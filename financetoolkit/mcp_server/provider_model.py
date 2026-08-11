"""
Centralized Finance Toolkit provider with SQLite-backed caching.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import time
from importlib import metadata
from threading import Lock
from typing import Any

import pandas as pd

from financetoolkit import Toolkit
from financetoolkit.cache import cache_controller, policy_model
from financetoolkit.cache.cache_controller import Cache
from financetoolkit.discovery.discovery_controller import Discovery
from financetoolkit.economics.economics_controller import Economics
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.mcp_server.auth_model import resolve_api_key, resolve_fred_api_key
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# Used as the Toolkit's default API key when set as an environment variable.
API_KEY: str = os.environ.get("FINANCIAL_MODELING_PREP_API_KEY", "")

# Optional — only gates the subset of Economics/FixedIncome tools backed by FRED.
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# Tool responses are computed output, so they sit under their own source.
MCP_CACHE_SOURCE = policy_model.MCP
MCP_CACHE_DATASET = "tool"

# Part of every tool response cache key, so a release that changes a formula can
# never be answered from the previous release's cached numbers.
try:
    TOOLKIT_VERSION: str = metadata.version("financetoolkit")
except metadata.PackageNotFoundError:  # pragma: no cover - running from a checkout
    TOOLKIT_VERSION = "unknown"


class ToolkitProvider:
    """
    Stateless provider that routes MCP tool calls to the appropriate
    Finance Toolkit module.
    """

    def __init__(
        self,
        cache_ttl: int,
        database_location: str,
        api_key: str = API_KEY,
        fred_api_key: str = FRED_API_KEY,
        cache_enabled: bool = True,
    ) -> None:
        """
        Initializes the ToolkitProvider.

        Args:
            cache_ttl (int): Time-to-live in seconds for SQLite-cached results.
                Set to 0 to disable caching.
            database_location (str): Path to the SQLite database file used for
                caching DataFrame results across calls.
            api_key (str, optional): FinancialModelingPrep API key used to
                initialize Toolkit and Discovery instances. Defaults to the value
                of the FINANCIAL_MODELING_PREP_API_KEY environment variable.
            fred_api_key (str, optional): FRED API key used to unlock the subset
                of Economics/FixedIncome tools backed by FRED data (e.g. Nonfarm
                Payrolls, ICE BofA bond indices). Optional — those specific tools
                simply error informatively when it's absent. Defaults to the
                value of the FRED_API_KEY environment variable.
            cache_enabled (bool, optional): Whether this server caches anything at
                all. When False no database is opened and every layer runs
                uncached, including the source data behind a tool response.
                Defaults to True, which suits a local single-user server; a hosted
                one is expected to pass False (see ``_resolve_cache_enabled`` in
                ``mcp_controller``). Defaults to True.
        """
        self._api_key = api_key
        self._fred_api_key = fred_api_key
        self._cache_enabled = cache_enabled

        # Zeroing the TTL and withholding the location is what turns both layers off.
        self._cache_ttl: int = cache_ttl if cache_enabled else 0
        self._cache_location = database_location

        # The same cache the library uses; a disabled one never opens the database.
        self._cache: Cache = cache_controller.get_cache(
            location=database_location, enabled=cache_enabled
        )

        # What Toolkit and Discovery get as `use_cached_data`: the path, or False.
        self._use_cached_data: bool | str = (
            database_location if cache_enabled else False
        )

        if cache_enabled:
            cache_controller.set_active_cache(self._cache)
        else:
            # set_active_cache refuses a downgrade, so withdrawing has to be explicit.
            cache_controller.clear_active_cache()

        self._toolkit_cache: dict[str, Any] = {}
        self._standalone_cache: dict[str, Any] = {}
        self._lock = Lock()
        self._last_eviction = 0.0

    def call_method(
        self,
        module_name: str,
        method_name: str,
        category: str,
        tickers: list[str] | None = None,
        countries: list[str] | None = None,
        start_date: str = "",
        end_date: str = "",
        quarterly: bool = False,
        benchmark_ticker: str = "SPY",
        **method_kwargs: Any,
    ) -> Any:
        """
        Route a tool call to the correct Finance Toolkit module.

        The ``tickers``, ``countries``, ``start_date``, ``end_date``,
        ``quarterly``, and ``benchmark_ticker`` parameters are all optional so
        that callers that do not require them (e.g. ``discovery`` category calls)
        can omit them without constructing dummy values.

        Args:
            module_name (str): Logical module name (e.g. ``"ratios"``,
                ``"economics"``, ``"toolkit"``).
            method_name (str): Public method to invoke on the module.
            category (str): Dispatch category — one of ``"ticker"``,
                ``"toolkit"``, ``"standalone"``, or ``"discovery"``.
            tickers (list[str] | None): Ticker symbols. Required for ``"ticker"``
                and ``"toolkit"`` categories; ignored otherwise.
            countries (list[str] | None): Country identifiers. Used by
                ``"standalone"`` (economics/fixedincome) modules.
            start_date (str): ISO-format start date (``YYYY-MM-DD``). Defaults to
                an empty string which is handled gracefully by the modules.
            end_date (str): ISO-format end date (``YYYY-MM-DD``). Defaults to an
                empty string.
            quarterly (bool): Whether to request quarterly granularity. Defaults
                to ``False`` (annual).
            benchmark_ticker (str): Benchmark symbol used by ticker-category
                modules. Defaults to ``"SPY"``.
            **method_kwargs: Additional keyword arguments forwarded verbatim to
                the underlying controller method.

        Returns:
            Any: The raw result from the underlying Finance Toolkit method —
                typically a ``pd.DataFrame``, ``pd.Series``, scalar, or dict.
        """
        # Per-request key for hosted HTTP, falling back to the env key on stdio.
        effective_key = resolve_api_key() or self._api_key
        effective_fred_key = resolve_fred_api_key() or self._fred_api_key
        current_time = time.time()

        if self._cache_ttl and (current_time - self._last_eviction) > self._cache_ttl:
            # Scoped to this server's responses: the same database holds price history.
            evicted_count = self._cache.remove_expired_entries(
                ttl=self._cache_ttl, source=MCP_CACHE_SOURCE
            )
            self._last_eviction = current_time
            if evicted_count > 0:
                logger.info(
                    f"Evicted {evicted_count} expired cache entries. "
                    f"Disable this by setting cache_ttl to 0 or None in the YAML configuration."
                )

        cache_params = {
            # Unlike every other entry in this database, an MCP tool response is
            # computed rather than fetched, so its meaning depends on the formulas
            # of the release that produced it. Keying by version means an upgrade
            # can never answer from the previous release's arithmetic.
            "financetoolkit_version": TOOLKIT_VERSION,
            "module": module_name,
            "method": method_name,
            "tickers": sorted(t.upper() for t in tickers) if tickers else [],
            "countries": sorted(countries) if countries else [],
            "start": start_date,
            "end": end_date,
            "quarterly": quarterly,
            "benchmark_ticker": benchmark_ticker,
            # Every keyword argument, whatever its type. Filtering to scalars here
            # dropped list-valued arguments from the key entirely, so `lag=[1, 4]`
            # and `lag=[5, 10]` hashed the same and the second call was answered
            # with the first one's frame. The key derivation canonicalises nested
            # structures itself, so nothing has to be excluded to keep it stable.
            **method_kwargs,
        }

        # A falsy TTL disables caching, so skip both the read and the write.
        if not self._cache_ttl:
            pass  # fall through directly to the live call below
        else:
            cached = self._cache.get(
                source=MCP_CACHE_SOURCE,
                dataset=MCP_CACHE_DATASET,
                entity=f"{module_name}.{method_name}",
                parameters=cache_params,
                ttl=self._cache_ttl,
            )
            if cached is not None:
                logger.info(
                    f"Acquired cache information ({module_name}, {method_name})"
                )
                return cached

        logger.info(
            f"Calling Finance Toolkit functionality ({module_name}, {method_name})"
        )

        if category == "ticker":
            # Functionality reached through a sub-module property (ratios, models, options).
            if not tickers:
                raise ValueError(
                    f"'{method_name}' requires one or more ticker symbols. "
                    "Provide them via the `tickers` parameter."
                )
            result = self.call_sub_module_functionality(
                module_name=module_name,
                method_name=method_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                quarterly=quarterly,
                benchmark_ticker=benchmark_ticker,
                api_key=effective_key,
                fred_api_key=effective_fred_key,
                **method_kwargs,
            )
        elif category == "toolkit":
            # Functionality on the Toolkit class itself rather than a sub-module property.
            if not tickers:
                raise ValueError(
                    f"'{method_name}' requires one or more ticker symbols. "
                    "Provide them via the `tickers` parameter."
                )
            result = self.call_toolkit_functionality(
                method_name=method_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                quarterly=quarterly,
                benchmark_ticker=benchmark_ticker,
                api_key=effective_key,
                fred_api_key=effective_fred_key,
                **method_kwargs,
            )
        elif category == "standalone":
            # Module such as Economics or FixedIncome, initialised without the Toolkit.
            result = self.call_standalone_module_functionality(
                module_name=module_name,
                method_name=method_name,
                start_date=start_date,
                end_date=end_date,
                quarterly=quarterly,
                countries=countries,
                api_key=effective_key,
                fred_api_key=effective_fred_key,
                **method_kwargs,
            )
        elif category == "discovery":
            # Also initialisable with the Toolkit, but needs nothing beyond the API key.
            instance = Discovery(
                api_key=effective_key, use_cached_data=self._use_cached_data
            )
            result = getattr(instance, method_name)(**method_kwargs)
        else:
            raise ValueError(
                f"Unknown category '{category}' for module '{module_name}'"
            )

        if isinstance(result, pd.Series):
            result = result.to_frame()
        if self._cache_ttl and isinstance(result, pd.DataFrame):
            self._cache.set(
                source=MCP_CACHE_SOURCE,
                dataset=MCP_CACHE_DATASET,
                entity=f"{module_name}.{method_name}",
                data=result,
                parameters=cache_params,
            )

        return result

    def get_transformation_notes(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        quarterly: bool,
        benchmark_ticker: str,
        api_key: str = "",
        fred_api_key: str = "",
    ) -> list[str]:
        """Return human-readable notes describing data transformations applied to
        the most recent result for the given Toolkit instance (fiscal-year
        relabelling and currency conversion).

        Returns an empty list when no transformations were applied or when the
        Toolkit instance has not yet fetched any financial statements.
        """
        notes: list[str] = []
        try:
            effective_key = resolve_api_key() or api_key or self._api_key
            effective_fred_key = (
                resolve_fred_api_key() or fred_api_key or self._fred_api_key
            )
            toolkit = self.get_toolkit_instance(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                quarterly=quarterly,
                benchmark_ticker=benchmark_ticker,
                api_key=effective_key,
                fred_api_key=effective_fred_key,
            )
        except Exception:
            return notes

        fy_adj: dict = getattr(toolkit, "_fiscal_year_adjustments", {})
        fy_tickers = []

        for ticker, adjustments in fy_adj.items():
            # The registry only ever records periods that were actually relabelled,
            # so an entry present here means a shift happened. Comparing the two
            # fields with .get() would turn an unexpected shape into "no shift at
            # all", which is exactly the silent miss this note exists to prevent.
            shifted = [
                (adjustment["fiscal_year"], adjustment["calendar_year"])
                for adjustment in adjustments
                if isinstance(adjustment, dict)
                and "fiscal_year" in adjustment
                and "calendar_year" in adjustment
                and adjustment["fiscal_year"] != adjustment["calendar_year"]
            ]
            malformed = len(adjustments) - len(shifted)

            if malformed:
                logger.warning(
                    "%s of the %s fiscal period adjustments recorded for %s do not "
                    "carry both a fiscal_year and a differing calendar_year. The "
                    "relabelling is reported without the period detail.",
                    malformed,
                    len(adjustments),
                    ticker,
                )

            if shifted:
                # Values are bare years for annual data and period labels such as
                # "2026Q1" for quarterly data, so an example carries both cases.
                fiscal, calendar = shifted[0]
                fy_tickers.append(
                    f"{ticker} ({len(shifted)} periods, e.g. {fiscal} to {calendar})"
                )
            elif malformed:
                fy_tickers.append(f"{ticker} (periods relabelled, detail unavailable)")

        if fy_tickers:
            notes.append(
                f"Fiscal Year to Calendar Year mapped for: {', '.join(fy_tickers)}"
            )

        stmt_currencies: pd.Series = getattr(
            toolkit, "_statement_currencies", pd.Series()
        )
        convert_currency: bool = bool(getattr(toolkit, "_convert_currency", False))
        if convert_currency and not stmt_currencies.empty:
            converted_currencies = []
            for ticker, pair in stmt_currencies.items():
                src, dst = str(pair)[:3], str(pair)[3:6]
                if src != dst:
                    converted_currencies.append(f"{ticker} ({src} to {dst})")
            if converted_currencies:
                notes.append(
                    f"Aligned Financial Statements with OHLC for: {', '.join(converted_currencies)}"
                )

        return notes

    def get_toolkit_instance(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        quarterly: bool,
        benchmark_ticker: str,
        api_key: str = "",
        fred_api_key: str = "",
    ) -> Toolkit:
        """
        Return a (potentially cached) Toolkit instance for the requested tickers and date range.

        This method will attempt to return a cached Toolkit instance if one exists for the
        combination of tickers, start_date, end_date, quarterly flag and the provider API hash.
        The cache key uses an uppercase, sorted representation of the tickers to ensure
        consistent caching regardless of input order. Access to the cache is guarded by an
        internal lock to ensure thread-safety. If no cached instance exists, a new Toolkit
        is instantiated with the provider's API key and the provided parameters, stored in
        the cache, and returned.

        Args:
            tickers (list[str]): List of ticker symbols to include in the Toolkit instance.
            start_date (str): Start date for the Toolkit (format YYYY-MM-DD).
            end_date (str): End date for the Toolkit (format YYYY-MM-DD).
            quarterly (bool): Whether to initialize the Toolkit for quarterly (True)
                or yearly (False) statements.
            benchmark_ticker (str): Benchmark ticker symbol to use for comparative analysis.
            api_key (str, optional): FinancialModelingPrep API key. Defaults to "".
            fred_api_key (str, optional): FRED API key, used by the Toolkit's
                `.economics`/`.fixedincome` properties. Optional. Defaults to "".

        Returns:
            Toolkit: A Toolkit instance configured for the requested tickers and parameters.
        """
        upper_tickers = [t.upper() for t in tickers]

        # The Toolkit drops a ticker that is also the benchmark, so pick another one.
        if benchmark_ticker and benchmark_ticker.upper() in upper_tickers:
            fallback_benchmarks = ["SPY", "QQQ", "^GSPC", "IWM", "DIA", "VTI"]
            resolved_benchmark: str | None = None
            for candidate in fallback_benchmarks:
                if candidate.upper() not in upper_tickers:
                    resolved_benchmark = candidate
                    break
            if resolved_benchmark:
                logger.info(
                    "benchmark_ticker '%s' conflicts with a requested ticker. "
                    "Automatically switching benchmark to '%s'.",
                    benchmark_ticker,
                    resolved_benchmark,
                )
                benchmark_ticker = resolved_benchmark
            else:
                logger.warning(
                    "benchmark_ticker '%s' conflicts with a requested ticker and no "
                    "non-conflicting fallback could be found. Setting benchmark_ticker to None.",
                    benchmark_ticker,
                )
                benchmark_ticker = None  # type: ignore[assignment]

        # Keyed by hashed FMP+FRED key so one user's Toolkit never reaches another.
        effective_key = api_key or self._api_key
        effective_fred_key = fred_api_key or self._fred_api_key
        key_hash = hashlib.sha256(
            f"{effective_key or ''}|{effective_fred_key or ''}".encode()
        ).hexdigest()
        cache_key = (
            f"{','.join(sorted(upper_tickers))}"
            f"|{start_date}|{end_date}|{quarterly}"
            f"|{benchmark_ticker or 'none'}|{key_hash}"
        )

        # Locked across check-create-store to stop two threads building duplicates.
        with self._lock:
            if cache_key in self._toolkit_cache:
                return self._toolkit_cache[cache_key]

            if not effective_key:
                raise ValueError(
                    "A FinancialModelingPrep API key is required for this tool. "
                    "Local setup: set FINANCIAL_MODELING_PREP_API_KEY in your "
                    "environment or .env file. Hosted setup: pass your key via the "
                    "`X-FMP-API-Key` header or a `?fmp_api_key=...` URL parameter. "
                    "Get a key with 15% off via https://www.jeroenbouma.com/fmp"
                )

            toolkit_instance: Toolkit = Toolkit(
                tickers=tickers,
                api_key=effective_key,
                fred_api_key=effective_fred_key,
                start_date=start_date,
                end_date=end_date,
                quarterly=quarterly,
                benchmark_ticker=benchmark_ticker,
                # The same database the provider opened, or False when running uncached.
                use_cached_data=self._use_cached_data,
            )
            self._toolkit_cache[cache_key] = toolkit_instance

        return toolkit_instance

    def call_sub_module_functionality(
        self,
        module_name: str,
        method_name: str,
        tickers: list[str],
        start_date: str,
        end_date: str,
        quarterly: bool,
        benchmark_ticker: str,
        api_key: str = "",
        fred_api_key: str = "",
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | dict | float | int | str:
        """
        Invoke a method on a Toolkit sub-module for a given set of tickers and date range.

        Args:
            module_name (str): Name of the Toolkit sub-module to access (must match a Toolkit
                    property name, e.g. "ratios", "models", "options", "performance").
            method_name (str): Name of the method to call on the resolved sub-module.
            tickers (list[str]): One or more ticker symbols to configure the Toolkit with.
            start_date (str): Start date for data used by the Toolkit, formatted as "YYYY-MM-DD".
            end_date (str): End date for data used by the Toolkit, formatted as "YYYY-MM-DD".
            quarterly (bool): If True, Toolkit is configured to use quarterly financial statements;
                    otherwise annual statements are used.
            benchmark_ticker (str): Ticker used as benchmark (e.g. "SPY"); passed to Toolkit initialization.
            api_key (str, optional): FinancialModelingPrep API key. Defaults to "".
            fred_api_key (str, optional): FRED API key, used by the Toolkit's
                `.economics`/`.fixedincome` sub-modules. Optional. Defaults to "".
            **kwargs: Arbitrary keyword arguments forwarded to the resolved sub-module method.

        Returns:
            The result returned by the invoked sub-module method (typically a pandas DataFrame,
                pd.Series, scalar or other domain-specific object).
        """
        toolkit_instance = self.get_toolkit_instance(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            quarterly=quarterly,
            benchmark_ticker=benchmark_ticker,
            api_key=api_key,
            fred_api_key=fred_api_key,
        )
        module = getattr(toolkit_instance, module_name)
        method = getattr(module, method_name)

        return method(**kwargs)

    def call_toolkit_functionality(
        self,
        method_name: str,
        tickers: list[str],
        start_date: str,
        end_date: str,
        quarterly: bool,
        benchmark_ticker: str,
        api_key: str = "",
        fred_api_key: str = "",
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | dict | float | int | str:
        """
        Call a Toolkit method on an instantiated Toolkit object for the specified tickers and date range.
        This helper obtains (or creates) a Toolkit instance via self.get_toolkit_instance(...)
        and invokes the requested method by name, forwarding any additional keyword arguments
        to that method.

        Args:
            method_name (str): Name of the Toolkit method to invoke (e.g., "get_historical_data",
                "get_profile", "get_quote").
            tickers (list[str]): List of ticker symbols used to initialize the Toolkit instance.
            start_date (str): Start date for the Toolkit data range in YYYY-MM-DD format.
            end_date (str): End date for the Toolkit data range in YYYY-MM-DD format.
            quarterly (bool): Whether to initialize the Toolkit for quarterly financial statements.
            benchmark_ticker (str): Benchmark ticker symbol used for comparative analyses (e.g., "SPY").
            api_key (str, optional): FinancialModelingPrep API key. Defaults to "".
            fred_api_key (str, optional): FRED API key. Optional. Defaults to "".
            **kwargs (Any): Additional keyword arguments forwarded to the invoked Toolkit method.

        Returns:
            The return value of the invoked Toolkit method (commonly a pandas.DataFrame,
            pd.Series, dict, float, int, or str).
        """
        toolkit_instance = self.get_toolkit_instance(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            quarterly=quarterly,
            benchmark_ticker=benchmark_ticker,
            api_key=api_key,
            fred_api_key=fred_api_key,
        )
        method = getattr(toolkit_instance, method_name)

        return method(**kwargs)

    def call_standalone_module_functionality(
        self,
        module_name: str,
        method_name: str,
        start_date: str,
        end_date: str,
        quarterly: bool,
        countries: list[str] | None = None,
        api_key: str = "",
        fred_api_key: str = "",
        **kwargs: Any,
    ) -> Any:
        """
        Invoke a standalone module method (Economics, FixedIncome, Discovery) in a
        thread-safe, cached manner and return its result.

        Args:
            module_name (str): Name of the standalone module to use. One of
                "economics", "fixedincome", "discovery".
            method_name (str): The name of the method to invoke on the module instance.
            start_date (str): ISO formatted start date (YYYY-MM-DD) used when creating
                module instances (ignored for discovery).
            end_date (str): ISO formatted end date (YYYY-MM-DD) used when creating
                module instances (ignored for discovery).
            quarterly (bool): Whether the module instance should operate on quarterly
                data (used for Economics and FixedIncome).
            countries (list[str] | None): Optional list of country identifiers to pass
                to the called method or to use for post-call column filtering if the
                    method does not accept a 'countries' parameter.
            api_key (str, optional): FinancialModelingPrep API key, used by Discovery. Defaults to "".
            fred_api_key (str, optional): FRED API key, used by Economics and FixedIncome
                for the subset of their tools backed by FRED data. Optional. Defaults to "".
            **kwargs: Additional keyword arguments forwarded to the target method.

        Returns:
            The raw return value from the invoked method. If countries were
            provided but the method does not accept them and the returned value is a
            pandas.DataFrame, a filtered DataFrame containing only the requested country
            columns (if present) is returned.
        """
        # Keyed by whichever key affects the module, so users and keys never collide.
        effective_key = api_key or self._api_key
        effective_fred_key = fred_api_key or self._fred_api_key
        if module_name == "discovery":
            key_hash = hashlib.sha256((effective_key or "").encode()).hexdigest()
            cache_key = f"discovery|{key_hash}"
        else:
            fred_key_hash = hashlib.sha256(
                (effective_fred_key or "").encode()
            ).hexdigest()
            cache_key = (
                f"{module_name}|{start_date}|{end_date}|{quarterly}|{fred_key_hash}"
            )

        # Locked across check-create-store to stop two threads building duplicates.
        with self._lock:
            instance = self._standalone_cache.get(cache_key)

            if instance is None:
                if module_name == "economics":
                    instance = Economics(
                        start_date=start_date,
                        end_date=end_date,
                        quarterly=quarterly,
                        fred_api_key=effective_fred_key,
                        cache=self._cache,
                    )
                elif module_name == "fixedincome":
                    instance = FixedIncome(
                        start_date=start_date,
                        end_date=end_date,
                        quarterly=quarterly,
                        fred_api_key=effective_fred_key,
                        cache=self._cache,
                    )
                elif module_name == "discovery":
                    instance = Discovery(
                        api_key=effective_key, use_cached_data=self._use_cached_data
                    )
                else:
                    raise ValueError(f"Unknown standalone module: {module_name}")

                self._standalone_cache[cache_key] = instance

        method = getattr(instance, method_name)

        # If the method accepts a 'countries' parameter, pass it through.
        countries_handled = False
        if countries:
            sig = inspect.signature(method)
            if "countries" in sig.parameters:
                kwargs["countries"] = countries
                countries_handled = True

        result = method(**kwargs)

        # Countries requested but not accepted by the method, so filter post-call.
        if countries and not countries_handled and isinstance(result, pd.DataFrame):
            available = [c for c in countries if c in result.columns]
            if available:
                result = result[available]

        return result
