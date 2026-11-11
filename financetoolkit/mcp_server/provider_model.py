"""
Centralized Finance Toolkit provider with SQLite-backed caching.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import hashlib
import inspect
import json
import os
import time
import weakref
from collections import OrderedDict
from importlib import metadata
from threading import Lock, RLock
from typing import Any

import pandas as pd

from financetoolkit import Toolkit
from financetoolkit.cache import cache_controller, policy_model
from financetoolkit.cache.cache_controller import Cache
from financetoolkit.discovery.discovery_controller import Discovery
from financetoolkit.economics.economics_controller import Economics
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.mcp_server.analytics_model import hold_lock
from financetoolkit.mcp_server.auth_model import resolve_api_key, resolve_fred_api_key
from financetoolkit.mcp_server.diagnostics_model import (
    FMP_KEY_HINT,
    add_call_input_info,
    call_input_info,
    current_call_messages,
)
from financetoolkit.portfolio.portfolio_controller import Portfolio
from financetoolkit.utilities import validation_model
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import collect_sources, record_source

logger = get_logger()

# Used as the Toolkit's default API key when set as an environment variable.
API_KEY: str = os.environ.get("FINANCIAL_MODELING_PREP_API_KEY", "")

# Optional — only gates the subset of Economics/FixedIncome tools backed by FRED.
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# The most Toolkit instances (per tickers, dates and key) and Economics, Fixed Income or
# Discovery instances (per dates and key) the provider keeps; the least recently used go
# first. An Economics instance holds the Global Macro Database, over 100 MB, so few.
MAXIMUM_TOOLKIT_INSTANCES = 64
# A Portfolio holds the daily prices and values of all its positions and its benchmark.
MAXIMUM_PORTFOLIO_INSTANCES = 16
MAXIMUM_STANDALONE_INSTANCES = 4
# An instance unused this long is dropped, so a long-running server (a stdio session open
# for days) gives back the memory of the data it no longer answers from.
INSTANCE_IDLE_SECONDS = 30 * 60


def release_memory() -> None:
    """
    Hands memory freed by dropped instances back to the operating system. The C library
    (glibc) keeps freed memory for later allocations, so without this the process keeps
    the size of its largest moment, even when it no longer uses it.
    """
    gc.collect()

    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


class InstanceCache(OrderedDict):
    """
    The instances the provider reuses, at most `maximum`, the least recently used first. An
    instance unused for idle_seconds is dropped when the cache is next used.
    """

    def __init__(self, maximum: int, idle_seconds: float = INSTANCE_IDLE_SECONDS):
        super().__init__()
        self.maximum = maximum
        self.idle_seconds = idle_seconds
        self._last_used: dict[str, float] = {}

    def recall(self, key: str) -> Any | None:
        """
        The instance kept under key, or None.

        Args:
            key (str): The cache key.

        Returns:
            Any | None: The instance, now the most recently used.
        """
        self.expire()
        instance = self.get(key)

        if instance is not None:
            self.move_to_end(key)
            self._last_used[key] = time.monotonic()

        return instance

    def keep(self, key: str, instance: Any) -> Any:
        """
        Keeps an instance, unless one is kept under key already, and drops the least
        recently used beyond the maximum.

        Args:
            key (str): The cache key.
            instance (Any): The instance.

        Returns:
            Any: The instance kept under key.
        """
        instance = self.setdefault(key, instance)
        self.move_to_end(key)
        self._last_used[key] = time.monotonic()

        while len(self) > self.maximum:
            self._last_used.pop(self.popitem(last=False)[0], None)

        return instance

    def expire(self) -> None:
        """Drops the instances unused for idle_seconds and releases their memory."""
        now = time.monotonic()
        expired = False

        # Ordered from least to most recently used, so the stale ones are at the front.
        while (
            self
            and now - self._last_used.get(next(iter(self)), now) > self.idle_seconds
        ):
            self._last_used.pop(self.popitem(last=False)[0], None)
            expired = True

        if expired:
            release_memory()


# Tool responses are computed output, so they sit under their own source.
MCP_CACHE_SOURCE = policy_model.MCP
MCP_CACHE_DATASET = "tool"
# The warnings logged while a cached response was computed, replayed with it.
MCP_CACHE_MESSAGES_DATASET = "tool_messages"

# Part of every tool response cache key, so a release that changes a formula can never be answered from the previous release's cached numbers.  # noqa: E501
try:
    TOOLKIT_VERSION: str = metadata.version("financetoolkit")
except metadata.PackageNotFoundError:  # pragma: no cover - running from a checkout
    TOOLKIT_VERSION = "unknown"

# Tried in order when the benchmark is the only requested ticker, so there is still something to analyse.
FALLBACK_BENCHMARKS = ["SPY", "QQQ", "^GSPC", "IWM", "DIA", "VTI"]


def resolve_benchmark_ticker(
    tickers: list[str], benchmark_ticker: str | None
) -> tuple[str | None, str | None]:
    """
    Decide which benchmark the Toolkit is built with when it overlaps the tickers.

    The Toolkit drops a ticker that is also the benchmark and reports it under
    "Benchmark" instead. That is kept whenever another ticker remains, so the
    benchmark is always the one that was asked for: silently swapping it made
    "Benchmark" a different security (e.g. QQQ instead of SPY) and skewed every
    benchmark-based metric. Only when the benchmark is the sole ticker, which would
    leave nothing to analyse, is a fallback benchmark used.

    Args:
        tickers (list[str]): The requested ticker symbols.
        benchmark_ticker (str | None): The requested benchmark ticker.

    Returns:
        tuple[str | None, str | None]: The benchmark to build the Toolkit with, and a
            note for the response describing what happened (None when nothing did).
    """
    upper_tickers = [ticker.upper() for ticker in tickers]

    if not benchmark_ticker or benchmark_ticker.upper() not in upper_tickers:
        return benchmark_ticker, None

    if any(ticker != benchmark_ticker.upper() for ticker in upper_tickers):
        return benchmark_ticker, (
            f"{benchmark_ticker} is the benchmark, so its data is reported under "
            f"'Benchmark' rather than as a separate {benchmark_ticker} column."
        )

    for candidate in FALLBACK_BENCHMARKS:
        if candidate.upper() not in upper_tickers:
            return candidate, (
                f"{benchmark_ticker} was requested as both the ticker and the benchmark, "
                f"so {candidate} is used as the benchmark: 'Benchmark' refers to {candidate}."
            )

    return None, (
        f"{benchmark_ticker} was requested as both the ticker and the benchmark, so the "
        "results are calculated without a benchmark."
    )


LOWERCASE_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
}


def readable_name(method_name: str) -> str:
    """
    Turns a method's name into a column name, e.g. get_value_at_risk into Value at Risk.

    Args:
        method_name (str): The method's name.

    Returns:
        str: The readable name.
    """
    words = method_name.removeprefix("get_").split("_")

    return " ".join(
        word if index and word in LOWERCASE_WORDS else word.capitalize()
        for index, word in enumerate(words)
    )


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
        excluded_sources: frozenset[str] = frozenset(),
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
                one is expected to pass False (see ``_resolve_cache_mode`` in
                ``mcp_controller``).
            excluded_sources (frozenset[str], optional): Sources whose data is never
                cached, such as FinancialModelingPrep on a hosted server, where it
                belongs to each subscriber. Defaults to an empty set.
        """
        # A placeholder from the docs counts as no key, so tools answer with the "key required" message.
        self._api_key = validation_model.resolve_api_key(api_key)
        self._fred_api_key = fred_api_key

        # Zeroing the TTL and withholding the location is what turns both layers off.
        self._cache_ttl: int = cache_ttl if cache_enabled else 0
        self._cache_location = database_location

        # The same cache the library uses; a disabled one never opens the database.
        self._cache: Cache = cache_controller.get_cache(
            location=database_location,
            enabled=cache_enabled,
            excluded_sources=excluded_sources,
        )

        # What Toolkit and Discovery get as `use_cached_data`: this cache, so they leave
        # out the same sources, or False.
        self._use_cached_data: bool | Cache = self._cache if cache_enabled else False

        if cache_enabled:
            cache_controller.set_active_cache(self._cache)
        else:
            # set_active_cache refuses a downgrade, so withdrawing has to be explicit.
            cache_controller.clear_active_cache()

        # Least recently used first; bounded, as a hosted server meets many distinct requests
        # and an Economics instance holds the Global Macro Database (over 100 MB) per range.
        self._toolkit_cache = InstanceCache(MAXIMUM_TOOLKIT_INSTANCES)
        self._standalone_cache = InstanceCache(MAXIMUM_STANDALONE_INSTANCES)
        self._portfolio_cache = InstanceCache(MAXIMUM_PORTFOLIO_INSTANCES)
        self._lock = Lock()
        # One lock per cached Toolkit or module instance, held for the whole call. Tools run
        # in worker threads, and an instance fills its statements, prices and conversions
        # lazily without locks of its own, so two calls on one instance at once would read
        # half-converted statements or download the same data twice. It is taken through
        # analytics_model.hold_lock, so a call waiting for it leaves its work slot to others.
        self._instance_locks: weakref.WeakKeyDictionary[Any, RLock] = (
            weakref.WeakKeyDictionary()
        )
        self._shared_instance_lock = RLock()
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
        toolkit_risk_free_rate: str = "10y",
        toolkit_basket: bool = False,
        portfolio: dict[str, Any] | None = None,
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
            toolkit_risk_free_rate (str): The Treasury yield the Toolkit uses as the risk-free
                rate ("13w", "5y", "10y" or "30y"). Defaults to ``"10y"``.
            toolkit_basket (bool): Whether the Toolkit adds up the tickers into a "Total"
                (Toolkit(basket=True)). Defaults to ``False``.
            portfolio (dict[str, Any] | None): The portfolio of a ``"portfolio"`` category
                call: transactions, holdings, weights or a preset, with optional
                start_date, initial_value, rebalancing and currency (see
                ``registry_controller._portfolio_inputs``). Defaults to None.
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
            # Unlike every other entry in this database, an MCP tool response is computed rather than fetched, so its meaning depends on the formulas of the release that produced it; keying by version means an upgrade can never answer from the previous release's arithmetic.  # noqa: E501
            "financetoolkit_version": TOOLKIT_VERSION,
            "module": module_name,
            "method": method_name,
            "tickers": sorted(t.upper() for t in tickers) if tickers else [],
            "countries": sorted(countries) if countries else [],
            "start": start_date,
            "end": end_date,
            "quarterly": quarterly,
            "benchmark_ticker": benchmark_ticker,
            "risk_free_rate": toolkit_risk_free_rate,
            "basket": toolkit_basket,
            "portfolio": portfolio or {},
            # Every keyword argument, whatever its type: filtering to scalars here dropped list-valued arguments from the key entirely, so `lag=[1, 4]` and `lag=[5, 10]` hashed the same and the second call was answered with the first one's frame. The key derivation canonicalises nested structures itself, so nothing has to be excluded to keep it stable.  # noqa: E501
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
                # The warnings of the call that produced it (a country not covered, a
                # ticker without data) qualify a cached answer just the same.
                cached_messages = self._cache.get(
                    source=MCP_CACHE_SOURCE,
                    dataset=MCP_CACHE_MESSAGES_DATASET,
                    entity=f"{module_name}.{method_name}",
                    parameters=cache_params,
                    ttl=self._cache_ttl,
                )
                call_messages = current_call_messages()
                if cached_messages is not None and call_messages is not None:
                    kinds = (
                        cached_messages["Kind"]
                        if "Kind" in cached_messages.columns
                        else pd.Series("warning", index=cached_messages.index)
                    )
                    call_messages.extend(
                        cached_messages.loc[kinds == "warning", "Message"].astype(str)
                    )
                    add_call_input_info(
                        list(
                            cached_messages.loc[kinds == "input", "Message"].astype(str)
                        )
                    )
                # The sources of the call that produced it, as nothing is retrieved now.
                if cached_messages is not None and "Kind" in cached_messages.columns:
                    for source in cached_messages.loc[
                        cached_messages["Kind"] == "source", "Message"
                    ].astype(str):
                        record_source(source)
                return cached

        logger.info(
            f"Calling Finance Toolkit functionality ({module_name}, {method_name})"
        )
        call_messages = current_call_messages()
        first_message = len(call_messages) if call_messages is not None else 0

        # The sources the answer's data comes from (see sources_model.track_sources).
        with collect_sources() as call_sources:
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
                    toolkit_risk_free_rate=toolkit_risk_free_rate,
                    toolkit_basket=toolkit_basket,
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
                    toolkit_risk_free_rate=toolkit_risk_free_rate,
                    toolkit_basket=toolkit_basket,
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
            elif category == "portfolio":
                # A Portfolio of the caller's transactions, holdings, weights or a preset.
                result = self.call_portfolio_functionality(
                    method_name=method_name,
                    portfolio=portfolio or {},
                    start_date=start_date,
                    benchmark_ticker=benchmark_ticker,
                    api_key=effective_key,
                    **method_kwargs,
                )
            elif category == "discovery":
                # Also initialisable with the Toolkit, but needs nothing beyond the API key.
                instance = Discovery(
                    api_key=effective_key, use_cached_data=self._use_cached_data
                )
                method = getattr(instance, method_name)
                # The calendars and filings take a range, passed on only when one is given.
                accepted = inspect.signature(method).parameters
                for name, value in (("start_date", start_date), ("end_date", end_date)):
                    if value and name in accepted:
                        method_kwargs[name] = value
                result = method(**method_kwargs)
            else:
                raise ValueError(
                    f"Unknown category '{category}' for module '{module_name}'"
                )

        # Passed on to the tool call's own collection, and kept with a cached answer.
        for source in call_sources:
            record_source(source)

        # An option chain carries its expiration date as an attribute, which no answer
        # would show; as a column it does, also from the cache.
        expiration = (
            getattr(result, "name", None) if isinstance(result, pd.DataFrame) else None
        )
        if (
            module_name == "options"
            and isinstance(expiration, str)
            and validation_model.is_valid_date(expiration)
            and "Expiration Date" not in result.columns
        ):
            result = result.copy()
            result.insert(0, "Expiration Date", expiration)

        if isinstance(result, pd.Series):
            # An unnamed result is named after its method, e.g. "Value at Risk", rather than
            # becoming a column called 0.
            result = result.to_frame(
                name=(
                    result.name
                    if result.name is not None
                    else readable_name(method_name)
                )
            )
        # An empty result is not cached: it usually means a missing API key or an
        # unreachable source, and a cached copy would keep answering "No data available"
        # after the key is added or the source is back.
        if (
            self._cache_ttl
            and isinstance(result, pd.DataFrame)
            and not result.empty
            and bool(result.notna().to_numpy().any())
        ):
            self._cache.set(
                source=MCP_CACHE_SOURCE,
                dataset=MCP_CACHE_DATASET,
                entity=f"{module_name}.{method_name}",
                data=result,
                parameters=cache_params,
            )
            warnings = (
                call_messages[first_message:] if call_messages is not None else []
            )
            inputs = call_input_info()
            sources = sorted(call_sources)
            if warnings or inputs or sources:
                self._cache.set(
                    source=MCP_CACHE_SOURCE,
                    dataset=MCP_CACHE_MESSAGES_DATASET,
                    entity=f"{module_name}.{method_name}",
                    data=pd.DataFrame(
                        {
                            "Message": [*warnings, *inputs, *sources],
                            "Kind": ["warning"] * len(warnings)
                            + ["input"] * len(inputs)
                            + ["source"] * len(sources),
                        }
                    ),
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
        toolkit_risk_free_rate: str = "10y",
        toolkit_basket: bool = False,
    ) -> list[str]:
        """Return human-readable notes describing data transformations applied to
        the most recent result for the given Toolkit instance (benchmark handling,
        fiscal-year relabelling and currency conversion).

        Returns an empty list when no transformations were applied or when the
        Toolkit instance has not yet fetched any financial statements, apart from
        the benchmark note, which depends on the request alone.
        """
        notes: list[str] = []

        # Derived from the request alone, so it is reported even when the data fetch below fails.
        _, benchmark_note = resolve_benchmark_ticker(tickers, benchmark_ticker)
        if benchmark_note:
            notes.append(benchmark_note)

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
                toolkit_risk_free_rate=toolkit_risk_free_rate,
                toolkit_basket=toolkit_basket,
                api_key=effective_key,
                fred_api_key=effective_fred_key,
            )
        except Exception:
            return notes

        with hold_lock(self.instance_lock(toolkit)):
            fy_adj: dict = dict(getattr(toolkit, "_fiscal_year_adjustments", {}))
        fy_tickers = []

        for ticker, adjustments in fy_adj.items():
            # The registry only ever records periods that were actually relabelled, so an entry present here means a shift happened. Comparing the two fields with .get() would turn an unexpected shape into "no shift at all", which is exactly the silent miss this note exists to prevent.  # noqa: E501
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
                # Values are bare years for annual data and period labels such as "2026Q1" for quarterly data, so an example carries both cases.  # noqa: E501
                fiscal, calendar = shifted[0]
                fy_tickers.append(f"{ticker} (e.g. fiscal {fiscal} as {calendar})")
            elif malformed:
                fy_tickers.append(f"{ticker} (periods relabelled, detail unavailable)")

        if fy_tickers:
            notes.append(
                "Fiscal years are shown as the calendar year they mostly fall in, so "
                f"companies compare like for like: {', '.join(fy_tickers)}"
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
                    "Financial statements are converted to the currency of the share "
                    f"price for: {', '.join(converted_currencies)}"
                )

        return notes

    def instance_lock(self, instance: Any) -> RLock:
        """
        Return the lock of a cached Toolkit or module instance, to hold while it is used.

        Args:
            instance (Any): The cached instance.

        Returns:
            RLock: The lock of that instance.
        """
        with self._lock:
            try:
                lock = self._instance_locks.get(instance)
                if lock is None:
                    lock = self._instance_locks[instance] = RLock()
            except TypeError:
                # An object without weak references (a stand-in in tests) shares one lock.
                lock = self._shared_instance_lock
            return lock

    def get_toolkit_instance(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        quarterly: bool,
        benchmark_ticker: str | None,
        api_key: str = "",
        fred_api_key: str = "",
        toolkit_risk_free_rate: str = "10y",
        toolkit_basket: bool = False,
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
            benchmark_ticker (str | None): Benchmark ticker symbol to use for comparative analysis,
                or None to run without a benchmark.
            api_key (str, optional): FinancialModelingPrep API key. Defaults to "".
            fred_api_key (str, optional): FRED API key, used by the Toolkit's
                `.economics`/`.fixedincome` properties. Optional. Defaults to "".
            toolkit_risk_free_rate (str, optional): The Treasury yield used as the risk-free rate
                ("13w", "5y", "10y" or "30y"). Defaults to "10y".
            toolkit_basket (bool, optional): Whether the Toolkit adds up the tickers into a
                "Total" (Toolkit(basket=True)). Defaults to False.

        Returns:
            Toolkit: A Toolkit instance configured for the requested tickers and parameters.
        """
        upper_tickers = [t.upper() for t in tickers]

        # The note half is surfaced to the model by get_transformation_notes.
        benchmark_ticker, benchmark_note = resolve_benchmark_ticker(
            tickers, benchmark_ticker
        )
        if benchmark_note:
            logger.info(benchmark_note)

        # Keyed by hashed FMP+FRED key so one user's Toolkit never reaches another.
        effective_key = api_key or self._api_key
        effective_fred_key = fred_api_key or self._fred_api_key
        key_hash = hashlib.sha256(
            f"{effective_key or ''}|{effective_fred_key or ''}".encode()
        ).hexdigest()
        cache_key = (
            f"{','.join(sorted(upper_tickers))}"
            f"|{start_date}|{end_date}|{quarterly}"
            f"|{benchmark_ticker or 'none'}|{toolkit_risk_free_rate}|{toolkit_basket}"
            f"|{key_hash}"
        )

        # Locked across check-create-store to stop two threads building duplicates.
        with self._lock:
            kept = self._toolkit_cache.recall(cache_key)
            if kept is not None:
                return kept

            if not effective_key:
                raise ValueError(FMP_KEY_HINT)

            toolkit_instance: Toolkit = Toolkit(
                tickers=tickers,
                api_key=effective_key,
                fred_api_key=effective_fred_key,
                start_date=start_date,
                end_date=end_date,
                quarterly=quarterly,
                benchmark_ticker=benchmark_ticker,
                risk_free_rate=toolkit_risk_free_rate,
                basket=toolkit_basket,
                # The same database the provider opened, or False when running uncached.
                use_cached_data=self._use_cached_data,
                # Unrounded, as the answer is rounded to significant digits when formatted,
                # so small values (a daily variance) keep their precision.
                rounding=None,
            )
            return self._toolkit_cache.keep(cache_key, toolkit_instance)

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
        toolkit_risk_free_rate: str = "10y",
        toolkit_basket: bool = False,
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
            toolkit_risk_free_rate (str, optional): The Treasury yield used as the risk-free rate
                ("13w", "5y", "10y" or "30y"). Defaults to "10y".
            toolkit_basket (bool, optional): Whether the Toolkit adds up the tickers into a
                "Total" (Toolkit(basket=True)). Defaults to False.
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
            toolkit_risk_free_rate=toolkit_risk_free_rate,
            toolkit_basket=toolkit_basket,
        )
        with hold_lock(self.instance_lock(toolkit_instance)):
            module = getattr(toolkit_instance, module_name)
            return getattr(module, method_name)(**kwargs)

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
        toolkit_risk_free_rate: str = "10y",
        toolkit_basket: bool = False,
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
            toolkit_risk_free_rate (str, optional): The Treasury yield used as the risk-free rate
                ("13w", "5y", "10y" or "30y"). Defaults to "10y".
            toolkit_basket (bool, optional): Whether the Toolkit adds up the tickers into a
                "Total" (Toolkit(basket=True)). Defaults to False.
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
            toolkit_risk_free_rate=toolkit_risk_free_rate,
            toolkit_basket=toolkit_basket,
        )
        with hold_lock(self.instance_lock(toolkit_instance)):
            return getattr(toolkit_instance, method_name)(**kwargs)

    def get_portfolio_instance(
        self,
        portfolio: dict[str, Any],
        start_date: str,
        benchmark_ticker: str,
        api_key: str = "",
    ) -> Portfolio:
        """
        Return a (potentially cached) Portfolio for the given transactions, holdings, weights
        or preset.

        The inputs are validated by the registry (registry_controller._portfolio_inputs) and
        never name a file: transactions arrive as a list of trades and become a DataFrame here.
        Instances are kept per portfolio, start date, benchmark and hashed API key, least
        recently used first, like the Toolkit instances.

        Args:
            portfolio (dict[str, Any]): The validated portfolio inputs.
            start_date (str): When holdings, weights or a preset are invested, or "" for
                their default.
            benchmark_ticker (str): The benchmark the portfolio is compared with.
            api_key (str, optional): FinancialModelingPrep API key. Defaults to "".

        Returns:
            Portfolio: The Portfolio.
        """
        effective_key = api_key or self._api_key
        key_hash = hashlib.sha256((effective_key or "").encode()).hexdigest()
        cache_key = hashlib.sha256(
            json.dumps(
                [portfolio, start_date, benchmark_ticker, key_hash],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

        with self._lock:
            kept = self._portfolio_cache.recall(cache_key)
            if kept is not None:
                return kept

        if not effective_key:
            raise ValueError(FMP_KEY_HINT)

        options: dict[str, Any] = {
            "api_key": effective_key,
            "benchmark_ticker": benchmark_ticker or None,
            "use_cached_data": self._use_cached_data,
            "currency": portfolio.get("currency"),
        }
        if "transactions" in portfolio:
            instance = Portfolio(
                transactions=pd.DataFrame(portfolio["transactions"]), **options
            )
        else:
            options["start_date"] = start_date or None
            if "holdings" in portfolio:
                instance = Portfolio(holdings=portfolio["holdings"], **options)
            else:
                options["initial_value"] = portfolio.get("initial_value", 10000.0)
                options["rebalancing"] = portfolio.get("rebalancing")
                instance = (
                    Portfolio(weights=portfolio["weights"], **options)
                    if "weights" in portfolio
                    else Portfolio.from_preset(portfolio["preset"], **options)
                )

        # Built outside the lock, as building collects prices; a portfolio built twice at
        # once keeps the first one stored.
        with self._lock:
            return self._portfolio_cache.keep(cache_key, instance)

    def call_portfolio_functionality(
        self,
        method_name: str,
        portfolio: dict[str, Any],
        start_date: str,
        benchmark_ticker: str,
        api_key: str = "",
        **kwargs: Any,
    ) -> Any:
        """
        Invoke a Portfolio method on the Portfolio of the given inputs.

        Args:
            method_name (str): The Portfolio method, e.g. "get_portfolio_overview".
            portfolio (dict[str, Any]): The validated portfolio inputs.
            start_date (str): When holdings, weights or a preset are invested, or "".
            benchmark_ticker (str): The benchmark the portfolio is compared with.
            api_key (str, optional): FinancialModelingPrep API key. Defaults to "".
            **kwargs (Any): The arguments of the method.

        Returns:
            Any: The method's result.
        """
        # The classic portfolios are a fixed list that needs no portfolio of its own.
        if method_name == "get_presets":
            return Portfolio.get_presets()

        instance = self.get_portfolio_instance(
            portfolio=portfolio,
            start_date=start_date,
            benchmark_ticker=benchmark_ticker,
            api_key=api_key,
        )
        with hold_lock(self.instance_lock(instance)):
            return getattr(instance, method_name)(**kwargs)

    @staticmethod
    def call_econometrics_with_data(
        method_name: str,
        data: pd.DataFrame,
        changes: str | None = "percentage",
        **method_kwargs: Any,
    ) -> Any:
        """
        Runs an Econometrics method on any data, such as the indicators an econometrics call
        names as its series, rather than on the prices of tickers.

        Args:
            method_name (str): The Econometrics method, e.g. "get_ols".
            data (pd.DataFrame): The series, a column each, by date.
            changes (str | None, optional): How their changes are calculated, see
                Econometrics (None for series that are changes already). Defaults to
                "percentage".
            **method_kwargs (Any): The arguments of the method.

        Returns:
            Any: The method's result.
        """
        from financetoolkit.econometrics.econometrics_controller import (  # noqa: PLC0415
            Econometrics,
        )

        return getattr(Econometrics(data=data, changes=changes), method_name)(
            **method_kwargs
        )

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
            # FixedIncome also holds the FMP key (treasury rates), so both keys are part of the cache key.
            key_hash = hashlib.sha256(
                f"{effective_key or ''}|{effective_fred_key or ''}".encode()
            ).hexdigest()
            cache_key = f"{module_name}|{start_date}|{end_date}|{quarterly}|{key_hash}"

        # Locked across check-create-store to stop two threads building duplicates.
        with self._lock:
            instance = self._standalone_cache.recall(cache_key)

            if instance is None:
                if module_name == "economics":
                    instance = Economics(
                        start_date=start_date,
                        end_date=end_date,
                        quarterly=quarterly,
                        fred_api_key=effective_fred_key,
                        cache=self._cache,
                        api_key=effective_key,
                        rounding=None,
                    )
                elif module_name == "fixedincome":
                    # get_treasury_rates is served by FMP, so without the key it silently returns no data.
                    instance = FixedIncome(
                        api_key=effective_key,
                        start_date=start_date,
                        end_date=end_date,
                        quarterly=quarterly,
                        fred_api_key=effective_fred_key,
                        cache=self._cache,
                        rounding=None,
                    )
                elif module_name == "discovery":
                    instance = Discovery(
                        api_key=effective_key, use_cached_data=self._use_cached_data
                    )
                else:
                    raise ValueError(f"Unknown standalone module: {module_name}")

                self._standalone_cache.keep(cache_key, instance)

        method = getattr(instance, method_name)

        # If the method accepts a 'countries' parameter, pass it through.
        countries_handled = False
        if countries:
            sig = inspect.signature(method)
            if "countries" in sig.parameters:
                kwargs["countries"] = countries
                countries_handled = True
            elif "country" in sig.parameters and module_name != "discovery":
                # An analysis of one country's data, such as the principal components of
                # its yield curve, takes the country on its own. (The stock screener's
                # country is an exchange country code, so Discovery is left out.)
                if len(countries) > 1:
                    raise ValueError(
                        f"{method_name} analyzes one country at a time, not "
                        f"{', '.join(countries)}. Please call it once per country."
                    )
                kwargs["country"] = countries[0]
                countries_handled = True

        with hold_lock(self.instance_lock(instance)):
            result = method(**kwargs)

        # Countries requested but not accepted by the method, so filter post-call.
        if countries and not countries_handled and isinstance(result, pd.DataFrame):
            available = [c for c in countries if c in result.columns]
            if available:
                result = result[available]

        return result
