"""Economics Base Module"""

__docformat__ = "google"

import os
from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache.cache_controller import Cache, resolve_cache, set_active_cache
from financetoolkit.economics.helpers import (
    buffered_start_date,
    change_over_periods,
    choose_sources,
    combine_sources,
    extend_with_recent,
    resample_to_period,
)
from financetoolkit.economics.sources import (
    bis_model,
    bls_model,
    boe_model,
    boj_model,
    ecb_model,
    eurostat_model,
    fred_model,
    gmdb_model,
    imf_model,
    oecd_model,
    ons_model,
    sbj_model,
)
from financetoolkit.fixedincome import (
    fed_model,
)
from financetoolkit.utilities import validation_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import (
    collect_sources,
    record_source,
    track_sources,
)

# ruff: noqa: E501

logger = get_logger()

FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")


@filter_shown_columns
@track_sources
class EconomicsBase:
    """
    The data every economics topic shares: the Global Macro Database, the default frequencies and the source combinations several topics use.
    """

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        gmdb_source: bool = True,
        quarterly: bool | None = None,
        rounding: int | None = 4,
        fred_api_key: str = FRED_API_KEY,
        allow_stale_oecd_cache: bool = True,
        cache: Cache | None = None,
        api_key: str = "",
        use_cached_data: bool | str | None = None,
    ):
        """
        Initializes the Economics Controller Class.

        Args:
            start_date (str | None, optional): The start date to retrieve data from. Defaults to None,
                which starts 100 years before today.
            end_date (str | None, optional): The end date to retrieve data from. Defaults to None,
                which uses today.
            gmdb_source (bool, optional): If True, retrieves data from the GMDB source. Defaults to True.
            quarterly (bool | None, optional): If True, methods with a period argument return quarterly
                data where the indicator has it. Otherwise they return the highest frequency the
                indicator is published at: daily, then weekly, monthly, quarterly and yearly. Methods
                without a period argument return the frequency their source publishes. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            fred_api_key (str, optional): A FRED API key. The US labor market and real-activity
                indicators (e.g. Nonfarm Payrolls, Initial Jobless Claims, Retail Sales) come from their
                publishers without it and from FRED with it, which gives Nonfarm Payrolls the history
                from 1939 instead of the last three years. Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html. Can also be set
                via the FRED_API_KEY environment variable. Defaults to the value of FRED_API_KEY if set,
                otherwise an empty string.
            allow_stale_oecd_cache (bool, optional): the OECD API enforces a hard rate limit (60
                downloads/hour). When True, a 429 response falls back to the most recently cached
                successful response for that exact query instead of returning empty data -- opt-in,
                since the served data may not be the most up-to-date. Every successful OECD response
                is cached regardless of this setting. Defaults to True.
            cache (Cache | None, optional): A cache to use for the data this module retrieves, which takes
                precedence over use_cached_data. Defaults to None.
            api_key (str, optional): A FinancialModelingPrep API key, needed for the economic
                calendar, the market risk premium and the MOVE and JGB VIX of the implied volatility,
                and used for the asset class proxies (Yahoo Finance otherwise). Obtain one at
                https://www.jeroenbouma.com/fmp. Defaults to an empty string.
            use_cached_data (bool | str | None, optional): Whether to cache the data retrieved from external
                sources. None or True uses the shared cache database in the user configuration directory, False
                retrieves everything every time and a string is the path to a dedicated cache folder or database
                file. Defaults to None, which caches unless the FINANCE_TOOLKIT_CACHE_ENABLED environment variable is 0.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2010-01-01", end_date="2025-12-31")

        cpi = economics.get_consumer_price_index()

        cpi.loc['2010':, ['United States', 'Netherlands', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Netherlands |    Japan |
        |:-----|----------------:|--------------:|---------:|
        | 2010 |         91.9999 |       91.5878 |  96.5429 |
        | 2011 |         94.9042 |       93.7329 |  96.2674 |
        | 2012 |         96.8681 |       96.0349 |  96.22   |
        | 2013 |         98.2871 |       98.4467 |  96.5429 |
        | 2014 |         99.8815 |       99.4026 |  99.2212 |
        | 2015 |        100      |      100      | 100      |
        | 2016 |        101.262  |      100.319  |  99.886  |
        | 2017 |        103.419  |      101.704  | 100.361  |
        | 2018 |        105.945  |      103.436  | 101.33   |
        | 2019 |        107.865  |      106.161  | 101.833  |
        | 2020 |        109.195  |      107.507  | 101.814  |
        | 2021 |        114.325  |      110.389  | 101.567  |
        | 2022 |        123.474  |      121.426  | 104.112  |
        | 2023 |        128.557  |      126.092  | 107.503  |
        | 2024 |        132.349  |      130.311  | 110.457  |
        | 2025 |        135.831  |      134.486  | 113.98   |
        """
        if start_date and not validation_model.is_valid_date(start_date):
            raise ValueError(
                f"Please input a valid start date (%Y-%m-%d) like '2010-01-01', not '{start_date}'"
            )
        if end_date and not validation_model.is_valid_date(end_date):
            raise ValueError(
                f"Please input a valid end date (%Y-%m-%d) like '2020-01-01', not '{end_date}'"
            )
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Please ensure the start date {start_date} is before the end date {end_date}"
            )

        self._start_date = (
            start_date
            if start_date
            else (datetime.now() - timedelta(days=365 * 100)).strftime("%Y-%m-%d")
        )
        self._end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")

        # The dates as passed, before the defaults are filled in: the economic calendar only
        # falls back to dates that were asked for, not to the 100 year default.
        self._requested_start_date = start_date
        self._requested_end_date = end_date
        # The data retrieved from external sources is cached, see use_cached_data.
        resolved_cache = resolve_cache(use_cached_data, cache)
        self._cache = resolved_cache if resolved_cache.enabled else None
        # A copied documentation example passes the placeholder key, treated as no key at all.
        self._api_key = validation_model.resolve_api_key(api_key)

        # Published once here so the OECD and FRED free functions read it back.
        set_active_cache(self._cache)
        oecd_model.configure_oecd_cache(allow_stale_oecd_cache)

        self._gmdb_source: bool = gmdb_source

        # The Global Macro Database is one file of around 60 MB, so it is only retrieved
        # when a method first needs it, see _get_gmdb_dataset.
        self._gmdb_datasets: dict[bool, pd.DataFrame] = {}
        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding
        self._fred_api_key: str = validation_model.resolve_fred_api_key(fred_api_key)

    def _get_gmdb_dataset(self, gmdb_forecasts: bool = False) -> pd.DataFrame:
        """
        Retrieves the Global Macro Database the first time a method needs it, through the
        cache when one is set, and keeps it for the methods called after.

        Args:
            gmdb_forecasts (bool, optional): Whether to include the projected years.
                Defaults to False.

        Returns:
            pd.DataFrame: The dataset, with a (variable, country) column per series.
        """
        if gmdb_forecasts not in self._gmdb_datasets:
            # Shared by every instance in the process (see gmdb_model.get_shared_dataset).
            self._gmdb_datasets[gmdb_forecasts] = gmdb_model.get_shared_dataset(
                cache=self._cache, include_forecasts=gmdb_forecasts
            )

        return self._gmdb_datasets[gmdb_forecasts]

    def _get_gmdb_series(
        self, variable: str, in_percent: bool = False, gmdb_forecasts: bool = False
    ) -> pd.DataFrame:
        """
        Retrieves one variable of the Global Macro Database with a column per country.

        Args:
            variable (str): The variable, e.g. "hcons_GDP".
            in_percent (bool, optional): Whether the variable is quoted in percentage points,
                which is divided by 100. Defaults to False.
            gmdb_forecasts (bool, optional): Whether to include the projected years.
                Defaults to False.

        Returns:
            pd.DataFrame: The variable, indexed by year with a column per country.
        """
        return gmdb_model.get_series(
            self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts), variable, in_percent
        )

    @staticmethod
    def _combine_sources_for(
        sources: list[tuple[Callable[[], pd.DataFrame], str | None]],
        countries: list[str] | str | None,
    ) -> pd.DataFrame:
        """
        Retrieves several sources at the same time and combines them with combine_sources,
        in the order given. A source that only publishes one country is left out when that
        country is not requested, since it would contribute nothing else.

        Args:
            sources (list[tuple[Callable, str | None]]): Per source, the function that
                retrieves it and the only country it publishes, or None for a source of
                several countries.
            countries (list[str] | str | None): The requested countries, None for all.

        Returns:
            pd.DataFrame: One column per country, from its most current source.
        """
        requested = (
            None
            if countries is None
            else {countries} if isinstance(countries, str) else set(countries)
        )
        needed = [
            fetch
            for fetch, only_country in sources
            if only_country is None or requested is None or only_country in requested
        ]

        def fetch_with_sources(
            fetch: Callable[[], pd.DataFrame],
        ) -> tuple[pd.DataFrame, set[str]]:
            # Held back until it is known which source each country is taken from.
            with collect_sources() as fetched:
                frame = fetch()
            return frame, fetched

        retrieved = helpers.run_in_parallel(
            fetch_with_sources, [(fetch,) for fetch in needed]
        )
        frames = [frame for frame, _ in retrieved]
        combined = combine_sources(frames)

        # Only the sources a requested country is taken from are named with the answer.
        for country, position in choose_sources(frames).items():
            if requested is None or country in requested:
                for source in retrieved[position][1]:
                    record_source(source)

        # A source's own index name (such as "Effective Date") is not kept, as it was not
        # when every source was combined.
        combined.index.name = None

        return combined

    def _get_monthly_price_data(
        self,
        measure: str,
        countries: list[str] | str | None = None,
        component: str = "all",
    ) -> pd.DataFrame:
        """
        Combines the monthly consumer prices of every country from its most current source.

        The national statistical offices publish first, so they are preferred: Eurostat
        (the euro area and the European Economic Area, including the flash estimate), the
        Office for National Statistics (United Kingdom) and the Statistics Bureau of Japan
        (Japan). Every other country comes from the Bank for International Settlements,
        which compiles the national consumer price indices of around sixty countries,
        including the United States, China, India and Brazil. Each country comes from the
        source with the most recent month. None of them have the tight request limit of
        the OECD API.

        The BIS republishes the US index of the Bureau of Labor Statistics (BLS), rebased
        to 2010 = 100, a few days after its release, so the months the BLS has published
        since are added to it: the index chained on with the BLS changes, the inflation
        rate as the BLS computes it.

        The core index and the components come from Eurostat (all but shelter) and, for
        the United States, from the BLS, through FRED when a FRED API key is set.

        Args:
            measure (str): "inflation_rate" for the annual rate of change or
                "consumer_price_index" for the index itself.
            countries (list[str] | str | None, optional): The requested countries, so the
                sources of other countries are not retrieved. Defaults to None.
            component (str, optional): "all", "core", "energy", "food", "goods",
                "services" or "shelter". Defaults to "all".

        Returns:
            pd.DataFrame: One column per country, indexed by month.
        """
        # Only the requested months are asked for, with room for growth and rolling.
        start_date = buffered_start_date(self._start_date, "monthly")
        end_date = self._end_date

        eurostat_prices = (
            eurostat_model.get_inflation_rate
            if measure == "inflation_rate"
            else eurostat_model.get_consumer_price_index
        )

        if component != "all":
            return self._combine_sources_for(
                [
                    (
                        lambda: (
                            eurostat_prices(start_date, end_date, component)
                            if component in eurostat_model.HICP_AGGREGATES
                            else pd.DataFrame()
                        ),
                        None,
                    ),
                    (
                        lambda: self._get_united_states_consumer_prices(
                            measure, component
                        ),
                        "United States",
                    ),
                ],
                countries,
            )

        inflation = measure == "inflation_rate"
        sources = [
            (lambda: eurostat_prices(start_date, end_date), None),
            (
                (
                    ons_model.get_inflation_rate
                    if inflation
                    else ons_model.get_consumer_price_index
                ),
                "United Kingdom",
            ),
            (
                (
                    sbj_model.get_inflation_rate
                    if inflation
                    else sbj_model.get_consumer_price_index
                ),
                "Japan",
            ),
            (
                lambda: bis_model.get_consumer_prices(measure, start_date, end_date),
                None,
            ),
        ]

        if inflation:
            # Around a hundred further countries the BIS does not cover.
            sources.append(
                (lambda: imf_model.get_inflation_rate(start_date, end_date), None)
            )

        prices = self._combine_sources_for(sources, countries)

        if "United States" not in prices.columns:
            return prices

        with collect_sources() as recent_sources:
            recent = self._get_united_states_consumer_prices(measure, "all")
        history = prices["United States"].dropna()
        shared = history.index.intersection(recent.dropna().index)

        if measure == "consumer_price_index" and not shared.empty:
            # Rebased to the BIS's 2010 = 100 on the latest month both have.
            recent = (
                recent * history[shared[-1]] / recent.loc[shared[-1], "United States"]
            )

        return extend_with_recent(prices, recent, "United States", recent_sources)

    def _get_united_states_consumer_prices(
        self, measure: str, component: str
    ) -> pd.DataFrame:
        """
        Retrieves the US Consumer Price Index for All Urban Consumers (CPI-U), not
        seasonally adjusted, or its annual rate of change. All items come from the BLS API
        (about the last three years, for the months it publishes before the BIS);
        the core index and the components from FRED when a FRED API key is set, for the
        full history, and from the BLS otherwise.

        Args:
            measure (str): "inflation_rate" or "consumer_price_index".
            component (str): "all", "core", "energy", "food", "goods", "services" or
                "shelter".

        Returns:
            pd.DataFrame: A single "United States" column indexed by month.
        """
        if component != "all" and self._fred_api_key:
            index = fred_model.get_price_index(
                f"consumer_{component}",
                # Two years before the start, for the annual rate of change of the first
                # months and room for growth and rolling.
                buffered_start_date(self._start_date, "monthly"),
                self._end_date,
                self._fred_api_key,
            )
        else:
            index = bls_model.get_consumer_price_index()
            index = (
                index[[component]].rename(columns={component: "United States"})
                if component in index.columns
                else pd.DataFrame()
            )

            if (
                component != "all"
                and not index.empty
                and index.index[0] > pd.Period(self._start_date, freq="M")
            ):
                logger.warning(
                    "Without a FRED API key, the US %s consumer price index starts in %s: "
                    "the keyless BLS API only returns about the last three years. Set a "
                    "FRED API key for the full history.",
                    component,
                    index.index[0].strftime("%Y-%m"),
                )

        if measure == "inflation_rate":
            # A month the BLS did not publish (October 2025) leaves its rate and the one a
            # year later empty rather than compare the wrong months.
            return change_over_periods(index, 12)

        return index

    def _default_period(
        self, period: str | None, highest: str, quarterly_available: bool = True
    ) -> str:
        """
        The period to use when none is given: quarterly when the class was created with
        quarterly=True and the indicator has it, otherwise the highest frequency the
        indicator is published at (daily, then weekly, monthly, quarterly and yearly).

        A source that cannot be reached, for example because of a rate limit, is reported
        rather than answered with a coarser frequency from another source.

        Args:
            period (str | None): The period passed to the method.
            highest (str): The highest frequency the indicator is published at.
            quarterly_available (bool): Whether the indicator is published quarterly.

        Returns:
            str: The period to retrieve.
        """
        if period is not None:
            return period

        if self._quarterly and quarterly_available:
            return "quarterly"

        return highest

    def _get_overnight_rate(
        self, countries: list[str] | str | None, period: str = "daily"
    ) -> pd.DataFrame:
        """
        Retrieves the daily overnight rates (€STR, SONIA, TONA and SOFR), unrounded and
        with history before the start date for the growth, lag and rolling windows of the
        period.

        Args:
            countries (list[str] | str | None): The requested countries, so the sources of
                other countries are not retrieved.
            period (str, optional): The period the rates are returned in, which sets how
                much history is retrieved. Defaults to "daily".

        Returns:
            pd.DataFrame: The daily rates as decimals, a column per country.
        """
        start_date = buffered_start_date(self._start_date, period)
        end_date = self._end_date

        def secured_overnight_financing_rate() -> pd.DataFrame:
            # The New York Fed publishes the full history of SOFR in one file.
            rate = fed_model.get_secured_overnight_financing_rate()
            return (
                rate[["Rate"]].rename(columns={"Rate": "United States"})
                if not rate.empty
                else rate
            )

        return self._combine_sources_for(
            [
                (
                    lambda: ecb_model.get_overnight_rate(start_date, end_date),
                    "Euro Area",
                ),
                (
                    lambda: boe_model.get_overnight_rate(start_date, end_date),
                    "United Kingdom",
                ),
                (
                    lambda: boj_model.get_overnight_rate(start_date, end_date),
                    "Japan",
                ),
                (secured_overnight_financing_rate, "United States"),
            ],
            countries,
        )

    def _collect_country_curves(
        self,
        sources: dict,
        countries: list[str] | str | None,
        period: str,
        indicator: str,
    ) -> pd.DataFrame:
        """
        Retrieves a curve per requested country and combines them with a column per country
        and maturity, the layout of FixedIncome.get_government_bond_yield_curve.

        Args:
            sources (dict): Per country, a function of the start date that returns its daily
                curve with a column per maturity, already in order.
            countries (list[str] | str | None): The countries to retrieve, None for all.
            period (str): "daily", "weekly" or "monthly".
            indicator (str): The name of the indicator, used in the log.

        Returns:
            pd.DataFrame: The curves, indexed by date with a (Country, Maturity) column.
        """
        if countries is not None and not isinstance(countries, str | list | tuple):
            raise TypeError(
                "The countries must be a country name or a list of country names, such as "
                f"'Germany' or ['Germany', 'United States'], not a {type(countries).__name__} ({countries!r})."
            )

        requested = (
            list(sources)
            if countries is None
            else [countries] if isinstance(countries, str) else list(countries)
        )

        if unavailable := [country for country in requested if country not in sources]:
            logger.warning(
                "The %s is not available for %s. It covers %s.",
                indicator,
                ", ".join(unavailable),
                ", ".join(sources),
            )

        start_date = buffered_start_date(self._start_date, period)
        curves = {}

        for country in requested:
            if country not in sources:
                continue

            curve = sources[country](start_date)

            if not curve.empty:
                curves[country] = resample_to_period(curve.dropna(how="all"), period)

        if not curves:
            return pd.DataFrame()

        combined = pd.concat(curves, axis=1).sort_index()
        combined.index.name = None
        combined.columns.names = ["Country", "Maturity"]

        return combined
