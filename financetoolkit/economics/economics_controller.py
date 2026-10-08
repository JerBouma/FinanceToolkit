"""Economics Module"""

__docformat__ = "google"


import os
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from financetoolkit import fmp_model, helpers, historical_model
from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import Cache, set_active_cache
from financetoolkit.economics import (
    bis_model,
    bls_model,
    boe_model,
    boj_model,
    cboe_model,
    dnb_model,
    ecb_model,
    eex_model,
    esrb_model,
    eurostat_model,
    fmp_model as economics_fmp_model,
    frb_model,
    fred_model,
    freddie_mac_model,
    gmdb_model,
    ibge_model,
    imf_model,
    macrohistory_model,
    mof_model,
    nber_model,
    ngfs_model,
    oecd_model,
    ons_model,
    sbj_model,
    shiller_model,
    stoxx_model,
    treasury_model,
    yfinance_model,
)
from financetoolkit.economics.helpers import (
    buffered_start_date,
    check_period_type,
    combine_sources,
    extend_with_recent,
    resample_to_period,
    validate_period,
)
from financetoolkit.fixedincome import bundesbank_model, fed_model
from financetoolkit.utilities import validation_model
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import finalize_dataset

logger = get_logger()

# The listed funds that track asset classes without a public return index, by asset class.
ASSET_CLASS_PROXIES = {
    "Private Equity": "PSP",
    "Infrastructure": "IGF",
    "Hedge Funds": "QAI",
    "Merger Arbitrage": "MNA",
    "Private Credit": "BIZD",
    "Real Estate": "VNQ",
    "Investment Grade Credit": "LQD",
    "High Yield Credit": "HYG",
}

FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,
# pylint: disable=too-many-locals,line-too-long,too-many-public-methods
# ruff: noqa: E501


class Economics:
    """
    The Economics module contains methods to retrieve economic data from the OECD.
    These can be anything ranging from Gross Domestic Product (GDP) to Inflation
    to Consumer Price Index (CPI) and more.
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
        gmdb_forecasts: bool = False,
    ):
        """
        Initializes the Economics Controller Class.

        Args:
            start_date (str | None, optional): The start date to retrieve data from. Defaults to None.
            end_date (str | None, optional): The end date to retrieve data from. Defaults to None.
            gmdb_source (bool, optional): If True, retrieves data from the GMDB source. Defaults to True.
            quarterly (bool | None, optional): If True, returns quarterly data; otherwise, returns yearly data.
                Defaults to None. This only works for data retrieved from the OECD source.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            fred_api_key (str, optional): A FRED API key used to retrieve US-specific labor market and
                real-activity indicators (e.g. Nonfarm Payrolls, Initial Jobless Claims, Retail Sales).
                Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html. Can also be set
                via the FRED_API_KEY environment variable. Defaults to the value of FRED_API_KEY if set,
                otherwise an empty string.
            allow_stale_oecd_cache (bool, optional): the OECD API enforces a hard rate limit (60
                downloads/hour). When True, a 429 response falls back to the most recently cached
                successful response for that exact query instead of returning empty data -- opt-in,
                since the served data may not be the most up-to-date. Every successful OECD response
                is cached regardless of this setting. Defaults to True.
            cache (Cache | None, optional): The incremental cache used for the OECD, FRED and Global
                Macro Database requests this module makes. Defaults to None, which disables caching.
            api_key (str, optional): A FinancialModelingPrep API key, only needed for the economic
                calendar and the market risk premium. Obtain one at https://www.jeroenbouma.com/fmp.
                Defaults to an empty string.
            gmdb_forecasts (bool, optional): The Global Macro Database extends most yearly
                series with the IMF's World Economic Outlook projections up to five years
                ahead. When True, those projected years are included; by default a series
                ends with its last observation. Defaults to False.

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
        self._cache = cache
        # A copied documentation example passes the placeholder key, treated as no key at all.
        self._api_key = validation_model.resolve_api_key(api_key)

        # Published once here so the OECD and FRED free functions read it back.
        set_active_cache(cache)
        oecd_model.configure_oecd_cache(allow_stale_oecd_cache)

        self._gmdb_source: bool = gmdb_source
        self._gmdb_forecasts: bool = gmdb_forecasts

        # The Global Macro Database is one file of around 60 MB, so it is only retrieved
        # when a method first needs it, see _get_gmdb_dataset.
        self._gmbd_dataset: pd.DataFrame = pd.DataFrame()
        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding
        self._fred_api_key: str = fred_api_key

    def _get_gmdb_dataset(self) -> pd.DataFrame:
        """
        Retrieves the Global Macro Database the first time a method needs it, through the
        cache when one is set, and keeps it for the methods called after.

        Returns:
            pd.DataFrame: The dataset, with a (variable, country) column per series.
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset(
                cache=self._cache, include_forecasts=self._gmdb_forecasts
            )

        return self._gmbd_dataset

    def _get_gmdb_series(self, variable: str, in_percent: bool = False) -> pd.DataFrame:
        """
        Retrieves one variable of the Global Macro Database with a column per country.

        Args:
            variable (str): The variable, e.g. "hcons_GDP".
            in_percent (bool, optional): Whether the variable is quoted in percentage points,
                which is divided by 100. Defaults to False.

        Returns:
            pd.DataFrame: The variable, indexed by year with a column per country.
        """
        return gmdb_model.get_series(self._get_gmdb_dataset(), variable, in_percent)

    @staticmethod
    def _consumption_variable(component: str) -> str:
        """
        Returns the Global Macro Database variable of a part of final consumption.

        Args:
            component (str): "total", "household" or "government".

        Returns:
            str: The variable, e.g. "hcons".

        Raises:
            ValueError: When the component is not one of the three.
        """
        variables = {"total": "cons", "household": "hcons", "government": "gcons"}

        if component not in variables:
            raise ValueError(
                f"The component must be one of {', '.join(map(repr, variables))}, not {component!r}."
            )

        return variables[component]

    @staticmethod
    def _government_variable(variable: str, level: str) -> str:
        """
        Returns the Global Macro Database variable of a government series at a level of
        government: "govdebt" is the consolidated debt, "gen_govdebt" the general government
        debt and "cgovdebt" the central government debt.

        Args:
            variable (str): The consolidated variable, e.g. "govdebt_GDP".
            level (str): "consolidated", "general" or "central".

        Returns:
            str: The variable at that level.

        Raises:
            ValueError: When the level is not one of the three.
        """
        prefixes = {"consolidated": "", "general": "gen_", "central": "c"}

        if level not in prefixes:
            raise ValueError(
                f"The level must be one of {', '.join(map(repr, prefixes))}, not {level!r}."
            )

        return f"{prefixes[level]}{variable}"

    def _require_fred_api_key(self) -> None:
        if not self._fred_api_key:
            logger.warning(
                "No FRED API key found. This indicator is sourced from FRED (Federal "
                "Reserve Economic Data) and requires a key to access — registration is "
                "entirely free and takes about a minute at "
                "https://fred.stlouisfed.org/docs/api/api_key.html. Once you have one, "
                "pass it via the fred_api_key argument or set the FRED_API_KEY "
                "environment variable."
            )
            raise ValueError(
                "A FRED API key is required to retrieve this indicator. Obtain a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and pass it via the "
                "fred_api_key argument or set the FRED_API_KEY environment variable."
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
        frames = helpers.run_in_parallel(
            lambda fetch: fetch(), [(fetch,) for fetch in needed]
        )
        combined = combine_sources(frames)

        # A source's own index name (such as "Effective Date") is not kept, as it was not
        # when every source was combined.
        combined.index.name = None

        return combined

    def _get_monthly_price_data(
        self, measure: str, countries: list[str] | str | None = None
    ) -> pd.DataFrame:
        """
        Combines the monthly consumer prices of every country from its most current source.

        The national statistical offices publish first, so they are preferred: Eurostat
        (the euro area and the European Economic Area, including the flash estimate), the
        Office for National Statistics (United Kingdom) and the Statistics Bureau of Japan
        (Japan). Every other country comes from the Bank for International Settlements,
        which compiles the national consumer price indices of around sixty countries,
        including the United States, China, India and Brazil. Each country comes from the
        source with the most recent month. None of them need an API key, and none have the
        tight request limit of the OECD API.

        Args:
            measure (str): "inflation_rate" for the annual rate of change or
                "consumer_price_index" for the index itself.
            countries (list[str] | str | None, optional): The requested countries, so the
                sources of other countries are not retrieved. Defaults to None.

        Returns:
            pd.DataFrame: One column per country, indexed by month.
        """
        # Only the requested months are asked for, with room for growth and rolling.
        start_date = buffered_start_date(self._start_date, "monthly")
        end_date = self._end_date

        if measure == "inflation_rate":
            return self._combine_sources_for(
                [
                    (
                        lambda: eurostat_model.get_inflation_rate(start_date, end_date),
                        None,
                    ),
                    (ons_model.get_inflation_rate, "United Kingdom"),
                    (sbj_model.get_inflation_rate, "Japan"),
                    (
                        lambda: bis_model.get_consumer_prices(
                            measure, start_date, end_date
                        ),
                        None,
                    ),
                    # Around a hundred further countries the BIS does not cover.
                    (lambda: imf_model.get_inflation_rate(start_date, end_date), None),
                ],
                countries,
            )

        return self._combine_sources_for(
            [
                (
                    lambda: eurostat_model.get_consumer_price_index(
                        start_date, end_date
                    ),
                    None,
                ),
                (ons_model.get_consumer_price_index, "United Kingdom"),
                (sbj_model.get_consumer_price_index, "Japan"),
                (
                    lambda: bis_model.get_consumer_prices(
                        measure, start_date, end_date
                    ),
                    None,
                ),
            ],
            countries,
        )

    def _get_daily_long_term_interest_rate(
        self, countries: list[str] | str | None = None
    ) -> pd.DataFrame:
        """
        Combines the daily 10-year government bond yields published without an API key:
        the U.S. Department of the Treasury, the Bank of England, the Japanese Ministry of
        Finance and the European Central Bank (the euro area yield curve).

        Returns:
            pd.DataFrame: One column per country, indexed by day.
        """
        start_date = buffered_start_date(self._start_date, "daily")
        end_date = self._end_date

        return self._combine_sources_for(
            [
                (
                    lambda: treasury_model.get_long_term_interest_rate(
                        start_date, end_date
                    ),
                    "United States",
                ),
                (
                    lambda: boe_model.get_long_term_interest_rate(start_date, end_date),
                    "United Kingdom",
                ),
                # The Ministry of Finance publishes one file with the full history.
                (mof_model.get_long_term_interest_rate, "Japan"),
                (
                    lambda: ecb_model.get_long_term_interest_rate(start_date, end_date),
                    "Euro Area",
                ),
            ],
            countries,
        )

    @handle_errors
    def get_gross_domestic_product(
        self,
        countries: list[str] | str | None = None,
        inflation_adjusted: bool = False,
        gmdb_source: bool | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Gross Domestic Product for a variety of countries over
        time from the OECD. The Gross Domestic Product is the total value
        of goods produced and services provided in a country during one year.

        Note that the OECD source reports GDP on a per capita basis, i.e. the total
        Gross Domestic Product divided by the population of the country, whereas the
        Global Macro Database (GMDB) source reports the total (not per capita) figure.
        The two are also expressed in different units: the OECD source is in current-price
        US dollars per person converted with Purchasing Power Parities (PPPs), which makes
        the level comparable across countries, while the GMDB source is in millions of
        national currency. Both are annual.

        The data is returned as levels. To obtain period-on-period changes (e.g. year
        on year or quarter on quarter growth), set `growth=True` and use `lag` to
        control how many periods back the comparison is made.

        See definition: https://data.oecd.org/gdp/gross-domestic-product-gdp.htm

        It is also possible to acquire the data from the Global Macro Database (GMDB) source which
        also provides inflation adjusted data. For more information see:
        https://www.globalmacrodata.com/documentation.html

        Also known as: GDP, national income, economic growth.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data. Defaults to False.
            gmdb_source (bool | None, optional): If True, retrieves data from the GMDB source. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. With inflation_adjusted=True, real GDP in US dollars. Always from the GMDB. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_gross_domestic_product(inflation_adjusted=True, countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Netherlands |     Germany |       China |
        |:-----|--------------:|------------:|------------:|
        | 2015 |        699213 | 3.08436e+06 | 7.02512e+07 |
        | 2016 |        716132 | 3.15026e+06 | 7.50097e+07 |
        | 2017 |        736049 | 3.24792e+06 | 8.01788e+07 |
        | 2018 |        752633 | 3.28523e+06 | 8.55972e+07 |
        | 2019 |        769904 | 3.31842e+06 | 9.07907e+07 |
        | 2020 |        740429 | 3.16955e+06 | 9.29147e+07 |
        | 2021 |        786377 | 3.29206e+06 | 1.00878e+08 |
        | 2022 |        825789 | 3.35415e+06 | 1.04039e+08 |
        | 2023 |        820955 | 3.33152e+06 | 1.09674e+08 |
        | 2024 |        829705 | 3.3158e+06  | 1.15158e+08 |
        | 2025 |        844596 | 3.32724e+06 | 1.20916e+08 |
        """
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source or inflation_adjusted or usd:
            if not gmdb_source:
                logger.info(
                    "OECD does not provide inflation adjusted or US dollar GDP data, using "
                    "GMDB source instead."
                )

            # The real and nominal GDP, in national currency or US dollars.
            variable = ("rGDP" if inflation_adjusted else "nGDP") + (
                "_USD" if usd else ""
            )
            gross_domestic_product = self._get_gmdb_series(variable)
        else:
            gross_domestic_product = oecd_model.get_annual_gross_domestic_product(
                start_date=self._start_date, end_date=self._end_date
            )

        return finalize_dataset(
            dataset=gross_domestic_product,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Gross Domestic Product",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_gross_domestic_product_growth(
        self,
        countries: list[str] | str | None = None,
        year_over_year: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The quarterly growth of real Gross Domestic Product (GDP) is the headline measure
        of how fast an economy grows: the change in the volume of everything produced,
        adjusted for inflation, seasonal patterns and the number of working days. Two
        consecutive quarters of negative growth is the common rule of thumb for a
        technical recession.

        By default the growth is measured on the previous quarter (quarter on quarter), as
        Eurostat and the Office for National Statistics report it. Set year_over_year=True
        for the change on the same quarter a year earlier. The growth is not annualized,
        so a US figure is about a quarter of the annualized rate the Bureau of Economic
        Analysis headlines.

        Every country comes from the most current source without an API key: Eurostat for
        the euro area and the countries it covers (including the preliminary flash estimate
        for the euro area, about thirty days after the quarter ends), the Office for
        National Statistics for the United Kingdom and the OECD Quarterly National Accounts
        for every other country, including the United States and Japan.

        The growth is expressed as a decimal fraction (0.006 for 0.6%).

        Also known as: GDP growth, real GDP growth, economic growth, quarter-on-quarter
        growth.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            year_over_year (bool, optional): Whether to return the change on the same quarter a
                year earlier instead of on the previous quarter. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the quarterly growth of real GDP per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-01-01', end_date='2026-09-30')

        economics.get_gross_domestic_product_growth(
            countries=['Euro Area', 'Germany', 'United Kingdom', 'United States', 'Japan']
        )
        ```

        Which returns:

        |        |   Euro Area |   Germany |   United Kingdom |   United States |   Japan |
        |:-------|------------:|----------:|-----------------:|----------------:|--------:|
        | 2025Q1 |       0.005 |     0.001 |            0.006 |          0.0004 |  0.005  |
        | 2025Q2 |       0     |     0     |            0     |          0.0099 |  0.0012 |
        | 2025Q3 |       0.003 |     0     |            0.002 |          0.0096 | -0.0037 |
        | 2025Q4 |       0.002 |     0.003 |            0     |          0.0005 |  0.0026 |
        | 2026Q1 |       0     |     0.004 |            0.006 |          0.0062 |  0.0048 |
        | 2026Q2 |       0.006 |     0.003 |            0.005 |          0.0055 |  0.0036 |
        """
        start_date = buffered_start_date(self._start_date, "monthly")

        gross_domestic_product_growth = self._combine_sources_for(
            [
                (
                    lambda: eurostat_model.get_gross_domestic_product_growth(
                        start_date, self._end_date, year_over_year=year_over_year
                    ),
                    None,
                ),
                (
                    lambda: ons_model.get_gross_domestic_product_growth(
                        year_over_year=year_over_year
                    ),
                    "United Kingdom",
                ),
                (
                    lambda: oecd_model.get_gross_domestic_product_growth(
                        year_over_year=year_over_year,
                        start_date=self._start_date,
                        end_date=self._end_date,
                    ),
                    None,
                ),
            ],
            countries,
        )

        return finalize_dataset(
            dataset=gross_domestic_product_growth,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Gross Domestic Product Growth",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_gross_domestic_product_deflator(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Gross Domestic Product Deflator for a variety of countries over
        time from the Global Macro Database (GMDB). The GDP deflator is a measure of
        the price of all domestically produced final goods and services in an economy
        relative to the price level in a base year which can vary per country.

        The deflator is an index, set to 100 in the base year, which can vary per country,
        and is annual.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: GDP deflator, implicit price deflator.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product Deflator

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_gross_domestic_product_deflator(countries=['United States', 'Canada', 'Russian Federation'])
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |         100     |  100     |              100     |
        | 2016 |         100.95  |  100.717 |              102.844 |
        | 2017 |         102.758 |  103.306 |              108.345 |
        | 2018 |         105.112 |  105.012 |              119.179 |
        | 2019 |         106.847 |  106.636 |              123.068 |
        | 2020 |         108.284 |  107.778 |              124.173 |
        | 2021 |         113.225 |  116.168 |              146.784 |
        | 2022 |         121.278 |  125.325 |              173.542 |
        | 2023 |         125.766 |  127.261 |              185.098 |
        | 2024 |         128.887 |  130.738 |              204.816 |
        | 2025 |         132.537 |  134.076 |              214.754 |

        """

        gross_domestic_product_deflator = (
            gmdb_model.get_gross_domestic_product_deflator(
                gmd_dataset=self._get_gmdb_dataset()
            )
        )

        return finalize_dataset(
            dataset=gross_domestic_product_deflator,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Gross Domestic Product Deflator",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_real_gross_domestic_product_usd(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Real Gross Domestic Product expressed in cross-country comparable US Dollars
        for a variety of countries over time from the Global Macro Database (GMDB). This is the
        inflation-adjusted GDP of a country converted into US Dollars, which makes it possible to
        directly compare the economic output of countries that use different currencies without
        having to perform the currency conversion or inflation adjustment yourself.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: real GDP in USD, cross-country comparable GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Real Gross Domestic Product in US Dollars

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_real_gross_domestic_product_usd(countries=['United States', 'Japan', 'Germany'])
        ```

        Which returns:

        |      |   United States |       Japan |     Germany |
        |:-----|----------------:|------------:|------------:|
        | 2015 |     1.8295e+07  | 4.53311e+06 | 3.41986e+06 |
        | 2016 |     1.86279e+07 | 4.56506e+06 | 3.49293e+06 |
        | 2017 |     1.90857e+07 | 4.63917e+06 | 3.60121e+06 |
        | 2018 |     1.96519e+07 | 4.67787e+06 | 3.64259e+06 |
        | 2019 |     2.01596e+07 | 4.66344e+06 | 3.67938e+06 |
        | 2020 |     1.974e+07   | 4.46369e+06 | 3.51432e+06 |
        | 2021 |     2.09545e+07 | 4.62279e+06 | 3.65016e+06 |
        | 2022 |     2.14834e+07 | 4.68435e+06 | 3.719e+06   |
        | 2023 |     2.21138e+07 | 4.71811e+06 | 3.69391e+06 |
        | 2024 |     2.27315e+07 | 4.71016e+06 | 3.67648e+06 |
        | 2025 |     2.32103e+07 | 4.75982e+06 | 3.68916e+06 |
        """

        real_gross_domestic_product_usd = (
            gmdb_model.get_real_gross_domestic_product_usd(
                gmd_dataset=self._get_gmdb_dataset()
            )
        )

        return finalize_dataset(
            dataset=real_gross_domestic_product_usd,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Real Gross Domestic Product (USD)",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_real_gross_domestic_product_per_capita(
        self,
        countries: list[str] | str | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Real Gross Domestic Product per Capita for a variety of countries over time from
        the Global Macro Database (GMDB). This is the inflation-adjusted Gross Domestic Product
        (GDP) divided by the total population of a country, which gives an indication of the
        average economic output (and by extension, living standard) per person.

        Formula:

            Real GDP per Capita = Real Gross Domestic Product / Population

        This uses the Global Macro Database's own precomputed per-capita series rather than
        dividing GDP by population manually, which avoids subtle mismatches that can arise from
        differences in population coverage or timing between the two underlying series.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: real GDP per capita, real income per capita, standard of living.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Real Gross Domestic Product per Capita

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_real_gross_domestic_product_per_capita(countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Netherlands |   Germany |   China |
        |:-----|--------------:|----------:|--------:|
        | 2015 |       41275.9 |   37916.3 | 50786.7 |
        | 2016 |       42051.2 |   38480.1 | 53873.9 |
        | 2017 |       42965.9 |   39590.4 | 57266.1 |
        | 2018 |       43676.5 |   39987.4 | 60905.5 |
        | 2019 |       44387.7 |   40368.7 | 64386.9 |
        | 2020 |       42450.9 |   38593.8 | 65798   |
        | 2021 |       44851.3 |   40149.7 | 71412.9 |
        | 2022 |       46652.1 |   40642.1 | 73695.2 |
        | 2023 |       45922.4 |   40000.8 | 77801.2 |
        | 2024 |       46110.1 |   39702.4 | 81884.3 |
        | 2025 |       46646.5 |   39835.9 | 86171.3 |
        """

        real_gross_domestic_product_per_capita = self._get_gmdb_series(
            "rGDP_pc_USD" if usd else "rGDP_pc"
        )

        return finalize_dataset(
            dataset=real_gross_domestic_product_per_capita,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Real Gross Domestic Product per Capita",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_output_gap(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Output Gap for a variety of countries over time from the OECD Economic Outlook.
        The output gap is the difference between actual Gross Domestic Product (GDP) and
        estimated potential GDP, expressed as a percentage of potential GDP. Potential GDP is the
        level of output an economy can sustain over the long term without generating excess
        inflationary or disinflationary pressure, based on the full, non-inflationary use of its
        productive resources (labour, capital and technology).

        A positive output gap indicates the economy is running above its long-run potential
        (an economic "boom", typically associated with rising inflationary pressure), while a
        negative output gap indicates the economy is running below potential (an economic
        "slack", typically associated with rising unemployment and disinflationary pressure).
        The output gap therefore complements indicators such as the Inflation Rate and
        Unemployment Rate as a measure of where an economy sits within the business cycle.

        Formula:

            Output Gap = (Actual GDP - Potential GDP) / Potential GDP

        This data is only available on a yearly basis, since the OECD Economic Outlook is
        published as a set of annual projections and estimates.

        Changed in v2.2.0: the result is now a decimal fraction (-0.0422) rather than the
        percentage of potential GDP the OECD publishes (-4.2231), matching every other rate
        and ratio in this class. Multiply by 100 to recover the published figure. Note that
        a small gap loses resolution at the default rounding of 4 decimals -- pass a larger
        `rounding` when the sub-basis-point detail matters.

        See definition: https://www.oecd.org/en/data/indicators/output-gaps.html

        Also known as: business cycle gap, GDP gap.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Output Gap as a decimal fraction of
            potential GDP.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date='2022-01-01')

        economics.get_output_gap(countries=['United States', 'Germany', 'Japan'])
        ```

        Which returns:

        |      |   United States |   Germany |   Japan |
        |:-----|----------------:|----------:|--------:|
        | 2018 |          0.0002 |    0.019  |  0.0194 |
        | 2019 |          0.0015 |    0.0201 |  0.0072 |
        | 2020 |         -0.0422 |   -0.0315 | -0.0422 |
        | 2021 |         -0.0073 |   -0.0005 | -0.0137 |
        | 2022 |         -0.0066 |    0.0108 | -0.0049 |
        """
        output_gap = oecd_model.get_output_gap(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=output_gap,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Output Gap",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_total_consumption(
        self,
        countries: list[str] | str | None = None,
        inflation_adjusted: bool = False,
        component: str = "total",
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Total Consumption for a variety of countries over time from the
        Global Macro Database (GMDB). Total Consumption is final consumption expenditure:
        the goods and services bought by households and the non-profit institutions serving
        them (household consumption) and by the government (government consumption).
        Select either part with the component parameter.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are; with
        usd=True the levels are in millions of US dollars. With inflation_adjusted=True the
        level is in 2015 prices: the nominal level divided by the GDP deflator.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: final consumption expenditure, household consumption, private
        consumption, government consumption.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data. Defaults to False.
            component (str, optional): The part of final consumption expenditure: "total", "household" (households and the non-profit institutions serving them) or "government". Defaults to "total".
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency. Cannot be combined with inflation_adjusted. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Total Consumption

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date="2025-12-31")

        total_consumption = economics.get_total_consumption()

        total_consumption.loc[:, ['Netherlands', 'France', 'Poland']]
        ```

        Which returns:

        |      |   Netherlands |      France |      Poland |
        |:-----|--------------:|------------:|------------:|
        | 2018 |        543044 | 1.84623e+06 | 1.64263e+06 |
        | 2019 |        566526 | 1.88924e+06 | 1.75019e+06 |
        | 2020 |        558421 | 1.82884e+06 | 1.78274e+06 |
        | 2021 |        606447 | 1.94983e+06 | 1.99473e+06 |
        | 2022 |        679151 | 2.08811e+06 | 2.36421e+06 |
        | 2023 |        734728 | 2.22836e+06 | 2.61398e+06 |
        | 2024 |        776762 | 2.30127e+06 | 2.85179e+06 |
        | 2025 |        813178 | 2.33891e+06 | 3.08623e+06 |
        """
        variable = self._consumption_variable(component)

        if inflation_adjusted and usd:
            raise ValueError(
                "The Global Macro Database has no consumption in constant US dollars, so "
                "inflation_adjusted and usd cannot be combined."
            )

        if inflation_adjusted:
            total_consumption = (
                gmdb_model.get_real_total_consumption(
                    gmd_dataset=self._get_gmdb_dataset()
                )
                if variable == "cons"
                else (
                    self._get_gmdb_series(variable)
                    / self._get_gmdb_series("deflator")
                    * 100
                ).dropna(how="all", axis="columns")
            )
        else:
            total_consumption = self._get_gmdb_series(
                f"{variable}_USD" if usd else variable
            )

        return finalize_dataset(
            dataset=total_consumption,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Total Consumption",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_total_consumption_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        component: str = "total",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Total Consumption to GDP Ratio for a variety of countries over time from the
        Global Macro Database (GMDB). The Total Consumption to GDP Ratio is the ratio of the
        total amount of money spent by households on consumer goods and services to the Gross
        Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.8100 for 81.00% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: consumption share of GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            component (str, optional): The part of final consumption expenditure: "total", "household" (households and the non-profit institutions serving them) or "government". Defaults to "total".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Total Consumption to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date="2025-12-31")

        total_consumption_to_gdp_ratio = economics.get_total_consumption_to_gdp_ratio()

        total_consumption_to_gdp_ratio.loc[:, ['Netherlands', 'France', 'Poland']]
        ```

        Which returns:

        |      |   Netherlands |   France |   Poland |
        |:-----|--------------:|---------:|---------:|
        | 2018 |        0.6896 |   0.7833 |   0.7655 |
        | 2019 |        0.6828 |   0.7757 |   0.7572 |
        | 2020 |        0.6838 |   0.7892 |   0.755  |
        | 2021 |        0.6805 |   0.7784 |   0.7496 |
        | 2022 |        0.6836 |   0.7871 |   0.7637 |
        | 2023 |        0.6997 |   0.7873 |   0.7662 |
        | 2024 |        0.6927 |   0.7878 |   0.7793 |
        | 2025 |        0.6895 |   0.7838 |   0.7901 |
        """

        total_consumption_to_gdp_ratio = self._get_gmdb_series(
            f"{self._consumption_variable(component)}_GDP", in_percent=True
        )

        return finalize_dataset(
            dataset=total_consumption_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Total Consumption to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_investment(
        self,
        countries: list[str] | str | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Investment for a variety of countries over time from the Global Macro Database (GMDB).
        Investment is the total amount of money spent by businesses on capital goods, such as machinery,
        equipment, and buildings.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: total investment, capital formation.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Investment

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2014-01-01', end_date="2025-12-31")

        investment = economics.get_investment()

        investment.loc[:, ['United States', 'Portugal', 'China']]
        ```

        Which returns:

        |      |   United States |   Portugal |       China |
        |:-----|----------------:|-----------:|------------:|
        | 2014 |     3.68031e+06 |    26574.5 | 2.97145e+07 |
        | 2015 |     3.91791e+06 |    28591.6 | 3.00218e+07 |
        | 2016 |     3.92796e+06 |    29873.8 | 3.20756e+07 |
        | 2017 |     4.14909e+06 |    34109.1 | 3.60559e+07 |
        | 2018 |     4.45542e+06 |    37975.2 | 4.05422e+07 |
        | 2019 |     4.66767e+06 |    40256   | 4.29201e+07 |
        | 2020 |     4.58237e+06 |    39037.7 | 4.40568e+07 |
        | 2021 |     5.07881e+06 |    45322.5 | 5.01014e+07 |
        | 2022 |     5.739e+06   |    52142.2 | 5.22678e+07 |
        | 2023 |     5.99898e+06 |    55414.9 | 5.3233e+07  |
        | 2024 |     6.30862e+06 |    58942.6 | 5.47878e+07 |
        | 2025 |     6.56842e+06 |    64530.2 | 5.47964e+07 |
        """

        investment = self._get_gmdb_series("inv_USD" if usd else "inv")

        return finalize_dataset(
            dataset=investment,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Investment",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_investment_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Investment to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Investment to GDP Ratio is the ratio of the total amount of money spent by businesses on capital goods,
        such as machinery, equipment, and buildings to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.2248 for 22.48% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: investment rate.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Investment to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2019-01-01', end_date="2025-12-31")

        investment_to_gdp_ratio = economics.get_investment_to_gdp_ratio()

        investment_to_gdp_ratio.loc[:, ['Australia', 'Japan', 'Turkey']]
        ```

        Which returns:

        |      |   Australia |   Japan |   Turkey |
        |:-----|------------:|--------:|---------:|
        | 2019 |      0.2265 |  0.2732 |   0.2751 |
        | 2020 |      0.2246 |  0.2689 |   0.3415 |
        | 2021 |      0.2333 |  0.2733 |   0.3438 |
        | 2022 |      0.2392 |  0.2831 |   0.386  |
        | 2023 |      0.2389 |  0.2778 |   0.3459 |
        | 2024 |      0.2453 |  0.2766 |   0.3073 |
        | 2025 |      0.2455 |  0.2778 |   0.3167 |
        """

        investment_to_gdp_ratio = gmdb_model.get_investment_to_gdp_ratio(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=investment_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Investment to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_fixed_investment(
        self,
        countries: list[str] | str | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Fixed Investment for a variety of countries over time from the Global Macro Database (GMDB).
        Fixed Investment is the total amount of money spent by businesses on capital goods, such as machinery,
        equipment, and buildings that are expected to last for more than one year.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: gross fixed capital formation, capital investment.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Fixed Investment

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date="2025-12-31")

        fixed_investment = economics.get_fixed_investment()

        fixed_investment.loc[:, ['United Kingdom', 'Germany', 'France']]
        ```

        Which returns:

        |      |   United Kingdom |   Germany |   France |
        |:-----|-----------------:|----------:|---------:|
        | 2020 |           387159 |    733569 |   519445 |
        | 2021 |           430541 |    775371 |   587672 |
        | 2022 |           490449 |    858069 |   622842 |
        | 2023 |           520091 |    894583 |   647114 |
        | 2024 |           539666 |    887571 |   645687 |
        | 2025 |           573803 |    911184 |   658386 |
        """

        fixed_investment = self._get_gmdb_series("finv_USD" if usd else "finv")

        return finalize_dataset(
            dataset=fixed_investment,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Fixed Investment",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_fixed_investment_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Fixed Investment to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Fixed Investment to GDP Ratio is the ratio of the total amount of money spent by businesses on capital goods,
        such as machinery, equipment, and buildings that are expected to last for more than one year to the Gross Domestic
        Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.2179 for 21.79% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: investment to GDP ratio.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Fixed Investment to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2000-01-01', end_date="2025-12-31")

        fixed_investment_to_gdp_ratio = economics.get_fixed_investment_to_gdp_ratio()

        fixed_investment_to_gdp_ratio.loc[:, ['Austria', 'Germany', 'Switzerland']]
        ```

        Which returns:

        |      |   Austria |   Germany |   Switzerland |
        |:-----|----------:|----------:|--------------:|
        | 2016 |    0.2332 |    0.2    |        0.2674 |
        | 2017 |    0.2385 |    0.2016 |        0.2751 |
        | 2018 |    0.2431 |    0.2086 |        0.2698 |
        | 2019 |    0.2508 |    0.2121 |        0.2699 |
        | 2020 |    0.2513 |    0.213  |        0.2763 |
        | 2021 |    0.2586 |    0.2111 |        0.268  |
        | 2022 |    0.2521 |    0.2154 |        0.2627 |
        | 2023 |    0.2469 |    0.2119 |        0.2689 |
        | 2024 |    0.2355 |    0.2049 |        0.2643 |
        | 2025 |    0.2353 |    0.2035 |        0.2599 |
        """

        fixed_investment_to_gdp_ratio = gmdb_model.get_fixed_investment_to_gdp_ratio(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=fixed_investment_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Fixed Investment to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_exports(
        self,
        countries: list[str] | str | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Exports for a variety of countries over time from the Global Macro Database (GMDB).
        Exports are the total amount of goods and services produced in a country that are sold to
        other countries.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: exports, trade exports.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exports

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='1980-01-01', end_date='1990-01-01')

        economics.get_exports(countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Netherlands |   Germany |    China |
        |:-----|--------------:|----------:|---------:|
        | 1980 |       89051.5 |    165542 |  31845.8 |
        | 1981 |      102962   |    187458 |  43100   |
        | 1982 |      106300   |    202401 |  48143   |
        | 1983 |      109580   |    205497 |  48825.2 |
        | 1984 |      124092   |    230732 |  64861.4 |
        | 1985 |      132292   |    254587 |  85285.2 |
        | 1986 |      114407   |    248907 | 101831   |
        | 1987 |      112478   |    248372 | 142745   |
        | 1988 |      122726   |    267089 | 188203   |
        | 1989 |      138150   |    301859 | 174600   |
        | 1990 |      144534   |    336413 | 264527   |
        """

        exports = self._get_gmdb_series("exports_USD" if usd else "exports")

        return finalize_dataset(
            dataset=exports,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Exports",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_exports_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Exports to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Exports to GDP Ratio is the ratio of the total amount of goods and services produced in a country
        that are sold to other countries to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.1016 for 10.16% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: exports to GDP ratio, trade openness.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exports to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_exports_to_gdp_ratio(countries=['United States', 'Canada', 'Russian Federation'])
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |          0.1241 |   0.3185 |               0.287  |
        | 2016 |          0.1189 |   0.315  |               0.2585 |
        | 2017 |          0.1218 |   0.3145 |               0.2609 |
        | 2018 |          0.1229 |   0.3233 |               0.3079 |
        | 2019 |          0.1179 |   0.3235 |               0.2843 |
        | 2020 |          0.1012 |   0.2947 |               0.2552 |
        | 2021 |          0.1083 |   0.3118 |               0.3    |
        | 2022 |          0.1165 |   0.3372 |               0.2771 |
        | 2023 |          0.1105 |   0.3334 |               0.2277 |
        | 2024 |          0.1097 |   0.3247 |               0.2179 |
        | 2025 |          0.1079 |   0.3133 |               0.1815 |
        """

        exports_to_gdp_ratio = gmdb_model.get_exports_to_gdp_ratio(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=exports_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Exports to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_imports(
        self,
        countries: list[str] | str | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Imports for a variety of countries over time from the Global Macro Database (GMDB).
        Imports are the total amount of goods and services produced in other countries that are
        bought by a country.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: imports, trade imports.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Imports

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date="2025-12-31")

        economics.get_imports(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |           Canada |      Mexico |
        |:-----|----------------:|-----------------:|------------:|
        | 2010 |     2.38956e+06 | 517153           | 4.22154e+06 |
        | 2011 |     2.69548e+06 | 564513           | 4.8355e+06  |
        | 2012 |     2.76932e+06 | 589137           | 5.41225e+06 |
        | 2013 |     2.76638e+06 | 606801           | 5.41757e+06 |
        | 2014 |     2.88744e+06 | 651176           | 5.92175e+06 |
        | 2015 |     2.79494e+06 | 683019           | 6.96894e+06 |
        | 2016 |     2.73883e+06 | 685868           | 8.05624e+06 |
        | 2017 |     2.93159e+06 | 720254           | 8.88029e+06 |
        | 2018 |     3.13117e+06 | 766265           | 9.93588e+06 |
        | 2019 |     3.11668e+06 | 782419           | 9.78484e+06 |
        | 2020 |     2.78239e+06 | 703532           | 9.07008e+06 |
        | 2021 |     3.41793e+06 | 787922           | 1.13375e+07 |
        | 2022 |     3.97432e+06 | 951814           | 1.35378e+07 |
        | 2023 |     3.85986e+06 | 985422           | 1.19891e+07 |
        | 2024 |     4.11383e+06 |      1.01502e+06 | 1.27547e+07 |
        | 2025 |     4.24615e+06 |      1.04369e+06 | 1.41931e+07 |

        """

        imports = self._get_gmdb_series("imports_USD" if usd else "imports")

        return finalize_dataset(
            dataset=imports,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Imports",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_imports_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Imports to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Imports to GDP Ratio is the ratio of the total amount of goods and services produced in other countries
        that are bought by a country to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.1213 for 12.13% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: imports to GDP ratio.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Imports to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date="2025-12-31")

        economics.get_imports_to_gdp_ratio(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |   Canada |   Mexico |
        |:-----|----------------:|---------:|---------:|
        | 2010 |          0.1588 |   0.3104 |   0.3022 |
        | 2011 |          0.1728 |   0.3182 |   0.3167 |
        | 2012 |          0.1704 |   0.3224 |   0.3277 |
        | 2013 |          0.1639 |   0.319  |   0.3195 |
        | 2014 |          0.164  |   0.3264 |   0.3264 |
        | 2015 |          0.1528 |   0.3431 |   0.3623 |
        | 2016 |          0.1456 |   0.3386 |   0.3885 |
        | 2017 |          0.1495 |   0.3365 |   0.3939 |
        | 2018 |          0.1516 |   0.3427 |   0.411  |
        | 2019 |          0.1447 |   0.3382 |   0.3894 |
        | 2020 |          0.1302 |   0.3168 |   0.377  |
        | 2021 |          0.1441 |   0.3107 |   0.4247 |
        | 2022 |          0.1525 |   0.3323 |   0.4585 |
        | 2023 |          0.1388 |   0.3323 |   0.3762 |
        | 2024 |          0.1404 |   0.3265 |   0.3811 |
        | 2025 |          0.138  |   0.3218 |   0.4025 |
        """

        imports_to_gdp_ratio = gmdb_model.get_imports_to_gdp_ratio(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=imports_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Imports to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_trade_balance(
        self,
        countries: list[str] | str | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Trade Balance for a variety of countries over time from the Global Macro
        Database (GMDB). The Trade Balance is the difference between the total value of goods
        and services a country exports and the total value of goods and services it imports. A
        positive trade balance (a "trade surplus") means a country exports more than it imports,
        while a negative trade balance (a "trade deficit") means a country imports more than it
        exports.

        Formula:

            Trade Balance = Exports - Imports

        The balance is annual and expressed in millions of national currency, with a negative
        value marking a trade deficit and a positive value a surplus.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: net exports, balance of trade.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Trade Balance

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date='2023-01-01')

        economics.get_trade_balance(countries=['United States', 'Germany', 'China'])
        ```

        Which returns:

        |      |   United States |   Germany |       China |
        |:-----|----------------:|----------:|------------:|
        | 2020 |         -618554 |    181408 | 2.57979e+06 |
        | 2021 |         -849233 |    185685 | 3.04482e+06 |
        | 2022 |         -937889 |     96043 | 3.97949e+06 |
        | 2023 |         -786495 |    167684 | 2.79265e+06 |
        """

        exports = self._get_gmdb_series("exports_USD" if usd else "exports")
        imports = self._get_gmdb_series("imports_USD" if usd else "imports")

        trade_balance = exports - imports

        return finalize_dataset(
            dataset=trade_balance,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Trade Balance",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_current_account_balance(
        self,
        countries: list[str] | str | None = None,
        usd: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Current Account Balance for a variety of countries over time from the Global Macro Database (GMDB).
        The Current Account Balance is the sum of the balance of trade (exports minus imports of goods and services),
        net factor income (such as interest and dividends) and net transfer payments (such as foreign aid).

        The balance is annual and expressed in millions of national currency, with a negative
        value marking a current account deficit and a positive value a surplus.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: current account, trade balance, balance of payments.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Current Account Balance

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_current_account_balance(countries=['France', 'Germany', 'Italy'])
        ```

        Which returns:

        |      |   France |   Germany |     Italy |
        |:-----|---------:|----------:|----------:|
        | 2015 |    -7158 |    252800 |  20667.3  |
        | 2016 |   -11787 |    285460 |  41949.2  |
        | 2017 |   -12529 |    272550 |  42551.2  |
        | 2018 |   -16441 |    296480 |  44465.2  |
        | 2019 |    14533 |    288300 |  56955.4  |
        | 2020 |   -47596 |    217260 |  62807    |
        | 2021 |     6941 |    256470 |  36974.5  |
        | 2022 |   -37663 |    164120 | -36550.2  |
        | 2023 |   -29398 |    233650 |   5323.05 |
        | 2024 |     2665 |    253790 |  24121.3  |
        | 2025 |    -9345 |    201850 |  25049.5  |
        """

        current_account_balance = self._get_gmdb_series("CA_USD" if usd else "CA")

        return finalize_dataset(
            dataset=current_account_balance,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Current Account Balance",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_current_account_balance_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Current Account Balance to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Current Account Balance to GDP Ratio is the ratio of the sum of the balance of trade (exports minus imports of goods
        and services), net factor income (such as interest and dividends) and net transfer payments (such as foreign aid) to the
        Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (-0.0211 for -2.11% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: current account to GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Current Account Balance to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_current_account_balance_to_gdp_ratio(countries=[
            'Poland', 'Turkey', 'United Kingdom'])
        ```

        Which returns:

        |      |   Poland |   Turkey |   United Kingdom |
        |:-----|---------:|---------:|-----------------:|
        | 2015 |  -0.013  |  -0.0245 |          -0.0478 |
        | 2016 |  -0.0105 |  -0.0254 |          -0.0525 |
        | 2017 |  -0.0117 |  -0.0408 |          -0.0338 |
        | 2018 |  -0.0197 |  -0.0132 |          -0.0372 |
        | 2019 |  -0.0027 |   0.0193 |          -0.0248 |
        | 2020 |   0.024  |  -0.0437 |          -0.0272 |
        | 2021 |  -0.0135 |  -0.0083 |          -0.0076 |
        | 2022 |  -0.0225 |  -0.0513 |          -0.0192 |
        | 2023 |   0.0156 |  -0.035  |          -0.0357 |
        | 2024 |   0.0031 |  -0.0098 |          -0.0295 |
        | 2025 |  -0.0086 |  -0.019  |          -0.0245 |
        """

        current_account_balance_to_gdp_ratio = (
            gmdb_model.get_current_account_balance_to_gdp(
                gmd_dataset=self._get_gmdb_dataset()
            )
        )

        return finalize_dataset(
            dataset=current_account_balance_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Current Account Balance to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_debt(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Debt for a variety of countries over time from the Global Macro Database (GMDB).
        Government Debt is the total amount of money that a government owes to creditors.

        The stock is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: national debt, sovereign debt.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Debt

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_debt(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |        Canada |        Mexico |
        |:-----|----------------:|--------------:|--------------:|
        | 2015 |     1.92907e+07 |   1.83683e+06 |   9.80508e+06 |
        | 2016 |     2.01975e+07 |   1.87318e+06 |   1.14065e+07 |
        | 2017 |     2.08678e+07 |   1.94774e+06 |   1.184e+07   |
        | 2018 |     2.22374e+07 |   2.02901e+06 |   1.26199e+07 |
        | 2019 |     2.34322e+07 |   2.08475e+06 |   1.30327e+07 |
        | 2020 |     2.83333e+07 |   2.62165e+06 |   1.40652e+07 |
        | 2021 |     2.96684e+07 |   2.85548e+06 |   1.51339e+07 |
        | 2022 |     3.10429e+07 |   2.96557e+06 |   1.58852e+07 |
        | 2023 |     3.33635e+07 |   3.11556e+06 |   1.68118e+07 |
        | 2024 |     3.58226e+07 |   3.41999e+06 |   1.97928e+07 |
        | 2025 |     3.811e+07   | nan           | nan           |
        """

        government_debt = self._get_gmdb_series(
            self._government_variable("govdebt", level)
        )

        return finalize_dataset(
            dataset=government_debt,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Debt",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_debt_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Debt to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Debt to GDP Ratio is the ratio of the total amount of money that a government owes to creditors
        to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (1.3173 for 131.73% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: debt-to-GDP ratio, fiscal sustainability.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Debt to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_debt_to_gdp_ratio(countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Netherlands |   Germany |    China |
        |:-----|--------------:|----------:|---------:|
        | 2015 |        0.6382 |    0.7118 |   0.4078 |
        | 2016 |        0.6087 |    0.683  |   0.5003 |
        | 2017 |        0.5597 |    0.6399 |   0.5389 |
        | 2018 |        0.5155 |    0.6076 |   0.5588 |
        | 2019 |        0.4765 |    0.5868 |   0.5984 |
        | 2020 |        0.5341 |    0.6804 |   0.7013 |
        | 2021 |        0.505  |    0.6794 |   0.7213 |
        | 2022 |        0.4841 |    0.644  |   0.7729 |
        | 2023 |        0.4585 |    0.6235 |   0.8408 |
        | 2024 |        0.4378 |    0.6223 |   0.9042 |
        | 2025 |        0.4439 |    0.635  | nan      |
        """

        government_debt_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govdebt_GDP", level), in_percent=True
        )

        return finalize_dataset(
            dataset=government_debt_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Debt to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_revenue(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Revenue for a variety of countries over time from the Global Macro Database (GMDB).
        Government Revenue is the total amount of money that a government collects from taxes and other sources.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: government income, public revenue.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Revenue

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2019-01-01', end_date="2025-12-31")

        economics.get_government_revenue(countries=['United Kingdom', 'Canada', 'Japan'])
        ```

        Which returns:

        |      |   United Kingdom |           Canada |       Japan |
        |:-----|-----------------:|-----------------:|------------:|
        | 2019 | 856511           | 938667           | 1.97133e+08 |
        | 2020 | 825025           | 919593           | 1.97357e+08 |
        | 2021 | 917948           |      1.07553e+06 | 2.07143e+08 |
        | 2022 |      1.04634e+06 |      1.17518e+06 | 2.16838e+08 |
        | 2023 |      1.10774e+06 |      1.23487e+06 | 2.24713e+08 |
        | 2024 |      1.1523e+06  |      1.31248e+06 | 2.33044e+08 |
        | 2025 |      1.23387e+06 |      1.36827e+06 | 2.44853e+08 |
        """

        government_revenue = self._get_gmdb_series(
            self._government_variable("govrev", level)
        )

        return finalize_dataset(
            dataset=government_revenue,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Revenue",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_revenue_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Revenue to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Revenue to GDP Ratio is the ratio of the total amount of money that a government collects from taxes
        and other sources to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.3104 for 31.04% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: revenue to GDP ratio.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Revenue to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_revenue_to_gdp_ratio(countries=['United States', 'Canada', 'Russian Federation'])
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |          0.3324 |   0.3996 |               0.3189 |
        | 2016 |          0.3275 |   0.403  |               0.3292 |
        | 2017 |          0.3343 |   0.4034 |               0.3336 |
        | 2018 |          0.3169 |   0.4102 |               0.3554 |
        | 2019 |          0.3167 |   0.4057 |               0.3568 |
        | 2020 |          0.3239 |   0.4141 |               0.3516 |
        | 2021 |          0.3351 |   0.4241 |               0.3572 |
        | 2022 |          0.348  |   0.4103 |               0.338  |
        | 2023 |          0.3146 |   0.4165 |               0.339  |
        | 2024 |          0.3174 |   0.4222 |               0.3506 |
        | 2025 |          0.3253 |   0.4219 |               0.3523 |
        """

        government_revenue_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govrev_GDP", level), in_percent=True
        )

        return finalize_dataset(
            dataset=government_revenue_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Revenue to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_tax_revenue(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Tax Revenue for a variety of countries over time from the Global Macro Database (GMDB).
        Government Tax Revenue is the total amount of money that a government collects from taxes.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: tax revenue, fiscal revenue.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Tax Revenue

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_tax_revenue(countries=['Kenya', 'Nigeria', 'South Africa'] )
        ```

        Which returns:

        |      |         Kenya |   South Africa |
        |:-----|--------------:|---------------:|
        | 2015 |   1.0216e+06  |    1.16848e+06 |
        | 2016 |   1.14692e+06 |    1.24935e+06 |
        | 2017 |   1.28657e+06 |    1.33015e+06 |
        | 2018 |   1.35099e+06 |    1.41653e+06 |
        | 2019 |   1.55782e+06 |    1.48503e+06 |
        | 2020 |   1.54288e+06 |    1.38495e+06 |
        | 2021 |   1.64048e+06 |    1.70199e+06 |
        | 2022 |   1.99797e+06 |    1.83491e+06 |
        | 2023 |   2.11082e+06 |    1.89676e+06 |
        | 2024 | nan           |    2.02766e+06 |
        | 2025 | nan           |    2.03308e+06 |
        """

        government_tax_revenue = self._get_gmdb_series(
            self._government_variable("govtax", level)
        )

        return finalize_dataset(
            dataset=government_tax_revenue,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Tax Revenue",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_tax_revenue_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Tax Revenue to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Tax Revenue to GDP Ratio is the ratio of the total amount of money that a government collects from taxes
        to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.1022 for 10.22% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: tax burden, tax to GDP ratio.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Tax Revenue to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_tax_revenue_to_gdp_ratio(
            countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |   Canada |   Mexico |
        |:-----|----------------:|---------:|---------:|
        | 2015 |          0.1994 |   0.2815 |   0.1323 |
        | 2016 |          0.1958 |   0.2855 |   0.1401 |
        | 2017 |          0.2031 |   0.2863 |   0.1358 |
        | 2018 |          0.1874 |   0.2903 |   0.136  |
        | 2019 |          0.1887 |   0.2857 |   0.1369 |
        | 2020 |          0.1928 |   0.2976 |   0.1484 |
        | 2021 |          0.2073 |   0.3021 |   0.1443 |
        | 2022 |          0.2203 |   0.2938 |   0.1449 |
        | 2023 |          0.1932 |   0.2958 |   0.1538 |
        | 2024 |          0.1942 |   0.2959 | nan      |
        | 2025 |          0.202  |   0.2925 | nan      |
        """

        government_tax_revenue_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govtax_GDP", level), in_percent=True
        )

        return finalize_dataset(
            dataset=government_tax_revenue_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Tax Revenue to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_expenditure(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Expenditure for a variety of countries over time from the Global Macro Database (GMDB).
        Government Expenditure is the total amount of money that a government spends on goods and services.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: government spending, public expenditure.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Expenditure

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_expenditure(countries=['Japan', 'China', 'India'])
        ```

        Which returns:

        |      |       Japan |       China |       India |
        |:-----|------------:|------------:|------------:|
        | 2015 | 2.06691e+08 | 2.1786e+07  | 3.65765e+07 |
        | 2016 | 2.08867e+08 | 2.41238e+07 | 4.12501e+07 |
        | 2017 | 2.09092e+08 | 2.71281e+07 | 4.4235e+07  |
        | 2018 | 2.10865e+08 | 3.05534e+07 | 4.91601e+07 |
        | 2019 | 2.14548e+08 | 3.38067e+07 | 5.35377e+07 |
        | 2020 | 2.47428e+08 | 3.6053e+07  | 6.12564e+07 |
        | 2021 | 2.43415e+08 | 3.74577e+07 | 7.03345e+07 |
        | 2022 | 2.41313e+08 | 4.02806e+07 | 7.82555e+07 |
        | 2023 | 2.39074e+08 | 4.23663e+07 | 8.60877e+07 |
        | 2024 | 2.43693e+08 | 4.44027e+07 | 9.31256e+07 |
        | 2025 | 2.51507e+08 | 4.44552e+07 | 9.84739e+07 |
        """

        government_expenditure = self._get_gmdb_series(
            self._government_variable("govexp", level)
        )

        return finalize_dataset(
            dataset=government_expenditure,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Expenditure",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_expenditure_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Expenditure to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Expenditure to GDP Ratio is the ratio of the total amount of money that a government spends on goods
        and services to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.3708 for 37.08% of GDP).

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: government spending to GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Expenditure to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_expenditure_to_gdp_ratio(
            countries=['United States', 'Japan', 'Netherlands'])
        ```

        Which returns:

        |      |   United States |   Japan |   Netherlands |
        |:-----|----------------:|--------:|--------------:|
        | 2015 |          0.3791 |  0.3766 |        0.4525 |
        | 2016 |          0.3818 |  0.3757 |        0.4391 |
        | 2017 |          0.3788 |  0.37   |        0.4275 |
        | 2018 |          0.3785 |  0.3705 |        0.4244 |
        | 2019 |          0.3836 |  0.3752 |        0.421  |
        | 2020 |          0.4736 |  0.4466 |        0.4781 |
        | 2021 |          0.4542 |  0.4244 |        0.4591 |
        | 2022 |          0.3849 |  0.4126 |        0.4328 |
        | 2023 |          0.3931 |  0.3881 |        0.4401 |
        | 2024 |          0.3966 |  0.3839 |        0.4429 |
        | 2025 |          0.3981 |  0.3791 |        0.4486 |
        """

        government_expenditure_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govexp_GDP", level), in_percent=True
        )

        return finalize_dataset(
            dataset=government_expenditure_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Expenditure to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_deficit(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Deficit for a variety of countries over time from the Global Macro Database (GMDB).
        Government Deficit is the total amount of money that a government spends more than it collects from taxes
        and other sources. A government deficit is usually financed by borrowing money.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The series is signed as a fiscal balance rather than as a deficit, in millions of
        national currency: a negative value is a deficit (spending exceeding revenue) and a
        positive value is a surplus.

        Also known as: budget deficit, fiscal deficit.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Deficit

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        economics.get_government_deficit(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |     United States |   Canada |            Mexico |
        |:-----|------------------:|---------:|------------------:|
        | 2015 | -853792           |    -1231 | -742271           |
        | 2016 |      -1.02094e+06 |    -9182 | -556067           |
        | 2017 | -872728           |    -2401 | -233109           |
        | 2018 |      -1.27096e+06 |     8046 | -517006           |
        | 2019 |      -1.44074e+06 |     -400 | -569358           |
        | 2020 |      -3.20097e+06 |  -243127 |      -1.03224e+06 |
        | 2021 |      -2.82551e+06 |   -77523 |      -1.00013e+06 |
        | 2022 | -961028           |    17672 |      -1.27619e+06 |
        | 2023 |      -2.18266e+06 |    -4808 |      -1.36965e+06 |
        | 2024 |      -2.31875e+06 |   -65919 |      -1.93368e+06 |
        | 2025 |      -2.23721e+06 |   -57263 |     nan           |
        """

        government_deficit = self._get_gmdb_series(
            self._government_variable("govdef", level)
        )

        return finalize_dataset(
            dataset=government_deficit,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Deficit",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_government_deficit_to_gdp_ratio(
        self,
        countries: list[str] | str | None = None,
        level: str = "consolidated",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Government Deficit to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Deficit to GDP Ratio is the ratio of the total amount of money that a government spends more than it
        collects from taxes and other sources to the Gross Domestic Product (GDP). A government deficit is usually financed
        by borrowing money.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The series is signed as a fiscal balance rather than as a deficit, and expressed as a
        decimal fraction of GDP: -0.07068 means a deficit of 7.068% of GDP, and a positive
        value is a surplus.

        Changed in v2.2.0: this used to be returned in percentage points. It is now a
        decimal fraction, matching every other ratio in the Finance Toolkit.

        Also known as: deficit-to-GDP, fiscal balance.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            level (str, optional): The government the figures cover: "consolidated" (the Global Macro Database's combination of general and central government figures, with the longest history), "general" (general government: central, state and local government and social security) or "central" (central government only). Defaults to "consolidated".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Deficit to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date="2025-12-31")

        government_deficit_to_gdp_ratio = economics.get_government_deficit_to_gdp_ratio()

        government_deficit_to_gdp_ratio.loc[:, ['New Zealand', 'Australia', 'United Kingdom']]
        ```

        Which returns:

        |      |   New Zealand |   Australia |   United Kingdom |
        |:-----|--------------:|------------:|-----------------:|
        | 2015 |        0.0012 |     -0.0104 |          -0.045  |
        | 2016 |        0.0115 |     -0.0155 |          -0.0327 |
        | 2017 |        0.0162 |     -0.0063 |          -0.0246 |
        | 2018 |        0.0083 |     -0.0056 |          -0.0219 |
        | 2019 |       -0.0028 |     -0.0121 |          -0.0246 |
        | 2020 |       -0.0758 |     -0.1142 |          -0.1291 |
        | 2021 |       -0.0388 |     -0.0417 |          -0.0774 |
        | 2022 |       -0.0292 |     -0.0193 |          -0.0473 |
        | 2023 |       -0.0348 |     -0.0135 |          -0.0604 |
        | 2024 |       -0.0299 |     -0.0277 |          -0.0613 |
        | 2025 |       -0.0333 |     -0.0393 |          -0.0546 |
        """

        government_deficit_to_gdp_ratio = self._get_gmdb_series(
            self._government_variable("govdef_GDP", level), in_percent=True
        )

        return finalize_dataset(
            dataset=government_deficit_to_gdp_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Government Deficit to GDP Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_trust_in_government(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Trust in government refers to the share of people who report having confidence
        in the national government. The data shown reflect the share of respondents
        answering “yes” (the other response categories being “no”, and “don’t know”)
        to the survey question: “In this country, do you have confidence in… national government?”

        Due to small sample sizes, country averages for horizontal inequalities (by age,
        gender and education) are pooled between 2010-18 to improve the accuracy of the
        estimates.

        The sample is ex ante designed to be nationally representative of the population
        aged 15 and over. The population-wide figure is returned -- both sexes, all education
        levels -- as an annual decimal fraction of that population (0.3933 for 39.33%). The
        breakdowns by sex and education level that the OECD publishes alongside it are not
        returned here.

        See definition: https://data.oecd.org/gga/trust-in-government.htm

        Also known as: political trust, institutional trust.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Trust in Government.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(end_date="2025-12-31")

        trust_in_government = economics.get_trust_in_government()

        trust_in_government.loc[:, ['United States', 'Greece', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Greece |   Japan |
        |:-----|----------------:|---------:|--------:|
        | 2006 |          0.4756 |   0.4345 |  0.2952 |
        | 2007 |          0.4756 |   0.4345 |  0.2952 |
        | 2008 |          0.4335 |   0.2764 |  0.2478 |
        | 2009 |          0.4335 |   0.2764 |  0.2478 |
        | 2010 |          0.4335 |   0.2764 |  0.2478 |
        | 2011 |          0.34   |   0.1483 |  0.2528 |
        | 2012 |          0.34   |   0.1483 |  0.2528 |
        | 2013 |          0.34   |   0.1483 |  0.2528 |
        | 2014 |          0.3309 |   0.2527 |  0.3649 |
        | 2015 |          0.3309 |   0.2527 |  0.3649 |
        | 2016 |          0.3309 |   0.2527 |  0.3649 |
        | 2017 |          0.3544 |   0.2311 |  0.4029 |
        | 2018 |          0.3544 |   0.2311 |  0.4029 |
        | 2019 |          0.3544 |   0.2311 |  0.4029 |
        | 2020 |          0.3933 |   0.3518 |  0.3819 |
        | 2021 |          0.3933 |   0.3518 |  0.3819 |
        | 2022 |          0.3933 |   0.3518 |  0.3819 |
        | 2023 |          0.2944 |   0.2732 |  0.3215 |
        | 2024 |          0.2944 |   0.2732 |  0.3215 |
        | 2025 |          0.2944 |   0.2732 |  0.3215 |
        """
        trust_in_government = oecd_model.get_trust_in_goverment(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=trust_in_government,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Trust in Government",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_consumer_price_index(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        oecd_source: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Consumer Price Index (CPI) is a measure that examines the average change in prices
        paid by consumers for goods and services over time. It is a measure of inflation.

        By default, data comes from the Global Macro Database (GMDB), which is annual-only
        (base year 2010). Set `oecd_source=True` to instead retrieve monthly or quarterly
        data from the OECD (base year varies per country), useful for tracking inflation
        more closely in real time.

        With period="monthly" and without oecd_source, every country comes from its most
        current source without an API key: Eurostat for the euro area and the European
        Economic Area (the HICP, 2015 = 100), the Office for National Statistics for the
        United Kingdom (CPI, 2015 = 100), the Statistics Bureau of Japan for Japan (CPI
        from 2015, on its latest base) and the Bank for International Settlements for
        every other country (2010 = 100). The base year differs between countries, so
        compare the changes rather than the levels.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: CPI, cost of living index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                The GMDB source is always annual, so without `oecd_source=True` only "monthly" changes
                the result, to the monthly sources described above. Defaults to None.
            oecd_source (bool, optional): Whether to get the data from the OECD instead of the
                Global Macro Database (GMDB). Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Consumer Price Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2008-09-01', end_date='2020-03-01')

        economics.get_consumer_price_index(countries=['Germany', 'France', 'Portugal'])
        ```

        Which returns:

        |      |   Germany |   France |   Portugal |
        |:-----|----------:|---------:|-----------:|
        | 2008 |   91.9238 |  93.2046 |    92.9047 |
        | 2009 |   92.2236 |  93.2862 |    92.1285 |
        | 2010 |   93.2552 |  94.7165 |    93.4207 |
        | 2011 |   95.1772 |  96.7132 |    96.8332 |
        | 2012 |   97.064  |  98.604  |    99.5189 |
        | 2013 |   98.5364 |  99.4547 |    99.7919 |
        | 2014 |   99.4445 |  99.9597 |    99.5142 |
        | 2015 |  100      | 100      |   100      |
        | 2016 |  100.467  | 100.184  |   100.608  |
        | 2017 |  101.975  | 101.216  |   101.984  |
        | 2018 |  103.765  | 103.09   |   102.998  |
        | 2019 |  105.237  | 104.231  |   103.346  |
        | 2020 |  105.801  | 104.729  |   103.333  |
        """
        check_period_type(period)

        if not oecd_source and period is not None and period.lower() == "monthly":
            consumer_price_index = self._get_monthly_price_data(
                "consumer_price_index", countries
            )
        elif oecd_source:
            period = (
                period
                if period is not None
                else "quarterly" if self._quarterly else "yearly"
            )
            consumer_price_index = oecd_model.get_consumer_price_index(
                period=period, start_date=self._start_date, end_date=self._end_date
            )
        else:

            consumer_price_index = gmdb_model.get_consumer_price_index(
                gmd_dataset=self._get_gmdb_dataset()
            )

        return finalize_dataset(
            dataset=consumer_price_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Consumer Price Index",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Rows empty for every selected country, e.g. a month only another source has.
            dropna=not oecd_source
            and period is not None
            and period.lower() == "monthly",
        )

    @handle_errors
    def get_inflation_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Inflation Rate is the percentage change in the Consumer Price Index (CPI) from one
        period to another. It is a measure of the rate of price increases in the economy.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The rate is expressed as a decimal fraction (0.0412 for 4.1166%) on an annual basis.
        The GMDB extends its series with IMF World Economic Outlook projections, so the
        current year and any later years the end_date reaches are forecasts rather than
        outturns.

        With period="monthly" the annual rate of change is returned for every month, the
        figure statistical offices publish, taken per country from the most current source
        without an API key: Eurostat for the euro area and the European Economic Area (the
        HICP, including the flash estimate at the end of the month itself), the Office for
        National Statistics for the United Kingdom (CPI), the Statistics Bureau of Japan
        for Japan (CPI, from 2016), the Bank for International Settlements for around sixty
        other countries, including the United States, China, India and Brazil, and the IMF
        for around a hundred more, all republishing the national consumer price indices.

        Changed in v2.2.0: this used to be returned in percentage points (4.1166 for
        4.1166%). It is now a decimal fraction, matching every other rate in the Finance
        Toolkit.

        Also known as: CPI-based inflation, price increases, consumer prices.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly or the annual data.
                Defaults to None, which is the annual data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Inflation Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2003-01-01', end_date='2009-03-01')

        economics.get_inflation_rate(countries=['Germany', 'France', 'Portugal'])
        ```

        Which returns:

        |      |   Germany |   France |   Portugal |
        |:-----|----------:|---------:|-----------:|
        | 2003 |    0.0107 |   0.0217 |     0.0324 |
        | 2004 |    0.0177 |   0.0234 |     0.0251 |
        | 2005 |    0.0195 |   0.0189 |     0.0213 |
        | 2006 |    0.0177 |   0.0189 |     0.0305 |
        | 2007 |    0.0229 |   0.016  |     0.0242 |
        | 2008 |    0.0275 |   0.0316 |     0.0265 |
        | 2009 |    0.0023 |   0.001  |    -0.009  |
        """
        check_period_type(period)

        period = validate_period(
            period or "yearly", ["monthly", "yearly"], "inflation rate"
        )

        if period == "monthly":
            inflation_rate = self._get_monthly_price_data("inflation_rate", countries)
        else:

            inflation_rate = gmdb_model.get_inflation_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )

        return finalize_dataset(
            dataset=inflation_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Inflation Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Rows empty for every selected country, e.g. a month only another source has.
            dropna=period == "monthly",
        )

    @handle_errors
    def get_producer_price_index(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Producer Price Index (PPI) for a variety of countries over time from the OECD.
        The PPI measures the average change over time in the prices received by domestic
        producers (manufacturing) for their output. Because producers tend to pass rising input
        costs on to their customers with a lag, the PPI is generally seen as a leading, upstream
        indicator of cost pressure that later shows up in the Consumer Price Index (CPI).

        The index covers manufacturing output only, is not seasonally adjusted, and is set to
        100 in the base year, which can vary per country.

        The OECD stopped updating this series in its Key Economic Indicators dataset during
        2023: annual values end in 2022, and monthly and quarterly values end in early 2023
        for all but a couple of countries. A start_date after that point returns an empty
        DataFrame.

        See definition: https://www.oecd.org/en/data/indicators/producer-prices-ppi.html

        Also known as: PPI, wholesale prices, factory gate prices, upstream inflation.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Producer Price Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date='2022-01-01')

        economics.get_producer_price_index(
            countries=['United States', 'Germany'],
            period='yearly'
        )
        ```

        Which returns:

        |      |   United States |   Germany |
        |:-----|----------------:|----------:|
        | 2018 |         106.059 |   102.758 |
        | 2019 |         106.067 |   103.65  |
        | 2020 |         103.849 |   103.15  |
        | 2021 |         116.511 |   108.242 |
        | 2022 |         134.46  |   122.75  |
        """
        check_period_type(period)

        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )

        producer_price_index = oecd_model.get_producer_price_index(
            period=period, start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=producer_price_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Producer Price Index",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_consumer_confidence_index(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        This consumer confidence indicator provides an indication of future developments of
        households consumption and saving, based upon answers regarding their expected
        financial situation, their sentiment about the general economic situation,
        unemployment and capability of savings.

        An indicator above 100 signals a boost in the consumers’ confidence towards
        the future economic situation, as a consequence of which they are less prone
        to save, and more inclined to spend money on major purchases in the next
        12 months. Values below 100 indicate a pessimistic attitude towards
        future developments in the economy, possibly resulting in a tendency to
        save more and consume less.

        See definition: https://data.oecd.org/leadind/consumer-confidence-index-cci.htm

        Also known as: consumer sentiment, spending outlook.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Consumer Confidence Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2008-09-01', end_date='2009-03-01')

        economics.get_consumer_confidence_index(countries=['Germany', 'France', 'Portugal'])
        ```

        Which returns:

        |         |   Germany |   France |   Portugal |
        |:--------|----------:|---------:|-----------:|
        | 2008-09 |   98.3529 |  97.6684 |    98.0214 |
        | 2008-10 |   98.228  |  97.6761 |    97.8728 |
        | 2008-11 |   98.1059 |  97.7517 |    97.4371 |
        | 2008-12 |   97.9237 |  97.7033 |    96.956  |
        | 2009-01 |   97.8036 |  97.6175 |    96.6248 |
        | 2009-02 |   97.7333 |  97.5525 |    96.6024 |
        | 2009-03 |   97.7251 |  97.6748 |    96.9086 |
        """
        consumer_confidence_index = oecd_model.get_consumer_confidence_index(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=consumer_confidence_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Consumer Confidence Index",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_business_confidence_index(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        This business confidence indicator provides information on future developments,
        based upon opinion surveys on developments in production, orders and stocks of
        finished goods in the industry sector. It can be used to monitor output growth
        and to anticipate turning points in economic activity.

        Numbers above 100 suggest an increased confidence in near future business
        performance, and numbers below 100 indicate pessimism towards future performance.

        See definition: https://data.oecd.org/leadind/business-confidence-index-bci.htm

        Also known as: BCI, business sentiment.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Business Confidence Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2022-09-01', end_date='2023-03-01')

        economics.get_business_confidence_index(countries=['Brazil', 'Canada', 'Costa Rica'])
        ```

        Which returns:

        |         |   Brazil |   Canada |   Costa Rica |
        |:--------|---------:|---------:|-------------:|
        | 2022-09 | 100.2    | 100.557  |      100.704 |
        | 2022-10 |  99.8479 | 100.162  |      100.693 |
        | 2022-11 |  99.5281 |  99.8368 |      100.689 |
        | 2022-12 |  99.3767 |  99.541  |      100.708 |
        | 2023-01 |  99.3373 |  99.2389 |      100.763 |
        | 2023-02 |  99.3642 |  98.909  |      100.879 |
        | 2023-03 |  99.4444 |  98.5588 |      101.063 |
        """
        business_confidence_index = oecd_model.get_business_confidence_index(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=business_confidence_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Business Confidence Index",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_composite_leading_indicator(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The composite leading indicator (CLI) is designed to provide early signals
        of turning points in business cycles showing fluctuation of the economic
        activity around its long term potential level. CLIs show short-term economic
        movements in qualitative rather than quantitative terms.

        The series returned is the OECD-harmonised, amplitude-adjusted index at monthly
        frequency, oscillating around a long-run average of 100: readings above 100 point to
        above-trend activity ahead and readings below 100 to below-trend activity. Coverage
        is narrower than the confidence indices -- around 22 countries and aggregates.

        See definition: https://data.oecd.org/leadind/composite-leading-indicator-cli.htm

        Also known as: CLI, leading economic indicator.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Composite Leading Indicator.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-06-01', end_date='2023-12-01')

        economics.get_composite_leading_indicator(countries=['United States', 'United Kingdom', 'Japan'])
        ```

        Which returns:

        |         |   United States |   United Kingdom |   Japan |
        |:--------|----------------:|-----------------:|--------:|
        | 2023-06 |         98.6692 |          98.2998 | 99.8519 |
        | 2023-07 |         98.7456 |          98.5586 | 99.8473 |
        | 2023-08 |         98.8022 |          98.7914 | 99.8444 |
        | 2023-09 |         98.8351 |          99.0011 | 99.8342 |
        | 2023-10 |         98.8582 |          99.1736 | 99.8154 |
        | 2023-11 |         98.9029 |          99.3102 | 99.7915 |
        | 2023-12 |         98.9852 |          99.4431 | 99.7713 |
        """
        composite_leading_indicator = oecd_model.get_composite_leading_indicator(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=composite_leading_indicator,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Composite Leading Indicator",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_house_prices(
        self,
        countries: list[str] | str | None = None,
        quarterly: bool | None = None,
        inflation_adjusted: bool = False,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        In most cases, the nominal house price index covers the sales of newly-built
        and existing dwellings, following the recommendations from the RPPI (Residential
        Property Prices Indices) manual.

        The real house price index is given by the ratio of the nominal house price index
        to the consumers' expenditure deflator in each country from the OECD national
        accounts database. Both indices are seasonally adjusted.

        Both are an index based on 2015 = 100.

        See definition: https://data.oecd.org/price/housing-prices.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Also known as: real estate prices, property prices, housing index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            quarterly (bool | None, optional): Whether to return the quarterly data or the annual data.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data or the nominal data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the House Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-12-31')

        economics.get_house_prices(
            countries=['Japan', 'Netherlands', 'Ireland'],
            quarterly=False,
            inflation_adjusted=True
        )
        ```

        Which returns:

        |      |   Japan |   Netherlands |   Ireland |
        |:-----|--------:|--------------:|----------:|
        | 2015 | 100     |       100     |   100     |
        | 2016 | 102.645 |       106.044 |   108.503 |
        | 2017 | 104.009 |       113.945 |   120.785 |
        | 2018 | 105.604 |       122.047 |   128.926 |
        | 2019 | 106.087 |       126.665 |   128.744 |
        | 2020 | 107.773 |       135.898 |   130.088 |
        | 2021 | 115.407 |       156.892 |   144.681 |
        | 2022 | 121.069 |       150.691 |   145.71  |
        | 2023 | 119.632 |       145.341 |   141.362 |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            # The real index is the nominal one deflated by consumer prices.
            house_prices = (
                self._get_gmdb_series("rHPI")
                if inflation_adjusted
                else gmdb_model.get_house_price_index(
                    gmd_dataset=self._get_gmdb_dataset()
                )
            )
        else:
            house_prices = oecd_model.get_house_prices(
                quarterly=quarterly,
                inflation_adjusted=inflation_adjusted,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        return finalize_dataset(
            dataset=house_prices,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="House Prices",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_rent_prices(
        self,
        countries: list[str] | str | None = None,
        quarterly: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The housing rent price index measures the prices paid for renting
        residential properties over time. Together with the house price index
        it is a key input into affordability and house ownership profitability
        measures such as the price to rent ratio.

        This is an index based on 2015 = 100.

        See definition: https://data.oecd.org/price/housing-prices.htm

        Also known as: rental prices, housing costs, rent index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            quarterly (bool | None, optional): Whether to return the quarterly data or the annual data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Rent Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-12-31')

        economics.get_rent_prices(
            countries=['Turkey', 'United States', 'United Kingdom'],
            quarterly=False)
        ```

        Which returns:

        |      |   Turkey |   United States |   United Kingdom |
        |:-----|---------:|----------------:|-----------------:|
        | 2015 |  100     |         100     |          100     |
        | 2016 |  108.66  |         103.773 |          101.725 |
        | 2017 |  118.577 |         107.731 |          102.699 |
        | 2018 |  130.041 |         111.627 |          103.174 |
        | 2019 |  143.185 |         115.765 |          103.924 |
        | 2020 |  156.574 |         119.382 |          105.399 |
        | 2021 |  172.625 |         122.062 |          107.148 |
        | 2022 |  221.217 |         129.426 |          110.897 |
        | 2023 |  405.025 |         139.72  |          117.271 |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly

        rent_prices = oecd_model.get_rent_prices(
            quarterly=quarterly, start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=rent_prices,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Rent Prices",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_household_savings_rate(
        self,
        countries: list[str] | str | None = None,
        quarterly: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Gross Household Savings Rate for a variety of countries over time from the
        OECD's Household Dashboard. The household savings rate is the share of household
        gross disposable income (adjusted for the net change in pension entitlements) that
        is saved rather than spent on final consumption.

        It is a key input to consumption-smoothing and life-cycle/permanent-income theories
        of household behaviour, and a closely watched signal of both near-term consumption
        momentum (a falling savings rate can temporarily prop up spending even as income
        growth slows) and a household sector's buffer against future income shocks. It
        complements Total Consumption (see `get_total_consumption`) — the two together
        show how much of household income is spent versus set aside.

        The rate is seasonally adjusted and returned as a decimal fraction of adjusted gross
        disposable income (0.1177 for 11.77%). Because the denominator is adjusted for the
        net change in pension entitlements and the numerator is gross rather than net saving,
        this sits above the personal saving rate the BEA publishes for the United States.

        See definition: https://data-explorer.oecd.org/vis?df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_HHDASH%40DF_HHDASH_INDIC

        Also known as: household savings ratio, personal savings rate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            quarterly (bool | None, optional): Whether to return the quarterly data or the annual data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Household Savings Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2022-12-31')

        economics.get_household_savings_rate(
            countries=['United States', 'Germany'],
            quarterly=False)
        ```

        Which returns:

        |      |   United States |   Germany |
        |:-----|----------------:|----------:|
        | 2015 |          0.1152 |    0.1724 |
        | 2016 |          0.1112 |    0.1715 |
        | 2017 |          0.1153 |    0.1757 |
        | 2018 |          0.1222 |    0.182  |
        | 2019 |          0.1305 |    0.1793 |
        | 2020 |          0.2063 |    0.2324 |
        | 2021 |          0.1707 |    0.2199 |
        | 2022 |          0.0981 |    0.189  |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly

        household_savings_rate = oecd_model.get_household_savings_rate(
            quarterly=quarterly,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        return finalize_dataset(
            dataset=household_savings_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Household Savings Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_household_debt_to_income_ratio(
        self,
        countries: list[str] | str | None = None,
        quarterly: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Household Debt to Disposable Income Ratio for a variety of countries over
        time from the OECD's Household Dashboard. This expresses total household gross
        debt (loans and debt securities) as a share of household gross disposable
        income, returned as a decimal ratio (1.0014 means debt equals 100.14% of income).

        It is a standard household-leverage indicator used in financial-stability analysis:
        a high or rapidly rising ratio signals households are more exposed to income shocks
        or interest rate increases (debt-servicing costs rise directly with rates on
        variable-rate or refinanced debt), and has historically preceded credit-cycle
        downturns (e.g. in the lead-up to the 2008 financial crisis). It is the household-
        sector analogue to government debt (see `get_government_debt_to_gdp_ratio`) — the
        two together give a fuller picture of an economy's overall leverage.

        See definition: https://data-explorer.oecd.org/vis?df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_HHDASH%40DF_HHDASH_INDIC

        Also known as: household leverage ratio, debt-to-income ratio.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            quarterly (bool | None, optional): Whether to return the quarterly data or the annual data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Household Debt to Income Ratio.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2022-12-31')

        economics.get_household_debt_to_income_ratio(
            countries=['United States', 'Australia'],
            quarterly=False)
        ```

        Which returns:

        |      |   United States |   Australia |
        |:-----|----------------:|------------:|
        | 2015 |          1.0396 |      1.86   |
        | 2016 |          1.0384 |      1.9308 |
        | 2017 |          1.0265 |      1.9843 |
        | 2018 |          1.0014 |      1.9888 |
        | 2019 |          0.9955 |      1.9676 |
        | 2020 |          0.9533 |      1.8811 |
        | 2021 |          0.9625 |      1.9168 |
        | 2022 |          1.0163 |      1.9253 |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly

        household_debt_to_income_ratio = oecd_model.get_household_debt_to_income_ratio(
            quarterly=quarterly,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        return finalize_dataset(
            dataset=household_debt_to_income_ratio,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Household Debt to Income Ratio",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_share_prices(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Share price indices are calculated from the prices of common shares of companies
        traded on national or foreign stock exchanges. They are usually determined by the
        stock exchange, using the closing daily values for the monthly data, and normally
        expressed as simple arithmetic averages of the daily data.

        A share price index measures how the value of the stocks in the index is changing,
        a share return index tells the investor what their “return” is, meaning how much
        money they would make as a result of investing in that basket of shares.

        A price index measures changes in the market capitalisation of the basket of shares
        in the index whereas a return index adds on to the price index the value of
        dividend payments, assuming they are re-invested in the same stocks.
        Occasionally agencies such as central banks will compile share indices.

        This uses 2015 as the base year (= 100)

        See definition: https://data.oecd.org/price/share-prices.htm

        Also known as: stock market index, equity index, market performance.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Share Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2013-01-01", end_date="2025-12-31")

        economics.get_share_prices(countries=['Turkey', 'Belgium', 'Australia'])
        ```

        Which returns:

        |      |    Turkey |   Belgium |   Australia |
        |:-----|----------:|----------:|------------:|
        | 2013 |   96.6029 |   74.3936 |     92.3054 |
        | 2014 |   93.2354 |   87.8382 |     98.611  |
        | 2015 |  100      |  100      |    100      |
        | 2016 |   95.6644 |   95.2324 |     96.0699 |
        | 2017 |  122.746  |  101.514  |    105.648  |
        | 2018 |  126.263  |   96.5515 |    109.205  |
        | 2019 |  123.056  |   92.6847 |    117.326  |
        | 2020 |  140.511  |   77.8758 |    111.188  |
        | 2021 |  187.146  |   91.6789 |    130.475  |
        | 2022 |  369.298  |   93.0484 |    128.367  |
        | 2023 |  785.903  |   97.9468 |    131.286  |
        | 2024 | 1190.71   |  106.289  |    143.996  |
        | 2025 | 1272.13   |  115.355  |    154.686  |
        """
        check_period_type(period)

        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )

        share_prices = oecd_model.get_share_prices(
            period=period, start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=share_prices,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Share Prices",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_exchange_rates(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        end_of_period: bool = False,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Exchange rates are defined as the price of one country's currency in relation
        to another country's currency. This indicator is measured in terms of
        national currency per US dollar.

        See definition: https://data.oecd.org/conversion/exchange-rates.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources are quoted the same way (national currency per US dollar), but the OECD
        source omits the United States itself, since the rate is trivially 1, whereas the GMDB
        source carries it as 1.0. Only the OECD source supports monthly and quarterly
        frequency; the GMDB is annual only, so the period argument has no effect when
        gmdb_source is True.

        Both are averages over each period. For daily and weekly rates (period="daily" or
        "weekly"), or the rate at the end of each month, quarter or year (end_of_period=True),
        the rates come from the Bank for International Settlements instead, without an API
        key: daily for around 80 currencies with history back to 1945 for some, and at the
        end of each month for around 130, back to the 1950s for most. Weekly and quarterly or
        yearly end-of-period rates take the last observation of each period.

        Also known as: currency exchange, FX rates, foreign exchange rates.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the daily, weekly, monthly, quarterly
                or the annual data. Defaults to None, which is quarterly when the Economics class
                was initialized with quarterly=True and annual otherwise.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            end_of_period (bool, optional): Whether to return the rate at the end of each month,
                quarter or year instead of its average. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exchange Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2000-01-01', end_date='2010-12-31')

        economics.get_exchange_rates(countries=['Japan', 'Indonesia', "China"])
        ```

        Which returns:

        |      |    Japan |   Indonesia |   China |
        |:-----|---------:|------------:|--------:|
        | 2000 | 107.746  |     8401.53 |  8.2771 |
        | 2001 | 121.409  |    10252.9  |  8.277  |
        | 2002 | 125.138  |     9315.13 |  8.277  |
        | 2003 | 115.929  |     8570.07 |  8.277  |
        | 2004 | 108.158  |     8945.34 |  8.2777 |
        | 2005 | 110.181  |     9714.85 |  8.1924 |
        | 2006 | 116.307  |     9170.49 |  7.9737 |
        | 2007 | 117.774  |     9138.88 |  7.6063 |
        | 2008 | 103.444  |     9682.61 |  6.9485 |
        | 2009 |  93.5426 |    10397.6  |  6.8311 |
        | 2010 |  87.7075 |     9082.76 |  6.7678 |
        """
        check_period_type(period)

        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source
        period = "yearly" if period.lower() == "annual" else period.lower()

        if period in ("daily", "weekly"):
            exchange_rates = resample_to_period(
                bis_model.get_exchange_rates(
                    "daily",
                    buffered_start_date(self._start_date, period),
                    self._end_date,
                ),
                period,
            )
        elif end_of_period:
            monthly_rates = bis_model.get_exchange_rates(
                "monthly",
                buffered_start_date(self._start_date, "monthly"),
                self._end_date,
            )
            exchange_rates = (
                monthly_rates
                if period == "monthly"
                else monthly_rates.groupby(
                    monthly_rates.index.asfreq("Q" if period == "quarterly" else "Y")
                ).last()
            )
        elif gmdb_source:

            exchange_rates = gmdb_model.get_usd_exchange_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            exchange_rates = oecd_model.get_exchange_rates(
                period=period, start_date=self._start_date, end_date=self._end_date
            )

        return finalize_dataset(
            dataset=exchange_rates,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Exchange Rates",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # The BIS lists weekends and holidays without a rate.
            dropna=period in ("daily", "weekly") or end_of_period,
        )

    @handle_errors
    def get_real_effective_exchange_rate(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Real Effective Exchange Rate (REER) for a variety of countries over time from the
        Global Macro Database (GMDB). The REER is a trade-weighted average of a country's currency
        relative to a basket of other major currencies, adjusted for relative price levels
        (inflation) between the country and its trading partners.

        Unlike a simple bilateral exchange rate, the REER captures a currency's overall
        competitiveness: a rising REER indicates that a country's exports are becoming more
        expensive (and imports cheaper) relative to its trading partners after accounting for
        inflation differentials, while a falling REER indicates the opposite. The index is set to
        100 in the base year, which can vary per country.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: REER, trade-weighted exchange rate, currency competitiveness index.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Real Effective Exchange Rate

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date="2025-12-31")

        economics.get_real_effective_exchange_rate(countries=['United States', 'Japan', 'Netherlands'])
        ```

        Which returns:

        |      |   United States |    Japan |   Netherlands |
        |:-----|----------------:|---------:|--------------:|
        | 2018 |         103.02  | 107.295  |       104.015 |
        | 2019 |         105.92  | 110.309  |       103.969 |
        | 2020 |         107.337 | 111.285  |       106.116 |
        | 2021 |         105.222 | 101.827  |       106.389 |
        | 2022 |         115.234 |  87.8727 |       106.548 |
        | 2023 |         116.036 |  83.6944 |       107.665 |
        | 2024 |         118.848 |  79.2776 |       109.002 |
        | 2025 |         119.249 |  80.8126 |       111.404 |
        """

        real_effective_exchange_rate = gmdb_model.get_real_effective_exchange_rate(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=real_effective_exchange_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Real Effective Exchange Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_money_supply(
        self,
        countries: list[str] | str | None = None,
        measure: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Money Supply is the total amount of money that is in circulation in a country.
        It includes currency, demand deposits, and other liquid assets that can be easily
        converted into cash. Money supply is an important economic indicator that the
        Federal Reserve uses to implement its monetary policy.

        Money supply can be divided into five categories: M0, M1, M2, M3 and M4.
            - M0: The total of all physical currency, plus accounts at the central bank that can be exchanged for physical currency.
            - M1: The total of all physical currency part of bank reserves + the amount in demand accounts ("checking" or "current" accounts).
            - M2: M1 + most savings accounts, money market accounts, retail money market mutual funds, and small denomination time deposits.
            - M3: M2 + large time deposits, institutional money market funds, short-term repurchase agreements, and other larger liquid assets.
            - M4: M3 + all other financial assets.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The aggregates are annual and expressed in millions of national currency, so levels
        are not comparable across countries with different currencies, but growth rates are.
        Not every country publishes every aggregate; the ones it does not are NaN.

        Also known as: M1, M2, M3, monetary aggregate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            measure (str | None, optional): Which single aggregate to return, one of 'M0', 'M1',
                'M2', 'M3' or 'M4'. Defaults to None, which returns all five with the aggregate
                as the first level of the column index.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Money Supply

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date='2020-12-31')

        money_supply = economics.get_money_supply(
            countries=['Netherlands', 'Germany', 'United States'],
            measure='M2'
        )
        ```

        Which returns:

        |      |   Netherlands |    Germany |   United States |
        |:-----|--------------:|-----------:|----------------:|
        | 2010 |        701718 | 1.9878e+06 |     8.4731e+06  |
        | 2011 |        727265 | 2.1053e+06 |     8.8445e+06  |
        | 2012 |        746482 | 2.2556e+06 |     9.7533e+06  |
        | 2013 |        741372 | 2.3144e+06 |     1.05023e+07 |
        | 2014 |        743043 | 2.4272e+06 |     1.11275e+07 |
        | 2015 |        822382 | 2.6518e+06 |     1.18062e+07 |
        | 2016 |        841302 | 2.8022e+06 |     1.25222e+07 |
        | 2017 |        851237 | 2.9236e+06 |     1.33189e+07 |
        | 2018 |        846513 | 3.0562e+06 |     1.39024e+07 |
        | 2019 |        889033 | 3.1968e+06 |     1.44602e+07 |
        | 2020 |        974276 | 3.4582e+06 |     1.5425e+07  |
        """

        money_supply = gmdb_model.get_money_supply(gmd_dataset=self._get_gmdb_dataset())

        money_supply = finalize_dataset(
            dataset=money_supply,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            rolling=rolling,
            trailing=trailing,
        )

        if measure:
            if measure not in money_supply.columns.get_level_values(0):
                logger.warning(
                    f"The following measure is not available for Money Supply: {measure}"
                )
            else:
                money_supply = money_supply[measure]

        if countries:
            if isinstance(countries, str):
                countries = [countries]

            if isinstance(money_supply.columns, pd.MultiIndex):
                available_countries = money_supply.columns.get_level_values(1).unique()
                missing_countries = [
                    country
                    for country in countries
                    if country not in available_countries
                ]
                money_supply = money_supply.loc[
                    :, money_supply.columns.get_level_values(1).isin(countries)
                ]
            else:
                missing_countries = [
                    country
                    for country in countries
                    if country not in money_supply.columns
                ]
                # A set would make the column order depend on hash randomisation.
                money_supply = money_supply[
                    [
                        country
                        for country in countries
                        if country not in missing_countries
                    ]
                ]

            if missing_countries:
                logger.warning(
                    f"The following countries are not available for Money Supply: {missing_countries}"
                )

        return money_supply

    @handle_errors
    def get_central_bank_policy_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The Central Bank Policy Rate is the interest rate that a central bank sets on its
        loans and advances to a commercial bank. This interest rate is used by the monetary
        authorities to control inflation and stabilize the country's currency.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The rate is annual and expressed as a decimal fraction per annum (0.0538 for 5.375%),
        taken at the end of the year rather than averaged over it.

        With period="daily", "weekly" or "monthly" the rates come from the Bank for
        International Settlements (BIS) instead, without an API key. It publishes the
        policy rates of around fifty central banks daily, including the Federal Reserve,
        the European Central Bank (as "Euro Area"), the Bank of England and the Bank of
        Japan, some back to the 1940s. Weekly and monthly take the rate on the last day of
        each period, so the current week or month shows the rate so far.

        Changed in v2.2.0: this used to be returned in percentage points (5.375 for 5.375%).
        It is now a decimal fraction, matching every other rate in the Finance Toolkit.

        Also known as: policy rate, benchmark rate, base rate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the daily, weekly, monthly or the annual
                data. Defaults to None, which is the annual data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Central Bank Policy Rate

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-01-01', end_date='2025-12-31')

        economics.get_central_bank_policy_rate(countries=['Netherlands', 'Germany', 'United States'])
        ```

        Which returns:

        |      |   Netherlands |   Germany |   United States |
        |:-----|--------------:|----------:|----------------:|
        | 2021 |       -0.005  |   -0.005  |          0.0012 |
        | 2022 |        0.0044 |    0.0044 |          0.0438 |
        | 2023 |        0.0362 |    0.0362 |          0.0538 |
        | 2024 |        0.0356 |    0.0356 |          0.0438 |
        | 2025 |        0.0213 |    0.0213 |          0.0362 |
        """
        check_period_type(period)

        period = validate_period(
            period or "yearly",
            ["daily", "weekly", "monthly", "yearly"],
            "central bank policy rate",
        )

        if period == "yearly":

            central_bank_policy_rate = gmdb_model.get_central_bank_policy_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            # Weeks and months are taken from the daily rates, which the BIS updates
            # sooner than its own end-of-month series.
            central_bank_policy_rate = resample_to_period(
                bis_model.get_central_bank_policy_rate(
                    buffered_start_date(self._start_date, period), self._end_date
                ),
                period,
            )

        return finalize_dataset(
            dataset=central_bank_policy_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Central Bank Policy Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Rows empty for every selected country, e.g. a month only another source has.
            dropna=period != "yearly",
        )

    @handle_errors
    def get_short_term_interest_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Short-term interest rates are the rates at which short-term borrowings are
        effected between financial institutions or the rate at which short-term government
        paper is issued or traded in the market. Short-term interest rates are generally
        averages of daily rates, measured as a percentage.

        Short-term interest rates are based on three-month money market rates where available.
        Typical standardised names are "money market rate" and "treasury bill rate". The OECD
        source specifically returns the 3-month interbank offered rate.

        See definition: https://data.oecd.org/interest/short-term-interest-rates.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources return the rate as a decimal fraction per annum (0.0513 for 5.13%), so
        the two are directly interchangeable. Only the OECD source supports monthly and
        quarterly frequency; the GMDB is annual only, so the period argument has no effect
        when gmdb_source is True.

        Changed in v2.2.0: the GMDB source previously returned percentage points (5.13 for
        5.13%) while the OECD source returned a decimal fraction. The GMDB series is now
        divided by 100 so both sources agree; divide any hard-coded comparison by 100.

        Also known as: 3-month rate, money market rate, short-term yield.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Short Term Interest Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-05-01', end_date="2025-12-31")

        economics.get_short_term_interest_rate(
            countries=['Japan', 'United States', 'China'],
            gmdb_source=False,
            period='quarterly'
        )
        ```

        Which returns:

        |        |   Japan |   United States |   China |
        |:-------|--------:|----------------:|--------:|
        | 2023Q2 | -0.0001 |          0.0513 |  0.0289 |
        | 2023Q3 |  0.0001 |          0.0543 |  0.0261 |
        | 2023Q4 |  0.0002 |          0.054  |  0.0288 |
        | 2024Q1 |  0.0005 |          0.0526 |  0.0267 |
        | 2024Q2 |  0.0013 |          0.0531 |  0.0235 |
        | 2024Q3 |  0.0023 |          0.051  |  0.021  |
        | 2024Q4 |  0.0033 |          0.0454 |  0.0205 |
        | 2025Q1 |  0.0079 |          0.0432 |  0.0202 |
        | 2025Q2 |  0.0078 |          0.0431 |  0.0188 |
        | 2025Q3 |  0.0079 |          0.042  |  0.0171 |
        | 2025Q4 |  0.009  |          0.0386 |  0.0168 |
        """
        check_period_type(period)

        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )

        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:

            short_term_interest_rate = gmdb_model.get_short_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            short_term_interest_rate = oecd_model.get_short_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        return finalize_dataset(
            dataset=short_term_interest_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Short Term Interest Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_long_term_interest_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Long-term interest rates refer to government bonds maturing in ten years.
        Rates are mainly determined by the price charged by the lender, the risk
        from the borrower and the fall in the capital value. Long-term interest rates
        are generally averages of daily rates, measured as a percentage. These interest
        rates are implied by the prices at which the government bonds are traded on
        financial markets, not the interest rates at which the loans were issued.

        In all cases, they refer to bonds whose capital repayment is guaranteed by governments.
        Long-term interest rates are one of the determinants of business investment. Low long
        term interest rates encourage investment in new equipment and high interest rates
        discourage it. Investment is, in turn, a major source of economic growth

        See definition: https://data.oecd.org/interest/long-term-interest-rates.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources return the rate as a decimal fraction per annum (0.0357 for 3.57%), so
        the two are directly interchangeable. Only the OECD source supports monthly and
        quarterly frequency; the GMDB is annual only, so the period argument has no effect
        when gmdb_source is True.

        With period="daily" or "weekly" the 10-year yields come from the issuers
        themselves, without an API key: the U.S. Department of the Treasury (par yield,
        from 1990), the Bank of England (nominal par yield of gilts, from 1993), the
        Japanese Ministry of Finance (JGBs, from 1974) and the European Central Bank (the
        euro area yield curve of all central government bonds, from 2004). Weekly takes the
        yield on the last trading day of each week. Other countries have no daily source
        and are not included.

        Changed in v2.2.0: the GMDB source previously returned percentage points (3.57 for
        3.57%) while the OECD source returned a decimal fraction. The GMDB series is now
        divided by 100 so both sources agree; divide any hard-coded comparison by 100.

        Also known as: 10-year yield, government bond rate, long-term yield.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the daily, weekly, monthly, quarterly
                or the annual data. Defaults to None, which is quarterly when the Economics class
                was initialized with quarterly=True and annual otherwise.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
                Not used for daily and weekly data.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Long Term Interest Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-05-01', end_date='2023-12-31')

        economics.get_long_term_interest_rate(
            countries=['Japan', 'United States', 'Brazil'],
            gmdb_source=False,
            period='monthly'
        )
        ```

        Which returns:

        |         |   Japan |   United States |   Brazil |
        |:--------|--------:|----------------:|---------:|
        | 2023-05 |  0.0043 |          0.0357 |   0.0728 |
        | 2023-06 |  0.004  |          0.0375 |   0.0728 |
        | 2023-07 |  0.0059 |          0.039  |   0.07   |
        | 2023-08 |  0.0064 |          0.0417 |   0.07   |
        | 2023-09 |  0.0076 |          0.0438 |   0.07   |
        | 2023-10 |  0.0095 |          0.048  |   0.0655 |
        | 2023-11 |  0.0066 |          0.045  |   0.0655 |
        | 2023-12 |  0.0062 |          0.0402 |   0.0655 |
        """
        check_period_type(period)

        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )

        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if period.lower() in ("daily", "weekly"):
            long_term_interest_rate = resample_to_period(
                self._get_daily_long_term_interest_rate(countries), period.lower()
            )
        elif gmdb_source:

            long_term_interest_rate = gmdb_model.get_long_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            long_term_interest_rate = oecd_model.get_long_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        return finalize_dataset(
            dataset=long_term_interest_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Long Term Interest Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Rows empty for every selected country, e.g. a month only another source has.
            dropna=period.lower() in ("daily", "weekly"),
        )

    @handle_errors
    def get_overnight_rate(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The overnight rate is the interest rate at which banks borrow from each other for
        a single day. It is the rate central banks steer with their policy rate, and the
        benchmark that replaced LIBOR for loans, derivatives and floating rate notes in
        each of these currencies, which makes it the most direct daily read of monetary
        conditions.

        Each currency has its own benchmark, published by its central bank:

        - Euro Area: the euro short-term rate (€STR) from the European Central Bank, since
          October 2019.
        - United Kingdom: the Sterling Overnight Index Average (SONIA) from the Bank of
          England, since 1997.
        - Japan: the uncollateralized overnight call rate, the Tokyo Overnight Average
          Rate (TONA), from the Bank of Japan, since 1985.
        - United States: the Secured Overnight Financing Rate (SOFR) from the Federal
          Reserve Bank of New York, since April 2018.

        None of these sources require an API key. The rates are returned as decimal
        fractions per annum (0.0244 for 2.44%). Weekly and monthly periods take the rate
        on the last day of each period (weeks end on Friday).

        Also known as: €STR, ESTR, SONIA, TONA, SOFR, overnight interbank rate, risk-free
        rate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str, optional): Whether to return the daily, weekly or monthly data. Defaults to "daily".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the overnight rate per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        economics.get_overnight_rate(period='monthly')
        ```

        Which returns:

        |         |   Euro Area |   Japan |   United Kingdom |   United States |
        |:--------|------------:|--------:|-----------------:|----------------:|
        | 2026-04 |      0.0193 |  0.0073 |           0.0373 |          0.0366 |
        | 2026-05 |      0.0193 |  0.0073 |           0.0373 |          0.0363 |
        | 2026-06 |      0.0218 |  0.0098 |           0.0373 |          0.0368 |
        | 2026-07 |      0.0218 |  0.0098 |           0.0373 |          0.0366 |
        | 2026-08 |      0.0218 |  0.0098 |           0.0373 |          0.0368 |
        | 2026-09 |      0.0244 |  0.0123 |           0.0373 |          0.039  |
        """
        check_period_type(period)

        period = validate_period(
            period, ["daily", "weekly", "monthly"], "overnight rate"
        )

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

        overnight_rate = self._combine_sources_for(
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

        return finalize_dataset(
            dataset=resample_to_period(overnight_rate, period),
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Overnight Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_real_interest_rate(
        self,
        countries: list[str] | str | None = None,
        rate_type: str = "long_term",
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Real Interest Rate for a variety of countries over time. The Real Interest Rate
        is the nominal interest rate adjusted for inflation, and reflects the true cost of
        borrowing (or the true return earned on savings) once the erosion of purchasing power
        by inflation is taken into account.

        Formula (Fisher equation, approximation):

            Real Interest Rate = Nominal Interest Rate - Inflation Rate

        The nominal interest rate is either the Long Term Interest Rate (the 10-year government
        bond yield) or the Short Term Interest Rate (the 3-month money market rate), selected via
        the rate_type parameter. The Inflation Rate is only available on an annual basis (see
        get_inflation_rate), which comes from the Global Macro Database (GMDB). Both legs are
        annual decimal fractions (0.05 for 5%) whichever source is used, so they line up
        directly for the subtraction and the result is itself a decimal fraction (-0.0016 for
        a real rate of -0.16%).

        Changed in v2.2.0: this used to be returned in percentage points, because the GMDB
        legs were percentage points and the OECD nominal rate was multiplied by 100 to match
        them. The GMDB series are now decimal fractions and that rescaling has been removed,
        so the result is 100x smaller than in v2.1.x.

        A negative real interest rate means that, after inflation, savers are effectively losing
        purchasing power and borrowers are being subsidized in real terms; this occurred in many
        countries during the 2021-2022 inflation surge.

        Also known as: real yield, inflation-adjusted interest rate.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rate_type (str, optional): Which nominal interest rate to use. Can be 'long_term'
                (10-year government bond yield) or 'short_term' (3-month money market rate).
                Defaults to 'long_term'.
            gmdb_source (bool | None, optional): Whether to get the nominal interest rate from
                the Global Macro Database (GMDB) instead of the OECD. Defaults to None, which
                falls back to the gmdb_source set on the Economics class (True by default).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Real Interest Rate

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date='2023-01-01')

        economics.get_real_interest_rate(countries=['United States', 'Germany', 'Japan'])
        ```

        Which returns:

        |      |   United States |   Germany |   Japan |
        |:-----|----------------:|----------:|--------:|
        | 2018 |          0.0047 |   -0.0151 | -0.009  |
        | 2019 |          0.0033 |   -0.0163 | -0.0057 |
        | 2020 |         -0.0037 |   -0.0088 |  0.0001 |
        | 2021 |         -0.0324 |   -0.0358 |  0.0032 |
        | 2022 |         -0.0505 |   -0.0753 | -0.023  |
        | 2023 |         -0.0017 |   -0.0359 | -0.0269 |
        """
        rate_type = rate_type.lower()

        if rate_type not in ["long_term", "short_term"]:
            raise ValueError(
                "Please choose either 'long_term' or 'short_term' for the rate_type parameter."
            )

        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            nominal_interest_rate = (
                gmdb_model.get_long_term_interest_rate(
                    gmd_dataset=self._get_gmdb_dataset()
                )
                if rate_type == "long_term"
                else gmdb_model.get_short_term_interest_rate(
                    gmd_dataset=self._get_gmdb_dataset()
                )
            )
        else:
            nominal_interest_rate = (
                oecd_model.get_long_term_interest_rate(
                    period="yearly",
                    start_date=self._start_date,
                    end_date=self._end_date,
                )
                if rate_type == "long_term"
                else oecd_model.get_short_term_interest_rate(
                    period="yearly",
                    start_date=self._start_date,
                    end_date=self._end_date,
                )
            )

        inflation_rate = gmdb_model.get_inflation_rate(
            gmd_dataset=self._get_gmdb_dataset()
        )

        real_interest_rate = nominal_interest_rate - inflation_rate

        return finalize_dataset(
            dataset=real_interest_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Real Interest Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_yield_curve_slope(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Yield Curve Slope for a variety of countries over time. The Yield Curve Slope is
        the difference between the Long Term Interest Rate (the 10-year government bond yield)
        and the Short Term Interest Rate (the 3-month money market rate), and summarizes the
        overall shape of the yield curve in a single number.

        Formula:

            Yield Curve Slope = Long Term Interest Rate - Short Term Interest Rate

        A positive (upward-sloping) yield curve is the historical norm and reflects investors
        demanding a premium for locking up money for longer. A negative (inverted) yield curve,
        where short-term rates exceed long-term rates, has historically been one of the more
        reliable leading indicators of an upcoming recession, as it signals that markets expect
        the central bank to cut rates in response to a weakening economy.

        Both legs are decimal fractions (0.05 for 5%) whichever source is used, so the result
        is itself a decimal fraction (-0.0122 for an inversion of 1.22 percentage points),
        matching the convention used by get_misery_index and get_real_interest_rate.

        Changed in v2.2.0: this used to be returned in percentage points, because the GMDB
        legs were percentage points and the OECD legs were multiplied by 100 to match them.
        The GMDB series are now decimal fractions and that rescaling has been removed, so the
        result is 100x smaller than in v2.1.x.

        Also known as: term spread, 10Y-3M spread, curve inversion.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Yield Curve Slope

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-01-01', end_date='2023-12-31')

        economics.get_yield_curve_slope(
            countries=['United States', 'Germany', 'Japan'],
            period='yearly'
        )
        ```

        Which returns:

        |      |   United States |   Germany |   Japan |
        |:-----|----------------:|----------:|--------:|
        | 2021 |          0.0139 |    0.0018 | -0      |
        | 2022 |          0.0071 |    0.008  |  0.0016 |
        | 2023 |         -0.0122 |   -0.01   |  0.0049 |
        """
        check_period_type(period)

        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:

            long_term_interest_rate = gmdb_model.get_long_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
            short_term_interest_rate = gmdb_model.get_short_term_interest_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            long_term_interest_rate = oecd_model.get_long_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )
            short_term_interest_rate = oecd_model.get_short_term_interest_rate(
                period=period,
                start_date=self._start_date,
                end_date=self._end_date,
            )

        yield_curve_slope = long_term_interest_rate - short_term_interest_rate

        return finalize_dataset(
            dataset=yield_curve_slope,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Yield Curve Slope",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_renewable_energy(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Renewable energy is defined as the contribution of renewables to total primary energy supply (TPES).
        Renewables include the primary energy equivalent of hydro (excluding pumped storage), geothermal,
        solar, wind, tide and wave sources.

        Energy derived from solid biofuels, biogasoline, biodiesels, other liquid biofuels, biogases and
        the renewable fraction of municipal waste are also included. Biofuels are defined as fuels derived
        directly or indirectly from biomass (material obtained from living or recently living organisms).

        This includes wood, vegetal waste (including wood waste and crops used for energy production), ethanol,
        animal materials/wastes and sulphite lyes. Municipal waste comprises wastes produced by the residential,
        commercial and public service sectors that are collected by local authorities for disposal in a central
        location for the production of heat and/or power.

        This indicator is the renewable share of total primary energy supply, returned as an
        annual decimal fraction (0.0872 for 8.72%).

        See definition: https://data.oecd.org/energy/renewable-energy.htm

        Also known as: clean energy, green energy, renewable energy share.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Renewable Energy Percentage.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date='2020-01-01')

        economics.get_renewable_energy(countries=['Austria', 'Germany', 'United States'])
        ```

        Which returns:

        |      |   Austria |   Germany |   United States |
        |:-----|----------:|----------:|----------------:|
        | 2010 |    0.2742 |    0.0933 |          0.0569 |
        | 2011 |    0.2696 |    0.102  |          0.0619 |
        | 2012 |    0.307  |    0.1137 |          0.0632 |
        | 2013 |    0.3011 |    0.1147 |          0.0666 |
        | 2014 |    0.3068 |    0.1192 |          0.0678 |
        | 2015 |    0.2985 |    0.1264 |          0.0678 |
        | 2016 |    0.3034 |    0.1253 |          0.071  |
        | 2017 |    0.2984 |    0.1332 |          0.0743 |
        | 2018 |    0.2944 |    0.1396 |          0.0767 |
        | 2019 |    0.3006 |    0.1485 |          0.078  |
        | 2020 |    0.3191 |    0.1637 |          0.0837 |
        """
        renewable_energy = oecd_model.get_renewable_energy(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=renewable_energy,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Renewable Energy",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_carbon_footprint(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The carbon footprint is a measure of the total amount of greenhouse gases produced
        to directly and indirectly support human activities, usually expressed in equivalent
        tons of carbon dioxide (CO2).

        The carbon footprint is a subset of the ecological footprint and of the more comprehensive
        Life Cycle Assessment (LCA). An individual, nation, or organization's carbon footprint can
        be measured by undertaking a GHG emissions assessment or other calculative activities
        denoted as carbon accounting.

        The data is sourced from the greenhouse gas emissions per capita indicator of the
        OECD's How's Life? well-being database (dataset ``DSD_HSL@DF_HSL_FWB``, indicator
        ``12_9``), so the figures are expressed in tonnes of CO2 equivalent per person, on
        an annual basis.

        This series currently ends in 2020, so a date range that starts after that returns
        an empty DataFrame and later years are absent rather than NaN.

        Also known as: CO2 emissions, carbon emissions, greenhouse gas.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Carbon Footprint.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2010-01-01", end_date="2020-01-01")

        economics.get_carbon_footprint(countries=['Germany', 'United States', 'Poland'])
        ```

        Which returns:

        |      |   Germany |   United States |   Poland |
        |:-----|----------:|----------------:|---------:|
        | 2010 |    14.574 |          22.85  |   10.557 |
        | 2011 |    14.512 |          22.081 |   10.479 |
        | 2012 |    13.938 |          21.212 |   10.14  |
        | 2013 |    14.139 |          21.404 |    9.749 |
        | 2014 |    13.74  |          21.551 |    9.703 |
        | 2015 |    13.318 |          21.253 |    9.55  |
        | 2016 |    13.37  |          20.577 |    9.772 |
        | 2017 |    13.205 |          20.353 |   10.228 |
        | 2018 |    13.267 |          20.8   |   10.377 |
        | 2019 |    12.582 |          20.191 |    9.838 |
        | 2020 |    11.498 |          18.34  |    9.451 |
        """
        carbon_footprint_df = oecd_model.get_carbon_footprint(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=carbon_footprint_df,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Carbon Footprint",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_unemployment_rate(
        self,
        countries: list[str] | str | None = None,
        period: str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The unemployed are people of working age who are without work,
        are available for work, and have taken specific steps to find work.
        The uniform application of this definition results in estimates of
        unemployment rates that are more internationally comparable than
        estimates based on national definitions of unemployment.

        This indicator is measured in numbers of unemployed people as a
        percentage of the labour force and it is seasonally adjusted.
        The labour force is defined as the total number of unemployed people
        plus those in employment. Data are based on labour force surveys (LFS).

        For European Union countries where monthly LFS information is not available,
        the monthly unemployed figures are estimated by Eurostat.

        See definition: https://data.oecd.org/unemp/unemployment-rate.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources return the rate as a decimal fraction of the labour force (0.036 for
        3.6%), so the two are directly interchangeable. Only the OECD source supports
        monthly and quarterly frequency; the GMDB is annual only, so the period argument
        has no effect when gmdb_source is True.

        With period="monthly" (unless gmdb_source=True is passed explicitly) the euro area
        and the countries Eurostat covers come from Eurostat and the United Kingdom from
        the Office for National Statistics, which publish weeks before the OECD republishes
        their figures, and Brazil from the IBGE (as the UK, the average of the three months
        ending in each month); every other country comes from the OECD. The United States is
        extended with the months the Bureau of Labor Statistics has published since, when
        both agree on the months they share. None of these need an API key. The UK figure is the average of the three months ending in each month, as the
        Labour Force Survey reports it.

        Changed in v2.2.0: the GMDB source previously returned percentage points (3.6 for
        3.6%) while the OECD source returned a decimal fraction. The GMDB series is now
        divided by 100 so both sources agree; divide any hard-coded comparison by 100.

        Also known as: jobless rate, labor market, unemployment level.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Unemployment Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-03-01', end_date='2023-01-01')

        economics.get_unemployment_rate(
            countries=['Germany', 'United States', 'Japan'],
            gmdb_source=False,
            period='quarterly'
        )
        ```

        Which returns:

        |        |   Germany |   United States |   Japan |
        |:-------|----------:|----------------:|--------:|
        | 2021Q1 |    0.039  |          0.0623 |  0.0287 |
        | 2021Q2 |    0.037  |          0.0593 |  0.029  |
        | 2021Q3 |    0.0343 |          0.0507 |  0.0277 |
        | 2021Q4 |    0.0337 |          0.0417 |  0.0273 |
        | 2022Q1 |    0.0323 |          0.0387 |  0.027  |
        | 2022Q2 |    0.031  |          0.0363 |  0.026  |
        | 2022Q3 |    0.031  |          0.0353 |  0.0253 |
        | 2022Q4 |    0.031  |          0.0357 |  0.0253 |
        | 2023Q1 |    0.0303 |          0.0353 |  0.026  |
        """
        check_period_type(period)

        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )
        # The GMDB is annual, so a monthly request goes to the monthly sources unless the
        # GMDB is asked for explicitly.
        monthly = period.lower() == "monthly" and gmdb_source is not True
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if monthly:
            # Eurostat and the ONS publish weeks before the OECD republishes their figures.
            unemployment_rate = self._combine_sources_for(
                [
                    (
                        lambda: eurostat_model.get_unemployment_rate(
                            buffered_start_date(self._start_date, "monthly"),
                            self._end_date,
                        ),
                        None,
                    ),
                    (ons_model.get_unemployment_rate, "United Kingdom"),
                    (ibge_model.get_unemployment_rate, "Brazil"),
                    # The OECD republishes the BLS figure a month later, so the US is
                    # extended with the months the BLS already has.
                    (
                        lambda: extend_with_recent(
                            oecd_model.get_unemployment_rate(
                                period="monthly",
                                start_date=self._start_date,
                                end_date=self._end_date,
                            ),
                            bls_model.get_unemployment_rate(),
                            "United States",
                        ),
                        None,
                    ),
                ],
                countries,
            )
        elif gmdb_source:

            unemployment_rate = gmdb_model.get_unemployment_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            unemployment_rate = oecd_model.get_unemployment_rate(
                period=period, start_date=self._start_date, end_date=self._end_date
            )

        return finalize_dataset(
            dataset=unemployment_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Unemployment Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            # Rows empty for every selected country, e.g. a month only another source has.
            dropna=monthly,
        )

    @handle_errors
    def get_misery_index(
        self,
        countries: list[str] | str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Misery Index for a variety of countries over time. The Misery Index is a simple
        gauge of the overall economic discomfort felt by the average person, combining the two
        economic ills that are most directly and visibly felt by households: unemployment and
        rising prices.

        Formula:

            Misery Index = Unemployment Rate + Inflation Rate

        The Unemployment Rate and Inflation Rate are both retrieved as annual decimal fractions
        (0.05 for 5%) whichever source is used, so they line up directly for the addition and
        the result is itself a decimal fraction (0.0774 for a Misery Index of 7.74).

        Changed in v2.2.0: this used to be returned in percentage points, because both GMDB
        legs were percentage points and the OECD unemployment rate was multiplied by 100 to
        match them. The GMDB series are now decimal fractions and that rescaling has been
        removed, so the result is 100x smaller than in v2.1.x.

        A higher Misery Index indicates a more uncomfortable economic climate for the average
        household, while a lower value indicates a more comfortable one. It was originally
        popularized by economist Arthur Okun.

        Also known as: economic discomfort index, Okun's misery index.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            gmdb_source (bool | None, optional): Whether to get the unemployment rate from the
                Global Macro Database (GMDB) instead of the OECD. Defaults to None, which falls
                back to the gmdb_source set on the Economics class (True by default).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Misery Index

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2018-01-01', end_date='2023-01-01')

        economics.get_misery_index(countries=['United States', 'Germany', 'Japan'])
        ```

        Which returns:

        |      |   United States |   Germany |   Japan |
        |:-----|----------------:|----------:|--------:|
        | 2018 |          0.0633 |    0.051  |  0.0341 |
        | 2019 |          0.0549 |    0.0433 |  0.0284 |
        | 2020 |          0.0935 |    0.0399 |  0.0279 |
        | 2021 |          0.1003 |    0.0681 |  0.0256 |
        | 2022 |          0.1164 |    0.1181 |  0.0511 |
        | 2023 |          0.0775 |    0.091  |  0.0582 |
        """
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            unemployment_rate = gmdb_model.get_unemployment_rate(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            unemployment_rate = oecd_model.get_unemployment_rate(
                period="yearly", start_date=self._start_date, end_date=self._end_date
            )

        inflation_rate = gmdb_model.get_inflation_rate(
            gmd_dataset=self._get_gmdb_dataset()
        )

        misery_index = unemployment_rate + inflation_rate

        return finalize_dataset(
            dataset=misery_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Misery Index",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_labour_productivity(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        GDP per hour worked is a measure of labour productivity. It measures
        how efficiently labour input is combined with other factors of production
        and used in the production process. Labour input is defined as total hours
        worked of all persons engaged in production. Labour productivity only partially
        reflects the productivity of labour in terms of the personal capacities of
        workers or the intensity of their effort.

        The ratio between the output measure and the labour input depends to a large
        degree on the presence and/or use of other inputs (e.g. capital, intermediate
        inputs, technical, organisational and efficiency change, economies of scale).

        The level is reported in US dollars per hour worked at constant prices (currently
        referenced to 2020), converted with Purchasing Power Parities (PPPs) so that it is
        comparable across countries, for the total economy and on an annual basis. It is a
        level rather than an index, so a value of 61.36 means 61.36 PPP-converted US dollars
        of GDP produced per hour worked.

        See definition: https://data.oecd.org/lprdty/gdp-per-hour-worked.htm

        Also known as: labor productivity, output per worker.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Labour Productivity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(end_date="2025-12-31")

        economics.get_labour_productivity(countries=['Bulgaria', 'Croatia', 'Spain'])
        ```

        Which returns:

        |      |   Bulgaria |   Croatia |   Spain |
        |:-----|-----------:|----------:|--------:|
        | 2016 |    28.0515 |   37.5651 | 60.3429 |
        | 2017 |    28.3199 |   37.9134 | 60.7889 |
        | 2018 |    28.9871 |   38.9686 | 60.7456 |
        | 2019 |    30.5046 |   39.5761 | 60.8858 |
        | 2020 |    30.7971 |   37.1191 | 60.9134 |
        | 2021 |    32.9483 |   41.2317 | 60.6022 |
        | 2022 |    33.9708 |   43.6926 | 61.3551 |
        | 2023 |    33.3439 |   44.313  | 61.2281 |
        | 2024 |    34.8229 |   43.9879 | 61.9601 |
        | 2025 |   nan      |   45.0352 | 62.42   |
        """
        labour_productivity = oecd_model.get_labour_productivity(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=labour_productivity,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Labour Productivity",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_income_inequality(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Income is defined as household disposable income in a particular year. It consists of earnings,
        self-employment and capital income and public cash transfers; income taxes and social
        security contributions paid by households are deducted. The income of the household is
        attributed to each of its members, with an adjustment to reflect differences in needs for
        households of different sizes.

        The Gini coefficient is based on the comparison of cumulative proportions of the population against
        cumulative proportions of income they receive, and it ranges between 0 in the case of perfect equality
        and 1 in the case of perfect inequality.

        One Gini coefficient is returned per country, for the total population and on the
        OECD's current income definition (in use since 2012). The other inequality measures
        published alongside it in the same dataflow (the P90/P10, P90/P50 and P50/P10 decile
        ratios, the Palma ratio and the S80/S20 quintile share) are not returned here.

        See definition: https://data.oecd.org/inequality/income-inequality.htm

        Also known as: Gini coefficient, income distribution.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Income Inequality.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2013-01-01', end_date='2021-12-31')

        economics.get_income_inequality(countries=['United States', 'Germany', 'Japan'])
        ```

        Which returns:

        |      |   United States |   Germany |   Japan |
        |:-----|----------------:|----------:|--------:|
        | 2013 |          0.396  |    0.2922 | nan     |
        | 2014 |          0.3938 |    0.2887 | nan     |
        | 2015 |          0.3896 |    0.2932 | nan     |
        | 2016 |          0.3912 |    0.2944 | nan     |
        | 2017 |          0.3899 |    0.2892 | nan     |
        | 2018 |          0.3927 |    0.2893 |   0.334 |
        | 2019 |          0.3949 |    0.2959 | nan     |
        | 2020 |          0.3773 |    0.3026 | nan     |
        | 2021 |          0.3752 |    0.3125 |   0.338 |
        """
        income_inequality = oecd_model.get_income_inequality(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=income_inequality,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Income Inequality",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_population_statistics(
        self,
        countries: list[str] | str | None = None,
        gmdb_source: bool | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Population is defined as all nationals present in, or temporarily absent from a country,
        and aliens permanently settled in a country. This indicator shows the number of people
        that usually live in an area. Growth rates are the annual changes in population resulting
        from births, deaths and net migration during the year.

        Total population includes the following:

            - national armed forces stationed abroad; merchant seamen at sea;
            - diplomatic personnel located abroad;
            - civilian aliens resident in the country;
            - displaced persons resident in the country.

        However, it excludes the following:

            - foreign armed forces stationed in the country;
            - foreign diplomatic personnel located in the country;
            - civilian aliens temporarily in the country.

        Population projections are a common demographic tool. They provide a basis for other
        statistical projections, helping governments in their decision making.

        The Global Macro Database (GMDB) source returns a single total population series
        per country, in millions of people. The OECD source additionally breaks the total
        down by gender, giving a Population, Men and Women series for each country, and
        reports a plain count of persons rather than millions. Both are annual, and the
        OECD source uses the historical (observed) series rather than its projections.

        See definition: https://data.oecd.org/pop/population.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Also known as: demographic data, census data.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Population Statistics.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date='2019-01-01')

        economics.get_population_statistics(countries='Japan')
        ```

        Which returns:

        |      |   Japan |
        |:-----|--------:|
        | 2010 | 128.043 |
        | 2011 | 127.831 |
        | 2012 | 127.552 |
        | 2013 | 127.333 |
        | 2014 | 127.262 |
        | 2015 | 127.11  |
        | 2016 | 126.933 |
        | 2017 | 126.706 |
        | 2018 | 126.443 |
        | 2019 | 126.167 |
        """
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:

            population_statistics_df = gmdb_model.get_population(
                gmd_dataset=self._get_gmdb_dataset()
            )
        else:
            population_statistics = {}

            population_statistics["Population"] = oecd_model.get_population(
                start_date=self._start_date, end_date=self._end_date
            )
            population_statistics["Men"] = oecd_model.get_population(
                gender="men", start_date=self._start_date, end_date=self._end_date
            )
            population_statistics["Women"] = oecd_model.get_population(
                gender="women", start_date=self._start_date, end_date=self._end_date
            )

            population_statistics_df = pd.concat(population_statistics, axis=0).unstack(
                level=0
            )

        return finalize_dataset(
            dataset=population_statistics_df,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Population Statistics",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_poverty_rate(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        The poverty rate is the ratio of the number of people (in a given age group) whose income
        falls below the poverty line; taken as half the median household income of the total population.

        However, two countries with the same poverty rates may differ in terms of the relative income-level of the poor.

        See definition: https://data.oecd.org/inequality/poverty-rate.htm

        Also known as: poverty rate, income poverty.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Poverty Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2012-01-01', end_date='2020-01-01')

        economics.get_poverty_rate(countries='Portugal')
        ```

        Which returns:

        |      |   Portugal |
        |:-----|-----------:|
        | 2012 |     0.1295 |
        | 2013 |     0.135  |
        | 2014 |     0.135  |
        | 2015 |     0.1255 |
        | 2016 |     0.1246 |
        | 2017 |     0.1067 |
        | 2018 |     0.1038 |
        | 2019 |     0.1058 |
        | 2020 |     0.1279 |
        """
        poverty_rate = oecd_model.get_poverty_rate(
            start_date=self._start_date, end_date=self._end_date
        )

        return finalize_dataset(
            dataset=poverty_rate,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Poverty Rate",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_sovereign_debt_crisis(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        rounding: int | None = None,
    ):
        """
        Get the Sovereign Debt Crisis dummy for a variety of countries over time from the Global
        Macro Database (GMDB). Unlike the other indicators in this module, this is a binary
        (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating series rather than a
        continuous economic series: a value of 1 marks a year in which a country was undergoing a
        sovereign debt crisis (e.g. a default or restructuring of government debt), and 0 marks a
        year in which it was not.

        The crisis dating stops well short of the present -- the series currently ends in 2017 --
        so recent years are NaN rather than 0.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: sovereign default, debt crisis dummy.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Sovereign Debt Crisis dummy

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='1980-01-01', end_date="2025-12-31")

        economics.get_sovereign_debt_crisis(countries='Argentina')
        ```

        Which returns:

        |      |   Argentina |
        |:-----|------------:|
        | 2021 |         nan |
        | 2022 |         nan |
        | 2023 |         nan |
        | 2024 |         nan |
        | 2025 |         nan |
        """

        sovereign_debt_crisis = gmdb_model.get_sovereign_debt_crisis(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=sovereign_debt_crisis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Sovereign Debt Crisis",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            rounding=rounding,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_currency_crisis(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        rounding: int | None = None,
    ):
        """
        Get the Currency Crisis dummy for a variety of countries over time from the Global Macro
        Database (GMDB). Unlike the other indicators in this module, this is a binary (0 = no
        crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating series rather than a continuous
        economic series: a value of 1 marks a year in which a country was undergoing a currency
        crisis (e.g. a sharp, disorderly depreciation or collapse of the exchange rate), and 0
        marks a year in which it was not.

        The crisis dating stops well short of the present -- the series currently ends in 2017 --
        so recent years are NaN rather than 0.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: currency collapse, exchange rate crisis dummy.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Currency Crisis dummy

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='1990-01-01', end_date="2025-12-31")

        economics.get_currency_crisis(countries='Turkey')
        ```

        Which returns:

        |      |   Turkey |
        |:-----|---------:|
        | 2015 |        0 |
        | 2016 |        0 |
        | 2017 |        0 |
        | 2018 |      nan |
        | 2019 |      nan |
        """

        currency_crisis = gmdb_model.get_currency_crisis(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=currency_crisis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Currency Crisis",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            rounding=rounding,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_banking_crisis(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        rounding: int | None = None,
    ):
        """
        Get the Banking Crisis dummy for a variety of countries over time from the Global Macro
        Database (GMDB). Unlike the other indicators in this module, this is a binary (0 = no
        crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating series rather than a continuous
        economic series: a value of 1 marks a year in which a country was undergoing a systemic
        banking crisis (e.g. bank runs, large-scale bank failures or government intervention to
        prevent them), and 0 marks a year in which it was not.

        The crisis dating stops well short of the present -- the series currently ends in 2020 --
        so recent years are NaN rather than 0.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: banking panic, financial crisis dummy, systemic banking crisis.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Banking Crisis dummy

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2005-01-01', end_date="2025-12-31")

        economics.get_banking_crisis(countries=['United States', 'United Kingdom'])
        ```

        Which returns:

        |      |   United States |   United Kingdom |
        |:-----|----------------:|-----------------:|
        | 2005 |               0 |                0 |
        | 2006 |               0 |                0 |
        | 2007 |               1 |                0 |
        | 2008 |               0 |                1 |
        | 2009 |               0 |                0 |
        | 2010 |               0 |                0 |
        | 2011 |               0 |                0 |
        | 2012 |               0 |                0 |
        | 2013 |               0 |                0 |
        | 2014 |               0 |                0 |
        | 2015 |               0 |                0 |
        | 2016 |               0 |                0 |
        | 2017 |               0 |                0 |
        | 2018 |               0 |                0 |
        | 2019 |               0 |                0 |
        | 2020 |               0 |                0 |
        | 2021 |               0 |                0 |
        | 2022 |               0 |                0 |
        | 2023 |               0 |                0 |
        | 2024 |               0 |                0 |
        | 2025 |               0 |                0 |
        """

        banking_crisis = gmdb_model.get_banking_crisis(
            gmd_dataset=self._get_gmdb_dataset()
        )

        return finalize_dataset(
            dataset=banking_crisis,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Banking Crisis",
            countries=countries,
            rolling=rolling,
            trailing=trailing,
            rounding=rounding,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_nonfarm_payrolls(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get Total Nonfarm Payroll Employment for the United States from the Bureau of
        Labor Statistics (via FRED).

        Nonfarm Payrolls is the headline monthly employment report and one of the most
        closely watched real-activity indicators in macroeconomics: it counts the
        number of paid US workers excluding farm employees, general government
        employees, private household employees and nonprofit organization employees.
        Sharp month-over-month changes are a core input to business-cycle dating (used
        directly by the NBER's Business Cycle Dating Committee) and, through Okun's
        Law, are closely tied to changes in the Unemployment Rate (see
        `get_unemployment_rate`).

        The series is the monthly level of employment in thousands of persons, seasonally
        adjusted, so 156857 means 156.857 million jobs -- not the monthly change that the
        headline "jobs added" number refers to. Use growth=True for that change.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/PAYEMS

        Also known as: NFP, nonfarm employment, the "jobs report".

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of nonfarm payroll
            employment, in thousands of persons.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
            fred_api_key='FRED_API_KEY',
            end_date="2025-12-31",
        )

        economics.get_nonfarm_payrolls()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-08-01 |          158472 |
        | 2025-09-01 |          158548 |
        | 2025-10-01 |          158408 |
        | 2025-11-01 |          158449 |
        | 2025-12-01 |          158432 |
        """
        self._require_fred_api_key()

        nonfarm_payrolls = fred_model.get_nonfarm_payrolls(
            self._start_date, self._end_date, self._fred_api_key
        )

        return finalize_dataset(
            dataset=nonfarm_payrolls,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_initial_jobless_claims(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get weekly Initial Claims for Unemployment Insurance for the United States
        from the Department of Labor (via FRED).

        Initial Jobless Claims counts the number of individuals filing for
        unemployment insurance for the first time in a given week. Because it is
        reported weekly (versus Nonfarm Payrolls' monthly cadence, see
        `get_nonfarm_payrolls`) and captures layoffs essentially in real time, it is
        one of the most timely leading indicators of labor-market deterioration and a
        core component of the Conference Board's Leading Economic Index.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/ICSA

        Also known as: initial claims, new unemployment claims.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of weekly initial
            jobless claims, seasonally adjusted.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
            fred_api_key='FRED_API_KEY',
            end_date="2025-12-31",
        )

        economics.get_initial_jobless_claims()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-11-29 |          216000 |
        | 2025-12-06 |          235000 |
        | 2025-12-13 |          224000 |
        | 2025-12-20 |          215000 |
        | 2025-12-27 |          203000 |
        """
        self._require_fred_api_key()

        initial_jobless_claims = fred_model.get_initial_jobless_claims(
            self._start_date, self._end_date, self._fred_api_key
        )

        return finalize_dataset(
            dataset=initial_jobless_claims,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_retail_sales(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get Advance Retail Sales (Retail and Food Services) for the United States from
        the Census Bureau (via FRED).

        Retail Sales measures nominal spending at retail and food-service
        establishments. Since Personal Consumption Expenditures make up roughly
        two-thirds to three-quarters of US GDP, this monthly, high-frequency series is
        a core input to real-time (nowcast) GDP estimates such as the Federal Reserve
        Bank of Atlanta's GDPNow.

        The series is monthly, in millions of US dollars, seasonally adjusted, and covers
        retail trade and food services. It is nominal, so growth=True mixes volume and price
        changes together; compare against the Consumer Price Index to separate the two.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/RSAFS

        Also known as: retail trade, consumer spending (proxy).

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of total retail
            and food services sales, in millions of dollars.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
            fred_api_key='FRED_API_KEY',
            end_date="2025-12-31",
        )

        economics.get_retail_sales()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-08-01 |          700218 |
        | 2025-09-01 |          701453 |
        | 2025-10-01 |          700790 |
        | 2025-11-01 |          705453 |
        | 2025-12-01 |          706264 |
        """
        self._require_fred_api_key()

        retail_sales = fred_model.get_retail_sales(
            self._start_date, self._end_date, self._fred_api_key
        )

        return finalize_dataset(
            dataset=retail_sales,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_industrial_production_index(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the Industrial Production Index for the United States from the Federal
        Reserve's G.17 statistical release (via FRED).

        The Industrial Production Index measures real output in manufacturing,
        mining, and electric and gas utilities. Unlike survey-based sentiment
        indices, it is a hard, quantity-based measure of physical production and is
        one of the four coincident indicators the NBER's Business Cycle Dating
        Committee uses to date US recessions (alongside real personal income, real
        manufacturing/trade sales and, see `get_nonfarm_payrolls`, nonfarm payroll
        employment).

        No API key is needed: without a FRED API key the data comes from
        the Federal Reserve's G.17 release directly, which gives the same figures. When a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/INDPRO

        Also known as: IP index, industrial output.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of the
            Industrial Production Index (2017 = 100).

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date="2025-12-31")

        economics.get_industrial_production_index()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-08-01 |         101.625 |
        | 2025-09-01 |         101.668 |
        | 2025-10-01 |         101.219 |
        | 2025-11-01 |         101.034 |
        | 2025-12-01 |         101.494 |
        """
        # FRED republishes these figures, so it is used when a key is set and
        # the Federal Reserve Board otherwise.
        industrial_production_index = (
            fred_model.get_industrial_production_index(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else frb_model.get_industrial_production_index()
        )

        return finalize_dataset(
            dataset=industrial_production_index,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_housing_starts(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get Housing Starts (Total New Privately-Owned Housing Units Started) for the
        United States from the Census Bureau (via FRED).

        Housing Starts counts the number of new residential construction projects
        that have begun in a given month. Residential investment is one of the most
        interest-rate-sensitive components of GDP, and construction activity leads
        the broader business cycle (it typically turns down before a recession and
        turns up before a recovery), making Housing Starts one of the ten components
        of the Conference Board's Leading Economic Index.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/HOUST

        Also known as: new residential construction.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of new housing
            starts, in thousands of units, seasonally adjusted annual rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
            fred_api_key='FRED_API_KEY',
            end_date="2025-12-31",
        )

        economics.get_housing_starts()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-08-01 |            1291 |
        | 2025-09-01 |            1319 |
        | 2025-10-01 |            1273 |
        | 2025-11-01 |            1319 |
        | 2025-12-01 |            1378 |
        """
        self._require_fred_api_key()

        housing_starts = fred_model.get_housing_starts(
            self._start_date, self._end_date, self._fred_api_key
        )

        return finalize_dataset(
            dataset=housing_starts,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_real_personal_income(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get Real Personal Income Excluding Current Transfer Receipts for the United
        States from the Bureau of Economic Analysis (via FRED).

        This is the exact series (not a proxy) the NBER's Business Cycle Dating
        Committee uses as one of its four primary coincident indicators for dating
        US recessions — alongside Nonfarm Payrolls (see `get_nonfarm_payrolls`),
        the Industrial Production Index (see `get_industrial_production_index`) and
        Real Personal Consumption Expenditures. It measures aggregate household
        income from wages, investments and proprietors' income, deliberately
        excluding government transfer payments (e.g. unemployment insurance, Social
        Security) so that the series reflects income generated by ongoing economic
        activity rather than the fiscal cushioning that automatically increases
        during a downturn.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/W875RX1

        Also known as: RPI less transfers, NBER real income indicator.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of real personal
            income excluding current transfer receipts, in billions of chained 2017
            dollars.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
            fred_api_key='FRED_API_KEY',
            end_date="2025-12-31",
        )

        economics.get_real_personal_income()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-08-01 |         16911.2 |
        | 2025-09-01 |         16923.1 |
        | 2025-10-01 |         16921.8 |
        | 2025-11-01 |         16960.6 |
        | 2025-12-01 |         16943.8 |
        """
        self._require_fred_api_key()

        real_personal_income = fred_model.get_real_personal_income(
            self._start_date, self._end_date, self._fred_api_key
        )

        return finalize_dataset(
            dataset=real_personal_income,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_mortgage_rate_30_year(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the weekly average 30-Year Fixed Rate Mortgage from FRED (Freddie
        Mac's Primary Mortgage Market Survey).

        The 30-year fixed mortgage rate is the primary interest rate US households
        actually borrow at for home purchases, and is one of the clearest single
        transmission points from Federal Reserve policy to the real economy: it
        moves with (but is not identical to) the 10-year Treasury yield plus a
        credit/prepayment spread, and directly drives housing affordability and
        demand. It is the natural interest-rate complement to Housing Starts (see
        `get_housing_starts`) — rate moves here lead construction activity, since
        higher borrowing costs price marginal buyers out of the market before
        builders scale back new projects.

        The rate is weekly (week ending Thursday), returned as a decimal fraction per annum
        (0.0648 for 6.48%) and not seasonally adjusted. FRED publishes it in percentage
        points; it is rescaled here so that every rate the Finance Toolkit returns is a
        decimal fraction.

        No API key is needed: without a FRED API key the data comes from
        Freddie Mac's Primary Mortgage Market Survey directly, which gives the same figures. When a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/MORTGAGE30US

        Also known as: 30-year mortgage rate, Freddie Mac PMMS rate.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of the weekly
            average 30-year fixed mortgage rate, as a decimal fraction per annum.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date="2025-12-31")

        economics.get_mortgage_rate_30_year()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-12-04 |          0.0619 |
        | 2025-12-11 |          0.0622 |
        | 2025-12-18 |          0.0621 |
        | 2025-12-24 |          0.0618 |
        | 2025-12-31 |          0.0615 |
        """
        # FRED republishes these figures, so it is used when a key is set and
        # Freddie Mac otherwise.
        mortgage_rate_30_year = (
            fred_model.get_mortgage_rate_30_year(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else freddie_mac_model.get_mortgage_rate_30_year()
        )

        return finalize_dataset(
            dataset=mortgage_rate_30_year,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_recession_indicator(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the NBER-based US Recession Indicator from FRED.

        This is the official US business-cycle chronology maintained by the
        National Bureau of Economic Research (NBER) Business Cycle Dating
        Committee, encoded as 1 during NBER-dated recession months (peak through
        trough) and 0 otherwise. The Committee determines recession dates
        retrospectively from a broad set of coincident indicators — including
        Nonfarm Payrolls (see `get_nonfarm_payrolls`) and the Industrial Production
        Index (see `get_industrial_production_index`) — rather than the popular
        "two consecutive quarters of negative GDP growth" rule of thumb, which the
        NBER does not use. This series is the standard ground-truth label used in
        academic and applied business-cycle research to backtest whether other
        indicators lead, lag or coincide with recessions.

        No API key is needed: without a FRED API key the data comes from
        the NBER's business cycle dates directly, built the way FRED builds USREC, which gives the same figures. When a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/USREC

        Also known as: USREC, NBER recession dummy, business cycle indicator.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame, 1 during
            NBER-dated recession months and 0 otherwise.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date="2025-12-31")

        economics.get_recession_indicator()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-08-01 |               0 |
        | 2025-09-01 |               0 |
        | 2025-10-01 |               0 |
        | 2025-11-01 |               0 |
        | 2025-12-01 |               0 |
        """
        # FRED republishes these figures, so it is used when a key is set and
        # the NBER otherwise.
        recession_indicator = (
            fred_model.get_recession_indicator(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else nber_model.get_recession_indicator()
        )

        return finalize_dataset(
            dataset=recession_indicator,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_commercial_real_estate_prices(
        self,
        countries: list[str] | str | None = None,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the quarterly growth of commercial real estate prices: the change in the price
        of commercial property (offices, retail, industrial and apartment buildings) on the
        same quarter a year earlier, for around twenty economies.

        This tracks commercial (office, retail, industrial, apartment) property
        prices, as distinct from residential house prices (see `get_house_prices`,
        which tracks a completely different asset class/market). It is a
        transaction-based index rather than the appraisal-smoothed methodology used
        by institutional benchmarks like the NCREIF Property Index -- which is not
        freely available anywhere -- so expect more volatility and less
        autocorrelation than an appraisal-based series would show.

        No API key is needed: the prices come from the commercial property price indices
        the Bank for International Settlements collects from national sources, for the euro
        area, the United States (from 1945), Japan, Denmark, Switzerland, Germany, France,
        Spain and around a dozen more. When a FRED API key is set (see the `fred_api_key`
        parameter of the `Economics` class), the United States instead comes from FRED's
        series sourced from the IMF's Financial Soundness Indicators, as in earlier versions.

        See definitions: https://data.bis.org/topics/CPP and
        https://fred.stlouisfed.org/series/COMREPUSQ159N

        Also known as: commercial property price index, CRE price index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The quarterly change in commercial real estate prices on a year
            earlier, as a decimal fraction, with a column per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2025-01-01', end_date="2025-12-31")

        economics.get_commercial_real_estate_prices(countries=['United States', 'Euro Area', 'Japan', 'Germany'])
        ```

        Which returns:

        |        |   United States |   Euro Area |   Japan |   Germany |
        |:-------|----------------:|------------:|--------:|----------:|
        | 2025Q1 |         -0.027  |      0.0232 |  0.0289 |    0.0221 |
        | 2025Q2 |         -0.0494 |      0.0331 |  0.0197 |    0.0266 |
        | 2025Q3 |          0.0209 |      0.0173 |  0.0222 |    0.0264 |
        | 2025Q4 |          0.0109 |      0.0189 |  0.0104 |    0.0207 |
        """
        # The index is turned into its change on the same quarter a year earlier, the
        # measure FRED publishes, so both sources mean the same thing.
        index = bis_model.get_commercial_property_prices(
            buffered_start_date(self._start_date, "monthly"), self._end_date
        )
        commercial_real_estate_prices = index.pct_change(4, fill_method=None)

        if self._fred_api_key:
            united_states = fred_model.get_commercial_real_estate_prices(
                self._start_date, self._end_date, self._fred_api_key
            )
            if not united_states.empty:
                united_states.index = united_states.index.asfreq("Q")
                commercial_real_estate_prices = commercial_real_estate_prices.drop(
                    columns="United States", errors="ignore"
                ).join(united_states, how="outer")

        return finalize_dataset(
            dataset=commercial_real_estate_prices,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            countries=countries,
            dropna=True,
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

    def _get_united_states_inflation_curve(
        self, curve: str, start_date: str
    ) -> pd.DataFrame:
        """
        Retrieves the US real yield curve or breakeven inflation curve, from FRED when a key
        is set and from the U.S. Department of the Treasury otherwise, with the maturities
        labelled as elsewhere ("5Y" and "5Y5Y" for the 5-year, 5-year forward rate).

        Args:
            curve (str): "real" or "breakeven".
            start_date (str): The start date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: The curve, indexed by day with a column per maturity.
        """
        # FRED republishes these figures, so it is used when a key is set and the U.S.
        # Treasury otherwise.
        if curve == "real":
            values = (
                fred_model.get_real_yield_curve(
                    start_date, self._end_date, self._fred_api_key
                )
                if self._fred_api_key
                else treasury_model.get_real_yield_curve(start_date, self._end_date)
            )
        else:
            values = (
                fred_model.get_breakeven_inflation_expectations(
                    start_date, self._end_date, self._fred_api_key
                )
                if self._fred_api_key
                else treasury_model.get_breakeven_inflation_expectations(
                    start_date, self._end_date
                )
            )

        return values.rename(
            columns=lambda column: (
                "5Y5Y"
                if column == "5 Year, 5 Year Forward"
                else column.replace(" Year", "Y")
            )
        )

    def _get_bank_of_england_curve(self, curve: str, start_date: str) -> pd.DataFrame:
        """
        Retrieves a Bank of England spot curve at whole-year maturities.

        Args:
            curve (str): "real" or "inflation".
            start_date (str): The start date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: The curve, indexed by day with a column per whole-year maturity.
        """
        values = boe_model.get_spot_curve(curve, start_date, self._end_date)

        # The Bank of England estimates the curve in steps of half a year; the whole years
        # keep the columns comparable with the other countries.
        return values[
            [column for column in values.columns if float(column[:-1]).is_integer()]
        ]

    @handle_errors
    def get_real_yield_curve(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the real yield curve of a variety of countries: the yields of inflation-linked
        government bonds, which pay a return on top of inflation, by maturity. Together with
        the nominal curve (see `fixedincome.get_government_bond_yield_curve`) it gives the
        inflation the bond market expects, see `get_breakeven_inflation_expectations`.

        Every curve comes from the official source without an API key:
        - United States: the U.S. Department of the Treasury's real par yield curve of
          Treasury Inflation-Protected Securities (TIPS), 5, 7, 10, 20 and 30 years, from
          2003. When a FRED API key is set (see the `fred_api_key` parameter of the
          `Economics` class), the same figures come from FRED.
        - United Kingdom: the Bank of England's real spot curve of index-linked gilts, 3 to
          40 years, from 1985. Index-linked gilts pay the Retail Prices Index (RPI).
        - Germany: the real yields of the inflation-linked German federal securities the
          Bundesbank publishes, interpolated to constant maturities between 3 and 30 years
          from 2014. Germany has four to six of these bonds outstanding at a time, so only
          the maturities between its shortest (at least two years) and longest bond are
          filled. The bonds are indexed to euro area inflation (HICP excluding tobacco).

        Yields are returned as a decimal fraction per annum (0.0174 for 1.74%), with one
        column per country and maturity. Weekly and monthly periods take the curve on the
        last trading day of each period.

        See definition: https://fred.stlouisfed.org/series/DFII10

        Also known as: TIPS yield curve, real Treasury yield curve, index-linked gilt curve,
        Bund linker real yields, inflation-linked bond yields.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "United States", "United Kingdom" and "Germany". Defaults to None, which
                retrieves every country.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The real yields as decimals, indexed by date with a column per
            country and maturity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        real_yield_curve = economics.get_real_yield_curve(period='monthly')

        real_yield_curve.xs('10Y', axis=1, level='Maturity')
        ```

        Which returns:

        |         |   United States |   United Kingdom |   Germany |
        |:--------|----------------:|-----------------:|----------:|
        | 2026-04 |          0.0194 |           0.0154 |    0.0075 |
        | 2026-05 |          0.0207 |           0.0155 |    0.0084 |
        | 2026-06 |          0.022  |           0.0172 |    0.0095 |
        | 2026-07 |          0.0247 |           0.0188 |    0.0106 |
        | 2026-08 |          0.0244 |           0.0184 |    0.0108 |
        | 2026-09 |          0.0293 |           0.02   |    0.0132 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "real yield curve"
        )

        real_yield_curve = self._collect_country_curves(
            {
                "United States": lambda start: self._get_united_states_inflation_curve(
                    "real", start
                ),
                "United Kingdom": lambda start: self._get_bank_of_england_curve(
                    "real", start
                ),
                "Germany": lambda start: bundesbank_model.get_real_yield_curve(
                    start, self._end_date
                ),
            },
            countries,
            period,
            "real yield curve",
        )

        return finalize_dataset(
            dataset=real_yield_curve,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_breakeven_inflation_expectations(
        self,
        countries: list[str] | str | None = None,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the market-implied (Q-measure) breakeven inflation of a variety of countries by
        maturity: the nominal government bond yield minus the real yield of an
        inflation-linked bond of the same maturity, which is the inflation rate at which
        both earn the same. It is the bond market's expectation of inflation, including an
        inflation risk premium. See `get_inflation_expectations` for the expectations of
        professional forecasters instead.

        Every curve comes from the official source without an API key:
        - United States: the U.S. Department of the Treasury's nominal minus real (TIPS) par
          yields, 5, 7, 10, 20 and 30 years from 2003, plus the 5-year, 5-year forward rate
          ("5Y5Y"), the average inflation expected over the five years starting five years
          from now, with the formulas FRED uses. When a FRED API key is set (see the
          `fred_api_key` parameter of the `Economics` class), the same figures come from
          FRED.
        - United Kingdom: the Bank of England's implied inflation spot curve, 3 to 40 years,
          from 1985. It is measured against the Retail Prices Index (RPI), which index-linked
          gilts pay and which has run above CPI inflation.
        - Germany: the nominal term structure of German federal securities at the remaining
          maturity of each inflation-linked federal security minus its real yield,
          interpolated to constant maturities between 3 and 30 years from 2014. The bonds are
          indexed to euro area inflation (HICP excluding tobacco), which makes this the
          euro area's market-implied inflation expectation, priced off its benchmark issuer,
          and a free proxy for euro area inflation swaps, which are licensed data. Only the
          maturities between the shortest (at least two years) and longest bond are filled.

        Rates are returned as a decimal fraction per annum (0.0221 for 2.21%), with one column
        per country and maturity. Weekly and monthly periods take the curve on the last
        trading day of each period.

        See definition: https://fred.stlouisfed.org/series/T10YIE

        Also known as: breakeven inflation rate, market-implied inflation expectations, UK
        implied inflation, euro area breakeven inflation, Bund linker breakeven.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "United States", "United Kingdom" and "Germany". Defaults to None, which
                retrieves every country.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The breakeven inflation as decimals, indexed by date with a column
            per country and maturity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        breakeven_inflation = economics.get_breakeven_inflation_expectations(period='monthly')

        breakeven_inflation.xs('10Y', axis=1, level='Maturity')
        ```

        Which returns:

        |         |   United States |   United Kingdom |   Germany |
        |:--------|----------------:|-----------------:|----------:|
        | 2026-04 |          0.0246 |           0.0352 |    0.0234 |
        | 2026-05 |          0.0238 |           0.0331 |    0.0211 |
        | 2026-06 |          0.0224 |           0.0311 |    0.019  |
        | 2026-07 |          0.0228 |           0.0322 |    0.0208 |
        | 2026-08 |          0.0231 |           0.033  |    0.0221 |
        | 2026-09 |          0.0236 |           0.0341 |    0.0224 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "breakeven inflation"
        )

        breakeven_inflation = self._collect_country_curves(
            {
                "United States": lambda start: self._get_united_states_inflation_curve(
                    "breakeven", start
                ),
                "United Kingdom": lambda start: self._get_bank_of_england_curve(
                    "inflation", start
                ),
                "Germany": lambda start: bundesbank_model.get_breakeven_inflation(
                    start, self._end_date
                ),
            },
            countries,
            period,
            "breakeven inflation",
        )

        return finalize_dataset(
            dataset=breakeven_inflation,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_inflation_expectations(
        self,
        countries: list[str] | str | None = None,
        period: str = "monthly",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the inflation professional forecasters expect over the longer term, from surveys.
        Where `get_breakeven_inflation_expectations` is the bond market's expectation,
        priced daily and including risk premia, this is the forecasters' own expectation,
        with no inflation risk or liquidity premium in it and a longer history.

        Every series comes from the official source without an API key:
        - Germany: the inflation Consensus Economics' forecasters expect over 5 and 10 years,
          monthly from 1989. The Bundesbank publishes the expected real interest rate, the
          yield on debt securities outstanding issued by German residents with a residual
          maturity of 5 to 6 (or 9 to 10) years minus the expected inflation, so adding that
          yield back gives the expected inflation.
        - Euro Area: the average longer-term (five years ahead) HICP inflation forecast of the
          ECB's quarterly Survey of Professional Forecasters, from 1999. With
          period="monthly" each survey round is shown from the first month of its quarter
          until the next round.

        Rates are returned as a decimal fraction (0.0223 for 2.23%), with one column per
        country and horizon.

        See definition: https://www.ecb.europa.eu/stats/ecb_surveys/survey_of_professional_forecasters/html/index.en.html

        Also known as: survey-based inflation expectations, long-term inflation expectations,
        Consensus Economics inflation forecasts, Survey of Professional Forecasters.

        Args:
            countries (list[str] | str | None, optional): The countries to retrieve, from
                "Germany" and "Euro Area". Defaults to None, which retrieves every country.
            period (str, optional): Whether to return the monthly or quarterly data. Quarterly
                data takes the last month of each quarter. Defaults to "monthly".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The expected inflation as decimals, indexed by month or quarter with
            a column per country and horizon.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        economics.get_inflation_expectations().xs('5Y', axis=1, level='Maturity')
        ```

        Which returns:

        |         |   Germany |   Euro Area |
        |:--------|----------:|------------:|
        | 2026-04 |    0.0218 |      0.0203 |
        | 2026-05 |    0.0216 |      0.0203 |
        | 2026-06 |    0.0212 |      0.0203 |
        | 2026-07 |    0.0218 |      0.0204 |
        | 2026-08 |    0.0219 |      0.0204 |
        | 2026-09 |    0.0223 |      0.0204 |
        """
        period = validate_period(
            period, ["monthly", "quarterly"], "inflation expectations"
        )
        start_date = buffered_start_date(self._start_date, "monthly")

        def euro_area() -> pd.DataFrame:
            survey = ecb_model.get_survey_inflation_expectations(
                start_date, self._end_date
            ).rename(columns={"Euro Area": "5Y"})

            if survey.empty or period == "quarterly":
                return survey

            # A survey round holds until the next one, a quarter later.
            survey.index = survey.index.asfreq("M", how="start")
            months = pd.period_range(survey.index[0], survey.index[-1] + 2, freq="M")

            return survey.reindex(months).ffill(limit=2)

        def germany() -> pd.DataFrame:
            expectations = bundesbank_model.get_survey_inflation_expectations(
                start_date, self._end_date
            )

            if expectations.empty or period == "monthly":
                return expectations

            return expectations.groupby(expectations.index.asfreq("Q")).last()

        sources = {"Germany": germany, "Euro Area": euro_area}

        if countries is not None and not isinstance(countries, str | list | tuple):
            raise TypeError(
                "The countries must be a country name or a list of country names, such as "
                f"'Germany' or ['Germany', 'Euro Area'], not a {type(countries).__name__} ({countries!r})."
            )

        requested = (
            list(sources)
            if countries is None
            else [countries] if isinstance(countries, str) else list(countries)
        )

        if unavailable := [country for country in requested if country not in sources]:
            logger.warning(
                "Inflation expectations are not available for %s. They cover %s.",
                ", ".join(unavailable),
                ", ".join(sources),
            )

        expectations = {
            country: values
            for country in requested
            if country in sources and not (values := sources[country]()).empty
        }

        if not expectations:
            return pd.DataFrame()

        inflation_expectations = pd.concat(expectations, axis=1).sort_index()
        inflation_expectations.index.name = None
        inflation_expectations.columns.names = ["Country", "Maturity"]

        return finalize_dataset(
            dataset=inflation_expectations,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_implied_volatility(
        self,
        markets: list[str] | str | None = None,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the implied volatility the option markets price, by market and time to expiry:
        the volatility indices of Cboe, STOXX and ICE. Implied volatility is the market's
        expectation of how much prices will move, and its term structure (how it rises or
        falls with the time to expiry) is the market-consistent target for calibrating the
        volatility of equity, rate and commodity models.

        The markets and their maturities:
        - "US Equity": S&P 500 options, the Cboe VIX family: 9 days (VIX9D, from 2011), 1
          month (VIX, from 1990), 3 months (VIX3M), 6 months (VIX6M) and 1 year (VIX1Y, from
          2007).
        - "Euro Area Equity": EURO STOXX 50 options, the VSTOXX sub-indices from 1 to 24
          months, from 1999.
        - "Emerging Markets Equity": options on the MSCI Emerging Markets ETF (Cboe VXEEM),
          1 month, from 2011.
        - "US Equity Volatility": the volatility of the VIX itself (Cboe VVIX), 1 month, from
          2006.
        - "US Treasury Bonds": options on the 20+ year Treasury bond ETF (Cboe VXTLT), the
          price volatility of long Treasuries, 1 month, from 2004.
        - "Crude Oil" and "Gold": options on the oil and gold ETFs (Cboe OVX and GVZ), 1
          month, from 2009.
        - "US Treasury Rates": the ICE BofA MOVE index, the normal (basis point) volatility of
          Treasury yields implied by 1-month options on 2, 5, 10 and 30-year Treasuries, from
          2002. Requires a FinancialModelingPrep API key (the api_key of the Economics class).
        - "Japan Government Bonds": the S&P/JPX JGB VIX, from options on JGB futures, 1
          month, from 2015. Requires a FinancialModelingPrep API key.

        The volatility is returned as a decimal fraction per annum (0.155 for a VIX of 15.5),
        and the MOVE as a decimal yield change per annum (0.0105 for a MOVE of 105 basis
        points), with one column per market and maturity. Weekly and monthly periods take the
        value on the last trading day of each period. The Cboe SKEW index is not included
        since it is not a volatility.

        See definition: https://www.cboe.com/tradable_products/vix/

        Also known as: VIX, VSTOXX, MOVE index, volatility term structure, implied
        volatility index, fear index.

        Args:
            markets (list[str] | str | None, optional): The markets to retrieve, from those
                listed above. Defaults to None, which retrieves every market that needs no
                API key, and the MOVE and JGB VIX as well when an API key is set.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The implied volatility as decimals, indexed by date with a column per
            market and maturity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        implied_volatility = economics.get_implied_volatility(
            markets=['US Equity', 'Euro Area Equity'], period='monthly'
        )

        implied_volatility['US Equity']
        ```

        Which returns:

        |         |     9D |     1M |     3M |     6M |     1Y |
        |:--------|-------:|-------:|-------:|-------:|-------:|
        | 2026-04 | 0.1437 | 0.1689 | 0.2008 | 0.2261 | 0.2365 |
        | 2026-05 | 0.1259 | 0.1532 | 0.1866 | 0.216  | 0.2306 |
        | 2026-06 | 0.1373 | 0.1645 | 0.19   | 0.215  | 0.2303 |
        | 2026-07 | 0.1305 | 0.1599 | 0.1902 | 0.2134 | 0.2294 |
        | 2026-08 | 0.1234 | 0.1492 | 0.1753 | 0.2017 | 0.2187 |
        | 2026-09 | 0.142  | 0.1634 | 0.1837 | 0.2035 | 0.218  |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "implied volatility"
        )

        def cboe(maturities: dict[str, str]) -> Callable[[str], pd.DataFrame]:
            return lambda start: pd.DataFrame(
                {
                    maturity: cboe_model.get_index(symbol)
                    for maturity, symbol in maturities.items()
                }
            )

        def fmp(symbol: str, scale: float) -> Callable[[str], pd.DataFrame]:
            return lambda start: (
                economics_fmp_model.get_index_history(
                    symbol, self._api_key, start, self._end_date
                )
                * scale
            ).to_frame("1M")

        sources: dict[str, Callable[[str], pd.DataFrame]] = {
            "US Equity": cboe(
                {
                    "9D": "VIX9D",
                    "1M": "VIX",
                    "3M": "VIX3M",
                    "6M": "VIX6M",
                    "1Y": "VIX1Y",
                }
            ),
            "Euro Area Equity": lambda start: stoxx_model.get_vstoxx_term_structure(),
            "Emerging Markets Equity": cboe({"1M": "VXEEM"}),
            "US Equity Volatility": cboe({"1M": "VVIX"}),
            "US Treasury Bonds": cboe({"1M": "VXTLT"}),
            "Crude Oil": cboe({"1M": "OVX"}),
            "Gold": cboe({"1M": "GVZ"}),
        }
        keyed: dict[str, Callable[[str], pd.DataFrame]] = {
            # The MOVE is in basis points and the JGB VIX in percent; both are divided by
            # 100 below with the other indices, so the MOVE is first divided by another 100.
            "US Treasury Rates": fmp("^MOVE", 0.01),
            "Japan Government Bonds": fmp("^SPJGBV", 1),
        }

        if markets is None and self._api_key:
            sources.update(keyed)
        elif markets is not None:
            requested = [markets] if isinstance(markets, str) else list(markets)
            if missing_key := [
                market for market in requested if market in keyed and not self._api_key
            ]:
                logger.warning(
                    "%s requires a FinancialModelingPrep API key, passed with the api_key "
                    "parameter of the Economics class.",
                    ", ".join(missing_key),
                )
            sources.update(
                {
                    market: source
                    for market, source in keyed.items()
                    if self._api_key and market in requested
                }
            )
            sources.update(
                {
                    market: lambda start: pd.DataFrame()
                    for market in keyed
                    if market in requested and market not in sources
                }
            )

        implied_volatility = self._collect_country_curves(
            sources, markets, period, "implied volatility"
        )

        if not implied_volatility.empty:
            # The indices are quoted in percent.
            implied_volatility = implied_volatility / 100
            implied_volatility.columns.names = ["Market", "Maturity"]

        return finalize_dataset(
            dataset=implied_volatility,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_stress_test_scenario(
        self,
        scenario: str = "adverse",
        authority: str = "federal_reserve",
        countries: list[str] | str | None = None,
        year: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the scenarios regulators publish for their stress tests: documented paths of
        the economy and financial markets under a baseline and an adverse scenario, which
        serve as calibration and validation targets for scenario generators and as the
        stresses banks and insurers are tested against.

        Two authorities are available, both without an API key:
        - "federal_reserve": the Federal Reserve's supervisory stress test, quarterly over
          13 quarters. For the United States: real and nominal GDP and disposable income
          growth, unemployment, CPI inflation, the 3-month, 5-year and 10-year Treasury
          rates, the BBB corporate yield, the mortgage and prime rate, the Dow Jones Total
          Stock Market Index, house and commercial real estate price indices and the VIX;
          for the euro area, developing Asia, Japan and the United Kingdom: GDP growth,
          inflation and the dollar exchange rate. scenario="historic" returns the history
          of the same variables from 1976. year selects the year of the test (published in
          February); by default the latest.
        - "esrb": the macro-financial scenario the European Systemic Risk Board designs for
          the EU-wide bank stress test of the EBA, yearly over three years: GDP growth,
          inflation, unemployment, residential and commercial real estate prices and
          long-term rates for every EU member and the main other economies, stock prices by
          region, commodity prices, iTraxx credit spreads and euro exchange rates.
          scenario="historic" returns the starting point. Always the latest exercise.

        Growth rates, inflation, interest rates, spreads and the VIX are decimal fractions
        (0.054 for 5.4%); price index levels, the stock market index and exchange rates are
        returned as published. The columns are per country (or region, index, commodity or
        currency pair) and variable.

        See definition: https://www.federalreserve.gov/supervisionreg/dfast-archive.htm

        Also known as: supervisory scenarios, CCAR, DFAST, EBA stress test, adverse
        scenario, severely adverse scenario.

        Args:
            scenario (str, optional): "baseline", "adverse" (the Federal Reserve's severely
                adverse scenario) or "historic". Defaults to "adverse".
            authority (str, optional): "federal_reserve" or "esrb". Defaults to
                "federal_reserve".
            countries (list[str] | str | None, optional): The countries (or regions) to
                include. Defaults to None, which includes all of them.
            year (int | None, optional): The year of the Federal Reserve's stress test.
                Defaults to None, which takes the latest.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The scenario, indexed by quarter or year with a column per country
            and variable.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(end_date="2025-12-31")

        stress_test = economics.get_stress_test_scenario(countries='United States')

        stress_test['United States'][
            ['Real GDP growth', 'Unemployment rate', '10-year Treasury yield', 'BBB corporate yield']
        ].head()
        ```

        Which returns:

        |        |   Real GDP growth |   Unemployment rate |   10-year Treasury yield |   BBB corporate yield |
        |:-------|------------------:|--------------------:|-------------------------:|----------------------:|
        | 2026Q1 |            -0.054 |               0.059 |                    0.031 |                 0.075 |
        | 2026Q2 |            -0.049 |               0.072 |                    0.027 |                 0.082 |
        | 2026Q3 |            -0.038 |               0.082 |                    0.024 |                 0.081 |
        | 2026Q4 |            -0.027 |               0.09  |                    0.023 |                 0.079 |
        | 2027Q1 |            -0.014 |               0.095 |                    0.023 |                 0.075 |
        """
        if scenario not in ("baseline", "adverse", "historic"):
            raise ValueError(
                f"The scenario must be 'baseline', 'adverse' or 'historic', not {scenario!r}."
            )

        if authority == "federal_reserve":
            stress_test = frb_model.get_supervisory_scenario(scenario, year)
        elif authority == "esrb":
            if year is not None:
                logger.info(
                    "The ESRB scenario is always the latest exercise, so year is not used."
                )
            stress_test = esrb_model.get_macro_financial_scenario(scenario)
        else:
            raise ValueError(
                f"The authority must be 'federal_reserve' or 'esrb', not {authority!r}."
            )

        if countries is not None and not stress_test.empty:
            requested = [countries] if isinstance(countries, str) else list(countries)
            available = stress_test.columns.get_level_values(0)

            if unavailable := [
                country for country in requested if country not in available
            ]:
                logger.warning(
                    "The %s scenario has no %s. It covers %s.",
                    authority,
                    ", ".join(unavailable),
                    ", ".join(dict.fromkeys(available)),
                )

            stress_test = stress_test.loc[:, available.isin(requested)]

        # A scenario runs into the future, so only the history is cut to the dates.
        historic = scenario == "historic"

        return finalize_dataset(
            dataset=stress_test,
            start_date=self._start_date if historic else None,
            end_date=self._end_date if historic else None,
            default_rounding=self._rounding,
            rounding=rounding,
            axis="rows",
            row_slice=historic,
            apply_slice=historic,
            dropna=True,
        )

    @handle_errors
    def get_carbon_price(
        self,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the price of emitting one tonne of CO2 under the EU Emissions Trading System
        (EU ETS): the clearing price of the primary auctions of EU emission allowances
        (EUAs) the European Energy Exchange holds for the European Union, Germany and Poland
        almost every trading day, daily from 2020. The carbon price is the main transition
        risk variable of climate scenarios; see get_climate_scenario for its projected paths.

        The price is the average of the day's auctions of general allowances, in euro per
        tonne of CO2. No API key is needed. Weekly and monthly periods take the price of the
        last auction of each period.

        See definition: https://www.eex.com/en/markets/environmental-markets/eu-ets-auctions

        Also known as: EU ETS price, EUA price, CO2 price, emission allowance price.

        Args:
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "daily".
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The price in euro per tonne of CO2, indexed by date with a "European
            Union" column.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        economics.get_carbon_price(period='monthly', rounding=2)
        ```

        Which returns:

        |         |   European Union |
        |:--------|-----------------:|
        | 2026-04 |            71.8  |
        | 2026-05 |            79.37 |
        | 2026-06 |            78.13 |
        | 2026-07 |            81.22 |
        | 2026-08 |            82.39 |
        | 2026-09 |            84.41 |
        """
        period = validate_period(period, ["daily", "weekly", "monthly"], "carbon price")
        carbon_price = resample_to_period(
            eex_model.get_carbon_price(
                buffered_start_date(self._start_date, period), self._end_date
            ),
            period,
        )
        if not carbon_price.empty:
            carbon_price.index.name = None

        return finalize_dataset(
            dataset=carbon_price,
            indicator_name="Carbon Price",
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            trailing=trailing,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_climate_scenario(
        self,
        variable: str = "long_term_interest_rate",
        scenario: str = "Net Zero 2050",
        model: str = "REMIND-MAgPIE",
        countries: list[str] | str | None = None,
        risk: str = "combined",
        deviation: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the path of an economic or financial variable in a climate scenario of the
        Network for Greening the Financial System (NGFS), the scenarios central banks and
        supervisors use for climate stress tests, yearly from 2022 to 2050 for around fifty
        countries and regions.

        The NGFS scenarios combine an integrated assessment model, which projects the energy
        system, emissions and the carbon price (model="REMIND-MAgPIE", "GCAM" or
        "MESSAGEix-GLOBIOM"), with NiGEM, the macroeconometric model of the National
        Institute of Economic and Social Research, which translates each scenario into GDP,
        inflation, interest rates, equity prices, exchange rates and energy prices. NiGEM
        separates the transition risk (the policies and technologies of the move to a
        low-carbon economy), the physical risk (the damage of climate change) and both
        together (risk="combined").

        The variables: "gdp" (2017 PPP US$ billion), "inflation", "long_term_interest_rate",
        "long_term_real_interest_rate", "policy_rate", "unemployment_rate", "equity_prices"
        (index, 2017 = 100), "effective_exchange_rate" (index), "domestic_demand", "exports",
        "imports" (2017 PPP US$ billion), "oil_price" (US$ per barrel), "gas_price" and
        "coal_price" (US$ per barrel of oil equivalent), and "carbon_price", which comes
        from the integrated assessment model itself: US$ of 2010 per tonne of CO2 for the
        world and the model's regions, every five years to 2100.

        By default the level is returned: the NiGEM baseline (a world without further
        climate change and policy, scenario="Baseline") with the scenario's deviation
        applied. With deviation=True the deviation from the baseline itself is returned:
        relative for levels (-0.034 for 3.4% lower GDP) and in decimal points for rates
        (0.0094 for 0.94 percentage points higher). Rates are decimal fractions (0.041 for
        4.1%).

        The data is the Phase 5 vintage (November 2024), hosted by IIASA and retrieved
        without an API key; the NGFS licence allows commercial use with attribution but
        not the redistribution of substantial parts, so it is retrieved when needed rather
        than stored with the Finance Toolkit.

        See definition: https://www.ngfs.net/ngfs-scenarios-portal/

        Also known as: NGFS scenarios, climate stress test scenarios, transition risk,
        physical risk, carbon price path.

        Args:
            variable (str, optional): The variable, from those listed above. Defaults to
                "long_term_interest_rate".
            scenario (str, optional): "Net Zero 2050", "Below 2°C", "Low demand", "Delayed
                transition", "Nationally Determined Contributions (NDCs)", "Current
                Policies", "Fragmented World" or "Baseline". Defaults to "Net Zero 2050".
            model (str, optional): The integrated assessment model behind the scenario:
                "REMIND-MAgPIE", "GCAM" or "MESSAGEix-GLOBIOM". Defaults to "REMIND-MAgPIE".
            countries (list[str] | str | None, optional): The countries or regions to
                include. Defaults to None, which includes all of them.
            risk (str, optional): "combined", "transition" or "physical". Defaults to
                "combined".
            deviation (bool, optional): Whether to return the deviation from the baseline
                instead of the level. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The path of the variable, indexed by year with a column per
            country or region.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(end_date="2025-12-31")

        climate_scenario = economics.get_climate_scenario(
            variable='long_term_interest_rate',
            scenario='Net Zero 2050',
            countries=['United States', 'Germany', 'United Kingdom', 'Japan'],
        )

        climate_scenario.loc[['2025', '2030', '2035', '2040', '2045', '2050']]
        ```

        Which returns:

        |      |   United States |   Germany |   United Kingdom |   Japan |
        |:-----|----------------:|----------:|-----------------:|--------:|
        | 2025 |          0.0492 |    0.0346 |           0.0435 |  0.0175 |
        | 2030 |          0.0428 |    0.041  |           0.041  |  0.0277 |
        | 2035 |          0.0378 |    0.0405 |           0.0393 |  0.0348 |
        | 2040 |          0.0349 |    0.0393 |           0.0405 |  0.0375 |
        | 2045 |          0.0341 |    0.0387 |           0.0413 |  0.0372 |
        | 2050 |          0.0341 |    0.0386 |           0.0412 |  0.0371 |
        """
        variables = [*ngfs_model.NIGEM_VARIABLES, "carbon_price"]
        if variable not in variables:
            raise ValueError(
                f"The variable must be one of {', '.join(map(repr, variables))}, not {variable!r}."
            )
        if scenario not in [*ngfs_model.SCENARIOS, "Baseline"]:
            raise ValueError(
                f"The scenario must be one of {', '.join(map(repr, ngfs_model.SCENARIOS))} or "
                f"'Baseline', not {scenario!r}."
            )
        if model not in ngfs_model.MODELS:
            raise ValueError(
                f"The model must be one of {', '.join(map(repr, ngfs_model.MODELS))}, not {model!r}."
            )
        if risk not in ngfs_model.RISKS:
            raise ValueError(
                f"The risk must be one of {', '.join(map(repr, ngfs_model.RISKS))}, not {risk!r}."
            )

        if variable == "carbon_price":
            if deviation:
                logger.info(
                    "The carbon price is projected by the integrated assessment model, not "
                    "relative to a baseline, so deviation is not used."
                )
            climate_scenario = ngfs_model.get_carbon_price(scenario, model)
        else:
            climate_scenario = ngfs_model.get_nigem_scenario(
                variable, scenario, model, risk, deviation
            )

        if countries is not None and not climate_scenario.empty:
            requested = [countries] if isinstance(countries, str) else list(countries)
            if unavailable := [
                country
                for country in requested
                if country not in climate_scenario.columns
            ]:
                logger.warning(
                    "The NGFS %s has no %s. It covers %s.",
                    variable,
                    ", ".join(unavailable),
                    ", ".join(climate_scenario.columns),
                )
            climate_scenario = climate_scenario[
                [
                    country
                    for country in requested
                    if country in climate_scenario.columns
                ]
            ]

        # A scenario runs into the future, so it is not cut at the dates.
        return finalize_dataset(
            dataset=climate_scenario,
            start_date=None,
            end_date=None,
            default_rounding=self._rounding,
            rounding=rounding,
            apply_slice=False,
            dropna=True,
        )

    @handle_errors
    def get_long_run_asset_returns(
        self,
        asset_class: str = "equity",
        countries: list[str] | str | None = None,
        real: bool = False,
        accept_licence: bool = False,
        rolling: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the yearly total returns of equity, housing, government bonds and bills for 18
        advanced economies from 1870 to 2020, from the Jordà-Schularick-Taylor Macrohistory
        Database (Jordà, Knoll, Kuvshinov, Schularick and Taylor, 2019, "The Rate of Return on
        Everything, 1870-2015", Quarterly Journal of Economics). A century and a half of
        returns across wars, depressions and inflations is what calibrates the long-run
        trend, the equity and housing risk premia and the cycles of scenario generators for
        50 to 100-year horizons.

        The asset classes: "equity" (dividends reinvested), "housing" (rental income
        included), "bonds" (long-term government bonds, coupons included), "bills"
        (short-term government bills), "risky" (a wealth-weighted mix of equity and housing),
        "safe" (a mix of bonds and bills) and "wealth" (all of them weighted by their share of
        national wealth). The countries: Australia, Belgium, Canada, Denmark, Finland,
        France, Germany, Ireland, Italy, Japan, the Netherlands, Norway, Portugal, Spain,
        Sweden, Switzerland, the United Kingdom and the United States, with gaps in the war
        years.

        Returns are decimal fractions (0.08 for 8%), in local currency, and with real=True
        deflated by consumer price inflation.

        The database is licensed under CC BY-NC-SA 4.0: it may be used for non-commercial
        purposes with attribution, and its authors expressly exclude commercial data
        providers. Pass accept_licence=True to confirm the use is non-commercial; the data
        is retrieved from macrohistory.net when needed and is not distributed with the
        Finance Toolkit. No API key is needed.

        See definition: https://www.macrohistory.net/database/

        Also known as: rate of return on everything, long-run returns, JST macrohistory,
        historical equity premium.

        Args:
            asset_class (str, optional): "equity", "housing", "bonds", "bills", "risky",
                "safe" or "wealth". Defaults to "equity".
            countries (list[str] | str | None, optional): The countries to include. Defaults
                to None, which includes all of them.
            real (bool, optional): Whether to deflate the returns by consumer price
                inflation. Defaults to False.
            accept_licence (bool, optional): Confirms that the data is used for
                non-commercial purposes, as its licence (CC BY-NC-SA 4.0) requires. Defaults
                to False, which returns nothing.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The returns as decimals, indexed by year with a column per country.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2011-01-01', end_date='2020-12-31')

        economics.get_long_run_asset_returns(
            asset_class='equity',
            countries=['United States', 'United Kingdom', 'Germany', 'Japan'],
            real=True,
            accept_licence=True,
        )
        ```

        Which returns:

        |      |   United States |   United Kingdom |   Germany |   Japan |
        |:-----|----------------:|-----------------:|----------:|--------:|
        | 2011 |         -0.0085 |          -0.0755 |   -0.1688 | -0.0456 |
        | 2012 |          0.1448 |           0.0906 |    0.2654 | -0.0431 |
        | 2013 |          0.2767 |           0.1752 |    0.2475 |  0.7158 |
        | 2014 |          0.1389 |          -0.0026 |    0.0231 |  0.0742 |
        | 2015 |          0.0193 |           0.0106 |    0.1058 |  0.1029 |
        | 2016 |          0.0233 |           0.1465 |    0.0529 |  0.0063 |
        | 2017 |          0.1679 |           0.1107 |    0.1442 |  0.2135 |
        | 2018 |          0.1149 |          -0.1176 |   -0.1843 | -0.16   |
        | 2019 |          0.0624 |           0.1629 |    0.2113 |  0.1793 |
        | 2020 |          0.1115 |          -0.1025 |    0.0719 |  0.0786 |
        """
        if not accept_licence:
            raise ValueError(
                "The Jordà-Schularick-Taylor Macrohistory Database is licensed under CC "
                "BY-NC-SA 4.0, for non-commercial use with attribution. Pass "
                "accept_licence=True to confirm the use is non-commercial."
            )
        if asset_class not in macrohistory_model.ASSET_CLASSES:
            raise ValueError(
                f"The asset_class must be one of "
                f"{', '.join(map(repr, macrohistory_model.ASSET_CLASSES))}, not {asset_class!r}."
            )

        asset_returns = macrohistory_model.get_asset_returns(asset_class, real)

        return finalize_dataset(
            dataset=asset_returns,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Long Run Asset Returns",
            countries=countries,
            rolling=rolling,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_millennium_of_macroeconomic_data(
        self,
        series: list[str] | str | None = None,
        rolling: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the headline annual series of "A millennium of macroeconomic data for the UK",
        the Bank of England's research dataset of British economic history (Thomas and
        Dimsdale, version 3.1): around eighty series of output, population, employment,
        prices, wages, interest rates, asset prices, exchange rates, money, credit and public
        finances, from as early as 1086 (GDP and population), 1209 (consumer prices), 1694
        (Bank Rate) and 1700 (share prices) to 2016. It is the longest consistent record of
        interest rates, inflation and asset prices of any economy, for calibrating the very
        long-run behaviour of scenario generators.

        The series keep the names of the dataset, such as "Bank Rate", "Consols / long-term
        government bond yields", "Corporate bond yields", "Share prices", "Consumer price
        index", "Consumer price inflation", "House price index" and "$/£ exchange rate"; a
        second column of a series, such as its growth rate or its share of GDP, carries that
        unit in brackets. Rates, growth rates and shares of GDP are decimal fractions (0.05
        for 5%); levels, indices and exchange rates are returned as published.

        The dataset is not updated anymore. No API key is needed; the workbook is large
        (around 28 MB), so it is cached for a year.

        See definition: https://www.bankofengland.co.uk/statistics/research-datasets

        Also known as: three centuries of macroeconomic data, BoE millennium dataset, long
        run UK data.

        Args:
            series (list[str] | str | None, optional): The series to include, matched on
                (part of) their name without regard to case, e.g. "Bank Rate" or ["consols",
                "share prices"]. Defaults to None, which includes all of them.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The series, indexed by year with a column per series.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='1700-01-01', end_date='2016-12-31')

        millennium = economics.get_millennium_of_macroeconomic_data(
            series=['Bank Rate', 'Consols', 'Consumer price inflation', 'Share prices']
        )

        millennium.loc[['1720', '1815', '1914', '1974', '2008']]
        ```

        Which returns:

        |      |   Bank Rate |   Consols / long-term government bond yields |   Consumer price inflation |   Share prices |
        |:-----|------------:|---------------------------------------------:|---------------------------:|---------------:|
        | 1720 |       0.05  |                                       0.04   |                     0.0515 |         2.416  |
        | 1815 |       0.05  |                                       0.0504 |                    -0.144  |         1.0776 |
        | 1914 |       0.05  |                                       0.0348 |                     0.0255 |        42.7811 |
        | 1974 |       0.115 |                                       0.1517 |                     0.1573 |        65.26   |
        | 2008 |       0.02  |                                       0.0459 |                     0.036  |      2128.8    |
        """
        millennium = boe_model.get_millennium_data()

        if series is not None and not millennium.empty:
            terms = [series] if isinstance(series, str) else list(series)
            selected = []

            for term in terms:
                # An exact name takes only that series; otherwise every series whose name
                # contains the term, except their second columns (growth rates and shares).
                matches = (
                    [term]
                    if term in millennium.columns
                    else [
                        column
                        for column in millennium.columns
                        if term.lower() in column.lower() and not column.endswith(")")
                    ]
                )
                if not matches:
                    logger.warning(
                        "The millennium dataset has no series named like %r.", term
                    )
                selected += [column for column in matches if column not in selected]

            millennium = millennium[selected]

        return finalize_dataset(
            dataset=millennium,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            growth=growth,
            lag=lag,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_scenario_set(
        self,
        authority: str = "dnb",
        measure: str = "real_world",
        variables: list[str] | str | None = None,
        quantiles: list[float] | None = None,
        scenarios: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get a regulator's published scenario set: the thousands of simulated paths of
        interest rates, equity returns and inflation that pension funds or insurers must
        use, generated by a documented and reviewed economic scenario generator. It is a
        reference to calibrate and validate an own scenario generator against, both for the
        real-world distribution and for the market-consistent one.

        Available is the scenario set of De Nederlandsche Bank (authority="dnb"), which
        Dutch pension funds use for their feasibility test and the transition to the new
        pension contract: 20,000 scenarios over a horizon of 100 years, generated every
        quarter with the model the Commissie Parameters 2022 specified, under the
        real-world measure (measure="real_world", the P-set) and the market-consistent
        measure (measure="market_consistent", the Q-set, arbitrage-free and calibrated to
        market prices including swaptions).

        The variables, as decimals per year of the horizon (year 0 is the end of the
        quarter the set is for): the annually compounded nominal zero rates "Nominal rate
        1Y", "5Y", "10Y", "20Y" and "30Y" and the euro area "Real rate EU 10Y", derived from
        the model's term structure, and the annual "Equity return", "Inflation EU" and
        "Inflation NL" (from year 1).

        By default the distribution per year is returned: the quantiles of the 20,000
        scenarios and their mean. With scenarios=True every scenario is returned instead
        (20,000 columns per variable). The set is a large workbook (around 180 MB) that
        takes a minute or two to download and read the first time; it is cached for 30
        days. No API key is needed.

        See definition: https://www.dnb.nl/voor-de-sector/open-boek-toezicht/sectoren/pensioenfondsen/

        Also known as: DNB scenarioset, uniform scenario set, CP2022 scenarios, P-set,
        Q-set, regulatory economic scenario generator.

        Args:
            authority (str, optional): The regulator: "dnb". Defaults to "dnb".
            measure (str, optional): "real_world" or "market_consistent". Defaults to
                "real_world".
            variables (list[str] | str | None, optional): The variables to include, from
                those listed above. Defaults to None, which includes all of them.
            quantiles (list[float] | None, optional): The quantiles of the distribution
                per year. Defaults to None, which is [0.01, 0.05, 0.25, 0.5, 0.75, 0.95,
                0.99].
            scenarios (bool, optional): Whether to return every scenario instead of the
                distribution. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: Indexed by the year of the horizon, with a column per variable and
            quantile (or "Mean"), or per variable and scenario with scenarios=True.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(end_date="2025-12-31")

        scenario_set = economics.get_scenario_set(
            measure='real_world', variables='Nominal rate 10Y', quantiles=[0.05, 0.5, 0.95]
        )

        scenario_set['Nominal rate 10Y'].loc[[0, 1, 5, 10, 30, 100]]
        ```

        Which returns:

        |   Horizon |      5% |    50% |    95% |   Mean |
        |----------:|--------:|-------:|-------:|-------:|
        |         0 |  0.0293 | 0.0293 | 0.0293 | 0.0293 |
        |         1 |  0.0203 | 0.0281 | 0.0363 | 0.0282 |
        |         5 |  0.0139 | 0.028  | 0.0435 | 0.0283 |
        |        10 |  0.0069 | 0.0235 | 0.0416 | 0.0238 |
        |        30 | -0.0001 | 0.0178 | 0.037  | 0.018  |
        |       100 |  0.0024 | 0.0197 | 0.039  | 0.0201 |
        """
        if authority != "dnb":
            raise ValueError(f"The authority must be 'dnb', not {authority!r}.")
        if measure not in dnb_model.MEASURES:
            raise ValueError(
                f"The measure must be one of {', '.join(map(repr, dnb_model.MEASURES))}, "
                f"not {measure!r}."
            )

        quantiles = (
            [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            if quantiles is None
            else quantiles
        )
        if any(not 0 <= quantile <= 1 for quantile in quantiles):
            raise ValueError(
                f"The quantiles must lie between 0 and 1, not {quantiles}."
            )

        scenario_set = dnb_model.get_scenario_set(measure)

        if scenario_set.empty:
            return scenario_set

        available = list(dict.fromkeys(scenario_set.columns.get_level_values(0)))
        requested = (
            available
            if variables is None
            else [variables] if isinstance(variables, str) else list(variables)
        )
        if unavailable := [
            variable for variable in requested if variable not in available
        ]:
            raise ValueError(
                f"The scenario set has no {', '.join(unavailable)}. It has "
                f"{', '.join(available)}."
            )

        scenario_set = scenario_set[requested]

        if not scenarios:
            distribution = {}

            # Returns and inflation have no value in year 0, which numpy warns about.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for variable in requested:
                    values = scenario_set[variable].to_numpy(dtype=float)
                    for quantile in quantiles:
                        distribution[(variable, f"{quantile:.0%}")] = np.nanquantile(
                            values, quantile, axis=1
                        )
                    distribution[(variable, "Mean")] = np.nanmean(values, axis=1)

            scenario_set = pd.DataFrame(distribution, index=scenario_set.index)
            scenario_set.columns.names = ["Variable", "Statistic"]

        return finalize_dataset(
            dataset=scenario_set.astype(float),
            start_date=None,
            end_date=None,
            default_rounding=self._rounding,
            rounding=rounding,
            apply_slice=False,
        )

    @handle_errors
    def get_asset_class_proxies(
        self,
        asset_classes: list[str] | str | None = None,
        period: str = "monthly",
        returns: bool = True,
        rolling: int | None = None,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the total returns of listed funds that track asset classes without a public
        return index: private equity, infrastructure, hedge funds, private credit, real
        estate and investment grade and high yield credit. Their returns approximate those
        of the asset classes for calibrating a scenario generator, with two caveats: listed
        prices are marked to market daily, so they are more volatile and more correlated
        with equities than the appraisal-based returns of private funds, and they include
        the funds' fees.

        The asset classes and the funds that track them, with the start of their history:
        - "Private Equity": the Invesco Global Listed Private Equity ETF (PSP), from 2006.
        - "Infrastructure": the iShares Global Infrastructure ETF (IGF), from 2007.
        - "Hedge Funds": the IQ Hedge Multi-Strategy Tracker ETF (QAI), which replicates the
          returns of hedge fund indices, from 2009.
        - "Merger Arbitrage": the IQ Merger Arbitrage ETF (MNA), from 2009.
        - "Private Credit": the VanEck BDC Income ETF (BIZD), business development companies
          that lend to private companies, from 2013.
        - "Real Estate": the Vanguard Real Estate ETF (VNQ), US REITs, from 2004.
        - "Investment Grade Credit": the iShares iBoxx $ Investment Grade Corporate Bond ETF
          (LQD), from 2002.
        - "High Yield Credit": the iShares iBoxx $ High Yield Corporate Bond ETF (HYG), from
          2007.

        Returns are the change of the dividend-adjusted price over each period, as decimals,
        in US dollars; with returns=False the dividend-adjusted price itself is returned.
        The prices come from FinancialModelingPrep when an API key is set (the api_key of the
        Economics class) and from Yahoo Finance otherwise.

        Also known as: private market proxies, listed private equity, alternative asset
        returns, hedge fund replication.

        Args:
            asset_classes (list[str] | str | None, optional): The asset classes to include,
                from those listed above. Defaults to None, which includes all of them.
            period (str, optional): Whether to return the daily, weekly or monthly data.
                Defaults to "monthly".
            returns (bool, optional): Whether to return the total return per period instead
                of the dividend-adjusted price. Defaults to True.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The returns (or prices), indexed by date with a column per asset
            class.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        economics.get_asset_class_proxies(
            asset_classes=['Private Equity', 'Infrastructure', 'Hedge Funds', 'Private Credit']
        )
        ```

        Which returns:

        |         |   Private Equity |   Infrastructure |   Hedge Funds |   Private Credit |
        |:--------|-----------------:|-----------------:|--------------:|-----------------:|
        | 2026-04 |           0.0778 |           0.0228 |        0.046  |           0.0675 |
        | 2026-05 |           0.001  |          -0.0282 |        0.0202 |          -0.0395 |
        | 2026-06 |          -0.0688 |           0.0141 |        0.0052 |           0.0024 |
        | 2026-07 |           0.0711 |           0.0032 |       -0.0142 |          -0.0017 |
        | 2026-08 |           0.0734 |          -0.025  |        0.0083 |           0.0895 |
        | 2026-09 |          -0.1149 |          -0.055  |       -0.0093 |          -0.0459 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "asset class proxies"
        )

        requested = (
            list(ASSET_CLASS_PROXIES)
            if asset_classes is None
            else (
                [asset_classes]
                if isinstance(asset_classes, str)
                else list(asset_classes)
            )
        )
        if unavailable := [
            name for name in requested if name not in ASSET_CLASS_PROXIES
        ]:
            raise ValueError(
                f"There is no proxy for {', '.join(unavailable)}. The asset classes are "
                f"{', '.join(ASSET_CLASS_PROXIES)}."
            )

        tickers = [ASSET_CLASS_PROXIES[name] for name in requested]
        start_date = buffered_start_date(self._start_date, period)
        historical_data, _ = historical_model.get_historical_data(
            tickers,
            api_key=self._api_key or None,
            start=start_date,
            end=self._end_date,
            fill_nan=False,
            show_ticker_seperation=False,
            cache=self._cache,
        )

        if historical_data.empty or "Adj Close" not in historical_data.columns:
            return pd.DataFrame()

        prices = historical_data["Adj Close"]
        prices = prices[[ticker for ticker in tickers if ticker in prices.columns]]
        prices = prices.rename(
            columns={ticker: name for name, ticker in ASSET_CLASS_PROXIES.items()}
        )
        prices = resample_to_period(prices.dropna(how="all"), period)
        prices.columns.name = None

        # The return of a period runs from the last price of the previous period.
        proxies = prices.pct_change(fill_method=None) if returns else prices

        return finalize_dataset(
            dataset=proxies,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rolling=rolling,
            rounding=rounding,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_commodity_forward_curve(
        self,
        commodity: str,
        contracts: int = 12,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the forward/futures curve for a commodity from Yahoo Finance -- the
        historical daily closing price of each dated futures contract over the next
        `contracts` calendar months (e.g. Crude Oil's December 2026, January 2027,
        ... contracts), rather than a single flat continuous/spot price.

        This is what a Schwartz-Smith (2000) two-factor commodity price model needs
        to back out the convenience-yield term structure under the risk-neutral (Q)
        measure -- the curve's shape (contango or backwardation) at each point in
        time is exactly what a single spot price series cannot reveal.

        Not every commodity has a listed contract for every calendar month (grains
        in particular only trade specific delivery months), so months with no
        listed contract are silently skipped -- the number of columns returned can
        be fewer than `contracts`.

        Also known as: futures term structure, forward curve.

        Args:
            commodity (str): The commodity to retrieve the curve for. One of "Crude
                Oil", "Natural Gas", "Gold", "Silver", "Copper", "Corn", "Wheat" or
                "Soybeans".
            contracts (int, optional): The number of sequential monthly contracts
                ahead of today to attempt to fetch. Defaults to 12.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Raises:
            ValueError: If `commodity` is not one of the supported names.

        Returns:
            pd.DataFrame: A DataFrame indexed by date, with one column per contract
            labeled by its delivery month (e.g. "2026-12"), containing that
            contract's daily closing price over its trading life. Columns are NaN
            outside the date range the contract actually traded in.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-01-01', end_date='2026-08-01')

        economics.get_commodity_forward_curve("Crude Oil", contracts=6)
        ```

        Which returns:

        | Date       |   2026-11 |   2026-12 |   2027-01 |   2027-02 |   2027-03 |
        |:-----------|----------:|----------:|----------:|----------:|----------:|
        | 2026-07-27 |     78.17 |     76.53 |     75.31 |     74.34 |     73.56 |
        | 2026-07-28 |     75.33 |     73.85 |     72.74 |     71.86 |     71.16 |
        | 2026-07-29 |     79.68 |     77.74 |     76.28 |     75.13 |     74.22 |
        | 2026-07-30 |     78.19 |     76.12 |     74.65 |     73.56 |     72.72 |
        | 2026-07-31 |     78.65 |     76.44 |     74.88 |     73.74 |     72.88 |
        """
        commodity_forward_curve = yfinance_model.get_commodity_forward_curve(
            commodity, self._start_date, self._end_date, contracts
        )

        return finalize_dataset(
            dataset=commodity_forward_curve,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            axis="rows",
            row_slice=True,
        )

    @handle_errors
    def get_life_table(
        self,
        countries: list[str] | str | None = None,
        measure: str = "death_probability",
        sex: str = "total",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves the life tables Eurostat compiles yearly for the countries of the European
        Economic Area, by single year of age from 0 to 95 (the last age group being 95 and
        over), from 1960 for most countries. A life table describes the mortality of a
        population: the basis of every pension, life insurance and annuity calculation, and
        of mortality and longevity assumptions in scenario models.

        The measures are the probability of dying within the year (death_probability, qx),
        the probability of surviving it (survival_probability, px), the age-specific death
        rate (death_rate, mx), the number of survivors out of 100,000 births (survivors, lx),
        the number dying (deaths, dx), the person-years lived within the year (person_years,
        Lx) and above the age (total_person_years, Tx), and the life expectancy at the age
        (life_expectancy, ex). Probabilities and rates are fractions; the others are counts
        or years.

        No API key is needed. For countries outside Europe the UN World Population Prospects
        publish life tables for every country (bulk files at population.un.org/wpp), and the
        Human Mortality Database long national histories (with free registration).

        Also known as: mortality table, qx table, survival table, actuarial table.

        Args:
            countries (list[str] | str | None, optional): The countries to include, e.g.
                'Germany' or ['Germany', 'France']. Defaults to None, which retrieves every country.
            measure (str, optional): The life table measure, see above. Defaults to
                "death_probability".
            sex (str, optional): "total", "male" or "female". Defaults to "total".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: The measure, indexed by year with a column per country and age.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date="2025-12-31")

        life_table = economics.get_life_table(countries='Germany', sex='male')

        life_table['Germany'][[0, 30, 65, 80, 95]]
        ```

        Which returns:

        |      |      0 |     30 |     65 |     80 |   95 |
        |:-----|-------:|-------:|-------:|-------:|-----:|
        | 2020 | 0.0032 | 0.0005 | 0.0156 | 0.0575 |    1 |
        | 2021 | 0.0033 | 0.0006 | 0.0163 | 0.0559 |    1 |
        | 2022 | 0.0032 | 0.0005 | 0.016  | 0.0596 |    1 |
        | 2023 | 0.0033 | 0.0006 | 0.0153 | 0.0571 |    1 |
        | 2024 | 0.0035 | 0.0005 | 0.0151 | 0.0566 |    1 |
        """
        if measure not in eurostat_model.LIFE_TABLE_MEASURES:
            raise ValueError(
                f"The measure must be one of {', '.join(eurostat_model.LIFE_TABLE_MEASURES)}, "
                f"not {measure!r}."
            )
        if sex not in eurostat_model.LIFE_TABLE_SEXES:
            raise ValueError(
                f"The sex must be 'total', 'male' or 'female', not {sex!r}."
            )

        requested = [countries] if isinstance(countries, str) else countries
        codes = None
        if requested:
            names_to_codes = {
                name: code for code, name in eurostat_model.COUNTRY_CODES.items()
            }
            codes = [
                names_to_codes[name] for name in requested if name in names_to_codes
            ]

            if unknown := [name for name in requested if name not in names_to_codes]:
                logger.warning("No life table is available for %s.", ", ".join(unknown))
            if not codes:
                return pd.DataFrame()

        life_table = eurostat_model.get_life_table(
            measure, sex, self._start_date, self._end_date, codes
        )

        return finalize_dataset(
            dataset=life_table,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
            dropna=True,
        )

    @handle_errors
    def get_shiller_stock_market_data(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
    ):
        """
        Retrieves Robert Shiller's monthly US stock market data from 1871, the data behind
        "Irrational Exuberance" and the CAPE ratio: the S&P Composite price, dividends and
        earnings, the consumer price index, the long-term interest rate, the same in real
        (inflation-adjusted) terms, and the cyclically adjusted price-earnings ratio (CAPE),
        the price over the average of ten years of real earnings. With more than 150 years of
        history it is the standard source for long-run equity returns, valuations and their
        relation to subsequent returns.

        Prices are monthly averages of daily closes; dividends and earnings are trailing
        twelve-month totals, with the latest months not yet published. The long interest
        rate is a decimal fraction (0.0475 for 4.75%).

        Shiller, R. J. (2015). Irrational Exuberance (3rd ed.). Princeton University Press.

        No API key is needed.

        See definition: https://shillerdata.com/

        Also known as: Shiller data, CAPE, Shiller PE, long-run stock returns.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.

        Returns:
            pd.DataFrame: One column per measure, indexed by month.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-01-01')

        economics.get_shiller_stock_market_data()[['Price', 'Dividend', 'Earnings', 'CAPE']]
        ```

        Which returns:

        |         |   Price |   Dividend |   Earnings |    CAPE |
        |:--------|--------:|-----------:|-----------:|--------:|
        | 2026-01 | 6929.12 |    79.8018 |    247.664 | 39.6408 |
        | 2026-02 | 6893.81 |    80.0835 |    254.693 | 39.014  |
        | 2026-03 | 6654.42 |    80.3653 |    261.723 | 37.0316 |
        | 2026-04 | 6957.01 |    80.8113 |    272.945 | 38.1383 |
        | 2026-05 | 7412.55 |    81.2572 |    284.166 | 40.1014 |
        | 2026-06 | 7450.03 |    81.7032 |    295.388 | 40.1502 |
        | 2026-07 | 7481.34 |   nan      |    nan     | 40.0086 |
        | 2026-08 | 7711.32 |   nan      |    nan     | 40.9874 |
        | 2026-09 | 7669.41 |   nan      |    nan     | 40.5809 |
        | 2026-10 | 7722.72 |   nan      |    nan     | 40.7106 |
        """
        stock_market_data = shiller_model.get_stock_market_data()

        return finalize_dataset(
            dataset=stock_market_data,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
        )

    def _require_api_key(self) -> None:
        """Warns and raises when a FinancialModelingPrep method is used without a key."""
        if not self._api_key:
            logger.warning(
                "No FinancialModelingPrep API key found. The economic calendar and the market "
                "risk premium require a key to access, obtain one (with 15% off) at "
                "https://www.jeroenbouma.com/fmp and pass it via the api_key argument."
            )
            raise ValueError(
                "A FinancialModelingPrep API key is required for this data. Obtain one at "
                "https://www.jeroenbouma.com/fmp and pass it via the api_key argument."
            )

    @handle_errors
    def get_economic_calendar(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        countries: str | list[str] | None = None,
        currencies: str | list[str] | None = None,
        impact: str | list[str] | None = None,
        events: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Returns the scheduled releases of economic data for every country, such as
        inflation, employment, GDP and central bank decisions. Each release comes with
        its previous value, the consensus estimate and, once published, the actual value
        and the change. The estimate versus the actual value is what moves markets: a
        release far from the consensus tends to cause the largest reaction.

        The calendar covers well over a hundred countries, so it is best narrowed down:
        select a date range and filter by country, currency, impact and/or event. Filtering
        on events="PMI" gives the latest purchasing managers' indices with their consensus,
        figures that are otherwise only available as licensed data. The "Impact"
        column rates how much a release usually moves the market (Low, Medium or High).
        The "Unit" column gives the unit of each release. Releases quoted in percent (for
        example an inflation or unemployment rate) are converted to decimals, as is the
        "Change %" column, so an unemployment rate of 4.1% is returned as 0.041. Releases
        in other units, such as thousands of jobs or index points, keep their values, as do
        survey levels such as PMIs and confidence indices, which the calendar sometimes
        labels "%" as well.

        Without a start_date and end_date the dates this Economics instance was created
        with are used, and without those the last 90 days. The endpoint returns at most 90
        days per request, so a longer range is retrieved in 90-day windows. This requires a
        FinancialModelingPrep API key, passed via api_key when initializing.

        Also known as: economic calendar, macro calendar, economic releases.

        Args:
            start_date (str, optional): The start date of the releases. Defaults to None, which
                uses this instance's start_date when it was given.
            end_date (str, optional): The end date of the releases. Defaults to None, which uses
                this instance's end_date when it was given, otherwise today.
            countries (str | list[str], optional): The countries to keep, as names (e.g.
                "United States", "Germany", "Euro Area") or codes (e.g. "US", "DE", "EU").
                Defaults to None, which keeps every country.
            currencies (str | list[str], optional): The currencies to keep, e.g. "USD" or
                ["USD", "EUR"]. Defaults to None, which keeps every currency.
            impact (str | list[str], optional): The market impact to keep: "Low", "Medium"
                and/or "High", or "All" for every release. Defaults to None, which keeps every release.
            events (str | list[str], optional): Parts of event names to keep, matched without
                regard to case, e.g. "PMI" for every purchasing managers' index or ["CPI",
                "Unemployment"]. Defaults to None, which keeps every event.

        Returns:
            pd.DataFrame: The economic data releases, indexed by date.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(api_key="FINANCIAL_MODELING_PREP_KEY", end_date="2025-12-31")

        economic_calendar = economics.get_economic_calendar(
            start_date="2026-09-01",
            end_date="2026-09-30",
            countries=["United States", "Euro Area"],
            impact="High",
        )

        economic_calendar[["Country", "Event", "Previous", "Estimate", "Actual"]].head()
        ```

        Which returns:

        | Date                | Country       | Event                       |   Previous |   Estimate |   Actual |
        |:--------------------|:--------------|:----------------------------|-----------:|-----------:|---------:|
        | 2026-09-01 14:00:00 | United States | ISM Manufacturing PMI (Aug) |     55.6   |     55.2   |   54.6   |
        | 2026-09-01 14:00:00 | United States | JOLTs Job Openings (Jul)    |      7.182 |      7.3   |    7.271 |
        | 2026-09-03 14:00:00 | United States | ISM Services PMI (Aug)      |     54.1   |     54.3   |   55.4   |
        | 2026-09-04 12:30:00 | United States | Non Farm Payrolls (Aug)     |     21     |     56     |  162     |
        | 2026-09-04 12:30:00 | United States | Unemployment Rate (Aug)     |      0.041 |      0.041 |    0.041 |
        """
        self._require_api_key()

        return economics_fmp_model.get_economic_calendar(
            api_key=self._api_key,
            start_date=start_date or self._requested_start_date,
            end_date=end_date or self._requested_end_date,
            countries=countries,
            currencies=currencies,
            impact=impact,
            events=events,
        )

    @handle_errors
    def get_market_risk_premium(self) -> pd.DataFrame:
        """
        Returns the market risk premium per country: the extra return investors demand
        for holding equities over a risk-free investment. It consists of the country risk
        premium, which compensates for the additional risk of the country itself, and the
        total equity risk premium, which adds the premium of a mature market. The market
        risk premium is a key input of the Capital Asset Pricing Model (CAPM) and with
        that of the cost of equity and the WACC.

        The same data is available as Toolkit.get_market_risk_premium; both share one
        cached copy. This requires a FinancialModelingPrep API key, passed via api_key
        when initializing.

        Also known as: equity risk premium, country risk premium, MRP.

        Returns:
            pd.DataFrame: The country and total equity risk premium per country, as decimals
            (0.0446 for 4.46%).

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(api_key="FINANCIAL_MODELING_PREP_KEY", end_date="2025-12-31")

        market_risk_premium = economics.get_market_risk_premium()

        market_risk_premium.loc[["United States", "Germany", "Japan", "Brazil", "India"]]
        ```

        Which returns:

        | Country       | Continent     |   Country Risk Premium |   Total Equity Risk Premium |
        |:--------------|:--------------|-----------------------:|----------------------------:|
        | United States | North America |                 0.0023 |                      0.0446 |
        | Germany       | Europe        |                 0      |                      0.0423 |
        | Japan         | Asia          |                 0.0091 |                      0.0514 |
        | Brazil        | South America |                 0.0324 |                      0.0747 |
        | India         | Asia          |                 0.0285 |                      0.0708 |
        """
        self._require_api_key()

        # Published per country rather than per ticker, so it is one cache entry, the
        # same one Toolkit.get_market_risk_premium uses.
        if self._cache is not None:
            cached_premium = self._cache.get(
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="market_risk_premium",
                entity=policy_model.MARKET_RISK_PREMIUM_ENTITY,
            )
            if cached_premium is not None:
                return cached_premium

        market_risk_premium = fmp_model.get_market_risk_premium(api_key=self._api_key)

        if self._cache is not None and not market_risk_premium.empty:
            self._cache.set(
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="market_risk_premium",
                entity=policy_model.MARKET_RISK_PREMIUM_ENTITY,
                data=market_risk_premium,
            )

        return market_risk_premium
