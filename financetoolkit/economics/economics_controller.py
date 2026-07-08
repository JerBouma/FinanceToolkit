"""Economics Module"""

__docformat__ = "google"


import os
import re
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit.cache.cache_controller import Cache, set_active_cache
from financetoolkit.economics import (
    fred_model,
    fxmacrodata_model,
    gmdb_model,
    oecd_model,
    yfinance_model,
)
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import finalize_dataset

logger = get_logger()

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

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2010-01-01")

        cpi = economics.get_consumer_price_index()

        cpi.loc['2010':, ['United States', 'Netherlands', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Netherlands |    Japan |
        |:-----|----------------:|--------------:|---------:|
        | 2010 |         100     |       100     | 100      |
        | 2011 |         103.14  |       102.472 |  99.7226 |
        | 2012 |         105.278 |       105.359 |  99.6741 |
        | 2013 |         106.822 |       108.052 | 100.004  |
        | 2014 |         108.547 |       108.397 | 102.762  |
        | 2015 |         108.679 |       108.635 | 103.583  |
        | 2016 |         110.056 |       108.759 | 103.455  |
        | 2017 |         112.402 |       110.165 | 103.958  |
        | 2018 |         115.143 |       111.927 | 104.986  |
        | 2019 |         117.231 |       114.913 | 105.477  |
        | 2020 |         118.695 |       116.185 | 105.449  |
        | 2021 |         124.253 |       119.459 | 105.202  |
        | 2022 |         134.183 |       133.336 | 107.828  |
        | 2023 |         139.722 |       138.827 | 111.353  |
        | 2024 |         143.896 |       143.228 | 113.839  |
        | 2025 |         146.562 |       146.58  | 116.102  |
        """
        if start_date and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date) is None:
            raise ValueError(
                "Please input a valid start date (%Y-%m-%d) like '2010-01-01'"
            )
        if end_date and re.match(r"^\d{4}-\d{2}-\d{2}$", end_date) is None:
            raise ValueError(
                "Please input a valid end date (%Y-%m-%d) like '2020-01-01'"
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

        self._cache = cache

        # Published once here so the OECD and FRED free functions read it back.
        set_active_cache(cache)
        oecd_model.configure_oecd_cache(allow_stale_oecd_cache)

        self._gmdb_source: bool = gmdb_source
        self._gmbd_dataset: pd.DataFrame = (
            gmdb_model.collect_global_macro_database_dataset(cache=cache)
            if self._gmdb_source
            else pd.DataFrame()
        )
        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding
        self._fred_api_key: str = fred_api_key

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

    @handle_errors
    def get_fxmacrodata_release_calendar(
        self,
        currency: str = "usd",
        limit: int = 100,
        min_tier: int | None = None,
        api_key: str | None = None,
    ) -> pd.DataFrame:
        """
        Get the FXMacroData economic release calendar for a currency.

        FXMacroData provides point-in-time macroeconomic and central-bank
        release events that can be joined to price, factor, or portfolio
        data when analyzing event risk.

        Args:
            currency (str, optional): Three-letter FX currency code. Defaults to "usd".
            limit (int, optional): Maximum number of events to return. Defaults to 100.
            min_tier (int | None, optional): Optional maximum market tier to keep.
                For example, 2 keeps tier 1 and tier 2 market-moving events.
            api_key (str | None, optional): FXMacroData API key. Defaults to the
                FXMACRODATA_API_KEY environment variable when available.

        Returns:
            pd.DataFrame: FXMacroData release calendar indexed by release date.
        """
        release_calendar = fxmacrodata_model.get_release_calendar(
            currency=currency,
            limit=limit,
            min_tier=min_tier,
            api_key=api_key,
        )

        if not release_calendar.empty:
            release_calendar = release_calendar.loc[self._start_date : self._end_date]

        return release_calendar

    @handle_errors
    def get_fxmacrodata_dataset(
        self,
        dataset: str,
        currency: str = "usd",
        indicator: str | None = None,
        base: str | None = None,
        quote: str | None = None,
        limit: int = 100,
        api_key: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get any public FXMacroData read dataset as a DataFrame.

        Supported datasets include data_catalogue, announcements,
        latest_announcements, announcement_changes, predictions, calendar,
        forex, cot, commodity, commodities_latest, curves, curve_proxies,
        forward_curves, rate_differentials, forward_differentials,
        market_sessions, risk_sentiment, news, and press_releases.
        """
        params = {
            "currency": currency,
            "indicator": indicator,
            "base": base,
            "quote": quote,
            "limit": limit,
            **kwargs,
        }
        params = {key: value for key, value in params.items() if value is not None}
        dataset_frame = fxmacrodata_model.get_fxmacrodata_dataset(
            dataset,
            api_key=api_key,
            **params,
        )
        if not dataset_frame.empty and isinstance(
            dataset_frame.index, pd.DatetimeIndex
        ):
            dataset_frame = dataset_frame.loc[self._start_date : self._end_date]
        return dataset_frame

    @handle_errors
    def get_gross_domestic_product(
        self,
        countries: list[str] | str | None = None,
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
        https://www.globalmacrodata.com/files/documentations/Variables/nGDP.pdf

        Also known as: GDP, national income, economic growth.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data. Defaults to False.
            gmdb_source (bool | None, optional): If True, retrieves data from the GMDB source. Defaults to None.
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

        economics = Economics(start_date='2015-01-01')

        economics.get_gross_domestic_product(inflation_adjusted=True, countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Netherlands |     Germany |       China |
        |:-----|--------------:|------------:|------------:|
        | 2015 |        792438 | 3.35252e+06 | 6.92094e+07 |
        | 2016 |        811653 | 3.42927e+06 | 7.39494e+07 |
        | 2017 |        834241 | 3.52232e+06 | 7.90868e+07 |
        | 2018 |        853097 | 3.56164e+06 | 8.44244e+07 |
        | 2019 |        872718 | 3.597e+06   | 8.94487e+07 |
        | 2020 |        838886 | 3.44953e+06 | 9.14542e+07 |
        | 2021 |        891550 | 3.57614e+06 | 9.91816e+07 |
        | 2022 |        936192 | 3.62504e+06 | 1.02108e+08 |
        | 2023 |        936871 | 3.61547e+06 | 1.07468e+08 |
        | 2024 |        942765 | 3.61572e+06 | 1.12652e+08 |
        | 2025 |        958100 | 3.64414e+06 | 1.17704e+08 |
        """
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source or inflation_adjusted:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            if inflation_adjusted:
                if not gmdb_source:
                    logger.info(
                        "OECD does not provide inflation adjusted GDP data, using GMDB source instead."
                    )

                gross_domestic_product = gmdb_model.get_real_gross_domestic_product(
                    gmd_dataset=self._gmbd_dataset
                )
            else:
                gross_domestic_product = gmdb_model.get_nominal_gross_domestic_product(
                    gmd_dataset=self._gmbd_dataset
                )
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

        economics = Economics(start_date='2015-01-01')

        economics.get_gross_domestic_product_deflator(countries=['United States', 'Canada', 'Russian Federation'])
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |         97.3159 |  96.7993 |              67.6025 |
        | 2016 |         98.2406 |  97.4935 |              69.5253 |
        | 2017 |        100      | 100      |              73.2441 |
        | 2018 |        102.291  | 101.651  |              80.5677 |
        | 2019 |        103.979  | 103.223  |              83.1968 |
        | 2020 |        105.361  | 104.328  |              83.9441 |
        | 2021 |        110.172  | 112.325  |             100      |
        | 2022 |        118.026  | 120.922  |             115.743  |
        | 2023 |        122.273  | 122.778  |             123.871  |
        | 2024 |        125.195  | 126.443  |             136.148  |
        | 2025 |        127.469  | 129.463  |             142.557  |

        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        gross_domestic_product_deflator = (
            gmdb_model.get_gross_domestic_product_deflator(
                gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2015-01-01')

        economics.get_real_gross_domestic_product_usd(countries=['United States', 'Japan', 'Germany'])
        ```

        Which returns:

        |      |     Germany |       Japan |   United States |
        |:-----|------------:|------------:|----------------:|
        | 2020 | 3.52029e+06 | 4.3728e+06  |     1.97236e+07 |
        | 2021 | 3.6495e+06  | 4.49117e+06 |     2.09179e+07 |
        | 2022 | 3.6994e+06  | 4.54319e+06 |     2.14434e+07 |
        | 2023 | 3.68964e+06 | 4.61947e+06 |     2.20626e+07 |
        | 2024 | 3.6899e+06  | 4.63432e+06 |     2.26726e+07 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        real_gross_domestic_product_usd = (
            gmdb_model.get_real_gross_domestic_product_usd(
                gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2015-01-01')

        economics.get_real_gross_domestic_product_per_capita(countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Germany |   China |   Netherlands |
        |:-----|----------:|--------:|--------------:|
        | 2022 |   43259.3 | 72327.1 |       53219.9 |
        | 2023 |   42779.5 | 76236.4 |       52600.7 |
        | 2024 |   42621.7 | 79949   |       52603.8 |
        | 2025 |   42882.3 | 83590.8 |       53142.2 |
        | 2026 |   43422.6 | 87097.8 |       53735.2 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        real_gross_domestic_product_per_capita = (
            gmdb_model.get_real_gross_domestic_product_per_capita(
                gmd_dataset=self._gmbd_dataset
            )
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
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Total Consumption for a variety of countries over time from the
        Global Macro Database (GMDB). Total Consumption is the total amount of money
        spent by households on consumer goods and services.

        The level is annual and expressed in millions of national currency, so levels are not
        comparable across countries with different currencies, but growth rates are.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: household consumption, private consumption.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data. Defaults to False.
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

        economics = Economics(start_date='2018-01-01')

        total_consumption = economics.get_total_consumption()

        total_consumption.loc[:, ['Netherlands', 'France', 'Poland']]
        ```

        Which returns:

        |      |   Netherlands |      France |      Poland |
        |:-----|--------------:|------------:|------------:|
        | 2018 |        542949 | 1.84554e+06 | 1.64362e+06 |
        | 2019 |        566538 | 1.888e+06   | 1.75043e+06 |
        | 2020 |        558446 | 1.82958e+06 | 1.78522e+06 |
        | 2021 |        606798 | 1.95042e+06 | 1.99581e+06 |
        | 2022 |        679345 | 2.087e+06   | 2.36461e+06 |
        | 2023 |        735272 | 2.2254e+06  | 2.60968e+06 |
        | 2024 |        776464 | 2.29617e+06 | 2.80908e+06 |
        | 2025 |        804450 | 2.3712e+06  | 3.03317e+06 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        if inflation_adjusted:
            total_consumption = gmdb_model.get_real_total_consumption(
                gmd_dataset=self._gmbd_dataset
            )
        else:
            total_consumption = gmdb_model.get_total_consumption(
                gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2018-01-01')

        total_consumption_to_gdp_ratio = economics.get_total_consumption_to_gdp_ratio()

        total_consumption_to_gdp_ratio.loc[:, ['Netherlands', 'France', 'Poland']]
        ```

        Which returns:

        |      |   Netherlands |   France |   Poland |
        |:-----|--------------:|---------:|---------:|
        | 2018 |        0.6897 |   0.7835 |   0.7653 |
        | 2019 |        0.6828 |   0.7762 |   0.7565 |
        | 2020 |        0.684  |   0.7892 |   0.7555 |
        | 2021 |        0.6806 |   0.7776 |   0.7499 |
        | 2022 |        0.6836 |   0.7859 |   0.7626 |
        | 2023 |        0.6887 |   0.7885 |   0.7672 |
        | 2024 |        0.6981 |   0.7891 |   0.7675 |
        | 2025 |        0.7002 |   0.7899 |   0.772  |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        total_consumption_to_gdp_ratio = gmdb_model.get_total_consumption_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2014-01-01')

        investment = economics.get_investment()

        investment.loc[:, ['United States', 'Portugal', 'China']]
        ```

        Which returns:

        |      |   United States |   Portugal |       China |
        |:-----|----------------:|-----------:|------------:|
        | 2014 |     3.68027e+06 |    26506.7 | 2.94903e+07 |
        | 2015 |     3.91787e+06 |    28493.5 | 2.97829e+07 |
        | 2016 |     3.92797e+06 |    29527   | 3.18198e+07 |
        | 2017 |     4.14914e+06 |    33755.8 | 3.57888e+07 |
        | 2018 |     4.45541e+06 |    37528.2 | 4.02584e+07 |
        | 2019 |     4.66771e+06 |    39644.4 | 4.26678e+07 |
        | 2020 |     4.57384e+06 |    38333.2 | 4.39554e+07 |
        | 2021 |     5.0519e+06  |    44565.3 | 4.95782e+07 |
        | 2022 |     5.70851e+06 |    50045.8 | 5.19792e+07 |
        | 2023 |     5.97132e+06 |    52005.7 | 5.22754e+07 |
        | 2024 |     6.36237e+06 |    54339.8 | 5.5217e+07  |
        | 2025 |     6.66113e+06 |    57349.5 | 5.84789e+07 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        investment = gmdb_model.get_investment(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics(start_date='2019-01-01')

        investment_to_gdp_ratio = economics.get_investment_to_gdp_ratio()

        investment_to_gdp_ratio.loc[:, ['Australia', 'Japan', 'Turkey']]
        ```

        Which returns:

        |      |   Australia |   Japan |   Turkey |
        |:-----|------------:|--------:|---------:|
        | 2019 |      0.2255 |  0.2579 |   0.2488 |
        | 2020 |      0.223  |  0.2522 |   0.3134 |
        | 2021 |      0.2331 |  0.258  |   0.314  |
        | 2022 |      0.2372 |  0.2681 |   0.3504 |
        | 2023 |      0.2398 |  0.264  |   0.2996 |
        | 2024 |      0.2415 |  0.2657 |   0.2557 |
        | 2025 |      0.2393 |  0.2664 |   0.2465 |
        | 2026 |      0.2403 |  0.2652 |   0.254  |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        investment_to_gdp_ratio = gmdb_model.get_investment_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2020-01-01')

        fixed_investment = economics.get_fixed_investment()

        fixed_investment.loc[:, ['United Kingdom', 'Germany', 'France']]
        ```

        Which returns:

        |      |   United Kingdom |   Germany |   France |
        |:-----|-----------------:|----------:|---------:|
        | 2020 |           362076 |    736476 |   520134 |
        | 2021 |           398052 |    779205 |   588983 |
        | 2022 |           443416 |    858253 |   628022 |
        | 2023 |           469685 |    899880 |   651792 |
        | 2024 |           473070 |    897275 |   657075 |
        | 2025 |           482008 |    925002 |   674350 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        fixed_investment = gmdb_model.get_fixed_investment(
            gmd_dataset=self._gmbd_dataset
        )

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

        economics = Economics(start_date='2000-01-01')

        fixed_investment_to_gdp_ratio = economics.get_fixed_investment_to_gdp_ratio()

        fixed_investment_to_gdp_ratio.loc[:, ['Austria', 'Germany', 'Switzerland']]
        ```

        Which returns:

        |      |   Austria |   Germany |   Switzerland |
        |:-----|----------:|----------:|--------------:|
        | 2000 |    0.2571 |    0.2288 |        0.2751 |
        | 2001 |    0.2493 |    0.2154 |        0.2687 |
        | 2002 |    0.2362 |    0.1989 |        0.2708 |
        | 2003 |    0.2419 |    0.1926 |        0.2653 |
        | 2004 |    0.2377 |    0.1883 |        0.2731 |
        | 2005 |    0.2323 |    0.1876 |        0.2726 |
        | 2006 |    0.2283 |    0.1946 |        0.2706 |
        | 2007 |    0.2314 |    0.1972 |        0.2721 |
        | 2008 |    0.2353 |    0.1996 |        0.2673 |
        | 2009 |    0.2266 |    0.1883 |        0.2521 |
        | 2010 |    0.2186 |    0.1918 |        0.2519 |
        | 2011 |    0.2271 |    0.2    |        0.2552 |
        | 2012 |    0.2295 |    0.1996 |        0.2627 |
        | 2013 |    0.2331 |    0.1957 |        0.2623 |
        | 2014 |    0.2297 |    0.1978 |        0.2642 |
        | 2015 |    0.229  |    0.1976 |        0.2641 |
        | 2016 |    0.2332 |    0.2004 |        0.265  |
        | 2017 |    0.2385 |    0.2015 |        0.2715 |
        | 2018 |    0.2431 |    0.2084 |        0.266  |
        | 2019 |    0.2508 |    0.2117 |        0.2662 |
        | 2020 |    0.2513 |    0.2135 |        0.2699 |
        | 2021 |    0.2588 |    0.2119 |        0.2634 |
        | 2022 |    0.2547 |    0.2171 |        0.2625 |
        | 2023 |    0.249  |    0.215  |        0.2593 |
        | 2024 |    0.2515 |    0.2067 |        0.2491 |
        | 2025 |    0.2525 |    0.2072 |        0.248  |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        fixed_investment_to_gdp_ratio = gmdb_model.get_fixed_investment_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
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
        | 1980 |       89636.1 |    164376 |  46573.7 |
        | 1981 |      103010   |    186137 |  61412.3 |
        | 1982 |      106456   |    200976 |  59212.3 |
        | 1983 |      109543   |    204049 |  57306   |
        | 1984 |      123555   |    229107 |  69340.1 |
        | 1985 |      131138   |    252794 |  75856.1 |
        | 1986 |      114516   |    247153 |  90398.9 |
        | 1987 |      112468   |    246623 | 151965   |
        | 1988 |      122716   |    265208 | 218329   |
        | 1989 |      138137   |    299732 | 203483   |
        | 1990 |      144521   |    334043 | 256949   |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        exports = gmdb_model.get_exports(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics(start_date='2015-01-01')

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
        | 2020 |          0.1007 |   0.2947 |               0.2552 |
        | 2021 |          0.1079 |   0.3122 |               0.2977 |
        | 2022 |          0.116  |   0.3385 |               0.2803 |
        | 2023 |          0.1101 |   0.3337 |               0.2308 |
        | 2024 |          0.1075 |   0.3235 |               0.2124 |
        | 2025 |          0.1059 |   0.3165 |               0.2122 |
        | 2026 |          0.1049 |   0.3124 |               0.2123 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        exports_to_gdp_ratio = gmdb_model.get_exports_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2010-01-01')

        economics.get_imports(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |           Canada |      Mexico |
        |:-----|----------------:|-----------------:|------------:|
        | 2010 |     2.38956e+06 | 517153           | 4.22619e+06 |
        | 2011 |     2.69548e+06 | 564513           | 4.84306e+06 |
        | 2012 |     2.76932e+06 | 589137           | 5.40808e+06 |
        | 2013 |     2.76638e+06 | 606801           | 5.41441e+06 |
        | 2014 |     2.88744e+06 | 651176           | 5.9193e+06  |
        | 2015 |     2.79494e+06 | 683019           | 6.97041e+06 |
        | 2016 |     2.73883e+06 | 685868           | 8.0699e+06  |
        | 2017 |     2.93159e+06 | 720254           | 8.88784e+06 |
        | 2018 |     3.13117e+06 | 766265           | 9.95323e+06 |
        | 2019 |     3.11668e+06 | 782419           | 9.78051e+06 |
        | 2020 |     2.77734e+06 | 703532           | 9.06124e+06 |
        | 2021 |     3.41546e+06 | 785539           | 1.13433e+07 |
        | 2022 |     3.97631e+06 | 948468           | 1.34558e+07 |
        | 2023 |     3.84981e+06 | 978214           | 1.18117e+07 |
        | 2024 |     4.03094e+06 | 990187           | 1.19409e+07 |
        | 2025 |     4.1031e+06  |      1.02675e+06 | 1.22266e+07 |

        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        imports = gmdb_model.get_imports(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics(start_date='2010-01-01')

        economics.get_imports_to_gdp_ratio(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |   Canada |   Mexico |
        |:-----|----------------:|---------:|---------:|
        | 2010 |          0.1588 |   0.3104 |   0.3026 |
        | 2011 |          0.1728 |   0.3182 |   0.3172 |
        | 2012 |          0.1704 |   0.3224 |   0.3272 |
        | 2013 |          0.1639 |   0.319  |   0.3194 |
        | 2014 |          0.164  |   0.3264 |   0.3264 |
        | 2015 |          0.1528 |   0.3431 |   0.3625 |
        | 2016 |          0.1456 |   0.3386 |   0.3887 |
        | 2017 |          0.1495 |   0.3365 |   0.3944 |
        | 2018 |          0.1516 |   0.3427 |   0.4117 |
        | 2019 |          0.1447 |   0.3382 |   0.3893 |
        | 2020 |          0.1301 |   0.3168 |   0.3762 |
        | 2021 |          0.1442 |   0.3121 |   0.425  |
        | 2022 |          0.1529 |   0.3371 |   0.4565 |
        | 2023 |          0.1389 |   0.3382 |   0.3718 |
        | 2024 |          0.1382 |   0.328  |   0.3491 |
        | 2025 |          0.1352 |   0.3244 |   0.3336 |
        | 2026 |          0.1313 |   0.3239 |   0.3253 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        imports_to_gdp_ratio = gmdb_model.get_imports_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
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

        |      |   Germany |       China |   United States |
        |:-----|----------:|------------:|----------------:|
        | 2020 |    184386 | 2.45079e+06 |         -626202 |
        | 2021 |    189652 | 2.97188e+06 |         -860029 |
        | 2022 |     98724 | 3.89305e+06 |         -958935 |
        | 2023 |    167656 | 2.73467e+06 |         -797342 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        exports = gmdb_model.get_exports(gmd_dataset=self._gmbd_dataset)
        imports = gmdb_model.get_imports(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics(start_date='2015-01-01')

        economics.get_current_account_balance(countries=['France', 'Germany', 'Italy'])
        ```

        Which returns:

        |      |    France |   Germany |     Italy |
        |:-----|----------:|----------:|----------:|
        | 2015 |  -7154.56 |    259781 |  20674.6  |
        | 2016 | -11784    |    270199 |  41956.5  |
        | 2017 | -12535.5  |    255962 |  42548.2  |
        | 2018 | -16440.4  |    267594 |  44461.4  |
        | 2019 |  14520.3  |    283851 |  56954.4  |
        | 2020 | -47594.2  |    222500 |  62809.1  |
        | 2021 |   6947.44 |    263455 |  38674.2  |
        | 2022 | -31095.1  |    164638 | -34928.5  |
        | 2023 | -28111.7  |    257704 |   -297.92 |
        | 2024 |   2650.74 |    286059 |  23619.7  |
        | 2025 |  -3590.1  |    285609 |  31890.9  |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        current_account_balance = gmdb_model.get_current_account_balance(
            gmd_dataset=self._gmbd_dataset
        )

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

        economics = Economics(start_date='2015-01-01')

        economics.get_current_account_balance_to_gdp_ratio(countries=[
            'Poland', 'Turkey', 'United Kingdom'])
        ```

        Which returns:

        |      |   Poland |   Turkey |   United Kingdom |
        |:-----|---------:|---------:|-----------------:|
        | 2015 |  -0.0129 |  -0.0246 |          -0.0495 |
        | 2016 |  -0.0102 |  -0.0255 |          -0.0545 |
        | 2017 |  -0.0116 |  -0.0409 |          -0.0349 |
        | 2018 |  -0.0193 |  -0.0183 |          -0.0393 |
        | 2019 |  -0.0025 |   0.0197 |          -0.0269 |
        | 2020 |   0.0248 |  -0.0434 |          -0.0293 |
        | 2021 |  -0.0124 |  -0.008  |          -0.0044 |
        | 2022 |  -0.0244 |  -0.0506 |          -0.021  |
        | 2023 |   0.0155 |  -0.0398 |          -0.0196 |
        | 2024 |   0.0085 |  -0.0216 |          -0.0279 |
        | 2025 |  -0.0002 |  -0.0207 |          -0.0283 |
        | 2026 |  -0.0043 |  -0.0201 |          -0.028  |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        current_account_balance_to_gdp_ratio = (
            gmdb_model.get_current_account_balance_to_gdp(
                gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_debt(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |      Canada |      Mexico |
        |:-----|----------------:|------------:|------------:|
        | 2015 |     1.91477e+07 | 1.83178e+06 | 9.8014e+06  |
        | 2016 |     2.00426e+07 | 1.87153e+06 | 1.1418e+07  |
        | 2017 |     2.06965e+07 | 1.94674e+06 | 1.18362e+07 |
        | 2018 |     2.20709e+07 | 2.02946e+06 | 1.26207e+07 |
        | 2019 |     2.3264e+07  | 2.08707e+06 | 1.30389e+07 |
        | 2020 |     2.81514e+07 | 2.62466e+06 | 1.4089e+07  |
        | 2021 |     2.94887e+07 | 2.85651e+06 | 1.51449e+07 |
        | 2022 |     3.08486e+07 | 3.02057e+06 | 1.59543e+07 |
        | 2023 |     3.29114e+07 | 3.10891e+06 | 1.68674e+07 |
        | 2024 |     3.52945e+07 | 3.20199e+06 | 1.97489e+07 |
        | 2025 |     3.76545e+07 | 3.26736e+06 | 2.12283e+07 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_debt = gmdb_model.get_government_debt(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_debt_to_gdp_ratio(countries=['Netherlands', 'Germany', 'China'])
        ```

        Which returns:

        |      |   Netherlands |   Germany |   China |
        |:-----|--------------:|----------:|--------:|
        | 2015 |        0.638  |    0.7056 |  0.4149 |
        | 2016 |        0.6088 |    0.6763 |  0.507  |
        | 2017 |        0.5599 |    0.6395 |  0.5495 |
        | 2018 |        0.5156 |    0.6073 |  0.5666 |
        | 2019 |        0.4758 |    0.5856 |  0.604  |
        | 2020 |        0.5334 |    0.6786 |  0.7016 |
        | 2021 |        0.5044 |    0.6788 |  0.7185 |
        | 2022 |        0.4835 |    0.6479 |  0.7739 |
        | 2023 |        0.4502 |    0.6266 |  0.8438 |
        | 2024 |        0.4426 |    0.6268 |  0.9012 |
        | 2025 |        0.4511 |    0.621  |  0.9384 |
        | 2026 |        0.4619 |    0.6095 |  0.9775 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_debt_to_gdp_ratio = gmdb_model.get_government_debt_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2019-01-01')

        economics.get_government_revenue(countries=['United Kingdom', 'Canada', 'Japan'])
        ```

        Which returns:

        |      |   United Kingdom |           Canada |       Japan |
        |:-----|-----------------:|-----------------:|------------:|
        | 2019 | 809863           | 938659           | 1.91079e+08 |
        | 2020 | 774335           | 919587           | 1.91365e+08 |
        | 2021 | 868383           |      1.07026e+06 | 2.01026e+08 |
        | 2022 | 994377           |      1.15747e+06 | 2.10432e+08 |
        | 2023 |      1.03927e+06 |      1.21236e+06 | 2.19057e+08 |
        | 2024 |      1.0989e+06  |      1.24586e+06 | 2.20353e+08 |
        | 2025 |      1.14061e+06 |      1.30501e+06 | 2.31967e+08 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_revenue = gmdb_model.get_government_revenue(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_revenue_to_gdp_ratio(countries=['United States', 'Canada', 'Russian Federation'])
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |          0.315  |   0.3996 |               0.3189 |
        | 2016 |          0.3098 |   0.403  |               0.3292 |
        | 2017 |          0.304  |   0.4034 |               0.3336 |
        | 2018 |          0.3001 |   0.4102 |               0.3554 |
        | 2019 |          0.3001 |   0.4057 |               0.3568 |
        | 2020 |          0.3065 |   0.4141 |               0.3516 |
        | 2021 |          0.3158 |   0.4252 |               0.3544 |
        | 2022 |          0.3238 |   0.4114 |               0.342  |
        | 2023 |          0.2921 |   0.4192 |               0.3426 |
        | 2024 |          0.299  |   0.4127 |               0.3545 |
        | 2025 |          0.3006 |   0.4124 |               0.3647 |
        | 2026 |          0.3065 |   0.4115 |               0.365  |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_revenue_to_gdp_ratio = (
            gmdb_model.get_government_revenue_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_tax_revenue(countries=['Kenya', 'Nigeria', 'South Africa'] )
        ```

        Which returns:

        |      |         Kenya |   Nigeria |   South Africa |
        |:-----|--------------:|----------:|---------------:|
        | 2015 |   1.0216e+06  |    815000 |    1.10735e+06 |
        | 2016 |   1.13656e+06 |       nan |    1.18172e+06 |
        | 2017 |   1.27696e+06 |       nan |    1.25783e+06 |
        | 2018 |   1.34139e+06 |       nan |    1.33516e+06 |
        | 2019 |   1.54591e+06 |       nan |    1.39884e+06 |
        | 2020 |   1.53224e+06 |       nan |    1.29417e+06 |
        | 2021 |   1.63031e+06 |       nan |    1.61069e+06 |
        | 2022 |   1.9694e+06  |       nan |    1.7308e+06  |
        | 2023 |   2.11419e+06 |       nan |  nan           |
        | 2024 | nan           |       nan |  nan           |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_tax_revenue = gmdb_model.get_government_tax_revenue(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_tax_revenue_to_gdp_ratio(
            countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |   United States |   Canada |   Mexico |
        |:-----|----------------:|---------:|---------:|
        | 2015 |          0.1994 |   0.1239 |   0.1318 |
        | 2016 |          0.1958 |   0.125  |   0.1386 |
        | 2017 |          0.2031 |   0.1261 |   0.134  |
        | 2018 |          0.1874 |   0.1306 |   0.1336 |
        | 2019 |          0.1888 |   0.1274 |   0.1348 |
        | 2020 |          0.1934 |   0.135  |   0.1452 |
        | 2021 |          0.2065 |   0.1322 |   0.1414 |
        | 2022 |          0.2156 |   0.1283 |   0.1368 |
        | 2023 |          0.1022 |   0.1401 |   0.1427 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_tax_revenue_to_gdp_ratio = (
            gmdb_model.get_government_tax_revenue_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_expenditure(countries=['Japan', 'China', 'India'])
        ```

        Which returns:

        |      |       Japan |       China |       India |
        |:-----|------------:|------------:|------------:|
        | 2015 | 2.00659e+08 | 2.18369e+07 | 3.72653e+07 |
        | 2016 | 2.02662e+08 | 2.41071e+07 | 4.19161e+07 |
        | 2017 | 2.029e+08   | 2.70539e+07 | 4.48306e+07 |
        | 2018 | 2.045e+08   | 3.04742e+07 | 4.97591e+07 |
        | 2019 | 2.08067e+08 | 3.38357e+07 | 5.39701e+07 |
        | 2020 | 2.40235e+08 | 3.63103e+07 | 6.15854e+07 |
        | 2021 | 2.34757e+08 | 3.74347e+07 | 7.00984e+07 |
        | 2022 | 2.35009e+08 | 4.02599e+07 | 7.85448e+07 |
        | 2023 | 2.44046e+08 | 4.17285e+07 | 8.59931e+07 |
        | 2024 | 2.57546e+08 | 4.45191e+07 | 9.43978e+07 |
        | 2025 | 2.50987e+08 | 4.77611e+07 | 1.02862e+08 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_expenditure = gmdb_model.get_government_expenditure(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_expenditure_to_gdp_ratio(
            countries=['United States', 'Japan', 'Netherlands'])
        ```

        Which returns:

        |      |   United States |   Japan |   Netherlands |
        |:-----|----------------:|--------:|--------------:|
        | 2015 |          0.3503 |  0.3729 |        0.4525 |
        | 2016 |          0.3533 |  0.3723 |        0.4392 |
        | 2017 |          0.3519 |  0.3669 |        0.4276 |
        | 2018 |          0.3535 |  0.3674 |        0.4244 |
        | 2019 |          0.3581 |  0.3729 |        0.421  |
        | 2020 |          0.4457 |  0.4452 |        0.4781 |
        | 2021 |          0.426  |  0.4244 |        0.4589 |
        | 2022 |          0.3631 |  0.4184 |        0.4327 |
        | 2023 |          0.3628 |  0.4116 |        0.432  |
        | 2024 |          0.3753 |  0.422  |        0.4416 |
        | 2025 |          0.3738 |  0.3983 |        0.448  |
        | 2026 |          0.374  |  0.3963 |        0.4513 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_expenditure_to_gdp_ratio = (
            gmdb_model.get_government_expenditure_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
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

        economics = Economics(start_date='2015-01-01')

        economics.get_government_deficit(countries=['United States', 'Canada', 'Mexico'])
        ```

        Which returns:

        |      |     United States |      Canada |            Mexico |
        |:-----|------------------:|------------:|------------------:|
        | 2015 | -645814           |   -1234.07  | -742032           |
        | 2016 | -819141           |   -9175.67  | -556543           |
        | 2017 | -940204           |   -2397.52  | -233024           |
        | 2018 |      -1.10203e+06 |    8048.43  | -517139           |
        | 2019 |      -1.24932e+06 |    -393.306 | -569235           |
        | 2020 |      -2.97292e+06 | -243126     |      -1.03335e+06 |
        | 2021 |      -2.61038e+06 |  -73474.8   |      -1.00008e+06 |
        | 2022 |      -1.02051e+06 |    3066.48  |      -1.25793e+06 |
        | 2023 |      -1.9593e+06  |  -16572     |      -1.36948e+06 |
        | 2024 |      -2.22521e+06 |  -59827.1   |      -2.01778e+06 |
        | 2025 |      -2.22159e+06 |  -32373.6   |      -1.28252e+06 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_deficit = gmdb_model.get_government_deficit(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2015-01-01')

        government_deficit_to_gdp_ratio = economics.get_government_deficit_to_gdp_ratio()

        government_deficit_to_gdp_ratio.loc[:, ['New Zealand', 'Australia', 'United Kingdom']]
        ```

        Which returns:

        |      |   New Zealand |   Australia |   United Kingdom |
        |:-----|--------------:|------------:|-----------------:|
        | 2015 |        0.0036 |     -0.0278 |          -0.0462 |
        | 2016 |        0.0098 |     -0.0242 |          -0.0334 |
        | 2017 |        0.0136 |     -0.0172 |          -0.0251 |
        | 2018 |        0.0127 |     -0.0126 |          -0.0227 |
        | 2019 |       -0.025  |     -0.044  |          -0.0248 |
        | 2020 |       -0.0433 |     -0.0872 |          -0.1314 |
        | 2021 |       -0.0324 |     -0.0635 |          -0.0786 |
        | 2022 |       -0.0351 |     -0.0219 |          -0.047  |
        | 2023 |       -0.0333 |     -0.0086 |          -0.0596 |
        | 2024 |       -0.0384 |     -0.0166 |          -0.0425 |
        | 2025 |       -0.0349 |     -0.0204 |          -0.0374 |
        | 2026 |       -0.0234 |     -0.013  |          -0.0354 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_deficit_to_gdp_ratio = (
            gmdb_model.get_government_deficit_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
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

        economics = Economics()

        trust_in_government = economics.get_trust_in_government()

        trust_in_government.loc[:, ['United States', 'Greece', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Greece |   Japan |
        |:-----|----------------:|---------:|--------:|
        | 2006 |          0.4959 |   0.4821 |  0.3248 |
        | 2007 |          0.4039 |   0.4821 |  0.3248 |
        | 2008 |          0.4065 | nan      |  0.238  |
        | 2009 |          0.3927 |   0.3153 |  0.238  |
        | 2010 |          0.3927 |   0.3153 |  0.238  |
        | 2011 |          0.3927 |   0.1556 |  0.2564 |
        | 2012 |          0.3927 |   0.1556 |  0.2564 |
        | 2013 |          0.3927 |   0.1556 |  0.2564 |
        | 2014 |          0.3463 |   0.2738 |  0.4084 |
        | 2015 |          0.3463 |   0.2738 |  0.3283 |
        | 2016 |          0.3463 |   0.2738 |  0.3283 |
        | 2017 |          0.3463 |   0.2415 |  0.387  |
        | 2018 |          0.3463 |   0.2415 |  0.387  |
        | 2019 |          0.3654 |   0.2415 |  0.387  |
        | 2020 |          0.3654 |   0.3695 |  0.3443 |
        | 2021 |          0.3654 |   0.3695 |  0.3443 |
        | 2022 |          0.3654 |   0.3086 |  0.3798 |
        | 2023 |          0.3654 |   0.3217 |  0.3798 |
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

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: CPI, cost of living index.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Only used when `oecd_source=True`; the GMDB source is always annual. Defaults to None.
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
        | 2008 |   98.6508 |  98.1924 |     99.527 |
        | 2009 |   98.8937 |  98.2913 |     98.628 |
        | 2010 |  100      | 100      |    100     |
        | 2011 |  102.482  | 102.287  |    103.555 |
        | 2012 |  104.695  | 104.553  |    106.43  |
        | 2013 |  106.377  | 105.589  |    106.897 |
        | 2014 |  107.196  | 106.236  |    106.727 |
        | 2015 |  107.924  | 106.328  |    107.269 |
        | 2016 |  108.32   | 106.653  |    107.951 |
        | 2017 |  110.164  | 107.896  |    109.631 |
        | 2018 |  112.296  | 110.162  |    110.91  |
        | 2019 |  113.815  | 111.591  |    111.243 |
        | 2020 |  114.239  | 112.18   |    111.108 |
        """
        if oecd_source:
            period = (
                period
                if period is not None
                else "quarterly" if self._quarterly else "yearly"
            )
            consumer_price_index = oecd_model.get_consumer_price_index(
                period=period, start_date=self._start_date, end_date=self._end_date
            )
        else:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            consumer_price_index = gmdb_model.get_consumer_price_index(
                gmd_dataset=self._gmbd_dataset
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
        )

    @handle_errors
    def get_inflation_rate(
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
        Inflation Rate is the percentage change in the Consumer Price Index (CPI) from one
        period to another. It is a measure of the rate of price increases in the economy.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The rate is expressed as a decimal fraction (0.0412 for 4.1166%) on an annual basis.
        The GMDB extends its series with IMF World Economic Outlook projections, so the
        current year and any later years the end_date reaches are forecasts rather than
        outturns.

        Changed in v2.2.0: this used to be returned in percentage points (4.1166 for
        4.1166%). It is now a decimal fraction, matching every other rate in the Finance
        Toolkit.

        Also known as: CPI-based inflation, price increases, consumer prices.

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
        | 2003 |    0.0103 |   0.021  |     0.0322 |
        | 2004 |    0.0167 |   0.0214 |     0.0237 |
        | 2005 |    0.0155 |   0.0175 |     0.0228 |
        | 2006 |    0.0158 |   0.0168 |     0.0311 |
        | 2007 |    0.023  |   0.0149 |     0.0245 |
        | 2008 |    0.0263 |   0.0281 |     0.0259 |
        | 2009 |    0.0031 |   0.0009 |    -0.0084 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        inflation_rate = gmdb_model.get_inflation_rate(gmd_dataset=self._gmbd_dataset)

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
        |:-----|-----------------:|----------:|
        | 2018 |          106.059 |   102.758 |
        | 2019 |          106.068 |   103.65  |
        | 2020 |          103.849 |   103.15  |
        | 2021 |          116.511 |   108.241 |
        | 2022 |          134.46  |   122.75  |
        """
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
        | 2008-09 |   98.4042 |  97.4657 |    97.8598 |
        | 2008-10 |   98.2065 |  97.4716 |    97.748  |
        | 2008-11 |   97.9886 |  97.5514 |    97.3693 |
        | 2008-12 |   97.7184 |  97.5094 |    96.9437 |
        | 2009-01 |   97.5575 |  97.4412 |    96.6658 |
        | 2009-02 |   97.4573 |  97.3785 |    96.658  |
        | 2009-03 |   97.4165 |  97.4899 |    96.9339 |
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
        | 2022-09 | 100.196  | 100.381  |      101.157 |
        | 2022-10 |  99.7735 |  99.9799 |      101.145 |
        | 2022-11 |  99.4016 |  99.6322 |      101.141 |
        | 2022-12 |  99.2565 |  99.3052 |      101.161 |
        | 2023-01 |  99.2264 |  98.9732 |      101.222 |
        | 2023-02 |  99.2644 |  98.6224 |      101.35  |
        | 2023-03 |  99.3837 |  98.2617 |      101.553 |
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
        | 2023-06 |         99.1511 |          99.9353 | 100.023 |
        | 2023-07 |         99.2797 |         100.196  | 100.037 |
        | 2023-08 |         99.3826 |         100.419  | 100.055 |
        | 2023-09 |         99.4504 |         100.622  | 100.067 |
        | 2023-10 |         99.4863 |         100.806  | 100.075 |
        | 2023-11 |         99.5104 |         100.998  | 100.085 |
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
        | 2016 | 102.559 |       104.557 |   106.626 |
        | 2017 | 104.76  |       110.834 |   116.945 |
        | 2018 | 106.053 |       118.68  |   127.047 |
        | 2019 | 107.254 |       124.372 |   127.837 |
        | 2020 | 106.994 |       131.653 |   128.345 |
        | 2021 | 112.714 |       144.382 |   135.141 |
        | 2022 | 118.739 |       152.287 |   141.162 |
        | 2023 | 118.74  |       139.601 |   134.022 |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            house_prices = gmdb_model.get_house_price_index(
                gmd_dataset=self._gmbd_dataset
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
        | 2016 |  108.667 |         103.773 |          101.725 |
        | 2017 |  118.586 |         107.731 |          102.699 |
        | 2018 |  130.05  |         111.627 |          103.174 |
        | 2019 |  143.192 |         115.765 |          103.924 |
        | 2020 |  156.58  |         119.382 |          105.399 |
        | 2021 |  172.63  |         122.062 |          107.148 |
        | 2022 |  221.225 |         129.426 |          110.897 |
        | 2023 |  398.003 |         139.543 |          117.179 |
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
        | 2018 |          1.0014 |      1.9888 |
        | 2019 |          0.9955 |      1.9676 |
        | 2020 |          0.9532 |      1.8811 |
        | 2021 |          0.9624 |      1.9168 |
        | 2022 |          1.0162 |      1.9253 |
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

        economics = Economics(start_date="2013-01-01")

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
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
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

        Also known as: currency exchange, FX rates, foreign exchange rates.

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
        | 2000 | 107.835  |     8394.53 |  8.2784 |
        | 2001 | 121.484  |    10253    |  8.2777 |
        | 2002 | 125.255  |     9318.73 |  8.2771 |
        | 2003 | 115.936  |     8573.73 |  8.278  |
        | 2004 | 108.147  |     8931.52 |  8.2782 |
        | 2005 | 110.133  |     9701.29 |  8.1942 |
        | 2006 | 116.354  |     9164.03 |  7.9724 |
        | 2007 | 117.755  |     9139.41 |  7.6074 |
        | 2008 | 103.388  |     9663.87 |  6.9502 |
        | 2009 |  93.5716 |    10376.8  |  6.8308 |
        | 2010 |  87.7606 |     9078.03 |  6.769  |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            exchange_rates = gmdb_model.get_usd_exchange_rate(
                gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2018-01-01')

        economics.get_real_effective_exchange_rate(countries=['United States', 'Japan', 'Netherlands'])
        ```

        Which returns:

        |      |   Japan |   Netherlands |   United States |
        |:-----|--------:|--------------:|----------------:|
        | 2021 | 70.6912 |       102.098 |         115.627 |
        | 2022 | 61.011  |       102.238 |         126.626 |
        | 2023 | 58.1149 |       103.352 |         127.54  |
        | 2024 | 55.9376 |       104.859 |         134.572 |
        | 2025 | 55.5007 |       104.174 |         134.22  |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        real_effective_exchange_rate = gmdb_model.get_real_effective_exchange_rate(
            gmd_dataset=self._gmbd_dataset
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
        | 2010 |        701718 | 1.9878e+06 |     8.478e+06   |
        | 2011 |        727265 | 2.1053e+06 |     8.8452e+06  |
        | 2012 |        746482 | 2.2556e+06 |     9.7505e+06  |
        | 2013 |        741372 | 2.3144e+06 |     1.04976e+07 |
        | 2014 |        743043 | 2.4272e+06 |     1.11176e+07 |
        | 2015 |        822382 | 2.6518e+06 |     1.17742e+07 |
        | 2016 |        841302 | 2.8022e+06 |     1.24908e+07 |
        | 2017 |        851237 | 2.9236e+06 |     1.32864e+07 |
        | 2018 |        846513 | 3.0562e+06 |     1.38692e+07 |
        | 2019 |        889033 | 3.1968e+06 |     1.44327e+07 |
        | 2020 |        974276 | 3.4582e+06 |     1.54013e+07 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        money_supply = gmdb_model.get_money_supply(gmd_dataset=self._gmbd_dataset)

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

        Changed in v2.2.0: this used to be returned in percentage points (5.375 for 5.375%).
        It is now a decimal fraction, matching every other rate in the Finance Toolkit.

        Also known as: policy rate, benchmark rate, base rate.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
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
        | 2024 |        0.0381 |    0.0381 |          0.0438 |
        | 2025 |        0.0288 |    0.0288 |          0.0426 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        central_bank_policy_rate = gmdb_model.get_central_bank_policy_rate(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='2023-05-01')

        economics.get_short_term_interest_rate(
            countries=['Japan', 'United States', 'China'],
            gmdb_source=False,
            period='quarterly'
        )
        ```

        Which returns:

        |        |    Japan |   United States |    China |
        |:-------|---------:|----------------:|---------:|
        | 2023Q2 |  -0.0001 |          0.0513 |   0.0289 |
        | 2023Q3 |   0.0001 |          0.0543 |   0.0261 |
        | 2023Q4 |   0.0002 |          0.054  |   0.0288 |
        | 2024Q1 |   0.0005 |          0.0526 |   0.0267 |
        | 2024Q2 |   0.0013 |          0.0531 |   0.0235 |
        | 2024Q3 |   0.0023 |          0.051  |   0.021  |
        | 2024Q4 |   0.0033 |          0.0454 |   0.0205 |
        | 2025Q1 |   0.0079 |          0.0432 |   0.0202 |
        | 2025Q2 |   0.0078 |          0.0431 |   0.0188 |
        | 2025Q3 |   0.0079 |          0.042  |   0.0171 |
        | 2025Q4 |   0.009  |          0.0386 |   0.0168 |
        | 2026Q1 |   0.012  |          0.0366 |   0.0172 |
        | 2026Q2 | nan      |          0.0375 | nan      |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            short_term_interest_rate = gmdb_model.get_short_term_interest_rate(
                gmd_dataset=self._gmbd_dataset
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

        Changed in v2.2.0: the GMDB source previously returned percentage points (3.57 for
        3.57%) while the OECD source returned a decimal fraction. The GMDB series is now
        divided by 100 so both sources agree; divide any hard-coded comparison by 100.

        Also known as: 10-year yield, government bond rate, long-term yield.

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
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            long_term_interest_rate = gmdb_model.get_long_term_interest_rate(
                gmd_dataset=self._gmbd_dataset
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
        | 2018 |          0.0047 |   -0.0133 | -0.0092 |
        | 2019 |          0.0033 |   -0.016  | -0.0059 |
        | 2020 |         -0.0034 |   -0.0102 |  0.0001 |
        | 2021 |         -0.0326 |   -0.0352 |  0.0031 |
        | 2022 |         -0.0505 |   -0.0573 | -0.0228 |
        | 2023 |         -0.0016 |   -0.0351 | -0.0271 |
        """
        rate_type = rate_type.lower()

        if rate_type not in ["long_term", "short_term"]:
            raise ValueError(
                "Please choose either 'long_term' or 'short_term' for the rate_type parameter."
            )

        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        if gmdb_source:
            nominal_interest_rate = (
                gmdb_model.get_long_term_interest_rate(gmd_dataset=self._gmbd_dataset)
                if rate_type == "long_term"
                else gmdb_model.get_short_term_interest_rate(
                    gmd_dataset=self._gmbd_dataset
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

        inflation_rate = gmdb_model.get_inflation_rate(gmd_dataset=self._gmbd_dataset)

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
        | 2021 |          0.0133 |    0.0017 |  0.0014 |
        | 2022 |          0.0072 |    0.008  |  0.0026 |
        | 2023 |         -0.0122 |   -0.01   |  0.0056 |
        """
        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            long_term_interest_rate = gmdb_model.get_long_term_interest_rate(
                gmd_dataset=self._gmbd_dataset
            )
            short_term_interest_rate = gmdb_model.get_short_term_interest_rate(
                gmd_dataset=self._gmbd_dataset
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
        | 2010 |    0.2742 |    0.0933 |          0.0568 |
        | 2011 |    0.2696 |    0.102  |          0.0619 |
        | 2012 |    0.307  |    0.1137 |          0.0631 |
        | 2013 |    0.3011 |    0.1147 |          0.0665 |
        | 2014 |    0.3068 |    0.1192 |          0.0677 |
        | 2015 |    0.2985 |    0.1264 |          0.0675 |
        | 2016 |    0.3034 |    0.1253 |          0.0707 |
        | 2017 |    0.2984 |    0.1332 |          0.074  |
        | 2018 |    0.2944 |    0.1396 |          0.0764 |
        | 2019 |    0.3006 |    0.1485 |          0.0776 |
        | 2020 |    0.3191 |    0.1637 |          0.083  |
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
        | 2010 |    11.893 |          19.644 |    7.967 |
        | 2011 |    11.702 |          18.733 |    7.818 |
        | 2012 |    11.405 |          17.921 |    7.611 |
        | 2013 |    11.599 |          18.119 |    7.283 |
        | 2014 |    11.021 |          18.072 |    7.106 |
        | 2015 |    10.6   |          17.885 |    7.043 |
        | 2016 |    10.662 |          17.447 |    7.21  |
        | 2017 |    10.661 |          17.211 |    7.497 |
        | 2018 |    10.437 |          17.551 |    7.539 |
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
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            unemployment_rate = gmdb_model.get_unemployment_rate(
                gmd_dataset=self._gmbd_dataset
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
        | 2018 |          0.0633 |    0.0494 |  0.0341 |
        | 2019 |          0.0549 |    0.0432 |  0.0284 |
        | 2020 |          0.0933 |    0.0413 |  0.0278 |
        | 2021 |          0.1005 |    0.0672 |  0.0258 |
        | 2022 |          0.1164 |    0.0994 |  0.051  |
        | 2023 |          0.0774 |    0.0897 |  0.0584 |
        """
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        if gmdb_source:
            unemployment_rate = gmdb_model.get_unemployment_rate(
                gmd_dataset=self._gmbd_dataset
            )
        else:
            unemployment_rate = oecd_model.get_unemployment_rate(
                period="yearly", start_date=self._start_date, end_date=self._end_date
            )

        inflation_rate = gmdb_model.get_inflation_rate(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics()

        economics.get_labour_productivity(countries=['Bulgaria', 'Croatia', 'Spain'])
        ```

        Which returns:

        |      |   Bulgaria |   Croatia |   Spain |
        |:-----|-----------:|----------:|--------:|
        | 2013 |    26.3805 |   36.169  | 59.2337 |
        | 2014 |    26.5482 |   35.2152 | 59.5158 |
        | 2015 |    27.3489 |   36.3951 | 60.1307 |
        | 2016 |    28.0515 |   37.5651 | 60.3429 |
        | 2017 |    28.3199 |   37.9134 | 60.7889 |
        | 2018 |    28.9871 |   38.9686 | 60.7456 |
        | 2019 |    30.5046 |   39.5761 | 60.8858 |
        | 2020 |    30.7971 |   37.1191 | 60.9134 |
        | 2021 |    32.9483 |   41.2317 | 60.6022 |
        | 2022 |    33.9708 |   43.6926 | 61.3551 |
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
        | 2010 | 127.594 |
        | 2011 | 127.831 |
        | 2012 | 127.552 |
        | 2013 | 127.333 |
        | 2014 | 127.12  |
        | 2015 | 126.978 |
        | 2016 | 126.96  |
        | 2017 | 126.746 |
        | 2018 | 126.495 |
        | 2019 | 126.221 |
        """
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            population_statistics_df = gmdb_model.get_population(
                gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='1980-01-01')

        economics.get_sovereign_debt_crisis(countries='Argentina')
        ```

        Which returns:

        |      |   Argentina |
        |:-----|------------:|
        | 2016 |           0 |
        | 2017 |           0 |
        | 2018 |         nan |
        | 2019 |         nan |
        | 2020 |         nan |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        sovereign_debt_crisis = gmdb_model.get_sovereign_debt_crisis(
            gmd_dataset=self._gmbd_dataset
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

        economics = Economics(start_date='1990-01-01')

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
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        currency_crisis = gmdb_model.get_currency_crisis(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics(start_date='2005-01-01')

        economics.get_banking_crisis(countries=['United States', 'United Kingdom'])
        ```

        Which returns:

        |      |   United Kingdom |   United States |
        |:-----|-----------------:|----------------:|
        | 2016 |                0 |               0 |
        | 2017 |                0 |               0 |
        | 2018 |                0 |               0 |
        | 2019 |                0 |               0 |
        | 2020 |                0 |               0 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        banking_crisis = gmdb_model.get_banking_crisis(gmd_dataset=self._gmbd_dataset)

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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_nonfarm_payrolls()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-02-01 |          158436 |
        | 2026-03-01 |          158650 |
        | 2026-04-01 |          158798 |
        | 2026-05-01 |          158927 |
        | 2026-06-01 |          158984 |
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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_initial_jobless_claims()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-06-27 |          217000 |
        | 2026-07-04 |          217000 |
        | 2026-07-11 |          209000 |
        | 2026-07-18 |          188000 |
        | 2026-07-25 |          197000 |
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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_retail_sales()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-02-01 |          741278 |
        | 2026-03-01 |          754013 |
        | 2026-04-01 |          759097 |
        | 2026-05-01 |          766876 |
        | 2026-06-01 |          768553 |
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

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_industrial_production_index()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-02-01 |         101.926 |
        | 2026-03-01 |         101.617 |
        | 2026-04-01 |         102.42  |
        | 2026-05-01 |         102.561 |
        | 2026-06-01 |         102.639 |
        """
        self._require_fred_api_key()

        industrial_production_index = fred_model.get_industrial_production_index(
            self._start_date, self._end_date, self._fred_api_key
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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_housing_starts()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-02-01 |            1346 |
        | 2026-03-01 |            1522 |
        | 2026-04-01 |            1414 |
        | 2026-05-01 |            1199 |
        | 2026-06-01 |            1427 |
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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_real_personal_income()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-02-01 |         16601.6 |
        | 2026-03-01 |         16598.1 |
        | 2026-04-01 |         16526.5 |
        | 2026-05-01 |         16567   |
        | 2026-06-01 |         16606.1 |
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

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_mortgage_rate_30_year()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-07-02 |          0.0643 |
        | 2026-07-09 |          0.0649 |
        | 2026-07-16 |          0.0655 |
        | 2026-07-23 |          0.0658 |
        | 2026-07-30 |          0.0666 |
        """
        self._require_fred_api_key()

        mortgage_rate_30_year = fred_model.get_mortgage_rate_30_year(
            self._start_date, self._end_date, self._fred_api_key
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

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

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

        economics = Economics(start_date='2020-01-01', fred_api_key='FRED_API_KEY')

        economics.get_recession_indicator()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2026-02-01 |               0 |
        | 2026-03-01 |               0 |
        | 2026-04-01 |               0 |
        | 2026-05-01 |               0 |
        | 2026-06-01 |               0 |
        """
        self._require_fred_api_key()

        recession_indicator = fred_model.get_recession_indicator(
            self._start_date, self._end_date, self._fred_api_key
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
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the quarterly Commercial Real Estate Price Index for the United States
        from FRED (sourced from the IMF's Financial Soundness Indicators).

        This tracks commercial (office, retail, industrial, apartment) property
        prices, as distinct from residential house prices (see `get_house_prices`,
        which tracks a completely different asset class/market). It is a
        transaction-based index rather than the appraisal-smoothed methodology used
        by institutional benchmarks like the NCREIF Property Index -- which is not
        freely available anywhere -- so expect more volatility and less
        autocorrelation than an appraisal-based series would show.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/COMREPUSQ159N

        Also known as: commercial property price index, CRE price index.

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
            quarterly Commercial Real Estate Price Index, as a year-over-year
            percent change expressed as a decimal fraction.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', fred_api_key='FRED_API_KEY')

        economics.get_commercial_real_estate_prices()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2024-04-01 |         -0.1067 |
        | 2024-07-01 |         -0.1058 |
        | 2024-10-01 |         -0.0273 |
        | 2025-01-01 |         -0.0301 |
        | 2025-04-01 |         -0.0701 |
        """
        self._require_fred_api_key()

        commercial_real_estate_prices = fred_model.get_commercial_real_estate_prices(
            self._start_date, self._end_date, self._fred_api_key
        )

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
        )

    @handle_errors
    def get_real_yield_curve(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the daily real (TIPS-implied) U.S. Treasury yield curve from FRED -- the
        Market Yield on Treasury Inflation-Protected Securities at Constant
        Maturity, for the 5, 7, 10, 20 and 30-Year maturities.

        This is genuine market-observed data, as distinct from
        `fixedincome.get_breakeven_inflation_rate`, which is a pure formula applied
        to a hand-specified sample curve rather than real TIPS market data. Use this
        together with `get_breakeven_inflation_expectations` to get the
        market-implied (Q-measure) inflation expectation at each maturity.

        Yields are daily, returned as a decimal fraction per annum (0.0174 for 1.74%) and
        not seasonally adjusted. FRED publishes them in percentage points; they are
        rescaled here so this curve is on the same decimal scale as
        `fixedincome.get_treasury_rates` and can be differenced against it directly. The
        20-Year series starts in July 2004 and the 30-Year in February 2010, so earlier
        dates are NaN for those two maturities.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/DFII10

        Also known as: TIPS yield curve, real Treasury yield curve.

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
            pd.DataFrame: A DataFrame with one column per maturity (5, 7, 10, 20, 30
            Year), as a decimal fraction per annum.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2024-01-01', end_date='2024-01-15', fred_api_key='FRED_API_KEY')

        economics.get_real_yield_curve()
        ```

        Which returns:

        | Date       |   5 Year |   7 Year |   10 Year |   20 Year |   30 Year |
        |:-----------|---------:|---------:|----------:|----------:|----------:|
        | 2024-01-02 |   0.0176 |   0.0175 |    0.0174 |    0.0184 |    0.0191 |
        | 2024-01-03 |   0.0173 |   0.0171 |    0.0171 |    0.0181 |    0.0189 |
        | 2024-01-04 |   0.0179 |   0.0178 |    0.0177 |    0.0188 |    0.0196 |
        | 2024-01-05 |   0.0183 |   0.0183 |    0.0183 |    0.0194 |    0.0202 |
        | 2024-01-08 |   0.0178 |   0.0179 |    0.0179 |    0.019  |    0.0198 |
        """
        self._require_fred_api_key()

        real_yield_curve = fred_model.get_real_yield_curve(
            self._start_date, self._end_date, self._fred_api_key
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
        )

    @handle_errors
    def get_breakeven_inflation_expectations(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Get the daily market-implied (Q-measure) breakeven inflation expectations
        from FRED -- nominal Treasury yield minus real TIPS yield -- at the 5, 7,
        10, 20 and 30-Year maturities, plus the 5-Year, 5-Year Forward Inflation
        Expectation Rate (the market's implied average inflation rate for the five
        years starting five years from now).

        FRED only publishes ready-made daily breakeven series for the 5 and 10-Year
        maturities; its 7, 20 and 30-Year breakeven series only exist at monthly
        frequency, so those three points are instead computed as nominal minus real
        from FRED's own daily Treasury and TIPS series, keeping every maturity on a
        daily frequency. See `get_real_yield_curve` for the underlying real yields
        on their own.

        Rates are daily, returned as a decimal fraction per annum (0.0221 for 2.21%) and
        not seasonally adjusted. FRED publishes them in percentage points; they are
        rescaled here to match the decimal convention used by every other rate surface in
        the toolkit. The 20-Year column starts in July 2004 and the 30-Year in February
        2010, limited by the TIPS leg of the calculation.

        Requires a free FRED API key, see the `fred_api_key` parameter of the
        `Economics` class.

        See definition: https://fred.stlouisfed.org/series/T10YIE

        Also known as: breakeven inflation rate, market-implied inflation expectations.

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
            pd.DataFrame: A DataFrame with one column per maturity (5, 7, 10, 20, 30
            Year) plus the 5-Year, 5-Year Forward Rate, as a decimal fraction per annum.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2024-01-01', end_date='2024-01-15', fred_api_key='FRED_API_KEY')

        economics.get_breakeven_inflation_expectations()
        ```

        Which returns:

        | Date       |   5 Year |   7 Year |   10 Year |   20 Year |   30 Year |   5 Year, 5 Year Forward |
        |:-----------|---------:|---------:|----------:|----------:|----------:|-------------------------:|
        | 2024-01-02 |   0.0217 |   0.022  |    0.0221 |    0.0241 |    0.0217 |                   0.0225 |
        | 2024-01-03 |   0.0217 |   0.0221 |    0.022  |    0.024  |    0.0216 |                   0.0223 |
        | 2024-01-04 |   0.0218 |   0.0221 |    0.0222 |    0.0242 |    0.0217 |                   0.0226 |
        | 2024-01-05 |   0.0219 |   0.0221 |    0.0222 |    0.0243 |    0.0219 |                   0.0225 |
        | 2024-01-08 |   0.0219 |   0.022  |    0.0222 |    0.0243 |    0.0219 |                   0.0225 |
        """
        self._require_fred_api_key()

        breakeven_inflation_expectations = (
            fred_model.get_breakeven_inflation_expectations(
                self._start_date, self._end_date, self._fred_api_key
            )
        )

        return finalize_dataset(
            dataset=breakeven_inflation_expectations,
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

        | Date       |   2026-09 |   2026-10 |   2026-11 |   2026-12 |   2027-01 |
        |:-----------|----------:|----------:|----------:|----------:|----------:|
        | 2026-07-27 |     82.61 |     80.25 |     78.17 |     76.53 |     75.31 |
        | 2026-07-28 |     79.26 |     77.17 |     75.33 |     73.85 |     72.74 |
        | 2026-07-29 |     84.46 |     82.04 |     79.68 |     77.74 |     76.28 |
        | 2026-07-30 |     83.59 |     80.8  |     78.19 |     76.12 |     74.65 |
        | 2026-07-31 |     84.67 |     81.49 |     78.65 |     76.44 |     74.88 |
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
