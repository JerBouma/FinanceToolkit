"""Economic Activity Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit import helpers
from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    check_period_type,
)
from financetoolkit.economics.sources import (
    bea_model,
    census_model,
    fmp_model as economics_fmp_model,
    frb_model,
    fred_model,
    nber_model,
    nyfed_model,
    oecd_model,
    portwatch_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import (
    track_sources,
)
from financetoolkit.utilities.statistics_model import (
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class ActivityEconomics(EconomicsBase):
    """
    Activity and sentiment: consumer and business confidence, the composite leading indicator, share prices, retail sales, industrial production, housing starts, real personal income, the recession indicator, supply chain pressure, chokepoint transits and the economic calendar.
    """

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
        show_columns: str | list[str] | None = None,
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
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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

        The CLI is read for its direction and turning points rather than its level: a reading
        below 100 that is still rising signals a recovery ahead, one above 100 that is falling
        a slowdown. The latest months are estimated from incomplete data and are revised,
        sometimes enough to change the signal, so a turning point is usually only confirmed
        after several months. It is used alongside business confidence and PMIs to anticipate
        changes in GDP growth by about six to nine months.

        See definition: https://data.oecd.org/leadind/composite-leading-indicator-cli.htm

        Also known as: CLI, leading economic indicator.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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

        This is an index based on 2015 = 100.

        See definition: https://data.oecd.org/price/share-prices.htm

        Also known as: stock market index, equity index, market performance.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which is quarterly when the Economics class was initialized with
                quarterly=True and monthly otherwise.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Share Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2013-01-01", end_date="2025-12-31")

        economics.get_share_prices(countries=['Turkey', 'Belgium', 'Australia'], period='yearly')
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

        period = self._default_period(period, "monthly")

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
    def get_retail_sales(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get Advance Retail Sales (Retail and Food Services) for the United States from
        the Census Bureau.

        Retail Sales measures nominal spending at retail and food-service
        establishments. Since Personal Consumption Expenditures make up roughly
        two-thirds to three-quarters of US GDP, this monthly, high-frequency series is
        a core input to real-time (nowcast) GDP estimates such as the Federal Reserve
        Bank of Atlanta's GDPNow.

        The series is monthly, in millions of US dollars, seasonally adjusted, and covers
        retail trade and food services. It is nominal, so growth=True mixes volume and price
        changes together; compare against the Consumer Price Index to separate the two.

        Without a FRED API key the data comes from the Census Bureau directly, which gives
        the same figures. When a FRED API key is set (see the `fred_api_key` parameter of
        the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/RSAFS

        Also known as: retail trade, consumer spending (proxy).

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of total retail
            and food services sales, in millions of dollars.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
            end_date="2025-12-31",
        )

        economics.get_retail_sales()
        ```

        Which returns:

        | Date       |   United States |
        |:-----------|----------------:|
        | 2025-08-01 |          700407 |
        | 2025-09-01 |          700948 |
        | 2025-10-01 |          700790 |
        | 2025-11-01 |          705453 |
        | 2025-12-01 |          706264 |
        """
        # FRED republishes these figures, so it is used when a key is set and the
        # publisher otherwise.
        retail_sales = (
            fred_model.get_retail_sales(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else census_model.get_retail_sales()
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
        show_columns: str | list[str] | None = None,
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

        Without a FRED API key the data comes from the Federal Reserve's G.17 release
        directly, which gives the same figures. When a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/INDPRO

        Also known as: IP index, industrial output.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get Housing Starts (Total New Privately-Owned Housing Units Started) for the
        United States from the Census Bureau.

        Housing Starts counts the number of new residential construction projects
        that have begun in a given month. Residential investment is one of the most
        interest-rate-sensitive components of GDP, and construction activity leads
        the broader business cycle (it typically turns down before a recession and
        turns up before a recovery), making Housing Starts one of the ten components
        of the Conference Board's Leading Economic Index.

        Without a FRED API key the data comes from the Census Bureau directly, which gives
        the same figures. When a FRED API key is set (see the `fred_api_key` parameter of
        the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/HOUST

        Also known as: new residential construction.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of new housing
            starts, in thousands of units, seasonally adjusted annual rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
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
        # FRED republishes these figures, so it is used when a key is set and the
        # publisher otherwise.
        housing_starts = (
            fred_model.get_housing_starts(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else census_model.get_housing_starts()
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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get Real Personal Income Excluding Current Transfer Receipts for the United
        States from the Bureau of Economic Analysis.

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

        Without a FRED API key the data comes from the Bureau of Economic Analysis
        directly, which gives the same figures. When a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/W875RX1

        Also known as: RPI less transfers, NBER real income indicator.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A single-column ("United States") DataFrame of real personal
            income excluding current transfer receipts, in billions of chained 2017
            dollars.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
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
        # FRED republishes these figures, so it is used when a key is set and the
        # publisher otherwise.
        real_personal_income = (
            fred_model.get_real_personal_income(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else bea_model.get_real_personal_income()
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
    def get_recession_indicator(
        self,
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
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

        Without a FRED API key the data comes from the NBER's business cycle dates
        directly, built the way FRED builds USREC, which gives the same figures. When a
        FRED API key is set (see the `fred_api_key` parameter of the `Economics` class), it
        comes from FRED.

        See definition: https://fred.stlouisfed.org/series/USREC

        Also known as: USREC, NBER recession dummy, business cycle indicator.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over
            trailing periods. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
    def get_supply_chain_pressure_index(
        self,
        rolling: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the Global Supply Chain Pressure Index (GSCPI) of the Federal Reserve Bank of New
        York, monthly from 1998: one measure of the pressure on global supply chains,
        combining 27 series. Shipping costs come from the Baltic Dry Index, the Harpex index
        of container ship charter rates and the US Bureau of Labor Statistics air freight
        price indices; the rest are the delivery times, backlogs and purchased stocks of the
        manufacturing purchasing managers' surveys of the euro area, China, Japan, South
        Korea, Taiwan, the United Kingdom and the United States. The index peaked in December
        2021 at more than four standard deviations, when the reopening after COVID-19 met
        congested ports, and is a common leading indicator of goods inflation.

        Benigno, G., di Giovanni, J., Groen, J. J. J., & Noble, A. I. (2022). The GSCPI: A New
        Barometer of Global Supply Chain Pressures. Federal Reserve Bank of New York Staff
        Reports, 1017. https://www.newyorkfed.org/research/staff_reports/sr1017

        The index is the first principal component of the series after removing the part
        explained by demand, measured in standard deviations from its average: zero is the
        average pressure and 1 is one standard deviation above it. It is not a rate, so it is
        not a decimal fraction. The New York Fed re-estimates the whole
        history with every release, so past months can change slightly.

        See definition: https://www.newyorkfed.org/research/policy/gscpi

        Also known as: GSCPI, supply chain pressure, supply chain index, shipping and delivery
        pressure.

        Args:
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The index in standard deviations from its average, indexed by month
            with a "World" column.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-05-01', end_date='2026-10-31')

        economics.get_supply_chain_pressure_index()
        ```

        Which returns:

        |         |   World |
        |:--------|--------:|
        | 2026-05 |  1.8017 |
        | 2026-06 |  1.2026 |
        | 2026-07 |  0.9511 |
        | 2026-08 |  1.271  |
        | 2026-09 |  1.3612 |
        | 2026-10 |  1.2619 |
        """
        supply_chain_pressure = nyfed_model.get_supply_chain_pressure_index()

        return finalize_dataset(
            dataset=supply_chain_pressure,
            indicator_name="Supply Chain Pressure Index",
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
    def get_chokepoint_transits(
        self,
        chokepoints: str | list[str] | None = None,
        measure: str = "transits",
        period: str = "daily",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Retrieves IMF PortWatch's daily ship transits through the world's maritime
        chokepoints from 2019, such as the Strait of Hormuz, the Suez Canal, the Bab
        el-Mandeb Strait, the Panama Canal, the Malacca Strait and the Bosporus, 28 in all.
        For each chokepoint and day it gives the number of ships that passed it (transits)
        or the estimated cargo they carried (trade volume, in metric tons), by ship
        category: Container, Dry Bulk, General Cargo, Ro-Ro and Tanker, with Cargo (all
        but tankers) and Total.

        The transits show disruptions to trade as they happen, weeks before trade
        statistics are published: tankers through the Strait of Hormuz fell from 36 on 28
        February 2026 to 5 on 1 March, and when ships avoided the Red Sea, the monthly
        transits of the Bab el-Mandeb Strait fell from 2,438 in October 2023 to 1,013 in
        February 2024 while those around the Cape of Good Hope rose from 1,561 to 2,487.

        PortWatch reads the transits from the Automatic Identification System (AIS)
        signals ships broadcast, as collected by satellites for the UN Global Platform. A
        ship is counted when it crosses a chokepoint's boundary, once per transit and not
        again within 48 hours. Its trade volume is its deadweight tonnage (the most it can
        carry) times its load factor, which PortWatch estimates from the ship's draft and
        dimensions. It is an estimate: on a day with few transits it can be zero while
        ships passed.

        With period, the days are summed by week (Monday to Sunday), month, quarter or
        year. Only complete periods are returned, since PortWatch adds the latest days
        once a week.

        Also known as: shipping traffic, strait of Hormuz tanker traffic, Suez Canal
        transits, Red Sea shipping, maritime trade disruptions, oil flows.

        Args:
            chokepoints (str | list[str] | None, optional): The chokepoints, e.g. "Strait
                of Hormuz" or ["Suez Canal", "Cape of Good Hope"], in any casing; a part of
                a name that identifies one chokepoint, such as "Hormuz" or "Suez", works
                too. Defaults to None, which returns all of them.
            measure (str, optional): "transits" for the number of ships or "trade_volume"
                for the estimated cargo they carried in metric tons. Defaults to
                "transits".
            period (str, optional): "daily", "weekly", "monthly", "quarterly" or "yearly".
                Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            growth (bool, optional): Whether to calculate the growth. Defaults to False.
            lag (int, optional): The lag in periods for the growth, e.g. 12 with monthly
                data for the change over a year. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result.
                Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The transits or trade volumes by day (or period), with a column
            per ship category, and per chokepoint and ship category when there are several
            chokepoints.

        Notes:
        - Sources: UN Global Platform; IMF PortWatch (portwatch.imf.org).
        - Arslanalp, S., Koepke, R. and Verschuur, J. (2021). "Tracking Trade from Space:
        An Application to Pacific Island Countries." IMF Working Paper No. 2021/225.
        - The days are in Coordinated Universal Time (UTC).

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2026-02-24", end_date="2026-03-03")

        economics.get_chokepoint_transits(chokepoints="Strait of Hormuz", show_columns=["Tanker", "Cargo", "Total"])
        ```

        Which returns:

        |            |   Tanker |   Cargo |   Total |
        |:-----------|---------:|--------:|--------:|
        | 2026-02-24 |       49 |      44 |      93 |
        | 2026-02-25 |       59 |      57 |     116 |
        | 2026-02-26 |       45 |      44 |      89 |
        | 2026-02-27 |       37 |      27 |      64 |
        | 2026-02-28 |       36 |      21 |      57 |
        | 2026-03-01 |        5 |      15 |      20 |
        | 2026-03-02 |        1 |       3 |       4 |
        | 2026-03-03 |        1 |       3 |       4 |
        """
        if measure not in ("transits", "trade_volume"):
            raise ValueError(
                f"measure='{measure}' is not available. Choose 'transits' or "
                "'trade_volume'."
            )

        frequencies = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q",
            "yearly": "Y",
        }

        if period not in frequencies:
            raise ValueError(
                f"period='{period}' is not available. Choose from "
                f"{', '.join(frequencies)}."
            )

        available = portwatch_model.get_chokepoint_names()

        if not available:
            return pd.DataFrame()

        selected: list[str] = []

        # A chokepoint is found by its name, else by a part of it that matches one name.
        for requested in (
            available if chokepoints is None else helpers.split_values(chokepoints)
        ):
            exact = [name for name in available if name.lower() == requested.lower()]
            partial = [name for name in available if requested.lower() in name.lower()]
            matches = exact or partial

            if len(matches) == 1:
                if matches[0] not in selected:
                    selected.append(matches[0])
            else:
                logger.warning(
                    "'%s' %s. The chokepoints are: %s.",
                    requested,
                    "matches several chokepoints" if matches else "is not a chokepoint",
                    ", ".join(available),
                )

        if not selected:
            return pd.DataFrame()

        prefix = "Transits " if measure == "transits" else "Trade Volume "
        transits = {}

        for name, data in zip(
            selected,
            helpers.run_in_parallel(
                portwatch_model.get_chokepoint_transits,
                [(name,) for name in selected],
            ),
            strict=True,
        ):
            if not data.empty:
                columns = [
                    column for column in data.columns if column.startswith(prefix)
                ]
                transits[name] = data[columns].rename(
                    columns=lambda column: column.removeprefix(prefix)
                )

        if not transits:
            return pd.DataFrame()

        dataset = pd.concat(transits, axis=1).sort_index()

        if period != "daily":
            # Only periods whose last day is in the data are complete.
            last_day = dataset.index[-1].to_timestamp()
            dataset = dataset.groupby(dataset.index.asfreq(frequencies[period])).sum(
                min_count=1
            )
            dataset = dataset[
                dataset.index.to_timestamp(how="end").normalize() <= last_day
            ]

        if len(selected) == 1:
            dataset = dataset.droplevel(0, axis=1)

        return finalize_dataset(
            dataset=dataset,
            start_date=self._start_date,
            end_date=self._end_date,
            default_rounding=self._rounding,
            indicator_name="Chokepoint Transits",
            rounding=rounding,
            growth=growth,
            lag=lag,
            standardize=standardize,
            axis="rows",
            row_slice=True,
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
        as_series: bool = False,
        show_columns: str | list[str] | None = None,
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
        labels "%" as well. The "Currency" column is the currency of the market a release
        concerns, not the unit of its value: China's balance of trade is in US dollars
        while its row shows CNY (the yuan figure is the separate "Balance of Trade Yuan").
        The "Previous" value is the source's own and can differ from the Actual of the
        release before it, e.g. after a revision.

        The calendar comes from FinancialModelingPrep and starts in 2013 for most
        countries; the first release of a series is the first period of its as_series
        answer.

        With as_series=True the releases become time series: the Actual value of every
        event by the period it covers, with a column per country and event and the
        consensus Estimate next to it ("<event> Estimate"). The calendar only names the
        period in the event, e.g. "Producer Price Index YoY (Oct)", so the period is read
        from it: a month, a quarter "(Q3)", the day a weekly figure ends "(Oct/10)", a year
        "(2026)" or, for an event without one (an interest rate decision), the day of the
        release. The year is that of the release, or the year before when the period starts
        after the release month (a December figure released in January). Several releases of
        the same period, such as a flash estimate and the final figure, keep the latest one
        published. A figure published for two consecutive periods on the same day, such as
        China's January and February trade figures, is one release of both and is kept once,
        under the later period (February), leaving January empty; a release without a
        period, of an event that otherwise has one, is left out. The index is monthly for
        monthly events, quarterly for quarterly ones and
        daily, at the last day of each period, when events of different frequencies are
        combined. The series can be analysed further with `Econometrics(data=...)`, e.g. how
        unusual the latest value is (get_level_significance) or how long it has been above or
        below zero (get_streaks).

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
            as_series (bool, optional): Whether to return the Actual and Estimate of every
                event as time series by the period they cover, with (Country, Event) columns,
                instead of the releases by date. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The economic data releases, indexed by date, or with as_series=True
            the values of every event by the period they cover.

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
        helpers.require_api_key(self._api_key, "the economic calendar")

        economic_calendar = economics_fmp_model.get_economic_calendar(
            api_key=self._api_key,
            start_date=start_date or self._requested_start_date,
            end_date=end_date or self._requested_end_date,
            countries=countries,
            currencies=currencies,
            impact=impact,
            events=events,
        )

        if as_series:
            return economics_fmp_model.get_economic_calendar_series(economic_calendar)

        return economic_calendar
