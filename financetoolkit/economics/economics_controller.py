"""Economics Module"""
__docformat__ = "google"


import re
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit.economics import oecd_model
from financetoolkit.helpers import calculate_growth, handle_errors

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
        quarterly: bool | None = None,
        rounding: int | None = 4,
    ):
        """
        Initializes the Economics Controller Class.

        Args:
            quarterly (bool | None, optional): Parameter that defines if the default data returned is quarterly
            or yearly. Defaults to None.
            start_date (str | None, optional): The start date to retrieve data from. Defaults to None.
            end_date (str | None, optional): The end date to retrieve data from. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "ASML"])

        cpi = toolkit.economics.get_consumer_price_index(period='yearly')

        cpi.loc['2015':, ['United States', 'Netherlands', 'Japan']]
        ```

        Which returns:

        |      |   United States |   Netherlands |    Japan |
        |:-----|----------------:|--------------:|---------:|
        | 2015 |         100     |       100     | 100      |
        | 2016 |         101.262 |       100.317 |  99.8727 |
        | 2017 |         103.419 |       101.703 | 100.356  |
        | 2018 |         105.945 |       103.435 | 101.349  |
        | 2019 |         107.865 |       106.159 | 101.824  |
        | 2020 |         109.195 |       107.51  | 101.799  |
        | 2021 |         114.325 |       110.387 | 101.561  |
        | 2022 |         123.474 |       121.427 | 104.098  |
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

        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding

    @handle_errors
    def get_gross_domestic_product(
        self,
        inflation_adjusted: bool = False,
        per_capita: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Gross Domestic Product for a variety of countries over
        time from the OECD. The Gross Domestic Product is the total value
        of goods produced and services provided in a country during one year.

        The data is available in two forms: compared to the previous year's value or
        compared to the previous period. The year on year data is the GDP compared to
        the same quarter in the previous year. The quarter on quarter data is the GDP
        compared to the previous quarter.

        See definition: https://data.oecd.org/gdp/gross-domestic-product-gdp.htm

        Args:
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data or the nominal data.
            per_capita (bool, optional): Whether to return the per capita data or the total data.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2021-01-01')

        real_gdp = economics.get_gross_domestic_product(inflation_adjusted=True)

        real_gdp.loc[:, ['Netherlands', 'Germany', 'China']]
        ```

        Which returns:

        |      |   Netherlands |     Germany |       China |
        |:-----|--------------:|------------:|------------:|
        | 2015 |        851994 | 3.88475e+06 | 1.77968e+07 |
        | 2016 |        870238 | 3.96797e+06 | 1.9007e+07  |
        | 2017 |        896473 | 4.08338e+06 | 2.03184e+07 |
        | 2018 |        917246 | 4.13627e+06 | 2.16798e+07 |
        | 2019 |        932198 | 4.16067e+06 | 2.29806e+07 |
        | 2020 |        897261 | 3.94717e+06 | 2.35091e+07 |
        | 2021 |        921282 | 4.07756e+06 | 2.55147e+07 |
        """
        if inflation_adjusted:
            gross_domestic_product = (
                oecd_model.get_gross_domestic_product_forecast_long_term()
            )
        else:
            gross_domestic_product = oecd_model.get_annual_gross_domestic_product(
                per_capita=per_capita
            )

        gross_domestic_product = gross_domestic_product.loc[
            self._start_date : self._end_date
        ]

        return gross_domestic_product.round(rounding if rounding else self._rounding)

    def get_gross_domestic_product_growth(
        self, quarterly: bool | None = None, rounding: int | None = None
    ):
        """
        Get the Gross Domestic Product growth rate for a variety of countries over
        time from the OECD. The Gross Domestic Product is the total value
        of goods produced and services provided in a country during one year.

        It is possible to view the growth rate on a quarterly or annual basis, the
        default is dependent on the quarterly parameter. The growth rate is the
        percentage change in the GDP compared to the previous period.

        See definition: https://data.oecd.org/gdp/quarterly-gdp.htm

        Args:
            quarterly (bool, optional): Whether to return the quarterly data or the annual data.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product growth rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-01-01', end_date='2022-01-01')

        gdp_growth = economics.get_gross_domestic_product_growth(quarterly=True)

        gdp_growth.loc[:, ['United Kingdom', 'United States', 'Belgium']]
        ```

        Which returns:

        |        |   United Kingdom |   United States |   Belgium |
        |:-------|-----------------:|----------------:|----------:|
        | 2021Q1 |          -0.0102 |          0.0129 |    0.0181 |
        | 2021Q2 |           0.0733 |          0.0152 |    0.0193 |
        | 2021Q3 |           0.0172 |          0.0081 |    0.0219 |
        | 2021Q4 |           0.0152 |          0.017  |    0.0076 |
        | 2022Q1 |           0.0053 |         -0.005  |    0.0012 |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly

        if quarterly:
            growth_gdp = oecd_model.get_quarterly_gross_domestic_product(
                year_on_year=False
            )
        else:
            growth_gdp = calculate_growth(
                oecd_model.get_annual_gross_domestic_product(per_capita=False),
                axis="rows",
            )

        growth_gdp = growth_gdp.loc[self._start_date : self._end_date]

        return growth_gdp.round(rounding if rounding else self._rounding)

    def get_gross_domestic_product_forecast(
        self,
        quarterly: bool = False,
        inflation_adjusted: bool = True,
        long_term: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Gross Domestic Product growth rate for a variety of countries over
        time from the OECD. The Gross Domestic Product is the total value
        of goods produced and services provided in a country during one year.

        It is possible to view the growth rate on a quarterly or annual basis, the
        default is dependent on the quarterly parameter. The growth rate is the
        percentage change in the GDP compared to the previous period.

        See definition: https://data.oecd.org/gdp/real-gdp-long-term-forecast.htm

        Args:
            quarterly (bool, optional): Whether to return the quarterly data or the annual data.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data or the nominal data.
            long_term (bool, optional): Whether to return the long term forecast or the short term forecast.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product forecast growth rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-01-01')

        gdp_growth_forecast = economics.get_gross_domestic_product_forecast(long_term=False)

        gdp_growth_forecast.loc[:, ['Indonesia', 'China', 'India']]
        ```

        Which returns:

        |      |   Indonesia |   China |   India |
        |:-----|------------:|--------:|--------:|
        | 2021 |      0.037  |  0.0845 |  0.0905 |
        | 2022 |      0.0531 |  0.0299 |  0.0724 |
        | 2023 |      0.0488 |  0.0516 |  0.0626 |
        | 2024 |      0.0519 |  0.047  |  0.0606 |
        | 2025 |      0.0519 |  0.0424 |  0.0648 |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly

        if long_term:
            if quarterly:
                print("Only annual long term forecasts are available.")
            if not inflation_adjusted:
                print("Only inflation adjusted long term forecasts are available.")

            forecast_gdp = oecd_model.get_gross_domestic_product_forecast_long_term()

            forecast_gdp_growth = calculate_growth(forecast_gdp, axis="rows")
        else:
            forecast_gdp_growth = (
                oecd_model.get_gross_domestic_product_forecast_short_term(
                    quarterly=quarterly, inflation_adjusted=inflation_adjusted
                )
            )

        forecast_gdp_growth = forecast_gdp_growth.loc[self._start_date :]

        return forecast_gdp_growth.round(rounding if rounding else self._rounding)

    def get_consumer_confidence_index(
        self, growth: bool = False, lag: int = 1, rounding: int | None = None
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

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Consumer Confidence Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2008-09-01', end_date='2009-03-01')

        consumer_confidence_index = economics.get_consumer_confidence_index()

        consumer_confidence_index.loc[:, ['Germany', 'France', 'Portugal']]
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
        consumer_confidence_index = oecd_model.get_consumer_confidence_index()

        if growth:
            consumer_confidence_index = calculate_growth(
                consumer_confidence_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        consumer_confidence_index = consumer_confidence_index.loc[
            self._start_date : self._end_date
        ]

        return consumer_confidence_index.round(rounding if rounding else self._rounding)

    def get_business_confidence_index(
        self, growth: bool = False, lag: int = 1, rounding: int | None = None
    ):
        """
        This business confidence indicator provides information on future developments,
        based upon opinion surveys on developments in production, orders and stocks of
        finished goods in the industry sector. It can be used to monitor output growth
        and to anticipate turning points in economic activity.

        Numbers above 100 suggest an increased confidence in near future business
        performance, and numbers below 100 indicate pessimism towards future performance.

        See definition: https://data.oecd.org/leadind/business-confidence-index-bci.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Business Confidence Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2022-09-01', end_date='2023-03-01')

        business_confidence_index = economics.get_business_confidence_index()

        business_confidence_index.loc[:, ['Brazil', 'Canada', 'Costa Rica']]
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
        business_confidence_index = oecd_model.get_business_confidence_index()

        if growth:
            business_confidence_index = calculate_growth(
                business_confidence_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        business_confidence_index = business_confidence_index.loc[
            self._start_date : self._end_date
        ]

        return business_confidence_index.round(rounding if rounding else self._rounding)

    def get_composite_leading_indicator(
        self, growth: bool = False, lag: int = 1, rounding: int | None = None
    ):
        """
        The composite leading indicator (CLI) is designed to provide early signals
        of turning points in business cycles showing fluctuation of the economic
        activity around its long term potential level. CLIs show short-term economic
        movements in qualitative rather than quantitative terms.

        See definition: https://data.oecd.org/leadind/composite-leading-indicator-cli.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Composite Leading Indicator.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-06-01', end_date='2023-12-01')

        composite_leading_indicator = economics.get_composite_leading_indicator()

        composite_leading_indicator.loc[:, ['United States', 'United Kingdom', 'Japan']]
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
        composite_leading_indicator = oecd_model.get_composite_leading_indicator()

        if growth:
            composite_leading_indicator = calculate_growth(
                composite_leading_indicator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        composite_leading_indicator = composite_leading_indicator.loc[
            self._start_date : self._end_date
        ]

        return composite_leading_indicator.round(
            rounding if rounding else self._rounding
        )

    def get_consumer_price_index(
        self,
        period: str | None = None,
        inflation_type: str = "total",
        forecast: bool = False,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Inflation measured by consumer price index (CPI) is defined as the change in the
        prices of a basket of goods and services that are typically purchased by specific
        groups of households. Inflation is measured in terms an index, 2015 base year
        with a breakdown for food, energy and total.

        Inflation measures the erosion of living standards. A consumer price index is
        estimated as a series of summary measures of the period-to-period proportional
        change in the prices of a fixed set of consumer goods and services of constant
        quantity and characteristics, acquired, used or paid for by the reference
        population.

        Each summary measure is constructed as a weighted average of a large number of
        elementary aggregate indices. Each of the elementary aggregate indices is estimated
        using a sample of prices for a defined set of goods and services obtained in, or
        by residents of, a specific region from a given set of outlets or other sources of
        consumption goods and services.

        This indicator uses index 2015 = 100.

        See definition: https://data.oecd.org/price/inflation-cpi.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Consumer Price Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-12-31')

        consumer_price_index = economics.get_consumer_price_index()

        consumer_price_index.loc[:, ['United States', 'South Africa', 'China']]
        ```

        Which returns:

        |      |   United States |   South Africa |   China |
        |:-----|----------------:|---------------:|--------:|
        | 2015 |         100     |        100     | 100     |
        | 2016 |         101.262 |        106.571 | 102     |
        | 2017 |         103.419 |        112.096 | 103.625 |
        | 2018 |         105.945 |        117.16  | 105.775 |
        | 2019 |         107.865 |        121.987 | 108.842 |
        | 2020 |         109.195 |        125.903 | 111.475 |
        | 2021 |         114.325 |        131.709 | 112.569 |
        | 2022 |         123.474 |        140.981 | 114.79  |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        if forecast:
            consumer_price_index = oecd_model.get_consumer_price_index_forecast(
                period=period
            )

            consumer_price_index = consumer_price_index.loc[self._start_date :]

            return consumer_price_index.round(rounding if rounding else self._rounding)

        consumer_price_index = oecd_model.get_consumer_price_index(
            period=period, inflation_type=inflation_type
        )

        if growth:
            consumer_price_index = calculate_growth(
                consumer_price_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        consumer_price_index = consumer_price_index.loc[
            self._start_date : self._end_date
        ]

        return consumer_price_index.round(rounding if rounding else self._rounding)

    def get_producer_price_index(
        self,
        period: str | None = None,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Producer price indices in manufacturing measure the rate of change in prices
        of products sold as they leave the producer. They exclude any taxes, transport
        and trade margins that the purchaser may have to pay. PPIs provide measures of
        average movements of prices received by the producers of various commodities.
        hey are often seen as advanced indicators of price changes throughout the economy,
        including changes in the prices of consumer goods and services.

        Manufacturing covers the production of semi-processed goods and other intermediate
        goods as well as final products such as consumer goods and capital equipment.
        A variety of price indices may be used to measure inflation in an economy. These
        include consumer price indices (CPI), price indices relating to specific goods and/or
        services, GDP deflators and producer price indices (PPI).

        This indicator is presented for total industry and uses index 2015 = 100.

        See definition: https://data.oecd.org/price/producer-price-indices-ppi.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Consumer Price Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2020-01-01', end_date='2023-12-31')

        producer_price_index = economics.get_producer_price_index(period='quarterly')

        producer_price_index.loc[:, ['Germany', 'Poland', 'Belgium']]
        ```

        Which returns:

        |        |   Germany |   Poland |   Belgium |
        |:-------|----------:|---------:|----------:|
        | 2020Q1 |   103.9   |  105.193 |   110.167 |
        | 2020Q2 |   102.9   |  103.926 |   105.1   |
        | 2020Q3 |   102.833 |  104.026 |   107.9   |
        | 2020Q4 |   102.967 |  104.693 |   108.233 |
        | 2021Q1 |   104.567 |  107.26  |   112.9   |
        | 2021Q2 |   106.9   |  110.594 |   118.233 |
        | 2021Q3 |   109.567 |  114.362 |   123.033 |
        | 2021Q4 |   111.933 |  118.663 |   130.3   |
        | 2022Q1 |   117     |  124.298 |   138.433 |
        | 2022Q2 |   123     |  135     |   148.533 |
        | 2022Q3 |   125.2   |  137.534 |   150.167 |
        | 2022Q4 |   125.8   |  138.601 |   149.933 |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        producer_price_index = oecd_model.get_producer_price_index(period=period)

        if growth:
            producer_price_index = calculate_growth(
                producer_price_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        producer_price_index = producer_price_index.loc[
            self._start_date : self._end_date
        ]

        return producer_price_index.round(rounding if rounding else self._rounding)

    def get_house_prices(
        self,
        quarterly: bool | None = None,
        inflation_adjusted: bool = False,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        In most cases, the nominal house price index covers the sales of newly-built
        and existing dwellings, following the recommendations from the RPPI (Residential
        Property Prices Indices) manual.

        The real house price index is given by the ratio of the nominal house price index
        to the consumers’ expenditure deflator in each country from the OECD national
        accounts database. Both indices are seasonally adjusted.

        Both are based on an 2015 = 100 as an index.

        See definition: https://data.oecd.org/price/housing-prices.htm

        Args:
            quarterly (bool | None, optional): Whether to return the quarterly data or the annual data.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data or the nominal data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the House Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-12-31')

        real_house_prices = economics.get_house_prices(quarterly=False, inflation_adjusted=True)

        real_house_prices.loc[:, ['Japan', 'Netherlands', 'Ireland']]
        ```

        Which returns:

        |      |   Japan |   Netherlands |   Ireland |
        |:-----|--------:|--------------:|----------:|
        | 2015 | 100     |       100     |   100     |
        | 2016 | 102.559 |       104.447 |   106.77  |
        | 2017 | 104.762 |       110.795 |   116.608 |
        | 2018 | 106.054 |       118.658 |   126.275 |
        | 2019 | 107.256 |       124.074 |   126.897 |
        | 2020 | 106.991 |       131.814 |   126.311 |
        | 2021 | 112.714 |       147.149 |   131.669 |
        | 2022 | 118.827 |       156.422 |   138.298 |
        """
        quarterly = quarterly if quarterly is not None else self._quarterly

        house_prices = oecd_model.get_house_prices(
            quarterly=quarterly, inflation_adjusted=inflation_adjusted
        )

        if growth:
            house_prices = calculate_growth(
                house_prices,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        house_prices = house_prices.loc[self._start_date : self._end_date]

        return house_prices.round(rounding if rounding else self._rounding)

    def get_rent_prices(
        self,
        quarterly: bool | None = None,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        The price to rent ratio is the nominal house price index divided by
        the housing rent price index and can be considered as a measure of
        the profitability of house ownership.

        This is based on an 2015 = 100 as an index.

        See definition: https://data.oecd.org/price/housing-prices.htm

        Args:
            quarterly (bool | None, optional): Whether to return the quarterly data or the annual data.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data or the nominal data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the House Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-12-31')

        rent_prices = economics.get_rent_prices(quarterly=False)

        rent_prices.loc[:, ['Turkey', 'United States', 'United Kingdom']]
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

        rent_prices = oecd_model.get_rent_prices(quarterly=quarterly)

        if growth:
            rent_prices = calculate_growth(
                rent_prices,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        rent_prices = rent_prices.loc[self._start_date : self._end_date]

        return rent_prices.round(rounding if rounding else self._rounding)

    def get_share_prices(
        self,
        period: str | None = None,
        growth: bool = False,
        lag: int = 1,
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

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exchange Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics()

        share_prices = economics.get_share_prices()

        share_prices.loc[:, ['Turkey', 'Belgium', 'Australia']]
        ```

        Which returns:

        |      |   Turkey |   Belgium |   Australia |
        |:-----|---------:|----------:|------------:|
        | 2013 |  96.6029 |   74.3936 |     92.3054 |
        | 2014 |  93.2354 |   87.8382 |     98.611  |
        | 2015 | 100      |  100      |    100      |
        | 2016 |  95.6644 |   95.2324 |     96.0699 |
        | 2017 | 122.746  |  101.514  |    105.648  |
        | 2018 | 126.263  |   96.5515 |    109.205  |
        | 2019 | 123.056  |   92.6847 |    117.326  |
        | 2020 | 140.511  |   77.8758 |    111.188  |
        | 2021 | 187.146  |   91.6789 |    130.475  |
        | 2022 | 369.298  |   93.0484 |    128.367  |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        share_prices = oecd_model.get_share_prices(period=period)

        if growth:
            share_prices = calculate_growth(
                share_prices,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        share_prices = share_prices.loc[self._start_date : self._end_date]

        return share_prices.round(rounding if rounding else self._rounding)

    def get_long_term_interest_rate(
        self,
        period: str | None = None,
        forecast: bool = False,
        growth: bool = False,
        lag: int = 1,
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

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Long Term Interest Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-05-01', end_date='2023-12-31')

        long_term_interest_rate = economics.get_long_term_interest_rate(period='monthly')

        long_term_interest_rate.loc[:, ['Japan', 'United States', 'Brazil']]
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
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        long_term_interest_rate = oecd_model.get_long_term_interest_rate(
            period=period, forecast=forecast
        )

        if growth:
            long_term_interest_rate = calculate_growth(
                long_term_interest_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        long_term_interest_rate = long_term_interest_rate.loc[
            self._start_date : None if forecast else self._end_date
        ]

        return long_term_interest_rate.round(rounding if rounding else self._rounding)

    def get_short_term_interest_rate(
        self,
        period: str | None = None,
        forecast: bool = False,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Short-term interest rates are the rates at which short-term borrowings are
        effected between financial institutions or the rate at which short-term government
        paper is issued or traded in the market. Short-term interest rates are generally
        averages of daily rates, measured as a percentage.

        Short-term interest rates are based on three-month money market rates where available.
        Typical standardised names are "money market rate" and "treasury bill rate".

        See definition: https://data.oecd.org/interest/short-term-interest-rates.htm

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Short Term Interest Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-05-01')

        short_term_interest_rate = economics.get_short_term_interest_rate(period='quarterly', forecast=True)

        short_term_interest_rate.loc[:, ['Japan', 'United States', 'China']]
        ```

        Which returns:

        |        |   Japan |   United States |   China |
        |:-------|--------:|----------------:|--------:|
        | 2023Q2 | -0.0003 |          0.0513 |  0.0435 |
        | 2023Q3 | -0.0003 |          0.0543 |  0.0435 |
        | 2023Q4 | -0.0003 |          0.0543 |  0.0435 |
        | 2024Q1 |  0.0007 |          0.0536 |  0.0435 |
        | 2024Q2 |  0.0017 |          0.0513 |  0.043  |
        | 2024Q3 |  0.0027 |          0.0488 |  0.043  |
        | 2024Q4 |  0.0037 |          0.0468 |  0.0425 |
        | 2025Q1 |  0.0047 |          0.0448 |  0.0425 |
        | 2025Q2 |  0.0057 |          0.0423 |  0.0425 |
        | 2025Q3 |  0.0067 |          0.0408 |  0.0425 |
        | 2025Q4 |  0.0077 |          0.0398 |  0.0425 |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        short_term_interest_rate = oecd_model.get_short_term_interest_rate(
            period=period, forecast=forecast
        )

        if growth:
            short_term_interest_rate = calculate_growth(
                short_term_interest_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        short_term_interest_rate = short_term_interest_rate.loc[
            self._start_date : None if forecast else self._end_date
        ]

        return short_term_interest_rate.round(rounding if rounding else self._rounding)

    def get_narrow_and_broad_money(
        self,
        period: str | None = None,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        M1 includes currency i.e. banknotes and coins, plus overnight deposits.
        M1 is expressed as a seasonally adjusted index based on 2015=100.

        Broad money (M3) includes currency, deposits with an agreed maturity of up
        to two years, deposits redeemable at notice of up to three months and
        repurchase agreements, money market fund shares/units and debt securities up
        to two years. M3 is measured as a seasonally adjusted index based on 2015=100.

        See definition: https://data.oecd.org/money/broad-money-m3.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Narrow and Broad Money.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2012-01-01', end_date='2020-01-01')

        narrow_and_broad_money = economics.get_narrow_and_broad_money(growth=True)

        narrow_and_broad_money.loc[:, 'Australia']
        ```

        Which returns:

        |      |   Narrow Money (M1) |   Broad Money (M3) |
        |:-----|--------------------:|-------------------:|
        | 2012 |              0.0462 |             0.0786 |
        | 2013 |              0.1402 |             0.0667 |
        | 2014 |              0.1279 |             0.0722 |
        | 2015 |              0.1466 |             0.068  |
        | 2016 |              0.0983 |             0.0594 |
        | 2017 |              0.0941 |             0.0684 |
        | 2018 |              0.0565 |             0.027  |
        | 2019 |              0.1066 |             0.0301 |
        | 2020 |              0.294  |             0.0835 |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        narrow_and_broad_money = {}

        narrow_and_broad_money["Narrow Money (M1)"] = oecd_model.get_narrow_money(
            period=period
        )
        narrow_and_broad_money["Broad Money (M3)"] = oecd_model.get_broad_money(
            period=period
        )

        narrow_and_broad_money_df = pd.concat(narrow_and_broad_money, axis=0).unstack(
            level=0
        )

        if growth:
            narrow_and_broad_money_df = calculate_growth(
                narrow_and_broad_money_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        narrow_and_broad_money_df = narrow_and_broad_money_df.loc[
            self._start_date : self._end_date
        ]

        return narrow_and_broad_money_df.round(rounding if rounding else self._rounding)

    def get_purchasing_power_parity(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Purchasing power parities (PPPs) are the rates of currency conversion that
        try to equalise the purchasing power of different currencies, by eliminating
        the differences in price levels between countries. The basket of goods and services
        priced is a sample of all those that are part of final expenditures: final consumption
        of households and government, fixed capital formation, and net exports.

        This indicator is measured in terms of national currency per US dollar.

        See definition: https://data.oecd.org/conversion/purchasing-power-parities-ppp.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Purchasing Power Parity.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01', end_date='2023-01-01')

        purchasing_power_parity = economics.get_purchasing_power_parity()

        purchasing_power_parity.loc[:, ['China', 'Chile', 'France']]
        ```

        Which returns:

        |      |   China |   Chile |   France |
        |:-----|--------:|--------:|---------:|
        | 2015 |  3.8707 | 391.179 |   0.8087 |
        | 2016 |  3.9888 | 397.251 |   0.78   |
        | 2017 |  4.1838 | 397.689 |   0.7701 |
        | 2018 |  4.2292 | 396.229 |   0.7562 |
        | 2019 |  4.2083 | 401.613 |   0.7163 |
        | 2020 |  4.1787 | 418.446 |   0.7104 |
        | 2021 |  4.1873 | 435.156 |   0.7187 |
        | 2022 |  4.0219 | 443.416 |   0.701  |
        """
        purchasing_power_parity = oecd_model.get_purchasing_power_parity()

        if growth:
            purchasing_power_parity = calculate_growth(
                purchasing_power_parity,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        purchasing_power_parity = purchasing_power_parity.loc[
            self._start_date : self._end_date
        ]

        return purchasing_power_parity.round(rounding if rounding else self._rounding)

    def get_exchange_rates(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Exchange rates are defined as the price of one country's' currency in relation
        to another country's currency. This indicator is measured in terms of
        national currency per US dollar.

        See definition: https://data.oecd.org/conversion/exchange-rates.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exchange Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics()

        exchange_rates = economics.get_exchange_rates()

        exchange_rates.loc[:, ['Netherlands', 'Japan', 'Indonesia']]
        ```

        Which returns:

        |      |   Netherlands |    Japan |   Indonesia |
        |:-----|--------------:|---------:|------------:|
        | 2013 |        0.7529 |  97.5957 |     10461.2 |
        | 2014 |        0.7527 | 105.945  |     11865.2 |
        | 2015 |        0.9013 | 121.044  |     13389.4 |
        | 2016 |        0.9034 | 108.793  |     13308.3 |
        | 2017 |        0.8852 | 112.166  |     13380.8 |
        | 2018 |        0.8468 | 110.423  |     14236.9 |
        | 2019 |        0.8933 | 109.01   |     14147.7 |
        | 2020 |        0.8755 | 106.775  |     14582.2 |
        | 2021 |        0.8455 | 109.754  |     14308.1 |
        | 2022 |        0.9496 | 131.498  |     14849.9 |
        """
        exchange_rates = oecd_model.get_exchange_rates()

        if growth:
            exchange_rates = calculate_growth(
                exchange_rates,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        exchange_rates = exchange_rates.loc[self._start_date : self._end_date]

        return exchange_rates.round(rounding if rounding else self._rounding)

    def get_renewable_energy(
        self,
        growth: bool = False,
        lag: int = 1,
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

        This indicator in percentage of total primary energy supply.

        See definition: https://data.oecd.org/energy/renewable-energy.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Renewable Energy Percentage.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date='2020-01-01')

        renewable_energy = economics.get_renewable_energy()

        renewable_energy.loc[:, ['Zambia', 'Albania', 'Austria']]
        ```

        Which returns:

        |      |   Zambia |   Albania |   Austria |
        |:-----|---------:|----------:|----------:|
        | 2010 |   0.9038 |    0.4049 |    0.2742 |
        | 2011 |   0.8882 |    0.2581 |    0.2696 |
        | 2012 |   0.8726 |    0.3121 |    0.307  |
        | 2013 |   0.874  |    0.3489 |    0.3011 |
        | 2014 |   0.8627 |    0.2722 |    0.3068 |
        | 2015 |   0.8486 |    0.3433 |    0.2985 |
        | 2016 |   0.8241 |    0.4209 |    0.3034 |
        | 2017 |   0.8097 |    0.273  |    0.2984 |
        | 2018 |   0.8081 |    0.4322 |    0.2943 |
        | 2019 |   0.8089 |    0.3172 |    0.3006 |
        | 2020 |   0.818  |    0.3388 |    0.3202 |
        """
        renewable_energy = oecd_model.get_renewable_energy()

        if growth:
            renewable_energy = calculate_growth(
                renewable_energy,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        renewable_energy = renewable_energy.loc[self._start_date : self._end_date]

        return renewable_energy.round(rounding if rounding else self._rounding)

    def get_environmental_tax(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Environmentally related taxes are an important instrument for governments to shape
        relative prices of goods and services.

        The characteristics of such taxes included in the database (e.g. revenue, tax base, tax
        rates, exemptions, etc.) are used to construct the environmentally related tax revenues with a breakdown
        by environmental domain:

        - Energy products (including vehicle fuels);
        - Motor vehicles and transport services;
        - Measured or estimated emissions to air and water, ozone depleting substances, certain non-point
        sources of water pollution, waste management and noise, as well as management of water, land, soil,
        forests, biodiversity, wildlife and fish stocks.

        The data have been cross-validated and complemented with Revenue statistics from the OECD Tax statistics
        database and official national sources.

        See definition: https://data.oecd.org/envpolicy/environmental-tax.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Environmental Tax.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2010-01-01", end_date="2020-01-01")

        environmental_tax = economics.get_environmental_tax()

        environmental_tax.loc[:, 'Netherlands']
        ```

        Which returns:

        |      |   Total |   Energy |   Transport |   Resource |   Pollution |
        |:-----|--------:|---------:|------------:|-----------:|------------:|
        | 2010 |    3.63 |     1.88 |        1.14 |       0.37 |        0.24 |
        | 2011 |    3.45 |     1.85 |        1.1  |       0.27 |        0.23 |
        | 2012 |    3.28 |     1.78 |        1.02 |       0.25 |        0.23 |
        | 2013 |    3.29 |     1.9  |        0.95 |       0.26 |        0.19 |
        | 2014 |    3.35 |     1.88 |        1    |       0.28 |        0.19 |
        | 2015 |    3.36 |     1.86 |        1.04 |       0.27 |        0.19 |
        | 2016 |    3.39 |     1.89 |        1.03 |       0.28 |        0.19 |
        | 2017 |    3.37 |     1.86 |        1.06 |       0.27 |        0.18 |
        | 2018 |    3.37 |     1.87 |        1.07 |       0.26 |        0.18 |
        | 2019 |    3.42 |     1.94 |        1.04 |       0.25 |        0.19 |
        | 2020 |    3.21 |     1.8  |        0.96 |       0.26 |        0.2  |
        """
        environmental_tax = {}

        environmental_tax["Total"] = oecd_model.get_environmental_tax(parameter="TOT")
        environmental_tax["Energy"] = oecd_model.get_environmental_tax(parameter="ENRG")
        environmental_tax["Transport"] = oecd_model.get_environmental_tax(
            parameter="TRANSPORT"
        )
        environmental_tax["Resource"] = oecd_model.get_environmental_tax(
            parameter="RES"
        )
        environmental_tax["Pollution"] = oecd_model.get_environmental_tax(
            parameter="POL"
        )

        environmental_tax_df = pd.concat(environmental_tax, axis=0).unstack(level=0)

        if growth:
            environmental_tax_df = calculate_growth(
                environmental_tax_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        environmental_tax_df = environmental_tax_df.loc[
            self._start_date : self._end_date
        ]

        return environmental_tax_df.round(rounding if rounding else self._rounding)

    def get_greenhouse_emissions(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Greenhouse gases refer to the sum of seven gases that have direct effects on climate change:

            - Carbon Dioxide (CO2)
            - Methane (CH4)
            - Nitrous Oxide (N2O)
            - Chlorofluorocarbons (CFCs)
            - Hydrofluorocarbons (HFCs)
            - Perfluorocarbons (PFCs)
            - Sulphur Hexafluoride (SF6)
            - Nitrogen Trifluoride (NF3).

        The data are expressed in CO2 equivalents and refer to gross direct emissions from human
        activities. CO2 refers to gross direct emissions from fuel combustion only and data are
        provided by the International Energy Agency. Other air emissions include emissions of
        sulphur oxides (SOx) and nitrogen oxides (NOx) given as quantities of SO2 and NO2, emissions of
        carbon monoxide (CO), and emissions of volatile organic compounds (VOC), excluding methane.

        Air and greenhouse gas emissions are measured in tonnes per capita and kilogram per capita
        in which all metrics are converted to tonnes (1000kg) per capita.

        See definition: https://data.oecd.org/air/air-and-ghg-emissions.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Environmental Tax.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date="2010-01-01", end_date="2020-01-01")

        greenhouse_emissions = economics.get_greenhouse_emissions()

        greenhouse_emissions.loc[:, 'United States']
        ```

        Which returns:

        |      |   Carbon Dioxide (CO2) |   Carbon Monoxide (CO) |   Greenhouse Gases (GHG) |   Nitrogen Oxides (NOX) |   Sulphur Oxides (SOX) |   Volatile Organic Compounds (VOC) |
        |:-----|-----------------------:|-----------------------:|-------------------------:|------------------------:|-----------------------:|-----------------------------------:|
        | 2010 |                  17.28 |                 0.1719 |                   22.818 |                  0.0449 |                 0.0203 |                             0.0388 |
        | 2011 |                  16.44 |                 0.1658 |                   22.168 |                  0.0423 |                 0.0186 |                             0.0389 |
        | 2012 |                  15.6  |                 0.1538 |                   21.252 |                  0.0395 |                 0.0147 |                             0.0382 |
        | 2013 |                  15.93 |                 0.1504 |                   21.647 |                  0.0368 |                 0.0139 |                             0.0364 |
        | 2014 |                  15.84 |                 0.1443 |                   21.667 |                  0.0345 |                 0.013  |                             0.036  |
        | 2015 |                  15.36 |                 0.1345 |                   21.006 |                  0.031  |                 0.0097 |                             0.0343 |
        | 2016 |                  14.97 |                 0.1259 |                   20.362 |                  0.0278 |                 0.0074 |                             0.0332 |
        | 2017 |                  14.64 |                 0.133  |                   20.183 |                  0.026  |                 0.0067 |                             0.0348 |
        | 2018 |                  15.02 |                 0.133  |                   20.667 |                  0.0248 |                 0.0064 |                             0.0351 |
        | 2019 |                  14.44 |                 0.1264 |                   20.156 |                  0.0236 |                 0.0054 |                             0.0332 |
        | 2020 |                  12.9  |                 0.1172 |                   18.178 |                  0.0207 |                 0.0047 |                             0.0329 |
        """
        greenhouse_gases = {}

        greenhouse_gases["Carbon Dioxide (CO2)"] = oecd_model.get_greenhouse_emissions(
            parameter="CO2"
        )
        greenhouse_gases["Carbon Monoxide (CO)"] = oecd_model.get_greenhouse_emissions(
            parameter="CO"
        )
        greenhouse_gases[
            "Greenhouse Gases (GHG)"
        ] = oecd_model.get_greenhouse_emissions(parameter="GHG")
        greenhouse_gases["Nitrogen Oxides (NOX)"] = oecd_model.get_greenhouse_emissions(
            parameter="NOX"
        )
        greenhouse_gases["Sulphur Oxides (SOX)"] = oecd_model.get_greenhouse_emissions(
            parameter="SOX"
        )
        greenhouse_gases[
            "Volatile Organic Compounds (VOC)"
        ] = oecd_model.get_greenhouse_emissions(parameter="VOC")

        greenhouse_gases_df = pd.concat(greenhouse_gases, axis=0).unstack(level=0)

        if growth:
            greenhouse_gases_df = calculate_growth(
                greenhouse_gases_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        greenhouse_gases_df = greenhouse_gases_df.loc[self._start_date : self._end_date]

        return greenhouse_gases_df.round(rounding if rounding else self._rounding)

    def get_crude_oil_production(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Crude oil production is defined as the quantities of oil extracted from the ground after
        the removal of inert matter or impurities. It includes crude oil, natural gas liquids (NGLs)
        and additives. This indicator is measured in thousand tonne of oil equivalent (toe).

        Crude oil is a mineral oil consisting of a mixture of hydrocarbons of natural origin, yellow
        to black in colour, and of variable density and viscosity. NGLs are the liquid or liquefied
        hydrocarbons produced in the manufacture, purification and stabilisation of natural gas.

        Additives are non-hydrocarbon substances added to or blended with a product to modify its
        properties, for example, to improve its combustion characteristics (e.g. MTBE and tetraethyl lead).
        Refinery production refers to the output of secondary oil products from an oil refinery.

        This indicator is measured in thousand tonne of oil equivalent (toe).

        See definition: https://data.oecd.org/energy/crude-oil-production.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Crude Oil Production.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2007-01-01')

        crude_oil_production = economics.get_crude_oil_production()

        crude_oil_production.loc[:, ['China', 'Saudi Arabia', 'Russia', 'Netherlands']]
        ```

        Which returns:

        |      |   China |   Saudi Arabia |   Russia |   Netherlands |
        |:-----|--------:|---------------:|---------:|--------------:|
        | 2007 |  186318 |         446438 |   470511 |      2109.1   |
        | 2008 |  190440 |         467050 |   471814 |      1765.4   |
        | 2009 |  189490 |         414458 |   479089 |      1338.07  |
        | 2010 |  203014 |         413505 |   487106 |      1040.27  |
        | 2011 |  202876 |         471515 |   494393 |      1102.48  |
        | 2012 |  207478 |         495778 |   499908 |      1133.08  |
        | 2013 |  209919 |         488039 |   499966 |      1144.3   |
        | 2014 |  211429 |         491857 |   505603 |      1555.3   |
        | 2015 |  214556 |         516157 |   512777 |      1423.99  |
        | 2016 |  199685 |         531161 |   524319 |       975.589 |
        | 2017 |  191506 |         504365 |   517105 |       970.892 |
        | 2018 |  189324 |         522375 |   525934 |       918.789 |
        | 2019 |  191014 |         496688 |   530219 |       761.583 |
        | 2020 |  194769 |         467840 |   484621 |       751.952 |
        | 2021 |  199264 |         463618 |   495677 |       763.855 |
        """
        crude_oil_production = oecd_model.get_crude_oil_production()

        if growth:
            crude_oil_production = calculate_growth(
                crude_oil_production,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        crude_oil_production = crude_oil_production.loc[
            self._start_date : self._end_date
        ]

        return crude_oil_production.round(rounding if rounding else self._rounding)

    def get_crude_oil_prices(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Crude oil import prices come from the IEA's Crude Oil Import Register and are influenced
        not only by traditional movements of supply and demand, but also by other factors such as
        geopolitics.

        Information is collected from national agencies according to the type of crude oil, by
        geographic origin and by quality of crude. Average prices are obtained by dividing value
        by volume as recorded by customs administrations for each tariff position.

        Values are recorded at the time of import and include cost, insurance and freight, but exclude
        import duties. The nominal crude oil spot price from 2003 to 2011 is for Dubai and from 1970 to
        2002 for Arabian Light. This indicator is measured in USD per barrel of oil.

        The real price was calculated using the deflator for GDP at market prices and rebased with
        reference year 1970 = 100.

        See definition: https://data.oecd.org/energy/crude-oil-import-prices.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Crude Oil Prices.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        crude_oil_prices = economics.get_crude_oil_prices()

        crude_oil_prices.loc[:, ['United Kingdom', 'Japan', 'Canada']]
        ```

        Which returns:

        |      |   United Kingdom |   Japan |   Canada |
        |:-----|-----------------:|--------:|---------:|
        | 2015 |            53.81 |   54.2  |    53.48 |
        | 2016 |            44.62 |   41.79 |    43.6  |
        | 2017 |            54.69 |   54.42 |    54.3  |
        | 2018 |            72.65 |   72.85 |    70.88 |
        | 2019 |            65.58 |   66.78 |    63.12 |
        | 2020 |            44.62 |   46.85 |    45.79 |
        | 2021 |            71.2  |   70.25 |    69.44 |
        | 2022 |           104.66 |  102.11 |   104.16 |
        """
        crude_oil_prices = oecd_model.get_crude_oil_prices()

        if growth:
            crude_oil_prices = calculate_growth(
                crude_oil_prices,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        crude_oil_prices = crude_oil_prices.loc[self._start_date : self._end_date]

        return crude_oil_prices.round(rounding if rounding else self._rounding)

    def get_government_statistics(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        This function contains a variety of general government statistics. The following statistics are provided:

            - General government deficit is defined as the balance of income and expenditure of government,
            including capital income and capital expenditures. "Net lending" means that government has a
            surplus, and is providing financial resources to other sectors, while "net borrowing" means
            that government has a deficit, and requires financial resources from other sectors.
            - General government revenue is defined as the revenue required to finance the goods
            and services they provide to citizens and businesses, and to fulfil their redistributive role.
            Comparing levels of government revenues across countries provides an indication of the
            importance of the government sector in the economy in terms of available financial resources.
            The total amount of revenues collected by governments is determined by past
            and current political decisions.
            - General government spending is defined as an indicator of the size of government
            across countries. The large variation in this indicator highlights the variety of
            countries' approaches to delivering public goods and services and providing
            social protection, not necessarily differences in resources spent
            - General government final consumption can be broken down into two distinct groups.
            The first reflects expenditures for collective consumption (defence, justice, etc.)
            that benefit the society as a whole, or large parts of society, and are often known
            as public goods and services. The second, referred to as "individual", relates to
            expenditures for individual consumption (health care, housing, education, etc.),
            incurred by government for the benefit of individual households.
            - General government debt-to-GDP ratio measures the gross debt of the general
            government as a percentage of GDP. It is a key indicator for the sustainability
            of government finance. Debt is calculated as the sum of the following liability
            categories (as applicable): currency and deposits; debt securities, loans;
            insurance, pensions and standardised guarantee schemes, and other accounts payable.
            Changes in government debt over time primarily reflect the impact of past government deficits.
            - The net financial worth of the general government sector is the total value of its
            financial assets minus the total value of its outstanding liabilities. The general
            government sector consists of central, state and local governments as well as social
            security funds.
            - General government production costs are decisions about the amount and type of goods
            and services governments produce, as well as on how best to produce them. They are
            often political in nature and based on a country's social and cultural context.
            Governments use a mix of their own employees, capital, and outside contractors
            (non-profit institutions or private sector entities) to produce goods and services.
            Government production costs include: compensation costs of general government employees;
            goods and services used and financed by general government (including intermediate consumption
            and social transfer in kind via market producers paid for by government); and other costs,
            including depreciation of capital and other taxes on production less other subsidies on
            production. The data include government employment and intermediate consumption for output
            produced by the government for its own use, such as roads and other capital investment
            projects built by government employees.

        All of these metrics are reported as a percentage of GDP which makes comparison between countries
        relatively straight forward.

        See definition: https://data.oecd.org/gga/general-government-deficit.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Statistics.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2012-01-01')

        goverment_statistics = economics.get_government_statistics()

        goverment_statistics.loc[:, 'Switzerland']
        ```

        Which returns:

        |      |   Deficit |   Revenue |   Spending |   Debt |   Financial Wealth |   Production Costs |
        |:-----|----------:|----------:|-----------:|-------:|-------------------:|-------------------:|
        | 2012 |    0.0024 |    0.3292 |     0.3269 | 0.4373 |            -0.0309 |             0.1559 |
        | 2013 |   -0.0043 |    0.332  |     0.3363 | 0.4299 |            -0.0393 |             0.1579 |
        | 2014 |   -0.0025 |    0.3302 |     0.3327 | 0.4293 |             0.0158 |             0.1589 |
        | 2015 |    0.0055 |    0.3416 |     0.3361 | 0.4287 |            -0.0218 |             0.1607 |
        | 2016 |    0.0024 |    0.3391 |     0.3367 | 0.4165 |             0.0056 |             0.1612 |
        | 2017 |    0.0113 |    0.3484 |     0.3371 | 0.4246 |             0.1066 |             0.1618 |
        | 2018 |    0.0129 |    0.3428 |     0.3299 | 0.4028 |             0.0927 |             0.1584 |
        | 2019 |    0.0134 |    0.346  |     0.3326 | 0.4007 |             0.1769 |             0.1596 |
        | 2020 |   -0.0306 |    0.3527 |     0.3834 | 0.4379 |             0.1763 |             0.1687 |
        | 2021 |   -0.0031 |    0.3545 |     0.3589 | 0.4137 |             0.2029 |             0.1683 |
        | 2022 |    0.012  |    0.3417 |   nan      | 0.3754 |             0.0436 |           nan      |
        """
        government_statistics = {}

        government_statistics["Deficit"] = oecd_model.get_general_government_deficit()
        government_statistics["Revenue"] = oecd_model.get_general_government_revenue()
        government_statistics["Spending"] = oecd_model.get_general_government_spending(
            parameter="TOT"
        )
        government_statistics[
            "Spending (Society)"
        ] = oecd_model.get_general_government_spending(parameter="SOCIETY")
        government_statistics[
            "Spending (Individual)"
        ] = oecd_model.get_general_government_spending(parameter="INDIVIDUAL")
        government_statistics["Debt"] = oecd_model.get_general_government_debt()
        government_statistics[
            "Financial Wealth"
        ] = oecd_model.get_general_government_financial_wealth()
        government_statistics[
            "Production Costs"
        ] = oecd_model.get_general_government_production_costs()

        government_statistics_df = pd.concat(government_statistics, axis=0).unstack(
            level=0
        )

        if growth:
            government_statistics_df = calculate_growth(
                government_statistics_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_statistics_df = government_statistics_df.loc[
            self._start_date : self._end_date
        ]

        return government_statistics_df.round(rounding if rounding else self._rounding)

    def get_central_government_spending(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Central government expenditure is defined as the central government budget expenditure
        as reported in the final central government accounts. Data are based on the System of
        National accounts (SNA), a set of internationally agreed concepts, definitions, classifications
        and rules for national accounting.

        Central government spending by function is the breakdown of expenditures on the basis of the
        activities governments support. The classification system used to provide this breakdown on an
        internationally comparable basis is known as Classification of Functions of Government (COFOG).

        COFOG expenditures are divided into in the following ten functions:

            - General Public Services;
            - Defence;
            - Public Order and Safety;
            - Economic Affairs;
            - Environmental Protection;
            - Housing and Community Amenities;
            - Health;
            - Recreation, Culture and Religion;
            - Education;
            - Social Protection.

        Data on central government expenditures by function include transfers between the different levels
        of government. The general government sector consists of central, state and local governments,
        and the social security funds controlled by these units. The political authority of central
        government extends over the entire territory of the country; central government has the authority
        to impose taxes on all resident and non-resident units engaged in economic activities within the
        country.

        The responsibility for the provision of public goods and services and redistribution of income is
        divided between different levels of government. Data on the distribution of government spending
        by both level and function can provide an indication of the extent to which key government activities
        are decentralised to sub-national governments. This indicator of central government spending by
        function is measured as a percentage of total expenditures.

        See definition: https://data.oecd.org/gga/central-government-spending.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Central Government Spending.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2012-01-01')

        goverment_spending = economics.get_central_government_spending()

        goverment_spending.loc[:, 'OECD - Total']
        ```

        Which returns:

        |      |   General Public Services |   Defence |   Public Order and Safety |   Economic Affairs |   Environmental Protection |   Housing and Community Amenities |   Health |   Recreation, Culture and Religion |   Education |   Social Protection |
        |:-----|--------------------------:|----------:|--------------------------:|-------------------:|---------------------------:|----------------------------------:|---------:|-----------------------------------:|------------:|--------------------:|
        | 2012 |                    0.2408 |    0.1005 |                    0.0298 |             0.0932 |                     0.0054 |                            0.0182 |   0.1478 |                             0.0085 |      0.0731 |              0.2827 |
        | 2013 |                    0.2415 |    0.0957 |                    0.0302 |             0.0912 |                     0.0059 |                            0.0159 |   0.1517 |                             0.0083 |      0.074  |              0.2856 |
        | 2014 |                    0.2371 |    0.0924 |                    0.0302 |             0.087  |                     0.0055 |                            0.016  |   0.1615 |                             0.0084 |      0.0748 |              0.287  |
        | 2015 |                    0.2297 |    0.0911 |                    0.0306 |             0.0877 |                     0.0055 |                            0.0152 |   0.1677 |                             0.0084 |      0.0748 |              0.2893 |
        | 2016 |                    0.2263 |    0.0901 |                    0.031  |             0.0862 |                     0.0056 |                            0.014  |   0.1719 |                             0.0092 |      0.0735 |              0.2923 |
        | 2017 |                    0.2236 |    0.0903 |                    0.0308 |             0.0892 |                     0.0053 |                            0.0147 |   0.1726 |                             0.0094 |      0.0736 |              0.2906 |
        | 2018 |                    0.2247 |    0.0922 |                    0.0311 |             0.0883 |                     0.0051 |                            0.0142 |   0.1753 |                             0.0093 |      0.074  |              0.286  |
        | 2019 |                    0.2205 |    0.0945 |                    0.0309 |             0.0888 |                     0.005  |                            0.014  |   0.1774 |                             0.0092 |      0.0729 |              0.2866 |
        """
        government_spending = {}

        government_spending[
            "General Public Services"
        ] = oecd_model.get_central_government_spending(parameter="GRALPUBSER")
        government_spending["Defence"] = oecd_model.get_central_government_spending(
            parameter="DEF"
        )
        government_spending[
            "Public Order and Safety"
        ] = oecd_model.get_central_government_spending(parameter="PUBORD")
        government_spending[
            "Economic Affairs"
        ] = oecd_model.get_central_government_spending(parameter="ECOAFF")
        government_spending[
            "Environmental Protection"
        ] = oecd_model.get_central_government_spending(parameter="ENVPROT")
        government_spending[
            "Housing and Community Amenities"
        ] = oecd_model.get_central_government_spending(parameter="HOUCOMM")
        government_spending["Health"] = oecd_model.get_central_government_spending(
            parameter="HEALTH"
        )
        government_spending[
            "Recreation, Culture and Religion"
        ] = oecd_model.get_central_government_spending(parameter="RECULTREL")
        government_spending["Education"] = oecd_model.get_central_government_spending(
            parameter="EDU"
        )
        government_spending[
            "Social Protection"
        ] = oecd_model.get_central_government_spending(parameter="SOCPROT")

        government_spending_df = pd.concat(government_spending, axis=0).unstack(level=0)

        if growth:
            government_spending_df = calculate_growth(
                government_spending_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_spending_df = government_spending_df.loc[
            self._start_date : self._end_date
        ]

        return government_spending_df.round(rounding if rounding else self._rounding)

    def get_trust_in_government(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Trust in government refers to the share of people who report having confidence
        in the national government. The data shown reflect the share of respondents
        answering “yes” (the other response categories being “no”, and “dont know”)
        to the survey question: “In this country, do you have confidence in… national government?

        Due to small sample sizes, country averages for horizontal inequalities (by age,
        gender and education) are pooled between 2010-18 to improve the accuracy of the
        estimates.

        The sample is ex ante designed to be nationally representative of the population
        aged 15 and over. This indicator is measured as a percentage of all survey respondents.

        See definition: https://data.oecd.org/gga/trust-in-government.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
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
        | 2006 |          0.558  |   0.4875 |  0.3503 |
        | 2007 |          0.3932 |   0.3814 |  0.24   |
        | 2008 |          0.3792 | nan      |  0.2212 |
        | 2009 |          0.503  |   0.3162 |  0.2518 |
        | 2010 |          0.4183 |   0.2365 |  0.2703 |
        | 2011 |          0.3825 |   0.1752 |  0.2311 |
        | 2012 |          0.3489 |   0.1262 |  0.1692 |
        | 2013 |          0.2886 |   0.1436 |  0.3581 |
        | 2014 |          0.3487 |   0.1883 |  0.3795 |
        | 2015 |          0.3469 |   0.4373 |  0.3529 |
        | 2016 |          0.2972 |   0.1325 |  0.3622 |
        | 2017 |          0.3865 |   0.1399 |  0.4125 |
        | 2018 |          0.3138 |   0.157  |  0.3849 |
        | 2019 |          0.3628 |   0.3964 |  0.4112 |
        | 2020 |          0.4649 |   0.3975 |  0.4234 |
        | 2021 |          0.4046 |   0.4017 |  0.2908 |
        | 2022 |          0.3102 |   0.2563 |  0.4315 |
        """
        trust_in_government = oecd_model.get_trust_in_goverment()

        if growth:
            trust_in_government = calculate_growth(
                trust_in_government,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        trust_in_government = trust_in_government.loc[self._start_date : self._end_date]

        return trust_in_government.round(rounding if rounding else self._rounding)

    def get_unemployment_rate(
        self,
        period: str | None = None,
        growth: bool = False,
        lag: int = 1,
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

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Unemployment Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-03-01', end_date='2023-01-01')

        unemployment_rate = economics.get_unemployment_rate(period='quarterly')

        unemployment_rate.loc[:, ['Germany', 'United States', 'Japan']]
        ```

        Which returns:

        |        |   Germany |   United States |   Japan |
        |:-------|----------:|----------------:|--------:|
        | 2021Q1 |    0.039  |          0.062  |  0.0283 |
        | 2021Q2 |    0.037  |          0.0593 |  0.029  |
        | 2021Q3 |    0.0343 |          0.0513 |  0.0277 |
        | 2021Q4 |    0.0323 |          0.042  |  0.0273 |
        | 2022Q1 |    0.031  |          0.038  |  0.0267 |
        | 2022Q2 |    0.03   |          0.036  |  0.026  |
        | 2022Q3 |    0.0307 |          0.0357 |  0.0257 |
        | 2022Q4 |    0.0303 |          0.036  |  0.0253 |
        | 2023Q1 |    0.0293 |          0.035  |  0.026  |
        """
        period = (
            period
            if period is not None
            else "quarterly"
            if self._quarterly
            else "yearly"
        )

        unemployment_rate = oecd_model.get_unemployment_rate(period=period)

        if growth:
            unemployment_rate = calculate_growth(
                unemployment_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        unemployment_rate = unemployment_rate.loc[self._start_date : self._end_date]

        return unemployment_rate.round(rounding if rounding else self._rounding)

    def get_labour_productivity(
        self,
        growth: bool = False,
        lag: int = 1,
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

        This uses 2015 as the base year (= 100)

        See definition: https://data.oecd.org/lprdty/gdp-per-hour-worked.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exchange Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics()

        labour_productivity = economics.get_exchange_rates()

        labour_productivity.loc[:, ['Bulgaria', 'Croatia', 'Spain']]
        ```

        Which returns:

        |      |   Bulgaria |   Croatia |   Spain |
        |:-----|-----------:|----------:|--------:|
        | 2013 |     1.4736 |    0.7572 |  0.7529 |
        | 2014 |     1.4742 |    0.7629 |  0.7527 |
        | 2015 |     1.7644 |    0.9103 |  0.9013 |
        | 2016 |     1.768  |    0.9033 |  0.9034 |
        | 2017 |     1.7355 |    0.8791 |  0.8852 |
        | 2018 |     1.657  |    0.8334 |  0.8468 |
        | 2019 |     1.747  |    0.879  |  0.8933 |
        | 2020 |     1.7163 |    0.8778 |  0.8755 |
        | 2021 |     1.6538 |    0.8441 |  0.8455 |
        | 2022 |     1.8601 |    0.9503 |  0.9496 |
        """
        labour_productivity = oecd_model.get_labour_productivity()

        if growth:
            labour_productivity = calculate_growth(
                labour_productivity,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        labour_productivity = labour_productivity.loc[self._start_date : self._end_date]

        return labour_productivity.round(rounding if rounding else self._rounding)

    def get_income_inequality(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Income is defined as household disposable income in a particular year. It consists of earnings,
        self-employment and capital income and public cash transfers; income taxes and social
        security contributions paid by households are deducted. The income of the household is
        attributed to each of its members, with an adjustment to reflect differences in needs for
        households of different sizes. Income inequality among individuals is measured here by five indicators.

            - The Gini coefficient is based on the comparison of cumulative proportions of the population against
            cumulative proportions of income they receive, and it ranges between 0 in the case of perfect equality
            and 1 in the case of perfect inequality.
            - P90/P10 is the ratio of the upper bound value of the ninth decile (i.e. the 10% of people
            with highest income) to that of the first decile;
            - P90/P50 of the upper bound value of the ninth decile to the median income;
            - and P50/P10 of median income to the upper bound value of the first decile.
            - The Palma ratio is the share of all income received by the 10% people with highest
            disposable income divided by the share of all income received by the 40%
            people with the lowest disposable income.
            - S80/S20 is the ratio of the average income of the 20% richest to the 20% poorest;

        See definition: https://data.oecd.org/inequality/income-inequality.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Population Statistics.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2013-01-01')

        income_inequality = economics.get_income_inequality()

        income_inequality.loc[:, 'United States']
        ```

        Which returns:

        |      |   Gini Coefficient |   P90/P10 |   P90/P50 |   P50/P10 |   Palma Ratio |   S80/S20 |
        |:-----|-------------------:|----------:|----------:|----------:|--------------:|----------:|
        | 2013 |              0.396 |       6.4 |       2.3 |       2.7 |          1.82 |       8.6 |
        | 2014 |              0.394 |       6.4 |       2.3 |       2.7 |          1.79 |       8.7 |
        | 2015 |              0.39  |       6.1 |       2.3 |       2.7 |          1.75 |       8.3 |
        | 2016 |              0.391 |       6.3 |       2.3 |       2.7 |          1.77 |       8.5 |
        | 2017 |              0.39  |       6.2 |       2.3 |       2.7 |          1.76 |       8.4 |
        | 2018 |              0.393 |       6.3 |       2.3 |       2.8 |          1.79 |       8.4 |
        | 2019 |              0.395 |       6.3 |       2.3 |       2.7 |          1.81 |       8.4 |
        | 2020 |              0.377 |       5.8 |       2.2 |       2.6 |          1.64 |       7.5 |
        | 2021 |              0.375 |       5.4 |       2.2 |       2.4 |          1.63 |       7.1 |
        """
        income_inequality = {}

        income_inequality["Gini Coefficient"] = oecd_model.get_income_inequality(
            parameter="GINI"
        )
        income_inequality["P90/P10"] = oecd_model.get_income_inequality(
            parameter="P90P10"
        )
        income_inequality["P90/P50"] = oecd_model.get_income_inequality(
            parameter="P90P50"
        )
        income_inequality["P50/P10"] = oecd_model.get_income_inequality(
            parameter="P50P10"
        )
        income_inequality["Palma Ratio"] = oecd_model.get_income_inequality(
            parameter="PALMA"
        )
        income_inequality["S80/S20"] = oecd_model.get_income_inequality(
            parameter="S80S20"
        )

        income_inequality_df = pd.concat(income_inequality, axis=0).unstack(level=0)

        if growth:
            income_inequality_df = calculate_growth(
                income_inequality_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        income_inequality_df = income_inequality_df.loc[
            self._start_date : self._end_date
        ]

        return income_inequality_df.round(rounding if rounding else self._rounding)

    def get_population_statistics(
        self,
        growth: bool = False,
        lag: int = 1,
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
        statistical projections, helping governments in their decision making. This indicator is
        measured in terms of thousands of people.

        Furthermore the following statistics are provided:

            - The youth population is defined as those people aged less than 15 as a percentage
            of the total population.
            - The working age population is defined as those aged 15 to 64 as a percentage of
            the total population.
            - The elderly population is defined as those aged 65 and over as a percentage of
            the total population.
            - The total fertility rate in a specific year is defined as the total number
            of children that would be born to each woman if she were to live to the end of
            her child-bearing years and give birth to children in alignment with the
            prevailing age-specific fertility rates.
            - The old-age to working-age demographic ratio is defined as the number of
            individuals aged 65 and over per 100 people of working age defined as
            those at ages 20 to 64.

        See definition: https://data.oecd.org/pop/population.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Population Statistics.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date='2019-01-01')

        population_statistics = economics.get_population_statistics()

        population_statistics.loc[:, 'Japan']
        ```

        Which returns:

        |      |   Population |   Young Population |   Working Age Population |   Elderly Population |
        |:-----|-------------:|-------------------:|-------------------------:|---------------------:|
        | 2010 |      128.057 |             0.1315 |                   0.6383 |               0.2302 |
        | 2011 |      127.834 |             0.1307 |                   0.6365 |               0.2328 |
        | 2012 |      127.593 |             0.1298 |                   0.6288 |               0.2415 |
        | 2013 |      127.414 |             0.1288 |                   0.6207 |               0.2506 |
        | 2014 |      127.237 |             0.1277 |                   0.6126 |               0.2597 |
        | 2015 |      127.095 |             0.1255 |                   0.6081 |               0.2665 |
        | 2016 |      127.042 |             0.1244 |                   0.6035 |               0.272  |
        | 2017 |      126.918 |             0.1232 |                   0.6003 |               0.2765 |
        | 2018 |      126.749 |             0.1221 |                   0.598  |               0.2799 |
        | 2019 |      126.555 |             0.1206 |                   0.5969 |               0.2825 |
        """

        population_statistics = {}

        population_statistics["Population"] = oecd_model.get_population()
        population_statistics["Men Ratio"] = (
            oecd_model.get_population(gender="men")
            / population_statistics["Population"]
        )
        population_statistics["Women Ratio"] = (
            oecd_model.get_population(gender="women")
            / population_statistics["Population"]
        )

        population_statistics[
            "Young Population"
        ] = oecd_model.get_young_population_percentage()
        population_statistics[
            "Working Age Population"
        ] = oecd_model.get_working_age_population()
        population_statistics[
            "Elderly Population"
        ] = oecd_model.get_elderly_population_percentage()

        population_statistics["Fertility Rate"] = oecd_model.get_fertility_rate()

        # This is done to align the Old Age Dependency Ratio with the other statistics
        latest_year = population_statistics["Population"].index[-1]
        population_statistics[
            "Old Age Dependency Ratio"
        ] = oecd_model.get_old_age_dependency_ratio()
        population_statistics["Old Age Dependency Ratio"] = population_statistics[
            "Old Age Dependency Ratio"
        ].loc[:latest_year]

        population_statistics_df = pd.concat(population_statistics, axis=0).unstack(
            level=0
        )

        if growth:
            population_statistics_df = calculate_growth(
                population_statistics_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        population_statistics_df = population_statistics_df.loc[
            self._start_date : self._end_date
        ]

        return population_statistics_df.round(rounding if rounding else self._rounding)

    def get_poverty_rate(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        The poverty rate is the ratio of the number of people (in a given age group) whose income
        falls below the poverty line; taken as half the median household income of the total population.

        It is also available by broad age group:

            - child poverty (0 to 17 year-olds);
            - working-age poverty (18 to 65 year-olds);
            - and elderly poverty (66 year-olds or more).

        However, two countries with the same poverty rates may differ in terms of the relative income-level of the poor.

        See definition: https://data.oecd.org/inequality/poverty-rate.htm

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Poverty Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2012-01-01', end_date='2020-01-01')

        poverty_rate = economics.get_poverty_rate()

        poverty_rate.loc[:, 'Portugal']
        ```

        Which returns:

        |      |   Total |   0-17 Year |   18-65 Year |   66 or More |
        |:-----|--------:|------------:|-------------:|-------------:|
        | 2012 |   0.13  |       0.178 |        0.129 |        0.082 |
        | 2013 |   0.135 |       0.183 |        0.133 |        0.097 |
        | 2014 |   0.135 |       0.182 |        0.133 |        0.097 |
        | 2015 |   0.125 |       0.155 |        0.123 |        0.108 |
        | 2016 |   0.125 |       0.155 |        0.126 |        0.095 |
        | 2017 |   0.107 |       0.122 |        0.105 |        0.101 |
        | 2018 |   0.104 |       0.122 |        0.103 |        0.09  |
        | 2019 |   0.106 |       0.131 |        0.098 |        0.107 |
        | 2020 |   0.128 |       0.152 |        0.118 |        0.138 |
        """
        poverty_rate = {}

        poverty_rate["Total"] = oecd_model.get_poverty_rate(parameter="TOT")
        poverty_rate["0-17 Year"] = oecd_model.get_poverty_rate(parameter="0_17")
        poverty_rate["18-65 Year"] = oecd_model.get_poverty_rate(parameter="18_65")
        poverty_rate["66 or More"] = oecd_model.get_poverty_rate(parameter="66MORE")

        poverty_rate_df = pd.concat(poverty_rate, axis=0).unstack(level=0)

        if growth:
            poverty_rate_df = calculate_growth(
                poverty_rate_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        poverty_rate_df = poverty_rate_df.loc[self._start_date : self._end_date]

        return poverty_rate_df.round(rounding if rounding else self._rounding)
