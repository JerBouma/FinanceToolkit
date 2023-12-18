"""Economics Module"""
__docformat__ = "google"


import re
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit.economics import ecb_model, fed_model, oecd_model
from financetoolkit.helpers import calculate_growth, handle_errors

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals


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
            else (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
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

    def get_european_central_bank_rates(self, rate: str | None = None):
        """
        The Governing Council of the ECB sets the key interest rates for the
        euro area. The available rates are:

        - Main refinancing operations (refinancing)
        - Marginal lending facility (lending)
        - Deposit facility (deposit)

        The main refinancing operations (MRO) rate is the interest rate banks
        pay when they borrow money from the ECB for one week. When they do this,
        they have to provide collateral to guarantee that the money will be paid back.

        The marginal lending facility rate is the interest rate banks pay when they
        borrow from the ECB overnight. When they do this, they have to provide collateral,
        for example securities, to guarantee that the money will be paid back.

        The deposit facility rate is one of the three interest rates the ECB sets every
        six weeks as part of its monetary policy. The rate defines the interest banks
        receive for depositing money with the central bank overnight.

        See source: https://data.ecb.europa.eu/main-figures/

        Args:
            rate (str, optional): The rate to return. Defaults to None, which returns all rates.
                Choose between 'refinancing', 'lending' or 'deposit'.

        Returns:
            pd.DataFrame: A DataFrame containing the ECB rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-12-01')

        economics.get_european_central_bank_rates()
        ```

        Which returns:

        |            |   Refinancing |   Lending |   Deposit |
        |:-----------|--------------:|----------:|----------:|
        | 2023-12-01 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-02 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-03 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-04 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-05 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-06 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-07 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-08 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-09 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-10 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-11 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-12 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-13 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-14 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-15 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-16 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-17 |         0.045 |    0.0475 |      0.04 |
        | 2023-12-18 |         0.045 |    0.0475 |      0.04 |
        """
        ecb_rates = pd.DataFrame()

        if rate and rate not in ["refinancing", "lending", "deposit"]:
            raise ValueError(
                "Rate must be one of 'refinancing', 'lending' or 'deposit' or left empty for all."
            )

        if not rate or rate == "refinancing":
            ecb_rates["Refinancing"] = ecb_model.get_main_refinancing_operations()
        if not rate or rate == "lending":
            ecb_rates["Lending"] = ecb_model.get_marginal_lending_facility()
        if not rate or rate == "deposit":
            ecb_rates["Deposit"] = ecb_model.get_deposit_facility()

        ecb_rates = ecb_rates.loc[self._start_date : self._end_date]

        return ecb_rates

    def get_federal_reserve_rates(self, rate: str = "EFFR"):
        """
        Get the Federal Reserve rates as published by the Federal Reserve Bank of New York.
        The federal funds market consists of domestic unsecured borrowings in U.S. dollars
        by depository institutions from other depository institutions and certain other
        entities, primarily government-sponsored enterprises.

        The following rates are available:

        - Effective Federal Funds Rate (EFFR)
        - Overnight Bank Funding Rate (OBFR)
        - Tri-Party General Collateral Rate (TGCR)
        - Broad General Collateral Rate (BGCR)
        - Secured Overnight Financing Rate (SOFR)

        The effective federal funds rate (EFFR) is calculated as a volume-weighted median
        of overnight federal funds transactions reported in the FR 2420 Report of Selected
        Money Market Rates.

        The overnight bank funding rate (OBFR) is calculated as a volume-weighted median
        of overnight federal funds transactions, Eurodollar transactions, and the
        domestic deposits reported as “Selected Deposits” in the FR 2420 Report.

        The TGCR is calculated as a volume-weighted median of transaction-level
        tri-party repo data collected from the Bank of New York Mellon.

        The BGCR is calculated as a volume-weighted median of transaction-level
        tri-party repo data collected from the Bank of New York Mellon as well
        as GCF Repo transaction data obtained from the U.S. Department of the
        Treasury’s Office of Financial Research (OFR).

        The SOFR is calculated as a volume-weighted median of transaction-level
        tri-party repo data collected from the Bank of New York Mellon as well as
        GCF Repo transaction data and data on bilateral Treasury repo transactions
        cleared through FICC's DVP service, which are obtained from the U.S.
        Department of the Treasury’s Office of Financial Research (OFR).

        The New York Fed publishes the rates for the prior business day on the New
        York Fed’s website between 8:00 and 9:00 a.m.

        See source: https://www.newyorkfed.org/markets/reference-rates/

        Args:
            rate (str): The rate to return. Defaults to 'EFFR' (Effective Federal Funds Rate).

        Returns:
            pd.DataFrame: A DataFrame containing the Federal Reserve rates including the rate,
                percentiles, volume and upper and lower bounds.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-12-01')

        effr = economics.get_federal_reserve_rates()

        effr.loc[:, ['Rate', '1st Percentile', '25th Percentile', '75th Percentile', '99th Percentile']]
        ```

        Which returns:

        | Effective Date   |   Rate |   1st Percentile |   25th Percentile |   75th Percentile |   99th Percentile |
        |:-----------------|-------:|-----------------:|------------------:|------------------:|------------------:|
        | 2023-12-01       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0544 |
        | 2023-12-04       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-05       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-06       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-07       | 0.0533 |            0.053 |            0.0531 |            0.0534 |            0.0545 |
        | 2023-12-08       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-11       | 0.0533 |            0.053 |            0.0532 |            0.0533 |            0.0545 |
        | 2023-12-12       | 0.0533 |            0.053 |            0.0531 |            0.0533 |            0.0544 |
        | 2023-12-13       | 0.0533 |            0.053 |            0.0531 |            0.0533 |            0.0545 |
        | 2023-12-14       | 0.0533 |            0.053 |            0.0531 |            0.0533 |            0.0535 |
        """
        rate = rate.upper()

        if rate == "EFFR":
            fed_data = fed_model.get_effective_federal_funds_rate()
        elif rate == "OBFR":
            fed_data = fed_model.get_overnight_banking_funding_rate()
        elif rate == "TGCR":
            fed_data = fed_model.get_tri_party_general_collateral_rate()
        elif rate == "BGCR":
            fed_data = fed_model.get_broad_general_collateral_rate()
        elif rate == "SOFR":
            fed_data = fed_model.get_secured_overnight_financing_rate()
        else:
            raise ValueError(
                "Rate must be one of 'EFFR', 'OBFR', 'TGCR', 'BGCR' or 'SOFR'."
            )

        fed_data = fed_data.loc[self._start_date : self._end_date]

        return fed_data
