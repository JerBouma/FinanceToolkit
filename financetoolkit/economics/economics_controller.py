"""Economics Module"""

__docformat__ = "google"


import re
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit.economics import gmdb_model, oecd_model
from financetoolkit.helpers import calculate_growth
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

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
        gmdb_source: bool = False,
        quarterly: bool | None = None,
        rounding: int | None = 4,
    ):
        """
        Initializes the Economics Controller Class.

        Args:
            start_date (str | None, optional): The start date to retrieve data from. Defaults to None.
            end_date (str | None, optional): The end date to retrieve data from. Defaults to None.
            gmdb_source (bool, optional): If True, retrieves data from the GMDB source. Defaults to False.
            quarterly (bool | None, optional): If True, returns quarterly data; otherwise, returns yearly data.
                Defaults to None. This only works for data retrieved from the OECD source.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

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

        self._gmdb_source: bool = gmdb_source
        self._gmbd_dataset: pd.DataFrame = (
            gmdb_model.collect_global_macro_database_dataset()
            if self._gmdb_source
            else pd.DataFrame()
        )
        self._quarterly: bool | None = quarterly
        self._rounding: int | None = rounding

    @handle_errors
    def get_gross_domestic_product(
        self,
        inflation_adjusted: bool = False,
        gmdb_source: bool | None = None,
        growth: bool = False,
        lag: int = 1,
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

        It is also possible to acquire the data from the Global Macro Database (GMDB) source which
        also provides inflation adjusted data. For more information see:
        https://www.globalmacrodata.com/files/documentations/Variables/nGDP.pdf

        Args:
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data. Defaults to False.
            gmdb_source (bool | None, optional): If True, retrieves data from the GMDB source. Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        real_gdp = economics.get_gross_domestic_product(inflation_adjusted=True)

        real_gdp.loc[:, ['Netherlands', 'Germany', 'China']]
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
            gross_domestic_product = oecd_model.get_annual_gross_domestic_product()

        if growth:
            gross_domestic_product = calculate_growth(
                gross_domestic_product,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        gross_domestic_product = gross_domestic_product.loc[
            self._start_date : self._end_date
        ]

        return gross_domestic_product.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_gross_domestic_product_deflator(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Gross Domestic Product Deflator for a variety of countries over
        time from the Global Macro Database (GMDB). The GDP deflator is a measure of
        the price of all domestically produced final goods and services in an economy
        relative to the price level in a base year which can vary per country.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Gross Domestic Product Deflator

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        gdp_deflator = economics.get_gross_domestic_product_deflator()

        gdp_deflator.loc[:, ['United States', 'Canada', 'Russian Federation']]
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

        if growth:
            gross_domestic_product_deflator = calculate_growth(
                gross_domestic_product_deflator,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        gross_domestic_product_deflator = gross_domestic_product_deflator.loc[
            self._start_date : self._end_date
        ]

        return gross_domestic_product_deflator.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_total_consumption(
        self,
        inflation_adjusted: bool = False,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Total Consumption for a variety of countries over time from the
        Global Macro Database (GMDB). Total Consumption is the total amount of money
        spent by households on consumer goods and services.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data. Defaults to False.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
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

        if growth:
            total_consumption = calculate_growth(
                total_consumption,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        total_consumption = total_consumption.loc[self._start_date : self._end_date]

        return total_consumption.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_total_consumption_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Total Consumption to GDP Ratio for a variety of countries over time from the
        Global Macro Database (GMDB). The Total Consumption to GDP Ratio is the ratio of the
        total amount of money spent by households on consumer goods and services to the Gross
        Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
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
        | 2018 |       68.9658 |  78.3547 |  76.5334 |
        | 2019 |       68.2768 |  77.6248 |  75.6476 |
        | 2020 |       68.3982 |  78.92   |  75.5519 |
        | 2021 |       68.061  |  77.7649 |  74.9877 |
        | 2022 |       68.357  |  78.5937 |  76.2568 |
        | 2023 |       68.8716 |  78.8461 |  76.7188 |
        | 2024 |       69.8097 |  78.906  |  76.7534 |
        | 2025 |       70.0162 |  78.995  |  77.1961 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        total_consumption_to_gdp_ratio = gmdb_model.get_total_consumption_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
        )

        if growth:
            total_consumption_to_gdp_ratio = calculate_growth(
                total_consumption_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        total_consumption_to_gdp_ratio = total_consumption_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return total_consumption_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_investment(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Investment for a variety of countries over time from the Global Macro Database (GMDB).
        Investment is the total amount of money spent by businesses on capital goods, such as machinery,
        equipment, and buildings.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
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

        if growth:
            investment = calculate_growth(
                investment,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        investment = investment.loc[self._start_date : self._end_date]

        return investment.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_investment_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Investment to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Investment to GDP Ratio is the ratio of the total amount of money spent by businesses on capital goods,
        such as machinery, equipment, and buildings to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
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
        | 2019 |      22.55  |  25.79  |   24.878 |
        | 2020 |      22.3   |  25.224 |   31.341 |
        | 2021 |      23.313 |  25.797 |   31.404 |
        | 2022 |      23.722 |  26.811 |   35.04  |
        | 2023 |      23.981 |  26.397 |   29.964 |
        | 2024 |      24.149 |  26.572 |   25.574 |
        | 2025 |      23.928 |  26.639 |   24.649 |

        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        investment_to_gdp_ratio = gmdb_model.get_investment_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
        )

        if growth:
            investment_to_gdp_ratio = calculate_growth(
                investment_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        investment_to_gdp_ratio = investment_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return investment_to_gdp_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_fixed_investment(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Fixed Investment for a variety of countries over time from the Global Macro Database (GMDB).
        Fixed Investment is the total amount of money spent by businesses on capital goods, such as machinery,
        equipment, and buildings that are expected to last for more than one year.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
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

        if growth:
            fixed_investment = calculate_growth(
                fixed_investment,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        fixed_investment = fixed_investment.loc[self._start_date : self._end_date]

        return fixed_investment.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_fixed_investment_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Fixed Investment to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Fixed Investment to GDP Ratio is the ratio of the total amount of money spent by businesses on capital goods,
        such as machinery, equipment, and buildings that are expected to last for more than one year to the Gross Domestic
        Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
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
        | 2000 |   25.7061 |   22.8829 |       27.512  |
        | 2001 |   24.9279 |   21.5434 |       26.8656 |
        | 2002 |   23.6231 |   19.8867 |       27.0789 |
        | 2003 |   24.1941 |   19.2639 |       26.5301 |
        | 2004 |   23.7683 |   18.8315 |       27.3119 |
        | 2005 |   23.2346 |   18.7644 |       27.263  |
        | 2006 |   22.8307 |   19.4624 |       27.0603 |
        | 2007 |   23.1394 |   19.7152 |       27.2073 |
        | 2008 |   23.5338 |   19.96   |       26.7343 |
        | 2009 |   22.6637 |   18.8314 |       25.2148 |
        | 2010 |   21.8612 |   19.1803 |       25.1886 |
        | 2011 |   22.7074 |   20.0016 |       25.5188 |
        | 2012 |   22.9528 |   19.9585 |       26.2689 |
        | 2013 |   23.3121 |   19.5717 |       26.2276 |
        | 2014 |   22.9672 |   19.7779 |       26.4182 |
        | 2015 |   22.896  |   19.7566 |       26.4112 |
        | 2016 |   23.3219 |   20.0382 |       26.4998 |
        | 2017 |   23.8459 |   20.146  |       27.1489 |
        | 2018 |   24.3136 |   20.8351 |       26.6027 |
        | 2019 |   25.0801 |   21.1667 |       26.6222 |
        | 2020 |   25.1343 |   21.3495 |       26.9939 |
        | 2021 |   25.8758 |   21.1944 |       26.3366 |
        | 2022 |   25.4742 |   21.7068 |       26.2505 |
        | 2023 |   24.9012 |   21.4997 |       25.9307 |
        | 2024 |   25.1474 |   20.6719 |       24.9148 |
        | 2025 |   25.2518 |   20.7174 |       24.8035 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        fixed_investment_to_gdp_ratio = gmdb_model.get_fixed_investment_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
        )

        if growth:
            fixed_investment_to_gdp_ratio = calculate_growth(
                fixed_investment_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        fixed_investment_to_gdp_ratio = fixed_investment_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return fixed_investment_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_exports(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Exports for a variety of countries over time from the Global Macro Database (GMDB).
        Exports are the total amount of goods and services produced in a country that are sold to
        other countries.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exports

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='1980-01-01', end_date='1990-01-01')

        exports = economics.get_exports()

        exports.loc[:, ['Netherlands', 'Germany', 'China']]
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

        if growth:
            exports = calculate_growth(
                exports,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        exports = exports.loc[self._start_date : self._end_date]

        return exports.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_exports_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Exports to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Exports to GDP Ratio is the ratio of the total amount of goods and services produced in a country
        that are sold to other countries to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exports to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        exports_to_gdp_ratio = economics.get_exports_to_gdp_ratio()

        exports_to_gdp_ratio.loc[:, ['United States', 'Canada', 'Russian Federation']]
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |         12.4112 |  31.85   |              28.7034 |
        | 2016 |         11.8882 |  31.5025 |              25.8545 |
        | 2017 |         12.1775 |  31.4544 |              26.0909 |
        | 2018 |         12.2871 |  32.3254 |              30.7933 |
        | 2019 |         11.7892 |  32.3527 |              28.4334 |
        | 2020 |         10.0736 |  29.4736 |              25.5222 |
        | 2021 |         10.791  |  31.2164 |              29.7712 |
        | 2022 |         11.6022 |  33.845  |              28.0257 |
        | 2023 |         11.0115 |  33.3693 |              23.083  |
        | 2024 |         10.7508 |  32.3514 |              21.242  |
        | 2025 |         10.5946 |  31.6492 |              21.2205 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        exports_to_gdp_ratio = gmdb_model.get_exports_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
        )

        if growth:
            exports_to_gdp_ratio = calculate_growth(
                exports_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        exports_to_gdp_ratio = exports_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return exports_to_gdp_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_imports(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Imports for a variety of countries over time from the Global Macro Database (GMDB).
        Imports are the total amount of goods and services produced in other countries that are
        bought by a country.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Imports

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01')

        imports = economics.get_imports()

        imports.loc[:, ['United States', 'Canada', 'Mexico']]
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

        if growth:
            imports = calculate_growth(
                imports,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        imports = imports.loc[self._start_date : self._end_date]

        return imports.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_imports_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Imports to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Imports to GDP Ratio is the ratio of the total amount of goods and services produced in other countries
        that are bought by a country to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Imports to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        imports_to_gdp_ratio = economics.get_imports_to_gdp_ratio()

        imports_to_gdp_ratio.loc[:, ['United States', 'Canada', 'Mexico']]
        ```

        Which returns:

        |      |   United States |   Canada |   Mexico |
        |:-----|----------------:|---------:|---------:|
        | 2010 |         15.8785 |  31.0407 |  30.2559 |
        | 2011 |         17.279  |  31.8203 |  31.7194 |
        | 2012 |         17.0378 |  32.2425 |  32.7185 |
        | 2013 |         16.3878 |  31.8991 |  31.9359 |
        | 2014 |         16.3984 |  32.6421 |  32.6354 |
        | 2015 |         15.2771 |  34.3149 |  36.2502 |
        | 2016 |         14.5644 |  33.8611 |  38.8746 |
        | 2017 |         14.9479 |  33.6467 |  39.438  |
        | 2018 |         15.1582 |  34.2745 |  41.1687 |
        | 2019 |         14.4693 |  33.8188 |  38.9323 |
        | 2020 |         13.0061 |  31.6831 |  37.6192 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        imports_to_gdp_ratio = gmdb_model.get_imports_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
        )

        if growth:
            imports_to_gdp_ratio = calculate_growth(
                imports_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        imports_to_gdp_ratio = imports_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return imports_to_gdp_ratio.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_current_account_balance(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Current Account Balance for a variety of countries over time from the Global Macro Database (GMDB).
        The Current Account Balance is the sum of the balance of trade (exports minus imports of goods and services),
        net factor income (such as interest and dividends) and net transfer payments (such as foreign aid).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Current Account Balance

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        current_account_balance = economics.get_current_account_balance()

        current_account_balance.loc[:, ['France', 'Germany', 'Italy']]
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

        if growth:
            current_account_balance = calculate_growth(
                current_account_balance,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        current_account_balance = current_account_balance.loc[
            self._start_date : self._end_date
        ]

        return current_account_balance.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_current_account_balance_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Current Account Balance to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Current Account Balance to GDP Ratio is the ratio of the sum of the balance of trade (exports minus imports of goods
        and services), net factor income (such as interest and dividends) and net transfer payments (such as foreign aid) to the
        Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Current Account Balance to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        current_account_balance_to_gdp_ratio = economics.get_current_account_balance_to_gdp_ratio()

        current_account_balance_to_gdp_ratio.loc[:, ['Poland', 'Turkey', 'United Kingdom']]
        ```

        Which returns:

        |      |   Poland |   Turkey |   United Kingdom |
        |:-----|---------:|---------:|-----------------:|
        | 2015 |   -1.292 |   -2.463 |           -4.948 |
        | 2016 |   -1.025 |   -2.55  |           -5.448 |
        | 2017 |   -1.156 |   -4.091 |           -3.493 |
        | 2018 |   -1.926 |   -1.831 |           -3.927 |
        | 2019 |   -0.246 |    1.974 |           -2.688 |
        | 2020 |    2.483 |   -4.335 |           -2.934 |
        | 2021 |   -1.245 |   -0.796 |           -0.437 |
        | 2022 |   -2.438 |   -5.056 |           -2.102 |
        | 2023 |    1.548 |   -3.983 |           -1.961 |
        | 2024 |    0.848 |   -2.161 |           -2.787 |
        | 2025 |   -0.024 |   -2.072 |           -2.829 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        current_account_balance_to_gdp_ratio = (
            gmdb_model.get_current_account_balance_to_gdp(
                gmd_dataset=self._gmbd_dataset
            )
        )

        if growth:
            current_account_balance_to_gdp_ratio = calculate_growth(
                current_account_balance_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        current_account_balance_to_gdp_ratio = current_account_balance_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return current_account_balance_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_government_debt(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Debt for a variety of countries over time from the Global Macro Database (GMDB).
        Government Debt is the total amount of money that a government owes to creditors.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Debt

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_debt = economics.get_government_debt()

        government_debt.loc[:, ['United States', 'Canada', 'Mexico']]
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

        if growth:
            government_debt = calculate_growth(
                government_debt,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_debt = government_debt.loc[self._start_date : self._end_date]

        return government_debt.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_government_debt_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Debt to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Debt to GDP Ratio is the ratio of the total amount of money that a government owes to creditors
        to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Debt to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_debt_to_gdp_ratio = economics.get_government_debt_to_gdp_ratio()

        government_debt_to_gdp_ratio.loc[:, ['Netherlands', 'Germany', 'China']]
        ```

        Which returns:

        |      |   Netherlands |   Germany |   China |
        |:-----|--------------:|----------:|--------:|
        | 2015 |        63.799 |    70.56  |  41.489 |
        | 2016 |        60.883 |    67.631 |  50.701 |
        | 2017 |        55.986 |    63.952 |  54.951 |
        | 2018 |        51.558 |    60.729 |  56.659 |
        | 2019 |        47.583 |    58.556 |  60.404 |
        | 2020 |        53.337 |    67.858 |  70.155 |
        | 2021 |        50.436 |    67.879 |  71.852 |
        | 2022 |        48.354 |    64.789 |  77.393 |
        | 2023 |        45.024 |    62.66  |  84.381 |
        | 2024 |        44.264 |    62.679 |  90.119 |
        | 2025 |        45.11  |    62.098 |  93.845 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_debt_to_gdp_ratio = gmdb_model.get_government_debt_to_gdp_ratio(
            gmd_dataset=self._gmbd_dataset
        )

        if growth:
            government_debt_to_gdp_ratio = calculate_growth(
                government_debt_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_debt_to_gdp_ratio = government_debt_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return government_debt_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_government_revenue(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Revenue for a variety of countries over time from the Global Macro Database (GMDB).
        Government Revenue is the total amount of money that a government collects from taxes and other sources.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Revenue

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2019-01-01')

        government_revenue = economics.get_government_revenue()

        government_revenue.loc[:, ['United Kingdom', 'Canada', 'Japan']]

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

        if growth:
            government_revenue = calculate_growth(
                government_revenue,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_revenue = government_revenue.loc[self._start_date : self._end_date]

        return government_revenue.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_government_revenue_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Revenue to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Revenue to GDP Ratio is the ratio of the total amount of money that a government collects from taxes
        and other sources to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Revenue to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_revenue_to_gdp_ratio = economics.get_government_revenue_to_gdp_ratio()

        government_revenue_to_gdp_ratio.loc[:, ['United States', 'Canada', 'Russian Federation']]
        ```

        Which returns:

        |      |   United States |   Canada |   Russian Federation |
        |:-----|----------------:|---------:|---------------------:|
        | 2015 |          31.501 |   39.957 |               31.887 |
        | 2016 |          30.977 |   40.3   |               32.916 |
        | 2017 |          30.4   |   40.343 |               33.361 |
        | 2018 |          30.014 |   41.022 |               35.544 |
        | 2019 |          30.011 |   40.572 |               35.682 |
        | 2020 |          30.646 |   41.413 |               35.164 |
        | 2021 |          31.576 |   42.519 |               35.44  |
        | 2022 |          32.385 |   41.143 |               34.2   |
        | 2023 |          29.211 |   41.919 |               34.257 |
        | 2024 |          29.897 |   41.274 |               35.446 |
        | 2025 |          30.06  |   41.238 |               36.466 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_revenue_to_gdp_ratio = (
            gmdb_model.get_government_revenue_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
        )

        if growth:
            government_revenue_to_gdp_ratio = calculate_growth(
                government_revenue_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_revenue_to_gdp_ratio = government_revenue_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return government_revenue_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_government_tax_revenue(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Tax Revenue for a variety of countries over time from the Global Macro Database (GMDB).
        Government Tax Revenue is the total amount of money that a government collects from taxes.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Tax Revenue

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_tax_revenue = economics.get_government_tax_revenue()

        government_tax_revenue.loc[:, ['Kenya', 'Nigeria', 'South Africa']]
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

        if growth:
            government_tax_revenue = calculate_growth(
                government_tax_revenue,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_tax_revenue = government_tax_revenue.loc[
            self._start_date : self._end_date
        ]

        return government_tax_revenue.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_government_tax_revenue_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Tax Revenue to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Tax Revenue to GDP Ratio is the ratio of the total amount of money that a government collects from taxes
        to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Tax Revenue to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_tax_revenue_to_gdp_ratio = economics.get_government_tax_revenue_to_gdp_ratio()

        government_tax_revenue_to_gdp_ratio.loc[:, ['United States', 'Canada', 'Mexico']]
        ```

        Which returns:

        |      |   United States |   Canada |   Mexico |
        |:-----|----------------:|---------:|---------:|
        | 2015 |         19.9356 |  12.3898 |  13.1765 |
        | 2016 |         19.5813 |  12.4966 |  13.8639 |
        | 2017 |         20.3141 |  12.612  |  13.4013 |
        | 2018 |         18.7359 |  13.0577 |  13.3632 |
        | 2019 |         18.8848 |  12.7417 |  13.4765 |
        | 2020 |         19.3433 |  13.5033 |  14.5167 |
        | 2021 |         20.6471 |  13.2199 |  14.1389 |
        | 2022 |         21.5601 |  12.826  |  13.6774 |
        | 2023 |         10.2238 |  14.0076 |  14.2666 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_tax_revenue_to_gdp_ratio = (
            gmdb_model.get_government_tax_revenue_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
        )

        if growth:
            government_tax_revenue_to_gdp_ratio = calculate_growth(
                government_tax_revenue_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_tax_revenue_to_gdp_ratio = government_tax_revenue_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return government_tax_revenue_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_government_expenditure(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Expenditure for a variety of countries over time from the Global Macro Database (GMDB).
        Government Expenditure is the total amount of money that a government spends on goods and services.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Expenditure

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_expenditure = economics.get_government_expenditure()

        government_expenditure.loc[:, ['Japan', 'China', 'India']]
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

        if growth:
            government_expenditure = calculate_growth(
                government_expenditure,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_expenditure = government_expenditure.loc[
            self._start_date : self._end_date
        ]

        return government_expenditure.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_government_expenditure_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Expenditure to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Expenditure to GDP Ratio is the ratio of the total amount of money that a government spends on goods
        and services to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Expenditure to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_expenditure_to_gdp_ratio = economics.get_government_expenditure_to_gdp_ratio()

        government_expenditure_to_gdp_ratio.loc[:, ['United States', 'Japan', 'Netherlands']]
        ```

        Which returns:

        |      |   United States |   Japan |   Netherlands |
        |:-----|----------------:|--------:|--------------:|
        | 2015 |          35.031 |  37.295 |        45.252 |
        | 2016 |          35.333 |  37.229 |        43.917 |
        | 2017 |          35.194 |  36.686 |        42.759 |
        | 2018 |          35.349 |  36.739 |        42.443 |
        | 2019 |          35.811 |  37.294 |        42.098 |
        | 2020 |          44.568 |  44.517 |        47.813 |
        | 2021 |          42.599 |  42.44  |        45.894 |
        | 2022 |          36.308 |  41.835 |        43.266 |
        | 2023 |          36.279 |  41.165 |        43.201 |
        | 2024 |          37.526 |  42.198 |        44.158 |
        | 2025 |          37.384 |  39.825 |        44.798 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_expenditure_to_gdp_ratio = (
            gmdb_model.get_government_expenditure_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
        )

        if growth:
            government_expenditure_to_gdp_ratio = calculate_growth(
                government_expenditure_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_expenditure_to_gdp_ratio = government_expenditure_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return government_expenditure_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
    def get_government_deficit(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Deficit for a variety of countries over time from the Global Macro Database (GMDB).
        Government Deficit is the total amount of money that a government spends more than it collects from taxes
        and other sources. A government deficit is usually financed by borrowing money.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Government Deficit

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2015-01-01')

        government_deficit = economics.get_government_deficit()

        government_deficit.loc[:, ['United States', 'Canada', 'Mexico']]
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

        if growth:
            government_deficit = calculate_growth(
                government_deficit,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_deficit = government_deficit.loc[self._start_date : self._end_date]

        return government_deficit.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_government_deficit_to_gdp_ratio(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Get the Government Deficit to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Government Deficit to GDP Ratio is the ratio of the total amount of money that a government spends more than it
        collects from taxes and other sources to the Gross Domestic Product (GDP). A government deficit is usually financed
        by borrowing money.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
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
        | 2015 |         0.365 |      -2.785 |           -4.622 |
        | 2016 |         0.978 |      -2.422 |           -3.338 |
        | 2017 |         1.358 |      -1.716 |           -2.505 |
        | 2018 |         1.271 |      -1.257 |           -2.273 |
        | 2019 |        -2.499 |      -4.404 |           -2.481 |
        | 2020 |        -4.328 |      -8.719 |          -13.145 |
        | 2021 |        -3.237 |      -6.353 |           -7.86  |
        | 2022 |        -3.511 |      -2.187 |           -4.703 |
        | 2023 |        -3.333 |      -0.857 |           -5.964 |
        | 2024 |        -3.845 |      -1.661 |           -4.25  |
        | 2025 |        -3.493 |      -2.043 |           -3.741 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        government_deficit_to_gdp_ratio = (
            gmdb_model.get_government_deficit_to_gdp_ratio(
                gmd_dataset=self._gmbd_dataset
            )
        )

        if growth:
            government_deficit_to_gdp_ratio = calculate_growth(
                government_deficit_to_gdp_ratio,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        government_deficit_to_gdp_ratio = government_deficit_to_gdp_ratio.loc[
            self._start_date : self._end_date
        ]

        return government_deficit_to_gdp_ratio.round(
            rounding if rounding else self._rounding
        )

    @handle_errors
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

    @handle_errors
    def get_consumer_price_index(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Consumer Price Index (CPI) is a measure that examines the average change in prices
        paid by consumers for goods and services over time. It is a measure of inflation.
        The base year (2010) is the year against which the index is set to 100.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Consumer Price Index.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2008-09-01', end_date='2020-03-01')

        consumer_price_index = economics.get_consumer_price_index()

        consumer_price_index.loc[:, ['Germany', 'France', 'Portugal']]
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
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        consumer_price_index = gmdb_model.get_consumer_price_index(
            gmd_dataset=self._gmbd_dataset
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

    @handle_errors
    def get_inflation_rate(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Inflation Rate is the percentage change in the Consumer Price Index (CPI) from one
        period to another. It is a measure of the rate of price increases in the economy.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Inflation Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2003-01-01', end_date='2009-03-01')

        inflation_rate = economics.get_inflation_rate()

        inflation_rate.loc[:, ['Germany', 'France', 'Portugal']]
        ```

        Which returns:

        |      |   Germany |   France |   Portugal |
        |:-----|----------:|---------:|-----------:|
        | 2003 |    1.0342 |   2.0985 |     3.219  |
        | 2004 |    1.6657 |   2.1421 |     2.3654 |
        | 2005 |    1.5469 |   1.7459 |     2.2772 |
        | 2006 |    1.5774 |   1.6751 |     3.1077 |
        | 2007 |    2.2983 |   1.488  |     2.454  |
        | 2008 |    2.6284 |   2.8129 |     2.5885 |
        | 2009 |    0.3127 |   0.0876 |    -0.8355 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        inflation_rate = gmdb_model.get_inflation_rate(gmd_dataset=self._gmbd_dataset)

        if growth:
            inflation_rate = calculate_growth(
                inflation_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        inflation_rate = inflation_rate.loc[self._start_date : self._end_date]

        return inflation_rate.round(rounding if rounding else self._rounding)

    @handle_errors
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

    @handle_errors
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

    @handle_errors
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

    @handle_errors
    def get_house_prices(
        self,
        quarterly: bool | None = None,
        inflation_adjusted: bool = False,
        gmdb_source: bool | None = None,
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

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Args:
            quarterly (bool | None, optional): Whether to return the quarterly data or the annual data.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data or the nominal data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
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

    @handle_errors
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

    @handle_errors
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

        economics = Economics(start_date="2013-01-01")

        share_prices = economics.get_share_prices()

        share_prices.loc[:, ['Turkey', 'Belgium', 'Australia']]
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
            else "quarterly" if self._quarterly else "yearly"
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

    @handle_errors
    def get_exchange_rates(
        self,
        period: str | None = None,
        gmdb_source: bool | None = None,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Exchange rates are defined as the price of one country's' currency in relation
        to another country's currency. This indicator is measured in terms of
        national currency per US dollar.

        See definition: https://data.oecd.org/conversion/exchange-rates.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Exchange Rates.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2000-01-01', end_date='2010-12-31')

        exchange_rates = economics.get_exchange_rates()

        exchange_rates.loc[:, ['Japan', 'Indonesia', "China"]]
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
            else "quarterly" if self._quarterly else "yearly"
        )
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            exchange_rates = gmdb_model.get_usd_exchange_rate(
                gmd_dataset=self._gmbd_dataset
            )
        else:
            exchange_rates = oecd_model.get_exchange_rates(period=period)

        if growth:
            exchange_rates = calculate_growth(
                exchange_rates,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        exchange_rates = exchange_rates.loc[self._start_date : self._end_date]

        return exchange_rates.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_money_supply(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        Money Supply is the total amount of money that is in circulation in a country.
        It includes currency, demand deposits, and other liquid assets that can be easily
        converted into cash. Money supply is an important economic indicator that the
        Federal Reserve uses to implement its monetary policy.

        Money supply can be divided into four categories: M0, M1, M2, M3 and M4.
            - M0: The total of all physical currency, plus accounts at the central bank that can be exchanged for physical currency.
            - M1: The total of all physical currency part of bank reserves + the amount in demand accounts ("checking" or "current" accounts).
            - M2: M1 + most savings accounts, money market accounts, retail money market mutual funds, and small denomination time deposits.
            - M3: M2 + large time deposits, institutional money market funds, short-term repurchase agreements, and other larger liquid assets.
            - M4: M3 + all other financial assets.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Money Supply

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2010-01-01', end_date='2020-12-31')

        money_supply = economics.get_money_supply()

        money_supply["M2"][["Netherlands", "Germany", "United States"]]
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

        if growth:
            money_supply = calculate_growth(
                money_supply,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        money_supply = money_supply.loc[self._start_date : self._end_date]

        return money_supply.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_central_bank_policy_rate(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        The Central Bank Policy Rate is the interest rate that a central bank sets on its
        loans and advances to a commercial bank. This interest rate is used by the monetary
        authorities to control inflation and stabilize the country's currency.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/files/GMD_TA.pdf

        Args:
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Central Bank Policy Rate

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2021-01-01', end_date='2025-12-31')

        central_bank_policy_rate = economics.get_central_bank_policy_rate()

        central_bank_policy_rate[["Netherlands", "Germany", "United States"]]
        ```

        Which returns:

        |      |   Netherlands |   Germany |   United States |
        |:-----|--------------:|----------:|----------------:|
        | 2021 |       -0.5    |   -0.5    |           0.125 |
        | 2022 |        0.4375 |    0.4375 |           4.375 |
        | 2023 |        3.625  |    3.625  |           5.375 |
        | 2024 |        3.8125 |    3.8125 |           4.375 |
        | 2025 |        2.875  |    2.875  |           4.255 |
        """
        if self._gmbd_dataset.empty:
            self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

        central_bank_policy_rate = gmdb_model.get_central_bank_policy_rate(
            gmd_dataset=self._gmbd_dataset
        )

        if growth:
            central_bank_policy_rate = calculate_growth(
                central_bank_policy_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        central_bank_policy_rate = central_bank_policy_rate.loc[
            self._start_date : self._end_date
        ]

        return central_bank_policy_rate.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_short_term_interest_rate(
        self,
        period: str | None = None,
        gmdb_source: bool | None = None,
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

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            growth (bool, optional): Whether to return the growth data or the actual data.
            lag (int, optional): The number of periods to lag the data by.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the Short Term Interest Rate.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2023-05-01')

        short_term_interest_rate = economics.get_short_term_interest_rate(period='quarterly')

        short_term_interest_rate.loc[:, ['Japan', 'United States', 'China']]
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
        """
        period = (
            period
            if period is not None
            else "quarterly" if self._quarterly else "yearly"
        )

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            short_term_interest_rate = gmdb_model.get_short_term_interest_rate(
                gmd_dataset=self._gmbd_dataset
            )
        else:
            short_term_interest_rate = oecd_model.get_short_term_interest_rate(
                period=period,
            )

        if growth:
            short_term_interest_rate = calculate_growth(
                short_term_interest_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        short_term_interest_rate = short_term_interest_rate.loc[
            self._start_date : self._end_date
        ]

        return short_term_interest_rate.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_long_term_interest_rate(
        self,
        period: str | None = None,
        gmdb_source: bool | None = None,
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

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
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
            else "quarterly" if self._quarterly else "yearly"
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
            )

        if growth:
            long_term_interest_rate = calculate_growth(
                long_term_interest_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        long_term_interest_rate = long_term_interest_rate.loc[
            self._start_date : self._end_date
        ]

        return long_term_interest_rate.round(rounding if rounding else self._rounding)

    @handle_errors
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

        renewable_energy.loc[:, ['Austria', 'Germany', 'United States']]
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

    @handle_errors
    def get_carbon_footprint(
        self,
        growth: bool = False,
        lag: int = 1,
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

        carbon_footprint = economics.get_carbon_footprint()

        carbon_footprint.loc[:, ["Germany", "United States", "Poland"]]
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
        carbon_footprint_df = oecd_model.get_carbon_footprint()

        if growth:
            carbon_footprint_df = calculate_growth(
                carbon_footprint_df,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        carbon_footprint_df = carbon_footprint_df.loc[self._start_date : self._end_date]

        return carbon_footprint_df.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_unemployment_rate(
        self,
        period: str | None = None,
        gmdb_source: bool | None = None,
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

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Args:
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
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
            else "quarterly" if self._quarterly else "yearly"
        )
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset()

            unemployment_rate = gmdb_model.get_unemployment_rate(
                gmd_dataset=self._gmbd_dataset
            )
        else:
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

    @handle_errors
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

    @handle_errors
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
        households of different sizes.

        The Gini coefficient is based on the comparison of cumulative proportions of the population against
        cumulative proportions of income they receive, and it ranges between 0 in the case of perfect equality
        and 1 in the case of perfect inequality.

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
        income_inequality = oecd_model.get_income_inequality()

        if growth:
            income_inequality = calculate_growth(
                income_inequality,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        income_inequality = income_inequality.loc[self._start_date : self._end_date]

        return income_inequality.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_population_statistics(
        self,
        gmdb_source: bool | None = None,
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

        See definition: https://data.oecd.org/pop/population.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Args:
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
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
        gmdb_source = gmdb_source if gmdb_source is not None else self._gmdb_source

        if gmdb_source:
            if self._gmbd_dataset.empty:
                self._gmbd_dataset = gmdb_model.collect_global_macro_database_dataset

            population_statistics = gmdb_model.get_population(
                gmd_dataset=self._gmbd_dataset
            )
        else:
            population_statistics = {}

            population_statistics["Population"] = oecd_model.get_population()
            population_statistics["Men"] = oecd_model.get_population(gender="men")
            population_statistics["Women"] = oecd_model.get_population(gender="women")

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

    @handle_errors
    def get_poverty_rate(
        self,
        growth: bool = False,
        lag: int = 1,
        rounding: int | None = None,
    ):
        """
        The poverty rate is the ratio of the number of people (in a given age group) whose income
        falls below the poverty line; taken as half the median household income of the total population.

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
        poverty_rate = oecd_model.get_poverty_rate()

        if growth:
            poverty_rate = calculate_growth(
                poverty_rate,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="rows",
            )

        poverty_rate = poverty_rate.loc[self._start_date : self._end_date]

        return poverty_rate.round(rounding if rounding else self._rounding)
