"""Output and Trade Module"""

__docformat__ = "google"

from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    buffered_start_date,
)
from financetoolkit.economics.sources import (
    eurostat_model,
    gmdb_model,
    oecd_model,
    ons_model,
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
class OutputEconomics(EconomicsBase):
    """
    National accounts and trade: GDP, its growth and deflator, real GDP per capita, the output gap, consumption, investment, exports, imports, the trade balance and the current account.
    """

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Gross Domestic Product for a variety of countries over
        time. The Gross Domestic Product is the total value of goods produced
        and services provided in a country during one year. By default the data
        comes from the Global Macro Database (GMDB); set gmdb_source=False to
        retrieve it from the OECD instead.

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

        The inflation adjusted and US dollar figures are only available from the Global Macro
        Database (GMDB), so they always come from it. For more information see:
        https://www.globalmacrodata.com/documentation.html

        Also known as: GDP, national income, economic growth, real GDP, real GDP in USD, cross-country comparable GDP.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            inflation_adjusted (bool, optional): Whether to return the inflation adjusted data. Defaults to False.
            gmdb_source (bool | None, optional): Whether to retrieve the data from the GMDB. Defaults to None, which falls back to the gmdb_source set on the Economics class (True by default).
            usd (bool, optional): Whether to return the values in millions of US dollars, converted by the Global Macro Database, instead of millions of national currency, which makes levels comparable across countries. With inflation_adjusted=True, real GDP in millions of US dollars at constant 2015 prices, converted at the 2015 market exchange rate (not at purchasing power parities), so its growth equals the growth of real GDP in the national currency. Always from the GMDB. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            gross_domestic_product = self._get_gmdb_series(
                variable, gmdb_forecasts=gmdb_forecasts
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
        show_columns: str | list[str] | None = None,
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

        Every country comes from the most current source: Eurostat for the euro area and
        the countries it covers (including the preliminary flash estimate
        for the euro area, about thirty days after the quarter ends), the Office for
        National Statistics for the United Kingdom and the OECD Quarterly National Accounts
        for every other country, including the United States and Japan.

        The growth is the source's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - GDP Growth (quarter on quarter) = Real GDP (t) / Real GDP (t-1) - 1
        - GDP Growth (year over year) = Real GDP (t) / Real GDP (t-4) - 1

        Where Real GDP is the seasonally and calendar adjusted GDP in chain-linked volumes.

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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Gross Domestic Product Deflator for a variety of countries over
        time from the Global Macro Database (GMDB). The GDP deflator is a measure of
        the price of all domestically produced final goods and services in an economy
        relative to the price level in a base year.

        The deflator is an annual index with 2015 = 100 for every country.

        The GDP deflator is the broadest measure of domestic inflation: it covers everything
        produced in the economy, including investment goods and government services, and
        leaves out imports, whereas the Consumer Price Index follows a fixed basket of what
        households buy, imports included. Its yearly change is compared with CPI inflation;
        a gap between the two often points to import prices or the terms of trade, as during
        an energy shock.

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Real Gross Domestic Product per Capita for a variety of countries over time from
        the Global Macro Database (GMDB). This is the inflation-adjusted Gross Domestic Product
        (GDP) divided by the total population of a country, which gives an indication of the
        average economic output (and by extension, living standard) per person.

        This uses the Global Macro Database's own precomputed per-capita series (the source's figure)
        rather than dividing GDP by population manually, which avoids subtle mismatches that can arise
        from differences in population coverage or timing between the two underlying series.

        The formula is as follows:

        - Real GDP per Capita = Real Gross Domestic Product / Population, in national currency at constant
          prices, or in US dollars as converted by the Global Macro Database when usd=True

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        Also known as: real GDP per capita, real income per capita, standard of living.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            usd (bool, optional): Whether to return the values in US dollars per person, converted by the Global Macro Database, instead of national currency per person, which makes levels comparable across countries. Defaults to False.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the growth data. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            "rGDP_pc_USD" if usd else "rGDP_pc", gmdb_forecasts=gmdb_forecasts
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
        show_columns: str | list[str] | None = None,
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

        The output gap is the OECD's own estimate (the source's figure, divided by 100 here).

        The formula is as follows:

        - Output Gap = (Actual GDP - Potential GDP) / Potential GDP, with both GDP measures in real terms

        This data is only available on a yearly basis, since the OECD Economic Outlook is
        published as a set of annual projections and estimates.

        The result is a decimal fraction of potential GDP (-0.0422), where the OECD publishes
        a percentage (-4.2231), like every other rate and ratio in this class. Multiply by 100
        to recover the published figure. A small gap loses resolution at the default rounding
        of 4 decimals, so pass a larger `rounding` when the sub-basis-point detail matters.

        Potential output cannot be observed, so the gap is an estimate, and the estimate for a
        given year is often revised substantially as later data comes in: the gap estimated
        at the time can differ by a percentage point or more from the one published years
        later. Treat recent values as provisional and compare the gap with other measures of
        slack, such as unemployment or capacity utilisation.

        See definition: https://www.oecd.org/en/data/indicators/output-gaps.html

        Also known as: business cycle gap, GDP gap.

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
                    gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
                )
                if variable == "cons"
                else (
                    self._get_gmdb_series(variable, gmdb_forecasts=gmdb_forecasts)
                    / self._get_gmdb_series("deflator", gmdb_forecasts=gmdb_forecasts)
                    * 100
                ).dropna(how="all", axis="columns")
            )
        else:
            total_consumption = self._get_gmdb_series(
                f"{variable}_USD" if usd else variable, gmdb_forecasts=gmdb_forecasts
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Total Consumption to GDP Ratio for a variety of countries over time from the
        Global Macro Database (GMDB). The Total Consumption to GDP Ratio is final consumption
        expenditure (households, the non-profit institutions serving them and the government)
        as a share of the Gross Domestic Product (GDP). Select the household or government
        part with the component parameter.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.8100 for 81.00% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Consumption to GDP Ratio = Final Consumption Expenditure / Nominal GDP, with the total (households, the
          non-profit institutions serving them and the government), household or government part selected with
          `component` ("total" by default)

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            f"{self._consumption_variable(component)}_GDP",
            in_percent=True,
            gmdb_forecasts=gmdb_forecasts,
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Investment for a variety of countries over time from the Global Macro Database (GMDB).
        Investment is gross capital formation: what businesses, households and the government spend on
        fixed assets such as machinery, equipment, buildings and dwellings, plus the change in inventories.

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        investment = self._get_gmdb_series(
            "inv_USD" if usd else "inv", gmdb_forecasts=gmdb_forecasts
        )

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Investment to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Investment to GDP Ratio is gross capital formation (what businesses, households and the government
        spend on fixed assets such as machinery, equipment, buildings and dwellings, plus the change in
        inventories) as a share of the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.2248 for 22.48% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Investment to GDP Ratio = Gross Capital Formation / Nominal GDP, with Gross Capital Formation = Gross Fixed
          Capital Formation + Change in Inventories

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Fixed Investment for a variety of countries over time from the Global Macro Database (GMDB).
        Fixed Investment is gross fixed capital formation: what businesses, households and the government
        spend on fixed assets such as machinery, equipment, buildings and dwellings, without the change in
        inventories.

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        fixed_investment = self._get_gmdb_series(
            "finv_USD" if usd else "finv", gmdb_forecasts=gmdb_forecasts
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Fixed Investment to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Fixed Investment to GDP Ratio is gross fixed capital formation (what businesses, households and the
        government spend on fixed assets such as machinery, equipment, buildings and dwellings, without the change
        in inventories) as a share of the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.2179 for 21.79% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Fixed Investment to GDP Ratio = Gross Fixed Capital Formation / Nominal GDP

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: A DataFrame containing the Fixed Investment to GDP Ratio

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2016-01-01', end_date="2025-12-31")

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
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        exports = self._get_gmdb_series(
            "exports_USD" if usd else "exports", gmdb_forecasts=gmdb_forecasts
        )

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Exports to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Exports to GDP Ratio is the ratio of the total amount of goods and services produced in a country
        that are sold to other countries to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.1016 for 10.16% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Exports to GDP Ratio = Exports of Goods and Services / Nominal GDP

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        imports = self._get_gmdb_series(
            "imports_USD" if usd else "imports", gmdb_forecasts=gmdb_forecasts
        )

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Imports to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Imports to GDP Ratio is the ratio of the total amount of goods and services produced in other countries
        that are bought by a country to the Gross Domestic Product (GDP).

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (0.1213 for 12.13% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Imports to GDP Ratio = Imports of Goods and Services / Nominal GDP

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Trade Balance for a variety of countries over time from the Global Macro
        Database (GMDB). The Trade Balance is the difference between the total value of goods
        and services a country exports and the total value of goods and services it imports. A
        positive trade balance (a "trade surplus") means a country exports more than it imports,
        while a negative trade balance (a "trade deficit") means a country imports more than it
        exports.

        The formula is as follows:

        - Trade Balance = Exports of Goods and Services - Imports of Goods and Services, both the Global Macro
          Database's figures in millions of national currency (or of US dollars when usd=True)

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        exports = self._get_gmdb_series(
            "exports_USD" if usd else "exports", gmdb_forecasts=gmdb_forecasts
        )
        imports = self._get_gmdb_series(
            "imports_USD" if usd else "imports", gmdb_forecasts=gmdb_forecasts
        )

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        current_account_balance = self._get_gmdb_series(
            "CA_USD" if usd else "CA", gmdb_forecasts=gmdb_forecasts
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Current Account Balance to GDP Ratio for a variety of countries over time from the Global Macro Database (GMDB).
        The Current Account Balance to GDP Ratio is the ratio of the sum of the balance of trade (exports minus imports of goods
        and services), net factor income (such as interest and dividends) and net transfer payments (such as foreign aid) to the
        Gross Domestic Product (GDP).

        The ratio shows whether a country saves more than it invests (a surplus) or relies on foreign financing (a
        deficit). Persistent deficits are not a problem in themselves, but they leave a country exposed when capital
        inflows stop, which is why emerging markets with large deficits are watched closely for currency stress. The
        European Union's macroeconomic imbalance procedure uses a three-year average between -4% and +6% of GDP as its
        threshold.

        Data comes from the Global Macro Database (GMDB), further information about the
        variable can be found within https://www.globalmacrodata.com/documentation.html

        The ratio is expressed as a decimal fraction (-0.0211 for -2.11% of GDP).

        The ratio is the Global Macro Database's own figure, published in percent and divided by 100 here.

        The formula is as follows:

        - Current Account Balance to GDP Ratio = (Exports - Imports of Goods and Services + Net Primary Income + Net
          Secondary Income) / Nominal GDP

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
            gmdb_forecasts (bool, optional): Whether to include the years the Global Macro
                Database projects, from the IMF's World Economic Outlook, up to five years
                ahead. Only applies to data from the Global Macro Database. Defaults to
                False, which ends a series with its last observation.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
