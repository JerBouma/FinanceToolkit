"""Jobs and Society Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    buffered_start_date,
    check_period_type,
    extend_with_recent,
    to_gmdb_country_names,
)
from financetoolkit.economics.sources import (
    bls_model,
    dol_model,
    eurostat_model,
    fred_model,
    gmdb_model,
    ibge_model,
    oecd_model,
    ons_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import (
    collect_sources,
    track_sources,
)
from financetoolkit.utilities.statistics_model import (
    finalize_dataset,
)

# ruff: noqa: E501

logger = get_logger()


@filter_shown_columns
@track_sources
class JobsEconomics(EconomicsBase):
    """
    Labour and society: unemployment, nonfarm payrolls, jobless claims, the misery index, labour productivity, income inequality, poverty, population, life tables and household savings and debt.
    """

    def _get_oecd_unemployment_rate_with_bls(self) -> pd.DataFrame:
        """
        The OECD's monthly unemployment rates, with the US extended by the months the BLS
        has published since (see extend_with_recent).

        Returns:
            pd.DataFrame: One column per country, indexed by month.
        """
        history = oecd_model.get_unemployment_rate(
            period="monthly", start_date=self._start_date, end_date=self._end_date
        )

        with collect_sources() as recent_sources:
            recent = bls_model.get_unemployment_rate()

        return extend_with_recent(history, recent, "United States", recent_sources)

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
        show_columns: str | list[str] | None = None,
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
            quarterly (bool | None, optional): Whether to return the quarterly data (True) or the
                annual data (False). Defaults to None, which returns the quarterly data, the
                highest frequency the OECD publishes.
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
        # Quarterly is the highest frequency these indicators are published at.
        quarterly = True if quarterly is None else quarterly

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
        show_columns: str | list[str] | None = None,
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
            quarterly (bool | None, optional): Whether to return the quarterly data (True) or the
                annual data (False). Defaults to None, which returns the quarterly data, the
                highest frequency the OECD publishes.
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
        # Quarterly is the highest frequency these indicators are published at.
        quarterly = True if quarterly is None else quarterly

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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
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

        The unemployment rate is a lagging indicator: it tends to rise only once a slowdown
        is under way, so its change matters more than its level. The Sahm rule, for example,
        flags a US recession when the three-month average rises 0.5 percentage points above
        its low of the previous twelve months. Levels differ between countries for structural
        reasons (labour market rules, participation), so compare a country with its own
        history or its estimated natural rate rather than with other countries.

        See definition: https://data.oecd.org/unemp/unemployment-rate.htm

        It is also possible to get the data from the Global Macro Database (GMDB) by setting
        the gmdb_source to True.

        Both sources return the rate as a decimal fraction of the labour force (0.036 for
        3.6%), so the two are directly interchangeable. Only the OECD source supports
        monthly and quarterly frequency; the GMDB is annual only, so a monthly or quarterly
        request uses the OECD unless gmdb_source=True is passed, in which case the period
        argument has no effect.

        With period="monthly" (unless gmdb_source=True is passed explicitly) the euro area
        and the countries Eurostat covers come from Eurostat and the United Kingdom from
        the Office for National Statistics, which publish weeks before the OECD republishes
        their figures, and Brazil from the IBGE (as the UK, the average of the three months
        ending in each month); every other country comes from the OECD. The United States is
        extended with the months the Bureau of Labor Statistics has published since, when
        both agree on the months they share. The UK figure is the average of the three
        months ending in each month, as the Labour Force Survey reports it.

        Both sources return a decimal fraction (0.036 for 3.6%).

        Also known as: jobless rate, labor market, unemployment level.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            period (str | None, optional): Whether to return the monthly, quarterly or the annual data.
                Defaults to None, which is quarterly when the Economics class was initialized with
                quarterly=True and monthly otherwise.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
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

        period = self._default_period(period, "monthly")
        # The GMDB is annual, so a monthly request goes to the monthly sources unless the
        # GMDB is asked for explicitly.
        monthly = period.lower() == "monthly" and gmdb_source is not True
        # The GMDB is annual, so a monthly or quarterly request goes to the sources that
        # have that frequency unless the GMDB is asked for explicitly.
        gmdb_source = (
            gmdb_source
            if gmdb_source is not None
            else self._gmdb_source and period.lower() not in ("monthly", "quarterly")
        )

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
                    (self._get_oecd_unemployment_rate_with_bls, None),
                ],
                countries,
            )
        elif gmdb_source:

            unemployment_rate = gmdb_model.get_unemployment_rate(
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Misery Index for a variety of countries over time. The Misery Index is a simple
        gauge of the overall economic discomfort felt by the average person, combining the two
        economic ills that are most directly and visibly felt by households: unemployment and
        rising prices.

        The formula is as follows:

        - Misery Index (year) = Unemployment Rate (year) + Inflation Rate (year), with the annual unemployment rate
          from the Global Macro Database (or the OECD with gmdb_source=False) and the Global Macro Database's annual
          CPI inflation

        The Unemployment Rate and Inflation Rate are both retrieved as annual decimal fractions
        (0.05 for 5%) whichever source is used, so they line up directly for the addition and
        the result is itself a decimal fraction (0.0774 for a Misery Index of 7.74).

        A higher Misery Index indicates a more uncomfortable economic climate for the average
        household, while a lower value indicates a more comfortable one. It was originally
        popularized by economist Arthur Okun.

        Also known as: economic discomfort index, Okun's misery index.

        Args:
            countries (list[str] | str | None, optional): A list of countries or a single country to include in the results. Defaults to None.
            gmdb_source (bool | None, optional): Whether to get the unemployment rate from the
                Global Macro Database (GMDB) instead of the OECD. Defaults to None, which falls
                back to the gmdb_source set on the Economics class (True by default).
                Either way the countries carry the GMDB names, such as "Russian Federation",
                "Eswatini" and "Congo DR", as the inflation rate comes from the GMDB.
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
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
            )
        else:
            unemployment_rate = oecd_model.get_unemployment_rate(
                period="yearly", start_date=self._start_date, end_date=self._end_date
            )

            # The inflation leg comes from the GMDB, which names a few countries differently.
            unemployment_rate = to_gmdb_country_names(unemployment_rate)

        inflation_rate = gmdb_model.get_inflation_rate(
            gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        show_columns: str | list[str] | None = None,
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

        The level is the OECD's own figure (the source's calculation).

        The formula is as follows:

        - Labour Productivity = Real GDP (constant prices, PPP-converted US dollars) / Total Hours Worked by all
          persons engaged in production

        See definition: https://data.oecd.org/lprdty/gdp-per-hour-worked.htm

        Also known as: labor productivity, output per worker.

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
        show_columns: str | list[str] | None = None,
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
        gmdb_forecasts: bool = False,
        show_columns: str | list[str] | None = None,
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

        By default the data comes from the Global Macro Database (GMDB), which returns a
        single total population series per country, in millions of people. Set
        gmdb_source=False for the OECD series, which breaks the total down by gender,
        giving a Population, Men and Women series for each country, and reports a plain
        count of persons rather than millions. Both are annual, and the OECD source uses
        the historical (observed) series rather than its projections.

        See definition: https://data.oecd.org/pop/population.htm

        Also known as: demographic data, census data.

        Args:
            countries (list[str] | str | None, optional): The countries to include in the data. Defaults to None.
            gmdb_source (bool | None, optional): Whether to get the data from the Global Macro Database (GMDB).
                Defaults to None, which falls back to the gmdb_source set on the Economics class (True by default).
            rolling (int, optional): The rolling window size to use for smoothing the data (simple moving average). Defaults to None.
            trailing (int, optional): The trailing window size to use for summing the data over trailing periods (e.g. a trailing-4-quarter sum). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
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
                gmd_dataset=self._get_gmdb_dataset(gmdb_forecasts=gmdb_forecasts)
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
        show_columns: str | list[str] | None = None,
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
    def get_nonfarm_payrolls(
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
        Get Total Nonfarm Payroll Employment for the United States from the Bureau of
        Labor Statistics.

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

        Without a FRED API key the data comes from the Bureau of Labor Statistics' API,
        which gives the same figures but only for about the last three years. The
        full history, from 1939, comes from FRED when a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class).

        See definition: https://fred.stlouisfed.org/series/PAYEMS

        Also known as: NFP, nonfarm employment, the "jobs report".

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
        # FRED republishes these figures, so it is used when a key is set and the
        # publisher otherwise.
        nonfarm_payrolls = (
            fred_model.get_nonfarm_payrolls(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else bls_model.get_nonfarm_payrolls()
        )

        if (
            not self._fred_api_key
            and not nonfarm_payrolls.empty
            and nonfarm_payrolls.index.min().to_timestamp()
            > pd.Timestamp(self._start_date)
        ):
            logger.warning(
                "Without a FRED API key, nonfarm payrolls start in %s: the keyless BLS API "
                "only returns about the last three years. Set a FRED API key for the "
                "history from 1939.",
                nonfarm_payrolls.index.min().strftime("%Y-%m"),
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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get weekly Initial Claims for Unemployment Insurance for the United States
        from the Department of Labor.

        Initial Jobless Claims counts the number of individuals filing for
        unemployment insurance for the first time in a given week. Because it is
        reported weekly (versus Nonfarm Payrolls' monthly cadence, see
        `get_nonfarm_payrolls`) and captures layoffs essentially in real time, it is
        one of the most timely leading indicators of labor-market deterioration and a
        core component of the Conference Board's Leading Economic Index.

        Without a FRED API key the data comes from the Department of Labor's weekly claims
        report directly, which gives the same figures, although FRED adds the latest weeks
        before the report includes them. When a FRED API key is set (see the
        `fred_api_key` parameter of the `Economics` class), it comes from FRED.

        See definition: https://fred.stlouisfed.org/series/ICSA

        Also known as: initial claims, new unemployment claims.

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
            pd.DataFrame: A single-column ("United States") DataFrame of weekly initial
            jobless claims, seasonally adjusted.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(
            start_date='2020-01-01',
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
        # FRED republishes these figures, so it is used when a key is set and the
        # publisher otherwise.
        initial_jobless_claims = (
            fred_model.get_initial_jobless_claims(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else dol_model.get_initial_jobless_claims()
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
    def get_life_table(
        self,
        countries: list[str] | str | None = None,
        measure: str = "death_probability",
        sex: str = "total",
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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

        For countries outside Europe the UN World Population Prospects
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
