"""Environment Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.economics.base.base_controller import EconomicsBase
from financetoolkit.economics.helpers import (
    buffered_start_date,
    resample_to_period,
    validate_period,
)
from financetoolkit.economics.sources import eex_model, ngfs_model, oecd_model
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
class EnvironmentEconomics(EconomicsBase):
    """
    Environment and climate: renewable energy, the carbon footprint, the EU carbon price and the NGFS climate scenarios.
    """

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
        show_columns: str | list[str] | None = None,
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
        show_columns: str | list[str] | None = None,
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
    def get_carbon_price(
        self,
        period: str = "daily",
        rolling: int | None = None,
        trailing: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the price of emitting one tonne of CO2 under the EU Emissions Trading System
        (EU ETS): the clearing price of the primary auctions of EU emission allowances
        (EUAs) the European Energy Exchange holds for the European Union, Germany and Poland
        almost every trading day, daily from 2020. The carbon price is the main transition
        risk variable of climate scenarios; see get_climate_scenario for its projected paths.

        The price is the average of the day's auctions of general allowances, in euro per
        tonne of CO2. Weekly and monthly periods take the price of the
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
        show_columns: str | list[str] | None = None,
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

        The data is the Phase 5 vintage (November 2024), hosted by IIASA. The NGFS
        licence allows commercial use with attribution but not the redistribution of
        substantial parts, so it is retrieved when needed rather
        than stored with the Finance Toolkit.

        See definition: https://www.ngfs.net/ngfs-scenarios-portal/

        Also known as: NGFS scenarios, climate stress test scenarios, transition risk,
        physical risk, carbon price path.

        Args:
            variable (str, optional): The variable: "gdp", "inflation",
                "long_term_interest_rate", "long_term_real_interest_rate", "policy_rate",
                "unemployment_rate", "equity_prices", "effective_exchange_rate",
                "domestic_demand", "exports", "imports", "oil_price", "gas_price",
                "coal_price" or "carbon_price" (units above). Defaults to
                "long_term_interest_rate".
            scenario (str, optional): "Net Zero 2050", "Below 2°C", "Low demand", "Delayed
                transition", "Nationally Determined Contributions (NDCs)", "Current
                Policies", "Fragmented World" or "Baseline". Defaults to "Net Zero 2050".
            model (str, optional): The integrated assessment model behind the scenario:
                "REMIND-MAgPIE", "GCAM" or "MESSAGEix-GLOBIOM". Defaults to "REMIND-MAgPIE".
            countries (list[str] | str | None, optional): The countries or regions to
                include. Defaults to None, which includes all of them.
            risk (str, optional): "combined", "transition" or "physical". Current Policies has
                physical risk only, which is also its combined risk. Defaults to "combined".
            deviation (bool, optional): Whether to return the deviation from the baseline
                instead of the level. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            # Current Policies has no transition (that is the scenario), so NGFS publishes
            # its physical risk only, which is then also its combined risk.
            if scenario == "Current Policies" and risk == "combined":
                risk = "physical"
            elif scenario == "Current Policies" and risk == "transition":
                raise ValueError(
                    "Current Policies has no transition risk: it is the scenario without a "
                    "transition. Use risk='physical' or 'combined' for its physical risk."
                )
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
