"""Long-Run Data and Scenarios Module"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd

from financetoolkit import helpers, historical_model, yfinance_model
from financetoolkit.economics.helpers import (
    buffered_start_date,
    resample_to_period,
    validate_period,
)
from financetoolkit.economics.prices.prices_controller import PricesEconomics
from financetoolkit.economics.sources import (
    boe_model,
    dnb_model,
    macrohistory_model,
    shiller_model,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    handle_errors,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import finalize_dataset

# ruff: noqa: E501

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


@filter_shown_columns
@track_sources
class HistoryEconomics(PricesEconomics):
    """
    Long-run data and scenario sets: long-run asset returns, the Bank of England's millennium of macroeconomic data, Shiller's stock market data, DNB's pension scenario set and asset class proxies.
    """

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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the yearly total returns of equity, housing, government bonds and bills of 18 advanced economies from
        1870, from the Jordà-Schularick-Taylor Macrohistory Database.

        The database is that of Jordà, Knoll, Kuvshinov, Schularick and Taylor (2019, "The Rate of Return on
        Everything, 1870-2015", Quarterly Journal of Economics) and runs to 2020. A century and a half of returns
        across wars, depressions and inflations is what calibrates the long-run trend, the equity and housing risk
        premia and the cycles of scenario generators for 50 to 100-year horizons.

        The asset classes are equity (dividends reinvested), housing (rental income included), long-term government
        bonds (coupons included), short-term government bills, risky assets (a wealth-weighted mix of equity and
        housing), safe assets (a mix of bonds and bills) and wealth (all of them weighted by their share of national
        wealth). The countries: Australia, Belgium, Canada, Denmark, Finland, France, Germany, Ireland, Italy, Japan,
        the Netherlands, Norway, Portugal, Spain, Sweden, Switzerland, the United Kingdom and the United States, with
        gaps in the war years.

        Returns are decimal fractions (0.08 for 8%), in local currency, and in real terms
        deflated by consumer price inflation.

        The database is licensed under CC BY-NC-SA 4.0: it may be used for non-commercial
        purposes with attribution, and its authors expressly exclude commercial data
        providers. Accept the licence to confirm the use is non-commercial; the data
        is retrieved from macrohistory.net when needed and is not distributed with the
        Finance Toolkit.

        See definition: https://www.macrohistory.net/database/

        Also known as: rate of return on everything, long-run returns, JST macrohistory,
        historical equity premium.

        Args:
            asset_class (str, optional): The asset class. Options are: "equity" (dividends reinvested),
                "housing" (rental income included), "bonds" (long-term government bonds, coupons included),
                "bills" (short-term government bills), "risky" (a wealth-weighted mix of equity and housing),
                "safe" (a mix of bonds and bills), "wealth" (all of them weighted by their share of national
                wealth). Defaults to "equity".
            countries (list[str] | str | None, optional): The countries to include. Defaults
                to None, which includes all of them.
            real (bool, optional): Whether to deflate the returns by consumer price
                inflation. Defaults to False.
            accept_licence (bool, optional): Confirms that the data is used for
                non-commercial purposes, as its licence (CC BY-NC-SA 4.0) requires. Defaults
                to False, which returns nothing.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
            raise InvalidArgumentError(
                "The Jordà-Schularick-Taylor Macrohistory Database is licensed under CC "
                "BY-NC-SA 4.0, for non-commercial use with attribution. Pass "
                "accept_licence=True to confirm the use is non-commercial."
            )
        check_option("asset_class", asset_class, macrohistory_model.ASSET_CLASSES)

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
        deflate_with: str | None = None,
        rolling: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the headline annual series of the Bank of England's "A millennium of macroeconomic data for the UK":
        British output, prices, rates and asset prices over centuries.

        The dataset (Thomas and Dimsdale, version 3.1) holds around eighty series of output, population, employment,
        prices, wages, interest rates, asset prices, exchange rates, money, credit and public finances, from as early
        as 1086 (GDP and population), 1209 (consumer prices), 1694 (Bank Rate) and 1700 (share prices) to 2016. It is
        the longest consistent record of interest rates, inflation and asset prices of any economy, for calibrating
        the very long-run behaviour of scenario generators.

        The series keep the names of the dataset, such as "Bank Rate", "Consols / long-term
        government bond yields", "Corporate bond yields", "Share prices", "Consumer price
        index", "Consumer price inflation", "House price index" and "$/£ exchange rate"; a
        second column of a series, such as its growth rate or its share of GDP, carries that
        unit in brackets. Rates, growth rates and shares of GDP are decimal fractions (0.05
        for 5%); levels, indices and exchange rates are returned as published.

        The dataset is not updated anymore. The workbook is large (around 28 MB), so it is
        cached for a year.

        Deflated with a country's prices, every series is in real terms: in the money of the latest year of
        that country's yearly consumer price index up to the end date (see
        get_consumer_price_index), such as the dollar oil price ("$ Oil prices") in 2025
        dollars deflated with the United States, which compares the oil price of 1864
        with that of 2008. A year before the first year of the consumer price index has no
        real value. Only deflate series in money, such as prices, in the currency of that
        country: rates, indices and growth rates are not amounts of money.

        - Real Value (t) = Value (t) * CPI (latest year) / CPI (t)

        See definition: https://www.bankofengland.co.uk/statistics/research-datasets

        Also known as: three centuries of macroeconomic data, BoE millennium dataset, long
        run UK data.

        Args:
            series (list[str] | str | None, optional): The series to include, matched on
                (part of) their name without regard to case, e.g. "Bank Rate" or ["consols",
                "share prices"]. Defaults to None, which includes all of them.
            deflate_with (str | None, optional): The country whose yearly consumer price index
                the series are deflated with, e.g. "United States" for the dollar oil price
                or "United Kingdom" for sterling series. Defaults to None, which returns the
                series as published.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            growth (bool, optional): Whether to return the growth data or the actual data. Defaults to False.
            lag (int, optional): The number of periods to lag the data by. Defaults to 1.
            standardize (bool, optional): Whether to standardize (Z-Score) the result. When
                combined with growth=True, standardizes the growth values instead of the raw
                values. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        if deflate_with is not None:
            consumer_price_index = self.get_consumer_price_index(
                countries=deflate_with, period="yearly", rounding=None
            )
            if consumer_price_index.empty:
                raise ValueError(
                    f"There is no yearly consumer price index for {deflate_with} to deflate "
                    "the series with."
                )
            millennium = helpers.deflate_with_price_index(
                millennium, consumer_price_index.iloc[:, 0]
            )

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
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get a regulator's published scenario set: the thousands of simulated paths of interest rates, equity returns
        and inflation that pension funds or insurers must use.

        The paths come from a documented and reviewed economic scenario generator. It is a reference to calibrate and
        validate an own scenario generator against, both for the real-world distribution and for the market-consistent
        one.

        Available is the scenario set of De Nederlandsche Bank (DNB), which
        Dutch pension funds use for their feasibility test and the transition to the new
        pension contract: 20,000 scenarios over a horizon of 100 years, generated every
        quarter with the model the Commissie Parameters 2022 specified, under the
        real-world measure (the P-set) and the market-consistent
        measure (the Q-set, arbitrage-free and calibrated to
        market prices including swaptions).

        The variables, as decimals per year of the horizon (year 0 is the end of the
        quarter the set is for): the annually compounded nominal zero rates "Nominal rate
        1Y", "5Y", "10Y", "20Y" and "30Y" and the euro area "Real rate EU 10Y", derived from
        the model's term structure, and the annual "Equity return", "Inflation EU" and
        "Inflation NL" (from year 1).

        By default the distribution per year is returned: the quantiles of the 20,000
        scenarios and their mean. On request every scenario is returned instead
        (20,000 columns per variable). The set is a large workbook (around 180 MB) that
        takes a minute or two to download and read the first time; it is cached for 30
        days.

        See definition: https://www.dnb.nl/voor-de-sector/open-boek-toezicht/sectoren/pensioenfondsen/

        Also known as: DNB scenarioset, uniform scenario set, CP2022 scenarios, P-set,
        Q-set, regulatory economic scenario generator.

        Args:
            authority (str, optional): The regulator. Options are: "dnb" (De Nederlandsche Bank). Defaults
                to "dnb".
            measure (str, optional): The probability measure. Options are: "real_world" (the P-set),
                "market_consistent" (the Q-set). Defaults to "real_world".
            variables (list[str] | str | None, optional): The variables to include. Options are:
                "Nominal rate 1Y", "Nominal rate 5Y", "Nominal rate 10Y", "Nominal rate 20Y",
                "Nominal rate 30Y", "Real rate EU 10Y", "Equity return", "Inflation EU" and
                "Inflation NL". Defaults to None, which includes all of them.
            quantiles (list[float] | None, optional): The quantiles of the distribution
                per year. Defaults to None, which is [0.01, 0.05, 0.25, 0.5, 0.75, 0.95,
                0.99].
            scenarios (bool, optional): Whether to return every scenario instead of the
                distribution. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Indexed by the year of the horizon, with a column per variable and
            quantile (or "Mean"), or per variable and scenario with every scenario.

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
        check_option("authority", authority, ("dnb",))
        check_option("measure", measure, tuple(dnb_model.MEASURES))

        quantiles = (
            [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            if quantiles is None
            else quantiles
        )
        if any(not 0 <= quantile <= 1 for quantile in quantiles):
            raise InvalidArgumentError(
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
            raise InvalidArgumentError(
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
        period: str = "daily",
        returns: bool = True,
        rolling: int | None = None,
        standardize: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get the total returns of listed funds that track asset classes without a public return index, such as private
        equity, infrastructure, hedge funds and private credit.

        The asset classes are private equity, infrastructure, hedge funds, private credit, real estate and investment
        grade and high yield credit. Their returns approximate those of the asset classes for calibrating a scenario
        generator, with two caveats: listed prices are marked to market daily, so they are more volatile and more
        correlated with equities than the appraisal-based returns of private funds, and they include the funds' fees.

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
        in US dollars; without returns the dividend-adjusted price itself is returned.
        The prices come from FinancialModelingPrep when an API key is set on the
        Economics class and from Yahoo Finance otherwise.

        Also known as: private market proxies, listed private equity, alternative asset
        returns, hedge fund replication.

        Args:
            asset_classes (list[str] | str | None, optional): The asset classes to include. Options are:
                "Private Equity", "Infrastructure", "Hedge Funds", "Merger Arbitrage",
                "Private Credit", "Real Estate", "Investment Grade Credit" and "High Yield
                Credit". Defaults to None, which includes all of them.
            period (str, optional): The frequency of the data. Options are: "daily" (daily data), "weekly" (weekly
                data), "monthly" (monthly data). Defaults to "daily".
            returns (bool, optional): Whether to return the total return per period instead
                of the dividend-adjusted price. Defaults to True.
            rolling (int, optional): The rolling window size to use for smoothing the data (simple
            moving average). Defaults to None.
            standardize (bool, optional): Whether to standardize (Z-Score) the result.
                Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The returns (or prices), indexed by date with a column per asset
            class.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-04-01', end_date='2026-09-30')

        economics.get_asset_class_proxies(
            asset_classes=['Private Equity', 'Infrastructure', 'Hedge Funds', 'Private Credit'],
            period='monthly',
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
            raise InvalidArgumentError(
                f"There is no proxy for {', '.join(unavailable)}. The asset classes are "
                f"{', '.join(ASSET_CLASS_PROXIES)}."
            )

        tickers = [ASSET_CLASS_PROXIES[name] for name in requested]
        start_date = buffered_start_date(self._start_date, period)
        # yfinance logs every ticker without prices; they are reported in one line below.
        with yfinance_model.quiet_yfinance():
            historical_data, _ = historical_model.get_historical_data(
                tickers,
                api_key=self._api_key or None,
                start=start_date,
                end=self._end_date,
                fill_nan=False,
                show_ticker_separation=False,
                cache=self._cache,
            )

        prices = (
            historical_data["Adj Close"]
            if not historical_data.empty and "Adj Close" in historical_data.columns
            else pd.DataFrame()
        )
        prices = prices[[ticker for ticker in tickers if ticker in prices.columns]]

        if missing := [
            name
            for name in requested
            if ASSET_CLASS_PROXIES[name] not in prices
            or prices[ASSET_CLASS_PROXIES[name]].isna().all()
        ]:
            logger.error(
                "No prices could be retrieved for the proxies of %s.",
                ", ".join(missing),
            )

        if prices.empty:
            return pd.DataFrame()

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
    def get_shiller_stock_market_data(
        self,
        rounding: int | None = None,
        growth: bool = False,
        lag: int = 1,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get Robert Shiller's monthly US stock market data from 1871: the S&P Composite price, dividends, earnings,
        interest rates and the CAPE ratio, nominal and real.

        It is the data behind "Irrational Exuberance" and the CAPE ratio, with the trailing price-earnings ratio (P/E)
        over the earnings of the last twelve months and the cyclically adjusted price-earnings ratio (CAPE) over the
        average of ten years of real earnings. With more than 150 years of history it is the standard source for
        long-run equity returns, valuations and their relation to subsequent returns.

        Prices are monthly averages of daily closes; dividends and earnings are trailing
        twelve-month totals, with the latest months not yet published. The long interest
        rate is a decimal fraction (0.0475 for 4.75%).

        The CAPE is mostly used as a long-run valuation gauge: across the history of the data,
        higher starting CAPE ratios have gone with lower average returns over the following
        ten years, while saying little about returns over the next year, so it is not a timing
        tool. Comparisons with the distant past are imperfect, as accounting rules, payout
        policy (buybacks instead of dividends) and the composition of the index have changed.
        The Total Return CAPE corrects for the shift in payout policy.

        Shiller, R. J. (2015). Irrational Exuberance (3rd ed.). Princeton University Press.

        See definition: https://shillerdata.com/

        Also known as: Shiller data, CAPE, Shiller PE, long-run stock returns.

        Args:
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Economics class.
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
            pd.DataFrame: One column per measure, indexed by month.

        As an example:

        ```python
        from financetoolkit import Economics

        economics = Economics(start_date='2026-01-01', end_date='2026-09-30')

        economics.get_shiller_stock_market_data(show_columns=['Price', 'Earnings', 'P/E', 'CAPE'])
        ```

        Which returns:

        |         |   Price |   Earnings |     P/E |    CAPE |
        |:--------|--------:|-----------:|--------:|--------:|
        | 2026-01 | 6929.12 |    237.839 | 29.1336 | 39.6591 |
        | 2026-02 | 6893.81 |    239.865 | 28.7403 | 39.0501 |
        | 2026-03 | 6654.42 |    241.891 | 27.51   | 37.0914 |
        | 2026-04 | 6957.01 |    248.79  | 27.9633 | 38.2345 |
        | 2026-05 | 7412.55 |    255.69  | 28.9904 | 40.246  |
        | 2026-06 | 7450.03 |    262.589 | 28.3715 | 40.3455 |
        | 2026-07 | 7481.34 |    273.565 | 27.3475 | 40.2609 |
        | 2026-08 | 7711.32 |    284.542 | 27.1008 | 41.0871 |
        | 2026-09 | 7669.41 |    295.518 | 25.9524 | 40.2683 |
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
