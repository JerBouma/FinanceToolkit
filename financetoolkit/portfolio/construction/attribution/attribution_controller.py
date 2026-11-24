"""Portfolio Attribution Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.attribution import attribution_model
from financetoolkit.portfolio.construction.optimization import estimation_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    handle_errors,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501

logger = get_logger()

# The trading days an asset's first price may come after the first day of an attribution.
START_PRICE_TOLERANCE_DAYS = 5

# The sub-periods of a linked attribution and their pandas frequencies. Days are left out:
# every sub-period is an attribution of its own, which over years of days takes long.
SUB_PERIODS = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


@filter_shown_columns
@track_sources
class AttributionConstruction:
    """
    Exposures and attribution: the stress test, the look-through exposure, the Brinson attribution and the
    factor attribution.
    """

    def _get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """The daily prices of other tickers over a date range."""
        if self._price_function is None:
            raise InvalidArgumentError(
                "The prices of other tickers come from the Toolkit. Please use this method from a "
                "Toolkit (toolkit.portfolio), or pass price_function when creating the "
                "PortfolioConstruction class."
            )
        return self._price_function(tickers, start, end)

    @handle_errors
    def get_stress_test(
        self,
        weights: dict[str, float] | None = None,
        scenarios: dict[str, tuple[str, str]] | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how a portfolio would have fared in historical market crises.

        It shows the return of every asset and of the buy-and-hold portfolio from the start to
        the end of each scenario, next to the benchmark. The default scenarios run from the peak
        to the trough of the S&P 500: the Dot-com Crash, the Global Financial Crisis, the Euro
        Area Debt Crisis, the 2015-2016 and 2018 Q4 sell-offs, the COVID-19 Crash and the 2022
        Rate Shock.

        The prices are retrieved for each scenario, so the scenarios do not have to fall
        within the Toolkit's date range.

        Also known as: historical scenario analysis, crisis replay, what-if analysis.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            scenarios (dict[str, tuple[str, str]] | None, optional): Custom scenarios as
                {"name": ("start date", "end date")}. Defaults to None, the scenarios
                above.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per scenario its Start Date and End Date, the Portfolio Return,
            the Benchmark Return, the Coverage (the share of the portfolio's gross
            exposure with prices in the scenario, from 0 to 1) and the return of every
            asset.

        Notes:
        - An asset that was not listed yet at the start of a scenario has no return there;
          the other weights are scaled up to a net weight of one and the Coverage shows
          the share of the gross exposure, Σ |w_i| over the assets with prices / Σ |w_i|,
          that had data.
        - When the assets with prices have a net weight of zero or less (only the short
          positions of a long-short portfolio), they cannot be scaled up to the
          portfolio, so the Portfolio Return of that scenario is empty and a warning is
          logged, as it is when none of the assets has prices.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_stress_test()
        ```

        Which returns:

        | Scenario                | Start Date   | End Date   |   Portfolio Return |   Benchmark Return |   Coverage |    AAPL |    MSFT |     JNJ |     XOM |
        |:------------------------|:-------------|:-----------|-------------------:|-------------------:|-----------:|--------:|--------:|--------:|--------:|
        | Dot-com Crash           | 2000-03-24   | 2002-10-09 |            -0.22   |            -0.4751 |          1 | -0.8044 | -0.6061 |  0.6314 | -0.1009 |
        | Global Financial Crisis | 2007-10-09   | 2009-03-09 |            -0.3836 |            -0.5519 |          1 | -0.504  | -0.4817 | -0.2658 | -0.2829 |
        | Euro Area Debt Crisis   | 2011-04-29   | 2011-10-03 |            -0.0479 |            -0.1861 |          1 |  0.0697 | -0.0415 | -0.0386 | -0.1812 |
        | 2015-2016 Sell-off      | 2015-05-21   | 2016-02-11 |            -0.0673 |            -0.1298 |          1 | -0.2769 |  0.0619 |  0.008  | -0.0624 |
        | 2018 Q4 Sell-off        | 2018-09-20   | 2018-12-24 |            -0.2118 |            -0.1935 |          1 | -0.3303 | -0.1676 | -0.1292 | -0.2199 |
        | COVID-19 Crash          | 2020-02-19   | 2020-03-23 |            -0.3271 |            -0.3372 |          1 | -0.3067 | -0.2739 | -0.2491 | -0.4789 |
        | 2022 Rate Shock         | 2022-01-03   | 2022-10-12 |             0.0017 |            -0.245  |          1 | -0.2368 | -0.3212 | -0.0329 |  0.5976 |
        """
        assets = self._get_assets()
        resolved = self._resolve_weights(weights, assets)
        scenarios = scenarios or attribution_model.HISTORICAL_SCENARIOS

        tickers = list(assets)
        if self._benchmark_name and self._benchmark_name not in tickers:
            tickers.append(self._benchmark_name)

        rows = []
        for name, (start, end) in scenarios.items():
            # A few days before the start, so the last price on the start date is found.
            buffer_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime(
                "%Y-%m-%d"
            )
            prices = self._get_prices(tickers, buffer_start, end)
            returns = attribution_model.get_scenario_returns(
                prices.reindex(columns=tickers), {name: (start, end)}
            )
            stress = attribution_model.get_stress_test(returns[assets], resolved)
            if stress.loc[name, "Coverage"] == 0:
                logger.warning(
                    "No price data is available for the %s (%s to %s): none of the "
                    "assets was listed yet, so its Portfolio Return is empty.",
                    name,
                    start,
                    end,
                )
            elif pd.isna(stress.loc[name, "Portfolio Return"]):
                logger.warning(
                    "Only part of the portfolio has price data for the %s (%s to %s), with "
                    "a net weight of zero or less (short positions without the long ones), "
                    "so its Portfolio Return is empty.",
                    name,
                    start,
                    end,
                )
            row = pd.Series(
                {
                    "Start Date": start,
                    "End Date": end,
                    "Portfolio Return": stress.loc[name, "Portfolio Return"],
                    "Benchmark Return": (
                        returns.loc[name, self._benchmark_name]
                        if self._benchmark_name
                        else np.nan
                    ),
                    "Coverage": stress.loc[name, "Coverage"],
                },
                name=name,
            )
            rows.append(pd.concat([row, returns.loc[name, assets]]))

        table = pd.DataFrame(rows)
        table.index.name = "Scenario"
        numeric = table.columns.drop(["Start Date", "End Date"])
        table[numeric] = table[numeric].astype(float)
        return self._round(table, rounding)

    def _get_exposures(
        self, assets: list[str], by: str, alternative: str = ""
    ) -> dict[str, pd.Series]:
        """
        The sector, country or holding exposure of every asset; the alternative is named
        when the company profiles the sectors and countries of stocks come from are missing.
        """
        if self._fund_function is None:
            raise InvalidArgumentError(
                "The fund holdings come from the Toolkit. Please use this method from a Toolkit "
                "(toolkit.portfolio), or pass fund_function when creating the "
                "PortfolioConstruction class."
            )

        funds = self._fund_function(by, assets)
        exposures: dict[str, pd.Series] = {}
        stocks = []

        for asset in assets:
            if funds is not None and asset in funds.columns:
                exposure = pd.to_numeric(funds[asset], errors="coerce").dropna()
                if not exposure.empty:
                    exposures[asset] = exposure
                    continue
            stocks.append(asset)

        if stocks:
            if by == "holdings":
                for asset in stocks:
                    exposures[asset] = pd.Series({asset: 1.0})
            else:
                classification = self._get_classification(
                    stocks,
                    "Sector" if by == "sector" else "Country",
                    alternative,
                )
                for asset, value in classification.dropna().items():
                    exposures[asset] = pd.Series({value: 1.0})

        return exposures

    @handle_errors
    def get_look_through_exposure(
        self,
        weights: dict[str, float] | None = None,
        by: str = "sector",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a portfolio's exposure after looking through its funds.

        An ETF's weight is spread over its own sectors, countries or holdings, a stock counts
        fully towards its own sector, country or itself, and everything is summed. It shows, for
        example, that a portfolio of a technology ETF and Apple holds far more Apple and
        technology than either position suggests. The benchmark's exposure and the difference
        (the active exposure) are shown next to it when the benchmark is a fund.

        Also known as: look-through analysis, underlying exposure, X-ray.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            by (str, optional): What to look through to. Options are: "sector" (the sectors), "country" (the
                countries), "holdings" (the holdings). Defaults to "sector".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Exposure per sector, country or holding, largest first, with
            the Benchmark Exposure and Active Exposure where the benchmark is a fund.

        Notes:
        - Fund holdings lists often show only the largest positions; the rest is reported as
          "Other". Assets without a known classification are "Unclassified".

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["QQQ", "AAPL", "XLE", "JNJ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_look_through_exposure()
        ```

        Which returns:

        | Sector                 |   Exposure |   Benchmark Exposure |   Active Exposure |
        |:-----------------------|-----------:|---------------------:|------------------:|
        | Technology             |     0.3965 |               0.3757 |            0.0208 |
        | Healthcare             |     0.261  |               0.0994 |            0.1616 |
        | Energy                 |     0.2512 |               0.0373 |            0.2139 |
        | Communication Services |     0.029  |               0.0923 |           -0.0633 |
        | Consumer Cyclical      |     0.0263 |               0.0896 |           -0.0633 |
        | Consumer Defensive     |     0.0156 |               0.0495 |           -0.0339 |
        | Industrials            |     0.0144 |               0.0803 |           -0.0659 |
        | Utilities              |     0.0027 |               0.0204 |           -0.0177 |
        | Basic Materials        |     0.0025 |               0.017  |           -0.0145 |
        | Financial Services     |     0.0005 |               0.1208 |           -0.1203 |
        | Cash & Others          |     0.0004 |               0      |            0.0004 |
        | Real Estate            |     0      |               0.0175 |           -0.0175 |
        """
        check_option("by", by, ("sector", "country", "holdings"))

        assets = self._get_assets()
        resolved = self._resolve_weights(weights, assets)
        exposure = attribution_model.get_look_through_exposure(
            resolved,
            self._get_exposures(
                assets, by, 'look through to the holdings with by="holdings"'
            ),
        ).to_frame()

        if self._benchmark_name and by != "holdings" and self._fund_function:
            benchmark = self._fund_function(by, [self._benchmark_name])
            if benchmark is not None and self._benchmark_name in benchmark.columns:
                weights_benchmark = pd.to_numeric(
                    benchmark[self._benchmark_name], errors="coerce"
                ).dropna()
                if not weights_benchmark.empty:
                    exposure["Benchmark Exposure"] = weights_benchmark.reindex(
                        exposure.index
                    ).fillna(0)
                    for missing in weights_benchmark.index.difference(exposure.index):
                        exposure.loc[missing] = [0.0, weights_benchmark[missing]]
                    exposure["Active Exposure"] = (
                        exposure["Exposure"] - exposure["Benchmark Exposure"]
                    )
                    exposure = exposure.sort_values("Exposure", ascending=False)

        return self._round(exposure, rounding)

    @handle_errors
    def get_brinson_attribution(
        self,
        weights: dict[str, float] | None = None,
        period: str | None = None,
        linking: str = "carino",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Brinson-Fachler attribution of the portfolio's return over the benchmark,
        by sector, over the Toolkit's date range.

        The excess return is split into three effects per sector:

        - Allocation: the reward for overweighting sectors that beat the benchmark (and
          underweighting those that lagged).
        - Selection: the reward for holding better stocks than the benchmark within a
          sector.
        - Interaction: the combined effect of both decisions.

        The portfolio's sector returns come from its holdings (buy and hold from the start
        date). A stock counts towards its own sector and a fund is looked through to its
        sector weightings, as in get_look_through_exposure, so that QQQ counts as mostly
        Technology rather than as the sector of its issuer. The benchmark's sector weights
        come from the benchmark fund and its sector returns from the Select Sector SPDR
        funds (XLK, XLF, XLV, ...), which track the sectors of the S&P 500.

        With a period, the range is split into sub-periods (weeks, months, quarters or
        years). Every sub-period gets its own attribution with the weights at its start, and
        the effects are linked into effects over the whole range that add up to the
        compounded excess return. The holdings are bought at the start and held without
        rebalancing, so the portfolio's weights drift with the returns of its assets; the
        benchmark's sector weights drift with the returns of the sector funds in the same
        way. The compounded returns of the sub-periods therefore equal the returns of the
        attribution without a period, and only the effects differ: they reflect the
        weights held in each sub-period rather than those at the start.

        The formula is as follows:

        - Allocation_t = (w_p,t - w_b,t) x (r_b,t - B_t)
        - Selection_t = w_b,t x (r_p,t - r_b,t)
        - Interaction_t = (w_p,t - w_b,t) x (r_p,t - r_b,t)
        - Cariño: Linked Effect = Σ_t (k_t / K) x Effect_t, with
          k_t = [ln(1 + R_t) - ln(1 + B_t)] / (R_t - B_t) and
          K = [ln(1 + R) - ln(1 + B)] / (R - B), or 1 / (1 + R_t) and 1 / (1 + R) when the
          returns are equal
        - Menchero: Linked Effect = Σ_t (M + α_t) x Effect_t, with
          M = [(R - B) / T] / [(1 + R)^(1 / T) - (1 + B)^(1 / T)] and
          α_t = [R - B - M x Σ_s (R_s - B_s)] / Σ_s (R_s - B_s)² x (R_t - B_t)

        Here w_p,t and w_b,t are the sector weights at the start of sub-period t, r_p,t and
        r_b,t the sector returns, R_t and B_t the portfolio and benchmark returns in the
        sub-period, R and B those returns compounded over all T sub-periods. Without a
        period the first three lines apply once to the whole range.

        For more information, see the following papers:

        - Brinson, G.P. and Fachler, N. (1985). "Measuring Non-US Equity Portfolio
          Performance." The Journal of Portfolio Management, 11(3), 73-76.
        - Brinson, G.P., Hood, L.R. and Beebower, G.L. (1986). "Determinants of Portfolio
          Performance." Financial Analysts Journal, 42(4), 39-44.
        - Cariño, D.R. (1999). "Combining Attribution Effects Over Time." The Journal of
          Performance Measurement, 3(4), 5-14.
        - Menchero, J. (2000). "An Optimized Approach to Linking Attribution Effects Over
          Time." The Journal of Performance Measurement, 5(1), 36-42.
        - Bacon, C. (2004). Practical Portfolio Performance Measurement and Attribution.
          Wiley, pp. 191-196.

        Also known as: performance attribution, BHB attribution, sector attribution.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset at the start of
                the date range. Defaults to None, which uses the latest weights of the
                Portfolio the Toolkit came from, or equal weights.
            period (str | None, optional): The sub-periods whose attributions are linked, each ending on the last
                trading day of the calendar period. Options are: "weekly", "monthly", "quarterly", "yearly". Defaults
                to None, which attributes the whole range as one period.
            linking (str, optional): How the effects of the sub-periods are linked when a period is given. Options
                are: "carino" (Cariño, 1999), "menchero" (Menchero, 2000). Defaults to "carino".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per sector and in total, the Portfolio Weight, Benchmark Weight
            (at the start of the range), Portfolio Return, Benchmark Return (over the whole
            range) and the Allocation, Selection, Interaction and Total Effect, linked over
            the sub-periods when a period is given. The Total row's effects add up to the
            excess return.

        Notes:
        - The benchmark fund's sector weights are its current weights, applied to the whole
          date range, and the sector funds stand in for the benchmark's sectors; for a
          benchmark other than the S&P 500 this is an approximation.
        - A fund's weight is spread over its current sector weightings and every part earns
          the fund's own return, as the return of a fund's sectors separately is not known.
          Assets without a sector are "Unclassified" and compared with the benchmark as a
          whole.
        - A sector whose sector fund has no price within a few trading days of the start of
          the range (XLC starts on 2018-06-19, XLRE on 2015-10-08) has no benchmark return,
          since the fund's return from its first price covers only part of the range. The
          sector keeps its benchmark weight rather than handing it to the other sectors, a
          warning names the fund and its first date, and the benchmark's total return, every
          allocation effect, the sector's selection and interaction effects and the Total
          row's effects are empty. With a period this holds for the sub-periods that start
          before the fund's first price, and as the linking of the effects depends on the
          benchmark return compounded over the whole range, all linked effects are empty. A
          range that starts at the fund's first price can be attributed in full.
        - Cariño's linking scales each sub-period by its logarithmic excess return relative
          to the whole range; Menchero's keeps the scaling as close as possible to one
          factor for every sub-period. Both make the linked effects add up exactly to the
          compounded portfolio return minus the compounded benchmark return.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM", "JPM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_brinson_attribution()
        ```

        Which returns:

        | Sector                 |   Portfolio Weight |   Benchmark Weight |   Portfolio Return |   Benchmark Return |   Allocation Effect |   Selection Effect |   Interaction Effect |   Total Effect |
        |:-----------------------|-------------------:|-------------------:|-------------------:|-------------------:|--------------------:|-------------------:|---------------------:|---------------:|
        | Basic Materials        |                0   |             0.017  |           nan      |             0.1115 |              0.0012 |             0      |              -0      |         0.0012 |
        | Communication Services |                0   |             0.0923 |           nan      |             0.2249 |             -0.0042 |             0      |              -0      |        -0.0042 |
        | Consumer Cyclical      |                0   |             0.0896 |           nan      |             0.0881 |              0.0082 |             0      |              -0      |         0.0082 |
        | Consumer Defensive     |                0   |             0.0495 |           nan      |             0.0183 |              0.008  |             0      |              -0      |         0.008  |
        | Energy                 |                0.2 |             0.0373 |             0.1625 |             0.0661 |             -0.0185 |             0.0036 |               0.0157 |         0.0008 |
        | Financial Services     |                0.2 |             0.1208 |             0.371  |             0.152  |             -0.0022 |             0.0265 |               0.0173 |         0.0416 |
        | Healthcare             |                0.2 |             0.0994 |             0.4809 |             0.1447 |             -0.0035 |             0.0334 |               0.0338 |         0.0637 |
        | Industrials            |                0   |             0.0803 |           nan      |             0.1976 |             -0.0014 |             0      |              -0      |        -0.0014 |
        | Real Estate            |                0   |             0.0175 |           nan      |             0.036  |              0.0025 |             0      |              -0      |         0.0025 |
        | Technology             |                0.4 |             0.3758 |             0.1419 |             0.2489 |              0.0017 |            -0.0402 |              -0.0026 |        -0.0411 |
        | Utilities              |                0   |             0.0204 |           nan      |             0.1511 |              0.0006 |             0      |              -0      |         0.0006 |
        | Total                  |                1   |             1      |             0.2596 |             0.1798 |             -0.0077 |             0.0233 |               0.0642 |         0.0798 |
        """
        if period is not None and period not in SUB_PERIODS:
            raise InvalidArgumentError(
                f"period must be {', '.join(SUB_PERIODS)}, not {period!r}. Please pass one of "
                "these as period, or leave it out to attribute the whole range as one period."
            )
        check_option("linking", linking, attribution_model.LINKING_METHODS)
        if not self._benchmark_name or self._fund_function is None:
            raise ValueError("The attribution needs a benchmark fund, e.g. SPY.")

        resolved = self._resolve_weights(weights, self._get_assets())
        # Only the assets held take part: one without weight adds no sector slice, and its
        # missing start price is no reason to leave out others.
        resolved = resolved[resolved != 0]
        assets = resolved.index.tolist()

        # Historical data supplied to the Toolkit can reach beyond its date range.
        prices = (
            self._historical_data["daily"]["Adj Close"][assets]
            .loc[self._start_date : self._end_date]
            .dropna(how="all")
        )

        # Each asset starts at its first price within a few trading days, so a holding
        # whose exchange is closed on the first day (a local holiday) still counts. An
        # asset without a price by then would be measured over a shorter span than the
        # benchmark sectors, or not at all (a NaN return that the sector sums would count
        # as 0%), so it is left out and the other weights are scaled back up to one.
        start_prices = prices.bfill(limit=START_PRICE_TOLERANCE_DAYS).iloc[0]
        unpriced = [asset for asset in assets if pd.isna(start_prices[asset])]
        if unpriced:
            if len(unpriced) == len(assets):
                raise ValueError(
                    "None of the assets has a price at the start of the range."
                )
            logger.warning(
                "No price data is available for %s at the start of the range, so the "
                "attribution leaves them out and scales the other weights up.",
                ", ".join(unpriced),
            )
            assets = [asset for asset in assets if asset not in unpriced]
            if resolved[assets].sum() <= 0:
                raise ValueError("The assets with a price at the start have no weight.")
            resolved = resolved[assets] / resolved[assets].sum()
            prices = prices[assets]
            start_prices = start_prices[assets]

        asset_returns = prices.ffill().iloc[-1] / start_prices - 1

        # Every asset as one or more sector slices: a stock in its own sector, a fund spread
        # over its sector weightings, each slice earning the asset's return.
        exposures = self._get_exposures(assets, "sector")
        slices = []
        for asset in assets:
            exposure = exposures.get(asset, pd.Series(dtype=float))
            exposure = exposure[exposure > 0]
            if exposure.empty:
                exposure = pd.Series({"Unclassified": 1.0})
            for sector, share in (exposure / exposure.sum()).items():
                slices.append(
                    (sector, asset, resolved[asset] * share, asset_returns[asset])
                )
        slices = pd.DataFrame(slices, columns=["Sector", "Asset", "Weight", "Return"])

        portfolio_weights = slices.groupby("Sector")["Weight"].sum()
        portfolio_returns = (slices["Weight"] * slices["Return"]).groupby(
            slices["Sector"]
        ).sum() / portfolio_weights.replace(0, np.nan)

        benchmark = self._fund_function("sector", [self._benchmark_name])
        if benchmark is None or self._benchmark_name not in benchmark.columns:
            raise ValueError(
                f"No sector weights are available for the benchmark {self._benchmark_name}."
            )
        benchmark_weights = pd.to_numeric(
            benchmark[self._benchmark_name], errors="coerce"
        ).dropna()

        dates = pd.to_datetime(prices.index.astype(str))
        sector_tickers = {
            sector: ticker
            for sector, ticker in attribution_model.SECTOR_ETFS.items()
            if sector in benchmark_weights.index or sector in portfolio_weights.index
        }
        sector_prices = self._get_prices(
            list(sector_tickers.values()),
            dates.min().strftime("%Y-%m-%d"),
            dates.max().strftime("%Y-%m-%d"),
        )
        funds = sector_prices.reindex(columns=list(sector_tickers.values())).set_axis(
            pd.to_datetime(sector_prices.index.astype(str))
        )
        funds.columns = list(sector_tickers)
        funds = funds.loc[dates.min() : dates.max()]
        # A sector fund stands in for its sector only from its first price, which may come
        # a few trading days after the first day, as for the assets. Before that the
        # sector's benchmark return is unknown rather than the return from the fund's
        # first price, and the sector keeps its benchmark weight instead of handing it to
        # the other sectors.
        first_dates = funds.notna().idxmax().where(funds.notna().any())
        covered = first_dates.notna() & (
            first_dates <= dates[min(START_PRICE_TOLERANCE_DAYS, len(dates) - 1)]
        )
        fund_start_prices = funds.bfill().iloc[0].where(covered)
        benchmark_returns = funds.ffill().iloc[-1] / fund_start_prices - 1
        late_funds = [
            f"{sector} ({sector_tickers[sector]}, "
            + (
                f"first price on {first_dates[sector]:%Y-%m-%d})"
                if pd.notna(first_dates[sector])
                else "no prices in the range)"
            )
            for sector in funds.columns
            if not covered[sector] and sector in benchmark_weights.index
        ]
        if late_funds:
            logger.warning(
                "No price at the start of the range (%s) for the sector fund of %s. The "
                "benchmark return of the sector is unknown %s, so it keeps its benchmark "
                "weight and the benchmark's total return and the effects that depend on it "
                "are empty. A range from the fund's first price onwards can be attributed.",
                dates.min().strftime("%Y-%m-%d"),
                ", ".join(late_funds),
                (
                    "in the sub-periods before the fund's first price"
                    if period
                    else "over the range"
                ),
            )
        # Portfolio sectors outside the sector funds (e.g. "Unclassified") are compared
        # with the benchmark as a whole, and so are the benchmark's own sectors without a
        # sector fund, such as cash.
        benchmark_weights = benchmark_weights[
            benchmark_weights.index.isin(benchmark_returns.index)
        ]

        if period is None:
            table = attribution_model.get_brinson_fachler_attribution(
                portfolio_weights,
                portfolio_returns,
                benchmark_weights,
                benchmark_returns,
            )
        else:
            # The sub-periods run from the first day (or the last trading day of the
            # previous calendar period) to the last trading day of each calendar period.
            positions = pd.Series(np.arange(len(dates)), index=dates)
            ends = positions.groupby(dates.to_period(SUB_PERIODS[period])).max()
            boundaries = dates[[0, *ends[ends > 0]]]

            # Buy and hold: each sector slice grows with its asset from the start, before
            # its first price at the start price, so the weights at every boundary are the
            # drifted ones and the sub-period returns compound to the whole-range returns.
            asset_growth = (
                prices.ffill().fillna(start_prices).set_axis(dates).loc[boundaries]
                / start_prices
            )
            slice_values = (
                asset_growth[slices["Asset"]].to_numpy() * slices["Weight"].to_numpy()
            )
            sector_values = (
                pd.DataFrame(slice_values, index=boundaries, columns=slices["Sector"])
                .T.groupby(level=0)
                .sum()
                .T
            )

            # A sector fund has a return in the sub-periods it has prices for from start
            # to end, the first one starting at its first price within a few trading days
            # as without a period. Its weight drifts with the fund and is held at its start
            # value before the fund's first price, as the sector's return there is unknown.
            fund_prices = (
                funds.reindex(funds.index.union(boundaries)).ffill().loc[boundaries]
            )
            fund_prices.iloc[0] = fund_start_prices
            weight_prices = fund_prices.bfill().fillna(1.0)
            benchmark_values = (
                weight_prices / weight_prices.iloc[0] * benchmark_weights
            ).dropna(axis=1, how="all")

            # Every sub-period is named after its last day and starts with the values at
            # the previous boundary; a sector without value has no return.
            starts, sub_periods = boundaries[:-1], boundaries[1:]
            portfolio_values = sector_values.loc[starts].set_axis(sub_periods)
            benchmark_values = benchmark_values.loc[starts].set_axis(sub_periods)
            table = attribution_model.get_linked_brinson_fachler_attribution(
                portfolio_values,
                sector_values.loc[sub_periods].div(portfolio_values.replace(0, np.nan))
                - 1,
                benchmark_values,
                fund_prices.loc[sub_periods]
                / fund_prices.loc[starts].set_axis(sub_periods)
                - 1,
                linking,
            )
        table.index.name = "Sector"
        return self._round(table, rounding)

    @handle_errors
    def get_factor_attribution(
        self,
        weights: dict[str, float] | None = None,
        selection: str = "ols",
        factors: pd.DataFrame | None = None,
        region: str = "United States",
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Attribute the return of a portfolio in every period to factors, by default the five
        factors of Fama and French (market, size, value, profitability and investment).

        It shows the part of the return its exposure to each factor explains, and the specific
        return that is left. The loadings of the assets on the factors come from a regression
        over the whole period (see get_factor_loadings), and the portfolio's exposure to a
        factor is the weighted sum of its assets' loadings.

        - Exposure_f,t = Σ_i w_i,t β_i,f
        - Contribution_f,t = Exposure_f,t × Factor Return_f,t
        - Common Return = Σ_f Contribution_f,t
        - Specific Return = Total Return - Common Return
        - Tilt Return = Σ_f Average Exposure_f × Factor Return_f,t
        - Timing Return = Common Return - Tilt Return

        The Specific Return holds the alphas of the assets (and the risk free rate, as the
        Fama and French factors are excess returns) and their residual returns. The Tilt
        Return is what the average exposure earned and the Timing Return what moving the
        exposures around it added, which is zero for weights that stay the same.

        The weights are those of the Portfolio the Toolkit came from at the end of the
        period before, or, when weights are given (or there is no Portfolio), those weights
        restored every period.

        For more information, see the following sources:

        - Grinold, R. C. and Kahn, R. N. (2000). "Active Portfolio Management." 2nd
          edition, McGraw-Hill, chapter 17.
        - Fama, E. F. and French, K. R. (2015). "A Five-Factor Asset Pricing Model."
          Journal of Financial Economics, 116(1), 1-22.

        Also known as: performance attribution to risk factors, factor return attribution,
        returns-based attribution, common and specific returns.

        Referenced against:
        - empyrical, perf_attrib: https://github.com/stefan-jansen/empyrical-reloaded/blob/0.7.3/src/empyrical/perf_attrib.py#L5

        Shares similar logic:
        - empyrical, compute_exposures: https://github.com/stefan-jansen/empyrical-reloaded/blob/0.7.3/src/empyrical/perf_attrib.py#L139
        - pyfolio, perf_attrib: https://github.com/stefan-jansen/pyfolio-reloaded/blob/0.9.9/src/pyfolio/perf_attrib.py#L29

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the weights of the Portfolio the Toolkit came from over
                time, or equal weights.
            selection (str, optional): How the loadings are estimated (see get_factor_loadings). Options are: "ols"
                (every factor), "forward" (forward selection), "backward" (backward selection), "pcr" (principal
                components regression). Defaults to "ols".
            factors (pd.DataFrame | None, optional): The factor returns, one column per
                factor, at the frequency of the period. Defaults to None, the Fama and
                French five factors of the region.
            region (str, optional): The region of the Fama and French factors. Options are: "United States",
                "Developed", "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan", "North America".
                Defaults to "United States".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per period the contribution of every factor and the Common,
            Specific, Total, Tilt and Timing Return, on the periods in which the assets and
            the factors all have a return.

        Notes:
        - The loadings are estimated on the whole period, so the attribution of an early
          period uses information from later ones; it explains the past, it does not
          forecast.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_factor_attribution(
            weights={"AAPL": 0.4, "MSFT": 0.3, "JNJ": 0.2, "XOM": 0.1}, period="monthly"
        ).tail(4).T
        ```

        Which returns:

        |                 |   2025-09 |   2025-10 |   2025-11 |   2025-12 |
        |:----------------|----------:|----------:|----------:|----------:|
        | Mkt-RF          |    0.0297 |    0.017  |   -0.0012 |   -0.003  |
        | SMB             |    0.0043 |    0.0028 |   -0.0033 |    0.0004 |
        | HML             |    0.0019 |    0.0058 |   -0.0068 |   -0.0042 |
        | RMW             |   -0.005  |   -0.013  |    0.0032 |    0.0006 |
        | CMA             |   -0.003  |   -0.0055 |    0.0008 |    0.0005 |
        | Common Return   |    0.0278 |    0.0071 |   -0.0072 |   -0.0058 |
        | Specific Return |    0.0256 |    0.0227 |    0.0284 |   -0.0055 |
        | Total Return    |    0.0534 |    0.0298 |    0.0213 |   -0.0113 |
        | Tilt Return     |    0.0278 |    0.0071 |   -0.0072 |   -0.0058 |
        | Timing Return   |    0      |    0      |    0      |    0      |
        """
        if weights is None and self._portfolio_weights:
            assets = self._get_loaded_assets(period, include_benchmark)
        else:
            assets = self._get_held_weights(
                weights, period, include_benchmark
            ).index.tolist()
        returns, factor_returns = self._align_factors(
            self._get_returns(period, include_benchmark, assets),
            self._get_factor_returns(period, factors, region),
        )
        loadings = estimation_model.get_factor_loadings(
            returns, factor_returns, selection
        ).drop(columns="Intercept")
        table = attribution_model.get_factor_attribution(
            returns,
            self._get_weights_over_time(weights, period, assets, returns.index),
            loadings,
            factor_returns,
        )
        table.index.name = "Date"
        return self._round(table, rounding)
