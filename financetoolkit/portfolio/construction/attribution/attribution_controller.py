"""Portfolio Attribution Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.attribution import attribution_model
from financetoolkit.portfolio.construction.base.base_controller import (
    PortfolioConstructionBase,
)
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501

logger = get_logger()

# The trading days an asset's first price may come after the first day of an attribution.
START_PRICE_TOLERANCE_DAYS = 5


@filter_shown_columns
@track_sources
class AttributionConstruction(PortfolioConstructionBase):
    """
    Exposures and attribution: the stress test, the look-through exposure and the Brinson attribution.
    """

    def _get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """The daily prices of other tickers over a date range."""
        if self._price_function is None:
            raise ValueError(
                "Prices of other tickers are not available for this Toolkit."
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
        Calculate how a portfolio would have fared in historical market crises: the return
        of every asset and of the buy-and-hold portfolio from the start to the end of each
        scenario, next to the benchmark. The default scenarios run from the peak to the
        trough of the S&P 500: the Dot-com Crash, the Global Financial Crisis, the Euro
        Area Debt Crisis, the 2015-2016 and 2018 Q4 sell-offs, the COVID-19 Crash and the
        2022 Rate Shock.

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
            the Benchmark Return, the Coverage (the share of the portfolio with prices in
            the scenario) and the return of every asset.

        Notes:
        - An asset that was not listed yet at the start of a scenario has no return there;
          the other weights are scaled up and the Coverage shows the share that had data.

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

    def _get_exposures(self, assets: list[str], by: str) -> dict[str, pd.Series]:
        """The sector, country or holding exposure of every asset."""
        if self._fund_function is None:
            raise ValueError("Fund data is not available for this Toolkit.")

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
                    stocks, "Sector" if by == "sector" else "Country"
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
        Calculate a portfolio's exposure after looking through its funds. An ETF's weight is
        spread over its own sectors, countries or holdings, a stock counts fully towards its
        own sector, country or itself, and everything is summed. It shows, for example, that
        a portfolio of a technology ETF and Apple holds far more Apple and technology than
        either position suggests. The benchmark's exposure and the difference (the active
        exposure) are shown next to it when the benchmark is a fund.

        Also known as: look-through analysis, underlying exposure, X-ray.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            by (str, optional): What to look through to: "sector", "country" or "holdings".
                Defaults to "sector".
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
        if by not in ("sector", "country", "holdings"):
            raise ValueError("by must be 'sector', 'country' or 'holdings'.")

        assets = self._get_assets()
        resolved = self._resolve_weights(weights, assets)
        exposure = attribution_model.get_look_through_exposure(
            resolved, self._get_exposures(assets, by)
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
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Brinson-Fachler attribution of the portfolio's return over the
        benchmark, by sector, over the Toolkit's date range. The excess return is split
        into three effects per sector:

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

        For more information, see the following papers:

        - Brinson, G.P. and Fachler, N. (1985). "Measuring Non-US Equity Portfolio
          Performance." The Journal of Portfolio Management, 11(3), 73-76.
        - Brinson, G.P., Hood, L.R. and Beebower, G.L. (1986). "Determinants of Portfolio
          Performance." Financial Analysts Journal, 42(4), 39-44.

        Also known as: performance attribution, BHB attribution, sector attribution.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset at the start of
                the date range. Defaults to None, which uses the latest weights of the
                Portfolio the Toolkit came from, or equal weights.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per sector and in total, the Portfolio Weight, Benchmark Weight,
            Portfolio Return, Benchmark Return and the Allocation, Selection, Interaction
            and Total Effect. The Total row's effects add up to the excess return.

        Notes:
        - The benchmark fund's sector weights are its current weights, applied to the whole
          date range, and the sector funds stand in for the benchmark's sectors; for a
          benchmark other than the S&P 500 this is an approximation.
        - A fund's weight is spread over its current sector weightings and every part earns
          the fund's own return, as the return of a fund's sectors separately is not known.
          Assets without a sector are "Unclassified" and compared with the benchmark as a
          whole.

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
        if not self._benchmark_name or self._fund_function is None:
            raise ValueError("The attribution needs a benchmark fund, e.g. SPY.")

        assets = self._get_assets()
        resolved = self._resolve_weights(weights, assets)

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
                slices.append((sector, resolved[asset] * share, asset_returns[asset]))
        slices = pd.DataFrame(slices, columns=["Sector", "Weight", "Return"])

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
        benchmark_returns = pd.Series(
            {
                sector: sector_prices[ticker].dropna().iloc[-1]
                / sector_prices[ticker].dropna().iloc[0]
                - 1
                for sector, ticker in sector_tickers.items()
                if ticker in sector_prices.columns
                and not sector_prices[ticker].dropna().empty
            }
        )
        # Portfolio sectors outside the sector funds (e.g. "Unclassified") are compared
        # with the benchmark as a whole.
        benchmark_weights = benchmark_weights[
            benchmark_weights.index.isin(benchmark_returns.index)
        ]

        table = attribution_model.get_brinson_fachler_attribution(
            portfolio_weights, portfolio_returns, benchmark_weights, benchmark_returns
        )
        table.index.name = "Sector"
        return self._round(table, rounding)
