"""Portfolio Analysis Module"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.analysis import backtest_model
from financetoolkit.portfolio.construction.base.base_controller import (
    BENCHMARK_TICKER,
    PORTFOLIO_TICKER,
    PortfolioConstructionBase,
)
from financetoolkit.portfolio.construction.optimization import optimization_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR

# ruff: noqa: E501

# The optimizers a backtest can compare, with the names they are shown under.
BACKTEST_METHODS = {
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "mean_variance": "Mean Variance",
    "risk_parity": "Risk Parity",
    "hierarchical_risk_parity": "Hierarchical Risk Parity",
    "inverse_volatility": "Inverse Volatility",
    "maximum_diversification": "Maximum Diversification",
    "hierarchical_equal_risk_contribution": "Hierarchical Equal Risk Contribution",
    "nested_clustered_optimization": "Nested Clustered Optimization",
    "minimum_cvar": "Minimum CVaR",
    "minimum_drawdown": "Minimum Drawdown",
    "maximum_omega": "Maximum Omega",
    "maximum_cvar_ratio": "Maximum CVaR Ratio",
    "equal_weight": "Equal Weight",
}

DEFAULT_BACKTEST_METHODS = (
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
    "hierarchical_risk_parity",
    "maximum_diversification",
    "inverse_volatility",
)


@filter_shown_columns
@track_sources
class AnalysisConstruction(PortfolioConstructionBase):
    """
    Analysis of a set of weights: risk contributions, diversification and risk metrics, the discrete allocation and the walk-forward optimization backtest.
    """

    @handle_errors
    def get_risk_contributions(
        self,
        weights: dict[str, float] | None = None,
        risk_measure: str = "volatility",
        confidence_level: float = 0.95,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how much each asset contributes to the risk of a portfolio, measured as
        its volatility, expected shortfall (CVaR), semideviation, mean absolute deviation or
        Conditional Drawdown at Risk (CDaR). The marginal contribution is the
        change in the portfolio's risk from a small increase in an asset's weight; the
        risk contribution is the weight times that, and the contributions add up to the
        portfolio's risk (Euler allocation). A large position in a calm asset can
        contribute less risk than a small one in a volatile asset.

        - risk_measure="volatility": the marginal contribution is (Σw)_i / σ_p, with Σ
          the annualized covariance matrix.
        - risk_measure="cvar": the expected shortfall is the average loss in the periods
          in which the portfolio's return is at or below its Value at Risk, and the
          marginal contribution of an asset is its average loss in those same periods,
          -E[r_i | r_p ≤ VaR], so the contributions show which assets lose in the
          portfolio's worst periods. These are per period (e.g. daily), like the
          Finance Toolkit's historical CVaR, and not annualized.
        - risk_measure="semideviation": the downside deviation of the portfolio's
          return below zero, SD⁻ = √(Σ min(r_p,t, 0)² / T), with the marginal
          contribution Σ min(r_p,t, 0) r_i,t / (T SD⁻): an asset contributes by losing
          in the periods in which the portfolio loses. Per period, not annualized.
        - risk_measure="mad": the mean absolute deviation of the portfolio's return,
          Σ |r_p,t - μ_p| / T, with the marginal contribution
          Σ sign(r_p,t - μ_p)(r_i,t - μ_i) / T. Per period, not annualized.
        - risk_measure="cdar": the Conditional Drawdown at Risk, the average of the
          worst 1 - α share of the drawdowns of the uncompounded cumulative return (the
          drawdowns the minimum drawdown optimization minimizes). Each drawdown is the
          sum of the assets' drawdowns from the portfolio's peak, so the marginal
          contribution of an asset is its average drawdown over the portfolio's worst
          drawdown periods (Goldberg and Mahmoud, 2017).

        For more information, see the following papers:

        - Litterman, R. (1996). "Hot Spots and Hedges." The Journal of Portfolio Management,
          23(5), 52-75.
        - Tasche, D. (2000). "Conditional Expectations as Quantile Derivatives." Working
          paper, arXiv math/0104190.
        - Scaillet, O. (2004). "Nonparametric Estimation and Sensitivity Analysis of
          Expected Shortfall." Mathematical Finance, 14(1), 115-129.
        - Sortino, F. A. and Price, L. N. (1994). "Performance Measurement in a Downside
          Risk Framework." The Journal of Investing, 3(3), 59-64.
        - Konno, H. and Yamazaki, H. (1991). "Mean-Absolute Deviation Portfolio
          Optimization Model and Its Applications to Tokyo Stock Market." Management
          Science, 37(5), 519-531.
        - Goldberg, L. R. and Mahmoud, O. (2017). "Drawdown: From Practice to Theory and
          Back Again." Mathematics and Financial Economics, 11(3), 275-297.

        Also known as: risk decomposition, marginal contribution to risk, risk budget,
        component expected shortfall, component CVaR, downside risk contribution,
        drawdown contribution.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            risk_measure (str, optional): "volatility", "cvar" (the expected shortfall),
                "semideviation" (below a return of zero), "mad" (the mean absolute
                deviation) or "cdar" (the Conditional Drawdown at Risk). Defaults to
                "volatility".
            confidence_level (float, optional): The confidence level of the expected
                shortfall and the CDaR, e.g. 0.95 for the average of the worst 5% of
                periods or drawdowns. Defaults to 0.95.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (towards the scaled identity, Ledoit and Wolf,
                2004), "ledoit_wolf_constant_correlation" (towards the average
                correlation, Ledoit and Wolf, 2004), "ledoit_wolf_single_factor"
                (towards the covariance that the benchmark as a single factor implies,
                Ledoit and Wolf, 2003), "oas" (Chen et al., 2010), "semicovariance" (the
                co-movement below a zero return, Estrada, 2007), "exponential" (recent
                periods weigh most, RiskMetrics) or "denoised" (the eigenvalues of the
                correlation matrix that random returns could produce averaged, Laloux et
                al., 2000). The shrinkage and denoised estimators stabilize the weights
                when there are many assets compared to the number of periods. Defaults
                to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and in total, the Weight, Marginal Contribution, Risk
            Contribution and Percentage Contribution: annualized volatility terms, or
            per period terms of the other measures, with losses and drawdowns positive.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_risk_contributions(
            weights={"AAPL": 0.4, "MSFT": 0.3, "JNJ": 0.2, "XOM": 0.1},
            risk_measure="cvar",
        )
        ```

        Which returns:

        | Asset   |   Weight |   Marginal Contribution |   Risk Contribution |   Percentage Contribution |
        |:--------|---------:|------------------------:|--------------------:|--------------------------:|
        | AAPL    |      0.4 |                  0.0352 |              0.0141 |                    0.5365 |
        | MSFT    |      0.3 |                  0.0306 |              0.0092 |                    0.3492 |
        | JNJ     |      0.2 |                  0.0082 |              0.0016 |                    0.0625 |
        | XOM     |      0.1 |                  0.0136 |              0.0014 |                    0.0518 |
        | Total   |      1   |                nan      |              0.0263 |                    1      |
        """
        measures = ("volatility", "cvar", *optimization_model.HISTORICAL_RISK_MEASURES)
        if risk_measure not in measures:
            raise ValueError(
                f"The risk measure must be one of {', '.join(measures)}, not "
                f"'{risk_measure}'."
            )
        returns, _, covariance_matrix, _ = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        resolved = self._resolve_weights(weights, covariance_matrix.index.tolist())
        if risk_measure == "volatility":
            table = optimization_model.get_risk_contributions(
                resolved, covariance_matrix
            )
        elif risk_measure == "cvar":
            table = optimization_model.get_expected_shortfall_contributions(
                resolved, returns, confidence_level
            )
        else:
            table = optimization_model.get_historical_risk_contributions(
                resolved, returns, risk_measure, confidence_level
            )
        table.index.name = "Asset"
        table.loc["Total"] = [
            table["Weight"].sum(),
            np.nan,
            table["Risk Contribution"].sum(),
            table["Percentage Contribution"].sum(),
        ]
        return self._round(table, rounding)

    @handle_errors
    def get_diversification_metrics(
        self,
        weights: dict[str, float] | None = None,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Calculate how diversified a portfolio is, by the concentration of its weights and
        by how its risk spreads over independent sources.

        - Herfindahl-Hirschman Index: Σ w_i², from 1/N for equal weights to 1 for a single
          asset.
        - Effective Number of Assets: 1 / Σ w_i², the number of equally weighted assets
          that would be as concentrated.
        - Diversification Ratio: the weighted average volatility of the assets over the
          portfolio's volatility, w'σ / √(w'Σw); one for a single asset, higher as the
          assets move less together (Choueifaty and Coignard, 2008).
        - Effective Number of Bets: the covariance matrix is split into uncorrelated
          principal components, p_k is the share of the portfolio's variance from
          component k, and the number of bets is exp(-Σ p_k ln p_k): 1 when all risk comes
          from one source and N when it spreads evenly over all of them (Meucci, 2009).
          Equal weights in assets that all follow the market are many assets but close
          to one bet.

        For more information, see the following papers:

        - Choueifaty, Y. and Coignard, Y. (2008). "Toward Maximum Diversification." The
          Journal of Portfolio Management, 35(1), 40-51.
        - Meucci, A. (2009). "Managing Diversification." Risk, 22(5), 74-79.

        Also known as: concentration, HHI, effective N, ENB, diversification ratio.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (towards the scaled identity, Ledoit and Wolf,
                2004), "ledoit_wolf_constant_correlation" (towards the average
                correlation, Ledoit and Wolf, 2004), "ledoit_wolf_single_factor"
                (towards the covariance that the benchmark as a single factor implies,
                Ledoit and Wolf, 2003), "oas" (Chen et al., 2010), "semicovariance" (the
                co-movement below a zero return, Estrada, 2007), "exponential" (recent
                periods weigh most, RiskMetrics) or "denoised" (the eigenvalues of the
                correlation matrix that random returns could produce averaged, Laloux et
                al., 2000). The shrinkage and denoised estimators stabilize the weights
                when there are many assets compared to the number of periods. Defaults
                to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.Series: The Herfindahl-Hirschman Index, Effective Number of Assets,
            Diversification Ratio and Effective Number of Bets of the portfolio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_diversification_metrics(
            weights={"AAPL": 0.4, "MSFT": 0.3, "JNJ": 0.2, "XOM": 0.1}
        )
        ```

        Which returns:

        |                            |   Portfolio |
        |:---------------------------|------------:|
        | Herfindahl-Hirschman Index |      0.3    |
        | Effective Number of Assets |      3.3333 |
        | Diversification Ratio      |      1.343  |
        | Effective Number of Bets   |      1.1298 |
        """
        _, _, covariance_matrix, _ = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        resolved = self._resolve_weights(weights, covariance_matrix.index.tolist())
        metrics = optimization_model.get_diversification_metrics(
            resolved, covariance_matrix
        )
        metrics.name = PORTFOLIO_TICKER
        return self._round(metrics.to_frame(), rounding)[PORTFOLIO_TICKER]

    @handle_errors
    def get_portfolio_risk_metrics(
        self,
        weights: dict[str, float] | None = None,
        confidence_level: float = 0.95,
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the return and risk of a portfolio with given weights over the
        Toolkit's date range, next to the benchmark. The portfolio's return in each
        period is the weighted return of its assets, r_p,t = Σ w_i r_i,t, which holds the
        weights constant by rebalancing to them every period. The measures use the
        formulas of the Performance and Risk modules:

        - Annualized Return: the geometric average, Π(1 + r_t)^(P/T) - 1, with P the
          periods per year and T the number of periods.
        - Annualized Volatility: the standard deviation (ddof=1) times √P.
        - Sharpe Ratio: the average return above the risk free rate over its standard
          deviation, annualized with √P (Sharpe, 1966).
        - Sortino Ratio: the average return above the risk free rate over its downside
          deviation below zero, annualized with √P (Sortino and Price, 1994).
        - Maximum Drawdown: the largest fall of the compounded value from its peak.
        - Value at Risk: the 1 - α quantile of the returns per period (historical).
        - Conditional Value at Risk: the average return at or below the Value at Risk
          per period (historical).

        Losses, drawdowns, the Value at Risk and the Conditional Value at Risk are
        negative returns, as in the Risk module.

        For more information, see the following papers:

        - Sharpe, W. F. (1966). "Mutual Fund Performance." The Journal of Business, 39(1),
          119-138.
        - Sortino, F. A. and Price, L. N. (1994). "Performance Measurement in a Downside
          Risk Framework." The Journal of Investing, 3(3), 59-64.
        - Rockafellar, R. T. and Uryasev, S. (2002). "Conditional Value-at-Risk for General
          Loss Distributions." Journal of Banking & Finance, 26(7), 1443-1471.

        Also known as: portfolio risk report, risk summary, risk metrics of weights.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            confidence_level (float, optional): The confidence level of the Value at Risk
                and the Conditional Value at Risk, e.g. 0.95 for the worst 5% of periods.
                Defaults to 0.95.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Annualized Return, Annualized Volatility, Sharpe Ratio,
            Sortino Ratio, Maximum Drawdown, Value at Risk and Conditional Value at Risk of
            the Portfolio and, when there is one, the Benchmark over the same periods.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_portfolio_risk_metrics(
            weights={"AAPL": 0.4, "MSFT": 0.3, "JNJ": 0.2, "XOM": 0.1}
        )
        ```

        Which returns:

        |                           |   Portfolio |   Benchmark |
        |:--------------------------|------------:|------------:|
        | Annualized Return         |      0.1847 |      0.1472 |
        | Annualized Volatility     |      0.1855 |      0.1711 |
        | Sharpe Ratio              |      0.8247 |      0.6912 |
        | Sortino Ratio             |      1.211  |      0.996  |
        | Maximum Drawdown          |     -0.1977 |     -0.245  |
        | Value at Risk             |     -0.0185 |     -0.0166 |
        | Conditional Value at Risk |     -0.0263 |     -0.0248 |
        """
        returns = self._get_returns(period, include_benchmark)
        resolved = self._resolve_weights(weights, returns.columns.tolist())
        series = {PORTFOLIO_TICKER: returns @ resolved}
        benchmark = self._get_benchmark_returns(period)
        if benchmark is not None:
            series[BENCHMARK_TICKER] = benchmark.reindex(returns.index)
        metrics = optimization_model.get_portfolio_risk_metrics(
            pd.DataFrame(series),
            PERIODS_PER_YEAR[period],
            self._get_risk_free_rate(period, returns.index),
            confidence_level,
        )
        return self._round(metrics, rounding)

    @handle_errors
    def get_discrete_allocation(
        self,
        weights: dict[str, float] | None = None,
        portfolio_value: float = 10000,
        method: str = "integer",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how many whole shares to buy of each asset to hold weights with a
        portfolio value, at the latest traded (split-adjusted) closing price. Weights
        rarely come out in whole shares, so some cash is left over and the weights held
        differ a little from the targets; the smaller the portfolio value compared to the
        share prices, the larger the difference. With t_i = w_i V the target amount of
        asset i and p_i its price:

        - method="integer": the number of shares n_i that minimizes the absolute
          deviation from the target amounts with the leftover cash c counted as a
          holding with a target of zero, min Σ |t_i - n_i p_i| + c with
          Σ n_i p_i + c = V and c ≥ 0, solved as an integer programme (Cornuejols, Peña
          and Tütüncü, 2018).
        - method="greedy": every position is first rounded down to whole shares and the
          leftover cash then buys one more share at a time of the asset furthest below
          its target amount, while its price fits the cash. Quick and close to the
          integer programme, which can do better by leaving out a share of one asset to
          buy more of another.

        A short weight becomes a negative number of shares, with the proceeds funding
        the long positions: weights that sum to one keep a net investment of the
        portfolio value. Short positions are rounded to a larger short, so the leftover
        cash is never negative.

        For more information, see the following book:

        - Cornuejols, G., Peña, J. and Tütüncü, R. (2018). "Optimization Methods in
          Finance." 2nd edition, Cambridge University Press.

        Also known as: share allocation, whole share rounding, lot sizing, order sizes.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset, e.g. those
                of one of the optimizers. Defaults to None, which uses the latest weights
                of the Portfolio the Toolkit came from, or equal weights.
            portfolio_value (float, optional): The amount to invest, in the currency of
                the prices. Defaults to 10000.
            method (str, optional): "integer" or "greedy". Defaults to "integer".
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset the Price, Shares, Amount (shares times price),
            Target Weight and Weight held (the amount over the portfolio value), a
            Leftover Cash row and a Total row that sums to the portfolio value.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        weights = toolkit.portfolio.get_maximum_sharpe_weights()["Weight"]

        toolkit.portfolio.get_discrete_allocation(
            weights=weights.drop("Portfolio").to_dict(), portfolio_value=25000
        )
        ```

        Which returns:

        | Asset         |   Price |   Shares |   Amount |   Target Weight |   Weight |
        |:--------------|--------:|---------:|---------:|----------------:|---------:|
        | AAPL          |  271.86 |        1 |   271.86 |          0.014  |   0.0109 |
        | MSFT          |  483.62 |       16 |  7737.92 |          0.3121 |   0.3095 |
        | JNJ           |  206.95 |       24 |  4966.8  |          0.1998 |   0.1987 |
        | XOM           |  120.34 |       99 | 11913.7  |          0.4741 |   0.4765 |
        | Leftover Cash |  nan    |      nan |   109.76 |          0      |   0.0044 |
        | Total         |  nan    |      nan | 25000    |          1      |   1      |
        """
        closes = self._historical_data["daily"]["Close"].loc[: self._end_date]
        assets = [
            asset
            for asset in self._get_assets(include_benchmark)
            if asset in closes.columns
        ]
        resolved = self._resolve_weights(weights, assets)
        # The latest traded price of each asset, which can be of an earlier day for an
        # asset that did not trade on the last day.
        prices = closes[assets].ffill().iloc[-1]
        table = optimization_model.get_discrete_allocation(
            resolved, prices, portfolio_value, method
        )
        table.index.name = "Asset"
        table.loc["Total"] = [
            np.nan,
            np.nan,
            table["Amount"].sum(),
            table["Target Weight"].sum(),
            table["Weight"].sum(),
        ]
        return self._round(table, rounding)

    @handle_errors
    def get_optimization_backtest(
        self,
        methods: list[str] | str | None = None,
        rebalancing: str = "monthly",
        lookback: int = 252,
        transaction_cost: float = 0.0,
        maximum_turnover: float | None = None,
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        output: str = "summary",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Backtest the portfolio optimizers out of sample against equal weights (1/N) and
        the benchmark. On every rebalancing date (the last trading day of each month,
        quarter or year) each optimizer sets its weights from the daily returns of the
        `lookback` trading days up to and including that date, and the portfolio holds
        them from the next day on, so no weight is estimated with a return it is then
        measured on (walk-forward). Between rebalancing dates the weights drift with the
        prices, as those of a portfolio that does not trade. The 1/N portfolio is
        rebalanced to equal weights on the same dates and the benchmark is bought and
        held.

        Each rebalancing trades from the drifted weights to the new ones. Its turnover is
        Σ |w_i,new - w_i,drifted| and its cost the transaction cost times the turnover,
        taken from the portfolio's value, so the returns shown are after costs
        (DeMiguel, Garlappi and Uppal, 2009). With a maximum turnover, a portfolio trades
        towards its new weights only as far as the limit allows. DeMiguel et al. found
        that none of fourteen optimizers consistently beat 1/N out of sample, because the
        estimation error in the expected returns outweighs the gain from optimizing; a
        backtest shows whether that holds for these assets.

        For more information, see the following paper:

        - DeMiguel, V., Garlappi, L. and Uppal, R. (2009). "Optimal Versus Naive
          Diversification: How Inefficient Is the 1/N Portfolio Strategy?" The Review of
          Financial Studies, 22(5), 1915-1953.

        Also known as: walk-forward backtest, out-of-sample backtest, rolling
        optimization, 1/N comparison.

        Args:
            methods (list[str] | str | None, optional): The optimizers to backtest, as a
                list or comma separated: "minimum_variance", "maximum_sharpe",
                "mean_variance" (a risk aversion of 2.5), "risk_parity",
                "hierarchical_risk_parity", "hierarchical_equal_risk_contribution",
                "nested_clustered_optimization" (minimum variance within and across
                the clusters), "inverse_volatility", "maximum_diversification",
                "minimum_cvar" (95%), "minimum_drawdown" (the 95% CDaR),
                "maximum_omega" (a minimum acceptable return of 0) and
                "maximum_cvar_ratio" (95%). Equal weights and the benchmark are always
                included. Defaults to None, which backtests
                minimum variance, maximum Sharpe, risk parity, Hierarchical Risk Parity,
                maximum diversification and inverse volatility.
            rebalancing (str, optional): How often the weights are set anew: "monthly",
                "quarterly" or "yearly". Defaults to "monthly".
            lookback (int, optional): The number of daily returns each estimation uses.
                The backtest starts on the first rebalancing date with that many returns
                up to and including it. Defaults to 252, a year of trading days.
            transaction_cost (float, optional): The cost per unit of weight traded, e.g.
                0.001 for 10 basis points. The first investment is free, so every
                portfolio starts from the same value. Defaults to 0.
            maximum_turnover (float | None, optional): The highest turnover of a
                rebalancing, e.g. 0.2. Defaults to None, no limit.
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (towards the scaled identity, Ledoit and Wolf,
                2004), "ledoit_wolf_constant_correlation" (towards the average
                correlation, Ledoit and Wolf, 2004), "ledoit_wolf_single_factor"
                (towards the covariance that the benchmark as a single factor implies,
                Ledoit and Wolf, 2003), "oas" (Chen et al., 2010), "semicovariance" (the
                co-movement below a zero return, Estrada, 2007), "exponential" (recent
                periods weigh most, RiskMetrics) or "denoised" (the eigenvalues of the
                correlation matrix that random returns could produce averaged, Laloux et
                al., 2000). The shrinkage and denoised estimators stabilize the weights
                when there are many assets compared to the number of periods. Defaults
                to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            expected_returns (str, optional): The estimator of the expected returns:
                "mean" (the historical average), "exponential" (the exponentially
                weighted average, recent periods weigh most), "capm" (the returns the
                Capital Asset Pricing Model implies from each asset's beta to the
                benchmark, Sharpe, 1964) or "james_stein" (the averages shrunk towards
                that of the minimum variance portfolio, Jorion, 1986). Defaults to
                "mean".
            expected_returns_span (int, optional): The span in periods of the
                "exponential" expected returns. Defaults to 180.
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, for the optimizers that take weight bounds (all
                but inverse volatility and nested clustered optimization). Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, for the optimizers that take weight bounds.
                Defaults to 1.
            output (str, optional): "summary" for the performance of each portfolio,
                "returns" for the daily returns after costs, or "weights" for the weights
                set on each rebalancing date. Defaults to "summary".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: With output="summary", per portfolio the Annualized Return
            (geometric), Annualized Volatility, Sharpe Ratio (the annualized average return
            above the risk free rate over the volatility), Maximum Drawdown, Average
            Turnover per rebalancing, Annualized Costs, and the In-Sample Return and
            In-Sample Sharpe Ratio: what the weights expected when they were set (see the
            Notes). With output="returns", the
            daily returns after costs of every portfolio and the benchmark. With
            output="weights", the weights each optimizer set on each rebalancing date.

        Notes:
        - The returns are daily total returns (from adjusted closing prices). The
          estimates use the days on which every asset has a return; the portfolios earn
          the returns of every day, so a day on which one market is closed (its assets
          return 0%, the next day covers both) still counts for the others. Each
          portfolio and the benchmark are measured over the days they have a return.
        - When an optimizer finds no weights on a rebalancing date (e.g. the maximum Sharpe
          ratio when no asset beat the risk free rate over the lookback), the portfolio
          keeps its drifted weights until the next date, and a warning says how often.
        - The In-Sample Return is the average over the rebalancing dates of w · mu, the
          annualized return the weights set on a date expected from the estimates of
          their own window (with the chosen expected return and covariance estimators),
          and the In-Sample Sharpe Ratio the average of (w · mu - rf) / sqrt(w' Sigma w).
          This is the promise an optimizer makes, on the same dates as the realised
          return next to it, so the gap between the two is the estimation error that
          DeMiguel et al. describe. The in-sample return is arithmetic, the realised
          Annualized Return geometric, which is lower by about half the variance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_optimization_backtest(transaction_cost=0.001)
        ```

        Which returns:

        | Portfolio                |   Annualized Return |   Annualized Volatility |   Sharpe Ratio |   Maximum Drawdown |   Average Turnover |   Annualized Costs |   In-Sample Return |   In-Sample Sharpe Ratio |
        |:-------------------------|--------------------:|------------------------:|---------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------------:|
        | Minimum Variance         |              0.0891 |                  0.1375 |         0.4061 |            -0.1268 |             0.0667 |             0.0008 |             0.1163 |                   0.6525 |
        | Maximum Sharpe           |              0.0828 |                  0.2124 |         0.2971 |            -0.2776 |             0.4378 |             0.0052 |             0.3317 |                   1.4328 |
        | Risk Parity              |              0.1242 |                  0.1469 |         0.6056 |            -0.1439 |             0.0428 |             0.0005 |             0.1429 |                   0.7982 |
        | Hierarchical Risk Parity |              0.1003 |                  0.1382 |         0.4784 |            -0.135  |             0.0474 |             0.0006 |             0.1176 |                   0.6484 |
        | Maximum Diversification  |              0.1225 |                  0.1426 |         0.6085 |            -0.1357 |             0.0617 |             0.0007 |             0.1468 |                   0.8404 |
        | Inverse Volatility       |              0.122  |                  0.151  |         0.5798 |            -0.1484 |             0.0415 |             0.0005 |             0.1408 |                   0.7745 |
        | Equal Weight             |              0.1407 |                  0.1624 |         0.6519 |            -0.1594 |             0.0388 |             0.0005 |             0.1582 |                   0.8381 |
        | Benchmark                |              0.1285 |                  0.1799 |         0.545  |            -0.221  |           nan      |           nan      |           nan      |                 nan      |
        """
        if output not in ("summary", "returns", "weights"):
            raise ValueError(
                f"The output must be 'summary', 'returns' or 'weights', not '{output}'."
            )
        if isinstance(methods, str):
            methods = [
                method.strip() for method in methods.split(",") if method.strip()
            ]
        methods = list(methods or DEFAULT_BACKTEST_METHODS)
        unknown = [method for method in methods if method not in BACKTEST_METHODS]
        if unknown:
            raise ValueError(
                f"Unknown methods: {', '.join(unknown)}; choose from "
                f"{', '.join(BACKTEST_METHODS)}."
            )
        if "equal_weight" not in methods:
            methods.append("equal_weight")

        returns = self._get_returns("daily", include_benchmark=False)
        if lookback < returns.shape[1] + 2:
            raise ValueError(
                f"A lookback of {lookback} days is too short to estimate the covariance "
                f"of {returns.shape[1]} assets."
            )
        benchmark = self._get_benchmark_returns("daily")
        if expected_returns == "capm" and benchmark is None:
            raise ValueError("The CAPM expected returns need a benchmark.")
        if covariance == "ledoit_wolf_single_factor" and benchmark is None:
            raise ValueError("The single factor covariance needs a benchmark.")
        periods_per_year = PERIODS_PER_YEAR["daily"]
        # The portfolios earn every day's return: a day on which only some markets traded
        # is a 0% return for the others, whose next return covers both days. A day on
        # which none of the assets traded (only the benchmark's market was open) is no
        # day of the portfolios.
        holding_returns = (
            self._historical_data["daily"]["Return"]
            .loc[returns.index[0] : returns.index[-1], returns.columns]
            .dropna(how="all")
            .fillna(0)
        )
        # The estimates of a window are the same for every method, so they are made once.
        estimates: dict = {}

        def window_estimates(window: pd.DataFrame) -> tuple:
            if window.index[-1] not in estimates:
                risk_free_rate = self._get_risk_free_rate("daily", window.index)
                estimates[window.index[-1]] = (
                    risk_free_rate,
                    *optimization_model.get_expected_returns_and_covariance(
                        window,
                        periods_per_year,
                        covariance,
                        covariance_span,
                        expected_returns,
                        expected_returns_span,
                        (
                            benchmark
                            if expected_returns == "capm"
                            or covariance == "ledoit_wolf_single_factor"
                            else None
                        ),
                        risk_free_rate,
                    ),
                )
            return estimates[window.index[-1]]

        def strategy(method: str) -> Callable[[pd.DataFrame], pd.Series]:
            def weights(window: pd.DataFrame) -> pd.Series:
                if method == "equal_weight":
                    return pd.Series(1 / window.shape[1], index=window.columns)
                if method == "minimum_cvar":
                    return optimization_model.get_minimum_cvar_weights(
                        window,
                        minimum_weight=minimum_weight,
                        maximum_weight=maximum_weight,
                    )
                if method == "minimum_drawdown":
                    return optimization_model.get_minimum_drawdown_weights(
                        window,
                        minimum_weight=minimum_weight,
                        maximum_weight=maximum_weight,
                    )
                risk_free_rate, mu, sigma = window_estimates(window)
                if method == "maximum_omega":
                    return optimization_model.get_maximum_omega_weights(
                        window,
                        minimum_weight=minimum_weight,
                        maximum_weight=maximum_weight,
                        expected_returns=mu / periods_per_year,
                    )
                if method == "maximum_cvar_ratio":
                    return optimization_model.get_maximum_cvar_ratio_weights(
                        window,
                        risk_free_rate=risk_free_rate / periods_per_year,
                        minimum_weight=minimum_weight,
                        maximum_weight=maximum_weight,
                        expected_returns=mu / periods_per_year,
                    )
                if method == "hierarchical_equal_risk_contribution":
                    return optimization_model.get_hierarchical_equal_risk_contribution_weights(
                        sigma,
                        minimum_weight=minimum_weight,
                        maximum_weight=maximum_weight,
                    )
                if method == "nested_clustered_optimization":
                    return optimization_model.get_nested_clustered_optimization_weights(
                        mu, sigma, risk_free_rate=risk_free_rate
                    )
                if method == "inverse_volatility":
                    return optimization_model.get_inverse_volatility_weights(sigma)
                if method == "hierarchical_risk_parity":
                    return optimization_model.get_hierarchical_risk_parity_weights(
                        sigma,
                        minimum_weight=minimum_weight,
                        maximum_weight=maximum_weight,
                    )
                if method == "risk_parity":
                    return optimization_model.get_risk_parity_weights(
                        sigma,
                        minimum_weight=minimum_weight,
                        maximum_weight=maximum_weight,
                    )
                if method == "minimum_variance":
                    return optimization_model.get_minimum_variance_weights(
                        sigma, minimum_weight, maximum_weight
                    )
                if method == "maximum_diversification":
                    return optimization_model.get_maximum_diversification_weights(
                        sigma, minimum_weight, maximum_weight
                    )
                if method == "maximum_sharpe":
                    return optimization_model.get_maximum_sharpe_weights(
                        mu, sigma, risk_free_rate, minimum_weight, maximum_weight
                    )
                return optimization_model.get_mean_variance_weights(
                    mu,
                    sigma,
                    minimum_weight=minimum_weight,
                    maximum_weight=maximum_weight,
                )

            return weights

        net_returns, weights_table, turnover = backtest_model.get_walk_forward_backtest(
            returns,
            {BACKTEST_METHODS[method]: strategy(method) for method in methods},
            backtest_model.get_rebalancing_dates(returns.index, rebalancing, lookback),
            lookback,
            transaction_cost,
            maximum_turnover,
            holding_returns,
        )

        if output == "weights":
            return self._round(weights_table, rounding)

        if benchmark is not None:
            net_returns[BENCHMARK_TICKER] = benchmark.reindex(net_returns.index)
        if output == "returns":
            return self._round(net_returns, rounding)

        summary = backtest_model.get_backtest_statistics(
            net_returns,
            turnover,
            periods_per_year,
            self._get_risk_free_rate("daily", net_returns.index),
            transaction_cost,
        )
        # What each optimizer's weights expected from their own window, so the gap
        # between the promise and the realised return shows in one table.
        for date in weights_table.index:
            position = returns.index.get_loc(date)
            window_estimates(returns.iloc[position - lookback + 1 : position + 1])
        summary = summary.join(
            backtest_model.get_in_sample_expectations(weights_table, estimates)
        )
        summary.index.name = "Portfolio"
        return self._round(summary, rounding)
