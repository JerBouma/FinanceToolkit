"""Portfolio Analysis Module"""

__docformat__ = "google"

import functools
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.analysis import backtest_model
from financetoolkit.portfolio.construction.attribution import attribution_model
from financetoolkit.portfolio.construction.optimization import (
    estimation_model,
    optimization_model,
)
from financetoolkit.portfolio.helpers import BENCHMARK_TICKER, PORTFOLIO_TICKER
from financetoolkit.risk.helpers import get_observations_per_year
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    check_option,
    check_period,
    handle_errors,
    validate_alpha,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR

# ruff: noqa: E501

logger = get_logger()

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
    "minimum_mad": "Minimum MAD",
    "minimum_semideviation": "Minimum Semideviation",
    "random": "Random",
    "equal_weight": "Equal Weight",
}

# The seed of the generator that draws the random portfolios, one per backtest, so the
# same backtest gives the same random portfolios.
RANDOM_BACKTEST_SEED = 0

# The expected return estimators, with the names they are shown under.
EXPECTED_RETURN_NAMES = {
    "mean": "Mean",
    "geometric": "Geometric",
    "exponential": "Exponential",
    "exponential_geometric": "Exponential Geometric",
    "capm": "CAPM",
    "bayes_stein": "Bayes-Stein",
    "james_stein": "James-Stein",
    "bodnar_okhrin_parolya": "Bodnar-Okhrin-Parolya",
    "centroid": "Centroid",
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
class AnalysisConstruction:
    """
    Analysis of a set of weights and of the estimates behind them.

    Risk contributions, diversification and risk metrics, the discrete allocation, the
    walk-forward optimization backtest, the expected return, covariance and codependence
    estimates, asset clusters, the coskewness and cokurtosis matrices and the uncertainty of
    the estimates.
    """

    @handle_errors
    def get_risk_contributions(
        self,
        weights: dict[str, float] | None = None,
        risk_measure: str = "volatility",
        alpha: float = 0.05,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how much each asset contributes to the risk of a portfolio.

        The risk is measured as the volatility, expected shortfall (CVaR), semideviation, mean
        absolute deviation, Conditional Drawdown at Risk (CDaR) or one of the Risk module's
        tail, dispersion and drawdown measures. The marginal contribution is the change in the
        portfolio's risk from a small increase in an asset's weight; the risk contribution is
        the weight times that, and the contributions add up to the portfolio's risk (Euler
        allocation). A large position in a calm asset can contribute less risk than a small one
        in a volatile asset.

        - The volatility: the marginal contribution is (Σw)_i / σ_p, with Σ
          the annualized covariance matrix.
        - The expected shortfall is the average loss in the periods
          in which the portfolio's return is at or below its Value at Risk, and the
          marginal contribution of an asset is its average loss in those same periods,
          -E[r_i | r_p ≤ VaR], so the contributions show which assets lose in the
          portfolio's worst periods. These are per period (e.g. daily), like the
          Finance Toolkit's historical CVaR, and not annualized.
        - The semideviation: the downside deviation of the portfolio's
          return below zero, SD⁻ = √(Σ min(r_p,t, 0)² / T), with the marginal
          contribution Σ min(r_p,t, 0) r_i,t / (T SD⁻): an asset contributes by losing
          in the periods in which the portfolio loses. Per period, not annualized.
        - The mean absolute deviation of the portfolio's return,
          Σ |r_p,t - μ_p| / T, with the marginal contribution
          Σ sign(r_p,t - μ_p)(r_i,t - μ_i) / T. Per period, not annualized.
        - The Conditional Drawdown at Risk, the average of the
          worst α share of the drawdowns of the uncompounded cumulative return (the
          drawdowns the minimum drawdown optimization minimizes). Each drawdown is the
          sum of the assets' drawdowns from the portfolio's peak, so the marginal
          contribution of an asset is its average drawdown over the portfolio's worst
          drawdown periods (Goldberg and Mahmoud, 2017).
        - The other measures: the historical Value at Risk and Entropic Value at Risk, the worst
          return, the Gini mean difference, the Tail Gini, the first lower partial moment below
          zero, the square root of the fourth central moment, and the maximum and average
          drawdown, Drawdown at Risk, Entropic Drawdown at Risk and Ulcer Index of the
          uncompounded cumulative return, as the Risk module calculates them, losses positive.
          The marginal contribution is found by central finite differences, (ρ(w + h
          e_i) - ρ(w - h e_i)) / 2h with h = 1e-7 (1e-4 for the entropic measures); the risk
          contribution is the weight times that (half of it for the kurtosis measure, whose
          square root of the fourth moment doubles when the weights double twice over), so the
          contributions add up to the measure. Per period, not annualized.

        For more information, see the following papers:

        - Litterman, R. (1997). "Hot Spots and Hedges." The Journal of Portfolio Management,
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

        Referenced against:
        - Riskfolio-Lib, Risk_Contribution (CVaR): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Risk_Contribution
        - Riskfolio-Lib, Risk_Contribution (other risk measures): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Risk_Contribution

        Shares similar logic:
        - Riskfolio-Lib, Risk_Contribution (volatility): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Risk_Contribution

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            risk_measure (str, optional): The risk measure. Options are: "volatility" (of the covariance estimate),
                "cvar" (the expected shortfall), "semideviation" (below a return of zero), "mad" (the mean absolute
                deviation), "cdar" (the Conditional Drawdown at Risk), "var" (historical Value at Risk), "evar"
                (Entropic Value at Risk), "worst_realization" (the largest loss), "gini_mean_difference" (Gini mean
                difference), "tail_gini" (Tail Gini), "lower_partial_moment" (first lower partial moment below
                zero), "kurtosis" (square root of the fourth central moment), "maximum_drawdown" (maximum drawdown),
                "average_drawdown" (average drawdown), "drawdown_at_risk" (Drawdown at Risk),
                "entropic_drawdown_at_risk" (Entropic Drawdown at Risk), "ulcer_index" (Ulcer Index). Defaults to
                "volatility".
            alpha (float, optional): The tail probability of the tail measures (the
                expected shortfall, CDaR, VaR, EVaR, Tail Gini, Drawdown at Risk and EDaR),
                e.g. 0.05 for the worst 5% of periods or drawdowns. Defaults to 0.05.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix. The shrinkage and denoised
                estimators stabilize the weights when there are many assets compared to the number of periods.
                Options are: "sample" (the sample covariance, ddof=1), "ledoit_wolf" (shrunk towards the scaled
                identity, Ledoit and Wolf, 2004), "ledoit_wolf_constant_correlation" (shrunk towards the average
                correlation, Ledoit and Wolf, 2004), "ledoit_wolf_single_factor" (shrunk towards the covariance that
                the benchmark as a single factor implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating
                Shrinkage, Chen et al., 2010), "shrunk" (a fixed 10% towards the scaled identity), "semicovariance"
                (the co-movement below a zero return, Estrada, 2008), "exponential" (exponentially weighted, recent
                periods weigh most, RiskMetrics), "denoised" (the eigenvalues of the correlation matrix that random
                returns could produce averaged, Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to
                zero), "denoised_shrink" (the noise correlations shrunk towards their diagonal, López de Prado,
                2020), "gerber" (co-movements beyond half a standard deviation, Gerber et al., 2022), "factor" (the
                covariance of the Fama and French five factor model with OLS loadings, Fama and French, 1993),
                "minimum_covariance_determinant" (the reweighted covariance of the half of the periods with the
                smallest covariance determinant, robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the
                sparse inverse covariance on the TMFG planar graph of the correlations, Barfuss et al., 2016),
                "geodesic" (10% towards the scaled identity along the geodesic of the positive definite matrices,
                Bhatia, 2007), "regime_adjusted_ew" (exponentially weighted with a half-life of 40 periods, rescaled
                by the recent realized against predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one
                part of the periods and the variances along them from the other, averaged over random splits, Lam,
                2016), "ccc_garch" (the next-period covariance of GARCH(1, 1) volatilities with constant
                correlations, Bollerslev, 1990), "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke"
                (Rocke's robust S-estimator, Maronna et al., 2019). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
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
            Contribution and Percentage Contribution: annualized volatility terms, or per period
            terms of the other measures, with losses and drawdowns positive. For the kurtosis
            measure the marginal contribution is the derivative of the measure and the risk
            contribution half the weight times it. The assets held (a weight other than zero)
            are listed, over the periods in which each of them has a return; a loaded asset
            without weight does not shorten the sample.

        Notes:
        - The Value at Risk and Drawdown at Risk are the Risk module's quantiles, interpolated
          between two periods, where Riskfolio-Lib takes a single period (the ⌈αT⌉-th worst), so
          their contributions mix two periods and differ from Riskfolio-Lib's; the other
          measures match Riskfolio-Lib's to 1e-8.

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
        validate_alpha(alpha)
        measures = (
            "volatility",
            "cvar",
            *optimization_model.HISTORICAL_RISK_MEASURES,
            *optimization_model.FINITE_DIFFERENCE_RISK_MEASURES,
        )
        check_option("risk_measure", risk_measure, measures)
        # Only the assets held decide the periods, so an asset left out of the weights with
        # a shorter history does not shorten the sample.
        resolved = self._get_held_weights(weights, period, include_benchmark)
        returns, _, covariance_matrix, _ = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            assets=resolved.index.tolist(),
        )
        if risk_measure == "volatility":
            table = optimization_model.get_risk_contributions(
                resolved, covariance_matrix
            )
        elif risk_measure == "cvar":
            table = optimization_model.get_expected_shortfall_contributions(
                resolved, returns, alpha
            )
        elif risk_measure in optimization_model.FINITE_DIFFERENCE_RISK_MEASURES:
            table = optimization_model.get_finite_difference_risk_contributions(
                resolved, returns, risk_measure, alpha
            )
        else:
            table = optimization_model.get_historical_risk_contributions(
                resolved, returns, risk_measure, alpha
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

        Referenced against:
        - skfolio, Portfolio (diversification and effective_number_assets): https://skfolio.org/generated/skfolio.portfolio.Portfolio.html

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix. The shrinkage and denoised
                estimators stabilize the weights when there are many assets compared to the number of periods.
                Options are: "sample" (the sample covariance, ddof=1), "ledoit_wolf" (shrunk towards the scaled
                identity, Ledoit and Wolf, 2004), "ledoit_wolf_constant_correlation" (shrunk towards the average
                correlation, Ledoit and Wolf, 2004), "ledoit_wolf_single_factor" (shrunk towards the covariance that
                the benchmark as a single factor implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating
                Shrinkage, Chen et al., 2010), "shrunk" (a fixed 10% towards the scaled identity), "semicovariance"
                (the co-movement below a zero return, Estrada, 2008), "exponential" (exponentially weighted, recent
                periods weigh most, RiskMetrics), "denoised" (the eigenvalues of the correlation matrix that random
                returns could produce averaged, Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to
                zero), "denoised_shrink" (the noise correlations shrunk towards their diagonal, López de Prado,
                2020), "gerber" (co-movements beyond half a standard deviation, Gerber et al., 2022), "factor" (the
                covariance of the Fama and French five factor model with OLS loadings, Fama and French, 1993),
                "minimum_covariance_determinant" (the reweighted covariance of the half of the periods with the
                smallest covariance determinant, robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the
                sparse inverse covariance on the TMFG planar graph of the correlations, Barfuss et al., 2016),
                "geodesic" (10% towards the scaled identity along the geodesic of the positive definite matrices,
                Bhatia, 2007), "regime_adjusted_ew" (exponentially weighted with a half-life of 40 periods, rescaled
                by the recent realized against predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one
                part of the periods and the variances along them from the other, averaged over random splits, Lam,
                2016), "ccc_garch" (the next-period covariance of GARCH(1, 1) volatilities with constant
                correlations, Bollerslev, 1990), "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke"
                (Rocke's robust S-estimator, Maronna et al., 2019). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
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
            Diversification Ratio and Effective Number of Bets of the portfolio, from
            the covariance of the assets held (a weight other than zero) over the periods
            in which each of them has a return.

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
        resolved = self._get_held_weights(weights, period, include_benchmark)
        _, _, covariance_matrix, _ = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            assets=resolved.index.tolist(),
        )
        metrics = optimization_model.get_diversification_metrics(
            resolved, covariance_matrix
        )
        metrics.name = PORTFOLIO_TICKER
        return self._round(metrics.to_frame(), rounding)[PORTFOLIO_TICKER]

    @handle_errors
    def get_portfolio_risk_metrics(
        self,
        weights: dict[str, float] | None = None,
        alpha: float = 0.05,
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the return and risk of a portfolio with given weights over the Toolkit's date
        range, next to the benchmark.

        The portfolio's return in each period is the weighted return of its assets, r_p,t = Σ
        w_i r_i,t, which holds the weights constant by rebalancing to them every period. The
        measures use the formulas of the Performance and Risk modules:

        - Annualized Return: the geometric average, Π(1 + r_t)^(P/T) - 1, with P the
          periods per year and T the number of periods.
        - Annualized Volatility: the standard deviation (ddof=1) times √P.
        - Sharpe Ratio: the average return above the risk free rate over its standard
          deviation, annualized with √P (Sharpe, 1966).
        - Sortino Ratio: the average return above the risk free rate over its downside
          deviation below zero, annualized with √P (Sortino and Price, 1994).
        - Maximum Drawdown: the largest fall of the compounded value from its peak.
        - Value at Risk: the α quantile of the returns per period (historical).
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

        Also known as: portfolio risk report, risk summary, risk metrics of weights, VaR and CVaR of
        a weighted portfolio.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            alpha (float, optional): The tail probability of the Value at Risk and the
                Conditional Value at Risk, e.g. 0.05 for the worst 5% of periods.
                Defaults to 0.05.
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
            pd.DataFrame: The Annualized Return, Annualized Volatility, Sharpe Ratio,
            Sortino Ratio, Maximum Drawdown, Value at Risk and Conditional Value at Risk of
            the Portfolio and, when there is one, the Benchmark over the same periods:
            those in which every asset held (a weight other than zero) has a return, so a
            loaded asset without weight does not shorten the sample.

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
        # Only the assets held decide the periods, so an asset left out of the weights with
        # a shorter history does not shorten the sample.
        validate_alpha(alpha)
        resolved = self._get_held_weights(weights, period, include_benchmark)
        returns = self._get_returns(period, include_benchmark, resolved.index.tolist())
        series = {PORTFOLIO_TICKER: returns @ resolved}
        benchmark = self._get_benchmark_returns(period)
        if benchmark is not None:
            series[BENCHMARK_TICKER] = benchmark.reindex(returns.index)
        metrics = optimization_model.get_portfolio_risk_metrics(
            pd.DataFrame(series),
            self._get_periods_per_year(period, returns.index),
            self._get_risk_free_rate(period, returns.index),
            alpha,
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
        Calculate how many whole shares to buy of each asset to hold weights with a portfolio
        value, at the latest traded (split-adjusted) closing price.

        Weights rarely come out in whole shares, so some cash is left over and the weights held
        differ a little from the targets; the smaller the portfolio value compared to the share
        prices, the larger the difference. With t_i = w_i V the target amount of asset i and p_i
        its price:

        - The integer programme: the number of shares n_i that minimizes the absolute
          deviation from the target amounts with the leftover cash c counted as a
          holding with a target of zero, min Σ |t_i - n_i p_i| + c with
          Σ n_i p_i + c = V and c ≥ 0, solved as an integer programme (Cornuejols, Peña
          and Tütüncü, 2018).
        - The greedy allocation: every position is first rounded down to whole shares and the
          leftover cash then buys one more share at a time of the asset furthest below
          its target amount, while its price fits the cash. Quick and close to the
          integer programme, which can do better by leaving out a share of one asset to
          buy more of another.

        A short weight becomes a negative number of shares, with the proceeds funding
        the long positions: weights that sum to one keep a net investment of the
        portfolio value. Short positions are rounded to a larger short, so the leftover
        cash is never negative. The integer programme holds a long position at no more
        than the whole shares that reach its target, but leaves the size of a short
        position open: one more share short raises the cash for a long share that is
        further below its own target, which can lower the total deviation.

        For more information, see the following book:

        - Cornuejols, G., Peña, J. and Tütüncü, R. (2018). "Optimization Methods in
          Finance." 2nd edition, Cambridge University Press.

        Also known as: share allocation, whole share rounding, lot sizing, order sizes.

        Shares similar logic:
        - PyPortfolioOpt, DiscreteAllocation.lp_portfolio (integer programme): https://pyportfolioopt.readthedocs.io/en/latest/Postprocessing.html#pypfopt.discrete_allocation.DiscreteAllocation.lp_portfolio
        - PyPortfolioOpt, DiscreteAllocation.greedy_portfolio (greedy allocation): https://pyportfolioopt.readthedocs.io/en/latest/Postprocessing.html#pypfopt.discrete_allocation.DiscreteAllocation.greedy_portfolio

        Args:
            weights (dict[str, float] | None, optional): The weight per asset, e.g. those
                of one of the optimizers. Defaults to None, which uses the latest weights
                of the Portfolio the Toolkit came from, or equal weights.
            portfolio_value (float, optional): The amount to invest, in the currency of
                the prices. Defaults to 10000.
            method (str, optional): Options are: "integer" (the integer programme), "greedy" (the greedy
                allocation). Defaults to "integer".
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
        Backtest the portfolio optimizers out of sample against equal weights (1/N) and the
        benchmark.

        On every rebalancing date (the last trading day of each month, quarter or year) each
        optimizer sets its weights from the daily returns of the `lookback` trading days up to
        and including that date, and the portfolio holds them from the next day on, so no weight
        is estimated with a return it is then measured on (walk-forward). Between rebalancing
        dates the weights drift with the prices, as those of a portfolio that does not trade.
        The 1/N portfolio is rebalanced to equal weights on the same dates and the benchmark is
        bought and held.

        Each rebalancing trades from the drifted weights to the new ones. Its turnover is
        Σ |w_i,new - w_i,drifted| and its cost the transaction cost times the turnover,
        taken from the portfolio's value, so the returns shown are after costs
        (DeMiguel, Garlappi and Uppal, 2009). With a maximum turnover, a portfolio trades
        towards its new weights only as far as the limit allows. DeMiguel et al. found
        that none of fourteen optimizers consistently beat 1/N out of sample, because the
        estimation error in the expected returns outweighs the gain from optimizing; a
        backtest shows whether that holds for these assets.

        For more information, see the following papers:

        - DeMiguel, V., Garlappi, L. and Uppal, R. (2009). "Optimal Versus Naive
          Diversification: How Inefficient Is the 1/N Portfolio Strategy?" The Review of
          Financial Studies, 22(5), 1915-1953.
        - Konno, H. and Yamazaki, H. (1991). "Mean-Absolute Deviation Portfolio
          Optimization Model and Its Applications to Tokyo Stock Market." Management
          Science, 37(5), 519-531.
        - Estrada, J. (2008). "Mean-Semivariance Optimization: A Heuristic Approach."
          Journal of Applied Finance, 18(1), 57-72.
        - Rubin, D. B. (1981). "The Bayesian Bootstrap." The Annals of Statistics, 9(1),
          130-134.

        Also known as: walk-forward backtest, out-of-sample backtest, rolling
        optimization, 1/N comparison.

        Args:
            methods (list[str] | str | None, optional): The optimizers to backtest, as a list or comma separated.
                Options are: "minimum_variance" (Minimum Variance), "maximum_sharpe" (Maximum Sharpe),
                "mean_variance" (a risk aversion of 2.5), "risk_parity" (Risk Parity), "hierarchical_risk_parity"
                (Hierarchical Risk Parity), "inverse_volatility" (Inverse Volatility), "maximum_diversification"
                (Maximum Diversification), "hierarchical_equal_risk_contribution" (Hierarchical Equal Risk
                Contribution), "nested_clustered_optimization" (minimum variance within and across the clusters),
                "minimum_cvar" (95%), "minimum_drawdown" (the 95% CDaR), "maximum_omega" (a minimum acceptable
                return of 0), "maximum_cvar_ratio" (95%), "minimum_mad" (the lowest mean absolute deviation, Konno
                and Yamazaki, 1991), "minimum_semideviation" (the lowest semideviation below a return of 0, Estrada,
                2008), "random" (weights drawn uniformly from all long-only portfolios, a flat Dirichlet, with a
                generator seeded at 0 so the backtest is reproducible), "equal_weight" (Equal Weight). Equal weights
                and the benchmark are always included. Defaults to None, which backtests minimum variance, maximum
                Sharpe, risk parity, Hierarchical Risk Parity, maximum diversification and inverse volatility.
            rebalancing (str, optional): How often the weights are set anew. Options are: "monthly" (every month),
                "quarterly" (every quarter), "yearly" (every year). Defaults to "monthly".
            lookback (int, optional): The number of daily returns each estimation uses.
                The backtest starts on the first rebalancing date with that many returns
                up to and including it. Defaults to 252, a year of trading days.
            transaction_cost (float, optional): The cost per unit of weight traded, e.g.
                0.001 for 10 basis points. The first investment is free, so every
                portfolio starts from the same value. Defaults to 0.
            maximum_turnover (float | None, optional): The highest turnover of a
                rebalancing, e.g. 0.2. Defaults to None, no limit.
            covariance (str, optional): The estimator of the covariance matrix. The shrinkage and denoised
                estimators stabilize the weights when there are many assets compared to the number of periods.
                Options are: "sample" (the sample covariance, ddof=1), "ledoit_wolf" (shrunk towards the scaled
                identity, Ledoit and Wolf, 2004), "ledoit_wolf_constant_correlation" (shrunk towards the average
                correlation, Ledoit and Wolf, 2004), "ledoit_wolf_single_factor" (shrunk towards the covariance that
                the benchmark as a single factor implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating
                Shrinkage, Chen et al., 2010), "shrunk" (a fixed 10% towards the scaled identity), "semicovariance"
                (the co-movement below a zero return, Estrada, 2008), "exponential" (exponentially weighted, recent
                periods weigh most, RiskMetrics), "denoised" (the eigenvalues of the correlation matrix that random
                returns could produce averaged, Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to
                zero), "denoised_shrink" (the noise correlations shrunk towards their diagonal, López de Prado,
                2020), "gerber" (co-movements beyond half a standard deviation, Gerber et al., 2022), "factor" (the
                covariance of the Fama and French five factor model with OLS loadings, Fama and French, 1993),
                "minimum_covariance_determinant" (the reweighted covariance of the half of the periods with the
                smallest covariance determinant, robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the
                sparse inverse covariance on the TMFG planar graph of the correlations, Barfuss et al., 2016),
                "geodesic" (10% towards the scaled identity along the geodesic of the positive definite matrices,
                Bhatia, 2007), "regime_adjusted_ew" (exponentially weighted with a half-life of 40 periods, rescaled
                by the recent realized against predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one
                part of the periods and the variances along them from the other, averaged over random splits, Lam,
                2016), "ccc_garch" (the next-period covariance of GARCH(1, 1) volatilities with constant
                correlations, Bollerslev, 1990), "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke"
                (Rocke's robust S-estimator, Maronna et al., 2019). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
            expected_returns (str, optional): The estimator of the expected returns. Options are: "mean" (the
                historical average), "geometric" (the compounded average), "exponential" (the exponentially weighted
                average, recent periods weigh most), "exponential_geometric" (that average compounded over a year),
                "capm" (the returns the Capital Asset Pricing Model implies from each asset's beta to the benchmark,
                Sharpe, 1964), "bayes_stein" (the averages shrunk towards that of the minimum variance portfolio,
                Jorion, 1986), "james_stein" (the averages shrunk towards their grand mean, James and Stein, 1961),
                "bodnar_okhrin_parolya" (the averages and their grand mean combined with optimal weights, Bodnar,
                Okhrin and Parolya, 2019). Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the exponentially weighted expected
                returns. Defaults to 180.
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, for the optimizers that take weight bounds (all
                but inverse volatility, nested clustered optimization and random).
                Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, for the optimizers that take weight bounds.
                Defaults to 1.
            output (str, optional): Options are: "summary" (the performance of each portfolio), "returns" (the daily
                returns after costs), "weights" (the weights set on each rebalancing date). Defaults to "summary".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: For the summary, per portfolio the Annualized Return
            (geometric), Annualized Volatility, Sharpe Ratio (the annualized average return
            above the risk free rate over the volatility), Maximum Drawdown, Average
            Turnover per rebalancing, Annualized Costs, and the In-Sample Return and
            In-Sample Sharpe Ratio: what the weights expected when they were set (see the
            Notes). For the returns, the
            daily returns after costs of every portfolio and the benchmark. For
            the weights, the weights each optimizer set on each rebalancing date.

        Notes:
        - The returns are daily total returns (from adjusted closing prices). The
          estimates use the days on which every asset has a return; the portfolios earn
          the returns of every day, so a day on which one market is closed (its assets
          return 0%, the next day covers both) still counts for the others, up to the
          last day of the data. An asset whose returns end earlier (delisted or
          acquired) ends the estimates and rebalancing on its last return; the portfolios
          then hold it at 0% up to the end, and a warning says so. Each portfolio and the
          benchmark are measured over the days they have a return.
        - Returns on every calendar day (such as those of cryptocurrencies, with returns
          on Saturdays and Sundays) are annualized with 365 days a year, returns on
          trading days with 252, for the estimates and for each portfolio and the
          benchmark on its own.
        - When an optimizer finds no weights on a rebalancing date (e.g. the maximum Sharpe
          ratio when no asset beat the risk free rate over the lookback), the portfolio
          keeps its drifted weights until the next date, and a warning says how often.
          Before its first weights it holds nothing and has no returns, so it starts
          later than the others, and with weight bounds no portfolio meets it has no
          returns at all.
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
        check_option("output", output, ("summary", "returns", "weights"))
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
        periods_per_year = self._get_periods_per_year("daily", returns.index)
        # The portfolios earn every day's return up to the end of the data: a day on
        # which only some markets traded is a 0% return for the others, whose next return
        # covers both days. A day on which none of the assets traded (only the benchmark's
        # market was open) is no day of the portfolios.
        holding_returns = (
            self._historical_data["daily"]["Return"]
            .loc[returns.index[0] : self._end_date, returns.columns]
            .dropna(how="all")
        )
        # An asset whose returns stop early (delisted or acquired) ends the estimates and
        # the rebalancing on its last return, which is worth a warning once a rebalancing
        # date of the data falls after it.
        if any(
            date > returns.index[-1]
            for date in backtest_model.get_rebalancing_dates(
                holding_returns.index, rebalancing, 1
            )
        ):
            ended = [
                asset
                for asset in returns.columns
                if holding_returns[asset].last_valid_index() <= returns.index[-1]
            ]
            logger.warning(
                "%s has no returns after %s, the last day on which every asset has a "
                "return, so the portfolios set no new weights after it and hold %s at a "
                "0%% return up to %s.",
                ", ".join(ended),
                returns.index[-1],
                "them" if len(ended) > 1 else "it",
                holding_returns.index[-1],
            )
        holding_returns = holding_returns.fillna(0)
        factors = self._get_factor_returns("daily") if covariance == "factor" else None
        # The estimates of a window are the same for every method, so they are made once.
        estimates: dict = {}

        estimate = functools.partial(
            backtest_model.get_window_estimates,
            estimates=estimates,
            risk_free_rate=functools.partial(self._get_risk_free_rate, "daily"),
            periods_per_year=periods_per_year,
            covariance=covariance,
            covariance_span=covariance_span,
            expected_returns=expected_returns,
            expected_returns_span=expected_returns_span,
            benchmark=(
                benchmark
                if expected_returns == "capm"
                or covariance == "ledoit_wolf_single_factor"
                else None
            ),
            factors=factors,
        )
        strategies = {
            BACKTEST_METHODS[method]: functools.partial(
                backtest_model.get_strategy_weights,
                method=method,
                generator=np.random.default_rng(RANDOM_BACKTEST_SEED),
                estimate=estimate,
                periods_per_year=periods_per_year,
                minimum_weight=minimum_weight,
                maximum_weight=maximum_weight,
            )
            for method in methods
        }

        net_returns, weights_table, turnover = backtest_model.get_walk_forward_backtest(
            returns,
            strategies,
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

        # Each column is annualized with its own days: 365 a year for a path with
        # returns on weekends (cryptocurrencies) and 252 for one on trading days only.
        summary = backtest_model.get_backtest_statistics(
            net_returns,
            turnover,
            get_observations_per_year(net_returns),
            self._get_risk_free_rate("daily", net_returns.index),
            transaction_cost,
        )
        # What each optimizer's weights expected from their own window, so the gap
        # between the promise and the realised return shows in one table.
        for date in weights_table.index:
            position = returns.index.get_loc(date)
            estimate(returns.iloc[position - lookback + 1 : position + 1])
        summary = summary.join(
            backtest_model.get_in_sample_expectations(weights_table, estimates)
        )
        summary.index.name = "Portfolio"
        return self._round(summary, rounding)

    @handle_errors
    def get_expected_returns_estimate(
        self,
        method: str | list[str] | None = None,
        period: str = "daily",
        span: int = 180,
        adjust: bool = True,
        ranking: list | dict | pd.Series | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate the annualized expected return of every asset with every estimator the
        optimizers take, side by side.

        It shows how much the choice moves the input that mean-variance weights are most
        sensitive to.

        - Mean: the historical average return per period times P, the number of periods
          in a year.
        - Geometric: the compounded average, (Π (1 + r_t))^(P / T) - 1 over the T periods.
        - Exponential: the exponentially weighted average with α = 2 / (span + 1), times
          P.
        - Exponential Geometric: that average compounded, (1 + m)^P - 1.
        - CAPM: r_f + β_i (E[r_m] - r_f) with the benchmark as the market (Sharpe, 1964).
        - Bayes-Stein: the averages shrunk towards the average return of the minimum
          variance portfolio, μ = (1 - φ) μ̂ + φ μ_0 1 with φ = (N + 2) / ((N + 2) +
          T (μ̂ - μ_0 1)'Σ⁻¹(μ̂ - μ_0 1)) and Σ = S (T - 1) / (T - N - 2) (Jorion, 1986).
        - James-Stein: the averages shrunk towards their grand mean μ̄, μ = (1 - β) μ̂ +
          β μ̄ 1 with β = (Σ λ_k - 2 max λ_k) / (T ‖μ̂ - μ̄ 1‖²) clipped to [0, 1] and λ_k the
          eigenvalues of the sample covariance.
        - Bodnar-Okhrin-Parolya: μ = α μ̂ + β b with the grand mean as the target b and
          the weights α and β of Bodnar, Okhrin and Parolya (2019).
        - Centroid: the returns that only an ordering of the assets implies (Almgren and
          Chriss, 2006), μ_i = c_i σ_i with σ_i the annualized volatility and c_i the
          expected value of the i-th largest of N standard normal draws, approximated by
          c_j = Φ⁻¹((N + 1 - j - α) / (N - 2α + 1)) with α = 0.4424 - 0.1185 N^(-0.21);
          assets ranked together share the average c of their positions. It takes the
          ranking, from the highest expected return to the lowest, from anywhere: a view,
          or the scores of any metric of the Toolkit.

        For more information, see the following papers:

        - James, W. and Stein, C. (1961). "Estimation with Quadratic Loss." Proceedings of
          the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1,
          361-379.
        - Jorion, P. (1986). "Bayes-Stein Estimation for Portfolio Analysis." Journal of
          Financial and Quantitative Analysis, 21(3), 279-292.
        - Bodnar, T., Okhrin, O. and Parolya, N. (2019). "Optimal Shrinkage Estimator for
          High-Dimensional Mean Vector." Journal of Multivariate Analysis, 170, 63-79.
        - Almgren, R. and Chriss, N. (2006). "Optimal Portfolios from Ordering
          Information." Journal of Risk, 9(1), 1-47.

        Also known as: shrunk mu, shrinkage expected returns, CAGR, geometric mean return,
        centroid returns, ranking returns, returns from an ordering.

        Referenced against:
        - PyPortfolioOpt, mean_historical_return (geometric average): https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.mean_historical_return
        - PyPortfolioOpt, ema_historical_return (exponentially weighted averages): https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.ema_historical_return
        - PyPortfolioOpt, capm_return (CAPM return): https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.capm_return
        - Riskfolio-Lib, mean_vector (the exponential recursion): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.mean_vector
        - skfolio, ShrunkMu (James-Stein): https://skfolio.org/generated/skfolio.moments.ShrunkMu.html
        - skfolio, ShrunkMu (Bodnar-Okhrin): https://skfolio.org/generated/skfolio.moments.ShrunkMu.html

        Shares similar logic:
        - skfolio, ShrunkMu (Bayes-Stein): https://skfolio.org/generated/skfolio.moments.ShrunkMu.html
        - Riskfolio-Lib, mean_vector (James-Stein): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.mean_vector
        - Riskfolio-Lib, mean_vector (Bodnar-Okhrin-Parolya): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.mean_vector
        - PortfolioAnalytics (R), ac.ranking (centroid): https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/ac_ranking.R#L31
        - PortfolioAnalytics (R), centroid.complete.mc (centroid): https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/ac_ranking.R#L97

        Args:
            method (str | list[str] | None, optional): The estimators. Options are: "mean" (the historical average),
                "geometric" (the compounded average), "exponential" (the exponentially weighted average, recent
                periods weigh most), "exponential_geometric" (that average compounded over a year), "capm" (the
                returns the Capital Asset Pricing Model implies from each asset's beta to the benchmark, Sharpe,
                1964), "bayes_stein" (the averages shrunk towards that of the minimum variance portfolio, Jorion,
                1986), "james_stein" (the averages shrunk towards their grand mean, James and Stein, 1961),
                "bodnar_okhrin_parolya" (the averages and their grand mean combined with optimal weights, Bodnar,
                Okhrin and Parolya, 2019), "centroid" (the returns an ordering of the assets implies, Almgren and
                Chriss, 2006). Defaults to None, all of them (CAPM only when there is a benchmark, Centroid only
                with a ranking).
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            span (int, optional): The span in periods of the exponential averages.
                Defaults to 180.
            adjust (bool, optional): Whether the exponential averages normalize the weights over the whole history
                rather than follow the recursion m_t = α r_t + (1 - α) m_(t-1). Defaults to True.
            ranking (list | dict | pd.Series | None, optional): The ordering of the centroid estimator, from the
                highest expected return to the lowest: a list of every asset (["MSFT", "AAPL", ...]), whose entries
                can be lists of assets ranked together (["MSFT", ["AAPL", "JNJ"], "XOM"]), or a score per asset
                ({"AAPL": 2.1, ...} or a pd.Series, higher is better, equal scores ranked together). Defaults to
                None.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The annualized expected return per asset (rows) and estimator
            (columns), from the periods in which every asset has a return.

        Notes:
        - The geometric and exponential geometric averages are compounded annual returns; the
          others are arithmetic returns per period times P, without compounding.
        - James-Stein needs at least three assets to shrink (with two, its β is negative
          and clipped to 0); skfolio and Riskfolio-Lib leave β unclipped. Bayes-Stein
          uses Jorion's Σ, where skfolio's and Riskfolio-Lib's Bayes-Stein estimators use S.
        - The centroid values follow, within 0.01, the mean of sorted normal draws that
          PortfolioAnalytics' centroid.complete.mc simulates. PortfolioAnalytics takes the order
          from the lowest return up and rescales the centroid to lie between -0.05 and 0.05,
          where here it is scaled by each asset's volatility, so every asset's return over its
          volatility is its centroid value.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_expected_returns_estimate(
            show_columns=["Mean", "Geometric", "Bayes-Stein", "James-Stein", "Bodnar-Okhrin-Parolya"]
        )
        ```

        Which returns:

        |      |   Mean |   Geometric |   Bayes-Stein |   James-Stein |   Bodnar-Okhrin-Parolya |
        |:-----|-------:|------------:|--------------:|--------------:|------------------------:|
        | AAPL | 0.1932 |      0.1672 |        0.1673 |        0.1932 |                  0.1291 |
        | MSFT | 0.2017 |      0.1838 |        0.1695 |        0.2017 |                  0.1222 |
        | JNJ  | 0.0987 |      0.0884 |        0.1418 |        0.0987 |                  0.2051 |
        | XOM  | 0.2911 |      0.2898 |        0.1937 |        0.2911 |                  0.0503 |
        """
        returns = self._get_returns(period, include_benchmark)
        benchmark = self._get_benchmark_returns(period)
        if method is None:
            methods = [
                name
                for name in optimization_model.EXPECTED_RETURN_METHODS
                if (name != "capm" or benchmark is not None)
                and (name != "centroid" or ranking is not None)
            ]
        else:
            methods = [method] if isinstance(method, str) else list(method)
        risk_free_rate = self._get_risk_free_rate(period, returns.index)
        table = pd.DataFrame(
            {
                EXPECTED_RETURN_NAMES.get(
                    name, name
                ): optimization_model.get_expected_returns(
                    returns,
                    self._get_periods_per_year(period, returns.index),
                    name,
                    span,
                    benchmark,
                    risk_free_rate,
                    adjust,
                    ranking,
                )
                for name in methods
            }
        )
        return self._round(table, rounding)

    @handle_errors
    def get_covariance_estimate(
        self,
        method: str = "sample",
        period: str = "daily",
        span: int = 180,
        shrinkage: float = 0.1,
        gerber_threshold: float = 0.5,
        gerber_statistic: int = 1,
        denoise_alpha: float = 0.0,
        components: int = 1,
        factor_selection: str = "ols",
        factors: pd.DataFrame | None = None,
        region: str = "United States",
        half_life: float = 40,
        seed: int | None = 0,
        correlation: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate the annualized covariance matrix of the assets, with one of the estimators the
        optimizers take.

        Its parameters can be set, and it can be shown as a correlation matrix.

        - Sample: the sample covariance (ddof=1).
        - Ledoit-Wolf, Ledoit-Wolf constant correlation, Ledoit-Wolf single factor and Oracle
          Approximating Shrinkage: shrinkage with an estimated intensity (Ledoit and Wolf, 2003,
          2004; Chen et al., 2010).
        - Fixed shrinkage: Σ = (1 - δ) S + δ (tr(S) / N) I with a fixed intensity δ, the
          shrinkage.
        - Semicovariance: the co-movement below a zero return (Estrada, 2008).
        - Exponential: exponentially weighted with α = 2 / (span + 1) (RiskMetrics).
        - Denoised: the eigenvalues of the correlation matrix below the Marčenko-Pastur upper
          edge averaged (constant residual eigenvalue), set to zero (spectral), or their part of
          the correlation shrunk towards its diagonal (targeted shrinkage), C_signal + α C_noise
          + (1 - α) diag(C_noise) with α the denoise alpha (López de Prado, 2020).
        - Gerber: Σ = D G D with G the Gerber statistic of the moves beyond the Gerber threshold
          in standard deviations: statistic 0 (n_c - n_d) / (n_c + n_d), 1 (n_c - n_d) / (T -
          n_n) or 2 H / √(diag(H) diag(H)') (Gerber et al., 2022).
        - Detoned: the first principal components of the correlation matrix, the market, removed
          and the rest rescaled to a unit diagonal (López de Prado, 2020). It is singular, so
          the optimizers do not take it; it suits clustering as the detoned codependence.
        - Factor: Σ = B Σ_f B' + D, the covariance a time series factor model implies, with B
          the loadings of the assets on the factors (the Fama and French five factors by
          default), Σ_f the covariance of the factors and D the variances of the residuals, all
          with ddof=1 (Fama and French, 1993; Connor, 1995). The loadings come from OLS on every
          factor, a stepwise selection or a principal components regression (see
          get_factor_loadings), on the periods both the assets and the factors have.
        - Minimum Covariance Determinant: the covariance of the h = ⌈(T + N + 1) / 2⌉ periods
          whose covariance matrix has the smallest determinant, found with the C-steps of
          FAST-MCD from random subsets (with the seed), made consistent at the normal
          distribution and reweighted: the periods with a Mahalanobis distance below the 97.5%
          quantile of χ²_N are kept and their covariance is multiplied by c = 0.975 /
          F_(N+2)(χ²_N(0.975)) (Rousseeuw and Van Driessen, 1999; Croux and Haesbroeck, 1999).
          Crashes and data errors barely move it.
        - j-LoGo: Σ = J⁻¹ with J = Σ_C [S_CC]⁻¹ - Σ_S [S_SS]⁻¹ the sparse inverse covariance
          over the 4-cliques C and separators S of the Triangulated Maximally Filtered Graph of
          the similarities (1 + ρ) / 2, a planar graph with 3(N - 2) links (Massara et al.,
          2017; Barfuss et al., 2016). With four or fewer assets every pair is linked and it is
          the sample covariance.
        - Geodesic: Σ = S^(1/2) (S^(-1/2) T S^(-1/2))^δ S^(1/2) with T = (tr(S) / N) I and δ the
          shrinkage, which keeps the eigenvectors of S and interpolates its eigenvalues
          geometrically, λ^(1 - δ) (tr(S) / N)^δ (Bhatia, 2007; Musolas et al., 2021).
        - Regime adjusted exponentially weighted: φ² S_T with S_t = λ S_(t-1) + (1 - λ) r_t r_t'
          (around zero, bias-corrected, λ = 2^(-1 / half_life)) and φ the exponential average
          (half-life half_life / 2) of |w' r_t| / √(w' S_(t-1) w) / √(2 / π) for the inverse
          volatility portfolio w, clipped to [0.7, 1.6] (Paleologo, 2025).
        - Denoised and detoned: the denoised correlation with its first principal components
          removed and rescaled to a unit diagonal, the order of López de Prado (2020, section
          2.6). Singular, like the detoned estimate.
        - NERCOME: Σ = (1 / M) Σ_s P_1 diag(P_1' S_2 P_1) P_1' over M = 50 random splits of the
          periods into a first part of m and the rest, with P_1 the eigenvectors of the first
          part's sample covariance and S_2 the second part's: eigenvalues that are not fitted to
          the same noise as the eigenvectors (Abadir, Distaso and Žikeš, 2014; Lam, 2016). m is
          chosen from 2√T, 0.2T, ..., 0.8T, T - 2.5√T and T - 1.5√T by the smallest ‖avg(Σ_m -
          S_2)‖² (Lam, 2016), and the splits drawn with the seed.
        - CCC-GARCH: the next-period covariance of the constant conditional correlation model of
          Bollerslev (1990), Σ_T+1 = D R D with D the GARCH(1, 1) volatilities forecast for the
          next period, σ²_i,T+1 = ω_i + α_i ε²_i,T + β_i σ²_i,T, and R the correlation of the
          standardized residuals ε_t / σ_t of the returns around their average. It follows the
          volatility now, not its average over the history.
        - MM-estimator: robust to almost half the periods being outliers and 95% as efficient in
          its shape (the correlations and variance ratios) as the sample covariance for normal
          returns: an S-estimator with Tukey's bisquare ρ(d) = 1 - (1 - (d / c)²)³ (the scatter
          with the smallest scale s solving avg ρ(d_t / √s) = (1 - N / T) / 2), followed by a
          bisquare M-step with a larger c at that scale (Tatsuoka and Tyler, 2000;
          Salibián-Barrera, Van Aelst and Willems, 2006). Both are iteratively reweighted means
          and covariances, with weights that fall to zero for periods far from the bulk.
        - Rocke's S-estimator: the translated biweight of Rocke (1996) on the squared distances,
          ρ(x) = ((x - 1) / (4γ)) (3 - ((x - 1) / γ)²) + ½ within γ of 1 (0 below, 1 above),
          with γ = min(χ²_N(0.995) / N - 1, 1), made consistent at the normal distribution
          through the median distance, which keeps its robustness for many assets (Maronna,
          Martin, Yohai and Salibián-Barrera, 2019).

        For more information, see the following papers:

        - Ledoit, O. and Wolf, M. (2004). "A Well-Conditioned Estimator for
          Large-Dimensional Covariance Matrices." Journal of Multivariate Analysis, 88(2),
          365-411.
        - Gerber, S., Markowitz, H. M., Ernst, P. A., Miao, Y., Javid, B. and Sargen, P.
          (2022). "The Gerber Statistic: A Robust Co-Movement Measure for Portfolio
          Optimization." The Journal of Portfolio Management, 48(3), 87-102.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge
          University Press, chapter 2.
        - Fama, E. F. and French, K. R. (1993). "Common Risk Factors in the Returns on
          Stocks and Bonds." Journal of Financial Economics, 33(1), 3-56.
        - Connor, G. (1995). "The Three Types of Factor Models: A Comparison of Their
          Explanatory Power." Financial Analysts Journal, 51(3), 42-46.
        - Rousseeuw, P. J. and Van Driessen, K. (1999). "A Fast Algorithm for the Minimum
          Covariance Determinant Estimator." Technometrics, 41(3), 212-223.
        - Croux, C. and Haesbroeck, G. (1999). "Influence Function and Efficiency of the
          Minimum Covariance Determinant Scatter Matrix Estimator." Journal of
          Multivariate Analysis, 71(2), 161-190.
        - Massara, G. P., Di Matteo, T. and Aste, T. (2017). "Network Filtering for Big
          Data: Triangulated Maximally Filtered Graph." Journal of Complex Networks, 5(2),
          161-178.
        - Barfuss, W., Massara, G. P., Di Matteo, T. and Aste, T. (2016). "Parsimonious
          Modeling with Information Filtering Networks." Physical Review E, 94(6), 062306.
        - Bhatia, R. (2007). "Positive Definite Matrices." Princeton University Press.
        - Musolas, A., Smith, S. T. and Marzouk, Y. (2021). "Geodesically Parameterized
          Covariance Estimation." SIAM Journal on Matrix Analysis and Applications, 42(2),
          528-556.
        - Paleologo, G. A. (2025). "The Elements of Quantitative Investing." Wiley Finance.
        - Abadir, K. M., Distaso, W. and Žikeš, F. (2014). "Design-Free Estimation of
          Variance Matrices." Journal of Econometrics, 181(2), 165-180.
        - Lam, C. (2016). "Nonparametric Eigenvalue-Regularized Precision or Covariance
          Matrix Estimator." The Annals of Statistics, 44(3), 928-953.
        - Bollerslev, T. (1990). "Modelling the Coherence in Short-Run Nominal Exchange
          Rates: A Multivariate Generalized ARCH Model." The Review of Economics and
          Statistics, 72(3), 498-505.
        - Rocke, D. M. (1996). "Robustness Properties of S-Estimators of Multivariate
          Location and Shape in High Dimension." The Annals of Statistics, 24(3),
          1327-1345.
        - Tatsuoka, K. S. and Tyler, D. E. (2000). "On the Uniqueness of S-Functionals and
          M-Functionals under Nonelliptical Distributions." The Annals of Statistics,
          28(4), 1219-1243.
        - Salibián-Barrera, M., Van Aelst, S. and Willems, G. (2006). "Principal Components
          Analysis Based on Multivariate MM Estimators with Fast and Robust Bootstrap."
          Journal of the American Statistical Association, 101(475), 1198-1211.
        - Maronna, R. A., Martin, R. D., Yohai, V. J. and Salibián-Barrera, M. (2019).
          "Robust Statistics: Theory and Methods (with R)." 2nd edition, Wiley.

        Also known as: covariance estimator, risk model, shrunk covariance, Gerber
        covariance, denoised covariance, detoned correlation, factor covariance, factor
        risk model, robust covariance, MCD, FastMCD, TMFG, LoGo, sparse inverse covariance,
        geodesic shrinkage, regime adjusted EWMA covariance, short-term volatility update,
        NERCOME, split-sample covariance, CCC-GARCH, constant conditional correlation,
        MM-estimator, S-estimator, Rocke estimator, translated biweight.

        Referenced against:
        - scikit-learn, LedoitWolf: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
        - scikit-learn, OAS: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
        - PyPortfolioOpt, CovarianceShrinkage.shrunk_covariance: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.shrunk_covariance
        - PyPortfolioOpt, semicovariance: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.semicovariance
        - PyPortfolioOpt, exp_cov: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.exp_cov
        - skfolio, DenoiseCovariance (constant residual eigenvalue): https://skfolio.org/generated/skfolio.moments.DenoiseCovariance.html
        - Riskfolio-Lib, denoiseCov (spectral and targeted shrinkage): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.denoiseCov
        - skfolio, GerberCovariance (statistics 1 and 2): https://skfolio.org/generated/skfolio.moments.GerberCovariance.html
        - Riskfolio-Lib, gerber_cov_stat0 (statistic 0): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#GerberStatistic.gerber_cov_stat0
        - skfolio, DetoneCovariance (sample covariance): https://skfolio.org/generated/skfolio.moments.DetoneCovariance.html
        - Riskfolio-Lib, risk_factors (factor): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.risk_factors
        - Riskfolio-Lib, loadings_matrix (backward selection, every factor kept): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.loadings_matrix
        - scikit-learn, MinCovDet: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
        - Riskfolio-Lib, covar_matrix (j-LoGo): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.covar_matrix
        - skfolio, GeodesicShrinkageCovariance: https://skfolio.org/generated/skfolio.moments.GeodesicShrinkageCovariance.html
        - skfolio, RegimeAdjustedEWCovariance (its defaults): https://skfolio.org/generated/skfolio.moments.RegimeAdjustedEWCovariance.html

        Shares similar logic:
        - PyPortfolioOpt, CovarianceShrinkage.ledoit_wolf: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf
        - PyPortfolioOpt, CovarianceShrinkage.oracle_approximating: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.oracle_approximating
        - scikit-learn, ShrunkCovariance: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html
        - skfolio, ShrunkCovariance: https://skfolio.org/generated/skfolio.moments.ShrunkCovariance.html
        - Riskfolio-Lib, covar_matrix (fixed shrinkage): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.covar_matrix
        - RiskLabAI, denoise_cov: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/denoising.py#L376
        - Riskfolio-Lib, gerber_cov_stat1 (statistic 1): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#GerberStatistic.gerber_cov_stat1
        - Riskfolio-Lib, gerber_cov_stat2 (statistic 2): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#GerberStatistic.gerber_cov_stat2
        - skfolio, FactorModel (factor): https://skfolio.org/generated/skfolio.prior.FactorModel.html
        - RiskLabAI, nercome_denoised_covariance: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/nercome.py#L54
        - PortfolioAnalytics (R), CCCgarch.MM: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/moment.functions.R#L22
        - PortfolioAnalytics (R), custom.covRob.MM: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/custom.covRob.R#L29
        - PortfolioAnalytics (R), custom.covRob.Rocke: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/custom.covRob.R#L57

        Args:
            method (str, optional): The estimator. Options are: "sample" (the sample covariance, ddof=1),
                "ledoit_wolf" (shrunk towards the scaled identity, Ledoit and Wolf, 2004),
                "ledoit_wolf_constant_correlation" (shrunk towards the average correlation, Ledoit and Wolf, 2004),
                "ledoit_wolf_single_factor" (shrunk towards the covariance that the benchmark as a single factor
                implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating Shrinkage, Chen et al., 2010), "shrunk"
                (a fixed 10% towards the scaled identity), "semicovariance" (the co-movement below a zero return,
                Estrada, 2008), "exponential" (exponentially weighted, recent periods weigh most, RiskMetrics),
                "denoised" (the eigenvalues of the correlation matrix that random returns could produce averaged,
                Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to zero), "denoised_shrink" (the
                noise correlations shrunk towards their diagonal, López de Prado, 2020), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "detoned" (the correlation without its first
                principal components, López de Prado, 2020), "factor" (the covariance a time series factor model
                implies, Connor, 1995), "minimum_covariance_determinant" (the reweighted covariance of the half of
                the periods with the smallest covariance determinant, robust to outliers, Rousseeuw and Van
                Driessen, 1999), "jlogo" (the sparse inverse covariance on the TMFG planar graph of the
                correlations, Barfuss et al., 2016), "geodesic" (10% towards the scaled identity along the geodesic
                of the positive definite matrices, Bhatia, 2007), "regime_adjusted_ew" (exponentially weighted with
                a half-life of 40 periods, rescaled by the recent realized against predicted risk, Paleologo, 2025),
                "denoised_detoned" (the denoised correlation with its first principal components removed, López de
                Prado, 2020), "nercome" (eigenvectors from one part of the periods and the variances along them from
                the other, averaged over random splits, Lam, 2016), "ccc_garch" (the next-period covariance of
                GARCH(1, 1) volatilities with constant correlations, Bollerslev, 1990), "mm" (the robust
                MM-estimator, Maronna et al., 2019), "rocke" (Rocke's robust S-estimator, Maronna et al., 2019).
                Defaults to "sample".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            span (int, optional): The span in periods of the exponentially weighted estimate. Defaults to 180.
            shrinkage (float, optional): The fixed intensity δ of the fixed and geodesic shrinkage, between 0 and 1.
                Defaults to 0.1 (PyPortfolioOpt defaults to 0.2 for the fixed shrinkage).
            gerber_threshold (float, optional): The threshold of the Gerber estimate in standard deviations, between
                0 and 1. Defaults to 0.5.
            gerber_statistic (int, optional): The Gerber statistic. Options are: 0 (the original statistic), 1 (the
                positive semidefinite variant), 2 (the matrix form). Defaults to 1.
            denoise_alpha (float, optional): The share α of the noise correlations the targeted shrinkage keeps,
                between 0 and 1. Defaults to 0.
            components (int, optional): The number of principal components the detoned estimates remove. Defaults to
                1.
            factor_selection (str, optional): How the factor estimate takes the loadings. Options are: "ols" (every
                factor), "forward" (stepwise by the p-values, 0.05), "backward" (stepwise by the p-values, 0.05),
                "pcr" (principal components regression keeping 95% of the variance). Defaults to "ols".
            factors (pd.DataFrame | None, optional): The factor returns of the factor estimate, one column per
                factor, at the frequency of the period. Defaults to None, the Fama and French five factors of the
                region.
            region (str, optional): The region of the Fama and French factors. Options are: "United States",
                "Developed", "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan", "North America".
                Defaults to "United States".
            half_life (float, optional): The half-life in periods of the exponential weights of the regime adjusted
                estimate. Defaults to 40, skfolio's default as well.
            seed (int | None, optional): The seed of the random subsets of the Minimum Covariance Determinant, which
                also starts the MM and Rocke estimators, and of the random splits of NERCOME (numpy's default
                generator), so the same returns give the same estimate. Defaults to 0.
            correlation (bool, optional): Whether to return the correlation matrix of
                the estimate instead. Defaults to False.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The annualized covariance matrix (the covariance per period
            times the number of periods in a year), or the correlation matrix, from the
            periods in which every asset has a return.

        Notes:
        - The Gerber thresholds and D use standard deviations with ddof=1, where skfolio
          and Riskfolio-Lib use ddof=0; their result equals this one with the threshold
          multiplied by √(T / (T - 1)) and the covariance by T / (T - 1).
        - The fixed shrinkage shrinks the sample covariance with ddof=1 (as PyPortfolioOpt);
          skfolio, scikit-learn and Riskfolio-Lib shrink the one divided by T.
        - The Minimum Covariance Determinant divides by the number of periods kept (ddof=0),
          as in scikit-learn, since the consistency factors assume it. Its random subsets
          differ from scikit-learn's, so the two can end in different local minima of the
          determinant, a few percent apart on small samples. PyPortfolioOpt's
          min_cov_determinant returns the raw estimate (scikit-learn's fast_mcd), without
          the consistency factor and the reweighting.
        - The regime adjusted estimate takes the returns around zero, as skfolio does by
          default, and uses the whole history, where the exponential estimate takes a span of
          180 periods.
        - NERCOME estimates the covariance itself, as Lam (2016) does; RiskLabAI's
          nercome_denoised_covariance splits at 2T / 3 and estimates the correlation of the
          standardized returns, so its variances are the sample ones.
        - CCC-GARCH fits every GARCH(1, 1) by Gaussian likelihood from the arch package's
          backcast (see get_garch); PortfolioAnalytics' CCCgarch.MM uses fGarch and a
          constant volatility for an asset whose α is below 0.01. A GARCH fit takes a few
          seconds per asset on daily returns.
        - The MM and Rocke estimators start from the minimum covariance determinant (with the
          seed) and from the sample covariance and keep the smaller S scale; RobStatTM's
          covRobMM and covRobRocke, which PortfolioAnalytics calls, start from the
          kurtosis-based estimate of Peña and Prieto (2007), so on small samples the two can end
          in different local minima. Rocke's estimator suits ten or more assets; with fewer, γ =
          1 and its weights are low for many ordinary periods, so the MM-estimator is the more
          efficient there (Maronna et al., 2019).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_covariance_estimate(method="gerber")
        ```

        Which returns:

        |      |   AAPL |   MSFT |    JNJ |    XOM |
        |:-----|-------:|-------:|-------:|-------:|
        | AAPL | 0.0776 | 0.0288 | 0.0044 | 0.0054 |
        | MSFT | 0.0288 | 0.0661 | 0.0021 | 0.0015 |
        | JNJ  | 0.0044 | 0.0021 | 0.028  | 0.0041 |
        | XOM  | 0.0054 | 0.0015 | 0.0041 | 0.073  |
        """
        # Checked whichever method is chosen, as only the factor estimate uses it.
        check_option(
            "factor_selection", factor_selection, estimation_model.FACTOR_SELECTIONS
        )
        returns = self._get_returns(period, include_benchmark)
        if method == "detoned":
            matrix = estimation_model.get_detoned_covariance(returns, components)
        elif method == "denoised_detoned":
            matrix = estimation_model.get_detoned_covariance(
                returns,
                components,
                optimization_model.get_denoised_covariance(returns)[0],
            )
        elif method == "factor":
            returns, factor_returns = self._align_factors(
                returns, self._get_factor_returns(period, factors, region)
            )
            matrix = estimation_model.get_factor_model(
                returns,
                factor_returns,
                estimation_model.get_factor_loadings(
                    returns, factor_returns, factor_selection
                ),
            )[1]
        else:
            matrix = optimization_model.get_covariance_matrix(
                returns,
                method,
                span,
                self._get_benchmark_returns(period),
                shrinkage,
                gerber_threshold,
                gerber_statistic,
                denoise_alpha,
                half_life=half_life,
                seed=seed,
            )
        if correlation:
            volatility = np.sqrt(np.diag(matrix.to_numpy()))
            return self._round(matrix / np.outer(volatility, volatility), rounding)
        return self._round(
            matrix * self._get_periods_per_year(period, returns.index), rounding
        )

    @handle_errors
    def get_random_matrix_statistics(
        self,
        output: str = "summary",
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit the Marčenko-Pastur distribution to the eigenvalues of the correlation matrix of the
        assets, to see how much of it random returns alone could produce.

        The eigenvalues of the correlation matrix of T periods of N unrelated returns with
        variance σ² fall between the edges λ- and λ+ of the Marčenko-Pastur distribution; an
        eigenvalue above λ+ is signal (the market, sectors, styles), the ones below it are
        indistinguishable from noise and are what the denoised covariance estimators average,
        remove or shrink (López de Prado, 2020, chapter 2).

        The formula is as follows:

        - f(λ) = q / (2π σ² λ) √((λ+ - λ)(λ - λ-)) for λ- ≤ λ ≤ λ+, with q = T / N
        - λ± = σ² (1 ± √(1 / q))²
        - σ² minimizes Σ_x (f(x) - k(x))² over 1,000 points x between λ- and λ+, with k a
          Gaussian kernel density of the eigenvalues (bandwidth 0.01), from σ² = 0.5 and
          between 0.00001 and 0.99999 (López de Prado, 2020, section 2.4)
        - signal eigenvalues: λ_k > λ+

        For more information, see the following sources:

        - Marčenko, V. A. and Pastur, L. A. (1967). "Distribution of Eigenvalues for Some
          Sets of Random Matrices." Mathematics of the USSR-Sbornik, 1(4), 457-483.
        - Laloux, L., Cizeau, P., Potters, M. and Bouchaud, J.-P. (2000). "Random Matrix
          Theory and Financial Correlations." International Journal of Theoretical and
          Applied Finance, 3(3), 391-397.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge
          University Press, chapter 2.

        Also known as: Marchenko-Pastur, Marcenko-Pastur fit, random matrix theory, RMT,
        eigenvalue spectrum, noise eigenvalues, signal eigenvalues.

        Shares similar logic:
        - RiskLabAI, find_max_eval: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/denoising.py#L138
        - RiskLabAI, marcenko_pastur_pdf: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/denoising.py#L26

        Args:
            output (str, optional): Options are: "summary" (one row: the Periods, Assets, Noise Variance σ², Lower
                Edge λ-, Upper Edge λ+, Largest Eigenvalue, the number of Signal Eigenvalues and their Signal Share
                of the trace N), "eigenvalues" (every eigenvalue from the largest, its Explained Variance λ / N and
                whether it is Signal). Defaults to "summary".
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
            pd.DataFrame: The summary of the fit or the table of eigenvalues, from the
            periods in which every asset has a return.

        Notes:
        - The fit is the one the denoised covariance estimators use. RiskLabAI's find_max_eval
          fits the same density with scikit-learn's KernelDensity at as many points as there are
          eigenvalues rather than the book's 1,000, which moves σ² by a few percent for 30
          assets. With few assets the eigenvalues are too few to pin the density down, so σ² is
          a rough guide there; the distribution describes many assets.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM", "JPM", "PG", "KO", "NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_random_matrix_statistics(output="eigenvalues")
        ```

        Which returns:

        |              |   Eigenvalue |   Explained Variance | Signal   |
        |:-------------|-------------:|---------------------:|:---------|
        | Eigenvalue 1 |       2.8514 |               0.3564 | True     |
        | Eigenvalue 2 |       1.7917 |               0.224  | True     |
        | Eigenvalue 3 |       1.0929 |               0.1366 | False    |
        | Eigenvalue 4 |       0.6159 |               0.077  | False    |
        | Eigenvalue 5 |       0.5604 |               0.0701 | False    |
        | Eigenvalue 6 |       0.4081 |               0.051  | False    |
        | Eigenvalue 7 |       0.3617 |               0.0452 | False    |
        | Eigenvalue 8 |       0.3178 |               0.0397 | False    |
        """
        outputs = ("summary", "eigenvalues")
        check_option("output", output, outputs)
        returns = self._get_returns(period, include_benchmark)
        periods, assets = returns.shape
        eigenvalues = np.linalg.eigvalsh(returns.corr().to_numpy())[::-1]
        variance, lower, upper = optimization_model.get_marcenko_pastur_fit(
            eigenvalues, periods / assets
        )
        signal = eigenvalues > upper
        if output == "eigenvalues":
            table = pd.DataFrame(
                {
                    "Eigenvalue": eigenvalues,
                    "Explained Variance": eigenvalues / assets,
                    "Signal": signal,
                },
                index=[f"Eigenvalue {number}" for number in range(1, assets + 1)],
            )
        else:
            table = pd.DataFrame(
                {
                    "Periods": periods,
                    "Assets": assets,
                    "Noise Variance": variance,
                    "Lower Edge": lower,
                    "Upper Edge": upper,
                    "Largest Eigenvalue": eigenvalues[0],
                    "Signal Eigenvalues": int(signal.sum()),
                    "Signal Share": float(eigenvalues[signal].sum() / assets),
                },
                index=["Marčenko-Pastur"],
            )
        return self._round(table, rounding)

    @handle_errors
    def get_codependence_matrix(
        self,
        method: Literal[
            "pearson",
            "spearman",
            "kendall",
            "gerber",
            "detoned",
            "distance",
            "mutual_information",
            "lower_tail",
            "absolute_angular",
            "squared_angular",
            "gnpr",
        ] = "pearson",
        period: str = "daily",
        covariance: str = "sample",
        distance: bool = False,
        gerber_threshold: float = 0.5,
        gerber_statistic: int = 1,
        components: int = 1,
        bins: str | int = "freedman",
        alpha: float = 0.05,
        gnpr_theta: float = 0.5,
        gnpr_bandwidth: float = 0.01,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how much every pair of assets moves together.

        It is the codependence that the hierarchical optimizers (HRP, HERC, NCO) and
        get_asset_clusters take as their codependence, or the distance they cluster on.

        - Pearson: the correlation of the covariance estimate, distance √((1 - ρ) / 2).
        - Spearman and Kendall: the rank correlations of the returns, distance √((1 - ρ) / 2).
        - Gerber: the Gerber statistic (see get_covariance_estimate), distance √((1 - ρ) / 2).
        - Detoned: the correlation without its first principal components, distance √((1 - ρ) /
          2).
        - Distance correlation: dCor = √(dCov²(x, y) / √(dCov²(x, x) dCov²(y, y))) from the
          doubly centred distances |x_s - x_t| (Székely, Rizzo and Bakirov, 2007), zero only for
          independent returns; distance √(1 - dCor).
        - Mutual information: I(X, Y) / min(H(X), H(Y)) from histograms with the given bins
          (natural logarithms); distance the variation of information (H(X) + H(Y) - 2 I(X, Y))
          / H(X, Y) (Meilă, 2007).
        - Lower tail dependence: P(r_j ≤ q_j(α) | r_i ≤ q_i(α)); distance -ln λ (λ at least
          1e-8).
        - Absolute angular: |ρ| of the Pearson correlation, distance √(1 - |ρ|) (López de Prado,
          2020, chapter 3), and squared angular: ρ², distance √(1 - ρ²), the same construction
          on the squared correlation, which put assets that move strongly against each other
          close together, as hedges of one another.
        - GNPR: the Generic Non-Parametric Representation distance of Marti et al. (2016), d² =
          θ d_1² + (1 - θ) d_0² with θ the GNPR theta, d_1² = 3 / (T (T² - 1)) Σ_t (R_i,t -
          R_j,t)² of the ranks R of the returns ((1 - ρ_S) / 2 without ties) and d_0² = ½ Σ_b
          (√g_i,b - √g_j,b)² the squared Hellinger distance between the histograms g of the
          returns, with bins the GNPR bandwidth wide; codependence 1 - d². θ = 1 compares only
          how the assets move together, θ = 0 only how their returns are distributed.

        For more information, see the following papers:

        - Mantegna, R. N. (1999). "Hierarchical Structure in Financial Markets." The
          European Physical Journal B, 11(1), 193-197.
        - Székely, G. J., Rizzo, M. L. and Bakirov, N. K. (2007). "Measuring and Testing
          Dependence by Correlation of Distances." The Annals of Statistics, 35(6),
          2769-2794.
        - Meilă, M. (2007). "Comparing Clusterings - An Information Based Distance."
          Journal of Multivariate Analysis, 98(5), 873-895.
        - Marti, G., Andler, S., Nielsen, F. and Donnat, P. (2016). "Clustering Financial
          Time Series: How Long Is Enough?" Proceedings of the Twenty-Fifth International
          Joint Conference on Artificial Intelligence (IJCAI-16), 2583-2589.
        - Hacine-Gharbi, A. and Ravier, P. (2018). "A Binning Formula of Bi-histogram for
          Joint Entropy Estimation Using Mean Square Error Minimization." Pattern
          Recognition Letters, 101, 21-28.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge
          University Press, chapter 3.

        Also known as: codependence, distance correlation, mutual information, variation
        of information, tail dependence matrix, correlation distance, angular distance,
        absolute angular distance, squared angular distance, GNPR, copula distance.

        Referenced against:
        - skfolio, DistanceCorrelation: https://skfolio.org/generated/skfolio.distance.DistanceCorrelation.html
        - skfolio, MutualInformation: https://skfolio.org/generated/skfolio.distance.MutualInformation.html
        - Riskfolio-Lib, mutual_info_matrix (Hacine-Gharbi bins): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.mutual_info_matrix
        - Riskfolio-Lib, ltdi_matrix: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.ltdi_matrix

        Shares similar logic:
        - Riskfolio-Lib, codep_dist: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.codep_dist
        - skfolio, CovarianceDistance (Pearson correlation): https://skfolio.org/generated/skfolio.distance.CovarianceDistance.html
        - skfolio, DistanceCorrelation: https://skfolio.org/generated/skfolio.distance.DistanceCorrelation.html
        - skfolio, MutualInformation: https://skfolio.org/generated/skfolio.distance.MutualInformation.html
        - Riskfolio-Lib, dcorr_matrix: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.dcorr_matrix
        - Riskfolio-Lib, var_info_matrix: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.var_info_matrix

        Args:
            method (str, optional): The codependence. Options are: "pearson" (Pearson correlation), "spearman"
                (Spearman correlation), "kendall" (Kendall correlation), "gerber" (Gerber statistic), "detoned"
                (detoned correlation), "distance" (distance correlation), "mutual_information" (mutual information),
                "lower_tail" (lower tail dependence), "absolute_angular" (absolute angular codependence),
                "squared_angular" (squared angular codependence), "gnpr" (GNPR codependence). Defaults to "pearson".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The covariance estimator whose correlation the Pearson codependence is (see
                get_covariance_estimate). Options are: "sample" (sample covariance), "ledoit_wolf" (Ledoit-Wolf),
                "ledoit_wolf_constant_correlation" (Ledoit-Wolf constant correlation), "ledoit_wolf_single_factor"
                (Ledoit-Wolf single factor), "oas" (Oracle Approximating Shrinkage), "shrunk" (fixed shrinkage),
                "semicovariance" (semicovariance), "exponential" (exponentially weighted), "denoised" (denoised,
                constant residual eigenvalue), "denoised_spectral" (denoised, spectral), "denoised_shrink"
                (denoised, targeted shrinkage), "gerber" (Gerber), "factor" (factor model),
                "minimum_covariance_determinant" (Minimum Covariance Determinant), "jlogo" (j-LoGo), "geodesic"
                (geodesic shrinkage), "regime_adjusted_ew" (regime adjusted exponentially weighted), "nercome"
                (NERCOME), "ccc_garch" (CCC-GARCH), "mm" (MM-estimator), "rocke" (Rocke's S-estimator). Defaults to
                "sample".
            distance (bool, optional): Whether to return the distance matrix the
                clustering works on instead. Defaults to False.
            gerber_threshold (float, optional): The threshold of the Gerber statistic in standard deviations.
                Defaults to 0.5.
            gerber_statistic (int, optional): The Gerber statistic. Options are: 0 (the original statistic), 1 (the
                positive semidefinite variant), 2 (the matrix form). Defaults to 1.
            components (int, optional): The number of principal components the detoned correlation removes. Defaults
                to 1.
            bins (str | int, optional): The number of histogram bins of the mutual information, or the rule per
                pair. Options are: "freedman" (Freedman and Diaconis, skfolio's default), "hacine_gharbi"
                (Hacine-Gharbi, López de Prado's choice), or an integer. Defaults to "freedman".
            alpha (float, optional): The share of the lowest returns that forms the tail of the lower tail
                dependence. Defaults to 0.05.
            gnpr_theta (float, optional): The weight θ of the co-movement in the GNPR distance, between 0 and 1.
                Defaults to 0.5.
            gnpr_bandwidth (float, optional): The width of the histogram bins of the GNPR distance, in returns per
                period. Defaults to 0.01.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The codependence (or distance) matrix of the assets, from the
            periods in which every asset has a return.

        Notes:
        - The distance correlation builds a matrix of T × T distances per asset, so it
          takes longer on many daily periods.
        - Riskfolio-Lib's mutual information defaults to Knuth's bins, skfolio's to
          Freedman-Diaconis; with the same bins, both equal this one.
        - The absolute and squared angular distances take the whole of 1 - |ρ| and 1 - ρ²;
          half of them, as the correlation distance takes of 1 - ρ, gives a distance √2 times
          smaller with the same clusters.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_codependence_matrix(method="distance")
        ```

        Which returns:

        |      |   AAPL |   MSFT |    JNJ |    XOM |
        |:-----|-------:|-------:|-------:|-------:|
        | AAPL | 1      | 0.5974 | 0.1659 | 0.1473 |
        | MSFT | 0.5974 | 1      | 0.1173 | 0.0876 |
        | JNJ  | 0.1659 | 0.1173 | 1      | 0.1513 |
        | XOM  | 0.1473 | 0.0876 | 0.1513 | 1      |
        """
        # Checked whichever method is chosen, as only some of them use them.
        validate_alpha(alpha)
        if isinstance(bins, str):
            check_option("bins", bins, estimation_model.BIN_METHODS)
        returns = self._get_returns(period, include_benchmark)
        covariance_matrix = optimization_model.get_covariance_matrix(
            returns,
            covariance,
            benchmark_returns=self._get_benchmark_returns(period),
            factors=(
                self._get_factor_returns(period) if covariance == "factor" else None
            ),
        )
        function = (
            optimization_model.get_codependence_distance
            if distance
            else optimization_model.get_codependence
        )
        matrix = function(
            returns,
            covariance_matrix,
            method,
            gerber_threshold,
            gerber_statistic,
            components,
            bins,
            alpha,
            gnpr_theta,
            gnpr_bandwidth,
        )
        return self._round(matrix, rounding)

    @handle_errors
    def get_asset_clusters(
        self,
        method: str = "hierarchical",
        period: str = "daily",
        codependence: str = "pearson",
        covariance: str = "sample",
        linkage_method: str = "ward",
        clusters: int | None = None,
        cluster_criterion: str = "two_difference_gap",
        threshold: float = 0.5,
        seed: int | None = 0,
        include_benchmark: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Group the assets into clusters of assets that move together, the clusters that
        HERC and NCO allocate across.

        - Hierarchical clustering: a hierarchical clustering on the distance of the codependence
          (see get_codependence_matrix), cut into the given number of clusters or, without a
          number, at the sharpest bend of the within-cluster dispersion (the two-order
          difference of the gap statistic, Yue, Wang and Wei, 2008), as HERC and NCO do,
          or at the highest standardized silhouette (López de Prado, 2020).
        - FTCA: the Fast Threshold Clustering Algorithm of Varadi (2013). Among the
          assets not yet clustered, take the one with the highest average codependence
          (H) and the lowest (L). If H and L are codependent above the threshold, they form
          a cluster with every remaining asset whose average codependence with H and L is
          above it; otherwise H forms a cluster with the remaining assets above the
          threshold with H, and then L with those above it with L. This repeats until
          every asset belongs to a cluster. It needs no number of clusters.
        - ONC: the Optimal Number of Clusters algorithm of López de Prado and Lewis
          (2019). Every asset is the point given by its row of the distance matrix. For
          k = 2 to N - 1 clusters, k-means runs ten times (from k-means++ starts drawn with
          the seed), and the clustering with the highest quality E[S] / √V[S] of the
          silhouette coefficients S_i = (b_i - a_i) / max(a_i, b_i) is kept. The clusters
          of below-average quality are then clustered again on their own, and their new
          clusters kept when they raise the average quality above that of the clusters
          they replace (López de Prado, 2020, snippets 4.1 and 4.2). It finds the number of
          clusters itself, needs no tree and suits uneven groups of assets.

        For more information, see the following papers:

        - Varadi, D. (2013). "The Fast Threshold Clustering Algorithm (FTCA)." CSS
          Analytics.
        - López de Prado, M. and Lewis, M. J. (2019). "Detection of False Investment
          Strategies Using Unsupervised Learning Methods." Quantitative Finance, 19(9),
          1555-1565.
        - Yue, S., Wang, X. and Wei, M. (2008). "Application of Two-Order Difference to
          Gap Statistic." Transactions of Tianjin University, 14(3), 217-221.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge
          University Press.

        Also known as: asset clustering, FTCA, correlation clusters, ONC, optimal number of
        clusters, k-means clustering.

        Referenced against:
        - skfolio, HierarchicalClustering (hierarchical): https://skfolio.org/generated/skfolio.cluster.HierarchicalClustering.html
        - ffn, calc_ftca: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L2297

        Shares similar logic:
        - RiskLabAI, cluster_k_means_top (López de Prado, Machine Learning for Asset Managers, snippet 4.2): https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/cluster/clustering.py#L194

        Args:
            method (str, optional): Options are: "hierarchical" (hierarchical clustering), "ftca" (Fast Threshold
                Clustering Algorithm), "onc" (Optimal Number of Clusters). Defaults to "hierarchical".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            codependence (str, optional): The codependence the assets are clustered on (see
                get_codependence_matrix). Defaults to. Options are: "pearson" (the linear correlation of the
                covariance matrix), "spearman" (the rank correlation of the returns, Spearman, 1904), "kendall" (the
                rank correlation τ_b of the returns, Kendall, 1938), "gerber" (co-movements beyond half a standard
                deviation, Gerber et al., 2022), "detoned" (the correlation without the market component, López de
                Prado, 2020), "distance" (the distance correlation, Székely et al., 2007, through √(1 - dCor)),
                "mutual_information" (through the variation of information, Meilă, 2007), "lower_tail" (how often
                two assets fall into their lowest 5% of returns together, through -ln λ), "absolute_angular" (|ρ|
                through √(1 - |ρ|), López de Prado, 2020), "squared_angular" (ρ² through √(1 - ρ²), López de Prado,
                2020), "gnpr" (the rank co-movement and the difference between the return distributions, θ = 0.5,
                Marti et al., 2016). Defaults to "pearson".
            covariance (str, optional): The covariance estimator whose correlation the Pearson codependence is.
                Options are: "sample" (sample covariance), "ledoit_wolf" (Ledoit-Wolf),
                "ledoit_wolf_constant_correlation" (Ledoit-Wolf constant correlation), "ledoit_wolf_single_factor"
                (Ledoit-Wolf single factor), "oas" (Oracle Approximating Shrinkage), "shrunk" (fixed shrinkage),
                "semicovariance" (semicovariance), "exponential" (exponentially weighted), "denoised" (denoised,
                constant residual eigenvalue), "denoised_spectral" (denoised, spectral), "denoised_shrink"
                (denoised, targeted shrinkage), "gerber" (Gerber), "factor" (factor model),
                "minimum_covariance_determinant" (Minimum Covariance Determinant), "jlogo" (j-LoGo), "geodesic"
                (geodesic shrinkage), "regime_adjusted_ew" (regime adjusted exponentially weighted), "nercome"
                (NERCOME), "ccc_garch" (CCC-GARCH), "mm" (MM-estimator), "rocke" (Rocke's S-estimator). Defaults to
                "sample".
            linkage_method (str, optional): The linkage of the hierarchical clustering. Options are: "single"
                (single linkage), "complete" (complete linkage), "average" (average linkage), "ward" (Ward's
                method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo and Aste, 2012, which
                finds the number of clusters itself in the TMFG planar graph of the codependence). Defaults to
                "ward".
            clusters (int | None, optional): The number of clusters of the hierarchical clustering, or the most
                clusters ONC tries. Defaults to None, chosen by the two-order difference of the gap statistic (by
                the DBHT itself for its linkage; up to N - 1 for ONC).
            cluster_criterion (str, optional): How the hierarchical clustering chooses the number of clusters when
                it is not given. Options are: "two_difference_gap" (the two-order difference of the gap statistic),
                "standardized_silhouette" (the highest mean over standard deviation of the silhouette coefficients).
                Defaults to "two_difference_gap".
            threshold (float, optional): The codependence above which FTCA puts assets in one cluster. Defaults to
                0.5.
            seed (int | None, optional): The seed of the k-means starts of ONC (numpy's default generator), so the
                same returns give the same clusters. Defaults to 0.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Cluster of every asset, numbered from 1 (for FTCA in the
            order the clusters are formed, for ONC in the order of their first asset).

        Notes:
        - ONC follows the comparisons of López de Prado (2020, snippet 4.2): it clusters
          again when at least two clusters are below the average quality and keeps the new
          clusters when their quality is the higher one. A cluster whose silhouettes are
          all equal (one asset alone) has no quality and is left as it is. Across seeds the
          clusters of well-separated groups agree; where groups overlap, a k-means start can
          settle elsewhere, which the seed fixes.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_asset_clusters(method="ftca")
        ```

        Which returns:

        |      |   Cluster |
        |:-----|----------:|
        | AAPL |         1 |
        | MSFT |         1 |
        | JNJ  |         2 |
        | XOM  |         3 |
        """
        check_option("method", method, ("hierarchical", "ftca", "onc"))
        returns = self._get_returns(period, include_benchmark)
        covariance_matrix = optimization_model.get_covariance_matrix(
            returns,
            covariance,
            benchmark_returns=self._get_benchmark_returns(period),
            factors=(
                self._get_factor_returns(period) if covariance == "factor" else None
            ),
        )
        if method == "onc":
            labels = estimation_model.get_optimal_number_of_clusters(
                optimization_model.get_codependence_distance(
                    returns, covariance_matrix, codependence
                ),
                clusters,
                seed=seed,
            )
        elif method == "ftca":
            labels = estimation_model.get_fast_threshold_clusters(
                optimization_model.get_codependence(
                    returns, covariance_matrix, codependence
                ),
                threshold,
            )
        else:
            _, labels = optimization_model.get_hierarchical_clusters(
                covariance_matrix,
                linkage_method,
                clusters,
                distance=optimization_model.get_codependence_distance(
                    returns, covariance_matrix, codependence
                ),
                cluster_criterion=cluster_criterion,
                similarity=(
                    optimization_model.get_codependence_similarity(
                        returns, covariance_matrix, codependence
                    )
                    if linkage_method == "dbht"
                    else None
                ),
            )
        return pd.DataFrame({"Cluster": labels + 1}, index=covariance_matrix.index)

    @handle_errors
    def get_coskewness_matrix(
        self,
        method: str = "sample",
        period: str = "daily",
        span: int = 180,
        standardized_moments: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the coskewness matrix M3 of the assets, the third co-moments of their returns:
        how the return of one asset moves with the joint variation of two others.

        The skewness of a portfolio with weights w is w'M3(w ⊗ w) / σ_p³, so M3 is the input of
        mean-variance-skewness optimizations and the modified (Cornish-Fisher) VaR of a
        portfolio.

        The formula is as follows:

        - m_ijk = Σ_t w_t (r_ti - μ_i)(r_tj - μ_j)(r_tk - μ_k)
        - Sample: w_t = 1 / T; exponential: w_t = λ^(T - t) / Σ λ^(T - s) with
          λ = 1 - 2 / (span + 1); semi: w_t = 1 / T with min(r_ti - μ_i, 0) in place of
          every deviation (only returns below the average count); CCC-GARCH: w_t = 1 / T
          on the standardized residuals of the CCC-GARCH(1, 1) model at the next-period
          volatilities, z_i,t σ_i,T+1 / s(z_i) (Bollerslev, 1990; see
          get_covariance_estimate), the next-period coskewness of the model
        - standardized: m_ijk / (σ_i σ_j σ_k) with σ² = Σ_t w_t (r_t - μ)²

        For more information, see the following papers:

        - Jondeau, E. and Rockinger, M. (2006). "Optimal Portfolio Allocation under Higher
          Moments." European Financial Management, 12(1), 29-55.
        - Martellini, L. and Ziemann, V. (2010). "Improved Estimates of Higher-Order
          Comoments and Implications for Portfolio Selection." The Review of Financial
          Studies, 23(4), 1467-1502.

        Also known as: coskewness, third co-moment matrix, M3.

        Shares similar logic:
        - PerformanceAnalytics (R), M3.MM (sample): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L34
        - PerformanceAnalytics (R), M3.ewma (exponential): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L1486
        - PortfolioAnalytics (R), CCCgarch.MM (CCC-GARCH): https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/moment.functions.R#L22

        Args:
            method (str, optional): The estimator of the co-moments. Options are: "sample" (equal weights),
                "exponential" (exponentially weighted), "semi" (lower semi co-moment), "ccc_garch" (CCC-GARCH
                scenarios). Defaults to "sample".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            span (int, optional): The span in periods of the exponential weights. Defaults to 180.
            standardized_moments (bool, optional): Whether to divide by the product of the
                standard deviations, so the diagonal m_iii is the skewness of each asset
                and the entries are comparable across frequencies. Defaults to True.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per asset i and one column per pair of assets (j, k), from
            the periods in which every asset has a return, per period (not annualized).

        Notes:
        - The unstandardized sample matrix follows the same formula as PerformanceAnalytics'
          M3.MM. Its M3.ewma (default λ = 0.97, a span of about 65.7) assumes returns with a
          zero mean; the exponential estimate takes the deviations from the average return and
          follows the same formula as M3.ewma on the demeaned returns.
        - Unstandardized co-moments of daily returns are of the order of 1e-6, below the
          default rounding of 4 decimals; a higher rounding shows them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # The coskewness of each pair of assets with AAPL
        toolkit.portfolio.get_coskewness_matrix().loc[:, "AAPL"]
        ```

        Which returns:

        |      |   AAPL |    MSFT |     JNJ |
        |:-----|-------:|--------:|--------:|
        | AAPL | 0.4539 |  0.3026 |  0.0257 |
        | MSFT | 0.3026 |  0.198  | -0.0169 |
        | JNJ  | 0.0257 | -0.0169 | -0.168  |
        """
        matrix = estimation_model.get_coskewness_matrix(
            self._get_returns(period, include_benchmark),
            method,
            span,
            standardized_moments,
        )
        return self._round(matrix, rounding)

    @handle_errors
    def get_cokurtosis_matrix(
        self,
        method: str = "sample",
        period: str = "daily",
        span: int = 180,
        standardized_moments: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the cokurtosis matrix M4 of the assets, the fourth co-moments of their
        returns: how often assets have extreme returns at the same time.

        The kurtosis of a portfolio with weights w is w'M4(w ⊗ w ⊗ w) / σ_p⁴, the input of
        kurtosis-aware optimizations and the modified (Cornish-Fisher) VaR of a portfolio.

        The formula is as follows:

        - m_ijkl = Σ_t w_t (r_ti - μ_i)(r_tj - μ_j)(r_tk - μ_k)(r_tl - μ_l)
        - Sample: w_t = 1 / T; exponential: w_t = λ^(T - t) / Σ λ^(T - s) with
          λ = 1 - 2 / (span + 1); semi: w_t = 1 / T with min(r_ti - μ_i, 0) in place of
          every deviation, the lower semi cokurtosis; CCC-GARCH: w_t = 1 / T on the
          standardized residuals of the CCC-GARCH(1, 1) model at the next-period
          volatilities (see get_coskewness_matrix), the next-period cokurtosis of the model
        - standardized: m_ijkl / (σ_i σ_j σ_k σ_l) with σ² = Σ_t w_t (r_t - μ)²

        For more information, see the following papers:

        - Jondeau, E. and Rockinger, M. (2006). "Optimal Portfolio Allocation under Higher
          Moments." European Financial Management, 12(1), 29-55.
        - Cajas, D. (2022). "Convex Optimization of Portfolio Kurtosis." SSRN 4202967.

        Also known as: cokurtosis, fourth co-moment matrix, M4, semi cokurtosis.

        Shares similar logic:
        - PerformanceAnalytics (R), M4.MM (sample): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L72
        - PerformanceAnalytics (R), M4.ewma (exponential): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L1543
        - Riskfolio-Lib, cokurt_matrix (sample and semi, as N² × N²): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.cokurt_matrix
        - PortfolioAnalytics (R), CCCgarch.MM (CCC-GARCH): https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/moment.functions.R#L22

        Args:
            method (str, optional): The estimator of the co-moments. Options are: "sample" (equal weights),
                "exponential" (exponentially weighted), "semi" (lower semi co-moment), "ccc_garch" (CCC-GARCH
                scenarios). Defaults to "sample".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            span (int, optional): The span in periods of the exponential weights. Defaults to 180.
            standardized_moments (bool, optional): Whether to divide by the product of the
                standard deviations, so the diagonal m_iiii is the kurtosis of each asset
                (3 for a normal distribution, not the excess). Defaults to True.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per asset i and one column per triple of assets
            (j, k, l), N × N³, from the periods in which every asset has a return, per
            period (not annualized).

        Notes:
        - The unstandardized sample matrix follows the same formula as PerformanceAnalytics'
          M4.MM; Riskfolio-Lib's cokurt_matrix holds the sample and semi cokurtosis as an
          N² × N² matrix. The exponential estimate follows the same formula as M4.ewma on the
          demeaned returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # The cokurtosis of each pair of assets with AAPL twice
        toolkit.portfolio.get_cokurtosis_matrix().loc[:, ("AAPL", "AAPL")]
        ```

        Which returns:

        |      |   AAPL |   MSFT |
        |:-----|-------:|-------:|
        | AAPL | 9.5182 | 6.0896 |
        | MSFT | 6.0896 | 4.8517 |
        """
        matrix = estimation_model.get_cokurtosis_matrix(
            self._get_returns(period, include_benchmark),
            method,
            span,
            standardized_moments,
        )
        return self._round(matrix, rounding)

    @handle_errors
    def get_estimation_uncertainty(
        self,
        method: str = "stationary",
        period: str = "daily",
        n_bootstrap: int = 1000,
        block_size: int = 3,
        confidence_level: float = 0.95,
        seed: int | None = 0,
        output: str = "expected_returns",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how uncertain the estimates of the expected returns and the covariance matrix
        are.

        Both are estimated again on many simulated histories of the same length, and the spread
        of those estimates gives a confidence interval for every expected return and covariance
        and a standard error for every expected return. These are the box uncertainty sets of
        robust (worst-case) mean-variance optimization.

        - The stationary bootstrap of Politis and Romano (1994), blocks of a random length with
          a geometric distribution of mean block_size, wrapping from the last period to the
          first.
        - The circular block bootstrap: blocks of block_size periods that wrap around.
        - The moving block bootstrap: blocks of block_size periods that start where a whole
          block fits (Künsch, 1989).
        - The normal distribution: histories drawn from a multivariate normal distribution with
          the average returns and the sample covariance.

        Block bootstraps keep the dependence between consecutive returns, such as
        volatility clustering, that drawing single periods would break.

        For more information, see the following papers:

        - Politis, D. N. and Romano, J. P. (1994). "The Stationary Bootstrap." Journal of
          the American Statistical Association, 89(428), 1303-1313.
        - Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary
          Observations." The Annals of Statistics, 17(3), 1217-1241.
        - Tütüncü, R. H. and Koenig, M. (2004). "Robust Asset Allocation." Annals of
          Operations Research, 132, 157-187.

        Also known as: bootstrapped expected returns, uncertainty set, estimation error,
        block bootstrap.

        Shares similar logic:
        - Riskfolio-Lib, bootstrapping: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.bootstrapping
        - Riskfolio-Lib, normal_simulation (normal distribution): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.normal_simulation
        - skfolio, BootstrapMuUncertaintySet: https://skfolio.org/generated/skfolio.uncertainty_set.BootstrapMuUncertaintySet.html

        Args:
            method (str, optional): How the histories are simulated. Options are: "stationary" (stationary
                bootstrap), "circular" (circular block bootstrap), "moving" (moving block bootstrap), "normal"
                (multivariate normal draws). Defaults to "stationary".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            n_bootstrap (int, optional): The number of simulated histories. Defaults to
                1000 (as skfolio; Riskfolio-Lib draws 6000).
            block_size (int, optional): The (average) number of periods in a block.
                Defaults to 3 (as Riskfolio-Lib's window).
            confidence_level (float, optional): The confidence level of the bounds, the
                (1 - c) / 2 and (1 + c) / 2 percentiles of the estimates. Defaults to 0.95.
            seed (int | None, optional): The seed of the random draws, so the result repeats; None draws a new one
                every time. Defaults to 0.
            output (str, optional): Options are: "expected_returns" (the Expected Return, Lower, Upper and Standard
                Error per asset), "covariance" (the Lower and Upper bound of every covariance). Defaults to
                "expected_returns".
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The annualized estimates and their bounds (the averages per
            period, their bounds and standard errors and the covariances times the number
            of periods in a year, without compounding).

        Notes:
        - The bounds of the covariances hold per entry and are not adjusted to form
          positive semidefinite matrices. Riskfolio-Lib draws with the arch package, so
          its bounds agree with these in distribution, not draw for draw.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_estimation_uncertainty()
        ```

        Which returns:

        |      |   Expected Return |   Lower |   Upper |   Standard Error |
        |:-----|------------------:|--------:|--------:|-----------------:|
        | AAPL |            0.1932 | -0.0568 |  0.4194 |           0.1207 |
        | MSFT |            0.2017 | -0.0032 |  0.3985 |           0.1045 |
        | JNJ  |            0.0987 | -0.0518 |  0.2328 |           0.0756 |
        | XOM  |            0.2911 |  0.063  |  0.5191 |           0.1179 |
        """
        check_option("output", output, ("expected_returns", "covariance"))
        returns = self._get_returns(period, include_benchmark)
        table, lower, upper = estimation_model.get_bootstrap_estimates(
            returns,
            method,
            n_bootstrap,
            block_size,
            confidence_level,
            seed,
        )
        if output == "covariance":
            table = pd.concat({"Lower": lower, "Upper": upper}, axis=1)
        return self._round(
            table * self._get_periods_per_year(period, returns.index), rounding
        )

    @handle_errors
    def get_factor_loadings(
        self,
        selection: str = "ols",
        criterion: str = "pvalue",
        threshold: float = 0.05,
        components: float = 0.95,
        factors: pd.DataFrame | None = None,
        region: str = "United States",
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Estimate the loadings of a time series factor model.

        It is the regression of every asset's returns on factor returns, by default the five
        factors of Fama and French (market, size, value, profitability and investment), r_i,t =
        α_i + Σ_f β_i,f F_f,t + ε_i,t. The loadings are what the factor covariance, the factor
        risk contributions, the factor attribution and the augmented and Bayesian
        Black-Litterman models build on.

        - Ordinary least squares on every factor.
        - Forward selection: starting without factors, add the factor with the lowest p-value as
          long as every p-value stays at or below the threshold (or, with another criterion, as
          long as it improves); the factors left out load zero.
        - Backward selection: starting with every factor, drop the one with the highest p-value
          until all are at or below the threshold (or as long as dropping one improves the
          criterion). Both stepwise selections keep at least the single best factor on the
          p-value.
        - Principal components regression: the standardized factors' first principal components
          that explain the given share of their variance as regressors, mapped back to loadings
          on the factors, which removes the instability that correlated factors give OLS
          (Jolliffe, 1982).

        The Alpha is the annualized intercept, α times the number of periods in a year.
        The R Squared is the share of an asset's variance the factors explain and the
        Residual Volatility (ddof=1, annualized) what they leave, the idiosyncratic risk
        the factor covariance keeps on the diagonal.

        For more information, see the following sources:

        - Fama, E. F. and French, K. R. (1993). "Common Risk Factors in the Returns on
          Stocks and Bonds." Journal of Financial Economics, 33(1), 3-56.
        - Fama, E. F. and French, K. R. (2015). "A Five-Factor Asset Pricing Model."
          Journal of Financial Economics, 116(1), 1-22.
        - Connor, G. (1995). "The Three Types of Factor Models: A Comparison of Their
          Explanatory Power." Financial Analysts Journal, 51(3), 42-46.
        - Jolliffe, I. T. (1982). "A Note on the Use of Principal Components in
          Regression." Journal of the Royal Statistical Society, Series C, 31(3), 300-303.

        Also known as: factor exposures, factor betas, loadings matrix, stepwise
        regression, principal components regression.

        Referenced against:
        - Riskfolio-Lib, loadings_matrix (backward selection, every factor kept): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.loadings_matrix
        - Riskfolio-Lib, PCR: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.PCR

        Shares similar logic:
        - Riskfolio-Lib, forward_regression: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.forward_regression
        - Riskfolio-Lib, backward_regression: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.backward_regression

        Args:
            selection (str, optional): Options are: "ols" (every factor), "forward" (forward selection), "backward"
                (backward selection), "pcr" (principal components regression). Defaults to "ols".
            criterion (str, optional): What the stepwise selections compare. Options are: "pvalue" (p-value), "aic"
                (Akaike information criterion), "bic" (Schwarz information criterion), "r_squared" (R squared),
                "adjusted_r_squared" (adjusted R squared). Defaults to "pvalue".
            threshold (float, optional): The highest p-value a selected factor may have.
                Defaults to 0.05.
            components (float, optional): The principal components the principal components regression keeps: a
                number of 1 or more, or between 0 and 1 the share of the variance of the standardized factors to
                explain. Defaults to 0.95.
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
            pd.DataFrame: Per asset the Alpha (annualized), the loading on every factor,
            the R Squared and the Residual Volatility (annualized), from the periods in
            which every asset and every factor has a return.

        Notes:
        - The Fama and French factors are excess returns while the asset returns are
          total returns, so the Alpha holds the risk free rate. Ken French publishes the
          factors about a month after each month ends, so the latest weeks are left out.
        - OLS gives the loadings of Riskfolio-Lib's loadings_matrix as a backward selection at a
          threshold of 1; the AIC and BIC are defined as statsmodels defines them
          (Riskfolio-Lib's AIC and SIC).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_factor_loadings(selection="forward")
        ```

        Which returns:

        |      |   Alpha |   Mkt-RF |     SMB |     HML |     RMW |     CMA |   R Squared |   Residual Volatility |
        |:-----|--------:|---------:|--------:|--------:|--------:|--------:|------------:|----------------------:|
        | AAPL |  0.0726 |   1.1791 | -0.2683 | -0.3519 |  0      |  0.3824 |      0.6004 |                0.1761 |
        | MSFT |  0.0851 |   1.0449 | -0.321  | -0.3811 |  0.3329 | -0.2635 |      0.6806 |                0.1453 |
        | JNJ  |  0.0387 |   0.2901 | -0.1045 |  0.1992 |  0.1657 |  0      |      0.0868 |                0.1599 |
        | XOM  |  0.1409 |   0.7905 | -0.2442 |  1.0268 | -0.4884 |  0.2562 |      0.3934 |                0.2105 |
        """
        returns, factor_returns = self._align_factors(
            self._get_returns(period, include_benchmark),
            self._get_factor_returns(period, factors, region),
        )
        loadings = estimation_model.get_factor_loadings(
            returns, factor_returns, selection, criterion, threshold, components
        )
        _, _, residual_variance = estimation_model.get_factor_model(
            returns, factor_returns, loadings
        )
        periods_per_year = self._get_periods_per_year(period, returns.index)
        table = pd.concat(
            [
                (loadings["Intercept"] * periods_per_year).rename("Alpha"),
                loadings.drop(columns="Intercept"),
                (1 - residual_variance / returns.var()).rename("R Squared"),
                np.sqrt(residual_variance * periods_per_year).rename(
                    "Residual Volatility"
                ),
            ],
            axis=1,
        )
        return self._round(table, rounding)

    def _get_risk_contribution_table(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        covariance_matrix: pd.DataFrame,
        risk_measure: str,
        alpha: float,
    ) -> pd.DataFrame:
        """The risk contributions of the assets to one of the risk measures."""
        if risk_measure == "volatility":
            return optimization_model.get_risk_contributions(weights, covariance_matrix)
        if risk_measure == "cvar":
            return optimization_model.get_expected_shortfall_contributions(
                weights, returns, alpha
            )
        if risk_measure in optimization_model.FINITE_DIFFERENCE_RISK_MEASURES:
            return optimization_model.get_finite_difference_risk_contributions(
                weights, returns, risk_measure, alpha
            )
        if risk_measure in optimization_model.HISTORICAL_RISK_MEASURES:
            return optimization_model.get_historical_risk_contributions(
                weights, returns, risk_measure, alpha
            )
        measures = (
            "volatility",
            "cvar",
            *optimization_model.HISTORICAL_RISK_MEASURES,
            *optimization_model.FINITE_DIFFERENCE_RISK_MEASURES,
        )
        raise ValueError(
            f"The risk measure must be one of {', '.join(measures)}, not "
            f"'{risk_measure}'."
        )

    @handle_errors
    def get_factor_risk_contributions(
        self,
        weights: dict[str, float] | None = None,
        risk_measure: str = "volatility",
        alpha: float = 0.05,
        selection: str = "ols",
        factors: pd.DataFrame | None = None,
        region: str = "United States",
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how much each factor, by default the five factors of Fama and French,
        contributes to the risk of a portfolio, rather than each asset.

        The portfolio's exposure to the factors is y = B'w, with B the loadings of the assets on
        the factors (see get_factor_loadings), and the marginal risk of the factors follows from
        that of the assets m = ∂ρ/∂w (see get_risk_contributions) through the pseudo-inverse of
        B': ∂ρ/∂y = pinv(B')'m. A factor contributes its exposure times its marginal risk, and
        the part of the weights the factors cannot span forms an Other factor, so the
        contributions add up to the risk measure (Meucci, 2009; Roncalli and Weisang, 2016).
        With fewer assets than factors the factors span every portfolio and Other is zero.

        The risk measure is any of get_risk_contributions: the volatility (annualized), the
        expected shortfall, the drawdowns or the other measures of the Risk module (per
        period).

        For more information, see the following sources:

        - Meucci, A. (2009). "Managing Diversification." Risk, 22(5), 74-79.
        - Roncalli, T. and Weisang, G. (2016). "Risk Parity Portfolios with Risk
          Factors." Quantitative Finance, 16(3), 377-388.
        - Fama, E. F. and French, K. R. (2015). "A Five-Factor Asset Pricing Model."
          Journal of Financial Economics, 116(1), 1-22.

        Also known as: factor risk decomposition, factor risk budget, risk factor
        contributions.

        Referenced against:
        - Riskfolio-Lib, Factors_Risk_Contribution: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Factors_Risk_Contribution

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            risk_measure (str, optional): A risk measure of get_risk_contributions. Options are: "volatility" (of
                the covariance estimate), "cvar" (the expected shortfall), "semideviation" (below a return of zero),
                "mad" (the mean absolute deviation), "cdar" (the Conditional Drawdown at Risk), "var" (historical
                Value at Risk), "evar" (Entropic Value at Risk), "worst_realization" (the largest loss),
                "gini_mean_difference" (Gini mean difference), "tail_gini" (Tail Gini), "lower_partial_moment"
                (first lower partial moment below zero), "kurtosis" (square root of the fourth central moment),
                "maximum_drawdown" (maximum drawdown), "average_drawdown" (average drawdown), "drawdown_at_risk"
                (Drawdown at Risk), "entropic_drawdown_at_risk" (Entropic Drawdown at Risk), "ulcer_index" (Ulcer
                Index). Defaults to "volatility".
            alpha (float, optional): The tail probability of the tail measures, e.g.
                0.05 for the worst 5% of periods. Defaults to 0.05.
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
            pd.DataFrame: Per factor, Other and in total, the Exposure, Marginal
            Contribution, Risk Contribution and Percentage Contribution, on the periods in
            which the assets held and the factors all have a return.

        Notes:
        - Riskfolio-Lib's Factors_Risk_Contribution uses the same inputs with OLS loadings: the
          volatility uses the sample covariance (ddof=1) of the asset returns, annualized,
          and the other measures their historical returns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_factor_risk_contributions(
            weights={"AAPL": 0.4, "MSFT": 0.3, "JNJ": 0.2, "XOM": 0.1}
        )
        ```

        Which returns:

        | Factor   |   Exposure |   Marginal Contribution |   Risk Contribution |   Percentage Contribution |
        |:---------|-----------:|------------------------:|--------------------:|--------------------------:|
        | Mkt-RF   |     0.9376 |                  0.1727 |              0.1619 |                    0.873  |
        | SMB      |    -0.1694 |                 -0.0254 |              0.0043 |                    0.0232 |
        | HML      |    -0.1765 |                 -0.0437 |              0.0077 |                    0.0415 |
        | RMW      |     0.2806 |                  0.0337 |              0.0095 |                    0.051  |
        | CMA      |     0.1144 |                  0.0184 |              0.0021 |                    0.0113 |
        | Other    |   nan      |                nan      |              0      |                    0      |
        | Total    |   nan      |                nan      |              0.1855 |                    1      |
        """
        validate_alpha(alpha)
        resolved = self._get_held_weights(weights, period, include_benchmark)
        returns, factor_returns = self._align_factors(
            self._get_returns(period, include_benchmark, resolved.index.tolist()),
            self._get_factor_returns(period, factors, region),
        )
        assets = self._get_risk_contribution_table(
            resolved,
            returns,
            returns.cov() * self._get_periods_per_year(period, returns.index),
            risk_measure,
            alpha,
        )
        # A measure homogeneous of a higher order splits its Euler sum by that order, so
        # the factor contributions add up to the measure itself.
        marginal = assets["Marginal Contribution"] / (
            optimization_model.FINITE_DIFFERENCE_RISK_MEASURES.get(risk_measure, 1)
        )
        loadings = estimation_model.get_factor_loadings(
            returns, factor_returns, selection
        ).drop(columns="Intercept")
        table = optimization_model.get_factor_risk_contributions(
            resolved, loadings, marginal
        )
        table.index.name = "Factor"
        table.loc["Total"] = [
            np.nan,
            np.nan,
            table["Risk Contribution"].sum(),
            table["Percentage Contribution"].sum(),
        ]
        return self._round(table, rounding)

    @handle_errors
    def get_entropy_pooling(
        self,
        views: dict | None = None,
        volatility_views: dict | None = None,
        correlation_views: dict | None = None,
        ranking: list[str] | None = None,
        opinions: list[dict] | None = None,
        opinion_weights: list[float] | None = None,
        pooling: str = "linear",
        divergence_penalty: float = 0.0,
        output: str = "moments",
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Reweight the historical periods so that they agree with views on the assets, by Entropy
        Pooling (Meucci, 2008).

        These are the probabilities of the periods closest to equal probabilities in relative
        entropy, Σ p_t ln(p_t T), under which the views hold. The returns themselves stay as
        they were, so the posterior keeps the fat tails and co-movements of the history; only
        how much each period counts changes. The posterior expected returns and covariance
        matrix follow from the probabilities.

        Views are annualized, like those of get_black_litterman_weights:

        - views: the expected return of an asset ({"AAPL": 0.10}, an arithmetic return a
          year) or the difference of two ({"AAPL-MSFT": 0.02}).
        - volatility_views: the annualized volatility of an asset ({"AAPL": 0.25}).
        - correlation_views: the correlation of two assets ({"AAPL-MSFT": 0.5}).
        - ranking: an ordering of the expected returns, from the highest to the lowest
          (["XOM", "MSFT", "AAPL"]), the ranking views of Meucci (2008): the inequalities
          E_p[r_(k+1)] ≤ E_p[r_(k)] for every pair of neighbours, imposed with the views on
          the expected returns. Only the order is stated, not by how much.

        A view is an equality (a number) or an inequality, a pair (">=", 0.10) or
        ("<=", 0.10). The views on volatilities hold the means found with the mean views,
        and the views on correlations hold both the means and the volatilities, the
        sequential treatment of Meucci, Ardia and Keel (2011), in the order skfolio's
        EntropyPooling takes them. The problem is solved through its dual with SciPy's
        L-BFGS-B.

        With opinions, a list of views of several experts, each a dictionary with the
        keys "views", "volatility_views", "correlation_views" and "ranking", every opinion gets its
        own posterior and these are pooled with the opinion weights (equal by default; any
        weight left to sum to one goes to the equal prior probabilities): linearly,
        p = Σ π_k p_k (Stone, 1961), or logarithmically, p ∝ Π p_k^π_k (Genest and Zidek,
        1986), optionally weighing down opinions far from the consensus by
        exp(-α KL(p_k ‖ c)) with α the divergence penalty.

        For more information, see the following sources:

        - Meucci, A. (2008). "Fully Flexible Views: Theory and Practice." Risk, 21(10),
          97-102.
        - Stone, M. (1961). "The Opinion Pool." The Annals of Mathematical Statistics,
          32(4), 1339-1342.
        - Genest, C. and Zidek, J. V. (1986). "Combining Probability Distributions: A
          Critique and an Annotated Bibliography." Statistical Science, 1(1), 114-135.

        Also known as: fully flexible views, fully flexible probabilities, opinion pooling,
        views on scenarios, ranking views, ordering views.

        Referenced against:
        - skfolio, EntropyPooling: https://skfolio.org/generated/skfolio.prior.EntropyPooling.html
        - skfolio, OpinionPooling: https://skfolio.org/generated/skfolio.prior.OpinionPooling.html

        Args:
            views (dict | None, optional): The annualized expected return views. Defaults
                to None.
            volatility_views (dict | None, optional): The annualized volatility views.
                Defaults to None.
            correlation_views (dict | None, optional): The correlation views. Defaults to
                None.
            ranking (list[str] | None, optional): Assets from the highest expected return
                to the lowest, at least two. Defaults to None.
            opinions (list[dict] | None, optional): The views of several experts to pool,
                instead of views, volatility_views, correlation_views and ranking. Defaults
                to None.
            opinion_weights (list[float] | None, optional): The weight of every opinion,
                summing to at most one. Defaults to None, equal weights.
            pooling (str, optional): How the opinions are pooled. Options are: "linear" (arithmetic mixture),
                "logarithmic" (geometric mixture). Defaults to "linear".
            divergence_penalty (float, optional): The penalty α on an opinion's relative
                entropy to the consensus. Defaults to 0.
            output (str, optional): Options are: "moments" (the prior and posterior expected return and volatility
                per asset), "covariance" (the posterior covariance matrix), "probabilities" (the prior and posterior
                probability of every period), "summary" (the relative entropy and the effective number of periods).
                Defaults to "moments".
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
            pd.DataFrame: The output chosen: the expected returns (arithmetic averages
            times the number of periods in a year) and volatilities (annualized), the
            annualized covariance matrix, the probabilities, or the Relative Entropy,
            Effective Number of Scenarios (exp of the entropy of the probabilities, T for
            equal ones) and the number of Scenarios.

        Notes:
        - The posterior covariance is the probability-weighted covariance Σ p_t (r_t - μ)
          (r_t - μ)' with μ = Σ p_t r_t, without a small-sample correction, the basis on
          which the views are imposed, so a volatility view comes back exactly. skfolio
          applies numpy's correction for weights, 1 / (1 - Σ p_t²), on top, which raises
          the variance by about 1 / T (0.1% for daily data, 3% for 36 monthly returns).
        - The prior volatility is the sample standard deviation of the returns (ddof=1),
          as elsewhere in the Finance Toolkit. With equal probabilities the posterior's
          basis gives a volatility √((T - 1) / T) times the prior's, so without views
          the two columns differ by that factor.
        - With mean and volatility views the probabilities agree with skfolio's EntropyPooling
          to 1e-9; with correlation views the views are met here more tightly and the posterior
          lies at least as close to the prior (a lower relative entropy).
        - A ranking is the set of relative views ("MSFT-XOM": ("<=", 0)) that
          PortfolioAnalytics' meucci.ranking imposes, which takes the order from the lowest
          expected return to the highest. A ranking the history already follows leaves the
          probabilities equal.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_entropy_pooling(
            views={"AAPL": 0.05, "XOM-JNJ": (">=", 0.02)},
            volatility_views={"MSFT": 0.30},
        )
        ```

        Which returns:

        |      |   Prior Return |   Posterior Return |   Prior Volatility |   Posterior Volatility |
        |:-----|---------------:|-------------------:|-------------------:|-----------------------:|
        | AAPL |         0.1932 |             0.05   |             0.2786 |                 0.3151 |
        | MSFT |         0.2017 |             0.118  |             0.2571 |                 0.3    |
        | JNJ  |         0.0987 |             0.0698 |             0.1673 |                 0.1696 |
        | XOM  |         0.2911 |             0.264  |             0.2702 |                 0.2762 |
        """
        outputs = ("moments", "covariance", "probabilities", "summary")
        check_option("output", output, outputs)
        # Checked also for a single opinion, which is not pooled.
        check_option("pooling", pooling, estimation_model.POOLING_METHODS)
        returns = self._get_returns(period, include_benchmark)
        periods_per_year = self._get_periods_per_year(period, returns.index)

        if opinions:
            if views or volatility_views or correlation_views or ranking:
                raise ValueError(
                    "Pass the views either directly or as opinions, not both."
                )
            probabilities = estimation_model.get_opinion_pooling(
                pd.concat(
                    [
                        estimation_model.get_opinion_entropy_pooling(
                            returns, opinion, periods_per_year
                        )
                        for opinion in opinions
                    ],
                    axis=1,
                ),
                opinion_weights,
                pooling,
                divergence_penalty,
            )
        else:
            probabilities = estimation_model.get_opinion_entropy_pooling(
                returns,
                {
                    "views": views,
                    "volatility_views": volatility_views,
                    "correlation_views": correlation_views,
                    "ranking": ranking,
                },
                periods_per_year,
            )

        weights = probabilities.to_numpy()
        prior = np.full(len(weights), 1 / len(weights))
        if output == "probabilities":
            table = pd.DataFrame(
                {"Prior Probability": prior, "Posterior Probability": weights},
                index=returns.index,
            )
        elif output == "summary":
            positive = weights[weights > 0]
            table = pd.DataFrame(
                {
                    "Relative Entropy": float(
                        np.sum(positive * np.log(positive * len(weights)))
                    ),
                    "Effective Number of Scenarios": float(
                        np.exp(-np.sum(positive * np.log(positive)))
                    ),
                    "Scenarios": len(weights),
                },
                index=["Entropy Pooling"],
            )
        else:
            covariance = pd.DataFrame(
                np.cov(returns.to_numpy(), rowvar=False, aweights=weights, bias=True),
                index=returns.columns,
                columns=returns.columns,
            )
            if output == "covariance":
                table = covariance * periods_per_year
            else:
                table = pd.DataFrame(
                    {
                        "Prior Return": returns.mean() * periods_per_year,
                        "Posterior Return": pd.Series(
                            weights @ returns.to_numpy(), index=returns.columns
                        )
                        * periods_per_year,
                        "Prior Volatility": returns.std() * np.sqrt(periods_per_year),
                        "Posterior Volatility": np.sqrt(
                            np.diag(covariance) * periods_per_year
                        ),
                    }
                )
        return self._round(table, rounding)

    @handle_errors
    def get_return_contributions(
        self,
        weights: dict[str, float] | None = None,
        period: str = "monthly",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how much each asset added to the portfolio's return in every period.

        The periods are weeks, months, quarters or years, and the contributions of the assets
        add up to the portfolio's return of that period. Every day an asset contributes its
        weight at the start of the day times its return, c_i,t = w_i,t-1 r_i,t, which add up to
        the portfolio's daily return R_t = Σ c_i,t. Over a longer period the daily contributions
        are carried by the growth of the portfolio since the period began: C_i = Σ_t c_i,t
        Π_(s<t) (1 + R_s), the contribution in money divided by the value at the start of the
        period, so the contributions add up to the compounded return Π (1 + R_t) - 1 (Bacon,
        2008).

        The weights are the latest weights of the Portfolio the Toolkit came from, as they
        moved from day to day, or, when weights are given (or there is no Portfolio), those
        weights restored every day.

        For more information, see the following sources:

        - Bacon, C. R. (2008). "Practical Portfolio Performance Measurement and
          Attribution." 2nd edition, Wiley, chapter 5.

        Also known as: contribution to return, return attribution by asset, performance
        contribution.

        Shares similar logic:
        - PerformanceAnalytics (R), to.period.contributions: https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/to.period.contributions.R#L45

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the weights of the Portfolio the Toolkit came from over
                time, or equal weights.
            period (str, optional): The period to add the contributions up over. Options are: "daily" (daily
                returns), "weekly" (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns),
                "yearly" (yearly returns). Defaults to "monthly".
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per period the contribution of every asset held and the
            Portfolio return, compounded within the period.

        Notes:
        - The contributions follow PerformanceAnalytics' to.period.contributions of the daily
          contributions; with given weights these are rebalanced every day, not left to drift.
        - With fixed weights (given, or equal ones) the periods start on the first day on
          which every asset held has a return, so an asset that was not listed yet does not
          count as a return of 0, as in get_portfolio_risk_metrics. Within the sample a day
          on which an asset has no return (it did not trade) is a contribution of 0.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_return_contributions(
            weights={"AAPL": 0.4, "MSFT": 0.3, "JNJ": 0.2, "XOM": 0.1}, period="yearly"
        )
        ```

        Which returns:

        | Date   |    AAPL |    MSFT |     JNJ |     XOM |   Portfolio |
        |:-------|--------:|--------:|--------:|--------:|------------:|
        | 2021   |  0.1698 |  0.1574 |  0.0289 |  0.0514 |      0.4074 |
        | 2022   | -0.1049 | -0.0914 |  0.0105 |  0.0655 |     -0.1204 |
        | 2023   |  0.1833 |  0.1649 | -0.0136 | -0.0046 |      0.3301 |
        | 2024   |  0.1263 |  0.0396 | -0.0087 |  0.0116 |      0.1689 |
        | 2025   |  0.0598 |  0.04   |  0.0889 |  0.0187 |      0.2074 |
        """
        check_period(period, PERIODS_PER_YEAR)
        resolved = self._get_held_weights(weights, "daily", include_benchmark)
        assets = resolved.index.tolist()
        fixed_weights = not (weights is None and self._portfolio_weights)
        if not fixed_weights:
            assets = self._get_loaded_assets("daily", include_benchmark)
        # The first day of the data has no return, as there is no close before it.
        returns = (
            self._historical_data["daily"]["Return"]
            .iloc[1:]
            .loc[self._start_date : self._end_date, assets]
            .dropna(how="all")
        )
        if fixed_weights:
            # Fixed weights hold every asset from the first day, which an asset that is not
            # listed yet cannot be; as in get_portfolio_risk_metrics, the sample starts on the
            # first day every asset held has a return, rather than counting the days before
            # its listing as returns of 0. Days without a trade later on stay a return of 0.
            complete = returns.notna().all(axis=1)
            if not complete.any():
                raise ValueError("No day has a return for every asset held.")
            returns = returns.loc[complete.idxmax() :]
        held = self._get_weights_over_time(weights, "daily", assets, returns.index)
        daily = (held * returns.fillna(0)).loc[:, (held != 0).any()]
        table = attribution_model.get_period_contributions(
            daily, self._historical_data[period]["Return"].index.freq
        )
        table.index.name = "Date"
        return self._round(table, rounding)
