"""Portfolio Optimization Module"""

__docformat__ = "google"

import functools
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.portfolio.construction.optimization import (
    estimation_model,
    optimization_model,
)
from financetoolkit.portfolio.helpers import BENCHMARK_TICKER, PORTFOLIO_TICKER
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    check_option,
    handle_errors,
    validate_alpha,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import track_sources

# ruff: noqa: E501

logger = get_logger()

# The largest gap between the sum of the weights and one that is not shown as cash, well
# above the rounding of the solvers.
CASH_TOLERANCE = 1e-6

# The optimizers that stacking combines, with the names they are shown under.
STACKING_METHODS = {
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    "inverse_volatility": "Inverse Volatility",
    "maximum_diversification": "Maximum Diversification",
    "hierarchical_risk_parity": "Hierarchical Risk Parity",
    "hierarchical_equal_risk_contribution": "Hierarchical Equal Risk Contribution",
    "nested_clustered_optimization": "Nested Clustered Optimization",
    "schur_complementary": "Schur Complementary",
    "minimum_cvar": "Minimum CVaR",
    "minimum_drawdown": "Minimum Drawdown",
    "minimum_mad": "Minimum MAD",
    "minimum_semideviation": "Minimum Semideviation",
    "equal_weight": "Equal Weight",
}
DEFAULT_STACKING_METHODS = (
    "inverse_volatility",
    "maximum_diversification",
    "hierarchical_risk_parity",
    "minimum_cvar",
)


@filter_shown_columns
@track_sources
class OptimizationConstruction:
    """
    Portfolio optimizers that set the weights of the assets of a Toolkit.

    They are minimum variance, maximum Sharpe, mean-variance, the efficient frontier, risk
    parity, hierarchical and nested clustered methods, inverse volatility, CVaR, drawdown,
    Omega, mean-risk (MAD, semideviation, Gini and other risk measures), the critical line
    algorithm, worst-case mean-variance, distributionally robust CVaR, diversification,
    tracking error, growth optimal, resampled and Black-Litterman weights.
    """

    def _get_current_weights(
        self,
        current_weights: dict[str, float] | None,
        assets: list[str],
        needed: bool,
    ) -> pd.Series | None:
        """
        The weights held now that the turnover is measured from: as given, otherwise,
        when a maximum turnover or transaction cost needs them, the latest weights of the
        Portfolio the Toolkit came from. None when neither applies.
        """
        if current_weights is not None:
            return pd.Series(current_weights, dtype=float)
        if needed and self._portfolio_weights:
            latest = self._portfolio_weights.get("daily", pd.DataFrame())
            if not latest.empty:
                return latest.iloc[-1].reindex(assets).fillna(0).astype(float)
        return None

    @staticmethod
    def _add_turnover(
        table: pd.DataFrame, weights: pd.Series, current: pd.Series | None
    ) -> pd.DataFrame:
        """
        The table with a Turnover column, |w_i - w0_i| per asset and their sum for the
        portfolio, when there are current weights.
        """
        if current is None:
            return table
        traded = (weights - current.reindex(weights.index).fillna(0)).abs()
        table["Turnover"] = traded
        table.loc[PORTFOLIO_TICKER, "Turnover"] = traded.sum()
        return table

    def _weights_table(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        risk_free_rate: float,
        rounding: int | None,
        current: pd.Series | None = None,
        transaction_cost: float = 0.0,
    ) -> pd.DataFrame:
        """
        The weights with, per asset and for the portfolio, the expected return, volatility,
        Sharpe ratio and share of the portfolio's risk. Weights that do not sum to one leave
        the rest, 1 - Σw, in a Cash row at the risk free rate (negative when borrowed above
        a net exposure of one), so the portfolio's expected return and Sharpe ratio include
        it. With current weights it adds the Turnover, and with a transaction cost the
        Transaction Cost, the cost times |w_i - w0_i| per asset, which the portfolio's
        expected return and Sharpe ratio are after, as the optimizers that take it maximize.
        """
        assets = weights
        cash = 1 - float(weights.sum())
        if abs(cash) > CASH_TOLERANCE:
            weights = pd.concat([weights, pd.Series({"Cash": cash})])
            expected_returns = pd.concat(
                [expected_returns, pd.Series({"Cash": risk_free_rate})]
            )
            with_cash = covariance.index.append(pd.Index(["Cash"]))
            covariance = covariance.reindex(
                index=with_cash, columns=with_cash, fill_value=0.0
            )
        volatility = pd.Series(np.sqrt(np.diag(covariance)), index=covariance.index)
        # An asset without volatility (cash, a constant price) has no Sharpe ratio rather
        # than an infinite one, as the Portfolio row has none without volatility.
        risky_volatility = volatility.where(
            volatility > optimization_model.ZERO_VOLATILITY_TOLERANCE
        )
        statistics = optimization_model.get_portfolio_statistics(
            weights, expected_returns, covariance, risk_free_rate
        )
        # A portfolio without volatility (a market neutral optimum that holds nothing) has
        # no risk to share out, so its risk contributions are NaN, as is its Sharpe ratio.
        if statistics["Volatility"] > optimization_model.ZERO_VOLATILITY_TOLERANCE:
            shares = optimization_model.get_risk_contributions(weights, covariance)[
                "Percentage Contribution"
            ]
        else:
            shares = pd.Series(np.nan, index=weights.index)
        table = pd.DataFrame(
            {
                "Weight": weights,
                "Expected Return": expected_returns,
                "Volatility": volatility,
                "Sharpe Ratio": (expected_returns - risk_free_rate) / risky_volatility,
                "Risk Contribution": shares,
            }
        )
        table.loc[PORTFOLIO_TICKER] = [
            weights.sum(),
            statistics["Expected Return"],
            statistics["Volatility"],
            statistics["Sharpe Ratio"],
            shares.sum(min_count=1),
        ]
        table = self._add_turnover(table, assets, current)
        if transaction_cost and current is not None:
            costs = transaction_cost * table["Turnover"].drop(PORTFOLIO_TICKER)
            table["Transaction Cost"] = costs
            table.loc[PORTFOLIO_TICKER, "Transaction Cost"] = costs.sum()
            table.loc[PORTFOLIO_TICKER, "Expected Return"] -= costs.sum()
            if not np.isnan(statistics["Sharpe Ratio"]):
                table.loc[PORTFOLIO_TICKER, "Sharpe Ratio"] = (
                    table.loc[PORTFOLIO_TICKER, "Expected Return"] - risk_free_rate
                ) / statistics["Volatility"]
        return self._round(table, rounding)

    def _get_constraints(
        self,
        assets: list[str],
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict:
        """
        The weight constraints of the optimizers: the per-asset bounds as given and the
        sector and country bounds with the sector and country of each asset from the
        company profiles, which are only retrieved when a sector or country is bounded.
        Countries can be named in full or by their two-letter code.
        """
        bounds = {
            classification: group_bounds
            for classification, group_bounds in (
                ("Sector", sector_bounds),
                ("Country", country_bounds),
            )
            if group_bounds
        }
        groups = None
        if bounds:
            groups = pd.DataFrame(
                {
                    classification: self._get_classification(
                        assets,
                        classification,
                        "bound the assets one by one with minimum_weight and "
                        "maximum_weight",
                    )
                    for classification in bounds
                }
            )
            if "Country" in bounds:
                from financetoolkit.economics.helpers import (  # noqa: PLC0415
                    COUNTRY_CODES,
                )

                bounds["Country"] = {
                    COUNTRY_CODES.get(country, country): bound
                    for country, bound in bounds["Country"].items()
                }
        return {
            "minimum_weight": minimum_weight,
            "maximum_weight": maximum_weight,
            "groups": groups,
            "group_bounds": bounds or None,
        }

    def _add_hierarchical_risk(
        self,
        table: pd.DataFrame,
        weights: pd.Series,
        returns: pd.DataFrame,
        risk_measure: str,
        alpha: float,
        minimum_acceptable_return: float | str,
        rounding: int | None,
    ) -> pd.DataFrame:
        """
        The table of a hierarchical optimizer with a column of its risk measure per asset
        and for the portfolio, when that is a measure of the returns rather than the
        variance or volatility the table already shows.
        """
        if risk_measure in ("variance", "volatility"):
            return table
        table[optimization_model.HIERARCHICAL_RISK_MEASURES[risk_measure]] = (
            optimization_model.get_mean_risk_measure(
                returns.assign(**{PORTFOLIO_TICKER: returns @ weights}),
                risk_measure,
                alpha,
                minimum_acceptable_return,
            )
        )
        return self._round(table, rounding)

    @handle_errors
    def get_minimum_variance_weights(
        self,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        l2_regularization: float = 0.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the minimum variance portfolio: the fully invested portfolio of
        the Toolkit's assets with the lowest volatility.

        It uses only the covariance matrix, not the expected returns, which makes it the most
        robust of the mean-variance portfolios to estimation error.

        The weights minimize w'Σw subject to the weights summing to one and lying between
        the minimum and maximum weight, where Σ is the annualized covariance matrix of the
        returns.

        An L2 regularization γ adds the penalty γ w'w, so the weights minimize
        w'(Σ + γI)w, a ridge on the weights. With the weights summing to one, w'w is the
        squared distance from the equal weights plus 1/N, so a higher γ moves the
        portfolio towards equal weights, which tempers the estimation error in Σ.

        For more information, see the following papers:

        - Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.
        - DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009). "A Generalized
          Approach to Portfolio Optimization: Improving Performance by Constraining
          Portfolio Norms." Management Science, 55(5), 798-812.
        - Hoerl, A. E. and Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for
          Nonorthogonal Problems." Technometrics, 12(1), 55-67.

        Also known as: global minimum variance portfolio, GMV, lowest risk portfolio.

        Referenced against:
        - PyPortfolioOpt, EfficientFrontier.min_volatility: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.min_volatility
        - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg
        - skfolio, MeanRisk: https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

        Args:
            period (str, optional): The frequency of the returns the covariance matrix is estimated from. Options
                are: "daily" (daily returns), "weekly" (weekly returns), "monthly" (monthly returns), "quarterly"
                (quarterly returns), "yearly" (yearly returns). Defaults to "daily".
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
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). At 0 the portfolio with the lowest variance holds nothing,
                so all weights are 0. The rest, 1 minus the net exposure, is a Cash row at
                the risk free rate (negative when borrowed above one), which the Portfolio's
                Expected Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover or transaction cost needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
                the weights, in annualized variance units, which pulls the portfolio
                towards equal weights and spreads it over more assets, e.g. 0.05. It is
                PyPortfolioOpt's L2_reg gamma on an annualized covariance matrix; on a
                covariance matrix per period, that gamma is γ divided by the number of
                periods in a year. Defaults to 0, no penalty.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution (the share of the portfolio's
            volatility). Returns and volatilities are annualized.

        Notes:
        - Only the periods in which every asset has a return are used, so an asset with a
          short history shortens the sample of all assets.
        - The expected returns are the historical average returns, annualized; the Sharpe
          ratios use the average risk free rate over the same periods.
        - Every optimizer with weight bounds takes the same constraints. Sector and
          country bounds limit the total weight of the assets in a group (an asset
          without a sector or country, such as the benchmark, is in none). A gross
          exposure limit is linearized with one extra variable t_i ≥ |w_i| per asset
          (t_i ≥ w_i, t_i ≥ -w_i, Σ t_i ≤ the limit), so the solution stays exact. A
          maximum turnover from the current weights is linearized the same way, with one
          variable d_i ≥ |w_i - w0_i| per asset and Σ d_i ≤ the limit (Lobo, Fazel and
          Boyd, 2007). A combination of constraints no portfolio can meet raises an
          error before the optimization runs.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_minimum_variance_weights(
            sector_bounds={"Healthcare": (0, 0.4)}, maximum_weight={"XOM": 0.15}
        )
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.1225 |            0.1932 |       0.2786 |         0.5725 |              0.1683 |
        | MSFT      |   0.3275 |            0.2017 |       0.2571 |         0.6534 |              0.4499 |
        | JNJ       |   0.4    |            0.0987 |       0.1673 |         0.3884 |              0.2624 |
        | XOM       |   0.15   |            0.2911 |       0.2702 |         0.9524 |              0.1194 |
        | Portfolio |   1      |            0.1729 |       0.1463 |         0.9513 |              1      |
        """
        _, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        current = self._get_current_weights(
            current_weights,
            covariance_matrix.index.tolist(),
            maximum_turnover is not None,
        )
        weights = optimization_model.get_minimum_variance_weights(
            covariance_matrix,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
            l2_regularization=l2_regularization,
        )
        table = self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_maximum_sharpe_weights(
        self,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        transaction_cost: float = 0.0,
        l2_regularization: float = 0.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the maximum Sharpe ratio portfolio.

        It is the fully invested portfolio with the highest expected return per unit of
        volatility above the risk free rate. It is the tangency portfolio, where a line from the
        risk free rate touches the efficient frontier.

        The weights maximize (w'μ - r_f) / sqrt(w'Σw) subject to the weights summing to one
        and lying between the minimum and maximum weight, with μ the annualized expected
        returns, Σ the annualized covariance matrix and r_f the risk free rate.

        An L2 regularization γ adds the penalty γ w'w to the variance in the denominator,
        (w'μ - r_f) / sqrt(w'(Σ + γI)w), which moves the portfolio towards equal weights.

        For more information, see the following papers:

        - Tobin, J. (1958). "Liquidity Preference as Behavior Towards Risk." The Review of
          Economic Studies, 25(2), 65-86.
        - Sharpe, W.F. (1966). "Mutual Fund Performance." The Journal of Business, 39(1),
          119-138.
        - DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009). "A Generalized
          Approach to Portfolio Optimization: Improving Performance by Constraining
          Portfolio Norms." Management Science, 55(5), 798-812.
        - Hoerl, A. E. and Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for
          Nonorthogonal Problems." Technometrics, 12(1), 55-67.

        Also known as: tangency portfolio, optimal risky portfolio, max Sharpe.

        Referenced against:
        - PyPortfolioOpt, EfficientFrontier.max_sharpe: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.max_sharpe
        - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg
        - skfolio, MeanRisk (with transaction costs): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

        Args:
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
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). The rest, 1 minus the net exposure, is a Cash row at the
                risk free rate (negative when borrowed above one), which the Portfolio's
                Expected Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover or transaction cost needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            transaction_cost (float, optional): The cost per unit of weight traded
                from the current weights, e.g. 0.001 for 10 basis points, subtracted once
                from the portfolio's annualized expected return. It adds a Transaction
                Cost column, the cost times |w_i - w0_i| per asset and their sum for the
                portfolio, whose Expected Return and Sharpe Ratio are after it. Defaults
                to 0.
            l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
                the weights, in annualized variance units, which pulls the portfolio
                towards equal weights and spreads it over more assets, e.g. 0.05. It is
                PyPortfolioOpt's L2_reg gamma on an annualized covariance matrix; on a
                covariance matrix per period, that gamma is γ divided by the number of
                periods in a year. Defaults to 0, no penalty.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution. Returns and volatilities are
            annualized.

        Notes:
        - The expected returns are the noisiest input of the optimization, so with the
          historical averages the weights tend to concentrate in the assets that happened
          to do best. A maximum weight, the CAPM or shrunk (Bayes-Stein, James-Stein,
          Bodnar-Okhrin-Parolya) expected returns or the Black-Litterman returns temper
          that.
        - A transaction cost lowers the expected return of the portfolio by the cost
          times the turnover from the current weights, so the optimizer only trades when
          the gain in Sharpe ratio pays for it. The cost
          is taken once from an annualized return, which spreads it over a year of
          holding the portfolio. The Portfolio row shows the expected return and Sharpe
          ratio after the cost, the ones the optimizer maximized.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_maximum_sharpe_weights(maximum_weight=0.4)
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.0293 |            0.1933 |       0.2786 |         0.5727 |              0.0287 |
        | MSFT      |   0.3129 |            0.2017 |       0.2572 |         0.6532 |              0.3207 |
        | JNJ       |   0.2578 |            0.0984 |       0.1673 |         0.3867 |              0.1137 |
        | XOM       |   0.4    |            0.291  |       0.2702 |         0.9522 |              0.5369 |
        | Portfolio |   1      |            0.2105 |       0.1568 |         1.1275 |              1      |
        """
        _, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        current = self._get_current_weights(
            current_weights,
            covariance_matrix.index.tolist(),
            maximum_turnover is not None or transaction_cost > 0,
        )
        weights = optimization_model.get_maximum_sharpe_weights(
            expected,
            covariance_matrix,
            risk_free_rate,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
            l2_regularization=l2_regularization,
            transaction_cost=transaction_cost,
        )
        return self._weights_table(
            weights,
            expected,
            covariance_matrix,
            risk_free_rate,
            rounding,
            current,
            transaction_cost,
        )

    @handle_errors
    def get_mean_variance_weights(
        self,
        target_return: float | None = None,
        risk_aversion: float = 2.5,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        transaction_cost: float = 0.0,
        target_volatility: float | None = None,
        l2_regularization: float = 0.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of a mean-variance efficient portfolio.

        With a target return, it is the portfolio with the lowest volatility that expects that
        return. With a target volatility σ*, it is the portfolio with the highest expected
        return whose annualized volatility is at most σ*: max w'μ subject to w'Σw ≤ σ*². Without
        either, it is the portfolio that maximizes the investor's utility, the expected return
        minus half the risk aversion times the variance: w'μ - (λ/2) w'Σw.

        A risk aversion of 2 to 4 is typical; 2.5 is the value commonly used for the
        market as a whole.

        An L2 regularization γ adds the penalty γ w'w, which moves the portfolio towards
        equal weights: with a target return the weights minimize w'(Σ + γI)w, the utility
        becomes w'μ - (λ/2) w'Σw - γ w'w and with a target volatility the weights
        maximize w'μ - γ w'w, with the volatility limit still on w'Σw.

        For more information, see the following papers:

        - Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.
        - Merton, R. C. (1972). "An Analytic Derivation of the Efficient Portfolio
          Frontier." Journal of Financial and Quantitative Analysis, 7(4), 1851-1872.
        - DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009). "A Generalized
          Approach to Portfolio Optimization: Improving Performance by Constraining
          Portfolio Norms." Management Science, 55(5), 798-812.
        - Hoerl, A. E. and Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for
          Nonorthogonal Problems." Technometrics, 12(1), 55-67.

        Also known as: Markowitz portfolio, efficient portfolio, quadratic utility
        portfolio.

        Referenced against:
        - PyPortfolioOpt, EfficientFrontier.efficient_return (target return): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.efficient_return
        - PyPortfolioOpt, EfficientFrontier.efficient_risk (target volatility): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.efficient_risk
        - PyPortfolioOpt, EfficientFrontier.max_quadratic_utility (risk aversion): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.max_quadratic_utility
        - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg

        Args:
            target_return (float | None, optional): The annualized return the portfolio must
                expect, e.g. 0.12 for 12%, with the cash of a net exposure other than one
                at the risk free rate. Defaults to None, which uses the target
                volatility or the risk aversion.
            risk_aversion (float, optional): The investor's risk aversion λ; higher values
                give less risky portfolios. Not used with a target return or target
                volatility. Defaults to 2.5.
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
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). The rest, 1 minus the net exposure, is a Cash row at the
                risk free rate (negative when borrowed above one), which the Portfolio's
                Expected Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover or transaction cost needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            transaction_cost (float, optional): The cost per unit of weight traded
                from the current weights, e.g. 0.001 for 10 basis points, subtracted once
                from the portfolio's annualized expected return. It adds a Transaction
                Cost column, the cost times |w_i - w0_i| per asset and their sum for the
                portfolio, whose Expected Return and Sharpe Ratio are after it. Defaults
                to 0.
            target_volatility (float | None, optional): The highest annualized volatility
                of the portfolio (the volatility per period times the square root of the
                number of periods in a year), e.g. 0.15 for 15%. Cannot be combined with
                a target return. Defaults to None, which uses the target return or the
                risk aversion.
            l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
                the weights, in annualized variance units, which pulls the portfolio
                towards equal weights and spreads it over more assets, e.g. 0.05. It is
                PyPortfolioOpt's L2_reg gamma on an annualized covariance matrix; on a
                covariance matrix per period, that gamma is γ divided by the number of
                periods in a year. Defaults to 0, no penalty.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution.

        Notes:
        - A target return outside what the assets can reach within the constraints
          raises an error that states the reachable range.
        - A target volatility below that of the minimum variance portfolio within the
          constraints raises an error that states that volatility. Above the volatility
          of the highest return portfolio the target does not bind, and that portfolio
          is returned.
        - A target return and a target volatility together are refused: on the efficient
          frontier the one fixes the other.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_mean_variance_weights(
            target_return=0.3, minimum_weight=-0.5, gross_exposure=1.6
        )
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.0057 |            0.1932 |       0.2786 |         0.5725 |              0.0027 |
        | MSFT      |   0.4234 |            0.2017 |       0.2571 |         0.6534 |              0.2149 |
        | JNJ       |  -0.2459 |            0.0987 |       0.1673 |         0.3884 |              0.0022 |
        | XOM       |   0.8168 |            0.2911 |       0.2702 |         0.9524 |              0.7803 |
        | Portfolio |   1      |            0.3    |       0.2521 |         1.0561 |              1      |
        """
        _, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        current = self._get_current_weights(
            current_weights,
            covariance_matrix.index.tolist(),
            maximum_turnover is not None or transaction_cost > 0,
        )
        weights = optimization_model.get_mean_variance_weights(
            expected,
            covariance_matrix,
            (
                target_return - (1 - net_exposure) * risk_free_rate
                if target_return is not None
                else None
            ),
            risk_aversion,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
            l2_regularization=l2_regularization,
            transaction_cost=transaction_cost,
            target_volatility=target_volatility,
        )
        return self._weights_table(
            weights,
            expected,
            covariance_matrix,
            risk_free_rate,
            rounding,
            current,
            transaction_cost,
        )

    @handle_errors
    def get_efficient_frontier(
        self,
        points: int = 20,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        l2_regularization: float = 0.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.

        These are the portfolios with the lowest volatility for each level of expected return,
        from the minimum variance portfolio up to the highest return the weight bounds allow.
        Every portfolio below the frontier takes more risk than needed for its return.

        An L2 regularization γ minimizes w'(Σ + γI)w for each target return, which moves
        every portfolio towards equal weights; the volatility shown stays √(w'Σw).

        For more information, see the following papers:

        - Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.
        - DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009). "A Generalized
          Approach to Portfolio Optimization: Improving Performance by Constraining
          Portfolio Norms." Management Science, 55(5), 798-812.
        - Hoerl, A. E. and Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for
          Nonorthogonal Problems." Technometrics, 12(1), 55-67.

        Also known as: Markowitz frontier, mean-variance frontier, risk-return frontier.

        Referenced against:
        - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg

        Args:
            points (int, optional): The number of portfolios on the frontier. Defaults to 20.
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
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). At 0 the portfolio with the lowest variance holds nothing,
                so the first portfolio of the frontier has all weights 0 and no Sharpe ratio
                (NaN). The rest, 1 minus the net exposure, earns the risk free rate
                (borrowed at it above one) in the Expected Return and Sharpe Ratio. Defaults
                to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover or transaction cost needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
                the weights, in annualized variance units, which pulls the portfolio
                towards equal weights and spreads it over more assets, e.g. 0.05. It is
                PyPortfolioOpt's L2_reg gamma on an annualized covariance matrix; on a
                covariance matrix per period, that gamma is γ divided by the number of
                periods in a year. Defaults to 0, no penalty.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per portfolio on the frontier (numbered from the lowest risk), its
            annualized Expected Return, Volatility and Sharpe Ratio and the weight per asset.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_efficient_frontier(points=5)
        ```

        Which returns:

        |   Portfolio |   Expected Return |   Volatility |   Sharpe Ratio |   AAPL |   MSFT |    JNJ |    XOM |
        |------------:|------------------:|-------------:|---------------:|-------:|-------:|-------:|-------:|
        |           1 |            0.1575 |       0.135  |         0.9166 | 0.0212 | 0.2166 | 0.5821 | 0.1801 |
        |           2 |            0.1909 |       0.1439 |         1.0917 | 0.0178 | 0.265  | 0.388  | 0.3292 |
        |           3 |            0.2243 |       0.1679 |         1.1349 | 0.0145 | 0.3133 | 0.194  | 0.4782 |
        |           4 |            0.2576 |       0.2016 |         1.111  | 0.0111 | 0.3616 | 0      | 0.6274 |
        |           5 |            0.291  |       0.2702 |         0.9522 | 0      | 0      | 0      | 1      |
        """
        _, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        current = self._get_current_weights(
            current_weights,
            covariance_matrix.index.tolist(),
            maximum_turnover is not None,
        )
        frontier = optimization_model.get_efficient_frontier(
            expected,
            covariance_matrix,
            risk_free_rate,
            points,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
            l2_regularization=l2_regularization,
        )
        return self._round(frontier, rounding)

    @handle_errors
    def get_risk_parity_weights(
        self,
        risk_budget: dict[str, float] | None = None,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        risk_measure: str = "volatility",
        alpha: float = 0.05,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the risk parity weights: the portfolio in which every asset contributes the
        same share of the volatility, or the share set by a risk budget.

        Volatile assets get smaller weights, and correlated assets share their risk budget, so
        no single asset dominates the portfolio's risk.

        The weights solve w_i (Σw)_i / σ_p = b_i σ_p for every asset, found by minimizing the
        convex function 1/2 w'Σw - Σ b_i ln(w_i) and scaling the result to sum to one.

        With another risk measure ρ of the historical returns, every asset contributes its
        share of that measure instead, w_i ∂ρ/∂w_i = b_i ρ(w): the solution of
        min ρ(y) subject to Σ b_i ln(y_i) ≥ 0 and y ≥ 0, scaled to w = y / Σy (Roncalli,
        2013). The measures that are linear programmes, and the Gini mean
        difference and Tail Gini measures, are solved exactly with cutting planes (Kelley,
        1960): the concave Σ b_i ln(y_i) lies below each of its tangents, which are added
        as linear constraints where the linear programme's solution needs them. The
        semideviation and the Ulcer Index are minimized with SLSQP as ρ(y) - Σ b_i ln(y_i).
        The measures are those of get_mean_risk_weights plus the Conditional Value
        at Risk (Rockafellar and Uryasev, 2000), and the Conditional Drawdown at
        Risk of the uncompounded cumulative return (Chekhlov, Uryasev and Zabarankin,
        2005), without the worst realization and the range, whose contributions come from
        a single period.

        For more information, see the following papers:

        - Maillard, S., Roncalli, T. and Teïletche, J. (2010). "The Properties of Equally
          Weighted Risk Contribution Portfolios." The Journal of Portfolio Management,
          36(4), 60-70.
        - Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights." SSRN Working
          Paper.
        - Richard, J.-C. and Roncalli, T. (2019). "Constrained Risk Budgeting Portfolios:
          Theory, Algorithms, Applications & Puzzles." arXiv:1902.05710.
        - Roncalli, T. (2013). "Introduction to Risk Parity and Budgeting." Chapman &
          Hall/CRC, Boca Raton.
        - Rockafellar, R. T. and Uryasev, S. (2000). "Optimization of Conditional
          Value-at-Risk." Journal of Risk, 2(3), 21-41.
        - Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in
          Portfolio Optimization." International Journal of Theoretical and Applied
          Finance, 8(1), 13-58.
        - Kelley, J. E. (1960). "The Cutting-Plane Method for Solving Convex Programs."
          Journal of the Society for Industrial and Applied Mathematics, 8(4), 703-712.

        Also known as: equal risk contribution, ERC, risk budgeting.

        Referenced against:
        - skfolio, RiskBudgeting (volatility): https://skfolio.org/generated/skfolio.optimization.RiskBudgeting.html
        - skfolio, RiskBudgeting (other risk measures): https://skfolio.org/generated/skfolio.optimization.RiskBudgeting.html
        - Riskfolio-Lib, Portfolio.rp_optimization (other risk measures): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.rp_optimization

        Args:
            risk_budget (dict[str, float] | None, optional): The share of the risk per
                asset, e.g. {"AAPL": 0.4, "MSFT": 0.2, ...}, positive for every asset;
                it is scaled to sum to one. A budget that names an asset outside the
                portfolio or leaves an asset out raises a ValueError that lists them.
                Defaults to None, an equal share for every asset.
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
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); it
                cannot be negative. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            risk_measure (str, optional): The risk the assets share. Options are: "volatility" (of the covariance
                estimator), "mad" (mean absolute deviation, Konno and Yamazaki, 1991), "semideviation" (below a zero
                return), "lower_partial_moment" (below a zero return), "gini_mean_difference" (Yitzhaki, 1982),
                "tail_gini" (Furman et al., 2017), "cvar_range" (the CVaR of the losses and of the gains),
                "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the uncompounded
                cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar" (Chekhlov et al., 2005). Defaults
                to "volatility".
            alpha (float, optional): The tail probability of the CVaR, CDaR, CVaR
                range and Tail Gini measures, e.g. 0.05 for the worst 5% of periods.
                Defaults to 0.05.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution (of the volatility). With
            another risk measure, also that measure (per period, positive for a loss,
            named after it, e.g. "CVaR") of every asset on its own and of the portfolio,
            and each asset's share of the portfolio's measure, e.g. "CVaR Contribution"
            (w_i ∂ρ/∂w_i / ρ(w) with central finite differences).

        Notes:
        - Weight, sector and country bounds turn it into constrained risk budgeting
          (Richard and Roncalli, 2019): the bounds are linear constraints on w = y / Σy,
          so the problem stays convex. Where a bound
          binds, the risk contributions can no longer all match the budget.
        - Risk parity holds long positions only and is fully invested, so it takes no
          negative minimum weight and no net or gross exposure.
        - The other risk measures are per period of the chosen frequency and taken of the
          historical returns, not of the covariance estimator. The CVaR, the CDaR, the
          Ulcer Index and the Tail Gini measures follow a few periods and have kinks: at
          the optimum the budget holds for one of the slopes on either side of a kink,
          while the central finite differences of the contribution column average them,
          so that column can differ from the budget by a few percent.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_risk_parity_weights(sector_bounds={"Healthcare": (0, 0.3)})
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.1991 |            0.1932 |       0.2786 |         0.5725 |              0.2763 |
        | MSFT      |   0.2388 |            0.2017 |       0.2571 |         0.6534 |              0.2816 |
        | JNJ       |   0.3    |            0.0987 |       0.1673 |         0.3884 |              0.1575 |
        | XOM       |   0.2621 |            0.2911 |       0.2702 |         0.9524 |              0.2846 |
        | Portfolio |   1      |            0.1925 |       0.1523 |         1.0429 |              1      |
        """
        validate_alpha(alpha)
        returns, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        budget = (
            pd.Series(risk_budget, dtype=float) if risk_budget is not None else None
        )
        constraints = self._get_constraints(
            covariance_matrix.index.tolist(),
            minimum_weight,
            maximum_weight,
            sector_bounds,
            country_bounds,
        )
        if risk_measure == "volatility":
            weights = optimization_model.get_risk_parity_weights(
                covariance_matrix, budget, **constraints
            )
            return self._weights_table(
                weights, expected_returns, covariance_matrix, risk_free_rate, rounding
            )

        weights = optimization_model.get_risk_budgeting_weights(
            returns, risk_measure, budget, alpha, **constraints
        )
        table = self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )
        name = optimization_model.RISK_BUDGETING_MEASURES[risk_measure]
        table[name] = optimization_model.get_mean_risk_measure(
            returns.assign(**{PORTFOLIO_TICKER: returns @ weights}),
            risk_measure,
            alpha,
        )
        contributions = optimization_model.get_mean_risk_contributions(
            weights, returns, risk_measure, alpha
        )
        contributions[PORTFOLIO_TICKER] = contributions.sum()
        table[f"{name} Contribution"] = contributions
        return self._round(table, rounding)

    @handle_errors
    def get_hierarchical_risk_parity_weights(
        self,
        linkage_method: str = "single",
        codependence: str = "pearson",
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        risk_measure: str = "variance",
        alpha: float = 0.05,
        minimum_acceptable_return: float | str = 0.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hierarchical Risk Parity (HRP) weights, which split the risk over clusters
        of similar assets without inverting the covariance matrix.

        The assets are clustered by their correlations, ordered so that similar assets sit next
        to each other, and the weight is split top-down between the two halves of every cluster
        in inverse proportion to their variance. It never inverts the covariance matrix, so it
        stays stable with many or highly correlated assets, where mean-variance weights become
        erratic.

        As in the paper, the clustering uses the Euclidean distance between the columns of
        the correlation distance matrix √((1 - ρ) / 2): two assets are close when they
        relate to all the other assets in the same way. Some libraries cluster on the
        correlation distance itself, which can order the assets differently.

        The risk of each half is that of its inverse risk portfolio, every asset weighted by
        the inverse of its own risk: the variance by default, as in the paper, or another
        risk measure of the historical returns, such as the CVaR, which splits by the
        losses in the worst periods, or the CDaR, which splits by the drawdowns. Raffinot (2017,
        2018) allocates across hierarchical clusters with such measures.

        For more information, see the following papers:

        - López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of
          Sample." The Journal of Portfolio Management, 42(4), 59-69.
        - Raffinot, T. (2017). "Hierarchical Clustering-Based Asset Allocation." The
          Journal of Portfolio Management, 44(2), 89-99.
        - Raffinot, T. (2018). "The Hierarchical Equal Risk Contribution Portfolio."
          SSRN 3237540.

        Also known as: HRP, hierarchical clustering portfolio.

        Referenced against:
        - Riskfolio-Lib, HCPortfolio (HRP): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/hcportfolio.html#HCPortfolio.HCPortfolio
        - skfolio, HierarchicalRiskParity: https://skfolio.org/generated/skfolio.optimization.HierarchicalRiskParity.html

        Shares similar logic:
        - PyPortfolioOpt, HRPOpt: https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html#pypfopt.hierarchical_portfolio.HRPOpt
        - RiskLabAI, hrp: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/optimization/hrp.py#L169

        Args:
            linkage_method (str, optional): The linkage of the hierarchical clustering. Options are: "single"
                (single linkage), "complete" (complete linkage), "average" (average linkage), "ward" (Ward's
                method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo and Aste, 2012, on the
                TMFG planar graph of the codependence, ordered by the distance itself). Defaults to "single".
            codependence (str, optional): The codependence the assets are clustered on. The rank correlations depend
                only on the order of the returns, so a few extreme returns weigh less; the
                distance correlation and mutual information also capture non-linear co-movement.
                Options are: "pearson" (the linear correlation of the covariance matrix),
                "spearman" (the rank correlation of the returns, Spearman, 1904), "kendall" (the
                rank correlation τ_b of the returns, Kendall, 1938), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "detoned" (the
                correlation without the market component, López de Prado, 2020), "distance" (the
                distance correlation, Székely et al., 2007, through √(1 - dCor)),
                "mutual_information" (through the variation of information, Meilă, 2007),
                "lower_tail" (how often two assets fall into their lowest 5% of returns
                together, through -ln λ), "absolute_angular" (|ρ| through √(1 - |ρ|), López de
                Prado, 2020), "squared_angular" (ρ² through √(1 - ρ²), López de Prado, 2020),
                "gnpr" (the rank co-movement and the difference between the return
                distributions, θ = 0.5, Marti et al., 2016). Defaults to "pearson".
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
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); it
                cannot be negative. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            risk_measure (str, optional): The risk measure of the splits. Options are: "variance" (of the covariance
                matrix), "volatility" (of the covariance matrix), "mad" (mean absolute deviation, Konno and
                Yamazaki, 1991), "semideviation" (below the minimum acceptable return), "lower_partial_moment"
                (first order), "worst_realization" (the worst return), "gini_mean_difference" (Yitzhaki, 1982),
                "tail_gini" (Furman et al., 2017), "range" (the best minus the worst return), "cvar_range" (the CVaR
                of the losses and of the gains), "tail_gini_range" (the Tail Gini of the losses and of the gains),
                "ulcer_index" (of the uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002),
                "cdar" (Chekhlov et al., 2005). Defaults to "variance".
            alpha (float, optional): The tail probability of the CVaR, CDaR, CVaR
                range and Tail Gini measures, e.g. 0.05 for the worst 5% of periods.
                Defaults to 0.05.
            minimum_acceptable_return (float | str, optional): The threshold of the semideviation and the lower
                partial moment, as a return per period of the chosen frequency, or "mean" for each asset's average
                return. Defaults to 0.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, and with a risk measure of the
            returns, that measure per period (e.g. "CVaR").

        Notes:
        - Weight bounds move each split just enough to keep both halves within the sum of
          their assets' bounds.
        - The risk measures of the returns are per period of the chosen frequency and not
          annualized, and only their ratios between the halves matter. They equal the weights of
          Riskfolio-Lib's HCPortfolio and skfolio's HierarchicalRiskParity with the same risk
          measure on the same asset order; those cluster on the correlation distance itself
          rather than the distance between its columns, which can order the assets differently.
        - A split only moves weight between the two halves of a cluster, so sector and
          country bounds, which cut across the clusters, do not apply; neither do short
          positions or a net or gross exposure, as HRP is long only and fully invested.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_hierarchical_risk_parity_weights(maximum_weight={"JNJ": 0.3})
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.1287 |            0.1932 |       0.2786 |         0.5725 |              0.1307 |
        | MSFT      |   0.1511 |            0.2017 |       0.2571 |         0.6534 |              0.1184 |
        | JNJ       |   0.3    |            0.0987 |       0.1673 |         0.3884 |              0.1503 |
        | XOM       |   0.4201 |            0.2911 |       0.2702 |         0.9524 |              0.6006 |
        | Portfolio |   1      |            0.2073 |       0.1569 |         1.1059 |              1      |
        """
        check_option(
            "codependence", codependence, optimization_model.CODEPENDENCE_METHODS
        )
        validate_alpha(alpha)
        returns, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        weights = optimization_model.get_hierarchical_risk_parity_weights(
            covariance_matrix,
            linkage_method,
            minimum_weight,
            maximum_weight,
            distance=optimization_model.get_codependence_distance(
                returns, covariance_matrix, codependence
            ),
            returns=returns,
            risk_measure=risk_measure,
            alpha=alpha,
            minimum_acceptable_return=minimum_acceptable_return,
            similarity=(
                optimization_model.get_codependence_similarity(
                    returns, covariance_matrix, codependence
                )
                if linkage_method == "dbht"
                else None
            ),
        )
        table = self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )
        return self._add_hierarchical_risk(
            table,
            weights,
            returns,
            risk_measure,
            alpha,
            minimum_acceptable_return,
            rounding,
        )

    @handle_errors
    def get_schur_complementary_weights(
        self,
        gamma: float = 0.5,
        linkage_method: str = "single",
        codependence: str = "pearson",
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Schur Complementary Allocation weights, which run from Hierarchical Risk
        Parity (γ = 0) towards the minimum variance portfolio (γ near 1).

        HRP splits every cluster between its two halves by their own variances and ignores how
        the two halves move together; the minimum variance portfolio uses exactly that, through
        the Schur complements of the inverse covariance matrix, but becomes unstable when the
        covariance matrix is noisy. Schur Complementary Allocation augments the covariance block
        of each half with a share γ of what the other half tells about it, and then splits as
        HRP does on the augmented blocks.

        For a cluster split into halves with covariance blocks A and D and the block B
        between them, the formula is as follows:

        - A_c = A - γ B D⁻¹ B'
        - Ã = (I - γ B D⁻¹ M')⁻¹ A_c, made symmetric, with M the step-up matrix that spreads
          the assets of the other half over those of this one (the identity for halves of
          equal size), so that Ã⁻¹1 = A_c⁻¹(1 - γ B D⁻¹ M'1), the minimum variance
          weights of the half up to a scale.
        - α = 1 - ν(Ã) / (ν(Ã) + ν(D̃)), the share of the first half, with ν the
          variance of the inverse variance portfolio of a block.

        The halves are then split further on Ã and D̃. The assets are clustered and
        ordered as in Hierarchical Risk Parity, so γ = 0 gives its weights.

        For more information, see the following papers:

        - Cotton, P. (2024). "Schur Complementary Allocation: A Unification of
          Hierarchical Risk Parity and Minimum Variance Portfolios." arXiv 2411.05807.
        - López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of
          Sample." The Journal of Portfolio Management, 42(4), 59-69.

        Also known as: Schur complementary portfolio, Schur allocation.

        Referenced against:
        - skfolio, SchurComplementary: https://skfolio.org/generated/skfolio.optimization.SchurComplementary.html

        Args:
            gamma (float, optional): How much of the covariance between the halves of a
                cluster the splits use, between 0 (Hierarchical Risk Parity) and 1.
                Defaults to 0.5.
            linkage_method (str, optional): The linkage of the hierarchical clustering, as in Hierarchical Risk
                Parity. Options are: "single" (single linkage), "complete" (complete linkage), "average" (average
                linkage), "ward" (Ward's method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo
                and Aste, 2012). Defaults to "single".
            codependence (str, optional): The codependence the assets are clustered on (see
                get_hierarchical_risk_parity_weights). Defaults to. Options are: "pearson" (the
                linear correlation of the covariance matrix), "spearman" (the rank correlation
                of the returns, Spearman, 1904), "kendall" (the rank correlation τ_b of the
                returns, Kendall, 1938), "gerber" (co-movements beyond half a standard
                deviation, Gerber et al., 2022), "detoned" (the correlation without the market
                component, López de Prado, 2020), "distance" (the distance correlation, Székely
                et al., 2007, through √(1 - dCor)), "mutual_information" (through the variation
                of information, Meilă, 2007), "lower_tail" (how often two assets fall into their
                lowest 5% of returns together, through -ln λ), "absolute_angular" (|ρ| through
                √(1 - |ρ|), López de Prado, 2020), "squared_angular" (ρ² through √(1 - ρ²),
                López de Prado, 2020), "gnpr" (the rank co-movement and the difference between
                the return distributions, θ = 0.5, Marti et al., 2016). Defaults to "pearson".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix (see
                get_hierarchical_risk_parity_weights). A shrinkage estimator such as "ledoit_wolf" keeps the
                augmented blocks positive definite at a higher gamma. Defaults to. Options are: "sample" (the sample
                covariance, ddof=1), "ledoit_wolf" (shrunk towards the scaled identity, Ledoit and Wolf, 2004),
                "ledoit_wolf_constant_correlation" (shrunk towards the average correlation, Ledoit and Wolf, 2004),
                "ledoit_wolf_single_factor" (shrunk towards the covariance that the benchmark as a single factor
                implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating Shrinkage, Chen et al., 2010), "shrunk"
                (a fixed 10% towards the scaled identity), "semicovariance" (the co-movement below a zero return,
                Estrada, 2008), "exponential" (exponentially weighted, recent periods weigh most, RiskMetrics),
                "denoised" (the eigenvalues of the correlation matrix that random returns could produce averaged,
                Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to zero), "denoised_shrink" (the
                noise correlations shrunk towards their diagonal, López de Prado, 2020), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "factor" (the covariance of the Fama and
                French five factor model with OLS loadings, Fama and French, 1993), "minimum_covariance_determinant"
                (the reweighted covariance of the half of the periods with the smallest covariance determinant,
                robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the sparse inverse covariance on the
                TMFG planar graph of the correlations, Barfuss et al., 2016), "geodesic" (10% towards the scaled
                identity along the geodesic of the positive definite matrices, Bhatia, 2007), "regime_adjusted_ew"
                (exponentially weighted with a half-life of 40 periods, rescaled by the recent realized against
                predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one part of the periods and the
                variances along them from the other, averaged over random splits, Lam, 2016), "ccc_garch" (the
                next-period covariance of GARCH(1, 1) volatilities with constant correlations, Bollerslev, 1990),
                "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke" (Rocke's robust S-estimator, Maronna
                et al., 2019). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); it
                cannot be negative. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution. Returns and volatilities are
            annualized.

        Notes:
        - Weight bounds move each split just enough to keep both halves within the sum of
          their assets' bounds, as in Hierarchical Risk Parity; the method is long only
          and fully invested.
        - The portfolio's variance need not fall as gamma rises: the augmented blocks can
          make a split worse when the covariance matrix is poorly conditioned. skfolio
          by default caps gamma where the variance stops falling;
          here gamma is used as given, so compare the Volatility with that of a lower
          gamma. When an augmented block is not positive definite, the method returns an
          error that asks for a lower gamma or a shrinkage covariance estimator.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_schur_complementary_weights(gamma=0.5)
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.1146 |            0.1932 |       0.2786 |         0.5725 |              0.136  |
        | MSFT      |   0.1448 |            0.2017 |       0.2571 |         0.6534 |              0.1365 |
        | JNJ       |   0.5231 |            0.0987 |       0.1673 |         0.3884 |              0.479  |
        | XOM       |   0.2175 |            0.2911 |       0.2702 |         0.9526 |              0.2485 |
        | Portfolio |   1      |            0.1663 |       0.1372 |         0.9663 |              1      |
        """
        check_option(
            "codependence", codependence, optimization_model.CODEPENDENCE_METHODS
        )
        returns, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        weights = optimization_model.get_schur_complementary_weights(
            covariance_matrix,
            gamma,
            linkage_method,
            minimum_weight,
            maximum_weight,
            distance=optimization_model.get_codependence_distance(
                returns, covariance_matrix, codependence
            ),
            similarity=(
                optimization_model.get_codependence_similarity(
                    returns, covariance_matrix, codependence
                )
                if linkage_method == "dbht"
                else None
            ),
        )
        return self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )

    @handle_errors
    def get_stacking_weights(
        self,
        methods: str | list[str] | None = None,
        final_method: str = "minimum_variance",
        folds: int | None = 5,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of a stacked portfolio.

        Several optimizers each give their portfolio, and a final optimizer divides the weight
        over those portfolios as if they were assets, judged by the returns each earns out of
        sample. Each optimizer has its own strengths (inverse volatility is robust, maximum
        diversification spreads the risk, HRP avoids inverting the covariance matrix, minimum
        CVaR guards the tail); stacking combines them by how they hold up rather than choosing
        one.

        The formula is as follows:

        - w = W v, with W the N × K matrix whose column k holds the weights of method k
          optimized on all periods, and v the weights the final method gives the K
          portfolios.
        - The final method is optimized on the returns of the K portfolios. With folds,
          the periods are cut into that many consecutive blocks; for each block the
          methods are optimized on the other periods and their portfolios earn the
          block's returns, so every return the final method sees is out of sample (a
          K-fold cross-validation without shuffling). Without folds it uses the
          in-sample returns R W, which favours the methods that fit the past best.

        For more information, see the following sources:

        - Wolpert, D. H. (1992). "Stacked Generalization." Neural Networks, 5(2),
          241-259.

        Also known as: stacking, stacked generalization, ensemble optimization.

        Referenced against:
        - skfolio, StackingOptimization: https://skfolio.org/generated/skfolio.optimization.StackingOptimization.html

        Args:
            methods (str | list[str] | None, optional): The optimizers to combine, a list or a comma-separated
                string, each with its default settings. Options are: "minimum_variance" (Minimum Variance),
                "maximum_sharpe" (Maximum Sharpe), "risk_parity" (Risk Parity), "inverse_volatility" (Inverse
                Volatility), "maximum_diversification" (Maximum Diversification), "hierarchical_risk_parity"
                (Hierarchical Risk Parity), "hierarchical_equal_risk_contribution" (Hierarchical Equal Risk
                Contribution), "nested_clustered_optimization" (Nested Clustered Optimization),
                "schur_complementary" (Schur Complementary), "minimum_cvar" (Minimum CVaR), "minimum_drawdown"
                (Minimum Drawdown), "minimum_mad" (Minimum MAD), "minimum_semideviation" (Minimum Semideviation),
                "equal_weight" (Equal Weight). Defaults to None, which uses "inverse_volatility",
                "maximum_diversification", "hierarchical_risk_parity" and "minimum_cvar".
            final_method (str, optional): The optimizer that weights the methods' portfolios. Options are:
                "minimum_variance" (Minimum Variance), "maximum_sharpe" (Maximum Sharpe), "risk_parity" (Risk
                Parity), "inverse_volatility" (Inverse Volatility), "maximum_diversification" (Maximum
                Diversification), "hierarchical_risk_parity" (Hierarchical Risk Parity),
                "hierarchical_equal_risk_contribution" (Hierarchical Equal Risk Contribution),
                "nested_clustered_optimization" (Nested Clustered Optimization), "schur_complementary" (Schur
                Complementary), "minimum_cvar" (Minimum CVaR), "minimum_drawdown" (Minimum Drawdown), "minimum_mad"
                (Minimum MAD), "minimum_semideviation" (Minimum Semideviation), "equal_weight" (Equal Weight).
                Defaults to "minimum_variance".
            folds (int | None, optional): The number of consecutive blocks of the
                cross-validation, at least 2, or None to use the in-sample returns.
                Defaults to 5.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix of the methods that use one (see
                get_minimum_variance_weights). Defaults to. Options are: "sample" (the sample covariance, ddof=1),
                "ledoit_wolf" (shrunk towards the scaled identity, Ledoit and Wolf, 2004),
                "ledoit_wolf_constant_correlation" (shrunk towards the average correlation, Ledoit and Wolf, 2004),
                "ledoit_wolf_single_factor" (shrunk towards the covariance that the benchmark as a single factor
                implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating Shrinkage, Chen et al., 2010), "shrunk"
                (a fixed 10% towards the scaled identity), "semicovariance" (the co-movement below a zero return,
                Estrada, 2008), "exponential" (exponentially weighted, recent periods weigh most, RiskMetrics),
                "denoised" (the eigenvalues of the correlation matrix that random returns could produce averaged,
                Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to zero), "denoised_shrink" (the
                noise correlations shrunk towards their diagonal, López de Prado, 2020), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "factor" (the covariance of the Fama and
                French five factor model with OLS loadings, Fama and French, 1993), "minimum_covariance_determinant"
                (the reweighted covariance of the half of the periods with the smallest covariance determinant,
                robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the sparse inverse covariance on the
                TMFG planar graph of the correlations, Barfuss et al., 2016), "geodesic" (10% towards the scaled
                identity along the geodesic of the positive definite matrices, Bhatia, 2007), "regime_adjusted_ew"
                (exponentially weighted with a half-life of 40 periods, rescaled by the recent realized against
                predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one part of the periods and the
                variances along them from the other, averaged over random splits, Lam, 2016), "ccc_garch" (the
                next-period covariance of GARCH(1, 1) volatilities with constant correlations, Bollerslev, 1990),
                "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke" (Rocke's robust S-estimator, Maronna
                et al., 2019). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
            expected_returns (str, optional): The estimator of the expected returns of "maximum_sharpe" (see
                get_maximum_sharpe_weights). Defaults to. Options are: "mean" (the historical average), "geometric"
                (the compounded average), "exponential" (the exponentially weighted average, recent periods weigh
                most), "exponential_geometric" (that average compounded over a year), "capm" (the returns the
                Capital Asset Pricing Model implies from each asset's beta to the benchmark, Sharpe, 1964),
                "bayes_stein" (the averages shrunk towards that of the minimum variance portfolio, Jorion, 1986),
                "james_stein" (the averages shrunk towards their grand mean, James and Stein, 1961),
                "bodnar_okhrin_parolya" (the averages and their grand mean combined with optimal weights, Bodnar,
                Okhrin and Parolya, 2019). Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the exponentially weighted expected
                returns. Defaults to 180.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution of the stacked portfolio, and a
            column per method with its weights, whose Portfolio row is the method's weight
            in the stacked portfolio. Returns and volatilities are annualized.

        Notes:
        - Every method and the final method are long only and fully invested and use their
          default settings; the risk free rate of the maximum Sharpe ratio is the Toolkit's
          average over the periods it is optimized on.
        - skfolio's StackingOptimization uses 5-fold cross-validation by default as well;
          its blocks follow scikit-learn's KFold, the same consecutive blocks with the
          first ones one period longer when the periods do not divide evenly.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_stacking_weights()[
            [
                "Weight",
                "Maximum Diversification",
                "Hierarchical Risk Parity",
                "Minimum CVaR",
            ]
        ]
        ```

        Which returns:

        |           |   Weight |   Maximum Diversification |   Hierarchical Risk Parity |   Minimum CVaR |
        |:----------|---------:|--------------------------:|---------------------------:|---------------:|
        | AAPL      |   0.1287 |                    0.0922 |                     0.1287 |         0      |
        | MSFT      |   0.1511 |                    0.2349 |                     0.1511 |         0.1823 |
        | JNJ       |   0.5207 |                    0.4195 |                     0.5207 |         0.5968 |
        | XOM       |   0.1995 |                    0.2534 |                     0.1995 |         0.2209 |
        | Portfolio |   1      |                    0      |                     1      |         0      |
        """
        if isinstance(methods, str):
            methods = [
                method.strip() for method in methods.split(",") if method.strip()
            ]
        methods = list(methods or DEFAULT_STACKING_METHODS)
        for method in methods:
            check_option("methods", method, STACKING_METHODS)
        check_option("final_method", final_method, STACKING_METHODS)
        returns, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        benchmark = (
            self._get_benchmark_returns(period)
            if expected_returns == "capm" or covariance == "ledoit_wolf_single_factor"
            else None
        )
        factors = self._get_factor_returns(period) if covariance == "factor" else None

        optimizer = functools.partial(
            optimization_model.get_stacking_method_weights,
            risk_free_rate=functools.partial(self._get_risk_free_rate, period),
            periods_per_year=self._get_periods_per_year(period, returns.index),
            covariance=covariance,
            covariance_span=covariance_span,
            expected_returns=expected_returns,
            expected_returns_span=expected_returns_span,
            benchmark=benchmark,
            factors=factors,
        )
        weights, method_weights, final_weights = (
            optimization_model.get_stacking_weights(
                returns,
                {
                    STACKING_METHODS[method]: functools.partial(
                        optimizer, method=method
                    )
                    for method in methods
                },
                functools.partial(optimizer, method=final_method),
                folds,
            )
        )
        table = self._weights_table(
            weights, expected, covariance_matrix, risk_free_rate, rounding
        )
        for name in method_weights.columns:
            table[name] = method_weights[name]
            table.loc[PORTFOLIO_TICKER, name] = final_weights[name]
        return self._round(table, rounding)

    @handle_errors
    def get_hierarchical_equal_risk_contribution_weights(
        self,
        linkage_method: str = "ward",
        clusters: int | None = None,
        cluster_criterion: str = "two_difference_gap",
        codependence: str = "pearson",
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        risk_measure: str = "variance",
        alpha: float = 0.05,
        minimum_acceptable_return: float | str = 0.0,
        within_clusters: str = "inverse_risk",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hierarchical Equal Risk Contribution (HERC) weights, which split the
        weight top-down over clusters of similar assets by their risk.

        The assets are clustered by their correlations and the dendrogram is cut into clusters.
        Within a cluster every asset gets a weight in inverse proportion to its variance;
        between clusters the weight is split top-down along the dendrogram, each branch getting
        a weight in inverse proportion to its risk, the sum of the variances of its clusters.
        Unlike Hierarchical Risk Parity, which halves the ordered list of assets, the splits
        follow the tree, so assets that belong together are never split up, and the weight is
        allocated over the clusters rather than over the assets: riskier clusters get less.

        The clustering uses the correlation distance √((1 - ρ) / 2) with Ward linkage, as
        in the paper. Without a number of clusters, it is the one at the sharpest bend of
        the pooled within-cluster dispersion, the two-order difference of the gap
        statistic (Yue, Wang and Wei, 2008; Tibshirani, Walther and Hastie, 2001), or
        with the standardized silhouette the one whose clusters have the
        highest mean over standard deviation of the silhouette coefficients (López de
        Prado, 2020; Rousseeuw, 1987).

        The risk of a cluster is that of its inverse risk portfolio: the variance by default,
        or, as in the paper, another risk measure of the historical returns, such as the CVaR or
        the CDaR. With equal weights within the clusters, the assets of a cluster share its
        weight equally while the clusters are still split by the risk of their inverse risk
        portfolios, the HERC2 variant of Cajas.

        For more information, see the following papers:

        - Raffinot, T. (2018). "The Hierarchical Equal Risk Contribution Portfolio."
          SSRN 3237540.
        - Raffinot, T. (2017). "Hierarchical Clustering-Based Asset Allocation." The
          Journal of Portfolio Management, 44(2), 89-99.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge
          University Press.
        - Rousseeuw, P. J. (1987). "Silhouettes: A Graphical Aid to the Interpretation and
          Validation of Cluster Analysis." Journal of Computational and Applied
          Mathematics, 20, 53-65.

        Also known as: HERC, hierarchical clustering-based risk parity.

        Referenced against:
        - skfolio, HierarchicalEqualRiskContribution: https://skfolio.org/generated/skfolio.optimization.HierarchicalEqualRiskContribution.html
        - Riskfolio-Lib, HCPortfolio (HERC and HERC2): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/hcportfolio.html#HCPortfolio.HCPortfolio

        Args:
            linkage_method (str, optional): The linkage of the hierarchical clustering; Ward's method is the
                paper's. Options are: "single" (single linkage), "complete" (complete linkage), "average" (average
                linkage), "ward" (Ward's method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo
                and Aste, 2012, whose own clusters are used without a number of clusters). Defaults to "ward".
            clusters (int | None, optional): The number of clusters. Defaults to None,
                chosen by the two-order difference of the gap statistic.
            cluster_criterion (str, optional): How the number of clusters is chosen when it is not given. Options
                are: "two_difference_gap" (the sharpest bend of the within-cluster dispersion, Yue, Wang and Wei,
                2008), "standardized_silhouette" (the highest mean over standard deviation of the silhouette
                coefficients, López de Prado, 2020). Defaults to "two_difference_gap".
            codependence (str, optional): The codependence the assets are clustered on. The rank correlations depend
                only on the order of the returns, so a few extreme returns weigh less; the
                distance correlation and mutual information also capture non-linear co-movement.
                Options are: "pearson" (the linear correlation of the covariance matrix),
                "spearman" (the rank correlation of the returns, Spearman, 1904), "kendall" (the
                rank correlation τ_b of the returns, Kendall, 1938), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "detoned" (the
                correlation without the market component, López de Prado, 2020), "distance" (the
                distance correlation, Székely et al., 2007, through √(1 - dCor)),
                "mutual_information" (through the variation of information, Meilă, 2007),
                "lower_tail" (how often two assets fall into their lowest 5% of returns
                together, through -ln λ), "absolute_angular" (|ρ| through √(1 - |ρ|), López de
                Prado, 2020), "squared_angular" (ρ² through √(1 - ρ²), López de Prado, 2020),
                "gnpr" (the rank co-movement and the difference between the return
                distributions, θ = 0.5, Marti et al., 2016). Defaults to "pearson".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix, which also gives the correlations
                that are clustered. The shrinkage and denoised estimators stabilize the weights when there are many
                assets compared to the number of periods. Options are: "sample" (the sample covariance, ddof=1),
                "ledoit_wolf" (shrunk towards the scaled identity, Ledoit and Wolf, 2004),
                "ledoit_wolf_constant_correlation" (shrunk towards the average correlation, Ledoit and Wolf, 2004),
                "ledoit_wolf_single_factor" (shrunk towards the covariance that the benchmark as a single factor
                implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating Shrinkage, Chen et al., 2010), "shrunk"
                (a fixed 10% towards the scaled identity), "semicovariance" (the co-movement below a zero return,
                Estrada, 2008), "exponential" (exponentially weighted, recent periods weigh most, RiskMetrics),
                "denoised" (the eigenvalues of the correlation matrix that random returns could produce averaged,
                Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to zero), "denoised_shrink" (the
                noise correlations shrunk towards their diagonal, López de Prado, 2020), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "factor" (the covariance of the Fama and
                French five factor model with OLS loadings, Fama and French, 1993), "minimum_covariance_determinant"
                (the reweighted covariance of the half of the periods with the smallest covariance determinant,
                robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the sparse inverse covariance on the
                TMFG planar graph of the correlations, Barfuss et al., 2016), "geodesic" (10% towards the scaled
                identity along the geodesic of the positive definite matrices, Bhatia, 2007), "regime_adjusted_ew"
                (exponentially weighted with a half-life of 40 periods, rescaled by the recent realized against
                predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one part of the periods and the
                variances along them from the other, averaged over random splits, Lam, 2016), "ccc_garch" (the
                next-period covariance of GARCH(1, 1) volatilities with constant correlations, Bollerslev, 1990),
                "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke" (Rocke's robust S-estimator, Maronna
                et al., 2019). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); it
                cannot be negative. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            risk_measure (str, optional): The risk measure of the clusters. Options are: "variance" (of the
                covariance matrix), "volatility" (of the covariance matrix), "mad" (mean absolute deviation, Konno
                and Yamazaki, 1991), "semideviation" (below the minimum acceptable return), "lower_partial_moment"
                (first order), "worst_realization" (the worst return), "gini_mean_difference" (Yitzhaki, 1982),
                "tail_gini" (Furman et al., 2017), "range" (the best minus the worst return), "cvar_range" (the CVaR
                of the losses and of the gains), "tail_gini_range" (the Tail Gini of the losses and of the gains),
                "ulcer_index" (of the uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002),
                "cdar" (Chekhlov et al., 2005). Defaults to "variance".
            alpha (float, optional): The tail probability of the CVaR, CDaR, CVaR
                range and Tail Gini measures, e.g. 0.05 for the worst 5% of periods.
                Defaults to 0.05.
            minimum_acceptable_return (float | str, optional): The threshold of the semideviation and the lower
                partial moment, as a return per period of the chosen frequency, or "mean" for each asset's average
                return. Defaults to 0.
            within_clusters (str, optional): The weights within a cluster. Options are: "inverse_risk" (inverse to
                each asset's risk), "equal" (equal weights, HERC2). Defaults to "inverse_risk".
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with a risk measure of the
            returns that measure per period (e.g. "CDaR"), and the Cluster of every asset
            (numbered from 1). Returns and volatilities are annualized.

        Notes:
        - Weight, sector and country bounds are applied after the allocation: when the
          weights fall outside them, they move to the weights within the bounds with the
          smallest relative change, Σ (w_i / w_HERC,i - 1)².
        - The risk measures of the returns are per period of the chosen frequency and not
          annualized. With the same clusters, the weights equal those of Riskfolio-Lib's
          HCPortfolio and skfolio's HierarchicalEqualRiskContribution with the same risk
          measure. Riskfolio-Lib's own
          two-order difference sums the standard deviation of the distances within each
          cluster, which can pick another number of clusters.
        - HERC is long only and fully invested, so short positions and a net or gross
          exposure do not apply.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_hierarchical_equal_risk_contribution_weights()[
            ["Weight", "Volatility", "Sharpe Ratio", "Risk Contribution", "Cluster"]
        ]
        ```

        Which returns:

        |           |   Weight |   Volatility |   Sharpe Ratio |   Risk Contribution |   Cluster |
        |:----------|---------:|-------------:|---------------:|--------------------:|----------:|
        | AAPL      |   0.2917 |       0.2786 |         0.5725 |              0.412  |         1 |
        | MSFT      |   0.3425 |       0.2571 |         0.6534 |              0.4348 |         1 |
        | JNJ       |   0.2645 |       0.1673 |         0.3884 |              0.1009 |         2 |
        | XOM       |   0.1014 |       0.2702 |         0.9524 |              0.0523 |         3 |
        | Portfolio |   1      |       0.1715 |         0.8592 |              1      |       nan |
        """
        check_option(
            "codependence", codependence, optimization_model.CODEPENDENCE_METHODS
        )
        validate_alpha(alpha)
        returns, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        distance = optimization_model.get_codependence_distance(
            returns, covariance_matrix, codependence
        )
        similarity = (
            optimization_model.get_codependence_similarity(
                returns, covariance_matrix, codependence
            )
            if linkage_method == "dbht"
            else None
        )
        weights = optimization_model.get_hierarchical_equal_risk_contribution_weights(
            covariance_matrix,
            linkage_method,
            clusters,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            distance=distance,
            returns=returns,
            risk_measure=risk_measure,
            alpha=alpha,
            minimum_acceptable_return=minimum_acceptable_return,
            within_clusters=within_clusters,
            cluster_criterion=cluster_criterion,
            similarity=similarity,
        )
        table = self._add_hierarchical_risk(
            self._weights_table(
                weights, expected_returns, covariance_matrix, risk_free_rate, rounding
            ),
            weights,
            returns,
            risk_measure,
            alpha,
            minimum_acceptable_return,
            rounding,
        )
        _, labels = optimization_model.get_hierarchical_clusters(
            covariance_matrix,
            linkage_method,
            clusters,
            distance=distance,
            cluster_criterion=cluster_criterion,
            similarity=similarity,
        )
        table["Cluster"] = pd.Series(labels + 1, index=covariance_matrix.index)
        return table

    @handle_errors
    def get_nested_clustered_optimization_weights(
        self,
        objective: str = "minimum_variance",
        linkage_method: str = "ward",
        clusters: int | None = None,
        cluster_criterion: str = "two_difference_gap",
        codependence: str = "pearson",
        period: str = "daily",
        covariance: str = "denoised",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        risk_measure: str = "variance",
        alpha: float = 0.05,
        minimum_acceptable_return: float | str = 0.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Nested Clustered Optimization (NCO) weights, optimized within clusters of
        similar assets and then across the clusters.

        The assets are clustered by their correlations, the objective is optimized within every
        cluster, and then once more across the clusters, each cluster acting as one asset that
        holds its own optimal portfolio. Estimation noise in the covariance matrix mostly sits
        among correlated assets, which is what makes mean-variance weights unstable; by
        optimizing within and across clusters separately, NCO keeps that noise contained and the
        weights steadier out of sample.

        With W the matrix whose column k holds the optimal weights within cluster k, the
        clusters have the expected returns W'μ and the covariance matrix W'ΣW; the weights
        across the clusters v are optimal for those, and the weight of every asset is
        w = W v. The clusters are found as in HERC (correlation distance, Ward linkage,
        the number of clusters from the two-order difference of the gap statistic) and the
        covariance matrix is denoised by default, as in the paper. With another risk
        measure than the variance, the minimum variance objective minimizes that measure of the
        historical returns and the risk parity objective equalizes its risk contributions,
        within the clusters and across them on the returns of the clusters' portfolios, as
        Riskfolio-Lib's HCPortfolio does.

        For more information, see the following papers:

        - López de Prado, M. (2019). "A Robust Estimator of the Efficient Frontier." SSRN
          3469961.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge
          University Press, chapter 7.

        Also known as: NCO, nested clustered optimisation.

        Referenced against:
        - Riskfolio-Lib, HCPortfolio (NCO): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/hcportfolio.html#HCPortfolio.HCPortfolio

        Shares similar logic:
        - skfolio, NestedClustersOptimization: https://skfolio.org/generated/skfolio.optimization.NestedClustersOptimization.html
        - RiskLabAI, get_optimal_portfolio_weights_nco: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/optimization/nco.py#L49

        Args:
            objective (str, optional): The objective at both levels. Options are: "minimum_variance" (the lowest
                risk), "maximum_sharpe" (the highest Sharpe ratio), "risk_parity" (equal risk contribution).
                Defaults to "minimum_variance".
            linkage_method (str, optional): The linkage of the hierarchical clustering. Options are: "single"
                (single linkage), "complete" (complete linkage), "average" (average linkage), "ward" (Ward's
                method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo and Aste, 2012, whose own
                clusters are used without a number of clusters). Defaults to "ward".
            clusters (int | None, optional): The number of clusters. Defaults to None,
                chosen by the two-order difference of the gap statistic.
            cluster_criterion (str, optional): How the number of clusters is chosen when it is not given. Options
                are: "two_difference_gap" (the sharpest bend of the within-cluster dispersion, Yue, Wang and Wei,
                2008), "standardized_silhouette" (the highest mean over standard deviation of the silhouette
                coefficients, López de Prado, 2020), "onc" (the Optimal Number of Clusters algorithm of López de
                Prado and Lewis, 2019). Defaults to "two_difference_gap".
            codependence (str, optional): The codependence the assets are clustered on. The rank correlations depend
                only on the order of the returns, so a few extreme returns weigh less; the
                distance correlation and mutual information also capture non-linear co-movement.
                Options are: "pearson" (the linear correlation of the covariance matrix),
                "spearman" (the rank correlation of the returns, Spearman, 1904), "kendall" (the
                rank correlation τ_b of the returns, Kendall, 1938), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "detoned" (the
                correlation without the market component, López de Prado, 2020), "distance" (the
                distance correlation, Székely et al., 2007, through √(1 - dCor)),
                "mutual_information" (through the variation of information, Meilă, 2007),
                "lower_tail" (how often two assets fall into their lowest 5% of returns
                together, through -ln λ), "absolute_angular" (|ρ| through √(1 - |ρ|), López de
                Prado, 2020), "squared_angular" (ρ² through √(1 - ρ²), López de Prado, 2020),
                "gnpr" (the rank co-movement and the difference between the return
                distributions, θ = 0.5, Marti et al., 2016). Defaults to "pearson".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix, which also gives the correlations
                that are clustered: "denoised" (the eigenvalues of the correlation matrix that random returns could
                produce averaged, Laloux et al., 2000). Options are: "sample" (the sample covariance, ddof=1),
                "ledoit_wolf" (shrunk towards the scaled identity, Ledoit and Wolf, 2004),
                "ledoit_wolf_constant_correlation" (shrunk towards the average correlation, Ledoit and Wolf, 2004),
                "ledoit_wolf_single_factor" (shrunk towards the covariance that the benchmark as a single factor
                implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating Shrinkage, Chen et al., 2010), "shrunk"
                (a fixed 10% towards the scaled identity), "semicovariance" (the co-movement below a zero return,
                Estrada, 2008), "exponential" (exponentially weighted, recent periods weigh most, RiskMetrics),
                "denoised" (the eigenvalues of the correlation matrix that random returns could produce averaged,
                Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to zero), "denoised_shrink" (the
                noise correlations shrunk towards their diagonal, López de Prado, 2020), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "factor" (the covariance of the Fama and
                French five factor model with OLS loadings, Fama and French, 1993), "minimum_covariance_determinant"
                (the reweighted covariance of the half of the periods with the smallest covariance determinant,
                robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the sparse inverse covariance on the
                TMFG planar graph of the correlations, Barfuss et al., 2016), "geodesic" (10% towards the scaled
                identity along the geodesic of the positive definite matrices, Bhatia, 2007), "regime_adjusted_ew"
                (exponentially weighted with a half-life of 40 periods, rescaled by the recent realized against
                predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one part of the periods and the
                variances along them from the other, averaged over random splits, Lam, 2016), "ccc_garch" (the
                next-period covariance of GARCH(1, 1) volatilities with constant correlations, Bollerslev, 1990),
                "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke" (Rocke's robust S-estimator, Maronna
                et al., 2019). Defaults to "denoised".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
            expected_returns (str, optional): The estimator of the expected returns of the "maximum_sharpe"
                objective. Options are: "mean" (the historical average), "geometric" (the compounded average),
                "exponential" (the exponentially weighted average, recent periods weigh most),
                "exponential_geometric" (that average compounded over a year), "capm" (the returns the Capital Asset
                Pricing Model implies from each asset's beta to the benchmark, Sharpe, 1964), "bayes_stein" (the
                averages shrunk towards that of the minimum variance portfolio, Jorion, 1986), "james_stein" (the
                averages shrunk towards their grand mean, James and Stein, 1961), "bodnar_okhrin_parolya" (the
                averages and their grand mean combined with optimal weights, Bodnar, Okhrin and Parolya, 2019).
                Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the exponentially weighted expected
                returns. Defaults to 180.
            risk_measure (str, optional): The risk measure of the minimum variance and risk parity objectives (see
                get_mean_risk_weights); the volatility gives the weights of the variance, and the maximum Sharpe
                ratio takes the variance only. Options are: "variance" (of the covariance matrix), "volatility" (of
                the covariance matrix), "mad" (mean absolute deviation, Konno and Yamazaki, 1991), "semideviation"
                (below the minimum acceptable return), "lower_partial_moment" (first order), "worst_realization"
                (the worst return), "gini_mean_difference" (Yitzhaki, 1982), "tail_gini" (Furman et al., 2017),
                "range" (the best minus the worst return), "cvar_range" (the CVaR of the losses and of the gains),
                "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the uncompounded
                cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar" (Chekhlov et al., 2005). Defaults
                to "variance".
            alpha (float, optional): The tail probability of the CVaR, CDaR, CVaR
                range and Tail Gini measures, e.g. 0.05 for the worst 5% of periods.
                Defaults to 0.05.
            minimum_acceptable_return (float | str, optional): The threshold of the semideviation and the lower
                partial moment, as a return per period of the chosen frequency, or "mean" for each asset's average
                return. Defaults to 0.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with a risk measure of the
            returns that measure per period (e.g. "MAD"), and the Cluster of every asset
            (numbered from 1). Returns and volatilities are annualized.

        Notes:
        - Both levels are long only and fully invested. Weight, sector and country
          bounds and a net or gross exposure do not apply: the weight of an asset is the
          product of its weight within its cluster and the cluster's weight, from two
          separate problems, so a bound on it is not a constraint either problem can
          hold.
        - With the maximum Sharpe ratio objective, a cluster without an asset whose expected
          return exceeds the risk free rate holds its minimum variance portfolio. Across the
          clusters, at least one cluster needs an expected return above it.
        - The paper clusters with k-means and solves both levels without bounds,
          allowing short positions. Here the clustering is hierarchical, as in HERC.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_nested_clustered_optimization_weights()[
            ["Weight", "Volatility", "Sharpe Ratio", "Risk Contribution", "Cluster"]
        ]
        ```

        Which returns:

        |           |   Weight |   Volatility |   Sharpe Ratio |   Risk Contribution |   Cluster |
        |:----------|---------:|-------------:|---------------:|--------------------:|----------:|
        | AAPL      |   0.1081 |       0.2786 |         0.5725 |              0.1106 |         1 |
        | MSFT      |   0.1388 |       0.2571 |         0.6534 |              0.1362 |         1 |
        | JNJ       |   0.575  |       0.1673 |         0.3884 |              0.575  |         3 |
        | XOM       |   0.1782 |       0.2702 |         0.9524 |              0.1782 |         2 |
        | Portfolio |   1      |       0.1366 |         0.9061 |              1      |       nan |
        """
        check_option(
            "codependence", codependence, optimization_model.CODEPENDENCE_METHODS
        )
        validate_alpha(alpha)
        returns, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        distance = optimization_model.get_codependence_distance(
            returns, covariance_matrix, codependence
        )
        similarity = (
            optimization_model.get_codependence_similarity(
                returns, covariance_matrix, codependence
            )
            if linkage_method == "dbht"
            else None
        )
        weights = optimization_model.get_nested_clustered_optimization_weights(
            expected,
            covariance_matrix,
            objective,
            risk_free_rate,
            linkage_method,
            clusters,
            distance=distance,
            returns=returns,
            risk_measure=risk_measure,
            alpha=alpha,
            minimum_acceptable_return=minimum_acceptable_return,
            cluster_criterion=cluster_criterion,
            similarity=similarity,
        )
        table = self._add_hierarchical_risk(
            self._weights_table(
                weights, expected, covariance_matrix, risk_free_rate, rounding
            ),
            weights,
            returns,
            risk_measure,
            alpha,
            minimum_acceptable_return,
            rounding,
        )
        _, labels = optimization_model.get_hierarchical_clusters(
            covariance_matrix,
            linkage_method,
            clusters,
            distance=distance,
            cluster_criterion=cluster_criterion,
            similarity=similarity,
        )
        table["Cluster"] = pd.Series(labels + 1, index=covariance_matrix.index)
        return table

    @handle_errors
    def get_inverse_volatility_weights(
        self,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate inverse volatility weights: each asset's weight is proportional to one over
        its volatility, w_i = (1/σ_i) / Σ(1/σ_j).

        It is the simplest risk-based allocation and equals risk parity when all correlations
        are the same.

        Also known as: naive risk parity, volatility weighting.

        Shares similar logic:
        - skfolio, InverseVolatility: https://skfolio.org/generated/skfolio.optimization.InverseVolatility.html

        Args:
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
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_inverse_volatility_weights()
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.2092 |            0.1933 |       0.2786 |         0.5727 |              0.3023 |
        | MSFT      |   0.2267 |            0.2017 |       0.2572 |         0.6532 |              0.2743 |
        | JNJ       |   0.3484 |            0.0984 |       0.1673 |         0.3867 |              0.2083 |
        | XOM       |   0.2157 |            0.291  |       0.2702 |         0.9522 |              0.2152 |
        | Portfolio |   1      |            0.1832 |       0.1491 |         1.0028 |              1      |
        """
        _, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        weights = optimization_model.get_inverse_volatility_weights(covariance_matrix)
        return self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )

    @handle_errors
    def get_maximum_omega_weights(
        self,
        minimum_acceptable_return: float = 0.0,
        period: str = "daily",
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the portfolio with the highest Omega ratio over the historical
        returns: the gains above a minimum acceptable return divided by the losses below it.

        The Omega ratio uses the whole return distribution instead of its mean and variance, so
        it rewards positive skew and penalizes fat left tails that a Sharpe ratio overlooks.

        With τ the minimum acceptable return, Ω(w) = 1 + (r̄'w - τ) / E[max(0, τ - r_t'w)].
        This ratio is maximized exactly as a linear programme with the Charnes-Cooper
        transformation: with y = κw and κ > 0, maximize r̄'y - τκ subject to
        Σ max(0, τκ - r_t'y) / T = 1 and the weight constraints, then w = y / κ.

        For more information, see the following papers:

        - Keating, C. and Shadwick, W. F. (2002). "A Universal Performance Measure."
          Journal of Performance Measurement, 6(3), 59-84.
        - Kapsos, M., Christofides, N. and Rustem, B. (2014). "Worst-Case Robust Omega
          Ratio." European Journal of Operational Research, 234(2), 499-507.

        Also known as: Omega optimization, maximum gain-loss ratio.

        Referenced against:
        - Riskfolio-Lib, Portfolio.optimization (first lower partial moment): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization

        Args:
            minimum_acceptable_return (float, optional): The threshold between gains and
                losses, as a return per period of the chosen frequency (e.g. 0.0002 a day).
                Defaults to 0.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
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
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one. The rest, 1 minus the net exposure,
                is a Cash row at the risk free rate (negative when borrowed above one),
                which the Portfolio's Expected Return and Sharpe Ratio include. Defaults to
                1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the Omega Ratio of every
            asset on its own and of the portfolio.

        Notes:
        - Some portfolio within the constraints needs an average return above the minimum
          acceptable return; otherwise no Omega ratio is above one and the method returns
          an error that says so.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_maximum_omega_weights()[
            ["Weight", "Expected Return", "Volatility", "Risk Contribution", "Omega Ratio"]
        ]
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Risk Contribution |   Omega Ratio |
        |:----------|---------:|------------------:|-------------:|--------------------:|--------------:|
        | AAPL      |   0.0594 |            0.1932 |       0.2786 |              0.0615 |        1.1306 |
        | MSFT      |   0.2822 |            0.2017 |       0.2571 |              0.2875 |        1.1459 |
        | JNJ       |   0.2697 |            0.0987 |       0.1673 |              0.1258 |        1.1083 |
        | XOM       |   0.3887 |            0.2911 |       0.2702 |              0.5251 |        1.1954 |
        | Portfolio |   1      |            0.2082 |       0.1551 |              1      |        1.2567 |
        """
        returns, expected, covariance, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            expected_returns=expected_returns,
            expected_returns_span=expected_returns_span,
        )
        # The numerator of the ratio takes the expected returns per period; the losses
        # in its denominator are those of the historical returns.
        expected_per_period = expected / self._get_periods_per_year(
            period, returns.index
        )
        current = self._get_current_weights(
            current_weights, returns.columns.tolist(), maximum_turnover is not None
        )
        weights = optimization_model.get_maximum_omega_weights(
            returns,
            minimum_acceptable_return,
            **self._get_constraints(
                returns.columns.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
            expected_returns=expected_per_period,
        )
        table = self._weights_table(
            weights, expected, covariance, risk_free_rate, rounding
        )
        # Ω = 1 + (μ - τ) / E[max(0, τ - r)], the ratio the optimizer maximizes, which
        # without losses below τ is undefined (NaN).
        with_portfolio = returns.assign(**{PORTFOLIO_TICKER: returns @ weights})
        shortfall = (minimum_acceptable_return - with_portfolio).clip(lower=0).mean()
        mean = expected_per_period.copy()
        mean[PORTFOLIO_TICKER] = expected_per_period @ weights
        table["Omega Ratio"] = 1 + (mean - minimum_acceptable_return) / shortfall.where(
            shortfall > 0
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_mean_risk_weights(
        self,
        risk_measure: Literal[
            "mad",
            "semideviation",
            "lower_partial_moment",
            "worst_realization",
            "gini_mean_difference",
            "tail_gini",
            "range",
            "cvar_range",
            "tail_gini_range",
            "ulcer_index",
            "cvar",
            "cdar",
            "maximum_drawdown",
            "average_drawdown",
        ] = "mad",
        objective: str = "minimum_risk",
        target_return: float | None = None,
        alpha: float = 0.05,
        minimum_acceptable_return: float | str = 0.0,
        period: str = "daily",
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of a mean-risk portfolio with a risk measure other than the
        volatility.

        It is the portfolio with the lowest risk, the lowest risk at a target return, or the
        highest ratio of the average excess return to the risk.

        The risk is measured on the historical returns, so measures that only count losses (the
        semideviation, the lower partial moment, the worst realization, the CVaR, the Tail Gini)
        leave upside swings unpenalized, and the Ulcer Index and the drawdown measures penalize
        deep and long drawdowns, which are what investors feel and what ends strategies. The
        minimum CVaR portfolio only looks at the losses in the tail, so it accounts for skewed
        and fat-tailed returns; its maximum ratio portfolio is the maximum STARR (CVaR ratio)
        portfolio of Martin, Rachev and Siboulet (2003), the maximum Sharpe ratio portfolio with
        the tail loss in place of the volatility.

        The risk measures ρ(w) of the portfolio's returns X_t = r_t'w over T periods, with
        α the tail probability and τ the minimum acceptable return, are:

        - The mean absolute deviation Σ |X_t - X̄| / T (Konno and Yamazaki, 1991).
        - The semideviation: √(Σ min(X_t - τ, 0)² / T), or √(Σ min(X_t - X̄, 0)² / (T - 1))
          below the mean (Markowitz, 1959; Estrada, 2008).
        - The first lower partial moment Σ max(τ - X_t, 0) / T
          (Bawa, 1975), or below each asset's mean.
        - The worst realization: the largest loss -min X_t (Young, 1998).
        - The Gini mean difference: the average absolute difference between the returns of
          two periods, Σ_i Σ_j |X_i - X_j| / (T(T - 1)) (Yitzhaki, 1982).
        - The Tail Gini: the Tail Gini of the losses beyond the α quantile (Furman, Wang
          and Zitikis, 2017; Cajas, 2021).
        - The range: the largest gain plus the largest loss, max X_t - min X_t.
        - The CVaR range: the CVaR of the losses plus that of the gains, the average of the
          best α share of the returns minus the average of the worst (Cajas, 2021).
        - The Tail Gini range: the Tail Gini of the losses plus that of the gains.
        - The Ulcer Index: √(Σ D_t² / T) with D_t the drawdowns of the uncompounded
          cumulative return (Martin and McCann, 1989).
        - The Conditional Value at Risk, the average loss in the worst α share of the
          periods, VaR + Σ max(0, -X_t - VaR) / (αT) at its minimum over the VaR
          (Rockafellar and Uryasev, 2000).
        - The Conditional Drawdown at Risk, the average of the worst α share of the
          drawdowns D_t, ζ + Σ max(0, D_t - ζ) / (αT) at its minimum over ζ
          (Chekhlov, Uryasev and Zabarankin, 2005).
        - The maximum drawdown: the largest drawdown max D_t.
        - The average drawdown Σ D_t / T.

        The drawdowns D_t = u_t - Y_t are those of the uncompounded cumulative return
        Y_t = Σ_(s≤t) X_s below its running peak u_t = max(0, max_(s≤t) Y_s), from u_0 = 0.

        The objectives are:

        - The minimum risk: minimize ρ(w), at the target return μ'w = target when given.
        - The maximum ratio: maximize (μ - r_f)'w / ρ(w) with the Charnes-Cooper
          transformation y = κw, κ > 0 (Charnes and Cooper, 1962): maximize (μ - r_f)'y
          subject to ρ(y) ≤ 1 and the weight constraints, then w = y / κ.

        The mean absolute deviation, the lower partial moment, the worst realization, the
        ranges, the CVaR range, the CVaR and the drawdown measures are solved exactly as
        linear programmes with HiGHS. The
        Gini mean difference and the Tail Gini measures are ordered weighted averages of
        the sorted returns, convex because their weights rise with the rank (Cajas, 2021),
        and are solved exactly with cutting planes (Kelley, 1960) stabilized as in Ben-Ameur
        and Neto (2007): every sorted order of the returns gives a linear lower bound of
        the measure, which is added where the linear programme's solution needs it. The
        semideviation and the Ulcer Index have continuous gradients and are minimized with
        SLSQP.

        For more information, see the following papers:

        - Konno, H. and Yamazaki, H. (1991). "Mean-Absolute Deviation Portfolio
          Optimization Model and Its Applications to Tokyo Stock Market." Management
          Science, 37(5), 519-531.
        - Markowitz, H. (1959). "Portfolio Selection: Efficient Diversification of
          Investments." John Wiley & Sons, New York.
        - Estrada, J. (2008). "Mean-Semivariance Optimization: A Heuristic Approach."
          Journal of Applied Finance, 18(1), 57-72.
        - Bawa, V. S. (1975). "Optimal Rules for Ordering Uncertain Prospects." Journal of
          Financial Economics, 2(1), 95-121.
        - Young, M. R. (1998). "A Minimax Portfolio Selection Rule with Linear Programming
          Solution." Management Science, 44(5), 673-683.
        - Yitzhaki, S. (1982). "Stochastic Dominance, Mean Variance, and Gini's Mean
          Difference." The American Economic Review, 72(1), 178-185.
        - Furman, E., Wang, R. and Zitikis, R. (2017). "Gini-Type Measures of Risk and
          Variability: Gini Shortfall, Capital Allocations, and Heavy-Tailed Risks."
          Journal of Banking & Finance, 83, 70-84.
        - Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming
          Framework." SSRN Working Paper 3988927.
        - Martin, P. G. and McCann, B. B. (1989). "The Investor's Guide to Fidelity Funds."
          John Wiley & Sons, New York.
        - Rockafellar, R. T. and Uryasev, S. (2000). "Optimization of Conditional
          Value-at-Risk." Journal of Risk, 2(3), 21-41.
        - Martin, R. D., Rachev, S. T. and Siboulet, F. (2003). "Phi-Alpha Optimal
          Portfolios and Extreme Risk Management." Wilmott Magazine, November, 70-83.
        - Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in
          Portfolio Optimization." International Journal of Theoretical and Applied
          Finance, 8(1), 13-58.
        - Charnes, A. and Cooper, W. W. (1962). "Programming with Linear Fractional
          Functionals." Naval Research Logistics Quarterly, 9(3-4), 181-186.
        - Kelley, J. E. (1960). "The Cutting-Plane Method for Solving Convex Programs."
          Journal of the Society for Industrial and Applied Mathematics, 8(4), 703-712.
        - Ben-Ameur, W. and Neto, J. (2007). "Acceleration of Cutting-Plane and Column
          Generation Algorithms: Applications to Network Design." Networks, 49(1), 3-17.

        Also known as: mean-MAD, mean-semivariance, minimax, mean-Gini, mean-Ulcer,
        Sortino-optimal or Gini-optimal portfolio, minimum CVaR weights, minimum expected
        shortfall portfolio, mean-CVaR, tail risk optimization, maximum CVaR ratio weights,
        maximum STARR, stable tail adjusted return ratio, maximum return-to-CVaR, mean-CVaR
        tangency portfolio, minimum drawdown weights, minimum CDaR, minimum maximum drawdown,
        minimum average drawdown, drawdown optimization.

        Referenced against:
        - skfolio, MeanRisk: https://skfolio.org/generated/skfolio.optimization.MeanRisk.html
        - Riskfolio-Lib, Portfolio.optimization: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization
        - Riskfolio-Lib, Portfolio.owa_optimization (Tail Gini): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.owa_optimization

        Args:
            risk_measure (str, optional): The risk measure. Options are: "mad" (mean absolute deviation, Konno and
                Yamazaki, 1991), "semideviation" (below the minimum acceptable return), "lower_partial_moment"
                (first order), "worst_realization" (the worst return), "gini_mean_difference" (Yitzhaki, 1982),
                "tail_gini" (Furman et al., 2017), "range" (the best minus the worst return), "cvar_range" (the CVaR
                of the losses and of the gains), "tail_gini_range" (the Tail Gini of the losses and of the gains),
                "ulcer_index" (of the uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002),
                "cdar" (Chekhlov et al., 2005), "maximum_drawdown" (of the uncompounded cumulative return),
                "average_drawdown" (of the uncompounded cumulative return). Defaults to "mad".
            objective (str, optional): Options are: "minimum_risk" (the lowest risk), "maximum_ratio" (the highest
                average excess return per unit of risk). Defaults to "minimum_risk".
            target_return (float | None, optional): The annualized expected return the
                minimum risk portfolio must have, e.g. 0.12 for 12%, with the cash of a
                net exposure other than one at the risk free rate. Defaults to None,
                no target.
            alpha (float, optional): The tail probability of the CVaR, the CVaR
                range, the CDaR and the Tail Gini measures, e.g. 0.05 for the worst 5% of
                periods (or drawdowns). Defaults to 0.05.
            minimum_acceptable_return (float | str, optional): The threshold of the semideviation and the lower
                partial moment, as a return per period of the chosen frequency (e.g. 0.0002 a day), or "mean" for
                each asset's average return. Defaults to 0.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            expected_returns (str, optional): The estimator of the expected returns of the ratio and the target
                return. Options are: "mean" (the historical average), "geometric" (the compounded average),
                "exponential" (the exponentially weighted average, recent periods weigh most),
                "exponential_geometric" (that average compounded over a year), "capm" (the returns the Capital Asset
                Pricing Model implies from each asset's beta to the benchmark, Sharpe, 1964), "bayes_stein" (the
                averages shrunk towards that of the minimum variance portfolio, Jorion, 1986), "james_stein" (the
                averages shrunk towards their grand mean, James and Stein, 1961), "bodnar_okhrin_parolya" (the
                averages and their grand mean combined with optimal weights, Bodnar, Okhrin and Parolya, 2019).
                Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the exponentially weighted expected
                returns. Defaults to 180.
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). At 0 the portfolio with the lowest CVaR holds nothing, so
                all weights are 0, and for the drawdown measures a net exposure of 0 is
                refused, as holding nothing has no drawdown. The rest, 1 minus the net
                exposure, is a Cash row at the risk free rate (negative when borrowed above
                one), which the Portfolio's Expected Return and Sharpe Ratio include.
                Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution (of the volatility), with the
            risk measure (per period, positive for a loss, named after it, e.g. "MAD")
            and its ratio (the average excess return per period over the measure, e.g.
            "MAD Ratio") of every asset on its own and of the portfolio.

        Notes:
        - The risk measures are per period of the chosen frequency and not annualized;
          the drawdowns of the Ulcer Index, the CDaR and the maximum and average drawdown
          are those of the uncompounded cumulative return (the sum of the returns, as in
          Chekhlov, Uryasev and Zabarankin), which scales with the weights and keeps the
          problem linear, as in Riskfolio-Lib and skfolio: they are absolute drawdowns of
          a portfolio that keeps its invested amount constant. The compounded drawdowns
          of the risk module, relative to the peak value, are close for small drawdowns
          and smaller for large ones.
        - The CVaR and the CDaR are the values the linear programmes minimize: the average
          of the worst α share of the periods (or drawdowns), with a fraction of the
          boundary period when αT is not a whole number (Rockafellar and Uryasev,
          2002), as in the expected shortfall contributions.
        - Riskfolio-Lib's MAD is half of the mean absolute deviation and its lower partial
          moments divide by T - 1, and skfolio's lower partial moment is below the mean;
          those scale the measure but leave the weights the same. The Gini mean
          difference here is that of the Risk module, Σ_i Σ_j |X_i - X_j| / (T(T - 1)).
        - The risk free rate is the Toolkit's, averaged over the returns and converted to
          a rate per period. For the maximum ratio, some portfolio within the constraints
          needs an average return above it; otherwise the method returns an error that
          says so.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_mean_risk_weights(
            risk_measure="gini_mean_difference", objective="maximum_ratio"
        )[["Weight", "Expected Return", "Gini Mean Difference", "Gini Mean Difference Ratio"]]
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Gini Mean Difference |   Gini Mean Difference Ratio |
        |:----------|---------:|------------------:|-----------------------:|-----------------------------:|
        | AAPL      |   0.0562 |            0.1932 |                 0.0185 |                       0.0342 |
        | MSFT      |   0.307  |            0.2017 |                 0.0174 |                       0.0384 |
        | JNJ       |   0.1846 |            0.0987 |                 0.0112 |                       0.0231 |
        | XOM       |   0.4521 |            0.2911 |                 0.0187 |                       0.0545 |
        | Portfolio |   1      |            0.2226 |                 0.0113 |                       0.0662 |
        """
        validate_alpha(alpha)
        check_option(
            "risk_measure", risk_measure, optimization_model.MEAN_RISK_MEASURES
        )
        returns, expected, covariance, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            expected_returns=expected_returns,
            expected_returns_span=expected_returns_span,
        )
        # The ratio and the target return take the expected returns per period; the
        # risk measures are those of the historical returns.
        periods_per_year = self._get_periods_per_year(period, returns.index)
        expected_per_period = expected / periods_per_year
        risk_free_per_period = risk_free_rate / periods_per_year
        current = self._get_current_weights(
            current_weights, returns.columns.tolist(), maximum_turnover is not None
        )
        weights = optimization_model.get_mean_risk_weights(
            returns,
            risk_measure,
            objective,
            (
                target_return / periods_per_year
                - (1 - net_exposure) * risk_free_per_period
                if target_return is not None
                else None
            ),
            risk_free_per_period,
            alpha,
            minimum_acceptable_return,
            **self._get_constraints(
                returns.columns.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
            expected_returns=expected_per_period,
        )
        table = self._weights_table(
            weights, expected, covariance, risk_free_rate, rounding
        )

        name = optimization_model.MEAN_RISK_MEASURES[risk_measure]
        measure = optimization_model.get_mean_risk_measure(
            returns.assign(**{PORTFOLIO_TICKER: returns @ weights}),
            risk_measure,
            alpha,
            minimum_acceptable_return,
        )
        excess = expected_per_period - risk_free_per_period
        excess[PORTFOLIO_TICKER] = excess @ weights
        table[name] = measure
        # A risk measure of zero (or a gain as the measure) gives no ratio, not ±inf.
        table[f"{name} Ratio"] = excess / measure.where(measure > 0)
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_critical_line_weights(
        self,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the corner portfolios of the efficient frontier with the critical line
        algorithm of Markowitz (1956).

        These are the exact points where an asset enters or leaves the portfolio. Between two
        neighbouring corners the efficient weights move along a straight line, so the corners
        describe the whole long-only (or bounded) frontier without a grid of target returns. The
        minimum variance portfolio (the last corner) and the maximum Sharpe ratio portfolio
        follow from them exactly.

        Every efficient portfolio minimizes ½ w'Σw - λ μ'w subject to the weights summing to
        one and lying between their bounds, for some λ ≥ 0, the risk tolerance (the inverse
        of the risk aversion). The algorithm starts at the portfolio with the highest
        expected return (λ = ∞) and lowers λ. For the assets strictly between their bounds
        (the free assets F, the others B are at a bound) the first-order conditions are a
        linear system:

        - Σ_FF w_F + γ 1 = λ μ_F - Σ_FB w_B
        - 1'w_F = 1 - 1'w_B

        so the weights are affine in λ until a free asset reaches a bound or the gradient
        (Σw)_j - λ μ_j + γ of a bound asset changes sign, which frees it. That λ is a corner.
        On the line between two corners the Sharpe ratio has a single stationary point,
        which gives the maximum Sharpe ratio portfolio in closed form.

        For more information, see the following papers:

        - Markowitz, H. (1956). "The Optimization of a Quadratic Function Subject to Linear
          Constraints." Naval Research Logistics Quarterly, 3(1-2), 111-133.
        - Bailey, D. H. and López de Prado, M. (2013). "An Open-Source Implementation of the
          Critical-Line Algorithm for Portfolio Optimization." Algorithms, 6(1), 169-196.

        Also known as: CLA, corner portfolios, turning points of the efficient frontier.

        Referenced against:
        - PyPortfolioOpt, CLA: https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html#pypfopt.cla.CLA

        Args:
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix. Options are: "sample" (the sample
                covariance, ddof=1), "ledoit_wolf" (shrunk towards the scaled identity, Ledoit and Wolf, 2004),
                "ledoit_wolf_constant_correlation" (shrunk towards the average correlation, Ledoit and Wolf, 2004),
                "ledoit_wolf_single_factor" (shrunk towards the covariance that the benchmark as a single factor
                implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating Shrinkage, Chen et al., 2010), "shrunk"
                (a fixed 10% towards the scaled identity), "semicovariance" (the co-movement below a zero return,
                Estrada, 2008), "exponential" (exponentially weighted, recent periods weigh most, RiskMetrics),
                "denoised" (the eigenvalues of the correlation matrix that random returns could produce averaged,
                Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to zero), "denoised_shrink" (the
                noise correlations shrunk towards their diagonal, López de Prado, 2020), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "factor" (the covariance of the Fama and
                French five factor model with OLS loadings, Fama and French, 1993), "minimum_covariance_determinant"
                (the reweighted covariance of the half of the periods with the smallest covariance determinant,
                robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the sparse inverse covariance on the
                TMFG planar graph of the correlations, Barfuss et al., 2016), "geodesic" (10% towards the scaled
                identity along the geodesic of the positive definite matrices, Bhatia, 2007), "regime_adjusted_ew"
                (exponentially weighted with a half-life of 40 periods, rescaled by the recent realized against
                predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one part of the periods and the
                variances along them from the other, averaged over random splits, Lam, 2016), "ccc_garch" (the
                next-period covariance of GARCH(1, 1) volatilities with constant correlations, Bollerslev, 1990),
                "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke" (Rocke's robust S-estimator, Maronna
                et al., 2019). Defaults to "sample".
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
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: One row per corner portfolio, from "Corner 1" (the highest
            expected return) to the minimum variance portfolio, followed by the "Minimum
            Variance" and "Maximum Sharpe" portfolios, each with its Lambda (the risk
            tolerance λ at which it is optimal), the annualized Expected Return,
            Volatility and Sharpe Ratio and the weight per asset.

        Notes:
        - The critical line algorithm takes the weight bounds and the budget only; for
          sector, country, gross exposure or turnover constraints use
          get_efficient_frontier, which solves each target return numerically.
        - λ is in the units of the annualized expected returns and covariance. The first
          corner is optimal for every λ at or above its own, and the last one, at λ = 0,
          is the minimum variance portfolio, so it is listed twice.
        - PyPortfolioOpt's CLA lists the first portfolio twice (at λ = ∞ and at the first
          corner); here it appears once.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_critical_line_weights().drop(columns="Sharpe Ratio")
        ```

        Which returns:

        | Portfolio        |   Lambda |   Expected Return |   Volatility |   AAPL |   MSFT |    JNJ |    XOM |
        |:-----------------|---------:|------------------:|-------------:|-------:|-------:|-------:|-------:|
        | Corner 1         |   0.7508 |            0.2911 |       0.2702 | 0      | 0      | 0      | 1      |
        | Corner 2         |   0.2405 |            0.2591 |       0.2031 | 0      | 0.3583 | 0      | 0.6417 |
        | Corner 3         |   0.224  |            0.2578 |       0.2016 | 0.0103 | 0.3619 | 0      | 0.6278 |
        | Corner 4         |   0      |            0.1577 |       0.135  | 0.0211 | 0.2167 | 0.5821 | 0.1801 |
        | Minimum Variance |   0      |            0.1577 |       0.135  | 0.0211 | 0.2167 | 0.5821 | 0.1801 |
        | Maximum Sharpe   |   0.147  |            0.2234 |       0.167  | 0.014  | 0.312  | 0.2    | 0.474  |
        """
        _, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        corners = optimization_model.get_critical_line_weights(
            expected,
            covariance_matrix,
            risk_free_rate,
            minimum_weight,
            maximum_weight,
        )
        return self._round(corners, rounding)

    @handle_errors
    def get_worst_case_mean_variance_weights(
        self,
        uncertainty: str = "box",
        confidence_level: float = 0.95,
        objective: str = "maximum_sharpe",
        risk_aversion: float = 2.5,
        target_return: float | None = None,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of a worst-case (robust) mean-variance portfolio.

        It is the portfolio that is optimal for the least favourable expected returns within a
        confidence region around the estimates, instead of for the estimates themselves. The
        expected returns are the noisiest input of a mean-variance optimization, and the worst
        case penalizes the portfolios whose return rests most on them.

        With μ the annualized expected returns, Σ the annualized covariance matrix,
        Σ_μ = Σ × (periods per year) / T the covariance of the annualized average over T
        periods and α the confidence level, the worst-case return R(w) is:

        - An ellipsoid: μ'w - κ √(w'Σ_μ w) with κ² = χ²_n(α), the lowest return over the
          joint confidence ellipsoid of the n expected returns (Goldfarb and Iyengar,
          2003).
        - A box: μ'w - δ'|w| with δ_i = z_(1+α)/2 √(Σ_μ,ii), the lowest return when each
          expected return can lie anywhere in its own confidence interval (Tütüncü and
          Koenig, 2004).

        The objectives replace the expected return μ'w by R(w):

        - The maximum Sharpe ratio: maximize (R(w) - r_f) / √(w'Σw), solved with the
          Charnes-Cooper transformation (Charnes and Cooper, 1962).
        - The utility: maximize R(w) - (λ/2) w'Σw with λ the risk aversion.
        - The minimum variance: minimize w'Σw subject to R(w) ≥ the target return.

        R(w) is concave, so every objective is a convex problem, solved with SLSQP.

        For more information, see the following papers:

        - Goldfarb, D. and Iyengar, G. (2003). "Robust Portfolio Selection Problems."
          Mathematics of Operations Research, 28(1), 1-38.
        - Tütüncü, R. H. and Koenig, M. (2004). "Robust Asset Allocation." Annals of
          Operations Research, 132, 157-187.
        - Charnes, A. and Cooper, W. W. (1962). "Programming with Linear Fractional
          Functionals." Naval Research Logistics Quarterly, 9(3-4), 181-186.

        Also known as: robust mean-variance, robust Markowitz, max-min portfolio,
        uncertainty set optimization.

        Referenced against:
        - Riskfolio-Lib, Portfolio.wc_optimization (box): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.wc_optimization
        - skfolio, MeanRisk (with an EmpiricalMuUncertaintySet): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

        Args:
            uncertainty (str, optional): The uncertainty set of the expected returns. Options are: "box" (a
                confidence interval per estimate), "ellipsoid" (the joint confidence region). Defaults to "box".
            confidence_level (float, optional): The confidence level of the uncertainty
                set, e.g. 0.95. A higher level widens the set and gives more cautious
                weights. Defaults to 0.95.
            objective (str, optional): Options are: "maximum_sharpe" (the highest worst-case Sharpe ratio),
                "utility" (the highest worst-case utility), "minimum_variance" (the lowest variance, which needs a
                target return). Defaults to "maximum_sharpe".
            risk_aversion (float, optional): The risk aversion λ of the utility; higher
                values give less risky portfolios. Defaults to 2.5.
            target_return (float | None, optional): The lowest annualized worst-case
                return of the minimum variance objective, e.g. 0.1 for 10%, with the cash
                of a net exposure other than one at the risk free rate. Defaults to
                None.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix, as in get_maximum_sharpe_weights.
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
            expected_returns (str, optional): The estimator of the expected returns, as in
                get_maximum_sharpe_weights. Options are: "mean" (the historical average), "geometric" (the
                compounded average), "exponential" (the exponentially weighted average, recent periods weigh most),
                "exponential_geometric" (that average compounded over a year), "capm" (the returns the Capital Asset
                Pricing Model implies from each asset's beta to the benchmark, Sharpe, 1964), "bayes_stein" (the
                averages shrunk towards that of the minimum variance portfolio, Jorion, 1986), "james_stein" (the
                averages shrunk towards their grand mean, James and Stein, 1961), "bodnar_okhrin_parolya" (the
                averages and their grand mean combined with optimal weights, Bodnar, Okhrin and Parolya, 2019).
                Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the exponentially weighted expected
                returns. Defaults to 180.
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio. The rest, 1 minus the net exposure, is a Cash row at the risk
                free rate (negative when borrowed above one), which the Portfolio's Expected
                Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the Worst-Case Return
            and the Worst-Case Sharpe Ratio of every asset on its own and of the
            portfolio. Returns and volatilities are annualized.

        Notes:
        - Only the uncertainty of the expected returns is modelled. Uncertainty sets on
          the covariance matrix (as in Riskfolio-Lib's worst-case optimization) need
          semidefinite programming, which is not available with SciPy's solvers.
        - Σ_μ = Σ / T is the covariance of the historical average. For the other
          expected return estimators it is an approximation of their uncertainty.
        - The box of each expected return is its own two-sided confidence interval
          from the normal distribution. Riskfolio-Lib estimates its box and ellipsoid
          by bootstrapping by default; with the same δ, Σ_μ and κ the weights agree.
        - skfolio's EmpiricalMuUncertaintySet uses only the variances of Σ_μ by default;
          the full Σ_μ here equals its full-matrix option.
        - With the ellipsoid, Σ_μ = Σ × (periods per year) / T is proportional to Σ, so
          every Sharpe ratio falls by the same κ √((periods per year) / T) and the
          maximum Sharpe portfolio equals that of get_maximum_sharpe_weights. The
          ellipsoid changes the weights of the utility and minimum variance objectives,
          and the box changes those of all three.
        - skfolio's EmpiricalMuUncertaintySet shares the ellipsoid κ² = χ²_n(α) and Σ_μ = Σ / T.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_worst_case_mean_variance_weights(
            confidence_level=0.9, maximum_weight=0.5
        )[["Weight", "Expected Return", "Worst-Case Return", "Worst-Case Sharpe Ratio"]]
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Worst-Case Return |   Worst-Case Sharpe Ratio |
        |:----------|---------:|------------------:|--------------------:|--------------------------:|
        | AAPL      |      0   |            0.1932 |             -0.0122 |                   -0.1649 |
        | MSFT      |      0.5 |            0.2017 |              0.0121 |                   -0.084  |
        | JNJ       |      0   |            0.0987 |             -0.0247 |                   -0.349  |
        | XOM       |      0.5 |            0.2911 |              0.0919 |                    0.2152 |
        | Portfolio |      1   |            0.2464 |              0.052  |                    0.0941 |
        """
        returns, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        # The covariance of the annualized average of T periods: Var(P r̄) = P Σ_annual / T.
        mean_covariance = (
            covariance_matrix
            * self._get_periods_per_year(period, returns.index)
            / len(returns)
        )
        current = self._get_current_weights(
            current_weights,
            covariance_matrix.index.tolist(),
            maximum_turnover is not None,
        )
        weights = optimization_model.get_worst_case_mean_variance_weights(
            expected,
            covariance_matrix,
            mean_covariance,
            uncertainty,
            confidence_level,
            objective,
            risk_free_rate,
            risk_aversion,
            target_return,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
        )
        table = self._weights_table(
            weights, expected, covariance_matrix, risk_free_rate, rounding
        )
        portfolios = {
            **{asset: pd.Series({asset: 1.0}) for asset in covariance_matrix.index},
            PORTFOLIO_TICKER: weights,
        }
        worst = pd.Series(
            {
                name: optimization_model.get_worst_case_return(
                    portfolio,
                    expected,
                    mean_covariance,
                    uncertainty,
                    confidence_level,
                )
                for name, portfolio in portfolios.items()
            }
        )
        volatility = pd.Series(
            {
                name: np.sqrt(
                    portfolio.reindex(covariance_matrix.index).fillna(0)
                    @ covariance_matrix
                    @ portfolio.reindex(covariance_matrix.index).fillna(0)
                )
                for name, portfolio in portfolios.items()
            }
        )
        # The cash of a net exposure other than one earns the risk free rate for certain.
        cash = 1 - float(weights.sum())
        worst[PORTFOLIO_TICKER] += cash * risk_free_rate
        if "Cash" in table.index:
            worst["Cash"] = risk_free_rate
        table["Worst-Case Return"] = worst
        table["Worst-Case Sharpe Ratio"] = (worst - risk_free_rate) / volatility.where(
            volatility > optimization_model.ZERO_VOLATILITY_TOLERANCE
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_distributionally_robust_cvar_weights(
        self,
        alpha: float = 0.05,
        wasserstein_radius: float = 0.0005,
        risk_aversion: float = 1.0,
        period: str = "daily",
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the distributionally robust mean-CVaR portfolio.

        It is the portfolio that minimizes the expected loss plus the risk aversion times the
        Conditional Value at Risk (CVaR) under the least favourable distribution of the returns
        within a Wasserstein ball around the historical returns. The historical returns are one
        sample of the distribution; the ball holds every distribution that lies within a
        transport distance ε of it, so the portfolio is protected against distributions with
        heavier losses than the sample happened to show.

        With X = ξ'w the portfolio return, α the tail probability, ρ the risk aversion and
        ε the radius, the weights minimize

        - sup over Q in the ball of E_Q[-X] + ρ CVaR_α,Q(-X),

        where the distance between two distributions is the least average 1-norm change
        in the returns of the assets that turns one into the other (the type-1
        Wasserstein distance), and returns stay at or above -100%. Mohajerin Esfahani
        and Kuhn (2018) show that this worst case is a linear programme: with the CVaR written as
        min over τ of τ + E[max(-X - τ, 0)] / α the loss is the maximum of two
        linear pieces, and its worst case is min λε + Σ s_t / T over λ ≥ 0 with s_t bounding
        each piece in period t after the most harmful shift of the returns. Per asset that
        shift is either none or down to -100%, so the programme has n + 2 + T + 2n
        variables and is solved exactly with HiGHS. At a radius of 0 it is the historical
        mean-CVaR portfolio.

        For more information, see the following papers:

        - Mohajerin Esfahani, P. and Kuhn, D. (2018). "Data-Driven Distributionally Robust
          Optimization Using the Wasserstein Metric: Performance Guarantees and Tractable
          Reformulations." Mathematical Programming, 171(1-2), 115-166.
        - Rockafellar, R. T. and Uryasev, S. (2000). "Optimization of Conditional
          Value-at-Risk." Journal of Risk, 2(3), 21-41.

        Also known as: Wasserstein robust CVaR, DRO CVaR, distributionally robust
        mean-CVaR.

        Referenced against:
        - skfolio, DistributionallyRobustCVaR: https://skfolio.org/generated/skfolio.optimization.DistributionallyRobustCVaR.html

        Args:
            alpha (float, optional): The tail probability α of the CVaR, e.g. 0.05
                for the average of the worst 5% of periods. Defaults to 0.05.
            wasserstein_radius (float, optional): The radius ε of the Wasserstein ball, in
                returns per period of the chosen frequency: the average total change in
                the returns of all assets that the distributions considered may have.
                Defaults to 0.0005, which suits daily returns (see the Notes).
            risk_aversion (float, optional): The weight ρ of the CVaR against the
                expected loss; a high value gives the minimum worst-case CVaR portfolio.
                Defaults to 1.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio. The rest, 1 minus the net exposure, is a Cash row at the risk
                free rate (negative when borrowed above one), which the Portfolio's Expected
                Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the historical CVaR and
            the Worst-Case CVaR over the Wasserstein ball (per period, positive for a
            loss) of every asset on its own and of the portfolio.

        Notes:
        - The radius is in returns per period, so it depends on the frequency. For a
          long-only portfolio and a small radius the worst case adds
          ε (1 + ρ / α) max_i w_i to the historical objective: the least
          favourable distribution moves the asset with the largest weight down in the
          tail periods, which favours spreading the weights. With daily returns a radius
          of about 0.002 or more already gives equal weights, so the default is 0.0005
          (skfolio's is 0.02).
        - The worst-case CVaR of an asset or portfolio is the same programme with the
          weights fixed, without the expected loss.
        - skfolio's DistributionallyRobustCVaR has the same formulation (a 1-norm transport
          cost, returns of at least -100%) and the same α = 0.05 (its β = 0.95) and ρ = 1 by
          default.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_distributionally_robust_cvar_weights()[
            ["Weight", "Expected Return", "CVaR", "Worst-Case CVaR"]
        ]
        ```

        Which returns:

        |           |   Weight |   Expected Return |   CVaR |   Worst-Case CVaR |
        |:----------|---------:|------------------:|-------:|------------------:|
        | AAPL      |   0.0581 |            0.1932 | 0.039  |            0.049  |
        | MSFT      |   0.2543 |            0.2017 | 0.0356 |            0.0456 |
        | JNJ       |   0.4117 |            0.0987 | 0.0227 |            0.0327 |
        | XOM       |   0.2759 |            0.2911 | 0.0373 |            0.0473 |
        | Portfolio |   1      |            0.1835 | 0.0199 |            0.0241 |
        """
        validate_alpha(alpha)
        returns, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        current = self._get_current_weights(
            current_weights,
            returns.columns.tolist(),
            maximum_turnover is not None,
        )
        weights = optimization_model.get_distributionally_robust_cvar_weights(
            returns,
            alpha,
            wasserstein_radius,
            risk_aversion,
            **self._get_constraints(
                returns.columns.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
        )
        table = self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, rounding
        )
        portfolios = {
            **{asset: pd.Series({asset: 1.0}) for asset in returns.columns},
            PORTFOLIO_TICKER: weights,
        }
        table["CVaR"] = (
            -returns.assign(**{PORTFOLIO_TICKER: returns @ weights})
        ).apply(optimization_model.get_expected_tail_loss, alpha=alpha)
        table["Worst-Case CVaR"] = pd.Series(
            {
                name: optimization_model.get_worst_case_cvar(
                    returns, portfolio, alpha, wasserstein_radius
                )
                for name, portfolio in portfolios.items()
            }
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_maximum_diversification_weights(
        self,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the maximum diversification portfolio.

        It is the fully invested portfolio with the highest diversification ratio, the weighted
        average volatility of its assets divided by its own volatility. The ratio is one for a
        single asset and grows as the assets move less together, so the portfolio holds the
        assets that diversify each other most. It needs no expected returns, and when every
        asset has the same Sharpe ratio it is also the maximum Sharpe ratio portfolio.

        The weights maximize DR(w) = w'σ / √(w'Σw) subject to the weights summing to one
        and lying between the minimum and maximum weight, with σ the volatilities of the
        assets and Σ their annualized covariance matrix. The square of the ratio is the
        number of independent risk factors a long-only portfolio effectively holds.

        For more information, see the following papers:

        - Choueifaty, Y. and Coignard, Y. (2008). "Toward Maximum Diversification." The
          Journal of Portfolio Management, 35(1), 40-51.
        - Choueifaty, Y., Froidure, T. and Reynier, J. (2013). "Properties of the Most
          Diversified Portfolio." Journal of Investment Strategies, 2(2), 49-70.

        Also known as: most diversified portfolio, MDP, maximum diversification ratio.

        Referenced against:
        - skfolio, MaximumDiversification: https://skfolio.org/generated/skfolio.optimization.MaximumDiversification.html

        Args:
            period (str, optional): The frequency of the returns the covariance matrix is estimated from. Options
                are: "daily" (daily returns), "weekly" (weekly returns), "monthly" (monthly returns), "quarterly"
                (quarterly returns), "yearly" (yearly returns). Defaults to "daily".
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
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). The rest, 1 minus the net exposure, is a Cash row at the
                risk free rate (negative when borrowed above one), which the Portfolio's
                Expected Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover or transaction cost needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the Diversification Ratio
            of the portfolio (one for every asset on its own). Returns and volatilities are
            annualized.

        Notes:
        - The ratio is maximized directly, so it works with any weight bounds, including
          short positions; with a negative minimum weight the ratio no longer counts
          independent risk factors.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_maximum_diversification_weights()[
            ["Weight", "Volatility", "Sharpe Ratio", "Risk Contribution", "Diversification Ratio"]
        ]
        ```

        Which returns:

        |           |   Weight |   Volatility |   Sharpe Ratio |   Risk Contribution |   Diversification Ratio |
        |:----------|---------:|-------------:|---------------:|--------------------:|------------------------:|
        | AAPL      |   0.0923 |       0.2786 |         0.5727 |              0.1145 |                     1   |
        | MSFT      |   0.2349 |       0.2572 |         0.6532 |              0.2687 |                     1   |
        | JNJ       |   0.4194 |       0.1673 |         0.3867 |              0.3122 |                     1   |
        | XOM       |   0.2533 |       0.2702 |         0.9522 |              0.3046 |                     1   |
        | Portfolio |   1      |       0.1405 |         1.043  |              1      |                     1.6 |
        """
        _, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        current = self._get_current_weights(
            current_weights,
            covariance_matrix.index.tolist(),
            maximum_turnover is not None,
        )
        weights = optimization_model.get_maximum_diversification_weights(
            covariance_matrix,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
        )
        table = self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )
        table["Diversification Ratio"] = 1.0
        table.loc[PORTFOLIO_TICKER, "Diversification Ratio"] = (
            optimization_model.get_diversification_ratio(weights, covariance_matrix)
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_minimum_tracking_error_weights(
        self,
        objective: str = "tracking_error",
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the portfolio of the Toolkit's assets that follows the
        benchmark most closely.

        It is the fully invested portfolio with the lowest tracking error, the volatility of its
        return minus the benchmark's. With the information ratio as the objective it is instead
        the portfolio with the highest information ratio, the expected return above the
        benchmark per unit of tracking error, which is what an active manager measured against
        the benchmark maximizes.

        The tracking error is TE(w) = √Var(R w - r_b) = √(w'Σ_AA w - 2 w'Σ_Ab + σ_b²), with
        Σ_AA the annualized covariance matrix of the assets, Σ_Ab their covariances with the
        benchmark and σ_b² the benchmark's variance. The information ratio is
        (w'μ + (1 - Σw) r_f - μ_b) / TE(w), with μ and μ_b the annualized expected returns
        and the part not invested at the risk free rate r_f, which leaves the tracking error
        as it is. The weights sum to the net exposure, one by default, and lie between the
        minimum and maximum weight; the benchmark is the Toolkit's (SPY by default) and is
        not one of the assets.

        For more information, see the following papers:

        - Roll, R. (1992). "A Mean/Variance Analysis of Tracking Error." The Journal of
          Portfolio Management, 18(4), 13-22.
        - Jorion, P. (2003). "Portfolio Optimization with Tracking-Error Constraints."
          Financial Analysts Journal, 59(5), 70-82.
        - Grinold, R.C. and Kahn, R.N. (2000). "Active Portfolio Management." 2nd edition,
          McGraw-Hill.

        Also known as: index tracking, benchmark tracking, minimum active risk, maximum
        information ratio portfolio.

        Referenced against:
        - skfolio, BenchmarkTracker (tracking error): https://skfolio.org/generated/skfolio.optimization.BenchmarkTracker.html
        - PyPortfolioOpt, objective_functions.ex_post_tracking_error (tracking error): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.ex_post_tracking_error
        - skfolio, MeanRisk (information ratio): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

        Args:
            objective (str, optional): Options are: "tracking_error" (the lowest tracking error),
                "information_ratio" (the highest information ratio). Defaults to "tracking_error".
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix of the assets and the benchmark.
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
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). The rest, 1 minus the net exposure, is a Cash row at the
                risk free rate (negative when borrowed above one), which the Portfolio's
                Expected Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover or transaction cost needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, and against the benchmark the
            Active Return, Tracking Error and Information Ratio (each asset on its own and
            the portfolio). Returns, volatilities and tracking errors are annualized.

        Notes:
        - Only the periods in which every asset and the benchmark have a return are used.
        - With the sample covariance the tracking error equals the annualized standard
          deviation (ddof=1) of the historical active returns R w - r_b.
        - The information ratio needs an asset that expects more than the benchmark;
          otherwise no portfolio has a positive information ratio and an error is raised.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_minimum_tracking_error_weights()[
            ["Weight", "Expected Return", "Active Return", "Tracking Error", "Information Ratio"]
        ]
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Active Return |   Tracking Error |   Information Ratio |
        |:----------|---------:|------------------:|----------------:|-----------------:|--------------------:|
        | AAPL      |   0.249  |            0.1933 |          0.0411 |           0.1861 |              0.2209 |
        | MSFT      |   0.3641 |            0.2017 |          0.0495 |           0.1693 |              0.2923 |
        | JNJ       |   0.2149 |            0.0984 |         -0.0538 |           0.2113 |             -0.2545 |
        | XOM       |   0.172  |            0.291  |          0.1388 |           0.2658 |              0.5224 |
        | Portfolio |   1      |            0.1928 |          0.0406 |           0.0881 |              0.4603 |
        """
        _, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            True,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        if BENCHMARK_TICKER not in covariance_matrix.index:
            raise ValueError("The tracking error needs the returns of a benchmark.")
        current = self._get_current_weights(
            current_weights,
            covariance_matrix.index.drop(BENCHMARK_TICKER).tolist(),
            maximum_turnover is not None,
        )
        weights = optimization_model.get_minimum_tracking_error_weights(
            expected,
            covariance_matrix,
            BENCHMARK_TICKER,
            objective,
            **self._get_constraints(
                covariance_matrix.index.drop(BENCHMARK_TICKER).tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
            risk_free_rate=risk_free_rate,
        )
        assets = weights.index
        table = self._weights_table(
            weights,
            expected[assets],
            covariance_matrix.loc[assets, assets],
            risk_free_rate,
            rounding,
        )
        # Each row on its own, the Cash row as cash only, against the benchmark.
        tracking = pd.DataFrame(
            {
                asset: optimization_model.get_tracking_error(
                    pd.Series({asset: 1.0}),
                    expected,
                    covariance_matrix,
                    BENCHMARK_TICKER,
                    risk_free_rate,
                )
                for asset in table.index.drop(PORTFOLIO_TICKER)
            }
        ).T
        tracking.loc[PORTFOLIO_TICKER] = optimization_model.get_tracking_error(
            weights, expected, covariance_matrix, BENCHMARK_TICKER, risk_free_rate
        )
        return self._round(
            self._add_turnover(pd.concat([table, tracking], axis=1), weights, current),
            rounding,
        )

    @handle_errors
    def get_growth_optimal_weights(
        self,
        method: str = "exact",
        fraction: float = 1.0,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the growth optimal (Kelly) portfolio.

        It is the fully invested portfolio with the highest expected logarithmic growth of
        wealth, which in the long run ends up with more wealth than any other strategy with near
        certainty. Because the full Kelly portfolio is volatile and sensitive to errors in the
        expected returns, investors often hold a fraction of it (fractional Kelly) and keep the
        rest in cash.

        The exact method maximizes the average log return over the historical returns,
        (1/T) Σ ln(1 + r_t'w). The approximate method maximizes its second-order
        approximation w'μ - ½ w'Σw, which is a mean-variance portfolio with a risk aversion
        of one; it uses the expected returns and covariance matrix instead of the returns
        themselves. Both keep the weights summing to one and between the minimum and
        maximum weight. The weights are then multiplied by the fraction and the remainder,
        1 - fraction, is held in cash at the risk free rate (borrowed when the fraction is
        above one).

        For more information, see the following papers:

        - Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System
          Technical Journal, 35(4), 917-926.
        - Breiman, L. (1961). "Optimal Gambling Systems for Favorable Games." Proceedings of
          the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1,
          65-78.
        - Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting and the
          Stock Market." Handbook of Asset and Liability Management, 1, 385-428.

        Also known as: Kelly portfolio, log-optimal portfolio, maximum geometric return,
        fractional Kelly.

        Referenced against:
        - Riskfolio-Lib, Portfolio.optimization (Kelly criterion): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization

        Args:
            method (str, optional): Options are: "exact" (the historical log returns), "approximate" (the
                mean-variance approximation). Defaults to "exact".
            fraction (float, optional): The share of the growth optimal portfolio to hold,
                e.g. 0.5 for half Kelly; the rest is cash. Defaults to 1.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix, used by the "approximate" method and
                for the volatility and risk contributions. Options are: "sample" (the sample covariance, ddof=1),
                "ledoit_wolf" (shrunk towards the scaled identity, Ledoit and Wolf, 2004),
                "ledoit_wolf_constant_correlation" (shrunk towards the average correlation, Ledoit and Wolf, 2004),
                "ledoit_wolf_single_factor" (shrunk towards the covariance that the benchmark as a single factor
                implies, Ledoit and Wolf, 2003), "oas" (Oracle Approximating Shrinkage, Chen et al., 2010), "shrunk"
                (a fixed 10% towards the scaled identity), "semicovariance" (the co-movement below a zero return,
                Estrada, 2008), "exponential" (exponentially weighted, recent periods weigh most, RiskMetrics),
                "denoised" (the eigenvalues of the correlation matrix that random returns could produce averaged,
                Laloux et al., 2000), "denoised_spectral" (those eigenvalues set to zero), "denoised_shrink" (the
                noise correlations shrunk towards their diagonal, López de Prado, 2020), "gerber" (co-movements
                beyond half a standard deviation, Gerber et al., 2022), "factor" (the covariance of the Fama and
                French five factor model with OLS loadings, Fama and French, 1993), "minimum_covariance_determinant"
                (the reweighted covariance of the half of the periods with the smallest covariance determinant,
                robust to outliers, Rousseeuw and Van Driessen, 1999), "jlogo" (the sparse inverse covariance on the
                TMFG planar graph of the correlations, Barfuss et al., 2016), "geodesic" (10% towards the scaled
                identity along the geodesic of the positive definite matrices, Bhatia, 2007), "regime_adjusted_ew"
                (exponentially weighted with a half-life of 40 periods, rescaled by the recent realized against
                predicted risk, Paleologo, 2025), "nercome" (eigenvectors from one part of the periods and the
                variances along them from the other, averaged over random splits, Lam, 2016), "ccc_garch" (the
                next-period covariance of GARCH(1, 1) volatilities with constant correlations, Bollerslev, 1990),
                "mm" (the robust MM-estimator, Maronna et al., 2019), "rocke" (Rocke's robust S-estimator, Maronna
                et al., 2019). Defaults to "sample".
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
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully
                invested portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset, for the Cash (when the fraction is not one) and for the
            portfolio, the Weight, Expected Return, Volatility, Sharpe Ratio and Risk
            Contribution, with the Log Growth, the annualized average log return of each
            asset on its own and of the portfolio. Returns and volatilities are annualized.

        Notes:
        - With historical average returns the full Kelly weights tend to concentrate in the
          assets that happened to grow fastest; a fraction below one or a maximum weight
          tempers that. The expected returns only enter the approximate method; the
          exact method uses the returns themselves.
        - The log growth is the average of ln(1 + r) per period times the number of periods
          in a year, so the growth of daily returns is comparable to that of monthly ones.
        - The weight bounds, sector and country bounds and exposures apply to the full
          Kelly portfolio; the fraction then scales it, so the weights sum to the fraction
          times the net exposure and the rest is cash.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_growth_optimal_weights(fraction=0.5, maximum_weight=0.4)
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |   Log Growth |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|-------------:|
        | AAPL      |      0.1 |            0.1933 |       0.2786 |         0.5727 |              0.2141 |       0.1547 |
        | MSFT      |      0.2 |            0.2017 |       0.2572 |         0.6532 |              0.4118 |       0.1687 |
        | JNJ       |      0   |            0.0984 |       0.1673 |         0.3867 |              0      |       0.0844 |
        | XOM       |      0.2 |            0.291  |       0.2702 |         0.9522 |              0.374  |       0.2544 |
        | Cash      |      0.5 |            0.0337 |       0      |       nan      |              0      |       0.0337 |
        | Portfolio |      1   |            0.1347 |       0.0959 |         1.0531 |              1      |       0.1301 |
        """
        returns, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        weights = optimization_model.get_growth_optimal_weights(
            returns,
            expected,
            covariance_matrix,
            method,
            fraction,
            **self._get_constraints(
                returns.columns.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
        )
        periods_per_year = self._get_periods_per_year(period, returns.index)
        cash = 1 - fraction * net_exposure
        growth = np.log1p(returns).mean() * periods_per_year
        portfolio_growth = (
            np.log1p(
                returns @ weights + cash * risk_free_rate / periods_per_year
            ).mean()
            * periods_per_year
        )

        # The cash is an asset without risk that earns the risk free rate; the table adds
        # it to the weights, so the portfolio's expected return and Sharpe ratio include it.
        growth["Cash"] = np.log1p(risk_free_rate / periods_per_year) * periods_per_year
        table = self._weights_table(
            weights, expected, covariance_matrix, risk_free_rate, rounding
        )
        table["Log Growth"] = growth
        table.loc[PORTFOLIO_TICKER, "Log Growth"] = portfolio_growth
        return self._round(table, rounding)

    @handle_errors
    def get_resampled_efficient_weights(
        self,
        objective: str = "maximum_sharpe",
        resamples: int = 500,
        resampling: str = "parametric",
        seed: int | None = 0,
        risk_aversion: float = 2.5,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        current_weights: dict[str, float] | None = None,
        maximum_turnover: float | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the resampled efficient portfolio (Michaud resampling).

        It is the average of the optimal weights over many return histories that could have been
        observed instead of the actual one. A single optimization takes the estimated expected
        returns and covariances at face value and piles into the assets whose estimates happen
        to be highest; averaging over resampled histories only gives an asset a large weight
        when it earns it in most of them, which makes the weights more diversified and more
        stable from one estimate to the next.

        Every draw has as many periods as the returns. With parametric resampling, as in
        Michaud, the draws come from a multivariate normal distribution with the
        estimated expected returns and covariance; with the bootstrap they are periods of
        the actual returns drawn with replacement, which keeps their fat tails and skew.
        The objective is optimized on the sample mean and sample covariance of every draw
        within the constraints, and the weights are averaged.

        For more information, see the following sources:

        - Michaud, R. O. (1998). "Efficient Asset Management: A Practical Guide to Stock
          Portfolio Optimization and Asset Allocation." Harvard Business School Press.
        - Scherer, B. (2002). "Portfolio Resampling: Review and Critique." Financial
          Analysts Journal, 58(6), 98-109.

        Also known as: Michaud resampling, resampled efficiency, resampled efficient
        frontier, portfolio resampling.

        Referenced against:
        - Riskfolio-Lib, Portfolio.optimization (every draw): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization

        Args:
            objective (str, optional): The portfolio optimized in every draw. Options are: "maximum_sharpe" (the
                highest Sharpe ratio), "minimum_variance" (the lowest variance), "mean_variance" (the utility with
                the risk aversion). Defaults to "maximum_sharpe".
            resamples (int, optional): The number of resampled return histories. The
                average settles as their number grows; 500 keeps the weights within
                about a percentage point of their limit for a handful of assets.
                Defaults to 500.
            resampling (str, optional): How the histories are drawn. Options are: "parametric" (a multivariate
                normal distribution with the estimated expected returns and covariance), "bootstrap" (the actual
                periods, drawn with replacement). Defaults to "parametric".
            seed (int | None, optional): The seed of the random draws; the same seed
                gives the same weights. Defaults to 0; None draws anew on every call.
            risk_aversion (float, optional): The risk aversion of the mean-variance objective. Defaults to 2.5.
            period (str, optional): The frequency of the returns. Options are: "daily" (daily returns), "weekly"
                (weekly returns), "monthly" (monthly returns), "quarterly" (quarterly returns), "yearly" (yearly
                returns). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix the "parametric" draws come from and
                the table reports. Options are: "sample" (the sample covariance, ddof=1), "ledoit_wolf" (shrunk
                towards the scaled identity, Ledoit and Wolf, 2004), "ledoit_wolf_constant_correlation" (shrunk
                towards the average correlation, Ledoit and Wolf, 2004), "ledoit_wolf_single_factor" (shrunk towards
                the covariance that the benchmark as a single factor implies, Ledoit and Wolf, 2003), "oas" (Oracle
                Approximating Shrinkage, Chen et al., 2010), "shrunk" (a fixed 10% towards the scaled identity),
                "semicovariance" (the co-movement below a zero return, Estrada, 2008), "exponential" (exponentially
                weighted, recent periods weigh most, RiskMetrics), "denoised" (the eigenvalues of the correlation
                matrix that random returns could produce averaged, Laloux et al., 2000), "denoised_spectral" (those
                eigenvalues set to zero), "denoised_shrink" (the noise correlations shrunk towards their diagonal,
                López de Prado, 2020), "gerber" (co-movements beyond half a standard deviation, Gerber et al.,
                2022), "factor" (the covariance of the Fama and French five factor model with OLS loadings, Fama and
                French, 1993), "minimum_covariance_determinant" (the reweighted covariance of the half of the
                periods with the smallest covariance determinant, robust to outliers, Rousseeuw and Van Driessen,
                1999), "jlogo" (the sparse inverse covariance on the TMFG planar graph of the correlations, Barfuss
                et al., 2016), "geodesic" (10% towards the scaled identity along the geodesic of the positive
                definite matrices, Bhatia, 2007), "regime_adjusted_ew" (exponentially weighted with a half-life of
                40 periods, rescaled by the recent realized against predicted risk, Paleologo, 2025), "nercome"
                (eigenvectors from one part of the periods and the variances along them from the other, averaged
                over random splits, Lam, 2016), "ccc_garch" (the next-period covariance of GARCH(1, 1) volatilities
                with constant correlations, Bollerslev, 1990), "mm" (the robust MM-estimator, Maronna et al., 2019),
                "rocke" (Rocke's robust S-estimator, Maronna et al., 2019). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the exponentially weighted covariance; the
                weight of a period halves after about span / 2.9 periods. Defaults to 180.
            expected_returns (str, optional): The estimator of the expected returns the "parametric" draws are
                centred on. Options are: "mean" (the historical average), "geometric" (the compounded average),
                "exponential" (the exponentially weighted average, recent periods weigh most),
                "exponential_geometric" (that average compounded over a year), "capm" (the returns the Capital Asset
                Pricing Model implies from each asset's beta to the benchmark, Sharpe, 1964), "bayes_stein" (the
                averages shrunk towards that of the minimum variance portfolio, Jorion, 1986), "james_stein" (the
                averages shrunk towards their grand mean, James and Stein, 1961), "bodnar_okhrin_parolya" (the
                averages and their grand mean combined with optimal weights, Bodnar, Okhrin and Parolya, 2019).
                Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the exponentially weighted expected
                returns. Defaults to 180.
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one. The rest, 1 minus the net exposure,
                is a Cash row at the risk free rate (negative when borrowed above one),
                which the Portfolio's Expected Return and Sharpe Ratio include. Defaults to
                1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            current_weights (dict[str, float] | None, optional): The weights held now,
                e.g. {"AAPL": 0.5, "MSFT": 0.5} (an asset left out is not held), which
                the turnover is measured from; they add a Turnover column, |w_i - w0_i|
                per asset and its sum for the portfolio. Defaults to None, which uses
                the latest weights of the Portfolio the Toolkit came from when a maximum
                turnover needs them.
            maximum_turnover (float | None, optional): The highest turnover
                Σ |w_i - w0_i| from the current weights, e.g. 0.2 to trade at most 10%
                of the portfolio each way. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight (the average over
            the draws), Expected Return, Volatility, Sharpe Ratio and Risk Contribution
            from the estimated expected returns and covariance, with the Weight Standard
            Deviation over the draws per asset: how uncertain the optimal weight of the
            asset is. Returns and volatilities are annualized.

        Notes:
        - The constraints are linear (or convex, for the gross exposure and turnover), so
          the average of weights that meet them meets them too.
        - The expected returns and covariance of every draw are its sample mean and
          sample covariance; the estimators above set the distribution the parametric
          draws come from. A draw without a solution (e.g. one in which no asset beats
          the risk free rate for the maximum Sharpe ratio) is left out of the average,
          with a warning that says how many were.
        - The resampled portfolio is not on the efficient frontier of the estimates: it
          gives up some expected Sharpe ratio, measured on those estimates, for weights
          that depend less on them (Scherer, 2002).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_resampled_efficient_weights()[
            ["Weight", "Expected Return", "Volatility", "Sharpe Ratio", "Weight Standard Deviation"]
        ]
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Weight Standard Deviation |
        |:----------|---------:|------------------:|-------------:|---------------:|----------------------------:|
        | AAPL      |   0.097  |            0.1932 |       0.2786 |         0.5725 |                      0.1559 |
        | MSFT      |   0.2598 |            0.2017 |       0.2571 |         0.6534 |                      0.212  |
        | JNJ       |   0.2171 |            0.0987 |       0.1673 |         0.3884 |                      0.2159 |
        | XOM       |   0.4261 |            0.2911 |       0.2702 |         0.9524 |                      0.2297 |
        | Portfolio |   1      |            0.2166 |       0.1622 |         1.1273 |                    nan      |
        """
        returns, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        current = self._get_current_weights(
            current_weights, returns.columns.tolist(), maximum_turnover is not None
        )
        draws = optimization_model.get_resampled_efficient_weights(
            returns,
            expected,
            covariance_matrix,
            self._get_periods_per_year(period, returns.index),
            objective,
            resamples,
            resampling,
            seed,
            risk_free_rate,
            risk_aversion,
            **self._get_constraints(
                returns.columns.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            current_weights=current,
            maximum_turnover=maximum_turnover,
        )
        if len(draws) < resamples:
            logger.warning(
                "%d of the %d resampled histories have no %s portfolio and are left "
                "out of the average.",
                resamples - len(draws),
                resamples,
                objective.replace("_", " "),
            )
        weights = draws.mean()
        table = self._weights_table(
            weights, expected, covariance_matrix, risk_free_rate, rounding
        )
        table["Weight Standard Deviation"] = draws.std(ddof=1)
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_black_litterman_weights(
        self,
        views: dict[str, float] | None = None,
        view_confidences: dict[str, float] | None = None,
        market_weights: dict[str, float] | None = None,
        risk_aversion: float | str = 2.5,
        tau: float = 0.05,
        method: Literal["standard", "augmented", "bayesian"] = "standard",
        factor_views: dict[str, float] | None = None,
        factor_view_confidences: dict[str, float] | None = None,
        factors: pd.DataFrame | None = None,
        region: str = "United States",
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        net_exposure: float = 1.0,
        gross_exposure: float | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Black-Litterman expected returns and the weights they lead to.

        The starting point is the return the market implies through its capitalization weights:
        the excess return Π = δΣw_mkt at which holding the market is optimal, plus the risk free
        rate, so the Implied Return is a total return. Views then move these returns towards the
        investor's expectations, in proportion to the confidence in each view. The result avoids
        the extreme weights that historical average returns give.

        Views are absolute ({"AAPL": 0.10}: AAPL returns 10% a year in total, risk free
        rate included) or relative ({"AAPL-MSFT": 0.02}: AAPL beats MSFT by 2% a year).
        Without a confidence, a view is as uncertain as the variance of the view portfolio
        scaled by tau; a confidence c scales that by (1 - c) / c, Idzorek's confidence
        method in the closed form of Walters (2014). The weights are the mean-variance
        portfolio of the Black-Litterman returns with the same risk aversion. Without views,
        or with relative views only, the risk free rate shifts every return by the same
        amount and leaves the weights unchanged; an absolute view is compared with the
        implied total return, so there it matters.

        The risk aversion is 2.5 by default, the value He and Litterman (1999) use for the
        world market, or the one the benchmark implies:
        δ = (E[r_m] - r_f) / σ²_m, the benchmark's annualized average excess return over
        its annualized variance on the same periods as the assets.

        Two variants put views on factors, by default the five factors of Fama and French
        (Mkt-RF, SMB, HML, RMW and CMA), with loadings from an OLS regression of the
        assets on the factors (see get_factor_loadings):

        - The augmented model (Cheung, 2009): the assets and the factors form one vector
          with covariance [[Σ, BΣ_f], [Σ_f B', Σ_f]] and implied returns [δΣw; δΣ_f B'w],
          the factor premiums the market implies, and the Black-Litterman update takes
          views on both, so a view on a factor ({"HML": 0.03}, value beats growth by 3% a
          year) moves every asset through its covariance with that factor. Factor views
          are premiums, without the risk free rate.
        - The Bayesian model (Kolm and Ritter, 2017): the prior is the factor model itself,
          the historical average factor returns μ_f with uncertainty τΣ_f, the views on
          the factors update them to Π̂ with uncertainty Ŝ, and the assets follow:
          μ = α + BΠ̂ and Σ_post = BΣ_f B' + D + BŜB', with α the intercepts and D the
          residual variances. It needs factor views and no market weights; the Prior
          Return is α + Bμ_f, the average return of every asset, and the weights are
          the mean-variance portfolio of μ and Σ_post.

        The Posterior Volatility is the square root of the diagonal of the posterior
        covariance matrix Σ + M (He and Litterman, 1999), with M = ((τΣ)⁻¹ + P'Ω⁻¹P)⁻¹
        the uncertainty that remains in the Black-Litterman returns: the volatility to
        expect for each asset once the estimation error of its return is counted too.
        The weights use Σ itself, so that without views they are the market weights.

        For more information, see the following papers:

        - Black, F. and Litterman, R. (1992). "Global Portfolio Optimization." Financial
          Analysts Journal, 48(5), 28-43.
        - He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman Model
          Portfolios." Goldman Sachs Investment Management Research.
        - Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model:
          Incorporating User-Specified Confidence Levels." Working paper.
        - Walters, J. (2014). "The Black-Litterman Model in Detail."
          https://ssrn.com/abstract=1314585
        - Cheung, W. (2009). "The Augmented Black-Litterman Model: A Ranking-Free
          Approach to Factor-Based Portfolio Construction and Beyond."
          https://ssrn.com/abstract=1347648
        - Kolm, P. and Ritter, G. (2017). "On the Bayesian Interpretation of
          Black-Litterman." European Journal of Operational Research, 258(2), 564-572.

        Also known as: BL model, equilibrium returns with views, augmented
        Black-Litterman, Bayesian Black-Litterman, factor views.

        Referenced against:
        - Riskfolio-Lib, augmented_black_litterman: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.augmented_black_litterman
        - Riskfolio-Lib, black_litterman_bayesian: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.black_litterman_bayesian

        Shares similar logic:
        - PyPortfolioOpt, BlackLittermanModel (standard): https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html#pypfopt.black_litterman.BlackLittermanModel
        - Riskfolio-Lib, black_litterman (standard): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.black_litterman
        - PyPortfolioOpt, market_implied_risk_aversion: https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html#pypfopt.black_litterman.market_implied_risk_aversion

        Args:
            views (dict[str, float] | None, optional): The annualized views, absolute views
                as total returns, as described above. Defaults to None, which returns the
                market implied returns.
            view_confidences (dict[str, float] | None, optional): The confidence per view,
                between 0 and 1. Defaults to None, which sets each view's uncertainty to the
                variance of the view scaled by tau.
            market_weights (dict[str, float] | None, optional): The market weight per asset.
                Defaults to None, which uses the market capitalizations of the assets.
            risk_aversion (float | str, optional): The risk aversion δ of the market, or
                "market" for the one the benchmark implies, (E[r_m] - r_f) / σ²_m, with
                the benchmark's arithmetic average return and variance (ddof=1) per
                period both annualized by the number of periods in a year. Defaults to
                2.5.
            tau (float, optional): The uncertainty of the implied returns relative to the
                covariance matrix. Defaults to 0.05.
            method (str, optional): Options are: "standard" (views on assets), "augmented" (views on assets and
                factors), "bayesian" (views on factors of the factor model). Defaults to "standard".
            factor_views (dict[str, float] | None, optional): The annualized views on factors, a premium ({"HML":
                0.03}) or the difference of two ({"SMB-HML": 0.01}), for the augmented and Bayesian models. Defaults
                to None.
            factor_view_confidences (dict[str, float] | None, optional): The confidence per
                factor view, between 0 and 1. Defaults to None, which sets each view's
                uncertainty to the variance of the view scaled by tau.
            factors (pd.DataFrame | None, optional): The factor returns, one column per
                factor, at the frequency of the period. Defaults to None, the Fama and
                French five factors of the region.
            region (str, optional): The region of the Fama and French factors. Defaults to
                "United States".
            period (str, optional): The frequency of the returns the covariance matrix is estimated from. Options
                are: "daily" (daily returns), "weekly" (weekly returns), "monthly" (monthly returns), "quarterly"
                (quarterly returns), "yearly" (yearly returns). Defaults to "daily".
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
            minimum_weight (float | dict[str, float], optional): The lowest weight of
                every asset, or per asset, e.g. {"AAPL": 0.05} (the others keep 0); a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float | dict[str, float], optional): The highest weight of
                every asset, or per asset, e.g. {"AAPL": 0.3} (the others keep 1).
                Defaults to 1.
            sector_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per sector, e.g. {"Technology": (0, 0.4)}, with
                the sector of each asset from its company profile. Defaults to None.
            country_bounds (dict[str, tuple[float, float]] | None, optional): The minimum
                and maximum total weight per country, named in full or by its two-letter
                code, e.g. {"United States": (0, 0.6)}, with the country of each asset
                from its company profile. Defaults to None.
            net_exposure (float, optional): The sum of the weights: 1 for a fully invested
                portfolio, 0 for a market neutral one (which needs a negative
                minimum_weight). The rest, 1 minus the net exposure, is a Cash row at the
                risk free rate (negative when borrowed above one), which the Portfolio's
                Expected Return and Sharpe Ratio include. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per asset its Market Weight, Implied Return, Black-Litterman Return
            (both annualized total returns), Posterior Volatility (annualized) and the
            resulting Weight; with the Bayesian model the Prior Return in place of the
            Market Weight and Implied Return.

        Notes:
        - The risk aversion the market implies is refused when the benchmark earned no more than
          the risk free rate over the period, as that implies no positive risk aversion.
        - The factor models are estimated on the periods in which the assets and the
          factors all have a return.
        - The augmented model agrees with Riskfolio-Lib's augmented_black_litterman (with τ =
          1 / T, its choice) when the loadings hold no intercept: Riskfolio-Lib adds the
          intercepts to the posterior returns when they do, which Cheung (2009) does not.
          It does not invert the stacked covariance, which is singular when an asset is
          itself a factor (the benchmark and Mkt-RF), where Riskfolio-Lib's inverse
          loses its accuracy.
        - The Bayesian model takes the prior N(μ_f, τΣ_f); Riskfolio-Lib's
          black_litterman_bayesian takes N(μ_f, Σ_f) with views of uncertainty
          τ p'Σ_f p, τ = 1 / T, which gives the result here with a tau of 1 and a confidence
          of T / (T + 1) for every view.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_black_litterman_weights(
            views={"XOM": 0.10, "AAPL-MSFT": 0.02}
        )
        ```

        Which returns:

        |      |   Market Weight |   Implied Return |   Black-Litterman Return |   Posterior Volatility |   Weight |
        |:-----|----------------:|-----------------:|-------------------------:|-----------------------:|---------:|
        | AAPL |          0.4861 |           0.1751 |                   0.1772 |                 0.2846 |   0.4727 |
        | MSFT |          0.3834 |           0.1538 |                   0.1556 |                 0.2631 |   0.3768 |
        | JNJ  |          0.0647 |           0.0511 |                   0.0522 |                 0.1714 |   0.016  |
        | XOM  |          0.0658 |           0.0697 |                   0.0847 |                 0.2736 |   0.1345 |
        """
        methods = ("standard", "augmented", "bayesian")
        check_option("method", method, methods)
        if method == "standard" and factor_views:
            raise ValueError(
                "Factor views need method='augmented' or method='bayesian'."
            )
        if method == "bayesian" and views:
            raise ValueError(
                "The Bayesian model takes views on factors only; use factor_views."
            )
        returns, _, covariance_matrix, risk_free_rate = self._get_inputs(
            period, False, covariance, covariance_span
        )
        assets = covariance_matrix.index.tolist()

        if risk_aversion == "market":
            benchmark = self._get_benchmark_returns(period)
            if benchmark is None:
                raise ValueError("The market risk aversion needs a benchmark.")
            risk_aversion = optimization_model.get_market_implied_risk_aversion(
                benchmark.reindex(returns.index),
                self._get_periods_per_year(period, returns.index),
                risk_free_rate,
            )
        elif isinstance(risk_aversion, str):
            raise ValueError(
                f"The risk aversion must be a number or 'market', not '{risk_aversion}'."
            )

        if method != "standard":
            periods_per_year = self._get_periods_per_year(period, returns.index)
            aligned, factor_returns = self._align_factors(
                returns, self._get_factor_returns(period, factors, region)
            )
            loadings = estimation_model.get_factor_loadings(aligned, factor_returns)
            factor_covariance = factor_returns.cov() * periods_per_year

        if method == "bayesian":
            _, _, residual_variance = estimation_model.get_factor_model(
                aligned, factor_returns, loadings
            )
            black_litterman, posterior_covariance = (
                optimization_model.get_bayesian_black_litterman_returns(
                    loadings.assign(Intercept=loadings["Intercept"] * periods_per_year),
                    factor_returns.mean() * periods_per_year,
                    factor_covariance,
                    residual_variance * periods_per_year,
                    factor_views or {},
                    factor_view_confidences,
                    tau,
                )
            )
            weights = optimization_model.get_mean_variance_weights(
                black_litterman["Black-Litterman Return"],
                posterior_covariance,
                risk_aversion=risk_aversion,
                **self._get_constraints(
                    assets,
                    minimum_weight,
                    maximum_weight,
                    sector_bounds,
                    country_bounds,
                ),
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
            )
            table = pd.concat(
                [
                    black_litterman,
                    pd.Series(
                        np.sqrt(np.diag(posterior_covariance)),
                        index=posterior_covariance.index,
                        name="Posterior Volatility",
                    ),
                    weights,
                ],
                axis=1,
            )
            return self._round(table, rounding)

        if market_weights is not None:
            market = pd.Series(market_weights, dtype=float).reindex(assets)
        else:
            profile = self._get_profile(
                "pass the market weights of the assets with market_weights"
            )
            market = pd.to_numeric(
                profile.loc["Market Capitalization"].reindex(assets), errors="coerce"
            )
        if market.isna().any():
            missing = ", ".join(market[market.isna()].index)
            raise ValueError(
                f"No market weight for {missing}; pass market_weights explicitly."
            )

        if method == "augmented":
            black_litterman, posterior_covariance = (
                optimization_model.get_augmented_black_litterman_returns(
                    covariance_matrix,
                    market,
                    factor_covariance,
                    loadings.drop(columns="Intercept"),
                    views=views,
                    view_confidences=view_confidences,
                    factor_views=factor_views,
                    factor_view_confidences=factor_view_confidences,
                    risk_aversion=risk_aversion,
                    tau=tau,
                    risk_free_rate=risk_free_rate,
                )
            )
        else:
            black_litterman, posterior_covariance = (
                optimization_model.get_black_litterman_returns(
                    covariance_matrix,
                    market,
                    views=views,
                    view_confidences=view_confidences,
                    risk_aversion=risk_aversion,
                    tau=tau,
                    risk_free_rate=risk_free_rate,
                )
            )
        weights = optimization_model.get_mean_variance_weights(
            black_litterman["Black-Litterman Return"],
            covariance_matrix,
            risk_aversion=risk_aversion,
            **self._get_constraints(
                assets,
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
        )
        table = pd.concat(
            [
                (market / market.sum()).rename("Market Weight"),
                black_litterman,
                pd.Series(
                    np.sqrt(np.diag(posterior_covariance)),
                    index=posterior_covariance.index,
                    name="Posterior Volatility",
                ),
                weights,
            ],
            axis=1,
        )
        return self._round(table, rounding)
