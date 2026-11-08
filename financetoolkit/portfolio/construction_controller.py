"""Portfolio Construction Module"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.portfolio import (
    attribution_model,
    backtest_model,
    optimization_model,
)
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR, apply_rounding

logger = get_logger()

# The trading days an asset's first price may come after the first day of an attribution.
START_PRICE_TOLERANCE_DAYS = 5

BENCHMARK_TICKER = "Benchmark"
PORTFOLIO_TICKER = "Portfolio"
PERIODS = ["daily", "weekly", "monthly", "quarterly", "yearly"]

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

# How far weights may sum from one before they are reported as scaled.
WEIGHT_SUM_TOLERANCE = 1e-6

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# ruff: noqa: E501


class PortfolioConstruction:
    """
    The Portfolio Construction module builds and examines portfolios from the assets of a
    Toolkit: optimal weights (minimum variance, maximum Sharpe ratio, mean-variance,
    risk parity, Hierarchical Risk Parity, Hierarchical Equal Risk Contribution, Nested
    Clustered Optimization, Black-Litterman, minimum CVaR, minimum drawdown, maximum
    Omega ratio, maximum CVaR ratio, maximum diversification, minimum tracking error,
    growth optimal, resampled efficient), the efficient frontier, a walk-forward
    backtest of the optimizers against equal weights, risk contributions to volatility
    and expected shortfall, diversification metrics, historical stress tests,
    look-through exposure and a Brinson-Fachler attribution against the benchmark. The
    optimizers take weight bounds per asset, sector and country bounds from the company
    profiles, for long-short portfolios a net and gross exposure and, from current
    weights, a maximum turnover.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame] | None = None,
        risk_free_rate_data: dict[str, pd.Series] | None = None,
        benchmark_name: str | None = None,
        portfolio_weights: dict | None = None,
        price_function: Callable[[list[str], str, str], pd.DataFrame] | None = None,
        profile_function: Callable[[], pd.DataFrame] | None = None,
        fund_function: Callable[[str, list[str]], pd.DataFrame] | None = None,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Portfolio Construction Controller Class.

        Args:
            tickers (str | list[str]): The tickers of the Toolkit instance.
            historical_data (dict[str, pd.DataFrame] | None, optional): The historical data
                per period. Defaults to None.
            risk_free_rate_data (dict[str, pd.Series] | None, optional): The annualized
                risk free rate per period. Defaults to None.
            benchmark_name (str | None, optional): The ticker of the benchmark, e.g. "SPY".
                Defaults to None.
            portfolio_weights (dict | None, optional): The weights of a Portfolio over time,
                per period, used when no weights are given. Defaults to None.
            price_function (Callable | None, optional): Retrieves the daily prices of other
                tickers over a date range, for stress tests and sector returns. Defaults
                to None.
            profile_function (Callable | None, optional): Retrieves the company profiles
                (sector, country, market capitalization). Defaults to None.
            fund_function (Callable | None, optional): Retrieves the sector, country or
                holding weights of funds. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results
                to. Defaults to 4.
            start_date (str | None, optional): The start date of the data. Defaults to None.
            end_date (str | None, optional): The end date of the data. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_maximum_sharpe_weights()
        ```
        """
        self._tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self._historical_data = historical_data if historical_data is not None else {}
        self._risk_free_rate_data = (
            risk_free_rate_data if risk_free_rate_data is not None else {}
        )
        self._benchmark_name = benchmark_name
        self._portfolio_weights = portfolio_weights
        self._price_function = price_function
        self._profile_function = profile_function
        self._fund_function = fund_function
        self._rounding: int | None = rounding
        self._start_date = start_date
        self._end_date = end_date
        self._profile: pd.DataFrame | None = None

    def _get_assets(self, include_benchmark: bool = False) -> list[str]:
        """The assets to build the portfolio from, without the Portfolio pseudo-ticker."""
        assets = [
            ticker
            for ticker in self._tickers
            if ticker not in (PORTFOLIO_TICKER, BENCHMARK_TICKER)
        ]
        if include_benchmark:
            assets.append(BENCHMARK_TICKER)
        return assets

    def _get_returns(self, period: str, include_benchmark: bool) -> pd.DataFrame:
        """The returns of the assets over the periods in which every asset has a return."""
        if period not in PERIODS:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        # Historical data supplied to the Toolkit can reach beyond its date range.
        returns = self._historical_data[period]["Return"].loc[
            self._start_date : self._end_date
        ]
        assets = [
            asset
            for asset in self._get_assets(include_benchmark)
            if asset in returns.columns
        ]
        if len(assets) < 2:  # noqa: PLR2004
            raise ValueError("A portfolio needs the returns of at least two assets.")

        returns = returns[assets].dropna(how="any")
        if len(returns) < len(assets) + 1:
            raise ValueError(
                f"There are only {len(returns)} periods in which every asset has a "
                "return; use a longer date range or a higher frequency."
            )
        return returns

    def _get_risk_free_rate(self, period: str, index: pd.Index) -> float:
        """The average annualized risk free rate over the periods, 0 when unknown."""
        risk_free = self._risk_free_rate_data.get(period)
        risk_free_rate = (
            float(risk_free.reindex(index).mean())
            if risk_free is not None and not risk_free.empty
            else 0.0
        )
        return 0.0 if np.isnan(risk_free_rate) else risk_free_rate

    def _get_benchmark_returns(self, period: str) -> pd.Series | None:
        """The returns of the benchmark per period, None when there is no benchmark."""
        returns = self._historical_data[period]["Return"]
        return returns.get(BENCHMARK_TICKER)

    def _get_inputs(
        self,
        period: str,
        include_benchmark: bool,
        covariance: str = "sample",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
        """The returns, annualized expected returns, covariance and risk free rate."""
        returns = self._get_returns(period, include_benchmark)
        risk_free_rate = self._get_risk_free_rate(period, returns.index)
        expected, covariance_matrix = (
            optimization_model.get_expected_returns_and_covariance(
                returns,
                PERIODS_PER_YEAR[period],
                covariance,
                covariance_span,
                expected_returns,
                expected_returns_span,
                (
                    self._get_benchmark_returns(period)
                    if expected_returns == "capm"
                    else None
                ),
                risk_free_rate,
            )
        )
        return returns, expected, covariance_matrix, risk_free_rate

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

    def _resolve_weights(
        self, weights: dict[str, float] | pd.Series | None, assets: list[str]
    ) -> pd.Series:
        """
        The weights to examine: as given, otherwise the latest weights of the Portfolio the
        Toolkit came from, otherwise equal weights. They are scaled to sum to one.
        """
        if weights is not None:
            resolved = pd.Series(weights, dtype=float)
            unknown = [asset for asset in resolved.index if asset not in assets]
            if unknown:
                raise ValueError(
                    f"The weights name assets that are not loaded: {', '.join(unknown)}."
                )
            resolved = resolved.reindex(assets).fillna(0)
        elif self._portfolio_weights:
            latest = self._portfolio_weights.get("daily", pd.DataFrame())
            latest = latest.iloc[-1] if not latest.empty else pd.Series(dtype=float)
            resolved = latest.reindex(assets).fillna(0).astype(float)
        else:
            resolved = pd.Series(1.0, index=assets)

        if resolved.sum() == 0:
            raise ValueError("The weights sum to zero.")
        # Scaling by a negative sum would flip every sign and turn a short portfolio long.
        if resolved.sum() < 0:
            raise ValueError(
                f"The weights sum to {resolved.sum():.4f}; a portfolio needs a positive net "
                "weight to be scaled to one."
            )
        if abs(resolved.sum() - 1) > WEIGHT_SUM_TOLERANCE:
            logger.info(
                "The weights sum to %.4f and are scaled to one.", resolved.sum()
            )
        return resolved / resolved.sum()

    def _weights_table(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        risk_free_rate: float,
        rounding: int | None,
    ) -> pd.DataFrame:
        """
        The weights with, per asset and for the portfolio, the expected return, volatility,
        Sharpe ratio and share of the portfolio's risk.
        """
        volatility = pd.Series(np.sqrt(np.diag(covariance)), index=covariance.index)
        contributions = optimization_model.get_risk_contributions(weights, covariance)
        table = pd.DataFrame(
            {
                "Weight": weights,
                "Expected Return": expected_returns,
                "Volatility": volatility,
                "Sharpe Ratio": (expected_returns - risk_free_rate) / volatility,
                "Risk Contribution": contributions["Percentage Contribution"],
            }
        )
        statistics = optimization_model.get_portfolio_statistics(
            weights, expected_returns, covariance, risk_free_rate
        )
        table.loc[PORTFOLIO_TICKER] = [
            weights.sum(),
            statistics["Expected Return"],
            statistics["Volatility"],
            statistics["Sharpe Ratio"],
            contributions["Percentage Contribution"].sum(),
        ]
        return self._round(table, rounding)

    def _round(self, table: pd.DataFrame, rounding: int | None) -> pd.DataFrame:
        """The table rounded to the given decimals, or to the Toolkit's rounding when None."""
        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )

    def _get_profile(self) -> pd.DataFrame:
        """The company profiles, retrieved once."""
        if self._profile is None:
            if self._profile_function is None:
                raise ValueError("Company profiles are not available for this Toolkit.")
            self._profile = self._profile_function()
        return self._profile

    def _get_classification(self, assets: list[str], row: str) -> pd.Series:
        """
        The sector or country of each asset from the company profiles, with countries
        named in full ("United States" rather than "US"); empty for an asset without one.
        """
        profile = self._get_profile()
        values = (
            profile.loc[row].reindex(assets)
            if row in profile.index
            else pd.Series(index=assets, dtype=object)
        )
        values = values.where(
            values.map(lambda value: isinstance(value, str) and value != "")
        )
        if row == "Country":
            # Lazily imported: the country names live with the Economics data.
            from financetoolkit.economics.helpers import COUNTRY_CODES  # noqa: PLC0415

            values = values.map(lambda value: COUNTRY_CODES.get(value, value))
        return values

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
                    classification: self._get_classification(assets, classification)
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

    def _get_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """The daily prices of other tickers over a date range."""
        if self._price_function is None:
            raise ValueError(
                "Prices of other tickers are not available for this Toolkit."
            )
        return self._price_function(tickers, start, end)

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
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the minimum variance portfolio: the fully invested
        portfolio of the Toolkit's assets with the lowest volatility. It uses only the
        covariance matrix, not the expected returns, which makes it the most robust of the
        mean-variance portfolios to estimation error.

        The weights minimize w'Σw subject to the weights summing to one and lying between
        the minimum and maximum weight, where Σ is the annualized covariance matrix of the
        returns.

        For more information, see the following paper:

        - Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.

        Also known as: global minimum variance portfolio, GMV, lowest risk portfolio.

        Args:
            period (str, optional): The frequency of the returns the covariance matrix is
                estimated from (daily, weekly, monthly, quarterly or yearly). Defaults to
                "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
          without a sector or country, such as the benchmark, is in none), as
          PyPortfolioOpt's add_sector_constraints and skfolio's group constraints do. A
          gross exposure limit is linearized with one extra variable t_i ≥ |w_i| per asset
          (t_i ≥ w_i, t_i ≥ -w_i, Σ t_i ≤ the limit), so the solution stays exact. A
          maximum turnover from the current weights is linearized the same way, with one
          variable d_i ≥ |w_i - w0_i| per asset and Σ d_i ≤ the limit (Lobo, Fazel and
          Boyd, 2007). A combination of constraints no portfolio can meet raises an error
          before the optimization runs.

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
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of the maximum Sharpe ratio portfolio: the fully invested
        portfolio with the highest expected return per unit of volatility above the risk
        free rate. It is the tangency portfolio, where a line from the risk free rate
        touches the efficient frontier.

        The weights maximize (w'μ - r_f) / sqrt(w'Σw) subject to the weights summing to one
        and lying between the minimum and maximum weight, with μ the annualized expected
        returns, Σ the annualized covariance matrix and r_f the risk free rate.

        For more information, see the following papers:

        - Tobin, J. (1958). "Liquidity Preference as Behavior Towards Risk." The Review of
          Economic Studies, 25(2), 65-86.
        - Sharpe, W.F. (1966). "Mutual Fund Performance." The Journal of Business, 39(1),
          119-138.

        Also known as: tangency portfolio, optimal risky portfolio, max Sharpe.

        Args:
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
                from the portfolio's annualized expected return. Defaults to 0.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution. Returns and volatilities are
            annualized.

        Notes:
        - The expected returns are the noisiest input of the optimization, so with the
          historical averages the weights tend to concentrate in the assets that happened
          to do best. A maximum weight, the CAPM or James-Stein expected returns or the
          Black-Litterman returns temper that.
        - A transaction cost lowers the expected return of the portfolio by the cost
          times the turnover from the current weights, so the optimizer only trades when
          the gain in Sharpe ratio pays for it (as skfolio's transaction_costs). The cost
          is taken once from an annualized return, which spreads it over a year of
          holding the portfolio.

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
            transaction_cost=transaction_cost,
        )
        table = self._weights_table(
            weights, expected, covariance_matrix, risk_free_rate, rounding
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

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
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the weights of a mean-variance efficient portfolio. With a target return,
        it is the portfolio with the lowest volatility that expects that return. Without
        one, it is the portfolio that maximizes the investor's utility, the expected return
        minus half the risk aversion times the variance: w'μ - (λ/2) w'Σw.

        A risk aversion of 2 to 4 is typical; 2.5 is the value commonly used for the
        market as a whole.

        For more information, see the following paper:

        - Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.

        Also known as: Markowitz portfolio, efficient portfolio, quadratic utility
        portfolio.

        Args:
            target_return (float | None, optional): The annualized return the portfolio must
                expect, e.g. 0.12 for 12%. Defaults to None, which uses the risk aversion.
            risk_aversion (float, optional): The investor's risk aversion λ; higher values
                give less risky portfolios. Defaults to 2.5.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
                from the portfolio's annualized expected return. Defaults to 0.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution.

        Notes:
        - A target return outside what the assets can reach within the constraints
          raises an error that states the reachable range.

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
            target_return,
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
            transaction_cost=transaction_cost,
        )
        table = self._weights_table(
            weights, expected, covariance_matrix, risk_free_rate, rounding
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

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
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier: the portfolios with the lowest volatility for
        each level of expected return, from the minimum variance portfolio up to the
        highest return the weight bounds allow. Every portfolio below the frontier takes
        more risk than needed for its return.

        For more information, see the following paper:

        - Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.

        Also known as: Markowitz frontier, mean-variance frontier, risk-return frontier.

        Args:
            points (int, optional): The number of portfolios on the frontier. Defaults to 20.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the risk parity weights: the portfolio in which every asset contributes
        the same share of the volatility, or the share set by a risk budget. Volatile
        assets get smaller weights, and correlated assets share their risk budget, so no
        single asset dominates the portfolio's risk.

        The weights solve w_i (Σw)_i / σ_p = b_i σ_p for every asset, found by minimizing the
        convex function 1/2 w'Σw - Σ b_i ln(w_i) and scaling the result to sum to one.

        For more information, see the following papers:

        - Maillard, S., Roncalli, T. and Teïletche, J. (2010). "The Properties of Equally
          Weighted Risk Contribution Portfolios." The Journal of Portfolio Management,
          36(4), 60-70.
        - Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights." SSRN Working
          Paper.
        - Richard, J.-C. and Roncalli, T. (2019). "Constrained Risk Budgeting Portfolios:
          Theory, Algorithms, Applications & Puzzles." arXiv:1902.05710.

        Also known as: equal risk contribution, ERC, risk budgeting.

        Args:
            risk_budget (dict[str, float] | None, optional): The share of the risk per
                asset, e.g. {"AAPL": 0.4, "MSFT": 0.2, ...}; it is scaled to sum to one.
                Defaults to None, an equal share for every asset.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
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
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution.

        Notes:
        - Weight, sector and country bounds turn it into constrained risk budgeting
          (Richard and Roncalli, 2019): the bounds are linear constraints on w = y / Σy,
          so the problem stays convex, the same as skfolio's RiskBudgeting. Where a bound
          binds, the risk contributions can no longer all match the budget.
        - Risk parity holds long positions only and is fully invested, so it takes no
          negative minimum weight and no net or gross exposure.

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
        _, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        weights = optimization_model.get_risk_parity_weights(
            covariance_matrix,
            pd.Series(risk_budget, dtype=float) if risk_budget is not None else None,
            **self._get_constraints(
                covariance_matrix.index.tolist(),
                minimum_weight,
                maximum_weight,
                sector_bounds,
                country_bounds,
            ),
        )
        return self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )

    @handle_errors
    def get_hierarchical_risk_parity_weights(
        self,
        linkage_method: str = "single",
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hierarchical Risk Parity (HRP) weights. The assets are clustered by
        their correlations, ordered so that similar assets sit next to each other, and the
        weight is split top-down between the two halves of every cluster in inverse
        proportion to their variance. It never inverts the covariance matrix, so it stays
        stable with many or highly correlated assets, where mean-variance weights become
        erratic.

        As in the paper, the clustering uses the Euclidean distance between the columns of
        the correlation distance matrix √((1 - ρ) / 2): two assets are close when they
        relate to all the other assets in the same way. Some libraries cluster on the
        correlation distance itself, which can order the assets differently.

        For more information, see the following paper:

        - López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of
          Sample." The Journal of Portfolio Management, 42(4), 59-69.

        Also known as: HRP, hierarchical clustering portfolio.

        Args:
            linkage_method (str, optional): The linkage of the hierarchical clustering:
                "single", "complete", "average" or "ward". Defaults to "single", as in the
                paper.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
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

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution.

        Notes:
        - Weight bounds move each split just enough to keep both halves within the sum of
          their assets' bounds, as skfolio and Riskfolio-Lib do.
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
        _, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        weights = optimization_model.get_hierarchical_risk_parity_weights(
            covariance_matrix, linkage_method, minimum_weight, maximum_weight
        )
        return self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )

    @handle_errors
    def get_hierarchical_equal_risk_contribution_weights(
        self,
        linkage_method: str = "ward",
        clusters: int | None = None,
        period: str = "daily",
        covariance: str = "sample",
        covariance_span: int = 180,
        minimum_weight: float | dict[str, float] = 0.0,
        maximum_weight: float | dict[str, float] = 1.0,
        sector_bounds: dict[str, tuple[float, float]] | None = None,
        country_bounds: dict[str, tuple[float, float]] | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Hierarchical Equal Risk Contribution (HERC) weights. The assets are
        clustered by their correlations and the dendrogram is cut into clusters. Within a
        cluster every asset gets a weight in inverse proportion to its variance; between
        clusters the weight is split top-down along the dendrogram, each branch getting a
        weight in inverse proportion to its risk, the sum of the variances of its clusters.
        Unlike Hierarchical Risk Parity, which halves the ordered list of assets, the
        splits follow the tree, so assets that belong together are never split up, and
        the weight is allocated over the clusters rather than over the assets: riskier
        clusters get less.

        The clustering uses the correlation distance √((1 - ρ) / 2) with Ward linkage, as
        in the paper. Without a number of clusters, it is the one at the sharpest bend of
        the pooled within-cluster dispersion, the two-order difference of the gap
        statistic (Yue, Wang and Wei, 2008; Tibshirani, Walther and Hastie, 2001).

        For more information, see the following papers:

        - Raffinot, T. (2018). "The Hierarchical Equal Risk Contribution Portfolio."
          SSRN 3237540.
        - Raffinot, T. (2017). "Hierarchical Clustering-Based Asset Allocation." The
          Journal of Portfolio Management, 44(2), 89-99.

        Also known as: HERC, hierarchical clustering-based risk parity.

        Args:
            linkage_method (str, optional): The linkage of the hierarchical clustering:
                "single", "complete", "average" or "ward". Defaults to "ward", as in the
                paper.
            clusters (int | None, optional): The number of clusters. Defaults to None,
                chosen by the two-order difference of the gap statistic.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix, which also
                gives the correlations that are clustered: "sample" (ddof=1),
                "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the clusters when there are many assets compared to the number
                of periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
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
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the Cluster of every asset
            (numbered from 1). Returns and volatilities are annualized.

        Notes:
        - Weight, sector and country bounds are applied after the allocation as skfolio
          does: when the weights fall outside them, they move to the weights within the
          bounds with the smallest relative change, Σ (w_i / w_HERC,i - 1)².
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
        _, expected_returns, covariance_matrix, risk_free_rate = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
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
        )
        table = self._weights_table(
            weights, expected_returns, covariance_matrix, risk_free_rate, rounding
        )
        _, labels = optimization_model.get_hierarchical_clusters(
            covariance_matrix, linkage_method, clusters
        )
        table["Cluster"] = pd.Series(labels + 1, index=covariance_matrix.index)
        return table

    @handle_errors
    def get_nested_clustered_optimization_weights(
        self,
        objective: str = "minimum_variance",
        linkage_method: str = "ward",
        clusters: int | None = None,
        period: str = "daily",
        covariance: str = "denoised",
        covariance_span: int = 180,
        expected_returns: str = "mean",
        expected_returns_span: int = 180,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Nested Clustered Optimization (NCO) weights. The assets are
        clustered by their correlations, the objective is optimized within every cluster,
        and then once more across the clusters, each cluster acting as one asset that holds
        its own optimal portfolio. Estimation noise in the covariance matrix mostly sits
        among correlated assets, which is what makes mean-variance weights unstable; by
        optimizing within and across clusters separately, NCO keeps that noise contained
        and the weights steadier out of sample.

        With W the matrix whose column k holds the optimal weights within cluster k, the
        clusters have the expected returns W'μ and the covariance matrix W'ΣW; the weights
        across the clusters v are optimal for those, and the weight of every asset is
        w = W v. The clusters are found as in HERC (correlation distance, Ward linkage,
        the number of clusters from the two-order difference of the gap statistic) and the
        covariance matrix is denoised by default, as in the paper.

        For more information, see the following papers:

        - López de Prado, M. (2019). "A Robust Estimator of the Efficient Frontier." SSRN
          3469961.
        - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge
          University Press, chapter 7.

        Also known as: NCO, nested clustered optimisation.

        Args:
            objective (str, optional): The objective at both levels: "minimum_variance",
                "maximum_sharpe" or "risk_parity" (equal risk contribution). Defaults to
                "minimum_variance".
            linkage_method (str, optional): The linkage of the hierarchical clustering:
                "single", "complete", "average" or "ward". Defaults to "ward".
            clusters (int | None, optional): The number of clusters. Defaults to None,
                chosen by the two-order difference of the gap statistic.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix, which also
                gives the correlations that are clustered: "denoised" (the eigenvalues of
                the correlation matrix that random returns could produce averaged, Laloux
                et al., 2000), "sample" (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004),
                "oas" (Chen et al., 2010), "semicovariance" (the co-movement below a zero
                return, Estrada, 2007) or "exponential" (recent periods weigh most,
                RiskMetrics). Defaults to "denoised", as in the paper.
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            expected_returns (str, optional): The estimator of the expected returns of
                the "maximum_sharpe" objective: "mean" (the historical average),
                "exponential" (the exponentially weighted average, recent periods weigh
                most), "capm" (the returns the Capital Asset Pricing Model implies from
                each asset's beta to the benchmark, Sharpe, 1964) or "james_stein" (the
                averages shrunk towards that of the minimum variance portfolio, Jorion,
                1986). Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the
                "exponential" expected returns. Defaults to 180.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the Cluster of every asset
            (numbered from 1). Returns and volatilities are annualized.

        Notes:
        - Both levels are long only and fully invested, as Riskfolio-Lib and skfolio do by
          default. Weight, sector and country bounds and a net or gross exposure do not
          apply: the weight of an asset is the product of its weight within its cluster
          and the cluster's weight, from two separate problems, so a bound on it is not a
          constraint either problem can hold.
        - With "maximum_sharpe", a cluster without an asset whose expected return
          exceeds the risk free rate holds its minimum variance portfolio. Across the
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
        _, expected, covariance_matrix, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            covariance,
            covariance_span,
            expected_returns,
            expected_returns_span,
        )
        weights = optimization_model.get_nested_clustered_optimization_weights(
            expected,
            covariance_matrix,
            objective,
            risk_free_rate,
            linkage_method,
            clusters,
        )
        table = self._weights_table(
            weights, expected, covariance_matrix, risk_free_rate, rounding
        )
        _, labels = optimization_model.get_hierarchical_clusters(
            covariance_matrix, linkage_method, clusters
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
    ) -> pd.DataFrame:
        """
        Calculate inverse volatility weights: each asset's weight is proportional to one
        over its volatility, w_i = (1/σ_i) / Σ(1/σ_j). It is the simplest risk-based
        allocation and equals risk parity when all correlations are the same.

        Also known as: naive risk parity, volatility weighting.

        Args:
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

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
    def get_minimum_cvar_weights(
        self,
        confidence_level: float = 0.95,
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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the minimum Conditional Value at Risk (CVaR) portfolio:
        the fully invested portfolio whose average loss in the worst periods is the
        smallest. Unlike variance, CVaR only looks at the losses in the tail, so it accounts
        for skewed and fat-tailed returns.

        The problem is solved as a linear programme over the historical returns: minimize
        VaR + 1/((1 - α)T) Σ max(0, -r_t'w - VaR), with α the confidence level and T the
        number of periods.

        For more information, see the following paper:

        - Rockafellar, R.T. and Uryasev, S. (2000). "Optimization of Conditional
          Value-at-Risk." Journal of Risk, 2(3), 21-41.

        Also known as: minimum expected shortfall portfolio, mean-CVaR, tail risk
        optimization.

        Args:
            confidence_level (float, optional): The confidence level of the CVaR, e.g. 0.95
                for the average of the worst 5% of periods. Defaults to 0.95.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the CVaR of every asset
            on its own and of the portfolio (per period, as a loss) in the CVaR column.

        Notes:
        - The CVaR column is the value the linear programme minimizes: the average of the
          worst 1 - α share of the periods, with a fraction of the boundary period when
          (1 - α)T is not a whole number (Rockafellar and Uryasev, 2002), as in the
          maximum CVaR ratio weights and the expected shortfall contributions.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_minimum_cvar_weights()
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |   CVaR |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|-------:|
        | AAPL      |   0      |            0.1932 |       0.2786 |         0.5725 |              0      | 0.039  |
        | MSFT      |   0.1827 |            0.2017 |       0.2571 |         0.6534 |              0.1513 | 0.0356 |
        | JNJ       |   0.5965 |            0.0987 |       0.1673 |         0.3884 |              0.6005 | 0.0227 |
        | XOM       |   0.2208 |            0.2911 |       0.2702 |         0.9524 |              0.2482 | 0.0373 |
        | Portfolio |   1      |            0.16   |       0.136  |         0.9287 |              1      | 0.0191 |
        """
        returns, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        current = self._get_current_weights(
            current_weights,
            returns.columns.tolist(),
            maximum_turnover is not None,
        )
        weights = optimization_model.get_minimum_cvar_weights(
            returns,
            confidence_level,
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

        # The CVaR of every asset alone and of the portfolio as a positive loss, the
        # Rockafellar-Uryasev tail average that the linear programme minimizes.
        table["CVaR"] = (
            -returns.assign(**{PORTFOLIO_TICKER: returns @ weights})
        ).apply(
            optimization_model.get_expected_tail_loss, confidence_level=confidence_level
        )
        return self._round(self._add_turnover(table, weights, current), rounding)

    @handle_errors
    def get_minimum_drawdown_weights(
        self,
        measure: str = "cdar",
        confidence_level: float = 0.95,
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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the portfolio with the lowest drawdown risk over the
        historical returns: the lowest Conditional Drawdown at Risk (CDaR, the average of
        the worst drawdowns), the lowest maximum drawdown or the lowest average drawdown.
        Drawdowns are what investors feel and what ends strategies, and unlike volatility
        or CVaR they account for losses that build up over consecutive periods.

        The problem is a linear programme over the uncompounded cumulative returns
        y_t = Σ_(s≤t) r_s'w, with the running peak u_t ≥ y_t, u_t ≥ u_(t-1) from u_0 = 0
        and the drawdown u_t - y_t. It minimizes ζ + Σ max(0, u_t - y_t - ζ) / ((1 - α)T)
        for the CDaR with α the confidence level and T the number of periods, the largest
        drawdown for the maximum and the average over all periods for the average.

        For more information, see the following paper:

        - Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in
          Portfolio Optimization." International Journal of Theoretical and Applied
          Finance, 8(1), 13-58.

        Also known as: minimum CDaR, minimum maximum drawdown, minimum average drawdown,
        drawdown optimization.

        Args:
            measure (str, optional): The drawdown measure to minimize: "cdar", "maximum" or
                "average". Defaults to "cdar".
            confidence_level (float, optional): The confidence level of the CDaR, e.g. 0.95
                for the average of the worst 5% of drawdowns. Defaults to 0.95.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
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
                invested portfolio. A net exposure of 0 is refused, as holding nothing
                has no drawdown. Defaults to 1.
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

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the drawdown measure that
            was minimized (CDaR, Maximum Drawdown or Average Drawdown) of every asset on its
            own and of the portfolio.

        Notes:
        - The drawdowns are on uncompounded returns (the cumulative sum of the returns, as
          in the paper, PyPortfolioOpt, Riskfolio-Lib and skfolio), which keeps the problem
          linear: they are absolute drawdowns of a portfolio that keeps its invested amount
          constant. The compounded drawdowns of the risk module, relative to the peak
          value, are close for small drawdowns and smaller for large ones.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_minimum_drawdown_weights()
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |   CDaR |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|-------:|
        | AAPL      |   0.152  |            0.1932 |       0.2786 |         0.5725 |              0.2007 | 0.2553 |
        | MSFT      |   0.1761 |            0.2017 |       0.2571 |         0.6534 |              0.1887 | 0.3173 |
        | JNJ       |   0.4312 |            0.0987 |       0.1673 |         0.3884 |              0.3284 | 0.1573 |
        | XOM       |   0.2407 |            0.2911 |       0.2702 |         0.9524 |              0.2822 | 0.178  |
        | Portfolio |   1      |            0.1775 |       0.1412 |         1.0185 |              1      | 0.0769 |
        """
        returns, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        current = self._get_current_weights(
            current_weights, returns.columns.tolist(), maximum_turnover is not None
        )
        weights = optimization_model.get_minimum_drawdown_weights(
            returns,
            measure,
            confidence_level,
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

        drawdowns = optimization_model.get_drawdowns(
            returns.assign(**{PORTFOLIO_TICKER: returns @ weights})
        )
        if measure == "cdar":
            table["CDaR"] = drawdowns.apply(
                optimization_model.get_expected_tail_loss,
                confidence_level=confidence_level,
            )
        elif measure == "maximum":
            table["Maximum Drawdown"] = drawdowns.max()
        else:
            table["Average Drawdown"] = drawdowns.mean()
        return self._round(self._add_turnover(table, weights, current), rounding)

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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the portfolio with the highest Omega ratio over the
        historical returns: the gains above a minimum acceptable return divided by the
        losses below it. The Omega ratio uses the whole return distribution instead of its
        mean and variance, so it rewards positive skew and penalizes fat left tails that a
        Sharpe ratio overlooks.

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

        Args:
            minimum_acceptable_return (float, optional): The threshold between gains and
                losses, as a return per period of the chosen frequency (e.g. 0.0002 a day).
                Defaults to 0.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
        expected_per_period = expected / PERIODS_PER_YEAR[period]
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
    def get_maximum_cvar_ratio_weights(
        self,
        confidence_level: float = 0.95,
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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the portfolio with the highest CVaR ratio: the average
        return above the risk free rate divided by the Conditional Value at Risk, the
        average loss in the worst periods. It is the maximum Sharpe ratio portfolio with
        the tail loss in place of the volatility, so upside swings are not penalized and
        crash risk is.

        The ratio (r̄ - r_f)'w / CVaR_α(w), with the CVaR of Rockafellar and Uryasev
        (2000), is maximized exactly as a linear programme with the Charnes-Cooper
        transformation y = κw, κ > 0: maximize (r̄ - r_f)'y subject to
        ζ + Σ max(0, -r_t'y - ζ) / ((1 - α)T) ≤ 1 and the weight constraints, then
        w = y / κ.

        For more information, see the following papers:

        - Martin, R. D., Rachev, S. T. and Siboulet, F. (2003). "Phi-Alpha Optimal
          Portfolios and Extreme Risk Management." Wilmott Magazine, November, 70-83.
        - Rockafellar, R. T. and Uryasev, S. (2000). "Optimization of Conditional
          Value-at-Risk." Journal of Risk, 2(3), 21-41.

        Also known as: maximum STARR, stable tail adjusted return ratio, maximum
        return-to-CVaR, mean-CVaR tangency portfolio.

        Args:
            confidence_level (float, optional): The confidence level of the CVaR, e.g. 0.95
                for the average of the worst 5% of periods. Defaults to 0.95.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the CVaR (per period, as a
            loss) and the CVaR Ratio (the average excess return per period over the CVaR)
            of every asset on its own and of the portfolio.

        Notes:
        - The risk free rate is the Toolkit's, averaged over the returns and converted to
          a rate per period. Some portfolio within the constraints needs an average return
          above it; otherwise no CVaR ratio is positive and the method returns an error
          that says so.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_maximum_cvar_ratio_weights()[
            ["Weight", "Expected Return", "Risk Contribution", "CVaR", "CVaR Ratio"]
        ]
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Risk Contribution |   CVaR |   CVaR Ratio |
        |:----------|---------:|------------------:|--------------------:|-------:|-------------:|
        | AAPL      |   0      |            0.1932 |              0      | 0.039  |       0.0162 |
        | MSFT      |   0.277  |            0.2017 |              0.2061 | 0.0356 |       0.0187 |
        | JNJ       |   0.1978 |            0.0987 |              0.064  | 0.0227 |       0.0113 |
        | XOM       |   0.5253 |            0.2911 |              0.7299 | 0.0373 |       0.0274 |
        | Portfolio |   1      |            0.2283 |              1      | 0.0237 |       0.0326 |
        """
        returns, expected, covariance, risk_free_rate = self._get_inputs(
            period,
            include_benchmark,
            expected_returns=expected_returns,
            expected_returns_span=expected_returns_span,
        )
        # The numerator of the ratio takes the expected returns per period; the losses
        # in its denominator are those of the historical returns.
        expected_per_period = expected / PERIODS_PER_YEAR[period]
        current = self._get_current_weights(
            current_weights, returns.columns.tolist(), maximum_turnover is not None
        )
        risk_free_per_period = risk_free_rate / PERIODS_PER_YEAR[period]
        weights = optimization_model.get_maximum_cvar_ratio_weights(
            returns,
            confidence_level,
            risk_free_per_period,
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

        with_portfolio = returns.assign(**{PORTFOLIO_TICKER: returns @ weights})
        cvar = (-with_portfolio).apply(
            optimization_model.get_expected_tail_loss, confidence_level=confidence_level
        )
        excess = expected_per_period - risk_free_per_period
        excess[PORTFOLIO_TICKER] = excess @ weights
        table["CVaR"] = cvar
        table["CVaR Ratio"] = excess / cvar
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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the maximum diversification portfolio: the fully invested
        portfolio with the highest diversification ratio, the weighted average volatility of
        its assets divided by its own volatility. The ratio is one for a single asset and
        grows as the assets move less together, so the portfolio holds the assets that
        diversify each other most. It needs no expected returns, and when every asset
        has the same Sharpe ratio it is also the maximum Sharpe ratio portfolio.

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

        Args:
            period (str, optional): The frequency of the returns the covariance matrix is
                estimated from (daily, weekly, monthly, quarterly or yearly). Defaults to
                "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the portfolio of the Toolkit's assets that follows the
        benchmark most closely: the fully invested portfolio with the lowest tracking error,
        the volatility of its return minus the benchmark's. With the objective
        "information_ratio" it is instead the portfolio with the highest information ratio,
        the expected return above the benchmark per unit of tracking error, which is what an
        active manager measured against the benchmark maximizes.

        The tracking error is TE(w) = √Var(R w - r_b) = √(w'Σ_AA w - 2 w'Σ_Ab + σ_b²), with
        Σ_AA the annualized covariance matrix of the assets, Σ_Ab their covariances with the
        benchmark and σ_b² the benchmark's variance. The information ratio is
        (w'μ - μ_b) / TE(w), with μ and μ_b the annualized expected returns. The weights sum
        to one and lie between the minimum and maximum weight; the benchmark is the
        Toolkit's (SPY by default) and is not one of the assets.

        For more information, see the following papers:

        - Roll, R. (1992). "A Mean/Variance Analysis of Tracking Error." The Journal of
          Portfolio Management, 18(4), 13-22.
        - Jorion, P. (2003). "Portfolio Optimization with Tracking-Error Constraints."
          Financial Analysts Journal, 59(5), 70-82.
        - Grinold, R.C. and Kahn, R.N. (2000). "Active Portfolio Management." 2nd edition,
          McGraw-Hill.

        Also known as: index tracking, benchmark tracking, minimum active risk, maximum
        information ratio portfolio.

        Args:
            objective (str, optional): "tracking_error" for the lowest tracking error or
                "information_ratio" for the highest information ratio. Defaults to
                "tracking_error".
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix of the assets
                and the benchmark: "sample" (ddof=1), "ledoit_wolf" (Ledoit and Wolf,
                2004), "oas" (Chen et al., 2010), "semicovariance" (the co-movement below a
                zero return, Estrada, 2007), "exponential" (recent periods weigh most,
                RiskMetrics) or "denoised" (the eigenvalues of the correlation matrix that
                random returns could produce averaged, Laloux et al., 2000). Defaults to
                "sample".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
        )
        assets = weights.index
        table = self._weights_table(
            weights,
            expected[assets],
            covariance_matrix.loc[assets, assets],
            risk_free_rate,
            rounding,
        )
        tracking = pd.DataFrame(
            {
                asset: optimization_model.get_tracking_error(
                    pd.Series({asset: 1.0}),
                    expected,
                    covariance_matrix,
                    BENCHMARK_TICKER,
                )
                for asset in assets
            }
        ).T
        tracking.loc[PORTFOLIO_TICKER] = optimization_model.get_tracking_error(
            weights, expected, covariance_matrix, BENCHMARK_TICKER
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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the growth optimal (Kelly) portfolio: the fully invested
        portfolio with the highest expected logarithmic growth of wealth, which in the long
        run ends up with more wealth than any other strategy with near certainty. Because
        the full Kelly portfolio is volatile and sensitive to errors in the expected
        returns, investors often hold a fraction of it (fractional Kelly) and keep the rest
        in cash.

        The "exact" method maximizes the average log return over the historical returns,
        (1/T) Σ ln(1 + r_t'w). The "approximate" method maximizes its second-order
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

        Args:
            method (str, optional): "exact" (the historical log returns) or "approximate"
                (the mean-variance approximation). Defaults to "exact".
            fraction (float, optional): The share of the growth optimal portfolio to hold,
                e.g. 0.5 for half Kelly; the rest is cash. Defaults to 1.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). The log growth compounds per period, so it needs
                non-overlapping returns; daily returns keep every period's wealth positive.
                Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix, used by
                the "approximate" method and for the volatility and risk contributions:
                "sample" (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et
                al., 2010), "semicovariance" (the co-movement below a zero return, Estrada,
                2007), "exponential" (recent periods weigh most, RiskMetrics) or "denoised"
                (the eigenvalues of the correlation matrix that random returns could
                produce averaged, Laloux et al., 2000). Defaults to "sample".
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset, for the Cash (when the fraction is not one) and for the
            portfolio, the Weight, Expected Return, Volatility, Sharpe Ratio and Risk
            Contribution, with the Log Growth, the annualized average log return of each
            asset on its own and of the portfolio. Returns and volatilities are annualized.

        Notes:
        - With historical average returns the full Kelly weights tend to concentrate in the
          assets that happened to grow fastest; a fraction below one or a maximum weight
          tempers that. The expected returns only enter the "approximate" method; the
          "exact" method uses the returns themselves.
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
        periods_per_year = PERIODS_PER_YEAR[period]
        cash = 1 - fraction * net_exposure
        growth = np.log1p(returns).mean() * periods_per_year
        portfolio_growth = (
            np.log1p(
                returns @ weights + cash * risk_free_rate / periods_per_year
            ).mean()
            * periods_per_year
        )

        # The cash is an asset without risk that earns the risk free rate, so the
        # portfolio's expected return and Sharpe ratio include it.
        if cash != 0:
            weights["Cash"] = cash
            expected["Cash"] = risk_free_rate
            covariance_matrix = covariance_matrix.reindex(
                index=weights.index, columns=weights.index, fill_value=0.0
            )
            growth["Cash"] = (
                np.log1p(risk_free_rate / periods_per_year) * periods_per_year
            )

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
    ) -> pd.DataFrame:
        """
        Calculate the weights of the resampled efficient portfolio (Michaud resampling):
        the average of the optimal weights over many return histories that could have
        been observed instead of the actual one. A single optimization takes the
        estimated expected returns and covariances at face value and piles into the
        assets whose estimates happen to be highest; averaging over resampled histories
        only gives an asset a large weight when it earns it in most of them, which makes
        the weights more diversified and more stable from one estimate to the next.

        Every draw has as many periods as the returns. With "parametric" resampling, as in
        Michaud, the draws come from a multivariate normal distribution with the
        estimated expected returns and covariance; with "bootstrap" they are periods of
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

        Args:
            objective (str, optional): The portfolio optimized in every draw:
                "maximum_sharpe", "minimum_variance" or "mean_variance" (with the risk
                aversion). Defaults to "maximum_sharpe".
            resamples (int, optional): The number of resampled return histories. The
                average settles as their number grows; 500 keeps the weights within
                about a percentage point of their limit for a handful of assets.
                Defaults to 500.
            resampling (str, optional): How the histories are drawn: "parametric" (a
                multivariate normal distribution with the estimated expected returns and
                covariance) or "bootstrap" (the actual periods, drawn with
                replacement). Defaults to "parametric".
            seed (int | None, optional): The seed of the random draws; the same seed
                gives the same weights. Defaults to 0; None draws anew on every call.
            risk_aversion (float, optional): The risk aversion of the "mean_variance"
                objective. Defaults to 2.5.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix the
                "parametric" draws come from and the table reports: "sample" (ddof=1),
                "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            expected_returns (str, optional): The estimator of the expected returns the
                "parametric" draws are centred on: "mean" (the historical average),
                "exponential" (the exponentially weighted average, recent periods weigh
                most), "capm" (the returns the Capital Asset Pricing Model implies from
                each asset's beta to the benchmark, Sharpe, 1964) or "james_stein" (the
                averages shrunk towards that of the minimum variance portfolio, Jorion,
                1986). Defaults to "mean".
            expected_returns_span (int, optional): The span in periods of the
                "exponential" expected returns. Defaults to 180.
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
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
          sample covariance; the estimators above set the distribution the "parametric"
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
            PERIODS_PER_YEAR[period],
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
        risk_aversion: float = 2.5,
        tau: float = 0.05,
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
    ) -> pd.DataFrame:
        """
        Calculate the Black-Litterman expected returns and the weights they lead to. The
        starting point is the return the market implies through its capitalization
        weights: the excess return Π = δΣw_mkt at which holding the market is optimal, plus
        the risk free rate, so the Implied Return is a total return. Views then move these
        returns towards the investor's expectations, in proportion to the confidence in
        each view. The result avoids the extreme weights that historical average returns
        give.

        Views are absolute ({"AAPL": 0.10}: AAPL returns 10% a year in total, risk free
        rate included) or relative ({"AAPL-MSFT": 0.02}: AAPL beats MSFT by 2% a year).
        Without a confidence, a view is as uncertain as the variance of the view portfolio
        scaled by tau; a confidence c scales that by (1 - c) / c, Idzorek's confidence
        method in the closed form of Walters (2014). The weights are the mean-variance
        portfolio of the Black-Litterman returns with the same risk aversion. Without views,
        or with relative views only, the risk free rate shifts every return by the same
        amount and leaves the weights unchanged; an absolute view is compared with the
        implied total return, so there it matters.

        For more information, see the following papers:

        - Black, F. and Litterman, R. (1992). "Global Portfolio Optimization." Financial
          Analysts Journal, 48(5), 28-43.
        - He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman Model
          Portfolios." Goldman Sachs Investment Management Research.
        - Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model:
          Incorporating User-Specified Confidence Levels." Working paper.
        - Walters, J. (2014). "The Black-Litterman Model in Detail."
          https://ssrn.com/abstract=1314585

        Also known as: BL model, equilibrium returns with views.

        Args:
            views (dict[str, float] | None, optional): The annualized views, absolute views
                as total returns, as described above. Defaults to None, which returns the
                market implied returns.
            view_confidences (dict[str, float] | None, optional): The confidence per view,
                between 0 and 1. Defaults to None, which sets each view's uncertainty to the
                variance of the view scaled by tau.
            market_weights (dict[str, float] | None, optional): The market weight per asset.
                Defaults to None, which uses the market capitalizations of the assets.
            risk_aversion (float, optional): The risk aversion δ of the market. Defaults to
                2.5.
            tau (float, optional): The uncertainty of the implied returns relative to the
                covariance matrix. Defaults to 0.05.
            period (str, optional): The frequency of the returns the covariance matrix is
                estimated from. Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
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
                invested portfolio, 0 for a market neutral one. Defaults to 1.
            gross_exposure (float | None, optional): The highest sum of the absolute
                weights, which limits the leverage of a long-short portfolio, e.g. 1.6 for
                a 130/30 portfolio. Defaults to None, no limit.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset its Market Weight, Implied Return, Black-Litterman Return
            (both annualized total returns) and the resulting Weight.

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

        |      |   Market Weight |   Implied Return |   Black-Litterman Return |   Weight |
        |:-----|----------------:|-----------------:|-------------------------:|---------:|
        | AAPL |          0.4899 |           0.1752 |                   0.1769 |   0.4738 |
        | MSFT |          0.376  |           0.1531 |                   0.155  |   0.3729 |
        | JNJ  |          0.0614 |           0.0509 |                   0.052  |   0.0145 |
        | XOM  |          0.0727 |           0.0709 |                   0.0853 |   0.1387 |
        """
        _, _, covariance_matrix, risk_free_rate = self._get_inputs(
            period, False, covariance, covariance_span
        )
        assets = covariance_matrix.index.tolist()

        if market_weights is not None:
            market = pd.Series(market_weights, dtype=float).reindex(assets)
        else:
            profile = self._get_profile()
            market = pd.to_numeric(
                profile.loc["Market Capitalization"].reindex(assets), errors="coerce"
            )
        if market.isna().any():
            missing = ", ".join(market[market.isna()].index)
            raise ValueError(
                f"No market weight for {missing}; pass market_weights explicitly."
            )

        returns = optimization_model.get_black_litterman_returns(
            covariance_matrix,
            market,
            views=views,
            view_confidences=view_confidences,
            risk_aversion=risk_aversion,
            tau=tau,
            risk_free_rate=risk_free_rate,
        )
        weights = optimization_model.get_mean_variance_weights(
            returns["Black-Litterman Return"],
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
            [(market / market.sum()).rename("Market Weight"), returns, weights], axis=1
        )
        return self._round(table, rounding)

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
    ) -> pd.DataFrame:
        """
        Calculate how much each asset contributes to the risk of a portfolio, measured as
        its volatility or its expected shortfall (CVaR). The marginal contribution is the
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

        For more information, see the following papers:

        - Litterman, R. (1996). "Hot Spots and Hedges." The Journal of Portfolio Management,
          23(5), 52-75.
        - Tasche, D. (2000). "Conditional Expectations as Quantile Derivatives." Working
          paper, arXiv math/0104190.
        - Scaillet, O. (2004). "Nonparametric Estimation and Sensitivity Analysis of
          Expected Shortfall." Mathematical Finance, 14(1), 115-129.

        Also known as: risk decomposition, marginal contribution to risk, risk budget,
        component expected shortfall, component CVaR.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            risk_measure (str, optional): "volatility" or "cvar" (the expected shortfall).
                Defaults to "volatility".
            confidence_level (float, optional): The confidence level of the expected
                shortfall, e.g. 0.95 for the average of the worst 5% of periods.
                Defaults to 0.95.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            covariance (str, optional): The estimator of the covariance matrix: "sample"
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and in total, the Weight, Marginal Contribution, Risk
            Contribution and Percentage Contribution: annualized volatility terms, or
            expected shortfall terms per period as positive losses.

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
        if risk_measure not in ("volatility", "cvar"):
            raise ValueError(
                f"The risk measure must be 'volatility' or 'cvar', not '{risk_measure}'."
            )
        returns, _, covariance_matrix, _ = self._get_inputs(
            period, include_benchmark, covariance, covariance_span
        )
        resolved = self._resolve_weights(weights, covariance_matrix.index.tolist())
        table = (
            optimization_model.get_risk_contributions(resolved, covariance_matrix)
            if risk_measure == "volatility"
            else optimization_model.get_expected_shortfall_contributions(
                resolved, returns, confidence_level
            )
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
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
            covariance_span (int, optional): The span in periods of the "exponential"
                covariance; the weight of a period halves after about span / 2.9
                periods. Defaults to 180.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

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
                (ddof=1), "ledoit_wolf" (Ledoit and Wolf, 2004), "oas" (Chen et al., 2010),
                "semicovariance" (the co-movement below a zero return, Estrada, 2007),
                "exponential" (recent periods weigh most, RiskMetrics) or "denoised" (the
                eigenvalues of the correlation matrix that random returns could produce
                averaged, Laloux et al., 2000). The shrinkage and denoised estimators
                stabilize the weights when there are many assets compared to the number of
                periods. Defaults to "sample".
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

        Returns:
            pd.DataFrame: With output="summary", per portfolio the Annualized Return
            (geometric), Annualized Volatility, Sharpe Ratio (the annualized average return
            above the risk free rate over the volatility), Maximum Drawdown, Average
            Turnover per rebalancing and Annualized Costs. With output="returns", the
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

        | Portfolio                |   Annualized Return |   Annualized Volatility |   Sharpe Ratio |   Maximum Drawdown |   Average Turnover |   Annualized Costs |
        |:-------------------------|--------------------:|------------------------:|---------------:|-------------------:|-------------------:|-------------------:|
        | Minimum Variance         |              0.0891 |                  0.1375 |         0.4061 |            -0.1268 |             0.0667 |             0.0008 |
        | Maximum Sharpe           |              0.0828 |                  0.2124 |         0.2971 |            -0.2776 |             0.4378 |             0.0052 |
        | Risk Parity              |              0.1242 |                  0.1469 |         0.6056 |            -0.1439 |             0.0428 |             0.0005 |
        | Hierarchical Risk Parity |              0.1003 |                  0.1382 |         0.4784 |            -0.135  |             0.0474 |             0.0006 |
        | Maximum Diversification  |              0.1225 |                  0.1426 |         0.6085 |            -0.1357 |             0.0617 |             0.0007 |
        | Inverse Volatility       |              0.122  |                  0.151  |         0.5798 |            -0.1484 |             0.0415 |             0.0005 |
        | Equal Weight             |              0.1407 |                  0.1624 |         0.6519 |            -0.1594 |             0.0388 |             0.0005 |
        | Benchmark                |              0.1285 |                  0.1799 |         0.545  |            -0.221  |           nan      |           nan      |
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
                        benchmark if expected_returns == "capm" else None,
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
        summary.index.name = "Portfolio"
        return self._round(summary, rounding)

    @handle_errors
    def get_stress_test(
        self,
        weights: dict[str, float] | None = None,
        scenarios: dict[str, tuple[str, str]] | None = None,
        rounding: int | None = None,
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
