"""Portfolio Construction Module"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd

from financetoolkit.portfolio import attribution_model, optimization_model
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR, apply_rounding

logger = get_logger()

BENCHMARK_TICKER = "Benchmark"
PORTFOLIO_TICKER = "Portfolio"
PERIODS = ["daily", "weekly", "monthly", "quarterly", "yearly"]

# How far weights may sum from one before they are reported as scaled.
WEIGHT_SUM_TOLERANCE = 1e-6

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# ruff: noqa: E501


class PortfolioConstruction:
    """
    The Portfolio Construction module builds and examines portfolios from the assets of a
    Toolkit: optimal weights (minimum variance, maximum Sharpe ratio, mean-variance,
    risk parity, Hierarchical Risk Parity, Black-Litterman, minimum CVaR), the efficient
    frontier, risk contributions, historical stress tests, look-through exposure and a
    Brinson-Fachler attribution against the benchmark.
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
        returns = self._historical_data[period]["Return"]
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

    def _get_inputs(
        self, period: str, include_benchmark: bool
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
        """The returns, annualized expected returns, covariance and risk free rate."""
        returns = self._get_returns(period, include_benchmark)
        expected_returns, covariance = (
            optimization_model.get_expected_returns_and_covariance(
                returns, PERIODS_PER_YEAR[period]
            )
        )
        risk_free = self._risk_free_rate_data.get(period)
        risk_free_rate = (
            float(risk_free.reindex(returns.index).mean())
            if risk_free is not None and not risk_free.empty
            else 0.0
        )
        if np.isnan(risk_free_rate):
            risk_free_rate = 0.0
        return returns, expected_returns, covariance, risk_free_rate

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
        minimum_weight: float = 0.0,
        maximum_weight: float = 1.0,
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
            minimum_weight (float, optional): The lowest weight an asset may have; a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float, optional): The highest weight an asset may have.
                Defaults to 1.
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

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.portfolio.get_minimum_variance_weights()
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.0212 |            0.1933 |       0.2786 |         0.5727 |              0.0212 |
        | MSFT      |   0.2166 |            0.2017 |       0.2572 |         0.6532 |              0.2166 |
        | JNJ       |   0.5821 |            0.0984 |       0.1673 |         0.3867 |              0.5821 |
        | XOM       |   0.1801 |            0.291  |       0.2702 |         0.9522 |              0.1801 |
        | Portfolio |   1      |            0.1575 |       0.135  |         0.9166 |              1      |
        """
        _, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        weights = optimization_model.get_minimum_variance_weights(
            covariance, minimum_weight, maximum_weight
        )
        return self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, rounding
        )

    @handle_errors
    def get_maximum_sharpe_weights(
        self,
        period: str = "daily",
        minimum_weight: float = 0.0,
        maximum_weight: float = 1.0,
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
            minimum_weight (float, optional): The lowest weight an asset may have; a
                negative value allows short positions. Defaults to 0.
            maximum_weight (float, optional): The highest weight an asset may have.
                Defaults to 1.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution. Returns and volatilities are
            annualized.

        Notes:
        - The expected returns are the historical average returns. They are the noisiest
          input of the optimization, so the weights tend to concentrate in the assets that
          happened to do best; a maximum weight or the Black-Litterman returns temper that.

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
        _, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        weights = optimization_model.get_maximum_sharpe_weights(
            expected_returns, covariance, risk_free_rate, minimum_weight, maximum_weight
        )
        return self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, rounding
        )

    @handle_errors
    def get_mean_variance_weights(
        self,
        target_return: float | None = None,
        risk_aversion: float = 2.5,
        period: str = "daily",
        minimum_weight: float = 0.0,
        maximum_weight: float = 1.0,
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
            minimum_weight (float, optional): The lowest weight an asset may have. Defaults
                to 0.
            maximum_weight (float, optional): The highest weight an asset may have.
                Defaults to 1.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution.

        Notes:
        - A target return outside what the assets can reach within the weight bounds
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

        toolkit.portfolio.get_mean_variance_weights(target_return=0.15)
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.0219 |            0.1933 |       0.2786 |         0.5727 |              0.0211 |
        | MSFT      |   0.2058 |            0.2017 |       0.2572 |         0.6532 |              0.1961 |
        | JNJ       |   0.6256 |            0.0984 |       0.1673 |         0.3867 |              0.6549 |
        | XOM       |   0.1467 |            0.291  |       0.2702 |         0.9522 |              0.1279 |
        | Portfolio |   1      |            0.15   |       0.1355 |         0.8583 |              1      |
        """
        _, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        weights = optimization_model.get_mean_variance_weights(
            expected_returns,
            covariance,
            target_return,
            risk_aversion,
            minimum_weight,
            maximum_weight,
        )
        return self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, rounding
        )

    @handle_errors
    def get_efficient_frontier(
        self,
        points: int = 20,
        period: str = "daily",
        minimum_weight: float = 0.0,
        maximum_weight: float = 1.0,
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
            minimum_weight (float, optional): The lowest weight an asset may have. Defaults
                to 0.
            maximum_weight (float, optional): The highest weight an asset may have.
                Defaults to 1.
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
        _, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        frontier = optimization_model.get_efficient_frontier(
            expected_returns,
            covariance,
            risk_free_rate,
            points,
            minimum_weight,
            maximum_weight,
        )
        return self._round(frontier, rounding)

    @handle_errors
    def get_risk_parity_weights(
        self,
        risk_budget: dict[str, float] | None = None,
        period: str = "daily",
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

        Also known as: equal risk contribution, ERC, risk budgeting.

        Args:
            risk_budget (dict[str, float] | None, optional): The share of the risk per
                asset, e.g. {"AAPL": 0.4, "MSFT": 0.2, ...}; it is scaled to sum to one.
                Defaults to None, an equal share for every asset.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
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

        toolkit.portfolio.get_risk_parity_weights()
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.1786 |            0.1933 |       0.2786 |         0.5727 |                0.25 |
        | MSFT      |   0.2129 |            0.2017 |       0.2572 |         0.6532 |                0.25 |
        | JNJ       |   0.3782 |            0.0984 |       0.1673 |         0.3867 |                0.25 |
        | XOM       |   0.2303 |            0.291  |       0.2702 |         0.9522 |                0.25 |
        | Portfolio |   1      |            0.1817 |       0.1452 |         1.0189 |                1    |
        """
        _, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        weights = optimization_model.get_risk_parity_weights(
            covariance,
            pd.Series(risk_budget, dtype=float) if risk_budget is not None else None,
        )
        return self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, rounding
        )

    @handle_errors
    def get_hierarchical_risk_parity_weights(
        self,
        linkage_method: str = "single",
        period: str = "daily",
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

        toolkit.portfolio.get_hierarchical_risk_parity_weights()
        ```

        Which returns:

        |           |   Weight |   Expected Return |   Volatility |   Sharpe Ratio |   Risk Contribution |
        |:----------|---------:|------------------:|-------------:|---------------:|--------------------:|
        | AAPL      |   0.1287 |            0.1933 |       0.2786 |         0.5727 |              0.1599 |
        | MSFT      |   0.1511 |            0.2017 |       0.2572 |         0.6532 |              0.1498 |
        | JNJ       |   0.5206 |            0.0984 |       0.1673 |         0.3867 |              0.4742 |
        | XOM       |   0.1996 |            0.291  |       0.2702 |         0.9522 |              0.2161 |
        | Portfolio |   1      |            0.1647 |       0.1374 |         0.9534 |              1      |
        """
        _, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        weights = optimization_model.get_hierarchical_risk_parity_weights(
            covariance, linkage_method
        )
        return self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, rounding
        )

    @handle_errors
    def get_inverse_volatility_weights(
        self,
        period: str = "daily",
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
        _, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        weights = optimization_model.get_inverse_volatility_weights(covariance)
        return self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, rounding
        )

    @handle_errors
    def get_minimum_cvar_weights(
        self,
        confidence_level: float = 0.95,
        period: str = "daily",
        minimum_weight: float = 0.0,
        maximum_weight: float = 1.0,
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
            minimum_weight (float, optional): The lowest weight an asset may have. Defaults
                to 0.
            maximum_weight (float, optional): The highest weight an asset may have.
                Defaults to 1.
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and for the portfolio, the Weight, Expected Return,
            Volatility, Sharpe Ratio and Risk Contribution, with the portfolio's CVaR (per
            period, as a loss) in the CVaR column.

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
        | AAPL      |   0      |            0.1933 |       0.2786 |         0.5727 |              0      | 0.0389 |
        | MSFT      |   0.1823 |            0.2017 |       0.2572 |         0.6532 |              0.1507 | 0.0355 |
        | JNJ       |   0.5966 |            0.0984 |       0.1673 |         0.3867 |              0.6006 | 0.0227 |
        | XOM       |   0.2211 |            0.291  |       0.2702 |         0.9522 |              0.2488 | 0.0369 |
        | Portfolio |   1      |            0.1598 |       0.136  |         0.9273 |              1      | 0.0189 |
        """
        returns, expected_returns, covariance, risk_free_rate = self._get_inputs(
            period, include_benchmark
        )
        weights = optimization_model.get_minimum_cvar_weights(
            returns, confidence_level, minimum_weight, maximum_weight
        )
        table = self._weights_table(
            weights, expected_returns, covariance, risk_free_rate, None
        )

        # The CVaR of every asset alone and of the portfolio, as a positive loss.
        def cvar(series: pd.Series) -> float:
            cutoff = series.quantile(1 - confidence_level)
            return float(-series[series <= cutoff].mean())

        table["CVaR"] = [cvar(returns[asset]) for asset in returns.columns] + [
            cvar(returns @ weights)
        ]
        return self._round(table, rounding)

    @handle_errors
    def get_black_litterman_weights(
        self,
        views: dict[str, float] | None = None,
        view_confidences: dict[str, float] | None = None,
        market_weights: dict[str, float] | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        period: str = "daily",
        minimum_weight: float = 0.0,
        maximum_weight: float = 1.0,
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
            minimum_weight (float, optional): The lowest weight an asset may have. Defaults
                to 0.
            maximum_weight (float, optional): The highest weight an asset may have.
                Defaults to 1.
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
        _, _, covariance, risk_free_rate = self._get_inputs(period, False)
        assets = covariance.index.tolist()

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
            covariance,
            market,
            views=views,
            view_confidences=view_confidences,
            risk_aversion=risk_aversion,
            tau=tau,
            risk_free_rate=risk_free_rate,
        )
        weights = optimization_model.get_mean_variance_weights(
            returns["Black-Litterman Return"],
            covariance,
            risk_aversion=risk_aversion,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
        table = pd.concat(
            [(market / market.sum()).rename("Market Weight"), returns, weights], axis=1
        )
        return self._round(table, rounding)

    @handle_errors
    def get_risk_contributions(
        self,
        weights: dict[str, float] | None = None,
        period: str = "daily",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate how much each asset contributes to the volatility of a portfolio. The
        marginal contribution is the change in portfolio volatility from a small increase in
        an asset's weight, (Σw)_i / σ_p; the risk contribution is the weight times that, and
        the contributions add up to the portfolio's volatility (Euler allocation). A large
        position in a calm asset can contribute less risk than a small one in a volatile
        asset.

        For more information, see the following paper:

        - Litterman, R. (1996). "Hot Spots and Hedges." The Journal of Portfolio Management,
          23(5), 52-75.

        Also known as: risk decomposition, marginal contribution to risk, risk budget.

        Args:
            weights (dict[str, float] | None, optional): The weight per asset. Defaults to
                None, which uses the latest weights of the Portfolio the Toolkit came from,
                or equal weights.
            period (str, optional): The frequency of the returns (daily, weekly, monthly,
                quarterly or yearly). Defaults to "daily".
            include_benchmark (bool, optional): Whether to include the benchmark as an
                asset. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per asset and in total, the Weight, Marginal Contribution, Risk
            Contribution and Percentage Contribution (annualized volatility terms).

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
            weights={"AAPL": 0.4, "MSFT": 0.3, "JNJ": 0.2, "XOM": 0.1}
        )
        ```

        Which returns:

        | Asset   |   Weight |   Marginal Contribution |   Risk Contribution |   Percentage Contribution |
        |:--------|---------:|------------------------:|--------------------:|--------------------------:|
        | AAPL    |      0.4 |                  0.2564 |              0.1025 |                    0.5529 |
        | MSFT    |      0.3 |                  0.2116 |              0.0635 |                    0.3423 |
        | JNJ     |      0.2 |                  0.0541 |              0.0108 |                    0.0584 |
        | XOM     |      0.1 |                  0.0862 |              0.0086 |                    0.0465 |
        | Total   |      1   |                nan      |              0.1855 |                    1      |
        """
        _, _, covariance, _ = self._get_inputs(period, include_benchmark)
        resolved = self._resolve_weights(weights, covariance.index.tolist())
        table = optimization_model.get_risk_contributions(resolved, covariance)
        table.index.name = "Asset"
        table.loc["Total"] = [
            table["Weight"].sum(),
            np.nan,
            table["Risk Contribution"].sum(),
            table["Percentage Contribution"].sum(),
        ]
        return self._round(table, rounding)

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
                profile = self._get_profile()
                row = "Sector" if by == "sector" else "Country"
                for asset in stocks:
                    value = (
                        profile.loc[row, asset]
                        if asset in profile.columns and row in profile.index
                        else None
                    )
                    if by == "country" and isinstance(value, str):
                        # Lazily imported: the country names live with the Economics data.
                        from financetoolkit.economics.helpers import (  # noqa: PLC0415
                            COUNTRY_CODES,
                        )

                        value = COUNTRY_CODES.get(value, value)
                    if isinstance(value, str) and value:
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

        prices = self._historical_data["daily"]["Adj Close"][assets].dropna(how="all")
        start_prices = prices.bfill().iloc[0]
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
