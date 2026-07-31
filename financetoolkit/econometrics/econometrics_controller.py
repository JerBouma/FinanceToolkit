"""Econometrics Module"""

__docformat__ = "google"

import itertools
import warnings

import numpy as np
import pandas as pd

from financetoolkit.econometrics import (
    causal_inference_model,
    causality_model,
    cointegration_model,
    diagnostics_model,
    forecast_evaluation_model,
    hypothesis_testing_model,
    panel_data_model,
    regression_model,
    specification_tests_model,
    time_series_model,
    unitroot_model,
)
from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk import risk_model
from financetoolkit.risk.helpers import determine_within_historical_data
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# Runtime errors are ignored on purpose given the nature of the calculations
# sometimes leading to division by zero or other mathematical errors. This is however
# for financial analysis purposes not an issue and should not be considered as a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals
# pylint: disable=too-many-boolean-expressions

# The synthetic benchmark series added to every Toolkit instance's historical data --
# excluded from default ticker-set derivation unless `include_benchmark=True`.
BENCHMARK_TICKER = "Benchmark"


class Econometrics:
    """
    The Econometrics module is meant to test the statistical properties of a price
    or return series -- stationarity (unit root tests), long-run equilibrium
    relationships between series (cointegration), predictive lead-lag relationships
    (Granger causality), and general model/residual diagnostics (ARCH effects,
    normality, autocorrelation, structural stability). These are the standard
    foundational tests behind approaches such as pairs trading, spread modeling,
    and choosing an appropriate volatility or return model in the first place.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: pd.DataFrame = pd.DataFrame(),
        intraday_period: str | None = None,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Econometrics Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the Toolkit instance.
            historical_data (pd.DataFrame, optional): The historical data containing all periods.
                Defaults to pd.DataFrame().
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_augmented_dickey_fuller(period='yearly')
        ```
        """
        self._historical_data = historical_data
        self._tickers = tickers
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None

        # Within Return Calculations
        daily_historical_data = self._historical_data["daily"].copy().fillna(0)
        intraday_historical_data = self._historical_data["intraday"].copy().fillna(0)

        daily_historical_data.index = pd.DatetimeIndex(
            daily_historical_data.to_timestamp().index
        )

        if not self._historical_data["intraday"].empty:
            intraday_historical_data.index = pd.DatetimeIndex(
                intraday_historical_data.to_timestamp().index
            )

        self._within_historical_data = determine_within_historical_data(
            daily_historical_data=daily_historical_data,
            intraday_historical_data=intraday_historical_data,
            intraday_period=intraday_period,
        )

    @handle_portfolio
    @handle_errors
    def get_arch_lm_test(
        self,
        period: str | None = None,
        within_period: bool = True,
        lags: int = 5,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate Engle's Lagrange Multiplier (LM) test for ARCH effects.

        The test regresses squared, mean-demeaned returns on `lags` of themselves and
        tests whether the resulting R-squared is significantly different from zero.
        A significant result (low p-value) indicates that the return series exhibits
        volatility clustering, and a GARCH-family model is an appropriate choice for
        it. A high p-value suggests fitting GARCH would not be meaningful, since there
        is no detectable time-varying volatility to model.

        For more information about the method, see the following paper:

        - Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with
        Estimates of the Variance of United Kingdom Inflation." Econometrica, 50(4),
        987-1008.

        Also known as: ARCH-LM test, Engle's ARCH test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "quarterly".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to True.
            lags (int, optional): The number of lags to test for ARCH effects. Defaults to 5.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The ARCH-LM statistic and its p-value per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and runs the ARCH-LM
        test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_arch_lm_test(period="quarterly")
        ```

        Which returns:

        |                   |     AMZN |   TSLA |
        |:------------------|---------:|-------:|
        | ARCH-LM Statistic |   4.0116 | 3.7793 |
        | P-Value           |   0.548  | 0.5817 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._filter_benchmark(
            (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            ).dropna(),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_arch_lm_test(returns, lags=lags)

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_jarque_bera_test(
        self,
        period: str | None = None,
        within_period: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Jarque-Bera test for normality.

        The test combines sample skewness and excess kurtosis into a single statistic
        that is chi-squared distributed with 2 degrees of freedom under the null
        hypothesis that returns are normally distributed. A significant result (low
        p-value) indicates that returns are not normally distributed, which is
        relevant when choosing between e.g. gaussian and Student-T based Value at Risk
        models.

        For more information about the method, see the following paper:

        - Jarque, C.M. and Bera, A.K. (1987). "A Test for Normality of Observations
        and Regression Residuals." International Statistical Review, 55(2), 163-172.

        Also known as: JB test, normality test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "quarterly".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to True.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The Jarque-Bera statistic and its p-value per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and runs the
        Jarque-Bera test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_jarque_bera_test(period="quarterly")
        ```

        Which returns:

        |                       |    AMZN |    TSLA |
        |:----------------------|--------:|--------:|
        | Jarque-Bera Statistic |  3.0505 |  1.9354 |
        | P-Value               |  0.2175 |  0.38   |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._filter_benchmark(
            (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            ).dropna(),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_jarque_bera_test(returns)

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_ljung_box_test(
        self,
        period: str | None = None,
        within_period: bool = True,
        lags: int = 10,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Ljung-Box test for autocorrelation.

        The test aggregates the squared Autocorrelation Function up to lag `h` into a single
        statistic that is chi-squared distributed with `h` degrees of freedom under the null
        hypothesis that the series exhibits no autocorrelation up to that lag. A significant
        result (low p-value) indicates that the series is autocorrelated, which is relevant
        both as a standalone diagnostic (e.g. to check whether a return series follows a random
        walk) and as a residual diagnostic after fitting a model (e.g. checking that GARCH
        residuals are no longer autocorrelated).

        For more information about the method, see the following paper:

        - Ljung, G.M., & Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time Series
        Models." Biometrika, 65(2), 297-303.

        Also known as: Ljung-Box Q test, portmanteau test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "quarterly".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to True.
            lags (int, optional): The number of lags to test for autocorrelation up to. Defaults to 10.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The Ljung-Box statistic and its p-value per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and runs the
        Ljung-Box test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_ljung_box_test(period="quarterly", within_period=False)
        ```

        Which returns:

        |                     |   AMZN |   TSLA |
        |:--------------------|-------:|-------:|
        | Ljung-Box Statistic | 8.7703 | 7.1814 |
        | P-Value             | 0.554  | 0.7082 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._filter_benchmark(
            (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            ).dropna(),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_ljung_box_test(returns, lags=lags)

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_variance_ratio_test(
        self,
        period: str | None = None,
        within_period: bool = True,
        q: int = 2,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Lo-MacKinlay Variance Ratio test for the random walk hypothesis.

        The test compares the Variance of `q`-period (overlapping) compounded returns to `q`
        times the Variance of single-period returns. Under the random walk hypothesis these
        should be equal, giving a Variance Ratio of 1. A Variance Ratio above 1 with a
        significant (low) p-value indicates positive autocorrelation (momentum/trending
        behavior), while a Variance Ratio below 1 with a significant p-value indicates negative
        autocorrelation (mean-reversion).

        For more information about the method, see the following paper:

        - Lo, A.W., & MacKinlay, A.C. (1988). "Stock Market Prices Do Not Follow Random
        Walks: Evidence from a Simple Specification Test." Review of Financial Studies,
        1(1), 41-66.

        Also known as: Lo-MacKinlay test, VR test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "quarterly".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to True.
            q (int, optional): The number of periods to compound returns over. Defaults to 2.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The Variance Ratio, the (homoskedastic) test statistic and its p-value
            per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and runs the
        Variance Ratio test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_variance_ratio_test(period="quarterly", within_period=False)
        ```

        Which returns:

        |                          |    AMZN |    TSLA |
        |:-------------------------|--------:|--------:|
        | Variance Ratio           |  0.8526 |  0.7523 |
        | Variance Ratio Statistic | -0.6757 | -1.1353 |
        | P-Value                  |  0.4993 |  0.2563 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._filter_benchmark(
            (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            ).dropna(),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_variance_ratio_test(returns, q=q)

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_cusum_test(
        self,
        period: str | None = None,
        within_period: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the CUSUM test for the stability of the mean of returns over time.

        The test builds a standardized cumulative sum of recursive (expanding-window)
        residuals, which behaves like a Brownian motion under the null hypothesis of a
        stable mean. A stable mean keeps the path close to zero, while a mean shift
        partway through the series (a structural break, e.g. a regime change) drags
        the path away from zero and keeps it there, since later residuals are
        computed against a recursive mean contaminated by the pre-break observations.

        For more information about the method, see the following paper:

        - Brown, R.L., Durbin, J., & Evans, J.M. (1975). "Techniques for Testing the
        Constancy of Regression Relationships over Time." Journal of the Royal
        Statistical Society, Series B, 37(2), 149-192.

        Also known as: CUSUM test, CUSUM of recursive residuals test, Brown-Durbin-Evans
        test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "quarterly".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to True.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The CUSUM statistic, its p-value, the number of observations used, the
            1%/5%/10% critical boundary values, and whether stability is rejected at the 5% level,
            per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and runs the
        CUSUM test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_cusum_test(period="quarterly", within_period=False)
        ```

        Which returns:

        |                        |    AAPL |    MSFT |
        |:-----------------------|--------:|--------:|
        | CUSUM Statistic        |  2.4844 |  2.2885 |
        | Observations           | 12      | 12      |
        | Critical Value 1%      |  6.5991 |  9.2388 |
        | Critical Value 5%      |  5.4733 |  7.6626 |
        | Critical Value 10%     |  4.9075 |  6.8705 |
        | Reject Stability (5%)  |  0      |  0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        returns = self._filter_benchmark(
            (
                self._within_historical_data[period]["Return"]
                if within_period
                else self._historical_data[period]["Return"]
            ).dropna(),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_cusum_test(returns)

        return result.round(rounding if rounding is not None else self._rounding)

    def _get_price_column(self, period: str, column: str) -> pd.DataFrame:
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        return self._historical_data[period][column].dropna()

    def _get_tickers(self, include_benchmark: bool = False) -> list[str]:
        """
        `self._tickers`, optionally with `"Benchmark"` filtered out -- shared by
        every method below that derives its own default ticker set (rather than
        having every relevant ticker passed in explicitly), so `include_benchmark`
        behaves identically everywhere it appears.
        """
        tickers = (
            [self._tickers] if isinstance(self._tickers, str) else list(self._tickers)
        )

        if not include_benchmark:
            tickers = [ticker for ticker in tickers if ticker != BENCHMARK_TICKER]

        return tickers

    def _filter_benchmark(
        self, data: pd.DataFrame, include_benchmark: bool
    ) -> pd.DataFrame:
        """
        Drops the `"Benchmark"` column from `data` unless `include_benchmark` is
        True -- shared by the per-asset diagnostic methods below, which loop over
        every column of the historical data directly rather than over
        `self._tickers`.
        """
        if include_benchmark or BENCHMARK_TICKER not in data.columns:
            return data

        return data.drop(columns=[BENCHMARK_TICKER])

    def _resolve_dependent_independent(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
    ) -> tuple[str, list[str]]:
        """
        Resolves a dependent ticker and one-or-more independent ticker(s) from
        `self._tickers` for every simple "one dependent, one-or-more
        independent(s)" regression-style method on this controller. Defaults to
        `self._tickers[0]` as the dependent ticker and every other ticker in
        `self._tickers` as independent, unless overridden.
        """
        tickers = self._get_tickers(include_benchmark=include_benchmark)

        dependent_ticker = (
            dependent_ticker if dependent_ticker is not None else tickers[0]
        )

        if independent_tickers is None:
            independent_tickers = [
                ticker for ticker in tickers if ticker != dependent_ticker
            ]
        else:
            independent_tickers = (
                [independent_tickers]
                if isinstance(independent_tickers, str)
                else list(independent_tickers)
            )

        if not independent_tickers:
            raise ValueError(
                "No independent ticker(s) resolved -- the Toolkit instance must "
                "contain at least 2 tickers, or independent_tickers must be given "
                "explicitly."
            )

        logger.info(
            "Dependent ticker: %s | Independent ticker(s): %s",
            dependent_ticker,
            independent_tickers,
        )

        return dependent_ticker, independent_tickers

    def _all_ordered_pairs(
        self, include_benchmark: bool = False
    ) -> list[tuple[str, str]]:
        """
        Every ordered pair `(ticker_a, ticker_b)`, `ticker_a != ticker_b`, of
        `self._tickers` -- used by the directional pairwise methods below (Granger
        causality, Engle-Granger cointegration), where swapping the pair changes
        which ticker is treated as dependent.
        """
        return list(
            itertools.permutations(
                self._get_tickers(include_benchmark=include_benchmark), 2
            )
        )

    def _all_unordered_pairs(
        self, include_benchmark: bool = False
    ) -> list[tuple[str, str]]:
        """
        Every unordered pair `(ticker_a, ticker_b)` of `self._tickers` -- used by
        the symmetric pairwise methods below (two-sample t-test, RMSE, MAE), where
        swapping the pair does not change the test.
        """
        return list(
            itertools.combinations(
                self._get_tickers(include_benchmark=include_benchmark), 2
            )
        )

    @handle_portfolio
    @handle_errors
    def get_augmented_dickey_fuller(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        regression: str = "c",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Augmented Dickey-Fuller (ADF) test for a unit root, per asset.

        The test regresses the first difference of the series on its own lagged level and
        `p` lags of its own first difference. The null hypothesis is that the series has a
        unit root (is a random walk, not mean-reverting); the alternative is that it is
        stationary. This is a standard first step before modeling a price series or spread
        with mean-reverting methods, since those methods assume stationarity.

        For more information about the method, see the following paper:

        - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
        Autoregressive Time Series with a Unit Root." Journal of the American Statistical
        Association, 74(366a), 427-431.

        Also known as: ADF test, unit root test, stationarity test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider. Defaults to
            the Schwert (1989) rule of thumb.
            regression (str, optional): Which deterministic terms to include, one of "n" (none), "c"
            (constant) or "ct" (constant and trend). Defaults to "c".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The ADF statistic, its p-value, lags used, observations used, 1%/5%/10%
            critical values, and whether the unit root is rejected at the 5% level, per asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the ADF
        test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_augmented_dickey_fuller(period="quarterly")
        ```

        Which returns:

        |                       |     AMZN |     TSLA |
        |:----------------------|---------:|---------:|
        | ADF Statistic         |   0.2892 |   0.1023 |
        | Lags Used             |  11      |  11      |
        | Observations          |  47      |  47      |
        | Critical Value 1%     |  -3.43   |  -3.43   |
        | Critical Value 5%     |  -2.86   |  -2.86   |
        | Critical Value 10%    |  -2.57   |  -2.57   |
        | Reject Unit Root (5%) |   0      |   0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._filter_benchmark(
            self._get_price_column(period, column), include_benchmark=include_benchmark
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_augmented_dickey_fuller(
                    prices[ticker], max_lag=max_lag, regression=regression
                )
                for ticker in prices.columns
            }
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_kpss_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        regression: str = "c",
        lags: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity, per asset.

        KPSS is the natural complement to the Augmented Dickey-Fuller test: where the ADF null
        hypothesis is that the series HAS a unit root, the KPSS null hypothesis is that the
        series IS stationary, with a unit root as the alternative. Running both together is
        standard practice to triangulate a confident conclusion -- ADF rejecting a unit root
        and KPSS failing to reject stationarity together give a confident stationarity
        conclusion, while the two tests disagreeing flags an ambiguous case.

        For more information about the method, see the following paper:

        - Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). "Testing the Null
        Hypothesis of Stationarity against the Alternative of a Unit Root." Journal of
        Econometrics, 54(1-3), 159-178.

        Also known as: KPSS test, stationarity test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            regression (str, optional): Which deterministic term to remove before testing, one of "c"
            (constant, level-stationarity) or "ct" (constant and trend, trend-stationarity).
            Defaults to "c".
            lags (int, optional): The truncation lag for the Newey-West long-run variance estimate.
            Defaults to the Schwert (1989) rule of thumb.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The KPSS statistic, its p-value, truncation lag used, observations used,
            1%/2.5%/5%/10% critical values, and whether stationarity is rejected at the 5% level,
            per asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        KPSS test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_kpss_test(period="quarterly")
        ```

        Which returns:

        |                           |    AMZN |    TSLA |
        |:--------------------------|--------:|--------:|
        | KPSS Statistic            |  0.5683 |  0.542  |
        | Lags Used                 |  9      |  9      |
        | Observations              | 47      | 47      |
        | Critical Value 1%         |  0.739  |  0.739  |
        | Critical Value 2.5%       |  0.574  |  0.574  |
        | Critical Value 5%         |  0.463  |  0.463  |
        | Critical Value 10%        |  0.347  |  0.347  |
        | Reject Stationarity (5%)  |  1      |  1      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._filter_benchmark(
            self._get_price_column(period, column), include_benchmark=include_benchmark
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_kpss_test(
                    prices[ticker], regression=regression, lags=lags
                )
                for ticker in prices.columns
            }
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_phillips_perron_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        regression: str = "c",
        lags: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Phillips-Perron (PP) test for a unit root, per asset.

        Phillips-Perron tests the same null hypothesis as the Augmented Dickey-Fuller test (a
        unit root), but corrects for heteroskedasticity and serial correlation in the errors
        nonparametrically via a Newey-West long-run variance estimate, rather than by adding
        lagged-difference terms to the regression as ADF does. PP and ADF should broadly agree
        on the same series since they test the same null with different correction methods.

        For more information about the method, see the following paper:

        - Phillips, P.C.B., & Perron, P. (1988). "Testing for a Unit Root in Time Series
        Regression." Biometrika, 75(2), 335-346.

        Also known as: PP test, Z_t test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            regression (str, optional): Which deterministic term to include, one of "c" (constant) or
            "ct" (constant and trend). Defaults to "c". Note "n" (no constant) is not supported, see
            `econometrics.unitroot_model.get_phillips_perron_test` for why.
            lags (int, optional): The truncation lag for the Newey-West long-run variance estimate.
            Defaults to the Schwert (1989) rule of thumb.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Phillips-Perron Z_t statistic, truncation lag used, observations used,
            1%/5%/10% critical values, and whether the unit root is rejected at the 5% level, per asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        Phillips-Perron test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_phillips_perron_test(period="quarterly")
        ```

        Which returns:

        |                           |     AMZN |     TSLA |
        |:--------------------------|---------:|---------:|
        | Phillips-Perron Statistic |  -0.6688 |  -1.2623 |
        | Lags Used                 |   9      |   9      |
        | Observations              |  46      |  46      |
        | Critical Value 1%         |  -3.43   |  -3.43   |
        | Critical Value 5%        |  -2.86   |  -2.86   |
        | Critical Value 10%       |  -2.57   |  -2.57   |
        | Reject Unit Root (5%)     |   0      |   0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._filter_benchmark(
            self._get_price_column(period, column), include_benchmark=include_benchmark
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_phillips_perron_test(
                    prices[ticker], regression=regression, lags=lags
                )
                for ticker in prices.columns
            }
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_zivot_andrews_test(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        regression: str = "c",
        trim: float = 0.15,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Zivot-Andrews test for a unit root, per asset, allowing for a
        single structural break at an unknown (endogenously estimated) date.

        The (A)DF/KPSS/PP tests above all assume the deterministic component of the
        series (its constant and/or trend) is stable throughout the sample. If a
        series instead has a single one-time break -- e.g. a permanent level shift or
        a change in trend slope, such as a stock split, spin-off, or a structural
        shift in the business -- the ordinary ADF test is biased towards not
        rejecting the unit root even for a genuinely (trend-)stationary series with a
        break. The Zivot-Andrews test corrects for this by adding a break dummy to
        the ADF regression and choosing, for each asset, the break date that is most
        favorable to the stationary alternative -- which is why it needs its own
        (more negative) critical values rather than the ordinary ADF ones.

        For more information about the method, see the following papers:

        - Zivot, E., & Andrews, D.W.K. (1992). "Further Evidence on the Great Crash,
        the Oil-Price Shock, and the Unit-Root Hypothesis." Journal of Business &
        Economic Statistics, 10(3), 251-270.
        - Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
        Hypothesis." Econometrica, 57(6), 1361-1401.

        Also known as: ZA test, structural break unit root test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider when
            selecting the (single, reused) lag length. Defaults to the Schwert (1989) rule of thumb.
            regression (str, optional): Which break to allow for, one of "c" (a break in the
            level/intercept), "t" (a break in the trend slope) or "ct" (both). Defaults to "c".
            trim (float, optional): The fraction of observations excluded from the candidate break
            date search at the start and end of the sample. Must be in [0, 1/3). Defaults to 0.15.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Zivot-Andrews statistic, its p-value, the (0-indexed) position of the
            selected break date within the used sample, the number of lags and observations used,
            the 1%/5%/10% critical values, and whether the unit root is rejected at the 5% level, per
            asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        Zivot-Andrews test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_zivot_andrews_test(period="weekly")
        ```

        Which returns:

        |                         |     AAPL |     MSFT |
        |:------------------------|---------:|---------:|
        | Zivot-Andrews Statistic |  -2.0684 |  -2.4191 |
        | Break Index             | 117      | 117      |
        | Lags Used               |  14      |  14      |
        | Observations            | 142      | 142      |
        | Critical Value 1%       |  -5.2764 |  -5.2764 |
        | Critical Value 5%       |  -4.8107 |  -4.8107 |
        | Critical Value 10%      |  -4.5662 |  -4.5662 |
        | Reject Unit Root (5%)   |   0      |   0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._filter_benchmark(
            self._get_price_column(period, column), include_benchmark=include_benchmark
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_zivot_andrews_test(
                    prices[ticker], max_lag=max_lag, regression=regression, trim=trim
                )
                for ticker in prices.columns
            }
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_engle_granger_cointegration(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Engle-Granger test for cointegration between every ordered
        pair of tickers in the Toolkit instance.

        Two individually non-stationary series (e.g. two stock prices, each following a
        random walk) are cointegrated if some linear combination of them is stationary,
        i.e. they share a long-run equilibrium relationship even though each wanders on
        its own in the short run. This is the classic statistical foundation for
        pairs-trading: if two assets are cointegrated, deviations of the spread from its
        equilibrium level tend to revert, making the spread itself tradeable.

        For more information about the method, see the following paper:

        - Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
        Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

        Also known as: EG test, residual-based cointegration test, pairs-trading test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider in the
            underlying ADF test on the residuals. Defaults to `statsmodels`' automatic selection.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: One row per ordered `(Dependent, Independent)` ticker pair, with the
            Engle-Granger statistic, its p-value, 1%/5%/10% critical values, and whether
            cointegration is found at the 5% level -- the test is not symmetric (normalizing on
            the dependent ticker matters), so both orderings of every pair are included.

        Notes:
        - The method retrieves historical price data based on the specified `period` for every
        ticker in the Toolkit instance and runs the Engle-Granger test on every ordered pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_engle_granger_cointegration(period="quarterly")
        ```

        Which returns (the 1%/10% critical value columns follow the same pattern as 5%,
        omitted here for width):

        | Dependent   | Independent   |   EG Statistic |   P-Value |   Crit. 5% | Cointegrated (5%)   |
        |:------------|:--------------|---------------:|----------:|-----------:|:--------------------|
        | AAPL        | MSFT          |        -1.4334 |    0.7858 |    -3.8927 | False               |
        | MSFT        | AAPL          |        -2.8297 |    0.1564 |    -3.8927 | False               |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._get_price_column(period, column)

        pairs = self._all_ordered_pairs(include_benchmark=include_benchmark)
        logger.info("Dependent/Independent ticker pairs: %s", pairs)

        rows = {
            (ticker_a, ticker_b): cointegration_model.get_engle_granger_cointegration(
                prices[ticker_a], prices[ticker_b], max_lag=max_lag
            )
            for ticker_a, ticker_b in pairs
        }

        result = pd.DataFrame(rows).T
        result.index = result.index.set_names(["Dependent", "Independent"])

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_johansen_cointegration(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        det_order: int = 0,
        k_ar_diff: int = 1,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Johansen test for cointegration among every ticker in the
        Toolkit instance.

        The Engle-Granger test only handles two assets and imposes an arbitrary
        normalization (which asset is "dependent"). Johansen's test generalizes this to
        `N >= 2` assets at once by testing the rank of the long-run coefficient matrix in a
        Vector Error Correction Model (VECM) fit to all assets jointly. The estimated rank
        equals the number of independent cointegrating (long-run equilibrium) relationships
        among the assets: rank 0 means none of them are cointegrated, rank `N` means the
        whole system is already stationary in levels, and a rank in between means that many
        independent stationary combinations exist among the `N` individually non-stationary
        price series.

        For more information about the method, see the following papers:

        - Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." Journal of
        Economic Dynamics and Control, 12(2-3), 231-254.
        - Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in
        Gaussian Vector Autoregressive Models." Econometrica, 59(6), 1551-1580.

        Also known as: Johansen test, Johansen procedure, VECM rank test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            det_order (int, optional): Which deterministic term to include: -1 (none), 0 (a
            constant, restricted to lie in the cointegrating relation) or 1 (a constant plus a
            linear trend, both restricted to lie in the cointegrating relation). Defaults to 0.
            k_ar_diff (int, optional): The number of lagged first differences to include as
            short-run dynamics. Defaults to 1.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers tested jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: One row per candidate cointegration rank `r = 0, ..., N-1`, with the
            corresponding eigenvalue, trace statistic, max-eigenvalue statistic, their 90%/95%/99%
            critical values, and whether each is rejected at the 5% level. The estimated rank is the
            first row (in rank order) that is NOT rejected.

        Notes:
        - The method retrieves historical price data based on the specified `period` for every ticker
        in the Toolkit instance and runs the Johansen test jointly across all of them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_johansen_cointegration(period="quarterly")
        ```

        Which returns (showing the trace-statistic columns; the max-eigenvalue-statistic
        columns follow the same pattern):

        |        |   Eigenvalue |   Trace Statistic |   Trace Critical Value 95% |   Reject (Trace, 5%)   |
        |:-------|-------------:|-------------------:|-----------------------------:|:------------------------|
        | r <= 0 |       0.5653 |             14.1993 |                       15.4943 | False                   |
        | r <= 1 |       0.3674 |              5.0363 |                        3.8415 | True                    |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._get_price_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        result = cointegration_model.get_johansen_cointegration(
            prices[tickers], det_order=det_order, k_ar_diff=k_ar_diff
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_granger_causality(
        self,
        period: str | None = None,
        column: str = "Return",
        max_lag: int = 5,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Granger causality test, for every ordered pair of tickers in
        the Toolkit instance, of whether the second helps predict the first.

        "Granger causality" is a statement about predictive power, not true causation:
        one asset is said to Granger-cause another if past values of the first,
        combined with past values of the second itself, predict the second
        significantly better than past values of the second alone.

        For more information about the method, see the following paper:

        - Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
        and Cross-Spectral Methods." Econometrica, 37(3), 424-438.

        Also known as: Granger causality test, predictive causality, lead-lag test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Return", since
            Granger causality assumes a stationary series (unlike the ADF/Engle-Granger tests, which
            operate on price levels on purpose).
            max_lag (int, optional): The number of lags of both assets to include in the regressions.
            Defaults to 5.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: One row per ordered `(Dependent, Independent)` ticker pair, with the
            F-statistic, its p-value, and whether the independent ticker is found to
            Granger-cause the dependent ticker at the 5% level -- Granger causality is inherently
            directional, so both orderings of every pair are included.

        Notes:
        - The method retrieves historical return data based on the specified `period` for every
        ticker in the Toolkit instance and runs the Granger causality test on every ordered pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_granger_causality(period="weekly", max_lag=3)
        ```

        Which returns:

        | Dependent   | Independent   |   F-Statistic |   P-Value | Granger-Causes (5%)   |
        |:------------|:--------------|--------------:|----------:|:----------------------|
        | AAPL        | MSFT          |        2.4852 |    0.0630 | False                 |
        | MSFT        | AAPL          |        0.3750 |    0.7712 | False                 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        pairs = self._all_ordered_pairs(include_benchmark=include_benchmark)
        logger.info("Dependent/Independent ticker pairs: %s", pairs)

        rows = {
            (ticker_a, ticker_b): causality_model.get_granger_causality(
                returns[ticker_a], returns[ticker_b], max_lag=max_lag
            )
            for ticker_a, ticker_b in pairs
        }

        result = pd.DataFrame(rows).T
        result.index = result.index.set_names(["Dependent", "Independent"])

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_portfolio
    @handle_errors
    def get_diebold_mariano_test(
        self,
        period: str | None = None,
        method_a: str = "ewma",
        method_b: str = "rolling",
        window_size: int = 22,
        lambda_: float = 0.94,
        loss: str = "squared",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Compare the forecast accuracy of two volatility forecasting methods against
        realized (squared return) Variance, per asset, via the Diebold-Mariano (1995)
        test.

        Rather than comparing two different assets, this compares two different ways
        of forecasting the *same* asset's next-day Variance -- e.g. the simpler
        exponentially weighted (EWMA, see `Risk.get_ewma_volatility`) approach against a
        plain rolling-window Standard Deviation -- to determine whether one is
        significantly more accurate than the other for a given asset.

        For more information about the method, see the following paper:

        - Diebold, F.X., & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
        Journal of Business & Economic Statistics, 13(3), 253-263.

        Also known as: DM test, forecast comparison test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "daily", since `window_size` is expressed in return observations of this
            frequency.
            method_a (str, optional): The first volatility forecasting method, one of "ewma" or "rolling".
            Defaults to "ewma".
            method_b (str, optional): The second (competing) volatility forecasting method, one of "ewma"
            or "rolling". Defaults to "rolling".
            window_size (int, optional): The rolling window size used by the "rolling" method. Defaults to
            22 (approximately one trading month).
            lambda_ (float, optional): The decay factor used by the "ewma" method. Defaults to 0.94.
            loss (str, optional): The loss function to compare forecast errors with, one of "squared" or
            "absolute". Defaults to "squared".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Diebold-Mariano statistic, its p-value, the mean loss differential (negative
            favors `method_a`) and the number of observations used, per asset.

        Notes:
        - Both forecasts are lagged by one period (i.e. use only information available up to, and
        including, the prior period) before being compared to the realized squared return, so this is a
        genuinely out-of-sample forecast comparison rather than an in-sample fit comparison.
        - Like `Risk.get_var_backtest`, this is calculated over the full return history rather than the
        `within_period` slices used elsewhere in this module, since the rolling/EWMA forecasts need more
        history than a single sub-period provides.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_diebold_mariano_test(method_a="ewma", method_b="rolling")
        ```

        Which returns:

        |                           |     AAPL |     MSFT |
        |:--------------------------|---------:|---------:|
        | Diebold-Mariano Statistic |  -2.4324 |  -2.8338 |
        | P-Value                   |   0.0152 |   0.0047 |
        | Mean Loss Differential    |  -0.0000 |  -0.0000 |
        | Observations              | 735      | 735      |
        """
        if method_a not in ("ewma", "rolling") or method_b not in ("ewma", "rolling"):
            raise ValueError("method_a and method_b must be 'ewma' or 'rolling'.")

        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._filter_benchmark(
            self._historical_data[period]["Return"].dropna(),
            include_benchmark=include_benchmark,
        )

        def _forecast(method: str) -> pd.DataFrame:
            if method == "ewma":
                volatility = risk_model.get_ewma_volatility(returns, lambda_)
            else:
                volatility = returns.rolling(window=window_size).std()

            return (volatility**2).shift(1)

        actual = returns**2
        forecast_a = _forecast(method_a)
        forecast_b = _forecast(method_b)

        result = forecast_evaluation_model.get_diebold_mariano_test(
            actual, forecast_a, forecast_b, loss=loss, horizon=1
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_ols(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        cov_type: str = "nonrobust",
        clusters: pd.Series | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit an Ordinary Least Squares (OLS) regression of `dependent_ticker` on
        `independent_tickers`.

        Also known as: linear regression, least squares regression.

        For more information about the method, see `regression_model.get_ols`.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            cov_type (str, optional): Which covariance estimator to use for the standard errors --
            one of "nonrobust" (classical, assumes homoskedastic errors), "HC0"/"HC1"/"HC2"/"HC3"
            (heteroskedasticity-robust) or "cluster" (cluster-robust, requires `clusters`). Use
            `get_breusch_pagan_test`/`get_white_test` to check for heteroskedasticity first.
            Defaults to "nonrobust".
            clusters (pd.Series | None, optional): The cluster label for each observation
            (e.g. a coarser time bucket derived from the return index, to correct for
            within-period correlation), required when `cov_type="cluster"`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AMZN (the first ticker) is dependent; TSLA is independent
        toolkit.econometrics.get_ols(period="quarterly")

        # Or, override which ticker is dependent/independent, and pull in Benchmark too
        toolkit.econometrics.get_ols(
            dependent_ticker="AMZN",
            independent_tickers=["TSLA", "Benchmark"],
            period="quarterly",
            cov_type="HC1",
        )
        # Or, simply opt every default independent ticker (here, just TSLA) into
        # including Benchmark too:
        toolkit.econometrics.get_ols(period="quarterly", include_benchmark=True)
        ```
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
            cov_type=cov_type,
            clusters=clusters,
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_wls(
        self,
        weights: pd.Series,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        cov_type: str = "nonrobust",
        clusters: pd.Series | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Weighted Least Squares (WLS) regression of `dependent_ticker` on
        `independent_tickers`.

        Also known as: weighted regression.

        For more information about the method, see `regression_model.get_wls`.

        Args:
            weights (pd.Series): The (positive) weight of each observation, aligned to
            the same period index as the return data (e.g. `1 / rolling_variance`).
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            cov_type (str, optional): Which covariance estimator to use, applied to the weighted/
            transformed problem -- see `get_ols`'s `cov_type` for the full list of options.
            Defaults to "nonrobust".
            clusters (pd.Series | None, optional): The cluster label for each observation,
            required when `cov_type="cluster"`. Defaults to None.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        import pandas as pd
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        returns = toolkit.econometrics._get_price_column("weekly", "Return")
        weights = pd.Series(1.0, index=returns.index)

        # AAPL (the first ticker) is dependent, MSFT is independent
        toolkit.econometrics.get_wls(weights, period="weekly")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0016 |       0.0024 |        0.6712 |    0.5031 |
        | MSFT      |        0.8681 |       0.0596 |       14.5659 |    0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        to_align = [returns[dependent_ticker], returns[independent_tickers], weights]
        if clusters is not None:
            to_align.append(clusters.rename("__clusters__"))

        aligned = pd.concat(to_align, axis=1, join="inner").dropna()
        aligned_clusters = aligned.pop("__clusters__") if clusters is not None else None

        result = regression_model.get_wls(
            aligned.iloc[:, 0],
            aligned[independent_tickers],
            aligned.iloc[:, -1],
            add_constant=add_constant,
            cov_type=cov_type,
            clusters=aligned_clusters,
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_gls(
        self,
        omega: pd.DataFrame,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Generalized Least Squares (GLS) regression of `dependent_ticker` on
        `independent_tickers`, given a known error covariance structure `omega`.

        Also known as: GLS.

        For more information about the method, see `regression_model.get_gls`.

        Args:
            omega (pd.DataFrame): The (symmetric, positive-definite) error covariance
            structure, up to a scalar, shape `(n, n)`.
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        import numpy as np
        import pandas as pd
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        returns = toolkit.econometrics._get_price_column("weekly", "Return")
        n = len(returns["AAPL"].dropna())
        omega = pd.DataFrame(np.eye(n))

        # AAPL (the first ticker) is dependent; MSFT is independent
        toolkit.econometrics.get_gls(omega, period="weekly")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0016 |       0.0024 |        0.6712 |    0.5031 |
        | MSFT      |        0.8681 |       0.0596 |       14.5659 |    0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = regression_model.get_gls(
            returns[dependent_ticker],
            returns[independent_tickers],
            omega.to_numpy(),
            add_constant=add_constant,
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_logistic_regression(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Logistic Regression (Logit model) of whether `dependent_ticker`'s
        return is positive on `independent_tickers`.

        Also known as: logit model, logit regression.

        For more information about the method, see `regression_model.get_logistic_regression`.

        Args:
            dependent_ticker (str | None, optional): The dependent asset (whose
            up/down direction is predicted). Defaults to None, meaning the Toolkit
            instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to derive returns from. Defaults to
            "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, z-Statistic, P-Value),
            indexed by regressor name.

        Notes:
        - The dependent variable is derived as `1` if the dependent ticker's return in the given
        period is positive, `0` otherwise -- this method predicts the *direction*, not the
        magnitude, of the dependent asset's return.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_logistic_regression(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   z-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0909 |       0.2195 |        0.4139 |    0.6789 |
        | MSFT      |       24.8481 |      10.1178 |        2.4559 |    0.0141 |
        | Benchmark |       63.677  |      15.5493 |        4.0952 |    0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        direction = (returns[dependent_ticker] > 0).astype(float)

        result = regression_model.get_logistic_regression(
            direction, returns[independent_tickers], add_constant=add_constant
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_probit_regression(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Probit Regression of whether `dependent_ticker`'s return is positive
        on `independent_tickers`.

        Also known as: probit model.

        For more information about the method, see `regression_model.get_probit_regression`.

        Args:
            dependent_ticker (str | None, optional): The dependent asset (whose
            up/down direction is predicted). Defaults to None, meaning the Toolkit
            instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to derive returns from. Defaults to
            "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, z-Statistic, P-Value),
            indexed by regressor name.

        Notes:
        - The dependent variable is derived as `1` if the dependent ticker's return in the given
        period is positive, `0` otherwise -- this method predicts the *direction*, not the
        magnitude, of the dependent asset's return.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_probit_regression(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   z-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0498 |       0.1266 |        0.3935 |    0.694  |
        | MSFT      |       15.7063 |       5.7385 |        2.737  |    0.0062 |
        | Benchmark |       35.3471 |       8.1355 |        4.3448 |    0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        direction = (returns[dependent_ticker] > 0).astype(float)

        result = regression_model.get_probit_regression(
            direction, returns[independent_tickers], add_constant=add_constant
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_quantile_regression(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        tau: float = 0.5,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        n_bootstrap: int = 0,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Quantile Regression of `dependent_ticker` on `independent_tickers` at
        quantile `tau`.

        Also known as: QR.

        For more information about the method, see `regression_model.get_quantile_regression`.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            tau (float, optional): The quantile to fit, in (0, 1). Defaults to 0.5 (the median).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            n_bootstrap (int, optional): The number of bootstrap resamples used for coefficient
            standard errors, overriding `statsmodels`' default analytic (kernel density-based)
            standard errors. Defaults to 0 (use the analytic standard errors).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error), indexed by regressor name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_quantile_regression(
            independent_tickers=["MSFT", "Benchmark"], tau=0.5, period="weekly"
        )
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |
        |:----------|--------------:|-------------:|
        | Intercept |        0.0007 |       0.0021 |
        | MSFT      |        0.3593 |       0.0854 |
        | Benchmark |        0.6885 |       0.1037 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = regression_model.get_quantile_regression(
            returns[dependent_ticker],
            returns[independent_tickers],
            tau=tau,
            add_constant=add_constant,
            n_bootstrap=n_bootstrap,
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    def _fit_ols_result(
        self,
        dependent_ticker: str,
        independent_tickers: str | list[str],
        period: str | None,
        column: str,
        add_constant: bool,
    ) -> regression_model.RegressionResult:
        """
        Shared helper for the hypothesis-testing methods below -- fits an OLS model
        via `regression_model.get_ols` and returns the raw (unrounded)
        `RegressionResult`, rather than the rounded coefficient table `get_ols`
        returns, since the F-test, Wald test and Likelihood Ratio test all operate
        directly on a `RegressionResult`'s residuals/coefficients/covariance matrix.
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        independent_tickers = (
            [independent_tickers]
            if isinstance(independent_tickers, str)
            else independent_tickers
        )

        return regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
        )

    @handle_errors
    def get_two_sample_t_test(
        self,
        period: str | None = None,
        column: str = "Return",
        equal_variance: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate a two-sample t-test for a difference in mean `column` between
        every unordered pair of tickers in the Toolkit instance.

        Also known as: independent samples t-test, Welch's t-test (default), Student's
        t-test (`equal_variance=True`).

        For more information about the method, see `hypothesis_testing_model.get_two_sample_t_test`.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to compare. Defaults to "Return".
            equal_variance (bool, optional): Whether to assume the two samples share a common
            variance (Student's pooled t-test) instead of Welch's (unequal-variance) t-test.
            Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: One row per unordered `(Ticker A, Ticker B)` pair, with the t-statistic,
            degrees of freedom, its p-value, and the two sample means.

        Notes:
        - The two return series are compared as independent samples (not paired by date), which is
        the standard use of a two-sample t-test -- if a date-by-date, paired comparison is needed
        instead, take the difference of the two return series and run a one-sample test on it.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_two_sample_t_test(period="weekly")
        ```

        Which returns:

        | Ticker A   | Ticker B   |   T-Statistic |   Degrees of Freedom |   P-Value |   Mean A |   Mean B |
        |:-----------|:-----------|--------------:|----------------------:|----------:|---------:|---------:|
        | AAPL       | MSFT       |        0.2318 |               306.6549 |    0.8168 |   0.0047 |   0.0036 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        rows = {
            (ticker_a, ticker_b): hypothesis_testing_model.get_two_sample_t_test(
                returns[ticker_a], returns[ticker_b], equal_variance=equal_variance
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        result = pd.DataFrame(rows).T
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_f_test(
        self,
        dependent_ticker: str,
        restricted_independent_tickers: str | list[str],
        unrestricted_independent_tickers: str | list[str],
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate a nested-model F-test for the joint significance of the regressors
        in `unrestricted_independent_tickers` that are not already in
        `restricted_independent_tickers`.

        Also known as: nested F-test, restricted vs. unrestricted F-test, partial F-test.

        Fits both a "restricted" and an "unrestricted" OLS regression of
        `dependent_ticker` internally (via `regression_model.get_ols`) and compares
        them. For more information about the method, see `hypothesis_testing_model.get_f_test`.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            restricted_independent_tickers (str | list[str]): The independent asset(s) in the
            restricted (smaller) model.
            unrestricted_independent_tickers (str | list[str]): The independent asset(s) in the
            unrestricted (larger) model -- must be a superset of `restricted_independent_tickers`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in both models. Defaults
            to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The F-statistic, its numerator/denominator degrees of freedom, its p-value,
            and whether the added regressors are jointly significant at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_f_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        | Metric                     |     Value |
        |:----------------------------|----------:|
        | F-Statistic                 |   43.6897 |
        | Df Numerator                 |    1      |
        | Df Denominator                |  154      |
        | P-Value                      |    0.0000 |
        | Reject Restrictions (5%)      |    1      |
        """
        restricted_result = self._fit_ols_result(
            dependent_ticker,
            restricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )
        unrestricted_result = self._fit_ols_result(
            dependent_ticker,
            unrestricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )

        result = hypothesis_testing_model.get_f_test(
            restricted_result, unrestricted_result
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_likelihood_ratio_test(
        self,
        dependent_ticker: str,
        restricted_independent_tickers: str | list[str],
        unrestricted_independent_tickers: str | list[str],
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate a nested-model Likelihood Ratio (LR) test, the Maximum Likelihood
        analogue of `get_f_test`, for the joint significance of the regressors in
        `unrestricted_independent_tickers` that are not already in
        `restricted_independent_tickers`.

        Also known as: LR test, Wilks' likelihood ratio test.

        Fits both a "restricted" and an "unrestricted" OLS regression of
        `dependent_ticker` internally (via `regression_model.get_ols`) and compares
        them. For more information about the method, see
        `hypothesis_testing_model.get_likelihood_ratio_test`.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            restricted_independent_tickers (str | list[str]): The independent asset(s) in the
            restricted (smaller) model.
            unrestricted_independent_tickers (str | list[str]): The independent asset(s) in the
            unrestricted (larger) model -- must be a superset of `restricted_independent_tickers`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in both models. Defaults
            to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The LR statistic, its degrees of freedom, its p-value, and whether the added
            regressors are jointly significant at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_likelihood_ratio_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        | Metric                     |    Value |
        |:----------------------------|---------:|
        | LR Statistic                 |  39.2102 |
        | Degrees of Freedom            |   1      |
        | P-Value                       |   0.0000 |
        | Reject Restrictions (5%)       |   1      |
        """
        restricted_result = self._fit_ols_result(
            dependent_ticker,
            restricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )
        unrestricted_result = self._fit_ols_result(
            dependent_ticker,
            unrestricted_independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )

        result = hypothesis_testing_model.get_likelihood_ratio_test(
            restricted_result, unrestricted_result
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_wald_test(
        self,
        restriction_matrix: pd.DataFrame | np.ndarray,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        restriction_values: pd.Series | np.ndarray | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate a Wald test of `q` general linear restriction(s) on the coefficients
        of an OLS regression of `dependent_ticker` on `independent_tickers`.

        Also known as: Wald chi-squared test.

        Fits a single OLS regression internally (via `regression_model.get_ols`) and
        tests `H0: restriction_matrix @ beta = restriction_values` on its
        coefficients. For more information about the method, see
        `hypothesis_testing_model.get_wald_test`.

        Args:
            restriction_matrix (pd.DataFrame | np.ndarray): The `(q, k)` restriction matrix `R`, one
            row per restriction, one column per coefficient in the same order as `add_constant`
            (if True, "Intercept" first) followed by the independent ticker(s).
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s), in the order the restriction matrix's columns
            (after "Intercept", if `add_constant`) refer to them. Defaults to None,
            meaning every other ticker in the Toolkit instance besides
            `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            restriction_values (pd.Series | np.ndarray | None, optional): The length-`q` vector of
            hypothesized values. Defaults to None, i.e. all restrictions equal zero.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The Wald (chi-squared) statistic and its p-value, the small-sample F-statistic
            and its p-value, the number of restrictions, and whether they are rejected at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # H0: the coefficients on both MSFT and Benchmark are jointly zero.
        toolkit.econometrics.get_wald_test(
            restriction_matrix=[[0, 1, 0], [0, 0, 1]],
            independent_tickers=["MSFT", "Benchmark"],
            period="weekly",
        )
        ```

        Which returns:

        | Metric                      |     Value |
        |:-----------------------------|----------:|
        | Wald Statistic (Chi2)         |  314.2908 |
        | Chi2 P-Value                  |    0.0000 |
        | F-Statistic                   |  157.1454 |
        | F P-Value                     |    0.0000 |
        | Restrictions (q)              |    2      |
        | Reject Restrictions (5%)       |    1      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._fit_ols_result(
            dependent_ticker,
            independent_tickers,
            period=period,
            column=column,
            add_constant=add_constant,
        )

        restriction_matrix_values = (
            restriction_matrix.to_numpy()
            if isinstance(restriction_matrix, pd.DataFrame)
            else np.asarray(restriction_matrix)
        )
        restriction_values_array = (
            restriction_values.to_numpy()
            if isinstance(restriction_values, pd.Series)
            else restriction_values
        )

        wald_result = hypothesis_testing_model.get_wald_test(
            result, restriction_matrix_values, restriction_values_array
        )

        return wald_result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_hausman_wu_test(
        self,
        dependent_ticker: str,
        suspect_ticker: str,
        instrument_tickers: str | list[str],
        other_independent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate a regression-based Hausman-Wu test for the endogeneity of
        `suspect_ticker` in a regression of `dependent_ticker` on `suspect_ticker`
        (and, optionally, `other_independent_tickers`), using `instrument_tickers` as
        instruments for `suspect_ticker`.

        Also known as: Hausman test, Durbin-Wu-Hausman test, regression test for
        endogeneity.

        For more information about the method, see
        `hypothesis_testing_model.get_hausman_wu_test`.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            suspect_ticker (str): The (possibly endogenous) asset being tested.
            instrument_tickers (str | list[str]): One or more instrument asset(s) for
            `suspect_ticker` -- assets correlated with `suspect_ticker` but assumed uncorrelated
            with `dependent_ticker`'s error term.
            other_independent_tickers (str | list[str] | None, optional): Any other (assumed
            exogenous) independent asset(s) to include. Defaults to None.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The coefficient on the first-stage residuals, its t-statistic and p-value, and
            whether `suspect_ticker` is flagged as endogenous at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_hausman_wu_test(
            "AAPL", "MSFT", "Benchmark", period="weekly"
        )
        ```

        Which returns:

        | Metric                  |    Value |
        |:-------------------------|---------:|
        | V-Hat Coefficient         |  -0.7162 |
        | T-Statistic               |  -6.6098 |
        | Degrees of Freedom         | 154      |
        | P-Value                   |   0.0000 |
        | Endogenous (5%)            |   1      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        instrument_tickers = (
            [instrument_tickers]
            if isinstance(instrument_tickers, str)
            else instrument_tickers
        )
        other_independent_tickers = (
            [other_independent_tickers]
            if isinstance(other_independent_tickers, str)
            else other_independent_tickers
        )

        result = hypothesis_testing_model.get_hausman_wu_test(
            returns[dependent_ticker],
            returns[suspect_ticker],
            returns[instrument_tickers],
            (
                returns[other_independent_tickers]
                if other_independent_tickers is not None
                else None
            ),
        )

        return result.round(rounding if rounding is not None else self._rounding)

    def _get_ols_result(
        self,
        dependent_ticker: str,
        independent_tickers: str | list[str],
        period: str | None,
        column: str,
        add_constant: bool,
    ) -> regression_model.RegressionResult:
        """
        Shared helper that fits the OLS regression of `dependent_ticker` on
        `independent_tickers` used internally by every post-estimation diagnostic
        below (`get_breusch_pagan_test`, `get_white_test`, `get_durbin_watson_test`,
        `get_ramsey_reset_test`, `get_chow_test`) -- each of these tests operates on
        an already-fitted `RegressionResult` (see `specification_tests_model`), so
        the controller's job is simply to assemble that same fit `get_ols` itself
        produces before handing it off to the requested diagnostic.
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        independent_tickers = (
            [independent_tickers]
            if isinstance(independent_tickers, str)
            else independent_tickers
        )

        return regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
        )

    @handle_errors
    def get_breusch_pagan_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Breusch-Pagan test for heteroskedasticity of a regression of
        `dependent_ticker` on `independent_tickers`.

        Also known as: BP test, Breusch-Pagan-Godfrey test.

        For more information about the method, see
        `specification_tests_model.get_breusch_pagan_test`.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The Breusch-Pagan LM statistic, its p-value, and whether homoskedasticity is
            rejected at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_breusch_pagan_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        | Metric                        |   Value |
        |:-------------------------------|--------:|
        | Breusch-Pagan Statistic       |  1.5544 |
        | P-Value                       |  0.4597 |
        | Reject Homoskedasticity (5%)  |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_breusch_pagan_test(result)

        return test_result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_white_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate White's test for heteroskedasticity of a regression of
        `dependent_ticker` on `independent_tickers`.

        Also known as: White's general test.

        For more information about the method, see `specification_tests_model.get_white_test`.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: White's LM statistic, its p-value, and whether homoskedasticity is rejected at
            the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_white_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        | Metric                        |   Value |
        |:-------------------------------|--------:|
        | White Statistic                |  2.2886 |
        | P-Value                        |  0.8079 |
        | Reject Homoskedasticity (5%)   |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_white_test(result)

        return test_result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_durbin_watson_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Durbin-Watson statistic for first-order autocorrelation in the
        residuals of a regression of `dependent_ticker` on `independent_tickers`.

        Also known as: DW statistic.

        For more information about the method, see
        `specification_tests_model.get_durbin_watson_test`.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The Durbin-Watson statistic and an approximate interpretation flag.

        Notes:
        - Unlike the other tests in this module, this does not carry a formal p-value/reject flag at a
        stated significance level -- see `specification_tests_model.get_durbin_watson_test` for why.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_durbin_watson_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        | Metric                    | Value               |
        |:---------------------------|:---------------------|
        | Durbin-Watson Statistic    | 2.0538               |
        | Interpretation             | No Strong Evidence   |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_durbin_watson_test(result)

        # The "Interpretation" entry is a string, so `.round()` (used elsewhere in this
        # module) is not applicable to the Series as a whole -- round only the statistic.
        rounding_value = rounding if rounding is not None else self._rounding
        if rounding_value is not None:
            test_result["Durbin-Watson Statistic"] = round(
                test_result["Durbin-Watson Statistic"], rounding_value
            )

        return test_result

    @handle_errors
    def get_vif(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Variance Inflation Factor (VIF) of every ticker in the
        Toolkit instance, treated as regressors against one another.

        Also known as: VIF.

        For more information about the method, see `specification_tests_model.get_vif`.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to use. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            regressors tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The VIF of each asset, indexed by ticker.

        Notes:
        - Unlike the other methods in this module, `get_vif` has no `dependent_ticker` -- VIF is a
        property of the regressors alone, independent of any particular dependent variable.
        - `VIF > 10` is the conventional rule-of-thumb threshold for concerning multicollinearity.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_vif(period="weekly")
        ```

        Which returns:

        |      |    VIF |
        |:-----|-------:|
        | AAPL | 2.3688 |
        | MSFT | 2.3688 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        test_result = specification_tests_model.get_vif(returns[tickers])

        return test_result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_ramsey_reset_test(
        self,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        power: int = 3,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate Ramsey's RESET test for functional form misspecification of a
        regression of `dependent_ticker` on `independent_tickers`.

        Also known as: RESET test, Ramsey RESET.

        For more information about the method, see
        `specification_tests_model.get_ramsey_reset_test`.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression. Defaults to True.
            power (int, optional): The highest power of the fitted values to add. Defaults to 3.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The RESET F-statistic, its p-value, and whether correct specification is
            rejected at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_ramsey_reset_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        | Metric                              |   Value |
        |:--------------------------------------|--------:|
        | RESET F-Statistic                    |  1.1591 |
        | P-Value                              |  0.3165 |
        | Reject Correct Specification (5%)    |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        result = self._get_ols_result(
            dependent_ticker, independent_tickers, period, column, add_constant
        )

        test_result = specification_tests_model.get_ramsey_reset_test(
            result, power=power
        )

        return test_result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_chow_test(
        self,
        break_date: str,
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Chow test for a structural break at `break_date` in a
        regression of `dependent_ticker` on `independent_tickers`.

        Also known as: Chow breakpoint test.

        For more information about the method, see `specification_tests_model.get_chow_test`.

        Args:
            break_date (str): The date (e.g. "2021-06-30") at which to split the sample -- all
            periods starting on or after this date form the "after" sub-sample, everything
            before it forms the "before" sub-sample.
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the underlying
            regression(s). Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The Chow F-statistic, its p-value, and whether no structural break is rejected
            at the 5% level.

        Raises:
            ValueError: If `break_date` does not leave enough observations on either side of the
            split to estimate the regression's parameters.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL (the first ticker) is dependent; MSFT and Benchmark are independent
        toolkit.econometrics.get_chow_test(
            break_date="2021-06-30",
            independent_tickers=["MSFT", "Benchmark"],
            period="weekly",
        )
        ```

        Which returns:

        | Metric                              |   Value |
        |:--------------------------------------|--------:|
        | Chow F-Statistic                     |  1.6807 |
        | P-Value                              |  0.1736 |
        | Reject No Structural Break (5%)      |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        result_full = regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
        )

        index = returns.index
        timestamps = (
            index.to_timestamp()
            if hasattr(index, "to_timestamp")
            else pd.DatetimeIndex(index)
        )
        break_index = int(timestamps.searchsorted(pd.Timestamp(break_date)))

        test_result = specification_tests_model.get_chow_test(
            result_full,
            returns[independent_tickers],
            returns[dependent_ticker],
            break_index=break_index,
            add_constant=add_constant,
        )

        return test_result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_iv_2sls(
        self,
        dependent_ticker: str,
        endogenous_ticker: str | list[str],
        instrument_tickers: str | list[str],
        exogenous_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit an Instrumental Variables regression via Two-Stage Least Squares (2SLS) of
        `dependent_ticker` on `endogenous_ticker`, instrumented by `instrument_tickers`.

        Also known as: IV, 2SLS, IV-2SLS.

        For more information about the method, and why plain OLS on an asset suspected of
        reverse-causality/omitted-confounder bias against another asset is unreliable, see
        `causal_inference_model.get_iv_2sls`.

        A natural use case in a multi-asset return panel: suppose `endogenous_ticker`'s
        return is suspected to be simultaneously determined together with
        `dependent_ticker`'s return (e.g. two closely related assets that react to each
        other intraday, or one asset's return partly reflects news about the other) --
        plain OLS of one on the other is then biased. An `instrument_tickers` asset that
        moves `endogenous_ticker` for reasons unrelated to `dependent_ticker`'s own error
        term (e.g. a supplier/peer whose moves affect `endogenous_ticker` but only reach
        `dependent_ticker`, if at all, THROUGH `endogenous_ticker`) allows recovering a
        cleaner estimate of the causal pass-through.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            endogenous_ticker (str | list[str]): The endogenous regressor asset(s) --
            suspected correlated with the error term.
            instrument_tickers (str | list[str]): The excluded instrument asset(s),
            correlated with `endogenous_ticker` but assumed uncorrelated with the error
            term. Must supply at least as many instruments as endogenous regressors.
            exogenous_tickers (str | list[str] | None, optional): Other, non-instrumented
            control asset(s) included as-is in both stages. Defaults to None.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
            yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to regress on. Defaults to
            "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, CORRECTED Std. Error, t-Statistic,
            P-Value), indexed by regressor name -- see `causal_inference_model.get_iv_2sls` for
            why these standard errors differ from (and correct) what a naive two-OLS-calls
            approach would report.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_iv_2sls("AAPL", "MSFT", "Benchmark", period="weekly")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|---------------:|--------------:|---------------:|-----------:|
        | Intercept |         0.0006 |        0.0025 |         0.2336 |     0.8156 |
        | MSFT      |         1.1453 |        0.0813 |        14.0938 |     0.0000 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        endogenous_ticker = (
            [endogenous_ticker]
            if isinstance(endogenous_ticker, str)
            else endogenous_ticker
        )
        instrument_tickers = (
            [instrument_tickers]
            if isinstance(instrument_tickers, str)
            else instrument_tickers
        )
        exogenous_tickers = (
            [exogenous_tickers]
            if isinstance(exogenous_tickers, str)
            else exogenous_tickers
        )

        result = causal_inference_model.get_iv_2sls(
            returns[dependent_ticker],
            returns[endogenous_ticker],
            returns[instrument_tickers],
            returns[exogenous_tickers] if exogenous_tickers else None,
            add_constant=add_constant,
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_difference_in_differences(
        self,
        treated_tickers: str | list[str],
        treatment_date: str,
        control_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Difference-in-Differences (DiD) regression estimating the effect of some
        event on `treatment_date` for `treated_tickers`, relative to `control_tickers`.

        Also known as: DiD, DD, difference-in-differences estimator.

        For more information about the method, and why the `Treated x Post` interaction
        coefficient IS the DiD treatment-effect estimate, see
        `causal_inference_model.get_difference_in_differences`.

        Builds a stacked (ticker, date) panel from every ticker in `treated_tickers` and
        `control_tickers`: each observation's outcome is that ticker's return on that
        date, `Treated` is 1 for every observation belonging to a `treated_tickers`
        asset (regardless of date) and 0 for `control_tickers` assets, and `Post` is 1
        for observations on or after `treatment_date` (regardless of asset). This
        answers "did `treated_tickers` behave differently after `treatment_date`,
        beyond both their normal average gap versus `control_tickers` and the common
        market-wide move over that same before/after window?" -- e.g. isolating the
        effect of an event (an index-inclusion announcement, a regulatory change
        affecting only some tickers, an earnings surprise) that hits `treated_tickers`
        but not `control_tickers`, at a known date.

        Args:
            treated_tickers (str | list[str]): The asset(s) subject to the event/treatment.
            treatment_date (str): The date the event/treatment occurs, in the same format
            accepted by `pd.Timestamp`. Observations on or after this date are `Post = 1`.
            control_tickers (str | list[str] | None, optional): The untreated comparison
            asset(s). Defaults to None, which uses every ticker (and "Benchmark", if
            present) NOT in `treated_tickers`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
            yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to use as the outcome.
            Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value)
            with rows `Intercept`, `Treated`, `Post` and `Treated x Post` -- the last of
            which is the DiD treatment-effect estimate.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_difference_in_differences(
            treated_tickers="AAPL", treatment_date="2021-06-30", period="weekly"
        )
        ```

        Which returns:

        |                 |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------------|---------------:|--------------:|---------------:|-----------:|
        | Intercept       |         0.0064 |        0.0031 |         2.0495 |     0.0410 |
        | Treated         |         0.0029 |        0.0054 |         0.5399 |     0.5896 |
        | Post            |        -0.0074 |        0.0045 |        -1.6572 |     0.0982 |
        | Treated x Post  |        -0.0020 |        0.0077 |        -0.2571 |     0.7972 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        treated_tickers = (
            [treated_tickers] if isinstance(treated_tickers, str) else treated_tickers
        )

        if control_tickers is None:
            control_tickers = [
                ticker for ticker in returns.columns if ticker not in treated_tickers
            ]
        elif isinstance(control_tickers, str):
            control_tickers = [control_tickers]

        panel_tickers = [*treated_tickers, *control_tickers]

        index = returns.index
        timestamps = (
            index.to_timestamp()
            if hasattr(index, "to_timestamp")
            else pd.DatetimeIndex(index)
        )
        post_flags = pd.Series(
            (timestamps >= pd.Timestamp(treatment_date)).astype(float),
            index=returns.index,
        )

        outcome_values = []
        treated_values = []
        post_values = []
        for ticker in panel_tickers:
            ticker_returns = returns[ticker].dropna()
            outcome_values.append(ticker_returns)
            treated_values.append(
                pd.Series(
                    1.0 if ticker in treated_tickers else 0.0,
                    index=ticker_returns.index,
                )
            )
            post_values.append(post_flags.loc[ticker_returns.index])

        outcome = pd.concat(outcome_values, ignore_index=True)
        treated = pd.concat(treated_values, ignore_index=True)
        post = pd.concat(post_values, ignore_index=True)

        result = causal_inference_model.get_difference_in_differences(
            outcome, treated, post, add_constant=add_constant
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_regression_discontinuity(
        self,
        dependent_ticker: str,
        running_variable_ticker: str,
        cutoff: float,
        period: str | None = None,
        column: str = "Return",
        bandwidth: float | None = None,
        kernel: str = "uniform",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Estimate a Sharp Regression Discontinuity (RDD): the jump in `dependent_ticker`
        exactly where `running_variable_ticker` crosses `cutoff`.

        Also known as: RDD, sharp RD.

        For more information about the method, see
        `causal_inference_model.get_regression_discontinuity`.

        Treats `running_variable_ticker`'s value as the running variable that
        (hypothetically) triggers some discrete change once it crosses `cutoff` -- e.g.
        testing whether `dependent_ticker`'s return behaves discontinuously around a
        round-number/threshold level of another asset or indicator (a psychological
        price level, an index-inclusion market-cap threshold, a macro indicator's
        policy-relevant threshold) fed in as `running_variable_ticker`. Fits separate
        local linear regressions of `dependent_ticker`'s return on the (cutoff-centered)
        `running_variable_ticker` value, one on each side of `cutoff`, and reports the
        gap between the two fitted lines exactly at the cutoff.

        Args:
            dependent_ticker (str): The outcome asset.
            running_variable_ticker (str): The asset (or column) whose value determines
            which side of `cutoff` an observation falls on.
            cutoff (float): The threshold value of `running_variable_ticker` at which the
            discontinuity is estimated.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
            yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to use for both series.
            Defaults to "Return".
            bandwidth (float | None, optional): The maximum distance from `cutoff` an
            observation may be to be included in either local regression. Defaults to
            None, which uses half of the running variable's observed range -- see
            `causal_inference_model.get_regression_discontinuity` for why this is a
            deliberately naive default.
            kernel (str, optional): One of "uniform" or "triangular". Defaults to "uniform".
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.DataFrame: A one-column results table with the discontinuity estimate, its
            standard error/t-statistic/p-value, and the cutoff/bandwidth/sample sizes used.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_regression_discontinuity(
            "AAPL", "MSFT", cutoff=0.0, period="weekly"
        )
        ```

        Which returns:

        |               |    Value |
        |:--------------|---------:|
        | Discontinuity |   0.0003 |
        | Std. Error    |   0.0077 |
        | t-Statistic   |   0.0343 |
        | P-Value       |   0.9726 |
        | Cutoff        |   0.0000 |
        | Bandwidth     |   0.1257 |
        | N Left        |  71      |
        | N Right       |  85      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        result = causal_inference_model.get_regression_discontinuity(
            returns[dependent_ticker],
            returns[running_variable_ticker],
            cutoff=cutoff,
            bandwidth=bandwidth,
            kernel=kernel,
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_propensity_score_matching(
        self,
        dependent_ticker: str,
        treatment_ticker: str,
        covariate_tickers: str | list[str],
        treatment_threshold: float = 0.0,
        period: str | None = None,
        column: str = "Return",
        caliper: float | None = None,
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Estimate the Average Treatment effect on the Treated (ATT) of `treatment_ticker`
        exceeding `treatment_threshold` on `dependent_ticker`'s return, via Propensity
        Score Matching (PSM) on `covariate_tickers`.

        Also known as: PSM, nearest-neighbor propensity matching.

        For more information about the method, see
        `causal_inference_model.get_propensity_score_matching`.

        Derives a binary "treatment" indicator the same way `get_logistic_regression`
        derives its binary outcome: 1 if `treatment_ticker`'s return in a given period
        exceeds `treatment_threshold` (0.0, i.e. a positive return, by default), else 0.
        This lets PSM answer e.g. "on periods where `treatment_ticker` has an
        outsized/positive move, is `dependent_ticker`'s return different than it would
        otherwise be -- comparing only periods that LOOK similar on `covariate_tickers`
        (to control for the possibility that `treatment_ticker` tends to move on the
        same periods/regimes that also independently affect `dependent_ticker`)?"

        Args:
            dependent_ticker (str): The outcome asset.
            treatment_ticker (str): The asset whose return, once it exceeds
            `treatment_threshold`, defines the treatment indicator.
            covariate_tickers (str | list[str]): The asset(s) used as covariates to
            estimate the propensity score -- should include asset(s) believed to drive
            selection into "treatment".
            treatment_threshold (float, optional): The return threshold defining
            treatment. Defaults to 0.0.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
            yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to use. Defaults to "Return".
            caliper (float | None, optional): The maximum allowed logit-propensity-score
            matching distance. Defaults to None, which uses Austin's (2011) rule of thumb
            -- see `causal_inference_model.get_propensity_score_matching`.
            add_constant (bool, optional): Whether to include an intercept in the
            propensity score model. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to.
            Defaults to None.

        Returns:
            pd.Series: The ATT estimate, its standard error/t-statistic/p-value, and the
            number of matched pairs/treated/control observations.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_propensity_score_matching(
            "AAPL", "MSFT", "Benchmark", period="weekly"
        )
        ```

        Which returns:

        | Metric        |    Value |
        |:--------------|---------:|
        | ATT           |   0.0335 |
        | Std. Error    |   0.0060 |
        | t-Statistic   |   5.5598 |
        | P-Value       |   0.0000 |
        | Matched Pairs |  33      |
        | N Treated     |  84      |
        | N Control     |  73      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        covariate_tickers = (
            [covariate_tickers]
            if isinstance(covariate_tickers, str)
            else covariate_tickers
        )

        treatment = (returns[treatment_ticker] > treatment_threshold).astype(float)

        result = causal_inference_model.get_propensity_score_matching(
            treatment,
            returns[dependent_ticker],
            returns[covariate_tickers],
            caliper=caliper,
            add_constant=add_constant,
        )

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    def _get_panel_data(
        self,
        independent_tickers: str | list[str] | None,
        independent_column: str | None,
        dependent_tickers: str | list[str] | None,
        period: str | None,
        column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Shared input-shaping step for `get_fixed_effects`/`get_random_effects`/
        `get_hausman_test`: unlike every other regression method in this module
        (which pairs one dependent ticker with one or more independent tickers,
        all on equal footing), panel data methods need a genuine entity dimension
        that a handful of single ticker return series don't have on their own.

        This module treats `dependent_tickers` (defaulting to every ticker in the
        `Toolkit` instance, minus `independent_tickers` if given) as the panel of
        ENTITIES -- e.g. a panel of stocks -- and supports two, mutually exclusive
        ways of building the regressor `x`:

        - `independent_tickers`: a COMMON, time-varying factor (or factors),
        broadcast identically to every entity at each point in time -- e.g. a
        market benchmark, mirroring a Fama-French-style factor regression.
        - `independent_column`: a PER-ENTITY regressor -- each entity's own value
        of a different historical data column (e.g. `"Volume"` explaining
        `"Return"`), rather than an external, shared series.

        Both describe "does X explain Y across a panel of stocks, controlling for
        stock fixed effects," but they are not interchangeable for every estimator
        here: `independent_tickers` carries no BETWEEN-entity variation once
        broadcast to every entity over the same dates (this `Toolkit`'s historical
        data is always aligned onto a common calendar across tickers), which makes
        Random Effects'/the Hausman test's between-entity regression step exactly
        rank-deficient (the entity-mean of a broadcast series is identical for
        every entity, perfectly collinear with the intercept). `independent_column`
        does not have this problem, since different entities genuinely have
        different average Volume (or whichever column), and so is the more
        reliable choice for `get_random_effects`/`get_hausman_test`, while
        `independent_tickers` remains natural (and works for every estimator here,
        including these two, granted enough cross-entity variation) for
        `get_fixed_effects`'s canonical Fama-French-style use case.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, list[str]]: `y` shaped entity (ticker)
            x time (a wide `pd.DataFrame`, accepted as-is by
            `panel_data_model.get_fixed_effects`/`get_random_effects`), `x` in the
            same shape (single regressor) or `(entity, time)`-MultiIndexed (multiple
            `independent_tickers`), and the resolved list of dependent (entity)
            tickers.

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, the period is invalid, or no dependent
            (entity) tickers remain.
        """
        if (independent_tickers is None) == (independent_column is None):
            raise ValueError(
                "Provide exactly one of independent_tickers (a common factor "
                "broadcast to every entity) or independent_column (a per-entity "
                "regressor taken from each dependent ticker's own data)."
            )

        period = period if period else "quarterly" if self._quarterly else "yearly"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        # Deliberately does NOT reuse `_get_price_column` (which drops any row
        # missing data for ANY ticker in the `Toolkit` instance): panel methods
        # need each entity's own NaN pattern preserved (e.g. a ticker that IPO'd
        # partway through the sample), so that `panel_data_model`'s internal
        # alignment step can drop only the genuinely missing `(entity, time)`
        # pairs, rather than every entity's observations on any date where a
        # single ticker anywhere in the `Toolkit` instance happens to be missing.
        prices = self._historical_data[period][column]

        if independent_tickers is not None:
            independent_tickers = (
                [independent_tickers]
                if isinstance(independent_tickers, str)
                else list(independent_tickers)
            )

            if dependent_tickers is None:
                dependent_tickers = [
                    ticker
                    for ticker in prices.columns
                    if ticker not in independent_tickers
                ]
            else:
                dependent_tickers = (
                    [dependent_tickers]
                    if isinstance(dependent_tickers, str)
                    else list(dependent_tickers)
                )

            if not dependent_tickers:
                raise ValueError(
                    "No dependent (entity) tickers remain after excluding "
                    "independent_tickers -- provide dependent_tickers explicitly."
                )

            y_panel = prices[dependent_tickers].T
            factor_values = prices[independent_tickers]
            x_panel = pd.concat(
                {entity: factor_values for entity in dependent_tickers},
                names=["entity", "time"],
            )
        else:
            if dependent_tickers is None:
                dependent_tickers = list(prices.columns)
            else:
                dependent_tickers = (
                    [dependent_tickers]
                    if isinstance(dependent_tickers, str)
                    else list(dependent_tickers)
                )

            if not dependent_tickers:
                raise ValueError(
                    "No dependent (entity) tickers -- provide dependent_tickers "
                    "explicitly."
                )

            independent_prices = self._historical_data[period][independent_column]

            y_panel = prices[dependent_tickers].T
            x_series = independent_prices[dependent_tickers].T.stack()
            x_series.name = independent_column
            x_panel = x_series.to_frame()

        return y_panel, x_panel, dependent_tickers

    @handle_errors
    def get_fixed_effects(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        entity_effects: bool = True,
        time_effects: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Fixed Effects ("within") estimator explaining a panel of
        `dependent_tickers` (the entities) by a regressor `x`, controlling for
        entity-specific (and/or time-specific) fixed effects.

        Also known as: within estimator, FE, least squares dummy variable (LSDV)
        estimator.

        Unlike this module's other regression methods (`get_ols`, `get_wls`, ...),
        which each compare a handful of individual ticker return series on equal
        footing, this treats `dependent_tickers` as a genuine panel of entities --
        e.g. every stock in the `Toolkit` instance -- observed over time. The
        regressor `x` is built in exactly one of two ways (provide exactly one of
        the two arguments below):

        - `independent_tickers`: a COMMON factor (or factors) applied identically
        to every entity at each date (e.g. a market benchmark, mirroring a
        Fama-French-style factor regression).
        - `independent_column`: a PER-ENTITY regressor -- each entity's own value
        of a *different* historical data column (e.g. does `"Volume"` explain
        `"Return"`, across the panel).

        Fixed Effects removes any purely entity-specific, time-invariant
        characteristic (e.g. a stock's typical risk premium) before estimating the
        regressor's coefficient(s), by demeaning every variable by its entity's
        mean. See `panel_data_model.get_fixed_effects` for the full formula and
        references.

        Args:
            independent_tickers (str | list[str] | None, optional): The factor
            ticker(s), whose `column` values are broadcast identically to every
            entity at each date. Mutually exclusive with `independent_column`.
            independent_column (str | None, optional): A different historical data
            column, taken per-entity from each of `dependent_tickers`' own data, to
            use as the regressor. Mutually exclusive with `independent_tickers`.
            dependent_tickers (str | list[str] | None, optional): The panel of
            entity tickers to explain. Defaults to None, meaning every ticker in
            the `Toolkit` instance (other than `independent_tickers`, if given).
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            entity_effects (bool, optional): Whether to control for time-invariant
            entity-specific characteristics. Defaults to True.
            time_effects (bool, optional): Whether to control for entity-invariant,
            time-specific shocks. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error,
            t-Statistic, P-Value) for the regressor(s), indexed by name, with the
            recovered entity (and/or time) fixed-effect intercepts appended as
            additional rows (only the Coefficient column is populated for those).

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, or (see `panel_data_model.
            get_fixed_effects`) if there are not enough observations for the
            requested fixed effects.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_fixed_effects(
            independent_tickers="Benchmark", period="weekly"
        )
        ```

        Which returns:

        |                        |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:-----------------------|---------------:|--------------:|---------------:|-----------:|
        | Benchmark              |         1.0772 |        0.0331 |        32.5314 |     0.0000 |
        | Entity Effect: AAPL    |         0.0022 |      nan      |       nan      |   nan      |
        | Entity Effect: MSFT    |         0.0017 |      nan      |       nan      |   nan      |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers, independent_column, dependent_tickers, period, column
        )

        result = panel_data_model.get_fixed_effects(
            y_panel, x_panel, entity_effects=entity_effects, time_effects=time_effects
        )

        rounding = rounding if rounding is not None else self._rounding
        summary = result.regression.summary()

        if result.entity_effects is not None:
            entity_rows = pd.DataFrame({"Coefficient": result.entity_effects})
            entity_rows.index = [
                f"Entity Effect: {entity}" for entity in entity_rows.index
            ]
            summary = pd.concat([summary, entity_rows])

        if result.time_effects is not None:
            time_rows = pd.DataFrame({"Coefficient": result.time_effects})
            time_rows.index = [f"Time Effect: {time}" for time in time_rows.index]
            summary = pd.concat([summary, time_rows])

        return summary.round(rounding)

    @handle_errors
    def get_random_effects(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Random Effects estimator explaining a panel of `dependent_tickers`
        (the entities) by a regressor `x`, via Swamy-Arora feasible Generalized
        Least Squares.

        Also known as: RE, GLS panel estimator, Swamy-Arora estimator.

        See `get_fixed_effects` for how `dependent_tickers`/`independent_tickers`/
        `independent_column` are shaped into a panel, and
        `panel_data_model.get_random_effects` for the full estimator formula,
        references, and how it compares to Fixed Effects. Unlike Fixed Effects,
        Random Effects retains and estimates a single, population-average
        intercept rather than one intercept per entity -- more efficient than
        Fixed Effects if entity effects are indeed uncorrelated with the
        regressor(s) (see `get_hausman_test` to check that assumption).

        Args:
            independent_tickers (str | list[str] | None, optional): The factor
            ticker(s), whose `column` values are broadcast identically to every
            entity at each date. Mutually exclusive with `independent_column`.
            Note: since this `Toolkit` instance's historical data is aligned onto a
            common calendar across tickers, a broadcast factor's entity mean is
            identical for every entity, leaving no between-entity variation to
            identify Random Effects' between-regression step -- prefer
            `independent_column` here unless `dependent_tickers` genuinely differ
            in their date coverage (e.g. different listing histories).
            independent_column (str | None, optional): A different historical data
            column, taken per-entity from each of `dependent_tickers`' own data, to
            use as the regressor. Mutually exclusive with `independent_tickers`.
            dependent_tickers (str | list[str] | None, optional): The panel of
            entity tickers to explain. Defaults to None, meaning every ticker in
            the `Toolkit` instance (other than `independent_tickers`, if given).
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error,
            t-Statistic, P-Value), indexed by regressor name plus "Intercept".

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, or (see `panel_data_model.
            get_random_effects`) if there are not enough entities or observations
            to estimate the model.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY"
        )

        toolkit.econometrics.get_random_effects(
            independent_column="Volume", period="weekly"
        )
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|---------------:|--------------:|---------------:|-----------:|
        | Intercept |         0.0094 |        0.0012 |         7.7341 |     0.0000 |
        | Volume    |        -0.0000 |        0.0000 |        -5.1258 |     0.0000 |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers, independent_column, dependent_tickers, period, column
        )

        result = panel_data_model.get_random_effects(y_panel, x_panel)

        return result.summary().round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_hausman_test(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Hausman specification test comparing a Fixed Effects and a
        Random Effects fit of `dependent_tickers` (the panel of entities) on a
        regressor `x`.

        Also known as: Hausman specification test, Hausman-Wu test.

        Random Effects is more efficient than Fixed Effects but relies on entity
        effects being uncorrelated with the regressor(s) -- if that assumption is
        violated, Random Effects is inconsistent while Fixed Effects remains
        consistent regardless. See `panel_data_model.get_hausman_test` for the full
        formula and references, and `get_fixed_effects`/`get_random_effects` for
        how the two models being compared are estimated (including how
        `independent_tickers`/`independent_column` shape the panel, and why
        `independent_column` is generally the more reliable choice here).

        Args:
            independent_tickers (str | list[str] | None, optional): The factor
            ticker(s), whose `column` values are broadcast identically to every
            entity at each date. Mutually exclusive with `independent_column`.
            independent_column (str | None, optional): A different historical data
            column, taken per-entity from each of `dependent_tickers`' own data, to
            use as the regressor. Mutually exclusive with `independent_tickers`.
            dependent_tickers (str | list[str] | None, optional): The panel of
            entity tickers to explain. Defaults to None, meaning every ticker in
            the `Toolkit` instance (other than `independent_tickers`, if given).
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.Series: The Hausman statistic, its degrees of freedom, its p-value,
            and whether Fixed Effects is preferred over Random Effects at the 5%
            level.

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, or (see `panel_data_model.
            get_hausman_test`) if there are not enough observations to fit either
            model, or the two estimators share no common regressors to compare.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY"
        )

        toolkit.econometrics.get_hausman_test(
            independent_column="Volume", period="weekly"
        )
        ```

        Which returns:

        | Metric                     |   Value |
        |:----------------------------|--------:|
        | Hausman Statistic           |  8.1289 |
        | Degrees of Freedom          |  1      |
        | P-Value                     |  0.0044 |
        | Prefer Fixed Effects (5%)   |  1      |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers, independent_column, dependent_tickers, period, column
        )

        result = panel_data_model.get_hausman_test(y_panel, x_panel)

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_arima_forecast(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        p: int = 1,
        d: int = 1,
        q: int = 1,
        forecast_steps: int = 5,
        include_constant: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit an ARIMA(p, d, q) model to every ticker's price series in the Toolkit
        instance and forecast `forecast_steps` periods ahead.

        Also known as: Box-Jenkins model, autoregressive integrated moving average.

        An ARIMA(p, d, q) model differences the series `d` times to remove a
        (stochastic) trend, then fits an autoregressive-moving-average model to the
        result -- see `time_series_model.get_arima_forecast` for the full formula,
        estimation method (Conditional Sum of Squares, not exact Maximum Likelihood --
        an important, deliberately documented limitation) and its practical caveats.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to fit. Defaults to
            "Adj Close".
            p (int, optional): The autoregressive order. Defaults to 1.
            d (int, optional): The number of times to difference the series. Defaults
            to 1 (the typical choice for a non-stationary price level series).
            q (int, optional): The moving-average order. Defaults to 1.
            forecast_steps (int, optional): The number of periods ahead to forecast.
            Defaults to 5.
            include_constant (bool, optional): Whether to estimate a free intercept.
            Defaults to True -- see `time_series_model.get_arima_forecast` for when
            to set this to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers forecast. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: The `forecast_steps`-ahead forecast, one column per ticker,
            indexed `1, ..., forecast_steps` (periods ahead, not a continuation of the
            historical date index). Each ticker's ARIMA model is fit independently.

        Notes:
        - The method retrieves historical price data based on the specified `period`
        for every ticker in the Toolkit instance and fits an ARIMA model to each.
        - See `time_series_model.get_arima_forecast`'s `Notes` for this estimator's
        practical limitations (in particular, keep `p + q` small).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_arima_forecast(period="quarterly")
        ```

        Which returns:

        | Step |    AAPL |    MSFT |
        |-----:|--------:|--------:|
        |    1 | 138.273 | 243.199 |
        |    2 | 147.039 | 250.190 |
        |    3 | 154.992 | 257.106 |
        |    4 | 162.323 | 264.014 |
        |    5 | 169.180 | 270.922 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._get_price_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        result = pd.DataFrame(
            {
                ticker: time_series_model.get_arima_forecast(
                    prices[ticker],
                    p=p,
                    d=d,
                    q=q,
                    forecast_steps=forecast_steps,
                    include_constant=include_constant,
                ).forecast
                for ticker in tickers
            }
        )
        result.index.name = "Step"

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_var_forecast(
        self,
        period: str | None = None,
        column: str = "Return",
        lags: int = 1,
        forecast_steps: int = 5,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Vector Autoregression (VAR) across every ticker in the Toolkit
        instance and forecast `forecast_steps` periods ahead.

        Also known as: VAR model, vector autoregressive model.

        A VAR jointly models every ticker's series, regressing each of them on `lags`
        lagged values of ALL of them (including itself) -- see
        `time_series_model.get_var_forecast` for the full formula and estimation
        method (equation-by-equation OLS, reusing `regression_model.get_ols`).

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to model. Defaults to
            "Return".
            lags (int, optional): The VAR order. Defaults to 1.
            forecast_steps (int, optional): The number of periods ahead to forecast.
            Defaults to 5.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: The `forecast_steps`-ahead forecast, one column per ticker,
            indexed `1, ..., forecast_steps` (periods ahead).

        Notes:
        - The method retrieves historical data based on the specified `period` for every
        ticker in the Toolkit instance and fits the VAR model jointly across all of them.
        - Unlike `get_vecm_forecast`, this operates on `column` directly (typically
        "Return", a stationary series) rather than price levels -- a VAR in levels is
        misspecified for non-stationary series (see
        `cointegration_model.get_johansen_cointegration`); use `get_vecm_forecast`
        instead if the assets' price levels are cointegrated.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_var_forecast(period="quarterly")
        ```

        Which returns:

        | Step |   AAPL |   MSFT |
        |-----:|-------:|-------:|
        |    1 | 0.1271 | 0.1046 |
        |    2 | 0.1069 | 0.0640 |
        |    3 | 0.0704 | 0.0428 |
        |    4 | 0.0661 | 0.0454 |
        |    5 | 0.0716 | 0.0496 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        result = time_series_model.get_var_forecast(
            returns[tickers], lags=lags, forecast_steps=forecast_steps
        )

        return result.forecast.round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_vecm_forecast(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        k_ar_diff: int = 1,
        forecast_steps: int = 5,
        significance: float = 0.05,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Vector Error Correction Model (VECM) across every (cointegrated)
        ticker in the Toolkit instance and forecast `forecast_steps` periods ahead.

        Also known as: VECM, error correction model (for the multivariate/cointegrated
        case).

        A VECM keeps a VAR's short-run dynamics while ALSO letting each asset's price
        change react to how far the system currently sits from its long-run
        equilibrium (the cointegrating relationship(s) among the tickers, taken from
        `cointegration_model.get_johansen_cointegration`) -- see
        `time_series_model.get_vecm_forecast` for the full formula, estimation method
        and verification notes.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to model. Defaults to
            "Adj Close" -- a VECM needs price LEVELS (non-stationary, cointegrated
            series), not returns, the same input `get_johansen_cointegration` expects.
            k_ar_diff (int, optional): The number of lagged first differences to
            include as short-run dynamics. Defaults to 1.
            forecast_steps (int, optional): The number of periods ahead to forecast.
            Defaults to 5.
            significance (float, optional): The significance level (one of 0.01, 0.05,
            0.10) at which the Johansen trace test determines the cointegrating rank.
            Defaults to 0.05.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: The `forecast_steps`-ahead forecast, on the price-level
            scale, one column per ticker, indexed `1, ..., forecast_steps` (periods
            ahead).

        Raises:
            ValueError: If the Johansen test does not reject a cointegrating rank of 0
            for the Toolkit instance's tickers at `significance` -- a VECM is not
            appropriate for non-cointegrated assets; use `get_var_forecast` on their
            returns instead.

        Notes:
        - The method retrieves historical price data based on the specified `period`
        for every ticker in the Toolkit instance and fits the VECM jointly across all
        of them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        # AAPL/MSFT alone aren't cointegrated in this sample -- add Benchmark to the
        # system to get one that is.
        toolkit.econometrics.get_vecm_forecast(period="quarterly", include_benchmark=True)
        ```

        Which returns:

        | Step |    AAPL |    MSFT |   Benchmark |
        |-----:|--------:|--------:|------------:|
        |    1 | 131.672 | 237.052 |     380.084 |
        |    2 | 136.414 | 266.892 |     404.920 |
        |    3 | 147.872 | 281.265 |     413.975 |
        |    4 | 154.833 | 289.048 |     413.568 |
        |    5 | 152.306 | 273.929 |     396.546 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._get_price_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        result = time_series_model.get_vecm_forecast(
            prices[tickers],
            k_ar_diff=k_ar_diff,
            forecast_steps=forecast_steps,
            significance=significance,
        )

        return result.forecast.round(
            rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_rmse(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Root Mean Squared Error (RMSE) between every unordered pair
        of tickers' series in the Toolkit instance.

        Also known as: RMSD (Root Mean Squared Deviation).

        See `forecast_evaluation_model.get_rmse` for the formula. This controller
        method compares two ASSETS' series directly, treating one as a naive
        "forecast" proxy for the other -- a simple, tracking-error style measure of
        how closely two series move together in absolute deviation terms (e.g. a
        portfolio versus a benchmark, or one asset as a naive stand-in forecast for
        a similar one). For evaluating an actual FORECASTING MODEL (ARIMA/VAR)
        rather than one asset as a naive proxy for another, use
        `get_out_of_sample_validation` instead.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to compare. Defaults to
            "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.Series: The RMSE, indexed by unordered `(Ticker A, Ticker B)` pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_rmse(period="quarterly")
        ```

        Which returns:

        | Ticker A   | Ticker B   |   RMSE |
        |:-----------|:-----------|-------:|
        | AAPL       | MSFT       | 0.1084 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        values = {
            (ticker_a, ticker_b): forecast_evaluation_model.get_rmse(
                returns[ticker_a], returns[ticker_b]
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        result = pd.Series(values, name="RMSE")
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_mae(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Mean Absolute Error (MAE) between every unordered pair of
        tickers' series in the Toolkit instance.

        Also known as: MAD (Mean Absolute Deviation).

        See `forecast_evaluation_model.get_mae` for the formula, and `get_rmse`'s
        docstring for why this controller method compares two ASSETS directly (rather
        than an asset against an actual forecasting model's output -- see
        `get_out_of_sample_validation` for that).

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to compare. Defaults to
            "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            tickers paired up. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.Series: The MAE, indexed by unordered `(Ticker A, Ticker B)` pair.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_mae(period="quarterly")
        ```

        Which returns:

        | Ticker A   | Ticker B   |    MAE |
        |:-----------|:-----------|-------:|
        | AAPL       | MSFT       | 0.0887 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_price_column(period, column)

        values = {
            (ticker_a, ticker_b): forecast_evaluation_model.get_mae(
                returns[ticker_a], returns[ticker_b]
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        result = pd.Series(values, name="MAE")
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_out_of_sample_validation(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        model: str = "arima",
        train_fraction: float = 0.8,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        include_constant: bool = True,
        lags: int = 1,
        other_tickers: list[str] | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Out-of-sample validate an ARIMA or VAR forecast of every ticker in the
        Toolkit instance -- fit only on a training portion of the history,
        forecast the held-out remainder, and score the forecast against what
        actually happened.

        Also known as: hold-out validation, train/test split validation.

        See `forecast_evaluation_model.get_out_of_sample_validation` for the general
        harness this wraps. Since that function takes a raw Python callable (not
        serializable for e.g. the MCP-facing tool layer), this controller method
        instead hardcodes the choice between the two Part-1 forecasting models via the
        `model` string:

        - `model="arima"`: `time_series_model.get_arima_forecast` is fit on each
          ticker's own training-period series (`p`, `d`, `q`, `include_constant`
          control the model, same as `get_arima_forecast`).
        - `model="var"`: `time_series_model.get_var_forecast` is fit on the
          training-period series of each ticker together with `other_tickers`
          (`lags` controls the VAR order, defaulting to every other ticker in the
          Toolkit instance if not given); only that ticker's own forecast column
          is scored against its holdout.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly,
            quarterly, or yearly). Defaults to "quarterly".
            column (str, optional): The historical data column to validate. Defaults
            to "Adj Close".
            model (str, optional): Either "arima" or "var". Defaults to "arima".
            train_fraction (float, optional): The fraction of observations used for
            training; the remainder is the holdout. Defaults to 0.8.
            p (int, optional): The ARIMA autoregressive order (`model="arima"` only).
            Defaults to 1.
            d (int, optional): The ARIMA differencing order (`model="arima"` only).
            Defaults to 1.
            q (int, optional): The ARIMA moving-average order (`model="arima"` only).
            Defaults to 1.
            include_constant (bool, optional): Whether the ARIMA model estimates a
            free intercept (`model="arima"` only). Defaults to True.
            lags (int, optional): The VAR order (`model="var"` only). Defaults to 1.
            other_tickers (list[str] | None, optional): The other assets to include in
            the VAR system alongside the ticker being validated (`model="var"` only).
            Defaults to None, meaning every other ticker in the Toolkit instance.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers validated (and, for `model="var"`, among the default
            `other_tickers`). Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: `RMSE`, `MAE` and `Holdout Observations`, one column per
            ticker.

        Raises:
            ValueError: If `model` is not "arima" or "var", or (for `model="var"`)
            fewer than 2 tickers are available to form a VAR system.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.econometrics.get_out_of_sample_validation(
            period="weekly", model="arima", p=1, d=1, q=1
        )
        ```

        Which returns:

        |                       |    AAPL |    MSFT |
        |:----------------------|--------:|--------:|
        | RMSE                  | 12.9091 | 24.2258 |
        | MAE                   | 10.2476 | 20.6824 |
        | Holdout Observations  | 32      | 32      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._get_price_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        if model not in ("arima", "var"):
            raise ValueError(f"model must be 'arima' or 'var', received '{model}'.")

        columns = {}
        for ticker in tickers:
            if model == "arima":
                columns[ticker] = (
                    forecast_evaluation_model.get_out_of_sample_validation(
                        prices[ticker],
                        time_series_model.get_arima_forecast,
                        train_fraction=train_fraction,
                        p=p,
                        d=d,
                        q=q,
                        include_constant=include_constant,
                    )
                )
                continue

            ticker_other_tickers = (
                other_tickers
                if other_tickers is not None
                else [
                    other_ticker for other_ticker in tickers if other_ticker != ticker
                ]
            )

            if not ticker_other_tickers:
                raise ValueError(
                    "other_tickers must be provided (and non-empty) when model='var' "
                    "-- a VAR needs at least one other series to model jointly with "
                    "each ticker."
                )

            system = prices[[ticker, *ticker_other_tickers]].dropna()
            split = int(len(system) * train_fraction)
            train, holdout = system.iloc[:split], system.iloc[split:]

            var_result = time_series_model.get_var_forecast(
                train, lags=lags, forecast_steps=len(holdout)
            )
            forecast = pd.Series(
                var_result.forecast[ticker].to_numpy(), index=holdout.index
            )

            columns[ticker] = pd.Series(
                {
                    "RMSE": forecast_evaluation_model.get_rmse(
                        holdout[ticker], forecast
                    ),
                    "MAE": forecast_evaluation_model.get_mae(holdout[ticker], forecast),
                    "Holdout Observations": len(holdout),
                }
            )

        result = pd.DataFrame(columns)

        return result.round(rounding if rounding is not None else self._rounding)
