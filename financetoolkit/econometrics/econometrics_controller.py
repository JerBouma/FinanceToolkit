"""Econometrics Module"""

__docformat__ = "google"

import itertools
import warnings
from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd

from financetoolkit.econometrics import (
    causal_inference_model,
    causality_model,
    cointegration_model,
    data_model,
    diagnostics_model,
    event_study_model,
    fama_macbeth_model,
    forecast_evaluation_model,
    hypothesis_testing_model,
    pairs_trading_model,
    panel_data_model,
    regime_model,
    regression_model,
    specification_tests_model,
    state_space_model,
    time_series_model,
    unitroot_model,
)
from financetoolkit.helpers import LazyDict, handle_portfolio
from financetoolkit.risk.helpers import determine_within_historical_data
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import apply_rounding

logger = get_logger()

# Division by zero is normal in these calculations, not a bug.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals
# pylint: disable=too-many-boolean-expressions
# The examples' output tables are wider than the line length.
# ruff: noqa: E501

# The synthetic benchmark series, excluded unless `include_benchmark=True`.
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

    Through the Toolkit (`toolkit.econometrics`) it analyzes the prices and returns of the
    Toolkit's tickers. Standalone, `Econometrics(data=...)` analyzes any metric: ratios,
    economic indicators, yield curves or a dataset of your own.
    """

    def __init__(
        self,
        tickers: str | list[str] | Callable[[], list[str]] | None = None,
        historical_data: dict[str, pd.DataFrame] | None = None,
        intraday_period: str | None = None,
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        data: pd.DataFrame | pd.Series | dict | None = None,
        changes: str | None = "percentage",
    ):
        """
        Initializes the Econometrics Controller Class, either with the historical data of a
        Toolkit (which `toolkit.econometrics` passes) or standalone with any data.

        Standalone, the data can be a DataFrame with a column per series, a Series, or a
        dictionary of them; the dates can be in the rows or, as in the ratios of the Toolkit
        (tickers by year), in the columns. Every series becomes an asset: the series as
        given is the "Adj Close" column (also "Level") that the unit root, cointegration and
        forecasting methods use by default, and its change from one observation to the next
        is the "Return" column (also "Change") that the regressions and diagnostics use. The
        "daily" period, the default, holds the series at the dates they are observed (so a
        monthly series is analyzed monthly); the other periods take the last observation of
        each week, month, quarter or year.

        The Toolkit passes the tickers as a function without arguments and the historical data
        as a LazyDict (financetoolkit.helpers), so a method retrieves the data of the period it
        is calculated over when it is first needed, and nothing else.

        Args:
            tickers (str | list[str] | Callable[[], list[str]] | None, optional): The tickers of the
                Toolkit, or a function without arguments that returns them. Defaults to None, which
                takes the names of the series when data is given.
            historical_data (dict[str, pd.DataFrame] | None, optional): The historical data per period,
                as the Toolkit passes it. Defaults to None, which is treated as an empty dictionary.
            intraday_period (str | None, optional): The intraday period used for within-period calculations.
                Defaults to None.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str | None, optional): The start date to use for the calculations. Defaults to None.
            end_date (str | None, optional): The end date to use for the calculations. Defaults to None.
            data (pd.DataFrame | pd.Series | dict | None, optional): Any data to analyze instead
                of the Toolkit's prices, with a series per column. Defaults to None.
            changes (str | None, optional): How the changes ("Return") of the data are calculated:
                "percentage" (relative changes, as returns), "log", "difference" (first
                differences, for rates and ratios that can be zero or negative) or None
                (the data are changes already). The percentage change divides by the
                absolute previous value, so a negative series that rises has a positive
                change. Defaults to "percentage".

        As an example:

        ```python
        from financetoolkit import Economics, Econometrics

        economics = Economics(start_date='2000-01-01', end_date='2025-12-31')

        unemployment = economics.get_unemployment_rate(
            countries=['United States', 'Germany'], period='monthly'
        )

        econometrics = Econometrics(data=unemployment, changes='difference')

        econometrics.get_augmented_dickey_fuller()
        ```

        Which returns:

        |                       |   United States |    Germany |
        |:----------------------|----------------:|-----------:|
        | ADF Statistic         |      -2.98304   |  -0.930407 |
        | P-Value               |       0.0365087 |   0.777794 |
        | Lags Used             |       0         |   3        |
        | Observations          |     310         | 307        |
        | Critical Value 1%     |      -3.45162   |  -3.45183  |
        | Critical Value 5%     |      -2.87091   |  -2.871    |
        | Critical Value 10%    |      -2.57176   |  -2.57181  |
        | Reject Unit Root (5%) |       1         |   0        |
        """
        if data is not None:
            if historical_data is not None:
                raise ValueError(
                    "Pass either data (any metric) or historical_data (from a Toolkit), not both."
                )
            historical_data, series = data_model.build_historical_data(
                data, changes=changes, start_date=start_date, end_date=end_date
            )
            tickers = series if tickers is None else tickers
        if tickers is None:
            raise ValueError(
                "Pass the data to analyze, e.g. Econometrics(data=dataframe), or use the "
                "Econometrics module of a Toolkit: toolkit.econometrics."
            )

        # Mirrors the Risk controller: the Toolkit always passes the period dict, and an
        # explicit empty dict keeps the lookups raising a plain KeyError.
        self._historical_data = historical_data if historical_data is not None else {}
        self._tickers_source = tickers
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._portfolio_weights: dict | None = None
        # Memo for the sliced-and-dropna'd frames handed to the calculations: nearly
        # every method starts from the same (period, column) slice, and re-copying it
        # per call adds up when many metrics are collected in a row (as the MCP server
        # does). The historical data is never mutated once retrieved, and the
        # slices are treated as read-only by every caller, so sharing them is safe.
        self._data_cache: dict[tuple, pd.DataFrame] = {}

        # The returns within each period, built when a method first asks for a period.
        # Missing returns stay NaN (not 0), so a ticker without data in part of a period is
        # not treated as flat there.
        self._within_historical_data = LazyDict(
            lambda period: determine_within_historical_data(
                period, self._historical_data, intraday_period
            )
        )

    @cached_property
    def _tickers(self) -> list[str]:
        """The tickers as a list, retrieved when first needed."""
        tickers = (
            self._tickers_source()
            if callable(self._tickers_source)
            else self._tickers_source
        )

        return [tickers] if isinstance(tickers, str) else list(tickers)

    @handle_portfolio
    @handle_errors
    def get_arch_lm_test(
        self,
        period: str | None = None,
        within_period: bool = False,
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

        The formula is as follows:

        - εₜ = rₜ - mean(r), the returns of `period` demeaned by their sample mean
        - Auxiliary regression: εₜ² = a₀ + Σ(i=1..q) aᵢ·εₜ₋ᵢ² + uₜ, with q = `lags` (5 by default)
        - ARCH-LM = (T - q)·R², chi-squared with q degrees of freedom, where T - q is the number of
          observations in the auxiliary regression

        For more information about the method, see the following paper:

        - Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with
        Estimates of the Variance of United Kingdom Inflation." Econometrica, 50(4),
        987-1008.

        Also known as: ARCH-LM test, Engle's ARCH test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to False. Note that this requires intraday data
            when period = 'daily', since a single day only nests observations when intraday data was fetched.
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

        toolkit = Toolkit(
            ["BTCUSD", "^GSPC", "EURUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_arch_lm_test()
        ```

        Which returns:

        |                   |   BTCUSD |   ^GSPC |   EURUSD |
        |:------------------|---------:|--------:|---------:|
        | ARCH-LM Statistic |  55.2071 | 165.389 |  67.7588 |
        | P-Value           |   0      |   0     |   0      |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_arch_lm_test(returns, lags=lags)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_jarque_bera_test(
        self,
        period: str | None = None,
        within_period: bool = False,
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

        The formula is as follows:

        - JB = (n / 6)·(S² + (K - 3)² / 4), chi-squared with 2 degrees of freedom
        - S = m₃ / m₂^(3/2) and K = m₄ / m₂², with mₖ = (1 / n)·Σ(rₜ - mean(r))ᵏ the (biased) central
          moments of the returns of `period`

        For more information about the method, see the following paper:

        - Jarque, C.M. and Bera, A.K. (1987). "A Test for Normality of Observations
        and Regression Residuals." International Statistical Review, 55(2), 163-172.

        Also known as: JB test, normality test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to False. Note that this requires intraday data
            when period = 'daily', since a single day only nests observations when intraday data was fetched.
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

        toolkit = Toolkit(
            ["^GSPC", "BTCUSD", "TLT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_jarque_bera_test(period="monthly")
        ```

        Which returns:

        |                       |   ^GSPC |   BTCUSD |    TLT |
        |:----------------------|--------:|---------:|-------:|
        | Jarque-Bera Statistic |  2.6001 |   1.0834 | 0.6166 |
        | P-Value               |  0.2725 |   0.5818 | 0.7347 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_jarque_bera_test(returns)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_ljung_box_test(
        self,
        period: str | None = None,
        within_period: bool = False,
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

        The formula is as follows:

        - Q = n·(n + 2)·Σ(k=1..h) ρₖ² / (n - k), chi-squared with h degrees of freedom, h = `lags` (10 by
          default)
        - ρₖ = Σ(t=k+1..n) (rₜ - mean(r))·(rₜ₋ₖ - mean(r)) / Σ(t=1..n) (rₜ - mean(r))², the sample
          autocorrelation of the returns of `period` at lag k

        For more information about the method, see the following paper:

        - Ljung, G.M., & Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time Series
        Models." Biometrika, 65(2), 297-303.

        Also known as: Ljung-Box Q test, portmanteau test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to False. Note that this requires intraday data
            when period = 'daily', since a single day only nests observations when intraday data was fetched.
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

        toolkit = Toolkit(
            ["WMT", "COST", "TGT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_ljung_box_test()
        ```

        Which returns:

        |                     |     WMT |    COST |     TGT |
        |:--------------------|--------:|--------:|--------:|
        | Ljung-Box Statistic | 12.8794 | 10.9863 | 11.9887 |
        | P-Value             |  0.2305 |  0.3586 |  0.2858 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_ljung_box_test(returns, lags=lags)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_variance_ratio_test(
        self,
        period: str | None = None,
        within_period: bool = False,
        q: int = 2,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Lo-MacKinlay Variance Ratio test for the random walk hypothesis.

        The test compares the Variance of `q`-period (overlapping) returns, the sum of `q`
        consecutive returns, to `q` times the Variance of single-period returns. Under the
        random walk hypothesis these should be equal, giving a Variance Ratio of 1. A Variance
        Ratio above 1 with a significant (low) p-value indicates positive autocorrelation
        (momentum/trending behavior), while a Variance Ratio below 1 with a significant p-value
        indicates negative autocorrelation (mean-reversion).

        The result holds two statistics. The homoskedastic z(q) assumes a constant volatility;
        daily returns cluster in volatility, and under such heteroskedasticity z(q) rejects the
        random walk too often. The heteroskedasticity-robust z*(q) of Lo and MacKinlay stays
        valid when volatility clusters and is the safer reading for returns.
        `get_arch_lm_test` shows whether a series has that volatility clustering.

        The formula is as follows:

        - VR(q) = σ²(q) / σ²(1), with σ²(1) = Σ(rₜ - μ)² / (n - 1) and σ²(q) = Σ(rₜ + ... + rₜ₋q₊₁ - q·μ)² / m
          over the overlapping q-period sums, m = q·(n - q + 1)·(1 - q / n) and μ the mean return
          (the unbiased estimators of Lo & MacKinlay), q = 2 by default
        - z(q) = (VR(q) - 1) / √φ(q), with φ(q) = 2·(2q - 1)·(q - 1) / (3q·n)
        - z*(q) = √n·(VR(q) - 1) / √θ(q), with θ(q) = Σ(k=1..q-1) 4·(1 - k / q)²·δ(k) and
          δ(k) = n·Σ(t=k+1..n) (rₜ - μ)²·(rₜ₋ₖ - μ)² / (Σ(t=1..n) (rₜ - μ)²)²
        - P-Value = 2·(1 - Φ(|z|)) for both statistics, with Φ the standard normal CDF

        For more information about the method, see the following paper:

        - Lo, A.W., & MacKinlay, A.C. (1988). "Stock Market Prices Do Not Follow Random
        Walks: Evidence from a Simple Specification Test." Review of Financial Studies,
        1(1), 41-66.

        Also known as: Lo-MacKinlay test, VR test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to False. Note that this requires intraday data
            when period = 'daily', since a single day only nests observations when intraday data was fetched.
            q (int, optional): The number of periods to sum returns over. Defaults to 2.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.

        Returns:
            pd.DataFrame: The Variance Ratio, the homoskedastic test statistic and its p-value,
            and the heteroskedasticity-robust test statistic and its p-value per asset.

        Notes:
        - The method retrieves historical return data based on the specified `period` and runs the
        Variance Ratio test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XLE", "XLF", "XLK"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_variance_ratio_test()
        ```

        Which returns:

        |                                 |    XLE |    XLF |     XLK |
        |:--------------------------------|-------:|-------:|--------:|
        | Variance Ratio                  | 1.0249 | 1.0249 |  0.9611 |
        | Variance Ratio Statistic        | 0.8815 | 0.8805 | -1.3774 |
        | P-Value                         | 0.3781 | 0.3786 |  0.1684 |
        | Robust Variance Ratio Statistic | 0.6193 | 0.6256 | -0.9679 |
        | Robust P-Value                  | 0.5358 | 0.5316 |  0.3331 |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_variance_ratio_test(returns, q=q)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_cusum_test(
        self,
        period: str | None = None,
        within_period: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the CUSUM test for the stability of the mean of returns over time.

        The test fits a constant-mean model to the returns and cumulates the (scaled)
        OLS residuals into a path that behaves like a Brownian Bridge under the null
        hypothesis of a stable mean. A stable mean keeps the path close to zero, while
        a mean shift partway through the series (a structural break, e.g. a regime
        change) drags the path away from zero; the test statistic is the maximum
        absolute value of that path.

        The formula is as follows:

        - êₜ = rₜ - mean(r), the OLS residuals of the constant-mean model rₜ = μ + eₜ
        - CUSUM = max over k of |Σ(t=1..k) êₜ| / (σ̂·√n), with σ̂² = Σêₜ² / (n - 1)
        - P-Value = P(sup|B(s)| > CUSUM) for a Brownian Bridge B (the Kolmogorov distribution), with
          stability rejected when the p-value is below 0.05

        For more information about the method, see the following paper:

        - Ploberger, W., & Kramer, W. (1992). "The CUSUM Test with OLS Residuals."
        Econometrica, 60(2), 271-285.

        Also known as: CUSUM test, CUSUM of OLS residuals test, Ploberger-Kramer test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            within_period (bool, optional): Whether to calculate the test within the specified period or for
            the entire period. Thus whether to look at the test within a specific year (if period = 'yearly')
            or look at the entirety of all years. Defaults to False. Note that this requires intraday data
            when period = 'daily', since a single day only nests observations when intraday data was fetched.
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

        toolkit = Toolkit(
            ["BA", "AIR.PA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # The critical values are the asymptotic Brownian Bridge boundaries, so they
        # are the same constants for every asset.
        toolkit.econometrics.get_cusum_test()
        ```

        Which returns:

        |                       |          BA |      AIR.PA |
        |:----------------------|------------:|------------:|
        | CUSUM Statistic       |    0.685425 |    0.441916 |
        | P-Value               |    0.735342 |    0.989762 |
        | Observations          | 1291        | 1291        |
        | Critical Value 1%     |    1.63     |    1.63     |
        | Critical Value 5%     |    1.36     |    1.36     |
        | Critical Value 10%    |    1.22     |    1.22     |
        | Reject Stability (5%) |    0        |    0        |
        """
        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise ValueError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_cusum_test(returns)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _get_series_column(
        self, period: str, column: str, tickers: list[str] | None = None
    ) -> pd.DataFrame:
        """
        The `column` slice of `period`, memoized -- see `_data_cache` in `__init__`. It
        serves the Toolkit's prices and returns and any data given with `data=` alike.

        Without `tickers` only the dates no series has are dropped: per-series and
        pairwise methods drop their own missing values, so each series keeps its own
        history and a series observed at other dates -- a monthly series next to a daily
        one, an asset listed later, another holiday calendar -- does not cut the others
        down to the shared dates. Methods that combine series date by date (regressions,
        VAR, cointegration systems) pass the `tickers` they use and keep the dates all of
        those series have, so a series outside the model never removes dates.
        """
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        # Deduplicated so a ticker named twice (e.g. as dependent and independent)
        # still selects a single column.
        tickers = list(dict.fromkeys(tickers)) if tickers is not None else None
        key = ("column", period, column, tuple(tickers) if tickers else None)

        if key not in self._data_cache:
            data = self._historical_data[period][column]
            self._data_cache[key] = (
                data.dropna(how="all") if tickers is None else data[tickers].dropna()
            )

        return self._data_cache[key]

    def _get_returns(self, period: str, within_period: bool) -> pd.DataFrame:
        """
        The Return slice the diagnostic tests run on, memoized per (period,
        within_period) pair -- see `_data_cache` in `__init__`. Only the dates no
        series has are dropped: every test runs per series on its own non-missing
        values, so a younger ticker (or a series observed at other dates) does not
        shorten another series' history.
        """
        if not within_period:
            return self._get_series_column(period, "Return")

        key = ("returns", period, within_period)

        if key not in self._data_cache:
            self._data_cache[key] = self._within_historical_data[period][
                "Return"
            ].dropna(how="all")

        return self._data_cache[key]

    def _get_tickers(self, include_benchmark: bool = False) -> list[str]:
        """
        `self._tickers`, optionally with `"Benchmark"` filtered out -- shared by
        every method below that derives its own default ticker set (rather than
        having every relevant ticker passed in explicitly), so `include_benchmark`
        behaves identically everywhere it appears.
        """
        tickers = list(self._tickers)

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

        The formula is as follows:

        - ΔYₜ = α + γ·Yₜ₋₁ + Σ(i=1..p) δᵢ·ΔYₜ₋ᵢ + εₜ, with Yₜ the `column` level (Adj Close by default)
          and regression="c" (the default); "ct" adds a trend β·t and "n" drops α
        - ADF = γ̂ / SE(γ̂), compared with MacKinnon (1994, 2010) critical values and p-values
        - p = the number of lags that minimises the AIC, searched from 0 up to `max_lag`, which defaults to
          min(ceil(12·(n / 100)^(1/4)), (n - 5) // 3)

        For more information about the method, see the following paper:

        - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
        Autoregressive Time Series with a Unit Root." Journal of the American Statistical
        Association, 74(366a), 427-431.

        Also known as: ADF test, unit root test, stationarity test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider. Defaults to
            the Schwert (1989) rule of thumb, capped at (n - 5) // 3 for short samples.
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

        toolkit = Toolkit(
            ["EURUSD", "USDJPY", "GBPUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_augmented_dickey_fuller()
        ```

        Which returns:

        |                       |      EURUSD |     USDJPY |      GBPUSD |
        |:----------------------|------------:|-----------:|------------:|
        | ADF Statistic         |   -1.83568  |   -1.68762 |   -1.81162  |
        | P-Value               |    0.362896 |    0.4375  |    0.374708 |
        | Lags Used             |    5        |    0       |    2        |
        | Observations          | 1345        | 1351       | 1349        |
        | Critical Value 1%     |   -3.43522  |   -3.4352  |   -3.43521  |
        | Critical Value 5%     |   -2.86369  |   -2.86368 |   -2.86368  |
        | Critical Value 10%    |   -2.56792  |   -2.56791 |   -2.56791  |
        | Reject Unit Root (5%) |    0        |    0       |    0        |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_augmented_dickey_fuller(
                    prices[ticker], max_lag=max_lag, regression=regression
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        The formula is as follows:

        - Yₜ = α + eₜ fitted by OLS on the `column` level (Adj Close by default), with regression="c"
          (the default); "ct" adds a trend β·t
        - Sₜ = Σ(i=1..t) êᵢ
        - KPSS = Σ(t=1..n) Sₜ² / (n²·λ²)
        - λ² = (1 / n)·Σêₜ² + (2 / n)·Σ(s=1..l) (1 - s / (l + 1))·Σ(t=s+1..n) êₜ·êₜ₋ₛ, with l = `lags` or,
          by default, the Hobijn, Franses and Ooms (2004) automatic bandwidth
        - The p-value is interpolated from the KPSS (1992) critical value table, so it is bounded
          between 0.01 and 0.10

        For more information about the method, see the following paper:

        - Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). "Testing the Null
        Hypothesis of Stationarity against the Alternative of a Unit Root." Journal of
        Econometrics, 54(1-3), 159-178.

        Also known as: KPSS test, stationarity test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            regression (str, optional): Which deterministic term to remove before testing, one of "c"
            (constant, level-stationarity) or "ct" (constant and trend, trend-stationarity).
            Defaults to "c".
            lags (int, optional): The truncation lag for the long-run variance estimate. Defaults
            to `statsmodels`' automatic (Hobijn, Franses & Ooms, 2004) bandwidth selection.
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

        toolkit = Toolkit(
            ["^GSPC", "^N225"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_kpss_test()
        ```

        Which returns:

        |                          |      ^GSPC |      ^N225 |
        |:-------------------------|-----------:|-----------:|
        | KPSS Statistic           |    4.73448 |    4.79639 |
        | P-Value                  |    0.01    |    0.01    |
        | Lags Used                |   21       |   21       |
        | Observations             | 1299       | 1298       |
        | Critical Value 1%        |    0.739   |    0.739   |
        | Critical Value 2.5%      |    0.574   |    0.574   |
        | Critical Value 5%        |    0.463   |    0.463   |
        | Critical Value 10%       |    0.347   |    0.347   |
        | Reject Stationarity (5%) |    1       |    1       |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_kpss_test(
                    prices[ticker], regression=regression, lags=lags
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        The formula is as follows:

        - Yₜ = α + ρ·Yₜ₋₁ + uₜ fitted by OLS on the `column` level (Adj Close by default), with
          regression="c" (the default); "ct" adds a trend β·t
        - t_ρ = (ρ̂ - 1) / SE(ρ̂), γ₀ = Σûₜ² / n and s² = Σûₜ² / (n - k), with k the number of coefficients
        - λ² = γ₀ + 2·Σ(j=1..l) (1 - j / (l + 1))·γⱼ, with γⱼ = (1 / n)·Σ(t=j+1..n) ûₜ·ûₜ₋ⱼ and l = `lags`,
          which defaults to int(12·(n / 100)^(1/4))
        - Z_t = √(γ₀ / λ²)·t_ρ - (λ² - γ₀) / (2·λ·s)·n·SE(ρ̂)
        - The unit root is rejected at 5% when Z_t is below the MacKinnon (1996) asymptotic critical value
          (-2.86 for "c", -3.41 for "ct"); no p-value is reported

        For more information about the method, see the following paper:

        - Phillips, P.C.B., & Perron, P. (1988). "Testing for a Unit Root in Time Series
        Regression." Biometrika, 75(2), 335-346.

        Also known as: PP test, Z_t test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            regression (str, optional): Which deterministic term to include, one of "c" (constant) or
            "ct" (constant and trend). Defaults to "c". Note "n" (no constant) is not supported, see
            `econometrics.unitroot_model.get_phillips_perron_test` for why.
            lags (int, optional): The truncation lag for the Newey-West long-run variance estimate.
            Defaults to None, meaning the Schwert (1989) rule ceil(12 * (n / 100) ** 0.25), capped
            for short samples as the ADF lag search is.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The Phillips-Perron Z_t statistic, its MacKinnon (1994) p-value, truncation
            lag used, observations used, the finite-sample MacKinnon (2010) 1%/5%/10% critical values,
            and whether the unit root is rejected at the 5% level, per asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the
        Phillips-Perron test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GCUSD", "SIUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_phillips_perron_test()
        ```

        Which returns:

        |                           |      GCUSD |      SIUSD |
        |:--------------------------|-----------:|-----------:|
        | Phillips-Perron Statistic |    3.39009 |    4.5285  |
        | P-Value                   |    1       |    1       |
        | Lags Used                 |   23       |   23       |
        | Observations              | 1300       | 1300       |
        | Critical Value 1%         |   -3.43539 |   -3.43539 |
        | Critical Value 5%         |   -2.86377 |   -2.86377 |
        | Critical Value 10%        |   -2.56796 |   -2.56796 |
        | Reject Unit Root (5%)     |    0       |    0       |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_phillips_perron_test(
                    prices[ticker], regression=regression, lags=lags
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        The formula is as follows:

        - ΔYₜ = μ + θ·DUₜ + β·t + α·Yₜ₋₁ + Σ(i=1..p) cᵢ·ΔYₜ₋ᵢ + εₜ, with Yₜ the `column` level (Adj Close
          by default) and DUₜ = 1 after the break date T_B and 0 before it (regression="c", the default)
        - regression="t" replaces θ·DUₜ by a trend break γ·DTₜ, with DTₜ = t - T_B after the break and 0
          before it, and regression="ct" includes both
        - ZA = min over T_B of α̂ / SE(α̂), searching every break date outside the first and last `trim`
          (15% by default) of the sample
        - p = the number of lags that minimises the AIC, up to `max_lag` (same default as
          `get_augmented_dickey_fuller`)

        For more information about the method, see the following papers:

        - Zivot, E., & Andrews, D.W.K. (1992). "Further Evidence on the Great Crash,
        the Oil-Price Shock, and the Unit-Root Hypothesis." Journal of Business &
        Economic Statistics, 10(3), 251-270.
        - Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
        Hypothesis." Econometrica, 57(6), 1361-1401.

        Also known as: ZA test, structural break unit root test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider when
            selecting the (single, reused) lag length. Defaults to the Schwert (1989) rule of thumb,
            capped at (n - 5) // 3 for short samples.
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

        toolkit = Toolkit(
            ["CLUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_zivot_andrews_test()
        ```

        Which returns:

        |                         |       CLUSD |
        |:------------------------|------------:|
        | Zivot-Andrews Statistic |   -4.1369   |
        | P-Value                 |    0.271885 |
        | Break Index             |  243        |
        | Observations            | 1280        |
        | Lags Used               |   10        |
        | Critical Value 1%       |   -5.27644  |
        | Critical Value 5%       |   -4.81067  |
        | Critical Value 10%      |   -4.56618  |
        | Reject Unit Root (5%)   |    0        |
        """
        period = period if period else "daily"
        prices = self._filter_benchmark(
            self._get_series_column(period, column),
            include_benchmark=include_benchmark,
        )

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_zivot_andrews_test(
                    prices[ticker], max_lag=max_lag, regression=regression, trim=trim
                )
                for ticker in prices.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        In pairs trading the slope of the regression of the dependent on the independent
        ticker is the hedge ratio, and its residual is the spread that is traded.
        Cointegration found in one window often breaks down in the next, so the relationship
        is usually tested again on a later sample before it is relied on. The result depends
        on which ticker is the dependent one, so the two orderings can disagree;
        `get_johansen_cointegration` tests the pair without that choice.

        The formula is as follows:

        - Step 1: Aₜ = α + β·Bₜ + uₜ, the OLS regression of the dependent on the independent `column`
          level (Adj Close by default)
        - Step 2: Δûₜ = γ·ûₜ₋₁ + Σ(i=1..p) δᵢ·Δûₜ₋ᵢ + εₜ, an ADF regression without constant on the residuals
        - EG = γ̂ / SE(γ̂), compared with the MacKinnon (1994, 2010) cointegration critical values and
          p-values for two variables
        - p = the number of lags that minimises the AIC, up to `max_lag` (same default as
          `get_augmented_dickey_fuller`)

        For more information about the method, see the following paper:

        - Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
        Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

        Also known as: EG test, residual-based cointegration test, pairs-trading test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider in the
            underlying ADF test on the residuals. Defaults to the Schwert (1989) rule of thumb,
            capped at (n - 5) // 3 for short samples.
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

        toolkit = Toolkit(
            ["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        engle_granger = toolkit.econometrics.get_engle_granger_cointegration()

        # The 1% and 10% critical values follow the same pattern as the 5% one.
        engle_granger[["Engle-Granger Statistic", "P-Value", "Critical Value 5%", "Cointegrated (5%)"]]
        ```

        Which returns:

        |               |   Engle-Granger Statistic |   P-Value |   Critical Value 5% | Cointegrated (5%)   |
        |:--------------|--------------------------:|----------:|--------------------:|:--------------------|
        | ('KO', 'PEP') |                   -1.0266 |    0.8977 |              -3.341 | False               |
        | ('PEP', 'KO') |                   -2.0565 |    0.4991 |              -3.341 | False               |
        """
        period = period if period else "daily"
        prices = self._get_series_column(period, column)

        pairs = self._all_ordered_pairs(include_benchmark=include_benchmark)
        logger.info("Computing %d dependent/independent ticker pairs", len(pairs))

        rows = {
            (ticker_a, ticker_b): cointegration_model.get_engle_granger_cointegration(
                prices[ticker_a], prices[ticker_b], max_lag=max_lag
            )
            for ticker_a, ticker_b in pairs
        }

        # Transposing rows that mix numbers and a boolean leaves object columns, which
        # pandas does not round, so the numeric columns are restored first.
        result = pd.DataFrame(rows).T.infer_objects()
        result.index = result.index.set_names(["Dependent", "Independent"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        The formula is as follows:

        - ΔYₜ = Π·Yₜ₋₁ + Σ(i=1..k) Γᵢ·ΔYₜ₋ᵢ + μ + εₜ, with Yₜ the vector of `column` levels (Adj Close by
          default), k = `k_ar_diff` (1 by default) and the deterministic term μ set by `det_order`: none
          (-1, Johansen's case 1), an unrestricted constant μ₀ (0, the default, case 3) or a linear trend
          (1: the levels are linearly detrended before the test, evaluated against the case 5 critical
          values)
        - λ₁ ≥ ... ≥ λ_N: the eigenvalues of the reduced-rank regression of ΔYₜ on Yₜ₋₁, both corrected for
          the lagged differences and μ
        - Trace(r) = -T·Σ(i=r+1..N) ln(1 - λᵢ)
        - Max-Eigenvalue(r) = -T·ln(1 - λᵣ₊₁)
        - Each statistic rejects "rank ≤ r" at the 5% level when it exceeds its 95% critical value

        For more information about the method, see the following papers:

        - Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." Journal of
        Economic Dynamics and Control, 12(2-3), 231-254.
        - Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in
        Gaussian Vector Autoregressive Models." Econometrica, 59(6), 1551-1580.
        - MacKinnon, J.G., Haug, A.A., & Michelis, L. (1999). "Numerical Distribution
        Functions of Likelihood Ratio Tests for Cointegration." Journal of Applied
        Econometrics, 14(5), 563-577.

        Also known as: Johansen test, Johansen procedure, VECM rank test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            det_order (int, optional): Which deterministic term to include: -1 (none,
            Johansen's case 1), 0 (an unrestricted constant, case 3) or 1 (a linear trend,
            evaluated against the case 5 critical values). The critical values are those of
            MacKinnon, Haug & Michelis (1999). Defaults to 0.
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

        toolkit = Toolkit(
            ["V", "MA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        johansen = toolkit.econometrics.get_johansen_cointegration()

        # The maximum eigenvalue statistic columns follow the same pattern as the trace ones.
        johansen[["Eigenvalue", "Trace Statistic", "Trace Critical Value 95%", "Reject (Trace, 5%)"]]
        ```

        Which returns:

        |        |   Eigenvalue |   Trace Statistic |   Trace Critical Value 95% | Reject (Trace, 5%)   |
        |:-------|-------------:|------------------:|---------------------------:|:---------------------|
        | r <= 0 |       0.0107 |           13.9015 |                    15.4943 | False                |
        | r <= 1 |       0.0003 |            0.3612 |                     3.8415 | False                |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        prices = self._get_series_column(period, column, tickers)

        result = cointegration_model.get_johansen_cointegration(
            prices[tickers], det_order=det_order, k_ar_diff=k_ar_diff
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        On daily data a significant result can reflect trading hours rather than
        information: of two markets that close at different times, the one that closes
        later already contains news the other only prices the next day, so it appears to
        predict it. The result also depends on `max_lag`, so it is usually checked at a few
        lag lengths, and across many pairs some significant results arise by chance alone.

        The formula is as follows:

        - Restricted: Aₜ = c + Σ(i=1..p) aᵢ·Aₜ₋ᵢ + eₜ
        - Unrestricted: Aₜ = c + Σ(i=1..p) aᵢ·Aₜ₋ᵢ + Σ(i=1..p) bᵢ·Bₜ₋ᵢ + uₜ
        - F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - 2p - 1)), F-distributed with (p, T - 2p - 1) degrees
          of freedom
        - A is the dependent and B the independent ticker's `column` (Return by default), p = `max_lag`
          (5 by default) and T the number of observations after the lags

        For more information about the method, see the following paper:

        - Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
        and Cross-Spectral Methods." Econometrica, 37(3), 424-438.

        Also known as: Granger causality test, predictive causality, lead-lag test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["CLUSD", "XLE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_granger_causality()
        ```

        Which returns:

        |                  |   F-Statistic |   P-Value | Granger-Causes (5%)   |
        |:-----------------|--------------:|----------:|:----------------------|
        | ('CLUSD', 'XLE') |        4.2749 |    0.0007 | True                  |
        | ('XLE', 'CLUSD') |        1.5533 |    0.1705 | False                 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        pairs = self._all_ordered_pairs(include_benchmark=include_benchmark)
        logger.info("Computing %d dependent/independent ticker pairs", len(pairs))

        rows = {
            (ticker_a, ticker_b): causality_model.get_granger_causality(
                returns[ticker_a], returns[ticker_b], max_lag=max_lag
            )
            for ticker_a, ticker_b in pairs
        }

        # Transposing rows that mix numbers and a boolean leaves object columns, which
        # pandas does not round, so the numeric columns are restored first.
        result = pd.DataFrame(rows).T.infer_objects()
        result.index = result.index.set_names(["Dependent", "Independent"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        The squared return is a noisy stand-in for the true Variance. Patton (2011) shows
        that the squared-error loss still ranks two volatility forecasts correctly with such
        a proxy, while the absolute-error loss can rank them the wrong way round, so
        `loss="squared"` is the safer choice. A negative statistic favors `method_a`.

        The formula is as follows:

        - Realized Variance: rₜ²
        - EWMA forecast ("ewma"): σₜ² = λ·σₜ₋₁² + (1 - λ)·rₜ₋₁², with λ = `lambda_` (0.94 by default), so
          the forecast for t uses returns up to t - 1
        - Rolling forecast ("rolling"): the sample variance (ddof=1) of the `window_size` (22 by default)
          returns up to t - 1
        - dₜ = L(rₜ² - f_a,ₜ) - L(rₜ² - f_b,ₜ), with L(e) = e² (loss="squared") or |e| (loss="absolute")
        - DM = mean(d) / √(γ₀ / n)·√((n - 1) / n), with γ₀ = (1 / n)·Σ(dₜ - mean(d))² and the Harvey,
          Leybourne and Newbold (1997) correction for a one-step horizon
        - P-Value = 2·P(T > |DM|) for a Student-t distribution with n - 1 degrees of freedom

        For more information about the method, see the following papers:

        - Diebold, F.X., & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
        Journal of Business & Economic Statistics, 13(3), 253-263.
        - Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility
        Proxies." Journal of Econometrics, 160(1), 246-256.

        Also known as: DM test, forecast comparison test.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
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
        - Both forecasts for period t use only the returns up to, and including, period t - 1, so this
        is a genuinely out-of-sample forecast comparison rather than an in-sample fit comparison.
        - Each asset uses its own return history, so an asset with a shorter history does not shorten
        the comparison of the others.
        - Like `Risk.get_var_backtest`, this is calculated over the full return history rather than the
        `within_period` slices used elsewhere in this module, since the rolling/EWMA forecasts need more
        history than a single sub-period provides.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "SHEL", "BP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_diebold_mariano_test(method_a="ewma", method_b="rolling")
        ```

        Which returns:

        |                           |       XOM |       CVX |      SHEL |        BP |
        |:--------------------------|----------:|----------:|----------:|----------:|
        | Diebold-Mariano Statistic |   -3.9115 |   -4.9425 |   -1.8262 |   -1.3358 |
        | P-Value                   |    0.0001 |    0      |    0.0681 |    0.1818 |
        | Mean Loss Differential    |    0      |    0      |    0      |    0      |
        | Observations              | 1232      | 1232      | 1232      | 1232      |
        """
        if method_a not in ("ewma", "rolling") or method_b not in ("ewma", "rolling"):
            raise ValueError("method_a and method_b must be 'ewma' or 'rolling'.")

        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        returns = self._filter_benchmark(
            self._historical_data[period]["Return"].dropna(how="all"),
            include_benchmark=include_benchmark,
        )

        actual = returns**2
        forecast_a = forecast_evaluation_model.get_volatility_forecast(
            returns, method_a, window_size, lambda_
        )
        forecast_b = forecast_evaluation_model.get_volatility_forecast(
            returns, method_b, window_size, lambda_
        )

        result = forecast_evaluation_model.get_diebold_mariano_test(
            actual, forecast_a, forecast_b, loss=loss, horizon=1
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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
        maxlags: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit an Ordinary Least Squares (OLS) regression of `dependent_ticker` on
        `independent_tickers`, using the selected `column` (returns by default) of each ticker.

        OLS estimates the coefficients that minimise the sum of squared residuals, i.e. the
        line (or plane) that best explains the dependent ticker from the independent tickers.

        The formula is as follows:

        - β̂ = (X'X)⁻¹X'y, with y the dependent ticker's `column` (Return by default) and X the
          independent tickers' `column` plus a constant column (add_constant=True)
        - Cov(β̂) = σ̂²·(X'X)⁻¹, with σ̂² = e'e / (n - k) (cov_type="nonrobust"); "HC0" to "HC3",
          "cluster" and "HAC" (Newey-West with `maxlags`) use statsmodels' sandwich estimators
        - t = β̂ⱼ / SE(β̂ⱼ), with p-values from a Student-t distribution with n - k degrees of freedom
        - R² = 1 - e'e / Σ(yₜ - mean(y))²

        Each coefficient is the expected change in `dependent_ticker` for a one-unit change in
        that independent ticker, holding the others fixed, and the intercept is what remains
        when all of them are zero. With returns this is the familiar way to estimate how
        strongly one asset moves with others (e.g. a beta against an index).

        Under the classical assumptions (linearity, no perfect multicollinearity, homoskedastic
        and uncorrelated errors) OLS is the Best Linear Unbiased Estimator (Gauss-Markov). The
        coefficients stay unbiased when the errors are heteroskedastic, but the classical
        standard errors do not: check with `get_breusch_pagan_test` or `get_white_test` and,
        if needed, refit with a robust `cov_type` such as "HC1" (or "HC3" for small samples),
        or "HAC" when the errors are also autocorrelated.

        For more information about the HAC covariance, see the following paper:

        - Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite,
        Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica,
        55(3), 703-708.

        Also known as: linear regression, least squares regression.

        Args:
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            cov_type (str, optional): Which covariance estimator to use for the standard errors --
            one of "nonrobust" (classical, assumes homoskedastic errors), "HC0"/"HC1"/"HC2"/"HC3"
            (heteroskedasticity-robust), "cluster" (cluster-robust, requires `clusters`) or "HAC"
            (Newey-West, heteroskedasticity-and-autocorrelation-consistent, uses `maxlags`).
            Use `get_breusch_pagan_test`/`get_white_test` to check for heteroskedasticity and
            `get_ljung_box_test` to check for autocorrelation first. Defaults to "nonrobust".
            clusters (pd.Series | None, optional): The cluster label for each observation
            (e.g. a coarser time bucket derived from the return index, to correct for
            within-period correlation), required when `cov_type="cluster"`. Aligned to the
            regression's own index before use, so it may be indexed by the full period
            index even though the regression drops periods with missing data. Defaults
            to None.
            maxlags (int | None, optional): The maximum lag to include when estimating the
            HAC (Newey-West) covariance matrix, only used when `cov_type="HAC"`. Defaults to
            None, meaning the Newey & West (1994) rule of thumb `floor(4 * (n / 100)^(2/9))`
            with `n` the number of observations.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "USO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # XOM (the first ticker) is dependent; USO and the Benchmark are independent.
        # dependent_ticker and independent_tickers choose others, cov_type='HC1' gives
        # heteroskedasticity-robust and cov_type='HAC' (with maxlags) Newey-West standard errors.
        toolkit.econometrics.get_ols(include_benchmark=True)
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0005 |       0.0004 |        1.5047 |    0.1327 |
        | USO       |        0.4878 |       0.0172 |       28.4214 |    0      |
        | Benchmark |        0.366  |       0.0333 |       10.9991 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        # Positionally aligned by statsmodels, so it has to be reindexed onto the regression's own index first -- `returns` has already dropped any period with missing data, which a caller-supplied cluster label will not have.  # noqa: E501
        aligned_clusters = (
            clusters.reindex(returns.index) if clusters is not None else None
        )

        result = regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
            cov_type=cov_type,
            clusters=aligned_clusters,
            maxlags=maxlags,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_wls(
        self,
        weights: pd.Series | str = "feasible",
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        cov_type: str = "nonrobust",
        clusters: pd.Series | None = None,
        maxlags: int | None = None,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Weighted Least Squares (WLS) regression of `dependent_ticker` on
        `independent_tickers`.

        When the error variance differs across observations (heteroskedasticity) with a known
        or estimable structure, `Var(e_i) = sigma^2 / w_i`, OLS remains unbiased but is no
        longer the most efficient estimator. WLS restores efficiency by minimising the
        weighted sum of squared residuals, so noisier observations (low weight) count less
        than precise ones (high weight).

        The formula is as follows:

        - β̂ = (X'WX)⁻¹X'Wy, with W = diag(weights), the same as OLS of √wₜ·yₜ on √wₜ·xₜ
        - Cov(β̂) = σ̂²·(X'WX)⁻¹, with σ̂² = Σwₜ·eₜ² / (n - k) (cov_type="nonrobust")
        - y is the dependent ticker's `column` (Return by default) and X the independent tickers'
          `column` plus a constant, on the dates where the returns and `weights` all have data

        With equal weights WLS reduces to OLS. A typical use is down-weighting periods of
        high volatility, so that a few turbulent periods do not dominate the estimated
        relationship. By default (weights="feasible") the weights are estimated from the
        regression itself with the feasible GLS procedure of Wooldridge (2016): the
        logarithm of the squared OLS residuals is regressed on the independent tickers, and
        each period's weight is 1 / exp(fitted value), the inverse of its estimated error
        variance. Use `get_breusch_pagan_test` or `get_white_test` to establish whether
        heteroskedasticity is present before choosing WLS over OLS.

        For more information about the feasible weights, see the following book:

        - Wooldridge, J.M. (2016). "Introductory Econometrics: A Modern Approach", 6th
        edition, Section 8.4. Cengage Learning.

        Also known as: weighted regression, feasible WLS.

        Args:
            weights (pd.Series | str, optional): The (positive) weight of each observation,
            aligned to the same period index as the return data (e.g. `1 / rolling_variance`),
            or "feasible" to estimate them from the regression's own residuals with the
            feasible GLS procedure of Wooldridge (2016). Defaults to "feasible".
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept. Defaults to True.
            cov_type (str, optional): Which covariance estimator to use, applied to the weighted/
            transformed problem -- see `get_ols`'s `cov_type` for the full list of options.
            Defaults to "nonrobust".
            clusters (pd.Series | None, optional): The cluster label for each observation,
            required when `cov_type="cluster"`. Defaults to None.
            maxlags (int | None, optional): The maximum lag to include when estimating the
            HAC (Newey-West) covariance matrix, only used when `cov_type="HAC"`. Defaults to
            None, meaning the Newey & West (1994) rule of thumb (see `get_ols`).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error, t-Statistic, P-Value),
            indexed by regressor name.

        As an example:

        ```python
        import pandas as pd
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        returns = toolkit.econometrics._get_series_column("weekly", "Return", ["AAPL", "MSFT"])
        weights = pd.Series(1.0, index=returns.index)

        # AAPL (the first ticker) is dependent, MSFT is independent
        toolkit.econometrics.get_wls(weights, period="weekly")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0012 |       0.0019 |        0.6328 |    0.5274 |
        | MSFT      |        0.6283 |       0.058  |       10.8323 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        estimate_weights = isinstance(weights, str)
        if estimate_weights and weights != "feasible":
            raise ValueError(
                f"weights must be a Series of weights or 'feasible', received '{weights}'."
            )

        to_align = [returns[dependent_ticker], returns[independent_tickers]]
        if not estimate_weights:
            to_align.append(weights.rename("__weights__"))
        if clusters is not None:
            to_align.append(clusters.rename("__clusters__"))

        aligned = pd.concat(to_align, axis=1, join="inner").dropna()
        aligned_clusters = aligned.pop("__clusters__") if clusters is not None else None
        # Estimated weights come from the same observations as the regression they weight.
        aligned_weights = (
            regression_model.get_feasible_weights(
                aligned.iloc[:, 0], aligned[independent_tickers], add_constant
            )
            if estimate_weights
            else aligned.pop("__weights__")
        )

        result = regression_model.get_wls(
            aligned.iloc[:, 0],
            aligned[independent_tickers],
            aligned_weights,
            add_constant=add_constant,
            cov_type=cov_type,
            clusters=aligned_clusters,
            maxlags=maxlags,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_gls(
        self,
        omega: pd.DataFrame | str = "ar1",
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

        GLS generalises WLS to an arbitrary error covariance matrix `Var(e) = sigma^2 * Omega`.
        Where WLS only corrects for heteroskedasticity (a diagonal `Omega`), GLS also
        corrects for correlation between the errors of different observations, such as the
        autocorrelation that is common in financial time series (a non-diagonal `Omega`).

        The formula is as follows:

        - β̂ = (X'Ω⁻¹X)⁻¹X'Ω⁻¹y
        - Cov(β̂) = σ̂²·(X'Ω⁻¹X)⁻¹, with σ̂² = e'Ω⁻¹e / (n - k)
        - y is the dependent ticker's `column` (Return by default) and X the independent tickers'
          `column` plus a constant (add_constant=True)

        Under a correctly specified `Omega` this is the most efficient linear unbiased
        estimator. By default (omega="ar1") `Omega` is estimated from the regression itself
        as an AR(1) error structure, eₜ = ρ·eₜ₋₁ + uₜ: ρ is the slope of the OLS residuals on
        their own lag and Ωᵢⱼ = ρ^|i - j|, which makes this the Prais-Winsten estimator. Any
        other structure can be supplied as `omega`. `get_durbin_watson_test` helps establish
        whether the residuals are autocorrelated in the first place.

        For more information about the method, see the following paper:

        - Aitken, A.C. (1935). "On Least Squares and Linear Combinations of Observations."
        Proceedings of the Royal Society of Edinburgh, 55, 42-48.
        - Prais, S.J. and Winsten, C.B. (1954). "Trend Estimators and Serial Correlation."
        Cowles Commission Discussion Paper No. 383.

        Also known as: GLS, Prais-Winsten.

        Args:
            omega (pd.DataFrame | str, optional): The (symmetric, positive-definite) error
            covariance structure, up to a scalar, shape `(n, n)`, or "ar1" to estimate an
            AR(1) structure from the regression's own residuals (Prais-Winsten). Defaults
            to "ar1".
            dependent_ticker (str | None, optional): The dependent (predicted) asset.
            Defaults to None, meaning the Toolkit instance's first ticker.
            independent_tickers (str | list[str] | None, optional): The independent
            (predictor) asset(s). Defaults to None, meaning every other ticker in
            the Toolkit instance besides `dependent_ticker`.
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default independent ticker(s) (has no effect when independent_tickers is given
            explicitly). Defaults to False.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        returns = toolkit.econometrics._get_series_column("weekly", "Return", ["AAPL", "MSFT"])
        n = len(returns)
        omega = pd.DataFrame(np.eye(n))

        # AAPL (the first ticker) is dependent; MSFT is independent
        toolkit.econometrics.get_gls(omega, period="weekly")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0012 |       0.0019 |        0.6328 |    0.5274 |
        | MSFT      |        0.6283 |       0.058  |       10.8323 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        if isinstance(omega, str):
            if omega != "ar1":
                raise ValueError(
                    f"omega must be a DataFrame or 'ar1', received '{omega}'."
                )
            omega_values = regression_model.get_ar1_covariance(
                returns[dependent_ticker], returns[independent_tickers], add_constant
            )
        else:
            omega_values = omega.to_numpy()

        result = regression_model.get_gls(
            returns[dependent_ticker],
            returns[independent_tickers],
            omega_values,
            add_constant=add_constant,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
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
        Fit a Logistic Regression (Logit model) of whether `dependent_ticker`'s return is
        positive on `independent_tickers`.

        The dependent variable is turned into a binary outcome (1 when the return is
        positive, 0 otherwise) and the model estimates the probability of that outcome
        through the logistic (sigmoid) function, which maps any value onto the 0 to 1 range.

        The formula is as follows:

        - yₜ = 1 if the dependent ticker's `column` (Return by default) is above 0, else 0
        - P(yₜ = 1 | xₜ) = 1 / (1 + exp(-xₜ'β)), with xₜ the independent tickers' `column` plus a
          constant
        - β̂ = argmax Σ[yₜ·ln(pₜ) + (1 - yₜ)·ln(1 - pₜ)], solved by Newton-Raphson (statsmodels Logit)
        - Pseudo R² = 1 - ln L(β̂) / ln L₀ (McFadden), with L₀ the likelihood of a constant-only model

        The model is fitted by Maximum Likelihood. Coefficients are log-odds: a one-unit
        increase in an independent ticker changes the log-odds of a positive return by that
        coefficient, holding the others fixed, so a positive coefficient means a higher
        probability of a positive return. This answers "does this asset tend to be up when
        the others move this way" rather than "by how much", which is what `get_ols`
        estimates. `get_probit_regression` is the closely related alternative with a normal
        instead of a logistic link.

        The coefficients are per unit of the independent ticker's return, and a unit is a
        return of 100%: a coefficient of 90 on the benchmark means the log-odds of an up
        period rise by about 0.9 for every 1% the benchmark moves. With returns of the same
        period as regressors the model describes co-movement and predicts nothing; a
        forecast needs regressors known in advance, such as the returns of the previous
        period.

        For more information about the method, see the following paper:

        - Berkson, J. (1944). "Application of the Logistic Function to Bio-Assay." Journal of
        the American Statistical Association, 39(227), 357-365.

        Also known as: logit model, logit regression.

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
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # AAPL (the first ticker) is dependent; the Benchmark is independent
        toolkit.econometrics.get_logistic_regression(include_benchmark=True)
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   z-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |       -0.0275 |       0.15   |       -0.1836 |    0.8543 |
        | Benchmark |      164.846  |      25.4127 |        6.4868 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        direction = (returns[dependent_ticker] > 0).astype(float)

        result = regression_model.get_logistic_regression(
            direction, returns[independent_tickers], add_constant=add_constant
        )

        return apply_rounding(
            regression_model.binary_regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
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
        Fit a Probit Regression of whether `dependent_ticker`'s return is positive on
        `independent_tickers`.

        As with `get_logistic_regression`, the dependent variable is turned into a binary
        outcome (1 when the return is positive, 0 otherwise). The probability of that outcome
        is modelled through the standard normal cumulative distribution function instead of
        the logistic function.

        The formula is as follows:

        - yₜ = 1 if the dependent ticker's `column` (Return by default) is above 0, else 0
        - P(yₜ = 1 | xₜ) = Φ(xₜ'β), with xₜ the independent tickers' `column` plus a constant
        - β̂ = argmax Σ[yₜ·ln(pₜ) + (1 - yₜ)·ln(1 - pₜ)], solved by Newton-Raphson (statsmodels Probit)
        - Pseudo R² = 1 - ln L(β̂) / ln L₀ (McFadden), with L₀ the likelihood of a constant-only model

        Where `Phi` is the standard normal CDF. The model is fitted by Maximum Likelihood. A
        positive coefficient means a higher probability of a positive return, but the size of
        a coefficient is not a log-odds ratio as in the Logit model. In practice Probit and
        Logit give very similar fitted probabilities; Probit is the convention in parts of the
        econometrics literature and underlies two-step estimators such as the Heckman
        selection correction.

        For more information about the method, see the following paper:

        - Bliss, C.I. (1934). "The Method of Probits." Science, 79(2037), 38-39.

        Also known as: probit model.

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
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["TSLA", "QQQ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # TSLA (the first ticker) is dependent; QQQ is independent
        toolkit.econometrics.get_probit_regression()
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   z-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |       -0.0848 |       0.0913 |       -0.9294 |    0.3527 |
        | QQQ       |       78.1324 |      11.1444 |        7.0109 |    0      |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        direction = (returns[dependent_ticker] > 0).astype(float)

        result = regression_model.get_probit_regression(
            direction, returns[independent_tickers], add_constant=add_constant
        )

        return apply_rounding(
            regression_model.binary_regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
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
        Fit a Quantile Regression of `dependent_ticker` on `independent_tickers` at quantile
        `tau`.

        Where OLS estimates how the average (conditional mean) of the dependent ticker depends
        on the independent tickers, Quantile Regression estimates how a chosen quantile of it
        does. It minimises an asymmetric ("pinball") loss that is smallest at the conditional
        `tau`-quantile.

        The formula is as follows:

        - β̂(τ) = argmin Σ(τ·max(uₜ, 0) + (1 - τ)·max(-uₜ, 0)), with uₜ = yₜ - xₜ'β and τ = `tau` (0.5
          by default)
        - y is the dependent ticker's `column` (Return by default) and X the independent tickers'
          `column` plus a constant (add_constant=True)
        - SE(β̂): statsmodels' kernel sandwich estimator (Epanechnikov kernel, Hall-Sheather bandwidth),
          or, when `n_bootstrap` > 0, the standard deviation (ddof=1) of β̂ over `n_bootstrap`
          resampled-pairs fits

        With `tau = 0.5` this is the median regression, which is far less sensitive to
        outliers than OLS. Fitting at several quantiles (e.g. 0.05, 0.5 and 0.95) shows how
        the whole distribution of returns responds to the independent tickers, for example
        whether an asset's sensitivity to the market is larger in sell-offs (low quantiles)
        than in normal periods, which is directly relevant for downside risk.

        Comparing the coefficients at a low `tau` with those at `tau=0.5`, or with the
        betas from `get_ols`, shows whether the sensitivity to the market rises in
        sell-offs. The table has no p-values: a coefficient more than about two standard
        errors from zero is significant at the 5% level.

        For more information about the method, see the following paper:

        - Koenker, R., & Bassett, G. (1978). "Regression Quantiles." Econometrica, 46(1),
        33-50.

        Also known as: QR.

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
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["BAC", "KRE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # BAC (the first ticker) is dependent; KRE and the Benchmark are independent.
        # tau=0.05 describes how the worst 5% of days respond.
        toolkit.econometrics.get_quantile_regression(include_benchmark=True, tau=0.05)
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |
        |:----------|--------------:|-------------:|
        | Intercept |       -0.0146 |       0.0006 |
        | KRE       |        0.5271 |       0.0484 |
        | Benchmark |        0.3965 |       0.1133 |
        """
        period = period if period else "daily"
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        result = regression_model.get_quantile_regression(
            returns[dependent_ticker],
            returns[independent_tickers],
            tau=tau,
            add_constant=add_constant,
            n_bootstrap=n_bootstrap,
        )

        return apply_rounding(
            regression_model.quantile_regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_fama_macbeth_regression(
        self,
        factor_tickers: str | list[str] | None = None,
        asset_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Fama-MacBeth (1973) two-pass cross-sectional regression: `asset_tickers` is
        treated as the cross-section of test assets, `factor_tickers` as the risk factor(s)
        whose risk premia are estimated -- the standard procedure for testing whether a
        proposed risk factor is actually priced.

        The method runs in two passes:

        1. Time-series pass (per asset): regress each asset's returns on the factor(s) to
        estimate its factor loadings (betas).
        2. Cross-sectional pass (per period): regress that period's returns of all assets on
        those betas, which gives one estimate of each factor's risk premium per period.

        The formula is as follows:

        - First pass: R_i,t = alpha_i + beta_i' * F_t + e_i,t, for each asset i, over the full sample
        - Second pass: R_i,t = gamma_0,t + gamma_t' * beta_i + u_i,t, for each period t (gamma_0,t only
          with add_constant=True)
        - Risk premium: gamma_hat = mean_t(gamma_t)
        - Standard error: SE(gamma_hat) = std_t(gamma_t) / sqrt(T), with ddof=1
        - t = gamma_hat / SE(gamma_hat), with p-values from a Student-t distribution with T - 1 degrees
          of freedom
        - R are the asset tickers' `column` (Return by default) and F the factor tickers' `column`
          (the Benchmark by default)

        The risk premium is the average of the per-period estimates, and its standard error
        comes from how much those estimates vary over time. That is the method's key
        contribution: it accounts for correlation between the assets' residuals without
        having to model it. A significant positive premium means investors are compensated
        for exposure to the factor. This uses a single full-sample set of betas (the
        textbook variant), not the original rolling-window betas.

        The first-pass betas are estimates, which makes the second-pass standard errors too
        small; the Shanken (1992) correction, not applied here, widens them. The returns are
        not in excess of the risk-free rate, so the intercept is the return of a zero-beta
        asset, which should be close to the risk-free rate if the factors price the assets.

        For more information about the method, see the following papers:

        - Fama, E.F. & MacBeth, J.D. (1973). "Risk, Return, and Equilibrium: Empirical Tests."
        Journal of Political Economy, 81(3), 607-636.
        - Shanken, J. (1992). "On the Estimation of Beta-Pricing Models." Review of Financial
        Studies, 5(1), 1-33.
        - Cochrane, J.H. (2005). "Asset Pricing." Princeton University Press, Chapter 12.

        Also known as: two-pass regression, Fama-MacBeth procedure.

        Args:
            factor_tickers (str | list[str] | None, optional): The ticker(s) whose
            returns are used as the risk factor(s) (e.g. "Benchmark" for a
            single-factor/CAPM-style test). Defaults to None, meaning `["Benchmark"]`.
            asset_tickers (str | list[str] | None, optional): The ticker(s) forming the
            cross-section of test assets. Defaults to None, meaning every Toolkit
            ticker (including "Benchmark") not already used as a factor.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            add_constant (bool, optional): Whether to include an intercept in the
            second-pass cross-sectional regression. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: A coefficient table (Risk Premium, Std. Error, t-Statistic,
            P-Value), indexed by factor name (plus "Intercept" if present).

        Notes:
        - The cross-sectional second pass needs strictly more assets than factors
        (plus an intercept, if `add_constant=True`) to be identified -- with only a
        couple of Toolkit tickers loaded, either pass `add_constant=False` or (better)
        construct the Toolkit with many tickers, since Fama-MacBeth is fundamentally a
        many-asset cross-sectional technique (dozens of assets is typical in practice),
        not a limitation of this method's implementation.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "CAT", "NEE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # The Benchmark (the default factor) is the single risk factor and the ten
        # stocks are the test assets.
        toolkit.econometrics.get_fama_macbeth_regression(period="monthly")
        ```

        Which returns:

        |           |   Risk Premium |   Std. Error |   t-Statistic |   P-Value |
        |:----------|---------------:|-------------:|--------------:|----------:|
        | Intercept |         0.0066 |       0.0058 |        1.1505 |    0.2547 |
        | Benchmark |         0.0101 |       0.008  |        1.2668 |    0.2103 |
        """
        period = period if period else "daily"
        factor_tickers = (
            [factor_tickers]
            if isinstance(factor_tickers, str)
            else (
                list(factor_tickers)
                if factor_tickers is not None
                else [BENCHMARK_TICKER]
            )
        )

        if asset_tickers is None:
            asset_tickers = [
                ticker
                for ticker in self._get_tickers(include_benchmark=True)
                if ticker not in factor_tickers
            ]
        else:
            asset_tickers = (
                [asset_tickers]
                if isinstance(asset_tickers, str)
                else list(asset_tickers)
            )

        returns = self._get_series_column(
            period, column, [*asset_tickers, *factor_tickers]
        )

        result = fama_macbeth_model.get_fama_macbeth_regression(
            returns[asset_tickers],
            returns[factor_tickers],
            add_constant=add_constant,
        )

        return apply_rounding(
            fama_macbeth_model.fama_macbeth_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    def _fit_ols_result(
        self,
        dependent_ticker: str,
        independent_tickers: str | list[str],
        period: str | None,
        column: str,
        add_constant: bool,
    ) -> dict:
        """
        Shared helper for the hypothesis-testing methods below -- fits an OLS model
        via `regression_model.get_ols` and returns the raw (unrounded) regression
        result dict, rather than the rounded coefficient table `get_ols` returns,
        since the F-test, Wald test and Likelihood Ratio test all operate directly
        on the result dict's residuals/coefficients/covariance matrix.
        """
        period = period if period else "daily"
        independent_tickers = (
            [independent_tickers]
            if isinstance(independent_tickers, str)
            else independent_tickers
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
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
        Calculate a two-sample t-test for a difference in mean `column` between every
        unordered pair of tickers in the Toolkit instance.

        The test checks the null hypothesis that two samples come from distributions with
        the same mean, against the two-sided alternative that the means differ. Applied to
        returns it answers whether one asset has had a genuinely different average return
        than another, or whether the difference could be down to chance. By default this is
        Welch's t-test, which does not assume the two tickers have the same variance.

        The formula is as follows:

        - Welch (the default): t = (mean_a - mean_b) / sqrt(s_a^2 / n_a + s_b^2 / n_b)
        - Welch-Satterthwaite: df = (s_a^2 / n_a + s_b^2 / n_b)^2 / ((s_a^2 / n_a)^2 / (n_a - 1)
          + (s_b^2 / n_b)^2 / (n_b - 1))
        - Student (equal_variance=True): t = (mean_a - mean_b) / (s_p·sqrt(1 / n_a + 1 / n_b)), with
          s_p^2 = ((n_a - 1)·s_a^2 + (n_b - 1)·s_b^2) / (n_a + n_b - 2) and df = n_a + n_b - 2
        - s_a^2 and s_b^2 are sample variances (ddof=1) of the two tickers' `column` (Return by
          default), with a two-sided p-value

        Setting `equal_variance=True` pools the two variances instead (Student's t-test), which is
        slightly more powerful but only valid when the variances are truly equal. Welch's
        version is the safer default since assets rarely share the same volatility. A low
        p-value (e.g. below 0.05) means the difference in means is statistically
        significant.

        For more information about the method, see the following papers:

        - Student (1908). "The Probable Error of a Mean." Biometrika, 6(1), 1-25.
        - Welch, B.L. (1947). "The Generalization of 'Student's' Problem when Several
        Different Population Variances are Involved." Biometrika, 34(1/2), 28-35.

        Also known as: independent samples t-test, Welch's t-test (default), Student's
        t-test (`equal_variance=True`).

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["MSFT", "ORCL", "SAP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_two_sample_t_test()
        ```

        Which returns:

        |                  |   T-Statistic |   Degrees of Freedom |   P-Value |   Mean A |   Mean B |
        |:-----------------|--------------:|---------------------:|----------:|---------:|---------:|
        | ('MSFT', 'ORCL') |       -0.528  |              2176.24 |    0.5975 |   0.0008 |   0.0012 |
        | ('MSFT', 'SAP')  |        0.1389 |              2504.05 |    0.8895 |   0.0008 |   0.0007 |
        | ('ORCL', 'SAP')  |        0.6326 |              2210.92 |    0.5271 |   0.0012 |   0.0007 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        rows = {
            (ticker_a, ticker_b): hypothesis_testing_model.get_two_sample_t_test(
                returns[ticker_a], returns[ticker_b], equal_variance=equal_variance
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        # Transposing rows that mix numbers and a boolean leaves object columns, which
        # pandas does not round, so the numeric columns are restored first.
        result = pd.DataFrame(rows).T.infer_objects()
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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
        Calculate a nested-model F-test for the joint significance of the regressors in
        `unrestricted_independent_tickers` that are not already in
        `restricted_independent_tickers`.

        Two OLS regressions of `dependent_ticker` are fitted internally: a "restricted" one on
        `restricted_independent_tickers` and an "unrestricted" one that adds the extra
        tickers. The test checks the null hypothesis that the `q` extra tickers are jointly
        zero, i.e. that adding them does not explain the dependent ticker any better, by
        comparing how much the Residual Sum of Squares (RSS) drops when they are added.

        The formula is as follows:

        - F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - k_unrestricted))
        - Both models are OLS regressions of the dependent ticker's `column` (Return by default),
          with a constant when add_constant=True

        Where `q = k_unrestricted - k_restricted` is the number of added regressors, `n` the
        number of observations and `k_unrestricted` the number of parameters of the larger
        model. Under the null, `F` follows an F(q, n - k_unrestricted) distribution. A low
        p-value means the extra tickers meaningfully improve the fit as a group, even when
        none of them is significant individually. `get_likelihood_ratio_test` is the
        asymptotic Maximum Likelihood counterpart.

        Also known as: nested F-test, restricted vs. unrestricted F-test, partial F-test.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            restricted_independent_tickers (str | list[str]): The independent asset(s) in the
            restricted (smaller) model.
            unrestricted_independent_tickers (str | list[str]): The independent asset(s) in the
            unrestricted (larger) model -- must be a superset of `restricted_independent_tickers`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_f_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        |                          |       0 |
        |:-------------------------|--------:|
        | F-Statistic              | 104.748 |
        | Df Numerator             |   1     |
        | Df Denominator           | 257     |
        | P-Value                  |   0     |
        | Reject Restrictions (5%) |   1     |
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

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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
        Calculate a nested-model Likelihood Ratio (LR) test, the Maximum Likelihood analogue
        of `get_f_test`, for the joint significance of the regressors in
        `unrestricted_independent_tickers` that are not already in
        `restricted_independent_tickers`.

        As with `get_f_test`, a "restricted" and an "unrestricted" OLS regression of
        `dependent_ticker` are fitted internally and the null hypothesis is that the `q` extra
        tickers are jointly zero. Instead of an F-statistic, the two models are compared
        through the ratio of their likelihoods, which for linear models with normal errors
        reduces to a function of the two Residual Sums of Squares (RSS).

        The formula is as follows:

        - LR = -2·(ln L_restricted - ln L_unrestricted) = n * ln(RSS_restricted / RSS_unrestricted)
        - Both models are OLS regressions of the dependent ticker's `column` (Return by default),
          with a constant when add_constant=True

        Under the null this is asymptotically chi-squared distributed with `q` degrees of
        freedom. Where the F-test is exact in small samples under the classical assumptions,
        the LR test relies on a large-sample approximation: both agree for long return
        histories and can differ somewhat for short ones. A low p-value means the extra
        tickers meaningfully improve the model.

        For more information about the method, see the following paper:

        - Wilks, S.S. (1938). "The Large-Sample Distribution of the Likelihood Ratio for
        Testing Composite Hypotheses." Annals of Mathematical Statistics, 9(1), 60-62.

        Also known as: LR test, Wilks' likelihood ratio test.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            restricted_independent_tickers (str | list[str]): The independent asset(s) in the
            restricted (smaller) model.
            unrestricted_independent_tickers (str | list[str]): The independent asset(s) in the
            unrestricted (larger) model -- must be a superset of `restricted_independent_tickers`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_likelihood_ratio_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
        ```

        Which returns:

        |                          |       0 |
        |:-------------------------|--------:|
        | LR Statistic             | 88.8863 |
        | Degrees of Freedom       |  1      |
        | P-Value                  |  0      |
        | Reject Restrictions (5%) |  1      |
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

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_wald_test(
        self,
        restriction_matrix: pd.DataFrame | np.ndarray | list[list[float]],
        dependent_ticker: str | None = None,
        independent_tickers: str | list[str] | None = None,
        include_benchmark: bool = False,
        restriction_values: pd.Series | np.ndarray | list[float] | None = None,
        period: str | None = None,
        column: str = "Return",
        add_constant: bool = True,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate a Wald test of `q` general linear restriction(s) on the coefficients of an
        OLS regression of `dependent_ticker` on `independent_tickers`.

        A single OLS regression is fitted internally and the test checks the null hypothesis
        `H0: R @ beta = r`, where `R` (`restriction_matrix`) encodes `q` linear restrictions
        on the coefficients and `r` (`restriction_values`) holds their hypothesised values
        (zero by default). This generalises the t-test reported for each coefficient to
        combinations of coefficients, for example that two coefficients are jointly zero,
        that a coefficient equals a specific value (such as a beta of exactly 1), or that two
        tickers have the same coefficient.

        The formula is as follows:

        - W = (R @ beta_hat - r)' @ (R @ Cov(beta_hat) @ R')^-1 @ (R @ beta_hat - r)
        - F = W / q, F-distributed with (q, n - k) degrees of freedom
        - beta_hat and the classical Cov(beta_hat) = σ̂²·(X'X)⁻¹ come from the OLS regression of the
          dependent ticker's `column` (Return by default), with a constant when add_constant=True

        Which is asymptotically chi-squared distributed with `q` degrees of freedom. `W / q`
        is reported alongside it as an F-statistic for small samples; for a single
        restriction it equals the squared t-statistic of that coefficient. A low p-value
        means the data reject the restriction(s).

        For more information about the method, see the following paper:

        - Wald, A. (1943). "Tests of Statistical Hypotheses Concerning Several Parameters When
        the Number of Observations is Large." Transactions of the American Mathematical
        Society, 54(3), 426-482.

        Also known as: Wald chi-squared test.

        Args:
            restriction_matrix (pd.DataFrame | np.ndarray | list[list[float]]): The `(q, k)`
            restriction matrix `R`, one row per restriction, one column per coefficient in the
            same order as `add_constant` (if True, "Intercept" first) followed by the
            independent ticker(s), e.g. [[0, 1, 0], [0, 0, 1]].
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
            restriction_values (pd.Series | np.ndarray | list[float] | None, optional): The
            length-`q` vector of hypothesized values. Defaults to None, i.e. all restrictions
            equal zero.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "daily".
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

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # H0: the coefficients on both MSFT and Benchmark are jointly zero.
        toolkit.econometrics.get_wald_test(
            restriction_matrix=[[0, 1, 0], [0, 0, 1]],
            independent_tickers=["MSFT", "Benchmark"],
            period="weekly",
        )
        ```

        Which returns:

        |                          |       0 |
        |:-------------------------|--------:|
        | Wald Statistic (Chi2)    | 269.191 |
        | Chi2 P-Value             |   0     |
        | F-Statistic              | 134.595 |
        | F P-Value                |   0     |
        | Restrictions (q)         |   2     |
        | Reject Restrictions (5%) |   1     |
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

        return apply_rounding(
            wald_result, rounding if rounding is not None else self._rounding
        )

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
        Calculate a regression-based Hausman-Wu test for the endogeneity of `suspect_ticker`
        in a regression of `dependent_ticker` on `suspect_ticker` (and, optionally,
        `other_independent_tickers`), using `instrument_tickers` as instruments for
        `suspect_ticker`.

        OLS requires every regressor to be uncorrelated with the error term. When that fails,
        for example because of an omitted common driver, reverse causality or measurement
        error, the regressor is "endogenous" and OLS is biased; an instrumental-variables
        approach such as `get_iv_2sls` is then needed. This test checks whether that is the
        case before committing to a full IV fit, using two auxiliary OLS regressions:

        1. Regress `suspect_ticker` on the instruments (and the other tickers, if given) and
        keep the residuals: the part of `suspect_ticker` that the instruments do not explain.
        2. Regress `dependent_ticker` on `suspect_ticker`, those residuals (and the other
        tickers) and test whether the coefficient on the residuals is zero.

        If `suspect_ticker` is exogenous the residuals add no information, so their coefficient
        should be statistically zero. A low p-value is evidence that `suspect_ticker` is
        endogenous and that the OLS estimate is unreliable. The test is only as good as the
        instruments: they must be correlated with `suspect_ticker` but not with the error term.

        The formula is as follows:

        - Stage 1: Xₜ = π₀ + π'Zₜ + δ'Wₜ + vₜ, the OLS regression of `suspect_ticker` (X) on a constant,
          the instruments (Z) and `other_independent_tickers` (W), keeping the residuals v̂ₜ
        - Stage 2: yₜ = β₀ + β₁·Xₜ + ρ·v̂ₜ + γ'Wₜ + εₜ, the OLS regression of `dependent_ticker` (y)
        - t = ρ̂ / SE(ρ̂), with the classical OLS standard error and a Student-t distribution with n - k
          degrees of freedom; endogeneity is flagged when the p-value is below 0.05
        - All variables are the tickers' `column` (Return by default)

        For more information about the method, see the following papers:

        - Wu, D-M. (1973). "Alternative Tests of Independence Between Stochastic Regressors
        and Disturbances." Econometrica, 41(4), 733-750.
        - Hausman, J.A. (1978). "Specification Tests in Econometrics." Econometrica, 46(6),
        1251-1271.

        Also known as: Hausman test, Durbin-Wu-Hausman test, regression test for
        endogeneity.

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            suspect_ticker (str): The (possibly endogenous) asset being tested.
            instrument_tickers (str | list[str]): One or more instrument asset(s) for
            `suspect_ticker` -- assets correlated with `suspect_ticker` but assumed uncorrelated
            with `dependent_ticker`'s error term.
            other_independent_tickers (str | list[str] | None, optional): Any other (assumed
            exogenous) independent asset(s) to include. Defaults to None.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to regress on. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The coefficient on the first-stage residuals, its t-statistic and p-value, and
            whether `suspect_ticker` is flagged as endogenous at the 5% level.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["SMCI", "NVDA", "TSM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_hausman_wu_test(
            dependent_ticker="SMCI", suspect_ticker="NVDA", instrument_tickers="TSM"
        )
        ```

        Which returns:

        |                    |    Value |
        |:-------------------|---------:|
        | V-Hat Coefficient  |  -0.4021 |
        | T-Statistic        |  -2.3377 |
        | Degrees of Freedom | 246      |
        | P-Value            |   0.0202 |
        | Endogenous (5%)    |   1      |
        """
        period = period if period else "daily"
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

        returns = self._get_series_column(
            period,
            column,
            [
                dependent_ticker,
                suspect_ticker,
                *instrument_tickers,
                *(other_independent_tickers or []),
            ],
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

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def _get_ols_result(
        self,
        dependent_ticker: str,
        independent_tickers: str | list[str],
        period: str | None,
        column: str,
        add_constant: bool,
    ) -> dict:
        """
        Shared helper that fits the OLS regression of `dependent_ticker` on
        `independent_tickers` used internally by every post-estimation diagnostic
        below (`get_breusch_pagan_test`, `get_white_test`, `get_durbin_watson_test`,
        `get_ramsey_reset_test`, `get_chow_test`) -- each of these tests operates on
        an already-fitted regression result dict (see `specification_tests_model`),
        so the controller's job is simply to assemble that same fit `get_ols` itself
        produces before handing it off to the requested diagnostic.
        """
        period = period if period else "daily"
        independent_tickers = (
            [independent_tickers]
            if isinstance(independent_tickers, str)
            else independent_tickers
        )
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
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

        OLS standard errors assume that the residual variance is constant
        (homoskedasticity). The test regresses the squared residuals of the regression on the
        original independent tickers and checks whether they explain any of the residual
        variance, by testing whether the R-squared of that auxiliary regression is
        significantly different from zero.

        The formula is as follows:

        - Auxiliary regression: e_t^2 = a_0 + a' * x_t + v_t, with e_t the OLS residuals and x_t the
          independent tickers' `column` (Return by default)
        - LM = n * R^2_auxiliary (Koenker's studentized form, the statsmodels default)

        This is the studentized statistic of Koenker (1981), which does not assume normally
        distributed errors. It is asymptotically chi-squared distributed with `k - 1` degrees
        of freedom under the null hypothesis of homoskedasticity, where `k` is the number of parameters of the
        auxiliary regression. A low p-value means the residual variance moves with the
        independent tickers (for example, errors grow when the market moves more), so the
        regression's standard errors, t-statistics and p-values are unreliable even though
        the coefficients themselves remain unbiased. Remedies are a robust `cov_type` in
        `get_ols` or `get_wls`. `get_white_test` is a more general alternative that also
        catches non-linear forms of heteroskedasticity.

        For more information about the method, see the following paper:

        - Breusch, T.S., & Pagan, A.R. (1979). "A Simple Test for Heteroscedasticity and
        Random Coefficient Variation." Econometrica, 47(5), 1287-1294.
        - Koenker, R. (1981). "A Note on Studentizing a Test for Heteroscedasticity."
        Journal of Econometrics, 17(1), 107-112.

        Also known as: BP test, Breusch-Pagan-Godfrey test.

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
            Defaults to "daily".
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

        toolkit = Toolkit(
            ["CAT", "DE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # CAT (the first ticker) is dependent; DE is independent
        toolkit.econometrics.get_breusch_pagan_test()
        ```

        Which returns:

        |                              |   Value |
        |:-----------------------------|--------:|
        | Breusch-Pagan Statistic      |  0.0129 |
        | P-Value                      |  0.9096 |
        | Reject Homoskedasticity (5%) |  0      |
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

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

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
        Calculate White's test for heteroskedasticity of a regression of `dependent_ticker` on
        `independent_tickers`.

        A more general version of `get_breusch_pagan_test`. Instead of regressing the squared
        residuals on only the original independent tickers, the auxiliary regression uses
        each ticker, its square and every pairwise cross-product.

        The formula is as follows:

        - e_t^2 = a_0 + SUM_i(a_i * x_it) + SUM_i(b_i * x_it^2) + SUM_{i<j}(c_ij * x_it * x_jt) + v_t
        - LM = n * R^2_auxiliary, chi-squared with as many degrees of freedom as the auxiliary
          regression has terms besides a_0
        - e_t are the OLS residuals and x_it the independent tickers' `column` (Return by default)

        The null hypothesis is homoskedasticity (constant residual variance). A low p-value means the
        regression's standard errors are unreliable and a robust `cov_type` should be used.
        Because it also detects non-linear patterns in the residual variance, White's test
        makes fewer assumptions than Breusch-Pagan, at the cost of power when there are many
        independent tickers, since the number of auxiliary terms grows quadratically.

        For more information about the method, see the following paper:

        - White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a
        Direct Test for Heteroskedasticity." Econometrica, 48(4), 817-838.

        Also known as: White's general test.

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
            Defaults to "daily".
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

        toolkit = Toolkit(
            ["GS", "MS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # GS (the first ticker) is dependent; MS is independent
        toolkit.econometrics.get_white_test()
        ```

        Which returns:

        |                              |   Value |
        |:-----------------------------|--------:|
        | White Statistic              | 11.5102 |
        | P-Value                      |  0.0032 |
        | Reject Homoskedasticity (5%) |  1      |
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

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

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

        OLS standard errors assume that the residuals of consecutive periods are
        uncorrelated. The Durbin-Watson statistic compares each residual with the one before
        it.

        The formula is as follows:

        - DW = SUM_{t=2}^{n}((e_t - e_(t-1))^2) / SUM_{t=1}^{n}(e_t^2)
        - e_t are the residuals of the OLS regression of the dependent ticker's `column` (Return by
          default) on the independent tickers' `column`
        - Interpretation: DW < 1.5 positive autocorrelation likely, DW > 2.5 negative autocorrelation
          likely, otherwise no strong evidence

        The statistic ranges from 0 to 4. A value near 2 means no first-order
        autocorrelation; values below 2 point to positive autocorrelation (consecutive
        residuals tend to share the same sign, often a sign of an omitted trend or lagged
        effect) and values above 2 to negative autocorrelation (residuals tend to alternate in
        sign). Like heteroskedasticity, autocorrelation leaves the coefficients unbiased but
        makes the standard errors unreliable; a "HAC" `cov_type` in `get_ols`, or `get_gls`
        with an AR(1) `omega`, are common remedies. There is no exact p-value: the returned
        interpretation is a rule of thumb (below 1.5 is likely positive, above 2.5 likely
        negative autocorrelation).

        For more information about the method, see the following paper:

        - Durbin, J., & Watson, G.S. (1950). "Testing for Serial Correlation in Least Squares
        Regression I." Biometrika, 37(3/4), 409-428.

        Also known as: DW statistic.

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
            Defaults to "daily".
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

        toolkit = Toolkit(
            ["HD", "LOW"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # HD (the first ticker) is dependent; LOW is independent
        toolkit.econometrics.get_durbin_watson_test()
        ```

        Which returns:

        |                         | Value              |
        |:------------------------|:-------------------|
        | Durbin-Watson Statistic | 1.8584             |
        | Interpretation          | No Strong Evidence |
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

        # The 'Interpretation' entry is a string, so only the statistic is rounded.
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
        Calculate the Variance Inflation Factor (VIF) of every ticker in the Toolkit
        instance, treated as regressors against one another.

        Multicollinearity, where one regressor can (nearly) be predicted from the others, does
        not bias regression coefficients but inflates their standard errors. Individual
        coefficients can then look insignificant, and swing wildly between samples, even when
        the regressors jointly explain the dependent variable well. VIF measures this for
        each ticker by regressing it on all the other tickers.

        The formula is as follows:

        - VIF_i = 1 / (1 - R^2_i)
        - R^2_i is the R-squared of the OLS regression of ticker i's `column` (Return by default) on a
          constant and every other ticker's `column`

        Where `R^2_i` is the R-squared of regressing ticker `i` on all other tickers. A VIF of
        1 means the ticker is uncorrelated with the others; a VIF of `V` means the variance of
        its coefficient is `V` times larger than it would be without multicollinearity. As a
        rule of thumb, a VIF above 10 (or above 5 for a stricter cutoff) signals problematic
        multicollinearity. This is common with assets that move closely together, such as
        several stocks from the same sector, and is worth checking before using them together
        as independent tickers in `get_ols` or the other regression methods.

        Also known as: VIF.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["GOOGL", "META", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_vif()
        ```

        Which returns:

        |       |    VIF |
        |:------|-------:|
        | GOOGL | 1.4431 |
        | META  | 1.8234 |
        | AMZN  | 2.0804 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        test_result = specification_tests_model.get_vif(returns[tickers])

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

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
        Calculate Ramsey's RESET test for functional form misspecification of a regression of
        `dependent_ticker` on `independent_tickers`.

        If the true relationship is non-linear, or a relevant transformation or variable is
        missing, the fitted values of a linear regression still carry information about that
        missing piece. RESET tests for this by adding powers of the fitted values (`fitted^2`,
        ..., `fitted^power`) to the regression and checking whether they are jointly
        significant with an F-test that compares the original (restricted) and augmented
        (unrestricted) model.

        The formula is as follows:

        - Augmented model: y_t = x_t' * beta + SUM_{j=2}^{power}(gamma_j * yhat_t^j) + u_t, with yhat_t
          the fitted values of the original OLS regression and `power` = 3 by default
        - F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u))

        Where `RSS_r` and `RSS_u` are the residual sums of squares of both models, `q` is the
        number of added terms (`power - 1`), `n` the number of observations and `k_u` the
        number of parameters of the augmented model. Under the null hypothesis that the linear
        form is correct, `F` follows an F(q, n - k_u) distribution. A low p-value suggests the
        model is missing something, for example a squared or interaction term or a log
        transformation, without saying what; it is a signal to reconsider the specification,
        not a prescription.

        For more information about the method, see the following paper:

        - Ramsey, J.B. (1969). "Tests for Specification Errors in Classical Linear
        Least-Squares Regression Analysis." Journal of the Royal Statistical Society, Series
        B, 31(2), 350-371.

        Also known as: RESET test, Ramsey RESET.

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
            Defaults to "daily".
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

        toolkit = Toolkit(
            ["NKE", "ADS.DE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        # NKE (the first ticker) is dependent; ADS.DE is independent
        toolkit.econometrics.get_ramsey_reset_test()
        ```

        Which returns:

        |                                   |   Value |
        |:----------------------------------|--------:|
        | RESET F-Statistic                 |  1.4847 |
        | P-Value                           |  0.2285 |
        | Reject Correct Specification (5%) |  0      |
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

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

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
        Calculate the Chow test for a structural break at `break_date` in a regression of
        `dependent_ticker` on `independent_tickers`.

        The sample is split at `break_date` into a "before" and "after" period, a regression
        is fitted on each part separately and on the full sample, and an F-test checks whether
        allowing the coefficients to differ between the two periods significantly reduces the
        residual sum of squares.

        The formula is as follows:

        - F = ((RSS_pooled - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n - 2k))
        - The second sub-period starts at the first date on or after `break_date`, and `k` includes
          the constant when add_constant=True

        Where `RSS_pooled` is the residual sum of squares of the full-sample regression,
        `RSS_1` and `RSS_2` those of the two sub-periods, `k` the number of parameters and `n`
        the total number of observations. Under the null hypothesis of no structural break
        (the same coefficients apply throughout), `F` follows an F(k, n - 2k) distribution. A
        low p-value means the relationship changed at `break_date`, for example an asset's
        beta shifting after a merger, a regulatory change or a market regime change, so a
        single regression over the full sample is misspecified. The break date must be chosen
        in advance; testing many candidate dates and keeping the most significant overstates
        the evidence.

        For more information about the method, see the following paper:

        - Chow, G.C. (1960). "Tests of Equality Between Sets of Coefficients in Two Linear
        Regressions." Econometrica, 28(3), 591-605.

        Also known as: Chow breakpoint test.

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
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # NVDA is dependent and the Benchmark independent, with a break at the launch of ChatGPT
        toolkit.econometrics.get_chow_test(break_date="2022-11-30", include_benchmark=True)
        ```

        Which returns:

        |                                 |   Value |
        |:--------------------------------|--------:|
        | Chow F-Statistic                |  0.811  |
        | P-Value                         |  0.4447 |
        | Reject No Structural Break (5%) |  0      |
        """
        dependent_ticker, independent_tickers = self._resolve_dependent_independent(
            dependent_ticker=dependent_ticker,
            independent_tickers=independent_tickers,
            include_benchmark=include_benchmark,
        )

        period = period if period else "daily"
        returns = self._get_series_column(
            period, column, [dependent_ticker, *independent_tickers]
        )

        result_full = regression_model.get_ols(
            returns[dependent_ticker],
            returns[independent_tickers],
            add_constant=add_constant,
        )

        index = returns.index
        timestamps = (
            index.to_timestamp()
            if isinstance(index, pd.PeriodIndex)
            else pd.DatetimeIndex(index)
        )
        break_index = int(
            timestamps.searchsorted(pd.Timestamp(break_date).to_datetime64())
        )

        test_result = specification_tests_model.get_chow_test(
            result_full,
            returns[independent_tickers],
            returns[dependent_ticker],
            break_index=break_index,
            add_constant=add_constant,
        )

        return apply_rounding(
            test_result, rounding if rounding is not None else self._rounding
        )

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

        The formula is as follows:

        - Z = [1, exogenous tickers, instruments] and X = [1, exogenous tickers, endogenous tickers], all
          in `column` (Return by default), the constant only with add_constant=True
        - Stage 1: X̂ = Z(Z'Z)⁻¹Z'X, the fitted values of each endogenous ticker regressed on Z
        - Stage 2: β̂ = (X̂'X)⁻¹X̂'y
        - Cov(β̂) = σ̂²·(X̂'X̂)⁻¹, with σ̂² = Σ(yₜ - xₜ'β̂)² / (n - k) (linearmodels IV2SLS with
          cov_type="unadjusted" and debiased=True)

        Args:
            dependent_ticker (str): The dependent (predicted) asset.
            endogenous_ticker (str | list[str]): The endogenous regressor asset(s) --
            suspected correlated with the error term.
            instrument_tickers (str | list[str]): The excluded instrument asset(s),
            correlated with `endogenous_ticker` but assumed uncorrelated with the error
            term. Must supply at least as many instruments as endogenous regressors.
            exogenous_tickers (str | list[str] | None, optional): Other, non-instrumented
            control asset(s) included as-is in both stages. Defaults to None.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["SMCI", "NVDA", "TSM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_iv_2sls(
            dependent_ticker="SMCI", endogenous_ticker="NVDA", instrument_tickers="TSM"
        )
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |       -0.0008 |       0.0027 |       -0.2931 |    0.7697 |
        | NVDA      |        1.2518 |       0.1094 |       11.441  |    0      |
        """
        period = period if period else "daily"
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

        returns = self._get_series_column(
            period,
            column,
            [
                dependent_ticker,
                *endogenous_ticker,
                *instrument_tickers,
                *(exogenous_tickers or []),
            ],
        )

        result = causal_inference_model.get_iv_2sls(
            returns[dependent_ticker],
            returns[endogenous_ticker],
            returns[instrument_tickers],
            returns[exogenous_tickers] if exogenous_tickers else None,
            add_constant=add_constant,
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
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

        With returns as the outcome, the `Treated x Post` coefficient is the change in the
        average return per period after the event, not the price impact around it, which
        is what `get_event_study` measures. The panel stacks tickers observed on the same
        dates, whose returns are correlated, while the standard errors treat every
        observation as independent, so the p-value tends to be too low for closely related
        tickers. The estimate rests on parallel trends, so the first check is whether both
        groups moved alike before the event.

        The formula is as follows:

        - yᵢₜ = β₀ + β₁·Treatedᵢ + β₂·Postₜ + δ·(Treatedᵢ × Postₜ) + εᵢₜ, fitted by OLS with classical
          standard errors on the stacked (ticker, date) panel of `column` (Return by default)
        - Treatedᵢ = 1 for `treated_tickers`, Postₜ = 1 on or after `treatment_date`
        - δ̂ = (mean(y_treated,post) - mean(y_treated,pre)) - (mean(y_control,post) - mean(y_control,pre))

        Args:
            treated_tickers (str | list[str]): The asset(s) subject to the event/treatment.
            treatment_date (str): The date the event/treatment occurs, in the same format
            accepted by `pd.Timestamp`. Observations on or after this date are `Post = 1`.
            control_tickers (str | list[str] | None, optional): The untreated comparison
            asset(s). Defaults to None, which uses every ticker NOT in `treated_tickers`,
            excluding "Benchmark" (pass `control_tickers="Benchmark"` to use the benchmark as
            the control group).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["NVDA", "INTC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_difference_in_differences(
            treated_tickers="NVDA",
            control_tickers="INTC",
            treatment_date="2023-06-01",
            period="monthly",
        )
        ```

        Which returns:

        |                |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:---------------|--------------:|-------------:|--------------:|----------:|
        | Intercept      |       -0.0126 |       0.0262 |       -0.4794 |    0.6325 |
        | Treated        |        0.0669 |       0.0371 |        1.8061 |    0.0735 |
        | Post           |        0.0291 |       0.0362 |        0.8044 |    0.4229 |
        | Treated x Post |       -0.0246 |       0.0511 |       -0.4817 |    0.6309 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        treated_tickers = (
            [treated_tickers] if isinstance(treated_tickers, str) else treated_tickers
        )

        if control_tickers is None:
            control_tickers = [
                ticker
                for ticker in self._get_tickers()
                if ticker not in treated_tickers
            ]
        elif isinstance(control_tickers, str):
            control_tickers = [control_tickers]

        panel_tickers = [*treated_tickers, *control_tickers]

        index = returns.index
        timestamps = (
            index.to_timestamp()
            if isinstance(index, pd.PeriodIndex)
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

        # Without one of the two groups the Treated dummy is constant and the design
        # is rank deficient, which says nothing about the cause.
        for group, flag in (("treated", 1.0), ("control", 0.0)):
            if not treated.eq(flag).any():
                raise ValueError(
                    f"There are no {group} observations: the {group} tickers have no "
                    f"{column} data in this period."
                )

        post = pd.concat(post_values, ignore_index=True)

        result = causal_inference_model.get_difference_in_differences(
            outcome, treated, post, add_constant=add_constant
        )

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
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

        The formula is as follows:

        - x̃ₜ = xₜ - cutoff, keeping the observations with |x̃ₜ| < h, where xₜ is the running ticker's
          `column` (Return by default) and h = `bandwidth`, or half the range of xₜ by default
        - Each side (x̃ₜ < 0 and x̃ₜ ≥ 0): yₜ = a + b·x̃ₜ + eₜ, fitted by OLS (kernel="uniform", the
          default) or by WLS with weights 1 - |x̃ₜ| / h (kernel="triangular")
        - Discontinuity = a_right - a_left, with SE = √(SE(a_right)² + SE(a_left)²)
        - t = Discontinuity / SE, with a Student-t distribution on the two sides' residual degrees of
          freedom combined

        Args:
            dependent_ticker (str): The outcome asset.
            running_variable_ticker (str): The asset (or column) whose value determines
            which side of `cutoff` an observation falls on.
            cutoff (float): The threshold value of `running_variable_ticker` at which the
            discontinuity is estimated.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["COIN", "BTCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2025-12-31",
        )

        # COIN's return on days Bitcoin rises versus days it falls
        toolkit.econometrics.get_regression_discontinuity(
            dependent_ticker="COIN", running_variable_ticker="BTCUSD", cutoff=0
        )
        ```

        Which returns:

        |               |    Value |
        |:--------------|---------:|
        | Discontinuity |   0.0031 |
        | Std. Error    |   0.0027 |
        | t-Statistic   |   1.1191 |
        | P-Value       |   0.2633 |
        | Cutoff        |   0      |
        | Bandwidth     |   0.1497 |
        | N Left        | 729      |
        | N Right       | 728      |
        """
        period = period if period else "daily"
        returns = self._get_series_column(
            period, column, [dependent_ticker, running_variable_ticker]
        )

        result = causal_inference_model.get_regression_discontinuity(
            returns[dependent_ticker],
            returns[running_variable_ticker],
            cutoff=cutoff,
            bandwidth=bandwidth,
            kernel=kernel,
        )

        return apply_rounding(
            causal_inference_model.regression_discontinuity_summary_table(result),
            rounding if rounding is not None else self._rounding,
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

        The formula is as follows:

        - Dₜ = 1 if the treatment ticker's `column` (Return by default) is above `treatment_threshold`
          (0 by default), else 0
        - e(xₜ) = 1 / (1 + exp(-xₜ'β)), a logit of Dₜ on the covariate tickers (plus a constant)
        - Matching on ℓₜ = ln(e / (1 - e)): each treated period, in order, takes the nearest unmatched
          control period (one-to-one, without replacement) if |ℓ_treated - ℓ_control| ≤ caliper, which
          defaults to 0.2 × the standard deviation (ddof=1) of ℓ
        - ATT = (1 / m)·Σ(y_treated - y_control) over the m matched pairs
        - SE = std(y_treated - y_control, ddof=1) / √m, with a Student-t distribution with m - 1 degrees
          of freedom

        Args:
            dependent_ticker (str): The outcome asset.
            treatment_ticker (str): The asset whose return, once it exceeds
            `treatment_threshold`, defines the treatment indicator.
            covariate_tickers (str | list[str]): The asset(s) used as covariates to
            estimate the propensity score -- should include asset(s) believed to drive
            selection into "treatment".
            treatment_threshold (float, optional): The return threshold defining
            treatment. Defaults to 0.0.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["XOM", "USO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # XOM's return on days USO rises more than 2%, matched on the Benchmark's return
        toolkit.econometrics.get_propensity_score_matching(
            dependent_ticker="XOM",
            treatment_ticker="USO",
            covariate_tickers="Benchmark",
            treatment_threshold=0.02,
        )
        ```

        Which returns:

        |               |     Value |
        |:--------------|----------:|
        | ATT           |    0.0163 |
        | Std. Error    |    0.0016 |
        | t-Statistic   |    9.9366 |
        | P-Value       |    0      |
        | Matched Pairs |  181      |
        | N Treated     |  185      |
        | N Control     | 1069      |
        """
        period = period if period else "daily"
        covariate_tickers = (
            [covariate_tickers]
            if isinstance(covariate_tickers, str)
            else covariate_tickers
        )

        returns = self._get_series_column(
            period, column, [dependent_ticker, treatment_ticker, *covariate_tickers]
        )
        treatment = (returns[treatment_ticker] > treatment_threshold).astype(float)

        result = causal_inference_model.get_propensity_score_matching(
            treatment,
            returns[dependent_ticker],
            returns[covariate_tickers],
            caliper=caliper,
            add_constant=add_constant,
        )

        return apply_rounding(
            causal_inference_model.propensity_score_matching_summary(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_synthetic_control(
        self,
        treated_ticker: str,
        treatment_period: str,
        donor_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Construct a Synthetic Control for `treated_ticker` from a weighted combination of
        `donor_tickers` (Abadie, Diamond & Hainmueller, 2010), and estimate the effect of an
        event/intervention as the post-`treatment_period` gap between `treated_ticker`'s
        actual and synthetic counterfactual return path.

        When a single company or market experiences an event (an acquisition, a regulatory
        change, an index inclusion) and there is no single obvious comparison, the synthetic
        control builds one from the data: a weighted average of unaffected "donor" tickers,
        with weights chosen so it tracks the treated ticker as closely as possible before the
        event. If that pre-event fit is good, the synthetic control's path after the event
        estimates what would have happened without it, and the gap is the estimated effect.

        The formula is as follows:

        - Weights: w* = argmin_w ||treated_pre - donors_pre @ w||^2, subject to w >= 0 and
          sum(w) = 1 (SLSQP), with pre the dates before `treatment_period`
        - Synthetic control: synthetic_t = donors_t @ w*, for every period t
        - Treatment effect: gap_t = treated_t - synthetic_t, for every post-event period t
        - Average Treatment Effect = mean of gap_t over the post-event periods
        - RMSPE ratio = sqrt(mean(gap_post^2)) / sqrt(mean(gap_pre^2))
        - P-Value = the share of the treated ticker's and every placebo donor's RMSPE ratios that are at
          least the treated ticker's ratio
        - All series are the tickers' `column` (Return by default)

        Unlike difference-in-differences, this does not require the treated and comparison
        units to follow parallel trends, as the synthetic control is built to match the
        treated ticker's own pre-event path. Because there is only one treated unit, the
        p-value comes from placebo tests: the procedure is rerun with each donor as if it were
        treated, and the treated ticker's post/pre fit ratio is compared with theirs. Matching
        is done on the pre-event outcome path only, a common simplification of the original
        method.

        For more information about the method, see the following papers:

        - Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for
        Comparative Case Studies: Estimating the Effect of California's Tobacco Control
        Program." Journal of the American Statistical Association, 105(490), 493-505.
        - Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and
        Methodological Aspects." Journal of Economic Literature, 59(2), 391-425.

        Also known as: SCM, synthetic control method.

        Args:
            treated_ticker (str): The asset believed to be affected by an event/
            intervention starting at `treatment_period`.
            treatment_period (str): The first post-treatment period -- periods at or
            after this value (within `period`'s index) are treated as post-treatment,
            everything before as pre-treatment (used to fit the synthetic control's
            weights).
            donor_tickers (str | list[str] | None, optional): The ticker(s) forming
            the donor pool the synthetic control is built from. Defaults to None,
            meaning every other Toolkit ticker (subject to `include_benchmark`).
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" in the
            default donor pool (has no effect when donor_tickers is given explicitly).
            Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The average post-treatment effect, pre/post RMSPE and their
            ratio, the placebo-based p-value, and the donor pool/period counts.

        Notes:
        - Needs at least 2 donor tickers -- with only 1-2 Toolkit tickers loaded,
        pass `donor_tickers` explicitly (e.g. including "Benchmark") or (better)
        construct the Toolkit with more tickers, since small donor pools give both a
        poorly-identified synthetic control and a very coarse placebo p-value (with
        `k` donors, the smallest achievable p-value is `1 / (k + 1)`).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BA", "AIR.PA", "LMT", "GD", "RTX", "NOC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2023-01-01",
            end_date="2024-12-31",
        )

        # Boeing after the 737 MAX 9 grounding, with the other aerospace companies as donors
        toolkit.econometrics.get_synthetic_control(treated_ticker="BA", treatment_period="2024-01-08")
        ```

        Which returns:

        |                          |    Value |
        |:-------------------------|---------:|
        | Average Treatment Effect |  -0.0017 |
        | Pre-Treatment RMSPE      |   0.0156 |
        | Post-Treatment RMSPE     |   0.0204 |
        | RMSPE Ratio              |   1.3094 |
        | P-Value                  |   0.5    |
        | N Donors                 |   5      |
        | N Pre-Periods            | 260      |
        | N Post-Periods           | 255      |
        """
        period = period if period else "daily"
        if donor_tickers is None:
            donor_tickers = [
                ticker
                for ticker in self._get_tickers(include_benchmark=include_benchmark)
                if ticker != treated_ticker
            ]
        else:
            donor_tickers = (
                [donor_tickers]
                if isinstance(donor_tickers, str)
                else list(donor_tickers)
            )

        returns = self._get_series_column(
            period, column, [treated_ticker, *donor_tickers]
        )

        result = causal_inference_model.get_synthetic_control(
            returns[treated_ticker],
            returns[donor_tickers],
            treatment_period=treatment_period,
        )

        return apply_rounding(
            causal_inference_model.synthetic_control_summary(result),
            rounding if rounding is not None else self._rounding,
        )

    def _get_panel_data(
        self,
        independent_tickers: str | list[str] | None,
        independent_column: str | None,
        dependent_tickers: str | list[str] | None,
        period: str | None,
        column: str,
        include_benchmark: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Shared input-shaping step for `get_fixed_effects`/`get_random_effects`/
        `get_hausman_test`: unlike every other regression method in this module
        (which pairs one dependent ticker with one or more independent tickers,
        all on equal footing), panel data methods need a genuine entity dimension
        that a handful of single ticker return series don't have on their own.

        This module treats `dependent_tickers` (defaulting to every ticker in the
        `Toolkit` instance, without "Benchmark" unless `include_benchmark` is True, minus
        `independent_tickers` if given) as the panel of ENTITIES -- e.g. a panel of stocks -- and supports
        two, mutually exclusive ways of building the regressor `x`:

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

        period = period if period else "daily"

        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )

        # The raw slice: panel methods need each entity's own NaN pattern on the full date index.
        prices = self._historical_data[period][column]

        if independent_tickers is not None:
            independent_tickers = (
                [independent_tickers]
                if isinstance(independent_tickers, str)
                else list(independent_tickers)
            )

            # The factor is read from the loaded data, so it has to be one of its tickers.
            missing = [ticker for ticker in independent_tickers if ticker not in prices]
            if missing:
                raise ValueError(
                    f"independent_tickers {', '.join(missing)} are not among the loaded "
                    "tickers: add them to the tickers, or use 'Benchmark' for the "
                    "benchmark."
                )

            if dependent_tickers is None:
                dependent_tickers = [
                    ticker
                    for ticker in prices.columns
                    if ticker not in independent_tickers
                    and (include_benchmark or ticker != BENCHMARK_TICKER)
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
                dependent_tickers = [
                    ticker
                    for ticker in prices.columns
                    if include_benchmark or ticker != BENCHMARK_TICKER
                ]
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
        include_benchmark: bool = False,
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

        The formula is as follows:

        - ỹᵢₜ = yᵢₜ - mean_t(yᵢₜ) and x̃ᵢₜ = xᵢₜ - mean_t(xᵢₜ) (entity_effects=True, the default);
          time_effects=True also removes each date's mean (two-way fixed effects)
        - β̂_FE = (X̃'X̃)⁻¹X̃'ỹ, without a separate constant (linearmodels PanelOLS, unadjusted covariance)
        - Entity Effect: αᵢ = mean_t(yᵢₜ - xᵢₜ'β̂_FE); Time Effect: λₜ = mean_i(yᵢₜ - xᵢₜ'β̂_FE)
        - yᵢₜ is each dependent ticker's `column` (Return by default)

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
            quarterly, or yearly). Defaults to "daily".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            entities. Defaults to False.
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
            requested fixed effects, or the fixed effects absorb the regressor: the
            common factor of `independent_tickers` is identical for every entity at
            each date, so it cannot be combined with `time_effects=True`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM", "CVX", "SHEL", "BP", "TTE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_fixed_effects(independent_tickers="Benchmark")
        ```

        Which returns:

        |                     |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:--------------------|--------------:|-------------:|--------------:|----------:|
        | Benchmark           |        0.5795 |       0.0183 |       31.7272 |         0 |
        | Entity Effect: BP   |        0.0004 |     nan      |      nan      |       nan |
        | Entity Effect: CVX  |        0.0004 |     nan      |      nan      |       nan |
        | Entity Effect: SHEL |        0.0005 |     nan      |      nan      |       nan |
        | Entity Effect: TTE  |        0.0004 |     nan      |      nan      |       nan |
        | Entity Effect: XOM  |        0.0008 |     nan      |      nan      |       nan |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers,
            independent_column,
            dependent_tickers,
            period,
            column,
            include_benchmark=include_benchmark,
        )

        result = panel_data_model.get_fixed_effects(
            y_panel, x_panel, entity_effects=entity_effects, time_effects=time_effects
        )

        rounding = rounding if rounding is not None else self._rounding
        summary = regression_model.regression_summary_table(result["regression"])

        if result["entity_effects"] is not None:
            entity_rows = pd.DataFrame({"Coefficient": result["entity_effects"]})
            entity_rows.index = [
                f"Entity Effect: {entity}" for entity in entity_rows.index
            ]
            summary = pd.concat([summary, entity_rows])

        if result["time_effects"] is not None:
            time_rows = pd.DataFrame({"Coefficient": result["time_effects"]})
            time_rows.index = [f"Time Effect: {time}" for time in time_rows.index]
            summary = pd.concat([summary, time_rows])

        return apply_rounding(summary, rounding)

    @handle_errors
    def get_random_effects(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
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

        The formula is as follows:

        - θᵢ = 1 - √(σ²_ε / (σ²_ε + Tᵢ·σ²_u)), with the Swamy-Arora estimates of the idiosyncratic (σ²_ε)
          and entity (σ²_u) variance components
        - y*ᵢₜ = yᵢₜ - θᵢ·mean_t(yᵢₜ) and x*ᵢₜ = xᵢₜ - θᵢ·mean_t(xᵢₜ), the constant becoming 1 - θᵢ
        - β̂_RE = OLS of y* on x* (linearmodels RandomEffects), with yᵢₜ each dependent ticker's `column`
          (Return by default)

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
            quarterly, or yearly). Defaults to "daily".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            entities. Defaults to False.
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
            ["JPM", "BAC", "C", "WFC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_random_effects(independent_tickers="Benchmark")
        ```

        Which returns:

        |           |   Coefficient |   Std. Error |   t-Statistic |   P-Value |
        |:----------|--------------:|-------------:|--------------:|----------:|
        | Intercept |        0.0003 |       0.0002 |        1.7192 |    0.0856 |
        | Benchmark |        0.9788 |       0.0183 |       53.4931 |    0      |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers,
            independent_column,
            dependent_tickers,
            period,
            column,
            include_benchmark=include_benchmark,
        )

        result = panel_data_model.get_random_effects(y_panel, x_panel)

        return apply_rounding(
            regression_model.regression_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_hausman_test(
        self,
        independent_tickers: str | list[str] | None = None,
        independent_column: str | None = None,
        dependent_tickers: str | list[str] | None = None,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Hausman specification test comparing a Fixed Effects and a
        Random Effects fit of `dependent_tickers` (the panel of entities) on a
        regressor `x`.

        Also known as: Hausman specification test, fixed versus random effects test.

        Random Effects is more efficient than Fixed Effects but relies on entity
        effects being uncorrelated with the regressor(s) -- if that assumption is
        violated, Random Effects is inconsistent while Fixed Effects remains
        consistent regardless. See `panel_data_model.get_hausman_test` for the full
        formula and references, and `get_fixed_effects`/`get_random_effects` for
        how the two models being compared are estimated (including how
        `independent_tickers`/`independent_column` shape the panel, and why
        `independent_column` is generally the more reliable choice here).

        The formula is as follows:

        - H = (β̂_FE - β̂_RE)'·[Var(β̂_FE) - Var(β̂_RE)]⁻¹·(β̂_FE - β̂_RE), over the k slope coefficients
          the two models share (a pseudo-inverse when the difference is singular, and H floored at 0)
        - H is chi-squared with k degrees of freedom; Random Effects is rejected when the p-value is
          below 0.05
        - β̂_FE comes from `get_fixed_effects` with entity effects only and β̂_RE from `get_random_effects`

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
            quarterly, or yearly). Defaults to "daily".
            column (str, optional): The dependent variable's historical data
            column. Defaults to "Return".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            entities. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.Series: The Hausman statistic, its degrees of freedom, its p-value,
            and whether Fixed Effects is preferred over Random Effects at the 5%
            level.

        Raises:
            ValueError: If both or neither of `independent_tickers`/
            `independent_column` are given, or (see `panel_data_model.
            get_hausman_test`) if there are not enough observations or entities to
            fit either model, or the two estimators share no common regressors to
            compare.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_hausman_test(independent_column="Volume", period="weekly")
        ```

        Which returns:

        |                           |   Value |
        |:--------------------------|--------:|
        | Hausman Statistic         |  0.5866 |
        | Degrees of Freedom        |  1      |
        | P-Value                   |  0.4438 |
        | Prefer Fixed Effects (5%) |  0      |
        """
        y_panel, x_panel, dependent_tickers = self._get_panel_data(
            independent_tickers,
            independent_column,
            dependent_tickers,
            period,
            column,
            include_benchmark=include_benchmark,
        )

        result = panel_data_model.get_hausman_test(y_panel, x_panel)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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
        estimation method (exact Maximum Likelihood via the Kalman filter) and its
        practical caveats.

        The formula is as follows:

        - y'ₜ = ΔᵈYₜ, the `column` level (Adj Close by default) differenced d times (d = 1 by default)
        - y'ₜ - μ = Σ(i=1..p) φᵢ·(y'ₜ₋ᵢ - μ) + εₜ + Σ(j=1..q) θⱼ·εₜ₋ⱼ, with p = q = 1 by default and μ the
          mean (d = 0) or drift (d ≥ 1) of y'ₜ when include_constant=True, and μ = 0 otherwise
        - Fitted by exact Maximum Likelihood via the Kalman filter (statsmodels ARIMA)
        - Forecast: the conditional mean E[Y_T+h | Y_1..Y_T] for h = 1..`forecast_steps`, in levels

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to fit. Defaults to
            "Adj Close".
            p (int, optional): The autoregressive order. Defaults to 1.
            d (int, optional): The number of times to difference the series. Defaults
            to 1 (the typical choice for a non-stationary price level series).
            q (int, optional): The moving-average order. Defaults to 1.
            forecast_steps (int, optional): The number of periods ahead to forecast.
            Defaults to 5.
            include_constant (bool, optional): Whether to estimate the mean of the
            differenced series (its drift for `d >= 1`). Defaults to True -- see
            `time_series_model.get_arima_forecast` for when to set this to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers forecast. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: The `forecast_steps`-ahead forecast, one column per ticker,
            indexed `1, ..., forecast_steps` (periods ahead, not a continuation of the
            historical date index). Each ticker's ARIMA model is fit independently on
            its own history; a ticker without enough data has an empty (NaN) column.

        Notes:
        - The method retrieves historical price data based on the specified `period`
        for every ticker in the Toolkit instance and fits an ARIMA model to each.
        - See `time_series_model.get_arima_forecast`'s `Notes` for this estimator's
        practical limitations (in particular, keep `p + q` small).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_arima_forecast(forecast_steps=10)
        ```

        Which returns:

        |   Step |   GCUSD |
        |-------:|--------:|
        |      1 | 4348.34 |
        |      2 | 4348.38 |
        |      3 | 4350.82 |
        |      4 | 4352.47 |
        |      5 | 4354.38 |
        |      6 | 4356.2  |
        |      7 | 4358.05 |
        |      8 | 4359.89 |
        |      9 | 4361.73 |
        |     10 | 4363.57 |
        """
        period = period if period else "daily"
        # Each ARIMA fit drops the missing values of its own series, so a series observed
        # at other dates does not cut the others down to the shared rows.
        prices = self._get_series_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        forecasts = {}
        for ticker in tickers:
            # Each ticker is fitted on its own history; one without enough data of its
            # own gets an empty column instead of failing the other tickers.
            try:
                forecasts[ticker] = time_series_model.get_arima_forecast(
                    prices[ticker],
                    p=p,
                    d=d,
                    q=q,
                    forecast_steps=forecast_steps,
                    include_constant=include_constant,
                )["forecast"]
            except ValueError as error:
                logger.warning("No ARIMA forecast for %s: %s", ticker, error)

        if not forecasts:
            raise ValueError("None of the tickers has enough data for an ARIMA fit.")

        result = pd.DataFrame(forecasts).reindex(columns=tickers)
        result.index.name = "Step"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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
        Fit a Vector Autoregression (VAR) across every ticker in the Toolkit instance and
        forecast `forecast_steps` periods ahead.

        A VAR models several series jointly: each ticker's value is regressed on `lags` past
        values of all tickers, including its own. Each asset's future can therefore depend on
        its own history and on the history of the others, without deciding in advance which
        one drives which.

        The formula is as follows:

        - Y_t = c + SUM_{l=1}^{lags}(Phi_l * Y_(t-l)) + e_t, with Y_t every ticker's `column` (Return by
          default) and `lags` = 1 by default
        - Each equation is an OLS regression on a constant and the `lags` lagged values of all tickers
        - Forecast: Yhat_(T+h) = c + SUM_{l=1}^{lags}(Phi_l * Yhat_(T+h-l)), with Yhat_s = Y_s for s <= T

        Where `Y_t` holds the values of all tickers at time `t` and each `Phi_l` is a matrix of
        coefficients. Because every equation has the same regressors, the model is estimated
        exactly by a separate OLS regression per ticker. Forecasts are produced by running the
        fitted equations forward one step at a time, feeding each forecast back in as the
        most recent lag. The coefficients show how shocks spread between assets;
        `get_impulse_response_function` and `get_variance_decomposition` summarise that, and
        `get_granger_causality` tests whether one ticker helps predict another. A VAR assumes
        stationary series, so returns are generally more suitable than prices (see
        `get_augmented_dickey_fuller`).

        For more information about the method, see the following paper:

        - Sims, C.A. (1980). "Macroeconomics and Reality." Econometrica, 48(1), 1-48.

        Also known as: VAR model, vector autoregressive model.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["^VIX", "^VIX3M"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_var_forecast(column="Adj Close", forecast_steps=20)
        ```

        Which returns:

        |   Step |    ^VIX |   ^VIX3M |
        |-------:|--------:|---------:|
        |      1 | 15.2712 |  18.3667 |
        |      2 | 15.5578 |  18.5342 |
        |      3 | 15.8138 |  18.6846 |
        |      4 | 16.0425 |  18.82   |
        |      5 | 16.2471 |  18.9418 |
        |      6 | 16.4301 |  19.0517 |
        |      7 | 16.5941 |  19.1509 |
        |      8 | 16.7411 |  19.2406 |
        |      9 | 16.873  |  19.322  |
        |     10 | 16.9915 |  19.3958 |
        |     11 | 17.0981 |  19.4629 |
        |     12 | 17.1942 |  19.5241 |
        |     13 | 17.2808 |  19.58   |
        |     14 | 17.359  |  19.6311 |
        |     15 | 17.4298 |  19.6781 |
        |     16 | 17.494  |  19.7213 |
        |     17 | 17.5523 |  19.7611 |
        |     18 | 17.6053 |  19.7979 |
        |     19 | 17.6536 |  19.8321 |
        |     20 | 17.6978 |  19.8639 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        result = time_series_model.get_var_forecast(
            returns[tickers], lags=lags, forecast_steps=forecast_steps
        )

        return result["forecast"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_impulse_response_function(
        self,
        period: str | None = None,
        column: str = "Return",
        lags: int = 1,
        periods: int = 10,
        orthogonalized: bool = True,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Vector Autoregression (VAR) across every ticker in the Toolkit instance
        and trace out the Impulse Response Function (IRF) -- how a shock to each
        ticker propagates through the whole system over `periods` periods ahead. The
        shock is one standard deviation of the orthogonalized innovation when
        `orthogonalized=True` (the default) and a unit shock to the reduced-form
        residual when `orthogonalized=False`.

        Also known as: IRF.

        A natural companion to `get_var_forecast`: rather than forecasting the levels
        forward, this traces out each ticker's dynamic response to a shock in every
        ticker (including itself) -- see `time_series_model.get_impulse_response_function`
        for the full formula, the Cholesky-orthogonalization used to identify the
        shocks, and why the ordering of tickers is an identifying assumption.

        The formula is as follows:

        - Ψ₀ = I and Ψₙ = Σ(l=1..min(n, p)) Φₗ·Ψₙ₋ₗ, the moving-average matrices of the VAR(p) fitted on
          every ticker's `column` (Return by default), p = `lags` (1 by default)
        - Σᵤ = Û'Û / (T - k·p - 1) = P·P', with P the lower-triangular Cholesky factor
        - Θₙ = Ψₙ·P (orthogonalized=True, a one-standard-deviation shock) or Θₙ = Ψₙ
          (orthogonalized=False, a unit reduced-form shock)
        - Response of ticker i, n periods after a shock to ticker j: Θₙ[i, j], for n = 0..`periods`

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to model. Defaults to
            "Return".
            lags (int, optional): The VAR order. Defaults to 1.
            periods (int, optional): The number of periods ahead to trace the response
            out to. Defaults to 10.
            orthogonalized (bool, optional): Whether to orthogonalize the shocks via a
            Cholesky decomposition of the residual covariance matrix -- see
            `time_series_model.get_impulse_response_function`. Defaults to True.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: Each shock ticker's response table, one column per response
            ticker, concatenated side by side under a top-level column per shock
            ticker, indexed `0, ..., periods` (horizon, `0` = impact response).

        Notes:
        - The ordering of the tickers modeled determines which ticker is treated as
        contemporaneously prior to the others. That ordering follows the Toolkit
        instance's own ticker order and is not configurable here -- reorder the
        Toolkit instance's tickers themselves to change the Cholesky identification
        order -- see `time_series_model.get_impulse_response_function`'s `Notes`.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^VIX", "^OVX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_impulse_response_function(column="Adj Close", periods=20)
        ```

        Which returns:

        |   Horizon |   ('^VIX', '^VIX') |   ('^VIX', '^OVX') |   ('^OVX', '^VIX') |   ('^OVX', '^OVX') |
        |----------:|-------------------:|-------------------:|-------------------:|-------------------:|
        |         0 |             1.7975 |             1.1106 |             0      |             2.4048 |
        |         1 |             1.6806 |             1.1595 |             0.022  |             2.2148 |
        |         2 |             1.5725 |             1.1956 |             0.0407 |             2.0416 |
        |         3 |             1.4723 |             1.2207 |             0.0565 |             1.8834 |
        |         4 |             1.3794 |             1.2361 |             0.0698 |             1.7389 |
        |         5 |             1.2933 |             1.2433 |             0.0807 |             1.6069 |
        |         6 |             1.2133 |             1.2434 |             0.0897 |             1.4861 |
        |         7 |             1.1389 |             1.2374 |             0.097  |             1.3755 |
        |         8 |             1.0698 |             1.2262 |             0.1027 |             1.2743 |
        |         9 |             1.0054 |             1.2106 |             0.1071 |             1.1814 |
        |        10 |             0.9454 |             1.1914 |             0.1104 |             1.0963 |
        |        11 |             0.8895 |             1.1691 |             0.1126 |             1.0181 |
        |        12 |             0.8374 |             1.1444 |             0.114  |             0.9462 |
        |        13 |             0.7887 |             1.1176 |             0.1146 |             0.8801 |
        |        14 |             0.7432 |             1.0893 |             0.1145 |             0.8193 |
        |        15 |             0.7006 |             1.0597 |             0.1139 |             0.7633 |
        |        16 |             0.6608 |             1.0292 |             0.1129 |             0.7117 |
        |        17 |             0.6236 |             0.9981 |             0.1114 |             0.664  |
        |        18 |             0.5886 |             0.9667 |             0.1096 |             0.6201 |
        |        19 |             0.5559 |             0.9351 |             0.1075 |             0.5794 |
        |        20 |             0.5252 |             0.9034 |             0.1052 |             0.5418 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        var_result = time_series_model.get_var_forecast(
            returns[tickers], lags=lags, forecast_steps=1
        )
        result = time_series_model.get_impulse_response_function(
            var_result, periods=periods, orthogonalized=orthogonalized
        )

        return apply_rounding(
            time_series_model.irf_summary_table(result),
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
    def get_variance_decomposition(
        self,
        period: str | None = None,
        column: str = "Return",
        lags: int = 1,
        periods: int = 10,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Vector Autoregression (VAR) across every ticker in the Toolkit instance
        and compute the (orthogonalized) Forecast Error Variance Decomposition (FEVD)
        -- what fraction of each ticker's `h`-step-ahead forecast error variance is
        attributable to each ticker's own structural shock, for `h = 1, ..., periods`.

        Also known as: FEVD, variance decomposition.

        The other natural companion to `get_var_forecast` (alongside
        `get_impulse_response_function`, which the FEVD is built from) -- see
        `time_series_model.get_variance_decomposition` for the full formula. A large
        own-shock share at short horizons that decays as the horizon grows is the
        classic signature of a ticker that is initially self-driven but increasingly
        explained by the rest of the system over time.

        The formula is as follows:

        - Θₙ = Ψₙ·P, the orthogonalized moving-average matrices of `get_impulse_response_function`, from
          a VAR(p) on every ticker's `column` (Return by default), p = `lags` (1 by default)
        - Share of ticker j's shock in ticker i's h-step forecast error variance:
          ωᵢⱼ(h) = Σ(n=0..h-1) Θₙ[i, j]² / Σ(m=1..k) Σ(n=0..h-1) Θₙ[i, m]², for h = 1..`periods`

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
            column (str, optional): The historical data column to model. Defaults to
            "Return".
            lags (int, optional): The VAR order. Defaults to 1.
            periods (int, optional): The forecast horizon to decompose out to.
            Defaults to 10.
            include_benchmark (bool, optional): Whether to include "Benchmark" among
            the tickers modeled jointly. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
            results to. Defaults to None.

        Returns:
            pd.DataFrame: Each response ticker's variance-share table (one column per
            shock ticker, rows summing to 1), concatenated side by side under a
            top-level column per response ticker, indexed `1, ..., periods` (horizon).

        Notes:
        - See `get_impulse_response_function`'s `Notes` on the Cholesky ordering --
        the same identifying assumption applies here.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^VIX", "^GVZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_variance_decomposition(column="Adj Close", periods=20)
        ```

        Which returns:

        |   Horizon |   ('^VIX', '^VIX') |   ('^VIX', '^GVZ') |   ('^GVZ', '^VIX') |   ('^GVZ', '^GVZ') |
        |----------:|-------------------:|-------------------:|-------------------:|-------------------:|
        |         1 |             1      |             0      |             0.1832 |             0.8168 |
        |         2 |             1      |             0      |             0.1925 |             0.8075 |
        |         3 |             0.9999 |             0.0001 |             0.2016 |             0.7984 |
        |         4 |             0.9998 |             0.0002 |             0.2104 |             0.7896 |
        |         5 |             0.9997 |             0.0003 |             0.219  |             0.781  |
        |         6 |             0.9996 |             0.0004 |             0.2273 |             0.7727 |
        |         7 |             0.9995 |             0.0005 |             0.2354 |             0.7646 |
        |         8 |             0.9993 |             0.0007 |             0.2432 |             0.7568 |
        |         9 |             0.9991 |             0.0009 |             0.2506 |             0.7494 |
        |        10 |             0.9989 |             0.0011 |             0.2578 |             0.7422 |
        |        11 |             0.9987 |             0.0013 |             0.2647 |             0.7353 |
        |        12 |             0.9985 |             0.0015 |             0.2714 |             0.7286 |
        |        13 |             0.9983 |             0.0017 |             0.2777 |             0.7223 |
        |        14 |             0.9981 |             0.0019 |             0.2837 |             0.7163 |
        |        15 |             0.9978 |             0.0022 |             0.2895 |             0.7105 |
        |        16 |             0.9976 |             0.0024 |             0.2949 |             0.7051 |
        |        17 |             0.9974 |             0.0026 |             0.3002 |             0.6998 |
        |        18 |             0.9971 |             0.0029 |             0.3051 |             0.6949 |
        |        19 |             0.9969 |             0.0031 |             0.3098 |             0.6902 |
        |        20 |             0.9967 |             0.0033 |             0.3142 |             0.6858 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        returns = self._get_series_column(period, column, tickers)

        var_result = time_series_model.get_var_forecast(
            returns[tickers], lags=lags, forecast_steps=1
        )
        result = time_series_model.get_variance_decomposition(
            var_result, periods=periods
        )

        return apply_rounding(
            time_series_model.variance_decomposition_summary_table(result),
            rounding if rounding is not None else self._rounding,
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

        The formula is as follows:

        - Rank r = the number of leading Johansen trace statistics (det_order=0, `k_ar_diff` lags) that
          exceed their critical value at `significance` (5% by default); r = 0 raises an error
        - ΔYₜ = α·β'·Yₜ₋₁ + Σ(i=1..k) Γᵢ·ΔYₜ₋ᵢ + c + εₜ, with Yₜ every ticker's `column` level (Adj Close by
          default), k = `k_ar_diff` (1 by default) and c a constant outside the cointegrating relation
          (statsmodels VECM with deterministic="co"), fitted by Maximum Likelihood
        - Forecast: the fitted recursion run forward `forecast_steps` periods, in levels

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["GOOGL", "GOOG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        # The two share classes of Alphabet are cointegrated: their prices cannot drift apart.
        toolkit.econometrics.get_vecm_forecast()
        ```

        Which returns:

        |   Step |   GOOGL |    GOOG |
        |-------:|--------:|--------:|
        |      1 | 312.569 | 313.352 |
        |      2 | 312.75  | 313.553 |
        |      3 | 312.929 | 313.747 |
        |      4 | 313.109 | 313.941 |
        |      5 | 313.289 | 314.135 |
        """
        period = period if period else "daily"
        tickers = self._get_tickers(include_benchmark=include_benchmark)
        prices = self._get_series_column(period, column, tickers)

        result = time_series_model.get_vecm_forecast(
            prices[tickers],
            k_ar_diff=k_ar_diff,
            forecast_steps=forecast_steps,
            significance=significance,
        )

        return result["forecast"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
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

        This controller method compares two ASSETS' series directly, treating one as a naive
        "forecast" proxy for the other -- a simple, tracking-error style measure of
        how closely two series move together in absolute deviation terms (e.g. a
        portfolio versus a benchmark, or one asset as a naive stand-in forecast for
        a similar one). For evaluating an actual FORECASTING MODEL (ARIMA/VAR)
        rather than one asset as a naive proxy for another, use
        `get_out_of_sample_validation` instead.

        For a fund against its index the RMSE comes close to the tracking error but is not
        the same: the tracking error (`Performance.get_tracking_error`) is the Standard
        Deviation of the return differences, which leaves out their average, while the RMSE
        includes it. The RMSE is per period, so a daily figure is usually multiplied by the
        square root of 252 to annualize it.

        The formula is as follows:

        - RMSE = √((1 / n)·Σ(aₜ - bₜ)²), with aₜ and bₜ the two tickers' `column` (Return by default) on
          the n dates where both have data

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["IVV", "VOO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_rmse(include_benchmark=True)
        ```

        Which returns:

        |                      |   RMSE |
        |:---------------------|-------:|
        | ('IVV', 'VOO')       | 0.0004 |
        | ('IVV', 'Benchmark') | 0.0009 |
        | ('VOO', 'Benchmark') | 0.001  |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        values = {
            (ticker_a, ticker_b): forecast_evaluation_model.get_rmse(
                returns[ticker_a], returns[ticker_b]
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        if not values:
            raise ValueError(
                "The RMSE compares two series, a forecast and the actual values, so it "
                "needs at least two tickers (or include_benchmark=True for the benchmark)."
            )

        result = pd.Series(values, name="RMSE")
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_mae(
        self,
        period: str | None = None,
        column: str = "Return",
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Mean Absolute Error (MAE) between every unordered pair of tickers'
        series in the Toolkit instance.

        MAE is the average absolute difference between two series.

        The formula is as follows:

        - MAE = (1 / n)·Σ|aₜ - bₜ|, with aₜ and bₜ the two tickers' `column` (Return by default) on the
          n dates where both have data

        It is expressed in the units of the series itself (e.g. a return difference) and
        answers "how far apart are the two series in a typical period". Every difference counts
        in proportion to its size, so unlike `get_rmse`, which squares the differences and is
        therefore dominated by a few large gaps, MAE is robust to occasional outliers. A
        lower value means the two tickers track each other more closely.

        This method compares two assets directly, for example how closely a fund tracks its
        index or how two related stocks move together, rather than an asset against a model's
        forecast; see `get_rmse` for the reasoning, and `get_out_of_sample_validation` for
        evaluating actual forecasts.

        Also known as: MAD (Mean Absolute Deviation).

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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

        toolkit = Toolkit(
            ["GLD", "IAU", "GLDM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_mae()
        ```

        Which returns:

        |                 |    MAE |
        |:----------------|-------:|
        | ('GLD', 'IAU')  | 0.0002 |
        | ('GLD', 'GLDM') | 0.0003 |
        | ('IAU', 'GLDM') | 0.0003 |
        """
        period = period if period else "daily"
        returns = self._get_series_column(period, column)

        values = {
            (ticker_a, ticker_b): forecast_evaluation_model.get_mae(
                returns[ticker_a], returns[ticker_b]
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }

        if not values:
            raise ValueError(
                "The MAE compares two series, a forecast and the actual values, so it "
                "needs at least two tickers (or include_benchmark=True for the benchmark)."
            )

        result = pd.Series(values, name="MAE")
        result.index = result.index.set_names(["Ticker A", "Ticker B"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

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

        The formula is as follows:

        - Training sample: the first int(`train_fraction`·n) observations (80% by default) of the
          `column` level (Adj Close by default); the remaining H observations are the holdout
        - Ŷₜ: the H-step forecast of the model fitted on the training sample only, an ARIMA(p, d, q)
          (model="arima", ARIMA(1, 1, 1) by default) or a VAR(`lags`) on the ticker and `other_tickers`
          (model="var")
        - RMSE = √((1 / H)·Σ(Yₜ - Ŷₜ)²) and MAE = (1 / H)·Σ|Yₜ - Ŷₜ|, over the holdout

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly). Defaults to
                "daily".
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
            ticker. With `model="arima"` each ticker uses its own history and a ticker
            without enough data has an empty (NaN) column.

        Raises:
            ValueError: If `model` is not "arima" or "var", or (for `model="var"`)
            fewer than 2 tickers are available to form a VAR system.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BTCUSD", "ETHUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_out_of_sample_validation()
        ```

        Which returns:

        |                      |   BTCUSD |   ETHUSD |
        |:---------------------|---------:|---------:|
        | RMSE                 | 11424.9  |  1003.11 |
        | MAE                  |  9896.95 |   842.37 |
        | Holdout Observations |   366    |   366    |
        """
        period = period if period else "daily"
        # Every series at its own observations: the ARIMA validation drops the missing
        # values per series and the VAR branch aligns its own system with dropna below.
        prices = self._get_series_column(period, column)

        tickers = self._get_tickers(include_benchmark=include_benchmark)

        if model not in ("arima", "var"):
            raise ValueError(f"model must be 'arima' or 'var', received '{model}'.")

        columns = {}
        for ticker in tickers:
            if model == "arima":
                # As in get_arima_forecast, a ticker without enough data of its own
                # gets an empty column instead of failing the other tickers.
                try:
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
                except ValueError as error:
                    logger.warning("No ARIMA validation for %s: %s", ticker, error)
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
                var_result["forecast"][ticker].to_numpy(), index=holdout.index
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

        if not columns:
            raise ValueError("None of the tickers has enough data for an ARIMA fit.")

        result = pd.DataFrame(columns).reindex(columns=tickers)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_event_study(
        self,
        event_date: str,
        dependent_ticker: str | None = None,
        column: str = "Return",
        estimation_window: int = 250,
        gap_days: int = 30,
        pre_event_days: int = 10,
        post_event_days: int = 10,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Perform a market-model event study around a single event date, following the
        methodology in MacKinlay, A.C. (1997), "Event Studies in Economics and
        Finance," Journal of Economic Literature, 35(1), 13-39 -- the canonical
        reference and still the dominant approach used to measure the stock-price
        impact of corporate events (earnings announcements, M&A deals, index
        additions/deletions, dividend changes, regulatory actions, etc.).

        A market model (`Return_t = alpha + beta * Benchmark_Return_t + e_t`) is fit
        via OLS over a clean "estimation window" ending `gap_days` before the event,
        then the Abnormal Return on each day of the "event window" around the event
        is the actual return minus the market-model-predicted expected return. These
        are cumulated into the Cumulative Abnormal Return (CAR) -- the stock-price
        impact attributable to the event, after stripping out what would have been
        expected from general market movements alone -- along with a t-test of
        whether CAR is significantly different from zero.

        The `event_date` is the first trading day on which the news can be priced, so for
        results released after the close it is the next trading day. The CAR runs from
        `pre_event_days` before to `post_event_days` after it (21 trading days by default),
        which also captures any leak ahead of the announcement. The t-test treats the
        abnormal returns as independent and uses the residual Variance of the estimation
        window, so a single event is weak evidence; studies average the CARs of many events
        of the same kind.

        The formula is as follows:

        - Market model: Rₜ = α + β·R_m,t + εₜ, fitted by OLS on the daily `column` (Return by default) of
          the ticker and the Benchmark over the `estimation_window` (250) trading days that end
          `gap_days` (30) trading days before the event
        - ARₜ = Rₜ - (α̂ + β̂·R_m,t), for t = -`pre_event_days` .. +`post_event_days` (10 and 10)
        - CAR = Σ ARₜ over the L = `pre_event_days` + `post_event_days` + 1 event-window days
        - t = CAR / √(L·σ̂²), with σ̂² = Σε̂ₜ² / (N - 2) over the N estimation-window days and a Student-t
          distribution with N - 2 degrees of freedom

        Also known as: CAR analysis, abnormal returns analysis, market model event
        study.

        Args:
            event_date (str): The date of the event (e.g. "2023-05-04"). Must fall
                within the Toolkit instance's daily historical data.
            dependent_ticker (str | None, optional): The ticker being studied.
                Defaults to the first ticker in the Toolkit instance.
            column (str, optional): The historical data column to use. Defaults to
                "Return".
            estimation_window (int, optional): Number of trading days used to
                estimate the market model. Defaults to 250 (~one trading year).
            gap_days (int, optional): Number of trading days between the end of the
                estimation window and the event date. Defaults to 30.
            pre_event_days (int, optional): Number of trading days before the event
                date included in the event window. Defaults to 10.
            post_event_days (int, optional): Number of trading days after the event
                date included in the event window. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the
                results to. Defaults to None.

        Returns:
            pd.Series: The Cumulative Abnormal Return (CAR), its t-statistic and
            p-value, the market-model alpha and beta, and the number of estimation-
            window observations used.

        Notes:
        - Always uses daily data -- event windows are measured in trading days, so
          quarterly/yearly granularity would not be meaningful here.
        - The market/benchmark return series is the Toolkit instance's `"Benchmark"`
          column.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2022-01-01",
            end_date="2023-12-31",
        )

        # NVDA's abnormal return around its May 2023 earnings release
        toolkit.econometrics.get_event_study(event_date="2023-05-25")
        ```

        Which returns:

        |                                |    Value |
        |:-------------------------------|---------:|
        | Cumulative Abnormal Return     |   0.2044 |
        | CAR t-statistic                |   1.9468 |
        | CAR p-value                    |   0.0527 |
        | Alpha                          |   0.0018 |
        | Beta                           |   2.0781 |
        | Estimation Window Observations | 250      |
        """
        dependent_ticker = dependent_ticker or self._get_tickers()[0]

        # The windows count trading days of the dependent ticker and the Benchmark
        # together, so only their common dates are kept: another ticker's missing
        # dates do not shorten or shift the estimation and event windows.
        returns = self._get_series_column(
            "daily", column, [dependent_ticker, BENCHMARK_TICKER]
        )

        result = event_study_model.get_event_study(
            returns=returns[dependent_ticker],
            market_returns=returns[BENCHMARK_TICKER],
            event_date=event_date,
            estimation_window=estimation_window,
            gap_days=gap_days,
            pre_event_days=pre_event_days,
            post_event_days=post_event_days,
        )

        summary = pd.Series(
            {
                "Cumulative Abnormal Return": result["cumulative_abnormal_return"],
                "CAR t-statistic": result["car_t_statistic"],
                "CAR p-value": result["car_p_value"],
                "Alpha": result["alpha"],
                "Beta": result["beta"],
                "Estimation Window Observations": result[
                    "estimation_window_observations"
                ],
            }
        )

        return apply_rounding(
            summary, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_markov_regime_switching(
        self,
        period: str | None = None,
        column: str = "Return",
        regimes: int = 2,
        switching_variance: bool = True,
        probabilities: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Fit a Markov regime switching model to every asset: the mean and volatility of its
        returns depend on a regime that is not observed and switches from one period to the
        next with fixed probabilities (Hamilton, 1989). With two regimes the model typically
        separates calm markets (a higher mean and a low volatility) from turbulent ones (a
        lower mean and a high volatility), and it says which regime the market is most
        likely in today, how persistent each regime is and how long it lasts on average.

        The regimes are ordered from the lowest to the highest volatility, so Regime 1 is
        the calmest. With switching_variance=False all regimes share one volatility, and
        they are ordered from the lowest to the highest mean instead, so Regime 1 is the
        one with the lowest mean. The expected duration of a regime is 1 / (1 - p), with p
        the probability of staying in it, in periods. The share is the part of the periods
        in which the regime was the most likely one.

        The likelihood of a mixture of regimes has local optima, so the model is fitted from
        its default starting values and from the best of 20 random perturbations of them,
        and the fit with the highest likelihood is kept. The random perturbations come from
        NumPy's global random state, so when competing optima are close, repeated calls can
        differ slightly; call np.random.seed beforehand for results that can be reproduced.

        The formula is as follows:

        - yₜ = μ_Sₜ + σ_Sₜ·εₜ, with yₜ the `column` series (Return by default), εₜ standard normal and
          Sₜ ∈ {1, ..., K} the regime, K = `regimes` (2 by default); σ_Sₜ = σ for every regime when
          switching_variance=False
        - P(Sₜ = j | Sₜ₋₁ = i) = pᵢⱼ, a first-order Markov chain with constant transition probabilities
        - μ, σ and pᵢⱼ are fitted by Maximum Likelihood with the Hamilton filter (statsmodels
          MarkovRegression), keeping the best of the default start and 20 random perturbations
        - Smoothed probabilities P(Sₜ = j | y₁..y_T) per date; Expected Duration = 1 / (1 - pⱼⱼ) periods;
          Share = the fraction of dates where regime j has the highest smoothed probability; Current
          Regime = that regime on the last date, with its smoothed probability

        For more information, see the following paper:

        - Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary
          Time Series and the Business Cycle." Econometrica, 57(2), 357-384.

        Also known as: Markov switching model, hidden Markov model, regime detection,
        bull and bear regimes.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to model. Defaults to "Return".
            regimes (int, optional): The number of regimes. Defaults to 2.
            switching_variance (bool, optional): Whether the volatility switches with the
                regime as well as the mean. Defaults to True.
            probabilities (bool, optional): Whether to return the smoothed probability of
                every regime per date instead of the summary. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.

        Returns:
            pd.DataFrame: Per asset the Mean, Volatility, Probability of Staying, Expected
            Duration and Share of every regime, and the Current Regime with its probability;
            or, with probabilities=True, the probability of every regime per date.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_markov_regime_switching(period="weekly")
        ```

        Which returns:

        |                                 |    AAPL |    MSFT |
        |:--------------------------------|--------:|--------:|
        | Regime 1 Mean                   |  0.009  |  0.0065 |
        | Regime 1 Volatility             |  0.0236 |  0.0217 |
        | Regime 1 Probability of Staying |  0.917  |  0.9528 |
        | Regime 1 Expected Duration      | 12.0479 | 21.194  |
        | Regime 1 Share                  |  0.5436 |  0.6411 |
        | Regime 2 Mean                   |  0.0006 |  0.0021 |
        | Regime 2 Volatility             |  0.0488 |  0.0455 |
        | Regime 2 Probability of Staying |  0.91   |  0.9209 |
        | Regime 2 Expected Duration      | 11.1105 | 12.6346 |
        | Regime 2 Share                  |  0.4564 |  0.3589 |
        | Current Regime                  |  1      |  1      |
        | Current Regime Probability      |  0.8877 |  0.8087 |
        """
        period = period if period else "daily"
        data = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        fits = {
            ticker: regime_model.get_markov_regime_switching(
                data[ticker], regimes=regimes, switching_variance=switching_variance
            )
            for ticker in data.columns
        }

        if probabilities:
            result = pd.concat(
                {ticker: smoothed for ticker, (_, smoothed) in fits.items()}, axis=1
            )
        else:
            result = pd.DataFrame(
                {ticker: summary for ticker, (summary, _) in fits.items()}
            )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_structural_breaks(
        self,
        period: str | None = None,
        column: str = "Return",
        max_breaks: int = 5,
        minimum_segment: float = 0.15,
        breaks: int | None = None,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Find the dates at which the mean of every asset's series shifts, with the Bai and
        Perron (1998, 2003) method for multiple structural breaks. For every number of breaks
        the break dates that minimize the sum of squared residuals around a mean per segment
        are found exactly (by dynamic programming), and the number of breaks is chosen by the
        Bayesian information criterion. Every segment covers at least a share of the
        observations (the integer part of that share times the number of observations, as
        in Bai and Perron), so a break is not fitted to a few outliers.

        The BIC works well for series without serial correlation, such as returns, but tends
        to choose too many breaks when the series is serially correlated (Bai and Perron,
        2003), as levels such as inflation or interest rates are: a persistent swing then
        looks like a shift in the mean. For such series, compare the segments for a few
        values of breaks rather than relying on the chosen number alone.

        Breaks show when a series changed character, such as the regimes of inflation or of
        interest rates, or a change in the average return of an asset. Unlike the Chow test,
        the break dates do not have to be known in advance.

        The formula is as follows:

        - yₜ = μⱼ + εₜ for the dates of segment j = 1, ..., m + 1, with yₜ the `column` series (Return by
          default) and m breaks
        - SSR(m) = min over the break dates of Σⱼ Σ(t in segment j) (yₜ - ȳⱼ)², found exactly by dynamic
          programming, with every segment at least h = max(⌊`minimum_segment`·T⌋, 2) observations
          (`minimum_segment` = 0.15 by default)
        - BIC(m) = T·ln(SSR(m) / T) + (2m + 1)·ln(T), for m = 0, ..., `max_breaks` (5 by default); the m
          with the lowest BIC is chosen unless `breaks` is given
        - Per segment: its Start, End, Observations, Mean ȳⱼ and Standard Deviation (ddof=1)

        For more information, see the following papers:

        - Bai, J. and Perron, P. (1998). "Estimating and Testing Linear Models with Multiple
          Structural Changes." Econometrica, 66(1), 47-78.
        - Bai, J. and Perron, P. (2003). "Computation and Analysis of Multiple Structural
          Change Models." Journal of Applied Econometrics, 18(1), 1-22.

        Also known as: Bai-Perron test, multiple structural breaks, change point detection,
        regime shifts.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to analyze. Defaults to
                "Return"; use "Adj Close" (or "Level" with standalone data) for the series
                itself, such as an inflation rate.
            max_breaks (int, optional): The most breaks considered. Defaults to 5.
            minimum_segment (float, optional): The smallest segment as a share of the
                observations (the trimming). Defaults to 0.15.
            breaks (int | None, optional): The number of breaks to find instead of choosing
                it by the BIC. Defaults to None.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.

        Returns:
            pd.DataFrame: Per asset and segment the Start, End, Observations, Mean and
            Standard Deviation, with the BIC of the chosen number of breaks.

        As an example:

        ```python
        from financetoolkit import Economics, Econometrics

        economics = Economics(start_date="1960-01-01", end_date="2025-12-31")

        inflation = economics.get_inflation_rate(countries="United States", period="yearly")

        econometrics = Econometrics(data=inflation)

        econometrics.get_structural_breaks(column="Level", max_breaks=3)
        ```

        Which returns:

        |                      | Start      | End        |   Observations |   Mean |   Standard Deviation |      BIC |
        |:---------------------|:-----------|:-----------|---------------:|-------:|---------------------:|---------:|
        | ('United States', 1) | 1960-12-31 | 1972-12-31 |             13 | 0.0282 |               0.0169 | -519.037 |
        | ('United States', 2) | 1973-12-31 | 1982-12-31 |             10 | 0.0876 |               0.0269 | -519.037 |
        | ('United States', 3) | 1983-12-31 | 2025-12-31 |             43 | 0.0285 |               0.0143 | -519.037 |
        """
        period = period if period else "daily"
        data = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        result = pd.concat(
            {
                ticker: regime_model.get_structural_breaks(
                    data[ticker],
                    max_breaks=max_breaks,
                    minimum_segment=minimum_segment,
                    breaks=breaks,
                )
                for ticker in data.columns
            }
        )
        result.index = result.index.set_names(["Asset", "Segment"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_time_varying_beta(
        self,
        dependent_tickers: str | list[str] | None = None,
        independent_ticker: str | None = None,
        period: str | None = None,
        column: str = "Return",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Estimate the alpha and beta of every asset as they change over time, with the Kalman
        filter. The regression y_t = α_t + β_t x_t + ε_t lets both coefficients take a random
        step every period; how large those steps are, and how much noise surrounds the
        regression, is estimated by maximum likelihood. A beta that drifts, such as a stock
        becoming more sensitive to the market, shows up without choosing the window of a
        rolling regression, while a beta that is constant stays flat.

        Every alpha and beta is the filtered estimate, updated with the observations up to
        its date only. The three variances that set how fast the estimates adapt are
        estimated on the whole sample, however, so the path is not entirely what would have
        been known at the time. The filter starts without prior knowledge of the
        coefficients, so the first date (or the first few, when the independent series
        starts with zeros) cannot separate the alpha from the beta and is NaN.

        By default every asset is regressed on the Benchmark of the Toolkit (or, with
        standalone data, on the second series).

        The formula is as follows:

        - yₜ = αₜ + βₜ·xₜ + εₜ, εₜ ~ N(0, σ²_ε), with yₜ the asset's and xₜ the independent asset's `column`
          series (Return by default), on the dates both have
        - αₜ = αₜ₋₁ + ηₜ and βₜ = βₜ₋₁ + ζₜ, ηₜ ~ N(0, σ²_α) and ζₜ ~ N(0, σ²_β): random walks
        - σ²_ε, σ²_α and σ²_β are fitted by Maximum Likelihood with the Kalman filter (diffusely
          initialized) on the series divided by their standard deviations, and scaled back
        - Alpha and Beta are the filtered states E[αₜ, βₜ | y₁..yₜ]; the Beta Standard Error is the square
          root of the filtered variance of βₜ

        For more information, see the following book:

        - Durbin, J. and Koopman, S.J. (2012). "Time Series Analysis by State Space
          Methods." Oxford University Press, 2nd edition.

        Also known as: Kalman filter beta, rolling beta, dynamic beta, state space
        regression.

        Args:
            dependent_tickers (str | list[str] | None, optional): The assets whose beta is
                estimated. Defaults to None, which uses every asset except the independent one.
            independent_ticker (str | None, optional): The asset the beta is measured
                against. Defaults to None, which uses the Benchmark or the second series.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.

        Returns:
            pd.DataFrame: Per date and asset the Alpha, the Beta and the standard error of
            the beta, NaN on the first date(s) where they are not yet identified.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        beta = toolkit.econometrics.get_time_varying_beta(period="monthly")

        beta.xs("Beta", axis=1, level=1).iloc[12::24]
        ```

        Which returns:

        | Date    |   AAPL |   MSFT |
        |:--------|-------:|-------:|
        | 2016-02 | 1.2708 | 1.5813 |
        | 2018-02 | 1.1427 | 1.3183 |
        | 2020-02 | 1.3197 | 1.1153 |
        | 2022-02 | 1.2496 | 0.9853 |
        | 2024-02 | 1.2526 | 0.9525 |
        """
        period = period if period else "daily"
        # Every pair is aligned on its own common dates by the model.
        data = self._get_series_column(period, column)

        tickers = list(data.columns)
        if independent_ticker is None:
            independent_ticker = (
                BENCHMARK_TICKER
                if BENCHMARK_TICKER in tickers
                else tickers[1] if len(tickers) > 1 else None
            )
        if independent_ticker is None or independent_ticker not in tickers:
            raise ValueError(
                f"The independent ticker {independent_ticker} is not available; choose from "
                f"{', '.join(tickers)}."
            )
        dependent = (
            [ticker for ticker in tickers if ticker != independent_ticker]
            if dependent_tickers is None
            else (
                [dependent_tickers]
                if isinstance(dependent_tickers, str)
                else list(dependent_tickers)
            )
        )

        result = pd.concat(
            {
                ticker: state_space_model.get_time_varying_beta(
                    data[ticker], data[independent_ticker]
                )[0]
                for ticker in dependent
            },
            axis=1,
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_dcc_garch(
        self,
        period: str | None = None,
        column: str = "Return",
        parameters: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Estimate the correlation between every pair of assets as it changes over time, with
        the dynamic conditional correlation (DCC) GARCH(1,1) model of Engle (2002). Every
        asset gets its own GARCH(1,1) volatility, and the correlation of the standardized
        returns follows a GARCH-like recursion around their long-run correlation. Correlations
        between assets tend to rise in crises, exactly when diversification is needed most.
        A rolling correlation sees that late and depends on its window; the DCC model
        updates the correlation every period, with a (how strongly it reacts to news) and b
        (how persistent it is) estimated by maximum likelihood.

        The formula is as follows:

        - zᵢ,ₜ = (rᵢ,ₜ - mean(rᵢ)) / σᵢ,ₜ, with rᵢ,ₜ the `column` series (Return by default) on the dates all
          assets have and σ²ᵢ,ₜ = ω + α·uᵢ,ₜ₋₁² + β·σ²ᵢ,ₜ₋₁ each asset's GARCH(1,1) variance of the demeaned
          returns u (GARCH Persistence = α + β)
        - Qₜ = (1 - a - b)·Q̄ + a·zₜ₋₁·zₜ₋₁' + b·Qₜ₋₁, with Q̄ the sample correlation of z and Q₀ = Q̄
        - Rₜ = diag(Qₜ)^(-1/2)·Qₜ·diag(Qₜ)^(-1/2), the conditional correlations
        - a and b maximize the correlation log-likelihood -½·Σₜ (ln|Rₜ| + zₜ'·Rₜ⁻¹·zₜ), with a, b ≥ 0 and
          a + b < 1

        For more information, see the following paper:

        - Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class of
          Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models."
          Journal of Business & Economic Statistics, 20(3), 339-350.

        Also known as: DCC-GARCH, dynamic conditional correlation, time-varying
        correlation, multivariate GARCH.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The historical data column to use. Defaults to "Return".
            parameters (bool, optional): Whether to return the estimated parameters (a, b and
                the GARCH persistence of every asset) instead of the correlations. Defaults
                to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.

        Returns:
            pd.DataFrame: The conditional correlation of every pair of assets per date, or
            the parameters of the model.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT", "XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2018-01-01",
            end_date="2025-12-31",
        )

        correlations = toolkit.econometrics.get_dcc_garch()

        correlations.resample("Y").mean().round(4)
        ```

        Which returns:

        | Date   |   AAPL / MSFT |   AAPL / XOM |   MSFT / XOM |
        |:-------|--------------:|-------------:|-------------:|
        | 2018   |        0.6047 |       0.2794 |       0.2729 |
        | 2019   |        0.578  |       0.2982 |       0.2641 |
        | 2020   |        0.6665 |       0.2879 |       0.226  |
        | 2021   |        0.6199 |       0.1384 |       0.0786 |
        | 2022   |        0.7084 |       0.2283 |       0.1692 |
        | 2023   |        0.5802 |       0.1706 |       0.0386 |
        | 2024   |        0.5378 |       0.1018 |       0.0237 |
        | 2025   |        0.4636 |       0.1991 |       0.114  |
        """
        period = period if period else "daily"
        # The model aligns the assets on the dates they share, after the Benchmark is left
        # out, so the Benchmark's own gaps do not shorten the sample when it is excluded.
        data = self._filter_benchmark(
            self._get_series_column(period, column), include_benchmark=include_benchmark
        )

        correlations, estimates = state_space_model.get_dcc_garch(data)
        result = (
            pd.Series(estimates, name="Value").to_frame()
            if parameters
            else correlations
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_pairs_trading(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        window: int | None = None,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        log_prices: bool = True,
        time_series: bool = False,
        include_benchmark: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the statistics of trading the spread between every pair of assets. The
        hedge ratio follows from regressing one log price on the other; what remains is the
        spread, and when the two are cointegrated the spread reverts to its mean. Its
        z-score says how far it is from that mean in standard deviations, and threshold
        rules trade on it: short the spread (sell the first asset, buy the hedge ratio of
        the second) once the z-score rises above the entry threshold, buy it once it falls
        below minus the threshold, and close the position once the z-score crosses the exit
        threshold on its way back (below it for a short, above minus it for a long). A
        z-score that jumps past the band in one period still closes the position, and one
        that lands beyond the entry threshold on the other side reverses it. The half-life
        says how many periods the spread takes to close half its gap to the mean.

        Gatev, Goetzmann and Rouwenhorst (2006) open a position when the normalized prices
        of a pair drift two standard deviations (of their formation period) apart and close
        it when the prices cross, which is an exit threshold of 0; closing at 0.5 is a
        common variant that takes profits before the spread is fully back at its mean.

        The hedge ratio and intercept are estimated on the whole period, so the spread, and
        with it every z-score and signal, uses information from after its date. A rolling
        window only makes the mean and standard deviation of the z-score backward-looking;
        a backtest without look-ahead estimates the hedge ratio on a formation period and
        trades in the period after it. The Engle-Granger p-value says whether there is a
        mean-reverting relationship to trade at all (below 0.05).

        The formula is as follows:

        - yₜ = c + h·xₜ + sₜ, fitted by OLS on the dates both have, with yₜ and xₜ the (log, when
          log_prices=True) `column` levels (Adj Close by default): h the Hedge Ratio, c the Intercept and
          sₜ the Spread
        - Z-Scoreₜ = (sₜ - mean(s)) / std(s) (ddof=1), over the full sample or a rolling `window`
        - Signal: -1 from Z > `entry_z_score` until Z < `exit_z_score`, +1 from Z < -`entry_z_score`
          until Z > -`exit_z_score`, 0 otherwise (2.0 and 0.5 by default)
        - Half-Life = -ln(2) / ln(1 + λ), with λ from Δsₜ = a + λ·sₜ₋₁ + εₜ (infinite when λ is (nearly)
          zero or positive)
        - Cointegration P-Value: the Engle-Granger test of yₜ on xₜ

        For more information, see the following papers:

        - Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G. (2006). "Pairs Trading:
          Performance of a Relative-Value Arbitrage Rule." The Review of Financial Studies,
          19(3), 797-827.
        - Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error Correction:
          Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

        Also known as: statistical arbitrage, spread trading, z-score strategy, half-life
        of mean reversion.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or
                yearly). Defaults to "daily".
            column (str, optional): The price column. Defaults to "Adj Close".
            window (int | None, optional): The rolling window of the z-score. Defaults to
                None, which uses the full sample.
            entry_z_score (float, optional): The z-score at which a position is opened.
                Defaults to 2.0.
            exit_z_score (float, optional): The z-score whose crossing on the way back closes
                a position (0 closes when the spread is back at its mean). Defaults to 0.5.
            log_prices (bool, optional): Whether to use log prices, so the hedge ratio is an
                elasticity. Defaults to True.
            time_series (bool, optional): Whether to return the Spread, Z-Score and Signal
                per date instead of the summary. Defaults to False.
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
                assets. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None.

        Returns:
            pd.DataFrame: Per pair the Hedge Ratio, Intercept, Half-Life (in periods),
            Cointegration P-Value, the latest Z-Score and Signal (1 long, -1 short, 0 flat)
            and the Number of Trades (positions opened, a reversal counting as a new one);
            or, with time_series=True, the Spread, Z-Score and Signal per date.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO", "PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2015-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_pairs_trading(window=60)
        ```

        Which returns:

        |               |   Hedge Ratio |   Intercept |   Half-Life |   Cointegration P-Value |   Z-Score |   Signal |   Number of Trades |
        |:--------------|--------------:|------------:|------------:|------------------------:|----------:|---------:|-------------------:|
        | ('KO', 'PEP') |        0.8685 |     -0.3098 |     290.448 |                  0.8345 |     0.684 |        0 |                 42 |
        """
        period = period if period else "daily"
        # Every pair is aligned on its own common dates by the model.
        prices = self._get_series_column(period, column)

        fits = {
            (ticker_a, ticker_b): pairs_trading_model.get_pairs_trading(
                prices[ticker_a],
                prices[ticker_b],
                window=window,
                entry=entry_z_score,
                exit_=exit_z_score,
                log_prices=log_prices,
            )
            for ticker_a, ticker_b in self._all_unordered_pairs(
                include_benchmark=include_benchmark
            )
        }
        if not fits:
            raise ValueError("Pairs trading needs at least two assets.")

        if time_series:
            result = pd.concat(
                {f"{a} / {b}": series for (a, b), (series, _) in fits.items()}, axis=1
            )
        else:
            result = pd.DataFrame(
                {pair: summary for pair, (_, summary) in fits.items()}
            ).T
            result.index = result.index.set_names(["Dependent", "Independent"])

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
