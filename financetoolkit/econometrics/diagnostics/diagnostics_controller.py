"""Diagnostic Tests Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.econometrics.base.base_controller import EconometricsBase
from financetoolkit.econometrics.diagnostics import diagnostics_model
from financetoolkit.helpers import handle_portfolio
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class DiagnosticsEconometrics(EconometricsBase):
    """
    Diagnostic tests of a series: ARCH-LM, Jarque-Bera, Ljung-Box, the variance ratio and CUSUM.
    """

    @handle_portfolio
    @handle_errors
    def get_arch_lm_test(
        self,
        period: str | None = None,
        within_period: bool = False,
        lags: int = 5,
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        show_columns: str | list[str] | None = None,
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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The CUSUM statistic, its p-value, the number of observations used, the
            1%/5%/10% critical boundary values, and whether stability is rejected at the 5% level,
            per asset. An asset with fewer than 10 observations has NaN values.

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
            self._data_cache[key] = (
                self._within_historical_data[period]["Return"]
                .loc[self._start_date : self._end_date]
                .dropna(how="all")
            )

        return self._data_cache[key]
