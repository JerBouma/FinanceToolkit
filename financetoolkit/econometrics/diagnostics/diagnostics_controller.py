"""Diagnostic Tests Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.econometrics.diagnostics import diagnostics_model
from financetoolkit.helpers import handle_portfolio
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_period,
    handle_errors,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class DiagnosticsEconometrics:
    """
    Diagnostic tests of a series: ARCH-LM, Jarque-Bera, Ljung-Box, the variance ratio and CUSUM, and of a GARCH model: the sign bias, adjusted Pearson goodness of fit and GMM orthogonality tests of its residuals and the Nyblom test of the stability of its parameters.
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
        weighted: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate Engle's ARCH-LM test: whether the volatility of every asset's returns clusters,
        with large squared returns following large ones, or the weighted portmanteau ARCH test of it.

        The Lagrange Multiplier (LM) test of Engle (1982) regresses squared, mean-demeaned
        returns on `lags` of themselves and tests whether the resulting R-squared is
        significantly different from zero. A significant result (low p-value) indicates that the
        return series exhibits volatility clustering, and a GARCH-family model is an appropriate
        choice for it. A high p-value suggests fitting GARCH would not be meaningful, since
        there is no detectable time-varying volatility to model.

        The formula is as follows:

        - εₜ = rₜ - mean(r), the returns of `period` demeaned by their sample mean
        - Auxiliary regression: εₜ² = a₀ + Σ(i=1..q) aᵢ·εₜ₋ᵢ² + uₜ, with q = `lags` (5 by default)
        - ARCH-LM = (T - q)·R², chi-squared with q degrees of freedom, where T - q is the number of
          observations in the auxiliary regression
        - Weighted test: Q_W = T·Σ(k=1..q) ((q - k + 1) / q)·ρₖ², with ρₖ the autocorrelation of εₜ² at lag k,
          and the p-value from the Gamma distribution with shape 3(q + 1)²q / (4(2q² + 3q + 1)) and
          scale 2(2q² + 3q + 1) / (3q(q + 1)), which has the mean and variance of Q_W

        The weighted test of Fisher and Gallagher (2012) gives the first lags the most
        weight, so a test over many lags is not diluted by the later ones and keeps its power;
        it is the ARCH test R's rugarch reports for a fitted GARCH model.

        For more information about the method, see the following papers:

        - Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with
        Estimates of the Variance of United Kingdom Inflation." Econometrica, 50(4),
        987-1008.
        - Fisher, T.J. and Gallagher, C.M. (2012). "New Weighted Portmanteau Statistics for
        Time Series Goodness of Fit Testing." Journal of the American Statistical
        Association, 107(498), 777-787.

        Also known as: ARCH-LM test, Engle's ARCH test, weighted ARCH-LM test, McLeod-Li test.

        Referenced against:
        - statsmodels, het_arch: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html
        - rugarch (R), Weighted.LM.test: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L1172

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
            weighted (bool, optional): Whether to calculate the weighted portmanteau test of
                Fisher and Gallagher (2012) instead of Engle's regression. Defaults to False.

        Returns:
            pd.DataFrame: The ARCH-LM (or weighted ARCH-LM) statistic and its p-value per asset.

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

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_arch_lm_test(
            returns, lags=lags, weighted=weighted
        )

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
        Calculate the Jarque-Bera test: whether every asset's returns are normally distributed,
        judged by their skewness and kurtosis.

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

        Referenced against:
        - statsmodels, jarque_bera: https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.jarque_bera.html

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

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
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
        weighted: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate the Ljung-Box test: whether every asset's returns are autocorrelated up to a
        number of lags, or the weighted Ljung-Box test of Fisher and Gallagher (2012).

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
        - Weighted test: Q_W = n·(n + 2)·Σ(k=1..h) ((h - k + 1) / h)·ρₖ² / (n - k), with the p-value from
          the Gamma distribution with shape 3(h + 1)²h / (4(2h² + 3h + 1)) and scale
          2(2h² + 3h + 1) / (3h(h + 1)), which has the mean and variance of Q_W

        The weighted test of Fisher and Gallagher (2012) gives the first lags the most
        weight, so a test over many lags is not diluted by the later ones and keeps its power;
        it is the Ljung-Box test R's rugarch reports for a fitted GARCH model.

        For more information about the method, see the following papers:

        - Ljung, G.M., & Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time Series
        Models." Biometrika, 65(2), 297-303.
        - Fisher, T.J. and Gallagher, C.M. (2012). "New Weighted Portmanteau Statistics for
        Time Series Goodness of Fit Testing." Journal of the American Statistical
        Association, 107(498), 777-787.

        Also known as: Ljung-Box Q test, portmanteau test, weighted Ljung-Box test.

        Referenced against:
        - statsmodels, acorr_ljungbox: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
        - rugarch (R), Weighted.Box.test: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L1056

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
            weighted (bool, optional): Whether to calculate the weighted Ljung-Box test of
                Fisher and Gallagher (2012). Defaults to False.

        Returns:
            pd.DataFrame: The Ljung-Box (or weighted Ljung-Box) statistic and its p-value per
            asset.

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

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_ljung_box_test(
            returns, lags=lags, weighted=weighted
        )

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
        Calculate the Lo-MacKinlay Variance Ratio test: whether every asset's returns follow a
        random walk, or trend (momentum) or mean-revert over several periods.

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

        Also known as: Lo-MacKinlay test, VR test, random walk test, mean reversion test,
        trending or mean-reverting.

        Referenced against:
        - arch, VarianceRatio: https://bashtage.github.io/arch/unitroot/generated/arch.unitroot.VarianceRatio.html

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

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
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
        method: Literal["ols", "recursive", "chu_stinchcombe_white"] = "ols",
        time_series: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate a CUSUM test: whether the mean of every asset's returns stayed stable over time
        or broke at some point.

        Three CUSUM tests are available: the CUSUM of the OLS residuals of Ploberger and Krämer
        (1992), the CUSUM of the recursive residuals of Brown, Durbin and Evans (1975) and the
        CUSUM of the log price of Chu, Stinchcombe and White (1996).

        The CUSUM of the OLS residuals fits a constant-mean model to the returns and cumulates the (scaled)
        OLS residuals into a path that behaves like a Brownian Bridge under the null
        hypothesis of a stable mean. A stable mean keeps the path close to zero, while
        a mean shift partway through the series (a structural break, e.g. a regime
        change) drags the path away from zero; the test statistic is the maximum
        absolute value of that path.

        The CUSUM of the recursive residuals cumulates the one-step-ahead forecast errors of the mean return,
        each from the returns before it (the recursive residuals). Under a stable mean they
        are independent, so their path is a Brownian motion; a break makes the errors after
        it share a sign and the path drifts out of two lines that widen with time. It reacts
        to a break later in the sample better than the OLS version, which compares every
        return with the mean of the whole sample.

        The Chu-Stinchcombe-White CUSUM asks whether the log price has risen by more since an
        earlier date than a random walk with the volatility up to now allows, the structural
        break test López de Prado (2018) uses on prices. It is one-sided: it detects a rise,
        such as a breakout or a bubble. It can also give the statistic, its critical value and
        the verdict per date, each from the returns up to that date only.

        The formula is as follows, with rₜ the returns of `period` and n the number of returns:

        - CUSUM of the OLS residuals: êₜ = rₜ - mean(r), CUSUM = max over k of |Σ(t=1..k) êₜ| / (σ̂·√n), with
          σ̂² = Σêₜ² / (n - 1), and P-Value = P(sup|B(s)| > CUSUM) for a Brownian Bridge B (the
          Kolmogorov distribution)
        - CUSUM of the recursive residuals: wₜ = (rₜ - mean(r₁..rₜ₋₁)) / √(1 + 1 / (t - 1)) for t = 2..n, Wⱼ =
          Σ(t=2..j) wₜ / σ̂ with σ̂² = Σwₜ² / (n - 1), CUSUM = max over j of |Wⱼ| / (√(n - 1) + 2(j - 1) / √(n - 1))
          and P-Value = 2·(1 - Φ(3·CUSUM) + exp(-4·CUSUM²)·Φ(CUSUM)), the probability that a Brownian motion crosses
          the lines ±CUSUM·(√(n - 1) + 2(j - 1) / √(n - 1))
        - Chu-Stinchcombe-White CUSUM: yₜ = Σ ln(1 + rₜ), the log price, Sₙ,ₜ = (yₜ - yₙ) / (σ̂ₜ·√(t - n))
          with σ̂ₜ² the mean of the squared changes of y up to t, CUSUM = max over n < t of Sₙ,ₜ at the
          last date t and its critical value √(4.6 + ln(t - n)) at that n (5%)
        - Stability is rejected at 5% when the p-value is below 0.05 (or the CUSUM above its
          critical value)

        For more information about the method, see the following papers:

        - Ploberger, W., & Kramer, W. (1992). "The CUSUM Test with OLS Residuals."
        Econometrica, 60(2), 271-285.
        - Brown, R.L., Durbin, J. and Evans, J.M. (1975). "Techniques for Testing the
        Constancy of Regression Relationships over Time." Journal of the Royal Statistical
        Society, Series B, 37(2), 149-192.
        - Chu, C.-S.J., Stinchcombe, M. and White, H. (1996). "Monitoring Structural
        Change." Econometrica, 64(5), 1045-1065.
        - Homm, U. and Breitung, J. (2012). "Testing for Speculative Bubbles in Stock
        Markets: A Comparison of Alternative Methods." Journal of Financial Econometrics,
        10(1), 198-231.
        - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley,
        section 17.3 (17.3.2 for the Chu-Stinchcombe-White test).

        Also known as: CUSUM test, CUSUM of OLS residuals test, Ploberger-Kramer test,
        recursive CUSUM test, Brown-Durbin-Evans test, Chu-Stinchcombe-White test, CSW CUSUM,
        structural break test.

        Referenced against:
        - statsmodels, breaks_cusumolsresid (CUSUM of the OLS residuals): https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html

        Shares similar logic:
        - statsmodels, recursive_olsresiduals (CUSUM of the recursive residuals): https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.recursive_olsresiduals.html

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
            method (Literal["ols", "recursive", "chu_stinchcombe_white"], optional): The CUSUM test.
                Options are: "ols" (CUSUM of the OLS residuals), "recursive" (CUSUM of the recursive
                residuals), "chu_stinchcombe_white" (Chu-Stinchcombe-White CUSUM of the log price).
                Defaults to "ols".
            time_series (bool, optional): Whether to return the statistic, its critical value and
                whether stability is rejected per date (method="chu_stinchcombe_white" only, as the
                other two scale their path with the whole sample). Defaults to False.

        Returns:
            pd.DataFrame: The CUSUM statistic, its p-value, the number of observations used, the
            1%/5%/10% critical values, and whether stability is rejected at the 5% level, per
            asset; for the Chu-Stinchcombe-White CUSUM the statistic, its 5% critical value, the
            observations and the verdict at the last date, or per date. An
            asset with fewer than 10 observations has NaN values.

        Notes:
        - The method retrieves historical return data based on the specified `period` and runs the
        CUSUM test for each asset in the Toolkit instance.
        - Unlike statsmodels' recursive_olsresiduals, which scales the recursive path with the
        standard deviation of the recursive residuals around their mean, the path here uses the
        residual variance of the full-sample fit, as Brown, Durbin and Evans (1975) do. The Chu-Stinchcombe-White
        statistic divides by the volatility σ̂ₜ and σ̂ₜ² averages the t - 1 squared changes, as in
        López de Prado (2018, section 17.3.2).
        - The critical value of the Chu-Stinchcombe-White test is the 5% value Homm and Breitung
        (2012) simulated; it has no p-value.

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

        |                       |        BA |    AIR.PA |
        |:----------------------|----------:|----------:|
        | CUSUM Statistic       |    0.6789 |    0.4439 |
        | P-Value               |    0.746  |    0.9892 |
        | Observations          | 1254      | 1280      |
        | Critical Value 1%     |    1.63   |    1.63   |
        | Critical Value 5%     |    1.36   |    1.36   |
        | Critical Value 10%    |    1.22   |    1.22   |
        | Reject Stability (5%) |    0      |    0      |
        """
        period = period if period else "daily"

        check_period(period, ["daily", "weekly", "monthly", "quarterly", "yearly"])
        if (
            period == "daily"
            and within_period
            and self._historical_data["intraday"].empty
        ):
            raise InvalidArgumentError(
                "Intraday data is required for within-period daily calculations. Either set "
                "within_period=False or initialise the Toolkit with an intraday_period."
            )

        returns = self._filter_benchmark(
            self._get_returns(period, within_period),
            include_benchmark=include_benchmark,
        )

        result = diagnostics_model.get_cusum_test(
            returns, method=method, time_series=time_series
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_sign_bias_test(
        self,
        period: str | None = None,
        method: str = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the sign bias test of Engle and Ng (1993): whether a fall raises the volatility
        of every asset by more than its GARCH-family model expects (sign bias), and whether large
        falls or large rises do (negative and positive size bias).

        A symmetric GARCH(1, 1) lets a fall and a rise of the same size raise the volatility
        equally. In equity returns falls tend to raise it by more (the leverage effect), so
        after a fall the squared standardized residuals stay larger than the model expects.
        The test regresses them on the sign and the size of the shock before them; a
        significant sign or negative size bias says the model misses that asymmetry and an
        asymmetric model, such as GJR-GARCH or APARCH, fits better.

        The formula is as follows:

        - eₜ the residuals of the `mean` model and zₜ = eₜ / σₜ the standardized residuals of the
          GARCH-family `method` (GARCH(1, 1) by default) fitted to the returns of `period`, see
          `toolkit.risk.get_garch`
        - zₜ² = c₀ + c₁·S⁻ₜ₋₁ + c₂·S⁻ₜ₋₁·eₜ₋₁ + c₃·S⁺ₜ₋₁·eₜ₋₁ + uₜ by OLS, with S⁻ₜ₋₁ = 1 when eₜ₋₁ < 0 and 0
          otherwise and S⁺ₜ₋₁ = 1 - S⁻ₜ₋₁
        - Sign Bias, Negative Size Bias and Positive Size Bias: the t-statistics of c₁, c₂ and c₃, with
          two-sided p-values from the t-distribution with n - 4 degrees of freedom
        - Joint Statistic = n·R², chi-squared with 3 degrees of freedom, with n the observations of the
          regression

        For more information about the method, see the following paper:

        - Engle, R.F. and Ng, V.K. (1993). "Measuring and Testing the Impact of News on
        Volatility." The Journal of Finance, 48(5), 1749-1778.

        Also known as: sign bias test, Engle-Ng test, size bias test, leverage effect test,
        asymmetric volatility test, GARCH residual diagnostics.

        Referenced against:
        - rugarch (R), signbias: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L195

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            method (str, optional): The GARCH-family model fitted to the returns, as in
                `toolkit.risk.get_garch`. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch"
                (TGARCH), "ngarch" (NGARCH), "avgarch" (AVGARCH), "nagarch" (NAGARCH), "component"
                (component GARCH), "figarch" (FIGARCH(1, d, 1)). Defaults to "garch".
            distribution (str, optional): The innovation distribution of the model. Options are:
                "normal" (normal), "t" (Student-t), "skewt" (skewed Student-t of Hansen),
                "skewt_fernandez_steel" (skewed Student-t of Fernández and Steel), "ged" (generalized
                error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
            mean (str, optional): The mean model. Options are: "zero" (the returns themselves are
                the shocks), "constant" (constant mean), "arma" (ARMA(1, 1)), "in_mean"
                (GARCH-in-mean). Defaults to "zero".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The t-statistic and p-value of the sign bias, negative size bias and
            positive size bias, the joint statistic and its p-value, and the observations of the
            regression, per asset.

        Notes:
        - Every asset is fitted on its own returns, as in `toolkit.risk.get_garch`; the
          fit takes a second or more per asset of daily returns.
        - The t-statistics reproduce those of the rugarch vignette.
        - Unlike rugarch, which reports the absolute t-statistics and a Wald statistic as the joint
          test, the signed t-statistics show the direction (a negative size bias is negative when
          larger falls raise the volatility by more) and the joint statistic is the n·R² of
          Engle and Ng (1993).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_sign_bias_test()
        ```

        Which returns:

        |                                |     ^GSPC |       GLD |
        |:-------------------------------|----------:|----------:|
        | Sign Bias t-Statistic          |    3.9418 |    0.6116 |
        | Sign Bias P-Value              |    0.0001 |    0.5409 |
        | Negative Size Bias t-Statistic |   -0.0432 |    1.864  |
        | Negative Size Bias P-Value     |    0.9656 |    0.0624 |
        | Positive Size Bias t-Statistic |   -0.0972 |    0.066  |
        | Positive Size Bias P-Value     |    0.9225 |    0.9474 |
        | Joint Statistic                |   27.0353 |    3.8098 |
        | Joint P-Value                  |    0      |    0.2827 |
        | Observations                   | 2512      | 2512      |
        """
        returns = self._filter_benchmark(
            self._get_returns(period if period else "daily", within_period=False),
            include_benchmark=include_benchmark,
        )
        residuals, standardized_residuals, _ = self._get_garch_residuals(
            returns, method, distribution, mean
        )

        result = diagnostics_model.get_sign_bias_test(residuals, standardized_residuals)

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_pearson_goodness_of_fit_test(
        self,
        period: str | None = None,
        bins: int = 20,
        method: str = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the adjusted Pearson goodness of fit test of Palm (1996) and Vlaar and Palm
        (1993): whether the standardized residuals of every asset's GARCH-family model follow the
        innovation distribution the model assumes.

        Each standardized residual is turned into the probability of a value at most as
        large under the fitted distribution, which is uniform between 0 and 1 when the
        distribution fits. The probabilities are counted in equally wide bins and the counts
        compared with the equal number every bin should hold. Daily returns have fatter tails
        than a GARCH model with normal innovations allows, so the test usually rejects the
        normal distribution; a Student-t or skewed Student-t fits better
        when its p-value is larger.

        The formula is as follows:

        - zₜ the standardized residuals of the GARCH-family `method` (GARCH(1, 1) by default) with the
          `distribution` and `mean` model fitted to the returns of `period`, see `toolkit.risk.get_garch`
        - uₜ = F(zₜ), with F the distribution function of the fitted standardized innovation distribution
          (with its fitted Shape and Skew)
        - nᵢ = the number of uₜ in [(i - 1) / g, i / g) for i = 1..g, with g = `bins` (20 by default)
        - Pearson Statistic = Σ(i=1..g) (nᵢ - n / g)² / (n / g), chi-squared with g - 1 degrees of freedom

        For more information about the method, see the following papers:

        - Palm, F.C. (1996). "GARCH Models of Volatility." In Handbook of Statistics, Vol.
        14, Elsevier, 209-240.
        - Vlaar, P.J.G. and Palm, F.C. (1993). "The Message in Weekly Exchange Rates in the
        European Monetary System: Mean Reversion, Conditional Heteroscedasticity, and
        Jumps." Journal of Business & Economic Statistics, 11(3), 351-360.

        Also known as: adjusted Pearson goodness of fit test, chi-squared goodness of fit test,
        probability integral transform test, GARCH distribution test, GARCH residual
        diagnostics.

        Referenced against:
        - rugarch (R), gof: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L223

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            bins (int, optional): The number of equally wide bins of the probabilities. Defaults to 20.
            method (str, optional): The GARCH-family model fitted to the returns, as in
                `toolkit.risk.get_garch`. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch"
                (TGARCH), "ngarch" (NGARCH), "avgarch" (AVGARCH), "nagarch" (NAGARCH), "component"
                (component GARCH), "figarch" (FIGARCH(1, d, 1)). Defaults to "garch".
            distribution (str, optional): The innovation distribution of the model and of the test.
                Options are: "normal" (normal), "t" (Student-t), "skewt" (skewed
                Student-t of Hansen), "skewt_fernandez_steel" (skewed Student-t of Fernández and
                Steel), "ged" (generalized error distribution), "johnson_su" (Johnson SU). Defaults
                to "normal".
            mean (str, optional): The mean model. Options are: "zero" (the returns themselves are
                the shocks), "constant" (constant mean), "arma" (ARMA(1, 1)), "in_mean"
                (GARCH-in-mean). Defaults to "zero".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Pearson statistic, its p-value, the number of bins and the
            observations, per asset.

        Notes:
        - Every asset is fitted on its own returns, as in `toolkit.risk.get_garch`; the
          fit takes a second or more per asset of daily returns.
        - The parameters of the distribution are estimated with the model, which makes the
          chi-squared p-value somewhat too large (Palm, 1996); read it as a guide.
        - rugarch reports the test for 20, 30, 40 and 50 bins.
        - Unlike rugarch, which leaves the empty bins above the highest occupied bin out of the
          sum, every empty bin counts here.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_pearson_goodness_of_fit_test(distribution="t")
        ```

        Which returns:

        |                   |     ^GSPC |       GLD |
        |:------------------|----------:|----------:|
        | Pearson Statistic |   82.4238 |   44.302  |
        | P-Value           |    0      |    0.0009 |
        | Bins              |   20      |   20      |
        | Observations      | 2513      | 2513      |
        """
        returns = self._filter_benchmark(
            self._get_returns(period if period else "daily", within_period=False),
            include_benchmark=include_benchmark,
        )
        _, standardized_residuals, parameters = self._get_garch_residuals(
            returns, method, distribution, mean
        )

        result = pd.DataFrame(
            {
                ticker: diagnostics_model.get_pearson_goodness_of_fit_test(
                    standardized_residuals[ticker],
                    bins=bins,
                    distribution=distribution,
                    parameters=parameters[ticker],
                )
                for ticker in standardized_residuals.columns
            }
        )

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_nyblom_test(
        self,
        period: str | None = None,
        method: str = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the parameter stability test of Nyblom (1989) and Hansen (1992): whether the
        parameters of every asset's GARCH-family model stayed the same over the sample, or the
        volatility process changed (a calmer or more turbulent regime, a
        change in how quickly shocks fade)?

        At the maximum of the likelihood the scores of every parameter (the derivatives of
        the log-likelihood of each return) sum to zero over the whole sample. When a parameter
        changes during the sample, their running sum wanders far from zero in between. The
        test measures that wandering for every parameter on its own and for all of them
        together, relative to the variance of the scores.

        The formula is as follows:

        - sₜ the scores of return t: the derivatives of its log-likelihood with respect to the
          parameters of the GARCH-family `method` (GARCH(1, 1) by default) with the `distribution` and
          `mean` model fitted to the returns of `period`, see `toolkit.risk.get_garch_parameters`
        - Sₜ = Σ(i=1..t) sᵢ and V = Σₜ sₜ·sₜ', with T the number of returns in the likelihood
        - Individual Statistic of parameter j = Σₜ Sₜ,ⱼ² / (T·Vⱼⱼ)
        - Joint Statistic = Σₜ Sₜ'·V⁻¹·Sₜ / T
        - P-values from the limiting distribution, Σ(j=1..k) ∫ Bⱼ(r)² dr over k independent Brownian
          bridges, with k = 1 for an individual statistic and the number of parameters for the joint
          statistic, by Imhof's (1961) integral

        Large statistics (small p-values) say the parameter is not stable. A parameter that the
        fit leaves on a bound of its search (such as an Omega or Alpha of 0) has scores that do
        not sum to zero, which inflates its statistic and the joint statistic.

        For more information about the method, see the following papers:

        - Nyblom, J. (1989). "Testing for the Constancy of Parameters Over Time." Journal of
        the American Statistical Association, 84(405), 223-230.
        - Hansen, B.E. (1992). "Testing for Parameter Instability in Linear Models." Journal
        of Policy Modeling, 14(4), 517-533.

        Also known as: Nyblom test, Hansen stability test, parameter constancy test, parameter
        instability test, GARCH stability test, structural change in GARCH.

        Shares similar logic:
        - rugarch (R), nyblom: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L110

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            method (str, optional): The GARCH-family model fitted to the returns, as in
                `toolkit.risk.get_garch`. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch"
                (TGARCH), "ngarch" (NGARCH), "avgarch" (AVGARCH), "nagarch" (NAGARCH), "component"
                (component GARCH), "figarch" (FIGARCH(1, d, 1)). Defaults to "garch".
            distribution (str, optional): The innovation distribution of the model. Options are:
                "normal" (normal), "t" (Student-t), "skewt" (skewed Student-t of Hansen),
                "skewt_fernandez_steel" (skewed Student-t of Fernández and Steel), "ged" (generalized
                error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
            mean (str, optional): The mean model. Options are: "zero" (the returns themselves are
                the shocks), "constant" (constant mean), "arma" (ARMA(1, 1)), "in_mean"
                (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The joint statistic, its p-value and 5% critical value, the statistic and
            p-value of every parameter, the 5% critical value of one parameter and the observations,
            per asset.

        Notes:
        - Every asset is fitted on its own returns, as in `toolkit.risk.get_garch`; the fit takes a
          second or more per asset of daily returns.
        - The scores are central finite differences of the log-likelihood, as for the robust
          standard errors of the diagnostics of `toolkit.risk.get_garch_parameters`.
        - Unlike rugarch, which compares the statistics with Hansen's tabulated critical values
          (0.470 for one parameter at 5%), the critical values here are those of the limiting
          distribution itself (0.461 for one parameter, the 5% point of the Cramér-von Mises
          distribution).
        - The test is conservative on a fitted GARCH: on 500 simulated GARCH(1, 1) samples of 1,000
          observations with normal innovations, refitted each time, the joint test at 5% rejects about 2% of
          the time and those of Omega, Alpha and Beta 2% to 2.6%, so a rejection is a strong signal and a
          parameter change of moderate size can go unnoticed.
        - The realized GARCH needs a realized measure the test does not take, so it is not among
          the models.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_nyblom_test()
        ```

        Which returns:

        |                                |     ^GSPC |       GLD |
        |:-------------------------------|----------:|----------:|
        | Joint Statistic                |    1.2495 |    1.2938 |
        | Joint P-Value                  |    0.0164 |    0.0134 |
        | Joint Critical Value (5%)      |    1.0002 |    1.0002 |
        | Omega Statistic                |    0.4851 |    0.4856 |
        | Omega P-Value                  |    0.0435 |    0.0433 |
        | Alpha Statistic                |    0.3316 |    0.4274 |
        | Alpha P-Value                  |    0.1104 |    0.0612 |
        | Beta Statistic                 |    0.682  |    0.3219 |
        | Beta P-Value                   |    0.0141 |    0.1174 |
        | Individual Critical Value (5%) |    0.4614 |    0.4614 |
        | Observations                   | 2513      | 2513      |
        """
        garch_model.validate_garch_family(method, distribution, mean)
        returns = self._filter_benchmark(
            self._get_returns(period if period else "daily", within_period=False),
            include_benchmark=include_benchmark,
        )
        names = garch_model.get_garch_family_parameter_names(method, distribution, mean)

        result = {}
        for ticker in returns.columns:
            values = returns[ticker].dropna().to_numpy(dtype=float)
            weights = garch_model.get_garch_family_weights(
                values, method, distribution, mean
            )
            if not np.all(np.isfinite(weights)):
                raise ValueError(f"The {method} model could not be fitted to {ticker}.")
            result[ticker] = diagnostics_model.get_nyblom_test(
                garch_model.get_garch_family_scores(
                    values, weights, method, distribution, mean
                ),
                names,
            )

        return apply_rounding(
            pd.DataFrame(result), rounding if rounding is not None else self._rounding
        )

    @handle_portfolio
    @handle_errors
    def get_gmm_moment_test(
        self,
        period: str | None = None,
        lags: int = 1,
        method: str = "garch",
        distribution: str = "normal",
        mean: str = "zero",
        include_benchmark: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the GMM orthogonality test of Hansen (1982): whether the standardized residuals
        of every asset's GARCH-family model have the mean, variance, skewness and kurtosis of the
        innovation distribution the model assumes, with squares, cubes and fourth powers free of
        autocorrelation.

        A correctly specified model leaves standardized residuals that are independent draws
        from its innovation distribution, so a set of moments has a known value. Each moment is
        tested on its own with a t-statistic; the autocorrelations of the squares (Q2: volatility
        clustering the model misses), cubes (Q3) and fourth powers (Q4) and all conditions
        together are tested with Wald statistics.

        The formula is as follows:

        - zₜ the standardized residuals of the GARCH-family `method` (GARCH(1, 1) by default) with the
          `distribution` and `mean` model fitted to the returns of `period`, see `toolkit.risk.get_garch`,
          and s and κ the skewness and kurtosis (not excess) of the fitted innovation distribution
        - Mean: E[zₜ] = 0; Variance: E[zₜ² - 1] = 0; Skewness: E[zₜ³ - s] = 0; Kurtosis: E[zₜ⁴ - κ] = 0
        - Q2: E[(zₜ² - 1)(zₜ₋ⱼ² - 1)] = 0, Q3: E[(zₜ³ - s)(zₜ₋ⱼ³ - s)] = 0 and Q4: E[(zₜ⁴ - κ)(zₜ₋ⱼ⁴ - κ)] = 0
          for j = 1, ..., `lags`
        - gₜ the conditions of return t over the T returns after the first `lags` and ḡ their average
        - S their covariance under the null of independent innovations: mₐ₊ᵦ - mₐ·mᵦ between the Mean, Variance,
          Skewness and Kurtosis conditions (powers a and b), with mₖ = E[zᵏ] of the fitted innovation distribution,
          (E[aₜ·cₜ])² between the products of the centred powers a and c at the same lag, and 0 between different
          lags and between the moments and the products
        - t-Statistic = ḡᵢ / √(Sᵢᵢ / T), with two-sided p-values from the normal distribution
        - Q2, Q3 and Q4 Statistic = T·ḡ'·S⁻¹·ḡ over their `lags` conditions, chi-squared with `lags`
          degrees of freedom; Joint Statistic over all 4 + 3·`lags` conditions, chi-squared with 4 + 3·`lags`
          degrees of freedom

        For more information about the method, see the following paper:

        - Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of Moments
        Estimators." Econometrica, 50(4), 1029-1054.

        Also known as: GMM orthogonality test, moment test, Hansen J test of standardized
        residuals, GARCH residual diagnostics, conditional moment test.

        Shares similar logic:
        - rugarch (R), GMMTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L534

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, monthly, quarterly, or yearly).
                Defaults to "daily".
            lags (int, optional): The number of lags of the Q2, Q3 and Q4 conditions. Defaults to 1.
            method (str, optional): The GARCH-family model fitted to the returns, as in
                `toolkit.risk.get_garch`. Options are: "garch" (GARCH(1, 1)), "gjr" (GJR-GARCH(1, 1, 1)),
                "egarch" (EGARCH(1, 1)), "igarch" (IGARCH(1, 1)), "aparch" (APARCH(1, 1)), "tgarch"
                (TGARCH), "ngarch" (NGARCH), "avgarch" (AVGARCH), "nagarch" (NAGARCH), "component"
                (component GARCH), "figarch" (FIGARCH(1, d, 1)). Defaults to "garch".
            distribution (str, optional): The innovation distribution of the model and of its
                skewness and kurtosis. Options are: "normal" (normal), "t" (Student-t), "skewt" (skewed
                Student-t of Hansen), "skewt_fernandez_steel" (skewed Student-t of Fernández and
                Steel), "ged" (generalized error distribution), "johnson_su" (Johnson SU). Defaults
                to "normal".
            mean (str, optional): The mean model. Options are: "zero" (the returns themselves are
                the shocks), "constant" (constant mean), "arma" (ARMA(1, 1)), "in_mean"
                (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
            include_benchmark (bool, optional): Whether to include "Benchmark" among the
            assets tested. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The t-statistic and p-value of the Mean, Variance, Skewness and Kurtosis
            conditions, the Q2, Q3, Q4 and Joint statistics with their p-values, the lags and the
            observations, per asset.

        Notes:
        - Every asset is fitted on its own returns, as in `toolkit.risk.get_garch`.
        - The conditions M1 to M4, Q2 to Q4 and the joint J test are those rugarch's
          documentation of GMMTest defines.
        - The parameters are estimated, which the covariance of the conditions leaves out (as in
          rugarch), so read the p-values as a guide.
        - The conditions are correlated (zₜ with zₜ³, zₜ² with zₜ⁴, Q2 with Q4), so the joint test can reject
          while none of the individual tests does: a negative skewness without the fatter tails that usually
          come with it is such a combination.
        - The covariance of the conditions is the one the null implies, where rugarch takes the sample covariance
          of all conditions. That sample covariance rests on 8th and 16th moments of the residuals: on simulated
          normal innovations its joint test at 5% rejects about 20% of the time at 2,500 observations, against
          about 5% here (7% at 1,000).
        - The Kurtosis condition needs a finite 8th moment of the innovations (a Student-t with a Shape above
          8); with a fitted Shape of 8 or less its variance is the sample one, and its test and the joint test
          reject too often.
        - On the residuals of a fitted GARCH(1, 1) with normal innovations, the Variance condition is close to
          zero by construction (the score of Omega is a weighted sum of zₜ² - 1), so its test, and the Kurtosis
          test that moves with it, hardly ever reject.
        - A Student-t with a Shape of 4 or less has no kurtosis (3 or less: no skewness); its Kurtosis
          and Q4 (Skewness and Q3) conditions are then NaN and left out of the joint statistic.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["^GSPC", "GLD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        toolkit.econometrics.get_gmm_moment_test(distribution="t")
        ```

        Which returns:

        |                      |     ^GSPC |       GLD |
        |:---------------------|----------:|----------:|
        | Mean t-Statistic     |    2.8645 |    3.1991 |
        | Mean P-Value         |    0.0042 |    0.0014 |
        | Variance t-Statistic |    0.0518 |   -0.2087 |
        | Variance P-Value     |    0.9587 |    0.8347 |
        | Skewness t-Statistic |   -3.1211 |   -0.0336 |
        | Skewness P-Value     |    0.0018 |    0.9732 |
        | Kurtosis t-Statistic |   -0.9237 |   -2.5345 |
        | Kurtosis P-Value     |    0.3556 |    0.0113 |
        | Q2 Statistic         |    0.8937 |    0.3027 |
        | Q2 P-Value           |    0.3445 |    0.5822 |
        | Q3 Statistic         |    0.0555 |    0.0347 |
        | Q3 P-Value           |    0.8138 |    0.8522 |
        | Q4 Statistic         |    0.0295 |    0.0019 |
        | Q4 P-Value           |    0.8635 |    0.9654 |
        | Joint Statistic      |  -26.0374 |  386.094  |
        | Joint P-Value        |    1      |    0      |
        | Lags                 |    1      |    1      |
        | Observations         | 2512      | 2512      |
        """
        returns = self._filter_benchmark(
            self._get_returns(period if period else "daily", within_period=False),
            include_benchmark=include_benchmark,
        )
        _, standardized_residuals, parameters = self._get_garch_residuals(
            returns, method, distribution, mean
        )

        result = {
            ticker: diagnostics_model.get_gmm_moment_test(
                standardized_residuals[ticker], lags, distribution, parameters[ticker]
            )
            for ticker in standardized_residuals.columns
        }

        return apply_rounding(
            pd.DataFrame(result), rounding if rounding is not None else self._rounding
        )

    @staticmethod
    def _get_garch_residuals(
        returns: pd.DataFrame, method: str, distribution: str, mean: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, ...]]]:
        """
        The residuals of the mean model, the standardized residuals and the fitted parameters
        of the innovation distribution of a GARCH-family model fitted to every asset's own
        returns, as `toolkit.risk.get_garch` fits it.
        """
        garch_model.validate_garch_family(method, distribution, mean)
        distribution_parameters = len(
            garch_model.GARCH_FAMILY_DISTRIBUTION_PARAMETERS[distribution]
        )

        residuals, standardized_residuals, parameters = {}, {}, {}
        for ticker in returns.columns:
            values = returns[ticker].dropna()
            weights = garch_model.get_garch_family_weights(
                values.to_numpy(dtype=float), method, distribution, mean
            )
            if not np.all(np.isfinite(weights)):
                raise ValueError(f"The {method} model could not be fitted to {ticker}.")

            arguments = {
                "weights": weights,
                "method": method,
                "distribution": distribution,
                "mean": mean,
            }
            variance = garch_model.get_garch_family(
                values.to_numpy(dtype=float), output="variance", **arguments
            )
            standardized = garch_model.get_garch_family(
                values.to_numpy(dtype=float),
                output="standardized_residuals",
                **arguments,
            )
            standardized_residuals[ticker] = pd.Series(standardized, index=values.index)
            residuals[ticker] = standardized_residuals[ticker] * np.sqrt(variance)
            parameters[ticker] = tuple(
                float(weight)
                for weight in weights[len(weights) - distribution_parameters :]
            )

        return (
            pd.DataFrame(residuals),
            pd.DataFrame(standardized_residuals),
            parameters,
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
