"""Risk Diagnostics Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.stats.stattools import jarque_bera

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2

# The Variance Ratio test requires at least a two-period compounding horizon.
MINIMUM_VARIANCE_RATIO_Q = 2

# `statsmodels.stats.stattools.jarque_bera` requires at least 2 observations.
MINIMUM_JARQUE_BERA_OBSERVATIONS = 2

# The CUSUM test needs at least a handful of OLS residuals to be meaningful.
MINIMUM_CUSUM_OBSERVATIONS = 3

# The conventional 5% significance level used for the "Reject" boolean flags.
SIGNIFICANCE_LEVEL = 0.05


def get_arch_lm_test(
    returns: pd.Series | pd.DataFrame, lags: int = 5
) -> pd.Series | pd.DataFrame:
    """
    Calculate Engle's Lagrange Multiplier (LM) test for ARCH effects, via
    `statsmodels.stats.diagnostic.het_arch`.

    The test regresses squared, mean-demeaned returns on `lags` of themselves and
    tests whether the resulting R-squared is significantly different from zero. Under
    the null hypothesis of no ARCH effects, `n * R-squared` is asymptotically
    chi-squared distributed with `lags` degrees of freedom.

    A significant result (low p-value) indicates that the return series exhibits
    volatility clustering, and a GARCH-family model is an appropriate choice for it.

    For more information about the method, see the following paper:

    - Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates
    of the Variance of United Kingdom Inflation." Econometrica, 50(4), 987-1008.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        lags (int): The number of lags to test for ARCH effects. Defaults to 5.

    Returns:
        pd.Series | pd.DataFrame: The ARCH-LM statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_arch_lm_test(returns.loc[sub_period], lags=lags)

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_arch_lm_test(returns[column], lags=lags)
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        residuals = (returns - returns.mean()).dropna().to_numpy()

        if len(residuals) <= lags + 1:
            return pd.Series({"ARCH-LM Statistic": np.nan, "P-Value": np.nan})

        try:
            lm_statistic, p_value, _, _ = het_arch(residuals, nlags=lags)
        except (ValueError, np.linalg.LinAlgError):
            return pd.Series({"ARCH-LM Statistic": np.nan, "P-Value": np.nan})

        return pd.Series({"ARCH-LM Statistic": lm_statistic, "P-Value": p_value})

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_jarque_bera_test(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Jarque-Bera test for normality, via
    `statsmodels.stats.stattools.jarque_bera`.

    The test combines sample skewness and excess kurtosis into a single statistic
    that is asymptotically chi-squared distributed with 2 degrees of freedom under
    the null hypothesis that the data is normally distributed:

    - JB = (n / 6) * (S^2 + (K^2) / 4)

    Where `S` is the skewness, `K` is the excess (Fisher) kurtosis and `n` is the
    number of observations.

    A significant result (low p-value) indicates that returns are not normally
    distributed, which is relevant when choosing between e.g. gaussian and Student-T
    based Value at Risk models.

    For more information about the method, see the following paper:

    - Jarque, C.M. and Bera, A.K. (1987). "A Test for Normality of Observations and
    Regression Residuals." International Statistical Review, 55(2), 163-172.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: The Jarque-Bera statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_jarque_bera_test(returns.loc[sub_period])

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_jarque_bera_test(returns[column])
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        values = returns.dropna().to_numpy()

        if len(values) < MINIMUM_JARQUE_BERA_OBSERVATIONS:
            return pd.Series({"Jarque-Bera Statistic": np.nan, "P-Value": np.nan})

        jarque_bera_statistic, p_value, _, _ = jarque_bera(values)

        return pd.Series(
            {"Jarque-Bera Statistic": jarque_bera_statistic, "P-Value": p_value}
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_ljung_box_test(
    returns: pd.Series | pd.DataFrame, lags: int = 10
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Ljung-Box test for autocorrelation, via
    `statsmodels.stats.diagnostic.acorr_ljungbox`.

    The test aggregates the squared Autocorrelation Function up to lag `h` into a
    single statistic that is asymptotically chi-squared distributed with `h` degrees
    of freedom under the null hypothesis that the series exhibits no autocorrelation
    up to that lag:

    - Q = n * (n + 2) * SUM_{k=1}^{h} (rho_k^2 / (n - k))

    Where `rho_k` is the Autocorrelation at lag `k` and `n` is the number of observations.

    A significant result (low p-value) indicates that the series is autocorrelated, which is
    relevant both as a standalone diagnostic (e.g. to check whether a return series follows a
    random walk) and as a residual diagnostic after fitting a model (e.g. checking that GARCH
    residuals are no longer autocorrelated).

    For more information about the method, see the following paper:

    - Ljung, G.M., & Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time Series Models."
    Biometrika, 65(2), 297-303.

    Also known as: Ljung-Box Q test, portmanteau test.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        lags (int): The number of lags to test for autocorrelation up to. Defaults to 10.

    Returns:
        pd.Series | pd.DataFrame: The Ljung-Box statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_ljung_box_test(returns.loc[sub_period], lags=lags)

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_ljung_box_test(returns[column], lags=lags)
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        values = returns.dropna()
        n = len(values)

        if n <= lags + 1:
            return pd.Series({"Ljung-Box Statistic": np.nan, "P-Value": np.nan})

        try:
            result = acorr_ljungbox(values.to_numpy(), lags=[lags])
        except (ValueError, np.linalg.LinAlgError):
            return pd.Series({"Ljung-Box Statistic": np.nan, "P-Value": np.nan})

        return pd.Series(
            {
                "Ljung-Box Statistic": float(result["lb_stat"].iloc[0]),
                "P-Value": float(result["lb_pvalue"].iloc[0]),
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_variance_ratio_test(
    returns: pd.Series | pd.DataFrame, q: int = 2
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Lo-MacKinlay Variance Ratio test for the random walk hypothesis.

    The test compares the Variance of `q`-period (overlapping) compounded returns to `q` times
    the Variance of single-period returns. Under the random walk hypothesis (i.e. returns have no
    autocorrelation), these two should be equal and the Variance Ratio should be 1:

    - VR(q) = Var(r_t(q)) / (q * Var(r_t))

    Where `r_t(q)` is the sum of `q` consecutive single-period returns. The (homoskedastic)
    test statistic is:

    - z = (VR(q) - 1) / SQRT(phi(q))
    - phi(q) = 2 * (2q - 1) * (q - 1) / (3 * q * n)

    Which is asymptotically standard normal under the null hypothesis, allowing for a two-sided
    p-value. A Variance Ratio above 1 with a significant (low) p-value indicates positive
    autocorrelation (momentum/trending behavior), while a Variance Ratio below 1 with a
    significant p-value indicates negative autocorrelation (mean-reversion).

    Neither `statsmodels` nor `linearmodels` ships a Variance Ratio test -- this is one of two
    tests in this module (along with `unitroot_model.get_phillips_perron_test`) that remain
    hand-built.

    For more information about the method, see the following paper:

    - Lo, A.W., & MacKinlay, A.C. (1988). "Stock Market Prices Do Not Follow Random Walks:
    Evidence from a Simple Specification Test." Review of Financial Studies, 1(1), 41-66.

    Also known as: Lo-MacKinlay test, VR test.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        q (int): The number of periods to compound returns over. Defaults to 2.

    Returns:
        pd.Series | pd.DataFrame: The Variance Ratio, the (homoskedastic) test statistic and
        its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_variance_ratio_test(returns.loc[sub_period], q=q)

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_variance_ratio_test(returns[column], q=q)
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        values = returns.dropna().to_numpy()
        n = len(values)

        minimum_observations = 2 * q
        if n <= minimum_observations or q < MINIMUM_VARIANCE_RATIO_Q:
            return pd.Series(
                {
                    "Variance Ratio": np.nan,
                    "Variance Ratio Statistic": np.nan,
                    "P-Value": np.nan,
                }
            )

        single_period_variance = values.var(ddof=1)

        # q-period overlapping compounded (summed log/simple) returns, following the
        # standard overlapping-sample implementation of the Lo-MacKinlay test.
        q_period_returns = pd.Series(values).rolling(window=q).sum().dropna().to_numpy()
        q_period_variance = q_period_returns.var(ddof=1)

        variance_ratio = q_period_variance / (q * single_period_variance)

        phi_q = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
        variance_ratio_statistic = (variance_ratio - 1) / np.sqrt(phi_q)
        p_value = 2 * stats.norm.sf(abs(variance_ratio_statistic))

        return pd.Series(
            {
                "Variance Ratio": variance_ratio,
                "Variance Ratio Statistic": variance_ratio_statistic,
                "P-Value": p_value,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_cusum_test(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the CUSUM test for the stability of the mean of returns over time, via
    `statsmodels.stats.diagnostic.breaks_cusumolsresid`.

    The test fits a constant-mean model (`returns_t = mu + e_t`) via
    `statsmodels.api.OLS` and cumulates the (scaled) OLS residuals into a path that,
    under the null hypothesis of a stable mean, behaves asymptotically like a
    Brownian Bridge. A shift in the mean partway through the series (a structural
    break) drags the path away from zero; the test statistic is the maximum absolute
    value of that path, compared against the asymptotic Brownian Bridge distribution
    (Ploberger & Kramer, 1992) for an exact p-value.

    A significant result (low p-value) indicates that the mean of the return series
    is not stable over the sample period, e.g. due to a regime change or structural
    break partway through.

    For more information about the method, see the following papers:

    - Ploberger, W., & Kramer, W. (1992). "The CUSUM Test with OLS Residuals."
    Econometrica, 60(2), 271-285.
    - Brown, R.L., Durbin, J., & Evans, J.M. (1975). "Techniques for Testing the
    Constancy of Regression Relationships over Time." Journal of the Royal
    Statistical Society, Series B, 37(2), 149-192.

    Also known as: CUSUM test, CUSUM of OLS residuals test.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: The CUSUM statistic, its p-value, the number of
        observations used, the 1%/5%/10% critical boundary values, and whether
        stability is rejected at the 5% level.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_cusum_test(returns.loc[sub_period])

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {column: get_cusum_test(returns[column]) for column in returns.columns}
        )
    if isinstance(returns, pd.Series):
        values = returns.dropna().to_numpy()
        n = len(values)

        nan_result = pd.Series(
            {
                "CUSUM Statistic": np.nan,
                "P-Value": np.nan,
                "Observations": n,
                "Critical Value 1%": np.nan,
                "Critical Value 5%": np.nan,
                "Critical Value 10%": np.nan,
                "Reject Stability (5%)": False,
            }
        )

        if n < MINIMUM_CUSUM_OBSERVATIONS:
            return nan_result

        sm_result = sm.OLS(values, np.ones((n, 1))).fit()

        try:
            cusum_statistic, p_value, critical_values = breaks_cusumolsresid(
                sm_result.resid, ddof=1
            )
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            return nan_result

        critical_value_by_alpha = {alpha: value for alpha, value in critical_values}

        return pd.Series(
            {
                "CUSUM Statistic": cusum_statistic,
                "P-Value": p_value,
                "Observations": n,
                "Critical Value 1%": critical_value_by_alpha[1],
                "Critical Value 5%": critical_value_by_alpha[5],
                "Critical Value 10%": critical_value_by_alpha[10],
                "Reject Stability (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
