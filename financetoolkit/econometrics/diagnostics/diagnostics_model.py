"""Risk Diagnostics Model"""

__docformat__ = "google"

import functools

import numpy as np
import pandas as pd
from scipy import optimize, stats

from financetoolkit.econometrics.least_squares_model import (
    fit_least_squares,
    get_autocorrelations,
    get_kolmogorov_survival,
    get_lagged_values,
)
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# The Variance Ratio test requires at least a two-period compounding horizon.
MINIMUM_VARIANCE_RATIO_Q = 2

# Skewness and kurtosis need at least 2 observations.
MINIMUM_JARQUE_BERA_OBSERVATIONS = 2

# The CUSUM test compares the path of the residuals with the asymptotic boundaries of a
# Brownian bridge, which a handful of residuals cannot follow; shorter series return NaN.
MINIMUM_CUSUM_OBSERVATIONS = 10

# The 1%, 5% and 10% quantiles of sup|B(t)| of a Brownian bridge, to two decimals as
# Ploberger and Krämer (1992) tabulate them.
CUSUM_CRITICAL_VALUES = {1: 1.63, 5: 1.36, 10: 1.22}

# The 1%, 5% and 10% values of a for which a Brownian motion crosses the lines
# ±a(√(n - k) + 2(t - k) / √(n - k)) with that probability (Brown, Durbin and Evans, 1975).
RECURSIVE_CUSUM_CRITICAL_VALUES = {1: 1.143, 5: 0.948, 10: 0.850}

CUSUM_METHODS = ("ols", "recursive", "chu_stinchcombe_white")

# b_0.05 of the Chu-Stinchcombe-White boundary √(b + ln(t - n)), from the simulations of
# Homm and Breitung (2012) as López de Prado (2018) quotes it.
CHU_STINCHCOMBE_WHITE_B = 4.6


# A variance below the floating-point resolution of the mean squared counts as zero, so
# a constant series has an undefined skewness rather than one made of rounding noise.
FLOAT_RESOLUTION = 1e-15
# A series whose spread is at most this share of its level per observation is constant up to
# rounding, which accumulates with the length (the steps of a cumulated constant).
ROUNDING_PER_OBSERVATION = 1e-14

# The conventional 5% significance level used for the "Reject" boolean flags.
SIGNIFICANCE_LEVEL = 0.05

# The sign bias regression has four coefficients and needs residual degrees of freedom.
MINIMUM_SIGN_BIAS_OBSERVATIONS = 10


def get_arch_lm_test(
    returns: pd.Series | pd.DataFrame,
    lags: int = 5,
    weighted: bool = False,
    model_df: int = 0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Engle's ARCH-LM test: whether the volatility of the returns clusters, with large
    squared returns following large ones, or the weighted portmanteau ARCH test of it.

    The Lagrange Multiplier (LM) test of Engle (1982) regresses squared, mean-demeaned returns on a constant and `lags` of
    themselves and tests whether the resulting R-squared is significantly different from
    zero. Under the null hypothesis of no ARCH effects, the LM statistic is asymptotically
    chi-squared distributed with `lags` degrees of freedom:

    - e_t = r_t - mean(r)
    - e_t^2 = a_0 + SUM_{i=1}^{q}(a_i * e_(t-i)^2) + v_t, for t = q + 1, ..., n
    - LM = (n - q) * R^2

    Where `q` is `lags` and `n - q` the number of observations of the regression.

    A significant result (low p-value) indicates that the return series exhibits
    volatility clustering, and a GARCH-family model is an appropriate choice for it.

    The weighted ARCH test is the weighted portmanteau test of Fisher and Gallagher (2012)
    on the autocorrelations of the squared series (the Li-Mak test with weights that fall
    linearly with the lag), which keeps its size better for many lags:

    - x_t = e_t^2, rho_k the autocorrelation of x at lag k
    - Q_W = n * SUM_{k=m+1}^{q}((q - k + m + 1) / q * rho_k^2), with m = `model_df`
    - P-Value = P(G > Q_W) for G ~ Gamma(shape, scale), with
      shape = 3/4 * (q + m + 1)^2 * (q - m) / D, scale = 2/3 * D / (q * (q + m + 1)) and
      D = 2q^2 + 3q + 2qm + 2m^2 + 3m + 1

    For more information about the method, see the following papers:

    - Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates
    of the Variance of United Kingdom Inflation." Econometrica, 50(4), 987-1008.
    - Fisher, T.J. and Gallagher, C.M. (2012). "New Weighted Portmanteau Statistics for
    Time Series Goodness of Fit Testing." Journal of the American Statistical Association,
    107(498), 777-787.

    Referenced against:
    - statsmodels, het_arch: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html
    - rugarch (R), Weighted.LM.test: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L1172

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns, or of the
            standardized residuals of a GARCH model.
        lags (int): The number of lags to test for ARCH effects. Defaults to 5.
        weighted (bool, optional): Whether to calculate the weighted portmanteau test of
            Fisher and Gallagher (2012) instead of Engle's regression. Defaults to False.
        model_df (int, optional): The number of lagged variance parameters (p + q) of the
            GARCH model whose standardized residuals are tested; the weighted test leaves out
            the first `model_df` autocorrelations. Defaults to 0, for returns.

    Returns:
        pd.Series | pd.DataFrame: The ARCH-LM (or weighted ARCH-LM) statistic and its
        p-value.

    Notes:
        - The weighted test reproduces the published values of the rugarch vignette.
    """
    if weighted and not 0 <= model_df < lags:
        raise ValueError(
            f"model_df ({model_df}) must be at least 0 and smaller than lags ({lags})."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_arch_lm_test(
                    returns.loc[sub_period],
                    lags=lags,
                    weighted=weighted,
                    model_df=model_df,
                )

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_arch_lm_test(
                    returns[column], lags=lags, weighted=weighted, model_df=model_df
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        residuals = (returns - returns.mean()).dropna().to_numpy()

        if weighted:
            return _get_weighted_arch_test(residuals**2, lags, model_df)

        if len(residuals) <= lags + 1:
            return pd.Series({"ARCH-LM Statistic": np.nan, "P-Value": np.nan})

        squared = residuals**2
        design = np.column_stack(
            [np.ones(len(squared) - lags), get_lagged_values(squared, lags)]
        )

        try:
            auxiliary = fit_least_squares(squared[lags:], design)
        except (ValueError, np.linalg.LinAlgError):
            return pd.Series({"ARCH-LM Statistic": np.nan, "P-Value": np.nan})

        lm_statistic = (len(squared) - lags) * auxiliary["r_squared"]
        p_value = stats.chi2.sf(lm_statistic, lags)

        return pd.Series({"ARCH-LM Statistic": lm_statistic, "P-Value": p_value})

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def _get_weighted_arch_test(squared: np.ndarray, lags: int, model_df: int) -> pd.Series:
    """
    The weighted portmanteau ARCH test of Fisher and Gallagher (2012) on a squared series,
    with its Gamma approximation, see `get_arch_lm_test`.
    """
    n = len(squared)

    if n <= lags + 1 or np.ptp(squared) == 0:
        return pd.Series({"Weighted ARCH-LM Statistic": np.nan, "P-Value": np.nan})

    lag_numbers = np.arange(model_df + 1, lags + 1)
    autocorrelations = get_autocorrelations(squared, lags)[model_df:]
    weights = (lags - lag_numbers + model_df + 1) / lags
    statistic = n * np.sum(weights * autocorrelations**2)

    denominator = (
        2 * lags**2
        + 3 * lags
        + 2 * lags * model_df
        + 2 * model_df**2
        + 3 * model_df
        + 1
    )
    shape = 0.75 * (lags + model_df + 1) ** 2 * (lags - model_df) / denominator
    scale = 2 / 3 * denominator / (lags * (lags + model_df + 1))

    return pd.Series(
        {
            "Weighted ARCH-LM Statistic": float(statistic),
            "P-Value": float(stats.gamma.sf(statistic, shape, scale=scale)),
        }
    )


def get_jarque_bera_test(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculate the Jarque-Bera test: whether the returns are normally distributed, judged by
    their skewness and kurtosis.

    The test combines sample skewness and excess kurtosis into a single statistic
    that is asymptotically chi-squared distributed with 2 degrees of freedom under
    the null hypothesis that the data is normally distributed:

    - JB = (n / 6) * (S^2 + (K^2) / 4)

    - S = m_3 / m_2^(3/2), K = m_4 / m_2^2 - 3, with m_j = SUM((x_t - mean(x))^j) / n

    Where `S` is the skewness, `K` is the excess (Fisher) kurtosis, both from the
    (biased) sample moments `m_j` as in Jarque and Bera, and `n` is the number of
    observations. A series without variation has no skewness or kurtosis and returns NaN.

    A significant result (low p-value) indicates that returns are not normally
    distributed, which is relevant when choosing between e.g. gaussian and Student-T
    based Value at Risk models.

    For more information about the method, see the following paper:

    - Jarque, C.M. and Bera, A.K. (1987). "A Test for Normality of Observations and
    Regression Residuals." International Statistical Review, 55(2), 163-172.

    Referenced against:
    - statsmodels, jarque_bera: https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.jarque_bera.html

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

        deviations = values - values.mean()
        second_moment = np.mean(deviations**2)

        if second_moment <= (FLOAT_RESOLUTION * values.mean()) ** 2:
            return pd.Series({"Jarque-Bera Statistic": np.nan, "P-Value": np.nan})

        skewness = np.mean(deviations**3) / second_moment**1.5
        excess_kurtosis = np.mean(deviations**4) / second_moment**2 - 3
        jarque_bera_statistic = len(values) / 6 * (skewness**2 + excess_kurtosis**2 / 4)
        p_value = stats.chi2.sf(jarque_bera_statistic, 2)

        return pd.Series(
            {"Jarque-Bera Statistic": jarque_bera_statistic, "P-Value": p_value}
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_ljung_box_test(
    returns: pd.Series | pd.DataFrame,
    lags: int = 10,
    model_df: int = 0,
    weighted: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Ljung-Box test: whether a series is autocorrelated up to a number of lags,
    or the weighted Ljung-Box test of Fisher and Gallagher (2012).

    The test aggregates the squared Autocorrelation Function up to lag `h` into a
    single statistic that is asymptotically chi-squared distributed with `h - model_df`
    degrees of freedom under the null hypothesis that the series exhibits no
    autocorrelation up to that lag:

    - Q = n * (n + 2) * SUM_{k=1}^{h} (rho_k^2 / (n - k))

    - rho_k = SUM_{t=k+1}^{n}((x_t - mean(x)) * (x_(t-k) - mean(x))) / SUM_{t=1}^{n}((x_t - mean(x))^2)

    Where `rho_k` is the Autocorrelation at lag `k` (every autocovariance divided by `n`)
    and `n` is the number of observations.

    The weighted Ljung-Box test weights every squared autocorrelation by (h - k + 1) / h, so the
    first lags count most and a large `h` does not dilute them, and the p-value follows from
    the Gamma distribution with the mean and variance of the weighted statistic (Fisher and
    Gallagher, 2012):

    - Q_W = n * (n + 2) * SUM_{k=1}^{h} ((h - k + 1) / h * rho_k^2 / (n - k))
    - P-Value = P(G > Q_W) for G ~ Gamma(shape, scale), with
      shape = 3/4 * (h + 1)^2 * h / (2h^2 + 3h + 1 - 6hm) and
      scale = 2/3 * (2h^2 + 3h + 1 - 6hm) / (h * (h + 1)), m = `model_df`

    A significant result (low p-value) indicates that the series is autocorrelated, which is
    relevant both as a standalone diagnostic (e.g. to check whether a return series follows a
    random walk) and as a residual diagnostic after fitting a model (e.g. checking that GARCH
    residuals are no longer autocorrelated).

    For more information about the method, see the following paper:

    - Ljung, G.M., & Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time Series Models."
    Biometrika, 65(2), 297-303.
    - Fisher, T.J. and Gallagher, C.M. (2012). "New Weighted Portmanteau Statistics for
    Time Series Goodness of Fit Testing." Journal of the American Statistical Association,
    107(498), 777-787.

    Also known as: Ljung-Box Q test, portmanteau test, weighted Ljung-Box test.

    Referenced against:
    - statsmodels, acorr_ljungbox: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
    - rugarch (R), Weighted.Box.test: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L1056

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        lags (int): The number of lags to test for autocorrelation up to. Defaults to 10.
        model_df (int, optional): The number of ARMA parameters estimated on the series
        that produced `returns`, subtracted from the chi-squared degrees of freedom.
        Defaults to 0, which is correct for an OBSERVED series (raw returns). It is NOT
        correct for the residuals of a fitted model: fitting `p + q` ARMA parameters uses
        up `p + q` of the autocorrelations the statistic aggregates, so leaving this at 0
        inflates the degrees of freedom and makes the test conservative (it under-rejects,
        reporting a larger p-value than it should). Pass `p + q` when testing ARIMA/GARCH
        residuals -- see Box, Pierce (1970) and Ljung, Box (1978).
        weighted (bool, optional): Whether to calculate the weighted Ljung-Box test of Fisher
        and Gallagher (2012). Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: The Ljung-Box statistic and its p-value.

    Raises:
        TypeError: If `returns` is not a pd.DataFrame or pd.Series.
        ValueError: If `model_df` is negative or leaves no degrees of freedom
        (`model_df >= lags`).

    Notes:
        - The weighted test reproduces the published values of the rugarch vignette.
    """
    if model_df < 0:
        raise ValueError(f"model_df must be non-negative, received {model_df}.")
    if model_df >= lags:
        raise ValueError(
            f"model_df ({model_df}) must be smaller than lags ({lags}) -- the "
            "chi-squared distribution needs at least one degree of freedom left."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_ljung_box_test(
                    returns.loc[sub_period],
                    lags=lags,
                    model_df=model_df,
                    weighted=weighted,
                )

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_ljung_box_test(
                    returns[column], lags=lags, model_df=model_df, weighted=weighted
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        values = returns.dropna()
        n = len(values)

        name = "Weighted Ljung-Box Statistic" if weighted else "Ljung-Box Statistic"

        array = values.to_numpy(dtype=float)

        # A series without variation has no autocorrelation: what is left after the mean is
        # rounding noise, whose autocorrelations are close to 1.
        if n <= lags + 1 or np.ptp(array) <= n * ROUNDING_PER_OBSERVATION * np.abs(
            array
        ).max(initial=0.0):
            return pd.Series({name: np.nan, "P-Value": np.nan})

        autocorrelations = get_autocorrelations(array, lags)

        if weighted:
            lag_numbers = np.arange(1, lags + 1)
            statistic = (
                n
                * (n + 2)
                * np.sum(
                    (lags - lag_numbers + 1)
                    / lags
                    * autocorrelations**2
                    / (n - lag_numbers)
                )
            )
            denominator = 2 * lags**2 + 3 * lags + 1 - 6 * lags * model_df
            if denominator <= 0:
                raise ValueError(
                    f"The weighted test needs lags ({lags}) of at least about three times "
                    f"model_df ({model_df})."
                )
            shape = 0.75 * (lags + 1) ** 2 * lags / denominator
            scale = 2 / 3 * denominator / (lags * (lags + 1))

            return pd.Series(
                {
                    name: float(statistic),
                    "P-Value": float(stats.gamma.sf(statistic, shape, scale=scale)),
                }
            )

        ljung_box_statistic = (
            n * (n + 2) * np.sum(autocorrelations**2 / (n - np.arange(1, lags + 1)))
        )

        return pd.Series(
            {
                "Ljung-Box Statistic": float(ljung_box_statistic),
                "P-Value": float(stats.chi2.sf(ljung_box_statistic, lags - model_df)),
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_variance_ratio_test(
    returns: pd.Series | pd.DataFrame, q: int = 2
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Lo-MacKinlay Variance Ratio test: whether returns follow a random walk, or
    trend (momentum) or mean-revert over several periods.

    The test compares the Variance of `q`-period (overlapping) compounded returns to `q` times
    the Variance of single-period returns. Under the random walk hypothesis (i.e. returns have no
    autocorrelation), these two should be equal and the Variance Ratio should be 1:

    - VR(q) = Var(r_t(q)) / (q * Var(r_t))

    Where `r_t(q)` is the sum of `q` consecutive single-period returns. Both variances use the
    unbiased estimators of Lo & MacKinlay (1988), with `mu` the mean single-period return and
    `n` the number of returns, so that VR(q) = sigma_c^2(q) / sigma_a^2 with:

    - sigma_a^2 = SUM((r_t - mu)^2) / (n - 1)
    - sigma_c^2(q) = SUM((r_t(q) - q * mu)^2) / m, with m = q * (n - q + 1) * (1 - q / n)

    The homoskedastic test statistic is:

    - z(q) = (VR(q) - 1) / SQRT(phi(q))
    - phi(q) = 2 * (2q - 1) * (q - 1) / (3 * q * n)

    And the heteroskedasticity-robust test statistic, which stays valid under volatility
    clustering, is:

    - z*(q) = SQRT(n) * (VR(q) - 1) / SQRT(theta(q))
    - theta(q) = SUM over k = 1..q-1 of 4 * (1 - k / q)^2 * delta(k)
    - delta(k) = n * SUM((r_t - mu)^2 * (r_t-k - mu)^2) / SUM((r_t - mu)^2)^2

    Both are asymptotically standard normal under the null hypothesis, allowing for two-sided
    p-values. A Variance Ratio above 1 with a significant (low) p-value indicates positive
    autocorrelation (momentum/trending behavior), while a Variance Ratio below 1 with a
    significant p-value indicates negative autocorrelation (mean-reversion). For financial
    returns, whose variance is rarely constant, the robust p-value is the safer reading.

    For more information about the method, see the following paper:

    - Lo, A.W., & MacKinlay, A.C. (1988). "Stock Market Prices Do Not Follow Random Walks:
    Evidence from a Simple Specification Test." Review of Financial Studies, 1(1), 41-66.

    Also known as: Lo-MacKinlay test, VR test.

    Referenced against:
    - arch, VarianceRatio: https://bashtage.github.io/arch/unitroot/generated/arch.unitroot.VarianceRatio.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        q (int): The number of periods to compound returns over. Defaults to 2.

    Returns:
        pd.Series | pd.DataFrame: The Variance Ratio, the homoskedastic test statistic and its
        p-value, and the heteroskedasticity-robust test statistic and its p-value.
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
                    "Robust Variance Ratio Statistic": np.nan,
                    "Robust P-Value": np.nan,
                }
            )

        # Lo & MacKinlay (1988) unbiased estimators: the single-period variance divides by
        # n - 1 and the overlapping q-period sums, centred on q times the mean, divide by
        # m = q(n - q + 1)(1 - q/n). The plain sample variances are biased towards a ratio below 1 in finite samples.  # noqa: E501
        mean_return = values.mean()
        squared_deviations = (values - mean_return) ** 2
        single_period_variance = np.sum(squared_deviations) / (n - 1)

        # q-period overlapping compounded returns, as the Lo-MacKinlay test defines them.
        q_period_returns = pd.Series(values).rolling(window=q).sum().dropna().to_numpy()
        m = q * (n - q + 1) * (1 - q / n)
        q_period_variance = np.sum((q_period_returns - q * mean_return) ** 2) / m

        variance_ratio = q_period_variance / single_period_variance

        phi_q = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
        variance_ratio_statistic = (variance_ratio - 1) / np.sqrt(phi_q)
        p_value = 2 * stats.norm.sf(abs(variance_ratio_statistic))

        # Heteroskedasticity-robust z*(q): the asymptotic variance theta(q) weights the
        # autocorrelation estimators delta(k) of the squared deviations, so volatility
        # clustering does not inflate the rejection rate (Lo & MacKinlay, 1988, eq. 19-20).
        squared_sum = np.sum(squared_deviations) ** 2
        theta_q = 0.0
        for k in range(1, q):
            delta_k = (
                n * (squared_deviations[k:] @ squared_deviations[:-k]) / squared_sum
            )
            theta_q += 4 * (1 - k / q) ** 2 * delta_k
        robust_statistic = (
            np.sqrt(n) * (variance_ratio - 1) / np.sqrt(theta_q)
            if theta_q > 0
            else np.nan
        )
        robust_p_value = 2 * stats.norm.sf(abs(robust_statistic))

        return pd.Series(
            {
                "Variance Ratio": variance_ratio,
                "Variance Ratio Statistic": variance_ratio_statistic,
                "P-Value": p_value,
                "Robust Variance Ratio Statistic": robust_statistic,
                "Robust P-Value": robust_p_value,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_cusum_test(
    returns: pd.Series | pd.DataFrame,
    method: str = "ols",
    time_series: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Calculate a CUSUM test: whether the mean of the returns stayed stable over time or broke
    at some point.

    Three CUSUM tests are available: the CUSUM of the OLS residuals (Ploberger and Krämer,
    1992), the CUSUM of the recursive residuals (Brown, Durbin and Evans, 1975) and the CUSUM
    of the log price level (Chu, Stinchcombe and White, 1996, as López de Prado, 2018, section
    17.3.2 defines it).

    The CUSUM of the OLS residuals fits a constant-mean model (`returns_t = mu + e_t`) by OLS and cumulates
    the (scaled) OLS residuals into a path that, under the null hypothesis of a stable
    mean, behaves asymptotically like a Brownian Bridge. A shift in the mean partway
    through the series (a structural break) drags the path away from zero; the test
    statistic is the maximum absolute value of that path, compared against the
    asymptotic distribution of the supremum of a Brownian Bridge for its p-value:

    - e_t = r_t - mean(r)
    - sigma^2 = SUM(e_t^2) / (n - 1)
    - B(j) = SUM_{t=1}^{j}(e_t) / (sigma * sqrt(n))
    - CUSUM = max_j |B(j)|
    - P-Value = P(sup|B(t)| > CUSUM) = 2 * SUM_{k>=1}((-1)^(k-1) * exp(-2 * k^2 * CUSUM^2))

    The variance uses the `n - 1` degrees of freedom of the one estimated mean, and the
    critical values are the 1.63, 1.36 and 1.22 Ploberger and Kramer (1992) tabulate.

    The CUSUM of the recursive residuals cumulates the one-step-ahead forecast errors of the mean of the
    returns before each date (the recursive residuals), which are independent with a
    constant variance under the null hypothesis, so their path is a Brownian motion and
    stability is rejected when it crosses one of two lines that widen with time:

    - w_t = (r_t - mean(r_1, ..., r_(t-1))) / sqrt(1 + 1 / (t - 1)), for t = 2, ..., n (k = 1)
    - sigma^2 = SUM(w_t^2) / (n - 1), the residual variance of the full-sample fit
    - W_j = SUM_{t=2}^{j}(w_t) / sigma
    - CUSUM = max_j |W_j| / (sqrt(n - 1) + 2 * (j - 1) / sqrt(n - 1)), the largest multiple a of
      the boundary a * (sqrt(n - 1) + 2 * (j - 1) / sqrt(n - 1)) the path reaches
    - P-Value = 2 * (1 - Phi(3a) + exp(-4a^2) * Phi(a)), the probability that a Brownian motion
      crosses the boundary, with the critical values 1.143, 0.948 and 0.850

    The Chu-Stinchcombe-White CUSUM works on the log price y_t = SUM ln(1 + r_t) (the log of
    the price up to a constant) and asks whether it has risen by more since an earlier date
    than a random walk with the volatility up to that date allows. It is one-sided, a test
    for a rise, as López de Prado (2018) defines it:

    - S_(n,t) = (y_t - y_n) / (sigma_t * sqrt(t - n)), with sigma_t^2 = SUM_{i=1}^{t}((y_i -
      y_(i-1))^2) / t the mean squared change up to t
    - S_t = max over n < t of S_(n,t), at the reference date n*
    - c_t = sqrt(4.6 + ln(t - n*)), the 5% critical value of Homm and Breitung (2012)
    - Stability is rejected at t when S_t > c_t; the test is at the last date

    Every statistic of a date only uses the returns up to that date, so the statistic, its
    critical value and the verdict per date have no look-ahead; they start at the 10th
    return. The 5% holds for every date on its own (on simulated random walks of 100 and 500
    returns the last date rejects about 4% to 5% of the time), so over many dates some cross
    the critical value by chance. The other two methods scale with the whole sample and only
    give the test.

    For more information about the method, see the following papers:

    - Ploberger, W., & Kramer, W. (1992). "The CUSUM Test with OLS Residuals."
    Econometrica, 60(2), 271-285.
    - Brown, R.L., Durbin, J., & Evans, J.M. (1975). "Techniques for Testing the
    Constancy of Regression Relationships over Time." Journal of the Royal
    Statistical Society, Series B, 37(2), 149-192.
    - Chu, C.-S.J., Stinchcombe, M. and White, H. (1996). "Monitoring Structural Change."
    Econometrica, 64(5), 1045-1065.
    - Homm, U. and Breitung, J. (2012). "Testing for Speculative Bubbles in Stock Markets: A
    Comparison of Alternative Methods." Journal of Financial Econometrics, 10(1), 198-231.
    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section
    17.3.

    Also known as: CUSUM test, CUSUM of OLS residuals test, recursive CUSUM test,
    Brown-Durbin-Evans test, Chu-Stinchcombe-White test.

    Referenced against:
    - statsmodels, breaks_cusumolsresid (CUSUM of the OLS residuals): https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html

    Shares similar logic:
    - statsmodels, recursive_olsresiduals (CUSUM of the recursive residuals): https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.recursive_olsresiduals.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        method (str, optional): The CUSUM test. Options are: "ols" (CUSUM of the OLS
            residuals), "recursive" (CUSUM of the recursive residuals), "chu_stinchcombe_white"
            (Chu-Stinchcombe-White CUSUM of the log price). Defaults to "ols".
        time_series (bool, optional): Whether to return the statistic, its critical value
            and the verdict per date (Chu-Stinchcombe-White CUSUM only). Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: The CUSUM statistic, its p-value, the number of
        observations used, the 1%/5%/10% critical values, and whether stability is rejected
        at the 5% level; for the Chu-Stinchcombe-White CUSUM the statistic, the 5% critical
        value, the observations and the verdict at the last date, or per date.

    Notes:
        - The CUSUM of the OLS residuals equals statsmodels' breaks_cusumolsresid with one
          degree of freedom for the mean.
        - Unlike statsmodels' recursive_olsresiduals, which scales the path with the standard
          deviation of the recursive residuals around their mean, the recursive CUSUM scales it
          with the residual variance of Brown, Durbin and Evans (1975).
        - The Chu-Stinchcombe-White CUSUM follows López de Prado (2018, "Advances in Financial
          Machine Learning", Wiley, section 17.3.2): it divides by sigma_t, with sigma_t^2 the
          mean of the t - 1 squared changes, and compares with sqrt(4.6 + ln(t - n)) at 5%.
    """
    check_option("method", method, CUSUM_METHODS)
    if time_series and method != "chu_stinchcombe_white":
        raise InvalidArgumentError(
            "time_series=True needs method='chu_stinchcombe_white': the OLS and recursive "
            "CUSUM paths are scaled with the variance of the whole sample, so a date's value "
            "would use returns after it."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = get_cusum_test(
                    returns.loc[sub_period], method=method, time_series=time_series
                )

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        result = pd.concat(
            {
                column: get_cusum_test(
                    returns[column], method=method, time_series=time_series
                )
                for column in returns.columns
            },
            axis=1,
        )

        if time_series:
            return result.swaplevel(0, 1, axis=1).sort_index(axis=1)

        return result
    if isinstance(returns, pd.Series):
        if method == "chu_stinchcombe_white":
            return _get_chu_stinchcombe_white_test(returns, time_series)

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

        # A series that is constant (up to rounding) leaves residuals of rounding noise
        # rather than zeros, which would otherwise give a statistic of noise over noise.
        if (
            np.ptp(values)
            <= len(values) * ROUNDING_PER_OBSERVATION * np.abs(values).max()
        ):
            return nan_result

        if method == "ols":
            residuals = fit_least_squares(values, np.ones((n, 1)))["residuals"]
            scaled_sum_of_squares = np.sum(residuals**2) / (n - 1) * n

            cusum_statistic = float(
                np.max(np.abs(np.cumsum(residuals) / np.sqrt(scaled_sum_of_squares)))
            )
            p_value = get_kolmogorov_survival(cusum_statistic)
            critical_values = CUSUM_CRITICAL_VALUES
        else:
            # The forecast of every return from the mean of the returns before it, scaled by
            # the standard error of that forecast, sqrt(1 + 1 / (t - 1)) with t - 1 returns.
            earlier = np.arange(1, n)
            earlier_means = np.cumsum(values)[:-1] / earlier
            recursive_residuals = (values[1:] - earlier_means) / np.sqrt(
                1 + 1 / earlier
            )
            # The sum of squared recursive residuals equals that of the OLS residuals, so this
            # is the residual variance of the full-sample fit (Brown, Durbin and Evans, 1975).
            sigma = np.sqrt(np.sum(recursive_residuals**2) / (n - 1))
            path = np.cumsum(recursive_residuals) / sigma
            boundary = np.sqrt(n - 1) + 2 * np.arange(1, n) / np.sqrt(n - 1)

            cusum_statistic = float(np.max(np.abs(path) / boundary))
            p_value = float(
                min(
                    2
                    * (
                        stats.norm.sf(3 * cusum_statistic)
                        + np.exp(-4 * cusum_statistic**2)
                        * stats.norm.cdf(cusum_statistic)
                    ),
                    1.0,
                )
            )
            critical_values = RECURSIVE_CUSUM_CRITICAL_VALUES

        return pd.Series(
            {
                "CUSUM Statistic": cusum_statistic,
                "P-Value": p_value,
                "Observations": n,
                "Critical Value 1%": critical_values[1],
                "Critical Value 5%": critical_values[5],
                "Critical Value 10%": critical_values[10],
                "Reject Stability (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def _get_chu_stinchcombe_white_test(
    returns: pd.Series, time_series: bool
) -> pd.Series | pd.DataFrame:
    """
    The Chu-Stinchcombe-White CUSUM on the log price of `returns` at its last date, or per
    date with `time_series=True`, see `get_cusum_test`.
    """
    returns = returns.dropna()
    n = len(returns)
    columns = ["CUSUM Statistic", "Critical Value 5%", "Reject Stability (5%)"]

    if n < MINIMUM_CUSUM_OBSERVATIONS or (returns <= -1).any():
        if time_series:
            return pd.DataFrame(np.nan, index=returns.index, columns=columns)
        return pd.Series(
            {
                "CUSUM Statistic": np.nan,
                "Critical Value 5%": np.nan,
                "Observations": n,
                "Reject Stability (5%)": False,
            }
        )

    changes = np.log1p(returns.to_numpy(dtype=float))
    # The log price with the price before the first return as 0, so y_0 = 0 and y_t is
    # the change in the log price over the first t returns.
    levels = np.concatenate([[0.0], np.cumsum(changes)])
    mean_squared_changes = np.cumsum(changes**2) / np.arange(1, n + 1)

    def statistic_at(t: int) -> tuple[float, float]:
        """S_t and c_t at the level after the t-th return."""
        distances = t - np.arange(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = (levels[t] - levels[:t]) / (
                np.sqrt(mean_squared_changes[t - 1]) * np.sqrt(distances)
            )
        if not np.isfinite(statistics).any():
            return np.nan, np.nan
        reference = int(np.nanargmax(statistics))
        return float(statistics[reference]), float(
            np.sqrt(CHU_STINCHCOMBE_WHITE_B + np.log(distances[reference]))
        )

    if not time_series:
        statistic, critical_value = statistic_at(n)
        return pd.Series(
            {
                "CUSUM Statistic": statistic,
                "Critical Value 5%": critical_value,
                "Observations": n,
                "Reject Stability (5%)": bool(statistic > critical_value),
            }
        )

    # The volatility of the first few changes is too noisy for a verdict, so the dates
    # start once there are as many returns as the test needs rather than after two.
    path = np.full((n, 2), np.nan)
    for t in range(MINIMUM_CUSUM_OBSERVATIONS, n + 1):
        path[t - 1] = statistic_at(t)

    result = pd.DataFrame(path, index=returns.index, columns=columns[:2])
    result["Reject Stability (5%)"] = (
        result["CUSUM Statistic"] > result["Critical Value 5%"]
    ).where(result["CUSUM Statistic"].notna())

    return result


def get_sign_bias_test(
    residuals: pd.Series | pd.DataFrame,
    standardized_residuals: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the sign bias test of Engle and Ng (1993): whether the squared standardized
    residuals of a volatility model still depend on the sign and the size of the shock
    before them. A symmetric GARCH model lets a fall and a rise of the same size raise the
    volatility equally; when falls raise it by more (the leverage effect), the squared
    standardized residuals after a fall stay larger than the model expects and the test
    rejects, which points to an asymmetric model such as GJR-GARCH or APARCH.

    The formula is as follows:

    - z_t^2 = c_0 + c_1 * S_(t-1)^- + c_2 * S_(t-1)^- * e_(t-1) + c_3 * S_(t-1)^+ * e_(t-1) + u_t, by OLS
    - S_(t-1)^- = 1 when e_(t-1) < 0 and 0 otherwise, S_(t-1)^+ = 1 - S_(t-1)^-
    - Sign Bias, Negative Size Bias and Positive Size Bias: the t-statistics of c_1, c_2 and
      c_3, with two-sided p-values from the t-distribution with n - 4 degrees of freedom
    - Joint Statistic = n * R^2 of the regression, chi-squared with 3 degrees of freedom

    Where e_t are the residuals of the mean model, z_t = e_t / sigma_t the standardized
    residuals and n the number of observations of the regression.

    For more information about the method, see the following paper:

    - Engle, R.F. and Ng, V.K. (1993). "Measuring and Testing the Impact of News on
    Volatility." The Journal of Finance, 48(5), 1749-1778.

    Also known as: sign bias test, Engle-Ng test, size bias test, leverage effect test.

    Referenced against:
    - rugarch (R), signbias: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L195

    Args:
        residuals (pd.Series | pd.DataFrame): The residuals e_t of the mean model.
        standardized_residuals (pd.Series | pd.DataFrame): The standardized residuals
            z_t = e_t / sigma_t, with the same index (and columns).

    Returns:
        pd.Series | pd.DataFrame: The t-statistic and p-value of the sign bias, negative
        size bias and positive size bias, the joint statistic and its p-value, and the
        observations of the regression.

    Notes:
        - The t-statistics reproduce those of the rugarch vignette. Unlike rugarch, which
          reports the absolute t-statistics and a Wald statistic with the OLS covariance as the
          joint test, this reports the signed t-statistics and the n * R^2 Lagrange multiplier
          statistic of Engle and Ng (1993).
    """
    if isinstance(residuals, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_sign_bias_test(
                    residuals[column], standardized_residuals[column]
                )
                for column in residuals.columns
            }
        )

    names = ["Sign Bias", "Negative Size Bias", "Positive Size Bias"]
    data = pd.concat([residuals, standardized_residuals], axis=1).dropna()
    n = len(data) - 1

    result = {}
    for name in names:
        result[f"{name} t-Statistic"] = np.nan
        result[f"{name} P-Value"] = np.nan
    result.update({"Joint Statistic": np.nan, "Joint P-Value": np.nan})
    result["Observations"] = max(n, 0)

    if n < MINIMUM_SIGN_BIAS_OBSERVATIONS:
        return pd.Series(result)

    shocks = data.iloc[:-1, 0].to_numpy(dtype=float)
    squared = data.iloc[1:, 1].to_numpy(dtype=float) ** 2
    negative = (shocks < 0).astype(float)
    design = np.column_stack(
        [np.ones(n), negative, negative * shocks, (1 - negative) * shocks]
    )

    fit = fit_least_squares(squared, design)
    if fit["rank"] < design.shape[1]:
        return pd.Series(result)

    for index, name in enumerate(names, start=1):
        result[f"{name} t-Statistic"] = float(fit["t_statistics"][index])
        result[f"{name} P-Value"] = float(fit["p_values"][index])

    joint_statistic = n * fit["r_squared"]
    result["Joint Statistic"] = float(joint_statistic)
    result["Joint P-Value"] = float(stats.chi2.sf(joint_statistic, 3))

    return pd.Series(result)


def get_pearson_goodness_of_fit_test(
    standardized_residuals: pd.Series | pd.DataFrame,
    bins: int = 20,
    distribution: str = "normal",
    parameters: tuple[float, ...] = (),
) -> pd.Series | pd.DataFrame:
    """
    Calculate the adjusted Pearson goodness of fit test of Palm (1996) and Vlaar and Palm
    (1993): whether the standardized residuals of a volatility model follow the innovation
    distribution the model assumes.

    Each residual is turned into the probability of a value
    at most as large under that distribution, which is uniform on (0, 1) when the
    distribution fits; the probabilities are counted in equally wide bins and compared with
    the equal count every bin should have.

    The formula is as follows:

    - u_t = F(z_t), with F the distribution function of the standardized innovation
      distribution (normal by default) and z_t the standardized residuals
    - n_i = the number of u_t in [(i - 1) / g, i / g), the last bin including 1, for i = 1, ..., g
      with g = `bins` (20 by default)
    - Pearson Statistic = SUM_{i=1}^{g}((n_i - n / g)^2 / (n / g)), chi-squared with g - 1
      degrees of freedom

    Classifying the residuals by their probability rather than by their value makes the
    test independent of the scale and keeps every bin's expected count the same, which
    Palm (1996) recommends for the dependent residuals of volatility models.

    For more information about the method, see the following papers:

    - Palm, F.C. (1996). "GARCH Models of Volatility." In Handbook of Statistics, Vol. 14,
    Elsevier, 209-240.
    - Vlaar, P.J.G. and Palm, F.C. (1993). "The Message in Weekly Exchange Rates in the
    European Monetary System: Mean Reversion, Conditional Heteroscedasticity, and Jumps."
    Journal of Business & Economic Statistics, 11(3), 351-360.

    Also known as: adjusted Pearson goodness of fit test, chi-squared goodness of fit test,
    probability integral transform test, GARCH distribution test.

    Referenced against:
    - rugarch (R), gof: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L223

    Args:
        standardized_residuals (pd.Series | pd.DataFrame): The standardized residuals.
        bins (int, optional): The number of bins g. Defaults to 20.
        distribution (str, optional): The standardized innovation distribution. Options are:
            "normal" (normal), "t" (Student-t), "skewt" (skewed Student-t of Hansen),
            "skewt_fernandez_steel" (skewed Student-t of Fernández and Steel), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (tuple[float, ...], optional): The Shape (and Skew) of the distribution.
            Defaults to (), for the normal distribution.

    Returns:
        pd.Series | pd.DataFrame: The Pearson statistic, its p-value, the bins and the
        observations.

    Notes:
        - Unlike rugarch, which leaves the empty bins above the last occupied one out of the
          sum (its tabulate stops at the highest bin with a value), every empty bin counts here.
    """
    if bins < 2:  # noqa: PLR2004
        raise ValueError(f"The test needs at least 2 bins, not {bins}.")

    if isinstance(standardized_residuals, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_pearson_goodness_of_fit_test(
                    standardized_residuals[column], bins, distribution, parameters
                )
                for column in standardized_residuals.columns
            }
        )

    values = standardized_residuals.dropna().to_numpy(dtype=float)
    n = len(values)

    if n < bins:
        return pd.Series(
            {
                "Pearson Statistic": np.nan,
                "P-Value": np.nan,
                "Bins": bins,
                "Observations": n,
            }
        )

    probabilities = garch_model.get_standardized_cdf(values, distribution, parameters)
    counts = np.bincount(
        np.minimum((probabilities * bins).astype(int), bins - 1), minlength=bins
    )
    expected = n / bins
    statistic = float(np.sum((counts - expected) ** 2) / expected)

    return pd.Series(
        {
            "Pearson Statistic": statistic,
            "P-Value": float(stats.chi2.sf(statistic, bins - 1)),
            "Bins": bins,
            "Observations": n,
        }
    )


# The Nyblom statistic of k parameters converges to Σ_(j=1..k) ∫ B_j(r)^2 dr over k independent
# Brownian bridges, Σ_n Σ_j Z_(n,j)^2 / (n^2 π^2) with independent standard normal Z. Its tail
# probability is Imhof's (1961) integral over the first 300 weights 1 / (n^2 π^2), with the sums of
# the weights beyond them, on a grid u = s^2 for s in (0, 80], where the integrand has fallen below
# e^-20; it reproduces the Cramér-von Mises quantiles 0.3473, 0.4614 and 0.7435 of one parameter
# to 1e-6.
NYBLOM_TERMS = 300
NYBLOM_GRID = np.linspace(0, 80, 16001)[1:]


@functools.cache
def _nyblom_integrand_terms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid u = s^2 and, on it, Σ_n arctan(λ_n u) and Σ_n ln(1 + λ_n^2 u^2) / 4 of one parameter."""
    weights = 1 / (np.arange(1, NYBLOM_TERMS + 1) ** 2 * np.pi**2)
    tail = 1 / (np.arange(NYBLOM_TERMS + 1, 200_000) ** 2 * np.pi**2)
    u = NYBLOM_GRID**2
    # Beyond the first weights λ_n u is small: arctan(x) = x - x^3 / 3 and ln(1 + x^2) = x^2.
    angle = (
        np.arctan(np.outer(u, weights)).sum(axis=1)
        + tail.sum() * u
        - np.sum(tail**3) * u**3 / 3
    )
    log_scale = 0.25 * (
        np.log1p(np.outer(u, weights) ** 2).sum(axis=1) + np.sum(tail**2) * u**2
    )

    return u, angle, log_scale


# The moment conditions of the GMM orthogonality test that need the skewness and the kurtosis.
GMM_SKEWNESS_MOMENTS = ("Skewness", "Q3")
GMM_KURTOSIS_MOMENTS = ("Kurtosis", "Q4")


def get_nyblom_p_value(statistic: float, parameters: int = 1) -> float:
    """
    Calculate the asymptotic p-value of the Nyblom (1989) parameter stability statistic: the
    probability that Σ_(j=1..k) ∫ B_j(r)^2 dr over k independent Brownian bridges, one per
    parameter, exceeds the statistic.

    The formula is as follows (Imhof, 1961), with λ_n = 1 / (n^2 π^2), each with multiplicity k:

    - P(L > x) = 1 / 2 + (1 / π) ∫_0^∞ sin θ(u) / (u ρ(u)) du
    - θ(u) = (k / 2) Σ_n arctan(λ_n u) - x u / 2 and ρ(u) = Π_n (1 + λ_n^2 u^2)^(k / 4)

    For one parameter this is the limiting distribution of the Cramér-von Mises statistic
    (5% critical value 0.461); Hansen (1992) tabulates its quantiles by simulation (0.470 for
    one parameter at 5%).

    For more information, see the following papers:

    - Imhof, J.P. (1961). "Computing the Distribution of Quadratic Forms in Normal
    Variables." Biometrika, 48(3/4), 419-426.
    - Hansen, B.E. (1992). "Testing for Parameter Instability in Linear Models." Journal of
    Policy Modeling, 14(4), 517-533.

    Args:
        statistic (float): The Nyblom statistic.
        parameters (int, optional): The number of parameters k the statistic covers.
            Defaults to 1.

    Returns:
        float: The p-value, NaN for a statistic that is not finite.
    """
    if not np.isfinite(statistic):
        return np.nan

    u, angle, log_scale = _nyblom_integrand_terms()

    with np.errstate(over="ignore"):
        integrand = (
            2
            * np.sin(0.5 * parameters * angle - 0.5 * statistic * u)
            / (NYBLOM_GRID * np.exp(parameters * log_scale))
        )

    # With u = s^2 the integrand is 2 sin θ / (s ρ), which is 0 at s = 0.
    value = np.trapezoid(
        np.concatenate([[0.0], integrand]), np.concatenate([[0.0], NYBLOM_GRID])
    )

    return float(np.clip(0.5 + value / np.pi, 0.0, 1.0))


def get_nyblom_test(
    scores: np.ndarray | pd.DataFrame, names: list[str] | None = None
) -> pd.Series:
    """
    Calculate the parameter stability test of Nyblom (1989) and Hansen (1992): whether the
    parameters of a model fitted by maximum likelihood stay the same over the sample.

    At the maximum the scores of every parameter sum to zero over the whole sample;
    when a parameter changes during the sample, their running sum wanders far from zero in
    between.

    The formula is as follows, with s_t the scores (the derivatives of the log-likelihood of
    observation t with respect to the parameters, a row per observation) and T the number of
    observations:

    - S_t = Σ_(i=1..t) s_i, the running sum of the scores
    - V = Σ_t s_t s_t', the outer product of the scores
    - Individual Statistic of parameter j = Σ_t S_t,j^2 / (T V_jj)
    - Joint Statistic = Σ_t S_t' V^-1 S_t / T
    - P-values from the limiting distribution, the sum of k squared Brownian bridges integrated over
      the sample with k = 1 for an individual and the number of parameters for the joint statistic
      (see `get_nyblom_p_value`)

    The statistics are large, and the p-values small, when a parameter changes during the
    sample. A parameter that sits on a bound of its search at the maximum (such as a Beta of
    0) has scores that do not sum to zero, which inflates its statistic and the joint one.

    For more information, see the following papers:

    - Nyblom, J. (1989). "Testing for the Constancy of Parameters Over Time." Journal of the
    American Statistical Association, 84(405), 223-230.
    - Hansen, B.E. (1992). "Testing for Parameter Instability in Linear Models." Journal of
    Policy Modeling, 14(4), 517-533.

    Shares similar logic:
    - rugarch (R), nyblom: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L110

    Args:
        scores (np.ndarray | pd.DataFrame): The scores, one row per observation and one
            column per parameter.
        names (list[str] | None, optional): The parameter names. Defaults to None, which uses
            the columns of a DataFrame or Parameter 1, 2, ...

    Returns:
        pd.Series: The Joint Statistic, its P-Value and 5% Critical Value, and the Statistic
        and P-Value of every parameter, with the 5% Critical Value of one parameter and the
        Observations.

    Notes:
        - Unlike rugarch, which reports Hansen's tabulated critical values, the p-values and
          critical values here are those of the limiting distribution itself.
    """
    if isinstance(scores, pd.DataFrame):
        names = list(scores.columns) if names is None else names
        scores = scores.to_numpy(dtype=float)

    scores = np.asarray(scores, dtype=float)
    observations, parameters = scores.shape
    names = (
        [f"Parameter {i + 1}" for i in range(parameters)] if names is None else names
    )

    cumulative = np.cumsum(scores, axis=0)
    outer = scores.T @ scores
    individual = np.sum(cumulative**2, axis=0) / (observations * np.diag(outer))

    try:
        joint = float(
            np.sum(cumulative * np.linalg.solve(outer, cumulative.T).T) / observations
        )
    except np.linalg.LinAlgError:
        joint = np.nan

    def critical_value(count: int) -> float:
        return float(
            optimize.brentq(
                lambda x: get_nyblom_p_value(x, count) - SIGNIFICANCE_LEVEL,
                1e-3,
                10.0 + count,
            )
        )

    result = {
        "Joint Statistic": joint,
        "Joint P-Value": get_nyblom_p_value(joint, parameters),
        "Joint Critical Value (5%)": critical_value(parameters),
    }
    for name, statistic in zip(names, individual, strict=True):
        result[f"{name} Statistic"] = float(statistic)
        result[f"{name} P-Value"] = get_nyblom_p_value(float(statistic), 1)
    result["Individual Critical Value (5%)"] = critical_value(1)
    result["Observations"] = observations

    return pd.Series(result)


def get_gmm_moment_test(
    standardized_residuals: pd.Series | pd.DataFrame,
    lags: int = 1,
    distribution: str = "normal",
    parameters: tuple[float, ...] = (),
) -> pd.Series | pd.DataFrame:
    """
    Calculate the GMM orthogonality test of Hansen (1982): whether the standardized residuals
    of a volatility model have the mean, variance, skewness and kurtosis of the
    innovation distribution, and are their squares, cubes and fourth powers uncorrelated over
    time, as a correctly specified model implies?

    The formula is as follows, with z_t the standardized residuals, m_k = E[z^k] the raw
    moments of the innovation distribution, s = m_3 and κ = m_4 its skewness and kurtosis (not
    excess) and j = 1, ..., p the lags:

    - Mean: E[z_t] = 0; Variance: E[z_t^2 - 1] = 0; Skewness: E[z_t^3 - s] = 0; Kurtosis: E[z_t^4 - κ] = 0
    - Q2: E[(z_t^2 - 1)(z_(t-j)^2 - 1)] = 0; Q3: E[(z_t^3 - s)(z_(t-j)^3 - s)] = 0;
      Q4: E[(z_t^4 - κ)(z_(t-j)^4 - κ)] = 0
    - g_t the moment conditions of observation t, over the T observations t = p + 1, ..., n, and ḡ
      their average
    - S their covariance under the null of independent innovations: Cov(z^a - m_a, z^b - m_b) =
      m_(a+b) - m_a m_b between the four moments, from the innovation distribution;
      Cov(a_t b_(t-j), c_t d_(t-k)) = E[a c] E[b d] for j = k and 0 otherwise between the lagged
      products, with E[a c] the average of the products of the centred moments; and 0 between the
      moments and the lagged products
    - t-Statistic of a moment = ḡ_i / √(S_ii / T), two-sided p-values from the normal distribution
    - Q2, Q3 and Q4 Statistic = T ḡ' S^-1 ḡ over their p conditions, chi-squared with p degrees of freedom
    - Joint Statistic = T ḡ' S^-1 ḡ over all 4 + 3 p conditions, chi-squared with 4 + 3 p degrees of
      freedom

    The parameters of the model are estimated, which the test leaves out of the covariance
    of the moments, as rugarch's GMMTest does; read the p-values as a guide. The moments are
    correlated with each other (z_t with z_t^3, z_t^2 with z_t^4, and Q2 with Q4), so the joint test
    can reject while none of the individual tests does. The covariance of the four moments is
    the one the distribution implies and that of the lagged products the one the null implies,
    rather than their sample covariance, which rests on 8th and 16th sample moments of z_t: on
    simulated independent normal innovations the joint test at 5% (one lag) then rejects about
    6.6% of the time at 1,000 observations and 5.3% at 2,500, against 10.5% and 8.2% with the
    sample covariance of the moments. On simulated Student-t innovations with a Shape of 10 (and
    the same distribution assumed) the joint test rejects about 5% of the time and the Kurtosis
    test about 3% at 1,000 and 2,500 observations. A moment the distribution does not have (m_6
    or m_8 of a Student-t with a Shape of 6 or 8 or less) takes its sample counterpart, which
    rests on moments that do not exist: the tests of the conditions it enters are then
    unreliable.

    For more information about the method, see the following paper:

    - Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of Moments
    Estimators." Econometrica, 50(4), 1029-1054.

    Shares similar logic:
    - rugarch (R), GMMTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L534

    Args:
        standardized_residuals (pd.Series | pd.DataFrame): The standardized residuals.
        lags (int, optional): The number of lags p of the Q2, Q3 and Q4 conditions. Defaults to 1.
        distribution (str, optional): The standardized innovation distribution. Options are:
            "normal" (normal), "t" (Student-t), "skewt" (skewed Student-t of Hansen),
            "skewt_fernandez_steel" (skewed Student-t of Fernández and Steel), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). A Student-t with a Shape of 3 or
            less has no skewness, which leaves the Skewness and Q3 conditions out, and one of 4
            or less no kurtosis, which leaves the Kurtosis and Q4 conditions out. Defaults to
            "normal".
        parameters (tuple[float, ...], optional): The Shape (and Skew) of the distribution.
            Defaults to ().

    Returns:
        pd.Series | pd.DataFrame: The t-statistic and p-value of the Mean, Variance, Skewness
        and Kurtosis conditions, the Q2, Q3, Q4 and Joint statistics with their p-values,
        the lags and the observations; NaN for a series without variation.

    Notes:
        - The moment conditions M1 to M4, Q2 to Q4 and the joint J test are those rugarch's
          documentation of GMMTest defines; rugarch takes the covariance of the moments from the
          sample, which this takes from the innovation distribution.
    """
    if lags < 1:
        raise ValueError(f"The test needs at least 1 lag, not {lags}.")

    if isinstance(standardized_residuals, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_gmm_moment_test(
                    standardized_residuals[column], lags, distribution, parameters
                )
                for column in standardized_residuals.columns
            }
        )

    z = standardized_residuals.dropna().to_numpy(dtype=float)
    raw_moments = garch_model.get_standardized_raw_moments(distribution, parameters, 8)
    skewness, kurtosis = raw_moments[3], raw_moments[4]
    centred = {
        "Mean": z,
        "Variance": z**2 - 1,
        "Skewness": z**3 - skewness,
        "Kurtosis": z**4 - kurtosis,
    }
    powers = {"Mean": 1, "Variance": 2, "Skewness": 3, "Kurtosis": 4}
    conditions = {name: values[lags:] for name, values in centred.items()}
    groups = {"Q2": "Variance", "Q3": "Skewness", "Q4": "Kurtosis"}
    for group, name in groups.items():
        for lag in range(1, lags + 1):
            conditions[f"{group} {lag}"] = (
                centred[name][lags:] * centred[name][lags - lag : len(z) - lag]
            )

    for moment, names in (
        (skewness, GMM_SKEWNESS_MOMENTS),
        (kurtosis, GMM_KURTOSIS_MOMENTS),
    ):
        if not np.isfinite(moment):
            conditions = {
                name: values
                for name, values in conditions.items()
                if not name.startswith(names)
            }

    observations = len(z) - lags
    result: dict[str, float] = {}
    names = list(conditions)
    # A series without variation (up to rounding) has no standardized residuals to test.
    constant = len(z) == 0 or np.ptp(z) <= len(z) * ROUNDING_PER_OBSERVATION * np.abs(
        z
    ).max(initial=0.0)
    if observations <= len(names) or constant:
        covariance = np.full((len(names), len(names)), np.nan)
    else:
        # The covariance of the conditions under the null of independent innovations. Between
        # the four moments it is the one the distribution implies, m_(a+b) - m_a m_b: their
        # sample covariance rests on 8th sample moments of z_t and gives the joint test at 5%
        # a size of about 10% at 1,000 normal observations. A moment the distribution does not
        # have takes the sample covariance instead. For the lagged products (centred moments
        # a and b at lags j and k) Cov = δ_jk E[a b]², uncorrelated with the moments: the
        # sample covariance of the products themselves rests on 16th moments of z_t and on
        # cross-products that are only zero in expectation.
        base = [name for name in centred if name in conditions]
        moments = np.column_stack([centred[name][lags:] for name in base])
        products = np.atleast_2d(moments.T @ moments / observations)
        order = np.array([powers[name] for name in base])
        implied = raw_moments[order[:, None] + order[None, :]] - np.outer(
            raw_moments[order], raw_moments[order]
        )
        covariance = np.zeros((len(names), len(names)))
        size = len(base)
        covariance[:size, :size] = np.where(
            np.isfinite(implied),
            implied,
            np.atleast_2d(np.cov(moments, rowvar=False)),
        )
        for row, first in enumerate(names[size:], size):
            for column, second in enumerate(names[size:], size):
                group_a, lag_a = first.split(" ")
                group_b, lag_b = second.split(" ")
                if lag_a == lag_b:
                    covariance[row, column] = (
                        products[
                            base.index(groups[group_a]), base.index(groups[group_b])
                        ]
                        ** 2
                    )
    index = {name: position for position, name in enumerate(names)}

    def wald(selected: list[str]) -> tuple[float, float]:
        if not selected or observations <= len(names) or constant:
            return np.nan, np.nan
        positions = [index[name] for name in selected]
        average = np.column_stack([conditions[name] for name in selected]).mean(axis=0)
        try:
            statistic = float(
                observations
                * average
                @ np.linalg.solve(covariance[np.ix_(positions, positions)], average)
            )
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        return statistic, float(stats.chi2.sf(statistic, len(selected)))

    for name in centred:
        if name in conditions and observations > len(names) and not constant:
            statistic = float(
                conditions[name].mean()
                / np.sqrt(covariance[index[name], index[name]] / observations)
            )
            result[f"{name} t-Statistic"] = statistic
            result[f"{name} P-Value"] = float(2 * stats.norm.sf(abs(statistic)))
        else:
            result[f"{name} t-Statistic"] = np.nan
            result[f"{name} P-Value"] = np.nan

    for group in groups:
        statistic, p_value = wald(
            [name for name in conditions if name.startswith(f"{group} ")]
        )
        result[f"{group} Statistic"] = statistic
        result[f"{group} P-Value"] = p_value

    statistic, p_value = wald(list(conditions))
    result["Joint Statistic"] = statistic
    result["Joint P-Value"] = p_value
    result["Lags"] = lags
    result["Observations"] = observations

    return pd.Series(result)


def get_level_significance(
    levels: pd.Series, level: float | None = None, difference: bool = False
) -> pd.Series:
    """
    Calculate how unusual the latest value of a series is compared with its own history: the
    percentile of the latest value among the earlier values, the last dates the series was
    at least as high and at least as low, its change since the first observation, and the
    highest and lowest values of the range with their dates and the change from them.

    The formula is as follows:

    - Percentile = (Count(x(s) < x(T)) + 0.5 * Count(x(s) = x(T))) / Count(s), over s < T
    - Rank = 1 + Count(x(s) > x(T)), over s < T: 1 for the highest of the history
    - Highest Since = the last date s < T with x(s) >= x(T)
    - Lowest Since = the last date s < T with x(s) <= x(T)
    - Change Since Start = (x(T) - x(0)) / |x(0)|
    - Change From Highest = (x(T) - max(x)) / |max(x)|
    - Change From Lowest = (x(T) - min(x)) / |min(x)|
    - Multiple Since Start = x(T) / x(0)
    - Annualised Change = (x(T) / x(0)) ^ (1 / years) - 1, years between the two dates
    - Largest Drawdown = min over t of (x(t) - max(x(s), s <= t)) / |max(x(s), s <= t)|
    - Recovery Date = the first date after the trough with x >= the peak before it
    - First Above Level = the first date s with x(s) > level
    - Above Level Since = the first date of the run of observations above the level that
      the latest one belongs to
    - Last Above Level = the last date s < T with x(s) > level
    - Last At Or Below Level = the last date s < T with x(s) <= level
    - Observations Above Level = Count(x(s) > level), over s <= T
    - Observations At Or Below Level = Count(x(s) <= level), over s <= T

    As differences, the changes are in the unit of the series
    (percentage points for a rate): Change Since Start = x(T) - x(0), Change From Highest =
    x(T) - max(x), Change From Lowest = x(T) - min(x) and Largest Drawdown = min over t of
    (x(t) - max(x(s), s <= t)); the Multiple Since Start and Annualised Change are NaN.

    Where x(T) is the latest value and s runs over every earlier observation. The percentile counts
    an earlier value equal to the latest one as half, the percentile rank, so a rate that has been
    unchanged for years is in the middle of its own ties rather than at their bottom. Highest Since
    is NaN when no earlier value was as high, which makes the latest value the highest of the
    history, and Lowest Since likewise. x(0) is the first observation; the change is relative to
    its absolute value, so a series that rises from a negative value has a positive change, and
    it is NaN when the first observation is zero. max(x) and min(x) run over every observation,
    the latest included, so the Highest equals the Latest Value when the series is at its peak and
    its change from it is 0. The Highest Date and Lowest Date are the first dates the series
    reached them; the changes from them are NaN when they are zero. The Multiple Since Start and
    Annualised Change (the compound annual rate between the first and the latest date) are NaN
    unless both values are positive. The Largest Drawdown is the largest fall from an earlier
    peak, with the dates of that peak and of the trough, and the Recovery Date the first date
    the series was back at the peak, NaN while it has not recovered (or when it never fell).
    With a level, such as an earlier peak, the First Above Level is the first date the series exceeded it ("the first
    month above the 1980 high") and the Above Level Since the date it has been above it since
    without a break, NaN when it never was or the latest value is not above it. The Last
    Above Level and Last At Or Below Level are the last earlier dates on either side of it
    ("the last week at or below 203,000"), NaN when no earlier observation was, and the
    Observations Above Level and Observations At Or Below Level count the observations on
    either side, the latest included, so they add up to the Observations.

    Args:
        levels (pd.Series): The values of a single series by date; missing values are left out.
        level (float | None, optional): A level to compare the series with. Defaults to None,
            which leaves out the Level and the rows that compare with it.
        difference (bool, optional): Whether the changes are differences (points) rather
            than relative changes, for rates and other series where a relative change has
            no meaning. Defaults to False.

    Returns:
        pd.Series: The Start Date, Start Value, Observations, Latest Date, Latest Value, Change
        Since Start (a decimal), Multiple Since Start, Annualised Change (a decimal),
        Percentile, Rank, Highest Since, Lowest Since, Highest, Highest Date, Change From Highest (a
        decimal), Lowest, Lowest Date, Change From Lowest (a decimal), Largest Drawdown (a
        decimal), Drawdown Peak Date (the last date at the peak before the trough), Drawdown
        Trough Date and Recovery Date, and with a level
        the Level, First Above Level, Above Level Since, Last Above Level, Last At Or Below
        Level, Observations Above Level and Observations At Or Below Level, the dates as
        text.
    """
    observed = levels.dropna()

    if observed.empty:
        raise ValueError("The series has no observations.")

    labels = observed.index.astype(str)
    values = observed.to_numpy(dtype=float)
    earlier = values[:-1]
    first = float(observed.iloc[0])
    highest = float(values.max())
    lowest = float(values.min())
    latest = float(observed.iloc[-1])

    as_high = np.flatnonzero(earlier >= latest)
    as_low = np.flatnonzero(earlier <= latest)

    # Years between the first and the latest observation, for the annualised change.
    try:
        dates = (
            observed.index.to_timestamp()
            if isinstance(observed.index, pd.PeriodIndex)
            else pd.DatetimeIndex(observed.index)
        )
        years = (dates[-1] - dates[0]).days / 365.25
    except (TypeError, ValueError):
        years = np.nan
    positive = first > 0 and latest > 0 and not difference
    multiple = latest / first if positive else np.nan
    annualised = (
        multiple ** (1 / years) - 1 if positive and years and years > 0 else np.nan
    )

    # The largest fall from a running peak, and the first date the peak was regained.
    running_peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        falls = (
            values - running_peak
            if difference
            else np.where(
                running_peak != 0,
                (values - running_peak) / np.abs(running_peak),
                np.nan,
            )
        )
    trough = int(np.nanargmin(falls)) if np.isfinite(falls).any() else 0
    largest_fall = float(falls[trough]) if np.isfinite(falls[trough]) else np.nan
    # The decline starts from the last visit to the peak before the trough.
    peak = int(np.flatnonzero(values[: trough + 1] == running_peak[trough])[-1])
    regained = np.flatnonzero(values[trough + 1 :] >= running_peak[trough])
    drawdown = {
        "Largest Drawdown": largest_fall,
        "Drawdown Peak Date": labels[peak] if largest_fall < 0 else np.nan,
        "Drawdown Trough Date": labels[trough] if largest_fall < 0 else np.nan,
        "Recovery Date": (
            labels[trough + 1 + regained[0]]
            if largest_fall < 0 and regained.size
            else np.nan
        ),
    }

    above_level = {}
    if level is not None:
        above = np.flatnonzero(values > level)
        below = np.flatnonzero(values <= level)
        earlier_above = np.flatnonzero(earlier > level)
        earlier_below = np.flatnonzero(earlier <= level)
        above_level = {
            "Level": level,
            "First Above Level": labels[above[0]] if above.size else np.nan,
            "Above Level Since": (
                labels[below[-1] + 1 if below.size else 0]
                if values[-1] > level
                else np.nan
            ),
            # The earlier observations only, so it is the last time before the latest one.
            "Last Above Level": (
                labels[earlier_above[-1]] if earlier_above.size else np.nan
            ),
            "Last At Or Below Level": (
                labels[earlier_below[-1]] if earlier_below.size else np.nan
            ),
            "Observations Above Level": int(above.size),
            "Observations At Or Below Level": int(below.size),
        }

    def change(start: float) -> float:
        # In points, or relative to the absolute start (NaN from zero).
        if difference:
            return latest - start
        return (latest - start) / abs(start) if start else np.nan

    return pd.Series(
        {
            "Start Date": labels[0],
            "Start Value": first,
            "Observations": len(observed),
            "Latest Date": labels[-1],
            "Latest Value": latest,
            "Change Since Start": change(first),
            "Multiple Since Start": multiple,
            "Annualised Change": annualised,
            "Percentile": (
                (np.sum(earlier < latest) + 0.5 * np.sum(earlier == latest))
                / len(earlier)
                if len(earlier)
                else np.nan
            ),
            # 1 for the highest of the history; equal earlier values share the rank.
            "Rank": int(np.sum(earlier > latest)) + 1,
            "Highest Since": labels[as_high[-1]] if as_high.size else np.nan,
            "Lowest Since": labels[as_low[-1]] if as_low.size else np.nan,
            "Highest": highest,
            "Highest Date": labels[int(np.argmax(values))],
            "Change From Highest": change(highest),
            "Lowest": lowest,
            "Lowest Date": labels[int(np.argmin(values))],
            "Change From Lowest": change(lowest),
            **drawdown,
            **above_level,
        },
        dtype=object,
    )


def get_window_change_by_year(
    levels: pd.Series, from_month: int, to_month: int, difference: bool = False
) -> pd.DataFrame:
    """
    Calculate the change of a series over the same months of every year, such as February
    to October, with the rank of every year's change and the last earlier year with a
    change at least as large ("the largest since").

    The formula is as follows:

    - Change (y) = (x(to_month, y) - x(from_month, y)) / |x(from_month, y)|
    - Rank (y) = the position of Change (y) among the changes of every year, 1 for the largest
    - Largest Since (y) = the last year s < y with Change (s) >= Change (y)

    Where x(m, y) is the value of month m in year y. A window that ends in an earlier month
    than it starts (November to February) runs into the next year and is labelled by the
    year it ends; the same month for both is the change over twelve months. A year without
    the value of either month has no change, so the current year is empty until its last
    month is published. Largest Since is <NA> when no earlier year had a change as large,
    which makes the change the largest of the history up to that year.

    Args:
        levels (pd.Series): The monthly values of a single series, indexed by a monthly
            PeriodIndex; missing values are left out.
        from_month (int): The month the window starts, 1 (January) to 12 (December).
        to_month (int): The month the window ends, 1 to 12.
        difference (bool, optional): Whether the Change is the difference x(to_month, y) -
            x(from_month, y) (points, for a rate) rather than the relative change. Defaults
            to False.

    Returns:
        pd.DataFrame: Per year the Start Value, End Value, Change (decimal), Rank and
        Largest Since, the last two as whole numbers (Int64) that are <NA> when missing.

    Raises:
        ValueError: When a month is not between 1 and 12.
    """
    if not (1 <= from_month <= 12 and 1 <= to_month <= 12):  # noqa: PLR2004
        raise ValueError("from_month and to_month must be between 1 and 12.")

    observed = levels.dropna()
    months = observed.index.month

    start = observed[months == from_month]
    end = observed[months == to_month]

    # A window that ends in an earlier month than it starts ends in the next year.
    start.index = start.index.year + int(to_month <= from_month)
    end.index = end.index.year

    window = pd.DataFrame({"Start Value": start, "End Value": end}).sort_index()
    window["Change"] = window["End Value"] - window["Start Value"]
    if not difference:
        window["Change"] /= window["Start Value"].abs().replace(0, np.nan)
    window["Rank"] = window["Change"].rank(ascending=False, method="min")

    changes = window["Change"].to_numpy()
    years = window.index.to_numpy()
    window["Largest Since"] = [
        (
            next(
                (
                    years[earlier]
                    for earlier in range(position - 1, -1, -1)
                    if changes[earlier] >= change
                ),
                np.nan,
            )
            if not np.isnan(change)
            else np.nan
        )
        for position, change in enumerate(changes)
    ]
    window.index.name = "Year"

    # Whole numbers (a rank and a year), with <NA> for a year without a change or without
    # an earlier change as large, rather than floats such as 2022.0.
    window["Rank"] = window["Rank"].astype("Int64")
    window["Largest Since"] = window["Largest Since"].astype("Int64")

    return window


def get_streaks(levels: pd.Series, level: float | None = None) -> pd.Series:
    """
    Calculate the streak a series is on: the sign of its latest value, since when the series
    has had that sign without a break, and how that run compares with the earlier ones. With
    a level the runs are above, below or at that level instead ("27 years below 2%").

    A run is a stretch of consecutive observations with the same sign (positive, negative or
    zero). The current run is the one the latest observation belongs to, the previous run the
    one before it, and the longest earlier run the longest run of the current sign before the
    current one (the most recent of them when several are as long).

    Args:
        levels (pd.Series): The values of a single series by date; missing values are left out.
        level (float | None, optional): The level the runs are measured against. Defaults to
            None, which is zero: the sign of the values.

    Returns:
        pd.Series: The Start Date and Observations of the history, the Latest Date and Latest
        Value, the Sign of the current run with its Run Start and Run Length, the Previous
        Sign, Previous Run Start, Previous Run End and Previous Run Length, and the Longest
        Earlier Run Start, Longest Earlier Run End and Longest Earlier Run Length, the dates
        as text and the lengths in observations. A run that does not exist is NaN.
    """
    observed = levels.dropna()

    if observed.empty:
        raise ValueError("The series has no observations.")

    labels = observed.index.astype(str)
    signs = np.sign(observed.to_numpy(dtype=float) - (level or 0.0))
    names = (
        {1.0: "Positive", -1.0: "Negative", 0.0: "Zero"}
        if level is None
        else {1.0: "Above Level", -1.0: "Below Level", 0.0: "At Level"}
    )

    starts = np.flatnonzero(np.r_[True, signs[1:] != signs[:-1]])
    ends = np.r_[starts[1:] - 1, len(signs) - 1]
    lengths = ends - starts + 1

    same_sign = np.flatnonzero(signs[starts[:-1]] == signs[-1])
    # The most recent of the longest runs, as "the longest since" reads from today.
    longest = (
        same_sign[::-1][np.argmax(lengths[same_sign][::-1])] if same_sign.size else None
    )
    previous = len(starts) - 2 if len(starts) > 1 else None

    return pd.Series(
        {
            "Start Date": labels[0],
            "Observations": len(observed),
            "Latest Date": labels[-1],
            "Latest Value": float(observed.iloc[-1]),
            "Sign": names[signs[-1]],
            "Run Start": labels[starts[-1]],
            "Run Length": int(lengths[-1]),
            "Previous Sign": (
                names[signs[starts[previous]]] if previous is not None else np.nan
            ),
            "Previous Run Start": (
                labels[starts[previous]] if previous is not None else np.nan
            ),
            "Previous Run End": (
                labels[ends[previous]] if previous is not None else np.nan
            ),
            "Previous Run Length": (
                int(lengths[previous]) if previous is not None else np.nan
            ),
            "Longest Earlier Run Start": (
                labels[starts[longest]] if longest is not None else np.nan
            ),
            "Longest Earlier Run End": (
                labels[ends[longest]] if longest is not None else np.nan
            ),
            "Longest Earlier Run Length": (
                int(lengths[longest]) if longest is not None else np.nan
            ),
        },
        dtype=object,
    )
