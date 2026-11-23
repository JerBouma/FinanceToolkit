"""Risk Diagnostics Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.econometrics.least_squares_model import (
    fit_least_squares,
    get_autocorrelations,
    get_kolmogorov_survival,
    get_lagged_values,
)
from financetoolkit.risk.volatility import garch_model

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
    Calculate Engle's Lagrange Multiplier (LM) test for ARCH effects, or the weighted
    portmanteau ARCH test of Fisher and Gallagher (2012).

    The test regresses squared, mean-demeaned returns on a constant and `lags` of
    themselves and tests whether the resulting R-squared is significantly different from
    zero. Under the null hypothesis of no ARCH effects, the LM statistic is asymptotically
    chi-squared distributed with `lags` degrees of freedom:

    - e_t = r_t - mean(r)
    - e_t^2 = a_0 + SUM_{i=1}^{q}(a_i * e_(t-i)^2) + v_t, for t = q + 1, ..., n
    - LM = (n - q) * R^2

    Where `q` is `lags` and `n - q` the number of observations of the regression.

    A significant result (low p-value) indicates that the return series exhibits
    volatility clustering, and a GARCH-family model is an appropriate choice for it.

    With `weighted=True` the statistic is the weighted portmanteau test of Fisher and
    Gallagher (2012) on the autocorrelations of the squared series (the Li-Mak test with
    weights that fall linearly with the lag), which keeps its size better for many lags:

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
        Referenced against statsmodels' `het_arch` and, for the weighted test, the weighted
        ARCH-LM test of R's rugarch (`Weighted.LM.test`,
        https://cran.r-project.org/package=rugarch), which reproduces the published values
        of the rugarch vignette.
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
    Calculate the Jarque-Bera test for normality.

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

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: The Jarque-Bera statistic and its p-value.

    Notes:
        Referenced against statsmodels' `jarque_bera`.
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
    Calculate the Ljung-Box test for autocorrelation, or the weighted Ljung-Box test of
    Fisher and Gallagher (2012).

    The test aggregates the squared Autocorrelation Function up to lag `h` into a
    single statistic that is asymptotically chi-squared distributed with `h - model_df`
    degrees of freedom under the null hypothesis that the series exhibits no
    autocorrelation up to that lag:

    - Q = n * (n + 2) * SUM_{k=1}^{h} (rho_k^2 / (n - k))

    - rho_k = SUM_{t=k+1}^{n}((x_t - mean(x)) * (x_(t-k) - mean(x))) / SUM_{t=1}^{n}((x_t - mean(x))^2)

    Where `rho_k` is the Autocorrelation at lag `k` (every autocovariance divided by `n`)
    and `n` is the number of observations.

    With `weighted=True` every squared autocorrelation is weighted by (h - k + 1) / h, so the
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
        Referenced against statsmodels' `acorr_ljungbox` and, for the weighted test, the
        weighted Ljung-Box test of R's rugarch (`Weighted.Box.test`,
        https://cran.r-project.org/package=rugarch), which reproduces the published values
        of the rugarch vignette.
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

        if n <= lags + 1:
            return pd.Series({name: np.nan, "P-Value": np.nan})

        autocorrelations = get_autocorrelations(values.to_numpy(dtype=float), lags)

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
    Calculate the Lo-MacKinlay Variance Ratio test for the random walk hypothesis.

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
    Calculate a CUSUM test for the stability of the returns over time: the CUSUM of the OLS
    residuals (Ploberger and Krämer, 1992), the CUSUM of the recursive residuals (Brown, Durbin
    and Evans, 1975) or the CUSUM of the log price level (Chu, Stinchcombe and White, 1996, as
    López de Prado, 2018, section 17.3.2 defines it).

    method="ols" fits a constant-mean model (`returns_t = mu + e_t`) by OLS and cumulates
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

    method="recursive" cumulates the one-step-ahead forecast errors of the mean of the
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

    method="chu_stinchcombe_white" works on the log price y_t = SUM ln(1 + r_t) (the log of
    the price up to a constant) and asks whether it has risen by more since an earlier date
    than a random walk with the volatility up to that date allows. It is one-sided, a test
    for a rise, as López de Prado (2018) defines it:

    - S_(n,t) = (y_t - y_n) / (sigma_t * sqrt(t - n)), with sigma_t^2 = SUM_{i=1}^{t}((y_i -
      y_(i-1))^2) / t the mean squared change up to t
    - S_t = max over n < t of S_(n,t), at the reference date n*
    - c_t = sqrt(4.6 + ln(t - n*)), the 5% critical value of Homm and Breitung (2012)
    - Stability is rejected at t when S_t > c_t; the test is at the last date

    Every statistic of a date only uses the returns up to that date, so with
    `time_series=True` the statistic, its critical value and the verdict per date have no
    look-ahead; they start at the 10th return. The 5% holds for every date on its own (on
    random walks the last date rejects about 5.5% of the time), so over many dates some
    cross the critical value by chance. The other two methods scale with the whole sample and
    only give the test.

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

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        method (str, optional): "ols", "recursive" or "chu_stinchcombe_white". Defaults to
            "ols".
        time_series (bool, optional): Whether to return the statistic, its critical value
            and the verdict per date (method="chu_stinchcombe_white" only). Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: The CUSUM statistic, its p-value, the number of
        observations used, the 1%/5%/10% critical values, and whether stability is rejected
        at the 5% level; for method="chu_stinchcombe_white" the statistic, the 5% critical
        value, the observations and the verdict at the last date, or with time_series=True
        per date.

    Notes:
        method="ols" is referenced against statsmodels' `breaks_cusumolsresid` with `ddof=1`
        and method="recursive" against statsmodels' `recursive_olsresiduals`, which scales
        the path with the standard deviation of the recursive residuals around their mean
        rather than with the residual variance of Brown, Durbin and Evans (1975).
        method="chu_stinchcombe_white" follows López de Prado (2018, "Advances in Financial
        Machine Learning", Wiley, section 17.3.2): it divides by sigma_t, with sigma_t^2 the
        mean of the t - 1 squared changes, and compares with sqrt(4.6 + ln(t - n)) at 5%.
    """
    if method not in CUSUM_METHODS:
        raise ValueError(
            f"The method must be one of {', '.join(CUSUM_METHODS)}, not {method!r}."
        )
    if time_series and method != "chu_stinchcombe_white":
        raise ValueError(
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

    Args:
        residuals (pd.Series | pd.DataFrame): The residuals e_t of the mean model.
        standardized_residuals (pd.Series | pd.DataFrame): The standardized residuals
            z_t = e_t / sigma_t, with the same index (and columns).

    Returns:
        pd.Series | pd.DataFrame: The t-statistic and p-value of the sign bias, negative
        size bias and positive size bias, the joint statistic and its p-value, and the
        observations of the regression.

    Notes:
        Referenced against `signbias` of R's rugarch
        (https://cran.r-project.org/package=rugarch), which reproduces the t-statistics of
        the rugarch vignette. rugarch reports the absolute t-statistics and a Wald statistic
        with the OLS covariance as the joint test; this reports the signed t-statistics and
        the n * R^2 Lagrange multiplier statistic of Engle and Ng (1993).
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
    distribution the model assumes. Each residual is turned into the probability of a value
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

    Args:
        standardized_residuals (pd.Series | pd.DataFrame): The standardized residuals.
        bins (int, optional): The number of bins g. Defaults to 20.
        distribution (str, optional): The standardized distribution, see
            `garch_model.get_standardized_log_density`. Defaults to "normal".
        parameters (tuple[float, ...], optional): The Shape (and Skew) of the distribution.
            Defaults to (), for the normal distribution.

    Returns:
        pd.Series | pd.DataFrame: The Pearson statistic, its p-value, the bins and the
        observations.

    Notes:
        Referenced against `gof` of R's rugarch (https://cran.r-project.org/package=rugarch).
        rugarch leaves the empty bins above the last occupied one out of the sum (its
        `tabulate` stops at the highest bin with a value); every empty bin counts here.
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

    With difference=True the changes are differences instead, in the unit of the series
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
    With a level, such as an
    earlier peak, the First Above Level is the first date the series exceeded it ("the first
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
