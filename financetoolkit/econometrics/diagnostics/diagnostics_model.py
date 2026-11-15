"""Risk Diagnostics Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.stats.stattools import jarque_bera

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# The Variance Ratio test requires at least a two-period compounding horizon.
MINIMUM_VARIANCE_RATIO_Q = 2

# `statsmodels.stats.stattools.jarque_bera` requires at least 2 observations.
MINIMUM_JARQUE_BERA_OBSERVATIONS = 2

# The CUSUM test compares the path of the residuals with the asymptotic boundaries of a
# Brownian bridge, which a handful of residuals cannot follow; shorter series return NaN.
MINIMUM_CUSUM_OBSERVATIONS = 10

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
            # het_arch calls acorr_lm, which on statsmodels 0.15 announces a result object for 0.16; the tuple used here is unaffected until then.  # noqa: E501
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="acorr_lm currently returns",
                    category=FutureWarning,
                )
                lm_statistic, p_value, _, _ = het_arch(residuals, nlags=lags)
        except (ValueError, np.linalg.LinAlgError):
            return pd.Series({"ARCH-LM Statistic": np.nan, "P-Value": np.nan})

        return pd.Series({"ARCH-LM Statistic": lm_statistic, "P-Value": p_value})

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_jarque_bera_test(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
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
    returns: pd.Series | pd.DataFrame, lags: int = 10, model_df: int = 0
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Ljung-Box test for autocorrelation, via
    `statsmodels.stats.diagnostic.acorr_ljungbox`.

    The test aggregates the squared Autocorrelation Function up to lag `h` into a
    single statistic that is asymptotically chi-squared distributed with `h - model_df`
    degrees of freedom under the null hypothesis that the series exhibits no
    autocorrelation up to that lag:

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
        model_df (int, optional): The number of ARMA parameters estimated on the series
        that produced `returns`, subtracted from the chi-squared degrees of freedom.
        Defaults to 0, which is correct for an OBSERVED series (raw returns). It is NOT
        correct for the residuals of a fitted model: fitting `p + q` ARMA parameters uses
        up `p + q` of the autocorrelations the statistic aggregates, so leaving this at 0
        inflates the degrees of freedom and makes the test conservative (it under-rejects,
        reporting a larger p-value than it should). Pass `p + q` when testing ARIMA/GARCH
        residuals -- see Box, Pierce (1970) and Ljung, Box (1978).

    Returns:
        pd.Series | pd.DataFrame: The Ljung-Box statistic and its p-value.

    Raises:
        TypeError: If `returns` is not a pd.DataFrame or pd.Series.
        ValueError: If `model_df` is negative or leaves no degrees of freedom
        (`model_df >= lags`).
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
                    returns.loc[sub_period], lags=lags, model_df=model_df
                )

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_ljung_box_test(
                    returns[column], lags=lags, model_df=model_df
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        values = returns.dropna()
        n = len(values)

        if n <= lags + 1:
            return pd.Series({"Ljung-Box Statistic": np.nan, "P-Value": np.nan})

        try:
            result = acorr_ljungbox(values.to_numpy(), lags=[lags], model_df=model_df)
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


def get_cusum_test(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
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

    With difference=True the changes are differences instead, in the unit of the series
    (percentage points for a rate): Change Since Start = x(T) - x(0), Change From Highest =
    x(T) - max(x), Change From Lowest = x(T) - min(x) and Largest Drawdown = min over t of
    (x(t) - max(x(s), s <= t)); the Multiple Since Start and Annualised Change are NaN.
    - Above Level Since = the first date of the run of observations above the level that
      the latest one belongs to

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
    without a break, NaN when it never was or the latest value is not above it.

    Args:
        levels (pd.Series): The values of a single series by date; missing values are left out.
        level (float | None, optional): A level to compare the series with. Defaults to None,
            which leaves out the Level, First Above Level and Above Level Since.
        difference (bool, optional): Whether the changes are differences (points) rather
            than relative changes, for rates and other series where a relative change has
            no meaning. Defaults to False.

    Returns:
        pd.Series: The Start Date, Start Value, Observations, Latest Date, Latest Value, Change
        Since Start (a decimal), Multiple Since Start, Annualised Change (a decimal),
        Percentile, Rank, Highest Since, Lowest Since, Highest, Highest Date, Change From Highest (a
        decimal), Lowest, Lowest Date, Change From Lowest (a decimal), Largest Drawdown (a
        decimal), Drawdown Peak Date, Drawdown Trough Date and Recovery Date, and with a level
        the Level, First Above Level and Above Level Since, the dates as text.
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
    peak = int(np.flatnonzero(values[: trough + 1] == running_peak[trough])[0])
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
        above_level = {
            "Level": level,
            "First Above Level": labels[above[0]] if above.size else np.nan,
            "Above Level Since": (
                labels[below[-1] + 1 if below.size else 0]
                if values[-1] > level
                else np.nan
            ),
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
    month is published. Largest Since is NaN when no earlier year had a change as large,
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
        Largest Since.

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
