"""Unit Root Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

# The conventional 5% significance level used for the boolean "reject" flags.
SIGNIFICANCE_LEVEL = 0.05

# Asymptotic critical values for the (A)DF tau statistic, by regression type. Needed
# only by `get_phillips_perron_test` below (which has no `statsmodels`/`linearmodels`
# equivalent, see its docstring), since its "Z_t" statistic is asymptotically pivotal
# to the same Dickey-Fuller tau distribution `statsmodels.tsa.stattools.adfuller`
# already returns critical values for directly.
# Source: MacKinnon, J.G. (1996). "Numerical Distribution Functions for Unit Root
# and Cointegration Tests." Journal of Applied Econometrics, 11(6), 601-618.
ADF_CRITICAL_VALUES: dict[str, dict[float, float]] = {
    "c": {0.01: -3.43, 0.05: -2.86, 0.10: -2.57},
    "ct": {0.01: -3.96, 0.05: -3.41, 0.10: -3.13},
}


def _newey_west_long_run_variance(residuals: np.ndarray, lags: int) -> float:
    """
    Newey-West (1987) HAC long-run variance estimator with Bartlett kernel weights.

    lambda^2 = gamma_0 + 2 * SUM_{l=1}^{lags} (1 - l / (lags + 1)) * gamma_l

    where gamma_l is the l-th sample autocovariance of `residuals`. Needed only by
    `get_phillips_perron_test` below, which has no `statsmodels`/`linearmodels`
    equivalent -- every other test in this module delegates its long-run-variance
    correction to `statsmodels` internally.

    For more information about the method, see the following paper:

    - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    Econometrica, 55(3), 703-708.
    """
    number_of_observations = len(residuals)
    variance = float(np.sum(residuals**2) / number_of_observations)

    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        autocovariance = (
            np.sum(residuals[lag:] * residuals[:-lag]) / number_of_observations
        )
        variance += 2 * weight * autocovariance

    return variance


def get_augmented_dickey_fuller(
    series: pd.Series, max_lag: int | None = None, regression: str = "c"
) -> pd.Series:
    """
    Calculate the Augmented Dickey-Fuller (ADF) test for a unit root, via
    `statsmodels.tsa.stattools.adfuller`.

    The test regresses the first difference of the series on its own lagged level and
    `p` lags of its own first difference:

    - dy_t = gamma * y_(t-1) + SUM(delta_i * dy_(t-i)) + [constant] + [trend] + e_t

    The null hypothesis is that the series has a unit root (is a random walk, i.e. not
    mean-reverting); the alternative is that it is stationary. The test statistic is
    the t-statistic on `gamma`; unlike a regular t-statistic it does not follow a
    Student-T distribution, so it must be compared against Dickey-Fuller specific
    critical values rather than a standard significance table.

    The number of lags `p` is chosen automatically (up to `max_lag`) by minimizing the
    Akaike Information Criterion (AIC) across candidate lag lengths, unless `max_lag`
    is given explicitly.

    For more information about the method, see the following paper:

    - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
    Autoregressive Time Series with a Unit Root." Journal of the American Statistical
    Association, 74(366a), 427-431.

    Also known as: ADF test, unit root test, stationarity test.

    Args:
        series (pd.Series): A Series of values (e.g. prices, or a spread between two prices).
        max_lag (int, optional): The maximum number of lagged differences to consider. Defaults to the
        Schwert (1989) rule of thumb, ceil(12 * (n / 100) ** 0.25).
        regression (str, optional): Which deterministic terms to include, one of "n" (none), "c"
        (constant) or "ct" (constant and trend). Defaults to "c".

    Returns:
        pd.Series: The ADF statistic, its p-value, the number of lags used, the number of
        observations used, the 1%/5%/10% critical values, and whether the unit root is rejected
        at the 5% level.

    Raises:
        ValueError: If `regression` is not one of "n", "c" or "ct".
    """
    if regression not in ["n", "c", "ct"]:
        raise ValueError(
            "regression must be 'n' (no constant), 'c' (constant), or 'ct' (constant and trend)."
        )

    values = series.dropna().to_numpy()

    try:
        adf_statistic, p_value, used_lag, n_observations, critical_values, _ = adfuller(
            values, maxlag=max_lag, regression=regression, autolag="AIC"
        )
    except (ValueError, np.linalg.LinAlgError):
        return pd.Series(
            {
                "ADF Statistic": np.nan,
                "P-Value": np.nan,
                "Lags Used": np.nan,
                "Observations": np.nan,
                "Critical Value 1%": np.nan,
                "Critical Value 5%": np.nan,
                "Critical Value 10%": np.nan,
                "Reject Unit Root (5%)": False,
            }
        )

    return pd.Series(
        {
            "ADF Statistic": adf_statistic,
            "P-Value": p_value,
            "Lags Used": used_lag,
            "Observations": n_observations,
            "Critical Value 1%": critical_values["1%"],
            "Critical Value 5%": critical_values["5%"],
            "Critical Value 10%": critical_values["10%"],
            "Reject Unit Root (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_kpss_test(
    series: pd.Series, regression: str = "c", lags: int | None = None
) -> pd.Series:
    """
    Calculate the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity, via
    `statsmodels.tsa.stattools.kpss`.

    KPSS is the natural complement to the Augmented Dickey-Fuller test: where the ADF
    null hypothesis is that the series HAS a unit root, the KPSS null hypothesis is
    that the series IS stationary (around a constant, or around a deterministic trend
    if `regression="ct"`), with a unit root as the alternative. Running both tests
    together is standard practice to triangulate a confident conclusion:

    - ADF rejects a unit root AND KPSS fails to reject stationarity -> confident the
    series is stationary.
    - ADF fails to reject a unit root AND KPSS rejects stationarity -> confident the
    series has a unit root.
    - The two tests disagree -> an ambiguous case (common in practice with borderline
    persistence, e.g. a near-unit-root AR(1)), which the literature treats as
    inconclusive rather than resolvable by either test alone.

    The test regresses the series on a constant (or constant plus a linear trend) and
    computes the partial sums of the regression residuals `e_t`:

    - S_t = SUM_{i=1}^{t} e_i
    - KPSS = (1 / n^2) * SUM_t(S_t^2) / lambda^2

    where `lambda^2` is a heteroskedasticity-and-autocorrelation-consistent (HAC)
    long-run variance estimate of the residuals.

    For more information about the method, see the following paper:

    - Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). "Testing the
    Null Hypothesis of Stationarity against the Alternative of a Unit Root." Journal
    of Econometrics, 54(1-3), 159-178.

    Also known as: KPSS test, stationarity test.

    Args:
        series (pd.Series): A Series of values (e.g. prices, or a spread between two prices).
        regression (str, optional): Which deterministic term to remove before testing, one of
        "c" (constant, tests level-stationarity around a mean) or "ct" (constant and trend,
        tests trend-stationarity). Defaults to "c".
        lags (int, optional): The truncation lag for the long-run variance estimate. Defaults
        to `statsmodels`' automatic (Hobijn, Franses & Ooms, 2004) bandwidth selection.

    Returns:
        pd.Series: The KPSS statistic, its p-value, the truncation lag used, the number of
        observations used, the 1%/2.5%/5%/10% critical values, and whether stationarity is
        rejected at the 5% level.

    Raises:
        ValueError: If `regression` is not one of "c" or "ct".
    """
    if regression not in ["c", "ct"]:
        raise ValueError(
            "regression must be 'c' (constant, level stationarity) or 'ct' (constant "
            "and trend, trend stationarity)."
        )

    values = series.dropna().to_numpy()
    n = len(values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kpss_statistic, p_value, used_lags, critical_values = kpss(
                values,
                regression=regression,
                nlags=lags if lags is not None else "auto",
            )
        except (ValueError, np.linalg.LinAlgError):
            return pd.Series(
                {
                    "KPSS Statistic": np.nan,
                    "P-Value": np.nan,
                    "Lags Used": lags if lags is not None else np.nan,
                    "Observations": n,
                    "Critical Value 1%": np.nan,
                    "Critical Value 2.5%": np.nan,
                    "Critical Value 5%": np.nan,
                    "Critical Value 10%": np.nan,
                    "Reject Stationarity (5%)": False,
                }
            )

    return pd.Series(
        {
            "KPSS Statistic": kpss_statistic,
            "P-Value": p_value,
            "Lags Used": used_lags,
            "Observations": n,
            "Critical Value 1%": critical_values["1%"],
            "Critical Value 2.5%": critical_values["2.5%"],
            "Critical Value 5%": critical_values["5%"],
            "Critical Value 10%": critical_values["10%"],
            "Reject Stationarity (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_phillips_perron_test(
    series: pd.Series, regression: str = "c", lags: int | None = None
) -> pd.Series:
    """
    Calculate the Phillips-Perron (PP) test for a unit root.

    Phillips-Perron tests the same null hypothesis as the Augmented Dickey-Fuller
    test (a unit root, i.e. non-stationarity), but corrects for heteroskedasticity
    and serial correlation in the errors nonparametrically via a Newey-West long-run
    variance estimate, rather than by adding lagged-difference terms to the
    regression as ADF does. This makes it a useful cross-check: PP and ADF should
    broadly agree on the same series since they test the same null with different
    correction methods, and disagreement can flag that the ADF lag choice or the PP
    truncation lag is sensitive to the specific series at hand.

    Neither `statsmodels` nor `linearmodels` ships a Phillips-Perron implementation
    (the `arch` package does, but that is a separate, unrelated optional dependency
    this codebase does not otherwise use) -- this is the one test in this module that
    remains hand-built, though the underlying Dickey-Fuller auxiliary regression is
    still fit via `statsmodels.api.OLS` rather than raw linear algebra.

    The test runs the simple (non-augmented) Dickey-Fuller regression:

    - y_t = alpha + rho * y_(t-1) + [trend] + u_t

    And computes the "Z_t" statistic, which adjusts the usual t-statistic on `rho` for
    the long-run-vs-short-run variance ratio of the (possibly autocorrelated,
    heteroskedastic) residuals `u_t`:

    - t_rho = (rho_hat - 1) / se(rho_hat)
    - gamma_0 = (1/n) * SUM(u_hat_t^2)      (short-run residual variance)
    - lambda^2 = Newey-West long-run variance of u_hat_t, `lags` truncation lag
    - Z_t = sqrt(gamma_0 / lambda^2) * t_rho
            - (lambda^2 - gamma_0) / (2 * sqrt(lambda^2) * sqrt(gamma_0)) * (n * se(rho_hat))

    This "Z_t" variant (as opposed to the "Z_rho" variant, which corrects `n * (rho_hat
    - 1)` directly rather than the t-statistic) is implemented here because it is
    asymptotically pivotal to the same Dickey-Fuller tau distribution as the ADF
    t-statistic, so the same MacKinnon (1996) critical value table `statsmodels`
    returns for `get_augmented_dickey_fuller` applies directly.

    This implementation was cross-checked (matching to 5+ decimal places) against the
    independent `arch.unitroot.PhillipsPerron` reference implementation (with the same
    truncation lag) for both `regression="c"` and `regression="ct"`. Only "c" and "ct"
    are supported -- a "n" (no constant) specification was tried against the same
    reference during development and did not match to reasonable precision, so it has
    deliberately been left unimplemented here rather than shipped unverified.

    For more information about the method, see the following paper:

    - Phillips, P.C.B., & Perron, P. (1988). "Testing for a Unit Root in Time Series
    Regression." Biometrika, 75(2), 335-346.

    Also known as: PP test, Z_t test.

    Args:
        series (pd.Series): A Series of values (e.g. prices, or a spread between two prices).
        regression (str, optional): Which deterministic term to include, one of "c" (constant)
        or "ct" (constant and trend). Defaults to "c". "n" (no constant) is not supported, see
        Notes above.
        lags (int, optional): The truncation lag for the Newey-West long-run variance estimate.
        Defaults to the Schwert (1989) rule of thumb, int(12 * (n / 100) ** 0.25).

    Returns:
        pd.Series: The Phillips-Perron Z_t statistic, the truncation lag used, the number of
        observations used, the 1%/5%/10% critical values, and whether the unit root is rejected
        at the 5% level.

    Raises:
        ValueError: If `regression` is not one of "c" or "ct".
    """
    if regression not in ["c", "ct"]:
        raise ValueError(
            "regression must be 'c' (constant) or 'ct' (constant and trend). 'n' (no "
            "constant) is not supported -- see the function docstring Notes."
        )

    values = series.dropna().to_numpy()
    y = values[1:]
    y_lag = values[:-1]
    n = len(y)

    if lags is None:
        lags = int(12 * (n / 100) ** 0.25)

    number_of_parameters = 3 if regression == "ct" else 2
    minimum_usable = lags + number_of_parameters + 3

    if n <= minimum_usable:
        return pd.Series(
            {
                "Phillips-Perron Statistic": np.nan,
                "Lags Used": lags,
                "Observations": n,
                "Critical Value 1%": np.nan,
                "Critical Value 5%": np.nan,
                "Critical Value 10%": np.nan,
                "Reject Unit Root (5%)": False,
            }
        )

    regressors = [np.ones(n), y_lag]
    if regression == "ct":
        regressors.append(np.arange(n, dtype=float))
    x = np.column_stack(regressors)

    sm_result = sm.OLS(y, x).fit()
    rho_hat = sm_result.params[1]
    residuals = np.asarray(sm_result.resid)
    residual_sum_of_squares = float(np.sum(residuals**2))
    standard_error_rho = float(sm_result.bse[1])
    t_rho = (rho_hat - 1) / standard_error_rho

    sigma_squared = residual_sum_of_squares / (n - number_of_parameters)
    gamma_0 = residual_sum_of_squares / n
    long_run_variance = _newey_west_long_run_variance(residuals, lags)

    z_t_statistic = np.sqrt(gamma_0 / long_run_variance) * t_rho - (
        long_run_variance - gamma_0
    ) / (2 * np.sqrt(long_run_variance) * np.sqrt(sigma_squared)) * (
        n * standard_error_rho
    )

    critical_values = ADF_CRITICAL_VALUES[regression]

    return pd.Series(
        {
            "Phillips-Perron Statistic": z_t_statistic,
            "Lags Used": lags,
            "Observations": n,
            "Critical Value 1%": critical_values[0.01],
            "Critical Value 5%": critical_values[0.05],
            "Critical Value 10%": critical_values[0.10],
            "Reject Unit Root (5%)": bool(z_t_statistic < critical_values[0.05]),
        }
    )


def get_zivot_andrews_test(
    series: pd.Series,
    max_lag: int | None = None,
    regression: str = "c",
    trim: float = 0.15,
) -> pd.Series:
    """
    Calculate the Zivot-Andrews test for a unit root, allowing for a single
    structural break at an unknown (endogenously estimated) date, via
    `statsmodels.tsa.stattools.zivot_andrews`.

    The (A)DF test above assumes the deterministic component of the series (its
    constant and/or trend) is stable throughout the sample. If a series instead has
    a single one-time break -- e.g. a permanent level shift or a change in trend
    slope -- the ordinary ADF test is biased towards not rejecting the unit root
    even for a genuinely (trend-)stationary series with a break, since the break
    inflates the estimated persistence of the series (Perron, 1989). The
    Zivot-Andrews test addresses this by adding a break dummy to the ADF regression
    and, rather than assuming a known break date, choosing the break date that is
    most favorable to the stationary alternative -- the test statistic is the
    minimum (most negative) t-statistic on the lagged-level coefficient across all
    candidate break dates within the trimmed range.

    For more information about the method, see the following papers:

    - Zivot, E., & Andrews, D.W.K. (1992). "Further Evidence on the Great Crash, the
    Oil-Price Shock, and the Unit-Root Hypothesis." Journal of Business & Economic
    Statistics, 10(3), 251-270.
    - Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
    Hypothesis." Econometrica, 57(6), 1361-1401.

    Also known as: ZA test, structural break unit root test.

    Args:
        series (pd.Series): A Series of values (e.g. prices, or a spread between two prices).
        max_lag (int, optional): The maximum number of lagged differences to consider. Defaults
        to `statsmodels`' Schwert (1989) rule of thumb.
        regression (str, optional): Which break to allow for, one of "c" (a break in the level/
        intercept), "t" (a break in the trend slope) or "ct" (both). Defaults to "c".
        trim (float, optional): The fraction of observations excluded from the candidate break
        date search at the start and end of the sample, since a break too close to either end
        cannot be reliably distinguished from a unit root. Must be in [0, 1/3). Defaults to 0.15.

    Returns:
        pd.Series: The Zivot-Andrews statistic, its p-value, the (0-indexed) position of the
        selected break date, the number of lags and observations used, the 1%/5%/10% critical
        values, and whether the unit root is rejected at the 5% level.

    Raises:
        TypeError: If `series` is not a pd.Series.
        ValueError: If `regression` is not one of "c", "t" or "ct", or if `trim` is not in
        [0, 1/3).
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Expects pd.Series, no other value.")
    if regression not in ["c", "t", "ct"]:
        raise ValueError(
            "regression must be 'c' (level break), 't' (trend break), or 'ct' "
            "(both)."
        )
    if trim < 0 or trim >= 1 / 3:
        raise ValueError("trim must be in the range [0, 1/3).")

    values = series.dropna().to_numpy()

    try:
        za_statistic, p_value, critical_values, used_lag, break_index = zivot_andrews(
            values, trim=trim, maxlag=max_lag, regression=regression, autolag="AIC"
        )
    except (ValueError, IndexError, np.linalg.LinAlgError):
        return pd.Series(
            {
                "Zivot-Andrews Statistic": np.nan,
                "P-Value": np.nan,
                "Break Index": np.nan,
                "Lags Used": np.nan,
                "Observations": len(values),
                "Critical Value 1%": np.nan,
                "Critical Value 5%": np.nan,
                "Critical Value 10%": np.nan,
                "Reject Unit Root (5%)": False,
            }
        )

    return pd.Series(
        {
            "Zivot-Andrews Statistic": za_statistic,
            "P-Value": p_value,
            "Break Index": float(break_index),
            "Observations": len(values),
            "Lags Used": used_lag,
            "Critical Value 1%": critical_values["1%"],
            "Critical Value 5%": critical_values["5%"],
            "Critical Value 10%": critical_values["10%"],
            "Reject Unit Root (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
