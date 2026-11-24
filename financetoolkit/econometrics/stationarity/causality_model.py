"""Granger Causality Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.econometrics.least_squares_model import (
    fit_least_squares,
    get_lagged_values,
)

SIGNIFICANCE_LEVEL = 0.05


def get_granger_causality(
    series_a: pd.Series, series_b: pd.Series, max_lag: int = 5
) -> pd.Series:
    """
    Calculate the Granger causality test: whether the past values of one series help predict
    another beyond that series' own past.

    "Granger causality" is a statement about predictive power, not true causation: `series_b`
    is said to Granger-cause `series_a` if past values of `series_b`, combined with past
    values of `series_a` itself, predict `series_a` significantly better than past values of
    `series_a` alone.

    The test compares two regressions of `series_a` on its own `max_lag` lags:

    - Restricted: y_t = c + SUM(alpha_i * y_(t-i))
    - Unrestricted: y_t = c + SUM(alpha_i * y_(t-i)) + SUM(beta_i * x_(t-i))

    Via an F-test on whether adding the lags of `series_b` (the beta coefficients)
    significantly reduces the residual sum of squares:

    - F = ((RSS_restricted - RSS_unrestricted) / p) / (RSS_unrestricted / (m - 2p - 1))

    With `p` the `max_lag` and `m = n - p` the observations both regressions use (the
    first `p` only serve as lags), so the denominator has the degrees of freedom of the
    unrestricted regression. Unlike the (A)DF/EG family of tests, this comparison
    follows a standard F-distribution, so an exact p-value is available.

    For more information about the method, see the following paper:

    - Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
    and Cross-Spectral Methods." Econometrica, 37(3), 424-438.

    Also known as: Granger causality test, predictive causality.

    Referenced against:
    - statsmodels, grangercausalitytests: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html

    Args:
        series_a (pd.Series): The series being predicted (the dependent series).
        series_b (pd.Series): The series being tested for predictive power over `series_a`.
        max_lag (int): The number of lags of both series to include in the regressions. Defaults to 5.

    Returns:
        pd.Series: The F-statistic, its p-value, and whether `series_b` is found to
        Granger-cause `series_a` at the 5% level. NaN when there are no more than
        `3 * max_lag + 1` observations, when a lagged series is constant (its lags
        duplicate the constant) or when the unrestricted regression fits the data
        perfectly, which leaves no residual variance to test against.

    Notes:
        - The F-test is the ssr_ftest of statsmodels' grangercausalitytests.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    data = aligned.to_numpy(dtype=float)
    nan_result = pd.Series(
        {
            "F-Statistic": np.nan,
            "P-Value": np.nan,
            "Granger-Causes (5%)": False,
        }
    )

    if max_lag <= 0 or len(data) <= 3 * max_lag + 1 or not np.isfinite(data).all():
        return nan_result

    target = data[max_lag:, 0]
    own_lags = get_lagged_values(data[:, 0], max_lag)
    ones = np.ones((len(target), 1))
    restricted_design = np.column_stack([own_lags, ones])
    unrestricted_design = np.column_stack(
        [own_lags, get_lagged_values(data[:, 1], max_lag), ones]
    )

    # A constant lagged column duplicates the intercept, which leaves the test undefined.
    constant_columns = unrestricted_design.max(axis=0) == unrestricted_design.min(
        axis=0
    )
    if constant_columns.sum() != 1:
        return nan_result

    restricted = fit_least_squares(target, restricted_design)
    unrestricted = fit_least_squares(target, unrestricted_design)
    residual_sum_of_squares = unrestricted["sum_of_squared_residuals"]
    total_sum_of_squares = np.sum((target - target.mean()) ** 2)

    if (
        total_sum_of_squares == 0
        or residual_sum_of_squares == 0
        or residual_sum_of_squares / total_sum_of_squares < np.finfo(float).eps
        or unrestricted["rank"] != unrestricted_design.shape[1]
    ):
        return nan_result

    degrees_of_freedom = unrestricted["degrees_of_freedom"]
    f_statistic = (
        (restricted["sum_of_squared_residuals"] - residual_sum_of_squares)
        / residual_sum_of_squares
        / max_lag
        * degrees_of_freedom
    )
    p_value = stats.f.sf(f_statistic, max_lag, degrees_of_freedom)

    return pd.Series(
        {
            "F-Statistic": f_statistic,
            "P-Value": p_value,
            "Granger-Causes (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
