"""Granger Causality Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

MINIMUM_USABLE_OBSERVATIONS_BUFFER = 3
SIGNIFICANCE_LEVEL = 0.05


def get_granger_causality(
    series_a: pd.Series, series_b: pd.Series, max_lag: int = 5
) -> pd.Series:
    """
    Calculate the Granger causality test for whether `series_b` helps predict `series_a`.

    "Granger causality" is a statement about predictive power, not true causation: `series_b`
    is said to Granger-cause `series_a` if past values of `series_b`, combined with past
    values of `series_a` itself, predict `series_a` significantly better than past values of
    `series_a` alone.

    The test compares two regressions of `series_a` on its own `max_lag` lags:

    - Restricted: y_t = c + SUM(alpha_i * y_(t-i))
    - Unrestricted: y_t = c + SUM(alpha_i * y_(t-i)) + SUM(beta_i * x_(t-i))

    Via an F-test on whether adding the lags of `series_b` (the beta coefficients)
    significantly reduces the residual sum of squares. Unlike the (A)DF/EG family of
    tests, this comparison follows a standard F-distribution, so an exact p-value is
    available.

    For more information about the method, see the following paper:

    - Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
    and Cross-Spectral Methods." Econometrica, 37(3), 424-438.

    Also known as: Granger causality test, predictive causality.

    Args:
        series_a (pd.Series): The series being predicted (the dependent series).
        series_b (pd.Series): The series being tested for predictive power over `series_a`.
        max_lag (int): The number of lags of both series to include in the regressions. Defaults to 5.

    Returns:
        pd.Series: The F-statistic, its p-value, and whether `series_b` is found to
        Granger-cause `series_a` at the 5% level.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1].to_numpy()
    n = len(y)

    usable = n - max_lag
    minimum_usable = 2 * max_lag + MINIMUM_USABLE_OBSERVATIONS_BUFFER

    if usable <= minimum_usable:
        return pd.Series(
            {
                "F-Statistic": np.nan,
                "P-Value": np.nan,
                "Granger-Causes (5%)": False,
            }
        )

    y_dependent = y[max_lag:]

    own_lags = np.column_stack(
        [y[max_lag - i : max_lag - i + usable] for i in range(1, max_lag + 1)]
    )
    other_lags = np.column_stack(
        [x[max_lag - i : max_lag - i + usable] for i in range(1, max_lag + 1)]
    )

    restricted_design = np.column_stack([own_lags, np.ones(usable)])
    unrestricted_design = np.column_stack([own_lags, other_lags, np.ones(usable)])

    restricted_coefficients, _, _, _ = np.linalg.lstsq(
        restricted_design, y_dependent, rcond=None
    )
    unrestricted_coefficients, _, _, _ = np.linalg.lstsq(
        unrestricted_design, y_dependent, rcond=None
    )

    restricted_residual_sum_of_squares = np.sum(
        (y_dependent - restricted_design @ restricted_coefficients) ** 2
    )
    unrestricted_residual_sum_of_squares = np.sum(
        (y_dependent - unrestricted_design @ unrestricted_coefficients) ** 2
    )

    degrees_of_freedom_denominator = usable - unrestricted_design.shape[1]

    f_statistic = (
        (restricted_residual_sum_of_squares - unrestricted_residual_sum_of_squares)
        / max_lag
    ) / (unrestricted_residual_sum_of_squares / degrees_of_freedom_denominator)
    f_statistic = max(f_statistic, 0.0)

    p_value = stats.f.sf(f_statistic, max_lag, degrees_of_freedom_denominator)

    return pd.Series(
        {
            "F-Statistic": f_statistic,
            "P-Value": p_value,
            "Granger-Causes (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
