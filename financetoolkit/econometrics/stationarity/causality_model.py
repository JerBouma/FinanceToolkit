"""Granger Causality Model"""

__docformat__ = "google"

import contextlib
import io

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

SIGNIFICANCE_LEVEL = 0.05


def get_granger_causality(
    series_a: pd.Series, series_b: pd.Series, max_lag: int = 5
) -> pd.Series:
    """
    Calculate the Granger causality test for whether `series_b` helps predict
    `series_a`, via `statsmodels.tsa.stattools.grangercausalitytests`.

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
    data = aligned.to_numpy()

    try:
        # Older statsmodels print a results table to stdout with no way to silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            result = grangercausalitytests(data, maxlag=[max_lag])
        f_statistic, p_value, _, _ = result[max_lag][0]["ssr_ftest"]
    except (ValueError, np.linalg.LinAlgError):
        return pd.Series(
            {
                "F-Statistic": np.nan,
                "P-Value": np.nan,
                "Granger-Causes (5%)": False,
            }
        )

    return pd.Series(
        {
            "F-Statistic": f_statistic,
            "P-Value": p_value,
            "Granger-Causes (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )
