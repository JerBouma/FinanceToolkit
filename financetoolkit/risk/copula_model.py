"""Copula (Tail Dependence) Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

MINIMUM_TAIL_OBSERVATIONS = 2
MEDIAN_QUANTILE = 0.5


def get_tail_dependence_coefficient(
    series_a: pd.Series,
    series_b: pd.Series,
    q: float = 0.95,
    method: str = "empirical",
    dof: float = 4.0,
) -> pd.Series:
    """
    Calculate the Upper and Lower Tail Dependence Coefficients between two series.

    Correlation (see e.g. `risk_controller.Risk.get_value_at_risk`'s use of the
    gaussian distribution) only captures the *average* co-movement between two
    series -- it says nothing about whether they are more likely to crash together
    than an equivalent gaussian relationship would imply. The Tail Dependence
    Coefficient answers that specific question: the probability that one series is
    in extreme distress, given that the other one already is.

    - Upper Tail Dependence: lambda_U = lim_(u -> 1) P(F_B(B) > u | F_A(A) > u)
    - Lower Tail Dependence: lambda_L = lim_(u -> 0) P(F_B(B) <= u | F_A(A) <= u)

    Where `F_A` and `F_B` are the marginal (cumulative) distributions of `series_a`
    and `series_b`. `lambda_L` is generally the more relevant figure for risk
    management purposes, since it measures joint crash risk (both assets suffering
    extreme losses together) rather than joint boom risk.

    Three ways of estimating the coefficient are supported via `method`:

    - "empirical" (default): a nonparametric plug-in estimate, replacing the limit at
    `u -> 0` / `u -> 1` with a finite threshold quantile `q`:
    `lambda_U(q) = P(B > quantile(B, q) | A > quantile(A, q))`, and symmetrically
    for `lambda_L(1 - q)`. This makes no distributional assumption but is
    sensitive to the choice of `q` and needs enough observations in the tail to be
    reliable.
    - "gaussian": the tail dependence implied by a bivariate gaussian copula with the
    same linear (Pearson) correlation as `series_a` and `series_b`. This is
    included mainly as a cautionary baseline: a gaussian copula has *zero*
    asymptotic tail dependence for any correlation below 1 (Embrechts, McNeil &
    Straumann, 1999), so `lambda_U = lambda_L = 0` is returned regardless of the
    observed correlation -- demonstrating why gaussian-based dependence models
    (e.g. the gaussian VaR in `var_model.get_var_gaussian`) can understate joint
    crash risk relative to the empirical estimate above.
    - "student-t": the tail dependence implied by a bivariate Student-T copula with
    `dof` degrees of freedom and the same linear correlation `rho` as `series_a`
    and `series_b`, which (unlike the gaussian copula) has nonzero, symmetric tail
    dependence given by the closed-form expression:
    `lambda = 2 * t_sf(SQRT((dof + 1) * (1 - rho) / (1 + rho)), dof + 1)`
    where `t_sf` is the Student-T survival function.

    For more information about the method, see the following papers:

    - Embrechts, P., McNeil, A., & Straumann, D. (1999). "Correlation: Pitfalls and
    Alternatives." RISK Magazine, 12, 69-71.
    - Poon, S.H., Rockinger, M., & Tawn, J. (2004). "Extreme Value Dependence in
    Financial Markets: Diagnostics, Models, and Financial Implications." Review of
    Financial Studies, 17(2), 581-610.

    Also known as: tail dependence, extremal dependence coefficient.

    Args:
        series_a (pd.Series): The first series (e.g. returns of one asset).
        series_b (pd.Series): The second series (e.g. returns of another asset).
        q (float, optional): The threshold quantile used for the "empirical" method,
        in (0.5, 1). Defaults to 0.95 (the top/bottom 5% of each series).
        method (str, optional): The estimation method, one of "empirical", "gaussian"
        or "student-t". Defaults to "empirical".
        dof (float, optional): The degrees of freedom of the Student-T copula, only
        used when `method="student-t"`. Lower values imply heavier tails and
        therefore higher tail dependence for the same correlation. Defaults to 4.0.

    Returns:
        pd.Series: The Lower and Upper Tail Dependence Coefficients, the linear
        (Pearson) correlation used by the "gaussian" and "student-t" methods, and the
        number of (paired, non-missing) observations used.

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If `method` is not one of "empirical", "gaussian" or "student-t",
        or if `q` is not in (0.5, 1).
    """
    if not isinstance(series_a, pd.Series) or not isinstance(series_b, pd.Series):
        raise TypeError("Expects pd.Series, no other value.")
    if method not in ("empirical", "gaussian", "student-t"):
        raise ValueError("method must be 'empirical', 'gaussian' or 'student-t'.")
    if q <= MEDIAN_QUANTILE or q >= 1:
        raise ValueError("q must be in the range (0.5, 1).")

    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    n = len(aligned)

    if n <= MINIMUM_TAIL_OBSERVATIONS:
        return pd.Series(
            {
                "Lower Tail Dependence": np.nan,
                "Upper Tail Dependence": np.nan,
                "Correlation": np.nan,
                "Observations": n,
            }
        )

    a = aligned.iloc[:, 0].to_numpy()
    b = aligned.iloc[:, 1].to_numpy()
    correlation = np.corrcoef(a, b)[0, 1]

    if method == "empirical":
        upper_a = a > np.quantile(a, q)
        upper_b = b > np.quantile(b, q)
        lower_a = a <= np.quantile(a, 1 - q)
        lower_b = b <= np.quantile(b, 1 - q)

        upper_tail_dependence = (
            np.sum(upper_a & upper_b) / np.sum(upper_a)
            if np.sum(upper_a) > 0
            else np.nan
        )
        lower_tail_dependence = (
            np.sum(lower_a & lower_b) / np.sum(lower_a)
            if np.sum(lower_a) > 0
            else np.nan
        )
    elif method == "gaussian":
        # A gaussian copula has zero asymptotic tail dependence for any correlation
        # strictly below 1 -- see the Embrechts, McNeil & Straumann (1999) reference
        # in the docstring above.
        upper_tail_dependence = 0.0 if correlation < 1 else 1.0
        lower_tail_dependence = upper_tail_dependence
    else:
        tail_dependence = 2 * stats.t.sf(
            np.sqrt((dof + 1) * (1 - correlation) / (1 + correlation)), dof + 1
        )
        upper_tail_dependence = tail_dependence
        lower_tail_dependence = tail_dependence

    return pd.Series(
        {
            "Lower Tail Dependence": lower_tail_dependence,
            "Upper Tail Dependence": upper_tail_dependence,
            "Correlation": correlation,
            "Observations": n,
        }
    )
