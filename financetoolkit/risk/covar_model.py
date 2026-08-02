"""CoVaR (Conditional Value at Risk Spillover) Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import optimize, sparse

MINIMUM_COVAR_OBSERVATIONS = 10
MEDIAN_QUANTILE = 0.5


def _quantile_regression(
    y: np.ndarray, x: np.ndarray, tau: float
) -> tuple[float, float]:
    """
    Fit a simple (one-predictor) linear Quantile Regression of `y` on `x` at quantile
    `tau`, via the standard Koenker & Bassett (1978) Linear Program formulation:

    - minimize: tau * SUM(u_plus) + (1 - tau) * SUM(u_minus)
    - subject to: intercept + slope * x_t + u_plus_t - u_minus_t = y_t,
    u_plus_t >= 0, u_minus_t >= 0

    Unlike Ordinary Least Squares (which minimizes squared residuals and therefore
    estimates the conditional mean), this minimizes an asymmetric ("pinball") loss
    that is minimized at the conditional `tau`-quantile of `y` given `x`. Shared by
    `covar_model.get_covar` (the CoVaR quantile regression is fit at `tau = alpha`,
    the tail quantile, and again at `tau = 0.5`, the median, to isolate the tail-
    specific spillover from the "normal state" relationship).

    Args:
        y (np.ndarray): The dependent variable.
        x (np.ndarray): The (single) predictor.
        tau (float): The quantile to fit, in (0, 1).

    Returns:
        tuple[float, float]: The fitted intercept and slope.
    """
    n = len(y)
    number_of_variables = 2 + 2 * n

    # Objective: no cost on the intercept/slope themselves, asymmetric cost on the
    # positive (u_plus) and negative (u_minus) residual parts.
    c = np.concatenate([[0.0, 0.0], np.full(n, tau), np.full(n, 1 - tau)])

    rows = np.repeat(np.arange(n), 4)
    cols = np.column_stack(
        [
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
            2 + np.arange(n),
            2 + n + np.arange(n),
        ]
    ).ravel()
    data = np.column_stack([np.ones(n), x, np.ones(n), -np.ones(n)]).ravel()

    a_eq = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n, number_of_variables)
    ).tocsr()

    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)

    result = optimize.linprog(c, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")

    if not result.success:
        return np.nan, np.nan

    return float(result.x[0]), float(result.x[1])


def get_covar(
    returns: pd.Series,
    conditioning_returns: pd.Series,
    alpha: float,
) -> pd.Series:
    """
    Calculate the (Delta-)CoVaR of `returns` conditional on `conditioning_returns`
    being in its own distress state.

    Ordinary Value at Risk (see `get_var_historic`) treats each asset in isolation,
    which misses systemic risk -- the fact that one institution's or asset's distress
    can spill over and worsen another's risk. CoVaR (short for "Conditional VaR",
    unrelated to `cvar_model`'s Conditional Value at Risk / Expected Shortfall)
    directly measures that spillover: it is the VaR of `returns`, conditional on
    `conditioning_returns` itself being at its own `alpha`-VaR:

    - P(returns <= CoVaR | conditioning_returns = VaR_alpha(conditioning_returns)) = alpha

    This is estimated by fitting a linear Quantile Regression of `returns` on
    `conditioning_returns` at quantile `alpha`, then evaluating the fitted line at
    `conditioning_returns`'s own historical VaR:

    - returns = a_alpha + b_alpha * conditioning_returns + e   (fit at quantile alpha)
    - CoVaR = a_alpha + b_alpha * VaR_alpha(conditioning_returns)

    The Delta-CoVaR isolates the marginal, distress-specific contribution by holding
    the *same* (single) alpha-quantile regression fixed and instead comparing what it
    predicts when `conditioning_returns` is at its own distress VaR versus at its
    median -- i.e. the same `a_alpha`/`b_alpha` used above, evaluated at two different
    values of `conditioning_returns` (this is the construction in Adrian & Brunnermeier
    (2016), Section II.B: they do not re-fit a separate median-quantile regression):

    - Delta-CoVaR = CoVaR - (a_alpha + b_alpha * median(conditioning_returns))
    - = b_alpha * (VaR_alpha(conditioning_returns) - median(conditioning_returns))

    A large (negative) Delta-CoVaR means `conditioning_returns` being in distress
    meaningfully worsens the tail risk of `returns` beyond their typical (median-state)
    relationship -- i.e. there is systemic spillover, not just an unconditionally
    higher/lower level of risk.

    For more information about the method, see the following paper:

    - Adrian, T., & Brunnermeier, M.K. (2016). "CoVaR." American Economic Review,
    106(7), 1705-1741.

    Also known as: Conditional Value at Risk (systemic risk sense), Delta-CoVaR.

    Args:
        returns (pd.Series): The returns of the asset/institution whose conditional
        VaR is being measured.
        conditioning_returns (pd.Series): The returns of the asset/institution (or
        e.g. a financial-system index) whose distress `returns` is conditioned on.
        alpha (float): The confidence level for both the tail quantile regression and
        the VaR of `conditioning_returns` (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series: The CoVaR, the Delta-CoVaR, the tail (alpha-quantile) Quantile
        Regression slope and intercept, and the number of (paired, non-missing)
        observations used.

    Raises:
        TypeError: If `returns` or `conditioning_returns` is not a pd.Series.
    """
    if not isinstance(returns, pd.Series) or not isinstance(
        conditioning_returns, pd.Series
    ):
        raise TypeError("Expects pd.Series, no other value.")

    aligned = pd.concat([returns, conditioning_returns], axis=1, join="inner").dropna()
    n = len(aligned)

    if n <= MINIMUM_COVAR_OBSERVATIONS:
        return pd.Series(
            {
                "CoVaR": np.nan,
                "Delta-CoVaR": np.nan,
                "Quantile Regression Slope": np.nan,
                "Quantile Regression Intercept": np.nan,
                "Observations": n,
            }
        )

    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1].to_numpy()

    intercept_alpha, slope_alpha = _quantile_regression(y, x, alpha)

    var_conditioning_alpha = np.percentile(x, alpha * 100)
    median_conditioning = np.percentile(x, MEDIAN_QUANTILE * 100)

    covar = intercept_alpha + slope_alpha * var_conditioning_alpha
    # Same (single) alpha-quantile regression, evaluated at the conditioning
    # variable's median instead of its distress VaR -- not a separately refit
    # median-quantile regression. See the docstring above.
    delta_covar = slope_alpha * (var_conditioning_alpha - median_conditioning)

    return pd.Series(
        {
            "CoVaR": covar,
            "Delta-CoVaR": delta_covar,
            "Quantile Regression Slope": slope_alpha,
            "Quantile Regression Intercept": intercept_alpha,
            "Observations": n,
        }
    )
