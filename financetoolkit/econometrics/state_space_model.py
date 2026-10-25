"""Time-Varying Beta and Dynamic Conditional Correlation Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, signal
from statsmodels.tsa.statespace.mlemodel import MLEModel

from financetoolkit.risk import garch_model

# pylint: disable=abstract-method,too-many-locals


class _TimeVaryingRegression(MLEModel):
    """
    y_t = α_t + β_t x_t + ε_t, with α and β random walks: the state space form of a
    regression whose coefficients drift, estimated with the Kalman filter.
    """

    def __init__(self, endog: np.ndarray, exog: np.ndarray):
        design = np.column_stack([np.ones_like(exog), exog]).T[np.newaxis]
        super().__init__(endog, k_states=2, initialization="diffuse")
        self["design"] = design
        self["transition"] = np.eye(2)
        self["selection"] = np.eye(2)

    @property
    def param_names(self) -> list[str]:
        return ["observation variance", "alpha variance", "beta variance"]

    @property
    def start_params(self) -> np.ndarray:
        variance = float(np.nanvar(self.endog))
        return np.array([variance, variance * 1e-3, 1e-3])

    def transform_params(self, unconstrained: np.ndarray) -> np.ndarray:
        return unconstrained**2

    def untransform_params(self, constrained: np.ndarray) -> np.ndarray:
        return constrained**0.5

    def update(
        self, params, transformed=True, includes_fixed=False, complex_step=False
    ):
        params = super().update(
            params,
            transformed=transformed,
            includes_fixed=includes_fixed,
            complex_step=complex_step,
        )
        self["obs_cov", 0, 0] = params[0]
        self["state_cov"] = np.diag(params[1:])


def get_time_varying_beta(
    dependent: pd.Series, independent: pd.Series
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Estimate a regression whose alpha and beta drift over time, with the Kalman filter:
    every period the coefficients take a random step, and the size of those steps and of
    the noise around the regression are estimated by maximum likelihood. Unlike a rolling
    regression there is no window to choose. Each alpha and beta is the filtered estimate,
    updated with the observations up to its date only; the three variances that set how
    fast the estimates adapt are estimated on the whole sample, so the path is not fully
    what would have been known at the time.

    The model starts without prior knowledge of the coefficients (an exact diffuse
    initialization), so the first observations only pin them down together: the dates
    before both are identified (normally just the first) are NaN.

    For more information, see the following book:

    - Durbin, J. and Koopman, S.J. (2012). "Time Series Analysis by State Space Methods."
      Oxford University Press, 2nd edition.

    Args:
        dependent (pd.Series): The dependent series, e.g. a stock's returns.
        independent (pd.Series): The independent series, e.g. the market's returns.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: Per date the Alpha and Beta with the standard
        error of the beta; and the estimated variances.
    """
    data = pd.concat([dependent, independent], axis=1).dropna()
    if len(data) < 30:  # noqa: PLR2004
        raise ValueError("At least 30 observations are needed for a time-varying beta.")

    y = data.iloc[:, 0].to_numpy(dtype=float)
    x = data.iloc[:, 1].to_numpy(dtype=float)
    scale_y, scale_x = float(np.std(y)), float(np.std(x))
    if not scale_y > 0 or not scale_x > 0:
        raise ValueError("A time-varying beta needs series that vary over time.")

    # The model is fitted to the standardized series and scaled back afterwards. With daily
    # returns the variances span orders of magnitude (an observation variance near 1e-4, a
    # beta variance near 0), and on that scale the optimizer reports convergence after a few
    # steps with the beta variance still at its starting value, far from the maximum of the
    # likelihood, which makes a constant beta wander. Scaling only changes the units:
    # y / s_y = α_t / s_y + (β_t s_x / s_y) x / s_x + ε_t / s_y.
    model = _TimeVaryingRegression(y / scale_y, x / scale_x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(disp=False, maxiter=200)
    filtered = np.asarray(fit.filtered_state)
    covariance = np.asarray(fit.filtered_state_cov)
    beta_scale = scale_y / scale_x

    result = pd.DataFrame(
        {
            "Alpha": filtered[0] * scale_y,
            "Beta": filtered[1] * beta_scale,
            "Beta Standard Error": np.sqrt(np.maximum(covariance[1, 1], 0))
            * beta_scale,
        },
        index=data.index,
    )
    # During the diffuse period the filtered covariance holds only its finite part, so the
    # unidentified first estimates would come with a misleadingly small standard error. The
    # filtered estimate becomes identified at the last diffuse observation.
    result.iloc[: max(fit.filter_results.nobs_diffuse - 1, 0)] = np.nan

    observation, alpha, beta = np.asarray(fit.params)
    variances = {
        "observation variance": float(observation * scale_y**2),
        "alpha variance": float(alpha * scale_y**2),
        "beta variance": float(beta * beta_scale**2),
    }
    return result, variances


def _dcc_correlations(
    residuals: np.ndarray, a: float, b: float, unconditional: np.ndarray
) -> np.ndarray:
    """
    The conditional correlation matrices of the DCC(1,1) model: Q_t is a linear recursion,
    Q_t = (1 - a - b) Q̄ + a z_{t-1} z_{t-1}' + b Q_{t-1}, computed for every element at once,
    and R_t is Q_t scaled to a unit diagonal.
    """
    t, n = residuals.shape
    outer = residuals[:, :, None] * residuals[:, None, :]
    driver = (1 - a - b) * unconditional + a * np.concatenate(
        [unconditional[None], outer[:-1]]
    )
    q = signal.lfilter([1.0], [1.0, -b], driver.reshape(t, -1), axis=0)
    # The recursion starts from Q̄: the first term carries b Q̄ as well.
    q = q.reshape(t, n, n) + (b ** np.arange(1, t + 1))[:, None, None] * unconditional
    scale = 1 / np.sqrt(np.einsum("tii->ti", q))
    return q * scale[:, :, None] * scale[:, None, :]


def get_dcc_garch(returns: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Estimate dynamic conditional correlations with the DCC-GARCH(1,1) model (Engle, 2002):
    every series gets its own GARCH(1,1) volatility, and the correlation of the
    standardized residuals follows Q_t = (1 - a - b) Q̄ + a z_{t-1} z_{t-1}' + b Q_{t-1},
    with Q̄ their unconditional correlation. Correlations between assets tend to rise in
    crises, exactly when diversification is needed most; a rolling correlation sees that
    late and depends on its window, the DCC model updates it every period.

    The model is estimated in two steps: the GARCH parameters per series, then a and b by
    maximizing the correlation part of the likelihood.

    For more information, see the following paper:

    - Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class of Multivariate
      Generalized Autoregressive Conditional Heteroskedasticity Models." Journal of
      Business & Economic Statistics, 20(3), 339-350.

    Args:
        returns (pd.DataFrame): The returns, a column per asset (at least two).

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: The conditional correlation of every pair
        per date, and the DCC parameters a and b with each series' GARCH persistence.
    """
    data = returns.dropna()
    if data.shape[1] < 2:  # noqa: PLR2004
        raise ValueError("The DCC model needs at least two series.")
    if len(data) < 100:  # noqa: PLR2004
        raise ValueError("At least 100 observations are needed for the DCC model.")

    demeaned = (data - data.mean()).to_numpy(dtype=float)
    parameters: dict[str, float] = {}
    residuals = np.empty_like(demeaned)
    for column, name in enumerate(data.columns):
        weights = garch_model.get_garch_weights(demeaned[:, column])
        if np.isnan(weights).any():
            raise ValueError(f"The GARCH model of {name} did not converge.")
        variance = garch_model.get_garch(demeaned[:, column], weights)
        residuals[:, column] = demeaned[:, column] / np.sqrt(variance)
        parameters[f"{name} GARCH Persistence"] = float(weights[1] + weights[2])

    unconditional = np.corrcoef(residuals, rowvar=False)

    def negative_likelihood(values: np.ndarray) -> float:
        a, b = values
        if a < 0 or b < 0 or a + b >= 1:
            return np.inf
        correlations = _dcc_correlations(residuals, a, b, unconditional)
        sign, log_determinant = np.linalg.slogdet(correlations)
        if (sign <= 0).any():
            return np.inf
        solved = np.linalg.solve(correlations, residuals[:, :, None])[:, :, 0]
        quadratic = np.einsum("ti,ti->t", residuals, solved)
        return 0.5 * float(np.sum(log_determinant + quadratic))

    best = min(
        (
            optimize.minimize(
                negative_likelihood,
                start,
                method="Nelder-Mead",
                options={"xatol": 1e-6},
            )
            for start in ((0.02, 0.95), (0.05, 0.9), (0.01, 0.98))
        ),
        key=lambda result: result.fun,
    )
    a, b = (float(value) for value in best.x)
    parameters = {"DCC Alpha": a, "DCC Beta": b, **parameters}

    correlations = _dcc_correlations(residuals, a, b, unconditional)
    pairs = {
        f"{data.columns[i]} / {data.columns[j]}": correlations[:, i, j]
        for i in range(data.shape[1])
        for j in range(i + 1, data.shape[1])
    }
    return pd.DataFrame(pairs, index=data.index), parameters
