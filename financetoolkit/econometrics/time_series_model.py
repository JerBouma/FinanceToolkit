"""Time Series Model"""

__docformat__ = "google"

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM

from financetoolkit.econometrics import cointegration_model, regression_model
from financetoolkit.econometrics.regression_model import RegressionResult

# A minimal safety margin (beyond the number of parameters actually being estimated)
# required for the ARIMA/VAR estimators below to be well-identified -- not a hard
# statistical requirement, just a guard against a degenerate (n <= k) fit that would
# otherwise fail deep inside `np.linalg`/`statsmodels` with an unhelpful error. VECM
# delegates this kind of validation to `statsmodels` itself (see its `except` block).
MINIMUM_ARIMA_OBSERVATIONS_BUFFER = 5
MINIMUM_VAR_OBSERVATIONS_BUFFER = 5
MINIMUM_VAR_SERIES = 2

# Significance levels tabulated by `cointegration_model.get_johansen_cointegration`
# and therefore the only ones `get_vecm_forecast` can determine a cointegrating rank
# at without inventing untabulated critical values.
JOHANSEN_TRACE_CRITICAL_VALUE_COLUMNS: dict[float, str] = {
    0.10: "Trace Critical Value 90%",
    0.05: "Trace Critical Value 95%",
    0.01: "Trace Critical Value 99%",
}


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


@dataclass
class ARIMAResult:
    """
    The fitted output of `get_arima_forecast`.

    Attributes:
        order (tuple[int, int, int]): The `(p, d, q)` order that was fit.
        constant (float): The fitted intercept/trend term, as `statsmodels` reports it
        -- for `d=0` this is the unconditional MEAN of `series` (not the recursion
        intercept `c`; recover `c` via `constant * (1 - sum(ar_coefficients))` if
        needed), for `d>=1` it is the drift coefficient of the differenced series
        (already equal to `c` in the differenced-scale recursion). 0.0 if
        `include_constant=False`.
        constant_std_error (float): The standard error of `constant`, `nan` if
        `include_constant=False`.
        ar_coefficients (np.ndarray): The fitted AR coefficients `phi_1, ..., phi_p`,
        shape `(p,)`.
        ar_std_errors (np.ndarray): The standard errors of `ar_coefficients`, shape `(p,)`.
        ma_coefficients (np.ndarray): The fitted MA coefficients `theta_1, ..., theta_q`,
        shape `(q,)`.
        ma_std_errors (np.ndarray): The standard errors of `ma_coefficients`, shape `(q,)`.
        residuals (pd.Series): The in-sample one-step-ahead prediction errors, on the
        ORIGINAL (undifferenced) scale of `series`, indexed like `series`.
        fitted_values (pd.Series): The in-sample fitted values, on the same original
        scale and index as `residuals`.
        forecast (pd.Series): The `forecast_steps`-ahead forecast, on the ORIGINAL
        scale of `series`. Indexed `1, ..., forecast_steps` (the number of periods
        ahead), not a continuation of `series`'s own index.
        sum_of_squared_residuals (float): The sum of squared one-step-ahead prediction
        errors.
        n_observations (int): The number of observations used in estimation.
        converged (bool): Whether the underlying Maximum Likelihood optimizer reported
        convergence.
    """

    order: tuple[int, int, int]
    constant: float
    constant_std_error: float
    ar_coefficients: np.ndarray
    ar_std_errors: np.ndarray
    ma_coefficients: np.ndarray
    ma_std_errors: np.ndarray
    residuals: pd.Series
    fitted_values: pd.Series
    forecast: pd.Series
    sum_of_squared_residuals: float
    n_observations: int
    converged: bool

    def summary(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: A coefficient table (Coefficient, Std. Error) of the
            constant, AR and MA coefficients.
        """
        p, _, q = self.order
        names = (
            ["Intercept"]
            + [f"AR{i}" for i in range(1, p + 1)]
            + [f"MA{j}" for j in range(1, q + 1)]
        )
        coefficients = [self.constant, *self.ar_coefficients, *self.ma_coefficients]
        standard_errors = [
            self.constant_std_error,
            *self.ar_std_errors,
            *self.ma_std_errors,
        ]
        return pd.DataFrame(
            {"Coefficient": coefficients, "Std. Error": standard_errors}, index=names
        )


def get_arima_forecast(
    series: pd.Series,
    p: int,
    d: int,
    q: int,
    forecast_steps: int = 1,
    include_constant: bool = True,
    max_iterations: int = 1000,
) -> ARIMAResult:
    """
    Fit an ARIMA(p, d, q) model and produce `forecast_steps`-ahead forecasts, via
    `statsmodels.tsa.arima.model.ARIMA`.

    Also known as: Box-Jenkins model, autoregressive integrated moving average.

    An ARIMA(p, d, q) model combines three ideas: `d` rounds of differencing to remove
    a (stochastic) trend and make the series stationary, an autoregressive (AR)
    component that regresses the (differenced) series on its own past `p` values, and
    a moving-average (MA) component that regresses it on the past `q` forecast errors:

    - Let `y'_t` be `series` differenced `d` times. Then:
    - y'_t = c + SUM_{i=1}^{p}(phi_i * y'_(t-i)) + SUM_{j=1}^{q}(theta_j * e_(t-j)) + e_t

    Estimated by exact Maximum Likelihood via the Kalman filter (a state-space
    representation of the ARMA recursion), which correctly handles the MA component's
    unobserved initial innovations rather than approximating them.

    For more information about the method, see the following book:

    - Box, G.E.P., & Jenkins, G.M. (1970). "Time Series Analysis: Forecasting and
    Control." Holden-Day.

    Args:
        series (pd.Series): The series to fit, in levels (before differencing).
        p (int): The autoregressive order.
        d (int): The number of times to difference the series.
        q (int): The moving-average order.
        forecast_steps (int, optional): The number of periods ahead to forecast.
        Defaults to 1.
        include_constant (bool, optional): Whether to estimate a free intercept `c`
        (defaults to True). For `d >= 1` and no `include_constant`, `c` is fixed at 0
        rather than estimated -- appropriate when the differenced series is not
        expected to have a nonzero mean/drift (the common case for `d >= 1`; leaving
        `include_constant=True` on a truly zero-drift differenced series lets small-
        sample noise in the estimated `c` compound into a spuriously trending
        multi-step forecast, since each step's drift accumulates).
        max_iterations (int, optional): The maximum number of Maximum Likelihood
        optimizer iterations. Defaults to 1000.

    Returns:
        ARIMAResult: The fitted constant/AR/MA coefficients and their standard
        errors, in-sample residuals and fitted values, and the `forecast_steps`-ahead
        forecast. Call `.summary()` for a coefficient table.

    Raises:
        TypeError: If `series` is not a `pd.Series`.
        ValueError: If `p`, `d` or `q` is negative, `p` and `q` are both 0,
        `forecast_steps` is not a positive integer, or there are not enough
        observations to fit the requested `(p, d, q)` order.

    Notes:
    - **ARMA parameter recovery is a genuine finite-sample identifiability issue, not
    an estimator quirk:** for `p + q >= 4` (e.g. ARMA(2, 2)), fitted coefficients can
    land noticeably far from the TRUE generating parameters even on several hundred
    observations, since near-canceling AR/MA roots make the likelihood surface close
    to flat in some directions. Keep orders small (e.g. ARIMA(1,1,1), ARIMA(2,0,1))
    for reliable parameter recovery.

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import time_series_model

    rng = np.random.default_rng(1)
    e = rng.standard_normal(300)
    y = np.zeros(300)
    for t in range(1, 300):
        y[t] = 0.6 * y[t - 1] + e[t]

    result = time_series_model.get_arima_forecast(pd.Series(y), p=1, d=0, q=0, forecast_steps=5)
    print(result.summary().round(4))
    print(result.forecast.round(4))
    ```
    """
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"series must be a pd.Series, received {type(series).__name__}."
        )
    if p < 0 or d < 0 or q < 0:
        raise ValueError("p, d and q must all be non-negative integers.")
    if p == 0 and q == 0:
        raise ValueError(
            "p and q cannot both be 0 -- an ARIMA(0, d, 0) has no dynamics to "
            "estimate; difference the series d times directly instead."
        )
    if forecast_steps < 1:
        raise ValueError("forecast_steps must be a positive integer.")

    clean_series = series.dropna()
    minimum_observations = d + p + q + MINIMUM_ARIMA_OBSERVATIONS_BUFFER
    if len(clean_series) <= minimum_observations:
        raise ValueError(
            f"Not enough observations ({len(clean_series)}) to fit an ARIMA({p}, "
            f"{d}, {q}) model -- need more than {minimum_observations}."
        )

    if not include_constant:
        trend = "n"
    elif d == 0:
        trend = "c"
    else:
        # A plain constant is eliminated by d rounds of differencing -- a trend
        # polynomial of order d (only its highest-order term) is `statsmodels`'
        # equivalent of "a nonzero mean/drift on the differenced series", which is
        # what `include_constant=True` means for d >= 1.
        trend = [0] * d + [1]

    try:
        sm_result = ARIMA(clean_series, order=(p, d, q), trend=trend).fit(
            method_kwargs={"maxiter": max_iterations}
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(
            f"Could not fit an ARIMA({p}, {d}, {q}) model on {len(clean_series)} "
            "observations."
        ) from error

    is_trend_param = ~sm_result.params.index.str.startswith(("ar.", "ma.")) & (
        sm_result.params.index != "sigma2"
    )
    if is_trend_param.any():
        constant = float(sm_result.params[is_trend_param].iloc[0])
        constant_std_error = float(sm_result.bse[is_trend_param].iloc[0])
    else:
        constant, constant_std_error = 0.0, np.nan
    ar_coefficients = np.asarray(sm_result.arparams)
    ar_std_errors = sm_result.bse.filter(regex=r"^ar\.").to_numpy()
    ma_coefficients = np.asarray(sm_result.maparams)
    ma_std_errors = sm_result.bse.filter(regex=r"^ma\.").to_numpy()

    forecast_index = pd.RangeIndex(1, forecast_steps + 1, name="Step")
    forecast_series = pd.Series(
        sm_result.get_forecast(steps=forecast_steps).predicted_mean.to_numpy(),
        index=forecast_index,
        name="Forecast",
    )

    return ARIMAResult(
        order=(p, d, q),
        constant=constant,
        constant_std_error=constant_std_error,
        ar_coefficients=ar_coefficients,
        ar_std_errors=ar_std_errors,
        ma_coefficients=ma_coefficients,
        ma_std_errors=ma_std_errors,
        residuals=sm_result.resid.rename("Residuals"),
        fitted_values=sm_result.fittedvalues.rename("Fitted Values"),
        forecast=forecast_series,
        sum_of_squared_residuals=float(sm_result.sse),
        n_observations=int(sm_result.nobs),
        converged=bool(sm_result.mle_retvals.get("converged", True)),
    )


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------


def _build_lagged_design(
    data: pd.DataFrame, lags: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the standard multivariate-lag design matrix used by `get_var_forecast`
    (lags of the levels): for each lag `l = 1, ..., lags`, appends every column of
    `data` shifted by `l` periods, then drops the leading `lags` rows (which contain
    NaNs from the shift). Column order is lag-major: all of lag 1's columns (in
    `data`'s own column order), then all of lag 2's, etc. -- callers rely on this
    exact order to slice per-lag coefficient blocks back out of a fitted
    `RegressionResult`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: `(design, target)`, sharing an index --
        `target` is simply `data.iloc[lags:]`.
    """
    if lags < 1:
        raise ValueError("lags must be a positive integer.")

    blocks = []
    for lag in range(1, lags + 1):
        shifted = data.shift(lag)
        shifted.columns = [f"{column}_lag{lag}" for column in data.columns]
        blocks.append(shifted)

    design = pd.concat(blocks, axis=1).iloc[lags:]
    target = data.iloc[lags:]
    return design, target


def _extract_lag_matrices(
    equations: dict[str, RegressionResult],
    variable_names: list[str],
    n_lags: int,
    coefficient_offset: int,
) -> dict[int, pd.DataFrame]:
    """
    Slices the shared lag-block coefficients (see `_build_lagged_design`'s column
    order) out of each variable's fitted `RegressionResult`, used by `get_var_forecast`
    (`coefficient_offset=1`, right after the intercept) to build the
    `{lag: (k, k) DataFrame}` matrices `Phi_l`, indexed and columned by variable name
    (row = equation i.e. dependent variable, column = predictor variable).
    """
    k = len(variable_names)
    matrices: dict[int, pd.DataFrame] = {}
    for lag in range(1, n_lags + 1):
        start = coefficient_offset + (lag - 1) * k
        end = start + k
        matrix = pd.DataFrame(
            {
                target_column: pd.Series(
                    equation.coefficients[start:end], index=variable_names
                )
                for target_column, equation in equations.items()
            }
        ).T
        matrix.index.name = "Equation"
        matrices[lag] = matrix
    return matrices


@dataclass
class VARResult:
    """
    The fitted output of `get_var_forecast`.

    Attributes:
        lags (int): The VAR order that was fit.
        intercept (pd.Series): The fitted constant of each equation, indexed by
        variable name.
        coefficient_matrices (dict[int, pd.DataFrame]): `{lag: (k, k) DataFrame}` --
        `coefficient_matrices[l].loc[i, j]` is the coefficient of variable `j`'s
        `l`-th lag in variable `i`'s equation.
        equations (dict[str, RegressionResult]): The full per-variable OLS fit (see
        `regression_model.get_ols`), keyed by variable name -- use this for
        standard errors, t-statistics, R-squared, etc. per equation.
        fitted_values (pd.DataFrame): The in-sample fitted values, one column per
        variable.
        residuals (pd.DataFrame): The in-sample residuals, one column per variable.
        forecast (pd.DataFrame): The `forecast_steps`-ahead forecast, one column per
        variable. Indexed `1, ..., forecast_steps` (the number of periods ahead), not
        a continuation of `data`'s own index.
        n_observations (int): The number of observations used in estimation
        (`len(data) - lags`).
        variable_names (list[str]): The fitted variables, in column order.
    """

    lags: int
    intercept: pd.Series
    coefficient_matrices: dict[int, pd.DataFrame]
    equations: dict[str, RegressionResult]
    fitted_values: pd.DataFrame
    residuals: pd.DataFrame
    forecast: pd.DataFrame
    n_observations: int
    variable_names: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: Each equation's coefficient table (see
            `RegressionResult.summary()`), concatenated side by side under a top-level
            column per variable.
        """
        return pd.concat(
            {name: equation.summary() for name, equation in self.equations.items()},
            axis=1,
        )


def get_var_forecast(
    data: pd.DataFrame, lags: int, forecast_steps: int = 1
) -> VARResult:
    """
    Fit a Vector Autoregression (VAR) of order `lags` and produce `forecast_steps`-
    ahead forecasts.

    Also known as: VAR model, vector autoregressive model.

    A VAR generalizes a univariate AR model to `k` series at once: every series is
    regressed on `lags` lagged values of ALL `k` series (including itself), letting
    each variable's future depend on its own past AND the others' past -- the natural
    tool for jointly modeling a small system of related series (e.g. several assets'
    returns) without imposing which one "drives" the others:

    - Y_t = c + SUM_{l=1}^{lags}(Phi_l * Y_(t-l)) + e_t

    Where `Y_t` is the `k`-vector of all series at time `t`. Because every equation
    has an identical set of regressors (the same lagged values of all `k` series), the
    Maximum Likelihood estimator of a VAR reduces to `k` separate Ordinary Least
    Squares regressions, one per equation -- no iterative fitting is needed, and no
    separate `statsmodels.tsa.vector_ar.var_model.VAR` call is needed either: each
    equation here is fit via `regression_model.get_ols`, i.e. `statsmodels.api.OLS`
    directly, which is mathematically identical to (and was cross-checked against)
    `statsmodels.tsa.vector_ar.var_model.VAR`'s own OLS-based estimation.

    Forecasts are produced by iterating the fitted recursion forward `forecast_steps`
    times, feeding each step's forecast back in as the next step's most recent lag.

    For more information about the method, see the following paper:

    - Sims, C.A. (1980). "Macroeconomic and Reality." Econometrica, 48(1), 1-48.

    Args:
        data (pd.DataFrame): One column per series (e.g. several assets' returns),
        at least 2 columns.
        lags (int): The VAR order (number of lagged periods of every series included
        in every equation).
        forecast_steps (int, optional): The number of periods ahead to forecast.
        Defaults to 1.

    Returns:
        VARResult: The fitted intercept and per-lag coefficient matrices, the
        underlying per-equation `RegressionResult`s, in-sample fitted values/residuals,
        and the `forecast_steps`-ahead forecast. Call `.summary()` for a combined
        coefficient table.

    Raises:
        TypeError: If `data` is not a `pd.DataFrame`.
        ValueError: If `data` has fewer than 2 columns, `lags` or `forecast_steps` is
        not a positive integer, or there are not enough observations to estimate the
        model.

    Notes:
    - Verified by simulating a synthetic bivariate VAR(1) with a known coefficient
    matrix and confirming the fitted coefficients recover it closely, and by
    cross-checking against `statsmodels.tsa.api.VAR` -- coefficients matched
    statsmodels to several decimal places, as expected since VAR estimation is
    equation-by-equation OLS in both implementations (no estimation-method
    ambiguity, unlike ARIMA).

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import time_series_model

    rng = np.random.default_rng(1)
    n = 300
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(1, n):
        y1[t] = 0.5 * y1[t - 1] + 0.2 * y2[t - 1] + rng.standard_normal() * 0.1
        y2[t] = 0.1 * y1[t - 1] + 0.6 * y2[t - 1] + rng.standard_normal() * 0.1

    data = pd.DataFrame({"Y1": y1, "Y2": y2})
    result = time_series_model.get_var_forecast(data, lags=1, forecast_steps=5)
    print(result.coefficient_matrices[1].round(4))
    print(result.forecast.round(4))
    ```
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pd.DataFrame, received {type(data).__name__}.")
    if lags < 1:
        raise ValueError("lags must be a positive integer.")
    if forecast_steps < 1:
        raise ValueError("forecast_steps must be a positive integer.")

    aligned = data.dropna()
    variable_names = list(aligned.columns)
    k = len(variable_names)

    if k < MINIMUM_VAR_SERIES:
        raise ValueError("data must have at least 2 columns (series) for a VAR model.")

    design, target = _build_lagged_design(aligned, lags)

    minimum_observations = k * lags + 1 + MINIMUM_VAR_OBSERVATIONS_BUFFER
    if len(design) <= minimum_observations:
        raise ValueError(
            f"Not enough observations ({len(design)}) to fit a VAR({lags}) model "
            f"across {k} series -- need more than {minimum_observations}."
        )

    equations: dict[str, RegressionResult] = {
        column: regression_model.get_ols(target[column], design, add_constant=True)
        for column in variable_names
    }

    intercept = pd.Series(
        {column: equation.coefficients[0] for column, equation in equations.items()}
    )
    coefficient_matrices = _extract_lag_matrices(
        equations, variable_names, n_lags=lags, coefficient_offset=1
    )

    fitted_values = pd.DataFrame(
        {column: equation.fitted_values for column, equation in equations.items()},
        index=target.index,
    )
    residuals = pd.DataFrame(
        {column: equation.residuals for column, equation in equations.items()},
        index=target.index,
    )

    intercept_vector = intercept[variable_names].to_numpy()
    coefficient_arrays = {
        lag: coefficient_matrices[lag].loc[variable_names, variable_names].to_numpy()
        for lag in range(1, lags + 1)
    }

    history = list(aligned.tail(lags)[variable_names].to_numpy())
    forecasts = []
    for _ in range(forecast_steps):
        y_new = intercept_vector.copy()
        for lag in range(1, lags + 1):
            y_new = y_new + coefficient_arrays[lag] @ history[-lag]
        history.append(y_new)
        forecasts.append(y_new)

    forecast_index = pd.RangeIndex(1, forecast_steps + 1, name="Step")
    forecast = pd.DataFrame(forecasts, index=forecast_index, columns=variable_names)

    return VARResult(
        lags=lags,
        intercept=intercept,
        coefficient_matrices=coefficient_matrices,
        equations=equations,
        fitted_values=fitted_values,
        residuals=residuals,
        forecast=forecast,
        n_observations=len(design),
        variable_names=variable_names,
    )


# ---------------------------------------------------------------------------
# VECM
# ---------------------------------------------------------------------------


@dataclass
class VECMResult:
    """
    The fitted output of `get_vecm_forecast`.

    Attributes:
        rank (int): The cointegrating rank used (from
        `cointegration_model.get_johansen_cointegration`).
        k_ar_diff (int): The number of lagged differences (short-run dynamics) fit.
        cointegrating_vectors (pd.DataFrame): `beta`, shape `(k, rank)` -- one
        cointegrating vector per column, indexed by variable name.
        adjustment_speeds (pd.DataFrame): `alpha`, shape `(k, rank)` -- how strongly
        each variable's equation responds to each cointegrating relation's
        disequilibrium, indexed by variable name (rows) and cointegrating vector
        (columns).
        short_run_coefficients (dict[int, pd.DataFrame]): `{lag: (k, k) DataFrame}` --
        the `Gamma_l` short-run dynamics matrices, same layout as
        `VARResult.coefficient_matrices`.
        intercept (pd.Series): The fitted constant of each equation.
        fitted_values (pd.DataFrame): The in-sample fitted values, on the
        FIRST-DIFFERENCED scale (`Delta_Y_t`), one column per variable.
        residuals (pd.DataFrame): The in-sample residuals, on the same
        first-differenced scale, one column per variable.
        forecast (pd.DataFrame): The `forecast_steps`-ahead forecast, on the ORIGINAL
        (levels) scale, one column per variable. Indexed `1, ..., forecast_steps`.
        n_observations (int): The number of observations used in estimation.
        variable_names (list[str]): The fitted variables, in column order.
    """

    rank: int
    k_ar_diff: int
    cointegrating_vectors: pd.DataFrame
    adjustment_speeds: pd.DataFrame
    short_run_coefficients: dict[int, pd.DataFrame]
    intercept: pd.Series
    fitted_values: pd.DataFrame
    residuals: pd.DataFrame
    forecast: pd.DataFrame
    n_observations: int
    variable_names: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: The cointegrating vector(s) (`beta`) and adjustment speeds
            (`alpha`), side by side, indexed by variable name.
        """
        return pd.concat(
            {
                "Cointegrating Vector": self.cointegrating_vectors,
                "Adjustment Speed": self.adjustment_speeds,
            },
            axis=1,
        )


def get_vecm_forecast(
    data: pd.DataFrame,
    k_ar_diff: int = 1,
    forecast_steps: int = 1,
    significance: float = 0.05,
) -> VECMResult:
    """
    Fit a Vector Error Correction Model (VECM) to cointegrated series and produce
    `forecast_steps`-ahead forecasts, via `statsmodels.tsa.vector_ar.vecm.VECM`.

    Also known as: VECM, error correction model (for the multivariate/cointegrated
    case).

    A plain VAR in levels is misspecified for non-stationary, cointegrated series (see
    `cointegration_model.get_johansen_cointegration`), and a VAR in differences throws
    away the long-run equilibrium relationship entirely. The VECM keeps both: it
    models the change in each series as reacting partly to how far the system
    currently sits from its long-run equilibrium/equilibria (the "error correction"
    term) and partly to recent short-run dynamics:

    - Delta_Y_t = Pi * Y_(t-1) + SUM_{i=1}^{k_ar_diff}(Gamma_i * Delta_Y_(t-i)) + c + e_t
    - Pi = alpha * beta'

    Where `beta` (the cointegrating vector(s)) and the cointegrating rank are taken
    from `cointegration_model.get_johansen_cointegration` -- reused here rather than
    re-derived, since that function already contains this codebase's verified
    rank-testing machinery. Given the rank, `alpha`, `beta` and the `Gamma_i`
    short-run matrices are then jointly estimated by `statsmodels.tsa.vector_ar.vecm.VECM`
    (with `deterministic="co"`, a constant restricted to lie in the cointegrating
    relation, matching the `det_order=0` convention used for rank determination).

    Forecasts are produced on the levels scale via the fitted model's own multi-step
    forecast recursion.

    For more information about the method, see the following paper:

    - Engle, R.F., & Granger, C.W.J. (1987). "Co-integration and Error Correction:
    Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

    Args:
        data (pd.DataFrame): One column per series, in LEVELS (e.g. price levels, not
        returns) -- a VECM only makes sense for non-stationary, cointegrated series,
        the same input `get_johansen_cointegration` expects.
        k_ar_diff (int, optional): The number of lagged first differences to include
        as short-run dynamics. Defaults to 1.
        forecast_steps (int, optional): The number of periods ahead to forecast.
        Defaults to 1.
        significance (float, optional): The significance level (one of 0.01, 0.05,
        0.10) at which the Johansen trace test determines the cointegrating rank.
        Defaults to 0.05.

    Returns:
        VECMResult: The fitted cointegrating vector(s), adjustment speeds, short-run
        coefficient matrices, in-sample fitted values/residuals (differenced scale),
        and the `forecast_steps`-ahead forecast (levels scale).

    Raises:
        TypeError: If `data` is not a `pd.DataFrame`.
        ValueError: If `k_ar_diff` or `forecast_steps` is not a positive integer,
        `significance` is not one of 0.01/0.05/0.10, there are not enough
        observations, or -- most importantly -- the Johansen test does not reject a
        cointegrating rank of 0 at `significance`, in which case a VECM is not an
        appropriate model (use `get_var_forecast` on the differenced data instead).

    As an example:

    ```python
    import numpy as np
    import pandas as pd
    from financetoolkit.econometrics import time_series_model

    rng = np.random.default_rng(3)
    n = 300
    common_trend = np.cumsum(rng.standard_normal(n))
    series_a = common_trend + rng.standard_normal(n) * 0.2
    series_b = common_trend * 1.5 + 3 + rng.standard_normal(n) * 0.2

    data = pd.DataFrame({"A": series_a, "B": series_b})
    result = time_series_model.get_vecm_forecast(data, k_ar_diff=1, forecast_steps=5)
    print(result.cointegrating_vectors.round(4))
    print(result.forecast.round(4))
    ```
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pd.DataFrame, received {type(data).__name__}.")
    if k_ar_diff < 1:
        raise ValueError(
            "k_ar_diff must be a positive integer -- k_ar_diff=0 leaves no short-run "
            "dynamics term to estimate."
        )
    if forecast_steps < 1:
        raise ValueError("forecast_steps must be a positive integer.")
    if significance not in JOHANSEN_TRACE_CRITICAL_VALUE_COLUMNS:
        raise ValueError(
            "significance must be one of 0.01, 0.05 or 0.10 -- the only levels "
            "tabulated by cointegration_model.get_johansen_cointegration."
        )

    aligned = data.dropna()
    variable_names = list(aligned.columns)
    k = len(variable_names)

    johansen_result = cointegration_model.get_johansen_cointegration(
        aligned, det_order=0, k_ar_diff=k_ar_diff
    )

    if johansen_result["Trace Statistic"].isna().all():
        raise ValueError(
            f"Not enough observations to fit a Johansen/VECM model with "
            f"k_ar_diff={k_ar_diff} across {k} series."
        )

    critical_value_column = JOHANSEN_TRACE_CRITICAL_VALUE_COLUMNS[significance]
    rank = 0
    for label in johansen_result.index:
        if (
            johansen_result.loc[label, "Trace Statistic"]
            > johansen_result.loc[label, critical_value_column]
        ):
            rank += 1
        else:
            break

    if rank == 0:
        raise ValueError(
            "The Johansen test does not reject a cointegrating rank of 0 (no "
            f"cointegration) at the {significance:.0%} level -- a VECM is not "
            "appropriate for these series; use get_var_forecast on the differenced "
            "data instead."
        )

    try:
        sm_result = VECM(
            aligned, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="co"
        ).fit()
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(
            f"Could not fit a VECM with rank {rank} and k_ar_diff={k_ar_diff} across "
            f"{k} series."
        ) from error

    cointegrating_vector_names = [f"EC{i + 1}" for i in range(rank)]

    cointegrating_vectors = pd.DataFrame(
        sm_result.beta, index=variable_names, columns=cointegrating_vector_names
    )
    adjustment_speeds = pd.DataFrame(
        sm_result.alpha, index=variable_names, columns=cointegrating_vector_names
    )
    intercept = pd.Series(sm_result.const.ravel(), index=variable_names)

    short_run_coefficients = {}
    for lag in range(1, k_ar_diff + 1):
        block = sm_result.gamma[:, (lag - 1) * k : lag * k]
        matrix = pd.DataFrame(block, index=variable_names, columns=variable_names)
        matrix.index.name = "Equation"
        short_run_coefficients[lag] = matrix

    residual_index = aligned.index[-sm_result.resid.shape[0] :]
    fitted_values = pd.DataFrame(
        sm_result.fittedvalues, index=residual_index, columns=variable_names
    )
    residuals = pd.DataFrame(
        sm_result.resid, index=residual_index, columns=variable_names
    )

    forecast_values = sm_result.predict(steps=forecast_steps)
    forecast_index = pd.RangeIndex(1, forecast_steps + 1, name="Step")
    forecast = pd.DataFrame(
        forecast_values, index=forecast_index, columns=variable_names
    )

    return VECMResult(
        rank=rank,
        k_ar_diff=k_ar_diff,
        cointegrating_vectors=cointegrating_vectors,
        adjustment_speeds=adjustment_speeds,
        short_run_coefficients=short_run_coefficients,
        intercept=intercept,
        fitted_values=fitted_values,
        residuals=residuals,
        forecast=forecast,
        n_observations=int(sm_result.nobs),
        variable_names=variable_names,
    )
