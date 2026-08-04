"""Time Series Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM

from financetoolkit.econometrics import cointegration_model, regression_model

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


def arima_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds the coefficient table (Coefficient, Std. Error) of the constant, AR and MA
    coefficients from a `get_arima_forecast` result dict.

    Args:
        result (dict): A fitted ARIMA result dict, as returned by `get_arima_forecast`.

    Returns:
        pd.DataFrame: The coefficient table.
    """
    p, _, q = result["order"]
    names = (
        ["Intercept"]
        + [f"AR{i}" for i in range(1, p + 1)]
        + [f"MA{j}" for j in range(1, q + 1)]
    )
    coefficients = [
        result["constant"],
        *result["ar_coefficients"],
        *result["ma_coefficients"],
    ]
    standard_errors = [
        result["constant_std_error"],
        *result["ar_std_errors"],
        *result["ma_std_errors"],
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
) -> dict:
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
        dict: The fitted constant/AR/MA coefficients and their standard errors,
        in-sample residuals and fitted values, and the `forecast_steps`-ahead
        forecast -- keys `order`, `constant`, `constant_std_error`,
        `ar_coefficients`, `ar_std_errors`, `ma_coefficients`, `ma_std_errors`,
        `residuals`, `fitted_values`, `forecast`, `sum_of_squared_residuals`,
        `n_observations`, `converged`. Call `arima_summary_table` for a coefficient
        table.

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
    print(time_series_model.arima_summary_table(result).round(4))
    print(result["forecast"].round(4))
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

    return {
        "order": (p, d, q),
        "constant": constant,
        "constant_std_error": constant_std_error,
        "ar_coefficients": ar_coefficients,
        "ar_std_errors": ar_std_errors,
        "ma_coefficients": ma_coefficients,
        "ma_std_errors": ma_std_errors,
        "residuals": sm_result.resid.rename("Residuals"),
        "fitted_values": sm_result.fittedvalues.rename("Fitted Values"),
        "forecast": forecast_series,
        "sum_of_squared_residuals": float(sm_result.sse),
        "n_observations": int(sm_result.nobs),
        "converged": bool(sm_result.mle_retvals.get("converged", True)),
    }


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
    regression result dict.

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
    equations: dict[str, dict],
    variable_names: list[str],
    n_lags: int,
    coefficient_offset: int,
) -> dict[int, pd.DataFrame]:
    """
    Slices the shared lag-block coefficients (see `_build_lagged_design`'s column
    order) out of each variable's fitted regression result dict, used by
    `get_var_forecast` (`coefficient_offset=1`, right after the intercept) to build
    the `{lag: (k, k) DataFrame}` matrices `Phi_l`, indexed and columned by variable
    name (row = equation i.e. dependent variable, column = predictor variable).
    """
    k = len(variable_names)
    matrices: dict[int, pd.DataFrame] = {}
    for lag in range(1, n_lags + 1):
        start = coefficient_offset + (lag - 1) * k
        end = start + k
        matrix = pd.DataFrame(
            {
                target_column: pd.Series(
                    equation["coefficients"][start:end], index=variable_names
                )
                for target_column, equation in equations.items()
            }
        ).T
        matrix.index.name = "Equation"
        matrices[lag] = matrix
    return matrices


def var_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds a combined coefficient table from a `get_var_forecast` result dict, each
    equation's coefficient table (see `regression_model.regression_summary_table`)
    concatenated side by side under a top-level column per variable.

    Args:
        result (dict): A fitted VAR result dict, as returned by `get_var_forecast`.

    Returns:
        pd.DataFrame: The combined coefficient table.
    """
    return pd.concat(
        {
            name: regression_model.regression_summary_table(equation)
            for name, equation in result["equations"].items()
        },
        axis=1,
    )


def get_var_forecast(data: pd.DataFrame, lags: int, forecast_steps: int = 1) -> dict:
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

    - Sims, C.A. (1980). "Macroeconomics and Reality." Econometrica, 48(1), 1-48.

    Args:
        data (pd.DataFrame): One column per series (e.g. several assets' returns),
        at least 2 columns.
        lags (int): The VAR order (number of lagged periods of every series included
        in every equation).
        forecast_steps (int, optional): The number of periods ahead to forecast.
        Defaults to 1.

    Returns:
        dict: The fitted intercept and per-lag coefficient matrices, the underlying
        per-equation regression result dicts, in-sample fitted values/residuals, and
        the `forecast_steps`-ahead forecast -- keys `lags`, `intercept`,
        `coefficient_matrices`, `equations`, `fitted_values`, `residuals`,
        `forecast`, `n_observations`, `variable_names`. Call `var_summary_table` for
        a combined coefficient table.

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
    print(result["coefficient_matrices"][1].round(4))
    print(result["forecast"].round(4))
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

    equations: dict[str, dict] = {
        column: regression_model.get_ols(target[column], design, add_constant=True)
        for column in variable_names
    }

    intercept = pd.Series(
        {column: equation["coefficients"][0] for column, equation in equations.items()}
    )
    coefficient_matrices = _extract_lag_matrices(
        equations, variable_names, n_lags=lags, coefficient_offset=1
    )

    fitted_values = pd.DataFrame(
        {column: equation["fitted_values"] for column, equation in equations.items()},
        index=target.index,
    )
    residuals = pd.DataFrame(
        {column: equation["residuals"] for column, equation in equations.items()},
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

    return {
        "lags": lags,
        "intercept": intercept,
        "coefficient_matrices": coefficient_matrices,
        "equations": equations,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "forecast": forecast,
        "n_observations": len(design),
        "variable_names": variable_names,
    }


# ---------------------------------------------------------------------------
# IRF / FEVD
# ---------------------------------------------------------------------------


def _wold_ma_matrices(
    coefficient_matrices: dict[int, pd.DataFrame],
    variable_names: list[str],
    lags: int,
    periods: int,
) -> list[np.ndarray]:
    """
    Recursively builds the Wold (moving-average) representation `Psi_0, ..., Psi_periods`
    of a fitted VAR(`lags`), via `Psi_0 = I` and
    `Psi_n = SUM_{l=1}^{min(n, lags)} Phi_l @ Psi_(n-l)` -- the standard MA(infinity)
    recursion (Lütkepohl, 2005, "New Introduction to Multiple Time Series Analysis",
    Section 2.1.2), truncated at `periods`. Shared by `get_impulse_response_function`
    (which orthogonalizes `Psi_n` via a Cholesky factor) and
    `get_variance_decomposition` (which accumulates `Psi_n`'s squared entries).
    """
    k = len(variable_names)
    phi = {
        lag: coefficient_matrices[lag].loc[variable_names, variable_names].to_numpy()
        for lag in range(1, lags + 1)
    }

    psi = [np.eye(k)]
    for n in range(1, periods + 1):
        psi_n = np.zeros((k, k))
        for lag in range(1, min(n, lags) + 1):
            psi_n = psi_n + phi[lag] @ psi[n - lag]
        psi.append(psi_n)

    return psi


def irf_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds a combined response table from a `get_impulse_response_function` result
    dict, each shock variable's response table concatenated side by side under a
    top-level column per shock variable.

    Args:
        result (dict): A fitted IRF result dict, as returned by
        `get_impulse_response_function`.

    Returns:
        pd.DataFrame: The combined response table.
    """
    return pd.concat(result["responses"], axis=1)


_VAR_RESULT_KEYS = {"coefficient_matrices", "variable_names", "lags", "residuals"}


def _require_var_result(var_result: dict) -> None:
    """
    Shared validation for `get_impulse_response_function`/`get_variance_decomposition`
    -- both need a fitted `get_var_forecast` result dict specifically (not just any
    dict), so this checks for the keys they actually read rather than accepting an
    arbitrary mapping.
    """
    if not isinstance(var_result, dict) or not _VAR_RESULT_KEYS.issubset(var_result):
        raise TypeError(
            "var_result must be a dict returned by get_var_forecast (missing one or "
            f"more of {sorted(_VAR_RESULT_KEYS)})."
        )


def get_impulse_response_function(
    var_result: dict, periods: int = 10, orthogonalized: bool = True
) -> dict:
    """
    Trace out the Impulse Response Function (IRF) of a fitted `get_var_forecast`
    result -- how a one-standard-deviation shock to each variable propagates through
    the whole system over `periods` periods ahead.

    Also known as: IRF.

    A VAR's moving-average (Wold) representation `Y_t = SUM_{n=0}^inf(Psi_n * e_(t-n))`
    gives the reduced-form response `Psi_n` of the system `n` periods after a unit
    reduced-form shock `e_t`. Because the reduced-form residuals are typically
    contemporaneously correlated (a shock to one variable's equation coincides with
    shocks to the others'), `orthogonalized=True` (the default, and the standard
    choice for interpretation) instead identifies structural shocks via a Cholesky
    decomposition of the residual covariance matrix `Sigma_u = P @ P.T` and reports
    `Theta_n = Psi_n @ P` -- the response to a one-standard-deviation structural shock
    that is, by construction, uncorrelated with the other shocks. This makes the
    ordering of `var_result.variable_names` an identifying assumption: the Cholesky
    factor is lower-triangular, so the first variable is treated as contemporaneously
    unaffected by (i.e. causally prior to) every other variable, the second is
    contemporaneously affected only by the first, and so on -- reorder the columns of
    the `data` passed to `get_var_forecast` to change this assumption.

    For more information about the method, see the following reference:

    - Lütkepohl, H. (2005). "New Introduction to Multiple Time Series Analysis."
      Springer, Chapter 2.3.2.

    Args:
        var_result (dict): A fitted VAR result dict, from `get_var_forecast`.
        periods (int, optional): The number of periods ahead to trace the response
        out to. Defaults to 10.
        orthogonalized (bool, optional): Whether to orthogonalize the shocks via a
        Cholesky decomposition of the residual covariance matrix (see above). If
        False, returns the reduced-form response to a unit (not orthogonalized, and
        not standardized) reduced-form shock instead -- rarely what's wanted given
        contemporaneously correlated residuals, but provided for completeness/
        cross-checking against other software. Defaults to True.

    Returns:
        dict: The horizon-by-horizon response of every variable to a shock in every
        other variable -- keys `periods`, `orthogonalized`, `responses`,
        `variable_names`. Call `irf_summary_table` for a combined table.

    Raises:
        TypeError: If `var_result` is not a `get_var_forecast` result dict.
        ValueError: If `periods` is not a positive integer.

    Notes:
    - Verified by simulating a synthetic bivariate VAR(1) with a known coefficient
    matrix and confirming the impact (horizon 0) response of a variable to its own
    shock equals that shock's standard deviation, and by confirming the sum of squared
    orthogonalized responses at each horizon matches `get_variance_decomposition`'s
    un-normalized forecast error variance exactly.

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
    var_result = time_series_model.get_var_forecast(data, lags=1)
    irf = time_series_model.get_impulse_response_function(var_result, periods=10)
    print(irf["responses"]["Y1"].round(4))
    ```
    """
    _require_var_result(var_result)
    if periods < 1:
        raise ValueError("periods must be a positive integer.")

    variable_names = var_result["variable_names"]
    psi = _wold_ma_matrices(
        var_result["coefficient_matrices"],
        variable_names,
        var_result["lags"],
        periods,
    )

    if orthogonalized:
        sigma_u = var_result["residuals"][variable_names].cov().to_numpy()
        cholesky_factor = np.linalg.cholesky(sigma_u)
        theta = [psi_n @ cholesky_factor for psi_n in psi]
    else:
        theta = psi

    horizon_index = pd.RangeIndex(0, periods + 1, name="Horizon")
    responses = {
        shock_name: pd.DataFrame(
            [theta_n[:, shock_index] for theta_n in theta],
            index=horizon_index,
            columns=variable_names,
        )
        for shock_index, shock_name in enumerate(variable_names)
    }

    return {
        "periods": periods,
        "orthogonalized": orthogonalized,
        "responses": responses,
        "variable_names": variable_names,
    }


def variance_decomposition_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds a combined decomposition table from a `get_variance_decomposition` result
    dict, each response variable's decomposition table concatenated side by side
    under a top-level column per response variable.

    Args:
        result (dict): A fitted FEVD result dict, as returned by
        `get_variance_decomposition`.

    Returns:
        pd.DataFrame: The combined decomposition table.
    """
    return pd.concat(result["decomposition"], axis=1)


def get_variance_decomposition(var_result: dict, periods: int = 10) -> dict:
    """
    Compute the (orthogonalized) Forecast Error Variance Decomposition (FEVD) of a
    fitted `get_var_forecast` result -- what fraction of each variable's `h`-step-
    ahead forecast error variance is attributable to each variable's own structural
    shock, for `h = 1, ..., periods`.

    Also known as: FEVD, variance decomposition.

    Using the same orthogonalized moving-average representation as
    `get_impulse_response_function` (`Theta_n = Psi_n @ P`, `P` the Cholesky factor of
    the residual covariance matrix), the `h`-step-ahead forecast error of variable `i`
    is `SUM_{n=0}^(h-1) Theta_n @ e_(t+h-n)`. Because the structural shocks `e` are
    orthogonal by construction, the total forecast error variance splits additively
    across shocks:

    - Var_i(h) = SUM_j SUM_{n=0}^(h-1) Theta_n[i, j]^2

    and shock `j`'s share is that inner sum divided by `Var_i(h)`. A large own-shock
    share at short horizons that decays as `h` grows is the classic signature of a
    variable that is initially self-driven but increasingly explained by the rest of
    the system over time. As with the IRF, the split depends on the Cholesky
    ordering of `var_result.variable_names` (see `get_impulse_response_function`).

    For more information about the method, see the following reference:

    - Lütkepohl, H. (2005). "New Introduction to Multiple Time Series Analysis."
      Springer, Chapter 2.3.3.

    Args:
        var_result (dict): A fitted VAR result dict, from `get_var_forecast`.
        periods (int, optional): The forecast horizon to decompose out to. Defaults
        to 10.

    Returns:
        dict: Each variable's forecast error variance share attributable to each
        shock, at every horizon `1, ..., periods` -- keys `periods`,
        `decomposition`, `variable_names`. Call `variance_decomposition_summary_table`
        for a combined table.

    Raises:
        TypeError: If `var_result` is not a `get_var_forecast` result dict.
        ValueError: If `periods` is not a positive integer.

    Notes:
    - Verified by confirming every row of every variable's decomposition sums to 1,
    and that horizon-1's decomposition for the first (Cholesky-ordered) variable
    attributes 100% of its variance to its own shock (by construction, the first
    variable has no contemporaneous exposure to the others' structural shocks).

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
    var_result = time_series_model.get_var_forecast(data, lags=1)
    fevd = time_series_model.get_variance_decomposition(var_result, periods=10)
    print(fevd["decomposition"]["Y2"].round(4))
    ```
    """
    _require_var_result(var_result)
    if periods < 1:
        raise ValueError("periods must be a positive integer.")

    variable_names = var_result["variable_names"]
    k = len(variable_names)
    psi = _wold_ma_matrices(
        var_result["coefficient_matrices"],
        variable_names,
        var_result["lags"],
        periods,
    )
    sigma_u = var_result["residuals"][variable_names].cov().to_numpy()
    cholesky_factor = np.linalg.cholesky(sigma_u)
    theta = [psi_n @ cholesky_factor for psi_n in psi]

    squared_theta = np.stack([t**2 for t in theta])  # shape (periods + 1, k, k)
    cumulative = np.cumsum(squared_theta, axis=0)  # cumulative over horizon

    # h-step-ahead forecast error variance sums n = 0, ..., h - 1 -- i.e. horizon h
    # (1-indexed) reads off `cumulative[h - 1]`, so horizons 1..periods are
    # `cumulative[0:periods]`.
    horizon_index = pd.RangeIndex(1, periods + 1, name="Horizon")
    decomposition = {
        variable_names[i]: pd.DataFrame(
            cumulative[0:periods, i, :]
            / cumulative[0:periods, i, :].sum(axis=1, keepdims=True),
            index=horizon_index,
            columns=variable_names,
        )
        for i in range(k)
    }

    return {
        "periods": periods,
        "decomposition": decomposition,
        "variable_names": variable_names,
    }


# ---------------------------------------------------------------------------
# VECM
# ---------------------------------------------------------------------------


def vecm_summary_table(result: dict) -> pd.DataFrame:
    """
    Builds a combined table of the cointegrating vector(s) (`beta`) and adjustment
    speeds (`alpha`) from a `get_vecm_forecast` result dict, side by side, indexed by
    variable name.

    Args:
        result (dict): A fitted VECM result dict, as returned by `get_vecm_forecast`.

    Returns:
        pd.DataFrame: The combined table.
    """
    return pd.concat(
        {
            "Cointegrating Vector": result["cointegrating_vectors"],
            "Adjustment Speed": result["adjustment_speeds"],
        },
        axis=1,
    )


def get_vecm_forecast(
    data: pd.DataFrame,
    k_ar_diff: int = 1,
    forecast_steps: int = 1,
    significance: float = 0.05,
) -> dict:
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
        dict: The fitted cointegrating vector(s), adjustment speeds, short-run
        coefficient matrices, in-sample fitted values/residuals (differenced scale),
        and the `forecast_steps`-ahead forecast (levels scale) -- keys `rank`,
        `k_ar_diff`, `cointegrating_vectors` (`beta`, shape `(k, rank)`),
        `adjustment_speeds` (`alpha`, shape `(k, rank)`), `short_run_coefficients`
        (`{lag: (k, k) DataFrame}` of the `Gamma_l` matrices, same layout as
        `get_var_forecast`'s `coefficient_matrices`), `intercept`, `fitted_values`,
        `residuals`, `forecast`, `n_observations`, `variable_names`. Call
        `vecm_summary_table` for the cointegrating vector(s)/adjustment speeds side
        by side.

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
    print(result["cointegrating_vectors"].round(4))
    print(result["forecast"].round(4))
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

    return {
        "rank": rank,
        "k_ar_diff": k_ar_diff,
        "cointegrating_vectors": cointegrating_vectors,
        "adjustment_speeds": adjustment_speeds,
        "short_run_coefficients": short_run_coefficients,
        "intercept": intercept,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "forecast": forecast,
        "n_observations": int(sm_result.nobs),
        "variable_names": variable_names,
    }
