"""GARCH Model"""

import hashlib
import math
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import optimize, special
from scipy.stats import qmc

from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    join_options,
)
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

ALPHA_CONSTRAINT = 0.5

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# Number of deterministic Sobol starting points a GARCH-family fit tries, in addition to the domain-informed initial guess.  # noqa: E501
N_MULTI_START_POINTS = 32
# The distance to a bound, as a fraction of the bounds' width, within which a weight of a
# warm-started fit counts as on that bound.
BOUND_DISTANCE = 1e-6
# The largest gradient, on the parameters scaled to [0, 1] within their bounds and relative to
# the size of the negative log-likelihood, at which an L-BFGS-B search counts as converged
# inside the bounds rather than stalled on the stationarity wall.
STATIONARY_GRADIENT = 1e-5
# L-BFGS-B with an analytic gradient converges to the last digits the likelihood resolves.
LBFGSB_OPTIONS = {"ftol": 1e-14, "gtol": 1e-9, "maxiter": 2000}
# The value an L-BFGS-B search with an analytic gradient sees outside the admissible region
# (a persistence of 1 or more): large but finite, so its line search backtracks to the
# admissible side instead of stopping on an infinite value.
INADMISSIBLE_VALUE = 1e10

# The GARCH and GJR-GARCH recursions start from a backcast: an exponentially weighted average of the first squared returns, with the same window and decay as the arch package (Kevin Sheppard), so the fitted parameters agree with it.  # noqa: E501
BACKCAST_OBSERVATIONS = 75
BACKCAST_DECAY = 0.94

# A fit needs at least two returns: one for the sample Variance its bounds anchor to and one more for the recursion.  # noqa: E501
MINIMUM_FIT_OBSERVATIONS = 2


def _apply_per_column(
    returns: pd.DataFrame, function, forecast: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Applies a single-series GARCH-family function to every column of `returns` on that
    column's own non-missing returns.

    Fitting all columns on the rows where every column has a return would cut each
    series down to the shortest history (a ticker that listed later, or one with a gap),
    and the recursion cannot run over missing values, so each column is fitted on its
    own observations instead.

    Args:
        returns (pd.DataFrame): A DataFrame of returns, one column per series, which may
            hold missing values.
        function (callable): The function to apply to the np.ndarray of each column's
            returns, such as `get_garch` or `get_garch_forecast`.
        forecast (bool, optional): Whether `function` returns a forecast (a fixed number
            of steps ahead, indexed 0 to time_steps - 1) rather than one value per
            observation (indexed by the column's own dates). Defaults to False.
        **kwargs: The keyword arguments passed on to `function`. A `realized_measure`
            DataFrame is passed on per column, on that column's return dates.

    Returns:
        pd.DataFrame: The results, one column per column of `returns`. Columns with fewer
        than two returns are NaN, and so are the dates a column has no return for.
    """
    results = {}
    measure = kwargs.pop("realized_measure", None)

    for column in returns.columns:
        column_returns = returns[column].dropna()

        if len(column_returns) < MINIMUM_FIT_OBSERVATIONS:
            continue

        # A realized measure (of the realized GARCH) is a frame like the returns, taken on the column's own return dates.  # noqa: E501
        column_kwargs = (
            {
                **kwargs,
                "realized_measure": measure[column]
                .reindex(column_returns.index)
                .to_numpy(dtype=float),
            }
            if isinstance(measure, pd.DataFrame)
            else kwargs
        )

        values = np.asarray(function(column_returns.to_numpy(), **column_kwargs))

        results[column] = (
            pd.Series(values)
            if forecast
            else pd.Series(values, index=column_returns.index[: len(values)])
        )

    if forecast:
        return pd.DataFrame(results, columns=returns.columns, dtype=float)

    return pd.DataFrame(
        results, index=returns.index, columns=returns.columns, dtype=float
    )


def _series_key(values) -> tuple[str, tuple[int, ...]]:
    """The SHA-256 hash and shape of an array, which identify a series in a weights cache."""
    values = np.ascontiguousarray(values, dtype=float)
    return hashlib.sha256(values.tobytes()).hexdigest(), values.shape


def _cached_weights(
    cache: OrderedDict, lock: threading.Lock, key: tuple, fit
) -> np.ndarray:
    """
    Returns the weights stored under `key`, or fits them with `fit()` and stores a copy,
    dropping the least recently used entry beyond EGARCH_WEIGHTS_CACHE_SIZE entries. The
    lock guards the cache, which the threads of the MCP server share.
    """
    with lock:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached.copy()

    weights = fit()

    with lock:
        cache[key] = weights.copy()
        if len(cache) > EGARCH_WEIGHTS_CACHE_SIZE:
            cache.popitem(last=False)

    return weights


# Outside the stationarity region the log-likelihood is infinite, so the finite-difference
# gradient there subtracts infinities; L-BFGS-B handles the NaN step, and the warning it
# raises says nothing to the user.
@np.errstate(invalid="ignore", over="ignore", divide="ignore")
def _fit_multi_start(
    wrapper_func,
    bounds: list[tuple[float, float]],
    initial_guess: list[float],
    model_name: str,
    number_of_points: int = N_MULTI_START_POINTS,
    gradient=None,
) -> np.ndarray:
    """
    Fits a GARCH-family log-likelihood by running `scipy.optimize.minimize` (L-BFGS-B)
    from many deterministic starting points spread across `bounds` and keeps the best. The
    search runs on the parameters scaled to [0, 1] within their bounds.

    With an analytic `gradient`, one more L-BFGS-B search from the best point converges to
    the maximum: it sees a large finite value (INADMISSIBLE_VALUE) beyond the stationarity
    wall rather than an infinite one, so its line search backs away from the wall. The
    result is finished with Nelder-Mead only when the search stalled: when it failed, or
    when its gradient on the scaled parameters, leaving out the components that point out
    of a bound the weights sit on (the projected gradient), is larger than
    STATIONARY_GRADIENT times the size of the negative log-likelihood. A maximum on a bound
    (an ARCH weight of 0, for example) is therefore accepted as it is.

    A global optimizer with a stochastic walk (such as simulated annealing) is avoided on
    purpose: the log-likelihood is not bit-identical across libm/BLAS backends, and tiny
    floating-point differences flip its accept/reject decisions, so the same seed reaches
    different optima on different platforms.

    Gradient descent from many fixed points doesn't have that failure mode: there is no
    stochastic accept/reject step, so tiny platform-level floating-point noise in the
    objective doesn't redirect the search, it only perturbs the final digits of wherever
    that descent was already headed. The starting points are drawn from a deterministic,
    seeded Sobol sequence (a low-discrepancy sequence generated by fixed bit arithmetic,
    not by evaluating the objective), so the set of points tried is itself the same on
    every platform. A start that lands outside the feasible region (`wrapper_func`
    returning a constraint-violation penalty) or whose local search doesn't converge is
    simply discarded rather than crashing the fit.

    Args:
        wrapper_func (Callable): The (constrained) negative log-likelihood to minimize.
        bounds (list[tuple[float, float]]): The parameter bounds every starting point is
        drawn from and every local search is confined to.
        initial_guess (list[float]): A domain-informed starting point, tried in addition
        to the Sobol-sampled ones.
        model_name (str): The model name to include in the warning message if every
        attempt fails.
        number_of_points (int, optional): The number of Sobol-sampled starting points tried
        in addition to the initial guess; 0 runs a single local search from the initial
        guess. Defaults to N_MULTI_START_POINTS (32).
        gradient (Callable | None, optional): The gradient of `wrapper_func` with respect to
        the parameters (analytic, see `_garch_family_score`). Defaults to None, which lets
        L-BFGS-B take finite differences: one more evaluation of the likelihood per
        parameter at every step, which the analytic gradient replaces by one recursion of
        the derivatives.

    Returns:
        np.ndarray: The fitted weights from the best-scoring successful attempt, or an
        array of NaN (same length as `initial_guess`) if every attempt failed.
    """
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)

    # The search runs on the parameters scaled to [0, 1] within their bounds: omega is of
    # the order of a daily variance (1e-5) and the other weights of 0.1, and on their own
    # scales the numerical gradient is so badly conditioned that L-BFGS-B stops close to
    # where it started.
    span = upper - lower

    def scaled_objective(unit: np.ndarray) -> float:
        return wrapper_func(lower + unit * span)

    def scaled_value_and_gradient(
        unit: np.ndarray, inadmissible: float = np.inf
    ) -> tuple[float, np.ndarray]:
        value = scaled_objective(unit)
        if gradient is None or not np.isfinite(value):
            return inadmissible, np.zeros(dim)
        values = np.asarray(gradient(lower + unit * span), dtype=float) * span
        return value, np.where(np.isfinite(values), values, 0.0)

    sampler = qmc.Sobol(d=dim, seed=42)
    starts = np.vstack(
        [
            (np.asarray(initial_guess, dtype=float) - lower) / span,
            sampler.random(number_of_points),
        ]
    )

    best_x = None
    best_value = np.inf
    best_gradient = None

    for start in starts:
        try:
            result = optimize.minimize(
                scaled_objective if gradient is None else scaled_value_and_gradient,
                np.clip(start, 0.0, 1.0),
                jac=gradient is not None,
                bounds=[(0.0, 1.0)] * dim,
                method="L-BFGS-B",
            )
        except (ValueError, FloatingPointError):
            continue

        if not result.success or not np.isfinite(result.fun):
            continue

        if result.fun < best_value:
            best_value = result.fun
            best_x = lower + result.x * span
            best_gradient = result.jac

    if best_x is None and number_of_points == 0:
        # A single search from earlier weights (a warm start) that stalls on the constraint
        # wall still starts the Nelder-Mead polish below from those weights, when admissible.
        start = np.clip(
            (np.asarray(initial_guess, dtype=float) - lower) / span, 0.0, 1.0
        )
        if np.isfinite(scaled_objective(start)):
            best_x = lower + start * span
            best_value = scaled_objective(start)

    if best_x is None:
        logger.warning(
            "The %s optimization did not converge from any of %d starting points. "
            "Returning NaN weights instead.",
            model_name,
            len(starts),
        )
        return np.full(dim, np.nan)

    if gradient is not None:
        # The searches above stop at the first step that reaches the stationarity wall, an infinite value their line search cannot back away from, so they only pick the best region. With the analytic gradient one more search from the best point sees a large finite value beyond the wall instead, backs away from it and converges to the maximum.  # noqa: E501
        refined = optimize.minimize(
            scaled_value_and_gradient,
            (best_x - lower) / span,
            args=(INADMISSIBLE_VALUE,),
            jac=True,
            bounds=[(0.0, 1.0)] * dim,
            method="L-BFGS-B",
            options=LBFGSB_OPTIONS,
        )
        best_gradient = None
        if refined.fun <= best_value:
            best_value = refined.fun
            best_x = lower + refined.x * span
            best_gradient = refined.jac if refined.success else None

    # The stationarity constraint is a wall of infinite values, along which gradient steps
    # stall, and a search can stop short along a bound; Nelder-Mead, which only compares
    # values, finishes the search from the best point in those cases. Where the gradient
    # vanishes, apart from a bound it points out of (the projected gradient), L-BFGS-B has
    # converged and the polish would only repeat it.
    unit = (best_x - lower) / span
    stalled = best_gradient is None or not np.all(np.isfinite(best_gradient))
    if not stalled:
        projected = np.where(
            ((unit < BOUND_DISTANCE) & (best_gradient > 0))
            | ((unit > 1 - BOUND_DISTANCE) & (best_gradient < 0)),
            0.0,
            best_gradient,
        )
        stalled = float(np.max(np.abs(projected))) > STATIONARY_GRADIENT * max(
            1.0, abs(best_value)
        )
    if stalled:
        polished = optimize.minimize(
            lambda unit: (
                scaled_objective(unit) if np.all((unit >= 0) & (unit <= 1)) else np.inf
            ),
            unit,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000 * dim},
        )
        if np.isfinite(polished.fun) and polished.fun < best_value:
            best_x = lower + polished.x * span

    return best_x


def get_backcast(returns: np.ndarray) -> float:
    """
    Calculate the backcast the GARCH variance recursions start from: the variance before the first
    return, an exponentially weighted average (decay 0.94) of the first 75 squared returns, with the
    first return weighted most.

    The recursion needs a variance and a squared return for the period before the first
    observation. Starting from the first squared return alone makes every fitted
    parameter depend on the size of that single return: a small one starts the recursion
    far below the true variance, the likelihood then prefers a lower persistence (beta)
    to forget that start quickly, and the estimates are biased. An average of the first
    returns is a far steadier estimate of the variance at the start of the sample. The
    window and decay are those of the arch package, so the fitted parameters agree with
    it.

    For more information, see:

    - Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
    Journal of Econometrics, 31(3), 307-327.

    Shares similar logic:
    - arch, VolatilityProcess.backcast: https://github.com/bashtage/arch/blob/v7.2.0/arch/univariate/volatility.py#L580

    Args:
        returns (np.ndarray): A np.ndarray of returns.

    Returns:
        float: The backcast, a variance in the units of the squared returns.
    """
    window = min(BACKCAST_OBSERVATIONS, len(returns))
    decay_weights = BACKCAST_DECAY ** np.arange(window)

    return float(np.sum(returns[:window] ** 2 * decay_weights / decay_weights.sum()))


def _linear_variance_recursion(
    driver: np.ndarray, persistence: float, first_variance: float, time_steps: int
) -> np.ndarray:
    """
    Computes the variance recursion of GARCH(1, 1) and GJR-GARCH(1, 1),
    sigma2[i] = driver[i - 1] + persistence * sigma2[i - 1], from sigma2[0].

    Args:
        driver (np.ndarray): The part of each step that does not depend on the previous
            variance, for steps 1 to time_steps - 1.
        persistence (float): The weight of the previous variance (beta).
        first_variance (float): The variance of the first step.
        time_steps (int): The number of steps.

    Returns:
        np.ndarray: The variance of every step.
    """
    sigma2 = np.empty(time_steps)
    sigma2[0] = first_variance

    if time_steps > 1:
        # scipy.signal takes around a second to import, so only where it is needed.
        from scipy import signal  # noqa: PLC0415

        sigma2[1:], _ = signal.lfilter(
            [1.0], [1.0, -persistence], driver, zi=[persistence * first_variance]
        )

    return sigma2


def garch_log_maximization(
    weights: list, returns: np.ndarray, t: int, p: int = 1, q: int = 1
) -> float:
    """
    Calculate the negative Gaussian log-likelihood of a GARCH(1, 1) model, up to constants: the
    objective the GARCH fit minimizes, lower for weights that fit the returns better.

    The formula is as follows:

    - -SUM(-ln(v_i) - (u_i ^ 2) / v_i)

    Args:
        weights (list): List with the values for omega, alpha and beta
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize GARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        int: The result of the calculation -SUM(-ln(v_i) - (u_i ^ 2) / v_i)
    """
    # Convert weights to a numpy array for vectorized operations
    weights_array = np.array(weights)

    # Compute GARCH values using a vectorized function
    garch = get_garch(returns, weights_array, t, p=p, q=q)

    # garch[i] is built from returns[i - 1] (garch[0] from the backcast), so it pairs with returns[i] itself and every observation, the first included, enters the likelihood.  # noqa: E501
    u = returns[:t]
    v = garch[:t]

    # Use np.sum to calculate the sum of the log-likelihood
    result = -np.sum(-np.log(v) - u**2 / v)

    return result


def _log_maximization_gradient(
    weights, returns: np.ndarray, t: int, method: str
) -> np.ndarray:
    """
    The gradient of `garch_log_maximization` ("garch") or `gjr_garch_log_maximization`
    ("gjr"): their objective, SUM(ln(v_i) + u_i ^ 2 / v_i), is -2 times the normal
    log-likelihood without its constant, whose gradient `_garch_family_score` gives.
    """
    return -2 * _garch_family_score(
        returns[:t], weights, method, "normal", "zero", get_backcast(returns)
    )


def get_garch_weights(
    returns: np.ndarray, t: int | None = None, p: int = 1, q: int = 1
) -> np.ndarray:
    """
    Calculate the weights (parameters) of a GARCH(p, q) model by multi-start optimization: how strongly
    the variance reacts to the last shock (alpha), how long it remembers (beta) and its constant (omega).

    The weights are estimated by trying many starting values of (1 - alpha - beta) sigma_l,
    alpha and beta, while maximizing: SUM(-ln(v_i) - (u_i ^ 2) / v_i).
    With the constraints:
    - alpha > 0
    - beta > 0
    - alpha + beta < 1
    - 0 < (1 - alpha - beta) sigma_l <= 2 * Var(returns), which follows from the long run
      variance sigma_l being of the same order as the sample Variance under stationarity.

    Referenced against:
    - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Args:
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize GARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.ndarray: The weights [omega, alpha, beta]. If the fit fails to converge
        from every starting point, this is `[nan, nan, nan]` (see `_fit_multi_start`).
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0].to_numpy()
    if t is None:
        t = len(returns)

    # Omega is (1 - alpha - beta) * long run variance, not the long run variance itself, so both the bound and the starting value are anchored to the sample Variance. Searching omega over [0, 1] instead puts the optimum of a daily return series in the bottom millionth of the range, which the global search then routinely misses.  # noqa: E501
    variance = float(np.var(returns[: t - 1]))
    omega_upper_bound = max(2 * variance, 1e-9)

    bounds = [(1e-12, omega_upper_bound), (1e-9, 1), (1e-9, 1)]
    initial_omega = min(max(variance * 0.1, 1e-12), omega_upper_bound)

    initial_guess = [initial_omega, 0.1, 0.8]

    # Define the wrapper function for optimization that applies the constraints
    def wrapper_func(parameters):
        alpha = parameters[1]
        beta = parameters[2]
        if alpha + beta >= 1:  # Constraint
            return np.inf  # Return a large number to represent an invalid solution
        return garch_log_maximization(parameters, returns, t, p, q)

    # Seeded so fitted parameters are reproducible across runs, and kept, since the GJR-GARCH fit, the DCC model and the factor models fit the same returns again.  # noqa: E501
    return _cached_weights(
        GARCH_FAMILY_WEIGHTS_CACHE,
        GARCH_FAMILY_WEIGHTS_CACHE_LOCK,
        ("get_garch_weights", *_series_key(returns), t, p, q),
        lambda: _fit_multi_start(
            wrapper_func,
            bounds,
            initial_guess,
            "GARCH",
            gradient=lambda parameters: _log_maximization_gradient(
                parameters, returns, t, "garch"
            ),
        ),
    )


def get_garch(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    time_steps: int | None = None,
    optimization_t: int | None = None,
    p: int = 1,
    q: int = 1,
    backcast: float | None = None,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Calculate the conditional variance of the GARCH model: how volatile each period is expected to be
    given the shocks and the variance before it.

    GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is for
    instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH models.

    The recursion starts from the backcast (see `get_backcast`), which stands in for both the squared return and the
    variance before the first observation: sigma2[0] = omega + (alpha + beta) * backcast.

    More information can be found in:
    - https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#GARCH
    - Generalized Autoregressive Conditional Heteroskedasticity, by Tim Bollerslev
    - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1
    - Options, Futures & other Derivates, by John C. Hull; ISBN: 0-13-022444-8

    Referenced against:
    - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha and beta. Note that these are used all columns
        in the returns.
        time_steps (int): Time steps to calculate GARCH for.
        optimization_t (int): Time steps to optimize GARCH for. It is only used if no weights are given.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.
        backcast (float | None): The squared return and variance before the first observation of a
        np.ndarray of returns. Defaults to None, which uses the backcast of the returns themselves
        (see `get_backcast`); a mean model passes the backcast of its starting residuals instead.

    Returns:
        np.array | pd.Series | pd.DataFrame: An object with sigma_2 values
    """
    if p != 1 or q != 1:
        raise InvalidArgumentError(
            "Invalid input for p or/and q, currently only GARCH(1, 1) is implemented."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = _apply_per_column(
                    returns.loc[sub_period],
                    get_garch,
                    weights=weights,
                    time_steps=time_steps,
                    optimization_t=optimization_t,
                    p=p,
                    q=q,
                )

                if not period_data.dropna(how="all").empty:
                    period_data_list.append(period_data)

            garch = pd.concat(period_data_list, axis=0)

            return garch
        return _apply_per_column(
            returns,
            get_garch,
            weights=weights,
            time_steps=time_steps,
            optimization_t=optimization_t,
            p=p,
            q=q,
        )
    if isinstance(returns, pd.Series):
        return get_garch(
            returns=returns.values,
            weights=weights,
            time_steps=time_steps,
            optimization_t=optimization_t,
            p=p,
            q=q,
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            if optimization_t is None:
                optimization_t = len(returns)
            weights = get_garch_weights(returns, optimization_t, p, q)
        if time_steps is None:
            time_steps = len(returns)

        omega, alpha, beta = weights[0], weights[1], weights[2]

        # The first step treats the squared return and the variance before the sample as the backcast, as the arch package does, see get_backcast.  # noqa: E501
        if backcast is None:
            backcast = get_backcast(returns)
        first_variance = omega + (alpha + beta) * backcast

        # The GARCH(1, 1) variance (omega plus alpha times the previous squared return plus
        # beta times the previous variance) is a first-order linear recursion, which lfilter
        # computes in compiled code with the same operations in the same order as the loop
        # it replaces.
        return _linear_variance_recursion(
            omega + alpha * returns[: time_steps - 1] ** 2,
            beta,
            first_variance,
            time_steps,
        )

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def get_garch_forecast(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    weights: list | np.ndarray | None = None,
    forecast_steps: int = 10,
    p: int = 1,
    q: int = 1,
):
    """Calculate the GARCH variance forecast (sigma_2): the conditional variance expected for each period
    after the last return, as a shock fades towards the long-run variance.

    GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is for
    instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH models.

    The forecasting with GARCH is done with the following formula, for a horizon of
    h = 1, 2, ... time steps ahead of the end of the sample:

    sigma_2(t + h) = long_run_variance + (sigma_2(t + 1) - long_run_variance) * (alpha + beta) ** (h - 1)

    Where `sigma_2(t + 1) = omega + alpha * u_t ** 2 + beta * sigma_2(t)` is the one
    step ahead conditional variance implied by the *last* observed return, so the
    first returned value is a genuine forecast rather than the in-sample conditional
    variance of the final observation.

    For more:
    - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

    Shares similar logic:
    - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha and beta
        forecast_steps (int): Time steps to calculate GARCH for
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.ndarray: sigma_2 sigma_2 forecasts, going from the forecast from 0 time period to t
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = _apply_per_column(
                    returns.loc[sub_period],
                    get_garch_forecast,
                    forecast=True,
                    weights=weights,
                    forecast_steps=forecast_steps,
                    p=p,
                    q=q,
                )
                period_data.columns = [
                    col + " " + str(sub_period) for col in period_data.columns
                ]

                if not period_data.dropna(how="all").empty:
                    period_data_list.append(period_data)

            garch_forecast = pd.concat(period_data_list, axis=1)

            return garch_forecast

        return _apply_per_column(
            returns,
            get_garch_forecast,
            forecast=True,
            weights=weights,
            forecast_steps=forecast_steps,
            p=p,
            q=q,
        )
    if isinstance(returns, pd.Series):
        return get_garch_forecast(
            returns=returns.values,
            weights=weights,
            forecast_steps=forecast_steps,
            p=p,
            q=q,
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            weights = get_garch_weights(returns, p=p, q=q)

        # One step past the sample so the recursion consumes the final observed return, making garch_values[-1] the one step ahead conditional variance rather than the in-sample variance of the last observation.  # noqa: E501
        garch_values = get_garch(returns, weights, len(returns) + 1, p=p, q=q)

        # Already a variance, so never squared; seeded from the one step ahead value.
        long_run_variance = weights[0] / (1 - weights[1] - weights[2])
        current_variance = garch_values[-1]
        persistence = weights[1] + weights[2]

        sigma_2 = np.zeros(forecast_steps)
        for i in range(forecast_steps):
            sigma_2[i] = (
                long_run_variance
                + (current_variance - long_run_variance) * persistence**i
            )

        return sigma_2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def gjr_garch_log_maximization(
    weights: list, returns: np.ndarray, t: int, p: int = 1, q: int = 1
) -> float:
    """
    Calculate the negative Gaussian log-likelihood of a GJR-GARCH model, up to constants: the
    objective the GJR-GARCH fit minimizes, lower for weights that fit the returns better.

    The formula is as follows:

    - -SUM(-ln(v_i) - (u_i ^ 2) / v_i)

    Args:
        weights (list): List with the values for omega, alpha, gamma and beta.
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize GJR-GARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        float: The result of the calculation -SUM(-ln(v_i) - (u_i ^ 2) / v_i)
    """
    weights_array = np.array(weights)

    garch = get_gjr_garch(returns, weights_array, t, p=p, q=q)

    # Every observation, the first included, enters the likelihood, see garch_log_maximization.
    u = returns[:t]
    v = garch[:t]

    return -np.sum(-np.log(v) - u**2 / v)


def get_gjr_garch_weights(
    returns: np.ndarray, t: int | None = None, p: int = 1, q: int = 1
) -> np.ndarray:
    """
    Calculate the weights (parameters) of a GJR-GARCH(1, 1, 1) model by multi-start optimization:
    how strongly the variance reacts to a shock, how much more to a negative one (gamma) and how long
    it remembers.

    The weights are estimated by maximizing the Gaussian log-likelihood, subject to:
    - omega > 0, alpha > 0, beta > 0
    - omega <= 2 * Var(returns), which follows from omega being
      (1 - alpha - gamma / 2 - beta) times the long run variance
    - alpha + gamma >= 0 (so the conditional variance stays non-negative regardless of
      the sign of the shock)
    - alpha + gamma / 2 + beta < 1 (the GJR-GARCH stationarity condition, assuming a
      symmetric error distribution)

    The fitted GARCH(1, 1) weights with gamma = 0 serve as an extra starting point and
    as a floor: GARCH is the special case gamma = 0, so a local search from them competes
    with the multi-start result, and when both end at a lower log-likelihood than the
    GARCH fit (a local optimum), the GARCH weights with gamma = 0 are returned instead.

    Referenced against:
    - arch, GARCH (GJR-GARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Args:
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize GJR-GARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.ndarray: The weights [omega, alpha, gamma, beta]. If the fit fails to
        converge from every starting point, this is `[nan, nan, nan, nan]` (see
        `_fit_multi_start`).
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0].to_numpy()
    if t is None:
        t = len(returns)

    # Anchored to the sample Variance for the same reason as in get_garch_weights: omega is (1 - alpha - gamma / 2 - beta) * long run variance, and searching it over [0, 1] leaves the optimum of a daily return series in the bottom millionth of the range, which cost this four-parameter fit far more than the three-parameter GARCH one (beta was routinely estimated near 0.2 rather than the 0.9 the likelihood actually prefers).  # noqa: E501
    variance = float(np.var(returns[: t - 1]))
    omega_upper_bound = max(2 * variance, 1e-9)

    bounds = [(1e-12, omega_upper_bound), (1e-9, 1), (-1, 1), (1e-9, 1)]
    initial_omega = min(max(variance * 0.1, 1e-12), omega_upper_bound)

    initial_guess = [initial_omega, 0.05, 0.1, 0.8]

    # GARCH is GJR-GARCH with gamma = 0, so the GJR-GARCH likelihood can never be lower at its optimum than the GARCH one. The four-parameter surface does have local optima the multi-start search can settle in (on simulated series it stopped below the GARCH likelihood), so the GARCH fit is both an extra starting point and a floor: a local search from it competes with the multi-start result, and the GARCH weights with gamma 0 are kept when neither does better.  # noqa: E501
    garch_weights = get_garch_weights(returns, t, p, q)
    nested_weights = (
        np.array([garch_weights[0], garch_weights[1], 0.0, garch_weights[2]])
        if np.all(np.isfinite(garch_weights))
        else None
    )

    def wrapper_func(parameters):
        alpha, gamma, beta = parameters[1], parameters[2], parameters[3]
        if alpha + gamma / 2 + beta >= 1:  # Stationarity constraint
            return np.inf
        if alpha + gamma < 0:  # Non-negativity constraint
            return np.inf
        return gjr_garch_log_maximization(parameters, returns, t, p, q)

    def gradient(parameters):
        return _log_maximization_gradient(parameters, returns, t, "gjr")

    weights = _fit_multi_start(
        wrapper_func, bounds, initial_guess, "GJR-GARCH", gradient=gradient
    )

    if nested_weights is None:
        return weights

    # A separate local search rather than one more start in the multi-start search: only the best start is polished there, so an extra start can displace the one that polishes to the higher likelihood.  # noqa: E501
    candidates = [
        weights,
        _fit_multi_start(
            wrapper_func,
            bounds,
            nested_weights,
            "GJR-GARCH (local search from the GARCH fit)",
            number_of_points=0,
            gradient=gradient,
        ),
        nested_weights,
    ]
    scores = [
        wrapper_func(candidate) if np.all(np.isfinite(candidate)) else np.inf
        for candidate in candidates
    ]

    # The first candidate wins ties, so the GARCH weights are only returned when the search ends below them.  # noqa: E501
    return candidates[int(np.argmin(scores))]


def get_gjr_garch(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    time_steps: int | None = None,
    optimization_t: int | None = None,
    p: int = 1,
    q: int = 1,
    backcast: float | None = None,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Calculate the conditional variance of the GJR-GARCH model: how volatile each period is expected to
    be, with negative shocks raising it by more than positive ones.

    GJR-GARCH extends GARCH with a leverage term that lets negative shocks (bad news)
    raise volatility by more than positive shocks of the same size, a well documented
    asymmetry in equity returns that symmetric GARCH cannot capture:

    - sigma_t^2 = omega + (alpha + gamma * 1[u_(t-1) < 0]) * u_(t-1)^2 + beta * sigma_(t-1)^2

    A positive gamma indicates the presence of a leverage effect.

    The recursion starts from the backcast (see `get_backcast`) for the squared return and the variance
    before the first observation, with the indicator at its expectation of one half:
    sigma2[0] = omega + (alpha + gamma / 2 + beta) * backcast.

    For more information about the method, see the following paper:

    - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
    between the Expected Value and the Volatility of the Nominal Excess Return on
    Stocks." The Journal of Finance, 48(5), 1779-1801.

    Referenced against:
    - arch, GARCH (GJR-GARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta. Note that these are used
        for all columns in the returns.
        time_steps (int): Time steps to calculate GJR-GARCH for.
        optimization_t (int): Time steps to optimize GJR-GARCH for. It is only used if no weights are given.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.
        backcast (float | None): The squared return and variance before the first observation of a
        np.ndarray of returns. Defaults to None, which uses the backcast of the returns themselves
        (see `get_backcast`); a mean model passes the backcast of its starting residuals instead.

    Returns:
        np.array | pd.Series | pd.DataFrame: An object with sigma_2 values
    """
    if p != 1 or q != 1:
        raise InvalidArgumentError(
            "Invalid input for p or/and q, currently only GJR-GARCH(1, 1, 1) is implemented."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = _apply_per_column(
                    returns.loc[sub_period],
                    get_gjr_garch,
                    weights=weights,
                    time_steps=time_steps,
                    optimization_t=optimization_t,
                    p=p,
                    q=q,
                )

                if not period_data.dropna(how="all").empty:
                    period_data_list.append(period_data)

            gjr_garch = pd.concat(period_data_list, axis=0)

            return gjr_garch
        return _apply_per_column(
            returns,
            get_gjr_garch,
            weights=weights,
            time_steps=time_steps,
            optimization_t=optimization_t,
            p=p,
            q=q,
        )
    if isinstance(returns, pd.Series):
        return get_gjr_garch(
            returns=returns.values,
            weights=weights,
            time_steps=time_steps,
            optimization_t=optimization_t,
            p=p,
            q=q,
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            if optimization_t is None:
                optimization_t = len(returns)
            weights = get_gjr_garch_weights(returns, optimization_t, p, q)
        if time_steps is None:
            time_steps = len(returns)

        omega, alpha, gamma, beta = weights
        lagged = returns[: time_steps - 1]
        indicator = np.where(lagged < 0, 1.0, 0.0)

        # The first step uses the backcast for the squared return and the variance before the sample, with the leverage term at its expectation (a negative shock half of the time), as the arch package does.  # noqa: E501
        if backcast is None:
            backcast = get_backcast(returns)
        first_variance = omega + (alpha + gamma / 2 + beta) * backcast

        # A first-order linear recursion like GARCH, see _linear_variance_recursion.
        return _linear_variance_recursion(
            omega + (alpha + gamma * indicator) * lagged**2,
            beta,
            first_variance,
            time_steps,
        )

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def get_gjr_garch_forecast(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    weights: list | np.ndarray | None = None,
    forecast_steps: int = 10,
    p: int = 1,
    q: int = 1,
):
    """Calculate the GJR-GARCH variance forecast (sigma_2): the conditional variance expected for each
    period after the last return, as a shock fades towards the long-run variance.

    The forecasting with GJR-GARCH is done with the following formula, using the
    "effective" persistence (alpha + gamma / 2 + beta), which reduces to the plain
    GARCH forecast formula when gamma = 0, for a horizon of h = 1, 2, ... time steps
    ahead of the end of the sample:

    sigma_2(t + h) = long_run_variance + (sigma_2(t + 1) - long_run_variance) * (alpha + gamma / 2 + beta) ** (h - 1)

    Where `sigma_2(t + 1)` is the one step ahead conditional variance implied by the
    *last* observed return, so the first returned value is a genuine forecast rather
    than the in-sample conditional variance of the final observation.

    For more information about the method, see the following paper:

    - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
    between the Expected Value and the Volatility of the Nominal Excess Return on
    Stocks." The Journal of Finance, 48(5), 1779-1801.

    Referenced against:
    - arch, GARCH (GJR-GARCH forecast): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta.
        forecast_steps (int): Time steps to calculate GJR-GARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.ndarray: sigma_2 forecasts, going from the forecast from 0 time period to t
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = _apply_per_column(
                    returns.loc[sub_period],
                    get_gjr_garch_forecast,
                    forecast=True,
                    weights=weights,
                    forecast_steps=forecast_steps,
                    p=p,
                    q=q,
                )
                period_data.columns = [
                    col + " " + str(sub_period) for col in period_data.columns
                ]

                if not period_data.dropna(how="all").empty:
                    period_data_list.append(period_data)

            gjr_garch_forecast = pd.concat(period_data_list, axis=1)

            return gjr_garch_forecast

        return _apply_per_column(
            returns,
            get_gjr_garch_forecast,
            forecast=True,
            weights=weights,
            forecast_steps=forecast_steps,
            p=p,
            q=q,
        )
    if isinstance(returns, pd.Series):
        return get_gjr_garch_forecast(
            returns=returns.values,
            weights=weights,
            forecast_steps=forecast_steps,
            p=p,
            q=q,
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            weights = get_gjr_garch_weights(returns, p=p, q=q)

        # One step past the sample so the recursion consumes the final observed return, making garch_values[-1] the one step ahead conditional variance rather than the in-sample variance of the last observation.  # noqa: E501
        garch_values = get_gjr_garch(returns, weights, len(returns) + 1, p=p, q=q)

        omega, alpha, gamma, beta = weights
        persistence = alpha + gamma / 2 + beta

        long_run_variance = omega / (1 - persistence)
        current_variance = garch_values[-1]

        sigma_2 = np.zeros(forecast_steps)
        for i in range(forecast_steps):
            sigma_2[i] = (
                long_run_variance
                + (current_variance - long_run_variance) * persistence**i
            )

        return sigma_2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def egarch_log_maximization(
    weights: list, returns: np.ndarray, t: int, p: int = 1, q: int = 1
) -> float:
    """
    Calculate the negative Gaussian log-likelihood of an EGARCH model, up to constants: the objective
    the EGARCH fit minimizes, lower for weights that fit the returns better.

    The formula is as follows:

    - -SUM(-ln(v_i) - (u_i ^ 2) / v_i)

    Args:
        weights (list): List with the values for omega, alpha, gamma and beta.
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize EGARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        float: The result of the calculation -SUM(-ln(v_i) - (u_i ^ 2) / v_i)
    """
    weights_array = np.array(weights)

    garch = get_egarch(returns, weights_array, t, p=p, q=q)

    u = returns[:t]
    v = garch[:t]

    return -np.sum(-np.log(v) - u**2 / v)


# The EGARCH fits of the latest return series. A fit takes seconds to a minute (33 starting
# points with numerical gradients), and get_egarch, get_egarch_forecast and
# get_egarch_parameters each fit the same returns, so a repeated fit is taken from here.
EGARCH_WEIGHTS_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
EGARCH_WEIGHTS_CACHE_SIZE = 64
# The cache is shared by threads (the MCP server runs tools in threads), and an OrderedDict
# is not safe to reorder and read concurrently.
EGARCH_WEIGHTS_CACHE_LOCK = threading.Lock()
# The fits of every other model, distribution and mean (and of GARCH(1, 1) itself, see
# get_garch_weights), by the returns and the settings of the fit: the variance, the
# forecast, the parameters, the diagnostics, the simulation and the econometric tests of
# one series each fit the same model again. Like the EGARCH cache it keeps the
# EGARCH_WEIGHTS_CACHE_SIZE fits used last.
GARCH_FAMILY_WEIGHTS_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
GARCH_FAMILY_WEIGHTS_CACHE_LOCK = threading.Lock()


def get_egarch_weights(
    returns: np.ndarray, t: int | None = None, p: int = 1, q: int = 1
) -> np.ndarray:
    """
    Calculate the weights (parameters) of an EGARCH(1, 1) model by multi-start optimization: how
    strongly the log variance reacts to the size and the sign of a shock and how long it remembers.

    Unlike GARCH and GJR-GARCH, EGARCH models the log of the conditional variance, so
    omega, alpha and gamma are unconstrained in sign; only |beta| < 1 is required for
    stationarity of the log-variance process.

    Referenced against:
    - arch, EGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html

    Args:
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize EGARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.ndarray: The weights [omega, alpha, gamma, beta]. If the fit fails to
        converge from every starting point, this is `[nan, nan, nan, nan]` (see
        `_fit_multi_start`).
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0].to_numpy()
    if t is None:
        t = len(returns)

    # Omega sits well below -1 on a log scale; the rest stay near empirical EGARCH.
    bounds = [(-15, 5), (-1, 1), (-1, 1), (-0.999, 0.999)]
    initial_guess = [0.0, 0.1, -0.1, 0.9]

    def wrapper_func(parameters):
        beta = parameters[3]
        if abs(beta) >= 1:  # Stationarity constraint
            return np.inf
        return egarch_log_maximization(parameters, returns, t, p, q)

    return _cached_weights(
        EGARCH_WEIGHTS_CACHE,
        EGARCH_WEIGHTS_CACHE_LOCK,
        (*_series_key(returns), t, p, q),
        lambda: _fit_multi_start(wrapper_func, bounds, initial_guess, "EGARCH"),
    )


def get_egarch(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    time_steps: int | None = None,
    optimization_t: int | None = None,
    p: int = 1,
    q: int = 1,
    backcast: float | None = None,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Calculate the conditional variance of the EGARCH model: how volatile each period is expected to be,
    modelled through its logarithm, with shocks of each sign moving it differently.

    EGARCH models the log of the conditional variance, which avoids having to
    constrain the parameters to keep the variance positive and, like GJR-GARCH, lets
    negative and positive shocks of the same size have a different impact on
    volatility (the leverage effect):

    - ln(sigma_t^2) = omega + beta * ln(sigma_(t-1)^2) + alpha * (|z_(t-1)| - E|z|) + gamma * z_(t-1)

    Where `z = u / sigma` is the standardized shock and `E|z| = sqrt(2 / pi)` under the
    assumption of standard normal innovations. A negative gamma indicates the presence
    of a leverage effect (negative shocks raise volatility by more than positive ones).
    The recursion starts from the logarithm of the backcast (see get_backcast), as the
    arch package does, and every observation enters the likelihood.

    For more information about the method, see the following paper:

    - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
    Approach." Econometrica, 59(2), 347-370.

    Referenced against:
    - arch, EGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta. Note that these are used
        for all columns in the returns.
        time_steps (int): Time steps to calculate EGARCH for.
        optimization_t (int): Time steps to optimize EGARCH for. It is only used if no weights are given.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.
        backcast (float | None): The squared return and variance before the first observation of a
        np.ndarray of returns. Defaults to None, which uses the backcast of the returns themselves
        (see `get_backcast`); a mean model passes the backcast of its starting residuals instead.

    Returns:
        np.array | pd.Series | pd.DataFrame: An object with sigma_2 values
    """
    if p != 1 or q != 1:
        raise InvalidArgumentError(
            "Invalid input for p or/and q, currently only EGARCH(1, 1) is implemented."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = _apply_per_column(
                    returns.loc[sub_period],
                    get_egarch,
                    weights=weights,
                    time_steps=time_steps,
                    optimization_t=optimization_t,
                    p=p,
                    q=q,
                )

                if not period_data.dropna(how="all").empty:
                    period_data_list.append(period_data)

            egarch = pd.concat(period_data_list, axis=0)

            return egarch
        return _apply_per_column(
            returns,
            get_egarch,
            weights=weights,
            time_steps=time_steps,
            optimization_t=optimization_t,
            p=p,
            q=q,
        )
    if isinstance(returns, pd.Series):
        return get_egarch(
            returns=returns.values,
            weights=weights,
            time_steps=time_steps,
            optimization_t=optimization_t,
            p=p,
            q=q,
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            if optimization_t is None:
                optimization_t = len(returns)
            weights = get_egarch_weights(returns, optimization_t, p, q)
        if time_steps is None:
            time_steps = len(returns)

        omega, alpha, gamma, beta = (float(weight) for weight in weights)
        expected_absolute_z = math.sqrt(2 / math.pi)

        # The log variance depends on the previous variance itself, so the recursion
        # cannot be vectorised. It runs on plain Python floats: indexing numpy arrays and
        # calling numpy on single numbers costs around a microsecond each, which made the
        # likelihood, evaluated thousands of times per fit, the bulk of an EGARCH fit.
        lagged = returns[: time_steps - 1].tolist()
        sigma2 = [0.0] * time_steps
        # The first step starts from the backcast (see get_backcast) with the shock terms
        # at their expected value of zero, as the arch package does.
        if backcast is None:
            backcast = get_backcast(returns)
        log_variance = omega + beta * math.log(backcast + 1e-12)
        log_variance = min(max(log_variance, -20.0), 20.0)
        sigma2[0] = math.exp(log_variance)

        for i, previous_return in enumerate(lagged, start=1):
            standardized_shock = previous_return / math.sqrt(sigma2[i - 1])
            log_variance = (
                omega
                + beta * log_variance
                + alpha * (abs(standardized_shock) - expected_absolute_z)
                + gamma * standardized_shock
            )
            # Without clipping, an overflow step makes the likelihood spuriously attractive.
            log_variance = min(max(log_variance, -20.0), 20.0)
            sigma2[i] = math.exp(log_variance)

        return np.array(sigma2)

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def get_egarch_forecast(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    weights: list | np.ndarray | None = None,
    forecast_steps: int = 10,
    p: int = 1,
    q: int = 1,
):
    """Calculate the EGARCH variance forecast (sigma_2): the conditional variance expected for each period
    after the last return, as the log variance reverts to its long-run level.

    Since EGARCH is nonlinear in log-variance, the multi-step-ahead forecast is
    approximated by forecasting the log-variance (whose forecast has a closed form,
    since the alpha and gamma terms have zero expectation under standard normal
    innovations) and then exponentiating:

    - E[ln(sigma_(t+h)^2)] = long_run_log_variance + beta^(h - 1) * (ln(sigma_(t+1)^2) - long_run_log_variance)
    - sigma_(t+h)^2 ~= exp(E[ln(sigma_(t+h)^2)])

    Where `long_run_log_variance = omega / (1 - beta)` and `sigma_(t+1)^2` is the one
    step ahead conditional variance implied by the *last* observed return, so the
    first returned value is a genuine forecast rather than the in-sample conditional
    variance of the final observation. This is the standard practical
    approximation (it understates the true forecast somewhat due to Jensen's
    inequality, since E[exp(X)] != exp(E[X])), not an exact closed form.

    For more information about the method, see the following paper:

    - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
    Approach." Econometrica, 59(2), 347-370.

    Shares similar logic:
    - arch, EGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta.
        forecast_steps (int): Time steps to calculate EGARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.ndarray: sigma_2 forecasts, going from the forecast from 0 time period to t
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = _apply_per_column(
                    returns.loc[sub_period],
                    get_egarch_forecast,
                    forecast=True,
                    weights=weights,
                    forecast_steps=forecast_steps,
                    p=p,
                    q=q,
                )
                period_data.columns = [
                    col + " " + str(sub_period) for col in period_data.columns
                ]

                if not period_data.dropna(how="all").empty:
                    period_data_list.append(period_data)

            egarch_forecast = pd.concat(period_data_list, axis=1)

            return egarch_forecast

        return _apply_per_column(
            returns,
            get_egarch_forecast,
            forecast=True,
            weights=weights,
            forecast_steps=forecast_steps,
            p=p,
            q=q,
        )
    if isinstance(returns, pd.Series):
        return get_egarch_forecast(
            returns=returns.values,
            weights=weights,
            forecast_steps=forecast_steps,
            p=p,
            q=q,
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            weights = get_egarch_weights(returns, p=p, q=q)

        # One step past the sample so the recursion consumes the final observed return, making garch_values[-1] the one step ahead conditional variance rather than the in-sample variance of the last observation.  # noqa: E501
        garch_values = get_egarch(returns, weights, len(returns) + 1, p=p, q=q)

        omega, _, _, beta = weights
        current_log_variance = np.log(garch_values[-1])
        long_run_log_variance = omega / (1 - beta)

        sigma_2 = np.zeros(forecast_steps)
        for i in range(forecast_steps):
            log_forecast = long_run_log_variance + (beta**i) * (
                current_log_variance - long_run_log_variance
            )
            sigma_2[i] = np.exp(log_forecast)

        return sigma_2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


# ──────────────────────────────────────────────────────────────────────────────
# The GARCH family: APARCH and the family GARCH sub-models, component and integrated
# GARCH, innovation distributions and mean models, written from the papers cited in
# each function. Every model is fitted by the same multi-start search as GARCH(1, 1).
# ──────────────────────────────────────────────────────────────────────────────

# The variance parameters of each model, in the order of its weights.
GARCH_FAMILY_VARIANCE_PARAMETERS = {
    "garch": ["Omega", "Alpha", "Beta"],
    "gjr": ["Omega", "Alpha", "Gamma", "Beta"],
    "egarch": ["Omega", "Alpha", "Gamma", "Beta"],
    "igarch": ["Omega", "Alpha"],
    "aparch": ["Omega", "Alpha", "Gamma", "Beta", "Delta"],
    "tgarch": ["Omega", "Alpha", "Gamma", "Beta"],
    "ngarch": ["Omega", "Alpha", "Beta", "Delta"],
    "avgarch": ["Omega", "Alpha", "Gamma", "Shift", "Beta"],
    "nagarch": ["Omega", "Alpha", "Shift", "Beta"],
    "component": ["Omega", "Alpha", "Beta", "Rho", "Phi"],
    "figarch": ["Omega", "Phi", "D", "Beta"],
    "realized": ["Omega", "Beta", "Gamma", "Xi", "Phi", "Tau 1", "Tau 2", "Sigma U"],
}

# The mean model parameters, after the variance parameters in the weights.
GARCH_FAMILY_MEAN_PARAMETERS = {
    "zero": [],
    "constant": ["Mu"],
    "arma": ["Mu", "AR(1)", "MA(1)"],
    "in_mean": ["Mu", "Lambda"],
    "arfima": ["Mu", "AR(1)", "Fractional D", "MA(1)"],
}

# The parameters of each standardized innovation distribution, after the mean parameters.
GARCH_FAMILY_DISTRIBUTION_PARAMETERS = {
    "normal": [],
    "t": ["Shape"],
    "skewt": ["Shape", "Skew"],
    "skewt_fernandez_steel": ["Shape", "Skew"],
    "ged": ["Shape"],
    "johnson_su": ["Shape", "Skew"],
}

# The search bounds and starting values of the variance parameters, on returns scaled to a unit Variance (see get_garch_family_weights), so a long-run variance of about 1. The rotation Gamma of APARCH and family GARCH stays within (-1, 1) so the shock term is never negative, and Delta within the (0.05, 4) of the arch package.  # noqa: E501
VARIANCE_BOUNDS = {
    "garch": ([(1e-9, 2), (1e-9, 1), (1e-9, 1)], [0.1, 0.1, 0.8]),
    "gjr": ([(1e-9, 2), (1e-9, 1), (-1, 1), (1e-9, 1)], [0.1, 0.05, 0.1, 0.8]),
    "egarch": ([(-2, 2), (-1, 1), (-1, 1), (-0.999, 0.999)], [0.0, 0.1, -0.1, 0.9]),
    "igarch": ([(1e-9, 2), (1e-9, 1)], [0.05, 0.1]),
    "aparch": (
        [(1e-9, 2), (1e-9, 1), (-0.9997, 0.9997), (1e-9, 1), (0.05, 4)],
        [0.1, 0.1, 0.1, 0.8, 1.5],
    ),
    "tgarch": (
        [(1e-9, 2), (1e-9, 1), (-0.9997, 0.9997), (1e-9, 1)],
        [0.1, 0.1, 0.1, 0.8],
    ),
    "ngarch": ([(1e-9, 2), (1e-9, 1), (1e-9, 1), (0.05, 4)], [0.1, 0.1, 0.8, 1.5]),
    "avgarch": (
        [(1e-9, 2), (1e-9, 1), (-0.9997, 0.9997), (-10, 10), (1e-9, 1)],
        [0.1, 0.1, 0.1, 0.0, 0.8],
    ),
    "nagarch": ([(1e-9, 2), (1e-9, 1), (-10, 10), (1e-9, 1)], [0.1, 0.1, 0.3, 0.8]),
    "component": (
        [(1e-9, 2), (1e-9, 1), (1e-9, 1), (1e-9, 1), (1e-9, 1)],
        [0.01, 0.05, 0.85, 0.99, 0.03],
    ),
    # Phi may be negative as long as every ARCH(∞) weight stays non-negative (see _figarch_weights).  # noqa: E501
    "figarch": (
        [(1e-9, 2), (-0.999, 0.999), (0, 1), (0, 0.999)],
        [0.05, 0.2, 0.4, 0.5],
    ),
    # The starting values of Omega and Xi of the realized GARCH follow from the realized measure (see get_garch_family_weights).  # noqa: E501
    "realized": (
        [
            (-10, 10),
            (0, 0.999),
            (0, 1),
            (-10, 10),
            (0, 3),
            (-2, 2),
            (-2, 2),
            (1e-3, 5),
        ],
        [0.0, 0.55, 0.4, 0.0, 1.0, 0.0, 0.05, 0.5],
    ),
}

# The FIGARCH variance is the ARCH(∞) representation truncated at 1,000 lags, as in the arch package and R's rugarch; the squared residuals before the sample are the backcast.  # noqa: E501
FIGARCH_TRUNCATION = 1000

# The variance models that need the whole history of the squared residuals (FIGARCH) or a realized measure (realized GARCH), so their recursion is not a one-step update and GARCH-in-mean, which runs one, does not take them.  # noqa: E501
HISTORY_METHODS = ("figarch", "realized")

# The Shape of the Student-t distributions is kept below 100, where they no longer differ from the normal distribution in any sample of returns.  # noqa: E501
DISTRIBUTION_BOUNDS = {
    "normal": ([], []),
    "t": ([(2.05, 100)], [8.0]),
    "skewt": ([(2.05, 100), (-0.99, 0.99)], [8.0, 0.0]),
    "skewt_fernandez_steel": ([(2.05, 100), (0.1, 10)], [8.0, 1.0]),
    "ged": ([(1.01, 50)], [1.5]),
    "johnson_su": ([(0.2, 20), (-10, 10)], [2.0, 0.0]),
}

# The models whose variance recursion is in a power of the volatility rather than in the variance, with that power: Delta for APARCH and NGARCH (a weight), 1 for the TGARCH and AVGARCH volatility and 2 for the rest.  # noqa: E501
POWER_OF_VOLATILITY = {"tgarch": 1.0, "avgarch": 1.0}

# The expectations over a standardized distribution (the persistence of APARCH and family GARCH, the share of negative shocks of GJR-GARCH) are integrals evaluated on a grid z = sinh(u), dense around zero and reaching far into the tails, with trapezoid weights.  # noqa: E501
_GRID_U = np.linspace(-11, 11, 4401)
EXPECTATION_GRID = np.sinh(_GRID_U)
EXPECTATION_WEIGHTS = np.cosh(_GRID_U) * (_GRID_U[1] - _GRID_U[0])
EXPECTATION_WEIGHTS[[0, -1]] /= 2

LOG_2_PI = math.log(2 * math.pi)
EXPECTED_ABSOLUTE_NORMAL = math.sqrt(2 / math.pi)


def validate_garch_family(method: str, distribution: str, mean: str) -> None:
    """
    Check that the model, the innovation distribution and the mean model are known: whether a
    GARCH-family specification can be fitted, raising an error when it cannot.

    Args:
        method (str): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch" (EGARCH), "igarch"
            (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH), "avgarch" (absolute
            value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH), "figarch" (FIGARCH),
            "realized" (Realized GARCH).
        distribution (str): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU).
        mean (str): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean" (GARCH-in-mean),
            "arfima" (ARFIMA(1, d, 1)).

    Raises:
        InvalidArgumentError: When one of them is not known.
    """
    for value, options, name in (
        (method, GARCH_FAMILY_VARIANCE_PARAMETERS, "method"),
        (distribution, GARCH_FAMILY_DISTRIBUTION_PARAMETERS, "distribution"),
        (mean, GARCH_FAMILY_MEAN_PARAMETERS, "mean"),
    ):
        if value not in options:
            raise InvalidArgumentError(
                f"{name} must be one of {join_options(options)}, not {value!r}. "
                f"Please pass one of these as {name}."
            )

    if method in HISTORY_METHODS and mean == "in_mean":
        raise InvalidArgumentError(
            f"GARCH-in-mean runs the variance one step at a time, which method={method!r} "
            "does not have. Please use mean='zero', 'constant', 'arma' or 'arfima'."
        )


def get_garch_family_parameter_names(
    method: str = "garch", distribution: str = "normal", mean: str = "zero"
) -> list[str]:
    """
    Calculate the names of the weights of a GARCH-family model: which parameter every weight is, the
    variance parameters, then the mean parameters and then those of the innovation distribution.

    Args:
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".

    Returns:
        list[str]: The parameter names, in the order of the weights.
    """
    validate_garch_family(method, distribution, mean)

    return (
        GARCH_FAMILY_VARIANCE_PARAMETERS[method]
        + GARCH_FAMILY_MEAN_PARAMETERS[mean]
        + GARCH_FAMILY_DISTRIBUTION_PARAMETERS[distribution]
    )


def _split_weights(
    weights, method: str, mean: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits the weights into the variance, mean and distribution parameters."""
    weights = np.asarray(weights, dtype=float)
    variance_end = len(GARCH_FAMILY_VARIANCE_PARAMETERS[method])
    mean_end = variance_end + len(GARCH_FAMILY_MEAN_PARAMETERS[mean])

    return weights[:variance_end], weights[variance_end:mean_end], weights[mean_end:]


def get_standardized_log_density(
    z: np.ndarray, distribution: str = "normal", parameters: tuple[float, ...] = ()
) -> np.ndarray:
    """
    Calculate the log density of a standardized innovation distribution (mean 0 and Variance 1) at the
    standardized residuals z = epsilon / sigma: how likely every shock is under the distribution.

    The formula is as follows, with Shape and Skew the distribution parameters:

    - Normal: ln f(z) = -(ln(2π) + z^2) / 2
    - Student-t (Bollerslev, 1987), Shape v > 2: ln f(z) = ln Γ((v + 1) / 2) - ln Γ(v / 2) - ln(π (v - 2)) / 2
      - (v + 1) / 2 * ln(1 + z^2 / (v - 2))
    - Hansen's skewed Student-t (Hansen, 1994), Shape η > 2 and Skew λ in (-1, 1):
      f(z) = b c (1 + ((b z + a) / (1 ± λ))^2 / (η - 2))^(-(η + 1) / 2), with 1 - λ below z = -a / b and 1 + λ
      above it, c = Γ((η + 1) / 2) / (√(π (η - 2)) Γ(η / 2)), a = 4 λ c (η - 2) / (η - 1) and b^2 = 1 + 3 λ^2 - a^2
    - Fernández and Steel's skewed Student-t (1998), Shape v > 2 and Skew ξ > 0: the standardized Student-t g
      scaled by ξ above zero and by 1 / ξ below it, f(x) = 2 / (ξ + 1 / ξ) * g(x / ξ) for x >= 0 and g(x ξ) for
      x < 0, then standardized with its mean m = M1 (ξ - 1 / ξ) and Variance s^2 = ξ^2 + 1 / ξ^2 - 1 - m^2,
      M1 = 2 √(v - 2) Γ((v + 1) / 2) / (√π (v - 1) Γ(v / 2)), so f(z) = s f(m + s z)
    - Generalized error distribution (Nelson, 1991), Shape v > 0: ln f(z) = ln v - ln λ - (1 + 1 / v) ln 2 - ln Γ(1 / v)
      - |z / λ|^v / 2, with λ = √(2^(-2 / v) Γ(1 / v) / Γ(3 / v))
    - Johnson SU (Johnson, 1949), Shape δ > 0 and Skew γ: x = sinh((n - γ) / δ) for a standard normal n,
      f(x) = δ / (√(2π) √(1 + x^2)) exp(-(γ + δ asinh(x))^2 / 2), standardized with its mean
      m = -exp(1 / (2 δ^2)) sinh(γ / δ) and Variance s^2 = (exp(1 / δ^2) - 1) (exp(1 / δ^2) cosh(2 γ / δ) + 1) / 2

    A negative Skew of Hansen's distribution, a Skew below 1 of Fernández and Steel's and a positive
    Skew of the Johnson SU make large negative shocks more likely than large positive ones.

    For more information, see the following papers:

    - Bollerslev, T. (1987). "A Conditionally Heteroskedastic Time Series Model for Speculative
    Prices and Rates of Return." The Review of Economics and Statistics, 69(3), 542-547.
    - Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation." International
    Economic Review, 35(3), 705-730.
    - Fernández, C. and Steel, M.F.J. (1998). "On Bayesian Modeling of Fat Tails and Skewness."
    Journal of the American Statistical Association, 93(441), 359-371.
    - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach."
    Econometrica, 59(2), 347-370.
    - Johnson, N.L. (1949). "Systems of Frequency Curves Generated by Methods of Translation."
    Biometrika, 36(1/2), 149-176.

    Referenced against:
    - arch, Normal: https://bashtage.github.io/arch/univariate/generated/arch.univariate.Normal.html
    - arch, StudentsT: https://bashtage.github.io/arch/univariate/generated/arch.univariate.StudentsT.html
    - arch, SkewStudent: https://bashtage.github.io/arch/univariate/generated/arch.univariate.SkewStudent.html
    - arch, GeneralizedError: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GeneralizedError.html
    - scipy, johnsonsu: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsu.html
    - scipy, gennorm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html

    Shares similar logic:
    - rugarch (R), ddist (sstd, jsu): https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-distributions.R#L1715

    Args:
        z (np.ndarray): The standardized residuals.
        distribution (str, optional): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (Sequence[float], optional): The Shape (and Skew) of the distribution.
            Defaults to (), for the normal distribution.

    Returns:
        np.ndarray: The log density at every z.
    """
    if distribution == "normal":
        return -0.5 * (LOG_2_PI + z * z)

    if distribution == "t":
        shape = parameters[0]
        return (
            special.gammaln((shape + 1) / 2)
            - special.gammaln(shape / 2)
            - 0.5 * math.log(math.pi * (shape - 2))
            - (shape + 1) / 2 * np.log1p(z * z / (shape - 2))
        )

    if distribution == "skewt":
        shape, skew = parameters
        c = math.exp(
            special.gammaln((shape + 1) / 2) - special.gammaln(shape / 2)
        ) / math.sqrt(math.pi * (shape - 2))
        a = 4 * skew * c * (shape - 2) / (shape - 1)
        b = math.sqrt(1 + 3 * skew**2 - a**2)
        x = b * z + a
        x = x / np.where(x < 0, 1 - skew, 1 + skew)
        return (
            math.log(b) + math.log(c) - (shape + 1) / 2 * np.log1p(x * x / (shape - 2))
        )

    if distribution == "skewt_fernandez_steel":
        shape, skew = parameters
        location, scale = _fernandez_steel_moments(shape, skew)
        x = location + scale * z
        x = np.where(x >= 0, x / skew, x * skew)
        return (
            math.log(2 / (skew + 1 / skew))
            + math.log(scale)
            + get_standardized_log_density(x, "t", (shape,))
        )

    if distribution == "ged":
        shape = parameters[0]
        scale = math.sqrt(
            2 ** (-2 / shape)
            * math.exp(special.gammaln(1 / shape) - special.gammaln(3 / shape))
        )
        return (
            math.log(shape)
            - math.log(scale)
            - (1 + 1 / shape) * math.log(2)
            - special.gammaln(1 / shape)
            - 0.5 * np.abs(z / scale) ** shape
        )

    if distribution == "johnson_su":
        shape, skew = parameters
        location, scale = _johnson_su_moments(shape, skew)
        x = location + scale * z
        return (
            math.log(scale)
            + math.log(shape)
            - 0.5 * LOG_2_PI
            - 0.5 * np.log1p(x * x)
            - 0.5 * (skew + shape * np.arcsinh(x)) ** 2
        )

    raise InvalidArgumentError(
        f"distribution must be one of {join_options(GARCH_FAMILY_DISTRIBUTION_PARAMETERS)}, "
        f"not {distribution!r}. Please pass one of these as distribution."
    )


def _fernandez_steel_moments(shape: float, skew: float) -> tuple[float, float]:
    """The mean and standard deviation of Fernández and Steel's skewed standardized Student-t."""
    absolute_moment = (
        2
        * math.sqrt(shape - 2)
        * math.exp(special.gammaln((shape + 1) / 2) - special.gammaln(shape / 2))
        / (math.sqrt(math.pi) * (shape - 1))
    )
    location = absolute_moment * (skew - 1 / skew)

    return location, math.sqrt(skew**2 + 1 / skew**2 - 1 - location**2)


def _johnson_su_moments(shape: float, skew: float) -> tuple[float, float]:
    """The mean and standard deviation of the Johnson SU distribution x = sinh((n - Skew) / Shape)."""
    w = math.exp(1 / shape**2)
    location = -math.sqrt(w) * math.sinh(skew / shape)
    variance = 0.5 * (w - 1) * (w * math.cosh(2 * skew / shape) + 1)

    return location, math.sqrt(variance)


def get_standardized_draws(
    distribution: str,
    parameters,
    size: int | tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Calculate random standardized innovations (mean 0 and Variance 1): shocks drawn from one of the
    distributions of `get_standardized_log_density`, for a simulation.

    The Student-t is drawn from numpy's Student-t scaled to a unit Variance; the skewed
    Student-t of Hansen and of Fernández and Steel draw the size of a standardized
    Student-t and put it below zero with probability (1 - λ) / 2 (Hansen) or 1 / (1 + ξ^2)
    (Fernández and Steel), scaled by the side's factor, and standardize the result; the GED
    takes |z| = λ (2 W)^(1 / v) with W from a Gamma(1 / v) distribution and a random sign; the
    Johnson SU transforms a standard normal draw.

    Shares similar logic:
    - rugarch (R), rdist: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-distributions.R#L1783

    Args:
        distribution (str): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's skewed
            Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized error
            distribution), "johnson_su" (Johnson SU).
        parameters (Sequence[float]): The Shape (and Skew) of the distribution.
        size (int | tuple[int, ...]): The number (or shape) of draws.
        rng (np.random.Generator): The random number generator.

    Returns:
        np.ndarray: The draws.
    """
    if distribution == "normal":
        return rng.standard_normal(size)

    if distribution == "johnson_su":
        shape, skew = parameters
        location, scale = _johnson_su_moments(shape, skew)
        return (np.sinh((rng.standard_normal(size) - skew) / shape) - location) / scale

    if distribution == "ged":
        shape = parameters[0]
        scale = math.sqrt(
            2 ** (-2 / shape)
            * math.exp(special.gammaln(1 / shape) - special.gammaln(3 / shape))
        )
        sign = rng.choice([-1.0, 1.0], size)
        return sign * scale * (2 * rng.gamma(1 / shape, 1.0, size)) ** (1 / shape)

    shape = parameters[0]
    draws = rng.standard_t(shape, size) * math.sqrt((shape - 2) / shape)

    if distribution == "t":
        return draws

    size_of_draws = np.abs(draws)
    uniform = rng.random(size)

    if distribution == "skewt":
        skew = parameters[1]
        c = math.exp(
            special.gammaln((shape + 1) / 2) - special.gammaln(shape / 2)
        ) / math.sqrt(math.pi * (shape - 2))
        a = 4 * skew * c * (shape - 2) / (shape - 1)
        b = math.sqrt(1 + 3 * skew**2 - a**2)
        x = np.where(
            uniform < (1 - skew) / 2,
            -(1 - skew) * size_of_draws,
            (1 + skew) * size_of_draws,
        )
        return (x - a) / b

    skew = parameters[1]
    location, scale = _fernandez_steel_moments(shape, skew)
    x = np.where(
        uniform < 1 / (1 + skew**2), -size_of_draws / skew, size_of_draws * skew
    )
    return (x - location) / scale


def get_standardized_expectation(
    values: np.ndarray, distribution: str = "normal", parameters: tuple[float, ...] = ()
) -> float:
    """
    Calculate the expectation E[h(z)] of a function h over a standardized innovation distribution: the
    average value of h(z) under the distribution, by the trapezoid rule on z = sinh(u).

    The function h is evaluated on EXPECTATION_GRID.

    Args:
        values (np.ndarray): h(z) at every point of EXPECTATION_GRID.
        distribution (str, optional): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (Sequence[float], optional): Its Shape (and Skew). Defaults to ().

    Returns:
        float: The expectation.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        density = np.exp(
            get_standardized_log_density(EXPECTATION_GRID, distribution, parameters)
        )
        return float(np.sum(values * density * EXPECTATION_WEIGHTS))


def get_standardized_raw_moments(
    distribution: str = "normal", parameters: tuple[float, ...] = (), order: int = 8
) -> np.ndarray:
    """
    Calculate the raw moments E[z^k], k = 0, ..., `order`, of a standardized innovation distribution
    (mean 0 and Variance 1): the averages of the powers of the shocks, which describe its shape.

    The normal moments are exact (0 for an odd k and (k - 1)!! for an even k); the other
    distributions are expectations over EXPECTATION_GRID (see `get_standardized_expectation`),
    which match the closed forms of the Student-t and the GED to four decimals up to the 8th
    moment.

    Args:
        distribution (str, optional): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (Sequence[float], optional): Its Shape (and Skew). Defaults to ().
        order (int, optional): The highest moment. Defaults to 8.

    Returns:
        np.ndarray: E[z^0] to E[z^order], NaN for a moment the distribution does not have (the
        k-th moment of a Student-t, skewed or not, with a Shape of at most k).
    """
    powers = np.arange(order + 1)

    if distribution == "normal":
        # (k - 1)!! for an even k: 1, 1, 3, 15, 105, ...
        return np.array(
            [0.0 if k % 2 else float(np.prod(np.arange(k - 1, 0, -2))) for k in powers]
        )

    student_t = distribution in ("t", "skewt", "skewt_fernandez_steel")

    return np.array(
        [
            (
                math.nan
                if student_t and parameters[0] <= k
                else (
                    1.0
                    if k == 0
                    else get_standardized_expectation(
                        EXPECTATION_GRID**k, distribution, parameters
                    )
                )
            )
            for k in powers
        ]
    )


def get_standardized_moments(
    distribution: str = "normal", parameters: tuple[float, ...] = ()
) -> tuple[float, float]:
    """
    Calculate the skewness E[z^3] and the kurtosis E[z^4] (not excess) of a standardized innovation
    distribution (mean 0 and Variance 1): how asymmetric and heavy-tailed its shocks are.

    The normal (0 and 3), the Student-t (0 and 3 + 6 / (v - 4)) and the GED (0 and
    Γ(5 / v) Γ(1 / v) / Γ(3 / v)^2) are exact; the skewed distributions and the Johnson SU are
    expectations over EXPECTATION_GRID (see `get_standardized_expectation`).

    Shares similar logic:
    - rugarch (R), dskewness: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-distributions.R#L1805
    - rugarch (R), dkurtosis: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-distributions.R#L1833

    Args:
        distribution (str, optional): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (Sequence[float], optional): Its Shape (and Skew). Defaults to ().

    Returns:
        tuple[float, float]: The skewness and the kurtosis, NaN for a moment the distribution
        does not have (a Student-t with a Shape of at most 3 or 4).
    """
    if distribution == "normal":
        return 0.0, 3.0
    if distribution == "ged":
        shape = parameters[0]
        return 0.0, math.exp(
            special.gammaln(5 / shape)
            + special.gammaln(1 / shape)
            - 2 * special.gammaln(3 / shape)
        )

    student_t = distribution in ("t", "skewt", "skewt_fernandez_steel")
    has_skewness = not student_t or parameters[0] > 3  # noqa: PLR2004
    has_kurtosis = not student_t or parameters[0] > 4  # noqa: PLR2004

    if distribution == "t":
        return (
            0.0 if has_skewness else math.nan,
            3 + 6 / (parameters[0] - 4) if has_kurtosis else math.nan,
        )

    return (
        (
            get_standardized_expectation(EXPECTATION_GRID**3, distribution, parameters)
            if has_skewness
            else math.nan
        ),
        (
            get_standardized_expectation(EXPECTATION_GRID**4, distribution, parameters)
            if has_kurtosis
            else math.nan
        ),
    )


def _power_weights(method: str, variance_weights) -> tuple[float, ...]:
    """Omega, Alpha, Gamma, Beta and Delta of APARCH and its TGARCH and NGARCH sub-models."""
    if method == "aparch":
        omega, alpha, gamma, beta, delta = variance_weights
    elif method == "tgarch":
        omega, alpha, gamma, beta = variance_weights
        delta = 1.0
    else:
        omega, alpha, beta, delta = variance_weights
        gamma = 0.0

    return omega, alpha, gamma, beta, delta


def _family_weights(method: str, variance_weights) -> tuple[float, ...]:
    """Omega, Alpha, Gamma (rotation), Shift, Beta and the power of AVGARCH and NAGARCH."""
    if method == "avgarch":
        omega, alpha, gamma, shift, beta = variance_weights
        return omega, alpha, gamma, shift, beta, 1.0

    omega, alpha, shift, beta = variance_weights
    return omega, alpha, 0.0, shift, beta, 2.0


def get_garch_family_persistence(
    variance_weights,
    method: str = "garch",
    distribution: str = "normal",
    distribution_weights: tuple[float, ...] = (),
) -> float:
    """
    Calculate the persistence of a GARCH-family model: how much of a shock to the (power of the)
    conditional variance carries over to the next period on average.

    The formula is as follows, with z a standardized innovation of the chosen distribution:

    - GARCH: Alpha + Beta
    - IGARCH: 1
    - GJR-GARCH: Alpha + Gamma * E[z^2 1(z < 0)] + Beta, with E[z^2 1(z < 0)] = 1 / 2 for a symmetric distribution
    - EGARCH: Beta, the persistence of the log variance
    - APARCH, TGARCH and NGARCH: Alpha * E[(|z| - Gamma * z)^Delta] + Beta
    - AVGARCH and NAGARCH: Alpha * E[(|z - Shift| - Gamma * (z - Shift))^d] + Beta, with d = 1 and 2
    - Component GARCH: Rho, the persistence of the long-run component (the short-run component fades with
      Alpha + Beta, which is smaller)
    - FIGARCH: the sum of the ARCH(∞) weights of the untruncated model, 1 for D > 0 (a shock fades
      hyperbolically, never as a fixed factor per period) and Phi for D = 0, where FIGARCH is GARCH(1, 1)
      with Alpha = Phi - Beta
    - Realized GARCH: Beta + Gamma * Phi, the persistence of the log variance (Hansen, Huang and Shek, 2012)

    For more information, see the following papers:

    - Ding, Z., Granger, C.W.J. and Engle, R.F. (1993). "A Long Memory Property of Stock Market
    Returns and a New Model." Journal of Empirical Finance, 1(1), 83-106.
    - Hentschel, L. (1995). "All in the Family: Nesting Symmetric and Asymmetric GARCH Models."
    Journal of Financial Economics, 39(1), 71-104.

    Shares similar logic:
    - rugarch (R), persistence: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2529

    Args:
        variance_weights (Sequence[float]): The variance parameters of the model.
        method (str, optional): The model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch" (EGARCH),
            "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH), "avgarch"
            (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH), "figarch"
            (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        distribution_weights (Sequence[float], optional): Its Shape (and Skew). Defaults to ().

    Returns:
        float: The persistence, infinite where the expectation does not exist (a Student-t with
        a Shape at or below Delta).
    """
    w = variance_weights

    if method == "garch":
        return float(w[1] + w[2])
    if method == "igarch":
        return 1.0
    if method == "egarch":
        return float(w[3])
    if method == "component":
        return float(w[3])
    if method == "figarch":
        return 1.0 if w[2] > 0 else float(w[1])
    if method == "realized":
        return float(w[1] + w[2] * w[4])
    if method == "gjr":
        # E[z^2 1(z < 0)], the share of the unit variance that comes from negative shocks: 1 / 2 for a symmetric distribution.  # noqa: E501
        negative_share = (
            0.5
            if distribution in ("normal", "t", "ged")
            else get_standardized_expectation(
                np.where(EXPECTATION_GRID < 0, EXPECTATION_GRID**2, 0.0),
                distribution,
                distribution_weights,
            )
        )
        return float(w[1] + w[2] * negative_share + w[3])

    if method in ("aparch", "tgarch", "ngarch"):
        _, alpha, gamma, beta, power = _power_weights(method, w)
        shift = 0.0
    else:
        _, alpha, gamma, shift, beta, power = _family_weights(method, w)

    # A Student-t has moments below its Shape only.
    if (
        distribution in ("t", "skewt", "skewt_fernandez_steel")
        and power >= distribution_weights[0]
    ):
        return math.inf

    centred = EXPECTATION_GRID - shift
    kappa = get_standardized_expectation(
        (np.abs(centred) - gamma * centred) ** power, distribution, distribution_weights
    )

    return float(alpha * kappa + beta)


def _is_admissible(
    method: str, variance_weights, distribution: str, distribution_weights
) -> bool:
    """Whether the weights give a positive, stationary variance (integrated for IGARCH)."""
    w = variance_weights

    if method == "gjr" and w[1] + w[2] < 0:
        return False
    if method == "component":
        _, alpha, beta, rho, phi = w
        # Engle and Lee (1999): 0 < Alpha + Beta < Rho < 1 and 0 < Phi < Beta keep both components positive.  # noqa: E501
        return alpha + beta < rho < 1 and phi < beta
    if method in ("igarch", "egarch"):
        return True
    if method == "figarch":
        # Conrad and Haag (2006): the variance is non-negative for every history of squared residuals exactly when every ARCH(∞) weight is.  # noqa: E501
        return bool(np.all(_figarch_weights(w[1], w[2], w[3]) >= 0))

    return (
        get_garch_family_persistence(w, method, distribution, distribution_weights) < 1
    )


def _first_variance(method: str, variance_weights, backcast: float) -> float:
    """
    The conditional variance of the first observation: the backcast (see get_backcast)
    stands in for the squared residual and the variance before the sample, as in the arch
    package. The leverage term of GJR-GARCH enters at its expectation, Gamma / 2, and the
    shock terms of EGARCH at zero; a power model takes the backcast to the power Delta / 2
    for both the shock and the (powered) volatility, and the component model starts both
    components at Omega + Rho * Backcast.
    """
    w = variance_weights

    if method == "garch":
        return w[0] + (w[1] + w[2]) * backcast
    if method == "igarch":
        return w[0] + backcast
    if method == "gjr":
        return w[0] + (w[1] + w[2] / 2 + w[3]) * backcast
    if method == "egarch":
        return math.exp(min(max(w[0] + w[3] * math.log(backcast + 1e-12), -20.0), 20.0))
    if method == "component":
        return w[0] + w[3] * backcast
    if method == "figarch":
        return (
            w[0] / (1 - w[3])
            + float(np.sum(_figarch_weights(w[1], w[2], w[3]))) * backcast
        )

    if method in ("aparch", "tgarch", "ngarch"):
        omega, alpha, _, beta, power = _power_weights(method, w)
    else:
        omega, alpha, _, _, beta, power = _family_weights(method, w)

    return (omega + (alpha + beta) * backcast ** (power / 2)) ** (2 / power)


def _variance_step(method: str, variance_weights):
    """
    Returns the one-step variance update of a model, variance_t = step(variance_(t-1),
    residual_(t-1), long_run_(t-1)), which also returns the long-run component of the
    component model (its third argument passed through for the other models). It takes
    floats or np.ndarrays, so it serves the recursions that cannot be vectorised over time
    (the family GARCH models and GARCH-in-mean) and the simulation over many paths.
    """
    if method in HISTORY_METHODS:
        raise ValueError(
            f"method={method!r} has no one-step variance update: FIGARCH needs the history "
            "of the squared residuals and the realized GARCH a realized measure."
        )

    w = [float(weight) for weight in variance_weights]

    if method in ("garch", "igarch"):
        omega, alpha = w[0], w[1]
        beta = w[2] if method == "garch" else 1 - alpha

        def step(variance, residual, long_run):
            return omega + alpha * residual * residual + beta * variance, long_run

    elif method == "gjr":
        omega, alpha, gamma, beta = w

        def step(variance, residual, long_run):
            return (
                omega
                + (alpha + gamma * (residual < 0)) * residual * residual
                + beta * variance,
                long_run,
            )

    elif method == "egarch":
        omega, alpha, gamma, beta = w

        def step(variance, residual, long_run):
            z = residual / variance**0.5
            log_variance = (
                omega
                + beta * np.log(variance)
                + alpha * (abs(z) - EXPECTED_ABSOLUTE_NORMAL)
                + gamma * z
            )
            return np.exp(np.clip(log_variance, -20.0, 20.0)), long_run

    elif method == "component":
        omega, alpha, beta, rho, phi = w

        def step(variance, residual, long_run):
            squared = residual * residual
            new_long_run = omega + rho * long_run + phi * (squared - variance)
            return (
                new_long_run
                + alpha * (squared - long_run)
                + beta * (variance - long_run),
                new_long_run,
            )

    elif method in ("aparch", "tgarch", "ngarch"):
        omega, alpha, gamma, beta, power = _power_weights(method, w)

        def step(variance, residual, long_run):
            powered = (
                omega
                + alpha * (abs(residual) - gamma * residual) ** power
                + beta * variance ** (power / 2)
            )
            return powered ** (2 / power), long_run

    else:
        omega, alpha, gamma, shift, beta, power = _family_weights(method, w)

        def step(variance, residual, long_run):
            powered_volatility = variance ** (power / 2)
            centred = residual / variance**0.5 - shift
            shock = abs(centred) - gamma * centred
            powered = omega + powered_volatility * (alpha * shock**power + beta)
            return powered ** (2 / power), long_run

    return step


def _fractional_weights(d: float, size: int) -> np.ndarray:
    """
    The weights pi_0, ..., pi_(size-1) of the fractional difference (1 - L)^d = Σ pi_k L^k of
    Granger and Joyeux (1980) and Hosking (1981): pi_0 = 1 and pi_k = pi_(k-1) * (k - 1 - d) / k.
    """
    k = np.arange(1, size)
    return np.cumprod(np.concatenate(([1.0], (k - 1 - d) / k)))


def _figarch_weights(phi: float, d: float, beta: float) -> np.ndarray:
    """
    The ARCH(∞) weights lambda_1, ..., lambda_1000 of FIGARCH(1, d, 1) (Baillie, Bollerslev and
    Mikkelsen, 1996), sigma_t^2 = Omega / (1 - Beta) + Σ lambda_k e_(t-k)^2, from
    lambda(L) = 1 - (1 - Phi L)(1 - L)^d / (1 - Beta L):

    - c_k = pi_k - Phi * pi_(k-1), the coefficients of (1 - Phi L)(1 - L)^d (see _fractional_weights)
    - g_0 = 1 and g_k = c_k + Beta * g_(k-1), the coefficients of that polynomial over (1 - Beta L)
    - lambda_k = -g_k, so lambda_1 = D + Phi - Beta
    """
    from scipy import signal  # noqa: PLC0415

    pi = _fractional_weights(d, FIGARCH_TRUNCATION + 1)
    coefficients = pi[1:] - phi * pi[:-1]
    g, _ = signal.lfilter([1.0], [1.0, -beta], coefficients, zi=[beta])

    return -g


def _figarch_variance(
    variance_weights, residuals: np.ndarray, backcast: float
) -> np.ndarray:
    """
    The FIGARCH(1, d, 1) variance of every residual and one step past the last: the ARCH(∞)
    representation truncated at 1,000 lags (see _figarch_weights), with the backcast for every
    squared residual before the sample. The sum over the lags is a convolution, computed with
    the FFT (the 1,000 weights times every residual otherwise dominate the fit).
    """
    from scipy import signal  # noqa: PLC0415

    omega, phi, d, beta = (float(weight) for weight in variance_weights)
    weights = _figarch_weights(phi, d, beta)
    squared = np.concatenate(
        [np.full(FIGARCH_TRUNCATION, backcast), residuals * residuals]
    )
    # sigma2[t] = Omega / (1 - Beta) + Σ_(i=1..1000) lambda_i squared[1000 + t - i], the convolution at 999 + t.  # noqa: E501
    convolution = signal.fftconvolve(squared, weights)

    return (
        omega / (1 - beta)
        + convolution[FIGARCH_TRUNCATION - 1 : FIGARCH_TRUNCATION + len(residuals)]
    )


def _realized_log_variance(
    variance_weights,
    residuals: np.ndarray,
    log_measure: np.ndarray,
    log_backcast: tuple[float, float],
) -> np.ndarray:
    """
    The log variance of the realized GARCH(1, 1) of Hansen, Huang and Shek (2012) for every
    residual and one step past the last:

    - ln sigma_t^2 = Omega + Beta * ln sigma_(t-1)^2 + Gamma * ln x_(t-1)
    - ln sigma_1^2 = Omega + Beta * ln(Backcast) + Gamma * ln(Measure Backcast)

    A day without a realized measure (a missing or zero range) takes its expectation given the
    return of that day, ln x_t = Xi + Phi * ln sigma_t^2 + Tau 1 * z_t + Tau 2 * (z_t^2 - 1), which
    makes the recursion run step by step; with a measure on every day it is linear in ln x.
    """
    omega, beta, gamma, xi, phi, tau_1, tau_2, _ = (
        float(weight) for weight in variance_weights
    )
    first = omega + beta * log_backcast[0] + gamma * log_backcast[1]

    if np.all(np.isfinite(log_measure)):
        return _linear_variance_recursion(
            omega + gamma * log_measure, beta, first, len(residuals) + 1
        )

    values = [first]
    log_variance = first
    for residual, observed in zip(
        residuals.tolist(), log_measure.tolist(), strict=True
    ):
        measure = observed
        if not math.isfinite(observed):
            # Kept within the range of a float, where a search far outside the optimum takes it.
            z = residual / math.exp(min(max(log_variance, -1400.0), 1400.0) / 2)
            measure = xi + phi * log_variance + tau_1 * z + tau_2 * (z * z - 1)
        log_variance = omega + beta * log_variance + gamma * measure
        values.append(log_variance)

    return np.array(values)


def _component_variance(
    variance_weights, residuals: np.ndarray, backcast: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    The conditional variance and its long-run component of the component GARCH model of
    Engle and Lee (1999), for every residual and one step past the last.

    The two components follow a linear system in the squared residuals e_t:

    - sigma_t^2 = Omega + (Rho - Alpha - Beta) * q_(t-1) + (Alpha + Phi) * e_(t-1) + (Beta - Phi) * sigma_(t-1)^2
    - q_t = Omega + Rho * q_(t-1) + Phi * (e_(t-1) - sigma_(t-1)^2)

    By the Cayley-Hamilton theorem the variance then follows a second-order linear
    recursion, sigma_t^2 = T * sigma_(t-1)^2 - D * sigma_(t-2)^2 + d_t, with T and D the trace
    and determinant of the system matrix, which lfilter computes in compiled code; the
    long-run component follows from the variance as a first-order recursion.
    """
    from scipy import signal  # noqa: PLC0415

    omega, alpha, beta, rho, phi = (float(weight) for weight in variance_weights)
    squared = residuals * residuals
    steps = len(residuals) + 1

    sigma2 = np.empty(steps)
    sigma2[0] = omega + rho * backcast

    if steps > 1:
        sigma2[1] = (
            omega
            + (rho - alpha - beta) * sigma2[0]
            + (alpha + phi) * squared[0]
            + (beta - phi) * sigma2[0]
        )

    if steps > 2:  # noqa: PLR2004
        trace = beta - phi + rho
        determinant = (beta - phi) * rho + phi * (rho - alpha - beta)
        driver_variance = omega + (alpha + phi) * squared
        driver_long_run = omega + phi * squared
        driver = (
            driver_variance[1:]
            - rho * driver_variance[:-1]
            + (rho - alpha - beta) * driver_long_run[:-1]
        )
        denominator = [1.0, -trace, determinant]
        sigma2[2:], _ = signal.lfilter(
            [1.0],
            denominator,
            driver,
            zi=signal.lfiltic([1.0], denominator, [sigma2[1], sigma2[0]]),
        )

    long_run = _linear_variance_recursion(
        omega + phi * (squared - sigma2[:-1]), rho, sigma2[0], steps
    )

    return sigma2, long_run


def _starting_residuals(returns: np.ndarray, mean: str) -> np.ndarray:
    """
    The residuals of the mean model at its starting values, whose backcast starts the
    variance recursion throughout the fit (as in the arch package): the returns themselves
    for a zero mean, the returns less their average for a constant mean and GARCH-in-mean,
    and the residuals of the least-squares AR(1) for the ARMA(1, 1) and ARFIMA(1, d, 1) means.
    """
    if mean == "zero":
        return returns
    if mean in ("constant", "in_mean"):
        return returns - returns.mean()

    regressors = np.column_stack([np.ones(len(returns) - 1), returns[:-1]])
    coefficients = np.linalg.lstsq(regressors, returns[1:], rcond=None)[0]

    return returns[1:] - regressors @ coefficients


def get_measure_backcast(measure: np.ndarray) -> float:
    """
    Calculate the backcast of the realized measure that starts the realized GARCH recursion: the
    measure before the first day, the exponentially weighted average of the first 75 measures (decay
    0.94) on the days with a positive measure, as `get_backcast` averages the first squared returns.

    Args:
        measure (np.ndarray): The realized measure per day, such as the daily Parkinson
            variance.

    Returns:
        float: The backcast of the realized measure.
    """
    measure = np.asarray(measure, dtype=float)
    measure = measure[np.isfinite(measure) & (measure > 0)]
    window = min(BACKCAST_OBSERVATIONS, len(measure))
    decay_weights = BACKCAST_DECAY ** np.arange(window)

    return float(np.sum(measure[:window] * decay_weights / decay_weights.sum()))


def _log_measure(measure: np.ndarray | None) -> np.ndarray:
    """The log of the realized measure, NaN on a day without a positive one."""
    if measure is None:
        raise ValueError(
            "method='realized' needs a realized measure, such as the daily Parkinson "
            "variance of the High and Low prices."
        )

    measure = np.asarray(measure, dtype=float)
    positive = np.isfinite(measure) & (measure > 0)

    return np.where(positive, np.log(np.where(positive, measure, 1.0)), np.nan)


def _garch_family_filter(
    returns: np.ndarray,
    weights,
    method: str,
    mean: str,
    backcast: float,
    realized_measure: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """
    Runs the mean model and the variance recursion of a GARCH-family model over the returns.

    The realized GARCH takes the realized measure of every return (in the units of the
    squared returns); the other models ignore it.

    Returns:
        tuple: The residuals, the conditional variance of every residual and one step past
        the last, the long-run component of the component model (None for the others) and
        the number of returns the mean model holds back (1 for the ARMA(1, 1) and ARFIMA(1, d,
        1) means).
    """
    variance_weights, mean_weights, _ = _split_weights(weights, method, mean)

    if mean == "in_mean":
        return _in_mean_filter(
            returns, variance_weights, mean_weights, method, backcast
        )

    if mean == "zero":
        residuals, held_back = returns, 0
    elif mean == "constant":
        residuals, held_back = returns - mean_weights[0], 0
    elif mean == "arfima":
        from scipy import signal  # noqa: PLC0415

        mu, autoregressive, d, moving_average = mean_weights
        # w = (1 - L)^d (r - Mu) over the returns from the first, with no values before it (the conditional sum of squares of Hosking, 1981), and then e_t + MA * e_(t-1) = w_t - AR * w_(t-1) from t = 1 as for the ARMA mean.  # noqa: E501
        centred = returns - mu
        differenced = signal.fftconvolve(centred, _fractional_weights(d, len(centred)))[
            : len(centred)
        ]
        residuals = signal.lfilter(
            [1.0],
            [1.0, moving_average],
            differenced[1:] - autoregressive * differenced[:-1],
        )
        held_back = 1
    else:
        from scipy import signal  # noqa: PLC0415

        mu, autoregressive, moving_average = mean_weights
        # e_t + MA * e_(t-1) = r_t - Mu - AR * r_(t-1) from t = 1, the residual before it zero.
        residuals = signal.lfilter(
            [1.0],
            [1.0, moving_average],
            returns[1:] - mu - autoregressive * returns[:-1],
        )
        held_back = 1

    steps = len(residuals) + 1
    long_run = None

    if method in ("garch", "igarch"):
        garch_weights = (
            variance_weights
            if method == "garch"
            else [variance_weights[0], variance_weights[1], 1 - variance_weights[1]]
        )
        sigma2 = get_garch(residuals, garch_weights, steps, backcast=backcast)
    elif method == "gjr":
        sigma2 = get_gjr_garch(residuals, variance_weights, steps, backcast=backcast)
    elif method == "egarch":
        sigma2 = get_egarch(residuals, variance_weights, steps, backcast=backcast)
    elif method == "component":
        sigma2, long_run = _component_variance(variance_weights, residuals, backcast)
    elif method == "figarch":
        sigma2 = _figarch_variance(variance_weights, residuals, backcast)
    elif method == "realized":
        log_measure = _log_measure(realized_measure)
        with np.errstate(over="ignore"):
            sigma2 = np.exp(
                _realized_log_variance(
                    variance_weights,
                    residuals,
                    log_measure[held_back:],
                    (
                        math.log(max(backcast, 1e-300)),
                        math.log(max(get_measure_backcast(realized_measure), 1e-300)),
                    ),
                )
            )
    elif method in ("aparch", "tgarch", "ngarch"):
        omega, alpha, gamma, beta, power = _power_weights(method, variance_weights)
        # The power of the volatility follows a first-order linear recursion in the shocks, like the GARCH variance.  # noqa: E501
        shocks = (np.abs(residuals) - gamma * residuals) ** power
        powered = _linear_variance_recursion(
            omega + alpha * shocks,
            beta,
            _first_variance(method, variance_weights, backcast) ** (power / 2),
            steps,
        )
        sigma2 = powered ** (2 / power)
    else:
        # The shock of the family GARCH models is measured in units of the volatility itself, so the recursion is not linear and runs step by step on plain Python floats (see get_egarch).  # noqa: E501
        step = _variance_step(method, variance_weights)
        variance = _first_variance(method, variance_weights, backcast)
        values = [variance]
        for residual in residuals.tolist():
            variance, _ = step(variance, residual, None)
            values.append(variance)
        sigma2 = np.array(values)

    return residuals, sigma2, long_run, held_back


def _in_mean_filter(
    returns: np.ndarray, variance_weights, mean_weights, method: str, backcast: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """
    The GARCH-in-mean filter of Engle, Lilien and Robins (1987), r_t = Mu + Lambda * sigma_t
    + e_t: the residual of each return depends on its own conditional volatility, so the mean
    and the variance run step by step.
    """
    mu, risk_premium = (float(weight) for weight in mean_weights)
    step = _variance_step(method, variance_weights)
    variance = _first_variance(method, variance_weights, backcast)
    long_run = variance if method == "component" else None

    sigma2 = [variance]
    long_runs = [long_run]
    residuals = []

    for value in returns.tolist():
        residual = value - mu - risk_premium * variance**0.5
        residuals.append(residual)
        variance, long_run = step(variance, residual, long_run)
        sigma2.append(variance)
        long_runs.append(long_run)

    return (
        np.array(residuals),
        np.array(sigma2, dtype=float),
        np.array(long_runs, dtype=float) if method == "component" else None,
        0,
    )


# The models and means whose log-likelihood has an analytic gradient (see _garch_family_score): the variance recursions that are linear in the variance (GARCH, IGARCH and GJR-GARCH) or in a power of the volatility (APARCH, TGARCH and NGARCH), with a mean whose residuals follow a linear filter. The others (EGARCH, family GARCH, the component model, FIGARCH, the realized GARCH, GARCH-in-mean and the ARFIMA mean) are fitted with finite differences.  # noqa: E501
ANALYTIC_SCORE_METHODS = ("garch", "igarch", "gjr", "aparch", "tgarch", "ngarch")
ANALYTIC_SCORE_MEANS = ("zero", "constant", "arma")

# The relative step of the central differences of the log density of the distributions without an analytic derivative (see _log_density_derivatives): the truncation error (step squared) and the rounding error (machine precision over the step) are then both of the order of 1e-10.  # noqa: E501
DENSITY_DIFFERENCE_STEP = 1e-5


def _linear_recursion_derivatives(
    driver_derivatives: np.ndarray, first_derivatives, persistence: float
) -> np.ndarray:
    """
    The derivatives of the recursion x_t = d_t + Beta * x_(t-1) (see
    `_linear_variance_recursion`) with respect to every weight: dx_t = dd_t + Beta * dx_(t-1),
    the same first-order recursion, run by lfilter for all weights at once. The derivatives
    of the driver d_t include x_(t-1) for the weight Beta itself.

    Args:
        driver_derivatives (np.ndarray): The derivatives of the driver of steps 1 to T - 1,
            one column per weight.
        first_derivatives (Sequence[float]): The derivatives of x_0.
        persistence (float): The weight of the previous value (Beta).

    Returns:
        np.ndarray: The derivatives of x_0 to x_(T-1), one column per weight.
    """
    from scipy import signal  # noqa: PLC0415

    first = np.asarray(first_derivatives, dtype=float)
    derivatives = np.empty((len(driver_derivatives) + 1, len(first)))
    derivatives[0] = first

    if len(driver_derivatives):
        derivatives[1:], _ = signal.lfilter(
            [1.0],
            [1.0, -persistence],
            driver_derivatives,
            axis=0,
            zi=persistence * first[None, :],
        )

    return derivatives


def _mean_residual_derivatives(
    returns: np.ndarray, mean_weights, mean: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    The residuals of the zero, constant or ARMA(1, 1) mean (as `_garch_family_filter`
    computes them) and their derivatives with respect to the mean weights. The ARMA
    residuals e_t = r_t - Mu - AR * r_(t-1) - MA * e_(t-1) give the derivatives
    de_t = -(1, r_(t-1), e_(t-1)) - MA * de_(t-1), the filter of the residuals itself.
    """
    if mean == "zero":
        return returns, np.zeros((len(returns), 0))
    if mean == "constant":
        return returns - mean_weights[0], -np.ones((len(returns), 1))

    from scipy import signal  # noqa: PLC0415

    mu, autoregressive, moving_average = mean_weights
    lagged = returns[:-1]
    residuals = signal.lfilter(
        [1.0], [1.0, moving_average], returns[1:] - mu - autoregressive * lagged
    )
    previous = np.concatenate(([0.0], residuals[:-1]))
    derivatives = signal.lfilter(
        [1.0],
        [1.0, moving_average],
        -np.column_stack([np.ones(len(lagged)), lagged, previous]),
        axis=0,
    )

    return residuals, derivatives


def _log_density_derivatives(
    z: np.ndarray, distribution: str, parameters
) -> tuple[np.ndarray, np.ndarray]:
    """
    The derivatives of the log density of a standardized distribution (see
    `get_standardized_log_density`) with respect to z and to its Shape (and Skew):

    - normal: d ln f / dz = -z
    - t: d ln f / dz = -(v + 1) z / (v - 2 + z^2) and d ln f / dv = (ψ((v + 1) / 2) - ψ(v / 2)) / 2
      - 1 / (2 (v - 2)) - ln(1 + q) / 2 + (v + 1) q / (2 (v - 2) (1 + q)), with q = z^2 / (v - 2) and ψ the
      digamma function

    The skewed Student-t distributions, the GED and the Johnson SU take central differences of
    the log density, with a step of 1e-5 relative to the value.

    Returns:
        tuple[np.ndarray, np.ndarray]: The derivative with respect to z at every z, and with
        respect to the distribution parameters (one column each).
    """
    if distribution == "normal":
        return -z, np.zeros((len(z), 0))

    if distribution == "t":
        shape = float(parameters[0])
        ratio = z * z / (shape - 2)
        slope = -(shape + 1) * z / (shape - 2 + z * z)
        score = (
            0.5 * (special.digamma((shape + 1) / 2) - special.digamma(shape / 2))
            - 0.5 / (shape - 2)
            - 0.5 * np.log1p(ratio)
            + 0.5 * (shape + 1) * ratio / ((shape - 2) * (1 + ratio))
        )
        return slope, score[:, None]

    parameters = np.asarray(parameters, dtype=float)
    step = DENSITY_DIFFERENCE_STEP * np.maximum(1.0, np.abs(z))
    slope = (
        get_standardized_log_density(z + step, distribution, parameters)
        - get_standardized_log_density(z - step, distribution, parameters)
    ) / (2 * step)

    scores = []
    for position, value in enumerate(parameters):
        parameter_step = DENSITY_DIFFERENCE_STEP * max(1.0, abs(value))
        higher, lower = parameters.copy(), parameters.copy()
        higher[position] += parameter_step
        lower[position] -= parameter_step
        scores.append(
            (
                get_standardized_log_density(z, distribution, higher)
                - get_standardized_log_density(z, distribution, lower)
            )
            / (2 * parameter_step)
        )

    return slope, np.column_stack(scores)


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def _garch_family_score(
    returns: np.ndarray,
    weights,
    method: str,
    distribution: str,
    mean: str,
    backcast: float,
) -> np.ndarray:
    """
    Calculates the gradient (score) of the log-likelihood of a GARCH-family model (see
    `get_garch_family_log_likelihood`) with respect to its weights, for the models of
    ANALYTIC_SCORE_METHODS with the means of ANALYTIC_SCORE_MEANS.

    The log-likelihood of every observation is l_t = ln f(z_t) - ln(sigma_t^2) / 2 with
    z_t = e_t / sigma_t, so with g = d ln f / dz (see `_log_density_derivatives`):

    - dl_t = g(z_t) / sigma_t * de_t - (g(z_t) z_t + 1) / (2 sigma_t^2) * dsigma_t^2 + d ln f / dθ

    The variance of GARCH, IGARCH and GJR-GARCH is a first-order linear recursion in itself
    and that of APARCH, TGARCH and NGARCH in the power Delta of the volatility, p_t = sigma_t^Delta:

    - sigma_t^2 = Omega + (Alpha + Gamma 1(e_(t-1) < 0)) e_(t-1)^2 + Beta sigma_(t-1)^2
    - p_t = Omega + Alpha (|e_(t-1)| - Gamma e_(t-1))^Delta + Beta p_(t-1), sigma_t^2 = p_t^(2 / Delta)

    so their derivatives with respect to every weight follow the same recursion (see
    `_linear_recursion_derivatives`), with Beta's own derivative driven by the previous
    value and the first one by the backcast (see `_first_variance`). The mean weights enter
    through the residuals (see `_mean_residual_derivatives`) and the shocks they drive.
    The indicator of GJR-GARCH has a derivative of zero wherever it is defined.

    Args:
        returns (np.ndarray): The returns.
        weights (Sequence[float]): The weights, see `get_garch_family_parameter_names`.
        method (str): The variance model, one of ANALYTIC_SCORE_METHODS.
        distribution (str): The innovation distribution.
        mean (str): The mean model, one of ANALYTIC_SCORE_MEANS.
        backcast (float): The backcast that starts the recursion.

    Returns:
        np.ndarray: The derivative of the log-likelihood with respect to every weight.
    """
    variance_weights, mean_weights, distribution_weights = _split_weights(
        weights, method, mean
    )
    residuals, residual_derivatives = _mean_residual_derivatives(
        returns, mean_weights, mean
    )
    observations = len(residuals)
    lagged = residuals[:-1]
    lagged_derivatives = residual_derivatives[:-1]
    ones = np.ones(observations - 1)
    mean_first = [0.0] * residual_derivatives.shape[1]

    if method in ("garch", "igarch", "gjr"):
        omega, alpha = variance_weights[0], variance_weights[1]
        if method == "gjr":
            gamma, beta = variance_weights[2], variance_weights[3]
        else:
            gamma = 0.0
            beta = variance_weights[2] if method == "garch" else 1 - alpha

        squared = lagged * lagged
        negative = np.where(lagged < 0, 1.0, 0.0)
        coefficient = alpha + gamma * negative
        sigma2 = _linear_variance_recursion(
            omega + coefficient * squared,
            beta,
            _first_variance(method, variance_weights, backcast),
            observations,
        )
        previous = sigma2[:-1]

        if method == "garch":
            columns = [ones, squared, previous]
            first = [1.0, backcast, backcast]
        elif method == "igarch":
            # Beta is 1 - Alpha, so Alpha also takes the previous variance away.
            columns = [ones, squared - previous]
            first = [1.0, 0.0]
        else:
            columns = [ones, squared, negative * squared, previous]
            first = [1.0, backcast, backcast / 2, backcast]

        mean_columns = (2 * coefficient * lagged)[:, None] * lagged_derivatives
        derivatives = _linear_recursion_derivatives(
            np.column_stack([*columns, mean_columns]), first + mean_first, beta
        )
    else:
        omega, alpha, gamma, beta, power = _power_weights(method, variance_weights)
        base = np.abs(lagged) - gamma * lagged
        shocks = base**power
        positive = base > 0
        log_base = np.where(positive, np.log(np.where(positive, base, 1.0)), 0.0)
        # The derivative of the shock with respect to its base, zero at a residual of zero.
        slope = np.where(positive, power * shocks / np.where(positive, base, 1.0), 0.0)
        start = backcast ** (power / 2)
        powered = _linear_variance_recursion(
            omega + alpha * shocks, beta, omega + (alpha + beta) * start, observations
        )

        # The derivatives with respect to Omega, Alpha, Gamma, Beta and Delta, of which every model takes its own.  # noqa: E501
        columns = [
            ones,
            shocks,
            -alpha * slope * lagged,
            powered[:-1],
            alpha * shocks * log_base,
        ]
        first = [
            1.0,
            start,
            0.0,
            start,
            (alpha + beta) * start * math.log(backcast) / 2,
        ]
        kept = {
            "aparch": [0, 1, 2, 3, 4],
            "tgarch": [0, 1, 2, 3],
            "ngarch": [0, 1, 3, 4],
        }[method]
        mean_columns = (alpha * slope * (np.sign(lagged) - gamma))[
            :, None
        ] * lagged_derivatives
        power_derivatives = _linear_recursion_derivatives(
            np.column_stack([*(columns[index] for index in kept), mean_columns]),
            [first[index] for index in kept] + mean_first,
            beta,
        )

        sigma2 = powered ** (2 / power)
        derivatives = (2 / power) * (sigma2 / powered)[:, None] * power_derivatives
        if method != "tgarch":
            # sigma^2 = p^(2 / Delta) also depends on Delta directly, Delta being the last variance weight.  # noqa: E501
            derivatives[:, len(variance_weights) - 1] -= (
                2 / power**2 * sigma2 * np.log(powered)
            )

    z = residuals / np.sqrt(sigma2)
    density_slope, distribution_score = _log_density_derivatives(
        z, distribution, distribution_weights
    )

    gradient = (-0.5 * (density_slope * z + 1) / sigma2) @ derivatives
    gradient[len(variance_weights) :] += (
        density_slope / np.sqrt(sigma2)
    ) @ residual_derivatives

    return np.concatenate([gradient, distribution_score.sum(axis=0)])


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def get_garch_family_log_likelihood(
    returns: np.ndarray,
    weights,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    backcast: float | None = None,
    per_observation: bool = False,
    realized_measure: np.ndarray | None = None,
) -> float | np.ndarray:
    """
    Calculate the log-likelihood of a GARCH-family model: how well its weights explain the returns,
    the quantity the fit maximizes.

    The formula is as follows:

    - Log-Likelihood = Σ (ln f(e_t / sigma_t) - ln(sigma_t^2) / 2)

    Where e_t are the residuals of the mean model, sigma_t^2 the conditional variance and f
    the density of the standardized innovation distribution (see
    `get_standardized_log_density`). The normal log-likelihood includes its constant, so it
    equals the log-likelihood the arch package reports.

    The realized GARCH of Hansen, Huang and Shek (2012) adds the normal density of the
    measurement equation on every day with a realized measure x_t (the joint likelihood of
    the returns and the measure):

    - u_t = ln x_t - Xi - Phi * ln sigma_t^2 - Tau 1 * z_t - Tau 2 * (z_t^2 - 1)
    - Log-Likelihood += Σ (-(ln(2π) + 2 ln(Sigma U) + u_t^2 / Sigma U^2) / 2)

    Referenced against:
    - arch, GARCH, EGARCH, APARCH and ARCHInMean (with ZeroMean, ConstantMean and ARX): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
    - arch, FIGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.FIGARCH.html

    Args:
        returns (np.ndarray): The returns.
        weights (Sequence[float]): The weights, see `get_garch_family_parameter_names`.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        backcast (float | None, optional): The backcast that starts the recursion. Defaults
            to None, which uses the backcast of the starting residuals of the mean model.
        per_observation (bool, optional): Whether to return the log-likelihood of every
            observation instead of their sum. Defaults to False.
        realized_measure (np.ndarray | None, optional): The realized measure of every return,
            in the units of the squared returns, for the Realized GARCH. Defaults to None.

    Returns:
        float | np.ndarray: The log-likelihood, -inf for weights outside the model's region.
    """
    returns = np.asarray(returns, dtype=float)
    variance_weights, _, distribution_weights = _split_weights(weights, method, mean)

    if backcast is None:
        backcast = get_backcast(_starting_residuals(returns, mean))

    residuals, sigma2, _, held_back = _garch_family_filter(
        returns, weights, method, mean, backcast, realized_measure
    )
    sigma2 = sigma2[: len(residuals)]
    z = residuals / np.sqrt(sigma2)

    log_likelihood = get_standardized_log_density(
        z, distribution, distribution_weights
    ) - 0.5 * np.log(sigma2)

    if method == "realized":
        _, _, _, xi, phi, tau_1, tau_2, sigma_u = variance_weights
        log_measure = _log_measure(realized_measure)[held_back:]
        measurement_error = (
            log_measure - xi - phi * np.log(sigma2) - tau_1 * z - tau_2 * (z * z - 1)
        )
        log_likelihood = log_likelihood + np.where(
            np.isfinite(log_measure),
            -0.5
            * (LOG_2_PI + 2 * math.log(sigma_u) + (measurement_error / sigma_u) ** 2),
            0.0,
        )

    if per_observation:
        return log_likelihood

    total = float(np.sum(log_likelihood))

    return total if np.isfinite(total) else -math.inf


def _scale_exponents(method: str, mean: str, weights) -> np.ndarray:
    """
    The power of the scale of the returns in every weight: returns divided by c give
    weights with Omega divided by c^2 (c^Delta for APARCH and NGARCH, c for TGARCH and
    AVGARCH) and Mu divided by c, while the other weights are unchanged. EGARCH's Omega
    shifts instead (see get_garch_family_weights).
    """
    variance_names = GARCH_FAMILY_VARIANCE_PARAMETERS[method]
    exponents = np.zeros(len(weights))

    if method not in ("egarch", "realized"):
        exponents[0] = (
            weights[variance_names.index("Delta")]
            if "Delta" in variance_names
            else POWER_OF_VOLATILITY.get(method, 2.0)
        )
    if mean != "zero":
        exponents[len(variance_names)] = 1.0

    return exponents


def _rescale_weights(weights, method: str, mean: str, scale: float) -> np.ndarray:
    """
    Converts the weights fitted on the returns divided by `scale` to the weights of the
    returns themselves (pass 1 / scale for the reverse).
    """
    weights = np.array(weights, dtype=float)
    rescaled = weights * scale ** _scale_exponents(method, mean, weights)

    if method == "egarch":
        # ln sigma_t^2 of the returns is that of the scaled returns plus 2 ln(scale), so Omega gains 2 (1 - Beta) ln(scale).  # noqa: E501
        rescaled[0] = weights[0] + 2 * (1 - weights[3]) * math.log(scale)
    elif method == "realized":
        # ln sigma_t^2 and ln x_t both gain 2 ln(scale), so Omega gains 2 (1 - Beta - Gamma) ln(scale) and Xi 2 (1 - Phi) ln(scale).  # noqa: E501
        rescaled[0] = weights[0] + 2 * (1 - weights[1] - weights[2]) * math.log(scale)
        rescaled[3] = weights[3] + 2 * (1 - weights[4]) * math.log(scale)

    return rescaled


def get_garch_family_weights(
    returns: np.ndarray,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    t: int | None = None,
    starting_weights: np.ndarray | None = None,
    realized_measure: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the weights of a GARCH-family model by maximum likelihood: the parameters of the variance,
    the mean and the innovation distribution that explain the returns best.

    The search is the same deterministic multi-start search as that of GARCH(1, 1) (see
    `_fit_multi_start`).

    With `starting_weights` (the weights of an earlier fit, such as the previous window of
    a rolling forecast) the search is a local search from them: the likelihood of a window
    that shares most of its returns with the previous one peaks close to the previous
    weights, and one search takes a fraction of the 33 of the multi-start search. A local
    search can stay on a bound the previous optimum sat on, or stop short along it, so a fit
    that starts or ends on a bound (or does not converge) also runs the multi-start search
    of a fit without starting weights and keeps the higher likelihood. A fall in the
    likelihood from one window to the next is no sign of a missed optimum: the windows
    differ by their returns, and a local optimum the search stays in can have a likelihood
    as high as the previous window's.

    The weights of the latest fits are kept by the returns and the settings (see
    GARCH_FAMILY_WEIGHTS_CACHE), so the variance, forecast and parameters of one series fit
    the model once.

    GARCH, GJR-GARCH and EGARCH with normal innovations and a zero mean are fitted by
    `get_garch_weights`, `get_gjr_garch_weights` and `get_egarch_weights`. Every other
    combination is fitted on the returns divided by their standard deviation, which puts
    every weight on a scale of about 1 (Omega of a daily return series would otherwise be
    of the order of 1e-6), and the weights are converted back: Omega is multiplied by the
    scale to the power of the model (2, Delta or 1), Mu by the scale, and EGARCH's Omega
    gains 2 (1 - Beta) ln(scale). The log-likelihood of the scaled returns differs by a
    constant only, so the maximum is the same.

    The constraints are those that keep the variance positive and stationary: a persistence
    below 1 (see `get_garch_family_persistence`; Alpha + Beta = 1 for IGARCH by construction),
    |Beta| < 1 for EGARCH, Alpha + Gamma >= 0 for GJR-GARCH, |Gamma| < 1 for the rotation of
    APARCH and family GARCH, 0 < Alpha + Beta < Rho < 1 and Phi < Beta for the component
    model, non-negative ARCH(∞) weights for FIGARCH (Conrad and Haag, 2006) and Beta + Gamma *
    Phi < 1 for the realized GARCH. The realized measure is divided by the square of the
    scale, and the Omega and Xi of the realized GARCH shift with its logarithm (see
    `_rescale_weights`); its search starts from Xi at the average log measure and Omega at
    -Gamma * Xi, where the log variance of the scaled returns is about 0.

    Referenced against:
    - arch, APARCH and GARCH (with other distributions and means): https://bashtage.github.io/arch/univariate/generated/arch.univariate.APARCH.html
    - arch, FIGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.FIGARCH.html

    Shares similar logic:
    - rugarch (R), ugarchspec (sGARCH, iGARCH, apARCH, fGARCH, csGARCH, realGARCH, ARFIMA mean): https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L21
    - rugarch (R), ugarchfit: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1059

    Args:
        returns (np.ndarray): The returns.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        t (int | None, optional): The number of returns, from the first, to fit on.
            Defaults to None, which uses every return.
        starting_weights (np.ndarray | None, optional): The weights to start a local
            search from, on the scale of the returns. Defaults to None, which runs the
            multi-start search.
        realized_measure (np.ndarray | None, optional): The realized measure of every return,
            in the units of the squared returns, which the Realized GARCH needs. Defaults to
            None.

    Returns:
        np.ndarray: The weights, see `get_garch_family_parameter_names`; NaN when the fit
        does not converge from any starting point.
    """
    validate_garch_family(method, distribution, mean)

    if method == "realized":
        _log_measure(realized_measure)
        realized_measure = np.asarray(realized_measure, dtype=float)
        if t is not None:
            realized_measure = realized_measure[:t]

    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0].to_numpy()

    returns = np.asarray(returns, dtype=float)
    warm_start = starting_weights is not None and bool(
        np.all(np.isfinite(starting_weights))
    )
    key = (
        *_series_key(returns),
        t,
        method,
        distribution,
        mean,
        tuple(np.round(starting_weights, 12)) if warm_start else None,
        _series_key(realized_measure) if method == "realized" else None,
    )

    return _cached_weights(
        GARCH_FAMILY_WEIGHTS_CACHE,
        GARCH_FAMILY_WEIGHTS_CACHE_LOCK,
        key,
        lambda: _fit_garch_family_weights(
            returns, method, distribution, mean, t, starting_weights, realized_measure
        ),
    )


def _fit_garch_family_weights(
    returns: np.ndarray,
    method: str,
    distribution: str,
    mean: str,
    t: int | None,
    starting_weights: np.ndarray | None,
    realized_measure: np.ndarray | None,
) -> np.ndarray:
    """Fits the weights of `get_garch_family_weights`, which keeps them in its cache."""
    warm_start = starting_weights is not None and bool(
        np.all(np.isfinite(starting_weights))
    )

    if distribution == "normal" and mean == "zero" and not warm_start:
        if method == "garch":
            return get_garch_weights(returns, t)
        if method == "gjr":
            return get_gjr_garch_weights(returns, t)
        if method == "egarch":
            return get_egarch_weights(returns, t)

    returns = returns[:t] if t is not None else returns
    scale = float(np.std(returns))
    scale = scale if scale > 0 else 1.0
    scaled = returns / scale
    scaled_measure = (
        realized_measure / scale**2
        if method == "realized" and realized_measure is not None
        else None
    )
    backcast = get_backcast(_starting_residuals(scaled, mean))

    variance_bounds, variance_guess = VARIANCE_BOUNDS[method]

    if method == "realized":
        xi = float(np.clip(np.nanmean(_log_measure(scaled_measure)), -9, 9))
        variance_guess = [
            -variance_guess[2] * xi,
            *variance_guess[1:3],
            xi,
            *variance_guess[4:],
        ]

    distribution_bounds, distribution_guess = DISTRIBUTION_BOUNDS[distribution]
    mean_bounds, mean_guess = {
        "zero": ([], []),
        "constant": ([(-1, 1)], [float(scaled.mean())]),
        "arma": (
            [(-1, 1), (-0.999, 0.999), (-0.999, 0.999)],
            [float(scaled.mean()), 0, 0],
        ),
        "in_mean": ([(-1, 1), (-1, 1)], [float(scaled.mean()), 0.0]),
        # The fractional difference of a stationary, invertible ARFIMA stays within (-0.5, 0.5).
        "arfima": (
            [(-1, 1), (-0.999, 0.999), (-0.499, 0.499), (-0.999, 0.999)],
            [float(scaled.mean()), 0, 0, 0],
        ),
    }[mean]

    bounds = variance_bounds + mean_bounds + distribution_bounds
    initial_guess = variance_guess + mean_guess + distribution_guess

    if warm_start:
        # The starting weights are on the scale of the returns; the search runs on the scaled returns, kept inside the bounds.  # noqa: E501
        initial_guess = list(
            np.clip(
                _rescale_weights(starting_weights, method, mean, 1 / scale),
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            )
        )

    def objective(weights):
        variance_weights, _, distribution_weights = _split_weights(
            weights, method, mean
        )
        if not _is_admissible(
            method, variance_weights, distribution, distribution_weights
        ):
            return np.inf
        log_likelihood = get_garch_family_log_likelihood(
            scaled,
            weights,
            method,
            distribution,
            mean,
            backcast,
            realized_measure=scaled_measure,
        )
        return -log_likelihood if np.isfinite(log_likelihood) else np.inf

    def gradient(weights):
        return -_garch_family_score(
            scaled, weights, method, distribution, mean, backcast
        )

    gradient = (
        gradient
        if method in ANALYTIC_SCORE_METHODS and mean in ANALYTIC_SCORE_MEANS
        else None
    )
    model_name = f"{method} ({distribution}, {mean} mean)"

    if warm_start:
        # A local search from weights on a bound (Alpha or Beta = 0 when the previous window had little volatility clustering) can stay in that corner, or stop short along the bound, while the new window peaks elsewhere. A fit that starts or ends on a bound therefore also runs the multi-start search of a fit without starting weights (the warm weights as a start of their own can displace the one that polishes to the highest likelihood) and keeps the higher likelihood.  # noqa: E501
        lower = np.array([bound[0] for bound in bounds])
        span = np.array([bound[1] for bound in bounds]) - lower
        weights = _fit_multi_start(
            objective,
            bounds,
            initial_guess,
            model_name,
            number_of_points=0,
            gradient=gradient,
        )
        value = objective(weights) if np.all(np.isfinite(weights)) else np.inf
        on_bound = [
            np.any(np.minimum(unit, 1 - unit) < BOUND_DISTANCE)
            for unit in (
                (np.asarray(initial_guess) - lower) / span,
                (weights - lower) / span,
            )
        ]
        if not np.isfinite(value) or any(on_bound):
            multi_start = _fit_multi_start(
                objective,
                bounds,
                variance_guess + mean_guess + distribution_guess,
                model_name,
                gradient=gradient,
            )
            if np.all(np.isfinite(multi_start)) and objective(multi_start) < value:
                weights = multi_start
    else:
        weights = _fit_multi_start(
            objective, bounds, initial_guess, model_name, gradient=gradient
        )

    return _rescale_weights(weights, method, mean, scale)


def _garch_family_path(
    returns: np.ndarray,
    weights,
    method: str,
    distribution: str,
    mean: str,
    output: str,
    time_steps: int | None,
    realized_measure: np.ndarray | None = None,
) -> np.ndarray:
    """
    The conditional variance (or the standardized residuals or the long-run component) of
    every return of one series, NaN for a return the mean model holds back.
    """
    backcast = get_backcast(_starting_residuals(returns, mean))
    residuals, sigma2, long_run, held_back = _garch_family_filter(
        returns, weights, method, mean, backcast, realized_measure
    )

    if output == "variance":
        values = sigma2[: len(residuals)]
    elif output == "standardized_residuals":
        values = residuals / np.sqrt(sigma2[: len(residuals)])
    elif long_run is None:
        raise InvalidArgumentError(
            "The long-run variance is only defined for method='component'."
        )
    else:
        values = long_run[: len(residuals)]

    path = np.concatenate([np.full(held_back, np.nan), values])

    return path[:time_steps] if time_steps is not None else path


def get_garch_family(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    output: str = "variance",
    time_steps: int | None = None,
    optimization_t: int | None = None,
    realized_measure: np.ndarray | pd.Series | pd.DataFrame | None = None,
) -> np.ndarray | pd.DataFrame:
    """
    Calculate the conditional variance of a GARCH-family model: how volatile each period is expected
    to be given the shocks before it, or the standardized residuals or (for the component model) the
    long-run component, with every model of `get_garch_family_weights`.

    Every series is fitted on its own returns (see `_apply_per_column`) and a 'within period'
    index fits every period separately, as in `get_garch`.

    Referenced against:
    - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
    - arch, GARCH (GJR-GARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html
    - arch, EGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html
    - arch, APARCH (APARCH, TGARCH, NGARCH): https://bashtage.github.io/arch/univariate/generated/arch.univariate.APARCH.html
    - arch, FIGARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.FIGARCH.html
    - arch, ARCHInMean (GARCH-in-mean): https://bashtage.github.io/arch/univariate/generated/arch.univariate.ARCHInMean.html

    Shares similar logic:
    - rugarch (R), ugarchspec (sGARCH, iGARCH, apARCH, fGARCH, csGARCH, realGARCH, ARFIMA mean): https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L21
    - rugarch (R), ugarchfit: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1059

    Args:
        returns (np.ndarray | pd.Series | pd.DataFrame): The returns.
        weights (np.ndarray | list | None, optional): The weights, see
            `get_garch_family_parameter_names`, used for every column. Defaults to None, which
            fits them per column.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        output (str, optional): What to return: the conditional variance, the standardized residuals e_t / sigma_t or
            the long-run component q_t of the component GARCH. Options are: "variance", "standardized_residuals",
            "long_run_variance" (component GARCH only). Defaults to "variance".
        time_steps (int | None, optional): The number of returns, from the first, to return.
            Defaults to None, which returns every return.
        optimization_t (int | None, optional): The number of returns, from the first, to fit on.
            Defaults to None, which uses every return.
        realized_measure (np.ndarray | pd.Series | pd.DataFrame | None, optional): The realized
            measure of every return (a frame with the columns of the returns), in the units of
            the squared returns, which the Realized GARCH needs. Defaults to None.

    Returns:
        np.ndarray | pd.DataFrame: The values, NaN for a return the mean model holds back
        (the first, for the ARMA(1, 1) and ARFIMA(1, d, 1) means).
    """
    validate_garch_family(method, distribution, mean)

    check_option(
        "output", output, ("variance", "standardized_residuals", "long_run_variance")
    )
    if output == "long_run_variance" and method != "component":
        raise InvalidArgumentError(
            "Only method='component' has a long-run variance component."
        )

    arguments = {
        "weights": weights,
        "method": method,
        "distribution": distribution,
        "mean": mean,
        "output": output,
        "time_steps": time_steps,
        "optimization_t": optimization_t,
        "realized_measure": realized_measure,
    }

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = [
                _apply_per_column(
                    returns.loc[sub_period], get_garch_family, **arguments
                )
                for sub_period in returns.index.get_level_values(0).unique()
            ]
            return pd.concat(
                [period for period in periods if not period.dropna(how="all").empty],
                axis=0,
            )
        return _apply_per_column(returns, get_garch_family, **arguments)

    if isinstance(returns, pd.Series):
        if isinstance(realized_measure, pd.Series):
            arguments["realized_measure"] = realized_measure.reindex(
                returns.index
            ).to_numpy(dtype=float)
        return get_garch_family(returns.to_numpy(), **arguments)

    returns = np.asarray(returns, dtype=float)

    if weights is None:
        weights = get_garch_family_weights(
            returns,
            method,
            distribution,
            mean,
            optimization_t,
            realized_measure=realized_measure,
        )

    return _garch_family_path(
        returns,
        weights,
        method,
        distribution,
        mean,
        output,
        time_steps,
        realized_measure,
    )


def get_garch_family_forecast(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    forecast_steps: int = 10,
    realized_measure: np.ndarray | pd.Series | pd.DataFrame | None = None,
) -> np.ndarray | pd.DataFrame:
    """
    Calculate the variance forecast of a GARCH-family model: the conditional variance expected for
    each of h = 1, ..., forecast_steps periods after the last return.

    The first value is the one-step-ahead variance of the fitted recursion, which consumes the
    last return. Further steps take the expectation of the shock terms over the innovation
    distribution, which makes the recursion linear in the variance (or its power):

    - GARCH, GJR-GARCH, IGARCH, APARCH, TGARCH, NGARCH, AVGARCH and NAGARCH: s_(n+h) = Omega + P * s_(n+h-1), with
      s = sigma^d the power of the model (2 for the variance models, Delta, or 1 for TGARCH and AVGARCH) and P
      the persistence (see `get_garch_family_persistence`), and sigma_(n+h)^2 = s_(n+h)^(2 / d)
    - Component GARCH: q_(n+h) = Omega + Rho * q_(n+h-1) and
      sigma_(n+h)^2 - q_(n+h) = (Alpha + Beta) * (sigma_(n+h-1)^2 - q_(n+h-1))
    - EGARCH: ln sigma_(n+h)^2 = Omega + Alpha * (E|z| - √(2 / π)) + Beta * ln sigma_(n+h-1)^2, and
      sigma_(n+h)^2 = exp(ln sigma_(n+h)^2)
    - FIGARCH: sigma_(n+h)^2 = Omega / (1 - Beta) + Σ_(i=1..1000) lambda_i E[e_(n+h-i)^2], with the squared
      residuals for the returns, their backcast before the first and the forecast variance
      sigma_(n+j)^2 for the periods after the last return
    - Realized GARCH: ln sigma_(n+h)^2 = Omega + Gamma * Xi + (Beta + Gamma * Phi) * ln sigma_(n+h-1)^2, the
      expected log measure Xi + Phi * ln sigma_t^2 in place of ln x_t (E[Tau(z)] = 0), and
      sigma_(n+h)^2 = exp(ln sigma_(n+h)^2)

    The forecast of a power other than 2, and of EGARCH and the realized GARCH, is that of
    the power (or the log) of the variance, transformed back, which understates the expected
    variance somewhat (Jensen's inequality), as the forecasts of `get_egarch_forecast` do. For IGARCH (P = 1)
    the variance grows by Omega every step. GARCH, GJR-GARCH and EGARCH with normal
    innovations and a zero mean are forecast by `get_garch_forecast`,
    `get_gjr_garch_forecast` and `get_egarch_forecast`.

    Referenced against:
    - arch, GARCH (GJR-GARCH forecast): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Shares similar logic:
    - rugarch (R), ugarchforecast: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1071

    Args:
        returns (np.ndarray | pd.Series | pd.DataFrame): The returns.
        weights (np.ndarray | list | None, optional): The weights, see
            `get_garch_family_parameter_names`. Defaults to None, which fits them per column.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        forecast_steps (int, optional): The number of periods to forecast. Defaults to 10.
        realized_measure (np.ndarray | pd.Series | pd.DataFrame | None, optional): The realized
            measure of every return, which the Realized GARCH needs. Defaults to None.

    Returns:
        np.ndarray | pd.DataFrame: The forecasted variance, one row per step.
    """
    validate_garch_family(method, distribution, mean)

    arguments = {
        "weights": weights,
        "method": method,
        "distribution": distribution,
        "mean": mean,
        "forecast_steps": forecast_steps,
        "realized_measure": realized_measure,
    }

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            period_data_list = []
            for sub_period in returns.index.get_level_values(0).unique():
                period_data = _apply_per_column(
                    returns.loc[sub_period],
                    get_garch_family_forecast,
                    forecast=True,
                    **arguments,
                )
                period_data.columns = [
                    f"{column} {sub_period}" for column in period_data.columns
                ]
                if not period_data.dropna(how="all").empty:
                    period_data_list.append(period_data)
            return pd.concat(period_data_list, axis=1)
        return _apply_per_column(
            returns, get_garch_family_forecast, forecast=True, **arguments
        )

    if isinstance(returns, pd.Series):
        if isinstance(realized_measure, pd.Series):
            arguments["realized_measure"] = realized_measure.reindex(
                returns.index
            ).to_numpy(dtype=float)
        return get_garch_family_forecast(returns.to_numpy(), **arguments)

    returns = np.asarray(returns, dtype=float)

    if weights is None:
        weights = get_garch_family_weights(
            returns, method, distribution, mean, realized_measure=realized_measure
        )

    if (
        distribution == "normal"
        and mean == "zero"
        and method in ("garch", "gjr", "egarch")
    ):
        forecast = {
            "garch": get_garch_forecast,
            "gjr": get_gjr_garch_forecast,
            "egarch": get_egarch_forecast,
        }[method]
        return forecast(returns, weights, forecast_steps)

    variance_weights, _, distribution_weights = _split_weights(weights, method, mean)
    backcast = get_backcast(_starting_residuals(returns, mean))
    residuals, sigma2, long_run, _ = _garch_family_filter(
        returns, weights, method, mean, backcast, realized_measure
    )

    forecast = np.empty(forecast_steps)

    if method == "figarch":
        omega, phi, d, beta = variance_weights
        arch_weights = _figarch_weights(phi, d, beta)
        # The squared residuals, latest first, with the backcast before the sample; every forecast variance joins them as the expected squared residual of its period.  # noqa: E501
        history = np.concatenate(
            [np.full(FIGARCH_TRUNCATION, backcast), residuals * residuals]
        )[::-1][:FIGARCH_TRUNCATION]
        variance = sigma2[-1]
        for step in range(forecast_steps):
            forecast[step] = variance
            history = np.concatenate([[variance], history[:-1]])
            variance = omega / (1 - beta) + float(arch_weights @ history)
        return forecast

    if method == "realized":
        omega, beta, gamma, xi, phi = variance_weights[:5]
        log_variance = math.log(sigma2[-1])
        for step in range(forecast_steps):
            forecast[step] = math.exp(log_variance)
            log_variance = omega + gamma * xi + (beta + gamma * phi) * log_variance
        return forecast

    if method == "component" and long_run is not None:
        _, alpha, beta, rho, _ = variance_weights
        level, short_run = long_run[-1], sigma2[-1] - long_run[-1]
        for step in range(forecast_steps):
            forecast[step] = level + short_run
            level = variance_weights[0] + rho * level
            short_run = (alpha + beta) * short_run
        return forecast

    if method == "egarch":
        omega, alpha, _, beta = variance_weights
        expected_absolute = get_standardized_expectation(
            np.abs(EXPECTATION_GRID), distribution, distribution_weights
        )
        log_variance = math.log(sigma2[-1])
        for step in range(forecast_steps):
            forecast[step] = math.exp(log_variance)
            log_variance = (
                omega
                + alpha * (expected_absolute - EXPECTED_ABSOLUTE_NORMAL)
                + beta * log_variance
            )
        return forecast

    power = (
        variance_weights[GARCH_FAMILY_VARIANCE_PARAMETERS[method].index("Delta")]
        if method in ("aparch", "ngarch")
        else POWER_OF_VOLATILITY.get(method, 2.0)
    )
    persistence = get_garch_family_persistence(
        variance_weights, method, distribution, distribution_weights
    )
    powered = sigma2[-1] ** (power / 2)
    for step in range(forecast_steps):
        forecast[step] = powered ** (2 / power)
        powered = variance_weights[0] + persistence * powered

    return forecast


def get_garch_family_unconditional_variance(
    variance_weights,
    method: str = "garch",
    distribution: str = "normal",
    distribution_weights=(),
) -> float:
    """
    Calculate the unconditional (long-run) variance of a GARCH-family model: the level the conditional
    variance reverts to after a shock.

    The formula is as follows:

    - sigma^2 = (Omega / (1 - P))^(2 / d), with P the persistence (see
      `get_garch_family_persistence`) and d the power of the model (2, Delta, or 1 for TGARCH and AVGARCH)
    - Component GARCH: Omega / (1 - Rho)
    - EGARCH: exp(Omega / (1 - Beta)), the exponential of the long-run log variance, which
      understates the expected variance somewhat (Jensen's inequality)
    - Realized GARCH: exp((Omega + Gamma * Xi) / (1 - Beta - Gamma * Phi)), likewise
    - FIGARCH: Omega / (1 - Phi) for D = 0 (GARCH(1, 1)); none for D > 0, where the model is not
      covariance stationary (Baillie, Bollerslev and Mikkelsen, 1996)

    Shares similar logic:
    - rugarch (R), uncvariance: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L3036

    Args:
        variance_weights (Sequence[float]): The variance parameters.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        distribution_weights (Sequence[float], optional): Its Shape (and Skew). Defaults to ().

    Returns:
        float: The unconditional variance, NaN for IGARCH (P = 1) and FIGARCH with D > 0,
        which have none.
    """
    w = variance_weights

    if method == "egarch":
        return math.exp(w[0] / (1 - w[3]))
    if method == "realized":
        persistence = w[1] + w[2] * w[4]
        return (
            math.exp((w[0] + w[2] * w[3]) / (1 - persistence))
            if persistence < 1
            else math.nan
        )
    if method == "figarch":
        return w[0] / (1 - w[1]) if w[2] == 0 and w[1] < 1 else math.nan

    persistence = get_garch_family_persistence(
        w, method, distribution, distribution_weights
    )

    if not persistence < 1:
        return math.nan

    if method == "component":
        return w[0] / (1 - w[3])

    power = (
        w[GARCH_FAMILY_VARIANCE_PARAMETERS[method].index("Delta")]
        if method in ("aparch", "ngarch")
        else POWER_OF_VOLATILITY.get(method, 2.0)
    )

    return (w[0] / (1 - persistence)) ** (2 / power)


def _numerical_hessian(function, point: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """The Hessian of a scalar function by central differences."""
    dimension = len(point)
    hessian = np.empty((dimension, dimension))

    for i in range(dimension):
        for j in range(i, dimension):
            values = []
            for sign_i, sign_j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                shifted = point.copy()
                shifted[i] += sign_i * steps[i]
                shifted[j] += sign_j * steps[j]
                values.append(function(shifted))
            hessian[i, j] = hessian[j, i] = (
                values[0] - values[1] - values[2] + values[3]
            ) / (4 * steps[i] * steps[j])

    return hessian


def _score_terms(
    returns: np.ndarray,
    weights,
    method: str,
    distribution: str,
    mean: str,
    realized_measure: np.ndarray | None,
):
    """
    The per-observation log-likelihood of the returns scaled to a unit Variance (where every
    weight is of the order of 1) as a function of the scaled weights, the scaled weights, the
    finite-difference steps, the scores of every observation by central differences and the
    Jacobian of the weights of the returns with respect to the scaled weights (see
    `_rescale_weights`).
    """
    returns = np.asarray(returns, dtype=float)
    scale = float(np.std(returns))
    scale = scale if scale > 0 else 1.0
    scaled = returns / scale
    scaled_measure = (
        np.asarray(realized_measure, dtype=float) / scale**2
        if realized_measure is not None
        else None
    )
    scaled_weights = _rescale_weights(weights, method, mean, 1 / scale)
    backcast = get_backcast(_starting_residuals(scaled, mean))
    steps = 1e-4 * np.maximum(np.abs(scaled_weights), 0.1)
    estimated = len(scaled_weights)

    def per_observation(point):
        return get_garch_family_log_likelihood(
            scaled,
            point,
            method,
            distribution,
            mean,
            backcast,
            per_observation=True,
            realized_measure=scaled_measure,
        )

    unit = np.eye(estimated)
    scores = np.column_stack(
        [
            (
                per_observation(scaled_weights + unit[i] * steps[i])
                - per_observation(scaled_weights - unit[i] * steps[i])
            )
            / (2 * steps[i])
            for i in range(estimated)
        ]
    )
    jacobian = np.column_stack(
        [
            (
                _rescale_weights(
                    scaled_weights + unit[i] * steps[i], method, mean, scale
                )
                - _rescale_weights(
                    scaled_weights - unit[i] * steps[i], method, mean, scale
                )
            )
            / (2 * steps[i])
            for i in range(estimated)
        ]
    )

    return per_observation, scaled_weights, steps, scores, jacobian


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def get_garch_family_scores(
    returns: np.ndarray,
    weights,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    realized_measure: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the score of every observation of a GARCH-family model: how the log-likelihood of each
    observation (see `get_garch_family_log_likelihood`) changes with every weight.

    The formula is as follows:

    - s_t,j = d ln l_t / d w_j

    The derivatives are central finite differences on returns scaled to a unit Variance,
    converted to the weights of the returns with the inverse of the Jacobian of
    `_rescale_weights` (the chain rule), as the robust standard errors of
    `get_garch_family_diagnostics` take them. At the maximum-likelihood weights the scores of
    a weight inside its bounds sum to about 0.

    Args:
        returns (np.ndarray): The returns.
        weights (Sequence[float]): The weights, see `get_garch_family_parameter_names`.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        realized_measure (np.ndarray | None, optional): The realized measure of every return,
            which the Realized GARCH needs. Defaults to None.

    Returns:
        np.ndarray: The scores, one row per observation in the likelihood and one column per
        weight.
    """
    _, _, _, scores, jacobian = _score_terms(
        returns, weights, method, distribution, mean, realized_measure
    )

    return scores @ np.linalg.inv(jacobian)


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def get_garch_family_diagnostics(
    returns: np.ndarray,
    weights,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    realized_measure: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calculate the fit diagnostics of a GARCH-family model: how persistent the volatility is and how
    well and how precisely the model fits, its persistence, half-life and unconditional variance, the
    log-likelihood and information criteria and the standard errors of the weights.

    The formula is as follows, with LL the log-likelihood, k the number of estimated
    weights and n the number of returns in the likelihood:

    - Persistence = P, see `get_garch_family_persistence`
    - Half-Life = ln(0.5) / ln(P), in periods
    - Unconditional Variance, see `get_garch_family_unconditional_variance`
    - AIC = (-2 LL + 2 k) / n
    - BIC = (-2 LL + k ln(n)) / n
    - Shibata = -2 LL / n + ln((n + 2 k) / n)
    - Hannan-Quinn = (-2 LL + 2 k ln(ln(n))) / n
    - Standard Error = √diag(H^-1), with H the negative Hessian of LL
    - Robust Standard Error = √diag(H^-1 J H^-1), with J = Σ s_t s_t' the outer product of the
      scores s_t of every observation (Bollerslev and Wooldridge, 1992)

    The information criteria are per observation, as in R's rugarch. The Hessian and the
    scores are central finite differences of the log-likelihood on returns scaled to a unit
    Variance (where every weight is of the order of 1), converted to the weights of the
    returns with the Jacobian of `_rescale_weights` (the delta method). The robust standard
    errors remain valid when the innovation distribution is misspecified (quasi maximum
    likelihood).

    For more information, see the following papers:

    - Bollerslev, T. and Wooldridge, J.M. (1992). "Quasi-Maximum Likelihood Estimation and
    Inference in Dynamic Models with Time-Varying Covariances." Econometric Reviews, 11(2),
    143-172.
    - Engle, R.F. and Ng, V.K. (1993). "Measuring and Testing the Impact of News on
    Volatility." The Journal of Finance, 48(5), 1749-1778.

    Referenced against:
    - arch, GARCH (classic and robust standard errors, log-likelihood, information criteria): https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Shares similar logic:
    - rugarch (R), persistence: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2529
    - rugarch (R), halflife: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2490
    - rugarch (R), uncvariance: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L3036
    - rugarch (R), infocriteria: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2063

    Args:
        returns (np.ndarray): The returns.
        weights (Sequence[float]): The fitted weights.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        realized_measure (np.ndarray | None, optional): The realized measure of every return,
            which the Realized GARCH needs. Defaults to None.

    Returns:
        dict[str, float]: The diagnostics, with a "<weight> Standard Error" and a
        "<weight> Robust Standard Error" for every estimated weight.

    Notes:
    - The arch package reports the information criteria in total, rugarch and the Toolkit per
    observation.
    """
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    names = get_garch_family_parameter_names(method, distribution, mean)
    variance_weights, _, distribution_weights = _split_weights(weights, method, mean)

    log_likelihoods = get_garch_family_log_likelihood(
        returns,
        weights,
        method,
        distribution,
        mean,
        per_observation=True,
        realized_measure=realized_measure,
    )
    log_likelihood = float(np.sum(log_likelihoods))
    observations = len(log_likelihoods)
    estimated = len(names)
    persistence = get_garch_family_persistence(
        variance_weights, method, distribution, distribution_weights
    )

    diagnostics = {
        "Persistence": persistence,
        "Half-Life": (
            math.log(0.5) / math.log(persistence) if 0 < persistence < 1 else math.nan
        ),
        "Unconditional Variance": get_garch_family_unconditional_variance(
            variance_weights, method, distribution, distribution_weights
        ),
        "Log-Likelihood": log_likelihood,
        "AIC": (-2 * log_likelihood + 2 * estimated) / observations,
        "BIC": (-2 * log_likelihood + estimated * math.log(observations))
        / observations,
        "Shibata": -2 * log_likelihood / observations
        + math.log((observations + 2 * estimated) / observations),
        "Hannan-Quinn": (
            -2 * log_likelihood + 2 * estimated * math.log(math.log(observations))
        )
        / observations,
    }

    per_observation, scaled_weights, steps, scores, jacobian = _score_terms(
        returns, weights, method, distribution, mean, realized_measure
    )
    hessian = _numerical_hessian(
        lambda point: float(np.sum(per_observation(point))), scaled_weights, steps
    )

    try:
        inverse = np.linalg.inv(-hessian)
        covariance = jacobian @ inverse @ jacobian.T
        robust = jacobian @ inverse @ (scores.T @ scores) @ inverse @ jacobian.T
    except np.linalg.LinAlgError:
        covariance = robust = np.full((estimated, estimated), np.nan)

    for i, name in enumerate(names):
        diagnostics[f"{name} Standard Error"] = (
            math.sqrt(covariance[i, i]) if covariance[i, i] >= 0 else math.nan
        )
    for i, name in enumerate(names):
        diagnostics[f"{name} Robust Standard Error"] = (
            math.sqrt(robust[i, i]) if robust[i, i] >= 0 else math.nan
        )

    return diagnostics


# The persistence from which a news impact curve treats a model as integrated, see get_news_impact_curve.  # noqa: E501
NEAR_INTEGRATED_PERSISTENCE = 1 - 1e-6


def get_news_impact_curve(
    weights,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    shocks: np.ndarray | None = None,
) -> pd.Series:
    """
    Calculate the news impact curve of Engle and Ng (1993): how a shock of each size moves the next
    period's volatility, the conditional variance of the next period as a function of this period's
    shock, with the variance of this period at the unconditional variance sigma^2 of the model.

    The formula is as follows:

    - News Impact(z) = sigma_(t+1)^2 given e_t = z * sigma and sigma_t^2 = sigma^2

    Where z is the shock in unconditional standard deviations, sigma^2 the unconditional
    variance (see `get_garch_family_unconditional_variance`) and, for the component model,
    the long-run component also at sigma^2. IGARCH and FIGARCH (with D > 0) have no
    unconditional variance, so their curve is NaN. For the realized GARCH the shock moves the
    next log variance through the measurement equation (Hansen, Huang and Shek, 2012), with
    the measurement error at its mean of 0:

    - News Impact(z) = exp(ln sigma^2 + Gamma * (Tau 1 * z + Tau 2 * (z^2 - 1)))
    A curve that rises faster to the left than to the right shows the
    leverage effect: bad news raises volatility by more than good news of the same size.

    For more information, see the following paper:

    - Engle, R.F. and Ng, V.K. (1993). "Measuring and Testing the Impact of News on
    Volatility." The Journal of Finance, 48(5), 1749-1778.

    Shares similar logic:
    - rugarch (R), newsimpact: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L2287

    Args:
        weights (Sequence[float]): The fitted weights.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        shocks (np.ndarray | None, optional): The shocks z, in unconditional standard
            deviations. Defaults to None, which uses -5, -4.5, ..., 5.

    Returns:
        pd.Series: The variance of the next period, indexed by the shock z.

    Notes:
    - A model with a persistence of at least 1 - 10^-6 is treated as integrated, as IGARCH: its
      unconditional variance Omega / (1 - P) is unbounded as P approaches 1 (a variance in the
      hundreds of millions for P = 1 - 10^-9), so its curve is NaN.
    """
    shocks = np.linspace(-5, 5, 21) if shocks is None else np.asarray(shocks, float)
    variance_weights, _, distribution_weights = _split_weights(weights, method, mean)
    variance = get_garch_family_unconditional_variance(
        variance_weights, method, distribution, distribution_weights
    )

    # A fit on the edge of stationarity (Alpha + Beta = 0.9999999) has an unconditional variance that only reflects how close P is to 1, so it is read as integrated.  # noqa: E501
    if (
        get_garch_family_persistence(
            variance_weights, method, distribution, distribution_weights
        )
        >= NEAR_INTEGRATED_PERSISTENCE
    ):
        variance = math.nan

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if method == "realized":
            _, _, gamma, _, _, tau_1, tau_2, _ = variance_weights
            news_impact = variance * np.exp(
                gamma * (tau_1 * shocks + tau_2 * (shocks**2 - 1))
            )
        elif method == "figarch":
            # Without an unconditional variance (D > 0) the curve has no level; with D = 0 FIGARCH is GARCH(1, 1).  # noqa: E501
            omega, phi, _, beta = variance_weights
            news_impact = (
                omega + (phi - beta) * shocks**2 * variance + beta * variance
            ) * np.ones(len(shocks))
        else:
            news_impact, _ = _variance_step(method, variance_weights)(
                np.full(len(shocks), variance), shocks * math.sqrt(variance), variance
            )

    return pd.Series(news_impact, index=pd.Index(shocks, name="Shock"))


# The quantiles that summarize the simulated paths per step.
SIMULATION_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _simulation_state(
    returns: np.ndarray,
    weights,
    method: str,
    mean: str,
    time_steps: int,
    realized_measure: np.ndarray | None,
) -> dict:
    """
    The state after the last return that a simulation continues from: the one-step-ahead
    variance (and long-run component), the last return and residual and, for the models with
    a longer memory, the part of every future step that the returns already fix: the
    FIGARCH sum over the squared residuals of the sample (with their backcast) and the
    ARFIMA sum Σ pi_k x_(n+h-k) over the centred returns of the sample.
    """
    backcast = get_backcast(_starting_residuals(returns, mean))
    residuals, sigma2, long_run, _ = _garch_family_filter(
        returns, weights, method, mean, backcast, realized_measure
    )
    variance_weights, mean_weights, _ = _split_weights(weights, method, mean)
    state = {
        "variance": float(sigma2[-1]),
        "long_run": float(long_run[-1]) if long_run is not None else None,
        "return": float(returns[-1]),
        "residual": float(residuals[-1]),
    }

    if method == "figarch":
        from scipy import signal  # noqa: PLC0415

        omega, phi, d, beta = variance_weights
        arch_weights = _figarch_weights(phi, d, beta)
        squared = np.concatenate(
            [np.full(FIGARCH_TRUNCATION, backcast), residuals * residuals]
        )
        # Step h (from 0) sums lambda_i over the squared residuals of the sample for i > h: the last len(squared) + h - 1 entries of the convolution with the weights shifted by h.  # noqa: E501
        padded = np.concatenate([squared, np.zeros(time_steps)])
        state["arch_weights"] = arch_weights
        state["fixed_variance"] = (
            omega / (1 - beta)
            + signal.fftconvolve(padded, arch_weights)[
                len(squared) - 1 : len(squared) - 1 + time_steps
            ]
        )

    if mean == "arfima":
        from scipy import signal  # noqa: PLC0415

        mu, autoregressive, d, moving_average = mean_weights
        centred = returns - mu
        fractional = _fractional_weights(d, len(centred) + time_steps + 1)
        differenced = signal.fftconvolve(centred, fractional[: len(centred)])
        state["fractional_weights"] = fractional
        state["fixed_mean"] = signal.fftconvolve(
            np.concatenate([centred, np.zeros(time_steps)]), fractional
        )[len(centred) : len(centred) + time_steps]
        state["differenced"] = float(differenced[len(centred) - 1])

    return state


def _simulate_paths(
    weights,
    method: str,
    mean: str,
    state: dict,
    draw,
    time_steps: int,
    paths: int,
) -> dict[str, np.ndarray]:
    """
    Simulates `paths` paths of `time_steps` returns of a GARCH-family model from `state`
    (see `_simulation_state`), with `draw(paths)` giving the standardized innovations z_t
    and, for the realized GARCH, the measurement errors u_t of every step.

    Returns:
        dict[str, np.ndarray]: The returns, the conditional volatility and, for the
        realized GARCH, the realized measure, each of shape (time_steps, paths).
    """
    variance_weights, mean_weights, _ = _split_weights(weights, method, mean)
    variance = np.full(paths, state["variance"])
    long_run = (
        np.full(paths, state["long_run"]) if state["long_run"] is not None else None
    )
    previous_return = np.full(paths, state["return"])
    previous_residual = np.full(paths, state["residual"])
    # FIGARCH and the realized GARCH have their own steps below.
    step = (
        (lambda variance, _, long_run: (variance, long_run))
        if method in HISTORY_METHODS
        else _variance_step(method, variance_weights)
    )

    returns = np.empty((time_steps, paths))
    volatilities = np.empty((time_steps, paths))
    # Only the model that needs them keeps a row per step.
    measures = np.empty((time_steps if method == "realized" else 0, paths))
    squared = np.empty((time_steps if method == "figarch" else 0, paths))
    centred = np.empty((time_steps if mean == "arfima" else 0, paths))
    differenced = np.full(paths, state.get("differenced", 0.0))

    with np.errstate(over="ignore", invalid="ignore"):
        for h in range(time_steps):
            volatility = np.sqrt(variance)
            z, u = draw(paths)
            residual = volatility * z

            if mean == "zero":
                value = residual
            elif mean == "constant":
                value = mean_weights[0] + residual
            elif mean == "arma":
                value = (
                    mean_weights[0]
                    + mean_weights[1] * previous_return
                    + mean_weights[2] * previous_residual
                    + residual
                )
            elif mean == "arfima":
                mu, autoregressive, _, moving_average = mean_weights
                # w_t = AR * w_(t-1) + e_t + MA * e_(t-1) and x_t = w_t - Σ_(k>=1) pi_k x_(t-k), the sample's part of the sum fixed in the state.  # noqa: E501
                differenced = (
                    autoregressive * differenced
                    + residual
                    + moving_average * previous_residual
                )
                weights_back = state["fractional_weights"][1 : h + 1][::-1]
                centred[h] = (
                    differenced
                    - state["fixed_mean"][h]
                    - (weights_back @ centred[:h] if h else 0.0)
                )
                value = mu + centred[h]
            else:
                value = mean_weights[0] + mean_weights[1] * volatility + residual

            returns[h] = value
            volatilities[h] = volatility

            if method == "figarch":
                squared[h] = residual * residual
                if h + 1 < time_steps:
                    # The squared residuals simulated so far, at most the 1,000 latest.
                    lags = min(h + 1, FIGARCH_TRUNCATION)
                    variance = state["fixed_variance"][h + 1] + (
                        state["arch_weights"][:lags][::-1]
                        @ squared[h + 1 - lags : h + 1]
                    )
            elif method == "realized":
                omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u = variance_weights
                log_variance = np.log(variance)
                log_measure = (
                    xi
                    + phi * log_variance
                    + tau_1 * z
                    + tau_2 * (z * z - 1)
                    + sigma_u * u
                )
                measures[h] = np.exp(log_measure)
                variance = np.exp(omega + beta * log_variance + gamma * log_measure)
            else:
                variance, long_run = step(variance, residual, long_run)

            previous_return, previous_residual = value, residual

    return {
        "returns": returns,
        "volatility": volatilities,
        "realized": measures if method == "realized" else None,
    }


def get_garch_simulation(
    returns: np.ndarray,
    weights=None,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "zero",
    forecast_steps: int = 10,
    simulations: int = 10000,
    seed: int | None = 0,
    output: str = "returns",
    bootstrap_fits: int | None = None,
    realized_measure: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate a GARCH simulation: the spread of the returns and conditional volatility a GARCH-family
    model expects after the end of the sample, from simulated paths summarized per step by the mean
    and the 5th, 25th, 50th, 75th and 95th percentiles.

    Every path starts from the fitted state after the last return (its one-step-ahead
    variance and, for the ARMA(1, 1) and ARFIMA(1, d, 1) means, the last return and
    residual, and for FIGARCH and the ARFIMA mean the sample's part of their long sums) and
    draws the standardized innovations z_t from the fitted distribution (see
    `get_standardized_draws`):

    - e_t = sigma_t * z_t
    - r_t = Mean_t + e_t, with Mean_t = 0, Mu, Mu + AR * r_(t-1) + MA * e_(t-1), Mu + Lambda * sigma_t or,
      for the ARFIMA mean, Mu + x_t with (1 - AR L)(1 - L)^d x_t = (1 + MA L) e_t
    - sigma_(t+1)^2 from the variance recursion of the model; the realized GARCH draws its
      measure from the measurement equation with u_t ~ N(0, Sigma U^2)

    With bootstrap fits, the forecast density of Pascual, Romo and Ruiz (2006) replaces
    the fitted distribution with the standardized residuals z_t of the fit (and, for the
    realized GARCH, the measurement errors of the same days), drawn with replacement, and
    adds the uncertainty of the weights:

    - Every bootstrap series r*_1..r*_n is simulated from the fitted weights, starting after the
      last return as rugarch's ugarchboot does, and the model is fitted to it again (a local
      search from the fitted weights, see `get_garch_family_weights`)
    - Each of these weights filters the original returns to the state after the last
      return and simulates simulations / bootstrap_fits of the paths from it
    - No bootstrap fits (0) keeps the fitted weights and only resamples the residuals (the
      partial bootstrap)

    For more information, see the following paper:

    - Pascual, L., Romo, J. and Ruiz, E. (2006). "Bootstrap Prediction for Returns and
    Volatilities in GARCH Models." Computational Statistics & Data Analysis, 50(9),
    2293-2312.

    Shares similar logic:
    - rugarch (R), ugarchsim: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1090
    - rugarch (R), ugarchpath: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1105
    - rugarch (R), ugarchboot (partial and full bootstrap): https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1210

    Args:
        returns (np.ndarray): The returns.
        weights (Sequence[float] | None, optional): The weights. Defaults to None, which fits them.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant", "arma" (ARMA(1, 1)), "in_mean"
            (GARCH-in-mean), "arfima" (ARFIMA(1, d, 1)). Defaults to "zero".
        forecast_steps (int, optional): The number of periods to simulate. Defaults to 10.
        simulations (int, optional): The number of paths. Defaults to 10,000.
        seed (int | None, optional): The seed of the random draws, None for a different
            simulation every time. Defaults to 0.
        output (str, optional): What to summarize: the return of each step, the compounded return from the end of the
            sample, (1 + r_1) ... (1 + r_h) - 1, or the conditional volatility sigma_t. Options are: "returns",
            "cumulative_returns", "volatility". Defaults to "returns".
        bootstrap_fits (int | None, optional): The number of bootstrap series the model is
            fitted to again, 0 for the partial bootstrap. Defaults to None, which draws from
            the fitted distribution instead.
        realized_measure (np.ndarray | None, optional): The realized measure of every return,
            which the Realized GARCH needs. Defaults to None.

    Returns:
        pd.DataFrame: The Mean and the percentiles per step (1 to forecast_steps).
    """
    validate_garch_family(method, distribution, mean)

    check_option("output", output, ("returns", "cumulative_returns", "volatility"))
    if bootstrap_fits is not None and bootstrap_fits < 0:
        raise ValueError("bootstrap_fits must be 0 or more.")

    returns = np.asarray(returns, dtype=float)

    if weights is None:
        weights = get_garch_family_weights(
            returns, method, distribution, mean, realized_measure=realized_measure
        )

    weights = np.asarray(weights, dtype=float)
    variance_weights, _, distribution_weights = _split_weights(weights, method, mean)
    rng = np.random.default_rng(seed)

    if bootstrap_fits is None:

        def draw(size):
            return (
                get_standardized_draws(distribution, distribution_weights, size, rng),
                rng.standard_normal(size) if method == "realized" else None,
            )

    else:
        backcast = get_backcast(_starting_residuals(returns, mean))
        residuals, sigma2, _, held_back = _garch_family_filter(
            returns, weights, method, mean, backcast, realized_measure
        )
        standardized = residuals / np.sqrt(sigma2[: len(residuals)])
        errors = np.zeros(len(standardized))
        if method == "realized":
            _, _, _, xi, phi, tau_1, tau_2, sigma_u = variance_weights
            # The standardized measurement errors of the days with a measure; a day without one draws a normal error.  # noqa: E501
            errors = (
                _log_measure(realized_measure)[held_back:]
                - xi
                - phi * np.log(sigma2[: len(residuals)])
                - tau_1 * standardized
                - tau_2 * (standardized**2 - 1)
            ) / sigma_u
            errors = np.where(
                np.isfinite(errors), errors, rng.standard_normal(len(errors))
            )

        def draw(size):
            index = rng.integers(0, len(standardized), size)
            return standardized[index], errors[index]

    fits = [weights]

    if bootstrap_fits:
        state = _simulation_state(
            returns, weights, method, mean, len(returns), realized_measure
        )
        fits = []
        for _ in range(bootstrap_fits):
            series = _simulate_paths(
                weights, method, mean, state, draw, len(returns), 1
            )
            series_measure = series["realized"][:, 0] if method == "realized" else None
            fitted = get_garch_family_weights(
                series["returns"][:, 0],
                method,
                distribution,
                mean,
                starting_weights=weights,
                realized_measure=series_measure,
            )
            if np.all(np.isfinite(fitted)):
                fits.append(fitted)
        if not fits:
            raise ValueError("The model could not be fitted to any bootstrap series.")

    paths_per_fit = -(-simulations // len(fits))
    simulated = []
    for fit_weights in fits:
        state = _simulation_state(
            returns, fit_weights, method, mean, forecast_steps, realized_measure
        )
        simulated.append(
            _simulate_paths(
                fit_weights, method, mean, state, draw, forecast_steps, paths_per_fit
            )
        )

    values = np.concatenate(
        [
            (
                np.cumprod(1 + path["returns"], axis=0) - 1
                if output == "cumulative_returns"
                else path[output]
            )
            for path in simulated
        ],
        axis=1,
    )[:, :simulations]

    return pd.DataFrame(
        {
            "Mean": values.mean(axis=1),
            **{
                f"{quantile:.0%}": np.quantile(values, quantile, axis=1)
                for quantile in SIMULATION_QUANTILES
            },
        },
        index=pd.Index(np.arange(1, forecast_steps + 1), name="Step"),
    )


def get_standardized_cdf(
    z: np.ndarray, distribution: str = "normal", parameters: tuple[float, ...] = ()
) -> np.ndarray:
    """
    Calculate the distribution function F(z) = P(Z <= z) of a standardized innovation distribution
    (mean 0 and Variance 1): the probability of a shock at or below z.

    The formula is as follows, with G the standardized Student-t distribution function
    G(x) = T_v(x √(v / (v - 2))) and the constants a, b, m and s of `get_standardized_log_density`:

    - Normal: F(z) = Φ(z)
    - Student-t: F(z) = G(z)
    - Hansen's skewed Student-t (Hansen, 1994): F(z) = (1 - λ) G((b z + a) / (1 - λ)) below z = -a / b and
      (1 - λ) / 2 + (1 + λ) (G((b z + a) / (1 + λ)) - 1 / 2) above it
    - Fernández and Steel's skewed Student-t: with x = m + s z, F = 2 / (ξ + 1 / ξ) G(x ξ) / ξ below zero and
      1 / (1 + ξ^2) + 2 ξ / (ξ + 1 / ξ) (G(x / ξ) - 1 / 2) above it
    - Generalized error distribution: F(z) = 1 / 2 + sign(z) P(Γ(1 / v), |z / λ|^v / 2) / 2, with P the regularized
      lower incomplete gamma function
    - Johnson SU: F(z) = Φ(γ + δ asinh(m + s z))

    Shares similar logic:
    - rugarch (R), pdist: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-distributions.R#L1736

    Args:
        z (np.ndarray): The standardized values.
        distribution (str, optional): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (Sequence[float], optional): The Shape (and Skew) of the distribution.
            Defaults to (), for the normal distribution.

    Returns:
        np.ndarray: The probability at every z.
    """
    z = np.asarray(z, dtype=float)

    if distribution == "normal":
        return special.ndtr(z)

    if distribution == "johnson_su":
        shape, skew = parameters
        location, scale = _johnson_su_moments(shape, skew)
        return special.ndtr(skew + shape * np.arcsinh(location + scale * z))

    if distribution == "ged":
        shape = parameters[0]
        scale = _ged_scale(shape)
        return 0.5 + 0.5 * np.sign(z) * special.gammainc(
            1 / shape, 0.5 * np.abs(z / scale) ** shape
        )

    shape = parameters[0]
    t_scale = math.sqrt(shape / (shape - 2))

    def student_t(x):
        return special.stdtr(shape, x * t_scale)

    if distribution == "t":
        return student_t(z)

    if distribution == "skewt":
        skew = parameters[1]
        a, b = _hansen_constants(shape, skew)
        x = b * z + a
        return np.where(
            x < 0,
            (1 - skew) * student_t(x / (1 - skew)),
            (1 - skew) / 2 + (1 + skew) * (student_t(x / (1 + skew)) - 0.5),
        )

    if distribution == "skewt_fernandez_steel":
        skew = parameters[1]
        location, scale = _fernandez_steel_moments(shape, skew)
        x = location + scale * z
        weight = 2 / (skew + 1 / skew)
        return np.where(
            x < 0,
            weight / skew * student_t(x * skew),
            1 / (1 + skew**2) + weight * skew * (student_t(x / skew) - 0.5),
        )

    raise InvalidArgumentError(
        f"distribution must be one of {join_options(GARCH_FAMILY_DISTRIBUTION_PARAMETERS)}, "
        f"not {distribution!r}. Please pass one of these as distribution."
    )


def get_standardized_quantile(
    probability: float | np.ndarray,
    distribution: str = "normal",
    parameters: tuple[float, ...] = (),
) -> np.ndarray:
    """
    Calculate the quantile F^-1(p) of a standardized innovation distribution (mean 0 and Variance 1):
    the shock that a share p of the shocks falls below, the inverse of `get_standardized_cdf`.

    The formula, in closed form, is as follows:

    - Normal: Φ^-1(p)
    - Student-t: G^-1(p) = T_v^-1(p) √((v - 2) / v)
    - Hansen's skewed Student-t (Hansen, 1994): z = ((1 - λ) G^-1(p / (1 - λ)) - a) / b for p < (1 - λ) / 2 and
      z = ((1 + λ) G^-1(1 / 2 + (p - (1 - λ) / 2) / (1 + λ)) - a) / b otherwise
    - Fernández and Steel's skewed Student-t: x = G^-1(p ξ (ξ + 1 / ξ) / 2) / ξ for p < 1 / (1 + ξ^2) and
      x = ξ G^-1(1 / 2 + (p - 1 / (1 + ξ^2)) (ξ + 1 / ξ) / (2 ξ)) otherwise, z = (x - m) / s
    - Generalized error distribution: z = sign(p - 1 / 2) λ (2 P^-1(1 / v, |2 p - 1|))^(1 / v)
    - Johnson SU: z = (sinh((Φ^-1(p) - γ) / δ) - m) / s

    Referenced against:
    - arch, Normal, StudentsT, SkewStudent and GeneralizedError (ppf): https://bashtage.github.io/arch/univariate/generated/arch.univariate.SkewStudent.html

    Shares similar logic:
    - rugarch (R), qdist: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-distributions.R#L1757

    Args:
        probability (float | np.ndarray): The probability (e.g., 0.05 for the 5% quantile).
        distribution (str, optional): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (Sequence[float], optional): The Shape (and Skew) of the distribution.
            Defaults to (), for the normal distribution.

    Returns:
        np.ndarray: The quantile.
    """
    p = np.asarray(probability, dtype=float)

    if distribution == "normal":
        return special.ndtri(p)

    if distribution == "johnson_su":
        shape, skew = parameters
        location, scale = _johnson_su_moments(shape, skew)
        return (np.sinh((special.ndtri(p) - skew) / shape) - location) / scale

    if distribution == "ged":
        shape = parameters[0]
        return (
            np.sign(p - 0.5)
            * _ged_scale(shape)
            * (2 * special.gammaincinv(1 / shape, np.abs(2 * p - 1))) ** (1 / shape)
        )

    shape = parameters[0]
    t_scale = math.sqrt((shape - 2) / shape)

    def inverse_student_t(q):
        return special.stdtrit(shape, q) * t_scale

    if distribution == "t":
        return inverse_student_t(p)

    if distribution == "skewt":
        skew = parameters[1]
        a, b = _hansen_constants(shape, skew)
        x = np.where(
            p < (1 - skew) / 2,
            (1 - skew) * inverse_student_t(np.minimum(p / (1 - skew), 1.0)),
            (1 + skew)
            * inverse_student_t(
                np.clip(0.5 + (p - (1 - skew) / 2) / (1 + skew), 0.0, 1.0)
            ),
        )
        return (x - a) / b

    if distribution == "skewt_fernandez_steel":
        skew = parameters[1]
        location, scale = _fernandez_steel_moments(shape, skew)
        weight = skew + 1 / skew
        below = p < 1 / (1 + skew**2)
        x = np.where(
            below,
            inverse_student_t(np.minimum(p * skew * weight / 2, 1.0)) / skew,
            skew
            * inverse_student_t(
                np.clip(0.5 + (p - 1 / (1 + skew**2)) * weight / (2 * skew), 0.0, 1.0)
            ),
        )
        return (x - location) / scale

    raise InvalidArgumentError(
        f"distribution must be one of {join_options(GARCH_FAMILY_DISTRIBUTION_PARAMETERS)}, "
        f"not {distribution!r}. Please pass one of these as distribution."
    )


def get_standardized_expected_shortfall(
    probability: float,
    distribution: str = "normal",
    parameters: tuple[float, ...] = (),
) -> float:
    """
    Calculate the expected shortfall of a standardized innovation distribution (mean 0 and Variance 1):
    the average shock in the worst tail, the mean of z below its quantile at `probability`.

    The formula is as follows:

    - ES(p) = E[z | z <= F^-1(p)] = (1 / p) ∫_0^p F^-1(u) du

    The normal and Student-t distributions have closed forms, ES = -φ(Φ^-1(p)) / p and
    ES = -√((v - 2) / v) (v + t^2) / (v - 1) f_v(t) / p with t = T_v^-1(p) and f_v the Student-t
    density; the other distributions integrate their quantile (see
    `get_standardized_quantile`) numerically.

    Args:
        probability (float): The tail probability (e.g., 0.05).
        distribution (str, optional): The distribution. Options are: "normal", "t" (Student-t), "skewt" (Hansen's
            skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged" (generalized
            error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        parameters (Sequence[float], optional): The Shape (and Skew) of the distribution.
            Defaults to (), for the normal distribution.

    Returns:
        float: The expected shortfall, negative for a probability below 1 / 2.
    """
    from scipy import integrate  # noqa: PLC0415

    if distribution == "normal":
        return float(
            -math.exp(-0.5 * special.ndtri(probability) ** 2 - 0.5 * LOG_2_PI)
            / probability
        )

    if distribution == "t":
        shape = parameters[0]
        t = float(special.stdtrit(shape, probability))
        density = math.exp(
            special.gammaln((shape + 1) / 2)
            - special.gammaln(shape / 2)
            - 0.5 * math.log(shape * math.pi)
            - (shape + 1) / 2 * math.log1p(t * t / shape)
        )
        return float(
            -math.sqrt((shape - 2) / shape)
            * (shape + t * t)
            / (shape - 1)
            * density
            / probability
        )

    integral, _ = integrate.quad(
        lambda u: float(get_standardized_quantile(u, distribution, parameters)),
        0,
        probability,
        limit=200,
    )

    return integral / probability


def _ged_scale(shape: float) -> float:
    """The λ that gives the generalized error distribution of Shape v a unit Variance."""
    return math.sqrt(
        2 ** (-2 / shape)
        * math.exp(special.gammaln(1 / shape) - special.gammaln(3 / shape))
    )


def _hansen_constants(shape: float, skew: float) -> tuple[float, float]:
    """The constants a and b of Hansen's (1994) skewed Student-t distribution."""
    c = math.exp(
        special.gammaln((shape + 1) / 2) - special.gammaln(shape / 2)
    ) / math.sqrt(math.pi * (shape - 2))
    a = 4 * skew * c * (shape - 2) / (shape - 1)

    return a, math.sqrt(1 + 3 * skew**2 - a**2)


ROLLING_FORECAST_CACHE_SIZE = 32
ROLLING_FORECAST_CACHE: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
ROLLING_FORECAST_CACHE_LOCK = threading.Lock()


def get_garch_family_rolling_forecast(
    returns: pd.Series,
    window_size: int | None = None,
    refit_every: int = 21,
    method: str = "garch",
    distribution: str = "normal",
    mean: str = "constant",
) -> pd.DataFrame:
    """
    Calculate a rolling GARCH forecast: the conditional mean and volatility each return was expected to
    have, from a GARCH-family model fitted only on the returns before it.

    The model is refitted every `refit_every` returns on a moving window of `window_size` returns, with
    the parameters of the innovation distribution of every fit.

    The model is fitted on the window of returns that ends at a refit date and the fitted
    weights filter every later return until the next refit, so the forecast for each return
    only uses the returns before it:

    - Mean_(t+1) = Mu (0 for the zero mean)
    - Volatility_(t+1) = √(sigma_(t+1)^2), the one-step variance of the recursion of the weights of
      the last refit on or before t, run over the returns from the start of that fit's window

    The first fit is the multi-start search of `get_garch_family_weights`; every later fit
    starts a local search from the previous weights, and a fit that starts or ends on a bound,
    or does not converge, also runs the multi-start search and keeps the higher likelihood
    (see `get_garch_family_weights`). The recursion of a
    refit starts from the backcast of its own window, as in the fit.

    Without `window_size`, the model is fitted once on every return and the result is the
    forecast for the period after the last return.

    Shares similar logic:
    - rugarch (R), ugarchroll: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1141

    Args:
        returns (pd.Series): The returns of one asset, without missing values.
        window_size (int | None, optional): The number of returns of every fit. Defaults to
            None, which fits once on every return.
        refit_every (int, optional): The number of returns between two fits. Defaults to 21.
        method (str, optional): The variance model. Options are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch"
            (EGARCH), "igarch" (IGARCH), "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH), "component" (component GARCH),
            "figarch" (FIGARCH), "realized" (Realized GARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal", "t" (Student-t), "skewt"
            (Hansen's skewed Student-t), "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson SU). Defaults to "normal".
        mean (str, optional): The mean model. Options are: "zero", "constant". Defaults to "constant".

    Returns:
        pd.DataFrame: The Mean and Volatility of the next return and the parameters of the
        innovation distribution (Shape, Skew), indexed by the date of the last return each
        forecast uses.

    Notes:
    - rugarch's ugarchroll refits every number of periods on a moving or expanding window and starts
    every fit from its default starting values, where every later fit here starts from the previous
    weights.
    """
    validate_garch_family(method, distribution, mean)

    if mean not in ("zero", "constant"):
        raise InvalidArgumentError(
            "The rolling forecast takes mean='zero' or mean='constant', whose next mean "
            "is known without the next return."
        )
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1.")
    if method == "realized":
        raise InvalidArgumentError(
            "The rolling forecast takes the models of the returns alone; method='realized' "
            "also needs a realized measure."
        )

    returns = returns.dropna()
    values = returns.to_numpy(dtype=float)
    n = len(values)
    window = n if window_size is None else int(window_size)

    if window < MINIMUM_FIT_OBSERVATIONS or n < window:
        raise ValueError(
            f"The GARCH forecast needs at least {max(window, MINIMUM_FIT_OBSERVATIONS)} "
            f"returns, received {n}."
        )

    key = (
        hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        n,
        window,
        refit_every,
        method,
        distribution,
        mean,
    )
    with ROLLING_FORECAST_CACHE_LOCK:
        cached = ROLLING_FORECAST_CACHE.get(key)
        if cached is not None:
            ROLLING_FORECAST_CACHE.move_to_end(key)
            return cached.copy()

    variance_end = len(GARCH_FAMILY_VARIANCE_PARAMETERS[method])
    distribution_names = GARCH_FAMILY_DISTRIBUTION_PARAMETERS[distribution]
    forecasts = np.full((n - window + 1, 2 + len(distribution_names)), np.nan)
    previous_weights = None

    for fit_end in range(window, n + 1, refit_every):
        fit_returns = values[fit_end - window : fit_end]
        weights = get_garch_family_weights(
            fit_returns, method, distribution, mean, starting_weights=previous_weights
        )

        # The forecasts of this fit run up to the next refit, the last one past the final return.  # noqa: E501
        block_end = min(fit_end + refit_every - 1, n)
        first, last = fit_end - window, block_end - window + 1

        if np.all(np.isfinite(weights)):
            previous_weights = weights
            backcast = get_backcast(_starting_residuals(fit_returns, mean))
            _, sigma2, _, _ = _garch_family_filter(
                values[fit_end - window : block_end], weights, method, mean, backcast
            )
            forecasts[first:last, 0] = (
                weights[variance_end] if mean == "constant" else 0.0
            )
            forecasts[first:last, 1] = np.sqrt(sigma2[window:])
            forecasts[first:last, 2:] = weights[
                len(weights) - len(distribution_names) :
            ]

    result = pd.DataFrame(
        forecasts,
        index=returns.index[window - 1 :],
        columns=["Mean", "Volatility", *distribution_names],
    )

    with ROLLING_FORECAST_CACHE_LOCK:
        ROLLING_FORECAST_CACHE[key] = result.copy()
        if len(ROLLING_FORECAST_CACHE) > ROLLING_FORECAST_CACHE_SIZE:
            ROLLING_FORECAST_CACHE.popitem(last=False)

    return result
