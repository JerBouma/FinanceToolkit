"""GARCH Model"""

import numpy as np
import pandas as pd
from scipy import optimize

ALPHA_CONSTRAINT = 0.5

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2


def garch_log_maximization(
    weights: list, returns: np.ndarray, t: int, p: int = 1, q: int = 1
) -> float:
    """
    Calculates -SUM(-ln(v_i) - (u_i ^ 2) / v_i)

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

    # Compute the log-likelihood in a vectorized manner. garch[i] is the conditional
    # variance for period i (built from returns[i - 1]), so it must be paired with
    # returns[i] itself, not returns[i - 1]'s variance.
    u = returns[1:t]
    v = garch[1:t]

    # Use np.sum to calculate the sum of the log-likelihood
    result = -np.sum(-np.log(v) - u**2 / v)

    return result


def get_garch_weights(
    returns: np.ndarray, t: int | None = None, p: int = 1, q: int = 1
) -> list:
    """
    Estimates the weights (parameters) for a GARCH(p, q) model using simulated annealing optimization.

    The weights are estimated by using simulated annealing, which goes over different values of
    (1 - alpha - beta) sigma_l, alpha and beta, while maximizing: SUM(-ln(v_i) - (u_i ^ 2) / v_i).
    With the constraints:
    - alpha > 0
    - beta > 0
    - alpha + beta < 1
    Note that there is no restriction on (1 - alpha - beta) sigma_l.

    Args:
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize GARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        list: A list with the weights
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0].to_numpy()
    if t is None:
        t = len(returns)

    bounds = [(1e-9, 1), (1e-9, 1), (1e-9, 1)]
    initial_guess = [np.var(returns[: t - 1]), 0.1, 0.8]

    # Define the wrapper function for optimization that applies the constraints
    def wrapper_func(parameters):
        alpha = parameters[1]
        beta = parameters[2]
        if alpha + beta >= 1:  # Constraint
            return np.inf  # Return a large number to represent an invalid solution
        return garch_log_maximization(parameters, returns, t, p, q)

    # Perform the optimization using simulated annealing. Seeded so that fitted
    # parameters (and anything derived from them) are reproducible across runs.
    result = optimize.dual_annealing(wrapper_func, bounds, x0=initial_guess, seed=42)

    return result.x


def get_garch(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    time_steps: int | None = None,
    optimization_t: int | None = None,
    p: int = 1,
    q: int = 1,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Calculates volatility forecasts based on the GARCH model.

    GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is for
    instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH models.

    More information can be found in:
    - https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity#GARCH
    - Generalized Autoregressive Conditional Heteroskedasticity, by Tim Bollerslev
    - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1
    - Options, Futures & other Derivates, by John C. Hull; ISBN: 0-13-022444-8

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha and beta. Note that these are used all columns
        in the returns.
        time_steps (int): Time steps to calculate GARCH for.
        optimization_t (int): Time steps to optimize GARCH for. It is only used if no weights are given.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.array | pd.Series | pd.DataFrame: An object with sigma_2 values
    """
    # TODO: support GARCH(p, q), for any p and q  # pylint: disable=W0511
    if p != 1 or q != 1:
        raise ValueError(
            "Invalid input for p or/and q, currently only GARCH(1, 1) is implemented."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_garch,
                    weights=weights,
                    time_steps=time_steps,
                    optimization_t=optimization_t,
                    p=p,
                    q=q,
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            garch = pd.concat(period_data_list, axis=0)

            return garch
        return returns.aggregate(
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

        # Initialize sigma2 with zeros and set the first value
        sigma2 = np.zeros(time_steps)
        sigma2[0] = returns[0] ** 2

        # Calculate sigma2 values using a vectorized approach
        for i in range(1, time_steps):
            sigma2[i] = (
                weights[0]
                + weights[1] * returns[i - 1] ** 2
                + weights[2] * sigma2[i - 1]
            )

        return sigma2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def get_garch_forecast(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    weights: list | None = None,
    time_steps: int = 10,
    p: int = 1,
    q: int = 1,
):
    """Calculates sigma_2 forecasts.

    GARCH (Generalized autoregressive conditional heteroskedasticity) is stochastic model for time series, which is for
    instance used to model volatility clusters, stock return and inflation. It is a generalisation of the ARCH models.

    The forecasting with GARCH is done with the following formula:
    long_run_variance + (current_variance - long_run_variance) * (alpha + beta) ** (t - 1)

    For more:
    - Finance Compact Plus Band 1, by Yvonne Seler Zimmerman and Heinz Zimmerman; ISBN: 978-3-907291-31-1

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha and beta
        time_steps (int): Time steps to calculate GARCH for
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
                period_data = returns.loc[sub_period].aggregate(
                    get_garch_forecast, weights=weights, time_steps=time_steps, p=p, q=q
                )
                period_data.name = sub_period
                period_data.columns = [
                    col + " " + str(sub_period) for col in period_data.columns
                ]

                if not period_data.empty:
                    period_data_list.append(period_data)

            garch_forecast = pd.concat(period_data_list, axis=1)

            return garch_forecast

        return returns.aggregate(
            get_garch_forecast, weights=weights, time_steps=time_steps, p=p, q=q
        )
    if isinstance(returns, pd.Series):
        return get_garch_forecast(
            returns=returns.values, weights=weights, time_steps=time_steps, p=p, q=q
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            weights = get_garch_weights(returns, p=p, q=q)

        garch_values = get_garch(returns, weights, time_steps, p=p, q=q)

        # weights[0] / (1 - alpha - beta) is already the long-run VARIANCE (not a
        # std-dev), so it must not be squared again. The forecast is seeded from the
        # most recently fitted conditional variance, not the initial (arbitrary)
        # first value of the fitted series.
        long_run_variance = weights[0] / (1 - weights[1] - weights[2])
        current_variance = garch_values[-1]
        sigma_2 = np.zeros(time_steps)
        sigma_2[0] = current_variance
        for i in range(1, time_steps):
            sigma_2[i] = long_run_variance + (current_variance - long_run_variance) * (
                weights[1] + weights[2]
            ) ** (i - 1)

        return sigma_2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def gjr_garch_log_maximization(
    weights: list, returns: np.ndarray, t: int, p: int = 1, q: int = 1
) -> float:
    """
    Calculates -SUM(-ln(v_i) - (u_i ^ 2) / v_i) for the GJR-GARCH model.

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

    u = returns[1:t]
    v = garch[1:t]

    return -np.sum(-np.log(v) - u**2 / v)


def get_gjr_garch_weights(
    returns: np.ndarray, t: int | None = None, p: int = 1, q: int = 1
) -> list:
    """
    Estimates the weights (parameters) for a GJR-GARCH(1, 1, 1) model using simulated
    annealing optimization.

    The weights are estimated by maximizing the Gaussian log-likelihood, subject to:
    - omega > 0, alpha > 0, beta > 0
    - alpha + gamma >= 0 (so the conditional variance stays non-negative regardless of
      the sign of the shock)
    - alpha + gamma / 2 + beta < 1 (the GJR-GARCH stationarity condition, assuming a
      symmetric error distribution)

    Args:
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize GJR-GARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        list: A list with the weights [omega, alpha, gamma, beta].
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0].to_numpy()
    if t is None:
        t = len(returns)

    bounds = [(1e-9, 1), (1e-9, 1), (-1, 1), (1e-9, 1)]
    initial_guess = [np.var(returns[: t - 1]), 0.05, 0.1, 0.8]

    def wrapper_func(parameters):
        alpha, gamma, beta = parameters[1], parameters[2], parameters[3]
        if alpha + gamma / 2 + beta >= 1:  # Stationarity constraint
            return np.inf
        if alpha + gamma < 0:  # Non-negativity constraint
            return np.inf
        return gjr_garch_log_maximization(parameters, returns, t, p, q)

    result = optimize.dual_annealing(wrapper_func, bounds, x0=initial_guess, seed=42)

    return result.x


def get_gjr_garch(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    time_steps: int | None = None,
    optimization_t: int | None = None,
    p: int = 1,
    q: int = 1,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Calculates volatility forecasts based on the GJR-GARCH model.

    GJR-GARCH extends GARCH with a leverage term that lets negative shocks (bad news)
    raise volatility by more than positive shocks of the same size, a well documented
    asymmetry in equity returns that symmetric GARCH cannot capture:

    - sigma_t^2 = omega + (alpha + gamma * 1[u_(t-1) < 0]) * u_(t-1)^2 + beta * sigma_(t-1)^2

    A positive gamma indicates the presence of a leverage effect.

    For more information about the method, see the following paper:

    - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
    between the Expected Value and the Volatility of the Nominal Excess Return on
    Stocks." The Journal of Finance, 48(5), 1779-1801.

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta. Note that these are used
        for all columns in the returns.
        time_steps (int): Time steps to calculate GJR-GARCH for.
        optimization_t (int): Time steps to optimize GJR-GARCH for. It is only used if no weights are given.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.array | pd.Series | pd.DataFrame: An object with sigma_2 values
    """
    if p != 1 or q != 1:
        raise ValueError(
            "Invalid input for p or/and q, currently only GJR-GARCH(1, 1, 1) is implemented."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_gjr_garch,
                    weights=weights,
                    time_steps=time_steps,
                    optimization_t=optimization_t,
                    p=p,
                    q=q,
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            gjr_garch = pd.concat(period_data_list, axis=0)

            return gjr_garch
        return returns.aggregate(
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

        sigma2 = np.zeros(time_steps)
        sigma2[0] = returns[0] ** 2

        for i in range(1, time_steps):
            indicator = 1.0 if returns[i - 1] < 0 else 0.0
            sigma2[i] = (
                omega
                + (alpha + gamma * indicator) * returns[i - 1] ** 2
                + beta * sigma2[i - 1]
            )

        return sigma2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def get_gjr_garch_forecast(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    weights: list | None = None,
    time_steps: int = 10,
    p: int = 1,
    q: int = 1,
):
    """Calculates sigma_2 forecasts based on the GJR-GARCH model.

    The forecasting with GJR-GARCH is done with the following formula, using the
    "effective" persistence (alpha + gamma / 2 + beta), which reduces to the plain
    GARCH forecast formula when gamma = 0:

    long_run_variance + (current_variance - long_run_variance) * (alpha + gamma / 2 + beta) ** (t - 1)

    For more information about the method, see the following paper:

    - Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
    between the Expected Value and the Volatility of the Nominal Excess Return on
    Stocks." The Journal of Finance, 48(5), 1779-1801.

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta.
        time_steps (int): Time steps to calculate GJR-GARCH for.
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
                period_data = returns.loc[sub_period].aggregate(
                    get_gjr_garch_forecast,
                    weights=weights,
                    time_steps=time_steps,
                    p=p,
                    q=q,
                )
                period_data.name = sub_period
                period_data.columns = [
                    col + " " + str(sub_period) for col in period_data.columns
                ]

                if not period_data.empty:
                    period_data_list.append(period_data)

            gjr_garch_forecast = pd.concat(period_data_list, axis=1)

            return gjr_garch_forecast

        return returns.aggregate(
            get_gjr_garch_forecast, weights=weights, time_steps=time_steps, p=p, q=q
        )
    if isinstance(returns, pd.Series):
        return get_gjr_garch_forecast(
            returns=returns.values, weights=weights, time_steps=time_steps, p=p, q=q
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            weights = get_gjr_garch_weights(returns, p=p, q=q)

        garch_values = get_gjr_garch(returns, weights, time_steps, p=p, q=q)

        omega, alpha, gamma, beta = weights
        persistence = alpha + gamma / 2 + beta

        long_run_variance = omega / (1 - persistence)
        current_variance = garch_values[-1]
        sigma_2 = np.zeros(time_steps)
        sigma_2[0] = current_variance
        for i in range(1, time_steps):
            sigma_2[i] = long_run_variance + (
                current_variance - long_run_variance
            ) * persistence ** (i - 1)

        return sigma_2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def egarch_log_maximization(
    weights: list, returns: np.ndarray, t: int, p: int = 1, q: int = 1
) -> float:
    """
    Calculates -SUM(-ln(v_i) - (u_i ^ 2) / v_i) for the EGARCH model.

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

    u = returns[1:t]
    v = garch[1:t]

    return -np.sum(-np.log(v) - u**2 / v)


def get_egarch_weights(
    returns: np.ndarray, t: int | None = None, p: int = 1, q: int = 1
) -> list:
    """
    Estimates the weights (parameters) for an EGARCH(1, 1) model using simulated
    annealing optimization.

    Unlike GARCH and GJR-GARCH, EGARCH models the log of the conditional variance, so
    omega, alpha and gamma are unconstrained in sign; only |beta| < 1 is required for
    stationarity of the log-variance process.

    Args:
        returns (np.ndarray): A np.ndarray of returns.
        t (int): Time steps to optimize EGARCH for.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        list: A list with the weights [omega, alpha, gamma, beta].
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0].to_numpy()
    if t is None:
        t = len(returns)

    # Alpha, gamma and beta are kept tight around typical empirical EGARCH
    # magnitudes (Nelson, 1991, and the applied EGARCH literature since) -- wider
    # bounds let a global optimizer wander into numerically pathological corners of
    # the log-variance recursion on small samples. Omega needs a much wider range:
    # since omega ~= (1 - beta) * long_run_log_variance and daily-return-scale
    # variances are small (e.g. ln(0.0002) =~ -8.5), omega genuinely sits well below
    # -1 for realistic return data even though it stays close to zero on a log scale.
    bounds = [(-15, 5), (-1, 1), (-1, 1), (-0.999, 0.999)]
    initial_guess = [0.0, 0.1, -0.1, 0.9]

    def wrapper_func(parameters):
        beta = parameters[3]
        if abs(beta) >= 1:  # Stationarity constraint
            return np.inf
        return egarch_log_maximization(parameters, returns, t, p, q)

    result = optimize.dual_annealing(wrapper_func, bounds, x0=initial_guess, seed=42)

    return result.x


def get_egarch(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    weights: np.ndarray | list | None = None,
    time_steps: int | None = None,
    optimization_t: int | None = None,
    p: int = 1,
    q: int = 1,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Calculates volatility forecasts based on the EGARCH model.

    EGARCH models the log of the conditional variance, which avoids having to
    constrain the parameters to keep the variance positive and, like GJR-GARCH, lets
    negative and positive shocks of the same size have a different impact on
    volatility (the leverage effect):

    - ln(sigma_t^2) = omega + beta * ln(sigma_(t-1)^2) + alpha * (|z_(t-1)| - E|z|) + gamma * z_(t-1)

    Where `z = u / sigma` is the standardized shock and `E|z| = sqrt(2 / pi)` under the
    assumption of standard normal innovations. A negative gamma indicates the presence
    of a leverage effect (negative shocks raise volatility by more than positive ones).

    For more information about the method, see the following paper:

    - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
    Approach." Econometrica, 59(2), 347-370.

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta. Note that these are used
        for all columns in the returns.
        time_steps (int): Time steps to calculate EGARCH for.
        optimization_t (int): Time steps to optimize EGARCH for. It is only used if no weights are given.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.array | pd.Series | pd.DataFrame: An object with sigma_2 values
    """
    if p != 1 or q != 1:
        raise ValueError(
            "Invalid input for p or/and q, currently only EGARCH(1, 1) is implemented."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_egarch,
                    weights=weights,
                    time_steps=time_steps,
                    optimization_t=optimization_t,
                    p=p,
                    q=q,
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            egarch = pd.concat(period_data_list, axis=0)

            return egarch
        return returns.aggregate(
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

        omega, alpha, gamma, beta = weights
        expected_absolute_z = np.sqrt(2 / np.pi)

        log_sigma2 = np.zeros(time_steps)
        sigma2 = np.zeros(time_steps)
        log_sigma2[0] = np.log(returns[0] ** 2 + 1e-12)
        sigma2[0] = np.exp(log_sigma2[0])

        for i in range(1, time_steps):
            standardized_shock = returns[i - 1] / np.sqrt(sigma2[i - 1])
            log_sigma2[i] = (
                omega
                + beta * log_sigma2[i - 1]
                + alpha * (abs(standardized_shock) - expected_absolute_z)
                + gamma * standardized_shock
            )
            # Clip to a generous but finite range. Without this, a simulated-annealing
            # exploration step that pushes the recursion toward overflow can make the
            # (otherwise well-behaved) likelihood spuriously attractive at extreme
            # parameter values, pulling the optimizer toward the edge of its bounds.
            log_sigma2[i] = np.clip(log_sigma2[i], -20, 20)
            sigma2[i] = np.exp(log_sigma2[i])

        return sigma2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")


def get_egarch_forecast(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    weights: list | None = None,
    time_steps: int = 10,
    p: int = 1,
    q: int = 1,
):
    """Calculates sigma_2 forecasts based on the EGARCH model.

    Since EGARCH is nonlinear in log-variance, the multi-step-ahead forecast is
    approximated by forecasting the log-variance (whose forecast has a closed form,
    since the alpha and gamma terms have zero expectation under standard normal
    innovations) and then exponentiating:

    - E[ln(sigma_(t+h)^2)] = long_run_log_variance + beta^(h - 1) * (current_log_variance - long_run_log_variance)
    - sigma_(t+h)^2 ~= exp(E[ln(sigma_(t+h)^2)])

    Where `long_run_log_variance = omega / (1 - beta)`. This is the standard practical
    approximation (it understates the true forecast somewhat due to Jensen's
    inequality, since E[exp(X)] != exp(E[X])), not an exact closed form.

    For more information about the method, see the following paper:

    - Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
    Approach." Econometrica, 59(2), 347-370.

    Args:
        returns (pd.Series | pd.DataFrame | np.ndarray): A Series or Dataframe or np.ndarray of returns.
        weights (list): List with the values for omega, alpha, gamma and beta.
        time_steps (int): Time steps to calculate EGARCH for.
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
                period_data = returns.loc[sub_period].aggregate(
                    get_egarch_forecast,
                    weights=weights,
                    time_steps=time_steps,
                    p=p,
                    q=q,
                )
                period_data.name = sub_period
                period_data.columns = [
                    col + " " + str(sub_period) for col in period_data.columns
                ]

                if not period_data.empty:
                    period_data_list.append(period_data)

            egarch_forecast = pd.concat(period_data_list, axis=1)

            return egarch_forecast

        return returns.aggregate(
            get_egarch_forecast, weights=weights, time_steps=time_steps, p=p, q=q
        )
    if isinstance(returns, pd.Series):
        return get_egarch_forecast(
            returns=returns.values, weights=weights, time_steps=time_steps, p=p, q=q
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            weights = get_egarch_weights(returns, p=p, q=q)

        garch_values = get_egarch(returns, weights, len(returns), p=p, q=q)

        omega, _, _, beta = weights
        current_log_variance = np.log(garch_values[-1])
        long_run_log_variance = omega / (1 - beta)

        sigma_2 = np.zeros(time_steps)
        for i in range(time_steps):
            log_forecast = long_run_log_variance + (beta**i) * (
                current_log_variance - long_run_log_variance
            )
            sigma_2[i] = np.exp(log_forecast)

        return sigma_2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")
