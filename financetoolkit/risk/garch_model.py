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

    # Compute the log-likelihood in a vectorized manner
    u = returns[1:t]
    v = garch[: t - 1]

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

    # Perform the optimization using simulated annealing
    result = optimize.dual_annealing(wrapper_func, bounds, x0=initial_guess)

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
        t (int): Time steps to calculate GARCH for.
        optimization_t (int): Time steps to optimize GRACH for. It is only used if no weights are given.
        p (int): Number of u_t datapoints to use. Note that currently only p=1 is supported.
        q: (int): Number of sigma_t datapoints to use. Note that currently only q=1 is supported.

    Returns:
        np.array | pd.Series | pd.DataFrame: A object with sigma_2 values
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
                period_data = returns.loc[sub_period].aggregate(get_garch)
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            garch = pd.concat(period_data_list, axis=0)

            return garch
        return returns.aggregate(get_garch)
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
    sigma_l ** 2 + (sigma_t ** 2 - sigma_l ** 2) * (alpha + beta) ** (t - 1)

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
                period_data = returns.loc[sub_period].aggregate(get_garch_forecast)
                period_data.name = sub_period
                period_data.columns = [
                    col + " " + str(sub_period) for col in period_data.columns
                ]

                if not period_data.empty:
                    period_data_list.append(period_data)

            garch_forecast = pd.concat(period_data_list, axis=1)

            return garch_forecast

        return returns.aggregate(get_garch_forecast)
    if isinstance(returns, pd.Series):
        return get_garch_forecast(
            returns=returns.values, weights=weights, time_steps=time_steps, p=p, q=q
        )
    if isinstance(returns, np.ndarray):
        if weights is None:
            weights = get_garch_weights(returns, p=p, q=q)

        garch_values = get_garch(returns, weights, time_steps, p=p, q=q)

        sigma_l = weights[0] / (1 - weights[1] - weights[2])
        sigma_2 = np.zeros(time_steps)
        sigma_2[0] = garch_values[0]
        for i in range(1, time_steps):
            sigma_2[i] = sigma_l**2 + (garch_values[0] - sigma_l**2) * (
                weights[1] + weights[2]
            ) ** (i - 1)

        return sigma_2

    raise TypeError("Expects pd.DataFrame or pd.Series or np.ndarry, no other value.")
