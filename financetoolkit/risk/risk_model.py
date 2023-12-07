"""Risk Model"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

ALPHA_CONSTRAINT = 0.5

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2


def get_var_historic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the historical Value at Risk (VaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_var_historic, alpha=alpha
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            value_at_risk = pd.concat(period_data_list, axis=1)

            return value_at_risk.T

        return returns.aggregate(get_var_historic, alpha=alpha)
    if isinstance(returns, pd.Series):
        return np.percentile(
            returns, alpha * 100
        )  # The actual calculation without data wrangling

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_var_gaussian(
    returns, alpha: float, cornish_fisher: bool = False
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Value at Risk (VaR) of returns based on the gaussian distribution.

    Adjust za according to the Cornish-Fischer expansion of the quantiles if
    Formula for quantile from "Finance Compact Plus" by Zimmerman; Part 1, page 130-131
    More material/resources:
     - "Numerical Methods and Optimization in Finance" by Gilli, Maringer & Schumann;
     - https://www.value-at-risk.net/the-cornish-fisher-expansion/;
     - https://www.diva-portal.org/smash/get/diva2:442078/FULLTEXT01.pdf, Section 2.4.2, p.18;
     - "Risk Management and Financial Institutions" by John C. Hull

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        cornish_fisher (bool): Whether to adjust the distriution for the skew and kurtosis of the returns
        based on the Cornish-Fischer quantile expansion. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_gaussian(
                returns.loc[sub_period], alpha, cornish_fisher
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    za = stats.norm.ppf(alpha, 0, 1)

    if cornish_fisher:
        S = get_skewness(returns)
        K = get_kurtosis(returns)
        za = (
            za
            + (za**2 - 1) * S / 6
            + (za**3 - 3 * za) * (K - 3) / 24
            - (2 * za**3 - 5 * za) * (S**2) / 36
        )

    return returns.mean() + za * returns.std(ddof=0)


def get_var_studentt(returns, alpha: float) -> pd.Series | pd.DataFrame:
    """
    Calculate the Value at Risk (VaR) of returns based on the Student-T distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_studentt(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # Fitting Student-T parameters to the data
    if isinstance(returns, pd.Series):
        v = np.array([stats.t.fit(returns)[0]])
    else:
        v = np.array([stats.t.fit(returns[col])[0] for col in returns.columns])
    za = stats.t.ppf(alpha, v, 1)

    return np.sqrt((v - 2) / v) * za * returns.std(ddof=0) + returns.mean()


def get_cvar_historic(returns: pd.Series | pd.DataFrame, alpha: float) -> pd.Series:
    """
    Calculate the historical Conditional Value at Risk (CVaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_cvar_historic, alpha=alpha
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            value_at_risk = pd.concat(period_data_list, axis=1)

            return value_at_risk.T
        return returns.aggregate(get_cvar_historic, alpha=alpha)
    if isinstance(returns, pd.Series):
        return returns[
            returns <= get_var_historic(returns, alpha)
        ].mean()  # The actual calculation without data wrangling

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_cvar_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the gaussian distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_gaussian(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    za = stats.norm.ppf(alpha, 0, 1)
    return returns.std(ddof=0) * -stats.norm.pdf(za) / alpha + returns.mean()


def get_cvar_studentt(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the Student-T distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_studentt(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    returns = pd.DataFrame(returns)

    # Fitting student-t parameters to the data
    v, scale = np.array([]), np.array([])
    for col in returns.columns:
        col_v, _, col_scale = stats.t.fit(returns[col])
        v = np.append(v, col_v)
        scale = np.append(scale, col_scale)
    za = stats.t.ppf(1 - alpha, v, 1)

    return (
        -scale * (v + za**2) / (v - 1) * stats.t.pdf(za, v) / alpha + returns.mean()
    )


def get_cvar_laplace(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the Laplace distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence). Note that `alpha` needs to be below 0.5.

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_laplace(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # For formula see: https://en.wikipedia.org/wiki/Expected_shortfall#Laplace_distribution

    # Fitting b (scale parameter) to the variance of the data
    # Since variance of the Laplace dist.: var = 2*b**2
    b = np.sqrt(returns.std(ddof=0) ** 2 / 2)

    if alpha <= ALPHA_CONSTRAINT:
        return -b * (1 - np.log(2 * alpha)) + returns.mean()

    print("Laplace Conditional VaR is not available for a level over 50%.")
    return 0


def get_cvar_logistic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the logistic distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_logistic(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # For formula see: https://en.wikipedia.org/wiki/Expected_shortfall#Logistic_distribution

    # Fitting b (scale parameter) to the variance of the data
    # Since variance of the Logistic dist.: var = b**2*pi**2/3
    scale = np.sqrt(3 * returns.std(ddof=0) ** 2 / np.pi**2)

    return -scale * np.log(((1 - alpha) ** (1 - 1 / alpha)) / alpha) + returns.mean()


def get_evar_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Entropic Value at Risk (EVaR) of returns based on the gaussian distribution.

    For more information see: https://en.wikipedia.org/wiki/Entropic_value_at_risk

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: EVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_laplace(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    return returns.mean() + returns.std(ddof=0) * np.sqrt(
        -2 * np.log(returns.std(ddof=0))
    )


def get_max_drawdown(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Maximum Drawdown (MDD) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame | float: MDD values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_max_drawdown(returns.loc[sub_period])
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        max_drawdown = pd.concat(period_data_list, axis=1)

        return max_drawdown.T

    cum_returns = (1 + returns).cumprod()  # type: ignore

    return (cum_returns / cum_returns.cummax() - 1).min()


def get_ui(
    returns: pd.Series | pd.DataFrame, rolling: int = 14
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Ulcer Index (UI), a measure of downside volatility.

    For more information see:
     - http://www.tangotools.com/ui/ui.htm
     - https://en.wikipedia.org/wiki/Ulcer_index

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        rolling (int, optional): The rolling period to use for the calculation.
        If you select period = 'monthly' and set rolling to 12 you obtain the rolling
        12-month Ulcer Index. If no value is given, then it calculates it for the
        entire period. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: UI values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index, if.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(get_ui)
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            ulcer_index = pd.concat(period_data_list, axis=1)

            return ulcer_index.T

        return returns.aggregate(get_ui)

    if isinstance(returns, pd.Series):
        cumulative_returns = (1 + returns).cumprod()
        cumulative_max = cumulative_returns.rolling(window=rolling).max()
        drawdowns = (cumulative_returns - cumulative_max) / cumulative_max

        ulcer_index_value = np.sqrt((drawdowns**2).mean())

        return ulcer_index_value

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


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


def get_skewness(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Computes the skewness of dataset.

    Args:
        dataset (pd.Series | pd.Dataframe): A single index dataframe or series

    Returns:
        pd.Series | pd.Dataframe: Skewness of the dataset
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(get_skewness)
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            skewness = pd.concat(period_data_list, axis=1)

            return skewness.T
        return returns.aggregate(get_skewness)
    if isinstance(returns, pd.Series):
        return returns.skew()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_kurtosis(
    returns: pd.Series | pd.DataFrame, fisher: bool = True
) -> pd.Series | pd.DataFrame:
    """
    Computes the kurtosis of dataset.

    Args:
        dataset (pd.Series | pd.Dataframe): A single index dataframe or series

    Returns:
        pd.Series | pd.Dataframe: Kurtosis of the dataset
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_kurtosis, fisher=fisher
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            kurtosis = pd.concat(period_data_list, axis=1)

            return kurtosis.T
        return returns.aggregate(get_kurtosis, fisher=fisher)
    if isinstance(returns, pd.Series):
        if fisher:
            return returns.kurtosis()
        return (((returns - returns.mean()) / returns.std(ddof=0)) ** 4).mean()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
