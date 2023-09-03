"""Risk Model"""

import pandas as pd
import numpy as np
from scipy import stats
from financetoolkit.base.helpers import skewness, kurtosis


def get_var_historic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the historical Value at Risk (VaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame | float: VaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == 1:
            return returns.aggregate(get_var_historic, alpha=alpha)
        else:
            periods = returns.index.get_level_values(0).unique()
            value_at_risk = pd.DataFrame()

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_var_historic, alpha=alpha
                )
                period_data.name = sub_period

                value_at_risk = pd.concat([value_at_risk, period_data], axis=1)
            return value_at_risk.T
    elif isinstance(returns, pd.Series):
        return np.percentile(
            returns, alpha * 100
        )  # The actual calculation without data wrangling
    else:
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_var_gaussian(
    returns, alpha: float, cornish_fisher: bool = False
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Value at Risk (VaR) of returns based on the gaussian distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        cornish_fisher (bool): Whether to adjust the distriution for the skew and kurtosis of the returns based on the Cornish-Fischer quantile expansion. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame | float: VaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame) and returns.index.nlevels > 1:
        periods = returns.index.get_level_values(0).unique()
        value_at_risk = pd.DataFrame()
        for sub_period in periods:
            period_data = get_var_gaussian(
                returns.loc[sub_period], alpha, cornish_fisher
            )
            period_data.name = sub_period

            value_at_risk = pd.concat([value_at_risk, period_data], axis=1)

        return value_at_risk.T
    else:
        za = stats.norm.ppf(alpha, 0, 1)

        if cornish_fisher:
            # Adjust za according to the Cornish-Fischer expansion of the quantiles.
            # Formula for quantile from "Finance Compact Plus" by Zimmerman; Part 1, page 130-131
            # More material/resources:
            #     - "Numerical Methods and Optimization in Finance" by Gilli, Maringer & Schumann;
            #     - https://www.value-at-risk.net/the-cornish-fisher-expansion/;
            #     - https://www.diva-portal.org/smash/get/diva2:442078/FULLTEXT01.pdf, Section 2.4.2, p.18;
            #     - "Risk Management and Financial Institutions" by John C. Hull
            S = skewness(returns)
            K = kurtosis(returns)
            za = (
                za
                + (za**2 - 1) * S / 6
                + (za**3 - 3 * za) * (K - 3) / 24
                - (2 * za**3 - 5 * za) * (S**2) / 36
            )

        return returns.mean() + za * returns.std(ddof=0)


def get_var_studentt(returns, alpha: float) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Value at Risk (VaR) of returns based on the Student-T distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame | float: VaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame) and returns.index.nlevels > 1:
        periods = returns.index.get_level_values(0).unique()
        value_at_risk = pd.DataFrame()
        for sub_period in periods:
            period_data = get_var_studentt(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            value_at_risk = pd.concat([value_at_risk, period_data], axis=1)

        return value_at_risk.T
    else:
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
        pd.Series | pd.DataFrame | float: CVaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == 1:
            return returns.aggregate(get_cvar_historic, alpha=alpha)
        else:
            periods = returns.index.get_level_values(0).unique()
            value_at_risk = pd.DataFrame()

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_cvar_historic, alpha=alpha
                )
                period_data.name = sub_period

                value_at_risk = pd.concat([value_at_risk, period_data], axis=1)
            return value_at_risk.T
    elif isinstance(returns, pd.Series):
        return returns[
            returns <= get_var_historic(returns, alpha)
        ].mean()  # The actual calculation without data wrangling
    else:
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_cvar_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the gaussian distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame | float: CVaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame) and returns.index.nlevels > 1:
        periods = returns.index.get_level_values(0).unique()
        value_at_risk = pd.DataFrame()
        for sub_period in periods:
            period_data = get_cvar_gaussian(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            value_at_risk = pd.concat([value_at_risk, period_data], axis=1)

        return value_at_risk.T
    else:
        za = stats.norm.ppf(alpha, 0, 1)
        return returns.std(ddof=0) * -stats.norm.pdf(za) / alpha + returns.mean()


def get_cvar_studentt(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the Student-T distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame | float: CVaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame) and returns.index.nlevels > 1:
        periods = returns.index.get_level_values(0).unique()
        value_at_risk = pd.DataFrame()
        for sub_period in periods:
            period_data = get_cvar_studentt(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            value_at_risk = pd.concat([value_at_risk, period_data], axis=1)

        return value_at_risk.T
    else:
        # Fitting student-t parameters to the data
        v, scale = np.array([]), np.array([])
        for col in returns.columns:
            col_v, _, col_scale = stats.t.fit(returns[col])
            v = np.append(v, col_v)
            scale = np.append(scale, col_scale)
        za = stats.t.ppf(1 - alpha, v, 1)

        return (
            -scale * (v + za**2) / (v - 1) * stats.t.pdf(za, v) / alpha
            + returns.mean()
        )


def get_cvar_laplace(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the Laplace distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence). Note that `alpha` needs to be below 0.5.

    Returns:
        pd.Series | pd.DataFrame | float: CVaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame) and returns.index.nlevels > 1:
        periods = returns.index.get_level_values(0).unique()
        value_at_risk = pd.DataFrame()
        for sub_period in periods:
            period_data = get_cvar_laplace(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            value_at_risk = pd.concat([value_at_risk, period_data], axis=1)

        return value_at_risk.T
    else:
        # For formula see: https://en.wikipedia.org/wiki/Expected_shortfall#Laplace_distribution

        # Fitting b (scale parameter) to the variance of the data
        # Since variance of the Laplace dist.: var = 2*b**2
        b = np.sqrt(returns.std(ddof=0) ** 2 / 2)

        if alpha <= 0.5:
            return -b * (1 - np.log(2 * alpha)) + returns.mean()
        else:
            print("Laplace Conditional VaR is not available for a level over 50%.")
            return 0


def get_cvar_logistic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the logistic distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame | float: CVaR values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame) and returns.index.nlevels > 1:
        periods = returns.index.get_level_values(0).unique()
        value_at_risk = pd.DataFrame()
        for sub_period in periods:
            period_data = get_cvar_logistic(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            value_at_risk = pd.concat([value_at_risk, period_data], axis=1)

        return value_at_risk.T
    else:
        # For formula see: https://en.wikipedia.org/wiki/Expected_shortfall#Logistic_distribution

        # Fitting b (scale parameter) to the variance of the data
        # Since variance of the Logistic dist.: var = b**2*pi**2/3
        scale = np.sqrt(3 * returns.std(ddof=0) ** 2 / np.pi**2)

        return (
            -scale * np.log(((1 - alpha) ** (1 - 1 / alpha)) / alpha) + returns.mean()
        )


def get_max_drawdown(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Maximum Drawdown (MDD) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame | float: MDD values as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame) and returns.index.nlevels > 1:
        periods = returns.index.get_level_values(0).unique()
        max_drawdown = pd.DataFrame()
        for sub_period in periods:
            period_data = get_max_drawdown(returns.loc[sub_period])
            period_data.name = sub_period

            max_drawdown = pd.concat([max_drawdown, period_data], axis=1)

        return max_drawdown.T

    cum_returns = (1 + returns).cumprod()

    return (cum_returns / cum_returns.cummax() - 1).min()
