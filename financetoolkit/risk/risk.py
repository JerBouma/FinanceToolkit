"""Risk Model"""

import pandas as pd


def get_var(returns: pd.Series, alpha: float) -> pd.Series:
    """
    Calculate the Value at Risk (VaR) of returns.

    Args:
        returns (pd.Series): A Series of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series: A Series of VaR values with time as index.
    """
    return returns.quantile(alpha)


def get_cvar(returns: pd.Series, alpha: float) -> pd.Series:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns.

    Args:
        returns (pd.Series): A Series of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series: A Series of CVaR values with time as index.
    """
    var = get_var(returns, alpha)
    return returns[returns <= var].mean()


def get_max_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculate the Maximum Drawdown (MDD) of returns.

    Args:
        returns (pd.Series): A Series of returns.

    Returns:
        pd.Series: A Series of MDD values with time as index.
    """
    cum_returns = (1 + returns).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    return max_drawdown
