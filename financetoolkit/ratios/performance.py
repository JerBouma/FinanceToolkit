"""Performanc Model"""

import pandas as pd


def get_sharpe_ratio(
    excess_returns: pd.Series, excess_volatility: pd.Series
) -> pd.Series:
    """
    Calculate the Sharpe ratio of returns.

    Args:
        excess_returns (pd.Series): A Series of returns with risk-free rate subtracted.
        excess_volatility (pd.Series): A Series of volatility numbers based on the excess returns.

    Returns:
        pd.Series: A Series of Sharpe ratios with time as index and assets as columns.
    """
    return excess_returns / excess_volatility
