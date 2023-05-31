"""Price Module"""

import pandas as pd


def get_returns(historical_data: pd.Series) -> pd.Series:
    """
    Calculate the returns of historical data for a given Series, with an option to include a rolling period.

    Args:
        historical_data (pd.Series): Series containing historical data.

    Returns:
        pd.Series: Series containing the calculated returns.
    """
    return historical_data.pct_change()


def get_volatility(returns: pd.Series) -> pd.Series:
    """
    Calculate the volatility of returns over a rolling window.

    Args:
        returns (pd.Series): A Series of returns with time as index
        rolling (int): The size of the rolling window.

    Returns:
        pd.Series: A Series of volatility with time as index and assets as columns.
    """
    return returns.std()


def get_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float | pd.Series
) -> pd.Series:
    """
    Calculate the Sharpe ratio of returns over a rolling window.

    Args:
        returns (pd.Series): A Series of returns with time as index
        risk_free_rate (float or pd.Series): The annualized risk-free rate.

    Returns:
        pd.Series: A Series of Sharpe ratios with time as index and assets as columns.
    """
    excess_returns = returns - risk_free_rate

    return excess_returns / returns.std()


def get_sortino_ratio(
    returns: pd.Series, risk_free_rate: float | pd.Series
) -> pd.Series:
    """
    Calculate the Sortino ratio of returns over a rolling window.

    Args:
        returns (pd.Series): A Series of returns with time as index
        risk_free_rate (float or pd.Series): The annualized risk-free rate.
        rolling_window (int): The size of the rolling window.

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """
    excess_returns = returns - risk_free_rate
    downside_volatility = excess_returns[excess_returns < 0].std()

    return excess_returns / downside_volatility


def get_beta(returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculate the beta of returns with respect to a benchmark over a rolling window.

    Args:
        returns (pd.DataFrame): A DataFrame of returns with time as index and assets as columns.
        benchmark_returns (pd.DataFrame): A DataFrame of benchmark returns with time as index and a single column.
        rolling_window (int): The size of the rolling window.

    Returns:
        pd.Series: A Series of beta values with time as index and assets as columns.
    """
    excess_returns = returns - returns.mean()
    excess_benchmark_returns = benchmark_returns - benchmark_returns.mean()
    cov = excess_returns.cov(excess_benchmark_returns)
    var = excess_benchmark_returns.var()

    return cov / var
