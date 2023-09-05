"""Performance Model"""

import warnings

import numpy as np
import pandas as pd

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2


def get_covariance(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the covariance of returns.

    A warnings filter is included given that the following error
    can occur:

    RuntimeWarning: Degrees of freedom <= 0 for slice
        return np.cov(a, b, ddof=ddof)[0, 1]

    Given that this is due to division by zero or NaN values, it does
    not have any impact on the result. The warning is therefore
    ignored.

    Args:
        returns (pd.Series | pd.DataFrame): _description_
        benchmark_returns (pd.Series | pd.DataFrame): _description_

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(returns, pd.DataFrame):
            covariance = pd.Series(index=returns.columns)

            for column in returns.columns:
                covariance.loc[column] = returns[column].cov(benchmark_returns)

            return covariance

        if isinstance(returns, pd.Series | pd.DataFrame.rolling | pd.Series.rolling):
            return returns.cov(benchmark_returns)

    return returns.cov(benchmark_returns)


def get_beta(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """_summary_

    Args:
        returns (pd.Series | pd.DataFrame): _description_
        benchmark_returns (pd.Series | pd.DataFrame): _description_

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            combination = pd.concat([returns, benchmark_returns], axis=1)

            # Calculate Sharpe ratio for each asset (ticker) in the DataFrame
            covariance = combination.groupby(level=0).apply(
                lambda x: get_covariance(x[returns.columns], x[benchmark_returns.name])
            )
            variance = benchmark_returns.groupby(level=0).apply(lambda x: x.var())

            return covariance.div(variance, axis=0)

        return get_covariance(returns, benchmark_returns) / benchmark_returns.var()

    if isinstance(returns, pd.Series):
        # Calculate Sharpe ratio for a single asset (ticker)
        return get_covariance(returns, benchmark_returns) / benchmark_returns.var()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_beta(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate rolling beta.

    Args:
        returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        window_size (int): Rolling window size.

    Returns:
        pd.Series | pd.DataFrame: Rolling beta values.
    """
    rolling_cov = pd.DataFrame(columns=returns.columns)

    for column in returns.columns:
        rolling_cov.loc[:, column] = get_covariance(
            returns[column].rolling(window=window_size), benchmark_returns
        )

    rolling_var = benchmark_returns.rolling(window=window_size).var()

    rolling_beta = rolling_cov.div(rolling_var, axis=0)

    return rolling_beta


def get_capital_asset_pricing_model(
    risk_free_rate: pd.Series | pd.DataFrame,
    beta: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series | pd.DataFrame:
    """_summary_

    Args:
        returns (pd.Series | pd.DataFrame): _description_
        benchmark_returns (pd.Series | pd.DataFrame): _description_

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    # Slightly different order with same result: Beta * (Benchmark - Risk Free) + Risk Free
    capital_asset_pricing_model = beta.multiply(
        (benchmark_returns - risk_free_rate), axis=0
    ).add(risk_free_rate, axis=0)

    return capital_asset_pricing_model


def get_sharpe_ratio(excess_returns: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Calculate the Sharpe ratio of returns.

    Args:
        excess_returns (pd.Series): A Series of returns with risk-free rate subtracted.

    Returns:
        pd.Series: A Series of Sharpe ratios with time as index and assets as columns.
    """
    if isinstance(excess_returns, pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # Calculate Sharpe ratio for each asset (ticker) in the DataFrame
            sharpe_ratios = excess_returns.groupby(level=0).apply(
                lambda x: x.mean() / x.std()
            )
            return sharpe_ratios

        return excess_returns / excess_returns.std()

    if isinstance(excess_returns, pd.Series):
        # Calculate Sharpe ratio for a single asset (ticker)
        return excess_returns / excess_returns.std()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_sharpe_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    window_size: int,
) -> pd.Series:
    """
    Calculate the rolling Sharpe ratio of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free rate subtracted.
        window_size (int): The size of the rolling window in months. Default is 12.

    Returns:
        pd.Series: A Series of rolling Sharpe ratios with time as index and assets as columns.
    """
    sharpe_ratio = (
        excess_returns.rolling(window=window_size).mean()
        / excess_returns.rolling(window=window_size).std()
    )

    return sharpe_ratio


def get_sortino_ratio(excess_returns: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Calculate the Sortino ratio of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free
        rate already subtracted.

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """
    if isinstance(excess_returns, pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # Calculate Sortino ratio for each asset (ticker) in the DataFrame
            sortino_ratios = excess_returns.groupby(level=0).apply(
                lambda x: x.mean() / x[x < 0].std()
            )
            return sortino_ratios

        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = downside_returns.std()

        return excess_returns.mean() / downside_volatility

    if isinstance(excess_returns, pd.Series):
        # Calculate Sortino ratio for a single asset (ticker)
        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = downside_returns.std()
        return excess_returns.mean() / downside_volatility

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


def get_treynor_ratio(excess_returns: pd.Series, beta: float) -> pd.Series:
    """
    Calculate the Treynor ratio of returns.

    Args:
        excess_returns (pd.Series): A Series of returns with risk-free rate subtracted.
        beta (float): The portfolio's beta (systematic risk).

    Returns:
        pd.Series: A Series of Treynor ratios with time as index and assets as columns.
    """
    return excess_returns / beta


def get_calmar_ratio(annualized_returns: pd.Series, max_drawdown: float) -> pd.Series:
    """
    Calculate the Calmar ratio of returns.

    Args:
        annualized_returns (pd.Series): A Series of annualized returns.
        max_drawdown (float): The maximum drawdown of the portfolio.

    Returns:
        pd.Series: A Series of Calmar ratios with time as index and assets as columns.
    """
    return annualized_returns / abs(max_drawdown)


def get_upside_capture_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate the Upside Capture Ratio.

    Args:
        portfolio_returns (pd.Series): A Series of portfolio returns.
        benchmark_returns (pd.Series): A Series of benchmark returns.

    Returns:
        pd.Series: A Series of Upside Capture Ratios with time as index.
    """
    upside_returns = np.maximum(portfolio_returns, 0)
    upside_capture = np.sum(upside_returns) / np.sum(benchmark_returns)
    return upside_capture


def get_downside_capture_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate the Downside Capture Ratio.

    Args:
        portfolio_returns (pd.Series): A Series of portfolio returns.
        benchmark_returns (pd.Series): A Series of benchmark returns.

    Returns:
        pd.Series: A Series of Downside Capture Ratios with time as index.
    """
    downside_returns = np.minimum(portfolio_returns, 0)
    downside_capture = np.sum(downside_returns) / np.sum(benchmark_returns)
    return downside_capture
