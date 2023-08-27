"""Profitability Module"""
__docformat__ = "google"

import pandas as pd


def get_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Moving Average (MA) of a given price series.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for the moving average.

    Returns:
        pd.Series: Moving Average values.
    """
    return prices.rolling(window=window).mean()


def get_exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) of a given price series.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for EMA calculation.

    Returns:
        pd.Series: EMA values.
    """
    return prices.ewm(span=window, min_periods=1, adjust=False).mean()


def get_double_exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Double Exponential Moving Average (DEMA) of a given price series.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for DEMA calculation.

    Returns:
        pd.Series: DEMA values.
    """
    ema_first = prices.ewm(span=window, min_periods=1, adjust=False).mean()
    ema_second = ema_first.ewm(span=window, min_periods=1, adjust=False).mean()
    dema = 2 * ema_first - ema_second

    return dema


def get_trix(prices_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Trix Indicator for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for Trix calculation.

    Returns:
        pd.Series: Trix Indicator values.
    """
    ema_1 = get_exponential_moving_average(prices_close, window)
    ema_2 = get_exponential_moving_average(ema_1, window)
    ema_3 = get_exponential_moving_average(ema_2, window)

    trix = (ema_3 - ema_3.shift(1)) / ema_3.shift(1) * 100

    return trix


def get_triangular_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Triangular Moving Average (TRIMA) of a given price series.

    The Triangular Moving Average is a type of moving average that provides
    smoothed values by taking an average of the middle values within a specified window.
    It reduces the impact of outliers and short-term fluctuations.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for TRIMA calculation.

    Returns:
        pd.Series: TRIMA values.
    """
    tri_sum = prices.rolling(window=window, min_periods=1).sum()
    tri_ma = tri_sum / ((window + 1) / 2)

    return tri_ma
