"""Overlap Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


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


def get_support_resistance_levels(
    prices: pd.Series, window: int = 5, sensitivity: float = 0.05
):
    """
    Calculate support and resistance levels from historical price data.

    Parameters:
        prices (pd.Series): A pandas Series of historical closing prices.
        window (int): The window size to use for identifying local maxima and minima.
        sensitivity (float): The sensitivity threshold for identifying levels.

    Returns:
        support_resistance_levels (pd.DataFrame): A DataFrame with support and resistance levels.
            The DataFrame has two columns: "Resistance" and "Support".
            The index of the DataFrame represents the dates, and the values represent the levels.
    """
    # Identify local maxima and minima
    local_maxima_indices = argrelextrema(prices.values, np.greater, order=window)[0]
    local_minima_indices = argrelextrema(prices.values, np.less, order=window)[0]

    local_maxima_prices = prices.iloc[local_maxima_indices]
    local_minima_prices = prices.iloc[local_minima_indices]

    # Initialize dictionaries for support and resistance levels
    resistance_levels: dict[pd.PeriodIndex, float] = {}
    support_levels: dict[pd.PeriodIndex, float] = {}

    # Calculate resistance levels
    for idx, price in zip(local_maxima_indices, local_maxima_prices):
        if not resistance_levels:
            resistance_levels[prices.index[idx]] = price
        else:
            close_to_existing = False
            for date, level in resistance_levels.items():
                if abs(price - level) / level < sensitivity:
                    resistance_levels[date] = (resistance_levels[date] + price) / 2
                    close_to_existing = True
                    break
            if not close_to_existing:
                resistance_levels[prices.index[idx]] = price

    # Calculate support levels
    for idx, price in zip(local_minima_indices, local_minima_prices):
        if not support_levels:
            support_levels[prices.index[idx]] = price
        else:
            close_to_existing = False
            for date, level in support_levels.items():
                if abs(price - level) / level < sensitivity:
                    support_levels[date] = (support_levels[date] + price) / 2
                    close_to_existing = True
                    break
            if not close_to_existing:
                support_levels[prices.index[idx]] = price

    support_resistance_levels = pd.DataFrame(
        {
            "Resistance": pd.Series(resistance_levels),
            "Support": pd.Series(support_levels),
        }
    ).sort_index()

    return support_resistance_levels
