"""Volatility Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.technicals.overlap_model import get_exponential_moving_average


def get_true_range(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series
) -> pd.Series:
    """
    Calculate the Average True Range (ATR) of a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for ATR calculation.

    Returns:
        pd.Series: ATR values.
    """
    true_range = pd.concat(
        [
            prices_high - prices_low,
            abs(prices_high - prices_close.shift(1)),
            abs(prices_low - prices_close.shift(1)),
        ],
        axis=1,
    ).max(axis=1)

    return true_range


def get_average_true_range(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Average True Range (ATR) of a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for ATR calculation.

    Returns:
        pd.Series: ATR values.
    """
    true_range = get_true_range(prices_high, prices_low, prices_close)

    atr = true_range.rolling(window=window, min_periods=1).mean()

    return atr


def get_keltner_channels(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    atr_window: int,
    atr_multiplier: float,
) -> pd.DataFrame:
    """
    Calculate the Keltner Channels for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the moving average.
        atr_window (int): Number of periods for ATR calculation.
        atr_multiplier (float): Multiplier for ATR to determine channel width.

    Returns:
        pd.DataFrame: Keltner Channels (upper, middle, lower).
    """
    average_true_range = get_average_true_range(
        prices_high, prices_low, prices_close, atr_window
    )
    middle_line = get_exponential_moving_average(prices_close, window)

    upper_line = middle_line + atr_multiplier * average_true_range
    lower_line = middle_line - atr_multiplier * average_true_range

    return pd.concat(
        [upper_line, middle_line, lower_line],
        keys=["Upper Line", "Middle Line", "Lower Line"],
        axis=1,
    )


def get_bollinger_bands(
    prices: pd.Series, window: int, num_std_dev: int
) -> pd.DataFrame:
    """
    Calculate the Bollinger Bands of a given price series.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods for the moving average.
        num_std_dev (int): Number of standard deviations for the bands.

    Returns:
        pd.DataFrame: Bollinger Bands (upper, middle, lower).
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    upper_band = rolling_mean + (num_std_dev * rolling_std)
    lower_band = rolling_mean - (num_std_dev * rolling_std)

    return pd.concat(
        [upper_band, rolling_mean, lower_band, prices],
        axis=1,
        keys=["Upper Band", "Middle Band", "Lower Band", "Close"],
    )
