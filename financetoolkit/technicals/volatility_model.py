"""Volatility Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.technicals.overlap_model import get_exponential_moving_average

# Number of trading days used to annualize realized volatility.
TRADING_DAYS_PER_YEAR = 252


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


def get_donchian_channels(
    prices_high: pd.Series, prices_low: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate the Donchian Channels of a given price series.

    Donchian Channels plot the highest high and lowest low over a specified window,
    with the middle line being the average of the two. They are used to identify
    breakouts and the overall volatility of the price range.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods to consider for the Donchian Channels.

    Returns:
        pd.DataFrame: Donchian Channels (upper, middle, lower).
    """
    upper_channel = prices_high.rolling(window=window).max()
    lower_channel = prices_low.rolling(window=window).min()
    middle_channel = (upper_channel + lower_channel) / 2

    return pd.concat(
        [upper_channel, middle_channel, lower_channel],
        keys=["Upper Channel", "Middle Channel", "Lower Channel"],
        axis=1,
    )


def get_volatility_cone(
    prices_close: pd.Series, windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Calculate the Volatility Cone of a given price series.

    The Volatility Cone summarizes the distribution of historical annualized realized
    volatility over a range of rolling windows, showing how the current realized
    volatility for each window compares to its own historical range. It is commonly
    used to judge whether current (or implied) volatility is cheap or expensive
    relative to history.

    Args:
        prices_close (pd.Series): Series of closing prices.
        windows (list[int] | None): The rolling windows (in periods) to calculate
            realized volatility for. Defaults to [10, 20, 30, 60, 90, 120].

    Returns:
        pd.DataFrame: Volatility Cone with, for each window, the historical minimum,
            10th, 25th, 50th (median), 75th and 90th percentiles, maximum and the
            current realized volatility.
    """
    if windows is None:
        windows = [10, 20, 30, 60, 90, 120]

    log_returns = np.log(prices_close / prices_close.shift(1))

    volatility_cone = {}
    for window in windows:
        realized_volatility = log_returns.rolling(window=window).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

        volatility_cone[window] = {
            "Min": realized_volatility.min(),
            "10th Percentile": realized_volatility.quantile(0.10),
            "25th Percentile": realized_volatility.quantile(0.25),
            "Median": realized_volatility.median(),
            "75th Percentile": realized_volatility.quantile(0.75),
            "90th Percentile": realized_volatility.quantile(0.90),
            "Max": realized_volatility.max(),
            "Current": (
                realized_volatility.iloc[-1]
                if not realized_volatility.empty
                else np.nan
            ),
        }

    volatility_cone_df = pd.DataFrame(volatility_cone).T
    volatility_cone_df.index.name = "Window"

    return volatility_cone_df


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
