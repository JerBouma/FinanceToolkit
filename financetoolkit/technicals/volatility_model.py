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


def get_supertrend(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Calculate the Supertrend indicator for a given price series.

    The Supertrend indicator plots a single trailing line that flips between sitting below
    price (in an uptrend) and above price (in a downtrend). The line is built from two bands
    offset from the median price ((High + Low) / 2) by a multiple of the Average True Range,
    which are then "ratcheted" period over period — each band can only move in the direction
    that tightens around price — so that the active band only flips to the other side once
    the closing price actually crosses it. This makes Supertrend both a trend filter (the
    flip direction signals a trend change) and a trailing stop-loss level.

    The formula is a follows:

    - Basic Upper Band = (High + Low) / 2 + multiplier * ATR(window)
    - Basic Lower Band = (High + Low) / 2 — multiplier * ATR(window)
    - Final Upper Band(t) = Basic Upper Band(t) if Basic Upper Band(t) < Final Upper Band(t-1)
      or Close(t-1) > Final Upper Band(t-1), else Final Upper Band(t-1)
    - Final Lower Band(t) = Basic Lower Band(t) if Basic Lower Band(t) > Final Lower Band(t-1)
      or Close(t-1) < Final Lower Band(t-1), else Final Lower Band(t-1)
    - While in an uptrend, Supertrend = Final Lower Band, until Close crosses below it, at
      which point the trend flips to a downtrend and Supertrend = Final Upper Band (and vice
      versa)

    Also known as: Supertrend, SuperTrend.

    Notes:
        - There is no academic journal citation for Supertrend. Like the Parabolic SAR, it is
          a practitioner-developed trailing-stop/trend indicator rather than one derived from
          a published financial paper.
        - The trend is initialized as an uptrend (Trend Direction = 1) on the first available
          period, since there is no prior period to determine the starting direction from.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the underlying Average True Range calculation.
            Defaults to 10.
        multiplier (float): Multiplier applied to the Average True Range to determine how
            far the bands sit from the median price. Defaults to 3.0.

    Returns:
        pd.DataFrame: Supertrend (the trailing indicator line) and Trend Direction (1 for an
            uptrend — Supertrend sits below price — and -1 for a downtrend — Supertrend sits
            above price).

    Raises:
        TypeError: If `prices_high`, `prices_low` or `prices_close` is not a pandas Series.
    """
    if not (
        isinstance(prices_high, pd.Series)
        and isinstance(prices_low, pd.Series)
        and isinstance(prices_close, pd.Series)
    ):
        raise TypeError(
            "prices_high, prices_low and prices_close must be pandas Series."
        )

    average_true_range = get_average_true_range(
        prices_high, prices_low, prices_close, window
    )
    median_price = (prices_high + prices_low) / 2

    basic_upper_band = median_price + multiplier * average_true_range
    basic_lower_band = median_price - multiplier * average_true_range

    length = len(prices_close)

    final_upper_band = pd.Series(index=prices_close.index, dtype="float64")
    final_lower_band = pd.Series(index=prices_close.index, dtype="float64")
    supertrend = pd.Series(index=prices_close.index, dtype="float64")
    trend_direction = pd.Series(index=prices_close.index, dtype="float64")

    if length == 0:
        return pd.concat(
            [supertrend, trend_direction],
            keys=["Supertrend", "Trend Direction"],
            axis=1,
        )

    final_upper_band.iloc[0] = basic_upper_band.iloc[0]
    final_lower_band.iloc[0] = basic_lower_band.iloc[0]
    trend_direction.iloc[0] = 1
    supertrend.iloc[0] = final_lower_band.iloc[0]

    for i in range(1, length):
        if pd.isna(basic_upper_band.iloc[i]) or pd.isna(basic_lower_band.iloc[i]):
            final_upper_band.iloc[i] = final_upper_band.iloc[i - 1]
            final_lower_band.iloc[i] = final_lower_band.iloc[i - 1]
            trend_direction.iloc[i] = trend_direction.iloc[i - 1]
            supertrend.iloc[i] = supertrend.iloc[i - 1]
            continue

        if (
            basic_upper_band.iloc[i] < final_upper_band.iloc[i - 1]
            or prices_close.iloc[i - 1] > final_upper_band.iloc[i - 1]
        ):
            final_upper_band.iloc[i] = basic_upper_band.iloc[i]
        else:
            final_upper_band.iloc[i] = final_upper_band.iloc[i - 1]

        if (
            basic_lower_band.iloc[i] > final_lower_band.iloc[i - 1]
            or prices_close.iloc[i - 1] < final_lower_band.iloc[i - 1]
        ):
            final_lower_band.iloc[i] = basic_lower_band.iloc[i]
        else:
            final_lower_band.iloc[i] = final_lower_band.iloc[i - 1]

        if trend_direction.iloc[i - 1] == 1:
            if prices_close.iloc[i] < final_lower_band.iloc[i]:
                trend_direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper_band.iloc[i]
            else:
                trend_direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower_band.iloc[i]
        elif prices_close.iloc[i] > final_upper_band.iloc[i]:
            trend_direction.iloc[i] = 1
            supertrend.iloc[i] = final_lower_band.iloc[i]
        else:
            trend_direction.iloc[i] = -1
            supertrend.iloc[i] = final_upper_band.iloc[i]

    return pd.concat(
        [supertrend, trend_direction], keys=["Supertrend", "Trend Direction"], axis=1
    )


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
