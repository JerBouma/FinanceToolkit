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
    Calculate the True Range (TR) of a given price series.

    The True Range is the greatest of three measures of a single period's price movement,
    designed to capture gaps between periods (e.g. an overnight gap) that a plain
    High — Low range would miss: the current high-low range, the distance from the prior
    close to the current high, and the distance from the prior close to the current low.
    It is the building block for the Average True Range and the Average Directional Index.

    The formula is a follows:

    - True Range = Max[(High — Low), |High — Previous Close|, |Low — Previous Close|]

    Also known as: TR.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: True Range values.
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


def get_wilder_moving_average(
    values: pd.Series | pd.DataFrame, window: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate Wilder's Smoothed Moving Average of a given series.

    Wilder's smoothing is the specific recursive smoothing method J. Welles Wilder Jr. used
    for every one of his original indicators (the Relative Strength Index, the Average True
    Range and the Average Directional Index). It is related to an Exponential Moving Average
    but uses a slower smoothing constant of 1/window (equivalent to an EMA with
    span = 2 * window — 1), rather than the standard EMA constant of 2/(window+1). Using a
    plain Simple Moving Average, or a standard EMA, instead of Wilder's specific smoothing
    is one of the most common sources of numerical discrepancies when re-implementing
    Wilder's indicators.

    The formula is a follows:

    - First value = SMA(values, window) — a simple average seeds the recursion
    - Value(t) = Value(t-1) + (1 / window) * (Series(t) — Value(t-1)), for every period after

    Also known as: Wilder's Moving Average, Modified Moving Average, Running Moving Average,
    Smoothed Moving Average.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Args:
        values (pd.Series | pd.DataFrame): Series (or DataFrame with one column per ticker)
            of values to smooth (e.g. True Range, +DM, -DM or DX).
        window (int): Number of periods for the smoothing.

    Returns:
        pd.Series | pd.DataFrame: Wilder's Smoothed Moving Average values.
    """
    if isinstance(values, pd.DataFrame):
        return pd.concat(
            {
                column: get_wilder_moving_average(values[column], window)
                for column in values.columns
            },
            axis=1,
        )

    wilder_average = pd.Series(index=values.index, dtype="float64")

    length = len(values)
    if length == 0:
        return wilder_average

    seed = values.rolling(window=window).mean()
    first_valid_index = seed.first_valid_index()
    if first_valid_index is None:
        return wilder_average

    start_position = values.index.get_loc(first_valid_index)
    wilder_average.iloc[start_position] = seed.iloc[start_position]

    for i in range(start_position + 1, length):
        current_value = values.iloc[i]
        previous_average = wilder_average.iloc[i - 1]

        if pd.isna(current_value):
            wilder_average.iloc[i] = previous_average
            continue

        wilder_average.iloc[i] = previous_average + (1 / window) * (
            current_value - previous_average
        )

    return wilder_average


def get_average_true_range(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Average True Range (ATR) of a given price series.

    The Average True Range measures volatility by smoothing the True Range (a single
    period's price movement that also accounts for gaps between periods) with Wilder's
    smoothing method, rather than a plain moving average.

    The formula is a follows:

    - True Range = Max[(High — Low), |High — Previous Close|, |Low — Previous Close|]
    - ATR = Wilder's Smoothed Moving Average of True Range over `window` periods

    Also known as: ATR.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for ATR calculation.

    Returns:
        pd.Series: ATR values.
    """
    true_range = get_true_range(prices_high, prices_low, prices_close)

    atr = get_wilder_moving_average(true_range, window)

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

        # The bands are NaN until the Average True Range has a full window, so the first
        # bar with valid bands seeds the recursion. Comparing against the NaN carried in
        # from the previous bar would fail both branches and propagate NaN indefinitely.
        if pd.isna(final_upper_band.iloc[i - 1]) or pd.isna(
            final_lower_band.iloc[i - 1]
        ):
            final_upper_band.iloc[i] = basic_upper_band.iloc[i]
            final_lower_band.iloc[i] = basic_lower_band.iloc[i]
            trend_direction.iloc[i] = 1
            supertrend.iloc[i] = final_lower_band.iloc[i]
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

    Keltner Channels plot a volatility-based envelope around an Exponential Moving Average of
    price, with the envelope width set by a multiple of the Average True Range rather than a
    fixed percentage or a standard deviation. This makes the channel width adapt to the
    asset's own volatility regime, similar in spirit to Bollinger Bands but based on true
    range instead of standard deviation.

    The formula is a follows:

    - Middle Line = EMA(Close, window)
    - Upper Line = Middle Line + atr_multiplier * ATR(atr_window)
    - Lower Line = Middle Line — atr_multiplier * ATR(atr_window)

    Also known as: Keltner Bands.

    Reference: Keltner, C.W. (1960). "How to Make Money in Commodities." The Keltner
    Statistical Service. The commonly used modern variant (EMA midline with an ATR-based
    envelope, as implemented here) was popularized by Linda Bradford Raschke in the 1980s.

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

    Donchian Channels plot the highest high and lowest low over the `window` periods
    *preceding* the current one, with the middle line being the average of the two. They
    are used to identify breakouts — Donchian's original rule buys when price exceeds the
    highest high of the preceding N periods — and to gauge the overall volatility of the
    price range.

    The formula is a follows:

    - Upper Channel = Max(High, window), ending one period before the current period
    - Lower Channel = Min(Low, window), ending one period before the current period
    - Middle Channel = (Upper Channel + Lower Channel) / 2

    Also known as: Donchian Bands, price channel.

    Reference: Donchian, R.D. (1960s). Popularized through his "5- and 20-Day Moving Average
    Trading Rule" commodity trading system; the channel itself is described in Kaufman, P.J.
    (2013). "Trading Systems and Methods." 5th ed. Wiley.

    Notes:
        - The current period is deliberately excluded from the lookback. StockCharts states
          the rule directly: "The Price Channel formula doesn't include the most recent
          period. Price Channels are based on prices prior to the current period... A
          channel break would not be possible if the most recent period was used." Including
          the current bar makes `High > Upper Channel` impossible by construction, which
          silently disables the indicator's primary use as a breakout signal.
        - Because the channel is built entirely from prior periods, it uses no information
          from the current bar and is safe to compare the current bar's price against.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods, ending one period ago, to consider for the
            Donchian Channels.

    Returns:
        pd.DataFrame: Donchian Channels (upper, middle, lower).
    """
    upper_channel = prices_high.rolling(window=window).max().shift(1)
    lower_channel = prices_low.rolling(window=window).min().shift(1)
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

    The formula is a follows:

    - Log Return(t) = ln(Close(t) / Close(t-1))
    - Realized Volatility(window) = StdDev(Log Return, window) * sqrt(252)
    - For each window, the cone reports the minimum, the 10th, 25th, 50th, 75th and 90th
      percentiles, the maximum and the most recent value of that realized volatility series

    Also known as: volatility cone, realized volatility cone.

    Reference: Burghardt, G. & Lane, M. (1990). "How to Tell if Options are Cheap." Journal
    of Portfolio Management, and the related treatment in Natenberg, S. (1994). "Option
    Volatility and Pricing." McGraw-Hill — the volatility cone is a standard tool in options
    trading practice for comparing realized volatility across lookback windows.

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

    Bollinger Bands plot an envelope around a Simple Moving Average of price, with the
    envelope width set to a multiple of the rolling standard deviation of price over the
    same window. Because the bands widen and narrow with the security's own volatility,
    they are commonly used to gauge whether price is relatively high or low versus its
    recent range, and periods of unusually narrow bands ("the squeeze") are read as a sign
    that a volatility expansion may be imminent.

    The formula is a follows:

    - Middle Band = SMA(Close, window)
    - Upper Band = Middle Band + num_std_dev * StdDev(Close, window)
    - Lower Band = Middle Band — num_std_dev * StdDev(Close, window)

    Also known as: Bollinger Bands.

    Reference: Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.

    Notes:
        - The rolling standard deviation uses the *population* convention (dividing by n,
          i.e. `ddof=0`), not pandas' default sample convention (dividing by n-1). This is
          what Bollinger himself specifies — "(We use the population calculation for
          standard deviation)", bollingerbands.com — and matches both TA-Lib (`ta_VAR.c`
          divides by the period) and StockCharts. For the default 20-period window the
          sample convention inflates the band half-width by a factor of sqrt(20/19), about
          2.6%.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods for the moving average.
        num_std_dev (int): Number of standard deviations for the bands.

    Returns:
        pd.DataFrame: Bollinger Bands (upper, middle, lower).
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std(ddof=0)

    upper_band = rolling_mean + (num_std_dev * rolling_std)
    lower_band = rolling_mean - (num_std_dev * rolling_std)

    return pd.concat(
        [upper_band, rolling_mean, lower_band, prices],
        axis=1,
        keys=["Upper Band", "Middle Band", "Lower Band", "Close"],
    )
