"""Overlap Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def get_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA) of a given price series.

    The Simple Moving Average is the unweighted arithmetic mean of price over a trailing
    window, used both as a standalone trend-following signal (e.g. price crossing above or
    below the average) and as a building block for many other indicators in this module.

    The formula is a follows:

    - SMA = Mean(Close, window)

    Also known as: SMA, moving average.

    Reference: The academic literature testing moving-average trading rules (rather than
    defining the SMA itself, which predates any single paper) includes Brock, W., Lakonishok,
    J., & LeBaron, B. (1992). "Simple Technical Trading Rules and the Stochastic Properties of
    Stock Returns." Journal of Finance, 47(5), 1731-1764.

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

    The Exponential Moving Average weights recent observations more heavily than older
    ones, with the weight decaying exponentially the further back in time an observation
    is, so that it reacts to new price information faster than a Simple Moving Average of
    the same window while still smoothing out noise.

    The formula is a follows:

    - Smoothing Factor (alpha) = 2 / (window + 1)
    - EMA(t) = alpha * Close(t) + (1 — alpha) * EMA(t-1)

    Also known as: EMA.

    Reference: The exponential smoothing technique underlying the EMA originates in Brown,
    R.G. (1956). "Exponential Smoothing for Predicting Demand." Arthur D. Little Inc.; the
    standard technical-analysis treatment is Murphy, J.J. (1999). "Technical Analysis of the
    Financial Markets." New York Institute of Finance.

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

    DEMA combines a single and a double-smoothed EMA to reduce the lag inherent in moving
    averages while retaining most of their smoothing benefit. Because the second EMA lags
    the first, subtracting it out (after doubling the first) removes much of the delay a
    plain EMA of the same window would have.

    The formula is a follows:

    - EMA1 = EMA(Close, window)
    - EMA2 = EMA(EMA1, window)
    - DEMA = 2 * EMA1 — EMA2

    Also known as: DEMA.

    Reference: Mulloy, P.G. (1994). "Smoothing Data with Faster Moving Averages." Technical
    Analysis of Stocks & Commodities, 12(1).

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

    Trix applies an Exponential Moving Average three times in succession (triple smoothing)
    to filter out short-term price fluctuations and insignificant cycles, then plots the
    percentage rate of change of that triple-smoothed series. Because insignificant
    fluctuations have already been smoothed away, the result oscillates around zero mainly
    in response to genuine trend changes.

    The formula is a follows:

    - EMA1 = EMA(Close, window)
    - EMA2 = EMA(EMA1, window)
    - EMA3 = EMA(EMA2, window)
    - Trix = ((EMA3(t) — EMA3(t-1)) / EMA3(t-1)) * 100

    Also known as: TRIX, Triple Exponential Average.

    Reference: Hutson, J. (1983). "TRIX — Triple Exponential Smoothing Oscillator."
    Technical Analysis of Stocks & Commodities, 1(5).

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

    The Triangular Moving Average is a doubly-smoothed Simple Moving Average: taking an SMA
    of an SMA produces a triangular weighting scheme where the middle observations of the
    combined window carry the most weight and the oldest/newest observations carry the
    least, which reduces the impact of outliers and short-term fluctuations more than a
    single SMA of the same overall length.

    The formula is a follows:

    - Sub-window Length = round((window + 1) / 2)
    - TRIMA = SMA(SMA(Close, Sub-window Length), Sub-window Length)

    Also known as: TRIMA.

    Reference: The standard textbook treatment is Colby, R.W. (2003). "The Encyclopedia of
    Technical Market Indicators." 2nd ed. McGraw-Hill.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for TRIMA calculation.

    Returns:
        pd.Series: TRIMA values.
    """
    sub_window = max(round((window + 1) / 2), 1)

    first_pass = prices.rolling(window=sub_window, min_periods=1).mean()
    tri_ma = first_pass.rolling(window=sub_window, min_periods=1).mean()

    return tri_ma


def get_weighted_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Weighted Moving Average (WMA) of a given price series.

    The Weighted Moving Average assigns a linearly increasing weight to more recent
    prices within the window, making it more responsive to recent price changes than
    a Simple Moving Average while remaining smoother than an Exponential Moving Average.

    The formula is a follows:

    - WMA = Sum(Price(i) * i, i = 1..window) / Sum(i, i = 1..window), where i = 1 is the
      oldest price in the window and i = window is the most recent

    Also known as: WMA, linearly weighted moving average.

    Reference: The standard textbook treatment is Colby, R.W. (2003). "The Encyclopedia of
    Technical Market Indicators." 2nd ed. McGraw-Hill.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for the WMA calculation.

    Returns:
        pd.Series: WMA values.
    """
    weights = np.arange(1, window + 1)

    return prices.rolling(window=window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def get_kaufman_adaptive_moving_average(
    prices: pd.Series,
    window: int = 10,
    fast_window: int = 2,
    slow_window: int = 30,
) -> pd.Series:
    """
    Calculate the Kaufman Adaptive Moving Average (KAMA) of a given price series.

    The Kaufman Adaptive Moving Average adjusts its own responsiveness to price changes
    based on how "efficiently" price is moving. It compares the net directional move over
    the window to the total (sum of absolute) movement over that same window — the
    Efficiency Ratio. When price trends strongly in one direction (an efficient move), the
    Efficiency Ratio is close to 1 and KAMA tracks price closely, behaving like a fast EMA.
    When price whipsaws sideways (an inefficient move), the Efficiency Ratio is close to 0
    and KAMA flattens out, behaving like a slow EMA — reducing whipsaw signals in choppy
    markets while still reacting quickly during strong trends.

    The formula is a follows:

    - Change = |Close(t) — Close(t - window)|
    - Volatility = Sum(|Close(i) — Close(i - 1)|, window)
    - Efficiency Ratio (ER) = Change / Volatility
    - Fastest SC = 2 / (fast_window + 1), Slowest SC = 2 / (slow_window + 1)
    - Smoothing Constant (SC) = [ER * (Fastest SC — Slowest SC) + Slowest SC]^2
    - KAMA(t) = KAMA(t-1) + SC * (Close(t) — KAMA(t-1))

    Also known as: KAMA, Kaufman's Adaptive Moving Average.

    Reference: Kaufman, P.J. (1998). "Trading Systems and Methods." 3rd ed. Wiley.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods over which the Efficiency Ratio is calculated.
            Defaults to 10.
        fast_window (int): The number of periods that corresponds to the fastest EMA
            constant used when the Efficiency Ratio is at its maximum (1.0). Defaults to 2.
        slow_window (int): The number of periods that corresponds to the slowest EMA
            constant used when the Efficiency Ratio is at its minimum (0.0). Defaults to 30.

    Returns:
        pd.Series: KAMA values.

    Raises:
        TypeError: If `prices` is not a pandas Series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")

    change = (prices - prices.shift(window)).abs()
    volatility = prices.diff().abs().rolling(window=window).sum()
    efficiency_ratio = change / volatility

    fastest_smoothing_constant = 2 / (fast_window + 1)
    slowest_smoothing_constant = 2 / (slow_window + 1)

    smoothing_constant = (
        efficiency_ratio * (fastest_smoothing_constant - slowest_smoothing_constant)
        + slowest_smoothing_constant
    ) ** 2

    kama = pd.Series(index=prices.index, dtype="float64")

    length = len(prices)
    if length == 0:
        return kama

    first_valid_index = smoothing_constant.first_valid_index()
    if first_valid_index is None:
        return kama

    start_position = prices.index.get_loc(first_valid_index)
    kama.iloc[start_position] = prices.iloc[start_position]

    for i in range(start_position + 1, length):
        smoothing_constant_value = smoothing_constant.iloc[i]
        if pd.isna(smoothing_constant_value):
            kama.iloc[i] = kama.iloc[i - 1]
            continue

        kama.iloc[i] = kama.iloc[i - 1] + smoothing_constant_value * (
            prices.iloc[i] - kama.iloc[i - 1]
        )

    return kama


def get_hull_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Hull Moving Average (HMA) of a given price series.

    The Hull Moving Average reduces the lag typically associated with moving averages
    while improving smoothing, by combining a WMA of half the window length, a WMA of
    the full window length, and a further WMA over the square root of the window length.

    The formula is a follows:

    - Raw HMA = 2 * WMA(Close, window / 2) — WMA(Close, window)
    - HMA = WMA(Raw HMA, sqrt(window))

    Also known as: HMA.

    Reference: Hull, A. (2005). "Hull Moving Average."

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for the HMA calculation.

    Returns:
        pd.Series: HMA values.
    """
    half_window_wma = get_weighted_moving_average(prices, max(window // 2, 1))
    full_window_wma = get_weighted_moving_average(prices, window)

    raw_hma = 2 * half_window_wma - full_window_wma

    return get_weighted_moving_average(raw_hma, max(int(np.sqrt(window)), 1))


def get_volume_weighted_average_price(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate the Volume Weighted Average Price (VWAP) of a given price series.

    VWAP weighs the typical price of each period by its traded volume over a rolling
    window, giving a more volume-informed view of the average price than a plain
    moving average. It is widely used as an institutional execution benchmark: a trade
    filled at or better than VWAP is generally considered to have avoided excess market
    impact.

    The formula is a follows:

    - Typical Price = (High + Low + Close) / 3
    - VWAP = Sum(Typical Price * Volume, window) / Sum(Volume, window)

    Also known as: VWAP.

    Reference: Berkowitz, S.A., Logue, D.E., & Noser, E.A. Jr. (1988). "The Total Cost of
    Transactions on the NYSE." Journal of Finance, 43(1), 97-112 — the paper that introduced
    the volume-weighted average price as a transaction-cost benchmark.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods to consider for the VWAP calculation.

    Returns:
        pd.Series: VWAP values.
    """
    typical_price = (prices_high + prices_low + prices_close) / 3
    price_volume = typical_price * volumes

    return (
        price_volume.rolling(window=window).sum() / volumes.rolling(window=window).sum()
    )


def get_parabolic_sar(
    prices_high: pd.Series,
    prices_low: pd.Series,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """
    Calculate the Parabolic Stop and Reverse (SAR) of a given price series.

    The Parabolic SAR is a trend-following indicator that trails price action, flipping
    from below to above price (and vice versa) whenever the trend reverses. The
    acceleration factor increases as the trend extends, causing the SAR to converge
    towards price over time.

    Also known as: SAR, Stop and Reverse.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        af_start (float): Initial acceleration factor. Defaults to 0.02.
        af_increment (float): Amount by which the acceleration factor increases every
            time a new extreme point is reached. Defaults to 0.02.
        af_max (float): Maximum value the acceleration factor can reach. Defaults to 0.2.

    Returns:
        pd.Series: Parabolic SAR values.
    """
    length = len(prices_high)
    sar = pd.Series(index=prices_high.index, dtype="float64")

    if length == 0:
        return sar

    uptrend = True
    af = af_start
    extreme_point = prices_high.iloc[0]
    sar.iloc[0] = prices_low.iloc[0]

    for i in range(1, length):
        prior_sar = sar.iloc[i - 1]

        if uptrend:
            current_sar = prior_sar + af * (extreme_point - prior_sar)
            current_sar = min(
                current_sar, prices_low.iloc[i - 1], prices_low.iloc[max(i - 2, 0)]
            )

            if prices_low.iloc[i] < current_sar:
                uptrend = False
                current_sar = extreme_point
                extreme_point = prices_low.iloc[i]
                af = af_start
            elif prices_high.iloc[i] > extreme_point:
                extreme_point = prices_high.iloc[i]
                af = min(af + af_increment, af_max)
        else:
            current_sar = prior_sar - af * (prior_sar - extreme_point)
            current_sar = max(
                current_sar, prices_high.iloc[i - 1], prices_high.iloc[max(i - 2, 0)]
            )

            if prices_high.iloc[i] > current_sar:
                uptrend = True
                current_sar = extreme_point
                extreme_point = prices_high.iloc[i]
                af = af_start
            elif prices_low.iloc[i] < extreme_point:
                extreme_point = prices_low.iloc[i]
                af = min(af + af_increment, af_max)

        sar.iloc[i] = current_sar

    return sar


def get_pivot_points(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series
) -> pd.DataFrame:
    """
    Calculate the Pivot Points of a given price series.

    Pivot Points are calculated from the previous period's high, low and close prices
    and are used to identify potential support and resistance levels for the current
    period.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.DataFrame: Pivot Points (pivot, resistance 1-3, support 1-3).
    """
    previous_high = prices_high.shift(1)
    previous_low = prices_low.shift(1)
    previous_close = prices_close.shift(1)

    pivot = (previous_high + previous_low + previous_close) / 3

    resistance_1 = (2 * pivot) - previous_low
    support_1 = (2 * pivot) - previous_high

    resistance_2 = pivot + (previous_high - previous_low)
    support_2 = pivot - (previous_high - previous_low)

    resistance_3 = previous_high + 2 * (pivot - previous_low)
    support_3 = previous_low - 2 * (previous_high - pivot)

    return pd.concat(
        [
            pivot,
            resistance_1,
            resistance_2,
            resistance_3,
            support_1,
            support_2,
            support_3,
        ],
        keys=[
            "Pivot Point",
            "Resistance 1",
            "Resistance 2",
            "Resistance 3",
            "Support 1",
            "Support 2",
            "Support 3",
        ],
        axis=1,
    )


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
            The index matches `prices`. A level is only identified on the handful of
            dates where a new local maximum or minimum is detected, so the result is
            forward-filled to the full price index — every date shows the most
            recently established level (NaN before the first level is found), rather
            than only the isolated dates where a new level was detected.
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
            "Resistance": pd.Series(resistance_levels, dtype="float64"),
            "Support": pd.Series(support_levels, dtype="float64"),
        }
    ).sort_index()

    return support_resistance_levels.reindex(prices.index).ffill()


def get_fibonacci_retracement_levels(
    high_prices: pd.Series,
    low_prices: pd.Series,
    levels: list[float] | None = None,
    trend: str = "uptrend",
) -> pd.DataFrame:
    """
    Calculate the Fibonacci Retracement Levels for a given high and low price series.

    Fibonacci Retracement Levels are horizontal price levels, derived from ratios found in the
    Fibonacci sequence, that traders watch as potential support (during a pullback within an
    uptrend) or resistance (during a bounce within a downtrend) zones. `high_prices` and
    `low_prices` are expected to already represent the swing high and swing low over the
    lookback window of interest — e.g. a rolling maximum and rolling minimum computed by the
    caller — so that a full set of retracement levels is produced for every date rather than
    for a single, hand-picked swing.

    The formula is a follows:

    - Uptrend (retracing down from the high): Level = High — Ratio * (High — Low)
    - Downtrend (retracing up from the low): Level = Low + Ratio * (High — Low)

    Also known as: Fibonacci retracement, Fib levels, retracement levels.

    Notes:
        - The 50% level is not actually a Fibonacci ratio. It is included purely by long-standing
          market convention, based on the Dow Theory observation that markets often retrace about
          half of a prior move.
        - The 78.6% level is the square root of 0.618, not a ratio drawn directly from the
          Fibonacci sequence itself (unlike 23.6%, 38.2% and 61.8%, which are).
        - There is no single canonical academic paper behind Fibonacci Retracement Levels — the
          indicator is a practitioner tool derived from the Fibonacci sequence's ratios rather
          than a published financial model. The standard textbook treatment is Murphy, J.J.
          (1999). "Technical Analysis of the Financial Markets." New York Institute of Finance.

    Args:
        high_prices (pd.Series): Series of high prices (e.g. a rolling maximum over the
            desired lookback window).
        low_prices (pd.Series): Series of low prices (e.g. a rolling minimum over the
            desired lookback window).
        levels (list[float] | None): The Fibonacci ratios to calculate levels for. Defaults to
            the standard [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0].
        trend (str): Whether to compute retracement levels for an "uptrend" (levels measured
            down from the high — the conventional direction, used when a prior move was up and
            price is now pulling back) or a "downtrend" (levels measured up from the low, used
            when a prior move was down and price is now bouncing). Defaults to "uptrend".

    Returns:
        pd.DataFrame: Fibonacci Retracement Levels, one column per ratio in `levels`, labelled
            by the ratio expressed as a percentage (e.g. "23.6%").

    Raises:
        ValueError: If `trend` is not "uptrend" or "downtrend".
    """
    if levels is None:
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    if trend not in ("uptrend", "downtrend"):
        raise ValueError("trend must be either 'uptrend' or 'downtrend'.")

    price_range = high_prices - low_prices

    retracement_levels = {}
    for level in levels:
        column_name = f"{level * 100:.1f}%"
        if trend == "uptrend":
            retracement_levels[column_name] = high_prices - level * price_range
        else:
            retracement_levels[column_name] = low_prices + level * price_range

    return pd.concat(
        retracement_levels.values(), keys=retracement_levels.keys(), axis=1
    )
