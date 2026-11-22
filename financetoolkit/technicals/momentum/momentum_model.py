"""Momentum Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.technicals.overlap.overlap_model import (
    apply_to_rolling_windows,
    get_double_exponential_moving_average,
    get_exponential_moving_average,
    get_hull_moving_average,
    get_kaufman_adaptive_moving_average,
    get_moving_average,
    get_triangular_moving_average,
    get_weighted_moving_average,
)
from financetoolkit.technicals.volatility.volatility_model import (
    get_true_range,
    get_wilder_moving_average,
)

# The Know Sure Thing combines exactly four smoothed Rate of Change components.
KST_COMPONENT_COUNT = 4

# The shortest window that has a deviation from its own mean (CCI) and a log10(window) above
# zero (Choppiness Index).
MINIMUM_WINDOW = 2

# The moving averages the oscillators built on two averages (MACD, APO, PPO, KST) can use, as
# TA-Lib's MA types: SMA (0), EMA (1), WMA (2), DEMA (3), TRIMA (5) and KAMA (6), plus the Hull
# moving average of get_moving_average.
MOVING_AVERAGES = {
    "simple": get_moving_average,
    "exponential": get_exponential_moving_average,
    "double_exponential": get_double_exponential_moving_average,
    "triangular": get_triangular_moving_average,
    "weighted": get_weighted_moving_average,
    "hull": get_hull_moving_average,
    "kaufman_adaptive": get_kaufman_adaptive_moving_average,
}

# TA-Lib's MACDFIX smooths the 12- and 26-period EMAs with the fixed constants 0.15 and 0.075
# that Appel used, instead of 2 / (window + 1) (0.1538 and 0.0741).
FIXED_MACD_WINDOWS = (12, 26)
FIXED_MACD_SMOOTHING = (0.15, 0.075)


def get_money_flow_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate the Money Flow Index (MFI) indicator for a given price series.

    The Money Flow Index is a volume-weighted version of the Relative Strength Index: instead
    of ranking the size of up versus down closes, it ranks the "money flow" (typical price
    times volume) that occurred on up days ("positive money flow", where the typical price
    rose from the prior period) against down days ("negative money flow"), then applies the
    same RSI-style transform to turn that ratio into a bounded 0-100 oscillator.

    The formula is as follows:

    - Typical Price = (High + Low + Close) / 3
    - Raw Money Flow = Typical Price * Volume
    - Money Ratio = Sum(Positive Raw Money Flow, window) / Sum(Negative Raw Money Flow, window)
    - MFI = 100 - (100 / (1 + Money Ratio))

    Also known as: MFI, volume-weighted RSI.

    Reference: Quong, G. & Soudack, A. (1989). "Volume-Weighted RSI: Money Flow."
    Technical Analysis of Stocks & Commodities, 7(3), 76-77.

    TA-Lib: MFI, the reference implementation (https://ta-lib.org/functions/mfi.html).

    Notes:
        - The first period has no prior typical price and therefore no money flow, so the
          first MFI value is on period `window` and covers the flows of periods 1 to
          `window`, as in TA-Lib's `ta_MFI.c`. A missing price or volume leaves its flow
          missing rather than counting it as zero.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for MFI calculation.

    Returns:
        pd.Series: MFI values.
    """
    typical_prices = (prices_high + prices_low + prices_close) / 3
    raw_money_flow = typical_prices * volumes

    # A flow needs a prior typical price to be classified as positive or negative, so the first period (and any period next to a missing price) has no flow at all rather than a zero flow.  # noqa: E501
    has_flow = raw_money_flow.notna() & typical_prices.shift(1).notna()

    positive_money_flow = (
        raw_money_flow.where(typical_prices > typical_prices.shift(1), 0)
        .where(has_flow)
        .rolling(window=window)
        .sum()
    )
    negative_money_flow = (
        raw_money_flow.where(typical_prices < typical_prices.shift(1), 0)
        .where(has_flow)
        .rolling(window=window)
        .sum()
    )

    money_ratio = positive_money_flow / negative_money_flow
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def get_williams_percent_r(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Williams %R indicator for a given price series.

    Williams %R measures where the current close sits within the high-low range of the
    lookback window, expressed as a percentage that runs from 0 (close at the window's
    highest high) down to -100 (close at the window's lowest low). It is mathematically a
    rescaled mirror image of the Fast Stochastic %K.

    The formula is as follows:

    - %R = -((Highest High - Close) / (Highest High - Lowest Low)) * 100

    Also known as: %R, Williams %R.

    Reference: Williams, L.R. (1973). "How I Made One Million Dollars... Last Year Trading
    Commodities." Windsor Books.

    TA-Lib: WILLR, the reference implementation (https://ta-lib.org/functions/willr.html).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for %R calculation.

    Returns:
        pd.Series: Williams %R values.
    """
    highest_high = prices_high.rolling(window=window).max()
    lowest_low = prices_low.rolling(window=window).min()

    percent_r = -((highest_high - prices_close) / (highest_high - lowest_low)) * 100
    return percent_r


def get_aroon_indicator(
    prices_high: pd.Series, prices_low: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate the Aroon Indicator for a given price series.

    The Aroon Indicator measures how recently price made a new high (Aroon Up) or a new low
    (Aroon Down) within the lookback window, expressed as a percentage. A reading of 100
    means the extreme (high or low) occurred on the current period; the reading decays
    linearly toward 0 the longer ago the extreme occurred within the window.

    The formula is as follows:

    - Periods Since High = number of periods between the current period and the most recent
      highest high over the current period plus the previous `window` periods (0 if the
      current period is itself the highest high, `window` if the oldest period in the
      lookback is)
    - Aroon Up = ((window - Periods Since High) / window) * 100
    - Aroon Down = ((window - Periods Since Low) / window) * 100, defined analogously using
      the lowest low
    - Aroon Oscillator = Aroon Up - Aroon Down, between -100 and 100

    Also known as: Aroon Up, Aroon Down, Aroon Oscillator, AROONOSC.

    Reference: Chande, T.S. (1995). "A Time Price Oscillator." Technical Analysis of Stocks &
    Commodities, 13(9); Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    TA-Lib: AROON and AROONOSC, the reference implementations
    (https://ta-lib.org/functions/aroon.html, https://ta-lib.org/functions/aroonosc.html).

    Notes:
        - The lookback spans `window + 1` bars — the current period plus the previous
          `window` — which is what TA-Lib's `ta_AROON.c` scans (`trailingIdx = startIdx -
          optInTimePeriod`, inclusive of `today`). A plain `window`-bar rolling lookback can
          never return 0 and mis-scales every reading.
        - Ties are broken on the most recent occurrence of the extreme, so a repeated high
          resets Aroon Up to 100 rather than leaving it decaying from the first occurrence.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods for Aroon calculation.

    Returns:
        pd.DataFrame: Aroon Up, Aroon Down and Aroon Oscillator values.
    """

    # The lookback spans today plus the previous window bars, so the extreme can be up to window periods old and the indicator can reach zero. Reversing before taking the extreme breaks ties on the most recent occurrence, as the definition requires.  # noqa: E501
    def periods_since(extreme):
        def calculation(windows):
            # The position of the extreme counted back from the most recent bar, missing
            # for a window that holds a missing price, as with rolling().apply().
            periods = extreme(windows[..., ::-1], axis=-1).astype(float)

            return np.where(np.isnan(windows).any(axis=-1), np.nan, periods)

        return calculation

    periods_since_high = apply_to_rolling_windows(
        prices_high, window + 1, periods_since(np.argmax)
    )
    periods_since_low = apply_to_rolling_windows(
        prices_low, window + 1, periods_since(np.argmin)
    )

    aroon_up = ((window - periods_since_high) / window) * 100
    aroon_down = ((window - periods_since_low) / window) * 100

    return pd.concat(
        [aroon_up, aroon_down, aroon_up - aroon_down],
        keys=["Aroon Up", "Aroon Down", "Aroon Oscillator"],
        axis=1,
    )


def get_commodity_channel_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    constant: float = 0.015,
) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI) for a given price series.

    The Commodity Channel Index measures how far the typical price has deviated from its own
    moving average, normalized by the average absolute deviation over the same window (the
    "mean deviation", a robust dispersion measure related to, but distinct from, the standard
    deviation). The 0.015 constant is chosen so that roughly 70-80% of CCI values fall within
    the +/-100 band, which is why those levels are conventionally read as overbought/oversold.

    The formula is as follows:

    - Typical Price = (High + Low + Close) / 3
    - Mean Deviation = Mean(|Typical Price - SMA(Typical Price, window)|, window)
    - CCI = (Typical Price - SMA(Typical Price, window)) / (constant * Mean Deviation)

    Also known as: CCI.

    Reference: Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic
    Trends." Technical Analysis of Stocks & Commodities, 1(5).

    TA-Lib: CCI, the reference implementation (https://ta-lib.org/functions/cci.html).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for CCI calculation.
        constant (float): Constant multiplier for CCI. Defaults to 0.015, the conventional
            value chosen so most readings fall within +/-100.

    Returns:
        pd.Series: CCI values.

    Raises:
        ValueError: If `window` is smaller than 2.
    """
    # A single-period window has no deviation from its own mean, so every value would be 0 / 0.
    if window < MINIMUM_WINDOW:
        raise ValueError("window must be at least 2 for the Commodity Channel Index.")

    typical_prices = (prices_high + prices_low + prices_close) / 3
    sma_typical_prices = typical_prices.rolling(window=window).mean()

    # Every point in the window is measured against the current window's mean, rather than each point against the mean of its own trailing window.  # noqa: E501
    mean_deviation = apply_to_rolling_windows(
        typical_prices,
        window,
        lambda windows: np.abs(windows - windows.mean(axis=-1, keepdims=True)).mean(
            axis=-1
        ),
    )

    cci_values = (typical_prices - sma_typical_prices) / (constant * mean_deviation)

    return cci_values


def get_relative_vigor_index(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
) -> pd.DataFrame:
    """
    Calculate the Relative Vigor Index (RVI) for a given price series.

    The Relative Vigor Index measures the conviction behind a move by comparing where each
    period closes relative to its open with the period's full high-low range, on the idea
    that in an uptrend prices tend to close above where they opened and in a downtrend
    below. Both the close-open move and the high-low range are first smoothed with a
    symmetric 4-period weighted average (weights 1, 2, 2, 1) to remove the shortest cycles,
    then averaged over `window` periods, and their ratio is the RVI. The signal line applies
    the same 4-period weighting to the RVI itself, and crossovers of the two lines are the
    usual trigger.

    The formula is as follows:

    - Numerator = [(Close - Open) + 2 * (Close(t-1) - Open(t-1)) + 2 * (Close(t-2) -
      Open(t-2)) + (Close(t-3) - Open(t-3))] / 6
    - Denominator = [(High - Low) + 2 * (High(t-1) - Low(t-1)) + 2 * (High(t-2) - Low(t-2)) +
      (High(t-3) - Low(t-3))] / 6
    - RVI = SMA(Numerator, window) / SMA(Denominator, window)
    - Signal Line = [RVI + 2 * RVI(t-1) + 2 * RVI(t-2) + RVI(t-3)] / 6

    Also known as: RVI, RVGI, Relative Vigor Index.

    Reference: Ehlers, J.F. (2002). "Relative Vigor Index." Technical Analysis of Stocks &
    Commodities.

    Notes:
        - The RVI is a fraction of the average range, so it usually lies between -1 and 1:
          positive when closes sit above opens, negative when they sit below.
        - The first RVI value is on period `window` + 2 (the weighting needs four periods and
          the average `window` weighted values), and the first signal value three periods
          later.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the averages of the numerator and denominator.

    Returns:
        pd.DataFrame: Relative Vigor Index and Signal Line values.
    """

    # Ehlers' symmetric weighting (1, 2, 2, 1) / 6 over the current and three prior periods.
    def symmetric_weighted_average(values: pd.Series) -> pd.Series:
        return (
            values + 2 * values.shift(1) + 2 * values.shift(2) + values.shift(3)
        ) / 6

    numerator = symmetric_weighted_average(prices_close - prices_open)
    denominator = symmetric_weighted_average(prices_high - prices_low)

    relative_vigor_index = (
        numerator.rolling(window=window).mean()
        / denominator.rolling(window=window).mean()
    )
    signal_line = symmetric_weighted_average(relative_vigor_index)

    return pd.concat(
        [relative_vigor_index, signal_line],
        keys=["Relative Vigor Index", "Signal Line"],
        axis=1,
    )


def get_force_index(
    prices_close: pd.Series, volumes: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Force Index for a given price series.

    The Force Index combines the direction and magnitude of a period's price change with
    the volume behind it, so that a price move on heavy volume registers as a stronger
    signal than the same move on light volume. The raw, single-period Force Index is
    smoothed with an Exponential Moving Average to filter out noise; Elder's own examples
    use a short window (2 periods) to time entries and a longer window (13 periods) to
    gauge the underlying trend.

    The formula is as follows:

    - Raw Force Index = (Close(t) - Close(t-1)) * Volume(t)
    - Force Index = EMA(Raw Force Index, window)

    Also known as: FI, Elder's Force Index.

    Reference: Elder, A. (1993). "Trading for a Living." Wiley.

    TA-Lib: EFI, the reference implementation (https://ta-lib.org/functions/efi.html).

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for the Exponential Moving Average used to smooth
            the raw Force Index, conventionally 13 (Elder's trend setting).

    Returns:
        pd.Series: Force Index values, missing until the first `window` raw values (from
            the second period on) seed the Exponential Moving Average.
    """
    raw_force_index = prices_close.diff(1) * volumes

    return get_exponential_moving_average(raw_force_index, window)


def get_ultimate_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window_1: int,
    window_2: int,
    window_3: int,
) -> pd.Series:
    """
    Calculate the Ultimate Oscillator for a given price series.

    The Ultimate Oscillator combines buying pressure measured over three different lookback
    periods (short, medium and long) into a single oscillator, weighting the shorter,
    more responsive period most heavily. This is intended to reduce the false divergence
    signals that single-period oscillators are prone to, by requiring some agreement across
    multiple timeframes.

    The formula is as follows:

    - Buying Pressure = Close - Min(Low, Previous Close)
    - True Range = Max[(High - Low), |High - Previous Close|, |Low - Previous Close|]
    - Average(i) = Sum(Buying Pressure, window_i) / Sum(True Range, window_i)
    - Ultimate Oscillator = 100 * [(4 * Average_1) + (2 * Average_2) + Average_3] / (4 + 2 + 1)

    Also known as: UO.

    Reference: Williams, L.R. (1985). "The Ultimate Oscillator." Technical Analysis of Stocks
    & Commodities, 3(4).

    TA-Lib: ULTOSC, the reference implementation (https://ta-lib.org/functions/ultosc.html).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window_1 (int): Number of periods for the first (shortest) time period.
        window_2 (int): Number of periods for the second time period.
        window_3 (int): Number of periods for the third (longest) time period.

    Returns:
        pd.Series: Ultimate Oscillator values, bounded between 0 and 100.
    """
    true_range = pd.concat(
        [
            prices_high - prices_low,
            abs(prices_high - prices_close.shift(1)),
            abs(prices_low - prices_close.shift(1)),
        ],
        axis=1,
    ).max(axis=1)

    sum_true_range_1 = true_range.rolling(window=window_1).sum()
    sum_true_range_2 = true_range.rolling(window=window_2).sum()
    sum_true_range_3 = true_range.rolling(window=window_3).sum()

    # The true low is the lower of the current low and the previous close.
    buying_pressure = prices_close - pd.concat(
        [prices_low, prices_close.shift(1)], axis=1
    ).min(axis=1)

    avg_buying_pressure_1 = buying_pressure.rolling(window=window_1).sum()
    avg_buying_pressure_2 = buying_pressure.rolling(window=window_2).sum()
    avg_buying_pressure_3 = buying_pressure.rolling(window=window_3).sum()

    average_1 = avg_buying_pressure_1 / sum_true_range_1
    average_2 = avg_buying_pressure_2 / sum_true_range_2
    average_3 = avg_buying_pressure_3 / sum_true_range_3

    ultimate_oscillator = (
        100 * ((average_1 * 4) + (average_2 * 2) + average_3) / (4 + 2 + 1)
    )

    return ultimate_oscillator


def get_percentage_price_oscillator(
    prices_close: pd.Series,
    short_window: int,
    long_window: int,
    signal_window: int = 9,
    moving_average: str = "exponential",
) -> pd.DataFrame:
    """
    Calculate the Percentage Price Oscillator (PPO) for a given price series.

    The Percentage Price Oscillator is the Moving Average Convergence Divergence line
    re-expressed as a percentage of the long-term EMA, rather than as a raw price
    difference. This makes readings comparable across securities trading at very different
    price levels, which the raw MACD line cannot do. As for the MACD, a signal line (an EMA
    of the PPO) and the histogram (the PPO minus its signal line) show when the PPO turns.

    The formula is as follows:

    - PPO = (MA(Close, short_window) - MA(Close, long_window)) / MA(Close, long_window)
    - Signal Line = EMA(PPO, signal_window)
    - Histogram = PPO - Signal Line

    with MA the Exponential Moving Average by default. The conventional parameters are 12,
    26 and 9 periods, as for MACD.

    Also known as: PPO, PPO signal line, PPO histogram.

    Reference: Appel, G. (2005). "Technical Analysis: Power Tools for Active Investors." FT
    Press. The PPO is a normalized variant of Appel's Moving Average Convergence Divergence
    (Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method." Signalert
    Corp.).

    TA-Lib: PPO, the reference implementation (https://ta-lib.org/functions/ppo.html), with
    the moving average as its `matype`; it returns the PPO in percent and no signal line.
    The signal line and histogram are those of ta's PercentagePriceOscillator, pandas-ta's
    ppo and finta's PPO.

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values (see
          `get_exponential_moving_average`), so the PPO starts on period `long_window` - 1
          and the signal line and histogram `signal_window` - 1 periods later.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        signal_window (int): Number of periods for the signal line EMA. Defaults to 9.
        moving_average (str): The moving average of the short and long term, one of
            `MOVING_AVERAGES`: "simple", "exponential", "double_exponential", "triangular",
            "weighted", "hull" or "kaufman_adaptive". Defaults to "exponential".

    Returns:
        pd.DataFrame: The Percentage Price Oscillator, Signal Line and Histogram as columns,
        as decimals (0.02 is 2%).

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    if moving_average not in MOVING_AVERAGES:
        raise ValueError(
            f"moving_average must be one of {', '.join(MOVING_AVERAGES)}, not {moving_average}."
        )

    short_average = MOVING_AVERAGES[moving_average](prices_close, short_window)
    long_average = MOVING_AVERAGES[moving_average](prices_close, long_window)

    ppo = (short_average - long_average) / long_average
    signal_line = get_exponential_moving_average(ppo, signal_window)

    return pd.concat(
        [ppo, signal_line, ppo - signal_line],
        keys=["Percentage Price Oscillator", "Signal Line", "Histogram"],
        axis=1,
    )


def get_percentage_volume_oscillator(
    volumes: pd.Series, short_window: int, long_window: int, signal_window: int
) -> pd.DataFrame:
    """
    Calculate the Percentage Volume Oscillator (PVO) for a given volume series.

    The Percentage Volume Oscillator is the Percentage Price Oscillator applied to volume:
    the difference between a short-term and a long-term EMA of volume as a share of the
    long-term EMA. It rises when recent volume runs above its longer-term average and
    falls when it runs below, which makes it a measure of volume momentum that can be
    compared across securities whatever the number of shares they trade.

    The formula is as follows:

    - PVO = (EMA(Volume, short_window) - EMA(Volume, long_window)) / EMA(Volume, long_window)
    - Signal Line = EMA(PVO, signal_window)
    - Histogram = PVO - Signal Line

    The conventional parameters are 12, 26 and 9 periods, as for MACD and the PPO.

    Also known as: PVO, volume oscillator.

    Reference: StockCharts ChartSchool, "Percentage Volume Oscillator (PVO)". The PVO is the
    volume counterpart of the Percentage Price Oscillator, a normalized variant of Appel's
    Moving Average Convergence Divergence (Appel, G. (1979). "The Moving Average
    Convergence-Divergence Trading Method." Signalert Corp.).

    TA-Lib: PVO, the reference implementation (https://ta-lib.org/functions/pvo.html).

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values (see
          `get_exponential_moving_average`), so the PVO starts on period `long_window` - 1
          and the signal line and histogram `signal_window` - 1 periods later.
        - A long-term EMA of zero (a series without volume, such as an index or a currency
          pair with no reported volume) has no PVO rather than an infinite one.

    Args:
        volumes (pd.Series): Series of trading volumes.
        short_window (int): Number of periods for the short-term EMA.
        long_window (int): Number of periods for the long-term EMA.
        signal_window (int): Number of periods for the signal line EMA.

    Returns:
        pd.DataFrame: The Percentage Volume Oscillator, Signal Line and Histogram as
        columns, as decimals (0.02 is 2%).
    """
    short_ema = get_exponential_moving_average(volumes, short_window)
    long_ema = get_exponential_moving_average(volumes, long_window)

    pvo = (short_ema - long_ema) / long_ema.where(long_ema != 0)
    signal_line = get_exponential_moving_average(pvo, signal_window)

    return pd.concat(
        [pvo, signal_line, pvo - signal_line],
        keys=["Percentage Volume Oscillator", "Signal Line", "Histogram"],
        axis=1,
    )


def get_detrended_price_oscillator(
    prices_close: pd.Series,
    window: int,
    moving_average_type: str = "sma",
) -> pd.Series:
    """
    Calculate the Detrended Price Oscillator (DPO) for a given price series.

    The Detrended Price Oscillator strips the trend out of price so that cycles above and
    below the trend line stand out. It does this by comparing a *past* closing price to the
    *current* moving average, offset by roughly half the moving average's window — since a
    non-shifted trailing moving average's own "center of mass" sits about half a window
    behind the current bar, comparing today's average to the price from that same point in
    the past effectively centers the average on the historical price it is being compared to.

    The formula is as follows:

    - Displacement = floor(window / 2) + 1
    - DPO = Close(t - Displacement) - Moving Average(t)

    Note that, unlike most oscillators, the moving average itself is *not* shifted — only the
    close price is looked up further back in time.

    Also known as: DPO.

    Reference: The standard textbook treatment is Murphy, J.J. (1999). "Technical Analysis of
    the Financial Markets." New York Institute of Finance; there is no single canonical
    academic paper behind DPO.

    TA-Lib: DPO, the reference implementation (https://ta-lib.org/functions/dpo.html).

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for DPO calculation.
        moving_average_type (str): The type of moving average to detrend against, either
            "sma" (Simple Moving Average) or "ema" (Exponential Moving Average). Defaults to
            "sma", the conventional choice.

    Returns:
        pd.Series: DPO values.

    Raises:
        ValueError: If `moving_average_type` is not "sma" or "ema".
    """
    if moving_average_type == "sma":
        moving_average = get_moving_average(prices_close, window)
    elif moving_average_type == "ema":
        moving_average = get_exponential_moving_average(prices_close, window)
    else:
        raise ValueError("Invalid moving average type. Choose either 'sma' or 'ema'.")

    displacement = int(window / 2) + 1

    dpo = prices_close.shift(displacement) - moving_average

    return dpo


def get_directional_movement_index(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate Wilder's Directional Movement System: the smoothed Plus and Minus Directional
    Movement (+DM, -DM), the Directional Indicators (+DI, -DI) and the Directional Movement
    Index (DX) of a given price series.

    The Directional Movement is the part of a period's range outside the previous period's
    range: upward when the high rises above the previous high by more than the low falls
    below the previous low, downward the other way around. Wilder sums each over the window
    with his smoothing and divides by the smoothed True Range, which gives the Directional
    Indicators: the share of the recent range that moved up (+DI) and down (-DI). The
    Directional Movement Index is their difference as a share of their sum, which is high
    in a trend in either direction and low in a range; its Wilder average is the ADX.

    The formula is as follows:

    - +DM(1) = High(t) - High(t-1), where positive and greater than Low(t-1) - Low(t), else 0
    - -DM(1) = Low(t-1) - Low(t), where positive and greater than High(t) - High(t-1), else 0
    - +DM = Wilder's sum of +DM(1) over window = window * Wilder's MA(+DM(1), window), and
      -DM analogously, where Wilder's sum(t) = Sum(t-1) - Sum(t-1) / window + Value(t)
    - +DI = 100 * Wilder's MA(+DM(1), window) / Wilder's MA(True Range, window)
    - -DI = 100 * Wilder's MA(-DM(1), window) / Wilder's MA(True Range, window)
    - DX = 100 * |+DI - -DI| / (+DI + -DI)

    Also known as: DMI, DMS, Directional Movement System, +DM/-DM, +DI/-DI, PLUS_DM,
    MINUS_DM, PLUS_DI, MINUS_DI, DX.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    TA-Lib: PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI and DX, the reference implementations
    (https://ta-lib.org/functions/plus_dm.html, https://ta-lib.org/functions/minus_dm.html,
    https://ta-lib.org/functions/plus_di.html, https://ta-lib.org/functions/minus_di.html,
    https://ta-lib.org/functions/dx.html).

    Notes:
        - The True Range and both directional movements start on the second period, the
          first with a previous period, so every column starts on period `window`, as
          TA-Lib's PLUS_DI, MINUS_DI and DX do. Wilder's smoothing is seeded with the sum
          (average) of the first `window` values, as in `get_average_directional_index`, of
          which DX is the input. TA-Lib seeds its running sums with `window` - 1 values and
          starts PLUS_DM and MINUS_DM one period earlier, a difference that decays by a
          factor 1 - 1 / `window` per period.
        - +DM and -DM are Wilder's running sums, as in his worksheets and TA-Lib, so they
          are in price units and `window` times the smoothed averages behind the +DI and -DI.
        - A period without any directional movement over the window (+DI and -DI both 0) has
          no DX.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of Wilder's smoothing.

    Returns:
        pd.DataFrame: +DM, -DM, +DI, -DI and DX as columns, the Directional Indicators and
        the DX between 0 and 100.
    """
    # The first period has no previous close or previous high and low, so it has no True
    # Range and no directional movement; starting both series on the second period gives
    # the first smoothed value on period `window` and the first ADX on period
    # 2 * window - 1, as in TA-Lib's ta_ADX.c, instead of averaging in a High - Low range
    # and two zero moves for the first period.
    has_previous_period = prices_close.shift(1).notna()
    true_range = get_true_range(prices_high, prices_low, prices_close).where(
        has_previous_period
    )

    # Wilder's rule: only the larger of the two moves counts and the other is set to zero. Counting both on an outside bar inflates each Directional Indicator and compresses the resulting Directional Index.  # noqa: E501
    up_move = prices_high.diff()
    down_move = -prices_low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0).where(
        up_move.notna() & has_previous_period
    )
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0).where(
        down_move.notna() & has_previous_period
    )

    smoothed_true_range = get_wilder_moving_average(true_range, window)
    smoothed_plus_dm = get_wilder_moving_average(plus_dm, window)
    smoothed_minus_dm = get_wilder_moving_average(minus_dm, window)

    plus_di = 100 * (smoothed_plus_dm / smoothed_true_range)
    minus_di = 100 * (smoothed_minus_dm / smoothed_true_range)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

    return pd.concat(
        [
            window * smoothed_plus_dm,
            window * smoothed_minus_dm,
            plus_di,
            minus_di,
            dx,
        ],
        keys=["+DM", "-DM", "+DI", "-DI", "DX"],
        axis=1,
    )


def get_average_directional_index(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Average Directional Movement Index (ADX) of a given price series.

    The Average Directional Movement Index (ADX) is a technical indicator used to quantify
    the strength of a trend. It combines the information from the Plus Directional Indicator (+DI)
    and Minus Directional Indicator (-DI) to provide a single value that represents the trend's strength.

    Every averaging step in the original ADX (the True Range, +DM and -DM smoothing, and the
    final DX-to-ADX smoothing) uses Wilder's specific smoothing method rather than a plain
    Simple Moving Average — substituting a plain SMA is one of the most common mistakes when
    re-implementing this indicator, and produces materially different values.

    The formula is as follows:

    - +DM = High(t) - High(t-1), where positive and greater than -(Low(t) - Low(t-1)), else 0
    - -DM = Low(t-1) - Low(t), where positive and greater than (High(t) - High(t-1)), else 0
    - True Range = Max[(High - Low), |High - Previous Close|, |Low - Previous Close|]
    - +DI = 100 * Wilder's MA(+DM, window) / Wilder's MA(True Range, window)
    - -DI = 100 * Wilder's MA(-DM, window) / Wilder's MA(True Range, window)
    - DX = 100 * |+DI - -DI| / (+DI + -DI)
    - ADX = Wilder's MA(DX, window)

    The +DM, -DM, +DI, -DI and DX are returned by `get_directional_movement_index`.

    Also known as: ADX, Average Directional Index.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    TA-Lib: ADX, the reference implementation (https://ta-lib.org/functions/adx.html).

    Notes:
        - The True Range and both directional movements start on the second period, the
          first with a previous period, so the first ADX value is on period 2 * window - 1,
          as in TA-Lib. Wilder's smoothing is seeded with the average of the first `window`
          values; TA-Lib seeds its running sums with `window` - 1 values instead, a
          difference that decays within a few windows.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to consider for ADX calculation.

    Returns:
        pd.Series: ADX values.
    """
    directional_movement_index = get_directional_movement_index(
        prices_high, prices_low, prices_close, window
    )

    adx = get_wilder_moving_average(directional_movement_index["DX"], window)

    return adx


def get_average_directional_index_rating(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Average Directional Movement Index Rating (ADXR) of a given price series.

    The ADXR is Wilder's smoothed ADX: the average of today's ADX and the ADX of `window` - 1
    periods ago. It rates how strongly a market trends, more slowly than the ADX itself,
    and Wilder used it to rank commodities by their directional movement and so choose
    which ones to trade with his trend-following systems.

    The formula is as follows:

    - ADXR = (ADX(t) + ADX(t - (window - 1))) / 2

    with the ADX of `get_average_directional_index`.

    Also known as: ADXR, Average Directional Movement Rating.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    TA-Lib: ADXR, the reference implementation (https://ta-lib.org/functions/adxr.html).

    Notes:
        - The ADX starts on period 2 * `window` - 1, so the ADXR starts on period
          3 * `window` - 2, as in TA-Lib. The ADX is seeded as in
          `get_average_directional_index`, so the first values differ from TA-Lib's until
          the ADX values converge.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the ADX and of the lag of the rating.

    Returns:
        pd.Series: ADXR values, between 0 and 100.
    """
    adx = get_average_directional_index(prices_high, prices_low, prices_close, window)

    return (adx + adx.shift(window - 1)) / 2


def get_chande_momentum_oscillator(
    prices_close: pd.Series, window: int, smoothed: bool = False
) -> pd.Series:
    """
    Calculate the Chande Momentum Oscillator (CMO) for a given price series.

    The Chande Momentum Oscillator is closely related to the Relative Strength Index: both
    compare the sum of up-moves to the sum of down-moves over a window. Where RSI computes
    the ratio of average gain to average loss and rescales it to 0-100, CMO instead uses the
    net difference between total gains and total losses, divided by their sum, which lets it
    range symmetrically from -100 to +100 (and, unlike RSI, uses a plain sum rather than
    Wilder's smoothing).

    The formula is as follows:

    - CMO = 100 * (Sum(Up Changes, window) - Sum(Down Changes, window)) / (Sum(Up Changes,
      window) + Sum(Down Changes, window))
    - Smoothed CMO = 100 * (Wilder's MA(Up Changes, window) - Wilder's MA(Down Changes,
      window)) / (Wilder's MA(Up Changes, window) + Wilder's MA(Down Changes, window))

    The smoothed CMO averages the gains and losses as the RSI does, so it equals
    2 * RSI - 100.

    Also known as: CMO, CMOU (the summed CMO), Chande momentum.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    TA-Lib: CMOU and CMO, the reference implementations
    (https://ta-lib.org/functions/cmou.html, https://ta-lib.org/functions/cmo.html): CMOU
    sums the gains and losses as Chande does, CMO smooths them with Wilder's method.

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for CMO calculation.
        smoothed (bool): Whether to smooth the gains and losses with Wilder's method (TA-Lib's
            CMO) instead of summing them (Chande's CMO, TA-Lib's CMOU). Defaults to False.

    Returns:
        pd.Series: CMO values, bounded between -100 and 100.
    """
    price_diff = prices_close.diff(1)

    # The first price has no change, which is missing rather than a zero change, so the
    # first sums cover a full window of changes.
    gains = price_diff.clip(lower=0)
    losses = (-price_diff).clip(lower=0)

    if smoothed:
        up_sum = get_wilder_moving_average(gains, window)
        down_sum = get_wilder_moving_average(losses, window)
    else:
        up_sum = gains.rolling(window=window).sum()
        down_sum = losses.rolling(window=window).sum()

    cmo = ((up_sum - down_sum) / (up_sum + down_sum)) * 100
    return cmo


def get_ichimoku_cloud(
    prices_high: pd.Series,
    prices_low: pd.Series,
    conversion_window: int,
    base_window: int,
    lead_span_b_window: int,
) -> pd.DataFrame:
    """
    Calculate the Ichimoku Cloud components for a given price series.

    The Ichimoku Cloud describes trend, support and resistance in one picture by plotting
    two fast/slow midpoint lines against a shaded band ("the cloud", or Kumo) spanned by two
    further midpoint lines that are projected forward in time. Each line is a midpoint of a
    lookback range — the average of the highest high and lowest low over its own window —
    rather than an average of closing prices, so every line sits in the middle of the range
    price actually traded over that window.

    The formula is as follows:

    - Conversion Line (Tenkan-sen) = (Max(High, conversion_window) + Min(Low,
      conversion_window)) / 2
    - Base Line (Kijun-sen) = (Max(High, base_window) + Min(Low, base_window)) / 2
    - Leading Span A (Senkou Span A) = (Conversion Line + Base Line) / 2, plotted
      `base_window` periods ahead
    - Leading Span B (Senkou Span B) = (Max(High, lead_span_b_window) + Min(Low,
      lead_span_b_window)) / 2, plotted `base_window` periods ahead

    Also known as: Ichimoku Kinko Hyo, Ichimoku, Kumo cloud.

    Reference: Hosoda, G. (1969). "Ichimoku Kinko Hyo." Published under the pen name Ichimoku
    Sanjin; the standard English treatment is Elliott, N. (2007). "Ichimoku Charts."
    Harriman House.

    TA-Lib: ICHIMOKU, the reference implementation (https://ta-lib.org/functions/ichimoku.html),
    which places the leading spans on the period they are calculated on.

    Notes:
        - Both leading spans are projected forward by `base_window` (26 by default) periods,
          not by their own lookback window. Projecting Leading Span B forward by
          `lead_span_b_window` (52) instead is a common re-implementation error.
        - Because a span plotted `base_window` periods into the future is, at any given
          period, derived from data `base_window` periods in the *past*, both leading spans
          are computable from information available at that period and carry no look-ahead.
        - The leading spans are shifted forward by exactly `base_window` periods, so the
          value on period t is the one calculated on period t - `base_window`, as
          StockCharts defines the cloud. Some charting platforms count the current period
          as the first of the 26 and shift by `base_window` - 1 instead, which places the
          same values one period earlier.
        - The Lagging Span (Chikou Span), the close plotted `base_window` periods into the
          past, is deliberately not returned: at any period it reports a *future* close and
          therefore cannot be used without look-ahead bias.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        conversion_window (int): Number of periods for the Conversion Line. Conventionally 9.
        base_window (int): Number of periods for the Base Line, and the number of periods
            both leading spans are projected forward by. Conventionally 26.
        lead_span_b_window (int): Number of periods for Leading Span B. Conventionally 52.

    Returns:
        pd.DataFrame: Ichimoku Cloud components (Conversion Line, Base Line, Leading Span A,
            Leading Span B).
    """
    conversion_line = (
        prices_high.rolling(window=conversion_window).max()
        + prices_low.rolling(window=conversion_window).min()
    ) / 2
    base_line = (
        prices_high.rolling(window=base_window).max()
        + prices_low.rolling(window=base_window).min()
    ) / 2
    lead_span_a = ((conversion_line + base_line) / 2).shift(base_window)
    lead_span_b = (
        (
            prices_high.rolling(window=lead_span_b_window).max()
            + prices_low.rolling(window=lead_span_b_window).min()
        )
        / 2
    ).shift(base_window)

    return pd.concat(
        [conversion_line, base_line, lead_span_a, lead_span_b],
        keys=["Conversion Line", "Base Line", "Leading Span A", "Leading Span B"],
        axis=1,
    )


def get_stochastic_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    smooth_window: int,
    slow: bool = False,
) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator of a given price series.

    The Stochastic Oscillator measures where the close sits within the high-low range of the
    lookback window, on the premise that in an uptrend prices tend to close near the high of
    the range (and near the low in a downtrend). %D is a moving average of %K used as a
    signal line for crossovers, the same role the signal line plays for MACD.

    The formula is as follows:

    - Fast %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    - Fast %D = SMA(Fast %K, smooth_window)
    - Slow %K = Fast %D = SMA(Fast %K, smooth_window)
    - Slow %D = SMA(Slow %K, smooth_window)

    The "Fast Stochastic" (the default) smooths the raw %K once to produce %D. The "Slow
    Stochastic" (`slow=True`) smooths %K itself with an SMA first, which is the stochastic
    Lane charted and most platforms show by default; its %K is the fast %D.

    Also known as: Stochastic Oscillator, Fast Stochastic, Slow Stochastic, %K/%D, STOCHF,
    STOCH.

    Reference: Lane, G.C. Developed in the late 1950s; see Lane, G.C. (1984). "Lane's
    Stochastics." Technical Analysis of Stocks & Commodities, 2(3).

    TA-Lib: STOCHF and STOCH, the reference implementations
    (https://ta-lib.org/functions/stochf.html, https://ta-lib.org/functions/stoch.html).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the stochastic calculation.
        smooth_window (int): Number of periods for smoothing the %K values into %D, and with
            `slow` the raw %K into the slow %K.
        slow (bool): Whether to return the slow stochastic instead of the fast one. Defaults
            to False.

    Returns:
        pd.DataFrame: Stochastic Oscillator (%K and %D).
    """
    lowest_low = prices_low.rolling(window=window).min()
    highest_high = prices_high.rolling(window=window).max()

    stochastic_k = ((prices_close - lowest_low) / (highest_high - lowest_low)) * 100
    if slow:
        stochastic_k = stochastic_k.rolling(window=smooth_window).mean()
    stochastic_d = stochastic_k.rolling(window=smooth_window).mean()

    return pd.concat(
        [stochastic_k, stochastic_d], keys=["Stochastic %K", "Stochastic %D"], axis=1
    )


def get_moving_average_convergence_divergence(
    prices: pd.Series,
    short_window: int,
    long_window: int,
    signal_window: int,
    moving_average: str = "exponential",
    signal_moving_average: str = "exponential",
    fixed: bool = False,
) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD) of a given price series.

    MACD tracks the convergence and divergence of a short-term and a long-term Exponential
    Moving Average of price. The MACD line crossing above/below its own EMA-based signal line
    is the classic trigger for a trend-following signal, and the histogram, the distance
    between the two lines, shows the momentum of the MACD line itself.

    The formula is as follows:

    - MACD Line = MA(Close, short_window) - MA(Close, long_window)
    - Signal Line = MA(MACD Line, signal_window)
    - Histogram = MACD Line - Signal Line

    with MA the Exponential Moving Average by default. With `fixed`, the 12- and 26-period
    EMAs use Appel's fixed smoothing constants 0.15 and 0.075 instead of 2 / (window + 1).

    Also known as: MACD, MACD histogram, MACDEXT (other moving averages), MACDFIX (fixed
    smoothing constants).

    Reference: Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method."
    Signalert Corp. The histogram is from Aspray, T. (1986). "Fine-tuning the MACD."
    Technical Analysis of Stocks & Commodities, 4(4).

    TA-Lib: MACD, MACDEXT and MACDFIX, the reference implementations
    (https://ta-lib.org/functions/macd.html, https://ta-lib.org/functions/macdext.html,
    https://ta-lib.org/functions/macdfix.html).

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values (see
          `get_exponential_moving_average`), so the MACD line starts on period
          `long_window` - 1 and the signal line `signal_window` - 1 periods later, as on
          StockCharts. TA-Lib's `ta_MACD.c` additionally starts the short EMA on the same
          period as the long one, which changes only the first values.
        - `moving_average` applies to both the short and the long average, as TA-Lib's
          MACDEXT with the same `fastmatype` and `slowmatype`.

    Args:
        prices (pd.Series): Series of prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        signal_window (int): Number of periods for the signal line moving average.
        moving_average (str): The moving average of the short and long term, one of
            `MOVING_AVERAGES`: "simple", "exponential", "double_exponential", "triangular",
            "weighted", "hull" or "kaufman_adaptive". Defaults to "exponential".
        signal_moving_average (str): The moving average of the signal line, one of
            `MOVING_AVERAGES`. Defaults to "exponential".
        fixed (bool): Whether to smooth the 12- and 26-period EMAs with the fixed constants
            0.15 and 0.075 (TA-Lib's MACDFIX). Needs the exponential moving average and
            windows of 12 and 26. Defaults to False.

    Returns:
        pd.DataFrame: The MACD Line, Signal Line and Histogram as columns.

    Raises:
        ValueError: If a moving average is not one of `MOVING_AVERAGES`, or `fixed` is
            combined with another moving average or other windows than 12 and 26.
    """
    for name in (moving_average, signal_moving_average):
        if name not in MOVING_AVERAGES:
            raise ValueError(
                f"moving_average must be one of {', '.join(MOVING_AVERAGES)}, not {name}."
            )

    if fixed:
        if (
            moving_average != "exponential"
            or (short_window, long_window) != FIXED_MACD_WINDOWS
        ):
            raise ValueError(
                "fixed needs the exponential moving average with windows of 12 and 26."
            )

        # The EMA of get_exponential_moving_average with the fixed smoothing constant in
        # place of 2 / (window + 1): seeded with the SMA of the first window values on the
        # last of them, NaN before and on periods without a price.
        averages = []
        for window, smoothing in zip(FIXED_MACD_WINDOWS, FIXED_MACD_SMOOTHING):
            simple_moving_average = prices.rolling(window=window).mean()
            has_seed = simple_moving_average.notna().astype(bool).cummax()
            seed_period = has_seed & ~has_seed.shift(1, fill_value=False).astype(bool)
            seeded_prices = prices.where(has_seed).mask(
                seed_period, simple_moving_average
            )
            averages.append(
                seeded_prices.ewm(alpha=smoothing, adjust=False, ignore_na=True)
                .mean()
                .where(has_seed & prices.notna())
            )
        short_average, long_average = averages
    else:
        short_average = MOVING_AVERAGES[moving_average](prices, short_window)
        long_average = MOVING_AVERAGES[moving_average](prices, long_window)

    macd_line = short_average - long_average
    signal_line = MOVING_AVERAGES[signal_moving_average](macd_line, signal_window)

    return pd.concat(
        [macd_line, signal_line, macd_line - signal_line],
        keys=["MACD Line", "Signal Line", "Histogram"],
        axis=1,
    )


def get_relative_strength_index(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) of a given price series.

    RSI measures the ratio of average gains to average losses over a lookback window and
    rescales it into a bounded 0-100 oscillator. Wilder's original method smooths the average
    gain and average loss using his own recursive smoothing method (equivalent to an EMA
    with a smoothing constant of 1/window, seeded with a simple average of the first `window`
    observations) rather than a plain Simple Moving Average — substituting a plain SMA is one
    of the most common mistakes when re-implementing RSI, and produces materially different
    values, particularly in trending markets.

    The formula is as follows:

    - Average Gain = Wilder's MA(Upward Changes, window)
    - Average Loss = Wilder's MA(Downward Changes, window)
    - RS = Average Gain / Average Loss
    - RSI = 100 - (100 / (1 + RS))

    Also known as: RSI.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    TA-Lib: RSI, the reference implementation (https://ta-lib.org/functions/rsi.html).

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for RSI calculation.

    Returns:
        pd.Series: RSI values, bounded between 0 and 100.
    """
    # Calculate price changes
    price_diff = prices.diff(1)

    # Calculate upward and downward price changes. The first price has no change, which
    # stays missing rather than counting as a zero change, so the first average covers
    # the first window of actual changes, as Wilder (1978) defines it.
    up_changes = price_diff.clip(lower=0)
    down_changes = (-price_diff).clip(lower=0)

    # Calculate average gains and losses over the specified window using Wilder's smoothing
    avg_gain = get_wilder_moving_average(up_changes, window)
    avg_loss = get_wilder_moving_average(down_changes, window)

    # Calculate the relative strength index
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_stochastic_relative_strength_index(
    prices: pd.Series, window: int, stochastic_window: int, smooth_window: int
) -> pd.DataFrame:
    """
    Calculate the Stochastic RSI of a given price series.

    The Stochastic RSI applies the Stochastic Oscillator to the Relative Strength Index
    instead of to price: it shows where the current RSI sits within its own high-low range
    over the last `stochastic_window` periods. Chande and Kroll designed it because the RSI
    can spend long stretches between 30 and 70 without reaching either extreme; the
    Stochastic RSI reaches its 0 and 100 extremes far more often, at the cost of being much
    more sensitive.

    The formula is as follows:

    - RSI = Relative Strength Index(Close, window), with Wilder's smoothing
    - %K = 100 * (RSI - Lowest RSI) / (Highest RSI - Lowest RSI), over the last stochastic_window RSI values
    - %D = SMA(%K, smooth_window)

    Chande and Kroll define it as a 14-period stochastic of a 14-period RSI, as StockCharts
    and TradingView do (`window=14`, `stochastic_window=14`). StockCharts shows this %K on a
    scale of 0 to 1; TradingView smooths it over 3 periods, so its %K equals this %D with
    `smooth_window=3`. TA-Lib's `STOCHRSI` defaults to a 5-period stochastic instead
    (timeperiod 14, fastk_period 5, fastd_period 3 with a simple average), which is
    `stochastic_window=5` here and returns its fast %K and %D.

    Also known as: StochRSI, Stoch RSI.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    TA-Lib: STOCHRSI, the reference implementation (https://ta-lib.org/functions/stochrsi.html).

    Notes:
        - The RSI is that of `get_relative_strength_index`, seeded with a simple average of
          the first `window` changes as Wilder and TA-Lib do, so %K starts on period
          `window` + `stochastic_window` - 1 and %D `smooth_window` - 1 periods later.
          TA-Lib starts both outputs on the first %D, with the same values from there.
        - An RSI that did not move over the stochastic window has no range, which leaves %K
          undefined (missing), as in `get_stochastic_oscillator`; TA-Lib sets it to 0.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods for the RSI.
        stochastic_window (int): Number of periods of RSI values for the high-low range of %K.
        smooth_window (int): Number of periods of the Simple Moving Average of %K that gives %D.

    Returns:
        pd.DataFrame: Stochastic RSI %K and %D as columns, bounded between 0 and 100.
    """
    relative_strength_index = get_relative_strength_index(prices, window)

    stochastic_rsi = get_stochastic_oscillator(
        relative_strength_index,
        relative_strength_index,
        relative_strength_index,
        stochastic_window,
        smooth_window,
    )
    stochastic_rsi.columns = ["Stochastic RSI %K", "Stochastic RSI %D"]

    return stochastic_rsi


def get_true_strength_index(
    prices: pd.Series, long_window: int, short_window: int, signal_window: int
) -> pd.DataFrame:
    """
    Calculate the True Strength Index (TSI) of a given price series.

    The True Strength Index smooths the one-period price change twice with Exponential Moving
    Averages and divides it by the same double smoothing of the absolute change. The double
    smoothing filters out noise while lagging less than one long average would, and dividing
    by the smoothed absolute change bounds the result between -100 (every change down) and
    +100 (every change up). Crossings of the zero line and of the signal line, and readings
    beyond about +25 or -25, are the usual signals.

    The formula is as follows:

    - Momentum(t) = Close(t) - Close(t - 1)
    - TSI = 100 * EMA(EMA(Momentum, long_window), short_window) / EMA(EMA(|Momentum|,
      long_window), short_window)
    - Signal Line = EMA(TSI, signal_window)

    Blau's parameters are 25 for the first and 13 for the second smoothing; the signal line
    of 7 periods is the one StockCharts uses.

    Also known as: TSI, double smoothed momentum.

    Reference: Blau, W. (1995). "Momentum, Direction, and Divergence." Wiley; first published
    in Blau, W. (1991). "True Strength Index." Technical Analysis of Stocks & Commodities,
    9(11).

    TA-Lib: TSI, the reference implementation (https://ta-lib.org/functions/tsi.html).

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values with an
          empty warm-up (see `get_exponential_moving_average`), as TA-Lib and StockCharts
          do, so the TSI starts on period `long_window` + `short_window` - 1 (the first
          price has no change) and the signal line `signal_window` - 1 periods later.
        - A stretch without any price change has no absolute change to divide by, which
          leaves the TSI undefined (missing).

    Args:
        prices (pd.Series): Series of prices.
        long_window (int): Number of periods of the first EMA of the price change.
        short_window (int): Number of periods of the second EMA.
        signal_window (int): Number of periods of the EMA of the TSI that gives the signal
            line.

    Returns:
        pd.DataFrame: True Strength Index and Signal Line as columns, bounded between -100
            and 100.
    """
    momentum = prices.diff(1)

    double_smoothed_momentum = get_exponential_moving_average(
        get_exponential_moving_average(momentum, long_window), short_window
    )
    double_smoothed_absolute_momentum = get_exponential_moving_average(
        get_exponential_moving_average(momentum.abs(), long_window), short_window
    )

    true_strength_index = (
        100
        * double_smoothed_momentum
        / double_smoothed_absolute_momentum.where(
            double_smoothed_absolute_momentum != 0
        )
    )
    signal_line = get_exponential_moving_average(true_strength_index, signal_window)

    return pd.concat(
        [true_strength_index, signal_line],
        keys=["True Strength Index", "Signal Line"],
        axis=1,
    )


def get_balance_of_power(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.Series:
    """
    Calculate the Balance of Power (BOP) indicator for a given price series.

    The Balance of Power (BOP) indicator measures the strength of buyers and sellers in the market.
    It considers the relationship between the closing price and the trading range (high - low) of each period.
    BOP values above zero suggest bullish buying pressure, while values below zero suggest bearish selling pressure.

    The formula is as follows:

    - BOP = (Close - Open) / (High - Low)

    Also known as: BOP.

    Reference: Created by Igor Livshin; featured in Kaufman, P.J. (2005). "New Trading
    Systems and Methods." 4th ed. Wiley.

    TA-Lib: BOP, the reference implementation (https://ta-lib.org/functions/bop.html).

    Notes:
        - A period whose high equals its low has no range to divide by and is set to 0, as
          in TA-Lib's `ta_BOP.c`. A period with a missing price stays missing.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: BOP values.
    """
    price_range = prices_high - prices_low

    # Only a flat period (High == Low) with known prices is a balance of zero; a missing price is unknown rather than balanced.  # noqa: E501
    bop_values = (prices_close - prices_open) / price_range.where(price_range != 0)
    flat_period = (price_range == 0) & prices_close.notna() & prices_open.notna()

    return bop_values.mask(flat_period, 0.0)


def get_awesome_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    short_window: int = 5,
    long_window: int = 34,
) -> pd.Series:
    """
    Calculate the Awesome Oscillator (AO) for a given price series.

    The Awesome Oscillator measures market momentum by comparing a short-term and a long-term
    Simple Moving Average of the median price (the midpoint of each period's high and low,
    rather than the closing price). It was developed by Bill Williams as part of his broader
    "Trading Chaos" collection of momentum indicators.

    The formula is as follows:

    - Median Price = (High + Low) / 2
    - AO = SMA(Median Price, short_window) - SMA(Median Price, long_window)

    Also known as: AO, Bill Williams Awesome Oscillator.

    Notes:
        - There is no academic journal citation for the Awesome Oscillator. Like most of Bill
          Williams' indicators, it is a practitioner-developed tool rather than one derived from
          a published financial paper. The standard textbook source is Williams, B. (1995).
          "Trading Chaos: Applying Expert Techniques to Maximize Your Profit." Wiley.
        - A cross of the AO above zero occurs exactly when the short-window SMA of the median
          price crosses above the long-window SMA, and vice versa for a cross below zero.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        short_window (int): Number of periods for the short-term SMA of the median price.
            Defaults to 5.
        long_window (int): Number of periods for the long-term SMA of the median price.
            Defaults to 34.

    Returns:
        pd.Series: Awesome Oscillator values. Positive values (and a cross above zero) signal
            building bullish momentum; negative values (and a cross below zero) signal building
            bearish momentum.
    """
    median_price = (prices_high + prices_low) / 2

    short_sma = get_moving_average(median_price, short_window)
    long_sma = get_moving_average(median_price, long_window)

    return short_sma - long_sma


def get_vortex_indicator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """
    Calculate the Vortex Indicator for a given price series.

    The Vortex Indicator quantifies the presence and strength of a directional trend by
    comparing each period's price movement away from the prior period's range to the period's
    overall volatility (True Range). It consists of two lines, VI+ and VI-, whose crossovers
    signal potential trend changes: VI+ above VI- suggests an uptrend is in control, VI- above
    VI+ suggests a downtrend is in control.

    The formula is as follows:

    - VM+ = |High(t) - Low(t-1)|
    - VM- = |Low(t) - High(t-1)|
    - VI+ = Sum(VM+, window) / Sum(True Range, window)
    - VI- = Sum(VM-, window) / Sum(True Range, window)

    Also known as: VI, Vortex Indicator +/-, trend direction indicator.

    Reference: Botes, E., & Siepman, D. (2010). "The Vortex Indicator." Technical Analysis of
    Stocks & Commodities, 28(1), 20-25.

    TA-Lib: VORTEX, the reference implementation (https://ta-lib.org/functions/vortex.html).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to sum the directional movement and true range over.
            Defaults to 14.

    Returns:
        pd.DataFrame: Vortex Indicator (VI+ and VI-) values.
    """
    positive_vortex_movement = (prices_high - prices_low.shift(1)).abs()
    negative_vortex_movement = (prices_low - prices_high.shift(1)).abs()

    true_range = get_true_range(prices_high, prices_low, prices_close)

    positive_vortex_indicator = (
        positive_vortex_movement.rolling(window=window).sum()
        / true_range.rolling(window=window).sum()
    )
    negative_vortex_indicator = (
        negative_vortex_movement.rolling(window=window).sum()
        / true_range.rolling(window=window).sum()
    )

    return pd.concat(
        [positive_vortex_indicator, negative_vortex_indicator],
        keys=["VI+", "VI-"],
        axis=1,
    )


def get_elder_ray_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 13,
) -> pd.DataFrame:
    """
    Calculate the Elder Ray Index (Bull Power and Bear Power) for a given price series.

    The Elder Ray Index measures buying and selling pressure in the market relative to a trend
    baseline (an Exponential Moving Average of the closing price). Bull Power captures how far
    the high extends above the EMA (buying pressure), while Bear Power captures how far the low
    extends below the EMA (selling pressure).

    The formula is as follows:

    - Bull Power = High - EMA(Close, window)
    - Bear Power = Low - EMA(Close, window)

    Also known as: Elder Ray, Bull Power, Bear Power.

    Reference: Elder, A. (1993). "Trading for a Living." Wiley.

    TA-Lib: ERI, the reference implementation (https://ta-lib.org/functions/eri.html).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the EMA used as the trend baseline. Defaults to 13,
            as originally proposed by Elder.

    Returns:
        pd.DataFrame: Elder Ray Index (Bull Power and Bear Power) values. When the close is
            above the EMA (uptrend), Bull Power tends to stay positive and Bear Power moves
            toward zero from below; when the close is below the EMA (downtrend), both tend to
            be negative.
    """
    exponential_moving_average = get_exponential_moving_average(prices_close, window)

    bull_power = prices_high - exponential_moving_average
    bear_power = prices_low - exponential_moving_average

    return pd.concat(
        [bull_power, bear_power], keys=["Bull Power", "Bear Power"], axis=1
    )


def get_rate_of_change(
    prices_close: pd.Series | pd.DataFrame, window: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Rate of Change (ROC) for a given price series.

    The Rate of Change is a pure momentum oscillator that measures the percentage change
    in price between the current period and the price a fixed number of periods ago. It
    oscillates around zero: positive values indicate price is higher than `window` periods
    ago (upward momentum), while negative values indicate price is lower (downward momentum).

    The formula is as follows:

    - ROC = Close(t) / Close(t - window) - 1

    Also known as: ROC, Price Rate of Change, momentum (in its unscaled form Close(t) -
    Close(t - window)).

    Reference: Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York
    Institute of Finance.

    TA-Lib: ROCP, the reference implementation (https://ta-lib.org/functions/rocp.html).

    Args:
        prices_close (pd.Series | pd.DataFrame): Series (or DataFrame with one column per
            ticker) of closing prices.
        window (int): Number of periods to look back for the rate of change calculation.

    Returns:
        pd.Series | pd.DataFrame: Rate of Change values as a decimal (0.05 is 5%).

    Raises:
        TypeError: If `prices_close` is not a pandas Series or DataFrame.
    """
    if not isinstance(prices_close, pd.Series | pd.DataFrame):
        raise TypeError("prices_close must be a pandas Series or DataFrame.")

    return prices_close / prices_close.shift(window) - 1


def get_choppiness_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    range_method: str = "high_low",
) -> pd.Series:
    """
    Calculate the Choppiness Index (CHOP) for a given price series.

    The Choppiness Index quantifies whether the market is trending or moving sideways
    ("choppy") by comparing the sum of True Range over the window (a measure of the total
    price path travelled) to the net range the price actually covered over that same window
    (the distance between the highest high and the lowest low). When price travels a long,
    winding path but ends up covering little net ground, the index is high (near 100),
    signalling a choppy, range-bound market. When price travels efficiently in one direction,
    the index is low (near 0), signalling a trending market.

    The formula is as follows:

    - CHOP = 100 * log10( Sum(True Range, window) / (Max(High, window) - Min(Low, window)) ) / log10(window)

    With `range_method="true_range"` the net range is that of the true highs and lows, the
    range the True Ranges themselves span, so a gap between periods counts in both:

    - True High = Max(High, Previous Close), True Low = Min(Low, Previous Close)
    - CHOP = 100 * log10( Sum(True Range, window) / (Max(True High, window) - Min(True Low, window)) ) / log10(window)

    Also known as: CHOP, CHOPTR (the true range box), Choppiness Index.

    Reference: Dreiss, E.W. (1990s). Developed by Australian commodities trader Bill Dreiss;
    there is no formal journal citation. The standard textbook treatment is Kaufman, P.J.
    (2013). "Trading Systems and Methods." 5th ed. Wiley.

    TA-Lib: CHOP and CHOPTR, the reference implementations
    (https://ta-lib.org/functions/chop.html, https://ta-lib.org/functions/choptr.html).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to consider for the Choppiness Index calculation.
        range_method (str): The net range the summed True Range is divided by: "high_low",
            the highest high minus the lowest low, or "true_range", the highest true high
            minus the lowest true low (TA-Lib's CHOPTR). Defaults to "high_low".

    Returns:
        pd.Series: Choppiness Index values, bounded between 0 and 100. Values above 61.8 are
            commonly read as signalling a choppy (range-bound) market, while values below 38.2
            are commonly read as signalling a trending market.

    Raises:
        TypeError: If `prices_high`, `prices_low` or `prices_close` is not a pandas Series.
        ValueError: If `window` is smaller than 2 or `range_method` is not "high_low" or
            "true_range".
    """
    if not (
        isinstance(prices_high, pd.Series)
        and isinstance(prices_low, pd.Series)
        and isinstance(prices_close, pd.Series)
    ):
        raise TypeError(
            "prices_high, prices_low and prices_close must be pandas Series."
        )
    # The index divides by log10(window), which is 0 for a single period.
    if window < MINIMUM_WINDOW:
        raise ValueError("window must be at least 2 for the Choppiness Index.")
    if range_method not in ("high_low", "true_range"):
        raise ValueError("range_method must be 'high_low' or 'true_range'.")

    true_range = get_true_range(prices_high, prices_low, prices_close)

    if range_method == "true_range":
        # The first period has no previous close, so its true high and low are its own
        # high and low, as its True Range is its High - Low range.
        previous_close = prices_close.shift(1)
        prices_high = prices_high.where(~(previous_close > prices_high), previous_close)
        prices_low = prices_low.where(~(previous_close < prices_low), previous_close)

    summed_true_range = true_range.rolling(window=window).sum()
    highest_high = prices_high.rolling(window=window).max()
    lowest_low = prices_low.rolling(window=window).min()

    choppiness_index = (
        100
        * np.log10(summed_true_range / (highest_high - lowest_low))
        / np.log10(window)
    )

    return choppiness_index


def get_know_sure_thing(
    prices_close: pd.Series,
    roc_windows: list[int] | None = None,
    sma_windows: list[int] | None = None,
    weights: list[int] | None = None,
    signal_window: int = 9,
    moving_average: str = "simple",
    signal_moving_average: str = "simple",
) -> pd.DataFrame:
    """
    Calculate the Know Sure Thing (KST) for a given price series.

    The Know Sure Thing is a momentum oscillator developed by Martin Pring that combines
    four smoothed Rate of Change series, each calculated over a progressively longer lookback
    period, into a single weighted sum. Smoothing each Rate of Change with a Simple Moving
    Average before combining them reduces noise, while the increasing weights on the longer
    lookback periods give more influence to the more significant, longer-term price cycles.
    A signal line (a Simple Moving Average of the KST itself) is used to spot crossovers, in
    the same way the MACD line is compared to its signal line.

    The formula is as follows:

    - RCMA(i) = SMA(ROC(Close, roc_windows[i]), sma_windows[i])
    - KST = Sum(RCMA(i) * weights[i]) for i = 1..4
    - Signal Line = SMA(KST, signal_window)

    Pring smooths with Simple Moving Averages; `moving_average` and `signal_moving_average`
    take another moving average for the smoothing of the Rates of Change and the signal
    line, as TA-Lib's KSTEXT does with its `rocmatype` and `signalmatype`.

    Also known as: KST, KSTEXT, Pring's Know Sure Thing, Summed Rate of Change.

    Reference: Pring, M.J. (1992). "The Know Sure Thing (KST)." Technical Analysis of Stocks
    & Commodities, 10(6).

    TA-Lib: KST and KSTEXT, the reference implementations
    (https://ta-lib.org/functions/kst.html, https://ta-lib.org/functions/kstext.html).

    Args:
        prices_close (pd.Series): Series of closing prices.
        roc_windows (list[int] | None): The four lookback periods used for the underlying
            Rate of Change calculations. Defaults to the standard [10, 15, 20, 30].
        sma_windows (list[int] | None): The four Simple Moving Average smoothing periods
            applied to each Rate of Change series. Defaults to the standard [10, 10, 10, 15].
        weights (list[int] | None): The four weights applied to each smoothed Rate of Change
            series before summing. Defaults to the standard [1, 2, 3, 4].
        signal_window (int): Number of periods for the Simple Moving Average of the KST used
            as the signal line. Defaults to 9.
        moving_average (str): The moving average that smooths each Rate of Change, one of
            `MOVING_AVERAGES`: "simple", "exponential", "double_exponential", "triangular",
            "weighted", "hull" or "kaufman_adaptive". Defaults to "simple".
        signal_moving_average (str): The moving average of the signal line, one of
            `MOVING_AVERAGES`. Defaults to "simple".

    Returns:
        pd.DataFrame: Know Sure Thing (KST and Signal Line) values, weighted sums of
            decimal Rates of Change.

    Raises:
        TypeError: If `prices_close` is not a pandas Series.
        ValueError: If `roc_windows`, `sma_windows` and `weights` are not all length 4, or a
            moving average is not one of `MOVING_AVERAGES`.
    """
    if not isinstance(prices_close, pd.Series):
        raise TypeError("prices_close must be a pandas Series.")
    for name in (moving_average, signal_moving_average):
        if name not in MOVING_AVERAGES:
            raise ValueError(
                f"moving_average must be one of {', '.join(MOVING_AVERAGES)}, not {name}."
            )

    if roc_windows is None:
        roc_windows = [10, 15, 20, 30]
    if sma_windows is None:
        sma_windows = [10, 10, 10, 15]
    if weights is None:
        weights = [1, 2, 3, 4]

    if (
        len(roc_windows) != KST_COMPONENT_COUNT
        or len(sma_windows) != KST_COMPONENT_COUNT
        or len(weights) != KST_COMPONENT_COUNT
    ):
        raise ValueError(
            "roc_windows, sma_windows and weights must each contain exactly 4 values."
        )

    kst = pd.Series(0.0, index=prices_close.index)
    for roc_window, sma_window, weight in zip(roc_windows, sma_windows, weights):
        rate_of_change = get_rate_of_change(prices_close, roc_window)
        smoothed_rate_of_change = MOVING_AVERAGES[moving_average](
            rate_of_change, sma_window
        )
        kst = kst + smoothed_rate_of_change * weight

    signal_line = MOVING_AVERAGES[signal_moving_average](kst, signal_window)

    return pd.concat([kst, signal_line], keys=["KST", "Signal Line"], axis=1)


def get_coppock_curve(
    prices_close: pd.Series | pd.DataFrame,
    long_roc_window: int = 14,
    short_roc_window: int = 11,
    wma_window: int = 10,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Coppock Curve for a given price series.

    The Coppock Curve is a long-term momentum indicator that Edwin Coppock designed for
    monthly closes of a stock index, to time buying after a bear market. He took the 11 and
    14 months from the time an Episcopal bishop told him mourning takes, as the period
    investors need to get over a loss. The classic signal is the curve turning up from below
    zero; Coppock did not use it for sell signals.

    The formula is as follows:

    - Coppock Curve = WMA(ROC(Close, long_roc_window) + ROC(Close, short_roc_window),
      wma_window)

    with ROC(Close, n) = Close(t) / Close(t - n) - 1 and WMA the linearly Weighted Moving
    Average, so the defaults give WMA10(ROC14 + ROC11), in months for the original curve.

    Also known as: Coppock Indicator, Coppock Guide.

    Reference: Coppock, E.S.C. (1962). "Practical Relative Strength Charting." Trendex
    Research Group; first presented in Barron's, October 15, 1962.

    TA-Lib: COPPOCK, the reference implementation (https://ta-lib.org/functions/coppock.html).

    Notes:
        - The Rates of Change are decimals, as in `get_rate_of_change`, so the curve is a
          weighted sum of decimal returns (0.30 where charting platforms show 30).
        - The curve starts on period `max(long_roc_window, short_roc_window)` +
          `wma_window` - 1.

    Args:
        prices_close (pd.Series | pd.DataFrame): Series (or DataFrame with one column per
            ticker) of closing prices, originally monthly.
        long_roc_window (int): Number of periods of the longer Rate of Change. Defaults to 14.
        short_roc_window (int): Number of periods of the shorter Rate of Change. Defaults
            to 11.
        wma_window (int): Number of periods of the Weighted Moving Average of the summed Rates
            of Change. Defaults to 10.

    Returns:
        pd.Series | pd.DataFrame: Coppock Curve values, a weighted sum of decimal Rates of
            Change.
    """
    summed_rate_of_change = get_rate_of_change(
        prices_close, long_roc_window
    ) + get_rate_of_change(prices_close, short_roc_window)

    return get_weighted_moving_average(summed_rate_of_change, wma_window)


def get_absolute_price_oscillator(
    prices_close: pd.Series,
    short_window: int,
    long_window: int,
    moving_average: str = "exponential",
) -> pd.Series:
    """
    Calculate the Absolute Price Oscillator (APO) for a given price series.

    The Absolute Price Oscillator is the difference between a short-term and a long-term
    moving average of price, in price units. With exponential moving averages it is the MACD
    line; the APO lets the moving average be chosen, and the Percentage Price Oscillator is
    the same difference as a share of the long-term average.

    The formula is as follows:

    - APO = MA(Close, short_window) - MA(Close, long_window)

    The conventional parameters are 12 and 26 periods, as for MACD.

    Also known as: APO, price oscillator, MACD line (with EMAs).

    Reference: Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method."
    Signalert Corp.; Achelis, S.B. (2000). "Technical Analysis from A to Z." 2nd ed.
    McGraw-Hill ("Price Oscillator").

    TA-Lib: APO, the reference implementation (https://ta-lib.org/functions/apo.html), with
    the moving average as its `matype` (EMA by default).

    Notes:
        - Each moving average starts on its own first full window, so with EMAs seeded with a
          Simple Moving Average the APO starts on period `long_window` - 1 and equals the
          MACD line of `get_moving_average_convergence_divergence`.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        moving_average (str): The moving average, one of `MOVING_AVERAGES`: "simple",
            "exponential", "double_exponential", "triangular", "weighted", "hull" or
            "kaufman_adaptive". Defaults to "exponential".

    Returns:
        pd.Series: APO values, in price units.

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    if moving_average not in MOVING_AVERAGES:
        raise ValueError(
            f"moving_average must be one of {', '.join(MOVING_AVERAGES)}, not {moving_average}."
        )

    return MOVING_AVERAGES[moving_average](
        prices_close, short_window
    ) - MOVING_AVERAGES[moving_average](prices_close, long_window)


def get_momentum(prices_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Momentum (MOM) of a given price series.

    Momentum is the change in price over a fixed number of periods, in price units: the
    simplest measure of the speed of a move. Above zero the price is higher than `window`
    periods ago, below zero lower; a falling momentum while the price still rises says the
    move slows down. The Rate of Change is the same change as a share of the earlier price.

    The formula is as follows:

    - Momentum = Close(t) - Close(t - window)

    Also known as: MOM, price momentum, momentum oscillator.

    Reference: Pring, M.J. (1991). "Technical Analysis Explained." 3rd ed. McGraw-Hill;
    Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York Institute
    of Finance.

    TA-Lib: MOM, the reference implementation (https://ta-lib.org/functions/mom.html).

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to look back.

    Returns:
        pd.Series: Momentum values, in price units.
    """
    return prices_close - prices_close.shift(window)


def get_accelerator_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    short_window: int = 5,
    long_window: int = 34,
    signal_window: int = 5,
) -> pd.Series:
    """
    Calculate the Accelerator/Decelerator Oscillator (AC) for a given price series.

    The Accelerator Oscillator is the Awesome Oscillator minus its own 5-period Simple Moving
    Average. Where the Awesome Oscillator measures the momentum of the median price, the
    Accelerator Oscillator measures how that momentum changes, its acceleration, on the
    premise that momentum changes direction before price does. Bill Williams developed it
    as part of his "Trading Chaos" indicators, next to the Awesome Oscillator.

    The formula is as follows:

    - Median Price = (High + Low) / 2
    - AO = SMA(Median Price, short_window) - SMA(Median Price, long_window)
    - AC = AO - SMA(AO, signal_window)

    Also known as: AC, Acceleration/Deceleration Oscillator, AC oscillator.

    Reference: Williams, B. (1998). "New Trading Dimensions: How to Profit from Chaos in
    Stocks, Bonds, and Commodities." Wiley; Williams, B. (1995). "Trading Chaos." Wiley.

    TA-Lib: AC, the reference implementation (https://ta-lib.org/functions/ac.html).

    Notes:
        - The AC starts on period `long_window` + `signal_window` - 2, the first full window
          of the Simple Moving Average of the Awesome Oscillator.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        short_window (int): Number of periods for the short-term SMA of the median price.
            Defaults to 5.
        long_window (int): Number of periods for the long-term SMA of the median price.
            Defaults to 34.
        signal_window (int): Number of periods for the SMA of the Awesome Oscillator.
            Defaults to 5.

    Returns:
        pd.Series: Accelerator Oscillator values, in price units.
    """
    awesome_oscillator = get_awesome_oscillator(
        prices_high, prices_low, short_window, long_window
    )

    return awesome_oscillator - get_moving_average(awesome_oscillator, signal_window)


def get_internal_bar_strength(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series
) -> pd.Series:
    """
    Calculate the Internal Bar Strength (IBS) of a given price series.

    The Internal Bar Strength is where the close sits within the period's own range: 0 for
    a close on the low, 1 for a close on the high. It is a mean-reversion measure: a close
    near the low of the day (a low IBS) has tended to be followed by higher returns in
    equity indices and ETFs, and a close near the high by lower returns.

    The formula is as follows:

    - IBS = (Close - Low) / (High - Low)

    Also known as: IBS, close location value (as a 0 to 1 ratio).

    Reference: Pagonidis, A.S. (2013). "The IBS Effect: Mean Reversion in Equity ETFs."
    NAAIM Wagner Award paper.

    TA-Lib: IBS, the reference implementation (https://ta-lib.org/functions/ibs.html).

    Notes:
        - A period without a range (High equal to Low) has no IBS (NaN), where TA-Lib
          returns 0.5.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: IBS values as a decimal between 0 and 1.
    """
    price_range = prices_high - prices_low

    return (prices_close - prices_low) / price_range.where(price_range != 0)


def get_intraday_momentum_index(
    prices_open: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Intraday Momentum Index (IMI) for a given price series.

    The Intraday Momentum Index is a Relative Strength Index of the moves within each
    period: it compares the sum of the bodies of the periods that closed above their open
    to the sum of all bodies over the window. Chande designed it for intraday traders, who
    care about the move from open to close rather than from close to close.

    The formula is as follows:

    - Gain = Close - Open where Close > Open, else 0
    - Loss = Open - Close where Close < Open, else 0
    - IMI = 100 * Sum(Gain, window) / (Sum(Gain, window) + Sum(Loss, window))

    Values above 70 are read as overbought and below 30 as oversold, as for the RSI.

    Also known as: IMI.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    TA-Lib: IMI, the reference implementation (https://ta-lib.org/functions/imi.html).

    Notes:
        - A window in which every period closed at its open has no IMI (NaN), where TA-Lib
          returns 50.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to sum the gains and losses over.

    Returns:
        pd.Series: IMI values, between 0 and 100.
    """
    body = prices_close - prices_open

    gains = body.clip(lower=0).rolling(window=window).sum()
    total = body.abs().rolling(window=window).sum()

    return 100 * gains / total.where(total != 0)


def get_qstick(
    prices_open: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Qstick indicator for a given price series.

    Qstick is the moving average of the candlestick bodies, Close - Open, over the window.
    It turns the colour of the recent candles into a number: above zero the periods closed
    above their open on average (white candles, buying pressure), below zero below their
    open (black candles, selling pressure).

    The formula is as follows:

    - Qstick = SMA(Close - Open, window)

    Also known as: Qstick, QSTICK, candlestick body average.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    TA-Lib: QSTICK, the reference implementation (https://ta-lib.org/functions/qstick.html).

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the moving average.

    Returns:
        pd.Series: Qstick values, in price units.
    """
    return get_moving_average(prices_close - prices_open, window)


def get_efficiency_ratio(prices_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate Kaufman's Efficiency Ratio (ER) for a given price series.

    The Efficiency Ratio compares the net change in price over the window to the path price
    travelled to get there, the sum of the absolute changes of each period. A price that
    moves in a straight line has a ratio of 1; a price that ends where it started however
    much it moved in between has a ratio of 0. Kaufman uses it to adapt the speed of his
    Adaptive Moving Average to the noise in the market.

    The formula is as follows:

    - Change = |Close(t) - Close(t - window)|
    - Volatility = Sum(|Close(i) - Close(i - 1)|, window)
    - ER = Change / Volatility

    Also known as: ER, Kaufman Efficiency Ratio, fractal efficiency.

    Reference: Kaufman, P.J. (1995). "Smarter Trading: Improving Performance in Changing
    Markets." McGraw-Hill.

    TA-Lib: ER, the reference implementation (https://ta-lib.org/functions/er.html).

    Notes:
        - The ratio is the one `get_kaufman_adaptive_moving_average` uses, except that a
          window without any change (flat prices) has no ratio (NaN), where TA-Lib's ER and
          KAMA use 1.

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the change and the path.

    Returns:
        pd.Series: Efficiency Ratio values as a decimal between 0 and 1.
    """
    change = (prices_close - prices_close.shift(window)).abs()
    volatility = prices_close.diff().abs().rolling(window=window).sum()

    return change / volatility.where(volatility != 0)


def get_vertical_horizontal_filter(prices_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Vertical Horizontal Filter (VHF) for a given price series.

    The Vertical Horizontal Filter tells a trending market from a congested one by dividing
    the range of the closes over the window (the vertical move) by the sum of the absolute
    changes from period to period (the path, horizontal in a range). A high VHF says the
    price moves in one direction, which suits trend-following indicators; a low VHF says it
    moves back and forth, which suits oscillators.

    The formula is as follows:

    - VHF = (Max(Close, window) - Min(Close, window)) / Sum(|Close(i) - Close(i - 1)|, window)

    Also known as: VHF.

    Reference: White, A. (1991). "Trending or Trading? The Vertical Horizontal Filter."
    Futures Magazine, August 1991.

    TA-Lib: VHF, the reference implementation (https://ta-lib.org/functions/vhf.html).

    Notes:
        - The range spans the last `window` closes and the path the last `window` changes,
          so the VHF starts on period `window`, as in TA-Lib.
        - A window without any change (flat prices) has no VHF (NaN), where TA-Lib returns
          0.

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the range and the path.

    Returns:
        pd.Series: VHF values, a ratio between 0 and 1.
    """
    highest_close = prices_close.rolling(window=window).max()
    lowest_close = prices_close.rolling(window=window).min()
    volatility = prices_close.diff().abs().rolling(window=window).sum()

    return (highest_close - lowest_close) / volatility.where(volatility != 0)
