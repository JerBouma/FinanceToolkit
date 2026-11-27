"""Momentum Module"""

__docformat__ = "google"


import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from financetoolkit.econometrics.statistics.statistics_model import (
    get_linear_regression,
    get_percent_rank,
)
from financetoolkit.technicals.overlap.overlap_model import (
    apply_to_rolling_windows,
    get_double_exponential_moving_average,
    get_elastic_volume_weighted_moving_average,
    get_exponential_moving_average,
    get_hull_moving_average,
    get_kaufman_adaptive_moving_average,
    get_moving_average,
    get_triangular_moving_average,
    get_volume_weighted_moving_average,
    get_weighted_moving_average,
)
from financetoolkit.technicals.volatility.volatility_model import (
    get_average_true_range,
    get_relative_volatility_index,
    get_true_range,
    get_wilder_moving_average,
)
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# The Know Sure Thing combines exactly four smoothed Rate of Change components.
KST_COMPONENT_COUNT = 4

# The shortest window that has a deviation from its own mean (CCI) and a log10(window) above
# zero (Choppiness Index).
MINIMUM_WINDOW = 2

# The moving averages the oscillators built on two averages (MACD, APO, PPO, KST) can use, as
# TA-Lib's MA types: SMA (0), EMA (1), WMA (2), DEMA (3), TRIMA (5) and KAMA (6), plus the Hull
# moving average, each one of the methods of get_moving_average.
MOVING_AVERAGES = {
    "simple": get_moving_average,
    "exponential": get_exponential_moving_average,
    "double_exponential": get_double_exponential_moving_average,
    "triangular": get_triangular_moving_average,
    "weighted": get_weighted_moving_average,
    "hull": get_hull_moving_average,
    "kaufman_adaptive": get_kaufman_adaptive_moving_average,
}

# The volume weighted averages the MACD can also use, which take the volume next to the
# prices: Dormeier's VWMA (the Volume Weighted MACD) and Fries' eVWMA (the Elastic Volume
# Weighted MACD).
VOLUME_MOVING_AVERAGES = {
    "volume_weighted": get_volume_weighted_moving_average,
    "elastic_volume_weighted": get_elastic_volume_weighted_moving_average,
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
    Calculate the Money Flow Index (MFI): how strongly money flows into or out of a security, using price and
    volume.

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

    Referenced against:
    - TA-Lib, MFI: https://ta-lib.org/functions/mfi.html

    Notes:
        - The first period has no prior typical price and therefore no money flow, so the
          first MFI value is on period `window` and covers the flows of periods 1 to
          `window`, the same start as TA-Lib's MFI. A missing price or volume leaves its flow
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
    Calculate the Williams %R: where the close lies within the high-low range of a window, from 0 (at the high) to
    -100 (at the low), to show overbought and oversold levels.

    Williams %R measures where the current close sits within the high-low range of the
    lookback window, expressed as a percentage that runs from 0 (close at the window's
    highest high) down to -100 (close at the window's lowest low). It is mathematically a
    rescaled mirror image of the Fast Stochastic %K.

    The formula is as follows:

    - %R = -((Highest High - Close) / (Highest High - Lowest Low)) * 100

    Also known as: %R, Williams %R.

    Reference: Williams, L.R. (1973). "How I Made One Million Dollars Last Year Trading
    Commodities." Windsor Books.

    Referenced against:
    - TA-Lib, WILLR: https://ta-lib.org/functions/willr.html

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for %R calculation.

    Returns:
        pd.Series: Williams %R values.
    """
    # %R is the Fast Stochastic %K moved down by 100: -(HH - C) / (HH - LL) = (C - LL) / (HH - LL) - 1.
    stochastic_k = get_stochastic_oscillator(
        prices_high, prices_low, prices_close, window, smooth_window=1
    )["Stochastic %K"]

    return (stochastic_k - 100).rename(None)


def get_aroon_indicator(
    prices_high: pd.Series, prices_low: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate the Aroon Indicator: how many periods ago the highest high and the lowest low of a window were, which
    shows whether a trend is starting and how strong it is.

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

    Reference: Chande, T.S. (1995). "The Time Price Oscillator." Technical Analysis of Stocks
    & Commodities, 13(9), 369-374; Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    Referenced against:
    - TA-Lib, AROON: https://ta-lib.org/functions/aroon.html
    - TA-Lib, AROONOSC: https://ta-lib.org/functions/aroonosc.html

    Notes:
        - The lookback spans `window + 1` bars — the current period plus the previous
          `window` — the span TA-Lib's AROON scans as well. A plain `window`-bar rolling
          lookback can never return 0 and mis-scales every reading.
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
    Calculate the Commodity Channel Index (CCI): how far the typical price lies from its average, in units of its
    mean deviation, to show cyclical turns and overbought and oversold levels.

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
    Trends." Commodities, October 1980; reprinted in Technical Analysis of Stocks &
    Commodities, 1(5), 120-122.

    Referenced against:
    - TA-Lib, CCI: https://ta-lib.org/functions/cci.html

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
    Calculate the Relative Vigor Index (RVI): how strongly the close ends above the open relative to the range,
    which shows the conviction behind a move.

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
    Commodities, 20(1), 16-20.

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
    Calculate the Force Index: the price change times the volume, smoothed, which shows the force of the buyers or
    sellers behind a move.

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

    Referenced against:
    - TA-Lib, EFI: https://ta-lib.org/functions/efi.html

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
    Calculate the Ultimate Oscillator: the buying pressure over three windows combined into one value from 0 to 100,
    to show overbought and oversold levels with fewer false signals.

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

    Referenced against:
    - TA-Lib, ULTOSC: https://ta-lib.org/functions/ultosc.html

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
    Calculate the Percentage Price Oscillator (PPO): the difference between a fast and a slow average of the prices
    as a share of the slow one, the MACD in percentage terms, so tickers of different prices compare.

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

    Referenced against:
    - TA-Lib, PPO: https://ta-lib.org/functions/ppo.html

    Shares similar logic:
    - pandas-ta-classic, ppo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/ppo.py#L48
    - ta, PercentagePriceOscillator: https://github.com/bukosabino/ta/blob/0.11.0/ta/momentum.py#L622
    - finta, PPO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L541

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values (see
          `get_exponential_moving_average`), so the PPO starts on period `long_window` - 1
          and the signal line and histogram `signal_window` - 1 periods later.
        - TA-Lib's PPO takes the moving average as its moving average type and returns the PPO in percent, without a
          signal line; the signal line and histogram are those of ta, pandas-ta-classic and finta.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        signal_window (int): Number of periods for the signal line EMA. Defaults to 9.
        moving_average (str): The moving average of the short and long term, one of `MOVING_AVERAGES`. Options
            are: "simple" (Simple Moving Average), "exponential" (Exponential Moving Average),
            "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular Moving Average),
            "weighted" (Weighted Moving Average), "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman
            Adaptive Moving Average). Defaults to "exponential".

    Returns:
        pd.DataFrame: The Percentage Price Oscillator, Signal Line and Histogram as columns,
        as decimals (0.02 is 2%).

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    check_option("moving_average", moving_average, MOVING_AVERAGES)

    short_average = MOVING_AVERAGES[moving_average](prices_close, short_window)
    long_average = MOVING_AVERAGES[moving_average](prices_close, long_window)

    # A long-term average of zero, such as that of a volume series without any volume, has no
    # oscillator rather than an infinite one.
    ppo = (short_average - long_average) / long_average.where(long_average != 0)
    signal_line = get_exponential_moving_average(ppo, signal_window)

    return pd.concat(
        [ppo, signal_line, ppo - signal_line],
        keys=["Percentage Price Oscillator", "Signal Line", "Histogram"],
        axis=1,
    )


def get_percentage_volume_oscillator(
    volumes: pd.Series,
    short_window: int,
    long_window: int,
    signal_window: int,
    moving_average: str = "exponential",
) -> pd.DataFrame:
    """
    Calculate the Percentage Volume Oscillator (PVO): the difference between a fast and a slow average of the volume
    as a share of the slow one, which shows whether volume is rising or falling.

    The Percentage Volume Oscillator is the Percentage Price Oscillator applied to volume:
    the difference between a short-term and a long-term EMA of volume as a share of the
    long-term EMA. It rises when recent volume runs above its longer-term average and
    falls when it runs below, which makes it a measure of volume momentum that can be
    compared across securities whatever the number of shares they trade.

    The formula is as follows:

    - PVO = (MA(Volume, short_window) - MA(Volume, long_window)) / MA(Volume, long_window)
    - Signal Line = EMA(PVO, signal_window)
    - Histogram = PVO - Signal Line

    with MA the Exponential Moving Average by default. With the Simple Moving Average
    it is the Volume Oscillator of StockCharts, pandas-ta-classic
    (`vosc`, 14 and 28 periods) and Tulip Indicators (`vosc`, 2 and 5 periods). The
    conventional parameters are 12, 26 and 9 periods, as for MACD and the PPO.

    Also known as: PVO, volume oscillator, VOSC.

    Reference: StockCharts ChartSchool, "Percentage Volume Oscillator (PVO)". The PVO is the
    volume counterpart of the Percentage Price Oscillator, a normalized variant of Appel's
    Moving Average Convergence Divergence (Appel, G. (1979). "The Moving Average
    Convergence-Divergence Trading Method." Signalert Corp.).

    Referenced against:
    - TA-Lib, PVO: https://ta-lib.org/functions/pvo.html
    - pandas-ta-classic, vosc: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/vosc.py#L12
    - Tulip Indicators, vosc: https://tulipindicators.org/vosc

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values (see
          `get_exponential_moving_average`), so the PVO starts on period `long_window` - 1
          and the signal line and histogram `signal_window` - 1 periods later.
        - With either moving average the PVO agrees with TA-Lib's PVO with the matching `matype`,
          pandas-ta-classic's `vosc` and Tulip's `vosc` (for the SMA) divided by 100: they
          are in percent and have no signal line.
        - A long-term EMA of zero (a series without volume, such as an index or a currency
          pair with no reported volume) has no PVO rather than an infinite one.
        - TA-Lib's PVO takes the moving average as its moving average type: the exponential or the simple moving
          average.

    Args:
        volumes (pd.Series): Series of trading volumes.
        short_window (int): Number of periods for the short-term EMA.
        long_window (int): Number of periods for the long-term EMA.
        signal_window (int): Number of periods for the signal line EMA.
        moving_average (str): The moving average of the volume. Options are: "exponential" (Exponential Moving
            Average), "simple" (Simple Moving Average). Defaults to "exponential".

    Returns:
        pd.DataFrame: The Percentage Volume Oscillator, Signal Line and Histogram as
        columns, as decimals (0.02 is 2%).

    Raises:
        ValueError: If `moving_average` is not "exponential" or "simple".
    """
    check_option("moving_average", moving_average, ("exponential", "simple"))

    # The PVO is the Percentage Price Oscillator of the volume.
    return get_percentage_price_oscillator(
        volumes, short_window, long_window, signal_window, moving_average
    ).rename(columns={"Percentage Price Oscillator": "Percentage Volume Oscillator"})


def get_detrended_price_oscillator(
    prices_close: pd.Series,
    window: int,
    moving_average_type: str = "sma",
) -> pd.Series:
    """
    Calculate the Detrended Price Oscillator (DPO): the price minus a moving average shifted back in time, which
    removes the trend to show the cycles.

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

    Referenced against:
    - TA-Lib, DPO: https://ta-lib.org/functions/dpo.html

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for DPO calculation.
        moving_average_type (str): The moving average to detrend against. Options are: "sma" (Simple Moving
            Average), "ema" (Exponential Moving Average). Defaults to "sma", the conventional choice.

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
        raise InvalidArgumentError(
            "Invalid moving average type. Choose either 'sma' or 'ema'."
        )

    displacement = int(window / 2) + 1

    dpo = prices_close.shift(displacement) - moving_average

    return dpo


def get_directional_movement_index(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate Wilder's Directional Movement System: the smoothed Plus and Minus Directional Movement (+DM, -DM), the
    Directional Indicators (+DI, -DI) and the Directional Movement Index (DX) of a given price series.

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

    Referenced against:
    - TA-Lib, PLUS_DM: https://ta-lib.org/functions/plus_dm.html
    - TA-Lib, MINUS_DM: https://ta-lib.org/functions/minus_dm.html
    - TA-Lib, PLUS_DI: https://ta-lib.org/functions/plus_di.html
    - TA-Lib, MINUS_DI: https://ta-lib.org/functions/minus_di.html
    - TA-Lib, DX: https://ta-lib.org/functions/dx.html

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
    # 2 * window - 1, the same start as TA-Lib's ADX, instead of averaging in a High - Low range
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
    Calculate the Average Directional Index (ADX): how strong a trend is, whatever its direction, from the smoothed
    difference between upward and downward movement.

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

    Referenced against:
    - TA-Lib, ADX: https://ta-lib.org/functions/adx.html

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
    Calculate the Average Directional Movement Index Rating (ADXR): the average of the ADX and its value a window
    earlier, a smoother measure of trend strength.

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

    Referenced against:
    - TA-Lib, ADXR: https://ta-lib.org/functions/adxr.html

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
    Calculate the Chande Momentum Oscillator (CMO): the gains minus the losses over a window as a share of their
    sum, from -100 to 100, to show momentum and overbought and oversold levels.

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

    Referenced against:
    - TA-Lib, CMOU: https://ta-lib.org/functions/cmou.html
    - TA-Lib, CMO: https://ta-lib.org/functions/cmo.html

    Notes:
        - TA-Lib's CMOU sums the gains and losses as Chande does, and its CMO smooths them with Wilder's method.

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
    Calculate the Ichimoku Cloud: the midpoints of the highs and lows over three windows, which show the trend, the
    momentum and the support and resistance of the prices at a glance.

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

    Referenced against:
    - TA-Lib, ICHIMOKU: https://ta-lib.org/functions/ichimoku.html

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
        - TA-Lib's ICHIMOKU places the leading spans on the period they are calculated on, without the shift
          forward.

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
    Calculate the Stochastic Oscillator: where the close lies within the high-low range of a window, from 0 to 100,
    to show momentum and overbought and oversold levels.

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
    Stochastic" smooths %K itself with an SMA first, which is the stochastic
    Lane charted and most platforms show by default; its %K is the fast %D.

    Also known as: Stochastic Oscillator, Fast Stochastic, Slow Stochastic, %K/%D, STOCHF,
    STOCH.

    Reference: Lane, G.C. Developed in the late 1950s; see Lane, G.C. (1984). "Lane's
    Stochastics." Technical Analysis of Stocks & Commodities, 2(3).

    Referenced against:
    - TA-Lib, STOCHF: https://ta-lib.org/functions/stochf.html
    - TA-Lib, STOCH: https://ta-lib.org/functions/stoch.html

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
    volumes: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD): the difference between a fast and a slow moving
    average of the prices and its signal line, which show the direction and the momentum of a trend.

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
    With the Volume Weighted Moving Average, MA(Close, n) = Sum(Close * Volume, n) /
    Sum(Volume, n), it is Dormeier's Volume Weighted MACD, and with Fries' Elastic Volume
    Weighted Moving Average the Elastic Volume Weighted MACD.

    Also known as: MACD, MACD histogram, MACDEXT (other moving averages), MACDFIX (fixed
    smoothing constants), VWMACD, volume weighted MACD, EVMACD, elastic volume weighted MACD.

    Reference: Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method."
    Signalert Corp. The histogram is Thomas Aspray's (1986), described in Aspray, T. (1988).
    "MACD Momentum." Technical Analysis of Stocks & Commodities, 6(8), 294-297. The volume
    weighted MACD is from
    Dormeier, B.P. (2011). "Investing with Volume Analysis." FT Press, and the elastic one
    builds on Fries, C.P. (2001). "Elastic Moving Averages." Technical Analysis of Stocks &
    Commodities, 19(6), 52-56.

    Referenced against:
    - TA-Lib, MACD: https://ta-lib.org/functions/macd.html
    - TA-Lib, MACDEXT: https://ta-lib.org/functions/macdext.html
    - TA-Lib, MACDFIX: https://ta-lib.org/functions/macdfix.html
    - pandas-ta-classic, vwmacd: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/vwmacd.py#L12

    Shares similar logic:
    - finta, VW_MACD: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L575
    - finta, EV_MACD: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L617

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values (see
          `get_exponential_moving_average`), so the MACD line starts on period
          `long_window` - 1 and the signal line `signal_window` - 1 periods later, as on
          StockCharts. TA-Lib's MACD additionally starts the short EMA on the same
          period as the long one, which changes only the first values.
        - `moving_average` applies to both the short and the long average, as TA-Lib's
          MACDEXT with the same `fastmatype` and `slowmatype`.
        - With "volume_weighted" for both averages the lines agree with pandas-ta-classic's
          vwmacd, which also weights the signal line by volume. finta's VW_MACD instead
          divides exponential averages of Close * Volume and of the volume, and its EV_MACD
          (20 and 40 periods by default) starts every eVWMA from 0 where the Toolkit seeds it
          with a simple average, so it equals this once that start has decayed.

    Args:
        prices (pd.Series): Series of prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        signal_window (int): Number of periods for the signal line moving average.
        moving_average (str): The moving average of the short and long term, one of `MOVING_AVERAGES` or
            `VOLUME_MOVING_AVERAGES`. Options are: "simple" (Simple Moving Average), "exponential" (Exponential
            Moving Average), "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular
            Moving Average), "weighted" (Weighted Moving Average), "hull" (Hull Moving Average),
            "kaufman_adaptive" (Kaufman Adaptive Moving Average), "volume_weighted" (Volume Weighted Moving
            Average, the Volume Weighted MACD), "elastic_volume_weighted" (Elastic Volume Weighted Moving
            Average, the Elastic Volume Weighted MACD). Defaults to "exponential".
        signal_moving_average (str): The moving average of the signal line, one of `MOVING_AVERAGES` or
            `VOLUME_MOVING_AVERAGES`. Options are: "simple" (Simple Moving Average), "exponential" (Exponential
            Moving Average), "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular
            Moving Average), "weighted" (Weighted Moving Average), "hull" (Hull Moving Average),
            "kaufman_adaptive" (Kaufman Adaptive Moving Average), "volume_weighted" (Volume Weighted Moving
            Average, the Volume Weighted MACD), "elastic_volume_weighted" (Elastic Volume Weighted Moving
            Average, the Elastic Volume Weighted MACD). Defaults to "exponential".
        fixed (bool): Whether to smooth the 12- and 26-period EMAs with the fixed constants
            0.15 and 0.075 (TA-Lib's MACDFIX). Needs the exponential moving average and
            windows of 12 and 26. Defaults to False.
        volumes (pd.Series | None): Series of trading volumes, needed by the volume weighted
            moving averages. Defaults to None.

    Returns:
        pd.DataFrame: The MACD Line, Signal Line and Histogram as columns.

    Raises:
        InvalidArgumentError: If a moving average is not one of `MOVING_AVERAGES` or
            `VOLUME_MOVING_AVERAGES` (a ValueError).
        ValueError: If a volume weighted moving average has no `volumes`, or `fixed` is
            combined with another moving average or other windows than 12 and 26.
    """
    for parameter, name in (
        ("moving_average", moving_average),
        ("signal_moving_average", signal_moving_average),
    ):
        check_option(parameter, name, [*MOVING_AVERAGES, *VOLUME_MOVING_AVERAGES])
        if name in VOLUME_MOVING_AVERAGES and volumes is None:
            raise ValueError(f"The {name} moving average needs the volumes.")

    def moving_average_of(values: pd.Series, name: str, window: int) -> pd.Series:
        if name in VOLUME_MOVING_AVERAGES:
            return VOLUME_MOVING_AVERAGES[name](values, volumes, window)
        return MOVING_AVERAGES[name](values, window)

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
        short_average = moving_average_of(prices, moving_average, short_window)
        long_average = moving_average_of(prices, moving_average, long_window)

    macd_line = short_average - long_average
    signal_line = moving_average_of(macd_line, signal_moving_average, signal_window)

    return pd.concat(
        [macd_line, signal_line, macd_line - signal_line],
        keys=["MACD Line", "Signal Line", "Histogram"],
        axis=1,
    )


def get_relative_strength_index(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI): the size of the gains against the losses over a window, from 0 to
    100, to show momentum and overbought and oversold levels.

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

    Referenced against:
    - TA-Lib, RSI: https://ta-lib.org/functions/rsi.html

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
    Calculate the Stochastic RSI: where the Relative Strength Index lies within its own range over a window, a
    faster measure of overbought and oversold levels.

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
    and TradingView do (a window and a stochastic window of 14). StockCharts shows this %K on a
    scale of 0 to 1; TradingView smooths it over 3 periods, so its %K equals this %D with
    a smoothing window of 3. TA-Lib's `STOCHRSI` defaults to a 5-period stochastic instead
    (timeperiod 14, fastk_period 5, fastd_period 3 with a simple average), which is
    a stochastic window of 5 here and returns its fast %K and %D.

    Also known as: StochRSI, Stoch RSI.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    Referenced against:
    - TA-Lib, STOCHRSI: https://ta-lib.org/functions/stochrsi.html

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
    Calculate the True Strength Index (TSI): the double-smoothed price change as a share of the double-smoothed
    absolute price change, which shows the direction and strength of momentum.

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

    Referenced against:
    - TA-Lib, TSI: https://ta-lib.org/functions/tsi.html

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
    Calculate the Balance of Power (BOP): how far the close lies from the open as a share of the range, which shows
    whether buyers or sellers control the period.

    The Balance of Power (BOP) indicator measures the strength of buyers and sellers in the market.
    It considers the relationship between the closing price and the trading range (high - low) of each period.
    BOP values above zero suggest bullish buying pressure, while values below zero suggest bearish selling pressure.

    The formula is as follows:

    - BOP = (Close - Open) / (High - Low)

    Also known as: BOP.

    Reference: Created by Igor Livshin; featured in Kaufman, P.J. (2005). "New Trading
    Systems and Methods." 4th ed. Wiley.

    Referenced against:
    - TA-Lib, BOP: https://ta-lib.org/functions/bop.html

    Notes:
        - A period whose high equals its low has no range to divide by and is set to 0, as
          in TA-Lib's BOP. A period with a missing price stays missing.

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
    Calculate the Awesome Oscillator (AO): the difference between a fast and a slow average of the median price,
    which shows the momentum of the market.

    The Awesome Oscillator measures market momentum by comparing a short-term and a long-term
    Simple Moving Average of the median price (the midpoint of each period's high and low,
    rather than the closing price). It was developed by Bill Williams as part of his broader
    "Trading Chaos" collection of momentum indicators.

    The formula is as follows:

    - Median Price = (High + Low) / 2
    - AO = SMA(Median Price, short_window) - SMA(Median Price, long_window)

    Also known as: AO, Bill Williams Awesome Oscillator.

    Referenced against:
    - TA-Lib, AO: https://ta-lib.org/functions/ao.html

    Notes:
        - There is no academic journal citation for the Awesome Oscillator. Like most of Bill
          Williams' indicators, it is a practitioner-developed tool rather than one derived from
          a published financial paper. The standard textbook source is Williams, B. (1995).
          "Trading Chaos: Applying Expert Techniques to Maximize Your Profits." Wiley.
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
    Calculate the Vortex Indicator: the upward and downward movement of the prices relative to their true range,
    whose crossings mark the start of a new trend.

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
    Stocks & Commodities, 28(1), 20-30.

    Referenced against:
    - TA-Lib, VORTEX: https://ta-lib.org/functions/vortex.html

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
    Calculate the Elder Ray Index: how far the high (Bull Power) and the low (Bear Power) lie from an exponential
    moving average, which shows the strength of the buyers and the sellers.

    The Elder Ray Index measures buying and selling pressure in the market relative to a trend
    baseline (an Exponential Moving Average of the closing price). Bull Power captures how far
    the high extends above the EMA (buying pressure), while Bear Power captures how far the low
    extends below the EMA (selling pressure).

    The formula is as follows:

    - Bull Power = High - EMA(Close, window)
    - Bear Power = Low - EMA(Close, window)

    Also known as: Elder Ray, Bull Power, Bear Power.

    Reference: Elder, A. (1993). "Trading for a Living." Wiley.

    Referenced against:
    - TA-Lib, ERI: https://ta-lib.org/functions/eri.html

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
    Calculate the Rate of Change (ROC): the percentage change of the price over a window, the speed at which the
    price moves.

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

    Referenced against:
    - TA-Lib, ROCP: https://ta-lib.org/functions/rocp.html

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
    Calculate the Choppiness Index (CHOP): how much of the range of a window the price travelled, from trending
    (near 0) to choppy and sideways (near 100).

    The Choppiness Index quantifies whether the market is trending or moving sideways
    ("choppy") by comparing the sum of True Range over the window (a measure of the total
    price path travelled) to the net range the price actually covered over that same window
    (the distance between the highest high and the lowest low). When price travels a long,
    winding path but ends up covering little net ground, the index is high (near 100),
    signalling a choppy, range-bound market. When price travels efficiently in one direction,
    the index is low (near 0), signalling a trending market.

    The formula is as follows:

    - CHOP = 100 * log10( Sum(True Range, window) / (Max(High, window) - Min(Low, window)) ) / log10(window)

    With the true range as the box, the net range is that of the true highs and lows, the
    range the True Ranges themselves span, so a gap between periods counts in both:

    - True High = Max(High, Previous Close), True Low = Min(Low, Previous Close)
    - CHOP = 100 * log10( Sum(True Range, window) / (Max(True High, window) - Min(True Low, window)) ) / log10(window)

    Also known as: CHOP, CHOPTR (the true range box), Choppiness Index.

    Reference: Dreiss, E.W. (1990s). Developed by Australian commodities trader Bill Dreiss;
    there is no formal journal citation. The standard textbook treatment is Kaufman, P.J.
    (2013). "Trading Systems and Methods." 5th ed. Wiley.

    Referenced against:
    - TA-Lib, CHOP: https://ta-lib.org/functions/chop.html
    - TA-Lib, CHOPTR: https://ta-lib.org/functions/choptr.html

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to consider for the Choppiness Index calculation.
        range_method (str): The net range the summed True Range is divided by. Options are: "high_low" (the
            highest high minus the lowest low), "true_range" (the highest true high minus the lowest true low,
            TA-Lib's CHOPTR). Defaults to "high_low".

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
    check_option("range_method", range_method, ("high_low", "true_range"))

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
    Calculate the Know Sure Thing (KST): a weighted sum of four smoothed rates of change over increasing windows,
    which shows the momentum of the major price cycles.

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

    Reference: Pring, M.J. (1992). "Summed Rate of Change (KST)." Technical Analysis of
    Stocks & Commodities, 10(9), 365-369.

    Referenced against:
    - TA-Lib, KST: https://ta-lib.org/functions/kst.html
    - TA-Lib, KSTEXT: https://ta-lib.org/functions/kstext.html

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
        moving_average (str): The moving average that smooths each Rate of Change, one of `MOVING_AVERAGES`.
            Options are: "simple" (Simple Moving Average), "exponential" (Exponential Moving Average),
            "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular Moving Average),
            "weighted" (Weighted Moving Average), "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman
            Adaptive Moving Average). Defaults to "simple".
        signal_moving_average (str): The moving average of the signal line, one of `MOVING_AVERAGES`. Options
            are: "simple" (Simple Moving Average), "exponential" (Exponential Moving Average),
            "double_exponential" (Double Exponential Moving Average), "triangular" (Triangular Moving Average),
            "weighted" (Weighted Moving Average), "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman
            Adaptive Moving Average). Defaults to "simple".

    Returns:
        pd.DataFrame: Know Sure Thing (KST and Signal Line) values, weighted sums of
            decimal Rates of Change.

    Raises:
        TypeError: If `prices_close` is not a pandas Series.
        InvalidArgumentError: If a moving average is not one of `MOVING_AVERAGES` (a
            ValueError).
        ValueError: If `roc_windows`, `sma_windows` and `weights` are not all length 4.
    """
    if not isinstance(prices_close, pd.Series):
        raise TypeError("prices_close must be a pandas Series.")
    check_option("moving_average", moving_average, list(MOVING_AVERAGES))
    check_option("signal_moving_average", signal_moving_average, list(MOVING_AVERAGES))

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
    Calculate the Coppock Curve: a weighted moving average of the sum of a long and a short rate of change, which
    shows long-term buying opportunities after a decline.

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

    Referenced against:
    - TA-Lib, COPPOCK: https://ta-lib.org/functions/coppock.html

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
    Calculate the Absolute Price Oscillator (APO): the difference between a fast and a slow moving average of the
    prices, which shows the direction and momentum of a trend.

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

    Referenced against:
    - TA-Lib, APO: https://ta-lib.org/functions/apo.html

    Notes:
        - Each moving average starts on its own first full window, so with EMAs seeded with a
          Simple Moving Average the APO starts on period `long_window` - 1 and equals the
          MACD line of `get_moving_average_convergence_divergence`.
        - TA-Lib's APO takes the moving average as its moving average type, the exponential moving average by
          default.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        moving_average (str): The moving average, one of `MOVING_AVERAGES`. Options are: "simple" (Simple Moving
            Average), "exponential" (Exponential Moving Average), "double_exponential" (Double Exponential
            Moving Average), "triangular" (Triangular Moving Average), "weighted" (Weighted Moving Average),
            "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman Adaptive Moving Average). Defaults to
            "exponential".

    Returns:
        pd.Series: APO values, in price units.

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    check_option("moving_average", moving_average, MOVING_AVERAGES)

    return MOVING_AVERAGES[moving_average](
        prices_close, short_window
    ) - MOVING_AVERAGES[moving_average](prices_close, long_window)


def get_momentum(prices_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Momentum (MOM): the change of the price over a window, the speed at which the price moves in price
    units.

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

    Referenced against:
    - TA-Lib, MOM: https://ta-lib.org/functions/mom.html

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
    Calculate the Accelerator/Decelerator Oscillator (AC): the Awesome Oscillator minus its own average, which shows
    whether momentum is speeding up or slowing down.

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

    Referenced against:
    - TA-Lib, AC: https://ta-lib.org/functions/ac.html

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
    Calculate the Internal Bar Strength (IBS): where the close lies within the range of its period, from 0 (at the
    low) to 1 (at the high).

    The Internal Bar Strength is where the close sits within the period's own range: 0 for
    a close on the low, 1 for a close on the high. It is a mean-reversion measure: a close
    near the low of the day (a low IBS) has tended to be followed by higher returns in
    equity indices and ETFs, and a close near the high by lower returns.

    The formula is as follows:

    - IBS = (Close - Low) / (High - Low)

    Also known as: IBS, close location value (as a 0 to 1 ratio).

    Reference: Pagonidis, A.S. (2014). "The IBS Effect: Mean Reversion in Equity ETFs."
    NAAIM Founders Award paper.

    Referenced against:
    - TA-Lib, IBS: https://ta-lib.org/functions/ibs.html

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
    Calculate the Intraday Momentum Index (IMI): the gains of the closes over the opens against their losses, from 0
    to 100, a Relative Strength Index of the candles.

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

    Referenced against:
    - TA-Lib, IMI: https://ta-lib.org/functions/imi.html

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
    Calculate the Qstick: the average of the close minus the open over a window, which shows whether candles have
    been mostly rising or falling.

    Qstick is the moving average of the candlestick bodies, Close - Open, over the window.
    It turns the colour of the recent candles into a number: above zero the periods closed
    above their open on average (white candles, buying pressure), below zero below their
    open (black candles, selling pressure).

    The formula is as follows:

    - Qstick = SMA(Close - Open, window)

    Also known as: Qstick, QSTICK, candlestick body average.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    Referenced against:
    - TA-Lib, QSTICK: https://ta-lib.org/functions/qstick.html

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
    Calculate Kaufman's Efficiency Ratio (ER): the net change over a window as a share of the path travelled, from 0
    (noise) to 1 (a straight trend).

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

    Referenced against:
    - TA-Lib, ER: https://ta-lib.org/functions/er.html

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
    Calculate the Vertical Horizontal Filter (VHF): the range of a window as a share of the path travelled, which
    shows whether prices trend or move sideways.

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

    Referenced against:
    - TA-Lib, VHF: https://ta-lib.org/functions/vhf.html

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


def get_fisher_transform(
    prices_high: pd.Series, prices_low: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate the Fisher Transform: the prices' position within their range turned into a near-normal distribution,
    which makes the turning points sharper and easier to see.

    The Fisher Transform turns the position of the median price within its recent range into
    a series that is close to normally distributed. Prices are not normally distributed, so
    an oscillator of their position spends most of its time in the middle and gives vague
    turning points; the Fisher transform stretches the values near the edges of the range,
    which makes the extremes, and the turns after them, sharp and rare. The trigger is the
    previous value of the transform, and the transform crossing it marks a turn.

    The formula is as follows:

    - Median Price = (High + Low) / 2
    - Position = (Median Price - Min(Median Price, window)) / (Max(Median Price, window) -
      Min(Median Price, window))
    - Value(t) = 0.66 * (Position - 0.5) + 0.67 * Value(t - 1), set to 0.999 above 0.99 and
      to -0.999 below -0.99
    - Fisher Transform(t) = 0.5 * ln((1 + Value(t)) / (1 - Value(t))) + 0.5 * Fisher
      Transform(t - 1)
    - Trigger(t) = Fisher Transform(t - 1)

    Value and Fisher Transform start from 0 on the period before the first full window, as
    in Ehlers' published formula.

    Also known as: Fisher, Ehlers Fisher Transform, FISHT.

    Reference: Ehlers, J.F. (2002). "Using the Fisher Transform." Technical Analysis of
    Stocks & Commodities, 20(11); Ehlers, J.F. (2004). "Cybernetic Analysis for Stocks and
    Futures." Wiley.

    Referenced against:
    - TA-Lib, FISHER: https://ta-lib.org/functions/fisher.html

    Shares similar logic:
    - pandas-ta-classic, fisher: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/fisher.py#L36
    - finta, FISH: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1827

    Notes:
        - A window in which the median price did not move has no range; its position is the
          neutral 0.5, as in TA-Lib, so the transform decays towards zero rather than
          stopping.
        - The Fisher Transform starts on period `window` - 1 and the trigger one period
          later; TA-Lib gives the first trigger the starting value 0. pandas-ta-classic's
          fisher starts one period later and therefore differs in the first values only.
          finta's FISH smooths with exponential averages of 5 and 3 periods instead of
          Ehlers' constants, a different series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods of the range of the median price.

    Returns:
        pd.DataFrame: The Fisher Transform and Trigger as columns, centred on zero.
    """
    median_price = (prices_high + prices_low) / 2
    highest = median_price.rolling(window=window).max()
    lowest = median_price.rolling(window=window).min()
    price_range = highest - lowest

    position = ((median_price - lowest) / price_range.where(price_range != 0)).where(
        price_range != 0, 0.5
    )
    position = position.where(price_range.notna())

    # The recursion runs on plain Python floats and carries its state across a period
    # without a price, which has no value itself.
    positions = position.to_numpy(dtype="float64").tolist()
    fisher = [math.nan] * len(positions)
    value = 0.0
    previous_fisher = 0.0
    for i, current_position in enumerate(positions):
        if math.isnan(current_position):
            continue

        value = 0.66 * (current_position - 0.5) + 0.67 * value
        if value > 0.99:  # noqa: PLR2004
            value = 0.999
        elif value < -0.99:  # noqa: PLR2004
            value = -0.999

        previous_fisher = (
            0.5 * math.log((1 + value) / (1 - value)) + 0.5 * previous_fisher
        )
        fisher[i] = previous_fisher

    fisher_transform = pd.Series(fisher, index=prices_high.index, dtype="float64")
    trigger = fisher_transform.shift(1)

    return pd.concat(
        [fisher_transform, trigger], keys=["Fisher Transform", "Trigger"], axis=1
    )


def get_inverse_fisher_transform_rsi(
    prices: pd.Series, rsi_window: int, wma_window: int
) -> pd.Series:
    """
    Calculate the Inverse Fisher Transform of the RSI: the smoothed Relative Strength Index squeezed between -1 and
    1, which makes its overbought and oversold signals clearer.

    The inverse Fisher transform (the hyperbolic tangent) squeezes a series into the range
    from -1 to 1, with most values close to the edges: applied to a rescaled and smoothed
    short RSI, it turns the RSI into a nearly binary signal that switches quickly between
    overbought and oversold. Ehlers buys when it crosses above -0.5 and sells when it
    crosses below 0.5.

    The formula is as follows:

    - Value 1 = 0.1 * (RSI(Close, rsi_window) - 50)
    - Value 2 = WMA(Value 1, wma_window)
    - IFT RSI = (exp(2 * Value 2) - 1) / (exp(2 * Value 2) + 1) = tanh(Value 2)

    Ehlers uses a 5-period RSI and a 9-period weighted moving average.

    Also known as: IFT RSI, IFTRSI, Inverse Fisher RSI.

    Reference: Ehlers, J.F. (2004). "The Inverse Fisher Transform." Technical Analysis of
    Stocks & Commodities, 22(5).

    Shares similar logic:
    - finta, IFT_RSI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L714

    Notes:
        - The RSI is that of `get_relative_strength_index` (Wilder's smoothing, seeded with
          a simple average), so the IFT RSI starts on period `rsi_window` + `wma_window` - 1.
        - finta's IFT_RSI returns (Value 2² - 1) / (Value 2² + 1) instead of the hyperbolic
          tangent, with an RSI from an exponential average without a seed; this follows
          Ehlers' formula.

    Args:
        prices (pd.Series): Series of prices.
        rsi_window (int): Number of periods of the RSI.
        wma_window (int): Number of periods of the weighted moving average of the rescaled
            RSI.

    Returns:
        pd.Series: IFT RSI values, between -1 and 1.
    """
    relative_strength_index = get_relative_strength_index(prices, rsi_window)

    smoothed_value = get_weighted_moving_average(
        0.1 * (relative_strength_index - 50), wma_window
    )

    return np.tanh(smoothed_value)


def get_kdj_indicator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    k_window: int,
    d_window: int,
) -> pd.DataFrame:
    """
    Calculate the KDJ indicator: the Stochastic Oscillator's %K and %D with the J line that leads them, to show
    overbought and oversold levels and turning points.

    The KDJ indicator is a stochastic oscillator, widely used in Chinese and other Asian
    markets, that smooths the raw stochastic with Wilder's average instead of a simple one
    and adds a third line, J, which extrapolates K away from D. J leaves the 0 to 100 range
    when the momentum is strong, so J above 100 (or below 0) is read as overbought (or
    oversold) before K and D get there, and K crossing D is the trade signal, as with the
    stochastic.

    The formula is as follows:

    - RSV = 100 * (Close - Lowest Low(window)) / (Highest High(window) - Lowest Low(window))
    - K = Wilder's MA(RSV, k_window)
    - D = Wilder's MA(K, d_window)
    - J = 3 * K - 2 * D

    Wilder's moving average is the recursion K(t) = K(t - 1) + (RSV(t) - K(t - 1)) /
    k_window, often written SMA(RSV, 3, 1) in the formula language of Asian platforms. The
    common parameters are 9, 3 and 3.

    Also known as: KDJ, KDJ stochastic, random index.

    Reference: The KDJ has no single primary source; it extends Lane's stochastic, see Lane,
    G.C. (1984). "Lane's Stochastics." Technical Analysis of Stocks & Commodities, 2(3).

    Referenced against:
    - TA-Lib, KDJ: https://ta-lib.org/functions/kdj.html

    Shares similar logic:
    - pandas-ta-classic, kdj: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/kdj.py#L17

    Notes:
        - Every Wilder average is seeded with the simple average of its first values (see
          `get_wilder_moving_average`), as TA-Lib does with its RMA moving average type, so
          K starts on period `window` + `k_window` - 2 and D `d_window` - 1 periods later.
          Platforms that start K and D at 50 differ in the first few dozen periods.
        - A window without a range (the highest high equal to the lowest low) has no RSV
          (NaN), where TA-Lib uses 0; the averages carry on over it.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the high-low range of the RSV.
        k_window (int): Number of periods of Wilder's average of the RSV that gives K.
        d_window (int): Number of periods of Wilder's average of K that gives D.

    Returns:
        pd.DataFrame: K, D and J as columns; K and D between 0 and 100, J unbounded.
    """
    raw_stochastic = get_stochastic_oscillator(
        prices_high, prices_low, prices_close, window, 1
    )["Stochastic %K"]

    k_line = get_wilder_moving_average(raw_stochastic, k_window)
    d_line = get_wilder_moving_average(k_line, d_window)

    return pd.concat(
        [k_line, d_line, 3 * k_line - 2 * d_line], keys=["K", "D", "J"], axis=1
    )


def get_premier_stochastic_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    smooth_window: int,
) -> pd.Series:
    """
    Calculate the Premier Stochastic Oscillator (PSO): the smoothed and normalised Stochastic %K squeezed between -1
    and 1, which gives clearer overbought and oversold signals.

    The Premier Stochastic Oscillator rescales a short stochastic around zero, smooths it
    twice with an exponential average and squeezes the result between -1 and 1 with the
    inverse Fisher transform. The smoothing removes the noise of the stochastic, and the
    transform keeps the oscillator near its extremes most of the time, so that a reading
    crossing back through 0.9 or -0.9 (or 0.2 and -0.2) stands out as a signal.

    The formula is as follows:

    - %K = 100 * (Close - Lowest Low(window)) / (Highest High(window) - Lowest Low(window))
    - Normalized Stochastic = 0.1 * (%K - 50)
    - Smoothed Stochastic = EMA(EMA(Normalized Stochastic, smooth_window), smooth_window)
    - PSO = (exp(Smoothed Stochastic) - 1) / (exp(Smoothed Stochastic) + 1) = tanh(Smoothed
      Stochastic / 2)

    Leibfarth uses an 8-period stochastic and a smoothing of 5 periods (the square root of
    25).

    Also known as: PSO, Premier Stochastic.

    Reference: Leibfarth, L. (2008). "Premier Stochastic Oscillator." Technical Analysis of
    Stocks & Commodities, 26(8).

    Referenced against:
    - TA-Lib, PSO: https://ta-lib.org/functions/pso.html

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values with an
          empty warm-up (see `get_exponential_moving_average`), as TA-Lib does, so the PSO
          starts on period `window` + 2 * `smooth_window` - 3.
        - A window without a range has no %K (NaN), where TA-Lib uses 50; the averages
          carry on over it.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the high-low range of %K.
        smooth_window (int): Number of periods of each of the two EMAs.

    Returns:
        pd.Series: PSO values, between -1 and 1.
    """
    stochastic_k = get_stochastic_oscillator(
        prices_high, prices_low, prices_close, window, 1
    )["Stochastic %K"]

    smoothed_stochastic = get_exponential_moving_average(
        get_exponential_moving_average(0.1 * (stochastic_k - 50), smooth_window),
        smooth_window,
    )

    return np.tanh(smoothed_stochastic / 2)


def get_stochastic_momentum_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    long_window: int,
    short_window: int,
    signal_window: int,
) -> pd.DataFrame:
    """
    Calculate the Stochastic Momentum Index (SMI): where the close lies relative to the middle of the high-low range
    of a window, smoothed, from -100 to 100.

    The Stochastic Momentum Index measures where the close sits relative to the midpoint of
    the high-low range, rather than to its low as the stochastic does, and smooths both the
    distance and the range twice with an exponential average before dividing. It runs from
    -100 (a close on the low) to +100 (a close on the high) around zero, is less erratic than
    the stochastic, and is read with its signal line and the levels -40 and 40.

    The formula is as follows:

    - Distance = Close - (Highest High(window) + Lowest Low(window)) / 2
    - Range = Highest High(window) - Lowest Low(window)
    - SMI = 100 * EMA(EMA(Distance, long_window), short_window) / (0.5 * EMA(EMA(Range,
      long_window), short_window))
    - Signal Line = EMA(SMI, signal_window)

    Blau uses a 13-period range, smoothed over 25 and then 2 periods.

    Also known as: SMI, Blau's stochastic momentum.

    Reference: Blau, W. (1993). "Stochastic Momentum." Technical Analysis of Stocks &
    Commodities, 11(1); Blau, W. (1995). "Momentum, Direction, and Divergence." Wiley.

    Referenced against:
    - TA-Lib, SMI: https://ta-lib.org/functions/smi.html

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values with an
          empty warm-up (see `get_exponential_moving_average`), as TA-Lib does, so the SMI
          starts on period `window` + `long_window` + `short_window` - 3 and the signal
          line `signal_window` - 1 periods later. TA-Lib starts both on the first signal
          value.
        - A stretch in which the smoothed range is zero has no SMI (NaN), where TA-Lib
          returns 0.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the high-low range.
        long_window (int): Number of periods of the first EMA of the distance and range.
        short_window (int): Number of periods of the second EMA.
        signal_window (int): Number of periods of the EMA of the SMI that gives the signal
            line.

    Returns:
        pd.DataFrame: Stochastic Momentum Index and Signal Line as columns, between -100 and
            100.
    """
    highest_high = prices_high.rolling(window=window).max()
    lowest_low = prices_low.rolling(window=window).min()

    smoothed_distance = get_exponential_moving_average(
        get_exponential_moving_average(
            prices_close - (highest_high + lowest_low) / 2, long_window
        ),
        short_window,
    )
    smoothed_range = get_exponential_moving_average(
        get_exponential_moving_average(highest_high - lowest_low, long_window),
        short_window,
    )

    stochastic_momentum_index = (
        100 * smoothed_distance / (0.5 * smoothed_range.where(smoothed_range != 0))
    )
    signal_line = get_exponential_moving_average(
        stochastic_momentum_index, signal_window
    )

    return pd.concat(
        [stochastic_momentum_index, signal_line],
        keys=["Stochastic Momentum Index", "Signal Line"],
        axis=1,
    )


def get_smi_ergodic_indicator(
    prices: pd.Series, long_window: int, short_window: int, signal_window: int
) -> pd.DataFrame:
    """
    Calculate the SMI Ergodic Indicator and Oscillator: the True Strength Index with its signal line and their
    difference, which show the direction and strength of momentum.

    The SMI Ergodic Indicator is Blau's True Strength Index with short smoothing periods,
    and its signal line an EMA of the indicator. The SMI Ergodic Oscillator is the
    difference between the two, a histogram in the way the MACD histogram is the MACD line
    minus its signal line. Blau trades the direction of the oscillator: a turn up from below
    zero is a buy signal, a turn down from above zero a sell signal.

    The formula is as follows:

    - Momentum(t) = Close(t) - Close(t - 1)
    - SMI Ergodic Indicator = 100 * EMA(EMA(Momentum, long_window), short_window) /
      EMA(EMA(|Momentum|, long_window), short_window)
    - Signal Line = EMA(SMI Ergodic Indicator, signal_window)
    - SMI Ergodic Oscillator (Histogram) = SMI Ergodic Indicator - Signal Line

    The common parameters are 20, 5 and 5.

    Also known as: SMI Ergodic, SMIIO, ergodic oscillator, TSI ergodic.

    Reference: Blau, W. (1995). "Momentum, Direction, and Divergence." Wiley.

    Referenced against:
    - pandas-ta-classic, smi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/smi.py#L13

    Notes:
        - The indicator is `get_true_strength_index` with these windows and on its scale of
          -100 to 100; pandas-ta-classic's smi uses a scalar of 1, a scale of -1 to 1.
        - Every EMA is seeded with the Simple Moving Average of its first values with an
          empty warm-up (see `get_exponential_moving_average`), so the indicator starts on
          period `long_window` + `short_window` - 1 and the signal line and histogram
          `signal_window` - 1 periods later.

    Args:
        prices (pd.Series): Series of prices.
        long_window (int): Number of periods of the first EMA of the price change.
        short_window (int): Number of periods of the second EMA.
        signal_window (int): Number of periods of the EMA of the indicator that gives the
            signal line.

    Returns:
        pd.DataFrame: SMI Ergodic Indicator, Signal Line and Histogram (the SMI Ergodic
            Oscillator) as columns.
    """
    true_strength_index = get_true_strength_index(
        prices, long_window, short_window, signal_window
    )
    ergodic_indicator = true_strength_index["True Strength Index"]
    signal_line = true_strength_index["Signal Line"]

    return pd.concat(
        [ergodic_indicator, signal_line, ergodic_indicator - signal_line],
        keys=["SMI Ergodic Indicator", "Signal Line", "Histogram"],
        axis=1,
    )


def get_schaff_trend_cycle(
    prices: pd.Series,
    short_window: int,
    long_window: int,
    cycle_window: int,
    factor: float = 0.5,
) -> pd.Series:
    """
    Calculate the Schaff Trend Cycle (STC): a stochastic of the MACD, applied twice, from 0 to 100, which shows the
    turns of a trend faster than the MACD.

    The Schaff Trend Cycle applies a smoothed stochastic twice to the MACD line: it measures
    where the MACD sits within its range over the last cycle, smooths that, and does the
    same again. The result moves between 0 and 100 and turns earlier than the MACD, as it
    reacts to the MACD's cycle rather than its level. A turn up from below 25 is read as
    bullish and a turn down from above 75 as bearish.

    The formula is as follows:

    - MACD = EMA(Close, short_window) - EMA(Close, long_window)
    - Fraction 1 = 100 * (MACD - Min(MACD, cycle_window)) / (Max(MACD, cycle_window) -
      Min(MACD, cycle_window))
    - PF(t) = PF(t - 1) + factor * (Fraction 1(t) - PF(t - 1))
    - Fraction 2 = 100 * (PF - Min(PF, cycle_window)) / (Max(PF, cycle_window) - Min(PF,
      cycle_window))
    - STC(t) = STC(t - 1) + factor * (Fraction 2(t) - STC(t - 1))

    A fraction over a window without a range keeps its previous value, and each smoothing
    starts on its first input, as in Schaff (2008). Schaff uses 23, 50 and 10 with a factor
    of 0.5.

    Also known as: STC, Schaff Trend Cycle indicator.

    Reference: Schaff, D. (2008). "Releasing the Code to the Schaff Trend Cycle."
    FX-Strategy.com; Schaff, D. (2002). "Catching Currency Moves with the Schaff Trend Cycle
    Indicator." Chartpoint, July/August 2002.

    Referenced against:
    - TA-Lib, STC: https://ta-lib.org/functions/stc.html

    Shares similar logic:
    - pandas-ta-classic, stc: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/stc.py#L100
    - ta, STCIndicator: https://github.com/bukosabino/ta/blob/0.11.0/ta/trend.py#L1093
    - finta, STC: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L2121

    Notes:
        - The EMAs of the MACD are seeded with the Simple Moving Average of their first
          values with an empty warm-up (see `get_exponential_moving_average`). TA-Lib starts
          the short EMA on the same period as the long one, so the two differ in the first
          values (up to 50 points) and agree within 0.01 after about 200 periods and within
          1e-5 after about 250 at the default windows.
        - pandas-ta-classic's stc seeds as this does and equals it with the same windows
          after the first values.
        - A fraction has a value from its first full window with a range, so the STC starts
          on period `long_window` + 2 * `cycle_window` - 3 at the earliest; TA-Lib counts a
          fraction without a value yet as 0.
        - pandas-ta-classic's stc defaults to 12 and 26 periods, and ta's STCIndicator and
          finta's STC smooth with exponential or simple moving averages of 3 periods.

    Args:
        prices (pd.Series): Series of prices.
        short_window (int): Number of periods of the short EMA of the MACD.
        long_window (int): Number of periods of the long EMA of the MACD.
        cycle_window (int): Number of periods of the range of each stochastic.
        factor (float): The smoothing factor of each stochastic. Defaults to 0.5.

    Returns:
        pd.Series: STC values, between 0 and 100.
    """
    macd_line = get_exponential_moving_average(
        prices, short_window
    ) - get_exponential_moving_average(prices, long_window)

    # The two stages are the same: a stochastic of the input over the cycle window that
    # keeps its previous value on a window without a range, smoothed with the factor from
    # its first value. A period without an input has no value and the state carries on.
    smoothed = macd_line
    for _ in range(2):
        values = smoothed.to_numpy(dtype="float64").tolist()
        lowest = smoothed.rolling(window=cycle_window).min().to_numpy().tolist()
        highest = smoothed.rolling(window=cycle_window).max().to_numpy().tolist()

        output = [math.nan] * len(values)
        fraction = math.nan
        previous = math.nan
        for i, value in enumerate(values):
            if math.isnan(value) or math.isnan(lowest[i]):
                continue

            if highest[i] > lowest[i]:
                fraction = 100 * (value - lowest[i]) / (highest[i] - lowest[i])
            if math.isnan(fraction):
                continue

            previous = (
                fraction
                if math.isnan(previous)
                else previous + factor * (fraction - previous)
            )
            output[i] = previous

        smoothed = pd.Series(output, index=prices.index, dtype="float64")

    return smoothed


def get_wavetrend_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    channel_window: int,
    average_window: int,
    signal_window: int,
) -> pd.DataFrame:
    """
    Calculate the WaveTrend Oscillator: how far the typical price lies from its average in units of its average
    distance, smoothed, to show overbought and oversold levels and turning points.

    The WaveTrend Oscillator is a Commodity Channel Index built from exponential averages:
    the distance of the typical price from its EMA, divided by 0.015 times the EMA of that
    absolute distance, and smoothed with a longer EMA. Its signal line is a short simple
    average. Readings beyond 53 or 60 (or -53 and -60) are read as overbought (or
    oversold), and the oscillator crossing its signal line there as a turn.

    The formula is as follows:

    - Typical Price = (High + Low + Close) / 3
    - ESA = EMA(Typical Price, channel_window)
    - D = EMA(|Typical Price - ESA|, channel_window)
    - CI = (Typical Price - ESA) / (0.015 * D)
    - WaveTrend = EMA(CI, average_window)
    - Signal Line = SMA(WaveTrend, signal_window)

    LazyBear's parameters are 10, 21 and 4.

    Also known as: WaveTrend, WT, WTO, WT1 and WT2.

    Reference: LazyBear (2014). "WaveTrend Oscillator [WT]." TradingView published script.

    Referenced against:
    - TA-Lib, WAVETREND: https://ta-lib.org/functions/wavetrend.html

    Shares similar logic:
    - finta, WTO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1802

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values with an
          empty warm-up (see `get_exponential_moving_average`), as TA-Lib does, so the
          WaveTrend starts on period 2 * `channel_window` + `average_window` - 3 and the
          signal line `signal_window` - 1 periods later. finta's WTO uses averages without a
          seed, which differ in the first values.
        - A stretch without any distance from the average has no CI (NaN), where TA-Lib
          returns 0.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        channel_window (int): Number of periods of the EMA of the typical price and of its
            distance.
        average_window (int): Number of periods of the EMA of the CI.
        signal_window (int): Number of periods of the SMA that gives the signal line.

    Returns:
        pd.DataFrame: WaveTrend and Signal Line as columns.
    """
    typical_price = (prices_high + prices_low + prices_close) / 3
    average = get_exponential_moving_average(typical_price, channel_window)
    distance = typical_price - average

    mean_distance = get_exponential_moving_average(distance.abs(), channel_window)
    channel_index = distance / (0.015 * mean_distance.where(mean_distance != 0))

    wavetrend = get_exponential_moving_average(channel_index, average_window)
    signal_line = get_moving_average(wavetrend, signal_window)

    return pd.concat(
        [wavetrend, signal_line], keys=["WaveTrend", "Signal Line"], axis=1
    )


def get_derivative_oscillator(
    prices: pd.Series,
    window: int,
    first_window: int,
    second_window: int,
    signal_window: int,
) -> pd.Series:
    """
    Calculate the Derivative Oscillator: a double-smoothed Relative Strength Index minus its average, which shows
    the momentum of the RSI itself.

    The Derivative Oscillator smooths the Relative Strength Index twice with an exponential
    average and subtracts a simple average of the result, the way the MACD histogram
    subtracts the signal line from the MACD. It thereby shows the momentum of the smoothed
    RSI: a crossing of zero marks a turn of the RSI, and the distance from zero how far the
    smoothed RSI has moved from its mean.

    The formula is as follows:

    - Double Smoothed RSI = EMA(EMA(RSI(Close, window), first_window), second_window)
    - Derivative Oscillator = Double Smoothed RSI - SMA(Double Smoothed RSI, signal_window)

    Brown uses a 14-period RSI, smoothed over 5 and then 3 periods, and a 9-period signal.

    Also known as: DOSC, Brown's Derivative Oscillator.

    Reference: Brown, C.M. (1994). "The Derivative Oscillator: A New Approach for an Old
    Problem." MTA Journal, 42; Brown, C.M. (1999). "Technical Analysis for the Trading
    Professional." McGraw-Hill.

    Referenced against:
    - TA-Lib, DOSC: https://ta-lib.org/functions/dosc.html

    Notes:
        - The RSI is that of `get_relative_strength_index` and every EMA is seeded with the
          Simple Moving Average of its first values, as in TA-Lib, so the oscillator starts
          on period `window` + `first_window` + `second_window` + `signal_window` - 3.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the RSI.
        first_window (int): Number of periods of the first EMA of the RSI.
        second_window (int): Number of periods of the second EMA.
        signal_window (int): Number of periods of the SMA subtracted from the smoothed RSI.

    Returns:
        pd.Series: Derivative Oscillator values, in RSI points around zero.
    """
    double_smoothed_rsi = get_exponential_moving_average(
        get_exponential_moving_average(
            get_relative_strength_index(prices, window), first_window
        ),
        second_window,
    )

    return double_smoothed_rsi - get_moving_average(double_smoothed_rsi, signal_window)


def get_swing_index(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    limit_move: float,
) -> pd.DataFrame:
    """
    Calculate Wilder's Swing Index and Accumulative Swing Index: the strength of each period's move compared with
    the previous period, and its running total, which shows the real direction of the prices.

    The Swing Index weighs the change of the close against the previous period's open and
    close and against the range of the current period, to measure the real direction of a
    period's move on a scale of -100 to 100. Wilder accumulates it into the Accumulative
    Swing Index, a line that follows the price but filters out the moves that the open,
    high and low do not confirm. A breakout of the ASI through a previous high or low swing
    point confirms a breakout of the price.

    The formula is as follows:

    - A = |High - Close(t - 1)|, B = |Low - Close(t - 1)|, C = |High - Low|
    - D = |Close(t - 1) - Open(t - 1)|
    - R = A - 0.5 * B + 0.25 * D when A is the largest of A, B and C, B - 0.5 * A + 0.25 * D
      when B is, else C + 0.25 * D
    - K = max(A, B)
    - N = (Close - Close(t - 1)) + 0.5 * (Close - Open) + 0.25 * (Close(t - 1) - Open(t - 1))
    - Swing Index = 50 * (N / R) * (K / limit_move)
    - Accumulative Swing Index = Sum of the Swing Index since the first period

    The limit move is the largest move a futures contract may make in one period. A stock
    has none, so `limit_move` is a scale in price units: the Swing Index stays within -100
    and 100 only for moves within it.

    Also known as: SI, ASI, Wilder Swing Index, Accumulation Swing Index.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research, section VIII "The Swing Index."

    Referenced against:
    - TA-Lib, SI: https://ta-lib.org/functions/si.html
    - TA-Lib, ASI: https://ta-lib.org/functions/asi.html

    Notes:
        - The first period has no previous period and no Swing Index, so the Accumulative
          Swing Index starts at the Swing Index of the second period, as in TA-Lib, which
          also gives the first period an ASI of 0.
        - A period without any move (R equal to 0) has no Swing Index (NaN), where TA-Lib
          uses 0; the Accumulative Swing Index carries on over it. Neither is rounded,
          where Wilder rounded to whole numbers by hand.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        limit_move (float): The limit move that scales the index, in price units.

    Returns:
        pd.DataFrame: Swing Index and Accumulative Swing Index as columns.
    """
    previous_close = prices_close.shift(1)
    previous_open = prices_open.shift(1)

    high_move = (prices_high - previous_close).abs()
    low_move = (prices_low - previous_close).abs()
    price_range = (prices_high - prices_low).abs()
    previous_body = 0.25 * (previous_close - previous_open).abs()

    swing_range = np.where(
        (high_move >= low_move) & (high_move >= price_range),
        high_move - 0.5 * low_move + previous_body,
        np.where(
            low_move >= price_range,
            low_move - 0.5 * high_move + previous_body,
            price_range + previous_body,
        ),
    )
    swing_range = pd.Series(swing_range, index=prices_close.index).where(
        previous_close.notna()
    )

    numerator = (
        (prices_close - previous_close)
        + 0.5 * (prices_close - prices_open)
        + 0.25 * (previous_close - previous_open)
    )

    swing_index = (
        50
        * numerator
        / swing_range.where(swing_range != 0)
        * np.maximum(high_move, low_move)
        / limit_move
    )
    accumulative_swing_index = swing_index.cumsum().where(prices_close.notna())

    return pd.concat(
        [swing_index, accumulative_swing_index],
        keys=["Swing Index", "Accumulative Swing Index"],
        axis=1,
    )


def get_quantitative_qualitative_estimation(
    prices: pd.Series, window: int, smooth_window: int, multiplier: float
) -> pd.DataFrame:
    """
    Calculate the Quantitative Qualitative Estimation (QQE): a smoothed Relative Strength Index with a trailing stop
    line a multiple of its own average move away, whose crossings show the turns of a trend.

    The QQE smooths the Relative Strength Index with an exponential average (the fast line)
    and trails it with a stop line (the QQE, the slow line), placed a multiple of the RSI's
    own average move away, in the way an Average True Range stop trails the price. The
    stop only moves towards the RSI while the RSI moves away from it, and jumps to the other
    side when the RSI crosses it, so the RSI crossing the QQE line marks a change of trend
    and the side of the line the trend itself.

    The formula is as follows:

    - RSI Moving Average = EMA(RSI(Close, window), smooth_window)
    - Average Move = EMA(EMA(|RSI Moving Average(t) - RSI Moving Average(t - 1)|, 2 * window -
      1), 2 * window - 1)
    - Distance = multiplier * Average Move
    - QQE(t) = max(QQE(t - 1), RSI Moving Average(t) - Distance(t)) while the RSI Moving
      Average stays above QQE, min(QQE(t - 1), RSI Moving Average(t) + Distance(t)) while it
      stays below, and RSI Moving Average(t) -/+ Distance(t) when it crosses QQE

    The EMA over 2 * window - 1 periods has the smoothing constant 1 / window of Wilder's
    average. The common parameters are 14, 5 and 4.236.

    Also known as: QQE, QQE fast and slow line.

    Reference: The QQE is a community indicator of the MetaTrader platform without a
    published primary source.

    Referenced against:
    - pandas-ta-classic, qqe: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/qqe.py#L71

    Notes:
        - The RSI is that of `get_relative_strength_index` and every EMA is seeded with the
          Simple Moving Average of its first values with an empty warm-up (see
          `get_exponential_moving_average`), so the RSI Moving Average starts on period
          `window` + `smooth_window` - 1 and the QQE on period `window` + `smooth_window` +
          2 * (2 * `window` - 1) - 2, at the RSI Moving Average minus the distance.
        - The QQE line switches sides on the period the RSI Moving Average crosses it, as
          in the original MetaTrader indicator. pandas-ta-classic's qqe switches when the RSI Moving
          Average crosses the band on the other side, which can keep its line on the old
          side for a few more periods; otherwise the two are equal.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the RSI.
        smooth_window (int): Number of periods of the EMA of the RSI.
        multiplier (float): The multiple of the average move of the RSI Moving Average that sets
            the distance of the QQE line.

    Returns:
        pd.DataFrame: QQE and RSI Moving Average as columns, in RSI points.
    """
    rsi_moving_average = get_exponential_moving_average(
        get_relative_strength_index(prices, window), smooth_window
    )
    wilder_window = 2 * window - 1
    distance = multiplier * get_exponential_moving_average(
        get_exponential_moving_average(rsi_moving_average.diff().abs(), wilder_window),
        wilder_window,
    )

    # The trailing line runs on plain Python floats; a period without a value has no line
    # and the line carries on from its last value.
    rsi_values = rsi_moving_average.to_numpy(dtype="float64").tolist()
    distances = distance.to_numpy(dtype="float64").tolist()
    qqe = [math.nan] * len(rsi_values)
    line = math.nan
    previous_rsi = math.nan
    for i, (current_rsi, current_distance) in enumerate(zip(rsi_values, distances)):
        if math.isnan(current_rsi) or math.isnan(current_distance):
            continue

        if math.isnan(line):
            line = current_rsi - current_distance
        elif current_rsi > line:
            candidate = current_rsi - current_distance
            line = max(line, candidate) if previous_rsi > line else candidate
        elif current_rsi < line:
            candidate = current_rsi + current_distance
            line = min(line, candidate) if previous_rsi < line else candidate

        qqe[i] = line
        previous_rsi = current_rsi

    return pd.concat(
        [pd.Series(qqe, index=prices.index, dtype="float64"), rsi_moving_average],
        keys=["QQE", "RSI Moving Average"],
        axis=1,
    )


def get_dynamic_momentum_index(
    prices: pd.Series,
    window: int,
    volatility_window: int,
    average_window: int,
    minimum_window: int,
    maximum_window: int,
) -> pd.Series:
    """
    Calculate the Dynamic Momentum Index (DMI): a Relative Strength Index whose window shortens when volatility
    rises, so it reacts faster in volatile markets.

    The Dynamic Momentum Index is a Relative Strength Index whose period changes with the
    volatility: the period shortens when the recent volatility is above its average and
    lengthens when it is below, between 5 and 30 periods. The index thereby reacts quickly in
    volatile markets and stays calm in quiet ones, and is read as the RSI, overbought above
    70 and oversold below 30.

    The formula is as follows:

    - Volatility Index = StdDev(Close, volatility_window) / SMA(StdDev(Close,
      volatility_window), average_window)
    - Period = Int(window / Volatility Index), bounded to minimum_window and maximum_window
    - DMI = 100 * Sum(Up Changes, Period) / (Sum(Up Changes, Period) + Sum(Down Changes,
      Period))

    Chande uses a 14-period base, a 5-period standard deviation and its 10-period average.

    Also known as: DYMI, Chande's Dynamic Momentum Index, variable-term RSI.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    Shares similar logic:
    - finta, DYMI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L752

    Notes:
        - The period changes every period, which leaves no single period for Wilder's
          recursive average, so the RSI of each period sums the gains and losses of its own
          last `Period` changes (Cutler's RSI), the summed form Chande also uses for his
          Momentum Oscillator.
        - The standard deviation is the sample standard deviation (ddof=1). The DMI starts
          on period `volatility_window` + `average_window` - 2, the first Volatility Index,
          once the period has that many changes.
        - finta's DYMI rounds the Volatility Index before the division, bounds the period to
          5 and 30, and takes an exponentially smoothed RSI over the periods before the
          current one, so it differs from this.

    Args:
        prices (pd.Series): Series of prices.
        window (int): The base period that the volatility index divides.
        volatility_window (int): Number of periods of the standard deviation.
        average_window (int): Number of periods of the average of the standard deviation.
        minimum_window (int): The shortest period of the RSI.
        maximum_window (int): The longest period of the RSI.

    Returns:
        pd.Series: DMI values, between 0 and 100.
    """
    standard_deviation = prices.rolling(window=volatility_window).std()
    volatility_index = standard_deviation / get_moving_average(
        standard_deviation, average_window
    )
    periods = np.floor(window / volatility_index).clip(minimum_window, maximum_window)

    price_diff = prices.diff()
    cumulative_gains = np.concatenate(
        [[0.0], price_diff.clip(lower=0).fillna(0).cumsum().to_numpy()]
    )
    cumulative_losses = np.concatenate(
        [[0.0], (-price_diff).clip(lower=0).fillna(0).cumsum().to_numpy()]
    )

    # The sums over the last Period changes of each period, as differences of the running
    # sums; a window holding fewer real price changes than its Period (a series starting
    # later, or a gap of missing prices) has no value.
    positions = np.arange(len(prices))
    period_values = periods.fillna(0).to_numpy().astype(int)
    in_range = periods.notna().to_numpy() & (positions >= period_values)
    starts = np.where(in_range, positions - period_values, 0)
    changes = price_diff.notna().cumsum().to_numpy()
    valid = in_range & (changes - changes[starts] >= period_values)

    gains = cumulative_gains[positions + 1] - cumulative_gains[starts + 1]
    losses = cumulative_losses[positions + 1] - cumulative_losses[starts + 1]
    total = gains + losses

    with np.errstate(divide="ignore", invalid="ignore"):
        dynamic_momentum_index = np.where(
            valid & (total != 0), 100 * gains / total, np.nan
        )

    return pd.Series(dynamic_momentum_index, index=prices.index).where(prices.notna())


def get_price_zone_oscillator(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Price Zone Oscillator (PZO): the share of the recent closes that moved up, from -100 to 100, which
    shows overbought and oversold levels and the direction of a trend.

    The Price Zone Oscillator counts each close as positive when it closed above the
    previous close and as negative when it closed below, and divides the EMA of these signed
    closes by the EMA of the closes. It is thereby the share of the recent price action that
    went up, on a scale of -100 to 100: above 40 is overbought and below -40 oversold in a
    trendless market, and crossings of zero, 15 and -5 are Khalil and Steckler's signals in
    a trend.

    The formula is as follows:

    - Signed Close = sign(Close(t) - Close(t - 1)) * Close(t)
    - PZO = 100 * EMA(Signed Close, window) / EMA(Close, window)

    Both EMAs run over the periods with a change, so they start on the same period.

    Also known as: PZO.

    Reference: Khalil, W. & Steckler, D. (2011). "Entering The Price Zone." Technical
    Analysis of Stocks & Commodities, 29(6), 28-35.

    Shares similar logic:
    - finta, PZO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1542

    Notes:
        - Every EMA is seeded with the Simple Moving Average of its first values with an
          empty warm-up (see `get_exponential_moving_average`), so the PZO starts on period
          `window`. finta's PZO uses averages without a seed and starts the EMA of the
          closes one period earlier, which differs in the first values.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the EMAs.

    Returns:
        pd.Series: PZO values, between -100 and 100.
    """
    price_diff = prices.diff()
    signed_close = np.sign(price_diff) * prices
    closes = prices.where(price_diff.notna())

    return (
        100
        * get_exponential_moving_average(signed_close, window)
        / get_exponential_moving_average(closes, window)
    )


def get_connors_relative_strength_index(
    prices: pd.Series, window: int = 3, streak_window: int = 2, rank_window: int = 100
) -> pd.Series:
    """
    Calculate the Connors Relative Strength Index (ConnorsRSI): the average of a short RSI, the RSI of the streak of
    rises or falls and the rank of the latest change, which shows short-term overbought and oversold levels.

    ConnorsRSI averages three short-term measures of how overbought or oversold a price is,
    each on a scale of 0 to 100: a 3-period RSI of the price, a 2-period RSI of the streak
    (the number of periods in a row the price closed up, positive, or down, negative) and
    the percent rank of the latest one-period return among the returns of the previous 100
    periods. Connors and Alvarez use it for mean reversion: below 10 or 5 is oversold and
    above 90 or 95 overbought.

    The formula is as follows:

    - Streak = Streak(t-1) + 1 if Close(t) > Close(t-1) (and Streak(t-1) >= 0, else 1),
      Streak(t-1) - 1 if Close(t) < Close(t-1) (and Streak(t-1) <= 0, else -1), 0 if unchanged
    - Return = Close(t) / Close(t-1) - 1
    - ConnorsRSI = (RSI(Close, window) + RSI(Streak, streak_window) + 100 * PercentRank(Return,
      rank_window)) / 3

    with the RSI of Wilder (`get_relative_strength_index`) and the percent rank of
    `get_percent_rank`: the share of the previous `rank_window` returns below the latest.

    Also known as: CRSI, ConnorsRSI, Connors RSI.

    Reference: Connors, L., Alvarez, C. & Radtke, M. (2012). "An Introduction to
    ConnorsRSI." Connors Research Trading Strategy Series; Connors, L. & Alvarez, C. (2009).
    "Short Term Trading Strategies That Work." TradingMarkets Publishing.

    Referenced against:
    - TA-Lib, CRSI: https://ta-lib.org/functions/crsi.html

    Notes:
        - Equals TA-Lib's CRSI.
        - The ConnorsRSI starts on period `rank_window` + 1, the first percent rank of a
          return.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the RSI of the price. Defaults to 3.
        streak_window (int): Number of periods of the RSI of the streak. Defaults to 2.
        rank_window (int): Number of previous returns of the percent rank. Defaults to 100.

    Returns:
        pd.Series: ConnorsRSI values, between 0 and 100.
    """
    direction = np.sign(prices.diff()).to_numpy(dtype=float)
    streak = np.full(len(direction), np.nan)
    previous = 0.0

    # The streak continues while the price keeps moving in the same direction and starts
    # again at 1 or -1 on a reversal; an unchanged price resets it to 0.
    for i, step in enumerate(direction.tolist()):
        if math.isnan(step):
            continue
        if step == 0:
            previous = 0.0
        elif step > 0:
            previous = previous + 1 if previous > 0 else 1.0
        else:
            previous = previous - 1 if previous < 0 else -1.0
        streak[i] = previous

    price_rsi = get_relative_strength_index(prices, window)
    streak_rsi = get_relative_strength_index(
        pd.Series(streak, index=prices.index), streak_window
    )
    return_rank = get_percent_rank(prices / prices.shift(1) - 1, rank_window)

    return (price_rsi + streak_rsi + 100 * return_rank) / 3


def get_inertia(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 20,
    rvi_window: int = 14,
    std_window: int = 10,
    method: str = "original",
) -> pd.Series:
    """
    Calculate Dorsey's Inertia: the Relative Volatility Index smoothed with a linear regression, which shows the
    direction of a trend, up above 50 and down below it.

    Inertia is the Relative Volatility Index smoothed with a linear regression: the end point
    of the least-squares line through the last `window` RVI values. Dorsey regarded a trend
    as momentum that persists, inertia, and its direction as given by the direction of
    volatility: above 50 the trend is up, below 50 down.

    The formula is as follows:

    - RVI = Relative Volatility Index(Close, rvi_window, std_window), or with `method`
      "refined" (RVI(High) + RVI(Low)) / 2
    - Inertia = Linear Regression(RVI, window), the end point of the least-squares line

    with the RVI of `get_relative_volatility_index` and the linear regression of
    `get_linear_regression`.

    Also known as: INERTIA, Dorsey Inertia.

    Reference: Dorsey, D. (1995). "Refining the Relative Volatility Index." Technical
    Analysis of Stocks & Commodities, 13(9).

    Referenced against:
    - pandas-ta-classic, inertia: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/inertia.py#L61

    Notes:
        - Equals pandas-ta-classic's inertia with Wilder's smoothing and a `std_window` equal to
          `rvi_window`. By default pandas-ta-classic smooths its RVI with an EMA and takes
          the standard deviation over the RVI window, where the Toolkit follows Dorsey (and
          TA-Lib's RVI) with Wilder's smoothing over 14 periods of a 10-period standard
          deviation.
        - The linear regression can overshoot the 0 to 100 range of the RVI.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the linear regression. Defaults to 20.
        rvi_window (int): Number of periods of Wilder's smoothing in the RVI. Defaults to 14.
        std_window (int): Number of periods of the standard deviation in the RVI.
            Defaults to 10.
        method (str): The Relative Volatility Index it uses. Options are: "original" (the RVI of the close),
            "refined" (the average of the RVI of the highs and of the lows). Defaults to "original".

    Returns:
        pd.Series: Inertia values, around 0 to 100.

    Raises:
        ValueError: If `method` is not "original" or "refined".
    """
    if method == "original":
        relative_volatility_index = get_relative_volatility_index(
            prices_close, rvi_window, std_window
        )
    elif method == "refined":
        relative_volatility_index = (
            get_relative_volatility_index(prices_high, rvi_window, std_window)
            + get_relative_volatility_index(prices_low, rvi_window, std_window)
        ) / 2
    else:
        raise InvalidArgumentError("The method must be 'original' or 'refined'.")

    return get_linear_regression(relative_volatility_index, window)["Linear Regression"]


def get_bias(
    prices: pd.Series, window: int = 26, moving_average: str = "simple"
) -> pd.Series:
    """
    Calculate the Bias: the distance of the price from its moving average, as a share of that average.

    The Bias (deviation rate) is widely used in Asian markets: a price far above its
    average is read as overbought and likely to return to it, far below as oversold. The
    level that counts as far depends on the security and the window.

    The formula is as follows:

    - Bias = Close / MA(Close, window) - 1

    with MA the Simple Moving Average by default.

    Also known as: BIAS, deviation rate, BIAS ratio, distance to the moving average.

    Reference: no single primary source; it is a standard indicator on Asian trading
    platforms. The definition is the one of pandas-ta (Kevin Johnson), continued as pandas-ta-classic,
    which notes the same.

    Referenced against:
    - pandas-ta-classic, bias: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/bias.py#L12

    Notes:
        - Equals pandas-ta-classic's bias. Both give a decimal (0.05 is 5% above the
          average); some platforms multiply it by 100.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the moving average. Defaults to 26.
        moving_average (str): The moving average, one of `MOVING_AVERAGES`. Options are: "simple" (Simple Moving
            Average), "exponential" (Exponential Moving Average), "double_exponential" (Double Exponential
            Moving Average), "triangular" (Triangular Moving Average), "weighted" (Weighted Moving Average),
            "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman Adaptive Moving Average). Defaults to
            "simple".

    Returns:
        pd.Series: Bias values as decimals.

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    check_option("moving_average", moving_average, MOVING_AVERAGES)

    return prices / MOVING_AVERAGES[moving_average](prices, window) - 1


def get_brar(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 26,
) -> pd.DataFrame:
    """
    Calculate the AR and BR sentiment indicators (BRAR), which show how popular a security is and how willing the
    market is to buy it.

    They set how far the prices rose against how far they fell, measured from the open (AR) and from the previous
    close (BR).

    AR (popularity) compares how far the price rose above the open with how far it fell below
    it over the window, and BR (willingness to buy) does the same against the previous
    close. Both are 100 when the upward and downward moves balance; AR from 150 and BR from
    300 are read as overheated, AR below 50 and BR below 40 as depressed, and BR crossing AR
    from below as a buy signal.

    The formula is as follows:

    - AR = 100 * Sum(High - Open, window) / Sum(Open - Low, window)
    - BR = 100 * Sum(Max(High - Close(t-1), 0), window) / Sum(Max(Close(t-1) - Low, 0), window)

    Also known as: BRAR, AR BR, popularity and willingness indicators, sentiment indicator.

    Reference: no single primary source; it is a standard sentiment indicator on Asian
    trading platforms. The definition is the one of pandas-ta (Kevin Johnson), continued as pandas-ta-classic,
    which notes the same.

    Referenced against:
    - pandas-ta-classic, brar: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/brar.py#L18

    Notes:
        - Equals pandas-ta-classic's brar, on its scale of 100 for a balance. A window
          without any downward move has no AR or BR (missing) where pandas-ta-classic gives
          an infinite value.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the sums. Defaults to 26.

    Returns:
        pd.DataFrame: AR and BR, 100 for a balance.
    """
    previous_close = prices_close.shift(1)
    upward_from_close = (prices_high - previous_close).clip(lower=0)
    downward_from_close = (previous_close - prices_low).clip(lower=0)

    open_low_sum = (prices_open - prices_low).rolling(window=window).sum()
    close_low_sum = downward_from_close.rolling(window=window).sum()

    return pd.DataFrame(
        {
            "AR": 100
            * (prices_high - prices_open).rolling(window=window).sum()
            / open_low_sum.where(open_low_sum != 0),
            "BR": 100
            * upward_from_close.rolling(window=window).sum()
            / close_low_sum.where(close_low_sum != 0),
        },
        index=prices_close.index,
    )


def get_pretty_good_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Calculate Johnson's Pretty Good Oscillator (PGO): how far the close lies from its moving average, in units of
    the Average True Range, which shows breakouts from a range.

    The Pretty Good Oscillator measures the distance of the close from its Simple Moving
    Average in units of the average true range, so the distance compares across securities
    and periods of different volatility. Johnson used it for breakouts: a close more than 3
    average ranges above the average is a long entry, more than 3 below a short entry, and
    the trade is closed when the oscillator returns to zero.

    The formula is as follows:

    - PGO = (Close - SMA(Close, window)) / EMA(ATR(window), window)

    with ATR the Average True Range of Wilder (`get_average_true_range`) and EMA the
    exponential moving average seeded with a simple average (`get_exponential_moving_average`).

    Also known as: PGO, Pretty Good Oscillator.

    Reference: Johnson, M. "Pretty Good Oscillator", as documented in FM Labs' technical
    analysis library (https://www.fmlabs.com/reference/default.htm?url=PGO.htm).

    Referenced against:
    - pandas-ta-classic, pgo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/pgo.py#L14

    Notes:
        - Equals pandas-ta-classic's pgo once Wilder's smoothing of the ATR has forgotten
          its seed: the Toolkit's ATR includes the High - Low of the first period, as
          StockCharts does. A period without any range has no PGO (0 / 0, missing) where
          pandas-ta-classic gives 0.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the SMA, the ATR and its EMA. Defaults to 14.

    Returns:
        pd.Series: PGO values, in average true ranges.
    """
    average_true_range = get_exponential_moving_average(
        get_average_true_range(prices_high, prices_low, prices_close, window), window
    )

    return (prices_close - get_moving_average(prices_close, window)) / (
        average_true_range.where(average_true_range != 0)
    )


def get_psychological_line(
    prices_open: pd.Series,
    prices_close: pd.Series,
    window: int = 12,
    method: str = "close",
) -> pd.Series:
    """
    Calculate the Psychological Line (PSL): the share of periods in a window that closed higher, which shows whether
    buyers or sellers have the upper hand.

    The Psychological Line is the percentage of the last `window` periods that closed
    higher, a measure of the mood of the market: above 75 many periods in a row went up and
    buyers may be exhausted (overbought), below 25 the reverse (oversold).

    The formula is as follows:

    - PSL = 100 * Count(Close(t) > Close(t-1), window) / window

    or with `method` "open", the periods that closed above their own open:
    100 * Count(Close > Open, window) / window.

    Also known as: PSL, PSY, psychological line, psychological index.

    Reference: no single primary source; it is a standard sentiment indicator on Asian
    trading platforms. The definition is the one of pandas-ta (Kevin Johnson), continued as pandas-ta-classic,
    which notes the same.

    Referenced against:
    - pandas-ta-classic, psl: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/psl.py#L19

    Notes:
        - Equals pandas-ta-classic's psl, also on the scale of 0 to 100. The first period
          has no previous close, so the PSL of the close starts on period `window`.

    Args:
        prices_open (pd.Series): Series of opening prices, used with `method` "open".
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods counted. Defaults to 12.
        method (str): What counts as an up period. Options are: "close" (a close above the previous close),
            "open" (a close above the period's open). Defaults to "close".

    Returns:
        pd.Series: PSL values, between 0 and 100.

    Raises:
        ValueError: If `method` is not "close" or "open".
    """
    if method == "close":
        change = prices_close.diff()
    elif method == "open":
        change = prices_close - prices_open
    else:
        raise InvalidArgumentError("The method must be 'close' or 'open'.")

    higher = (change > 0).astype(float).where(change.notna())

    return 100 * higher.rolling(window=window).sum() / window


def get_volatility_based_momentum(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 12,
    atr_window: int = 26,
) -> pd.Series:
    """
    Calculate the Volatility-Based Momentum (VBM): the change of the price over a window in units of the Average
    True Range, which compares momentum across tickers of different volatility.

    Volatility-Based Momentum is the change in price over `window` periods in units of the
    Average True Range, instead of as a share of the earlier price as in the Rate of Change.
    A move of 3 average ranges is then large for any security, whatever its price or
    volatility, which makes the momentum comparable across securities.

    The formula is as follows:

    - VBM = (Close(t) - Close(t - window)) / ATR(atr_window)

    with ATR the Average True Range of Wilder (`get_average_true_range`).

    Also known as: VBM, volatility-based momentum, volatility adjusted momentum.

    Reference: the VBM is defined in finta (Peerchemist), without a published primary source;
    the measure is the momentum of Pring (1991) normalized by Wilder's (1978) Average True
    Range.

    Shares similar logic:
    - finta, VBM: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L664

    Notes:
        - finta documents this formula, but its VBM subtracts the earlier close twice
          (Close.diff(n) - Close.shift(n)) and averages the true range with a simple moving
          average, so its values differ.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the price change. Defaults to 12.
        atr_window (int): Number of periods of the Average True Range. Defaults to 26.

    Returns:
        pd.Series: VBM values, in average true ranges.
    """
    average_true_range = get_average_true_range(
        prices_high, prices_low, prices_close, atr_window
    )

    return (prices_close - prices_close.shift(window)) / average_true_range.where(
        average_true_range != 0
    )


def get_archer_moving_average_trends(
    prices: pd.Series,
    short_window: int = 8,
    long_window: int = 21,
    lookback: int = 2,
    moving_average: str = "exponential",
) -> pd.DataFrame:
    """
    Calculate Archer's Moving Averages Trends (AMAT): whether a fast and a slow moving average both rise (a long
    run) or both fall (a short run), which shows the direction of a trend.

    AMAT flags a long run when the short moving average rose over the last `lookback`
    periods while the long one moved (a rising long average confirms the trend, a falling
    one marks a potential bottom), and a short run when the short average fell while the
    long one moved (a potential top when the long one still rose).

    The formula is as follows:

    - Long Run = 1 if Short MA(t) > Short MA(t - lookback) and Long MA(t) != Long MA(t -
      lookback), else 0
    - Short Run = 1 if Short MA(t) < Short MA(t - lookback) and Long MA(t) != Long MA(t -
      lookback), else 0

    with MA the exponential moving average by default.

    Also known as: AMAT, Archer Moving Averages Trends.

    Reference: Archer's TradingView script "Archer Moving Averages Trends", which defines the
    indicator; there is no published primary source.

    Referenced against:
    - pandas-ta-classic, amat: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/amat.py#L15

    Notes:
        - Equals pandas-ta-classic's amat (its long_run and short_run). Both flags are
          missing until both averages have `lookback` earlier values.

    Args:
        prices (pd.Series): Series of prices.
        short_window (int): Number of periods of the short moving average. Defaults to 8.
        long_window (int): Number of periods of the long moving average. Defaults to 21.
        lookback (int): Number of periods over which the averages rise or fall.
            Defaults to 2.
        moving_average (str): The moving average, one of `MOVING_AVERAGES`. Options are: "simple" (Simple Moving
            Average), "exponential" (Exponential Moving Average), "double_exponential" (Double Exponential
            Moving Average), "triangular" (Triangular Moving Average), "weighted" (Weighted Moving Average),
            "hull" (Hull Moving Average), "kaufman_adaptive" (Kaufman Adaptive Moving Average). Defaults to
            "exponential".

    Returns:
        pd.DataFrame: Long Run and Short Run, 1 when the run holds and 0 otherwise.

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    check_option("moving_average", moving_average, MOVING_AVERAGES)

    short_change = MOVING_AVERAGES[moving_average](prices, short_window).diff(lookback)
    long_change = MOVING_AVERAGES[moving_average](prices, long_window).diff(lookback)
    known = short_change.notna() & long_change.notna()
    long_moved = long_change != 0

    return pd.DataFrame(
        {
            "Long Run": ((short_change > 0) & long_moved).astype(float).where(known),
            "Short Run": ((short_change < 0) & long_moved).astype(float).where(known),
        },
        index=prices.index,
    )


def get_ttm_trend(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 6,
) -> pd.Series:
    """
    Calculate Carter's TTM Trend: whether the close lies above the average midpoint of the previous bars (1) or not
    (-1), which shows the direction of the trend.

    The TTM Trend colours each bar by the trend: up (1) when the close is above the average
    midpoint (High + Low) / 2 of the previous `window` bars, and down (-1) when it is not.
    Carter uses it to stay in a trade while the bars keep their colour and to avoid trading
    against it: two bars of the opposite colour in a row signal the trend has turned.

    The formula is as follows:

    - Average Midpoint = SMA((High + Low) / 2, window), of the `window` bars before the
      current one
    - TTM Trend = 1 if Close > Average Midpoint, else -1

    Also known as: TTM Trend, TTM_TRND, trade the markets trend.

    Reference: Carter, J.F. (2005). "Mastering the Trade: Proven Techniques for Profiting
    from Intraday and Swing Trading Setups." McGraw-Hill.

    Referenced against:
    - pandas-ta-classic, ttm_trend: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/ttm_trend.py#L12

    Notes:
        - Equals pandas-ta-classic's ttm_trend. The TTM Trend starts on period `window`,
          the first with `window` earlier bars.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of earlier bars of the average midpoint. Defaults to 6.

    Returns:
        pd.Series: 1 for an up trend and -1 for a down trend.
    """
    average_midpoint = ((prices_high + prices_low) / 2).shift(1).rolling(window).mean()

    return pd.Series(
        np.where(prices_close > average_midpoint, 1.0, -1.0), index=prices_close.index
    ).where(average_midpoint.notna() & prices_close.notna())


def get_correlation_trend_indicator(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the Correlation Trend Indicator (CTI) of a price series: the Pearson correlation of the prices with a
    straight line rising over time.

    Ehlers (2020) measures the trend by how closely the prices follow an upward-sloping
    line: +1 when they rise in a straight line, -1 when they fall in one, and around 0 when
    they move sideways or in a cycle. Unlike the slope, it is bounded and does not depend on
    the price level, so it compares across securities.

    The formula is as follows:

    - CTI = Correlation(Close, x, window), x = 0, ..., window - 1 from the oldest price to
      the latest

    Also known as: CTI, correlation trend, Ehlers correlation trend indicator.

    Reference: Ehlers, J.F. (2020). "Correlation As A Trend Indicator." Technical Analysis of
    Stocks & Commodities, 38(5), 24-27.

    Referenced against:
    - TA-Lib, CTI: https://ta-lib.org/functions/cti.html
    - pandas-ta-classic, cti: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/cti.py#L12

    Notes:
        - Equals TA-Lib's CTI and pandas-ta-classic's cti. TA-Lib defaults to a window of
          20 and pandas-ta-classic to 12.
        - A window of flat prices has no correlation and is missing, where TA-Lib gives 0.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods, at least 2. Defaults to 20.

    Returns:
        pd.Series: CTI values between -1 and 1.
    """
    if window < 2:  # noqa: PLR2004
        raise ValueError("The window of a correlation is at least 2 periods.")

    values = prices.to_numpy(dtype=float)
    correlation = np.full(len(values), np.nan)

    if len(values) >= window:
        windows = sliding_window_view(values, window)
        deviations = windows - windows.mean(axis=1, keepdims=True)
        time = np.arange(window) - (window - 1) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation[window - 1 :] = (deviations @ time) / np.sqrt(
                (deviations**2).sum(axis=1) * (time @ time)
            )

    # A window without variation has no correlation (0 / 0), rather than an infinite one.
    correlation = pd.Series(correlation, index=prices.index)

    return correlation.where(np.isfinite(correlation))


def get_forecast_oscillator(prices: pd.Series, window: int = 5) -> pd.Series:
    """
    Calculate the Forecast Oscillator of a price series: how far the price lies from the Time Series Forecast made
    for it one period earlier, as a share of the price.

    Chande's Forecast Oscillator measures the error of the linear regression forecast: it
    is positive when the price comes in above the forecast of the previous period and
    negative when below it. A forecast that keeps being too low points to a trend that
    accelerates, and the oscillator crossing zero to a change in the direction of the error.

    The formula is as follows:

    - Time Series Forecast(t - 1) = b0 + b1 * window, the least-squares line over the
      `window` periods up to t - 1, extended to t
    - Forecast Oscillator = (Close - Time Series Forecast(t - 1)) / Close

    Also known as: FOSC, Chande Forecast Oscillator, CFO, forecast error.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    Referenced against:
    - TA-Lib, FOSC: https://ta-lib.org/functions/fosc.html
    - Tulip Indicators, fosc: https://tulipindicators.org/fosc

    Shares similar logic:
    - pandas-ta-classic, fosc: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/fosc.py#L12
    - pandas-ta-classic, cfo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/cfo.py#L17

    Notes:
        - Equals TA-Lib's FOSC and Tulip Indicators' fosc divided by 100: the Toolkit gives
          a decimal (0.01 is 1%) where they give a percentage.
        - pandas-ta-classic's fosc and cfo compare the price with the forecast of the same
          period, which includes that price, rather than with the forecast made the period
          before.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods the forecast line is fitted over, at least 2.
            Defaults to 5.

    Returns:
        pd.Series: Forecast Oscillator values as decimals.
    """
    forecast = get_linear_regression(prices, window)["Time Series Forecast"]

    return (prices - forecast.shift(1)) / prices


def get_projection_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Calculate the Projection Oscillator of a price series: where the close lies within Widner's Projection Bands, 0
    on the lower band and 1 on the upper band.

    Widner (1995) projects every high and low of the window forward to the latest period
    along the least-squares slope of the highs and of the lows over the window. The upper
    Projection Band is the highest of the projected highs and the lower band the lowest of
    the projected lows, so the bands follow the trend of the window rather than lying flat
    like the highest high and lowest low of a stochastic oscillator. The Projection
    Oscillator is the stochastic within those bands.

    The formula is as follows:

    - Slope High, Slope Low = the least-squares slopes of the highs and of the lows over the window
    - Upper Band = Max(High(t - i) + i * Slope High), i = 0, ..., window - 1
    - Lower Band = Min(Low(t - i) + i * Slope Low), i = 0, ..., window - 1
    - Projection Oscillator = (Close - Lower Band) / (Upper Band - Lower Band)

    Also known as: PO, Projection Oscillator, Projection Bands, Widner.

    Reference: Widner, M. (1995). "Signaling Change with Projection Bands." Technical Analysis
    of Stocks & Commodities, 13(7), 275-281.

    Shares similar logic:
    - pandas-ta-classic, po: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/po.py#L12

    Notes:
        - Widner gives the oscillator from 0 to 100; the Toolkit gives a decimal from 0 to 1.
        - pandas-ta-classic's po is a simpler measure under the same name, the distance of
          the close from the end point of its linear regression in percent,
          100 * (Close - Linear Regression) / Linear Regression, which follows from
          `get_linear_regression`.
        - A window whose bands coincide has no position within them and is missing.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods, at least 2. Defaults to 14.

    Returns:
        pd.Series: Projection Oscillator values as decimals.
    """
    # The oldest price of each window is window - 1 periods back from the latest, which is 0.
    steps = np.arange(window)[::-1]
    bands = {}
    for name, prices, extreme in (
        ("Upper", prices_high, np.max),
        ("Lower", prices_low, np.min),
    ):
        slope = get_linear_regression(prices, window)["Slope"].to_numpy()
        values = prices.to_numpy(dtype=float)
        bands[name] = np.full(len(values), np.nan)
        if len(values) >= window:
            projected = sliding_window_view(values, window) + np.outer(
                slope[window - 1 :], steps
            )
            bands[name][window - 1 :] = extreme(projected, axis=1)

    upper_band, lower_band = bands["Upper"], bands["Lower"]

    with np.errstate(divide="ignore", invalid="ignore"):
        oscillator = (prices_close.to_numpy(dtype=float) - lower_band) / (
            upper_band - lower_band
        )

    oscillator = pd.Series(oscillator, index=prices_close.index)

    return oscillator.where(np.isfinite(oscillator))


def get_trend_detection_index(
    prices: pd.Series, window: int = 20, multiple: int = 2
) -> pd.DataFrame:
    """
    Calculate the Trend Detection Index (TDI) and its Direction Indicator: how strongly the momentum keeps one sign
    over a window, which shows whether the prices trend or consolidate and in which direction.

    M.H. Pee's Trend Detection Index tells a trending market from a consolidating one. It
    compares the net momentum over the window with how much the momentum moved over a longer
    window: when the momentum keeps one sign, its sum outweighs the absolute momentum of the
    earlier periods and the TDI is positive (a trend); when it changes sign, the sum cancels out
    and the TDI is negative (a consolidation). The Direction Indicator, the sum of the momentum,
    says which way the trend goes. Pee buys when both are positive and sells when the TDI is
    positive and the Direction Indicator negative.

    The formula is as follows:

    - Momentum = Close(t) - Close(t - window)
    - Direction Indicator = Sum(Momentum, window)
    - TDI = |Direction Indicator| - (Sum(|Momentum|, multiple * window) - Sum(|Momentum|, window))

    Also known as: TDI, trend detection index, Pee's trend detection index.

    Reference: Pee, M.H. (2001). "Trend Detection Index." Technical Analysis of Stocks &
    Commodities, 19(10).

    Referenced against:
    - TTR (R), TDI: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/TDI.R#L63

    Notes:
        - Every sum covers momentum values only, so the TDI starts on period
          (multiple + 1) * window - 1. TTR's TDI sets the first `window` momentum values,
          which have no earlier price, to 0 and starts at period multiple * window - 1, with
          sums that include those zeros until the window has passed them.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the momentum and of the Direction Indicator.
            Defaults to 20.
        multiple (int): The longer window of the absolute momentum, as a multiple of
            `window`. Defaults to 2.

    Returns:
        pd.DataFrame: The TDI and the Direction Indicator, in price units.
    """
    momentum = get_momentum(prices, window)
    absolute_momentum = momentum.abs()

    direction_indicator = momentum.rolling(window=window).sum()
    trend_detection_index = direction_indicator.abs() - (
        absolute_momentum.rolling(window=multiple * window).sum()
        - absolute_momentum.rolling(window=window).sum()
    )

    return pd.concat(
        [trend_detection_index, direction_indicator],
        axis=1,
        keys=["TDI", "Direction Indicator"],
    )


def get_dv_intermediate_oscillator(
    prices: pd.Series, window: int = 252, magnitude_weight: float = 0.8
) -> pd.DataFrame:
    """
    Calculate David Varadi's DV Intermediate Oscillator (DVI): a blend of rankings of how far the price stands from
    its short average and of the balance of up and down days, from 0 to 1, which shows how stretched the price is.

    The DVI is a smooth oscillator between 0 and 1 that combines two rankings over the last
    `window` periods. The magnitude ranks how far the price stands above or below its 3-period
    average, smoothed over a short and a long window; the stretch ranks the number of up days
    against down days over a short and a long window. Varadi weights the magnitude 0.8 and the
    stretch 0.2. A DVI above 0.5 marks a price that is stretched upwards (Varadi's mean
    reversion rules sell there) and below 0.5 one stretched downwards.

    The formula is as follows:

    - Return = Close(t) / SMA(Close, 3) - 1
    - Magnitude = SMA((SMA(Return, 5) + SMA(Return, 100) / 10) / 2, 5)
    - Direction = +1 if Close(t) > Close(t - 1), else -1
    - Stretch = SMA((Sum(Direction, 10) + Sum(Direction, 100) / 10) / 2, 2)
    - DVI Magnitude = Percent Rank(Magnitude, window)
    - DVI Stretch = Percent Rank(Stretch, window)
    - DVI = magnitude_weight * DVI Magnitude + (1 - magnitude_weight) * DVI Stretch

    The Percent Rank is `get_percent_rank`: the share of the previous `window` values that are
    lower than the latest one.

    Also known as: DVI, DV Intermediate, DV indicator.

    Reference: Varadi, D. (2009). "What is the DVI?" CSS Analytics, December 13, 2009
    (https://cssanalytics.wordpress.com/2009/12/13/what-is-the-dvi/).

    Referenced against:
    - TTR (R), DVI: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/DVI.R#L50

    Notes:
        - The Percent Rank compares the latest value with the `window` values before it and
          lies between 0 (a new low) and 1 (a new high). TTR's DVI ranks within a window of `n`
          values that includes the latest one and counts it, and equal values, as lower
          (an exact multiplier of 1), so its rank lies between 1/n and 1. Without equal values,
          TTR's rank over n = window + 1 values equals (window * Percent Rank + 1) / (window + 1).
        - An unchanged close counts as a down day, as in TTR's DVI.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of previous periods of the Percent Ranks. Defaults to 252, a year
            of trading days.
        magnitude_weight (float): Weight of the magnitude in the DVI, between 0 and 1; the
            stretch gets the rest. Defaults to 0.8.

    Returns:
        pd.DataFrame: The DVI Magnitude, the DVI Stretch and the DVI, as decimals between 0
        and 1.

    Raises:
        ValueError: If `magnitude_weight` is not between 0 and 1.
    """
    if not 0 <= magnitude_weight <= 1:
        raise ValueError(
            f"magnitude_weight must be between 0 and 1, not {magnitude_weight}."
        )

    price_return = prices / get_moving_average(prices, 3) - 1
    magnitude = get_moving_average(
        (
            get_moving_average(price_return, 5)
            + get_moving_average(price_return, 100) / 10
        )
        / 2,
        5,
    )

    previous_prices = prices.shift(1)
    direction = pd.Series(
        np.where(prices > previous_prices, 1.0, -1.0), index=prices.index
    ).where(prices.notna() & previous_prices.notna())
    # The stretch only takes multiples of 1/40, and equal values must rank as equal: rounding
    # removes the floating-point noise the running sums of the rolling windows leave behind.
    stretch = get_moving_average(
        (direction.rolling(window=10).sum() + direction.rolling(window=100).sum() / 10)
        / 2,
        2,
    ).round(10)

    dvi_magnitude = get_percent_rank(magnitude, window)
    dvi_stretch = get_percent_rank(stretch, window)
    dvi = magnitude_weight * dvi_magnitude + (1 - magnitude_weight) * dvi_stretch

    return pd.concat(
        [dvi_magnitude, dvi_stretch, dvi],
        axis=1,
        keys=["DVI Magnitude", "DVI Stretch", "DVI"],
    )
