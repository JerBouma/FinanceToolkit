"""Momentum Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.technicals.overlap_model import (
    get_exponential_moving_average,
    get_moving_average,
)
from financetoolkit.technicals.volatility_model import (
    get_true_range,
    get_wilder_moving_average,
)

# The Know Sure Thing combines exactly four smoothed Rate of Change components.
KST_COMPONENT_COUNT = 4


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

    The formula is a follows:

    - Typical Price = (High + Low + Close) / 3
    - Raw Money Flow = Typical Price * Volume
    - Money Ratio = Sum(Positive Raw Money Flow, window) / Sum(Negative Raw Money Flow, window)
    - MFI = 100 — (100 / (1 + Money Ratio))

    Also known as: MFI, volume-weighted RSI.

    Reference: Quong, G. & Soudack, A. (1989). "Money Flow Index." Technical Analysis of
    Stocks & Commodities, 7(6).

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

    positive_money_flow = (
        raw_money_flow.where(typical_prices > typical_prices.shift(1), 0)
        .rolling(window=window)
        .sum()
    )
    negative_money_flow = (
        raw_money_flow.where(typical_prices < typical_prices.shift(1), 0)
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

    The formula is a follows:

    - %R = —((Highest High — Close) / (Highest High — Lowest Low)) * 100

    Also known as: %R, Williams %R.

    Reference: Williams, L.R. (1973). "How I Made One Million Dollars... Last Year Trading
    Commodities." Windsor Books.

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

    The formula is a follows:

    - Periods Since High = number of periods between the current period and the most recent
      highest high over the current period plus the previous `window` periods (0 if the
      current period is itself the highest high, `window` if the oldest period in the
      lookback is)
    - Aroon Up = ((window — Periods Since High) / window) * 100
    - Aroon Down = ((window — Periods Since Low) / window) * 100, defined analogously using
      the lowest low

    Also known as: Aroon Up, Aroon Down.

    Reference: Chande, T.S. (1995). "The New Technical Trader." Wiley.

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
        pd.DataFrame: Aroon Up and Aroon Down values.
    """
    # The lookback spans today plus the previous window bars, so the extreme can be up to window periods old and the indicator can reach zero. Reversing before taking the extreme breaks ties on the most recent occurrence, as the definition requires.  # noqa: E501
    periods_since_high = prices_high.rolling(window=window + 1).apply(
        lambda values: window - (len(values) - 1 - values[::-1].argmax()), raw=True
    )
    periods_since_low = prices_low.rolling(window=window + 1).apply(
        lambda values: window - (len(values) - 1 - values[::-1].argmin()), raw=True
    )

    aroon_up = ((window - periods_since_high) / window) * 100
    aroon_down = ((window - periods_since_low) / window) * 100

    return pd.concat([aroon_up, aroon_down], keys=["Aroon Up", "Aroon Down"], axis=1)


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

    The formula is a follows:

    - Typical Price = (High + Low + Close) / 3
    - Mean Deviation = Mean(|Typical Price — SMA(Typical Price, window)|, window)
    - CCI = (Typical Price — SMA(Typical Price, window)) / (constant * Mean Deviation)

    Also known as: CCI.

    Reference: Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic
    Trends." Technical Analysis of Stocks & Commodities, 1(5).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for CCI calculation.
        constant (float): Constant multiplier for CCI. Defaults to 0.015, the conventional
            value chosen so most readings fall within +/-100.

    Returns:
        pd.Series: CCI values.
    """
    typical_prices = (prices_high + prices_low + prices_close) / 3
    sma_typical_prices = typical_prices.rolling(window=window).mean()

    # Every point in the window is measured against the current window's mean, rather than each point against the mean of its own trailing window.  # noqa: E501
    mean_deviation = typical_prices.rolling(window=window).apply(
        lambda window_values: np.abs(window_values - window_values.mean()).mean(),
        raw=True,
    )

    cci_values = (typical_prices - sma_typical_prices) / (constant * mean_deviation)

    return cci_values


def get_relative_vigor_index(
    prices_open: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the (Close-Open) Vigor Index for a given price series.

    This indicator measures the proportion of "upward vigor" — how much of the total
    close-minus-open movement over the window was upward — as a bounded oscillator between 0
    and 1. It is conceptually related to John Ehlers' "Relative Vigor Index" (which compares
    close-open movement to the high-low range using a symmetric 4-bar weighted average) but
    is not numerically identical to Ehlers' published formula: this implementation uses a
    plain rolling sum of upward versus downward close-open movement rather than Ehlers'
    weighted numerator/denominator construction, and does not require high/low data.

    The formula is a follows:

    - Upward Movement = (Close — Open) where positive, else 0
    - Downward Movement = (Open — Close) where positive, else 0
    - RVI = Sum(Upward Movement, window) / (Sum(Upward Movement, window) + Sum(Downward
      Movement, window))

    Also known as: vigor index.

    Notes:
        - Unlike Ehlers' original Relative Vigor Index, this version has no dependency on
          trading volume or the high/low range; it is purely a ratio of upward to total
          close-open movement, in the same spirit as the numerator of the Relative Strength
          Index but based on each period's own open-to-close change rather than period-to-
          period close-to-close changes.
        - There is no single canonical academic citation for this specific formulation; the
          standard textbook treatment of the underlying Relative Vigor Index concept is
          Ehlers, J.F. (2002). "Relative Vigor Index." Technical Analysis of Stocks &
          Commodities, 20(7).

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for RVI calculation.

    Returns:
        pd.Series: RVI values, bounded between 0 and 1.
    """
    close_open_diff = prices_close - prices_open

    up_close_open = close_open_diff.where(close_open_diff > 0, 0)
    down_close_open = -close_open_diff.where(close_open_diff < 0, 0)

    up_sum = up_close_open.rolling(window=window).sum()
    down_sum = down_close_open.rolling(window=window).sum()

    rvi = up_sum / (up_sum + down_sum)

    return rvi


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

    The formula is a follows:

    - Raw Force Index = (Close(t) — Close(t-1)) * Volume(t)
    - Force Index = EMA(Raw Force Index, window)

    Also known as: FI, Elder's Force Index.

    Reference: Elder, A. (1993). "Trading for a Living." Wiley.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for the Exponential Moving Average used to smooth
            the raw Force Index.

    Returns:
        pd.Series: Force Index values.
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

    The formula is a follows:

    - Buying Pressure = Close — Min(Low, Previous Close)
    - True Range = Max[(High — Low), |High — Previous Close|, |Low — Previous Close|]
    - Average(i) = Sum(Buying Pressure, window_i) / Sum(True Range, window_i)
    - Ultimate Oscillator = 100 * [(4 * Average_1) + (2 * Average_2) + Average_3] / (4 + 2 + 1)

    Also known as: UO.

    Reference: Williams, L.R. (1985). "The Ultimate Oscillator." Technical Analysis of Stocks
    & Commodities, 3(4).

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
    prices_close: pd.Series, short_window: int, long_window: int
) -> pd.Series:
    """
    Calculate the Percentage Price Oscillator (PPO) for a given price series.

    The Percentage Price Oscillator is the Moving Average Convergence Divergence line
    re-expressed as a percentage of the long-term EMA, rather than as a raw price
    difference. This makes readings comparable across securities trading at very different
    price levels, which the raw MACD line cannot do.

    The formula is a follows:

    - PPO = ((EMA(Close, short_window) — EMA(Close, long_window)) / EMA(Close, long_window)) * 100

    Also known as: PPO.

    Reference: Appel, G. (2005). "Technical Analysis: Power Tools for Active Investors." FT
    Press. The PPO is a normalized variant of Appel's Moving Average Convergence Divergence
    (Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method." Signalert
    Corp.).

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_window (int): Number of periods for the short-term EMA.
        long_window (int): Number of periods for the long-term EMA.

    Returns:
        pd.Series: PPO values, expressed as a percentage.
    """
    short_ema = get_exponential_moving_average(prices_close, short_window)
    long_ema = get_exponential_moving_average(prices_close, long_window)

    ppo = ((short_ema - long_ema) / long_ema) * 100
    return ppo


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

    The formula is a follows:

    - Displacement = floor(window / 2) + 1
    - DPO = Close(t — Displacement) — Moving Average(t)

    Note that, unlike most oscillators, the moving average itself is *not* shifted — only the
    close price is looked up further back in time.

    Also known as: DPO.

    Reference: The standard textbook treatment is Murphy, J.J. (1999). "Technical Analysis of
    the Financial Markets." New York Institute of Finance; there is no single canonical
    academic paper behind DPO.

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

    The formula is a follows:

    - +DM = High(t) — High(t-1), where positive and greater than —(Low(t) — Low(t-1)), else 0
    - -DM = Low(t-1) — Low(t), where positive and greater than (High(t) — High(t-1)), else 0
    - True Range = Max[(High — Low), |High — Previous Close|, |Low — Previous Close|]
    - +DI = 100 * Wilder's MA(+DM, window) / Wilder's MA(True Range, window)
    - -DI = 100 * Wilder's MA(-DM, window) / Wilder's MA(True Range, window)
    - DX = 100 * |+DI — -DI| / (+DI + -DI)
    - ADX = Wilder's MA(DX, window)

    Also known as: ADX, Average Directional Index.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to consider for ADX calculation.

    Returns:
        pd.Series: ADX values.
    """
    true_range = get_true_range(prices_high, prices_low, prices_close)

    # Wilder's rule: only the larger of the two moves counts and the other is set to zero. Counting both on an outside bar inflates each Directional Indicator and compresses the resulting Directional Index.  # noqa: E501
    up_move = prices_high.diff()
    down_move = -prices_low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smoothed_true_range = get_wilder_moving_average(true_range, window)
    smoothed_plus_dm = get_wilder_moving_average(plus_dm, window)
    smoothed_minus_dm = get_wilder_moving_average(minus_dm, window)

    plus_di = 100 * (smoothed_plus_dm / smoothed_true_range)
    minus_di = 100 * (smoothed_minus_dm / smoothed_true_range)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

    adx = get_wilder_moving_average(dx, window)

    return adx


def get_chande_momentum_oscillator(prices_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Chande Momentum Oscillator (CMO) for a given price series.

    The Chande Momentum Oscillator is closely related to the Relative Strength Index: both
    compare the sum of up-moves to the sum of down-moves over a window. Where RSI computes
    the ratio of average gain to average loss and rescales it to 0-100, CMO instead uses the
    net difference between total gains and total losses, divided by their sum, which lets it
    range symmetrically from -100 to +100 (and, unlike RSI, uses a plain sum rather than
    Wilder's smoothing).

    The formula is a follows:

    - CMO = 100 * (Sum(Up Changes, window) — Sum(Down Changes, window)) / (Sum(Up Changes,
      window) + Sum(Down Changes, window))

    Also known as: CMO.

    Reference: Chande, T.S. (1994). "Adapting Moving Averages to Market Volatility."
    Technical Analysis of Stocks & Commodities, 10(3).

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for CMO calculation.

    Returns:
        pd.Series: CMO values, bounded between -100 and 100.
    """
    price_diff = prices_close.diff(1)

    up_sum = price_diff.where(price_diff > 0, 0).rolling(window=window).sum()
    down_sum = abs(price_diff.where(price_diff < 0, 0)).rolling(window=window).sum()

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

    The formula is a follows:

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

    Notes:
        - Both leading spans are projected forward by `base_window` (26 by default) periods,
          not by their own lookback window. Projecting Leading Span B forward by
          `lead_span_b_window` (52) instead is a common re-implementation error.
        - Because a span plotted `base_window` periods into the future is, at any given
          period, derived from data `base_window` periods in the *past*, both leading spans
          are computable from information available at that period and carry no look-ahead.
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
) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator of a given price series.

    The Stochastic Oscillator measures where the close sits within the high-low range of the
    lookback window, on the premise that in an uptrend prices tend to close near the high of
    the range (and near the low in a downtrend). %D is a moving average of %K used as a
    signal line for crossovers, the same role the signal line plays for MACD.

    The formula is a follows:

    - %K = ((Close — Lowest Low) / (Highest High — Lowest Low)) * 100
    - %D = SMA(%K, smooth_window)

    This is the "Fast Stochastic" formulation (raw %K smoothed once to produce %D). The "Slow
    Stochastic" variant additionally smooths %K itself with an SMA before computing %D.

    Also known as: Stochastic Oscillator, Fast Stochastic, %K/%D.

    Reference: Lane, G.C. Developed in the late 1950s; see Lane, G.C. (1984). "Lane's
    Stochastics." Technical Analysis of Stocks & Commodities, 2(3).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the stochastic calculation.
        smooth_window (int): Number of periods for smoothing the %K values into %D.

    Returns:
        pd.DataFrame: Stochastic Oscillator (%K and %D).
    """
    lowest_low = prices_low.rolling(window=window).min()
    highest_high = prices_high.rolling(window=window).max()

    stochastic_k = ((prices_close - lowest_low) / (highest_high - lowest_low)) * 100
    stochastic_d = stochastic_k.rolling(window=smooth_window).mean()

    return pd.concat(
        [stochastic_k, stochastic_d], keys=["Stochastic %K", "Stochastic %D"], axis=1
    )


def get_moving_average_convergence_divergence(
    prices: pd.Series, short_window: int, long_window: int, signal_window: int
) -> pd.Series:
    """
    Calculate the Moving Average Convergence Divergence (MACD) of a given price series.

    MACD tracks the convergence and divergence of a short-term and a long-term Exponential
    Moving Average of price. The MACD line crossing above/below its own EMA-based signal line
    is the classic trigger for a trend-following signal.

    The formula is a follows:

    - MACD Line = EMA(Close, short_window) — EMA(Close, long_window)
    - Signal Line = EMA(MACD Line, signal_window)

    Also known as: MACD.

    Reference: Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method."
    Signalert Corp.

    Args:
        prices (pd.Series): Series of prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        signal_window (int): Number of periods for the signal line moving average.

    Returns:
        pd.Series: MACD values.
    """
    short_ema = prices.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()

    return pd.concat(
        [macd_line, signal_line], keys=["MACD Line", "Signal Line"], axis=1
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

    The formula is a follows:

    - Average Gain = Wilder's MA(Upward Changes, window)
    - Average Loss = Wilder's MA(Downward Changes, window)
    - RS = Average Gain / Average Loss
    - RSI = 100 — (100 / (1 + RS))

    Also known as: RSI.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for RSI calculation.

    Returns:
        pd.Series: RSI values, bounded between 0 and 100.
    """
    # Calculate price changes
    price_diff = prices.diff(1)

    # Calculate upward and downward price changes
    up_changes = price_diff.where(price_diff > 0, 0)
    down_changes = -price_diff.where(price_diff < 0, 0)

    # Calculate average gains and losses over the specified window using Wilder's smoothing
    avg_gain = get_wilder_moving_average(up_changes, window)
    avg_loss = get_wilder_moving_average(down_changes, window)

    # Calculate the relative strength index
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


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

    The formula is a follows:

    - BOP = (Close — Open) / (High — Low)

    Also known as: BOP.

    Reference: Created by Igor Livshin; featured in Kaufman, P.J. (2005). "New Trading
    Systems and Methods." 4th ed. Wiley.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: BOP values.
    """
    bop_values = ((prices_close - prices_open) / (prices_high - prices_low)).fillna(0)

    return bop_values


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

    The formula is a follows:

    - Median Price = (High + Low) / 2
    - AO = SMA(Median Price, short_window) — SMA(Median Price, long_window)

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

    The formula is a follows:

    - VM+ = |High(t) — Low(t-1)|
    - VM- = |Low(t) — High(t-1)|
    - VI+ = Sum(VM+, window) / Sum(True Range, window)
    - VI- = Sum(VM-, window) / Sum(True Range, window)

    Also known as: VI, Vortex Indicator +/-, trend direction indicator.

    Reference: Botes, E., & Siepman, D. (2010). "The Vortex Indicator." Technical Analysis of
    Stocks & Commodities, 28(1), 20-25.

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

    The formula is a follows:

    - Bull Power = High — EMA(Close, window)
    - Bear Power = Low — EMA(Close, window)

    Also known as: Elder Ray, Bull Power, Bear Power.

    Reference: Elder, A. (1993). "Trading for a Living." Wiley.

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

    The formula is a follows:

    - ROC = (Close(t) / Close(t - window) — 1) * 100

    Also known as: ROC, Price Rate of Change, momentum (in its unscaled form Close(t) —
    Close(t - window)).

    Reference: Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York
    Institute of Finance.

    Args:
        prices_close (pd.Series | pd.DataFrame): Series (or DataFrame with one column per
            ticker) of closing prices.
        window (int): Number of periods to look back for the rate of change calculation.

    Returns:
        pd.Series | pd.DataFrame: Rate of Change values, expressed as a percentage.

    Raises:
        TypeError: If `prices_close` is not a pandas Series or DataFrame.
    """
    if not isinstance(prices_close, pd.Series | pd.DataFrame):
        raise TypeError("prices_close must be a pandas Series or DataFrame.")

    return (prices_close / prices_close.shift(window) - 1) * 100


def get_choppiness_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
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

    The formula is a follows:

    - CHOP = 100 * log10( Sum(True Range, window) / (Max(High, window) — Min(Low, window)) ) / log10(window)

    Also known as: CHOP, Choppiness Index.

    Reference: Dreiss, E.W. (1990s). Developed by Australian commodities trader Bill Dreiss;
    there is no formal journal citation. The standard textbook treatment is Kaufman, P.J.
    (2013). "Trading Systems and Methods." 5th ed. Wiley.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to consider for the Choppiness Index calculation.

    Returns:
        pd.Series: Choppiness Index values, bounded between 0 and 100. Values above 61.8 are
            commonly read as signalling a choppy (range-bound) market, while values below 38.2
            are commonly read as signalling a trending market.

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

    true_range = get_true_range(prices_high, prices_low, prices_close)

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

    The formula is a follows:

    - RCMA(i) = SMA(ROC(Close, roc_windows[i]), sma_windows[i])
    - KST = Sum(RCMA(i) * weights[i]) for i = 1..4
    - Signal Line = SMA(KST, signal_window)

    Also known as: KST, Pring's Know Sure Thing, Summed Rate of Change.

    Reference: Pring, M.J. (1992). "The Know Sure Thing (KST)." Technical Analysis of Stocks
    & Commodities, 10(6).

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

    Returns:
        pd.DataFrame: Know Sure Thing (KST and Signal Line) values.

    Raises:
        TypeError: If `prices_close` is not a pandas Series.
        ValueError: If `roc_windows`, `sma_windows` and `weights` are not all length 4.
    """
    if not isinstance(prices_close, pd.Series):
        raise TypeError("prices_close must be a pandas Series.")

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
        smoothed_rate_of_change = get_moving_average(rate_of_change, sma_window)
        kst = kst + smoothed_rate_of_change * weight

    signal_line = get_moving_average(kst, signal_window)

    return pd.concat([kst, signal_line], keys=["KST", "Signal Line"], axis=1)
