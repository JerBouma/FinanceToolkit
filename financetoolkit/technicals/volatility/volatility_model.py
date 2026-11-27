"""Volatility Module"""

__docformat__ = "google"


import math

import numpy as np
import pandas as pd

from financetoolkit.econometrics.statistics.statistics_model import (
    get_linear_regression,
)
from financetoolkit.technicals.overlap import overlap_model
from financetoolkit.technicals.overlap.overlap_model import (
    get_exponential_moving_average,
)
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# Number of trading days used to annualize realized volatility.
TRADING_DAYS_PER_YEAR = 252


def get_true_range(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series
) -> pd.Series:
    """
    Calculate the True Range (TR): the range of a period including any gap from the previous close, the basic
    measure of how far the price moved.

    The True Range is the greatest of three measures of a single period's price movement,
    designed to capture gaps between periods (e.g. an overnight gap) that a plain
    High - Low range would miss: the current high-low range, the distance from the prior
    close to the current high, and the distance from the prior close to the current low.
    It is the building block for the Average True Range and the Average Directional Index.

    The formula is as follows:

    - True Range = Max[(High - Low), |High - Previous Close|, |Low - Previous Close|]

    Also known as: TR.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Referenced against:
    - TA-Lib, TRANGE: https://ta-lib.org/functions/trange.html

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
    Calculate Wilder's Smoothed Moving Average: an exponential moving average with a smoothing factor of one over
    the window, which smooths slowly and is used in Wilder's indicators.

    Wilder's smoothing is the specific recursive smoothing method J. Welles Wilder Jr. used
    for every one of his original indicators (the Relative Strength Index, the Average True
    Range and the Average Directional Index). It is related to an Exponential Moving Average
    but uses a slower smoothing constant of 1/window (equivalent to an EMA with
    span = 2 * window - 1), rather than the standard EMA constant of 2/(window+1). Using a
    plain Simple Moving Average, or a standard EMA, instead of Wilder's specific smoothing
    is one of the most common sources of numerical discrepancies when re-implementing
    Wilder's indicators.

    The formula is as follows:

    - First value = SMA(values, window) — a simple average seeds the recursion
    - Value(t) = Value(t-1) + (1 / window) * (Series(t) - Value(t-1)), for every period after

    A period without a value has no average (NaN), and the next value continues from the
    last average.

    Also known as: Wilder's Moving Average, Modified Moving Average, Running Moving Average,
    Smoothed Moving Average.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Shares similar logic:
    - TA-Lib, RMA: https://ta-lib.org/functions/rma.html

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

    # The recursion runs on plain Python floats, since writing a pandas Series one
    # element at a time costs tens of microseconds per element.
    current_values = values.to_numpy(dtype="float64").tolist()
    averages = [math.nan] * length
    averages[start_position] = float(seed.iloc[start_position])

    for i in range(start_position + 1, length):
        current_value = current_values[i]
        previous_average = averages[i - 1]

        if math.isnan(current_value):
            averages[i] = previous_average
            continue

        averages[i] = previous_average + (1 / window) * (
            current_value - previous_average
        )

    # A period without a value has no average, while the recursion carries on from the last
    # one on the next value, so a series that ends early (a delisted ticker) stops instead
    # of repeating its last average.
    return pd.Series(averages, index=values.index, dtype="float64").where(
        values.notna()
    )


# The moving averages of get_moving_average that take only a window, which the Bollinger Bands can take as their
# middle band as TA-Lib's BBANDS takes a matype.
MOVING_AVERAGES = {
    "simple": overlap_model.get_moving_average,
    "exponential": overlap_model.get_exponential_moving_average,
    "double_exponential": overlap_model.get_double_exponential_moving_average,
    "triangular": overlap_model.get_triangular_moving_average,
    "weighted": overlap_model.get_weighted_moving_average,
    "hull": overlap_model.get_hull_moving_average,
    "triple_exponential": overlap_model.get_triple_exponential_moving_average,
    "zero_lag": overlap_model.get_zero_lag_exponential_moving_average,
    "wilder": get_wilder_moving_average,
    "sine": overlap_model.get_sine_weighted_moving_average,
    "fibonacci": overlap_model.get_fibonacci_weighted_moving_average,
    "pascal": overlap_model.get_pascal_weighted_moving_average,
}


def get_average_true_range(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Average True Range (ATR): the average range of the periods, gaps included, a measure of how much
    the price moves.

    The Average True Range measures volatility by smoothing the True Range (a single
    period's price movement that also accounts for gaps between periods) with Wilder's
    smoothing method, rather than a plain moving average.

    The formula is as follows:

    - True Range = Max[(High - Low), |High - Previous Close|, |Low - Previous Close|]
    - ATR = Wilder's Smoothed Moving Average of True Range over `window` periods

    Also known as: ATR.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Referenced against:
    - TA-Lib, ATR: https://ta-lib.org/functions/atr.html

    Notes:
        - This follows the StockCharts convention: the first period, which has no previous
          close, has a True Range of High - Low, and the first ATR is the average of the
          first `window` True Ranges, on period `window` - 1. TA-Lib's ATR starts
          the True Range on the second period instead, so its first ATR is one period later
          and averages periods 1 to `window`; the two converge as the smoothing forgets the
          seed.

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


def get_mass_index(
    prices_high: pd.Series, prices_low: pd.Series, ema_window: int, sum_window: int
) -> pd.Series:
    """
    Calculate the Mass Index: the sum of the ratios of an exponential moving average of the high-low range to its
    own average, which signals a reversal when the range widens and then narrows.

    The Mass Index looks for reversals in the width of the trading range rather than in
    price. It divides an EMA of the High - Low range by an EMA of that EMA, which is above
    1 while the range widens and below 1 while it narrows, and sums that ratio over
    `sum_window` periods. Dorsey's "reversal bulge" is the Mass Index rising above 27 and
    then falling back below 26.5, after which the trend is expected to reverse; the index
    does not say in which direction.

    The formula is as follows:

    - Single EMA = EMA(High - Low, ema_window)
    - Double EMA = EMA(Single EMA, ema_window)
    - Mass Index = Sum(Single EMA / Double EMA, sum_window)

    The conventional parameters are 9 periods for the EMAs and 25 for the sum, so the
    Mass Index of a range that neither widens nor narrows is 25.

    Also known as: Mass Index, MI.

    Reference: Dorsey, D. (1992). "The Mass Index." Technical Analysis of Stocks &
    Commodities, 10(6). The calculation follows StockCharts ChartSchool, "Mass Index".

    Referenced against:
    - TA-Lib, MASSI: https://ta-lib.org/functions/massi.html

    Notes:
        - Both EMAs are seeded with the Simple Moving Average of their first values (see
          `overlap_model.get_exponential_moving_average`), so the ratio starts on the
          2 * `ema_window` - 1st period and the Mass Index `sum_window` - 1 periods later,
          on the 2 * (`ema_window` - 1) + `sum_window`th period (the 41st with 9 and 25).
        - The calculation runs over the periods with both a high and a low, so a missing
          period does not empty the next `sum_window` sums; the missing periods stay NaN.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        ema_window (int): Number of periods of the single and double EMA.
        sum_window (int): Number of periods over which the EMA ratio is summed.

    Returns:
        pd.Series: Mass Index values.
    """
    observed = prices_high.notna() & prices_low.notna()
    price_range = prices_high[observed] - prices_low[observed]

    single_ema = get_exponential_moving_average(price_range, ema_window)
    double_ema = get_exponential_moving_average(single_ema, ema_window)
    mass_index = (single_ema / double_ema).rolling(window=sum_window).sum()

    return mass_index.reindex(prices_high.index)


def get_supertrend(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Calculate the Supertrend: a trailing stop a multiple of the Average True Range away from the median price, which
    shows the direction of the trend and flips when the price crosses it.

    The Supertrend indicator plots a single trailing line that flips between sitting below
    price (in an uptrend) and above price (in a downtrend). The line is built from two bands
    offset from the median price ((High + Low) / 2) by a multiple of the Average True Range,
    which are then "ratcheted" period over period — each band can only move in the direction
    that tightens around price — so that the active band only flips to the other side once
    the closing price actually crosses it. This makes Supertrend both a trend filter (the
    flip direction signals a trend change) and a trailing stop-loss level.

    The formula is as follows:

    - Basic Upper Band = (High + Low) / 2 + multiplier * ATR(window)
    - Basic Lower Band = (High + Low) / 2 - multiplier * ATR(window)
    - Final Upper Band(t) = Basic Upper Band(t) if Basic Upper Band(t) < Final Upper Band(t-1)
      or Close(t-1) > Final Upper Band(t-1), else Final Upper Band(t-1)
    - Final Lower Band(t) = Basic Lower Band(t) if Basic Lower Band(t) > Final Lower Band(t-1)
      or Close(t-1) < Final Lower Band(t-1), else Final Lower Band(t-1)
    - While in an uptrend, Supertrend = Final Lower Band, until Close crosses below it, at
      which point the trend flips to a downtrend and Supertrend = Final Upper Band (and vice
      versa)

    Also known as: Supertrend, SuperTrend.

    Referenced against:
    - TA-Lib, SUPERTREND: https://ta-lib.org/functions/supertrend.html

    Notes:
        - There is no academic journal citation for Supertrend. Like the Parabolic SAR, it is
          a practitioner-developed trailing-stop/trend indicator rather than one derived from
          a published financial paper.
        - The trend is initialized as an uptrend (Trend Direction = 1) on the first available
          period, since there is no prior period to determine the starting direction from.
        - A period with a missing price has no Supertrend; the bands carry on from the last
          period with prices.

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

    supertrend = pd.Series(index=prices_close.index, dtype="float64")
    trend_direction = pd.Series(index=prices_close.index, dtype="float64")

    if length == 0:
        return pd.concat(
            [supertrend, trend_direction],
            keys=["Supertrend", "Trend Direction"],
            axis=1,
        )

    # The recursion runs on plain Python lists, since writing pandas Series one element
    # at a time costs tens of microseconds per element.
    basic_upper = basic_upper_band.to_numpy(dtype="float64").tolist()
    basic_lower = basic_lower_band.to_numpy(dtype="float64").tolist()
    close = prices_close.to_numpy(dtype="float64").tolist()
    final_upper = [math.nan] * length
    final_lower = [math.nan] * length
    direction = [math.nan] * length
    trend = [math.nan] * length

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    # Without valid bands there is no trend yet, so the direction stays NaN until they start.
    direction[0] = (
        math.nan if math.isnan(basic_upper[0]) or math.isnan(basic_lower[0]) else 1.0
    )
    trend[0] = final_lower[0]

    for i in range(1, length):
        if math.isnan(basic_upper[i]) or math.isnan(basic_lower[i]):
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            direction[i] = direction[i - 1]
            trend[i] = trend[i - 1]
            continue

        # The bands are NaN until the Average True Range has a full window, so the first bar with valid bands seeds the recursion. Comparing against the NaN carried in from the previous bar would fail both branches and propagate NaN indefinitely.  # noqa: E501
        if math.isnan(final_upper[i - 1]) or math.isnan(final_lower[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            direction[i] = 1.0
            trend[i] = final_lower[i]
            continue

        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if direction[i - 1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1.0
                trend[i] = final_upper[i]
            else:
                direction[i] = 1.0
                trend[i] = final_lower[i]
        elif close[i] > final_upper[i]:
            direction[i] = 1.0
            trend[i] = final_lower[i]
        else:
            direction[i] = -1.0
            trend[i] = final_upper[i]

    # A period without bands (a missing price) has no Supertrend, while the recursion
    # carries on from the last bands, so a series that ends early stops there.
    has_bands = basic_upper_band.notna() & basic_lower_band.notna()
    supertrend = pd.Series(trend, index=prices_close.index, dtype="float64").where(
        has_bands
    )
    trend_direction = pd.Series(
        direction, index=prices_close.index, dtype="float64"
    ).where(has_bands)

    return pd.concat(
        [supertrend, trend_direction], keys=["Supertrend", "Trend Direction"], axis=1
    )


def get_chandelier_exit(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Calculate the Chandelier Exit: a trailing stop a multiple of the Average True Range below the highest high (or
    above the lowest low), which keeps a trade open while the trend lasts.

    The Chandelier Exit is a trailing stop set a multiple of the Average True Range away
    from the most extreme price of the last `window` periods, so the stop "hangs" from the
    highest high of a long position like a chandelier from a ceiling. The long exit sits
    below the highest high and the short exit above the lowest low; because the distance
    is in ATRs, the stop is wider in volatile markets and tighter in quiet ones.

    The formula is as follows:

    - Long Exit = Highest High(window) - multiplier * ATR(window)
    - Short Exit = Lowest Low(window) + multiplier * ATR(window)

    The conventional parameters are 22 periods (about a month of trading days) and a
    multiplier of 3.

    Also known as: Chandelier Exit, Chandelier Stop, CE.

    Reference: Le Beau, C., & Lucas, D.W. (1992). "Technical Traders Guide to Computer
    Analysis of the Futures Market." Business One Irwin; popularized in Elder, A. (2002).
    "Come Into My Trading Room." John Wiley & Sons. The calculation follows StockCharts
    ChartSchool, "Chandelier Exit".

    Shares similar logic:
    - TA-Lib, CE: https://ta-lib.org/functions/ce.html

    Notes:
        - The ATR is `get_average_true_range`: Wilder's smoothing of the True Range, the
          first True Range High - Low and the first ATR the average of the first `window`
          True Ranges (the StockCharts convention), so the exits start on period
          `window` - 1, together with the first full window of highs and lows.
        - The highest high and lowest low include the current period, as on StockCharts.
        - The calculation runs over the periods with a high, a low and a close, so a missing
          period does not empty the next `window` extremes; the missing periods stay NaN.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the highest high, the lowest low and the ATR.
        multiplier (float): Number of ATRs the exits sit from the highest high and the
            lowest low.

    Returns:
        pd.DataFrame: The Long Exit and Short Exit as columns, in price units.
    """
    observed = prices_high.notna() & prices_low.notna() & prices_close.notna()
    high, low = prices_high[observed], prices_low[observed]

    average_true_range = get_average_true_range(
        high, low, prices_close[observed], window
    )
    long_exit = high.rolling(window=window).max() - multiplier * average_true_range
    short_exit = low.rolling(window=window).min() + multiplier * average_true_range

    return pd.concat(
        [long_exit, short_exit], keys=["Long Exit", "Short Exit"], axis=1
    ).reindex(prices_high.index)


def get_keltner_channels(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    atr_window: int,
    multiplier: float,
    method: str = "exponential",
) -> pd.DataFrame:
    """
    Calculate the Keltner Channels: bands a multiple of the Average True Range above and below an exponential moving
    average, which show breakouts and overbought and oversold levels.

    Keltner Channels plot a volatility-based envelope around an Exponential Moving Average of
    price, with the envelope width set by a multiple of the Average True Range rather than a
    fixed percentage or a standard deviation. This makes the channel width adapt to the
    asset's own volatility regime, similar in spirit to Bollinger Bands but based on true
    range instead of standard deviation.

    The formula is as follows:

    - Middle Line = EMA(Close, window)
    - Upper Line = Middle Line + multiplier * ATR(atr_window)
    - Lower Line = Middle Line - multiplier * ATR(atr_window)

    Keltner's original channel is centred on a simple average of
    the typical price with the average daily range as its width:

    - Typical Price = (High + Low + Close) / 3
    - Middle Line = SMA(Typical Price, window)
    - Upper Line = Middle Line + multiplier * SMA(High - Low, window)
    - Lower Line = Middle Line - multiplier * SMA(High - Low, window)

    Keltner used a 10-day average and one average range (a `multiplier` of 1) on either
    side; `atr_window` is not used by the original channel.

    Also known as: Keltner Bands, Keltner's ten-day moving average rule.

    Reference: Keltner, C.W. (1960). "How to Make Money in Commodities." The Keltner
    Statistical Service. The commonly used modern variant (EMA midline with an ATR-based
    envelope, the default here) was popularized by Linda Bradford Raschke in the 1980s.

    Referenced against:
    - TA-Lib, KC: https://ta-lib.org/functions/kc.html
    - ta, KeltnerChannel: https://github.com/bukosabino/ta/blob/0.11.0/ta/volatility.py#L183

    Notes:
        - The conventional parameters, as on StockCharts, are a 20-period EMA, a 10-period
          ATR and a multiplier of 2.
        - The EMA is seeded with the Simple Moving Average of the first `window` closes (see
          `overlap_model.get_exponential_moving_average`), so the channel starts on period
          `window` - 1 or `atr_window` - 1, whichever is later.
        - The original channel agrees with ta's `KeltnerChannel` in its original version with the same
          window and a `multiplier` of 1; it starts on period `window` - 1.
        - TA-Lib's KC centres the channel on an exponential moving average of the typical price instead of the
          close.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the Exponential Moving Average (the middle line).
        atr_window (int): Number of periods for ATR calculation.
        multiplier (float): Multiplier for ATR (or, for the original channel, the
            average range) to determine channel width.
        method (str): The channels. Options are: "exponential" (the EMA of the close with ATR bands),
            "original" (Keltner's simple average of the typical price with average range bands). Defaults to
            "exponential".

    Returns:
        pd.DataFrame: Keltner Channels (upper, middle, lower).
    """
    if method == "original":
        typical_price = (prices_high + prices_low + prices_close) / 3
        middle_line = typical_price.rolling(window=window).mean()
        band_width = (prices_high - prices_low).rolling(window=window).mean()
    elif method == "exponential":
        middle_line = get_exponential_moving_average(prices_close, window)
        band_width = get_average_true_range(
            prices_high, prices_low, prices_close, atr_window
        )
    else:
        raise InvalidArgumentError("The method must be 'exponential' or 'original'.")

    upper_line = middle_line + multiplier * band_width
    lower_line = middle_line - multiplier * band_width

    return pd.concat(
        [upper_line, middle_line, lower_line],
        keys=["Upper Line", "Middle Line", "Lower Line"],
        axis=1,
    )


def get_donchian_channels(
    prices_high: pd.Series,
    prices_low: pd.Series,
    window: int,
    include_current: bool = False,
) -> pd.DataFrame:
    """
    Calculate the Donchian Channels: the highest high and the lowest low of the previous periods, whose breakouts
    signal the start of a trend.

    Donchian Channels plot the highest high and lowest low over the `window` periods
    *preceding* the current one, with the middle line being the average of the two. They
    are used to identify breakouts — Donchian's original rule buys when price exceeds the
    highest high of the preceding N periods — and to gauge the overall volatility of the
    price range.

    The formula is as follows:

    - Upper Channel = Max(High, window), ending one period before the current period
    - Lower Channel = Min(Low, window), ending one period before the current period
    - Middle Channel = (Upper Channel + Lower Channel) / 2

    Also known as: Donchian Bands, price channel.

    Reference: Donchian, R.D. (1960s). Popularized through his "5- and 20-Day Moving Average
    Trading Rule" commodity trading system; the channel itself is described in Kaufman, P.J.
    (2013). "Trading Systems and Methods." 5th ed. Wiley.

    Referenced against:
    - TA-Lib, DONCHIAN: https://ta-lib.org/functions/donchian.html

    Notes:
        - The current period is deliberately excluded from the lookback. StockCharts states
          the rule directly: "The Price Channel formula doesn't include the most recent
          period. Price Channels are based on prices prior to the current period... A
          channel break would not be possible if the most recent period was used." Including
          the current bar makes `High > Upper Channel` impossible by construction, which
          silently disables the indicator's primary use as a breakout signal.
        - Because the channel is built entirely from prior periods, it uses no information
          from the current bar and is safe to compare the current bar's price against.
        - Including the current period, the window ends on the current period, which equals
          TA-Lib's DONCHIAN, pandas-ta-classic's donchian and ta's DonchianChannel (the
          channel then traces the extremes, as used for the squeeze momentum and the
          Ichimoku lines, rather than a breakout level).
        - TA-Lib's DONCHIAN includes the current period.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods, ending one period ago, to consider for the
            Donchian Channels.
        include_current (bool): Whether the window ends on the current period instead of
            the period before it. Defaults to False.

    Returns:
        pd.DataFrame: Donchian Channels (upper, middle, lower).
    """
    upper_channel = prices_high.rolling(window=window).max()
    lower_channel = prices_low.rolling(window=window).min()

    if not include_current:
        upper_channel = upper_channel.shift(1)
        lower_channel = lower_channel.shift(1)

    middle_channel = (upper_channel + lower_channel) / 2

    return pd.concat(
        [upper_channel, middle_channel, lower_channel],
        keys=["Upper Channel", "Middle Channel", "Lower Channel"],
        axis=1,
    )


def get_volatility_cone(
    prices_close: pd.Series,
    windows: list[int] | None = None,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Calculate the Volatility Cone: the range of the historical annualised realised volatility over several windows,
    which shows whether the current volatility is high or low for each window.

    The Volatility Cone summarizes the distribution of historical annualized realized
    volatility over a range of rolling windows, showing how the current realized
    volatility for each window compares to its own historical range. It is commonly
    used to judge whether current (or implied) volatility is cheap or expensive
    relative to history.

    The formula is as follows:

    - Log Return(t) = ln(Close(t) / Close(t-1))
    - Realized Volatility(window) = StdDev(Log Return, window) * sqrt(Periods per Year)
    - For each window, the cone reports the minimum, the 10th, 25th, 50th, 75th and 90th
      percentiles, the maximum and the most recent value of that realized volatility series

    Also known as: volatility cone, realized volatility cone.

    Reference: Burghardt, G. & Lane, M. (1990). "How to Tell if Options are Cheap." Journal
    of Portfolio Management, 16(2), 72-78, and the related treatment in Natenberg, S. (1994). "Option
    Volatility and Pricing." McGraw-Hill — the volatility cone is a standard tool in options
    trading practice for comparing realized volatility across lookback windows.

    Args:
        prices_close (pd.Series): Series of closing prices.
        windows (list[int] | None): The rolling windows (in periods) to calculate
            realized volatility for. Defaults to [10, 20, 30, 60, 90, 120].
        periods_per_year (float): The observations per year the volatility is annualized
            with, e.g. 252 for daily and 52 for weekly prices. Defaults to 252.

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
            periods_per_year
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
    prices: pd.Series,
    window: int,
    num_std_dev: float,
    moving_average: str = "simple",
) -> pd.DataFrame:
    """
    Calculate the Bollinger Bands: a moving average with bands a number of standard deviations above and below it,
    which show how high or low the price is relative to its recent volatility.

    Bollinger Bands plot an envelope around a Simple Moving Average of price, with the
    envelope width set to a multiple of the rolling standard deviation of price over the
    same window. Because the bands widen and narrow with the security's own volatility,
    they are commonly used to gauge whether price is relatively high or low versus its
    recent range, and periods of unusually narrow bands ("the squeeze") are read as a sign
    that a volatility expansion may be imminent.

    The formula is as follows:

    - Middle Band = SMA(Close, window), or another moving average of `moving_average`
    - Upper Band = Middle Band + num_std_dev * StdDev(Close, window)
    - Lower Band = Middle Band - num_std_dev * StdDev(Close, window)

    Also known as: Bollinger Bands.

    Reference: Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.

    Referenced against:
    - TA-Lib, BBANDS: https://ta-lib.org/functions/bbands.html

    Shares similar logic:
    - pandas-ta-classic, bbands: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/bbands.py#L48

    Notes:
        - The rolling standard deviation uses the *population* convention (dividing by n,
          i.e. `ddof=0`), not pandas' default sample convention (dividing by n-1). This is
          what Bollinger himself specifies — "(We use the population calculation for
          standard deviation)", bollingerbands.com — and matches both TA-Lib (whose variance
          divides by the period) and StockCharts. For the default 20-period window the
          sample convention inflates the band half-width by a factor of sqrt(20/19), about
          2.6%.
        - With another `moving_average` the middle band is that average while the standard
          deviation stays the one around the simple average of the window, as in TA-Lib's
          BBANDS with a `matype`.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods for the moving average.
        num_std_dev (float): Number of standard deviations for the bands.
        moving_average (str): The moving average of the middle band, one of `MOVING_AVERAGES` (the window-only
            methods of `get_moving_average`). Options are: "simple" (Simple Moving Average),
            "exponential" (Exponential Moving Average), "double_exponential" (Double Exponential Moving
            Average), "triple_exponential" (Triple Exponential Moving Average), "zero_lag" (Zero Lag Exponential
            Moving Average), "wilder" (Wilder's Moving Average), "triangular" (Triangular Moving Average),
            "weighted" (Weighted Moving Average), "hull" (Hull Moving Average), "sine" (Sine Weighted Moving
            Average), "fibonacci" (Fibonacci Weighted Moving Average), "pascal" (Pascal Weighted Moving
            Average). Defaults to "simple".

    Returns:
        pd.DataFrame: Bollinger Bands (upper, middle, lower).

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    check_option("moving_average", moving_average, MOVING_AVERAGES)

    rolling_mean = MOVING_AVERAGES[moving_average](prices, window)
    rolling_std = prices.rolling(window=window).std(ddof=0)

    upper_band = rolling_mean + (num_std_dev * rolling_std)
    lower_band = rolling_mean - (num_std_dev * rolling_std)

    return pd.concat(
        [upper_band, rolling_mean, lower_band, prices],
        axis=1,
        keys=["Upper Band", "Middle Band", "Lower Band", "Close"],
    )


def get_bollinger_percent_b_and_bandwidth(
    prices: pd.Series, window: int, num_std_dev: float
) -> pd.DataFrame:
    """
    Calculate the Bollinger %B and BandWidth: where the close lies within the Bollinger Bands and how wide the bands
    are, which show overbought and oversold levels and squeezes of volatility.

    Bollinger derived two indicators from his bands. %B places the close within the bands:
    0 is on the lower band, 0.5 on the middle band and 1 on the upper band, with values
    below 0 or above 1 for a close outside the bands. BandWidth is the distance between the
    bands relative to the middle band, a normalized measure of volatility in which a
    six-month low marks "the Squeeze" that often precedes a volatility expansion.

    The formula is as follows:

    - %B = (Close - Lower Band) / (Upper Band - Lower Band)
    - BandWidth = (Upper Band - Lower Band) / Middle Band

    Both are returned as decimals (a %B of 1 is a close on the upper band, a BandWidth of
    0.10 is bands 10% of the middle band apart); StockCharts shows BandWidth multiplied by
    100.

    Also known as: Percent B, %b, Bollinger Band Width, BBW.

    Reference: Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill, chapters
    7 (%b) and 8 (BandWidth).

    Referenced against:
    - TA-Lib, PERCENTB: https://ta-lib.org/functions/percentb.html
    - TA-Lib, BBW: https://ta-lib.org/functions/bbw.html

    Notes:
        - The bands are those of `get_bollinger_bands`: a Simple Moving Average with the
          *population* standard deviation (ddof=0), as Bollinger specifies.
        - A window of identical prices has bands with no width, which leaves %B undefined
          (missing) and BandWidth at 0.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods for the moving average and standard deviation.
        num_std_dev (float): Number of standard deviations for the bands.

    Returns:
        pd.DataFrame: %B and BandWidth as columns, both decimals.
    """
    bollinger_bands = get_bollinger_bands(prices, window, num_std_dev)
    band_range = bollinger_bands["Upper Band"] - bollinger_bands["Lower Band"]

    percent_b = (prices - bollinger_bands["Lower Band"]) / band_range.where(
        band_range != 0
    )
    bandwidth = band_range / bollinger_bands["Middle Band"]

    return pd.concat([percent_b, bandwidth], keys=["%B", "BandWidth"], axis=1)


def get_normalized_average_true_range(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Normalized Average True Range (NATR): the Average True Range as a share of the close, which
    compares volatility across tickers of different prices.

    The Average True Range is in price units, so it does not compare across securities with
    different prices or over long periods in which the price changed a lot. Dividing it by
    the close expresses the average range as a share of the price, which makes it
    comparable across markets.

    The formula is as follows:

    - NATR = ATR(window) / Close

    The result is a decimal (0.02 is an average range of 2% of the close); TA-Lib and
    pandas-ta-classic multiply it by 100.

    Also known as: NATR, ATR percent, ATRP.

    Reference: Forman, J. (2006). "Cross-Market Evaluations With Normalized Average True
    Range." Technical Analysis of Stocks & Commodities, 24(5).

    Referenced against:
    - TA-Lib, NATR: https://ta-lib.org/functions/natr.html

    Shares similar logic:
    - pandas-ta-classic, natr: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/natr.py#L20
    - Tulip Indicators, natr: https://tulipindicators.org/natr

    Notes:
        - The ATR is `get_average_true_range` (the StockCharts seeding), so the NATR starts
          one period before TA-Lib's and equals it, divided by 100, once the seeds have
          converged.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the Average True Range.

    Returns:
        pd.Series: NATR values as decimals.
    """
    average_true_range = get_average_true_range(
        prices_high, prices_low, prices_close, window
    )

    return average_true_range / prices_close


def get_average_day_range(
    prices_high: pd.Series, prices_low: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Average Day Range (ADR): the average high-low range over a window, how much the price usually
    moves within a period.

    The Average Day Range is the simple average of the High - Low range over `window`
    periods. Unlike the Average True Range it ignores gaps between periods, so it measures
    the range traded within the periods only.

    The formula is as follows:

    - ADR = SMA(High - Low, window)

    Also known as: ADR, average daily range, average high-low range.

    Reference: a common practitioner measure without a single primary source; the same
    average range is the width of Keltner's original channel (Keltner, C.W. (1960). "How
    to Make Money in Commodities." The Keltner Statistical Service).

    Referenced against:
    - TA-Lib, ADR: https://ta-lib.org/functions/adr.html

    Notes:
        - Equals TA-Lib's ADR.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods of the average.

    Returns:
        pd.Series: ADR values in price units.
    """
    return (prices_high - prices_low).rolling(window=window).mean()


def get_chaikin_volatility(
    prices_high: pd.Series, prices_low: pd.Series, window: int, roc_window: int
) -> pd.Series:
    """
    Calculate Chaikin's Volatility: the percentage change of the average high-low range over a window, which shows
    whether the range is widening or narrowing.

    Chaikin's Volatility measures how fast the trading range widens or narrows: it is the
    rate of change of an Exponential Moving Average of the High - Low range. Chaikin read a
    sharp rise in a short time as a sign of an approaching bottom (panic selling) and a
    long decline in volatility as typical of a maturing top.

    The formula is as follows:

    - Range Average = EMA(High - Low, window)
    - Chaikin Volatility = Range Average / Range Average(t - roc_window) - 1

    The result is a decimal (0.25 is a range average 25% wider than `roc_window` periods
    ago); TA-Lib, Tulip Indicators and pandas-ta-classic multiply it by 100.

    Also known as: CV, CVI, Chaikin Volatility Indicator.

    Reference: Chaikin, M.; described in Achelis, S.B. (2000). "Technical Analysis from A
    to Z." 2nd ed. McGraw-Hill.

    Referenced against:
    - TA-Lib, CVI: https://ta-lib.org/functions/cvi.html

    Shares similar logic:
    - pandas-ta-classic, cvi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/cvi.py#L12
    - Tulip Indicators, cvi: https://tulipindicators.org/cvi

    Notes:
        - The EMA is seeded with the Simple Moving Average of its first `window` ranges, as
          TA-Lib does, so this agrees with TA-Lib's CVI divided by 100. pandas-ta-classic seeds
          its EMA differently and converges to the same values.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods of the EMA of the range.
        roc_window (int): Number of periods the rate of change looks back.

    Returns:
        pd.Series: Chaikin Volatility as decimals.
    """
    range_average = get_exponential_moving_average(prices_high - prices_low, window)
    previous_average = range_average.shift(roc_window)

    return range_average / previous_average.where(previous_average != 0) - 1


def get_relative_volatility_index(
    prices: pd.Series, window: int, std_window: int
) -> pd.Series:
    """
    Calculate Dorsey's Relative Volatility Index (RVI): the standard deviation of the up periods against that of the
    down periods, from 0 to 100, which shows the direction of volatility.

    The Relative Volatility Index is built like the Relative Strength Index, but measures
    the direction of volatility instead of price changes: on a period that closes higher,
    the standard deviation of the last `std_window` prices counts as upward volatility,
    and on a period that closes lower as downward volatility. Dorsey designed it as a
    confirming filter for other indicators: above 50 favours long trades and below 50
    short trades.

    The formula is as follows:

    - Standard Deviation = StdDev(Price, std_window)
    - Up = Standard Deviation if Price(t) > Price(t-1), else 0
    - Down = Standard Deviation if Price(t) < Price(t-1), else 0
    - RVI = 100 * Wilder(Up, window) / (Wilder(Up, window) + Wilder(Down, window))

    Wilder(x, window) is Wilder's smoothing (`get_wilder_moving_average`), seeded with the
    simple average of the first `window` values. The RVI is on Dorsey's 0 to 100 scale.

    Also known as: RVI, Relative Volatility Index.

    Reference: Dorsey, D. (1993). "The Relative Volatility Index." Technical Analysis of
    Stocks & Commodities, 11(6). The refined version, applied to the highs and lows, is in
    Dorsey, D. (1995). "Refining the Relative Volatility Index." Technical Analysis of
    Stocks & Commodities, 13(9).

    Referenced against:
    - TA-Lib, RVI: https://ta-lib.org/functions/rvi.html
    - TA-Lib, RVIR: https://ta-lib.org/functions/rvir.html

    Shares similar logic:
    - pandas-ta-classic, rvi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/rvi.py#L61

    Notes:
        - Equals TA-Lib's RVI on the close and, averaged over the highs and lows, TA-Lib's
          RVIR. pandas-ta-classic's rvi smooths with an EMA by default, which gives
          different values.
        - The standard deviation is the sample one (ddof=1); the RVI is a ratio of the
          deviations, so the convention does not change it.
        - A window in which the price did not move has no RVI (0 / 0, missing) instead of 0.

    Args:
        prices (pd.Series): Series of prices: the close for the RVI, or the high and the
            low for the two halves of the refined RVI.
        window (int): Number of periods of Wilder's smoothing.
        std_window (int): Number of periods of the standard deviation.

    Returns:
        pd.Series: RVI values between 0 and 100.
    """
    standard_deviation = prices.rolling(window=std_window).std()
    price_change = prices.diff()
    has_value = standard_deviation.notna() & price_change.notna()

    upward = standard_deviation.where(price_change > 0, 0.0).where(has_value)
    downward = standard_deviation.where(price_change < 0, 0.0).where(has_value)

    upward_average = get_wilder_moving_average(upward, window)
    downward_average = get_wilder_moving_average(downward, window)
    total = upward_average + downward_average

    return 100 * upward_average / total.where(total != 0)


def get_acceleration_bands(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Calculate Headley's Acceleration Bands: bands around a moving average that widen with the range of the candles,
    whose breakouts show an accelerating trend.

    Acceleration Bands place an envelope around a simple average of the close whose width
    grows with the relative High - Low range of each period. Headley used a close outside
    the bands as the start of an accelerating move (a breakout) and a close back inside
    them as the end of it.

    The formula is as follows:

    - Range Ratio = multiplier * (High - Low) / (High + Low)
    - Upper Band = SMA(High * (1 + Range Ratio), window)
    - Middle Band = SMA(Close, window)
    - Lower Band = SMA(Low * (1 - Range Ratio), window)

    Also known as: ACCBANDS, Price Headley's Acceleration Bands.

    Reference: Headley, P. (2002). "Big Trends in Trading: Strategies to Master Major Market
    Moves." John Wiley & Sons.

    Referenced against:
    - TA-Lib, ACCBANDS: https://ta-lib.org/functions/accbands.html
    - pandas-ta-classic, accbands: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/accbands.py#L17

    Notes:
        - Equals TA-Lib's ACCBANDS (multiplier 4) and pandas-ta-classic's accbands.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the simple averages.
        multiplier (float): Multiplier of the relative range, 4 in Headley's definition.

    Returns:
        pd.DataFrame: Upper Band, Middle Band and Lower Band in price units.
    """
    range_ratio = multiplier * (prices_high - prices_low) / (prices_high + prices_low)

    upper_band = (prices_high * (1 + range_ratio)).rolling(window=window).mean()
    middle_band = prices_close.rolling(window=window).mean()
    lower_band = (prices_low * (1 - range_ratio)).rolling(window=window).mean()

    return pd.concat(
        [upper_band, middle_band, lower_band],
        keys=["Upper Band", "Middle Band", "Lower Band"],
        axis=1,
    )


def get_chande_kroll_stop(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    multiplier: float,
    stop_window: int,
) -> pd.DataFrame:
    """
    Calculate the Chande Kroll Stop: a long and a short stop a multiple of the Average True Range from the recent
    extremes, which shows where a trend would be broken.

    The Chande Kroll Stop is a pair of trailing stops a multiple of the Average True Range
    from the extreme price of the last `window` periods, smoothed a second time by taking
    the extreme of those first stops over `stop_window` periods. The high stop trails below
    the highs and the low stop above the lows; a close below the high stop exits a long
    position and a close above the low stop a short one, and the order of the two lines
    shows the trend.

    The formula is as follows:

    - First High Stop = Highest High(window) - multiplier * ATR(window)
    - First Low Stop = Lowest Low(window) + multiplier * ATR(window)
    - High Stop = Highest(First High Stop, stop_window)
    - Low Stop = Lowest(First Low Stop, stop_window)

    Chande and Kroll call the high stop the "stop short" and the low stop the "stop long".

    Also known as: CKSP, Chande Kroll stop loss.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." John Wiley &
    Sons.

    Referenced against:
    - TA-Lib, CKSP: https://ta-lib.org/functions/cksp.html

    Shares similar logic:
    - pandas-ta-classic, cksp: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/cksp.py#L12

    Notes:
        - The ATR is `get_average_true_range` (the StockCharts seeding), so the stops start
          one period before TA-Lib's CKSP and equal them once the seeds have converged.
        - pandas-ta-classic's cksp in its TradingView mode uses the same defaults (10, 1, 9);
          otherwise it uses a simple average of the True Range with 3 ATRs and
          20 periods.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the highest high, lowest low and ATR.
        multiplier (float): Number of ATRs the first stops sit from the extremes.
        stop_window (int): Number of periods over which the first stops are smoothed.

    Returns:
        pd.DataFrame: High Stop and Low Stop in price units.
    """
    average_true_range = get_average_true_range(
        prices_high, prices_low, prices_close, window
    )
    first_high_stop = (
        prices_high.rolling(window=window).max() - multiplier * average_true_range
    )
    first_low_stop = (
        prices_low.rolling(window=window).min() + multiplier * average_true_range
    )

    return pd.concat(
        [
            first_high_stop.rolling(window=stop_window).max(),
            first_low_stop.rolling(window=stop_window).min(),
        ],
        keys=["High Stop", "Low Stop"],
        axis=1,
    )


def get_adaptive_price_zone(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Calculate Leibfarth's Adaptive Price Zone (APZ): bands around a double-smoothed exponential moving average that
    adapt to the range, which show turning points in a sideways market.

    The Adaptive Price Zone is a band around a fast, double-smoothed Exponential Moving
    Average of the close, with a width that is the same double-smoothed average of the
    High - Low range. Both averages use the square root of `window` as their period, so
    the zone reacts quickly. Leibfarth designed it for markets without a trend (an ADX
    below 30): a close above the zone is a chance to sell, and one below it to buy, in
    anticipation of a reversal.

    The formula is as follows:

    - Smoothing Period = sqrt(window), with alpha = 2 / (sqrt(window) + 1)
    - Middle Band = EMA(EMA(Close, Smoothing Period), Smoothing Period)
    - Range = EMA(EMA(High - Low, Smoothing Period), Smoothing Period)
    - Upper Band = Middle Band + multiplier * Range
    - Lower Band = Middle Band - multiplier * Range

    Also known as: APZ, Adaptive Price Zone.

    Reference: Leibfarth, L. (2006). "Trading With An Adaptive Price Zone." Technical
    Analysis of Stocks & Commodities, 24(9).

    Shares similar logic:
    - finta, APZ: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1907

    Notes:
        - Each EMA is seeded with the simple average of its first ceil(sqrt(window))
          values, the Toolkit's seeding of exponential averages, and then runs with the
          exact smoothing constant of the square-root period, as Leibfarth's formula does.
        - finta's APZ differs by design: it smooths over the full `window` instead of its
          square root and centres the zone on a Double Exponential Moving Average
          (2 * EMA - EMA of EMA) instead of the EMA of the EMA.
        - The article's examples use 20 to 30 periods with 1.4 to 2.1 ranges; the defaults
          of the Toolkit are 20 periods and 2 ranges.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods whose square root is the smoothing period.
        multiplier (float): Number of smoothed ranges between the middle and outer bands.

    Returns:
        pd.DataFrame: Upper Band, Middle Band and Lower Band in price units.
    """
    smoothing_period = math.sqrt(window)
    seed_window = math.ceil(smoothing_period)

    smoothed = {}
    for name, series in (
        ("close", prices_close),
        ("range", prices_high - prices_low),
    ):
        values = series
        for _ in range(2):
            # The seed is the simple average of the first full window, after which ewm
            # with adjust=False runs the recursion with the square-root period's constant.
            seed = values.rolling(window=seed_window).mean()
            has_seed = seed.notna().astype(bool).cummax()
            seed_period = has_seed & ~has_seed.shift(1, fill_value=False).astype(bool)
            values = (
                values.where(has_seed)
                .mask(seed_period, seed)
                .ewm(span=smoothing_period, adjust=False, ignore_na=True)
                .mean()
                .where(has_seed & values.notna())
            )
        smoothed[name] = values

    middle_band = smoothed["close"]
    upper_band = middle_band + multiplier * smoothed["range"]
    lower_band = middle_band - multiplier * smoothed["range"]

    return pd.concat(
        [upper_band, middle_band, lower_band],
        keys=["Upper Band", "Middle Band", "Lower Band"],
        axis=1,
    )


def get_aberration(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    atr_window: int,
) -> pd.DataFrame:
    """
    Calculate the Aberration bands: bands one Average True Range above and below a short average of the typical
    price, whose breakouts are the signals of Fitschen's Aberration system.

    The Aberration bands place one Average True Range above and below a short simple
    average of the typical price. They come from Keith Fitschen's Aberration trading
    system, a breakout system that buys a close above the upper band and sells a close
    below the lower band.

    The formula is as follows:

    - Typical Price = (High + Low + Close) / 3
    - Middle Band = SMA(Typical Price, window)
    - Upper Band = Middle Band + ATR(atr_window)
    - Lower Band = Middle Band - ATR(atr_window)

    Also known as: ABER, Aberration.

    Reference: attributed to Fitschen, K. (1993), the Aberration trading system; the bands
    are pandas-ta-classic's definition, which notes that no definitive source of the formula
    exists. Fitschen describes his breakout systems in Fitschen, K. (2013).
    "Building Reliable Trading Systems." John Wiley & Sons.

    Referenced against:
    - pandas-ta-classic, aberration: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/aberration.py#L14

    Notes:
        - The ATR is `get_average_true_range` (Wilder's smoothing with the StockCharts
          seeding); pandas-ta-classic seeds its ATR differently and converges to the same
          bands.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the simple average of the typical price.
        atr_window (int): Number of periods of the Average True Range.

    Returns:
        pd.DataFrame: Upper Band, Middle Band and Lower Band in price units.
    """
    typical_price = (prices_high + prices_low + prices_close) / 3
    middle_band = typical_price.rolling(window=window).mean()
    average_true_range = get_average_true_range(
        prices_high, prices_low, prices_close, atr_window
    )

    return pd.concat(
        [
            middle_band + average_true_range,
            middle_band,
            middle_band - average_true_range,
        ],
        keys=["Upper Band", "Middle Band", "Lower Band"],
        axis=1,
    )


def get_profit_maximizer(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    atr_window: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Calculate the Profit Maximizer (PMax): a trailing stop a multiple of the Average True Range from an exponential
    moving average of the median price, which shows the trend and flips less often than the Supertrend.

    The Profit Maximizer applies the Supertrend's trailing logic to a moving average instead
    of the price: two stops a multiple of the Average True Range below and above an
    Exponential Moving Average of the median price, each of which only tightens while the
    average stays on its side. The trend turns down when the average crosses below the long
    stop and up when it crosses above the short stop, and PMax is the active stop. Because
    the average is smoother than the price, it flips less often than the Supertrend.

    The formula is as follows:

    - Average = EMA((High + Low) / 2, window)
    - Long Stop(t) = Average - multiplier * ATR(atr_window), raised to Long Stop(t-1) while
      Average(t) > Long Stop(t-1)
    - Short Stop(t) = Average + multiplier * ATR(atr_window), lowered to Short Stop(t-1)
      while Average(t) < Short Stop(t-1)
    - The trend turns down when Average(t) < Long Stop(t-1) and up when Average(t) >
      Short Stop(t-1)
    - PMax = Long Stop in an uptrend, Short Stop in a downtrend

    Also known as: PMax, Profit Maximizer.

    Reference: Ozbilgic, K. (2020). "Profit Maximizer - PMax." TradingView script by
    KivancOzbilgic, a practitioner indicator without a published source.

    Shares similar logic:
    - pandas-ta-classic, pmax: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/pmax.py#L51

    Notes:
        - This follows Ozbilgic's script: the EMA (seeded with a simple average) of the
          median price, a 10-period ATR with Wilder's smoothing and 3 ATRs, starting in an
          uptrend on the first period with an average and an ATR.
        - pandas-ta-classic's pmax differs by design: it averages the close and turns the
          trend on the close crossing the previous stops instead of the average.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the EMA of the median price.
        atr_window (int): Number of periods of the Average True Range.
        multiplier (float): Number of ATRs between the average and the stops.

    Returns:
        pd.DataFrame: PMax (the active stop, in price units) and Trend Direction (1 for an
            uptrend, -1 for a downtrend).
    """
    observed = prices_high.notna() & prices_low.notna() & prices_close.notna()
    high, low, close = (
        prices_high[observed],
        prices_low[observed],
        prices_close[observed],
    )

    average = get_exponential_moving_average((high + low) / 2, window)
    average_true_range = get_average_true_range(high, low, close, atr_window)

    averages = average.to_numpy(dtype="float64").tolist()
    basic_long = (average - multiplier * average_true_range).to_numpy(dtype="float64")
    basic_short = (average + multiplier * average_true_range).to_numpy(dtype="float64")

    length = len(averages)
    profit_maximizer = [math.nan] * length
    direction = [math.nan] * length
    long_stop = short_stop = trend = math.nan

    # The recursion runs on plain Python floats, since writing a pandas Series one element
    # at a time costs tens of microseconds per element.
    for i, (mean, new_long, new_short) in enumerate(
        zip(averages, basic_long.tolist(), basic_short.tolist())
    ):
        if math.isnan(new_long) or math.isnan(new_short):
            continue

        if math.isnan(trend):
            long_stop, short_stop, trend = new_long, new_short, 1.0
        else:
            previous_long, previous_short = long_stop, short_stop
            long_stop = (
                max(new_long, previous_long) if mean > previous_long else new_long
            )
            short_stop = (
                min(new_short, previous_short) if mean < previous_short else new_short
            )
            if trend == -1 and mean > previous_short:
                trend = 1.0
            elif trend == 1 and mean < previous_long:
                trend = -1.0

        profit_maximizer[i] = long_stop if trend == 1 else short_stop
        direction[i] = trend

    return pd.DataFrame(
        {"PMax": profit_maximizer, "Trend Direction": direction}, index=average.index
    ).reindex(prices_high.index)


def get_price_distance(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.Series:
    """
    Calculate the Price Distance: the distance the price travelled within each period, including the gap from the
    previous close, a measure of how much the price moves.

    The Price Distance estimates the distance the price travelled within a period: twice
    the High - Low range (down to the low and up to the high, or the reverse), less the
    open-to-close body that the range already contains once, plus the gap from the
    previous close to the open.

    The formula is as follows:

    - Price Distance = 2 * (High - Low) - |Close - Open| + |Open - Close(t-1)|

    Also known as: PDIST, price distance volatility.

    Reference: ProRealCode, "Price Distance" indicator, a practitioner measure without a
    published source.

    Referenced against:
    - pandas-ta-classic, pdist: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/pdist.py#L17

    Notes:
        - Equals pandas-ta-classic's pdist; the first period has no previous close and no
          Price Distance.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: Price Distance values in price units.
    """
    return (
        2 * (prices_high - prices_low)
        - (prices_close - prices_open).abs()
        + (prices_open - prices_close.shift(1)).abs()
    )


def get_market_thermometer(
    prices_high: pd.Series, prices_low: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate Elder's Market Thermometer: how far each period's range pushes outside that of the previous period,
    which shows how hot or quiet the market is.

    The Market Thermometer measures how far each period's range pushes outside the previous
    period's: the larger of the rise of the high above the previous high and the fall of the
    low below the previous low, or 0 for an inside period. Elder compares it with its
    Exponential Moving Average: a reading far above the average marks a burst of volatility
    in which to take profits, and several readings below it a quiet market that often
    precedes an explosive move.

    The formula is as follows:

    - Thermometer = Max(High - High(t-1), Low(t-1) - Low, 0)
    - Thermometer Average = EMA(Thermometer, window)

    Also known as: THERMO, Elder's thermometer, market temperature.

    Reference: Elder, A. (2002). "Come Into My Trading Room." John Wiley & Sons, which
    uses a 22-day EMA.

    Shares similar logic:
    - pandas-ta-classic, thermo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/thermo.py#L18

    Notes:
        - Only a higher high or a lower low raises the temperature, and an inside period (a
          lower high and a higher low) has a temperature of 0, as Elder defines it.
          pandas-ta-classic's thermo takes the absolute values of both changes instead, so a
          falling high or a rising low also counts; the two agree on periods with a higher
          high and a lower low. It also defaults to 20 periods and adds long and short flags.
        - The EMA is seeded with the simple average of its first `window` values.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods of the EMA.

    Returns:
        pd.DataFrame: Thermometer and Thermometer Average in price units.
    """
    high_change = prices_high - prices_high.shift(1)
    low_change = prices_low.shift(1) - prices_low
    thermometer = (
        pd.concat([high_change, low_change], axis=1).max(axis=1).clip(lower=0)
    ).where(high_change.notna() & low_change.notna())

    return pd.concat(
        [thermometer, get_exponential_moving_average(thermometer, window)],
        keys=["Thermometer", "Thermometer Average"],
        axis=1,
    )


def get_squeeze_momentum(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    num_std_dev: float,
    keltner_multipliers: list[float],
) -> pd.DataFrame:
    """
    Calculate the Squeeze Momentum (TTM Squeeze): whether the Bollinger Bands lie inside the Keltner Channels (a
    squeeze of volatility) and the momentum that shows the direction of the breakout.

    The squeeze compares two volatility bands around the same simple average of the close:
    Bollinger Bands, which widen and narrow with the standard deviation, and a Keltner
    Channel, which follows the average True Range. When the Bollinger Bands contract inside
    the Keltner Channel, volatility is unusually low and the market is "in a squeeze"; Carter
    trades the release of that squeeze in the direction of the momentum. The momentum is
    LazyBear's: the linear regression end point of the close relative to the middle of its
    recent range and its average.

    The formula is as follows:

    - Middle = SMA(Close, window)
    - Bollinger Bands = Middle +/- num_std_dev * StdDev(Close, window)
    - Keltner Channel(m) = Middle +/- m * SMA(True Range, window)
    - Squeeze = the number of `keltner_multipliers` m for which the Bollinger Bands lie
      inside Keltner Channel(m)
    - Momentum = LinReg(Close - ((Highest High(window) + Lowest Low(window)) / 2 + Middle) / 2,
      window), the end point of the least-squares line (`statistics_model.get_linear_regression`)

    With the default multipliers 2, 1.5 and 1 (the TTM Squeeze Pro's wide, normal and narrow
    channels), the Squeeze is 0 without a squeeze and 1, 2 or 3 for a low, normal or high
    compression; Carter's original squeeze (Keltner at 1.5) is on when it is 2 or more.

    Also known as: TTM Squeeze, Squeeze Pro, SQZ, SQZMOM, squeeze momentum indicator.

    Reference: Carter, J.F. (2005). "Mastering the Trade." McGraw-Hill, chapter 11 (the TTM
    Squeeze). The momentum follows LazyBear's "Squeeze Momentum Indicator" TradingView
    script (2014).

    Referenced against:
    - TA-Lib, SQZMOM: https://ta-lib.org/functions/sqzmom.html
    - pandas-ta-classic, squeeze: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/squeeze.py#L94

    Shares similar logic:
    - pandas-ta-classic, squeeze_pro: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/squeeze_pro.py#L17
    - finta, SQZMI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1941

    Notes:
        - The momentum agrees with TA-Lib's SQZMOM and pandas-ta-classic's
          `squeeze` in LazyBear's form. The Squeeze agrees with TA-Lib's squeeze output with 0 instead
          of -1 for no squeeze; its Keltner Channel is LazyBear's simple average of the True
          Range, not `get_keltner_channels`.
        - The Bollinger standard deviation is the population one (ddof=0), as in
          `get_bollinger_bands`. The Squeeze starts on period `window` - 1 (the True Range
          of the first period is High - Low); TA-Lib reports its squeeze output from the first
          Momentum, on period 2 * (`window` - 1), and pandas-ta-classic one period later.
        - pandas-ta-classic's squeeze without `lazybear` smooths a 12-period momentum
          instead, and finta's SQZMI returns only whether the squeeze is on.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the bands, channels, range and regression.
        num_std_dev (float): Number of standard deviations of the Bollinger Bands.
        keltner_multipliers (list[float]): The widths, in average True Ranges, of the
            Keltner Channels the Bollinger Bands are compared with.

    Returns:
        pd.DataFrame: Momentum (in price units) and Squeeze (the number of Keltner Channels
            the Bollinger Bands lie inside).
    """
    middle = prices_close.rolling(window=window).mean()
    bollinger_width = num_std_dev * prices_close.rolling(window=window).std(ddof=0)
    average_range = (
        get_true_range(prices_high, prices_low, prices_close)
        .where(prices_close.notna())
        .rolling(window=window)
        .mean()
    )

    squeeze = sum(
        (bollinger_width < multiplier * average_range).astype(float)
        for multiplier in keltner_multipliers
    ).where(bollinger_width.notna() & average_range.notna())

    range_middle = (
        prices_high.rolling(window=window).max()
        + prices_low.rolling(window=window).min()
    ) / 2
    momentum = get_linear_regression(
        prices_close - (range_middle + middle) / 2, window
    )["Linear Regression"]

    return pd.concat([momentum, squeeze], keys=["Momentum", "Squeeze"], axis=1)


def get_standard_deviation_channel(
    prices: pd.Series,
    window: int | None = None,
    num_std_dev: float | list[float] | None = None,
) -> pd.DataFrame:
    """
    Calculate the Standard Deviation Channel: a least-squares line through the prices with bands a number of
    standard deviations above and below it.

    The line runs through all prices of the period, or the last `window` of them.

    The channel shows the trend of the whole period and how far the prices stray from it: a
    price near an outer band lies far from the trend of the period. It is the thinkorswim
    study StDevAll, which draws the regression line of all bars (InertiaAll) with bands at
    1, 2 and 3 standard deviations.

    The formula is as follows:

    - Linear Regression = b0 + b1 * x, the least-squares line through the prices, x = 0, ..., n - 1
    - Upper Band k = Linear Regression + k * StdDev(Close)
    - Lower Band k = Linear Regression - k * StdDev(Close)

    Also known as: standard deviation channel, StDevAll, linear regression channel, regression
    bands.

    Reference: thinkorswim (TD Ameritrade). thinkScript studies StDevAll and InertiaAll.

    Referenced against:
    - pandas-ta-classic, tos_stdevall: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/tos_stdevall.py#L15

    Notes:
        - Equals pandas-ta-classic's tos_stdevall, with the sample standard deviation of
          the prices (ddof=1).
        - The line is fitted over the whole period at once, so every value depends on the
          later prices of the period and the channel is redrawn as prices are added. It
          describes a period, and is not a signal that was known at the time.
        - Missing prices are left out of the fit and stay missing.

    Args:
        prices (pd.Series): Series of prices.
        window (int | None): The number of latest periods the line is fitted over, at
            least 2. Defaults to None, which uses all prices.
        num_std_dev (float | list[float] | None): The number or numbers of standard
            deviations of the bands. Defaults to None, which uses 1, 2 and 3.

    Returns:
        pd.DataFrame: Linear Regression with a Lower Band k and Upper Band k per number of
        standard deviations k.
    """
    if num_std_dev is None:
        num_std_dev = [1, 2, 3]
    elif not isinstance(num_std_dev, list | tuple):
        num_std_dev = [num_std_dev]

    fitted = prices.dropna()
    if window is not None:
        if window < 2:  # noqa: PLR2004
            raise ValueError("The window of a linear regression is at least 2 periods.")
        fitted = fitted.iloc[-window:]

    channel = pd.DataFrame(index=prices.index)
    if len(fitted) < 2:  # noqa: PLR2004
        channel["Linear Regression"] = np.nan
        for deviations in num_std_dev:
            channel[f"Lower Band {deviations:g}"] = np.nan
            channel[f"Upper Band {deviations:g}"] = np.nan
        return channel

    time = np.arange(len(fitted))
    slope, intercept = np.polyfit(time, fitted.to_numpy(dtype=float), 1)
    line = pd.Series(intercept + slope * time, index=fitted.index)
    standard_deviation = fitted.std(ddof=1)

    channel["Linear Regression"] = line
    for deviations in num_std_dev:
        channel[f"Lower Band {deviations:g}"] = line - deviations * standard_deviation
        channel[f"Upper Band {deviations:g}"] = line + deviations * standard_deviation

    return channel


def get_signal_to_noise_ratio(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate the Signal-to-Noise Ratio (SNR): the net move of the close over a window divided by the Average True
    Range, which shows how cleanly the price trends.

    The Signal-to-Noise Ratio divides the net move of the close over the window (the signal)
    by the Average True Range over the same window (the noise). A ratio of 5 means the price
    moved five times its average range in `window` periods: a market that trends cleanly,
    where a ratio near 0 is one that moved a lot but went nowhere. Skeggs and Hill use it to
    measure how much trend there is for a trend follower to capture.

    The formula is as follows:

    - Signal = |Close(t) - Close(t - window)|
    - Noise = ATR(window), Wilder's Smoothed Moving Average of the True Range
    - SNR = Signal / Noise

    Also known as: SNR, signal to noise ratio, trend strength ratio.

    Reference: Skeggs, J. and Hill, A. (2015). "Back in Black Part 2: The Opportunity Set for
    Trend Following." Man AHL.

    Referenced against:
    - TTR (R), SNR: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/SNR.R#L47

    Notes:
        - The ATR is `get_average_true_range`, whose first True Range is High - Low (the
          StockCharts convention). TTR's ATR leaves the first True Range missing and starts
          one period later, so the two ratios differ until the seeds of the averages have
          converged (within 0.01% after about 100 periods with a 14-period window).
        - A window without any range (an ATR of 0) has no ratio.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods of the move and of the Average True Range.

    Returns:
        pd.Series: Signal-to-Noise Ratio values, in multiples of the Average True Range.
    """
    average_true_range = get_average_true_range(
        prices_high, prices_low, prices_close, window
    )
    signal = (prices_close - prices_close.shift(window)).abs()

    return signal / average_true_range.where(average_true_range != 0)


def get_price_bands(
    prices: pd.Series,
    window: int = 20,
    num_std_dev: float = 2,
    short_window: int = 2,
    moving_average: str = "simple",
) -> pd.DataFrame:
    """
    Calculate the Price Bands: bands around a moving average whose width comes from its distance to a fast moving
    average of the price, so they react to sustained moves rather than single spikes.

    Price Bands are a variant of the Bollinger Bands whose width comes from the distance between
    the moving average and a fast moving average of the price, rather than from the distance
    between the price and the moving average. The fast average filters out the noise of single
    periods, so a spike widens the bands less than it widens the Bollinger Bands, and the bands
    react to sustained moves away from the average.

    The formula is as follows:

    - Middle Band = SMA(Close, window), or another moving average of `moving_average`
    - Deviation = Middle Band - SMA(Close, short_window)
    - Width = Population Standard Deviation(Deviation, window)
    - Upper Band = Middle Band + num_std_dev * Width
    - Lower Band = Middle Band - num_std_dev * Width

    Also known as: PBands, price bands, smoothed Bollinger Bands.

    Reference: the measure is defined by Brian G. Peterson in TTR's PBands, after the bands of
    Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.

    Referenced against:
    - TTR (R), PBands: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/priceBands.R#L68

    Notes:
        - The standard deviation is the population one (dividing by `window`), as in TTR's
          PBands and in the Bollinger Bands.
        - With `short_window` = 1 the fast average is the price itself, and the width is the
          standard deviation of the price around a moving average, not around its own mean.
        - TTR's PBands can also centre the bands (`centered=TRUE`), which adds the moving
          average of the deviation divided by the width to the middle band. That adds a number
          of standard deviations to a price, so the shift does not scale with the price, and
          the Toolkit leaves it out.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the moving average and the standard deviation.
            Defaults to 20.
        num_std_dev (float): Number of standard deviations of the bands. Defaults to 2.
        short_window (int): Number of periods of the fast moving average. Defaults to 2.
        moving_average (str): The moving average of both averages, one of `MOVING_AVERAGES` (the window-only
            methods of `get_moving_average`). Options are: "simple" (Simple Moving Average),
            "exponential" (Exponential Moving Average), "double_exponential" (Double Exponential Moving
            Average), "triple_exponential" (Triple Exponential Moving Average), "zero_lag" (Zero Lag Exponential
            Moving Average), "wilder" (Wilder's Moving Average), "triangular" (Triangular Moving Average),
            "weighted" (Weighted Moving Average), "hull" (Hull Moving Average), "sine" (Sine Weighted Moving
            Average), "fibonacci" (Fibonacci Weighted Moving Average), "pascal" (Pascal Weighted Moving
            Average). Defaults to "simple".

    Returns:
        pd.DataFrame: Price Bands (upper, middle, lower).

    Raises:
        ValueError: If `moving_average` is not one of `MOVING_AVERAGES`.
    """
    check_option("moving_average", moving_average, MOVING_AVERAGES)

    middle_band = MOVING_AVERAGES[moving_average](prices, window)
    fast_average = MOVING_AVERAGES[moving_average](prices, short_window)
    width = (middle_band - fast_average).rolling(window=window).std(ddof=0)

    return pd.concat(
        [
            middle_band + num_std_dev * width,
            middle_band,
            middle_band - num_std_dev * width,
        ],
        axis=1,
        keys=["Upper Band", "Middle Band", "Lower Band"],
    )
