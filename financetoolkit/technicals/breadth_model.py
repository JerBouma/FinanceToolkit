"""Breadth Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.technicals.overlap_model import get_exponential_moving_average


def get_mcclellan_oscillator(
    prices_close: pd.Series, short_ema_window: int, long_ema_window: int
) -> pd.Series:
    """
    Calculate the McClellan Oscillator for a given price series.

    The McClellan Oscillator is traditionally a market-wide breadth indicator, calculated
    from the daily count of advancing minus declining issues across an entire exchange or
    index. It is the difference between a short-term and a long-term Exponential Moving
    Average of that daily advance-decline figure, with the classic parameters (19 and 39
    periods) chosen to approximate 10% and 5% trend weightings in Sherman McClellan's
    original per-day smoothing constants.

    The formula is as follows:

    - McClellan Oscillator = EMA(Advancers - Decliners, short_ema_window) - EMA(Advancers -
      Decliners, long_ema_window)

    Also known as: McClellan Oscillator.

    Reference: McClellan, S. and McClellan, M. (1970). "Patterns for Profit: The McClellan
    Oscillator and Summation Index." Trade Levels Inc.

    Notes:
        - Both EMAs are seeded with the Simple Moving Average of their first values (see
          `overlap_model.get_exponential_moving_average`), so the oscillator starts once the
          long EMA has `long_ema_window` daily signals, which begin on the second period.
        - This implementation is applied to a single security's own daily up/down signal
          (see `get_advancers_decliners`) rather than to a true cross-sectional count of
          advancing versus declining issues across a market universe, which is how the
          indicator was originally defined and is conventionally used. Treat the result as a
          smoothed measure of that single security's own directional persistence rather than
          broad market breadth.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_ema_window (int): Number of periods for the short-term EMA.
        long_ema_window (int): Number of periods for the long-term EMA.

    Returns:
        pd.Series: McClellan Oscillator values.
    """
    # The first price has no previous close and therefore no advance or decline; leaving
    # it missing keeps a zero that never happened out of the seed of both EMAs.
    advancers_decliners = get_advancers_decliners(prices_close).where(
        prices_close.shift(1).notna()
    )
    short_ema = get_exponential_moving_average(advancers_decliners, short_ema_window)
    long_ema = get_exponential_moving_average(advancers_decliners, long_ema_window)

    return short_ema - long_ema


def get_advancers_decliners(
    prices_close: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate a daily advance/decline signal for a given price series.

    Traditionally, "advancers minus decliners" is a market-wide breadth statistic: the count
    of stocks that rose minus the count of stocks that fell across an entire exchange or
    index on a given day. Applied here to a single security's own price series, it instead
    yields a simple directional signal for that security: +1 on a day it closed higher than
    the prior day, -1 on a day it closed lower, and 0 on an unchanged day.

    The formula is as follows:

    - Signal = +1 if Close(t) > Close(t-1), -1 if Close(t) < Close(t-1), else 0

    Also known as: advance/decline signal, market breadth (in its traditional, cross-
    sectional form).

    Notes:
        - This is not the traditional cross-sectional advance/decline statistic (which
          requires the full universe of constituent prices, as used by `get_trin` and
          `get_new_highs_new_lows` elsewhere in this module); it is a per-security proxy
          intended to feed `get_mcclellan_oscillator` for a single ticker.

    Args:
        prices_close (pd.Series | pd.DataFrame): Closing prices, one column per ticker
            when a DataFrame is given.

    Returns:
        pd.Series | pd.DataFrame: Advance/decline signal values (+1, 0 or -1), in the
            shape of the input.
    """
    price_change = prices_close.diff(1)

    # apply() hands a ufunc straight to pandas, which keeps the Series or DataFrame
    # (np.sign called directly is typed as returning a bare array).
    return price_change.apply(np.sign).fillna(0)


def get_on_balance_volume(prices_close: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV) of a given price series.

    On-Balance Volume is a running total of volume that adds the period's volume when price
    closes higher than the prior period, subtracts it when price closes lower, and leaves the
    running total unchanged on a flat close. Granville's premise was that volume tends to
    lead price: a rising OBV alongside flat or falling price signals accumulation that may
    precede a price breakout, and vice versa for a falling OBV.

    The formula is as follows:

    - OBV(t) = OBV(t-1) + Volume(t), if Close(t) > Close(t-1)
    - OBV(t) = OBV(t-1) - Volume(t), if Close(t) < Close(t-1)
    - OBV(t) = OBV(t-1), if Close(t) = Close(t-1)

    Also known as: OBV.

    Reference: Granville, J.E. (1963). "Granville's New Key to Stock Market Profits."
    Prentice-Hall.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.

    Returns:
        pd.Series: OBV values.
    """
    price_diff = prices_close.diff(1)
    direction = price_diff.apply(np.sign).fillna(0)
    obv = direction * volumes

    return obv.cumsum()


def _get_money_flow_multiplier(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series
) -> pd.Series:
    """
    Calculate the Money Flow Multiplier shared by every Accumulation/Distribution based
    indicator in this module (the Accumulation/Distribution Line, the Chaikin Oscillator
    and the Chaikin Money Flow).

    The Money Flow Multiplier maps where the close settled within the period's high-low
    range onto a scale from -1 (closed at the low) to +1 (closed at the high), which is then
    used to weight that period's volume as either buying or selling pressure.

    The formula is as follows:

    - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: Money Flow Multiplier values, bounded between -1 and 1.
    """
    return ((prices_close - prices_low) - (prices_high - prices_close)) / (
        prices_high - prices_low
    )


def get_accumulation_distribution_line(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
) -> pd.Series:
    """
    Calculate the Accumulation/Distribution Line for a given price series.

    The Accumulation/Distribution Line is a running (cumulative) total of Money Flow Volume:
    each period's volume, weighted by where the close settled within that period's high-low
    range (the Money Flow Multiplier). A rising line suggests volume is flowing in on
    up-weighted days (accumulation); a falling line suggests the opposite (distribution).

    The formula is as follows:

    - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    - Money Flow Volume = Money Flow Multiplier * Volume
    - Accumulation/Distribution Line = Cumulative Sum(Money Flow Volume)

    Also known as: A/D Line, ADL.

    Reference: Chaikin, M. (1980s). There is no formal journal citation; the standard
    textbook treatment is Murphy, J.J. (1999). "Technical Analysis of the Financial Markets."
    New York Institute of Finance.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.

    Returns:
        pd.Series: Accumulation/Distribution Line values.
    """
    money_flow_multiplier = _get_money_flow_multiplier(
        prices_high, prices_low, prices_close
    )
    money_flow_volume = money_flow_multiplier * volumes

    return money_flow_volume.cumsum()


def get_trin(prices_close: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series:
    """
    Calculate the TRIN (Arms Index) for a universe of tickers.

    TRIN compares the ratio of advancing to declining issues against the ratio of
    volume in advancing issues to volume in declining issues. It is a cross-sectional
    market breadth indicator, meaning it is calculated across the columns (tickers) of
    the provided data rather than per ticker over time.

    The formula is as follows:

    - TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

    Also known as: Arms Index, TRIN.

    Reference: Arms, R.W. (1967). Originally published as a short-term breadth/volume
    indicator; see Arms, R.W. (1989). "The Arms Index (TRIN): An Introduction to the Volume
    Analysis of Stock and Bond Markets." Business One Irwin, for the definitive treatment.

    Args:
        prices_close (pd.DataFrame): DataFrame of closing prices with tickers as columns.
        volumes (pd.DataFrame): DataFrame of trading volumes with tickers as columns.

    Returns:
        pd.Series: TRIN values.
    """
    price_change = prices_close.diff()

    advancing = price_change > 0
    declining = price_change < 0

    advancing_issues = advancing.sum(axis=1)
    declining_issues = declining.sum(axis=1)

    advancing_volume = volumes.where(advancing, 0).sum(axis=1)
    declining_volume = volumes.where(declining, 0).sum(axis=1)

    return (advancing_issues / declining_issues) / (advancing_volume / declining_volume)


def get_new_highs_new_lows(prices_close: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the New Highs - New Lows for a universe of tickers.

    New Highs - New Lows measures the number of tickers reaching a new high over the
    specified window minus the number of tickers reaching a new low over the same
    window. It is a cross-sectional market breadth indicator, meaning it is calculated
    across the columns (tickers) of the provided data rather than per ticker over time.

    The formula is as follows:

    - New Highs - New Lows = (Number of tickers at a window-period high) - (Number of tickers at a window-period low)

    Also known as: new highs minus new lows, record high percent.

    Reference: A standard market-breadth statistic with no single named inventor; see Colby,
    R.W. (2003). "The Encyclopedia of Technical Market Indicators." 2nd ed. McGraw-Hill.

    Args:
        prices_close (pd.DataFrame): DataFrame of closing prices with tickers as columns.
        window (int): Number of periods to consider for the new high / new low lookback.

    Returns:
        pd.Series: New Highs - New Lows values.
    """
    rolling_high = prices_close.rolling(window=window, min_periods=1).max()
    rolling_low = prices_close.rolling(window=window, min_periods=1).min()

    new_highs = (prices_close >= rolling_high).sum(axis=1)
    new_lows = (prices_close <= rolling_low).sum(axis=1)

    return new_highs - new_lows


def get_chaikin_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Calculate the Chaikin Oscillator for a given price series.

    The Chaikin Oscillator applies MACD-style momentum analysis to the Accumulation/
    Distribution Line itself, taking the difference between a short-term and a long-term
    EMA of the ADL rather than of price. This surfaces changes in the momentum of volume
    flow before they necessarily show up in price, and is typically read as an early signal
    of a change in accumulation/distribution pressure.

    The formula is as follows:

    - Chaikin Oscillator = EMA(Accumulation/Distribution Line, short_window) - EMA
      (Accumulation/Distribution Line, long_window)

    Also known as: Chaikin Oscillator.

    Reference: Chaikin, M. (1980s). There is no formal journal citation; the standard
    textbook treatment is Murphy, J.J. (1999). "Technical Analysis of the Financial Markets."
    New York Institute of Finance.

    Notes:
        - Both EMAs are seeded with the Simple Moving Average of their first values (see
          `overlap_model.get_exponential_moving_average`), as StockCharts calculates them, so
          the oscillator starts on period `long_window` - 1. TA-Lib's `ta_ADOSC.c` seeds both
          with the first Accumulation/Distribution value instead, which only changes the
          first values.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        short_window (int): Number of periods for the short-term EMA.
        long_window (int): Number of periods for the long-term EMA.

    Returns:
        pd.Series: Chaikin Oscillator values.
    """
    adl = get_accumulation_distribution_line(
        prices_high, prices_low, prices_close, volumes
    )
    short_ema = get_exponential_moving_average(adl, short_window)
    long_ema = get_exponential_moving_average(adl, long_window)

    return short_ema - long_ema


def get_chaikin_money_flow(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate the Chaikin Money Flow (CMF) for a given price series.

    The Chaikin Money Flow sums the same Money Flow Volume used by the Accumulation/
    Distribution Line over a rolling window and normalizes it by the window's total volume,
    turning the running (unbounded) Accumulation/Distribution Line into a bounded oscillator.
    Sustained readings above zero indicate buying pressure (accumulation) is dominating over
    the window, while sustained readings below zero indicate selling pressure (distribution).

    The formula is as follows:

    - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    - Money Flow Volume = Money Flow Multiplier * Volume
    - CMF = Sum(Money Flow Volume, window) / Sum(Volume, window)

    Also known as: CMF, Chaikin Money Flow.

    Reference: Chaikin, M. (1980s). There is no formal journal citation; the standard
    textbook treatment is Murphy, J.J. (1999). "Technical Analysis of the Financial Markets."
    New York Institute of Finance.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods to sum the Money Flow Volume and volume over.

    Returns:
        pd.Series: Chaikin Money Flow values, bounded between -1 and 1.

    Raises:
        TypeError: If any of the price or volume arguments is not a pandas Series.
    """
    if not (
        isinstance(prices_high, pd.Series)
        and isinstance(prices_low, pd.Series)
        and isinstance(prices_close, pd.Series)
        and isinstance(volumes, pd.Series)
    ):
        raise TypeError(
            "prices_high, prices_low, prices_close and volumes must be pandas Series."
        )

    money_flow_multiplier = _get_money_flow_multiplier(
        prices_high, prices_low, prices_close
    )
    money_flow_volume = money_flow_multiplier * volumes

    return (
        money_flow_volume.rolling(window=window).sum()
        / volumes.rolling(window=window).sum()
    )


def get_ease_of_movement(
    prices_high: pd.Series,
    prices_low: pd.Series,
    volumes: pd.Series,
    window: int,
    volume_divisor: float = 100_000_000,
) -> pd.Series:
    """
    Calculate the Ease of Movement (EMV) for a given price series.

    The Ease of Movement indicator relates how far price moved (the change in the midpoint
    of the high-low range from one period to the next) to the volume required to move it
    (via the "Box Ratio", volume scaled down and divided by the period's high-low range).
    High positive readings mean price is moving up easily on relatively little volume;
    high negative readings mean price is moving down easily on relatively little volume.
    Readings near zero mean price needed a lot of volume to move very little, i.e. movement
    was "difficult". The raw daily reading is typically smoothed with a Simple Moving
    Average to reduce noise.

    The formula is as follows:

    - Distance Moved = (High(t) + Low(t)) / 2 - (High(t-1) + Low(t-1)) / 2
    - Box Ratio = (Volume / volume_divisor) / (High - Low)
    - Raw EMV = Distance Moved / Box Ratio
    - EMV = SMA(Raw EMV, window)

    Also known as: EMV, Ease of Movement.

    Reference: Arms, R.W. (1989). "The Arms Index (TRIN): An Introduction to the Volume
    Analysis of Stock and Bond Markets." Business One Irwin.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods used to smooth the raw Ease of Movement values with
            a Simple Moving Average.
        volume_divisor (float): Scaling constant applied to volume so that the Box Ratio (and
            therefore EMV) stays in a readable range regardless of an asset's typical share
            volume. Defaults to 100,000,000, the conventional scale used for equities.

    Returns:
        pd.Series: Ease of Movement values.

    Raises:
        TypeError: If `prices_high`, `prices_low` or `volumes` is not a pandas Series.
    """
    if not (
        isinstance(prices_high, pd.Series)
        and isinstance(prices_low, pd.Series)
        and isinstance(volumes, pd.Series)
    ):
        raise TypeError("prices_high, prices_low and volumes must be pandas Series.")

    midpoint = (prices_high + prices_low) / 2
    distance_moved = midpoint - midpoint.shift(1)

    box_ratio = (volumes / volume_divisor) / (prices_high - prices_low)
    raw_ease_of_movement = distance_moved / box_ratio

    return raw_ease_of_movement.rolling(window=window).mean()


def _get_volume_based_index(
    prices_close: pd.Series,
    volumes: pd.Series,
    direction: str,
    start_value: float = 1000.0,
) -> pd.Series:
    """
    Calculate the cumulative volume-based index shared by the Negative Volume Index and the
    Positive Volume Index. Both indices track a running index that only updates on days that
    qualify (a volume decrease for the Negative Volume Index, a volume increase for the
    Positive Volume Index) by compounding that day's percentage price change onto the prior
    index value; on non-qualifying days the index is carried forward unchanged.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        direction (str): Either "negative" (index updates on volume-decrease days) or
            "positive" (index updates on volume-increase days).
        start_value (float): The index value to start the series at. Defaults to 1000.0.

    Returns:
        pd.Series: The cumulative volume-based index values.
    """
    price_change_pct = prices_close.pct_change()
    volume_change = volumes.diff()

    qualifying_day = volume_change < 0 if direction == "negative" else volume_change > 0

    adjusted_change = price_change_pct.where(qualifying_day, 0.0).fillna(0.0)

    return start_value * (1 + adjusted_change).cumprod()


def get_negative_volume_index(
    prices_close: pd.Series, volumes: pd.Series, start_value: float = 1000.0
) -> pd.Series:
    """
    Calculate the Negative Volume Index (NVI) for a given price series.

    The Negative Volume Index is a cumulative index that only updates on days where volume
    decreases from the prior period, compounding that day's percentage price change onto the
    running index; on days where volume increases (or stays flat), the index is carried
    forward unchanged. The premise, per Fosback, is that "smart money" tends to be active on
    low-volume (quiet) days, so tracking price behaviour specifically on those days isolates
    informed trading from the noise of high-volume, crowd-driven days.

    The formula is as follows:

    - Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) - 1)) if Volume(t) < Volume(t-1)
    - Index(t) = Index(t-1) otherwise

    Also known as: NVI, Negative Volume Index.

    Reference: Fosback, N.G. (1976). "Stock Market Logic: A Sophisticated Approach to
    Profits on Wall Street." The Institute for Econometric Research.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        start_value (float): The index value to start the series at. Defaults to 1000.0.

    Returns:
        pd.Series: Negative Volume Index values.

    Raises:
        TypeError: If `prices_close` or `volumes` is not a pandas Series.
    """
    if not (isinstance(prices_close, pd.Series) and isinstance(volumes, pd.Series)):
        raise TypeError("prices_close and volumes must be pandas Series.")

    return _get_volume_based_index(prices_close, volumes, "negative", start_value)


def get_positive_volume_index(
    prices_close: pd.Series, volumes: pd.Series, start_value: float = 1000.0
) -> pd.Series:
    """
    Calculate the Positive Volume Index (PVI) for a given price series.

    The Positive Volume Index mirrors the Negative Volume Index: it is a cumulative index
    that only updates on days where volume increases from the prior period, compounding that
    day's percentage price change onto the running index; on days where volume decreases (or
    stays flat), the index is carried forward unchanged. Per Fosback, the Positive Volume
    Index isolates price behaviour on high-volume (crowd-driven) days, which is traditionally
    read as tracking less-informed, sentiment-driven trading.

    The formula is as follows:

    - Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) - 1)) if Volume(t) > Volume(t-1)
    - Index(t) = Index(t-1) otherwise

    Also known as: PVI, Positive Volume Index.

    Reference: Fosback, N.G. (1976). "Stock Market Logic: A Sophisticated Approach to
    Profits on Wall Street." The Institute for Econometric Research.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        start_value (float): The index value to start the series at. Defaults to 1000.0.

    Returns:
        pd.Series: Positive Volume Index values.

    Raises:
        TypeError: If `prices_close` or `volumes` is not a pandas Series.
    """
    if not (isinstance(prices_close, pd.Series) and isinstance(volumes, pd.Series)):
        raise TypeError("prices_close and volumes must be pandas Series.")

    return _get_volume_based_index(prices_close, volumes, "positive", start_value)


def _get_seeded_exponential_moving_average(values: pd.Series, span: int) -> pd.Series:
    """
    The exponential moving average with smoothing constant 2 / (span + 1), seeded with the
    simple average of the first `span` values rather than with the first value alone, so the
    average does not carry the noise of a single day for weeks. Dates without a value stay
    NaN and are skipped by the recursion, as is everything before the seed.

    Args:
        values (pd.Series): The values to average.
        span (int): The span of the average.

    Returns:
        pd.Series: The exponential moving average, NaN until `span` values are available.
    """
    observed = values.dropna()
    if len(observed) < span:
        return pd.Series(np.nan, index=values.index)

    seeded = observed.iloc[span - 1 :].copy()
    seeded.iloc[0] = observed.iloc[:span].mean()

    return seeded.ewm(span=span, adjust=False).mean().reindex(values.index)


def get_market_breadth(
    prices_close: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    high_low_window: int = 252,
) -> pd.DataFrame:
    """
    Calculate the breadth of a market from the prices of its constituents: how many of them
    take part in a move. An index can rise on the strength of a few large companies while
    most of its members fall; breadth shows that divergence, which often comes before the
    index itself turns.

    - Advancers, Decliners and Unchanged: the number of constituents whose close rose, fell or
      stayed the same.
    - Advance-Decline Line: the running total of advancers minus decliners.
    - Advance-Decline Ratio: advancers divided by decliners.
    - McClellan Oscillator: the ratio-adjusted version, the 19-period minus the 39-period
      exponential moving average (smoothing constants of 0.10 and 0.05) of the net advances
      divided by advancers plus decliners (times 1,000, so it does not depend on the number
      of constituents); unchanged issues are left out. Each average starts from the simple
      average of its first 19 or 39 values, so the oscillator starts once the 39-period
      average exists (NaN before), and so does its running total, the McClellan Summation
      Index.
    - Percentage Above the Moving Averages: the share of constituents whose close is above its
      short and long moving average.
    - New Highs and New Lows: the constituents whose close is above the highest (below the
      lowest) close of the previous `high_low_window` - 1 periods, so the highest or lowest
      of the last `high_low_window` periods. A close that only equals the previous high is no
      new high. NaN until at least one constituent has `high_low_window` closes.

    For more information, see the following book:

    - McClellan, S. and McClellan, M. (1970). "Patterns for Profit: The McClellan Oscillator
      and Summation Index." Trade Levels Inc.

    Args:
        prices_close (pd.DataFrame): The closes of the constituents, a column each.
        short_window (int, optional): The short moving average window. Defaults to 50.
        long_window (int, optional): The long moving average window. Defaults to 200.
        high_low_window (int, optional): The number of periods, including the current one, of
            the new highs and lows. Defaults to 252 (a year of daily closes).

    Returns:
        pd.DataFrame: The breadth measures per date.
    """
    changes = prices_close.diff()
    observed = changes.notna()
    advancers = (changes > 0).sum(axis=1)
    decliners = (changes < 0).sum(axis=1)
    unchanged = ((changes == 0) & observed).sum(axis=1)
    net_advances = advancers - decliners

    # The ratio-adjusted net advances (RANA) of StockCharts and McClellan: unchanged issues are left out of the denominator. A date on which every observed constituent closed unchanged has no net advance (zero) and a date without observations is unknown (NaN).  # noqa: E501
    ratio_adjusted = (
        (net_advances / (advancers + decliners).replace(0, np.nan) * 1000)
        .fillna(0)
        .where(observed.any(axis=1))
    )
    short_average = _get_seeded_exponential_moving_average(ratio_adjusted, 19)
    long_average = _get_seeded_exponential_moving_average(ratio_adjusted, 39)
    mcclellan = short_average - long_average

    def share_above(window: int) -> pd.Series:
        average = prices_close.rolling(window).mean()
        counted = average.notna() & prices_close.notna()
        above = (prices_close > average).astype(float).where(counted)
        return above.sum(axis=1) / counted.sum(axis=1).replace(0, np.nan)

    # Against the closes before today, so a close that merely equals the previous high (or low) is not counted as a new one day after day.  # noqa: E501
    if high_low_window > 1:
        previous_closes = prices_close.shift(1).rolling(high_low_window - 1)
        previous_high, previous_low = previous_closes.max(), previous_closes.min()
    else:
        # A window of a single period has no earlier close to set a high or a low.
        previous_high = previous_low = prices_close * np.nan
    counted = previous_high.notna() & prices_close.notna()

    def count_new(extremes: pd.DataFrame) -> pd.Series:
        # NaN rather than zero while no constituent has a full window yet.
        return extremes.astype(float).where(counted).sum(axis=1, min_count=1)

    breadth = pd.DataFrame(
        {
            "Advancers": advancers,
            "Decliners": decliners,
            "Unchanged": unchanged,
            "Net Advances": net_advances,
            "Advance-Decline Line": net_advances.cumsum(),
            "Advance-Decline Ratio": advancers / decliners.replace(0, np.nan),
            "McClellan Oscillator": mcclellan,
            "McClellan Summation Index": mcclellan.cumsum(),
            f"Percentage Above {short_window}-Period Average": share_above(
                short_window
            ),
            f"Percentage Above {long_window}-Period Average": share_above(long_window),
            "New Highs": count_new(prices_close > previous_high),
            "New Lows": count_new(prices_close < previous_low),
        }
    )
    # The first date has no change to count.
    return breadth.iloc[1:]
