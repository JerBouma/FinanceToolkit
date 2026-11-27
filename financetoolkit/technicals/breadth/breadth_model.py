"""Breadth Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.technicals.helpers import apply_on_own_closes
from financetoolkit.technicals.overlap.overlap_model import (
    get_exponential_moving_average,
)
from financetoolkit.technicals.volatility.volatility_model import (
    get_wilder_moving_average,
)

# Katsanos measures the volatility of the typical price, which sets the cutoff of the Volume
# Flow Indicator, over 30 periods whatever the window of the indicator itself.
VOLUME_FLOW_VOLATILITY_WINDOW = 30


def get_mcclellan_oscillator(
    prices_close: pd.Series, short_ema_window: int, long_ema_window: int
) -> pd.Series:
    """
    Calculate the McClellan Oscillator: the momentum of the market's breadth, as the difference between a fast and a
    slow average of the advancing minus declining tickers.

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
    # Over the observed closes, so a close after a gap compares with the last close before
    # it and the EMAs carry on over the gap; days without a close stay NaN. The first close
    # has no previous one and therefore no advance or decline; leaving it missing keeps a
    # zero that never happened out of the seed of both EMAs.
    observed = prices_close.dropna()
    advancers_decliners = get_advancers_decliners(observed).iloc[1:]
    short_ema = get_exponential_moving_average(advancers_decliners, short_ema_window)
    long_ema = get_exponential_moving_average(advancers_decliners, long_ema_window)

    return (short_ema - long_ema).reindex(prices_close.index)


def get_advancers_decliners(
    prices_close: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the advance/decline signal: whether each ticker closed higher (1) or lower (-1) than the period
    before, the building block of the market breadth measures.

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

    def signal(closes: pd.Series) -> pd.Series:
        # Over the observed closes, so the day after a missing one compares with the last
        # close before it, and days without a close (before listing, gaps) stay NaN. The
        # first close has no earlier one and counts as unchanged.
        observed = closes.dropna()
        return np.sign(observed.diff()).fillna(0).reindex(closes.index)

    if isinstance(prices_close, pd.DataFrame):
        return prices_close.apply(signal)

    return signal(prices_close)


def get_on_balance_volume(
    prices_close: pd.Series | pd.DataFrame, volumes: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the On-Balance Volume (OBV): a running total of volume that adds the volume of up periods and
    subtracts that of down periods, to show whether volume flows into or out of a security.

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

    Referenced against:
    - TA-Lib, OBV: https://ta-lib.org/functions/obv.html

    Notes:
        - Unlike TA-Lib's OBV, whose running total starts at the first volume, the running total starts at 0.

    Args:
        prices_close (pd.Series | pd.DataFrame): Closing prices, one column per ticker when
            a DataFrame is given.
        volumes (pd.Series | pd.DataFrame): Trading volumes, in the shape of `prices_close`.

    Returns:
        pd.Series | pd.DataFrame: OBV values, in the shape of the input.
    """
    if isinstance(prices_close, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_on_balance_volume(prices_close[column], volumes[column])
                for column in prices_close.columns
            },
            index=prices_close.index,
        )

    # Over the days with both a close and a volume, so the day after a missing one compares
    # with the last close before it; days without data stay NaN.
    observed = prices_close.notna() & volumes.notna()
    direction = np.sign(prices_close[observed].diff()).fillna(0)
    obv = direction * volumes[observed]

    return obv.cumsum().reindex(prices_close.index)


def get_price_volume_trend(prices_close: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Calculate the Price Volume Trend (PVT): a running total of volume weighted by the percentage change of the
    close, which shows how strongly buying or selling pressure backs a trend.

    The Price Volume Trend is a running total of volume weighted by the period's percentage
    change in price. Where On-Balance Volume adds or subtracts the whole volume depending on
    the direction of the close, the PVT adds only the share of the volume that matches the
    size of the move, so a close 0.1% higher adds a tenth of what a close 1% higher adds.
    A rising PVT confirms an uptrend carried by volume, and a divergence from price warns
    that the move lacks participation.

    The formula is as follows:

    - PVT(t) = PVT(t-1) + Volume(t) * (Close(t) - Close(t-1)) / Close(t-1)
    - PVT = 0 on the first observation, which has no previous close

    Also known as: PVT, Price and Volume Trend, Volume Price Trend.

    Reference: StockCharts ChartSchool, "Price Volume Trend (PVT)". The PVT has no single
    published origin; it refines Granville's On-Balance Volume (Granville, J.E. (1963).
    "Granville's New Key to Stock Market Profits." Prentice-Hall) by weighting volume with
    the percentage change in price.

    Referenced against:
    - TA-Lib, PVT: https://ta-lib.org/functions/pvt.html

    Notes:
        - The running total starts at 0 on the first observation with both a close and a
          volume, so its level depends on the first date of the data; only its direction
          and its divergences from price carry meaning.
        - The change is taken over the observations with both a close and a volume, so the
          observation after a missing one compares with the last close before it; the
          missing observations stay NaN.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.

    Returns:
        pd.Series: PVT values, in units of volume.
    """
    observed = prices_close.notna() & volumes.notna()
    weighted_volume = prices_close[observed].pct_change() * volumes[observed]

    return weighted_volume.fillna(0).cumsum().reindex(prices_close.index)


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

    A period whose high equals its low has no range to place the close in, so its
    multiplier is 0 and its volume adds nothing, as in TA-Lib, instead of a NaN that would
    break the running total.

    Returns:
        pd.Series: Money Flow Multiplier values, bounded between -1 and 1.
    """
    multiplier = ((prices_close - prices_low) - (prices_high - prices_close)) / (
        prices_high - prices_low
    )

    return multiplier.mask(prices_high == prices_low, 0.0)


def get_accumulation_distribution_line(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
) -> pd.Series:
    """
    Calculate the Accumulation/Distribution Line: a running total of volume weighted by where each close lies within
    its high-low range, to show whether a security is being accumulated or distributed.

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

    Referenced against:
    - TA-Lib, AD: https://ta-lib.org/functions/ad.html

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
    Calculate the TRIN (Arms Index): the ratio of advancing to declining tickers divided by the ratio of their
    volumes, which shows whether volume favours the rising or the falling tickers.

    TRIN compares the ratio of advancing to declining issues against the ratio of
    volume in advancing issues to volume in declining issues. It is a cross-sectional
    market breadth indicator, meaning it is calculated across the columns (tickers) of
    the provided data rather than per ticker over time.

    The formula is as follows:

    - TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

    Also known as: Arms Index, TRIN.

    Reference: Arms, R.W. (1967). Originally published as a short-term breadth/volume
    indicator; see Arms, R.W. (1989). "The Arms Index (TRIN): An Introduction to the Volume
    Analysis of Stock and Bond Markets." Dow Jones-Irwin, for the definitive treatment.

    Args:
        prices_close (pd.DataFrame): DataFrame of closing prices with tickers as columns,
            NaN on the dates a ticker did not trade. A ticker's change runs from its own
            previous close, and a ticker without a close on a date is not counted on it.
        volumes (pd.DataFrame): DataFrame of trading volumes with tickers as columns.

    Returns:
        pd.Series: TRIN values.
    """
    # Each ticker's change since its own previous close, so a ticker without a close on a date (it did not trade) is neither advancing nor declining.  # noqa: E501
    price_change = apply_on_own_closes(prices_close, lambda close: close.diff())

    advancing = price_change > 0
    declining = price_change < 0

    advancing_issues = advancing.sum(axis=1)
    declining_issues = declining.sum(axis=1)

    advancing_volume = volumes.where(advancing, 0).sum(axis=1)
    declining_volume = volumes.where(declining, 0).sum(axis=1)

    return (advancing_issues / declining_issues) / (advancing_volume / declining_volume)


def get_new_highs_new_lows(prices_close: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the New Highs - New Lows: the number of tickers at a new high over a window minus the number at a new
    low, which shows how broad a rise or a fall is.

    New Highs - New Lows measures the number of tickers reaching a new high over the
    specified window minus the number of tickers reaching a new low over the same
    window. It is a cross-sectional market breadth indicator, meaning it is calculated
    across the columns (tickers) of the provided data rather than per ticker over time.

    The formula is as follows:

    - New Highs - New Lows = (Number of tickers at a window-period high) - (Number of tickers at a window-period low)

    A new high is a close above the highest close of the previous window - 1 periods, so
    the highest of the last `window` periods; a close that only equals the previous high is
    no new high (and the same for lows), as in get_market_breadth. The result is NaN until
    at least one ticker has `window` closes.

    Also known as: new highs minus new lows, record high percent.

    Reference: A standard market-breadth statistic with no single named inventor; see Colby,
    R.W. (2003). "The Encyclopedia of Technical Market Indicators." 2nd ed. McGraw-Hill.

    Args:
        prices_close (pd.DataFrame): DataFrame of closing prices with tickers as columns,
            NaN on the dates a ticker did not trade (its window counts its own closes).
        window (int): Number of periods to consider for the new high / new low lookback.

    Returns:
        pd.Series: New Highs - New Lows values.
    """
    breadth = get_market_breadth(prices_close, high_low_window=window)

    # The breadth starts at the first daily change, so the first date (never a full window)
    # is added back as NaN to keep the dates of the prices.
    return (breadth["New Highs"] - breadth["New Lows"]).reindex(prices_close.index)


def get_chaikin_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Calculate the Chaikin Oscillator: the momentum of the Accumulation/Distribution Line, as the difference between
    its fast and slow exponential moving averages.

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

    Referenced against:
    - TA-Lib, ADOSC: https://ta-lib.org/functions/adosc.html

    Notes:
        - Both EMAs are seeded with the Simple Moving Average of their first values (see
          `overlap_model.get_exponential_moving_average`), as StockCharts calculates them, so
          the oscillator starts on period `long_window` - 1. TA-Lib's ADOSC seeds both
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
    Calculate the Chaikin Money Flow (CMF): the share of volume over a window that came with closes near the highs
    rather than the lows, a measure of buying versus selling pressure.

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

    Referenced against:
    - TA-Lib, CMF: https://ta-lib.org/functions/cmf.html

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
    Calculate the Ease of Movement (EMV): how far the price moves per unit of volume, so a large value shows a price
    rising or falling on little volume.

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
    Analysis of Stock and Bond Markets." Dow Jones-Irwin.

    Referenced against:
    - TA-Lib, EMV: https://ta-lib.org/functions/emv.html

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
    # Over the days with both a close and a volume, so a missing day neither drops the next
    # day's change nor prints the start value before the first close; those days stay NaN.
    observed = prices_close.notna() & volumes.notna()
    closes, volumes_observed = prices_close[observed], volumes[observed]

    price_change_pct = closes.pct_change()
    volume_change = volumes_observed.diff()

    qualifying_day = volume_change < 0 if direction == "negative" else volume_change > 0

    adjusted_change = price_change_pct.where(qualifying_day, 0.0).fillna(0.0)

    return (start_value * (1 + adjusted_change).cumprod()).reindex(prices_close.index)


def get_negative_volume_index(
    prices_close: pd.Series, volumes: pd.Series, start_value: float = 1000.0
) -> pd.Series:
    """
    Calculate the Negative Volume Index (NVI): a running total of the price changes on the periods volume falls,
    which follows what informed investors do on quiet days.

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

    Referenced against:
    - TA-Lib, NVI: https://ta-lib.org/functions/nvi.html

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
    Calculate the Positive Volume Index (PVI): a running total of the price changes on the periods volume rises,
    which follows the crowd on busy days.

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

    Referenced against:
    - TA-Lib, PVI: https://ta-lib.org/functions/pvi.html

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
    Calculate the breadth of a market from the prices of its constituents: how many of them take part in a move.

    An index can rise on the strength of a few large companies while most of its members fall; breadth shows that
    divergence, which often comes before the index itself turns.

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
      Index. When a constituent has dates without a close inside its history, such as a
      stock next to a cryptocurrency, the averages run over the dates on which such a
      constituent traded and are NaN on the others (a cryptocurrency's weekend).
    - Percentage Above the Moving Averages: the share of constituents whose close is above its
      short and long moving average.
    - New Highs and New Lows: the constituents whose close is above the highest (below the
      lowest) close of the previous `high_low_window` - 1 periods, so the highest or lowest
      of the last `high_low_window` periods. A close that only equals the previous high is no
      new high. NaN until at least one constituent has `high_low_window` closes.

    Every change, moving average and high or low of a constituent runs over its own closes,
    and a constituent is counted on the dates it has a close, so a stock is not counted as
    unchanged on the weekends of a cryptocurrency in the same universe.

    For more information, see the following book:

    - McClellan, S. and McClellan, M. (1970). "Patterns for Profit: The McClellan Oscillator
      and Summation Index." Trade Levels Inc.

    Args:
        prices_close (pd.DataFrame): The closes of the constituents, a column each, NaN on
            the dates a constituent did not trade.
        short_window (int, optional): The short moving average window. Defaults to 50.
        long_window (int, optional): The long moving average window. Defaults to 200.
        high_low_window (int, optional): The number of periods, including the current one, of
            the new highs and lows. Defaults to 252 (a year of daily closes).

    Returns:
        pd.DataFrame: The breadth measures per date.
    """
    # Each constituent's change since its own previous close: a constituent without a close on a date (such as a stock on a weekend next to a cryptocurrency) is not counted on it, and its next change spans the days it did not trade.  # noqa: E501
    changes = apply_on_own_closes(prices_close, lambda close: close.diff())
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
    # A constituent with dates inside its history without a close, such as a stock next to a cryptocurrency, sets the dates of the averages: they run over the dates on which such a constituent traded, so the weekends of a cryptocurrency in the universe do not add periods that the 19 and 39 count, while a universe that trades every date (only cryptocurrencies, or stocks of one exchange) keeps all its dates.  # noqa: E501
    inside_gaps = (
        prices_close.isna()
        & prices_close.ffill().notna()
        & prices_close.bfill().notna()
    )
    with_gaps = inside_gaps.any()
    if with_gaps.any():
        ratio_adjusted = ratio_adjusted.where(
            prices_close.loc[:, with_gaps].notna().any(axis=1)
        )
    short_average = _get_seeded_exponential_moving_average(ratio_adjusted, 19)
    long_average = _get_seeded_exponential_moving_average(ratio_adjusted, 39)
    mcclellan = short_average - long_average

    def share_above(window: int) -> pd.Series:
        average = apply_on_own_closes(
            prices_close, lambda close: close.rolling(window).mean()
        )
        counted = average.notna() & prices_close.notna()
        above = (prices_close > average).astype(float).where(counted)
        return above.sum(axis=1) / counted.sum(axis=1).replace(0, np.nan)

    # Against the closes before today, so a close that merely equals the previous high (or low) is not counted as a new one day after day.  # noqa: E501
    if high_low_window > 1:
        previous_high = apply_on_own_closes(
            prices_close,
            lambda close: close.shift(1).rolling(high_low_window - 1).max(),
        )
        previous_low = apply_on_own_closes(
            prices_close,
            lambda close: close.shift(1).rolling(high_low_window - 1).min(),
        )
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


def get_klinger_volume_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    short_window: int,
    long_window: int,
    signal_window: int,
) -> pd.DataFrame:
    """
    Calculate the Klinger Volume Oscillator (KVO): the difference between a fast and a slow average of the volume
    force, which compares the volume of up and down trends to show long-term money flow.

    The Klinger Volume Oscillator turns volume into a signed "volume force" that depends on
    the direction of the trend (the sum of the high, low and close against that of the
    previous period) and on how the period's range compares with the range accumulated since
    the trend last changed direction. The oscillator is the difference between a short and a
    long Exponential Moving Average of that force, with a signal line to show when it turns.
    Klinger meant it to catch the long-term flow of money while staying sensitive enough for
    short-term reversals.

    The formula is as follows:

    - Trend = +1 if High + Low + Close > the previous High + Low + Close, -1 if lower, and the
      previous Trend if unchanged
    - dm = High - Low
    - cm = cm(t-1) + dm if the Trend is unchanged, else dm(t-1) + dm
    - Volume Force = Volume * |2 * dm / cm - 1| * Trend * 100
    - KVO = EMA(Volume Force, short_window) - EMA(Volume Force, long_window)
    - Signal Line = EMA(KVO, signal_window)

    The conventional parameters are 34, 55 and 13 periods.

    Also known as: KVO, Klinger Oscillator, Klinger Volume Force.

    Reference: Klinger, S.J. (1997). "Identifying Trends With Volume Analysis." Technical
    Analysis of Stocks & Commodities, 15(12).

    Referenced against:
    - TA-Lib, KVO: https://ta-lib.org/functions/kvo.html
    - Tulip Indicators, kvo: https://tulipindicators.org/kvo

    Notes:
        - The Volume Force agrees with TA-Lib's and Tulip Indicators' (`kvo`), with |2 * dm / cm - 1|.
          pandas-ta-classic's `kvo` uses a simplified force, the volume signed by the change
          in the typical price, and differs.
        - Every EMA is seeded with the Simple Moving Average of its first values (see
          `overlap_model.get_exponential_moving_average`). The first Volume Force is on the
          second period, so the KVO starts on period `long_window` + 1, as TA-Lib's, and the
          Signal Line `signal_window` - 1 periods later. TA-Lib seeds both averages with the
          first Volume Force and its Signal Line with the first KVO, which only changes the
          values during the warm-up.
        - A period whose cm is zero (no range since the trend changed) has no Volume Force.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        short_window (int): Number of periods for the short-term EMA of the Volume Force.
        long_window (int): Number of periods for the long-term EMA of the Volume Force.
        signal_window (int): Number of periods for the signal line EMA.

    Returns:
        pd.DataFrame: The Klinger Volume Oscillator and its Signal Line as columns, in units
        of volume.
    """
    high_low_close = prices_high + prices_low + prices_close
    daily_measurement = prices_high - prices_low

    # An unchanged sum keeps the previous trend; the first period has none.
    trend = np.sign(high_low_close.diff()).replace(0, np.nan).ffill()

    # The cumulative measurement restarts from the previous range whenever the trend turns,
    # including the first trend on the second period, as in Klinger's article and TA-Lib.
    trend_run = (trend != trend.shift(1)).cumsum()
    cumulative_measurement = daily_measurement.groupby(
        trend_run
    ).cumsum() + daily_measurement.shift(1).groupby(trend_run).transform("first")

    volume_force = (
        volumes
        * (
            2
            * daily_measurement
            / cumulative_measurement.where(cumulative_measurement != 0)
            - 1
        ).abs()
        * trend
        * 100
    )

    klinger_volume_oscillator = get_exponential_moving_average(
        volume_force, short_window
    ) - get_exponential_moving_average(volume_force, long_window)
    signal_line = get_exponential_moving_average(
        klinger_volume_oscillator, signal_window
    )

    return pd.concat(
        [klinger_volume_oscillator, signal_line],
        keys=["Klinger Volume Oscillator", "Signal Line"],
        axis=1,
    )


def get_market_facilitation_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    volumes: pd.Series,
    volume_divisor: float = 1.0,
) -> pd.Series:
    """
    Calculate Bill Williams' Market Facilitation Index (MFI): the range of a period per unit of volume, which shows
    how efficiently volume moves the price.

    The Market Facilitation Index measures how far price moved per unit of volume: the
    period's range divided by its volume. Williams read it together with the change in
    volume: a rising index on rising volume ("green") shows a market that moves easily with
    new participants, a falling index on falling volume ("fade") a market that loses
    interest, a falling index on rising volume ("squat") a struggle that often comes before
    a breakout, and a rising index on falling volume ("fake") a move without participation.

    The formula is as follows:

    - Market Facilitation Index = (High - Low) / (Volume / volume_divisor)

    Also known as: BW MFI, Market Facilitation Index, MARKETFI.

    Reference: Williams, B.M. (1995). "Trading Chaos: Applying Expert Techniques to
    Maximize Your Profits." John Wiley & Sons.

    Referenced against:
    - TA-Lib, MARKETFI: https://ta-lib.org/functions/marketfi.html
    - pandas-ta-classic, marketfi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/marketfi.py#L16
    - Tulip Indicators, marketfi: https://tulipindicators.org/marketfi

    Notes:
        - With `volume_divisor` = 1 it agrees with TA-Lib's MARKETFI, pandas-ta-classic's
          `marketfi` and Tulip Indicators' `marketfi`. A period without volume has no index
          (NaN) rather than an infinite one.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        volumes (pd.Series): Series of trading volumes.
        volume_divisor (float): Scaling constant the volume is divided by, so the index stays
            in a readable range. Defaults to 1.0.

    Returns:
        pd.Series: Market Facilitation Index values, in price per unit of
        volume / volume_divisor.
    """
    scaled_volumes = volumes / volume_divisor

    return (prices_high - prices_low) / scaled_volumes.where(scaled_volumes != 0)


def get_williams_accumulation_distribution(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series
) -> pd.Series:
    """
    Calculate Larry Williams' Accumulation/Distribution (WAD): a running total of how far each close moved from the
    opposite end of its true range, which shows accumulation and distribution from prices alone.

    Williams' Accumulation/Distribution is a running total of how far each close moved away
    from the opposite end of the period's true range: on an up close, the distance from the
    true low (accumulation); on a down close, the distance from the true high (distribution).
    It uses no volume. Williams traded the divergences: price making a new high while the
    line does not points to distribution, and the other way around.

    The formula is as follows:

    - True High = max(High, Close(t-1)), True Low = min(Low, Close(t-1))
    - A/D(t) = Close - True Low if Close > Close(t-1), Close - True High if Close < Close(t-1),
      else 0
    - WAD(t) = WAD(t-1) + A/D(t), starting at 0 on the first observation

    Also known as: WAD, Williams A/D, Williams Accumulation Distribution.

    Reference: Williams, L.R. (1973). "How I Made One Million Dollars Last Year Trading
    Commodities." Windsor Books.

    Referenced against:
    - TA-Lib, WAD: https://ta-lib.org/functions/wad.html
    - pandas-ta-classic, wad: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/wad.py#L13
    - Tulip Indicators, wad: https://tulipindicators.org/wad

    Notes:
        - It agrees with TA-Lib's WAD and Tulip Indicators' `wad` (which starts on the second
          period) and pandas-ta-classic's `wad` (NaN on the first period).
        - The running total starts at 0 on the first observation with a high, low and close,
          so its level depends on the first date of the data; only its direction and its
          divergences from price carry meaning. The observation after a missing one compares
          with the last close before it, and the missing observations stay NaN.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: Williams' Accumulation/Distribution values, in price units.
    """
    observed = prices_high.notna() & prices_low.notna() & prices_close.notna()
    high, low, close = (
        prices_high[observed],
        prices_low[observed],
        prices_close[observed],
    )

    previous_close = close.shift(1)
    true_high = np.fmax(high, previous_close)
    true_low = np.fmin(low, previous_close)

    # The first close has no previous one, so it compares with neither and adds 0.
    accumulation_distribution = (close - true_low).where(
        close > previous_close, (close - true_high).where(close < previous_close, 0.0)
    )

    return accumulation_distribution.cumsum().reindex(prices_close.index)


def get_relative_volume(volumes: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Relative Volume (RVOL): the volume of a period divided by its average over a window, which shows
    how unusually busy the period is.

    Relative Volume compares a period's volume with the average volume of the periods before
    it. A value of 2 means twice the usual volume: a breakout or a gap on high relative
    volume carries more conviction than one on thin trading, and traders use it to screen for
    stocks "in play".

    The formula is as follows:

    - RVOL = Volume(t) / SMA(Volume, window)(t-1)

    The average covers the `window` periods before the current one, so a single spike does
    not raise its own benchmark.

    Also known as: RVOL, Relative Volume, volume ratio.

    Reference: A common screening measure with no single published origin; see Bulkowski,
    T.N. (2005). "Encyclopedia of Chart Patterns." 2nd ed. John Wiley & Sons, for the role
    of above-average volume in breakouts.

    Referenced against:
    - TA-Lib, RVOL: https://ta-lib.org/functions/rvol.html

    Notes:
        - It agrees with TA-Lib's RVOL. An average volume of zero (a series without volume, such
          as an index) gives NaN rather than an infinite ratio.

    Args:
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of earlier periods to average the volume over.

    Returns:
        pd.Series: Relative Volume values, as a ratio (1 is the average volume).
    """
    average_volume = volumes.rolling(window=window).mean().shift(1)

    return volumes / average_volume.where(average_volume != 0)


def get_volume_flow_indicator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
    coefficient: float,
    volume_coefficient: float,
    smooth_window: int,
) -> pd.Series:
    """
    Calculate Markos Katsanos' Volume Flow Indicator (VFI): the volume of the periods with a meaningful move of the
    typical price, signed by the direction of the move, which shows whether money flows in or out.

    The Volume Flow Indicator refines On-Balance Volume in three ways: it uses the typical
    price instead of the close, it ignores moves too small to matter (below a cutoff that
    scales with the recent volatility of the typical price) and it caps the volume of a
    single period at a multiple of the average volume, so one spike cannot dominate. The sum
    of that signed volume over the window, divided by the average volume, is smoothed with a
    short EMA. Katsanos read a VFI above zero as a bullish flow of money and below zero as a
    bearish one, and divergences from price as early warnings.

    The formula is as follows:

    - Typical Price = (High + Low + Close) / 3
    - Volatility = Population Standard Deviation(ln(Typical Price / Typical Price(t-1)), 30)
    - Cutoff = coefficient * Volatility * Close
    - Average Volume = SMA(Volume, window)(t-1)
    - Capped Volume = min(Volume, volume_coefficient * Average Volume)
    - Signed Volume = +Capped Volume if Typical Price - Typical Price(t-1) > Cutoff, -Capped
      Volume if it is below -Cutoff, else 0
    - VFI = EMA(Sum(Signed Volume, window) / Average Volume, smooth_window)

    The conventional parameters are a window of 130 days (about six months), a coefficient
    of 0.2, a volume coefficient of 2.5 and a smoothing of 3 periods.

    Also known as: VFI, Volume Flow Indicator.

    Reference: Katsanos, M. (2004). "Using Money Flow to Stay with the Trend." Technical
    Analysis of Stocks & Commodities, 22(6), and Katsanos, M. (2004). "Volume Flow Indicator
    Performance." Technical Analysis of Stocks & Commodities, 22(7).

    Referenced against:
    - pandas-ta-classic, vfi: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/vfi.py#L17

    Shares similar logic:
    - finta, VFI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L2026

    Notes:
        - The volatility is the population standard deviation (ddof=0) of the logarithmic
          change in the typical price, as in Katsanos' MetaStock formula (whose Stdev divides by
          n): the deviation only sets the cutoff of the indicator's definition, so it follows
          the primary source, as the Bollinger Bands do. The VFI therefore equals
          pandas-ta-classic's `vfi` exactly. finta uses the sample deviation (ddof=1), a
          1.7% larger cutoff over 30 periods, seeds its EMA with the first value
          (`adjust=True`) and counts volume while the average is unknown, so it differs
          wherever a move falls between the two cutoffs.
        - The EMA is seeded with the Simple Moving Average of its first values. The cutoff
          needs 30 changes of the typical price and the average volume `window` earlier
          periods, the sum `window` signed volumes and the EMA `smooth_window` sums, so
          the first value comes after max(`window`, 30) + `window` + `smooth_window` - 2
          periods (261 with the defaults).
        - The VFI is in units of the average volume, a sum of up to `window` periods, not a
          ratio between -1 and 1.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for the average volume and the sum of signed volume.
        coefficient (float): The multiple of the volatility times the close a move of the
            typical price needs to count.
        volume_coefficient (float): The multiple of the average volume a single period's
            volume is capped at.
        smooth_window (int): Number of periods for the EMA that smooths the result.

    Returns:
        pd.Series: Volume Flow Indicator values.
    """
    typical_price = (prices_high + prices_low + prices_close) / 3
    volatility = (
        np.log(typical_price)
        .diff()
        .rolling(window=VOLUME_FLOW_VOLATILITY_WINDOW)
        .std(ddof=0)
    )
    cutoff = coefficient * volatility * prices_close

    average_volume = volumes.rolling(window=window).mean().shift(1)
    capped_volume = np.minimum(volumes, volume_coefficient * average_volume)

    price_change = typical_price.diff()
    signed_volume = capped_volume.where(
        price_change > cutoff, (-capped_volume).where(price_change < -cutoff, 0.0)
    )
    # A comparison with a missing value is False and would read as no flow.
    signed_volume = signed_volume.where(
        price_change.notna() & cutoff.notna() & capped_volume.notna()
    )

    volume_flow = signed_volume.rolling(window=window).sum() / average_volume.where(
        average_volume != 0
    )

    return get_exponential_moving_average(volume_flow, smooth_window)


def get_finite_volume_elements(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
    cutoff: float,
) -> pd.Series:
    """
    Calculate Markos Katsanos' Finite Volume Elements (FVE): the share of volume over a window classified as buying
    rather than selling, leaving out moves too small to matter, a measure of money flow.

    Finite Volume Elements is a money flow indicator that classifies each period's whole
    volume as buying, selling or neutral. It looks at both the intraday move (the close
    against the middle of the range) and the interday move (the change in the typical
    price), and leaves out periods whose combined move is smaller than a cutoff, so noise
    does not count as flow. The result is the net signed volume as a share of all volume
    over the window: a rising FVE while price consolidates points to accumulation.

    The formula is as follows:

    - Money Flow = Close - (High + Low) / 2 + Typical Price - Typical Price(t-1), with
      Typical Price = (High + Low + Close) / 3
    - Signed Volume = +Volume if Money Flow > cutoff * Close, -Volume if Money Flow < -cutoff
      * Close, else 0
    - FVE = Sum(Signed Volume, window) / Sum(Volume, window)

    The conventional parameters are 22 periods (a month of trading days) and a cutoff of
    0.3% of the close.

    Also known as: FVE, Finite Volume Elements.

    Reference: Katsanos, M. (2003). "Detecting Breakouts In Intraday Charts." Technical
    Analysis of Stocks & Commodities, 21(9), and Katsanos, M. (2003). "Detecting Breakouts."
    Technical Analysis of Stocks & Commodities, 21(4).

    Referenced against:
    - finta, FVE: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1998

    Notes:
        - The FVE is a decimal between -1 and 1 (0.25 is 25% net buying volume); Katsanos and
          finta multiply it by 100. finta's `factor` is the cutoff in percent (0.3), here a
          decimal (0.003). finta counts the first period as neutral volume, here it has no
          Money Flow, so the FVE starts on period `window` + 1 instead of `window`; after
          that the two are equal.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods to sum the signed and total volume over.
        cutoff (float): The share of the close the money flow needs to exceed to count, as a
            decimal (0.003 is 0.3%).

    Returns:
        pd.Series: Finite Volume Elements values, as decimals between -1 and 1.
    """
    typical_price = (prices_high + prices_low + prices_close) / 3
    money_flow = prices_close - (prices_high + prices_low) / 2 + typical_price.diff()
    threshold = cutoff * prices_close

    signed_volume = volumes.where(
        money_flow > threshold, (-volumes).where(money_flow < -threshold, 0.0)
    ).where(money_flow.notna())

    total_volume = volumes.where(money_flow.notna()).rolling(window=window).sum()

    return signed_volume.rolling(window=window).sum() / total_volume.where(
        total_volume != 0
    )


def get_twiggs_money_flow(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate Colin Twiggs' Money Flow (TMF): the share of volume that came with closes near the top of the true
    range rather than its bottom, smoothed, a measure of buying versus selling pressure.

    Twiggs Money Flow is a refinement of the Chaikin Money Flow. It places the close within
    the true range (which includes a gap from the previous close) instead of the high-low
    range, so a gap does not show up as a close at the top or bottom of a narrow range, and
    it smooths the money flow volume and the volume with Wilder's moving average instead of
    summing them over a window, so a large volume does not distort the result when it leaves
    the window. Readings above zero show buying pressure, below zero selling pressure.

    The formula is as follows:

    - True High = max(High, Close(t-1)), True Low = min(Low, Close(t-1))
    - Money Flow Volume = Volume * ((Close - True Low) - (True High - Close)) / (True High -
      True Low)
    - TMF = Wilder MA(Money Flow Volume, window) / Wilder MA(Volume, window)

    The conventional window is 21 periods.

    Also known as: TMF, Twiggs Money Flow.

    Reference: Twiggs, C. (2003). "Twiggs Money Flow." Incredible Charts
    (https://www.incrediblecharts.com/indicators/twiggs_money_flow.php).

    Notes:
        - None of TA-Lib, pandas-ta-classic, ta or Tulip Indicators has it, and finta's TMF is
          an unfinished stub that raises NotImplementedError, so it is checked against
          Twiggs' formula as Incredible Charts and the Chart manual of Kevin Ryde describe it.
        - Wilder's moving average is seeded with the Simple Moving Average of its first
          `window` values (see `volatility_model.get_wilder_moving_average`). The first
          period has no previous close, so its true range is its high-low range. A period
          whose true high equals its true low adds no money flow volume, as in the Chaikin
          Money Flow.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for Wilder's moving average.

    Returns:
        pd.Series: Twiggs Money Flow values, bounded between -1 and 1.
    """
    previous_close = prices_close.shift(1)
    true_high = np.fmax(prices_high, previous_close)
    true_low = np.fmin(prices_low, previous_close)

    multiplier = ((prices_close - true_low) - (true_high - prices_close)) / (
        true_high - true_low
    )
    money_flow_volume = multiplier.mask(true_high == true_low, 0.0) * volumes

    average_volume = get_wilder_moving_average(volumes, window)

    return get_wilder_moving_average(money_flow_volume, window) / average_volume.where(
        average_volume != 0
    )


def get_volume_zone_oscillator(
    prices_close: pd.Series, volumes: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Volume Zone Oscillator (VZO): the volume of up closes minus that of down closes as a share of the
    total volume, smoothed, which shows whether buying or selling volume dominates.

    The Volume Zone Oscillator divides the volume into the volume of up closes and that of
    down closes and expresses their difference, smoothed with an EMA, as a share of the
    smoothed total volume. Khalil and Steckler read it in zones: above 0.05 the trend is
    bullish and above 0.40 overbought (0.60 extremely), below -0.05 bearish and below -0.40
    oversold (-0.60 extremely), with crossings of zero and of the 0.40 lines as signals.

    The formula is as follows:

    - Signed Volume = sign(Close - Close(t-1)) * Volume
    - VZO = EMA(Signed Volume, window) / EMA(Volume, window)

    Also known as: VZO, Volume Zone Oscillator.

    Reference: Khalil, W. and Steckler, D. (2011). "In The Volume Zone." Technical Analysis
    of Stocks & Commodities, 29(5).

    Referenced against:
    - finta, VZO: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1521

    Notes:
        - The VZO is a decimal between -1 and 1; Khalil and Steckler and finta multiply it by
          100, so their zones are at 5, 40 and 60.
        - Both EMAs are seeded with the Simple Moving Average of their first values and run
          over the same periods, from the second one (the first has no change), so the VZO
          stays within its bounds and starts on period `window` + 1. finta seeds with
          pandas' adjusted weights from the first period, which only changes the warm-up.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for both EMAs.

    Returns:
        pd.Series: Volume Zone Oscillator values, as decimals between -1 and 1.
    """
    direction = np.sign(prices_close.diff())
    signed_volume = direction * volumes

    average_volume = get_exponential_moving_average(
        volumes.where(direction.notna()), window
    )

    return get_exponential_moving_average(signed_volume, window) / average_volume.where(
        average_volume != 0
    )


def get_cumulative_force_index(
    prices_close: pd.Series, volumes: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Cumulative Force Index (CFI): a running total of the price change times the volume, which shows
    the force behind a trend.

    The Cumulative Force Index is the running total of Elder's Force Index: each period's
    change in price times its volume, smoothed with an EMA, added up over time. Where
    On-Balance Volume adds the whole volume of an up close however small the move, the
    force weights the volume with the size of the move, so a heavy session that barely moves
    price adds little. With a window of 1 there is no smoothing and it is the Weighted
    On-Balance Volume.

    The formula is as follows:

    - Force(t) = (Close(t) - Close(t-1)) * Volume(t)
    - CFI(t) = CFI(t-1) + EMA(Force, window)(t)

    Also known as: CFI, Cumulative Force Index, Weighted On-Balance Volume (WOBV, with a
    window of 1).

    Reference: Elder, A. (1993). "Trading for a Living: Psychology, Trading Tactics, Money
    Management." John Wiley & Sons, for the Force Index. The cumulative form and the Weighted
    On-Balance Volume are defined in finta (Peerchemist).

    Referenced against:
    - finta, CFI: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1585
    - finta, WOBV: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1506

    Notes:
        - With a window of 1 it agrees with finta's WOBV. With a larger window it agrees with finta's
          CFI apart from the seed: the EMA is seeded with the Simple Moving Average of its
          first `window` values (see `overlap_model.get_exponential_moving_average`), while
          finta averages from the first value (pandas' adjusted weights). The EMAs converge, but the
          running total keeps the difference of the warm-up as a constant offset.
        - The running total starts at the first smoothed force, so its level depends on the
          first date of the data; only its direction and its divergences from price carry
          meaning.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for the EMA of the force.

    Returns:
        pd.Series: Cumulative Force Index values, in price times volume.
    """
    observed = prices_close.notna() & volumes.notna()
    force = prices_close[observed].diff() * volumes[observed]

    return (
        get_exponential_moving_average(force, window)
        .cumsum()
        .reindex(prices_close.index)
    )


def get_price_volume(
    prices_close: pd.Series, volumes: pd.Series, signed: bool = False
) -> pd.Series:
    """
    Calculate the Price-Volume (dollar volume): the close times the volume, the amount of money that traded in each
    period.

    The Price-Volume is the value traded in a period: the close times the volume. Unlike the
    number of shares, it can be compared across stocks with very different prices and is the
    usual measure of liquidity for position sizing. Signed, it carries the direction of the
    close, as a money flow per period.

    The formula is as follows:

    - Price-Volume = Close * Volume
    - Signed Price-Volume = sign(Close - Close(t-1)) * Close * Volume

    Also known as: PVOL, dollar volume, traded value.

    Reference: A common liquidity measure; see Amihud, Y. (2002). "Illiquidity and Stock
    Returns: Cross-Section and Time-Series Effects." Journal of Financial Markets, 5(1),
    31-56, which scales returns by the dollar volume.

    Referenced against:
    - pandas-ta-classic, pvol: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/pvol.py#L17

    Notes:
        - It agrees with pandas-ta-classic's `pvol`, also in its signed form, where the first
          period has no change and therefore no signed value.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        signed (bool): Whether to sign the value with the direction of the close. Defaults to
            False.

    Returns:
        pd.Series: Price-Volume values, in the currency of the prices.
    """
    price_volume = prices_close * volumes

    if signed:
        return price_volume * np.sign(prices_close.diff())

    return price_volume


def get_price_volume_rank(prices_close: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Calculate the Price Volume Rank (PVR): a code from 1 (a rising price on rising volume) to 4 (a falling price on
    falling volume), which shows whether volume confirms a move.

    The Price Volume Rank classifies each period by the direction of the close and of the
    volume, from the most bullish to the most bearish combination: a rising price on rising
    volume (1), a rising price on falling volume (2), a falling price on rising volume (3)
    and a falling price on falling volume (4). An unchanged close or volume counts as
    rising. A smoothed rank below 2.5 is read as bullish, above it as bearish.

    The formula is as follows:

    - PVR = 1 if Close >= Close(t-1) and Volume >= Volume(t-1)
    - PVR = 2 if Close >= Close(t-1) and Volume < Volume(t-1)
    - PVR = 3 if Close < Close(t-1) and Volume >= Volume(t-1)
    - PVR = 4 if Close < Close(t-1) and Volume < Volume(t-1)

    Also known as: PVR, Price Volume Rank.

    Reference: Fulks Maxwell Laboratories (fmlabs), "Price Volume Rank," Technical Indicators
    Reference.

    Referenced against:
    - pandas-ta-classic, pvr: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/pvr.py#L12

    Notes:
        - It agrees with pandas-ta-classic's `pvr`. The first period has no change and no rank.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.

    Returns:
        pd.Series: Price Volume Rank values (1, 2, 3 or 4).
    """
    price_change = prices_close.diff()
    volume_change = volumes.diff()

    rank = pd.Series(np.nan, index=prices_close.index)
    rank[(price_change >= 0) & (volume_change >= 0)] = 1
    rank[(price_change >= 0) & (volume_change < 0)] = 2
    rank[(price_change < 0) & (volume_change >= 0)] = 3
    rank[(price_change < 0) & (volume_change < 0)] = 4

    return rank


def get_volume_profile(
    prices_close: pd.Series, volumes: pd.Series, bins: int
) -> pd.DataFrame:
    """
    Calculate the Volume Profile (Volume by Price): how much volume traded at each price level, which shows the
    prices the market accepted most.

    The Volume Profile is not a series over time but a histogram: it divides the range of
    the closes into price bins of equal width and adds up the volume traded at a close in
    each bin, split into the volume of up closes and that of down closes. Bins with a lot of
    volume are prices at which many positions were opened, which tend to act as support or
    resistance when price returns to them; the bin with the most volume is the point of
    control.

    The formula is as follows:

    - Bin edges = Min(Close) + k * (Max(Close) - Min(Close)) / bins, for k = 0 to bins
    - Up Volume = Sum(Volume of the periods with Close > Close(t-1) and Close in the bin)
    - Down Volume = Sum(Volume of the periods with Close < Close(t-1) and Close in the bin)
    - Total Volume = Sum(Volume of all periods with Close in the bin)

    Also known as: VP, Volume by Price, Volume at Price, Market Profile (in its volume form).

    Reference: Steidlmayer, J.P. and Koy, K. (1986). "Markets and Market Logic." Porcupine
    Press, for the Market Profile; StockCharts ChartSchool, "Volume by Price," for the
    volume form.

    Referenced against:
    - pandas-ta-classic, vp: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/vp.py#L13

    Notes:
        - The profile describes the whole series, so it looks at every close in it; it is
          meant for the period between a start and an end date, not for a backtest.
        - The bins are those of pandas-ta-classic's `vp` sorted by close (by price),
          with the exact range of the closes as edges and every bin kept, also those without
          volume. Its default, unsorted, splits the series into periods of time
          instead. Its total is the up plus the down volume; here the Total Volume also
          includes the volume of periods with an unchanged close, and of the first period,
          which has no change.
        - A close on an inner edge belongs to the lower bin, and the lowest close to the
          first bin.
        - A constant close has no range to divide: every edge equals that close and all the
          volume is in the first bin, the others are 0. Without any close and volume, the
          prices are NaN and every bin has a volume of 0.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        bins (int): Number of price bins of equal width.

    Returns:
        pd.DataFrame: A row per price bin, from the lowest prices to the highest, with its
        Low Price, High Price, Up Volume, Down Volume and Total Volume.
    """
    observed = prices_close.notna() & volumes.notna()
    close, volume = prices_close[observed], volumes[observed]
    direction = np.sign(close.diff())

    edges = np.linspace(close.min(), close.max(), bins + 1)
    if close.empty or not np.all(np.diff(edges) > 0):
        # Without a range (or one of rounding only) the edges coincide and no bin can be
        # cut: a constant close puts all the volume in the first bin, and without closes
        # every bin is empty.
        price_bin = pd.Series(1, index=close.index)
    else:
        price_bin = pd.cut(close, edges, labels=False, include_lowest=True) + 1

    profile = (
        pd.DataFrame(
            {
                "Up Volume": volume.where(direction > 0, 0.0),
                "Down Volume": volume.where(direction < 0, 0.0),
                "Total Volume": volume,
            }
        )
        .groupby(price_bin)
        .sum()
    )

    profile = profile.reindex(range(1, bins + 1), fill_value=0.0)
    profile.insert(0, "Low Price", edges[:-1])
    profile.insert(1, "High Price", edges[1:])
    profile.index.name = "Price Bin"

    return profile


def get_buying_selling_pressure(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
    smooth_window: int,
) -> pd.DataFrame:
    """
    Calculate the Buying and Selling Pressure (BASP), which shows whether buyers or sellers control the market.

    It compares the distance of the close above the low (buying) and of the high above the close (selling) with
    their averages, weighted by volume.

    The Buying Pressure is the distance of the close above the low, the Selling Pressure the
    distance of the high above the close. Each is divided by its own EMA, so it shows whether
    the pressure is larger than usual, and multiplied by the volume relative to its EMA, so a
    session with heavy volume weighs more. A short EMA smooths both lines. Buying pressure
    above selling pressure points to accumulation, and a crossing of the two to a change in
    who is in control.

    The formula is as follows:

    - Buying Pressure (raw) = Close - Low, Selling Pressure (raw) = High - Close
    - Relative Volume = Volume / EMA(Volume, window)
    - Buying Pressure = EMA(Buying Pressure (raw) / EMA(Buying Pressure (raw), window) *
      Relative Volume, smooth_window)
    - Selling Pressure = EMA(Selling Pressure (raw) / EMA(Selling Pressure (raw), window) *
      Relative Volume, smooth_window)

    Also known as: BASP, BASPN (normalized Buying and Selling Pressure).

    Reference: BASP and BASPN are defined in finta (Peerchemist), which cites no primary
    source. The buying pressure resembles that of Larry Williams' Ultimate Oscillator
    (Williams, L.R. (1985). "The Ultimate Oscillator." Technical Analysis of Stocks &
    Commodities, 3(4)), which measures it from the lower of the low and the previous close.

    Referenced against:
    - finta, BASP: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1671
    - finta, BASPN: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1691

    Notes:
        - A `smooth_window` of 1 gives finta's BASP and the default of 20 its BASPN.
          Every EMA is seeded with the Simple Moving Average of its first values (see
          `overlap_model.get_exponential_moving_average`), while finta averages from the
          first value (pandas' adjusted weights), so the two differ during the warm-up and converge
          after it.
        - Both lines are ratios around 1 (the usual pressure on the usual volume).

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for the EMAs of the pressure and the volume.
        smooth_window (int): Number of periods for the EMA that smooths both lines.

    Returns:
        pd.DataFrame: The Buying Pressure and Selling Pressure as columns.
    """
    buying_pressure = prices_close - prices_low
    selling_pressure = prices_high - prices_close

    average_volume = get_exponential_moving_average(volumes, window)
    relative_volume = volumes / average_volume.where(average_volume != 0)

    pressures = {}
    for name, pressure in (
        ("Buying Pressure", buying_pressure),
        ("Selling Pressure", selling_pressure),
    ):
        average_pressure = get_exponential_moving_average(pressure, window)
        pressures[name] = get_exponential_moving_average(
            pressure / average_pressure.where(average_pressure != 0) * relative_volume,
            smooth_window,
        )

    return pd.DataFrame(pressures)


def get_archer_on_balance_volume(
    prices_close: pd.Series,
    volumes: pd.Series,
    short_window: int,
    long_window: int,
    extreme_window: int,
    run_window: int,
) -> pd.DataFrame:
    """
    Calculate the Archer On-Balance Volume (AOBV): a fast and a slow average of the On-Balance Volume and whether
    the volume trend is up or down.

    The Archer On-Balance Volume adds a fast and a slow EMA of On-Balance Volume and two
    flags for the trend of those averages, so OBV can be read like a moving average
    crossover system. A "long run" is a period in which the fast average has risen over the
    last `run_window` periods while the slow one rose or fell (a rising OBV trend or a
    potential bottom); a "short run" is one in which the fast average fell (a falling trend
    or a potential top). The rolling minimum and maximum of OBV mark its recent range.

    The formula is as follows:

    - OBV = Cumulative Sum(sign(Close - Close(t-1)) * Volume)
    - Fast Average = EMA(OBV, short_window), Slow Average = EMA(OBV, long_window)
    - Minimum = Min(OBV, extreme_window), Maximum = Max(OBV, extreme_window)
    - Long Run = 1 if Fast Average(t) > Fast Average(t - run_window) and Slow Average(t) !=
      Slow Average(t - run_window), else 0
    - Short Run = 1 if Fast Average(t) < Fast Average(t - run_window) and Slow Average(t) !=
      Slow Average(t - run_window), else 0

    Also known as: AOBV, Archer OBV.

    Reference: Archer, "Archer On-Balance Volume" (a TradingView script without a published
    paper), built on Granville, J.E. (1963). "Granville's New Key to Stock Market Profits."
    Prentice-Hall.

    Referenced against:
    - pandas-ta-classic, aobv: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volume/aobv.py#L17

    Notes:
        - It agrees with pandas-ta-classic's `aobv` (whose EMAs are also seeded with a Simple
          Moving Average), except that its On-Balance Volume starts at the first period's
          volume instead of 0, as TA-Lib's: the OBV, its minimum, maximum and averages are
          that volume higher there, while the flags are equal.
        - The flags are NaN until both averages have `run_window` earlier values.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        short_window (int): Number of periods for the fast EMA of OBV.
        long_window (int): Number of periods for the slow EMA of OBV.
        extreme_window (int): Number of periods for the rolling minimum and maximum of OBV.
        run_window (int): Number of periods over which the averages must rise or fall.

    Returns:
        pd.DataFrame: The On-Balance Volume, its Minimum and Maximum, the Fast and Slow
        Average and the Long Run and Short Run flags (1 or 0) as columns.
    """
    on_balance_volume = get_on_balance_volume(prices_close, volumes)
    fast_average = get_exponential_moving_average(on_balance_volume, short_window)
    slow_average = get_exponential_moving_average(on_balance_volume, long_window)

    fast_change = fast_average.diff(run_window)
    slow_change = slow_average.diff(run_window)
    known = fast_change.notna() & slow_change.notna()
    slow_moved = slow_change != 0

    return pd.DataFrame(
        {
            "On-Balance Volume": on_balance_volume,
            "Minimum": on_balance_volume.rolling(window=extreme_window).min(),
            "Maximum": on_balance_volume.rolling(window=extreme_window).max(),
            "Fast Average": fast_average,
            "Slow Average": slow_average,
            "Long Run": ((fast_change > 0) & slow_moved).astype(float).where(known),
            "Short Run": ((fast_change < 0) & slow_moved).astype(float).where(known),
        }
    )
