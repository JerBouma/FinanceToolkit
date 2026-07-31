"""Breadth Module"""

__docformat__ = "google"

import pandas as pd


def get_mcclellan_oscillator(
    prices_close: pd.Series, short_ema_window: int, long_ema_window: int
) -> pd.Series:
    """
    Calculate the McClellan Oscillator for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_ema_window (int): Number of periods for the short-term EMA.
        long_ema_window (int): Number of periods for the long-term EMA.

    Returns:
        pd.Series: McClellan Oscillator values.
    """
    advancers_decliners = get_advancers_decliners(prices_close)
    short_ema = advancers_decliners.ewm(
        span=short_ema_window, min_periods=1, adjust=False
    ).mean()
    long_ema = advancers_decliners.ewm(
        span=long_ema_window, min_periods=1, adjust=False
    ).mean()

    return short_ema - long_ema


def get_advancers_decliners(prices_close: pd.Series) -> pd.Series:
    """
    Calculate the difference between advancers and decliners for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: Advancers - Decliners values.
    """
    advancers = prices_close.where(prices_close > prices_close.shift(1), 0)
    decliners = -prices_close.where(prices_close < prices_close.shift(1), 0)

    return advancers - decliners


def get_on_balance_volume(prices_close: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV) of a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.

    Returns:
        pd.Series: OBV values.
    """
    price_diff = prices_close.diff(1)
    obv = (price_diff / abs(price_diff)) * volumes

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

    The formula is a follows:

    - Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low)

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

    The formula is a follows:

    - TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

    Also known as: Arms Index, TRIN.

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
    Calculate the New Highs — New Lows for a universe of tickers.

    New Highs — New Lows measures the number of tickers reaching a new high over the
    specified window minus the number of tickers reaching a new low over the same
    window. It is a cross-sectional market breadth indicator, meaning it is calculated
    across the columns (tickers) of the provided data rather than per ticker over time.

    The formula is a follows:

    - New Highs — New Lows = (Number of tickers at a window-period high) — (Number of tickers at a window-period low)

    Also known as: new highs minus new lows, record high percent.

    Args:
        prices_close (pd.DataFrame): DataFrame of closing prices with tickers as columns.
        window (int): Number of periods to consider for the new high / new low lookback.

    Returns:
        pd.Series: New Highs — New Lows values.
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
    short_ema = adl.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = adl.ewm(span=long_window, min_periods=1, adjust=False).mean()

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

    The formula is a follows:

    - Money Flow Multiplier = ((Close — Low) — (High — Close)) / (High — Low)
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

    The formula is a follows:

    - Distance Moved = (High(t) + Low(t)) / 2 — (High(t-1) + Low(t-1)) / 2
    - Box Ratio = (Volume / volume_divisor) / (High — Low)
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

    The formula is a follows:

    - Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) — 1)) if Volume(t) < Volume(t-1)
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

    The formula is a follows:

    - Index(t) = Index(t-1) * (1 + (Close(t) / Close(t-1) — 1)) if Volume(t) > Volume(t-1)
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
