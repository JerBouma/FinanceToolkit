"""Candlestick Pattern Module"""

__docformat__ = "google"

import pandas as pd


def _validate_ohlc(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> None:
    """
    Validate that every OHLC input is a pandas Series, raising a TypeError otherwise. Shared
    by every candlestick pattern function in this module.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Raises:
        TypeError: If any of the arguments is not a pandas Series.
    """
    if not (
        isinstance(prices_open, pd.Series)
        and isinstance(prices_high, pd.Series)
        and isinstance(prices_low, pd.Series)
        and isinstance(prices_close, pd.Series)
    ):
        raise TypeError(
            "prices_open, prices_high, prices_low and prices_close must be pandas Series."
        )


def get_doji(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    threshold: float = 0.1,
) -> pd.Series:
    """
    Detect the Doji candlestick pattern for a given price series.

    A Doji forms when a period opens and closes at (almost) the same price, leaving a real
    body that is negligible relative to the period's total trading range. It reflects
    indecision between buyers and sellers and, especially after a sustained trend, is watched
    as an early warning that the trend may be losing momentum.

    The formula is a follows:

    - Doji = |Close — Open| <= threshold * (High — Low)

    Also known as: Doji, Doji star.

    Reference: Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York
    Institute of Finance.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        threshold (float): The maximum size of the real body, expressed as a fraction of the
            period's high-low range, for the period to still qualify as a Doji. Defaults to
            0.1 (i.e. the real body is at most 10% of the total range).

    Returns:
        pd.Series: Boolean Series, True where the period is a Doji.

    Raises:
        TypeError: If any of the price arguments is not a pandas Series.
    """
    _validate_ohlc(prices_open, prices_high, prices_low, prices_close)

    real_body = (prices_close - prices_open).abs()
    price_range = prices_high - prices_low

    return real_body <= threshold * price_range


def get_bullish_engulfing(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.Series:
    """
    Detect the Bullish Engulfing candlestick pattern for a given price series.

    A Bullish Engulfing pattern forms across two periods: a down (bearish) period followed
    by an up (bullish) period whose real body fully engulfs — opens below and closes above —
    the prior period's real body. Appearing after a downtrend, it signals that buyers have
    forcefully overwhelmed the selling pressure that dominated the prior period, and is
    watched as a potential bullish reversal signal.

    The formula is a follows:

    - Prior period bearish: Close(t-1) < Open(t-1)
    - Current period bullish: Close(t) > Open(t)
    - Current body engulfs prior body: Open(t) <= Close(t-1) and Close(t) >= Open(t-1)
    - Bullish Engulfing = all of the above are true

    Also known as: Bullish Engulfing, Engulfing Bullish Line.

    Reference: Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York
    Institute of Finance.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: Boolean Series, True where the period completes a Bullish Engulfing
            pattern (i.e. the engulfing period itself, not the engulfed one).

    Raises:
        TypeError: If any of the price arguments is not a pandas Series.
    """
    _validate_ohlc(prices_open, prices_high, prices_low, prices_close)

    prior_open = prices_open.shift(1)
    prior_close = prices_close.shift(1)

    prior_bearish = prior_close < prior_open
    current_bullish = prices_close > prices_open
    engulfs_prior_body = (prices_open <= prior_close) & (prices_close >= prior_open)

    return prior_bearish & current_bullish & engulfs_prior_body


def get_bearish_engulfing(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.Series:
    """
    Detect the Bearish Engulfing candlestick pattern for a given price series.

    A Bearish Engulfing pattern forms across two periods: an up (bullish) period followed by
    a down (bearish) period whose real body fully engulfs — opens above and closes below —
    the prior period's real body. Appearing after an uptrend, it signals that sellers have
    forcefully overwhelmed the buying pressure that dominated the prior period, and is
    watched as a potential bearish reversal signal.

    The formula is a follows:

    - Prior period bullish: Close(t-1) > Open(t-1)
    - Current period bearish: Close(t) < Open(t)
    - Current body engulfs prior body: Open(t) >= Close(t-1) and Close(t) <= Open(t-1)
    - Bearish Engulfing = all of the above are true

    Also known as: Bearish Engulfing, Engulfing Bearish Line.

    Reference: Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York
    Institute of Finance.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: Boolean Series, True where the period completes a Bearish Engulfing
            pattern (i.e. the engulfing period itself, not the engulfed one).

    Raises:
        TypeError: If any of the price arguments is not a pandas Series.
    """
    _validate_ohlc(prices_open, prices_high, prices_low, prices_close)

    prior_open = prices_open.shift(1)
    prior_close = prices_close.shift(1)

    prior_bullish = prior_close > prior_open
    current_bearish = prices_close < prices_open
    engulfs_prior_body = (prices_open >= prior_close) & (prices_close <= prior_open)

    return prior_bullish & current_bearish & engulfs_prior_body


def get_hammer(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    body_threshold: float = 0.3,
    lower_wick_multiplier: float = 2.0,
) -> pd.Series:
    """
    Detect the Hammer candlestick pattern for a given price series.

    A Hammer forms when a period has a small real body positioned near the top of its
    trading range, a long lower wick and little to no upper wick — the visual result of
    sellers pushing price sharply lower during the period before buyers step in and drive it
    back up to close near the open. Appearing after a downtrend, it is watched as a potential
    bullish reversal signal.

    The formula is a follows:

    - Real Body = |Close — Open|
    - Range = High — Low
    - Lower Wick = Min(Open, Close) — Low
    - Upper Wick = High — Max(Open, Close)
    - Small body: Real Body <= body_threshold * Range
    - Long lower wick: Lower Wick >= lower_wick_multiplier * Real Body
    - Small upper wick: Upper Wick <= Real Body
    - Hammer = all of the above are true (and Range > 0)

    Also known as: Hammer.

    Notes:
        - The same body/wick shape appearing after an uptrend (rather than a downtrend) is
          conventionally called a Hanging Man instead — this function only evaluates the
          candle's shape, not the prevailing trend, so callers wanting to distinguish the two
          should combine the result with their own trend context (e.g. price relative to a
          moving average).

    Reference: Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York
    Institute of Finance.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        body_threshold (float): The maximum size of the real body, expressed as a fraction of
            the period's high-low range, for the period to still qualify as having a "small"
            body. Defaults to 0.3.
        lower_wick_multiplier (float): The minimum size of the lower wick, expressed as a
            multiple of the real body, for the period to qualify as having a "long" lower
            wick. Defaults to 2.0.

    Returns:
        pd.Series: Boolean Series, True where the period is a Hammer.

    Raises:
        TypeError: If any of the price arguments is not a pandas Series.
    """
    _validate_ohlc(prices_open, prices_high, prices_low, prices_close)

    real_body = (prices_close - prices_open).abs()
    price_range = prices_high - prices_low

    body_top = pd.concat([prices_open, prices_close], axis=1).max(axis=1)
    body_bottom = pd.concat([prices_open, prices_close], axis=1).min(axis=1)

    lower_wick = body_bottom - prices_low
    upper_wick = prices_high - body_top

    small_body = real_body <= body_threshold * price_range
    long_lower_wick = lower_wick >= lower_wick_multiplier * real_body
    small_upper_wick = upper_wick <= real_body
    has_range = price_range > 0

    return small_body & long_lower_wick & small_upper_wick & has_range
