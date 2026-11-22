"""Pattern Module: candlestick patterns and indicator signals"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.technicals.momentum import momentum_model
from financetoolkit.technicals.overlap import overlap_model
from financetoolkit.technicals.volatility import volatility_model

# TA-Lib's default candle settings (ta_global.c): what a candle part is measured against, over how many
# previous candles it is averaged and the factor applied. "Shadows" averages are halved, as in TA-Lib, so they
# measure one shadow against the average of a candle's two.
CANDLE_SETTINGS = {
    "BodyLong": ("RealBody", 10, 1.0),
    "BodyVeryLong": ("RealBody", 10, 3.0),
    "BodyShort": ("RealBody", 10, 1.0),
    "BodyDoji": ("HighLow", 10, 0.1),
    "ShadowLong": ("RealBody", 0, 1.0),
    "ShadowVeryLong": ("RealBody", 0, 2.0),
    "ShadowShort": ("Shadows", 10, 1.0),
    "ShadowVeryShort": ("HighLow", 10, 0.1),
    "Near": ("HighLow", 5, 0.2),
    "Far": ("HighLow", 5, 0.6),
    "Equal": ("HighLow", 5, 0.05),
}

# The candles TA-Lib needs before a pattern's first value (each function's _Lookback).
PATTERN_LOOKBACK = {
    "Doji": 10,
    "Hammer": 11,
    "Hanging Man": 11,
    "Inverted Hammer": 11,
    "Shooting Star": 11,
    "Engulfing": 2,
    "Harami": 11,
    "Piercing Line": 11,
    "Dark Cloud Cover": 11,
    "Morning Star": 12,
    "Evening Star": 12,
    "Three White Soldiers": 12,
    "Three Black Crows": 13,
    "Marubozu": 10,
    "Spinning Top": 10,
    "Belt-hold": 10,
    "Closing Marubozu": 10,
    "Dragonfly Doji": 10,
    "Gravestone Doji": 10,
    "High-Wave Candle": 10,
    "Long-Legged Doji": 10,
    "Long Line Candle": 10,
    "Rickshaw Man": 10,
    "Short Line Candle": 10,
    "Takuri": 10,
    "Inside Bar": 1,
    "Counterattack": 11,
    "Doji Star": 11,
    "Harami Cross": 11,
    "Homing Pigeon": 11,
    "In-Neck": 11,
    "On-Neck": 11,
    "Thrusting": 11,
    "Kicking": 11,
    "Kicking by Length": 11,
    "Matching Low": 6,
    "Separating Lines": 11,
    "Three Inside": 12,
    "Three Outside": 3,
    "Morning Doji Star": 12,
    "Evening Doji Star": 12,
}

# How far the third candle of a (doji) star, and the second candle of a dark cloud cover, must reach into the
# first candle's real body: TA-Lib's default optInPenetration.
STAR_PENETRATION = 0.3
DARK_CLOUD_PENETRATION = 0.5


def get_candle_average(candles: pd.DataFrame, setting: str) -> pd.Series:
    """
    Calculate the reference size a candle part is compared with, as TA-Lib defines it.

    The reference is the factor times the average range (real body, high-low range or the
    sum of both shadows) of the candles before the candle itself, so a candle is never
    compared with itself. A setting with an average period of 0 compares with the candle's
    own range instead (e.g. a long shadow is longer than the candle's own real body).

    Args:
        candles (pd.DataFrame): The candles, as returned by `get_candle_parts`.
        setting (str): The name of the setting in `CANDLE_SETTINGS`, e.g. "BodyLong".

    Returns:
        pd.Series: The reference size per candle.
    """
    range_type, period, factor = CANDLE_SETTINGS[setting]

    if range_type == "RealBody":
        ranges = candles["Body"]
    elif range_type == "HighLow":
        ranges = candles["Range"]
    else:
        ranges = candles["Upper Shadow"] + candles["Lower Shadow"]

    average = ranges if period == 0 else ranges.shift(1).rolling(period).mean()

    return factor * average / (2.0 if range_type == "Shadows" else 1.0)


def get_candle_parts(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.DataFrame:
    """
    Split each candle into its real body and shadows.

    Args:
        prices_open (pd.Series): The open prices.
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.

    Returns:
        pd.DataFrame: The open, high, low and close with whether the candle is white (a close at
        or above the open, as TA-Lib counts it), its colour (1 white, -1 black), the real body,
        both shadows and the high-low range.
    """
    white = prices_close >= prices_open
    body_top = prices_close.where(white, prices_open)
    body_bottom = prices_open.where(white, prices_close)

    return pd.DataFrame(
        {
            "Open": prices_open,
            "High": prices_high,
            "Low": prices_low,
            "Close": prices_close,
            "White": white,
            "Color": np.where(white, 1.0, -1.0),
            "Body": (prices_close - prices_open).abs(),
            "Upper Shadow": prices_high - body_top,
            "Lower Shadow": body_bottom - prices_low,
            "Range": prices_high - prices_low,
        }
    )


def get_candlestick_patterns(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Recognise 41 candlestick patterns, following the definitions of TA-Lib's CDL functions.

    Each pattern is 1 on a candle that completes it as a bullish pattern, -1 as a bearish
    pattern and 0 otherwise, which is TA-Lib's +100 / -100 divided by 100. TA-Lib's weaker
    form of the engulfing, harami and harami cross patterns, where one end of the two real
    bodies is equal, is 0.8 (TA-Lib's 80). The dojis (doji, dragonfly, gravestone and
    long-legged doji, rickshaw man and takuri) are indecision rather than a direction and are 1
    when present, as in TA-Lib; a marubozu, a spinning top, a belt-hold, a closing marubozu, a
    high-wave, long line and short line candle and an inside bar take the colour of the candle
    (1 white, -1 black).

    Sizes are relative, as in TA-Lib (ta_global.c): a real body is long when it is longer than
    the average real body of the 10 previous candles and short when it is shorter; it is a
    doji's body when it is at most 10% of the average high-low range of the 10 previous
    candles; a shadow is very short when it is shorter than 10% of that range, short when it
    is shorter than half the average sum of both shadows of the 10 previous candles (the
    average shadow), long when it is longer than the candle's own real body and very long
    when it is longer than twice that body; "near" is within 20% and "far" at least 60% of the
    average high-low range of the 5 previous candles, and two prices are "equal" when they are
    within 5% of that range.

    The patterns and what they need:

    - Doji: a real body of a doji.
    - Hammer (bullish): a short real body, a long lower shadow, a very short upper shadow and
      the body near or below the previous low.
    - Hanging Man (bearish): the hammer's shape with the body near or above the previous high.
    - Inverted Hammer (bullish): a short real body that gaps down, a long upper shadow and a
      very short lower shadow.
    - Shooting Star (bearish): a short real body that gaps up, a long upper shadow and a very
      short lower shadow.
    - Engulfing: a real body that engulfs the previous one of the opposite colour.
    - Harami: a short real body inside the previous long real body, opposite to the colour of
      that body.
    - Piercing Line (bullish): a long black candle followed by a long white candle that opens
      below the previous low and closes above the middle of the previous real body.
    - Dark Cloud Cover (bearish): a long white candle followed by a black candle that opens
      above the previous high and closes below the middle of the previous real body.
    - Morning Star (bullish) and Evening Star (bearish): a long real body, a short real body
      that gaps away from it and a real body of the opposite colour that closes at least 30%
      into the first body.
    - Three White Soldiers (bullish): three white candles with higher closes and very short
      upper shadows, each opening within or near the previous body and none far shorter than
      the one before.
    - Three Black Crows (bearish): a white candle followed by three declining black candles
      with very short lower shadows, each opening within the previous body.
    - Marubozu: a long real body with very short shadows.
    - Spinning Top: a short real body with both shadows longer than the body.
    - Belt-hold: a long real body with a very short shadow on the side of the open, a white
      candle opening on its low (bullish) or a black candle opening on its high (bearish).
    - Closing Marubozu: a long real body with a very short shadow on the side of the close.
    - Dragonfly Doji: a doji with a very short upper shadow and a lower shadow that is not.
    - Gravestone Doji: a doji with a very short lower shadow and an upper shadow that is not.
    - High-Wave Candle: a short real body with very long upper and lower shadows.
    - Long-Legged Doji: a doji with at least one long shadow.
    - Long Line Candle: a long real body with short upper and lower shadows.
    - Rickshaw Man: a doji with two long shadows and the body near the middle of the
      high-low range.
    - Short Line Candle: a short real body with short upper and lower shadows.
    - Takuri: a doji with a very short upper shadow and a very long lower shadow.
    - Inside Bar: a high below and a low above the previous candle's, the price-action
      pattern of pandas-ta-classic's cdl_inside (not part of TA-Lib's CDL functions).
    - Counterattack: two long real bodies of opposite colour with equal closes, in the
      direction of the second candle.
    - Doji Star: a long real body followed by a doji that gaps away from it (up after a white
      body, down after a black one), opposite to the colour of the first candle.
    - Harami Cross: a doji inside the previous long real body, opposite to the colour of that
      body.
    - Homing Pigeon (bullish): a long black candle followed by a short black candle whose real
      body lies inside it (a lower open and a higher close).
    - In-Neck (bearish): a long black candle followed by a white candle that opens below its
      low and closes at or just above its close (within "equal").
    - On-Neck (bearish): the same two candles with the white close equal to the previous low.
    - Thrusting (bearish): the same two candles with the white close above the previous close
      by more than "equal" but not above the middle of the previous real body.
    - Kicking: a marubozu (a long real body with very short shadows) followed by a marubozu
      of the opposite colour that gaps away from it (above the high of a black one, below the
      low of a white one), in the direction of the second candle.
    - Kicking by Length: the kicking pattern in the direction of the longer of the two
      marubozus (the first when both are equally long, as in TA-Lib).
    - Matching Low (bullish): two black candles with equal closes.
    - Separating Lines: two candles of opposite colour with equal opens, the second a long
      real body with a very short shadow on the side of the open (a belt-hold), in the
      direction of the second candle.
    - Three Inside (Up or Down): a harami (a short real body strictly inside the previous long
      real body) followed by a candle of the opposite colour to the first that closes beyond
      the first candle's open, opposite to the colour of the first candle.
    - Three Outside (Up or Down): an engulfing pattern followed by a candle that closes beyond
      the close of the engulfing candle, in the direction of the engulfing candle.
    - Morning Doji Star (bullish) and Evening Doji Star (bearish): the morning and evening
      star with a doji as the middle candle.

    Reference: Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York Institute
    of Finance; Bulkowski, T.N. (2008). "Encyclopedia of Candlestick Charts." Wiley. The
    thresholds are TA-Lib's (github.com/TA-Lib/ta-lib, src/ta_func/ta_CDL*.c).

    TA-Lib: the CDL functions, the reference implementation (https://ta-lib.org/functions/);
    pandas-ta-classic: cdl_inside for the inside bar
    (https://github.com/xgboosted/pandas-ta-classic).

    Notes:
        - A pattern is empty (NaN) until TA-Lib's lookback has passed: the candles its
          averages and the pattern itself need (3 to 13 candles, 1 for the inside bar), so a
          ticker is evaluated from the same candle on as TA-Lib. pandas-ta-classic's inside
          bar is 0 instead of empty on the first candle.
        - TA-Lib marks the gravestone doji, which Nison reads as bearish after a rise, as 1
          like every doji; so does this function.
        - A pattern only describes the candles; the trend before it, which Nison requires for
          e.g. a hammer (after a decline) and a hanging man (after a rise), is not checked, as
          in TA-Lib. The direction is therefore TA-Lib's: a counterattack, a kicking and
          separating lines follow the second candle, while Nison reads separating lines as a
          continuation of the trend, and the in-neck, on-neck and thrusting are bearish
          continuation patterns (Nison, Bulkowski) that TA-Lib marks -1 whatever the trend.
        - The doji stars use the same 30% penetration as the stars (STAR_PENETRATION, TA-Lib's
          default optInPenetration); Nison only asks for a close well into the first body.

    Args:
        prices_open (pd.Series): The open prices.
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.
        patterns (list[str] | None, optional): The patterns to return, names of
            `PATTERN_LOOKBACK` in the order given. Defaults to None, which returns all 41.

    Returns:
        pd.DataFrame: The patterns as columns, with 1 (bullish), -1 (bearish), 0.8 / -0.8 (the
        weaker engulfing, harami and harami cross) or 0 per candle.
    """
    unknown = set(patterns or []) - set(PATTERN_LOOKBACK)
    if unknown:
        raise ValueError(
            f"Unknown pattern(s) {sorted(unknown)}, choose from {list(PATTERN_LOOKBACK)}."
        )

    candles = get_candle_parts(prices_open, prices_high, prices_low, prices_close)
    o, h, low, c = candles["Open"], candles["High"], candles["Low"], candles["Close"]
    white, color, body = candles["White"], candles["Color"], candles["Body"]
    upper, lower = candles["Upper Shadow"], candles["Lower Shadow"]

    body_long = get_candle_average(candles, "BodyLong")
    body_short = get_candle_average(candles, "BodyShort")
    body_doji = get_candle_average(candles, "BodyDoji")
    shadow_long = get_candle_average(candles, "ShadowLong")
    shadow_very_short = get_candle_average(candles, "ShadowVeryShort")
    near = get_candle_average(candles, "Near")
    far = get_candle_average(candles, "Far")
    shadow_very_long = get_candle_average(candles, "ShadowVeryLong")
    shadow_short = get_candle_average(candles, "ShadowShort")
    equal = get_candle_average(candles, "Equal")

    def previous(series: pd.Series, candles_back: int = 1) -> pd.Series:
        return series.shift(candles_back)

    o1, h1, l1, c1 = previous(o), previous(h), previous(low), previous(c)
    o2, c2 = previous(o, 2), previous(c, 2)
    white1, white2 = previous(white).eq(True), previous(white, 2).eq(True)
    black1, black2 = previous(white).eq(False), previous(white, 2).eq(False)
    body1, body2 = previous(body), previous(body, 2)

    # The hammer family: a small body at one end of a long shadow.
    small_body = body < body_short
    doji = body <= body_doji
    midpoint = low + (h - low) / 2
    hammer_shape = small_body & (lower > shadow_long) & (upper < shadow_very_short)
    inverted_shape = small_body & (upper > shadow_long) & (lower < shadow_very_short)
    near1 = previous(near)
    gap_down = (o < o1) & (o < c1) & (c < o1) & (c < c1)
    gap_up = (o > o1) & (o > c1) & (c > o1) & (c > c1)

    # Engulfing and harami: TA-Lib's 100 when the bodies differ at both ends, 80 when one end is equal.
    engulfs = (white & black1 & (((c >= o1) & (o < c1)) | ((c > o1) & (o <= c1)))) | (
        ~white & white1 & (((o >= c1) & (c < o1)) | ((o > c1) & (c <= o1)))
    )
    engulfing_strength = np.where((o != c1) & (c != o1), 1.0, 0.8)

    harami_sizes = (body1 > previous(body_long)) & (body <= body_short)
    strictly_inside = (
        ((c < c1) | (c < o1))
        & ((o < c1) | (o < o1))
        & ((c > c1) | (c > o1))
        & ((o > c1) | (o > o1))
    )
    inside = (
        ((c <= c1) | (c <= o1))
        & ((o <= c1) | (o <= o1))
        & ((c >= c1) | (c >= o1))
        & ((o >= c1) | (o >= o1))
    )
    harami_strength = np.where(strictly_inside, 1.0, np.where(inside, 0.8, 0.0))

    star_sizes = (
        (body2 > previous(body_long, 2))
        & (body1 <= previous(body_short))
        & (body > body_short)
    )
    doji_star_sizes = (
        (body2 > previous(body_long, 2))
        & (body1 <= previous(body_doji))
        & (body > body_short)
    )
    gap_up1 = (o1 > o2) & (o1 > c2) & (c1 > o2) & (c1 > c2)
    gap_down1 = (o1 < o2) & (o1 < c2) & (c1 < o2) & (c1 < c2)

    # Two candles of opposite colour, and a black candle followed by a white one that opens below its low:
    # the start of the counterattack, kicking and separating lines, and of the in-neck, on-neck and thrusting.
    opposite = white != previous(white)
    long_black1 = black1 & (body1 > previous(body_long))
    white_below_low1 = long_black1 & white & (o < l1)
    equal1 = previous(equal)

    # A marubozu as the kicking patterns need it: a long real body with very short shadows on both sides.
    marubozu = (
        (body > body_long) & (upper < shadow_very_short) & (lower < shadow_very_short)
    )
    kicking = (
        opposite
        & ((black1 & (low > h1)) | (white1 & (h < l1)))
        & previous(marubozu).eq(True)
        & marubozu
    )

    recognised = {
        "Doji": (body <= body_doji).astype(float),
        "Hammer": (hammer_shape & ((c <= l1 + near1) | (o <= l1 + near1))).astype(
            float
        ),
        "Hanging Man": -(hammer_shape & (c >= h1 - near1) & (o >= h1 - near1)).astype(
            float
        ),
        "Inverted Hammer": (inverted_shape & gap_down).astype(float),
        "Shooting Star": -(inverted_shape & gap_up).astype(float),
        "Engulfing": color * engulfing_strength * engulfs,
        "Harami": -previous(color) * harami_strength * harami_sizes,
        "Piercing Line": (
            black1
            & white
            & (o < l1)
            & (body1 > previous(body_long))
            & (body > body_long)
            & (c < o1)
            & (c > c1 + body1 * 0.5)
        ).astype(float),
        "Dark Cloud Cover": -(
            white1
            & ~white
            & (o > h1)
            & (body1 > previous(body_long))
            & (c > o1)
            & (c < c1 - body1 * DARK_CLOUD_PENETRATION)
        ).astype(float),
        "Morning Star": (
            black2
            & white
            & (o1 < o2)
            & (o1 < c2)
            & (c1 < o2)
            & (c1 < c2)
            & (c > c2 + body2 * STAR_PENETRATION)
            & star_sizes
        ).astype(float),
        "Evening Star": -(
            white2
            & ~white
            & (o1 > o2)
            & (o1 > c2)
            & (c1 > o2)
            & (c1 > c2)
            & (c < c2 - body2 * STAR_PENETRATION)
            & star_sizes
        ).astype(float),
        "Three White Soldiers": (
            white2
            & white1
            & white
            & (previous(upper, 2) < previous(shadow_very_short, 2))
            & (previous(upper) < previous(shadow_very_short))
            & (upper < shadow_very_short)
            & (c > c1)
            & (c1 > c2)
            & (o1 > o2)
            & (o1 <= c2 + previous(near, 2))
            & (o > o1)
            & (o <= c1 + near1)
            & (body1 > body2 - previous(far, 2))
            & (body > body1 - previous(far))
            & (body > body_short)
        ).astype(float),
        "Three Black Crows": -(
            previous(white, 3).eq(True)
            & black2
            & black1
            & ~white
            & (o1 < o2)
            & (o1 > c2)
            & (o < o1)
            & (o > c1)
            & (previous(h, 3) > c2)
            & (c2 > c1)
            & (c1 > c)
            & (previous(lower, 2) < previous(shadow_very_short, 2))
            & (previous(lower) < previous(shadow_very_short))
            & (lower < shadow_very_short)
        ).astype(float),
        "Marubozu": color
        * (
            (body > body_long)
            & (upper < shadow_very_short)
            & (lower < shadow_very_short)
        ),
        "Spinning Top": color * ((upper > body) & (lower > body) & small_body),
        # The open side of a belt-hold is the lower shadow of a white candle and the upper of a black one.
        "Belt-hold": color
        * ((body > body_long) & (lower.where(white, upper) < shadow_very_short)),
        "Closing Marubozu": color
        * ((body > body_long) & (upper.where(white, lower) < shadow_very_short)),
        "Dragonfly Doji": (
            doji & (upper < shadow_very_short) & (lower > shadow_very_short)
        ).astype(float),
        "Gravestone Doji": (
            doji & (lower < shadow_very_short) & (upper > shadow_very_short)
        ).astype(float),
        "High-Wave Candle": color
        * (small_body & (upper > shadow_very_long) & (lower > shadow_very_long)),
        "Long-Legged Doji": (
            doji & ((lower > shadow_long) | (upper > shadow_long))
        ).astype(float),
        "Long Line Candle": color
        * ((body > body_long) & (upper < shadow_short) & (lower < shadow_short)),
        "Rickshaw Man": (
            doji
            & (lower > shadow_long)
            & (upper > shadow_long)
            & ((o <= midpoint + near) | (c <= midpoint + near))
            & ((o >= midpoint - near) | (c >= midpoint - near))
        ).astype(float),
        "Short Line Candle": color
        * (small_body & (upper < shadow_short) & (lower < shadow_short)),
        "Takuri": (
            doji & (upper < shadow_very_short) & (lower > shadow_very_long)
        ).astype(float),
        "Inside Bar": color * ((h < h1) & (low > l1)),
        "Counterattack": color
        * (
            opposite
            & (body1 > previous(body_long))
            & (body > body_long)
            & (c <= c1 + equal1)
            & (c >= c1 - equal1)
        ),
        "Doji Star": -previous(color)
        * (
            (body1 > previous(body_long))
            & doji
            & ((white1 & gap_up) | (black1 & gap_down))
        ),
        "Harami Cross": -previous(color)
        * harami_strength
        * ((body1 > previous(body_long)) & doji),
        "Homing Pigeon": (
            long_black1 & ~white & (c > c1) & (o < o1) & (body <= body_short)
        ).astype(float),
        "In-Neck": -(white_below_low1 & (c <= c1 + equal1) & (c >= c1)).astype(float),
        "On-Neck": -(white_below_low1 & (c <= l1 + equal1) & (c >= l1 - equal1)).astype(
            float
        ),
        "Thrusting": -(
            white_below_low1 & (c > c1 + equal1) & (c <= c1 + body1 * 0.5)
        ).astype(float),
        "Kicking": color * kicking,
        # The longer of the two marubozus decides the direction.
        "Kicking by Length": pd.Series(
            np.where(body > body1, color, previous(color)), index=candles.index
        )
        * kicking,
        "Matching Low": (
            black1 & ~white & (c <= c1 + equal1) & (c >= c1 - equal1)
        ).astype(float),
        "Separating Lines": color
        * (
            opposite
            & (o <= o1 + equal1)
            & (o >= o1 - equal1)
            & (body > body_long)
            & (lower.where(white, upper) < shadow_very_short)
        ),
        "Three Inside": -previous(color, 2)
        * (
            ((c1 < c2) | (c1 < o2))
            & ((o1 < c2) | (o1 < o2))
            & ((c1 > c2) | (c1 > o2))
            & ((o1 > c2) | (o1 > o2))
            & ((white2 & ~white & (c < o2)) | (black2 & white & (c > o2)))
            & (body2 > previous(body_long, 2))
            & (body1 <= previous(body_short))
        ),
        "Three Outside": previous(color)
        * (
            (white1 & black2 & (c1 > o2) & (o1 < c2) & (c > c1))
            | (black1 & white2 & (o1 > c2) & (c1 < o2) & (c < c1))
        ),
        "Morning Doji Star": (
            black2
            & white
            & gap_down1
            & (c > c2 + body2 * STAR_PENETRATION)
            & doji_star_sizes
        ).astype(float),
        "Evening Doji Star": -(
            white2
            & ~white
            & gap_up1
            & (c < c2 - body2 * STAR_PENETRATION)
            & doji_star_sizes
        ).astype(float),
    }

    position = pd.Series(np.arange(len(candles)), index=candles.index)

    # A pattern is evaluated from the candle on which TA-Lib gives its first value; the -0.0 of a
    # negated False becomes 0.
    return pd.DataFrame(
        {
            name: (recognised[name] + 0.0).where(position >= PATTERN_LOOKBACK[name])
            for name in (patterns or PATTERN_LOOKBACK)
        }
    )


def get_heikin_ashi(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.DataFrame:
    """
    Calculate the Heikin-Ashi candles, which smooth the candles with two-period averages.

    The formula is as follows:

    - HA Close = (Open + High + Low + Close) / 4
    - HA Open = (previous HA Open + previous HA Close) / 2, starting from (Open + Close) / 2
      on the first candle
    - HA High = max(High, HA Open, HA Close)
    - HA Low = min(Low, HA Open, HA Close)

    Reference: Valcu, D. (2004). "Using the Heikin-Ashi Technique." Technical Analysis of
    Stocks & Commodities, 22(2).

    TA-Lib: HA, the reference implementation (https://ta-lib.org/functions/);
    pandas-ta-classic: ha (https://github.com/xgboosted/pandas-ta-classic).

    Notes:
        - The HA Open halves the weight of the first candle's seed every period, so it is an
          exponential average with a smoothing factor of 1/2 of the previous HA Closes. The
          values equal TA-Lib's and pandas-ta-classic's on every candle, from the first candle
          on (TA-Lib's default unstable period of HA is 0).

    Args:
        prices_open (pd.Series): The open prices.
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.

    Returns:
        pd.DataFrame: The Heikin-Ashi Open, High, Low and Close per candle.
    """
    ha_close = prices_open + prices_high + prices_low + prices_close
    ha_close = ha_close / 4.0

    # HA Open(t) = 1/2 HA Open(t-1) + 1/2 HA Close(t-1): an exponential average of the previous
    # HA Closes that starts from the first candle's (Open + Close) / 2.
    previous_close = ha_close.shift(1)
    previous_close.iloc[:1] = (prices_open.iloc[:1] + prices_close.iloc[:1]) / 2.0
    ha_open = previous_close.ewm(alpha=0.5, adjust=False).mean()

    return pd.DataFrame(
        {
            "Open": ha_open,
            "High": pd.concat([prices_high, ha_open, ha_close], axis=1).max(axis=1),
            "Low": pd.concat([prices_low, ha_open, ha_close], axis=1).min(axis=1),
            "Close": ha_close,
        }
    )


def get_williams_fractal(
    prices_high: pd.Series,
    prices_low: pd.Series,
    left_bars: int = 2,
    right_bars: int = 2,
) -> pd.DataFrame:
    """
    Mark Bill Williams' fractals: a high above the highs of the bars on either side of it (an
    up fractal) or a low below their lows (a down fractal).

    A fractal needs the `right_bars` bars after it, so it is only known `right_bars` periods
    later. It is therefore marked on the bar that confirms it, not on the bar of the high or
    low itself: the value on a date only uses the prices up to that date.

    The formula is as follows:

    - Up Fractal (t) = 1 if High (t - right_bars) > every other High from t - left_bars -
      right_bars to t, else 0
    - Down Fractal (t) = 1 if Low (t - right_bars) < every other Low from t - left_bars -
      right_bars to t, else 0

    Reference: Williams, B.M. (1995). "Trading Chaos: Applying Expert Techniques to Maximize
    Your Profits." Wiley.

    TA-Lib: FRACTAL, the reference implementation (https://ta-lib.org/functions/); finta:
    WILLIAMS_FRACTAL (https://github.com/peerchemist/finta).

    Notes:
        - This equals TA-Lib's FRACTAL (swinghigh, swinglow) divided by 100, which marks the
          confirming bar too. finta's WILLIAMS_FRACTAL marks the bar of the high or low itself,
          which uses the later bars (look-ahead).
        - The comparison is strict, as in TA-Lib: a high equal to a neighbouring high is no
          fractal.

    Args:
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        left_bars (int, optional): The bars before the high or low it must exceed. Defaults to 2.
        right_bars (int, optional): The bars after the high or low it must exceed. Defaults to 2.

    Returns:
        pd.DataFrame: The Up Fractal and Down Fractal, 1 on the bar that confirms a fractal and
        0 otherwise, empty for the first left_bars + right_bars bars.
    """
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be at least 1.")

    window = left_bars + right_bars
    position = pd.Series(np.arange(len(prices_high)), index=prices_high.index)

    def is_extreme(prices: pd.Series, sign: float) -> pd.Series:
        # The highs (or negated lows) of the bars before and after the pivot, without the pivot itself.
        prices = sign * prices
        others = pd.concat(
            [
                prices.shift(right_bars + 1).rolling(left_bars).max(),
                prices.rolling(right_bars).max(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        return (
            (prices.shift(right_bars) > others).astype(float).where(position >= window)
        )

    return pd.DataFrame(
        {
            "Up Fractal": is_extreme(prices_high, 1.0),
            "Down Fractal": is_extreme(prices_low, -1.0),
        }
    )


def get_crossover(series: pd.Series, reference: pd.Series | float) -> pd.Series:
    """
    Mark where a series crosses a reference line.

    The series crosses when it reaches the other side of the line than the side it was on last.
    A period exactly on the line keeps the side it came from, so touching the line and turning
    back is no crossing, while moving through it over two periods counts on the period the
    other side is reached.

    Args:
        series (pd.Series): The series, e.g. a fast moving average.
        reference (pd.Series | float): The line it crosses, e.g. a slow moving average or a
            threshold.

    Returns:
        pd.Series: 1 where the series crosses above the reference, -1 where it crosses below it
        and 0 otherwise, empty while either value of either period is missing.
    """
    if not isinstance(reference, pd.Series):
        reference = pd.Series(reference, index=series.index)

    side = np.sign(series - reference).replace(0.0, np.nan).ffill()
    crossover = np.sign(side.diff()).fillna(0.0)

    complete = (
        series.notna()
        & reference.notna()
        & series.shift(1).notna()
        & reference.shift(1).notna()
    )

    return crossover.where(complete)


def get_signals(
    prices: pd.Series,
    short_window: int,
    long_window: int,
    macd_short_window: int,
    macd_long_window: int,
    macd_signal_window: int,
    rsi_window: int,
    rsi_overbought: float,
    rsi_oversold: float,
    bollinger_window: int,
    bollinger_std_dev: float,
    high_low_window: int,
) -> pd.DataFrame:
    """
    Turn moving averages, the MACD, the RSI, the Bollinger Bands and new highs and lows into
    the events traders act on.

    Every signal is 1 on the period a line crosses upwards, -1 on the period it crosses
    downwards and 0 otherwise:

    - Moving Average Crossover: the short Simple Moving Average crossing the long one, a
      golden cross (1) or a death cross (-1) with the 50 and 200 periods.
    - MACD Crossover: the MACD line crossing its signal line.
    - RSI Overbought: the RSI entering (1) or leaving (-1) the overbought zone above its
      upper threshold.
    - RSI Oversold: the RSI leaving (1) or entering (-1) the oversold zone below its lower
      threshold.
    - Upper Bollinger Band: the price breaking out above (1) or falling back below (-1) the
      upper band.
    - Lower Bollinger Band: the price recovering above (1) or breaking down below (-1) the
      lower band.
    - New High/Low: a close above every close of the previous `high_low_window` periods (1) or
      below every one of them (-1).

    Reference: Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York
    Institute of Finance; Appel, G. (1979) for the MACD; Wilder, J.W. Jr. (1978) for the RSI;
    Bollinger, J. (2001) for the Bollinger Bands.

    Args:
        prices (pd.Series): The close prices.
        short_window (int): The window of the short moving average.
        long_window (int): The window of the long moving average.
        macd_short_window (int): The short window of the MACD.
        macd_long_window (int): The long window of the MACD.
        macd_signal_window (int): The window of the MACD's signal line.
        rsi_window (int): The window of the RSI.
        rsi_overbought (float): The RSI above which a price counts as overbought.
        rsi_oversold (float): The RSI below which a price counts as oversold.
        bollinger_window (int): The window of the Bollinger Bands.
        bollinger_std_dev (float): The number of standard deviations of the Bollinger Bands.
        high_low_window (int): The number of previous periods a new high or low is measured
            against.

    Returns:
        pd.DataFrame: The signals as columns, with 1, -1 or 0 per period and empty until the
        indicator behind a signal has its first two values.
    """
    macd = momentum_model.get_moving_average_convergence_divergence(
        prices, macd_short_window, macd_long_window, macd_signal_window
    )
    rsi = momentum_model.get_relative_strength_index(prices, rsi_window)
    bands = volatility_model.get_bollinger_bands(
        prices, bollinger_window, bollinger_std_dev
    )

    previous_high = prices.shift(1).rolling(high_low_window).max()
    previous_low = prices.shift(1).rolling(high_low_window).min()
    new_high_low = pd.Series(
        np.where(
            prices > previous_high, 1.0, np.where(prices < previous_low, -1.0, 0.0)
        ),
        index=prices.index,
    ).where(prices.notna() & previous_high.notna())

    return pd.DataFrame(
        {
            "Moving Average Crossover": get_crossover(
                overlap_model.get_moving_average(prices, short_window),
                overlap_model.get_moving_average(prices, long_window),
            ),
            "MACD Crossover": get_crossover(macd["MACD Line"], macd["Signal Line"]),
            "RSI Overbought": get_crossover(rsi, rsi_overbought),
            "RSI Oversold": get_crossover(rsi, rsi_oversold),
            "Upper Bollinger Band": get_crossover(prices, bands["Upper Band"]),
            "Lower Bollinger Band": get_crossover(prices, bands["Lower Band"]),
            "New High/Low": new_high_low,
        }
    )


def get_indicator_summary(
    values: pd.Series,
    prices: pd.Series,
    below: float | None = None,
    above: float | None = None,
    forward_periods: list[int] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Summarise the history of an indicator: where its latest value ranks, its extremes, the
    periods spent beyond a threshold, the crossings into that zone and the return of the price
    after each crossing.

    Each value is ranked against its own history up to that date, never against later values:

    - Percentile(t) = share of the values up to and including t that are at most Value(t)
    - Rank(t) = number of values up to and including t that are at most Value(t), so 1 is the
      lowest value so far and the number of observations the highest
    - Forward Return(t, n) = Price(t + n) / Price(t) - 1, with t + n the n-th price after t

    A crossing below `below` is the period the indicator reaches the other side of the line
    (see get_crossover): a value exactly on the line keeps the side it came from.

    Args:
        values (pd.Series): The indicator per date.
        prices (pd.Series): The prices the forward returns are measured on, per date.
        below (float | None, optional): The line the indicator falls below, e.g. 30 for an
            oversold RSI. Defaults to None.
        above (float | None, optional): The line the indicator rises above, e.g. 70 for an
            overbought RSI. Defaults to None.
        forward_periods (list[int] | None, optional): The numbers of periods after each
            crossing to measure the return over. Defaults to None.

    Returns:
        tuple[pd.Series, pd.DataFrame]: The summary, and a row per crossing with its zone, the
        indicator value, its percentile and the forward returns.
    """
    values = values.dropna()
    forward_periods = forward_periods or []
    if values.empty:
        return pd.Series(dtype=object), pd.DataFrame()

    labels = values.index.astype(str)
    rank = values.expanding().rank(method="max")
    observations = np.arange(1, len(values) + 1)
    percentile = rank / observations

    summary = {
        "Latest": values.iloc[-1],
        "Latest Date": labels[-1],
        "Percentile": percentile.iloc[-1],
        "Rank": int(rank.iloc[-1]),
        "Observations": len(values),
        "Mean": values.mean(),
        "Minimum": values.min(),
        "Minimum Date": labels[np.argmin(values.to_numpy())],
        "Maximum": values.max(),
        "Maximum Date": labels[np.argmax(values.to_numpy())],
    }

    # Returns on the ticker's own trading days, measured from each date the indicator has a
    # value; the last n dates have no return over n periods yet and are left out of averages.
    prices = prices.dropna()
    forward_returns = {
        periods: (prices.shift(-periods) / prices - 1).reindex(values.index)
        for periods in forward_periods
    }
    for periods, returns in forward_returns.items():
        summary[f"Average {periods}-Period Return"] = returns.mean()

    crossings = []
    for zone, line, direction in (("Below", below, -1), ("Above", above, 1)):
        if line is None:
            continue

        beyond = values < line if zone == "Below" else values > line
        entries = get_crossover(values, line).eq(direction)
        dates = values.index[entries.to_numpy()]

        summary[f"Periods {zone}"] = int(beyond.sum())
        summary[f"Crossings {zone}"] = len(dates)
        summary[f"Last Crossing {zone}"] = (
            labels[entries.to_numpy()][-1] if len(dates) else np.nan
        )
        for periods, returns in forward_returns.items():
            summary[f"Average {periods}-Period Return After Crossing {zone}"] = (
                returns.loc[dates].mean()
            )

        crossings.append(
            pd.DataFrame(
                {
                    "Zone": zone,
                    "Value": values.loc[dates],
                    "Percentile": percentile.loc[dates],
                    **{
                        f"{periods}-Period Return": returns.loc[dates]
                        for periods, returns in forward_returns.items()
                    },
                },
                index=dates,
            )
        )

    crossing_table = pd.concat(crossings).sort_index() if crossings else pd.DataFrame()

    return pd.Series(summary, dtype=object), crossing_table
