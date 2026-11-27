"""Pattern Module: candlestick patterns and indicator signals"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.technicals.helpers import get_observations_per_year
from financetoolkit.technicals.momentum import momentum_model
from financetoolkit.technicals.overlap import overlap_model
from financetoolkit.technicals.volatility import volatility_model
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR

# The first candle each pattern is evaluated on: the previous candles its reference sizes average plus the candles
# before its last one. This is the warm-up of TA-Lib's CDL functions, which also gives the engulfing 2 and the
# three outside 3 candles.
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
    "Two Crows": 12,
    "Three Stars in the South": 12,
    "Abandoned Baby": 12,
    "Advance Block": 12,
    "Side-by-Side White Lines": 7,
    "Identical Three Crows": 12,
    "Stalled Pattern": 12,
    "Stick Sandwich": 7,
    "Tasuki Gap": 7,
    "Tri-Star": 12,
    "Unique Three River": 12,
    "Upside Gap Two Crows": 12,
    "Gap Three Methods": 2,
    "Three-Line Strike": 8,
    "Concealing Baby Swallow": 13,
    "Breakaway": 14,
    "Ladder Bottom": 14,
    "Mat Hold": 14,
    "Rising/Falling Three Methods": 14,
    "Hikkake": 5,
    "Modified Hikkake": 10,
}

# How far the last candle of a (doji) star and of an abandoned baby, and the second candle of a dark cloud cover,
# must reach into the first candle's real body: the default penetrations of TA-Lib.
STAR_PENETRATION = 0.3
DARK_CLOUD_PENETRATION = 0.5
# How far the small candles of a mat hold may reach into the first candle's real body (TA-Lib's default).
MAT_HOLD_PENETRATION = 0.5
# The candles after a hikkake (and a modified hikkake) within which a close beyond the inside bar confirms it.
HIKKAKE_CONFIRMATION = 3

# DeMark's TD Sequential: a setup of 9 closes beyond the close 4 bars earlier, then a countdown of 13 closes
# beyond the low (high) 2 bars earlier whose 13th bar is measured against the close of its 8th.
TD_SETUP_BARS, TD_SETUP_LAG = 9, 4
TD_COUNTDOWN_BARS, TD_COUNTDOWN_LAG, TD_QUALIFIER_BAR = 13, 2, 8


def get_candle_features(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.DataFrame:
    """
    Describe each candle by its parts and by the reference sizes its parts are compared with: the building blocks of
    the candlestick patterns.

    A candle is white when it closes at or above its open and black otherwise. Its real body
    runs from the open to the close, the upper shadow from the top of the body to the high and
    the lower shadow from the bottom of the body to the low. Whether a part is long, short or
    tiny depends on the candles before it, so each candle also carries reference sizes measured
    on the previous candles only (never on the candle itself). These are the default candle
    settings of TA-Lib:

    - Average Body = the average real body of the previous 10 candles: a body above it is long,
      a body below it short.
    - Tiny = 10% of the average high-low range of the previous 10 candles: a body at most this
      size is a doji's, a shadow below it very short.
    - Average Shadow = half the average sum of both shadows of the previous 10 candles (the
      average single shadow): a shadow below it is short.
    - Near = 20%, Far = 60% and Equal = 5% of the average high-low range of the previous 5
      candles: the distances within which prices are near or equal and beyond which they are
      far apart.

    A shadow is long when it is longer than the candle's own real body and very long when it is
    longer than twice that body.

    Args:
        prices_open (pd.Series): The open prices.
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.

    Returns:
        pd.DataFrame: Per candle the open, high, low and close, its colour (1 white, -1 black),
        the top and bottom of the real body, the real body, the upper and lower shadow, the
        high-low range and the reference sizes average_body, tiny, average_shadow, near, far
        and equal.
    """
    white = prices_close >= prices_open
    body = (prices_close - prices_open).abs()
    upper_shadow = prices_high - prices_close.where(white, prices_open)
    lower_shadow = prices_open.where(white, prices_close) - prices_low
    high_low = prices_high - prices_low

    average_range = high_low.shift(1).rolling(10).mean()
    average_range_5 = high_low.shift(1).rolling(5).mean()

    return pd.DataFrame(
        {
            "open": prices_open,
            "high": prices_high,
            "low": prices_low,
            "close": prices_close,
            "color": np.where(white, 1.0, -1.0),
            "top": np.maximum(prices_open, prices_close),
            "bottom": np.minimum(prices_open, prices_close),
            "body": body,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "high_low": high_low,
            "average_body": body.shift(1).rolling(10).mean(),
            "tiny": 0.1 * average_range,
            "average_shadow": (upper_shadow + lower_shadow).shift(1).rolling(10).mean()
            / 2.0,
            "near": 0.2 * average_range_5,
            "far": 0.6 * average_range_5,
            "equal": 0.05 * average_range_5,
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
    Recognise 62 candlestick patterns of Nison, Bulkowski and Chesler in their measurable form: the shapes of one to
    five candles read as a reversal or a continuation of the trend.

    Each pattern is 1 on a candle that completes it as a bullish pattern, -1 as a bearish
    pattern and 0 otherwise, which agrees with TA-Lib's +100 / -100 divided by 100. TA-Lib's weaker
    form of the engulfing, harami and harami cross patterns, where one end of the two real
    bodies is equal, is 0.8 (TA-Lib's 80). The dojis (doji, dragonfly, gravestone and
    long-legged doji, rickshaw man and takuri) are indecision rather than a direction and are 1
    when present, as in TA-Lib; a marubozu, a spinning top, a belt-hold, a closing marubozu, a
    high-wave, long line and short line candle and an inside bar take the colour of the candle
    (1 white, -1 black). A hikkake and a modified hikkake are 1 or -1 on the candle that
    completes them and 2 or -2 on the later candle that confirms them, which agrees with TA-Lib's +200 /
    -200 divided by 100 like every other value.

    Each pattern is a set of conditions on the features of its candles (see
    `get_candle_features`): their colour, real body, shadows, the gaps between their real
    bodies and how each part compares with the candles before it. Sizes are relative: a real
    body is long when it is longer than the average real body of the 10 previous candles and
    short when it is shorter; it is a doji's body when it is at most 10% of the average
    high-low range of the 10 previous candles; a shadow is very short when it is shorter than
    10% of that range, short when it is shorter than half the average sum of both shadows of
    the 10 previous candles (the average shadow), long when it is longer than the candle's own
    real body and very long when it is longer than twice that body; "near" is within 20% and
    "far" at least 60% of the average high-low range of the 5 previous candles, and two prices
    are "equal" when they are within 5% of that range. These reference sizes are the default
    candle settings of TA-Lib.

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
    - Two Crows (bearish): a long white candle, a black real body that gaps above it and a
      black candle that opens within the second real body and closes within the first.
    - Upside Gap Two Crows (bearish): a long white candle, a short black real body that gaps
      above it and a black real body that engulfs the second and still closes above the first
      candle's close.
    - Three Stars in the South (bullish): a long black candle with a long lower shadow, a
      smaller black candle that opens above the first close within its range and has a low
      inside the first candle's range and a lower shadow, and a short black candle with very
      short shadows inside the second candle's range.
    - Abandoned Baby: a long real body, a doji whose shadows gap away from it and a real body of
      the opposite colour whose shadows gap away from the doji in the other direction and that
      closes at least 30% into the first body, in the direction of the third candle.
    - Advance Block (bearish): three white candles with higher closes, each opening within or
      near the previous real body, the first long with a short upper shadow, and an advance
      that weakens: the second body far shorter than the first (and the third not longer than
      the second), the third far shorter than the second, ever shorter bodies with an upper
      shadow that is not short on the second or third candle, or a shorter third body with a
      long upper shadow.
    - Side-by-Side White Lines: a real body gap followed by two white candles of nearly the same
      size ("near") that open at the same price ("equal"), both beyond the gap; bullish after a
      gap up and bearish after a gap down.
    - Identical Three Crows (bearish): three declining black candles with very short lower
      shadows, the second and third opening at the previous close ("equal").
    - Stalled Pattern (bearish): two long white candles with higher closes, the second with a
      very short upper shadow and opening within or near the first real body, followed by a
      short white real body with a higher close that rides on the shoulder of the second body
      (it opens at most "near" below the second close minus its own body).
    - Stick Sandwich (bullish): a black candle, a white candle whose low is above the first
      close and a black candle that closes at the first close ("equal").
    - Tasuki Gap: a real body gap in the direction of the first candle after it, followed by a
      candle of the opposite colour that opens within that candle's real body and closes inside
      the gap, with the two real bodies of nearly the same size; in the direction of the gap.
    - Tri-Star: three dojis, the second gapping away from the first and the third not beyond
      the second: bullish when the second gaps down, bearish when it gaps up.
    - Unique Three River (bullish): a long black candle, a black harami (a higher close and an
      open at most the first open) with a lower low, and a short white real body that opens
      above the second candle's low.
    - Gap Three Methods (Upside or Downside): two candles of the same colour with a real body
      gap between them, followed by a candle of the opposite colour that opens within the
      second real body and closes within the first, in the direction of the gap.
    - Three-Line Strike: three candles of one colour with higher (white) or lower (black)
      closes, each opening within or near the previous real body, followed by a candle of the
      other colour that opens beyond the third close and closes beyond the first open; in the
      direction of the three candles.
    - Concealing Baby Swallow (bullish): two black candles with very short shadows, a black
      candle whose real body gaps down and whose upper shadow (not very short) reaches into the
      second real body, and a black candle whose high and low lie beyond the third candle's.
    - Breakaway: a long real body, a candle of the same colour whose real body gaps away from it,
      a candle and a fourth candle of the first colour that each move further away (both a
      higher high and low after a white first candle, both lower after a black one), and a
      candle of the other colour that closes inside the gap; in the direction of the last
      candle.
    - Ladder Bottom (bullish): three black candles with lower opens and closes, a black candle
      with an upper shadow that is not very short and a white candle that opens above the
      fourth open and closes above its high.
    - Mat Hold (bullish): a long white candle, a short black real body that gaps above it, two
      more short real bodies that each end lower (the top of each below the top of the one
      before) and both reach below the first close but no further than half the first real
      body, and a white candle that opens above the fourth close and closes above
      the highs of the three.
    - Rising/Falling Three Methods: a long real body, three short real bodies of the other
      colour with falling (after white) or rising (after black) closes, each partly within the
      first candle's high-low range, and a long real body of the first colour that opens beyond
      the fourth close and closes beyond the first close; in the direction of the first candle.
    - Hikkake: an inside bar (a lower high and a higher low than the candle before) followed by
      a candle that breaks out of it on one side, with a lower high and a lower low (1) or a
      higher high and a higher low (-1); the breakout is read as a false move. It is confirmed
      (2 or -2) by the first close above the inside bar's high (below its low) within the next
      3 candles, unless a new hikkake comes first.
    - Modified Hikkake: the hikkake after two inside bars in a row, the first of them closing
      near its low (within "near", bullish) or near its high (bearish), confirmed in the same
      way beyond the second inside bar.

    Reference: Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York Institute
    of Finance; Bulkowski, T.N. (2008). "Encyclopedia of Candlestick Charts." Wiley; Chesler,
    D. (2004). "Trading False Moves with the Hikkake Pattern." Active Trader, April 2004. The
    measurable rules and thresholds are those TA-Lib (Mario Fortier) documents for these
    patterns, which turn the descriptions of Nison and others into candle-size comparisons.

    Referenced against:
    - TA-Lib, CDL2CROWS: https://ta-lib.org/functions/cdl2crows.html
    - TA-Lib, CDL3BLACKCROWS: https://ta-lib.org/functions/cdl3blackcrows.html
    - TA-Lib, CDL3INSIDE: https://ta-lib.org/functions/cdl3inside.html
    - TA-Lib, CDL3LINESTRIKE: https://ta-lib.org/functions/cdl3linestrike.html
    - TA-Lib, CDL3OUTSIDE: https://ta-lib.org/functions/cdl3outside.html
    - TA-Lib, CDL3STARSINSOUTH: https://ta-lib.org/functions/cdl3starsinsouth.html
    - TA-Lib, CDL3WHITESOLDIERS: https://ta-lib.org/functions/cdl3whitesoldiers.html
    - TA-Lib, CDLABANDONEDBABY: https://ta-lib.org/functions/cdlabandonedbaby.html
    - TA-Lib, CDLADVANCEBLOCK: https://ta-lib.org/functions/cdladvanceblock.html
    - TA-Lib, CDLBELTHOLD: https://ta-lib.org/functions/cdlbelthold.html
    - TA-Lib, CDLBREAKAWAY: https://ta-lib.org/functions/cdlbreakaway.html
    - TA-Lib, CDLCLOSINGMARUBOZU: https://ta-lib.org/functions/cdlclosingmarubozu.html
    - TA-Lib, CDLCONCEALBABYSWALL: https://ta-lib.org/functions/cdlconcealbabyswall.html
    - TA-Lib, CDLCOUNTERATTACK: https://ta-lib.org/functions/cdlcounterattack.html
    - TA-Lib, CDLDARKCLOUDCOVER: https://ta-lib.org/functions/cdldarkcloudcover.html
    - TA-Lib, CDLDOJI: https://ta-lib.org/functions/cdldoji.html
    - TA-Lib, CDLDOJISTAR: https://ta-lib.org/functions/cdldojistar.html
    - TA-Lib, CDLDRAGONFLYDOJI: https://ta-lib.org/functions/cdldragonflydoji.html
    - TA-Lib, CDLENGULFING: https://ta-lib.org/functions/cdlengulfing.html
    - TA-Lib, CDLEVENINGDOJISTAR: https://ta-lib.org/functions/cdleveningdojistar.html
    - TA-Lib, CDLEVENINGSTAR: https://ta-lib.org/functions/cdleveningstar.html
    - TA-Lib, CDLGAPSIDESIDEWHITE: https://ta-lib.org/functions/cdlgapsidesidewhite.html
    - TA-Lib, CDLGRAVESTONEDOJI: https://ta-lib.org/functions/cdlgravestonedoji.html
    - TA-Lib, CDLHAMMER: https://ta-lib.org/functions/cdlhammer.html
    - TA-Lib, CDLHANGINGMAN: https://ta-lib.org/functions/cdlhangingman.html
    - TA-Lib, CDLHARAMI: https://ta-lib.org/functions/cdlharami.html
    - TA-Lib, CDLHARAMICROSS: https://ta-lib.org/functions/cdlharamicross.html
    - TA-Lib, CDLHIGHWAVE: https://ta-lib.org/functions/cdlhighwave.html
    - TA-Lib, CDLHIKKAKE: https://ta-lib.org/functions/cdlhikkake.html
    - TA-Lib, CDLHIKKAKEMOD: https://ta-lib.org/functions/cdlhikkakemod.html
    - TA-Lib, CDLHOMINGPIGEON: https://ta-lib.org/functions/cdlhomingpigeon.html
    - TA-Lib, CDLIDENTICAL3CROWS: https://ta-lib.org/functions/cdlidentical3crows.html
    - TA-Lib, CDLINNECK: https://ta-lib.org/functions/cdlinneck.html
    - TA-Lib, CDLINVERTEDHAMMER: https://ta-lib.org/functions/cdlinvertedhammer.html
    - TA-Lib, CDLKICKING: https://ta-lib.org/functions/cdlkicking.html
    - TA-Lib, CDLKICKINGBYLENGTH: https://ta-lib.org/functions/cdlkickingbylength.html
    - TA-Lib, CDLLADDERBOTTOM: https://ta-lib.org/functions/cdlladderbottom.html
    - TA-Lib, CDLLONGLEGGEDDOJI: https://ta-lib.org/functions/cdllongleggeddoji.html
    - TA-Lib, CDLLONGLINE: https://ta-lib.org/functions/cdllongline.html
    - TA-Lib, CDLMARUBOZU: https://ta-lib.org/functions/cdlmarubozu.html
    - TA-Lib, CDLMATCHINGLOW: https://ta-lib.org/functions/cdlmatchinglow.html
    - TA-Lib, CDLMATHOLD: https://ta-lib.org/functions/cdlmathold.html
    - TA-Lib, CDLMORNINGDOJISTAR: https://ta-lib.org/functions/cdlmorningdojistar.html
    - TA-Lib, CDLMORNINGSTAR: https://ta-lib.org/functions/cdlmorningstar.html
    - TA-Lib, CDLONNECK: https://ta-lib.org/functions/cdlonneck.html
    - TA-Lib, CDLPIERCING: https://ta-lib.org/functions/cdlpiercing.html
    - TA-Lib, CDLRICKSHAWMAN: https://ta-lib.org/functions/cdlrickshawman.html
    - TA-Lib, CDLRISEFALL3METHODS: https://ta-lib.org/functions/cdlrisefall3methods.html
    - TA-Lib, CDLSEPARATINGLINES: https://ta-lib.org/functions/cdlseparatinglines.html
    - TA-Lib, CDLSHOOTINGSTAR: https://ta-lib.org/functions/cdlshootingstar.html
    - TA-Lib, CDLSHORTLINE: https://ta-lib.org/functions/cdlshortline.html
    - TA-Lib, CDLSPINNINGTOP: https://ta-lib.org/functions/cdlspinningtop.html
    - TA-Lib, CDLSTALLEDPATTERN: https://ta-lib.org/functions/cdlstalledpattern.html
    - TA-Lib, CDLSTICKSANDWICH: https://ta-lib.org/functions/cdlsticksandwich.html
    - TA-Lib, CDLTAKURI: https://ta-lib.org/functions/cdltakuri.html
    - TA-Lib, CDLTASUKIGAP: https://ta-lib.org/functions/cdltasukigap.html
    - TA-Lib, CDLTHRUSTING: https://ta-lib.org/functions/cdlthrusting.html
    - TA-Lib, CDLTRISTAR: https://ta-lib.org/functions/cdltristar.html
    - TA-Lib, CDLUNIQUE3RIVER: https://ta-lib.org/functions/cdlunique3river.html
    - TA-Lib, CDLUPSIDEGAP2CROWS: https://ta-lib.org/functions/cdlupsidegap2crows.html
    - TA-Lib, CDLXSIDEGAP3METHODS: https://ta-lib.org/functions/cdlxsidegap3methods.html

    Shares similar logic:
    - pandas-ta-classic, cdl_inside: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/candles/cdl_inside.py#L11

    Notes:
        - A pattern is empty (NaN) during its warm-up: the previous candles its reference
          sizes average and the candles before its last one (2 to 14 candles, 1 for the
          inside bar, `PATTERN_LOOKBACK`), the warm-up of TA-Lib's CDL functions, so a ticker
          is evaluated from the same candle on as TA-Lib. pandas-ta-classic's inside bar is 0
          instead of empty on the first candle.
        - TA-Lib marks the gravestone doji, which Nison reads as bearish after a rise, as 1
          like every doji; so does this function.
        - A pattern only describes the candles; the trend before it, which Nison requires for
          e.g. a hammer (after a decline) and a hanging man (after a rise), is not checked, as
          in TA-Lib. The direction is therefore TA-Lib's: a counterattack, a kicking and
          separating lines follow the second candle, while Nison reads separating lines as a
          continuation of the trend, and the in-neck, on-neck and thrusting are bearish
          continuation patterns (Nison, Bulkowski) that TA-Lib marks -1 whatever the trend.
        - The doji stars and the abandoned baby use the same 30% penetration as the stars
          (STAR_PENETRATION, TA-Lib's default penetration); Nison only asks for a close
          well into the first body.
        - The tasuki gap, side-by-side white lines and gap three methods are continuation
          patterns (Nison, Bulkowski) and follow the direction of the gap; the stick sandwich,
          the three stars in the south and the unique three river are bullish and the two
          crows, upside gap two crows, identical three crows, advance block and stalled pattern
          bearish, as in TA-Lib.
        - The breakaway follows its last candle (a reversal), and the three-line strike, mat
          hold and rising/falling three methods follow their first candles (continuations), as
          in TA-Lib; the concealing baby swallow and ladder bottom are bullish.
        - The mat hold's small candles may reach at most 50% into the first real body
          (MAT_HOLD_PENETRATION, TA-Lib's default penetration).
        - A hikkake is confirmed by a close beyond the inside bar (TA-Lib's measurable form of
          Chesler's breakout) within HIKKAKE_CONFIRMATION (3) candles. A hikkake's sign is the
          direction it points to, the opposite of its false breakout, and its confirmation
          keeps that sign at twice the size, so a filter on 1 / -1 only finds the setups and
          one on 2 / -2 only the confirmations.

    Args:
        prices_open (pd.Series): The open prices.
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.
        patterns (list[str] | None, optional): The patterns to return, names of
            `PATTERN_LOOKBACK` in the order given. Defaults to None, which returns all 62.

    Returns:
        pd.DataFrame: The patterns as columns, with 1 (bullish), -1 (bearish), 0.8 / -0.8 (the
        weaker engulfing, harami and harami cross), 2 / -2 (a confirmed hikkake) or 0 per
        candle.
    """
    unknown = set(patterns or []) - set(PATTERN_LOOKBACK)
    if unknown:
        raise ValueError(
            f"Unknown pattern(s) {sorted(unknown)}, choose from {list(PATTERN_LOOKBACK)}."
        )

    # The last candle of a pattern and the one to four candles before it.
    last = get_candle_features(prices_open, prices_high, prices_low, prices_close)
    prev, prev2, prev3, prev4 = (last.shift(candles) for candles in range(1, 5))

    def white(candle: pd.DataFrame) -> pd.Series:
        return candle.color == 1

    def black(candle: pd.DataFrame) -> pd.Series:
        return candle.color == -1

    def long_body(candle: pd.DataFrame) -> pd.Series:
        return candle.body > candle.average_body

    def short_body(candle: pd.DataFrame) -> pd.Series:
        return candle.body < candle.average_body

    def doji(candle: pd.DataFrame) -> pd.Series:
        return candle.body <= candle.tiny

    def shaven(candle: pd.DataFrame) -> pd.Series:
        # Both shadows very short.
        return (candle.upper_shadow < candle.tiny) & (candle.lower_shadow < candle.tiny)

    def marubozu(candle: pd.DataFrame) -> pd.Series:
        return long_body(candle) & shaven(candle)

    def gap_up(candle: pd.DataFrame, before: pd.DataFrame) -> pd.Series:
        # A real body gap: the whole body above the body of the candle before.
        return candle.bottom > before.top

    def gap_down(candle: pd.DataFrame, before: pd.DataFrame) -> pd.Series:
        return candle.top < before.bottom

    def within(price: pd.Series, level: pd.Series, distance: pd.Series) -> pd.Series:
        return (price <= level + distance) & (price >= level - distance)

    def steps_up(candle: pd.DataFrame, before: pd.DataFrame) -> pd.Series:
        # A white candle opening above the open of the white candle before, within or near its body.
        return (candle.open > before.open) & (candle.open <= before.close + before.near)

    def opens_near_body(candle: pd.DataFrame, before: pd.DataFrame) -> pd.Series:
        # An open within the real body of the candle before or near it, on either side.
        return (candle.open >= before.bottom - before.near) & (
            candle.open <= before.top + before.near
        )

    def in_range(candle: pd.DataFrame, before: pd.DataFrame) -> pd.Series:
        # Part of the real body within the high-low range of the candle before.
        return (candle.bottom < before.high) & (candle.top > before.low)

    def with_confirmation(
        pattern: pd.Series, inside_high: pd.Series, inside_low: pd.Series
    ) -> pd.Series:
        # A hikkake is confirmed by the first close beyond its inside bar (above the high after a bullish one,
        # below the low after a bearish one) within HIKKAKE_CONFIRMATION candles, before another hikkake
        # appears; the confirming candle is 2 or -2.
        position = pd.Series(np.arange(len(pattern)), index=pattern.index)
        found = pattern != 0
        since = position.where(found).ffill()
        direction = pattern.where(found).ffill()
        level = (
            inside_high.where(found)
            .ffill()
            .where(direction > 0, inside_low.where(found).ffill())
        )
        breakout = direction * (last.close - level) > 0
        candidate = (
            ~found & (position - since <= HIKKAKE_CONFIRMATION) & breakout
        ).astype(int)
        first = candidate.groupby(since).cumsum().eq(1) & candidate.eq(1)
        return pattern.where(found, np.where(first, 2.0 * direction, 0.0))

    # The direction in which the last candle's real body covers the one before: 1 when both of its ends lie
    # beyond, 0.8 when one end is equal (the weaker form) and 0 otherwise.
    engulfing = (
        (last.color == -prev.color)
        & (
            ((last.top >= prev.top) & (last.bottom < prev.bottom))
            | ((last.top > prev.top) & (last.bottom <= prev.bottom))
        )
    ) * np.where((last.top != prev.top) & (last.bottom != prev.bottom), 1.0, 0.8)
    harami = np.where(
        (last.top < prev.top) & (last.bottom > prev.bottom),
        1.0,
        np.where((last.top <= prev.top) & (last.bottom >= prev.bottom), 0.8, 0.0),
    )

    opposite = last.color != prev.color
    open_shadow = last.lower_shadow.where(white(last), last.upper_shadow)
    close_shadow = last.upper_shadow.where(white(last), last.lower_shadow)
    middle = last.low + last.high_low / 2
    opens_in_previous_body = (last.open < prev.top) & (last.open > prev.bottom)

    long_black_then_white_below = (
        black(prev) & long_body(prev) & white(last) & (last.open < prev.low)
    )
    kicking = (
        opposite
        & (
            (black(prev) & (last.low > prev.high))
            | (white(prev) & (last.high < prev.low))
        )
        & marubozu(prev)
        & marubozu(last)
    )
    # A long white candle followed by a black real body gapping above it (two crows, upside gap two crows).
    black_gap_above_white = (
        white(prev2)
        & long_body(prev2)
        & black(prev)
        & gap_up(prev, prev2)
        & black(last)
    )
    star = long_body(prev2) & long_body(last)
    rising_whites = (
        white(prev2)
        & white(prev)
        & white(last)
        & (last.close > prev.close)
        & (prev.close > prev2.close)
        & steps_up(prev, prev2)
    )

    bullish = {
        "Hammer": short_body(last)
        & (last.lower_shadow > last.body)
        & (last.upper_shadow < last.tiny)
        & (last.bottom <= prev.low + prev.near),
        "Inverted Hammer": short_body(last)
        & (last.upper_shadow > last.body)
        & (last.lower_shadow < last.tiny)
        & gap_down(last, prev),
        "Piercing Line": black(prev)
        & long_body(prev)
        & white(last)
        & long_body(last)
        & (last.open < prev.low)
        & (last.close < prev.open)
        & (last.close > prev.close + prev.body * 0.5),
        "Morning Star": black(prev2)
        & gap_down(prev, prev2)
        & (prev.body <= prev.average_body)
        & white(last)
        & (last.close > prev2.close + prev2.body * STAR_PENETRATION)
        & star,
        "Morning Doji Star": black(prev2)
        & gap_down(prev, prev2)
        & doji(prev)
        & white(last)
        & (last.close > prev2.close + prev2.body * STAR_PENETRATION)
        & star,
        "Three White Soldiers": rising_whites
        & (prev2.upper_shadow < prev2.tiny)
        & (prev.upper_shadow < prev.tiny)
        & (last.upper_shadow < last.tiny)
        & steps_up(last, prev)
        & (prev.body > prev2.body - prev2.far)
        & (last.body > prev.body - prev.far)
        & long_body(last),
        "Homing Pigeon": black(prev)
        & long_body(prev)
        & black(last)
        & (last.body <= last.average_body)
        & (last.open < prev.open)
        & (last.close > prev.close),
        "Matching Low": black(prev)
        & black(last)
        & within(last.close, prev.close, prev.equal),
        "Three Stars in the South": black(prev2)
        & long_body(prev2)
        & (prev2.lower_shadow > prev2.body)
        & black(prev)
        & (prev.body < prev2.body)
        & (prev.open > prev2.close)
        & (prev.open <= prev2.high)
        & (prev.low < prev2.close)
        & (prev.low >= prev2.low)
        & (prev.lower_shadow > prev.tiny)
        & black(last)
        & short_body(last)
        & (last.lower_shadow < last.tiny)
        & (last.upper_shadow < last.tiny)
        & (last.low > prev.low)
        & (last.high < prev.high),
        "Stick Sandwich": black(prev2)
        & white(prev)
        & (prev.low > prev2.close)
        & black(last)
        & within(last.close, prev2.close, prev2.equal),
        "Unique Three River": black(prev2)
        & long_body(prev2)
        & black(prev)
        & (prev.close > prev2.close)
        & (prev.open <= prev2.open)
        & (prev.low < prev2.low)
        & white(last)
        & short_body(last)
        & (last.open > prev.low),
        # Two black candles without shadows, a black candle that gaps down with an upper shadow reaching into
        # the second body, and a black candle that engulfs it including its shadows.
        "Concealing Baby Swallow": black(prev3)
        & shaven(prev3)
        & black(prev2)
        & shaven(prev2)
        & black(prev)
        & gap_down(prev, prev2)
        & (prev.upper_shadow > prev.tiny)
        & (prev.high > prev2.close)
        & black(last)
        & (last.high > prev.high)
        & (last.low < prev.low),
        "Ladder Bottom": black(prev4)
        & black(prev3)
        & black(prev2)
        & (prev3.open < prev4.open)
        & (prev2.open < prev3.open)
        & (prev3.close < prev4.close)
        & (prev2.close < prev3.close)
        & black(prev)
        & (prev.upper_shadow > prev.tiny)
        & white(last)
        & (last.open > prev.open)
        & (last.close > prev.high),
        # A long white candle, a short black candle gapping above it and two more short candles that hold in
        # the upper part of the first body while falling, then a white candle above all three.
        "Mat Hold": white(prev4)
        & long_body(prev4)
        & black(prev3)
        & short_body(prev3)
        & gap_up(prev3, prev4)
        & short_body(prev2)
        & short_body(prev)
        & (prev2.bottom < prev4.close)
        & (prev.bottom < prev4.close)
        & (prev2.bottom > prev4.close - prev4.body * MAT_HOLD_PENETRATION)
        & (prev.bottom > prev4.close - prev4.body * MAT_HOLD_PENETRATION)
        & (prev2.top < prev3.top)
        & (prev.top < prev2.top)
        & white(last)
        & (last.open > prev.close)
        & (last.close > prev3.high)
        & (last.close > prev2.high)
        & (last.close > prev.high),
    }
    bearish = {
        "Hanging Man": short_body(last)
        & (last.lower_shadow > last.body)
        & (last.upper_shadow < last.tiny)
        & (last.bottom >= prev.high - prev.near),
        "Shooting Star": short_body(last)
        & (last.upper_shadow > last.body)
        & (last.lower_shadow < last.tiny)
        & gap_up(last, prev),
        "Dark Cloud Cover": white(prev)
        & long_body(prev)
        & black(last)
        & (last.open > prev.high)
        & (last.close > prev.open)
        & (last.close < prev.close - prev.body * DARK_CLOUD_PENETRATION),
        "Evening Star": white(prev2)
        & gap_up(prev, prev2)
        & (prev.body <= prev.average_body)
        & black(last)
        & (last.close < prev2.close - prev2.body * STAR_PENETRATION)
        & star,
        "Evening Doji Star": white(prev2)
        & gap_up(prev, prev2)
        & doji(prev)
        & black(last)
        & (last.close < prev2.close - prev2.body * STAR_PENETRATION)
        & star,
        "Three Black Crows": white(prev3)
        & (prev3.high > prev2.close)
        & black(prev2)
        & black(prev)
        & black(last)
        & (prev2.lower_shadow < prev2.tiny)
        & (prev.lower_shadow < prev.tiny)
        & (last.lower_shadow < last.tiny)
        & (prev.open < prev2.open)
        & (prev.open > prev2.close)
        & (last.open < prev.open)
        & (last.open > prev.close)
        & (prev2.close > prev.close)
        & (prev.close > last.close),
        "In-Neck": long_black_then_white_below
        & (last.close <= prev.close + prev.equal)
        & (last.close >= prev.close),
        "On-Neck": long_black_then_white_below
        & within(last.close, prev.low, prev.equal),
        "Thrusting": long_black_then_white_below
        & (last.close > prev.close + prev.equal)
        & (last.close <= prev.close + prev.body * 0.5),
        "Two Crows": black_gap_above_white
        & (last.open < prev.open)
        & (last.open > prev.close)
        & (last.close > prev2.open)
        & (last.close < prev2.close),
        "Upside Gap Two Crows": black_gap_above_white
        & (prev.body <= prev.average_body)
        & (last.open > prev.open)
        & (last.close < prev.close)
        & (last.close > prev2.close),
        "Identical Three Crows": black(prev2)
        & black(prev)
        & black(last)
        & (prev2.lower_shadow < prev2.tiny)
        & (prev.lower_shadow < prev.tiny)
        & (last.lower_shadow < last.tiny)
        & (prev2.close > prev.close)
        & (prev.close > last.close)
        & within(prev.open, prev2.close, prev2.equal)
        & within(last.open, prev.close, prev.equal),
        # The advance weakens: a far smaller second body, a far smaller third body, ever smaller bodies with
        # an upper shadow that is not short, or a smaller third body with a long upper shadow.
        "Advance Block": rising_whites
        & long_body(prev2)
        & (prev2.upper_shadow < prev2.average_shadow)
        & steps_up(last, prev)
        & (
            ((prev.body < prev2.body - prev2.far) & (last.body < prev.body + prev.near))
            | (last.body < prev.body - prev.far)
            | (
                (last.body < prev.body)
                & (prev.body < prev2.body)
                & (
                    (last.upper_shadow > last.average_shadow)
                    | (prev.upper_shadow > prev.average_shadow)
                )
            )
            | ((last.body < prev.body) & (last.upper_shadow > last.body))
        ),
        "Stalled Pattern": rising_whites
        & long_body(prev2)
        & long_body(prev)
        & (prev.upper_shadow < prev.tiny)
        & short_body(last)
        & (last.open >= prev.close - last.body - prev.near),
    }
    # An inside bar followed by a candle that breaks out of it on one side (a lower high and low, or a higher
    # high and low): a false breakout, read in the other direction. The modified hikkake has two inside bars
    # in a row, the first closing near its low (bullish) or near its high (bearish).
    inside = (prev.high < prev2.high) & (prev.low > prev2.low)
    breaks_down = inside & (last.high < prev.high) & (last.low < prev.low)
    breaks_up = inside & (last.high > prev.high) & (last.low > prev.low)
    twice_inside = (prev2.high < prev3.high) & (prev2.low > prev3.low)
    hikkake = pd.Series(
        np.where(breaks_down, 1.0, np.where(breaks_up, -1.0, 0.0)), index=last.index
    )
    modified_hikkake = pd.Series(
        np.where(
            twice_inside & breaks_down & (prev2.close <= prev2.low + prev2.near),
            1.0,
            np.where(
                twice_inside & breaks_up & (prev2.close >= prev2.high - prev2.near),
                -1.0,
                0.0,
            ),
        ),
        index=last.index,
    )

    # Patterns that are 1 when present (the dojis), take the colour of the last candle, or point in a
    # direction set by the pattern's candles.
    signed = {
        "Doji": doji(last),
        "Dragonfly Doji": doji(last)
        & (last.upper_shadow < last.tiny)
        & (last.lower_shadow > last.tiny),
        "Gravestone Doji": doji(last)
        & (last.lower_shadow < last.tiny)
        & (last.upper_shadow > last.tiny),
        "Long-Legged Doji": doji(last)
        & ((last.lower_shadow > last.body) | (last.upper_shadow > last.body)),
        "Rickshaw Man": doji(last)
        & (last.lower_shadow > last.body)
        & (last.upper_shadow > last.body)
        & (last.bottom <= middle + last.near)
        & (last.top >= middle - last.near),
        "Takuri": doji(last)
        & (last.upper_shadow < last.tiny)
        & (last.lower_shadow > 2.0 * last.body),
        "Marubozu": last.color * marubozu(last),
        "Spinning Top": last.color
        * (
            short_body(last)
            & (last.upper_shadow > last.body)
            & (last.lower_shadow > last.body)
        ),
        "Belt-hold": last.color * (long_body(last) & (open_shadow < last.tiny)),
        "Closing Marubozu": last.color * (long_body(last) & (close_shadow < last.tiny)),
        "High-Wave Candle": last.color
        * (
            short_body(last)
            & (last.upper_shadow > 2.0 * last.body)
            & (last.lower_shadow > 2.0 * last.body)
        ),
        "Long Line Candle": last.color
        * (
            long_body(last)
            & (last.upper_shadow < last.average_shadow)
            & (last.lower_shadow < last.average_shadow)
        ),
        "Short Line Candle": last.color
        * (
            short_body(last)
            & (last.upper_shadow < last.average_shadow)
            & (last.lower_shadow < last.average_shadow)
        ),
        "Inside Bar": last.color * ((last.high < prev.high) & (last.low > prev.low)),
        "Engulfing": last.color * engulfing,
        "Harami": -prev.color
        * harami
        * (long_body(prev) & (last.body <= last.average_body)),
        "Harami Cross": -prev.color * harami * (long_body(prev) & doji(last)),
        "Doji Star": -prev.color
        * (
            long_body(prev)
            & doji(last)
            & (
                (white(prev) & gap_up(last, prev))
                | (black(prev) & gap_down(last, prev))
            )
        ),
        "Counterattack": last.color
        * (
            opposite
            & long_body(prev)
            & long_body(last)
            & within(last.close, prev.close, prev.equal)
        ),
        "Kicking": last.color * kicking,
        # The longer of the two marubozus decides the direction, the first when both are equally long.
        "Kicking by Length": last.color.where(last.body > prev.body, prev.color)
        * kicking,
        "Separating Lines": last.color
        * (
            opposite
            & within(last.open, prev.open, prev.equal)
            & long_body(last)
            & (open_shadow < last.tiny)
        ),
        # A harami followed by a close beyond the first candle's open, against the first candle's colour.
        "Three Inside": -prev2.color
        * (
            long_body(prev2)
            & (prev.body <= prev.average_body)
            & (prev.top < prev2.top)
            & (prev.bottom > prev2.bottom)
            & (
                (white(prev2) & black(last) & (last.close < prev2.open))
                | (black(prev2) & white(last) & (last.close > prev2.open))
            )
        ),
        # An engulfing pattern followed by a close beyond the engulfing candle's close, in its direction.
        "Three Outside": prev.color
        * (
            (prev.color == -prev2.color)
            & (prev.top > prev2.top)
            & (prev.bottom < prev2.bottom)
            & (
                (white(prev) & (last.close > prev.close))
                | (black(prev) & (last.close < prev.close))
            )
        ),
        # A doji whose shadows gap away from both neighbours, in the direction of the last candle.
        "Abandoned Baby": last.color
        * (
            star
            & doji(prev)
            & (
                (
                    white(prev2)
                    & (prev.low > prev2.high)
                    & black(last)
                    & (last.high < prev.low)
                    & (last.close < prev2.close - prev2.body * STAR_PENETRATION)
                )
                | (
                    black(prev2)
                    & (prev.high < prev2.low)
                    & white(last)
                    & (last.low > prev.high)
                    & (last.close > prev2.close + prev2.body * STAR_PENETRATION)
                )
            )
        ),
        # Bullish after a gap up, bearish after a gap down.
        "Side-by-Side White Lines": pd.Series(
            np.where(prev.bottom > prev2.top, 1.0, -1.0), index=last.index
        )
        * (
            white(prev)
            & white(last)
            & (
                (gap_up(prev, prev2) & gap_up(last, prev2))
                | (gap_down(prev, prev2) & gap_down(last, prev2))
            )
            & within(last.body, prev.body, prev.near)
            & within(last.open, prev.open, prev.equal)
        ),
        # The second candle's colour, after a gap in its direction; the last candle opens in the second real
        # body and closes inside the gap.
        "Tasuki Gap": prev.color
        * (
            (
                (
                    gap_up(prev, prev2)
                    & white(prev)
                    & black(last)
                    & (last.close < prev.bottom)
                    & (last.close > prev2.top)
                )
                | (
                    gap_down(prev, prev2)
                    & black(prev)
                    & white(last)
                    & (last.close > prev.top)
                    & (last.close < prev2.bottom)
                )
            )
            & opens_in_previous_body
            & ((prev.body - last.body).abs() < prev.near)
        ),
        # Three dojis against the reference size before the first, the middle one gapping away from the first
        # and the last not beyond the middle: bullish when the middle one gaps down, bearish when it gaps up.
        "Tri-Star": pd.Series(
            np.where(
                gap_down(prev, prev2) & (last.bottom > prev.bottom),
                1.0,
                np.where(gap_up(prev, prev2) & (last.top < prev.top), -1.0, 0.0),
            ),
            index=last.index,
        )
        * (
            (prev2.body <= prev2.tiny)
            & (prev.body <= prev2.tiny)
            & (last.body <= prev2.tiny)
        ),
        # Two candles of one colour with a real body gap, then a candle of the other colour that opens in the
        # second real body and closes in the first, in the direction of the gap.
        "Gap Three Methods": prev2.color
        * (
            (prev2.color == prev.color)
            & opposite
            & opens_in_previous_body
            & (last.close < prev2.top)
            & (last.close > prev2.bottom)
            & (
                (white(prev2) & gap_up(prev, prev2))
                | (black(prev2) & gap_down(prev, prev2))
            )
        ),
        # Three candles of one colour that advance (white, higher closes) or decline (black, lower closes),
        # each opening within or near the previous real body, struck back by a candle of the other colour that
        # opens beyond the third close and closes beyond the first open; in the direction of the three.
        "Three-Line Strike": prev.color
        * (
            (prev3.color == prev2.color)
            & (prev2.color == prev.color)
            & opposite
            & opens_near_body(prev2, prev3)
            & opens_near_body(prev, prev2)
            & (
                (
                    white(prev)
                    & (prev2.close > prev3.close)
                    & (prev.close > prev2.close)
                    & (last.open > prev.close)
                    & (last.close < prev3.open)
                )
                | (
                    black(prev)
                    & (prev2.close < prev3.close)
                    & (prev.close < prev2.close)
                    & (last.open < prev.close)
                    & (last.close > prev3.open)
                )
            )
        ),
        # A long candle, a candle of the same colour gapping away from it, two candles moving further away
        # (the fourth of the same colour), and a candle of the other colour that closes inside the gap; in the
        # direction of the last candle.
        "Breakaway": last.color
        * (
            long_body(prev4)
            & (prev3.color == prev4.color)
            & (prev.color == prev4.color)
            & opposite
            & (
                (
                    black(prev4)
                    & gap_down(prev3, prev4)
                    & (prev2.high < prev3.high)
                    & (prev2.low < prev3.low)
                    & (prev.high < prev2.high)
                    & (prev.low < prev2.low)
                    & (last.close > prev3.open)
                    & (last.close < prev4.close)
                )
                | (
                    white(prev4)
                    & gap_up(prev3, prev4)
                    & (prev2.high > prev3.high)
                    & (prev2.low > prev3.low)
                    & (prev.high > prev2.high)
                    & (prev.low > prev2.low)
                    & (last.close < prev3.open)
                    & (last.close > prev4.close)
                )
            )
        ),
        # A long candle, three short candles of the other colour against it that stay within its range, and a
        # long candle of the first colour that opens beyond the third and closes beyond the first close; in the
        # direction of the first candle.
        "Rising/Falling Three Methods": prev4.color
        * (
            long_body(prev4)
            & (prev3.color == -prev4.color)
            & (prev2.color == prev3.color)
            & (prev.color == prev3.color)
            & (last.color == prev4.color)
            & short_body(prev3)
            & short_body(prev2)
            & short_body(prev)
            & long_body(last)
            & in_range(prev3, prev4)
            & in_range(prev2, prev4)
            & in_range(prev, prev4)
            & (
                (
                    white(prev4)
                    & (prev2.close < prev3.close)
                    & (prev.close < prev2.close)
                    & (last.open > prev.close)
                    & (last.close > prev4.close)
                )
                | (
                    black(prev4)
                    & (prev2.close > prev3.close)
                    & (prev.close > prev2.close)
                    & (last.open < prev.close)
                    & (last.close < prev4.close)
                )
            )
        ),
        # 1 (bullish) or -1 (bearish) on the candle that breaks out of the inside bar, 2 or -2 on the candle
        # that confirms it.
        "Hikkake": with_confirmation(hikkake, prev.high, prev.low),
        "Modified Hikkake": with_confirmation(modified_hikkake, prev.high, prev.low),
    }

    recognised = (
        {name: pattern.astype(float) for name, pattern in bullish.items()}
        | {name: -pattern.astype(float) for name, pattern in bearish.items()}
        | {name: pattern.astype(float) for name, pattern in signed.items()}
    )

    position = pd.Series(np.arange(len(last)), index=last.index)

    # Each pattern is empty during its warm-up; adding 0.0 turns the -0.0 of a negated False into 0.
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
    Calculate the Heikin-Ashi candles: candles built from two-period averages, which smooth the noise so the trend
    stands out.

    The formula is as follows:

    - HA Close = (Open + High + Low + Close) / 4
    - HA Open = (previous HA Open + previous HA Close) / 2, starting from (Open + Close) / 2
      on the first candle
    - HA High = max(High, HA Open, HA Close)
    - HA Low = min(Low, HA Open, HA Close)

    Reference: Valcu, D. (2004). "Using the Heikin-Ashi Technique." Technical Analysis of
    Stocks & Commodities, 22(2).

    Referenced against:
    - TA-Lib, HA: https://ta-lib.org/functions/ha.html

    Shares similar logic:
    - pandas-ta-classic, ha: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/candles/ha.py#L25

    Notes:
        - The HA Open halves the weight of the first candle's seed every period, so it is an
          exponential average with a smoothing factor of 1/2 of the previous HA Closes. The
          values agree with TA-Lib's and pandas-ta-classic's on every candle, from the first candle
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
    Mark Bill Williams' fractals: a high above the highs of the bars on either side of it (an up fractal) or a low
    below their lows (a down fractal).

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

    Referenced against:
    - TA-Lib, FRACTAL: https://ta-lib.org/functions/fractal.html

    Shares similar logic:
    - finta, WILLIAMS_FRACTAL: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L2205

    Notes:
        - This agrees with TA-Lib's FRACTAL (swinghigh, swinglow) divided by 100, which marks the
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


def get_td_sequential(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.DataFrame:
    """
    Count Tom DeMark's TD Sequential: the TD Setup of nine closes in a row beyond the close four bars earlier and
    the TD Countdown to 13 that follows a completed setup.

    The rules are DeMark's:

    - Price Flip: a buy setup starts on a close below the close four bars earlier right after a
      close above the close four bars before that one (a sell setup on the reverse).
    - TD Buy Setup: from the flip on, every close below the close four bars earlier counts 1 to
      9; a close that is not below breaks the setup. The setup is complete on its ninth bar,
      after which the count stops until the next flip. The TD Sell Setup counts closes above
      the close four bars earlier.
    - TD Buy Countdown: starts on the bar that completes a buy setup and counts every bar, not
      necessarily in a row, whose close is at or below the low two bars earlier, up to 13. The
      13th bar must also have a low at or below the close of the eighth countdown bar; a bar
      that misses this only is not counted and the countdown waits at 12. The TD Sell Countdown
      counts closes at or above the high two bars earlier, its 13th bar a high at or above the
      close of the eighth.
    - Cancellation: a buy countdown ends when a sell setup completes (which starts the sell
      countdown) or when a close rises above the TDST level of its setup, the highest true
      high (the high, or the previous close when that is higher) of the nine setup bars; a
      sell countdown ends on a completed buy setup or a close below the lowest true low of its
      setup.

    Reference: DeMark, T.R. (1994). "The New Science of Technical Analysis." Wiley; Perl, J.
    (2008). "DeMark Indicators." Bloomberg Press.

    Referenced against:
    - pandas-ta-classic, td_seq: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/td_seq.py#L31

    Notes:
        - A new setup in the same direction during a countdown does not restart it (DeMark's
          recycling, which compares the size of the two setups, is left out), and the
          countdown has no "perfection" or "intersection" filters.
        - pandas-ta-classic's td_seq counts every run of closes beyond the close four bars
          earlier, also without a price flip before it, and continues past 9 up to 13; on the
          bars of a setup that starts on a flip the counts 1 to 9 are the same.
        - pandas-ta-classic's td_seq counts the TD Setup only.

    Args:
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.

    Returns:
        pd.DataFrame: The Buy Setup and Sell Setup (1 to 9 on the bars of a setup), and the Buy
        Countdown and Sell Countdown (1 to 13 on the bars that count), 0 otherwise and empty for
        the first four bars.
    """
    high = prices_high.to_numpy(dtype=float)
    low = prices_low.to_numpy(dtype=float)
    close = prices_close.to_numpy(dtype=float)
    bars = len(close)

    # 1 for a close above the close four bars earlier, -1 below and 0 otherwise.
    direction = np.zeros(bars)
    direction[TD_SETUP_LAG:] = np.sign(close[TD_SETUP_LAG:] - close[:-TD_SETUP_LAG])
    previous_close = np.concatenate([[np.nan], close[:-1]])
    true_high, true_low = np.fmax(high, previous_close), np.fmin(low, previous_close)

    # Per side, 1 for the sell side (closes above) and -1 for the buy side (closes below).
    setup = {1: np.zeros(bars), -1: np.zeros(bars)}
    countdown = {1: np.zeros(bars), -1: np.zeros(bars)}
    # The closes in a row beyond the close four bars earlier since a price flip.
    run = {1: 0, -1: 0}

    # The countdown under way: its side (0 for none), its count, its setup's TDST level and the close of its
    # eighth bar.
    side, counted, tdst, eighth_close = 0, 0, np.nan, np.nan

    for bar in range(TD_SETUP_LAG + 1, bars):
        for flip_side in (1, -1):
            if direction[bar] != flip_side:
                run[flip_side] = 0
            elif run[flip_side] > 0 or direction[bar - 1] == -flip_side:
                run[flip_side] += 1
            if 0 < run[flip_side] <= TD_SETUP_BARS:
                setup[flip_side][bar] = run[flip_side]

            # A completed setup starts its countdown and cancels the other side's.
            if run[flip_side] == TD_SETUP_BARS and side != flip_side:
                side, counted, eighth_close = flip_side, 0, np.nan
                setup_bars = slice(bar - TD_SETUP_BARS + 1, bar + 1)
                tdst = (
                    true_high[setup_bars].max()
                    if side == -1
                    else true_low[setup_bars].min()
                )

        if side == -1:
            cancelled = close[bar] > tdst
            qualifies = close[bar] <= low[bar - TD_COUNTDOWN_LAG]
            thirteenth = low[bar] <= eighth_close
        elif side == 1:
            cancelled = close[bar] < tdst
            qualifies = close[bar] >= high[bar - TD_COUNTDOWN_LAG]
            thirteenth = high[bar] >= eighth_close
        else:
            continue

        if cancelled:
            side = 0
        elif qualifies and (counted < TD_COUNTDOWN_BARS - 1 or thirteenth):
            counted += 1
            countdown[side][bar] = counted
            if counted == TD_QUALIFIER_BAR:
                eighth_close = close[bar]
            if counted == TD_COUNTDOWN_BARS:
                side = 0

    warm_up = np.arange(bars) < TD_SETUP_LAG

    return pd.DataFrame(
        {
            "Buy Setup": np.where(warm_up, np.nan, setup[-1]),
            "Sell Setup": np.where(warm_up, np.nan, setup[1]),
            "Buy Countdown": np.where(warm_up, np.nan, countdown[-1]),
            "Sell Countdown": np.where(warm_up, np.nan, countdown[1]),
        },
        index=prices_close.index,
    )


def get_liquidity_sweep(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int = 15,
    wick_multiple: float = 1.5,
) -> pd.Series:
    """
    Mark a liquidity sweep: a candle that trades through the lowest low (or highest high) of the previous bars,
    where stop orders tend to rest, and closes back inside with a long wick.

    The formula is as follows:

    - Swing Low (t) = lowest Low of the `window` bars before t, Swing High (t) = highest High
    - Bullish Sweep (1): Low < Swing Low, Close > Swing Low, Close > Open and lower shadow >
      wick_multiple x real body
    - Bearish Sweep (-1): High > Swing High, Close < Swing High, Close < Open and upper shadow >
      wick_multiple x real body

    The liquidity sweep (or stop hunt) is a pattern defined by the ICT (Inner Circle Trader) and
    Smart Money Concepts community; it has no primary publication. The measurable form with a
    swing over the previous 15 bars and a wick of more than 1.5 times the body is
    pandas-ta-classic's.

    Referenced against:
    - pandas-ta-classic, smc_sweep: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/smc_sweep.py#L12

    Args:
        prices_open (pd.Series): The open prices.
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.
        window (int, optional): The number of previous bars the swing low and high are taken
            over. Defaults to 15.
        wick_multiple (float, optional): How many times the real body the wick through the
            swing must exceed. Defaults to 1.5.

    Returns:
        pd.Series: 1 for a bullish sweep, -1 for a bearish sweep and 0 otherwise, empty for the
        first `window` bars.
    """
    swing_low = prices_low.rolling(window).min().shift(1)
    swing_high = prices_high.rolling(window).max().shift(1)
    body = (prices_close - prices_open).abs()
    lower_shadow = np.minimum(prices_open, prices_close) - prices_low
    upper_shadow = prices_high - np.maximum(prices_open, prices_close)

    bullish = (
        (prices_low < swing_low)
        & (prices_close > swing_low)
        & (prices_close > prices_open)
        & (lower_shadow > wick_multiple * body)
    )
    bearish = (
        (prices_high > swing_high)
        & (prices_close < swing_high)
        & (prices_close < prices_open)
        & (upper_shadow > wick_multiple * body)
    )

    return (bullish.astype(float) - bearish.astype(float)).where(
        swing_low.notna() & swing_high.notna()
    )


def get_crossover(series: pd.Series, reference: pd.Series | float) -> pd.Series:
    """
    Mark where a series crosses a reference line: the periods a line moves from below to above it (1) or from above
    to below (-1).

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
    Turn moving averages, the MACD, the RSI, the Bollinger Bands and new highs and lows into the events traders act
    on: the period a line is crossed.

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


# The signals of get_signals, in its order.
SIGNAL_NAMES = [
    "Moving Average Crossover",
    "MACD Crossover",
    "RSI Overbought",
    "RSI Oversold",
    "Upper Bollinger Band",
    "Lower Bollinger Band",
    "New High/Low",
]


def get_period_signals(
    prices: pd.Series,
    period: str,
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
    high_low_window: int | None = None,
) -> pd.DataFrame:
    """
    Detect the signals of get_signals on the prices of one ticker, with a year of periods as
    the window of the new highs and lows unless another window is given.

    Args:
        prices (pd.Series): The prices of one ticker, NaN on the dates it did not trade.
        period (str): The period of the prices, such as "daily" or "weekly".
        short_window (int): The window of the short moving average.
        long_window (int): The window of the long moving average.
        macd_short_window (int): The short window of the MACD.
        macd_long_window (int): The long window of the MACD.
        macd_signal_window (int): The signal window of the MACD.
        rsi_window (int): The window of the RSI.
        rsi_overbought (float): The RSI level above which the price is overbought.
        rsi_oversold (float): The RSI level below which the price is oversold.
        bollinger_window (int): The window of the Bollinger Bands.
        bollinger_std_dev (float): The number of standard deviations of the Bollinger Bands.
        high_low_window (int | None, optional): The window of the new highs and lows.
            Defaults to None, a year of periods.

    Returns:
        pd.DataFrame: The signals on the dates of the prices.
    """
    # A year of periods, which is 365 days for a ticker that trades on weekends.
    if high_low_window is not None:
        window = high_low_window
    elif period in ("daily", "intraday"):
        window = get_observations_per_year(prices)
    else:
        window = PERIODS_PER_YEAR[period]

    return get_signals(
        prices.dropna(),
        short_window=short_window,
        long_window=long_window,
        macd_short_window=macd_short_window,
        macd_long_window=macd_long_window,
        macd_signal_window=macd_signal_window,
        rsi_window=rsi_window,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        bollinger_window=bollinger_window,
        bollinger_std_dev=bollinger_std_dev,
        high_low_window=window,
    ).reindex(prices.index)


def get_indicator_summary(
    values: pd.Series,
    prices: pd.Series,
    below: float | None = None,
    above: float | None = None,
    forward_periods: list[int] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Summarise the history of an indicator: where its latest value ranks, its extremes, the periods spent beyond a
    threshold, the crossings into that zone and the return of the price after each crossing.

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
