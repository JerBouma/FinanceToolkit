"""Pattern Module: candlestick patterns and indicator signals"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.technicals import momentum_model, overlap_model, volatility_model

# TA-Lib's default candle settings (ta_global.c): what a candle part is measured against, over how many
# previous candles it is averaged and the factor applied. "Shadows" averages are halved, as in TA-Lib.
CANDLE_SETTINGS = {
    "BodyLong": ("RealBody", 10, 1.0),
    "BodyShort": ("RealBody", 10, 1.0),
    "BodyDoji": ("HighLow", 10, 0.1),
    "ShadowLong": ("RealBody", 0, 1.0),
    "ShadowVeryShort": ("HighLow", 10, 0.1),
    "Near": ("HighLow", 5, 0.2),
    "Far": ("HighLow", 5, 0.6),
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
}

# How far the third candle of a star, and the second candle of a dark cloud cover, must reach into the first
# candle's real body: TA-Lib's default optInPenetration.
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
) -> pd.DataFrame:
    """
    Recognise 15 candlestick patterns, following the definitions of TA-Lib's CDL functions.

    Each pattern is 1 on a candle that completes it as a bullish pattern, -1 as a bearish
    pattern and 0 otherwise, which is TA-Lib's +100 / -100 divided by 100. TA-Lib's weaker
    form of the engulfing and harami patterns, where one end of the two real bodies is equal,
    is 0.8 (TA-Lib's 80). The doji is indecision rather than a direction and is 1 when present,
    as in TA-Lib; a marubozu and a spinning top take the colour of the candle.

    Sizes are relative, as in TA-Lib (ta_global.c): a real body is long when it is longer than
    the average real body of the 10 previous candles and short when it is shorter; it is a
    doji's body when it is at most 10% of the average high-low range of the 10 previous
    candles; a shadow is very short when it is shorter than 10% of that range and long when it
    is longer than the candle's own real body; "near" is within 20% and "far" at least 60% of
    the average high-low range of the 5 previous candles.

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

    Reference: Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York Institute
    of Finance; Bulkowski, T.N. (2008). "Encyclopedia of Candlestick Charts." Wiley. The
    thresholds are TA-Lib's (github.com/TA-Lib/ta-lib, src/ta_func/ta_CDL*.c).

    Notes:
        - A pattern is empty (NaN) until TA-Lib's lookback has passed: the candles its
          averages and the pattern itself need (10 to 13 candles), so a ticker is evaluated
          from the same candle on as TA-Lib.
        - A pattern only describes the candles; the trend before it, which Nison requires for
          e.g. a hammer (after a decline) and a hanging man (after a rise), is not checked, as
          in TA-Lib.

    Args:
        prices_open (pd.Series): The open prices.
        prices_high (pd.Series): The high prices.
        prices_low (pd.Series): The low prices.
        prices_close (pd.Series): The close prices.

    Returns:
        pd.DataFrame: The patterns as columns, with 1 (bullish), -1 (bearish), 0.8 / -0.8 (the
        weaker engulfing and harami) or 0 per candle.
    """
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

    def previous(series: pd.Series, candles_back: int = 1) -> pd.Series:
        return series.shift(candles_back)

    o1, h1, l1, c1 = previous(o), previous(h), previous(low), previous(c)
    o2, c2 = previous(o, 2), previous(c, 2)
    white1, white2 = previous(white).eq(True), previous(white, 2).eq(True)
    black1, black2 = previous(white).eq(False), previous(white, 2).eq(False)
    body1, body2 = previous(body), previous(body, 2)

    # The hammer family: a small body at one end of a long shadow.
    small_body = body < body_short
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

    patterns = {
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
    }

    position = pd.Series(np.arange(len(candles)), index=candles.index)

    # A pattern is evaluated from the candle on which TA-Lib gives its first value; the -0.0 of a
    # negated False becomes 0.
    return pd.DataFrame(
        {
            name: (pattern + 0.0).where(position >= PATTERN_LOOKBACK[name])
            for name, pattern in patterns.items()
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
