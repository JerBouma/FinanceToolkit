"""Pattern Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.technicals.helpers import (
    FACTOR_COLLECTIONS,
    PERIODS,
    calculate_on_complete_bars,
    calculate_per_ticker,
    report_missing_values,
)
from financetoolkit.technicals.pattern import pattern_model
from financetoolkit.utilities import factor_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_period,
    handle_errors,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
)

# The examples' output tables are wider than the line length.
# ruff: noqa: E501


@report_missing_values
@filter_shown_columns
@track_sources
class PatternTechnicals:
    """
    Candlestick patterns, the signals of the indicators and the summary of an indicator's history.

    The candlestick patterns are those described by Steve Nison, Thomas Bulkowski and Daniel Chesler (the
    hikkake), in the measurable form TA-Lib (Mario Fortier) defines them.
    """

    @handle_errors
    def get_candlestick_patterns(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        patterns: str | list[str] | None = None,
        signals_only: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Recognise 62 classic candlestick patterns in every ticker's candles: the shapes of one to five candles that
        traders read as a reversal or a continuation of the trend.

        Candlestick charting reads the open, high, low and close of a period as one candle: the real body between
        the open and the close and the shadows up to the high and down to the low. Certain shapes and sequences are
        read as a sign that a move is ending or continuing.

        Each pattern is 1 on the candle that completes it as a bullish pattern, -1 as a bearish
        pattern and 0 otherwise (TA-Lib's +100 and -100 divided by 100). The weaker engulfing,
        harami and harami cross, where one end of the two real bodies is equal, are 0.8 and -0.8
        (TA-Lib's 80). The dojis (doji, dragonfly, gravestone and long-legged doji, rickshaw man
        and takuri) are indecision rather than a direction and are 1 when present; a marubozu, a
        spinning top, a belt-hold, a closing marubozu, a high-wave, long line and short line candle
        and an inside bar take the colour of the candle (1 white, -1 black). A hikkake and a
        modified hikkake are 1 or -1 on the candle that completes them and 2 or -2 on the later
        candle that confirms them (TA-Lib's +200 and -200 divided by 100, like every other value).

        The patterns:

        - Doji: an open and close nearly equal.
        - Hammer (bullish) and Hanging Man (bearish): a short real body at the top of a long lower
          shadow, near the previous low or the previous high respectively.
        - Inverted Hammer (bullish) and Shooting Star (bearish): a short real body at the bottom
          of a long upper shadow that gaps down or up from the previous real body.
        - Engulfing: a real body that engulfs the previous real body of the opposite colour.
        - Harami: a short real body inside the previous long real body, opposite to its colour.
        - Piercing Line (bullish) and Dark Cloud Cover (bearish): a long candle followed by one
          of the opposite colour that opens beyond the previous low (high) and closes past the
          middle of the previous real body.
        - Morning Star (bullish) and Evening Star (bearish): a long real body, a short real body
          that gaps away from it and a real body of the opposite colour that closes at least 30%
          into the first.
        - Three White Soldiers (bullish) and Three Black Crows (bearish): three long-enough
          candles of one colour with ever higher (lower) closes, each opening within the previous
          real body.
        - Marubozu: a long real body with (almost) no shadows.
        - Spinning Top: a short real body with both shadows longer than the body.
        - Belt-hold: a long real body without (or with a very short) shadow on the side of the
          open: a white candle opening on its low (bullish) or a black candle opening on its high
          (bearish).
        - Closing Marubozu: a long real body without (or with a very short) shadow on the side of
          the close.
        - Dragonfly Doji and Takuri: a doji at the top of the range, with a lower shadow (a very
          long one, more than twice the body, for the takuri).
        - Gravestone Doji: a doji at the bottom of the range, with an upper shadow.
        - Long-Legged Doji: a doji with at least one long shadow, and Rickshaw Man: a doji with
          two long shadows and the body near the middle of the range.
        - High-Wave Candle: a short real body with very long upper and lower shadows.
        - Long Line Candle and Short Line Candle: a long (short) real body with short shadows.
        - Inside Bar: a candle whose high is below and low is above the previous candle's, a
          price-action pattern (pandas-ta-classic's cdl_inside) and the start of a hikkake.
        - Counterattack: two long real bodies of opposite colour that close at the same price, in
          the direction of the second.
        - Doji Star: a long real body followed by a doji that gaps away from it, opposite to the
          colour of the first candle; Harami Cross: a doji inside the previous long real body,
          opposite to its colour.
        - Homing Pigeon (bullish): a short black real body inside the previous long black one.
        - In-Neck, On-Neck and Thrusting (bearish): a long black candle followed by a white
          candle that opens below its low and closes at its close (in-neck), at its low (on-neck)
          or above its close but not above the middle of its real body (thrusting).
        - Kicking: a marubozu followed by a marubozu of the opposite colour that gaps away from it,
          in the direction of the second; Kicking by Length: in the direction of the longer one.
        - Matching Low (bullish): two black candles that close at the same price.
        - Separating Lines: two candles of opposite colour that open at the same price, the second
          a long real body without (or with a very short) shadow on the side of the open, in its
          direction.
        - Three Inside and Three Outside (Up or Down): a harami or an engulfing pattern confirmed
          by a third candle that closes beyond the first candle's open (inside) or the second
          candle's close (outside).
        - Morning Doji Star (bullish) and Evening Doji Star (bearish): the morning and evening star
          with a doji as the middle candle.
        - Two Crows (bearish): a long white candle, a black candle that gaps above it and a black
          candle that opens within the second real body and closes within the first; Upside Gap
          Two Crows (bearish): a long white candle, a short black candle that gaps above it and a
          black candle that engulfs the second and still closes above the first close.
        - Three Stars in the South (bullish): three black candles that shrink: a long one with a
          long lower shadow, a smaller one with a low within the first's range and a lower
          shadow, and a short one with (almost) no shadows inside the second candle's range.
        - Abandoned Baby: a long real body, a doji whose shadows gap away from it and a candle of
          the opposite colour that gaps away from the doji the other way and closes at least 30%
          into the first body, in the direction of the third candle.
        - Advance Block (bearish) and Stalled Pattern (bearish): three white candles with higher
          closes whose advance weakens: ever shorter bodies or long upper shadows (advance block),
          or two long white candles followed by a short one riding on the second (stalled).
        - Side-by-Side White Lines: two white candles of the same size and open beyond a gap,
          bullish after a gap up and bearish after a gap down.
        - Identical Three Crows (bearish): three declining black candles, each opening at the
          previous close.
        - Stick Sandwich (bullish): two black candles that close at the same price around a white
          candle trading above the first close.
        - Tasuki Gap: a gap followed by a candle in its direction and a candle of the opposite
          colour of about the same size that closes inside the gap, in the direction of the gap.
        - Tri-Star: three dojis, the middle one gapping away; bullish after a gap down, bearish
          after a gap up.
        - Unique Three River (bullish): a long black candle, a black harami with a lower low and a
          short white candle opening above that low.
        - Gap Three Methods (Upside or Downside): two candles of one colour with a gap between them
          and a candle of the opposite colour that closes the gap, in the direction of the gap.
        - Three-Line Strike: three candles of one colour with ever higher (white) or lower (black)
          closes, struck back by a candle of the other colour that closes beyond the first open,
          in the direction of the three.
        - Concealing Baby Swallow (bullish): two black candles without shadows, a black candle
          that gaps down with an upper shadow reaching into the second body, and a black candle
          that engulfs it including its shadows.
        - Breakaway: a long candle, a candle of its colour that gaps away from it, two candles that
          move further away and a candle of the other colour that closes inside the gap, in the
          direction of the last candle.
        - Ladder Bottom (bullish): three black candles with lower opens and closes, a black candle
          with an upper shadow and a white candle that opens above its open and closes above its
          high.
        - Mat Hold (bullish): a long white candle, a short black candle that gaps above it and two
          more short falling candles that hold in the upper half of its body, followed by a white
          candle that closes above all three.
        - Rising/Falling Three Methods: a long candle, three short falling (rising) candles of the
          other colour within its range and a long candle of the first colour that closes beyond
          the first close, in the direction of the first candle.
        - Hikkake: an inside bar followed by a candle that breaks out of it with a lower high and
          low (1, a false breakdown) or a higher high and low (-1, a false breakout), confirmed
          (2 or -2) by the first close beyond the other side of the inside bar within 3 candles;
          Modified Hikkake: the same after two inside bars in a row, the first closing near its
          low (bullish) or high (bearish).

        Sizes are relative to the ticker's own recent candles, as in TA-Lib: a real body is long
        (short) when it is longer (shorter) than the average real body of the 10 previous candles,
        a doji's body is at most 10% and a very short shadow less than 10% of their average
        high-low range, a short shadow is shorter than their average shadow (half the average sum
        of both shadows), a long shadow is longer than the candle's own real body and a very long
        shadow longer than twice that body, "near" is within 20% and "far" at least 60% of the
        average high-low range of the 5 previous candles, and two prices are the same ("equal")
        when they are within 5% of that range.

        For more information, see the following sources:

        - Nison, S. (1991). "Japanese Candlestick Charting Techniques." New York Institute of
          Finance.
        - Bulkowski, T.N. (2008). "Encyclopedia of Candlestick Charts." Wiley.
        - Chesler, D. (2004). "Trading False Moves with the Hikkake Pattern." Active Trader, April 2004.

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

        Also known as: candlestick patterns, candle patterns, CDL, Japanese candlesticks, belt
        hold, dragonfly doji, gravestone doji, long-legged doji, rickshaw man, takuri, high wave,
        inside bar, counterattack lines, meeting lines, doji star, harami cross, homing pigeon,
        in neck, on neck, thrusting line, kicker, matching low, separating lines, three inside up,
        three inside down, three outside up, three outside down, morning doji star, evening doji
        star, two crows, three stars in the south, abandoned baby, advance block, deliberation,
        stalled pattern, upside gap side-by-side white lines, downside gap side-by-side white
        lines, identical three crows, stick sandwich, upside tasuki gap, downside tasuki gap, tri
        star, unique three river bottom, upside gap two crows, upside gap three methods, downside
        gap three methods, three line strike, concealing baby swallow, breakaway, ladder bottom, mat
        hold, rising three methods, falling three methods, hikkake, modified hikkake.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way, so an ex-dividend day doesn't read as a gap. "Close"
                uses the candles as traded (split-adjusted).
            patterns (str | list[str] | None, optional): The patterns to return, e.g. ["Hammer",
                "Engulfing"] or "hammer, engulfing" (in any casing). Defaults to None, which returns
                all 62.
            signals_only (bool, optional): Whether to return only the periods in which at least one
                of the returned patterns occurs for at least one ticker. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The patterns per date, with a column per pattern and ticker: 1 (bullish),
            -1 (bearish), 0.8 / -0.8 (the weaker engulfing, harami and harami cross), 2 / -2 (a
            confirmed hikkake) or 0.

        Notes:
        - The patterns are TA-Lib's definitions (Mario Fortier, https://ta-lib.org/functions/) of the
          patterns described by Nison and Bulkowski, and match TA-Lib candle for candle, including its
          default candle settings and penetrations (30% for the stars, doji stars and abandoned baby,
          50% for the dark cloud cover and the mat hold); the inside bar agrees with pandas-ta-classic's cdl_inside.
        - A pattern is empty until TA-Lib's lookback has passed: the 2 to 14 candles its
          averages and the pattern itself need (1 for the inside bar, which pandas-ta-classic
          sets to 0 on the first candle), counted from the ticker's own first candle.
        - TA-Lib marks the gravestone doji, which candlestick analysis reads as bearish after a
          rise, as 1 like every doji; so does this method.
        - A pattern only describes the candles. The trend before it, which candlestick analysis
          requires for a reversal (a hammer after a decline, a hanging man after a rise), is not
          checked, as in TA-Lib; combine a pattern with a trend measure such as
          `get_moving_average` or `get_signals`. The direction follows TA-Lib: separating lines, a
          continuation pattern in candlestick analysis, follow the second candle, the in-neck,
          on-neck and thrusting, bearish continuation patterns after a decline, are -1 whatever
          the trend, and the tasuki gap, side-by-side white lines and gap three methods follow
          the direction of their gap. The breakaway follows its last candle, and the three-line
          strike, mat hold and rising/falling three methods their first candles.
        - A hikkake is confirmed by a close beyond the inside bar (TA-Lib's measurable form of
          Chesler's breakout) within 3 candles. Its confirmation keeps its sign at twice the size,
          so 1 / -1 marks the setups and 2 / -2 the confirmations.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        patterns = toolkit.technicals.get_candlestick_patterns(
            patterns="Engulfing", signals_only=True
        )

        patterns["Engulfing"].tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-10 |      0 |      0 |           1 |
        | 2025-12-15 |     -1 |      0 |           0 |
        | 2025-12-17 |     -1 |      0 |           0 |
        | 2025-12-19 |      1 |      0 |           0 |
        | 2025-12-31 |      0 |     -1 |           0 |
        """
        self._check_arguments(period, close_column)
        check_period(period, PERIODS)
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        if isinstance(patterns, str):
            patterns = [pattern.strip() for pattern in patterns.split(",")]
        # Matched in any casing, as the names are typed by hand ("belt-hold" for "Belt-hold").
        by_lowercase = {name.lower(): name for name in pattern_model.PATTERN_LOOKBACK}
        patterns = [by_lowercase.get(name.lower(), name) for name in patterns or []]
        unknown = set(patterns) - set(pattern_model.PATTERN_LOOKBACK)
        if unknown:
            raise ValueError(
                f"Unknown pattern(s) {sorted(unknown)}, choose from "
                f"{list(pattern_model.PATTERN_LOOKBACK)}."
            )
        patterns = patterns or list(pattern_model.PATTERN_LOOKBACK)

        historical_data = self._get_historical_data(period, close_column)

        # Each ticker is evaluated on its own candles, so a later listing or a missing
        # day doesn't shift the candles a pattern compares.
        candlestick_patterns = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            calculate_on_complete_bars,
            indicator=pattern_model.get_candlestick_patterns,
            patterns=patterns,
        ).loc[self._start_date : self._end_date][patterns]
        candlestick_patterns = candlestick_patterns.reindex(
            columns=historical_data[close_column].columns, level=1
        )

        if signals_only:
            # A pattern is a rare event, so most periods are all zeros; the empty warm-up
            # (NaN) doesn't count as a signal either.
            candlestick_patterns = candlestick_patterns[
                candlestick_patterns.abs().gt(0).any(axis=1)
            ]

        return candlestick_patterns

    @handle_errors
    def get_heikin_ashi(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Heikin-Ashi ("average bar") candles of every ticker: candles built from two-period averages,
        which smooth the noise so the trend stands out.

        Each candle averages the prices of its period, and its open is the midpoint of the previous Heikin-Ashi
        candle's body, so the candles smooth out the noise of single periods: a trend shows as a run of candles of
        one colour, often without a shadow on the side of the open, and a small body with long shadows on both sides
        hints at a turn.

        The formula is as follows:

        - HA Close = (Open + High + Low + Close) / 4
        - HA Open = (previous HA Open + previous HA Close) / 2, starting from (Open + Close) / 2
          on the first candle
        - HA High = max(High, HA Open, HA Close)
        - HA Low = min(Low, HA Open, HA Close)

        The HA Open is an exponential average of the previous HA Closes with a smoothing factor
        of 1/2.

        For more information, see the following sources:

        - Valcu, D. (2004). "Using the Heikin-Ashi Technique." Technical Analysis of Stocks &
          Commodities, 22(2).

        Referenced against:
        - TA-Lib, HA: https://ta-lib.org/functions/ha.html

        Shares similar logic:
        - pandas-ta-classic, ha: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/candles/ha.py#L25

        Also known as: Heikin-Ashi, Heikin Ashi, Heiken Ashi, HA candles, average bar.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way. "Close" uses the candles as traded (split-adjusted).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Heikin-Ashi Open, High, Low and Close per date and ticker.

        Notes:
        - The values agree with TA-Lib's HA and pandas-ta-classic's ha on every candle, from the
          ticker's own first candle on.
        - The first candle's open is the seed (Open + Close) / 2, whose weight halves every
          period, so the candles of the first few periods after the Toolkit's start date depend
          on where the history starts.
        - The Heikin-Ashi prices are averages, not traded prices: an order is filled at the
          prices of the actual candles.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        heikin_ashi = toolkit.technicals.get_heikin_ashi()

        heikin_ashi.xs("AAPL", level=1, axis="columns").tail()
        ```

        Which returns:

        | Date       |    Open |    High |     Low |   Close |
        |:-----------|--------:|--------:|--------:|--------:|
        | 2025-12-24 | 270.938 | 274.474 | 270.938 | 272.496 |
        | 2025-12-26 | 271.717 | 274.414 | 271.717 | 272.996 |
        | 2025-12-29 | 272.357 | 273.408 | 271.404 | 272.341 |
        | 2025-12-30 | 272.349 | 273.128 | 271.335 | 272.115 |
        | 2025-12-31 | 272.232 | 272.73  | 270.807 | 271.641 |
        """
        self._check_arguments(period, close_column)
        check_period(period, PERIODS)
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        # The HA Open carries over from candle to candle, so it starts on the ticker's own
        # first complete candle.
        heikin_ashi = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            calculate_on_complete_bars,
            indicator=pattern_model.get_heikin_ashi,
        ).loc[self._start_date : self._end_date][["Open", "High", "Low", "Close"]]
        heikin_ashi = heikin_ashi.reindex(
            columns=historical_data[close_column].columns, level=1
        )

        return apply_rounding(
            heikin_ashi, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_williams_fractal(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        left_bars: int = 2,
        right_bars: int = 2,
        signals_only: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Mark Bill Williams' fractals in every ticker's prices: an up fractal is a high above the highs of the two
        bars before and the two bars after it, a down fractal a low below their lows.

        They mark the swing highs and lows of a market; Williams buys a break above the last up fractal and sells a
        break below the last down fractal.

        A fractal needs the bars after it, so it is only known `right_bars` periods later. It is
        therefore marked on the bar that confirms it, two periods after the high or low with the
        defaults, and never on the bar of the high or low itself: the value on a date only uses
        the prices up to that date, so it can be used in a backtest as it is. The fractal's price
        is the High (Low) `right_bars` periods before the marked date.

        The formula is as follows:

        - Up Fractal (t) = 1 if High (t - right_bars) > every other High from t - left_bars -
          right_bars to t, else 0
        - Down Fractal (t) = 1 if Low (t - right_bars) < every other Low from t - left_bars -
          right_bars to t, else 0

        For more information, see the following sources:

        - Williams, B.M. (1995). "Trading Chaos: Applying Expert Techniques to Maximize Your
          Profits." Wiley.

        Referenced against:
        - TA-Lib, FRACTAL: https://ta-lib.org/functions/fractal.html

        Shares similar logic:
        - finta, WILLIAMS_FRACTAL: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L2205

        Also known as: Williams Fractal, fractals, swing high, swing low, FRACTAL.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted for
                dividends in the same way. "Close" uses the prices as traded (split-adjusted).
            left_bars (int, optional): The bars before the high or low it must exceed. Defaults to 2.
            right_bars (int, optional): The bars after the high or low it must exceed, which is
                also the delay before a fractal is confirmed. Defaults to 2.
            signals_only (bool, optional): Whether to return only the periods in which at least one
                ticker has a fractal. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Up Fractal and Down Fractal per date and ticker, 1 on the period that
            confirms a fractal and 0 otherwise.

        Notes:
        - The fractals agree with TA-Lib's FRACTAL (swinghigh and swinglow, 100 there), which also marks
          the confirming bar. finta's WILLIAMS_FRACTAL marks the bar of the high or low itself,
          which uses the prices of the following bars (look-ahead).
        - The comparison is strict, as in TA-Lib: a high equal to one of the other highs is no
          fractal.
        - The first left_bars + right_bars periods of a ticker are empty.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        fractals = toolkit.technicals.get_williams_fractal(signals_only=True)

        fractals["Up Fractal"].tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-12-19 |      0 |      0 |           0 |
        | 2025-12-22 |      0 |      1 |           0 |
        | 2025-12-26 |      0 |      0 |           0 |
        | 2025-12-29 |      1 |      1 |           0 |
        | 2025-12-30 |      0 |      0 |           1 |
        """
        self._check_arguments(period, close_column)
        check_period(period, PERIODS)
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        # The bars around a high are the ticker's own complete bars.
        fractals = calculate_per_ticker(
            historical_data,
            ["High", "Low"],
            calculate_on_complete_bars,
            indicator=pattern_model.get_williams_fractal,
            left_bars=left_bars,
            right_bars=right_bars,
        ).loc[self._start_date : self._end_date][["Up Fractal", "Down Fractal"]]
        fractals = fractals.reindex(
            columns=historical_data[close_column].columns, level=1
        )

        if signals_only:
            # A fractal is a rare event, so most periods are all zeros.
            fractals = fractals[fractals.abs().gt(0).any(axis=1)]

        return fractals

    @handle_errors
    def get_td_sequential(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Count Tom DeMark's TD Sequential for every ticker: the TD Setup of nine closes in a row beyond the close
        four bars earlier, and the TD Countdown to 13 that follows it.

        DeMark designed it to find where a trend exhausts itself: a completed buy setup (nine lower closes) and
        above all a completed buy countdown (13) point to a low, a completed sell setup and countdown to a high.

        The rules are DeMark's:

        - Price Flip: a buy setup starts on a close below the close four bars earlier right
          after a close above the close four bars before that one, a sell setup on the reverse.
        - TD Buy Setup: from the flip on, every close below the close four bars earlier counts 1
          to 9, and a close that is not below breaks it. The setup is complete on its ninth bar;
          the count then stops until the next flip. The TD Sell Setup counts closes above the
          close four bars earlier.
        - TD Buy Countdown: starts on the bar that completes a buy setup and counts every bar, not
          necessarily in a row, that closes at or below the low two bars earlier, up to 13. The
          13th bar also needs a low at or below the close of the eighth countdown bar, else the
          countdown waits at 12. The TD Sell Countdown counts closes at or above the high two bars
          earlier, its 13th bar a high at or above the close of the eighth.
        - Cancellation: a buy countdown ends when a sell setup completes (which starts the sell
          countdown) or when a close rises above the TDST level of its setup, the highest true
          high (the high, or the previous close when higher) of its nine bars; a sell countdown
          ends on a completed buy setup or a close below the lowest true low of its setup.

        For more information, see the following sources:

        - DeMark, T.R. (1994). "The New Science of Technical Analysis." Wiley.
        - Perl, J. (2008). "DeMark Indicators." Bloomberg Press.

        Referenced against:
        - pandas-ta-classic, td_seq: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/td_seq.py#L31

        TD Sequential, TD Setup, TD Countdown and TDST are Tom DeMark's names for his indicator
        and its parts.

        Also known as: TD Sequential, DeMark Sequential, Tom DeMark, TD Setup, TD Countdown, TDST,
        TD9, TD13, DeMark 9 13, td_seq.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the high and low prices are adjusted for
                dividends in the same way. "Close" uses the prices as traded (split-adjusted).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Buy Setup and Sell Setup (1 to 9 on the bars of a setup) and the Buy
            Countdown and Sell Countdown (1 to 13 on the bars that count) per date and ticker, 0
            otherwise.

        Notes:
        - A new setup in the same direction during a countdown does not restart it: DeMark's
          recycling, which compares the size of the two setups, is left out, as are the
          "perfection" of a setup and the "intersection" filter of the countdown.
        - pandas-ta-classic's td_seq counts every run of closes beyond the close four bars
          earlier, also one without a price flip before it, and continues past 9 up to 13; on the
          bars of a setup that starts on a flip, its counts 1 to 9 are the same as here.
        - The counts carry over from the ticker's own first bar, so the first setups after the
          Toolkit's start date depend on where the history starts; the first four bars are empty.
        - pandas-ta-classic's td_seq counts the TD Setup only.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        td_sequential = toolkit.technicals.get_td_sequential()

        td_sequential.xs("AAPL", level=1, axis="columns").tail()
        ```

        Which returns:

        | Date       |   Buy Setup |   Sell Setup |   Buy Countdown |   Sell Countdown |
        |:-----------|------------:|-------------:|----------------:|-----------------:|
        | 2025-12-24 |           0 |            2 |               0 |                0 |
        | 2025-12-26 |           1 |            0 |               0 |                1 |
        | 2025-12-29 |           0 |            1 |               0 |                0 |
        | 2025-12-30 |           0 |            2 |               0 |                0 |
        | 2025-12-31 |           1 |            0 |               0 |                0 |
        """
        self._check_arguments(period, close_column)
        check_period(period, PERIODS)
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        # The counts run over the ticker's own complete bars.
        columns = ["Buy Setup", "Sell Setup", "Buy Countdown", "Sell Countdown"]
        td_sequential = calculate_per_ticker(
            historical_data,
            ["High", "Low", close_column],
            calculate_on_complete_bars,
            indicator=pattern_model.get_td_sequential,
        ).loc[self._start_date : self._end_date][columns]

        return td_sequential.reindex(
            columns=historical_data[close_column].columns, level=1
        )

    @handle_errors
    def get_liquidity_sweep(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        window: int = 15,
        wick_multiple: float = 1.5,
        signals_only: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Mark the liquidity sweeps in every ticker's candles: a candle that trades through the lowest low (or highest
        high) of the previous 15 bars, where stop orders tend to rest, and closes back inside with a long wick.

        Traders read it as a stop hunt that has run its course, so a sweep below the lows (bullish) hints at a move
        up and a sweep above the highs (bearish) at a move down.

        The formula is as follows:

        - Swing Low = lowest Low of the `window` bars before the candle, Swing High = their
          highest High
        - Bullish Sweep (1): Low < Swing Low, Close > Swing Low, Close > Open and lower shadow >
          wick_multiple x real body
        - Bearish Sweep (-1): High > Swing High, Close < Swing High, Close < Open and upper
          shadow > wick_multiple x real body

        The real body is |Close - Open|, the lower shadow runs from the bottom of the body to
        the Low and the upper shadow from the top of the body to the High.

        The liquidity sweep (or stop hunt) is a pattern defined by the ICT (Inner Circle Trader)
        and Smart Money Concepts community, without a primary publication. The measurable form
        with a swing over the previous 15 bars and a wick longer than 1.5 times the body is
        pandas-ta-classic's.

        Also known as: liquidity sweep, liquidity grab, stop hunt, stop run, smart money concepts,
        SMC, ICT, swing failure, smc_sweep.

        Referenced against:
        - pandas-ta-classic, smc_sweep: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/smc_sweep.py#L12

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close", in which case the open, high and low prices are adjusted
                for dividends in the same way. "Close" uses the candles as traded (split-adjusted).
            window (int, optional): The number of previous bars the swing low and high are taken
                over. Defaults to 15.
            wick_multiple (float, optional): How many times the real body the wick through the
                swing must exceed. Defaults to 1.5.
            signals_only (bool, optional): Whether to return only the periods in which at least one
                ticker has a sweep. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The sweeps per date and ticker: 1 (bullish), -1 (bearish) or 0.

        Notes:
        - The values agree with pandas-ta-classic's smc_sweep, also in its first `window` bars, which
          are empty.
        - Smart Money Concepts traders often take the swing low and high from swing points (see
          `get_williams_fractal`) rather than from the extremes of a fixed window.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        sweeps = toolkit.technicals.get_liquidity_sweep(signals_only=True)

        sweeps.tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-10-02 |      0 |     -1 |           0 |
        | 2025-10-03 |      0 |      0 |          -1 |
        | 2025-10-30 |     -1 |      0 |           0 |
        | 2025-11-25 |      0 |      1 |           0 |
        | 2025-12-26 |      0 |      0 |          -1 |
        """
        self._check_arguments(period, close_column, window=window)
        check_period(period, PERIODS)
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._get_historical_data(period, close_column)

        # The swing is taken over the ticker's own complete candles.
        sweeps = calculate_per_ticker(
            historical_data,
            ["Open", "High", "Low", close_column],
            calculate_on_complete_bars,
            indicator=pattern_model.get_liquidity_sweep,
            window=window,
            wick_multiple=wick_multiple,
        ).loc[self._start_date : self._end_date]
        sweeps = sweeps[historical_data[close_column].columns]

        if signals_only:
            # A sweep is a rare event, so most periods are all zeros.
            sweeps = sweeps[sweeps.abs().gt(0).any(axis=1)]

        return sweeps

    @handle_errors
    def get_signals(
        self,
        period: str = "daily",
        close_column: str = "Adj Close",
        short_window: int = 50,
        long_window: int = 200,
        macd_short_window: int = 12,
        macd_long_window: int = 26,
        macd_signal_window: int = 9,
        rsi_window: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        bollinger_window: int = 20,
        bollinger_std_dev: float = 2,
        high_low_window: int | None = None,
        signals_only: bool = False,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Turn moving averages, the MACD, the RSI, the Bollinger Bands and new highs and lows into the events traders
        act on: the period a line is crossed.

        The indicators themselves are available as `get_moving_average`,
        `get_moving_average_convergence_divergence`, `get_relative_strength_index` and `get_bollinger_bands`, and
        give the same values.

        Every signal is 1 on the period a line is crossed upwards, -1 on the period it is crossed
        downwards and 0 otherwise:

        - Moving Average Crossover: the short Simple Moving Average crossing the long one; with
          the default 50 and 200 periods a golden cross (1) or a death cross (-1).
        - MACD Crossover: the MACD line crossing its signal line, the classic trigger of the MACD.
        - RSI Overbought: the RSI entering (1) or leaving (-1) the overbought zone above
          `rsi_overbought`. Leaving it is the classic sell signal.
        - RSI Oversold: the RSI leaving (1) or entering (-1) the oversold zone below
          `rsi_oversold`. Leaving it is the classic buy signal.
        - Upper Bollinger Band: the price breaking out above (1) or falling back below (-1) the
          upper band.
        - Lower Bollinger Band: the price recovering above (1) or breaking down below (-1) the
          lower band.
        - New High/Low: a close above every close of the previous `high_low_window` periods (1),
          a 52-week high with daily prices by default, or below every one of them (-1).

        For more information, see the following books and papers:

        - Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York Institute
          of Finance.
        - Appel, G. (1979). "The Moving Average Convergence-Divergence Trading Method." Signalert
          Corp.
        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Bollinger, J. (2001). "Bollinger on Bollinger Bands." McGraw-Hill.
        - Brock, W., Lakonishok, J., and LeBaron, B. (1992). "Simple Technical Trading Rules and
          the Stochastic Properties of Stock Returns." The Journal of Finance, 47(5), 1731-1764.

        Also known as: trading signals, golden cross, death cross, crossovers, breakouts, MACD
        crossover, RSI signal, 52-week high, new high.

        Args:
            period (str, optional): The time period to consider for historical data. Options are:
                "intraday" (intraday prices), "daily" (daily prices), "weekly" (weekly prices),
                "monthly" (monthly prices), "quarterly" (quarterly prices), "yearly" (yearly prices). Defaults
                to "daily".
            close_column (str, optional): The column name for closing prices in the historical data.
                Defaults to "Adj Close".
            short_window (int, optional): The window of the short moving average. Defaults to 50.
            long_window (int, optional): The window of the long moving average. Defaults to 200.
            macd_short_window (int, optional): The short window of the MACD. Defaults to 12.
            macd_long_window (int, optional): The long window of the MACD. Defaults to 26.
            macd_signal_window (int, optional): The window of the MACD's signal line. Defaults to 9.
            rsi_window (int, optional): The window of the RSI. Defaults to 14.
            rsi_overbought (float, optional): The RSI above which a price is overbought. Defaults to 70.
            rsi_oversold (float, optional): The RSI below which a price is oversold. Defaults to 30.
            bollinger_window (int, optional): The window of the Bollinger Bands. Defaults to 20.
            bollinger_std_dev (float, optional): The number of standard deviations of the Bollinger
                Bands. Defaults to 2.
            high_low_window (int | None, optional): The number of previous periods a new high or
                low is measured against. Defaults to None, which uses a year of periods: 252 daily
                (and intraday), or 365 for a ticker that also trades on weekends such as a
                cryptocurrency, 52 weekly, 12 monthly, 4 quarterly and 1 yearly.
            signals_only (bool, optional): Whether to return only the periods in which at least one
                signal occurs for at least one ticker. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The signals per date, with a column per signal and ticker: 1 (crossed
            upwards), -1 (crossed downwards) or 0.

        Notes:
        - A signal is empty until its indicator has two values to compare, e.g. 200 periods
          for the default long moving average, counted from the ticker's own first price; start
          the Toolkit early enough for the signals of interest.
        - A crossover is the period the line reaches the other side than the side it was on
          last: touching a line and turning back is no signal, while moving through it over two
          periods counts on the period the other side is reached.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2020-01-01",
            end_date="2025-12-31",
        )

        signals = toolkit.technicals.get_signals()

        crossovers = signals["Moving Average Crossover"]
        crossovers[crossovers.ne(0).any(axis=1)].tail()
        ```

        Which returns:

        | Date       |   AAPL |   MSFT |   Benchmark |
        |:-----------|-------:|-------:|------------:|
        | 2025-04-07 |     -1 |      0 |           0 |
        | 2025-04-16 |      0 |      0 |          -1 |
        | 2025-06-09 |      0 |      1 |           0 |
        | 2025-06-27 |      0 |      0 |           1 |
        | 2025-09-15 |      1 |      0 |           0 |
        """
        self._check_arguments(
            period,
            close_column,
            short_window=short_window,
            long_window=long_window,
            macd_short_window=macd_short_window,
            macd_long_window=macd_long_window,
            macd_signal_window=macd_signal_window,
            rsi_window=rsi_window,
            bollinger_window=bollinger_window,
            high_low_window=high_low_window,
        )
        check_period(period, PERIODS)
        if period == "intraday":
            if self._historical_data[period].empty:
                raise InvalidArgumentError(
                    "Please define the 'intraday_period' parameter when initializing the Toolkit."
                )
            close_column = "Close"

        historical_data = self._historical_data[period]

        # The signals in the order the model gives them and the tickers in their own order.
        signals = (
            calculate_per_ticker(
                historical_data,
                [close_column],
                pattern_model.get_period_signals,
                period=period,
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
                high_low_window=high_low_window,
            )
            .loc[self._start_date : self._end_date][pattern_model.SIGNAL_NAMES]
            .reindex(columns=historical_data[close_column].columns, level=1)
        )

        if signals_only:
            # Most periods have no signal at all; the empty warm-up (NaN) doesn't count either.
            signals = signals[signals.abs().gt(0).any(axis=1)]

        return signals

    @handle_errors
    def get_indicator_summary(
        self,
        metric: str = "get_relative_strength_index",
        below: float | None = None,
        above: float | None = None,
        forward_periods: list[int] | int | None = None,
        show_crossings: bool = False,
        column: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Summarise the history of a technical indicator per ticker in a single row instead of the series itself.

        The row holds where the latest value ranks in its own history, the lowest and highest value with their
        dates, how many periods the indicator spent below or above a threshold, when it last crossed into that zone
        and how the price did afterwards. This answers questions such as "how oversold is the RSI compared with the
        last five years, and what happened after the previous times it fell below 30?" without the thousands of
        daily values behind it.

        Each value is ranked against its own history up to that date, so the percentile of a
        crossing in the past only uses the values known at that time:

        - Percentile = share of the values up to and including the date that are at most the
          value on that date
        - Rank = number of values up to and including the date that are at most the value on
          that date: 1 is the lowest value so far, the number of observations the highest
        - Forward Return (n periods) = Price(t + n) / Price(t) - 1, from the close of the
          crossing to the close n trading periods later, on the dividend and split adjusted
          close

        A crossing below `below` is the period the indicator falls below the line, entering the
        zone, and a crossing above `above` the period it rises above the line (the same
        crossings as `get_signals`). The average forward return after the crossings is shown
        next to the average forward return over every period, so the effect of the signal can
        be read against the drift of the share itself.

        For more information, see the following books:

        - Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend Research.
        - Murphy, J.J. (1999). "Technical Analysis of the Financial Markets." New York Institute
          of Finance.

        Also known as: indicator statistics, indicator percentile, oversold history, event
        study, forward returns after a signal, RSI below 30.

        Args:
            metric (str, optional): The indicator method name, e.g. "get_stochastic_oscillator"
                or "get_rate_of_change", with method= among the other parameters for a variant
                (e.g. method="connors"). Defaults to "get_relative_strength_index".
            below (float | None, optional): The line the indicator falls below, e.g. 30 for an
                oversold RSI. Defaults to None.
            above (float | None, optional): The line the indicator rises above, e.g. 70 for an
                overbought RSI. Defaults to None.
            forward_periods (list[int] | int | None, optional): The numbers of trading periods
                after each date to measure the return over, e.g. [5, 21, 63] for about a week,
                a month and a quarter of daily data. Defaults to None, which measures no
                returns.
            show_crossings (bool, optional): Whether to return a row per crossing (ticker and
                date) with its zone, the indicator value, its percentile at the time and the
                forward returns instead of the summary. Defaults to False.
            column (str | None, optional): The output to use of an indicator with several,
                e.g. "Signal Line" of the MACD. Defaults to None, which uses the first output
                other than a price, a histogram or a signal line, e.g. "MACD Line".
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.
            **kwargs: Parameters passed to the indicator, e.g. window=14 or period="weekly".
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The summary per ticker: Latest, Latest Date, Percentile, Rank,
            Observations, Mean, Minimum, Minimum Date, Maximum and Maximum Date; with
            forward_periods the Average n-Period Return; with below (or above) the Periods
            Below, Crossings Below, Last Crossing Below and the Average n-Period Return After
            Crossing Below. With show_crossings a row per ticker and crossing date.

        Notes:
            - The history is the Toolkit's date range, after the warm-up the indicator needs
              (e.g. 14 periods for the RSI), so start early enough for the comparison you are
              after.
            - The forward returns of the last n periods are not known yet: those crossings are
              listed without a return and left out of the averages.
            - Crossings close together overlap in their forward returns, so a handful of
              crossings says little about the signal in general.
            - The returns are measured over the periods of the indicator: trading days by
              default, weeks for weekly periods.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-11-01",
            end_date="2026-11-05",
        )

        toolkit.technicals.get_indicator_summary(
            "get_relative_strength_index", below=30, forward_periods=[5, 21, 63]
        ).T
        ```

        Which returns:

        |                                               | NVDA       |
        |:----------------------------------------------|:-----------|
        | Latest                                        | 25.3948    |
        | Latest Date                                   | 2026-11-05 |
        | Percentile                                    | 0.0008     |
        | Rank                                          | 1          |
        | Observations                                  | 1259       |
        | Mean                                          | 54.7452    |
        | Minimum                                       | 25.3948    |
        | Minimum Date                                  | 2026-11-05 |
        | Maximum                                       | 89.1421    |
        | Maximum Date                                  | 2021-11-08 |
        | Average 5-Period Return                       | 0.0103     |
        | Average 21-Period Return                      | 0.044      |
        | Average 63-Period Return                      | 0.1495     |
        | Periods Below                                 | 12         |
        | Crossings Below                               | 6          |
        | Last Crossing Below                           | 2026-11-05 |
        | Average 5-Period Return After Crossing Below  | 0.0625     |
        | Average 21-Period Return After Crossing Below | 0.0313     |
        | Average 63-Period Return After Crossing Below | 0.1968     |
        """
        names = factor_model.select_metrics(metric, FACTOR_COLLECTIONS)
        if len(names) != 1:
            raise InvalidArgumentError(
                "Please name a single indicator, e.g. 'get_relative_strength_index'."
            )
        if isinstance(forward_periods, int):
            forward_periods = [forward_periods]
        if forward_periods and below is None and above is None:
            raise ValueError(
                "Please set below or above: the forward returns are measured after the "
                "crossings of that line."
            )

        factors = factor_model.get_daily_factors(
            self, names, kwargs, {}, self._tickers, column
        )
        label = factor_model.get_metric_label(names[0])
        if not factors:
            raise ValueError(f"{label} returned no data.")
        values = factors[label]

        period = kwargs.get("period", "daily")
        prices = factor_model.to_daily_factor(
            self._historical_data[period][
                "Close" if period == "intraday" else "Adj Close"
            ],
            self._tickers,
        )

        summaries, crossings = {}, {}
        for ticker in values.columns:
            summaries[ticker], crossings[ticker] = pattern_model.get_indicator_summary(
                values[ticker],
                prices[ticker] if ticker in prices else pd.Series(dtype=float),
                below,
                above,
                forward_periods,
            )

        if show_crossings:
            crossings = {
                ticker: table for ticker, table in crossings.items() if not table.empty
            }
            result = (
                pd.concat(crossings, names=["Ticker", "Date"])
                if crossings
                else pd.DataFrame()
            )
        else:
            result = pd.DataFrame(summaries).T.infer_objects()

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )
