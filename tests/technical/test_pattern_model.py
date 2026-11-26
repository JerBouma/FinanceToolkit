"""Candlestick Pattern and Signal Tests"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.pattern import pattern_model


def candles_after_a_calm_market(*pattern_candles):
    """Fifteen identical white candles (body 1, shadows 0.5) followed by the given candles.

    Against the calm candles a long real body is longer than 1, a short one shorter than 1, a
    doji's body and a very short shadow at most 0.2, "near" 0.4 and "far" 1.2.
    """
    calm = [(100.0, 101.5, 99.5, 101.0)] * 15
    rows = calm + [
        (open_, max(open_, close) + upper, min(open_, close) - lower, close)
        for open_, close, upper, lower in pattern_candles
    ]
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    return pattern_model.get_candlestick_patterns(
        frame["Open"], frame["High"], frame["Low"], frame["Close"]
    )


def test_a_hammer_near_the_previous_low():
    # A small body at the top of a long lower shadow, closing near the previous low (99.5).
    patterns = candles_after_a_calm_market((99.4, 99.6, 0.05, 0.9))

    assert patterns["Hammer"].iloc[-1] == 1
    assert patterns["Hanging Man"].iloc[-1] == 0


def test_a_white_body_engulfing_a_black_one_and_its_weaker_form():
    strict = candles_after_a_calm_market(
        (101.0, 100.0, 0.1, 0.1), (99.8, 101.2, 0.1, 0.1)
    )
    # Opening at the previous close: one end of the bodies is equal, TA-Lib's 80.
    weaker = candles_after_a_calm_market(
        (101.0, 100.0, 0.1, 0.1), (100.0, 101.2, 0.1, 0.1)
    )

    assert strict["Engulfing"].iloc[-1] == 1
    assert weaker["Engulfing"].iloc[-1] == 0.8


def test_a_short_body_inside_a_long_white_one_is_a_bearish_harami():
    patterns = candles_after_a_calm_market(
        (100.0, 103.0, 0.1, 0.1), (101.0, 101.5, 0.1, 0.1)
    )

    assert patterns["Harami"].iloc[-1] == -1


def test_a_morning_star():
    patterns = candles_after_a_calm_market(
        (103.0, 100.0, 0.1, 0.1),  # a long black body
        (99.5, 99.6, 0.1, 0.1),  # a short body gapping down
        (99.8, 102.0, 0.1, 0.1),  # a white body closing 30% into the first
    )

    assert patterns["Morning Star"].iloc[-1] == 1
    assert patterns["Evening Star"].iloc[-1] == 0


def test_three_black_crows_after_a_white_candle():
    patterns = candles_after_a_calm_market(
        (101.0, 100.0, 0.1, 0.05),
        (100.5, 99.5, 0.1, 0.05),
        (99.9, 99.0, 0.1, 0.05),
    )

    assert patterns["Three Black Crows"].iloc[-1] == -1


def test_a_marubozu_and_a_spinning_top_take_the_colour_of_the_candle():
    marubozu = candles_after_a_calm_market((103.0, 100.0, 0.05, 0.05))
    spinning_top = candles_after_a_calm_market((100.6, 100.4, 0.8, 0.8))

    assert marubozu["Marubozu"].iloc[-1] == -1
    assert spinning_top["Spinning Top"].iloc[-1] == -1


def test_patterns_are_empty_until_the_lookback_of_ta_lib():
    patterns = candles_after_a_calm_market()

    for name, lookback in pattern_model.PATTERN_LOOKBACK.items():
        assert patterns[name].iloc[:lookback].isna().all()
        assert patterns[name].iloc[lookback:].notna().all()
    # Identical white candles of an average size form no pattern.
    assert (patterns.iloc[14:] == 0).all().all()


def test_a_crossover_marks_the_period_a_line_is_crossed():
    crossover = pattern_model.get_crossover(pd.Series([1.0, 2.0, 3.0, 2.0, 1.0]), 2.5)

    assert np.isnan(crossover.iloc[0])
    assert crossover.iloc[1:].tolist() == [0.0, 1.0, -1.0, 0.0]


def test_touching_a_line_and_turning_back_is_no_crossover():
    crossover = pattern_model.get_crossover(pd.Series([1.0, 2.5, 1.0]), 2.5)

    assert crossover.iloc[1:].tolist() == [0.0, 0.0]


@pytest.fixture(name="rise_and_fall")
def fixture_rise_and_fall():
    return pd.Series(np.r_[np.linspace(100, 200, 60), np.linspace(200, 100, 60)])


def test_a_rise_and_fall_gives_a_golden_and_a_death_cross(rise_and_fall):
    signals = pattern_model.get_signals(
        rise_and_fall,
        short_window=5,
        long_window=20,
        macd_short_window=12,
        macd_long_window=26,
        macd_signal_window=9,
        rsi_window=14,
        rsi_overbought=70,
        rsi_oversold=30,
        bollinger_window=20,
        bollinger_std_dev=2,
        high_low_window=10,
    )

    crossovers = signals["Moving Average Crossover"]
    assert crossovers.isna().sum() == 20
    # A steady rise has the short average above the long one throughout, so only the fall
    # crosses: a death cross after the top.
    assert crossovers.eq(-1).sum() == 1
    assert crossovers.idxmin() > 59
    # Every close of the rise is a new high and every close of the fall after the top a new low.
    assert (signals["New High/Low"].iloc[10:60] == 1).all()
    assert (signals["New High/Low"].iloc[70:] == -1).all()
    assert signals["RSI Overbought"].eq(-1).sum() == 1


def test_moving_through_a_line_over_two_periods_is_one_crossover():
    crossover = pattern_model.get_crossover(pd.Series([1.0, 2.5, 3.0, 2.5, 3.0]), 2.5)

    assert crossover.iloc[1:].tolist() == [0.0, 1.0, 0.0, 0.0]


def indicator_with_three_dips():
    """An RSI-like series that falls below 30 three times; the price rises by 1 a day."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    values = pd.Series([50, 40, 25, 20, 35, 28, 30, 45, 29, 60.0], index=dates)
    prices = pd.Series(np.arange(100.0, 110.0), index=dates)
    return values, prices


def test_indicator_summary_ranks_the_latest_value_in_its_history():
    values, prices = indicator_with_three_dips()
    summary, _ = pattern_model.get_indicator_summary(values, prices)

    assert summary["Latest"] == 60
    assert summary["Latest Date"] == "2024-01-12"
    assert summary["Percentile"] == 1
    assert summary["Rank"] == 10
    assert summary["Observations"] == 10
    assert summary["Minimum"] == 20
    assert summary["Minimum Date"] == "2024-01-04"
    assert summary["Maximum Date"] == "2024-01-12"
    assert summary["Mean"] == pytest.approx(36.2)
    assert "Periods Below" not in summary


def test_indicator_summary_counts_the_crossings_and_their_forward_returns():
    values, prices = indicator_with_three_dips()
    summary, crossings = pattern_model.get_indicator_summary(
        values, prices, below=30, forward_periods=[2]
    )

    # 25, 20, 28 and 29 are below 30; 30 itself is not.
    assert summary["Periods Below"] == 4
    # Exactly 30 after 28 keeps the side it came from, so 45 -> 29 is the third crossing.
    assert list(crossings.index.strftime("%Y-%m-%d")) == [
        "2024-01-03",
        "2024-01-08",
        "2024-01-11",
    ]
    assert summary["Crossings Below"] == 3
    assert summary["Last Crossing Below"] == "2024-01-11"

    # Without look-ahead: 28 on 2024-01-08 is ranked among the six values up to then.
    assert crossings["Percentile"].tolist() == pytest.approx([1 / 3, 3 / 6, 4 / 9])

    returns = crossings["2-Period Return"]
    assert returns.iloc[0] == pytest.approx(104 / 102 - 1)
    assert returns.iloc[1] == pytest.approx(107 / 105 - 1)
    # Two periods after the last crossing lie beyond the data.
    assert np.isnan(returns.iloc[2])
    assert summary["Average 2-Period Return After Crossing Below"] == pytest.approx(
        returns.iloc[:2].mean()
    )
    every_day = (prices.shift(-2) / prices - 1).mean()
    assert summary["Average 2-Period Return"] == pytest.approx(every_day)


def test_indicator_summary_of_the_zone_above():
    values, prices = indicator_with_three_dips()
    summary, crossings = pattern_model.get_indicator_summary(values, prices, above=40)

    assert summary["Periods Above"] == 3
    assert summary["Crossings Above"] == 2
    assert crossings["Zone"].eq("Above").all()


def test_single_candle_patterns_of_a_long_body():
    # Long bodies (2 against an average of 1) with a very short (0.05) or short (0.3) shadow.
    white_belt_hold = candles_after_a_calm_market((100.0, 102.0, 0.3, 0.05))
    black_closing_marubozu = candles_after_a_calm_market((102.0, 100.0, 0.3, 0.05))
    long_line = candles_after_a_calm_market((100.0, 102.0, 0.3, 0.3))

    assert white_belt_hold["Belt-hold"].iloc[-1] == 1
    assert white_belt_hold["Closing Marubozu"].iloc[-1] == 0
    # A black candle's open side is its upper shadow, 0.3 here: no belt-hold.
    assert black_closing_marubozu["Closing Marubozu"].iloc[-1] == -1
    assert black_closing_marubozu["Belt-hold"].iloc[-1] == 0
    # Shadows below the average shadow (0.5) and a long body.
    assert long_line["Long Line Candle"].iloc[-1] == 1
    assert long_line["Belt-hold"].iloc[-1] == 0


def test_single_candle_patterns_of_a_short_body():
    short_line = candles_after_a_calm_market((100.5, 100.0, 0.3, 0.3))
    high_wave = candles_after_a_calm_market((100.0, 100.5, 1.2, 1.2))

    assert short_line["Short Line Candle"].iloc[-1] == -1
    assert short_line["High-Wave Candle"].iloc[-1] == 0
    # Both shadows longer than twice the body of 0.5.
    assert high_wave["High-Wave Candle"].iloc[-1] == 1
    assert high_wave["Short Line Candle"].iloc[-1] == 0


def test_the_dojis():
    dragonfly = candles_after_a_calm_market((100.0, 100.05, 0.05, 1.5))
    gravestone = candles_after_a_calm_market((100.0, 100.05, 1.5, 0.05))
    rickshaw_man = candles_after_a_calm_market((100.0, 100.05, 1.0, 1.0))

    # A very long lower shadow (more than twice the body) also makes the dragonfly a takuri.
    assert dragonfly[["Doji", "Dragonfly Doji", "Takuri", "Long-Legged Doji"]].iloc[
        -1
    ].tolist() == [1, 1, 1, 1]
    assert dragonfly[["Gravestone Doji", "Rickshaw Man"]].iloc[-1].tolist() == [0, 0]
    assert gravestone["Gravestone Doji"].iloc[-1] == 1
    assert gravestone[["Dragonfly Doji", "Takuri"]].iloc[-1].tolist() == [0, 0]
    # Two long shadows with the body in the middle of the range.
    assert rickshaw_man[["Rickshaw Man", "Long-Legged Doji"]].iloc[-1].tolist() == [
        1,
        1,
    ]
    assert rickshaw_man["Dragonfly Doji"].iloc[-1] == 0


def test_an_inside_bar_takes_the_colour_of_the_candle():
    # High 101.0 below the previous 101.5 and low 99.7 above the previous 99.5.
    patterns = candles_after_a_calm_market((100.5, 100.2, 0.5, 0.5))

    assert patterns["Inside Bar"].iloc[-1] == -1
    assert np.isnan(patterns["Inside Bar"].iloc[0])


def test_the_patterns_selector_keeps_the_order_given():
    frame = pd.DataFrame(
        [(100.0, 101.5, 99.5, 101.0)] * 15, columns=["Open", "High", "Low", "Close"]
    )
    patterns = pattern_model.get_candlestick_patterns(
        frame["Open"], frame["High"], frame["Low"], frame["Close"], ["Takuri", "Doji"]
    )

    assert list(patterns.columns) == ["Takuri", "Doji"]
    with pytest.raises(ValueError, match="Unknown pattern"):
        pattern_model.get_candlestick_patterns(
            frame["Open"], frame["High"], frame["Low"], frame["Close"], ["Hamer"]
        )


def test_single_candle_patterns_match_ta_lib_on_the_test_data():
    data = pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )
    candles = (
        data[["Open", "High", "Low", "Close"]].xs("AAPL", axis=1, level=1).dropna()
    )
    patterns = pattern_model.get_candlestick_patterns(
        candles["Open"], candles["High"], candles["Low"], candles["Close"]
    )

    # TA-Lib 0.8.3's CDL functions on these candles: (+100 count, -100 count).
    ta_lib_counts = {
        "Belt-hold": (62, 42),
        "Closing Marubozu": (76, 64),
        "Dragonfly Doji": (10, 0),
        "Gravestone Doji": (5, 0),
        "High-Wave Candle": (39, 44),
        "Long-Legged Doji": (96, 0),
        "Long Line Candle": (98, 63),
        "Rickshaw Man": (77, 0),
        "Short Line Candle": (54, 42),
        "Takuri": (10, 0),
    }
    for name, (bullish, bearish) in ta_lib_counts.items():
        assert (patterns[name].eq(1).sum(), patterns[name].eq(-1).sum()) == (
            bullish,
            bearish,
        ), name


def test_heikin_ashi_matches_ta_lib():
    candles = pd.DataFrame(
        {
            "Open": [72.48, 74.06, 74.29, 73.45, 74.96, 74.29, 76.81, 77.65],
            "High": [73.42, 75.15, 75.15, 74.99, 75.22, 76.11, 77.61, 78.17],
            "Low": [72.38, 73.8, 74.13, 73.19, 74.37, 74.29, 76.55, 77.06],
            "Close": [73.41, 75.09, 74.36, 74.95, 74.6, 75.8, 77.41, 77.58],
        }
    )
    heikin_ashi = pattern_model.get_heikin_ashi(
        candles["Open"], candles["High"], candles["Low"], candles["Close"]
    )

    # TA-Lib 0.8.3's HA on the same candles.
    assert heikin_ashi["Open"].tolist() == pytest.approx(
        [72.945, 72.93375, 73.729375, 74.1059375, 74.12546875, 74.456484375,
         74.7894921875, 75.94224609375]
    )  # fmt: skip
    assert heikin_ashi["Close"].tolist() == pytest.approx(
        [72.9225, 74.525, 74.4825, 74.145, 74.7875, 75.1225, 77.095, 77.615]
    )
    assert heikin_ashi["High"].tolist() == pytest.approx(candles["High"].tolist())
    assert heikin_ashi["Low"].tolist() == pytest.approx(
        [72.38, 72.93375, 73.729375, 73.19, 74.12546875, 74.29, 74.7894921875,
         75.94224609375]
    )  # fmt: skip


def test_williams_fractal_is_marked_on_the_confirming_bar():
    high = pd.Series([1.0, 2.0, 5.0, 3.0, 2.0, 2.5, 2.0])
    low = pd.Series([3.0, 2.0, 1.0, 2.0, 2.5, 3.0, 2.8])
    fractals = pattern_model.get_williams_fractal(high, low)

    # The high of 5 and the low of 1 at bar 2 are confirmed two bars later, at bar 4.
    assert fractals["Up Fractal"].iloc[:4].isna().all()
    assert fractals["Up Fractal"].iloc[4:].tolist() == [1, 0, 0]
    assert fractals["Down Fractal"].iloc[4:].tolist() == [1, 0, 0]
    # Without look-ahead: the bars up to the confirmation give the same value.
    truncated = pattern_model.get_williams_fractal(high.iloc[:5], low.iloc[:5])
    assert truncated["Up Fractal"].iloc[-1] == 1


def test_williams_fractal_matches_ta_lib_on_the_test_data():
    data = pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )
    prices = data[["High", "Low"]].xs("AAPL", axis=1, level=1).dropna().iloc[:30]
    fractals = pattern_model.get_williams_fractal(prices["High"], prices["Low"])

    # TA-Lib 0.8.3's FRACTAL (2, 2): the positions of its swinghigh and swinglow.
    positions = np.arange(len(prices))
    assert positions[fractals["Up Fractal"].eq(1).to_numpy()].tolist() == [
        11,
        18,
        21,
        27,
    ]
    assert positions[fractals["Down Fractal"].eq(1).to_numpy()].tolist() == [
        5,
        12,
        19,
        24,
        29,
    ]


def test_kicking_and_kicking_by_length():
    # A black marubozu and a white one gapping above its high, with bodies of the same length.
    patterns = candles_after_a_calm_market(
        (101.0, 98.5, 0.05, 0.05), (101.5, 104.0, 0.05, 0.05)
    )
    longer_white = candles_after_a_calm_market(
        (101.0, 98.5, 0.05, 0.05), (101.5, 104.5, 0.05, 0.05)
    )

    assert patterns["Kicking"].iloc[-1] == 1
    # The white body is not longer than the black one, so the black marubozu decides.
    assert patterns["Kicking by Length"].iloc[-1] == -1
    assert longer_white["Kicking by Length"].iloc[-1] == 1


def test_a_white_candle_opening_below_a_long_black_one():
    # A long black candle closing at 99.0 with its low at 98.9; "equal" is 0.1 here.
    first = (101.0, 99.0, 0.1, 0.1)
    in_neck = candles_after_a_calm_market(first, (98.5, 99.05, 0.1, 0.1))
    on_neck = candles_after_a_calm_market(first, (98.0, 98.9, 0.1, 0.1))
    thrusting = candles_after_a_calm_market(first, (98.0, 99.6, 0.1, 0.1))

    names = ["In-Neck", "On-Neck", "Thrusting"]
    assert in_neck[names].iloc[-1].tolist() == [-1, 0, 0]
    assert on_neck[names].iloc[-1].tolist() == [0, -1, 0]
    assert thrusting[names].iloc[-1].tolist() == [0, 0, -1]


def test_two_black_candles_matching_low_and_homing_pigeon():
    matching_low = candles_after_a_calm_market(
        (101.0, 99.0, 0.1, 0.1), (100.5, 99.02, 0.1, 0.1)
    )
    homing_pigeon = candles_after_a_calm_market(
        (101.0, 99.0, 0.1, 0.1), (100.5, 99.8, 0.1, 0.1)
    )

    assert matching_low[["Matching Low", "Homing Pigeon"]].iloc[-1].tolist() == [1, 0]
    assert homing_pigeon[["Matching Low", "Homing Pigeon"]].iloc[-1].tolist() == [0, 1]


def test_counterattack_and_separating_lines():
    # A long white candle closing at the previous close, and one opening at the previous open.
    counterattack = candles_after_a_calm_market(
        (102.0, 100.0, 0.1, 0.1), (98.0, 100.05, 0.1, 0.1)
    )
    separating_lines = candles_after_a_calm_market(
        (102.0, 100.0, 0.1, 0.1), (102.0, 104.0, 0.1, 0.05)
    )

    assert counterattack["Counterattack"].iloc[-1] == 1
    assert counterattack["Separating Lines"].iloc[-1] == 0
    assert separating_lines["Separating Lines"].iloc[-1] == 1
    assert separating_lines["Counterattack"].iloc[-1] == 0


def test_three_inside_and_three_outside():
    three_inside_down = candles_after_a_calm_market(
        (100.0, 103.0, 0.1, 0.1), (102.0, 101.5, 0.1, 0.1), (101.4, 99.5, 0.1, 0.1)
    )
    three_outside_up = candles_after_a_calm_market(
        (101.0, 100.0, 0.1, 0.1), (99.8, 101.5, 0.1, 0.1), (101.5, 102.0, 0.1, 0.1)
    )

    assert three_inside_down["Harami"].iloc[-2] == -1
    assert three_inside_down["Three Inside"].iloc[-1] == -1
    assert three_outside_up["Engulfing"].iloc[-2] == 1
    assert three_outside_up["Three Outside"].iloc[-1] == 1


def test_the_doji_patterns_of_two_and_three_candles():
    morning_doji_star = candles_after_a_calm_market(
        (103.0, 100.0, 0.1, 0.1),  # a long black body
        (99.5, 99.52, 0.1, 0.1),  # a doji gapping down
        (99.8, 102.0, 0.1, 0.1),  # a white body closing 30% into the first
    )
    harami_cross = candles_after_a_calm_market(
        (100.0, 103.0, 0.1, 0.1), (101.5, 101.52, 0.1, 0.1)
    )

    # The doji after the black body is a bullish doji star.
    assert morning_doji_star["Doji Star"].iloc[-2] == 1
    assert morning_doji_star["Morning Doji Star"].iloc[-1] == 1
    assert morning_doji_star["Evening Doji Star"].iloc[-1] == 0
    assert harami_cross["Harami Cross"].iloc[-1] == -1


def test_two_and_three_candle_patterns_match_ta_lib_on_the_test_data():
    data = pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )
    candles = (
        data[["Open", "High", "Low", "Close"]].xs("AAPL", axis=1, level=1).dropna()
    )
    patterns = pattern_model.get_candlestick_patterns(
        candles["Open"], candles["High"], candles["Low"], candles["Close"]
    )

    # TA-Lib 0.8.3's CDL functions on these candles: (+100 count, -100 count).
    ta_lib_counts = {
        "Counterattack": (1, 0),
        "Doji Star": (10, 13),
        "Harami Cross": (5, 15),
        "Homing Pigeon": (5, 0),
        "In-Neck": (0, 1),
        "On-Neck": (0, 7),
        "Thrusting": (0, 11),
        "Matching Low": (11, 0),
        "Separating Lines": (2, 1),
        "Three Inside": (5, 9),
        "Three Outside": (12, 19),
        "Morning Doji Star": (2, 0),
        "Evening Doji Star": (0, 1),
    }
    for name, (bullish, bearish) in ta_lib_counts.items():
        assert (patterns[name].eq(1).sum(), patterns[name].eq(-1).sum()) == (
            bullish,
            bearish,
        ), name


def test_the_bearish_crows():
    two_crows = candles_after_a_calm_market(
        (100.0, 103.0, 0.1, 0.1), (104.0, 103.5, 0.1, 0.1), (103.6, 101.5, 0.1, 0.1)
    )
    # A short second body and a third body engulfing it that closes above the first close.
    upside_gap_two_crows = candles_after_a_calm_market(
        (100.0, 103.0, 0.1, 0.1), (104.0, 103.6, 0.1, 0.1), (104.2, 103.2, 0.1, 0.1)
    )
    identical_three_crows = candles_after_a_calm_market(
        (104.0, 103.0, 0.1, 0.01), (103.0, 102.0, 0.1, 0.01), (102.0, 101.0, 0.1, 0.01)
    )

    names = ["Two Crows", "Upside Gap Two Crows"]
    assert two_crows[names].iloc[-1].tolist() == [-1, 0]
    assert upside_gap_two_crows[names].iloc[-1].tolist() == [0, -1]
    assert identical_three_crows["Identical Three Crows"].iloc[-1] == -1


def test_an_abandoned_baby_and_a_tri_star():
    abandoned_baby = candles_after_a_calm_market(
        (103.0, 100.0, 0.1, 0.1),  # a long black body with its low at 99.9
        (99.5, 99.52, 0.1, 0.1),  # a doji whose high (99.62) gaps below that low
        (
            99.8,
            102.0,
            0.1,
            0.05,
        ),  # a white body gapping above the doji, 60% into the first
    )
    tri_star = candles_after_a_calm_market(
        (101.0, 101.01, 0.5, 0.5), (102.0, 102.01, 0.5, 0.5), (101.5, 101.51, 0.5, 0.5)
    )

    assert abandoned_baby["Abandoned Baby"].iloc[-1] == 1
    assert abandoned_baby["Morning Doji Star"].iloc[-1] == 1
    # The middle doji gaps up and the third doesn't rise above it.
    assert tri_star["Tri-Star"].iloc[-1] == -1


def test_the_bullish_three_candle_bottoms():
    stick_sandwich = candles_after_a_calm_market(
        (102.0, 100.0, 0.1, 0.1), (100.5, 101.5, 0.1, 0.1), (101.6, 100.05, 0.1, 0.1)
    )
    unique_three_river = candles_after_a_calm_market(
        (103.0, 100.0, 0.1, 0.1), (101.5, 101.0, 0.1, 1.5), (99.8, 100.2, 0.1, 0.1)
    )

    assert stick_sandwich["Stick Sandwich"].iloc[-1] == 1
    assert unique_three_river["Unique Three River"].iloc[-1] == 1


def test_the_gap_continuation_patterns():
    side_by_side = candles_after_a_calm_market(
        (100.0, 101.0, 0.1, 0.1), (101.5, 102.5, 0.1, 0.1), (101.5, 102.5, 0.1, 0.1)
    )
    gap_three_methods = candles_after_a_calm_market(
        (100.0, 101.0, 0.1, 0.1), (101.5, 102.5, 0.1, 0.1), (102.0, 100.5, 0.1, 0.1)
    )
    tasuki_gap = candles_after_a_calm_market(
        (100.0, 101.0, 0.1, 0.1), (101.5, 102.5, 0.1, 0.1), (102.0, 101.2, 0.1, 0.1)
    )

    names = ["Side-by-Side White Lines", "Gap Three Methods", "Tasuki Gap"]
    assert side_by_side[names].iloc[-1].tolist() == [1, 0, 0]
    assert gap_three_methods[names].iloc[-1].tolist() == [0, 1, 0]
    assert tasuki_gap[names].iloc[-1].tolist() == [0, 0, 1]


def test_three_candle_patterns_match_ta_lib_on_the_test_data():
    data = pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )

    # TA-Lib 0.8.3's CDL functions on these candles: (+100 count, -100 count) for AAPL, MSFT
    # and the Benchmark.
    ta_lib_counts = {
        "Two Crows": [(0, 0), (0, 1), (0, 0)],
        "Advance Block": [(0, 0), (0, 5), (0, 1)],
        "Side-by-Side White Lines": [(0, 0), (3, 0), (3, 0)],
        "Stalled Pattern": [(0, 2), (0, 2), (0, 4)],
        "Stick Sandwich": [(0, 0), (3, 0), (2, 0)],
        "Tasuki Gap": [(3, 1), (2, 2), (7, 1)],
        "Unique Three River": [(0, 0), (3, 0), (1, 0)],
        "Upside Gap Two Crows": [(0, 1), (0, 1), (0, 1)],
        "Gap Three Methods": [(3, 2), (2, 5), (0, 2)],
    }
    for position, ticker in enumerate(["AAPL", "MSFT", "Benchmark"]):
        candles = (
            data[["Open", "High", "Low", "Close"]].xs(ticker, axis=1, level=1).dropna()
        )
        patterns = pattern_model.get_candlestick_patterns(
            candles["Open"], candles["High"], candles["Low"], candles["Close"]
        )
        for name, counts in ta_lib_counts.items():
            assert (
                patterns[name].eq(1).sum(),
                patterns[name].eq(-1).sum(),
            ) == counts[
                position
            ], (ticker, name)


def test_four_and_five_candle_patterns():
    three_line_strike = candles_after_a_calm_market(
        (100.0, 101.5, 0.1, 0.1),
        (101.3, 102.8, 0.1, 0.1),
        (102.6, 104.0, 0.1, 0.1),
        (
            104.3,
            99.8,
            0.1,
            0.1,
        ),  # opens above the third close, closes below the first open
    )
    concealing_baby_swallow = candles_after_a_calm_market(
        (103.0, 101.5, 0.05, 0.05),
        (101.5, 100.0, 0.05, 0.05),
        (99.5, 99.0, 0.8, 0.1),  # gaps down with an upper shadow up to 100.3
        (100.5, 98.5, 0.2, 0.2),
    )
    breakaway = candles_after_a_calm_market(
        (103.0, 100.0, 0.1, 0.1),
        (99.5, 99.0, 0.1, 0.1),
        (98.9, 98.6, 0.1, 0.1),
        (98.5, 98.1, 0.1, 0.1),
        (98.2, 99.8, 0.1, 0.1),  # closes inside the gap between 99.5 and 100
    )
    ladder_bottom = candles_after_a_calm_market(
        (103.0, 102.0, 0.1, 0.1),
        (102.5, 101.5, 0.1, 0.1),
        (102.0, 101.0, 0.1, 0.1),
        (101.5, 100.5, 0.6, 0.1),
        (101.6, 102.5, 0.1, 0.1),
    )
    mat_hold = candles_after_a_calm_market(
        (100.0, 103.0, 0.05, 0.05),
        (103.8, 103.5, 0.05, 0.05),
        (103.2, 102.9, 0.05, 0.05),
        (103.05, 102.7, 0.05, 0.05),
        (103.0, 104.8, 0.05, 0.05),
    )

    assert three_line_strike["Three-Line Strike"].iloc[-1] == 1
    assert concealing_baby_swallow["Concealing Baby Swallow"].iloc[-1] == 1
    assert breakaway["Breakaway"].iloc[-1] == 1
    assert ladder_bottom["Ladder Bottom"].iloc[-1] == 1
    assert mat_hold["Mat Hold"].iloc[-1] == 1


def test_rising_and_falling_three_methods():
    rising = candles_after_a_calm_market(
        (100.0, 103.0, 0.2, 0.2),
        (102.8, 102.5, 0.1, 0.1),
        (102.6, 102.2, 0.1, 0.1),
        (102.3, 101.9, 0.1, 0.1),
        (102.0, 104.0, 0.1, 0.1),
    )
    falling = candles_after_a_calm_market(
        (103.0, 100.0, 0.2, 0.2),
        (100.2, 100.5, 0.1, 0.1),
        (100.4, 100.8, 0.1, 0.1),
        (100.7, 101.1, 0.1, 0.1),
        (101.0, 99.0, 0.1, 0.1),
    )

    assert rising["Rising/Falling Three Methods"].iloc[-1] == 1
    assert falling["Rising/Falling Three Methods"].iloc[-1] == -1


def test_a_hikkake_and_its_confirmation():
    hikkake = candles_after_a_calm_market(
        (100.0, 101.0, 1.0, 1.0),
        (100.3, 100.7, 0.1, 0.1),  # an inside bar from 100.2 to 100.8
        (100.2, 99.8, 0.1, 0.2),  # a false breakdown: a lower high and a lower low
        (99.9, 101.2, 0.1, 0.1),  # a close above the inside bar's high
    )
    modified_hikkake = candles_after_a_calm_market(
        (100.0, 101.0, 1.0, 1.0),
        (100.2, 99.6, 0.3, 0.05),  # an inside bar closing near its low
        (99.8, 100.0, 0.1, 0.1),  # a second inside bar from 99.7 to 100.1
        (99.7, 99.5, 0.2, 0.2),
        (99.6, 100.5, 0.1, 0.1),
    )

    assert hikkake["Hikkake"].iloc[-3:].tolist() == [0, 1, 2]
    assert modified_hikkake["Modified Hikkake"].iloc[-3:].tolist() == [0, 1, 2]
    assert modified_hikkake["Hikkake"].iloc[-3:].tolist() == [0, 1, 2]


def test_a_hikkake_is_confirmed_once_and_only_within_three_candles():
    pattern = [
        (100.0, 101.0, 1.0, 1.0),
        (100.3, 100.7, 0.1, 0.1),
        (100.2, 99.8, 0.1, 0.2),
    ]
    quiet = [(99.9, 100.0, 0.05, 0.05)] * 3

    late = candles_after_a_calm_market(*pattern, *quiet, (100.0, 101.2, 0.1, 0.1))
    twice = candles_after_a_calm_market(
        *pattern, (99.9, 101.2, 0.1, 0.1), (101.2, 101.5, 0.1, 0.1)
    )

    assert late["Hikkake"].iloc[-5:].tolist() == [1, 0, 0, 0, 0]
    assert twice["Hikkake"].iloc[-3:].tolist() == [1, 2, 0]


def test_four_and_five_candle_patterns_match_ta_lib_on_the_test_data():
    data = pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )

    # TA-Lib 0.8.3's CDL functions on these candles: the counts of +100, -100, +200 and -200 for
    # AAPL, MSFT and the Benchmark (200 is a confirmed hikkake, 2 here).
    ta_lib_counts = {
        "Three-Line Strike": [(0, 0, 0, 0), (0, 0, 0, 0), (0, 1, 0, 0)],
        "Ladder Bottom": [(2, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
        "Rising/Falling Three Methods": [(1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
        "Hikkake": [(20, 37, 6, 15), (22, 35, 4, 11), (24, 26, 8, 4)],
    }
    for position, ticker in enumerate(["AAPL", "MSFT", "Benchmark"]):
        candles = (
            data[["Open", "High", "Low", "Close"]].xs(ticker, axis=1, level=1).dropna()
        )
        patterns = pattern_model.get_candlestick_patterns(
            candles["Open"], candles["High"], candles["Low"], candles["Close"]
        )
        for name, counts in ta_lib_counts.items():
            assert (
                tuple(int(patterns[name].eq(value).sum()) for value in (1, -1, 2, -2))
                == counts[position]
            ), (ticker, name)


def test_td_sequential_counts_a_setup_and_a_countdown():
    # Ten rising closes and then a steady fall of 1.5 a day, with a high and low 0.5 around each close.
    close = pd.Series(np.r_[np.arange(100.0, 110.0), np.arange(107.5, 55.0, -1.5)])
    td = pattern_model.get_td_sequential(close + 0.5, close - 0.5, close)

    buy_setup = td["Buy Setup"].to_numpy()
    first = int(np.flatnonzero(buy_setup == 1)[0])
    # The 12th close is the first below the close four bars earlier, right after one above it: the flip.
    assert first == 11
    assert buy_setup[first : first + 9].tolist() == list(range(1, 10))
    assert (buy_setup[first + 9 :] == 0).all()
    # Every close from the ninth setup bar on is below the low two bars earlier.
    countdown = td["Buy Countdown"].to_numpy()
    assert countdown[first + 8 : first + 21].tolist() == list(range(1, 14))
    assert (countdown[first + 21 :] == 0).all()
    assert td["Sell Countdown"].sum() == 0
    assert td.iloc[:4].isna().all().all()


def test_a_close_above_the_setup_cancels_the_buy_countdown():
    # The setup completes on the 20th bar and the countdown counts four bars, then a close jumps above the
    # highest true high of the setup (107.5) and the fall resumes.
    close = pd.Series(
        np.r_[
            np.arange(100.0, 110.0),
            np.arange(107.5, 89.0, -1.5),
            110.0,
            88.0,
            86.5,
            85.0,
        ]
    )
    td = pattern_model.get_td_sequential(close + 0.5, close - 0.5, close)

    countdown = td["Buy Countdown"].to_numpy()
    assert np.nanmax(countdown) == 4
    assert (countdown[np.flatnonzero(close.to_numpy() == 110.0)[0] :] == 0).all()


def test_td_sequential_matches_pandas_ta_classic_on_the_test_data():
    data = pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )

    # pandas-ta-classic 0.9.0's td_seq on these candles: the bars counted 9 for a sell (up) and a
    # buy (down) setup per ticker. AAPL has one run of nine without a price flip before it.
    pandas_ta_nines = {"AAPL": (14, 8), "MSFT": (14, 7), "Benchmark": (18, 7)}
    flip_nines = {"AAPL": (13, 8), "MSFT": (14, 7), "Benchmark": (18, 7)}
    thirteens = {"AAPL": (2, 4), "MSFT": (3, 4), "Benchmark": (2, 10)}
    for ticker, nines in pandas_ta_nines.items():
        candles = data[["High", "Low", "Close"]].xs(ticker, axis=1, level=1).dropna()
        td = pattern_model.get_td_sequential(
            candles["High"], candles["Low"], candles["Close"]
        )
        ours = (int(td["Sell Setup"].eq(9).sum()), int(td["Buy Setup"].eq(9).sum()))
        assert ours == flip_nines[ticker]
        assert ours[0] <= nines[0] and ours[1] <= nines[1]
        assert (
            int(td["Buy Countdown"].eq(13).sum()),
            int(td["Sell Countdown"].eq(13).sum()),
        ) == thirteens[ticker]


def test_liquidity_sweep():
    calm = [(100.0, 101.5, 99.5, 101.0)] * 15
    # A white candle wicking below the lows to 98 and closing back above them, a long lower shadow.
    frame = pd.DataFrame(
        calm + [(99.8, 100.0, 98.0, 99.9), (101.0, 102.5, 100.8, 100.9)],
        columns=["Open", "High", "Low", "Close"],
    )
    sweeps = pattern_model.get_liquidity_sweep(
        frame["Open"], frame["High"], frame["Low"], frame["Close"]
    )

    assert sweeps.iloc[:15].isna().all()
    assert sweeps.iloc[-2:].tolist() == [1, -1]


def test_liquidity_sweep_matches_pandas_ta_classic_on_the_test_data():
    data = pd.read_pickle(
        Path(__file__).resolve().parents[1] / "datasets" / "historical_dataset.pickle"
    )

    # pandas-ta-classic 0.9.0's smc_sweep (length 15, wick_mult 1.5): bullish and bearish sweeps.
    pandas_ta_counts = {"AAPL": (11, 11), "MSFT": (7, 10), "Benchmark": (5, 8)}
    for ticker, counts in pandas_ta_counts.items():
        candles = (
            data[["Open", "High", "Low", "Close"]].xs(ticker, axis=1, level=1).dropna()
        )
        sweeps = pattern_model.get_liquidity_sweep(
            candles["Open"], candles["High"], candles["Low"], candles["Close"]
        )
        assert (int(sweeps.eq(1).sum()), int(sweeps.eq(-1).sum())) == counts


def test_the_signal_names_are_the_columns_get_signals_gives():
    prices = pd.Series(
        np.linspace(100, 120, 300),
        index=pd.period_range("2024-01-01", periods=300, freq="D"),
    )

    signals = pattern_model.get_signals(
        prices, 50, 200, 12, 26, 9, 14, 70, 30, 20, 2, 252
    )

    assert list(signals.columns) == pattern_model.SIGNAL_NAMES
