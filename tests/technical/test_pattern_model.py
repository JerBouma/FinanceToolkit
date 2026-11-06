"""Candlestick Pattern and Signal Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals import pattern_model

# pylint: disable=missing-function-docstring


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
    assert (patterns.iloc[13:] == 0).all().all()


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
