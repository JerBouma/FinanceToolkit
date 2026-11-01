"""Breadth Model Tests"""

import pandas as pd
import pytest

from financetoolkit.technicals import breadth_model

# pylint: disable=missing-function-docstring


def test_get_mcclellan_oscillator(recorder):
    recorder.capture(
        breadth_model.get_mcclellan_oscillator(
            pd.Series([100, 110, 120, 130, 80]), 10, 20
        ).round(2)
    )


def test_get_advancers_decliners(recorder):
    recorder.capture(
        breadth_model.get_advancers_decliners(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02])
        )
    )


def test_get_on_balance_volume(recorder):
    recorder.capture(
        breadth_model.get_on_balance_volume(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
        )
    )


def test_get_accumulation_distribution_line(recorder):
    recorder.capture(
        breadth_model.get_accumulation_distribution_line(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.1, 103, 0.05, 0.01, -0.02]),
        )
    )


def test_get_chaikin_oscillator(recorder):
    recorder.capture(
        breadth_model.get_chaikin_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            10,
            20,
        )
    )


def test_get_chaikin_money_flow(recorder):
    recorder.capture(
        breadth_model.get_chaikin_money_flow(
            pd.Series([10, 11, 12, 11, 13, 14, 13, 15]),
            pd.Series([9, 10, 10, 9, 11, 12, 11, 13]),
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
            4,
        ).round(3)
    )


def test_get_ease_of_movement(recorder):
    recorder.capture(
        breadth_model.get_ease_of_movement(
            pd.Series([10, 11, 12, 11, 13, 14, 13, 15]),
            pd.Series([9, 10, 10, 9, 11, 12, 11, 13]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
            3,
        ).round(3)
    )


def test_get_negative_volume_index(recorder):
    recorder.capture(
        breadth_model.get_negative_volume_index(
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
        ).round(3)
    )


def test_get_positive_volume_index(recorder):
    recorder.capture(
        breadth_model.get_positive_volume_index(
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
        ).round(3)
    )


def test_mcclellan_oscillator_seeds_on_actual_advances_and_declines():
    prices = pd.Series([10.0, 11, 10, 12, 13, 12, 14, 15, 14, 16])

    oscillator = breadth_model.get_mcclellan_oscillator(prices, 2, 4)

    # The first signal is on the second price, so the 4-period EMA is seeded on periods
    # 1 to 4 and the 2-period EMA on periods 1 to 2 (then updated up to period 4).
    signals = [1, -1, 1, 1]
    short = sum(signals[:2]) / 2
    for signal in signals[2:]:
        short = short + 2 / 3 * (signal - short)

    assert oscillator.first_valid_index() == 4
    assert oscillator.iloc[4] == pytest.approx(short - sum(signals) / 4)


def test_volume_indicators_are_nan_before_listing_and_on_missing_days():
    index = pd.date_range("2024-01-01", periods=6)
    close = pd.Series([float("nan"), 10.0, 11.0, float("nan"), 12.0, 11.0], index=index)
    volume = pd.Series(
        [float("nan"), 100.0, 200.0, float("nan"), 150.0, 120.0], index=index
    )

    for result in [
        breadth_model.get_on_balance_volume(close, volume),
        breadth_model.get_negative_volume_index(close, volume),
        breadth_model.get_positive_volume_index(close, volume),
        breadth_model.get_advancers_decliners(close),
    ]:
        assert result.iloc[[0, 3]].isna().all()
        assert result.iloc[[1, 2, 4, 5]].notna().all()

    # The day after the gap compares with the last close before it (12 against 11).
    assert (
        breadth_model.get_on_balance_volume(close, volume).tolist()[4] == 0 + 200 + 150
    )
    assert breadth_model.get_advancers_decliners(close).iloc[4] == 1
    # The negative volume index moves only on lower volume: 150 < 200 on 12 against 11.
    assert breadth_model.get_negative_volume_index(close, volume).iloc[
        4
    ] == pytest.approx(1000 * 12 / 11)


def test_accumulation_distribution_line_adds_nothing_when_high_equals_low():
    index = pd.date_range("2024-01-01", periods=3)
    high = pd.Series([11.0, 10.0, 12.0], index=index)
    low = pd.Series([9.0, 10.0, 10.0], index=index)
    close = pd.Series([11.0, 10.0, 10.0], index=index)
    volume = pd.Series([100.0, 500.0, 50.0], index=index)

    adl = breadth_model.get_accumulation_distribution_line(high, low, close, volume)

    # As in TA-Lib: a bar without range has a multiplier of 0 and the line runs on.
    assert adl.tolist() == [100.0, 100.0, 50.0]


def test_new_highs_new_lows_use_full_windows_and_strict_comparisons():
    index = pd.date_range("2024-01-01", periods=5)
    closes = pd.DataFrame(
        {
            "A": [10.0, 11.0, 11.0, 12.0, 9.0],
            "B": [float("nan"), float("nan"), 5.0, 5.0, 6.0],
        },
        index=index,
    )

    result = breadth_model.get_new_highs_new_lows(closes, window=3)

    # No ticker has three closes until the third day; A's 11 equals its previous high, so it
    # is no new high.
    assert result.index.equals(index)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 0
    assert result.iloc[3] == 1
    # A makes a new low and B, with its first full window, makes a new high.
    assert result.iloc[4] == 0
