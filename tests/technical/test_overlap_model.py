"""Overlap Model Tests"""

import pandas as pd
import pytest

from financetoolkit.technicals import overlap_model

# pylint: disable=missing-function-docstring


def test_get_moving_average(recorder):
    recorder.capture(
        overlap_model.get_moving_average(pd.Series([100, 110, 120, 130, 80]), 10).round(
            3
        )
    )


def test_get_exponential_moving_average(recorder):
    recorder.capture(
        overlap_model.get_exponential_moving_average(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10
        ).round(2)
    )


def test_get_double_exponential_moving_average(recorder):
    recorder.capture(
        overlap_model.get_double_exponential_moving_average(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10
        ).round(2)
    )


def test_get_trix(recorder):
    recorder.capture(
        overlap_model.get_trix(pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10).round(3)
    )


def test_get_triangular_moving_average(recorder):
    recorder.capture(
        overlap_model.get_triangular_moving_average(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 20
        ).round(3)
    )


def test_get_fibonacci_retracement_levels(recorder):
    recorder.capture(
        overlap_model.get_fibonacci_retracement_levels(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([90, 95, 100, 110, 60]),
        ).round(3)
    )


def test_get_fibonacci_retracement_levels_downtrend(recorder):
    recorder.capture(
        overlap_model.get_fibonacci_retracement_levels(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([90, 95, 100, 110, 60]),
            trend="downtrend",
        ).round(3)
    )


def test_get_fibonacci_retracement_levels_custom_levels(recorder):
    recorder.capture(
        overlap_model.get_fibonacci_retracement_levels(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([90, 95, 100, 110, 60]),
            levels=[0.0, 0.5, 1.0],
        ).round(3)
    )


def test_get_kaufman_adaptive_moving_average(recorder):
    recorder.capture(
        overlap_model.get_kaufman_adaptive_moving_average(
            pd.Series(
                [100, 102, 101, 103, 106, 105, 107, 110, 108, 111, 115, 113, 116]
            ),
            5,
            2,
            30,
        ).round(3)
    )


def test_get_kaufman_adaptive_moving_average_default_parameters(recorder):
    recorder.capture(
        overlap_model.get_kaufman_adaptive_moving_average(
            pd.Series(
                [
                    100,
                    102,
                    101,
                    103,
                    106,
                    105,
                    107,
                    110,
                    108,
                    111,
                    115,
                    113,
                    116,
                    120,
                    118,
                ]
            )
        ).round(3)
    )


def _talib_exponential_moving_average(values, window):
    # TA-Lib's ta_EMA.c: the first output is the SMA of the first window values, on the
    # last of them, followed by the plain recursion.
    output = [float("nan")] * len(values)
    alpha = 2 / (window + 1)
    output[window - 1] = sum(values[:window]) / window
    for i in range(window, len(values)):
        output[i] = output[i - 1] + alpha * (values[i] - output[i - 1])
    return output


PRICES = [100, 102, 101, 103, 106, 105, 107, 110, 108, 111, 115, 113, 116, 120, 118]


def test_exponential_moving_average_is_seeded_with_simple_moving_average():
    ema = overlap_model.get_exponential_moving_average(
        pd.Series(PRICES, dtype=float), 5
    )
    expected = _talib_exponential_moving_average(PRICES, 5)

    assert ema.iloc[:4].isna().all()
    assert ema.iloc[4] == pytest.approx(sum(PRICES[:5]) / 5)
    assert ema.iloc[4:].tolist() == pytest.approx(expected[4:])


def test_exponential_moving_average_seeds_every_column_separately():
    late_start = [float("nan")] * 3 + PRICES[3:]
    prices = pd.DataFrame({"A": PRICES, "B": late_start}, dtype=float)

    ema = overlap_model.get_exponential_moving_average(prices, 5)

    assert ema["A"].first_valid_index() == 4
    assert ema["B"].first_valid_index() == 7
    assert ema["B"].iloc[7] == pytest.approx(sum(PRICES[3:8]) / 5)
    assert ema["B"].iloc[7:].tolist() == pytest.approx(
        _talib_exponential_moving_average(PRICES[3:], 5)[4:]
    )


def test_double_exponential_moving_average_starts_after_both_seeds():
    dema = overlap_model.get_double_exponential_moving_average(
        pd.Series(PRICES, dtype=float), 4
    )
    first = _talib_exponential_moving_average(PRICES, 4)
    second = [float("nan")] * 3 + _talib_exponential_moving_average(first[3:], 4)

    assert dema.first_valid_index() == 6
    assert dema.iloc[6:].tolist() == pytest.approx(
        [2 * a - b for a, b in zip(first[6:], second[6:])]
    )


def test_triangular_moving_average_has_no_partial_windows():
    for window in (4, 5):
        trima = overlap_model.get_triangular_moving_average(
            pd.Series(PRICES, dtype=float), window
        )
        half = (window + 1) // 2 if window % 2 else window // 2
        rising = list(range(1, half + 1))
        weights = rising + (rising[::-1] if window % 2 == 0 else rising[-2::-1])
        expected = sum(p * w for p, w in zip(PRICES[:window], weights)) / sum(weights)

        assert trima.iloc[: window - 1].isna().all()
        assert trima.iloc[window - 1] == pytest.approx(expected)


def test_kaufman_adaptive_moving_average_first_value_is_one_update_of_the_seed():
    window = 10
    kama = overlap_model.get_kaufman_adaptive_moving_average(
        pd.Series(PRICES, dtype=float), window
    )
    change = abs(PRICES[window] - PRICES[0])
    volatility = sum(abs(PRICES[i] - PRICES[i - 1]) for i in range(1, window + 1))
    constant = ((change / volatility) * (2 / 3 - 2 / 31) + 2 / 31) ** 2
    expected = PRICES[window - 1] + constant * (PRICES[window] - PRICES[window - 1])

    assert kama.iloc[:window].isna().all()
    assert kama.iloc[window] == pytest.approx(expected)
