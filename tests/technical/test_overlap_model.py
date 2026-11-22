"""Overlap Model Tests"""

import pandas as pd
import pytest

from financetoolkit.technicals.overlap import overlap_model
from tests.technical.sample_prices import sample_prices


def test_get_moving_average(recorder):
    prices = sample_prices()
    result = overlap_model.get_moving_average(prices["Close"], 10).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_exponential_moving_average(recorder):
    prices = sample_prices()
    result = overlap_model.get_exponential_moving_average(prices["Close"], 10).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_double_exponential_moving_average(recorder):
    prices = sample_prices()
    result = overlap_model.get_double_exponential_moving_average(
        prices["Close"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_trix(recorder):
    prices = sample_prices()
    result = overlap_model.get_trix(prices["Close"], 10).round(6)

    # The windows fill within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_triangular_moving_average(recorder):
    prices = sample_prices()
    result = overlap_model.get_triangular_moving_average(prices["Close"], 20).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


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


def test_kaufman_adaptive_moving_average_follows_flat_prices():
    # On flat prices the efficiency ratio is 1 (TA-Lib), so KAMA keeps tracking the price
    # instead of turning NaN on 0 / 0.
    prices = pd.Series([10.0] * 15 + [11.0, 12.0, 13.0], dtype=float)
    kama = overlap_model.get_kaufman_adaptive_moving_average(prices, window=5)

    assert kama.iloc[:5].isna().all()
    assert kama.iloc[5:15].eq(10.0).all()
    assert kama.iloc[15:].notna().all()
    assert kama.iloc[-1] > 10.0


def test_kaufman_adaptive_moving_average_is_nan_only_for_missing_prices():
    prices = pd.Series([10.0, 10.5, None, 11.0, 11.5, 12.0, 12.5, 13.0], dtype=float)
    kama = overlap_model.get_kaufman_adaptive_moving_average(prices, window=2)

    assert kama.iloc[:5].isna().all()
    assert kama.iloc[5:].notna().all()


def test_parabolic_sar_starts_at_each_ticker_first_bar():
    index = pd.date_range("2024-01-01", periods=30)
    high = pd.Series(
        [10 + i * 0.5 + (i % 3) for i in range(30)], index=index, dtype="float64"
    )
    low = high - 1.5

    full = overlap_model.get_parabolic_sar(high, low)

    later_high, later_low = high.copy(), low.copy()
    later_high.iloc[:5] = float("nan")
    later_low.iloc[:5] = float("nan")
    later = overlap_model.get_parabolic_sar(later_high, later_low)

    assert later.iloc[:5].isna().all()
    pd.testing.assert_series_equal(
        later.iloc[5:], overlap_model.get_parabolic_sar(high.iloc[5:], low.iloc[5:])
    )
    # The first bar only sets the direction, so the SAR starts on the second bar.
    assert pd.isna(full.iloc[0])
    assert full.iloc[1:].notna().all()

    gap_high, gap_low = high.copy(), low.copy()
    gap_high.iloc[12] = float("nan")
    assert pd.isna(overlap_model.get_parabolic_sar(gap_high, gap_low).iloc[12])


def test_support_resistance_follow_each_ticker_own_data():
    index = pd.date_range("2024-01-01", periods=80)
    prices = pd.Series(
        [
            100 + 10 * ((i % 20) - 10 if (i // 20) % 2 else 10 - (i % 20))
            for i in range(80)
        ],
        index=index,
        dtype="float64",
    )

    listed_later = prices.copy()
    listed_later.iloc[:15] = float("nan")

    levels = overlap_model.get_support_resistance_levels(listed_later, window=5)
    truncated = overlap_model.get_support_resistance_levels(prices.iloc[15:], window=5)

    assert levels.iloc[:15].isna().all().all()
    pd.testing.assert_frame_equal(levels.iloc[15:], truncated)


def test_parabolic_sar_matches_ta_lib_from_a_short_start():
    """TA-Lib's ta_SAR.c (SAR(0.02, 0.2)) on bars whose second low falls further than
    the high rises, so the trade starts short at the first high and reverses to a long.
    """
    high = pd.Series(
        [96.5944, 95.4067, 96.7179, 97.5011, 96.4311, 98.0733, 96.3935, 96.4349]
        + [95.0034, 93.9741, 94.7635, 92.3434]
    )
    low = pd.Series(
        [95.2593, 94.4887, 96.1772, 95.6457, 95.1031, 95.7938, 94.3958, 94.9741]
        + [94.6838, 92.6554, 93.7713, 91.8332]
    )
    ta_lib = [
        float("nan"),
        96.5944,
        94.4887,
        94.4887,
        94.6092,
        94.7249,
        98.0733,
        98.0733,
        97.9998,
        97.9277,
        97.7168,
        97.5144,
    ]

    result = overlap_model.get_parabolic_sar(high, low)

    # The prices are rounded to four decimals, which moves the last decimal at most.
    assert result.tolist()[1:] == pytest.approx(ta_lib[1:], abs=2e-4)
    assert pd.isna(result.iloc[0])


def test_parabolic_sar_takes_separate_start_and_increment():
    """TA-Lib's SAREXT with an initial factor of 0.01, steps of 0.03 and a maximum of 0.25
    is the same recursion with a different start and increment of the factor."""
    high = pd.Series([10.0, 10.5, 11.0, 11.6, 12.0, 11.8, 12.5])
    low = pd.Series([9.5, 9.9, 10.4, 11.0, 11.3, 11.1, 11.9])

    result = overlap_model.get_parabolic_sar(high, low, 0.01, 0.03, 0.25)

    # Long from the start (the second low is higher): the SAR starts at the first low and
    # the extreme point at the second high. Each bar shows the SAR set the bar before, then
    # a new high raises the factor and the next SAR stays below this and the prior low.
    sar, af, extreme_point = 9.5, 0.01, 10.5
    expected = []
    for i in range(1, len(high)):
        expected.append(sar)
        if high.iloc[i] > extreme_point:
            extreme_point, af = high.iloc[i], min(af + 0.03, 0.25)
        sar = min(
            sar + af * (extreme_point - sar), low.iloc[max(i - 1, 1)], low.iloc[i]
        )

    assert result.iloc[1:].tolist() == pytest.approx(expected)


def test_exponential_and_adaptive_averages_stop_where_the_prices_stop():
    """The EMA (and with it the DEMA, TRIX and MACD built on it) and KAMA are NaN on the
    dates without a price instead of repeating their last value, and the recursion
    continues from the last average on the next price."""
    import numpy as np  # noqa: PLC0415

    close = sample_prices()["Close"].copy()
    close.iloc[-5:] = np.nan
    close.iloc[40] = np.nan

    ema = overlap_model.get_exponential_moving_average(close, 10)
    for result in (
        ema,
        overlap_model.get_double_exponential_moving_average(close, 5),
        overlap_model.get_trix(close, 5),
        overlap_model.get_kaufman_adaptive_moving_average(close, 5),
    ):
        assert result.iloc[-5:].isna().all()
        assert pd.isna(result.iloc[40])

    alpha = 2 / 11
    assert ema.iloc[41] == pytest.approx(
        alpha * close.iloc[41] + (1 - alpha) * ema.iloc[39]
    )


def _sample_close():
    return sample_prices()["Close"].reset_index(drop=True)


@pytest.mark.parametrize(
    ("function", "window", "expected"),
    [
        # TA-Lib's TEMA(5), ZLEMA(10), ALMA(9, 6, 0.85) and RMA(10) on the sample closes.
        (
            overlap_model.get_triple_exponential_moving_average,
            5,
            {12: 100.75899484, 30: 106.79819427, 59: 120.77284896},
        ),
        (
            overlap_model.get_zero_lag_exponential_moving_average,
            10,
            {13: 102.81230492, 30: 108.04664238, 59: 119.3767005},
        ),
        (
            overlap_model.get_arnaud_legoux_moving_average,
            9,
            {8: 105.10209182, 30: 108.84582542, 59: 117.43482412},
        ),
        # pandas-ta-classic's sinwma(10), fwma(10), pwma(10) and swma(4), the last the
        # triangular moving average of four periods (weights 1, 2, 2, 1).
        (
            overlap_model.get_sine_weighted_moving_average,
            10,
            {9: 104.2356613, 30: 109.95994981, 59: 114.52446796},
        ),
        (
            overlap_model.get_fibonacci_weighted_moving_average,
            10,
            {9: 104.11680672, 30: 108.13368517, 59: 118.42361213},
        ),
        (
            overlap_model.get_pascal_weighted_moving_average,
            10,
            {9: 104.86909262, 30: 110.46700675, 59: 114.12374555},
        ),
        (
            overlap_model.get_triangular_moving_average,
            4,
            {3: 102.27173332, 30: 108.26443887, 59: 118.43487169},
        ),
    ],
)
def test_moving_averages_match_reference_libraries(function, window, expected):
    result = function(_sample_close(), window)

    first = min(expected)
    assert result.iloc[:first].isna().all()
    assert result.iloc[list(expected)].tolist() == pytest.approx(
        list(expected.values())
    )


def test_t3_moving_average_matches_talib():
    result = overlap_model.get_t3_moving_average(_sample_close(), 5, 0.7)

    # Six EMAs of five periods, each seeded on its first five values: 6 * (5 - 1).
    assert result.iloc[:24].isna().all()
    assert result.iloc[[24, 40, 59]].tolist() == pytest.approx(
        [109.35472636, 110.82298153, 116.70945961]
    )


def test_volume_weighted_moving_average_matches_talib():
    prices = sample_prices().reset_index(drop=True)
    result = overlap_model.get_volume_weighted_moving_average(
        prices["Close"], prices["Volume"].astype(float), 10
    )

    assert result.iloc[:9].isna().all()
    assert result.iloc[[9, 30, 59]].tolist() == pytest.approx(
        [103.92016157, 109.66672188, 114.74797246]
    )

    # A window without any volume has no average instead of 0 / 0.
    no_volume = overlap_model.get_volume_weighted_moving_average(
        prices["Close"], prices["Volume"].astype(float) * 0, 10
    )
    assert no_volume.isna().all()


def test_mcginley_dynamic_is_seeded_with_simple_moving_average():
    close = _sample_close()
    result = overlap_model.get_mcginley_dynamic(close, 10)

    seed = close.iloc[:10].mean()
    step = seed + (close.iloc[10] - seed) / (10 * (close.iloc[10] / seed) ** 4)

    assert result.iloc[:9].isna().all()
    assert result.iloc[9] == pytest.approx(seed)
    assert result.iloc[10] == pytest.approx(step)
    # TA-Lib's MCGD(10) starts from the first close; 50 periods later the seeds have
    # decayed to within 0.02 of each other.
    assert result.iloc[59] == pytest.approx(114.04219027, abs=0.02)


def test_variable_index_dynamic_average_matches_pandas_ta_with_equal_windows():
    # pandas-ta-classic's vidya(10) seeds with the simple average of ten closes, as here.
    result = overlap_model.get_variable_index_dynamic_average(_sample_close(), 10, 10)

    assert result.iloc[:9].isna().all()
    assert result.iloc[[9, 10, 30, 59]].tolist() == pytest.approx(
        [103.74406291, 103.67382461, 108.31405927, 115.07059807]
    )


def test_variable_index_dynamic_average_starts_after_both_windows():
    close = _sample_close()
    result = overlap_model.get_variable_index_dynamic_average(close, 12, 9)

    # The seed is the simple average of twelve closes on period 11 (TA-Lib starts from
    # the first close and its VIDYA(12, 9) is 114.97523489 on period 59).
    assert result.iloc[:11].isna().all()
    assert result.iloc[11] == pytest.approx(close.iloc[:12].mean())
    assert result.iloc[59] == pytest.approx(114.97523489, abs=0.01)

    # Flat prices give an oscillator of 0 / 0, which leaves the average where it is.
    flat = overlap_model.get_variable_index_dynamic_average(
        pd.Series([10.0] * 30), 5, 5
    )
    assert flat.iloc[4:].eq(10.0).all()


def test_fractal_adaptive_moving_average_matches_talib_after_the_seed():
    prices = sample_prices().reset_index(drop=True)
    median = (prices["High"] + prices["Low"]) / 2
    result = overlap_model.get_fractal_adaptive_moving_average(
        prices["High"], prices["Low"], 10
    )

    assert result.iloc[:9].isna().all()
    assert result.iloc[9] == pytest.approx(median.iloc[:10].mean())
    # TA-Lib's FRAMA(10) seeds with the median price of period 9; the seeds have decayed
    # by period 59.
    assert result.iloc[59] == pytest.approx(118.40431387)

    with pytest.raises(ValueError, match="even"):
        overlap_model.get_fractal_adaptive_moving_average(
            prices["High"], prices["Low"], 9
        )


def test_new_moving_averages_stop_where_the_prices_stop():
    close = _sample_close().copy()
    close.iloc[-3:] = float("nan")
    close.iloc[40] = float("nan")

    for result in (
        overlap_model.get_triple_exponential_moving_average(close, 5),
        overlap_model.get_zero_lag_exponential_moving_average(close, 5),
        overlap_model.get_mcginley_dynamic(close, 5),
        overlap_model.get_variable_index_dynamic_average(close, 5, 5),
        overlap_model.get_sine_weighted_moving_average(close, 5),
    ):
        assert result.iloc[-3:].isna().all()
        assert pd.isna(result.iloc[40])
