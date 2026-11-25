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


def _sample_bars():
    prices = sample_prices().reset_index(drop=True)
    prices["Volume"] = prices["Volume"].astype(float)
    return prices


def test_price_transforms_and_midpoints_match_talib():
    prices = _sample_bars()
    bars = prices["Open"], prices["High"], prices["Low"], prices["Close"]

    # TA-Lib's AVGPRICE, MEDPRICE, TYPPRICE and WCLPRICE on periods 9, 30 and 59.
    for method, expected in [
        ("average", [103.3598375345, 106.9055796207, 120.1246590122]),
        ("median", [102.9728897996, 106.5282750095, 120.3107218486]),
        ("typical", [103.0700865438, 106.6268218585, 120.4127101254]),
        ("weighted_close", [103.1186849159, 106.676095283, 120.4637042639]),
    ]:
        result = overlap_model.get_price_transform(*bars, method=method)
        assert result.iloc[[9, 30, 59]].tolist() == pytest.approx(expected)

    with pytest.raises(ValueError, match="method"):
        overlap_model.get_price_transform(*bars, method="close")

    # TA-Lib's MIDPOINT(10) and MIDPRICE(10).
    midpoint = overlap_model.get_midpoint(prices["Close"], prices["Close"], 10)
    midpoint_price = overlap_model.get_midpoint(prices["High"], prices["Low"], 10)
    assert midpoint.iloc[:9].isna().all()
    assert midpoint.iloc[[9, 30, 59]].tolist() == pytest.approx(
        [102.7408159155, 108.9906742715, 116.0355483558]
    )
    assert midpoint_price.iloc[[9, 30, 59]].tolist() == pytest.approx(
        [102.5405300116, 109.0959191924, 115.8813672431]
    )


def test_holt_winters_moving_average_and_channel_match_pandas_ta():
    close = _sample_close()

    # pandas-ta-classic's hwma and hwc with their defaults.
    average = overlap_model.get_holt_winters_moving_average(close)
    channel = overlap_model.get_holt_winters_channel(close)

    assert average.iloc[[9, 30, 59]].tolist() == pytest.approx(
        [105.28968738, 110.96041966, 117.16266437]
    )
    assert channel.loc[[9, 30, 59], "Upper Band"].tolist() == pytest.approx(
        [106.52468191, 113.13412507, 119.60358688]
    )
    assert channel.loc[[9, 30, 59], "Lower Band"].tolist() == pytest.approx(
        [104.05469284, 108.78671426, 114.72174186]
    )
    pd.testing.assert_series_equal(channel["Middle Band"], average, check_names=False)
    # The average starts on the first price, so the first three bands have no deviation.
    assert channel.loc[[0, 1, 2], "Upper Band"].tolist() == average.iloc[:3].tolist()
    assert channel.loc[3, "Upper Band"] > average.iloc[3]

    # A missing price is missing, and the recursion continues on the next price.
    gapped = close.copy()
    gapped.iloc[40] = float("nan")
    result = overlap_model.get_holt_winters_moving_average(gapped)
    assert pd.isna(result.iloc[40])
    assert result.iloc[:40].tolist() == pytest.approx(average.iloc[:40].tolist())


def test_elastic_volume_weighted_moving_average_follows_fries():
    prices = _sample_bars()
    close, volume = prices["Close"], prices["Volume"]
    result = overlap_model.get_elastic_volume_weighted_moving_average(close, volume, 10)

    # The seed is the simple average of ten closes, after which every period replaces its
    # volume out of the window's volume at the previous average (finta's EVWMA recursion).
    volume_sum = volume.rolling(10).sum()
    expected = close.iloc[:10].mean()
    assert result.iloc[:9].isna().all()
    assert result.iloc[9] == pytest.approx(expected)
    for i in range(10, 60):
        expected = (
            (volume_sum.iloc[i] - volume.iloc[i]) * expected
            + volume.iloc[i] * close.iloc[i]
        ) / volume_sum.iloc[i]
    assert result.iloc[59] == pytest.approx(expected)


def test_volume_adjusted_moving_average_matches_finta():
    prices = _sample_bars()
    result = overlap_model.get_volume_adjusted_moving_average(
        prices["Close"], prices["Volume"], 8
    )

    # finta's VAMA(8) needs two windows: the average volume and the weighted sum.
    assert result.iloc[:14].isna().all()
    assert result.iloc[[30, 59]].tolist() == pytest.approx([109.70278375, 115.55700028])


def test_gann_hilo_activator_follows_the_side_of_the_trend():
    prices = _sample_bars()
    result = overlap_model.get_gann_hilo_activator(
        prices["High"], prices["Low"], prices["Close"], 3, 3
    )
    high_average = prices["High"].rolling(3).mean()
    low_average = prices["Low"].rolling(3).mean()
    up = result["Trend Direction"] == 1

    assert result["Trend Direction"].dropna().isin([1, -1]).all()
    assert result.loc[up, "Gann HiLo Activator"].tolist() == pytest.approx(
        low_average[up].tolist()
    )
    assert result.loc[
        ~up & result["Trend Direction"].notna(), "Gann HiLo Activator"
    ].tolist() == (
        pytest.approx(high_average[result["Trend Direction"] == -1].tolist())
    )

    # pandas-ta-classic's hilo(3, 3) on the periods with a close beyond an average, where
    # both switch to the average of the new side.
    breaks = (prices["Close"] > high_average.shift(1)) | (
        prices["Close"] < low_average.shift(1)
    )
    assert breaks.iloc[[9, 30, 47]].all()
    assert result["Gann HiLo Activator"].iloc[[9, 30, 47]].tolist() == pytest.approx(
        [104.86182200398115, 108.61403330847348, 115.9653273207414]
    )


def test_zigzag_matches_talib():
    prices = _sample_bars()
    result = overlap_model.get_zigzag(prices["High"], prices["Low"], 0.03, 2)

    # TA-Lib's ZIGZAG(3, 2): the extreme of the current leg, its trend and its position.
    assert result.iloc[:2].isna().all().all()
    assert result["Zig Zag"].iloc[[2, 9, 15, 26, 34, 59]].tolist() == pytest.approx(
        [104.34363036521302, 101.7866974667683, 98.98836463616024]
        + [112.57935996585978, 104.64671998699143, 121.44752707370303]
    )
    assert result["Trend Direction"].iloc[[2, 9, 16, 27, 35, 47, 54]].tolist() == [
        1,
        -1,
        1,
        -1,
        1,
        -1,
        1,
    ]
    # TA-Lib's pivot positions 5, 14, 24 and 43 on periods 8, 15, 26 and 46.
    assert result["Periods Since Pivot"].iloc[[8, 15, 26, 46]].tolist() == [3, 1, 2, 3]


def test_pivot_point_variants_match_pandas_ta():
    prices = _sample_bars()
    bars = prices["High"], prices["Low"], prices["Close"], prices["Open"]

    classic = overlap_model.get_pivot_points(*bars[:3])
    assert list(classic.columns) == [
        "Pivot Point",
        "Resistance 1",
        "Resistance 2",
        "Resistance 3",
        "Support 1",
        "Support 2",
        "Support 3",
    ]

    # pandas-ta-classic's cpr(method="camarilla") and cpr(method="woodie") on period 30.
    camarilla = overlap_model.get_pivot_points(
        *bars, method="camarilla", central_pivot_range=True
    ).iloc[30]
    assert camarilla[
        ["Top Central Pivot", "Pivot Point", "Bottom Central Pivot"]
    ].tolist() == pytest.approx(
        [107.63944015484265, 107.53702740220655, 107.43461464957045]
    )
    assert camarilla[["Resistance 4", "Support 4"]].tolist() == pytest.approx(
        [108.98460664444409, 106.4990991705134]
    )
    woodie = overlap_model.get_pivot_points(*bars, method="woodie").iloc[30]
    assert woodie[
        ["Pivot Point", "Resistance 2", "Support 2"]
    ].tolist() == pytest.approx(
        [107.5882337785246, 109.84778602755249, 105.3286815294967]
    )

    # DeMark's X adds the low again after a falling period, the high after a rising one.
    demark = overlap_model.get_pivot_points(*bars, method="demark")
    high, low, close, opening = (series.iloc[29] for series in bars)
    extreme = low if close < opening else high if close > opening else close
    total = high + low + close + extreme
    assert demark.iloc[30].tolist() == pytest.approx(
        [total / 4, total / 2 - low, total / 2 - high]
    )
    with pytest.raises(ValueError, match="open"):
        overlap_model.get_pivot_points(*bars[:3], method="demark")


def test_trix_signal_line_matches_talib():
    result = overlap_model.get_trix(_sample_close(), 5, 3)

    # TA-Lib's EMA(TRIX(5) / 100, 3).
    assert result.loc[[20, 30, 59], "Signal Line"].tolist() == pytest.approx(
        [0.0020975598, 0.0023620344, 0.0042020924]
    )
    assert result["Histogram"].iloc[59] == pytest.approx(0.0058112345 - 0.0042020924)
    pd.testing.assert_series_equal(
        result["TRIX"], overlap_model.get_trix(_sample_close(), 5), check_names=False
    )


def test_parabolic_sar_matches_talib_sarext():
    prices = _sample_bars()
    result = overlap_model.get_parabolic_sar(
        prices["High"], prices["Low"], 0.01, 0.03, 0.25, 0.015, 0.02, 0.15, 0.0, 0.01
    )

    # The absolute value of TA-Lib's SAREXT(0, 0.01, 0.01, 0.03, 0.25, 0.015, 0.02, 0.15).
    assert result.iloc[[1, 20, 40, 59]].tolist() == pytest.approx(
        [99.0, 98.88972868096964, 105.64951174172081, 111.14141839984173]
    )

    # A negative start value starts a short at its absolute value.
    short = overlap_model.get_parabolic_sar(
        prices["High"], prices["Low"], start_value=-110.0
    )
    assert short.iloc[1] == pytest.approx(110.0)


def test_volume_weighted_average_price_anchors():
    prices = _sample_bars()
    prices.index = pd.period_range("2024-01-01", periods=60, freq="D")
    bars = prices["High"], prices["Low"], prices["Close"], prices["Volume"]

    # TA-Lib's VWAP accumulates from the first period.
    cumulative = overlap_model.get_volume_weighted_average_price(*bars, 14, "all")
    assert cumulative.iloc[[9, 30, 59]].tolist() == pytest.approx(
        [103.8046574946, 105.1557612481, 108.8683995253]
    )

    # pandas-ta-classic's vwap(anchor="W") starts again every Monday (2024-01-08).
    weekly = overlap_model.get_volume_weighted_average_price(*bars, 14, "weekly")
    assert weekly.iloc[[6, 7, 30, 59]].tolist() == pytest.approx(
        [103.76381638, 104.81575927, 107.53558314, 118.3897567]
    )
    typical_price = (prices["High"] + prices["Low"] + prices["Close"]) / 3
    assert weekly.iloc[7] == pytest.approx(typical_price.iloc[7])

    dated = overlap_model.get_volume_weighted_average_price(*bars, 14, "2024-02-01")
    assert dated.iloc[:31].isna().all()
    assert dated.iloc[31] == pytest.approx(typical_price.iloc[31])

    with pytest.raises(ValueError, match="anchor"):
        overlap_model.get_volume_weighted_average_price(*bars, 14, "session")


def test_new_overlap_indicators_stop_where_the_prices_stop():
    prices = _sample_bars()
    prices.iloc[-3:] = float("nan")
    prices.iloc[40] = float("nan")
    high, low, close, volume = (
        prices["High"],
        prices["Low"],
        prices["Close"],
        prices["Volume"],
    )

    for result in (
        overlap_model.get_holt_winters_moving_average(close),
        overlap_model.get_holt_winters_channel(close),
        overlap_model.get_elastic_volume_weighted_moving_average(close, volume, 5),
        overlap_model.get_gann_hilo_activator(high, low, close, 3, 3),
        overlap_model.get_zigzag(high, low),
    ):
        assert result.iloc[-3:].isna().all(axis=None)
        assert result.iloc[[40]].isna().all(axis=None)


def test_guppy_multiple_moving_averages_are_guppys_exponential_averages(recorder):
    prices = sample_prices()["Close"]
    result = overlap_model.get_guppy_multiple_moving_averages(prices)

    windows = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
    assert list(result.columns) == [
        *[f"Short EMA {window}" for window in windows[:6]],
        *[f"Long EMA {window}" for window in windows[6:]],
        "Short Group Spread",
        "Long Group Spread",
        "Group Spread",
    ]
    for window in windows:
        group = "Short" if window < 30 else "Long"
        pd.testing.assert_series_equal(
            result[f"{group} EMA {window}"],
            overlap_model.get_exponential_moving_average(prices, window),
            check_names=False,
        )

    short = result.iloc[:, :6]
    pd.testing.assert_series_equal(
        result["Short Group Spread"].iloc[14:],
        short.max(axis=1).iloc[14:] - short.min(axis=1).iloc[14:],
        check_names=False,
    )
    # The 60-period average of 60 prices starts on the last one, as do the long spreads.
    assert result["Short Group Spread"].first_valid_index() == prices.index[14]
    assert result["Long Group Spread"].first_valid_index() == prices.index[59]
    assert result["Group Spread"].iloc[-1] == pytest.approx(
        result.iloc[-1, :6].mean() - result.iloc[-1, 6:12].mean()
    )
    recorder.capture(result.round(3))


def test_guppy_multiple_moving_averages_take_their_own_windows():
    prices = sample_prices()["Close"]
    result = overlap_model.get_guppy_multiple_moving_averages(prices, [2, 4], [10, 20])

    assert list(result.columns[:4]) == [
        "Short EMA 2",
        "Short EMA 4",
        "Long EMA 10",
        "Long EMA 20",
    ]
    assert result["Group Spread"].first_valid_index() == prices.index[19]


def test_pascal_weighted_moving_average_long_window():
    """The binomial weights of a window beyond about 1,030 exceed the float range; they are scaled instead."""
    import math

    import numpy as np

    prices = pd.Series(np.linspace(100.0, 200.0, 1300))
    result = overlap_model.get_pascal_weighted_moving_average(prices, 1100)

    assert result.notna().sum() == 201
    # Symmetric weights on a straight line give the price in the middle of the window.
    assert result.iloc[-1] == pytest.approx(prices.iloc[-1100:].mean(), rel=1e-9)

    # Small windows keep the exact binomial coefficients.
    small = overlap_model.get_pascal_weighted_moving_average(prices, 10)
    weights = np.array([math.comb(9, i) for i in range(10)], dtype=float)
    expected = (prices.iloc[-10:].to_numpy() @ weights) / weights.sum()
    assert small.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_fibonacci_weighted_moving_average_long_window():
    """The Fibonacci numbers of a window beyond about 1,475 exceed the float range; they are scaled instead."""
    from fractions import Fraction

    import numpy as np

    prices = pd.Series(np.random.default_rng(0).normal(0, 1, 2100).cumsum() + 100)
    result = overlap_model.get_fibonacci_weighted_moving_average(prices, 2000)

    fibonacci = [1, 1]
    while len(fibonacci) < 2000:
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    window = [Fraction(price) for price in prices.iloc[-2000:]]
    expected = sum(f * p for f, p in zip(fibonacci, window)) / sum(fibonacci)

    assert result.notna().sum() == 101
    assert result.iloc[-1] == pytest.approx(float(expected), rel=1e-12)
