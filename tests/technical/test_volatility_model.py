"""Volatility Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.volatility import volatility_model
from tests.technical.sample_prices import sample_prices


def test_get_true_range(recorder):
    recorder.capture(
        volatility_model.get_true_range(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
        ).round(3)
    )


def test_get_average_true_range(recorder):
    prices = sample_prices()
    result = volatility_model.get_average_true_range(
        prices["High"], prices["Low"], prices["Close"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_keltner_channels(recorder):
    prices = sample_prices()
    result = volatility_model.get_keltner_channels(
        prices["High"], prices["Low"], prices["Close"], 10, 5, 2
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_bollinger_bands(recorder):
    recorder.capture(
        volatility_model.get_bollinger_bands(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10, 2
        ).round(3)
    )


def test_get_supertrend(recorder):
    recorder.capture(
        volatility_model.get_supertrend(
            pd.Series([100, 110, 120, 130, 80, 90, 100, 110, 105, 95]),
            pd.Series([90, 95, 100, 110, 60, 80, 90, 95, 90, 80]),
            pd.Series([95, 105, 110, 120, 70, 85, 95, 105, 95, 85]),
            5,
            3.0,
        ).round(3)
    )


def test_volatility_cone_annualizes_with_the_periods_per_year():
    import numpy as np

    rng = np.random.default_rng(0)
    index = pd.bdate_range("2018-01-01", periods=2000)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))), index)

    daily = volatility_model.get_volatility_cone(prices, [20])
    weekly = volatility_model.get_volatility_cone(
        prices.resample("W").last(), [20], periods_per_year=52
    )

    # Both annualize to the same volatility of about 1% a day times the square root of 252.
    assert daily.loc[20, "Median"] == pytest.approx(0.01 * np.sqrt(252), rel=0.1)
    assert weekly.loc[20, "Median"] == pytest.approx(daily.loc[20, "Median"], rel=0.1)


def test_supertrend_direction_is_nan_until_the_bands_start():
    index = pd.date_range("2024-01-01", periods=30)
    close = pd.Series([100 + i for i in range(30)], index=index, dtype="float64")

    supertrend = volatility_model.get_supertrend(close + 1, close - 1, close, window=10)

    direction = supertrend["Trend Direction"]
    assert direction.iloc[:9].isna().all()
    assert direction.iloc[9:].notna().all()


def test_recursive_averages_stop_where_the_prices_stop():
    """A series that ends early (a delisted ticker) has no Wilder average, Average True
    Range or Supertrend after its last price, while a single missing day in the middle is
    NaN on that day and the recursion carries on from the last average after it."""
    import numpy as np  # noqa: PLC0415

    prices = sample_prices()
    high, low, close = (
        prices["High"].copy(),
        prices["Low"].copy(),
        prices["Close"].copy(),
    )
    for series in (high, low, close):
        series.iloc[-5:] = np.nan
        series.iloc[40] = np.nan

    true_range = volatility_model.get_true_range(high, low, close)
    wilder = volatility_model.get_wilder_moving_average(true_range, 14)
    average_true_range = volatility_model.get_average_true_range(high, low, close, 14)
    supertrend = volatility_model.get_supertrend(high, low, close)

    for result in (wilder, average_true_range, supertrend["Supertrend"]):
        assert result.iloc[-5:].isna().all()
        assert pd.isna(result.iloc[40])
    assert supertrend["Trend Direction"].iloc[-5:].isna().all()

    # After the gap the recursion continues from the average of the day before it.
    assert wilder.iloc[41] == pytest.approx(
        wilder.iloc[39] + (true_range.iloc[41] - wilder.iloc[39]) / 14
    )


def test_mass_index_sums_the_ratio_of_the_single_and_double_ema():
    from financetoolkit.technicals.overlap.overlap_model import (  # noqa: PLC0415
        get_exponential_moving_average,
    )

    prices = sample_prices()
    result = volatility_model.get_mass_index(prices["High"], prices["Low"], 9, 25)

    single_ema = get_exponential_moving_average(prices["High"] - prices["Low"], 9)
    double_ema = get_exponential_moving_average(single_ema, 9)
    expected = (single_ema / double_ema).rolling(25).sum()

    pd.testing.assert_series_equal(result, expected)
    # The double EMA starts on the 17th period and the 25-period sum 24 periods later.
    assert result.first_valid_index() == prices.index[40]


def test_mass_index_of_a_constant_range_is_the_sum_window():
    index = pd.RangeIndex(80)
    result = volatility_model.get_mass_index(
        pd.Series(101.0, index=index), pd.Series(99.0, index=index), 9, 25
    )

    assert result.dropna().round(12).eq(25).all()


def test_chandelier_exit_hangs_from_the_extremes_by_the_average_true_range():
    prices = sample_prices()
    result = volatility_model.get_chandelier_exit(
        prices["High"], prices["Low"], prices["Close"], 22, 3.0
    )
    average_true_range = volatility_model.get_average_true_range(
        prices["High"], prices["Low"], prices["Close"], 22
    )

    pd.testing.assert_series_equal(
        result["Long Exit"],
        prices["High"].rolling(22).max() - 3.0 * average_true_range,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["Short Exit"],
        prices["Low"].rolling(22).min() + 3.0 * average_true_range,
        check_names=False,
    )
    assert result["Long Exit"].first_valid_index() == prices.index[21]


def test_get_bollinger_percent_b_and_bandwidth(recorder):
    prices = sample_prices()
    result = volatility_model.get_bollinger_percent_b_and_bandwidth(
        prices["Close"], 20, 2
    ).round(4)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_bollinger_percent_b_and_bandwidth_match_talib():
    """%B and BandWidth from TA-Lib's BBANDS(20, 2, 2) with a simple average."""
    rng = np.random.default_rng(7)
    closes = pd.Series(100 + rng.normal(0, 1, 80).cumsum())

    result = volatility_model.get_bollinger_percent_b_and_bandwidth(closes, 20, 2)

    assert result["%B"].first_valid_index() == 19
    assert result["%B"].iloc[-3:].tolist() == pytest.approx(
        [0.285528, 0.26088, 0.573318], abs=1e-6
    )
    assert result["BandWidth"].iloc[-3:].tolist() == pytest.approx(
        [0.064015, 0.060603, 0.057505], abs=1e-6
    )


def test_bollinger_percent_b_without_band_width_is_missing():
    result = volatility_model.get_bollinger_percent_b_and_bandwidth(
        pd.Series([10.0] * 5 + [11.0]), 5, 2
    )

    assert np.isnan(result["%B"].iloc[4])
    assert result["BandWidth"].iloc[4] == 0
    # The jump to 11 lifts the mean to 10.2 and the deviation to 0.4, which puts the close
    # exactly on the upper band (10.2 + 2 * 0.4), a %B of 1.
    assert result["%B"].iloc[5] == pytest.approx(1.0)


def _sample():
    return sample_prices().reset_index(drop=True)


def test_volatility_indicators_match_reference_libraries():
    prices = _sample()
    high, low, close = prices["High"], prices["Low"], prices["Close"]

    # TA-Lib's ADR(14), CVI(10, 10) / 100, RVI(14, 10) and RVIR(14, 10).
    average_day_range = volatility_model.get_average_day_range(high, low, 14)
    assert average_day_range.iloc[:13].isna().all()
    assert average_day_range.iloc[[19, 30, 59]].tolist() == pytest.approx(
        [2.17497329, 2.2168346, 2.24288844]
    )

    chaikin_volatility = volatility_model.get_chaikin_volatility(high, low, 10, 10)
    assert chaikin_volatility.iloc[:19].isna().all()
    assert chaikin_volatility.iloc[[19, 30, 59]].tolist() == pytest.approx(
        [-0.028927425, 0.1319172289, -0.0913495994]
    )

    relative_volatility_index = volatility_model.get_relative_volatility_index(
        close, 14, 10
    )
    assert relative_volatility_index.iloc[:22].isna().all()
    assert relative_volatility_index.iloc[[22, 30, 59]].tolist() == pytest.approx(
        [70.48207403, 50.78430053, 69.90723712]
    )

    refined = (
        volatility_model.get_relative_volatility_index(high, 14, 10)
        + volatility_model.get_relative_volatility_index(low, 14, 10)
    ) / 2
    assert refined.iloc[[22, 30, 59]].tolist() == pytest.approx(
        [63.46709017, 49.33711895, 67.64057523]
    )

    # TA-Lib's ACCBANDS(20).
    acceleration_bands = volatility_model.get_acceleration_bands(
        high, low, close, 20, 4.0
    )
    assert acceleration_bands.iloc[:19].isna().all().all()
    assert acceleration_bands.iloc[59].tolist() == pytest.approx(
        [120.30699137, 115.0135282, 109.45529132]
    )
    assert acceleration_bands["Upper Band"].iloc[30] == pytest.approx(111.7766954)

    # TA-Lib's DONCHIAN(10), which includes the current period.
    donchian = volatility_model.get_donchian_channels(
        high, low, 10, include_current=True
    )
    assert donchian.iloc[[9, 30, 59]].to_numpy().ravel().tolist() == pytest.approx(
        [106.08106002, 102.54053001, 99.0]
        + [112.57935997, 109.09591919, 105.61247842]
        + [121.44752707, 115.88136724, 110.31520741]
    )

    # pandas-ta-classic's pdist and aberration(5, 15) middle band, and ta's
    # KeltnerChannel(10, original_version=True).
    price_distance = volatility_model.get_price_distance(
        prices["Open"], high, low, close
    )
    assert pd.isna(price_distance.iloc[0])
    assert price_distance.iloc[[1, 30, 59]].tolist() == pytest.approx(
        [3.49865262, 2.74524901, 3.19103989]
    )

    aberration = volatility_model.get_aberration(high, low, close, 5, 15)
    assert aberration["Middle Band"].iloc[[4, 30, 59]].tolist() == pytest.approx(
        [102.78426987, 108.6400658, 117.44573573]
    )

    keltner = volatility_model.get_keltner_channels(
        high, low, close, 10, 10, 1, method="original"
    )
    assert keltner["Upper Line"].iloc[[9, 30, 59]].tolist() == pytest.approx(
        [105.80239872, 111.92276449, 116.90299419]
    )
    assert keltner["Middle Line"].iloc[[9, 30, 59]].tolist() == pytest.approx(
        [103.63891716, 109.50717826, 114.73884451]
    )


def test_squeeze_momentum_matches_talib():
    prices = _sample()
    result = volatility_model.get_squeeze_momentum(
        prices["High"], prices["Low"], prices["Close"], 20, 2.0, [2.0, 1.5, 1.0]
    )

    # TA-Lib's SQZMOM momentum starts on period 2 * (20 - 1).
    assert result["Momentum"].iloc[:38].isna().all()
    assert result["Momentum"].iloc[[38, 45, 59]].tolist() == pytest.approx(
        [0.27079719, 4.98444165, 1.36496926]
    )

    # TA-Lib's squeeze code from period 38 on, with -1 (no squeeze) as 0.
    talib_code = [1, 1, 1] + [0] * 14 + [1, 1, 1, 1, 0]
    assert result["Squeeze"].iloc[38:].tolist() == talib_code
    assert pd.isna(result["Squeeze"].iloc[18])

    # One multiplier gives Carter's squeeze as 1 (on) or 0 (off).
    classic = volatility_model.get_squeeze_momentum(
        prices["High"], prices["Low"], prices["Close"], 20, 2.0, [1.5]
    )
    assert set(classic["Squeeze"].dropna().unique()) <= {0.0, 1.0}


def test_atr_based_volatility_indicators_converge_to_talib():
    prices = _sample()
    high, low, close = prices["High"], prices["Low"], prices["Close"]
    average_true_range = volatility_model.get_average_true_range(high, low, close, 14)

    natr = volatility_model.get_normalized_average_true_range(high, low, close, 14)
    pd.testing.assert_series_equal(natr, average_true_range / close)
    # TA-Lib's NATR(14) / 100; the ATR seeds converge over the sample.
    assert natr.iloc[59] == pytest.approx(0.0191143326, rel=1e-3)

    # TA-Lib's CKSP(10, 1, 9), equal once the ATR seeds have converged.
    stops = volatility_model.get_chande_kroll_stop(high, low, close, 10, 1.0, 9)
    assert stops.iloc[:17].isna().all().all()
    assert stops.iloc[59].tolist() == pytest.approx(
        [119.15614663, 112.60658786], rel=1e-5
    )


def test_adaptive_price_zone_smooths_over_the_square_root_of_the_window():
    prices = _sample()
    high, low, close = prices["High"], prices["Low"], prices["Close"]
    result = volatility_model.get_adaptive_price_zone(high, low, close, 20, 2.0)

    # Two EMAs of ceil(sqrt(20)) = 5 seed periods start on period 8.
    assert result.iloc[:8].isna().all().all()
    width = result["Upper Band"] - result["Middle Band"]
    assert (result["Middle Band"] - result["Lower Band"]).equals(width)

    # The first EMA is seeded with five closes and runs with the constant of the
    # square-root period; the second is seeded with the first five values of the first.
    alpha = 2 / (20**0.5 + 1)
    first_ema = [close.iloc[:5].mean()]
    for price in close.iloc[5:9]:
        first_ema.append(alpha * price + (1 - alpha) * first_ema[-1])
    assert result["Middle Band"].iloc[8] == pytest.approx(np.mean(first_ema))
    assert result["Middle Band"].iloc[8:].notna().all()


def test_profit_maximizer_follows_the_average():
    prices = _sample()
    high, low, close = prices["High"], prices["Low"], prices["Close"]
    result = volatility_model.get_profit_maximizer(high, low, close, 10, 10, 3.0)

    # The EMA and ATR of ten periods start on period 9, in an uptrend on the long stop.
    assert result.iloc[:9].isna().all().all()
    assert result["Trend Direction"].iloc[9] == 1
    average = (high + low).div(2).iloc[:10].mean()
    atr = volatility_model.get_average_true_range(high, low, close, 10).iloc[9]
    assert result["PMax"].iloc[9] == pytest.approx(average - 3 * atr)
    assert set(result["Trend Direction"].dropna().unique()) <= {-1.0, 1.0}

    # A falling market turns the trend down onto the short stop above the average.
    falling = pd.Series(np.linspace(200, 100, 80))
    down = volatility_model.get_profit_maximizer(
        falling + 1, falling - 1, falling, 10, 10, 1.0
    )
    assert down["Trend Direction"].iloc[-1] == -1
    assert down["PMax"].iloc[-1] > falling.iloc[-1]


def test_market_thermometer_is_zero_on_inside_periods():
    high = pd.Series([10.0, 11.0, 10.5, 12.0, 11.0])
    low = pd.Series([9.0, 9.5, 9.8, 9.0, 9.5])
    result = volatility_model.get_market_thermometer(high, low, 2)

    # A higher high, an inside period, a higher high and lower low, a lower high and
    # higher low.
    assert pd.isna(result["Thermometer"].iloc[0])
    assert result["Thermometer"].iloc[1:].tolist() == pytest.approx(
        [1.0, 0.0, 1.5, 0.0]
    )
    assert result["Thermometer Average"].iloc[2] == pytest.approx(0.5)


def test_relative_volatility_index_is_missing_without_moves():
    flat = pd.Series([10.0] * 40)
    assert volatility_model.get_relative_volatility_index(flat, 14, 10).isna().all()


def test_new_volatility_indicators_stop_where_the_prices_stop():
    prices = _sample().copy()
    prices.iloc[-3:] = np.nan
    high, low, close = prices["High"], prices["Low"], prices["Close"]

    for result in (
        volatility_model.get_normalized_average_true_range(high, low, close, 5),
        volatility_model.get_chaikin_volatility(high, low, 5, 5),
        volatility_model.get_relative_volatility_index(close, 5, 5),
        volatility_model.get_profit_maximizer(high, low, close, 5, 5, 3.0)["PMax"],
        volatility_model.get_market_thermometer(high, low, 5)["Thermometer Average"],
        volatility_model.get_adaptive_price_zone(high, low, close, 9, 2.0)[
            "Middle Band"
        ],
    ):
        assert result.iloc[-3:].isna().all()
        assert result.iloc[-10:-3].notna().all()


def random_prices() -> pd.DataFrame:
    """80 bars of a random walk with a high and low around each close."""
    rng = np.random.default_rng(7)
    close = 100 + rng.normal(0, 1, 80).cumsum()
    return pd.DataFrame(
        {
            "High": close + 1 + rng.random(80),
            "Low": close - 1 - rng.random(80),
            "Close": close,
        }
    )


def test_standard_deviation_channel_matches_pandas_ta_classic():
    """pandas-ta-classic's tos_stdevall(length=40): the line and the upper band at 2."""
    closes = random_prices()["Close"]
    result = volatility_model.get_standard_deviation_channel(closes, 40)

    assert result["Linear Regression"].first_valid_index() == 40
    assert [
        result["Linear Regression"].iloc[40],
        result["Upper Band 2"].iloc[40],
        result["Linear Regression"].iloc[-1],
        result["Upper Band 2"].iloc[-1],
    ] == pytest.approx([84.766764, 87.478202, 85.791096, 88.502533], abs=1e-6)
    assert list(result.columns) == [
        "Linear Regression",
        "Lower Band 1",
        "Upper Band 1",
        "Lower Band 2",
        "Upper Band 2",
        "Lower Band 3",
        "Upper Band 3",
    ]


def test_standard_deviation_channel_takes_one_width():
    result = volatility_model.get_standard_deviation_channel(
        random_prices()["Close"], num_std_dev=1.5
    )

    assert list(result.columns) == [
        "Linear Regression",
        "Lower Band 1.5",
        "Upper Band 1.5",
    ]
    assert result.notna().all().all()


def test_signal_to_noise_ratio_of_a_steady_trend():
    """A close that rises 1 a day with a high and low 0.5 around it has a True Range of 1.5
    after the first day, so the 5-day move of 5 is 5 / 1.5 ATRs once the seed is gone.
    """
    close = pd.Series(100.0 + np.arange(80))
    result = volatility_model.get_signal_to_noise_ratio(
        close + 0.5, close - 0.5, close, 5
    )

    assert result.first_valid_index() == 5
    # The ATR seeds on the first five True Ranges: (1 + 4 * 1.5) / 5 = 1.4 on period 4, and
    # on period 5 it is 1.4 + (1.5 - 1.4) / 5 = 1.42.
    assert result.iloc[5] == pytest.approx(5 / 1.42)
    assert result.iloc[-1] == pytest.approx(5 / 1.5)


def test_get_signal_to_noise_ratio(recorder):
    prices = random_prices()
    result = volatility_model.get_signal_to_noise_ratio(
        prices["High"], prices["Low"], prices["Close"], 14
    )
    atr = volatility_model.get_average_true_range(
        prices["High"], prices["Low"], prices["Close"], 14
    )

    pd.testing.assert_series_equal(
        result, (prices["Close"] - prices["Close"].shift(14)).abs() / atr
    )
    recorder.capture(result.round(6))


def test_signal_to_noise_ratio_without_a_range_is_missing():
    flat = pd.Series(10.0, index=range(30))

    assert volatility_model.get_signal_to_noise_ratio(flat, flat, flat, 5).isna().all()


def test_price_bands_of_a_straight_line_have_no_width():
    """On a price that rises by the same step every period, the 20 and 2-period averages
    stay 9 apart, so the deviation has no spread and the bands lie on the middle band.
    """
    close = pd.Series(np.arange(60, dtype=float))
    result = volatility_model.get_price_bands(close, 20, 2, 2)

    assert result["Middle Band"].first_valid_index() == 19
    assert result["Upper Band"].first_valid_index() == 38
    pd.testing.assert_series_equal(
        result["Upper Band"].dropna(),
        result["Middle Band"].iloc[38:],
        check_names=False,
        atol=1e-9,
    )


def test_get_price_bands(recorder):
    closes = random_prices()["Close"]
    result = volatility_model.get_price_bands(closes, 10, 2, 3)

    middle = closes.rolling(10).mean()
    deviation = middle - closes.rolling(3).mean()
    # The population standard deviation of the last 10 deviations, window by window.
    expected_width = deviation.rolling(10).apply(lambda values: values.std(ddof=0))
    assert result["Upper Band"].first_valid_index() == 18
    pd.testing.assert_series_equal(
        result["Upper Band"], middle + 2 * expected_width, check_names=False
    )
    pd.testing.assert_series_equal(
        result["Lower Band"], middle - 2 * expected_width, check_names=False
    )
    assert list(result.columns) == ["Upper Band", "Middle Band", "Lower Band"]
    recorder.capture(result.round(6))


def test_price_bands_reject_an_unknown_moving_average():
    with pytest.raises(ValueError, match="moving_average must be one of"):
        volatility_model.get_price_bands(random_prices()["Close"], moving_average="x")
