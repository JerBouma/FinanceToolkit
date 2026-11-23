"""Momentum Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.momentum import momentum_model
from tests.technical.sample_prices import sample_prices


def test_get_money_flow_index(recorder):
    prices = sample_prices()
    result = momentum_model.get_money_flow_index(
        prices["High"], prices["Low"], prices["Close"], prices["Volume"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_williams_percent_r(recorder):
    prices = sample_prices()
    result = momentum_model.get_williams_percent_r(
        prices["High"], prices["Low"], prices["Close"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_aroon_indicator(recorder):
    prices = sample_prices()
    result = momentum_model.get_aroon_indicator(
        prices["High"], prices["Low"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_commodity_channel_index(recorder):
    prices = sample_prices()
    result = momentum_model.get_commodity_channel_index(
        prices["High"], prices["Low"], prices["Close"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_relative_vigor_index(recorder):
    recorder.capture(
        momentum_model.get_relative_vigor_index(
            pd.Series([10.0, 10.5, 10.2, 10.8, 11.0, 10.6, 10.9, 11.3, 11.1, 11.6]),
            pd.Series([10.8, 11.0, 10.9, 11.4, 11.5, 11.0, 11.4, 11.8, 11.6, 12.0]),
            pd.Series([9.8, 10.1, 10.0, 10.5, 10.6, 10.3, 10.6, 11.0, 10.9, 11.2]),
            pd.Series([10.4, 10.3, 10.7, 11.1, 10.7, 10.8, 11.2, 11.2, 11.5, 11.8]),
            3,
        ).round(3)
    )


def test_get_force_index(recorder):
    prices = sample_prices()
    result = momentum_model.get_force_index(
        prices["Close"], prices["Volume"], 20
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_ultimate_oscillator(recorder):
    prices = sample_prices()
    result = momentum_model.get_ultimate_oscillator(
        prices["High"], prices["Low"], prices["Close"], 7, 14, 28
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_detrended_price_oscillator(recorder):
    prices = sample_prices()
    result = momentum_model.get_detrended_price_oscillator(
        prices["Close"], 20, "sma"
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_average_directional_index(recorder):
    prices = sample_prices()
    result = momentum_model.get_average_directional_index(
        prices["High"], prices["Low"], prices["Close"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_chande_momentum_oscillator(recorder):
    prices = sample_prices()
    result = momentum_model.get_chande_momentum_oscillator(prices["Close"], 10).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_ichimoku_cloud(recorder):
    recorder.capture(
        momentum_model.get_ichimoku_cloud(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
            5,
            3,
        ).round(3)
    )


def test_get_stochastic_oscillator(recorder):
    prices = sample_prices()
    result = momentum_model.get_stochastic_oscillator(
        prices["High"], prices["Low"], prices["Close"], 10, 5
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_moving_average_convergence_divergence(recorder):
    prices = sample_prices()
    result = momentum_model.get_moving_average_convergence_divergence(
        prices["Close"], 5, 10, 5
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_relative_strength_index(recorder):
    prices = sample_prices()
    result = momentum_model.get_relative_strength_index(prices["Close"], 5).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_balance_of_power(recorder):
    recorder.capture(
        momentum_model.get_balance_of_power(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
        ).round(3)
    )


def test_get_awesome_oscillator(recorder):
    recorder.capture(
        momentum_model.get_awesome_oscillator(
            pd.Series([100, 110, 120, 130, 80, 90, 100, 110]),
            pd.Series([90, 95, 100, 110, 60, 80, 90, 95]),
            2,
            5,
        ).round(3)
    )


def test_get_vortex_indicator(recorder):
    recorder.capture(
        momentum_model.get_vortex_indicator(
            pd.Series([100, 110, 120, 130, 80, 90, 100, 110]),
            pd.Series([90, 95, 100, 110, 60, 80, 90, 95]),
            pd.Series([95, 105, 110, 120, 70, 85, 95, 105]),
            5,
        ).round(3)
    )


def test_get_elder_ray_index(recorder):
    recorder.capture(
        momentum_model.get_elder_ray_index(
            pd.Series([100, 110, 120, 130, 80, 90, 100, 110]),
            pd.Series([90, 95, 100, 110, 60, 80, 90, 95]),
            pd.Series([95, 105, 110, 120, 70, 85, 95, 105]),
            5,
        ).round(3)
    )


def test_get_rate_of_change(recorder):
    recorder.capture(
        momentum_model.get_rate_of_change(
            pd.Series([100, 110, 120, 130, 80, 90, 100, 110]), 3
        ).round(3)
    )


def test_get_choppiness_index(recorder):
    recorder.capture(
        momentum_model.get_choppiness_index(
            pd.Series([100, 110, 120, 130, 80, 90, 100, 110]),
            pd.Series([90, 95, 100, 110, 60, 80, 90, 95]),
            pd.Series([95, 105, 110, 120, 70, 85, 95, 105]),
            5,
        ).round(3)
    )


def test_get_know_sure_thing(recorder):
    prices = sample_prices()
    result = momentum_model.get_know_sure_thing(prices["Close"]).round(6)

    # The windows fill within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_know_sure_thing_custom_parameters(recorder):
    recorder.capture(
        momentum_model.get_know_sure_thing(
            pd.Series([100, 102, 101, 105, 110, 108, 112, 115, 111, 120]),
            roc_windows=[2, 3, 4, 5],
            sma_windows=[2, 2, 2, 2],
            weights=[1, 1, 1, 1],
            signal_window=3,
        ).round(3)
    )


def test_relative_strength_index_matches_wilder():
    """The first average covers the first 14 actual changes, as Wilder and TA-Lib do."""
    closes = pd.Series(
        [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89]
        + [46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21]
    )

    rsi = momentum_model.get_relative_strength_index(closes, 14)

    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[15:20].round(2).tolist() == [66.25, 66.48, 69.35, 66.29, 57.92]


HIGH = pd.Series(
    [10.8, 11.0, 10.9, 11.4, 11.5, 11.0, 11.4, 11.8, 11.6, 12.0, 12.3, 11.9]
)
LOW = pd.Series([9.8, 10.1, 10.0, 10.5, 10.6, 10.3, 10.6, 11.0, 10.9, 11.2, 11.5, 11.1])
OPEN = pd.Series(
    [10.0, 10.5, 10.2, 10.8, 11.0, 10.6, 10.9, 11.3, 11.1, 11.6, 11.8, 11.7]
)
CLOSE = pd.Series(
    [10.4, 10.3, 10.7, 11.1, 10.7, 10.8, 11.2, 11.2, 11.5, 11.8, 11.6, 11.3]
)
VOLUME = pd.Series(
    [1e6, 1.2e6, 0.9e6, 1.5e6, 1.1e6, 1e6, 1.3e6, 0.8e6, 1.4e6, 1.2e6, 1e6, 0.9e6]
)


def test_relative_vigor_index_matches_ehlers():
    window = 3
    result = momentum_model.get_relative_vigor_index(OPEN, HIGH, LOW, CLOSE, window)

    def weighted(values, i):
        return (values[i] + 2 * values[i - 1] + 2 * values[i - 2] + values[i - 3]) / 6

    close_open = (CLOSE - OPEN).tolist()
    high_low = (HIGH - LOW).tolist()
    rvi = [float("nan")] * len(CLOSE)
    for i in range(window + 2, len(CLOSE)):
        numerator = sum(weighted(close_open, j) for j in range(i - window + 1, i + 1))
        denominator = sum(weighted(high_low, j) for j in range(i - window + 1, i + 1))
        rvi[i] = numerator / denominator

    assert result["Relative Vigor Index"].first_valid_index() == window + 2
    assert result["Relative Vigor Index"].iloc[window + 2 :].tolist() == pytest.approx(
        rvi[window + 2 :]
    )
    assert result["Signal Line"].first_valid_index() == window + 5
    assert result["Signal Line"].iloc[-1] == pytest.approx(weighted(rvi, len(rvi) - 1))


def test_money_flow_index_excludes_the_first_period():
    window = 4
    mfi = momentum_model.get_money_flow_index(HIGH, LOW, CLOSE, VOLUME, window)

    typical = ((HIGH + LOW + CLOSE) / 3).tolist()
    flows = [t * v for t, v in zip(typical, VOLUME)]
    positive = sum(
        flows[i] for i in range(1, window + 1) if typical[i] > typical[i - 1]
    )
    negative = sum(
        flows[i] for i in range(1, window + 1) if typical[i] < typical[i - 1]
    )

    assert mfi.first_valid_index() == window
    assert mfi.iloc[window] == pytest.approx(100 * positive / (positive + negative))


def test_average_directional_index_starts_movement_on_the_second_period():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(0, 1, 80).cumsum())
    high = close + rng.uniform(0.2, 1.0, 80)
    low = close - rng.uniform(0.2, 1.0, 80)
    window = 5

    adx = momentum_model.get_average_directional_index(high, low, close, window)

    assert adx.first_valid_index() == 2 * window - 1


def test_balance_of_power_only_zeroes_flat_periods():
    bop = momentum_model.get_balance_of_power(
        pd.Series([10.0, 10.0, np.nan, 10.0]),
        pd.Series([11.0, 10.0, 11.0, 11.0]),
        pd.Series([9.0, 10.0, 9.0, 9.0]),
        pd.Series([10.5, 10.0, 10.5, np.nan]),
    )

    assert bop.iloc[0] == pytest.approx(0.25)
    assert bop.iloc[1] == 0
    assert bop.iloc[2:].isna().all()


def test_moving_average_convergence_divergence_seeds_every_average():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 + rng.normal(0, 1, 60).cumsum())

    macd = momentum_model.get_moving_average_convergence_divergence(prices, 12, 26, 9)

    assert macd["MACD Line"].first_valid_index() == 25
    assert macd["Signal Line"].first_valid_index() == 33
    assert macd["Signal Line"].iloc[33] == pytest.approx(
        macd["MACD Line"].iloc[25:34].mean()
    )


def test_rate_of_change_and_oscillators_are_decimals():
    index = pd.date_range("2024-01-01", periods=60)
    close = pd.Series([100 * 1.01**i for i in range(60)], index=index)

    rate_of_change = momentum_model.get_rate_of_change(close, window=10)
    assert rate_of_change.iloc[-1] == pytest.approx(1.01**10 - 1)

    ppo = momentum_model.get_percentage_price_oscillator(close, 12, 26)
    assert 0 < ppo["Percentage Price Oscillator"].iloc[-1] < 1


def test_windows_of_one_period_are_refused():
    index = pd.date_range("2024-01-01", periods=10)
    close = pd.Series(range(10, 20), index=index, dtype="float64")

    with pytest.raises(ValueError, match="at least 2"):
        momentum_model.get_choppiness_index(close + 1, close - 1, close, window=1)
    with pytest.raises(ValueError, match="at least 2"):
        momentum_model.get_commodity_channel_index(
            close + 1, close - 1, close, window=1
        )


def test_wilder_indicators_stop_where_the_prices_stop():
    """The RSI, ADX and MACD are NaN after a series' last price rather than repeating the
    last smoothed values."""
    prices = sample_prices()
    high, low, close = (
        prices["High"].copy(),
        prices["Low"].copy(),
        prices["Close"].copy(),
    )
    for series in (high, low, close):
        series.iloc[-5:] = np.nan

    for result in (
        momentum_model.get_relative_strength_index(close, 14),
        momentum_model.get_average_directional_index(high, low, close, 7),
        momentum_model.get_moving_average_convergence_divergence(close, 5, 10, 4),
    ):
        assert result.iloc[-5:].isna().all().all()
        assert result.iloc[-6:-5].notna().all().all()


def test_percentage_volume_oscillator_matches_its_emas():
    volumes = sample_prices()["Volume"]
    result = momentum_model.get_percentage_volume_oscillator(volumes, 12, 26, 9)

    # Each EMA seeds on the simple average of its first window, as in TA-Lib's ta_EMA.c.
    def ema(values, window):
        values = values.dropna()
        averages = [values.iloc[:window].mean()]
        for value in values.iloc[window:]:
            averages.append(averages[-1] + 2 / (window + 1) * (value - averages[-1]))
        return pd.Series(averages, index=values.index[window - 1 :])

    pvo = (ema(volumes, 12) - ema(volumes, 26)) / ema(volumes, 26)
    signal_line = ema(pvo, 9)

    pd.testing.assert_series_equal(
        result["Percentage Volume Oscillator"].dropna(), pvo.dropna(), check_names=False
    )
    pd.testing.assert_series_equal(
        result["Signal Line"].dropna(), signal_line, check_names=False
    )
    assert (
        result["Histogram"]
        .dropna()
        .equals(
            (result["Percentage Volume Oscillator"] - result["Signal Line"]).dropna()
        )
    )
    # A decimal, as the Percentage Price Oscillator, and no value before the long EMA.
    assert (
        result["Percentage Volume Oscillator"].first_valid_index() == volumes.index[25]
    )


def test_percentage_volume_oscillator_without_volume_is_nan():
    result = momentum_model.get_percentage_volume_oscillator(
        pd.Series(0.0, index=range(60)), 12, 26, 9
    )

    assert result.isna().all().all()


def random_walk() -> pd.Series:
    """80 closes of a seeded random walk, which moves both ways unlike the sample prices."""
    rng = np.random.default_rng(7)
    return pd.Series(100 + rng.normal(0, 1, 80).cumsum())


def test_get_stochastic_relative_strength_index(recorder):
    result = momentum_model.get_stochastic_relative_strength_index(
        random_walk(), 14, 5, 3
    ).round(3)

    # The windows fill within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_stochastic_relative_strength_index_matches_talib():
    """The values of TA-Lib's STOCHRSI(14, 5, 3, SMA) and STOCHRSI(14, 14, 3, SMA)."""
    result = momentum_model.get_stochastic_relative_strength_index(
        random_walk(), 14, 5, 3
    )

    # %K starts on period 14 + 5 - 1, %D two periods later, where TA-Lib starts both.
    assert result["Stochastic RSI %K"].first_valid_index() == 18
    assert result["Stochastic RSI %D"].first_valid_index() == 20
    assert result["Stochastic RSI %K"].iloc[-4:].tolist() == pytest.approx(
        [85.752577, 69.635095, 0.0, 100.0], abs=1e-6
    )
    assert result["Stochastic RSI %D"].iloc[-4:].tolist() == pytest.approx(
        [84.999198, 85.129224, 51.795891, 56.545032], abs=1e-6
    )

    longer = momentum_model.get_stochastic_relative_strength_index(
        random_walk(), 14, 14, 3
    )
    assert longer["Stochastic RSI %D"].iloc[-2:].tolist() == pytest.approx(
        [42.871038, 52.826448], abs=1e-6
    )


def test_stochastic_relative_strength_index_without_a_range_is_missing():
    # A steady rise has an RSI of 100 throughout, so there is no range to place it in.
    result = momentum_model.get_stochastic_relative_strength_index(
        pd.Series(range(1, 31), dtype="float64"), 5, 3, 2
    )

    assert result["Stochastic RSI %K"].isna().all()


def test_get_true_strength_index(recorder):
    result = momentum_model.get_true_strength_index(random_walk(), 25, 13, 7).round(3)

    # The windows fill within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_true_strength_index_matches_blau():
    """An independent loop of Blau's double smoothing with SMA-seeded EMAs."""

    def exponential_moving_average(values, window):
        averages, seed, previous = [np.nan] * len(values), [], None
        for i, value in enumerate(values):
            if np.isnan(value):
                continue
            if previous is None:
                seed.append(value)
                if len(seed) == window:
                    previous = averages[i] = sum(seed) / window
            else:
                previous = averages[i] = previous + 2 / (window + 1) * (
                    value - previous
                )
        return averages

    closes = random_walk().tolist()
    momentum = [np.nan] + [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    numerator = exponential_moving_average(exponential_moving_average(momentum, 25), 13)
    denominator = exponential_moving_average(
        exponential_moving_average([abs(value) for value in momentum], 25), 13
    )
    expected = [100 * n / d for n, d in zip(numerator, denominator)]

    result = momentum_model.get_true_strength_index(random_walk(), 25, 13, 7)

    # The first change is on period 1, so the TSI starts on 1 + 25 - 1 + 13 - 1.
    assert result["True Strength Index"].first_valid_index() == 37
    assert result["Signal Line"].first_valid_index() == 43
    assert result["True Strength Index"].iloc[37:].tolist() == pytest.approx(
        expected[37:]
    )
    assert result["Signal Line"].iloc[-1] == pytest.approx(
        exponential_moving_average(expected, 7)[-1]
    )
    assert result["True Strength Index"].abs().max() <= 100


def test_get_coppock_curve(recorder):
    result = momentum_model.get_coppock_curve(random_walk()).round(6)

    # The windows fill within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_coppock_curve_matches_talib():
    """TA-Lib's WMA(ROCP(14) + ROCP(11), 10), and the same per ticker of a DataFrame."""
    closes = random_walk()
    result = momentum_model.get_coppock_curve(closes)

    assert result.first_valid_index() == 23
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-0.048365, -0.04872, -0.038604], abs=1e-6
    )

    tickers = pd.DataFrame({"A": closes, "B": closes.iloc[::-1].to_numpy()})
    per_ticker = momentum_model.get_coppock_curve(tickers)
    pd.testing.assert_series_equal(per_ticker["A"], result, check_names=False)


def random_bars(length: int = 300) -> pd.DataFrame:
    """300 seeded random bars, long enough for Wilder's smoothing to converge to TA-Lib's."""
    rng = np.random.default_rng(11)
    close = 100 + rng.normal(0, 1, length).cumsum()
    opened = np.r_[close[0], close[:-1]] + rng.normal(0, 0.3, length)
    high = np.maximum(opened, close) + rng.uniform(0, 1, length)
    low = np.minimum(opened, close) - rng.uniform(0, 1, length)
    return pd.DataFrame({"Open": opened, "High": high, "Low": low, "Close": close})


def test_get_directional_movement_index(recorder):
    prices = sample_prices()
    result = momentum_model.get_directional_movement_index(
        prices["High"], prices["Low"], prices["Close"], 10
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_directional_movement_index_matches_talib_and_the_adx():
    """TA-Lib's PLUS_DM, MINUS_DM, PLUS_DI, MINUS_DI, DX and ADXR(14) once the seeds have
    decayed, and the ADX as Wilder's average of the DX."""
    bars = random_bars()
    high, low, close = bars["High"], bars["Low"], bars["Close"]
    result = momentum_model.get_directional_movement_index(high, low, close, 14)

    # Every column starts on period 14, the first with a full window of movements.
    assert (result.apply(lambda column: column.first_valid_index()) == 14).all()
    for column, expected in {
        "+DM": [2.843691, 2.64057, 2.451958],
        "-DM": [6.117981, 5.680982, 6.100133],
        "+DI": [10.939967, 10.640924, 9.869091],
        "-DI": [23.536494, 22.893124, 24.552938],
        "DX": [36.536599, 36.536599, 42.658284],
    }.items():
        assert result[column].iloc[-3:].tolist() == pytest.approx(expected, abs=1e-6)

    adx = momentum_model.get_average_directional_index(high, low, close, 14)
    pd.testing.assert_series_equal(
        adx,
        momentum_model.get_wilder_moving_average(result["DX"], 14),
        check_names=False,
    )

    rating = momentum_model.get_average_directional_index_rating(high, low, close, 14)
    assert rating.first_valid_index() == 3 * 14 - 2
    assert rating.iloc[-3:].tolist() == pytest.approx(
        [15.526234, 15.786392, 16.288776], abs=1e-6
    )


def test_get_average_directional_index_rating(recorder):
    prices = sample_prices()
    result = momentum_model.get_average_directional_index_rating(
        prices["High"], prices["Low"], prices["Close"], 7
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_aroon_oscillator_matches_talib():
    bars = random_bars()
    result = momentum_model.get_aroon_indicator(bars["High"], bars["Low"], 14)

    pd.testing.assert_series_equal(
        result["Aroon Oscillator"],
        result["Aroon Up"] - result["Aroon Down"],
        check_names=False,
    )
    assert result["Aroon Oscillator"].iloc[-3:].tolist() == pytest.approx(
        [-50.0, -50.0, -64.285714], abs=1e-6
    )


def test_get_absolute_price_oscillator(recorder):
    result = momentum_model.get_absolute_price_oscillator(random_walk(), 12, 26).round(
        6
    )

    # The windows fill within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_price_oscillators_match_talib():
    """TA-Lib's APO and PPO with their moving average types, and the MACD line."""
    closes = random_walk()

    simple = momentum_model.get_absolute_price_oscillator(closes, 12, 26, "simple")
    exponential = momentum_model.get_absolute_price_oscillator(closes, 12, 26)
    assert simple.first_valid_index() == exponential.first_valid_index() == 25
    assert simple.iloc[-3:].tolist() == pytest.approx(
        [-0.695263, -0.827153, -0.963935], abs=1e-6
    )
    assert exponential.iloc[-3:].tolist() == pytest.approx(
        [-0.495551, -0.509523, -0.393128], abs=1e-6
    )
    macd = momentum_model.get_moving_average_convergence_divergence(closes, 12, 26, 9)
    pd.testing.assert_series_equal(macd["MACD Line"], exponential, check_names=False)

    ppo = momentum_model.get_percentage_price_oscillator(closes, 12, 26, 9, "weighted")
    assert ppo["Percentage Price Oscillator"].iloc[-3:].tolist() == pytest.approx(
        [-0.00825328, -0.00834461, -0.00673998], abs=1e-8
    )

    with pytest.raises(ValueError, match="moving_average"):
        momentum_model.get_absolute_price_oscillator(closes, 12, 26, "wilder")


def test_percentage_price_oscillator_has_an_ema_signal_line():
    closes = random_walk()
    result = momentum_model.get_percentage_price_oscillator(closes, 12, 26, 9)

    signal_line = momentum_model.get_exponential_moving_average(
        result["Percentage Price Oscillator"], 9
    )
    pd.testing.assert_series_equal(
        result["Signal Line"], signal_line, check_names=False
    )
    pd.testing.assert_series_equal(
        result["Histogram"],
        result["Percentage Price Oscillator"] - result["Signal Line"],
        check_names=False,
    )
    assert result["Signal Line"].first_valid_index() == 33


def test_moving_average_convergence_divergence_variants_match_talib():
    """The histogram, TA-Lib's MACDEXT with simple moving averages (equal from the first
    value) and MACDFIX once the seeds have decayed."""
    closes = random_walk()
    simple = momentum_model.get_moving_average_convergence_divergence(
        closes, 5, 10, 4, "simple", "simple"
    )
    for column, expected in {
        "MACD Line": [-0.174394, 0.195711, 0.387302],
        "Signal Line": [-0.69479, -0.420308, -0.068449],
        "Histogram": [0.520396, 0.616019, 0.45575],
    }.items():
        assert simple[column].iloc[-3:].tolist() == pytest.approx(expected, abs=1e-6)

    fixed = momentum_model.get_moving_average_convergence_divergence(
        random_bars()["Close"], 12, 26, 9, fixed=True
    )
    for column, expected in {
        "MACD Line": [0.176816, 0.023121, -0.125808],
        "Signal Line": [0.45395, 0.367784, 0.269066],
        "Histogram": [-0.277133, -0.344663, -0.394873],
    }.items():
        assert fixed[column].iloc[-3:].tolist() == pytest.approx(expected, abs=1e-6)

    with pytest.raises(ValueError, match="fixed"):
        momentum_model.get_moving_average_convergence_divergence(
            closes, 5, 10, 4, fixed=True
        )


def test_know_sure_thing_takes_other_moving_averages():
    """TA-Lib's KSTEXT with weighted moving averages, in decimals instead of percent."""
    result = momentum_model.get_know_sure_thing(
        random_bars()["Close"],
        moving_average="weighted",
        signal_moving_average="weighted",
    )

    assert result["KST"].iloc[-3:].tolist() == pytest.approx(
        [0.0392952, -0.00958326, -0.04866799], abs=1e-8
    )
    assert result["Signal Line"].iloc[-3:].tolist() == pytest.approx(
        [0.10531716, 0.07793091, 0.04710109], abs=1e-8
    )


def test_get_momentum(recorder):
    result = momentum_model.get_momentum(random_walk(), 10).round(6)

    assert result.first_valid_index() == 10
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-2.773514, -1.700181, 0.61799], abs=1e-6
    )
    recorder.capture(result)


def test_get_accelerator_oscillator(recorder):
    bars = random_bars()
    result = momentum_model.get_accelerator_oscillator(bars["High"], bars["Low"])

    # The AC starts on period 34 + 5 - 2, the first full SMA of the Awesome Oscillator.
    assert result.first_valid_index() == 37
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-0.399343, -0.608252, -0.658767], abs=1e-6
    )
    recorder.capture(result.round(6))


def test_get_internal_bar_strength(recorder):
    bars = random_bars()
    result = momentum_model.get_internal_bar_strength(
        bars["High"], bars["Low"], bars["Close"]
    )

    assert result.iloc[-3:].tolist() == pytest.approx(
        [0.231665, 0.209847, 0.330882], abs=1e-6
    )
    recorder.capture(result.round(6))


def test_internal_bar_strength_without_a_range_is_missing():
    result = momentum_model.get_internal_bar_strength(
        pd.Series([11.0, 10.0]), pd.Series([9.0, 10.0]), pd.Series([10.0, 10.0])
    )

    assert result.iloc[0] == pytest.approx(0.5)
    assert np.isnan(result.iloc[1])


def test_get_intraday_momentum_index(recorder):
    bars = random_bars()
    result = momentum_model.get_intraday_momentum_index(bars["Open"], bars["Close"], 14)

    assert result.first_valid_index() == 13
    assert result.iloc[-3:].tolist() == pytest.approx(
        [42.155642, 47.153421, 46.821301], abs=1e-6
    )
    recorder.capture(result.round(6))


def test_get_qstick(recorder):
    bars = random_bars()
    result = momentum_model.get_qstick(bars["Open"], bars["Close"], 10)

    assert result.iloc[-3:].tolist() == pytest.approx(
        [-0.217948, -0.1948, -0.334352], abs=1e-6
    )
    recorder.capture(result.round(6))


def test_get_efficiency_ratio(recorder):
    result = momentum_model.get_efficiency_ratio(random_walk(), 10)

    assert result.first_valid_index() == 10
    assert result.iloc[-3:].tolist() == pytest.approx(
        [0.348424, 0.246874, 0.081147], abs=1e-6
    )
    recorder.capture(result.round(6))


def test_get_vertical_horizontal_filter(recorder):
    result = momentum_model.get_vertical_horizontal_filter(random_walk(), 28)

    assert result.first_valid_index() == 28
    assert result.iloc[-3:].tolist() == pytest.approx(
        [0.248857, 0.255715, 0.251329], abs=1e-6
    )
    recorder.capture(result.round(6))


def test_ratios_of_flat_prices_are_missing():
    """A window without a change has no Efficiency Ratio, VHF or IMI, where TA-Lib returns 1,
    0 and 50."""
    flat = pd.Series(10.0, index=range(30))

    assert momentum_model.get_efficiency_ratio(flat, 10).isna().all()
    assert momentum_model.get_vertical_horizontal_filter(flat, 10).isna().all()
    assert momentum_model.get_intraday_momentum_index(flat, flat, 10).isna().all()


def test_smoothed_chande_momentum_oscillator_matches_talib_and_the_rsi():
    closes = random_walk()
    result = momentum_model.get_chande_momentum_oscillator(closes, 14, smoothed=True)

    assert result.first_valid_index() == 14
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-9.806629, -12.179579, 4.32708], abs=1e-6
    )
    pd.testing.assert_series_equal(
        result,
        2 * momentum_model.get_relative_strength_index(closes, 14) - 100,
        check_names=False,
    )


def test_choppiness_index_with_the_true_range_box_matches_talib():
    bars = random_bars()
    result = momentum_model.get_choppiness_index(
        bars["High"], bars["Low"], bars["Close"], 14, range_method="true_range"
    )

    assert result.iloc[-3:].tolist() == pytest.approx(
        [66.120535, 64.503945, 61.158911], abs=1e-6
    )
    with pytest.raises(ValueError, match="range_method"):
        momentum_model.get_choppiness_index(
            bars["High"], bars["Low"], bars["Close"], 14, range_method="close"
        )


def test_slow_stochastic_oscillator_matches_talib():
    """TA-Lib's STOCH(14, 3, SMA, 3, SMA), whose slow %K is the fast %D."""
    bars = random_bars()
    fast = momentum_model.get_stochastic_oscillator(
        bars["High"], bars["Low"], bars["Close"], 14, 3
    )
    slow = momentum_model.get_stochastic_oscillator(
        bars["High"], bars["Low"], bars["Close"], 14, 3, slow=True
    )

    pd.testing.assert_series_equal(
        slow["Stochastic %K"], fast["Stochastic %D"], check_names=False
    )
    assert slow["Stochastic %K"].iloc[-3:].tolist() == pytest.approx(
        [35.375979, 21.845389, 13.05297], abs=1e-6
    )
    assert slow["Stochastic %D"].iloc[-3:].tolist() == pytest.approx(
        [37.172728, 30.870737, 23.424779], abs=1e-6
    )


def test_simple_percentage_volume_oscillator_matches_talib():
    volumes = sample_prices()["Volume"]
    result = momentum_model.get_percentage_volume_oscillator(
        volumes, 5, 10, 9, moving_average="simple"
    )

    # TA-Lib 0.8.3 PVO(5, 10, matype=0), the volume oscillator of pandas-ta-classic and
    # Tulip Indicators, in percent.
    assert result["Percentage Volume Oscillator"].iloc[-1] == pytest.approx(
        -0.023622047244094486, rel=1e-12
    )
    assert (
        result["Percentage Volume Oscillator"].first_valid_index() == volumes.index[9]
    )
    with pytest.raises(ValueError):
        momentum_model.get_percentage_volume_oscillator(
            volumes, 5, 10, 9, moving_average="hull"
        )


def test_fisher_transform_matches_talib():
    """TA-Lib's FISHER(10): the transform starts on period 9 and the trigger one later."""
    bars = random_bars()
    result = momentum_model.get_fisher_transform(bars["High"], bars["Low"], 10)

    assert result["Fisher Transform"].first_valid_index() == 9
    assert result["Trigger"].first_valid_index() == 10
    assert result["Fisher Transform"].iloc[-3:].tolist() == pytest.approx(
        [-0.813607, -1.305017, -1.777609], abs=1e-6
    )
    assert result["Trigger"].iloc[-3:].tolist() == pytest.approx(
        [-0.316505, -0.813607, -1.305017], abs=1e-6
    )


def test_fisher_transform_of_a_flat_range_decays_to_zero():
    flat = pd.Series(100.0, index=range(40))
    result = momentum_model.get_fisher_transform(flat, flat, 10)

    assert result["Fisher Transform"].iloc[9:].eq(0).all()


def test_inverse_fisher_transform_rsi_is_the_tanh_of_the_smoothed_rsi():
    """tanh(WMA(0.1 * (TA-Lib RSI(5) - 50), 9)), Ehlers' formula."""
    closes = random_bars()["Close"]
    result = momentum_model.get_inverse_fisher_transform_rsi(closes, 5, 9)

    assert result.first_valid_index() == 5 + 9 - 1
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-0.626315, -0.811348, -0.905228], abs=1e-6
    )
    assert result.abs().max() < 1


def test_kdj_indicator_matches_talib():
    """TA-Lib's KDJ(9, 3, RMA, 3, RMA), also pandas-ta-classic's kdj(9, 3)."""
    bars = random_bars()
    result = momentum_model.get_kdj_indicator(
        bars["High"], bars["Low"], bars["Close"], 9, 3, 3
    )

    assert result["K"].first_valid_index() == 10
    assert result["D"].first_valid_index() == 12
    for column, expected in {
        "K": [34.680925, 26.885087, 22.097447],
        "D": [45.269642, 39.141457, 33.46012],
        "J": [13.503491, 2.372346, -0.627899],
    }.items():
        assert result[column].iloc[-3:].tolist() == pytest.approx(expected, abs=1e-6)


def test_premier_stochastic_oscillator_matches_talib():
    bars = random_bars()
    result = momentum_model.get_premier_stochastic_oscillator(
        bars["High"], bars["Low"], bars["Close"], 8, 5
    )

    assert result.first_valid_index() == 8 + 2 * 5 - 3
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-0.2865, -0.525715, -0.694269], abs=1e-6
    )


def test_stochastic_momentum_index_matches_talib():
    """TA-Lib's SMI(13, 2, 25, 9), which starts both outputs on the first signal value."""
    bars = random_bars()
    result = momentum_model.get_stochastic_momentum_index(
        bars["High"], bars["Low"], bars["Close"], 13, 25, 2, 9
    )

    assert result["Stochastic Momentum Index"].first_valid_index() == 13 + 25 + 2 - 3
    assert result["Signal Line"].first_valid_index() == 13 + 25 + 2 - 3 + 8
    assert result["Stochastic Momentum Index"].iloc[-3:].tolist() == pytest.approx(
        [13.879485, 7.355389, 0.523929], abs=1e-6
    )
    assert result["Signal Line"].iloc[-3:].tolist() == pytest.approx(
        [22.502469, 19.473053, 15.683228], abs=1e-6
    )


def test_smi_ergodic_indicator_matches_pandas_ta_classic():
    """pandas-ta-classic's smi(5, 20, 5) times 100; the indicator is TA-Lib's TSI(20, 5)."""
    closes = random_bars()["Close"]
    result = momentum_model.get_smi_ergodic_indicator(closes, 20, 5, 5)

    for column, expected in {
        "SMI Ergodic Indicator": [-7.767662, -13.35796, -18.163198],
        "Signal Line": [0.847174, -3.887871, -8.646313],
        "Histogram": [-8.614836, -9.470089, -9.516885],
    }.items():
        assert result[column].iloc[-3:].tolist() == pytest.approx(expected, abs=1e-6)


def test_schaff_trend_cycle_matches_talib_once_the_seeds_decayed():
    """TA-Lib's STC(23, 50, 10), which seeds its short EMA later, after 300 bars."""
    closes = random_bars()["Close"]
    result = momentum_model.get_schaff_trend_cycle(closes, 23, 50, 10)

    # The MACD starts on period 49 and each stochastic needs a full window with a range.
    assert result.first_valid_index() >= 49 + 2 * 9
    assert result.iloc[-3:].tolist() == pytest.approx(
        [2.783986, 1.391993, 0.695996], abs=1e-5
    )
    assert result.dropna().between(0, 100).all()


def test_wavetrend_oscillator_matches_talib():
    bars = random_bars()
    result = momentum_model.get_wavetrend_oscillator(
        bars["High"], bars["Low"], bars["Close"], 10, 21, 4
    )

    assert result["WaveTrend"].first_valid_index() == 2 * 10 + 21 - 3
    assert result["WaveTrend"].iloc[-3:].tolist() == pytest.approx(
        [-3.599824, -15.647894, -24.506883], abs=1e-6
    )
    assert result["Signal Line"].iloc[-3:].tolist() == pytest.approx(
        [11.108466, 1.491716, -8.5128], abs=1e-6
    )


def test_derivative_oscillator_matches_talib():
    closes = random_bars()["Close"]
    result = momentum_model.get_derivative_oscillator(closes, 14, 5, 3, 9)

    assert result.first_valid_index() == 14 + 5 + 3 + 9 - 3
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-4.839607, -5.94887, -6.796227], abs=1e-6
    )


def test_swing_index_matches_talib():
    """TA-Lib's SI and ASI with a limit move of 3; the ASI is the running sum of the SI."""
    bars = random_bars()
    result = momentum_model.get_swing_index(
        bars["Open"], bars["High"], bars["Low"], bars["Close"], 3.0
    )

    assert result["Swing Index"].first_valid_index() == 1
    assert result["Swing Index"].iloc[-3:].tolist() == pytest.approx(
        [-29.299229, -3.897017, -5.982416], abs=1e-6
    )
    assert result["Accumulative Swing Index"].iloc[-3:].tolist() == pytest.approx(
        [95.945772, 92.048754, 86.066338], abs=1e-6
    )


def test_quantitative_qualitative_estimation_matches_pandas_ta_classic():
    closes = random_bars()["Close"]
    result = momentum_model.get_quantitative_qualitative_estimation(
        closes, 14, 5, 4.236
    )

    assert result["RSI Moving Average"].first_valid_index() == 14 + 5 - 1
    assert result["QQE"].first_valid_index() == 14 + 5 + 2 * 27 - 2
    assert result["QQE"].iloc[-3:].tolist() == pytest.approx(
        [54.682067, 52.31387, 50.218422], abs=1e-6
    )
    assert result["RSI Moving Average"].iloc[-3:].tolist() == pytest.approx(
        [48.86828, 46.494955, 44.379044], abs=1e-6
    )


def test_dynamic_momentum_index_sums_the_changes_of_its_own_period():
    """The summed RSI over Int(14 / volatility index) changes, bounded to 5 and 30."""
    closes = random_bars()["Close"]
    result = momentum_model.get_dynamic_momentum_index(closes, 14, 5, 10, 5, 30)

    assert result.iloc[-3:].tolist() == pytest.approx(
        [28.121299, 29.245696, 27.358302], abs=1e-6
    )
    assert result.dropna().between(0, 100).all()


def test_price_zone_oscillator_matches_ema_of_the_signed_closes():
    """100 * EMA(sign(change) * close, 14) / EMA(close, 14) with TA-Lib's EMA."""
    closes = random_bars()["Close"]
    result = momentum_model.get_price_zone_oscillator(closes, 14)

    assert result.first_valid_index() == 14
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-11.746324, -23.335871, -33.393339], abs=1e-6
    )


def test_connors_relative_strength_index_matches_talib():
    """TA-Lib's CRSI(3, 2, 100)."""
    closes = random_bars()["Close"]
    result = momentum_model.get_connors_relative_strength_index(closes, 3, 2, 100)

    assert result.first_valid_index() == 101
    assert result.iloc[-3:].tolist() == pytest.approx(
        [16.732125, 24.08288, 19.184298], abs=1e-6
    )


def test_connors_relative_strength_index_counts_the_streak():
    """Three rises give a streak of 3, a fall restarts it at -1 and no change resets it to 0."""
    closes = pd.Series([10.0, 11, 12, 13, 12, 12, 13, 14, 15, 16, 17, 18])
    streak = pd.Series([np.nan, 1, 2, 3, -1, 0, 1, 2, 3, 4, 5, 6])
    result = momentum_model.get_connors_relative_strength_index(closes, 2, 2, 3)

    # The last return, 18 / 17 - 1, is below the three before it: a percent rank of 0.
    expected = (
        momentum_model.get_relative_strength_index(closes, 2)
        + momentum_model.get_relative_strength_index(streak, 2)
    ) / 3

    assert result.iloc[-1] == pytest.approx(expected.iloc[-1])


def test_inertia_matches_pandas_ta_classic():
    """pandas-ta-classic's inertia(20, 14, mamode='rma'), which takes the standard deviation
    over the RVI window."""
    bars = random_bars()
    high, low, close = bars["High"], bars["Low"], bars["Close"]

    original = momentum_model.get_inertia(high, low, close, 20, 14, 14)
    refined = momentum_model.get_inertia(high, low, close, 20, 14, 14, "refined")

    assert original.first_valid_index() == 45
    assert original.iloc[-3:].tolist() == pytest.approx(
        [58.827239, 56.756035, 53.74869], abs=1e-6
    )
    assert refined.iloc[-3:].tolist() == pytest.approx(
        [65.164302, 64.431529, 63.433379], abs=1e-6
    )


def test_bias_matches_pandas_ta_classic():
    closes = random_bars()["Close"]
    result = momentum_model.get_bias(closes, 26)

    assert result.first_valid_index() == 25
    assert result.iloc[-3:].tolist() == pytest.approx(
        [-0.013261, -0.015268, -0.018741], abs=1e-6
    )


def test_brar_matches_pandas_ta_classic():
    bars = random_bars()
    result = momentum_model.get_brar(
        bars["Open"], bars["High"], bars["Low"], bars["Close"], 26
    )

    assert result["AR"].iloc[-3:].tolist() == pytest.approx(
        [97.36672, 109.338875, 109.656967], abs=1e-6
    )
    assert result["BR"].first_valid_index() == 26
    assert result["BR"].iloc[-3:].tolist() == pytest.approx(
        [89.784805, 99.756681, 98.112754], abs=1e-6
    )


def test_brar_is_missing_without_a_downward_move():
    """A window in which the price never fell below the open has no AR, not an infinite one."""
    opened = pd.Series([10.0, 10, 10, 10])
    result = momentum_model.get_brar(opened, opened + 1, opened, opened + 0.5, 2)

    assert result["AR"].isna().all()


def test_pretty_good_oscillator_matches_pandas_ta_classic():
    """pandas-ta-classic's pgo(14), once Wilder's ATR has forgotten its first range."""
    bars = random_bars()
    result = momentum_model.get_pretty_good_oscillator(
        bars["High"], bars["Low"], bars["Close"], 14
    )

    assert result.iloc[-3:].tolist() == pytest.approx(
        [-1.146025, -1.191006, -1.334609], abs=1e-6
    )


def test_psychological_line_matches_pandas_ta_classic():
    bars = random_bars()

    close = momentum_model.get_psychological_line(bars["Open"], bars["Close"], 12)
    opened = momentum_model.get_psychological_line(
        bars["Open"], bars["Close"], 12, "open"
    )

    assert close.first_valid_index() == 12
    assert close.iloc[-3:].tolist() == pytest.approx(
        [50.0, 41.666667, 33.333333], abs=1e-6
    )
    assert opened.first_valid_index() == 11
    assert opened.iloc[-3:].tolist() == pytest.approx(
        [58.333333, 50.0, 41.666667], abs=1e-6
    )


def test_volume_weighted_macd_matches_pandas_ta_classic():
    """pandas-ta-classic's vwmacd(12, 26, 9), which also weights the signal line by volume."""
    bars = random_bars()
    volumes = pd.Series(np.random.default_rng(5).uniform(1e5, 1e6, len(bars)))
    result = momentum_model.get_moving_average_convergence_divergence(
        bars["Close"],
        12,
        26,
        9,
        "volume_weighted",
        "volume_weighted",
        volumes=volumes,
    )

    for column, expected in {
        "MACD Line": [0.554314, 0.288027, -0.012207],
        "Signal Line": [0.674294, 0.63705, 0.604113],
        "Histogram": [-0.11998, -0.349023, -0.61632],
    }.items():
        assert result[column].iloc[-3:].tolist() == pytest.approx(expected, abs=1e-6)


def test_volume_weighted_macd_needs_the_volumes():
    with pytest.raises(ValueError, match="needs the volumes"):
        momentum_model.get_moving_average_convergence_divergence(
            random_bars()["Close"], 12, 26, 9, "elastic_volume_weighted"
        )


def test_volatility_based_momentum_divides_the_change_by_the_atr():
    """(Close - Close 12 periods ago) / pandas-ta-classic's atr(26), Wilder's ATR."""
    bars = random_bars()
    result = momentum_model.get_volatility_based_momentum(
        bars["High"], bars["Low"], bars["Close"], 12, 26
    )

    assert result.iloc[-3:].tolist() == pytest.approx(
        [-0.556143, -1.557564, -1.954465], abs=1e-5
    )


def test_archer_moving_average_trends_matches_pandas_ta_classic():
    closes = random_bars()["Close"]
    result = momentum_model.get_archer_moving_average_trends(closes, 8, 21, 2)

    assert result.first_valid_index() == 22
    assert result["Long Run"].iloc[-10:].tolist() == [1.0] * 3 + [0.0] * 7
    assert result["Short Run"].iloc[-10:].tolist() == [0.0] * 3 + [1.0] * 7


def test_ttm_trend_matches_pandas_ta_classic():
    bars = random_bars()
    result = momentum_model.get_ttm_trend(bars["High"], bars["Low"], bars["Close"], 6)

    assert result.first_valid_index() == 6
    assert result.iloc[-10:].tolist() == [1.0, 1.0] + [-1.0] * 8


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


def last_three(values: pd.Series) -> list[float]:
    return values.iloc[-3:].tolist()


def test_correlation_trend_indicator_matches_talib():
    """TA-Lib's CTI(20); a straight rise is +1 and flat prices have no correlation."""
    result = momentum_model.get_correlation_trend_indicator(
        random_prices()["Close"], 20
    )

    assert last_three(result) == pytest.approx(
        [-0.809991, -0.78511, -0.673336], abs=1e-6
    )
    assert momentum_model.get_correlation_trend_indicator(
        pd.Series(np.arange(10.0)), 5
    ).iloc[-1] == pytest.approx(1)
    assert np.isnan(
        momentum_model.get_correlation_trend_indicator(pd.Series([3.0] * 10), 5).iloc[
            -1
        ]
    )


def test_forecast_oscillator_matches_talib():
    """TA-Lib's FOSC(5), as a decimal rather than a percentage."""
    result = momentum_model.get_forecast_oscillator(random_prices()["Close"], 5)

    assert result.first_valid_index() == 5
    assert last_three(result * 100) == pytest.approx(
        [-1.45417, -1.272173, 1.713765], abs=1e-6
    )


def test_projection_oscillator_against_the_projected_bands():
    prices = random_prices()
    result = momentum_model.get_projection_oscillator(
        prices["High"], prices["Low"], prices["Close"], 14
    )

    # The bands of the last window, projected along the slopes of the highs and lows.
    window = prices.iloc[-14:]
    steps = np.arange(14)[::-1]
    upper = (window["High"] + steps * np.polyfit(range(14), window["High"], 1)[0]).max()
    lower = (window["Low"] + steps * np.polyfit(range(14), window["Low"], 1)[0]).min()

    assert result.iloc[-1] == pytest.approx(
        (prices["Close"].iloc[-1] - lower) / (upper - lower)
    )
    assert result.first_valid_index() == 13
    assert result.dropna().between(0, 1).all()


def test_dynamic_momentum_index_of_a_late_start_equals_its_own_history():
    """A series that starts later (leading NaN) sums only its own Period price changes."""
    closes = random_bars()["Close"]
    late_start = closes.copy()
    late_start.iloc[:40] = np.nan

    result = momentum_model.get_dynamic_momentum_index(late_start, 14, 5, 10, 5, 30)
    own_history = momentum_model.get_dynamic_momentum_index(
        closes.iloc[40:], 14, 5, 10, 5, 30
    )

    pd.testing.assert_series_equal(result.iloc[40:], own_history)
    assert result.iloc[:40].isna().all()


def test_dynamic_momentum_index_needs_a_full_window_of_changes_after_a_gap():
    """A window that spans missing prices holds fewer real changes than its Period."""
    closes = random_bars()["Close"]
    gapped = closes.copy()
    gapped.iloc[100:103] = np.nan

    result = momentum_model.get_dynamic_momentum_index(gapped, 14, 5, 10, 5, 30)
    changes = gapped.diff().notna().cumsum().to_numpy()
    standard_deviation = gapped.rolling(window=5).std()
    volatility_index = standard_deviation / standard_deviation.rolling(10).mean()
    periods = np.floor(14 / volatility_index).clip(5, 30)

    for position in range(103, 140):
        period = periods.iloc[position]
        if np.isnan(period) or position < period:
            continue
        real_changes = changes[position] - changes[position - int(period)]
        if real_changes < period:
            assert np.isnan(result.iloc[position])


def test_trend_detection_index_by_hand():
    """With a window of 2: Momentum = 2, 0, 1, 3, 0, ... from period 2, so on period 5 the
    Direction Indicator is 1 + 3 = 4 and the TDI |4| - ((2 + 0 + 1 + 3) - (1 + 3)) = 2.
    """
    prices = pd.Series([10.0, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17])
    result = momentum_model.get_trend_detection_index(prices, 2, 2)

    assert result["Direction Indicator"].first_valid_index() == 3
    assert result["TDI"].first_valid_index() == 5
    assert result["Direction Indicator"].iloc[5] == pytest.approx(4)
    assert result["TDI"].iloc[5] == pytest.approx(2)
    # Period 6: the Direction Indicator is 3 + 0 = 3 and the TDI |3| - ((0 + 1 + 3 + 0) - 3).
    assert result["Direction Indicator"].iloc[6] == pytest.approx(3)
    assert result["TDI"].iloc[6] == pytest.approx(2)


def test_get_trend_detection_index(recorder):
    result = momentum_model.get_trend_detection_index(random_walk(), 10, 2)

    assert result["TDI"].first_valid_index() == 29
    assert list(result.columns) == ["TDI", "Direction Indicator"]
    recorder.capture(result.round(6))


def test_dv_intermediate_oscillator_by_hand():
    """The ranks against a direct count of lower values in the previous window."""
    rng = np.random.default_rng(3)
    prices = pd.Series(100 * np.exp(rng.normal(0, 0.01, 200).cumsum()))
    result = momentum_model.get_dv_intermediate_oscillator(prices, 20, 0.8)

    ratio = prices / prices.rolling(3).mean() - 1
    magnitude = (
        ((ratio.rolling(5).mean() + ratio.rolling(100).mean() / 10) / 2)
        .rolling(5)
        .mean()
    )
    direction = np.sign(prices.diff()).replace(0, -1)
    stretch = (
        ((direction.rolling(10).sum() + direction.rolling(100).sum() / 10) / 2)
        .rolling(2)
        .mean()
    )

    def rank(values: pd.Series, position: int) -> float:
        window = values.iloc[position - 20 : position]
        return float((window < values.iloc[position] - 1e-9).sum() / 20)

    # The magnitude starts on period 105 and the stretch on period 101, so the ranks of a
    # 20-period window on periods 125 and 121.
    assert result["DVI Magnitude"].first_valid_index() == 125
    assert result["DVI Stretch"].first_valid_index() == 121
    for position in (125, 150, 199):
        assert result["DVI Magnitude"].iloc[position] == pytest.approx(
            rank(magnitude, position)
        )
        assert result["DVI Stretch"].iloc[position] == pytest.approx(
            rank(stretch, position)
        )
        assert result["DVI"].iloc[position] == pytest.approx(
            0.8 * rank(magnitude, position) + 0.2 * rank(stretch, position)
        )
    assert result.min().min() >= 0
    assert result.max().max() <= 1


def test_get_dv_intermediate_oscillator(recorder):
    rng = np.random.default_rng(5)
    prices = pd.Series(100 * np.exp(rng.normal(0, 0.01, 200).cumsum()))
    result = momentum_model.get_dv_intermediate_oscillator(prices, 50)

    assert result["DVI"].first_valid_index() == 155
    assert list(result.columns) == ["DVI Magnitude", "DVI Stretch", "DVI"]
    recorder.capture(result.round(6))


def test_dv_intermediate_oscillator_rejects_a_weight_outside_zero_and_one():
    with pytest.raises(ValueError, match="magnitude_weight"):
        momentum_model.get_dv_intermediate_oscillator(random_walk(), 20, 1.5)
