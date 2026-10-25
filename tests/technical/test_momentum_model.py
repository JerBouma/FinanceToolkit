"""Momentum Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals import momentum_model

# pylint: disable=missing-function-docstring


def test_get_money_flow_index(recorder):
    recorder.capture(
        momentum_model.get_money_flow_index(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            10,
        ).round(3)
    )


def test_get_williams_percent_r(recorder):
    recorder.capture(
        momentum_model.get_williams_percent_r(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        ).round(3)
    )


def test_get_aroon_indicator(recorder):
    recorder.capture(
        momentum_model.get_aroon_indicator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            10,
        ).round(3)
    )


def test_get_commodity_channel_index(recorder):
    recorder.capture(
        momentum_model.get_commodity_channel_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        ).round(3)
    )


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
    recorder.capture(
        momentum_model.get_force_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            20,
        ).round(3)
    )


def test_get_ultimate_oscillator(recorder):
    recorder.capture(
        momentum_model.get_ultimate_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            20,
            30,
            40,
        ).round(3)
    )


def test_get_detrended_price_oscillator(recorder):
    recorder.capture(
        momentum_model.get_detrended_price_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 30, "sma"
        ).round(3)
    )


def test_get_average_directional_index(recorder):
    recorder.capture(
        momentum_model.get_average_directional_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        ).round(3)
    )


def test_get_chande_momentum_oscillator(recorder):
    recorder.capture(
        momentum_model.get_chande_momentum_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10
        ).round(3)
    )


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
    recorder.capture(
        momentum_model.get_stochastic_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
            5,
        ).round(3)
    )


def test_get_moving_average_convergence_divergence(recorder):
    recorder.capture(
        momentum_model.get_moving_average_convergence_divergence(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            5,
            10,
            5,
        ).round(3)
    )


def test_get_relative_strength_index(recorder):
    recorder.capture(
        momentum_model.get_relative_strength_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            5,
        ).round(3)
    )


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
    recorder.capture(
        momentum_model.get_know_sure_thing(
            pd.Series(
                [
                    100,
                    102,
                    101,
                    105,
                    110,
                    108,
                    112,
                    115,
                    111,
                    120,
                    118,
                    123,
                    127,
                    130,
                    128,
                    132,
                    135,
                    140,
                    138,
                    145,
                    150,
                    148,
                    152,
                    155,
                    160,
                    158,
                    162,
                    165,
                    170,
                    168,
                    172,
                    175,
                    180,
                    178,
                    182,
                    185,
                ]
            )
        ).round(3)
    )


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
