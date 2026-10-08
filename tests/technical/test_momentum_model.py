"""Momentum Model Tests"""

import pandas as pd

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
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            20,
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
