"""Momentum Tests"""
import pandas as pd

from financetoolkit.technical import momentum

# pylint: disable=missing-function-docstring


def test_get_money_flow_index(recorder):
    recorder.capture(
        momentum.get_money_flow_index(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            10,
        )
    )


def test_get_williams_percent_r(recorder):
    recorder.capture(
        momentum.get_williams_percent_r(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        )
    )


def test_get_aroon_indicator(recorder):
    recorder.capture(
        momentum.get_aroon_indicator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            10,
        )
    )


def test_get_commodity_channel_index(recorder):
    recorder.capture(
        momentum.get_commodity_channel_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        )
    )


def test_get_relative_vigor_index(recorder):
    recorder.capture(
        momentum.get_relative_vigor_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            20,
        )
    )


def test_get_force_index(recorder):
    recorder.capture(
        momentum.get_force_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            20,
        )
    )


def test_get_ultimate_oscillator(recorder):
    recorder.capture(
        momentum.get_ultimate_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            20,
            30,
            40,
        )
    )


def test_get_detrended_price_oscillator(recorder):
    recorder.capture(
        momentum.get_detrended_price_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 30, "sma"
        )
    )


def test_get_average_directional_index(recorder):
    recorder.capture(
        momentum.get_average_directional_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        )
    )


def test_get_chande_momentum_oscillator(recorder):
    recorder.capture(
        momentum.get_chande_momentum_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10
        )
    )


def test_get_ichimoku_cloud(recorder):
    recorder.capture(
        momentum.get_ichimoku_cloud(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
            5,
            3,
        )
    )


def test_get_stochastic_oscillator(recorder):
    recorder.capture(
        momentum.get_stochastic_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
            5,
        )
    )


def test_get_moving_average_convergence_divergence(recorder):
    recorder.capture(
        momentum.get_moving_average_convergence_divergence(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            5,
            10,
            5,
        )
    )


def test_get_relative_strength_index(recorder):
    recorder.capture(
        momentum.get_relative_strength_index(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            5,
        )
    )


def test_get_balance_of_power(recorder):
    recorder.capture(
        momentum.get_balance_of_power(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
        )
    )
