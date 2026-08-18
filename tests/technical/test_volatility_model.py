"""Volatility Model Tests"""

import pandas as pd

from financetoolkit.technicals import volatility_model

# pylint: disable=missing-function-docstring


def test_get_true_range(recorder):
    recorder.capture(
        volatility_model.get_true_range(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
        ).round(3)
    )


def test_get_average_true_range(recorder):
    recorder.capture(
        volatility_model.get_average_true_range(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        ).round(3)
    )


def test_get_keltner_channels(recorder):
    recorder.capture(
        volatility_model.get_keltner_channels(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            10,
            5,
            2,
        ).round(3)
    )


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
