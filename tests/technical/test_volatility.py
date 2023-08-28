"""Volatility Tests"""
import pandas as pd

from financetoolkit.technical import volatility

# pylint: disable=missing-function-docstring


def test_get_true_range(recorder):
    recorder.capture(
        volatility.get_true_range(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([100, 110, 120, 130, 80]),
        )
    )


def test_get_average_true_range(recorder):
    recorder.capture(
        volatility.get_average_true_range(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            10,
        )
    )


def test_get_keltner_channels(recorder):
    recorder.capture(
        volatility.get_keltner_channels(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            10,
            5,
            2,
        )
    )


def test_get_bollinger_bands(recorder):
    recorder.capture(
        volatility.get_bollinger_bands(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10, 2
        )
    )
