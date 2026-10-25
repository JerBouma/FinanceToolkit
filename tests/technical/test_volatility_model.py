"""Volatility Model Tests"""

import pandas as pd
import pytest

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
