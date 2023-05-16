"""Price Tests""" ""
import pandas as pd

from financialtoolkit.historical import price

# pylint: disable=missing-function-docstring


def test_get_returns(recorder):
    recorder.capture(price.get_returns(pd.Series([100, 110, 120, 130, 80])))


def test_get_volatility(recorder):
    recorder.capture(price.get_volatility(pd.Series([0.01, -0.03, 0.05, 0.01, -0.02])))


def test_get_sharpe_ratio(recorder):
    recorder.capture(
        price.get_sharpe_ratio(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]) / 100,
        )
    )
    recorder.capture(
        price.get_sharpe_ratio(pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 0.001)
    )


def test_get_sortino_ratio(recorder):
    recorder.capture(
        price.get_sortino_ratio(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]) / 100,
        )
    )
    recorder.capture(
        price.get_sortino_ratio(pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 0.001)
    )


def test_get_beta(recorder):
    recorder.capture(
        price.get_beta(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
        )
    )
