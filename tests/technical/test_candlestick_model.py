"""Candlestick Model Tests"""

import pandas as pd

from financetoolkit.technicals import candlestick_model

# pylint: disable=missing-function-docstring


def test_get_doji(recorder):
    recorder.capture(
        candlestick_model.get_doji(
            pd.Series([100.0, 100.0, 100.0]),
            pd.Series([102.0, 102.0, 105.0]),
            pd.Series([98.0, 98.0, 95.0]),
            pd.Series([100.05, 101.5, 100.0]),
        )
    )


def test_get_bullish_engulfing(recorder):
    recorder.capture(
        candlestick_model.get_bullish_engulfing(
            pd.Series([110.0, 99.0]),
            pd.Series([111.0, 113.0]),
            pd.Series([99.0, 98.0]),
            pd.Series([100.0, 112.0]),
        )
    )


def test_get_bearish_engulfing(recorder):
    recorder.capture(
        candlestick_model.get_bearish_engulfing(
            pd.Series([100.0, 112.0]),
            pd.Series([111.0, 113.0]),
            pd.Series([99.0, 98.0]),
            pd.Series([110.0, 99.0]),
        )
    )


def test_get_hammer(recorder):
    recorder.capture(
        candlestick_model.get_hammer(
            pd.Series([100.0]),
            pd.Series([101.2]),
            pd.Series([95.0]),
            pd.Series([101.0]),
        )
    )


def test_get_hammer_not_a_hammer(recorder):
    recorder.capture(
        candlestick_model.get_hammer(
            pd.Series([100.0]),
            pd.Series([106.0]),
            pd.Series([99.5]),
            pd.Series([101.0]),
        )
    )
