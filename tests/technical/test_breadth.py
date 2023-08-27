"""Breadth Tests"""
import pandas as pd

from financetoolkit.technical import breadth

# pylint: disable=missing-function-docstring


def test_get_mcclellan_oscillator(recorder):
    recorder.capture(
        breadth.get_mcclellan_oscillator(pd.Series([100, 110, 120, 130, 80]), 10, 20)
    )


def test_get_advancers_decliners(recorder):
    recorder.capture(
        breadth.get_advancers_decliners(pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]))
    )


def test_get_on_balance_volume(recorder):
    recorder.capture(
        breadth.get_on_balance_volume(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
        )
    )


def test_get_accumulation_distribution_line(recorder):
    recorder.capture(
        breadth.get_accumulation_distribution_line(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.1, 103, 0.05, 0.01, -0.02]),
        )
    )


def test_get_chaikin_oscillator(recorder):
    recorder.capture(
        breadth.get_chaikin_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            10,
            20,
        )
    )
