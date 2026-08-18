"""Breadth Model Tests"""

import pandas as pd

from financetoolkit.technicals import breadth_model

# pylint: disable=missing-function-docstring


def test_get_mcclellan_oscillator(recorder):
    recorder.capture(
        breadth_model.get_mcclellan_oscillator(
            pd.Series([100, 110, 120, 130, 80]), 10, 20
        ).round(2)
    )


def test_get_advancers_decliners(recorder):
    recorder.capture(
        breadth_model.get_advancers_decliners(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02])
        )
    )


def test_get_on_balance_volume(recorder):
    recorder.capture(
        breadth_model.get_on_balance_volume(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
        )
    )


def test_get_accumulation_distribution_line(recorder):
    recorder.capture(
        breadth_model.get_accumulation_distribution_line(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.1, 103, 0.05, 0.01, -0.02]),
        )
    )


def test_get_chaikin_oscillator(recorder):
    recorder.capture(
        breadth_model.get_chaikin_oscillator(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            pd.Series([0.005, -0.02, 0.06, 0.005, 0.0]),
            10,
            20,
        )
    )


def test_get_chaikin_money_flow(recorder):
    recorder.capture(
        breadth_model.get_chaikin_money_flow(
            pd.Series([10, 11, 12, 11, 13, 14, 13, 15]),
            pd.Series([9, 10, 10, 9, 11, 12, 11, 13]),
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
            4,
        ).round(3)
    )


def test_get_ease_of_movement(recorder):
    recorder.capture(
        breadth_model.get_ease_of_movement(
            pd.Series([10, 11, 12, 11, 13, 14, 13, 15]),
            pd.Series([9, 10, 10, 9, 11, 12, 11, 13]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
            3,
        ).round(3)
    )


def test_get_negative_volume_index(recorder):
    recorder.capture(
        breadth_model.get_negative_volume_index(
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
        ).round(3)
    )


def test_get_positive_volume_index(recorder):
    recorder.capture(
        breadth_model.get_positive_volume_index(
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
        ).round(3)
    )
