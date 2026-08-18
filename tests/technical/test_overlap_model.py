"""Overlap Model Tests"""

import pandas as pd

from financetoolkit.technicals import overlap_model

# pylint: disable=missing-function-docstring


def test_get_moving_average(recorder):
    recorder.capture(
        overlap_model.get_moving_average(pd.Series([100, 110, 120, 130, 80]), 10).round(
            3
        )
    )


def test_get_exponential_moving_average(recorder):
    recorder.capture(
        overlap_model.get_exponential_moving_average(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10
        ).round(2)
    )


def test_get_double_exponential_moving_average(recorder):
    recorder.capture(
        overlap_model.get_double_exponential_moving_average(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10
        ).round(2)
    )


def test_get_trix(recorder):
    recorder.capture(
        overlap_model.get_trix(pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 10).round(3)
    )


def test_get_triangular_moving_average(recorder):
    recorder.capture(
        overlap_model.get_triangular_moving_average(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]), 20
        ).round(3)
    )


def test_get_fibonacci_retracement_levels(recorder):
    recorder.capture(
        overlap_model.get_fibonacci_retracement_levels(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([90, 95, 100, 110, 60]),
        ).round(3)
    )


def test_get_fibonacci_retracement_levels_downtrend(recorder):
    recorder.capture(
        overlap_model.get_fibonacci_retracement_levels(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([90, 95, 100, 110, 60]),
            trend="downtrend",
        ).round(3)
    )


def test_get_fibonacci_retracement_levels_custom_levels(recorder):
    recorder.capture(
        overlap_model.get_fibonacci_retracement_levels(
            pd.Series([100, 110, 120, 130, 80]),
            pd.Series([90, 95, 100, 110, 60]),
            levels=[0.0, 0.5, 1.0],
        ).round(3)
    )


def test_get_kaufman_adaptive_moving_average(recorder):
    recorder.capture(
        overlap_model.get_kaufman_adaptive_moving_average(
            pd.Series(
                [100, 102, 101, 103, 106, 105, 107, 110, 108, 111, 115, 113, 116]
            ),
            5,
            2,
            30,
        ).round(3)
    )


def test_get_kaufman_adaptive_moving_average_default_parameters(recorder):
    recorder.capture(
        overlap_model.get_kaufman_adaptive_moving_average(
            pd.Series(
                [
                    100,
                    102,
                    101,
                    103,
                    106,
                    105,
                    107,
                    110,
                    108,
                    111,
                    115,
                    113,
                    116,
                    120,
                    118,
                ]
            )
        ).round(3)
    )
