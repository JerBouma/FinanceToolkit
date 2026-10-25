"""Zmijewski Model Tests""" ""
import pandas as pd

from financetoolkit.models import zmijewski_model

# pylint: disable=missing-function-docstring


def test_get_net_income_to_total_assets_ratio(recorder):
    recorder.capture(
        zmijewski_model.get_net_income_to_total_assets_ratio(
            net_income=pd.Series([100, -110, 120, 130, -80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_total_liabilities_to_total_assets_ratio(recorder):
    recorder.capture(
        zmijewski_model.get_total_liabilities_to_total_assets_ratio(
            total_liabilities=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_current_assets_to_current_liabilities_ratio(recorder):
    recorder.capture(
        zmijewski_model.get_current_assets_to_current_liabilities_ratio(
            current_assets=pd.Series([300, 320, 340, 360, 380]),
            current_liabilities=pd.Series([200, 210, 220, 230, 240]),
        )
    )


def test_get_zmijewski_score(recorder):
    recorder.capture(
        zmijewski_model.get_zmijewski_score(
            net_income_to_total_assets_ratio=pd.Series([0.1, 0.05, 0.0, -0.05, -0.1]),
            total_liabilities_to_total_assets_ratio=pd.Series(
                [0.5, 0.6, 0.7, 0.8, 0.9]
            ),
            current_assets_to_current_liabilities_ratio=pd.Series(
                [1.5, 1.2, 1.0, 0.8, 0.5]
            ),
        )
    )


def test_get_zmijewski_bankruptcy_probability(recorder):
    recorder.capture(
        zmijewski_model.get_zmijewski_bankruptcy_probability(
            zmijewski_score=pd.Series([-3.0, -1.0, 0.0, 1.0, 3.0]),
        )
    )


def test_zmijewski_score_uses_the_full_precision_coefficients():
    # Zmijewski (1984): -4.336 - 4.513 NI/TA + 5.679 TL/TA - 0.004 CA/CL
    score = zmijewski_model.get_zmijewski_score(
        net_income_to_total_assets_ratio=0.1,
        total_liabilities_to_total_assets_ratio=0.5,
        current_assets_to_current_liabilities_ratio=2.0,
    )

    assert abs(score - (-4.336 - 0.4513 + 2.8395 - 0.008)) < 1e-12
