"""Grover Model Tests""" ""
import pandas as pd
import pytest

from financetoolkit.models import grover_model

# pylint: disable=missing-function-docstring


def test_get_working_capital_to_total_assets_ratio(recorder):
    recorder.capture(
        grover_model.get_working_capital_to_total_assets_ratio(
            working_capital=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_ebit_to_total_assets_ratio(recorder):
    recorder.capture(
        grover_model.get_ebit_to_total_assets_ratio(
            ebit=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_return_on_assets_ratio(recorder):
    recorder.capture(
        grover_model.get_return_on_assets_ratio(
            net_income=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_grover_score(recorder):
    recorder.capture(
        grover_model.get_grover_score(
            working_capital_to_total_assets_ratio=pd.Series([0.2, -0.05]),
            ebit_to_total_assets_ratio=pd.Series([0.15, -0.03]),
            return_on_assets_ratio=pd.Series([0.1, -0.05]),
        )
    )
    # Synthetic ground truth: healthy is well above the -0.02 cutoff, distressed below.
    healthy = grover_model.get_grover_score(
        working_capital_to_total_assets_ratio=0.2,
        ebit_to_total_assets_ratio=0.15,
        return_on_assets_ratio=0.1,
    )
    distressed = grover_model.get_grover_score(
        working_capital_to_total_assets_ratio=-0.05,
        ebit_to_total_assets_ratio=-0.03,
        return_on_assets_ratio=-0.05,
    )
    assert round(healthy, 5) == 0.896
    assert healthy > -0.02
    assert round(distressed, 5) == -0.12682
    assert distressed <= -0.02


def test_get_grover_score_type_error():
    with pytest.raises(TypeError):
        grover_model.get_grover_score(
            working_capital_to_total_assets_ratio="not a number",
            ebit_to_total_assets_ratio=0.15,
            return_on_assets_ratio=0.1,
        )
