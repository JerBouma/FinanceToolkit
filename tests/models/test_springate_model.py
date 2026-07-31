"""Springate Model Tests""" ""
import pandas as pd
import pytest

from financetoolkit.models import springate_model

# pylint: disable=missing-function-docstring


def test_get_working_capital_to_total_assets_ratio(recorder):
    recorder.capture(
        springate_model.get_working_capital_to_total_assets_ratio(
            working_capital=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_ebit_to_total_assets_ratio(recorder):
    recorder.capture(
        springate_model.get_ebit_to_total_assets_ratio(
            ebit=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_ebt_to_current_liabilities_ratio(recorder):
    recorder.capture(
        springate_model.get_ebt_to_current_liabilities_ratio(
            ebt=pd.Series([100, 110, 120, 130, 80]),
            current_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_sales_to_total_assets_ratio(recorder):
    recorder.capture(
        springate_model.get_sales_to_total_assets_ratio(
            sales=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_springate_score(recorder):
    recorder.capture(
        springate_model.get_springate_score(
            working_capital_to_total_assets_ratio=pd.Series([0.2, -0.05]),
            ebit_to_total_assets_ratio=pd.Series([0.15, -0.03]),
            ebt_to_current_liabilities_ratio=pd.Series([0.4, -0.08]),
            sales_to_total_assets_ratio=pd.Series([1.2, 0.4]),
        )
    )
    # Synthetic ground-truth: a healthy company (well above the 0.862 cutoff)
    # and a distressed company (below it, even negative).
    healthy = springate_model.get_springate_score(
        working_capital_to_total_assets_ratio=0.2,
        ebit_to_total_assets_ratio=0.15,
        ebt_to_current_liabilities_ratio=0.4,
        sales_to_total_assets_ratio=1.2,
    )
    distressed = springate_model.get_springate_score(
        working_capital_to_total_assets_ratio=-0.05,
        ebit_to_total_assets_ratio=-0.03,
        ebt_to_current_liabilities_ratio=-0.08,
        sales_to_total_assets_ratio=0.4,
    )
    assert round(healthy, 4) == 1.4105
    assert healthy > 0.862
    assert round(distressed, 4) == -0.0364
    assert distressed < 0.862


def test_get_springate_score_type_error():
    with pytest.raises(TypeError):
        springate_model.get_springate_score(
            working_capital_to_total_assets_ratio="not a number",
            ebit_to_total_assets_ratio=0.15,
            ebt_to_current_liabilities_ratio=0.4,
            sales_to_total_assets_ratio=1.2,
        )
