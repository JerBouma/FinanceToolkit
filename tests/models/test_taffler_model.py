"""Taffler Model Tests"""

import pandas as pd

from financetoolkit.models import taffler_model


def test_get_profit_before_tax_to_current_liabilities_ratio(recorder):
    recorder.capture(
        taffler_model.get_profit_before_tax_to_current_liabilities_ratio(
            profit_before_tax=pd.Series([100, -50, 120, 130, 80]),
            current_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_current_assets_to_total_liabilities_ratio(recorder):
    recorder.capture(
        taffler_model.get_current_assets_to_total_liabilities_ratio(
            current_assets=pd.Series([100, 110, 120, 130, 80]),
            total_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_current_liabilities_to_total_assets_ratio(recorder):
    recorder.capture(
        taffler_model.get_current_liabilities_to_total_assets_ratio(
            current_liabilities=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_no_credit_interval():
    # (400 - 300) / ((1,000 - 100 - 50) / 365) = 100 / (850 / 365) days.
    interval = taffler_model.get_no_credit_interval(
        quick_assets=400,
        current_liabilities=300,
        sales=1_000,
        profit_before_tax=100,
        depreciation=50,
    )
    assert abs(interval - 100 / (850 / 365)) < 1e-12

    # A quarter of flows covers a quarter of the days, which gives the same interval.
    quarterly = taffler_model.get_no_credit_interval(
        quick_assets=400,
        current_liabilities=300,
        sales=250,
        profit_before_tax=25,
        depreciation=12.5,
        days=365 / 4,
    )
    assert abs(quarterly - interval) < 1e-12


def test_get_taffler_z_score(recorder):
    score = taffler_model.get_taffler_z_score(
        profit_before_tax_to_current_liabilities_ratio=pd.Series([0.2, -0.1]),
        current_assets_to_total_liabilities_ratio=pd.Series([0.9, 0.4]),
        current_liabilities_to_total_assets_ratio=pd.Series([0.3, 0.6]),
        no_credit_interval=pd.Series([10.0, -60.0]),
    )
    recorder.capture(score)

    expected = 3.20 + 12.18 * 0.2 + 2.50 * 0.9 - 10.68 * 0.3 + 0.029 * 10.0
    assert abs(score.iloc[0] - expected) < 1e-12
    # A loss, thin current assets and a negative no-credit interval put it at risk.
    assert score.iloc[1] < 0
