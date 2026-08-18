"""Enterprise Model Tests""" ""
import pandas as pd
import pytest

from financetoolkit.models import enterprise_model

# pylint: disable=missing-function-docstring


def test_get_enterprise_value_breakdown(recorder):
    recorder.capture(
        enterprise_model.get_enterprise_value_breakdown(
            share_price=pd.Series([100, 110, 120, 130, 80]),
            shares_outstanding=pd.Series([200, 150, 130, 200, 110]),
            total_debt=pd.Series([500, 400, 300, 200, 100]),
            minority_interest=pd.Series([500, 430, 340, 240, 140]),
            preferred_equity=pd.Series([400, 300, 200, 150, 80]),
            cash_and_cash_equivalents=pd.Series([430, 340, 240, 150, 80]),
        )
    )


def test_get_tobins_q_ratio(recorder):
    recorder.capture(
        enterprise_model.get_tobins_q_ratio(
            market_value_of_equity=pd.Series([5000, 300, 6000, 250]),
            total_liabilities=pd.Series([1000, 800, 1200, 900]),
            total_assets=pd.Series([2000, 2000, 2500, 2000]),
        )
    )
    # Synthetic ground truth: one firm well above book cost, one below it.
    assert (
        enterprise_model.get_tobins_q_ratio(
            market_value_of_equity=5000,
            total_liabilities=1000,
            total_assets=2000,
        )
        == 3.0
    )
    assert (
        enterprise_model.get_tobins_q_ratio(
            market_value_of_equity=300,
            total_liabilities=800,
            total_assets=2000,
        )
        == 0.55
    )


def test_get_tobins_q_ratio_type_error():
    with pytest.raises(TypeError):
        enterprise_model.get_tobins_q_ratio(
            market_value_of_equity="not a number",
            total_liabilities=1000,
            total_assets=2000,
        )
