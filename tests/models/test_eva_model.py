"""EVA Model Tests""" ""
import pandas as pd
import pytest

from financetoolkit.models import eva_model

# pylint: disable=missing-function-docstring


def test_get_net_operating_profit_after_taxes(recorder):
    recorder.capture(
        eva_model.get_net_operating_profit_after_taxes(
            ebit=pd.Series([100, 110, 120, 130, 80]),
            effective_tax_rate=pd.Series([0.2, 0.21, 0.19, 0.22, 0.18]),
        )
    )


def test_get_invested_capital(recorder):
    recorder.capture(
        eva_model.get_invested_capital(
            total_equity=pd.Series([500, 520, 540, 560, 580]),
            total_debt=pd.Series([200, 210, 220, 230, 240]),
        )
    )


def test_get_economic_value_added(recorder):
    recorder.capture(
        eva_model.get_economic_value_added(
            net_operating_profit_after_taxes=pd.Series([80, 87, 97, 101, 66]),
            weighted_average_cost_of_capital=pd.Series([0.08, 0.09, 0.085, 0.095, 0.1]),
            invested_capital=pd.Series([700, 730, 760, 790, 820]),
        )
    )


def test_get_market_value_added(recorder):
    recorder.capture(
        eva_model.get_market_value_added(
            market_value_of_equity=pd.Series([5000, 5200, 4800, 5300, 5600]),
            market_value_of_debt=pd.Series([1000, 1050, 1100, 1150, 1200]),
            invested_capital=pd.Series([3000, 3100, 3200, 3300, 3400]),
        )
    )
    # Synthetic ground-truth: MVA = (5000 + 1000) - 3000 = 3000
    assert (
        eva_model.get_market_value_added(
            market_value_of_equity=5000,
            market_value_of_debt=1000,
            invested_capital=3000,
        )
        == 3000
    )


def test_get_market_value_added_type_error():
    with pytest.raises(TypeError):
        eva_model.get_market_value_added(
            market_value_of_equity="not a number",
            market_value_of_debt=1000,
            invested_capital=3000,
        )
