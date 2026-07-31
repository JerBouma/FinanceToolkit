"""Intrinsic Model Tests""" ""

import pandas as pd
import pytest

from financetoolkit.models import intrinsic_model

# pylint: disable=missing-function-docstring


def test_get_intrinsic_value(recorder):
    recorder.capture(
        intrinsic_model.get_intrinsic_value(
            cash_flow=500,
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.1,
            cash_and_cash_equivalents=100,
            total_debt=100,
            shares_outstanding=100,
            periods=5,
        )
    )


def test_get_free_cash_flow_to_firm(recorder):
    recorder.capture(
        intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes=pd.Series([750, 800, 820, 780, 900]),
            depreciation_and_amortization=pd.Series([100, 105, 110, 108, 115]),
            capital_expenditure=pd.Series([200, 210, 220, 190, 230]),
            change_in_net_working_capital=pd.Series([50, -20, 30, 10, -5]),
        )
    )
    # Synthetic ground-truth: FCFF = 750 + 100 - 200 - 50 = 600
    assert (
        intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes=750,
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
        )
        == 600
    )


def test_get_free_cash_flow_to_firm_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes="not a number",
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
        )


def test_get_free_cash_flow_to_equity(recorder):
    recorder.capture(
        intrinsic_model.get_free_cash_flow_to_equity(
            net_income=pd.Series([800, 850, 870, 830, 950]),
            depreciation_and_amortization=pd.Series([100, 105, 110, 108, 115]),
            capital_expenditure=pd.Series([200, 210, 220, 190, 230]),
            change_in_net_working_capital=pd.Series([50, -20, 30, 10, -5]),
            net_borrowing=pd.Series([30, -10, 15, 5, -25]),
        )
    )
    # Synthetic ground-truth: FCFE = 800 + 100 - 200 - 50 + 30 = 680
    assert (
        intrinsic_model.get_free_cash_flow_to_equity(
            net_income=800,
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
            net_borrowing=30,
        )
        == 680
    )


def test_get_free_cash_flow_to_equity_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_free_cash_flow_to_equity(
            net_income="not a number",
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
            net_borrowing=30,
        )


def test_get_two_stage_dividend_discount_model(recorder):
    recorder.capture(
        intrinsic_model.get_two_stage_dividend_discount_model(
            dividends_per_share=2.0,
            rate_of_return=0.10,
            high_growth_rate=0.20,
            stable_growth_rate=0.03,
            high_growth_periods=3,
        )
    )
    # Synthetic ground-truth computed by hand (see task verification):
    # dividends grow 2.0 -> 2.4 -> 2.88 -> 3.456, discounted at 10% gives a
    # high-growth phase PV of ~7.1585, a terminal value of ~50.8526 (PV ~38.2063),
    # for a total intrinsic value of ~45.3648.
    result = intrinsic_model.get_two_stage_dividend_discount_model(
        dividends_per_share=2.0,
        rate_of_return=0.10,
        high_growth_rate=0.20,
        stable_growth_rate=0.03,
        high_growth_periods=3,
    )
    intrinsic_value = result.loc["Intrinsic Value"].iloc[0]
    assert round(intrinsic_value, 4) == 45.3648


def test_get_two_stage_dividend_discount_model_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_two_stage_dividend_discount_model(
            dividends_per_share="not a number",
            rate_of_return=0.10,
            high_growth_rate=0.20,
            stable_growth_rate=0.03,
        )


def test_get_residual_income(recorder):
    recorder.capture(
        intrinsic_model.get_residual_income(
            net_income=pd.Series([1000, 1050, 1100, 950, 1200]),
            cost_of_equity=pd.Series([0.10, 0.11, 0.095, 0.12, 0.09]),
            book_value_of_equity=pd.Series([6000, 6100, 6200, 6300, 6400]),
        )
    )
    # Synthetic ground-truth: RI = 1000 - (0.10 * 6000) = 400
    assert (
        intrinsic_model.get_residual_income(
            net_income=1000,
            cost_of_equity=0.10,
            book_value_of_equity=6000,
        )
        == 400
    )


def test_get_residual_income_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_residual_income(
            net_income="not a number",
            cost_of_equity=0.10,
            book_value_of_equity=6000,
        )
