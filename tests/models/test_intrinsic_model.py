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
    # Ground truth by hand: PV 7.1585 plus a terminal 38.2063 gives about 45.3648.
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


def test_graham_number_needs_positive_earnings_and_book_value():
    import numpy as np

    graham_number = intrinsic_model.get_graham_number(
        earnings_per_share=pd.Series([2.0, -2.0, 2.0]),
        book_value_per_share=pd.Series([10.0, -10.0, -10.0]),
    )

    assert graham_number.iloc[0] == np.sqrt(22.5 * 2 * 10)
    assert graham_number.iloc[1:].isna().all()


def test_get_implied_growth_rate_recovers_the_dcf_growth():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    value = float(
        intrinsic_model.get_intrinsic_value(100, 0.05, 0.025, 0.09, 50, 80, 10, 5)
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    implied = intrinsic_model.get_implied_growth_rate(
        value, 100, 0.025, 0.09, 50, 80, 10, 5
    )
    assert implied == pytest.approx(0.05, abs=1e-8)
    assert np.isnan(
        intrinsic_model.get_implied_growth_rate(10, -5, 0.02, 0.09, 0, 0, 1)
    )


def test_get_earnings_power_value():
    import pandas as pd  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    result = intrinsic_model.get_earnings_power_value(
        revenue=pd.Series([100.0, 120.0]),
        operating_income=pd.Series([20.0, 30.0]),
        income_tax_expense=pd.Series([4.0, 5.0]),
        income_before_tax=pd.Series([20.0, 25.0]),
        weighted_average_cost_of_capital=0.1,
        cash_and_cash_equivalents=10,
        total_debt=30,
        shares_outstanding=10,
    )
    margin = (0.2 + 0.25) / 2
    nopat = margin * 120 * (1 - 0.2)
    assert result["Normalized NOPAT"] == pytest.approx(nopat)
    assert result["Earnings Power Value per Share"] == pytest.approx(
        (nopat / 0.1 - 20) / 10
    )


def test_get_monte_carlo_intrinsic_value_without_uncertainty():
    import pytest  # noqa: PLC0415

    values = intrinsic_model.get_monte_carlo_intrinsic_value(
        100, 0.05, 0.0, 0.025, 0.0, 0.09, 0.0, 50, 80, 10, 5, simulations=50
    )
    deterministic = float(
        intrinsic_model.get_intrinsic_value(100, 0.05, 0.025, 0.09, 50, 80, 10, 5)
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    assert values == pytest.approx(deterministic)


def test_implied_growth_rate_is_missing_with_a_missing_input():
    import numpy as np

    growth = intrinsic_model.get_implied_growth_rate(
        cash_flow=100.0,
        weighted_average_cost_of_capital=np.nan,
        perpetual_growth_rate=0.02,
        cash_and_cash_equivalents=10.0,
        total_debt=20.0,
        shares_outstanding=10.0,
        share_price=50.0,
    )

    assert np.isnan(growth)
