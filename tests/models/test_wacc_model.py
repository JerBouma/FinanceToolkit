"""WACC Model Tests""" ""
import numpy as np
import pandas as pd
import pytest

from financetoolkit.models.valuation import wacc_model


@pytest.mark.parametrize(
    "total_debt, interest_expense", [(1000.0, 50.0), (0.0, 0.0), (1000.0, 10.0)]
)
def test_weighted_average_cost_of_capital_accepts_scalar_inputs(
    total_debt, interest_expense
):
    inputs = {
        "share_price": 10.0,
        "total_shares_outstanding": 100.0,
        "interest_expense": interest_expense,
        "total_debt": total_debt,
        "risk_free_rate": 0.02,
        "beta": 1.0,
        "benchmark_returns": 0.08,
        "income_tax_expense": 21.0,
        "income_before_tax": 100.0,
    }
    expected = wacc_model.get_weighted_average_cost_of_capital(
        **{name: pd.Series([value]) for name, value in inputs.items()}
    )

    result = wacc_model.get_weighted_average_cost_of_capital(**inputs)

    pd.testing.assert_frame_equal(result, expected)
    equity = inputs["share_price"] * inputs["total_shares_outstanding"]
    expected_wacc = (equity * 0.08 + interest_expense * 0.79) / (equity + total_debt)
    assert result.loc["Weighted Average Cost of Capital", 0] == pytest.approx(
        expected_wacc
    )


def test_get_cost_of_equity(recorder):
    recorder.capture(
        wacc_model.get_cost_of_equity(
            risk_free_rate=pd.Series([0.01, 0.02, 0.03, 0.0, 0.05]),
            beta=pd.Series([1, 1.2, 1.1, 1.3, 0.9]),
            benchmark_returns=pd.Series([0.03, 0.04, 0.06, 0.05, 0.10]),
        )
    )


def test_get_cost_of_debt(recorder):
    recorder.capture(
        wacc_model.get_cost_of_debt(
            interest_expense=pd.Series([10, 20, 33, 0, 5]),
            total_debt=pd.Series([100, 102, 101, 103, 90]),
        )
    )


def test_get_weighted_average_cost_of_capital(recorder):
    recorder.capture(
        wacc_model.get_weighted_average_cost_of_capital(
            share_price=pd.Series([10, 20, 33, 20, 5]),
            total_shares_outstanding=pd.Series([100, 102, 101, 103, 90]),
            interest_expense=pd.Series([10, 20, 33, 0, 5]),
            total_debt=pd.Series([100, 102, 101, 103, 90]),
            risk_free_rate=pd.Series([0.01, 0.02, 0.03, 0.0, 0.05]),
            beta=pd.Series([1, 1.2, 1.1, 1.3, 0.9]),
            benchmark_returns=pd.Series([0.03, 0.04, 0.06, 0.05, 0.10]),
            income_tax_expense=pd.Series([10, 20, 33, 0, 5]),
            income_before_tax=pd.Series([100, 102, 101, 103, 90]),
        )
    )


def test_weighted_average_cost_of_capital_without_debt_is_the_cost_of_equity():
    wacc = wacc_model.get_weighted_average_cost_of_capital(
        share_price=pd.Series([10.0, 10.0]),
        total_shares_outstanding=pd.Series([100.0, 100.0]),
        interest_expense=pd.Series([0.0, 5.0]),
        total_debt=pd.Series([0.0, 1000.0]),
        risk_free_rate=pd.Series([0.02, 0.02]),
        beta=pd.Series([1.0, 1.0]),
        benchmark_returns=pd.Series([0.08, 0.08]),
        income_tax_expense=pd.Series([0.0, 0.0]),
        income_before_tax=pd.Series([0.0, 100.0]),
    )

    # The components are the rows.
    assert wacc.loc["Cost of Debt"].tolist() == [0.0, 0.005]
    assert (
        wacc.loc["Weighted Average Cost of Capital", 0] == wacc.loc["Cost of Equity", 0]
    )


def test_weighted_average_cost_of_capital_keeps_the_tax_rate_within_zero_and_one():
    wacc = wacc_model.get_weighted_average_cost_of_capital(
        share_price=pd.Series([10.0, 10.0, 10.0]),
        total_shares_outstanding=pd.Series([100.0, 100.0, 100.0]),
        interest_expense=pd.Series([50.0, 50.0, 50.0]),
        total_debt=pd.Series([1000.0, 1000.0, 1000.0]),
        risk_free_rate=pd.Series([0.02, 0.02, 0.02]),
        beta=pd.Series([1.0, 1.0, 1.0]),
        benchmark_returns=pd.Series([0.08, 0.08, 0.08]),
        # A tax benefit on a profit, a loss (no taxable income to shield) and an ordinary rate.
        income_tax_expense=pd.Series([-20.0, 30.0, 21.0]),
        income_before_tax=pd.Series([100.0, -10.0, 100.0]),
    )

    assert wacc.loc["Corporate Tax Rate"].tolist() == [0.0, 0.0, 0.21]
    # Half equity at 8% and half debt at 5% before tax.
    assert wacc.loc["Weighted Average Cost of Capital"].tolist() == pytest.approx(
        [0.065, 0.065, 0.5 * 0.08 + 0.5 * 0.05 * 0.79]
    )


def test_corporate_tax_rate_is_clipped_and_zero_without_a_pre_tax_profit():
    tax_rate = wacc_model.get_corporate_tax_rate(
        income_tax_expense=pd.Series([-20.0, 30.0, 21.0, 150.0, -5.0]),
        income_before_tax=pd.Series([100.0, -10.0, 100.0, 100.0, -50.0]),
    )

    # A tax benefit on a profit, a loss, an ordinary rate, tax above the profit and two losses.
    assert tax_rate.tolist() == [0.0, 0.0, 0.21, 1.0, 0.0]
    assert wacc_model.get_corporate_tax_rate(-5.0, -50.0) == 0.0
    assert wacc_model.get_corporate_tax_rate(21.0, 100.0) == pytest.approx(0.21)


def test_debt_without_an_interest_expense_has_an_unknown_cost_of_debt():
    cost_of_debt = wacc_model.get_cost_of_debt(
        interest_expense=pd.Series([50.0, 0.0, 0.0]),
        total_debt=pd.Series([1000.0, 1000.0, 0.0]),
    )
    # Debt without a reported interest expense is not free debt; no debt at all is 0 / 0.
    assert cost_of_debt.iloc[0] == 0.05
    assert cost_of_debt.iloc[1:].isna().all()
    assert np.isnan(wacc_model.get_cost_of_debt(0.0, 1000.0))

    wacc = wacc_model.get_weighted_average_cost_of_capital(
        share_price=pd.Series([10.0, 10.0, 10.0]),
        total_shares_outstanding=pd.Series([100.0, 100.0, 100.0]),
        interest_expense=pd.Series([50.0, 0.0, 0.0]),
        total_debt=pd.Series([1000.0, 1000.0, 0.0]),
        risk_free_rate=pd.Series([0.02, 0.02, 0.02]),
        beta=pd.Series([1.0, 1.0, 1.0]),
        benchmark_returns=pd.Series([0.08, 0.08, 0.08]),
        income_tax_expense=pd.Series([21.0, 21.0, 21.0]),
        income_before_tax=pd.Series([100.0, 100.0, 100.0]),
    )

    weighted = wacc.loc["Weighted Average Cost of Capital"]
    assert weighted.iloc[0] == pytest.approx(0.5 * 0.08 + 0.5 * 0.05 * 0.79)
    assert pd.isna(weighted.iloc[1])
    # Without debt the WACC is the cost of equity.
    assert weighted.iloc[2] == pytest.approx(0.08)


def test_unlevered_and_relevered_beta_follow_hamada():
    # Damodaran's example: a levered beta of 1.2 at a D/E of 50% and a 40% tax rate.
    unlevered = wacc_model.get_unlevered_beta(1.2, 0.5, 0.4)
    assert unlevered == pytest.approx(1.2 / 1.3)
    assert wacc_model.get_relevered_beta(unlevered, 0.5, 0.4) == pytest.approx(1.2)
    # Without debt the levered and unlevered betas are the same.
    assert wacc_model.get_unlevered_beta(1.1, 0.0, 0.25) == pytest.approx(1.1)
