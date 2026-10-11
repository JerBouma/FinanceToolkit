"""WACC Model Tests""" ""
import pandas as pd

from financetoolkit.models import wacc_model

# pylint: disable=missing-function-docstring


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
