"""Profitability Model Tests""" ""
import pandas as pd

from financetoolkit.ratios import profitability_model


# pylint: disable=missing-function-docstring
def test_get_gross_margin(recorder):
    recorder.capture(
        profitability_model.get_gross_margin(
            revenue=pd.Series([100, 110, 120, 130, 80]),
            cost_of_goods_sold=pd.Series([30, 40, 60, 60, 20]),
        )
    )


def test_get_operating_margin(recorder):
    recorder.capture(
        profitability_model.get_operating_margin(
            operating_income=pd.Series([80, 60, 70, 80, 90]),
            revenue=pd.Series([100, 110, 120, 130, 80]),
        )
    )


def test_get_net_profit_margin(recorder):
    recorder.capture(
        profitability_model.get_net_profit_margin(
            net_income=pd.Series([60, 50, 40, 30, 50]),
            revenue=pd.Series([100, 110, 120, 130, 80]),
        )
    )


def test_get_interest_coverage_ratio(recorder):
    recorder.capture(
        profitability_model.get_interest_coverage_ratio(
            operating_income=pd.Series([60, 50, 40, 30, 50]),
            interest_expense=pd.Series([80, 60, 70, 80, 90]),
        )
    )


def test_get_income_before_tax_profit_margin(recorder):
    recorder.capture(
        profitability_model.get_income_before_tax_profit_margin(
            income_before_tax=pd.Series([60, 50, 40, 30, 50]),
            revenue=pd.Series([200, 150, 130, 100, 90]),
        )
    )


def test_get_effective_tax_rate(recorder):
    recorder.capture(
        profitability_model.get_effective_tax_rate(
            income_tax_expense=pd.Series([80, 40, 40, 30, 20]),
            income_before_tax=pd.Series([200, 150, 130, 100, 90]),
        )
    )


def test_get_return_on_assets(recorder):
    recorder.capture(
        profitability_model.get_return_on_assets(
            net_income=pd.Series([80, 40, 40, 30, 20]),
            average_total_assets=pd.Series([200, 300, 400, 500, 450]),
        )
    )


def test_get_return_on_equity(recorder):
    recorder.capture(
        profitability_model.get_return_on_equity(
            net_income=pd.Series([80, 40, 40, 30, 20]),
            average_total_equity=pd.Series([130, 200, 210, 200, 150]),
        )
    )


def test_get_return_on_invested_capital(recorder):
    recorder.capture(
        profitability_model.get_return_on_invested_capital(
            net_income=pd.Series([80, 40, 40, 30, 20]),
            dividends=pd.Series([4, 3, 2, 1, 10]),
            average_total_equity=pd.Series([130, 200, 210, 200, 150]),
            average_total_debt=pd.Series([130, 200, 210, 200, 150]),
        )
    )


def test_get_income_quality_ratio(recorder):
    recorder.capture(
        profitability_model.get_income_quality_ratio(
            cash_flow_from_operating_activities=pd.Series([130, 200, 210, 200, 150]),
            net_income=pd.Series([80, 40, 40, 30, 20]),
        )
    )


def test_get_return_on_tangible_assets(recorder):
    recorder.capture(
        profitability_model.get_return_on_tangible_assets(
            net_income=pd.Series([80, 40, 40, 30, 20]),
            average_total_assets=pd.Series([200, 300, 400, 500, 450]),
            average_intangible_assets=pd.Series([20, 20, 21, 20, 15]),
            average_total_liabilities=pd.Series([130, 200, 210, 200, 150]),
        )
    )


def test_get_return_on_capital_employed(recorder):
    recorder.capture(
        profitability_model.get_return_on_capital_employed(
            net_income=pd.Series([80, 40, 40, 30, 20]),
            interest_expense=pd.Series([20, 20, 21, 20, 15]),
            tax_expense=pd.Series([5, 10, 20, 10, 5]),
            total_assets=pd.Series([200, 300, 400, 500, 450]),
            total_current_liabilities=pd.Series([130, 200, 210, 200, 150]),
        )
    )


def test_get_net_income_per_ebt(recorder):
    recorder.capture(
        profitability_model.get_net_income_per_ebt(
            net_income=pd.Series([80, 40, 40, 30, 20]),
            income_tax_expense=pd.Series([5, 10, 20, 10, 5]),
        )
    )


def test_get_free_cash_flow_operating_cash_flow_ratio(recorder):
    recorder.capture(
        profitability_model.get_free_cash_flow_operating_cash_flow_ratio(
            free_cash_flow=pd.Series([80, 40, 40, 30, 20]),
            operating_cash_flow=pd.Series([200, 100, 200, 150, 50]),
        )
    )


def test_get_tax_burden_ratio(recorder):
    recorder.capture(
        profitability_model.get_tax_burden_ratio(
            net_income=pd.Series([80, 40, 40, 30, 20]),
            income_before_tax=pd.Series([200, 100, 50, 150, 50]),
        )
    )


def test_get_EBIT_to_revenue(recorder):
    recorder.capture(
        profitability_model.get_EBIT_to_revenue(
            earnings_before_interest_and_taxes=pd.Series([80, 50, 40, 30, 20]),
            revenue=pd.Series([230, 150, 50, 150, 50]),
        )
    )
