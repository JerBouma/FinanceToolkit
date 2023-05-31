"""Solvency Ratios Tests""" ""
import pandas as pd

from financetoolkit.ratios import solvency

# pylint: disable=missing-function-docstring


def test_get_debt_to_assets_ratio(recorder):
    recorder.capture(
        solvency.get_debt_to_assets_ratio(
            total_debt=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_debt_to_equity_ratio(recorder):
    recorder.capture(
        solvency.get_debt_to_equity_ratio(
            total_debt=pd.Series([100, 110, 120, 130, 80]),
            total_equity=pd.Series([150, 130, 120, 100, 50]),
        )
    )


def test_get_interest_coverage_ratio(recorder):
    recorder.capture(
        solvency.get_interest_coverage_ratio(
            operating_income=pd.Series([100, 110, 120, 130, 80]),
            depreciation_and_amortization=pd.Series([30, 20, 10, 10, 0]),
            interest_expense=pd.Series([15, 13, 12, 10.3, 5]),
        )
    )


def test_get_debt_service_coverage_ratio(recorder):
    recorder.capture(
        solvency.get_debt_service_coverage_ratio(
            operating_income=pd.Series([100, 110, 120, 130, 80]),
            current_liabilities=pd.Series([300, 200, 100, 100, 150]),
        )
    )


def test_get_equity_multiplier(recorder):
    recorder.capture(
        solvency.get_equity_multiplier(
            total_assets_begin=pd.Series([100, 110, 120, 130, 180]),
            total_assets_end=pd.Series([200, 210, 220, 230, 280]),
            total_equity_begin=pd.Series([150, 130, 120, 100, 50]),
            total_equity_end=pd.Series([250, 120, 200, 200, 150]),
        )
    )


def test_get_free_cash_flow_yield(recorder):
    recorder.capture(
        solvency.get_free_cash_flow_yield(
            free_cash_flow=pd.Series([200, 210, 220, 230, 280]),
            market_capitalization=pd.Series([1000, 1300, 1200, 1000, 500]),
        )
    )


def test_get_net_debt_to_ebitda_ratio(recorder):
    recorder.capture(
        solvency.get_net_debt_to_ebitda_ratio(
            operating_income=pd.Series([200, 210, 220, 230, 280]),
            depreciation_and_amortization=pd.Series([20, 21, 22, 23, 28]),
            net_debt=pd.Series([1000, 1300, 1200, 1000, 500]),
        )
    )


def test_get_cash_flow_coverage_ratio(recorder):
    recorder.capture(
        solvency.get_cash_flow_coverage_ratio(
            operating_cash_flow=pd.Series([200, 210, 220, 230, 280]),
            total_debt=pd.Series([1000, 1300, 1200, 1000, 500]),
        )
    )


def test_get_capex_coverage_ratio(recorder):
    recorder.capture(
        solvency.get_capex_coverage_ratio(
            cash_flow_from_operations=pd.Series([200, 210, 220, 230, 280]),
            capital_expenditure=pd.Series([10, 5, 3, 5, 0]),
        )
    )


def test_get_dividend_capex_coverage_ratio(recorder):
    recorder.capture(
        solvency.get_dividend_capex_coverage_ratio(
            cash_flow_from_operations=pd.Series([200, 210, 220, 230, 280]),
            capital_expenditure=pd.Series([10, 5, 3, 5, 0]),
            dividends=pd.Series([1, 3, 2, 0, 5]),
        )
    )
