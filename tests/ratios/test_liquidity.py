"""Liquidity Ratios Tests""" ""
import pandas as pd

from financetoolkit.ratios import liquidity

# pylint: disable=missing-function-docstring


def test_get_current_ratio(recorder):
    recorder.capture(
        liquidity.get_current_ratio(
            current_assets=pd.Series([100, 110, 120, 130, 80]),
            current_liabilities=pd.Series([130, 120, 110, 150, 100]),
        )
    )


def test_get_quick_ratio(recorder):
    recorder.capture(
        liquidity.get_quick_ratio(
            cash_and_equivalents=pd.Series([100, 110, 120, 130, 80]),
            accounts_receivable=pd.Series([30, 20, 30, 20, 40]),
            marketable_securities=pd.Series([30, 10, 30, 20, 40]),
            current_liabilities=pd.Series([130, 120, 110, 150, 100]),
        )
    )


def test_get_cash_ratio(recorder):
    recorder.capture(
        liquidity.get_cash_ratio(
            cash_and_equivalents=pd.Series([100, 110, 120, 130, 80]),
            marketable_securities=pd.Series([30, 20, 30, 20, 40]),
            current_liabilities=pd.Series([130, 120, 110, 150, 100]),
        )
    )


def test_get_working_capital(recorder):
    recorder.capture(
        liquidity.get_working_capital(
            current_assets=pd.Series([100, 110, 120, 130, 80]),
            current_liabilities=pd.Series([130, 120, 110, 150, 100]),
        )
    )


def test_get_operating_cash_flow_ratio(recorder):
    recorder.capture(
        liquidity.get_operating_cash_flow_ratio(
            operating_cash_flow=pd.Series([100, 110, 120, 130, 80]),
            current_liabilities=pd.Series([130, 120, 110, 150, 100]),
        )
    )


def test_get_operating_cash_flow_sales_ratio(recorder):
    recorder.capture(
        liquidity.get_operating_cash_flow_sales_ratio(
            operating_cash_flow=pd.Series([100, 110, 120, 130, 80]),
            revenue=pd.Series([200, 150, 150, 130, 100]),
        )
    )


def test_get_short_term_coverage_ratio(recorder):
    recorder.capture(
        liquidity.get_short_term_coverage_ratio(
            operating_cash_flow=pd.Series([400, 300, 350, 200, 150]),
            accounts_receivable=pd.Series([50, 100, 30, 130, 100]),
            inventory=pd.Series([30, 50, 50, 30, 10]),
            accounts_payable=pd.Series([20, 15, 10, 13, 10]),
        )
    )
