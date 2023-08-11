"""Valuation Ratios Tests""" ""
import pandas as pd

from financetoolkit.ratios import valuation

# pylint: disable=missing-function-docstring


def test_get_earnings_per_share(recorder):
    recorder.capture(
        valuation.get_earnings_per_share(
            net_income=pd.Series([100, 110, 120, 130, 80]),
            preferred_dividends=pd.Series([0, 0, 0, 0, 2]),
            average_outstanding_shares=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_revenue_per_share(recorder):
    recorder.capture(
        valuation.get_revenue_per_share(
            total_revenue=pd.Series([3000, 1100, 1200, 1300, 3000]),
            shares_outstanding=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_price_earnings_ratio(recorder):
    recorder.capture(
        valuation.get_price_earnings_ratio(
            stock_price=pd.Series([30, 11, 12, 10, 30]),
            earnings_per_share=pd.Series([0.5, 0.4, 0.1, 1.1, 1.2]),
        )
    )


def test_get_earnings_per_share_growth(recorder):
    recorder.capture(
        valuation.get_earnings_per_share_growth(
            earnings_per_share=pd.Series([0.5, 0.4, 0.1, 1.1, 1.2])
        )
    )


def test_get_price_to_earnings_growth_ratio(recorder):
    recorder.capture(
        valuation.get_price_to_earnings_growth_ratio(
            price_earnings=pd.Series([30, 11, 12, 10, 30]),
            earnings_per_share_growth=pd.Series([0.2, 0.1, 0.1, 0.05, 0.01]),
        )
    )


def test_get_book_value_per_share(recorder):
    recorder.capture(
        valuation.get_book_value_per_share(
            total_shareholder_equity=pd.Series([3000, 1100, 1200, 1300, 3000]),
            preferred_equity=pd.Series([200, 140, 160, 160, 300]),
            common_shares_outstanding=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_price_to_book_ratio(recorder):
    recorder.capture(
        valuation.get_price_to_book_ratio(
            price_per_share=pd.Series([30, 11, 10, 5, 30]),
            book_value_per_share=pd.Series([0.5, 0.5, 0.1, 1.1, 1.2]),
        )
    )


def test_get_interest_debt_per_share(recorder):
    recorder.capture(
        valuation.get_interest_debt_per_share(
            interest_expense=pd.Series([5, 1, 23, 5, 10]),
            total_debt=pd.Series([200, 140, 160, 160, 300]),
            shares_outstanding=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_capex_per_share(recorder):
    recorder.capture(
        valuation.get_capex_per_share(
            capital_expenditures=pd.Series([10, 10, 23, 15, 0]),
            shares_outstanding=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_dividend_yield(recorder):
    recorder.capture(
        valuation.get_dividend_yield(
            dividends=pd.Series([0.3, 0.31, 0.5, 0.25, 0.4]),
            stock_price=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_weighted_dividend_yield(recorder):
    recorder.capture(
        valuation.get_weighted_dividend_yield(
            dividends_paid=pd.Series([10, 10, 23, 15, 0]),
            shares_outstanding=pd.Series([1000, 500, 300, 160, 300]),
            stock_price=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_price_to_cash_flow_ratio(recorder):
    recorder.capture(
        valuation.get_price_to_cash_flow_ratio(
            market_cap=pd.Series([1000, 500, 300, 160, 300]),
            operations_cash_flow=pd.Series([10, 10, 23, 15, 30]),
        )
    )


def test_get_price_to_free_cash_flow_ratio(recorder):
    recorder.capture(
        valuation.get_price_to_free_cash_flow_ratio(
            market_cap=pd.Series([1000, 500, 300, 160, 300]),
            free_cash_flow=pd.Series([5, 5, 5, 3, 100]),
        )
    )


def test_get_market_cap(recorder):
    recorder.capture(
        valuation.get_market_cap(
            share_price=pd.Series([1000, 500, 300, 160, 300]),
            total_shares_outstanding=pd.Series([500, 1000, 1500, 1000, 2000]),
        )
    )


def test_get_enterprise_value(recorder):
    recorder.capture(
        valuation.get_enterprise_value(
            market_cap=pd.Series([100000, 5000, 3000, 16000, 30000]),
            total_debt=pd.Series([500, 1000, 1500, 1000, 2000]),
            minority_interest=pd.Series([50, 100, 150, 10, 20]),
            preferred_equity=pd.Series([5, 100, 150, 100, 200]),
            cash_and_cash_equivalents=pd.Series([50, 100, 10, 100, 200]),
        )
    )


def test_get_ev_to_sales_ratio(recorder):
    recorder.capture(
        valuation.get_ev_to_sales_ratio(
            enterprise_value=pd.Series([100000, 5000, 3000, 16000, 30000]),
            total_revenue=pd.Series([1000, 1000, 1500, 1000, 2000]),
        )
    )


def test_get_ev_to_ebitda_ratio(recorder):
    recorder.capture(
        valuation.get_ev_to_ebitda_ratio(
            enterprise_value=pd.Series([100000, 5000, 3000, 16000, 30000]),
            operating_income=pd.Series([1000, 1000, 1500, 1000, 2000]),
            depreciation_and_amortization=pd.Series([100, 100, 150, 100, 200]),
        )
    )


def test_get_ev_to_operating_cashflow_ratio(recorder):
    recorder.capture(
        valuation.get_ev_to_operating_cashflow_ratio(
            enterprise_value=pd.Series([100000, 5000, 3000, 16000, 30000]),
            operating_cashflow=pd.Series([500, 300, 150.5, 100, 220]),
        )
    )


def test_get_earnings_yield(recorder):
    recorder.capture(
        valuation.get_earnings_yield(
            earnings_per_share=pd.Series([0.5, 0.4, 0.4, 1.4, 0.25]),
            market_price_per_share=pd.Series([100, 110, 80, 50, 30]),
        )
    )


def test_get_payout_ratio(recorder):
    recorder.capture(
        valuation.get_payout_ratio(
            dividends=pd.Series([0.6, 0.4, 0.15, 0.4, 0.25]),
            net_income=pd.Series([100, 120, 80, 50, 30]),
        )
    )


def test_get_tangible_asset_value(recorder):
    recorder.capture(
        valuation.get_tangible_asset_value(
            total_assets=pd.Series([100, 120, 80, 50, 30]),
            total_liabilities=pd.Series([50, 30, 80, 20, 50]),
            goodwill=pd.Series([1.6, 1.4, 1.15, 1.4, 1.25]),
        )
    )


def test_get_net_current_asset_value(recorder):
    recorder.capture(
        valuation.get_net_current_asset_value(
            total_current_assets=pd.Series([100, 120, 80, 50, 30]),
            total_current_liabilities=pd.Series([50, 30, 80, 20, 50]),
        )
    )
