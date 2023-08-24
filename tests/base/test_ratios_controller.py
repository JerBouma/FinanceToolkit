"""Ratios Controller Tests""" ""
import pandas as pd

from financetoolkit import Toolkit

balance_dataset = pd.read_pickle("tests/base/datasets/balance_dataset.pickle")
income_dataset = pd.read_pickle("tests/base/datasets/income_dataset.pickle")
cash_dataset = pd.read_pickle("tests/base/datasets/cash_dataset.pickle")
historical = pd.read_pickle("tests/base/datasets/historical_dataset.pickle")

toolkit = Toolkit(
    tickers=["AAPL", "MSFT"],
    historical=historical,
    balance=balance_dataset,
    income=income_dataset,
    cash=cash_dataset,
)

ratios_module = toolkit.ratios

# pylint: disable=missing-function-docstring


def test_collect_all_ratios(recorder):
    recorder.capture(ratios_module.collect_all_ratios())
    recorder.capture(ratios_module.collect_all_ratios(growth=True))
    recorder.capture(ratios_module.collect_all_ratios(growth=True, lag=[1, 2, 3]))


def test_collect_efficiency_ratios(recorder):
    recorder.capture(ratios_module.collect_efficiency_ratios())
    recorder.capture(ratios_module.collect_efficiency_ratios(growth=True))
    recorder.capture(
        ratios_module.collect_efficiency_ratios(growth=True, lag=[1, 2, 3])
    )


def test_collect_solvency_ratios(recorder):
    recorder.capture(ratios_module.collect_solvency_ratios())
    recorder.capture(ratios_module.collect_solvency_ratios(growth=True))
    recorder.capture(ratios_module.collect_solvency_ratios(growth=True, lag=[1, 2, 3]))


def test_collect_liquidity_ratios(recorder):
    recorder.capture(ratios_module.collect_liquidity_ratios())
    recorder.capture(ratios_module.collect_liquidity_ratios(growth=True))
    recorder.capture(ratios_module.collect_liquidity_ratios(growth=True, lag=[1, 2, 3]))


def test_collect_profitability_ratios(recorder):
    recorder.capture(ratios_module.collect_profitability_ratios())
    recorder.capture(ratios_module.collect_profitability_ratios(growth=True))
    recorder.capture(
        ratios_module.collect_profitability_ratios(growth=True, lag=[1, 2, 3])
    )


def test_collect_valuation_ratios(recorder):
    recorder.capture(ratios_module.collect_valuation_ratios())
    recorder.capture(ratios_module.collect_valuation_ratios(growth=True))
    recorder.capture(ratios_module.collect_valuation_ratios(growth=True, lag=[1, 2, 3]))


def test_get_asset_turnover_ratio(recorder):
    recorder.capture(ratios_module.get_asset_turnover_ratio())


def test_get_operating_ratio(recorder):
    recorder.capture(ratios_module.get_operating_ratio())


def test_get_inventory_turnover_ratio(recorder):
    recorder.capture(ratios_module.get_inventory_turnover_ratio())


def test_get_days_of_inventory_outstanding(recorder):
    recorder.capture(ratios_module.get_days_of_inventory_outstanding())


def test_get_days_of_sales_outstanding(recorder):
    recorder.capture(ratios_module.get_days_of_sales_outstanding())


def test_get_operating_cycle(recorder):
    recorder.capture(ratios_module.get_operating_cycle())


def test_get_days_of_accounts_payable_outstanding(recorder):
    recorder.capture(ratios_module.get_days_of_accounts_payable_outstanding())


def test_get_accounts_payables_turnover_ratio(recorder):
    recorder.capture(ratios_module.get_accounts_payables_turnover_ratio())


def test_get_cash_conversion_cycle(recorder):
    recorder.capture(ratios_module.get_cash_conversion_cycle())


def test_get_receivables_turnover(recorder):
    recorder.capture(ratios_module.get_receivables_turnover())


def test_get_sga_to_revenue_ratio(recorder):
    recorder.capture(ratios_module.get_sga_to_revenue_ratio())


def test_get_fixed_asset_turnover(recorder):
    recorder.capture(ratios_module.get_fixed_asset_turnover())


def test_get_current_ratio(recorder):
    recorder.capture(ratios_module.get_current_ratio())


def test_get_quick_ratio(recorder):
    recorder.capture(ratios_module.get_quick_ratio())


def test_get_cash_ratio(recorder):
    recorder.capture(ratios_module.get_cash_ratio())


def test_get_working_capital(recorder):
    recorder.capture(ratios_module.get_working_capital())


def test_get_operating_cash_flow_ratio(recorder):
    recorder.capture(ratios_module.get_operating_cash_flow_ratio())


def test_get_operating_cash_flow_sales_ratio(recorder):
    recorder.capture(ratios_module.get_operating_cash_flow_sales_ratio())


def test_get_short_term_coverage_ratio(recorder):
    recorder.capture(ratios_module.get_short_term_coverage_ratio())


def test_get_gross_margin(recorder):
    recorder.capture(ratios_module.get_gross_margin())


def test_get_operating_margin(recorder):
    recorder.capture(ratios_module.get_operating_margin())


def test_get_net_profit_margin(recorder):
    recorder.capture(ratios_module.get_net_profit_margin())


def test_get_income_before_tax_profit_margin(recorder):
    recorder.capture(ratios_module.get_income_before_tax_profit_margin())


def test_get_return_on_assets(recorder):
    recorder.capture(ratios_module.get_return_on_assets())


def test_get_return_on_equity(recorder):
    recorder.capture(ratios_module.get_return_on_equity())


def test_get_return_on_invested_capital(recorder):
    recorder.capture(ratios_module.get_return_on_invested_capital())


def test_get_income_quality_ratio(recorder):
    recorder.capture(ratios_module.get_income_quality_ratio())


def test_get_return_on_tangible_assets(recorder):
    recorder.capture(ratios_module.get_return_on_tangible_assets())


def test_get_return_on_capital_employed(recorder):
    recorder.capture(ratios_module.get_return_on_capital_employed())


def test_get_net_income_per_ebt(recorder):
    recorder.capture(ratios_module.get_net_income_per_ebt())


def test_get_free_cash_flow_operating_cash_flow_ratio(recorder):
    recorder.capture(ratios_module.get_free_cash_flow_operating_cash_flow_ratio())


def test_get_tax_burden_ratio(recorder):
    recorder.capture(ratios_module.get_tax_burden_ratio())


def test_get_interest_coverage_ratio(recorder):
    recorder.capture(ratios_module.get_interest_coverage_ratio())


def test_get_EBT_to_EBIT(recorder):
    recorder.capture(ratios_module.get_EBT_to_EBIT())


def test_get_EBIT_to_revenue(recorder):
    recorder.capture(ratios_module.get_EBIT_to_revenue())


def test_get_debt_to_assets_ratio(recorder):
    recorder.capture(ratios_module.get_debt_to_assets_ratio())


def test_get_debt_to_equity_ratio(recorder):
    recorder.capture(ratios_module.get_debt_to_equity_ratio())


def test_get_interest_burden_ratio(recorder):
    recorder.capture(ratios_module.get_interest_burden_ratio())


def test_get_equity_multiplier(recorder):
    recorder.capture(ratios_module.get_equity_multiplier())


def test_get_free_cash_flow_yield(recorder):
    recorder.capture(ratios_module.get_free_cash_flow_yield())


def test_get_net_debt_to_ebitda_ratio(recorder):
    recorder.capture(ratios_module.get_net_debt_to_ebitda_ratio())


def test_get_cash_flow_coverage_ratio(recorder):
    recorder.capture(ratios_module.get_cash_flow_coverage_ratio())


def test_get_capex_coverage_ratio(recorder):
    recorder.capture(ratios_module.get_capex_coverage_ratio())


def test_get_capex_dividend_coverage_ratio(recorder):
    recorder.capture(ratios_module.get_capex_dividend_coverage_ratio())


def test_get_earnings_per_share(recorder):
    recorder.capture(ratios_module.get_earnings_per_share())


def test_get_earnings_per_share_growth(recorder):
    recorder.capture(ratios_module.get_earnings_per_share_growth())


def test_get_revenue_per_share(recorder):
    recorder.capture(ratios_module.get_revenue_per_share())


def test_get_price_earnings_ratio(recorder):
    recorder.capture(ratios_module.get_price_earnings_ratio())


def test_get_price_to_earnings_growth_ratio(recorder):
    recorder.capture(ratios_module.get_price_to_earnings_growth_ratio())


def test_get_book_value_per_share(recorder):
    recorder.capture(ratios_module.get_book_value_per_share())


def test_get_price_to_book_ratio(recorder):
    recorder.capture(ratios_module.get_price_to_book_ratio())


def test_get_interest_debt_per_share(recorder):
    recorder.capture(ratios_module.get_interest_debt_per_share())


def test_get_capex_per_share(recorder):
    recorder.capture(ratios_module.get_capex_per_share())


def test_get_dividend_yield(recorder):
    recorder.capture(ratios_module.get_dividend_yield())


def test_get_price_to_cash_flow_ratio(recorder):
    recorder.capture(ratios_module.get_price_to_cash_flow_ratio())


def test_get_price_to_free_cash_flow_ratio(recorder):
    recorder.capture(ratios_module.get_price_to_free_cash_flow_ratio())


def test_get_market_cap(recorder):
    recorder.capture(ratios_module.get_market_cap())


def test_get_enterprise_value(recorder):
    recorder.capture(ratios_module.get_enterprise_value())


def test_get_ev_to_sales_ratio(recorder):
    recorder.capture(ratios_module.get_ev_to_sales_ratio())


def test_get_ev_to_ebitda_ratio(recorder):
    recorder.capture(ratios_module.get_ev_to_ebitda_ratio())


def test_get_ev_to_operating_cashflow_ratio(recorder):
    recorder.capture(ratios_module.get_ev_to_operating_cashflow_ratio())


def test_get_earnings_yield(recorder):
    recorder.capture(ratios_module.get_earnings_yield())


def test_get_payout_ratio(recorder):
    recorder.capture(ratios_module.get_payout_ratio())


def test_get_tangible_asset_value(recorder):
    recorder.capture(ratios_module.get_tangible_asset_value())


def test_get_net_current_asset_value(recorder):
    recorder.capture(ratios_module.get_net_current_asset_value())
