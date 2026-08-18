"""Ratios Controller Tests""" ""
# pylint: disable=missing-function-docstring


def test_collect_all_ratios(recorder, ratios_module):
    recorder.capture(ratios_module.collect_all_ratios())
    recorder.capture(ratios_module.collect_all_ratios(growth=True))
    recorder.capture(ratios_module.collect_all_ratios(growth=True, lag=[1, 2, 3]))
    recorder.capture(ratios_module.collect_all_ratios(growth=False, trailing=2))
    recorder.capture(ratios_module.collect_all_ratios(growth=True, trailing=2))


def test_collect_custom_ratios(recorder, ratios_module):
    recorder.capture(
        ratios_module.collect_custom_ratios(
            custom_ratios_dict={
                "WC / Net Income as %": "(Working Capital / Net Income) * 100",
                "Quick Assets": "Cash and Short Term Investments + Accounts Receivable",
            }
        )
    )
    recorder.capture(
        ratios_module.collect_custom_ratios(
            custom_ratios_dict={
                "WC / Net Income as %": "(Working Capital / Net Income) * 100",
                "Quick Assets": "Cash and Short Term Investments + Accounts Receivable",
            },
            growth=True,
        )
    )


def test_collect_efficiency_ratios(recorder, ratios_module):
    recorder.capture(ratios_module.collect_efficiency_ratios())
    recorder.capture(ratios_module.collect_efficiency_ratios(growth=True))
    recorder.capture(
        ratios_module.collect_efficiency_ratios(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(ratios_module.collect_efficiency_ratios(growth=False, trailing=2))
    recorder.capture(ratios_module.collect_efficiency_ratios(growth=True, trailing=2))


def test_collect_solvency_ratios(recorder, ratios_module):
    recorder.capture(ratios_module.collect_solvency_ratios())
    recorder.capture(ratios_module.collect_solvency_ratios(growth=True))
    recorder.capture(ratios_module.collect_solvency_ratios(growth=True, lag=[1, 2, 3]))
    recorder.capture(ratios_module.collect_solvency_ratios(growth=False, trailing=2))
    recorder.capture(ratios_module.collect_solvency_ratios(growth=True, trailing=2))


def test_collect_liquidity_ratios(recorder, ratios_module):
    recorder.capture(ratios_module.collect_liquidity_ratios())
    recorder.capture(ratios_module.collect_liquidity_ratios(growth=True))
    recorder.capture(ratios_module.collect_liquidity_ratios(growth=True, lag=[1, 2, 3]))
    recorder.capture(ratios_module.collect_liquidity_ratios(growth=False, trailing=2))
    recorder.capture(ratios_module.collect_liquidity_ratios(growth=True, trailing=2))


def test_collect_profitability_ratios(recorder, ratios_module):
    recorder.capture(ratios_module.collect_profitability_ratios())
    recorder.capture(ratios_module.collect_profitability_ratios(growth=True))
    recorder.capture(
        ratios_module.collect_profitability_ratios(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        ratios_module.collect_profitability_ratios(growth=False, trailing=2)
    )
    recorder.capture(
        ratios_module.collect_profitability_ratios(growth=True, trailing=2)
    )


def test_collect_valuation_ratios(recorder, ratios_module):
    recorder.capture(ratios_module.collect_valuation_ratios())
    recorder.capture(ratios_module.collect_valuation_ratios(growth=True))
    recorder.capture(ratios_module.collect_valuation_ratios(growth=True, lag=[1, 2, 3]))
    recorder.capture(ratios_module.collect_valuation_ratios(growth=False, trailing=2))
    recorder.capture(ratios_module.collect_valuation_ratios(growth=True, trailing=2))


def test_get_asset_turnover_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_asset_turnover_ratio())


def test_get_operating_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_ratio())


def test_get_inventory_turnover_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_inventory_turnover_ratio())


def test_get_days_of_inventory_outstanding(recorder, ratios_module):
    recorder.capture(ratios_module.get_days_of_inventory_outstanding())


def test_get_days_of_sales_outstanding(recorder, ratios_module):
    recorder.capture(ratios_module.get_days_of_sales_outstanding())


def test_get_operating_cycle(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_cycle())


def test_get_days_of_accounts_payable_outstanding(recorder, ratios_module):
    recorder.capture(ratios_module.get_days_of_accounts_payable_outstanding())


def test_get_accounts_payables_turnover_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_accounts_payables_turnover_ratio())


def test_get_cash_conversion_cycle(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_conversion_cycle())


def test_get_cash_conversion_efficiency(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_conversion_efficiency())


def test_get_receivables_turnover(recorder, ratios_module):
    recorder.capture(ratios_module.get_receivables_turnover())


def test_get_sga_to_revenue_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_sga_to_revenue_ratio())


def test_get_fixed_asset_turnover(recorder, ratios_module):
    recorder.capture(ratios_module.get_fixed_asset_turnover())


def test_get_working_capital_turnover_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_working_capital_turnover_ratio())


def test_get_current_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_current_ratio())


def test_get_quick_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_quick_ratio())


def test_get_cash_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_ratio())


def test_get_working_capital(recorder, ratios_module):
    recorder.capture(ratios_module.get_working_capital())


def test_get_operating_cash_flow_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_cash_flow_ratio())


def test_get_operating_cash_flow_sales_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_cash_flow_sales_ratio())


def test_get_short_term_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_short_term_coverage_ratio())


def test_get_defensive_interval_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_defensive_interval_ratio())


def test_get_gross_margin(recorder, ratios_module):
    recorder.capture(ratios_module.get_gross_margin())


def test_get_operating_margin(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_margin())


def test_get_net_profit_margin(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_profit_margin())


def test_get_ebitda_margin(recorder, ratios_module):
    recorder.capture(ratios_module.get_ebitda_margin())


def test_get_free_cash_flow_margin(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_margin())


def test_get_income_before_tax_profit_margin(recorder, ratios_module):
    recorder.capture(ratios_module.get_income_before_tax_profit_margin())


def test_get_effective_tax_rate(recorder, ratios_module):
    recorder.capture(ratios_module.get_effective_tax_rate())


def test_get_return_on_assets(recorder, ratios_module):
    recorder.capture(ratios_module.get_return_on_assets())


def test_get_cash_return_on_assets(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_return_on_assets())


def test_get_return_on_equity(recorder, ratios_module):
    recorder.capture(ratios_module.get_return_on_equity())


def test_get_return_on_invested_capital(recorder, ratios_module):
    recorder.capture(ratios_module.get_return_on_invested_capital())


def test_get_income_quality_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_income_quality_ratio())


def test_get_return_on_tangible_assets(recorder, ratios_module):
    recorder.capture(ratios_module.get_return_on_tangible_assets())


def test_get_return_on_capital_employed(recorder, ratios_module):
    recorder.capture(ratios_module.get_return_on_capital_employed())


def test_get_net_income_per_ebt(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_income_per_ebt())


def test_get_free_cash_flow_operating_cash_flow_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_operating_cash_flow_ratio())


def test_get_tax_burden_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_tax_burden_ratio())


def test_get_interest_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_interest_coverage_ratio())


def test_get_EBT_to_EBIT(recorder, ratios_module):
    recorder.capture(ratios_module.get_EBT_to_EBIT())


def test_get_EBIT_to_revenue(recorder, ratios_module):
    recorder.capture(ratios_module.get_EBIT_to_revenue())


def test_get_debt_to_assets_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_debt_to_assets_ratio())


def test_get_asset_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_asset_coverage_ratio())


def test_get_debt_to_equity_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_debt_to_equity_ratio())


def test_get_interest_burden_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_interest_burden_ratio())


def test_get_equity_multiplier(recorder, ratios_module):
    recorder.capture(ratios_module.get_equity_multiplier())


def test_get_debt_service_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_debt_service_coverage_ratio())


def test_get_free_cash_flow_yield(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_yield())


def test_get_net_debt_to_ebitda_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_debt_to_ebitda_ratio())


def test_get_gross_debt_to_ebitda_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_gross_debt_to_ebitda_ratio())


def test_get_cash_flow_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_flow_coverage_ratio())


def test_get_capex_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_capex_coverage_ratio())


def test_get_capex_dividend_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_capex_dividend_coverage_ratio())


def test_get_earnings_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_earnings_per_share())


def test_get_revenue_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_revenue_per_share())


def test_get_price_to_earnings_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_price_to_earnings_ratio())
    recorder.capture(ratios_module.get_price_to_earnings_ratio(show_daily=True))


def test_get_price_to_earnings_growth_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_price_to_earnings_growth_ratio())
    recorder.capture(
        ratios_module.get_price_to_earnings_growth_ratio(use_ebitda_growth_rate=True)
    )


def test_get_book_value_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_book_value_per_share())


def test_get_price_to_book_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_price_to_book_ratio())
    recorder.capture(ratios_module.get_price_to_book_ratio(show_daily=True))


def test_get_interest_debt_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_interest_debt_per_share())


def test_get_capex_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_capex_per_share())


def test_get_dividend_yield(recorder, ratios_module):
    recorder.capture(ratios_module.get_dividend_yield())
    recorder.capture(ratios_module.get_dividend_yield(show_daily=True))


def test_get_weighted_dividend_yield(recorder, ratios_module):
    recorder.capture(ratios_module.get_weighted_dividend_yield())
    recorder.capture(ratios_module.get_weighted_dividend_yield(show_daily=True))


def test_get_price_to_cash_flow_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_price_to_cash_flow_ratio())
    recorder.capture(ratios_module.get_price_to_cash_flow_ratio(show_daily=True))


def test_get_price_to_free_cash_flow_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_price_to_free_cash_flow_ratio())
    recorder.capture(ratios_module.get_price_to_free_cash_flow_ratio(show_daily=True))


def test_get_price_to_sales_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_price_to_sales_ratio())
    recorder.capture(ratios_module.get_price_to_sales_ratio(show_daily=True))


def test_get_market_cap(recorder, ratios_module):
    recorder.capture(ratios_module.get_market_cap())
    recorder.capture(ratios_module.get_market_cap(show_daily=True))


def test_get_enterprise_value(recorder, ratios_module):
    recorder.capture(ratios_module.get_enterprise_value())
    recorder.capture(ratios_module.get_enterprise_value(show_daily=True))


def test_get_ev_to_sales_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_ev_to_sales_ratio())
    recorder.capture(ratios_module.get_ev_to_sales_ratio(show_daily=True))


def test_get_ev_to_ebitda_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_ev_to_ebitda_ratio())
    recorder.capture(ratios_module.get_ev_to_ebitda_ratio(show_daily=True))


def test_get_ev_to_operating_cashflow_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_ev_to_operating_cashflow_ratio())
    recorder.capture(ratios_module.get_ev_to_operating_cashflow_ratio(show_daily=True))


def test_get_earnings_yield(recorder, ratios_module):
    recorder.capture(ratios_module.get_earnings_yield())
    recorder.capture(ratios_module.get_earnings_yield(show_daily=True))


def test_get_dividend_payout_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_dividend_payout_ratio())


def test_get_reinvestment_rate(recorder, ratios_module):
    recorder.capture(ratios_module.get_reinvestment_rate())


def test_get_tangible_asset_value(recorder, ratios_module):
    recorder.capture(ratios_module.get_tangible_asset_value())


def test_get_net_current_asset_value(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_current_asset_value())


def test_get_ev_to_ebit(recorder, ratios_module):
    recorder.capture(ratios_module.get_ev_to_ebit())
    recorder.capture(ratios_module.get_ev_to_ebit(show_daily=True))
