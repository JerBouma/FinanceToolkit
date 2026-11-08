"""Ratios Controller Tests""" ""
# pylint: disable=missing-function-docstring

import copy

import numpy as np
import pandas as pd
import pytest


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


def test_get_credit_ratios(recorder, ratios_module):
    recorder.capture(ratios_module.get_ebitda_interest_coverage_ratio())
    recorder.capture(ratios_module.get_ebitda_minus_capex_interest_coverage_ratio())
    recorder.capture(ratios_module.get_cash_interest_coverage_ratio(trailing=2))
    recorder.capture(ratios_module.get_funds_from_operations_to_debt_ratio())
    recorder.capture(
        ratios_module.get_discretionary_cash_flow_to_debt_ratio(trailing=2)
    )


def test_get_degrees_of_leverage(recorder, ratios_module):
    recorder.capture(ratios_module.get_degree_of_operating_leverage())
    recorder.capture(ratios_module.get_degree_of_financial_leverage())
    recorder.capture(ratios_module.get_degree_of_total_leverage(trailing=2))


def test_get_payout_and_book_multiples(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_payout_ratio())
    recorder.capture(ratios_module.get_ev_to_invested_capital_ratio())
    recorder.capture(ratios_module.get_ev_to_invested_capital_ratio(show_daily=True))
    recorder.capture(ratios_module.get_price_to_tangible_book_ratio())
    recorder.capture(ratios_module.get_price_to_tangible_book_ratio(show_daily=True))


# The tests below check the definitions behind the ratios against values calculated by hand.
# pylint: disable=protected-access,import-outside-toplevel

YEAR = "2022"


def _approx(expected):
    # The Toolkit rounds to 4 decimals by default, so compare with a relative tolerance.
    return pytest.approx(expected, rel=1e-3)


def _row(statement, item, ticker="AAPL"):
    return statement.loc[(ticker, item)]


def test_days_ratios_cover_the_whole_trailing_window(ratios_module):
    # With trailing=2 on annual data, COGS covers two years, so the window is 730 days.
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    inventory = _row(balance, "Inventory")
    cogs = _row(income, "Cost of Goods Sold")
    year = inventory.index.get_loc(next(p for p in inventory.index if str(p) == YEAR))

    expected = (
        inventory.iloc[year - 1 : year + 1].mean()
        / cogs.iloc[year - 1 : year + 1].sum()
        * 730
    )
    result = ratios_module.get_days_of_inventory_outstanding(trailing=2, rounding=None)

    assert result.loc["AAPL"].loc[YEAR] == _approx(expected)
    assert ratios_module._get_days(None, 4) == 1460
    assert ratios_module._get_days(100, 4) == 100


def test_defensive_interval_ratio_counts_all_cash_expenditures(ratios_module):
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    cash = ratios_module._cash_flow_statement

    def value(statement, item):
        return _row(statement, item).loc[YEAR]

    daily_expenditures = (
        value(income, "Cost of Goods Sold")
        + value(income, "Operating Expenses")
        - value(cash, "Depreciation and Amortization")
    ) / 365
    defensive_assets = (
        value(balance, "Cash and Cash Equivalents")
        + value(balance, "Short Term Investments")
        + value(balance, "Accounts Receivable")
    )
    result = ratios_module.get_defensive_interval_ratio(rounding=None)

    assert result.loc["AAPL"].loc[YEAR] == _approx(
        defensive_assets / daily_expenditures
    )


def test_return_on_tangible_assets_and_tangible_asset_value(ratios_module):
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    total_assets = _row(balance, "Total Assets", "MSFT")
    intangibles = _row(balance, "Goodwill and Intangible Assets", "MSFT")
    liabilities = _row(balance, "Total Liabilities", "MSFT")
    net_income = _row(income, "Net Income", "MSFT")
    year = total_assets.index.get_loc(
        next(p for p in total_assets.index if str(p) == YEAR)
    )

    tangible_assets = (
        total_assets.iloc[year - 1 : year + 1].mean()
        - intangibles.iloc[year - 1 : year + 1].mean()
    )
    rota = ratios_module.get_return_on_tangible_assets(rounding=None)
    tav = ratios_module.get_tangible_asset_value(rounding=None)

    assert rota.loc["MSFT"].loc[YEAR] == _approx(
        net_income.iloc[year] / tangible_assets
    )
    assert tav.loc["MSFT"].loc[YEAR] == _approx(
        total_assets.iloc[year] - liabilities.iloc[year] - intangibles.iloc[year]
    )


def test_enterprise_value_deducts_cash_and_short_term_investments(ratios_module):
    balance = ratios_module._balance_sheet_statement
    market_cap = ratios_module.get_market_cap(rounding=None)
    enterprise_value = ratios_module.get_enterprise_value(rounding=None)

    def value(item):
        return _row(balance, item).loc[YEAR]

    expected = (
        market_cap.loc["AAPL"].loc[YEAR]
        + value("Total Debt")
        + value("Minority Interest")
        + value("Preferred Stock")
        - value("Cash and Cash Equivalents")
        - value("Short Term Investments")
    )

    assert enterprise_value.loc["AAPL"].loc[YEAR] == _approx(expected)


def test_ev_to_ebit_uses_operating_income(ratios_module):
    enterprise_value = ratios_module.get_enterprise_value(rounding=None)
    ev_to_ebit = ratios_module.get_ev_to_ebit(rounding=None)
    operating_income = _row(ratios_module._income_statement, "Operating Income")

    assert ev_to_ebit.loc["AAPL"].loc[YEAR] == _approx(
        enterprise_value.loc["AAPL"].loc[YEAR] / operating_income.loc[YEAR]
    )


def test_free_cash_flow_yield_with_trailing_uses_the_period_end_price(ratios_module):
    market_cap = ratios_module.get_market_cap(rounding=None)
    free_cash_flow = _row(ratios_module._cash_flow_statement, "Free Cash Flow")
    year = free_cash_flow.index.get_loc(
        next(p for p in free_cash_flow.index if str(p) == YEAR)
    )
    result = ratios_module.get_free_cash_flow_yield(trailing=2, rounding=None)

    assert result.loc["AAPL"].loc[YEAR] == _approx(
        free_cash_flow.iloc[year - 1 : year + 1].sum()
        / market_cap.loc["AAPL"].loc[YEAR]
    )


def test_map_period_data_to_daily_data_does_not_look_ahead():
    from financetoolkit.ratios.helpers import map_period_data_to_daily_data

    period_data = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=["XYZ"],
        columns=pd.period_range("2021Q1", periods=3, freq="Q"),
    )
    daily_dates = pd.period_range("2021-01-01", "2021-10-15", freq="D")

    daily = map_period_data_to_daily_data(period_data, daily_dates, quarterly=True)[
        "XYZ"
    ]

    # Nothing before the first quarter has closed, then each quarter from its last day on.
    assert daily.loc[: pd.Period("2021-03-30", "D")].isna().all()
    assert daily.loc[pd.Period("2021-03-31", "D")] == 1
    assert daily.loc[pd.Period("2021-06-29", "D")] == 1
    assert daily.loc[pd.Period("2021-06-30", "D")] == 2
    assert daily.loc[pd.Period("2021-10-15", "D")] == 3


def _quarterly_ratios():
    from financetoolkit.ratios.ratios_controller import Ratios

    ticker = "XYZ"
    quarters = pd.period_range("2021Q1", periods=12, freq="Q")
    growth = 1.03 ** np.arange(12)

    def statement(items):
        return pd.DataFrame(
            list(items.values()),
            index=pd.MultiIndex.from_tuples([(ticker, item) for item in items]),
            columns=quarters,
        )

    income = statement(
        {
            "Revenue": 100 * growth,
            "Net Income": 10 * growth,
            "Operating Income": 15 * growth,
            "Weighted Average Shares Diluted": np.full(12, 10.0),
            "Weighted Average Shares": np.full(12, 10.0),
        }
    )
    balance = statement(
        {
            "Total Debt": 50 * growth,
            "Minority Interest": np.zeros(12),
            "Preferred Stock": np.zeros(12),
            "Cash and Cash Equivalents": 10 * growth,
        }
    )
    cash = statement(
        {
            "Free Cash Flow": 8 * growth,
            "Depreciation and Amortization": 3 * growth,
            "Preferred Dividends Paid": np.zeros(12),
        }
    )

    days = pd.bdate_range("2021-01-01", "2023-12-29")
    dividends = pd.Series(0.0, index=days)
    for quarter in quarters:
        dividends[
            days[days.searchsorted(quarter.start_time + pd.Timedelta(days=45))]
        ] = 0.2
    daily_index = pd.PeriodIndex(days, freq="D")
    daily = pd.concat(
        {
            "Close": pd.DataFrame({ticker: 50.0}, index=daily_index),
            "Dividends": pd.DataFrame(
                {ticker: dividends.to_numpy()}, index=daily_index
            ),
        },
        axis=1,
    )
    period = pd.concat(
        {
            "Close": pd.DataFrame({ticker: 50.0}, index=quarters),
            "Dividends": pd.DataFrame({ticker: 0.2}, index=quarters),
        },
        axis=1,
    )

    return (
        Ratios(
            [ticker],
            {"period": period, "daily": daily},
            balance,
            income,
            cash,
            quarterly=True,
            rounding=None,
        ),
        income,
    )


def test_quarterly_valuation_ratios_default_to_the_trailing_twelve_months():
    ratios, income = _quarterly_ratios()
    ttm_revenue = income.loc["XYZ"].loc["Revenue"].iloc[-4:].sum()

    price_to_earnings = ratios.get_price_to_earnings_ratio().iloc[0, -1]

    assert ratios.get_earnings_yield().iloc[0, -1] == pytest.approx(
        1 / price_to_earnings
    )
    assert ratios.get_price_to_sales_ratio().iloc[0, -1] == pytest.approx(
        50 * 10 / ttm_revenue
    )
    assert ratios.get_dividend_yield().iloc[0, -1] == pytest.approx(0.8 / 50)


def test_quarterly_peg_compares_with_four_quarters_earlier():
    ratios, income = _quarterly_ratios()
    ttm_eps = income.loc["XYZ"].loc["Net Income"].rolling(4).sum() / 10
    yearly_growth = ttm_eps.iloc[-1] / ttm_eps.iloc[-5] - 1
    price_to_earnings = ratios.get_price_to_earnings_ratio().iloc[0, -1]

    assert ratios.get_price_to_earnings_growth_ratio().iloc[0, -1] == pytest.approx(
        price_to_earnings / (yearly_growth * 100), rel=1e-3
    )


def test_daily_ratios_sum_flows_over_periods_not_days():
    ratios, _ = _quarterly_ratios()
    daily = ratios.get_price_to_sales_ratio(show_daily=True)["XYZ"]
    period = ratios.get_price_to_sales_ratio()

    # 2023-09-29 is the last trading day before the end of 2023Q3 (a Saturday): it still
    # uses the figures up to 2023Q2. From the next trading day on, 2023Q3 applies.
    assert daily.loc[pd.Period("2023-09-29", "D")] == pytest.approx(
        period.loc["XYZ", pd.Period("2023Q2", "Q")]
    )
    assert daily.loc[pd.Period("2023-10-02", "D")] == pytest.approx(
        period.loc["XYZ", pd.Period("2023Q3", "Q")]
    )


def test_daily_dividend_yield_sums_a_year_of_dividends():
    ratios, _ = _quarterly_ratios()
    daily = ratios.get_dividend_yield(show_daily=True)["XYZ"].dropna()

    # Four quarterly dividends of 0.2 over a constant price of 50, on every day.
    assert (daily > 0).all()
    assert daily.median() == pytest.approx(0.8 / 50)


def test_tax_rates_are_nan_for_pre_tax_losses(ratios_module):
    from financetoolkit.ratios.ratios_controller import Ratios

    income = ratios_module._income_statement.copy()
    income.loc[("AAPL", "Income Before Tax"), YEAR] = -1e9
    ratios = Ratios(
        ratios_module._tickers,
        {
            "period": ratios_module._historical_data,
            "daily": ratios_module._daily_historical_data,
        },
        ratios_module._balance_sheet_statement,
        income,
        ratios_module._cash_flow_statement,
        rounding=None,
    )

    assert pd.isna(ratios.get_effective_tax_rate().loc["AAPL"].loc[YEAR])
    assert pd.isna(ratios.get_cash_tax_rate().loc["AAPL"].loc[YEAR])
    assert ratios.get_effective_tax_rate().loc["MSFT"].notna().all()


def test_map_period_data_to_daily_data_starts_at_the_fiscal_period_end():
    from financetoolkit.ratios.helpers import map_period_data_to_daily_data

    # A fiscal year ending in January is labelled with the calendar year before it.
    periods = pd.PeriodIndex(["2022", "2023"], freq="Y")
    period_data = pd.DataFrame(
        [[1.0, 2.0], [1.0, 2.0]], index=["NVDA", "XYZ"], columns=periods
    )
    period_end_dates = pd.DataFrame(
        [pd.to_datetime(["2023-01-29", "2024-01-28"])], index=["NVDA"], columns=periods
    )
    daily_dates = pd.period_range("2023-12-28", "2024-02-02", freq="D")

    daily = map_period_data_to_daily_data(
        period_data, daily_dates, period_end_dates=period_end_dates
    )

    # NVDA's fiscal 2023 figures apply from 28 January 2024, not from 31 December 2023.
    assert daily.loc[pd.Period("2024-01-27", "D"), "NVDA"] == 1
    assert daily.loc[pd.Period("2024-01-28", "D"), "NVDA"] == 2
    # A ticker without end dates uses the end of the label.
    assert daily.loc[pd.Period("2023-12-31", "D"), "XYZ"] == 2


def test_debt_service_coverage_ratio_scales_the_principal_to_the_window(ratios_module):
    income = ratios_module._income_statement.loc["AAPL"]
    balance = ratios_module._balance_sheet_statement.loc["AAPL"]
    cash = ratios_module._cash_flow_statement.loc["AAPL"]

    result = ratios_module.get_debt_service_coverage_ratio(trailing=2, rounding=None)

    window = lambda frame, item: frame.loc[item].rolling(2)  # noqa: E731
    # Two years of earnings and interest against two years of the principal due in a year.
    expected = (
        window(income, "Operating Income").sum()
        + window(cash, "Depreciation and Amortization").sum()
    ) / (
        window(income, "Interest Expense").sum()
        + window(balance, "Short Term Debt").mean() * 2
    )

    assert result.loc["AAPL"].loc[YEAR] == _approx(expected.loc[YEAR])


def test_quarterly_forward_ratios_use_a_year_of_estimates():
    ratios, income = _quarterly_ratios()
    estimate_quarters = pd.period_range("2024Q1", periods=6, freq="Q")
    estimates = pd.DataFrame(
        [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]],
        index=pd.MultiIndex.from_tuples([("XYZ", "Estimated EPS Average")]),
        columns=estimate_quarters,
    )
    ratios._analyst_estimates_cache["data"] = estimates
    eps = income.loc["XYZ"].loc["Net Income"] / 10
    ratios._income_statement = pd.concat(
        [
            income,
            pd.DataFrame(
                [eps.to_numpy()],
                index=pd.MultiIndex.from_tuples([("XYZ", "EPS Diluted")]),
                columns=income.columns,
            ),
        ]
    )

    forward = ratios.get_forward_price_earnings_ratio(rounding=None).loc["XYZ"]
    forward_peg = ratios.get_forward_price_earnings_growth_ratio(rounding=None).loc[
        "XYZ"
    ]

    # The price over the estimates of the quarter and the next three.
    assert forward[pd.Period("2024Q1", "Q")] == pytest.approx(50 / 4.6)
    assert forward[pd.Period("2024Q2", "Q")] == pytest.approx(50 / 5.0)
    assert pd.isna(forward[pd.Period("2024Q4", "Q")])

    # The next twelve months against the last four reported quarters, a year apart.
    growth = 4.6 / eps.iloc[-4:].sum() - 1
    assert forward_peg[pd.Period("2024Q1", "Q")] == pytest.approx(
        (50 / 4.6) / (growth * 100)
    )


def test_ratios_call_only_the_loaders_a_ratio_needs(ratios_module):
    from financetoolkit.helpers import LazyDict
    from financetoolkit.ratios.ratios_controller import Ratios

    loaded = []

    def loader(name, data):
        def load():
            loaded.append(name)
            return data

        return load

    historical = {
        "period": ratios_module._historical_data,
        "daily": ratios_module._daily_historical_data,
    }
    ratios = Ratios(
        loader("tickers", ratios_module._tickers),
        LazyDict(lambda key: loader(key, historical[key])()),
        loader("balance", ratios_module._balance_sheet_statement),
        loader("income", ratios_module._income_statement),
        loader("cash", ratios_module._cash_flow_statement),
    )

    assert loaded == []

    ratios.get_gross_margin()
    assert sorted(loaded) == ["income", "tickers"]

    ratios.get_price_to_earnings_ratio()
    assert sorted(loaded) == ["income", "period", "tickers"]


def test_collected_z_scores_equal_those_of_each_ratio(ratios_module, monkeypatch):
    # Both describe the periods shown, so a later start date gives the same Z-Scores.
    periods = ratios_module.get_gross_margin().columns
    monkeypatch.setattr(ratios_module, "_start_date", str(periods[2]))

    collected = ratios_module.collect_profitability_ratios(standardize=True)
    single = ratios_module.get_gross_margin(standardize=True)

    ticker = single.index[0]
    pd.testing.assert_series_equal(
        collected.loc[(ticker, "Gross Margin")],
        single.loc[ticker],
        check_names=False,
    )
    assert collected.columns[0] == periods[2]


def test_daily_growth_and_standardization_run_over_time(ratios_module):
    daily = ratios_module.get_price_to_earnings_ratio(show_daily=True, rounding=None)
    growth = ratios_module.get_price_to_earnings_ratio(
        show_daily=True, growth=True, rounding=None
    )

    expected = daily.pct_change(fill_method=None)
    pd.testing.assert_frame_equal(
        growth.dropna(how="all"),
        expected.loc[growth.dropna(how="all").index],
        atol=1e-3,
        check_freq=False,
    )

    standardized = ratios_module.get_price_to_earnings_ratio(
        show_daily=True, standardize=True
    )
    # A Z-Score over time, not across two tickers (which would be +/-0.7071 every day).
    assert standardized.abs().round(4).ne(0.7071).any().all()


def test_collect_ratios_growth_with_one_ticker():
    from financetoolkit import Toolkit

    toolkit = Toolkit(
        tickers=["AAPL"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        balance=pd.read_pickle("tests/datasets/balance_dataset.pickle"),
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    assert not toolkit.ratios.collect_profitability_ratios(growth=True).empty
    assert not toolkit.ratios.collect_profitability_ratios(
        growth=True, lag=[1, 2]
    ).empty


def test_collect_all_ratios_groups_the_rows_per_ticker(ratios_module):
    tickers = ratios_module.collect_all_ratios().index.get_level_values(0)

    assert (tickers[1:] != tickers[:-1]).sum() == 1


def test_compound_growth_of_a_ratio_is_the_cagr_of_its_values(ratios_module):
    values = ratios_module.get_revenue_per_share(rounding=None)
    cagr = ratios_module.get_revenue_per_share(
        growth=True, lag=2, compound=True, rounding=None
    )
    collected = ratios_module.collect_valuation_ratios(
        growth=True, lag=2, compound=True, rounding=None
    )

    periods = values.columns
    expected = (values[periods[-1]] / values[periods[-3]]) ** (1 / 2) - 1
    expected = expected.where((values[periods[-1]] > 0) & (values[periods[-3]] > 0))

    # The growth is rounded to the Toolkit's four decimals.
    pd.testing.assert_series_equal(
        cagr[periods[-1]], expected, check_names=False, atol=1e-4
    )
    for ticker in values.index:
        assert collected.loc[
            (ticker, "Revenue per Share"), periods[-1]
        ] == pytest.approx(expected[ticker], nan_ok=True, abs=1e-4)


def test_credit_ratios_from_the_statement_lines(ratios_module):
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    cash = ratios_module._cash_flow_statement

    def flow(statement, item):
        return _row(statement, item).loc[YEAR]

    ebitda = flow(income, "Operating Income") + flow(
        cash, "Depreciation and Amortization"
    )
    interest_paid = flow(cash, "Interest Paid")
    taxes_paid = flow(cash, "Income Taxes Paid")
    total_debt = flow(balance, "Total Debt")
    funds_from_operations = (
        ebitda + flow(cash, "Stock Based Compensation") - interest_paid - taxes_paid
    )
    discretionary_cash_flow = (
        flow(cash, "Cash Flow from Operations")
        - abs(flow(cash, "Capital Expenditure"))
        - abs(flow(cash, "Dividends Paid"))
        - abs(flow(cash, "Common Stock Purchased"))
    )

    expected = {
        "get_ebitda_interest_coverage_ratio": ebitda / flow(income, "Interest Expense"),
        "get_ebitda_minus_capex_interest_coverage_ratio": (
            ebitda - abs(flow(cash, "Capital Expenditure"))
        )
        / flow(income, "Interest Expense"),
        "get_cash_interest_coverage_ratio": (
            flow(cash, "Cash Flow from Operations") + interest_paid + taxes_paid
        )
        / interest_paid,
        "get_funds_from_operations_to_debt_ratio": funds_from_operations / total_debt,
        "get_discretionary_cash_flow_to_debt_ratio": discretionary_cash_flow
        / total_debt,
        "get_free_cash_flow_payout_ratio": abs(flow(cash, "Dividends Paid"))
        / flow(cash, "Free Cash Flow"),
    }

    for method, value in expected.items():
        result = getattr(ratios_module, method)(rounding=None)
        assert result.loc["AAPL"].loc[YEAR] == _approx(value), method


def test_degrees_of_leverage_compare_period_over_period_changes(ratios_module):
    income = ratios_module._income_statement
    previous = str(int(YEAR) - 1)

    def change(item):
        return (_row(income, item).loc[YEAR] - _row(income, item).loc[previous]) / abs(
            _row(income, item).loc[previous]
        )

    operating = change("Operating Income") / change("Revenue")
    financial = change("Net Income") / change("Operating Income")

    assert ratios_module.get_degree_of_operating_leverage(rounding=None).loc[
        "AAPL"
    ].loc[YEAR] == _approx(operating)
    assert ratios_module.get_degree_of_financial_leverage(rounding=None).loc[
        "AAPL"
    ].loc[YEAR] == _approx(financial)
    # The degree of total leverage is the product of the other two.
    assert ratios_module.get_degree_of_total_leverage(rounding=None).loc["AAPL"].loc[
        YEAR
    ] == _approx(operating * financial)


def test_book_multiples_from_the_statement_lines(ratios_module):
    balance = ratios_module._balance_sheet_statement
    market_cap = ratios_module.get_market_cap(rounding=None).loc["AAPL"].loc[YEAR]
    enterprise_value = (
        ratios_module.get_enterprise_value(rounding=None).loc["AAPL"].loc[YEAR]
    )

    def item(name):
        return _row(balance, name).loc[YEAR]

    invested_capital = (
        item("Total Equity")
        + item("Total Debt")
        - item("Cash and Cash Equivalents")
        - item("Short Term Investments")
    )
    tangible_book_value = (
        item("Total Shareholder Equity")
        - item("Preferred Stock")
        - item("Goodwill and Intangible Assets")
    )

    assert ratios_module.get_ev_to_invested_capital_ratio(rounding=None).loc[
        "AAPL"
    ].loc[YEAR] == _approx(enterprise_value / invested_capital)
    assert ratios_module.get_price_to_tangible_book_ratio(rounding=None).loc[
        "AAPL"
    ].loc[YEAR] == _approx(market_cap / tangible_book_value)


def assert_rounded_equal(result, expected, **kwargs):
    """The Toolkit rounds to four decimals, so a value is within half a unit of the fourth."""
    kwargs.pop("check_names", None)
    assertion = (
        pd.testing.assert_series_equal
        if isinstance(result, pd.Series)
        else pd.testing.assert_frame_equal
    )
    assertion(result, expected, check_names=False, check_exact=False, rtol=0, atol=5e-5)


def _item(statement: pd.DataFrame, item: str, columns) -> pd.DataFrame:
    """A statement row per ticker over the columns of a ratio, unrounded."""
    return statement.loc[:, item, :].loc[:, columns].astype(float)


@pytest.mark.parametrize(
    ("method", "numerator", "numerator_statement"),
    [
        (
            "get_research_and_development_ratio",
            "Research and Development Expenses",
            "_income_statement",
        ),
        (
            "get_selling_and_marketing_ratio",
            "Selling and Marketing Expenses",
            "_income_statement",
        ),
        (
            "get_general_and_administrative_ratio",
            "General and Administrative Expenses",
            "_income_statement",
        ),
        (
            "get_stock_based_compensation_ratio",
            "Stock Based Compensation",
            "_cash_flow_statement",
        ),
        ("get_deferred_revenue_ratio", "Deferred Revenue", "_balance_sheet_statement"),
    ],
)
def test_expense_ratios_divide_the_item_by_revenue(
    ratios_module, method, numerator, numerator_statement
):
    result = getattr(ratios_module, method)(rounding=None)
    columns = result.columns

    expected = _item(
        getattr(ratios_module, numerator_statement), numerator, columns
    ) / _item(ratios_module._income_statement, "Revenue", columns)

    assert not result.empty
    assert_rounded_equal(result, expected, check_names=False)


def test_expense_ratios_sum_flows_over_the_trailing_window(ratios_module):
    result = ratios_module.get_research_and_development_ratio(trailing=2, rounding=None)
    income = ratios_module._income_statement
    research = income.loc[:, "Research and Development Expenses", :]
    revenue = income.loc[:, "Revenue", :]

    last, previous = result.columns[-1], result.columns[-2]
    expected = (research[last] + research[previous]) / (
        revenue[last] + revenue[previous]
    )

    assert_rounded_equal(result[last], expected.astype(float), check_names=False)


def test_debt_to_capital_is_debt_over_debt_plus_equity(ratios_module):
    result = ratios_module.get_debt_to_capital_ratio(rounding=None)
    balance = ratios_module._balance_sheet_statement
    debt = _item(balance, "Total Debt", result.columns)
    equity = _item(balance, "Total Equity", result.columns)

    assert_rounded_equal(result, debt / (debt + equity), check_names=False)
    # Debt over all capital stays a share as long as the equity is positive.
    assert ((result > 0) & (result < 1))[equity > 0].fillna(True).all().all()


def test_interest_paid_to_expense_compares_cash_with_accrual(ratios_module):
    result = ratios_module.get_interest_paid_to_expense_ratio(rounding=None)
    expected = _item(
        ratios_module._cash_flow_statement, "Interest Paid", result.columns
    ) / _item(ratios_module._income_statement, "Interest Expense", result.columns)

    assert_rounded_equal(result, expected, check_names=False)


def test_tax_rate_divergence_is_cash_minus_effective_tax_rate(ratios_module):
    result = ratios_module.get_tax_rate_divergence(rounding=None)
    expected = ratios_module.get_cash_tax_rate(
        rounding=None
    ) - ratios_module.get_effective_tax_rate(rounding=None)

    assert_rounded_equal(result, expected.loc[:, result.columns])


def test_buyback_yield_is_net_repurchases_over_market_cap(ratios_module):
    result = ratios_module.get_buyback_yield(rounding=None)
    cash_flow = ratios_module._cash_flow_statement
    purchased = _item(cash_flow, "Common Stock Purchased", result.columns)
    issued = _item(cash_flow, "Common Stock Issued", result.columns)
    market_cap = ratios_module.get_market_cap().loc[:, result.columns]

    assert_rounded_equal(result, -(purchased + issued) / market_cap, check_names=False)
    # Apple and Microsoft bought back far more stock than they issued in every year with
    # a market capitalisation (the last period lies past the end date of the prices).
    assert (result.dropna(axis=1, how="all") > 0).all().all()


def test_shareholder_yield_adds_dividend_and_buyback_yield(ratios_module):
    result = ratios_module.get_shareholder_yield(rounding=None)
    dividend_yield = ratios_module.get_dividend_yield(rounding=None)
    buyback_yield = ratios_module.get_buyback_yield(rounding=None)

    expected = (
        dividend_yield.loc[result.index, result.columns]
        + buyback_yield.loc[result.index, result.columns]
    )

    assert_rounded_equal(result, expected)


def test_ev_to_free_cash_flow_divides_enterprise_value_by_free_cash_flow(
    ratios_module,
):
    result = ratios_module.get_ev_to_free_cash_flow_ratio(rounding=None)
    enterprise_value = ratios_module.get_enterprise_value().loc[:, result.columns]
    free_cash_flow = _item(
        ratios_module._cash_flow_statement, "Free Cash Flow", result.columns
    )

    assert_rounded_equal(result, enterprise_value / free_cash_flow, check_names=False)


def test_sbc_adjusted_free_cash_flow_deducts_stock_based_compensation(
    ratios_module,
):
    result = ratios_module.get_sbc_adjusted_free_cash_flow(rounding=None)
    cash_flow = ratios_module._cash_flow_statement
    expected = _item(cash_flow, "Free Cash Flow", result.columns) - _item(
        cash_flow, "Stock Based Compensation", result.columns
    )

    assert_rounded_equal(result, expected, check_names=False)
    assert (result < _item(cash_flow, "Free Cash Flow", result.columns)).all().all()


def test_get_gross_profitability(recorder, ratios_module):
    recorder.capture(ratios_module.get_gross_profitability())
    recorder.capture(ratios_module.get_gross_profitability(trailing=2))


def test_get_operating_profitability(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_profitability())
    recorder.capture(ratios_module.get_operating_profitability(growth=True))


def test_get_cash_based_operating_profitability(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_based_operating_profitability())
    recorder.capture(ratios_module.get_cash_based_operating_profitability(trailing=2))


def test_get_cash_return_on_invested_capital(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_return_on_invested_capital())
    recorder.capture(ratios_module.get_cash_return_on_invested_capital(trailing=2))


def test_get_free_cash_flow_conversion(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_conversion())
    recorder.capture(ratios_module.get_free_cash_flow_conversion(trailing=2))


def test_get_free_cash_flow_to_ebitda_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_to_ebitda_ratio())
    recorder.capture(ratios_module.get_free_cash_flow_to_ebitda_ratio(trailing=2))


def test_get_capex_intensity(recorder, ratios_module):
    recorder.capture(ratios_module.get_capex_intensity())
    recorder.capture(ratios_module.get_capex_intensity(trailing=2))


def test_get_capex_to_depreciation_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_capex_to_depreciation_ratio())
    recorder.capture(ratios_module.get_capex_to_depreciation_ratio(trailing=2))


def test_get_asset_growth(recorder, ratios_module):
    recorder.capture(ratios_module.get_asset_growth())
    recorder.capture(ratios_module.get_asset_growth(trailing=2))


def test_get_net_operating_assets(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_operating_assets())
    recorder.capture(ratios_module.get_net_operating_assets(growth=True))


def test_get_percent_accruals(recorder, ratios_module):
    recorder.capture(ratios_module.get_percent_accruals())
    recorder.capture(ratios_module.get_percent_accruals(trailing=2))


def test_get_net_share_issuance(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_share_issuance())
    recorder.capture(ratios_module.get_net_share_issuance(diluted=True, trailing=2))


def test_get_composite_equity_issuance(recorder, ratios_module):
    recorder.capture(ratios_module.get_composite_equity_issuance(trailing=1))
    recorder.capture(
        ratios_module.get_composite_equity_issuance(diluted=True, trailing=2)
    )


def test_get_net_external_financing(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_external_financing())
    recorder.capture(ratios_module.get_net_external_financing(trailing=2))


def test_investment_and_issuance_anomalies_from_the_statement_lines(ratios_module):
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    cash = ratios_module._cash_flow_statement
    previous = str(int(YEAR) - 1)

    def at(statement, item, year=YEAR):
        return _row(statement, item).loc[year]

    total_assets = at(balance, "Total Assets")
    beginning_total_assets = at(balance, "Total Assets", previous)

    assert ratios_module.get_asset_growth(rounding=8).loc["AAPL", YEAR] == _approx(
        total_assets / beginning_total_assets - 1
    )
    # Operating assets minus operating liabilities is the debt and equity minus the cash.
    assert ratios_module.get_net_operating_assets(rounding=8).loc[
        "AAPL", YEAR
    ] == _approx(
        (
            at(balance, "Short Term Debt")
            + at(balance, "Long Term Debt")
            + at(balance, "Total Equity")
            - at(balance, "Cash and Short Term Investments")
        )
        / beginning_total_assets
    )
    net_income = at(income, "Net Income")
    assert ratios_module.get_percent_accruals(rounding=8).loc["AAPL", YEAR] == _approx(
        (net_income - at(cash, "Operating Cash Flow")) / abs(net_income)
    )
    assert ratios_module.get_net_share_issuance(rounding=8).loc[
        "AAPL", YEAR
    ] == _approx(
        np.log(
            at(income, "Weighted Average Shares")
            / at(income, "Weighted Average Shares", previous)
        )
    )
    # Bradshaw, Richardson and Sloan: equity issued minus repurchased minus dividends, plus
    # debt issued minus repaid, over the average total assets (outflows are negative).
    assert ratios_module.get_net_external_financing(rounding=8).loc[
        "AAPL", YEAR
    ] == _approx(
        (
            at(cash, "Common Stock Issued")
            + at(cash, "Common Stock Purchased")
            + at(cash, "Net Preferred Stock Issued")
            + at(cash, "Dividends Paid")
            + at(cash, "Long Term Debt Issued")
            + at(cash, "Short Term Debt Issued")
        )
        / ((total_assets + beginning_total_assets) / 2)
    )


def test_composite_equity_issuance_is_market_cap_growth_minus_the_log_return(
    ratios_module,
):
    prices = ratios_module._historical["yearly"]
    income = ratios_module._income_statement
    previous = str(int(YEAR) - 1)

    def price(column, year):
        return prices.loc[year, (column, "AAPL")]

    shares = _row(income, "Weighted Average Shares")
    market_cap_growth = (price("Close", YEAR) * shares.loc[YEAR]) / (
        price("Close", previous) * shares.loc[previous]
    )

    result = ratios_module.get_composite_equity_issuance(trailing=1, rounding=8)

    assert result.loc["AAPL", YEAR] == _approx(
        np.log(market_cap_growth)
        - np.log(price("Adj Close", YEAR) / price("Adj Close", previous))
    )
    # The default window is five years, longer than the test data.
    assert ratios_module.get_composite_equity_issuance().isna().all().all()


def test_quarterly_investment_and_issuance_anomalies_are_year_over_year():
    from financetoolkit.ratios.ratios_controller import Ratios

    quarters = pd.period_range("2021Q1", periods=8, freq="Q")
    growth = 1.05 ** np.arange(8)

    def statement(items):
        return pd.DataFrame(
            list(items.values()),
            index=pd.MultiIndex.from_tuples([("XYZ", item) for item in items]),
            columns=quarters,
        )

    income = statement(
        {
            "Net Income": 10 * growth,
            "Weighted Average Shares": 100 * 0.99 ** np.arange(8),
        }
    )
    balance = statement(
        {
            "Total Assets": 1000 * growth,
            "Cash and Short Term Investments": 100 * growth,
            "Short Term Debt": 50 * growth,
            "Long Term Debt": 200 * growth,
            "Total Equity": 400 * growth,
        }
    )
    cash = statement(
        {
            "Operating Cash Flow": 12 * growth,
            "Common Stock Issued": np.full(8, 1.0),
            "Common Stock Purchased": np.full(8, -5.0),
            "Net Preferred Stock Issued": np.zeros(8),
            "Dividends Paid": np.full(8, -2.0),
            "Long Term Debt Issued": np.full(8, 3.0),
            "Short Term Debt Issued": np.zeros(8),
        }
    )
    ratios = Ratios(["XYZ"], {}, balance, income, cash, quarterly=True, rounding=None)

    total_assets = balance.loc[("XYZ", "Total Assets")]
    shares = income.loc[("XYZ", "Weighted Average Shares")]

    def trailing_year(statement_data, item):
        return statement_data.loc[("XYZ", item)].iloc[-4:].sum()

    assert ratios.get_asset_growth().iloc[0, -1] == pytest.approx(
        total_assets.iloc[-1] / total_assets.iloc[-5] - 1
    )
    assert ratios.get_net_share_issuance().iloc[0, -1] == pytest.approx(
        np.log(shares.iloc[-1] / shares.iloc[-5])
    )
    assert ratios.get_percent_accruals().iloc[0, -1] == pytest.approx(
        (
            trailing_year(income, "Net Income")
            - trailing_year(cash, "Operating Cash Flow")
        )
        / trailing_year(income, "Net Income")
    )
    # A year of financing flows over the average of the total assets a year apart.
    assert ratios.get_net_external_financing().iloc[0, -1] == pytest.approx(
        4 * (1 - 5 - 2 + 3) / ((total_assets.iloc[-1] + total_assets.iloc[-5]) / 2)
    )
    # Net operating assets are a balance, scaled by the total assets of the quarter before.
    assert ratios.get_net_operating_assets().iloc[0, -1] == pytest.approx(
        (50 + 200 + 400 - 100) * growth[-1] / total_assets.iloc[-2]
    )
    assert ratios.get_asset_growth().iloc[0, :4].isna().all()


def test_cash_based_operating_profitability_from_the_statement_lines(ratios_module):
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    previous = str(int(YEAR) - 1)

    def change(item):
        return _row(balance, item).loc[YEAR] - _row(balance, item).loc[previous]

    operating_profit = (
        _row(income, "Revenue").loc[YEAR]
        - _row(income, "Cost of Goods Sold").loc[YEAR]
        - _row(income, "Selling, General and Administrative Expenses").loc[YEAR]
    )
    accruals = (
        change("Net Receivables")
        + change("Inventory")
        + change("Prepaids")
        - change("Deferred Revenue")
        - change("Deferred Revenue Non Current")
        - change("Accounts Payable")
        - change("Accrued Expenses")
    )

    result = ratios_module.get_cash_based_operating_profitability(rounding=None)

    assert result.loc["AAPL"].loc[YEAR] == _approx(
        # Scaled by the total assets at the start of the year, as in Ball et al. (2016).
        (operating_profit - accruals)
        / _row(balance, "Total Assets").loc[previous]
    )


def test_quarterly_profitability_anomalies_default_to_the_trailing_year():
    from financetoolkit.ratios.ratios_controller import Ratios

    quarters = pd.period_range("2021Q1", periods=8, freq="Q")
    growth = 1.05 ** np.arange(8)

    def statement(items):
        return pd.DataFrame(
            list(items.values()),
            index=pd.MultiIndex.from_tuples([("XYZ", item) for item in items]),
            columns=quarters,
        )

    zeros = np.zeros(8)
    income = statement(
        {
            "Revenue": 100 * growth,
            "Cost of Goods Sold": 40 * growth,
            "Selling, General and Administrative Expenses": 20 * growth,
            "Research and Development Expenses": 5 * growth,
            "Interest Expense": np.full(8, 1.0),
        }
    )
    balance = statement(
        {
            "Total Assets": 1000 * growth,
            "Total Shareholder Equity": 400 * growth,
            "Deferred Tax Liabilities": zeros,
            "Preferred Stock": zeros,
            "Net Receivables": 30 * growth,
            "Inventory": zeros,
            "Prepaids": zeros,
            "Deferred Revenue": zeros,
            "Deferred Revenue Non Current": zeros,
            "Accounts Payable": zeros,
            "Accrued Expenses": zeros,
        }
    )
    ratios = Ratios(
        ["XYZ"], {}, balance, income, pd.DataFrame(), quarterly=True, rounding=None
    )

    def trailing_year(item):
        return income.loc[("XYZ", item)].iloc[-4:].sum()

    total_assets = balance.loc[("XYZ", "Total Assets")].iloc[-1]
    gross_profit = trailing_year("Revenue") - trailing_year("Cost of Goods Sold")
    receivables = balance.loc[("XYZ", "Net Receivables")]

    assert ratios.get_gross_profitability().iloc[0, -1] == pytest.approx(
        gross_profit / total_assets
    )
    assert ratios.get_operating_profitability().iloc[0, -1] == pytest.approx(
        (
            gross_profit
            - trailing_year("Selling, General and Administrative Expenses")
            - trailing_year("Research and Development Expenses")
            - trailing_year("Interest Expense")
        )
        / balance.loc[("XYZ", "Total Shareholder Equity")].iloc[-1]
    )
    # The receivables change over the same four quarters as the flows, and the total
    # assets are those at the start of them.
    assert ratios.get_cash_based_operating_profitability().iloc[0, -1] == pytest.approx(
        (
            gross_profit
            - trailing_year("Selling, General and Administrative Expenses")
            - (receivables.iloc[-1] - receivables.iloc[-5])
        )
        / balance.loc[("XYZ", "Total Assets")].iloc[-5]
    )
    # The first three quarters have no full trailing year.
    assert ratios.get_gross_profitability().iloc[0, :3].isna().all()


def test_a_zero_denominator_gives_nan_and_keeps_the_z_scores():
    from financetoolkit import Toolkit

    balance = pd.read_pickle("tests/datasets/balance_dataset.pickle")
    income = pd.read_pickle("tests/datasets/income_dataset.pickle")
    balance.loc[("AAPL", "Total Equity")] = [63.09e9, 0.0, 50.672e9, 62.146e9]
    income.loc[("MSFT", "Revenue")] = [0.0, 100.0, np.nan, 200.0]

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        balance=balance,
        income=income,
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )

    debt_to_equity = toolkit.ratios.get_debt_to_equity_ratio()
    standardized = toolkit.ratios.get_debt_to_equity_ratio(standardize=True)
    gross_margin = toolkit.ratios.get_gross_margin()

    assert debt_to_equity.loc["AAPL"].isna().tolist() == [False, True, False, False]
    # The undefined year no longer turns every Z-Score of the ticker into NaN.
    assert standardized.loc["AAPL"].isna().tolist() == [False, True, False, False]
    assert np.isnan(gross_margin.loc["MSFT"].iloc[0])


def test_a_custom_ratio_with_a_zero_denominator_is_nan(ratios_module):
    custom_ratios = copy.deepcopy(ratios_module).collect_custom_ratios(
        custom_ratios_dict={"Revenue over Nothing": "Revenue / 0"}
    )

    values = custom_ratios.xs("Revenue over Nothing", level=1)
    assert values.isna().all().all()
