"""Ratios Controller Tests""" ""

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


def test_get_free_cash_flow_operating_cash_flow_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_operating_cash_flow_ratio())


def test_get_tax_burden_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_tax_burden_ratio())


def test_get_interest_coverage_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_interest_coverage_ratio())


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


def test_get_retention_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_retention_ratio())


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

YEAR = "2022"


def _approx(expected):
    # The Toolkit rounds to 4 decimals by default, so compare with a relative tolerance.
    return pytest.approx(expected, rel=1e-3)


def _row(statement, item, ticker="AAPL"):
    return statement.loc[(ticker, item)]


def test_days_ratios_cover_the_whole_trailing_window(ratios_module):
    # With trailing=2 on annual data, COGS covers two years, so the window is 730 days, and
    # the inventory is averaged over the three year ends that open and close those years.
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    inventory = _row(balance, "Inventory")
    cogs = _row(income, "Cost of Goods Sold")
    year = inventory.index.get_loc(next(p for p in inventory.index if str(p) == YEAR))

    expected = (
        inventory.iloc[year - 2 : year + 1].mean()
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


def test_forward_ratios_leave_out_periods_with_too_few_analysts():
    ratios, _ = _quarterly_ratios()
    estimate_quarters = pd.period_range("2024Q1", periods=6, freq="Q")
    estimates = pd.DataFrame(
        [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5], [12, 12, 12, 12, 3, 12]],
        index=pd.MultiIndex.from_tuples(
            [("XYZ", "Estimated EPS Average"), ("XYZ", "Number of Analysts EPS")]
        ),
        columns=estimate_quarters,
    )
    ratios._analyst_estimates_cache["data"] = estimates

    every_period = ratios.get_forward_price_earnings_ratio(rounding=None).loc["XYZ"]
    deep_periods = ratios.get_forward_price_earnings_ratio(
        min_analysts=5, rounding=None
    ).loc["XYZ"]

    assert every_period[pd.Period("2024Q2", "Q")] == pytest.approx(50 / 5.0)
    # 2024Q1 covers 2024Q1-Q4; 2024Q2 includes 2025Q1, which three analysts cover.
    assert deep_periods[pd.Period("2024Q1", "Q")] == pytest.approx(50 / 4.6)
    assert pd.isna(deep_periods[pd.Period("2024Q2", "Q")])


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
    # The two parts are added unrounded, so the sum matches to the last digit.
    result = ratios_module.get_shareholder_yield(rounding=12)
    dividend_yield = ratios_module.get_dividend_yield(rounding=15)
    buyback_yield = ratios_module.get_buyback_yield(rounding=15)

    expected = (
        dividend_yield.loc[result.index, result.columns]
        + buyback_yield.loc[result.index, result.columns]
    ).round(12)

    pd.testing.assert_frame_equal(result, expected, check_names=False)


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
            "Cash and Cash Equivalents": 100 * growth,
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


def test_asset_coverage_ratio_deducts_goodwill_with_the_other_intangibles(
    ratios_module,
):
    # Microsoft's fixture carries separate, nonzero goodwill and other intangible assets.
    # Subtracting only the "Intangible Assets" row would leave the goodwill in the asset
    # base and give 5.0648 for 2023 (4.6597 for the trailing two-period average).
    balance = ratios_module._balance_sheet_statement.loc["MSFT"]
    assert (balance.loc["Goodwill"] > 0).all()

    assert ratios_module.get_asset_coverage_ratio().loc["MSFT"].tolist() == [
        pytest.approx(value, abs=1e-4) for value in [2.5685, 2.8872, 3.1612, 3.9327]
    ]
    assert ratios_module.get_asset_coverage_ratio(trailing=2).loc[
        "MSFT"
    ].dropna().tolist() == [
        pytest.approx(value, abs=1e-4) for value in [2.7242, 3.0173, 3.5428]
    ]


def test_asset_coverage_ratio_does_not_need_the_combined_intangibles_row(
    ratios_module, monkeypatch
):
    expected = ratios_module.get_asset_coverage_ratio()
    expected_trailing = ratios_module.get_asset_coverage_ratio(trailing=2)

    # A provider can return goodwill and the other intangible assets without the row
    # that combines them. Selected separately, the two give the same ratio without it.
    balance = ratios_module._balance_sheet_statement.copy()
    combined_row = balance.index.get_level_values(1) == "Goodwill and Intangible Assets"
    assert combined_row.any()
    balance.loc[combined_row] = np.nan
    monkeypatch.setattr(ratios_module, "_balance_sheet_statement", balance)

    pd.testing.assert_frame_equal(ratios_module.get_asset_coverage_ratio(), expected)
    pd.testing.assert_frame_equal(
        ratios_module.get_asset_coverage_ratio(trailing=2), expected_trailing
    )


def test_get_market_cap_basket(recorder, ratios_module, basket_toolkit, monkeypatch):
    basket_ratios = basket_toolkit.ratios
    for module in (ratios_module, basket_ratios):
        monkeypatch.setattr(
            module,
            "_currency_function",
            lambda: pd.Series({"AAPL": "USD", "MSFT": "USD"}),
        )

    market_cap = ratios_module.get_market_cap(rounding=None)
    basket = basket_ratios.get_market_cap(rounding=None)
    share = ratios_module.get_market_cap(share=True)
    daily_basket = basket_ratios.get_market_cap(show_daily=True)

    # The Total adds up the tickers and each share is a part of it.
    pd.testing.assert_series_equal(
        basket.loc["Total"],
        market_cap.sum(axis=0, skipna=False),
        check_names=False,
    )
    assert share.loc["Total"].dropna().eq(1).all()
    assert share.drop(index="Total").sum(min_count=1).dropna().round(3).eq(1).all()
    assert daily_basket.columns.tolist() == ["AAPL", "MSFT", "Total"]

    recorder.capture(basket)
    recorder.capture(share)
    recorder.capture(daily_basket.tail(5))


def test_get_market_cap_basket_refuses_mixed_currencies(
    ratios_module, basket_toolkit, monkeypatch
):
    basket_ratios = basket_toolkit.ratios
    for module in (ratios_module, basket_ratios):
        monkeypatch.setattr(
            module,
            "_currency_function",
            lambda: pd.Series({"AAPL": "USD", "MSFT": "EUR"}),
        )

    # Market caps in different currencies can't be summed.
    assert basket_ratios.get_market_cap().empty
    assert ratios_module.get_market_cap(share=True).empty
    assert ratios_module.get_market_cap_to_gdp().empty
    assert not ratios_module.get_market_cap().empty


def test_get_market_cap_to_gdp(recorder, ratios_module, monkeypatch):
    monkeypatch.setattr(
        ratios_module,
        "_currency_function",
        lambda: pd.Series({"AAPL": "USD", "MSFT": "USD"}),
    )
    # The annual GDP in millions of US dollars, as the Global Macro Database publishes it.
    gross_domestic_product = pd.DataFrame(
        {
            "United States": [21_380_000.0, 21_060_000.0, 23_320_000.0, 25_740_000.0],
            "China": [14_280_000.0, 14_690_000.0, 17_820_000.0, 17_880_000.0],
        },
        index=pd.period_range("2019", "2022", freq="Y"),
    )
    monkeypatch.setattr(
        ratios_module,
        "_gross_domestic_product_function",
        lambda: gross_domestic_product,
    )

    basket = ratios_module.get_market_cap(rounding=None).sum(axis=0, skipna=False)
    market_cap_to_gdp = ratios_module.get_market_cap_to_gdp(rounding=None)
    daily = ratios_module.get_market_cap_to_gdp(country="China", show_daily=True)

    # Each year is divided by the GDP of the year before it, in US dollars.
    assert market_cap_to_gdp.index.tolist() == ["United States"]
    assert market_cap_to_gdp.loc["United States", pd.Period("2022", "Y")] == (
        pytest.approx(basket[pd.Period("2022", "Y")] / 23_320_000e6, abs=1e-4)
    )
    assert daily.columns.tolist() == ["China"]
    assert ratios_module.get_market_cap_to_gdp(country="Atlantis").empty

    recorder.capture(market_cap_to_gdp)
    recorder.capture(daily.tail(5))


def test_basket_ratios_are_calculated_on_the_summed_statements(
    recorder, ratios_module, basket_toolkit
):
    basket_ratios = basket_toolkit.ratios
    cash_flow = basket_toolkit.get_cash_flow_statement(rounding=None)
    coverage = basket_ratios.get_capex_coverage_ratio(rounding=None)

    # The Total is a company of its own: its ratio is the ratio of the summed figures,
    # not an average of the ratios of the tickers.
    pd.testing.assert_series_equal(
        coverage.loc["Total"],
        cash_flow.loc[("Total", "Operating Cash Flow")]
        / cash_flow.loc[("Total", "Capital Expenditure")].abs(),
        check_names=False,
    )
    # The tickers themselves are not affected by the basket.
    pd.testing.assert_frame_equal(
        coverage.drop(index="Total"),
        ratios_module.get_capex_coverage_ratio(rounding=None),
    )
    pd.testing.assert_frame_equal(
        basket_ratios.collect_all_ratios(rounding=None).drop(index="Total", level=0),
        ratios_module.collect_all_ratios(rounding=None),
    )

    # Per-share ratios have no Total, the ratios built on the market cap use the summed one.
    assert basket_ratios.get_earnings_per_share().loc["Total"].isna().all()
    assert basket_ratios.get_ev_to_ebitda_ratio().loc["Total"].notna().any()

    recorder.capture(coverage)
    recorder.capture(basket_ratios.get_free_cash_flow_yield())


def test_get_net_capital_expenditure(recorder, ratios_module):
    net_capital_expenditure = ratios_module.get_net_capital_expenditure(rounding=None)
    cash_flow = ratios_module._cash_flow_statement

    expected = (
        cash_flow.loc[:, "Capital Expenditure", :].abs()
        - cash_flow.loc[:, "Depreciation and Amortization", :]
    )
    pd.testing.assert_frame_equal(
        net_capital_expenditure,
        expected.loc[:, net_capital_expenditure.columns],
        check_names=False,
    )

    recorder.capture(net_capital_expenditure)
    recorder.capture(ratios_module.get_net_capital_expenditure(trailing=2))


def _ratios_with(ratios_module, balance=None, period_end_dates=None):
    from financetoolkit.ratios.ratios_controller import Ratios

    return Ratios(
        ratios_module._tickers,
        {
            "period": ratios_module._historical_data,
            "daily": ratios_module._daily_historical_data,
        },
        ratios_module._balance_sheet_statement if balance is None else balance,
        ratios_module._income_statement,
        ratios_module._cash_flow_statement,
        rounding=None,
        period_end_dates=period_end_dates,
    )


def test_period_prices_are_taken_at_the_fiscal_period_end(ratios_module):
    periods = ratios_module._balance_sheet_statement.columns
    # A fiscal year ending in January is labelled with the calendar year before it.
    fiscal_ends = pd.DataFrame(
        {
            period: [
                pd.Timestamp(f"{period.year + 1}-01-31"),
                pd.Timestamp(f"{period.year}-06-30"),
            ]
            for period in periods
        },
        index=["AAPL", "MSFT"],
    )
    ratios = _ratios_with(ratios_module, period_end_dates=fiscal_ends)

    for method in ["get_price_to_earnings_ratio", "get_price_to_book_ratio"]:
        period = getattr(ratios, method)()
        daily = getattr(ratios, method)(show_daily=True)

        # The period ratio equals the daily ratio on the day the period ends.
        assert period.loc["MSFT"].loc["2021"] == pytest.approx(
            daily.loc["2021-06-30", "MSFT"]
        )
        assert period.loc["AAPL"].loc["2021"] == pytest.approx(
            daily.loc["2022-01-31", "AAPL"]
        )

    # Fiscal years ending on 31 December are priced as without end dates.
    december_ends = pd.DataFrame(
        {period: [period.end_time.normalize()] * 2 for period in periods},
        index=["AAPL", "MSFT"],
    )
    pd.testing.assert_frame_equal(
        _ratios_with(ratios_module, period_end_dates=december_ends).get_market_cap(),
        _ratios_with(ratios_module).get_market_cap(),
    )


def test_collected_growth_is_calculated_before_rounding(ratios_module):
    collected = ratios_module.collect_valuation_ratios(growth=True)

    for item, method in [
        ("Dividend Yield", "get_dividend_yield"),
        ("Earnings Yield", "get_earnings_yield"),
    ]:
        single = getattr(ratios_module, method)(growth=True)

        for ticker in ["AAPL", "MSFT"]:
            columns = collected.columns.intersection(single.columns)
            pd.testing.assert_series_equal(
                collected.loc[(ticker, item), columns],
                single.loc[ticker, columns],
                check_names=False,
            )


def _add_row(statement, item, values):
    return pd.concat(
        [
            statement,
            pd.DataFrame(
                [values],
                index=pd.MultiIndex.from_tuples([("XYZ", item)]),
                columns=statement.columns,
            ),
        ]
    )


def test_quarterly_debt_ratios_default_to_the_trailing_twelve_months():
    ratios, income = _quarterly_ratios()
    balance = ratios._balance_sheet_statement
    cash = ratios._cash_flow_statement
    total_debt = balance.loc["XYZ"].loc["Total Debt"]
    net_debt = total_debt - balance.loc["XYZ"].loc["Cash and Cash Equivalents"]
    operating_cash_flow = 1.5 * cash.loc["XYZ"].loc["Free Cash Flow"]

    ratios._balance_sheet_statement = _add_row(balance, "Net Debt", net_debt.to_numpy())
    ratios._cash_flow_statement = _add_row(
        cash, "Cash Flow from Operations", operating_cash_flow.to_numpy()
    )

    ebitda = (
        income.loc["XYZ"].loc["Operating Income"].iloc[-4:].sum()
        + cash.loc["XYZ"].loc["Depreciation and Amortization"].iloc[-4:].sum()
    )

    # A year of flows against the balance at the end of the last quarter.
    assert ratios.get_net_debt_to_ebitda_ratio().iloc[0, -1] == pytest.approx(
        net_debt.iloc[-1] / ebitda
    )
    assert ratios.get_gross_debt_to_ebitda_ratio().iloc[0, -1] == pytest.approx(
        total_debt.iloc[-1] / ebitda
    )
    assert ratios.get_cash_flow_coverage_ratio().iloc[0, -1] == pytest.approx(
        operating_cash_flow.iloc[-4:].sum() / total_debt.iloc[-1]
    )
    # trailing=1 gives single quarters.
    assert ratios.get_cash_flow_coverage_ratio(trailing=1).iloc[0, -1] == pytest.approx(
        operating_cash_flow.iloc[-1] / total_debt.iloc[-1]
    )


def test_market_value_ratios_use_the_balance_sheet_at_the_period_end(ratios_module):
    import inspect

    items = ["Price-to-Book", "Price-to-Tangible-Book", "EV-to-Invested-Capital"]
    trailing = ratios_module.collect_valuation_ratios(trailing=2)
    default = ratios_module.collect_valuation_ratios()

    # A market value is compared with the balance sheet on the same date, so these
    # ratios take no trailing window and the collection gives them unchanged.
    pd.testing.assert_frame_equal(
        trailing.loc[(slice(None), items), :],
        default.loc[(slice(None), items), :],
    )

    for method in [
        "get_price_to_book_ratio",
        "get_price_to_tangible_book_ratio",
        "get_ev_to_invested_capital_ratio",
    ]:
        assert (
            "trailing"
            not in inspect.signature(getattr(ratios_module, method)).parameters
        )


def test_trailing_one_averages_the_opening_and_closing_balance(ratios_module):
    for method in [
        "get_days_of_sales_outstanding",
        "get_asset_turnover_ratio",
        "get_return_on_equity",
        "get_cash_conversion_cycle",
    ]:
        pd.testing.assert_frame_equal(
            getattr(ratios_module, method)(trailing=1),
            getattr(ratios_module, method)(),
        )


def test_dividend_yield_is_nan_for_a_period_before_the_first_price(ratios_module):
    dividend_yield = ratios_module.get_dividend_yield(rounding=10)
    daily = ratios_module._daily_historical_data
    first_price = daily.index[0]

    # The first period starts before the first price, so its dividends are incomplete.
    first_period = dividend_yield.columns[0]
    assert first_period.start_time < first_price.start_time
    assert dividend_yield[first_period].isna().all()

    dividends = daily["Dividends"].loc["2021"].sum()
    close = daily["Close"].loc["2021"].iloc[-1]
    pd.testing.assert_series_equal(
        dividend_yield["2021"].astype(float),
        (dividends / close).reindex(dividend_yield.index),
        check_names=False,
        rtol=1e-3,
    )


def test_price_to_earnings_growth_divides_unrounded_figures(ratios_module):
    income = ratios_module._income_statement
    peg = ratios_module.get_price_to_earnings_growth_ratio(rounding=10)
    price_to_earnings = ratios_module.get_price_to_earnings_ratio(rounding=15)

    for ticker in ["AAPL", "MSFT"]:
        eps = (
            income.loc[(ticker, "Net Income")]
            / income.loc[(ticker, "Weighted Average Shares Diluted")]
        )
        growth = eps.loc[YEAR] / eps.shift(1).loc[YEAR] - 1

        assert peg.loc[ticker].loc[YEAR] == pytest.approx(
            price_to_earnings.loc[ticker].loc[YEAR] / (growth * 100), rel=1e-8
        )


def test_return_on_equity_excludes_minority_interest(ratios_module):
    balance = ratios_module._balance_sheet_statement.copy()
    # A minority interest raises the total equity but not that of the shareholders.
    balance.loc[("AAPL", "Total Equity")] += 1e10

    pd.testing.assert_frame_equal(
        _ratios_with(ratios_module, balance=balance).get_return_on_equity(),
        _ratios_with(ratios_module).get_return_on_equity(),
    )


def test_get_cyclically_adjusted_price_to_earnings_ratio(
    recorder, ratios_module, monkeypatch
):
    # A monthly consumer price index that rises by one point a month from 100 in January 2018.
    consumer_price_index = pd.Series(
        range(100, 160),
        index=pd.period_range("2018-01", periods=60, freq="M"),
        dtype=float,
    )
    monkeypatch.setattr(
        ratios_module,
        "_consumer_price_index_function",
        lambda country: consumer_price_index,
    )

    cape = ratios_module.get_cyclically_adjusted_price_to_earnings_ratio(
        years=2, rounding=None
    )
    nominal_cape = ratios_module.get_cyclically_adjusted_price_to_earnings_ratio(
        years=2, inflation_adjusted=False, rounding=None
    )
    one_year = ratios_module.get_cyclically_adjusted_price_to_earnings_ratio(
        years=1, inflation_adjusted=False, rounding=None
    )
    price_to_earnings = ratios_module.get_price_to_earnings_ratio(rounding=None)
    eps = ratios_module.get_earnings_per_share(rounding=None)
    price = price_to_earnings * eps

    recorder.capture(cape)
    recorder.capture(nominal_cape)

    # With one year and no inflation adjustment it is the P/E (the earnings are positive).
    pd.testing.assert_frame_equal(one_year, price_to_earnings, atol=1e-3)

    # 2021 against the 2020 and 2021 EPS, the 2020 EPS in December 2021 prices (CPI 147 / 135).
    year, previous = pd.Period("2021", "Y"), pd.Period("2020", "Y")
    real_average = (eps.loc["AAPL", previous] * 147 / 135 + eps.loc["AAPL", year]) / 2
    assert cape.loc["AAPL", year] == pytest.approx(
        price.loc["AAPL", year] / real_average, rel=1e-3
    )
    nominal_average = (eps.loc["AAPL", previous] + eps.loc["AAPL", year]) / 2
    assert nominal_cape.loc["AAPL", year] == pytest.approx(
        price.loc["AAPL", year] / nominal_average, rel=1e-3
    )

    # The first year has no year before it, so the two-year window is incomplete.
    assert cape[previous].isna().all()

    # A country without a consumer price index gives an empty result.
    monkeypatch.setattr(
        ratios_module,
        "_consumer_price_index_function",
        lambda country: pd.Series(dtype=float),
    )
    assert ratios_module.get_cyclically_adjusted_price_to_earnings_ratio(
        country="Atlantis"
    ).empty


def test_consumer_price_index_function_accepts_country_aliases(
    ratios_module, monkeypatch
):
    from financetoolkit.economics.economics_controller import Economics

    # Economics returns the names its sources use, whatever form of the name was requested.
    consumer_price_index = pd.DataFrame(
        {"United States": [100.0, 101.0], "Netherlands": [100.0, 102.0]},
        index=pd.period_range("2020-01", periods=2, freq="M"),
    )
    monkeypatch.setattr(
        Economics,
        "get_consumer_price_index",
        lambda self, countries=None, period=None: consumer_price_index,
    )

    function = ratios_module._consumer_price_index_function
    pd.testing.assert_series_equal(
        function("USA"), consumer_price_index["United States"]
    )
    pd.testing.assert_series_equal(
        function("netherlands"), consumer_price_index["Netherlands"]
    )
    assert function("Atlantis").empty


def test_get_return_on_net_operating_assets(recorder, ratios_module):
    recorder.capture(ratios_module.get_return_on_net_operating_assets())
    recorder.capture(ratios_module.get_return_on_net_operating_assets(trailing=2))


def test_get_net_borrowing_cost(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_borrowing_cost())


def test_get_financial_leverage(recorder, ratios_module):
    recorder.capture(ratios_module.get_financial_leverage())


def test_nissim_penman_ratios_follow_the_statements(ratios_module):
    balance = ratios_module._balance_sheet_statement
    income = ratios_module._income_statement
    period = pd.Period("2022", freq="Y")

    def row(statement, name, ticker, at):
        value = statement.loc[(ticker, name), at]
        return 0.0 if pd.isna(value) else value

    rnoa = ratios_module.get_return_on_net_operating_assets(rounding=10)
    nbc = ratios_module.get_net_borrowing_cost(rounding=10)
    flev = ratios_module.get_financial_leverage(rounding=10)
    for ticker in ["AAPL", "MSFT"]:

        def nfo(at, ticker=ticker):
            return (
                row(balance, "Total Debt", ticker, at)
                + row(balance, "Preferred Stock", ticker, at)
                - row(balance, "Cash and Cash Equivalents", ticker, at)
                - row(balance, "Short Term Investments", ticker, at)
                - row(balance, "Long Term Investments", ticker, at)
            )

        def cse(at, ticker=ticker):
            return row(balance, "Total Shareholder Equity", ticker, at) - row(
                balance, "Preferred Stock", ticker, at
            )

        before_tax = row(income, "Income Before Tax", ticker, period)
        tax_rate = min(
            max(row(income, "Income Tax Expense", ticker, period) / before_tax, 0), 1
        )
        nfe = (
            row(income, "Interest Expense", ticker, period)
            - row(income, "Interest Income", ticker, period)
        ) * (1 - tax_rate)
        average_nfo = (nfo(period) + nfo(period - 1)) / 2
        average_cse = (cse(period) + cse(period - 1)) / 2
        net_income = row(income, "Net Income", ticker, period)

        assert rnoa.loc[ticker, period] == pytest.approx(
            (net_income + nfe) / (average_nfo + average_cse)
        )
        assert nbc.loc[ticker, period] == pytest.approx(nfe / average_nfo)
        assert flev.loc[ticker, period] == pytest.approx(average_nfo / average_cse)


def test_nissim_penman_ratios_are_nan_without_interest_rows(ratios_module):
    ratios = copy.copy(ratios_module)
    income = ratios_module._income_statement.copy()
    period = pd.Period("2022", freq="Y")
    # Neither interest row reported for AAPL in 2022 (FMP gives zeros, as for Apple from 2024).
    income.loc[("AAPL", "Interest Expense"), period] = 0
    income.loc[("AAPL", "Interest Income"), period] = float("nan")
    # Interest income alone missing for MSFT: counted as zero, the ratios remain.
    income.loc[("MSFT", "Interest Income"), period] = float("nan")
    ratios._income_statement = income

    base_rnoa = ratios_module.get_return_on_net_operating_assets(rounding=10)
    rnoa = ratios.get_return_on_net_operating_assets(rounding=10)
    nbc = ratios.get_net_borrowing_cost(rounding=10)
    flev = ratios.get_financial_leverage(rounding=10)

    assert pd.isna(rnoa.loc["AAPL", period]) and pd.isna(nbc.loc["AAPL", period])
    assert flev.loc["AAPL", period] == pytest.approx(
        ratios_module.get_financial_leverage(rounding=10).loc["AAPL", period]
    )
    # With the same average obligations, only the interest income dropping out changes the NBC.
    base = ratios_module._income_statement
    expense = base.loc[("MSFT", "Interest Expense"), period]
    interest_income = base.loc[("MSFT", "Interest Income"), period]
    base_nbc = ratios_module.get_net_borrowing_cost(rounding=10)
    assert nbc.loc["MSFT", period] == pytest.approx(
        base_nbc.loc["MSFT", period] * expense / (expense - interest_income)
    )
    assert pd.notna(rnoa.loc["MSFT", period])
    # Only the period itself is NaN; a trailing window over it is NaN as well.
    other = period + 1
    assert rnoa.loc["AAPL", other] == pytest.approx(base_rnoa.loc["AAPL", other])
    assert (
        ratios.get_net_borrowing_cost(trailing=2)
        .loc["AAPL", [period, other]]
        .isna()
        .all()
    )


def test_get_dividend_adjusted_price_to_earnings_growth_ratio(recorder, ratios_module):
    recorder.capture(
        ratios_module.get_dividend_adjusted_price_to_earnings_growth_ratio()
    )

    # The P/E and EPS growth of the PEG ratio, with the dividend yield of the same period
    # added to the growth, both in percent.
    pegy = ratios_module.get_dividend_adjusted_price_to_earnings_growth_ratio(
        rounding=10
    )
    peg = ratios_module.get_price_to_earnings_growth_ratio(rounding=15)
    price_earnings = ratios_module.get_price_to_earnings_ratio(rounding=15)
    dividend_yield = ratios_module.get_dividend_yield(rounding=15)
    growth_rate = price_earnings / peg / 100
    expected = price_earnings / ((growth_rate + dividend_yield) * 100)
    pd.testing.assert_frame_equal(
        pegy,
        expected.loc[pegy.index, pegy.columns].round(10),
        check_names=False,
        rtol=1e-8,
    )


def test_get_net_debt_to_equity_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_debt_to_equity_ratio())
    recorder.capture(ratios_module.get_net_debt_to_equity_ratio(trailing=2))

    # The net debt of the balance sheet (total debt minus cash) over the total equity.
    balance = ratios_module._balance_sheet_statement
    expected = (
        balance.loc[:, "Total Debt", :] - balance.loc[:, "Cash and Cash Equivalents", :]
    ) / balance.loc[:, "Total Equity", :]
    ratio = ratios_module.get_net_debt_to_equity_ratio(rounding=10)
    pd.testing.assert_frame_equal(
        ratio, expected.loc[ratio.index, ratio.columns].round(10), check_names=False
    )


@pytest.mark.parametrize(
    ("quarterly", "trailing"), [(False, None), (False, 2), (True, None), (True, 4)]
)
def test_cash_return_on_invested_capital_uses_the_invested_capital_of_the_roic(
    quarterly, trailing
):
    from financetoolkit.ratios.ratios_controller import Ratios  # noqa: PLC0415

    balance, income, cash = [
        pd.read_pickle(f"tests/datasets/{name}_dataset.pickle")
        for name in ("balance", "income", "cash")
    ]
    if quarterly:
        balance, income, cash = [
            pd.concat([statement, statement * 1.07], axis=1).set_axis(
                pd.period_range("2019Q1", periods=8, freq="Q"), axis=1
            )
            for statement in (balance, income, cash)
        ]

    ratios = Ratios(
        ["AAPL", "MSFT"], {}, balance, income, cash, quarterly=quarterly, rounding=None
    )
    cash_return = ratios.get_cash_return_on_invested_capital(trailing=trailing)

    # The return on invested capital averages the balances over the trailing + 1 period
    # ends, or the opening and closing balance without trailing.
    invested_capital = (
        (balance.xs("Total Equity", level=1) + balance.xs("Total Debt", level=1))
        .T.rolling((trailing or 1) + 1)
        .mean()
        .T
    )
    free_cash_flow = cash.xs("Free Cash Flow", level=1)
    if trailing:
        free_cash_flow = free_cash_flow.T.rolling(trailing).sum().T

    expected = free_cash_flow / invested_capital
    assert cash_return.notna().sum().sum() >= 4
    pd.testing.assert_frame_equal(
        cash_return, expected, check_names=False, check_freq=False, rtol=1e-12
    )


def _fiscal_year_ends(periods):
    # AAPL with a fiscal year ending on the last Saturday of September (a 52-53 week year)
    # and MSFT with one ending on 30 June, both labelled with the calendar year they end in.
    return pd.DataFrame(
        {
            period: [
                pd.Timestamp(f"{period.year}-10-01") - pd.offsets.Week(weekday=5),
                pd.Timestamp(f"{period.year}-06-30"),
            ]
            for period in periods
        },
        index=["AAPL", "MSFT"],
    )


def test_dividend_yield_covers_the_fiscal_year_of_the_statements(ratios_module):
    periods = ratios_module._balance_sheet_statement.columns
    fiscal_ends = _fiscal_year_ends(periods)
    ratios = _ratios_with(ratios_module, period_end_dates=fiscal_ends)

    dividend_yield = ratios.get_dividend_yield(rounding=15)
    daily = ratios_module._daily_historical_data
    dates = daily.index.to_timestamp()
    dividends = daily["Dividends"].set_axis(dates)
    close = daily["Close"].set_axis(dates)
    price_to_earnings_prices = ratios._get_share_prices(
        str(dividend_yield.columns[0]), str(dividend_yield.columns[-1])
    )

    for ticker in ["AAPL", "MSFT"]:
        for year in [pd.Period("2021", "Y"), pd.Period("2022", "Y")]:
            start = fiscal_ends.loc[ticker, year - 1]
            end = fiscal_ends.loc[ticker, year]
            paid = dividends.loc[(dates > start) & (dates <= end), ticker].sum()
            price = close.loc[:end, ticker].iloc[-1]

            # The dividends of the fiscal year over the close the price-to-earnings ratio
            # of the same column takes, not those of the calendar year.
            assert price == price_to_earnings_prices.loc[ticker, year]
            assert dividend_yield.loc[ticker, year] == pytest.approx(paid / price)

    # The calendar year 2021 of MSFT holds other dividends than its fiscal year to June.
    calendar = dividends.loc["2021", "MSFT"].sum() / close.loc["2021", "MSFT"].iloc[-1]
    assert dividend_yield.loc["MSFT", pd.Period("2021", "Y")] != pytest.approx(calendar)


def test_dividend_yield_of_december_fiscal_years_equals_the_calendar_yield(
    ratios_module,
):
    periods = ratios_module._balance_sheet_statement.columns
    december_ends = pd.DataFrame(
        {period: [period.end_time.normalize()] * 2 for period in periods},
        index=["AAPL", "MSFT"],
    )

    pd.testing.assert_frame_equal(
        _ratios_with(ratios_module, period_end_dates=december_ends).get_dividend_yield(
            rounding=15
        ),
        _ratios_with(ratios_module).get_dividend_yield(rounding=15),
    )


def test_quarterly_dividend_yield_sums_the_trailing_four_fiscal_quarters():
    ratios, _ = _quarterly_ratios()
    quarters = ratios._balance_sheet_statement.columns
    # Fiscal quarters ending two months before the calendar quarters they are labelled with,
    # before the dividend paid in the middle of each calendar quarter.
    fiscal_ends = pd.DataFrame(
        {
            quarter: [(quarter.end_time - pd.offsets.MonthEnd(2)).normalize()]
            for quarter in quarters
        },
        index=["XYZ"],
    )
    ratios._period_end_dates = fiscal_ends
    # A dividend that rises with every payment, so the fiscal and calendar windows differ.
    daily = ratios._daily_historical_data
    paid = daily[("Dividends", "XYZ")] > 0
    daily.loc[paid, ("Dividends", "XYZ")] = 0.2 + 0.01 * np.arange(paid.sum())

    dividend_yield = ratios.get_dividend_yield(rounding=15)
    dates = daily.index.to_timestamp()
    dividends = daily["Dividends"]["XYZ"].set_axis(dates)

    # The last quarter differs from the trailing calendar year of its label.
    calendar = dividends.loc[str(quarters[-4].start_time.date()) :].sum() / 50
    assert dividend_yield.loc["XYZ", quarters[-1]] != pytest.approx(calendar)

    for position in range(4, len(quarters)):
        quarter = quarters[position]
        start, end = fiscal_ends.iloc[0, position - 4], fiscal_ends.iloc[0, position]
        paid = dividends.loc[(dates > start) & (dates <= end)].sum()

        assert dividend_yield.loc["XYZ", quarter] == pytest.approx(paid / 50)

    # trailing=1 gives the dividends of the single fiscal quarter.
    single = ratios.get_dividend_yield(trailing=1, rounding=15)
    start, end = fiscal_ends.iloc[0, -2], fiscal_ends.iloc[0, -1]
    assert single.loc["XYZ", quarters[-1]] == pytest.approx(
        dividends.loc[(dates > start) & (dates <= end)].sum() / 50
    )


def test_dividend_yield_of_an_unfinished_period_covers_the_year_up_to_the_last_close():
    ratios, _ = _quarterly_ratios()
    quarters = ratios._balance_sheet_statement.columns
    daily = ratios._daily_historical_data
    paid = daily[("Dividends", "XYZ")] > 0
    daily.loc[paid, ("Dividends", "XYZ")] = 0.2 + 0.01 * np.arange(paid.sum())
    daily.loc[daily.index[-1], ("Close", "XYZ")] = 40.0

    # The last fiscal quarter ends a month after the last close.
    fiscal_ends = pd.DataFrame(
        {quarter: [quarter.end_time.normalize()] for quarter in quarters},
        index=["XYZ"],
    )
    fiscal_ends[quarters[-1]] = pd.Timestamp("2024-01-31")
    ratios._period_end_dates = fiscal_ends

    dates = daily.index.to_timestamp()
    dividends = daily["Dividends"]["XYZ"].set_axis(dates)
    last_close = dates[-1]
    year = dividends.loc[
        (dates > last_close - pd.DateOffset(months=12)) & (dates <= last_close)
    ]

    # Four dividends against the last close, not the part of the period seen so far.
    assert (year > 0).sum() == 4  # noqa: PLR2004
    assert ratios.get_dividend_yield(rounding=15).loc[
        "XYZ", quarters[-1]
    ] == pytest.approx(year.sum() / 40)


def test_trailing_windows_of_a_half_yearly_reporter_count_calendar_quarters():
    """
    A company reporting half-yearly has no first and third quarter columns. Its ratios
    equal those of the same statements with those quarters as NaN columns: a trailing
    window or a comparison a year back counts quarters, not reported periods.
    """

    def half_yearly(keep_columns: bool):
        ratios, _ = _quarterly_ratios()

        for name in (
            "_income_statement",
            "_balance_sheet_statement",
            "_cash_flow_statement",
        ):
            statement = getattr(ratios, name)
            unreported = statement.columns.quarter.isin([1, 3])
            if name == "_balance_sheet_statement":
                statement = _add_row(
                    statement, "Total Assets", 200 * 1.03 ** np.arange(12)
                )
            statement = (
                statement.mask(np.broadcast_to(unreported, statement.shape))
                if keep_columns
                else statement.loc[:, ~unreported]
            )
            setattr(ratios, name, statement)

        return ratios

    reported = half_yearly(keep_columns=False)
    calendar = half_yearly(keep_columns=True)
    quarters = reported._balance_sheet_statement.columns
    assert list(quarters.quarter.unique()) == [2, 4]

    for method, kwargs in (
        ("get_price_to_earnings_ratio", {}),
        ("get_price_to_free_cash_flow_ratio", {}),
        ("get_net_profit_margin", {"trailing": 4}),
        ("get_return_on_assets", {}),
        ("get_return_on_assets", {"trailing": 4}),
        ("get_asset_turnover_ratio", {}),
        ("get_asset_growth", {}),
        ("get_net_share_issuance", {}),
    ):
        expected = getattr(calendar, method)(**kwargs)
        result = getattr(reported, method)(**kwargs)
        shared = result.columns.intersection(expected.columns)
        pd.testing.assert_frame_equal(
            result[shared], expected[shared], check_names=False, obj=method
        )

    # Four quarters back is the same half a year earlier, not two half-years back.
    asset_growth = reported.get_asset_growth()
    assert asset_growth.loc["XYZ", quarters[-1]] == pytest.approx(1.03**4 - 1)

    # A trailing twelve months needs all four quarters, which a half-yearly reporter lacks.
    assert reported.get_price_to_earnings_ratio().isna().all().all()


def test_get_operating_cash_flow_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_cash_flow_per_share())
    recorder.capture(ratios_module.get_operating_cash_flow_per_share(diluted=False))
    recorder.capture(ratios_module.get_operating_cash_flow_per_share(trailing=2))


def test_get_free_cash_flow_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_free_cash_flow_per_share())
    recorder.capture(ratios_module.get_free_cash_flow_per_share(growth=True))


def test_get_cash_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_cash_per_share())
    recorder.capture(ratios_module.get_cash_per_share(trailing=2))


def test_get_tangible_book_value_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_tangible_book_value_per_share())
    recorder.capture(ratios_module.get_tangible_book_value_per_share(diluted=False))


def test_get_net_net_working_capital_per_share(recorder, ratios_module):
    recorder.capture(ratios_module.get_net_net_working_capital_per_share())
    recorder.capture(
        ratios_module.get_net_net_working_capital_per_share(
            receivables_weight=0.9, inventory_weight=0.75
        )
    )


def test_get_long_term_debt_to_capitalization_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_long_term_debt_to_capitalization_ratio())
    recorder.capture(
        ratios_module.get_long_term_debt_to_capitalization_ratio(trailing=2)
    )


def test_get_debt_to_market_cap_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_debt_to_market_cap_ratio())
    recorder.capture(ratios_module.get_debt_to_market_cap_ratio(show_daily=True))


def test_get_solvency_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_solvency_ratio())
    recorder.capture(ratios_module.get_solvency_ratio(trailing=2))


def test_get_capex_to_operating_cash_flow_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_capex_to_operating_cash_flow_ratio())
    recorder.capture(ratios_module.get_capex_to_operating_cash_flow_ratio(trailing=2))


def test_get_intangibles_to_total_assets_ratio(recorder, ratios_module):
    recorder.capture(ratios_module.get_intangibles_to_total_assets_ratio())
    recorder.capture(ratios_module.get_intangibles_to_total_assets_ratio(trailing=2))


def test_get_operating_return_on_assets(recorder, ratios_module):
    recorder.capture(ratios_module.get_operating_return_on_assets())
    recorder.capture(ratios_module.get_operating_return_on_assets(trailing=2))


def test_get_continuing_operations_profit_margin(recorder, ratios_module):
    recorder.capture(ratios_module.get_continuing_operations_profit_margin())
    recorder.capture(ratios_module.get_continuing_operations_profit_margin(growth=True))


def _with_employee_counts(ratios_module):
    # The headcounts the annual filings of the test companies state, on the statement periods.
    ratios = copy.copy(ratios_module)
    periods = ratios._income_statement.columns
    employee_count = pd.DataFrame(
        [[147000, 154000, 164000, 161000], [163000, 181000, 221000, 221000]],
        index=["AAPL", "MSFT"],
        columns=periods,
    )
    ratios._employee_count_function = lambda: employee_count
    return ratios


def test_get_revenue_per_employee(recorder, ratios_module):
    ratios = _with_employee_counts(ratios_module)
    recorder.capture(ratios.get_revenue_per_employee())
    recorder.capture(ratios.get_revenue_per_employee(growth=True))

    # Revenue / employees: Apple's 2022 revenue of 394,328 million over 164,000 employees.
    assert ratios.get_revenue_per_employee(rounding=None).loc[
        "AAPL", ratios._income_statement.columns[2]
    ] == pytest.approx(394328e6 / 164000)


def test_get_net_income_per_employee(recorder, ratios_module):
    ratios = _with_employee_counts(ratios_module)
    recorder.capture(ratios.get_net_income_per_employee())


def test_figures_per_employee_are_empty_without_employee_counts(ratios_module):
    assert ratios_module.get_revenue_per_employee().empty
    assert ratios_module.get_net_income_per_employee().empty


def test_align_employee_count_matches_the_fiscal_year_ends():
    from financetoolkit.ratios.helpers import align_employee_count

    # The counts are labelled with the year of the fiscal year end: NVIDIA's fiscal year
    # ending January 2023 is labelled 2023 but is the statement column 2022.
    counts = pd.DataFrame(
        {"AAPL": [154000, 164000, 161000], "NVDA": [22473, 26196, 29600]},
        index=pd.PeriodIndex(["2021", "2022", "2023"], freq="Y"),
    )
    yearly = pd.DataFrame(
        {
            pd.Period("2021", "Y"): ["2021-09-25", "2022-01-30"],
            pd.Period("2022", "Y"): ["2022-09-24", None],
        },
        index=["AAPL", "NVDA"],
    )
    aligned = align_employee_count(counts, yearly)
    assert aligned.loc["AAPL"].tolist() == [154000, 164000]
    assert aligned.loc["NVDA", pd.Period("2021", "Y")] == 26196
    assert pd.isna(aligned.loc["NVDA", pd.Period("2022", "Y")])

    # A quarter uses the latest count of a fiscal year that ended in an earlier calendar
    # year, or in its own when the quarter ends in December (or the first days of January).
    quarterly = pd.DataFrame(
        {
            pd.Period("2022Q3", "Q"): ["2022-09-24", "2022-10-30"],
            pd.Period("2022Q4", "Q"): ["2022-12-31", "2023-01-01"],
        },
        index=["AAPL", "NVDA"],
    )
    aligned = align_employee_count(counts, quarterly, quarterly=True)
    assert aligned.loc["AAPL"].tolist() == [154000, 164000]
    assert aligned.loc["NVDA"].tolist() == [22473, 26196]

    # A 52/53-week year ending on 25 December (Intel) is that December's count.
    intel = pd.DataFrame(
        [["2021-09-25", "2021-12-25", "2022-12-31"]],
        index=["INTC"],
        columns=[pd.Period(quarter, "Q") for quarter in ["2021Q3", "2021Q4", "2022Q4"]],
    )
    counts = pd.DataFrame(
        {"INTC": [100, 110, 120]},
        index=pd.PeriodIndex(["2020", "2021", "2022"], freq="Y"),
    )
    aligned = align_employee_count(counts, intel, quarterly=True)
    assert aligned.loc["INTC"].tolist() == [100, 110, 120]


def test_toolkit_passes_the_employee_counts_on_the_fiscal_year_ends(
    test_toolkit, monkeypatch
):
    # The counts as the Toolkit collects them: per year of the fiscal year end.
    counts = pd.DataFrame(
        {
            "AAPL": [147000, 154000, 164000, 161000],
            "MSFT": [163000, 181000, 221000, 221000],
        },
        index=pd.PeriodIndex(["2020", "2021", "2022", "2023"], freq="Y", name="Period"),
    )
    monkeypatch.setattr(test_toolkit, "_collect_employee_count", lambda: counts)

    revenue_per_employee = test_toolkit.ratios.get_revenue_per_employee(rounding=None)

    revenue = test_toolkit.ratios._income_statement.loc[:, "Revenue", :]
    expected = revenue / counts.T.set_axis(revenue.columns, axis=1)
    pd.testing.assert_frame_equal(
        revenue_per_employee,
        expected.loc[:, revenue_per_employee.columns],
        check_names=False,
    )


def test_align_employee_count_puts_a_year_ending_in_early_january_in_the_year_before():
    from financetoolkit.ratios.helpers import align_employee_count

    # A 52/53-week year ending 2 January 2021 is fiscal 2020, labelled as get_employee_count
    # labels it; the yearly statement of that year gets its own count, not the next one.
    counts = pd.DataFrame(
        {"K": [100, 200, 300, 400]},
        index=pd.PeriodIndex(["2019", "2020", "2021", "2022"], freq="Y"),
    )
    yearly = pd.DataFrame(
        [["2019-12-28", "2021-01-02", "2022-01-01", "2022-12-31"]],
        index=["K"],
        columns=[pd.Period(year, "Y") for year in ["2019", "2020", "2021", "2022"]],
    )
    assert align_employee_count(counts, yearly).loc["K"].tolist() == [
        100,
        200,
        300,
        400,
    ]

    quarterly = pd.DataFrame(
        [["2021-01-02", "2021-07-03", "2022-01-01", "2022-07-02"]],
        index=["K"],
        columns=[pd.Period(q, "Q") for q in ["2020Q4", "2021Q2", "2021Q4", "2022Q2"]],
    )
    aligned = align_employee_count(counts, quarterly, quarterly=True)
    assert aligned.loc["K"].tolist() == [200, 200, 300, 300]


def test_align_employee_count_puts_a_year_ending_on_7_january_in_the_year_before():
    from financetoolkit.ratios.helpers import align_employee_count

    counts = pd.DataFrame(
        {"K": [100, 200]}, index=pd.PeriodIndex(["2017", "2018"], freq="Y")
    )
    yearly = pd.DataFrame(
        [["2018-01-07", "2019-01-06"]],
        index=["K"],
        columns=[pd.Period("2017", "Y"), pd.Period("2018", "Y")],
    )
    assert align_employee_count(counts, yearly).loc["K"].tolist() == [100, 200]


def _offline_toolkit(balance=None, income=None, cash=None, **settings):
    """The offline Toolkit of the fixtures, with statements changed by the test."""
    from financetoolkit import Toolkit  # noqa: PLC0415

    def read(name):
        return pd.read_pickle(f"tests/datasets/{name}_dataset.pickle")

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=read("historical"),
        balance=read("balance") if balance is None else balance,
        income=read("income") if income is None else income,
        cash=read("cash") if cash is None else cash,
        convert_currency=False,
        sleep_timer=False,
        use_cached_data=False,
        **{"start_date": "2019-12-31", "end_date": "2023-01-01", **settings},
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")
    return toolkit


def test_a_ratio_built_on_a_missing_one_names_it_instead_of_failing(caplog):
    # Without the dividends the payout ratio fails, and the ratios built on it report the
    # missing input (an empty result) rather than an AttributeError on its empty Series.
    cash = pd.read_pickle("tests/datasets/cash_dataset.pickle")
    toolkit = _offline_toolkit(cash=cash.drop(index="Dividends Paid", level=1))

    assert toolkit.ratios.get_retention_ratio().empty
    assert "dividend payout ratio is not available" in caplog.text

    # The collection leaves the ratio out and keeps the others.
    valuation = toolkit.ratios.collect_valuation_ratios()
    assert "Dividend Payout Ratio" not in valuation.index.get_level_values(1)
    assert "Retention Ratio" not in valuation.index.get_level_values(1)
    assert "Price-to-Earnings" in valuation.index.get_level_values(1)
    assert not toolkit.ratios.collect_all_ratios().empty


def test_the_price_to_earnings_growth_ratios_name_the_missing_earnings(caplog):
    income = pd.read_pickle("tests/datasets/income_dataset.pickle")
    cash = pd.read_pickle("tests/datasets/cash_dataset.pickle")
    toolkit = _offline_toolkit(
        income=income.drop(index="Net Income", level=1),
        cash=cash.drop(index="Net Income", level=1),
    )

    assert toolkit.ratios.get_price_to_earnings_growth_ratio().empty
    assert toolkit.ratios.get_dividend_adjusted_price_to_earnings_growth_ratio().empty
    assert "earnings growth rate is not available" in caplog.text
    assert "AttributeError" not in caplog.text


def test_the_market_cap_to_gdp_without_the_gdp_is_empty(ratios_module, caplog):
    ratios = copy.copy(ratios_module)
    ratios._gross_domestic_product_function = pd.DataFrame
    ratios._get_basket_currency = lambda: "USD"

    assert ratios.get_market_cap_to_gdp().empty
    assert "Gross Domestic Product is not available" in caplog.text


def test_the_basic_shares_stand_in_for_missing_diluted_shares(caplog):
    income = pd.read_pickle("tests/datasets/income_dataset.pickle")
    toolkit = _offline_toolkit(
        income=income.drop(index=[("MSFT", "Weighted Average Shares Diluted")])
    )

    earnings_per_share = toolkit.ratios.get_earnings_per_share(rounding=None)
    basic = toolkit.ratios.get_earnings_per_share(diluted=False, rounding=None)

    pd.testing.assert_series_equal(earnings_per_share.loc["MSFT"], basic.loc["MSFT"])
    assert earnings_per_share.loc["AAPL"].notna().all()
    assert not earnings_per_share.loc["AAPL"].equals(basic.loc["AAPL"])
    assert "no Weighted Average Shares Diluted for MSFT" in caplog.text
    # The statement is shown as it was supplied.
    assert ("MSFT", "Weighted Average Shares Diluted") not in (
        toolkit.get_income_statement().index
    )


def test_supplied_statements_keep_the_year_before_the_start_date():
    toolkit = _offline_toolkit(start_date="2022-06-01")

    # The return on assets averages the assets with those of the year before, which the
    # statements keep, while only the periods from the start date on are shown.
    return_on_assets = toolkit.ratios.get_return_on_assets()
    assert return_on_assets.columns[0] == pd.Period("2022", "Y")
    assert return_on_assets.loc[:, "2022"].notna().all()
    assert toolkit.get_income_statement().columns[0] == pd.Period("2022", "Y")


def test_a_lag_below_one_is_refused_and_one_too_long_is_reported(ratios_module, caplog):
    from financetoolkit.utilities.error_model import (
        InvalidArgumentError,
    )  # noqa: PLC0415

    for lag in (0, -1, [1, 0], []):
        with pytest.raises(InvalidArgumentError):
            ratios_module.get_current_ratio(growth=True, lag=lag)
    with pytest.raises(InvalidArgumentError):
        ratios_module.collect_liquidity_ratios(growth=True, lag=0)

    assert ratios_module.get_current_ratio(growth=True, lag=100).isna().all().all()
    assert caplog.text.count("The lag of 100 periods is at least as long") == 1


def test_get_cyclically_adjusted_price_to_earnings_ratio_refuses_zero_years(
    ratios_module,
):
    from financetoolkit.utilities.error_model import (
        InvalidArgumentError,
    )  # noqa: PLC0415

    with pytest.raises(InvalidArgumentError, match="years"):
        ratios_module.get_cyclically_adjusted_price_to_earnings_ratio(years=0)


def test_ratios_without_values_say_why(ratios_module, caplog):
    caplog.set_level("INFO", logger="financetoolkit")

    assert ratios_module.get_preferred_dividend_coverage_ratio().isna().all().all()
    assert "AAPL, MSFT paid no preferred dividends" in caplog.text

    # Five years of market caps back, with four years of statements.
    assert ratios_module.get_composite_equity_issuance().isna().all().all()
    assert "with that of 5 periods earlier" in caplog.text


def test_get_factor_analysis_refuses_an_unknown_metric(ratios_module):
    from financetoolkit.utilities.error_model import (
        InvalidArgumentError,
    )  # noqa: PLC0415

    with pytest.raises(InvalidArgumentError, match="bogus"):
        ratios_module.get_factor_analysis(metrics="bogus")


def test_a_ratio_built_on_another_reports_a_window_without_data_once(caplog):
    # The statements end in 2023, so a window in 2024 holds none of their periods; the
    # price-to-sales ratio reports it once, not again for the market cap it builds on.
    toolkit = _offline_toolkit(start_date="2024-06-01", end_date="2024-12-31")

    assert toolkit.ratios.get_price_to_sales_ratio().empty

    assert caplog.text.count("There are no observations between") == 1
    assert "market cap is not available" in caplog.text


def test_a_trailing_window_below_one_is_refused_and_one_too_long_is_reported(
    ratios_module, caplog
):
    from financetoolkit.utilities.error_model import (  # noqa: PLC0415
        InvalidArgumentError,
    )

    for trailing in (0, -1, 2.5):
        with pytest.raises(InvalidArgumentError, match="trailing"):
            ratios_module.get_asset_growth(trailing=trailing)
        with pytest.raises(InvalidArgumentError, match="trailing"):
            ratios_module.collect_efficiency_ratios(trailing=trailing)

    with caplog.at_level("WARNING", logger="financetoolkit"):
        ratios_module.collect_efficiency_ratios(trailing=100)
    # The collection calls every ratio with the window, the window is reported once.
    assert caplog.text.count("The trailing window of 100 periods is longer") == 1


def test_a_trailing_window_as_long_as_the_statements_is_reported_for_averaged_balances(
    ratios_module, caplog
):
    periods = len(ratios_module._income_statement.columns)

    # The averaged equity of a window of every period also needs the period end before it.
    with caplog.at_level("WARNING", logger="financetoolkit"):
        return_on_equity = ratios_module.get_return_on_equity(trailing=periods)
    assert return_on_equity.isna().all().all()
    assert (
        caplog.text.count(f"The trailing window of {periods} periods is as long as")
        == 1
    )

    caplog.clear()
    with caplog.at_level("WARNING", logger="financetoolkit"):
        return_on_equity = ratios_module.get_return_on_equity(trailing=periods - 1)
    assert return_on_equity.notna().any().any()
    assert "trailing window" not in caplog.text
