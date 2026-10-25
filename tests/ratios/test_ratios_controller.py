"""Ratios Controller Tests""" ""
# pylint: disable=missing-function-docstring

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
