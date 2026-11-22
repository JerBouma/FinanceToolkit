"""Valuation Model Tests""" ""
import numpy as np
import pandas as pd
import pytest

from financetoolkit.ratios import valuation_model


@pytest.mark.parametrize(
    "estimated_eps, trailing_eps, years",
    [
        (2.0, 0.0, 1),
        (2.0, 1.0, 0),
        (0.0, 0.0, 0),
        (0.0, 1.0, 1),
        (-2.0, 1.0, 2),
        (2.0, -1.0, 2),
        (2.0, 1.0, -1),
    ],
)
def test_estimated_eps_growth_is_missing_for_invalid_scalars(
    estimated_eps, trailing_eps, years
):
    assert np.isnan(
        valuation_model.get_estimated_eps_growth_rate(
            estimated_eps, trailing_eps, years
        )
    )


@pytest.mark.parametrize("years, expected", [(1, 1.25), (2, 0.5)])
def test_estimated_eps_growth_valid_scalar_formula(years, expected):
    assert valuation_model.get_estimated_eps_growth_rate(
        9.0, 4.0, years
    ) == pytest.approx(expected)


@pytest.mark.parametrize("zero_input", ["trailing_eps", "years"])
def test_estimated_eps_growth_broadcasts_an_invalid_scalar(zero_input):
    estimates = pd.DataFrame({"2024": [2.0, 3.0], "2025": [4.0, 5.0]}, index=["A", "B"])
    inputs = {"estimated_eps": estimates, "trailing_eps": 1.0, "years": 1}
    inputs[zero_input] = 0

    result = valuation_model.get_estimated_eps_growth_rate(**inputs)

    pd.testing.assert_frame_equal(result, estimates * np.nan)


def test_estimated_eps_growth_preserves_series_alignment():
    result = valuation_model.get_estimated_eps_growth_rate(
        estimated_eps=pd.Series([9.0, 2.0], index=["B", "A"]),
        trailing_eps=pd.Series([1.0, 4.0], index=["A", "B"]),
        years=pd.Series([0.0, 2.0], index=["A", "B"]),
    )

    pd.testing.assert_series_equal(result, pd.Series([np.nan, 0.5], index=["A", "B"]))


def test_estimated_eps_growth_preserves_dataframe_alignment():
    estimates = pd.DataFrame(
        [[9.0, 4.0], [16.0, 25.0]], index=["A", "B"], columns=["2025", "2024"]
    )
    trailing = pd.DataFrame(
        [[1.0, 4.0], [4.0, 9.0]], index=["B", "A"], columns=["2024", "2025"]
    )

    result = valuation_model.get_estimated_eps_growth_rate(estimates, trailing, years=2)

    pd.testing.assert_frame_equal(result, (estimates / trailing) ** 0.5 - 1)


def test_get_earnings_per_share(recorder):
    recorder.capture(
        valuation_model.get_earnings_per_share(
            net_income=pd.Series([100, 110, 120, 130, 80]),
            preferred_dividends=pd.Series([0, 0, 0, 0, 2]),
            average_outstanding_shares=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_revenue_per_share(recorder):
    recorder.capture(
        valuation_model.get_revenue_per_share(
            total_revenue=pd.Series([3000, 1100, 1200, 1300, 3000]),
            shares_outstanding=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_price_to_earnings_ratio(recorder):
    recorder.capture(
        valuation_model.get_price_to_earnings_ratio(
            stock_price=pd.Series([30, 11, 12, 10, 30]),
            earnings_per_share=pd.Series([0.5, 0.4, 0.1, 1.1, 1.2]),
        )
    )


def test_get_price_to_earnings_growth_ratio(recorder):
    recorder.capture(
        valuation_model.get_price_to_earnings_growth_ratio(
            price_earnings=pd.Series([30, 11, 12, 10, 30]),
            growth_rate=pd.Series([0.2, 0.1, 0.1, 0.05, 0.01]),
        )
    )


def test_get_estimated_eps_growth_rate(recorder):
    recorder.capture(
        valuation_model.get_estimated_eps_growth_rate(
            estimated_eps=pd.Series([7.3, 12.8]),
            trailing_eps=pd.Series([6.6, 11.2]),
        )
    )


def test_get_book_value_per_share(recorder):
    recorder.capture(
        valuation_model.get_book_value_per_share(
            total_shareholder_equity=pd.Series([3000, 1100, 1200, 1300, 3000]),
            preferred_equity=pd.Series([200, 140, 160, 160, 300]),
            common_shares_outstanding=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_price_to_book_ratio(recorder):
    recorder.capture(
        valuation_model.get_price_to_book_ratio(
            price_per_share=pd.Series([30, 11, 10, 5, 30]),
            book_value_per_share=pd.Series([0.5, 0.5, 0.1, 1.1, 1.2]),
        )
    )


def test_get_interest_debt_per_share(recorder):
    recorder.capture(
        valuation_model.get_interest_debt_per_share(
            interest_expense=pd.Series([5, 1, 23, 5, 10]),
            total_debt=pd.Series([200, 140, 160, 160, 300]),
            shares_outstanding=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_capex_per_share(recorder):
    recorder.capture(
        valuation_model.get_capex_per_share(
            capital_expenditures=pd.Series([10, 10, 23, 15, 0]),
            shares_outstanding=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_dividend_yield(recorder):
    recorder.capture(
        valuation_model.get_dividend_yield(
            dividends=pd.Series([0.3, 0.31, 0.5, 0.25, 0.4]),
            stock_price=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_weighted_dividend_yield(recorder):
    recorder.capture(
        valuation_model.get_weighted_dividend_yield(
            dividends_paid=pd.Series([10, 10, 23, 15, 0]),
            shares_outstanding=pd.Series([1000, 500, 300, 160, 300]),
            stock_price=pd.Series([100, 130, 200, 160, 200]),
        )
    )


def test_get_price_to_cash_flow_ratio(recorder):
    recorder.capture(
        valuation_model.get_price_to_cash_flow_ratio(
            market_cap=pd.Series([1000, 500, 300, 160, 300]),
            operations_cash_flow=pd.Series([10, 10, 23, 15, 30]),
        )
    )


def test_get_price_to_free_cash_flow_ratio(recorder):
    recorder.capture(
        valuation_model.get_price_to_free_cash_flow_ratio(
            market_cap=pd.Series([1000, 500, 300, 160, 300]),
            free_cash_flow=pd.Series([5, 5, 5, 3, 100]),
        )
    )


def test_get_market_cap(recorder):
    recorder.capture(
        valuation_model.get_market_cap(
            share_price=pd.Series([1000, 500, 300, 160, 300]),
            total_shares_outstanding=pd.Series([500, 1000, 1500, 1000, 2000]),
        )
    )


def test_get_enterprise_value(recorder):
    recorder.capture(
        valuation_model.get_enterprise_value(
            market_cap=pd.Series([100000, 5000, 3000, 16000, 30000]),
            total_debt=pd.Series([500, 1000, 1500, 1000, 2000]),
            minority_interest=pd.Series([50, 100, 150, 10, 20]),
            preferred_equity=pd.Series([5, 100, 150, 100, 200]),
            cash_and_cash_equivalents=pd.Series([50, 100, 10, 100, 200]),
        )
    )


def test_get_ev_to_sales_ratio(recorder):
    recorder.capture(
        valuation_model.get_ev_to_sales_ratio(
            enterprise_value=pd.Series([100000, 5000, 3000, 16000, 30000]),
            total_revenue=pd.Series([1000, 1000, 1500, 1000, 2000]),
        )
    )


def test_get_ev_to_ebitda_ratio(recorder):
    recorder.capture(
        valuation_model.get_ev_to_ebitda_ratio(
            enterprise_value=pd.Series([100000, 5000, 3000, 16000, 30000]),
            operating_income=pd.Series([1000, 1000, 1500, 1000, 2000]),
            depreciation_and_amortization=pd.Series([100, 100, 150, 100, 200]),
        )
    )


def test_get_ev_to_operating_cashflow_ratio(recorder):
    recorder.capture(
        valuation_model.get_ev_to_operating_cashflow_ratio(
            enterprise_value=pd.Series([100000, 5000, 3000, 16000, 30000]),
            operating_cashflow=pd.Series([500, 300, 150.5, 100, 220]),
        )
    )


def test_get_earnings_yield(recorder):
    recorder.capture(
        valuation_model.get_earnings_yield(
            earnings_per_share=pd.Series([0.5, 0.4, 0.4, 1.4, 0.25]),
            market_price_per_share=pd.Series([100, 110, 80, 50, 30]),
        )
    )


def test_get_dividend_payout_ratio(recorder):
    recorder.capture(
        valuation_model.get_dividend_payout_ratio(
            dividends=pd.Series([0.6, 0.4, 0.15, 0.4, 0.25]),
            net_income=pd.Series([100, 120, 80, 50, 30]),
        )
    )


def test_issuance_measures_are_log_changes():
    issuance = valuation_model.get_net_share_issuance(
        shares=pd.Series([110.0, 90.0, 100.0]),
        beginning_shares=pd.Series([100.0, 100.0, 0.0]),
    )
    composite = valuation_model.get_composite_equity_issuance(
        market_cap=pd.Series([1200.0, 1000.0]),
        beginning_market_cap=pd.Series([1000.0, 1000.0]),
        total_return_price=pd.Series([11.0, 11.0]),
        beginning_total_return_price=pd.Series([10.0, 10.0]),
    )

    assert issuance.iloc[0] == pytest.approx(np.log(1.1))
    assert issuance.iloc[1] == pytest.approx(np.log(0.9))
    assert pd.isna(issuance.iloc[2])
    # A market cap that grows with the return alone is no issuance; one that lags it is
    # equity returned (here the whole return paid out).
    assert composite.iloc[0] == pytest.approx(np.log(1.2) - np.log(1.1))
    assert composite.iloc[1] == pytest.approx(-np.log(1.1))


def test_net_external_financing_takes_the_cash_flow_signs():
    result = valuation_model.get_net_external_financing(
        net_stock_issued=pd.Series([-30.0, 10.0]),
        dividends_paid=pd.Series([-10.0, 0.0]),
        net_debt_issued=pd.Series([20.0, 5.0]),
        average_total_assets=pd.Series([200.0, 0.0]),
    )

    assert result.iloc[0] == pytest.approx(-20 / 200)
    assert pd.isna(result.iloc[1])


def test_get_tangible_asset_value(recorder):
    recorder.capture(
        valuation_model.get_tangible_asset_value(
            total_assets=pd.Series([100, 120, 80, 50, 30]),
            total_liabilities=pd.Series([50, 30, 80, 20, 50]),
            goodwill_and_intangible_assets=pd.Series([1.6, 1.4, 1.15, 1.4, 1.25]),
        )
    )


def test_get_net_current_asset_value(recorder):
    recorder.capture(
        valuation_model.get_net_current_asset_value(
            total_current_assets=pd.Series([100, 120, 80, 50, 30]),
            total_liabilities=pd.Series([50, 30, 80, 20, 50]),
        )
    )


def test_free_cash_flow_payout_ratio_needs_a_positive_free_cash_flow():
    result = valuation_model.get_free_cash_flow_payout_ratio(
        dividends=pd.Series([-30.0, -30.0]),
        free_cash_flow=pd.Series([100.0, -10.0]),
    )

    assert result.iloc[0] == 0.3
    assert pd.isna(result.iloc[1])


def test_ev_to_invested_capital_and_price_to_tangible_book():
    ev_to_invested_capital = valuation_model.get_ev_to_invested_capital_ratio(
        enterprise_value=pd.Series([500.0, 500.0]),
        total_equity=pd.Series([150.0, 10.0]),
        total_debt=pd.Series([100.0, 10.0]),
        cash_and_short_term_investments=pd.Series([50.0, 40.0]),
    )
    price_to_tangible_book = valuation_model.get_price_to_tangible_book_ratio(
        market_cap=pd.Series([400.0, 400.0]),
        total_shareholder_equity=pd.Series([150.0, 150.0]),
        preferred_stock=pd.Series([10.0, 10.0]),
        goodwill_and_intangible_assets=pd.Series([40.0, 200.0]),
    )

    assert ev_to_invested_capital.iloc[0] == 2.5
    assert pd.isna(ev_to_invested_capital.iloc[1])
    assert price_to_tangible_book.iloc[0] == 4
    assert pd.isna(price_to_tangible_book.iloc[1])


def test_get_retention_ratio_is_what_is_not_paid_out():
    result = valuation_model.get_retention_ratio(pd.Series([0.25, 0.0, 1.2]))

    # A payout above earnings means a negative reinvestment.
    pd.testing.assert_series_equal(result, pd.Series([0.75, 1.0, -0.2]))


def test_get_basket_market_cap_totals_and_shares():
    market_cap = pd.DataFrame(
        {"A": [300.0, 200.0, np.nan], "B": [100.0, 200.0, 50.0]},
        index=pd.period_range("2021", "2023", freq="Y"),
    )

    basket = valuation_model.get_basket_market_cap(market_cap)
    share = valuation_model.get_basket_market_cap(market_cap, share=True)

    # A year without the market cap of one of the companies has no total.
    assert basket["Total"].tolist()[:2] == [400.0, 400.0]
    assert pd.isna(basket["Total"].iloc[2])
    assert share.iloc[0].tolist() == [0.75, 0.25, 1.0]
    assert share.iloc[2].isna().all()


def test_get_market_cap_to_gdp_uses_the_year_before():
    gross_domestic_product = pd.DataFrame(
        {"Country": [100.0, 200.0]}, index=pd.period_range("2024", "2025", freq="Y")
    )
    daily = pd.Series(
        [300.0, 400.0, 500.0],
        index=pd.PeriodIndex(["2025-12-31", "2026-01-02", "2027-01-04"], freq="D"),
    )
    yearly = pd.Series([300.0], index=pd.period_range("2025", "2025", freq="Y"))

    result = valuation_model.get_market_cap_to_gdp(daily, gross_domestic_product)

    # 2025 is divided by the 2024 GDP and 2026 by the 2025 GDP; 2026 isn't there for 2027.
    assert result["Country"].tolist()[:2] == [3.0, 2.0]
    assert pd.isna(result["Country"].iloc[2])
    assert valuation_model.get_market_cap_to_gdp(yearly, gross_domestic_product).iloc[
        0, 0
    ] == pytest.approx(3.0)


def test_multiples_of_non_positive_denominators_are_nan():
    ev_to_ebitda = valuation_model.get_ev_to_ebitda_ratio(
        enterprise_value=pd.Series([1000.0, 1000.0, 1000.0]),
        operating_income=pd.Series([100.0, -50.0, -20.0]),
        depreciation_and_amortization=pd.Series([20.0, 30.0, 20.0]),
    )
    price_to_book = valuation_model.get_price_to_book_ratio(
        price_per_share=pd.Series([50.0, 50.0, 50.0]),
        book_value_per_share=pd.Series([10.0, -5.0, 0.0]),
    )

    assert ev_to_ebitda.iloc[0] == pytest.approx(1000 / 120)
    assert ev_to_ebitda.iloc[1:].isna().all()
    assert price_to_book.iloc[0] == pytest.approx(5.0)
    assert price_to_book.iloc[1:].isna().all()


def test_get_cyclically_adjusted_price_to_earnings_ratio():
    periods = pd.period_range("2020", "2023", freq="Y")
    price = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]], index=["A"], columns=periods)
    eps = pd.DataFrame([[1.0, 2.0, -9.0, 4.0]], index=["A"], columns=periods)
    cpi = pd.DataFrame([[100.0, 110.0, 120.0, 132.0]], index=["A"], columns=periods)

    cape = valuation_model.get_cyclically_adjusted_price_to_earnings_ratio(
        price, eps, cpi, years=2
    )
    nominal = valuation_model.get_cyclically_adjusted_price_to_earnings_ratio(
        price, eps, years=2
    )

    # 2021: the 2020 EPS in 2021 prices is 1 x 110 / 100.
    assert cape.loc["A", periods[1]] == pytest.approx(20 / ((1.1 + 2) / 2))
    assert nominal.loc["A", periods[1]] == pytest.approx(20 / 1.5)
    # 2022 averages to a loss, so it is NaN; 2023 is (-9 x 132 / 120 + 4) / 2 < 0 as well.
    assert cape.loc["A", periods[2:]].isna().all()
    # 2020 has no year before it.
    assert pd.isna(cape.loc["A", periods[0]])

    # Quarterly: trailing twelve month EPS of the same quarter one year apart (lags 0 and 4).
    quarters = pd.period_range("2020Q1", "2021Q4", freq="Q")
    quarterly_eps = pd.DataFrame([[1.0] * 4 + [3.0] * 4], index=["A"], columns=quarters)
    quarterly_price = pd.DataFrame([[40.0] * 8], index=["A"], columns=quarters)
    quarterly = valuation_model.get_cyclically_adjusted_price_to_earnings_ratio(
        quarterly_price, quarterly_eps, years=2, periods_per_year=4
    )

    assert quarterly.loc["A", quarters[:4]].isna().all()
    assert quarterly.loc["A", quarters[4:]].tolist() == pytest.approx([20.0] * 4)


def test_get_dividend_adjusted_price_to_earnings_growth_ratio():
    price_earnings = pd.Series([30.0, 11.0, 12.0])
    growth_rate = pd.Series([0.2, 0.1, 0.05])
    dividend_yield = pd.Series([0.0, 0.02, 0.03])

    pegy = valuation_model.get_dividend_adjusted_price_to_earnings_growth_ratio(
        price_earnings, growth_rate, dividend_yield
    )

    # Growth and yield enter in percent: 12 / (5 + 3) = 1.5.
    assert pegy.tolist() == pytest.approx([30 / 20, 11 / 12, 12 / 8])
    # Without a dividend it is the PEG ratio.
    assert pegy.iloc[0] == pytest.approx(
        valuation_model.get_price_to_earnings_growth_ratio(30.0, 0.2 * 100)
    )


def test_cyclically_adjusted_price_to_earnings_ratio_counts_calendar_years():
    # 2021 is not reported, so the two-year window of 2022 lacks a year.
    years = pd.PeriodIndex(["2019", "2020", "2022", "2023"], freq="Y")
    eps = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=["XYZ"], columns=years)
    price = pd.DataFrame([[10.0] * 4], index=["XYZ"], columns=years)

    cape = valuation_model.get_cyclically_adjusted_price_to_earnings_ratio(
        price, eps, years=2
    )

    # 2020 averages 2019 and 2020, 2022 has no 2021 and 2023 averages 2022 and 2023.
    assert cape.loc["XYZ", years[1]] == pytest.approx(10 / 1.5)
    assert pd.isna(cape.loc["XYZ", years[2]])
    assert cape.loc["XYZ", years[3]] == pytest.approx(10 / 3.5)


def test_net_net_working_capital_per_share_reference_values():
    # Apple's fiscal 2022 balance sheet (millions) and diluted shares: (48,304 + 0.75 x 60,932 +
    # 0.5 x 4,946 - 302,083) / 16,325.819 = -12.594, the hand calculation of FinancialModelingPrep's
    # grahamNetNet definition (on the diluted instead of the basic shares).
    result = valuation_model.get_net_net_working_capital_per_share(
        48304.0, 60932.0, 4946.0, 302083.0, 16325.819
    )
    assert result == pytest.approx(-12.593978, abs=1e-6)

    # Graham's upper ends of the liquidation ranges value the current assets higher.
    upper = valuation_model.get_net_net_working_capital_per_share(
        48304.0, 60932.0, 4946.0, 302083.0, 16325.819, 0.9, 0.75
    )
    assert upper > result


def test_per_share_cash_flows_and_tangible_book_value():
    assert valuation_model.get_operating_cash_flow_per_share(
        122151.0, 16325.819
    ) == pytest.approx(7.482075, abs=1e-6)
    assert valuation_model.get_free_cash_flow_per_share(
        111443.0, 16325.819
    ) == pytest.approx(6.826181, abs=1e-6)
    assert valuation_model.get_cash_per_share(48304.0, 16325.819) == pytest.approx(
        2.958749, abs=1e-6
    )
    # Microsoft 2022: (166,542 - 0 - 78,822) / 7,540.
    assert valuation_model.get_tangible_book_value_per_share(
        166542.0, 0.0, 78822.0, 7540.0
    ) == pytest.approx(11.633952, abs=1e-6)
