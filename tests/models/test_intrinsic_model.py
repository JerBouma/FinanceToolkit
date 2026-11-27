"""Intrinsic Model Tests""" ""

import pandas as pd
import pytest

from financetoolkit.models.valuation import intrinsic_model


def test_get_intrinsic_value(recorder):
    recorder.capture(
        intrinsic_model.get_intrinsic_value(
            cash_flow=500,
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.1,
            cash_and_cash_equivalents=100,
            total_debt=100,
            shares_outstanding=100,
            periods=5,
        )
    )


def test_get_free_cash_flow_to_firm(recorder):
    recorder.capture(
        intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes=pd.Series([750, 800, 820, 780, 900]),
            depreciation_and_amortization=pd.Series([100, 105, 110, 108, 115]),
            capital_expenditure=pd.Series([200, 210, 220, 190, 230]),
            change_in_net_working_capital=pd.Series([50, -20, 30, 10, -5]),
        )
    )
    # Synthetic ground-truth: FCFF = 750 + 100 - 200 - 50 = 600
    assert (
        intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes=750,
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
        )
        == 600
    )


def test_get_free_cash_flow_to_firm_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_free_cash_flow_to_firm(
            net_operating_profit_after_taxes="not a number",
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
        )


def test_get_free_cash_flow_to_equity(recorder):
    recorder.capture(
        intrinsic_model.get_free_cash_flow_to_equity(
            net_income=pd.Series([800, 850, 870, 830, 950]),
            depreciation_and_amortization=pd.Series([100, 105, 110, 108, 115]),
            capital_expenditure=pd.Series([200, 210, 220, 190, 230]),
            change_in_net_working_capital=pd.Series([50, -20, 30, 10, -5]),
            net_borrowing=pd.Series([30, -10, 15, 5, -25]),
        )
    )
    # Synthetic ground-truth: FCFE = 800 + 100 - 200 - 50 + 30 = 680
    assert (
        intrinsic_model.get_free_cash_flow_to_equity(
            net_income=800,
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
            net_borrowing=30,
        )
        == 680
    )


def test_get_free_cash_flow_to_equity_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_free_cash_flow_to_equity(
            net_income="not a number",
            depreciation_and_amortization=100,
            capital_expenditure=200,
            change_in_net_working_capital=50,
            net_borrowing=30,
        )


def test_get_two_stage_dividend_discount_model(recorder):
    recorder.capture(
        intrinsic_model.get_two_stage_dividend_discount_model(
            dividends_per_share=2.0,
            rate_of_return=0.10,
            high_growth_rate=0.20,
            stable_growth_rate=0.03,
            high_growth_periods=3,
        )
    )
    # Ground truth by hand: PV 7.1585 plus a terminal 38.2063 gives about 45.3648.
    result = intrinsic_model.get_two_stage_dividend_discount_model(
        dividends_per_share=2.0,
        rate_of_return=0.10,
        high_growth_rate=0.20,
        stable_growth_rate=0.03,
        high_growth_periods=3,
    )
    intrinsic_value = result.loc["Intrinsic Value"].iloc[0]
    assert round(intrinsic_value, 4) == 45.3648


def test_get_two_stage_dividend_discount_model_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_two_stage_dividend_discount_model(
            dividends_per_share="not a number",
            rate_of_return=0.10,
            high_growth_rate=0.20,
            stable_growth_rate=0.03,
        )


def test_get_residual_income(recorder):
    recorder.capture(
        intrinsic_model.get_residual_income(
            net_income=pd.Series([1000, 1050, 1100, 950, 1200]),
            cost_of_equity=pd.Series([0.10, 0.11, 0.095, 0.12, 0.09]),
            book_value_of_equity=pd.Series([6000, 6100, 6200, 6300, 6400]),
        )
    )
    # Synthetic ground-truth: RI = 1000 - (0.10 * 6000) = 400
    assert (
        intrinsic_model.get_residual_income(
            net_income=1000,
            cost_of_equity=0.10,
            book_value_of_equity=6000,
        )
        == 400
    )


def test_get_residual_income_type_error():
    with pytest.raises(TypeError):
        intrinsic_model.get_residual_income(
            net_income="not a number",
            cost_of_equity=0.10,
            book_value_of_equity=6000,
        )


def test_graham_number_needs_positive_earnings_and_book_value():
    import numpy as np

    graham_number = intrinsic_model.get_graham_number(
        earnings_per_share=pd.Series([2.0, -2.0, 2.0]),
        book_value_per_share=pd.Series([10.0, -10.0, -10.0]),
    )

    assert graham_number.iloc[0] == np.sqrt(22.5 * 2 * 10)
    assert graham_number.iloc[1:].isna().all()


def test_valuations_require_a_discount_rate_above_the_perpetual_growth_rate():
    with pytest.raises(ValueError, match="Perpetual Growth Rate"):
        intrinsic_model.get_intrinsic_value(
            cash_flow=500,
            growth_rate=0.05,
            perpetual_growth_rate=0.1,
            weighted_average_cost_of_capital=0.1,
            cash_and_cash_equivalents=100,
            total_debt=100,
            shares_outstanding=100,
        )

    with pytest.raises(ValueError, match="Growth Rate"):
        intrinsic_model.get_gordon_growth_model(
            dividends_per_share=1.0, rate_of_return=0.05, growth_rate=0.06
        )

    with pytest.raises(ValueError, match="Stable Growth Rate"):
        intrinsic_model.get_two_stage_dividend_discount_model(
            dividends_per_share=1.0,
            rate_of_return=0.08,
            high_growth_rate=0.15,
            stable_growth_rate=0.08,
        )

    # The high-growth phase is discounted explicitly and may exceed the rate of return.
    two_stage = intrinsic_model.get_two_stage_dividend_discount_model(
        dividends_per_share=1.0,
        rate_of_return=0.08,
        high_growth_rate=0.15,
        stable_growth_rate=0.03,
    )
    assert two_stage.loc["Intrinsic Value"].iloc[0] > 0


def test_residual_income_charges_the_beginning_book_value():
    # RI_t = NI_t - r * B_t-1
    residual_income = intrinsic_model.get_residual_income(
        net_income=pd.Series([10.0, 12.0]),
        cost_of_equity=0.1,
        book_value_of_equity=pd.Series([100.0, 110.0]).shift(1),
    )

    assert residual_income.isna().iloc[0]
    assert residual_income.iloc[1] == pytest.approx(12.0 - 0.1 * 100.0)


def test_intrinsic_value_adds_back_the_short_term_investments():
    inputs = {
        "cash_flow": 500,
        "growth_rate": 0.05,
        "perpetual_growth_rate": 0.02,
        "weighted_average_cost_of_capital": 0.1,
        "cash_and_cash_equivalents": 100,
        "total_debt": 100,
        "shares_outstanding": 100,
    }
    without = intrinsic_model.get_intrinsic_value(**inputs)
    with_investments = intrinsic_model.get_intrinsic_value(
        **inputs, short_term_investments=50
    )

    assert with_investments.loc["Enterprise Value"].iloc[0] == pytest.approx(
        without.loc["Enterprise Value"].iloc[0]
    )
    assert with_investments.loc["Equity Value"].iloc[0] == pytest.approx(
        without.loc["Equity Value"].iloc[0] + 50
    )


def test_get_implied_growth_rate_recovers_the_dcf_growth():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    value = float(
        intrinsic_model.get_intrinsic_value(100, 0.05, 0.025, 0.09, 50, 80, 10, 5)
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    implied = intrinsic_model.get_implied_growth_rate(
        value, 100, 0.025, 0.09, 50, 80, 10, 5
    )
    assert implied == pytest.approx(0.05, abs=1e-8)
    assert np.isnan(
        intrinsic_model.get_implied_growth_rate(10, -5, 0.02, 0.09, 0, 0, 1)
    )


def test_get_earnings_power_value():
    import pandas as pd  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    result = intrinsic_model.get_earnings_power_value(
        revenue=pd.Series([100.0, 120.0]),
        operating_income=pd.Series([20.0, 30.0]),
        income_tax_expense=pd.Series([4.0, 5.0]),
        income_before_tax=pd.Series([20.0, 25.0]),
        weighted_average_cost_of_capital=0.1,
        cash_and_cash_equivalents=10,
        total_debt=30,
        shares_outstanding=10,
    )
    margin = (0.2 + 0.25) / 2
    nopat = margin * 120 * (1 - 0.2)
    assert result["Normalized NOPAT"] == pytest.approx(nopat)
    assert result["Earnings Power Value per Share"] == pytest.approx(
        (nopat / 0.1 - 20) / 10
    )


def test_get_monte_carlo_intrinsic_value_without_uncertainty():
    import pytest  # noqa: PLC0415

    values = intrinsic_model.get_monte_carlo_intrinsic_value(
        100, 0.05, 0.0, 0.025, 0.0, 0.09, 0.0, 50, 80, 10, 5, simulations=50
    )
    deterministic = float(
        intrinsic_model.get_intrinsic_value(100, 0.05, 0.025, 0.09, 50, 80, 10, 5)
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    assert values == pytest.approx(deterministic)


def test_implied_growth_rate_is_missing_with_a_missing_input():
    import numpy as np

    growth = intrinsic_model.get_implied_growth_rate(
        cash_flow=100.0,
        weighted_average_cost_of_capital=np.nan,
        perpetual_growth_rate=0.02,
        cash_and_cash_equivalents=10.0,
        total_debt=20.0,
        shares_outstanding=10.0,
        share_price=50.0,
    )

    assert np.isnan(growth)


def test_h_model_is_gordon_plus_the_extra_growth():
    value = intrinsic_model.get_h_model(
        dividends_per_share=2.0,
        rate_of_return=0.10,
        high_growth_rate=0.20,
        stable_growth_rate=0.05,
        high_growth_periods=8,
    )
    # Gordon at 5%: 2 * 1.05 / 0.05 = 42; extra growth: 2 * 4 * 0.15 / 0.05 = 24.
    assert value == pytest.approx(66.0)

    # Without extra growth it is the Gordon Growth Model.
    assert intrinsic_model.get_h_model(2.0, 0.10, 0.05, 0.05) == pytest.approx(
        intrinsic_model.get_gordon_growth_model(2.0, 0.10, 0.05)
    )

    with pytest.raises(ValueError):
        intrinsic_model.get_h_model(2.0, 0.05, 0.20, 0.05)


def test_three_stage_model_discounts_a_linearly_declining_growth_rate():
    result = intrinsic_model.get_two_stage_dividend_discount_model(
        dividends_per_share=1.0,
        rate_of_return=0.09,
        high_growth_rate=0.15,
        stable_growth_rate=0.03,
        high_growth_periods=3,
        transition_periods=3,
    )

    # The dividends grow 15% for three years, then 12%, 9% and 6%, then 3% forever.
    growth_rates = [0.15, 0.15, 0.15, 0.12, 0.09, 0.06]
    dividends, dividend = [], 1.0
    for growth_rate in growth_rates:
        dividend *= 1 + growth_rate
        dividends.append(dividend)
    expected = sum(value / 1.09**year for year, value in enumerate(dividends, 1))
    expected += dividends[-1] * 1.03 / (0.09 - 0.03) / 1.09**6

    assert result.loc["Intrinsic Value"].iloc[0] == pytest.approx(expected)
    assert "Transition Phase Present Value" in result.index


def test_two_stage_model_without_transition_is_unchanged():
    result = intrinsic_model.get_two_stage_dividend_discount_model(
        1.0, 0.09, 0.15, 0.03, 3
    )

    assert "Transition Phase Present Value" not in result.index
    assert list(result.columns) == ["High-Growth Periods = 3"]


def test_justified_multiples():
    payout_ratio = pd.Series([0.6, 0.6])
    return_on_equity = pd.Series([0.15, 0.15])
    growth_rate = pd.Series([0.06, 0.12])

    trailing = intrinsic_model.get_justified_price_to_earnings(
        payout_ratio, 0.10, growth_rate
    )
    leading = intrinsic_model.get_justified_price_to_earnings(
        payout_ratio, 0.10, growth_rate, leading=True
    )
    price_to_book = intrinsic_model.get_justified_price_to_book(
        return_on_equity, 0.10, growth_rate
    )

    assert trailing.iloc[0] == pytest.approx(0.6 * 1.06 / 0.04)
    assert leading.iloc[0] == pytest.approx(0.6 / 0.04)
    assert price_to_book.iloc[0] == pytest.approx(0.09 / 0.04)
    # Growth at or above the rate of return has no finite value.
    assert (
        trailing.isna().iloc[1]
        and leading.isna().iloc[1]
        and price_to_book.isna().iloc[1]
    )


def test_intrinsic_value_deducts_minority_interest_and_preferred_stock():
    base = intrinsic_model.get_intrinsic_value(100, 0.05, 0.02, 0.10, 50, 200, 10)
    claims = intrinsic_model.get_intrinsic_value(
        100,
        0.05,
        0.02,
        0.10,
        50,
        200,
        10,
        short_term_investments=30,
        minority_interest=40,
        preferred_stock=25,
    )
    # Damodaran: Equity = EV + cash + marketable securities - debt - minority interest
    # - preferred stock; the Enterprise Value itself does not change.
    assert claims.loc["Enterprise Value"].iloc[0] == pytest.approx(
        base.loc["Enterprise Value"].iloc[0]
    )
    assert claims.loc["Equity Value"].iloc[0] == pytest.approx(
        base.loc["Equity Value"].iloc[0] + 30 - 40 - 25
    )
    assert claims.loc["Intrinsic Value"].iloc[0] == pytest.approx(
        (base.loc["Equity Value"].iloc[0] + 30 - 40 - 25) / 10
    )


def test_net_cash_paths_round_trip_with_the_intrinsic_value():
    capital = {
        "cash_and_cash_equivalents": 50,
        "short_term_investments": 30,
        "total_debt": 200,
        "minority_interest": 40,
        "preferred_stock": 25,
        "shares_outstanding": 10,
    }
    value = (
        intrinsic_model.get_intrinsic_value(100, 0.07, 0.02, 0.10, periods=5, **capital)
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    # The growth the intrinsic value implies is the growth it was calculated with.
    implied = intrinsic_model.get_implied_growth_rate(
        share_price=value,
        cash_flow=100,
        perpetual_growth_rate=0.02,
        weighted_average_cost_of_capital=0.10,
        periods=5,
        **capital,
    )
    assert implied == pytest.approx(0.07, abs=1e-8)

    simulated = intrinsic_model.get_monte_carlo_intrinsic_value(
        100, 0.07, 0, 0.02, 0, 0.10, 0, periods=5, simulations=3, **capital
    )
    assert simulated == pytest.approx([value] * 3)

    epv = intrinsic_model.get_earnings_power_value(
        revenue=pd.Series([1000.0]),
        operating_income=pd.Series([200.0]),
        income_tax_expense=pd.Series([40.0]),
        income_before_tax=pd.Series([200.0]),
        weighted_average_cost_of_capital=0.10,
        **capital,
    )
    assert epv["Equity Value"] == pytest.approx(
        200 * 0.8 / 0.10 + 50 + 30 - 200 - 40 - 25
    )


def test_graham_intrinsic_value_matches_grahams_table():
    # The Intelligent Investor (chapter 11) tabulates the multiple of current earnings his
    # formula gives per expected growth rate; at the 1962 AAA yield of 4.4% the 4.4 / Y
    # adjustment is 1, so an EPS of 1 returns that multiple.
    growth_rates = pd.Series([0.0, 0.025, 0.05, 0.072, 0.10, 0.143, 0.20])
    multiples = [8.5, 13.5, 18.5, 22.9, 28.5, 37.1, 48.5]

    values = intrinsic_model.get_graham_intrinsic_value(
        earnings_per_share=1.0, growth_rate=growth_rates, bond_yield=0.044
    )

    assert values.round(1).tolist() == multiples

    # A higher bond yield scales the value down by 4.4 / Y.
    assert intrinsic_model.get_graham_intrinsic_value(
        2.0, 0.05, 0.055
    ) == pytest.approx(2.0 * 18.5 * 4.4 / 5.5)


def test_graham_intrinsic_value_needs_positive_earnings_and_bond_yield():
    import numpy as np

    values = intrinsic_model.get_graham_intrinsic_value(
        earnings_per_share=pd.Series([2.0, -2.0, 2.0, 0.0]),
        growth_rate=0.05,
        bond_yield=pd.Series([0.05, 0.05, 0.0, 0.05]),
    )

    assert values.iloc[0] == pytest.approx(2.0 * 18.5 * 4.4 / 5)
    assert values.iloc[1:].isna().all()
    assert np.isnan(intrinsic_model.get_graham_intrinsic_value(-1.0, 0.05, 0.05))


def test_get_owner_earnings_adds_back_depreciation_and_subtracts_maintenance():
    result = intrinsic_model.get_owner_earnings(
        net_income=pd.Series([100.0]),
        depreciation_and_amortization=pd.Series([30.0]),
        maintenance_capital_expenditure=pd.Series([25.0]),
        change_in_working_capital=pd.Series([5.0]),
    )
    assert result[0] == pytest.approx(100.0)


def test_get_greenwald_maintenance_capital_expenditure():
    result = intrinsic_model.get_greenwald_maintenance_capital_expenditure(
        capital_expenditure=pd.Series([50.0, 50.0, 10.0]),
        property_plant_and_equipment_to_revenue=pd.Series([0.5, 0.5, 0.5]),
        change_in_revenue=pd.Series([40.0, -40.0, 100.0]),
    )
    # 50 - 0.5 * 40; a fall in revenue requires no growth investment; growth above the
    # capital expenditure leaves no maintenance part rather than a negative one.
    assert list(result) == pytest.approx([30.0, 50.0, 0.0])


def test_get_intrinsic_value_present_values_add_up_to_the_enterprise_value():
    # Grown at 37.99% for five years, discounted at 12.61% with 2.5% perpetual growth:
    # a terminal value of 4,642.66 is worth 2,563.79 today, the rest of the enterprise
    # value of 3,441.03 being the present value of the five projected cash flows.
    growth, perpetual, wacc = 0.3799, 0.025, 0.1261
    last = 4642.66 * (wacc - perpetual) / (1 + perpetual)
    result = intrinsic_model.get_intrinsic_value(
        cash_flow=last / (1 + growth) ** 5,
        growth_rate=growth,
        perpetual_growth_rate=perpetual,
        weighted_average_cost_of_capital=wacc,
        cash_and_cash_equivalents=0,
        total_debt=0,
        shares_outstanding=1,
        periods=5,
    ).iloc[:, 0]
    projections = [last / (1 + growth) ** (5 - t) for t in range(1, 6)]
    assert result["Present Value of Cash Flows"] == pytest.approx(
        sum(cf / (1 + wacc) ** t for t, cf in enumerate(projections, 1)), rel=1e-12
    )
    assert result["Present Value of Terminal Value"] == pytest.approx(2563.79, abs=0.01)
    assert result["Enterprise Value"] == pytest.approx(3441.03, abs=0.01)
    assert result["Present Value of Cash Flows"] == pytest.approx(877.25, abs=0.01)
    assert result["Present Value of Cash Flows"] + result[
        "Present Value of Terminal Value"
    ] == pytest.approx(result["Enterprise Value"], rel=1e-12)
