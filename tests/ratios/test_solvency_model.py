"""Solvency Model Tests""" ""
import pandas as pd
import pytest

from financetoolkit.ratios import solvency_model


def test_get_debt_to_assets_ratio(recorder):
    recorder.capture(
        solvency_model.get_debt_to_assets_ratio(
            total_debt=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([200, 140, 160, 160, 300]),
        )
    )


def test_get_debt_to_equity_ratio(recorder):
    recorder.capture(
        solvency_model.get_debt_to_equity_ratio(
            total_debt=pd.Series([100, 110, 120, 130, 80]),
            total_equity=pd.Series([150, 130, 120, 100, 50]),
        )
    )


def test_get_interest_coverage_ratio(recorder):
    recorder.capture(
        solvency_model.get_interest_coverage_ratio(
            operating_income=pd.Series([100, 110, 120, 130, 80]),
            interest_expense=pd.Series([15, 13, 12, 10.3, 5]),
        )
    )


def test_get_debt_service_coverage_ratio(recorder):
    recorder.capture(
        solvency_model.get_debt_service_coverage_ratio(
            operating_income=pd.Series([100, 110, 120, 130, 80]),
            depreciation_and_amortization=pd.Series([30, 20, 10, 10, 0]),
            interest_expense=pd.Series([15, 13, 12, 10.3, 5]),
            short_term_debt=pd.Series([30, 20, 10, 10, 15]),
        )
    )


def test_get_equity_multiplier(recorder):
    recorder.capture(
        solvency_model.get_equity_multiplier(
            average_total_assets=pd.Series([100, 110, 120, 130, 180]),
            average_total_equity=pd.Series([150, 130, 120, 100, 50]),
        )
    )


def test_get_free_cash_flow_yield(recorder):
    recorder.capture(
        solvency_model.get_free_cash_flow_yield(
            free_cash_flow=pd.Series([200, 210, 220, 230, 280]),
            market_capitalization=pd.Series([1000, 1300, 1200, 1000, 500]),
        )
    )


def test_get_net_debt_to_ebitda_ratio(recorder):
    recorder.capture(
        solvency_model.get_net_debt_to_ebitda_ratio(
            operating_income=pd.Series([200, 210, 220, 230, 280]),
            depreciation_and_amortization=pd.Series([20, 21, 22, 23, 28]),
            net_debt=pd.Series([1000, 1300, 1200, 1000, 500]),
        )
    )


def test_get_cash_flow_coverage_ratio(recorder):
    recorder.capture(
        solvency_model.get_cash_flow_coverage_ratio(
            operating_cash_flow=pd.Series([200, 210, 220, 230, 280]),
            total_debt=pd.Series([1000, 1300, 1200, 1000, 500]),
        )
    )


def test_get_capex_coverage_ratio(recorder):
    recorder.capture(
        solvency_model.get_capex_coverage_ratio(
            cash_flow_from_operations=pd.Series([200, 210, 220, 230, 280]),
            capital_expenditure=pd.Series([10, 5, 3, 5, 0]),
        )
    )


def test_get_dividend_capex_coverage_ratio(recorder):
    recorder.capture(
        solvency_model.get_dividend_capex_coverage_ratio(
            cash_flow_from_operations=pd.Series([200, 210, 220, 230, 280]),
            capital_expenditure=pd.Series([10, 5, 3, 5, 0]),
            dividends=pd.Series([1, 3, 2, 0, 5]),
        )
    )


def test_get_interest_coverage_ratio_uses_ebit_and_masks_zero_interest():
    result = solvency_model.get_interest_coverage_ratio(
        operating_income=pd.Series([100.0, 50.0]),
        interest_expense=pd.Series([20.0, 0.0]),
    )

    assert result.iloc[0] == 5
    assert pd.isna(result.iloc[1])


def test_get_debt_service_coverage_ratio_uses_interest_and_principal():
    # (EBIT + D&A) / (interest + short-term debt); no debt service gives NaN.
    result = solvency_model.get_debt_service_coverage_ratio(
        operating_income=pd.Series([80.0, 10.0]),
        depreciation_and_amortization=pd.Series([20.0, 5.0]),
        interest_expense=pd.Series([10.0, 0.0]),
        short_term_debt=pd.Series([40.0, 0.0]),
    )

    assert result.iloc[0] == 2
    assert pd.isna(result.iloc[1])


def test_get_preferred_dividend_coverage_ratio_masks_zero_dividends():
    result = solvency_model.get_preferred_dividend_coverage_ratio(
        net_income=pd.Series([100.0, 100.0]),
        preferred_dividends=pd.Series([-20.0, 0.0]),
    )

    assert result.iloc[0] == 5
    assert pd.isna(result.iloc[1])


def test_get_debt_service_coverage_ratio_scales_the_principal_to_a_quarter():
    # A quarter of earnings and interest against a quarter of the principal due in a year.
    result = solvency_model.get_debt_service_coverage_ratio(
        operating_income=pd.Series([20.0]),
        depreciation_and_amortization=pd.Series([5.0]),
        interest_expense=pd.Series([2.5]),
        short_term_debt=pd.Series([40.0]),
        window_years=0.25,
    )

    assert result.iloc[0] == 2


def test_ebitda_coverages_need_an_interest_expense():
    coverage = solvency_model.get_ebitda_interest_coverage_ratio(
        operating_income=pd.Series([80.0, 80.0]),
        depreciation_and_amortization=pd.Series([20.0, 20.0]),
        interest_expense=pd.Series([10.0, 0.0]),
    )
    minus_capex = solvency_model.get_ebitda_minus_capex_interest_coverage_ratio(
        operating_income=pd.Series([80.0]),
        depreciation_and_amortization=pd.Series([20.0]),
        capital_expenditure=pd.Series([-30.0]),
        interest_expense=pd.Series([10.0]),
    )

    assert coverage.iloc[0] == 10
    assert pd.isna(coverage.iloc[1])
    assert minus_capex.iloc[0] == 7


def test_cash_interest_coverage_adds_back_interest_and_taxes_paid():
    result = solvency_model.get_cash_interest_coverage_ratio(
        cash_flow_from_operations=pd.Series([80.0, 80.0]),
        interest_paid=pd.Series([10.0, 0.0]),
        income_taxes_paid=pd.Series([10.0, 10.0]),
    )

    assert result.iloc[0] == 10
    assert pd.isna(result.iloc[1])


def test_funds_from_operations_to_debt_follows_s_and_p():
    result = solvency_model.get_funds_from_operations_to_debt_ratio(
        operating_income=pd.Series([80.0, 80.0, 80.0]),
        depreciation_and_amortization=pd.Series([20.0, 20.0, 20.0]),
        stock_based_compensation=pd.Series([10.0, 10.0, 10.0]),
        interest_paid=pd.Series([5.0, 0.0, 5.0]),
        income_taxes_paid=pd.Series([25.0, 25.0, 25.0]),
        total_debt=pd.Series([200.0, 200.0, 0.0]),
    )

    # FFO = 80 + 20 + 10 - 5 - 25 = 80; an unreported interest paid or no debt is NaN.
    assert result.iloc[0] == 0.4
    assert result.iloc[1:].isna().all()


def test_discretionary_cash_flow_to_debt_deducts_capex_dividends_and_buybacks():
    result = solvency_model.get_discretionary_cash_flow_to_debt_ratio(
        cash_flow_from_operations=pd.Series([100.0]),
        capital_expenditure=pd.Series([-20.0]),
        dividends_paid=pd.Series([-10.0]),
        common_stock_purchased=pd.Series([-30.0]),
        total_debt=pd.Series([200.0]),
    )

    assert result.iloc[0] == 0.2


def test_degree_of_leverage_divides_percentage_changes():
    result = solvency_model.get_degree_of_leverage(
        result=pd.DataFrame([[100.0, 130.0, 130.0, 65.0]]),
        driver=pd.DataFrame([[200.0, 220.0, 220.0, 198.0]]),
    )

    # +30% on +10%, no change in the driver, -50% on -10%.
    assert pd.isna(result.iloc[0, 0])
    assert result.iloc[0, 1] == pytest.approx(3)
    assert pd.isna(result.iloc[0, 2])
    assert result.iloc[0, 3] == pytest.approx(5)


def test_degree_of_leverage_keeps_the_sign_from_a_negative_base():
    result = solvency_model.get_degree_of_leverage(
        result=pd.DataFrame([[-100.0, -50.0]]),
        driver=pd.DataFrame([[200.0, 220.0]]),
    )

    # A loss narrowing by half is a change of +50%, on a revenue change of +10%.
    assert result.iloc[0, 1] == pytest.approx(5)
