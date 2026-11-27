"""Financial Constraints Model Tests"""

import math

import pandas as pd

from financetoolkit.models.constraints import constraints_model


def test_get_cash_flow_to_capital(recorder):
    recorder.capture(
        constraints_model.get_cash_flow_to_capital(
            income_before_extraordinary_items=pd.Series([100, -50, 20, 0]),
            depreciation_and_amortization=pd.Series([20, 30, 10, 5]),
            lagged_capital=pd.Series([400, 300, 0, 100]),
        )
    )


def test_get_kaplan_zingales_q(recorder):
    recorder.capture(
        constraints_model.get_kaplan_zingales_q(
            total_assets=pd.Series([1000, 500, 200]),
            market_value_of_equity=pd.Series([3000, 200, 50]),
            common_equity=pd.Series([400, 250, 100]),
            deferred_taxes=pd.Series([20, 0, 5]),
        )
    )


def test_get_debt_to_total_capital(recorder):
    recorder.capture(
        constraints_model.get_debt_to_total_capital(
            short_term_debt=pd.Series([50, 0, 20]),
            long_term_debt=pd.Series([150, 100, 30]),
            stockholders_equity=pd.Series([300, -40, -50]),
        )
    )


def test_get_dividend_payer_indicator():
    indicator = constraints_model.get_dividend_payer_indicator(
        pd.Series([-15.0, 0.0, float("nan"), 3.0])
    )

    assert indicator.iloc[[0, 1, 3]].tolist() == [1.0, 0.0, 1.0]
    assert math.isnan(indicator.iloc[2])


def test_get_log_of_total_assets_is_in_millions():
    log_assets = constraints_model.get_log_of_total_assets_in_millions(
        pd.Series([1_000_000.0, 100e9, 0.0])
    )

    assert log_assets.iloc[0] == 0.0
    assert abs(log_assets.iloc[1] - math.log(100_000)) < 1e-12
    assert math.isnan(log_assets.iloc[2])


def test_get_industry_sales_growth_averages_the_industry_and_keeps_loners_own():
    sales_growth = pd.DataFrame(
        {
            "2023": [0.10, 0.30, 0.05, float("nan")],
            "2024": [0.2, float("nan"), 0.1, 0.4],
        },
        index=["F", "GM", "SMP", "Total"],
    )
    industries = pd.Series({"F": "Autos", "GM": "Autos", "SMP": None})

    industry = constraints_model.get_industry_sales_growth(sales_growth, industries)

    assert industry.loc[["F", "GM"], "2023"].tolist() == [0.2, 0.2]
    # GM has no growth in 2024, so F's industry is F alone.
    assert industry.loc[["F", "GM"], "2024"].tolist() == [0.2, 0.2]
    assert industry.loc["SMP"].tolist() == [0.05, 0.1]
    assert industry.loc["Total", "2024"] == 0.4


def test_kaplan_zingales_index_uses_the_lamont_polk_saa_requejo_coefficients():
    # Lamont, Polk and Saa-Requejo (2001), Appendix.
    index = constraints_model.get_kaplan_zingales_index(
        cash_flow_to_capital=0.5,
        q=2.0,
        debt_to_total_capital=0.4,
        dividends_to_capital=0.02,
        cash_to_capital=0.3,
    )
    expected = (
        -1.001909 * 0.5
        + 0.2826389 * 2.0
        + 3.139193 * 0.4
        - 39.3678 * 0.02
        - 1.314759 * 0.3
    )

    assert abs(index - expected) < 1e-12


def test_whited_wu_index_uses_equation_13():
    # Whited and Wu (2006), equation 13 (Table 1, column 4).
    index = constraints_model.get_whited_wu_index(
        cash_flow_to_total_assets=0.1,
        dividend_payer_indicator=1.0,
        long_term_debt_to_total_assets=0.2,
        log_of_total_assets=math.log(500),
        industry_sales_growth=0.05,
        sales_growth=0.08,
    )
    expected = (
        -0.091 * 0.1
        - 0.062
        + 0.021 * 0.2
        - 0.044 * math.log(500)
        + 0.102 * 0.05
        - 0.035 * 0.08
    )

    assert abs(index - expected) < 1e-12


def test_size_age_index_caps_size_and_age():
    size = constraints_model.get_size_age_size(
        total_assets=pd.Series([100e6, 1e12, 200e6, -5.0]),
        price_level=pd.Series([1.0, 1.7, 2.0, 1.0]),
    )
    age = constraints_model.get_size_age_age(pd.Series([10, 60, -1, 5]))
    index = constraints_model.get_size_age_index(size, age)

    assert abs(size.iloc[0] - math.log(100)) < 1e-12
    assert abs(size.iloc[1] - math.log(4_500)) < 1e-12
    # $200 million at twice the 2004 price level is $100 million of 2004 dollars.
    assert abs(size.iloc[2] - math.log(100)) < 1e-12
    assert math.isnan(size.iloc[3])
    assert age.iloc[:2].tolist() == [10, 37]
    assert math.isnan(age.iloc[2])
    # A firm listed 10 years with $100 million of 2004 assets.
    assert abs(index.iloc[0] - (-2.8821)) < 1e-4
    # Hadlock and Pierce's floor, a firm at both caps.
    assert abs(index.iloc[1] - (-4.6369)) < 1e-4
