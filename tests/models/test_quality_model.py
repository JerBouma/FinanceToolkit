"""Quality Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models import quality_model

# pylint: disable=missing-function-docstring

PERIODS = ["2021", "2022", "2023"]


def frame(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=PERIODS).T


def test_magic_formula_ranks_cheap_and_good_companies_first():
    result = quality_model.get_magic_formula(
        operating_income=frame({"A": [10, 10, 10], "B": [10, 10, 10]}),
        enterprise_value=frame({"A": [100, 100, 100], "B": [200, 200, 200]}),
        net_working_capital=frame({"A": [10, 10, 10], "B": [40, 40, 40]}),
        net_fixed_assets=frame({"A": [10, 10, 10], "B": [10, 10, 10]}),
    )
    assert (result["Magic Formula Rank"].loc["A"] == 1).all()
    assert result["Earnings Yield"].loc["A", "2021"] == pytest.approx(0.1)
    assert result["Return on Capital"].loc["B", "2021"] == pytest.approx(0.2)


def test_dechow_f_score_matches_the_published_formula():
    ones = frame({"A": [1.0, 1.0, 1.0]})
    result = quality_model.get_dechow_f_score(
        total_assets=frame({"A": [100.0, 110.0, 120.0]}),
        current_assets=frame({"A": [40.0, 45.0, 50.0]}),
        current_liabilities=frame({"A": [20.0, 22.0, 25.0]}),
        total_liabilities=frame({"A": [50.0, 52.0, 55.0]}),
        cash_and_short_term_investments=frame({"A": [10.0, 11.0, 12.0]}),
        short_term_investments=ones * 2,
        long_term_investments=ones * 5,
        short_term_debt=ones * 3,
        long_term_debt=ones * 20,
        preferred_stock=ones * 0,
        accounts_receivable=frame({"A": [10.0, 12.0, 15.0]}),
        inventory=frame({"A": [5.0, 6.0, 6.0]}),
        property_plant_and_equipment=frame({"A": [30.0, 32.0, 35.0]}),
        revenue=frame({"A": [200.0, 220.0, 250.0]}),
        net_income=frame({"A": [10.0, 12.0, 15.0]}),
        debt_issued=frame({"A": [0.0, 5.0, 0.0]}),
        stock_issued=ones * 0,
    )
    inputs = {
        name: result[name].loc["A", "2023"]
        for name in quality_model.DECHOW_COEFFICIENTS
    }
    predicted = quality_model.DECHOW_INTERCEPT + sum(
        coefficient * inputs[name]
        for name, coefficient in quality_model.DECHOW_COEFFICIENTS.items()
    )
    assert result["Predicted Value"].loc["A", "2023"] == pytest.approx(predicted)
    assert result["F-Score"].loc["A", "2023"] == pytest.approx(
        1 / (1 + np.exp(-predicted)) / 0.0037
    )
    assert result["Actual Issuance"].loc["A", "2022"] == 1
    # Soft assets leave out cash and short-term investments (Compustat CHE).
    assert result["Soft Assets"].loc["A", "2021"] == pytest.approx(
        (100 - 30 - 10) / 100
    )


def test_montier_c_score_counts_warning_signs():
    result = quality_model.get_montier_c_score(
        net_income=frame({"A": [10.0, 12.0, 14.0]}),
        operating_cash_flow=frame({"A": [10.0, 10.0, 10.0]}),
        accounts_receivable=frame({"A": [10.0, 15.0, 20.0]}),
        revenue=frame({"A": [100.0, 100.0, 100.0]}),
        inventory=frame({"A": [10.0, 12.0, 14.0]}),
        cost_of_goods_sold=frame({"A": [50.0, 50.0, 50.0]}),
        other_current_assets=frame({"A": [5.0, 6.0, 7.0]}),
        depreciation_and_amortization=frame({"A": [10.0, 9.0, 8.0]}),
        property_plant_and_equipment=frame({"A": [100.0, 100.0, 100.0]}),
        total_assets=frame({"A": [100.0, 120.0, 140.0]}),
    )
    assert result["C-Score"].loc["A", "2023"] == 6
    assert np.isnan(result["C-Score"].loc["A", "2021"])


def test_mohanram_and_accruals():
    g_score = quality_model.get_mohanram_g_score(
        net_income=frame({"A": [10, 12, 14], "B": [5, 5, 5], "C": [1, 1, 1]}),
        operating_cash_flow=frame({"A": [12, 14, 16], "B": [4, 4, 4], "C": [1, 1, 1]}),
        total_assets=frame(
            {"A": [100, 100, 100], "B": [100, 100, 100], "C": [100, 100, 100]}
        ),
        revenue=frame({"A": [50, 55, 60], "B": [50, 50, 50], "C": [50, 40, 60]}),
        research_and_development=frame(
            {"A": [5, 5, 5], "B": [1, 1, 1], "C": [0, 0, 0]}
        ),
        capital_expenditure=frame(
            {"A": [-5, -5, -5], "B": [-1, -1, -1], "C": [0, 0, 0]}
        ),
        selling_and_marketing=frame({"A": [5, 5, 5], "B": [1, 1, 1], "C": [0, 0, 0]}),
        variance_periods=2,
    )
    # A's return on assets and revenue growth vary more than the median company's, so it
    # misses G4 and G5.
    assert g_score["G-Score"].loc["A", "2023"] == 8 - 2
    assert g_score["G4 Earnings Variability"].loc["A", "2023"] == 0
    assert g_score["G3 Cash Flow above Net Income"].loc["B", "2023"] == 0

    accruals = quality_model.get_accruals_ratios(
        current_assets=frame({"A": [50.0, 60.0, 60.0]}),
        cash_and_cash_equivalents=frame({"A": [10.0, 10.0, 10.0]}),
        current_liabilities=frame({"A": [30.0, 30.0, 30.0]}),
        short_term_debt=frame({"A": [5.0, 5.0, 5.0]}),
        tax_payables=frame({"A": [1.0, 1.0, 1.0]}),
        depreciation_and_amortization=frame({"A": [4.0, 4.0, 4.0]}),
        net_income=frame({"A": [10.0, 10.0, 10.0]}),
        operating_cash_flow=frame({"A": [8.0, 8.0, 8.0]}),
        investing_cash_flow=frame({"A": [-5.0, -5.0, -5.0]}),
        total_assets=frame({"A": [100.0, 100.0, 100.0]}),
    )
    assert accruals["Balance Sheet Accruals Ratio"].loc["A", "2022"] == pytest.approx(
        0.06
    )
    assert accruals["Cash Flow Accruals Ratio"].loc["A", "2022"] == pytest.approx(0.07)


def test_mohanram_scales_by_beginning_assets_and_scores_missing_signals_zero():
    g_score = quality_model.get_mohanram_g_score(
        # A's assets doubled during the year: against the assets at the start its return
        # (10%) beats B's (9%), against the average assets (6.7%) it would not.
        net_income=frame({"A": [10, 10, 10], "B": [9, 9, 9], "C": [1, 1, 1]}),
        operating_cash_flow=frame({"A": [5, 5, 5], "B": [5, 5, 5], "C": [5, 5, 5]}),
        total_assets=frame(
            {"A": [100, 100, 200], "B": [100, 100, 100], "C": [100, 100, 100]}
        ),
        revenue=frame({"A": [50, 55, 60], "B": [50, 50, 50], "C": [50, 40, 60]}),
        research_and_development=frame(
            {"A": [5, 5, 5], "B": [1, 1, 1], "C": [np.nan, np.nan, np.nan]}
        ),
        capital_expenditure=frame(
            {"A": [-5, -5, -5], "B": [-1, -1, -1], "C": [0, 0, 0]}
        ),
        selling_and_marketing=frame({"A": [5, 5, 5], "B": [1, 1, 1], "C": [0, 0, 0]}),
    )
    assert g_score["G1 Return on Assets"].loc["A", "2023"] == 1
    # C reports no research and development and has too short a history for the
    # variances (three growth rates are needed): those signals score 0, the G-Score stays.
    assert g_score["G6 Research and Development Intensity"].loc["C", "2023"] == 0
    assert g_score["G5 Growth Variability"].loc["C", "2023"] == 0
    assert g_score["G-Score"].loc[["A", "B", "C"], "2023"].notna().all()

    with pytest.raises(ValueError):
        quality_model.get_mohanram_g_score(
            *[frame({"A": [1.0, 1.0, 1.0]})] * 7, variance_periods=1
        )


def test_variance_takes_one_value_a_year_with_quarterly_lag():
    # A seasonal pattern that repeats every year: one value a year does not vary, while
    # the consecutive quarters do.
    quarters = [f"Q{quarter}" for quarter in range(12)]
    seasonal = pd.DataFrame([[1.0, 2.0, 3.0, 4.0] * 3], index=["A"], columns=quarters)

    yearly = quality_model._variance(seasonal, periods=3, lag=4, min_periods=3)
    quarterly = quality_model._variance(seasonal, periods=3, lag=1, min_periods=3)

    assert yearly.loc["A", "Q11"] == pytest.approx(0)
    assert np.isnan(yearly.loc["A", "Q7"])
    assert quarterly.loc["A", "Q11"] > 0
    # With a lag of 1 it is a rolling variance.
    pd.testing.assert_frame_equal(
        quarterly, seasonal.T.rolling(3, min_periods=3).var().T
    )


def test_quarterly_lag_compares_with_the_year_before():
    quarters = [f"Q{quarter}" for quarter in range(8)]
    assets = pd.DataFrame(
        [[100 * 1.03**quarter for quarter in range(8)]], index=["A"], columns=quarters
    )
    flat = pd.DataFrame(1.0, index=["A"], columns=quarters)

    signals = {
        lag: quality_model.get_montier_c_score(
            net_income=flat,
            operating_cash_flow=flat,
            accounts_receivable=flat,
            revenue=flat,
            inventory=flat,
            cost_of_goods_sold=flat,
            other_current_assets=flat,
            depreciation_and_amortization=flat,
            property_plant_and_equipment=flat,
            total_assets=assets,
            lag=lag,
        )["Total Asset Growth"]
        for lag in (1, 4)
    }
    # 3% a quarter is 12.55% a year, above Montier's 10%.
    assert signals[4].loc["A", "Q7"] == 1
    assert signals[1].loc["A", "Q7"] == 0
    assert np.isnan(signals[4].loc["A", "Q3"])
