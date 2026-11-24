"""Quality Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models.quality import quality_model
from financetoolkit.utilities.error_model import InvalidArgumentError

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


def test_rank_z_score_standardizes_the_ranks_per_period():
    frame = pd.DataFrame(
        {"2024": [0.30, 0.10, 0.20, None], "2025": [1.0, 1.0, 5.0, 2.0]},
        index=["A", "B", "C", "D"],
    )

    z_score = quality_model.get_rank_z_score(frame)

    # Ranks 3, 1, 2 have a mean of 2 and a standard deviation of 1.
    assert z_score["2024"].tolist()[:3] == pytest.approx([1.0, -1.0, 0.0])
    assert pd.isna(z_score.loc["D", "2024"])
    # Ties share the average rank, and outliers count as one rank.
    assert z_score.loc["A", "2025"] == z_score.loc["B", "2025"]
    assert z_score["2025"].mean() == pytest.approx(0.0)
    assert z_score["2025"].std() == pytest.approx(1.0)


def test_quality_minus_junk_orders_the_tickers_by_every_component():
    better = pd.DataFrame({"2025": [3.0, 2.0, 1.0]}, index=["A", "B", "C"])
    components = {
        "Profitability": {"Return on Equity": better, "Gross Margin": better * 2},
        "Safety": {"Low Leverage": better},
    }

    scores = quality_model.get_quality_minus_junk(components)

    assert list(scores) == ["Profitability", "Safety", "Quality Score"]
    assert scores["Quality Score"]["2025"].tolist() == pytest.approx([1.0, 0.0, -1.0])


def test_a_ticker_without_a_component_has_no_quality_score():
    values = pd.DataFrame({"2025": [3.0, 2.0, 1.0]}, index=["A", "B", "C"])
    missing = values.copy()
    missing.loc["C", "2025"] = float("nan")

    scores = quality_model.get_quality_minus_junk(
        {"Profitability": {"ROE": values}, "Payout": {"Net Payout Ratio": missing}}
    )

    assert pd.isna(scores["Quality Score"].loc["C", "2025"])
    assert scores["Quality Score"].loc[["A", "B"], "2025"].notna().all()


def jones_company(seed: int, years: int = 15) -> dict[str, pd.DataFrame]:
    """Random but plausible statements of one company, ticker by year."""
    rng = np.random.default_rng(seed)
    columns = pd.period_range("2005", periods=years, freq="Y")

    def row(values):
        return pd.DataFrame([values], index=[f"T{seed}"], columns=columns)

    assets = 1000 * np.cumprod(1 + rng.normal(0.05, 0.1, years))
    return {
        "total_assets": row(assets),
        "revenue": row(assets * rng.uniform(0.5, 1.5, years)),
        "accounts_receivable": row(assets * rng.uniform(0.05, 0.2, years)),
        "property_plant_and_equipment": row(assets * rng.uniform(0.2, 0.6, years)),
    }


def jones_accruals(company, coefficients, revenue_change):
    """The total accruals that the Jones model explains exactly, in currency."""
    a1, a2, a3 = coefficients
    return a1 + a2 * revenue_change + a3 * company["property_plant_and_equipment"]


def test_modified_jones_time_series_matches_statsmodels_on_earlier_years():
    sm = pytest.importorskip("statsmodels.api")
    company = jones_company(3)
    rng = np.random.default_rng(4)
    net_income = company["total_assets"] * rng.normal(0.08, 0.03, 15)
    operating_cash_flow = company["total_assets"] * rng.normal(0.1, 0.03, 15)

    result = quality_model.get_modified_jones_accruals(
        net_income=net_income, operating_cash_flow=operating_cash_flow, **company
    )

    # 2016 (position 11) is estimated on 2006-2015, the ten years with beginning assets.
    beginning = company["total_assets"].shift(1, axis=1).iloc[0]
    revenue = company["revenue"].iloc[0]
    receivables = company["accounts_receivable"].iloc[0]
    ppe = company["property_plant_and_equipment"].iloc[0]
    accruals = (net_income - operating_cash_flow).iloc[0] / beginning
    design = pd.DataFrame(
        {
            "1/A": 1 / beginning,
            "dREV": revenue.diff() / beginning,
            "PPE": ppe / beginning,
        }
    )
    fit = sm.OLS(accruals.iloc[1:11], design.iloc[1:11]).fit()
    event = design.iloc[11].copy()
    event["dREV"] = (revenue.diff() - receivables.diff()).iloc[11] / beginning.iloc[11]
    expected = float(event @ fit.params)

    non_discretionary = result["Non-Discretionary Accruals"].iloc[0]
    assert non_discretionary.iloc[:11].isna().all()
    assert non_discretionary.iloc[11] == pytest.approx(expected, abs=1e-10)
    assert result["Discretionary Accruals"].iloc[0, 11] == pytest.approx(
        accruals.iloc[11] - expected, abs=1e-10
    )
    assert result["Total Accruals"].iloc[0, 11] == pytest.approx(accruals.iloc[11])


def test_modified_jones_accruals_that_follow_the_model_are_not_discretionary():
    company = jones_company(5)
    # Receivables that do not change leave the modification without effect.
    company["accounts_receivable"] = company["accounts_receivable"] * 0 + 100
    coefficients = (20.0, 0.05, -0.08)
    accruals = jones_accruals(company, coefficients, company["revenue"].diff(axis=1))

    result = quality_model.get_modified_jones_accruals(
        net_income=accruals,
        operating_cash_flow=accruals * 0,
        **company,
    )
    discretionary = result["Discretionary Accruals"].iloc[0]
    assert discretionary.notna().sum() == 15 - 11
    np.testing.assert_allclose(discretionary.dropna(), 0, atol=1e-10)

    # Revenue booked on credit is counted as discretionary: a2 times the change in
    # receivables in the year itself, while the estimation keeps the change in revenues.
    company["accounts_receivable"].iloc[0, -1] = 160.0
    result = quality_model.get_modified_jones_accruals(
        net_income=accruals,
        operating_cash_flow=accruals * 0,
        **company,
    )
    beginning = company["total_assets"].iloc[0, -2]
    assert result["Discretionary Accruals"].iloc[0, -1] == pytest.approx(
        0.05 * 60 / beginning
    )


def test_modified_jones_cross_section_explains_accruals_that_follow_the_model():
    companies = [jones_company(seed, years=4) for seed in range(12)]
    statements = {
        name: pd.concat([company[name] for company in companies])
        for name in companies[0]
    }
    coefficients = (15.0, 0.1, -0.05)
    accruals = jones_accruals(
        statements,
        coefficients,
        statements["revenue"].diff(axis=1)
        - statements["accounts_receivable"].diff(axis=1),
    )

    result = quality_model.get_modified_jones_accruals(
        net_income=accruals,
        operating_cash_flow=accruals * 0,
        method="cross_section",
        **statements,
    )
    discretionary = result["Discretionary Accruals"]
    assert discretionary.iloc[:, 0].isna().all()
    assert discretionary.iloc[:, 1:].notna().all().all()
    np.testing.assert_allclose(discretionary.iloc[:, 1:], 0, atol=1e-10)

    # Twelve tickers are fewer than thirteen observations.
    fewer = quality_model.get_modified_jones_accruals(
        net_income=accruals,
        operating_cash_flow=accruals * 0,
        method="cross_section",
        min_observations=13,
        **statements,
    )
    assert fewer["Discretionary Accruals"].isna().all().all()


def test_modified_jones_rejects_unknown_methods_and_too_few_observations():
    company = jones_company(6)
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        quality_model.get_modified_jones_accruals(
            net_income=company["revenue"],
            operating_cash_flow=company["revenue"],
            method="panel",
            **company,
        )
    with pytest.raises(ValueError, match="at least"):
        quality_model.get_modified_jones_accruals(
            net_income=company["revenue"],
            operating_cash_flow=company["revenue"],
            min_observations=2,
            **company,
        )


def test_modified_jones_accruals_do_not_depend_on_the_currency_unit():
    company = jones_company(8)
    rng = np.random.default_rng(9)
    net_income = company["total_assets"] * rng.normal(0.08, 0.03, 15)
    operating_cash_flow = company["total_assets"] * rng.normal(0.1, 0.03, 15)

    for method, min_observations in [("time_series", 10), ("cross_section", 3)]:
        statements = company
        income, cash_flow = net_income, operating_cash_flow
        if method == "cross_section":
            # Five companies, so the cross-section has enough tickers.
            companies = [jones_company(seed) for seed in range(8, 13)]
            statements = {
                name: pd.concat([each[name] for each in companies])
                for name in companies[0]
            }
            income = statements["total_assets"] * 0.08 + statements["revenue"] * 0.01
            cash_flow = statements["total_assets"] * 0.1
        base = quality_model.get_modified_jones_accruals(
            net_income=income,
            operating_cash_flow=cash_flow,
            method=method,
            min_observations=min_observations,
            **statements,
        )["Discretionary Accruals"]
        assert base.notna().any().any()
        # Total assets around 1e3 times 1e13 are around 1e16, as in rupiah or dong, where
        # one over the beginning assets is around 1e-16.
        for scale in (1e5, 1e13):
            scaled = quality_model.get_modified_jones_accruals(
                net_income=income * scale,
                operating_cash_flow=cash_flow * scale,
                method=method,
                min_observations=min_observations,
                **{name: frame * scale for name, frame in statements.items()},
            )["Discretionary Accruals"]
            pd.testing.assert_frame_equal(scaled, base, rtol=1e-9, atol=1e-12)


def test_modified_jones_with_an_intercept_matches_statsmodels_with_a_constant():
    sm = pytest.importorskip("statsmodels.api")
    company = jones_company(3)
    rng = np.random.default_rng(4)
    net_income = company["total_assets"] * rng.normal(0.08, 0.03, 15)
    operating_cash_flow = company["total_assets"] * rng.normal(0.1, 0.03, 15)

    result = quality_model.get_modified_jones_accruals(
        net_income=net_income,
        operating_cash_flow=operating_cash_flow,
        intercept=True,
        **company,
    )

    beginning = company["total_assets"].shift(1, axis=1).iloc[0]
    revenue = company["revenue"].iloc[0]
    receivables = company["accounts_receivable"].iloc[0]
    ppe = company["property_plant_and_equipment"].iloc[0]
    accruals = (net_income - operating_cash_flow).iloc[0] / beginning
    design = sm.add_constant(
        pd.DataFrame(
            {
                "1/A": 1 / beginning,
                "dREV": revenue.diff() / beginning,
                "PPE": ppe / beginning,
            }
        ).iloc[1:],
        prepend=False,
    )
    for position in (11, 14):
        fit = sm.OLS(accruals.iloc[1:position], design.iloc[: position - 1]).fit()
        event = design.iloc[position - 1].copy()
        event["dREV"] = (revenue.diff() - receivables.diff()).iloc[position] / (
            beginning.iloc[position]
        )
        expected = float(event @ fit.params)
        assert result["Non-Discretionary Accruals"].iloc[0, position] == pytest.approx(
            expected, abs=1e-10
        )

    # The cross-section with a constant, against statsmodels in one year.
    companies = [jones_company(seed, years=4) for seed in range(12)]
    statements = {
        name: pd.concat([each[name] for each in companies]) for name in companies[0]
    }
    income = statements["total_assets"] * rng.normal(0.08, 0.03, (12, 4))
    cash_flow = statements["total_assets"] * rng.normal(0.1, 0.03, (12, 4))
    cross_section = quality_model.get_modified_jones_accruals(
        net_income=income,
        operating_cash_flow=cash_flow,
        method="cross_section",
        intercept=True,
        **statements,
    )
    beginning = statements["total_assets"].iloc[:, 1]
    cash_revenue_change = (
        statements["revenue"].diff(axis=1)
        - statements["accounts_receivable"].diff(axis=1)
    ).iloc[:, 2] / statements["total_assets"].iloc[:, 1]
    design = pd.DataFrame(
        {
            "1/A": 1 / beginning,
            "dREV": cash_revenue_change,
            "PPE": statements["property_plant_and_equipment"].iloc[:, 2] / beginning,
            "const": 1.0,
        }
    )
    accruals = (income - cash_flow).iloc[:, 2] / beginning
    fit = sm.OLS(accruals, design).fit()
    np.testing.assert_allclose(
        cross_section["Non-Discretionary Accruals"].iloc[:, 2],
        design @ fit.params,
        rtol=0,
        atol=1e-10,
    )
    # Four coefficients need at least four observations.
    with pytest.raises(ValueError, match="4 coefficients"):
        quality_model.get_modified_jones_accruals(
            net_income=income,
            operating_cash_flow=cash_flow,
            min_observations=3,
            intercept=True,
            **statements,
        )


def long_frame(values) -> pd.DataFrame:
    return pd.DataFrame([list(values)], index=["A"], columns=range(len(values)))


def test_get_accrual_quality_is_the_residual_deviation_of_the_regression():
    rng = np.random.default_rng(3)
    cash_flow = rng.normal(0.1, 0.03, 12)
    # Accruals that the cash flows explain exactly have no estimation error.
    accruals = (
        0.01
        + 0.2 * np.roll(cash_flow, 1)
        - 0.6 * cash_flow
        + 0.3 * np.roll(cash_flow, -1)
    )
    accruals[0] = accruals[-1] = np.nan
    exact = quality_model.get_accrual_quality(
        long_frame(accruals), long_frame(cash_flow), window=8
    )
    assert exact.iloc[0, 9:].max() == pytest.approx(0, abs=1e-12)
    assert exact.iloc[0, :9].isna().all()

    # With noise, the value equals the standard deviation of the least squares residuals
    # over the eight years before the period (computed here with numpy's polyfit-free OLS).
    noisy = accruals + rng.normal(0, 0.01, 12)
    quality = quality_model.get_accrual_quality(
        long_frame(noisy), long_frame(cash_flow), window=8
    )
    years = np.arange(2, 10)
    design = np.column_stack(
        [np.ones(8), cash_flow[years - 1], cash_flow[years], cash_flow[years + 1]]
    )
    coefficients = np.linalg.lstsq(design, noisy[years], rcond=None)[0]
    expected = np.std(noisy[years] - design @ coefficients, ddof=1)
    assert quality.iloc[0, 10] == pytest.approx(expected)

    with pytest.raises(ValueError):
        quality_model.get_accrual_quality(
            long_frame(noisy), long_frame(cash_flow), window=4
        )


def test_get_earnings_attributes():
    rng = np.random.default_rng(5)
    earnings = 1 + np.cumsum(rng.normal(0.05, 0.2, 14))
    income = rng.normal(0.08, 0.02, 14)
    cash = rng.normal(0.1, 0.04, 14)
    result = quality_model.get_earnings_attributes(
        long_frame(earnings), long_frame(income), long_frame(cash), window=10
    )
    years = np.arange(4, 14)
    slope, intercept = np.polyfit(earnings[years - 1], earnings[years], 1)
    residuals = earnings[years] - (intercept + slope * earnings[years - 1])
    assert result["Persistence"].iloc[0, 13] == pytest.approx(slope)
    assert result["Predictability"].iloc[0, 13] == pytest.approx(
        np.sqrt((residuals**2).sum() / 8)
    )
    assert result["Smoothness"].iloc[0, 13] == pytest.approx(
        np.std(income[years], ddof=1) / np.std(cash[years], ddof=1)
    )
    # Ten years and the year before are needed for the persistence regression.
    assert result["Persistence"].iloc[0, :10].isna().all()


def test_get_graham_criteria():
    years = 25
    ones = np.ones(years)
    criteria = quality_model.get_graham_criteria(
        revenue=long_frame(1e9 * ones),
        total_current_assets=long_frame(500 * ones),
        total_current_liabilities=long_frame(200 * ones),
        long_term_debt=long_frame(250 * ones),
        net_income=long_frame(np.r_[-1.0, 10 * ones[1:]]),
        dividends_paid=long_frame(-5 * ones),
        earnings_per_share=long_frame(np.linspace(1, 3, years)),
        share_price=long_frame(30 * ones),
        book_value_per_share=long_frame(20 * ones),
    )
    last = {name: frame.iloc[0, -1] for name, frame in criteria.items()}
    assert last["Adequate Size"] == 1
    # Current ratio 2.5 and long-term debt 250 below the net current assets of 300.
    assert last["Strong Financial Condition"] == 1
    assert last["Earnings Stability"] == 1
    assert last["Dividend Record"] == 1
    assert last["Earnings Growth"] == 1
    # The price of 30 over the average EPS of the last three years (about 2.92) is above 10
    # but below 15, and that P/E times the P/B of 1.5 is about 15.4, below 22.5.
    assert last["Moderate Price to Earnings"] == 1
    assert last["Moderate Price to Assets"] == 1
    # The loss of the first year fails the stability test until it leaves the ten years,
    # and there are fewer than 20 years of dividends before year 19.
    assert criteria["Earnings Stability"].iloc[0, 9] == 0
    assert criteria["Earnings Stability"].iloc[0, 10] == 1
    assert pd.isna(criteria["Dividend Record"].iloc[0, 18])
    assert criteria["Dividend Record"].iloc[0, 19] == 1
