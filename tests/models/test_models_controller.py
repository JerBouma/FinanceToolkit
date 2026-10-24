"""Models Controller Tests""" ""
# pylint: disable=missing-function-docstring


def test_get_dupont_analysis(recorder, models_module):
    recorder.capture(models_module.get_dupont_analysis())
    recorder.capture(models_module.get_dupont_analysis(growth=True))
    recorder.capture(models_module.get_dupont_analysis(growth=True, lag=[1, 2, 3]))


def test_get_extended_dupont_analysis(recorder, models_module):
    recorder.capture(models_module.get_extended_dupont_analysis())
    recorder.capture(models_module.get_extended_dupont_analysis(growth=True))
    recorder.capture(
        models_module.get_extended_dupont_analysis(growth=True, lag=[1, 2, 3])
    )


def test_get_enterprise_value_breakdown(recorder, models_module):
    recorder.capture(models_module.get_enterprise_value_breakdown())
    recorder.capture(models_module.get_enterprise_value_breakdown(growth=True))
    recorder.capture(
        models_module.get_enterprise_value_breakdown(growth=True, lag=[1, 2, 3])
    )


def test_get_weighted_average_cost_of_capital(recorder, models_module):
    recorder.capture(models_module.get_weighted_average_cost_of_capital().round(1))
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True).round(1)
    )
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True, lag=[1, 2, 3])
    )


def test_get_intrinsic_valuation(recorder, models_module):
    recorder.capture(
        models_module.get_intrinsic_valuation(
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
        )
    )
    recorder.capture(
        models_module.get_intrinsic_valuation(
            growth_rate=[0.05, 0.08],
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
        )
    )


def test_get_gorden_growth_model(recorder, models_module):
    recorder.capture(
        models_module.get_gorden_growth_model(rate_of_return=0.1, growth_rate=0.09)
    )
    recorder.capture(
        models_module.get_gorden_growth_model(
            rate_of_return=0.2, growth_rate=0.1, project_periods=10
        )
    )


def test_get_altman_z_score(recorder, models_module):
    recorder.capture(models_module.get_altman_z_score())
    recorder.capture(models_module.get_altman_z_score(growth=True))
    recorder.capture(models_module.get_altman_z_score(growth=True, lag=[1, 2, 3]))


def test_get_piotroski_score(recorder, models_module):
    recorder.capture(models_module.get_piotroski_score())


def test_get_present_value_of_growth_opportunities(recorder, models_module):
    recorder.capture(models_module.get_present_value_of_growth_opportunities())
    recorder.capture(
        models_module.get_present_value_of_growth_opportunities(calculate_daily=True)
    )


def test_get_economic_value_added(recorder, models_module):
    recorder.capture(models_module.get_economic_value_added())
    recorder.capture(models_module.get_economic_value_added(growth=True))


def test_get_beneish_m_score(recorder, models_module):
    recorder.capture(models_module.get_beneish_m_score())
    recorder.capture(models_module.get_beneish_m_score(growth=True))


def test_get_ohlson_o_score(recorder, models_module):
    recorder.capture(models_module.get_ohlson_o_score())
    recorder.capture(models_module.get_ohlson_o_score(growth=True))
    recorder.capture(models_module.get_ohlson_o_score(growth=True, lag=[1, 2, 3]))


def test_get_zmijewski_score(recorder, models_module):
    recorder.capture(models_module.get_zmijewski_score())
    recorder.capture(models_module.get_zmijewski_score(growth=True))
    recorder.capture(models_module.get_zmijewski_score(growth=True, lag=[1, 2, 3]))


def test_get_sustainable_growth_rate(recorder, models_module):
    recorder.capture(models_module.get_sustainable_growth_rate())
    recorder.capture(models_module.get_sustainable_growth_rate(growth=True))


def test_get_internal_growth_rate(recorder, models_module):
    recorder.capture(models_module.get_internal_growth_rate())
    recorder.capture(models_module.get_internal_growth_rate(growth=True))


def test_get_market_value_added(recorder, models_module):
    recorder.capture(models_module.get_market_value_added())
    recorder.capture(models_module.get_market_value_added(growth=True))


def test_get_tobins_q_ratio(recorder, models_module):
    recorder.capture(models_module.get_tobins_q_ratio())
    recorder.capture(models_module.get_tobins_q_ratio(growth=True))


def test_get_free_cash_flow_to_firm(recorder, models_module):
    recorder.capture(models_module.get_free_cash_flow_to_firm())
    recorder.capture(models_module.get_free_cash_flow_to_firm(growth=True))


def test_get_free_cash_flow_to_equity(recorder, models_module):
    recorder.capture(models_module.get_free_cash_flow_to_equity())
    recorder.capture(models_module.get_free_cash_flow_to_equity(growth=True))


def test_get_two_stage_dividend_discount_model(recorder, models_module):
    recorder.capture(
        models_module.get_two_stage_dividend_discount_model(
            rate_of_return=0.10, high_growth_rate=0.12, stable_growth_rate=0.03
        )
    )
    recorder.capture(
        models_module.get_two_stage_dividend_discount_model(
            rate_of_return=[0.10, 0.09],
            high_growth_rate=0.12,
            stable_growth_rate=0.03,
            high_growth_periods=3,
        )
    )


def test_get_residual_income(recorder, models_module):
    recorder.capture(models_module.get_residual_income())
    recorder.capture(models_module.get_residual_income(growth=True))


def test_get_springate_score(recorder, models_module):
    recorder.capture(models_module.get_springate_score())
    recorder.capture(models_module.get_springate_score(growth=True))


def test_get_grover_score(recorder, models_module):
    recorder.capture(models_module.get_grover_score())
    recorder.capture(models_module.get_grover_score(growth=True))


def test_get_fulmer_h_score(recorder, models_module):
    recorder.capture(models_module.get_fulmer_h_score())
    recorder.capture(models_module.get_fulmer_h_score(growth=True))


def test_get_reverse_discounted_cash_flow(recorder, models_module):
    recorder.capture(
        models_module.get_reverse_discounted_cash_flow(
            weighted_average_cost_of_capital=0.09
        )
    )
    recorder.capture(
        models_module.get_reverse_discounted_cash_flow(
            perpetual_growth_rate={"AAPL": 0.02, "MSFT": 0.03},
            weighted_average_cost_of_capital=0.1,
            cash_flow_type="Operating Cash Flow",
        )
    )


def test_get_earnings_power_value(recorder, models_module):
    recorder.capture(
        models_module.get_earnings_power_value(weighted_average_cost_of_capital=0.08)
    )
    recorder.capture(
        models_module.get_earnings_power_value(
            weighted_average_cost_of_capital=0.08, periods=2
        )
    )


def test_get_monte_carlo_intrinsic_valuation(recorder, models_module):
    recorder.capture(
        models_module.get_monte_carlo_intrinsic_valuation(
            growth_rate=0.08, weighted_average_cost_of_capital=0.09, simulations=2000
        )
    )


def test_get_merton_distance_to_default(recorder, models_module):
    recorder.capture(models_module.get_merton_distance_to_default())
    recorder.capture(models_module.get_merton_distance_to_default(time_horizon=2))


def test_get_synthetic_credit_rating(recorder, models_module, monkeypatch):
    import pandas as pd  # noqa: PLC0415

    from financetoolkit.models import damodaran_model  # noqa: PLC0415

    bounds = [
        -100000,
        0.2,
        0.65,
        0.8,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        3,
        4.25,
        5.5,
        6.5,
        8.5,
    ]
    ratings = [
        "D",
        "C",
        "CC",
        "CCC",
        "B-",
        "B",
        "B+",
        "BB",
        "BB+",
        "BBB",
        "A-",
        "A",
        "A+",
        "AA",
        "AAA",
    ]
    spreads = [
        0.19,
        0.16,
        0.13,
        0.09,
        0.05,
        0.03,
        0.028,
        0.018,
        0.014,
        0.011,
        0.009,
        0.008,
        0.007,
        0.0055,
        0.004,
    ]
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "Firm Type": firm_type,
                    "Lower": [bound * scale for bound in bounds],
                    "Upper": [bound * scale for bound in bounds[1:]] + [100000],
                    "Rating": ratings,
                    "Default Spread": spreads,
                }
            )
            for firm_type, scale in (("Large", 1), ("Small", 1.5), ("Financial", 0.3))
        ]
    )
    monkeypatch.setattr(damodaran_model, "get_synthetic_rating_table", lambda: table)

    recorder.capture(models_module.get_synthetic_credit_rating())
    recorder.capture(models_module.get_synthetic_credit_rating(firm_type="small"))
    assert models_module.get_synthetic_credit_rating(firm_type="tiny").empty


def test_get_peer_comparison_without_peer_data(models_module):
    assert models_module.get_peer_comparison().empty


def test_get_magic_formula(recorder, models_module):
    recorder.capture(models_module.get_magic_formula())


def test_get_mohanram_g_score(recorder, models_module):
    recorder.capture(models_module.get_mohanram_g_score(variance_periods=2))


def test_get_dechow_f_score(recorder, models_module):
    recorder.capture(models_module.get_dechow_f_score())


def test_get_montier_c_score(recorder, models_module):
    recorder.capture(models_module.get_montier_c_score())


def test_get_accruals_ratio(recorder, models_module):
    recorder.capture(models_module.get_accruals_ratio())


GROWTH = 1.03
QUARTERS = 20


def synthetic_models(items, quarterly=True):
    """
    A Models instance on synthetic statements, without the Toolkit around it. Items map a
    statement row to its values per ticker; the same frame serves as every statement.
    """
    import pandas as pd  # noqa: PLC0415

    from financetoolkit.models.models_controller import Models  # noqa: PLC0415

    tickers = sorted({ticker for values in items.values() for ticker in values})
    length = len(next(iter(next(iter(items.values())).values())))
    columns = pd.period_range(
        "2019Q1" if quarterly else "2010",
        periods=length,
        freq="Q" if quarterly else "Y",
    )
    statement = (
        pd.DataFrame(
            {
                (ticker, row): values[ticker]
                for row, values in items.items()
                for ticker in values
            },
            index=columns,
            dtype=float,
        )
        .T.sort_index()
        .rename_axis([None, None])
    )
    models = object.__new__(Models)
    models._tickers = tickers
    models._balance_sheet_statement = statement
    models._income_statement = statement
    models._cash_flow_statement = statement
    models._quarterly = quarterly
    models._rounding = 10
    models._start_date = None
    models._end_date = None
    models._historical_data = {
        "daily": pd.DataFrame(
            {("Close", ticker): [100.0] for ticker in tickers},
            index=pd.period_range("2023-12-29", periods=1, freq="D"),
        )
    }
    return models


def growing(start, length=QUARTERS):
    """Values that grow 3% a period."""
    return [start * GROWTH**period for period in range(length)]


def growing_company():
    """A company whose every item grows 3% a quarter, so 12.55% a year."""
    rows = {
        "Total Assets": 1000,
        "Total Current Assets": 400,
        "Total Current Liabilities": 200,
        "Total Liabilities": 500,
        "Cash and Short Term Investments": 100,
        "Cash and Cash Equivalents": 80,
        "Short Term Investments": 20,
        "Long Term Investments": 50,
        "Short Term Debt": 30,
        "Long Term Debt": 150,
        "Total Debt": 180,
        "Accounts Receivable": 60,
        "Inventory": 40,
        "Property, Plant and Equipment": 300,
        "Other Current Assets": 10,
        "Tax Payables": 5,
        "Revenue": 250,
        "Cost of Goods Sold": 150,
        "Operating Income": 50,
        "Income Before Tax": 40,
        "Income Tax Expense": 8,
        "Net Income": 25,
        "Research and Development Expenses": 12,
        "Selling and Marketing Expenses": 15,
        "Depreciation and Amortization": 10,
        "Operating Cash Flow": 30,
        "Free Cash Flow": 20,
        "Capital Expenditure": -10,
        "Cash Flow from Investing": -20,
    }
    items = {row: {"A": growing(start)} for row, start in rows.items()}
    items["Weighted Average Shares Diluted"] = {"A": [10.0] * QUARTERS}
    return items


def test_quarterly_scores_compare_year_over_year():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    models = synthetic_models(growing_company())
    g = GROWTH ** np.arange(QUARTERS)
    yearly_growth = GROWTH**4 - 1
    average_assets = 1000 * (g[-1] + g[-5]) / 2

    montier = models.get_montier_c_score(trailing=4).loc["A"]
    quarter_over_quarter = models.get_montier_c_score(trailing=1).loc["A"]
    # 12.55% a year is above Montier's 10%, 3% a quarter is not.
    assert montier.loc["Total Asset Growth"].iloc[-1] == 1
    assert quarter_over_quarter.loc["Total Asset Growth"].iloc[-1] == 0
    # A full trailing sum and the one a year back are needed for the first signs.
    assert str(montier.columns[0]) == "2020Q4"

    dechow = models.get_dechow_f_score(trailing=4).loc["A"]
    assert dechow.loc["Change in Cash Sales"].iloc[-1] == pytest.approx(yearly_growth)
    assert dechow.loc["Change in Receivables"].iloc[-1] == pytest.approx(
        60 * (g[-1] - g[-5]) / average_assets
    )
    assert dechow.loc["Soft Assets"].iloc[-1] == pytest.approx(
        (1000 - 300 - 100) / 1000
    )

    accruals = models.get_accruals_ratio(trailing=4).loc["A"]
    assert accruals.loc["Cash Flow Accruals Ratio"].iloc[-1] == pytest.approx(
        (25 - 30 + 20) * g[-4:].sum() / average_assets
    )
    balance_sheet = (
        (400 - 80) * (g[-1] - g[-5])
        - (200 - 30 - 5) * (g[-1] - g[-5])
        - 10 * g[-4:].sum()
    )
    assert accruals.loc["Balance Sheet Accruals Ratio"].iloc[-1] == pytest.approx(
        balance_sheet / average_assets
    )

    # Three yearly growth rates of trailing sums take sixteen quarters.
    mohanram = models.get_mohanram_g_score(trailing=4).loc["A"]
    assert str(mohanram.columns[0]) == "2022Q4"
    assert mohanram.loc["G-Score"].notna().all()


def test_quarterly_scores_default_to_a_trailing_year():
    import pandas as pd  # noqa: PLC0415

    models = synthetic_models(growing_company())
    for method in (
        "get_dechow_f_score",
        "get_montier_c_score",
        "get_mohanram_g_score",
        "get_accruals_ratio",
    ):
        pd.testing.assert_frame_equal(
            getattr(models, method)(), getattr(models, method)(trailing=4)
        )


def test_default_wacc_uses_the_trailing_periods():
    import pandas as pd  # noqa: PLC0415

    models = synthetic_models(growing_company())
    requested = []

    def weighted_average_cost_of_capital(show_full_results=True, trailing=None):
        requested.append(trailing)
        return pd.DataFrame({"2023Q4": [0.09]}, index=["A"])

    models.get_weighted_average_cost_of_capital = weighted_average_cost_of_capital

    models.get_earnings_power_value()
    models.get_reverse_discounted_cash_flow()
    models.get_monte_carlo_intrinsic_valuation(growth_rate=0.05, simulations=10)
    models.get_earnings_power_value(trailing=2)

    # A year of interest expense on quarterly data, the same trailing as the cash flows.
    assert requested == [4, 4, 4, 2]


def test_quality_scores_use_gross_debt_issuance_and_fixed_assets():
    items = {
        "Total Assets": {"A": [1000.0] * 3, "B": [1000.0] * 3},
        "Property, Plant and Equipment": {
            "A": [500.0, 500.0, 400.0],
            "B": [500.0, 500.0, 400.0],
        },
        "Gross Property, Plant and Equipment": {"A": [1000.0] * 3},
        "Depreciation and Amortization": {
            "A": [100.0, 100.0, 90.0],
            "B": [100.0, 100.0, 90.0],
        },
        "Gross Long Term Debt Issued": {"A": [0.0, 0.0, 50.0]},
        "Long Term Debt Issued": {"A": [0.0, 0.0, -10.0], "B": [0.0, 0.0, -10.0]},
        "Common Stock Issued": {"A": [0.0] * 3, "B": [0.0] * 3},
        "Revenue": {"A": [100.0] * 3, "B": [100.0] * 3},
    }
    models = synthetic_models(items, quarterly=False)

    # Both issued 50 of debt and repaid 60; only the gross issuance, which A reports,
    # shows the issuance.
    dechow = models.get_dechow_f_score()
    assert dechow.loc[("A", "Actual Issuance")].iloc[-1] == 1
    assert dechow.loc[("B", "Actual Issuance")].iloc[-1] == 0

    # Depreciation fell relative to gross fixed assets (10% to 9%) while it rose relative
    # to the net amount (20% to 22.5%), which only the gross amount reveals.
    montier = models.get_montier_c_score()
    assert montier.loc[("A", "Depreciation to Fixed Assets")].iloc[-1] == 1
    assert montier.loc[("B", "Depreciation to Fixed Assets")].iloc[-1] == 0


def test_quarterly_valuations_use_a_year_of_cash_flow_and_revenue():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    from financetoolkit.models import intrinsic_model  # noqa: PLC0415

    models = synthetic_models(growing_company())
    g = GROWTH ** np.arange(QUARTERS)
    last_year = g[-4:].sum()

    epv = models.get_earnings_power_value(weighted_average_cost_of_capital=0.1)
    assert epv.loc["A", "Normalized Operating Income"] == pytest.approx(
        0.2 * 250 * last_year
    )
    assert epv.loc["A", "Normalized NOPAT"] == pytest.approx(
        0.2 * 250 * last_year * 0.8
    )

    reverse = models.get_reverse_discounted_cash_flow(
        weighted_average_cost_of_capital=0.09
    )
    assert reverse.loc["A", "Cash Flow"] == pytest.approx(20 * last_year)
    # Trailing sums that grow 3% a quarter grow 12.55% a year.
    assert reverse.loc["A", "Historical Growth Rate"] == pytest.approx(GROWTH**4 - 1)

    simulated = models.get_monte_carlo_intrinsic_valuation(
        growth_rate=0.05,
        growth_rate_volatility=0,
        perpetual_growth_rate=0.02,
        perpetual_growth_rate_volatility=0,
        weighted_average_cost_of_capital=0.09,
        weighted_average_cost_of_capital_volatility=0,
        simulations=10,
    )
    expected = float(
        intrinsic_model.get_intrinsic_value(
            20 * last_year, 0.05, 0.02, 0.09, 80 * g[-1], 180 * g[-1], 10, 5
        )
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    assert simulated.loc["A", "Mean"] == pytest.approx(expected)


def test_market_values_use_the_traded_close(models_module):
    import pytest  # noqa: PLC0415

    prices = models_module._period_prices()
    close = models_module._historical_data["yearly"]["Close"]
    period = prices.columns[prices.loc["AAPL"].notna()][-1]
    assert prices.loc["AAPL", period] == pytest.approx(close.loc[period, "AAPL"])
    assert models_module._latest_price("AAPL") == pytest.approx(
        models_module._historical_data["daily"]["Close"]["AAPL"].dropna().iloc[-1]
    )


def test_synthetic_credit_rating_without_interest_expense(models_module, monkeypatch):
    import copy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from financetoolkit.models import damodaran_model  # noqa: PLC0415

    table = pd.DataFrame(
        {
            "Firm Type": "Large",
            "Lower": [-10.0, 1.0, 5.0],
            "Upper": [0.999, 4.999, 100000.0],
            "Rating": ["D", "BBB", "AAA"],
            "Default Spread": [0.19, 0.011, 0.004],
        }
    )
    monkeypatch.setattr(damodaran_model, "get_synthetic_rating_table", lambda: table)

    models = copy.copy(models_module)
    income = models_module._income_statement.copy()
    period = income.columns[-1]
    # A reported zero means nothing to cover, a missing interest expense is unknown.
    income.loc[("AAPL", "Interest Expense"), period] = 0.0
    income.loc[("MSFT", "Interest Expense"), period] = np.nan
    models._income_statement = income

    rating = models.get_synthetic_credit_rating(firm_type="large")
    assert rating.loc[("AAPL", "Rating"), period] == "AAA"
    assert pd.isna(rating.loc[("MSFT", "Rating"), period])
