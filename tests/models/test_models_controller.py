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


def test_capital_asset_pricing_inputs_cover_the_trailing_span():
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    from financetoolkit.models import intrinsic_model, wacc_model
    from financetoolkit.models.models_controller import Models

    quarters = pd.period_range("2023Q1", "2024Q4", freq="Q")
    risk_free_rate = pd.Series(0.04, index=quarters)
    benchmark_returns = pd.Series(0.0225, index=quarters)

    # Quarterly data with a trailing 4 quarters: the flows are trailing twelve months, so
    # the rates are annual (4% and 4% + 5%), not the quarterly 1% and 2.25%.
    risk_free, market_return = Models._capital_asset_pricing_inputs(
        SimpleNamespace(_quarterly=True), risk_free_rate, benchmark_returns, 0.05, 4
    )
    assert risk_free.iloc[-1] == pytest.approx(0.04)
    assert market_return.iloc[-1] == pytest.approx(0.09)

    # Net Income of 100 over the year on a Book Value of 1000, with a Beta of 1.2, earns
    # exactly the annual Cost of Equity of 4% + 1.2 * 5% = 10%.
    beta = pd.DataFrame({"AAPL": 1.2}, index=quarters)
    cost_of_equity = wacc_model.get_cost_of_equity(risk_free, beta, market_return)
    assert cost_of_equity["AAPL"].iloc[-1] == pytest.approx(0.10)
    residual_income = intrinsic_model.get_residual_income(
        net_income=100,
        cost_of_equity=cost_of_equity["AAPL"].iloc[-1],
        book_value_of_equity=1000,
    )
    assert residual_income == pytest.approx(0)

    # Without trailing the rates stay per quarter, and the default premium (the average
    # benchmark return above the per-quarter rate, 1.25%) scales the same way.
    risk_free, market_return = Models._capital_asset_pricing_inputs(
        SimpleNamespace(_quarterly=True), risk_free_rate, benchmark_returns, None
    )
    assert risk_free.iloc[-1] == pytest.approx(0.01)
    assert market_return.iloc[-1] == pytest.approx(0.0225)
    risk_free, market_return = Models._capital_asset_pricing_inputs(
        SimpleNamespace(_quarterly=True), risk_free_rate, benchmark_returns, None, 4
    )
    assert market_return.iloc[-1] == pytest.approx(0.09)


def test_cost_of_equity_follows_trailing(models_module):
    import pandas as pd

    # A trailing sum of two years of Net Income or Interest Expense is charged the Cost of
    # Equity of two years, in residual income and in the WACC (and so EVA and PVGO).
    single = models_module.get_residual_income(rounding=10).xs(
        "Cost of Equity", level=1
    )
    double = models_module.get_residual_income(rounding=10, trailing=2).xs(
        "Cost of Equity", level=1
    )
    pd.testing.assert_frame_equal(double, 2 * single, rtol=1e-6)

    wacc_single = models_module.get_weighted_average_cost_of_capital(rounding=10).xs(
        "Cost of Equity", level=1
    )
    wacc_double = models_module.get_weighted_average_cost_of_capital(
        rounding=10, trailing=2
    ).xs("Cost of Equity", level=1)
    pd.testing.assert_frame_equal(wacc_double, 2 * wacc_single, rtol=1e-6)


def test_intrinsic_valuation_discounts_the_free_cash_flow_to_firm(models_module):
    import pandas as pd

    from financetoolkit.models import intrinsic_model

    valuation = models_module.get_intrinsic_valuation(
        growth_rate=0.05,
        perpetual_growth_rate=0.02,
        weighted_average_cost_of_capital=0.10,
        rounding=10,
    )
    fcff = models_module.get_free_cash_flow_to_firm(rounding=15).xs(
        "Free Cash Flow to Firm", level=1
    )

    for ticker in models_module._tickers:
        expected = intrinsic_model.get_intrinsic_value(
            cash_flow=fcff.loc[ticker].dropna().iloc[-1],
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
            cash_and_cash_equivalents=models_module._balance_sheet_statement.loc[
                ticker, "Cash and Cash Equivalents"
            ]
            .dropna()
            .iloc[-1],
            total_debt=models_module._balance_sheet_statement.loc[ticker, "Total Debt"]
            .dropna()
            .iloc[-1],
            shares_outstanding=models_module._income_statement.loc[
                ticker, "Weighted Average Shares Diluted"
            ]
            .dropna()
            .iloc[-1],
        ).round(10)
        pd.testing.assert_frame_equal(valuation.loc[ticker], expected)

    # The capital expenditure and working capital are not cash flows to value a company on.
    assert models_module.get_intrinsic_valuation(
        0.05, 0.02, 0.10, cash_flow_type="Capital Expenditure"
    ).empty
    # Without a discount rate above the perpetual growth rate there is no valuation.
    assert models_module.get_intrinsic_valuation(0.05, 0.10, 0.10).empty


def test_quarterly_valuations_use_the_trailing_twelve_months(
    models_module, monkeypatch
):
    import pandas as pd

    monkeypatch.setattr(models_module, "_quarterly", True)

    pd.testing.assert_frame_equal(
        models_module.get_intrinsic_valuation(0.05, 0.02, 0.10),
        models_module.get_intrinsic_valuation(0.05, 0.02, 0.10, trailing=4),
    )
    pd.testing.assert_frame_equal(
        models_module.get_graham_number(),
        models_module.get_graham_number(trailing=4),
    )


def test_present_value_of_growth_opportunities_uses_the_cost_of_equity(models_module):
    import pandas as pd

    pvgo = models_module.get_present_value_of_growth_opportunities(rounding=10)
    cost_of_equity = models_module.get_weighted_average_cost_of_capital(rounding=15).xs(
        "Cost of Equity", level=1
    )
    income = models_module._income_statement
    earnings_per_share = (
        income.loc[:, "Net Income", :]
        / income.loc[:, "Weighted Average Shares Diluted", :]
    )
    close = models_module._historical_data["yearly"]["Close"]

    expected = close[models_module._tickers] - (earnings_per_share / cost_of_equity).T
    expected = expected.loc[pvgo.index].round(10)
    pd.testing.assert_frame_equal(pvgo, expected, check_names=False, rtol=1e-8)


def test_residual_income_charges_the_beginning_book_value(models_module):
    import pandas as pd

    residual_income = models_module.get_residual_income(rounding=10)
    balance = models_module._balance_sheet_statement
    beginning_book_value = (
        balance.loc[:, "Total Shareholder Equity", :]
        - balance.loc[:, "Preferred Stock", :]
    ).shift(1, axis=1)

    book_value = residual_income.xs("Book Value of Equity", level=1)
    pd.testing.assert_frame_equal(
        book_value,
        beginning_book_value.loc[book_value.index, book_value.columns],
        check_names=False,
    )
    expected = (
        residual_income.xs("Net Income", level=1)
        - residual_income.xs("Cost of Equity", level=1) * book_value
    )
    pd.testing.assert_frame_equal(
        residual_income.xs("Residual Income", level=1), expected, rtol=1e-6
    )


def test_economic_value_added_uses_the_unrounded_wacc(models_module):
    import pandas as pd

    eva = models_module.get_economic_value_added(rounding=15)
    wacc = models_module.get_weighted_average_cost_of_capital(
        show_full_results=False, rounding=15
    )

    eva_wacc = eva.xs("Weighted Average Cost of Capital", level=1).dropna(
        axis=1, how="all"
    )
    pd.testing.assert_frame_equal(
        eva_wacc, wacc.loc[eva_wacc.index, eva_wacc.columns], check_names=False
    )


def test_models_take_operating_income_as_ebit(models_module):
    import pandas as pd

    altman = models_module.get_altman_z_score(rounding=10)
    income = models_module._income_statement
    balance = models_module._balance_sheet_statement
    expected = (
        income.loc[:, "Operating Income", :] / balance.loc[:, "Total Assets", :]
    ).round(10)

    ebit_to_assets = altman.xs("EBIT to Total Assets", level=1)
    pd.testing.assert_frame_equal(
        ebit_to_assets,
        expected.loc[ebit_to_assets.index, ebit_to_assets.columns],
        check_names=False,
    )


def test_growth_rates_use_beginning_equity_and_ending_assets(models_module):
    import pandas as pd

    income = models_module._income_statement
    balance = models_module._balance_sheet_statement
    cash = models_module._cash_flow_statement
    net_income = income.loc[:, "Net Income", :]
    retention = 1 - cash.loc[:, "Dividends Paid", :].abs() / net_income

    sustainable = models_module.get_sustainable_growth_rate(rounding=10)
    expected = (
        net_income / balance.loc[:, "Total Equity", :].shift(1, axis=1) * retention
    ).round(10)
    pd.testing.assert_frame_equal(
        sustainable,
        expected.loc[sustainable.index, sustainable.columns],
        check_names=False,
    )

    internal = models_module.get_internal_growth_rate(rounding=10)
    retained_return = net_income / balance.loc[:, "Total Assets", :] * retention
    expected = (retained_return / (1 - retained_return)).round(10)
    pd.testing.assert_frame_equal(
        internal, expected.loc[internal.index, internal.columns], check_names=False
    )


def test_dividend_discount_models_drop_an_unfinished_period():
    from types import SimpleNamespace

    import pandas as pd

    from financetoolkit.models.models_controller import Models

    years = pd.period_range("2022", "2024", freq="Y")
    dividends = pd.DataFrame({"AAPL": [0.9, 0.95, 0.5]}, index=years)

    def complete(last_day: str) -> pd.DataFrame:
        days = pd.period_range("2022-01-03", last_day, freq="D")
        historical = {
            "yearly": pd.concat({"Dividends": dividends}, axis=1),
            "daily": pd.DataFrame(index=days),
        }
        return Models._complete_period_dividends(
            SimpleNamespace(_quarterly=False, _historical_data=historical)
        )

    # Data up to the last trading day of the year covers the whole year.
    assert complete("2024-12-31").index[-1] == years[-1]
    # Data ending in June leaves 2024 unfinished, so 2023 is the last complete year.
    assert complete("2024-06-28").index[-1] == years[-2]
