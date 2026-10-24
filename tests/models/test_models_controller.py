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
