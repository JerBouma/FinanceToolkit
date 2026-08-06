"""Risk Controller Tests""" ""
# pylint: disable=missing-function-docstring


def test_collect_all_metrics(recorder, risk_module):
    recorder.capture(risk_module.collect_all_metrics())
    recorder.capture(risk_module.collect_all_metrics(growth=True))
    recorder.capture(risk_module.collect_all_metrics(growth=True, lag=[1, 2, 3]))


def test_get_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_value_at_risk())
    recorder.capture(risk_module.get_value_at_risk(within_period=False))
    recorder.capture(risk_module.get_value_at_risk(period="monthly"))
    recorder.capture(risk_module.get_value_at_risk(growth=True))
    recorder.capture(risk_module.get_value_at_risk(growth=True, lag=[1, 2, 3]))
    recorder.capture(
        risk_module.get_value_at_risk(
            period="monthly", within_period=False, distribution="evt"
        )
    )
    recorder.capture(
        risk_module.get_value_at_risk(
            period="monthly", within_period=False, distribution="cornish-fisher"
        )
    )
    recorder.capture(
        risk_module.get_value_at_risk(period="monthly", within_period=False, rolling=6)
    )


def test_get_conditional_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_conditional_value_at_risk())
    recorder.capture(risk_module.get_conditional_value_at_risk(within_period=False))
    recorder.capture(risk_module.get_conditional_value_at_risk(period="monthly"))
    recorder.capture(risk_module.get_conditional_value_at_risk(growth=True))
    recorder.capture(
        risk_module.get_conditional_value_at_risk(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, rolling=6
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, distribution="cornish-fisher"
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, distribution="evt"
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, distribution="studentt"
        )
    )


def test_get_conditional_value_at_risk_studentt_differs_from_var(recorder, risk_module):
    # Regression test: studentt used to dispatch to VaR, so CVaR must be as extreme.
    var = risk_module.get_value_at_risk(
        period="monthly", within_period=False, distribution="studentt"
    )
    cvar = risk_module.get_conditional_value_at_risk(
        period="monthly", within_period=False, distribution="studentt"
    )
    assert (cvar <= var).all().all()


def test_get_conditional_drawdown_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_conditional_drawdown_at_risk())
    recorder.capture(risk_module.get_conditional_drawdown_at_risk(within_period=False))
    recorder.capture(risk_module.get_conditional_drawdown_at_risk(period="monthly"))
    recorder.capture(risk_module.get_conditional_drawdown_at_risk(growth=True))
    recorder.capture(
        risk_module.get_conditional_drawdown_at_risk(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        risk_module.get_conditional_drawdown_at_risk(period="monthly", rolling=6)
    )


def test_get_tail_ratio(recorder, risk_module):
    recorder.capture(risk_module.get_tail_ratio())
    recorder.capture(risk_module.get_tail_ratio(within_period=False))
    recorder.capture(risk_module.get_tail_ratio(period="monthly"))
    recorder.capture(risk_module.get_tail_ratio(growth=True))
    recorder.capture(risk_module.get_tail_ratio(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_tail_ratio(period="monthly", rolling=6))


def test_get_entropic_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_entropic_value_at_risk())
    recorder.capture(risk_module.get_entropic_value_at_risk(within_period=False))
    recorder.capture(risk_module.get_entropic_value_at_risk(period="monthly"))
    recorder.capture(risk_module.get_entropic_value_at_risk(growth=True))
    recorder.capture(risk_module.get_entropic_value_at_risk(growth=True, lag=[1, 2, 3]))


def test_get_garch(recorder, risk_module):
    recorder.capture(risk_module.get_garch())
    recorder.capture(risk_module.get_garch(period="monthly"))
    recorder.capture(risk_module.get_garch(growth=True))
    recorder.capture(risk_module.get_garch(growth=True, lag=[1, 2, 3]))


def test_get_garch_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_garch_forecast())
    recorder.capture(risk_module.get_garch_forecast(period="monthly"))
    recorder.capture(risk_module.get_garch_forecast(growth=True))
    recorder.capture(risk_module.get_garch_forecast(growth=True, lag=[1, 2, 3]))


def test_get_garch_parameters(recorder, risk_module):
    recorder.capture(risk_module.get_garch_parameters())
    recorder.capture(risk_module.get_garch_parameters(period="monthly"))


def test_get_gjr_garch(recorder, risk_module):
    recorder.capture(risk_module.get_gjr_garch())
    recorder.capture(risk_module.get_gjr_garch(period="monthly"))
    recorder.capture(risk_module.get_gjr_garch(growth=True))


def test_get_gjr_garch_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_gjr_garch_forecast())
    recorder.capture(risk_module.get_gjr_garch_forecast(period="monthly"))


def test_get_gjr_garch_parameters(recorder, risk_module):
    recorder.capture(risk_module.get_gjr_garch_parameters())
    recorder.capture(risk_module.get_gjr_garch_parameters(period="monthly"))


def test_get_egarch(recorder, risk_module):
    recorder.capture(risk_module.get_egarch())
    recorder.capture(risk_module.get_egarch(period="monthly"))
    recorder.capture(risk_module.get_egarch(growth=True))


def test_get_egarch_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_egarch_forecast())
    recorder.capture(risk_module.get_egarch_forecast(period="monthly"))


def test_get_egarch_parameters(recorder, risk_module):
    recorder.capture(risk_module.get_egarch_parameters())
    recorder.capture(risk_module.get_egarch_parameters(period="monthly"))


def test_get_var_backtest(recorder, risk_module):
    recorder.capture(risk_module.get_var_backtest(window_size=100))
    recorder.capture(
        risk_module.get_var_backtest(window_size=100, distribution="gaussian")
    )
    recorder.capture(risk_module.get_var_backtest(window_size=100, test="kupiec"))
    recorder.capture(
        risk_module.get_var_backtest(window_size=100, test="christoffersen")
    )


def test_get_maximum_drawdown(recorder, risk_module):
    recorder.capture(risk_module.get_maximum_drawdown())
    recorder.capture(risk_module.get_maximum_drawdown(within_period=False))
    recorder.capture(risk_module.get_maximum_drawdown(period="monthly"))
    recorder.capture(risk_module.get_maximum_drawdown(growth=True))
    recorder.capture(risk_module.get_maximum_drawdown(growth=True, lag=[1, 2, 3]))


def test_get_maximum_drawdown_duration(recorder, risk_module):
    recorder.capture(risk_module.get_maximum_drawdown_duration())
    recorder.capture(risk_module.get_maximum_drawdown_duration(within_period=False))
    recorder.capture(risk_module.get_maximum_drawdown_duration(period="monthly"))
    recorder.capture(risk_module.get_maximum_drawdown_duration(growth=True))
    recorder.capture(
        risk_module.get_maximum_drawdown_duration(growth=True, lag=[1, 2, 3])
    )


def test_get_maximum_drawdown_recovery_time(recorder, risk_module):
    recorder.capture(risk_module.get_maximum_drawdown_recovery_time())
    recorder.capture(
        risk_module.get_maximum_drawdown_recovery_time(within_period=False)
    )
    recorder.capture(risk_module.get_maximum_drawdown_recovery_time(period="monthly"))
    recorder.capture(risk_module.get_maximum_drawdown_recovery_time(growth=True))
    recorder.capture(
        risk_module.get_maximum_drawdown_recovery_time(growth=True, lag=[1, 2, 3])
    )


def test_get_ulcer_index(recorder, risk_module):
    recorder.capture(risk_module.get_ulcer_index())
    recorder.capture(risk_module.get_ulcer_index(rolling=5))
    recorder.capture(risk_module.get_ulcer_index(period="monthly"))
    recorder.capture(risk_module.get_ulcer_index(growth=True))
    recorder.capture(risk_module.get_ulcer_index(growth=True, lag=[1, 2, 3]))


def test_get_skewness(recorder, risk_module):
    recorder.capture(risk_module.get_skewness())
    recorder.capture(risk_module.get_skewness(within_period=False))
    recorder.capture(risk_module.get_skewness(period="monthly"))
    recorder.capture(risk_module.get_skewness(growth=True))
    recorder.capture(risk_module.get_skewness(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_skewness(period="monthly", rolling=6))


def test_get_kurtosis(recorder, risk_module):
    recorder.capture(risk_module.get_kurtosis())
    recorder.capture(round(risk_module.get_kurtosis(within_period=False), 4))
    recorder.capture(risk_module.get_kurtosis(period="monthly"))
    recorder.capture(risk_module.get_kurtosis(growth=True))
    recorder.capture(risk_module.get_kurtosis(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_kurtosis(period="monthly", rolling=6))


def test_get_downside_deviation(recorder, risk_module):
    recorder.capture(risk_module.get_downside_deviation())
    recorder.capture(risk_module.get_downside_deviation(within_period=False))
    recorder.capture(risk_module.get_downside_deviation(period="monthly"))
    recorder.capture(risk_module.get_downside_deviation(minimum_acceptable_return=0.01))
    recorder.capture(risk_module.get_downside_deviation(growth=True))
    recorder.capture(risk_module.get_downside_deviation(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_downside_deviation(period="monthly", rolling=6))


def test_get_variance(recorder, risk_module):
    recorder.capture(risk_module.get_variance())
    recorder.capture(risk_module.get_variance(period="monthly"))
    recorder.capture(risk_module.get_variance(growth=True))
    recorder.capture(risk_module.get_variance(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_variance(period="monthly", rolling=6))


def test_get_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_volatility())
    recorder.capture(risk_module.get_volatility(period="monthly"))
    recorder.capture(risk_module.get_volatility(growth=True))
    recorder.capture(risk_module.get_volatility(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_volatility(period="monthly", rolling=6))


def test_get_parkinson_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_parkinson_volatility())
    recorder.capture(risk_module.get_parkinson_volatility(period="monthly"))
    recorder.capture(risk_module.get_parkinson_volatility(growth=True))
    recorder.capture(risk_module.get_parkinson_volatility(growth=True, lag=[1, 2, 3]))


def test_get_garman_klass_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_garman_klass_volatility())
    recorder.capture(risk_module.get_garman_klass_volatility(period="monthly"))
    recorder.capture(risk_module.get_garman_klass_volatility(growth=True))


def test_get_rogers_satchell_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_rogers_satchell_volatility())
    recorder.capture(risk_module.get_rogers_satchell_volatility(period="monthly"))
    recorder.capture(risk_module.get_rogers_satchell_volatility(growth=True))


def test_get_yang_zhang_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_yang_zhang_volatility())
    recorder.capture(risk_module.get_yang_zhang_volatility(period="monthly"))
    recorder.capture(risk_module.get_yang_zhang_volatility(growth=True))


def test_get_excess_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_excess_volatility())
    recorder.capture(risk_module.get_excess_volatility(period="monthly"))
    recorder.capture(risk_module.get_excess_volatility(growth=True))
    recorder.capture(risk_module.get_excess_volatility(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_excess_volatility(period="monthly", rolling=6))


def test_get_hill_estimator(recorder, risk_module):
    recorder.capture(risk_module.get_hill_estimator(period="yearly"))
    recorder.capture(
        risk_module.get_hill_estimator(period="yearly", tail="right", k=0.15)
    )


def test_get_amihud_illiquidity(recorder, risk_module):
    recorder.capture(risk_module.get_amihud_illiquidity(period="quarterly", scale=1e12))
    recorder.capture(risk_module.get_amihud_illiquidity(period="monthly", scale=1e12))
    recorder.capture(risk_module.get_amihud_illiquidity(growth=True, scale=1e12))


def test_get_roll_spread(recorder, risk_module):
    recorder.capture(risk_module.get_roll_spread(period="quarterly"))
    recorder.capture(
        risk_module.get_roll_spread(period="quarterly", within_period=False)
    )


def test_get_har_rv_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_har_rv_forecast().tail())
    recorder.capture(risk_module.get_har_rv_forecast(estimator="parkinson").tail())
    recorder.capture(risk_module.get_har_rv_forecast(estimator="garman_klass").tail())
    recorder.capture(
        risk_module.get_har_rv_forecast(estimator="rogers_satchell").tail()
    )


def test_get_har_rv_forecast_invalid_estimator(risk_module):
    # @handle_errors returns an empty Series rather than propagating the ValueError.
    result = risk_module.get_har_rv_forecast(estimator="bad")
    assert result.empty


def test_get_tail_dependence_coefficient(recorder, risk_module):
    recorder.capture(
        risk_module.get_tail_dependence_coefficient("AAPL", "MSFT", period="yearly")
    )
    recorder.capture(
        risk_module.get_tail_dependence_coefficient(
            "AAPL", "MSFT", period="yearly", method="gaussian"
        )
    )


def test_get_covar(recorder, risk_module):
    recorder.capture(risk_module.get_covar("AAPL", "MSFT", period="yearly"))


def test_get_marginal_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_marginal_value_at_risk(period="yearly"))
    recorder.capture(
        risk_module.get_marginal_value_at_risk(
            weights={"AAPL": 0.7, "MSFT": 0.3}, period="yearly"
        )
    )
    recorder.capture(
        risk_module.get_marginal_value_at_risk(period="yearly", distribution="gaussian")
    )


def test_get_component_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_component_value_at_risk(period="yearly"))
    recorder.capture(
        risk_module.get_component_value_at_risk(
            weights={"AAPL": 0.7, "MSFT": 0.3}, period="yearly"
        )
    )
    recorder.capture(
        risk_module.get_component_value_at_risk(
            period="yearly", distribution="gaussian"
        )
    )


def test_get_acerbi_szekely_test(recorder, risk_module):
    recorder.capture(risk_module.get_acerbi_szekely_test(window_size=100))
    recorder.capture(
        risk_module.get_acerbi_szekely_test(window_size=100, distribution="gaussian")
    )
