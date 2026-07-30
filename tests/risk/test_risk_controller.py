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


def test_get_arch_lm_test(recorder, risk_module):
    recorder.capture(risk_module.get_arch_lm_test())
    recorder.capture(risk_module.get_arch_lm_test(within_period=False))
    recorder.capture(risk_module.get_arch_lm_test(period="monthly", lags=3))


def test_get_jarque_bera_test(recorder, risk_module):
    recorder.capture(risk_module.get_jarque_bera_test())
    recorder.capture(risk_module.get_jarque_bera_test(within_period=False))
    recorder.capture(risk_module.get_jarque_bera_test(period="monthly"))


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


def test_get_excess_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_excess_volatility())
    recorder.capture(risk_module.get_excess_volatility(period="monthly"))
    recorder.capture(risk_module.get_excess_volatility(growth=True))
    recorder.capture(risk_module.get_excess_volatility(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_excess_volatility(period="monthly", rolling=6))
