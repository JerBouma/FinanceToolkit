"""Econometrics Controller Tests""" ""
# pylint: disable=missing-function-docstring

import numpy as np
import pandas as pd


def test_get_arch_lm_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_arch_lm_test())
    recorder.capture(econometrics_module.get_arch_lm_test(within_period=False))
    recorder.capture(econometrics_module.get_arch_lm_test(period="monthly", lags=3))


def test_get_arch_lm_test_include_benchmark(recorder, econometrics_module):
    # Benchmark is excluded by default -- include_benchmark=True brings it back as a
    # third column.
    default = econometrics_module.get_arch_lm_test(period="monthly", lags=3)
    with_benchmark = econometrics_module.get_arch_lm_test(
        period="monthly", lags=3, include_benchmark=True
    )
    assert "Benchmark" not in default.columns
    assert "Benchmark" in with_benchmark.columns
    recorder.capture(with_benchmark)


def test_get_jarque_bera_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_jarque_bera_test())
    recorder.capture(econometrics_module.get_jarque_bera_test(within_period=False))
    recorder.capture(econometrics_module.get_jarque_bera_test(period="monthly"))


def test_get_ljung_box_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_ljung_box_test())
    recorder.capture(econometrics_module.get_ljung_box_test(within_period=False))
    recorder.capture(econometrics_module.get_ljung_box_test(period="monthly", lags=5))


def test_get_variance_ratio_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_variance_ratio_test())
    recorder.capture(econometrics_module.get_variance_ratio_test(within_period=False))
    recorder.capture(econometrics_module.get_variance_ratio_test(period="monthly", q=4))


def test_get_augmented_dickey_fuller(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_augmented_dickey_fuller(period="quarterly")
    )
    recorder.capture(
        econometrics_module.get_augmented_dickey_fuller(
            period="monthly", regression="n"
        )
    )


def test_get_kpss_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_kpss_test(period="quarterly"))
    recorder.capture(
        econometrics_module.get_kpss_test(period="monthly", regression="ct")
    )


def test_get_phillips_perron_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_phillips_perron_test(period="quarterly"))
    recorder.capture(
        econometrics_module.get_phillips_perron_test(period="monthly", regression="ct")
    )


def test_get_engle_granger_cointegration(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_engle_granger_cointegration(period="quarterly")
    )


def test_get_engle_granger_cointegration_include_benchmark(
    recorder, econometrics_module
):
    # 2 ordered pairs (AAPL, MSFT) by default -- Benchmark brings the ordered pair
    # count up to 6 (3 tickers, permutations of 2).
    default = econometrics_module.get_engle_granger_cointegration(period="quarterly")
    with_benchmark = econometrics_module.get_engle_granger_cointegration(
        period="quarterly", include_benchmark=True
    )
    assert len(default) == 2
    assert len(with_benchmark) == 6
    recorder.capture(with_benchmark)


def test_get_johansen_cointegration(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_johansen_cointegration(period="quarterly"))


def test_get_johansen_cointegration_include_benchmark(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_johansen_cointegration(
            period="quarterly", include_benchmark=True
        )
    )


def test_get_granger_causality(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_granger_causality(period="weekly", max_lag=3)
    )


def test_get_zivot_andrews_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_zivot_andrews_test(period="weekly"))
    recorder.capture(
        econometrics_module.get_zivot_andrews_test(period="weekly", regression="ct")
    )


def test_get_cusum_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_cusum_test(period="quarterly"))
    recorder.capture(
        econometrics_module.get_cusum_test(period="quarterly", within_period=False)
    )


def test_get_diebold_mariano_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_diebold_mariano_test())
    recorder.capture(
        econometrics_module.get_diebold_mariano_test(method_a="ewma", method_b="ewma")
    )


def test_get_diebold_mariano_test_invalid_method(econometrics_module):
    # @handle_errors catches the ValueError and returns an empty Series rather than
    # propagating it.
    result = econometrics_module.get_diebold_mariano_test(method_a="bad")
    assert result.empty


def test_get_ols(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_ols(period="weekly"))


def test_get_ols_include_benchmark(recorder, econometrics_module):
    # Benchmark is excluded from the default independent ticker(s) unless
    # include_benchmark=True is passed.
    default = econometrics_module.get_ols(period="weekly")
    with_benchmark = econometrics_module.get_ols(
        period="weekly", include_benchmark=True
    )
    assert "Benchmark" not in default.index
    assert "Benchmark" in with_benchmark.index
    recorder.capture(with_benchmark)


def test_get_ols_explicit_independent_tickers(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_ols(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_ols_explicit_dependent_ticker(recorder, econometrics_module):
    # Overriding dependent_ticker to MSFT should give the same result as manually
    # passing independent_tickers as every other non-benchmark Toolkit ticker (AAPL).
    override = econometrics_module.get_ols(dependent_ticker="MSFT", period="weekly")
    manual = econometrics_module.get_ols(
        dependent_ticker="MSFT",
        independent_tickers=["AAPL"],
        period="weekly",
    )
    pd.testing.assert_frame_equal(override, manual)
    recorder.capture(override)


def test_get_wls(recorder, econometrics_module):
    returns = econometrics_module._get_price_column("weekly", "Return")
    weights = pd.Series(1.0, index=returns.index)
    recorder.capture(econometrics_module.get_wls(weights, period="weekly"))


def test_get_gls(recorder, econometrics_module):
    returns = econometrics_module._get_price_column("weekly", "Return")
    n = len(returns["AAPL"].dropna())
    omega = pd.DataFrame(np.eye(n))
    recorder.capture(econometrics_module.get_gls(omega, period="weekly"))


def test_get_logistic_regression(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_logistic_regression(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_probit_regression(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_probit_regression(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_quantile_regression(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_quantile_regression(
            independent_tickers=["MSFT", "Benchmark"], tau=0.5, period="weekly"
        )
    )


def test_get_two_sample_t_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_two_sample_t_test(period="weekly"))
    recorder.capture(
        econometrics_module.get_two_sample_t_test(period="weekly", equal_variance=True)
    )


def test_get_two_sample_t_test_include_benchmark(recorder, econometrics_module):
    # 1 unordered pair by default -- Benchmark brings the pair count up to 3.
    default = econometrics_module.get_two_sample_t_test(period="weekly")
    with_benchmark = econometrics_module.get_two_sample_t_test(
        period="weekly", include_benchmark=True
    )
    assert len(default) == 1
    assert len(with_benchmark) == 3
    recorder.capture(with_benchmark)


def test_get_f_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_f_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_f_test_invalid_nesting(econometrics_module):
    # @handle_errors catches the ValueError (unrestricted has fewer/equal parameters
    # than restricted) and returns an empty Series rather than propagating it.
    result = econometrics_module.get_f_test(
        "AAPL", ["MSFT", "Benchmark"], "MSFT", period="weekly"
    )
    assert result.empty


def test_get_likelihood_ratio_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_likelihood_ratio_test(
            "AAPL", "MSFT", ["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_wald_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_wald_test(
            restriction_matrix=[[0, 1, 0], [0, 0, 1]],
            independent_tickers=["MSFT", "Benchmark"],
            period="weekly",
        )
    )


def test_get_wald_test_single_coefficient_matches_ols_t_test(econometrics_module):
    # Testing a single coefficient via the Wald test's F-statistic should exactly
    # match that coefficient's own squared t-statistic from get_ols.
    ols_summary = econometrics_module.get_ols(
        independent_tickers=["MSFT", "Benchmark"], period="weekly", rounding=10
    )
    wald_result = econometrics_module.get_wald_test(
        restriction_matrix=[[0, 1, 0]],
        independent_tickers=["MSFT", "Benchmark"],
        period="weekly",
        rounding=10,
    )

    t_statistic = ols_summary.loc["MSFT", "t-Statistic"]
    assert np.isclose(wald_result["F-Statistic"], t_statistic**2, rtol=1e-6)


def test_get_wald_test_invalid_shape(econometrics_module):
    result = econometrics_module.get_wald_test(
        restriction_matrix=[[1, 0]],
        independent_tickers=["MSFT", "Benchmark"],
        period="weekly",
    )
    assert result.empty


def test_get_hausman_wu_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_hausman_wu_test(
            "AAPL", "MSFT", "Benchmark", period="weekly"
        )
    )


def test_get_hausman_wu_test_reversed_dependent_and_suspect(
    recorder, econometrics_module
):
    # The 3-ticker test fixture (AAPL, MSFT, Benchmark) doesn't leave room for a
    # genuinely distinct `other_independent_tickers` on top of the dependent/suspect/
    # instrument roles, so this exercises a different dependent/suspect/period
    # combination and monthly data instead.
    recorder.capture(
        econometrics_module.get_hausman_wu_test(
            "MSFT", "AAPL", "Benchmark", period="monthly"
        )
    )


def test_get_breusch_pagan_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_breusch_pagan_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_white_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_white_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_durbin_watson_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_durbin_watson_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
    )


def test_get_vif(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_vif(period="weekly"))


def test_get_vif_include_benchmark(recorder, econometrics_module):
    default = econometrics_module.get_vif(period="weekly")
    with_benchmark = econometrics_module.get_vif(
        period="weekly", include_benchmark=True
    )
    assert "Benchmark" not in default.index
    assert "Benchmark" in with_benchmark.index
    recorder.capture(with_benchmark)


def test_get_ramsey_reset_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_ramsey_reset_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly"
        )
    )
    recorder.capture(
        econometrics_module.get_ramsey_reset_test(
            independent_tickers=["MSFT", "Benchmark"], period="weekly", power=2
        )
    )


def test_get_ramsey_reset_test_invalid_power(econometrics_module):
    # @handle_errors catches the ValueError (power < 2) and returns an empty Series
    # rather than propagating it.
    result = econometrics_module.get_ramsey_reset_test(
        independent_tickers=["MSFT", "Benchmark"], period="weekly", power=1
    )
    assert result.empty


def test_get_chow_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_chow_test(
            break_date="2021-06-30",
            independent_tickers=["MSFT", "Benchmark"],
            period="weekly",
        )
    )


def test_get_chow_test_break_date_near_edge_of_sample(econometrics_module):
    # @handle_errors catches the ValueError (too few observations on one side of the
    # split) and returns an empty Series rather than propagating it.
    result = econometrics_module.get_chow_test(
        break_date="2019-12-31",
        independent_tickers=["MSFT", "Benchmark"],
        period="weekly",
    )
    assert result.empty


def test_get_iv_2sls(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_iv_2sls("AAPL", "MSFT", "Benchmark", period="weekly")
    )


def test_get_iv_2sls_with_exogenous(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_iv_2sls(
            "AAPL",
            "MSFT",
            instrument_tickers="Benchmark",
            exogenous_tickers="Benchmark",
            period="weekly",
        )
    )


def test_get_iv_2sls_underidentified(econometrics_module):
    # @handle_errors catches the ValueError (fewer instruments than endogenous
    # regressors) and returns an empty Series rather than propagating it.
    result = econometrics_module.get_iv_2sls(
        "AAPL",
        ["MSFT", "Benchmark"],
        instrument_tickers="Benchmark",
        period="weekly",
    )
    assert result.empty


def test_get_difference_in_differences(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_difference_in_differences(
            treated_tickers="AAPL", treatment_date="2021-06-30", period="weekly"
        )
    )


def test_get_difference_in_differences_explicit_control(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_difference_in_differences(
            treated_tickers="AAPL",
            treatment_date="2021-06-30",
            control_tickers="MSFT",
            period="weekly",
        )
    )


def test_get_regression_discontinuity(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_regression_discontinuity(
            "AAPL", "MSFT", cutoff=0.0, period="weekly"
        )
    )


def test_get_regression_discontinuity_triangular_kernel(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_regression_discontinuity(
            "AAPL", "MSFT", cutoff=0.0, period="weekly", kernel="triangular"
        )
    )


def test_get_regression_discontinuity_invalid_kernel(econometrics_module):
    # @handle_errors catches the ValueError (invalid kernel) and returns an empty
    # Series rather than propagating it.
    result = econometrics_module.get_regression_discontinuity(
        "AAPL", "MSFT", cutoff=0.0, period="weekly", kernel="gaussian"
    )
    assert result.empty


def test_get_propensity_score_matching(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_propensity_score_matching(
            "AAPL", "MSFT", "Benchmark", period="weekly"
        )
    )


def test_get_propensity_score_matching_custom_threshold(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_propensity_score_matching(
            "AAPL",
            "MSFT",
            "Benchmark",
            treatment_threshold=0.01,
            period="weekly",
        )
    )


def test_get_arima_forecast(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_arima_forecast(period="quarterly"))


def test_get_arima_forecast_no_constant(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_arima_forecast(
            period="quarterly", include_constant=False
        )
    )


def test_get_arima_forecast_invalid_order(econometrics_module):
    # @handle_errors catches the ValueError (p and q both 0) and returns an empty
    # DataFrame rather than propagating it.
    result = econometrics_module.get_arima_forecast(period="quarterly", p=0, q=0)
    assert result.empty


def test_get_var_forecast(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_var_forecast(period="quarterly"))


def test_get_var_forecast_multiple_lags(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_var_forecast(period="weekly", lags=2, forecast_steps=3)
    )


def test_get_var_forecast_invalid_lags(econometrics_module):
    result = econometrics_module.get_var_forecast(period="quarterly", lags=0)
    assert result.empty


def test_get_vecm_forecast_not_cointegrated_returns_empty(econometrics_module):
    # The test fixture's AAPL/MSFT prices are not (reliably) cointegrated at the 5%
    # level -- @handle_errors catches the resulting ValueError and returns an empty
    # DataFrame rather than propagating it.
    result = econometrics_module.get_vecm_forecast(period="weekly")
    assert result.empty


def test_get_vecm_forecast_invalid_k_ar_diff(econometrics_module):
    result = econometrics_module.get_vecm_forecast(period="weekly", k_ar_diff=0)
    assert result.empty


def test_get_rmse(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rmse(period="quarterly"))


def test_get_mae(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_mae(period="quarterly"))


def test_get_out_of_sample_validation_arima(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_out_of_sample_validation(
            period="weekly", model="arima", p=1, d=1, q=1
        )
    )


def test_get_out_of_sample_validation_var(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_out_of_sample_validation(
            period="weekly", model="var", lags=1
        )
    )


def test_get_out_of_sample_validation_var_missing_other_tickers(econometrics_module):
    # Explicitly passing an empty other_tickers override (rather than relying on the
    # default, which derives every other Toolkit ticker automatically) exercises the
    # "no other series to model jointly with" ValueError -- @handle_errors catches it
    # and returns an empty DataFrame rather than propagating it.
    result = econometrics_module.get_out_of_sample_validation(
        period="weekly", model="var", other_tickers=[]
    )
    assert result.empty


def test_get_out_of_sample_validation_invalid_model(econometrics_module):
    result = econometrics_module.get_out_of_sample_validation(
        period="weekly", model="bad"
    )
    assert result.empty


def test_get_fixed_effects_broadcast_factor(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_fixed_effects(
            independent_tickers="Benchmark", period="weekly"
        )
    )


def test_get_fixed_effects_entity_column(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_fixed_effects(
            independent_column="Volume", period="weekly"
        )
    )


def test_get_fixed_effects_time_effects(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_fixed_effects(
            independent_column="Volume",
            period="weekly",
            entity_effects=True,
            time_effects=False,
        )
    )


def test_get_fixed_effects_missing_regressor_args(econometrics_module):
    # @handle_errors catches the ValueError (neither independent_tickers nor
    # independent_column given) and returns an empty result rather than
    # propagating it.
    result = econometrics_module.get_fixed_effects(period="weekly")
    assert result.empty


def test_get_fixed_effects_both_regressor_args(econometrics_module):
    # @handle_errors catches the ValueError (both independent_tickers and
    # independent_column given) and returns an empty result rather than
    # propagating it.
    result = econometrics_module.get_fixed_effects(
        independent_tickers="Benchmark", independent_column="Volume", period="weekly"
    )
    assert result.empty


def test_get_random_effects(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_random_effects(
            independent_column="Volume", period="weekly"
        )
    )


def test_get_random_effects_not_enough_entities(econometrics_module):
    # @handle_errors catches the ValueError (only 2 entities for 1 regressor plus
    # an intercept) and returns an empty result rather than propagating it.
    result = econometrics_module.get_random_effects(
        independent_column="Volume",
        dependent_tickers=["AAPL", "MSFT"],
        period="weekly",
    )
    assert result.empty


def test_get_hausman_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_hausman_test(
            independent_column="Volume", period="weekly"
        )
    )


def test_get_hausman_test_not_enough_entities(econometrics_module):
    result = econometrics_module.get_hausman_test(
        independent_column="Volume",
        dependent_tickers=["AAPL", "MSFT"],
        period="weekly",
    )
    assert result.empty
