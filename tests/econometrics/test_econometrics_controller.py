"""Econometrics Controller Tests""" ""
# pylint: disable=missing-function-docstring

import numpy as np
import pandas as pd
import pytest


def test_get_arch_lm_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_arch_lm_test())
    recorder.capture(econometrics_module.get_arch_lm_test(within_period=False))
    recorder.capture(econometrics_module.get_arch_lm_test(period="monthly", lags=3))


def test_get_arch_lm_test_include_benchmark(recorder, econometrics_module):
    # Benchmark is excluded by default; include_benchmark=True adds a third column.
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


def test_default_period_is_daily(econometrics_module):
    # The Toolkit instance here has no intraday data, so a daily default only works
    # because within_period defaults to False.
    pd.testing.assert_frame_equal(
        econometrics_module.get_jarque_bera_test(),
        econometrics_module.get_jarque_bera_test(period="daily", within_period=False),
    )


def test_within_period_daily_requires_intraday(econometrics_module):
    # Nesting observations inside a single day needs intraday data, which this Toolkit
    # instance does not have. @handle_errors returns an empty Series rather than
    # propagating the ValueError.
    assert econometrics_module.get_jarque_bera_test(
        period="daily", within_period=True
    ).empty


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
    # 2 ordered pairs by default; Benchmark takes the count up to 6.
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
    # @handle_errors returns an empty Series rather than propagating the ValueError.
    result = econometrics_module.get_diebold_mariano_test(method_a="bad")
    assert result.empty


def test_get_ols(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_ols(period="weekly"))


def test_get_ols_include_benchmark(recorder, econometrics_module):
    # Benchmark is excluded from the default independent tickers.
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
    # Overriding dependent_ticker should match passing the others explicitly.
    override = econometrics_module.get_ols(dependent_ticker="MSFT", period="weekly")
    manual = econometrics_module.get_ols(
        dependent_ticker="MSFT",
        independent_tickers=["AAPL"],
        period="weekly",
    )
    pd.testing.assert_frame_equal(override, manual)
    recorder.capture(override)


def test_get_ols_hac(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_ols(period="weekly", cov_type="HAC", maxlags=4)
    )


def test_get_wls_hac(recorder, econometrics_module):
    returns = econometrics_module._get_series_column("weekly", "Return")
    weights = pd.Series(1.0, index=returns.index)
    recorder.capture(
        econometrics_module.get_wls(weights, period="weekly", cov_type="HAC", maxlags=4)
    )


def test_get_wls(recorder, econometrics_module):
    returns = econometrics_module._get_series_column("weekly", "Return")
    weights = pd.Series(1.0, index=returns.index)
    recorder.capture(econometrics_module.get_wls(weights, period="weekly"))


def test_get_gls(recorder, econometrics_module):
    returns = econometrics_module._get_series_column("weekly", "Return")
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


def test_get_fama_macbeth_regression(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_fama_macbeth_regression(
            period="weekly", add_constant=False
        )
    )


def test_get_fama_macbeth_regression_explicit_tickers(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_fama_macbeth_regression(
            factor_tickers="Benchmark",
            asset_tickers=["AAPL", "MSFT"],
            period="weekly",
            add_constant=False,
        )
    )


def test_get_fama_macbeth_regression_too_few_assets(econometrics_module):
    # add_constant=True needs more assets than factors + 1, so this is unidentified.
    result = econometrics_module.get_fama_macbeth_regression(period="weekly")
    assert result.empty


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
    # @handle_errors returns an empty Series rather than propagating the ValueError.
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
    # A single-coefficient Wald F should equal that coefficient's squared t.
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
    # The 3-ticker fixture has no room for a distinct other_independent_tickers.
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
    # @handle_errors returns an empty Series rather than propagating the ValueError.
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
    # @handle_errors returns an empty Series rather than propagating the ValueError.
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
    # @handle_errors returns an empty Series rather than propagating the ValueError.
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
    # @handle_errors returns an empty Series rather than propagating the ValueError.
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


def test_get_synthetic_control(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_synthetic_control(
            "AAPL",
            treatment_period="2021-07-01",
            donor_tickers=["MSFT", "Benchmark"],
            period="weekly",
        )
    )


def test_get_synthetic_control_default_donor_tickers(recorder, econometrics_module):
    # include_benchmark=True brings in MSFT and Benchmark, meeting the 2-donor floor.
    recorder.capture(
        econometrics_module.get_synthetic_control(
            "AAPL",
            treatment_period="2021-07-01",
            period="weekly",
            include_benchmark=True,
        )
    )


def test_get_synthetic_control_too_few_donors(econometrics_module):
    # include_benchmark=False leaves only MSFT, under the minimum of 2 donors.
    result = econometrics_module.get_synthetic_control(
        "AAPL", treatment_period="2021-07-01", period="weekly"
    )
    assert result.empty


def test_get_arima_forecast(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_arima_forecast(period="quarterly"))


def test_get_arima_forecast_no_constant(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_arima_forecast(
            period="quarterly", include_constant=False
        )
    )


def test_get_arima_forecast_invalid_order(econometrics_module):
    # @handle_errors returns an empty DataFrame rather than propagating the ValueError.
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


def test_get_impulse_response_function(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_impulse_response_function(period="weekly", periods=5)
    )


def test_get_impulse_response_function_reduced_form(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_impulse_response_function(
            period="weekly", periods=5, orthogonalized=False
        )
    )


def test_get_impulse_response_function_invalid_periods(econometrics_module):
    result = econometrics_module.get_impulse_response_function(
        period="weekly", periods=0
    )
    assert result.empty


def test_get_variance_decomposition(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_variance_decomposition(period="weekly", periods=5)
    )


def test_get_variance_decomposition_invalid_periods(econometrics_module):
    result = econometrics_module.get_variance_decomposition(period="weekly", periods=0)
    assert result.empty


def test_get_vecm_forecast_not_cointegrated_returns_empty(econometrics_module):
    # The fixture's prices are not reliably cointegrated, so this returns empty.
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
    # q=0: ARMA(1, 1) on near-white-noise differences is unidentified on this fixture.
    recorder.capture(
        econometrics_module.get_out_of_sample_validation(
            period="weekly", model="arima", p=1, d=1, q=0
        )
    )


def test_get_out_of_sample_validation_var(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_out_of_sample_validation(
            period="weekly", model="var", lags=1
        )
    )


def test_get_out_of_sample_validation_var_missing_other_tickers(econometrics_module):
    # An explicit empty other_tickers exercises the 'nothing to model jointly' error.
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
    # @handle_errors returns an empty result rather than propagating the ValueError.
    result = econometrics_module.get_fixed_effects(period="weekly")
    assert result.empty


def test_get_fixed_effects_both_regressor_args(econometrics_module):
    # @handle_errors returns an empty result rather than propagating the ValueError.
    result = econometrics_module.get_fixed_effects(
        independent_tickers="Benchmark", independent_column="Volume", period="weekly"
    )
    assert result.empty


def test_get_random_effects(recorder, econometrics_module):
    # The Benchmark is a third entity next to AAPL and MSFT, which Random Effects needs.
    recorder.capture(
        econometrics_module.get_random_effects(
            independent_column="Volume", period="weekly", include_benchmark=True
        )
    )


def test_get_random_effects_excludes_the_benchmark_by_default(econometrics_module):
    # Without the Benchmark only AAPL and MSFT remain, too few entities to fit.
    result = econometrics_module.get_random_effects(
        independent_column="Volume", period="weekly"
    )
    assert result.empty


def test_get_random_effects_not_enough_entities(econometrics_module):
    # @handle_errors returns an empty result rather than propagating the ValueError.
    result = econometrics_module.get_random_effects(
        independent_column="Volume",
        dependent_tickers=["AAPL", "MSFT"],
        period="weekly",
    )
    assert result.empty


def test_get_hausman_test(recorder, econometrics_module):
    # The Benchmark is a third entity next to AAPL and MSFT, which Random Effects needs.
    recorder.capture(
        econometrics_module.get_hausman_test(
            independent_column="Volume", period="weekly", include_benchmark=True
        )
    )


def test_get_hausman_test_excludes_the_benchmark_by_default(econometrics_module):
    # Without the Benchmark only AAPL and MSFT remain, too few entities to fit.
    result = econometrics_module.get_hausman_test(
        independent_column="Volume", period="weekly"
    )
    assert result.empty


def test_get_hausman_test_not_enough_entities(econometrics_module):
    result = econometrics_module.get_hausman_test(
        independent_column="Volume",
        dependent_tickers=["AAPL", "MSFT"],
        period="weekly",
    )
    assert result.empty


def test_get_event_study(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_event_study(event_date="2021-03-11"))


def test_get_event_study_default_ticker(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_event_study(
            event_date="2021-03-11", dependent_ticker="MSFT"
        )
    )


def test_get_fixed_effects_names_a_factor_that_is_not_loaded(
    econometrics_module, monkeypatch
):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="SPY are not among the loaded tickers"):
        econometrics_module.get_fixed_effects(independent_tickers="SPY")


def test_pair_tables_are_rounded(econometrics_module):
    # Each pair's row mixes numbers and a boolean, which left object columns that the
    # rounding skipped.
    for result in (
        econometrics_module.get_engle_granger_cointegration(period="quarterly"),
        econometrics_module.get_granger_causality(period="weekly", max_lag=3),
        econometrics_module.get_two_sample_t_test(period="weekly"),
    ):
        numeric = result.select_dtypes("number")

        assert not numeric.empty
        assert (numeric.round(4) == numeric).all().all()


def test_per_ticker_statistics_ignore_a_younger_ticker(econometrics_module):
    from financetoolkit.econometrics.econometrics_controller import Econometrics

    # MSFT only starts halfway through the sample: AAPL's per-ticker statistics, and
    # the AAPL/Benchmark pair statistics, must not lose the dates MSFT is missing.
    historical_data = {}
    # Read per period: the Toolkit's module loads a period when it is first asked for.
    for period in ("daily", "weekly", "monthly", "quarterly", "yearly"):
        frame = econometrics_module._historical_data[period].copy()
        if not frame.empty:
            msft_columns = [column for column in frame.columns if column[1] == "MSFT"]
            frame.iloc[: len(frame) // 2, frame.columns.get_indexer(msft_columns)] = (
                np.nan
            )
        historical_data[period] = frame

    younger = Econometrics(
        tickers=econometrics_module._tickers,
        historical_data=historical_data,
        rounding=econometrics_module._rounding,
    )

    for method, kwargs in [
        ("get_augmented_dickey_fuller", {"period": "weekly"}),
        ("get_kpss_test", {"period": "weekly"}),
        ("get_phillips_perron_test", {"period": "weekly"}),
        ("get_zivot_andrews_test", {"period": "monthly"}),
        ("get_arch_lm_test", {}),
        ("get_jarque_bera_test", {}),
        ("get_ljung_box_test", {}),
        ("get_variance_ratio_test", {}),
        ("get_cusum_test", {}),
        ("get_diebold_mariano_test", {}),
    ]:
        original = getattr(econometrics_module, method)(rounding=None, **kwargs)
        result = getattr(younger, method)(rounding=None, **kwargs)
        pd.testing.assert_series_equal(
            result["AAPL"], original["AAPL"], obj=method, check_like=True
        )
        assert not result["MSFT"].equals(original["MSFT"]), method

    for method in ("get_two_sample_t_test", "get_rmse", "get_mae"):
        original = getattr(econometrics_module, method)(
            include_benchmark=True, rounding=None
        )
        result = getattr(younger, method)(include_benchmark=True, rounding=None)
        pd.testing.assert_series_equal(
            pd.Series(result.loc[("AAPL", "Benchmark")]),
            pd.Series(original.loc[("AAPL", "Benchmark")]),
            obj=method,
        )

    # The two independent samples keep their own lengths: AAPL's mean is its full mean.
    t_test = younger.get_two_sample_t_test(rounding=None).loc[("AAPL", "MSFT")]
    returns = econometrics_module._historical_data["daily"]["Return"]
    assert np.isclose(t_test["Mean A"], returns["AAPL"].dropna().mean(), atol=1e-4)
    assert np.isclose(
        t_test["Mean B"],
        historical_data["daily"]["Return"]["MSFT"].dropna().mean(),
        atol=1e-4,
    )


def test_get_difference_in_differences_default_controls_exclude_benchmark(
    econometrics_module,
):
    default = econometrics_module.get_difference_in_differences(
        treated_tickers="AAPL", treatment_date="2021-06-30", period="weekly"
    )
    explicit = econometrics_module.get_difference_in_differences(
        treated_tickers="AAPL",
        treatment_date="2021-06-30",
        control_tickers="MSFT",
        period="weekly",
    )
    pd.testing.assert_frame_equal(default, explicit)


def _econometrics_with_msft(econometrics_module, rows):
    from financetoolkit.econometrics.econometrics_controller import Econometrics

    # `rows` is the slice of each period's history where MSFT has no data; None
    # removes MSFT from the Toolkit altogether.
    historical_data = {}
    # Read per period: the Toolkit's module loads a period when it is first asked for.
    for period in ("daily", "weekly", "monthly", "quarterly", "yearly"):
        frame = econometrics_module._historical_data[period].copy()
        msft_columns = [column for column in frame.columns if column[1] == "MSFT"]
        if rows is None:
            frame = frame.drop(columns=msft_columns)
        elif not frame.empty:
            frame[msft_columns] = frame[msft_columns].astype(float)
            frame.iloc[rows, frame.columns.get_indexer(msft_columns)] = np.nan
        historical_data[period] = frame

    return Econometrics(
        tickers=[
            ticker
            for ticker in econometrics_module._tickers
            if rows is not None or ticker != "MSFT"
        ],
        historical_data=historical_data,
        rounding=econometrics_module._rounding,
    )


def test_models_ignore_tickers_they_do_not_use(econometrics_module):
    # MSFT misses its first 300 days ("young") or all of its data ("empty"): AAPL's
    # ARIMA forecast, the Benchmark/AAPL Engle-Granger test and the AAPL~Benchmark
    # regression do not use MSFT, so they equal the results without MSFT.
    without_msft = _econometrics_with_msft(econometrics_module, None)

    def results(econometrics):
        return (
            econometrics.get_arima_forecast(rounding=None)["AAPL"],
            econometrics.get_engle_granger_cointegration(
                include_benchmark=True, rounding=None
            ).loc[("Benchmark", "AAPL")],
            econometrics.get_ols(
                dependent_ticker="AAPL",
                independent_tickers="Benchmark",
                rounding=None,
            ),
        )

    expected = results(without_msft)
    assert not expected[0].empty and not expected[2].empty

    for rows in (slice(0, 300), slice(None)):
        arima, engle_granger, ols = results(
            _econometrics_with_msft(econometrics_module, rows)
        )
        pd.testing.assert_series_equal(arima, expected[0])
        pd.testing.assert_series_equal(engle_granger, expected[1])
        pd.testing.assert_frame_equal(ols, expected[2])


def test_get_difference_in_differences_without_control_data(
    econometrics_module, monkeypatch
):
    empty = _econometrics_with_msft(econometrics_module, slice(None))
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")

    with pytest.raises(ValueError, match="no control observations"):
        empty.get_difference_in_differences(
            treated_tickers="AAPL", treatment_date="2021-06-30", period="weekly"
        )
