"""Econometrics Controller Tests""" ""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.statistics import entropy_model


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


def test_get_iv_2sls_with_the_instrument_as_exogenous_is_refused(econometrics_module):
    # A ticker that is both the instrument and an exogenous regressor leaves the first stage
    # without an excluded instrument, so there is no fit (the fit with exogenous controls is
    # covered in test_causal_inference_model.py). @handle_errors returns an empty Series.
    result = econometrics_module.get_iv_2sls(
        "AAPL",
        "MSFT",
        instrument_tickers="Benchmark",
        exogenous_tickers="Benchmark",
        period="weekly",
    )
    assert result.empty


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
    # A fit on a dozen quarters lands slightly differently per platform's libraries (in the
    # fourth digit), so the price forecast is compared in whole units.
    recorder.capture(
        econometrics_module.get_arima_forecast(period="quarterly").round(0)
    )


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


def test_get_wls_estimates_feasible_weights_by_default(econometrics_module):
    from financetoolkit.econometrics.regression import regression_model  # noqa: PLC0415

    returns = econometrics_module._get_series_column("weekly", "Return")[
        ["AAPL", "MSFT"]
    ].dropna()
    weights = pd.Series(
        regression_model.get_feasible_weights(returns["AAPL"], returns["MSFT"]),
        index=returns.index,
    )

    pd.testing.assert_frame_equal(
        econometrics_module.get_wls(period="weekly"),
        econometrics_module.get_wls(weights, period="weekly"),
    )
    assert econometrics_module.get_wls("unknown", period="weekly").empty


def test_get_gls_estimates_an_ar1_structure_by_default(econometrics_module):
    from financetoolkit.econometrics.regression import regression_model  # noqa: PLC0415

    returns = econometrics_module._get_series_column("weekly", "Return")
    omega = pd.DataFrame(
        regression_model.get_ar1_covariance(returns["AAPL"], returns[["MSFT"]])
    )

    pd.testing.assert_frame_equal(
        econometrics_module.get_gls(period="weekly"),
        econometrics_module.get_gls(omega, period="weekly"),
    )


def test_get_wald_test_takes_lists(econometrics_module):
    """The MCP server passes the restriction matrix and values as lists."""
    from_lists = econometrics_module.get_wald_test(
        [[0, 1]], period="weekly", restriction_values=[0.5]
    )
    from_arrays = econometrics_module.get_wald_test(
        np.array([[0.0, 1.0]]), period="weekly", restriction_values=np.array([0.5])
    )

    assert not from_lists.empty
    pd.testing.assert_series_equal(from_lists, from_arrays)


def test_level_significance_measures_the_ratio_of_two_series():
    from financetoolkit import Econometrics

    months = pd.period_range("2026-01", "2026-06", freq="M")
    data = pd.DataFrame(
        {"Exports": [10.0, 12, 14, 16, 18, 20], "GDP": [100.0] * 6}, index=months
    )

    significance = Econometrics(data=data).get_level_significance(
        spread_over="GDP", ratio=True
    )

    assert significance.index.tolist() == ["Exports / GDP"]
    assert significance.loc["Exports / GDP", "Latest Value"] == pytest.approx(0.2)


def test_get_move_significance_measures_yield_moves_in_points():
    from financetoolkit import Econometrics

    days = pd.bdate_range("2025-01-01", periods=60)
    rng = np.random.default_rng(1)
    yields = pd.DataFrame(
        {"2 Year": 0.04 + np.cumsum(rng.normal(0, 0.0003, len(days)))}, index=days
    )
    # Day 30 rises 0.002, the last day 0.0017: the largest rise since day 30.
    yields.iloc[30:, 0] += 0.002
    yields.iloc[-1, 0] = yields.iloc[-2, 0] + 0.0017

    significance = Econometrics(
        data=yields, changes="difference"
    ).get_move_significance(window=20, rounding=6)

    latest = significance.iloc[-1].unstack(level=0)
    assert latest.loc["2 Year", "Return"] == pytest.approx(0.0017)
    assert latest.loc["2 Year", "Largest Since"] == str(days[30].date())
    assert latest.loc["2 Year", "Volatility Multiple"] > 3  # noqa: PLR2004


def _prices_with_a_holiday():
    """
    Three days of a Toolkit price history in which ASML did not trade on the second day
    while the benchmark did: the last close is carried forward without a Return.
    """
    dates = pd.period_range("2025-04-30", periods=3, freq="D")
    close = pd.DataFrame(
        {"ASML": [582.5, 582.5, 608.9], "Benchmark": [554.54, 558.47, 566.76]},
        index=dates,
    )
    returns = pd.DataFrame(
        {"ASML": [np.nan, np.nan, 0.0453], "Benchmark": [np.nan, 0.0071, 0.0148]},
        index=dates,
    )

    return pd.concat({"Close": close, "Return": returns}, axis=1)


def test_a_carried_forward_price_is_not_an_observation():
    from financetoolkit.econometrics.econometrics_controller import Econometrics
    from financetoolkit.risk.helpers import mask_non_trading_days

    frame = _prices_with_a_holiday()

    # The first price has no Return by construction and is kept; the filled day is not.
    masked = mask_non_trading_days(frame, "Close")
    assert masked["ASML"].tolist()[0] == 582.5
    assert np.isnan(masked["ASML"].tolist()[1])
    assert masked["Benchmark"].notna().all()

    econometrics = Econometrics(
        tickers=["ASML", "Benchmark"], historical_data={"daily": frame}, rounding=None
    )
    significance = econometrics.get_level_significance(period="daily")

    assert significance.loc["ASML", "Observations"] == 2
    assert significance.loc["ASML", "Start Date"] == "2025-04-30"


def test_a_data_series_keeps_its_observations():
    from financetoolkit.econometrics.econometrics_controller import Econometrics

    # Series given with data= hold their own observations: an unchanged value is one.
    series = pd.Series(
        [1.0, 1.0, 2.0], index=pd.period_range("2025-04-30", periods=3, freq="D")
    )
    econometrics = Econometrics(data={"Rate": series}, rounding=None)

    assert (
        econometrics.get_level_significance(period="daily").loc["Rate", "Observations"]
        == 3
    )


def test_get_linear_regression(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_linear_regression())
    recorder.capture(econometrics_module.get_linear_regression(window=20))
    recorder.capture(econometrics_module.get_linear_regression(include_benchmark=True))


def test_get_slope(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_slope())
    recorder.capture(econometrics_module.get_slope(window=5))
    recorder.capture(econometrics_module.get_slope(as_angle=True))
    recorder.capture(econometrics_module.get_slope(period="weekly"))


def test_get_rolling_correlation(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rolling_correlation(column="Adj Close"))
    recorder.capture(
        econometrics_module.get_rolling_correlation(column="High", against="Low")
    )
    recorder.capture(
        econometrics_module.get_rolling_correlation(column="Adj Close", against="MSFT")
    )
    recorder.capture(econometrics_module.get_rolling_correlation())


def test_get_rolling_median(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rolling_median())
    recorder.capture(econometrics_module.get_rolling_median(window=9))
    recorder.capture(econometrics_module.get_rolling_median(column="Return"))


def test_get_rolling_percentile(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rolling_percentile())
    recorder.capture(
        econometrics_module.get_rolling_percentile(window=30, percentile=0.75)
    )
    recorder.capture(econometrics_module.get_rolling_percentile(method="linear"))
    recorder.capture(econometrics_module.get_rolling_percentile(period="weekly"))


def test_get_percent_rank(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_percent_rank())
    recorder.capture(econometrics_module.get_percent_rank(window=30))
    recorder.capture(econometrics_module.get_percent_rank(column="Return"))


def test_get_rolling_z_score(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rolling_z_score())
    recorder.capture(econometrics_module.get_rolling_z_score(window=20))
    recorder.capture(econometrics_module.get_rolling_z_score(column="Return"))


def test_get_rolling_mean_absolute_deviation(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rolling_mean_absolute_deviation())
    recorder.capture(econometrics_module.get_rolling_mean_absolute_deviation(window=30))
    recorder.capture(
        econometrics_module.get_rolling_mean_absolute_deviation(column="Return")
    )


def test_get_rolling_standard_error(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rolling_standard_error())
    recorder.capture(econometrics_module.get_rolling_standard_error(window=30))
    recorder.capture(
        econometrics_module.get_rolling_standard_error(include_benchmark=True)
    )


def test_get_rolling_entropy(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_rolling_entropy())
    recorder.capture(econometrics_module.get_rolling_entropy(window=20, base=np.e))
    recorder.capture(econometrics_module.get_rolling_entropy(period="weekly"))


def test_rolling_correlation_with_an_unknown_series_is_empty(econometrics_module):
    assert econometrics_module.get_rolling_correlation(against="UNKNOWN").empty


def test_rolling_correlation_leaves_out_the_series_it_is_correlated_with(
    econometrics_module,
):
    # MSFT against itself would be 1 everywhere; the Benchmark is left out by default.
    correlation = econometrics_module.get_rolling_correlation(against="MSFT")
    assert list(correlation.columns) == ["AAPL"]

    with_benchmark = econometrics_module.get_rolling_correlation(
        against="MSFT", include_benchmark=True
    )
    assert list(with_benchmark.columns) == ["AAPL", "Benchmark"]


def test_rolling_statistics_skip_the_dates_a_series_has_no_value():
    from financetoolkit.econometrics.econometrics_controller import Econometrics

    # A monthly series next to a daily one: each window counts its own observations.
    dates = pd.period_range("2025-01-01", periods=90, freq="D")
    daily = pd.Series(np.arange(90.0) + 1, index=dates)
    monthly = daily[daily.index.day == 1]
    econometrics = Econometrics(
        data={"Daily": daily, "Monthly": monthly}, rounding=None
    )

    median = econometrics.get_rolling_median(window=3)

    assert median["Monthly"].dropna().tolist() == [32.0]
    assert median["Daily"].dropna().iloc[0] == 2.0


def test_rolling_statistics_default_windows():
    import inspect

    from financetoolkit.econometrics.econometrics_controller import Econometrics

    # The TA-Lib defaults, and pandas-ta-classic's where TA-Lib has no such function.
    for method, window in (
        (Econometrics.get_linear_regression, 14),
        (Econometrics.get_rolling_correlation, 30),
        (Econometrics.get_rolling_median, 30),
        (Econometrics.get_rolling_percentile, 100),
        (Econometrics.get_percent_rank, 100),
        (Econometrics.get_rolling_mean_absolute_deviation, 14),
        (Econometrics.get_rolling_z_score, 30),
        (Econometrics.get_rolling_standard_error, 14),
        (Econometrics.get_rolling_entropy, 10),
        (Econometrics.get_slope, 1),
    ):
        assert inspect.signature(method).parameters["window"].default == window


def test_the_portfolio_is_no_default_independent_of_its_holdings():
    """A holding regressed on the others and their Portfolio is collinear by construction."""
    from types import SimpleNamespace

    from financetoolkit.econometrics.econometrics_controller import EconometricsBase

    stub = SimpleNamespace(
        _get_tickers=lambda include_benchmark=False: ["AAPL", "MSFT", "Portfolio"]
    )

    assert EconometricsBase._resolve_dependent_independent(stub) == ("AAPL", ["MSFT"])
    assert EconometricsBase._resolve_dependent_independent(
        stub, dependent_ticker="Portfolio"
    ) == ("Portfolio", ["AAPL", "MSFT"])
    assert EconometricsBase._resolve_dependent_independent(
        stub, independent_tickers=["MSFT", "Portfolio"]
    ) == ("AAPL", ["MSFT", "Portfolio"])


def test_get_fractional_difference(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_fractional_difference(period="weekly", threshold=1e-3)
    )
    recorder.capture(
        econometrics_module.get_fractional_difference(
            period="weekly", d=0.6, method="expanding"
        )
    )


def test_get_minimum_fractional_difference(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_minimum_fractional_difference(period="weekly")
    )
    recorder.capture(
        econometrics_module.get_minimum_fractional_difference(
            period="weekly", threshold=1e-3, grid=True
        )
    )


def test_get_bubble_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_bubble_test(period="weekly", replications=100)
    )
    recorder.capture(
        econometrics_module.get_bubble_test(
            period="weekly", method="sadf", lag_order=1, replications=100
        )
    )


def test_get_bubble_episodes(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_bubble_episodes(period="weekly", replications=100)
    )
    recorder.capture(
        econometrics_module.get_bubble_episodes(
            period="weekly", replications=100, time_series=True
        )
    )
    # The runs above the critical value in the test data are shorter than the default
    # minimum duration (round(log(157)) = 5 weeks), so these episodes only appear with 1.
    recorder.capture(
        econometrics_module.get_bubble_episodes(
            period="weekly", replications=100, minimum_duration=1
        )
    )


def test_get_chow_dickey_fuller_test(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_chow_dickey_fuller_test(
            period="weekly", replications=200
        )
    )
    recorder.capture(
        econometrics_module.get_chow_dickey_fuller_test(
            period="weekly", time_series=True
        )
    )


def test_log_prices_need_positive_values(econometrics_module):
    # @handle_errors returns an empty Series for the ValueError on a non-positive series.
    assert econometrics_module.get_fractional_difference(column="Return").empty


def test_get_entropy(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_entropy())
    recorder.capture(econometrics_module.get_entropy(method="approximate", window=100))
    recorder.capture(econometrics_module.get_entropy(method="permutation", order=4))
    recorder.capture(
        econometrics_module.get_entropy(method="spectral", estimator="welch")
    )
    recorder.capture(econometrics_module.get_entropy(method="svd", period="weekly"))
    recorder.capture(econometrics_module.get_entropy(method="shannon", word_length=2))
    recorder.capture(
        econometrics_module.get_entropy(method="lempel_ziv", encoding="quantile")
    )
    recorder.capture(
        econometrics_module.get_entropy(
            method="kontoyiannis", window=252, encoding="sigma"
        )
    )


def test_get_lempel_ziv_complexity(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_lempel_ziv_complexity())
    recorder.capture(econometrics_module.get_lempel_ziv_complexity(window=100))
    recorder.capture(
        econometrics_module.get_lempel_ziv_complexity(encoding="quantile", bins=4)
    )


def test_get_hjorth_parameters(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_hjorth_parameters())
    recorder.capture(econometrics_module.get_hjorth_parameters(window=100))


def test_entropy_of_each_series_over_its_own_dates(econometrics_module):
    entropy = econometrics_module.get_entropy(method="permutation", rounding=10)
    returns = econometrics_module._get_series_column("daily", "Return")

    for ticker in entropy.index:
        assert entropy[ticker] == pytest.approx(
            entropy_model.get_permutation_entropy(returns[ticker].dropna().to_numpy())
        )


def test_entropy_with_an_unknown_method_is_empty(econometrics_module):
    assert econometrics_module.get_entropy(method="unknown").empty


def test_get_berkowitz_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_berkowitz_test(refit_every=63))
    recorder.capture(
        econometrics_module.get_berkowitz_test(model="gaussian", alpha=0.05)
    )


def test_get_hong_li_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_hong_li_test(model="gaussian"))
    # The GARCH refits move the statistics by up to 5e-7 with the fit, so two decimals.
    recorder.capture(econometrics_module.get_hong_li_test(refit_every=63, rounding=2))


def test_get_directional_accuracy_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_directional_accuracy_test())
    recorder.capture(
        econometrics_module.get_directional_accuracy_test(
            method="anatolyev_gerko", forecast="autoregressive", window_size=60
        )
    )


def test_get_model_confidence_set(recorder, econometrics_module):
    recorder.capture(
        econometrics_module.get_model_confidence_set(
            models=["ewma", "rolling", "garch"], refit_every=63
        )
    )
    recorder.capture(
        econometrics_module.get_model_confidence_set(
            models=["ewma", "rolling"], loss="squared", statistic="range", rounding=12
        )
    )


def test_get_model_confidence_set_needs_two_models(econometrics_module):
    assert econometrics_module.get_model_confidence_set(models=["ewma"]).empty


def test_get_arch_lm_test_weighted(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_arch_lm_test(weighted=True))
    recorder.capture(
        econometrics_module.get_arch_lm_test(period="weekly", lags=10, weighted=True)
    )


def test_get_ljung_box_test_weighted(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_ljung_box_test(weighted=True))
    recorder.capture(
        econometrics_module.get_ljung_box_test(period="weekly", lags=5, weighted=True)
    )


def test_get_cusum_test_methods(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_cusum_test(method="recursive"))
    recorder.capture(econometrics_module.get_cusum_test(method="chu_stinchcombe_white"))
    recorder.capture(
        econometrics_module.get_cusum_test(
            period="weekly", method="chu_stinchcombe_white", time_series=True
        ).astype(float)
    )


def test_get_cusum_test_time_series_needs_chu_stinchcombe_white(econometrics_module):
    # @handle_errors returns an empty Series rather than propagating the ValueError.
    assert econometrics_module.get_cusum_test(time_series=True).empty


def test_get_sign_bias_test(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_sign_bias_test())
    recorder.capture(econometrics_module.get_sign_bias_test(method="gjr"))


def test_get_pearson_goodness_of_fit_test(recorder, econometrics_module):
    normal = econometrics_module.get_pearson_goodness_of_fit_test()
    student = econometrics_module.get_pearson_goodness_of_fit_test(distribution="t")
    # Daily returns have fatter tails than normal innovations allow.
    assert (student.loc["Pearson Statistic"] < normal.loc["Pearson Statistic"]).all()
    recorder.capture(normal)
    recorder.capture(student)
    recorder.capture(econometrics_module.get_pearson_goodness_of_fit_test(bins=10))


def test_get_nyblom_test(recorder, econometrics_module):
    # The fit stops within its tolerance: a move of 1e-12 in the prices moves the statistics
    # by up to 1e-6, so they are recorded at two decimals.
    recorder.capture(econometrics_module.get_nyblom_test(rounding=2))


def test_get_gmm_moment_test(recorder, econometrics_module):
    # As for the Nyblom test: the Joint Statistic moves by up to 4e-6 with the fit.
    recorder.capture(econometrics_module.get_gmm_moment_test(rounding=2))
    recorder.capture(
        econometrics_module.get_gmm_moment_test(distribution="t", lags=2, rounding=2)
    )


def test_get_change_points(recorder, econometrics_module):
    recorder.capture(econometrics_module.get_change_points())
    recorder.capture(
        econometrics_module.get_change_points(period="weekly", cost="variance")
    )
    recorder.capture(
        econometrics_module.get_change_points(
            column="Adj Close", period="monthly", cost="mean"
        )
    )


def test_get_optimal_trading_rule(recorder, econometrics_module):
    # Monte Carlo results are recorded to two decimals: a path that lands on a threshold
    # after a last-bit difference on another platform moves a mean by at most
    # (profit-taking + stop-loss) / paths, and the simulation error is larger still.
    recorder.capture(
        econometrics_module.get_optimal_trading_rule(
            grid_size=5, paths=2000, rounding=2
        )
    )
    recorder.capture(
        econometrics_module.get_optimal_trading_rule(
            period="weekly",
            grid_size=3,
            maximum_holding_period=20,
            paths=1000,
            grid=True,
            rounding=2,
        )
    )
