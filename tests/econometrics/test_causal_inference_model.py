"""Causal Inference Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import causal_inference_model, regression_model

# pylint: disable=missing-function-docstring


# --- IV / 2SLS -----------------------------------------------------------------------


def test_get_iv_2sls_recovers_true_effect_unlike_naive_ols(recorder):
    # x is endogenous: driven by both the instrument and a confounder that also
    # affects y directly, so naive OLS of y on x is biased. The instrument is
    # correlated with x but (by construction) uncorrelated with the confounder, so
    # 2SLS should recover the true slope of 2.0 while naive OLS does not.
    rng = np.random.default_rng(42)
    n = 3000
    confounder = rng.standard_normal(n)
    instrument = pd.Series(rng.standard_normal(n), name="Instrument")
    x = pd.Series(
        0.8 * instrument.to_numpy() + 0.6 * confounder + rng.standard_normal(n) * 0.3,
        name="X",
    )
    true_slope = 2.0
    y = pd.Series(
        1.0
        + true_slope * x.to_numpy()
        + 0.9 * confounder
        + rng.standard_normal(n) * 0.3
    )

    naive_ols = regression_model.get_ols(y, x)
    iv_result = causal_inference_model.get_iv_2sls(y, x, instrument)

    naive_error = abs(naive_ols["coefficients"][1] - true_slope)
    iv_error = abs(iv_result["coefficients"][1] - true_slope)

    # The whole point of IV: naive OLS is biased away from the truth, 2SLS is not.
    assert naive_error > 0.1
    assert iv_error < 0.05
    assert iv_error < naive_error

    recorder.capture(regression_model.regression_summary_table(iv_result).round(4))


def test_get_iv_2sls_corrected_standard_errors_differ_from_naive_stage_two():
    # The naive "just run OLS on the Stage-1 fitted values and read off its own
    # standard errors" approach gives WRONG standard errors -- confirm our corrected
    # standard errors are materially different from that naive approach's.
    rng = np.random.default_rng(11)
    n = 1000
    confounder = rng.standard_normal(n)
    instrument = pd.Series(rng.standard_normal(n))
    x = pd.Series(
        0.8 * instrument.to_numpy() + 0.6 * confounder + rng.standard_normal(n) * 0.3
    )
    y = pd.Series(
        1.0 + 2.0 * x.to_numpy() + 0.9 * confounder + rng.standard_normal(n) * 0.3
    )

    first_stage = regression_model.get_ols(x, instrument)
    naive_second_stage = regression_model.get_ols(
        y, pd.Series(first_stage["fitted_values"])
    )

    corrected = causal_inference_model.get_iv_2sls(y, x, instrument)

    # Coefficients should match (the naive two-OLS-calls approach gets the point
    # estimate right)...
    assert np.isclose(
        corrected["coefficients"][1], naive_second_stage["coefficients"][1], atol=1e-6
    )
    # ...but the standard error should NOT match the naive (wrong) one -- the naive
    # approach both uses the wrong residuals (y - fitted(x_hat) instead of
    # y - fitted(x_actual)) and ignores the extra estimation uncertainty introduced by
    # Stage 1, so it is generically biased (in either direction, depending on the sign
    # of the correlation between the first-stage residuals and the structural error)
    # rather than simply "too small" or "too large".
    assert not np.isclose(
        corrected["standard_errors"][1],
        naive_second_stage["standard_errors"][1],
        rtol=0.05,
    )


def test_get_iv_2sls_with_exogenous_controls(recorder):
    rng = np.random.default_rng(13)
    n = 2000
    confounder = rng.standard_normal(n)
    instrument = pd.Series(rng.standard_normal(n), name="Instrument")
    control = pd.Series(rng.standard_normal(n), name="Control")
    x = pd.Series(
        0.8 * instrument.to_numpy() + 0.6 * confounder + rng.standard_normal(n) * 0.3,
        name="X",
    )
    y = pd.Series(
        1.0
        + 2.0 * x.to_numpy()
        - 0.5 * control.to_numpy()
        + 0.9 * confounder
        + rng.standard_normal(n) * 0.3
    )

    result = causal_inference_model.get_iv_2sls(y, x, instrument, x_exogenous=control)

    assert result["feature_names"] == ["Intercept", "Control", "X"]
    assert abs(result["coefficients"][-1] - 2.0) < 0.1
    assert abs(result["coefficients"][1] - (-0.5)) < 0.1

    recorder.capture(regression_model.regression_summary_table(result).round(4))


def test_get_iv_2sls_underidentified():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    instruments = pd.DataFrame(
        {"Z1": [1.0, 2.0, 3.0, 4.0, 5.0], "Z2": [2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    x_endog = pd.DataFrame({"X1": x, "X2": x * 2})
    try:
        causal_inference_model.get_iv_2sls(y, x_endog, instruments.iloc[:, :1])
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_iv_2sls_mismatched_lengths():
    y = pd.Series([1.0, 2.0, 3.0])
    x = pd.Series([1.0, 2.0])
    instrument = pd.Series([1.0, 2.0, 3.0])
    try:
        causal_inference_model.get_iv_2sls(y, x, instrument)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# --- Difference-in-Differences ---------------------------------------------------------


def test_get_difference_in_differences_recovers_treatment_effect(recorder):
    rng = np.random.default_rng(1)
    n_units, n_periods = 200, 2
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    treated = (unit % 2 == 0).astype(float)
    post = (period == 1).astype(float)

    true_effect = 3.0
    y = (
        1.0
        + 0.5 * treated
        + 0.2 * post
        + true_effect * treated * post
        + rng.standard_normal(n_units * n_periods) * 0.2
    )

    result = causal_inference_model.get_difference_in_differences(
        pd.Series(y), pd.Series(treated), pd.Series(post)
    )
    interaction_index = result["feature_names"].index("Treated x Post")

    assert abs(result["coefficients"][interaction_index] - true_effect) < 0.1
    assert result["p_values"][interaction_index] < 0.05

    recorder.capture(regression_model.regression_summary_table(result).round(4))


def test_get_difference_in_differences_placebo_test_is_insignificant():
    # Placebo: apply a FAKE treatment date that lies entirely within the pre-period
    # (before the true treatment ever occurs) -- since there is no real effect before
    # the true treatment date, the placebo DiD estimate should be small/insignificant.
    rng = np.random.default_rng(2)
    n_units, n_periods = 200, 4
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    treated = (unit % 2 == 0).astype(float)
    true_post = (period >= 2).astype(float)

    true_effect = 3.0
    y = (
        1.0
        + 0.5 * treated
        + 0.1 * period
        + true_effect * treated * true_post
        + rng.standard_normal(n_units * n_periods) * 0.2
    )

    # Restrict to the genuinely pre-treatment periods (0 and 1) and apply a placebo
    # "post" cutoff at period >= 1.
    pre_mask = period < 2
    placebo_post = (period >= 1).astype(float)

    placebo_result = causal_inference_model.get_difference_in_differences(
        pd.Series(y[pre_mask]),
        pd.Series(treated[pre_mask]),
        pd.Series(placebo_post[pre_mask]),
    )
    interaction_index = placebo_result["feature_names"].index("Treated x Post")

    assert abs(placebo_result["coefficients"][interaction_index]) < 0.3
    assert placebo_result["p_values"][interaction_index] > 0.05


def test_get_difference_in_differences_with_controls(recorder):
    rng = np.random.default_rng(3)
    n = 400
    treated = (rng.uniform(size=n) < 0.5).astype(float)
    post = (rng.uniform(size=n) < 0.5).astype(float)
    control = rng.standard_normal(n)
    true_effect = 1.5
    y = (
        1.0
        + 0.3 * treated
        + 0.2 * post
        + true_effect * treated * post
        + 0.7 * control
        + rng.standard_normal(n) * 0.1
    )

    result = causal_inference_model.get_difference_in_differences(
        pd.Series(y),
        pd.Series(treated),
        pd.Series(post),
        x_controls=pd.Series(control, name="Control"),
    )
    interaction_index = result["feature_names"].index("Treated x Post")

    assert abs(result["coefficients"][interaction_index] - true_effect) < 0.1
    recorder.capture(regression_model.regression_summary_table(result).round(4))


def test_get_difference_in_differences_invalid_treated():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    treated = pd.Series([0.0, 1.0, 2.0, 1.0])
    post = pd.Series([0.0, 0.0, 1.0, 1.0])
    try:
        causal_inference_model.get_difference_in_differences(y, treated, post)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_difference_in_differences_invalid_post():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    treated = pd.Series([0.0, 1.0, 0.0, 1.0])
    post = pd.Series([0.0, 0.0, 1.0, 5.0])
    try:
        causal_inference_model.get_difference_in_differences(y, treated, post)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_difference_in_differences_mismatched_lengths():
    y = pd.Series([1.0, 2.0, 3.0])
    treated = pd.Series([0.0, 1.0])
    post = pd.Series([0.0, 1.0, 1.0])
    try:
        causal_inference_model.get_difference_in_differences(y, treated, post)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# --- Regression Discontinuity -----------------------------------------------------------


def test_get_regression_discontinuity_recovers_known_jump(recorder):
    rng = np.random.default_rng(7)
    n = 2000
    running = pd.Series(rng.uniform(-10, 10, n), name="Running")
    true_jump = 4.0
    y = (
        1.0
        + 0.3 * running.to_numpy()
        + true_jump * (running.to_numpy() >= 0)
        + rng.standard_normal(n) * 0.5
    )

    result = causal_inference_model.get_regression_discontinuity(
        pd.Series(y), running, cutoff=0.0
    )

    assert abs(result["discontinuity"] - true_jump) < 0.2
    assert result["p_value"] < 0.05

    recorder.capture(
        causal_inference_model.regression_discontinuity_summary_table(result).round(4)
    )


def test_get_regression_discontinuity_no_jump_is_close_to_zero():
    rng = np.random.default_rng(8)
    n = 2000
    running = pd.Series(rng.uniform(-10, 10, n), name="Running")
    # Purely continuous underlying function -- no discontinuity anywhere.
    y = 1.0 + 0.3 * running.to_numpy() + rng.standard_normal(n) * 0.5

    result = causal_inference_model.get_regression_discontinuity(
        pd.Series(y), running, cutoff=0.0
    )

    assert abs(result["discontinuity"]) < 0.3
    assert result["p_value"] > 0.05


def test_get_regression_discontinuity_triangular_kernel_recovers_known_jump():
    rng = np.random.default_rng(9)
    n = 2000
    running = pd.Series(rng.uniform(-10, 10, n), name="Running")
    true_jump = 4.0
    y = (
        1.0
        + 0.3 * running.to_numpy()
        + true_jump * (running.to_numpy() >= 0)
        + rng.standard_normal(n) * 0.5
    )

    result = causal_inference_model.get_regression_discontinuity(
        pd.Series(y), running, cutoff=0.0, kernel="triangular"
    )

    assert abs(result["discontinuity"] - true_jump) < 0.3


def test_get_regression_discontinuity_invalid_kernel():
    y = pd.Series(np.arange(10.0))
    running = pd.Series(np.arange(10.0))
    try:
        causal_inference_model.get_regression_discontinuity(
            y, running, cutoff=5.0, kernel="gaussian"
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_regression_discontinuity_invalid_bandwidth():
    rng = np.random.default_rng(10)
    y = pd.Series(rng.standard_normal(50))
    running = pd.Series(rng.standard_normal(50))
    try:
        causal_inference_model.get_regression_discontinuity(
            y, running, cutoff=0.0, bandwidth=-1.0
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_regression_discontinuity_too_few_observations():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    running = pd.Series([-2.0, -1.0, 1.0, 2.0])
    try:
        causal_inference_model.get_regression_discontinuity(
            y, running, cutoff=0.0, bandwidth=0.5
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# --- Propensity Score Matching -----------------------------------------------------------


def test_get_propensity_score_matching_recovers_true_effect_unlike_naive(recorder):
    # Treatment assignment is correlated with the covariate (selection bias), and the
    # covariate ALSO drives the outcome directly -- so a naive mean difference between
    # treated and control is biased, but PSM (matching on similar propensity scores)
    # should recover the true, constant treatment effect of 2.0.
    rng = np.random.default_rng(3)
    n = 2000
    covariate = pd.Series(rng.standard_normal(n), name="Size")
    propensity = 1 / (1 + np.exp(-1.5 * covariate.to_numpy()))
    treatment = pd.Series((rng.uniform(size=n) < propensity).astype(float))
    true_effect = 2.0
    outcome = pd.Series(
        1.0
        + 3.0 * covariate.to_numpy()
        + true_effect * treatment.to_numpy()
        + rng.standard_normal(n) * 0.5
    )

    naive_diff = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    result = causal_inference_model.get_propensity_score_matching(
        treatment, outcome, covariate
    )

    naive_error = abs(naive_diff - true_effect)
    psm_error = abs(result["att"] - true_effect)

    # The whole point of PSM: the naive mean difference is badly biased, PSM is not.
    assert naive_error > 1.0
    assert psm_error < 0.3
    assert psm_error < naive_error
    assert result["n_matched_pairs"] > 0

    recorder.capture(
        causal_inference_model.propensity_score_matching_summary(result).round(4)
    )


def test_get_propensity_score_matching_invalid_treatment():
    treatment = pd.Series([0.0, 1.0, 2.0, 0.0])
    outcome = pd.Series([1.0, 2.0, 3.0, 4.0])
    covariates = pd.Series([1.0, 2.0, 3.0, 4.0])
    try:
        causal_inference_model.get_propensity_score_matching(
            treatment, outcome, covariates
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_propensity_score_matching_mismatched_lengths():
    treatment = pd.Series([0.0, 1.0, 0.0])
    outcome = pd.Series([1.0, 2.0])
    covariates = pd.Series([1.0, 2.0, 3.0])
    try:
        causal_inference_model.get_propensity_score_matching(
            treatment, outcome, covariates
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_propensity_score_matching_no_treated_or_control():
    treatment = pd.Series([0.0, 0.0, 0.0])
    outcome = pd.Series([1.0, 2.0, 3.0])
    covariates = pd.Series([1.0, 2.0, 3.0])
    try:
        causal_inference_model.get_propensity_score_matching(
            treatment, outcome, covariates
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_propensity_score_matching_infinite_caliper_matches_every_treated_unit():
    rng = np.random.default_rng(4)
    n = 200
    covariate = pd.Series(rng.standard_normal(n))
    treatment = pd.Series((rng.uniform(size=n) < 0.3).astype(float))
    outcome = pd.Series(rng.standard_normal(n))

    result = causal_inference_model.get_propensity_score_matching(
        treatment, outcome, covariate, caliper=np.inf
    )

    assert result["n_matched_pairs"] == result["n_treated"]


def _synthetic_control_setup(
    n_periods=40, treatment_period=30, true_effect=3.0, seed=1
):
    rng = np.random.default_rng(seed)
    common_trend = np.cumsum(rng.standard_normal(n_periods) * 0.5)
    donors = pd.DataFrame(
        {
            f"Donor_{i}": common_trend + rng.standard_normal(n_periods) * 0.3
            for i in range(5)
        }
    )
    treated_values = common_trend + rng.standard_normal(n_periods) * 0.3
    treated_values[treatment_period:] += true_effect
    treated = pd.Series(treated_values, name="Treated")
    return treated, donors


def test_get_synthetic_control_recovers_known_treatment_effect(recorder):
    treated, donors = _synthetic_control_setup(true_effect=3.0)
    result = causal_inference_model.get_synthetic_control(
        treated, donors, treatment_period=30
    )

    assert abs(result["average_treatment_effect"] - 3.0) < 1.0
    assert result["n_donors"] == 5
    assert result["n_pre_periods"] == 30
    assert result["n_post_periods"] == 10

    # Weights are a valid weighted average: non-negative and summing to 1.
    assert (result["weights"] >= -1e-6).all()
    assert abs(result["weights"].sum() - 1.0) < 1e-6

    recorder.capture(causal_inference_model.synthetic_control_summary(result).round(4))


def test_get_synthetic_control_weights_concentrate_on_close_match():
    # One donor is (deliberately) an almost-exact pre-treatment match for the
    # treated unit -- the fitted weights should concentrate heavily on it.
    rng = np.random.default_rng(2)
    n_periods, treatment_period = 30, 20
    base = np.cumsum(rng.standard_normal(n_periods) * 0.5)

    donors = pd.DataFrame(
        {
            "Close": base + rng.standard_normal(n_periods) * 0.01,
            "Far1": rng.standard_normal(n_periods) * 2,
            "Far2": rng.standard_normal(n_periods) * 2 + 5,
        }
    )
    treated = pd.Series(base, name="Treated")

    result = causal_inference_model.get_synthetic_control(
        treated, donors, treatment_period=treatment_period
    )

    assert result["weights"]["Close"] > 0.95
    assert result["pre_treatment_rmspe"] < 0.1


def test_get_synthetic_control_no_effect_gives_small_ratio():
    # No treatment effect at all -- the post-treatment gap should look like the
    # pre-treatment gap, so the RMSPE ratio should be close to 1, not large.
    rng = np.random.default_rng(5)
    n_periods, treatment_period = 40, 30
    common_trend = np.cumsum(rng.standard_normal(n_periods) * 0.5)
    donors = pd.DataFrame(
        {
            f"Donor_{i}": common_trend + rng.standard_normal(n_periods) * 0.3
            for i in range(5)
        }
    )
    treated = pd.Series(
        common_trend + rng.standard_normal(n_periods) * 0.3, name="Treated"
    )

    result = causal_inference_model.get_synthetic_control(
        treated, donors, treatment_period=treatment_period
    )

    assert result["rmspe_ratio"] < 3.0


def test_get_synthetic_control_p_value_bounded():
    treated, donors = _synthetic_control_setup()
    result = causal_inference_model.get_synthetic_control(
        treated, donors, treatment_period=30
    )

    assert 0.0 <= result["p_value"] <= 1.0
    assert len(result["placebo_ratios"]) == 5


def test_get_synthetic_control_invalid_treated_type():
    _, donors = _synthetic_control_setup()
    try:
        causal_inference_model.get_synthetic_control(
            [1, 2, 3], donors, treatment_period=30
        )
        raise AssertionError("Expected a TypeError to be raised.")
    except TypeError:
        pass


def test_get_synthetic_control_invalid_donors_type():
    treated, _ = _synthetic_control_setup()
    try:
        causal_inference_model.get_synthetic_control(
            treated, [1, 2, 3], treatment_period=30
        )
        raise AssertionError("Expected a TypeError to be raised.")
    except TypeError:
        pass


def test_get_synthetic_control_too_few_donors():
    treated, donors = _synthetic_control_setup()
    try:
        causal_inference_model.get_synthetic_control(
            treated, donors[["Donor_0"]], treatment_period=30
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_synthetic_control_too_few_pre_periods():
    treated, donors = _synthetic_control_setup()
    try:
        causal_inference_model.get_synthetic_control(
            treated, donors, treatment_period=1
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_synthetic_control_too_few_post_periods():
    treated, donors = _synthetic_control_setup(n_periods=40)
    try:
        causal_inference_model.get_synthetic_control(
            treated, donors, treatment_period=40
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_synthetic_control_no_overlapping_periods():
    rng = np.random.default_rng(1)
    treated = pd.Series(
        rng.standard_normal(30), index=pd.date_range("2020-01-01", periods=30)
    )
    donors = pd.DataFrame(
        rng.standard_normal((30, 3)),
        index=pd.date_range("2030-01-01", periods=30),
        columns=["A", "B", "C"],
    )
    try:
        causal_inference_model.get_synthetic_control(
            treated, donors, treatment_period=pd.Timestamp("2020-06-01")
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass
