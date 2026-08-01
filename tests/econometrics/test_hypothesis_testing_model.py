"""Hypothesis Testing Model Tests"""

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.econometrics import hypothesis_testing_model, regression_model

# pylint: disable=missing-function-docstring


# ---------------------------------------------------------------------------
# get_two_sample_t_test
# ---------------------------------------------------------------------------


def test_get_two_sample_t_test_detects_known_large_difference(recorder):
    # Two samples with a known, large true mean difference (5 vs 0, unit variance)
    # should be flagged as significantly different, and should exactly match scipy.
    rng = np.random.default_rng(0)
    sample_a = pd.Series(rng.normal(loc=5.0, scale=1.0, size=200))
    sample_b = pd.Series(rng.normal(loc=0.0, scale=1.0, size=200))

    result = hypothesis_testing_model.get_two_sample_t_test(sample_a, sample_b)
    expected = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    assert np.isclose(result["T-Statistic"], expected.statistic)
    assert np.isclose(result["P-Value"], expected.pvalue)
    assert np.isclose(result["Degrees of Freedom"], expected.df)
    assert result["P-Value"] < 0.001

    recorder.capture(result.round(4))


def test_get_two_sample_t_test_no_true_difference():
    # Two samples drawn from the identical distribution should not be flagged as
    # significantly different.
    rng = np.random.default_rng(1)
    sample_a = pd.Series(rng.normal(loc=0.0, scale=1.0, size=200))
    sample_b = pd.Series(rng.normal(loc=0.0, scale=1.0, size=200))

    result = hypothesis_testing_model.get_two_sample_t_test(sample_a, sample_b)

    assert result["P-Value"] > 0.05


def test_get_two_sample_t_test_equal_variance_matches_scipy_pooled():
    rng = np.random.default_rng(2)
    sample_a = rng.normal(loc=5.0, scale=1.0, size=150)
    sample_b = rng.normal(loc=0.0, scale=1.0, size=150)

    result = hypothesis_testing_model.get_two_sample_t_test(
        sample_a, sample_b, equal_variance=True
    )
    expected = stats.ttest_ind(sample_a, sample_b, equal_var=True)

    assert np.isclose(result["T-Statistic"], expected.statistic)
    assert np.isclose(result["P-Value"], expected.pvalue)
    # Pooled (Student's) degrees of freedom is exactly n_a + n_b - 2.
    assert result["Degrees of Freedom"] == len(sample_a) + len(sample_b) - 2


def test_get_two_sample_t_test_accepts_numpy_arrays():
    rng = np.random.default_rng(3)
    sample_a = rng.standard_normal(50)
    sample_b = rng.standard_normal(50) + 3

    result = hypothesis_testing_model.get_two_sample_t_test(sample_a, sample_b)

    assert result["P-Value"] < 0.05


# ---------------------------------------------------------------------------
# get_f_test / get_likelihood_ratio_test -- shared synthetic nested-model setup
# ---------------------------------------------------------------------------


def _fit_nested_models(y: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
    restricted = regression_model.get_ols(y, pd.DataFrame({"x1": x1}))
    unrestricted = regression_model.get_ols(
        y, pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    )
    return restricted, unrestricted


def test_get_f_test_fails_to_reject_when_coefficients_are_truly_zero(recorder):
    # x2, x3 are pure noise regressors -- the true model only depends on x1, so the
    # F-test should NOT reject the (correct) restriction that x2, x3 are jointly zero.
    rng = np.random.default_rng(4)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1 + 2 * x1 + rng.standard_normal(n) * 0.5

    restricted, unrestricted = _fit_nested_models(y, x1, x2, x3)
    result = hypothesis_testing_model.get_f_test(restricted, unrestricted)

    assert not result["Reject Restrictions (5%)"]
    recorder.capture(result.round(4))


def test_get_f_test_rejects_when_coefficients_are_truly_nonzero():
    # x2, x3 now have large true coefficients -- the F-test should reject.
    rng = np.random.default_rng(5)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1 + 2 * x1 + 5 * x2 - 4 * x3 + rng.standard_normal(n) * 0.5

    restricted, unrestricted = _fit_nested_models(y, x1, x2, x3)
    result = hypothesis_testing_model.get_f_test(restricted, unrestricted)

    assert result["Reject Restrictions (5%)"]
    assert result["P-Value"] < 0.001


def test_get_f_test_invalid_nesting_direction():
    rng = np.random.default_rng(6)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1 + x1 + rng.standard_normal(n) * 0.1

    smaller = regression_model.get_ols(y, pd.Series(x1, name="x1"))
    larger = regression_model.get_ols(y, pd.DataFrame({"x1": x1, "x2": x2}))
    try:
        # Swapped -- "restricted" has more parameters than "unrestricted".
        hypothesis_testing_model.get_f_test(larger, smaller)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_f_test_mismatched_observations():
    rng = np.random.default_rng(7)
    small_result = regression_model.get_ols(
        rng.standard_normal(50), pd.Series(rng.standard_normal(50))
    )
    large_result = regression_model.get_ols(
        rng.standard_normal(100),
        pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100)}),
    )
    try:
        hypothesis_testing_model.get_f_test(small_result, large_result)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_likelihood_ratio_test_fails_to_reject_when_coefficients_are_truly_zero(
    recorder,
):
    rng = np.random.default_rng(4)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1 + 2 * x1 + rng.standard_normal(n) * 0.5

    restricted, unrestricted = _fit_nested_models(y, x1, x2, x3)
    result = hypothesis_testing_model.get_likelihood_ratio_test(
        restricted, unrestricted
    )

    assert not result["Reject Restrictions (5%)"]
    recorder.capture(result.round(4))


def test_get_likelihood_ratio_test_rejects_when_coefficients_are_truly_nonzero():
    rng = np.random.default_rng(5)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1 + 2 * x1 + 5 * x2 - 4 * x3 + rng.standard_normal(n) * 0.5

    restricted, unrestricted = _fit_nested_models(y, x1, x2, x3)
    result = hypothesis_testing_model.get_likelihood_ratio_test(
        restricted, unrestricted
    )

    assert result["Reject Restrictions (5%)"]
    assert result["P-Value"] < 0.001


def test_get_likelihood_ratio_test_agrees_with_f_test_direction():
    # For large samples the F-test and LR-test should agree on the decision, even
    # though their exact statistics/distributions differ.
    rng = np.random.default_rng(8)
    n = 2000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1 + 2 * x1 + 3 * x2 + rng.standard_normal(n) * 0.5

    restricted, unrestricted = _fit_nested_models(y, x1, x2, x3)
    f_result = hypothesis_testing_model.get_f_test(restricted, unrestricted)
    lr_result = hypothesis_testing_model.get_likelihood_ratio_test(
        restricted, unrestricted
    )

    assert f_result["Reject Restrictions (5%)"] == lr_result["Reject Restrictions (5%)"]


def test_get_likelihood_ratio_test_invalid_nesting_direction():
    rng = np.random.default_rng(6)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1 + x1 + rng.standard_normal(n) * 0.1

    smaller = regression_model.get_ols(y, pd.Series(x1, name="x1"))
    larger = regression_model.get_ols(y, pd.DataFrame({"x1": x1, "x2": x2}))
    try:
        hypothesis_testing_model.get_likelihood_ratio_test(larger, smaller)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# get_wald_test
# ---------------------------------------------------------------------------


def test_get_wald_test_single_restriction_equals_t_squared_exactly():
    # The core algebraic identity: a chi-sq(1) Wald statistic for a single
    # coefficient restriction must exactly equal that coefficient's own squared
    # t-statistic, since both use the identical Cov(beta_hat) estimate.
    rng = np.random.default_rng(9)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1 + 2 * x1 + 5 * x2 + rng.standard_normal(n) * 0.5

    result = regression_model.get_ols(y, pd.DataFrame({"x1": x1, "x2": x2}))
    # Coefficients are [Intercept, x1, x2] -- pick out x2 (index 2).
    restriction_matrix = np.array([[0.0, 0.0, 1.0]])

    wald_result = hypothesis_testing_model.get_wald_test(result, restriction_matrix)

    t_statistic = result["t_statistics"][2]
    p_value = result["p_values"][2]

    assert np.isclose(
        wald_result["Wald Statistic (Chi2)"], t_statistic**2, atol=1e-10, rtol=1e-10
    )
    assert np.isclose(
        wald_result["F-Statistic"], t_statistic**2, atol=1e-10, rtol=1e-10
    )
    assert np.isclose(wald_result["F P-Value"], p_value, atol=1e-12, rtol=1e-10)


def test_get_wald_test_fails_to_reject_when_coefficients_are_truly_zero(recorder):
    rng = np.random.default_rng(4)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1 + 2 * x1 + rng.standard_normal(n) * 0.5

    result = regression_model.get_ols(y, pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}))
    restriction_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    wald_result = hypothesis_testing_model.get_wald_test(result, restriction_matrix)

    assert not wald_result["Reject Restrictions (5%)"]
    recorder.capture(wald_result.round(4))


def test_get_wald_test_rejects_when_coefficients_are_truly_nonzero():
    rng = np.random.default_rng(5)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    y = 1 + 2 * x1 + 5 * x2 - 4 * x3 + rng.standard_normal(n) * 0.5

    result = regression_model.get_ols(y, pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}))
    restriction_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    wald_result = hypothesis_testing_model.get_wald_test(result, restriction_matrix)

    assert wald_result["Reject Restrictions (5%)"]


def test_get_wald_test_custom_restriction_value():
    # H0: coefficient on x1 equals its true value (2.0) -- should not reject.
    rng = np.random.default_rng(10)
    n = 1000
    x1 = rng.standard_normal(n)
    y = 1 + 2 * x1 + rng.standard_normal(n) * 0.3

    result = regression_model.get_ols(y, pd.Series(x1, name="x1"))
    restriction_matrix = np.array([[0.0, 1.0]])

    true_value_result = hypothesis_testing_model.get_wald_test(
        result, restriction_matrix, restriction_values=np.array([2.0])
    )
    false_value_result = hypothesis_testing_model.get_wald_test(
        result, restriction_matrix, restriction_values=np.array([100.0])
    )

    assert not true_value_result["Reject Restrictions (5%)"]
    assert false_value_result["Reject Restrictions (5%)"]


def test_get_wald_test_equality_restriction_between_two_coefficients():
    # H0: coefficient on x2 equals coefficient on x3 -- true here (both = 3), should
    # not reject; a second case with hugely different true coefficients should reject.
    rng = np.random.default_rng(11)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    restriction_matrix = np.array([[0, 0, 1, -1]], dtype=float)

    y_equal = 1 + 2 * x1 + 3 * x2 + 3 * x3 + rng.standard_normal(n) * 0.5
    result_equal = regression_model.get_ols(
        y_equal, pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    )
    equal_result = hypothesis_testing_model.get_wald_test(
        result_equal, restriction_matrix
    )
    assert not equal_result["Reject Restrictions (5%)"]

    y_unequal = 1 + 2 * x1 + 10 * x2 - 10 * x3 + rng.standard_normal(n) * 0.5
    result_unequal = regression_model.get_ols(
        y_unequal, pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    )
    unequal_result = hypothesis_testing_model.get_wald_test(
        result_unequal, restriction_matrix
    )
    assert unequal_result["Reject Restrictions (5%)"]


def test_get_wald_test_invalid_restriction_matrix_shape():
    rng = np.random.default_rng(12)
    result = regression_model.get_ols(
        rng.standard_normal(100), pd.Series(rng.standard_normal(100))
    )
    try:
        # Model has 2 coefficients (Intercept, X1); restriction_matrix has 3 columns.
        hypothesis_testing_model.get_wald_test(result, np.array([[1.0, 0.0, 0.0]]))
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_wald_test_invalid_restriction_values_length():
    rng = np.random.default_rng(13)
    result = regression_model.get_ols(
        rng.standard_normal(100),
        pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100)}),
    )
    try:
        hypothesis_testing_model.get_wald_test(
            result,
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            restriction_values=np.array([0.0]),
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# get_hausman_wu_test
# ---------------------------------------------------------------------------


def _endogenous_setup(rng, n=1000):
    # A textbook endogeneity setup: an instrument z drives x, an unobserved
    # confounder u drives both x and y's error -- x is therefore correlated with
    # y's error term (endogenous), while z is not (a valid instrument).
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x = 0.8 * z + 0.6 * u + rng.standard_normal(n) * 0.3
    y = 2 + 3 * x + 2 * u + rng.standard_normal(n) * 0.5
    return y, x, z, u


def test_get_hausman_wu_test_flags_endogeneity_when_present(recorder):
    rng = np.random.default_rng(42)
    y, x, z, _ = _endogenous_setup(rng)

    result = hypothesis_testing_model.get_hausman_wu_test(
        pd.Series(y), pd.Series(x, name="X"), pd.Series(z, name="Z")
    )

    assert result["Endogenous (5%)"]
    assert result["P-Value"] < 0.05
    recorder.capture(result.round(4))


def test_get_hausman_wu_test_does_not_flag_genuinely_exogenous_x():
    rng = np.random.default_rng(43)
    n = 1000
    z = rng.standard_normal(n)
    # x depends only on z and independent noise -- no shared confounder with y.
    x = 0.8 * z + rng.standard_normal(n) * 0.3
    y = 2 + 3 * x + rng.standard_normal(n) * 0.5

    result = hypothesis_testing_model.get_hausman_wu_test(
        pd.Series(y), pd.Series(x, name="X"), pd.Series(z, name="Z")
    )

    assert not result["Endogenous (5%)"]
    assert result["P-Value"] > 0.05


def test_get_hausman_wu_test_matches_manual_two_step_ols():
    rng = np.random.default_rng(44)
    y, x, z, _ = _endogenous_setup(rng)

    result = hypothesis_testing_model.get_hausman_wu_test(
        pd.Series(y), pd.Series(x, name="X"), pd.Series(z, name="Z")
    )

    # Manual replication of the exact same two-step procedure using get_ols directly.
    stage_one = regression_model.get_ols(x, pd.Series(z, name="Z"))
    v_hat = stage_one["residuals"]
    stage_two = regression_model.get_ols(y, pd.DataFrame({"X": x, "Residual": v_hat}))
    residual_index = stage_two["feature_names"].index("Residual")

    assert np.isclose(result["T-Statistic"], stage_two["t_statistics"][residual_index])
    assert np.isclose(result["P-Value"], stage_two["p_values"][residual_index])


def test_get_hausman_wu_test_with_additional_exogenous_regressor():
    rng = np.random.default_rng(45)
    n = 1000
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    w = rng.standard_normal(n)
    x = 0.8 * z + 0.6 * u + rng.standard_normal(n) * 0.3
    y = 2 + 3 * x + 1.5 * w + 2 * u + rng.standard_normal(n) * 0.5

    result = hypothesis_testing_model.get_hausman_wu_test(
        pd.Series(y),
        pd.Series(x, name="X"),
        pd.Series(z, name="Z"),
        x_other=pd.Series(w, name="W"),
    )

    assert result["Endogenous (5%)"]
