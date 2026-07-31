"""Specification Tests Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import regression_model, specification_tests_model

# pylint: disable=missing-function-docstring


def test_get_breusch_pagan_test_detects_heteroskedasticity(recorder):
    # Error variance grows with x -- textbook heteroskedasticity.
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(1, 10, n)
    y = 1 + 2 * x + rng.standard_normal(n) * x
    result = regression_model.get_ols(y, x)

    bp_result = specification_tests_model.get_breusch_pagan_test(result)

    assert bp_result["P-Value"] < 0.05
    assert bp_result["Reject Homoskedasticity (5%)"]
    recorder.capture(bp_result.round(4))


def test_get_breusch_pagan_test_homoskedastic_not_flagged(recorder):
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1 + 2 * x1 - 0.5 * x2 + rng.standard_normal(n)
    result = regression_model.get_ols(y, pd.DataFrame({"A": x1, "B": x2}))

    bp_result = specification_tests_model.get_breusch_pagan_test(result)

    assert bp_result["P-Value"] > 0.05
    assert not bp_result["Reject Homoskedasticity (5%)"]
    recorder.capture(bp_result.round(4))


def test_get_white_test_detects_heteroskedasticity(recorder):
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(1, 10, n)
    x2 = rng.standard_normal(n)
    y = 1 + 2 * x1 - 0.5 * x2 + rng.standard_normal(n) * x1
    result = regression_model.get_ols(y, pd.DataFrame({"A": x1, "B": x2}))

    white_result = specification_tests_model.get_white_test(result)

    assert white_result["P-Value"] < 0.05
    assert white_result["Reject Homoskedasticity (5%)"]
    recorder.capture(white_result.round(4))


def test_get_white_test_homoskedastic_not_flagged(recorder):
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1 + 2 * x1 - 0.5 * x2 + rng.standard_normal(n)
    result = regression_model.get_ols(y, pd.DataFrame({"A": x1, "B": x2}))

    white_result = specification_tests_model.get_white_test(result)

    assert white_result["P-Value"] > 0.05
    assert not white_result["Reject Homoskedasticity (5%)"]
    recorder.capture(white_result.round(4))


def test_get_white_test_detects_nonlinear_heteroskedasticity_missed_by_bp():
    # Variance depends on x^2 (a curved, non-linear pattern) -- White's polynomial
    # expansion should catch this more decisively than Breusch-Pagan's linear-only
    # auxiliary regression.
    rng = np.random.default_rng(11)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1 + 0.5 * x1 + 0.5 * x2 + rng.standard_normal(n) * (1 + 5 * x1**2)
    result = regression_model.get_ols(y, pd.DataFrame({"A": x1, "B": x2}))

    bp_result = specification_tests_model.get_breusch_pagan_test(result)
    white_result = specification_tests_model.get_white_test(result)

    assert white_result["Reject Homoskedasticity (5%)"]
    # White's p-value should be far more decisive than Breusch-Pagan's here.
    assert white_result["P-Value"] < bp_result["P-Value"]


def test_get_durbin_watson_test_no_autocorrelation(recorder):
    rng = np.random.default_rng(7)
    n = 300
    x = rng.standard_normal(n)
    y = 1 + 2 * x + rng.standard_normal(n) * 0.5
    result = regression_model.get_ols(y, x)

    dw_result = specification_tests_model.get_durbin_watson_test(result)

    assert 1.5 <= dw_result["Durbin-Watson Statistic"] <= 2.5
    assert dw_result["Interpretation"] == "No Strong Evidence"
    recorder.capture(dw_result)


def test_get_durbin_watson_test_detects_positive_autocorrelation(recorder):
    # AR(1) errors with a high persistence coefficient -- classic positive
    # autocorrelation, DW should fall well below 2.
    rng = np.random.default_rng(7)
    n = 300
    x = rng.standard_normal(n)
    ar_errors = np.zeros(n)
    innovations = rng.standard_normal(n) * 0.3
    for t in range(1, n):
        ar_errors[t] = 0.9 * ar_errors[t - 1] + innovations[t]
    y = 1 + 2 * x + ar_errors
    result = regression_model.get_ols(y, x)

    dw_result = specification_tests_model.get_durbin_watson_test(result)

    assert dw_result["Durbin-Watson Statistic"] < 1.5
    assert dw_result["Interpretation"] == "Positive Autocorrelation Likely"
    recorder.capture(dw_result)


def test_get_durbin_watson_test_detects_negative_autocorrelation(recorder):
    # Alternating-sign errors -- classic negative autocorrelation, DW should
    # fall well above 2.
    rng = np.random.default_rng(7)
    n = 300
    x = rng.standard_normal(n)
    alt_errors = (
        np.array([((-1) ** t) for t in range(n)]) + rng.standard_normal(n) * 0.1
    )
    y = 1 + 2 * x + alt_errors
    result = regression_model.get_ols(y, x)

    dw_result = specification_tests_model.get_durbin_watson_test(result)

    assert dw_result["Durbin-Watson Statistic"] > 2.5
    assert dw_result["Interpretation"] == "Negative Autocorrelation Likely"
    recorder.capture(dw_result)


def test_get_vif_low_for_independent_regressors(recorder):
    rng = np.random.default_rng(7)
    n = 300
    x = pd.DataFrame(
        {
            "A": rng.standard_normal(n),
            "B": rng.standard_normal(n),
            "C": rng.standard_normal(n),
        }
    )

    vif_result = specification_tests_model.get_vif(x)

    assert (vif_result < 5).all()
    recorder.capture(vif_result.round(4))


def test_get_vif_detects_collinearity(recorder):
    rng = np.random.default_rng(7)
    n = 300
    a = rng.standard_normal(n)
    x = pd.DataFrame(
        {
            "A": a,
            "B": 2 * a + rng.standard_normal(n) * 0.01,  # near-perfectly collinear
            "C": rng.standard_normal(n),
        }
    )

    vif_result = specification_tests_model.get_vif(x)

    assert vif_result["A"] > 10
    assert vif_result["B"] > 10
    assert vif_result["C"] < 5
    recorder.capture(vif_result.round(4))


def test_get_vif_excludes_intercept_column():
    rng = np.random.default_rng(7)
    n = 300
    x = pd.DataFrame(
        {
            "Intercept": 1.0,
            "A": rng.standard_normal(n),
            "B": rng.standard_normal(n),
        }
    )

    vif_result = specification_tests_model.get_vif(x)

    assert "Intercept" not in vif_result.index
    assert list(vif_result.index) == ["A", "B"]


def test_get_vif_invalid_type():
    try:
        specification_tests_model.get_vif([1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_ramsey_reset_test_linear_not_flagged(recorder):
    rng = np.random.default_rng(11)
    n = 400
    x = rng.standard_normal(n)
    y = 1 + 2 * x + rng.standard_normal(n) * 0.5
    result = regression_model.get_ols(y, x)

    reset_result = specification_tests_model.get_ramsey_reset_test(result)

    assert reset_result["P-Value"] > 0.05
    assert not reset_result["Reject Correct Specification (5%)"]
    recorder.capture(reset_result.round(4))


def test_get_ramsey_reset_test_detects_misspecification(recorder):
    # True relationship is quadratic; fit with a linear model -- RESET should
    # pick up the missing non-linear term via the fitted-value powers.
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(-3, 3, n)
    y = 1 + 2 * x + 1.5 * x**2 + rng.standard_normal(n) * 0.5
    result = regression_model.get_ols(y, x)

    reset_result = specification_tests_model.get_ramsey_reset_test(result)

    assert reset_result["P-Value"] < 0.05
    assert reset_result["Reject Correct Specification (5%)"]
    recorder.capture(reset_result.round(4))


def test_get_ramsey_reset_test_configurable_power():
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(-3, 3, n)
    y = 1 + 2 * x + 1.5 * x**2 + rng.standard_normal(n) * 0.5
    result = regression_model.get_ols(y, x)

    reset_power_2 = specification_tests_model.get_ramsey_reset_test(result, power=2)
    reset_power_3 = specification_tests_model.get_ramsey_reset_test(result, power=3)

    assert reset_power_2["RESET F-Statistic"] != reset_power_3["RESET F-Statistic"]


def test_get_ramsey_reset_test_invalid_power():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(50)
    y = 1 + 2 * x + rng.standard_normal(50) * 0.5
    result = regression_model.get_ols(y, x)

    try:
        specification_tests_model.get_ramsey_reset_test(result, power=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_chow_test_no_structural_break(recorder):
    rng = np.random.default_rng(21)
    n = 400
    x = rng.standard_normal(n)
    y = 1 + 2 * x + rng.standard_normal(n) * 0.5
    result_full = regression_model.get_ols(y, x)

    chow_result = specification_tests_model.get_chow_test(
        result_full, x, y, break_index=n // 2
    )

    assert chow_result["P-Value"] > 0.05
    assert not chow_result["Reject No Structural Break (5%)"]
    recorder.capture(chow_result.round(4))


def test_get_chow_test_detects_structural_break(recorder):
    # The coefficients genuinely differ before/after the midpoint.
    rng = np.random.default_rng(21)
    n = 400
    half = n // 2
    x = rng.standard_normal(n)
    y_before = 1 + 2 * x[:half] + rng.standard_normal(half) * 0.3
    y_after = -3 - 5 * x[half:] + rng.standard_normal(n - half) * 0.3
    y = np.concatenate([y_before, y_after])
    result_full = regression_model.get_ols(y, x)

    chow_result = specification_tests_model.get_chow_test(
        result_full, x, y, break_index=half
    )

    assert chow_result["P-Value"] < 0.05
    assert chow_result["Reject No Structural Break (5%)"]
    recorder.capture(chow_result.round(4))


def test_get_chow_test_break_index_out_of_range():
    rng = np.random.default_rng(21)
    x = rng.standard_normal(50)
    y = 1 + 2 * x + rng.standard_normal(50) * 0.5
    result_full = regression_model.get_ols(y, x)

    try:
        specification_tests_model.get_chow_test(result_full, x, y, break_index=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    try:
        specification_tests_model.get_chow_test(result_full, x, y, break_index=50)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_chow_test_insufficient_observations_in_subsample():
    rng = np.random.default_rng(21)
    x = rng.standard_normal(20)
    y = 1 + 2 * x + rng.standard_normal(20) * 0.5
    result_full = regression_model.get_ols(y, x)

    try:
        specification_tests_model.get_chow_test(result_full, x, y, break_index=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
