"""Regression Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import regression_model

# pylint: disable=missing-function-docstring


def test_get_ols_recovers_known_coefficients(recorder):
    # y = 2 + 3*x + noise -- OLS on a large sample should recover ~[2, 3].
    rng = np.random.default_rng(1)
    x = pd.Series(rng.standard_normal(1000), name="X")
    y = 2 + 3 * x + rng.standard_normal(1000) * 0.1
    result = regression_model.get_ols(y, x)

    assert abs(result["coefficients"][0] - 2) < 0.05
    assert abs(result["coefficients"][1] - 3) < 0.05
    assert result["r_squared"] > 0.99

    recorder.capture(regression_model.regression_summary_table(result).round(4))


def test_get_ols_matches_hand_solved_two_point_case():
    # y = 1 + 2x through three collinear points, so the fit is perfect and R^2 = 1.
    x = pd.Series([0.0, 1.0, 2.0])
    y = pd.Series([1.0, 3.0, 5.0])
    result = regression_model.get_ols(y, x)

    assert np.allclose(result["coefficients"], [1.0, 2.0], atol=1e-8)
    assert abs(result["r_squared"] - 1.0) < 1e-8
    assert np.allclose(result["residuals"], 0, atol=1e-8)


def test_get_ols_no_constant():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    result = regression_model.get_ols(y, x, add_constant=False)

    assert result["feature_names"] == ["X1"]
    assert abs(result["coefficients"][0] - 2.0) < 1e-8


def test_get_ols_multivariate(recorder):
    rng = np.random.default_rng(2)
    x = pd.DataFrame({"A": rng.standard_normal(500), "B": rng.standard_normal(500)})
    y = 1 - 2 * x["A"] + 0.5 * x["B"] + rng.standard_normal(500) * 0.05
    result = regression_model.get_ols(y, x)

    assert np.allclose(result["coefficients"], [1, -2, 0.5], atol=0.05)
    recorder.capture(regression_model.regression_summary_table(result).round(4))


def test_get_ols_invalid_type():
    try:
        regression_model.get_ols([1, 2, 3], [1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_ols_rank_deficient():
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [2.0, 4.0, 6.0, 8.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    try:
        regression_model.get_ols(y, x)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_ols_too_few_observations():
    x = pd.Series([1.0, 2.0])
    y = pd.Series([1.0, 2.0])
    try:
        regression_model.get_ols(y, x)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_ols_hac_matches_nonrobust_coefficients():
    # HAC reweights only the standard errors, so the coefficients must be identical.
    rng = np.random.default_rng(6)
    x = pd.Series(rng.standard_normal(200), name="X")
    y = 1 + 2 * x + rng.standard_normal(200) * 0.1

    nonrobust_result = regression_model.get_ols(y, x)
    hac_result = regression_model.get_ols(y, x, cov_type="HAC", maxlags=4)

    assert np.allclose(
        nonrobust_result["coefficients"], hac_result["coefficients"], atol=1e-8
    )
    assert hac_result["cov_type"] == "HAC"


def test_get_ols_hac_reweights_standard_errors():
    # AR(1) errors with rho near 1, so HAC and nonrobust must disagree on the errors.
    rng = np.random.default_rng(7)
    n = 300
    errors = np.zeros(n)
    for i in range(1, n):
        errors[i] = 0.9 * errors[i - 1] + rng.standard_normal()

    x = pd.Series(rng.standard_normal(n), name="X")
    y = 1 + 0.5 * x + errors

    nonrobust_result = regression_model.get_ols(y, x)
    hac_result = regression_model.get_ols(y, x, cov_type="HAC", maxlags=10)

    assert not np.allclose(
        hac_result["standard_errors"], nonrobust_result["standard_errors"]
    )


def test_get_ols_hac_missing_maxlags():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    try:
        regression_model.get_ols(y, x, cov_type="HAC")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_ols_hac_negative_maxlags():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    try:
        regression_model.get_ols(y, x, cov_type="HAC", maxlags=-1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_wls_hac_matches_nonrobust_coefficients():
    rng = np.random.default_rng(8)
    x = pd.Series(rng.standard_normal(200), name="X")
    y = 1 + 2 * x + rng.standard_normal(200) * 0.1
    weights = pd.Series(np.ones(200))

    nonrobust_result = regression_model.get_wls(y, x, weights)
    hac_result = regression_model.get_wls(y, x, weights, cov_type="HAC", maxlags=4)

    assert np.allclose(
        nonrobust_result["coefficients"], hac_result["coefficients"], atol=1e-8
    )
    assert hac_result["cov_type"] == "HAC"


def test_get_wls_downweights_noisy_observations():
    # Weighting the precise group heavily should pull the fit toward its relationship.
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 15.0, -5.0, 20.0, -10.0])
    weights = pd.Series([100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0])

    result = regression_model.get_wls(y, x, weights)
    ols_result = regression_model.get_ols(y, x)

    # WLS should be much closer to the true slope (1.0) than unweighted OLS.
    assert abs(result["coefficients"][1] - 1.0) < abs(
        ols_result["coefficients"][1] - 1.0
    )


def test_get_wls_invalid_weights():
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([1.0, 2.0, 3.0])
    weights = pd.Series([1.0, -1.0, 1.0])
    try:
        regression_model.get_wls(y, x, weights)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_gls_identity_omega_matches_ols():
    rng = np.random.default_rng(3)
    n = 200
    x = pd.Series(rng.standard_normal(n))
    y = 1 + 2 * x + rng.standard_normal(n) * 0.2

    ols_result = regression_model.get_ols(y, x)
    gls_result = regression_model.get_gls(y, x, np.eye(n))

    assert np.allclose(
        ols_result["coefficients"], gls_result["coefficients"], atol=1e-6
    )


def test_get_gls_invalid_omega_shape():
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([1.0, 2.0, 3.0])
    try:
        regression_model.get_gls(y, x, np.eye(2))
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_gls_non_positive_definite_omega():
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([1.0, 2.0, 3.0])
    bad_omega = np.zeros((3, 3))
    try:
        regression_model.get_gls(y, x, bad_omega)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_logistic_regression_separates_classes(recorder):
    rng = np.random.default_rng(4)
    n = 1000
    x = pd.Series(rng.standard_normal(n), name="X")
    linear = 3 * x
    prob = 1 / (1 + np.exp(-linear))
    y = pd.Series((rng.uniform(size=n) < prob).astype(float))

    result = regression_model.get_logistic_regression(y, x)

    # Recovered slope should be close to the true value of 3, and positive.
    assert result["coefficients"][1] > 2.0
    assert result["converged"]
    recorder.capture(regression_model.binary_regression_summary_table(result).round(4))


def test_get_logistic_regression_invalid_y():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([0.0, 1.0, 2.0, 1.0])
    try:
        regression_model.get_logistic_regression(y, x)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_probit_regression_separates_classes(recorder):
    rng = np.random.default_rng(5)
    n = 1000
    x = pd.Series(rng.standard_normal(n), name="X")
    linear = 2 * x
    from scipy import stats as spstats

    prob = spstats.norm.cdf(linear)
    y = pd.Series((rng.uniform(size=n) < prob).astype(float))

    result = regression_model.get_probit_regression(y, x)

    assert result["coefficients"][1] > 1.0
    assert result["converged"]
    recorder.capture(regression_model.binary_regression_summary_table(result).round(4))


def test_get_quantile_regression_median_close_to_ols_for_symmetric_noise(recorder):
    rng = np.random.default_rng(6)
    n = 1000
    x = pd.Series(rng.standard_normal(n), name="X")
    y = 1 + 2 * x + rng.standard_normal(n) * 0.3

    ols_result = regression_model.get_ols(y, x)
    qr_result = regression_model.get_quantile_regression(y, x, tau=0.5)

    # For Gaussian noise the median and mean regression coincide asymptotically.
    assert np.allclose(ols_result["coefficients"], qr_result["coefficients"], atol=0.1)
    recorder.capture(
        regression_model.quantile_regression_summary_table(qr_result).round(4)
    )


def test_get_quantile_regression_upper_tail_shifted():
    # Spread grows with x, so the 0.9 quantile line should be steeper than the 0.1.
    rng = np.random.default_rng(7)
    n = 2000
    x = pd.Series(rng.uniform(0, 10, n), name="X")
    y = 1 + x + rng.standard_normal(n) * (0.1 + 0.5 * x)

    low = regression_model.get_quantile_regression(y, x, tau=0.1)
    high = regression_model.get_quantile_regression(y, x, tau=0.9)

    assert high["coefficients"][1] > low["coefficients"][1]


def test_get_quantile_regression_with_bootstrap_standard_errors():
    rng = np.random.default_rng(8)
    n = 300
    x = pd.Series(rng.standard_normal(n), name="X")
    y = 1 + 2 * x + rng.standard_normal(n) * 0.3

    result = regression_model.get_quantile_regression(y, x, n_bootstrap=50, seed=1)

    assert result["standard_errors"] is not None
    assert len(result["standard_errors"]) == 2
    assert np.all(result["standard_errors"] > 0)


def test_get_quantile_regression_invalid_tau():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    try:
        regression_model.get_quantile_regression(y, x, tau=1.5)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
