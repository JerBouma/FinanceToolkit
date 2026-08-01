"""Fama-MacBeth Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import fama_macbeth_model

# pylint: disable=missing-function-docstring


def _synthetic_single_factor(n_periods=200, n_assets=25, true_premium=0.005, seed=1):
    rng = np.random.default_rng(seed)
    true_betas = rng.uniform(0.5, 1.5, n_assets)

    factor = pd.Series(rng.standard_normal(n_periods) * 0.02, name="Market")
    returns = pd.DataFrame(
        {
            f"Asset_{i}": true_betas[i] * true_premium
            + true_betas[i] * factor.to_numpy()
            + rng.standard_normal(n_periods) * 0.01
            for i in range(n_assets)
        }
    )
    return returns, factor, true_betas


def test_get_fama_macbeth_regression_recovers_known_risk_premium(recorder):
    returns, factor, true_betas = _synthetic_single_factor(true_premium=0.005)
    result = fama_macbeth_model.get_fama_macbeth_regression(returns, factor)

    assert abs(result["risk_premia"]["Market"] - 0.005) < 0.005
    assert np.allclose(result["betas"]["Market"].to_numpy(), true_betas, atol=0.15)
    assert result["n_assets"] == 25
    assert result["n_periods"] == 200

    recorder.capture(fama_macbeth_model.fama_macbeth_summary_table(result).round(4))


def test_get_fama_macbeth_regression_noise_factor_is_insignificant():
    # A factor uncorrelated with returns should produce a risk premium
    # indistinguishable from zero -- the classic "is this factor priced" null.
    rng = np.random.default_rng(2)
    n_periods, n_assets = 200, 25
    returns = pd.DataFrame(
        rng.standard_normal((n_periods, n_assets)) * 0.01,
        columns=[f"Asset_{i}" for i in range(n_assets)],
    )
    noise_factor = pd.Series(rng.standard_normal(n_periods) * 0.02, name="Noise")

    result = fama_macbeth_model.get_fama_macbeth_regression(returns, noise_factor)

    assert result["p_values"]["Noise"] > 0.05


def test_get_fama_macbeth_regression_series_factor_gets_named_column():
    returns, factor, _ = _synthetic_single_factor()
    unnamed_factor = pd.Series(factor.to_numpy(), index=factor.index)

    result = fama_macbeth_model.get_fama_macbeth_regression(returns, unnamed_factor)

    assert result["factor_names"] == ["Factor"]


def test_get_fama_macbeth_regression_multi_factor(recorder):
    rng = np.random.default_rng(3)
    n_periods, n_assets = 250, 30
    true_betas = rng.uniform(0.3, 1.2, (n_assets, 2))
    true_premia = np.array([0.004, 0.002])

    factors = pd.DataFrame(
        rng.standard_normal((n_periods, 2)) * 0.02, columns=["Market", "Size"]
    )
    returns = pd.DataFrame(
        {
            f"Asset_{i}": true_betas[i] @ true_premia
            + factors.to_numpy() @ true_betas[i]
            + rng.standard_normal(n_periods) * 0.01
            for i in range(n_assets)
        }
    )

    result = fama_macbeth_model.get_fama_macbeth_regression(returns, factors)

    assert list(result["factor_names"]) == ["Market", "Size"]
    assert result["betas"].shape == (n_assets, 2)
    recorder.capture(fama_macbeth_model.fama_macbeth_summary_table(result).round(4))


def test_get_fama_macbeth_regression_no_constant():
    returns, factor, _ = _synthetic_single_factor()
    result = fama_macbeth_model.get_fama_macbeth_regression(
        returns, factor, add_constant=False
    )

    assert "Intercept" not in result["risk_premia"].index
    assert list(result["cross_sectional_coefficients"].columns) == ["Market"]


def test_get_fama_macbeth_regression_invalid_returns_type():
    factor = pd.Series(np.random.default_rng(1).standard_normal(50))
    try:
        fama_macbeth_model.get_fama_macbeth_regression([1, 2, 3], factor)
        raise AssertionError("Expected a TypeError to be raised.")
    except TypeError:
        pass


def test_get_fama_macbeth_regression_invalid_factors_type():
    returns = pd.DataFrame(np.random.default_rng(1).standard_normal((50, 5)))
    try:
        fama_macbeth_model.get_fama_macbeth_regression(returns, [1, 2, 3])
        raise AssertionError("Expected a TypeError to be raised.")
    except TypeError:
        pass


def test_get_fama_macbeth_regression_no_overlapping_periods():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        rng.standard_normal((50, 5)),
        index=pd.date_range("2020-01-01", periods=50, freq="D"),
    )
    factor = pd.Series(
        rng.standard_normal(50),
        index=pd.date_range("2030-01-01", periods=50, freq="D"),
    )
    try:
        fama_macbeth_model.get_fama_macbeth_regression(returns, factor)
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_fama_macbeth_regression_too_few_assets():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.standard_normal((50, 2)), columns=["Asset_0", "Asset_1"])
    factor = pd.Series(rng.standard_normal(50), name="Market")
    try:
        fama_macbeth_model.get_fama_macbeth_regression(returns, factor)
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass
