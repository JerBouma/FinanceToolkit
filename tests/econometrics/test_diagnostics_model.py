"""Risk Diagnostics Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import diagnostics_model

# pylint: disable=missing-function-docstring

# Thresholds used to assert statistical significance (or lack thereof) in the tests
# below, kept as named constants to avoid PLR2004 "magic value" lint warnings.
HIGH_P_VALUE_THRESHOLD = 0.05
LOW_P_VALUE_THRESHOLD = 0.001
SIGNIFICANCE_LEVEL = 0.05
VARIANCE_RATIO_LOWER_BOUND = 0.8
VARIANCE_RATIO_UPPER_BOUND = 1.2


def test_get_arch_lm_test(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(200) * 0.01)
    recorder.capture(diagnostics_model.get_arch_lm_test(returns).round(4))


def test_get_arch_lm_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_arch_lm_test(returns_df).round(4))


def test_get_arch_lm_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_arch_lm_test(returns, lags=5))


def test_get_jarque_bera_test(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(200) * 0.01)
    recorder.capture(diagnostics_model.get_jarque_bera_test(returns).round(4))


def test_get_jarque_bera_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_exponential(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_jarque_bera_test(returns_df).round(4))


def test_get_ljung_box_test_white_noise(recorder):
    # White noise should fail to reject the null hypothesis of no autocorrelation,
    # i.e. a high p-value.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    result = diagnostics_model.get_ljung_box_test(returns, lags=10).round(4)
    recorder.capture(result)
    recorder.capture(bool(result["P-Value"] > HIGH_P_VALUE_THRESHOLD))


def test_get_ljung_box_test_autocorrelated(recorder):
    # A strongly autocorrelated AR(1) series should reject the null hypothesis,
    # i.e. a low p-value.
    rng = np.random.default_rng(42)
    values = [0.0]
    for _ in range(500):
        values.append(0.8 * values[-1] + rng.standard_normal() * 0.01)
    returns = pd.Series(values[1:])

    result = diagnostics_model.get_ljung_box_test(returns, lags=10).round(4)
    recorder.capture(result)
    recorder.capture(bool(result["P-Value"] < LOW_P_VALUE_THRESHOLD))


def test_get_ljung_box_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_ljung_box_test(returns_df).round(4))


def test_get_ljung_box_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_ljung_box_test(returns, lags=10))


def test_get_variance_ratio_test_iid(recorder):
    # IID returns should give a Variance Ratio close to 1 and a high p-value.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(1000) * 0.01)
    result = diagnostics_model.get_variance_ratio_test(returns, q=5).round(4)
    recorder.capture(result)
    recorder.capture(
        bool(
            VARIANCE_RATIO_LOWER_BOUND
            < result["Variance Ratio"]
            < VARIANCE_RATIO_UPPER_BOUND
        )
    )


def test_get_variance_ratio_test_momentum(recorder):
    # A positively autocorrelated (momentum) series should give a Variance Ratio > 1
    # and a significant (low) p-value.
    rng = np.random.default_rng(42)
    values = [0.0]
    for _ in range(1000):
        values.append(0.3 * values[-1] + rng.standard_normal() * 0.01)
    returns = pd.Series(values[1:])

    result = diagnostics_model.get_variance_ratio_test(returns, q=5).round(4)
    recorder.capture(result)
    recorder.capture(
        bool(result["Variance Ratio"] > 1 and result["P-Value"] < SIGNIFICANCE_LEVEL)
    )


def test_get_variance_ratio_test_mean_reversion(recorder):
    # A negatively autocorrelated (mean-reverting) series should give a Variance Ratio
    # < 1 and a significant (low) p-value.
    rng = np.random.default_rng(42)
    values = [0.0]
    for _ in range(1000):
        values.append(-0.3 * values[-1] + rng.standard_normal() * 0.01)
    returns = pd.Series(values[1:])

    result = diagnostics_model.get_variance_ratio_test(returns, q=5).round(4)
    recorder.capture(result)
    recorder.capture(
        bool(result["Variance Ratio"] < 1 and result["P-Value"] < SIGNIFICANCE_LEVEL)
    )


def test_get_variance_ratio_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(
        diagnostics_model.get_variance_ratio_test(returns_df, q=4).round(4)
    )


def test_get_variance_ratio_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_variance_ratio_test(returns, q=5))


def test_get_cusum_test_stable_mean(recorder):
    rng = np.random.default_rng(7)
    stable = pd.Series(rng.standard_normal(300) * 0.01)
    result = diagnostics_model.get_cusum_test(stable)
    assert not result["Reject Stability (5%)"]
    recorder.capture(result.round(4))


def test_get_cusum_test_mean_shift(recorder):
    rng = np.random.default_rng(7)
    shifted = pd.Series(
        np.concatenate(
            [
                rng.standard_normal(150) * 0.01,
                rng.standard_normal(150) * 0.01 + 0.05,
            ]
        )
    )
    result = diagnostics_model.get_cusum_test(shifted)
    assert result["Reject Stability (5%)"]
    recorder.capture(result.round(4))


def test_get_cusum_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_cusum_test(returns_df).round(4))


def test_get_cusum_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_cusum_test(returns))


def test_get_cusum_test_invalid_type():
    try:
        diagnostics_model.get_cusum_test(123)
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass
