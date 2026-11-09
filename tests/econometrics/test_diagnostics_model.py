"""Risk Diagnostics Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics import diagnostics_model

# Thresholds for asserting significance, or the lack of it, in the tests below.
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
    # White noise should fail to reject no autocorrelation, so a high p-value.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    result = diagnostics_model.get_ljung_box_test(returns, lags=10).round(4)
    recorder.capture(result)
    recorder.capture(bool(result["P-Value"] > HIGH_P_VALUE_THRESHOLD))


def test_get_ljung_box_test_autocorrelated(recorder):
    # A strongly autocorrelated AR(1) should reject, so a low p-value.
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
    # A momentum series should give a Variance Ratio > 1 and a low p-value.
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
    # A mean-reverting series should give a Variance Ratio < 1 and a low p-value.
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


def test_get_level_significance():
    months = pd.period_range("2002-03", periods=5, freq="M")
    rates = pd.Series([0.0528, 0.0516, 0.04, np.nan, 0.0513], index=months)

    significance = diagnostics_model.get_level_significance(rates)

    # The last month at least as high is 2002-04, the last at least as low 2002-05; of the
    # three earlier values one is below the latest (the missing month is left out).
    assert significance["Highest Since"] == "2002-04"
    assert significance["Lowest Since"] == "2002-05"
    assert significance["Percentile"] == pytest.approx(1 / 3)
    assert significance["Start Date"] == "2002-03"
    assert significance["Observations"] == 4  # noqa: PLR2004
    assert significance["Latest Date"] == "2002-07"
    assert significance["Latest Value"] == pytest.approx(0.0513)


def test_get_level_significance_ties_and_records():
    days = pd.period_range("2026-01-01", periods=4, freq="D")

    # An unchanged rate is in the middle of its ties, and was as high and as low the day before.
    flat = diagnostics_model.get_level_significance(pd.Series([0.02] * 4, index=days))
    assert flat["Percentile"] == pytest.approx(0.5)
    assert flat["Highest Since"] == flat["Lowest Since"] == "2026-01-03"

    # A record high has no earlier date as high.
    record = diagnostics_model.get_level_significance(
        pd.Series([0.01, 0.03, 0.02, 0.04], index=days)
    )
    assert record["Percentile"] == 1
    assert pd.isna(record["Highest Since"])
    assert record["Lowest Since"] == "2026-01-03"

    # A single observation has no history to compare with.
    single = diagnostics_model.get_level_significance(pd.Series([0.01], index=days[:1]))
    assert pd.isna(single["Percentile"]) and pd.isna(single["Highest Since"])

    with pytest.raises(ValueError, match="no observations"):
        diagnostics_model.get_level_significance(pd.Series([np.nan], index=days[:1]))
