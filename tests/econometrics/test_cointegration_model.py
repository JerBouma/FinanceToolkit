"""Cointegration Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import cointegration_model

# pylint: disable=missing-function-docstring


def test_get_engle_granger_cointegration_independent(recorder):
    rng = np.random.default_rng(2)
    series_a = pd.Series(np.cumsum(rng.standard_normal(500)))
    series_b = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(
        cointegration_model.get_engle_granger_cointegration(series_a, series_b).round(4)
    )


def test_get_engle_granger_cointegration_cointegrated(recorder):
    rng = np.random.default_rng(2)
    common_trend = np.cumsum(rng.standard_normal(500))
    series_a = pd.Series(common_trend + rng.standard_normal(500) * 0.1)
    series_b = pd.Series(common_trend * 2 + 5 + rng.standard_normal(500) * 0.1)
    recorder.capture(
        cointegration_model.get_engle_granger_cointegration(series_a, series_b).round(4)
    )


def test_get_johansen_cointegration_independent(recorder):
    # Three fully independent random walks -- no shared stochastic trend, so the
    # Johansen procedure should fail to reject rank 0 (no cointegration).
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        np.column_stack([np.cumsum(rng.standard_normal(500)) for _ in range(3)]),
        columns=["A", "B", "C"],
    )
    result = cointegration_model.get_johansen_cointegration(data).round(4)

    assert not result.loc["r <= 0", "Reject (Trace, 5%)"]

    recorder.capture(result)


def test_get_johansen_cointegration_one_cointegrating_relation(recorder):
    # A and B share a common stochastic trend (B is a scaled/shifted version of A
    # plus noise), C is an independent random walk -- exactly one cointegrating
    # relation should be detected: reject rank <= 0, fail to reject rank <= 1.
    rng = np.random.default_rng(7)
    common_trend = np.cumsum(rng.standard_normal(500))
    series_a = common_trend + rng.standard_normal(500) * 0.1
    series_b = 2 * common_trend + 5 + rng.standard_normal(500) * 0.1
    series_c = np.cumsum(rng.standard_normal(500))
    data = pd.DataFrame({"A": series_a, "B": series_b, "C": series_c})

    result = cointegration_model.get_johansen_cointegration(data).round(4)

    assert result.loc["r <= 0", "Reject (Trace, 5%)"]
    assert not result.loc["r <= 1", "Reject (Trace, 5%)"]

    recorder.capture(result)


def test_get_johansen_cointegration_bivariate(recorder):
    rng = np.random.default_rng(2)
    common_trend = np.cumsum(rng.standard_normal(500))
    series_a = pd.Series(common_trend + rng.standard_normal(500) * 0.1)
    series_b = pd.Series(common_trend * 2 + 5 + rng.standard_normal(500) * 0.1)
    data = pd.DataFrame({"A": series_a, "B": series_b})

    result = cointegration_model.get_johansen_cointegration(data).round(4)

    assert result.loc["r <= 0", "Reject (Trace, 5%)"]

    recorder.capture(result)


def test_get_johansen_cointegration_k_ar_diff(recorder):
    rng = np.random.default_rng(7)
    common_trend = np.cumsum(rng.standard_normal(500))
    series_a = common_trend + rng.standard_normal(500) * 0.1
    series_b = 2 * common_trend + 5 + rng.standard_normal(500) * 0.1
    series_c = np.cumsum(rng.standard_normal(500))
    data = pd.DataFrame({"A": series_a, "B": series_b, "C": series_c})

    recorder.capture(
        cointegration_model.get_johansen_cointegration(data, k_ar_diff=2).round(4)
    )


def test_get_johansen_cointegration_too_few_observations(recorder):
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    recorder.capture(cointegration_model.get_johansen_cointegration(data))


def test_get_johansen_cointegration_invalid_det_order():
    data = pd.DataFrame(
        {"A": np.arange(20, dtype=float), "B": np.arange(20, dtype=float) * 2}
    )
    try:
        cointegration_model.get_johansen_cointegration(data, det_order=2)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_johansen_cointegration_invalid_number_of_series():
    data = pd.DataFrame({"A": np.arange(20, dtype=float)})
    try:
        cointegration_model.get_johansen_cointegration(data)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_johansen_cointegration_invalid_k_ar_diff():
    data = pd.DataFrame(
        {"A": np.arange(20, dtype=float), "B": np.arange(20, dtype=float) * 2}
    )
    try:
        cointegration_model.get_johansen_cointegration(data, k_ar_diff=-1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
