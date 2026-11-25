"""Cointegration Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.stationarity import (
    cointegration_model,
    unitroot_model,
)


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
    # Three independent random walks, so Johansen should fail to reject rank 0.
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        np.column_stack([np.cumsum(rng.standard_normal(500)) for _ in range(3)]),
        columns=["A", "B", "C"],
    )
    result = cointegration_model.get_johansen_cointegration(data).round(4)

    assert not result.loc["r <= 0", "Reject (Trace, 5%)"]

    recorder.capture(result)


def test_get_johansen_cointegration_one_cointegrating_relation(recorder):
    # A and B share a trend, C does not, so exactly one relation should be found.
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


def test_get_johansen_cointegration_lags(recorder):
    rng = np.random.default_rng(7)
    common_trend = np.cumsum(rng.standard_normal(500))
    series_a = common_trend + rng.standard_normal(500) * 0.1
    series_b = 2 * common_trend + 5 + rng.standard_normal(500) * 0.1
    series_c = np.cumsum(rng.standard_normal(500))
    data = pd.DataFrame({"A": series_a, "B": series_b, "C": series_c})

    recorder.capture(
        cointegration_model.get_johansen_cointegration(data, lags=2).round(4)
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


def test_get_johansen_cointegration_invalid_lags():
    data = pd.DataFrame(
        {"A": np.arange(20, dtype=float), "B": np.arange(20, dtype=float) * 2}
    )
    try:
        cointegration_model.get_johansen_cointegration(data, lags=-1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_engle_granger_cointegration_short_sample():
    # With 16 observations the uncapped default lag search leaves the residual ADF regression
    # without degrees of freedom and returns a statistic of 0 with a p-value of 0.9859.
    rng = np.random.default_rng(0)
    common_trend = np.cumsum(rng.standard_normal(16))
    series_a = pd.Series(common_trend + rng.standard_normal(16) * 0.5)
    series_b = pd.Series(common_trend * 2 + 5 + rng.standard_normal(16) * 0.5)

    result = cointegration_model.get_engle_granger_cointegration(series_a, series_b)

    assert result["Engle-Granger Statistic"] < -1
    assert result["P-Value"] < 0.5  # noqa: PLR2004


def test_cointegration_tests_need_enough_observations():
    rng = np.random.default_rng(0)
    walk = pd.Series(np.cumsum(rng.normal(size=4)))
    other = walk + rng.normal(scale=0.1, size=4)

    engle_granger = cointegration_model.get_engle_granger_cointegration(walk, other)
    johansen = cointegration_model.get_johansen_cointegration(
        pd.concat([walk, other], axis=1)
    )

    # Four points used to report cointegration at p = 0.0015.
    assert np.isnan(engle_granger["P-Value"])
    assert not engle_granger["Cointegrated (5%)"]
    assert johansen["Trace Statistic"].isna().all()


def test_get_engle_granger_cointegration_matches_statsmodels_coint():
    # The residual-based test with the lag search from one decomposition gives statsmodels' coint statistic, p-value and critical values.  # noqa: E501
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(11)

    for observations in [30, 250, 1000]:
        common = np.cumsum(rng.standard_normal(observations))
        independent = common + rng.standard_normal(observations)
        for dependent in [
            2 * common + rng.standard_normal(observations),
            np.cumsum(rng.standard_normal(observations)),
        ]:
            result = cointegration_model.get_engle_granger_cointegration(
                pd.Series(dependent), pd.Series(independent)
            )
            statistic, p_value, critical_values = stattools.coint(
                dependent,
                independent,
                trend="c",
                maxlag=unitroot_model.default_max_lag(observations),
                autolag="aic",
            )

            np.testing.assert_allclose(
                result["Engle-Granger Statistic"], statistic, rtol=0, atol=1e-10
            )
            np.testing.assert_allclose(result["P-Value"], p_value, rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                result[
                    ["Critical Value 1%", "Critical Value 5%", "Critical Value 10%"]
                ].to_numpy(dtype=float),
                critical_values,
            )


def test_the_tests_run_without_the_mackinnon_functions_in_stattools(monkeypatch):
    # statsmodels 0.15 no longer has mackinnonp and mackinnoncrit in stattools, only in
    # adfvalues; without them the ADF and Engle-Granger tests give the same results.
    from statsmodels.tsa import stattools

    rng = np.random.default_rng(2)
    series_a = pd.Series(np.cumsum(rng.standard_normal(500)))
    series_b = series_a + rng.standard_normal(500)
    expected_adf = unitroot_model.get_augmented_dickey_fuller(series_a)
    expected_engle_granger = cointegration_model.get_engle_granger_cointegration(
        series_a, series_b
    )

    monkeypatch.delattr(stattools, "mackinnonp", raising=False)
    monkeypatch.delattr(stattools, "mackinnoncrit", raising=False)

    pd.testing.assert_series_equal(
        unitroot_model.get_augmented_dickey_fuller(series_a), expected_adf
    )
    pd.testing.assert_series_equal(
        cointegration_model.get_engle_granger_cointegration(series_a, series_b),
        expected_engle_granger,
    )
