"""Unit Root Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics.stationarity import unitroot_model

# The simulated break sits at index 150, so the last pre-break index is expected.
EXPECTED_BREAK_INDEX = 149


def test_get_augmented_dickey_fuller_random_walk(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(unitroot_model.get_augmented_dickey_fuller(random_walk).round(4))


def test_get_augmented_dickey_fuller_stationary(recorder):
    rng = np.random.default_rng(1)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.standard_normal()
    stationary = pd.Series(values)
    recorder.capture(unitroot_model.get_augmented_dickey_fuller(stationary).round(4))


def test_get_augmented_dickey_fuller_regression_types(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(
        unitroot_model.get_augmented_dickey_fuller(random_walk, regression="n").round(4)
    )
    recorder.capture(
        unitroot_model.get_augmented_dickey_fuller(random_walk, regression="ct").round(
            4
        )
    )


def test_get_augmented_dickey_fuller_too_few_observations(recorder):
    series = pd.Series([1.0, 2.0, 3.0])
    recorder.capture(unitroot_model.get_augmented_dickey_fuller(series))


def test_get_augmented_dickey_fuller_invalid_regression():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        unitroot_model.get_augmented_dickey_fuller(series, regression="invalid")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_kpss_test_random_walk(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(unitroot_model.get_kpss_test(random_walk).round(4))


def test_get_kpss_test_stationary(recorder):
    rng = np.random.default_rng(1)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.standard_normal()
    stationary = pd.Series(values)
    recorder.capture(unitroot_model.get_kpss_test(stationary).round(4))


def test_get_kpss_test_agrees_with_adf(recorder):
    # seed=2 rather than 1: at n=500 the Schwert lag makes KPSS oversized on that draw.
    rng = np.random.default_rng(2)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))

    rng = np.random.default_rng(2)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.standard_normal()
    stationary = pd.Series(values)

    adf_random_walk = unitroot_model.get_augmented_dickey_fuller(random_walk)
    adf_stationary = unitroot_model.get_augmented_dickey_fuller(stationary)
    kpss_random_walk = unitroot_model.get_kpss_test(random_walk)
    kpss_stationary = unitroot_model.get_kpss_test(stationary)

    assert not adf_random_walk["Reject Unit Root (5%)"]
    assert kpss_random_walk["Reject Stationarity (5%)"]

    assert adf_stationary["Reject Unit Root (5%)"]
    assert not kpss_stationary["Reject Stationarity (5%)"]

    recorder.capture(
        pd.DataFrame(
            {
                "ADF Random Walk": adf_random_walk,
                "KPSS Random Walk": kpss_random_walk,
                "ADF Stationary": adf_stationary,
                "KPSS Stationary": kpss_stationary,
            }
        ).round(4),
        strip=True,
    )


def test_get_kpss_test_regression_types(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(
        unitroot_model.get_kpss_test(random_walk, regression="ct").round(4)
    )


def test_get_kpss_test_too_few_observations(recorder):
    series = pd.Series([1.0, 2.0, 3.0])
    recorder.capture(unitroot_model.get_kpss_test(series))


def test_get_kpss_test_invalid_regression():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        unitroot_model.get_kpss_test(series, regression="n")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_phillips_perron_test_random_walk(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(unitroot_model.get_phillips_perron_test(random_walk).round(4))


def test_get_phillips_perron_test_stationary(recorder):
    rng = np.random.default_rng(1)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.standard_normal()
    stationary = pd.Series(values)
    recorder.capture(unitroot_model.get_phillips_perron_test(stationary).round(4))


def test_get_phillips_perron_test_agrees_with_adf(recorder):
    # PP and ADF test the same null differently, so they should broadly agree.
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))

    rng = np.random.default_rng(1)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.5 * values[i - 1] + rng.standard_normal()
    stationary = pd.Series(values)

    adf_random_walk = unitroot_model.get_augmented_dickey_fuller(random_walk)
    adf_stationary = unitroot_model.get_augmented_dickey_fuller(stationary)
    pp_random_walk = unitroot_model.get_phillips_perron_test(random_walk)
    pp_stationary = unitroot_model.get_phillips_perron_test(stationary)

    assert (
        adf_random_walk["Reject Unit Root (5%)"]
        == pp_random_walk["Reject Unit Root (5%)"]
    )
    assert (
        adf_stationary["Reject Unit Root (5%)"]
        == pp_stationary["Reject Unit Root (5%)"]
    )
    assert not pp_random_walk["Reject Unit Root (5%)"]
    assert pp_stationary["Reject Unit Root (5%)"]

    recorder.capture(
        pd.DataFrame(
            {
                "ADF Random Walk": adf_random_walk,
                "PP Random Walk": pp_random_walk,
                "ADF Stationary": adf_stationary,
                "PP Stationary": pp_stationary,
            }
        ).round(4),
        strip=True,
    )


def test_get_phillips_perron_test_regression_types(recorder):
    rng = np.random.default_rng(1)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(
        unitroot_model.get_phillips_perron_test(random_walk, regression="ct").round(4)
    )


def test_get_phillips_perron_test_too_few_observations(recorder):
    series = pd.Series([1.0, 2.0, 3.0])
    recorder.capture(unitroot_model.get_phillips_perron_test(series))


def test_get_phillips_perron_test_invalid_regression():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    try:
        unitroot_model.get_phillips_perron_test(series, regression="n")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_zivot_andrews_test_stationary_with_break(recorder):
    rng = np.random.default_rng(5)
    n = 300
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.6 * values[i - 1] + rng.standard_normal() * 0.5
    values[150:] += 8.0
    series = pd.Series(values)
    result = unitroot_model.get_zivot_andrews_test(series, regression="c")
    assert result["Reject Unit Root (5%)"]
    assert result["Break Index"] == EXPECTED_BREAK_INDEX
    recorder.capture(result.round(4))


def test_get_zivot_andrews_test_random_walk(recorder):
    rng = np.random.default_rng(5)
    n = 300
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.6 * values[i - 1] + rng.standard_normal() * 0.5
    values[150:] += 8.0
    rng = np.random.default_rng(5)
    random_walk = pd.Series(np.cumsum(rng.standard_normal(n)))
    result = unitroot_model.get_zivot_andrews_test(random_walk, regression="c")
    assert not result["Reject Unit Root (5%)"]
    recorder.capture(result.round(4))


def test_get_zivot_andrews_test_regression_types(recorder):
    rng = np.random.default_rng(11)
    n = 300
    values = np.cumsum(rng.standard_normal(n) * 0.3) + np.arange(n) * 0.05
    values[150:] += np.arange(n - 150) * 0.15
    series = pd.Series(values)
    recorder.capture(
        unitroot_model.get_zivot_andrews_test(series, regression="t").round(4)
    )
    recorder.capture(
        unitroot_model.get_zivot_andrews_test(series, regression="ct").round(4)
    )


def test_get_zivot_andrews_test_too_few_observations(recorder):
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    recorder.capture(unitroot_model.get_zivot_andrews_test(series))


def test_get_zivot_andrews_test_invalid_regression():
    rng = np.random.default_rng(1)
    series = pd.Series(np.cumsum(rng.standard_normal(300)))
    try:
        unitroot_model.get_zivot_andrews_test(series, regression="n")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_zivot_andrews_test_invalid_trim():
    rng = np.random.default_rng(1)
    series = pd.Series(np.cumsum(rng.standard_normal(300)))
    try:
        unitroot_model.get_zivot_andrews_test(series, trim=0.4)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_zivot_andrews_test_invalid_type():
    try:
        unitroot_model.get_zivot_andrews_test([1.0, 2.0, 3.0])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_default_max_lag():
    # Short samples are capped, long samples keep the Schwert (1989) rule of thumb.
    assert unitroot_model.default_max_lag(16) == 3  # noqa: PLR2004
    assert unitroot_model.default_max_lag(100) == 12  # noqa: PLR2004
    assert unitroot_model.default_max_lag(1000) == 22  # noqa: PLR2004
    assert unitroot_model.default_max_lag(4) == 0


def test_get_augmented_dickey_fuller_short_sample():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.standard_normal(16))

    result = unitroot_model.get_augmented_dickey_fuller(series)

    assert result["Lags Used"] <= 3  # noqa: PLR2004


def test_unit_root_tests_need_enough_observations():
    series = pd.Series([1.0, 1.2, 0.9, 1.4, 1.1])

    assert np.isnan(unitroot_model.get_augmented_dickey_fuller(series)["P-Value"])
    assert np.isnan(unitroot_model.get_kpss_test(series)["P-Value"])


def test_get_phillips_perron_test_rejects_from_the_p_value():
    # A short AR(1) whose statistic lies between the asymptotic and the finite-sample
    # 5% critical value: the p-value is below 0.05, the critical value for 29
    # observations is not reached, and the flag follows the p-value as in the ADF test.
    rng = np.random.default_rng(17)
    shocks = rng.normal(size=30)
    values = np.zeros(30)
    for position in range(1, 30):
        values[position] = 0.6 * values[position - 1] + shocks[position]

    result = unitroot_model.get_phillips_perron_test(pd.Series(values))

    assert result["P-Value"] < 0.05  # noqa: PLR2004
    assert result["Phillips-Perron Statistic"] > result["Critical Value 5%"]
    assert result["Reject Unit Root (5%)"]


def test_get_adf_regression_matches_statsmodels_adfuller():
    # The lag search from one QR decomposition picks the lag of statsmodels' search over a regression per lag, with the same statistic, observations and AIC.  # noqa: E501
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(7)

    for observations in [15, 40, 120, 500, 1500]:
        shocks = rng.standard_t(5, observations)
        stationary = np.zeros(observations)
        for i in range(1, observations):
            stationary[i] = 0.6 * stationary[i - 1] + shocks[i]

        for values in [np.cumsum(shocks), stationary, np.round(np.cumsum(shocks))]:
            for regression in ["n", "c", "ct"]:
                max_lag = unitroot_model.default_max_lag(observations)
                fast = unitroot_model.get_adf_regression(values, max_lag, regression)
                statistic, _, lag, used, _, criterion = stattools.adfuller(
                    values, maxlag=max_lag, regression=regression, autolag="AIC"
                )

                assert fast is not None
                assert fast[1:3] == (lag, used)
                np.testing.assert_allclose(fast[0], statistic, rtol=0, atol=1e-10)
                np.testing.assert_allclose(fast[3], criterion, rtol=1e-12)


def test_get_adf_regression_leaves_singular_designs_to_statsmodels():
    # A constant series has a design with a column of zeros, and a lag range statsmodels refuses stays refused.  # noqa: E501
    assert unitroot_model.get_adf_regression(np.ones(50), 3) is None
    assert unitroot_model.get_adf_regression(np.arange(30.0), 20) is None
