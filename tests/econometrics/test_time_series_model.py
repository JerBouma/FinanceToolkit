"""Time Series Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import time_series_model

# pylint: disable=missing-function-docstring


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


def test_get_arima_forecast_recovers_ar1_coefficients(recorder):
    # y_t = 0.5 + 0.6 * y_(t-1) + e_t -- a large sample AR(1) should recover phi
    # close to 0.6 and forecasts should decay towards the unconditional mean
    # (0.5 / (1 - 0.6) = 1.25), not explode.
    rng = np.random.default_rng(1)
    n = 3000
    phi_true, const_true = 0.6, 0.5
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = const_true + phi_true * y[t - 1] + rng.standard_normal()

    result = time_series_model.get_arima_forecast(
        pd.Series(y), p=1, d=0, q=0, forecast_steps=20
    )

    assert abs(result.ar_coefficients[0] - phi_true) < 0.05
    assert result.converged

    # For d=0, `result.constant` is the unconditional mean directly (as `statsmodels`
    # reports it), not the recursion intercept `c` -- see `ARIMAResult.constant`.
    unconditional_mean = result.constant
    # A stationary AR(1) forecast's distance from the unconditional mean decays
    # geometrically (|distance_h| = |distance_0| * phi^h) -- it should be much closer
    # to the mean at the end of a 20-step horizon than at the start, and essentially
    # converged by then.
    distance_first_step = abs(result.forecast.iloc[0] - unconditional_mean)
    distance_last_step = abs(result.forecast.iloc[-1] - unconditional_mean)
    assert distance_last_step < distance_first_step
    assert distance_last_step < 1e-3

    recorder.capture(result.summary().round(4))


def test_get_arima_forecast_recovers_arma11_coefficients():
    # y_t = 0.5 * y_(t-1) + 0.3 * e_(t-1) + e_t
    rng = np.random.default_rng(42)
    n = 2000
    phi_true, theta_true = 0.5, 0.3
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi_true * y[t - 1] + theta_true * e[t - 1] + e[t]

    result = time_series_model.get_arima_forecast(
        pd.Series(y), p=1, d=0, q=1, forecast_steps=5, include_constant=False
    )

    assert abs(result.ar_coefficients[0] - phi_true) < 0.1
    assert abs(result.ma_coefficients[0] - theta_true) < 0.1
    assert result.constant == 0.0


def test_get_arima_forecast_differencing_undoes_correctly():
    # A pure random walk (d=1, differences are i.i.d. noise -- AR(1) on the noise
    # should find phi close to 0) -- the un-differenced forecast should stay close
    # to the last observed level, not run away.
    rng = np.random.default_rng(3)
    n = 500
    y = np.cumsum(rng.standard_normal(n)) + 50

    result = time_series_model.get_arima_forecast(
        pd.Series(y), p=1, d=1, q=0, forecast_steps=10, include_constant=False
    )

    assert abs(result.ar_coefficients[0]) < 0.15
    # The forecast should stay in the neighborhood of the last actual value (a
    # driftless random walk's best forecast is roughly the last observed level).
    assert abs(result.forecast.iloc[-1] - y[-1]) < 5 * np.std(np.diff(y))


def test_get_arima_forecast_invalid_type():
    try:
        time_series_model.get_arima_forecast([1, 2, 3], p=1, d=0, q=0)
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_arima_forecast_negative_order():
    try:
        time_series_model.get_arima_forecast(pd.Series([1.0, 2.0, 3.0]), p=-1, d=0, q=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_arima_forecast_p_and_q_both_zero():
    try:
        time_series_model.get_arima_forecast(pd.Series(np.arange(20.0)), p=0, d=1, q=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_arima_forecast_invalid_forecast_steps():
    try:
        time_series_model.get_arima_forecast(
            pd.Series(np.arange(20.0)), p=1, d=0, q=0, forecast_steps=0
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_arima_forecast_too_few_observations():
    try:
        time_series_model.get_arima_forecast(pd.Series([1.0, 2.0, 3.0]), p=1, d=0, q=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------


def test_get_var_forecast_recovers_known_coefficients(recorder):
    rng = np.random.default_rng(5)
    n = 5000
    phi = np.array([[0.5, 0.2], [0.1, 0.6]])
    const = np.array([0.1, -0.2])
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = const + phi @ y[t - 1] + rng.standard_normal(2) * 0.3

    data = pd.DataFrame(y, columns=["Y1", "Y2"])
    result = time_series_model.get_var_forecast(data, lags=1, forecast_steps=5)

    fitted_phi = (
        result.coefficient_matrices[1].loc[["Y1", "Y2"], ["Y1", "Y2"]].to_numpy()
    )
    assert np.allclose(fitted_phi, phi, atol=0.05)
    assert np.allclose(result.intercept[["Y1", "Y2"]].to_numpy(), const, atol=0.05)

    recorder.capture(result.coefficient_matrices[1].round(4))
    recorder.capture(result.forecast.round(4))


def test_get_var_forecast_one_step_matches_hand_computed_recursion():
    # Two independent (noiseless), non-collinear AR(1)-shaped recursions:
    # Y1_t = 1 + 0.5 * Y1_(t-1), Y2_t = 2 - 0.3 * Y2_(t-1). Since the generating
    # process is an exact, noiseless VAR(1), the fitted equations should recover it
    # (near) exactly and the one-step forecast should exactly match one more turn of
    # the same recursion applied to the last observed row.
    n = 30
    y1 = [0.0]
    y2 = [0.0]
    for _ in range(1, n):
        y1.append(1 + 0.5 * y1[-1])
        y2.append(2 - 0.3 * y2[-1])
    data = pd.DataFrame({"Y1": y1, "Y2": y2})

    result = time_series_model.get_var_forecast(data, lags=1, forecast_steps=1)

    expected_y1 = 1 + 0.5 * y1[-1]
    expected_y2 = 2 - 0.3 * y2[-1]
    assert abs(result.forecast["Y1"].iloc[0] - expected_y1) < 1e-6
    assert abs(result.forecast["Y2"].iloc[0] - expected_y2) < 1e-6


def test_get_var_forecast_invalid_type():
    try:
        time_series_model.get_var_forecast([1, 2, 3], lags=1)
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_var_forecast_invalid_lags():
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    try:
        time_series_model.get_var_forecast(data, lags=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_var_forecast_single_column():
    data = pd.DataFrame({"A": np.arange(20.0)})
    try:
        time_series_model.get_var_forecast(data, lags=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_var_forecast_too_few_observations():
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    try:
        time_series_model.get_var_forecast(data, lags=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_var_forecast_invalid_forecast_steps():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.standard_normal((50, 2)), columns=["A", "B"])
    try:
        time_series_model.get_var_forecast(data, lags=1, forecast_steps=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# VECM
# ---------------------------------------------------------------------------


def test_get_vecm_forecast_identifies_rank_one_and_mean_reverts(recorder):
    # Same construction as tests/econometrics/test_cointegration_model.py's Johansen
    # tests: A and B share a common stochastic trend -- exactly one cointegrating
    # relation should be found, and the forecasted spread (A - beta_ratio * B) should
    # converge to a fixed equilibrium and stay there (mean reversion), rather than
    # drift indefinitely.
    rng = np.random.default_rng(3)
    n = 500
    common_trend = np.cumsum(rng.standard_normal(n))
    series_a = common_trend + rng.standard_normal(n) * 0.2
    series_b = common_trend * 1.5 + 3 + rng.standard_normal(n) * 0.2
    data = pd.DataFrame({"A": series_a, "B": series_b})

    result = time_series_model.get_vecm_forecast(data, k_ar_diff=1, forecast_steps=20)

    assert result.rank == 1
    assert result.cointegrating_vectors.shape == (2, 1)

    beta = result.cointegrating_vectors.to_numpy()
    spread_forecast = (result.forecast.to_numpy() @ beta).flatten()

    # The spread should stabilize: the change over the last few forecast steps
    # should be much smaller than the change over the first few steps.
    early_movement = abs(spread_forecast[2] - spread_forecast[0])
    late_movement = abs(spread_forecast[-1] - spread_forecast[-3])
    assert late_movement < early_movement

    # And it should have essentially flattened out by the end of the horizon.
    assert late_movement < 0.05

    recorder.capture(result.cointegrating_vectors.round(4))


def test_get_vecm_forecast_mean_reverts_more_than_naive_var_in_differences():
    # Own-consistency check: a VECM's forecasted spread should converge to a fixed
    # equilibrium faster/more tightly than a naive VAR-in-differences forecast (which
    # has no error-correction term and can keep drifting).
    rng = np.random.default_rng(3)
    n = 500
    common_trend = np.cumsum(rng.standard_normal(n))
    series_a = common_trend + rng.standard_normal(n) * 0.2
    series_b = common_trend * 1.5 + 3 + rng.standard_normal(n) * 0.2
    data = pd.DataFrame({"A": series_a, "B": series_b})

    vecm_result = time_series_model.get_vecm_forecast(
        data, k_ar_diff=1, forecast_steps=20
    )
    beta = vecm_result.cointegrating_vectors.to_numpy()

    differences = data.diff().dropna()
    var_result = time_series_model.get_var_forecast(
        differences, lags=1, forecast_steps=20
    )
    level_path = data.to_numpy()[-1] + np.cumsum(var_result.forecast.to_numpy(), axis=0)
    naive_spread = level_path @ beta

    vecm_spread = (vecm_result.forecast.to_numpy() @ beta).flatten()

    vecm_late_movement = abs(vecm_spread[-1] - vecm_spread[-3])
    naive_late_movement = abs(naive_spread[-1] - naive_spread[-3])

    assert vecm_late_movement < naive_late_movement


def test_get_vecm_forecast_no_cointegration_raises():
    rng = np.random.default_rng(2)
    n = 500
    data = pd.DataFrame(
        {
            "A": np.cumsum(rng.standard_normal(n)),
            "B": np.cumsum(rng.standard_normal(n)),
        }
    )
    try:
        time_series_model.get_vecm_forecast(data, k_ar_diff=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_vecm_forecast_invalid_type():
    try:
        time_series_model.get_vecm_forecast([1, 2, 3])
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_vecm_forecast_invalid_k_ar_diff():
    data = pd.DataFrame({"A": np.arange(20.0), "B": np.arange(20.0) * 2})
    try:
        time_series_model.get_vecm_forecast(data, k_ar_diff=0)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_get_vecm_forecast_invalid_significance():
    data = pd.DataFrame({"A": np.arange(20.0), "B": np.arange(20.0) * 2})
    try:
        time_series_model.get_vecm_forecast(data, significance=0.5)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
