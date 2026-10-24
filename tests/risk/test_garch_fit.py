"""GARCH Fit and HAR-RV Forecast Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk import garch_model, realized_volatility_model

# pylint: disable=missing-function-docstring


def simulate_garch(omega, alpha, beta, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    variance = omega / (1 - alpha - beta)
    for step in range(n):
        returns[step] = np.sqrt(variance) * rng.standard_normal()
        variance = omega + alpha * returns[step] ** 2 + beta * variance
    return returns


def test_garch_fit_reaches_the_maximum_likelihood():
    returns = simulate_garch(2e-6, 0.08, 0.90)

    weights = garch_model.get_garch_weights(returns)
    fitted = garch_model.garch_log_maximization(weights, returns, len(returns))
    true = garch_model.garch_log_maximization([2e-6, 0.08, 0.90], returns, len(returns))

    # The fit does not stop at its starting point (Alpha 0.1, Beta 0.8): it scores at least
    # as well as the parameters the returns were simulated with.
    assert fitted <= true + 1e-6
    assert weights[1] == pytest.approx(0.08, abs=0.04)
    assert weights[2] == pytest.approx(0.90, abs=0.05)


def test_gjr_garch_fit_leaves_its_starting_point():
    returns = simulate_garch(2e-6, 0.08, 0.90, seed=1)

    weights = garch_model.get_gjr_garch_weights(returns)

    assert not np.allclose(weights[1:], [0.05, 0.1, 0.8])
    assert weights[1] + weights[2] / 2 + weights[3] < 1


def test_har_rv_forecasts_the_last_day():
    rng = np.random.default_rng(2)
    variance = pd.Series(
        rng.gamma(2, 1e-4, 300), index=pd.bdate_range("2024-01-01", periods=300)
    )

    forecast = realized_volatility_model.get_har_rv_forecast(variance)

    # The last day has no observed target, but its inputs exist, so it gets a forecast.
    assert np.isfinite(forecast.iloc[-1])
    assert forecast.iloc[:21].isna().all()
