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


def simulate_gjr_garch(omega, alpha, gamma, beta, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    variance = omega / (1 - alpha - gamma / 2 - beta)
    for step in range(n):
        returns[step] = np.sqrt(variance) * rng.standard_normal()
        variance = (
            omega
            + (alpha + gamma * (returns[step] < 0)) * returns[step] ** 2
            + beta * variance
        )
    return returns


def test_garch_backcast_is_the_exponentially_weighted_start():
    returns = np.linspace(0.001, 0.02, 100)

    weights = 0.94 ** np.arange(75)
    expected = np.sum(returns[:75] ** 2 * weights) / weights.sum()

    assert garch_model.get_backcast(returns) == pytest.approx(expected)
    # A series shorter than the window averages every return it has.
    assert garch_model.get_backcast(returns[:3]) == pytest.approx(
        np.sum(returns[:3] ** 2 * 0.94 ** np.arange(3)) / np.sum(0.94 ** np.arange(3))
    )

    # The first conditional variance is one GARCH step from a pre-sample squared return
    # and variance that both equal the backcast, and it is independent of the first return.
    sigma_2 = garch_model.get_garch(returns, [1e-6, 0.1, 0.85])
    assert sigma_2[0] == pytest.approx(1e-6 + 0.95 * garch_model.get_backcast(returns))
    assert sigma_2[1] == pytest.approx(1e-6 + 0.1 * returns[0] ** 2 + 0.85 * sigma_2[0])


def test_garch_parameters_match_the_arch_package():
    arch = pytest.importorskip("arch")

    # The first squared return of this path is 4% of its Variance: a recursion started from
    # it fits a Beta of 0.72, while the backcast start agrees with arch (0.86).
    returns = simulate_gjr_garch(2e-6, 0.08, 0.0, 0.85, seed=2)

    weights = garch_model.get_garch_weights(returns)
    reference = arch.arch_model(
        returns * 100, mean="Zero", vol="GARCH", p=1, q=1, rescale=False
    ).fit(disp="off")

    assert weights[0] == pytest.approx(reference.params.iloc[0] / 1e4, rel=1e-2)
    assert weights[1] == pytest.approx(reference.params.iloc[1], abs=1e-3)
    assert weights[2] == pytest.approx(reference.params.iloc[2], abs=1e-3)


def test_gjr_garch_parameters_match_the_arch_package():
    arch = pytest.importorskip("arch")

    returns = simulate_gjr_garch(2e-6, 0.03, 0.12, 0.85, seed=1)

    weights = garch_model.get_gjr_garch_weights(returns)
    reference = arch.arch_model(
        returns * 100, mean="Zero", vol="GARCH", p=1, o=1, q=1, rescale=False
    ).fit(disp="off")

    assert weights[0] == pytest.approx(reference.params.iloc[0] / 1e4, rel=1e-2)
    np.testing.assert_allclose(weights[1:], reference.params.iloc[1:], atol=1e-3)


def simulate_egarch(omega, alpha, gamma, beta, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    log_variance = np.empty(n)
    log_variance[0] = omega / (1 - beta)
    for t in range(1, n):
        log_variance[t] = (
            omega
            + beta * log_variance[t - 1]
            + alpha * (abs(shocks[t - 1]) - np.sqrt(2 / np.pi))
            + gamma * shocks[t - 1]
        )
    return np.exp(log_variance / 2) * shocks


def test_egarch_parameters_match_the_arch_package():
    arch = pytest.importorskip("arch")

    returns = simulate_egarch(-0.3, 0.12, -0.08, 0.97, seed=1)

    weights = np.array(garch_model.get_egarch_weights(returns, len(returns)))
    reference = np.array(
        arch.arch_model(
            returns * 100, mean="Zero", vol="EGARCH", p=1, o=1, q=1, rescale=False
        )
        .fit(disp="off")
        .params,
        dtype=float,
    )
    # Scaling the returns by 100 adds 2 ln(100) to the log variance, which the constant
    # absorbs as 2 ln(100) (1 - beta).
    reference[0] -= 2 * np.log(100) * (1 - reference[3])

    np.testing.assert_allclose(weights, reference, atol=1e-3)
