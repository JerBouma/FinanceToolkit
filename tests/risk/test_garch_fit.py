"""GARCH Fit and HAR-RV Forecast Tests"""

import math

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk.volatility import garch_model, realized_volatility_model


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


# The multi-start EGARCH fit on 1,500 observations takes about a minute on its own, so the
# suite's 60-second limit failed it whenever the machine was busy.
@pytest.mark.timeout(300)
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


@pytest.mark.parametrize(
    ("variance_function", "forecast_function", "weights"),
    [
        (garch_model.get_garch, garch_model.get_garch_forecast, [2e-6, 0.08, 0.9]),
        (
            garch_model.get_gjr_garch,
            garch_model.get_gjr_garch_forecast,
            [2e-6, 0.03, 0.1, 0.85],
        ),
        (
            garch_model.get_egarch,
            garch_model.get_egarch_forecast,
            [-0.3, 0.12, -0.08, 0.97],
        ),
    ],
)
def test_garch_family_runs_each_column_on_its_own_returns(
    variance_function, forecast_function, weights
):
    dates = pd.bdate_range("2023-01-02", periods=400)
    returns = pd.DataFrame(
        {
            "Full": simulate_garch(2e-6, 0.08, 0.9, n=400, seed=3),
            "Late": simulate_garch(2e-6, 0.08, 0.9, n=400, seed=4),
        },
        index=dates,
    )
    # Late lists after 150 days and misses two days later on.
    returns.iloc[:150, 1] = np.nan
    returns.iloc[[220, 300], 1] = np.nan

    variance = variance_function(returns, weights)
    forecast = forecast_function(returns, weights, forecast_steps=5)

    # The full column is untouched by the other column's missing days.
    np.testing.assert_allclose(
        variance["Full"], variance_function(returns["Full"].to_numpy(), weights)
    )
    np.testing.assert_allclose(
        forecast["Full"], forecast_function(returns["Full"].to_numpy(), weights, 5)
    )

    # The late column runs on its own observations and is missing where it has none.
    late = returns["Late"].dropna()
    np.testing.assert_allclose(
        variance["Late"].dropna(), variance_function(late.to_numpy(), weights)
    )
    assert variance["Late"].dropna().index.equals(late.index)
    np.testing.assert_allclose(
        forecast["Late"], forecast_function(late.to_numpy(), weights, 5)
    )


def simulate_garch_student_t(omega, alpha, beta, n, seed):
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    variance = omega / (1 - alpha - beta)
    for step in range(n):
        returns[step] = np.sqrt(variance) * rng.standard_t(5) / np.sqrt(5 / 3)
        variance = omega + alpha * returns[step] ** 2 + beta * variance
    return returns


def test_gjr_garch_fit_is_never_below_the_garch_fit(monkeypatch):
    # On this path the multi-start search alone settles in a local optimum (Beta 0.45)
    # with a log-likelihood 0.49 below the GARCH fit; arch reaches the GARCH optimum.
    returns = simulate_garch_student_t(1e-5, 0.1, 0.8, n=260, seed=15)
    garch_weights = garch_model.get_garch_weights(returns)
    nested_weights = [garch_weights[0], garch_weights[1], 0.0, garch_weights[2]]

    # GARCH is GJR-GARCH with gamma = 0: the likelihoods agree at that point.
    assert garch_model.gjr_garch_log_maximization(
        nested_weights, returns, len(returns)
    ) == pytest.approx(
        garch_model.garch_log_maximization(garch_weights, returns, len(returns))
    )

    fitted = garch_model.get_gjr_garch_weights(returns)
    assert garch_model.gjr_garch_log_maximization(
        fitted, returns, len(returns)
    ) <= garch_model.garch_log_maximization(garch_weights, returns, len(returns))
    assert fitted[3] == pytest.approx(0.9315, abs=1e-3)

    arch = pytest.importorskip("arch")
    reference = arch.arch_model(
        returns * 100, mean="Zero", vol="GARCH", p=1, o=1, q=1, rescale=False
    ).fit(disp="off")
    # The log-likelihood of the unscaled returns, with the constant the fit leaves out.
    log_likelihood = -0.5 * (
        garch_model.gjr_garch_log_maximization(fitted, returns, len(returns))
        + len(returns) * np.log(2 * np.pi)
    )
    assert log_likelihood == pytest.approx(
        reference.loglikelihood + len(returns) * np.log(100), abs=1e-3
    )

    # A search that settles in a worse local optimum falls back to the GARCH weights.
    original = garch_model._fit_multi_start

    def stuck(wrapper_func, bounds, initial_guess, model_name, **kwargs):
        if model_name.startswith("GJR-GARCH"):
            return np.array([garch_weights[0], 0.01, 0.02, 0.5])
        return original(wrapper_func, bounds, initial_guess, model_name, **kwargs)

    monkeypatch.setattr(garch_model, "_fit_multi_start", stuck)

    np.testing.assert_allclose(
        garch_model.get_gjr_garch_weights(returns), nested_weights
    )


def test_egarch_weights_are_fitted_once_per_series(monkeypatch):
    returns = np.random.default_rng(3).normal(0, 0.01, 200)
    calls = []
    original = garch_model._fit_multi_start

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(garch_model, "_fit_multi_start", counting)
    garch_model.EGARCH_WEIGHTS_CACHE.clear()

    first = garch_model.get_egarch_weights(returns)
    second = garch_model.get_egarch_weights(returns.copy())

    np.testing.assert_array_equal(first, second)
    assert len(calls) == 1


def test_egarch_weights_cache_is_safe_across_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(
        garch_model, "_fit_multi_start", lambda *args, **kwargs: np.ones(4)
    )
    garch_model.EGARCH_WEIGHTS_CACHE.clear()
    series = [np.full(50, value / 1000) for value in range(200)]

    # Many threads storing and reading at once, past the size of the cache.
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(garch_model.get_egarch_weights, series * 3))

    assert all(np.array_equal(result, np.ones(4)) for result in results)
    assert (
        len(garch_model.EGARCH_WEIGHTS_CACHE) <= garch_model.EGARCH_WEIGHTS_CACHE_SIZE
    )


def simulate_garch_family_returns(n=1500, seed=5):
    """A GJR-GARCH path with skewed Student-t shocks and a small drift, in percent."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(6, n) / np.sqrt(1.5)
    shocks = np.where(shocks < 0, shocks * 1.1, shocks * 0.9)
    returns = np.empty(n)
    variance = 0.5
    for step in range(n):
        returns[step] = 0.03 + np.sqrt(variance) * shocks[step]
        residual = returns[step] - 0.03
        variance = 0.05 + (0.05 + 0.1 * (residual < 0)) * residual**2 + 0.85 * variance
    return returns


def _arch_model(arch, returns, method, distribution, mean):
    from arch.univariate import (  # noqa: PLC0415
        APARCH,
        ARX,
        EGARCH,
        GARCH,
        ARCHInMean,
        ConstantMean,
        GeneralizedError,
        Normal,
        SkewStudent,
        StudentsT,
        ZeroMean,
    )

    volatility = {
        "garch": GARCH(1, 0, 1),
        "igarch": GARCH(1, 0, 1),
        "gjr": GARCH(1, 1, 1),
        "egarch": EGARCH(1, 1, 1),
        "aparch": APARCH(1, 1, 1),
        "tgarch": APARCH(1, 1, 1, delta=1.0),
        "ngarch": APARCH(1, 0, 1),
    }[method]
    model = {
        "zero": lambda: ZeroMean(returns, rescale=False),
        "constant": lambda: ConstantMean(returns, rescale=False),
        "arma": lambda: ARX(returns, lags=1, rescale=False),
        "in_mean": lambda: ARCHInMean(
            returns, constant=True, form="vol", volatility=volatility, rescale=False
        ),
    }[mean]()
    model.volatility = volatility
    model.distribution = {
        "normal": Normal,
        "t": StudentsT,
        "skewt": SkewStudent,
        "ged": GeneralizedError,
    }[distribution]()
    return model


# The weights in the arch package's order (the mean parameters first), and the same model in
# the Toolkit's order (the variance parameters first), with Beta = 1 - Alpha for IGARCH and a
# zero MA(1) term for arch's AR(1).
LIKELIHOOD_CASES = [
    ("garch", "t", "zero", [0.02, 0.08, 0.88, 6.0], [0.02, 0.08, 0.88, 6.0]),
    (
        "gjr",
        "skewt",
        "zero",
        [0.02, 0.04, 0.1, 0.88, 6.0, -0.1],
        [0.02, 0.04, 0.1, 0.88, 6.0, -0.1],
    ),
    (
        "egarch",
        "t",
        "zero",
        [0.0, 0.15, -0.08, 0.96, 6.0],
        [0.0, 0.15, -0.08, 0.96, 6.0],
    ),
    (
        "aparch",
        "ged",
        "zero",
        [0.03, 0.07, 0.4, 0.88, 1.6, 1.4],
        [0.03, 0.07, 0.4, 0.88, 1.6, 1.4],
    ),
    ("tgarch", "normal", "zero", [0.03, 0.08, 0.5, 0.88], [0.03, 0.08, 0.5, 0.88]),
    ("ngarch", "t", "zero", [0.03, 0.08, 0.88, 1.7, 6.0], [0.03, 0.08, 0.88, 1.7, 6.0]),
    ("igarch", "normal", "zero", [0.01, 0.1, 0.9], [0.01, 0.1]),
    (
        "garch",
        "skewt",
        "constant",
        [0.03, 0.02, 0.08, 0.88, 6.0, -0.1],
        [0.02, 0.08, 0.88, 0.03, 6.0, -0.1],
    ),
    (
        "garch",
        "normal",
        "arma",
        [0.03, 0.1, 0.02, 0.08, 0.88],
        [0.02, 0.08, 0.88, 0.03, 0.1, 0.0],
    ),
    (
        "garch",
        "normal",
        "in_mean",
        [0.03, 0.05, 0.02, 0.08, 0.88],
        [0.02, 0.08, 0.88, 0.03, 0.05],
    ),
]


@pytest.mark.parametrize(
    ("method", "distribution", "mean", "arch_weights", "weights"), LIKELIHOOD_CASES
)
def test_garch_family_log_likelihood_matches_the_arch_package(
    method, distribution, mean, arch_weights, weights
):
    arch = pytest.importorskip("arch")
    returns = simulate_garch_family_returns()

    reference = _arch_model(arch, returns, method, distribution, mean).fix(arch_weights)

    assert garch_model.get_garch_family_log_likelihood(
        returns, weights, method, distribution, mean
    ) == pytest.approx(reference.loglikelihood, rel=1e-9)


@pytest.mark.parametrize(
    ("method", "distribution", "mean"),
    [
        ("garch", "t", "zero"),
        ("aparch", "normal", "zero"),
        ("tgarch", "skewt", "zero"),
        ("gjr", "ged", "constant"),
    ],
)
def test_garch_family_fit_matches_the_arch_package(method, distribution, mean):
    arch = pytest.importorskip("arch")
    returns = simulate_garch_family_returns()

    weights = garch_model.get_garch_family_weights(returns, method, distribution, mean)
    reference = _arch_model(arch, returns, method, distribution, mean).fit(disp="off")

    # The weights in arch's order, the mean parameter first.
    variance_count = len(garch_model.GARCH_FAMILY_VARIANCE_PARAMETERS[method])
    mean_count = len(garch_model.GARCH_FAMILY_MEAN_PARAMETERS[mean])
    reordered = np.concatenate(
        [
            weights[variance_count : variance_count + mean_count],
            weights[:variance_count],
            weights[variance_count + mean_count :],
        ]
    )

    assert (
        garch_model.get_garch_family_log_likelihood(
            returns, weights, method, distribution, mean
        )
        >= reference.loglikelihood - 1e-4
    )
    # The likelihood of TGARCH with skewed-t shocks is flat along Gamma and the Shape: arch
    # stops 0.02 below the maximum the multi-start search reaches, 1.5% away in Gamma.
    np.testing.assert_allclose(reordered, reference.params, rtol=2e-2, atol=2e-3)


def test_garch_family_standard_errors_match_the_arch_package():
    arch = pytest.importorskip("arch")
    returns = simulate_garch_family_returns()

    weights = garch_model.get_garch_family_weights(returns, "gjr", "t")
    diagnostics = garch_model.get_garch_family_diagnostics(returns, weights, "gjr", "t")
    model = _arch_model(arch, returns, "gjr", "t", "zero")
    names = ["Omega", "Alpha", "Gamma", "Beta", "Shape"]

    np.testing.assert_allclose(
        [diagnostics[f"{name} Standard Error"] for name in names],
        model.fit(disp="off", cov_type="classic").std_err,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        [diagnostics[f"{name} Robust Standard Error"] for name in names],
        model.fit(disp="off").std_err,
        rtol=1e-2,
    )
    reference = model.fit(disp="off")
    assert diagnostics["Log-Likelihood"] == pytest.approx(
        reference.loglikelihood, rel=1e-6
    )
    # arch reports the information criteria in total, rugarch (and the Toolkit) per observation.
    assert diagnostics["AIC"] == pytest.approx(reference.aic / len(returns), rel=1e-6)
    assert diagnostics["BIC"] == pytest.approx(reference.bic / len(returns), rel=1e-6)


@pytest.mark.parametrize(
    ("distribution", "parameters"),
    [
        ("normal", ()),
        ("t", (5.0,)),
        ("skewt", (6.0, -0.3)),
        ("skewt_fernandez_steel", (6.0, 0.8)),
        ("ged", (1.3,)),
        ("johnson_su", (1.5, 0.7)),
    ],
)
def test_standardized_distributions_have_mean_zero_and_unit_variance(
    distribution, parameters
):
    from scipy import integrate  # noqa: PLC0415

    def density(z):
        return np.exp(
            garch_model.get_standardized_log_density(
                np.array([z]), distribution, parameters
            )[0]
        )

    for power, expected in ((0, 1.0), (1, 0.0), (2, 1.0)):
        assert integrate.quad(lambda z: z**power * density(z), -np.inf, np.inf)[
            0
        ] == pytest.approx(expected, abs=1e-7)

    draws = garch_model.get_standardized_draws(
        distribution, parameters, 200_000, np.random.default_rng(1)
    )
    assert draws.mean() == pytest.approx(0, abs=0.01)
    assert draws.var() == pytest.approx(1, abs=0.02)


def test_johnson_su_and_ged_densities_match_scipy():
    from scipy import stats  # noqa: PLC0415

    z = np.linspace(-4, 4, 17)
    location, scale = garch_model._johnson_su_moments(1.5, 0.7)
    np.testing.assert_allclose(
        garch_model.get_standardized_log_density(z, "johnson_su", (1.5, 0.7)),
        stats.johnsonsu.logpdf(location + scale * z, 0.7, 1.5) + np.log(scale),
    )
    np.testing.assert_allclose(
        garch_model.get_standardized_log_density(z, "ged", (1.3,)),
        stats.gennorm.logpdf(
            z, 1.3, scale=math.sqrt(math.gamma(1 / 1.3) / math.gamma(3 / 1.3))
        ),
    )


def test_hansen_and_fernandez_steel_skewed_t_are_the_same_distribution():
    # Both scale a Student-t differently on each side of its mode, so a Skew lambda of Hansen
    # equals xi = sqrt((1 + lambda) / (1 - lambda)) of Fernández and Steel.
    z = np.linspace(-5, 5, 41)
    skew = -0.3
    np.testing.assert_allclose(
        garch_model.get_standardized_log_density(z, "skewt", (6.0, skew)),
        garch_model.get_standardized_log_density(
            z, "skewt_fernandez_steel", (6.0, np.sqrt((1 + skew) / (1 - skew)))
        ),
    )


def test_family_garch_nests_garch_and_tgarch():
    returns = simulate_garch_family_returns()

    # NAGARCH without a Shift is GARCH(1, 1) and AVGARCH without a Shift is TGARCH.
    assert garch_model.get_garch_family_log_likelihood(
        returns, [0.02, 0.08, 0.0, 0.88], "nagarch"
    ) == pytest.approx(
        garch_model.get_garch_family_log_likelihood(returns, [0.02, 0.08, 0.88]),
        rel=1e-12,
    )
    assert garch_model.get_garch_family_log_likelihood(
        returns, [0.03, 0.08, 0.5, 0.0, 0.88], "avgarch"
    ) == pytest.approx(
        garch_model.get_garch_family_log_likelihood(
            returns, [0.03, 0.08, 0.5, 0.88], "tgarch"
        ),
        rel=1e-12,
    )


def test_component_garch_follows_its_two_recursions():
    returns = simulate_garch_family_returns(n=300)
    omega, alpha, beta, rho, phi = 0.005, 0.06, 0.85, 0.99, 0.03
    backcast = garch_model.get_backcast(returns)

    sigma2, long_run = garch_model._component_variance(
        [omega, alpha, beta, rho, phi], returns, backcast
    )

    variance = level = omega + rho * backcast
    for step, value in enumerate(returns):
        assert sigma2[step] == pytest.approx(variance, rel=1e-10)
        assert long_run[step] == pytest.approx(level, rel=1e-10)
        new_level = omega + rho * level + phi * (value**2 - variance)
        variance = new_level + alpha * (value**2 - level) + beta * (variance - level)
        level = new_level


@pytest.mark.parametrize(
    ("method", "distribution", "mean", "weights"),
    [
        ("gjr", "skewt", "zero", [0.02, 0.06, 0.1, 0.86, 6.0, -0.2]),
        ("component", "t", "zero", [0.007, 0.077, 0.869, 0.99, 0.031, 6.0]),
        ("nagarch", "normal", "zero", [0.02, 0.095, 0.4, 0.86]),
        ("igarch", "normal", "constant", [0.01, 0.12, 0.05]),
    ],
)
def test_garch_family_forecast_is_the_expected_simulated_variance(
    method, distribution, mean, weights
):
    returns = simulate_garch_family_returns()
    forecast = garch_model.get_garch_family_forecast(
        returns, weights, method, distribution, mean, forecast_steps=10
    )

    variance_weights, _, distribution_weights = garch_model._split_weights(
        weights, method, mean
    )
    backcast = garch_model.get_backcast(garch_model._starting_residuals(returns, mean))
    _, sigma2, long_run, _ = garch_model._garch_family_filter(
        returns, weights, method, mean, backcast
    )
    rng = np.random.default_rng(3)
    paths = 200_000
    variance = np.full(paths, sigma2[-1])
    level = np.full(paths, long_run[-1]) if long_run is not None else None
    step = garch_model._variance_step(method, variance_weights)
    simulated = []
    for _ in range(10):
        simulated.append(variance.mean())
        residual = np.sqrt(variance) * garch_model.get_standardized_draws(
            distribution, distribution_weights, paths, rng
        )
        variance, level = step(variance, residual, level)

    np.testing.assert_allclose(forecast, simulated, rtol=5e-3)


def test_garch_family_diagnostics_and_news_impact():
    weights = [0.02, 0.08, 0.88]
    returns = simulate_garch_family_returns()
    diagnostics = garch_model.get_garch_family_diagnostics(returns, weights)

    assert diagnostics["Persistence"] == pytest.approx(0.96)
    assert diagnostics["Half-Life"] == pytest.approx(np.log(0.5) / np.log(0.96))
    assert diagnostics["Unconditional Variance"] == pytest.approx(0.5)

    news_impact = garch_model.get_news_impact_curve(weights)
    # sigma_(t+1)^2 = Omega + Alpha * z^2 * 0.5 + Beta * 0.5, symmetric for GARCH.
    assert news_impact.loc[2.0] == pytest.approx(0.02 + 0.08 * 4 * 0.5 + 0.88 * 0.5)
    assert news_impact.loc[-2.0] == pytest.approx(news_impact.loc[2.0])

    gjr = garch_model.get_news_impact_curve([0.02, 0.04, 0.1, 0.88], "gjr")
    assert gjr.loc[-2.0] > gjr.loc[2.0]

    # APARCH with Delta = 2 under the normal distribution has E[(|z| - Gamma z)^2] = 1 + Gamma^2.
    assert garch_model.get_garch_family_persistence(
        [0.02, 0.1, 0.3, 0.8, 2.0], "aparch"
    ) == pytest.approx(0.1 * 1.09 + 0.8, rel=1e-9)
    assert np.isnan(
        garch_model.get_garch_family_unconditional_variance([0.01, 0.1], "igarch")
    )


def test_garch_simulation_is_reproducible_with_a_seed():
    returns = simulate_garch_family_returns(n=500)
    weights = [0.02, 0.08, 0.88, 6.0]

    first = garch_model.get_garch_simulation(
        returns, weights, distribution="t", simulations=2000, seed=7
    )
    second = garch_model.get_garch_simulation(
        returns, weights, distribution="t", simulations=2000, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["Mean", "5%", "25%", "50%", "75%", "95%"]
    assert (first["5%"] < first["95%"]).all()


def test_garch_family_refuses_unknown_options():
    with pytest.raises(ValueError, match="distribution"):
        garch_model.get_garch_family_weights(np.ones(10), distribution="cauchy")
    with pytest.raises(ValueError, match="long-run"):
        garch_model.get_garch_family(
            np.ones(10), [1, 0.1, 0.8], output="long_run_variance"
        )


STANDARDIZED_CASES = [
    ("normal", ()),
    ("t", (5.0,)),
    ("skewt", (6.0, -0.3)),
    ("skewt_fernandez_steel", (5.0, 0.8)),
    ("ged", (1.3,)),
    ("johnson_su", (1.5, 0.5)),
]


@pytest.mark.parametrize(("distribution", "parameters"), STANDARDIZED_CASES)
def test_standardized_cdf_quantile_and_expected_shortfall_integrate_the_density(
    distribution, parameters
):
    from scipy import integrate  # noqa: PLC0415

    def density(z):
        return float(
            np.exp(
                garch_model.get_standardized_log_density(
                    np.array([z]), distribution, parameters
                )[0]
            )
        )

    for z in (-2.5, -0.4, 0.0, 1.1):
        assert garch_model.get_standardized_cdf(
            np.array([z]), distribution, parameters
        )[0] == pytest.approx(integrate.quad(density, -np.inf, z)[0], abs=1e-8)

    for probability in (0.01, 0.05, 0.5, 0.9):
        quantile = garch_model.get_standardized_quantile(
            probability, distribution, parameters
        )
        assert garch_model.get_standardized_cdf(
            np.array([quantile]), distribution, parameters
        )[0] == pytest.approx(probability, abs=1e-12)

    quantile = float(
        garch_model.get_standardized_quantile(0.05, distribution, parameters)
    )
    tail_mean = integrate.quad(lambda z: z * density(z), -np.inf, quantile)[0] / 0.05
    assert garch_model.get_standardized_expected_shortfall(
        0.05, distribution, parameters
    ) == pytest.approx(tail_mean, rel=1e-7)


@pytest.mark.parametrize(
    ("distribution", "parameters", "arch_distribution"),
    [
        ("normal", (), "Normal"),
        ("t", (5.0,), "StudentsT"),
        ("skewt", (6.0, -0.3), "SkewStudent"),
        ("ged", (1.3,), "GeneralizedError"),
    ],
)
def test_standardized_quantile_matches_the_arch_package(
    distribution, parameters, arch_distribution
):
    univariate = pytest.importorskip("arch.univariate")
    reference = getattr(univariate, arch_distribution)()

    for probability in (0.01, 0.05, 0.3, 0.7):
        expected = reference.ppf(probability, list(parameters) or None)
        assert garch_model.get_standardized_quantile(
            probability, distribution, parameters
        ) == pytest.approx(float(np.asarray(expected)), rel=1e-9)


def test_rolling_forecast_refits_on_the_returns_before_each_forecast():
    returns = pd.Series(
        simulate_garch_family_returns(n=400) / 100,
        index=pd.bdate_range("2020-01-01", periods=400),
    )
    forecast = garch_model.get_garch_family_rolling_forecast(
        returns, window_size=300, refit_every=40, method="garch", mean="constant"
    )

    # One forecast per return from the end of the first window, the last past the final return.
    assert len(forecast) == 101
    assert forecast.index[0] == returns.index[299]
    assert forecast.index[-1] == returns.index[-1]

    # The forecast made on the last day of a window is the one-step variance of the model
    # fitted on that window, so it does not use any later return.
    first_fit = garch_model.get_garch_family_weights(
        returns.to_numpy()[:300], "garch", "normal", "constant"
    )
    expected = garch_model.get_garch_family_forecast(
        returns.to_numpy()[:300], first_fit, "garch", "normal", "constant", 1
    )[0]
    assert forecast["Volatility"].iloc[0] ** 2 == pytest.approx(expected, rel=1e-9)
    assert forecast["Mean"].iloc[0] == pytest.approx(first_fit[3])

    changed = returns.copy()
    changed.iloc[350:] *= 3
    altered = garch_model.get_garch_family_rolling_forecast(
        changed, window_size=300, refit_every=40, method="garch", mean="constant"
    )
    pd.testing.assert_series_equal(
        forecast["Volatility"].iloc[:51], altered["Volatility"].iloc[:51]
    )


def test_warm_started_fit_reaches_the_multi_start_maximum():
    returns = simulate_garch_family_returns(n=800)
    full = garch_model.get_garch_family_weights(returns[:760], "gjr", "t", "constant")
    warm = garch_model.get_garch_family_weights(
        returns[40:], "gjr", "t", "constant", starting_weights=full
    )
    multi_start = garch_model.get_garch_family_weights(
        returns[40:], "gjr", "t", "constant"
    )

    assert garch_model.get_garch_family_log_likelihood(
        returns[40:], warm, "gjr", "t", "constant"
    ) == pytest.approx(
        garch_model.get_garch_family_log_likelihood(
            returns[40:], multi_start, "gjr", "t", "constant"
        ),
        abs=1e-4,
    )


def test_every_rolling_refit_reaches_the_multi_start_maximum():
    # A series whose first window peaks at Beta = 0: a local search from those weights stays
    # in that corner while the later windows peak inside the bounds.
    rng = np.random.default_rng(2)
    returns = np.empty(400)
    variance = 1e-4
    for step in range(400):
        returns[step] = 0.0003 + np.sqrt(variance) * rng.standard_t(6) * np.sqrt(4 / 6)
        variance = 2e-6 + 0.08 * (returns[step] - 0.0003) ** 2 + 0.9 * variance

    def log_likelihood(window, weights):
        backcast = garch_model.get_backcast(
            garch_model._starting_residuals(window, "constant")
        )
        return garch_model.get_garch_family_log_likelihood(
            window, weights, "garch", "normal", "constant", backcast
        )

    previous = garch_model.get_garch_family_weights(
        returns[:200], "garch", "normal", "constant"
    )
    for end in range(250, 401, 50):
        window = returns[end - 200 : end]
        warm = garch_model.get_garch_family_weights(
            window, "garch", "normal", "constant", starting_weights=previous
        )
        multi_start = garch_model.get_garch_family_weights(
            window, "garch", "normal", "constant"
        )
        # At least as high as the multi-start: on some platforms the multi-start's own polish
        # stops a little short (by about 2e-3) and the warm refit is the higher of the two.
        assert (
            log_likelihood(window, warm) >= log_likelihood(window, multi_start) - 1e-6
        )
        previous = warm


def simulate_figarch(phi=0.1, d=0.45, beta=0.35, omega=1e-6, n=1500, seed=3):
    """A FIGARCH(1, d, 1) path from its ARCH(∞) form, with 1,000 burn-in returns."""
    rng = np.random.default_rng(seed)
    weights = garch_model._figarch_weights(phi, d, beta)
    squared = np.full(1000, 1e-4)
    returns = []
    for _ in range(n + 1000):
        variance = omega / (1 - beta) + weights @ squared[::-1]
        returns.append(np.sqrt(variance) * rng.standard_normal())
        squared = np.append(squared[1:], returns[-1] ** 2)
    return np.array(returns[1000:])


def test_figarch_weights_follow_the_expansion():
    weights = garch_model._figarch_weights(0.2, 0.4, 0.5)

    # lambda_1 = D + Phi - Beta and lambda_2 = Beta * lambda_1 + (1 - D) / 2 * D - Phi * D.
    assert weights[0] == pytest.approx(0.4 + 0.2 - 0.5)
    assert weights[1] == pytest.approx(0.5 * weights[0] + 0.3 * 0.4 - 0.2 * 0.4)
    assert len(weights) == garch_model.FIGARCH_TRUNCATION
    # D = 0 is GARCH(1, 1) with Alpha = Phi - Beta: lambda_k = Alpha * Beta^(k - 1).
    garch = garch_model._figarch_weights(0.9, 0.0, 0.8)
    np.testing.assert_allclose(garch[:5], 0.1 * 0.8 ** np.arange(5), rtol=1e-12)


def test_figarch_matches_the_arch_package():
    pytest.importorskip("arch")
    from arch.univariate import FIGARCH, ZeroMean  # noqa: PLC0415

    returns = simulate_figarch()
    model = ZeroMean(returns * 100, rescale=False)
    model.volatility = FIGARCH()

    # The likelihood of fixed weights: arch's of the returns in percent, less n ln(100).
    fixed = [1e-6, 0.2, 0.4, 0.5]
    reference = model.fix([1e-2, 0.2, 0.4, 0.5]).loglikelihood + len(returns) * np.log(
        100
    )
    assert garch_model.get_garch_family_log_likelihood(
        returns, fixed, "figarch"
    ) == pytest.approx(reference, rel=1e-10)

    # The fit stays inside arch's bounds on this path, so both reach the same maximum.
    weights = garch_model.get_garch_family_weights(returns, "figarch")
    fit = model.fit(disp="off", options={"ftol": 1e-12, "maxiter": 1000})
    expected = fit.params.to_numpy() * [1e-4, 1, 1, 1]
    np.testing.assert_allclose(weights, expected, rtol=1e-3, atol=1e-6)

    forecast = garch_model.get_garch_family_forecast(
        returns, expected, "figarch", forecast_steps=5
    )
    np.testing.assert_allclose(
        forecast,
        model.fix(fit.params).forecast(horizon=5).variance.to_numpy()[-1] / 1e4,
        rtol=1e-8,
    )


def test_figarch_simulation_starts_from_the_forecast():
    returns = simulate_figarch(n=800)
    weights = [1e-6, 0.1, 0.45, 0.35]

    simulation = garch_model.get_garch_simulation(
        returns,
        weights,
        "figarch",
        forecast_steps=3,
        simulations=20000,
        output="volatility",
    )
    forecast = garch_model.get_garch_family_forecast(
        returns, weights, "figarch", forecast_steps=3
    )

    # The first volatility is the forecast itself; later the mean simulated variance is the forecast.
    assert simulation["Mean"].iloc[0] == pytest.approx(np.sqrt(forecast[0]), rel=1e-12)
    rng = np.random.default_rng(1)
    paths = garch_model._simulate_paths(
        weights,
        "figarch",
        "zero",
        garch_model._simulation_state(returns, weights, "figarch", "zero", 3, None),
        lambda size: (rng.standard_normal(size), None),
        3,
        200000,
    )
    np.testing.assert_allclose(
        (paths["volatility"] ** 2).mean(axis=1), forecast, rtol=5e-3
    )
    assert garch_model.get_garch_family_persistence(weights, "figarch") == 1.0
    assert np.isnan(
        garch_model.get_garch_family_unconditional_variance(weights, "figarch")
    )


def _realized_log_likelihood(returns, measure, weights):
    """The realized GARCH likelihood, written out step by step."""
    omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u = weights
    weights_ewma = 0.94 ** np.arange(75)
    log_variance = (
        omega
        + beta * np.log(np.sum(returns[:75] ** 2 * weights_ewma) / weights_ewma.sum())
        + gamma * np.log(np.sum(measure[:75] * weights_ewma) / weights_ewma.sum())
    )
    total = 0.0
    for value, x in zip(returns, measure, strict=True):
        z = value / np.exp(log_variance / 2)
        u = np.log(x) - xi - phi * log_variance - tau_1 * z - tau_2 * (z * z - 1)
        total += -0.5 * (np.log(2 * np.pi) + log_variance + z * z)
        total += -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma_u) + (u / sigma_u) ** 2)
        log_variance = omega + beta * log_variance + gamma * np.log(x)
    return total


def simulate_realized_garch(n=1500, seed=5):
    rng = np.random.default_rng(seed)
    beta, gamma, xi, phi, tau_1, tau_2, sigma_u = (
        0.55,
        0.41,
        -0.18,
        1.04,
        -0.07,
        0.07,
        0.38,
    )
    omega = np.log(1e-4) * (1 - beta - gamma * phi) - gamma * xi
    log_variance = np.log(1e-4)
    returns, measure = np.empty(n), np.empty(n)
    for step in range(n):
        z = rng.standard_normal()
        returns[step] = np.exp(log_variance / 2) * z
        log_measure = xi + phi * log_variance + tau_1 * z + tau_2 * (z * z - 1)
        log_measure += sigma_u * rng.standard_normal()
        measure[step] = np.exp(log_measure)
        log_variance = omega + beta * log_variance + gamma * log_measure
    return returns, measure, [omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u]


def test_realized_garch_log_likelihood_and_fit():
    returns, measure, true = simulate_realized_garch()

    assert garch_model.get_garch_family_log_likelihood(
        returns, true, "realized", realized_measure=measure
    ) == pytest.approx(_realized_log_likelihood(returns, measure, true), rel=1e-12)

    weights = garch_model.get_garch_family_weights(
        returns, "realized", realized_measure=measure
    )
    fitted = garch_model.get_garch_family_log_likelihood(
        returns, weights, "realized", realized_measure=measure
    )
    assert fitted >= garch_model.get_garch_family_log_likelihood(
        returns, true, "realized", realized_measure=measure
    )
    # The persistence and the measurement equation are well identified on 1,500 days.
    assert weights[1] + weights[2] * weights[4] == pytest.approx(
        0.55 + 0.41 * 1.04, abs=0.03
    )
    np.testing.assert_allclose(weights[5:], true[5:], atol=0.03)

    # The forecast of the log variance is linear with the persistence Beta + Gamma * Phi.
    forecast = garch_model.get_garch_family_forecast(
        returns, weights, "realized", forecast_steps=3, realized_measure=measure
    )
    omega, beta, gamma, xi, phi = weights[:5]
    assert np.log(forecast[1]) == pytest.approx(
        omega + gamma * xi + (beta + gamma * phi) * np.log(forecast[0]), rel=1e-12
    )

    with pytest.raises(ValueError, match="realized measure"):
        garch_model.get_garch_family_weights(returns, "realized")


def test_realized_garch_fills_a_missing_measure_with_its_expectation():
    returns, measure, true = simulate_realized_garch(n=300)
    gap = measure.copy()
    gap[100] = np.nan

    variance = garch_model.get_garch_family(
        returns, true, "realized", realized_measure=gap
    )
    log_variance = np.log(variance)
    omega, beta, gamma, xi, phi, tau_1, tau_2, _ = true
    z = returns[100] / np.sqrt(variance[100])
    expected = xi + phi * log_variance[100] + tau_1 * z + tau_2 * (z * z - 1)
    assert log_variance[101] == pytest.approx(
        omega + beta * log_variance[100] + gamma * expected, rel=1e-12
    )
    # The day without a measure leaves the measurement equation out of the likelihood.
    per_day = garch_model.get_garch_family_log_likelihood(
        returns, true, "realized", per_observation=True, realized_measure=gap
    )
    assert per_day[100] == pytest.approx(
        -0.5 * (np.log(2 * np.pi) + log_variance[100] + z * z), rel=1e-12
    )


def test_arfima_mean_nests_the_arma_mean_and_follows_its_filter():
    returns = simulate_garch_family_returns(n=600)
    # The ARFIMA Mu is the mean of the returns, the ARMA Mu the intercept Mu * (1 - AR).
    arma = [0.02, 0.08, 0.88, 0.03 * 0.9, 0.1, -0.05]
    arfima = [0.02, 0.08, 0.88, 0.03, 0.1, 0.0, -0.05]

    assert garch_model.get_garch_family_log_likelihood(
        returns, arfima, "garch", "normal", "arfima"
    ) == pytest.approx(
        garch_model.get_garch_family_log_likelihood(
            returns, arma, "garch", "normal", "arma"
        ),
        rel=1e-10,
    )

    # With a fractional difference the residuals follow (1 - AR L)(1 - L)^d (r - Mu) = (1 + MA L) e.
    weights = [0.02, 0.08, 0.88, 0.03, 0.1, 0.3, -0.05]
    residuals, _, _, held_back = garch_model._garch_family_filter(
        returns, weights, "garch", "arfima", 1.0
    )
    pi = [1.0]
    for k in range(1, len(returns)):
        pi.append(pi[-1] * (k - 1 - 0.3) / k)
    centred = returns - 0.03
    differenced = [
        sum(pi[k] * centred[t - k] for k in range(t + 1)) for t in range(len(returns))
    ]
    expected, previous = [], 0.0
    for t in range(1, len(returns)):
        previous = differenced[t] - 0.1 * differenced[t - 1] + 0.05 * previous
        expected.append(previous)
    assert held_back == 1
    np.testing.assert_allclose(residuals, expected, rtol=1e-9, atol=1e-12)


def test_garch_bootstrap_is_reproducible_and_adds_parameter_uncertainty():
    returns = simulate_garch(2e-6, 0.08, 0.90, n=600)
    weights = garch_model.get_garch_weights(returns)

    partial = garch_model.get_garch_simulation(
        returns, weights, output="volatility", simulations=2000, bootstrap_fits=0
    )
    full = garch_model.get_garch_simulation(
        returns, weights, output="volatility", simulations=2000, bootstrap_fits=5
    )

    # Without refits the first volatility is the fitted one on every path; the refits spread it.
    assert partial.iloc[0, 1] == partial.iloc[0, 5]
    assert full.iloc[0, 5] > full.iloc[0, 1]
    pd.testing.assert_frame_equal(
        full,
        garch_model.get_garch_simulation(
            returns, weights, output="volatility", simulations=2000, bootstrap_fits=5
        ),
    )
    with pytest.raises(ValueError, match="bootstrap_fits"):
        garch_model.get_garch_simulation(returns, weights, bootstrap_fits=-1)


def test_garch_family_scores_match_the_analytical_garch_scores():
    returns = simulate_garch(2e-6, 0.08, 0.90, n=1000)
    weights = np.array([3e-6, 0.07, 0.9])
    scores = garch_model.get_garch_family_scores(returns, weights)

    # d l_t / d w = (r_t^2 / sigma_t^2 - 1) / (2 sigma_t^2) * d sigma_t^2 / d w, with
    # d sigma_t^2 / d w = (1, r_(t-1)^2, sigma_(t-1)^2) + Beta * d sigma_(t-1)^2 / d w from the backcast.
    backcast = garch_model.get_backcast(returns)
    variance = weights[0] + (weights[1] + weights[2]) * backcast
    derivative = np.array([1.0, backcast, backcast])
    expected = []
    for value in returns:
        expected.append((value**2 / variance - 1) / (2 * variance) * derivative)
        derivative = np.array([1.0, value**2, variance]) + weights[2] * derivative
        variance = weights[0] + weights[1] * value**2 + weights[2] * variance

    np.testing.assert_allclose(scores, np.array(expected), rtol=1e-5, atol=1e-3)


def test_history_models_refuse_garch_in_mean():
    with pytest.raises(ValueError, match="GARCH-in-mean"):
        garch_model.validate_garch_family("figarch", "normal", "in_mean")
    with pytest.raises(ValueError, match="one-step"):
        garch_model._variance_step("figarch", [0.1, 0.2, 0.4, 0.5])


@pytest.mark.parametrize("method", garch_model.ANALYTIC_SCORE_METHODS)
@pytest.mark.parametrize("mean", garch_model.ANALYTIC_SCORE_MEANS)
@pytest.mark.parametrize(
    "distribution",
    ["normal", "t", "skewt", "skewt_fernandez_steel", "ged", "johnson_su"],
)
def test_analytic_score_matches_finite_differences(method, mean, distribution):
    # The gradient the fits pass to L-BFGS-B is that of the log-likelihood they maximise: central differences of the log-likelihood agree to their own accuracy.  # noqa: E501
    returns = simulate_garch(0.05, 0.08, 0.88, n=800, seed=3)
    variance = {
        "garch": [0.05, 0.08, 0.88],
        "igarch": [0.03, 0.1],
        "gjr": [0.05, 0.04, 0.1, 0.85],
        "aparch": [0.05, 0.08, 0.3, 0.85, 1.6],
        "tgarch": [0.05, 0.08, 0.3, 0.85],
        "ngarch": [0.05, 0.08, 0.85, 1.4],
    }[method]
    mean_weights = {"zero": [], "constant": [0.03], "arma": [0.02, 0.1, -0.05]}[mean]
    shape = {
        "normal": [],
        "t": [7.0],
        "skewt": [7.0, -0.1],
        "skewt_fernandez_steel": [7.0, 0.9],
        "ged": [1.4],
        "johnson_su": [2.0, 0.3],
    }[distribution]
    weights = np.array(variance + mean_weights + shape, dtype=float)
    backcast = garch_model.get_backcast(garch_model._starting_residuals(returns, mean))

    def log_likelihood(values):
        return garch_model.get_garch_family_log_likelihood(
            returns, values, method, distribution, mean, backcast
        )

    step = 1e-6
    numerical = np.array(
        [
            (
                log_likelihood(weights + step * unit)
                - log_likelihood(weights - step * unit)
            )
            / (2 * step)
            for unit in np.eye(len(weights))
        ]
    )
    analytic = garch_model._garch_family_score(
        returns, weights, method, distribution, mean, backcast
    )

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("method", "weights", "objective"),
    [
        ("garch", [2e-6, 0.08, 0.88], garch_model.garch_log_maximization),
        ("gjr", [2e-6, 0.04, 0.1, 0.85], garch_model.gjr_garch_log_maximization),
    ],
)
def test_garch_and_gjr_objective_gradients_match_finite_differences(
    method, weights, objective
):
    # The objectives of get_garch_weights and get_gjr_garch_weights on the scale of daily returns, with a step relative to every weight.  # noqa: E501
    returns = simulate_garch(2e-6, 0.08, 0.88, n=1200, seed=4)
    weights = np.array(weights)
    steps = np.abs(weights) * 1e-6
    numerical = np.array(
        [
            (
                objective(weights + unit * step, returns, 1100)
                - objective(weights - unit * step, returns, 1100)
            )
            / (2 * step)
            for unit, step in zip(np.eye(len(weights)), steps, strict=True)
        ]
    )
    analytic = garch_model._log_maximization_gradient(weights, returns, 1100, method)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5)


def test_a_nearly_integrated_garch_has_no_news_impact_curve():
    # Alpha + Beta = 1 - 10^-9 gives Omega / (1 - P) = 10^3 instead of a level of the data.
    nearly_integrated = [1e-6, 0.05, 0.95 - 1e-9]

    assert garch_model.get_news_impact_curve(nearly_integrated).isna().all()
    assert garch_model.get_news_impact_curve([1e-6, 0.05, 0.9]).notna().all()
