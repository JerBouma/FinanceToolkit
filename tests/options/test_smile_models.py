"""SABR, SSVI and Dupire local volatility model tests against reference values."""

import numpy as np
import pytest

from financetoolkit.options.volatility import (
    local_volatility_model,
    sabr_model,
    svi_model,
)

# QuantLib 1.44 sabrVolatility (and shiftedSabrVolatility with a shift of 2) for F = 100,
# T = 1, rho = -0.6, nu = 0.8 and alpha = 0.2 * F^(1 - beta): (beta, alpha, strike, vol, shifted vol).
QUANTLIB_SABR = [
    (0.0, 20.0, 60.0, 0.4009651055677761, 0.39078964521356907),
    (0.0, 20.0, 95.0, 0.2237362040438788, 0.219222214603418),
    (0.0, 20.0, 125.0, 0.15592708768100874, 0.15317495150378022),
    (0.5, 2.0, 60.0, 0.3665284471844939, 0.3601819929655815),
    (0.5, 2.0, 95.0, 0.2181729356779186, 0.21585790590058332),
    (0.5, 2.0, 125.0, 0.16282599329446062, 0.16136620886390207),
    (1.0, 0.2, 60.0, 0.3358831956602096, 0.3325288275673458),
    (1.0, 0.2, 95.0, 0.21295813772447525, 0.21269111182163342),
    (1.0, 0.2, 125.0, 0.16991590130400752, 0.16991577333336125),
]


@pytest.mark.parametrize(
    "beta, alpha, strike, expected, expected_shifted", QUANTLIB_SABR
)
def test_sabr_volatility_matches_quantlib(
    beta, alpha, strike, expected, expected_shifted
):
    arguments = {
        "forward_price": 100.0,
        "time_to_expiration": 1.0,
        "alpha": alpha,
        "beta": beta,
        "rho": -0.6,
        "nu": 0.8,
    }
    assert sabr_model.get_sabr_volatility(strike, **arguments) == pytest.approx(
        expected, abs=1e-12
    )
    assert sabr_model.get_sabr_volatility(
        strike, shift=2.0, **arguments
    ) == pytest.approx(expected_shifted, abs=1e-12)


def test_obloj_equals_hagan_at_the_money_and_for_a_lognormal_backbone():
    strikes = np.linspace(60, 160, 41)
    lognormal = {"alpha": 0.25, "beta": 1.0, "rho": -0.4, "nu": 0.9}
    np.testing.assert_allclose(
        sabr_model.get_sabr_volatility(strikes, 100, 2.0, **lognormal, method="obloj"),
        sabr_model.get_sabr_volatility(strikes, 100, 2.0, **lognormal),
        atol=1e-14,
    )

    parameters = {"alpha": 2.0, "beta": 0.5, "rho": -0.4, "nu": 0.9}
    at_the_money = sabr_model.get_sabr_volatility(100.0, 100, 2.0, **parameters)
    assert sabr_model.get_sabr_volatility(
        100.0, 100, 2.0, **parameters, method="obloj"
    ) == pytest.approx(at_the_money, abs=1e-14)
    # Continuous through the forward, where the leading term switches to its series.
    for strike in (100 * (1 + 1e-7), 100 * (1 - 1e-5), 100 * (1 + 1e-3)):
        assert sabr_model.get_sabr_volatility(
            strike, 100, 2.0, **parameters, method="obloj"
        ) == pytest.approx(
            sabr_model.get_sabr_volatility(strike, 100, 2.0, **parameters), abs=1e-5
        )


def test_sabr_volatility_is_nan_for_a_non_positive_shifted_strike():
    volatility = sabr_model.get_sabr_volatility(
        np.array([-0.01, 0.01]), 0.02, 1.0, alpha=0.01, beta=0.5, rho=0.0, nu=0.3
    )
    assert np.isnan(volatility[0]) and np.isfinite(volatility[1])
    assert np.isfinite(
        sabr_model.get_sabr_volatility(
            -0.01, 0.02, 1.0, alpha=0.01, beta=0.5, rho=0.0, nu=0.3, shift=0.03
        )
    )


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_sabr_alpha_reproduces_the_at_the_money_volatility(beta):
    alpha = sabr_model.get_sabr_alpha(0.25, 100.0, 2.0, beta, -0.5, 1.0)
    assert sabr_model.get_sabr_volatility(
        100.0, 100.0, 2.0, alpha, beta, -0.5, 1.0
    ) == pytest.approx(0.25, abs=1e-14)


@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_sabr_fit_recovers_the_parameters_of_a_sabr_smile(beta):
    forward, time = 100.0, 0.5
    true = {"beta": beta, "rho": -0.45, "nu": 1.2}
    true["alpha"] = sabr_model.get_sabr_alpha(0.3, forward, time, **true)
    strikes = np.linspace(70, 130, 25)
    market = sabr_model.get_sabr_volatility(strikes, forward, time, **true)

    fitted = sabr_model.get_sabr_parameters(strikes, market, forward, time, beta=beta)

    assert fitted["beta"] == beta
    for name in ("alpha", "rho", "nu"):
        assert fitted[name] == pytest.approx(true[name], abs=1e-4)


def test_sabr_fit_needs_the_forward_among_the_strikes():
    with pytest.raises(ValueError, match="outside the quoted strikes"):
        sabr_model.get_sabr_parameters(
            np.array([110.0, 120, 130, 140]), np.full(4, 0.2), 100.0, 1.0
        )


def test_ssvi_is_the_raw_svi_slice_it_converts_to():
    log_moneyness = np.linspace(-1.5, 1.5, 61)
    for theta in (0.01, 0.1, 0.6):
        parameters = {"theta": theta, "rho": -0.6, "eta": 1.1, "gamma": 0.4}
        np.testing.assert_allclose(
            svi_model.get_svi_total_variance(
                log_moneyness, **svi_model.convert_ssvi_to_svi(**parameters)
            ),
            svi_model.get_ssvi_total_variance(log_moneyness, **parameters),
            atol=1e-15,
        )
        assert svi_model.get_ssvi_total_variance(0.0, **parameters) == pytest.approx(
            theta
        )


def test_butterfly_function_matches_financepy():
    # FinancePy 1.1.2 volatility_fns.g for an SSVI slice with theta = 0.08 and its
    # Heston-like phi(theta) = 0.49339947..., rho = -0.7.
    theta, phi, rho = 0.08000000000000002, 0.4933994702019883, -0.7
    raw = {
        "a": theta / 2 * (1 - rho**2),
        "b": theta * phi / 2,
        "rho": rho,
        "m": -rho / phi,
        "sigma": np.sqrt(1 - rho**2) / phi,
    }
    assert svi_model.get_svi_butterfly_function(-0.3, **raw) == pytest.approx(
        0.9037961058722639, abs=1e-12
    )
    assert svi_model.get_svi_butterfly_function(0.4, **raw) == pytest.approx(
        1.1517545572829486, abs=1e-12
    )


def test_butterfly_check_flags_the_arbitrage_of_a_known_svi_slice():
    # Axel Vogt's slice, the example of Gatheral and Jacquier (2014) of an SVI slice with a
    # negative density.
    vogt = {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153}
    violations = svi_model.check_butterfly_arbitrage(
        {1.0: vogt}, np.linspace(-1.5, 1.5, 301)
    )
    assert not violations.empty
    assert (violations["g(k)"] < 0).all()

    ssvi = svi_model.convert_ssvi_to_svi(theta=0.05, rho=-0.7, eta=2 / 1.7, gamma=0.5)
    assert svi_model.check_butterfly_arbitrage(
        {1.0: ssvi}, np.linspace(-3, 3, 601)
    ).empty


def test_ssvi_fit_recovers_the_parameters_of_an_ssvi_surface():
    true = {"rho": -0.55, "eta": 0.9, "gamma": 0.35}
    times = [0.05, 0.2, 0.5, 1.0]
    thetas = [0.2**2 * t * (1 + t) for t in times]
    log_moneyness = {t: np.linspace(-0.4, 0.3, 15) * np.sqrt(t) for t in times}
    total_variance = {
        t: svi_model.get_ssvi_total_variance(log_moneyness[t], theta, **true)
        for t, theta in zip(times, thetas)
    }

    fitted = svi_model.get_ssvi_parameters(log_moneyness, total_variance)

    assert sorted(fitted) == times
    for t, theta in zip(times, thetas):
        assert fitted[t]["theta"] == pytest.approx(theta, rel=1e-3)
        for name, value in true.items():
            assert fitted[t][name] == pytest.approx(value, abs=2e-3)


def test_local_volatility_of_a_flat_surface_is_the_flat_volatility():
    flat = {
        t: {"a": 0.2**2 * t, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
        for t in (0.25, 0.5, 1.0)
    }
    local = local_volatility_model.get_local_volatility(
        np.linspace(-0.5, 0.5, 11), flat
    )
    np.testing.assert_allclose(local.to_numpy(), 0.2, atol=1e-14)


def test_local_volatility_of_a_variance_linear_in_time_has_the_closed_form():
    # w(k, T) = T * f(k) with f a raw SVI curve, so dw/dT = f(k) exactly, and the
    # derivatives in k are checked here by finite differences rather than analytically.
    shape = {"a": 0.02, "b": 0.1, "rho": -0.5, "m": 0.05, "sigma": 0.2}
    times = (0.25, 0.6, 1.0, 2.0)
    surface = {t: {**shape, "a": shape["a"] * t, "b": shape["b"] * t} for t in times}
    log_moneyness = np.linspace(-0.6, 0.6, 13)
    local = local_volatility_model.get_local_volatility(log_moneyness, surface)

    step = 1e-4
    for t in times:

        def total_variance(k, t=t):
            return t * svi_model.get_svi_total_variance(k, **shape)

        w = total_variance(log_moneyness)
        first = (
            total_variance(log_moneyness + step) - total_variance(log_moneyness - step)
        ) / (2 * step)
        second = (
            total_variance(log_moneyness + step)
            - 2 * w
            + total_variance(log_moneyness - step)
        ) / step**2
        denominator = (
            1
            - log_moneyness / w * first
            + 0.25 * (-0.25 - 1 / w + log_moneyness**2 / w**2) * first**2
            + 0.5 * second
        )
        expected = np.sqrt(
            svi_model.get_svi_total_variance(log_moneyness, **shape) / denominator
        )
        np.testing.assert_allclose(local[t].to_numpy(), expected, rtol=1e-6)


def test_local_volatility_matches_quantlib_and_financepy():
    # QuantLib 1.44 LocalVolSurface on a BlackVarianceSurface (bicubic, 721 strikes, five-day
    # steps) of the SSVI surface theta_t = 0.25^2 * t, rho = -0.5, eta = 1, gamma = 0.4, spot
    # 100 and r = q = 0, at T = 1.
    quantlib = {
        70: 0.37544359332094507,
        100: 0.24211117674299698,
        140: 0.20579669713983753,
    }
    step = 1e-3
    surface = {
        t: svi_model.convert_ssvi_to_svi(0.25**2 * t, -0.5, 1.0, 0.4)
        for t in (1 - step, 1.0, 1 + step)
    }
    local = local_volatility_model.get_local_volatility(
        np.log(np.array(list(quantlib)) / 100), surface
    )[1.0]
    np.testing.assert_allclose(local.to_numpy(), list(quantlib.values()), atol=1e-5)

    # FinancePy 1.1.2 ssvi_local_varg(x=-0.3, gamma=0.5, sigma=0.4, rho=-0.7, t=0.5): its
    # Heston-like phi with theta_t = 0.4^2 * t, central difference in t with a step of 1e-4.
    def heston_like(t):
        theta = 0.4**2 * t
        phi = (1 / 0.5 / theta) * (1 - (1 - np.exp(-0.5 * theta)) / 0.5 / theta)
        # gamma = 0 makes the power law phi = eta / (1 + theta).
        return svi_model.convert_ssvi_to_svi(theta, -0.7, phi * (1 + theta), 0.0)

    surface = {t: heston_like(t) for t in (0.5 - 1e-4, 0.5, 0.5 + 1e-4)}
    local_variance = (
        local_volatility_model.get_local_volatility(np.array([-0.3]), surface)[
            0.5
        ].iloc[0]
        ** 2
    )
    assert local_variance == pytest.approx(0.19556683416864504, rel=1e-8)


def test_local_volatility_is_nan_where_the_surface_has_calendar_arbitrage():
    shrinking = {
        0.5: {"a": 0.04, "b": 0.1, "rho": -0.3, "m": 0.0, "sigma": 0.2},
        1.0: {"a": 0.02, "b": 0.1, "rho": -0.3, "m": 0.0, "sigma": 0.2},
    }
    local = local_volatility_model.get_local_volatility(np.array([0.0]), shrinking)
    assert local.isna().all().all()

    with pytest.raises(ValueError, match="at least two expiries"):
        local_volatility_model.get_local_volatility(
            np.array([0.0]), {1.0: shrinking[1.0]}
        )
