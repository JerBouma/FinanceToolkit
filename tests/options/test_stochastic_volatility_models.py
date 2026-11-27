"""Heston, Bates, Merton, Variance Gamma and Heston-Nandi model tests against reference values."""

import numpy as np
import pytest

from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.options.volatility import (
    fourier_model,
    garch_option_model,
    heston_model,
    levy_model,
)

# Every reference below prices with S = 100, r = 0.04 and q = 0.01 (continuous) and an
# expiration of 182 days on an Actual/365 basis, at these strikes, calls then puts.
STOCK_PRICE, RISK_FREE_RATE, DIVIDEND_YIELD = 100.0, 0.04, 0.01
TIME_TO_EXPIRATION = 182 / 365
STRIKES = np.array([60.0, 90.0, 100.0, 130.0])
HESTON = {"v0": 0.04, "kappa": 2.0, "theta": 0.05, "sigma": 0.6, "rho": -0.7}
JUMPS = {"jump_intensity": 0.5, "jump_mean": -0.1, "jump_volatility": 0.15}

# QuantLib 1.44 AnalyticHestonEngine (relative tolerance 1e-14).
QUANTLIB_HESTON = [
    [40.76767834974016, 13.298216523419482, 6.152874994668645, 0.036753484752526294],
    [0.08021044801992437, 2.0183201505378827, 4.675502464733257, 27.966952483655778],
]
# FinancePy 1.1.2 Heston.value_weber (calls).
FINANCEPY_HESTON_CALLS = [
    40.76767834973049,
    13.298216523419413,
    6.15287499466865,
    0.03675348475251272,
]
# QuantLib 1.44 BatesEngine (relative tolerance 1e-14).
QUANTLIB_BATES = [
    [40.81127605994565, 13.956730312215683, 7.09479784376591, 0.14186944706159688],
    [0.12380815822541358, 2.6768339393340845, 5.617425313830523, 28.07206844596486],
]
# QuantLib 1.44 BatesEngine with the Merton jumps and a nearly constant variance
# (v0 = theta = 0.04, kappa = 1, sigma = 0.001, rho = 0): the Merton price up to the small
# volatility of the variance, as QuantLib's Python bindings leave out the JumpDiffusionEngine.
QUANTLIB_MERTON_LIMIT = [
    [40.719453444815784, 13.458248787749739, 7.11973581289557, 0.4250141882360916],
    [0.03198554309555274, 2.17835241486814, 5.642363282960176, 28.35521318713934],
]
# Variance Gamma calls (sigma = 0.2, theta = -0.15, nu = 0.2) by integrating the conditional
# Black-Scholes price over the gamma clock with scipy's quad, an independent route; and
# QuantLib 1.44 VarianceGammaEngine, whose own integration agrees within 2e-8.
VARIANCE_GAMMA = {"volatility": 0.2, "drift": -0.15, "variance_rate": 0.2}
GAMMA_CLOCK_CALLS = [
    40.70756355199388,
    13.048686088459647,
    6.320685069072287,
    0.23511381482083718,
]
QUANTLIB_VARIANCE_GAMMA = [
    [40.70756353328724, 13.048686081380353, 6.320685065883618, 0.2351138145644073],
    [0.02009565027667044, 1.7687897156687336, 4.843312539271647, 28.165312805514546],
]

# Each method against the exact price: Lewis' adaptive integral, COS with 1024 terms and the
# Carr-Madan transform with 4096 points and spline interpolation.
METHOD_TOLERANCE = {"lewis": 1e-12, "cos": 5e-9, "fft": 1e-6}


@pytest.mark.parametrize("method", ["lewis", "cos", "fft"])
@pytest.mark.parametrize("put_option", [False, True])
def test_heston_matches_quantlib(method, put_option):
    prices = heston_model.get_heston_price(
        STOCK_PRICE,
        STRIKES,
        RISK_FREE_RATE,
        TIME_TO_EXPIRATION,
        **HESTON,
        dividend_yield=DIVIDEND_YIELD,
        put_option=put_option,
        method=method,
    )
    np.testing.assert_allclose(
        prices, QUANTLIB_HESTON[put_option], rtol=0, atol=METHOD_TOLERANCE[method]
    )


def test_heston_matches_financepy():
    prices = heston_model.get_heston_price(
        STOCK_PRICE,
        STRIKES,
        RISK_FREE_RATE,
        TIME_TO_EXPIRATION,
        **HESTON,
        dividend_yield=DIVIDEND_YIELD,
    )
    np.testing.assert_allclose(prices, FINANCEPY_HESTON_CALLS, rtol=0, atol=1e-10)


def test_heston_keeps_the_principal_branch_for_long_expirations():
    # The little Heston trap: the original form of the characteristic function jumps between
    # branches of the logarithm for long expirations and a high volatility of variance; the
    # form of Albrecher et al. gives a smooth price and put-call parity.
    parameters = {"v0": 0.09, "kappa": 0.5, "theta": 0.09, "sigma": 1.5, "rho": -0.9}
    times = np.linspace(5, 30, 26)
    calls = np.array(
        [
            heston_model.get_heston_price(100.0, 100.0, 0.02, time, **parameters)
            for time in times
        ]
    )
    puts = np.array(
        [
            heston_model.get_heston_price(
                100.0, 100.0, 0.02, time, **parameters, put_option=True
            )
            for time in times
        ]
    )
    np.testing.assert_allclose(
        calls - puts, 100.0 - 100.0 * np.exp(-0.02 * times), atol=1e-10
    )
    assert np.all(np.diff(calls) > 0)
    # QuantLib 1.44 AnalyticHestonEngine for 5 and 30 years.
    np.testing.assert_allclose(
        calls[[0, -1]], [19.253401161016342, 59.52346842759482], atol=1e-12
    )
    # The fat left tail widens the COS range: 1024 terms are 1e-3 off, 4096 within 1e-7.
    for time, expected in ((5.0, 19.253401161016342), (30.0, 59.52346842759482)):

        def characteristic_function(u, time=time):
            return heston_model.get_heston_characteristic_function(
                u, time, *parameters.values()
            )

        for terms, tolerance in ((1024, 1e-3), (4096, 1e-7)):
            assert fourier_model.get_fourier_price(
                characteristic_function,
                100.0,
                100.0,
                0.02,
                time,
                method="cos",
                cos_terms=terms,
            ) == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("method", ["lewis", "cos", "fft"])
@pytest.mark.parametrize("put_option", [False, True])
def test_bates_matches_quantlib(method, put_option):
    prices = heston_model.get_bates_price(
        STOCK_PRICE,
        STRIKES,
        RISK_FREE_RATE,
        TIME_TO_EXPIRATION,
        **HESTON,
        **JUMPS,
        dividend_yield=DIVIDEND_YIELD,
        put_option=put_option,
        method=method,
    )
    np.testing.assert_allclose(
        prices, QUANTLIB_BATES[put_option], rtol=0, atol=10 * METHOD_TOLERANCE[method]
    )


def test_bates_without_jumps_is_heston():
    np.testing.assert_allclose(
        heston_model.get_bates_price(
            STOCK_PRICE,
            STRIKES,
            RISK_FREE_RATE,
            TIME_TO_EXPIRATION,
            **HESTON,
            jump_intensity=0.0,
            jump_mean=-0.1,
            jump_volatility=0.15,
        ),
        heston_model.get_heston_price(
            STOCK_PRICE, STRIKES, RISK_FREE_RATE, TIME_TO_EXPIRATION, **HESTON
        ),
        atol=1e-13,
    )


@pytest.mark.parametrize("put_option", [False, True])
def test_merton_series_matches_the_fourier_methods_and_quantlib(put_option):
    arguments = {
        "stock_price": STOCK_PRICE,
        "strike_price": STRIKES,
        "risk_free_rate": RISK_FREE_RATE,
        "volatility": 0.2,
        "time_to_expiration": TIME_TO_EXPIRATION,
        **JUMPS,
        "dividend_yield": DIVIDEND_YIELD,
        "put_option": put_option,
    }
    series = levy_model.get_merton_jump_diffusion_price(**arguments)
    for method in ("lewis", "cos", "fft"):
        np.testing.assert_allclose(
            levy_model.get_merton_jump_diffusion_price(**arguments, method=method),
            series,
            atol=METHOD_TOLERANCE[method],
        )
    # QuantLib's Bates limit carries a volatility of variance of 0.001.
    np.testing.assert_allclose(series, QUANTLIB_MERTON_LIMIT[put_option], atol=5e-6)
    # Our Bates model at the same limit agrees with QuantLib's to its integration precision.
    np.testing.assert_allclose(
        heston_model.get_bates_price(
            STOCK_PRICE,
            STRIKES,
            RISK_FREE_RATE,
            TIME_TO_EXPIRATION,
            v0=0.04,
            kappa=1.0,
            theta=0.04,
            sigma=1e-3,
            rho=0.0,
            **JUMPS,
            dividend_yield=DIVIDEND_YIELD,
            put_option=put_option,
        ),
        QUANTLIB_MERTON_LIMIT[put_option],
        atol=1e-9,
    )


def test_merton_without_jumps_is_black_scholes():
    np.testing.assert_allclose(
        levy_model.get_merton_jump_diffusion_price(
            STOCK_PRICE,
            STRIKES,
            RISK_FREE_RATE,
            0.2,
            TIME_TO_EXPIRATION,
            jump_intensity=0.0,
            jump_mean=-0.1,
            jump_volatility=0.15,
            dividend_yield=DIVIDEND_YIELD,
        ),
        get_black_scholes(
            STOCK_PRICE,
            STRIKES,
            RISK_FREE_RATE,
            0.2,
            TIME_TO_EXPIRATION,
            DIVIDEND_YIELD,
        ),
        atol=1e-13,
    )


@pytest.mark.parametrize("method", ["lewis", "cos", "fft"])
def test_variance_gamma_matches_the_gamma_clock_integral_and_quantlib(method):
    calls = levy_model.get_variance_gamma_price(
        STOCK_PRICE,
        STRIKES,
        RISK_FREE_RATE,
        time_to_expiration=TIME_TO_EXPIRATION,
        **VARIANCE_GAMMA,
        dividend_yield=DIVIDEND_YIELD,
        method=method,
    )
    np.testing.assert_allclose(calls, GAMMA_CLOCK_CALLS, atol=METHOD_TOLERANCE[method])
    puts = levy_model.get_variance_gamma_price(
        STOCK_PRICE,
        STRIKES,
        RISK_FREE_RATE,
        time_to_expiration=TIME_TO_EXPIRATION,
        **VARIANCE_GAMMA,
        dividend_yield=DIVIDEND_YIELD,
        put_option=True,
        method=method,
    )
    np.testing.assert_allclose(
        calls, QUANTLIB_VARIANCE_GAMMA[0], atol=METHOD_TOLERANCE[method] + 5e-8
    )
    np.testing.assert_allclose(
        puts, QUANTLIB_VARIANCE_GAMMA[1], atol=METHOD_TOLERANCE[method] + 5e-8
    )


def test_variance_gamma_rejects_an_infinite_forward():
    with pytest.raises(ValueError):
        levy_model.get_variance_gamma_price(
            100.0, 100.0, 0.04, 0.6, 1.0, drift=0.5, variance_rate=2.0
        )


@pytest.mark.parametrize("method", ["lewis", "cos", "fft"])
def test_the_fourier_pricer_gives_black_scholes_for_a_normal_log_return(method):
    def normal(u):
        return np.exp(-0.5 * 0.2**2 * TIME_TO_EXPIRATION * (1j * u + u * u))

    for put_option in (False, True):
        np.testing.assert_allclose(
            fourier_model.get_fourier_price(
                normal,
                STOCK_PRICE,
                STRIKES,
                RISK_FREE_RATE,
                TIME_TO_EXPIRATION,
                DIVIDEND_YIELD,
                put_option,
                method=method,
            ),
            get_black_scholes(
                STOCK_PRICE,
                STRIKES,
                RISK_FREE_RATE,
                0.2,
                TIME_TO_EXPIRATION,
                DIVIDEND_YIELD,
                put_option,
            ),
            atol=METHOD_TOLERANCE[method],
        )


def test_the_fourier_pricer_returns_the_exercise_value_at_expiration_and_checks_its_inputs():
    assert heston_model.get_heston_price(100.0, 90.0, 0.04, 0.0, **HESTON) == 10.0
    assert (
        heston_model.get_heston_price(100.0, 90.0, 0.04, 0.0, **HESTON, put_option=True)
        == 0.0
    )
    with pytest.raises(ValueError):
        heston_model.get_heston_price(100.0, 90.0, 0.04, 1.0, **HESTON, method="quad")
    with pytest.raises(ValueError):
        heston_model.get_heston_price(100.0, 90.0, 0.04, 1.0, **{**HESTON, "rho": -1.5})
    with pytest.raises(TypeError):
        heston_model.get_heston_price(100.0, 90.0, "0.04", 1.0, **HESTON)


@pytest.mark.parametrize("method", ["lewis", "cos", "fft"])
def test_the_prices_are_stable_under_tiny_perturbations(method):
    prices = heston_model.get_bates_price(
        STOCK_PRICE,
        STRIKES,
        RISK_FREE_RATE,
        TIME_TO_EXPIRATION,
        **HESTON,
        **JUMPS,
        method=method,
    )
    perturbed = heston_model.get_bates_price(
        STOCK_PRICE * (1 + 1e-12),
        STRIKES,
        RISK_FREE_RATE + 1e-12,
        TIME_TO_EXPIRATION,
        **{name: value * (1 + 1e-12) for name, value in HESTON.items()},
        **JUMPS,
        method=method,
    )
    np.testing.assert_allclose(prices, perturbed, atol=1e-9)


# A Heston surface priced by QuantLib's AnalyticHestonEngine (v0 = 0.04, kappa = 1.5,
# theta = 0.06, sigma = 0.5, rho = -0.7) as Black implied volatilities at the strikes
# 80, 85, ..., 120, by days to expiration. QuantLib's HestonModelHelper calibration
# (Levenberg-Marquardt) recovers theta 0.0599999, kappa 1.5000044, sigma 0.5000002,
# rho -0.6999999 and v0 0.0399999 from it.
CALIBRATION_STRIKES = np.arange(80.0, 121.0, 5.0)
CALIBRATION_SURFACE = {
    30: [
        0.27673333630106434,
        0.2587663685012525,
        0.24011439242848645,
        0.22050866112303832,
        0.1998077847312059,
        0.17899736232783633,
        0.16307811979907225,
        0.15620543018960176,
        0.15529042263219167,
    ],
    91: [
        0.2717473783157009,
        0.25473921441616254,
        0.237214623098533,
        0.21900017762409427,
        0.2001138941014388,
        0.18145880569525238,
        0.16606871771715961,
        0.1573135208544027,
        0.15443598494979738,
    ],
    182: [
        0.26536915899818525,
        0.249818687592727,
        0.23406905143671425,
        0.2181202565504074,
        0.2021563460498542,
        0.1867547921123543,
        0.17313141721680902,
        0.16291874674144202,
        0.15695416525148342,
    ],
    365: [
        0.25695083277174185,
        0.2443000171309636,
        0.23187697033560264,
        0.21972568459433778,
        0.20794537741710345,
        0.19671096335897756,
        0.18629165756979185,
        0.17704579010153662,
        0.16935458280156285,
    ],
}


def test_the_heston_calibration_recovers_the_parameters_of_a_heston_surface():
    parameters = heston_model.get_heston_parameters(
        STOCK_PRICE,
        np.tile(CALIBRATION_STRIKES, 4),
        np.repeat([days / 365 for days in CALIBRATION_SURFACE], 9),
        np.concatenate(list(CALIBRATION_SURFACE.values())),
        RISK_FREE_RATE,
        DIVIDEND_YIELD,
    )
    expected = {"v0": 0.04, "kappa": 1.5, "theta": 0.06, "sigma": 0.5, "rho": -0.7}
    for name, value in expected.items():
        assert parameters[name] == pytest.approx(value, abs=1e-6)
    assert parameters["feller_ratio"] == pytest.approx(
        2 * 1.5 * 0.06 / 0.5**2, abs=1e-5
    )
    assert parameters["rmse"] < 1e-8
    # The far out-of-the-money strikes of the 30-day expiration carry a vega below 5% of
    # the at-the-money vega and are left out.
    assert parameters["quotes"] == 33


def test_the_heston_calibration_needs_enough_quotes():
    with pytest.raises(ValueError):
        heston_model.get_heston_parameters(
            100.0,
            np.array([90.0, 100.0]),
            np.array([0.5, 0.5]),
            np.array([0.2, 0.2]),
            0.04,
        )


# Heston-Nandi (Heston and Nandi 2000, daily parameters) priced for 63 trading days.
HESTON_NANDI = {
    "next_variance": 1.5e-4,
    "omega": 5e-7,
    "alpha": 1.3e-6,
    "beta": 0.89,
    "gamma": 250.0,
    "risk_premium": 2.0,
}
# Monte Carlo of the risk-neutral process: 1,000,000 paths (numpy default_rng(7)) with the
# discounted stock price as a control variate, puts at 80, 95, 100 and 120 with their standard
# errors.
HESTON_NANDI_MONTE_CARLO = [
    (80.0, 0.02735099840574801, 0.0003853133185862715),
    (95.0, 1.219874846149616, 0.0021227365021427953),
    (100.0, 2.881225024491145, 0.0024263396991802324),
    (120.0, 19.064557275641572, 0.00017616156962840725),
]


@pytest.mark.parametrize("strike, expected, standard_error", HESTON_NANDI_MONTE_CARLO)
def test_heston_nandi_matches_a_monte_carlo_simulation(
    strike, expected, standard_error
):
    price = garch_option_model.get_heston_nandi_price(
        STOCK_PRICE,
        strike,
        RISK_FREE_RATE,
        63,
        **HESTON_NANDI,
        dividend_yield=DIVIDEND_YIELD,
        put_option=True,
    )
    assert abs(price - expected) < 3 * standard_error
    for method in ("cos", "fft"):
        assert garch_option_model.get_heston_nandi_price(
            STOCK_PRICE,
            strike,
            RISK_FREE_RATE,
            63,
            **HESTON_NANDI,
            dividend_yield=DIVIDEND_YIELD,
            put_option=True,
            method=method,
        ) == pytest.approx(price, abs=METHOD_TOLERANCE[method])


@pytest.mark.parametrize("method", ["lewis", "cos"])
def test_heston_nandi_with_a_constant_variance_is_black_scholes(method):
    parameters = {
        **HESTON_NANDI,
        "next_variance": 2e-4,
        "omega": 2e-4,
        "alpha": 0.0,
        "beta": 0.0,
    }
    for put_option in (False, True):
        np.testing.assert_allclose(
            garch_option_model.get_heston_nandi_price(
                STOCK_PRICE,
                STRIKES,
                RISK_FREE_RATE,
                63,
                **parameters,
                dividend_yield=DIVIDEND_YIELD,
                put_option=put_option,
                method=method,
            ),
            get_black_scholes(
                STOCK_PRICE,
                STRIKES,
                RISK_FREE_RATE,
                np.sqrt(2e-4 * 252),
                63 / 252,
                DIVIDEND_YIELD,
                put_option,
            ),
            atol=1e-12,
        )


def test_the_heston_nandi_fit_recovers_simulated_parameters():
    rng = np.random.default_rng(1)
    variance, returns = 1e-4, []
    for shock in rng.standard_normal(3000):
        returns.append(2.0 * variance + np.sqrt(variance) * shock)
        variance = (
            5e-7 + 0.89 * variance + 1.3e-6 * (shock - 250.0 * np.sqrt(variance)) ** 2
        )

    parameters = garch_option_model.get_heston_nandi_parameters(np.array(returns))
    # Within about two standard errors of 3,000 daily returns.
    assert parameters["alpha"] == pytest.approx(1.3e-6, rel=0.2)
    assert parameters["beta"] == pytest.approx(0.89, abs=0.05)
    assert parameters["gamma"] == pytest.approx(250.0, rel=0.3)
    assert parameters["persistence"] == pytest.approx(
        0.89 + 1.3e-6 * 250.0**2, abs=0.02
    )
    # The optimum: a direct search from it finds no higher likelihood.
    log_likelihood, _ = garch_option_model.get_heston_nandi_log_likelihood(
        np.array(returns),
        parameters["omega"],
        parameters["alpha"],
        parameters["beta"],
        parameters["gamma"] * 1.01,
        parameters["lambda"],
        garch_option_model.get_backcast(np.array(returns)),
    )
    assert log_likelihood < parameters["log_likelihood"]
    with pytest.raises(ValueError):
        garch_option_model.get_heston_nandi_parameters(np.array(returns[:50]))


def test_heston_moment_explosion_time_matches_the_riccati_equation():
    from scipy.integrate import solve_ivp  # noqa: PLC0415

    # B' = σ²B²/2 + (ρσp - κ)B + p(p - 1)/2 from B(0) = 0 blows up at the explosion time.
    for order, kappa, sigma, rho in ((2.5, 1.0, 1.0, 0.99), (4.0, 1.0, 1.0, 0.3)):

        def blow_up(_time, value):
            return value[0] - 1e8

        blow_up.terminal = True
        solution = solve_ivp(
            lambda _time, value, order=order, kappa=kappa, sigma=sigma, rho=rho: [
                0.5 * sigma**2 * value[0] ** 2
                + (rho * sigma * order - kappa) * value[0]
                + 0.5 * order * (order - 1)
            ],
            [0, 50],
            [0.0],
            events=blow_up,
            rtol=1e-11,
            atol=1e-12,
        )
        assert heston_model.get_heston_moment_explosion_time(
            order, kappa, sigma, rho
        ) == pytest.approx(solution.t_events[0][0], rel=1e-6)

    # A fast mean reversion against a negative correlation keeps every moment finite, and
    # moments of an order up to 1 never explode.
    assert np.isinf(heston_model.get_heston_moment_explosion_time(2.5, 2.0, 0.5, -0.7))
    assert np.isinf(heston_model.get_heston_moment_explosion_time(0.5, 0.1, 5.0, 0.9))


def test_heston_and_bates_fft_fall_back_to_lewis_when_the_moment_explodes(caplog):
    # ρ = 0.99 and σ = 1: the Carr-Madan transform was 0.57 off at K = 40 and priced puts
    # below zero, as E[S_T^2.5] nearly explodes at T = 1.
    parameters = {"v0": 0.04, "kappa": 1.0, "theta": 0.04, "sigma": 1.0, "rho": 0.99}
    strikes = np.array([40.0, 100.0, 250.0])
    lewis = heston_model.get_heston_price(100.0, strikes, 0.03, 1.0, **parameters)
    fft = heston_model.get_heston_price(
        100.0, strikes, 0.03, 1.0, **parameters, method="fft"
    )
    np.testing.assert_allclose(fft, lewis, rtol=0, atol=1e-12)
    assert "priced with the Lewis method instead" in caplog.text

    bates = {"jump_intensity": 0.5, "jump_mean": -0.1, "jump_volatility": 0.15}
    np.testing.assert_allclose(
        heston_model.get_bates_price(
            100.0, strikes, 0.03, 1.0, **parameters, **bates, method="fft"
        ),
        heston_model.get_bates_price(100.0, strikes, 0.03, 1.0, **parameters, **bates),
        rtol=0,
        atol=1e-12,
    )

    # Where the moment is finite with room to spare the transform is kept, within 1e-6.
    assert heston_model.get_heston_price(
        STOCK_PRICE, STRIKES, RISK_FREE_RATE, TIME_TO_EXPIRATION, **HESTON, method="fft"
    ) == pytest.approx(
        heston_model.get_heston_price(
            STOCK_PRICE, STRIKES, RISK_FREE_RATE, TIME_TO_EXPIRATION, **HESTON
        ),
        abs=1e-6,
    )


def test_variance_gamma_fft_falls_back_to_cos_when_the_moment_explodes(caplog):
    # 1 - θν·4 - σ²ν·16/2 < 0: the transform was 0.1 off.
    arguments = (100.0, np.array([60.0, 100.0, 150.0]), 0.03, 0.4, 1.0, 0.1, 1.0)
    np.testing.assert_allclose(
        levy_model.get_variance_gamma_price(*arguments, method="fft"),
        levy_model.get_variance_gamma_price(*arguments, method="cos"),
        rtol=0,
        atol=1e-12,
    )
    assert "priced with the COS method instead" in caplog.text


def test_cos_terms_grow_until_the_series_converges():
    # A volatile variance with a slow mean reversion over five years: 1,024 terms were 2e-2
    # off. The Lewis integral agrees with QuantLib's AnalyticHestonEngine within 1e-12.
    for time in (1.0, 5.0):

        def characteristic_function(u, time=time):
            return heston_model.get_heston_characteristic_function(
                u, time, 0.04, 0.5, 0.04, 1.5, -0.9
            )

        strikes = np.array([40.0, 80.0, 100.0, 120.0, 250.0])
        lewis = fourier_model.get_fourier_price(
            characteristic_function, 100.0, strikes, 0.03, time
        )
        cos = fourier_model.get_fourier_price(
            characteristic_function, 100.0, strikes, 0.03, time, method="cos"
        )
        np.testing.assert_allclose(cos, lewis, rtol=0, atol=1e-6)
        fixed = fourier_model.get_fourier_price(
            characteristic_function,
            100.0,
            strikes,
            0.03,
            time,
            method="cos",
            cos_terms=1024,
            cos_tolerance=None,
        )
        assert np.abs(fixed - lewis).max() > 1e-4


def test_fourier_prices_stay_above_the_no_arbitrage_bound():
    # A deep out-of-the-money option a day from expiration: the integration noise gave
    # -1e-13 (Lewis, COS) and -2e-7 (FFT).
    def characteristic_function(u):
        return heston_model.get_heston_characteristic_function(
            u, 1 / 365, 0.04, 2.0, 0.04, 0.5, -0.7
        )

    strikes = np.array([40.0, 60.0, 80.0, 120.0, 150.0, 250.0])
    for method in fourier_model.FOURIER_METHODS:
        for put_option in (False, True):
            prices = fourier_model.get_fourier_price(
                characteristic_function,
                100.0,
                strikes,
                0.03,
                1 / 365,
                put_option=put_option,
                method=method,
            )
            sign = -1 if put_option else 1
            bound = np.maximum(sign * (100.0 - strikes * np.exp(-0.03 / 365)), 0.0)
            assert (prices >= bound).all()


def test_heston_calibration_recovers_a_fat_tailed_surface():
    # κ = 0.3 and σ = 1.2: with 256 COS terms the prices were 2e-2 off at a year and the fit
    # returned κ = 0.249 and θ = 0.0562 with an RMSE of 0.001 on this exact surface.
    from scipy.optimize import brentq  # noqa: PLC0415

    truth = (0.04, 0.3, 0.05, 1.2, -0.8)
    strikes, times, volatilities = [], [], []
    for time in (1 / 12, 0.25, 0.5, 1.0, 1.5):
        expiry_strikes = 100 * np.exp(np.linspace(-0.35, 0.25, 9) * np.sqrt(time / 0.5))
        prices = heston_model.get_heston_price(
            100.0, expiry_strikes, 0.03, time, *truth, dividend_yield=0.01
        )
        for strike, price in zip(expiry_strikes, prices, strict=True):
            strikes.append(strike)
            times.append(time)
            volatilities.append(
                brentq(
                    lambda volatility, strike=strike, price=price, time=time: (
                        get_black_scholes(100.0, strike, 0.03, volatility, time, 0.01)
                        - price
                    ),
                    1e-4,
                    5.0,
                )
            )

    fit = heston_model.get_heston_parameters(
        100.0, np.array(strikes), np.array(times), np.array(volatilities), 0.03, 0.01
    )

    for name, value in zip(
        ("v0", "kappa", "theta", "sigma", "rho"), truth, strict=True
    ):
        assert fit[name] == pytest.approx(value, rel=1e-3)
    assert fit["rmse"] < 1e-5
