"""Constant Elasticity of Variance Model Tests"""

import numpy as np
import pytest

from financetoolkit.options.pricing import black_scholes_model, cev_model

# Each case: stock price, strike price, risk-free rate, dividend yield, volatility, days to
# expiration, β, put option and the QuantLib 1.44 AnalyticCEVEngine price. QuantLib prices a
# futures price with a constant coefficient α, which is √(v / T) of get_cev.
QUANTLIB_CASES = [
    (100, 90, 0.04, 0.00, 0.25, 365, 0.5, True, 4.236309585630),
    (100, 110, 0.03, 0.02, 0.30, 730, 0.3, False, 12.792312736130),
    (100, 100, 0.05, 0.00, 0.20, 182, 0.0, True, 4.419968348035),
    (100, 120, 0.02, 0.01, 0.40, 365, 0.8, False, 9.129470881176),
]


@pytest.mark.parametrize("case", QUANTLIB_CASES)
def test_the_cev_price_matches_quantlib(case):
    stock, strike, rate, dividend, volatility, days, beta, put, quantlib = case
    assert cev_model.get_cev(
        float(stock),
        float(strike),
        rate,
        volatility,
        days / 365,
        beta,
        dividend,
        put,
    ) == pytest.approx(quantlib, abs=1e-9)


def test_the_cev_price_matches_a_simulation_with_drift():
    # An Euler simulation of dS = (r - q) * S * dt + δ * S^β * dW, absorbed at zero, with
    # 200,000 paths of 400 steps (NumPy's default_rng, seed 1) gives 13.0739 ± 0.0436 for
    # this call and 10.1080 ± 0.0323 for this put; the formula (13.0309 and 10.1340) lies
    # within one standard error of both.
    call = cev_model.get_cev(100.0, 100.0, 0.05, 0.3, 1.0, 0.5, 0.02)
    put = cev_model.get_cev(100.0, 100.0, 0.05, 0.3, 1.0, 0.5, 0.02, put_option=True)
    assert call == pytest.approx(13.0739, abs=0.0436)
    assert put == pytest.approx(10.1080, abs=0.0323)


def test_the_cev_price_satisfies_put_call_parity_and_tends_to_black_scholes():
    for beta in (0.0, 0.5, 0.9, 0.999):
        call = cev_model.get_cev(100.0, 95.0, 0.05, 0.3, 2.0, beta, 0.02)
        put = cev_model.get_cev(100.0, 95.0, 0.05, 0.3, 2.0, beta, 0.02, True)
        parity = 100 * np.exp(-0.02 * 2) - 95 * np.exp(-0.05 * 2)
        assert call - put == pytest.approx(parity, abs=1e-10)

    black_scholes = float(
        black_scholes_model.get_black_scholes(100.0, 95.0, 0.05, 0.3, 2.0, 0.02)
    )
    assert cev_model.get_cev(100.0, 95.0, 0.05, 0.3, 2.0, 1.0, 0.02) == black_scholes
    assert cev_model.get_cev(100.0, 95.0, 0.05, 0.3, 2.0, 0.999, 0.02) == pytest.approx(
        black_scholes, abs=1e-3
    )


def test_a_lower_elasticity_makes_out_of_the_money_puts_dearer():
    puts = [
        cev_model.get_cev(100.0, 80.0, 0.03, 0.25, 0.5, beta, put_option=True)
        for beta in (1.0, 0.8, 0.5, 0.0)
    ]
    assert puts == sorted(puts)


def test_the_cev_model_refuses_an_elasticity_outside_zero_and_one():
    with pytest.raises(ValueError, match="beta must be between 0 and 1"):
        cev_model.get_cev(100.0, 100.0, 0.05, 0.3, 1.0, 1.5)
    with pytest.raises(ValueError, match="volatility must be positive"):
        cev_model.get_cev(100.0, 100.0, 0.05, 0.0, 1.0)
    assert cev_model.get_cev(100.0, 90.0, 0.05, 0.3, 0.0) == 10.0


def _noncentralities(
    strike, days, beta, stock=100.0, rate=0.04, dividend=0.01, volatility=0.3
):
    """The degrees of freedom and the x and y of get_cev."""
    time = days / 365
    carry = rate - dividend
    growth = 2 * carry * (1 - beta)
    variance = (
        (volatility * stock ** (1 - beta)) ** 2 * np.expm1(growth * time) / growth
    )
    scale = (1 - beta) ** 2 * variance
    forward = stock * np.exp(carry * time)
    return (
        1 / (1 - beta),
        forward ** (2 * (1 - beta)) / scale,
        strike ** (2 * (1 - beta)) / scale,
    )


# The two probabilities of the CEV call, P(χ²(ν + 2, x) > y) and P(χ²(ν, y) ≤ x), from a
# quadrature of the non-central chi-squared density in 40-digit arithmetic (mpmath 1.4).
# scipy's distribution gives NaN for the second at β = 0.9999 and one day, and 0.5881 for the
# first at β = 0.99999.
MPMATH_CASES = [
    (100, 1, 0.9999, 0.505220562537, 0.498956234735),
    (100, 1, 0.99999, 0.50522028059, 0.498955952766),
    (120, 1, 0.99999, 2.11571465435e-31, 1.7604607008e-31),
    (80, 182.5, 0.99999, 0.890405110826, 0.845316971958),
    (120, 182.5, 0.999, 0.247407415396, 0.185451393645),
    (100, 182.5, 0.99, 0.570574676779, 0.486321052683),
]


@pytest.mark.parametrize("strike, days, beta, upper, lower", MPMATH_CASES)
def test_the_chi_squared_probabilities_near_beta_one_match_a_quadrature(
    strike, days, beta, upper, lower
):
    degrees, x, y = _noncentralities(strike, days, beta)
    assert cev_model._noncentral_chi_squared(y, degrees + 2, x, upper=True) == (
        pytest.approx(upper, rel=1e-8, abs=1e-12)
    )
    assert cev_model._noncentral_chi_squared(x, degrees, y, upper=False) == (
        pytest.approx(lower, rel=1e-8, abs=1e-12)
    )


def test_the_price_runs_into_black_scholes_as_beta_approaches_one():
    strikes = np.repeat([50.0, 90, 100, 110, 200], 3)
    times = np.tile([1 / 365, 0.5, 5], 5)
    black_scholes = black_scholes_model.get_black_scholes(
        100, strikes, 0.04, 0.3, times, 0.01
    )
    gaps = []
    for beta in (0.999, 0.9999, 0.99999, 0.999999, 1 - 1e-9):
        values = cev_model.get_cev(100, strikes, 0.04, 0.3, times, beta, 0.01)
        assert np.all(np.isfinite(values))
        gaps.append(np.abs(values - black_scholes).max())
    # The CEV price differs from Black-Scholes by about (1 - β) times its sensitivity to β.
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3] and gaps[0] < 1e-2
    # Within 1e-6 of 1 the elasticity is priced as Black-Scholes.
    assert gaps[4] == 0


def test_a_grid_of_strikes_and_expirations_is_priced_at_once():
    strikes = np.repeat([0.0, 80, 100, 130], 3)
    times = np.tile([0.0, 1 / 365, 1.0], 4)
    for beta in (0.3, 0.995):
        for put_option in (False, True):
            grid = cev_model.get_cev(
                100, strikes, 0.04, 0.3, times, beta, 0.01, put_option
            )
            one_by_one = [
                cev_model.get_cev(
                    100, float(strike), 0.04, 0.3, float(time), beta, 0.01, put_option
                )
                for strike, time in zip(strikes, times, strict=True)
            ]
            np.testing.assert_allclose(grid, one_by_one, rtol=1e-12, atol=1e-12)
    # At a zero strike the call is the discounted forward, at β = 1 as well.
    for beta in (0.5, 1.0):
        assert cev_model.get_cev(100, 0.0, 0.04, 0.3, 1.0, beta, 0.01) == pytest.approx(
            100 * np.exp(-0.01), rel=1e-12
        )
