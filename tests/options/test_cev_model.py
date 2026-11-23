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
