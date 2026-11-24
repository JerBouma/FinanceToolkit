"""LIBOR Market Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.fixedincome.short_rates import lmm_model

CURVE = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])
TIMES = np.arange(1, 8.0)
VOLATILITIES = np.linspace(0.25, 0.17, 6)


def black(forward, strike, volatility, expiry, sign=1.0):
    deviation = volatility * np.sqrt(expiry)
    d1 = (np.log(forward / strike) + deviation**2 / 2) / deviation
    return sign * (
        forward * norm.cdf(sign * d1) - strike * norm.cdf(sign * (d1 - deviation))
    )


@pytest.mark.parametrize(
    "correlation_decay, factors, option_type",
    [(None, None, "caplet"), (0.1, 3, "floorlet"), (0.1, None, "caplet")],
)
def test_caplets_converge_to_black(correlation_decay, factors, option_type):
    # Every caplet depends on its own forward only, which is lognormal under its payment
    # measure: the Monte Carlo price must match Black's formula within its standard error.
    caplets = lmm_model.get_lmm_cap(
        CURVE,
        TIMES,
        VOLATILITIES,
        0.04,
        correlation_decay=correlation_decay,
        factors=factors,
        option_type=option_type,
        simulations=20000,
    )
    discounts = discount_factors(CURVE, TIMES)
    forwards = (discounts[:-1] / discounts[1:] - 1) / np.diff(TIMES)
    sign = 1.0 if option_type == "caplet" else -1.0
    expected = (
        discounts[1:]
        * np.diff(TIMES)
        * black(forwards, 0.04, VOLATILITIES, TIMES[:-1], sign)
    )

    assert np.all(np.abs(caplets["Price"] - expected) < 4 * caplets["Standard Error"])
    # The standard errors are below 4% of the prices, so the check above is a tight one.
    assert np.all(caplets["Standard Error"] < 0.04 * expected)


def test_a_fixed_first_caplet_is_its_intrinsic_value():
    times = np.array([0.0, 0.5, 1.0])
    caplets = lmm_model.get_lmm_cap(CURVE, times, 0.2, 0.01, simulations=100)
    discounts = discount_factors(CURVE, times)
    forward = (discounts[0] / discounts[1] - 1) / 0.5
    assert caplets["Price"].iloc[0] == pytest.approx(
        discounts[1] * 0.5 * (forward - 0.01), rel=1e-14
    )
    assert caplets["Standard Error"].iloc[0] == pytest.approx(0, abs=1e-15)


def test_one_factor_rebonato_volatility_is_the_forward_volatility():
    # With perfectly correlated forwards of one volatility the swap rate has that volatility.
    assert lmm_model.get_rebonato_swaption_volatility(
        CURVE, [2, 3, 4, 5, 6, 7], 0.2
    ) == pytest.approx(0.2, rel=1e-14)
    # Less correlation lowers it.
    assert (
        lmm_model.get_rebonato_swaption_volatility(
            CURVE, [2, 3, 4, 5, 6, 7], 0.2, correlation_decay=0.1
        )
        < 0.2
    )


@pytest.mark.parametrize(
    "correlation_decay, factors", [(None, None), (0.1, 2), (0.1, None)]
)
def test_swaption_matches_black_with_the_rebonato_volatility(
    correlation_decay, factors
):
    times = np.array([2, 3, 4, 5, 6, 7.0])
    swaption = lmm_model.get_lmm_swaption(
        CURVE,
        times,
        0.2,
        0.04,
        correlation_decay=correlation_decay,
        factors=factors,
        simulations=20000,
    )
    volatility = lmm_model.get_rebonato_swaption_volatility(
        CURVE, times, 0.2, correlation_decay=correlation_decay, factors=factors
    )
    discounts = discount_factors(CURVE, times)
    annuity = np.sum(discounts[1:])
    swap_rate = (discounts[0] - discounts[-1]) / annuity

    # Rebonato's approximation freezes the swap rate's weights: about 1% from the model.
    assert swaption["Price"] == pytest.approx(
        annuity * black(swap_rate, 0.04, volatility, 2.0), rel=0.03
    )


def test_swaption_matches_financepy_monte_carlo():
    # FinancePy 1.1.2 lmm_simulate_fwds_nf (200,000 paths, seed 42) and lmm_swaption_pricer:
    # forwards 3%, 3.1%, ..., 4% on yearly periods from today, ρ_ij = e^(-0.1 |i - j|), all
    # volatilities 20%, a payer at 3.6% expiring in 2 years into a 5-year swap.
    forwards = np.array([0.03, 0.031, 0.033, 0.035, 0.036, 0.038, 0.039, 0.04])
    maturities = np.arange(1, 9.0)
    discounts = 1 / np.cumprod(1 + forwards)
    curve = pd.Series(discounts ** (-1 / maturities) - 1, index=maturities)
    swaption = lmm_model.get_lmm_swaption(
        curve,
        [2, 3, 4, 5, 6, 7],
        0.2,
        0.036,
        correlation_decay=0.1,
        simulations=200000,
        steps_per_year=1,
    )
    assert swaption["Price"] == pytest.approx(
        0.01608273322856879, abs=3 * swaption["Standard Error"]
    )


def test_payer_less_receiver_is_the_forward_swap():
    times = np.array([2, 3, 4, 5, 6, 7.0])
    common = {"correlation_decay": 0.1, "factors": 3, "simulations": 20000}
    payer = lmm_model.get_lmm_swaption(CURVE, times, 0.2, 0.04, **common)
    receiver = lmm_model.get_lmm_swaption(
        CURVE, times, 0.2, 0.04, swaption_type="receiver", **common
    )
    discounts = discount_factors(CURVE, times)
    forward_swap = discounts[0] - discounts[-1] - 0.04 * np.sum(discounts[1:])
    error = np.hypot(payer["Standard Error"], receiver["Standard Error"])
    assert payer["Price"] - receiver["Price"] == pytest.approx(
        forward_swap, abs=4 * error
    )


def test_factor_loadings():
    times = np.arange(1, 12.0)
    np.testing.assert_allclose(
        lmm_model.get_lmm_factor_loadings(times), np.ones((10, 1))
    )
    full = lmm_model.get_lmm_factor_loadings(times, correlation_decay=0.15)
    resets = times[:-1]
    np.testing.assert_allclose(
        full @ full.T,
        np.exp(-0.15 * np.abs(resets[:, None] - resets[None, :])),
        atol=1e-12,
    )
    reduced = lmm_model.get_lmm_factor_loadings(
        times, correlation_decay=0.15, factors=3
    )
    assert reduced.shape == (10, 3)
    np.testing.assert_allclose(np.linalg.norm(reduced, axis=1), 1.0, rtol=1e-14)
    # The first factor is a level shift: every forward loads on it with the same sign.
    assert np.all(reduced[:, 0] > 0)


def test_simulation_is_seeded_and_stable():
    arguments = ([0.03, 0.032, 0.034], [1, 2, 3, 4], [0.2, 0.19, 0.18])
    first = lmm_model.simulate_lmm_forwards(
        *arguments, correlation_decay=0.2, factors=2, simulations=500, seed=7
    )
    again = lmm_model.simulate_lmm_forwards(
        *arguments, correlation_decay=0.2, factors=2, simulations=500, seed=7
    )
    np.testing.assert_array_equal(first, again)
    assert first.shape == (500, 3, 3)
    assert np.isnan(first[:, 1, 0]).all() and np.isnan(first[:, 2, :2]).all()

    moved = lmm_model.simulate_lmm_forwards(
        [0.03 + 1e-12, 0.032, 0.034],
        [1, 2, 3, 4],
        [0.2, 0.19, 0.18 + 1e-12],
        correlation_decay=0.2 + 1e-12,
        factors=2,
        simulations=500,
        seed=7,
    )
    np.testing.assert_allclose(moved, first, rtol=1e-9)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="increasing"):
        lmm_model.simulate_lmm_forwards([0.03, 0.03], [1, 1, 2], 0.2)
    with pytest.raises(ValueError, match="one positive forward"):
        lmm_model.simulate_lmm_forwards([0.03], [1, 2, 3], 0.2)
    with pytest.raises(ValueError, match="factors"):
        lmm_model.get_lmm_factor_loadings([1, 2, 3], correlation_decay=0.1, factors=3)
    with pytest.raises(ValueError, match="symmetric"):
        lmm_model.get_lmm_factor_loadings([1, 2, 3], np.eye(3))
    with pytest.raises(ValueError, match="after today"):
        lmm_model.get_lmm_swaption(CURVE, [0, 1, 2], 0.2, 0.03)
    with pytest.raises(ValueError, match="caplet"):
        lmm_model.get_lmm_cap(CURVE, [1, 2, 3], 0.2, 0.03, option_type="cap")


def test_a_scalar_correlation_is_refused():
    # A number is not a correlation of the forwards: correlation=1.0 would otherwise be read
    # as the decay β = 1, which gives forwards a year apart a correlation of only e^(-1).
    with pytest.raises(ValueError, match="correlation_decay"):
        lmm_model.get_lmm_factor_loadings([1, 2, 3, 4], correlation=1.0)
    with pytest.raises(ValueError, match="correlation_decay"):
        lmm_model.get_lmm_swaption(CURVE, [2, 3, 4], 0.2, 0.03, correlation=0.5)
    with pytest.raises(ValueError, match="not both"):
        lmm_model.get_lmm_factor_loadings(
            [1, 2, 3], correlation=np.eye(2), correlation_decay=0.1
        )
