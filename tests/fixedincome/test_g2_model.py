"""G2++ Two-Factor Gaussian Short-Rate Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.fixedincome.short_rates import g2_model, short_rate_model

# QuantLib 1.44 references: G2(a, σ, b, η, ρ) on a flat 3.5% annually compounded curve
# (SimpleDayCounter, so the times are whole years); swaptions with the G2SwaptionEngine over
# 8 standard deviations in 4,000 intervals.
FLAT_CURVE = pd.Series([0.035, 0.035], index=[0.0, 60.0])
PARAMETERS = {
    "A": {
        "speed_x": 0.5,
        "volatility_x": 0.01,
        "speed_y": 0.05,
        "volatility_y": 0.008,
        "correlation": -0.7,
    },
    "B": {
        "speed_x": 0.1,
        "volatility_x": 0.012,
        "speed_y": 0.8,
        "volatility_y": 0.006,
        "correlation": 0.3,
    },
}


@pytest.mark.parametrize(
    "case, time, maturity, x, y, price",
    [
        ("A", 2, 10, 0.005, -0.002, 0.760370933005587),
        ("A", 1, 3, -0.01, 0.004, 0.938137626171831),
        ("A", 0, 7, 0, 0, 0.785990960683819),
        ("B", 2, 10, 0.005, -0.002, 0.736770929508348),
        ("B", 1, 3, -0.01, 0.004, 0.946425700687401),
    ],
)
def test_bond_price_matches_quantlib(case, time, maturity, x, y, price):
    assert g2_model.get_g2_bond_price(
        maturity, FLAT_CURVE, time=time, factor_x=x, factor_y=y, **PARAMETERS[case]
    ) == pytest.approx(price, abs=1e-14)


def test_bond_price_fits_the_curve_today():
    curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])
    maturities = np.array([0.5, 1, 3, 7, 10, 20])
    np.testing.assert_allclose(
        g2_model.get_g2_bond_price(maturities, curve, **PARAMETERS["A"]),
        discount_factors(curve, maturities),
        rtol=1e-15,
    )


@pytest.mark.parametrize(
    "case, expiry, maturity, strike, call, put",
    [
        ("A", 2, 5, 0.9, 0.00860003271044885, 0.00678649618168736),
        ("A", 1, 10, 0.72, 0.0208561100505242, 0.0075894702537955),
        ("A", 0.5, 1, 0.98, 0.00297259863279964, 7.64706322335523e-05),
        ("B", 2, 5, 0.9, 0.0149225870044541, 0.0131090504756926),
        ("B", 1, 10, 0.72, 0.0268256388222032, 0.0135589990254745),
        ("B", 0.5, 1, 0.98, 0.00360746079883378, 0.000711332798267905),
    ],
)
def test_zero_coupon_bond_option_matches_quantlib(
    case, expiry, maturity, strike, call, put
):
    arguments = (expiry, maturity, strike, FLAT_CURVE)
    assert g2_model.get_g2_zero_coupon_bond_option(
        *arguments, **PARAMETERS[case]
    ) == pytest.approx(call, abs=1e-14)
    assert g2_model.get_g2_zero_coupon_bond_option(
        *arguments, option_type="put", **PARAMETERS[case]
    ) == pytest.approx(put, abs=1e-14)


@pytest.mark.parametrize(
    "case, expiry, tenor, fixed_rate, frequency, payer, receiver",
    [
        ("A", 2, 5, 0.04, 1, 0.00533700976346123, 0.0264112582895452),
        ("A", 1, 5, 0.035, 1, 0.00954959415514285, 0.00954959415514311),
        ("A", 5, 10, 0.03, 2, 0.0520511583916106, 0.018861676808319),
        ("B", 2, 5, 0.04, 1, 0.0130400683656824, 0.0341143168917665),
        ("B", 1, 5, 0.035, 1, 0.01698177885411, 0.0169817788541103),
        ("B", 5, 10, 0.03, 2, 0.0587800872090115, 0.0255906056257192),
    ],
)
def test_swaption_matches_quantlib(
    case, expiry, tenor, fixed_rate, frequency, payer, receiver
):
    # QuantLib integrates over 8 standard deviations in 4,000 intervals: 1e-11 covers it.
    common = {"fixed_frequency": frequency, **PARAMETERS[case]}
    assert g2_model.get_g2_swaption(
        expiry, tenor, fixed_rate, FLAT_CURVE, **common
    ) == pytest.approx(payer, abs=1e-11)
    assert g2_model.get_g2_swaption(
        expiry, tenor, fixed_rate, FLAT_CURVE, swaption_type="receiver", **common
    ) == pytest.approx(receiver, abs=1e-11)


def test_payer_less_receiver_is_the_forward_swap():
    curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])
    for parameters in PARAMETERS.values():
        payer = g2_model.get_g2_swaption(
            3, 7, 0.037, curve, fixed_frequency=2, **parameters
        )
        receiver = g2_model.get_g2_swaption(
            3,
            7,
            0.037,
            curve,
            swaption_type="receiver",
            fixed_frequency=2,
            **parameters,
        )
        times = 3 + np.arange(1, 15) / 2
        forward = (
            discount_factors(curve, 3.0)
            - np.sum(0.037 / 2 * discount_factors(curve, times))
            - discount_factors(curve, 10.0)
        )
        assert payer - receiver == pytest.approx(float(forward), abs=1e-13)


def test_without_the_second_factor_the_model_is_hull_white():
    curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])
    g1 = {
        "speed_x": 0.08,
        "volatility_x": 0.012,
        "speed_y": 0.3,
        "volatility_y": 0.0,
        "correlation": 0.4,
    }
    hull_white = {
        "speed": 0.08,
        "volatility": 0.012,
        "zero_curve": curve,
        "model": "hull_white",
    }

    for option_type in ("call", "put"):
        assert g2_model.get_g2_zero_coupon_bond_option(
            1, 5, 0.87, curve, option_type=option_type, **g1
        ) == pytest.approx(
            short_rate_model.get_zero_coupon_bond_option(
                1, 5, 0.87, option_type=option_type, **hull_white
            ),
            abs=1e-15,
        )
    # The swaption integral needs η > 0; as η falls it tends to Jamshidian's Hull-White price
    # (uncorrelated, so the gap is of order η² rather than ρση).
    expected = short_rate_model.get_swaption(
        2, 10, 0.04, fixed_frequency=2, **hull_white
    )
    for eta, tolerance in ((1e-4, 1e-6), (1e-6, 1e-10)):
        assert g2_model.get_g2_swaption(
            2,
            10,
            0.04,
            curve,
            fixed_frequency=2,
            **{**g1, "volatility_y": eta, "correlation": 0.0},
        ) == pytest.approx(expected, abs=tolerance)


def test_swaption_is_stable_under_tiny_perturbations():
    base = g2_model.get_g2_swaption(2, 5, 0.04, FLAT_CURVE, **PARAMETERS["A"])
    moved = g2_model.get_g2_swaption(
        2,
        5,
        0.04 + 1e-12,
        FLAT_CURVE,
        **{**PARAMETERS["A"], "volatility_x": 0.01 + 1e-12},
    )
    assert moved == pytest.approx(base, abs=1e-11)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="speeds"):
        g2_model.get_g2_bond_price(5, FLAT_CURVE, 0.0, 0.01, 0.1, 0.01, 0.0)
    with pytest.raises(ValueError, match="correlation"):
        g2_model.get_g2_bond_price(5, FLAT_CURVE, 0.1, 0.01, 0.1, 0.01, 1.5)
    with pytest.raises(ValueError, match="Hull-White"):
        g2_model.get_g2_swaption(1, 5, 0.03, FLAT_CURVE, 0.1, 0.01, 0.1, 0.0, 0.0)
    with pytest.raises(ValueError, match="expire"):
        g2_model.get_g2_zero_coupon_bond_option(
            5, 5, 0.9, FLAT_CURVE, **PARAMETERS["A"]
        )
    with pytest.raises(ValueError, match="payer"):
        g2_model.get_g2_swaption(
            1, 5, 0.03, FLAT_CURVE, swaption_type="call", **PARAMETERS["A"]
        )
