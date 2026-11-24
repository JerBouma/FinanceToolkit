"""Derivative Model Tests""" ""

import math

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.bonds import derivative_model
from financetoolkit.fixedincome.inflation import inflation_swap_model
from financetoolkit.utilities.error_model import InvalidArgumentError


def test_get_annuity_factor(recorder):
    recorder.capture(
        derivative_model._get_annuity_factor(
            risk_free_rate=0.03,
            years_to_maturity=1,
            tenor=5,
            payment_frequency=2,
        )
    )


def test_get_annuity_factor_scales_with_tenor(recorder):
    short_tenor = derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=1, payment_frequency=2
    )
    long_tenor = derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=10, payment_frequency=2
    )
    recorder.capture(bool(long_tenor > short_tenor))


def test_get_black_price_receiver(recorder):
    recorder.capture(
        derivative_model.get_black_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.2,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=True,
        )
    )


def test_get_black_price_payer(recorder):
    recorder.capture(
        derivative_model.get_black_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.2,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=False,
        )
    )


def test_get_black_price_default_tenor_matches_maturity(recorder):
    with_default_tenor = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=5,
        risk_free_rate=0.03,
        notional=10_000_000,
        is_receiver=True,
    )
    with_explicit_tenor = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=5,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=5,
        is_receiver=True,
    )
    recorder.capture(with_default_tenor == with_explicit_tenor)


def test_get_black_price_scales_with_tenor(recorder):
    short_tenor_price, _ = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=1,
        is_receiver=True,
    )
    long_tenor_price, _ = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=10,
        is_receiver=True,
    )
    recorder.capture(bool(long_tenor_price > short_tenor_price))


def test_get_bachelier_price_receiver(recorder):
    recorder.capture(
        derivative_model.get_bachelier_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.01,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=True,
        )
    )


def test_get_bachelier_price_payer(recorder):
    recorder.capture(
        derivative_model.get_bachelier_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.01,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=False,
        )
    )


def test_get_bachelier_price_scales_with_tenor(recorder):
    short_tenor_price, _ = derivative_model.get_bachelier_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.01,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=1,
        is_receiver=True,
    )
    long_tenor_price, _ = derivative_model.get_bachelier_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.01,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=10,
        is_receiver=True,
    )
    recorder.capture(bool(long_tenor_price > short_tenor_price))


def test_swaptions_without_volatility_or_time_pay_their_intrinsic_value():
    _, black_payoff = derivative_model.get_black_price(0.03, 0.03, 0.2, 0, 0.03)
    _, bachelier_payoff = derivative_model.get_bachelier_price(
        0.03, 0.02, 0.0, 1, 0.03, is_receiver=False
    )

    assert black_payoff == 0
    assert abs(bachelier_payoff - 0.01) < 1e-12


def test_get_annuity_factor_ends_with_a_short_period():
    # A 1.25-year swap paying twice a year accrues 0.5, 0.5 and a last 0.25 year, paid at
    # the end of the tenor: 2.5, 3.0 and 3.25 years from today.
    annuity = derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=2, tenor=1.25, payment_frequency=2
    )
    expected = (
        0.5 * math.exp(-0.03 * 2.5)
        + 0.5 * math.exp(-0.03 * 3.0)
        + 0.25 * math.exp(-0.03 * 3.25)
    )
    assert annuity == pytest.approx(expected)

    # A tenor shorter than one period is a single payment accruing the tenor.
    assert derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=0.3, payment_frequency=2
    ) == pytest.approx(0.3 * math.exp(-0.03 * 1.3))

    # A whole number of periods in floating point noise has no short period.
    assert derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=0.1 * 3, payment_frequency=10
    ) == pytest.approx(0.1 * sum(math.exp(-0.03 * (1 + 0.1 * k)) for k in (1, 2, 3)))


# Rate options checked against QuantLib 1.44 (BlackCapFloorEngine with a displacement,
# BachelierCapFloorEngine, BlackSwaptionEngine and BachelierSwaptionEngine with physical and
# cash settlement on the par yield curve method, blackFormulaImpliedStdDev and
# bachelierBlackFormulaImpliedVol) on the same discount factors: annual periods of 365 days on
# ACT/365F, so the conventions coincide and the values agree to rounding (rel 1e-12).

CAP_CURVE = pd.Series([0.031, 0.034, 0.037, 0.039, 0.041], index=[0.5, 1, 2, 3, 5])
NEGATIVE_CURVE = pd.Series(
    [-0.006, -0.004, -0.002, 0.001, 0.003], index=[0.5, 1, 2, 3, 5]
)
SWAPTION_CURVE = pd.Series(
    [0.031, 0.034, 0.037, 0.039, 0.041, 0.045], index=[0.5, 1, 2, 3, 5, 10]
)
SWAPTION_FORWARD = 0.0447471922895601


@pytest.mark.parametrize(
    "curve, strike, floor_strike, volatility, method, shift, expected",
    [
        (CAP_CURVE, 0.038, 0.033, 0.25, "black", 0.0,
         (304748.0308210631, 141259.8848600936, 230590.50999307248)),
        (CAP_CURVE, 0.038, 0.033, 0.18, "black", 0.02,
         (320974.9633333704, 157486.81737240072, 229746.94940506388)),
        (CAP_CURVE, 0.038, 0.033, 0.009, "bachelier", 0.0,
         (283711.0975265916, 120222.95156562218, 216087.11685387103)),
        (NEGATIVE_CURVE, 0.0, -0.004, 0.18, "black", 0.02,
         (225382.17834833005, 36562.142173396154, 218879.15290825756)),
        (NEGATIVE_CURVE, 0.0, -0.004, 0.009, "bachelier", 0.0,
         (329497.29006727063, 140677.25389233674, 240101.9686305553)),
    ],
)  # fmt: skip
def test_cap_floor_and_collar_match_quantlib(
    curve, strike, floor_strike, volatility, method, shift, expected
):
    terms = {
        "zero_curve": curve,
        "strike_rate": strike,
        "volatility": volatility,
        "maturity": 5,
        "frequency": 1,
        "method": method,
        "shift": shift,
    }
    cap, caplets = derivative_model.get_cap_floor(**terms)
    floor, _ = derivative_model.get_cap_floor(**terms, option_type="floor")
    collar, _ = derivative_model.get_cap_floor(
        **terms, option_type="collar", floor_strike_rate=floor_strike
    )

    assert (cap, floor, collar) == pytest.approx(expected, rel=1e-12)
    # The first period fixes today and is left out: four caplets from year 1 to 5.
    assert caplets["Start"].tolist() == [1, 2, 3, 4]
    assert derivative_model.get_cap_floor_implied_volatility(
        cap, curve, strike, 5, 1, method=method, shift=shift
    ) == pytest.approx(volatility, rel=1e-10)


def test_cap_floor_parity_and_per_caplet_volatilities():
    terms = {"zero_curve": CAP_CURVE, "strike_rate": 0.036, "maturity": 3}
    volatilities = np.linspace(0.18, 0.26, 11)
    cap, caplets = derivative_model.get_cap_floor(volatility=volatilities, **terms)
    floor, _ = derivative_model.get_cap_floor(
        volatility=volatilities, option_type="floor", **terms
    )
    swap = (
        (caplets["End"] - caplets["Start"])
        * caplets["Discount Factor"]
        * (caplets["Forward Rate"] - 0.036)
    )

    # Cap - Floor = the forward starting payer swap on the same periods.
    assert cap - floor == pytest.approx(10_000_000 * swap.sum(), rel=1e-12)
    assert caplets["Volatility"].tolist() == pytest.approx(list(volatilities))


def test_cap_floor_input_errors():
    with pytest.raises(ValueError, match="floor"):
        derivative_model.get_cap_floor(CAP_CURVE, 0.04, 0.2, 2, option_type="collar")
    with pytest.raises(ValueError, match="Bachelier"):
        derivative_model.get_cap_floor(NEGATIVE_CURVE, 0.0, 0.2, 2)
    with pytest.raises(ValueError, match="method"):
        derivative_model.get_cap_floor(CAP_CURVE, 0.04, 0.2, 2, method="sabr")
    with pytest.raises(ValueError, match="after today"):
        derivative_model.get_cap_floor(CAP_CURVE, 0.04, 0.2, 0.25)


def test_cap_implied_volatility_outside_the_bounds_is_nan():
    floor_value, _ = derivative_model.get_cap_floor(CAP_CURVE, 0.036, 0.0, 3)
    upper, _ = derivative_model.get_cap_floor(CAP_CURVE, 0.036, 1e6, 3)

    assert np.isnan(
        derivative_model.get_cap_floor_implied_volatility(
            floor_value * 0.99, CAP_CURVE, 0.036, 3
        )
    )
    assert np.isnan(
        derivative_model.get_cap_floor_implied_volatility(
            upper * 1.01, CAP_CURVE, 0.036, 3
        )
    )
    assert (
        derivative_model.get_cap_floor_implied_volatility(
            floor_value, CAP_CURVE, 0.036, 3
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "strike, method, volatility, shift, is_receiver, physical, cash, delta, vega",
    [
        (0.025, "black", 0.22, 0.0, False, 813331.6164362191, 811724.2497008755,
         40058823.5792125, 132540.54965734808),
        (0.04, "black", 0.15, 0.02, True, 132191.07559383658, 131929.8297092247,
         -13137675.12113745, 1342020.1299074593),
        (0.025, "bachelier", 0.0095, 0.0, True, 17258.105627285735, 17223.998868935563,
         -2898197.9310946814, 7840987.101665018),
        (0.04, "bachelier", 0.0095, 0.0, False, 330103.42453455593, 329451.0494724079,
         26118504.4058744, 21696195.97938428),
    ],
)  # fmt: skip
def test_swaption_settlement_and_greeks_match_quantlib(
    strike, method, volatility, shift, is_receiver, physical, cash, delta, vega
):
    terms = {
        "forward_rate": SWAPTION_FORWARD,
        "strike_rate": strike,
        "volatility": volatility,
        "years_to_maturity": 2,
        "notional": 10_000_000,
        "tenor": 5,
        "payment_frequency": 1,
        "is_receiver": is_receiver,
        "discount_curve": SWAPTION_CURVE,
    }
    price = (
        derivative_model.get_black_price
        if method == "black"
        else derivative_model.get_bachelier_price
    )
    extra = {"shift": shift} if method == "black" else {}

    assert price(**terms, **extra)[0] == pytest.approx(physical, rel=1e-12)
    assert price(**terms, **extra, settlement="cash")[0] == pytest.approx(
        cash, rel=1e-12
    )
    greeks = derivative_model.get_swaption_greeks(**terms, method=method, shift=shift)
    assert greeks["Price"] == pytest.approx(physical, rel=1e-12)
    assert greeks["Delta"] == pytest.approx(delta, rel=1e-12)
    assert greeks["Vega"] == pytest.approx(vega, rel=1e-12)
    assert greeks["Annuity"] * 10_000_000 == pytest.approx(40932649.55959951, rel=1e-12)


def test_cash_annuity_discounts_at_the_swap_rate():
    annuity = derivative_model._get_annuity_factor(
        0.03, 1, 3, 2, settlement="cash", forward_rate=0.04
    )

    assert annuity == pytest.approx(
        np.exp(-0.03) * sum(0.5 / 1.02**k for k in range(1, 7)), rel=1e-14
    )
    with pytest.raises(ValueError, match="settlement"):
        derivative_model._get_annuity_factor(0.03, 1, 3, 2, settlement="net")


def test_shifted_black_without_shift_is_black():
    terms = {
        "forward_rate": 0.035,
        "strike_rate": 0.03,
        "volatility": 0.2,
        "years_to_maturity": 1,
        "risk_free_rate": 0.03,
        "tenor": 5,
    }

    assert derivative_model.get_black_price(
        **terms, shift=0.0
    ) == derivative_model.get_black_price(**terms)
    # A shift makes negative rates priceable.
    price, _ = derivative_model.get_black_price(
        forward_rate=-0.002,
        strike_rate=-0.001,
        volatility=0.2,
        years_to_maturity=1,
        risk_free_rate=0.0,
        shift=0.02,
    )
    assert price > 0


@pytest.mark.parametrize(
    "price, forward, strike, years, option_type, method, shift, volatility",
    [
        (0.0033101632738933694, 0.03, 0.035, 2.0, "call", "black", 0.0, 0.3),
        (0.010321601878113217, 0.03, 0.02, 1.5, "call", "black", 0.0, 0.25),
        (0.005324349622016448, 0.03, 0.02, 5.0, "put", "black", 0.0, 0.45),
        (0.005728474740659913, -0.002, 0.001, 3.0, "put", "black", 0.03, 0.2),
        (0.01000001288746723, 0.04, 0.03, 0.5, "call", "black", 0.0, 0.1),
        (0.0024472534501547762, 0.03, 0.035, 2.0, "call", "bachelier", 0.0, 0.008),
        (0.007116736274742176, -0.004, 0.001, 3.0, "put", "bachelier", 0.0, 0.006),
        (0.020084907026168296, 0.03, 0.01, 1.0, "call", "bachelier", 0.0, 0.01),
        (0.015138795132120964, 0.02, 0.02, 10.0, "put", "bachelier", 0.0, 0.012),
    ],
)
def test_implied_volatility_matches_quantlib(
    price, forward, strike, years, option_type, method, shift, volatility
):
    # The prices are QuantLib's blackFormula and bachelierBlackFormula; its implied
    # volatility solvers return the same volatilities (to 1e-12).
    assert derivative_model.get_implied_volatility(
        price,
        forward,
        strike,
        years,
        option_type=option_type,
        method=method,
        shift=shift,
    ) == pytest.approx(volatility, rel=1e-9, abs=1e-12)


def test_implied_volatility_bounds():
    # Below the intrinsic value, at or above the forward (Black call) and at intrinsic.
    assert np.isnan(derivative_model.get_implied_volatility(0.004, 0.035, 0.03, 1))
    assert np.isnan(derivative_model.get_implied_volatility(0.035, 0.035, 0.03, 1))
    assert derivative_model.get_implied_volatility(0.005, 0.035, 0.03, 1) == 0.0
    assert np.isnan(
        derivative_model.get_implied_volatility(
            -0.001, 0.035, 0.03, 1, option_type="put", method="bachelier"
        )
    )
    # A very high normal volatility is still found by widening the bracket.
    price = derivative_model._bachelier_formula(0.03, 0.03, 0.5, 4.0, True)
    assert derivative_model.get_implied_volatility(
        price, 0.03, 0.03, 4.0, method="bachelier"
    ) == pytest.approx(0.5, rel=1e-10)
    with pytest.raises(InvalidArgumentError, match="option_type must be one of"):
        derivative_model.get_implied_volatility(0.01, 0.03, 0.03, 1, option_type="x")
    with pytest.raises(ValueError, match="positive"):
        derivative_model.get_implied_volatility(0.01, 0.03, 0.03, 0)


NOMINAL_CURVE = pd.Series([0.040, 0.042, 0.045], index=[1, 5, 10])


@pytest.mark.parametrize(
    "method, volatility, shift, expected",
    [
        ("black", 0.3, 0.0, (180496.71289867643, 241721.65991275854, 59959.63817231609)),
        ("black", 0.012, 1.0, (329355.4604655211, 390580.40747959627, 44269.363688390484)),
        ("bachelier", 0.01, 0.0, (262648.81263350515, 323873.75964758726, 42005.25955023955)),
    ],
)  # fmt: skip
def test_year_on_year_inflation_cap_floor_matches_quantlib(
    method, volatility, shift, expected
):
    # QuantLib's YoYInflationBlackCapFloorEngine, YoYInflationUnitDisplacedBlackCapFloorEngine
    # and YoYInflationBachelierCapFloorEngine on a YoY curve of the same forward rates, with
    # no observation lag. The curve is the zero-coupon inflation curve of the nominal curve
    # and a real curve of 1.5%, 1.8% and 2.1% (1, 5 and 10 years).
    zero_inflation = inflation_curve()
    terms = {
        "zero_curve": NOMINAL_CURVE,
        "zero_inflation_curve": zero_inflation,
        "strike_rate": 0.025,
        "volatility": volatility,
        "maturity": 5,
        "method": method,
        "shift": shift,
    }
    cap, caplets = derivative_model.get_year_on_year_inflation_cap_floor(**terms)
    floor, _ = derivative_model.get_year_on_year_inflation_cap_floor(
        **terms, option_type="floor"
    )
    collar, _ = derivative_model.get_year_on_year_inflation_cap_floor(
        **terms, option_type="collar", floor_strike_rate=0.02
    )

    assert (cap, floor, collar) == pytest.approx(expected, rel=1e-12)
    # Every year is an option, the first included.
    assert caplets["Start"].tolist() == [0, 1, 2, 3, 4]


def inflation_curve() -> pd.Series:
    real = pd.Series([0.015, 0.018, 0.021], index=[1, 5, 10])
    return inflation_swap_model.get_zero_coupon_inflation_swap_rate(
        NOMINAL_CURVE, real, range(1, 11)
    )


def test_year_on_year_inflation_cap_floor_needs_whole_years():
    with pytest.raises(ValueError, match="whole number"):
        derivative_model.get_year_on_year_inflation_cap_floor(
            NOMINAL_CURVE, inflation_curve(), 0.02, 0.01, 2.5, method="bachelier"
        )
