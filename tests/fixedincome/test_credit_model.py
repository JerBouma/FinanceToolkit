"""Credit Curve and CDS Model Tests

The reference values come from QuantLib 1.44 and FinancePy 1.1.2, given the discount factors of
the same curve (annually compounded zero rates, linear in between) at every node they need and
the same piecewise flat hazard rates, on dates today + 365 t so that the ACT/365 Fixed times are
exact. The closed forms are those of a flat hazard rate on a flat curve.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from financetoolkit.fixedincome.credit import cds_model, credit_curve_model
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.utilities.error_model import InvalidArgumentError

CURVE = pd.Series(
    [0.038, 0.04, 0.041, 0.043, 0.044, 0.045, 0.047], index=[1, 2, 3, 5, 7, 10, 30.0]
)
HAZARD_RATES = [0.01, 0.015, 0.02, 0.022, 0.025]
HAZARD_TIMES = [1, 3, 5, 7, 10.0]

# A flat 4% annually compounded curve and a flat 2% hazard rate.
FLAT_CURVE = pd.Series([0.04, 0.04], index=[1, 30.0])
FLAT_RATE = np.log(1.04)
FLAT_HAZARD = 0.02


def test_survival_probability_matches_quantlib():
    # QuantLib HazardRateCurve (backward flat) survivalProbability.
    survival = credit_curve_model.get_survival_probability(
        HAZARD_RATES, HAZARD_TIMES, [1, 2, 4, 6, 12]
    )

    assert survival == pytest.approx(
        [
            0.9900498337491681,
            0.9753099120283326,
            0.9417645335842487,
            0.9030295516688768,
            0.7795799733847004,
        ],
        rel=1e-12,
    )
    assert credit_curve_model.get_survival_probability(
        HAZARD_RATES, HAZARD_TIMES, [0, 0.5]
    ) == pytest.approx([1, np.exp(-0.005)])


def test_hazard_rate_is_piecewise_flat():
    hazard = credit_curve_model.get_hazard_rate(
        HAZARD_RATES, HAZARD_TIMES, [0, 1, 1.5, 10, 40]
    )

    assert hazard == pytest.approx([0.01, 0.01, 0.015, 0.025, 0.025])


def test_default_integrals_match_the_closed_form():
    default_value, accrual_value = credit_curve_model.get_default_integrals(
        FLAT_CURVE, [FLAT_HAZARD], [5], 0, 5
    )
    rate = FLAT_RATE + FLAT_HAZARD

    assert default_value == pytest.approx(
        FLAT_HAZARD / rate * (1 - np.exp(-rate * 5)), rel=1e-12
    )
    assert accrual_value == pytest.approx(
        FLAT_HAZARD * (1 - np.exp(-rate * 5) * (1 + rate * 5)) / rate**2, rel=1e-12
    )


def test_credit_default_swap_matches_the_closed_form():
    swap = cds_model.get_credit_default_swap(
        5, FLAT_CURVE, [FLAT_HAZARD], [5], coupon=0.01, frequency=1
    )
    rate = FLAT_RATE + FLAT_HAZARD
    years = np.arange(1, 6)
    coupons = np.exp(-rate * years).sum()
    accrual_on_default = (
        FLAT_HAZARD
        * np.exp(-rate * (years - 1)).sum()
        * (1 - np.exp(-rate) * (1 + rate))
        / rate**2
    )
    rpv01 = 365 / 360 * (coupons + accrual_on_default)
    protection = 0.6 * FLAT_HAZARD / rate * (1 - np.exp(-rate * 5))

    assert swap["RPV01"] == pytest.approx(rpv01, rel=1e-12)
    assert swap["Protection Leg"] == pytest.approx(protection, rel=1e-12)
    assert swap["Par Spread"] == pytest.approx(protection / rpv01, rel=1e-12)
    # QuantLib IsdaCdsEngine defaultLegNPV on the same flat curves.
    assert swap["Protection Leg"] == pytest.approx(0.05193242403064197, rel=1e-12)


def test_credit_default_swap_matches_quantlib():
    swap = cds_model.get_credit_default_swap(
        5, CURVE, HAZARD_RATES, HAZARD_TIMES, coupon=0.01, frequency=1
    )

    # QuantLib CreditDefaultSwap with the IsdaCdsEngine (Taylor, no bias, piecewise):
    # defaultLegNPV, couponLegBPS, fairSpread and NPV. The protection legs agree to
    # rounding; the risky annuities differ by the engine's accrual conventions (4e-5).
    assert swap["Protection Leg"] == pytest.approx(0.04127320733809537, rel=1e-12)
    assert swap["RPV01"] == pytest.approx(4.347761814530229, rel=1e-4)
    assert swap["Par Spread"] == pytest.approx(0.009492978019209845, rel=1e-4)
    assert swap["Upfront"] == pytest.approx(-0.0022044108072069232, rel=1e-3)
    # IntegralCdsEngine with a one-day step and MidPointCdsEngine.
    assert swap["RPV01"] == pytest.approx(4.347664327057417, rel=1e-4)
    assert swap["Par Spread"] == pytest.approx(0.009492637558144335, rel=1e-4)
    assert swap["Par Spread"] == pytest.approx(0.0094921265870985, rel=2e-4)


def test_credit_default_swap_upfront_and_accrued_premium():
    # 4.9 years of quarterly premiums: the current period started 0.1 years ago.
    swap = cds_model.get_credit_default_swap(
        4.9, CURVE, HAZARD_RATES, HAZARD_TIMES, coupon=0.05, notional=1e7
    )

    assert swap["Accrued Premium"] == pytest.approx(1e7 * 0.05 * 0.1 * 365 / 360)
    assert swap["Full RPV01"] - swap["RPV01"] == pytest.approx(0.1 * 365 / 360)
    assert swap["Upfront"] == pytest.approx((swap["Par Spread"] - 0.05) * swap["RPV01"])
    assert swap["Mark-to-Market"] == pytest.approx(
        1e7 * swap["Upfront"] - swap["Accrued Premium"]
    )
    assert swap["Premium Leg"] == pytest.approx(1e7 * 0.05 * swap["Full RPV01"])
    # At the par spread the clean value is zero.
    at_par = cds_model.get_credit_default_swap(
        4.9, CURVE, HAZARD_RATES, HAZARD_TIMES, coupon=swap["Par Spread"]
    )
    assert at_par["Upfront"] == pytest.approx(0, abs=1e-14)


def test_cds_implied_hazard_rates_reprice_the_quotes():
    spreads = [0.006, 0.009, 0.012, 0.014]
    hazard = cds_model.get_cds_implied_hazard_rates(
        [1, 3, 5, 7], spreads, CURVE, frequency=1
    )

    for maturity, spread in zip([1, 3, 5, 7], spreads):
        assert cds_model.get_credit_default_swap(
            maturity, CURVE, hazard, hazard.index, frequency=1
        )["Par Spread"] == pytest.approx(spread, rel=1e-10)

    # QuantLib PiecewiseFlatHazardRate on SpreadCdsHelpers with the ISDA model.
    survival = credit_curve_model.get_survival_probability(
        hazard, hazard.index, [1, 3, 5, 7]
    )
    assert survival == pytest.approx(
        [
            0.9900976318699292,
            0.9558570486106578,
            0.9027506504665002,
            0.8437773136018945,
        ],
        abs=2e-5,
    )


def test_cds_implied_hazard_rates_match_financepy():
    # FinancePy CDSCurve of quarterly CDS maturing on 20 December of 2025, 2027, 2029 and
    # 2031 from 20 December 2024. FinancePy dates the schedule on the calendar, so the
    # survival probabilities agree to 2e-4.
    maturities = np.array([365, 1095, 1826, 2556]) / 365
    hazard = cds_model.get_cds_implied_hazard_rates(
        maturities, [0.006, 0.009, 0.012, 0.014], CURVE
    )
    survival = credit_curve_model.get_survival_probability(
        hazard, hazard.index, maturities
    )

    assert survival == pytest.approx(
        [
            0.9898764330295223,
            0.9550856398961629,
            0.9010798276326456,
            0.841207902515993,
        ],
        abs=2e-4,
    )


def test_cds_implied_hazard_rates_follow_the_credit_triangle():
    hazard = cds_model.get_cds_implied_hazard_rates([5], [0.012], FLAT_CURVE)

    assert hazard.iloc[0] == pytest.approx(0.012 / 0.6, rel=0.01)


def test_risky_bond_price_matches_quantlib():
    price = credit_curve_model.get_risky_bond_price(
        0.05, 7, CURVE, HAZARD_RATES, HAZARD_TIMES, frequency=1
    )

    # QuantLib RiskyBondEngine, which takes the default at the middle of each coupon period.
    assert price == pytest.approx(97.36706789494045, rel=1e-5)


def test_risky_bond_price_without_default_risk_is_the_risk_free_price():
    price = credit_curve_model.get_risky_bond_price(0.05, 3, FLAT_CURVE, [0.0], [3])

    assert price == pytest.approx(
        100 * (0.025 * sum(1.04 ** -(t / 2) for t in range(1, 7)) + 1.04**-3)
    )


def test_risky_bond_price_of_a_zero_coupon_bond_without_recovery():
    # A zero-coupon bond that recovers nothing is worth P(T) Q(T).
    price = credit_curve_model.get_risky_bond_price(
        0.0, 5, FLAT_CURVE, [FLAT_HAZARD], [5], recovery_rate=0.0, par_value=1
    )

    assert price == pytest.approx(np.exp(-(FLAT_RATE + FLAT_HAZARD) * 5))


def test_bond_implied_hazard_rates_reprice_par_bonds():
    maturities = [2, 4, 6, 8.5, 12.5, 20]
    spreads = [0.0062, 0.0084, 0.0098, 0.0113, 0.011, 0.0111]
    hazard = credit_curve_model.get_bond_implied_hazard_rates(
        maturities, spreads, CURVE
    )

    for maturity, spread in zip(maturities, spreads):
        times = np.arange(1, 2 * maturity + 1) / 2
        spread_discount = (1 + np.interp(times, CURVE.index, CURVE)) ** -times * (
            1 + spread / 2
        ) ** (-2 * times)
        coupon = 2 * (1 - spread_discount[-1]) / spread_discount.sum()

        assert credit_curve_model.get_risky_bond_price(
            coupon, maturity, CURVE, hazard, hazard.index
        ) == pytest.approx(100, rel=1e-10)

    # About the credit triangle.
    assert hazard.iloc[0] == pytest.approx(0.0062 / 0.6, rel=0.03)


def test_bootstraps_reject_bad_quotes():
    with pytest.raises(ValueError):
        credit_curve_model.get_bond_implied_hazard_rates([5, 3], [0.01, 0.01], CURVE)
    with pytest.raises(ValueError):
        cds_model.get_cds_implied_hazard_rates([5], [-0.01], CURVE)
    with pytest.raises(ValueError):
        cds_model.get_credit_default_swap(5, CURVE, [0.01], [5], recovery_rate=1)


OPTION_CURVE = pd.Series([0.035, 0.037, 0.040], index=[1, 2, 5])


@pytest.mark.parametrize(
    "hazard_rate, expiry, maturity, strike, volatility, knockout, option_type, value",
    [
        (0.02, 1, 6, 0.012, 0.4, True, "payer", 79205.91099899824),
        (0.02, 1, 6, 0.012, 0.4, True, "receiver", 76097.99399322078),
        (0.02, 1, 6, 0.012, 0.4, False, "payer", 193996.2106699823),
        (0.035, 2, 7, 0.015, 0.6, True, "payer", 339035.78524256084),
        (0.035, 2, 7, 0.015, 0.6, True, "receiver", 118946.78793425323),
        (0.035, 2, 7, 0.025, 0.6, False, "payer", 584827.7650711037),
    ],
)
def test_cds_option_matches_quantlib(
    hazard_rate, expiry, maturity, strike, volatility, knockout, option_type, value
):
    # QuantLib's BlackCdsOptionEngine on a forward CDS priced with the IntegralCdsEngine
    # (daily steps), annual premiums on dates today + 365 t: the forward legs differ by its
    # daily integration (rel 3e-5 on the annuity, 1e-6 on the spread), the value by up to 4e-4.
    option = cds_model.get_cds_option(
        expiry,
        maturity,
        OPTION_CURVE,
        [hazard_rate],
        [50.0],
        strike,
        volatility,
        notional=10_000_000,
        frequency=1,
        option_type=option_type,
        knockout=knockout,
    )

    assert option["Value"] == pytest.approx(value, rel=4e-4)


def test_cds_option_forward_legs_by_quadrature():
    # The forward legs integrated numerically, with the forward rate constant between the
    # curve nodes and the premium dates as the model's exact integrals assume.
    hazard_rate, expiry, maturity, frequency = 0.05, 0.5, 5.25, 4
    ends = expiry + np.arange(1, 20) / frequency
    starts = np.maximum(ends - 1 / frequency, expiry)
    grid = np.unique(np.r_[0.0, OPTION_CURVE.index, expiry, ends, 50.0])
    log_discount = np.log(discount_factors(OPTION_CURVE, grid))

    def density(t):
        return (
            np.exp(np.interp(t, grid, log_discount))
            * hazard_rate
            * np.exp(-hazard_rate * t)
        )

    protection = 0.6 * sum(
        quad(density, a, b, epsrel=1e-13)[0] for a, b in zip(starts, ends)
    )
    rpv01 = sum(
        (end - start) * 365 / 360 * density(end) / hazard_rate
        + 365
        / 360
        * quad(lambda t, s=start: (t - s) * density(t), start, end, epsrel=1e-13)[0]
        for start, end in zip(starts, ends)
    )
    option = cds_model.get_cds_option(
        expiry, maturity, OPTION_CURVE, [hazard_rate], [50.0], 0.012, 0.4, frequency=4
    )

    assert option["Forward RPV01"] == pytest.approx(rpv01, rel=1e-12)
    assert option["Forward Spread"] == pytest.approx(protection / rpv01, rel=1e-12)


def test_cds_option_parity_and_front_end_protection():
    terms = {
        "option_expiry": 1,
        "maturity": 6,
        "zero_curve": OPTION_CURVE,
        "hazard_rates": [0.02],
        "hazard_times": [50.0],
        "strike_spread": 0.011,
        "volatility": 0.5,
    }
    payer = cds_model.get_cds_option(**terms)
    receiver = cds_model.get_cds_option(**terms, option_type="receiver")
    no_knockout = cds_model.get_cds_option(**terms, knockout=False)

    # Payer - Receiver = a forward CDS paying the strike, on the forward annuity.
    assert payer["Value"] - receiver["Value"] == pytest.approx(
        payer["Forward RPV01"] * (payer["Forward Spread"] - 0.011), rel=1e-12
    )
    assert no_knockout["Front-End Protection"] == pytest.approx(
        0.6 * (1 - np.exp(-0.02)) / 1.035, rel=1e-12
    )
    assert no_knockout["Value"] == pytest.approx(
        payer["Value"] + no_knockout["Front-End Protection"], rel=1e-12
    )
    with pytest.raises(InvalidArgumentError, match="option_type must be one of"):
        cds_model.get_cds_option(**terms, option_type="straddle")
    with pytest.raises(ValueError, match="before the maturity"):
        cds_model.get_cds_option(**{**terms, "option_expiry": 7})


def test_cds_needs_a_positive_maturity():
    zero_curve = pd.Series([0.03, 0.037], index=[1, 10])
    with pytest.raises(ValueError, match="maturity of a CDS must be positive"):
        cds_model.get_credit_default_swap(0, zero_curve, [0.02], [5])


def test_bootstraps_report_a_negative_hazard_rate_on_an_inverted_curve():
    # A 3-year spread of 1% after a 1-year spread of 5% needs a negative hazard rate from 1
    # to 3 years; the root search failed with "f(a) and f(b) must have different signs".
    zero_curve = pd.Series([0.03, 0.032, 0.035, 0.037], index=[1, 2, 5, 10])
    for bootstrap in (
        cds_model.get_cds_implied_hazard_rates,
        credit_curve_model.get_bond_implied_hazard_rates,
    ):
        with pytest.raises(ValueError, match="implies a negative hazard rate"):
            bootstrap([1, 3, 5], [0.05, 0.01, 0.005], zero_curve)
