"""Swap and Forward Model Tests

The reference values come from QuantLib 1.44, FinancePy 1.1.2 and rateslib 3.0.0, each given
the discount factors of the same curve (annually compounded zero rates, linear in between) at
every date it needs, on a 30/360 schedule for the swaps and the asset swap and on ACT/365
Fixed times with an ACT/360 accrual for the FRAs.
"""

import math

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.bonds import derivative_model
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.fixedincome.swaps import forward_model, swap_model

CURVE = pd.Series(
    [0.035, 0.038, 0.04, 0.042, 0.045, 0.047], index=[0.25, 1, 2, 5, 10, 30]
)

# Relative tolerance: the references agree to rounding (1e-12 or better).
REFERENCE = 1e-9


def test_discount_factors_follow_the_annual_zero_curve():
    assert discount_factors(CURVE, 0) == pytest.approx(1)
    assert discount_factors(CURVE, 1.5) == pytest.approx(1.039**-1.5)
    # Flat beyond the curve's maturities.
    assert discount_factors(CURVE, 40) == pytest.approx(1.047**-40)


def test_swap_rate_matches_quantlib_and_rateslib():
    rate, annuity = swap_model.get_swap_rate(CURVE, 5)

    # QuantLib VanillaSwap.fairRate and fixedLegBPS; rateslib IRS.rate and analytic_delta.
    assert rate == pytest.approx(0.041878722030031165, rel=REFERENCE)
    assert annuity == pytest.approx(4.439740236220727, rel=REFERENCE)
    assert annuity * 1e6 * 1e-4 == pytest.approx(443.97402362207276, rel=REFERENCE)


def test_forward_starting_swap_rate_matches_quantlib():
    rate, annuity = swap_model.get_swap_rate(
        CURVE, 5, fixed_frequency=2, forward_start=2
    )

    assert rate == pytest.approx(0.043894217828709156, rel=REFERENCE)
    assert annuity == pytest.approx(4.119136163568449, rel=REFERENCE)


def test_swap_rate_with_a_short_first_period():
    rate, annuity = swap_model.get_swap_rate(CURVE, 1.5, fixed_frequency=1)

    # Payments after half a year (accruing half a year) and 1.5 years.
    expected_annuity = 0.5 * discount_factors(CURVE, 0.5) + discount_factors(CURVE, 1.5)
    assert annuity == pytest.approx(float(expected_annuity))
    assert rate == pytest.approx(
        float((1 - discount_factors(CURVE, 1.5)) / expected_annuity)
    )


def test_swap_value_matches_quantlib():
    value = swap_model.get_swap_value(CURVE, 0.04, 7, fixed_frequency=2)

    assert value["Value"] == pytest.approx(151916.6470388337, rel=REFERENCE)
    assert value["Fixed Leg"] == pytest.approx(2410583.8230815213, rel=REFERENCE)
    assert value["Floating Leg"] == pytest.approx(2562500.470120355, rel=REFERENCE)
    assert value["PV01"] == pytest.approx(6026.459557703799, rel=REFERENCE)
    assert value["Par Swap Rate"] == pytest.approx(0.042520827454066865, rel=REFERENCE)
    # QuantLib's value on the curve with every zero rate 1 basis point higher and lower.
    assert value["DV01"] == pytest.approx(5818.322956462856, rel=REFERENCE)


def test_swap_value_of_a_receiver_matches_rateslib():
    value = swap_model.get_swap_value(
        CURVE, 0.04, 5, notional=1_000_000, is_receiver=True
    )

    assert value["Value"] == pytest.approx(-8341.03778940448, rel=REFERENCE)
    assert value["DV01"] < 0


def test_running_swap_value_matches_quantlib():
    # Started 2025-10-15, ends 2030-10-15, valued 2026-02-15 with the fixing of 2026-01-15.
    value = swap_model.get_swap_value(
        CURVE, 0.039, 4 + 8 / 12, current_fixing=0.041, is_receiver=True
    )

    assert value["Value"] == pytest.approx(-27454.947830330115, rel=REFERENCE)
    assert value["PV01"] == pytest.approx(4502.4737092866635, rel=REFERENCE)


def test_swap_at_the_par_rate_is_worth_nothing():
    rate, _ = swap_model.get_swap_rate(CURVE, 10)
    value = swap_model.get_swap_value(CURVE, rate, 10)

    assert value["Value"] == pytest.approx(0, abs=1e-6)


def test_swap_value_needs_a_maturity():
    with pytest.raises(ValueError):
        swap_model.get_swap_value(CURVE, 0.04, 0)


@pytest.mark.parametrize(
    ("start", "end", "rate", "value"),
    [
        (
            0.2465753424657534,
            0.4958904109589041,
            0.03595597200937569,
            -10044.763276092932,
        ),
        (0.4958904109589041, 1.0, 0.039056194670079016, -4647.296636305429),
        (1.0, 2.0, 0.04142845831023841, 13390.329225145368),
    ],
)
def test_forward_rate_agreement_matches_quantlib(start, end, rate, value):
    fra_rate, fra_value = forward_model.get_forward_rate_agreement(
        CURVE, start, end, fixed_rate=0.04
    )

    assert fra_rate == pytest.approx(rate, rel=REFERENCE)
    assert fra_value == pytest.approx(value, rel=REFERENCE)


def test_forward_rate_agreement_day_counts():
    act_360, value = forward_model.get_forward_rate_agreement(CURVE, 0.5, 1)
    act_365, _ = forward_model.get_forward_rate_agreement(
        CURVE, 0.5, 1, day_count="act/365"
    )

    assert act_360 == pytest.approx(act_365 * 360 / 365)
    assert math.isnan(value)
    with pytest.raises(ValueError):
        forward_model.get_forward_rate_agreement(CURVE, 1, 0.5)


def test_futures_convexity_adjustment_matches_financepy():
    start, end = 1.9095890410958904, 2.164383561643836

    hull_white = forward_model.get_futures_convexity_adjustment(0.012, start, end, 0.05)
    ho_lee = forward_model.get_futures_convexity_adjustment(0.012, start, end)

    assert hull_white == pytest.approx(0.00026879494186860095, rel=REFERENCE)
    assert ho_lee == pytest.approx(0.00029758198536310756, rel=REFERENCE)
    # Mean reversion lowers the adjustment, and a tiny one tends to Ho-Lee.
    assert forward_model.get_futures_convexity_adjustment(
        0.012, start, end, 1e-7
    ) == pytest.approx(ho_lee, rel=1e-5)


@pytest.mark.parametrize(
    ("years", "domestic_rate", "foreign_rate", "forward"),
    [
        (0.2465753424657534, 0.03, 0.045, 1.096085474309906),
        (0.9863013698630136, 0.03, 0.045, 1.0844252819102391),
        (
            4.931506849315069,
            0.03493150684931507,
            0.041045662100456626,
            1.0685062106333363,
        ),
    ],
)
def test_fx_forward_rate_matches_rateslib(years, domestic_rate, foreign_rate, forward):
    forward_rate, _ = forward_model.get_fx_forward_rate(
        1.1, domestic_rate, foreign_rate, years
    )

    assert forward_rate == pytest.approx(forward, rel=REFERENCE)


def test_fx_forward_value():
    forward_rate, value = forward_model.get_fx_forward_rate(
        1.1, 0.03, 0.045, 2, strike_rate=1.05, notional=1_000_000
    )

    assert value == pytest.approx(1_000_000 * (forward_rate - 1.05) / 1.03**2)


def test_equity_forward_matches_financepy():
    forward, value = forward_model.get_equity_forward(
        100, 0.04, 2, dividend_yield=0.015, strike_price=105
    )

    assert forward == pytest.approx(105.1271096376024, rel=REFERENCE)
    assert value == pytest.approx(0.11733698425405857, rel=REFERENCE)


def test_equity_forward_with_cash_dividends():
    forward, value = forward_model.get_equity_forward(
        100, 0.04, 1, dividends=[(0.5, 2), (1.5, 3)]
    )

    # The dividend after delivery does not count (Hull, 2022, Example 5.2 layout).
    assert forward == pytest.approx((100 - 2 * np.exp(-0.02)) * np.exp(0.04))
    assert math.isnan(value)


def test_asset_swap_spread_matches_quantlib():
    par, market_value = swap_model.get_asset_swap_spread(CURVE, 0.05, 4.25, 99)

    assert par == pytest.approx(0.011530511616100261, rel=REFERENCE)
    assert market_value == pytest.approx(0.011501757223042652, rel=REFERENCE)


def test_bond_priced_on_the_curve_has_no_asset_swap_spread():
    dirty = 100 * (
        0.05 / 2 * discount_factors(CURVE, np.arange(1, 11) / 2).sum()
        + discount_factors(CURVE, 5)
    )
    par, market_value = swap_model.get_asset_swap_spread(CURVE, 0.05, 5, dirty)

    assert par == pytest.approx(0, abs=1e-12)
    assert market_value == pytest.approx(0, abs=1e-12)


def test_i_spread_is_the_yield_over_the_swap_rate():
    swap_rate, _ = swap_model.get_swap_rate(CURVE, 7, fixed_frequency=2)

    assert swap_model.get_i_spread(CURVE, 0.052, 7) == pytest.approx(0.052 - swap_rate)


def test_swaption_annuity_on_a_curve_is_the_forward_swap_annuity():
    _, annuity = swap_model.get_swap_rate(CURVE, 5, fixed_frequency=2, forward_start=2)

    assert derivative_model._get_annuity_factor(
        None, years_to_maturity=2, tenor=5, payment_frequency=2, discount_curve=CURVE
    ) == pytest.approx(annuity)

    price, payoff = derivative_model.get_bachelier_price(
        forward_rate=0.0439,
        strike_rate=0.04,
        volatility=0.01,
        years_to_maturity=2,
        tenor=5,
        discount_curve=CURVE,
    )
    assert price == pytest.approx(10_000_000 * annuity * payoff)


def test_swaption_needs_a_rate_or_a_curve():
    with pytest.raises(ValueError):
        derivative_model.get_black_price(
            forward_rate=0.04, strike_rate=0.04, volatility=0.2, years_to_maturity=1
        )
