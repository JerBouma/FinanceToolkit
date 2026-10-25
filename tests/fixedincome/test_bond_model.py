"""Bond Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome import bond_model

# pylint: disable=missing-function-docstring

TAYLOR_APPROXIMATION_TOLERANCE = 0.0001


def test_get_bond_price_from_curve_matches_flat_yield(recorder):
    flat_curve = pd.Series({1: 0.08, 2: 0.08, 3: 0.08, 5: 0.08, 10: 0.08})
    price_from_curve = (
        bond_model._get_bond_price_from_curve(  # pylint: disable=protected-access
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=flat_curve,
            frequency=1,
        )
    )
    price_from_ytm = bond_model.get_bond_price(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        yield_to_maturity=0.08,
        frequency=1,
    )
    recorder.capture(bool(round(price_from_curve, 6) == round(price_from_ytm, 6)))


def test_get_z_spread(recorder):
    benchmark = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 10: 0.05})
    recorder.capture(
        bond_model.get_z_spread(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            bond_price=96.038065,
            spot_rates=benchmark,
            frequency=1,
        )
    )


def test_get_z_spread_recovers_synthetic_spread(recorder):
    benchmark = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 10: 0.05})
    true_spread = 0.015
    synthetic_price = (
        bond_model._get_bond_price_from_curve(  # pylint: disable=protected-access
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=benchmark + true_spread,
            frequency=1,
        )
    )
    recovered_spread = bond_model.get_z_spread(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        bond_price=synthetic_price,
        spot_rates=benchmark,
        frequency=1,
    )
    recorder.capture(bool(round(recovered_spread, 3) == round(true_spread, 3)))


def test_get_z_spread_type_error():
    with pytest.raises(TypeError):
        bond_model.get_z_spread(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            bond_price=96,
            spot_rates=[0.03, 0.04, 0.05],
        )


def test_get_bond_equivalent_yield(recorder):
    recorder.capture(
        bond_model.get_bond_equivalent_yield(discount_yield=0.05, days_to_maturity=182)
    )


def test_get_bond_equivalent_yield_matches_hand_calculation(recorder):
    bey = bond_model.get_bond_equivalent_yield(
        discount_yield=0.05, days_to_maturity=182
    )
    expected = 365 * 0.05 / (360 - 182 * 0.05)
    recorder.capture(bool(round(bey, 10) == round(expected, 10)))


def test_get_bond_equivalent_yield_type_error():
    with pytest.raises(TypeError):
        bond_model.get_bond_equivalent_yield(
            discount_yield="0.05", days_to_maturity=182
        )


def test_get_key_rate_duration(recorder):
    spot_rates = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 7: 0.048, 10: 0.05})
    recorder.capture(
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=spot_rates,
            key_rate_maturity=5,
            frequency=1,
        )
    )


def test_get_key_rate_duration_sums_to_approx_effective_duration(recorder):
    spot_rates = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 7: 0.048, 10: 0.05})
    total_krd = sum(
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=spot_rates,
            key_rate_maturity=maturity,
            frequency=1,
        )
        for maturity in spot_rates.index
    )
    approx_flat_rate = float(np.interp(5, spot_rates.index, spot_rates.to_numpy()))
    effective_duration = bond_model.get_effective_duration(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        yield_to_maturity=approx_flat_rate,
        frequency=1,
    )
    # Summed key rate durations should approximate the parallel-shift duration.
    recorder.capture(bool(abs(total_krd - effective_duration) < 1))


def test_get_key_rate_duration_type_error():
    with pytest.raises(TypeError):
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=[0.03, 0.04, 0.05],
            key_rate_maturity=5,
        )


def test_get_key_rate_duration_invalid_key_rate_maturity():
    spot_rates = pd.Series({1: 0.03, 2: 0.04, 3: 0.05})
    with pytest.raises(ValueError):
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=3,
            spot_rates=spot_rates,
            key_rate_maturity=99,
        )


def test_get_taylor_price_change(recorder):
    recorder.capture(
        bond_model.get_taylor_price_change(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            yield_to_maturity=0.08,
            frequency=1,
            yield_change=0.01,
        )
    )


def test_get_taylor_price_change_matches_actual_repricing(recorder):
    par_value, coupon_rate, years_to_maturity, ytm, frequency = 100, 0.05, 5, 0.08, 1
    yield_change = 0.001

    taylor_pct = bond_model.get_taylor_price_change(
        par_value, coupon_rate, years_to_maturity, ytm, frequency, yield_change
    )

    price_before = bond_model.get_bond_price(
        par_value, coupon_rate, years_to_maturity, ytm, frequency
    )
    price_after = bond_model.get_bond_price(
        par_value, coupon_rate, years_to_maturity, ytm + yield_change, frequency
    )
    actual_pct = (price_after - price_before) / price_before

    # For a small yield shift the Taylor approximation should match the reprice.
    recorder.capture(
        bool(abs(taylor_pct - actual_pct) < TAYLOR_APPROXIMATION_TOLERANCE)
    )


def test_periods_are_rounded_not_truncated():
    """0.29 years of 100 periods is 28.999999999999996 in floating point, not 28."""
    assert bond_model.get_number_of_periods(0.29, 100) == 29  # noqa: PLR2004
    assert bond_model.get_bond_price(100, 0.05, 0.29, 0.05, 100) == pytest.approx(
        bond_model.get_bond_price(100, 0.05, 29 / 100, 0.05, 100)
    )


def test_yield_to_maturity_from_a_guess_of_zero():
    """A guess of 0 starts the secant method from two distinct points."""
    price = bond_model.get_bond_price(1000, 0.05, 10, 0.06, 2)

    assert bond_model.get_yield_to_maturity(
        1000, 0.05, 10, price, 2, guess=0
    ) == pytest.approx(0.06, abs=1e-6)


def test_macaulay_duration_matches_the_price_periods():
    """The principal is weighted at the same moment the price discounts it to."""
    # Duration is a sensitivity of the full price, the clean price plus accrued interest.
    price = bond_model.get_bond_price(
        100, 0.05, 2.75, 0.05, 2
    ) + bond_model.get_accrued_interest(100, 0.05, 2.75, 2)
    shock = 1e-6
    effective_duration = (
        bond_model.get_bond_price(100, 0.05, 2.75, 0.05 - shock, 2)
        - bond_model.get_bond_price(100, 0.05, 2.75, 0.05 + shock, 2)
    ) / (2 * shock * price)
    modified_duration = bond_model.get_macaulays_duration(100, 0.05, 2.75, 0.05, 2) / (
        1 + 0.05 / 2
    )

    assert modified_duration == pytest.approx(effective_duration, rel=1e-6)


def test_fractional_maturity_matches_quantlib():
    """2.5 years of annual coupons: QuantLib (30/360) gives dirty 100.2043, clean 97.7043."""
    assert bond_model.get_accrued_fraction(2.5, 1) == pytest.approx(0.5)
    assert bond_model.get_accrued_interest(100, 0.05, 2.5, 1) == pytest.approx(2.5)
    np.testing.assert_allclose(
        bond_model.get_cash_flow_periods(2.5, 1), [0.5, 1.5, 2.5]
    )
    assert bond_model.get_bond_price(100, 0.05, 2.5, 0.06, 1) == pytest.approx(
        97.70426717046045, abs=1e-9
    )
    assert bond_model.get_macaulays_duration(100, 0.05, 2.5, 0.06, 1) == pytest.approx(
        2.3573474352559565, abs=1e-9
    )


def test_fractional_maturity_semi_annual_matches_quantlib():
    """2.75 years of semi-annual coupons, reference values from QuantLib (30/360)."""
    assert bond_model.get_bond_price(100, 0.05, 2.75, 0.045, 2) == pytest.approx(
        101.27289635, abs=1e-6
    )
    assert bond_model.get_macaulays_duration(
        100, 0.05, 2.75, 0.045, 2
    ) == pytest.approx(2.57435708, abs=1e-6)
    assert bond_model.get_modified_duration(100, 0.05, 2.75, 0.045, 2) == pytest.approx(
        2.51770864, abs=1e-6
    )
    assert bond_model.get_convexity(100, 0.05, 2.75, 0.045, 2) == pytest.approx(
        7.851, abs=1e-3
    )


def test_macaulay_duration_never_exceeds_the_maturity():
    for maturity in (0.5, 2.5, 7.25, 10.5):
        assert (
            bond_model.get_macaulays_duration(100, 0.05, maturity, 0.06, 1)
            <= maturity + 1e-12
        )


def test_bond_at_its_coupon_yield_prices_near_par_between_coupon_dates():
    """The clean price excludes the accrued coupon, so it stays near par, not par plus accrued."""
    assert bond_model.get_bond_price(100, 0.05, 2.5, 0.05, 1) == pytest.approx(
        100, abs=0.05
    )
    assert bond_model.get_bond_price(100, 0.04, 3.3, 0.04, 4) == pytest.approx(
        100, abs=0.05
    )


def test_yield_to_maturity_inverts_a_fractional_maturity_price():
    price = bond_model.get_bond_price(100, 0.05, 2.75, 0.045, 2)

    assert bond_model.get_yield_to_maturity(
        100, 0.05, 2.75, price, 2, tolerance=1e-10
    ) == pytest.approx(0.045, abs=1e-9)


def test_curve_price_uses_fractional_times_on_a_flat_curve():
    """On a flat curve the curve price equals the yield price, also between coupon dates."""
    flat_curve = pd.Series({0.25: 0.06, 1: 0.06, 5: 0.06, 10: 0.06})

    for maturity, frequency in ((2.5, 1), (2.75, 2), (7.3, 4)):
        assert (
            bond_model._get_bond_price_from_curve(  # pylint: disable=protected-access
                100, 0.05, maturity, flat_curve, frequency
            )
            == pytest.approx(
                bond_model.get_bond_price(100, 0.05, maturity, 0.06, frequency)
            )
        )

    # A bond priced on the curve has a Z-spread of zero.
    price = bond_model._get_bond_price_from_curve(  # pylint: disable=protected-access
        100, 0.05, 2.5, pd.Series({1: 0.03, 2: 0.035, 5: 0.04}), 1
    )
    assert bond_model.get_z_spread(
        100,
        0.05,
        2.5,
        price,
        pd.Series({1: 0.03, 2: 0.035, 5: 0.04}),
        1,
        tolerance=1e-10,
    ) == pytest.approx(0, abs=1e-8)
