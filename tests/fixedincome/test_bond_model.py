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
    # Key rate durations, summed across the curve, should approximate the
    # parallel-shift effective duration (same order of magnitude).
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

    # For a small yield shift, the Taylor approximation should closely match
    # the actual repriced bond value.
    recorder.capture(
        bool(abs(taylor_pct - actual_pct) < TAYLOR_APPROXIMATION_TOLERANCE)
    )
