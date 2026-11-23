"""Yield Curve Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.curves import yieldcurve_model

FLAT_CURVE_RATE = 0.05


def test_get_forward_rate(recorder):
    recorder.capture(
        yieldcurve_model.get_forward_rate(
            near_rate=0.03, far_rate=0.05, near_maturity=1, far_maturity=3
        )
    )


def test_get_forward_rate_flat_curve_equals_spot_rate(recorder):
    forward_rate = yieldcurve_model.get_forward_rate(
        near_rate=0.05, far_rate=0.05, near_maturity=1, far_maturity=3
    )
    recorder.capture(bool(round(forward_rate, 8) == FLAT_CURVE_RATE))


def test_get_forward_rate_with_series(recorder):
    near_rate = pd.Series([0.03, 0.032])
    far_rate = pd.Series([0.05, 0.052])
    near_maturity = pd.Series([1, 2])
    far_maturity = pd.Series([3, 4])
    recorder.capture(
        yieldcurve_model.get_forward_rate(
            near_rate=near_rate,
            far_rate=far_rate,
            near_maturity=near_maturity,
            far_maturity=far_maturity,
        )
    )


def test_get_forward_rate_invalid_maturity_order():
    with pytest.raises(ValueError):
        yieldcurve_model.get_forward_rate(
            near_rate=0.03, far_rate=0.05, near_maturity=5, far_maturity=1
        )


def test_get_forward_rate_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_forward_rate(
            near_rate="0.03", far_rate=0.05, near_maturity=1, far_maturity=3
        )


def test_get_par_yield(recorder):
    spot_rates = pd.Series({1: 0.03, 2: 0.04, 3: 0.05})
    recorder.capture(
        yieldcurve_model.get_par_yield(
            spot_rates=spot_rates, years_to_maturity=3, frequency=1
        )
    )


def test_get_par_yield_flat_curve_equals_spot_rate(recorder):
    spot_rates = pd.Series({1: 0.05, 2: 0.05, 3: 0.05, 5: 0.05, 10: 0.05})
    par_yield = yieldcurve_model.get_par_yield(
        spot_rates=spot_rates, years_to_maturity=5, frequency=1
    )
    recorder.capture(bool(round(par_yield, 8) == FLAT_CURVE_RATE))


def test_get_par_yield_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_par_yield(
            spot_rates=[0.03, 0.04, 0.05], years_to_maturity=3
        )


def test_get_par_yield_invalid_maturity():
    spot_rates = pd.Series({1: 0.03, 2: 0.04, 3: 0.05})
    with pytest.raises(ValueError):
        yieldcurve_model.get_par_yield(spot_rates=spot_rates, years_to_maturity=0)


def test_get_yield_curve_spread(recorder):
    recorder.capture(
        yieldcurve_model.get_yield_curve_spread(long_yield=0.045, short_yield=0.02)
    )


def test_get_yield_curve_spread_with_series(recorder):
    long_yield = pd.Series([0.045, 0.05])
    short_yield = pd.Series([0.02, 0.025])
    recorder.capture(
        yieldcurve_model.get_yield_curve_spread(
            long_yield=long_yield, short_yield=short_yield
        )
    )


def test_get_yield_curve_spread_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_yield_curve_spread(long_yield="0.045", short_yield=0.02)


def test_get_breakeven_inflation_rate(recorder):
    recorder.capture(
        yieldcurve_model.get_breakeven_inflation_rate(
            nominal_yield=0.045, real_yield=0.018
        )
    )


def test_get_breakeven_inflation_rate_with_series(recorder):
    nominal_yield = pd.Series([0.045, 0.05])
    real_yield = pd.Series([0.018, 0.02])
    recorder.capture(
        yieldcurve_model.get_breakeven_inflation_rate(
            nominal_yield=nominal_yield, real_yield=real_yield
        )
    )


def test_get_breakeven_inflation_rate_type_error():
    with pytest.raises(TypeError):
        yieldcurve_model.get_breakeven_inflation_rate(
            nominal_yield="0.045", real_yield=0.018
        )


def test_get_par_yield_prices_at_par_between_coupon_dates():
    """A bond with the par yield as coupon prices at par on the same curve, at any maturity."""
    from financetoolkit.fixedincome.bonds import bond_model

    spot_rates = pd.Series({1: 0.03, 2: 0.032, 5: 0.038, 10: 0.044})

    for maturity, frequency in ((2.5, 1), (2.75, 2), (5, 1), (7.3, 4)):
        par_yield = yieldcurve_model.get_par_yield(
            spot_rates=spot_rates, years_to_maturity=maturity, frequency=frequency
        )
        assert bond_model.get_bond_price_from_curve(
            100, par_yield, maturity, spot_rates, frequency
        ) == pytest.approx(100)


def test_get_forward_rate_invalid_maturity_order_with_series():
    # One pair with the far maturity before the near one is rejected, not turned into a
    # forward rate over a negative period.
    with pytest.raises(ValueError, match="far maturity"):
        yieldcurve_model.get_forward_rate(
            near_rate=pd.Series([0.03, 0.03]),
            far_rate=pd.Series([0.04, 0.04]),
            near_maturity=pd.Series([1.0, 5.0]),
            far_maturity=pd.Series([2.0, 2.0]),
        )
    with pytest.raises(ValueError, match="far maturity"):
        yieldcurve_model.get_forward_rate(
            near_rate=0.03,
            far_rate=pd.Series([0.04]),
            near_maturity=5.0,
            far_maturity=pd.Series([5.0]),
        )


def test_forward_rate_aligns_differently_indexed_maturities():
    near = pd.Series([1.0, 2.0], index=[0, 1])
    far = pd.Series([3.0, 4.0], index=[1, 2])
    near_rate = pd.Series([0.03, 0.035], index=[0, 1])
    far_rate = pd.Series([0.04, 0.045], index=[1, 2])

    result = yieldcurve_model.get_forward_rate(near_rate, far_rate, near, far)
    shared = yieldcurve_model.get_forward_rate(0.035, 0.04, 2.0, 3.0)

    # Only the shared label holds both maturities; the others are NaN, not an error.
    assert result.loc[1] == pytest.approx(shared)
    assert result.drop(1).isna().all()


# A Treasury-like zero-coupon curve (annually compounded) that rises to 10 years and then
# flattens, so the roll-down changes sign along the curve.
CARRY_CURVE = pd.Series(
    [0.040, 0.041, 0.042, 0.043, 0.045, 0.047, 0.048, 0.047],
    index=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)


def test_get_carry_and_roll_down(recorder):
    recorder.capture(
        yieldcurve_model.get_carry_and_roll_down(
            spot_rates=CARRY_CURVE, maturities=[0.5, 2, 5, 10, 25, 30], horizon=0.25
        )
    )


def test_get_carry_and_roll_down_hand_example():
    # z(1Y) = 4% and z(2Y) = 5% over a year: the 2-year bond earns 5% against 4% financing
    # (carry 1.05 - 1.04) and is then a 1-year bond yielding 4%, priced at 1 / 1.04 instead
    # of 1.05 / 1.05^2 (roll-down 1.05^2 / 1.04 - 1.05).
    result = yieldcurve_model.get_carry_and_roll_down(
        pd.Series([0.04, 0.05], index=[1.0, 2.0]), [2.0], horizon=1.0
    ).loc[2.0]

    assert result["Carry"] == pytest.approx(0.01, abs=1e-15)
    assert result["Roll-Down"] == pytest.approx(1.05**2 / 1.04 - 1.05, abs=1e-15)
    assert result["Total"] == pytest.approx(result["Carry"] + result["Roll-Down"])
    # The duration approximation of Koijen et al. (2018) is close but not exact.
    assert result["Roll-Down"] == pytest.approx(-(1 / 1.05) * (0.04 - 0.05), abs=6e-4)


def test_get_carry_and_roll_down_reprices_on_the_unchanged_curve():
    maturities = np.array([0.5, 1.0, 3.0, 7.0, 10.0, 15.0, 30.0])
    horizon = 0.5
    result = yieldcurve_model.get_carry_and_roll_down(CARRY_CURVE, maturities, horizon)

    def price(maturity):
        rate = np.interp(maturity, CARRY_CURVE.index, CARRY_CURVE.to_numpy())
        return (1 + rate) ** -maturity

    # The total is the return of holding the bond on the unchanged curve minus the return
    # of the bond maturing at the horizon (the financing).
    expected = price(maturities - horizon) / price(maturities) - 1 / price(horizon)
    assert result["Total"].to_numpy() == pytest.approx(expected, abs=1e-14)
    # A bond maturing at the horizon earns exactly the financing rate.
    assert result.loc[0.5].to_numpy() == pytest.approx([0, 0, 0], abs=1e-15)
    # Up to 10 years the curve rises, so the bonds roll down it; beyond 20 it falls.
    assert (result.loc[[1.0, 3.0, 7.0], "Roll-Down"] > 0).all()
    assert result.loc[30.0, "Roll-Down"] < 0


def test_get_carry_and_roll_down_flat_curve_and_own_financing():
    flat = pd.Series(FLAT_CURVE_RATE, index=[0.25, 1.0, 10.0, 30.0])
    result = yieldcurve_model.get_carry_and_roll_down(flat, [2.0, 10.0], horizon=0.25)

    # On a flat curve financed at the curve's own rate, there is neither carry nor roll-down.
    assert result.to_numpy() == pytest.approx(np.zeros((2, 3)), abs=1e-15)

    financed = yieldcurve_model.get_carry_and_roll_down(
        flat, [2.0, 10.0], horizon=0.25, short_rate=0.03
    )
    assert financed["Carry"].to_numpy() == pytest.approx([1.05**0.25 - 1.03**0.25] * 2)
    assert financed["Roll-Down"].to_numpy() == pytest.approx([0, 0], abs=1e-15)


def test_get_carry_and_roll_down_maturity_before_the_horizon():
    result = yieldcurve_model.get_carry_and_roll_down(CARRY_CURVE, [0.5, 2.0], 1.0)

    # A bond repaid before the horizon has no price at the horizon.
    assert result.loc[0.5].isna().all()
    assert result.loc[2.0].notna().all()


def test_get_carry_and_roll_down_invalid_input():
    with pytest.raises(TypeError, match="spot_rates"):
        yieldcurve_model.get_carry_and_roll_down([0.04, 0.05], [2.0])
    with pytest.raises(TypeError, match="horizon"):
        yieldcurve_model.get_carry_and_roll_down(CARRY_CURVE, [2.0], horizon="3M")
    with pytest.raises(TypeError, match="short_rate"):
        yieldcurve_model.get_carry_and_roll_down(CARRY_CURVE, [2.0], short_rate="4%")
    with pytest.raises(ValueError, match="horizon"):
        yieldcurve_model.get_carry_and_roll_down(CARRY_CURVE, [2.0], horizon=0)


def test_get_yield_curve_butterfly():
    assert yieldcurve_model.get_yield_curve_butterfly(
        0.04, 0.045, 0.047
    ) == pytest.approx(0.003)
    with pytest.raises(TypeError):
        yieldcurve_model.get_yield_curve_butterfly(0.04, "0.045", 0.047)
