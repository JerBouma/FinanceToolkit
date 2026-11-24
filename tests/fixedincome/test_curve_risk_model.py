"""Curve Risk Model Tests"""

import pandas as pd
import pytest

from financetoolkit.fixedincome.bonds import bond_model, curve_risk_model
from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.fixedincome.helpers import discount_factors

# A Treasury par curve by maturity in years, bills quoted as bond-equivalent yields.
PAR_CURVE = pd.Series(
    {
        1 / 12: 0.039,
        0.25: 0.0395,
        0.5: 0.0405,
        1: 0.042,
        2: 0.0445,
        3: 0.0452,
        5: 0.046,
        7: 0.047,
        10: 0.0484,
        20: 0.0527,
        30: 0.0525,
    }
)
CONVENTION = "bond-equivalent"


def zero_curve(par_curve: pd.Series) -> pd.Series:
    nodes, rates = curve_model.bootstrap_zero_rates(
        par_curve.index.to_numpy(dtype=float),
        par_curve.to_numpy(dtype=float),
        2,
        CONVENTION,
        return_grid=True,
    )
    return pd.Series(rates, index=nodes)


def present_value(cash_flows: pd.Series, par_curve: pd.Series) -> float:
    return curve_risk_model._present_value(cash_flows, zero_curve(par_curve))


def test_par_rate_delta_matches_bumping_and_bootstrapping_again():
    """The Jacobian chain rule equals a full rebuild per par yield to second order."""
    cash_flows = bond_model.get_bond_cash_flows(100, 0.03, 13.3, 2)
    deltas = curve_risk_model.get_par_rate_delta(cash_flows, PAR_CURVE, 2, CONVENTION)

    for maturity in PAR_CURVE.index:
        up, down = PAR_CURVE.copy(), PAR_CURVE.copy()
        up[maturity] += 1e-4
        down[maturity] -= 1e-4
        rebuilt = (present_value(cash_flows, up) - present_value(cash_flows, down)) / 2
        assert deltas[maturity] == pytest.approx(rebuilt, abs=1e-8)

    # Summed, a parallel shift of the par curve.
    parallel = (
        present_value(cash_flows, PAR_CURVE + 1e-4)
        - present_value(cash_flows, PAR_CURVE - 1e-4)
    ) / 2
    assert deltas.sum() == pytest.approx(parallel, rel=1e-6)


def test_par_bond_has_a_delta_to_its_own_par_yield_only():
    """A par bond at a quoted maturity prices at par whatever the other par yields are."""
    cash_flows = bond_model.get_bond_cash_flows(100, 0.0484, 10, 2)
    assert present_value(cash_flows, PAR_CURVE) == pytest.approx(100, abs=1e-10)

    deltas = curve_risk_model.get_par_rate_delta(cash_flows, PAR_CURVE, 2, CONVENTION)
    assert deltas.drop(10).abs().max() < 1e-9
    # Its coupon moves with the par yield, so the delta is minus its annuity per basis point.
    annuity = (
        0.5 * discount_factors(zero_curve(PAR_CURVE), cash_flows.index.to_numpy()).sum()
    )
    assert deltas[10] == pytest.approx(-100 * annuity * 1e-4, rel=1e-6)


def test_par_to_zero_jacobian():
    """No par yield moves the zero rates before the previous quoted maturity."""
    jacobian = curve_risk_model.get_par_to_zero_jacobian(PAR_CURVE, 2, CONVENTION)

    assert jacobian.shape == (len(zero_curve(PAR_CURVE)), len(PAR_CURVE))
    # A bill's zero rate moves with its own yield alone.
    assert jacobian.loc[0.25, 0.25] > 0.9
    assert jacobian.loc[0.25].drop(0.25).abs().max() == 0
    # The 5-year par yield moves nothing up to 3 years, the nodes up to 7 years most and the
    # later ones through the discount factors of the earlier coupons only.
    assert jacobian.loc[jacobian.index <= 3, 5].abs().max() == 0
    assert jacobian.loc[5, 5] > 1
    assert jacobian.loc[jacobian.index > 7, 5].abs().max() < 0.1


def test_forward_rate_delta_adds_up_to_a_continuous_parallel_shift(recorder):
    """The buckets cover every payment, so together they shift the continuous zero rates."""
    cash_flows = bond_model.get_bond_cash_flows(100, 0.03, 13.3, 2)
    curve = zero_curve(PAR_CURVE)
    deltas = curve_risk_model.get_forward_rate_delta(cash_flows, curve, PAR_CURVE.index)

    times = cash_flows.index.to_numpy()
    discounted = cash_flows.to_numpy() * discount_factors(curve, times)
    assert deltas.sum() == pytest.approx(-(discounted * times).sum() * 1e-4, rel=1e-8)
    # Nothing is paid after 20 years, and the 13.3-year payments reach into the 20Y bucket.
    assert deltas[30] == 0
    assert deltas[20] < 0
    recorder.capture(deltas.round(10))


def test_parallel_shift_ladder_on_a_flat_curve():
    """On a flat curve delta and gamma are the bond's price times duration and convexity."""
    cash_flows = bond_model.get_bond_cash_flows(100, 0.05, 7, 1)
    ladder = curve_risk_model.get_parallel_shift_ladder(
        cash_flows, pd.Series(0.05, index=[0.5, 30.0])
    )

    assert ladder.index.tolist() == list(curve_risk_model.DEFAULT_SHIFTS)
    assert ladder.loc[0.0, "Present Value"] == pytest.approx(100)
    assert ladder.loc[0.0, "Delta"] == pytest.approx(
        -100 * bond_model.get_modified_duration(100, 0.05, 7, 0.05, 1), rel=1e-6
    )
    assert ladder.loc[0.0, "Gamma"] == pytest.approx(
        100 * bond_model.get_convexity(100, 0.05, 7, 0.05, 1), rel=1e-6
    )
    for shift in (-0.02, 0.01):
        assert ladder.loc[shift, "Present Value"] == pytest.approx(
            bond_model.get_bond_price(100, 0.05, 7, 0.05 + shift, 1)
        )
    # The second-order estimate holds for small shifts and drifts off for large ones.
    error = (ladder["P&L"] - ladder["Delta-Gamma P&L"]).abs()
    assert error[0.0025] < 1e-3
    assert 0.01 < error[0.02] < 0.1


def test_parallel_shift_ladder_of_a_portfolio(recorder):
    """The ladder of a portfolio is the sum of the ladders of its bonds."""
    bond = bond_model.get_bond_cash_flows(100, 0.04, 9.5, 2)
    curve = zero_curve(PAR_CURVE)
    single = curve_risk_model.get_parallel_shift_ladder(bond, curve, [-0.01, 0, 0.01])
    portfolio = curve_risk_model.get_parallel_shift_ladder(
        bond + bond_model.get_bond_cash_flows(50, 0.04, 9.5, 2),
        curve,
        [-0.01, 0, 0.01],
    )

    assert portfolio.loc[0.0, "P&L"] == 0
    pd.testing.assert_frame_equal(portfolio, 1.5 * single, rtol=1e-6)
    # The gamma is a second difference over 1 basis point, so it is recorded to 0.1.
    recorder.capture(
        portfolio.round(
            {"Present Value": 6, "P&L": 6, "Delta-Gamma P&L": 4, "Delta": 4, "Gamma": 1}
        )
    )


def test_a_bump_that_is_not_positive_is_refused():
    # The differences divide by the bump: a bump of 0 gave NaN deltas.
    cash_flows = bond_model.get_bond_cash_flows(100, 0.04, 5, 2)
    with pytest.raises(ValueError, match="bump must be positive"):
        curve_risk_model.get_par_rate_delta(
            cash_flows, PAR_CURVE, bill_convention=CONVENTION, bump=0
        )
    with pytest.raises(ValueError, match="bump must be positive"):
        curve_risk_model.get_forward_rate_delta(
            cash_flows, zero_curve(PAR_CURVE), bump=0
        )
    with pytest.raises(ValueError, match="bump must be positive"):
        curve_risk_model.get_parallel_shift_ladder(
            cash_flows, zero_curve(PAR_CURVE), bump=-0.0001
        )
