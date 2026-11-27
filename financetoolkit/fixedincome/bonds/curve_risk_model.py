"""Curve Risk Model

The interest rate risk of a set of cash flows (a bond or a portfolio of them) on a yield
curve: the bucketed par-rate delta through the Jacobian of the bootstrap, the bucketed
forward-rate delta, and the parallel shift ladder with its delta and gamma.

The cash flows are amounts by payment time in years, as `bond_model.get_bond_cash_flows` and
`bond_model.get_amortizing_cash_flows` return them; a portfolio is the sum of the cash flows of
its holdings. The zero-coupon curve is read as the Toolkit's curves are returned: annually
compounded zero rates by maturity in years, interpolated linearly in between and held flat
beyond the first and last maturity (see `financetoolkit.fixedincome.helpers.discount_factors`).
"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.fixedincome.helpers import discount_factors

# A basis point, the move the bucketed deltas are expressed per.
BASIS_POINT = 0.0001

# The parallel shifts of the ladder when none are given, from -200 to +200 basis points.
DEFAULT_SHIFTS = (-0.02, -0.01, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.01, 0.02)


def _present_value(cash_flows: pd.Series, zero_curve: pd.Series) -> float:
    """
    The present value of cash flows on an annually compounded zero-coupon curve.

    Args:
        cash_flows (pd.Series): The amounts by payment time in years.
        zero_curve (pd.Series): The annually compounded zero rates by maturity in years.

    Returns:
        float: The sum of the discounted cash flows.
    """
    times = cash_flows.index.to_numpy(dtype=float)

    return float(
        (cash_flows.to_numpy(dtype=float) * discount_factors(zero_curve, times)).sum()
    )


def get_par_to_zero_jacobian(
    par_curve: pd.Series,
    frequency: int = 2,
    bill_convention: str | None = None,
    bump: float = BASIS_POINT,
) -> pd.DataFrame:
    """
    Calculate the Jacobian of the bootstrapped zero-coupon curve with respect to the par
    yields it is bootstrapped from: how much the zero rate of every node of the curve moves
    per unit move of each par yield, the others held fixed.

    The zero curve is the one `curve_model.bootstrap_zero_rates` builds on its full grid:
    the bills (maturities up to a year) keep their own zero rate and every coupon date beyond
    them gets the rate on which a bond paying the par yield interpolated linearly to that date
    prices at par. Its nodes depend on the maturities only, so every bumped curve has the same
    nodes. Each column is a central difference of the bootstrap:

    - J(t, s_i) = (z(t; s_i + h) - z(t; s_i - h)) / (2h)

    with z(t; s) the annually compounded zero rate of node t bootstrapped from the par yields s
    and h = bump. A par yield leaves the zero rates up to the previous quoted maturity alone,
    so the Jacobian is lower triangular in blocks; it moves the nodes up to the next quoted
    maturity most (through the interpolated par yields) and the later ones a little (through
    the discount factors of the earlier coupons).

    For more information, see Darbyshire (2022), Pricing and Trading Interest Rate Derivatives,
    3rd edition, chapters 7 and 8, and Andersen and Piterbarg (2010), Interest Rate Modeling,
    volume 3, chapter 26.

    Args:
        par_curve (pd.Series): The par yields (decimals, compounded `frequency` times a year
            beyond a year, the bills in `bill_convention`) by maturity in years.
        frequency (int, optional): The coupons per year of the par bonds. Defaults to 2.
        bill_convention (str | None, optional): How the bill yields are quoted, see
            `curve_model.bill_yields_to_annual`. Options are: "bond-equivalent" (the US
            Treasury), "act/365" (Canada), "act/360" (Sweden), None (yields compounded
            `frequency` times a year). Defaults to None.
        bump (float, optional): The move of each par yield up and down (decimal). Defaults to
            0.0001 (1 basis point).

    Returns:
        pd.DataFrame: The zero rate moves per unit par yield move, with the nodes of the zero
        curve (maturity in years) as rows and the par maturities as columns.

    Raises:
        ValueError: If the bump is not positive.
    """
    if bump <= 0:
        raise ValueError(f"bump must be positive, received {bump!r}.")
    par_curve = par_curve.dropna().sort_index()
    maturities = par_curve.index.to_numpy(dtype=float)
    par_yields = par_curve.to_numpy(dtype=float)

    nodes, _ = curve_model.bootstrap_zero_rates(
        maturities, par_yields, frequency, bill_convention, return_grid=True
    )
    jacobian = np.empty((nodes.size, maturities.size))
    for column in range(maturities.size):
        bumped = np.zeros(maturities.size)
        bumped[column] = bump
        _, zero_up = curve_model.bootstrap_zero_rates(
            maturities, par_yields + bumped, frequency, bill_convention, True
        )
        _, zero_down = curve_model.bootstrap_zero_rates(
            maturities, par_yields - bumped, frequency, bill_convention, True
        )
        jacobian[:, column] = (zero_up - zero_down) / (2 * bump)

    return pd.DataFrame(jacobian, index=nodes, columns=par_curve.index)


def get_par_rate_delta(
    cash_flows: pd.Series,
    par_curve: pd.Series,
    frequency: int = 2,
    bill_convention: str | None = None,
    bump: float = BASIS_POINT,
) -> pd.Series:
    """
    Calculate the bucketed par-rate delta of a set of cash flows: the change in their
    present value when one par yield of the curve rises by a basis point and the zero curve
    is bootstrapped again, for every par yield.

    These are the sensitivities to the instruments a trader hedges with (a 2-year note, a
    10-year note), where `bond_model.get_key_rate_duration` bumps a zero rate with a tent
    around it.

    The delta follows from the chain rule through the par-to-zero Jacobian of the bootstrap
    rather than from a full rebuild per par yield:

    - PV = SUM(CF_k * (1 + z(t_k))^(-t_k))
    - dPV / dz_j = -SUM(t_k * CF_k * (1 + z(t_k))^(-t_k - 1) * w_kj)
    - Delta_i = SUM_j(dPV / dz_j * J(z_j, s_i)) * 0.0001

    where z(t_k) is the annually compounded zero rate at the k-th payment time, interpolated
    linearly between the nodes z_j of the bootstrapped curve with weights w_kj (held flat
    beyond the first and last node), and J is the Jacobian of `get_par_to_zero_jacobian`. The
    result is the present value change per +1 basis point of each par yield, in the currency
    of the cash flows: negative for a long bond position. It equals bumping each par yield,
    bootstrapping again and repricing to second order in the bump.

    A par bond that matures at a quoted maturity on a coupon date has a delta to its own par
    yield only: the bootstrap prices it at par whatever the other par yields are. Summed over
    the par yields, the deltas give the present value change for a parallel shift of the par
    curve.

    For more information, see Darbyshire (2022), Pricing and Trading Interest Rate Derivatives,
    3rd edition, chapters 7 and 8.

    Shares similar logic:
    - rateslib, Solver.delta: https://rateslib.com/py/en/3.0.x/api/rateslib.solver.Solver.html#rateslib.solver.Solver.delta
    - FinancePy, par_rate_risk_report: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/market/curves/ibor_curve_risk_engine.py#L25

    Args:
        cash_flows (pd.Series): The cash flow amounts by payment time in years.
        par_curve (pd.Series): The par yields (decimals) by maturity in years, see
            `get_par_to_zero_jacobian`.
        frequency (int, optional): The coupons per year of the par bonds. Defaults to 2.
        bill_convention (str | None, optional): How the bill yields are quoted, see
            `curve_model.bill_yields_to_annual`. Options are: "bond-equivalent" (the US
            Treasury), "act/365" (Canada), "act/360" (Sweden), None (yields compounded
            `frequency` times a year). Defaults to None.
        bump (float, optional): The move of each par yield for the Jacobian (decimal).
            Defaults to 0.0001 (1 basis point).

    Returns:
        pd.Series: The present value change per +1 basis point of each par yield, by par
        maturity.

    Raises:
        ValueError: If the bump is not positive.
    """
    par_curve = par_curve.dropna().sort_index()
    maturities = par_curve.index.to_numpy(dtype=float)
    nodes, zero_rates = curve_model.bootstrap_zero_rates(
        maturities,
        par_curve.to_numpy(dtype=float),
        frequency,
        bill_convention,
        return_grid=True,
    )
    jacobian = get_par_to_zero_jacobian(par_curve, frequency, bill_convention, bump)

    times = cash_flows.index.to_numpy(dtype=float)
    amounts = cash_flows.to_numpy(dtype=float)
    rates = np.interp(times, nodes, zero_rates)

    # The interpolation weight of every node in the zero rate of every payment time, the
    # flat extrapolation beyond the ends included.
    weights = np.column_stack(
        [np.interp(times, nodes, np.eye(nodes.size)[j]) for j in range(nodes.size)]
    )
    rate_sensitivity = -times * amounts * (1 + rates) ** (-times - 1)
    node_delta = rate_sensitivity @ weights

    return pd.Series(
        node_delta @ jacobian.to_numpy() * BASIS_POINT,
        index=par_curve.index,
        name="Par Rate Delta",
    )


def get_forward_rate_delta(
    cash_flows: pd.Series,
    zero_curve: pd.Series,
    buckets: list[float] | np.ndarray | None = None,
    bump: float = BASIS_POINT,
) -> pd.Series:
    """
    Calculate the bucketed forward-rate delta of a set of cash flows: the change in their
    present value when the instantaneous forward rate over one bucket of maturities rises by
    a basis point, for every bucket.

    Unlike a zero-rate bump, a forward bump also moves every discount factor beyond the
    bucket by the same factor, so the deltas show where along the curve the forward rates
    the cash flows depend on lie.

    The instantaneous forward rate is continuously compounded, so raising it by h over the
    bucket (a, b] multiplies the discount factor of a payment at t by exp(-h * overlap), the
    part of (a, b] before t:

    - overlap_k = max(0, min(t_k, b) - a)
    - PV(h) = SUM(CF_k * P(t_k) * exp(-h * overlap_k))
    - Delta = (PV(+h) - PV(-h)) / 2 * 0.0001 / h

    where P(t) is the discount factor of the annually compounded zero curve and h = bump. The
    first bucket starts today and the last one has no end, so the deltas add up to the present
    value change for a parallel shift of the continuously compounded zero rates.

    For more information, see Andersen and Piterbarg (2010), Interest Rate Modeling, volume 3,
    chapter 26.

    Shares similar logic:
    - FinancePy, forward_rate_risk_report: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/market/curves/ibor_curve_risk_engine.py#L84

    Args:
        cash_flows (pd.Series): The cash flow amounts by payment time in years.
        zero_curve (pd.Series): The annually compounded zero rates (decimals) by maturity in
            years.
        buckets (list[float] | np.ndarray | None, optional): The ends of the buckets in years,
            ascending. Defaults to None, the maturities of the zero curve.
        bump (float, optional): The move of the forward rate up and down (decimal). Defaults
            to 0.0001 (1 basis point).

    Returns:
        pd.Series: The present value change per +1 basis point of the forward rate of each
        bucket, by the end of the bucket in years.

    Raises:
        ValueError: If the bump is not positive.
    """
    if bump <= 0:
        raise ValueError(f"bump must be positive, received {bump!r}.")
    ends = np.sort(
        np.asarray(
            zero_curve.dropna().index if buckets is None else buckets, dtype=float
        )
    )
    starts = np.concatenate([[0.0], ends[:-1]])
    # The last bucket runs on beyond its end, so every payment falls in some bucket.
    limits = np.concatenate([ends[:-1], [np.inf]])

    times = cash_flows.index.to_numpy(dtype=float)
    present_values = cash_flows.to_numpy(dtype=float) * discount_factors(
        zero_curve, times
    )

    deltas = []
    for start, limit in zip(starts, limits, strict=True):
        overlap = np.clip(np.minimum(times, limit) - start, 0.0, None)
        value_up = (present_values * np.exp(-bump * overlap)).sum()
        value_down = (present_values * np.exp(bump * overlap)).sum()
        deltas.append((value_up - value_down) / 2 * BASIS_POINT / bump)

    return pd.Series(deltas, index=ends, name="Forward Rate Delta")


def get_parallel_shift_ladder(
    cash_flows: pd.Series,
    zero_curve: pd.Series,
    shifts: list[float] | np.ndarray | None = None,
    bump: float = BASIS_POINT,
) -> pd.DataFrame:
    """
    Calculate the parallel shift ladder of a set of cash flows: their present value and
    profit or loss when the whole zero-coupon curve shifts by each of a range of amounts,
    with the delta and gamma at every shift.

    Where duration and convexity extrapolate from today's curve, the ladder reprices at
    every shift, so it shows how far the second-order estimate holds.

    The formula is as follows:

    - PV(Δ) = SUM(CF_k * (1 + z(t_k) + Δ)^(-t_k))
    - P&L(Δ) = PV(Δ) - PV(0)
    - Delta(Δ) = (PV(Δ + h) - PV(Δ - h)) / (2h)
    - Gamma(Δ) = (PV(Δ + h) - 2 * PV(Δ) + PV(Δ - h)) / h^2
    - Delta-Gamma P&L(Δ) = Delta(0) * Δ + 0.5 * Gamma(0) * Δ^2

    where z(t) is the annually compounded zero rate of the curve at the payment time t (so Δ
    shifts annually compounded rates), and h = bump. Delta and gamma are per unit of rate (a
    shift of 1.0, i.e. 10,000 basis points), so a shift Δ given as a decimal multiplies them
    directly: a delta of -700 loses 0.07 for a 1 basis point rise. On a flat curve they are
    minus the price times the modified duration and the price times the convexity of an
    annual-pay bond at that yield.

    For more information, see Tuckman and Serrat (2022), Fixed Income Securities, 4th edition,
    chapters 4 to 6.

    Shares similar logic:
    - FinancePy, parallel_shift_ladder_report: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/market/curves/ibor_curve_risk_engine.py#L291

    Args:
        cash_flows (pd.Series): The cash flow amounts by payment time in years.
        zero_curve (pd.Series): The annually compounded zero rates (decimals) by maturity in
            years.
        shifts (list[float] | np.ndarray | None, optional): The parallel shifts (decimals).
            Defaults to None, -200, -100, -50, -25, 0, 25, 50, 100 and 200 basis points.
        bump (float, optional): The step of the delta and gamma differences (decimal).
            Defaults to 0.0001 (1 basis point).

    Returns:
        pd.DataFrame: By shift (decimal), the Present Value, P&L, Delta-Gamma P&L, Delta and
        Gamma.

    Raises:
        ValueError: If the bump is not positive.
    """
    if bump <= 0:
        raise ValueError(f"bump must be positive, received {bump!r}.")
    shifts = np.asarray(DEFAULT_SHIFTS if shifts is None else shifts, dtype=float)
    zero_curve = zero_curve.dropna().sort_index()

    def present_value(shift: float) -> float:
        return _present_value(cash_flows, zero_curve + shift)

    base_value = present_value(0.0)
    base_up, base_down = present_value(bump), present_value(-bump)
    base_delta = (base_up - base_down) / (2 * bump)
    base_gamma = (base_up - 2 * base_value + base_down) / bump**2

    rows = {}
    for shift in shifts:
        value = present_value(shift)
        value_up, value_down = present_value(shift + bump), present_value(shift - bump)
        rows[float(shift)] = {
            "Present Value": value,
            "P&L": value - base_value,
            "Delta-Gamma P&L": base_delta * shift + 0.5 * base_gamma * shift**2,
            "Delta": (value_up - value_down) / (2 * bump),
            "Gamma": (value_up - 2 * value + value_down) / bump**2,
        }

    ladder = pd.DataFrame.from_dict(rows, orient="index")
    ladder.index.name = "Shift"

    return ladder
