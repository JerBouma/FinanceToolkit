"""Yield Curve Interpolation and Bootstrapping Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.utilities.error_model import check_option

# The ways a zero-coupon curve is read between its maturities. "linear_par" is the curve
# of get_zero_coupon_curve: the par yields interpolated linearly to every coupon date
# before the bootstrap, and the annually compounded zero rates linearly between those
# dates. The others interpolate the zero curve the bootstrap solves for, so every quoted
# par bond prices at par on the interpolated curve itself (Hagan and West, 2006).
INTERPOLATIONS = (
    "linear_par",
    "linear_zero",
    "log_linear",
    "natural_cubic",
    "monotone_cubic",
    "monotone_convex",
)

# The bootstrap of the interpolations whose shape between two maturities depends on the
# maturities further out (the cubic splines and monotone convex) is repeated until no
# zero rate moves by more than this, and gives up after this many passes.
BOOTSTRAP_TOLERANCE = 1e-13
BOOTSTRAP_PASSES = 200

# Maturities, in years, closer than this count as the same (floating point noise).
EPSILON = 1e-9


def _monotone_convex(
    maturities: np.ndarray, zero_rates: np.ndarray, years: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    The monotone convex interpolation of Hagan and West (2006): the integral of the
    instantaneous forward curve (r(t) t) and the forward itself at `years`, for
    t up to the last maturity.

    The discrete forwards f^d_i between the maturities are kept exactly: the forward
    on each interval is f^d_i + g(x), with x the position in the interval and g a
    quadratic that integrates to zero, through the instantaneous forwards at the
    interval's ends. Where that quadratic would make the forward overshoot (lose
    monotonicity or convexity), it is replaced by the piecewise forms of sectors (ii)
    to (iv) of Hagan and West, which are flat over part of the interval.
    """
    times = np.concatenate([[0.0], maturities])
    integrals = np.concatenate([[0.0], zero_rates * maturities])
    discrete = np.diff(integrals) / np.diff(times)

    # The instantaneous forwards at the maturities: a weighted average of the discrete
    # forwards on either side, and at both ends the extrapolation that keeps the
    # forward's slope, as Hagan and West (2006) set them. The forwards are not collared at
    # zero (their positivity option), as government curves can have negative rates.
    nodes = np.empty(len(times))
    if len(discrete) > 1:
        widths = np.diff(times)
        nodes[1:-1] = (widths[:-1] * discrete[1:] + widths[1:] * discrete[:-1]) / (
            widths[:-1] + widths[1:]
        )
        nodes[0] = discrete[0] - 0.5 * (nodes[1] - discrete[0])
        nodes[-1] = discrete[-1] - 0.5 * (nodes[-2] - discrete[-1])
    else:
        nodes[:] = discrete[0]

    interval = np.clip(np.searchsorted(times, years, side="right"), 1, len(times) - 1)
    start, width = times[interval - 1], np.diff(times)[interval - 1]
    x = np.clip((years - start) / width, 0.0, 1.0)
    level = discrete[interval - 1]
    g0 = nodes[interval - 1] - level
    g1 = nodes[interval] - level

    g = np.zeros_like(x)
    integral = np.zeros_like(x)

    # Sector (i): the quadratic g0 (1 - 4x + 3x²) + g1 (-2x + 3x²) stays monotone.
    sector_1 = ((g0 < 0) & (-0.5 * g0 <= g1) & (g1 <= -2 * g0)) | (
        (g0 > 0) & (-0.5 * g0 >= g1) & (g1 >= -2 * g0)
    )
    # Sector (ii): g1 is too large for g0, so g stays at g0 up to η and then rises.
    sector_2 = ((g0 < 0) & (g1 > -2 * g0)) | ((g0 > 0) & (g1 < -2 * g0))
    # Sector (iii): g1 is too small, so g falls to g1 at η and stays there.
    sector_3 = ((g0 > 0) & (g1 < 0) & (g1 > -0.5 * g0)) | (
        (g0 < 0) & (g1 > 0) & (g1 < -0.5 * g0)
    )
    # Sector (iv): g0 and g1 have the same sign, so g passes through A at η.
    sector_4 = ((g0 > 0) & (g1 > 0)) | ((g0 < 0) & (g1 < 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        g[sector_1] = (g0 * (1 - 4 * x + 3 * x**2) + g1 * (-2 * x + 3 * x**2))[sector_1]
        integral[sector_1] = (
            g0 * (x - 2 * x**2 + x**3) + g1 * (-(x**2) + x**3)
        )[sector_1]

        eta = (g1 + 2 * g0) / (g1 - g0)
        after = x > eta
        rise = np.where(after, (x - eta) / (1 - eta), 0.0)
        g[sector_2] = (g0 + (g1 - g0) * rise**2)[sector_2]
        integral[sector_2] = (
            g0 * x + np.where(after, (g1 - g0) * (x - eta) * rise**2 / 3, 0.0)
        )[sector_2]

        eta = 3 * g1 / (g1 - g0)
        before = x < eta
        fall = np.where(before, (eta - x) / eta, 0.0)
        g[sector_3] = (g1 + (g0 - g1) * fall**2)[sector_3]
        integral[sector_3] = (
            g1 * x
            + (g0 - g1)
            * np.where(before, (eta**3 - (eta - x) ** 3), eta**3)
            / (3 * eta**2)
        )[sector_3]

        eta = g1 / (g1 + g0)
        a = -g0 * g1 / (g0 + g1)
        before = x <= eta
        fall = np.where(before, (eta - x) / eta, 0.0)
        rise = np.where(before, 0.0, (x - eta) / (1 - eta))
        g[sector_4] = np.where(
            before, a + (g0 - a) * fall**2, a + (g1 - a) * rise**2
        )[sector_4]
        integral[sector_4] = (
            a * x
            + (g0 - a)
            * np.where(before, eta**3 - (eta - x) ** 3, eta**3)
            / (3 * eta**2)
            + np.where(before, 0.0, (g1 - a) * (x - eta) * rise**2 / 3)
        )[sector_4]

    forwards = level + g
    log_discount = integrals[interval - 1] + width * (level * x + integral)

    return log_discount, forwards


def _interpolate(
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    years: np.ndarray,
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The integral of the instantaneous forward curve, r(t) t = -ln P(t), and the
    instantaneous forward f(t) = -d ln P(t) / dt, at `years` within the last maturity.

    Args:
        maturities (np.ndarray): The maturities of the curve in years, ascending and above 0.
        zero_rates (np.ndarray): The continuously compounded zero rates at the maturities.
        years (np.ndarray): The times to read the curve at, from 0 up to the last maturity.
        interpolation (str): One of INTERPOLATIONS.

    Returns:
        tuple[np.ndarray, np.ndarray]: r(t) t and f(t).
    """
    if interpolation == "monotone_convex":
        return _monotone_convex(maturities, zero_rates, years)

    if interpolation == "log_linear":
        # ln P(t) linear between the maturities and from P(0) = 1: the forward is flat
        # between two maturities (Hagan and West's raw interpolation).
        times = np.concatenate([[0.0], maturities])
        integrals = np.concatenate([[0.0], zero_rates * maturities])
        forwards = np.diff(integrals) / np.diff(times)
        interval = np.clip(
            np.searchsorted(times, years, side="right"), 1, len(times) - 1
        )
        return (
            integrals[interval - 1]
            + forwards[interval - 1] * (years - times[interval - 1]),
            forwards[interval - 1],
        )

    if interpolation == "linear_par":
        # Linear on the annually compounded zero rates, flat before the first maturity,
        # as get_forward_rate and get_par_yield read the curve of get_zero_coupon_curve.
        annual = np.expm1(zero_rates)
        rates = np.interp(years, maturities, annual)
        interval = np.clip(
            np.searchsorted(maturities, years, side="right"), 1, len(maturities) - 1
        )
        slopes = np.diff(annual) / np.diff(maturities) if len(maturities) > 1 else [0.0]
        slope = np.where(years < maturities[0], 0.0, np.asarray(slopes)[interval - 1])
        return years * np.log1p(rates), np.log1p(rates) + years * slope / (1 + rates)

    # The zero rate curve starts at t = 0 with the rate of the first maturity, so it is
    # flat before it.
    times = np.concatenate([[0.0], maturities])
    rates = np.concatenate([[zero_rates[0]], zero_rates])

    if interpolation == "linear_zero":
        interval = np.clip(
            np.searchsorted(times, years, side="right"), 1, len(times) - 1
        )
        slope = np.diff(rates)[interval - 1] / np.diff(times)[interval - 1]
        rate = np.interp(years, times, rates)
        return rate * years, rate + years * slope

    spline = CubicSpline(times, rates, bc_type="natural")

    if interpolation == "monotone_cubic":
        # Hyman's (1983) filter on the slopes of the natural spline: at a maturity where
        # the zero rates keep rising (or falling) the slope is held between zero and three
        # times the smaller of the neighbouring secants, so the curve does not overshoot;
        # at a local extremum it is zero. The ends are held within three times their secant.
        secants = np.diff(rates) / np.diff(times)
        slopes = spline(times, 1)
        filtered = np.zeros_like(slopes)
        for node in range(len(times)):
            sides = secants[max(node - 1, 0) : node + 1]
            if np.all(sides > 0) or np.all(sides < 0):
                sign = np.sign(sides[0])
                filtered[node] = sign * min(
                    max(0.0, sign * slopes[node]), 3 * np.min(np.abs(sides))
                )
        spline = CubicHermiteSpline(times, rates, filtered)

    rate = spline(years)
    return rate * years, rate + years * spline(years, 1)


def _read(
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    years: np.ndarray | float,
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    r(t) t and f(t) at any times: within the curve as `_interpolate` reads it, beyond its
    last maturity with the zero rate held at the last one, as the other curves of the
    Finance Toolkit do.
    """
    check_option("interpolation", interpolation, INTERPOLATIONS)
    maturities = np.asarray(maturities, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)
    if maturities.size == 0 or maturities.shape != zero_rates.shape:
        raise ValueError(
            f"The curve needs a zero rate for each maturity, got {maturities.size} "
            f"maturities and {zero_rates.size} zero rates."
        )
    order = np.argsort(maturities)
    maturities, zero_rates = maturities[order], zero_rates[order]
    years = np.atleast_1d(np.asarray(years, dtype=float))
    if np.any(years < 0):
        raise ValueError("The times to read the curve at must be 0 or more.")

    beyond = years > maturities[-1]
    log_discount, forwards = _interpolate(
        maturities, zero_rates, np.where(beyond, maturities[-1], years), interpolation
    )
    log_discount = np.where(beyond, zero_rates[-1] * years, log_discount)
    forwards = np.where(beyond, zero_rates[-1], forwards)

    return log_discount, forwards


def get_discount_factors(
    maturities: np.ndarray | list,
    zero_rates: np.ndarray | list,
    years: np.ndarray | list | float,
    interpolation: str = "monotone_convex",
) -> np.ndarray:
    """
    Calculate the discount factors of a zero-coupon curve at any times: what a payment at
    each time is worth today, with a choice of how the curve is read between its maturities
    (Hagan and West, 2006):

    - Linear zero: the zero rate r(t) is linear between the maturities.
    - Log-linear: ln P(t) = -r(t) t is linear between the maturities and from P(0) = 1,
      so the instantaneous forward is flat between two maturities. Hagan and West call
      this raw interpolation; it is also known as piecewise flat forward.
    - Natural cubic: a natural cubic spline (zero second derivative at both ends) on
      the zero rates.
    - Monotone cubic: a cubic on the zero rates with the slopes of the natural spline,
      held by Hyman's (1983) filter so the curve does not overshoot where the zero rates
      rise or fall monotonically.
    - Monotone convex: the monotone convex method of Hagan and West (2006): the
      instantaneous forward on each interval is the interval's discrete forward plus a
      quadratic (or, where that would overshoot, a partly flat form) that integrates to
      zero, so the forward curve is continuous, keeps the discrete forwards exactly and
      does not oscillate.
    - Linear par: linear on the annually compounded zero rates, the way
      `get_forward_rate` and `get_par_yield` read the curve of `get_zero_coupon_curve`.

    The zero rates are continuously compounded, as Hagan and West define them, so a
    payment in t years is discounted with P(t) = e^(-r(t) t). Before the first maturity the
    zero rate is that of the first maturity (monotone convex extrapolates the forward
    instead, as Hagan and West do), and beyond the last maturity it is held at the last
    one.

    For more information, see the following papers:

    - Hagan, P.S. and West, G. (2006). "Interpolation Methods for Curve Construction."
      Applied Mathematical Finance, 13(2), 89-129.
    - Hagan, P.S. and West, G. (2008). "Methods for Constructing a Yield Curve." Wilmott
      Magazine, May, 70-81.
    - Hyman, J.M. (1983). "Accurate Monotonicity Preserving Cubic Interpolation." SIAM
      Journal on Scientific and Statistical Computing, 4(4), 645-654.

    QuantLib's convex monotone forward curve is taken without forced positivity.

    Referenced against:
    - QuantLib, ZeroCurve (linear_zero): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
    - QuantLib, DiscountCurve (log_linear): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_discount_curve.html
    - QuantLib, NaturalCubicZeroCurve (natural_cubic): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
    - QuantLib, MonotonicCubicZeroCurve (monotone_cubic): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
    - QuantLib, PiecewiseYieldCurve with ConvexMonotone (monotone_convex): https://www.quantlib.org/reference/class_quant_lib_1_1_convex_monotone.html

    Args:
        maturities (np.ndarray | list): The maturities of the curve in years, above 0.
        zero_rates (np.ndarray | list): The continuously compounded zero rates (decimals),
            one for each maturity.
        years (np.ndarray | list | float): The times to discount from, in years.
        interpolation (str, optional): How the curve is read between its maturities. Options
            are: "linear_zero" (linear zero rates), "log_linear" (linear log discount
            factors), "natural_cubic" (a natural cubic spline), "monotone_cubic" (a monotone
            cubic spline), "monotone_convex" (Hagan and West's monotone convex method),
            "linear_par" (linear annually compounded zero rates). Defaults to
            "monotone_convex".

    Returns:
        np.ndarray: The discount factor at each time.

    Raises:
        ValueError: If the interpolation does not exist, a time is negative, or there is not
            one zero rate for each maturity.
    """
    log_discount, _ = _read(maturities, zero_rates, years, interpolation)

    return np.exp(-log_discount)


def get_coupon_spot_rates(
    maturities: np.ndarray | list,
    zero_rates: np.ndarray | list,
    years_to_maturity: float,
    frequency: int,
    interpolation: str = "monotone_convex",
) -> pd.Series:
    """
    Calculate the spot rates, compounded at the coupon frequency, that a par yield discounts
    with: on the bootstrap's coupon dates for linear par, between which it interpolates them
    linearly, and otherwise read from the interpolated curve at the bond's own coupon dates.

    Args:
        maturities (np.ndarray | list): The maturities of the curve in years.
        zero_rates (np.ndarray | list): The continuously compounded zero rates.
        years_to_maturity (float): The maturity of the bond in years.
        frequency (int): The number of coupons a year.
        interpolation (str, optional): How the curve is read between its maturities, see
            get_discount_factors. Defaults to "monotone_convex".

    Returns:
        pd.Series: The spot rates, indexed by time in years.
    """
    times = (
        np.asarray(maturities, dtype=float)
        if interpolation == "linear_par"
        else bond_model.get_cash_flow_periods(years_to_maturity, frequency) / frequency
    )
    discount = get_discount_factors(maturities, zero_rates, times, interpolation)
    return pd.Series(
        frequency * (discount ** (-1 / (times * frequency)) - 1), index=times
    )


def get_instantaneous_forward_rates(
    maturities: np.ndarray | list,
    zero_rates: np.ndarray | list,
    years: np.ndarray | list | float,
    interpolation: str = "monotone_convex",
) -> np.ndarray:
    """
    Calculate the instantaneous forward rates of a zero-coupon curve: the continuously
    compounded rate, implied by the curve, for borrowing over an infinitesimal period
    starting at t, f(t) = -d ln P(t) / dt = r(t) + t r'(t).

    It is the rate the discrete forward rates of the curve converge to as their period
    shrinks, and shows the shape of the curve more sharply than the zero rates, which
    average it: r(t) = (1 / t) ∫ f(s) ds over 0 to t.

    The curve is read between its maturities as `get_discount_factors` describes. The
    forward of the interpolations on the zero rates follows from the slope of the zero
    rates, so it jumps at the maturities of "linear_zero" and is flat between those of
    "log_linear"; "monotone_convex" gives a continuous forward curve. At a maturity of the
    curve itself, the forward is that of the interval that starts there; beyond the last
    maturity it is the last zero rate.

    For more information, see: Hagan, P.S. and West, G. (2006). "Interpolation Methods for
    Curve Construction." Applied Mathematical Finance, 13(2), 89-129.

    Referenced against:
    - QuantLib, ZeroCurve (linear_zero): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
    - QuantLib, DiscountCurve (log_linear): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_discount_curve.html
    - QuantLib, NaturalCubicZeroCurve (natural_cubic): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
    - QuantLib, MonotonicCubicZeroCurve (monotone_cubic): https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_zero_curve.html
    - QuantLib, PiecewiseYieldCurve with ConvexMonotone (monotone_convex): https://www.quantlib.org/reference/class_quant_lib_1_1_convex_monotone.html

    Args:
        maturities (np.ndarray | list): The maturities of the curve in years, above 0.
        zero_rates (np.ndarray | list): The continuously compounded zero rates (decimals),
            one for each maturity.
        years (np.ndarray | list | float): The times of the forward rates, in years.
        interpolation (str, optional): How the curve is read between its maturities, see
            `get_discount_factors`. Options are: "linear_zero" (linear zero rates),
            "log_linear" (linear log discount factors), "natural_cubic" (a natural cubic
            spline), "monotone_cubic" (a monotone cubic spline), "monotone_convex" (Hagan
            and West's monotone convex method), "linear_par" (linear annually compounded
            zero rates). Defaults to "monotone_convex".

    Returns:
        np.ndarray: The instantaneous forward rate (continuously compounded) at each time.

    Raises:
        ValueError: If the interpolation does not exist, a time is negative, or there is not
            one zero rate for each maturity.
    """
    _, forwards = _read(maturities, zero_rates, years, interpolation)

    return forwards


def bootstrap_zero_curve(
    maturities: np.ndarray | list,
    par_yields: np.ndarray | list,
    frequency: int = 2,
    bill_convention: str | None = None,
    interpolation: str = "linear_par",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap a zero-coupon curve from par yields with a choice of interpolation: the zero
    rates at which every quoted par bond prices at par on the interpolated curve (Hagan and
    West, 2006).

    Maturities up to a year are bills, whose zero rate follows from their yield alone (see
    `curve_model.bill_yields_to_annual`). A longer maturity T is a bond paying its par
    yield c in `frequency` coupons a year, whose discount factor solves

        P(T) = (1 + c/f a - c/f Σ_{t_i < T} P(t_i)) / (1 + c/f)

    with a the elapsed share of the current coupon period (0 on a coupon date) and the
    discount factors of the earlier coupon dates read from the curve with the chosen
    interpolation (see `get_discount_factors`). Those coupon dates lie between maturities
    whose zero rates are not known yet, so the bootstrap starts from the curve of
    "linear_par" and repeats until no zero rate moves by more than 1e-13: for the
    interpolations whose shape between two maturities depends on the maturities further
    out (the cubic splines and monotone convex) this is the global bootstrap of Hagan and
    West.

    With linear par interpolation the par yields themselves are interpolated linearly
    to every coupon date first, as `curve_model.bootstrap_zero_rates` does, and the curve
    holds the zero rate of every coupon date.

    Referenced against:
    - QuantLib, PiecewiseYieldCurve: https://www.quantlib.org/reference/class_quant_lib_1_1_piecewise_yield_curve.html
    - QuantLib, FixedRateBondHelper: https://www.quantlib.org/reference/class_quant_lib_1_1_fixed_rate_bond_helper.html
    - QuantLib, DepositRateHelper: https://www.quantlib.org/reference/class_quant_lib_1_1_deposit_rate_helper.html

    Args:
        maturities (np.ndarray | list): The maturities in years; at least one of a year or
            less.
        par_yields (np.ndarray | list): The par yields (decimals), one for each maturity,
            bond-equivalent with `frequency` coupons a year beyond a year and in the bill
            convention up to it.
        frequency (int, optional): The coupons per year, at least 1. Defaults to 2.
        bill_convention (str | None, optional): How the bill yields are quoted, see
            `curve_model.bill_yields_to_annual`. Options are: "bond-equivalent" (the US
            Treasury), "act/365" (Canada), "act/360" (Sweden), None (yields compounded
            `frequency` times a year). Defaults to None.
        interpolation (str, optional): How the curve is read between its maturities, see
            `get_discount_factors`. Options are: "linear_zero" (linear zero rates),
            "log_linear" (linear log discount factors), "natural_cubic" (a natural cubic
            spline), "monotone_cubic" (a monotone cubic spline), "monotone_convex" (Hagan
            and West's monotone convex method), "linear_par" (linear annually compounded
            zero rates). Defaults to "linear_par".

    Returns:
        tuple[np.ndarray, np.ndarray]: The maturities of the curve (ascending) and their
        continuously compounded zero rates.

    Raises:
        ValueError: If the interpolation does not exist, there is not one par yield for each
            maturity, the frequency is below 1, no maturity is a year or less, or the bootstrap
            does not converge.
    """
    check_option("interpolation", interpolation, INTERPOLATIONS)
    maturities = np.asarray(maturities, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    if maturities.size == 0 or maturities.shape != par_yields.shape:
        raise ValueError(
            f"The curve needs a par yield for each maturity, got {maturities.size} "
            f"maturities and {par_yields.size} par yields."
        )
    if not frequency >= 1:
        raise ValueError(
            f"The frequency must be at least 1 coupon a year, got {frequency}."
        )
    order = np.argsort(maturities)
    maturities, par_yields = maturities[order], par_yields[order]

    if maturities.min() > 1 + EPSILON:
        raise ValueError(
            "The curve needs a maturity of a year or less to bootstrap zero-coupon rates from."
        )

    grid, annual = curve_model.bootstrap_zero_rates(
        maturities, par_yields, frequency, bill_convention, return_grid=True
    )
    if interpolation == "linear_par":
        return grid, np.log1p(annual)

    # The start: the zero rates of the quoted maturities on the linear_par curve, exact for
    # the bills.
    zero_rates = np.log1p(np.interp(maturities, grid, annual))
    bonds = np.flatnonzero(maturities > 1 + EPSILON)
    schedules = []
    for bond in bonds:
        periods = bond_model.get_cash_flow_periods(maturities[bond], frequency)
        accrued = bond_model.get_accrued_fraction(maturities[bond], frequency)
        schedules.append((periods[:-1] / frequency, accrued))

    for _ in range(BOOTSTRAP_PASSES):
        previous = zero_rates.copy()
        for bond, (coupon_dates, accrued) in zip(bonds, schedules, strict=True):
            coupon = par_yields[bond] / frequency
            earlier = get_discount_factors(
                maturities, zero_rates, coupon_dates, interpolation
            ).sum()
            discount = (1 + coupon * accrued - coupon * earlier) / (1 + coupon)
            zero_rates[bond] = -np.log(discount) / maturities[bond]
        if np.max(np.abs(zero_rates - previous)) < BOOTSTRAP_TOLERANCE:
            return maturities, zero_rates

    raise ValueError(
        f"The {interpolation} bootstrap did not converge in {BOOTSTRAP_PASSES} passes."
    )
