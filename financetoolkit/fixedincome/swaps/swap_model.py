"""Swap Model

Interest rate swaps on a single curve: the par swap rate and annuity, the value and risk of
a swap, and the asset swap and I-spreads of a fixed rate bond. One curve both projects the
floating rate and discounts every cash flow, so a floating leg that resets at the start of
each period is worth par at its start (Hull, 2022, ch. 7).

Every function takes the zero-coupon curve as the Toolkit's curves are returned: annually
compounded zero rates by maturity in years, interpolated linearly in between (see
`financetoolkit.fixedincome.helpers.discount_factors`). Times are year fractions: a period
of a leg paid `frequency` times a year accrues 1 / frequency of a year, with no calendar or
business-day adjustment (a regular schedule on a 30/360 day count).
"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.helpers import discount_factors

# A basis point, the shift of the curve and the fixed rate the risk measures are per.
BASIS_POINT = 0.0001


def _schedule(
    years: float, frequency: int, start: float = 0.0, running: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    The payment times and accrual fractions of a leg that ends `years` after `start`.

    The payments are counted back from the end in periods of 1 / frequency years, so an
    irregular period comes first. A new leg (running=False) accrues that first period from
    its start only (a short stub); a leg that is already running (running=True) pays the
    full period on its first payment date, as it started accruing before today.

    Args:
        years (float): The length of the leg, from its start to its last payment, in years.
        frequency (int): The number of payments per year.
        start (float): The start of the leg, in years from today. Defaults to 0.
        running (bool): Whether the leg started before today. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: The payment times (years from today) and the
        accrual fraction of each payment.
    """
    times = start + bond_model.get_cash_flow_periods(years, frequency) / frequency
    accruals = np.diff(times, prepend=start)
    if running:
        accruals[0] = 1 / frequency

    return times, accruals


def get_swap_rate(
    zero_curve: pd.Series,
    tenor: float,
    fixed_frequency: int = 1,
    forward_start: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate the par swap rate and the annuity of an interest rate swap on a single curve:
    the fixed rate at which the swap is worth nothing today, and the value of one basis
    point of it.

    The par swap rate is the fixed rate at which a new swap is worth nothing: the present
    value of its fixed leg equals that of its floating leg. On a single curve the floating
    leg of a swap starting at t_0 is worth P(t_0) - P(t_n), whatever its reset frequency,
    so the rate follows from the discount factors alone (Hull, 2022, ch. 7; Andersen and
    Piterbarg, 2010, ch. 5).

    The formula is as follows:

    - Annuity = SUM(τ_i * P(t_i)), over the fixed payment dates t_1, ..., t_n
    - Swap Rate = (P(t_0) - P(t_n)) / Annuity

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero
    rate r(t) of the curve, t_0 = forward_start the start of the swap and τ_i the accrual
    fraction of the i-th fixed period: 1 / fixed_frequency, apart from a short first period
    when the tenor is not a whole number of periods. The swap rate is a nominal annual
    rate paid fixed_frequency times a year on that accrual basis.

    The annuity is the present value of receiving 1 a year on the fixed leg, so the value
    of a basis point on the fixed rate (the PV01) is Annuity * Notional * 0.0001.

    Referenced against:
    - QuantLib, VanillaSwap.fairRate: https://www.quantlib.org/reference/class_quant_lib_1_1_vanilla_swap.html
    - QuantLib, DiscountingSwapEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_discounting_swap_engine.html
    - rateslib, IRS.rate: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.IRS.html#rateslib.instruments.IRS.rate

    Shares similar logic:
    - FinancePy, IborSwap.swap_rate: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/rates/ibor_swap.py#L280
    - FinancePy, IborSwap.pv01: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/rates/ibor_swap.py#L269

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        tenor (float): The length of the swap from its start, in years.
        fixed_frequency (int, optional): The number of fixed payments per year.
            Defaults to 1 (annual).
        forward_start (float, optional): The start of the swap, in years from today,
            for a forward starting swap. Defaults to 0, a swap that starts today.

    Returns:
        tuple[float, float]: The par swap rate (decimal) and the annuity (years, per unit
        of notional).

    Raises:
        ValueError: If the tenor is not positive.

    """
    if tenor <= 0:
        raise ValueError(f"The tenor of a swap must be positive, not {tenor}.")

    times, accruals = _schedule(tenor, fixed_frequency, forward_start)
    annuity = float(np.sum(accruals * discount_factors(zero_curve, times)))
    start_discount, end_discount = discount_factors(
        zero_curve, np.array([forward_start, times[-1]])
    )

    return float((start_discount - end_discount) / annuity), annuity


def _swap_legs(
    zero_curve: pd.Series,
    fixed_rate: float,
    years_to_maturity: float,
    fixed_frequency: int,
    float_frequency: int,
    current_fixing: float | None,
) -> tuple[float, float, float]:
    """
    The present values of the legs of a running swap, per unit of notional.

    Args:
        zero_curve (pd.Series): The annually compounded zero rates by maturity in years.
        fixed_rate (float): The fixed rate of the swap.
        years_to_maturity (float): The years until the last payment.
        fixed_frequency (int): The number of fixed payments per year.
        float_frequency (int): The number of floating payments per year.
        current_fixing (float | None): The floating rate fixed for the current period.

    Returns:
        tuple[float, float, float]: The fixed leg, the floating leg and the annuity.
    """
    fixed_times, fixed_accruals = _schedule(
        years_to_maturity, fixed_frequency, running=True
    )
    annuity = float(np.sum(fixed_accruals * discount_factors(zero_curve, fixed_times)))

    end_discount = float(discount_factors(zero_curve, years_to_maturity))
    if current_fixing is None:
        floating_leg = 1 - end_discount
    else:
        float_times, float_accruals = _schedule(
            years_to_maturity, float_frequency, running=True
        )
        next_discount = float(discount_factors(zero_curve, float_times[0]))
        floating_leg = (
            1 + current_fixing * float_accruals[0]
        ) * next_discount - end_discount

    return fixed_rate * annuity, floating_leg, annuity


def get_swap_value(
    zero_curve: pd.Series,
    fixed_rate: float,
    years_to_maturity: float,
    notional: float = 10_000_000,
    fixed_frequency: int = 1,
    float_frequency: int = 4,
    current_fixing: float | None = None,
    is_receiver: bool = False,
) -> dict[str, float]:
    """
    Calculate the value, the par swap rate and the risk of an interest rate swap on a single
    curve: what a new swap or one that is already running is worth, and how its value
    changes with the rates.

    A payer swap pays the fixed rate and receives the floating rate; a receiver swap does the
    opposite. The fixed leg is worth the fixed rate times the annuity. The floating leg is worth
    par at its next reset, so today it is worth the current period's floating payment and par,
    discounted from the next payment date, minus the principal at maturity (Hull, 2022, ch. 7).

    The formula is as follows:

    - Annuity = SUM(τ_i * P(t_i)), over the remaining fixed payment dates
    - Fixed Leg = Notional * Fixed Rate * Annuity
    - Floating Leg = Notional * ((1 + L * δ) * P(s_1) - P(T)), or Notional * (1 - P(T))
      when no current fixing is given
    - Payer Swap Value = Floating Leg - Fixed Leg (the receiver swap is its negative)
    - Par Swap Rate = Floating Leg / (Notional * Annuity)
    - PV01 = Notional * Annuity * 0.0001
    - DV01 = (Value(r + 0.0001) - Value(r - 0.0001)) / 2

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero rate
    r(t), T the years to maturity, L the current fixing paid at the next floating payment
    date s_1 with accrual δ = 1 / float_frequency, and τ_i = 1 / fixed_frequency. The payment
    dates are counted back from maturity in whole periods, so a running swap's first fixed and
    floating payments are full periods. Without a current fixing, the floating leg is valued as
    if it reset today, which is exact on a reset date (a new swap) only.

    The PV01 is the change in value of a 1 basis point higher fixed rate (positive, the
    receiver gains it). The DV01 is the change in value of a parallel shift of the
    annually compounded zero rates by 1 basis point, by central difference: positive for a
    payer swap, which gains when rates rise.

    Referenced against:
    - QuantLib, VanillaSwap: https://www.quantlib.org/reference/class_quant_lib_1_1_vanilla_swap.html
    - rateslib, IRS.npv: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.IRS.html#rateslib.instruments.IRS.npv

    Shares similar logic:
    - FinancePy, IborSwap: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/rates/ibor_swap.py#L29

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        fixed_rate (float): The fixed rate of the swap (decimal).
        years_to_maturity (float): The years until the swap's last payment.
        notional (float, optional): The notional of the swap. Defaults to 10,000,000.
        fixed_frequency (int, optional): The number of fixed payments per year.
            Defaults to 1 (annual).
        float_frequency (int, optional): The number of floating payments per year.
            Defaults to 4 (quarterly).
        current_fixing (float | None, optional): The floating rate already fixed for the
            current floating period (decimal, simple interest on 1 / float_frequency of a
            year). Defaults to None, which values the floating leg as if it reset today.
        is_receiver (bool, optional): Whether the swap receives the fixed rate. Defaults
            to False, a payer swap.

    Returns:
        dict[str, float]: The Value, the Fixed Leg and Floating Leg present values, the Par
        Swap Rate, the Annuity (per unit of notional), the PV01 and the DV01, all in the
        currency of the notional apart from the rate and the annuity.

    Raises:
        ValueError: If the years to maturity are not positive.

    """
    if years_to_maturity <= 0:
        raise ValueError(
            f"The years to maturity of a swap must be positive, not {years_to_maturity}."
        )

    def payer_value(curve: pd.Series) -> float:
        fixed_leg, floating_leg, _ = _swap_legs(
            curve,
            fixed_rate,
            years_to_maturity,
            fixed_frequency,
            float_frequency,
            current_fixing,
        )
        return notional * (floating_leg - fixed_leg)

    fixed_leg, floating_leg, annuity = _swap_legs(
        zero_curve,
        fixed_rate,
        years_to_maturity,
        fixed_frequency,
        float_frequency,
        current_fixing,
    )
    sign = -1 if is_receiver else 1
    dv01 = (
        payer_value(zero_curve + BASIS_POINT) - payer_value(zero_curve - BASIS_POINT)
    ) / 2

    return {
        "Value": sign * notional * (floating_leg - fixed_leg),
        "Fixed Leg": notional * fixed_leg,
        "Floating Leg": notional * floating_leg,
        "Par Swap Rate": floating_leg / annuity,
        "Annuity": annuity,
        "PV01": notional * annuity * BASIS_POINT,
        "DV01": sign * dv01,
    }


def get_asset_swap_spread(
    zero_curve: pd.Series,
    coupon_rate: float,
    years_to_maturity: float,
    clean_price: float,
    frequency: int = 2,
    float_frequency: int = 4,
) -> tuple[float, float]:
    """
    Calculate the par and the market value asset swap spread of a fixed rate bond over a
    swap curve: the spread over the floating rate that swapping the bond's fixed coupons
    pays, a measure of the bond's credit relative to the swap curve.

    An asset swap combines a bond with a swap that pays away its coupons for the floating
    rate plus a spread, turning the bond into a floating rate asset. In a par asset swap the
    investor pays par for the package, whatever the bond's price, and the spread makes the
    package worth par; in a market value asset swap the investor pays the bond's price and
    the floating leg runs on that amount. The spread measures the bond's credit and
    liquidity over the swap curve (O'Kane, 2000; Tuckman and Serrat, 2022, ch. 18).

    The formula is as follows:

    - Curve Price = 100 * (SUM(c / f * P(t_i)) + P(t_n)), the bond's dirty price on the swap curve
    - Annuity = SUM(δ_j * P(s_j)), over the floating payment dates s_j
    - Par Asset Swap Spread = (Curve Price - Dirty Price) / (100 * Annuity)
    - Market Value Asset Swap Spread = (Curve Price - Dirty Price) / (Dirty Price * Annuity)

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded swap zero
    rate r(t), c the coupon rate, f the coupon frequency, t_i the coupon dates counted back
    from maturity, Dirty Price = Clean Price + Accrued Interest (see
    `bond_model.get_accrued_interest`) per 100 of par, and δ_j the accrual fractions of the
    floating leg, paid float_frequency times a year from today to maturity with a short first
    period. The spreads are simple annual rates on the floating leg's accrual basis.

    Referenced against:
    - QuantLib, AssetSwap.fairSpread: https://www.quantlib.org/reference/class_quant_lib_1_1_asset_swap.html

    Args:
        zero_curve (pd.Series): The annually compounded swap zero rates (decimals), indexed
            by maturity in years.
        coupon_rate (float): The bond's annual coupon rate (decimal).
        years_to_maturity (float): The years until the bond matures.
        clean_price (float): The bond's clean price per 100 of par.
        frequency (int, optional): The number of coupons per year. Defaults to 2.
        float_frequency (int, optional): The number of floating payments per year of the
            swap. Defaults to 4 (quarterly).

    Returns:
        tuple[float, float]: The par and the market value asset swap spread (decimals).

    """
    periods = bond_model.get_cash_flow_periods(years_to_maturity, frequency)
    coupon_discounts = discount_factors(zero_curve, periods / frequency)
    curve_price = 100 * (
        coupon_rate / frequency * coupon_discounts.sum() + coupon_discounts[-1]
    )
    dirty_price = clean_price + bond_model.get_accrued_interest(
        100, coupon_rate, years_to_maturity, frequency
    )

    float_times, float_accruals = _schedule(years_to_maturity, float_frequency)
    annuity = float(np.sum(float_accruals * discount_factors(zero_curve, float_times)))

    return (
        float((curve_price - dirty_price) / (100 * annuity)),
        float((curve_price - dirty_price) / (dirty_price * annuity)),
    )


def get_i_spread(
    zero_curve: pd.Series,
    yield_to_maturity: float,
    years_to_maturity: float,
    frequency: int = 2,
) -> float:
    """
    Calculate the I-spread (interpolated spread) of a bond: its yield to maturity over the
    par swap rate of its maturity, the extra yield the bond pays over the swap curve.

    The I-spread compares a bond with the swap curve the way the G-spread compares it with
    government bonds: the yield of the bond minus the swap rate interpolated at the bond's
    maturity (Tuckman and Serrat, 2022, ch. 18). The swap rate is the par rate of a swap
    paying fixed at the bond's coupon frequency, so both rates are compounded alike.

    The formula is as follows:

    - I-Spread = Yield to Maturity - Swap Rate(T)

    where Swap Rate(T) is the par swap rate of `get_swap_rate` for the bond's years to
    maturity T, with fixed payments `frequency` times a year (a short first period when T
    is not a whole number of periods), on the annually compounded swap zero curve.

    Args:
        zero_curve (pd.Series): The annually compounded swap zero rates (decimals), indexed
            by maturity in years.
        yield_to_maturity (float): The bond's yield to maturity (decimal), a nominal annual
            rate compounded `frequency` times a year.
        years_to_maturity (float): The years until the bond matures.
        frequency (int, optional): The number of coupons per year, which sets the fixed
            frequency of the swap. Defaults to 2.

    Returns:
        float: The I-spread (decimal).

    """
    swap_rate, _ = get_swap_rate(zero_curve, years_to_maturity, frequency)

    return yield_to_maturity - swap_rate
