"""Bond Model Module"""

import numpy as np
import pandas as pd


def get_bond_price(
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    yield_to_maturity: float,
    frequency: int = 1,
):
    """
    Calculate the price of a bond as the present value of its remaining cash flows.

    Every cash flow is discounted at the yield to maturity, which is treated as a
    nominal annual rate compounded `frequency` times per year — i.e. the per-period
    discount rate is `yield_to_maturity / frequency` and the number of periods is
    `years_to_maturity * frequency`. This is the standard bond-market convention
    (semi-annual compounding for U.S. Treasuries and corporates, annual for most
    European government bonds) rather than continuous compounding.

    Settlement is assumed to fall exactly on a coupon date, so the price returned is
    a clean price with zero accrued interest and there is no fractional first period.

    Also known as: present value of a bond, full price on a coupon date.

    The formula is as follows:

        Price = SUM_(t=1)^(n) [ (Coupon Rate * Par Value / m) / (1 + y / m)^t ]
                + Par Value / (1 + y / m)^n

    where m = frequency, y = yield to maturity and n = years_to_maturity * m.

    A yield of zero, a coupon rate of zero (a zero-coupon bond) and a negative yield
    are all handled correctly; only a yield of −m * 100% or below is undefined.

    For more information, see: https://en.wikipedia.org/wiki/Bond_valuation

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year. Defaults to 1.

    Returns:
        float: The price of the bond.
    """
    coupon_payment = (par_value * coupon_rate) / frequency
    total_periods = int(years_to_maturity * frequency)
    present_value: int | float = 0

    # Calculate the present value of coupon payments
    for t in range(1, total_periods + 1):
        present_value += coupon_payment / ((1 + yield_to_maturity / frequency) ** t)

    # Add the present value of the face value (at maturity)
    present_value += par_value / ((1 + yield_to_maturity / frequency) ** total_periods)

    return present_value


def get_current_yield(par_value: float, coupon_rate: float, bond_price: float):
    """
    Calculate the current yield of a bond.

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        bond_price (float): The current market price of the bond.

    Returns:
        float: The current yield of the bond.
    """
    current_yield = (coupon_rate * par_value) / bond_price

    return current_yield


def get_effective_yield(coupon_rate: float, frequency: int) -> float:
    """
    Calculate the effective yield of a bond, taking into account reinvestment of coupon payments.

    The effective yield restates a nominal (stated) annual coupon rate on a compounded
    basis, by assuming every coupon received during the year is reinvested at the same
    rate until year-end. Whenever coupons are paid more than once a year the effective
    yield therefore exceeds the nominal coupon rate; when they are paid annually the two
    are identical.

    Note the convention this function deliberately follows: the input is the bond's
    **coupon rate**, not its yield to maturity, so the result is the effective annual
    rate earned on the coupon stream of a bond bought at par. It is not the effective
    annual yield to maturity of a bond trading away from par — for that, pass the yield
    to maturity in place of the coupon rate, since the algebra is the same conversion of
    a nominal annual rate compounded `frequency` times per year into an effective annual
    rate.

    Also known as: effective annual yield, effective annual rate, annual equivalent rate.

    The formula is as follows:

        Effective Yield = (1 + Coupon Rate / m)^m - 1

    where m = frequency.

    For more information, see: https://en.wikipedia.org/wiki/Effective_interest_rate

    Args:
        coupon_rate (float): The annual coupon rate (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The effective yield of the bond.
    """
    return (1 + (coupon_rate / frequency)) ** frequency - 1


def get_yield_to_maturity(
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    bond_price: float,
    frequency: int,
    guess: float = 0.05,
    tolerance: float = 0.0001,
    max_iterations: int = 100,
):
    """
    Calculate the yield to maturity of a bond using the secant method.

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (int): The number of years until the bond matures.
        bond_price (float): The current market price of the bond.
        frequency (int): The number of coupon payments per year.
        guess (float): Initial guess for the yield to maturity.
        tolerance (float): The desired level of accuracy.
        max_iterations (int): Maximum number of iterations to perform.

    Returns:
        float: The yield to maturity of the bond.
    """

    # Define the function to solve
    def bond_value(ytm):
        value = 0
        total_periods = int(years_to_maturity * frequency)
        for t in range(1, total_periods + 1):
            value += coupon_rate * par_value / frequency / ((1 + ytm / frequency) ** t)
        value += par_value / ((1 + ytm / frequency) ** total_periods)
        return value - bond_price

    # Initial values
    ytm0 = guess
    ytm1 = guess * 1.1  # Slightly higher guess for the secant method

    # Iterative process using the secant method
    for _ in range(max_iterations):
        ytm_next = ytm1 - bond_value(ytm1) * (ytm1 - ytm0) / (
            bond_value(ytm1) - bond_value(ytm0)
        )
        if abs(ytm_next - ytm1) < tolerance:
            return ytm_next
        ytm0 = ytm1
        ytm1 = ytm_next

    # If the method fails to converge
    return np.nan


def get_macaulays_duration(
    par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
):
    """
    Calculate Macaulay's duration of a bond.

    Macaulay's duration is a measure of the weighted average time until the bond's cash flows are received.
    It takes into account the timing and amount of each cash flow, as well as the yield to maturity.

    Because each cash flow is discounted with the same per-period rate that
    `get_bond_price` uses, the weights are the present values of the cash flows as a
    share of the bond's price, and the time attached to each weight is expressed in
    **years** (t / frequency), so the result is in years regardless of the coupon
    frequency.

    Also known as: Macaulay duration, weighted average time to cash flow.

    The formula is as follows:

        Macaulay's Duration = SUM_(t=1)^(n) [ (t / m) * PV(CF_t) ] / Price

    where m = frequency, n = years_to_maturity * m and PV(CF_t) is the present value of
    the cash flow received in period t, discounted at (1 + y / m)^t.

    For more information, see: https://en.wikipedia.org/wiki/Bond_duration#Macaulay_duration

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The Macaulay's duration of the bond.
    """
    total_periods = int(years_to_maturity * frequency)
    present_value_sum = 0
    cash_flow_weighted_sum = 0

    # Discounted by period count as get_bond_price does, then weighted by t/frequency.
    for t in range(1, total_periods + 1):
        coupon_payment = (par_value * coupon_rate) / frequency
        present_value = coupon_payment / ((1 + yield_to_maturity / frequency) ** t)
        present_value_sum += present_value
        cash_flow_weighted_sum += (t / frequency) * present_value

    # Add the present value of the face value (at maturity)
    present_value_sum += par_value / (
        (1 + yield_to_maturity / frequency) ** total_periods
    )
    cash_flow_weighted_sum += years_to_maturity * (
        par_value / ((1 + yield_to_maturity / frequency) ** total_periods)
    )

    # Calculate Macaulay's duration
    macaulays_duration = cash_flow_weighted_sum / present_value_sum

    return macaulays_duration


def get_modified_duration(
    par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
):
    """
    Calculate the modified duration of a bond.

    The modified duration of a bond measures the sensitivity of its price to changes in yield to maturity.
    It is a useful metric for assessing the interest rate risk associated with a bond investment.

    It is Macaulay's duration divided by one plus the **per-period** yield, i.e. the yield
    to maturity divided by the coupon frequency. Dividing by (1 + y) instead of
    (1 + y / m) is a common error that overstates the sensitivity of a semi-annual bond;
    the per-period division is what makes modified duration equal to the analytical
    -(dP/dy) / P of the pricing formula.

    Also known as: modified duration, adjusted duration, volatility of a bond.

    The formula is as follows:

        Modified Duration = Macaulay's Duration / (1 + y / m)

    where m = frequency and y = yield to maturity. It estimates the percentage price
    change for a one unit (100 percentage point) change in yield, so a modified duration
    of 4.18 implies a 4.18% price fall for a 100 basis point rise in yield.

    For more information, see: https://en.wikipedia.org/wiki/Bond_duration#Modified_duration

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The modified duration of the bond.
    """
    macaulays_duration = get_macaulays_duration(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )

    # Calculate modified duration
    modified_duration = macaulays_duration / (1 + yield_to_maturity / frequency)

    return modified_duration


def get_effective_duration(
    par_value,
    coupon_rate,
    years_to_maturity,
    yield_to_maturity,
    frequency,
    yield_change=0.01,
):
    """
    Calculate the effective duration of a bond.

    The effective duration of a bond measures the sensitivity of the bond's price to changes in the yield to maturity.
    It provides an estimate of the percentage change in the bond's price for a given change in yield.

    Unlike Macaulay's and modified duration, which are analytical yield-duration statistics
    derived from a fixed schedule of cash flows, effective duration is a curve-duration
    statistic obtained by actually repricing the bond after shifting the benchmark yield
    up and down by the same amount. The bond is repriced in **both** directions and the
    two prices are differenced symmetrically around the starting price: a one-sided shift
    would inherit the curvature (convexity) of the price-yield relationship and
    systematically understate the true sensitivity — by roughly 3% for a 5-year bond and
    by more than 10% for a 30-year bond at a 100 basis point shift.

    Also known as: curve duration, option-adjusted duration, OAD.

    The formula is as follows:

        Effective Duration = (V- - V+) / (2 * V0 * Δy)

    where V- is the price after the yield is lowered by Δy, V+ is the price after the
    yield is raised by Δy, V0 is the starting price and Δy = yield_change.

    For more information, see: https://en.wikipedia.org/wiki/Bond_duration#Effective_duration

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The initial yield to maturity of the bond (in decimal).
        yield_change (float): The size of the shift applied to the yield, up and down
            (in decimal). Defaults to 0.01 (100 basis points).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The effective duration of the bond.
    """
    # Calculate bond price at initial yield
    initial_price = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )

    # Reprice symmetrically around the starting yield, so that the curvature of the price-yield relationship cancels out instead of biasing the estimate downwards.  # noqa: E501
    price_increased = get_bond_price(
        par_value,
        coupon_rate,
        years_to_maturity,
        yield_to_maturity + yield_change,
        frequency,
    )
    price_decreased = get_bond_price(
        par_value,
        coupon_rate,
        years_to_maturity,
        yield_to_maturity - yield_change,
        frequency,
    )

    # Calculate effective duration
    effective_duration = (price_decreased - price_increased) / (
        2 * initial_price * yield_change
    )

    return effective_duration


def get_dollar_duration(
    par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
):
    """
    Calculate the bond's dollar duration.

    The dollar duration is calculated by multiplying the bond's modified duration by the bond's price
    and dividing the result by 100, so that it expresses the price change for a 1 percentage point
    change in the yield to maturity.

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The dollar duration of the bond.
    """
    # Calculate modified duration
    modified_duration = get_modified_duration(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )

    # Calculate bond price
    bond_price = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )

    # Calculate dollar duration
    dollar_duration = modified_duration * bond_price / 100

    return dollar_duration


def get_dv01(par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency):
    """
    Calculate DV01 (Dollar Value of 01) of a bond.

    DV01 is the change in a bond's price, expressed as a currency amount per `par_value`
    of face, caused by a one basis point (0.01%) change in its yield. Where modified
    duration is a percentage sensitivity, DV01 is an absolute one, which is what makes it
    the natural unit for sizing a hedge: a position is hedged when the DV01 of the hedge
    instrument, multiplied by its notional, offsets the DV01 of the position.

    It is computed by repricing the bond one basis point above and one basis point below
    the current yield and halving the difference, so the estimate is centred on the
    current yield rather than biased by the curvature of the price-yield relationship.

    Also known as: PV01, BPV, price value of a basis point, basis point value.

    The formula is as follows:

        DV01 = (Price(y - 0.0001) - Price(y + 0.0001)) / 2

    Numerically this equals Modified Duration * Price / 10,000.

    For more information, see: https://en.wikipedia.org/wiki/Bond_duration#DV01

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The DV01 of the bond, in the same currency units as par_value.
    """
    # Average absolute price change over a symmetric 1bp shift, via get_bond_price.
    yield_decreased = yield_to_maturity - 0.0001  # 1 basis point decrease
    yield_increased = yield_to_maturity + 0.0001  # 1 basis point increase

    price_decreased = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_decreased, frequency
    )
    price_increased = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_increased, frequency
    )

    # Calculate the change in bond price for a 1 basis point change in yield
    dv01 = (price_decreased - price_increased) / 2

    return dv01


def get_convexity(
    par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
):
    """
    Calculate the convexity of a bond.

    Convexity measures the curvature of the price-yield relationship, i.e. the rate at
    which a bond's duration itself changes as the yield changes. Duration alone treats
    that relationship as a straight line, which understates the price gain from a yield
    fall and overstates the price loss from a yield rise; convexity is the second-order
    term that corrects for this.

    It is computed here as the annualised effective convexity, obtained by repricing the
    bond one basis point above and one basis point below the current yield rather than
    from a closed-form cash flow sum. Being expressed per unit (not per period) of annual
    yield, it plugs directly into the second-order price approximation without any
    frequency rescaling, and it already carries the 1 / (1 + y / m)^2 curvature of the
    discount function rather than requiring it to be applied separately.

    Also known as: effective convexity, second-order price sensitivity.

    The formula is as follows:

        Convexity = (Price(y + Δy) + Price(y - Δy) - 2 * Price(y)) / (Price(y) * Δy^2)

    with Δy = 0.0001. It is consumed as the second term of
    %ΔPrice ≈ -Modified Duration * Δy + 0.5 * Convexity * Δy^2 — see
    `get_taylor_price_change`.

    For more information, see: https://en.wikipedia.org/wiki/Bond_convexity

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The convexity of the bond.
    """
    # Calculate bond price at current yield
    bond_price = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )

    # Calculate bond price when yield decreases by 1 basis point
    yield_decreased = yield_to_maturity - 0.0001  # 1 basis point decrease
    bond_price_down = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_decreased, frequency
    )

    # Calculate bond price when yield increases by 1 basis point
    yield_increased = yield_to_maturity + 0.0001  # 1 basis point increase
    bond_price_up = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_increased, frequency
    )

    # Calculate convexity
    convexity = (bond_price_up + bond_price_down - 2 * bond_price) / (
        bond_price * 0.0001**2
    )

    return convexity


def _get_bond_price_from_curve(
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    spot_rates: pd.Series,
    frequency: int,
) -> float:
    """
    Calculate the price of a bond by discounting each of its cash flows with the
    zero/spot rate for that specific cash flow's maturity, rather than a single
    flat yield to maturity as `get_bond_price` does.

    The rate applied to a cash flow that falls in between two maturities present in
    `spot_rates` is obtained through linear interpolation. This is the building block
    used by `get_z_spread` (which shifts the curve uniformly) and
    `get_key_rate_duration` (which bumps a single point on the curve).

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        spot_rates (pd.Series): The zero-coupon (spot) yield curve, indexed by
            maturity in years (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The price of the bond as implied by the spot curve.
    """
    spot_rates_sorted = spot_rates.sort_index()
    total_periods = int(round(years_to_maturity * frequency))
    coupon_payment = (par_value * coupon_rate) / frequency

    bond_price = 0.0
    for t in range(1, total_periods + 1):
        period_time = t / frequency
        period_rate = np.interp(
            period_time, spot_rates_sorted.index, spot_rates_sorted.to_numpy()
        )
        bond_price += coupon_payment / (1 + period_rate / frequency) ** t

    maturity_rate = np.interp(
        years_to_maturity, spot_rates_sorted.index, spot_rates_sorted.to_numpy()
    )
    bond_price += par_value / (1 + maturity_rate / frequency) ** total_periods

    return bond_price


def get_z_spread(
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    bond_price: float,
    spot_rates: pd.Series,
    frequency: int = 1,
    guess: float = 0.01,
    tolerance: float = 0.0001,
    max_iterations: int = 100,
) -> float:
    """
    Calculate the zero-volatility spread (Z-spread) of a bond using the secant method.

    The Z-spread is the constant spread that, when added uniformly to every point of a
    benchmark zero-coupon (spot) yield curve, makes the present value of the bond's
    discounted cash flows equal to its observed market price. Unlike a simple yield
    spread (the bond's yield to maturity minus a benchmark yield of the same maturity),
    the Z-spread is measured against the entire curve, which makes it a more accurate
    measure of a bond's compensation for credit and liquidity risk because it does not
    assume a flat curve between now and maturity.

    Also known as: zero-volatility spread, static spread.

    The formula to solve for the Z-spread (Z) is as follows:

        Bond Price = SUM_(t=1)^(n) [ Coupon / frequency / (1 + (spot_rate(t) + Z) / frequency)^t ]
                     + Par Value / (1 + (spot_rate(n) + Z) / frequency)^n

    Because this equation cannot be solved for Z in closed form, it is solved
    iteratively using the secant method — the same numerical approach used by
    `get_yield_to_maturity`.

    For more information, see: https://en.wikipedia.org/wiki/Z-spread

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        bond_price (float): The current market price of the bond.
        spot_rates (pd.Series): The benchmark zero-coupon (spot) yield curve, indexed
            by maturity in years (in decimal).
        frequency (int): The number of coupon payments per year. Defaults to 1.
        guess (float): Initial guess for the Z-spread. Defaults to 0.01.
        tolerance (float): The desired level of accuracy. Defaults to 0.0001.
        max_iterations (int): Maximum number of iterations to perform. Defaults to 100.

    Returns:
        float: The Z-spread of the bond (in decimal), or numpy.nan if the method fails to converge.

    Raises:
        TypeError: If spot_rates is not a pandas Series.
    """
    if not isinstance(spot_rates, pd.Series):
        raise TypeError(
            f"Expected spot_rates to be a pandas Series, received {type(spot_rates)} instead."
        )

    # Define the function to solve
    def price_difference(spread):
        shifted_curve = spot_rates + spread
        return (
            _get_bond_price_from_curve(
                par_value, coupon_rate, years_to_maturity, shifted_curve, frequency
            )
            - bond_price
        )

    # Initial values
    spread0 = guess
    spread1 = guess * 1.1  # Slightly higher guess for the secant method

    # Iterative process using the secant method
    for _ in range(max_iterations):
        spread_next = spread1 - price_difference(spread1) * (spread1 - spread0) / (
            price_difference(spread1) - price_difference(spread0)
        )
        if abs(spread_next - spread1) < tolerance:
            return spread_next
        spread0 = spread1
        spread1 = spread_next

    # If the method fails to converge
    return np.nan


def get_bond_equivalent_yield(discount_yield: float, days_to_maturity: float) -> float:
    """
    Convert a money-market discount yield (e.g. quoted for Treasury bills) into a
    bond-equivalent yield (BEY).

    Money-market instruments such as Treasury bills are often quoted on a discount-yield
    basis, which understates the actual return an investor earns because it is computed
    on the face value rather than the (lower) purchase price, and uses a 360-day year
    rather than a 365-day year. The bond-equivalent yield restates the discount yield on
    a basis that is comparable to coupon-bearing bonds, making it possible to compare a
    T-bill's return directly against notes and bonds.

    Also known as: BEY, coupon-equivalent yield, investment rate.

    Two formulas apply, because a bill maturing in more than half a year would pay a
    coupon along the way if it were a note, so its equivalent yield has to be compounded
    once. For a bill with half a year (365 / 2 days) or less remaining the simple
    restatement suffices:

        BEY = 365 * Discount Yield / (360 - Days to Maturity * Discount Yield)

    For a bill with more than 182 days remaining, the U.S. Treasury (31 CFR 356,
    Appendix B) solves the semi-annually compounded relationship
    1 / Price = (1 + BEY / 2) * (1 + (t - 365 / 2) * BEY / 365) for BEY, which gives:

        BEY = (-2t/365 + 2 * SQRT((t/365)^2 - (2t/365 - 1) * (1 - 1/Price)))
              / (2t/365 - 1)

    where t = days_to_maturity and Price = 1 - Discount Yield * t / 360 is the purchase
    price per unit of face value. Applying the simple formula beyond 182 days ignores
    that intermediate compounding and understates the yield.

    A 365-day year is assumed throughout; the Treasury substitutes 366 when the holding
    period spans a 29 February, which shifts the result by well under a basis point.

    For more information, see: https://en.wikipedia.org/wiki/Bond_equivalent_yield

    Args:
        discount_yield (float): The money-market discount yield of the instrument (in decimal).
        days_to_maturity (float): The number of days until the instrument matures.

    Returns:
        float: The bond-equivalent yield (in decimal).

    Raises:
        TypeError: If discount_yield or days_to_maturity are not a float or int.
        ValueError: If days_to_maturity is not positive.
    """
    if not isinstance(discount_yield, int | float):
        raise TypeError(
            f"Expected discount_yield to be a float or int, received {type(discount_yield)} instead."
        )
    if not isinstance(days_to_maturity, int | float):
        raise TypeError(
            f"Expected days_to_maturity to be a float or int, received {type(days_to_maturity)} instead."
        )
    if days_to_maturity <= 0:
        raise ValueError("days_to_maturity must be greater than 0.")

    # A bill of half a year or less pays nothing before maturity, so the discount yield only has to be restated onto the purchase price and a 365-day year. At exactly 365 / 2 days the compounded formula below degenerates to this same expression while its denominator goes to zero, so that point is routed here as well.  # noqa: E501
    if days_to_maturity <= 365 / 2:
        return (365 * discount_yield) / (360 - days_to_maturity * discount_yield)

    # Beyond that, an equivalent note would have paid a coupon at the six month point, so the Treasury's semi-annually compounded solution is used instead.  # noqa: E501
    purchase_price = 1 - discount_yield * days_to_maturity / 360
    year_fraction = days_to_maturity / 365

    discriminant = year_fraction**2 - (2 * year_fraction - 1) * (1 - 1 / purchase_price)

    bond_equivalent_yield = (-2 * year_fraction + 2 * np.sqrt(discriminant)) / (
        2 * year_fraction - 1
    )

    return bond_equivalent_yield


def get_key_rate_duration(
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    spot_rates: pd.Series,
    key_rate_maturity: float,
    frequency: int = 1,
    yield_change: float = 0.0001,
) -> float:
    """
    Calculate the key rate duration of a bond for a single maturity point on the yield curve.

    Whereas `get_effective_duration` assumes the entire yield curve shifts in parallel by
    the same amount, key rate duration measures the bond's price sensitivity to a shock
    at a single tenor of the curve (e.g. only the 5-year point), while every other point
    on the curve is left unchanged. Because the bond's cash flows are discounted using
    linear interpolation between the curve's tenors, a shock at one tenor tapers off
    linearly towards the neighboring tenors and has no effect beyond them. Summing the
    key rate durations across every tenor on the curve approximately reproduces the
    bond's effective (parallel-shift) duration, but key rate duration additionally
    reveals which segment of the curve the bond's price is most exposed to — information
    that is essential for constructing curve-neutral hedges or identifying "twist" risk.

    Also known as: partial duration, rate-specific duration.

    The formula is as follows:

        Key Rate Duration = -(Price(rate_k + Δy) - Price(rate_k - Δy)) / (2 * Price(rate) * Δy)

    where only the rate at the key_rate_maturity tenor (rate_k) is bumped up or down by
    Δy = yield_change, all other tenors of the curve are held fixed, and Price(...) is
    calculated by discounting each cash flow with the (possibly interpolated) curve rate
    for that cash flow's maturity — see `_get_bond_price_from_curve`.

    For more information, see: https://en.wikipedia.org/wiki/Key_rate_duration

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        spot_rates (pd.Series): The zero-coupon (spot) yield curve used to discount the
            bond's cash flows, indexed by maturity in years (in decimal).
        key_rate_maturity (float): The maturity, in years, of the curve point to shock.
            Must be one of the maturities present in the index of `spot_rates`.
        frequency (int): The number of coupon payments per year. Defaults to 1.
        yield_change (float): The size of the shock applied to the key rate, up and down
            (in decimal). Defaults to 0.0001 (1 basis point).

    Returns:
        float: The key rate duration of the bond for the given maturity point.

    Raises:
        TypeError: If spot_rates is not a pandas Series.
        ValueError: If key_rate_maturity is not one of the maturities in spot_rates.
    """
    if not isinstance(spot_rates, pd.Series):
        raise TypeError(
            f"Expected spot_rates to be a pandas Series, received {type(spot_rates)} instead."
        )
    if key_rate_maturity not in spot_rates.index:
        raise ValueError(
            f"key_rate_maturity {key_rate_maturity} was not found in the index of "
            "spot_rates. Please provide one of the maturities present in the curve: "
            f"{list(spot_rates.index)}"
        )

    base_price = _get_bond_price_from_curve(
        par_value, coupon_rate, years_to_maturity, spot_rates, frequency
    )

    spot_rates_up = spot_rates.copy()
    spot_rates_up.loc[key_rate_maturity] += yield_change
    price_up = _get_bond_price_from_curve(
        par_value, coupon_rate, years_to_maturity, spot_rates_up, frequency
    )

    spot_rates_down = spot_rates.copy()
    spot_rates_down.loc[key_rate_maturity] -= yield_change
    price_down = _get_bond_price_from_curve(
        par_value, coupon_rate, years_to_maturity, spot_rates_down, frequency
    )

    key_rate_duration = -(price_up - price_down) / (2 * base_price * yield_change)

    return key_rate_duration


def get_taylor_price_change(
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    yield_to_maturity: float,
    frequency: int,
    yield_change: float,
) -> float:
    """
    Estimate the percentage change in a bond's price for a given change in yield using
    a second-order Taylor series expansion that combines modified duration and convexity.

    Modified duration alone only captures the first-order (linear) relationship between
    a bond's price and its yield, which understates the price increase for a yield
    decrease and overstates the price decrease for a yield increase because the true
    price-yield relationship is curved (convex), not linear. Adding a convexity term
    corrects for this curvature and produces a substantially more accurate estimate,
    especially for larger yield changes.

    Also known as: duration-convexity approximation, second-order price approximation.

    The formula is as follows:

        %ΔPrice ≈ -Modified Duration * Δy + 0.5 * Convexity * Δy^2

    This function calls `get_modified_duration` and `get_convexity` directly rather than
    recomputing them.

    For more information, see: https://en.wikipedia.org/wiki/Bond_convexity

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The current yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year.
        yield_change (float): The hypothetical change in yield to maturity (in decimal),
            e.g. 0.01 for a 100 basis point increase or -0.005 for a 50 basis point decrease.

    Returns:
        float: The estimated percentage change in the bond's price (in decimal).
    """
    modified_duration = get_modified_duration(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )
    convexity = get_convexity(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )

    price_change_pct = (
        -modified_duration * yield_change + 0.5 * convexity * yield_change**2
    )

    return price_change_pct
