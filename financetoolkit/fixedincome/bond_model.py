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
    Calculate the price of a bond.

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

    # Calculate present value of each cash flow and the sum of present values.
    # Cash flows are discounted using the period count (t) as the exponent since
    # the per-period rate is yield_to_maturity / frequency — matching get_bond_price.
    # Each cash flow is then weighted by the time it is received, in years (t / frequency).
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

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The initial yield to maturity of the bond (in decimal).
        yield_change (float): The change in yield (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The effective duration of the bond.
    """
    # Calculate bond price at initial yield
    initial_price = get_bond_price(
        par_value, coupon_rate, years_to_maturity, yield_to_maturity, frequency
    )

    # Calculate bond price at yield change
    new_yield = yield_to_maturity + yield_change
    new_price = get_bond_price(
        par_value, coupon_rate, years_to_maturity, new_yield, frequency
    )

    # Calculate effective duration
    effective_duration = -((new_price - initial_price) / (initial_price * yield_change))

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

    Args:
        par_value (float): The face value of the bond.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).
        frequency (int): The number of coupon payments per year.

    Returns:
        float: The DV01 of the bond.
    """
    # DV01 is the average absolute price change resulting from a symmetric 1 basis
    # point (0.0001) shift of the yield to maturity up and down. Reuses get_bond_price
    # so the discounting is guaranteed to be consistent with the rest of the module.
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

    Also known as: BEY, coupon-equivalent yield.

    The formula is as follows:

        BEY = 365 * Discount Yield / (360 - Days to Maturity * Discount Yield)

    For more information, see: https://en.wikipedia.org/wiki/Bond_equivalent_yield

    Args:
        discount_yield (float): The money-market discount yield of the instrument (in decimal).
        days_to_maturity (float): The number of days until the instrument matures.

    Returns:
        float: The bond-equivalent yield (in decimal).

    Raises:
        TypeError: If discount_yield or days_to_maturity are not a float or int.
    """
    if not isinstance(discount_yield, int | float):
        raise TypeError(
            f"Expected discount_yield to be a float or int, received {type(discount_yield)} instead."
        )
    if not isinstance(days_to_maturity, int | float):
        raise TypeError(
            f"Expected days_to_maturity to be a float or int, received {type(days_to_maturity)} instead."
        )

    bond_equivalent_yield = (365 * discount_yield) / (
        360 - days_to_maturity * discount_yield
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
