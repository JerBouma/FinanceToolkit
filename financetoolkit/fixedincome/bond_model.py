"""Bond Model Module"""

import numpy as np


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
        years_to_maturity (int): The number of years until the bond matures.
        yield_to_maturity (float): The yield to maturity of the bond (in decimal).

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

    # Calculate present value of each cash flow and the sum of present values
    for t in range(1, total_periods + 1):
        coupon_payment = (par_value * coupon_rate) / frequency
        present_value = coupon_payment / (
            (1 + yield_to_maturity / frequency) ** (t / frequency)
        )
        present_value_sum += present_value
        cash_flow_weighted_sum += (t / frequency) * present_value

    # Add the present value of the face value (at maturity)
    present_value_sum += par_value / (
        (1 + yield_to_maturity / frequency) ** years_to_maturity
    )
    cash_flow_weighted_sum += years_to_maturity * (
        par_value / ((1 + yield_to_maturity / frequency) ** years_to_maturity)
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
    # Calculate present value of each cash flow
    total_periods = int(years_to_maturity * frequency)
    present_value_sum = 0

    for t in range(1, total_periods + 1):
        coupon_payment = (par_value * coupon_rate) / frequency
        present_value = coupon_payment / (
            (1 + yield_to_maturity / frequency) ** (t / frequency)
        )
        present_value_sum += present_value

    # Add the present value of the face value (at maturity)
    present_value_sum += par_value / (
        (1 + yield_to_maturity / frequency) ** years_to_maturity
    )

    # Calculate bond price when yield decreases by 1 basis point
    yield_decreased = yield_to_maturity - 0.0001  # 1 basis point decrease
    present_value_sum_decreased = 0

    for t in range(1, total_periods + 1):
        coupon_payment = (par_value * coupon_rate) / frequency
        present_value = coupon_payment / (
            (1 + yield_decreased / frequency) ** (t / frequency)
        )
        present_value_sum_decreased += present_value

    # Add the present value of the face value (at maturity)
    present_value_sum_decreased += par_value / (
        (1 + yield_decreased / frequency) ** years_to_maturity
    )

    # Calculate bond price when yield increases by 1 basis point
    yield_increased = yield_to_maturity + 0.0001  # 1 basis point increase
    present_value_sum_increased = 0

    for t in range(1, total_periods + 1):
        coupon_payment = (par_value * coupon_rate) / frequency
        present_value = coupon_payment / (
            (1 + yield_increased / frequency) ** (t / frequency)
        )
        present_value_sum_increased += present_value

    # Add the present value of the face value (at maturity)
    present_value_sum_increased += par_value / (
        (1 + yield_increased / frequency) ** years_to_maturity
    )

    # Calculate the change in bond price for a 1 basis point change in yield
    dv01 = -0.01 * (present_value_sum_increased - present_value_sum_decreased) / 2

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
