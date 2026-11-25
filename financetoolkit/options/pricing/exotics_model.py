"""Exotic Options Model"""

__docformat__ = "google"

import numpy as np
from scipy.special import ndtr

from financetoolkit.options.helpers import (
    validate_numeric_inputs,
    validate_positive_inputs,
)
from financetoolkit.utilities.error_model import check_option


def _barrier_term(
    phi: int,
    eta: int,
    stock_price: float,
    strike_price: float,
    barrier: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    term: str,
) -> float:
    """
    Helper function that calculates one of the six building-block terms (A, B, C, D, E
    or F) of the Reiner & Rubinstein (1991) closed-form barrier option pricing formulas.

    Args:
        phi (int): 1 for a call option, -1 for a put option.
        eta (int): 1 for a down-barrier, -1 for an up-barrier.
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        barrier (float): The barrier level.
        risk_free_rate (float): The risk-free interest rate.
        cost_of_carry (float): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option.
        term (str): Which of the six terms ("A", "B", "C" or "D") to compute.

    Returns:
        float: The value of the requested term.
    """
    volatility_time = volatility * np.sqrt(time_to_expiration)
    mu = (cost_of_carry - volatility**2 / 2) / volatility**2

    if term == "A":
        x1 = (
            np.log(stock_price / strike_price) / volatility_time
            + (1 + mu) * volatility_time
        )
        return phi * stock_price * np.exp(
            (cost_of_carry - risk_free_rate) * time_to_expiration
        ) * ndtr(phi * x1) - phi * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * ndtr(
            phi * x1 - phi * volatility_time
        )

    if term == "B":
        x2 = (
            np.log(stock_price / barrier) / volatility_time + (1 + mu) * volatility_time
        )
        return phi * stock_price * np.exp(
            (cost_of_carry - risk_free_rate) * time_to_expiration
        ) * ndtr(phi * x2) - phi * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * ndtr(
            phi * x2 - phi * volatility_time
        )

    if term == "C":
        y1 = (
            np.log(barrier**2 / (stock_price * strike_price)) / volatility_time
            + (1 + mu) * volatility_time
        )
        return phi * stock_price * np.exp(
            (cost_of_carry - risk_free_rate) * time_to_expiration
        ) * (barrier / stock_price) ** (2 * (mu + 1)) * ndtr(
            eta * y1
        ) - phi * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * (
            barrier / stock_price
        ) ** (
            2 * mu
        ) * ndtr(
            eta * y1 - eta * volatility_time
        )

    # The remaining term is D.
    y2 = np.log(barrier / stock_price) / volatility_time + (1 + mu) * volatility_time
    return phi * stock_price * np.exp(
        (cost_of_carry - risk_free_rate) * time_to_expiration
    ) * (barrier / stock_price) ** (2 * (mu + 1)) * ndtr(
        eta * y2
    ) - phi * strike_price * np.exp(
        -risk_free_rate * time_to_expiration
    ) * (
        barrier / stock_price
    ) ** (
        2 * mu
    ) * ndtr(
        eta * y2 - eta * volatility_time
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_barrier_option(
    stock_price: float | np.ndarray,
    strike_price: float | np.ndarray,
    barrier: float | np.ndarray,
    risk_free_rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float | np.ndarray = 0.0,
    put_option: bool = False,
    barrier_direction: str = "down",
    knock_type: str = "out",
    rebate: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Calculate the closed-form price of a single-barrier European option: what an option is worth
    that knocks in or out when the stock price touches a barrier below (down) or above (up) it,
    with the formulas of Reiner & Rubinstein (1991).

    Also known as: knock-in option, knock-out option, down-and-out, down-and-in,
    up-and-out, up-and-in option.

    A barrier option is a path-dependent option whose payoff (and existence) depends on
    whether the underlying stock price touches a pre-specified barrier level at any
    point before expiration:

    - Knock-out: the option becomes worthless if the barrier is touched.
    - Knock-in: the option only comes into existence if the barrier is touched;
      otherwise, at expiration it pays the (optional) rebate.
    - Down barrier: the barrier is below the current stock price.
    - Up barrier: the barrier is above the current stock price.

    A useful identity that this function satisfies (and that is used to validate the
    implementation) is in-out parity: for identical parameters and no rebate, a knock-in
    option plus its corresponding knock-out option (same direction) equals the price of the
    equivalent vanilla European option, since the underlying either does or does not
    touch the barrier:

    - down-and-in + down-and-out = vanilla option
    - up-and-in + up-and-out = vanilla option

    This holds only when the rebate is zero. A rebate is an extra cash payment attached
    to whichever leg does not survive, so it is added to both legs rather than split
    between them and the sum then exceeds the vanilla price.

    The numeric inputs may be arrays, which price a grid of strikes and expirations in one
    call with the same values as one call per combination.

    Shares similar logic:
    - QuantLib, AnalyticBarrierEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_barrier_engine.html

    Args:
        stock_price (float | np.ndarray): The current stock price.
        strike_price (float | np.ndarray): The option's strike price.
        barrier (float | np.ndarray): The barrier level. Should be below stock_price when
            barrier_direction is "down" and above stock_price when barrier_direction is
            "up".
        risk_free_rate (float | np.ndarray): The risk-free interest rate.
        volatility (float | np.ndarray): The volatility of the stock.
        time_to_expiration (float | np.ndarray): The time to expiration of the option.
        dividend_yield (float | np.ndarray): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.
        barrier_direction (str): The side of the barrier. Options are: "down" (below the
            stock price), "up" (above the stock price). Defaults to "down".
        knock_type (str): The knock. Options are: "in" (knock-in), "out" (knock-out).
            Defaults to "out".
        rebate (float | np.ndarray): The fixed cash amount paid out (a) immediately for a knock-out
            option if/when the barrier is breached, or (b) at expiration for a knock-in
            option that never breaches the barrier. Defaults to 0.

    Returns:
        float | np.ndarray: The Barrier Option value, an array in the shape of the inputs
        when any of them is an array.

    Raises:
        TypeError: if any of the numeric inputs is not a float, int or np.ndarray.
        ValueError: if barrier_direction is not "down"/"up", knock_type is not
        "in"/"out" or volatility is not positive.

    Notes:
        Reference: Reiner, E., & Rubinstein, M. (1991). "Breaking Down the Barriers."
        Risk Magazine, 4(8), 28-35.
    """
    validate_numeric_inputs(
        arrays=True,
        stock_price=stock_price,
        strike_price=strike_price,
        barrier=barrier,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        rebate=rebate,
    )
    # The barrier terms divide by the prices and take their logarithms.
    validate_positive_inputs(
        stock_price=stock_price, strike_price=strike_price, barrier=barrier
    )

    check_option("barrier_direction", barrier_direction, ("down", "up"))

    check_option("knock_type", knock_type, ("in", "out"))

    # The barrier terms divide by the variance, which a constant price does not have.
    if np.any(np.asarray(volatility) <= 0):
        raise ValueError(f"volatility must be positive, received {volatility!r}.")

    cost_of_carry = risk_free_rate - dividend_yield
    volatility_time = volatility * np.sqrt(time_to_expiration)
    mu = (cost_of_carry - volatility**2 / 2) / volatility**2
    lambda_ = np.sqrt(mu**2 + 2 * risk_free_rate / volatility**2)

    phi = -1 if put_option else 1
    eta = 1 if barrier_direction == "down" else -1

    kwargs = {
        "phi": phi,
        "eta": eta,
        "stock_price": stock_price,
        "strike_price": strike_price,
        "barrier": barrier,
        "risk_free_rate": risk_free_rate,
        "cost_of_carry": cost_of_carry,
        "volatility": volatility,
        "time_to_expiration": time_to_expiration,
    }

    A = _barrier_term(term="A", **kwargs)
    B = _barrier_term(term="B", **kwargs)
    C = _barrier_term(term="C", **kwargs)
    D = _barrier_term(term="D", **kwargs)

    x2 = np.log(stock_price / barrier) / volatility_time + (1 + mu) * volatility_time
    y2 = np.log(barrier / stock_price) / volatility_time + (1 + mu) * volatility_time
    z = np.log(barrier / stock_price) / volatility_time + lambda_ * volatility_time

    # Rebate terms: E pays at expiration for a knock-in, F on a knock-out touch.
    E = (
        rebate
        * np.exp(-risk_free_rate * time_to_expiration)
        * (
            ndtr(eta * x2 - eta * volatility_time)
            - (barrier / stock_price) ** (2 * mu)
            * ndtr(eta * y2 - eta * volatility_time)
        )
    )
    F = rebate * (
        (barrier / stock_price) ** (mu + lambda_) * ndtr(eta * z)
        + (barrier / stock_price) ** (mu - lambda_)
        * ndtr(eta * z - 2 * eta * lambda_ * volatility_time)
    )

    # The formula per option, barrier direction and knock type, as the value with the strike
    # above the barrier and the value with the strike at or below it.
    if not put_option:
        if barrier_direction == "down" and knock_type == "in":
            beyond, other = C + E, A - B + D + E
        elif barrier_direction == "down" and knock_type == "out":
            beyond, other = A - C + F, B - D + F
        elif barrier_direction == "up" and knock_type == "in":
            beyond, other = A + E, B - C + D + E
        else:  # up-and-out
            beyond, other = F, A - B + C - D + F
    elif barrier_direction == "down" and knock_type == "in":
        beyond, other = B - C + D + E, A + E
    elif barrier_direction == "down" and knock_type == "out":
        beyond, other = A - B + C - D + F, F
    elif barrier_direction == "up" and knock_type == "in":
        beyond, other = A - B + D + E, C + E
    else:  # up-and-out
        beyond, other = B - D + F, A - C + F

    value = np.where(strike_price > barrier, beyond, other)

    # The formulas assume the barrier has not been reached yet. Once it has, a knock-out
    # option is gone and pays its rebate, and a knock-in option is a regular option;
    # the formulas would otherwise give e.g. a negative knock-out price.
    breached = (
        stock_price <= barrier
        if barrier_direction == "down"
        else stock_price >= barrier
    )

    if knock_type == "out":
        value = np.where(breached, rebate, value)
    elif np.any(breached):
        d1 = (
            np.log(stock_price / strike_price)
            + (cost_of_carry + volatility**2 / 2) * time_to_expiration
        ) / volatility_time
        d2 = d1 - volatility_time
        value = np.where(
            breached,
            phi
            * (
                stock_price
                * np.exp((cost_of_carry - risk_free_rate) * time_to_expiration)
                * ndtr(phi * d1)
                - strike_price
                * np.exp(-risk_free_rate * time_to_expiration)
                * ndtr(phi * d2)
            ),
            value,
        )

    return value if np.ndim(value) else float(value)


@np.errstate(divide="ignore", invalid="ignore")
def get_asian_option(
    stock_price: float | np.ndarray,
    strike_price: float | np.ndarray,
    risk_free_rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float | np.ndarray = 0.0,
    put_option: bool = False,
) -> float | np.ndarray:
    """
    Calculate the closed-form price of a geometric-average Asian option: what an option on the
    continuously sampled geometric average of the stock price is worth, with the Kemna & Vorst
    (1990) formula.

    Also known as: geometric Asian option, average rate option, average price option.

    An Asian option's payoff depends on the average price of the underlying stock over
    the option's life, rather than the price at a single point in time, which typically
    makes it cheaper than the equivalent vanilla option (the averaging reduces
    variance). The package prices the continuously sampled geometric-average Asian
    option only. The arithmetic-average Asian option has no closed-form solution and
    needs a path simulation that averages the price along each path, which
    options_model.get_monte_carlo_option_price does not do: it prices European
    options from the terminal price. The geometric-average version has an elegant
    closed form instead and, since the geometric mean never exceeds the arithmetic mean,
    is a lower bound for the arithmetic-average call and an upper bound for the
    arithmetic-average put. Since the geometric average
    of a Geometric Brownian Motion is itself lognormally distributed, the Asian option
    can be priced with a Black-Scholes-style formula using an adjusted volatility and
    an adjusted cost of carry:

    - σ_A = σ / √3
    - b_A = 0.5 * (b — σ²/6), where b = r — q is the cost of carry
    - d1 = (ln(S / K) + (b_A + σ_A²/2) * t) / (σ_A * √t)
    - d2 = d1 — σ_A * √t
    - Call Price = S * e^((b_A — r) * t) * N(d1) — K * e^(—r * t) * N(d2)
    - Put Price = K * e^(—r * t) * N(—d2) — S * e^((b_A — r) * t) * N(—d1)

    Where S is the stock price, K is the strike price, r is the risk-free rate, q is
    the dividend yield, σ is the volatility, t is the time to expiration, N(d1) is the
    cumulative normal distribution of d1 and N(d2) is the cumulative normal
    distribution of d2.

    Shares similar logic:
    - QuantLib, AnalyticContinuousGeometricAveragePriceAsianEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_geometric_average_price_asian_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option.
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.

    Returns:
        float | np.ndarray: The geometric-average Asian Option value, an array in the shape
        of the inputs when any of them is an array.

    Raises:
        TypeError: if any of the numeric inputs is not a float, int or np.ndarray.

    Notes:
        Reference: Kemna, A. G. Z., & Vorst, A. C. F. (1990). "A Pricing Method for
        Options Based on Average Asset Values." Journal of Banking & Finance, 14(1),
        113-129.
    """
    validate_numeric_inputs(
        arrays=True,
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    validate_positive_inputs(stock_price=stock_price, strike_price=strike_price)

    cost_of_carry = risk_free_rate - dividend_yield

    adjusted_volatility = volatility / np.sqrt(3)
    adjusted_cost_of_carry = 0.5 * (cost_of_carry - volatility**2 / 6)

    d1 = (
        np.log(stock_price / strike_price)
        + (adjusted_cost_of_carry + adjusted_volatility**2 / 2) * time_to_expiration
    ) / (adjusted_volatility * np.sqrt(time_to_expiration))
    d2 = d1 - adjusted_volatility * np.sqrt(time_to_expiration)

    if put_option:
        return strike_price * np.exp(-risk_free_rate * time_to_expiration) * ndtr(
            -d2
        ) - stock_price * np.exp(
            (adjusted_cost_of_carry - risk_free_rate) * time_to_expiration
        ) * ndtr(
            -d1
        )

    return stock_price * np.exp(
        (adjusted_cost_of_carry - risk_free_rate) * time_to_expiration
    ) * ndtr(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiration) * ndtr(
        d2
    )
