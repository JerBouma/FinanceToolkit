"""Exotic Options Model"""

__docformat__ = "google"

import numpy as np
from scipy.stats import norm

# pylint: disable=too-many-arguments,too-many-locals,invalid-name


def _validate_numeric_inputs(**kwargs) -> None:
    """
    Validate that the provided keyword arguments are numeric (float or int) and raise
    a TypeError otherwise.

    Args:
        **kwargs: the arguments to validate, keyed by argument name.

    Raises:
        TypeError: if any of the provided values is not a float or int.
    """
    for name, value in kwargs.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a float or int, received {type(value).__name__}."
            )


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
        ) * norm.cdf(phi * x1) - phi * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * norm.cdf(
            phi * x1 - phi * volatility_time
        )

    if term == "B":
        x2 = (
            np.log(stock_price / barrier) / volatility_time + (1 + mu) * volatility_time
        )
        return phi * stock_price * np.exp(
            (cost_of_carry - risk_free_rate) * time_to_expiration
        ) * norm.cdf(phi * x2) - phi * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * norm.cdf(
            phi * x2 - phi * volatility_time
        )

    if term == "C":
        y1 = (
            np.log(barrier**2 / (stock_price * strike_price)) / volatility_time
            + (1 + mu) * volatility_time
        )
        return phi * stock_price * np.exp(
            (cost_of_carry - risk_free_rate) * time_to_expiration
        ) * (barrier / stock_price) ** (2 * (mu + 1)) * norm.cdf(
            eta * y1
        ) - phi * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * (
            barrier / stock_price
        ) ** (
            2 * mu
        ) * norm.cdf(
            eta * y1 - eta * volatility_time
        )

    # term == "D"
    y2 = np.log(barrier / stock_price) / volatility_time + (1 + mu) * volatility_time
    return phi * stock_price * np.exp(
        (cost_of_carry - risk_free_rate) * time_to_expiration
    ) * (barrier / stock_price) ** (2 * (mu + 1)) * norm.cdf(
        eta * y2
    ) - phi * strike_price * np.exp(
        -risk_free_rate * time_to_expiration
    ) * (
        barrier / stock_price
    ) ** (
        2 * mu
    ) * norm.cdf(
        eta * y2 - eta * volatility_time
    )


def get_barrier_option(
    stock_price: float,
    strike_price: float,
    barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    barrier_direction: str = "down",
    knock_type: str = "out",
    rebate: float = 0.0,
) -> float:
    """
    Calculate the closed-form price of a single-barrier (knock-in or knock-out, up or
    down) European option using the Reiner & Rubinstein (1991) formulas.

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
    implementation) is in-out parity: for identical parameters, a knock-in option plus
    its corresponding knock-out option (same direction) always equals the price of the
    equivalent vanilla European option, since the underlying either does or does not
    touch the barrier:

    - down-and-in + down-and-out = vanilla option
    - up-and-in + up-and-out = vanilla option

    This holds only when ``rebate`` is zero. A rebate is an extra cash payment attached
    to whichever leg does not survive, so it is added to both legs rather than split
    between them and the sum then exceeds the vanilla price.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        barrier (float): The barrier level. Should be below stock_price when
            barrier_direction is "down" and above stock_price when barrier_direction is
            "up".
        risk_free_rate (float): The risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option.
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.
        barrier_direction (str): Either "down" or "up". Defaults to "down".
        knock_type (str): Either "in" or "out". Defaults to "out".
        rebate (float): The fixed cash amount paid out (a) immediately for a knock-out
            option if/when the barrier is breached, or (b) at expiration for a knock-in
            option that never breaches the barrier. Defaults to 0.

    Returns:
        float: The Barrier Option value.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if barrier_direction is not "down"/"up" or knock_type is not
        "in"/"out".

    Notes:
        Reference: Reiner, E., & Rubinstein, M. (1991). "Breaking Down the Barriers."
        Risk Magazine, 4(8), 28-35.
    """
    _validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        barrier=barrier,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        rebate=rebate,
    )

    if barrier_direction not in ("down", "up"):
        raise ValueError(
            f"barrier_direction must be 'down' or 'up', received {barrier_direction!r}."
        )

    if knock_type not in ("in", "out"):
        raise ValueError(f"knock_type must be 'in' or 'out', received {knock_type!r}.")

    np.seterr(divide="ignore", invalid="ignore")

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
            norm.cdf(eta * x2 - eta * volatility_time)
            - (barrier / stock_price) ** (2 * mu)
            * norm.cdf(eta * y2 - eta * volatility_time)
        )
    )
    F = rebate * (
        (barrier / stock_price) ** (mu + lambda_) * norm.cdf(eta * z)
        + (barrier / stock_price) ** (mu - lambda_)
        * norm.cdf(eta * z - 2 * eta * lambda_ * volatility_time)
    )

    in_the_money_barrier_side = strike_price > barrier

    if not put_option:
        if barrier_direction == "down" and knock_type == "in":
            value = (A - B + D + E) if not in_the_money_barrier_side else (C + E)
        elif barrier_direction == "down" and knock_type == "out":
            value = (B - D + F) if not in_the_money_barrier_side else (A - C + F)
        elif barrier_direction == "up" and knock_type == "in":
            value = (A + E) if in_the_money_barrier_side else (B - C + D + E)
        else:  # up-and-out
            value = F if in_the_money_barrier_side else (A - B + C - D + F)
    elif barrier_direction == "down" and knock_type == "in":
        value = (B - C + D + E) if in_the_money_barrier_side else (A + E)
    elif barrier_direction == "down" and knock_type == "out":
        value = (A - B + C - D + F) if in_the_money_barrier_side else F
    elif barrier_direction == "up" and knock_type == "in":
        value = (A - B + D + E) if in_the_money_barrier_side else (C + E)
    else:  # up-and-out
        value = (B - D + F) if in_the_money_barrier_side else (A - C + F)

    return value


def get_asian_option(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the closed-form price of a geometric-average Asian option using the
    Kemna & Vorst (1990) formula.

    Also known as: geometric Asian option, average rate option, average price option.

    An Asian option's payoff depends on the average price of the underlying stock over
    the option's life, rather than the price at a single point in time, which typically
    makes it cheaper than the equivalent vanilla option (the averaging reduces
    variance). While the arithmetic-average Asian option has no closed-form solution
    (it requires simulation, see `options_model.get_monte_carlo_option_price`), the
    geometric-average version has an elegant closed form: since the geometric average
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

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option.
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.

    Returns:
        float: The geometric-average Asian Option value.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.

    Notes:
        Reference: Kemna, A. G. Z., & Vorst, A. C. F. (1990). "A Pricing Method for
        Options Based on Average Asset Values." Journal of Banking & Finance, 14(1),
        113-129.
    """
    _validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )

    np.seterr(divide="ignore", invalid="ignore")

    cost_of_carry = risk_free_rate - dividend_yield

    adjusted_volatility = volatility / np.sqrt(3)
    adjusted_cost_of_carry = 0.5 * (cost_of_carry - volatility**2 / 6)

    d1 = (
        np.log(stock_price / strike_price)
        + (adjusted_cost_of_carry + adjusted_volatility**2 / 2) * time_to_expiration
    ) / (adjusted_volatility * np.sqrt(time_to_expiration))
    d2 = d1 - adjusted_volatility * np.sqrt(time_to_expiration)

    if put_option:
        return strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(
            -d2
        ) - stock_price * np.exp(
            (adjusted_cost_of_carry - risk_free_rate) * time_to_expiration
        ) * norm.cdf(
            -d1
        )

    return stock_price * np.exp(
        (adjusted_cost_of_carry - risk_free_rate) * time_to_expiration
    ) * norm.cdf(d1) - strike_price * np.exp(
        -risk_free_rate * time_to_expiration
    ) * norm.cdf(
        d2
    )
