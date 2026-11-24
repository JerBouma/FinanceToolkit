"""Compound-Style Exotic Option Model"""

__docformat__ = "google"

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from financetoolkit.options.helpers import (
    bivariate_normal_cdf,
    validate_numeric_inputs,
)
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.utilities.error_model import check_option

# The critical stock prices are searched between these many e-folds below and above a first
# guess (1e-26 to 1e26 times it), beyond which an option value no longer moves.
_SEARCH_WIDTH = 60.0


def _validate_volatility(volatility: float) -> None:
    """Raises a ValueError for a volatility that is not positive, which the formulas divide by."""
    if np.any(np.asarray(volatility) <= 0):
        raise ValueError(f"volatility must be positive, received {volatility!r}.")


def _critical_price(objective: Callable[[float], float], guess: float) -> float:
    """
    The stock price at which a monotonic function of the stock price crosses zero, searched on
    the log price around the guess. A function that keeps one sign over the whole search
    crosses beyond it: at infinity when it moves towards zero as the price rises (increasing
    and negative, or decreasing and positive), and at 0 otherwise.

    Args:
        objective (Callable[[float], float]): The function of the stock price.
        guess (float): A positive stock price to search around.

    Returns:
        float: The critical stock price, possibly 0 or np.inf.
    """
    low = np.log(guess) - _SEARCH_WIDTH
    high = np.log(guess) + _SEARCH_WIDTH
    at_low = objective(float(np.exp(low)))
    at_high = objective(float(np.exp(high)))

    if at_low == 0:
        return float(np.exp(low))
    if at_high == 0:
        return float(np.exp(high))
    if np.sign(at_low) == np.sign(at_high):
        increasing = at_high > at_low
        return np.inf if increasing == (at_high < 0) else 0.0

    return float(
        np.exp(
            brentq(
                lambda log_price: objective(float(np.exp(log_price))),
                low,
                high,
                xtol=1e-14,
                rtol=1e-15,
            )
        )
    )


def _decision_terms(
    stock_price: float,
    boundary: float,
    risk_free_rate: float,
    volatility: float,
    decision_time: float,
    dividend_yield: float,
) -> tuple[float, float]:
    """
    The terms a1 = (ln(S / B) + (r - q + σ²/2) * t) / (σ√t) and a2 = a1 - σ√t, which decide
    whether the stock price at time t ends above the boundary B: under the risk-neutral
    measure P(S(t) > B) = N(a2) and E[S(t) * 1{S(t) > B}] = S * e^((r - q)t) * N(a1). A
    boundary of 0 gives a1 = a2 = infinity and an infinite boundary minus infinity.
    """
    if boundary <= 0:
        return np.inf, np.inf
    if np.isposinf(boundary):
        return -np.inf, -np.inf

    deviation = volatility * np.sqrt(decision_time)
    first = (
        np.log(stock_price / boundary)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * decision_time
    ) / deviation

    return float(first), float(first - deviation)


def _option_beyond(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    decision_time: float,
    time_to_expiration: float,
    boundary: float,
    side: int,
    put_option: bool,
) -> float:
    """
    The present value of a European option (strike K, expiring at T) held at a decision time
    t < T only where the stock price then lies beyond a boundary B, the building block of
    Geske (1979):

    - e^(-rt) * E[V(S(t), T - t) * 1{ηS(t) > ηB}] = φ * [S * e^(-qT) * M(ηa1, φd1; ηφρ) - K * e^(-rT) * M(ηa2, φd2; ηφρ)]

    With a1 and a2 the decision terms at B over t (see _decision_terms), d1 and d2 the
    Black-Scholes terms of K over T, ρ = √(t / T), φ = 1 for a call and -1 for a put, η = 1
    for the region above B and -1 for the region below it, and M the bivariate standard normal
    distribution function.

    Returns:
        float: The present value.
    """
    sign = -1 if put_option else 1
    first, second = _decision_terms(
        stock_price,
        boundary,
        risk_free_rate,
        volatility,
        decision_time,
        dividend_yield,
    )
    deviation = volatility * np.sqrt(time_to_expiration)
    d1 = (
        np.log(stock_price / strike_price)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiration
    ) / deviation
    correlation = side * sign * np.sqrt(decision_time / time_to_expiration)

    return float(
        sign
        * (
            stock_price
            * np.exp(-dividend_yield * time_to_expiration)
            * float(bivariate_normal_cdf(side * first, sign * d1, correlation))
            - strike_price
            * np.exp(-risk_free_rate * time_to_expiration)
            * float(
                bivariate_normal_cdf(
                    side * second, sign * (d1 - deviation), correlation
                )
            )
        )
    )


def get_chooser_option(
    stock_price: float | np.ndarray,
    strike_price: float | np.ndarray,
    risk_free_rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiration: float | np.ndarray,
    choice_time: float | np.ndarray,
    dividend_yield: float | np.ndarray = 0.0,
    put_strike_price: float | None = None,
    put_time_to_expiration: float | None = None,
) -> float | np.ndarray:
    """
    Calculate the price of a chooser option: what the right to decide at the choice time whether
    the option becomes a European call or a European put is worth (Rubinstein, 1991).

    A simple chooser (put_strike_price and put_time_to_expiration left out) chooses between a
    call and a put with the same strike K and expiration T. By put-call parity at the choice
    time t it is a call on (K, T) plus a put on (K * e^(-(r - q)(T - t)), t), scaled by
    e^(-q(T - t)):

    - Simple Chooser = S * e^(-qT) * N(d) - K * e^(-rT) * N(d - σ√T) - S * e^(-qT) * N(-y) + K * e^(-rT) * N(-y + σ√t)
    - d = (ln(S / K) + (r - q + σ²/2) * T) / (σ√T)
    - y = (ln(S / K) + (r - q) * T + σ²t/2) / (σ√t)

    A complex chooser chooses between a call (Kc, Tc) and a put (Kp, Tp) with their own strike
    and expiration. At t the holder takes the call above the critical price I where both are
    worth the same, c(I, Kc, Tc - t) = p(I, Kp, Tp - t), and the put below it:

    - Complex Chooser = S * e^(-q Tc) * M(a1, d1c; ρc) - Kc * e^(-r Tc) * M(a2, d2c; ρc)
      - S * e^(-q Tp) * M(-a1, -d1p; ρp) + Kp * e^(-r Tp) * M(-a2, -d2p; ρp)
    - a1 = (ln(S / I) + (r - q + σ²/2) * t) / (σ√t), a2 = a1 - σ√t
    - d1c, d2c and d1p, d2p are the Black-Scholes terms of (Kc, Tc) and (Kp, Tp), ρc = √(t / Tc) and ρp = √(t / Tp)

    Where S is the stock price, r the continuously compounded risk-free rate, q the continuous
    dividend yield, σ the volatility, N the standard normal and M the bivariate standard normal
    distribution function. At a choice time of 0 the chooser is worth the larger of the call
    and the put, and at the call's expiration (simple chooser) a straddle.

    Referenced against:
    - QuantLib, AnalyticSimpleChooserEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_simple_chooser_engine.html

    Shares similar logic:
    - FinancePy, EquityChooserOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_chooser_option.py#L67

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The strike price of the call, and of the put for a simple chooser.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the call in years, and of the
            put for a simple chooser.
        choice_time (float): The time in years at which the holder chooses, at least 0 and at
            most the earliest expiration.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_strike_price (float | None, optional): The strike price of the put of a complex
            chooser. Defaults to None, which uses strike_price.
        put_time_to_expiration (float | None, optional): The time to expiration of the put of a
            complex chooser in years. Defaults to None, which uses time_to_expiration.

    Returns:
        float | np.ndarray: The chooser option value. The inputs of a simple chooser may be
        arrays (a grid of strikes and expirations), which give an array in their shape.

    Raises:
        ValueError: If volatility is not positive or choice_time is not between 0 and the
            earliest expiration.

    Notes:
        References: Rubinstein, M. (1991). "Options for the Undecided." Risk, 4(4), 43.
        Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas, 2nd ed., section
        4.10.

    As an example:

    ```python
    from financetoolkit.options.pricing import compound_model

    simple = compound_model.get_chooser_option(
        stock_price=50,
        strike_price=50,
        risk_free_rate=0.08,
        volatility=0.25,
        time_to_expiration=0.5,
        choice_time=0.25,
    )
    complex_chooser = compound_model.get_chooser_option(
        stock_price=50,
        strike_price=55,
        risk_free_rate=0.1,
        volatility=0.35,
        time_to_expiration=0.5,
        choice_time=0.25,
        dividend_yield=0.05,
        put_strike_price=48,
        put_time_to_expiration=0.5833,
    )

    round(simple, 4), round(complex_chooser, 4)
    ```

    Which returns:

    ```
    (6.1071, 6.0507)
    ```
    """
    # A simple chooser has a closed form, which prices arrays of strikes, expirations and
    # choice times (a grid) in one call.
    simple = put_strike_price is None and put_time_to_expiration is None
    validate_numeric_inputs(
        arrays=simple,
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        choice_time=choice_time,
        dividend_yield=dividend_yield,
    )
    _validate_volatility(volatility)

    put_strike = strike_price if put_strike_price is None else put_strike_price
    put_expiration = (
        time_to_expiration if put_time_to_expiration is None else put_time_to_expiration
    )
    validate_numeric_inputs(
        arrays=simple,
        put_strike_price=put_strike,
        put_time_to_expiration=put_expiration,
    )

    if not (
        np.all(np.asarray(choice_time) >= 0)
        and np.all(choice_time <= np.minimum(time_to_expiration, put_expiration))
    ):
        raise ValueError(
            "choice_time must be at least 0 and at most the earliest expiration, "
            f"received {choice_time!r}."
        )

    inputs = {"risk_free_rate": risk_free_rate, "volatility": volatility}
    call = np.asarray(
        get_black_scholes(
            stock_price,
            strike_price,
            time_to_expiration=time_to_expiration,
            dividend_yield=dividend_yield,
            **inputs,
        ),
        dtype=float,
    )

    if simple or (put_strike == strike_price and put_expiration == time_to_expiration):
        # At a choice time of 0 the chooser is the larger of the call and the put. Later,
        # the put leg is a put on K * e^(-(r - q)(T - t)) expiring at t, scaled by
        # e^(-q(T - t)).
        remaining = time_to_expiration - choice_time
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.where(
                np.asarray(choice_time) == 0,
                np.maximum(
                    call,
                    get_black_scholes(
                        stock_price,
                        put_strike,
                        time_to_expiration=put_expiration,
                        dividend_yield=dividend_yield,
                        put_option=True,
                        **inputs,
                    ),
                ),
                call
                + np.exp(-dividend_yield * remaining)
                * get_black_scholes(
                    stock_price,
                    strike_price
                    * np.exp(-(risk_free_rate - dividend_yield) * remaining),
                    time_to_expiration=choice_time,
                    dividend_yield=dividend_yield,
                    put_option=True,
                    **inputs,
                ),
            )

        return value if value.ndim else float(value)

    if choice_time == 0:
        return max(
            float(call),
            float(
                get_black_scholes(
                    stock_price,
                    put_strike,
                    time_to_expiration=put_expiration,
                    dividend_yield=dividend_yield,
                    put_option=True,
                    **inputs,
                )
            ),
        )

    def call_minus_put(price: float) -> float:
        return float(
            get_black_scholes(
                price,
                strike_price,
                time_to_expiration=time_to_expiration - choice_time,
                dividend_yield=dividend_yield,
                **inputs,
            )
        ) - float(
            get_black_scholes(
                price,
                put_strike,
                time_to_expiration=put_expiration - choice_time,
                dividend_yield=dividend_yield,
                put_option=True,
                **inputs,
            )
        )

    critical_price = _critical_price(call_minus_put, strike_price)
    shared = {
        "stock_price": stock_price,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
        "decision_time": choice_time,
        "boundary": critical_price,
    }

    return _option_beyond(
        **shared,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
        side=1,
        put_option=False,
    ) + _option_beyond(
        **shared,
        strike_price=put_strike,
        time_to_expiration=put_expiration,
        side=-1,
        put_option=True,
    )


def get_compound_option(
    stock_price: float,
    strike_price: float,
    compound_strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    compound_time_to_expiration: float,
    dividend_yield: float = 0.0,
    option_type: Literal[
        "call_on_call", "call_on_put", "put_on_call", "put_on_put"
    ] = "call_on_call",
) -> float:
    """
    Calculate the price of a compound option: what an option to buy (call) or sell (put) a
    European option on the stock for the compound strike K1 at time T1 is worth (Geske, 1979).

    The underlying option has strike K2 and expires at T2 ≥ T1.

    At T1 the compound option is exercised where the underlying option V is worth more (call)
    or less (put) than K1, which for a call on a call happens above the critical price S* with
    V(S*, K2, T2 - T1) = K1. With ψ = 1 for a compound call and -1 for a compound put, φ = 1 for
    an underlying call and -1 for an underlying put, and η = ψφ the side of S* on which it is
    exercised:

    - Compound = ψ * [φ * (S * e^(-q T2) * M(ηa1, φd1; ηφρ) - K2 * e^(-r T2) * M(ηa2, φd2; ηφρ))
      - K1 * e^(-r T1) * N(ηa2)]
    - a1 = (ln(S / S*) + (r - q + σ²/2) * T1) / (σ√T1), a2 = a1 - σ√T1
    - d1 = (ln(S / K2) + (r - q + σ²/2) * T2) / (σ√T2), d2 = d1 - σ√T2, ρ = √(T1 / T2)

    which for a call on a call is the formula of Geske (1979): S * e^(-q T2) * M(a1, d1; ρ)
    - K2 * e^(-r T2) * M(a2, d2; ρ) - K1 * e^(-r T1) * N(a2). Where S is the stock price, r the
    continuously compounded risk-free rate, q the continuous dividend yield, σ the volatility,
    N the standard normal and M the bivariate standard normal distribution function. A put is
    worth at most K2 * e^(-r(T2 - T1)) at T1, so for a compound strike above it S* is 0: a call
    on that put is worthless and a put on it is a forward.

    Referenced against:
    - QuantLib, AnalyticCompoundOptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_compound_option_engine.html

    Shares similar logic:
    - FinancePy, EquityCompoundOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_compound_option.py#L26

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The strike price K2 of the underlying option.
        compound_strike_price (float): The compound strike K1, the price paid (call) or
            received (put) for the underlying option at T1.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration T2 of the underlying option in years.
        compound_time_to_expiration (float): The time to expiration T1 of the compound option
            in years, at least 0 and at most T2.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        option_type (str, optional): The compound option first, then the option it is on.
            Options are: "call_on_call" (a call on a call), "call_on_put" (a call on a put),
            "put_on_call" (a put on a call), "put_on_put" (a put on a put). Defaults to
            "call_on_call".

    Returns:
        float: The compound option value.

    Raises:
        ValueError: If option_type is not one of the four types, volatility is not positive or
            compound_time_to_expiration is not between 0 and time_to_expiration.

    Notes:
        References: Geske, R. (1979). "The Valuation of Compound Options." Journal of
        Financial Economics, 7(1), 63-81. Haug, E. G. (2007). The Complete Guide to Option
        Pricing Formulas, 2nd ed., section 4.9.

    As an example:

    ```python
    from financetoolkit.options.pricing import compound_model

    put_on_call = compound_model.get_compound_option(
        stock_price=500,
        strike_price=520,
        compound_strike_price=50,
        risk_free_rate=0.08,
        volatility=0.35,
        time_to_expiration=0.5,
        compound_time_to_expiration=0.25,
        dividend_yield=0.03,
        option_type="put_on_call",
    )

    round(put_on_call, 4)
    ```

    Which returns:

    ```
    21.1964
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        compound_strike_price=compound_strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        compound_time_to_expiration=compound_time_to_expiration,
        dividend_yield=dividend_yield,
    )
    _validate_volatility(volatility)
    check_option(
        "option_type",
        option_type,
        ("call_on_call", "call_on_put", "put_on_call", "put_on_put"),
    )
    if not 0 <= compound_time_to_expiration <= time_to_expiration:
        raise ValueError(
            "compound_time_to_expiration must be at least 0 and at most time_to_expiration, "
            f"received {compound_time_to_expiration!r}."
        )

    compound_sign = 1 if option_type.startswith("call") else -1
    put_option = option_type.endswith("put")
    remaining = time_to_expiration - compound_time_to_expiration

    def underlying(price: float) -> float:
        return float(
            get_black_scholes(
                price,
                strike_price,
                risk_free_rate,
                volatility,
                remaining,
                dividend_yield,
                put_option,
            )
        )

    if compound_time_to_expiration == 0:
        return max(
            compound_sign * (underlying(stock_price) - compound_strike_price), 0.0
        )

    critical_price = _critical_price(
        lambda price: underlying(price) - compound_strike_price, strike_price
    )
    side = compound_sign * (-1 if put_option else 1)
    _, second = _decision_terms(
        stock_price,
        critical_price,
        risk_free_rate,
        volatility,
        compound_time_to_expiration,
        dividend_yield,
    )

    return compound_sign * (
        _option_beyond(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            dividend_yield,
            compound_time_to_expiration,
            time_to_expiration,
            critical_price,
            side,
            put_option,
        )
        - compound_strike_price
        * float(np.exp(-risk_free_rate * compound_time_to_expiration))
        * float(ndtr(side * second))
    )


def get_forward_start_option(
    stock_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    start_time: float,
    moneyness: float = 1.0,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    performance: bool = False,
) -> float:
    """
    Calculate the price of a forward-start option: what a European option is worth whose strike
    is set at the start time t1 to moneyness α times the stock price then (Rubinstein, 1990).

    The performance option, which pays the return over [t1, T] beyond α instead, is priced as
    well.

    The option on S(T) with strike α * S(t1) is S(t1) times an option on a unit stock price
    with strike α, so its value at t1 is S(t1) * BS(1, α, T - t1). Taken back to today:

    - Forward Start = S * e^(-q t1) * BS(1, α, T - t1), the payoff max(S(T) - α * S(t1), 0) for a call
    - Performance = e^(-r t1) * BS(1, α, T - t1), the payoff max(S(T) / S(t1) - α, 0) for a call
    - BS(1, α, τ) = e^(-qτ) * N(d1) - α * e^(-rτ) * N(d1 - σ√τ), d1 = (ln(1 / α) + (r - q + σ²/2) * τ) / (σ√τ)

    for a call, and the Black-Scholes put on the unit stock price for a put. Where S is the
    stock price, r the continuously compounded risk-free rate, q the continuous dividend
    yield, σ the volatility and N the standard normal distribution function. A performance
    option is a forward-start option divided by S only when r = q.

    Referenced against:
    - QuantLib, ForwardEuropeanEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_forward_vanilla_engine.html

    Shares similar logic:
    - FinancePy, EquityForwardStartOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_forward_start_option.py#L32

    Args:
        stock_price (float): The current stock price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration T in years.
        start_time (float): The time t1 in years at which the strike is set, at least 0 and at
            most T.
        moneyness (float, optional): The strike as a share α of the stock price at t1.
            Defaults to 1, at the money.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        performance (bool, optional): Whether to price the performance option, which pays the
            return S(T) / S(t1) beyond α, a decimal per unit of notional. Defaults to False.

    Returns:
        float: The forward-start (or performance) option value.

    Raises:
        ValueError: If volatility or moneyness is not positive or start_time is not between 0
            and time_to_expiration.

    Notes:
        References: Rubinstein, M. (1990). "Pay Now, Choose Later." Risk, 3(2). Haug, E. G.
        (2007). The Complete Guide to Option Pricing Formulas, 2nd ed., section 4.6.

    As an example:

    ```python
    from financetoolkit.options.pricing import compound_model

    forward_start = compound_model.get_forward_start_option(
        stock_price=60,
        risk_free_rate=0.08,
        volatility=0.3,
        time_to_expiration=1,
        start_time=0.25,
        moneyness=1.1,
        dividend_yield=0.04,
    )

    round(forward_start, 4)
    ```

    Which returns:

    ```
    4.4065
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        start_time=start_time,
        moneyness=moneyness,
        dividend_yield=dividend_yield,
    )
    _validate_volatility(volatility)
    if moneyness <= 0:
        raise ValueError(f"moneyness must be positive, received {moneyness!r}.")
    if not 0 <= start_time <= time_to_expiration:
        raise ValueError(
            "start_time must be at least 0 and at most time_to_expiration, "
            f"received {start_time!r}."
        )

    unit_option = float(
        get_black_scholes(
            1.0,
            moneyness,
            risk_free_rate,
            volatility,
            time_to_expiration - start_time,
            dividend_yield,
            put_option,
        )
    )

    if performance:
        return float(np.exp(-risk_free_rate * start_time) * unit_option)

    return float(stock_price * np.exp(-dividend_yield * start_time) * unit_option)


def get_cliquet_option(
    stock_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    reset_times: Sequence[float],
    moneyness: float = 1.0,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    performance: bool = False,
) -> float:
    """
    Calculate the price of a cliquet (ratchet) option: what a series of forward-start options is
    worth, each with its strike reset to moneyness α times the stock price at the start of its
    period and paid at the end of that period.

    The same series of performance options is priced as well.

    With reset times t1 < t2 < ... < tn and t(n+1) = T the last period's end, the cliquet is the
    sum of the forward-start options over each period (see get_forward_start_option):

    - Cliquet = Σ S * e^(-q ti) * BS(1, α, t(i+1) - ti)
    - Performance Cliquet = Σ e^(-r ti) * BS(1, α, t(i+1) - ti)

    Where S is the stock price, r the continuously compounded risk-free rate, q the continuous
    dividend yield and BS(1, α, τ) the Black-Scholes value of an option on a unit stock price
    with strike α and expiration τ. A reset time of 0 starts the first period today; with one
    reset time the cliquet is a forward-start option.

    Referenced against:
    - QuantLib, AnalyticCliquetEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_cliquet_engine.html
    - QuantLib, AnalyticPerformanceEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_performance_engine.html

    Args:
        stock_price (float): The current stock price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration T in years, the end of the last
            period.
        reset_times (Sequence[float]): The start of each period in years, increasing, at
            least 0 and below T.
        moneyness (float, optional): The strike as a share α of the stock price at each reset.
            Defaults to 1, at the money.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the periods are puts. Defaults to False.
        performance (bool, optional): Whether each period pays its return S(t(i+1)) / S(ti)
            beyond α, a decimal per unit of notional. Defaults to False.

    Returns:
        float: The cliquet option value.

    Raises:
        ValueError: If reset_times is empty, not increasing or not within [0, T), or
            volatility or moneyness is not positive.

    Notes:
        References: Rubinstein, M. (1990). "Pay Now, Choose Later." Risk, 3(2). Haug, E. G.
        (2007). The Complete Guide to Option Pricing Formulas, 2nd ed., section 4.6.

    As an example:

    ```python
    from financetoolkit.options.pricing import compound_model

    cliquet = compound_model.get_cliquet_option(
        stock_price=100,
        risk_free_rate=0.05,
        volatility=0.3,
        time_to_expiration=1,
        reset_times=[0, 0.25, 0.5, 0.75],
        dividend_yield=0.02,
    )

    round(cliquet, 4)
    ```

    Which returns:

    ```
    25.0344
    ```
    """
    times = [float(time) for time in reset_times]
    if not times:
        raise ValueError("reset_times must hold at least one reset time.")
    validate_numeric_inputs(
        time_to_expiration=time_to_expiration,
        **{f"reset_times[{index}]": time for index, time in enumerate(reset_times)},
    )
    ends = [*times[1:], time_to_expiration]
    if times[0] < 0 or any(end <= start for start, end in zip(times, ends)):
        raise ValueError(
            "reset_times must be increasing, at least 0 and below time_to_expiration, "
            f"received {list(reset_times)!r}."
        )

    return sum(
        get_forward_start_option(
            stock_price,
            risk_free_rate,
            volatility,
            end,
            start,
            moneyness,
            dividend_yield,
            put_option,
            performance,
        )
        for start, end in zip(times, ends)
    )


def get_extendible_option(
    stock_price: float,
    strike_price: float,
    extended_strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    extended_time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: Literal["holder", "writer"] = "holder",
    extension_fee: float = 0.0,
) -> float:
    """
    Calculate the price of an extendible option: what a European option with strike K1 expiring
    at t1 is worth when its life can be extended to T2 with a new strike K2 (Longstaff, 1990).

    - Holder-extendible: at t1 the holder exercises, lets the option expire
      or pays the extension fee A to hold an option on (K2, T2) instead, whichever is worth
      most.
    - Writer-extendible: an option that is out of the money at t1 is extended
      to (K2, T2) automatically, so the holder receives the extended option instead of nothing.

    Both are the vanilla option on (K1, t1) plus the extended option over the region where it
    replaces the vanilla payoff. With V the Black-Scholes option on (K2, T2 - t1), the
    holder of a call extends between I1 with V(I1) = A and I2 with V(I2) - A = I2 - K1, and the
    holder of a put between I1 with V(I1) - A = K1 - I1 and I2 with V(I2) = A:

    - Holder Call = S * e^(-q t1) * N(a1(I2)) - K1 * e^(-r t1) * N(a2(I2)) + C(I1) - C(I2)
      - A * e^(-r t1) * [N(a2(I1)) - N(a2(I2))]
    - Holder Put = K1 * e^(-r t1) * N(-a2(I1)) - S * e^(-q t1) * N(-a1(I1)) + C(I1) - C(I2)
      - A * e^(-r t1) * [N(a2(I1)) - N(a2(I2))]
    - Writer Call = c(S, K1, t1) + S * e^(-q T2) * M(-a1(K1), d1; -ρ) - K2 * e^(-r T2) * M(-a2(K1), d2; -ρ)
    - Writer Put = p(S, K1, t1) + K2 * e^(-r T2) * M(a2(K1), -d2; -ρ) - S * e^(-q T2) * M(a1(K1), -d1; -ρ)
    - a1(B) = (ln(S / B) + (r - q + σ²/2) * t1) / (σ√t1), a2(B) = a1(B) - σ√t1
    - d1 = (ln(S / K2) + (r - q + σ²/2) * T2) / (σ√T2), d2 = d1 - σ√T2, ρ = √(t1 / T2)
    - C(B) = φ * [S * e^(-q T2) * M(a1(B), φd1; φρ) - K2 * e^(-r T2) * M(a2(B), φd2; φρ)], the extended
      option where S(t1) > B, with φ = 1 for a call and -1 for a put

    Where S is the stock price, r the continuously compounded risk-free rate, q the continuous
    dividend yield, σ the volatility, N the standard normal and M the bivariate standard normal
    distribution function. A holder never extends when V(K1) ≤ A, which leaves the vanilla
    option; a boundary that is never reached is 0 or infinity (A = 0 makes I1 = 0 for a call).

    Referenced against:
    - QuantLib, AnalyticHolderExtensibleOptionEngine (holder-extendible): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_holder_extensible_option_engine.html
    - QuantLib, AnalyticWriterExtensibleOptionEngine (writer-extendible): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_writer_extensible_option_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The strike price K1 of the original option.
        extended_strike_price (float): The strike price K2 of the extended option.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The original time to expiration t1 in years, above 0.
        extended_time_to_expiration (float): The extended time to expiration T2 in years, at
            least t1.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        method (str, optional): Who decides on the extension. Options are: "holder"
            (holder-extendible), "writer" (writer-extendible). Defaults to "holder".
        extension_fee (float, optional): The fee A the holder pays at t1 to extend, ignored
            for a writer-extendible option. Defaults to 0.

    Returns:
        float: The extendible option value.

    Raises:
        ValueError: If method is not "holder" or "writer", volatility is not positive,
            extension_fee is negative or the times are not 0 < t1 ≤ T2.

    Notes:
        References: Longstaff, F. A. (1990). "Pricing Options with Extendible Maturities:
        Analysis and Applications." The Journal of Finance, 45(3), 935-957. Haug, E. G.
        (2007). The Complete Guide to Option Pricing Formulas, 2nd ed., section 4.11.

    As an example:

    ```python
    from financetoolkit.options.pricing import compound_model

    writer_extendible = compound_model.get_extendible_option(
        stock_price=80,
        strike_price=90,
        extended_strike_price=82,
        risk_free_rate=0.1,
        volatility=0.3,
        time_to_expiration=0.5,
        extended_time_to_expiration=0.75,
        method="writer",
    )

    round(writer_extendible, 4)
    ```

    Which returns:

    ```
    6.8238
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        extended_strike_price=extended_strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        extended_time_to_expiration=extended_time_to_expiration,
        dividend_yield=dividend_yield,
        extension_fee=extension_fee,
    )
    _validate_volatility(volatility)
    check_option("method", method, ("holder", "writer"))
    if extension_fee < 0:
        raise ValueError(
            f"extension_fee must be at least 0, received {extension_fee!r}."
        )
    if not 0 < time_to_expiration <= extended_time_to_expiration:
        raise ValueError(
            "The times must satisfy 0 < time_to_expiration <= extended_time_to_expiration, "
            f"received {time_to_expiration!r} and {extended_time_to_expiration!r}."
        )

    inputs = {
        "stock_price": stock_price,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
    }
    vanilla = float(
        get_black_scholes(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiration,
            dividend_yield,
            put_option,
        )
    )

    def extended(boundary: float, side: int) -> float:
        return _option_beyond(
            **inputs,
            strike_price=extended_strike_price,
            decision_time=time_to_expiration,
            time_to_expiration=extended_time_to_expiration,
            boundary=boundary,
            side=side,
            put_option=put_option,
        )

    if method == "writer":
        # The out-of-the-money region at t1: below K1 for a call, above it for a put.
        return vanilla + extended(strike_price, 1 if put_option else -1)

    remaining = extended_time_to_expiration - time_to_expiration

    def extended_value(price: float) -> float:
        return (
            float(
                get_black_scholes(
                    price,
                    extended_strike_price,
                    risk_free_rate,
                    volatility,
                    remaining,
                    dividend_yield,
                    put_option,
                )
            )
            - extension_fee
        )

    if extended_value(strike_price) <= 0:
        return vanilla

    sign = -1 if put_option else 1

    def against_exercise(price: float) -> float:
        if put_option:
            return extended_value(price) - (strike_price - price)
        # c(S) - (S - K1) - A by put-call parity, p(S) + S * (e^(-qτ) - 1) + K1 - K2 * e^(-rτ) - A,
        # which keeps its sign far above the strike, where c(S) - S cancels to rounding noise.
        return (
            float(
                get_black_scholes(
                    price,
                    extended_strike_price,
                    risk_free_rate,
                    volatility,
                    remaining,
                    dividend_yield,
                    True,
                )
            )
            + price * np.expm1(-dividend_yield * remaining)
            + strike_price
            - extended_strike_price * np.exp(-risk_free_rate * remaining)
            - extension_fee
        )

    if put_option:
        lower = _critical_price(against_exercise, strike_price)
        upper = _critical_price(extended_value, strike_price)
    else:
        lower = _critical_price(extended_value, strike_price)
        upper = _critical_price(against_exercise, strike_price)

    terms = {
        boundary: _decision_terms(
            stock_price,
            boundary,
            risk_free_rate,
            volatility,
            time_to_expiration,
            dividend_yield,
        )
        for boundary in (lower, upper)
    }
    # The exercise region: above I2 for a call, below I1 for a put.
    first, second = terms[lower if put_option else upper]
    exercise = float(sign) * float(
        stock_price
        * np.exp(-dividend_yield * time_to_expiration)
        * float(ndtr(sign * first))
        - strike_price
        * np.exp(-risk_free_rate * time_to_expiration)
        * float(ndtr(sign * second))
    )
    extension_probability = float(ndtr(terms[lower][1]) - ndtr(terms[upper][1]))

    return (
        exercise
        + extended(lower, 1)
        - extended(upper, 1)
        - extension_fee
        * float(np.exp(-risk_free_rate * time_to_expiration))
        * extension_probability
    )


def get_quanto_option(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    foreign_risk_free_rate: float,
    volatility: float,
    exchange_rate_volatility: float,
    correlation: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    fixed_exchange_rate: float = 1.0,
    start_time: float = 0.0,
) -> float:
    """
    Calculate the price of a quanto option: what a European option on a stock quoted in a
    foreign currency is worth when its payoff is paid in the domestic currency at a fixed
    exchange rate E (Reiner, 1992).

    Paying at a fixed rate makes the stock drift at r_f - q - ρσσ_E in the domestic risk-neutral
    measure, so the option is a Black-Scholes option with a quanto-adjusted dividend yield:

    - Quanto = E * BS(S, K, r, q', σ, T), q' = q + r - r_f + ρ * σ * σ_E
    - Quanto Call = E * [S * e^((r_f - q - ρσσ_E - r)T) * N(d1) - K * e^(-rT) * N(d1 - σ√T)]
    - d1 = (ln(S / K) + (r_f - q - ρσσ_E + σ²/2) * T) / (σ√T)

    With a start time t1 above 0 it is a quanto forward-start option, whose strike is set at t1
    to the same share K / S of the stock price then (see get_forward_start_option):

    - Quanto Forward Start = E * S * e^(-q' t1) * BS(1, K / S, r, q', σ, T - t1)

    Where S and K are the stock and strike price in the foreign currency, r the domestic and
    r_f the foreign continuously compounded risk-free rate, q the continuous dividend yield, σ
    the volatility of the stock, σ_E the volatility of the exchange rate (domestic currency per
    unit of foreign currency), ρ the correlation between the stock's and the exchange rate's
    returns and N the standard normal distribution function.

    Referenced against:
    - QuantLib, QuantoEuropeanEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_quanto_engine.html
    - QuantLib, QuantoForwardEuropeanEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_quanto_engine.html

    Args:
        stock_price (float): The current stock price in the foreign currency.
        strike_price (float): The strike price in the foreign currency, or with a start time
            its share of the stock price (moneyness K / S) at which the strike is set.
        risk_free_rate (float): The continuously compounded domestic risk-free rate.
        foreign_risk_free_rate (float): The continuously compounded foreign risk-free rate.
        volatility (float): The volatility of the stock.
        exchange_rate_volatility (float): The volatility of the exchange rate.
        correlation (float): The correlation between the stock and the exchange rate, between
            -1 and 1.
        time_to_expiration (float): The time to expiration T in years.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        fixed_exchange_rate (float, optional): The fixed exchange rate E in domestic currency
            per unit of foreign currency. Defaults to 1.
        start_time (float, optional): The time t1 in years at which the strike is set, at most
            T. Defaults to 0, a quanto option with the strike set today.

    Returns:
        float: The quanto option value in the domestic currency.

    Raises:
        ValueError: If a volatility is negative (or the stock's is 0), the correlation is not
            between -1 and 1 or start_time is not between 0 and time_to_expiration.

    Notes:
        References: Reiner, E. (1992). "Quanto Mechanics." Risk, 5(3), 59-63. Haug, E. G.
        (2007). The Complete Guide to Option Pricing Formulas, 2nd ed., section 5.15.

    As an example:

    ```python
    from financetoolkit.options.pricing import compound_model

    quanto = compound_model.get_quanto_option(
        stock_price=100,
        strike_price=105,
        risk_free_rate=0.08,
        foreign_risk_free_rate=0.05,
        volatility=0.2,
        exchange_rate_volatility=0.1,
        correlation=0.3,
        time_to_expiration=0.5,
        dividend_yield=0.04,
        fixed_exchange_rate=1.5,
    )

    round(quanto, 4)
    ```

    Which returns:

    ```
    5.328
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        foreign_risk_free_rate=foreign_risk_free_rate,
        volatility=volatility,
        exchange_rate_volatility=exchange_rate_volatility,
        correlation=correlation,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        fixed_exchange_rate=fixed_exchange_rate,
        start_time=start_time,
    )
    _validate_volatility(volatility)
    if exchange_rate_volatility < 0:
        raise ValueError(
            "exchange_rate_volatility must be at least 0, "
            f"received {exchange_rate_volatility!r}."
        )
    if not -1 <= correlation <= 1:
        raise ValueError(
            f"correlation must be between -1 and 1, received {correlation!r}."
        )

    quanto_yield = (
        dividend_yield
        + risk_free_rate
        - foreign_risk_free_rate
        + correlation * volatility * exchange_rate_volatility
    )

    if start_time == 0:
        value = float(
            get_black_scholes(
                stock_price,
                strike_price,
                risk_free_rate,
                volatility,
                time_to_expiration,
                quanto_yield,
                put_option,
            )
        )
    else:
        value = get_forward_start_option(
            stock_price,
            risk_free_rate,
            volatility,
            time_to_expiration,
            start_time,
            strike_price / stock_price,
            quanto_yield,
            put_option,
        )

    return fixed_exchange_rate * value
