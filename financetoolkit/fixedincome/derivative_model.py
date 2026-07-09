"""Derivative Models"""

import numpy as np
from scipy.stats import norm


def _get_annuity_factor(
    risk_free_rate: float,
    years_to_maturity: float,
    tenor: float,
    payment_frequency: int,
) -> float:
    """
    Calculate the annuity factor (present value of a basis point, PVBP) of the swap
    underlying a swaption.

    Exercising a swaption does not pay out a single lump sum at expiration — it grants
    the right to enter a swap that exchanges cash flows at every payment date over the
    swap's tenor. The annuity factor is the sum of the discount factors to each of
    those payment dates, weighted by the accrual fraction between them, and is what
    converts a per-period rate differential into the present value of the full stream
    of swap cash flows.

    The formula is as follows:

        Annuity = SUM_(k=1)^(n) [ accrual * DiscountFactor(years_to_maturity + k * accrual) ]

    Where accrual = 1 / payment_frequency and n = tenor * payment_frequency is the
    number of payments made over the life of the underlying swap.

    Args:
        risk_free_rate (float): The risk-free interest rate used to discount each
        payment date, assumed flat across the curve.
        years_to_maturity (float): Years to expiration of the swaption, i.e. the time
        at which the underlying swap would start if exercised.
        tenor (float): The tenor (length in years) of the underlying swap.
        payment_frequency (int): The number of fixed-leg payments per year on the
        underlying swap (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly).

    Returns:
        float: The annuity factor (PVBP) of the underlying swap.
    """
    number_of_payments = max(round(tenor * payment_frequency), 1)
    accrual = 1 / payment_frequency
    payment_times = years_to_maturity + accrual * np.arange(1, number_of_payments + 1)

    return accrual * np.sum(np.exp(-risk_free_rate * payment_times))


def get_black_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float,
    notional: float = 10_000_000,
    tenor: float | None = None,
    payment_frequency: int = 2,
    is_receiver: bool = True,
) -> tuple[float, float]:
    """
    Black's Model for pricing financial derivatives.

    Black's Model is a mathematical model used for pricing financial derivatives, its primary applications are for
    pricing options on future contracts, bond options, interest rate cap and floors, and swaptions.

    Swaption is an option on an interest rate swap that gives the holder the right, but not the obligation,
    to enter into the swap at a predetermined fixed rate (strike rate) at a specified future time (maturity).

    Exercising a swaption is not a single payment at expiration but the right to enter a swap that exchanges
    cash flows at every payment date over the underlying swap's tenor. The Black-76 swaption price therefore
    discounts the option payoff by the swap's annuity (present value of a basis point) rather than a single
    discount factor to expiration — see `_get_annuity_factor`.

    For more information, see: https://en.wikipedia.org/wiki/Black_model

    Args:
        forward_rate (float): Forward rate of the underlying swap.
        strike_rate (float): Strike rate of the swaption.
        volatility (float): Volatility of the underlying swap.
        years_to_maturity (float): years to maturity of the swaption.
        risk_free_rate (float): The risk-free interest rate.
        notional (float, optional): Notional amount of the swap. Default is 10,000,000.
        tenor (float | None, optional): Tenor of the underlying swap. Defaults to being equal to years to maturity.
        payment_frequency (int, optional): Number of fixed-leg payments per year on the underlying swap
        (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly). Defaults to 2 (semi-annual).
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.
    """
    tenor = years_to_maturity if tenor is None else tenor

    d1 = (
        np.log(forward_rate / strike_rate) + 0.5 * volatility**2 * years_to_maturity
    ) / (volatility * np.sqrt(years_to_maturity))
    d2 = d1 - volatility * np.sqrt(years_to_maturity)

    if is_receiver:
        payoff = -forward_rate * norm.cdf(-d1) + strike_rate * norm.cdf(-d2)
    else:
        payoff = forward_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2)

    annuity = _get_annuity_factor(
        risk_free_rate=risk_free_rate,
        years_to_maturity=years_to_maturity,
        tenor=tenor,
        payment_frequency=payment_frequency,
    )
    swaption_price = notional * annuity * payoff

    return swaption_price, payoff


def get_bachelier_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float,
    notional: float = 10_000_000,
    tenor: float | None = None,
    payment_frequency: int = 2,
    is_receiver: bool = True,
) -> tuple[float, float]:
    """
    Bachelier Model for pricing future contracts.

    The Bachelier Model is an alternative to Black's Model for pricing options on futures contracts.
    It assumes that the distribution of the underlying asset follows a normal distribution with constant volatility.

    Exercising a swaption is not a single payment at expiration but the right to enter a swap that exchanges
    cash flows at every payment date over the underlying swap's tenor. The swaption price therefore discounts
    the option payoff by the swap's annuity (present value of a basis point) rather than a single discount
    factor to expiration — see `_get_annuity_factor`.

    For more information, see: https://en.wikipedia.org/wiki/Bachelier_model

    Args:
        forward_rate (float): Forward rate of the underlying swap.
        strike_rate (float): Strike rate of the swaption.
        volatility (float): Volatility of the underlying swap.
        years_to_maturity (float): years to maturity of the swaption.
        risk_free_rate (float): The risk-free interest rate.
        notional (float, optional): Notional amount of the swap. Default is 10,000,000.
        tenor (float | None, optional): Tenor of the underlying swap. Defaults to being equal to years to maturity.
        payment_frequency (int, optional): Number of fixed-leg payments per year on the underlying swap
        (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly). Defaults to 2 (semi-annual).
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.
    """
    tenor = years_to_maturity if tenor is None else tenor

    d = (forward_rate - strike_rate) / (volatility * np.sqrt(years_to_maturity))

    if is_receiver:
        payoff = (strike_rate - forward_rate) * norm.cdf(-d) + volatility * np.sqrt(
            years_to_maturity
        ) * norm.pdf(-d)
    else:
        payoff = (forward_rate - strike_rate) * norm.cdf(d) + volatility * np.sqrt(
            years_to_maturity
        ) * norm.pdf(d)

    annuity = _get_annuity_factor(
        risk_free_rate=risk_free_rate,
        years_to_maturity=years_to_maturity,
        tenor=tenor,
        payment_frequency=payment_frequency,
    )
    swaption_price = notional * annuity * payoff

    return swaption_price, payoff
