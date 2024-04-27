"""Derivative Models"""

import numpy as np
from scipy.stats import norm


def get_black_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float,
    notional: float = 10_000_000,
    is_receiver: bool = True,
) -> tuple[float, float]:
    """
    Black's Model for pricing financial derivatives.

    Black's Model is a mathematical model used for pricing financial derivatives, its primary applications are for
    pricing options on future contracts, bond options, interest rate cap and floors, and swaptions.

    Swaption is an option on an interest rate swap that gives the holder the right, but not the obligation,
    to enter into the swap at a predetermined fixed rate (strike rate) at a specified future time (maturity).

    For more information, see: https://en.wikipedia.org/wiki/Black_model

    Args:
        forward_rate (float): Forward rate of the underlying swap.
        strike_rate (float): Strike rate of the swaption.
        volatility (float): Volatility of the underlying swap.
        years_to_maturity (float): years to maturity of the swaption.
        current_time (float, optional): Current time. Default is 0.
        notional (float, optional): Notional amount of the swap. Default is 100.
        tenor (float, optional): Tenor of the underlying swap. Defaults to being equal to years to maturity.
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.
    """
    d1 = (
        np.log(forward_rate / strike_rate) + 0.5 * volatility**2 * years_to_maturity
    ) / (volatility * np.sqrt(years_to_maturity))
    d2 = d1 - volatility * np.sqrt(years_to_maturity)

    if is_receiver:
        payoff = -forward_rate * norm.cdf(-d1) + strike_rate * norm.cdf(-d2)
    else:
        payoff = forward_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2)

    present_value = np.exp(-risk_free_rate * years_to_maturity)
    swaption_price = notional * present_value * payoff

    return swaption_price, payoff


def get_bachelier_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float,
    notional: float = 10_000_000,
    is_receiver: bool = True,
) -> tuple[float, float]:
    """
    Bachelier Model for pricing future contracts.

    The Bachelier Model is an alternative to Black's Model for pricing options on futures contracts.
    It assumes that the distribution of the underlying asset follows a normal distribution with constant volatility.

    For more information, see: https://en.wikipedia.org/wiki/Bachelier_model

    Args:
        forward_rate (float): Forward rate of the underlying swap.
        strike_rate (float): Strike rate of the swaption.
        volatility (float): Volatility of the underlying swap.
        years_to_maturity (float): years to maturity of the swaption.
        risk_free_rate (float): The risk-free interest rate.
        notional (float, optional): Notional amount of the swap. Default is 100.
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.
    """
    d = (forward_rate - strike_rate) / (volatility * np.sqrt(years_to_maturity))

    if is_receiver:
        payoff = (strike_rate - forward_rate) * norm.cdf(-d) + volatility * np.sqrt(
            years_to_maturity
        ) * norm.pdf(-d)
    else:
        payoff = (forward_rate - strike_rate) * norm.cdf(d) + volatility * np.sqrt(
            years_to_maturity
        ) * norm.pdf(d)

    present_value = np.exp(-risk_free_rate * years_to_maturity)
    swaption_price = notional * present_value * payoff

    return swaption_price, payoff
