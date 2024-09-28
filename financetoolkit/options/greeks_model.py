"""Black Scholes Greeks Model"""

__docformat__ = "google"

import numpy as np
from scipy.stats import norm

from financetoolkit.options import black_scholes_model


def get_delta(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the Option Delta using the Black-Scholes formula.

    Args:
        stock_price (float): Series of stock prices.
        strike_price (float): Option strike price.
        time_to_expiry (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.

    Returns:
        float: Option Delta values.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )

    if put_option:
        option_delta = -np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1)

    else:
        option_delta = np.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1)

    return option_delta


def get_dual_delta(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate Dual Delta using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiry (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Dual Delta value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )

    if put_option:
        dual_delta = np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2)
    else:
        dual_delta = -np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)

    return dual_delta


def get_vega(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate the Option Vega using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.

    Returns:
        float: Option Vega value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    vega = (
        stock_price
        * np.exp(-dividend_yield * time_to_expiration)
        * norm.pdf(d1)
        * np.sqrt(time_to_expiration)
    )

    # Divide by 100 to get the vega value
    vega = vega / 100

    return vega


def get_theta(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the Option Theta using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Option Theta value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    if put_option:
        theta = -np.exp(-dividend_yield * time_to_expiration) * (
            (stock_price * norm.pdf(d1) * volatility)
            / (2 * np.sqrt(time_to_expiration))
            + risk_free_rate
            * strike_price
            * np.exp(-risk_free_rate * time_to_expiration)
            * norm.cdf(-d2)
            - dividend_yield
            * stock_price
            * np.exp(-dividend_yield * time_to_expiration)
            * norm.cdf(-d1)
        )
    else:
        theta = (
            -np.exp(-dividend_yield * time_to_expiration)
            * (stock_price * norm.pdf(d1) * volatility)
            / (2 * np.sqrt(time_to_expiration))
            - risk_free_rate
            * strike_price
            * np.exp(-risk_free_rate * time_to_expiration)
            * norm.cdf(d2)
            + dividend_yield
            * stock_price
            * np.exp(-dividend_yield * time_to_expiration)
            * norm.cdf(d1)
        )

    # Divide by 365 to get the theta value
    theta = theta / 365

    return theta


def get_rho(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the Option Rho using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Option Rho value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    if put_option:
        rho = (
            -strike_price
            * time_to_expiration
            * np.exp(-risk_free_rate * time_to_expiration)
            * norm.cdf(-d2)
        )
    else:
        rho = (
            strike_price
            * time_to_expiration
            * np.exp(-risk_free_rate * time_to_expiration)
            * norm.cdf(d2)
        )

    return rho


def get_epsilon(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the Option Epsilon using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Option Epsilon value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )

    if put_option:
        epsilon = (
            stock_price
            * time_to_expiration
            * np.exp(-dividend_yield * time_to_expiration)
            * norm.cdf(-d1)
        )
    else:
        epsilon = (
            -stock_price
            * time_to_expiration
            * np.exp(-dividend_yield * time_to_expiration)
            * norm.cdf(d1)
        )

    return epsilon


def get_lambda(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the Option Lambda using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.

    Returns:
        float: Option Lambda value.
    """
    delta = get_delta(
        stock_price=stock_price,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        put_option=put_option,
    )

    option_price = black_scholes_model.get_black_scholes(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        put_option=put_option,
    )

    lambda_value = delta * (stock_price / option_price)

    return lambda_value


def get_gamma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate the Option Gamma using the Black-Scholes formula.

    Args:
        stock_price (float): Series of stock prices.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

    Returns:
        float: Option Gamma values.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )
    gamma = np.exp(-dividend_yield * time_to_expiration) * (
        norm.pdf(d1) / (stock_price * volatility * np.sqrt(time_to_expiration))
    )

    return gamma


def get_dual_gamma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Dual Gamma using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiry (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Dual Gamma value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )

    dual_gamma = np.exp(-risk_free_rate * time_to_expiration) * (
        norm.pdf(d2) / (strike_price * volatility * np.sqrt(time_to_expiration))
    )

    return dual_gamma


def get_vanna(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Vanna using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Vanna value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    vanna = (
        -np.exp(-dividend_yield * time_to_expiration) * norm.pdf(d1) * (d2 / volatility)
    )

    return vanna


def get_charm(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate Charm using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Charm value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    if put_option:
        charm = -dividend_yield * np.exp(
            -dividend_yield * time_to_expiration
        ) * norm.cdf(-d1) + np.exp(-dividend_yield * time_to_expiration) * norm.pdf(
            -d1
        ) * (
            2 * (risk_free_rate - dividend_yield) * time_to_expiration
            - d2 * volatility * np.sqrt(time_to_expiration)
        ) / (
            2 * time_to_expiration * volatility * np.sqrt(time_to_expiration)
        )

    else:
        charm = dividend_yield * np.exp(
            -dividend_yield * time_to_expiration
        ) * norm.cdf(d1) - np.exp(-dividend_yield * time_to_expiration) * norm.pdf(
            d1
        ) * (
            2 * (risk_free_rate - dividend_yield) * time_to_expiration
            - d2 * volatility * np.sqrt(time_to_expiration)
        ) / (
            2 * time_to_expiration * volatility * np.sqrt(time_to_expiration)
        )

    return charm


def get_vomma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Vomma using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

    Returns:
        float: Vomma value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    vomma = (
        stock_price
        * np.exp(-dividend_yield * time_to_expiration)
        * norm.pdf(d1)
        * np.sqrt(time_to_expiration)
        * (d1 * d2)
        / volatility
    )

    return vomma


def get_vera(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Vera using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

    Returns:
        float: Vera value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    vera = (
        -strike_price
        * time_to_expiration
        * np.exp(-risk_free_rate * time_to_expiration)
        * norm.pdf(d2)
        * (d1 / volatility)
    )

    return vera


def get_veta(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Veta using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

    Returns:
        float: Veta value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    veta = (
        -stock_price
        * np.exp(-dividend_yield * time_to_expiration)
        * norm.pdf(d1)
        * np.sqrt(time_to_expiration)
        * (
            dividend_yield
            + ((risk_free_rate - dividend_yield) * d1)
            / (volatility * np.sqrt(time_to_expiration))
            - (1 + d1 * d2) / (2 * time_to_expiration)
        )
    )

    # Divide by 100 and multiply by 365 to get the veta value
    veta = veta / 100 * 365

    return veta


def get_second_order_partial_derivative(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
) -> float:
    """
    Calculate the second order partial derivative of the Black-Scholes model
    with respect to the Strike Price.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

    Returns:
        float: The second order partial derivative of the Black-Scholes model with
        respect to the strike price.
    """
    second_order_partial_derivative = (
        np.exp(-risk_free_rate * time_to_expiration)
        * (1 / strike_price)
        * (1 / np.sqrt(2 * np.pi * volatility**2 * time_to_expiration))
        * np.exp(
            -(1 / (2 * volatility**2 * time_to_expiration))
            * (
                np.log(strike_price / stock_price)
                - ((risk_free_rate - dividend_yield) - (0.5 * volatility**2))
                * time_to_expiration
            )
            ** 2
        )
    )

    return second_order_partial_derivative


def get_speed(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Speed using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

    Returns:
        float: Speed value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )

    speed = (
        -np.exp(-dividend_yield * time_to_expiration)
        * (norm.pdf(d1) / (stock_price**2 * volatility * np.sqrt(time_to_expiration)))
        * ((d1 / (volatility * np.sqrt(time_to_expiration))) + 1)
    )

    return speed


def get_zomma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Zomma using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiry (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.

    Returns:
        float: Zomma value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    zomma = np.exp(-dividend_yield * time_to_expiration) * (
        (norm.pdf(d1) * (d1 * d2 - 1))
        / (stock_price * volatility**2 * np.sqrt(time_to_expiration))
    )

    return zomma


def get_color(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Color using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiry (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.

    Returns:
        float: Color value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )

    color = (
        -np.exp(-dividend_yield * time_to_expiration)
        * (
            norm.pdf(d1)
            / (
                2
                * stock_price
                * time_to_expiration
                * volatility
                * np.sqrt(time_to_expiration)
            )
        )
        * (
            2 * dividend_yield * time_to_expiration
            + 1
            + (
                (
                    2 * (risk_free_rate - dividend_yield) * time_to_expiration
                    - d2 * volatility * np.sqrt(time_to_expiration)
                )
                / (volatility * np.sqrt(time_to_expiration))
            )
            * d1
        )
    )

    return color


def get_ultima(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Ultima using the Black-Scholes formula.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiry (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.

    Returns:
        float: Ultima value.
    """
    d1 = black_scholes_model.get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )

    vega = get_vega(
        stock_price=stock_price,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    ultima = (-vega / volatility**2) * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)

    return ultima
