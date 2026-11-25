"""Black Scholes Greeks Model"""

__docformat__ = "google"

import numpy as np
from scipy.stats import norm

from financetoolkit.options import helpers
from financetoolkit.options.pricing import black_scholes_model


@np.errstate(divide="ignore", invalid="ignore")
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
    Calculate the Option Delta using the Black-Scholes formula: how much the option value
    changes for a 1.00 change in the price of the underlying asset.

    Args:
        stock_price (float): Series of stock prices.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.
        put_option (bool): True if it's a put option, False for a call option.

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


@np.errstate(divide="ignore", invalid="ignore")
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
    Calculate Dual Delta using the Black-Scholes formula: how much the option value changes
    for a 1.00 change in the strike price.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
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


@np.errstate(divide="ignore", invalid="ignore")
def get_vega(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate the Option Vega using the Black-Scholes formula: how much the option value
    changes for a 1.00 change in the volatility of the underlying asset.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.

    Returns:
        float: Option Vega value.

    Notes:
        The raw derivative dV/dsigma is divided by 100, so the returned value is the
        change in the option value per 1 percentage point change in volatility. The
        higher order volatility Greeks (Vanna, Vomma, Zomma, Vera, Ultima) are returned
        unscaled, per 1.00 of volatility; only Vega and Veta carry the 1/100 factor.
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


@np.errstate(divide="ignore", invalid="ignore")
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
    Calculate the Option Theta using the Black-Scholes formula: how much the option value
    changes as time passes, its time decay.

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

    Notes:
        Theta is dV/dt, the change in the option value as calendar time passes, which is
        the negative of the derivative with respect to the time to maturity. It is
        divided by 365, so the returned value is the change per calendar day rather than
        per year. A long option normally has a negative Theta.
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
        theta = (
            -np.exp(-dividend_yield * time_to_expiration)
            * (stock_price * norm.pdf(d1) * volatility)
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

    # Without uncertainty left (at expiration or without volatility) the option is worth its
    # discounted intrinsic value off the strike, so only the carry of the strike and of the
    # stock remains: -r K e^(-r t) + q S e^(-q t) for a call in the money, mirrored for a put.  # noqa: E501
    if put_option:
        limit = risk_free_rate * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * norm.cdf(-d2) - dividend_yield * stock_price * np.exp(
            -dividend_yield * time_to_expiration
        ) * norm.cdf(
            -d1
        )
    else:
        limit = -risk_free_rate * strike_price * np.exp(
            -risk_free_rate * time_to_expiration
        ) * norm.cdf(d2) + dividend_yield * stock_price * np.exp(
            -dividend_yield * time_to_expiration
        ) * norm.cdf(
            d1
        )

    theta = helpers.apply_expiration_limit(
        theta,
        limit,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    # Divide by 365 to get the theta value
    theta = theta / 365

    return theta


@np.errstate(divide="ignore", invalid="ignore")
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
    Calculate the Option Rho using the Black-Scholes formula: how much the option value
    changes for a 1.00 change in the risk free rate.

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

    Notes:
        Rho is returned unscaled, as the change in the option value per 1.00 (i.e. 100
        percentage points) change in the risk-free rate. Divide by 100 for the more
        common per-basis-point-style quote of one percentage point. Epsilon and Vera
        follow the same unscaled convention.
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


@np.errstate(divide="ignore", invalid="ignore")
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
    Calculate the Option Epsilon using the Black-Scholes formula: how much the option value
    changes for a 1.00 change in the dividend yield.

    It is the derivative of the option value with respect to the dividend yield, ∂V/∂q, per
    1.00 change in the dividend yield, not a percentage change.

    - Call Epsilon = —S * t * e^(—q * t) * N(d1)
    - Put Epsilon = S * t * e^(—q * t) * N(—d1)

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


@np.errstate(divide="ignore", over="ignore", invalid="ignore")
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
    Calculate the Option Lambda using the Black-Scholes formula: the percentage change in
    the option value per percentage change in the price of the underlying asset, the
    leverage the option gives.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.
        put_option (bool): True if it's a put option, False for a call option.

    Returns:
        float: Option Lambda value, NaN where the option is worth nothing or so little that
            the elasticity is not finite.
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

    # An option worth nothing (out of the money at expiration, or deep out of the money) has
    # no elasticity, so its lambda is NaN rather than the inf of a division by zero. A price so
    # small it underflows (about 1e-300 a day before expiry) overflows the division the same way.
    lambda_value = np.asarray(delta * np.divide(stock_price, option_price), dtype=float)
    lambda_value = np.where(np.isfinite(lambda_value), lambda_value, np.nan)

    return float(lambda_value) if lambda_value.ndim == 0 else lambda_value


@np.errstate(divide="ignore", invalid="ignore")
def get_gamma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate the Option Gamma using the Black-Scholes formula: how much the delta changes
    for a 1.00 change in the price of the underlying asset.

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
        dividend_yield=dividend_yield,
    )
    gamma = np.exp(-dividend_yield * time_to_expiration) * (
        norm.pdf(d1) / (stock_price * volatility * np.sqrt(time_to_expiration))
    )

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        gamma,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_dual_gamma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Dual Gamma using the Black-Scholes formula: how much the dual delta changes
    for a 1.00 change in the strike price.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.

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

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        dual_gamma,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_vanna(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Vanna using the Black-Scholes formula: how much the delta changes for a 1.00
    change in the volatility, or equally the vega for a 1.00 change in the price of the
    underlying asset.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

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

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        vanna,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
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
    Calculate Charm using the Black-Scholes formula: how much the delta changes as time
    passes, the delta decay.

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

    Notes:
        Charm is dDelta/dt, the change in Delta as calendar time passes, which is the
        negative of the derivative with respect to the time to maturity. Unlike Theta it
        is returned per year rather than per day; divide by 365 for delta decay per
        calendar day. Color follows the same per-year convention.
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
        ) * norm.cdf(-d1) - np.exp(-dividend_yield * time_to_expiration) * norm.pdf(
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

    # Without uncertainty left Delta is e^(-q t) in the money and 0 out of it, which only
    # changes through the dividend yield.
    limit = (
        -dividend_yield * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1)
        if put_option
        else dividend_yield
        * np.exp(-dividend_yield * time_to_expiration)
        * norm.cdf(d1)
    )

    return helpers.apply_expiration_limit(
        charm,
        limit,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_vomma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Vomma using the Black-Scholes formula: how much the vega changes for a 1.00
    change in the volatility, the convexity of the option value in the volatility.

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

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        vomma,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_vera(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Vera using the Black-Scholes formula: how much the rho changes for a 1.00
    change in the volatility.

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
        # np.divide gives inf rather than raising for a volatility of 0.0.
        * np.divide(d1, volatility)
    )

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        vera,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_veta(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Veta using the Black-Scholes formula: how much the vega changes as time
    passes.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Default is 0.0.

    Returns:
        float: Veta value.

    Notes:
        Veta is the rate of change of Vega as calendar time passes, dVega/dt, which is
        the negative of the derivative with respect to the remaining time to maturity t.
        The closed form below equals dVega/dt as it stands, so no sign is changed:

        - Veta = S * e^(—q * t) * n(d1) * √t * (q + (r — q) * d1 / (σ√t) — (1 + d1 * d2) / (2t))

        This points Veta the same way as Theta, Charm and Color, which all measure the
        change per unit of time elapsed rather than per unit of remaining maturity: an
        option close to the money loses Vega as expiry approaches, so Veta is typically
        negative for it.

        The result is divided by (100 * 365), expressing it per 1 percentage point of
        volatility per calendar day.
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

    # The published closed form is dVega/dt as it stands, so Veta measures the change per unit of time elapsed, as Theta, Charm and Color do.  # noqa: E501
    veta = (
        stock_price
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

    # Expressed per 1% change in volatility per day, matching the documented convention.
    veta = veta / (100 * 365)

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        veta,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_second_order_partial_derivative(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
) -> float:
    """
    Calculate the second order partial derivative of the Black-Scholes model with respect to
    the Strike Price: how fast the dual delta changes with the strike price, the second
    derivative of the option value with respect to the strike price.

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
    # The density is per unit of strike and in logarithms of both prices.
    helpers.validate_positive_inputs(stock_price=stock_price, strike_price=strike_price)
    # The variance up to expiration as a NumPy value, so that a time to expiration or a
    # volatility of 0.0 divides to inf (and the limit below) instead of raising.
    variance = np.multiply(volatility**2, time_to_expiration)

    second_order_partial_derivative = (
        np.exp(-risk_free_rate * time_to_expiration)
        * (1 / strike_price)
        * (1 / np.sqrt(2 * np.pi * variance))
        * np.exp(
            -(1 / (2 * variance))
            * (
                np.log(strike_price / stock_price)
                - ((risk_free_rate - dividend_yield) - (0.5 * volatility**2))
                * time_to_expiration
            )
            ** 2
        )
    )

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        second_order_partial_derivative,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_speed(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Speed using the Black-Scholes formula: how much the gamma changes for a 1.00
    change in the price of the underlying asset.

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

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        speed,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_zomma(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Zomma using the Black-Scholes formula: how much the gamma changes for a 1.00
    change in the volatility.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
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
        dividend_yield=dividend_yield,
    )
    d2 = black_scholes_model.get_d2(
        d1=d1, volatility=volatility, time_to_expiration=time_to_expiration
    )

    zomma = np.exp(-dividend_yield * time_to_expiration) * (
        (norm.pdf(d1) * (d1 * d2 - 1))
        / (stock_price * volatility**2 * np.sqrt(time_to_expiration))
    )

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        zomma,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_color(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Color using the Black-Scholes formula: how much the gamma changes as time
    passes.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
        risk_free_rate (float): Risk-free interest rate (annualized).
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float): Dividend yield (annualized). Defaults to 0.

    Returns:
        float: Color value.

    Notes:
        Color is dGamma/dt, the change in Gamma as calendar time passes, which is the
        negative of the derivative with respect to the remaining time to maturity t. The
        closed form below equals dGamma/dt as it stands:

        - Color = e^(—q * t) * n(d1) / (2 * S * t * σ√t) * (2qt + 1 + (2(r — q)t — d2 * σ√t) / (σ√t) * d1)

        Tables that print this expression with a leading minus sign give the derivative
        with respect to the time to maturity instead. Color here points the same way as
        Theta, Charm and Veta. The result is per year, matching Charm; divide by 365 for
        gamma decay per calendar day.
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
        np.exp(-dividend_yield * time_to_expiration)
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

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        color,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_ultima(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Ultima using the Black-Scholes formula: how much the vomma changes for a 1.00
    change in the volatility.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Option strike price.
        time_to_expiration (float): Time to option expiry (in years).
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

    # get_vega is scaled by 1/100, and Ultima's formula needs the raw vega.
    vega = (
        get_vega(
            stock_price=stock_price,
            strike_price=strike_price,
            time_to_expiration=time_to_expiration,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )
        * 100
    )

    ultima = (-vega / volatility**2) * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)

    # At expiration or without volatility the curvature is gone off the strike.
    return helpers.apply_expiration_limit(
        ultima,
        0.0,
        stock_price,
        strike_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
        dividend_yield,
    )
