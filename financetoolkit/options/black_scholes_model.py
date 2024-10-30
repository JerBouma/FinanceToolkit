"""Black Scholes Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.stats import norm


def get_d1(
    stock_price: float | pd.Series,
    strike_price: float | pd.Series,
    risk_free_rate: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
    dividend_yield: float | pd.Series = 0,
):
    """
    Calculate d1 in the Black-Scholes model for option pricing.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.
        dividend_yield (float or pd.Series): The dividend yield of the stock. Defaults to 0.

    Returns:
        float | pd.Series: The d1 value.
    """
    np.seterr(divide="ignore", invalid="ignore")

    return (
        np.log(stock_price / strike_price)
        + (risk_free_rate - dividend_yield + (volatility**2) / 2) * time_to_expiration
    ) / (volatility * np.sqrt(time_to_expiration))


def get_d2(
    d1: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
):
    """
    Calculate d2 in the Black-Scholes model for option pricing.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.

    Returns:
        float | pd.Series: The d2 value.
    """
    np.seterr(divide="ignore", invalid="ignore")

    return d1 - volatility * np.sqrt(time_to_expiration)


def get_black_scholes(
    stock_price: float | pd.Series,
    strike_price: float | pd.Series,
    risk_free_rate: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
    dividend_yield: float | pd.Series = 0,
    put_option: bool = False,
):
    """
    Calculate the Black-Scholes model for option pricing.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.
        dividend_yield (float or pd.Series): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.

    Returns:
        float | pd.Series: The Black-Scholes value.
    """
    d1 = get_d1(
        stock_price, strike_price, risk_free_rate, volatility, time_to_expiration
    )
    d2 = get_d2(d1, volatility, time_to_expiration)

    if put_option:
        return strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(
            -d2
        ) - stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1)

    return stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(
        d1
    ) - strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)
