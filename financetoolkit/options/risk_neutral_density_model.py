"""Risk-Neutral Density (Breeden-Litzenberger) Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options import black_scholes_model, svi_model

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals


def get_risk_neutral_density(
    stock_price: float,
    forward_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float,
    svi_parameters: dict[str, float],
    strike_price_range: float = 0.5,
    number_of_strikes: int = 200,
) -> pd.Series:
    """
    Extract the market-implied risk-neutral probability density of the underlying's
    price at expiration, via the Breeden-Litzenberger (1978) theorem, from a
    calibrated SVI implied volatility smile (see `svi_model.get_svi_parameters`) --
    as opposed to a single flat (constant-across-strikes) assumed volatility, which
    can only ever recover a lognormal density and defeats the purpose of the
    theorem, since the entire point is to recover whatever (typically non-lognormal,
    fat-tailed and/or skewed) density the market's actual smile implies.

    The theorem states that the risk-neutral density is the second partial
    derivative of the (discounted) call price with respect to the strike price:

    - f(K) = e^(r * t) * d^2 C(K) / dK^2

    Where C(K) is the Black-Scholes call price at strike K, using the SVI-smoothed
    implied volatility at that strike (rather than the raw, noisy market quotes),
    r is the risk-free rate and t is the time to expiration. The second derivative
    is approximated numerically via a central finite difference on a fine,
    evenly-spaced strike grid, since the smile only gives implied volatility at a
    sparse set of traded strikes.

    See the paper: Breeden, D.T., & Litzenberger, R.H. (1978), "Prices of
    State-Contingent Claims Implicit in Option Prices", Journal of Business, 51(4),
    621-651. https://www.jstor.org/stable/2352653

    Also known as: Breeden-Litzenberger, implied risk-neutral distribution.

    Notes:
        Negative density values indicate a butterfly-arbitrage violation (the call
        price is not convex in the strike) at that point, either in the underlying
        market quotes or introduced by the SVI fit -- a well-calibrated, liquid
        smile should not produce any.

    Args:
        stock_price (float): The current price of the underlying.
        forward_price (float): The forward price of the underlying, F = S * e^((r -
            q) * t).
        time_to_expiration (float): The time to expiration, in years.
        risk_free_rate (float): The risk-free rate.
        dividend_yield (float): The dividend yield.
        svi_parameters (dict[str, float]): The calibrated SVI parameters ("a", "b",
            "rho", "m", "sigma") for this expiry, see `svi_model.get_svi_parameters`.
        strike_price_range (float): The range of strikes to evaluate the density
            over, as a fraction of the forward price in each direction. Defaults to
            0.5, i.e. from 50% to 150% of the forward price.
        number_of_strikes (int): The number of strikes in the evaluation grid. A
            finer grid gives a smoother density but the theorem is only exact in
            the continuum limit. Defaults to 200.

    Returns:
        pd.Series: The risk-neutral probability density, indexed by strike price.
            The first and last grid points are dropped, since the central finite
            difference needs a neighbor on each side.
    """
    strike_grid = np.linspace(
        forward_price * (1 - strike_price_range),
        forward_price * (1 + strike_price_range),
        number_of_strikes,
    )
    strike_step = strike_grid[1] - strike_grid[0]

    log_moneyness = np.log(strike_grid / forward_price)
    implied_volatility = svi_model.get_svi_implied_volatility(
        log_moneyness, time_to_expiration, **svi_parameters
    )

    call_price = black_scholes_model.get_black_scholes(
        stock_price=stock_price,
        strike_price=strike_grid,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        put_option=False,
    )

    density = np.exp(risk_free_rate * time_to_expiration) * np.gradient(
        np.gradient(call_price, strike_step), strike_step
    )

    return pd.Series(density[1:-1], index=strike_grid[1:-1], name="Density")
