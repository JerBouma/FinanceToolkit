"""Constant Elasticity of Variance Model"""

__docformat__ = "google"

import numpy as np
from scipy.stats import ncx2

from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes


def get_cev(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    beta: float = 0.5,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the price of a European option in the constant elasticity of variance (CEV)
    model of Cox (1975), with the non-central chi-squared formula of Schroder (1989).

    Also known as: CEV option price, constant elasticity of variance, local volatility
    power model.

    In the CEV model the stock's volatility depends on its price, σ(S) = δ * S^(β — 1).
    With β below 1 the volatility rises as the price falls (the leverage effect), which
    gives the downward sloping skew of equity options that Black-Scholes (β = 1) does not
    have; β = 1/2 is the square-root process and β = 0 an absolute (normal) volatility.
    The stock is absorbed at zero, its default.

    The volatility passed is the local volatility at today's stock price, so that δ =
    σ * S^(1 — β) and the same σ gives the Black-Scholes price at β = 1. The futures price
    F = S * e^((r — q)T) of the expiration then follows dF = δ * e^((r — q)(1 — β)(T — t)) *
    F^β dW, a CEV process with a time-dependent coefficient, whose prices are those of a
    constant δ over the variance v below.

    The formulas are as follows:

    - δ = σ * S^(1 — β), F = S * e^((r — q)T), ν = 1 / (1 — β)
    - v = δ² * (e^(2(r — q)(1 — β)T) — 1) / (2(r — q)(1 — β)), or δ² * T when r = q
    - x = F^(2(1 — β)) / ((1 — β)² * v), y = K^(2(1 — β)) / ((1 — β)² * v)
    - Call = e^(—rT) * [F * (1 — χ²(y; ν + 2, x)) — K * χ²(x; ν, y)]
    - Put = e^(—rT) * [K * (1 — χ²(x; ν, y)) — F * χ²(y; ν + 2, x)]

    Where S is the stock price, K is the strike price, r is the continuously compounded
    risk-free rate, q is the continuous dividend yield, σ is the local volatility at S,
    β is the elasticity, T is the time to expiration in years and χ²(z; k, λ) is the
    distribution function of the non-central chi-squared distribution with k degrees of
    freedom and non-centrality λ. The call and put satisfy put-call parity.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The local volatility at today's stock price, σ = δ * S^(β — 1).
        time_to_expiration (float): The time to expiration of the option, in years. At zero
            the option is worth its exercise value.
        beta (float): The elasticity β, between 0 and 1; 1 is the Black-Scholes model.
            Defaults to 0.5, the square-root process.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float: The CEV value of the option.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if beta is outside 0 to 1 or the volatility is not positive.

    Notes:
        References: Cox, J. C. (1975). "Notes on Option Pricing I: Constant Elasticity of
        Variance Diffusions." Working paper, Stanford University (reprinted in The Journal
        of Portfolio Management, 23(2), 1996); Schroder, M. (1989). "Computing the Constant
        Elasticity of Variance Option Pricing Formula." The Journal of Finance, 44(1),
        211-219. Referenced against QuantLib's AnalyticCEVEngine (https://www.quantlib.org),
        which takes the coefficient of the futures price, √(v / T) here, and agrees within
        1e-12, and against a Monte Carlo simulation of the stock with drift r — q. A β
        above 1 is not accepted: the futures price is then a strict local martingale, so
        the call and put no longer satisfy put-call parity. The put of the example below is
        above the Black-Scholes price of the same put, 4.0241: with β = 0.5 the volatility
        rises as the price falls towards the strike.

    As an example:

    ```python
    from financetoolkit.options.pricing import cev_model

    round(
        cev_model.get_cev(
            stock_price=100, strike_price=90, risk_free_rate=0.04, volatility=0.25,
            time_to_expiration=1, beta=0.5, put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    4.2363
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        beta=beta,
        dividend_yield=dividend_yield,
    )
    if not 0 <= beta <= 1:
        raise ValueError(f"beta must be between 0 and 1, received {beta!r}.")
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, received {volatility!r}.")

    if time_to_expiration <= 0:
        return max(
            (
                (strike_price - stock_price)
                if put_option
                else (stock_price - strike_price)
            ),
            0.0,
        )

    if beta == 1:
        return float(
            get_black_scholes(
                stock_price=stock_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_expiration=time_to_expiration,
                dividend_yield=dividend_yield,
                put_option=put_option,
            )
        )

    carry = risk_free_rate - dividend_yield
    forward_price = stock_price * np.exp(carry * time_to_expiration)
    coefficient = volatility * stock_price ** (1 - beta)
    growth = 2 * carry * (1 - beta)
    # ∫_0^T δ² * e^(2(r — q)(1 — β)(T — t)) dt, with expm1 so a small carry keeps its digits.
    variance = coefficient**2 * (
        time_to_expiration
        if growth == 0
        else np.expm1(growth * time_to_expiration) / growth
    )
    degrees = 1 / (1 - beta)
    scale = (1 - beta) ** 2 * variance
    x = forward_price ** (2 * (1 - beta)) / scale
    y = strike_price ** (2 * (1 - beta)) / scale
    discount = np.exp(-risk_free_rate * time_to_expiration)

    if put_option:
        return float(
            discount
            * (
                strike_price * ncx2.sf(x, degrees, y)
                - forward_price * ncx2.cdf(y, degrees + 2, x)
            )
        )

    return float(
        discount
        * (
            forward_price * ncx2.sf(y, degrees + 2, x)
            - strike_price * ncx2.cdf(x, degrees, y)
        )
    )
