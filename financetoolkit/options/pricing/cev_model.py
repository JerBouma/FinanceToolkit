"""Constant Elasticity of Variance Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import ncx2

from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes

# From this many degrees of freedom (β of 0.99 and above) or from this non-centrality (a short
# expiration or a low volatility) the distribution function of the non-central chi-squared
# distribution is the approximation of Sankaran (1963). scipy's distribution sums a series
# over the non-centrality, which takes a time that grows with its square root (about 0.2
# milliseconds per value at 10^8) and loses its accuracy as the degrees of freedom grow as well
# (NaN at β = 0.9999 and one day to expiration). The approximation is within about 1e-5 per 100
# of the stock price of scipy's prices for β from 0 to 0.98 and volatilities from 10% to 100%,
# and within 1e-9 of a quadrature of the density in 40-digit arithmetic from β = 0.99.
SANKARAN_DEGREES = 100
SANKARAN_NONCENTRALITY = 1e4
# Within this distance of 1 the elasticity is priced as Black-Scholes (β = 1). The CEV price
# differs from Black-Scholes by about (1 — β) times the price's sensitivity to β, so by at
# most about 2e-5 here, while the formula itself loses digits: x and y are then ratios of
# powers that differ from 1 by (1 — β) and errors of 1e-6 or more appear below 1e-7.
BLACK_SCHOLES_BETA_TOLERANCE = 1e-6


# A zero strike gives the power of log1p(-1) = -inf, whose expm1 is the right limit of -1.
@np.errstate(divide="ignore")
def _noncentral_chi_squared(
    value: np.ndarray, degrees: float, noncentrality: np.ndarray, upper: bool
) -> np.ndarray:
    """
    Helper function with the distribution function P(χ² ≤ value) of the non-central
    chi-squared distribution or, with upper, its complement P(χ² > value): scipy's below
    SANKARAN_DEGREES degrees of freedom and SANKARAN_NONCENTRALITY, and otherwise the
    approximation of Sankaran (1963), which Schroder (1989) recommends for the CEV model with
    large arguments:

    - h = 1 — 2/3 * (k + λ)(k + 3λ) / (k + 2λ)², p = (k + 2λ) / (k + λ)², m = (h — 1)(1 — 3h)
    - P(χ² ≤ z) = N([(z / (k + λ))^h — 1 + h * p * (1 — h + (2 — h) * m * p / 2)] /
      (h * √(2p) * (1 + m * p / 2)))

    The power less one is taken as expm1(h * log1p(·)), as z lies within a few standard
    deviations √(2(k + 2λ)) of k + λ, which is huge, and the complement as N(—·), so that
    neither loses its digits.
    """
    value, noncentrality = np.broadcast_arrays(
        np.asarray(value, dtype=float), np.asarray(noncentrality, dtype=float)
    )
    mean = degrees + noncentrality
    power = (
        1
        - 2
        / 3
        * mean
        * (degrees + 3 * noncentrality)
        / (degrees + 2 * noncentrality) ** 2
    )
    ratio = (degrees + 2 * noncentrality) / mean**2
    shape = (power - 1) * (1 - 3 * power)
    standardized = (
        np.expm1(power * np.log1p((value - mean) / mean))
        + power * ratio * (1 - power + 0.5 * (2 - power) * shape * ratio)
    ) / (power * np.sqrt(2 * ratio) * (1 + 0.5 * shape * ratio))
    probability = np.array(ndtr(-standardized if upper else standardized), dtype=float)

    if degrees < SANKARAN_DEGREES:
        exact = noncentrality < SANKARAN_NONCENTRALITY
        probability[exact] = (ncx2.sf if upper else ncx2.cdf)(
            value[exact], degrees, noncentrality[exact]
        )

    return probability


def get_cev(
    stock_price: float,
    strike_price: float | pd.Series | np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float | pd.Series | np.ndarray,
    beta: float = 0.5,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float | np.ndarray:
    """
    Calculate the price of a European option in the constant elasticity of variance (CEV) model
    of Cox (1975): what a call or put is worth when the volatility rises as the stock price
    falls, with the non-central chi-squared formula of Schroder (1989).

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

    As β approaches 1 the degrees of freedom ν and the non-centralities x and y grow without
    bound (x is about 1 / ((1 — β)² * σ² * T)), where the distribution's series no longer
    converges in floating point, and a short expiration or a low volatility makes x and y
    large at any β, where the series takes long. From ν = 100 (β of 0.99 and above), or from
    a non-centrality of 10,000, χ² is therefore the approximation of Sankaran (1963), as
    Schroder (1989) recommends for large arguments. It is within about 1e-5 per 100 of the
    stock price of the exact series below β = 0.99, and within 1e-9 in probability from
    β = 0.99 on, better closer to 1, so the price runs continuously into Black-Scholes at
    β = 1. A β within 1e-6 of 1 is priced as
    Black-Scholes, which it then differs from by at most about 2e-5, while the formula itself
    loses digits there.

    Referenced against:
    - QuantLib, AnalyticCEVEngine: https://github.com/lballabio/QuantLib/blob/v1.44/ql/pricingengines/vanilla/analyticcevengine.hpp#L62

    Args:
        stock_price (float): The current stock price.
        strike_price (float | pd.Series | np.ndarray): The option's strike price, or a range
            of strike prices priced at once.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The local volatility at today's stock price, σ = δ * S^(β — 1).
        time_to_expiration (float | pd.Series | np.ndarray): The time to expiration of the
            option, in years, or a range of them of the same shape as the strike prices. At
            zero the option is worth its exercise value.
        beta (float): The elasticity β, between 0 and 1; 1 (or within 1e-6 of it) is the
            Black-Scholes model. Defaults to 0.5, the square-root process.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float | np.ndarray: The CEV value of the option, an array for a range of strike
            prices or times to expiration.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int (or, for the strike
            price and time to expiration, an np.ndarray or pd.Series).
        ValueError: if beta is outside 0 to 1, the volatility is not positive or the
            formula has no finite value.

    Notes:
        References: Cox, J. C. (1975). "Notes on Option Pricing I: Constant Elasticity of
        Variance Diffusions." Working paper, Stanford University (reprinted in The Journal
        of Portfolio Management, 23(2), 1996); Schroder, M. (1989). "Computing the Constant
        Elasticity of Variance Option Pricing Formula." The Journal of Finance, 44(1),
        211-219.
        - QuantLib's AnalyticCEVEngine takes the coefficient of the futures price, √(v / T)
          here, and agrees within 1e-12; the price also matches a Monte Carlo simulation of
          the stock with drift r — q.

        A β above 1 is not accepted: the futures price is then a strict local martingale, so
        the call and put no longer satisfy put-call parity. The put of the example below is
        above the Black-Scholes price of the same put, 4.0241: with β = 0.5 the volatility
        rises as the price falls towards the strike. The Sankaran approximation is checked
        against a quadrature of the non-central chi-squared density in 40-digit arithmetic
        (mpmath) for β from 0.5 to 0.99999 and expirations from one day to two years.

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
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        beta=beta,
        dividend_yield=dividend_yield,
    )
    validate_numeric_inputs(
        arrays=True, strike_price=strike_price, time_to_expiration=time_to_expiration
    )
    if not 0 <= beta <= 1:
        raise ValueError(f"beta must be between 0 and 1, received {beta!r}.")
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, received {volatility!r}.")

    if 1 - beta < BLACK_SCHOLES_BETA_TOLERANCE:
        black_scholes = np.asarray(
            get_black_scholes(
                stock_price=stock_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_expiration=time_to_expiration,
                dividend_yield=dividend_yield,
                put_option=put_option,
            ),
            dtype=float,
        )
        return float(black_scholes) if black_scholes.ndim == 0 else black_scholes

    strike = np.asarray(strike_price, dtype=float)
    time = np.asarray(time_to_expiration, dtype=float)
    expired = time <= 0
    # An expired option is worth its exercise value, set at the end; the formula gets a year
    # in its place so that it does not divide by a zero variance.
    time = np.where(expired, 1.0, time)

    carry = risk_free_rate - dividend_yield
    forward_price = stock_price * np.exp(carry * time)
    coefficient = volatility * stock_price ** (1 - beta)
    growth = 2 * carry * (1 - beta)
    # ∫_0^T δ² * e^(2(r — q)(1 — β)(T — t)) dt, with expm1 so a small carry keeps its digits.
    variance = coefficient**2 * (
        time if growth == 0 else np.expm1(growth * time) / growth
    )
    degrees = 1 / (1 - beta)
    scale = (1 - beta) ** 2 * variance
    x = forward_price ** (2 * (1 - beta)) / scale
    y = strike ** (2 * (1 - beta)) / scale
    discount = np.exp(-risk_free_rate * time)

    if put_option:
        value = discount * (
            strike * _noncentral_chi_squared(x, degrees, y, upper=True)
            - forward_price * _noncentral_chi_squared(y, degrees + 2, x, upper=False)
        )
        exercise_value = np.maximum(strike - stock_price, 0.0)
    else:
        value = discount * (
            forward_price * _noncentral_chi_squared(y, degrees + 2, x, upper=True)
            - strike * _noncentral_chi_squared(x, degrees, y, upper=False)
        )
        exercise_value = np.maximum(stock_price - strike, 0.0)

    value = np.where(expired, exercise_value, value)

    if not np.all(np.isfinite(value)):
        raise ValueError(
            "The non-central chi-squared distribution of the CEV formula has no finite value "
            f"for beta={beta!r} at these strikes and times to expiration."
        )

    return float(value) if value.ndim == 0 else value
