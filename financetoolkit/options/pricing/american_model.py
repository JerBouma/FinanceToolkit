"""American Option Model"""

__docformat__ = "google"

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.pricing.black_scholes_model import (
    MAXIMUM_IMPLIED_VOLATILITY,
    get_black_scholes,
)

# The lowest volatility the American implied volatility is searched from. Below it the
# quadratic approximations raise the exponent λ to powers that no longer fit a float, and
# a listed option is not quoted at such a volatility.
MINIMUM_AMERICAN_IMPLIED_VOLATILITY = 1e-3

# The Gauss-Legendre points of the early exercise premium integral of QD+, on the square
# root of the time from now, where the integrand is smooth.
_QD_PLUS_POINTS = 64
_QD_PLUS_NODES, _QD_PLUS_WEIGHTS = np.polynomial.legendre.leggauss(_QD_PLUS_POINTS)

# The QD+ exercise boundary is solved on the logarithm of boundary / strike over an interval
# of 60 natural log units: 12 halvings narrow it to 0.015, after which at most 50 steps of the
# Illinois method take it to 1e-13.
_BOUNDARY_BISECTIONS = 12
_BOUNDARY_ITERATIONS = 50
_BOUNDARY_TOLERANCE = 1e-13

# The QD fixed point boundary is held at 25 Chebyshev-Lobatto nodes in √τ and the iteration
# stops once the boundary changes less than 1e-12 relative (about 15 iterations, at most 100).
# Its integrals use the tanh-sinh rule of Takahasi and Mori (1974) with a step of 1/12 (77
# points): near expiration the boundary moves like √(τ * ln τ), which Gauss-Legendre points
# converge on slowly (64 of them are 3e-6 off) and the tanh-sinh points, crowded at both ends,
# within about 1e-9.
_QD_FIXED_POINT_NODES = 24
_TANH_SINH_STEP = 1 / 12
_TANH_SINH_GRID = _TANH_SINH_STEP * np.arange(-38, 39)
_QD_FIXED_POINT_ROOTS = np.tanh(0.5 * np.pi * np.sinh(_TANH_SINH_GRID))
_QD_FIXED_POINT_WEIGHTS = (
    _TANH_SINH_STEP
    * 0.5
    * np.pi
    * np.cosh(_TANH_SINH_GRID)
    / np.cosh(0.5 * np.pi * np.sinh(_TANH_SINH_GRID)) ** 2
)
_QD_FIXED_POINT_ITERATIONS = 100
_QD_FIXED_POINT_TOLERANCE = 1e-12


def _european_and_derivatives(
    stock_price,
    strike_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_expiration,
    sign: int,
):
    """
    Helper function that returns the generalized Black-Scholes price of a European option
    together with N(φ * d1) and the derivative of the price to the time to expiration, the
    three terms the quadratic approximations need at a candidate exercise boundary.

    - V = φ * (S * e^(—qT) * N(φ * d1) — K * e^(—rT) * N(φ * d2))
    - ∂V/∂T = S * e^(—qT) * n(d1) * σ / (2√T) — φ * q * S * e^(—qT) * N(φ * d1) +
      φ * r * K * e^(—rT) * N(φ * d2)

    With φ = 1 for a call and —1 for a put.

    Args:
        stock_price (float | np.ndarray): The stock price(s).
        strike_price (float): The strike price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        dividend_yield (float): The continuous dividend yield.
        volatility (float): The volatility.
        time_to_expiration (float | np.ndarray): The time(s) to expiration in years.
        sign (int): φ, 1 for a call and —1 for a put.

    Returns:
        tuple: The European price, N(φ * d1) and ∂V/∂T.
    """
    root_time = np.sqrt(time_to_expiration)
    d1 = (
        np.log(stock_price / strike_price)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiration
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time

    stock_discount = stock_price * np.exp(-dividend_yield * time_to_expiration)
    strike_discount = strike_price * np.exp(-risk_free_rate * time_to_expiration)
    probability_d1 = ndtr(sign * d1)
    probability_d2 = ndtr(sign * d2)

    value = sign * (stock_discount * probability_d1 - strike_discount * probability_d2)
    time_derivative = (
        # n(d1), the standard normal density, written out as it is evaluated on scalars.
        stock_discount
        * np.exp(-0.5 * d1**2)
        / np.sqrt(2 * np.pi)
        * volatility
        / (2 * root_time)
        - sign * dividend_yield * stock_discount * probability_d1
        + sign * risk_free_rate * strike_discount * probability_d2
    )

    return value, probability_d1, time_derivative


def _early_exercise_is_optimal(
    risk_free_rate: float, dividend_yield: float, put_option: bool
) -> bool:
    """
    Whether exercising before expiration can be optimal: for a call only when the dividend
    yield exceeds the risk-free rate's share of the carry (q > 0 with b = r — q < r), for a
    put only with a positive risk-free rate. Otherwise the American option is worth its
    European value (floored at the exercise value).
    """
    if put_option:
        return risk_free_rate > 0
    return dividend_yield > 0


def _quadratic_exponent(
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_expiration: float,
    sign: int,
) -> tuple[float, float, float, float]:
    """
    Helper function with the exponent λ of the quadratic approximation of Barone-Adesi and
    Whaley (1987), as written by Ju and Zhong (1999), and its derivative to h:

    - α = 2r / σ²,  β = 2(r — q) / σ²,  h = 1 — e^(—rT)
    - λ = [—(β — 1) + φ * √((β — 1)² + 4α / h)] / 2
    - λ' = —φ * α / (h² * √((β — 1)² + 4α / h))

    Returns:
        tuple[float, float, float, float]: λ, λ', α and β.
    """
    alpha = 2 * risk_free_rate / volatility**2
    beta = 2 * (risk_free_rate - dividend_yield) / volatility**2
    h = -np.expm1(-risk_free_rate * time_to_expiration)
    root = np.sqrt((beta - 1) ** 2 + 4 * alpha / h)
    exponent = (-(beta - 1) + sign * root) / 2
    exponent_derivative = -sign * alpha / (h**2 * root)

    return exponent, exponent_derivative, alpha, beta


def _quadratic_critical_price(
    strike_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_expiration: float,
    sign: int,
    exponent: float,
) -> float:
    """
    Helper function that solves the critical stock price S* of Barone-Adesi and Whaley
    (1987), at which the approximation meets the exercise value with the same slope:

    - φ * (S* — K) = V_E(S*) + φ * (1 — e^(—qT) * N(φ * d1(S*))) * S* / λ

    The root lies above the strike for a call and below it for a put, and is found with
    Brent's method on that side of the strike, the bracket of a call widened until the
    equation changes sign.

    Returns:
        float: The critical stock price S*.
    """

    def objective(price: float) -> float:
        value, probability_d1, _ = _european_and_derivatives(
            price,
            strike_price,
            risk_free_rate,
            dividend_yield,
            volatility,
            time_to_expiration,
            sign,
        )
        return float(
            sign * (price - strike_price)
            - value
            - sign
            * (1 - np.exp(-dividend_yield * time_to_expiration) * probability_d1)
            * price
            / exponent
        )

    if sign == -1:
        return brentq(
            objective,
            strike_price * 1e-12,
            strike_price,
            xtol=1e-12 * strike_price,
            maxiter=500,
        )

    upper = 2 * strike_price
    while objective(upper) < 0:
        upper *= 2

    return brentq(
        objective, strike_price, upper, xtol=1e-12 * strike_price, maxiter=500
    )


def _validate(
    stock_price,
    strike_price,
    risk_free_rate,
    volatility,
    time_to_expiration,
    dividend_yield,
) -> None:
    """Rejects inputs that are not numbers and a volatility that is not positive."""
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    # The approximations divide by the variance, which a constant price does not have.
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, received {volatility!r}.")


def _quadratic_approximation(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
    put_option: bool,
    ju_zhong: bool,
) -> float:
    """
    Helper function with the Barone-Adesi-Whaley price and, with ju_zhong, the correction of
    Ju and Zhong (1999), as they share the exponent and the critical stock price.
    """
    sign = -1 if put_option else 1
    exercise_value = max(sign * (stock_price - strike_price), 0.0)

    if time_to_expiration <= 0:
        return exercise_value

    european_value = float(
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

    if not _early_exercise_is_optimal(risk_free_rate, dividend_yield, put_option):
        return max(european_value, exercise_value)

    if risk_free_rate <= 0:
        # A call on a stock with a dividend yield above a non-positive rate: the exponent
        # divides by h = 1 — e^(—rT), so the call is priced as the put it equals under the
        # put-call symmetry of McDonald and Schroder (1998), with S and K and r and q
        # swapped, which has a positive rate, and floored at the European call like every
        # other case.  # noqa: E501
        return max(
            european_value,
            _quadratic_approximation(
                stock_price=strike_price,
                strike_price=stock_price,
                risk_free_rate=dividend_yield,
                volatility=volatility,
                time_to_expiration=time_to_expiration,
                dividend_yield=risk_free_rate,
                put_option=True,
                ju_zhong=ju_zhong,
            ),
        )

    exponent, exponent_derivative, alpha, beta = _quadratic_exponent(
        risk_free_rate, dividend_yield, volatility, time_to_expiration, sign
    )
    critical_price = _quadratic_critical_price(
        strike_price,
        risk_free_rate,
        dividend_yield,
        volatility,
        time_to_expiration,
        sign,
        exponent,
    )

    if sign * (stock_price - critical_price) >= 0:
        # Beyond the critical price the option is exercised right away.
        return exercise_value

    critical_value, critical_probability, critical_time_derivative = (
        _european_and_derivatives(
            critical_price,
            strike_price,
            risk_free_rate,
            dividend_yield,
            volatility,
            time_to_expiration,
            sign,
        )
    )

    if not ju_zhong:
        # A = φ * (1 — e^(—qT) * N(φ * d1(S*))) * S* / λ, the early exercise premium at S*.
        premium = (
            sign
            * (1 - np.exp(-dividend_yield * time_to_expiration) * critical_probability)
            * critical_price
            / exponent
        )
        return float(
            max(
                european_value + premium * (stock_price / critical_price) ** exponent,
                exercise_value,
            )
        )

    h = -np.expm1(-risk_free_rate * time_to_expiration)
    # The early exercise premium at S*, hA = φ * (S* — K) — V_E(S*).
    premium = sign * (critical_price - strike_price) - critical_value
    # ∂V_E/∂h = ∂V_E/∂T / (dh/dT), with dh/dT = r * e^(—rT) = r * (1 — h).
    european_h_derivative = critical_time_derivative / (risk_free_rate * (1 - h))
    denominator = 2 * exponent + beta - 1

    b_term = (1 - h) * alpha * exponent_derivative / (2 * denominator)
    c_term = (
        -(1 - h)
        * alpha
        / denominator
        * (european_h_derivative / premium + 1 / h + exponent_derivative / denominator)
    )
    log_ratio = np.log(stock_price / critical_price)
    chi = b_term * log_ratio**2 + c_term * log_ratio

    return float(
        max(
            european_value
            + premium * (stock_price / critical_price) ** exponent / (1 - chi),
            exercise_value,
        )
    )


def get_barone_adesi_whaley(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the price of an American option with the quadratic approximation of
    Barone-Adesi and Whaley (1987).

    The early exercise premium satisfies the Black-Scholes partial differential equation, as
    the European price does. Barone-Adesi and Whaley write the premium as h(T) * f(S, h) with
    h = 1 — e^(—rT) and drop the term with ∂f/∂h, which turns the equation into an ordinary
    one that is solved by a power of S. The power is matched to the exercise value at the
    critical stock price S*, found numerically:

    - α = 2r / σ²,  β = 2(r — q) / σ²,  h = 1 — e^(—rT)
    - λ = [—(β — 1) + φ * √((β — 1)² + 4α / h)] / 2, with φ = 1 for a call and —1 for a put
    - φ * (S* — K) = V_E(S*) + φ * (1 — e^(—qT) * N(φ * d1(S*))) * S* / λ
    - A = φ * (1 — e^(—qT) * N(φ * d1(S*))) * S* / λ
    - V = V_E(S) + A * (S / S*)^λ when φ * (S* — S) > 0, otherwise φ * (S — K)

    Where V_E is the generalized Black-Scholes price of the European option with the same
    inputs and d1(S*) its d1 at the stock price S*.

    A call on a stock without dividends (q ≤ 0) is never exercised early, and neither is a
    put at a non-positive rate; these are worth their European price, floored at the
    exercise value. A call at a non-positive rate with a positive dividend yield is priced
    as the put it equals under the put-call symmetry of McDonald and Schroder (1998).

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option, in years.
            At zero the option is worth its exercise value.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float: The Barone-Adesi-Whaley approximation of the American option value.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if volatility is not positive.

    Notes:
        Reference: Barone-Adesi, G., & Whaley, R. E. (1987). "Efficient Analytic
        Approximation of American Option Values." The Journal of Finance, 42(2), 301-320.
        Referenced against QuantLib's BaroneAdesiWhaleyApproximationEngine
        (https://www.quantlib.org) and FinancePy's black_scholes_analytic.baw_value
        (https://github.com/domokane/FinancePy). For S = K = 100, r = 5%, q = 3%, σ = 25%
        and T = 1 the put is 8.9030, where the exact American value is 8.8827.
    """
    _validate(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    )

    return _quadratic_approximation(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        put_option=put_option,
        ju_zhong=False,
    )


def get_ju_zhong(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the price of an American option with the approximation of Ju and Zhong
    (1999), which corrects the Barone-Adesi-Whaley approximation for the term it drops.

    Barone-Adesi and Whaley (1987) leave the derivative of the early exercise premium to
    h = 1 — e^(—rT) out of the partial differential equation. Ju and Zhong keep it to first
    order, which divides the premium by 1 — χ(S), a quadratic in ln(S / S*), with S*, λ and
    the premium hA at S* as in Barone-Adesi-Whaley:

    - α = 2r / σ²,  β = 2(r — q) / σ²,  h = 1 — e^(—rT)
    - λ = [—(β — 1) + φ * √((β — 1)² + 4α / h)] / 2, with φ = 1 for a call and —1 for a put
    - λ' = —φ * α / (h² * √((β — 1)² + 4α / h))
    - hA = φ * (S* — K) — V_E(S*)
    - b = (1 — h) * α * λ' / (2 * (2λ + β — 1))
    - c = —(1 — h) * α / (2λ + β — 1) * [∂V_E/∂h(S*) / hA + 1 / h + λ' / (2λ + β — 1)]
    - χ(S) = b * (ln(S / S*))² + c * ln(S / S*)
    - V = V_E(S) + hA * (S / S*)^λ / (1 — χ(S)) when φ * (S* — S) > 0, otherwise φ * (S — K)

    Where V_E is the generalized Black-Scholes price of the European option and
    ∂V_E/∂h = (∂V_E/∂T) / (r * e^(—rT)) its derivative to h at S*.

    A call on a stock without dividends (q ≤ 0) is never exercised early, and neither is a
    put at a non-positive rate; these are worth their European price, floored at the
    exercise value. A call at a non-positive rate with a positive dividend yield is priced
    as the put it equals under the put-call symmetry of McDonald and Schroder (1998).

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option, in years.
            At zero the option is worth its exercise value.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float: The Ju-Zhong approximation of the American option value.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if volatility is not positive.

    Notes:
        Reference: Ju, N., & Zhong, R. (1999). "An Approximate Formula for Pricing
        American Options." The Journal of Derivatives, 7(2), 31-40. Referenced against
        QuantLib's JuQuadraticApproximationEngine (https://www.quantlib.org). For S = K =
        100, r = 5%, q = 3%, σ = 25% and T = 1 the put is 8.8573, where the exact American
        value is 8.8827.
    """
    _validate(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    )

    return _quadratic_approximation(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        put_option=put_option,
        ju_zhong=True,
    )


@np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
def _qd_plus_put_boundary(
    strike_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    times: np.ndarray,
) -> np.ndarray:
    """
    Helper function with the early exercise boundary B(τ) of an American put at the times
    to expiration τ, from the QD+ equation of Li (2010), solved for every τ at once.

    The Ju-Zhong form of the premium, hA * (S / B)^λ / (1 — χ(S)), meets the exercise value
    K — S with slope —1 at the boundary when:

    - (1 — e^(—qτ) * N(—d1(B))) * B + (λ + c0) * (K — B — p(B)) = 0

    With λ, λ' and c0 = c of the Ju-Zhong approximation evaluated at B (so through
    hA = K — B — p(B) and ∂p/∂h at B). The boundary lies below K * min(1, r / q), its value
    at expiration, and the root is found by bisection on ln(B / K) below that limit.

    Args:
        strike_price (float): The strike price.
        risk_free_rate (float): The positive continuously compounded risk-free rate.
        dividend_yield (float): The continuous dividend yield.
        volatility (float): The volatility.
        times (np.ndarray): The times to expiration τ, all positive.

    Returns:
        np.ndarray: The boundary B(τ) at each time.
    """
    exponent, exponent_derivative, alpha, beta = _quadratic_exponent(
        risk_free_rate, dividend_yield, volatility, times, -1
    )
    h = -np.expm1(-risk_free_rate * times)
    denominator = 2 * exponent + beta - 1

    def objective(boundary: np.ndarray) -> np.ndarray:
        value, probability_d1, time_derivative = _european_and_derivatives(
            boundary,
            strike_price,
            risk_free_rate,
            dividend_yield,
            volatility,
            times,
            -1,
        )
        premium = strike_price - boundary - value
        european_h_derivative = time_derivative / (risk_free_rate * (1 - h))
        c_zero = (
            -(1 - h)
            * alpha
            / denominator
            * (
                european_h_derivative / premium
                + 1 / h
                + exponent_derivative / denominator
            )
        )
        return (1 - np.exp(-dividend_yield * times) * probability_d1) * boundary + (
            exponent + c_zero
        ) * premium

    limit = strike_price * (
        min(1.0, risk_free_rate / dividend_yield) if dividend_yield > 0 else 1.0
    )
    # Just below the limit the premium K — B — p(B) is positive and tends to zero, where the
    # objective is positive; far below it the objective is negative.
    upper = np.full_like(times, np.log(limit / strike_price) - 1e-12)
    lower = np.full_like(times, np.log(limit / strike_price) - 60.0)

    # Bisection narrows the bracket to where the objective is smooth, after which the
    # Illinois variant of the false position method (Dowell and Jarratt, 1971) converges
    # superlinearly while keeping the root bracketed.
    for _ in range(_BOUNDARY_BISECTIONS):
        middle = 0.5 * (lower + upper)
        positive = objective(strike_price * np.exp(middle)) > 0
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)

    lower_value = objective(strike_price * np.exp(lower))
    upper_value = objective(strike_price * np.exp(upper))
    side = np.zeros_like(times)

    for _ in range(_BOUNDARY_ITERATIONS):
        middle = (lower * upper_value - upper * lower_value) / (
            upper_value - lower_value
        )
        middle = np.where(np.isfinite(middle), middle, 0.5 * (lower + upper))
        value = objective(strike_price * np.exp(middle))
        positive = value > 0

        # The end that stays put a second time has its value halved (Illinois).
        upper_value = np.where(
            positive, value, np.where(side == -1, upper_value / 2, upper_value)
        )
        lower_value = np.where(
            positive, np.where(side == 1, lower_value / 2, lower_value), value
        )
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)
        side = np.where(positive, 1, -1)

        if np.all(upper - lower < _BOUNDARY_TOLERANCE) or np.all(
            np.abs(value) < _BOUNDARY_TOLERANCE * strike_price
        ):
            break

    return strike_price * np.exp(middle)


@np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
def _qd_plus_put(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
) -> float:
    """
    Helper function with the QD+ price of an American put with a positive risk-free rate:
    the European put plus the early exercise premium integrated over the QD+ boundary.

    - V = p(S) + ∫_0^T [r * K * e^(—rt) * N(—d2(t, S / B(T — t))) —
      q * S * e^(—qt) * N(—d1(t, S / B(T — t)))] dt

    With d1(t, x) = [ln(x) + (r — q + σ² / 2) * t] / (σ√t) and d2 = d1 — σ√t. The integral
    is taken over u = √t with Gauss-Legendre quadrature, as the integrand changes fastest
    near t = 0.
    """
    roots = 0.5 * np.sqrt(time_to_expiration) * (_QD_PLUS_NODES + 1)
    weights = 0.5 * np.sqrt(time_to_expiration) * _QD_PLUS_WEIGHTS
    times_from_now = roots**2

    # The boundary at the quadrature times and, last, today's boundary B(T): at or below it
    # the put is exercised right away.
    boundary = _qd_plus_put_boundary(
        strike_price,
        risk_free_rate,
        dividend_yield,
        volatility,
        np.append(time_to_expiration - times_from_now, time_to_expiration),
    )
    if stock_price <= boundary[-1]:
        return strike_price - stock_price
    boundary = boundary[:-1]

    spread = volatility * roots
    d1 = (
        np.log(stock_price / boundary)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * times_from_now
    ) / spread
    d2 = d1 - spread

    integrand = risk_free_rate * strike_price * np.exp(
        -risk_free_rate * times_from_now
    ) * ndtr(-d2) - dividend_yield * stock_price * np.exp(
        -dividend_yield * times_from_now
    ) * ndtr(
        -d1
    )
    # dt = 2u du.
    premium = float(np.sum(weights * integrand * 2 * roots))

    european_value = float(
        get_black_scholes(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=dividend_yield,
            put_option=True,
        )
    )

    return max(european_value + premium, strike_price - stock_price)


def get_qd_plus(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the price of an American option with the QD+ exercise boundary of Li (2010)
    and the integral representation of the early exercise premium, as Andersen, Lake and
    Offengenden (2016) combine them.

    The American put is its European value plus the value of exercising early, an integral
    over the early exercise boundary B (Kim, 1990; Jacka, 1991; Carr, Jarrow and Myneni,
    1992). QD+ approximates the boundary at every time to expiration τ with the
    smooth-pasting condition of the Ju-Zhong form of the premium, which is more accurate
    than the quadratic approximations that value the premium in closed form:

    - α = 2r / σ²,  β = 2(r — q) / σ²,  h(τ) = 1 — e^(—rτ)
    - λ = [—(β — 1) — √((β — 1)² + 4α / h)] / 2,  λ' = α / (h² * √((β — 1)² + 4α / h))
    - c0 = —(1 — h) * α / (2λ + β — 1) * [∂p/∂h(B) / (K — B — p(B)) + 1 / h + λ' / (2λ + β — 1)]
    - B(τ) solves (1 — e^(—qτ) * N(—d1(τ, B / K))) * B + (λ + c0) * (K — B — p(τ, B)) = 0
    - P = p(S) + ∫_0^T [r * K * e^(—rt) * N(—d2(t, S / B(T — t))) —
      q * S * e^(—qt) * N(—d1(t, S / B(T — t)))] dt when S > B(T), otherwise K — S
    - C(S, K, r, q) = P(K, S, q, r), the put-call symmetry of McDonald and Schroder (1998)

    Where p is the European put, d1(t, x) = [ln(x) + (r — q + σ² / 2) * t] / (σ√t),
    d2 = d1 — σ√t and ∂p/∂h = (∂p/∂τ) / (r * e^(—rτ)). The boundary is solved at each of the
    64 Gauss-Legendre points of the integral, taken over √t, so it is not interpolated.

    A put at a non-positive rate is never exercised early and a call without dividends
    neither; these are worth their European price, floored at the exercise value.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option, in years.
            At zero the option is worth its exercise value.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float: The QD+ value of the American option.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if volatility is not positive.

    Notes:
        References: Li, M. (2010). "Analytical Approximations for the Critical Stock Prices
        of American Options: A Performance Comparison." Review of Derivatives Research,
        13(1), 75-99; Andersen, L., Lake, M., & Offengenden, D. (2016). "High-Performance
        American Option Pricing." Journal of Computational Finance, 20(1), 39-87; Kim, I. J.
        (1990). "The Analytic Valuation of American Options." The Review of Financial
        Studies, 3(4), 547-572. Referenced against QuantLib's QdPlusAmericanEngine
        (https://www.quantlib.org), which interpolates the boundary between eight
        Chebyshev nodes and values a stock at or beyond the boundary with the integral as
        well, where this function returns the exercise value. For S = K = 100, r = 5%,
        q = 3%, σ = 25% and T = 1 the put is 8.8886, where the exact American value is
        8.8827; across typical inputs QD+ lies within 0.3% of the exact value.
    """
    _validate(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    )

    exercise_value = max(
        (strike_price - stock_price) if put_option else (stock_price - strike_price),
        0.0,
    )
    if time_to_expiration <= 0:
        return exercise_value

    if not put_option:
        # A call is the put with S and K and r and q swapped (McDonald and Schroder, 1998).
        return get_qd_plus(
            stock_price=strike_price,
            strike_price=stock_price,
            risk_free_rate=dividend_yield,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=risk_free_rate,
            put_option=True,
        )

    if risk_free_rate <= 0:
        european_value = float(
            get_black_scholes(
                stock_price=stock_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_expiration=time_to_expiration,
                dividend_yield=dividend_yield,
                put_option=True,
            )
        )
        return max(european_value, exercise_value)

    return _qd_plus_put(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    )


def _qd_fixed_point_interpolate(
    nodes: np.ndarray, values: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Helper function with the polynomial through values at the Chebyshev-Lobatto nodes
    z_j = —cos(jπ / n), evaluated at points in [—1, 1] with the barycentric formula of the
    second kind (weights (—1)^j, halved at both ends), which is stable at every degree.
    """
    weights = (-1.0) ** np.arange(nodes.size)
    weights[[0, -1]] *= 0.5
    difference = points[..., None] - nodes
    on_node = difference == 0
    terms = weights / np.where(on_node, 1.0, difference)
    interpolated = (terms * values).sum(-1) / terms.sum(-1)
    # A point on a node takes that node's value, where the formula divides by zero.
    return np.where(on_node.any(-1), (on_node * values).sum(-1), interpolated)


@np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
def _qd_fixed_point_put(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
) -> float:
    """
    Helper function with the American put of a positive risk-free rate from the exercise
    boundary that solves the integral equation of Kim (1990) as a fixed point, the method
    of Andersen, Lake and Offengenden (2016).

    The boundary is held at the Chebyshev-Lobatto nodes of z = 2√(τ / T) — 1 as
    H = ln(B / X)², with X = K * min(1, r / q) its limit at expiration, and interpolated in
    between. Starting from the QD+ boundary, each iteration sets at every node:

    - B(τ) = K * e^(—(r — q)τ) * N(τ, B) / D(τ, B)
    - N = n(d—(τ, B / K)) / (σ√τ) + r * ∫_0^τ e^(ru) * n(d—(τ — u, B(τ) / B(u))) / (σ√(τ — u)) du
    - D = n(d+(τ, B / K)) / (σ√τ) + N(d+(τ, B / K)) + q * ∫_0^τ e^(qu) *
      [n(d+(τ — u, B(τ) / B(u))) / (σ√(τ — u)) + N(d+(τ — u, B(τ) / B(u)))] du

    With d±(t, x) = [ln(x) + (r — q ± σ² / 2) * t] / (σ√t), the equation the paper calls
    FP-A. When FP-A does not converge (it oscillates at a low volatility or a negative
    dividend yield), the iteration starts again from the QD+ boundary with FP-B, the same
    fixed point written with the normal distribution:

    - N = N(d—(τ, B / K)) + r * ∫_0^τ e^(ru) * N(d—(τ — u, B(τ) / B(u))) du
    - D = N(d+(τ, B / K)) + q * ∫_0^τ e^(qu) * N(d+(τ — u, B(τ) / B(u))) du

    The integrals are taken over s = √(τ — u), which removes the 1 / √(τ — u), with the
    tanh-sinh rule, and the iteration stops once the boundary changes less than 1e-12
    relative between two iterations (at most 100 iterations per equation). A boundary that
    converges with neither returns NaN.
    """
    limit = strike_price * (
        min(1.0, risk_free_rate / dividend_yield) if dividend_yield > 0 else 1.0
    )
    nodes = -np.cos(
        np.pi * np.arange(_QD_FIXED_POINT_NODES + 1) / _QD_FIXED_POINT_NODES
    )
    times = time_to_expiration * (1 + nodes[1:]) ** 2 / 4

    def boundary_at(log_distance: np.ndarray, time: np.ndarray) -> np.ndarray:
        # A time next to zero can round to just below it, where the square root has no value.
        squared = _qd_fixed_point_interpolate(
            nodes,
            log_distance,
            2 * np.sqrt(np.maximum(time, 0) / time_to_expiration) - 1,
        )
        return limit * np.exp(-np.sqrt(np.maximum(squared, 0)))

    def d_plus(time, ratio):
        return (
            np.log(ratio)
            + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time
        ) / (volatility * np.sqrt(time))

    def density(x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    starting_boundary = np.minimum(
        _qd_plus_put_boundary(
            strike_price, risk_free_rate, dividend_yield, volatility, times
        ),
        limit,
    )

    # u = τ — s², so du = 2s ds and the 1 / (σ√(τ — u)) = 1 / (σs) of FP-A cancels against it.
    roots = 0.5 * np.sqrt(times)[:, None] * (_QD_FIXED_POINT_ROOTS + 1)
    weights = 0.5 * np.sqrt(times)[:, None] * _QD_FIXED_POINT_WEIGHTS
    earlier_times = times[:, None] - roots**2
    root_times = np.sqrt(times)

    def iterate(equation: str) -> tuple[np.ndarray, bool]:
        boundary = starting_boundary
        for _ in range(_QD_FIXED_POINT_ITERATIONS):
            log_distance = np.append(0.0, np.log(boundary / limit) ** 2)
            earlier_boundary = boundary_at(log_distance, earlier_times)

            plus = d_plus(roots**2, boundary[:, None] / earlier_boundary)
            minus = plus - volatility * roots
            plus_today = d_plus(times, boundary / strike_price)
            minus_today = plus_today - volatility * root_times
            rate_weights = weights * np.exp(risk_free_rate * earlier_times)
            yield_weights = weights * np.exp(dividend_yield * earlier_times)

            if equation == "FP-A":
                numerator = density(minus_today) / (
                    volatility * root_times
                ) + risk_free_rate * np.sum(
                    rate_weights * 2 * density(minus) / volatility, axis=1
                )
                denominator = (
                    density(plus_today) / (volatility * root_times)
                    + ndtr(plus_today)
                    + dividend_yield
                    * np.sum(
                        yield_weights
                        * (2 * density(plus) / volatility + 2 * roots * ndtr(plus)),
                        axis=1,
                    )
                )
            else:
                numerator = ndtr(minus_today) + risk_free_rate * np.sum(
                    rate_weights * 2 * roots * ndtr(minus), axis=1
                )
                denominator = ndtr(plus_today) + dividend_yield * np.sum(
                    yield_weights * 2 * roots * ndtr(plus), axis=1
                )

            updated = np.minimum(
                strike_price
                * np.exp(-(risk_free_rate - dividend_yield) * times)
                * numerator
                / denominator,
                limit,
            )
            change = np.max(np.abs(updated / boundary - 1))
            boundary = updated

            if not np.all(np.isfinite(boundary) & (boundary > 0)):
                return boundary, False
            if change < _QD_FIXED_POINT_TOLERANCE:
                return boundary, True

        return boundary, False

    boundary, converged = iterate("FP-A")
    if not converged:
        # FP-A converges fastest, but it oscillates and diverges in some regimes (a low
        # volatility, a negative dividend yield). FP-B, the same fixed point, then converges.
        boundary, converged = iterate("FP-B")
    if not converged:
        return np.nan

    if stock_price <= boundary[-1]:
        return strike_price - stock_price

    log_distance = np.append(0.0, np.log(boundary / limit) ** 2)
    roots = 0.5 * np.sqrt(time_to_expiration) * (_QD_FIXED_POINT_ROOTS + 1)
    weights = 0.5 * np.sqrt(time_to_expiration) * _QD_FIXED_POINT_WEIGHTS
    times_from_now = roots**2
    plus = d_plus(
        times_from_now,
        stock_price / boundary_at(log_distance, time_to_expiration - times_from_now),
    )
    minus = plus - volatility * roots
    integrand = risk_free_rate * strike_price * np.exp(
        -risk_free_rate * times_from_now
    ) * ndtr(-minus) - dividend_yield * stock_price * np.exp(
        -dividend_yield * times_from_now
    ) * ndtr(
        -plus
    )
    # dt = 2u du, with the integral taken over u = √t.
    premium = float(np.sum(weights * integrand * 2 * roots))

    european_value = float(
        get_black_scholes(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=dividend_yield,
            put_option=True,
        )
    )

    return max(european_value + premium, strike_price - stock_price)


def get_qd_fixed_point(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the price of an American option by solving the early exercise boundary as the
    fixed point of its integral equation, the method of Andersen, Lake and Offengenden
    (2016). Unlike the approximations (QD+, Barone-Adesi-Whaley, Ju-Zhong), this converges
    to the exact American value, within about 1e-6 of it and 1e-8 for most inputs.

    The American put is its European value plus the value of exercising early, an integral
    over the early exercise boundary B (Kim, 1990; Jacka, 1991; Carr, Jarrow and Myneni,
    1992). At the boundary the put is worth its exercise value K — B, which gives an
    integral equation in B that Andersen, Lake and Offengenden rewrite as a fixed point
    (their FP-A equation):

    - B(τ) = K * e^(—(r — q)τ) * N(τ, B) / D(τ, B)
    - N(τ, B) = n(d—(τ, B(τ) / K)) / (σ√τ) + r * ∫_0^τ e^(ru) * n(d—(τ — u, B(τ) / B(u))) /
      (σ√(τ — u)) du
    - D(τ, B) = n(d+(τ, B(τ) / K)) / (σ√τ) + N(d+(τ, B(τ) / K)) + q * ∫_0^τ e^(qu) *
      [n(d+(τ — u, B(τ) / B(u))) / (σ√(τ — u)) + N(d+(τ — u, B(τ) / B(u)))] du
    - P = p(S) + ∫_0^T [r * K * e^(—rt) * N(—d—(t, S / B(T — t))) —
      q * S * e^(—qt) * N(—d+(t, S / B(T — t)))] dt when S > B(T), otherwise K — S
    - C(S, K, r, q) = P(K, S, q, r), the put-call symmetry of McDonald and Schroder (1998)

    Where d±(t, x) = [ln(x) + (r — q ± σ² / 2) * t] / (σ√t), n and N are the standard
    normal density and distribution and p is the European put. The boundary is held at 25
    Chebyshev-Lobatto nodes in √τ, as ln(B / X)² with X = K * min(1, r / q) its value at
    expiration, starts from the QD+ boundary and is iterated until it changes less than
    1e-12; the integrals use the 77 point tanh-sinh rule (Takahasi and Mori, 1974) over the
    square root of the time, whose points crowd at the ends where the boundary near
    expiration changes fastest.

    A put at a non-positive rate is never exercised early and a call without dividends
    neither; these are worth their European price, floored at the exercise value.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option, in years.
            At zero the option is worth its exercise value.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float: The value of the American option.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if volatility is not positive.

    Notes:
        References: Andersen, L., Lake, M., & Offengenden, D. (2016). "High-Performance
        American Option Pricing." Journal of Computational Finance, 20(1), 39-87; Kim, I. J.
        (1990). "The Analytic Valuation of American Options." The Review of Financial
        Studies, 3(4), 547-572; Li, M. (2010) for the QD+ starting boundary. Referenced
        against QuantLib's QdFpAmericanEngine with its high precision scheme
        (https://www.quantlib.org) and a Crank-Nicolson finite difference grid extrapolated
        to zero step size. For S = K = 100, r = 5%, q = 3%, σ = 25% and T = 1 the put is
        8.8827, where QD+ gives 8.8886.
    """
    _validate(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    )

    exercise_value = max(
        (strike_price - stock_price) if put_option else (stock_price - strike_price),
        0.0,
    )
    if time_to_expiration <= 0:
        return exercise_value

    if not put_option:
        # A call is the put with S and K and r and q swapped (McDonald and Schroder, 1998).
        return get_qd_fixed_point(
            stock_price=strike_price,
            strike_price=stock_price,
            risk_free_rate=dividend_yield,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=risk_free_rate,
            put_option=True,
        )

    if risk_free_rate <= 0:
        european_value = float(
            get_black_scholes(
                stock_price=stock_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_expiration=time_to_expiration,
                dividend_yield=dividend_yield,
                put_option=True,
            )
        )
        return max(european_value, exercise_value)

    return _qd_fixed_point_put(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    )


def get_american_option(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: str = "qd_plus",
) -> float:
    """
    Calculate the price of an American option with one of the approximations: QD+
    ("qd_plus", see get_qd_plus), Barone-Adesi-Whaley ("barone_adesi_whaley", see
    get_barone_adesi_whaley) or Ju-Zhong ("ju_zhong", see get_ju_zhong), or with the
    exercise boundary solved as a fixed point ("qd_fixed_point", see get_qd_fixed_point).

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option, in years.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.
        method (str): The method, "qd_plus", "barone_adesi_whaley", "ju_zhong" or
            "qd_fixed_point". Defaults to "qd_plus".

    Returns:
        float: The value of the American option.

    Raises:
        ValueError: if method is not one of the approximations or volatility is not
            positive.
    """
    functions = {
        "qd_plus": get_qd_plus,
        "barone_adesi_whaley": get_barone_adesi_whaley,
        "ju_zhong": get_ju_zhong,
        "qd_fixed_point": get_qd_fixed_point,
    }
    if method not in functions:
        raise ValueError(
            f"method must be one of {', '.join(functions)}, received {method!r}."
        )
    return functions[method](
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        put_option=put_option,
    )


def get_american_implied_volatility(
    market_price: float,
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: str = "qd_plus",
) -> float:
    """
    Numerically solve for the volatility at which the American option price of
    get_american_option equals an observed market price: the root of American price(σ) —
    market price, found with Brent's method on the volatility interval [0.001, 5].

    Listed options on single stocks are American. Their price includes the right to
    exercise early, which a European (Black-Scholes) implied volatility attributes to a
    higher volatility instead: the in-the-money puts and the calls before a large dividend
    then show a European implied volatility above the American one.

    Args:
        market_price (float): The observed market price of the option.
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        time_to_expiration (float): The time to expiration of the option, in years.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.
        method (str): The American method, "qd_plus", "barone_adesi_whaley", "ju_zhong"
            or "qd_fixed_point". Defaults to "qd_plus".

    Returns:
        float: The implied volatility, or NaN when the observed price admits no solution.

    Notes:
        - The American price rises with the volatility, so the root is bracketed. A price
          at or below the exercise value, or one that the lowest volatility already
          exceeds or the highest does not reach, has no solution and returns NaN.
        - Referenced against QuantLib's QdPlusAmericanEngine and
          BaroneAdesiWhaleyApproximationEngine (https://www.quantlib.org): their prices invert
          to the volatility they were priced at.
    """
    # A strike without a usable quote has no price (NaN) and so no volatility.
    if time_to_expiration <= 0 or not market_price > 0:
        return np.nan

    exercise_value = max(
        (strike_price - stock_price) if put_option else (stock_price - strike_price),
        0.0,
    )
    if market_price <= exercise_value:
        return np.nan

    def objective(volatility: float) -> float:
        return (
            get_american_option(
                stock_price=stock_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                time_to_expiration=time_to_expiration,
                dividend_yield=dividend_yield,
                put_option=put_option,
                method=method,
            )
            - market_price
        )

    if (
        objective(MINIMUM_AMERICAN_IMPLIED_VOLATILITY) > 0
        or objective(MAXIMUM_IMPLIED_VOLATILITY) < 0
    ):
        return np.nan

    return brentq(
        objective,
        MINIMUM_AMERICAN_IMPLIED_VOLATILITY,
        MAXIMUM_IMPLIED_VOLATILITY,
        xtol=1e-10,
    )
