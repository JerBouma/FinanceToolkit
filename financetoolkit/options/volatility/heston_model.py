"""Heston Stochastic Volatility and Bates Models"""

__docformat__ = "google"

from typing import Literal

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from financetoolkit.options.greeks.greeks_model import get_vega
from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.options.volatility.fourier_model import (
    COS_MAXIMUM_TERMS,
    COS_TOLERANCE,
    FFT_MOMENT_ORDER,
    get_fourier_price,
)
from financetoolkit.options.volatility.levy_model import (
    get_merton_characteristic_function,
)
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

MINIMUM_QUOTES_FOR_HESTON_FIT = 5

# The vega, relative to the at-the-money vega of the expiration, below which a quote is left
# out of the calibration.
MINIMUM_RELATIVE_VEGA = 0.05

# The bounds of the calibration: (v0, kappa, theta, sigma, rho). The variances allow a
# volatility of up to 200%, the speed of mean reversion a half-life of a few weeks, and the
# correlation stays off ±1 where the variance process becomes degenerate.
HESTON_BOUNDS = (
    np.array([1e-4, 1e-3, 1e-4, 1e-2, -0.999]),
    np.array([4.0, 20.0, 4.0, 5.0, 0.999]),
)

# The scale of the soft-l1 loss of the calibration, in volatility: errors up to about one
# volatility point count in full, larger ones (stale or wide quotes) grow only linearly.
HESTON_LOSS_SCALE = 0.01

# The speed of mean reversion and volatility of variance of the starting points of the
# calibration, which start v0 and theta from the at-the-money variance and rho at -0.6: a slow
# reversion with a calm variance and a fast one with a volatile variance.
HESTON_STARTS = ((1.0, 0.4), (4.0, 1.2))

# The number of COS terms the calibration starts with (the COS method prices a whole expiry
# for the cost of one characteristic function evaluation per term), and the largest
# difference in implied volatility between those prices and the Lewis prices at the fitted
# parameters before the fit is repeated with more terms.
HESTON_CALIBRATION_TERMS = 256
HESTON_PRICER_TOLERANCE = 1e-5

# The Newton steps that invert the fitted prices to implied volatilities for the RMSE,
# starting from the market's implied volatility.
IMPLIED_VOLATILITY_ITERATIONS = 8


def get_heston_characteristic_function(
    u: np.ndarray,
    time_to_expiration: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> np.ndarray:
    """
    Calculate the characteristic function of the log return X = ln(S_T / F) in the Heston
    (1993) model: the Fourier transform of its distribution, from which the option prices
    follow, in the form of Albrecher, Mayer, Schoutens and Tistaert (2007) ("the little
    Heston trap").

    This form keeps the complex logarithm on its principal branch for every expiration, so
    no count of branch crossings is needed.

    The formulas are as follows:

    - ξ = κ — ρσiu, d = √(ξ² + σ²(iu + u²)), g = (ξ — d) / (ξ + d)
    - C = κθ / σ² * ((ξ — d)T — 2 * ln((1 — g * e^(—dT)) / (1 — g)))
    - D = (ξ — d) / σ² * (1 — e^(—dT)) / (1 — g * e^(—dT))
    - φ(u) = e^(C + D * v0)

    Where v0 is the variance today, κ the speed of mean reversion of the variance, θ its
    long-run level, σ the volatility of the variance, ρ the correlation between the stock and
    its variance and T the time to expiration in years.

    Args:
        u (np.ndarray): The arguments, real or complex.
        time_to_expiration (float): The time to expiration in years.
        v0 (float): The variance today.
        kappa (float): The speed of mean reversion κ.
        theta (float): The long-run variance θ.
        sigma (float): The volatility of the variance σ, above 0.
        rho (float): The correlation ρ between the stock and its variance.

    Returns:
        np.ndarray: E[e^(iuX)] for every u.

    As an example, φ(—i) = E[S_T / F] = 1 (the forward is the expected price) and φ(1):

    ```python
    import numpy as np

    from financetoolkit.options.volatility import heston_model

    np.round(
        heston_model.get_heston_characteristic_function(
            np.array([-1j, 1.0]), time_to_expiration=0.5, v0=0.04, kappa=2.0,
            theta=0.05, sigma=0.6, rho=-0.7,
        ),
        6,
    ).tolist()
    ```

    Which returns:

    ```
    [(1+0j), (0.988337-0.009853j)]
    ```
    """
    u = np.asarray(u, dtype=complex)
    xi = kappa - rho * sigma * 1j * u
    d = np.sqrt(xi * xi + sigma**2 * (1j * u + u * u))
    g = (xi - d) / (xi + d)
    decay = np.exp(-d * time_to_expiration)

    c = (
        kappa
        * theta
        / sigma**2
        * ((xi - d) * time_to_expiration - 2 * np.log((1 - g * decay) / (1 - g)))
    )
    d_term = (xi - d) / sigma**2 * (1 - decay) / (1 - g * decay)

    return np.exp(c + d_term * v0)


def get_heston_moment_explosion_time(
    order: float,
    kappa: float,
    sigma: float,
    rho: float,
) -> float:
    """
    Calculate the time at which the moment E[S_T^p] of the stock price explodes in the
    Heston (1993) model, following Andersen and Piterbarg (2007): the expiration from which
    the moment is infinite, while it is finite for every expiration before it.

    Moments of an order between 0 and 1 never explode, and neither does a moment of a higher
    order when the variance reverts fast enough against the correlation.

    The moment is E[S_T^p] = F^p * e^(A(T) + B(T) * v0), where B solves the Riccati equation
    dB / dt = σ²B² / 2 + (ρσp — κ)B + p(p — 1) / 2 from B(0) = 0, and the moment explodes when B
    does. The time follows from integrating that equation from 0 to infinity.

    The formulas are as follows:

    - b = ρσp — κ, D = b² — σ²p(p — 1)
    - T* = ∞ if p(p — 1) ≤ 0, or if D ≥ 0 and b < 0
    - T* = 1 / √D * ln((b + √D) / (b — √D)) if D > 0 and b ≥ 0
    - T* = 2 / √(—D) * (π / 2 — arctan(b / √(—D))) if D < 0

    Where p is the order of the moment, κ is the speed of mean reversion of the variance, σ the
    volatility of the variance and ρ the correlation between the stock and its variance. The
    jumps of the Bates model have finite moments of every order, so the same time holds there.

    Args:
        order (float): The order p of the moment E[S_T^p].
        kappa (float): The speed of mean reversion κ, above 0.
        sigma (float): The volatility of the variance σ, above 0.
        rho (float): The correlation ρ between the stock and its variance, between -1 and 1.

    Returns:
        float: The explosion time T* in years, np.inf if the moment never explodes.

    Notes:
        The Carr-Madan transform (the fast Fourier transform of get_heston_price and
        get_bates_price) needs
        the moment of order α + 1 of the damping α to be finite, and is accurate only with a
        margin above it (see fourier_model.FFT_MOMENT_ORDER). References: Andersen, L. B. G.
        and Piterbarg, V. V. (2007). "Moment Explosions in Stochastic Volatility Models."
        Finance and Stochastics, 11(1), 29-50. Checked against a numerical integration of the
        Riccati equation, which blows up at the same time to 1e-8.

    As an example:

    ```python
    from financetoolkit.options.volatility import heston_model

    round(
        heston_model.get_heston_moment_explosion_time(
            order=2.5, kappa=1.0, sigma=1.0, rho=0.99
        ),
        4,
    )
    ```

    Which returns:

    ```
    1.1236
    ```
    """
    curvature = order * (order - 1)
    if curvature <= 0:
        return np.inf
    slope = rho * sigma * order - kappa
    discriminant = slope**2 - sigma**2 * curvature
    if discriminant >= 0:
        if slope < 0:
            return np.inf
        if discriminant == 0:
            return 2 / slope
        root = np.sqrt(discriminant)
        return float(np.log((slope + root) / (slope - root)) / root)
    root = np.sqrt(-discriminant)
    return float(2 / root * (np.pi / 2 - np.arctan(slope / root)))


def get_heston_price(
    stock_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    time_to_expiration: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: Literal["lewis", "cos", "fft"] = "lewis",
) -> float | np.ndarray:
    """
    Calculate the price of a European option in the stochastic volatility model of Heston
    (1993): what a call or put is worth when the variance of the stock follows a
    mean-reverting square-root process correlated with the stock.

    A negative correlation makes the volatility rise as the price falls, which gives the
    downward sloping skew of equity options, and the volatility of the variance curves the
    smile.

    The model is dS = (r — q)S dt + √v S dW_1 and dv = κ(θ — v) dt + σ√v dW_2, with
    dW_1 dW_2 = ρ dt. The price follows from the characteristic function in the form of
    Albrecher et al. (2007) (see get_heston_characteristic_function) with the Fourier pricer
    of the Toolkit (see fourier_model.get_fourier_price): Lewis' integral by default, or the
    COS method or the Carr-Madan fast Fourier transform.

    The formulas are as follows:

    - φ(u) = e^(C + D * v0), with C and D of get_heston_characteristic_function
    - Call = S * e^(—qT) — √(F * K) * e^(—rT) / π * ∫_0^∞ Re[e^(iu * ln(F / K)) * φ(u — i/2)] / (u² + 1/4) du
    - Put = Call — S * e^(—qT) + K * e^(—rT)

    Where S is the stock price, K is the strike price, F = S * e^((r — q)T) the forward price,
    r is the continuously compounded risk-free rate, q is the continuous dividend yield, T is
    the time to expiration in years, v0 is the variance today, κ the speed of mean reversion,
    θ the long-run variance, σ the volatility of the variance and ρ the correlation. The
    variance stays above zero when the Feller condition 2κθ ≥ σ² holds; the price is defined
    either way.

    The Carr-Madan transform needs the moment E[S_T^(α + 1)] of its damping α
    to be finite, and in the Heston model that moment explodes at a finite time when ρσ(α + 1)
    exceeds κ, or σ is large (see get_heston_moment_explosion_time). Close to the explosion the
    transform aliases the damped price into errors of 0.1 or more. When the moment of order
    4 (α + 1 with a margin of 1.5, see fourier_model.FFT_MOMENT_ORDER) is infinite at the
    expiration, the price is taken with the Lewis method instead and a warning says so.

    Referenced against:
    - QuantLib, AnalyticHestonEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_heston_engine.html
    - FinancePy, Heston.value_weber: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/heston.py#L363

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The strike price, or an array of strike prices.
        risk_free_rate (float): The continuously compounded risk-free rate.
        time_to_expiration (float): The time to expiration in years. At zero the option is
            worth its exercise value.
        v0 (float): The variance today (the squared volatility), above 0.
        kappa (float): The speed of mean reversion κ of the variance, above 0.
        theta (float): The long-run variance θ, above 0.
        sigma (float): The volatility of the variance σ, above 0.
        rho (float): The correlation ρ between the stock and its variance, between -1 and 1.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether to price a put instead of a call. Defaults to False.
        method (str, optional): The Fourier pricer. Options are: "lewis" (Lewis' integral), "cos" (the COS method),
            "fft" (the Carr-Madan fast Fourier transform). Defaults to "lewis".

    Returns:
        float | np.ndarray: The option price, a float for a single strike price and an array
        for an array of strike prices.

    Raises:
        TypeError: If a scalar input is not a float or int.
        ValueError: If a variance, κ or σ is not positive or ρ is outside -1 to 1.

    Notes:
        References: Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic
        Volatility with Applications to Bond and Currency Options." The Review of Financial
        Studies, 6(2), 327-343; Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J.
        (2007). "The Little Heston Trap." Wilmott Magazine, January, 83-92; Lewis, A. L.
        (2000). Option Valuation under Stochastic Volatility. Finance Press.
        - QuantLib's AnalyticHestonEngine is used on nine parameter sets from a day to ten
          years, with ρ from -0.999 to 0.99 and σ up to 2: the Lewis
        method agrees within 1e-10, the COS method within 1e-8 for typical parameters and 6e-7
        for a volatile variance with a slow mean reversion (σ = 1.5, κ = 0.5, ρ = -0.9 over
        five years), and the FFT within 1e-6 where it applies.
        - FinancePy's Heston.value_weber agrees with the Lewis method within 1e-10.

    As an example:

    ```python
    from financetoolkit.options.volatility import heston_model

    round(
        heston_model.get_heston_price(
            stock_price=100, strike_price=90, risk_free_rate=0.04, time_to_expiration=0.5,
            v0=0.04, kappa=2.0, theta=0.05, sigma=0.6, rho=-0.7, put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    1.9543
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        risk_free_rate=risk_free_rate,
        time_to_expiration=time_to_expiration,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        dividend_yield=dividend_yield,
    )
    if min(v0, kappa, theta, sigma) <= 0:
        raise ValueError("v0, kappa, theta and sigma must be positive.")
    if not -1 <= rho <= 1:
        raise ValueError(f"rho must be between -1 and 1, received {rho!r}.")

    if method == "fft" and (
        get_heston_moment_explosion_time(FFT_MOMENT_ORDER, kappa, sigma, rho)
        <= time_to_expiration
    ):
        logger.warning(
            "The moment E[S_T^%s] of this Heston model is infinite at the expiration, which "
            "the Carr-Madan transform needs finite, so the option is priced with the Lewis "
            "method instead.",
            FFT_MOMENT_ORDER,
        )
        method = "lewis"

    return get_fourier_price(
        lambda u: get_heston_characteristic_function(
            u, time_to_expiration, v0, kappa, theta, sigma, rho
        ),
        stock_price,
        strike_price,
        risk_free_rate,
        time_to_expiration,
        dividend_yield,
        put_option,
        method=method,
    )


def get_bates_price(
    stock_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    time_to_expiration: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: Literal["lewis", "cos", "fft"] = "lewis",
) -> float | np.ndarray:
    """
    Calculate the price of a European option in the stochastic volatility jump diffusion of
    Bates (1996): what a call or put is worth in the Heston model with lognormal jumps in
    the stock as in Merton (1976).

    The stochastic volatility shapes the smile of the longer expirations and the jumps the
    steep smile of the short ones, which the Heston model alone fits poorly.

    The jumps are independent of the diffusion, so the characteristic function of the log
    return is the product of the Heston one and the compensated jump part of the Merton model
    (see get_heston_characteristic_function and get_merton_characteristic_function), priced
    with the Fourier pricer of the Toolkit (see fourier_model.get_fourier_price).

    The formulas are as follows:

    - φ(u) = φ_Heston(u) * exp(λT * (e^(iuμ — δ²u² / 2) — 1 — iu * k)), k = e^(μ + δ² / 2) — 1
    - Call = S * e^(—qT) — √(F * K) * e^(—rT) / π * ∫_0^∞ Re[e^(iu * ln(F / K)) * φ(u — i/2)] / (u² + 1/4) du
    - Put = Call — S * e^(—qT) + K * e^(—rT)

    Where S is the stock price, K is the strike price, F = S * e^((r — q)T) the forward price,
    r is the continuously compounded risk-free rate, q is the continuous dividend yield, T is
    the time to expiration in years, v0, κ, θ, σ and ρ are the Heston parameters (see
    get_heston_price), λ is the jump intensity (jumps per year) and μ and δ are the mean and
    standard deviation of the log jump size. With λ = 0 it is the Heston price.

    The lognormal jumps have finite moments of every order, so the moment condition of the
    Carr-Madan transform is that of the Heston part: when its moment of order 4
    is infinite at the expiration (see get_heston_price), the price is taken with the Lewis
    method instead and a warning says so.

    Referenced against:
    - QuantLib, BatesEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_bates_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The strike price, or an array of strike prices.
        risk_free_rate (float): The continuously compounded risk-free rate.
        time_to_expiration (float): The time to expiration in years. At zero the option is
            worth its exercise value.
        v0 (float): The variance today, above 0.
        kappa (float): The speed of mean reversion κ of the variance, above 0.
        theta (float): The long-run variance θ, above 0.
        sigma (float): The volatility of the variance σ, above 0.
        rho (float): The correlation ρ between the stock and its variance, between -1 and 1.
        jump_intensity (float): The jump intensity λ, the expected number of jumps a year, 0 or
            above.
        jump_mean (float): The mean μ of the log jump size (negative for downward jumps).
        jump_volatility (float): The standard deviation δ of the log jump size, 0 or above.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether to price a put instead of a call. Defaults to False.
        method (str, optional): The Fourier pricer. Options are: "lewis" (Lewis' integral), "cos" (the COS method),
            "fft" (the Carr-Madan fast Fourier transform). Defaults to "lewis".

    Returns:
        float | np.ndarray: The option price, a float for a single strike price and an array
        for an array of strike prices.

    Raises:
        TypeError: If a scalar input is not a float or int.
        ValueError: If a Heston parameter is out of range or the jump intensity or jump
            volatility is negative.

    Notes:
        References: Bates, D. S. (1996). "Jumps and Stochastic Volatility: Exchange Rate
        Processes Implicit in Deutsche Mark Options." The Review of Financial Studies, 9(1),
        69-107; Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns Are
        Discontinuous." Journal of Financial Economics, 3(1-2), 125-144.
        - QuantLib's BatesEngine with a BatesModel agrees with the Lewis method within
          1e-12.

    As an example:

    ```python
    from financetoolkit.options.volatility import heston_model

    round(
        heston_model.get_bates_price(
            stock_price=100, strike_price=90, risk_free_rate=0.04, time_to_expiration=0.5,
            v0=0.04, kappa=2.0, theta=0.05, sigma=0.6, rho=-0.7, jump_intensity=0.5,
            jump_mean=-0.1, jump_volatility=0.15, put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    2.6013
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        risk_free_rate=risk_free_rate,
        time_to_expiration=time_to_expiration,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
        dividend_yield=dividend_yield,
    )
    if min(v0, kappa, theta, sigma) <= 0:
        raise ValueError("v0, kappa, theta and sigma must be positive.")
    if not -1 <= rho <= 1:
        raise ValueError(f"rho must be between -1 and 1, received {rho!r}.")
    if jump_intensity < 0 or jump_volatility < 0:
        raise ValueError("jump_intensity and jump_volatility must be 0 or above.")

    if method == "fft" and (
        get_heston_moment_explosion_time(FFT_MOMENT_ORDER, kappa, sigma, rho)
        <= time_to_expiration
    ):
        logger.warning(
            "The moment E[S_T^%s] of the Heston part of this Bates model is infinite at the expiration, which "
            "the Carr-Madan transform needs finite, so the option is priced with the Lewis "
            "method instead.",
            FFT_MOMENT_ORDER,
        )
        method = "lewis"

    return get_fourier_price(
        lambda u: get_heston_characteristic_function(
            u, time_to_expiration, v0, kappa, theta, sigma, rho
        )
        * get_merton_characteristic_function(
            u, time_to_expiration, 0.0, jump_intensity, jump_mean, jump_volatility
        ),
        stock_price,
        strike_price,
        risk_free_rate,
        time_to_expiration,
        dividend_yield,
        put_option,
        method=method,
    )


def get_heston_parameters(
    stock_price: float,
    strike_price: np.ndarray,
    time_to_expiration: np.ndarray,
    implied_volatility: np.ndarray,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """
    Calibrate the Heston (1993) model to the implied volatilities of an option chain: the
    variance today v0, the speed of mean reversion κ, the long-run variance θ, the
    volatility of the variance σ and the correlation ρ that best reproduce the market's
    smile across every expiration at once.

    The fit minimizes the vega-weighted price errors, the price error of each quote divided
    by its Black-Scholes vega, which to first order is its error in implied volatility and is
    far cheaper than inverting every model price. A soft-l1 loss with a scale of one
    volatility point keeps a few stale or wide quotes from dominating. The search is a
    bounded trust-region least squares (scipy.optimize.least_squares) from two fixed
    starting points around the at-the-money variance, keeping the best, and prices each
    expiration with the COS method of 256 terms. That is within 1e-8 of the exact price for
    the typical parameters of equity smiles, but a volatile variance with a slow mean reversion
    fattens the tails and 256 terms are then off by 1e-2 or more (Fang and Oosterlee 2008,
    section 5). So the fitted prices are compared with a COS series converged to 1e-10 per
    unit of the strike, and while they differ by more than 1e-5 in implied volatility the fit
    is repeated from its own result with four times the terms. Quotes whose vega is below 5% of the
    at-the-money vega of their expiration (far from the money, where the price hardly moves
    with the volatility) are left out.

    The formulas are as follows:

    - minimize Σ_i ρ_soft-l1([C_Heston(K_i, T_i; v0, κ, θ, σ, ρ) — C_BS(K_i, T_i, σ_i)] / Vega_BS(K_i, T_i, σ_i))
    - ρ_soft-l1(z) = 2s² * (√(1 + (z / s)²) — 1), s = 0.01
    - Feller ratio = 2κθ / σ²

    Where K_i, T_i and σ_i are the strike price, time to expiration and market implied
    volatility of quote i, C_Heston is the Heston call price (see get_heston_price) and C_BS
    and Vega_BS are the Black-Scholes price and vega. The call and the put of a strike carry
    the same implied volatility error (both models satisfy put-call parity), so calls are
    fitted throughout. The Feller condition (ratio ≥ 1) is reported, not imposed: fits to
    equity smiles commonly break it, as the steep short-dated skew asks for a high σ, and
    imposing it worsens the fit.

    Referenced against:
    - QuantLib, HestonModelHelper: https://www.quantlib.org/reference/class_quant_lib_1_1_heston_model_helper.html

    Args:
        stock_price (float): The current stock price.
        strike_price (np.ndarray): The strike price of every quote.
        time_to_expiration (np.ndarray): The time to expiration of every quote, in years.
        implied_volatility (np.ndarray): The market implied volatility of every quote.
        risk_free_rate (float): The continuously compounded risk-free rate.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.

    Returns:
        dict[str, float]: v0, kappa, theta, sigma and rho, the Feller ratio 2κθ / σ²,
        the root mean squared error in implied volatility (from inverting the converged
        prices of the fit) and the number of quotes fitted.

    Raises:
        ValueError: If the inputs differ in length or fewer than five quotes have a vega of
            at least 5% of the at-the-money vega.

    Notes:
        With a single expiration κ and θ are hardly identified separately (only the variance
        averaged over the expiration matters), so a surface of several expirations gives
        more meaningful parameters. Even then κ and σ trade off: a faster mean reversion with
        a more volatile variance gives much the same smile, so on a market surface the
        objective is flat along that direction and two similar smiles can give a quite
        different κ, while θ, the Feller ratio and the at-the-money term structure are far
        better determined. References: Heston, S. L. (1993). "A Closed-Form
        Solution for Options with Stochastic Volatility with Applications to Bond and
        Currency Options." The Review of Financial Studies, 6(2), 327-343; Fang, F. and
        Oosterlee, C. W. (2008). "A Novel Pricing Method for European Options Based on
        Fourier-Cosine Series Expansions." SIAM Journal on Scientific Computing, 31(2),
        826-848.
        - QuantLib's HestonModelHelper calibration and this one both recover the parameters
          of a known Heston surface from its implied volatilities.

    As an example:

    ```python
    import numpy as np

    from financetoolkit.options.volatility import heston_model

    strikes = np.tile(np.linspace(80, 120, 9), 3)
    times = np.repeat([0.25, 0.5, 1.0], 9)
    volatility = 0.22 - 0.25 * np.log(strikes / 100) / np.sqrt(times) * 0.3

    {
        name: round(value, 4)
        for name, value in heston_model.get_heston_parameters(
            100, strikes, times, volatility, 0.04
        ).items()
    }
    ```

    Which returns:

    ```
    {'v0': 0.0493, 'kappa': 4.7506, 'theta': 0.0486, 'sigma': 0.2827, 'rho': -0.6788,
     'feller_ratio': 5.7758, 'rmse': 0.0005, 'quotes': 27}
    ```
    """
    strikes = np.asarray(strike_price, dtype=float)
    times = np.asarray(time_to_expiration, dtype=float)
    volatilities = np.asarray(implied_volatility, dtype=float)

    if not len(strikes) == len(times) == len(volatilities):
        raise ValueError(
            "strike_price, time_to_expiration and implied_volatility differ in length."
        )
    # The vega per unit of volatility (the Toolkit's vega is per volatility point). A quote
    # whose price hardly moves with the volatility says little about it, and its price error
    # over its vega is mostly the rounding of two prices close to the exercise value.
    vegas = 100 * np.asarray(
        get_vega(
            stock_price, strikes, times, risk_free_rate, volatilities, dividend_yield
        )
    )
    at_the_money_vega = (
        stock_price * np.exp(-dividend_yield * times) * np.sqrt(times / (2 * np.pi))
    )
    informative = vegas >= MINIMUM_RELATIVE_VEGA * at_the_money_vega
    strikes, times, volatilities, vegas = (
        strikes[informative],
        times[informative],
        volatilities[informative],
        vegas[informative],
    )

    if len(strikes) < MINIMUM_QUOTES_FOR_HESTON_FIT:
        raise ValueError(
            f"The Heston fit needs at least {MINIMUM_QUOTES_FOR_HESTON_FIT} quotes with a "
            f"vega of at least {MINIMUM_RELATIVE_VEGA:.0%} of the at-the-money vega, "
            f"received {len(strikes)}."
        )

    market_prices = get_black_scholes(
        stock_price, strikes, risk_free_rate, volatilities, times, dividend_yield
    )

    expirations = np.unique(times)
    groups = [times == expiration for expiration in expirations]

    def model_prices(parameters: np.ndarray, terms: int | None) -> np.ndarray:
        # COS with a fixed number of terms during the search, which keeps the residuals a
        # smooth function of the parameters for the finite-difference Jacobian, and COS with
        # as many terms as it needs to converge (terms=None) for the check of the result.
        prices = np.empty_like(strikes)
        for expiration, group in zip(expirations, groups, strict=True):
            prices[group] = get_fourier_price(
                lambda u, expiration=expiration: get_heston_characteristic_function(
                    u, expiration, *parameters
                ),
                stock_price,
                strikes[group],
                risk_free_rate,
                expiration,
                dividend_yield,
                method="cos",
                cos_terms=terms or HESTON_CALIBRATION_TERMS,
                cos_tolerance=COS_TOLERANCE if terms is None else None,
            )
        return prices

    def fit(start: np.ndarray, terms: int) -> OptimizeResult:
        return least_squares(
            lambda parameters: (model_prices(parameters, terms) - market_prices)
            / vegas,
            start,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=HESTON_LOSS_SCALE,
            x_scale=np.array([0.01, 1.0, 0.01, 0.1, 0.1]),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

    # The at-the-money variance of the nearest and the farthest expiration start v0 and θ,
    # with a slow and a fast mean reversion and a low and a high volatility of variance.
    log_moneyness = np.abs(
        np.log(strikes / stock_price) - (risk_free_rate - dividend_yield) * times
    )
    nearest = groups[0]
    farthest = groups[-1]
    start_v0 = float(volatilities[nearest][np.argmin(log_moneyness[nearest])] ** 2)
    start_theta = float(volatilities[farthest][np.argmin(log_moneyness[farthest])] ** 2)
    lower, upper = HESTON_BOUNDS

    terms = HESTON_CALIBRATION_TERMS
    best = min(
        (
            fit(
                np.clip(
                    [start_v0, kappa, start_theta, sigma, -0.6],
                    lower * 1.01,
                    upper * 0.99,
                ),
                terms,
            )
            for kappa, sigma in HESTON_STARTS
        ),
        key=lambda result: result.cost,
    )

    # The COS series converges slowly where a volatile variance or a slow mean reversion
    # fattens the tails (Fang and Oosterlee 2008, section 5), so the fit is checked against
    # the Lewis prices and repeated from its own result with four times the terms until the
    # two agree within HESTON_PRICER_TOLERANCE in implied volatility. The Lewis prices of the
    # accepted fit give the RMSE, which so measures the fit alone.
    fitted_prices = model_prices(best.x, None)
    while (
        np.max(np.abs(model_prices(best.x, terms) - fitted_prices) / vegas)
        > HESTON_PRICER_TOLERANCE
        and terms < COS_MAXIMUM_TERMS
    ):
        terms *= 4
        best = fit(best.x, terms)
        fitted_prices = model_prices(best.x, None)

    v0, kappa, theta, sigma, rho = (float(value) for value in best.x)

    # The implied volatility of the fitted prices by Newton's method from the market's, which
    # lies close by: every quote kept has a vega of at least 5% of the at-the-money vega.
    fitted_volatility = volatilities.copy()
    for _ in range(IMPLIED_VOLATILITY_ITERATIONS):
        fitted_volatility = np.clip(
            fitted_volatility
            - (
                get_black_scholes(
                    stock_price,
                    strikes,
                    risk_free_rate,
                    fitted_volatility,
                    times,
                    dividend_yield,
                )
                - fitted_prices
            )
            / (
                100
                * get_vega(
                    stock_price,
                    strikes,
                    times,
                    risk_free_rate,
                    fitted_volatility,
                    dividend_yield,
                )
            ),
            1e-4,
            5.0,
        )

    return {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
        "feller_ratio": 2 * kappa * theta / sigma**2,
        "rmse": float(np.sqrt(np.mean((fitted_volatility - volatilities) ** 2))),
        "quotes": len(strikes),
    }
