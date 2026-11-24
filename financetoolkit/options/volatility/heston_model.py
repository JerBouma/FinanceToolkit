"""Heston Stochastic Volatility and Bates Models"""

__docformat__ = "google"

from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from financetoolkit.options.greeks.greeks_model import get_vega
from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.options.volatility.fourier_model import get_fourier_price
from financetoolkit.options.volatility.levy_model import (
    get_merton_characteristic_function,
)

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

# The number of COS terms the calibration prices with: the COS method prices a whole expiry
# for the cost of one characteristic function evaluation per term.
HESTON_CALIBRATION_TERMS = 256

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
    (1993) model, in the form of Albrecher, Mayer, Schoutens and Tistaert (2007) ("the little
    Heston trap"), which keeps the complex logarithm on its principal branch for every
    expiration, so no count of branch crossings is needed.

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
    (1993), in which the variance of the stock follows a mean-reverting square-root process
    correlated with the stock. A negative correlation makes the volatility rise as the price
    falls, which gives the downward sloping skew of equity options, and the volatility of the
    variance curves the smile.

    The model is dS = (r — q)S dt + √v S dW_1 and dv = κ(θ — v) dt + σ√v dW_2, with
    dW_1 dW_2 = ρ dt. The price follows from the characteristic function in the form of
    Albrecher et al. (2007) (see get_heston_characteristic_function) with the Fourier pricer
    of the Toolkit (see fourier_model.get_fourier_price): Lewis' integral by default, or the
    COS method (method="cos") or the Carr-Madan fast Fourier transform (method="fft").

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
        method (str, optional): "lewis", "cos" or "fft". Defaults to "lewis".

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
        (2000). Option Valuation under Stochastic Volatility. Finance Press. Referenced
        against QuantLib's AnalyticHestonEngine (https://www.quantlib.org): the Lewis
        method agrees within 1e-13, the COS method within 1e-8 and the FFT within 1e-6.
        FinancePy's Heston.value_weber (https://github.com/domokane/FinancePy) agrees with
        the Lewis method within 1e-10.

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
    Bates (1996): the Heston model with lognormal jumps in the stock as in Merton (1976). The
    stochastic volatility shapes the smile of the longer expirations and the jumps the steep
    smile of the short ones, which the Heston model alone fits poorly.

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
        method (str, optional): "lewis", "cos" or "fft". Defaults to "lewis".

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
        Discontinuous." Journal of Financial Economics, 3(1-2), 125-144. Referenced against
        QuantLib's BatesEngine with a BatesModel (https://www.quantlib.org), which agrees
        with the Lewis method within 1e-12.

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
    variance today v0, the speed of mean reversion κ, the long-run variance θ, the volatility
    of the variance σ and the correlation ρ that best reproduce the market's smile across
    every expiration at once.

    The fit minimizes the vega-weighted price errors, the price error of each quote divided
    by its Black-Scholes vega, which to first order is its error in implied volatility and is
    far cheaper than inverting every model price. A soft-l1 loss with a scale of one
    volatility point keeps a few stale or wide quotes from dominating. The search is a
    bounded trust-region least squares (scipy.optimize.least_squares) from two fixed
    starting points around the at-the-money variance, keeping the best, and prices each
    expiration with the COS method (256 terms, within about 1e-8 of the exact price for
    typical parameters). Quotes whose vega is below 5% of the
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

    Args:
        stock_price (float): The current stock price.
        strike_price (np.ndarray): The strike price of every quote.
        time_to_expiration (np.ndarray): The time to expiration of every quote, in years.
        implied_volatility (np.ndarray): The market implied volatility of every quote.
        risk_free_rate (float): The continuously compounded risk-free rate.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.

    Returns:
        dict[str, float]: v0, kappa, theta, sigma and rho, the Feller ratio 2κθ / σ²,
        the root mean squared error in implied volatility (from inverting the fitted
        prices) and the number of quotes fitted.

    Raises:
        ValueError: If the inputs differ in length or fewer than five quotes have a vega of
            at least 5% of the at-the-money vega.

    Notes:
        With a single expiration κ and θ are hardly identified separately (only the variance
        averaged over the expiration matters), so a surface of several expirations gives
        more meaningful parameters. References: Heston, S. L. (1993). "A Closed-Form
        Solution for Options with Stochastic Volatility with Applications to Bond and
        Currency Options." The Review of Financial Studies, 6(2), 327-343; Fang, F. and
        Oosterlee, C. W. (2008). "A Novel Pricing Method for European Options Based on
        Fourier-Cosine Series Expansions." SIAM Journal on Scientific Computing, 31(2),
        826-848. Referenced against QuantLib's HestonModelHelper calibration
        (https://www.quantlib.org) on the implied volatilities of a known Heston surface,
        whose parameters both recover.

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

    def model_prices(parameters: np.ndarray) -> np.ndarray:
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
                cos_terms=HESTON_CALIBRATION_TERMS,
            )
        return prices

    def residuals(parameters: np.ndarray) -> np.ndarray:
        return (model_prices(parameters) - market_prices) / vegas

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

    best = None
    for kappa, sigma in HESTON_STARTS:
        result = least_squares(
            residuals,
            np.clip(
                [start_v0, kappa, start_theta, sigma, -0.6], lower * 1.01, upper * 0.99
            ),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=HESTON_LOSS_SCALE,
            x_scale=np.array([0.01, 1.0, 0.01, 0.1, 0.1]),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if best is None or result.cost < best.cost:
            best = result

    v0, kappa, theta, sigma, rho = (float(value) for value in best.x)

    # The implied volatility of the fitted prices by Newton's method from the market's, which
    # lies close by: every quote kept has a vega of at least 5% of the at-the-money vega.
    fitted_prices = model_prices(best.x)
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
