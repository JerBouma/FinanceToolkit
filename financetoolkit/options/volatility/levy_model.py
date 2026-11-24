"""Jump Diffusion and Lévy Process Models"""

__docformat__ = "google"

from typing import Literal

import numpy as np
from scipy.stats import poisson

from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.options.volatility.fourier_model import (
    FFT_MOMENT_ORDER,
    get_fourier_price,
)
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# The Poisson probability left out of the Merton series: the terms beyond the
# (1 - 1e-16) quantile of the number of jumps add less than a double can hold.
MERTON_SERIES_TAIL = 1e-16


def get_merton_characteristic_function(
    u: np.ndarray,
    time_to_expiration: float,
    volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
) -> np.ndarray:
    """
    Calculate the characteristic function of the log return X = ln(S_T / F) in the jump
    diffusion of Merton (1976): the Fourier transform of its distribution, a geometric
    Brownian motion with lognormal jumps arriving as a Poisson process, compensated so that
    the forward F is the expected price.

    The formulas are as follows:

    - φ(u) = exp(—σ² * T / 2 * (iu + u²) + λT * (e^(iuμ — δ²u² / 2) — 1 — iu * k))
    - k = e^(μ + δ² / 2) — 1

    Where σ is the volatility of the diffusion, λ is the jump intensity (the expected number
    of jumps a year), μ and δ are the mean and standard deviation of the log of a jump's size
    and k is the expected relative jump. With σ = 0 it is the jump part the Bates model adds
    to the Heston model.

    Args:
        u (np.ndarray): The arguments, real or complex.
        time_to_expiration (float): The time to expiration in years.
        volatility (float): The volatility σ of the diffusion.
        jump_intensity (float): The jump intensity λ, jumps per year.
        jump_mean (float): The mean μ of the log jump size.
        jump_volatility (float): The standard deviation δ of the log jump size.

    Returns:
        np.ndarray: E[e^(iuX)] for every u.

    As an example, φ(—i) = E[S_T / F] = 1 (the forward is the expected price) and φ(1):

    ```python
    import numpy as np

    from financetoolkit.options.volatility import levy_model

    np.round(
        levy_model.get_merton_characteristic_function(
            np.array([-1j, 1.0]), time_to_expiration=0.5, volatility=0.2,
            jump_intensity=0.5, jump_mean=-0.1, jump_volatility=0.15,
        ),
        6,
    ).tolist()
    ```

    Which returns:

    ```
    [(1+0j), (0.985977-0.01326j)]
    ```
    """
    u = np.asarray(u, dtype=complex)
    expected_jump = np.expm1(jump_mean + 0.5 * jump_volatility**2)

    return np.exp(
        -0.5 * volatility**2 * time_to_expiration * (1j * u + u * u)
        + jump_intensity
        * time_to_expiration
        * (
            np.exp(1j * u * jump_mean - 0.5 * jump_volatility**2 * u * u)
            - 1
            - 1j * u * expected_jump
        )
    )


def get_merton_jump_diffusion_price(
    stock_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: Literal["series", "lewis", "cos", "fft"] = "series",
) -> float | np.ndarray:
    """
    Calculate the price of a European option in the jump diffusion model of Merton (1976):
    what a call or put is worth when the stock follows a geometric Brownian motion with
    lognormal jumps arriving as a Poisson process, which gives the fat tails and short-dated
    smile that Black-Scholes has none of.

    Conditional on n jumps the log return is normal, so the price is a Poisson-weighted sum
    of Black-Scholes prices (the series, the closed form). The other methods price from
    the characteristic function (see get_merton_characteristic_function) with the Fourier
    pricer of the Toolkit (see fourier_model.get_fourier_price).

    The formulas are as follows:

    - Price = Σ_n e^(—λ'T) * (λ'T)^n / n! * BS(S, K, r_n, σ_n, T, q)
    - λ' = λ * (1 + k), k = e^(μ + δ² / 2) — 1
    - σ_n² = σ² + n * δ² / T
    - r_n = r — λk + n * ln(1 + k) / T

    Where S is the stock price, K is the strike price, r is the continuously compounded
    risk-free rate, q is the continuous dividend yield, σ is the volatility of the diffusion,
    T is the time to expiration in years, λ is the jump intensity (jumps per year), μ and δ
    are the mean and standard deviation of the log jump size and BS is the Black-Scholes
    price. The series runs until the Poisson probability left out is below 1e-16. With λ = 0
    it is the Black-Scholes price.

    Referenced against:
    - QuantLib, BatesEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_bates_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The strike price, or an array of strike prices.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility σ of the diffusion, above 0.
        time_to_expiration (float): The time to expiration in years. At zero the option is
            worth its exercise value.
        jump_intensity (float): The jump intensity λ, the expected number of jumps a year, 0 or
            above.
        jump_mean (float): The mean μ of the log jump size (negative for downward jumps).
        jump_volatility (float): The standard deviation δ of the log jump size, 0 or above.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether to price a put instead of a call. Defaults to False.
        method (str, optional): The pricer. Options are: "series" (the closed form), "lewis"
            (Lewis' integral), "cos" (the COS method), "fft" (the Carr-Madan fast Fourier
            transform). Defaults to "series".

    Returns:
        float | np.ndarray: The option price, a float for a single strike price and an array
        for an array of strike prices.

    Raises:
        TypeError: If a scalar input is not a float or int.
        ValueError: If the volatility is not positive or the jump intensity or jump volatility
            is negative.

    Notes:
        References: Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns Are
        Discontinuous." Journal of Financial Economics, 3(1-2), 125-144. The series agrees
        with the Lewis method within 1e-13, an independent route through the characteristic
        function, and with QuantLib's BatesEngine at the Merton
        limit of a nearly constant variance within 2e-6, the effect of the variance's small
        volatility there.

    As an example:

    ```python
    from financetoolkit.options.volatility import levy_model

    round(
        levy_model.get_merton_jump_diffusion_price(
            stock_price=100, strike_price=90, risk_free_rate=0.04, volatility=0.2,
            time_to_expiration=0.5, jump_intensity=0.5, jump_mean=-0.1, jump_volatility=0.15,
            put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    2.0911
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
        dividend_yield=dividend_yield,
    )
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, received {volatility!r}.")
    if jump_intensity < 0 or jump_volatility < 0:
        raise ValueError("jump_intensity and jump_volatility must be 0 or above.")

    if method != "series" or time_to_expiration <= 0:
        return get_fourier_price(
            lambda u: get_merton_characteristic_function(
                u,
                time_to_expiration,
                volatility,
                jump_intensity,
                jump_mean,
                jump_volatility,
            ),
            stock_price,
            strike_price,
            risk_free_rate,
            time_to_expiration,
            dividend_yield,
            put_option,
            method="lewis" if method == "series" else method,
        )

    strikes = np.asarray(strike_price, dtype=float)
    expected_jump = np.expm1(jump_mean + 0.5 * jump_volatility**2)
    jump_rate = jump_intensity * (1 + expected_jump) * time_to_expiration
    last_term = (
        int(poisson.ppf(1 - MERTON_SERIES_TAIL, jump_rate)) if jump_rate > 0 else 0
    )

    price = np.zeros_like(strikes)
    for jumps in range(last_term + 1):
        price = price + poisson.pmf(jumps, jump_rate) * get_black_scholes(
            stock_price=stock_price,
            strike_price=strikes,
            risk_free_rate=risk_free_rate
            - jump_intensity * expected_jump
            + jumps * np.log1p(expected_jump) / time_to_expiration,
            volatility=np.sqrt(
                volatility**2 + jumps * jump_volatility**2 / time_to_expiration
            ),
            time_to_expiration=time_to_expiration,
            dividend_yield=dividend_yield,
            put_option=put_option,
        )

    return float(price) if np.ndim(strike_price) == 0 else np.asarray(price)


def get_variance_gamma_characteristic_function(
    u: np.ndarray,
    time_to_expiration: float,
    volatility: float,
    drift: float,
    variance_rate: float,
) -> np.ndarray:
    """
    Calculate the characteristic function of the log return X = ln(S_T / F) in the Variance
    Gamma model of Madan, Carr and Chang (1998): the Fourier transform of its distribution,
    a Brownian motion with drift θ and volatility σ run on a gamma-distributed clock whose
    variance per year is ν.

    The formulas are as follows:

    - φ(u) = e^(iuωT) * (1 — iuθν + σ²νu² / 2)^(—T / ν)
    - ω = ln(1 — θν — σ²ν / 2) / ν

    Where σ is the volatility, θ the drift (the skew: negative for a heavier left tail),
    ν the variance rate (the kurtosis) and ω the compensator that makes the forward the
    expected price. It needs 1 — θν — σ²ν / 2 > 0.

    Args:
        u (np.ndarray): The arguments, real or complex.
        time_to_expiration (float): The time to expiration in years.
        volatility (float): The volatility σ.
        drift (float): The drift θ of the subordinated Brownian motion.
        variance_rate (float): The variance rate ν of the gamma clock.

    Returns:
        np.ndarray: E[e^(iuX)] for every u.

    As an example, φ(—i) = E[S_T / F] = 1 (the forward is the expected price) and φ(1):

    ```python
    import numpy as np

    from financetoolkit.options.volatility import levy_model

    np.round(
        levy_model.get_variance_gamma_characteristic_function(
            np.array([-1j, 1.0]), time_to_expiration=0.5, volatility=0.2, drift=-0.15,
            variance_rate=0.2,
        ),
        6,
    ).tolist()
    ```

    Which returns:

    ```
    [(1+0j), (0.988911-0.010393j)]
    ```
    """
    u = np.asarray(u, dtype=complex)
    compensator = (
        np.log(1 - drift * variance_rate - 0.5 * volatility**2 * variance_rate)
        / variance_rate
    )

    return np.exp(
        1j * u * compensator * time_to_expiration
        - time_to_expiration
        / variance_rate
        * np.log(
            1
            - 1j * u * drift * variance_rate
            + 0.5 * volatility**2 * variance_rate * u * u
        )
    )


def get_variance_gamma_price(
    stock_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    drift: float,
    variance_rate: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: Literal["lewis", "cos", "fft"] = "lewis",
) -> float | np.ndarray:
    """
    Calculate the price of a European option in the Variance Gamma model of Madan, Carr and
    Chang (1998): what a call or put is worth when the stock follows a pure-jump Lévy
    process, a Brownian motion with drift θ and volatility σ evaluated at a
    gamma-distributed time.

    The drift sets the skew and the variance rate ν the kurtosis of the returns, so the
    model fits a smile that Black-Scholes cannot, with three parameters.

    The price follows from the characteristic function (see
    get_variance_gamma_characteristic_function) with the Fourier pricer of the Toolkit (see
    fourier_model.get_fourier_price); the fast Fourier transform is that of Carr and
    Madan (1999), the method they introduced with this model.

    The formulas are as follows:

    - φ(u) = e^(iuωT) * (1 — iuθν + σ²νu² / 2)^(—T / ν), ω = ln(1 — θν — σ²ν / 2) / ν
    - Call = S * e^(—qT) — √(F * K) * e^(—rT) / π * ∫_0^∞ Re[e^(iu * ln(F / K)) * φ(u — i/2)] / (u² + 1/4) du

    Where S is the stock price, K is the strike price, F = S * e^((r — q)T) the forward price,
    r is the continuously compounded risk-free rate, q is the continuous dividend yield, T is
    the time to expiration in years, σ is the volatility, θ the drift and ν the variance
    rate. As ν goes to zero the price goes to the Black-Scholes price with volatility σ when
    θ = 0.

    Referenced against:
    - QuantLib, VarianceGammaEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_variance_gamma_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The strike price, or an array of strike prices.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility σ, above 0.
        time_to_expiration (float): The time to expiration in years. At zero the option is
            worth its exercise value.
        drift (float): The drift θ (negative for a heavier left tail).
        variance_rate (float): The variance rate ν, above 0.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether to price a put instead of a call. Defaults to False.
        method (str, optional): The Fourier pricer. Options are: "lewis" (Lewis' integral),
            "cos" (the COS method), "fft" (the Carr-Madan fast Fourier transform). Defaults to
            "lewis".

    Returns:
        float | np.ndarray: The option price, a float for a single strike price and an array
        for an array of strike prices.

    Raises:
        TypeError: If a scalar input is not a float or int.
        ValueError: If the volatility or variance rate is not positive, or
            1 — θν — σ²ν / 2 is not positive (the forward is then infinite).

    Notes:
        References: Madan, D. B., Carr, P. and Chang, E. C. (1998). "The Variance Gamma
        Process and Option Pricing." European Finance Review, 2(1), 79-105; Carr, P. and
        Madan, D. B. (1999). "Option Valuation Using the Fast Fourier Transform." Journal of
        Computational Finance, 2(4), 61-73. The Lewis method agrees within 1e-13 with an
        independent integration of the conditional Black-Scholes price over the gamma clock,
        and QuantLib's VarianceGammaEngine within 2e-8, the
        precision of its own integration. The Carr-Madan transform needs
        the moment E[S_T^p] to exist, 1 — θνp — σ²νp² / 2 > 0, for p = α + 1 of the damping
        α = 1.5, and is accurate only with a margin above it, so with p = 4 (see
        fourier_model.FFT_MOMENT_ORDER). That fails for a large variance rate or volatility,
        and the price is then taken with the COS method instead, with a warning: the
        characteristic function decays only like |u|^(-2T / ν), so the Lewis integral takes
        seconds where the COS series converges in milliseconds.

    As an example:

    ```python
    from financetoolkit.options.volatility import levy_model

    round(
        levy_model.get_variance_gamma_price(
            stock_price=100, strike_price=90, risk_free_rate=0.04, volatility=0.2,
            time_to_expiration=0.5, drift=-0.15, variance_rate=0.2, put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    1.6947
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        drift=drift,
        variance_rate=variance_rate,
        dividend_yield=dividend_yield,
    )
    if volatility <= 0 or variance_rate <= 0:
        raise ValueError("volatility and variance_rate must be positive.")
    if 1 - drift * variance_rate - 0.5 * volatility**2 * variance_rate <= 0:
        raise ValueError(
            "1 - drift * variance_rate - volatility^2 * variance_rate / 2 must be positive."
        )
    if (
        method == "fft"
        and 1
        - drift * variance_rate * FFT_MOMENT_ORDER
        - 0.5 * volatility**2 * variance_rate * FFT_MOMENT_ORDER**2
        <= 0
    ):
        # The COS method rather than Lewis' integral: the characteristic function decays only
        # like a power |u|^(-2T / ν), which makes the adaptive integral take seconds.
        logger.warning(
            "The moment E[S_T^%s] of this Variance Gamma model is infinite, which the "
            "Carr-Madan transform needs finite, so the option is priced with the COS "
            "method instead.",
            FFT_MOMENT_ORDER,
        )
        method = "cos"

    return get_fourier_price(
        lambda u: get_variance_gamma_characteristic_function(
            u, time_to_expiration, volatility, drift, variance_rate
        ),
        stock_price,
        strike_price,
        risk_free_rate,
        time_to_expiration,
        dividend_yield,
        put_option,
        method=method,
    )
