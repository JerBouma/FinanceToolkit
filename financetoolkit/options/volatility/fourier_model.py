"""Fourier Option Pricing Model"""

__docformat__ = "google"

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline

from financetoolkit.utilities.error_model import check_option

FOURIER_METHODS = ("lewis", "cos", "fft")

# The absolute and relative tolerance of the adaptive Lewis integral, per unit of the
# strike: close to the precision a double holds, so the method serves as the reference
# the COS and FFT prices are checked against.
LEWIS_TOLERANCE = 1e-13

# The width of the COS truncation range in standard deviations (Fang and Oosterlee 2008,
# eq. 49, with the fourth cumulant to cover fat tails).
COS_RANGE_WIDTH = 10

# The number of COS terms per standard deviation of the log return in the truncation range,
# which sets the first number of terms tried: 256 for a normal log return (a range of 20
# standard deviations), more where a fat tail widens the range.
COS_TERMS_PER_DEVIATION = 8

# The largest difference, per unit of the strike, between the COS prices from N and 2N terms
# at which the series counts as converged, and the most terms it may double to.
COS_TOLERANCE = 1e-10
COS_MAXIMUM_TERMS = 2**16

# The order p of the moment E[S_T^p] that must be finite for the Carr-Madan transform with its
# default damping α = 1.5. Carr and Madan (1999) need p = α + 1; as the moment that explodes
# approaches α + 1, the damped call price decays ever more slowly in the log strike and the
# transform aliases it, so the margin of 1.5 keeps that error near 1e-6 (measured on Heston
# parameter sets against the Lewis integral: an error of 1e-3 at a margin of 0.8 and 4e-7 at
# 1.2).
FFT_MOMENT_ORDER = 4.0

# The step of the central differences that give the cumulants for the COS range, on the
# cumulant generating function of the log return. Only the range depends on them, and the
# price converges for any range that holds the density, so a step this coarse is precise
# enough and stays well inside the region where the moment generating function exists.
CUMULANT_STEP = 0.02


def get_fourier_price(
    characteristic_function: Callable[[np.ndarray], np.ndarray],
    stock_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    method: Literal["lewis", "cos", "fft"] = "lewis",
    cos_terms: int = 256,
    cos_tolerance: float | None = COS_TOLERANCE,
    fft_points: int = 4096,
    fft_spacing: float = 0.25,
    fft_damping: float = 1.5,
) -> float | np.ndarray:
    """
    Calculate the price of European options from the characteristic function of the log
    return: what calls or puts are worth in any model whose characteristic function is
    known, with Lewis' single integral (2001), the COS method of Fang and Oosterlee (2008)
    or the fast Fourier transform of Carr and Madan (1999).

    This is the pricer behind the Heston, Bates, Merton jump diffusion and Variance Gamma
    models of the Finance Toolkit: a model only supplies the characteristic function.

    The characteristic function is that of X = ln(S_T / F), the log of the stock price at
    expiration over the forward price F = S * e^((r — q)T), so φ(u) = E[e^(iuX)] with φ(—i) = 1
    (the forward is the expected price). Every model below is written in that form, which
    leaves the rate and dividend yield to this function.

    The formulas are as follows:

    - Lewis: Call = S * e^(—qT) — √(F * K) * e^(—rT) / π * ∫_0^∞ Re[e^(iu * ln(F / K)) * φ(u — i/2)] / (u² + 1/4) du
    - COS: Put = e^(—rT) * Σ'_k Re[φ(u_k) * e^(—i * u_k * a)] * V_k, u_k = kπ / (b — a)
    - V_k = 2K / (b — a) * (ψ_k(a', 0) — χ_k(a', 0)), on the log-moneyness y = ln(F / K) + X over [a', b']
    - Carr-Madan: Call(k) = e^(—αk) / π * ∫_0^∞ Re[e^(—ivk) * ψ(v)] dv, k = ln K
    - ψ(v) = e^(—rT) * φ_lnS(v — (α + 1)i) / (α² + α — v² + i(2α + 1)v)
    - Put = Call — S * e^(—qT) + K * e^(—rT) (put-call parity) for Lewis and Carr-Madan

    Where S is the stock price, K is the strike price, r is the continuously compounded
    risk-free rate, q is the continuous dividend yield, T is the time to expiration in years,
    Σ' halves the first term, [a, b] is the truncation range of X (its first cumulant plus and
    minus ten times √(c2 + √c4), the cumulants taken by central differences of ln E[e^(sX)]),
    a' and b' are the same range around ln(F / K), χ_k and ψ_k are the cosine coefficients of
    e^y and 1 on [a', 0], α is the damping factor and φ_lnS(v) = e^(iv * ln F) * φ(v).

    The Lewis integral is integrated adaptively to a tolerance of 1e-13 per unit of the strike,
    which makes it the most precise and the default. The COS method needs one evaluation of φ
    per term for every strike at once and converges exponentially in the number of terms for
    smooth densities, but fat tails widen the truncation range and need more terms: a Heston
    model with a volatile variance and a slow mean reversion (σ = 1.5, κ = 0.5, ρ = —0.9) over
    five years is 2e-2 off with 1024 terms and 6e-6 with 4096. The number of terms therefore
    starts from the width of the range, 8 per standard deviation of X and at least cos_terms,
    and doubles until the sums of N and 2N terms agree within cos_tolerance per unit of the
    strike (up to 65,536 terms). The truncation range then sets the precision: within about
    1e-9 of the exact price for typical equity parameters and 6e-7 for the volatile case. The
    Carr-Madan transform prices fft_points log-strikes spaced 2π / (fft_points * fft_spacing)
    apart around the forward in one transform, with Simpson weights, and interpolates them to
    the strikes with a cubic spline; it is the fastest for a whole chain and the least precise
    (about 1e-6 per unit of the strike with the defaults). It needs E[S_T^(α + 1)] to be
    finite and is accurate only with a margin above it (see FFT_MOMENT_ORDER), which the
    Heston, Bates and Variance Gamma pricers check before using it.

    Every price is kept at or above its no-arbitrage lower bound, max(S * e^(—qT) — K * e^(—rT), 0)
    for a call and max(K * e^(—rT) — S * e^(—qT), 0) for a put, which takes out the integration
    noise of prices close to it (such as —1e-13 for a deep out-of-the-money option a day from
    expiration).

    Referenced against:
    - QuantLib, AnalyticHestonEngine (Heston): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_heston_engine.html
    - QuantLib, BatesEngine (Bates): https://www.quantlib.org/reference/class_quant_lib_1_1_bates_engine.html
    - QuantLib, VarianceGammaEngine (Variance Gamma): https://www.quantlib.org/reference/class_quant_lib_1_1_variance_gamma_engine.html

    Args:
        characteristic_function (Callable): The characteristic function φ(u) of
            X = ln(S_T / F), taking a complex np.ndarray of u and returning E[e^(iuX)] for each.
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The strike price, or an array of strike prices.
        risk_free_rate (float): The continuously compounded risk-free rate.
        time_to_expiration (float): The time to expiration in years. At zero the option is
            worth its exercise value.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether to price puts instead of calls. Defaults to False.
        method (str, optional): The Fourier pricer. Options are: "lewis" (Lewis' integral),
            "cos" (the COS method), "fft" (the Carr-Madan fast Fourier transform). Defaults to
            "lewis".
        cos_terms (int, optional): The smallest number of cosine terms of the COS method.
            Defaults to 256.
        cos_tolerance (float | None, optional): The largest difference per unit of the strike
            between the COS prices of N and 2N terms at which the series counts as converged,
            or None to sum exactly cos_terms terms. Defaults to 1e-10.
        fft_points (int, optional): The number of points of the Carr-Madan transform, a power of
            2. Defaults to 4096.
        fft_spacing (float, optional): The spacing η of the integration grid of the Carr-Madan
            transform. Defaults to 0.25.
        fft_damping (float, optional): The damping factor α of the Carr-Madan transform, above
            0. Defaults to 1.5.

    Returns:
        float | np.ndarray: The option price, a float for a single strike price and an array
        for an array of strike prices.

    Raises:
        ValueError: If the method is unknown or a strike price is not positive.

    Notes:
        References: Lewis, A. L. (2001). "A Simple Option Formula for General Jump-Diffusion
        and Other Exponential Lévy Processes." Envision Financial Systems and
        OptionCity.net; Fang, F. and Oosterlee, C. W. (2008). "A Novel Pricing Method for
        European Options Based on Fourier-Cosine Series Expansions." SIAM Journal on
        Scientific Computing, 31(2), 826-848; Carr, P. and Madan, D. B. (1999). "Option
        Valuation Using the Fast Fourier Transform." Journal of Computational Finance, 2(4),
        61-73.
        - The models built on this function agree with QuantLib's engines within the
          precision of the method stated above.

    As an example, the Black-Scholes model, whose log return is normal:

    ```python
    import numpy as np

    from financetoolkit.options.volatility import fourier_model

    def black_scholes(u):
        return np.exp(-0.5 * 0.2**2 * (1j * u + u**2))

    [
        round(fourier_model.get_fourier_price(
            black_scholes, 100, 110, 0.05, 1, method=method
        ), 6)
        for method in ("lewis", "cos", "fft")
    ]
    ```

    Which returns:

    ```
    [6.040088, 6.040088, 6.040088]
    ```
    """
    check_option("method", method, FOURIER_METHODS)

    strikes = np.atleast_1d(np.asarray(strike_price, dtype=float))
    if np.any(strikes <= 0):
        raise ValueError("Every strike price must be positive.")

    discounted_stock = stock_price * np.exp(-dividend_yield * time_to_expiration)
    discounted_strikes = strikes * np.exp(-risk_free_rate * time_to_expiration)

    if time_to_expiration <= 0:
        prices = np.maximum(
            (strikes - stock_price) if put_option else (stock_price - strikes), 0.0
        )
        return float(prices[0]) if np.ndim(strike_price) == 0 else prices

    forward_price = stock_price * np.exp(
        (risk_free_rate - dividend_yield) * time_to_expiration
    )
    log_moneyness = np.log(forward_price / strikes)
    discount = np.exp(-risk_free_rate * time_to_expiration)

    if method == "lewis":
        # The integrand decays like φ itself, and with the shift of the contour to Im u = -1/2
        # it has no pole at zero, so a single adaptive integral covers every strike at once.
        integral = quad_vec(
            lambda u: np.real(
                np.exp(1j * u * log_moneyness) * characteristic_function(u - 0.5j)
            )
            / (u * u + 0.25),
            0,
            np.inf,
            epsabs=LEWIS_TOLERANCE,
            epsrel=LEWIS_TOLERANCE,
            norm="max",
        )[0]
        calls = (
            discounted_stock
            - np.sqrt(forward_price * strikes) * discount / np.pi * integral
        )
        prices = calls - discounted_stock + discounted_strikes if put_option else calls

    elif method == "cos":
        # ln E[e^(sX)] at s = -2h, -h, 0, h and 2h, from φ(-is), in one evaluation.
        step = CUMULANT_STEP
        values = np.log(
            np.real(characteristic_function(-1j * step * np.arange(-2.0, 3.0)))
        )
        first = (values[3] - values[1]) / (2 * step)
        second = (values[3] - 2 * values[2] + values[1]) / step**2
        fourth = (
            values[4] - 4 * values[3] + 6 * values[2] - 4 * values[1] + values[0]
        ) / step**4
        half_width = COS_RANGE_WIDTH * np.sqrt(abs(second) + np.sqrt(abs(fourth)))

        # The range of y = ln(F / K) + X, the log of S_T / K: the same width for every strike,
        # so the frequencies u_k, and with them φ(u_k), are shared by the whole chain.
        lower = log_moneyness[:, None] + first - half_width
        width = 2 * half_width
        # The put pays K(1 - e^y) for y below zero, so its coefficients run over [a', min(0, b')].
        upper_end = np.minimum(0.0, lower + width)
        # A strike far below the range has a worthless put: its interval [a', 0] is empty.
        empty = (lower >= 0)[:, 0]

        # The series resolves the density with a number of terms that grows with the width of
        # the range in standard deviations of X, which the fourth cumulant of a fat tail widens
        # (Fang and Oosterlee 2008, section 5). The sum of the first N terms is compared with
        # that of 2N, both from one evaluation of φ at 2N frequencies, and N doubles until they
        # agree: the error of a converging series is about the difference.
        terms = (
            cos_terms
            if cos_tolerance is None
            else max(
                cos_terms,
                int(
                    2
                    ** np.ceil(
                        np.log2(
                            COS_TERMS_PER_DEVIATION
                            * width
                            / np.sqrt(max(abs(second), 1e-300))
                        )
                    )
                ),
            )
        )
        while True:
            frequencies = (
                np.arange(terms if cos_tolerance is None else 2 * terms) * np.pi / width
            )
            weights = characteristic_function(frequencies.astype(complex)) * np.exp(
                -1j * frequencies * (first - half_width)
            )
            weights[0] *= 0.5
            angle = frequencies * (upper_end - lower)
            exponential = np.exp(upper_end)
            chi = (
                np.cos(angle) * exponential
                - np.exp(lower)
                + frequencies * np.sin(angle) * exponential
            ) / (1 + frequencies**2)
            psi = np.empty_like(chi)
            psi[:, 0] = (upper_end - lower)[:, 0]
            psi[:, 1:] = np.sin(angle[:, 1:]) / frequencies[1:]
            coefficients = 2 / width * strikes[:, None] * (psi - chi)
            coefficients[empty] = 0.0
            terms_sum = np.real(weights[None, :] * coefficients)
            puts = discount * terms_sum.sum(axis=1)
            if (
                cos_tolerance is None
                or 2 * terms >= COS_MAXIMUM_TERMS
                or np.max(
                    np.abs(puts - discount * terms_sum[:, :terms].sum(axis=1)) / strikes
                )
                <= cos_tolerance
            ):
                break
            terms *= 2
        prices = puts if put_option else puts + discounted_stock - discounted_strikes

    else:
        frequencies = np.arange(fft_points) * fft_spacing
        log_strike_spacing = 2 * np.pi / (fft_points * fft_spacing)
        first_log_strike = np.log(forward_price) - fft_points * log_strike_spacing / 2
        shifted = frequencies - (fft_damping + 1) * 1j
        log_price_characteristic = np.exp(
            1j * shifted * np.log(forward_price)
        ) * characteristic_function(shifted)
        transform = (
            discount
            * log_price_characteristic
            / (
                fft_damping**2
                + fft_damping
                - frequencies**2
                + 1j * (2 * fft_damping + 1) * frequencies
            )
        )
        simpson = (3 + (-1.0) ** (np.arange(fft_points) + 1)) / 3
        simpson[0] = 1 / 3
        log_strikes = first_log_strike + log_strike_spacing * np.arange(fft_points)
        grid_calls = (
            np.exp(-fft_damping * log_strikes)
            / np.pi
            * np.real(
                np.fft.fft(
                    np.exp(-1j * first_log_strike * frequencies)
                    * transform
                    * fft_spacing
                    * simpson
                )
            )
        )
        calls = CubicSpline(log_strikes, grid_calls)(np.log(strikes))
        prices = calls - discounted_stock + discounted_strikes if put_option else calls

    # A price below the no-arbitrage bound max(S * e^(-qT) - K * e^(-rT), 0) (calls) or
    # max(K * e^(-rT) - S * e^(-qT), 0) (puts) is the integration error of a price close to the
    # bound, such as -1e-13 for a deep out-of-the-money option a day from expiration.
    prices = np.maximum(
        prices,
        np.maximum(
            (
                (discounted_strikes - discounted_stock)
                if put_option
                else (discounted_stock - discounted_strikes)
            ),
            0.0,
        ),
    )

    return float(prices[0]) if np.ndim(strike_price) == 0 else prices
