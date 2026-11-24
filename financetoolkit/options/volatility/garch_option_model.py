"""Heston-Nandi GARCH Option Pricing Model"""

__docformat__ = "google"

import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.volatility.fourier_model import get_fourier_price
from financetoolkit.risk.volatility.garch_model import get_backcast

TRADING_DAYS_PER_YEAR = 252

MINIMUM_OBSERVATIONS_FOR_HESTON_NANDI_FIT = 100

# The value of the objective outside the stationary region (or where the likelihood is not
# finite), far above any negative log-likelihood per observation.
INFEASIBLE_OBJECTIVE = 1e10

# The bounds of the fit on the scale of the sample variance s² of the excess returns:
# (ω / s², α / s², β, γ * s, λ * s), all of the order of one. Typical daily estimates on
# stocks and indices (α ≈ 1e-6, γ ≈ 100-500 and λ ≈ 0-5 with s ≈ 0.01) lie well inside.
HESTON_NANDI_BOUNDS = (
    np.array([1e-8, 1e-8, 0.0, -50.0, -1.0]),
    np.array([1.0, 1.0, 0.9999, 50.0, 1.0]),
)

# The fixed starting points of the fit, on the same scale: a persistence of about 0.9 split
# between β and αγ², without and with a leverage effect.
HESTON_NANDI_STARTS = (
    (0.02, 0.01, 0.90, 0.0, 0.0),
    (0.02, 0.015, 0.85, 2.0, 0.0),
)


def get_heston_nandi_log_likelihood(
    excess_returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    risk_premium: float,
    first_variance: float,
) -> tuple[float, float]:
    """
    Calculate the Gaussian log-likelihood of the Heston and Nandi (2000) GARCH(1, 1) model:
    how well its parameters explain a series of excess returns, and the variance it
    forecasts for the period after the last return.

    The formulas are as follows:

    - y_t = λ * h_t + √h_t * z_t
    - h_(t+1) = ω + β * h_t + α * (z_t — γ * √h_t)²
    - log-likelihood = —½ * Σ_t (ln(2π) + ln(h_t) + z_t²)

    Where y_t is the log return over the risk-free rate of period t, h_t its conditional
    variance, z_t the standard normal shock, ω, α and β the GARCH weights, γ the asymmetry
    (the leverage effect: a positive γ raises the variance more after a fall) and λ the price
    of risk.

    Args:
        excess_returns (np.ndarray): The log returns over the risk-free rate, per period.
        omega (float): The constant ω.
        alpha (float): The weight α of the shock.
        beta (float): The weight β of the previous variance.
        gamma (float): The asymmetry γ.
        risk_premium (float): The price of risk λ.
        first_variance (float): The variance h_1 of the first period.

    Returns:
        tuple[float, float]: The log-likelihood and the variance of the next period.

    As an example, five daily excess returns:

    ```python
    import numpy as np

    from financetoolkit.options.volatility import garch_option_model

    log_likelihood, next_variance = garch_option_model.get_heston_nandi_log_likelihood(
        np.array([0.01, -0.02, 0.005, -0.01, 0.015]), omega=1e-6, alpha=3e-6, beta=0.9,
        gamma=150.0, risk_premium=2.0, first_variance=1e-4,
    )

    round(log_likelihood, 4), round(next_variance, 8)
    ```

    Which returns:

    ```
    (14.075, 0.00011283)
    ```
    """
    returns = np.asarray(excess_returns, dtype=float)
    # (z_t — γ√h_t)² = (y_t — (λ + γ)h_t)² / h_t, so the recursion needs no square root, and
    # with ω > 0 and α, β ≥ 0 every variance is positive.
    shift = risk_premium + gamma
    variances = [first_variance]
    variance = first_variance
    for excess_return in returns.tolist():
        deviation = excess_return - shift * variance
        variance = omega + beta * variance + alpha * deviation * deviation / variance
        variances.append(variance)

    path = np.array(variances[:-1])
    log_likelihood = -0.5 * float(
        np.sum(np.log(2 * np.pi * path) + (returns - risk_premium * path) ** 2 / path)
    )

    return log_likelihood, variances[-1]


def get_heston_nandi_parameters(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float | pd.Series | np.ndarray = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """
    Fit the GARCH(1, 1) model of Heston and Nandi (2000) to a series of log returns by
    maximum likelihood: the variance dynamics, with a leverage effect, that best explain the
    returns and whose European option prices have a closed form (see
    get_heston_nandi_price).

    The variance responds to the shock z_t shifted by γ√h_t, so with a positive γ a fall
    raises the variance more than a rise of the same size (the leverage effect), and the
    expected excess return λ * h_t is the price of risk of the variance.

    The formulas are as follows:

    - y_t = R_t — r / periods_per_year = λ * h_t + √h_t * z_t
    - h_(t+1) = ω + β * h_t + α * (z_t — γ * √h_t)²
    - maximize —½ * Σ_t (ln(2π) + ln(h_t) + z_t²), subject to β + αγ² < 1
    - Unconditional variance = (ω + α) / (1 — β — αγ²)

    Where R_t is the log return of period t, r the continuously compounded annual risk-free
    rate, y_t the excess return, h_t its conditional variance, z_t the standard normal shock,
    ω, α and β the GARCH weights, γ the asymmetry and λ the price of risk. The recursion
    starts from the backcast of the GARCH models of the Toolkit (an exponentially weighted
    average of the first 75 squared excess returns). The fit runs on the parameters scaled by
    the sample variance s² of the excess returns (ω / s², α / s², β, γ * s, λ * s), from two
    fixed starting points with L-BFGS-B within bounds, and keeps the best.

    Args:
        returns (pd.Series | np.ndarray): The log returns, one per period (daily by default).
            Missing values (and returns without a rate) are dropped.
        risk_free_rate (float | pd.Series | np.ndarray, optional): The continuously
            compounded annual risk-free rate, spread evenly over the periods of a year: one
            rate, or one per return. Defaults to 0.
        periods_per_year (int, optional): The number of periods in a year. Defaults to 252.

    Returns:
        dict[str, float]: omega, alpha, beta, gamma and lambda (the risk premium), the
        persistence β + αγ², the unconditional and next-period variance (per period) and the
        log-likelihood.

    Raises:
        ValueError: If there are fewer than 100 returns or no fit converges.

    Notes:
        References: Heston, S. L. and Nandi, S. (2000). "A Closed-Form GARCH Option Valuation
        Model." The Review of Financial Studies, 13(3), 585-625. Checked against a fit by
        direct likelihood maximization with Nelder-Mead from the recovered optimum on
        simulated Heston-Nandi returns, which recovers the simulated parameters within their
        standard errors.

    As an example:

    ```python
    import numpy as np

    from financetoolkit.options.volatility import garch_option_model

    rng = np.random.default_rng(1)
    omega, alpha, beta, gamma, risk_premium = 5e-7, 1.3e-6, 0.89, 250.0, 2.0
    variance, returns = 1e-4, []
    for shock in rng.standard_normal(3000):
        returns.append(risk_premium * variance + np.sqrt(variance) * shock)
        variance = omega + beta * variance + alpha * (shock - gamma * np.sqrt(variance)) ** 2

    parameters = garch_option_model.get_heston_nandi_parameters(np.array(returns))
    [f"{parameters[name]:.4g}" for name in ("omega", "alpha", "beta", "gamma", "lambda")]
    ```

    Which returns:

    ```
    ['7.921e-07', '1.181e-06', '0.8581', '306.3', '1.446']
    ```
    """
    values = np.asarray(returns, dtype=float)
    rates = np.broadcast_to(np.asarray(risk_free_rate, dtype=float), values.shape)
    observed = np.isfinite(values) & np.isfinite(rates)
    values, rates = values[observed], rates[observed]

    if len(values) < MINIMUM_OBSERVATIONS_FOR_HESTON_NANDI_FIT:
        raise ValueError(
            f"The Heston-Nandi fit needs at least {MINIMUM_OBSERVATIONS_FOR_HESTON_NANDI_FIT} "
            f"returns, received {len(values)}."
        )

    excess_returns = values - rates / periods_per_year
    scale = float(np.var(excess_returns, ddof=1))
    deviation = math.sqrt(scale)
    first_variance = get_backcast(excess_returns)

    def unscale(parameters: np.ndarray) -> tuple[float, float, float, float, float]:
        return (
            parameters[0] * scale,
            parameters[1] * scale,
            parameters[2],
            parameters[3] / deviation,
            parameters[4] / deviation,
        )

    def objective(parameters: np.ndarray) -> float:
        # β + αγ² is β + (α / s²)(γs)², so the stationarity condition holds on either scale.
        if parameters[2] + parameters[1] * parameters[3] ** 2 >= 1:
            return INFEASIBLE_OBJECTIVE
        log_likelihood, _ = get_heston_nandi_log_likelihood(
            excess_returns, *unscale(parameters), first_variance
        )
        # Per observation, so the tolerances of the search do not depend on the sample size.
        return (
            -log_likelihood / len(excess_returns)
            if math.isfinite(log_likelihood)
            else INFEASIBLE_OBJECTIVE
        )

    lower, upper = HESTON_NANDI_BOUNDS
    best = None
    for start in HESTON_NANDI_STARTS:
        result = minimize(
            objective,
            np.array(start),
            method="L-BFGS-B",
            bounds=list(zip(lower, upper, strict=True)),
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
        )
        if result.fun < INFEASIBLE_OBJECTIVE and (
            best is None or result.fun < best.fun
        ):
            best = result

    if best is None:
        raise ValueError(
            "The Heston-Nandi fit did not converge from any starting point."
        )

    omega, alpha, beta, gamma, risk_premium = (
        float(value) for value in unscale(best.x)
    )
    log_likelihood, next_variance = get_heston_nandi_log_likelihood(
        excess_returns, omega, alpha, beta, gamma, risk_premium, first_variance
    )
    persistence = beta + alpha * gamma**2

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lambda": risk_premium,
        "persistence": persistence,
        "unconditional_variance": (omega + alpha) / (1 - persistence),
        "next_variance": next_variance,
        "log_likelihood": log_likelihood,
    }


def get_heston_nandi_price(
    stock_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    periods_to_expiration: int,
    next_variance: float,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    risk_premium: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    method: Literal["lewis", "cos", "fft"] = "lewis",
) -> float | np.ndarray:
    """
    Calculate the price of a European option in the GARCH(1, 1) model of Heston and Nandi
    (2000): what a call or put is worth when the variance of the stock follows a GARCH
    process with a leverage effect, the GARCH option pricing model with a closed form.

    Its parameters can be fitted on the stock's own returns (see
    get_heston_nandi_parameters), and the price follows from the moment generating function
    of the log price by Fourier inversion.

    Under the risk-neutral measure the price of risk λ becomes —½ and the asymmetry γ becomes
    γ* = γ + λ + ½. The moment generating function f(φ) = E[S_T^φ] = S^φ * exp(A + B * h)
    then follows from a backward recursion over the periods to expiration, and the
    characteristic function of X = ln(S_T / F) is priced with the Fourier pricer of the
    Toolkit (see fourier_model.get_fourier_price).

    The formulas are as follows:

    - A_t = A_(t+1) + φ * (r — q) / periods_per_year + B_(t+1) * ω — ½ * ln(1 — 2α * B_(t+1))
    - B_t = φ * (λ* + γ*) — ½ * γ*² + β * B_(t+1) + ½ * (φ — γ*)² / (1 — 2α * B_(t+1))
    - A_T = B_T = 0, λ* = —½, γ* = γ + λ + ½
    - Call = S * e^(—qT) — √(F * K) * e^(—rT) / π * ∫_0^∞ Re[e^(iu * ln(F / K)) * φ_X(u — i/2)] / (u² + 1/4) du

    Where S is the stock price, K is the strike price, r is the continuously compounded annual
    risk-free rate, q is the continuous dividend yield, T = periods_to_expiration /
    periods_per_year, F = S * e^((r — q)T) the forward price, h the variance of the next
    period, ω, α, β, γ and λ the parameters of get_heston_nandi_parameters (per period) and
    φ_X(u) = f(iu) / F^(iu). With α = 0 and β = 0 the variance is the constant ω, and the
    price is the Black-Scholes price with volatility √(ω * periods_per_year).

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The strike price, or an array of strike prices.
        risk_free_rate (float): The continuously compounded annual risk-free rate.
        periods_to_expiration (int): The number of periods (trading days by default) to
            expiration. At zero the option is worth its exercise value.
        next_variance (float): The conditional variance h of the next period.
        omega (float): The constant ω of the variance, per period.
        alpha (float): The weight α of the shock.
        beta (float): The weight β of the previous variance.
        gamma (float): The asymmetry γ.
        risk_premium (float): The price of risk λ.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether to price a put instead of a call. Defaults to False.
        periods_per_year (int, optional): The number of periods in a year. Defaults to 252.
        method (str, optional): The Fourier pricer. Options are: "lewis" (Lewis' integral),
            "cos" (the COS method), "fft" (the Carr-Madan fast Fourier transform). Defaults to
            "lewis".

    Returns:
        float | np.ndarray: The option price, a float for a single strike price and an array
        for an array of strike prices.

    Raises:
        TypeError: If a scalar input is not a float or int.
        ValueError: If the variance, ω, α or β is negative.

    Notes:
        References: Heston, S. L. and Nandi, S. (2000). "A Closed-Form GARCH Option
        Valuation Model." The Review of Financial Studies, 13(3), 585-625. Checked
        against a Monte Carlo simulation of the risk-neutral process (1,000,000 seeded paths,
        within three standard errors) and the Black-Scholes price at α = β = 0 (within
        1e-12). The dividend yield enters as a lower drift, which the paper leaves out.

    As an example:

    ```python
    from financetoolkit.options.volatility import garch_option_model

    round(
        garch_option_model.get_heston_nandi_price(
            stock_price=100, strike_price=95, risk_free_rate=0.04, periods_to_expiration=63,
            next_variance=1.5e-4, omega=5e-7, alpha=1.3e-6, beta=0.89, gamma=250.0,
            risk_premium=2.0, put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    1.1667
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        risk_free_rate=risk_free_rate,
        periods_to_expiration=periods_to_expiration,
        next_variance=next_variance,
        omega=omega,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        risk_premium=risk_premium,
        dividend_yield=dividend_yield,
    )
    if min(next_variance, omega, alpha, beta) < 0:
        raise ValueError("next_variance, omega, alpha and beta must be 0 or above.")

    risk_neutral_gamma = gamma + risk_premium + 0.5
    steps = int(periods_to_expiration)

    def characteristic_function(u: np.ndarray) -> np.ndarray:
        # The drift (r - q) per period only moves the forward, which X = ln(S_T / F) takes
        # out again, so the recursion runs without it.
        phi = 1j * np.asarray(u, dtype=complex)
        a = np.zeros_like(phi)
        b = np.zeros_like(phi)
        for _ in range(steps):
            denominator = 1 - 2 * alpha * b
            a = a + b * omega - 0.5 * np.log(denominator)
            b = (
                phi * (risk_neutral_gamma - 0.5)
                - 0.5 * risk_neutral_gamma**2
                + beta * b
                + 0.5 * (phi - risk_neutral_gamma) ** 2 / denominator
            )
        return np.exp(a + b * next_variance)

    return get_fourier_price(
        characteristic_function,
        stock_price,
        strike_price,
        risk_free_rate,
        steps / periods_per_year,
        dividend_yield,
        put_option,
        method=method,
    )
