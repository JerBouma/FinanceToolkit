"""SVI (Stochastic Volatility Inspired) Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MINIMUM_OBSERVATIONS_FOR_SVI_FIT = 5

# SSVI shares rho, eta and gamma across the expiries (theta comes from the quotes).
MINIMUM_OBSERVATIONS_FOR_SSVI_FIT = 3


def get_svi_total_variance(
    log_moneyness: pd.Series | np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> pd.Series | np.ndarray:
    """
    Calculate the raw SVI (Stochastic Volatility Inspired, Gatheral 2004) total implied
    variance: the implied variance times the time to expiration as a function of
    log-moneyness, for a single expiry ("slice").

    The formula is as follows:

    - w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Where w is the total implied variance (Black-Scholes implied volatility squared,
    times time to expiration), k is the log-moneyness (ln(K / F), K the strike, F
    the forward price), a shifts the overall variance level, b controls the angle
    between the left and right wings, rho controls the skew/rotation, m shifts the
    curve horizontally and sigma controls the curvature (smoothness) at the money.

    See: Gatheral, J. (2004), "A parsimonious arbitrage-free implied volatility
    parameterization with application to the valuation of volatility derivatives".

    Args:
        log_moneyness (pd.Series | np.ndarray): ln(K / F) for each strike K, given
            forward price F.
        a (float): The overall variance level.
        b (float): The angle between the left and right wings. Must be >= 0.
        rho (float): The skew/rotation. Must be in (-1, 1).
        m (float): The horizontal shift of the curve.
        sigma (float): The curvature at the money. Must be > 0.

    Returns:
        pd.Series | np.ndarray: The total implied variance at each log-moneyness.
    """
    return a + b * (
        rho * (log_moneyness - m) + np.sqrt((log_moneyness - m) ** 2 + sigma**2)
    )


def get_svi_implied_volatility(
    log_moneyness: pd.Series | np.ndarray,
    time_to_expiration: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> pd.Series | np.ndarray:
    """
    Calculate the Black-Scholes implied volatility of a raw SVI slice: the volatility smile
    that the SVI total variance curve (see `get_svi_total_variance`) gives across
    log-moneyness.

    The formula is as follows:

    - Implied Volatility = sqrt(w(k) / t)

    Where w(k) is the SVI total variance at log-moneyness k, and t is the time to
    expiration, in years.

    Args:
        log_moneyness (pd.Series | np.ndarray): ln(K / F) for each strike K, given
            forward price F.
        time_to_expiration (float): The time to expiration, in years. Must be > 0.
        a (float): The overall variance level.
        b (float): The angle between the left and right wings.
        rho (float): The skew/rotation.
        m (float): The horizontal shift of the curve.
        sigma (float): The curvature at the money.

    Returns:
        pd.Series | np.ndarray: The Black-Scholes implied volatility at each
            log-moneyness.
    """
    total_variance = get_svi_total_variance(log_moneyness, a, b, rho, m, sigma)

    return np.sqrt(np.clip(total_variance, a_min=0, a_max=None) / time_to_expiration)


def get_svi_parameters(
    log_moneyness: pd.Series | np.ndarray,
    total_variance: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Calibrate the raw SVI parameters (a, b, rho, m, sigma) for a single expiry ("slice"):
    the SVI curve closest to a set of market total-variance observations, by minimizing the
    sum of squared differences between the SVI curve (see `get_svi_total_variance`) and the
    market total variance at each observed log-moneyness.

    Also known as: SVI calibration, SVI slice fit.

    Args:
        log_moneyness (pd.Series | np.ndarray): ln(K / F) for each observed strike
            K, given forward price F.
        total_variance (pd.Series | np.ndarray): The market total implied variance
            (implied volatility squared, times time to expiration) observed at each
            log-moneyness.

    Raises:
        ValueError: If fewer than 5 observations are given (the SVI slice has 5
            free parameters) or if the calibration fails to converge.

    Returns:
        dict[str, float]: The calibrated {"a", "b", "rho", "m", "sigma"} parameters.

    Notes:
        The fit enforces Gatheral's condition a + b * sigma * sqrt(1 - rho^2) >= 0: the
        minimum of the SVI curve, reached at k = m - rho * sigma / sqrt(1 - rho^2), is
        exactly that expression, so the condition keeps the total variance non-negative
        at every log-moneyness. Without it a least-squares fit can return a curve that
        dips below zero between or beyond the quoted strikes, where no implied volatility
        exists. The condition is imposed exactly by optimizing over
        c = a + b * sigma * sqrt(1 - rho^2) with the bound c >= 0 and recovering
        a = c - b * sigma * sqrt(1 - rho^2).

        The raw SVI least-squares surface is not convex and has a large flat region in
        which b collapses to zero, which a local optimizer started from a single guess
        will happily settle in -- returning a straight line through a pronounced smile
        while still reporting success. The fit is therefore run from a small grid of
        starting points spanning the plausible location (m), width (sigma) and skew
        (rho) of the slice, and the best of those runs is kept.
    """
    log_moneyness = np.asarray(log_moneyness, dtype=float)
    total_variance = np.asarray(total_variance, dtype=float)

    if len(log_moneyness) < MINIMUM_OBSERVATIONS_FOR_SVI_FIT:
        raise ValueError(
            "At least 5 (log-moneyness, total variance) observations are required "
            "to calibrate the 5 raw SVI parameters."
        )

    def to_raw_parameters(parameters: np.ndarray) -> tuple[float, ...]:
        # The optimizer works with the minimum total variance c instead of a, so the
        # bound c >= 0 enforces a + b * sigma * sqrt(1 - rho^2) >= 0 exactly.
        minimum_variance, b, rho, m, sigma = parameters
        a = minimum_variance - b * sigma * np.sqrt(1 - rho**2)
        return a, b, rho, m, sigma

    def objective(parameters: np.ndarray) -> float:
        a, b, rho, m, sigma = to_raw_parameters(parameters)
        model_variance = get_svi_total_variance(log_moneyness, a, b, rho, m, sigma)

        return float(np.sum((model_variance - total_variance) ** 2))

    # The minimum of the observed smile is the natural location for m, but a smile whose minimum sits at the edge of the quoted strikes needs the mid-point as a fallback.  # noqa: E501
    log_moneyness_span = max(log_moneyness.max() - log_moneyness.min(), 1e-6)
    # Each guess starts the curve's minimum (c) at the lowest observed total variance.
    initial_guesses = [
        [
            max(total_variance.min(), 1e-8),
            0.1,
            initial_rho,
            initial_m,
            initial_sigma,
        ]
        for initial_m in (
            float(log_moneyness[np.argmin(total_variance)]),
            float(np.mean(log_moneyness)),
        )
        for initial_sigma in (0.1 * log_moneyness_span, 0.5 * log_moneyness_span)
        for initial_rho in (-0.7, 0.0, 0.7)
    ]

    best_result = None

    for initial_guess in initial_guesses:
        result = minimize(
            objective,
            x0=initial_guess,
            method="L-BFGS-B",
            bounds=[
                (0, None),  # c = a + b * sigma * sqrt(1 - rho^2) >= 0
                (1e-6, None),  # b >= 0
                (-0.999, 0.999),  # rho in (-1, 1)
                (None, None),  # m
                (1e-6, None),  # sigma > 0
            ],
        )

        if result.success and (best_result is None or result.fun < best_result.fun):
            best_result = result

    if best_result is None:
        raise ValueError(
            "SVI calibration failed to converge from any of the "
            f"{len(initial_guesses)} starting points."
        )

    a, b, rho, m, sigma = to_raw_parameters(best_result.x)

    return {
        "a": float(a),
        "b": float(b),
        "rho": float(rho),
        "m": float(m),
        "sigma": float(sigma),
    }


def check_calendar_arbitrage(
    svi_parameters: dict[float, dict[str, float]],
    log_moneyness_grid: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Check a multi-expiry SVI surface for calendar-spread arbitrage: whether total implied
    variance is non-decreasing in time to expiration at every log-moneyness, as an
    arbitrage-free surface needs.

    A shorter-dated option can never carry more total variance than a longer-dated option on
    the same underlying, at the same log-moneyness, or a riskless calendar-spread arbitrage
    exists.

    See: Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility
    surfaces", Quantitative Finance, 14(1), 59-71.

    Args:
        svi_parameters (dict[float, dict[str, float]]): The calibrated SVI
            parameters (see `get_svi_parameters`) for each expiry, keyed by time to
            expiration in years.
        log_moneyness_grid (pd.Series | np.ndarray, optional): The log-moneyness
            grid to check the condition on. Defaults to None, which uses a grid from -1 to 1 (strikes
            from 63% below to 172% above the forward price, since e^-1 = 0.37 and
            e^1 = 2.72) with 200 points.

    Returns:
        pd.DataFrame: One row per log-moneyness/expiry-pair where the condition is
            violated, with the shorter and longer time to expiration, the
            log-moneyness and the (illegal) drop in total variance. Empty if the
            surface is free of calendar-spread arbitrage.
    """
    if log_moneyness_grid is None:
        log_moneyness_grid = np.linspace(-1, 1, 200)

    expirations = sorted(svi_parameters)

    violations = []
    for shorter_expiration, longer_expiration in zip(expirations, expirations[1:]):
        shorter_variance = get_svi_total_variance(
            log_moneyness_grid, **svi_parameters[shorter_expiration]
        )
        longer_variance = get_svi_total_variance(
            log_moneyness_grid, **svi_parameters[longer_expiration]
        )

        variance_drop = shorter_variance - longer_variance
        violating_points = variance_drop > 0

        for k, drop in zip(
            np.asarray(log_moneyness_grid)[violating_points],
            variance_drop[violating_points],
        ):
            violations.append(
                {
                    "Shorter Expiration": shorter_expiration,
                    "Longer Expiration": longer_expiration,
                    "Log-Moneyness": k,
                    "Total Variance Drop": drop,
                }
            )

    return pd.DataFrame(violations)


def get_svi_butterfly_function(
    log_moneyness: pd.Series | np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> pd.Series | np.ndarray:
    """
    Calculate the function g(k) of Gatheral and Jacquier (2014) for a raw SVI slice: whether
    the slice is free of butterfly arbitrage, as the density of the underlying at expiration
    implied by the slice is non-negative exactly where g(k) >= 0.

    The formula is as follows:

    - g(k) = (1 - k * w'(k) / (2 * w(k)))² - w'(k)² / 4 * (1 / w(k) + 1 / 4) + w''(k) / 2
    - w'(k) = b * (rho + (k - m) / sqrt((k - m)² + sigma²))
    - w''(k) = b * sigma² / ((k - m)² + sigma²)^(3/2)

    Where w(k) is the SVI total implied variance at log-moneyness k (see
    `get_svi_total_variance`) and a, b, rho, m and sigma the raw SVI parameters. The density
    of ln(S_T / F) equals g(k) / sqrt(2 * pi * w(k)) * exp(-d2(k)² / 2) with
    d2(k) = -k / sqrt(w(k)) - sqrt(w(k)) / 2, so a negative g means a negative density: a
    butterfly spread of calls around that strike would have a negative price.

    g(k) is also the denominator of Dupire's local variance in total variance terms (see
    `local_volatility_model.get_local_volatility`).

    See: Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces",
    Quantitative Finance, 14(1), 59-71, Section 2.

    Referenced against:
    - FinancePy, volatility_fns.g: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/volatility_fns.py#L247

    Args:
        log_moneyness (pd.Series | np.ndarray): ln(K / F) for each strike K, given forward
            price F.
        a (float): The overall variance level.
        b (float): The angle between the left and right wings.
        rho (float): The skew/rotation.
        m (float): The horizontal shift of the curve.
        sigma (float): The curvature at the money.

    Returns:
        pd.Series | np.ndarray: g(k) at each log-moneyness, negative where the slice has
            butterfly arbitrage.
    """
    shifted = log_moneyness - m
    root = np.sqrt(shifted**2 + sigma**2)

    total_variance = a + b * (rho * shifted + root)
    first_derivative = b * (rho + shifted / root)
    second_derivative = b * sigma**2 / root**3

    return (
        (1 - log_moneyness * first_derivative / (2 * total_variance)) ** 2
        - first_derivative**2 / 4 * (1 / total_variance + 0.25)
        + second_derivative / 2
    )


def check_butterfly_arbitrage(
    svi_parameters: dict[float, dict[str, float]],
    log_moneyness_grid: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Check every slice of an SVI surface for butterfly arbitrage: whether g(k) of Gatheral
    and Jacquier (2014) (see `get_svi_butterfly_function`) is non-negative at every
    log-moneyness, which keeps the implied density of the underlying non-negative and the
    call price convex in the strike.

    SSVI slices are checked the same way after converting them to raw SVI (see
    `convert_ssvi_to_svi`).

    See: Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces",
    Quantitative Finance, 14(1), 59-71.

    Args:
        svi_parameters (dict[float, dict[str, float]]): The raw SVI parameters (see
            `get_svi_parameters`) for each expiry, keyed by time to expiration in years.
        log_moneyness_grid (pd.Series | np.ndarray, optional): The log-moneyness grid to
            check the condition on. Defaults to None, which uses a grid from -1 to 1 with 200
            points, as `check_calendar_arbitrage` does.

    Returns:
        pd.DataFrame: One row per expiry and log-moneyness where g(k) is negative, with the
            time to expiration, the log-moneyness and g(k). Empty if every slice is free of
            butterfly arbitrage.
    """
    if log_moneyness_grid is None:
        log_moneyness_grid = np.linspace(-1, 1, 200)

    log_moneyness_grid = np.asarray(log_moneyness_grid, dtype=float)

    violations = []
    for expiration in sorted(svi_parameters):
        butterfly_function = get_svi_butterfly_function(
            log_moneyness_grid, **svi_parameters[expiration]
        )
        for k, value in zip(
            log_moneyness_grid[butterfly_function < 0],
            butterfly_function[butterfly_function < 0],
        ):
            violations.append(
                {"Expiration": expiration, "Log-Moneyness": k, "g(k)": value}
            )

    return pd.DataFrame(violations)


def get_ssvi_total_variance(
    log_moneyness: pd.Series | np.ndarray,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> pd.Series | np.ndarray:
    """
    Calculate the SSVI (Surface SVI) total implied variance of Gatheral and Jacquier (2014):
    the implied variance times the time to expiration across log-moneyness, with the
    power-law curvature function, for the expiry whose at-the-money total variance is theta.

    The formula is as follows:

    - w(k, θ) = θ / 2 * (1 + ρ * φ(θ) * k + sqrt((φ(θ) * k + ρ)² + 1 - ρ²))
    - φ(θ) = η / (θ^γ * (1 + θ)^(1 - γ))

    Where k is the log-moneyness ln(K / F), θ the at-the-money total implied variance of the
    expiry (σ_ATM² * t), ρ the skew shared by every expiry, and η and γ set how the curvature
    φ(θ) of the smile falls with the expiry. Three parameters describe the whole surface, and
    w(0, θ) = θ at every expiry.

    The surface is free of calendar-spread arbitrage when θ rises with the expiry (Theorem 4.1:
    the power law with 0 < γ <= 1/2 meets the conditions on φ), and free of butterfly
    arbitrage when η * (1 + |ρ|) <= 2 with 0 < γ <= 1/2 (Theorem 4.2 and the power-law example of Section 4).

    See: Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces",
    Quantitative Finance, 14(1), 59-71.

    Shares similar logic:
    - FinancePy, volatility_fns.ssvi: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/volatility_fns.py#L148

    Args:
        log_moneyness (pd.Series | np.ndarray): ln(K / F) for each strike K, given forward
            price F.
        theta (float): The at-the-money total implied variance θ of the expiry. Must be > 0.
        rho (float): The skew ρ, in (-1, 1).
        eta (float): The curvature level η. Must be > 0.
        gamma (float): The curvature decay γ, in (0, 1].

    Returns:
        pd.Series | np.ndarray: The total implied variance at each log-moneyness.
    """
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))

    return (
        theta
        / 2
        * (
            1
            + rho * phi * log_moneyness
            + np.sqrt((phi * log_moneyness + rho) ** 2 + 1 - rho**2)
        )
    )


def convert_ssvi_to_svi(
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> dict[str, float]:
    """
    Convert one SSVI slice (see `get_ssvi_total_variance`) to the raw SVI parameters: the
    SVI slice with the same total variance curve, so that the SVI functions (implied
    volatility, the arbitrage checks and the local volatility) apply to SSVI as well.

    The formula is as follows:

    - a = θ / 2 * (1 - ρ²)
    - b = θ * φ(θ) / 2
    - rho = ρ
    - m = -ρ / φ(θ)
    - sigma = sqrt(1 - ρ²) / φ(θ)

    See: Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces",
    Quantitative Finance, 14(1), 59-71, Section 3.

    Args:
        theta (float): The at-the-money total implied variance θ.
        rho (float): The skew ρ.
        eta (float): The curvature level η.
        gamma (float): The curvature decay γ.

    Returns:
        dict[str, float]: The raw SVI {"a", "b", "rho", "m", "sigma"} parameters.
    """
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))

    return {
        "a": float(theta / 2 * (1 - rho**2)),
        "b": float(theta * phi / 2),
        "rho": float(rho),
        "m": float(-rho / phi),
        "sigma": float(np.sqrt(1 - rho**2) / phi),
    }


def get_ssvi_parameters(
    log_moneyness: dict[float, np.ndarray],
    total_variance: dict[float, np.ndarray],
) -> dict[float, dict[str, float]]:
    """
    Calibrate an SSVI surface (see `get_ssvi_total_variance`) to the market total variance
    of several expiries at once: the surface free of butterfly arbitrage closest to the
    market's, with θ of each expiry its market at-the-money total variance and ρ, η and γ
    shared by every expiry, fitted by least squares on the implied volatility.

    θ is the market total variance at k = 0, interpolated linearly in the log-moneyness between
    the two quotes around the forward. The fit then keeps 0 < γ <= 1/2 and
    η * (1 + |ρ|) <= 2, the sufficient conditions of Gatheral and Jacquier (2014) for a surface
    free of butterfly arbitrage, by searching u in [0, 1] with η = 2 * u / (1 + |ρ|). The surface
    is then free of calendar-spread arbitrage too when θ rises with the expiry, which
    `check_calendar_arbitrage` confirms on the market's own at-the-money variances.

    The errors are taken in implied volatility rather than total variance, so the longer
    expiries, whose total variance is larger, do not outweigh the shorter ones.

    Also known as: SSVI calibration, surface SVI fit.

    See: Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces",
    Quantitative Finance, 14(1), 59-71.

    Args:
        log_moneyness (dict[float, np.ndarray]): ln(K / F) of the quotes of each expiry, keyed
            by time to expiration in years.
        total_variance (dict[float, np.ndarray]): The market total implied variance of the
            same quotes, keyed the same way.

    Raises:
        ValueError: If an expiry's quotes do not straddle the forward (no at-the-money total
            variance), fewer than 3 quotes are given in total or no starting point converges.

    Returns:
        dict[float, dict[str, float]]: The {"theta", "rho", "eta", "gamma"} parameters of each
            expiry, keyed by time to expiration in years; rho, eta and gamma are the same for
            every expiry.

    Notes:
        The fit runs L-BFGS-B with ρ in [-0.999, 0.999], u in [0.001, 1] and γ in [0.01, 0.5]
        from a fixed grid of twelve starting points (ρ of -0.5, 0 and 0.5, u of 0.25 and 0.75
        and γ of 0.25 and 0.5) and keeps the best, so the result is deterministic.
    """
    expirations = sorted(log_moneyness)

    thetas = {}
    for expiration in expirations:
        moneyness = np.asarray(log_moneyness[expiration], dtype=float)
        variance = np.asarray(total_variance[expiration], dtype=float)
        order = np.argsort(moneyness)
        moneyness, variance = moneyness[order], variance[order]

        if not moneyness[0] <= 0 <= moneyness[-1]:
            raise ValueError(
                f"The quotes of the expiry at {expiration:.4f} years do not straddle the "
                "forward, so its at-the-money total variance is unknown."
            )
        thetas[expiration] = float(np.interp(0.0, moneyness, variance))

    if (
        sum(len(log_moneyness[expiration]) for expiration in expirations)
        < MINIMUM_OBSERVATIONS_FOR_SSVI_FIT
    ):
        raise ValueError("At least 3 quotes are required to calibrate SSVI.")

    market_volatility = {
        expiration: np.sqrt(
            np.clip(np.asarray(total_variance[expiration], dtype=float), 0, None)
            / expiration
        )
        for expiration in expirations
    }

    def to_parameters(parameters: np.ndarray) -> tuple[float, float, float]:
        rho, eta_share, gamma = parameters
        return rho, 2 * eta_share / (1 + abs(rho)), gamma

    def objective(parameters: np.ndarray) -> float:
        rho, eta, gamma = to_parameters(parameters)
        error = 0.0
        for expiration in expirations:
            model_variance = get_ssvi_total_variance(
                np.asarray(log_moneyness[expiration], dtype=float),
                thetas[expiration],
                rho,
                eta,
                gamma,
            )
            model_volatility = np.sqrt(np.clip(model_variance, 0, None) / expiration)
            error += float(
                np.sum((model_volatility - market_volatility[expiration]) ** 2)
            )
        return error

    best_result = None
    for initial_rho in (-0.5, 0.0, 0.5):
        for initial_eta_share in (0.25, 0.75):
            for initial_gamma in (0.25, 0.5):
                result = minimize(
                    objective,
                    x0=[initial_rho, initial_eta_share, initial_gamma],
                    method="L-BFGS-B",
                    bounds=[(-0.999, 0.999), (1e-3, 1.0), (0.01, 0.5)],
                )
                if result.success and (
                    best_result is None or result.fun < best_result.fun
                ):
                    best_result = result

    if best_result is None:
        raise ValueError("SSVI calibration failed to converge from any starting point.")

    rho, eta, gamma = (float(value) for value in to_parameters(best_result.x))

    return {
        expiration: {
            "theta": thetas[expiration],
            "rho": rho,
            "eta": eta,
            "gamma": gamma,
        }
        for expiration in expirations
    }
