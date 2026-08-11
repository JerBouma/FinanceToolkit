"""SVI (Stochastic Volatility Inspired) Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments

MINIMUM_OBSERVATIONS_FOR_SVI_FIT = 5


def get_svi_total_variance(
    log_moneyness: pd.Series | np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> pd.Series | np.ndarray:
    """
    Evaluate the raw SVI (Stochastic Volatility Inspired, Gatheral 2004) parametric
    curve for total implied variance as a function of log-moneyness, for a single
    expiry ("slice").

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
    Convert the raw SVI total variance curve (see `get_svi_total_variance`) back to
    Black-Scholes implied volatility.

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
    Calibrate the raw SVI parameters (a, b, rho, m, sigma) for a single expiry
    ("slice") to a set of market total-variance observations, by minimizing the sum
    of squared differences between the SVI curve (see `get_svi_total_variance`) and
    the market total variance at each observed log-moneyness.

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

    def objective(parameters: np.ndarray) -> float:
        a, b, rho, m, sigma = parameters
        model_variance = get_svi_total_variance(log_moneyness, a, b, rho, m, sigma)

        return float(np.sum((model_variance - total_variance) ** 2))

    # The minimum of the observed smile is the natural location for m, but a smile whose
    # minimum sits at the edge of the quoted strikes needs the mid-point as a fallback.
    log_moneyness_span = max(log_moneyness.max() - log_moneyness.min(), 1e-6)
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
                (None, None),  # a
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

    a, b, rho, m, sigma = best_result.x

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
    Check a multi-expiry SVI surface for calendar-spread arbitrage -- for a surface
    to be arbitrage-free, total implied variance must be non-decreasing in time to
    expiration at every log-moneyness (a shorter-dated option can never carry more
    total variance than a longer-dated option on the same underlying, at the same
    log-moneyness, or a riskless calendar-spread arbitrage exists).

    See: Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility
    surfaces", Quantitative Finance, 14(1), 59-71.

    Args:
        svi_parameters (dict[float, dict[str, float]]): The calibrated SVI
            parameters (see `get_svi_parameters`) for each expiry, keyed by time to
            expiration in years.
        log_moneyness_grid (pd.Series | np.ndarray, optional): The log-moneyness
            grid to check the condition on. Defaults to a grid from -1 to 1 (roughly
            37% below to 172% above the forward price) with 200 points.

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
