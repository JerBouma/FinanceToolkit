"""SABR (Stochastic Alpha Beta Rho) Model"""

__docformat__ = "google"

from typing import Literal

import numpy as np
from scipy.optimize import minimize

from financetoolkit.utilities.error_model import check_option

MINIMUM_OBSERVATIONS_FOR_SABR_FIT = 4

# Below this |z| the ratio z / x(z) is replaced by its expansion 1 - rho * z / 2, since
# x(z) itself goes to zero and the division loses all precision at the money.
SMALL_Z = 1e-7

# Below this |ln(F / K)| the Obłój leading term takes the series of Hagan's denominator,
# the limit of (F^(1 - beta) - K^(1 - beta)) / ((1 - beta) * ln(F / K)) at K = F.
SMALL_LOG_MONEYNESS = 1e-6

# The imaginary part below which a root of the cubic for alpha counts as real (rounding).
REAL_ROOT_TOLERANCE = 1e-10


def get_sabr_volatility(
    strike_price: float | np.ndarray,
    forward_price: float,
    time_to_expiration: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    method: Literal["hagan", "obloj"] = "hagan",
) -> float | np.ndarray:
    """
    Calculate the Black (lognormal) implied volatility of the SABR model: the volatility
    smile across strikes when the forward and its volatility follow correlated processes,
    with the asymptotic expansion of Hagan, Kumar, Lesniewski and Woodward (2002) or with
    the leading term corrected by Obłój (2008).

    The SABR model lets the forward price F follow dF = a * F^beta * dW with a stochastic
    volatility da = nu * a * dZ, where the two Brownian motions have correlation rho and a
    starts at alpha. Hagan et al. expand the implied volatility of a European option in the
    time to expiration, which gives a closed formula for the whole smile of one expiry: beta
    sets the backbone (how the at-the-money volatility moves with the forward), rho the skew
    and nu the curvature of the wings.

    The formula for the expansion of Hagan et al. is as follows:

    - σ(K) = α / ((F * K)^((1 - β) / 2) * D) * (z / x(z)) * (1 + I * T)
    - D = 1 + (1 - β)² / 24 * ln²(F / K) + (1 - β)⁴ / 1920 * ln⁴(F / K)
    - z = ν / α * (F * K)^((1 - β) / 2) * ln(F / K)
    - x(z) = ln((sqrt(1 - 2 * ρ * z + z²) + z - ρ) / (1 - ρ))
    - I = (1 - β)² / 24 * α² / (F * K)^(1 - β) + ρ * β * ν * α / (4 * (F * K)^((1 - β) / 2)) + (2 - 3 * ρ²) / 24 * ν²

    With Obłój's correction the leading term is replaced by:

    - σ(K) = ν * ln(F / K) / x(z) * (1 + I * T)
    - z = ν / α * (F^(1 - β) - K^(1 - β)) / (1 - β), or ν / α * ln(F / K) when β = 1

    Where K is the strike price, F the forward price, T the time to expiration in years, α
    the initial volatility, β the elasticity (0 is normal, 1 lognormal), ρ the correlation
    between the forward and its volatility and ν the volatility of the volatility. At the
    money z / x(z) equals 1 and both methods reduce to σ = α / F^(1 - β) * (1 + I * T). For
    β = 1 the two methods are the same. Obłój's leading term is exact in the limit T → 0
    (Berestycki, Busca and Florent 2004), while Hagan's is not for β < 1, which shows in the
    far wings.

    The shifted SABR model applies the same formula to F + s and K + s, so it allows
    negative forwards and strikes down to -s (rates); the volatility it returns is then the
    lognormal volatility of the shifted forward.

    See: Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002), "Managing
    smile risk", Wilmott Magazine, 84-108, and Obłój, J. (2008), "Fine-tune your smile:
    Correction to Hagan et al.", Wilmott Magazine.

    Referenced against:
    - QuantLib, sabrVolatility: https://github.com/lballabio/QuantLib/blob/v1.44/ql/termstructures/volatility/sabr.cpp#L163
    - QuantLib, shiftedSabrVolatility: https://github.com/lballabio/QuantLib/blob/v1.44/ql/termstructures/volatility/sabr.cpp#L182

    Shares similar logic:
    - FinancePy, SABR: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/sabr.py#L157
    - rateslib, IRSabrSmile: https://rateslib.com/py/en/2.7.x/api/rateslib.volatility.IRSabrSmile.html

    Args:
        strike_price (float | np.ndarray): The strike price(s) K.
        forward_price (float): The forward price F.
        time_to_expiration (float): The time to expiration T, in years.
        alpha (float): The initial volatility α. Must be > 0.
        beta (float): The elasticity β, between 0 and 1.
        rho (float): The correlation ρ between the forward and its volatility, in (-1, 1).
        nu (float): The volatility of the volatility ν. Must be >= 0.
        shift (float, optional): The shift s added to the forward and the strikes. Defaults
            to 0.0, the unshifted model.
        method (str, optional): The approximation. Options are: "hagan" (the expansion of Hagan et al., 2002),
            "obloj" (Obłój's, 2008, leading term). Defaults to "hagan".

    Returns:
        float | np.ndarray: The Black implied volatility at each strike (of the shifted
            forward when shift is not 0), NaN where F + s or K + s is not positive.
    """
    check_option("method", method, ("hagan", "obloj"))

    strike = np.asarray(strike_price, dtype=float) + shift
    forward = float(forward_price) + shift
    scalar_input = strike.ndim == 0
    strike = np.atleast_1d(strike)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_moneyness = np.log(forward / strike)
        one_minus_beta = 1.0 - beta
        forward_strike = (forward * strike) ** (one_minus_beta / 2)

        correction = 1 + time_to_expiration * (
            one_minus_beta**2 / 24 * alpha**2 / forward_strike**2
            + rho * beta * nu * alpha / (4 * forward_strike)
            + (2 - 3 * rho**2) / 24 * nu**2
        )

        # The series of the sinh-like term (F^(1 - β) - K^(1 - β)) / (1 - β) around ln(F / K) = 0.
        denominator = forward_strike * (
            1
            + one_minus_beta**2 / 24 * log_moneyness**2
            + one_minus_beta**4 / 1920 * log_moneyness**4
        )

        if method == "hagan":
            z = nu / alpha * forward_strike * log_moneyness
            leading = alpha / denominator
        elif beta == 1:
            z = nu / alpha * log_moneyness
            leading = np.full_like(strike, alpha)
        else:
            exact_denominator = (
                forward**one_minus_beta - strike**one_minus_beta
            ) / one_minus_beta
            z = nu / alpha * exact_denominator
            leading = np.where(
                np.abs(log_moneyness) < SMALL_LOG_MONEYNESS,
                alpha / denominator,
                alpha * log_moneyness / exact_denominator,
            )

        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        z_over_x = np.where(np.abs(z) < SMALL_Z, 1 - rho * z / 2, z / x)

        volatility = leading * z_over_x * correction

    volatility = np.where((strike > 0) & (forward > 0), volatility, np.nan)

    return float(volatility[0]) if scalar_input else volatility


def get_sabr_alpha(
    at_the_money_volatility: float,
    forward_price: float,
    time_to_expiration: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> float:
    """
    Calculate the SABR α that reproduces a given at-the-money Black volatility: the initial
    volatility at which the SABR smile passes through the market's at-the-money point, the
    smallest positive root of the cubic in α that the at-the-money SABR volatility gives
    (West 2005).

    The formula is as follows:

    - c3 * α³ + c2 * α² + c1 * α - σ_ATM * F^(1 - β) = 0
    - c3 = (1 - β)² * T / (24 * F^(2 - 2β))
    - c2 = ρ * β * ν * T / (4 * F^(1 - β))
    - c1 = 1 + (2 - 3ρ²) * ν² * T / 24

    Where σ_ATM is the at-the-money Black volatility, F the forward price (plus the shift),
    T the time to expiration in years and β, ρ and ν the other SABR parameters. Fixing α
    this way lets a fit of ρ and ν match the at-the-money volatility exactly.

    See: West, G. (2005), "Calibration of the SABR model in illiquid markets", Applied
    Mathematical Finance, 12(4), 371-385.

    Shares similar logic:
    - FinancePy, SABR.set_alpha_from_atm_black_vol: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/sabr.py#L283

    Args:
        at_the_money_volatility (float): The Black volatility at the forward.
        forward_price (float): The forward price F.
        time_to_expiration (float): The time to expiration T, in years.
        beta (float): The elasticity β, between 0 and 1.
        rho (float): The correlation ρ, in (-1, 1).
        nu (float): The volatility of the volatility ν.
        shift (float, optional): The shift s added to the forward. Defaults to 0.0.

    Raises:
        ValueError: If the cubic has no positive real root.

    Returns:
        float: The α at which the SABR at-the-money volatility equals the given one.
    """
    forward = float(forward_price) + shift
    one_minus_beta = 1.0 - beta
    forward_power = forward**one_minus_beta

    roots = np.roots(
        [
            one_minus_beta**2 * time_to_expiration / (24 * forward_power**2),
            rho * beta * nu * time_to_expiration / (4 * forward_power),
            1 + (2 - 3 * rho**2) * nu**2 * time_to_expiration / 24,
            -at_the_money_volatility * forward_power,
        ]
    )
    # A root with a tiny imaginary part from rounding is still a real root.
    positive_roots = [
        root.real
        for root in roots
        if abs(root.imag) < REAL_ROOT_TOLERANCE and root.real > 0
    ]

    if not positive_roots:
        raise ValueError(
            "No positive SABR alpha reproduces the at-the-money volatility of "
            f"{at_the_money_volatility}."
        )

    return float(min(positive_roots))


def get_sabr_parameters(
    strike_prices: np.ndarray,
    implied_volatility: np.ndarray,
    forward_price: float,
    time_to_expiration: float,
    beta: float = 1.0,
    shift: float = 0.0,
    method: Literal["hagan", "obloj"] = "hagan",
) -> dict[str, float]:
    """
    Calibrate the SABR parameters (alpha, rho, nu) of a single expiry ("slice") to the
    market implied volatilities with beta fixed: the SABR smile closest to the market's, by
    minimizing the sum of squared differences between the SABR volatility (see
    `get_sabr_volatility`) and the market volatility at each strike.

    The at-the-money volatility is the market volatility at the forward, interpolated linearly
    in the strike between the two quotes around it, and α follows from it, ρ and ν through the
    cubic of West (2005) (see `get_sabr_alpha`), so the fit only searches ρ and ν and matches
    the at-the-money volatility exactly. β is not fitted: it trades off against ρ (both tilt
    the smile), so it is chosen up front, 1 for equities (a lognormal backbone) and 0 or 0.5
    for rates.

    Also known as: SABR calibration, SABR slice fit.

    See: Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002), "Managing
    smile risk", Wilmott Magazine, and West, G. (2005), "Calibration of the SABR model in
    illiquid markets", Applied Mathematical Finance, 12(4).

    Shares similar logic:
    - QuantLib, SABRInterpolation: https://www.quantlib.org/reference/class_quant_lib_1_1_s_a_b_r_interpolation.html

    Args:
        strike_prices (np.ndarray): The strike price of each quote.
        implied_volatility (np.ndarray): The market Black implied volatility of each quote.
        forward_price (float): The forward price of the expiry.
        time_to_expiration (float): The time to expiration, in years.
        beta (float, optional): The elasticity β, between 0 and 1. Defaults to 1.0.
        shift (float, optional): The shift s added to the forward and the strikes. Defaults
            to 0.0.
        method (str, optional): The approximation, see `get_sabr_volatility`. Options are:
            "hagan" (the expansion of Hagan et al., 2002), "obloj" (Obłój's, 2008, leading
            term). Defaults to "hagan".

    Raises:
        ValueError: If fewer than 4 quotes are given, the forward lies outside the quoted
            strikes or no starting point converges.

    Returns:
        dict[str, float]: The calibrated {"alpha", "beta", "rho", "nu"} parameters.

    Notes:
        The fit runs L-BFGS-B with ρ in [-0.999, 0.999] and ν in [0.0001, 25] from a fixed grid
        of nine starting points (ρ of -0.5, 0 and 0.5 with ν of 0.3, 1 and 3) and keeps the
        best, so the result is deterministic and does not depend on a single starting guess.
    """
    strike_prices = np.asarray(strike_prices, dtype=float)
    implied_volatility = np.asarray(implied_volatility, dtype=float)

    if len(strike_prices) < MINIMUM_OBSERVATIONS_FOR_SABR_FIT:
        raise ValueError(
            "At least 4 (strike price, implied volatility) observations are required "
            "to calibrate SABR."
        )

    order = np.argsort(strike_prices)
    strike_prices, implied_volatility = strike_prices[order], implied_volatility[order]

    if not strike_prices[0] <= forward_price <= strike_prices[-1]:
        raise ValueError(
            f"The forward price {forward_price:.4f} lies outside the quoted strikes "
            f"({strike_prices[0]} to {strike_prices[-1]}), so there is no at-the-money "
            "volatility to fit SABR to."
        )

    at_the_money_volatility = float(
        np.interp(forward_price, strike_prices, implied_volatility)
    )

    def alpha_for(rho: float, nu: float) -> float:
        return get_sabr_alpha(
            at_the_money_volatility,
            forward_price,
            time_to_expiration,
            beta,
            rho,
            nu,
            shift,
        )

    def objective(parameters: np.ndarray) -> float:
        rho, nu = parameters
        try:
            alpha = alpha_for(rho, nu)
        except ValueError:
            return 1e10
        model_volatility = get_sabr_volatility(
            strike_prices,
            forward_price,
            time_to_expiration,
            alpha,
            beta,
            rho,
            nu,
            shift,
            method,
        )
        residuals = model_volatility - implied_volatility
        if not np.all(np.isfinite(residuals)):
            return 1e10
        return float(np.sum(residuals**2))

    best_result = None
    for initial_rho in (-0.5, 0.0, 0.5):
        for initial_nu in (0.3, 1.0, 3.0):
            result = minimize(
                objective,
                x0=[initial_rho, initial_nu],
                method="L-BFGS-B",
                bounds=[(-0.999, 0.999), (1e-4, 25.0)],
            )
            if result.success and (best_result is None or result.fun < best_result.fun):
                best_result = result

    if best_result is None:
        raise ValueError("SABR calibration failed to converge from any starting point.")

    rho, nu = (float(value) for value in best_result.x)

    return {
        "alpha": alpha_for(rho, nu),
        "beta": float(beta),
        "rho": rho,
        "nu": nu,
    }
