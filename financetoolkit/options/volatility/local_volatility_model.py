"""Local Volatility (Dupire) Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options.volatility import svi_model

MINIMUM_EXPIRATIONS_FOR_LOCAL_VOLATILITY = 2


def get_local_volatility(
    log_moneyness: pd.Series | np.ndarray,
    svi_parameters: dict[float, dict[str, float]],
) -> pd.DataFrame:
    """
    Calculate the Dupire (1994) local volatility at every expiry of an SVI volatility
    surface: the volatility of the stock at each price and time that reproduces every option
    price of the surface, from the total implied variance with the formula of Gatheral
    (2006, eq. 1.10).

    The local volatility σ_loc(K, T) is the volatility, as a function of the price and the
    time, of the only diffusion dS / S = (r - q) dt + σ_loc(S, t) dW that reprices every
    European option of the surface. Dupire showed it follows from the call prices through
    their derivatives in strike and expiry; written in the total implied variance
    w(k, T) = σ_BS²(k, T) * T and the log-moneyness k = ln(K / F_T), it reads as below.

    The formula is as follows:

    - σ_loc²(k, T) = (∂w / ∂T) / g(k)
    - g(k) = 1 - k / w * ∂w/∂k + 1/4 * (-1/4 - 1/w + k² / w²) * (∂w/∂k)² + 1/2 * ∂²w/∂k²

    Where the derivatives in k come analytically from each SVI slice and ∂w/∂T, taken at fixed
    k, by finite differences between the slices: the three-point formula for unequal spacing
    at the inner expiries, which is exact for a total variance quadratic in T, and the
    one-sided difference with the neighbouring expiry at the first and the last. The
    denominator g(k) is the butterfly function of Gatheral and Jacquier (2014) (see
    `svi_model.get_svi_butterfly_function`), so the local volatility exists exactly where the
    surface is free of butterfly arbitrage (g > 0) and of calendar-spread arbitrage
    (∂w/∂T >= 0); elsewhere it is NaN.

    The interest rate and the dividend yield enter only through the forward F_T in k, so the
    surface needs no separate rates.

    See: Dupire, B. (1994), "Pricing with a smile", Risk, 7(1), 18-20; Gatheral, J. (2006),
    "The Volatility Surface: A Practitioner's Guide", Wiley, eq. 1.10; and Gatheral, J., &
    Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces", Quantitative Finance, 14(1).

    Referenced against:
    - QuantLib, LocalVolSurface: https://www.quantlib.org/reference/class_quant_lib_1_1_local_vol_surface.html
    - FinancePy, volatility_fns.ssvi_local_varg: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/volatility_fns.py#L318

    Args:
        log_moneyness (pd.Series | np.ndarray): The log-moneyness ln(K / F_T) at which to
            evaluate the local volatility, the same for every expiry.
        svi_parameters (dict[float, dict[str, float]]): The raw SVI parameters of each expiry
            (see `svi_model.get_svi_parameters`, or `svi_model.convert_ssvi_to_svi` for SSVI),
            keyed by time to expiration in years.

    Raises:
        ValueError: If fewer than two expiries are given, since ∂w/∂T needs two.

    Returns:
        pd.DataFrame: The local volatility, indexed by log-moneyness, with one column per time
            to expiration in years; NaN where the surface has arbitrage.
    """
    expirations = sorted(svi_parameters)

    if len(expirations) < MINIMUM_EXPIRATIONS_FOR_LOCAL_VOLATILITY:
        raise ValueError(
            "The local volatility needs at least two expiries to take the derivative of the "
            "total variance in time."
        )

    log_moneyness = np.asarray(log_moneyness, dtype=float)
    times = np.asarray(expirations, dtype=float)

    total_variance = np.array(
        [
            svi_model.get_svi_total_variance(
                log_moneyness, **svi_parameters[expiration]
            )
            for expiration in expirations
        ]
    )

    time_derivative = np.empty_like(total_variance)
    time_derivative[0] = (total_variance[1] - total_variance[0]) / (times[1] - times[0])
    time_derivative[-1] = (total_variance[-1] - total_variance[-2]) / (
        times[-1] - times[-2]
    )
    for index in range(1, len(times) - 1):
        before = times[index] - times[index - 1]
        after = times[index + 1] - times[index]
        time_derivative[index] = (
            -after / (before * (before + after)) * total_variance[index - 1]
            + (after - before) / (before * after) * total_variance[index]
            + before / (after * (before + after)) * total_variance[index + 1]
        )

    local_volatility = {}
    for index, expiration in enumerate(expirations):
        butterfly_function = svi_model.get_svi_butterfly_function(
            log_moneyness, **svi_parameters[expiration]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            local_variance = time_derivative[index] / butterfly_function
        local_volatility[expiration] = np.where(
            (butterfly_function > 0) & (time_derivative[index] >= 0),
            np.sqrt(np.abs(local_variance)),
            np.nan,
        )

    result = pd.DataFrame(local_volatility, index=log_moneyness)
    result.index.name = "Log-Moneyness"

    return result
