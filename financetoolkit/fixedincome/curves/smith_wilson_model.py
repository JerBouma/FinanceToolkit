"""Smith-Wilson Extrapolation Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# EIOPA's convergence criterion: the forward intensity at the convergence point lies
# within a basis point of the ultimate forward intensity, with alpha at least 0.05. The
# search for alpha stops at 1, far beyond the values EIOPA publishes (about 0.05 to 0.2).
CONVERGENCE_TOLERANCE = 0.0001
MINIMUM_ALPHA = 0.05
MAXIMUM_ALPHA = 1.0


def _wilson(
    years: np.ndarray, cash_flow_times: np.ndarray, alpha: float, omega: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    The Wilson function W(t, u) for every time t (rows) and cash flow time u (columns), and
    its derivative to t:

        W(t, u) = e^(-ω (t + u)) [α min(t, u) - e^(-α max(t, u)) sinh(α min(t, u))]
    """
    t = years[:, None]
    u = cash_flow_times[None, :]
    low, high = np.minimum(t, u), np.maximum(t, u)
    scale = np.exp(-omega * (t + u))
    shape = alpha * low - np.exp(-alpha * high) * np.sinh(alpha * low)
    # Before the cash flow the bracket grows with t through min(t, u), after it through
    # max(t, u).
    slope = np.where(
        t < u,
        alpha - alpha * np.exp(-alpha * u) * np.cosh(alpha * t),
        alpha * np.exp(-alpha * t) * np.sinh(alpha * u),
    )
    return scale * shape, scale * (slope - omega * shape)


def _coefficients(
    cash_flow_times: np.ndarray,
    cash_flows: np.ndarray,
    prices: np.ndarray,
    alpha: float,
    omega: float,
) -> np.ndarray:
    """
    The weight ζ_j of every cash flow time in P(t) = e^(-ω t) + Σ_j ζ_j W(t, u_j), so that
    every instrument is priced exactly: ζ = Cᵀ (C W Cᵀ)⁻¹ (m - C μ) with μ_j = e^(-ω u_j).
    """
    weights, _ = _wilson(cash_flow_times, cash_flow_times, alpha, omega)
    gaps = prices - cash_flows @ np.exp(-omega * cash_flow_times)
    return cash_flows.T @ np.linalg.solve(cash_flows @ weights @ cash_flows.T, gaps)


def get_smith_wilson_curve(
    cash_flow_times: np.ndarray | list,
    cash_flows: np.ndarray | list,
    prices: np.ndarray | list,
    years: np.ndarray | list | float,
    ultimate_forward_rate: float,
    convergence_speed: float,
) -> pd.DataFrame:
    """
    Build a discount curve with the Smith-Wilson method (Smith and Wilson, 2001): the
    risk-free term structure that prices every input instrument exactly and, beyond the last
    of them, converges to an ultimate forward rate (UFR) at a speed set by the convergence
    speed α (EIOPA's alpha), the method Solvency II prescribes for the risk-free rate term
    structures of EIOPA.

    The formula is as follows:

    - P(t) = e^(-ω t) + Σ_j ζ_j W(t, u_j)
    - W(t, u) = e^(-ω (t + u)) [α min(t, u) - ½ e^(-α max(t, u)) (e^(α min(t, u)) - e^(-α min(t, u)))]
    - ζ = Cᵀ (C W Cᵀ)⁻¹ (m - C μ), with W_jk = W(u_j, u_k) and μ_j = e^(-ω u_j)
    - f(t) = -P'(t) / P(t), the instantaneous forward rate

    where ω = ln(1 + UFR) is the ultimate forward intensity, u_j the cash flow times, C the
    cash flows of every instrument (rows) at those times (columns) and m the instruments'
    prices. A zero-coupon bond paying 1 at u is a row with a 1 at u and price P(u); a par
    swap or par bond with coupon c paid f times a year is a row of c / f at every coupon time
    and 1 + c / f at maturity, with price 1. Smith-Wilson keeps the curve exactly on the
    inputs; between and beyond them it is the smoothest curve (in the sense of Smith and
    Wilson) that does, and the forward rate tends to the UFR as t grows.

    The zero rates are annually compounded, as EIOPA publishes them, r(t) = P(t)^(-1/t) - 1,
    and the instantaneous forward rate is continuously compounded, so it converges to
    ω = ln(1 + UFR).

    For more information, see the following sources:

    - Smith, A. and Wilson, T. (2001). "Fitting Yield Curves with Long Term Constraints."
      Research Notes, Bacon and Woodrow.
    - EIOPA (2022). "Technical Documentation of the Methodology to Derive EIOPA's Risk-Free
      Interest Rate Term Structures." EIOPA-BoS-22/409, section on the Smith-Wilson method.

    EIOPA's published curves and Smith-Wilson coefficients (Qb) are reproduced, see
    tests/fixedincome/test_curve_construction.py::test_smith_wilson_reproduces_eiopa.

    Referenced against:
    - rateslib, SmithWilsonCurve: https://rateslib.com/py/en/3.0.x/api/rateslib.curves.SmithWilsonCurve.html

    Args:
        cash_flow_times (np.ndarray | list): The times of the cash flows in years, after 0.
        cash_flows (np.ndarray | list): The cash flows, a row per instrument and a column per
            cash flow time.
        prices (np.ndarray | list): The price of every instrument, per 1 of notional.
        years (np.ndarray | list | float): The maturities to return the curve at, in years.
        ultimate_forward_rate (float): The UFR, annually compounded (decimal), e.g. 0.033.
        convergence_speed (float): The speed α of convergence to the UFR, EIOPA's alpha,
            above 0; see `get_smith_wilson_alpha` for EIOPA's choice.

    Returns:
        pd.DataFrame: The Zero Rate (annually compounded), the Discount Factor and the
        Instantaneous Forward Rate (continuously compounded) by maturity in years.

    Raises:
        ValueError: If the convergence speed is not positive or the cash flows do not fit the
            prices and times.
    """
    if convergence_speed <= 0:
        raise ValueError("The convergence speed must be greater than 0.")
    cash_flow_times = np.asarray(cash_flow_times, dtype=float)
    cash_flows = np.atleast_2d(np.asarray(cash_flows, dtype=float))
    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    if cash_flows.shape != (len(prices), len(cash_flow_times)):
        raise ValueError(
            "The cash flows need a row per instrument (price) and a column per cash flow time."
        )
    if np.any(cash_flow_times <= 0):
        raise ValueError("The cash flow times must be after today.")
    years = np.atleast_1d(np.asarray(years, dtype=float))
    omega = np.log1p(ultimate_forward_rate)

    coefficients = _coefficients(
        cash_flow_times, cash_flows, prices, convergence_speed, omega
    )
    wilson, slope = _wilson(years, cash_flow_times, convergence_speed, omega)
    discount = np.exp(-omega * years) + wilson @ coefficients
    derivative = -omega * np.exp(-omega * years) + slope @ coefficients

    with np.errstate(divide="ignore", invalid="ignore"):
        zero_rates = np.where(years > 0, discount ** (-1 / years) - 1, np.nan)

    return pd.DataFrame(
        {
            "Zero Rate": zero_rates,
            "Discount Factor": discount,
            "Instantaneous Forward Rate": -derivative / discount,
        },
        index=pd.Index(years, name="Maturity"),
    )


def get_smith_wilson_alpha(
    cash_flow_times: np.ndarray | list,
    cash_flows: np.ndarray | list,
    prices: np.ndarray | list,
    ultimate_forward_rate: float,
    convergence_point: float,
    tolerance: float = CONVERGENCE_TOLERANCE,
) -> float:
    """
    Calculate the alpha of the Smith-Wilson method as EIOPA sets it: the smallest alpha of
    at least 0.05 for which the instantaneous forward rate (continuously compounded) at the
    convergence point lies within the tolerance (a basis point) of the ultimate forward
    intensity ω = ln(1 + UFR).

    The formula is as follows:

    - α = min {α ≥ 0.05 : |f(T) - ω| ≤ tolerance}

    where f(T) is the forward of `get_smith_wilson_curve` at the convergence point T, which
    EIOPA sets at the later of 60 years and the last liquid point plus 40 years. A higher
    alpha converges faster but bends the curve more just after the last liquid point, so
    EIOPA takes the lowest alpha that meets the criterion. The gap |f(T) - ω| falls as alpha
    rises; the alpha is found with Brent's method to 1e-10.

    For more information, see: EIOPA (2022). "Technical Documentation of the Methodology to
    Derive EIOPA's Risk-Free Interest Rate Term Structures", section on the convergence
    criterion.

    Args:
        cash_flow_times (np.ndarray | list): The times of the cash flows in years, after 0.
        cash_flows (np.ndarray | list): The cash flows, a row per instrument and a column per
            cash flow time.
        prices (np.ndarray | list): The price of every instrument, per 1 of notional.
        ultimate_forward_rate (float): The UFR, annually compounded (decimal).
        convergence_point (float): The maturity, in years, at which the forward must be
            within the tolerance of the UFR.
        tolerance (float, optional): The largest gap between the forward intensity and ω at
            the convergence point. Defaults to 0.0001 (a basis point).

    Returns:
        float: The alpha.

    Raises:
        ValueError: If no alpha up to 1 meets the criterion.
    """
    omega = np.log1p(ultimate_forward_rate)

    def gap(alpha: float) -> float:
        forward = get_smith_wilson_curve(
            cash_flow_times,
            cash_flows,
            prices,
            convergence_point,
            ultimate_forward_rate,
            alpha,
        )["Instantaneous Forward Rate"].iloc[0]
        return abs(forward - omega) - tolerance

    if gap(MINIMUM_ALPHA) <= 0:
        return MINIMUM_ALPHA
    if gap(MAXIMUM_ALPHA) > 0:
        raise ValueError(
            f"No alpha up to {MAXIMUM_ALPHA} brings the forward rate at {convergence_point} "
            "years within the tolerance of the ultimate forward rate."
        )

    return float(brentq(gap, MINIMUM_ALPHA, MAXIMUM_ALPHA, xtol=1e-10))
