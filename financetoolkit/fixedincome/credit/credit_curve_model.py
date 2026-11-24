"""Credit Curve Model

Survival curves of a reduced-form (intensity) credit model: the default time is the first jump of
a process with a hazard rate λ(t), so the probability of surviving to t is Q(t) = exp(-∫ λ(u) du)
(Jarrow and Turnbull, 1995; O'Kane, 2008, ch. 3). The hazard rate is piecewise flat: constant
between the maturities of the instruments the curve is bootstrapped from and held at its last
value beyond them, as the ISDA CDS Standard Model and QuantLib's PiecewiseFlatHazardRate define
it. A default pays the recovery rate R times the face value (recovery of face value), at the
time of default.

Every function takes the zero-coupon curve as the Toolkit's curves are returned: annually
compounded zero rates by maturity in years, interpolated linearly in between (see
`financetoolkit.fixedincome.helpers.discount_factors`). Times are year fractions from today.

The integrals over the default time are exact, as in the ISDA CDS Standard Model: on the joint
grid of the curve maturities, the hazard rate maturities and the payment dates both the hazard
rate and the forward rate f(t) = -d ln P(t) / dt are constant between two grid points, so

- ∫_a^b P(t) λ Q(t) dt = λ P(a) Q(a) (1 - e^(-(λ + f)(b - a))) / (λ + f)

on every piece of the grid.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.helpers import discount_factors

# The share of the face value a defaulted senior unsecured bond recovers on average, the
# ISDA standard recovery rate of senior unsecured CDS.
DEFAULT_RECOVERY_RATE = 0.4

# The highest hazard rate the bootstrap searches: a 99.995% probability of default within
# a year, beyond any quoted spread.
MAXIMUM_HAZARD_RATE = 10.0

# Below this exponent the integrals use their Taylor expansion, as (1 - e^(-x)) / x loses
# its digits when x is close to zero.
TAYLOR_THRESHOLD = 1e-8


def get_survival_probability(
    hazard_rates: np.ndarray | list | pd.Series,
    hazard_times: np.ndarray | list,
    years: float | np.ndarray,
) -> np.ndarray:
    """
    Calculate the survival probabilities of a piecewise flat hazard rate curve: the
    probability that the reference entity has not defaulted by each time.

    The hazard rate λ_i holds between the maturities T_(i-1) and T_i (T_0 = 0), and the
    last one beyond the last maturity, so the probability of no default before t is

    - Q(t) = exp(-SUM(λ_i * (min(t, T_i) - T_(i-1))^+))

    Referenced against:
    - QuantLib, HazardRateCurve: https://www.quantlib.org/reference/class_quant_lib_1_1_interpolated_hazard_rate_curve.html

    Args:
        hazard_rates (np.ndarray | list | pd.Series): The hazard rates λ_i, continuous and
            per year.
        hazard_times (np.ndarray | list): The maturities T_i in years up to which each
            hazard rate holds, in increasing order.
        years (float | np.ndarray): The times, in years from today.

    Returns:
        np.ndarray: The survival probability of each time.
    """
    hazard_rates = np.asarray(hazard_rates, dtype=float)
    hazard_times = np.asarray(hazard_times, dtype=float)
    years = np.asarray(years, dtype=float)

    starts = np.concatenate([[0.0], hazard_times[:-1]])
    lengths = np.append(np.diff(starts), np.inf)
    exposure = np.clip(years[..., None] - starts, 0, lengths)

    return np.exp(-(exposure * hazard_rates).sum(axis=-1))


def get_hazard_rate(
    hazard_rates: np.ndarray | list | pd.Series,
    hazard_times: np.ndarray | list,
    years: float | np.ndarray,
) -> np.ndarray:
    """
    Calculate the hazard rate of a piecewise flat hazard rate curve at any times: the rate
    at which the reference entity defaults at that time, given that it survived until then.

    Args:
        hazard_rates (np.ndarray | list | pd.Series): The hazard rates λ_i per year.
        hazard_times (np.ndarray | list): The maturities T_i up to which each holds.
        years (float | np.ndarray): The times, in years from today.

    Returns:
        np.ndarray: The hazard rate in force at each time, λ_i on (T_(i-1), T_i].
    """
    hazard_rates = np.asarray(hazard_rates, dtype=float)
    position = np.searchsorted(np.asarray(hazard_times, dtype=float), years)

    return hazard_rates[np.minimum(position, len(hazard_rates) - 1)]


def get_default_integrals(
    zero_curve: pd.Series,
    hazard_rates: np.ndarray | list | pd.Series,
    hazard_times: np.ndarray | list,
    start: float,
    end: float,
    accrual_start: float | None = None,
) -> tuple[float, float]:
    """
    Calculate the discounted probability of a default between two times, and the discounted
    time accrued since a date at that default: the two integrals of the default leg of a
    bond or CDS (O'Kane, 2008, ch. 6; ISDA CDS Standard Model).

    The formula is as follows:

    - Default Value = ∫_start^end P(t) λ(t) Q(t) dt = ∫ P(t) (-dQ(t))
    - Accrual Value = ∫_start^end (t - accrual_start) P(t) λ(t) Q(t) dt

    Both are exact on the joint grid of the curve maturities, the hazard rate maturities and
    the two times, with a constant forward rate f and hazard rate λ on every piece [a, b]:

    - ∫_a^b P λ Q dt = λ P(a) Q(a) E_1, E_1 = (1 - e^(-kΔ)) / k
    - ∫_a^b (t - s) P λ Q dt = λ P(a) Q(a) ((a - s) E_1 + E_2), E_2 = (E_1 - Δ e^(-kΔ)) / k

    with k = λ + f and Δ = b - a.

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        hazard_rates (np.ndarray | list | pd.Series): The piecewise flat hazard rates.
        hazard_times (np.ndarray | list): The maturities up to which each holds.
        start (float): The start of the period, in years from today.
        end (float): The end of the period, in years from today.
        accrual_start (float | None, optional): The date premium accrues from, for the
            Accrual Value. Defaults to None, the start.

    Returns:
        tuple[float, float]: The Default Value and the Accrual Value (in years).
    """
    accrual_start = start if accrual_start is None else accrual_start
    nodes = np.concatenate(
        [
            np.asarray(hazard_times, dtype=float),
            zero_curve.index.to_numpy(dtype=float),
        ]
    )
    grid = np.unique(
        np.concatenate([[start, end], nodes[(nodes > start) & (nodes < end)]])
    )

    discount = discount_factors(zero_curve, grid)
    survival = get_survival_probability(hazard_rates, hazard_times, grid)
    length = np.diff(grid)
    forward = np.log(discount[:-1] / discount[1:]) / length
    hazard = np.log(survival[:-1] / survival[1:]) / length
    rate = forward + hazard
    exponent = rate * length
    small = np.abs(exponent) < TAYLOR_THRESHOLD
    safe_rate = np.where(small, 1.0, rate)

    first = np.where(small, length, -np.expm1(-exponent) / safe_rate)
    second = np.where(
        small,
        length**2 / 2,
        (first - length * np.exp(-exponent)) / safe_rate,
    )
    weight = hazard * discount[:-1] * survival[:-1]

    return (
        float(np.sum(weight * first)),
        float(np.sum(weight * ((grid[:-1] - accrual_start) * first + second))),
    )


def get_risky_bond_price(
    coupon_rate: float,
    maturity: float,
    zero_curve: pd.Series,
    hazard_rates: np.ndarray | list | pd.Series,
    hazard_times: np.ndarray | list,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
    frequency: int = 2,
    par_value: float = 100,
) -> float:
    """
    Calculate the price of a fixed rate bond that can default: what its coupons and
    principal are worth given the chance that the issuer defaults and pays only the
    recovery, from a risk-free zero-coupon curve and a survival curve (Jarrow and Turnbull,
    1995; O'Kane, 2008, ch. 3).

    Every coupon and the principal are paid only when the issuer survives to their date, and
    a default pays the recovery rate times the face value at the time of default:

    The formula is as follows:

    - Price = Par Value * (c / m * SUM(P(t_k) Q(t_k)) + P(T) Q(T) + R * ∫_0^T P(t) (-dQ(t)))

    where c is the coupon rate paid m times a year on the dates t_k counted back from the
    maturity T, P(t) = (1 + r(t))^(-t) the discount factor of the annually compounded zero
    rate r(t), Q(t) the survival probability of the piecewise flat hazard rate curve and R
    the recovery rate. The default integral is exact on the grid of the curve and hazard
    rate maturities (see `get_default_integrals`). The price is the full (dirty) price: a
    maturity between coupon dates pays the whole first coupon.

    For more information, see the following sources:

    - Merton, R.C. (1974). "On the Pricing of Corporate Debt: The Risk Structure of
      Interest Rates." The Journal of Finance, 29(2), 449-470.
    - Jarrow, R.A. and Turnbull, S.M. (1995). "Pricing Derivatives on Financial Securities
      Subject to Credit Risk." The Journal of Finance, 50(1), 53-85.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley.

    Referenced against:
    - QuantLib, RiskyBondEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_risky_bond_engine.html

    Args:
        coupon_rate (float): The annual coupon rate, as a decimal.
        maturity (float): The years until the bond matures.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        hazard_rates (np.ndarray | list | pd.Series): The piecewise flat hazard rates per
            year.
        hazard_times (np.ndarray | list): The maturities up to which each holds.
        recovery_rate (float, optional): The share of the face value recovered on default.
            Defaults to 0.4.
        frequency (int, optional): The number of coupons per year. Defaults to 2.
        par_value (float, optional): The face value. Defaults to 100.

    Returns:
        float: The full price of the bond.

    Raises:
        ValueError: If the maturity is not positive or the recovery rate is not between 0
            and 1.
    """
    if not 0 <= recovery_rate < 1:
        raise ValueError("The recovery rate must lie between 0 and 1.")

    times = bond_model.get_cash_flow_periods(maturity, frequency) / frequency
    risky_discount = discount_factors(zero_curve, times) * get_survival_probability(
        hazard_rates, hazard_times, times
    )
    # Integrated per coupon period, so the coupon dates are part of the grid.
    default_value = sum(
        get_default_integrals(zero_curve, hazard_rates, hazard_times, begin, end)[0]
        for begin, end in zip(np.concatenate([[0.0], times[:-1]]), times)
    )

    return par_value * float(
        coupon_rate / frequency * risky_discount.sum()
        + risky_discount[-1]
        + recovery_rate * default_value
    )


def get_bond_implied_hazard_rates(
    maturities: np.ndarray | list,
    spreads: np.ndarray | list | pd.Series,
    zero_curve: pd.Series,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
    frequency: int = 2,
) -> pd.Series:
    """
    Bootstrap a piecewise flat hazard rate curve from the spreads of bonds over the
    risk-free curve, such as the option-adjusted spreads of a corporate bond index by
    maturity: the default intensity over time that the bond market prices in.

    Each spread s_i is taken as the spread over the zero-coupon curve of a bond that matures
    at T_i and trades at par, so its coupon is the par yield of the spread curve. The hazard
    rate up to T_i is then solved, in maturity order with the earlier ones fixed, so that the
    risky bond price (see `get_risky_bond_price`) of that bond is par (O'Kane, 2008, ch. 7).

    The formula is as follows:

    - D_s(t) = P(t) * (1 + s_i / m)^(-m t), the discount factor of the spread curve
    - c_i = m * (1 - D_s(T_i)) / SUM(D_s(t_k)), the par coupon
    - Solve λ_i: c_i / m * SUM(P(t_k) Q(t_k)) + P(T_i) Q(T_i) + R ∫_0^T_i P (-dQ) = 1

    where the spread is compounded m times a year (twice for the semi-annual spreads of US
    corporate bonds) and P(t) = (1 + r(t))^(-t) is the discount factor of the annually
    compounded zero curve. The credit triangle λ ≈ s / (1 - R) is the first-order answer.

    For more information, see the following sources:

    - Jarrow, R.A. and Turnbull, S.M. (1995). "Pricing Derivatives on Financial Securities
      Subject to Credit Risk." The Journal of Finance, 50(1), 53-85.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 7.

    Args:
        maturities (np.ndarray | list): The maturities of the bonds in years, increasing.
        spreads (np.ndarray | list | pd.Series): The spread of each bond over the zero
            curve, as decimals compounded `frequency` times a year.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        recovery_rate (float, optional): The share of the face value recovered on default.
            Defaults to 0.4.
        frequency (int, optional): The number of coupons per year of the bonds and the
            compounding of the spreads. Defaults to 2.

    Returns:
        pd.Series: The hazard rate (continuous, per year) up to each maturity.

    Raises:
        ValueError: If the maturities do not increase, a spread is negative or missing, or a
            spread is so far below the ones before it that it implies a negative hazard rate.
    """
    maturities = np.asarray(maturities, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    if np.any(np.diff(maturities) <= 0) or not np.all(np.isfinite(spreads)):
        raise ValueError(
            "The maturities must increase and every maturity needs a spread."
        )
    if np.any(spreads < 0):
        raise ValueError("A negative spread implies no default risk to bootstrap.")

    hazard_rates = np.zeros(len(maturities))
    for position, (maturity, spread) in enumerate(zip(maturities, spreads)):
        times = bond_model.get_cash_flow_periods(maturity, frequency) / frequency
        spread_discount = discount_factors(zero_curve, times) * (
            1 + spread / frequency
        ) ** (-frequency * times)
        coupon_rate = frequency * (1 - spread_discount[-1]) / spread_discount.sum()

        def pricing_error(hazard_rate: float, position=position, coupon=coupon_rate):
            hazard_rates[position] = hazard_rate
            return (
                get_risky_bond_price(
                    coupon,
                    maturities[position],
                    zero_curve,
                    hazard_rates[: position + 1],
                    maturities[: position + 1],
                    recovery_rate,
                    frequency,
                    par_value=1,
                )
                - 1
            )

        if spread == 0:
            hazard_rates[position] = 0.0
            continue
        # The price falls as the hazard rate rises, so a bond below par at a hazard rate of
        # zero after the earlier ones needs a negative hazard rate, which a steeply inverted
        # spread curve can ask for.
        if pricing_error(0.0) < 0:
            raise ValueError(
                f"The spread of maturity {maturity:g} is too low for the spreads before it: "
                "it implies a negative hazard rate up to that maturity."
            )
        if pricing_error(MAXIMUM_HAZARD_RATE) > 0:
            raise ValueError(
                f"The spread of maturity {maturity:g} implies a hazard rate above "
                f"{MAXIMUM_HAZARD_RATE:g} a year."
            )
        hazard_rates[position] = brentq(
            pricing_error, 0.0, MAXIMUM_HAZARD_RATE, xtol=1e-14
        )

    return pd.Series(hazard_rates, index=maturities, name="Hazard Rate")
