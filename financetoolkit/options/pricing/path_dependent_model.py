"""Path-Dependent Option Model"""

__docformat__ = "google"

import numbers

import numpy as np
from scipy.special import exprel, log_ndtr, ndtr

from financetoolkit.options.helpers import (
    bivariate_normal_cdf,
    validate_numeric_inputs,
    validate_positive_inputs,
)
from financetoolkit.options.options_model import (
    MAX_SIMULATED_VALUES,
    SIMULATION_CHUNK_SIZE,
)
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.options.pricing.exotics_model import (
    _barrier_term,
    get_barrier_option,
)
from financetoolkit.utilities.error_model import check_option

# The shift of Broadie, Glasserman and Kou (1997), β = -ζ(1/2) / √(2π), with ζ the Riemann
# zeta function: a barrier watched every Δt years prices as a continuous barrier moved
# away from the stock price by a factor e^(β * σ * √Δt).
DISCRETE_MONITORING_SHIFT = 0.5825971579390106

# The years between two observations of a barrier watched daily (252 trading days a year),
# weekly or monthly.
MONITORING_INTERVALS = {"daily": 1 / 252, "weekly": 1 / 52, "monthly": 1 / 12}

# The Gauss-Legendre points of the integrals over the averaging period (the second moment of
# a continuous average and Curran's sums for a continuous average) and over the band of a
# soft barrier. The integrands are smooth, so these points are exact to machine precision.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(64)

# Below this cost of carry the lookback formulas, which divide by b, are evaluated at
# b = ±_SMALL_CARRY and averaged, which differs from the limit at b = 0 by O(b²).
_SMALL_CARRY = 1e-5

# Below this σ√T a touch option is valued on the deterministic forward path: the powers of
# its formulas grow like 1/σ², so their logarithms lose all precision near σ√T = 1e-12, while
# at 1e-7 the formulas agree with the deterministic value to 1e-9.
SMALL_DEVIATION = 1e-7


def _validate_volatility(volatility: float) -> None:
    """Raises a ValueError for a volatility that is not positive, which the formulas divide by."""
    if np.any(np.asarray(volatility) <= 0):
        raise ValueError(f"volatility must be positive, received {volatility!r}.")


def _expected_term(
    mean: float,
    variance: float,
    power: float,
    intercept: float = np.inf,
    slope: float = 0.0,
    side: int | None = None,
    cutoff: float = 0.0,
) -> float:
    """
    The expectation E[e^(aY) * N(α + βY) * 1{ηY > ηc}] of a normal variable Y with the given
    mean m and variance v, the building block of the partial-time barriers and lookbacks,
    whose value at an intermediate date is a sum of such terms of the log stock price:

    - E = e^(am + a²v/2) * M((α + β(m + av)) / √(1 + β²v), η(m + av - c) / √v; ηβ√v / √(1 + β²v))

    Weighting the density by e^(aY) moves its mean to m + av, after which N(α + βY) is the
    probability that an independent standard normal variable stays below α + βY. Without an
    indicator (side None) the bivariate normal M becomes N of its first argument, and
    without a normal distribution function (α infinite) N of its second.

    Args:
        mean (float): The mean m of Y.
        variance (float): The variance v of Y.
        power (float): The exponent a.
        intercept (float, optional): α. Defaults to infinity, which leaves out N.
        slope (float, optional): β. Defaults to 0.
        side (int | None, optional): η, 1 for Y > c and -1 for Y < c. Defaults to None, which
            leaves out the indicator.
        cutoff (float, optional): c. Defaults to 0.

    Returns:
        float: The expectation.
    """
    shifted_mean = mean + power * variance
    scale = np.exp(power * mean + 0.5 * power**2 * variance)
    spread = np.sqrt(1 + slope**2 * variance)
    first = (intercept + slope * shifted_mean) / spread

    if side is None:
        return scale * ndtr(first)

    second = side * (shifted_mean - cutoff) / np.sqrt(variance)

    if np.all(np.isposinf(intercept)):
        return scale * ndtr(second)

    return scale * bivariate_normal_cdf(
        first, second, side * slope * np.sqrt(variance) / spread
    )


def _expected_value(
    terms: list[tuple[float, float, float, float]],
    mean: float,
    variance: float,
    side: int | None = None,
    cutoff: float = 0.0,
) -> float:
    """
    The expectation of a sum of terms coefficient * e^(aY) * N(α + βY), each a tuple
    (coefficient, a, α, β), over the region ηY > ηc (see _expected_term). The inputs may be
    arrays of one shape (a grid of strikes and expirations); a term is left out where its
    coefficient is zero, where its expectation may not be defined.
    """
    value = 0.0
    for coefficient, power, intercept, slope in terms:
        if not np.any(coefficient != 0):
            continue
        term = coefficient * _expected_term(
            mean, variance, power, intercept, slope, side, cutoff
        )
        value = value + (
            np.where(coefficient != 0, term, 0.0) if np.ndim(coefficient) else term
        )
    return value


def get_discrete_monitoring_barrier(
    barrier: float,
    stock_price: float,
    volatility: float,
    monitoring_interval: float,
) -> float:
    """
    Calculate the continuous barrier that prices a discretely watched barrier: the barrier
    shifted away from the stock price by the continuity correction of Broadie, Glasserman and
    Kou (1997).

    A barrier that is only checked at the close of each day (or week, or month) is crossed less
    often than one watched continuously, so a discretely monitored barrier option is priced
    with the continuous formulas after moving the barrier away from the stock price:

    - H_discrete = H * e^(β * σ * √Δt) for a barrier above the stock price
    - H_discrete = H * e^(-β * σ * √Δt) for a barrier below the stock price
    - β = -ζ(1/2) / √(2π) ≈ 0.5826

    Where H is the barrier, σ the volatility, Δt the years between two observations and ζ the
    Riemann zeta function. The error of the correction is of order Δt, small for daily and
    weekly monitoring.

    Args:
        barrier (float): The barrier level.
        stock_price (float): The current stock price, which decides on which side the barrier
            lies.
        volatility (float): The volatility of the stock.
        monitoring_interval (float): The years between two observations of the barrier, 1/252
            for daily monitoring. Zero leaves the barrier as it is (continuous monitoring).

    Returns:
        float: The shifted barrier.

    Notes:
        Reference: Broadie, M., Glasserman, P., & Kou, S. (1997). "A Continuity Correction for
        Discrete Barrier Options." Mathematical Finance, 7(4), 325-349.
    """
    validate_numeric_inputs(
        barrier=barrier,
        stock_price=stock_price,
        volatility=volatility,
        monitoring_interval=monitoring_interval,
    )
    direction = 1 if barrier > stock_price else -1

    return barrier * np.exp(
        direction
        * DISCRETE_MONITORING_SHIFT
        * volatility
        * np.sqrt(max(monitoring_interval, 0.0))
    )


def _averaging_times(time_to_expiration: float, fixings: int) -> np.ndarray:
    """The fixing times of an average of equally spaced prices, the last at expiration."""
    return time_to_expiration * np.arange(1, fixings + 1) / fixings


def _geometric_moments(
    stock_price: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    fixings: int | None,
) -> tuple[float, float]:
    """
    The mean and variance of the logarithm of the geometric average G, which is normal:

    - Continuous: E[ln G] = ln S + (b - σ²/2) * T / 2, Var[ln G] = σ² * T / 3
    - n fixings t_i = iT / n: E[ln G] = ln S + (b - σ²/2) * T * (n + 1) / (2n),
      Var[ln G] = σ² * T * (n + 1)(2n + 1) / (6n²)
    """
    drift = cost_of_carry - 0.5 * volatility**2

    if fixings is None:
        return (
            np.log(stock_price) + drift * time_to_expiration / 2,
            volatility**2 * time_to_expiration / 3,
        )

    return (
        np.log(stock_price)
        + drift * time_to_expiration * (fixings + 1) / (2 * fixings),
        volatility**2
        * time_to_expiration
        * (fixings + 1)
        * (2 * fixings + 1)
        / (6 * fixings**2),
    )


def _lognormal_option(
    forward: float,
    log_variance: float,
    strike_price: float,
    discount_factor: float,
    put_option: bool,
) -> float:
    """
    The price of an option on a lognormal variable with the given forward (mean) and variance
    of its logarithm: φ * e^(-rT) * (F * N(φ * d1) - K * N(φ * d2)), d1 = (ln(F / K) + v / 2) / √v.
    """
    sign = -1 if put_option else 1
    deviation = np.sqrt(log_variance)
    d1 = (np.log(forward / strike_price) + 0.5 * log_variance) / deviation
    d2 = d1 - deviation

    return (
        sign
        * discount_factor
        * (forward * ndtr(sign * d1) - strike_price * ndtr(sign * d2))
    )


def _validate_asian(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
    fixings: int | None,
    arrays: bool = False,
) -> None:
    """
    Validates the inputs of the Asian option formulas, which may be arrays for the formulas
    that price a grid of strikes and expirations at once (arrays=True).
    """
    validate_numeric_inputs(
        arrays=arrays,
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    _validate_volatility(volatility)
    if fixings is not None and (
        isinstance(fixings, bool)
        or not isinstance(fixings, numbers.Integral)
        or fixings < 1
    ):
        raise ValueError(
            f"fixings must be a whole number of at least 1 or None, received {fixings!r}."
        )


def get_geometric_asian_option(
    stock_price: float | np.ndarray,
    strike_price: float | np.ndarray,
    risk_free_rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float | np.ndarray = 0.0,
    put_option: bool = False,
    fixings: int | None = None,
) -> float | np.ndarray:
    """
    Calculate the closed-form price of a geometric-average price Asian option: what an option on
    the geometric average of the stock price is worth, averaged continuously (Kemna & Vorst,
    1990) or over n equally spaced fixings.

    The logarithm of the geometric average G of a lognormal stock price is normal, so the
    option is a Black-Scholes option on G:

    - Continuous: E[ln G] = ln S + (b - σ²/2) * T / 2, Var[ln G] = σ² * T / 3
    - n fixings at t_i = iT / n: E[ln G] = ln S + (b - σ²/2) * T * (n + 1) / (2n),
      Var[ln G] = σ² * T * (n + 1)(2n + 1) / (6n²)
    - Call = e^(-rT) * (E[G] * N(d1) - K * N(d2)), E[G] = e^(E[ln G] + Var[ln G] / 2)
    - d1 = (E[ln G] - ln K + Var[ln G]) / √Var[ln G], d2 = d1 - √Var[ln G]

    With b = r - q. The continuous version equals exotics_model.get_asian_option.

    Referenced against:
    - QuantLib, AnalyticDiscreteGeometricAveragePriceAsianEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_discrete_geometric_average_price_asian_engine.html

    Args:
        stock_price (float | np.ndarray): The current stock price.
        strike_price (float | np.ndarray): The option's strike price.
        risk_free_rate (float | np.ndarray): The continuously compounded risk-free rate.
        volatility (float | np.ndarray): The volatility of the stock.
        time_to_expiration (float | np.ndarray): The time to expiration in years.
        dividend_yield (float | np.ndarray): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the option is a put option. Defaults to False.
        fixings (int | None): The number of equally spaced fixings, the first a 1/n-th of
            the life from now and the last at expiration. Defaults to None, which averages
            continuously.

    Returns:
        float | np.ndarray: The geometric-average Asian option value, an array in the shape
        of the inputs when any of them is an array (a grid of strikes and expirations is
        priced in one call).

    Notes:
        Reference: Kemna, A. G. Z., & Vorst, A. C. F. (1990). "A Pricing Method for Options
        Based on Average Asset Values." Journal of Banking & Finance, 14(1), 113-129.
    """
    _validate_asian(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        fixings,
        arrays=True,
    )
    mean, variance = _geometric_moments(
        stock_price,
        risk_free_rate - dividend_yield,
        volatility,
        time_to_expiration,
        fixings,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(
            np.asarray(time_to_expiration) <= 0,
            np.maximum((strike_price - stock_price) * (1 if put_option else -1), 0.0),
            _lognormal_option(
                np.exp(mean + 0.5 * variance),
                variance,
                strike_price,
                np.exp(-risk_free_rate * time_to_expiration),
                put_option,
            ),
        )

    return value if value.ndim else float(value)


def _arithmetic_moments(
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    fixings: int | None,
) -> tuple[float, float]:
    """
    The first two moments of the arithmetic average A per unit of stock price, E[A] / S and
    E[A²] / S², averaged continuously or over n fixings t_i = iT / n:

    - Continuous: E[A] / S = (e^(bT) - 1) / (bT),
      E[A²] / S² = 2 / T² * ∫_0^T e^(bt) * (e^((b + σ²)t) - 1) / (b + σ²) dt
    - Fixings: E[A] / S = Σ e^(b t_i) / n,
      E[A²] / S² = Σ_i Σ_j e^(b(t_i + t_j) + σ² min(t_i, t_j)) / n²

    The continuous integral is taken with Gauss-Legendre points, since its closed form divides
    by b, b + σ² and 2b + σ², and the double sum in O(n) with the tail sums of e^(b t_j).
    """
    if fixings is None:
        nodes = 0.5 * (_LEGENDRE_NODES + 1)
        weights = 0.5 * _LEGENDRE_WEIGHTS
        scaled = cost_of_carry * time_to_expiration * nodes
        inner = (
            nodes
            * exprel((cost_of_carry + volatility**2) * time_to_expiration * nodes)
            * np.exp(scaled)
        )
        return float(exprel(cost_of_carry * time_to_expiration)), float(
            2 * np.sum(weights * inner)
        )

    times = _averaging_times(time_to_expiration, fixings)
    growth = np.exp(cost_of_carry * times)
    later_growth = np.concatenate((np.cumsum(growth[::-1])[::-1][1:], [0.0]))
    second = np.sum(
        np.exp((2 * cost_of_carry + volatility**2) * times)
        + 2 * np.exp((cost_of_carry + volatility**2) * times) * later_growth
    )

    return float(growth.mean()), float(second / fixings**2)


def get_turnbull_wakeman(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    fixings: int | None = None,
) -> float:
    """
    Calculate the price of an arithmetic-average price Asian option with the approximation of
    Turnbull and Wakeman (1991): what an option on the average stock price is worth when that
    average is taken to be lognormal with the first two moments of the true average.

    - M1 = E[A] / S, M2 = E[A²] / S² (see the moments below)
    - F_A = S * M1, v = ln(M2 / M1²)
    - Call = e^(-rT) * (F_A * N(d1) - K * N(d2)), Put = e^(-rT) * (K * N(-d2) - F_A * N(-d1))
    - d1 = (ln(F_A / K) + v / 2) / √v, d2 = d1 - √v

    The moments are those of a continuous average or of n equally spaced fixings t_i = iT / n:

    - Continuous: M1 = (e^(bT) - 1) / (bT), M2 = 2 / T² * ∫_0^T e^(bt) * (e^((b + σ²)t) - 1) / (b + σ²) dt
    - Fixings: M1 = Σ e^(b t_i) / n, M2 = Σ_i Σ_j e^(b(t_i + t_j) + σ² min(t_i, t_j)) / n²

    Written in the generalized Black-Scholes form of Haug (2007) as b_A = ln(M1) / T and
    σ_A² = ln(M2) / T - 2b_A, which gives the same price. For a continuous average that starts
    today, Levy (1992) matches the same two moments with a lognormal and gives the same price;
    the two differ only for an average that has already started.

    Referenced against:
    - QuantLib, TurnbullWakemanAsianEngine (fixings): https://www.quantlib.org/reference/class_quant_lib_1_1_turnbull_wakeman_asian_engine.html
    - QuantLib, ContinuousArithmeticAsianLevyEngine (continuous): https://www.quantlib.org/reference/class_quant_lib_1_1_continuous_arithmetic_asian_levy_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the option is a put option. Defaults to False.
        fixings (int | None): The number of equally spaced fixings, the last at expiration.
            Defaults to None, which averages continuously.

    Returns:
        float: The arithmetic-average Asian option value.

    Notes:
        Reference: Turnbull, S. M., & Wakeman, L. M. (1991). "A Quick Algorithm for Pricing
        European Average Options." Journal of Financial and Quantitative Analysis, 26(3),
        377-389. Levy, E. (1992). "Pricing European Average Rate Currency Options." Journal of
        International Money and Finance, 11(5), 474-491. Haug, E. G. (2007). The Complete Guide
        to Option Pricing Formulas, 2nd ed., sections 4.20.2 and 4.20.3.
    """
    _validate_asian(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        fixings,
        arrays=True,
    )
    # The moments are matched in logarithms of the strike price.
    validate_positive_inputs(stock_price=stock_price, strike_price=strike_price)
    if time_to_expiration <= 0:
        payoff = np.maximum(
            (strike_price - stock_price) * (1 if put_option else -1), 0.0
        )
        return float(payoff) if np.ndim(payoff) == 0 else payoff

    first, second = _arithmetic_moments(
        risk_free_rate - dividend_yield, volatility, time_to_expiration, fixings
    )

    value = _lognormal_option(
        stock_price * first,
        np.log(second / first**2),
        strike_price,
        np.exp(-risk_free_rate * time_to_expiration),
        put_option,
    )
    return float(value) if np.ndim(value) == 0 else value


def _curran_call(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    times: np.ndarray,
    weights: np.ndarray,
    covariances: np.ndarray,
    geometric_variance: float,
) -> tuple[float, float]:
    """
    Curran's call price on the average Σ w_i S(t_i), with the covariances σ_iG of each log
    price with the log geometric average and its variance σ_G². Returns the call and the
    discounted forward of the average, from which the put follows by parity.
    """
    drift = cost_of_carry - 0.5 * volatility**2
    log_means = np.log(stock_price) + drift * times
    log_variances = volatility**2 * times
    geometric_mean = float(np.sum(weights * log_means))
    geometric_deviation = np.sqrt(geometric_variance)
    discount_factor = np.exp(-risk_free_rate * time_to_expiration)
    expected_prices = np.exp(log_means + 0.5 * log_variances)
    forward = float(np.sum(weights * expected_prices))

    # The geometric average at which the expected arithmetic average equals the strike,
    # from the conditional means E[S(t_i) | G = K] to first order around G = K, with a row
    # per strike price for a grid of strikes of one expiration.
    strikes = np.asarray(strike_price, dtype=float)[..., None]
    adjusted_strike = 2 * strikes[..., 0] - np.sum(
        weights
        * np.exp(
            log_means
            + covariances / geometric_variance * (np.log(strikes) - geometric_mean)
            + 0.5 * (log_variances - covariances**2 / geometric_variance)
        ),
        axis=-1,
    )
    # A non-positive adjusted strike is where the first-order expansion breaks down, which
    # happens at both ends. Deep in the money (a strike below the forward of the average) the
    # average ends above the strike for (almost) every path and the call is the forward minus
    # the strike; deep out of the money the expansion would give that negative difference too,
    # so the call on the geometric average, a lower bound of about zero there, is used instead.
    positive = adjusted_strike > 0
    threshold = (
        geometric_mean - np.log(np.where(positive, adjusted_strike, 1.0))
    ) / geometric_deviation
    geometric_threshold = (
        geometric_mean - np.log(strikes[..., 0])
    ) / geometric_deviation
    geometric_call = discount_factor * (
        np.exp(geometric_mean + 0.5 * geometric_variance)
        * ndtr(geometric_threshold + geometric_deviation)
        - strike_price * ndtr(geometric_threshold)
    )
    call = np.where(
        positive,
        discount_factor
        * (
            np.sum(
                weights
                * expected_prices
                * ndtr(threshold[..., None] + covariances / geometric_deviation),
                axis=-1,
            )
            - strike_price * ndtr(threshold)
        ),
        np.where(
            strike_price < forward,
            discount_factor * (forward - strike_price),
            geometric_call,
        ),
    )
    # The sums of the approximation leave rounding noise below zero far out of the money.
    call = np.maximum(call, 0.0)
    return (float(call) if np.ndim(call) == 0 else call), discount_factor * forward


def get_curran(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    fixings: int | None = None,
) -> float:
    """
    Calculate the price of an arithmetic-average price Asian option with the approximation of
    Curran (1994): what an option on the average stock price is worth, conditioning on the
    geometric average G.

    The arithmetic average A is at least G, so when G ends above the strike the call pays
    A - K exactly; only the paths with G below the strike are approximated, by the threshold
    at which the expected average given G equals the strike:

    - μ_i = ln S + (b - σ²/2) * t_i, σ_i² = σ² * t_i, μ_G = Σ μ_i / n
    - σ_G² = σ² * Σ_i Σ_j min(t_i, t_j) / n², σ_iG = σ² * Σ_j min(t_i, t_j) / n
    - K̂ = 2K - Σ e^(μ_i + σ_iG / σ_G² * (ln K - μ_G) + (σ_i² - σ_iG² / σ_G²) / 2) / n
    - Call = e^(-rT) * [Σ e^(μ_i + σ_i² / 2) * N((μ_G - ln K̂) / σ_G + σ_iG / σ_G) / n - K * N((μ_G - ln K̂) / σ_G)]
    - Put = Call - e^(-rT) * (Σ e^(μ_i + σ_i² / 2) / n - K)

    With n equally spaced fixings t_i = iT / n. A continuous average replaces the sums by
    integrals over [0, T] (σ_tG = σ² * (t - t² / (2T)), σ_G² = σ² * T / 3), taken with
    Gauss-Legendre points. When K̂ is not positive the expansion breaks down: for a strike
    below E[A] the average ends above the strike on every path and the call is
    e^(-rT) * (E[A] - K); for a strike above it (deep out of the money) the call is that on
    the geometric average, e^(-rT) * [e^(μ_G + σ_G² / 2) * N(d + σ_G) - K * N(d)] with
    d = (μ_G - ln K) / σ_G, a lower bound that is about zero there. Calls and puts are floored
    at zero.

    Referenced against:
    - QuantLib, MCDiscreteArithmeticAPEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_discrete_arithmetic_a_p_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the option is a put option. Defaults to False.
        fixings (int | None): The number of equally spaced fixings, the last at expiration.
            Defaults to None, which averages continuously.

    Returns:
        float: The arithmetic-average Asian option value.

    Notes:
        Reference: Curran, M. (1994). "Valuing Asian and Portfolio Options by Conditioning on
        the Geometric Mean Price." Management Science, 40(12), 1705-1711.
    """
    _validate_asian(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        fixings,
        arrays=True,
    )
    if time_to_expiration <= 0:
        payoff = np.maximum(
            (strike_price - stock_price) * (1 if put_option else -1), 0.0
        )
        return float(payoff) if np.ndim(payoff) == 0 else payoff

    if fixings is None:
        times = 0.5 * time_to_expiration * (_LEGENDRE_NODES + 1)
        weights = 0.5 * _LEGENDRE_WEIGHTS
        covariances = volatility**2 * (times - times**2 / (2 * time_to_expiration))
        geometric_variance = volatility**2 * time_to_expiration / 3
    else:
        steps = np.arange(1, fixings + 1)
        times = _averaging_times(time_to_expiration, fixings)
        weights = np.full(fixings, 1 / fixings)
        # Σ_j min(i, j) = i(i + 1) / 2 + i(n - i) in units of the fixing interval.
        covariances = (
            volatility**2
            * (time_to_expiration / fixings)
            * (steps * (steps + 1) / 2 + steps * (fixings - steps))
            / fixings
        )
        geometric_variance = (
            volatility**2
            * time_to_expiration
            * (fixings + 1)
            * (2 * fixings + 1)
            / (6 * fixings**2)
        )

    call, discounted_forward = _curran_call(
        stock_price,
        strike_price,
        risk_free_rate,
        risk_free_rate - dividend_yield,
        volatility,
        time_to_expiration,
        times,
        weights,
        covariances,
        geometric_variance,
    )

    if put_option:
        # Floored at zero, as the parity leaves rounding noise below it deep out of the money.
        put = np.maximum(
            call
            - discounted_forward
            + strike_price * np.exp(-risk_free_rate * time_to_expiration),
            0.0,
        )
        return float(put) if np.ndim(put) == 0 else put
    return call


def get_monte_carlo_asian_options(
    stock_price: float,
    strike_prices: np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    fixings: int = 100,
    simulations: int = 10_000,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Monte Carlo price of arithmetic-average price Asian options: what options on
    the average stock price are worth on simulated paths, with the geometric-average option as
    control variate (Kemna & Vorst, 1990).

    Every path is a Geometric Brownian Motion observed at n equally spaced fixings. The
    geometric-average option on the same paths has a known price and moves closely with the
    arithmetic one, so its simulation error is subtracted:

    - S(t_i) = S(t_i-1) * e^((b - σ²/2) * Δt + σ * √Δt * Z), Δt = T / n
    - Y = e^(-rT) * max(φ * (A - K), 0), X = e^(-rT) * max(φ * (G - K), 0)
    - Price = MEAN(Y) - β * (MEAN(X) - E[X]), β = Cov(Y, X) / Var(X)
    - Standard Error = √(Var(Y - βX) / Simulations)

    Where A and G are the arithmetic and geometric averages of the fixings of a path, E[X] is
    the closed-form discrete geometric price (get_geometric_asian_option) and Var and Cov use
    ddof=1. All strike prices are priced on the same paths (common random numbers).

    Referenced against:
    - QuantLib, MCDiscreteArithmeticAPEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_discrete_arithmetic_a_p_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_prices (np.ndarray): The strike prices.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the options are put options. Defaults to False.
        fixings (int): The number of equally spaced fixings, the last at expiration, which are
            also the steps of each path. Defaults to 100.
        simulations (int): The number of simulated paths. Defaults to 10,000.
        seed (int | None): The seed of the random number generator, None for new paths on
            every call. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: The price and its standard error per strike price.

    Raises:
        ValueError: if simulations times fixings exceeds MAX_SIMULATED_VALUES.

    Notes:
        Reference: Kemna, A. G. Z., & Vorst, A. C. F. (1990). "A Pricing Method for Options
        Based on Average Asset Values." Journal of Banking & Finance, 14(1), 113-129.
    """
    strike_prices = np.asarray(strike_prices, dtype=float)
    _validate_asian(
        stock_price,
        float(strike_prices.min()) if strike_prices.size else 1.0,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        fixings,
    )
    if simulations <= 1:
        raise ValueError(f"simulations must be at least 2, received {simulations!r}.")
    if simulations * fixings > MAX_SIMULATED_VALUES:
        raise ValueError(
            f"{simulations:,} simulations of {fixings:,} fixings is too large; keep "
            f"simulations times fixings under {MAX_SIMULATED_VALUES:,}."
        )

    sign = -1 if put_option else 1

    if time_to_expiration <= 0:
        return np.maximum(sign * (stock_price - strike_prices), 0.0), np.zeros(
            strike_prices.size
        )

    random_number_generator = np.random.default_rng(seed)
    time_delta = time_to_expiration / fixings
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_delta
    diffusion = volatility * np.sqrt(time_delta)
    arithmetic_averages = np.empty(simulations)
    geometric_averages = np.empty(simulations)

    # The generator fills the shocks row by row, so drawing the paths in chunks of rows gives
    # the same paths as one draw while holding only a chunk in memory.
    for first in range(0, simulations, SIMULATION_CHUNK_SIZE):
        rows = min(SIMULATION_CHUNK_SIZE, simulations - first)
        log_paths = np.cumsum(
            drift
            + diffusion * random_number_generator.standard_normal((rows, fixings)),
            axis=1,
        )
        arithmetic_averages[first : first + rows] = stock_price * np.exp(
            log_paths
        ).mean(axis=1)
        geometric_averages[first : first + rows] = stock_price * np.exp(
            log_paths.mean(axis=1)
        )

    discount_factor = np.exp(-risk_free_rate * time_to_expiration)
    prices = np.empty(strike_prices.size)
    standard_errors = np.empty(strike_prices.size)

    for position, strike_price in enumerate(strike_prices):
        arithmetic = discount_factor * np.maximum(
            sign * (arithmetic_averages - strike_price), 0
        )
        geometric = discount_factor * np.maximum(
            sign * (geometric_averages - strike_price), 0
        )
        geometric_price = get_geometric_asian_option(
            float(stock_price),
            float(strike_price),
            float(risk_free_rate),
            float(volatility),
            float(time_to_expiration),
            float(dividend_yield),
            put_option,
            fixings,
        )
        geometric_variance = geometric.var(ddof=1)
        # Without variation in the control (every path out of the money) there is nothing
        # to correct and the estimate is the plain average.
        beta = (
            np.cov(arithmetic, geometric, ddof=1)[0, 1] / geometric_variance
            if geometric_variance > 0
            else 0.0
        )
        controlled = arithmetic - beta * (geometric - geometric_price)
        prices[position] = controlled.mean()
        standard_errors[position] = controlled.std(ddof=1) / np.sqrt(simulations)

    return prices, standard_errors


def _lookback_terms(
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    risk_free_rate: float,
    log_extreme: float | None,
    strike_price: float,
    fixed: bool,
    sign: int,
) -> list[tuple[float, float, float, float]]:
    """
    The value of a lookback option as a sum of terms coefficient * S^a * N(α + β ln S) of the
    log stock price ln S (see _expected_term), with the extreme observed so far E (None for an
    extreme that equals the stock price itself, so that E = S):

    - x = (ln(S / E) + (b + σ²/2) * T) / (σ√T)
    - Correction(s) = s * σ² / (2b) * [e^(bT) * N(s * x) - (S / E)^(-2b/σ²) * N(s * (x - 2b√T / σ))]
    - Fixed call (E = max(S_max, K)) = e^(-rT) * (E - K) + S * e^((b - r)T) * N(x) - E * e^(-rT) * N(x - σ√T)
      + S * e^(-rT) * Correction(1)
    - Fixed put (E = min(S_min, K)) = e^(-rT) * (K - E) - S * e^((b - r)T) * N(-x) + E * e^(-rT) * N(-x + σ√T)
      + S * e^(-rT) * Correction(-1)
    - Floating call (E = S_min) = S * e^((b - r)T) * N(x) - E * e^(-rT) * N(x - σ√T) + S * e^(-rT) * Correction(-1)
    - Floating put (E = S_max) = E * e^(-rT) * N(-x + σ√T) - S * e^((b - r)T) * N(-x) + S * e^(-rT) * Correction(1)

    With s = 1 for a fixed call and floating put and -1 for a fixed put and floating call.
    """
    root_time = volatility * np.sqrt(time_to_expiration)
    discount = np.exp(-risk_free_rate * time_to_expiration)
    carry_discount = np.exp((cost_of_carry - risk_free_rate) * time_to_expiration)
    ratio = volatility**2 / (2 * cost_of_carry)
    exponent = -2 * cost_of_carry / volatility**2
    shift = 2 * cost_of_carry * np.sqrt(time_to_expiration) / volatility
    offset = (cost_of_carry + 0.5 * volatility**2) * time_to_expiration / root_time

    # The correction of s, -1 for a call on the lowest and +1 for a put on the highest price.
    correction = sign if fixed else -sign
    if log_extreme is None:
        # x does not depend on S: every term is a constant times S (a = 1).
        x = offset
        constant = sign * (
            carry_discount * ndtr(sign * x) - discount * ndtr(sign * (x - root_time))
        ) + discount * correction * ratio * (
            np.exp(cost_of_carry * time_to_expiration) * ndtr(correction * x)
            - ndtr(correction * (x - shift))
        )
        if fixed:
            # E = S: e^(-rT) * (E - K) adds S * e^(-rT) and -K * e^(-rT).
            return [
                (constant + sign * discount, 1.0, np.inf, 0.0),
                (-sign * strike_price * discount, 0.0, np.inf, 0.0),
            ]
        return [(constant, 1.0, np.inf, 0.0)]

    slope = 1 / root_time
    intercept = -log_extreme * slope + offset
    extreme = np.exp(log_extreme)
    terms = [
        (sign * carry_discount, 1.0, sign * intercept, sign * slope),
        (-sign * extreme * discount, 0.0, sign * (intercept - root_time), sign * slope),
        (
            correction * discount * ratio * np.exp(cost_of_carry * time_to_expiration),
            1.0,
            correction * intercept,
            correction * slope,
        ),
        (
            -correction * discount * ratio * extreme ** (-exponent),
            1.0 + exponent,
            correction * (intercept - shift),
            correction * slope,
        ),
    ]
    if fixed:
        terms.append((sign * discount * (extreme - strike_price), 0.0, np.inf, 0.0))

    return terms


def _evaluate_terms(
    terms: list[tuple[float, float, float, float]], stock_price: float
) -> float:
    """
    The value of a sum of terms coefficient * S^a * N(α + β ln S) at a stock price. S^a and
    N(α + β ln S) are multiplied as e^(a ln S + ln N(α + β ln S)): at a low volatility the power
    a = 1 - 2b/σ² is in the hundreds or more, so S^a overflows (or underflows) where N is 0
    (or huge), while their product is finite.
    """
    log_price = np.log(stock_price)
    value = sum(
        coefficient
        * np.where(
            np.isposinf(intercept),
            stock_price**power,
            np.exp(power * log_price + log_ndtr(intercept + slope * log_price)),
        )
        for coefficient, power, intercept, slope in terms
    )
    return float(value) if np.ndim(value) == 0 else value


def _lookback_value(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    put_option: bool,
    fixed: bool,
    lookback_period: float,
) -> float:
    """
    The value of a lookback option for a cost of carry away from zero, see get_lookback_option.
    """
    sign = -1 if put_option else 1

    if lookback_period >= 1:
        extreme = stock_price
        if fixed:
            extreme = (
                np.minimum(stock_price, strike_price)
                if put_option
                else np.maximum(stock_price, strike_price)
            )
        # The value is homogeneous of degree one in S, E and K, so it is taken for an extreme
        # of 1 and scaled back: the term in E^(2b/σ²) then needs no power of the extreme,
        # which overflows at a low volatility.
        return extreme * _evaluate_terms(
            _lookback_terms(
                cost_of_carry,
                volatility,
                time_to_expiration,
                risk_free_rate,
                0.0,
                strike_price / extreme,
                fixed,
                sign,
            ),
            stock_price / extreme,
        )

    if fixed:
        # The extreme is watched over the last part of the life only: from t1 = (1 - share) * T
        # the option is a lookback that starts at S(t1), worth e^(-r t1) * E[V(S(t1))] today.
        start = (1 - lookback_period) * time_to_expiration
        remaining = time_to_expiration - start
        mean = np.log(stock_price) + (cost_of_carry - 0.5 * volatility**2) * start
        variance = volatility**2 * start
        log_strike = np.log(strike_price)
        # Beyond the strike the extreme starts at S(t1) itself; before it at the strike.
        own_extreme = _lookback_terms(
            cost_of_carry,
            volatility,
            remaining,
            risk_free_rate,
            None,
            strike_price,
            True,
            sign,
        )
        strike_extreme = _lookback_terms(
            cost_of_carry,
            volatility,
            remaining,
            risk_free_rate,
            log_strike,
            strike_price,
            True,
            sign,
        )
        value = np.exp(-risk_free_rate * start) * (
            _expected_value(own_extreme, mean, variance, sign, log_strike)
            + _expected_value(strike_extreme, mean, variance, -sign, log_strike)
        )
        return float(value) if np.ndim(value) == 0 else value

    return _partial_floating_lookback(
        stock_price,
        risk_free_rate,
        cost_of_carry,
        volatility,
        time_to_expiration,
        put_option,
        lookback_period * time_to_expiration,
    )


def _partial_floating_lookback(
    stock_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    put_option: bool,
    lookback_end: float,
) -> float:
    """
    The partial-time floating strike lookback of Heynen and Kat (1994): the strike is the
    lowest (call) or highest (put) price up to t1 = lookback_end, before expiration T.

    With x = ln(S(t1) / E) ≥ 0 the distance of S(t1) from that extreme E (in absolute terms),
    the option is a call (put) at t1 on S(t1) with strike E: S(t1) * c(e^(-x)) for a call and
    S(t1) * p(e^(x)) for a put, with c and p the Black-Scholes prices for a unit stock price.
    Under the measure with S(t1) as numeraire, x is distributed as the running maximum of a
    Brownian motion with drift ν' = s * (b + σ²/2) (s = 1 for a call, -1 for a put) by time
    reversal, so:

    - Value = S * e^((b - r) * t1) * ∫_0^∞ f(x) * g(x) dx
    - P(X ≤ x) = N((x - ν' t1) / (σ√t1)) - e^(2ν'x / σ²) * N((-x - ν' t1) / (σ√t1))

    Where g is the unit option and f the density of X, integrated over the square root of x
    with Gauss-Legendre points (the integrand is smooth there).
    """
    sign = -1 if put_option else 1
    remaining = time_to_expiration - lookback_end
    root_start = volatility * np.sqrt(lookback_end)
    drift = sign * (cost_of_carry + 0.5 * volatility**2)

    # The maximum stays within about twelve standard deviations plus its drift.
    upper = np.sqrt(abs(drift) * lookback_end + 12 * root_start)
    roots = 0.5 * upper * (_LEGENDRE_NODES + 1)
    weights = 0.5 * upper * _LEGENDRE_WEIGHTS
    distances = roots**2

    # The density of the running maximum, the derivative of its distribution above.
    density = (
        np.exp(-0.5 * ((distances - drift * lookback_end) / root_start) ** 2)
        / (root_start * np.sqrt(2 * np.pi))
        + np.exp(
            2 * drift * distances / volatility**2
            - 0.5 * ((-distances - drift * lookback_end) / root_start) ** 2
        )
        / (root_start * np.sqrt(2 * np.pi))
        - 2
        * drift
        / volatility**2
        * np.exp(2 * drift * distances / volatility**2)
        * ndtr((-distances - drift * lookback_end) / root_start)
    )

    unit_strikes = np.exp(-sign * distances)
    root_remaining = volatility * np.sqrt(remaining)
    d1 = (
        -np.log(unit_strikes) + (cost_of_carry + 0.5 * volatility**2) * remaining
    ) / root_remaining
    d2 = d1 - root_remaining
    unit_options = sign * (
        np.exp((cost_of_carry - risk_free_rate) * remaining) * ndtr(sign * d1)
        - unit_strikes * np.exp(-risk_free_rate * remaining) * ndtr(sign * d2)
    )

    # dx = 2u du on x = u².
    return float(
        stock_price
        * np.exp((cost_of_carry - risk_free_rate) * lookback_end)
        * np.sum(weights * 2 * roots * density * unit_options)
    )


def get_lookback_option(
    stock_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    strike_type: str = "fixed",
    lookback_period: float = 1.0,
) -> float | np.ndarray:
    """
    Calculate the price of a lookback option: what an option is worth whose payoff depends on
    the highest or lowest stock price over (part of) the option's life, for an option starting
    today.

    - Floating strike call: S(T) - min S, put: max S - S(T) (Goldman, Sosin & Gatto, 1979)
    - Fixed strike call: max(max S - K, 0), put: max(K - min S, 0) (Conze & Viswanathan, 1991)

    With b = r - q, x = (ln(S / E) + (b + σ²/2) * T) / (σ√T) and E the extreme so far (S for
    an option starting today, or the strike for a fixed strike beyond it):

    - Floating call = S * e^((b - r)T) * N(x) - E * e^(-rT) * N(x - σ√T)
      + S * e^(-rT) * σ² / (2b) * [(S / E)^(-2b/σ²) * N(-x + 2b√T / σ) - e^(bT) * N(-x)]
    - Floating put = E * e^(-rT) * N(-x + σ√T) - S * e^((b - r)T) * N(-x)
      + S * e^(-rT) * σ² / (2b) * [-(S / E)^(-2b/σ²) * N(x - 2b√T / σ) + e^(bT) * N(x)]
    - Fixed call (E = max(S, K)) = e^(-rT) * (E - K) + S * e^((b - r)T) * N(x) - E * e^(-rT) * N(x - σ√T)
      + S * e^(-rT) * σ² / (2b) * [-(S / E)^(-2b/σ²) * N(x - 2b√T / σ) + e^(bT) * N(x)]
    - Fixed put (E = min(S, K)) = e^(-rT) * (K - E) - S * e^((b - r)T) * N(-x) + E * e^(-rT) * N(-x + σ√T)
      + S * e^(-rT) * σ² / (2b) * [(S / E)^(-2b/σ²) * N(-x + 2b√T / σ) - e^(bT) * N(-x)]

    A partial-time lookback (Heynen & Kat, 1994) watches the extreme over part of the life,
    lookback_period times T:

    - Floating strike: the extreme of [0, t1], t1 = lookback_period * T, after which the option
      is a vanilla option with that extreme as strike. Priced as the vanilla option at t1 on the
      distance between S(t1) and the extreme, which by time reversal is distributed as the
      running maximum of a Brownian motion, integrated over its density.
    - Fixed strike: the extreme of [t1, T], t1 = (1 - lookback_period) * T, so the option is a
      fresh fixed strike lookback from t1: e^(-r t1) * E[V(S(t1))], in closed form with bivariate
      normal distributions, since V is a sum of terms S(t1)^a * N(α + β ln S(t1)).

    Where b = 0 (r = q) the formulas are taken at b = ±1e-5, their limit to O(b²).

    Referenced against:
    - QuantLib, AnalyticContinuousFixedLookbackEngine (fixed strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_fixed_lookback_engine.html
    - QuantLib, AnalyticContinuousFloatingLookbackEngine (floating strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_floating_lookback_engine.html
    - QuantLib, AnalyticContinuousPartialFixedLookbackEngine (partial fixed strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_partial_fixed_lookback_engine.html
    - QuantLib, AnalyticContinuousPartialFloatingLookbackEngine (partial floating strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_partial_floating_lookback_engine.html

    Args:
        stock_price (float): The current stock price, which is also the extreme so far.
        strike_price (float): The option's strike price, ignored for a floating strike.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the option is a put option. Defaults to False.
        strike_type (str): The strike. Options are: "fixed" (fixed strike), "floating"
            (floating strike). Defaults to "fixed".
        lookback_period (float): The share of the life over which the extreme is watched,
            above 0 and at most 1: the start for a floating strike, the end for a fixed
            strike. Defaults to 1, the whole life.

    Returns:
        float: The lookback option value.

    Raises:
        ValueError: if strike_type is not "fixed" or "floating", lookback_period is not in
            (0, 1] or volatility is not positive.

    Notes:
        References: Goldman, M. B., Sosin, H. B., & Gatto, M. A. (1979). "Path Dependent
        Options: Buy at the Low, Sell at the High." The Journal of Finance, 34(5), 1111-1127.
        Conze, A., & Viswanathan (1991). "Path Dependent Options: The Case of Lookback
        Options." The Journal of Finance, 46(5), 1893-1907. Heynen, R. C., & Kat, H. M.
        (1994). "Selective Memory." Risk, 7(11), 73-76.
    """
    validate_numeric_inputs(
        arrays=True,
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        lookback_period=lookback_period,
    )
    _validate_volatility(volatility)
    check_option("strike_type", strike_type, ("fixed", "floating"))
    if not 0 < lookback_period <= 1:
        raise ValueError(
            f"lookback_period must be above 0 and at most 1, received {lookback_period!r}."
        )

    fixed = strike_type == "fixed"

    if np.ndim(strike_price) or np.ndim(time_to_expiration):
        strike_price, time_to_expiration = np.broadcast_arrays(
            np.asarray(strike_price, dtype=float),
            np.asarray(time_to_expiration, dtype=float),
        )
        if lookback_period < 1 and not fixed:
            # The partial-time floating strike integrates per option.
            return np.vectorize(get_lookback_option, otypes=[float])(
                stock_price,
                strike_price,
                risk_free_rate,
                volatility,
                time_to_expiration,
                dividend_yield,
                put_option,
                strike_type,
                lookback_period,
            )
    # The full-life formulas are evaluated on a whole grid at once, with the options at
    # expiration (worth their payoff) priced a year out and replaced below.
    expired = np.asarray(time_to_expiration) <= 0
    if np.ndim(expired):
        time_to_expiration = np.where(expired, 1.0, time_to_expiration)
    elif time_to_expiration <= 0:
        if not fixed:
            return 0.0
        return max((strike_price - stock_price) * (1 if put_option else -1), 0.0)

    cost_of_carry = risk_free_rate - dividend_yield
    arguments = (
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        put_option,
        fixed,
        lookback_period,
    )

    # An overflow of the partial-time formulas is reported below, not as a warning.
    with np.errstate(over="ignore", invalid="ignore"):
        if abs(cost_of_carry) < _SMALL_CARRY:
            value = 0.5 * (
                _lookback_value(*arguments[:3], _SMALL_CARRY, *arguments[3:])
                + _lookback_value(*arguments[:3], -_SMALL_CARRY, *arguments[3:])
            )
        else:
            value = _lookback_value(*arguments[:3], cost_of_carry, *arguments[3:])

    # The partial-time formulas integrate powers e^(2bx/σ²) over the lognormal price at t1,
    # which overflow once 2|b|/σ² reaches the hundreds (a volatility of about 1% at b = 4%).
    if not np.all(np.isfinite(value)):
        raise ValueError(
            f"The partial-time lookback can not be evaluated at a volatility of {volatility!r}: "
            f"2|r - q| / σ² = {2 * abs(cost_of_carry) / volatility**2:,.0f} overflows its "
            "formulas. The price path is then close to deterministic; a lookback_period of 1 "
            "takes any volatility."
        )

    if np.ndim(expired):
        payoff = (
            np.maximum((strike_price - stock_price) * (1 if put_option else -1), 0.0)
            if fixed
            else 0.0
        )
        return np.where(expired, payoff, value)
    return float(value)


def _double_barrier_integral(
    stock_price: float,
    strike_price: float | np.ndarray,
    lower_barrier: float,
    upper_barrier: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float | np.ndarray,
    lower_limit: float | np.ndarray,
    upper_limit: float | np.ndarray,
    stock_weight: float,
    strike_weight: float,
) -> float | np.ndarray:
    """
    The undiscounted expectation of (stock_weight * S(T) - strike_weight * K) over the paths
    that stay between the barriers and end between lower_limit and upper_limit, with the
    method of images of Ikeda and Kunitomo (1992) for flat barriers.

    With x = ln(S(T) / S), l = ln(L / S), u = ln(U / S), Z = u - l, ν = b - σ²/2 and f the
    normal density of x, the density of the paths that stay inside is:

    - p(x) = Σ_n [e^(2nZν / σ²) * f(x - 2nZ) - e^(2ν(l - nZ) / σ²) * f(x - 2l + 2nZ)]

    The sum is taken over n = -N, ..., N with N large enough that the next image lies twelve
    standard deviations of x beyond the corridor at the longest expiration. The strikes,
    expirations and limits are arrays of one shape, priced at once (or numbers, for one
    option).
    """
    scalar = np.ndim(time_to_expiration) == 0
    strike_price, time_to_expiration, lower_limit, upper_limit = (
        np.atleast_1d(np.asarray(value, dtype=float))
        for value in (strike_price, time_to_expiration, lower_limit, upper_limit)
    )
    deviation = (volatility * np.sqrt(time_to_expiration))[:, None]
    drift = cost_of_carry - 0.5 * volatility**2
    lower = np.log(lower_barrier / stock_price)
    width = np.log(upper_barrier / lower_barrier)
    low = np.log(lower_limit / stock_price)[:, None]
    high = np.log(upper_limit / stock_price)[:, None]
    images = int(min(np.ceil(6 * np.max(deviation, initial=0.0) / width) + 2, 1_000))
    steps = np.arange(-images, images + 1)
    time = time_to_expiration[:, None]

    value = np.zeros(time_to_expiration.shape)
    for shifts, weights, side in (
        (2 * steps * width, 2 * steps * width * drift / volatility**2, 1.0),
        (
            2 * lower - 2 * steps * width,
            2 * drift * (lower - steps * width) / volatility**2,
            -1.0,
        ),
    ):
        mean = shifts + drift * time
        stock_part = np.exp(shifts + cost_of_carry * time) * (
            ndtr((high - mean - volatility**2 * time) / deviation)
            - ndtr((low - mean - volatility**2 * time) / deviation)
        )
        strike_part = ndtr((high - mean) / deviation) - ndtr((low - mean) / deviation)
        part = (
            stock_weight * stock_price * stock_part
            - strike_weight * strike_price[:, None] * strike_part
        )
        # An image far beyond the corridor adds nothing, also where its weight overflows.
        value += side * np.sum(np.where(part == 0, 0.0, np.exp(weights) * part), axis=1)

    value = np.where(upper_limit <= lower_limit, 0.0, value)
    return float(value[0]) if scalar else value


def get_double_barrier_option(
    stock_price: float,
    strike_price: float | np.ndarray,
    lower_barrier: float,
    upper_barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    knock_type: str = "out",
) -> float | np.ndarray:
    """
    Calculate the price of a double barrier European option: what an option is worth that knocks
    out (or in) when the stock price touches either of two flat, continuously watched barriers,
    with the formula of Ikeda and Kunitomo (1992).

    The knock-out option pays the vanilla payoff only when the stock price stays between the
    lower barrier L and the upper barrier U up to expiration. With the density of those paths
    from the method of images (Σ over n of the normal density and its reflections in the two
    barriers), the price is a series of Black-Scholes terms:

    - Call = S * e^((b - r)T) * Σ_n [(U^n / L^n)^μ * (N(d1) - N(d2)) - (L^(n+1) / (U^n * S))^μ * (N(d3) - N(d4))]
      - K * e^(-rT) * Σ_n [(U^n / L^n)^(μ - 2) * (N(d1 - σ√T) - N(d2 - σ√T))
      - (L^(n+1) / (U^n * S))^(μ - 2) * (N(d3 - σ√T) - N(d4 - σ√T))]
    - d1 = (ln(S * U^2n / (K' * L^2n)) + (b + σ²/2)T) / (σ√T), d2 = (ln(S * U^2n / (U * L^2n)) + (b + σ²/2)T) / (σ√T)
    - d3 = (ln(L^(2n+2) / (K' * S * U^2n)) + (b + σ²/2)T) / (σ√T),
      d4 = (ln(L^(2n+2) / (U * S * U^2n)) + (b + σ²/2)T) / (σ√T)
    - μ = 2b / σ² + 1, K' = max(K, L) the lowest price at which the call pays
    - Knock-in = Black-Scholes price - Knock-out

    And the put the same way over the prices between L and min(K, U). The series is summed
    until the terms are below machine precision. With the stock price outside the corridor
    the knock-out is worth 0 and the knock-in the vanilla price.

    Referenced against:
    - QuantLib, AnalyticDoubleBarrierEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_double_barrier_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The option's strike price, or an array of them.
        lower_barrier (float): The lower barrier L, below the stock price.
        upper_barrier (float): The upper barrier U, above the stock price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float | np.ndarray): The time to expiration in years, or an
            array of them (broadcast against the strike prices).
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the option is a put option. Defaults to False.
        knock_type (str): The knock. Options are: "out" (knock-out), "in" (knock-in).
            Defaults to "out".

    Returns:
        float | np.ndarray: The double barrier option value, an array in the shape of the
        strike prices and expirations when either is an array.

    Raises:
        ValueError: if knock_type is not "in" or "out", the lower barrier is not below the
            upper barrier or volatility is not positive.

    Notes:
        Reference: Ikeda, M., & Kunitomo, N. (1992). "Pricing Options with Curved
        Boundaries." Mathematical Finance, 2(4), 275-298. Haug, E. G. (2007). The Complete
        Guide to Option Pricing Formulas, 2nd ed., section 4.17.6.
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    validate_numeric_inputs(
        arrays=True,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
    )
    _validate_volatility(volatility)
    validate_positive_inputs(stock_price=stock_price, lower_barrier=lower_barrier)
    check_option("knock_type", knock_type, ("in", "out"))
    if lower_barrier >= upper_barrier:
        raise ValueError(
            f"lower_barrier ({lower_barrier}) must be below upper_barrier ({upper_barrier})."
        )

    sign = -1 if put_option else 1
    cost_of_carry = risk_free_rate - dividend_yield
    strike_prices, times = np.broadcast_arrays(
        np.asarray(strike_price, dtype=float),
        np.asarray(time_to_expiration, dtype=float),
    )
    shape = strike_prices.shape
    strike_prices, times = strike_prices.ravel(), times.ravel()
    expired = times <= 0

    # get_black_scholes gives the exercise value at expiration.
    vanilla = np.asarray(
        get_black_scholes(
            stock_price,
            strike_prices,
            risk_free_rate,
            volatility,
            times,
            dividend_yield,
            put_option,
        ),
        dtype=float,
    )

    inside = lower_barrier < stock_price < upper_barrier

    if not inside:
        knock_out = np.zeros(times.shape)
    else:
        knock_out = vanilla.copy()
        if not expired.all():
            alive = ~expired
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                knock_out[alive] = np.exp(
                    -risk_free_rate * times[alive]
                ) * _double_barrier_integral(
                    stock_price,
                    strike_prices[alive],
                    lower_barrier,
                    upper_barrier,
                    cost_of_carry,
                    volatility,
                    times[alive],
                    (
                        np.full(alive.sum(), float(lower_barrier))
                        if put_option
                        else np.maximum(strike_prices[alive], lower_barrier)
                    ),
                    (
                        np.minimum(strike_prices[alive], upper_barrier)
                        if put_option
                        else np.full(alive.sum(), float(upper_barrier))
                    ),
                    float(sign),
                    float(sign),
                )

    knock_out = np.maximum(knock_out, 0.0)
    value = (
        knock_out if knock_type == "out" else np.maximum(vanilla - knock_out, 0.0)
    ).reshape(shape)

    return value if value.ndim else float(value)


def _barrier_terms(
    stock_price_power: str,
    sign: int,
    direction: int,
    strike_price: float,
    barrier: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
) -> list[tuple[float, float, float, float]]:
    """
    A term A, B, C or D of Reiner and Rubinstein (1991) (see exotics_model._barrier_term) as
    a sum of terms coefficient * S^a * N(α + β ln S) of the log stock price, so that its
    expectation at a future date has a closed form (see _expected_term).
    """
    root_time = volatility * np.sqrt(time_to_expiration)
    mu = (cost_of_carry - 0.5 * volatility**2) / volatility**2
    carry_discount = np.exp((cost_of_carry - risk_free_rate) * time_to_expiration)
    discount = np.exp(-risk_free_rate * time_to_expiration)
    log_strike = np.log(strike_price)
    log_barrier = np.log(barrier)
    offset = (1 + mu) * root_time

    if stock_price_power in ("A", "B"):
        level = log_strike if stock_price_power == "A" else log_barrier
        intercept = -level / root_time + offset
        return [
            (sign * carry_discount, 1.0, sign * intercept, sign / root_time),
            (
                -sign * strike_price * discount,
                0.0,
                sign * (intercept - root_time),
                sign / root_time,
            ),
        ]

    level = 2 * log_barrier - log_strike if stock_price_power == "C" else log_barrier
    intercept = level / root_time + offset
    return [
        (
            sign * carry_discount * barrier ** (2 * (mu + 1)),
            1.0 - 2 * (mu + 1),
            direction * intercept,
            -direction / root_time,
        ),
        (
            -sign * strike_price * discount * barrier ** (2 * mu),
            -2 * mu,
            direction * (intercept - root_time),
            -direction / root_time,
        ),
    ]


# The Reiner and Rubinstein (1991) knock-out without rebate as a combination of their terms
# A, B, C and D, by put option, barrier direction and whether the strike lies beyond the barrier.
_KNOCK_OUT_COMBINATIONS: dict[tuple[bool, str, bool], dict[str, int]] = {
    (False, "down", True): {"A": 1, "C": -1},
    (False, "down", False): {"B": 1, "D": -1},
    (False, "up", True): {},
    (False, "up", False): {"A": 1, "B": -1, "C": 1, "D": -1},
    (True, "down", True): {"A": 1, "B": -1, "C": 1, "D": -1},
    (True, "down", False): {},
    (True, "up", True): {"B": 1, "D": -1},
    (True, "up", False): {"A": 1, "C": -1},
}


def _knock_out_terms(
    put_option: bool,
    barrier_direction: str,
    strike_price: float,
    barrier: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
) -> list[tuple[float, float, float, float]]:
    """
    The Reiner and Rubinstein (1991) knock-out price without rebate as terms of the log stock
    price, for a stock price on the side of the barrier where the option is alive.
    """
    sign = -1 if put_option else 1
    direction = 1 if barrier_direction == "down" else -1
    beyond = np.asarray(strike_price > barrier)

    # The weight of each term where the strike lies beyond the barrier and where it does
    # not, so that an array of strikes on both sides takes its combination per strike.
    terms = []
    for name in "ABCD":
        weight = np.where(
            beyond,
            _KNOCK_OUT_COMBINATIONS[(put_option, barrier_direction, True)].get(name, 0),
            _KNOCK_OUT_COMBINATIONS[(put_option, barrier_direction, False)].get(
                name, 0
            ),
        )
        if not weight.any():
            continue
        weight = weight if weight.ndim else int(weight)
        terms.extend(
            (weight * coefficient, power, intercept, slope)
            for coefficient, power, intercept, slope in _barrier_terms(
                name,
                sign,
                direction,
                strike_price,
                barrier,
                risk_free_rate,
                cost_of_carry,
                volatility,
                time_to_expiration,
            )
        )
    return terms


def get_partial_time_barrier_option(
    stock_price: float,
    strike_price: float | np.ndarray,
    barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    barrier_direction: str = "down",
    knock_type: str = "out",
    barrier_window: str = "start",
    window_share: float = 0.5,
) -> float | np.ndarray:
    """
    Calculate the price of a partial-time single barrier option (Heynen & Kat, 1994): what an
    option is worth whose barrier is only watched during part of its life, from today to t1 (a
    start window) or from t1 to expiration (an end window).

    With t1 the end (start window) or the start (end window) of the watched period, Y = ln S(t1)
    normal with mean ln S + (b - σ²/2) * t1 and variance σ² * t1, and h = ln(H / S):

    - Start window, knock-out = e^(-r t1) * {E[BS(S(t1)) * 1{ηY > η ln H}]
      - (H / S)^(2μ) * E[BS(S(t1) * (H / S)²) * 1{η(Y + 2h) > η ln H}]}
    - End window, knock-out = e^(-r t1) * E[RR(S(t1)) * 1{ηY > η ln H}]
    - Knock-in = Black-Scholes price - Knock-out

    Where BS is the Black-Scholes price over T - t1 and RR the knock-out of Reiner and
    Rubinstein (1991) over T - t1 (see exotics_model.get_barrier_option), μ = (b - σ²/2) / σ²
    and η = 1 for a down barrier and -1 for an up barrier. The second term of the start window
    removes, by the reflection principle, the paths that cross the barrier before t1. Both
    prices are sums of terms S(t1)^a * N(α + β ln S(t1)), of which the expectations are closed
    form with the bivariate normal distribution (correlation √(t1 / T) between the price at t1
    and at T), the form in which Heynen and Kat give them. In the end window the option is
    knocked out when the stock price is beyond the barrier at t1 (Heynen and Kat's type B2).

    Referenced against:
    - QuantLib, AnalyticPartialTimeBarrierOptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_partial_time_barrier_option_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The option's strike price, or an array of them.
        barrier (float): The barrier level.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float | np.ndarray): The time to expiration in years, or an
            array of them (broadcast against the strike prices).
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the option is a put option. Defaults to False.
        barrier_direction (str): The side of the barrier. Options are: "down" (below the
            stock price), "up" (above the stock price). Defaults to "down".
        knock_type (str): The knock. Options are: "out" (knock-out), "in" (knock-in).
            Defaults to "out".
        barrier_window (str): The watched window. Options are: "start" (watched from today
            to t1), "end" (watched from t1 to expiration). Defaults to "start".
        window_share (float): The share of the life during which the barrier is watched, above
            0 and at most 1, so t1 = window_share * T for a start window and
            t1 = (1 - window_share) * T for an end window. Defaults to 0.5.

    Returns:
        float | np.ndarray: The partial-time barrier option value, an array in the shape of
        the strike prices and expirations when either is an array.

    Raises:
        ValueError: if a choice is not one of its values, window_share is not in (0, 1] or
            volatility is not positive.

    Notes:
        Reference: Heynen, R. C., & Kat, H. M. (1994). "Partial Barrier Options." The Journal
        of Financial Engineering, 3(3/4), 253-274. Haug, E. G. (2007). The Complete Guide to
        Option Pricing Formulas, 2nd ed., section 4.17.3.
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        barrier=barrier,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        window_share=window_share,
    )
    validate_numeric_inputs(
        arrays=True,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
    )
    _validate_volatility(volatility)
    validate_positive_inputs(stock_price=stock_price, barrier=barrier)
    check_option("barrier_direction", barrier_direction, ("down", "up"))
    check_option("knock_type", knock_type, ("in", "out"))
    check_option("barrier_window", barrier_window, ("start", "end"))
    if not 0 < window_share <= 1:
        raise ValueError(
            f"window_share must be above 0 and at most 1, received {window_share!r}."
        )

    strike_prices, times = np.broadcast_arrays(
        np.asarray(strike_price, dtype=float),
        np.asarray(time_to_expiration, dtype=float),
    )
    shape = strike_prices.shape
    strike_prices, times = strike_prices.ravel(), times.ravel()
    arguments = {
        "stock_price": stock_price,
        "barrier": barrier,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
        "put_option": put_option,
        "barrier_direction": barrier_direction,
    }

    # A barrier watched over the whole life is a regular barrier option, as is one at
    # expiration.
    value = np.asarray(
        get_barrier_option(
            **arguments,
            strike_price=strike_prices,
            time_to_expiration=times,
            knock_type=knock_type,
        ),
        dtype=float,
    ).reshape(-1)
    partial = (times > 0) if window_share < 1 else np.zeros(times.shape, dtype=bool)

    if partial.any():
        value[partial] = _partial_time_barrier_values(
            **arguments,
            strike_price=strike_prices[partial],
            time_to_expiration=times[partial],
            knock_type=knock_type,
            barrier_window=barrier_window,
            window_share=window_share,
        )

    value = value.reshape(shape)

    return value if value.ndim else float(value)


@np.errstate(divide="ignore", invalid="ignore", over="ignore")
def _partial_time_barrier_values(
    stock_price: float,
    strike_price: np.ndarray,
    barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: np.ndarray,
    dividend_yield: float,
    put_option: bool,
    barrier_direction: str,
    knock_type: str,
    barrier_window: str,
    window_share: float,
) -> np.ndarray:
    """
    The partial-time barrier option values of get_partial_time_barrier_option for arrays of
    strike prices and times to expiration above zero, with a window share below one.
    """
    vanilla = np.asarray(
        get_black_scholes(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiration,
            dividend_yield,
            put_option,
        ),
        dtype=float,
    )
    direction = 1 if barrier_direction == "down" else -1
    breached = direction * stock_price <= direction * barrier
    sign = -1 if put_option else 1
    cost_of_carry = risk_free_rate - dividend_yield
    mu = (cost_of_carry - 0.5 * volatility**2) / volatility**2
    log_barrier = np.log(barrier)

    if barrier_window == "start":
        if breached:
            knock_out = np.zeros(time_to_expiration.shape)
        else:
            middle = window_share * time_to_expiration
            remaining = time_to_expiration - middle
            mean = np.log(stock_price) + (cost_of_carry - 0.5 * volatility**2) * middle
            variance = volatility**2 * middle
            root_time = volatility * np.sqrt(remaining)
            intercept = (
                (cost_of_carry + 0.5 * volatility**2) * remaining - np.log(strike_price)
            ) / root_time
            black_scholes = [
                (
                    sign * np.exp((cost_of_carry - risk_free_rate) * remaining),
                    1.0,
                    sign * intercept,
                    sign / root_time,
                ),
                (
                    -sign * strike_price * np.exp(-risk_free_rate * remaining),
                    0.0,
                    sign * (intercept - root_time),
                    sign / root_time,
                ),
            ]
            reflection = 2 * np.log(barrier / stock_price)
            knock_out = np.exp(-risk_free_rate * middle) * (
                _expected_value(black_scholes, mean, variance, direction, log_barrier)
                - (barrier / stock_price) ** (2 * mu)
                * _expected_value(
                    black_scholes,
                    mean + reflection,
                    variance,
                    direction,
                    log_barrier,
                )
            )
    else:
        start = (1 - window_share) * time_to_expiration
        mean = np.log(stock_price) + (cost_of_carry - 0.5 * volatility**2) * start
        variance = volatility**2 * start
        knock_out = np.exp(-risk_free_rate * start) * _expected_value(
            _knock_out_terms(
                put_option,
                barrier_direction,
                strike_price,
                barrier,
                risk_free_rate,
                cost_of_carry,
                volatility,
                time_to_expiration - start,
            ),
            mean,
            variance,
            direction,
            log_barrier,
        )

    knock_out = np.minimum(np.maximum(knock_out, 0.0), vanilla)

    return knock_out if knock_type == "out" else vanilla - knock_out


def get_soft_barrier_option(
    stock_price: float,
    strike_price: float | np.ndarray,
    first_barrier: float,
    second_barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    barrier_direction: str = "down",
    knock_type: str = "out",
) -> float | np.ndarray:
    """
    Calculate the price of a soft barrier option (Hart & Ross, 1994): what an option is worth
    that knocks in or out gradually as the stock price moves through a band of barriers [L, U]
    instead of at once at a single barrier.

    A soft down-and-out option pays the vanilla payoff times the share of the band that the
    lowest price did not reach, (min S - L) / (U - L) between 0 and 1, and the soft up-and-out
    times (U - max S) / (U - L). That share is the average over the band of the indicators that
    a single barrier H was not touched, so the price is the average of the hard barrier prices:

    - Soft Knock-out = 1 / (U - L) * ∫_L^U Knock-out(H) dH
    - Soft Knock-in = Black-Scholes price - Soft Knock-out

    Where Knock-out(H) is the Reiner and Rubinstein (1991) price with barrier H (see
    exotics_model.get_barrier_option). The integral is taken with 64 Gauss-Legendre points on
    each side of the strike and the stock price, where the hard barrier price is smooth. A band
    of zero width is the hard barrier.

    Referenced against:
    - QuantLib, AnalyticSoftBarrierEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_soft_barrier_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float | np.ndarray): The option's strike price, or an array of them.
        first_barrier (float): One edge of the band of barriers.
        second_barrier (float): The other edge of the band.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float | np.ndarray): The time to expiration in years, or an
            array of them (broadcast against the strike prices).
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        put_option (bool): Whether the option is a put option. Defaults to False.
        barrier_direction (str): The side of the barriers. Options are: "down" (below the
            stock price), "up" (above the stock price). Defaults to "down".
        knock_type (str): The knock. Options are: "out" (knock-out), "in" (knock-in).
            Defaults to "out".

    Returns:
        float | np.ndarray: The soft barrier option value, an array in the shape of the
        strike prices and expirations when either is an array.

    Raises:
        ValueError: if barrier_direction or knock_type is not one of its values or volatility
            is not positive.

    Notes:
        Reference: Hart, I., & Ross, M. (1994). "Striking Continuity." Risk, 7(6), 51-56.
        Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas, 2nd ed., section
        4.17.5.
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        first_barrier=first_barrier,
        second_barrier=second_barrier,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    validate_numeric_inputs(
        arrays=True,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
    )
    _validate_volatility(volatility)
    check_option("barrier_direction", barrier_direction, ("down", "up"))
    check_option("knock_type", knock_type, ("in", "out"))

    arguments = {
        "stock_price": stock_price,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
        "put_option": put_option,
        "barrier_direction": barrier_direction,
    }
    lower, upper = sorted((first_barrier, second_barrier))

    if upper == lower:
        return get_barrier_option(
            **arguments,
            strike_price=strike_price,
            time_to_expiration=time_to_expiration,
            barrier=lower,
            knock_type=knock_type,
        )

    strike_prices, times = np.broadcast_arrays(
        np.asarray(strike_price, dtype=float),
        np.asarray(time_to_expiration, dtype=float),
    )
    shape = strike_prices.shape
    strike_prices, times = strike_prices.ravel()[:, None], times.ravel()[:, None]
    direction = 1 if barrier_direction == "down" else -1

    # The hard barrier price has kinks at the strike (the formula changes) and at the stock
    # price (the barrier is breached beyond it), so each smooth piece gets its own points, on
    # which the knock-out is one combination of the terms A to D, evaluated for all at once.
    # The pieces lie between the band edges, the strike and the stock price in order; a piece
    # of zero width (a strike outside the band) adds nothing.
    edges = np.sort(
        np.clip(
            np.hstack(
                np.broadcast_arrays(lower, upper, strike_prices, float(stock_price))
            ),
            lower,
            upper,
        ),
        axis=1,
    )
    term_inputs = {
        "phi": -1 if put_option else 1,
        "eta": direction,
        "stock_price": stock_price,
        "strike_price": strike_prices,
        "risk_free_rate": risk_free_rate,
        "cost_of_carry": risk_free_rate - dividend_yield,
        "volatility": volatility,
        "time_to_expiration": times,
    }
    knock_out = np.zeros(strike_prices.shape)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for piece in range(edges.shape[1] - 1):
            left, right = edges[:, [piece]], edges[:, [piece + 1]]
            middle = 0.5 * (left + right)
            # Beyond the stock price the barrier is breached and the knock-out is worth nothing.
            alive = (right > left) & (direction * middle < direction * stock_price)
            if not alive.any():
                continue
            barriers = left + 0.5 * (right - left) * (_LEGENDRE_NODES + 1)
            prices = 0.0
            for name in "ABCD":
                weight = np.where(
                    strike_prices > middle,
                    _KNOCK_OUT_COMBINATIONS[(put_option, barrier_direction, True)].get(
                        name, 0
                    ),
                    _KNOCK_OUT_COMBINATIONS[(put_option, barrier_direction, False)].get(
                        name, 0
                    ),
                )
                if weight.any():
                    prices = prices + np.where(
                        weight != 0,
                        weight
                        * _barrier_term(**term_inputs, barrier=barriers, term=name),
                        0.0,
                    )
            # Some knock-outs are worth nothing on a whole piece (an up-and-out call struck
            # above it).
            if np.ndim(prices) == 0:
                continue
            knock_out += np.where(
                alive, 0.5 * (right - left) * (prices @ _LEGENDRE_WEIGHTS)[:, None], 0.0
            )

        knock_out = (knock_out / (upper - lower)).ravel()

        if knock_type == "in":
            knock_out = (
                np.asarray(
                    get_black_scholes(
                        stock_price,
                        strike_prices.ravel(),
                        risk_free_rate,
                        volatility,
                        times.ravel(),
                        dividend_yield,
                        put_option,
                    ),
                    dtype=float,
                )
                - knock_out
            )

    # At expiration the knock-out pays the vanilla payoff times the share of the band not
    # reached.
    vanilla = np.maximum(
        (strike_prices.ravel() - stock_price) * (1 if put_option else -1), 0.0
    )
    alive_share = min(
        max(
            (
                stock_price - lower
                if barrier_direction == "down"
                else upper - stock_price
            )
            / (upper - lower),
            0.0,
        ),
        1.0,
    )
    at_expiration = vanilla * alive_share
    value = np.where(
        times.ravel() <= 0,
        at_expiration if knock_type == "out" else vanilla - at_expiration,
        knock_out,
    ).reshape(shape)

    return value if value.ndim else float(value)


def get_touch_option(
    stock_price: float,
    barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    option_type: str = "one-touch",
    cash_payout: float = 1.0,
    payment: str = "expiration",
) -> float:
    """
    Calculate the price of a one-touch or no-touch option (Reiner & Rubinstein, 1991): what a
    binary option on whether the stock price touches a barrier before expiration is worth.

    - One-touch: pays the cash payout when the barrier is touched, at that moment or at
      expiration.
    - No-touch: pays the cash payout at expiration when the barrier is never touched.

    With b = r - q, ν = b - σ²/2, μ = ν / σ², λ = √(μ² + 2r / σ²), h = ln(H / S) and η = 1 for
    a barrier below the stock price and -1 for one above it:

    - P(touch) = N(η(h - νT) / (σ√T)) + (H / S)^(2μ) * N(η(h + νT) / (σ√T))
    - One-touch at expiration = Cash * e^(-rT) * P(touch)
    - One-touch at hit = Cash * [(H / S)^(μ + λ) * N(ηz) + (H / S)^(μ - λ) * N(ηz - 2ηλσ√T)],
      z = h / (σ√T) + λσ√T
    - No-touch = Cash * e^(-rT) * (1 - P(touch))

    A barrier at or beyond the stock price is touched already: the one-touch pays at once (or
    the discounted cash at expiration) and the no-touch is worth 0.

    Referenced against:
    - QuantLib, AnalyticDigitalAmericanEngine (one-touch): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_digital_american_engine.html
    - QuantLib, AnalyticBinaryBarrierEngine (no-touch): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_binary_barrier_engine.html

    Args:
        stock_price (float): The current stock price.
        barrier (float): The barrier level, below or above the stock price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        option_type (str): The kind of touch option. Options are: "one-touch" (one-touch),
            "no-touch" (no-touch). Defaults to "one-touch".
        cash_payout (float): The cash paid. Defaults to 1.
        payment (str): When a one-touch pays. Options are: "expiration" (at expiration),
            "hit" (when the barrier is touched). A no-touch pays at expiration. Defaults to "expiration".

    Returns:
        float: The touch option value.

    Raises:
        ValueError: if option_type or payment is not one of its values, a no-touch is asked to
            pay at the hit or volatility is not positive.

    Notes:
        Reference: Reiner, E., & Rubinstein, M. (1991). "Unscrambling the Binary Code." Risk,
        4(9), 75-83. Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas, 2nd
        ed., section 4.19.
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        barrier=barrier,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        cash_payout=cash_payout,
    )
    _validate_volatility(volatility)
    validate_positive_inputs(stock_price=stock_price, barrier=barrier)
    check_option("option_type", option_type, ("one-touch", "no-touch"))
    check_option("payment", payment, ("expiration", "hit"))
    if option_type == "no-touch" and payment == "hit":
        raise ValueError("A no-touch option pays at expiration, not at a hit.")

    discount = np.exp(-risk_free_rate * max(time_to_expiration, 0.0))
    one_touch_paid = cash_payout if payment == "hit" else cash_payout * discount

    if barrier == stock_price or time_to_expiration <= 0:
        touched = barrier == stock_price
        if option_type == "one-touch":
            return float(one_touch_paid) if touched else 0.0
        return 0.0 if touched else float(cash_payout * discount)

    direction = 1 if barrier < stock_price else -1
    deviation = volatility * np.sqrt(time_to_expiration)
    drift = risk_free_rate - dividend_yield - 0.5 * volatility**2
    mu = drift / volatility**2
    log_barrier = np.log(barrier / stock_price)

    # Below this deviation even the logarithms of the powers lose their precision, and the
    # price path is the forward path S * e^((r - q)t), which touches H at t_h = ln(H / S) /
    # (r - q) when that lies within the life.
    if deviation < SMALL_DEVIATION:
        carry = risk_free_rate - dividend_yield
        hit_time = log_barrier / carry if carry else np.inf
        touched = 0 < hit_time <= time_to_expiration
        if option_type == "no-touch":
            return 0.0 if touched else float(cash_payout * discount)
        if not touched:
            return 0.0
        return float(
            cash_payout * np.exp(-risk_free_rate * hit_time)
            if payment == "hit"
            else cash_payout * discount
        )

    # (H / S)^p * N(y) is taken as e^(p * h + ln N(y)): at a low volatility the powers
    # μ ± λ and 2μ grow like 1/σ², so the power overflows where N is 0, while the product
    # tends to the deterministic answer (the forward path S * e^((r - q)t) reaches H or not).
    if option_type == "one-touch" and payment == "hit":
        lambda_ = np.sqrt(mu**2 + 2 * risk_free_rate / volatility**2)
        z = log_barrier / deviation + lambda_ * deviation
        return float(
            cash_payout
            * (
                np.exp((mu + lambda_) * log_barrier + log_ndtr(direction * z))
                + np.exp(
                    (mu - lambda_) * log_barrier
                    + log_ndtr(direction * z - 2 * direction * lambda_ * deviation)
                )
            )
        )

    touch_probability = ndtr(
        direction * (log_barrier - drift * time_to_expiration) / deviation
    ) + np.exp(
        2 * mu * log_barrier
        + log_ndtr(direction * (log_barrier + drift * time_to_expiration) / deviation)
    )
    touch_probability = min(max(float(touch_probability), 0.0), 1.0)

    if option_type == "one-touch":
        return float(cash_payout * discount * touch_probability)

    return float(cash_payout * discount * (1 - touch_probability))


def get_double_touch_option(
    stock_price: float,
    lower_barrier: float,
    upper_barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    option_type: str = "double-one-touch",
    cash_payout: float = 1.0,
) -> float:
    """
    Calculate the price of a double one-touch or double no-touch option (Hui, 1996), paid at
    expiration: what a binary option on whether the stock price touches either of two barriers
    is worth.

    The double no-touch pays when the stock price stays between the two barriers up to
    expiration, the double one-touch when it touches either barrier.

    - P(stay) = Σ_n [e^(2nZν / σ²) * (N((u - 2nZ - νT) / (σ√T)) - N((l - 2nZ - νT) / (σ√T)))
      - e^(2ν(l - nZ) / σ²) * (N((u - 2l + 2nZ - νT) / (σ√T)) - N((l - 2l + 2nZ - νT) / (σ√T)))]
    - Double no-touch = Cash * e^(-rT) * P(stay)
    - Double one-touch = Cash * e^(-rT) * (1 - P(stay))

    With l = ln(L / S), u = ln(U / S), Z = u - l and ν = b - σ²/2: the probability that the
    paths stay inside, from the method of images (see get_double_barrier_option). Hui gives
    the same probability as the sine series Σ_i 2πi / Z² * ... * e^(-(iπ / Z)²σ²T / 2), which
    converges fast for long and the image series for short times to expiration.

    Referenced against:
    - QuantLib, AnalyticDoubleBarrierBinaryEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_double_barrier_binary_engine.html

    Args:
        stock_price (float): The current stock price.
        lower_barrier (float): The lower barrier L.
        upper_barrier (float): The upper barrier U.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The continuous dividend yield. Defaults to 0.
        option_type (str): The kind of double touch option. Options are:
            "double-one-touch" (double one-touch), "double-no-touch" (double no-touch).
            Defaults to "double-one-touch".
        cash_payout (float): The cash paid. Defaults to 1.

    Returns:
        float: The double touch option value.

    Raises:
        ValueError: if option_type is not one of its values, the lower barrier is not below
            the upper barrier or volatility is not positive.

    Notes:
        Reference: Hui, C. H. (1996). "One-Touch Double Barrier Binary Option Values."
        Applied Financial Economics, 6(4), 343-346. Haug, E. G. (2007). The Complete Guide
        to Option Pricing Formulas, 2nd ed., section 4.19.5.
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        cash_payout=cash_payout,
    )
    _validate_volatility(volatility)
    validate_positive_inputs(stock_price=stock_price, lower_barrier=lower_barrier)
    check_option("option_type", option_type, ("double-one-touch", "double-no-touch"))
    if lower_barrier >= upper_barrier:
        raise ValueError(
            f"lower_barrier ({lower_barrier}) must be below upper_barrier ({upper_barrier})."
        )

    discount = np.exp(-risk_free_rate * max(time_to_expiration, 0.0))

    if not lower_barrier < stock_price < upper_barrier:
        stay = 0.0
    elif time_to_expiration <= 0:
        stay = 1.0
    else:
        # The expectation of a unit payoff (stock weight 0, strike weight -1 on a strike of 1).
        stay = _double_barrier_integral(
            stock_price,
            1.0,
            lower_barrier,
            upper_barrier,
            risk_free_rate - dividend_yield,
            volatility,
            time_to_expiration,
            lower_barrier,
            upper_barrier,
            0.0,
            -1.0,
        )
        stay = min(max(stay, 0.0), 1.0)

    if option_type == "double-no-touch":
        return float(cash_payout * discount * stay)

    return float(cash_payout * discount * (1 - stay))


def get_barrier_option_value(
    strike_price: float | np.ndarray,
    time_to_expiration: float | np.ndarray,
    stock_price: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    put_option: bool,
    knock_type: str,
    barrier_direction: str,
    barrier: float,
    upper_barrier: float,
    second_barrier: float | None = None,
    barrier_window: str = "full",
    window_share: float = 0.5,
    rebate: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the price of the barrier option a barrier setup describes: a double barrier
    option, a partial-time barrier option when the barrier is watched in a window, a soft
    barrier option when a second barrier is given, and a single barrier option otherwise.

    Args:
        strike_price (float | np.ndarray): The strike price(s).
        time_to_expiration (float | np.ndarray): The time(s) to expiration in years.
        stock_price (float): The current stock price.
        risk_free_rate (float): The continuously compounded risk free rate.
        volatility (float): The volatility of the stock.
        dividend_yield (float): The continuously compounded dividend yield.
        put_option (bool): Whether the option is a put.
        knock_type (str): Whether the barrier knocks the option "in" or "out".
        barrier_direction (str): "down", "up" or "double".
        barrier (float): The barrier, the lower barrier of a double barrier option and the
            first barrier of a soft barrier option.
        upper_barrier (float): The upper barrier of a double barrier option.
        second_barrier (float | None, optional): The second barrier of a soft barrier option.
            Defaults to None, no soft barrier.
        barrier_window (str, optional): The part of the option's life the barrier is watched:
            "full", "start" or "end". Defaults to "full".
        window_share (float, optional): The share of the option's life a partial-time window
            covers. Defaults to 0.5.
        rebate (float, optional): The rebate of a single barrier option. Defaults to 0.0.

    Returns:
        float | np.ndarray: The price of the option per strike price and time to expiration.
    """
    arguments = {
        "stock_price": stock_price,
        "strike_price": strike_price,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "time_to_expiration": time_to_expiration,
        "dividend_yield": dividend_yield,
        "put_option": put_option,
        "knock_type": knock_type,
    }
    if barrier_direction == "double":
        return get_double_barrier_option(
            **arguments, lower_barrier=barrier, upper_barrier=upper_barrier
        )
    if barrier_window != "full":
        return get_partial_time_barrier_option(
            **arguments,
            barrier=barrier,
            barrier_direction=barrier_direction,
            barrier_window=barrier_window,
            window_share=window_share,
        )
    if second_barrier is not None:
        return get_soft_barrier_option(
            **arguments,
            first_barrier=barrier,
            second_barrier=second_barrier,
            barrier_direction=barrier_direction,
        )
    return get_barrier_option(
        **arguments, barrier=barrier, barrier_direction=barrier_direction, rebate=rebate
    )
