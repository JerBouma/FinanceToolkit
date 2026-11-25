"""Short-Rate Models and Callable Bond Valuation Model"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import ncx2, norm

from financetoolkit.fixedincome.bonds.bond_model import get_cash_flow_periods
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.fixedincome.short_rates import trinomial_tree_model
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

MODELS = ("vasicek", "cir")
# The models of the option formulas: Hull-White and CIR++ are Vasicek and CIR shifted to fit
# today's curve (Brigo and Mercurio, 2006, sections 3.3 and 3.9).
OPTION_MODELS = ("vasicek", "hull_white", "cir", "cir++")
SCHEMES = ("euler", "exact")
# The tree steps per coupon period. With one step per period the call is valued too coarsely
# (the effective convexity of a callable bond can even turn positive); twenty steps match a
# continuous-time valuation to a fraction of a basis point.
STEPS_PER_PERIOD = 20
# The most steps an option-adjusted spread tree may have: its nodes and repricing time grow with
# the square of the steps, and 2,400 steps (30 years of semi-annual or 10 years of monthly
# coupons) price in a few seconds.
MAX_TREE_STEPS = 2_400
# Below this product x = aτ of the speed of mean reversion and the time, the Vasicek and
# Hull-White terms are taken from their Taylor series in x. The closed forms divide by a, a²
# or a³ terms that cancel to leading order: ln A of Vasicek and the variance of Hull-White
# lose a relative precision of about 1e-16 / x² (6e-8 at x = 1e-4, all of it near x = 1e-8),
# and are NaN at a = 0. At x = 0.1 the closed forms are within 1e-13, and the series of
# SPEED_SERIES_TERMS terms is exact to double precision for |x| up to 0.1 (the next term is
# below 1e-22), so the two agree across the switch.
SPEED_SERIES_SWITCH = 0.1
SPEED_SERIES_TERMS = 14
# The coefficients of x^m of the series (Taylor series of the closed forms in x = aτ):
# (1 - e^(-x)) / x, the Vasicek -(B - τ) / (2a²τ³) - B² / (4aτ³) and the Hull-White
# [τ - 2(1 - e^(-x)) / a + (1 - e^(-2x)) / (2a)] / (a²τ³).
DECAY_SERIES = np.array(
    [(-1) ** m / math.factorial(m + 1) for m in range(SPEED_SERIES_TERMS)]
)
VASICEK_VARIANCE_SERIES = np.array(
    [
        (-1) ** m * (2 ** (m + 1) - 1) / math.factorial(m + 3)
        for m in range(SPEED_SERIES_TERMS)
    ]
)
HULL_WHITE_VARIANCE_SERIES = np.array(
    [
        (-1) ** m * (2 ** (m + 2) - 2) / math.factorial(m + 3)
        for m in range(SPEED_SERIES_TERMS)
    ]
)


def _decay_integral(speed: float, times: np.ndarray) -> np.ndarray:
    """
    The integral (1 - e^(-aτ)) / a of e^(-as) from 0 to τ, which is τ at a = 0, from its
    series below |aτ| = SPEED_SERIES_SWITCH. It is the B(τ) of Vasicek and Hull-White.
    """
    times = np.asarray(times, dtype=float)
    x = speed * times
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.abs(x) < SPEED_SERIES_SWITCH,
            times * np.polynomial.polynomial.polyval(x, DECAY_SERIES),
            -np.expm1(-x) / speed,
        )


def calibrate_short_rate_model(
    rates: pd.Series, model: str = "vasicek", periods_per_year: int = 252
) -> dict[str, float]:
    """
    Calibrate the Vasicek (1977) or Cox-Ingersoll-Ross (1985) model to a history of a short
    rate by regressing the change of the rate on its level (the discretized dynamics):

        - Vasicek: dr = a (b - r) dt + σ dW
        - CIR: dr = a (b - r) dt + σ √r dW, estimated on Δr / √r

    with a the speed of mean reversion, b the long-run mean and σ the volatility. In both
    models the expected rate keeps φ = e^(-aΔt) of its distance to the long-run mean over a
    step of Δt, so the slope β of the regression gives the speed exactly, a = -ln(1 + β) / Δt,
    whatever the frequency of the data; the volatility follows from the variance of a step,
    σ² (1 - φ²) / (2a) for Vasicek and r σ² φ (1 - φ) / a + b σ² (1 - φ)² / (2a) for CIR
    (divided by r for the regression on Δr / √r, so the second term enters with the sample
    mean of 1 / r). Over a few decades of data the speed of a slowly reverting rate is
    overestimated (the small-sample bias of the regression near a unit root), so a short
    half-life of such a rate calls for caution.

    Args:
        rates (pd.Series): The short rate over time, as decimals.
        model (str, optional): The model. Options are: "vasicek" (Vasicek, 1977), "cir"
            (Cox-Ingersoll-Ross, 1985). Defaults to "vasicek".
        periods_per_year (int, optional): The observations per year. Defaults to 252.

    Returns:
        dict[str, float]: The Speed of Mean Reversion, Long-Run Mean, Volatility, Current
        Rate and Half-Life (in years) of the model.
    """
    check_option("model", model, MODELS)

    values = rates.dropna().to_numpy(dtype=float)
    if len(values) < 30:  # noqa: PLR2004
        raise ValueError("At least 30 observations are needed to calibrate the model.")
    if np.ptp(values) == 0:
        raise ValueError(
            "The rate does not move over the period (a flat history), so it has no volatility "
            "or mean reversion to calibrate the model to. Use a longer period or another maturity."
        )
    if model == "cir" and (rates.dropna() <= 0).any():
        at_zero = rates.dropna()[rates.dropna() <= 0]
        raise ValueError(
            f"The CIR model needs a rate above zero throughout, while the rate was at or below "
            f"zero on {len(at_zero)} dates, the last on {at_zero.index[-1]}. Start after that "
            "date or use a longer maturity."
        )

    dt = 1 / periods_per_year
    level, change = values[:-1], np.diff(values)

    if model == "vasicek":
        design = np.column_stack([np.ones_like(level), level])
        target = change
    else:
        root = np.sqrt(level)
        design = np.column_stack([1 / root, root])
        target = change / root
    (alpha, beta), *_ = np.linalg.lstsq(design, target, rcond=None)
    deviation = (target - design @ np.array([alpha, beta])).std(ddof=2)

    persistence = 1 + beta
    if 0 < persistence < 1:
        speed = -np.log(persistence) / dt
        mean = alpha / (1 - persistence)
        if model == "vasicek":
            volatility = deviation * np.sqrt(2 * speed / (1 - persistence**2))
        else:
            # The exact conditional variance of a CIR step is r σ² φ (1 - φ) / a
            # + b σ² (1 - φ)² / (2a); divided by r (the regression is on Δr / √r) and averaged
            # over the sample, the second term weighs with the mean of 1 / r. Leaving it out
            # overstates σ, the more so the faster the rate reverts and the lower it sits.
            volatility = deviation / np.sqrt(
                persistence * (1 - persistence) / speed
                + max(mean, 0.0)
                * (1 - persistence) ** 2
                / (2 * speed)
                * np.mean(1 / level)
            )
    else:
        # A rate that does not revert to a mean over the sample has no exact mapping; the
        # Euler discretization reports its (non-positive) speed.
        speed = -beta / dt
        mean = alpha / (speed * dt) if speed else np.nan
        volatility = deviation / np.sqrt(dt)

    return {
        "Speed of Mean Reversion": float(speed),
        "Long-Run Mean": float(mean),
        "Volatility": float(volatility),
        "Current Rate": float(values[-1]),
        "Half-Life": float(np.log(2) / speed) if speed > 0 else np.nan,
    }


def _bond_coefficients(
    model: str, speed: float, mean: float, volatility: float, maturities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    ln A(τ) and B(τ) of the Vasicek or CIR zero-coupon bond price P = A(τ) e^(-B(τ) r) of a
    bond maturing in τ years, see `get_model_yields`.
    """
    t = np.asarray(maturities, dtype=float)
    a, b, sigma = speed, mean, volatility

    if model == "vasicek":
        big_b = _decay_integral(a, t)
        # ln A = b(B - τ) + σ²[-(B - τ) / (2a²) - B² / (4a)], whose σ² terms cancel to
        # σ²τ³/6 as a goes to 0 (the Merton, or Ho-Lee, model), so below the switch they come
        # from their series, σ²τ³(1/6 - aτ/8 + 7(aτ)²/120 - ...).
        x = a * t
        with np.errstate(divide="ignore", invalid="ignore"):
            log_a = b * (big_b - t) + sigma**2 * np.where(
                np.abs(x) < SPEED_SERIES_SWITCH,
                t**3 * np.polynomial.polynomial.polyval(x, VASICEK_VARIANCE_SERIES),
                -(big_b - t) / (2 * a**2) - big_b**2 / (4 * a),
            )
    elif model == "cir":
        # B and ln A divided by e^(γT) above and below, as e^(γT) overflows once γT passes
        # 709.78 (a fitted speed of 35 at 30 years). With e = e^(-γT), x = 2σ²/(γ + a)² and
        # γ - a = 2σ²/(γ + a), ln A = -2abT/(γ + a) + (2ab/σ²)·ln(1 + x(1 - e)/(1 + xe)),
        # whose log1p keeps its precision when σ is small and 2ab/σ² is very large.  # noqa: E501
        gamma = np.sqrt(a**2 + 2 * sigma**2)
        decay = np.exp(-gamma * t)
        growth = -np.expm1(-gamma * t)
        big_b = 2 * growth / ((gamma + a) * growth + 2 * gamma * decay)
        ratio = 2 * sigma**2 / (gamma + a) ** 2
        log_a = -2 * a * b * t / (gamma + a) + (2 * a * b / sigma**2) * np.log1p(
            ratio * growth / (1 + ratio * decay)
        )
    else:
        raise InvalidArgumentError(f"The model must be one of {', '.join(MODELS)}.")

    return log_a, big_b


def get_model_yields(
    maturities: np.ndarray,
    model: str,
    speed: float,
    mean: float,
    volatility: float,
    rate: float,
) -> np.ndarray:
    """
    Calculate the zero-coupon yields the Vasicek or CIR model implies: the yield curve that
    the model's parameters and today's short rate give, from the closed-form bond price P(T)
    = A(T) e^(-B(T) r):

        - Vasicek: B = (1 - e^(-aT)) / a,
          ln A = (B - T)(a²b - σ²/2) / a² - σ² B² / (4a)
        - CIR: γ = √(a² + 2σ²), B = 2(e^(γT) - 1) / ((γ + a)(e^(γT) - 1) + 2γ),
          A = [2γ e^((a + γ)T/2) / ((γ + a)(e^(γT) - 1) + 2γ)]^(2ab/σ²), evaluated with
          e^(-γT) in place of e^(γT) so that it holds for any speed and maturity

    Args:
        maturities (np.ndarray): The maturities in years.
        model (str): The model. Options are: "vasicek" (Vasicek, 1977), "cir"
            (Cox-Ingersoll-Ross, 1985).
        speed (float): The speed of mean reversion a.
        mean (float): The long-run mean b.
        volatility (float): The volatility σ.
        rate (float): The current short rate r.

    Returns:
        np.ndarray: The continuously compounded yields.
    """
    t = np.asarray(maturities, dtype=float)
    log_a, big_b = _bond_coefficients(model, speed, mean, volatility, t)

    return -(log_a - big_b * rate) / t


def get_feller_ratio(speed: float, mean: float, volatility: float) -> float:
    """
    Calculate the Feller ratio of the Cox-Ingersoll-Ross (1985) model dr = a (b - r) dt + σ
    √r dW: whether the mean reversion is strong enough to keep the short rate away from
    zero.

        - Feller Ratio = 2ab / σ²

    The short rate never reaches zero when the ratio is at least 1 (the Feller condition
    2ab ≥ σ², Feller 1951); below 1 it touches zero and is reflected back. The exact
    simulation of `simulate_short_rate` holds either way, while an Euler scheme has to cut
    off negative rates, which biases it most when the ratio is low. The ratio is also the
    degrees of freedom of the non-central chi-squared distribution of the rate divided by 2.

    For more information, see the following sources:

    - Feller, W. (1951). "Two Singular Diffusion Problems." Annals of Mathematics, 54(1),
      173-182.
    - Cox, J.C., Ingersoll, J.E. and Ross, S.A. (1985). "A Theory of the Term Structure of
      Interest Rates." Econometrica, 53(2), 385-407.

    Shares similar logic:
    - FinancePy, satisfies_feller: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/cir_montecarlo.py#L56

    Args:
        speed (float): The speed of mean reversion a.
        mean (float): The long-run mean b.
        volatility (float): The volatility σ (of the rate divided by its square root).

    Returns:
        float: The Feller ratio; the condition holds when it is at least 1.

    As an example:

    ```python
    from financetoolkit.fixedincome.short_rates import short_rate_model

    short_rate_model.get_feller_ratio(speed=0.15, mean=0.03, volatility=0.05)
    ```

    Which returns:

    ```
    3.599999999999999
    ```
    """
    if volatility <= 0:
        raise ValueError("The volatility must be above zero.")
    return float(2 * speed * mean / volatility**2)


def simulate_short_rate(
    model: str,
    speed: float,
    mean: float | np.ndarray,
    volatility: float,
    rate: float,
    years: float = 10.0,
    steps_per_year: int = 12,
    simulations: int = 10000,
    seed: int | None = 0,
    scheme: str = "euler",
) -> np.ndarray:
    """
    Simulate short-rate paths with an Euler scheme or by sampling the exact distribution of
    the rate at the end of every step: the paths the short rate may follow in the Vasicek,
    Cox-Ingersoll-Ross or Hull-White model.

    For Hull-White (1990) the long-run mean varies over time, θ(t)/a, so that the model
    reproduces today's curve; pass it as an array with one value per step (`hull_white_mean`
    with the same scheme).

    The schemes are as follows, with Δ the step, φ = e^(-aΔ) and Z a standard normal draw:

        - Euler: r' = r + a (b - r) Δ + σ √Δ Z, with √r in the volatility for CIR (whose
          rate is kept at or above zero)
        - Exact, Vasicek and Hull-White: r' = r φ + b (1 - φ) + σ √((1 - φ²) / (2a)) Z, with
          the mean b held over the step
        - Exact, CIR: r' = c χ'²_d(λ), a non-central chi-squared draw with d = 4ab / σ²
          degrees of freedom and non-centrality λ = r φ / c, c = σ² (1 - φ) / (4a)

    The exact schemes have no discretization error at any step size: the distribution of
    the simulated rate is that of the model at every step. The CIR rate stays at or above
    zero by construction, whether or not the Feller condition 2ab ≥ σ² holds (see
    `get_feller_ratio`).

    For more information, see the following book:

    - Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering." Springer,
      sections 3.3 and 3.4.

    numpy's generator draws the non-central chi-squared values of the exact scheme, so the
    paths match FinancePy's in distribution rather than draw by draw.

    Shares similar logic:
    - FinancePy, CIRMonteCarlo: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/cir_montecarlo.py#L10
    - FinancePy, draw: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/cir_montecarlo.py#L126

    Args:
        model (str): The model. Options are: "vasicek" (Vasicek, 1977), "cir"
            (Cox-Ingersoll-Ross, 1985), "hull_white" (Hull-White, 1990).
        speed (float): The speed of mean reversion.
        mean (float | np.ndarray): The long-run mean, or one per step for Hull-White.
        volatility (float): The volatility.
        rate (float): The starting rate.
        years (float, optional): The horizon. Defaults to 10.
        steps_per_year (int, optional): The steps per year, at least 1. Defaults to 12.
        simulations (int, optional): The number of paths, at least 1. Defaults to 10,000.
        seed (int | None, optional): The random seed. Defaults to 0.
        scheme (str, optional): The simulation scheme. Options are: "euler" (an Euler
            scheme), "exact" (sampling the exact distribution of the rate). Defaults to
            "euler".

    Returns:
        np.ndarray: The rates, paths by steps (including the start).

    Raises:
        InvalidArgumentError: If the model or scheme is not one of its options.
        ValueError: If steps_per_year or simulations is below 1.

    As an example:

    ```python
    from financetoolkit.fixedincome.short_rates import short_rate_model

    paths = short_rate_model.simulate_short_rate(
        "cir", speed=0.15, mean=0.03, volatility=0.05, rate=0.04, years=5,
        steps_per_year=1, scheme="exact"
    )

    paths.mean(axis=0).round(4)
    ```

    Which returns:

    ```
    array([0.04  , 0.0387, 0.0374, 0.0363, 0.0355, 0.0347])
    ```
    """
    check_option("model", model, (*MODELS, "hull_white"))
    check_option("scheme", scheme, SCHEMES)
    if steps_per_year < 1 or simulations < 1:
        raise ValueError(
            f"steps_per_year and simulations must be at least 1, received {steps_per_year!r} "
            f"and {simulations!r}."
        )
    rng = np.random.default_rng(seed)
    steps = int(round(years * steps_per_year))
    dt = 1 / steps_per_year
    means = np.broadcast_to(np.asarray(mean, dtype=float), (steps,))
    rates = np.empty((simulations, steps + 1))
    rates[:, 0] = rate

    if scheme == "exact":
        persistence = np.exp(-speed * dt)
        if model == "cir":
            if means.min() <= 0:
                raise ValueError(
                    "The exact CIR scheme needs a long-run mean above zero."
                )
            scale = volatility**2 * (1 - persistence) / (4 * speed)
            degrees = 4 * speed * means / volatility**2
            for step in range(steps):
                rates[:, step + 1] = scale * rng.noncentral_chisquare(
                    degrees[step], rates[:, step] * persistence / scale
                )
        else:
            deviation = volatility * np.sqrt((1 - persistence**2) / (2 * speed))
            shocks = rng.standard_normal((simulations, steps))
            for step in range(steps):
                rates[:, step + 1] = (
                    rates[:, step] * persistence
                    + means[step] * (1 - persistence)
                    + deviation * shocks[:, step]
                )
        return rates

    shocks = rng.standard_normal((simulations, steps))
    for step in range(steps):
        current = rates[:, step]
        diffusion = volatility * np.sqrt(dt) * shocks[:, step]
        if model == "cir":
            diffusion = diffusion * np.sqrt(np.maximum(current, 0))
        rates[:, step + 1] = current + speed * (means[step] - current) * dt + diffusion
        if model == "cir":
            rates[:, step + 1] = np.maximum(rates[:, step + 1], 0)

    return rates


def hull_white_mean(
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    speed: float,
    volatility: float,
    rate: float,
    years: float,
    steps_per_year: int,
    scheme: str = "euler",
) -> np.ndarray:
    """
    Calculate the long-run mean per step that makes the Hull-White model fit today's curve:
    the time-varying level the short rate reverts to so that the model reproduces the zero
    curve.

        E[r(t)] = f(t) + σ² (1 - e^(-at))² / (2a²)

    with f the forward rate of the (continuously compounded) zero curve. Rather than
    differentiating the forward curve for θ(t), which spikes where an interpolated curve has
    a kink, the mean of every step is set so that the expected rate reaches E[r(t)] at the
    end of the step exactly, whatever the step size: m = E_i + (E_(i+1) - E_i) / (aΔ) for
    the Euler scheme and m = (E_(i+1) - E_i φ) / (1 - φ), φ = e^(-aΔ), for the exact scheme
    of `simulate_short_rate`. The forward rate of a step is the one over that step of the
    zero curve interpolated linearly.

    Args:
        maturities (np.ndarray): The maturities of the zero curve in years.
        zero_rates (np.ndarray): The continuously compounded zero rates.
        speed (float): The speed of mean reversion a.
        volatility (float): The volatility σ.
        rate (float): The short rate today.
        years (float): The horizon.
        steps_per_year (int): The steps per year.
        scheme (str, optional): The scheme of the simulation. Options are: "euler" (an Euler
            scheme), "exact" (sampling the exact distribution of the rate). Defaults to
            "euler".

    Returns:
        np.ndarray: The long-run mean per step.
    """
    check_option("scheme", scheme, SCHEMES)
    steps = int(round(years * steps_per_year))
    dt = 1 / steps_per_year
    times = np.arange(steps + 2) * dt
    log_discount = np.interp(times, maturities, zero_rates) * times
    forward = np.diff(log_discount) / dt
    convexity = volatility**2 * (1 - np.exp(-speed * times[:-1])) ** 2 / (2 * speed**2)
    # The expected rate of every step: today's rate, then the forward rate of the step plus
    # the convexity term.
    expected = np.concatenate([[rate], (forward + convexity)[1:]])
    if scheme == "exact":
        persistence = np.exp(-speed * dt)
        return (expected[1:] - expected[:-1] * persistence) / (1 - persistence)
    return expected[:-1] + (expected[1:] - expected[:-1]) / (speed * dt)


def _check_option_model(
    model: str,
    mean: float | None,
    rate: float | None,
    zero_curve: pd.Series | None,
    volatility: float,
) -> None:
    """
    Check that the inputs of a short-rate option formula fit the model: Vasicek and CIR take
    their own curve from the mean and today's rate, Hull-White and CIR++ fit a zero curve,
    and the CIR formulas divide by the volatility.
    """
    check_option("model", model, OPTION_MODELS)
    if model in ("hull_white", "cir++") and zero_curve is None:
        raise ValueError(f"The {model} model is fitted to a curve; pass zero_curve.")
    if model in ("vasicek", "cir") and zero_curve is not None:
        raise ValueError(
            f"The {model} model implies its own curve; use 'hull_white' or 'cir++' to fit "
            "the zero_curve."
        )
    if model != "hull_white" and (mean is None or rate is None):
        raise ValueError(f"The {model} model needs the mean and the rate.")
    if model in ("cir", "cir++") and volatility <= 0:
        raise ValueError(
            f"The {model} model needs a volatility above zero, received {volatility!r}."
        )


def _discount(
    model: str,
    speed: float,
    mean: float | None,
    volatility: float,
    rate: float | None,
    zero_curve: pd.Series | None,
    times: np.ndarray,
) -> np.ndarray:
    """
    The discount factors P(0, t) of a short-rate model: the closed form of Vasicek and CIR,
    the zero curve itself for Hull-White and CIR++, which reproduce it.
    """
    if model in ("hull_white", "cir++"):
        return discount_factors(zero_curve, times)
    log_a, big_b = _bond_coefficients(model, speed, mean, volatility, times)
    return np.exp(log_a - big_b * rate)


def _bond_price_at(
    model: str,
    speed: float,
    mean: float | None,
    volatility: float,
    rate: float | None,
    zero_curve: pd.Series | None,
    expiry: float,
    maturities: np.ndarray,
    state: float,
) -> np.ndarray:
    """
    The price at time T = expiry of the zero-coupon bonds maturing at the maturities, given
    the state of the model at T: the short rate for Vasicek and CIR, the process x = r - φ
    without the deterministic shift for Hull-White (written as the G1++ model, x(0) = 0) and
    CIR++ (Brigo and Mercurio, 2006, equations 3.39 and 3.77):

        - Vasicek and CIR: P(T, S) = A(S - T) e^(-B(S - T) r)
        - Hull-White: P(T, S) = P(0, S) / P(0, T) · e^(½ [V(T, S) - V(0, S) + V(0, T)] - B x),
          V(t, T) = σ²/a² [T - t + 2/a e^(-a(T - t)) - 1/(2a) e^(-2a(T - t)) - 3/(2a)]
        - CIR++: P(T, S) = Φ(T, S) A(S - T) e^(-B(S - T) x), with
          Φ(T, S) = P(0, S) P_CIR(0, T) / (P(0, T) P_CIR(0, S))
    """
    tau = maturities - expiry
    if model == "hull_white":
        a = speed

        def variance(t):
            # σ²/a² [t - 2(1 - e^(-at)) / a + (1 - e^(-2at)) / (2a)], which cancels to
            # σ²t³/3 as a goes to 0 (the Ho-Lee model), so its series below the switch.
            t = np.asarray(t, dtype=float)
            x = a * t
            with np.errstate(divide="ignore", invalid="ignore"):
                return volatility**2 * np.where(
                    np.abs(x) < SPEED_SERIES_SWITCH,
                    t**3
                    * np.polynomial.polynomial.polyval(x, HULL_WHITE_VARIANCE_SERIES),
                    (t - 2 * _decay_integral(a, t) + _decay_integral(2 * a, t)) / a**2,
                )

        ratio = discount_factors(zero_curve, maturities) / discount_factors(
            zero_curve, np.array([expiry])
        )
        exponent = 0.5 * (variance(tau) - variance(maturities) + variance(expiry))
        return ratio * np.exp(exponent - _decay_integral(a, tau) * state)

    log_a, big_b = _bond_coefficients(
        "vasicek" if model == "vasicek" else "cir", speed, mean, volatility, tau
    )
    price = np.exp(log_a - big_b * state)
    if model == "cir++":
        market = discount_factors(zero_curve, maturities) / discount_factors(
            zero_curve, np.array([expiry])
        )
        cir = _discount("cir", speed, mean, volatility, rate, None, maturities) / (
            _discount("cir", speed, mean, volatility, rate, None, np.array([expiry]))
        )
        price = price * market / cir
    return price


def _cir_bond_call(
    speed: float,
    mean: float,
    volatility: float,
    rate: float,
    expiry: np.ndarray,
    maturity: np.ndarray,
    strike: np.ndarray,
) -> np.ndarray:
    """
    The Cox-Ingersoll-Ross (1985) price of a European call on a zero-coupon bond, see
    `get_zero_coupon_bond_option`. The terms in e^(hT) are written with e^(-hT) so that
    they hold for any speed and expiry.
    """
    a, b, sigma = speed, mean, volatility
    h = np.sqrt(a**2 + 2 * sigma**2)
    log_a, big_b = _bond_coefficients("cir", a, b, sigma, maturity - expiry)
    p_expiry = _discount("cir", a, b, sigma, rate, None, expiry)
    p_maturity = _discount("cir", a, b, sigma, rate, None, maturity)
    decay = np.exp(-h * expiry)
    rho = 2 * h * decay / (sigma**2 * -np.expm1(-h * expiry))
    psi = (a + h) / sigma**2
    # ρ² r e^(hT), rewritten with e^(-hT) above and below.
    scaled_rate = 4 * h**2 * decay / (sigma**4 * np.expm1(-h * expiry) ** 2) * rate
    critical = (log_a - np.log(strike)) / big_b
    # The non-central chi-squared distribution is undefined at 0 degrees of freedom, the CIR
    # rate without drift (a speed or mean of 0, absorbed at zero), and continuous in them
    # there: 1e-12 degrees give its limit within 1e-11.
    degrees = max(4 * a * b / sigma**2, 1e-12)

    return p_maturity * ncx2.cdf(
        2 * critical * (rho + psi + big_b),
        degrees,
        2 * scaled_rate / (rho + psi + big_b),
    ) - strike * p_expiry * ncx2.cdf(
        2 * critical * (rho + psi), degrees, 2 * scaled_rate / (rho + psi)
    )


def get_zero_coupon_bond_option(
    option_expiry: float | np.ndarray,
    bond_maturity: float | np.ndarray,
    strike: float | np.ndarray,
    speed: float,
    volatility: float,
    mean: float | None = None,
    rate: float | None = None,
    zero_curve: pd.Series | None = None,
    model: str = "vasicek",
    option_type: str = "call",
) -> float | np.ndarray:
    """
    Calculate the price of a European option on a zero-coupon bond in a short-rate model, in
    closed form: what a call or put expiring at T on a bond paying 1 at S > T, struck at K,
    is worth (a bond price, so a put on the bond is a call on its yield).

    These options are the building blocks of caplets and floorlets (`get_caplet`) and,
    through Jamshidian's decomposition, of European swaptions (`get_swaption`).

    The formula is as follows:

    - Vasicek and Hull-White: ZBC = P(0, S) N(h) - K P(0, T) N(h - σ_P), with
      σ_P = σ (1 - e^(-a(S - T))) / a · √((1 - e^(-2aT)) / (2a)) and
      h = ln(P(0, S) / (K P(0, T))) / σ_P + σ_P / 2
    - CIR: ZBC = P(0, S) χ²(2r̂ [ρ + ψ + B(S - T)]; d, 2ρ² r e^(hT) / (ρ + ψ + B(S - T)))
      - K P(0, T) χ²(2r̂ [ρ + ψ]; d, 2ρ² r e^(hT) / (ρ + ψ)), with h = √(a² + 2σ²),
      ρ = 2h / (σ² (e^(hT) - 1)), ψ = (a + h) / σ², d = 4ab / σ² and
      r̂ = ln(A(S - T) / K) / B(S - T) the rate at which the bond at T is worth K
    - CIR++: ZBC = P(0, S) / P_CIR(0, S) · ZBC_CIR(K / Φ(T, S)), with
      Φ(T, S) = P(0, S) P_CIR(0, T) / (P(0, T) P_CIR(0, S))
    - Put (all models): ZBP = ZBC - P(0, S) + K P(0, T)

    where N is the standard normal and χ²(x; d, λ) the non-central chi-squared distribution
    function, a the speed of mean reversion, b the long-run mean, σ the volatility, r the
    short rate today and P(0, t) the discount factors: those of the model's own curve for
    Vasicek and CIR (see `get_model_yields`), and those of the zero curve for Hull-White and
    CIR++, which add a deterministic shift to Vasicek and CIR so that the model reproduces
    the curve (Hull-White is Vasicek with a time-varying mean). Hull-White and Vasicek share
    the formula: only P(0, t) differs.

    All rates are continuously compounded, as the short-rate models describe them; the zero
    curve is annually compounded, as the Toolkit's curves are returned, and read with
    P(t) = (1 + r(t))^(-t) (see `financetoolkit.fixedincome.helpers.discount_factors`).

    For more information, see the following sources:

    - Jamshidian, F. (1989). "An Exact Bond Option Formula." Journal of Finance, 44(1),
      205-209.
    - Hull, J. and White, A. (1990). "Pricing Interest-Rate-Derivative Securities." The
      Review of Financial Studies, 3(4), 573-592.
    - Cox, J.C., Ingersoll, J.E. and Ross, S.A. (1985). "A Theory of the Term Structure of
      Interest Rates." Econometrica, 53(2), 385-407.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, sections 3.2, 3.3 and 3.9 (CIR++, Brigo and Mercurio, 2001,
      Finance and Stochastics, 5(3), 369-387).

    Referenced against:
    - QuantLib, Vasicek.discountBondOption (vasicek): https://www.quantlib.org/reference/class_quant_lib_1_1_vasicek.html
    - QuantLib, HullWhite.discountBondOption (hull_white): https://www.quantlib.org/reference/class_quant_lib_1_1_hull_white.html
    - QuantLib, CoxIngersollRoss.discountBondOption (cir): https://www.quantlib.org/reference/class_quant_lib_1_1_cox_ingersoll_ross.html
    - QuantLib, ExtendedCoxIngersollRoss.discountBondOption (cir++): https://www.quantlib.org/reference/class_quant_lib_1_1_extended_cox_ingersoll_ross.html

    Shares similar logic:
    - FinancePy, HWTree.option_on_zcb (hull_white): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/hw_tree.py#L946

    Args:
        option_expiry (float | np.ndarray): The expiry T of the option in years.
        bond_maturity (float | np.ndarray): The maturity S of the bond in years, after the
            expiry.
        strike (float | np.ndarray): The strike K, as a price per 1 of face value.
        speed (float): The speed of mean reversion a.
        volatility (float): The volatility σ of the short rate (for CIR and CIR++ of the
            rate divided by its square root).
        mean (float | None, optional): The long-run mean b; not used by Hull-White.
            Defaults to None.
        rate (float | None, optional): The short rate today (the start x(0) of the CIR
            process for CIR++); not used by Hull-White. Defaults to None.
        zero_curve (pd.Series | None, optional): The annually compounded zero-coupon rates
            (decimals) by maturity in years, which Hull-White and CIR++ fit. Defaults to
            None.
        model (str, optional): The model. Options are: "vasicek" (Vasicek, 1977),
            "hull_white" (Hull-White, 1990), "cir" (Cox-Ingersoll-Ross, 1985), "cir++"
            (CIR++ of Brigo and Mercurio, 2001). Defaults to "vasicek".
        option_type (str, optional): The option. Options are: "call" (the right to buy the
            bond), "put" (the right to sell it). Defaults to "call".

    Returns:
        float | np.ndarray: The option price per 1 of face value.

    As an example:

    ```python
    from financetoolkit.fixedincome.short_rates import short_rate_model

    short_rate_model.get_zero_coupon_bond_option(
        option_expiry=1, bond_maturity=5, strike=0.85, speed=0.1, volatility=0.01,
        mean=0.04, rate=0.03
    )
    ```

    Which returns:

    ```
    0.030226802572760114
    ```
    """
    _check_option_model(model, mean, rate, zero_curve, volatility)
    check_option("option_type", option_type, ("call", "put"))
    expiry = np.asarray(option_expiry, dtype=float)
    maturity = np.asarray(bond_maturity, dtype=float)
    strike = np.asarray(strike, dtype=float)
    if np.any(expiry <= 0) or np.any(maturity <= expiry):
        raise ValueError(
            "The option must expire after today and before the bond matures."
        )

    p_expiry = _discount(model, speed, mean, volatility, rate, zero_curve, expiry)
    p_maturity = _discount(model, speed, mean, volatility, rate, zero_curve, maturity)

    if model in ("vasicek", "hull_white"):
        # σ (1 - e^(-a(S - T))) / a · √((1 - e^(-2aT)) / (2a)), which is σ (S - T) √T at
        # a = 0 (the Ho-Lee model).
        sigma_p = (
            volatility
            * _decay_integral(speed, maturity - expiry)
            * np.sqrt(_decay_integral(2 * speed, expiry))
        )
        h = np.log(p_maturity / (strike * p_expiry)) / sigma_p + sigma_p / 2
        call = p_maturity * norm.cdf(h) - strike * p_expiry * norm.cdf(h - sigma_p)
    elif model == "cir":
        call = _cir_bond_call(speed, mean, volatility, rate, expiry, maturity, strike)
    else:
        cir_expiry = _discount("cir", speed, mean, volatility, rate, None, expiry)
        cir_maturity = _discount("cir", speed, mean, volatility, rate, None, maturity)
        shift = p_maturity * cir_expiry / (p_expiry * cir_maturity)
        call = (
            p_maturity
            / cir_maturity
            * _cir_bond_call(
                speed, mean, volatility, rate, expiry, maturity, strike / shift
            )
        )

    value = call if option_type == "call" else call - p_maturity + strike * p_expiry
    return float(value) if np.ndim(value) == 0 else value


def get_caplet(
    reset_time: float | np.ndarray,
    payment_time: float | np.ndarray,
    strike: float | np.ndarray,
    speed: float,
    volatility: float,
    mean: float | None = None,
    rate: float | None = None,
    zero_curve: pd.Series | None = None,
    model: str = "vasicek",
    option_type: str = "caplet",
    notional: float = 1.0,
) -> float | np.ndarray:
    """
    Calculate the price of a caplet or floorlet in a short-rate model: what an option on the
    simple rate L that is fixed at the reset time T and paid over τ = S - T at the payment
    time S is worth.

    A caplet pays τ max(L - K, 0) at S, which is worth (1 + Kτ) times a put on the
    zero-coupon bond from T to S struck at 1 / (1 + Kτ) at T; a floorlet is the matching
    call. A cap or floor is the sum of its caplets or floorlets, so pass the reset and
    payment times as arrays and sum the result.

    The formula is as follows:

    - Caplet = Notional · (1 + K τ) · ZBP(T, S, 1 / (1 + K τ))
    - Floorlet = Notional · (1 + K τ) · ZBC(T, S, 1 / (1 + K τ))

    with ZBP and ZBC the zero-coupon bond put and call of `get_zero_coupon_bond_option`.
    The strike K is a simple rate on the accrual τ in years (no day count), while the
    short rates and the model parameters are continuously compounded; the zero curve of
    Hull-White and CIR++ is annually compounded.

    For more information, see the following sources:

    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      chapter 32.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 2.6.

    Referenced against:
    - QuantLib, AnalyticCapFloorEngine (hull_white): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_cap_floor_engine.html

    Args:
        reset_time (float | np.ndarray): The time T at which the rate is fixed, in years.
        payment_time (float | np.ndarray): The time S at which it is paid, in years.
        strike (float | np.ndarray): The cap or floor rate K (decimal, simple interest).
        speed (float): The speed of mean reversion a.
        volatility (float): The volatility σ of the short rate.
        mean (float | None, optional): The long-run mean b; not used by Hull-White.
            Defaults to None.
        rate (float | None, optional): The short rate today (x(0) for CIR++); not used by
            Hull-White. Defaults to None.
        zero_curve (pd.Series | None, optional): The annually compounded zero-coupon rates
            by maturity in years, which Hull-White and CIR++ fit. Defaults to None.
        model (str, optional): The model. Options are: "vasicek" (Vasicek, 1977),
            "hull_white" (Hull-White, 1990), "cir" (Cox-Ingersoll-Ross, 1985), "cir++"
            (CIR++ of Brigo and Mercurio, 2001). Defaults to "vasicek".
        option_type (str, optional): The option. Options are: "caplet" (an option on a
            rising rate), "floorlet" (an option on a falling rate). Defaults to "caplet".
        notional (float, optional): The notional. Defaults to 1.

    Returns:
        float | np.ndarray: The price, in the currency of the notional.

    As an example:

    ```python
    from financetoolkit.fixedincome.short_rates import short_rate_model

    short_rate_model.get_caplet(
        reset_time=1, payment_time=1.25, strike=0.03, speed=0.1, volatility=0.01,
        mean=0.04, rate=0.03, notional=1_000_000
    )
    ```

    Which returns:

    ```
    1051.5510824737437
    ```
    """
    check_option("option_type", option_type, ("caplet", "floorlet"))
    accrual = np.asarray(payment_time, dtype=float) - np.asarray(
        reset_time, dtype=float
    )
    growth = 1 + np.asarray(strike, dtype=float) * accrual

    value = (
        notional
        * growth
        * get_zero_coupon_bond_option(
            reset_time,
            payment_time,
            1 / growth,
            speed,
            volatility,
            mean,
            rate,
            zero_curve,
            model,
            "put" if option_type == "caplet" else "call",
        )
    )
    return float(value) if np.ndim(value) == 0 else value


def get_swaption(
    option_expiry: float,
    tenor: float,
    fixed_rate: float,
    speed: float,
    volatility: float,
    mean: float | None = None,
    rate: float | None = None,
    zero_curve: pd.Series | None = None,
    model: str = "vasicek",
    swaption_type: str = "payer",
    fixed_frequency: int = 1,
    notional: float = 1.0,
) -> float:
    """
    Calculate the price of a European swaption in a one-factor short-rate model with
    Jamshidian's (1989) decomposition: what the right to enter a swap at the fixed rate is
    worth in the model.

    At expiry T a payer swaption is worth max(1 - CB, 0), with CB the value of a coupon bond
    paying the fixed rate on the fixed leg's dates and 1 at the end. In a one-factor model
    every bond price falls as the state of the model rises, so there is one state x* at
    which the coupon bond is worth exactly 1, and the option on the coupon bond splits into
    options on its zero-coupon bonds, each struck at its price in that state:

    The formula is as follows:

    - c_i = K τ_i, plus 1 on the last date T_n
    - x* solves SUM(c_i P(T, T_i; x*)) = 1, X_i = P(T, T_i; x*)
    - Payer Swaption = Notional · SUM(c_i ZBP(T, T_i, X_i))
    - Receiver Swaption = Notional · SUM(c_i ZBC(T, T_i, X_i))

    with K the fixed rate, τ_i the accrual of the i-th fixed period (1 / fixed_frequency,
    no day count, a short first period when the tenor is not a whole number of periods),
    P(T, T_i; x) the bond price at T in state x (the short rate for Vasicek and CIR, the
    rate less its deterministic shift for Hull-White and CIR++) and ZBP and ZBC the
    zero-coupon bond options of `get_zero_coupon_bond_option`. The floating leg is worth
    par at the start of the swap (one curve both projects and discounts). When the coupon
    bond stays below 1 in every state, which CIR and CIR++ allow as their rate cannot fall
    below zero (or the shift), the payer swaption is always exercised and worth the forward
    swap, P(0, T) - SUM(c_i P(0, T_i)), and the receiver nothing.

    The fixed rate is a nominal rate paid fixed_frequency times a year; the short rates are
    continuously compounded and the zero curve of Hull-White and CIR++ annually compounded.

    For more information, see the following sources:

    - Jamshidian, F. (1989). "An Exact Bond Option Formula." Journal of Finance, 44(1),
      205-209.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, sections 3.2.2 and 3.3.2.

    Referenced against:
    - QuantLib, JamshidianSwaptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_jamshidian_swaption_engine.html

    Args:
        option_expiry (float): The expiry T of the swaption, the start of the swap, in years.
        tenor (float): The length of the swap in years.
        fixed_rate (float): The fixed rate K of the swap (decimal).
        speed (float): The speed of mean reversion a.
        volatility (float): The volatility σ of the short rate.
        mean (float | None, optional): The long-run mean b; not used by Hull-White.
            Defaults to None.
        rate (float | None, optional): The short rate today (x(0) for CIR++); not used by
            Hull-White. Defaults to None.
        zero_curve (pd.Series | None, optional): The annually compounded zero-coupon rates
            by maturity in years, which Hull-White and CIR++ fit. Defaults to None.
        model (str, optional): The model. Options are: "vasicek" (Vasicek, 1977),
            "hull_white" (Hull-White, 1990), "cir" (Cox-Ingersoll-Ross, 1985), "cir++"
            (CIR++ of Brigo and Mercurio, 2001). Defaults to "vasicek".
        swaption_type (str, optional): The swaption. Options are: "payer" (the right to pay
            the fixed rate), "receiver" (the right to receive it). Defaults to "payer".
        fixed_frequency (int, optional): The fixed payments per year. Defaults to 1.
        notional (float, optional): The notional. Defaults to 1.

    Returns:
        float: The price, in the currency of the notional.

    As an example:

    ```python
    from financetoolkit.fixedincome.short_rates import short_rate_model

    short_rate_model.get_swaption(
        option_expiry=1, tenor=5, fixed_rate=0.035, speed=0.1, volatility=0.01,
        mean=0.04, rate=0.03, notional=1_000_000
    )
    ```

    Which returns:

    ```
    9586.798853610037
    ```
    """
    _check_option_model(model, mean, rate, zero_curve, volatility)
    check_option("swaption_type", swaption_type, ("payer", "receiver"))
    if option_expiry <= 0 or tenor <= 0:
        raise ValueError("The expiry and the tenor of the swaption must be positive.")

    times = (
        option_expiry + get_cash_flow_periods(tenor, fixed_frequency) / fixed_frequency
    )
    cash_flows = fixed_rate * np.diff(times, prepend=option_expiry)
    cash_flows[-1] += 1

    def coupon_bond(state: float) -> float:
        return (
            float(
                np.sum(
                    cash_flows
                    * _bond_price_at(
                        model,
                        speed,
                        mean,
                        volatility,
                        rate,
                        zero_curve,
                        option_expiry,
                        times,
                        state,
                    )
                )
            )
            - 1
        )

    # The coupon bond falls as the state rises. CIR's state cannot fall below zero; the
    # Gaussian models' bracket widens until it holds the root.
    low, high = (0.0, 1.0) if model in ("cir", "cir++") else (-1.0, 1.0)
    if model in ("cir", "cir++") and coupon_bond(0.0) <= 0:
        start, *payments = _discount(
            model,
            speed,
            mean,
            volatility,
            rate,
            zero_curve,
            np.concatenate([[option_expiry], times]),
        )
        forward = start - np.sum(cash_flows * np.array(payments))
        return float(notional * forward) if swaption_type == "payer" else 0.0
    while coupon_bond(low) < 0:
        low *= 2
    while coupon_bond(high) > 0:
        high *= 2
    state = brentq(coupon_bond, low, high, xtol=1e-14)
    strikes = _bond_price_at(
        model, speed, mean, volatility, rate, zero_curve, option_expiry, times, state
    )
    options = get_zero_coupon_bond_option(
        np.full_like(times, option_expiry),
        times,
        strikes,
        speed,
        volatility,
        mean,
        rate,
        zero_curve,
        model,
        "put" if swaption_type == "payer" else "call",
    )
    return float(notional * np.sum(cash_flows * options))


def get_cir_plus_plus_shift(
    times: float | np.ndarray,
    zero_curve: pd.Series,
    speed: float,
    mean: float,
    volatility: float,
    rate: float,
) -> float | np.ndarray:
    """
    Calculate the deterministic shift φ(t) of the CIR++ model of Brigo and Mercurio (2001):
    what it adds to a Cox-Ingersoll-Ross process x so that the short rate r = x + φ
    reproduces today's zero curve exactly while x keeps CIR's closed forms.

    The shift is the market's instantaneous forward rate less the one the CIR process
    implies:

    The formula is as follows:

    - φ(t) = f(0, t) - f_CIR(0, t)
    - f_CIR(0, t) = 2ab (e^(th) - 1) / (2h + (a + h)(e^(th) - 1))
      + x(0) · 4h² e^(th) / (2h + (a + h)(e^(th) - 1))², with h = √(a² + 2σ²)
    - f(0, t) = -∂ ln P(0, t) / ∂t, by central difference over 0.0001 years

    with P(0, t) = (1 + r(t))^(-t) the discount factors of the annually compounded zero
    curve, interpolated linearly between its maturities (so the market forward jumps at
    every maturity of the curve), and a, b, σ and x(0) the CIR parameters. The rate stays
    positive when φ(t) is at least zero for every t and the CIR parameters meet the Feller
    condition (`get_feller_ratio`). Bond and option prices follow from the CIR closed forms
    times the ratio of the market's and the CIR discount factors, see
    `get_zero_coupon_bond_option` and `get_swaption` for the CIR++ model. The shift is a
    continuously compounded rate.

    For more information, see the following sources:

    - Brigo, D. and Mercurio, F. (2001). "A Deterministic-Shift Extension of Analytically
      Tractable and Time-Homogeneous Short-Rate Models." Finance and Stochastics, 5(3),
      369-387.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 3.9.

    Shares similar logic:
    - QuantLib, ExtendedCoxIngersollRoss: https://www.quantlib.org/reference/class_quant_lib_1_1_extended_cox_ingersoll_ross.html

    Args:
        times (float | np.ndarray): The times t in years.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals) by
            maturity in years.
        speed (float): The speed of mean reversion a of the CIR process.
        mean (float): The long-run mean b of the CIR process.
        volatility (float): The volatility σ of the CIR process.
        rate (float): The start x(0) of the CIR process.

    Returns:
        float | np.ndarray: The shift φ(t) per time (continuously compounded).

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import short_rate_model

    zero_curve = pd.Series([0.04, 0.038, 0.037, 0.039, 0.042], index=[0.25, 1, 2, 5, 10])

    short_rate_model.get_cir_plus_plus_shift(
        [0.5, 1, 3, 7], zero_curve, speed=0.15, mean=0.03, volatility=0.05, rate=0.035
    ).round(4)
    ```

    Which returns:

    ```
    array([0.0027, 0.0013, 0.006 , 0.0125])
    ```
    """
    t = np.asarray(times, dtype=float)
    a, b, sigma = speed, mean, volatility
    epsilon = 1e-4
    low = np.maximum(t - epsilon, 0)
    high = t + epsilon
    forward = (
        np.log(discount_factors(zero_curve, low))
        - np.log(discount_factors(zero_curve, high))
    ) / (high - low)

    h = np.sqrt(a**2 + 2 * sigma**2)
    decay = np.exp(-h * t)
    growth = -np.expm1(-h * t)
    denominator = 2 * h * decay + (a + h) * growth
    cir_forward = (
        2 * a * b * growth / denominator + rate * 4 * h**2 * decay / denominator**2
    )
    shift = forward - cir_forward
    return float(shift) if np.ndim(shift) == 0 else shift


def get_option_adjusted_spread(
    price: float,
    coupon_rate: float,
    years_to_maturity: float,
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    volatility: float = 0.01,
    call_price: float | None = 100.0,
    years_to_first_call: float = 0.0,
    frequency: int = 2,
    face_value: float = 100.0,
    put_price: float | None = None,
    years_to_first_put: float = 0.0,
    model: str = "ho_lee",
    speed: float = 0.0,
) -> dict[str, float]:
    """
    Calculate the option-adjusted spread (OAS) of a callable or puttable bond: the spread
    over a short-rate tree calibrated to today's zero curve at which the model price, with
    the issuer's call and the investor's put exercised whenever optimal, equals the market
    price.

    Unlike the Z-spread it removes the value of the embedded options, so it measures the
    credit and liquidity premium alone. A put raises the bond's value, so a puttable bond's
    option value (the call less the put) is negative and its effective convexity positive.
    The effective duration and convexity follow from shifting the curve by 100 basis points
    up and down at an unchanged OAS. Black-Karasinski and Black-Derman-Toy need positive
    forward rates, so on a curve whose lowest forward rate on the tree is below 2% they
    shift it by half that forward rate instead.

    The bond is valued on a trinomial tree of the short rate fitted exactly to the zero
    curve (`trinomial_tree_model.build_short_rate_tree`, the two-stage procedure of Hull and
    White, 1994) in one of four models: Ho-Lee and Hull-White with normal rates,
    Black-Karasinski and Black-Derman-Toy with lognormal rates (so the volatility is of the
    rate for the first two and of its logarithm for the last two, and only Hull-White and
    Ho-Lee allow a curve with negative forward rates). Hull-White and Black-Karasinski
    revert to the curve at the speed a, which lowers the volatility of long rates and so the
    value of a long-dated call.

    The tree has twenty steps per coupon period and every coupon date among its dates. A
    bond bought between coupon dates also pays the interest accrued since the last coupon,
    so the market (clean) price plus the accrued interest is what the tree values. The call
    and the put can be exercised on the coupon dates from their first date on (Bermudan), at
    a clean price. The OAS and the tree rates are continuously compounded, the zero rates
    annually compounded.

    The formula is as follows:

    - OAS solves Tree Price(s) = Price + Accrued Interest, see
      `trinomial_tree_model.get_callable_bond_price`
    - Option Value = Straight Bond Price(OAS) - Price
    - Effective Duration = (P₋ - P₊) / (2 · P · Δy), Effective Convexity =
      (P₋ + P₊ - 2P) / (P · Δy²)

    with P₊ and P₋ the dirty prices at the OAS on trees fitted to the curve shifted 100
    basis points (Δy) up and down, or half the lowest forward rate for the lognormal models
    when that is less.

    For more information, see the following sources:

    - Ho, T.S.Y. and Lee, S.-B. (1986). "Term Structure Movements and Pricing Interest Rate
      Contingent Claims." Journal of Finance, 41(5), 1011-1029.
    - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure
      Models I: Single-Factor Models." Journal of Derivatives, 2(1), 7-16.
    - Fabozzi, F.J. (2016). "Bond Markets, Analysis, and Strategies." Pearson, 9th edition,
      chapter 18.

    Referenced against:
    - QuantLib, TreeCallableFixedRateBondEngine (ho_lee): https://www.quantlib.org/reference/class_quant_lib_1_1_tree_callable_fixed_rate_bond_engine.html

    Shares similar logic:
    - QuantLib, CallableFixedRateBond: https://www.quantlib.org/reference/class_quant_lib_1_1_callable_fixed_rate_bond.html
    - FinancePy, BDTTree.callable_puttable_bond_tree (black_derman_toy): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bdt_tree.py#L831

    Args:
        price (float): The market (clean) price of the bond.
        coupon_rate (float): The annual coupon rate.
        years_to_maturity (float): The years to maturity.
        maturities (np.ndarray): The maturities of the zero curve in years.
        zero_rates (np.ndarray): The zero rates (annually compounded).
        volatility (float, optional): The volatility σ: of the rate for Ho-Lee and Hull-White
            (0.01 is 1% a year), of its logarithm for Black-Karasinski and Black-Derman-Toy
            (0.2 is 20% of the rate). Defaults to 0.01.
        call_price (float | None, optional): The call price; None for a straight bond.
            Defaults to 100.
        years_to_first_call (float, optional): The years until the bond can first be called.
            Defaults to 0.
        frequency (int, optional): The coupons per year. Defaults to 2.
        face_value (float, optional): The face value. Defaults to 100.
        put_price (float | None, optional): The put price; None for a bond without a put.
            Defaults to None.
        years_to_first_put (float, optional): The years until the bond can first be put.
            Defaults to 0.
        model (str, optional): The short-rate model of the tree. Options are: "ho_lee"
            (Ho-Lee), "hull_white" (Hull-White), "black_karasinski" (Black-Karasinski),
            "black_derman_toy" (Black-Derman-Toy). Defaults to "ho_lee".
        speed (float, optional): The speed of mean reversion a of Hull-White and
            Black-Karasinski. Defaults to 0.

    Returns:
        dict[str, float]: The Option-Adjusted Spread, the Straight Bond Price (clean) and
        Option Value (the call less the put) at that spread, and the Effective Duration and
        Effective Convexity.

    As an example:

    ```python
    import numpy as np

    from financetoolkit.fixedincome.short_rates import short_rate_model

    short_rate_model.get_option_adjusted_spread(
        price=104, coupon_rate=0.04, years_to_maturity=10, maturities=np.array([0.5, 30]),
        zero_rates=np.array([0.035, 0.035]), call_price=None, put_price=100,
        years_to_first_put=3
    )
    ```

    Which returns:

    ```
    {'Option-Adjusted Spread': 0.006554025670571298,
     'Straight Bond Price': 98.88111024145738,
     'Option Value': -5.118889758542622,
     'Effective Duration': 5.639697383692862,
     'Effective Convexity': 138.6028126744038}
    ```
    """
    periods = years_to_maturity * frequency
    coupons = int(np.ceil(periods - 1e-9))
    if coupons < 1:
        raise ValueError("The bond must have at least one coupon left.")
    if coupons * STEPS_PER_PERIOD > MAX_TREE_STEPS:
        raise ValueError(
            f"A tree of {coupons * STEPS_PER_PERIOD:,} steps ({coupons:,} coupons of "
            f"{STEPS_PER_PERIOD} steps) is too large; keep the coupons (years to maturity "
            f"times frequency) under {MAX_TREE_STEPS // STEPS_PER_PERIOD:,}."
        )
    zero_curve = pd.Series(zero_rates, index=maturities)
    coupon_times = get_cash_flow_periods(years_to_maturity, frequency) / frequency
    accrued = float(
        face_value * coupon_rate / frequency * (1 - coupon_times[0] * frequency)
    )
    dirty_price = price + accrued
    bond = (coupon_rate, years_to_maturity, frequency, face_value)
    options = (call_price, years_to_first_call, put_price, years_to_first_put)

    def tree(curve_shift: float) -> dict:
        return trinomial_tree_model.build_short_rate_tree(
            zero_curve,
            coupon_times,
            speed,
            volatility,
            model,
            STEPS_PER_PERIOD * frequency,
            curve_shift,
        )

    base_tree = tree(0.0)
    oas = brentq(
        lambda spread: trinomial_tree_model.price_callable_bond(
            base_tree, *bond, *options, spread
        )
        - dirty_price,
        -0.2,
        1.0,
    )
    straight = (
        trinomial_tree_model.price_callable_bond(base_tree, *bond, spread=oas) - accrued
    )

    # A 100 basis point parallel shift of the continuously compounded curve at an unchanged
    # OAS: at 25 basis points the convexity of a callable bond is mostly lattice noise (it can
    # even turn positive at a 2% volatility). The lognormal models need a positive forward rate
    # at every step of the tree, so on a curve with a forward below 2% the shift is half the
    # lowest forward instead, which keeps the shifted-down curve positive.
    shift = 0.01
    if model in trinomial_tree_model.LOGNORMAL_MODELS:
        times = base_tree["Times"]
        forwards = -np.diff(np.log(discount_factors(zero_curve, times))) / np.diff(
            times
        )
        shift = min(shift, 0.5 * float(forwards.min()))
    up, down = (
        trinomial_tree_model.price_callable_bond(tree(move), *bond, *options, oas)
        for move in (shift, -shift)
    )

    return {
        "Option-Adjusted Spread": float(oas),
        "Straight Bond Price": straight,
        "Option Value": straight - price,
        "Effective Duration": (down - up) / (2 * dirty_price * shift),
        "Effective Convexity": (down + up - 2 * dirty_price) / (dirty_price * shift**2),
    }


def get_implied_default_probability(
    spread: float,
    recovery_rate: float = 0.4,
    horizons: tuple[float, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Calculate the default probabilities a credit spread implies with a constant hazard rate
    (the credit triangle): how likely a default within each horizon is when investors are
    compensated for the expected loss, so

        λ = Spread / (1 - Recovery Rate),  P(default before t) = 1 - e^(-λt)

    Spreads also pay for liquidity and the uncertainty of losses, so these risk-neutral
    probabilities are higher than historical default rates.

    For more information, see the following book:

    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      chapter 24.

    Args:
        spread (float): The credit spread, as a decimal.
        recovery_rate (float, optional): The share recovered on default. Defaults to 0.4.
        horizons (tuple[float, ...], optional): The horizons in years. Defaults to (1, 3, 5,
            10).

    Returns:
        dict[str, float]: The Hazard Rate and the default probability per horizon.
    """
    if not 0 <= recovery_rate < 1:
        raise ValueError("The recovery rate must lie between 0 and 1.")
    hazard = spread / (1 - recovery_rate)
    result = {"Hazard Rate": hazard}
    for horizon in horizons:
        result[f"{horizon:g}Y Default Probability"] = 1 - np.exp(-hazard * horizon)
    return result
