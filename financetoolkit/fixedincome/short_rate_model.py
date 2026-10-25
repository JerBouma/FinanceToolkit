"""Short-Rate Models and Callable Bond Valuation Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# pylint: disable=too-many-arguments,too-many-locals

MODELS = ("vasicek", "cir")
# The tree steps per coupon period. With one step per period the call is valued too coarsely
# (the effective convexity of a callable bond can even turn positive); twenty steps match a
# continuous-time valuation to a fraction of a basis point.
STEPS_PER_PERIOD = 20


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
        model (str, optional): "vasicek" or "cir". Defaults to "vasicek".
        periods_per_year (int, optional): The observations per year. Defaults to 252.

    Returns:
        dict[str, float]: The Speed of Mean Reversion, Long-Run Mean, Volatility, Current
        Rate and Half-Life (in years) of the model.
    """
    if model not in MODELS:
        raise ValueError(f"The model must be one of {', '.join(MODELS)}.")

    values = rates.dropna().to_numpy(dtype=float)
    if len(values) < 30:  # noqa: PLR2004
        raise ValueError("At least 30 observations are needed to calibrate the model.")
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


def get_model_yields(
    maturities: np.ndarray,
    model: str,
    speed: float,
    mean: float,
    volatility: float,
    rate: float,
) -> np.ndarray:
    """
    The zero-coupon yields the Vasicek or CIR model implies, from the closed-form bond price
    P(T) = A(T) e^(-B(T) r):

        - Vasicek: B = (1 - e^(-aT)) / a,
          ln A = (B - T)(a²b - σ²/2) / a² - σ² B² / (4a)
        - CIR: γ = √(a² + 2σ²), B = 2(e^(γT) - 1) / ((γ + a)(e^(γT) - 1) + 2γ),
          A = [2γ e^((a + γ)T/2) / ((γ + a)(e^(γT) - 1) + 2γ)]^(2ab/σ²)

    Args:
        maturities (np.ndarray): The maturities in years.
        model (str): "vasicek" or "cir".
        speed (float): The speed of mean reversion a.
        mean (float): The long-run mean b.
        volatility (float): The volatility σ.
        rate (float): The current short rate r.

    Returns:
        np.ndarray: The continuously compounded yields.
    """
    t = np.asarray(maturities, dtype=float)
    a, b, sigma = speed, mean, volatility

    if model == "vasicek":
        big_b = (1 - np.exp(-a * t)) / a
        log_a = (big_b - t) * (
            a**2 * b - sigma**2 / 2
        ) / a**2 - sigma**2 * big_b**2 / (4 * a)
    elif model == "cir":
        gamma = np.sqrt(a**2 + 2 * sigma**2)
        growth = np.exp(gamma * t) - 1
        denominator = (gamma + a) * growth + 2 * gamma
        big_b = 2 * growth / denominator
        log_a = (2 * a * b / sigma**2) * np.log(
            2 * gamma * np.exp((a + gamma) * t / 2) / denominator
        )
    else:
        raise ValueError(f"The model must be one of {', '.join(MODELS)}.")

    return -(log_a - big_b * rate) / t


def simulate_short_rate(
    model: str,
    speed: float,
    mean: float | np.ndarray,
    volatility: float,
    rate: float,
    years: float = 10.0,
    steps_per_year: int = 12,
    paths: int = 10000,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Simulate short-rate paths with an Euler scheme. For Hull-White (1990) the long-run mean
    varies over time, θ(t)/a, so that the model reproduces today's curve; pass it as an
    array with one value per step.

    Args:
        model (str): "vasicek", "cir" or "hull_white".
        speed (float): The speed of mean reversion.
        mean (float | np.ndarray): The long-run mean, or one per step for Hull-White.
        volatility (float): The volatility.
        rate (float): The starting rate.
        years (float, optional): The horizon. Defaults to 10.
        steps_per_year (int, optional): The steps per year. Defaults to 12.
        paths (int, optional): The number of paths. Defaults to 10,000.
        seed (int | None, optional): The random seed. Defaults to 0.

    Returns:
        np.ndarray: The rates, paths by steps (including the start).
    """
    rng = np.random.default_rng(seed)
    steps = int(round(years * steps_per_year))
    dt = 1 / steps_per_year
    means = np.broadcast_to(np.asarray(mean, dtype=float), (steps,))
    rates = np.empty((paths, steps + 1))
    rates[:, 0] = rate
    shocks = rng.standard_normal((paths, steps))

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
) -> np.ndarray:
    """
    The long-run mean per step that makes the Hull-White model fit today's curve. In the
    model dr = [θ(t) - a r] dt + σ dW the expected short rate is

        E[r(t)] = f(t) + σ² (1 - e^(-at))² / (2a²)

    with f the forward rate of the (continuously compounded) zero curve. Rather than
    differentiating the forward curve for θ(t), which spikes where an interpolated curve has
    a kink, the mean of every Euler step is set so that the expected rate reaches E[r(t)] at
    the end of the step exactly, whatever the step size. The forward rate of a step is the
    one over that step of the zero curve interpolated linearly.

    Args:
        maturities (np.ndarray): The maturities of the zero curve in years.
        zero_rates (np.ndarray): The continuously compounded zero rates.
        speed (float): The speed of mean reversion a.
        volatility (float): The volatility σ.
        rate (float): The short rate today.
        years (float): The horizon.
        steps_per_year (int): The steps per year.

    Returns:
        np.ndarray: The long-run mean per step.
    """
    steps = int(round(years * steps_per_year))
    dt = 1 / steps_per_year
    times = np.arange(steps + 2) * dt
    log_discount = np.interp(times, maturities, zero_rates) * times
    forward = np.diff(log_discount) / dt
    convexity = volatility**2 * (1 - np.exp(-speed * times[:-1])) ** 2 / (2 * speed**2)
    # The expected rate of every step: today's rate, then the forward rate of the step plus
    # the convexity term.
    expected = np.concatenate([[rate], (forward + convexity)[1:]])
    return expected[:-1] + (expected[1:] - expected[:-1]) / (speed * dt)


def _ho_lee_tree(
    discount_factors: np.ndarray, volatility: float, dt: float
) -> list[np.ndarray]:
    """
    A Ho-Lee binomial tree of the short rate calibrated to the discount factors: at step i
    the rates are θ_i + σ √dt (2j - i), j = 0..i, with θ_i solved so that the tree prices the
    zero-coupon bond maturing one step later exactly (forward induction on Arrow-Debreu
    prices with up and down probabilities of one half), which has the closed form
    θ_i = [ln Σ_j Q_j e^(-σ √dt (2j - i) dt) - ln P(i + 1)] / dt.
    """
    steps = len(discount_factors)
    rates = []
    arrow = np.array([1.0])
    for i in range(steps):
        spread = volatility * np.sqrt(dt) * (2 * np.arange(i + 1) - i)
        theta = (
            np.log(np.sum(arrow * np.exp(-spread * dt))) - np.log(discount_factors[i])
        ) / dt
        step_rates = theta + spread
        rates.append(step_rates)
        discounted = arrow * np.exp(-step_rates * dt) / 2
        arrow = np.zeros(i + 2)
        arrow[:-1] += discounted
        arrow[1:] += discounted
    return rates


def price_callable_bond(
    tree: list[np.ndarray],
    coupon_rate: float,
    frequency: int,
    face_value: float = 100.0,
    call_price: float | None = None,
    first_call_step: int = 0,
    spread: float = 0.0,
    steps_per_period: int = 1,
    first_coupon_step: int | None = None,
) -> float:
    """
    The price of a bond on a short-rate tree, discounting at the tree rate plus a spread:
    the value of all its remaining payments, so including the interest accrued since the
    last coupon (the dirty price). A callable bond is redeemed by the issuer whenever its
    value exceeds the call price on a coupon date from the first call date on, so its value
    at those nodes is capped at the call price.

    Args:
        tree (list[np.ndarray]): The short rates per step; the last coupon is paid at the
            end of the last step.
        coupon_rate (float): The annual coupon rate.
        frequency (int): The coupons per year.
        face_value (float, optional): The face value. Defaults to 100.
        call_price (float | None, optional): The call price; None for a straight bond.
            Defaults to None.
        first_call_step (int, optional): The first step on which the bond can be called (on
            the coupon dates from then on). Defaults to 0.
        spread (float, optional): The spread added to every rate (the OAS). Defaults to 0.
        steps_per_period (int, optional): The tree steps per coupon period. Defaults to 1.
        first_coupon_step (int | None, optional): The step of the next coupon. Defaults to
            None, which is one coupon period from now.

    Returns:
        float: The price.
    """
    dt = 1 / (frequency * steps_per_period)
    coupon = face_value * coupon_rate / frequency
    first = steps_per_period if first_coupon_step is None else first_coupon_step
    value = np.full(len(tree) + 1, face_value + coupon)
    for step in range(len(tree) - 1, -1, -1):
        value = (value[:-1] + value[1:]) / 2 * np.exp(-(tree[step] + spread) * dt)
        if step > 0 and step >= first and (step - first) % steps_per_period == 0:
            if call_price is not None and step >= first_call_step:
                value = np.minimum(value, call_price)
            value = value + coupon
    return float(value[0])


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
) -> dict[str, float]:
    """
    The option-adjusted spread (OAS) of a callable bond: the spread over a short-rate tree
    calibrated to today's zero curve at which the model price, with the issuer's call
    exercised whenever optimal, equals the market price. Unlike the Z-spread it removes the
    value of the embedded option, so it measures the credit and liquidity premium alone.
    The effective duration and convexity follow from shifting the curve by 100 basis points
    up and down at an unchanged OAS.

    The tree has twenty steps per coupon period. A bond bought between coupon dates also pays
    the interest accrued since the last coupon, so the market (clean) price plus the accrued
    interest is what the tree values; the next coupon date is placed on the nearest step
    (within a twentieth of a period).

    For more information, see the following sources:

    - Ho, T.S.Y. and Lee, S.-B. (1986). "Term Structure Movements and Pricing Interest Rate
      Contingent Claims." Journal of Finance, 41(5), 1011-1029.
    - Fabozzi, F.J. (2016). "Bond Markets, Analysis, and Strategies." Pearson, 9th edition,
      chapter 18.

    Args:
        price (float): The market (clean) price of the bond.
        coupon_rate (float): The annual coupon rate.
        years_to_maturity (float): The years to maturity.
        maturities (np.ndarray): The maturities of the zero curve in years.
        zero_rates (np.ndarray): The zero rates (annually compounded).
        volatility (float, optional): The volatility of the short rate (Ho-Lee, in absolute
            terms per year, e.g. 0.01 for 1%). Defaults to 0.01.
        call_price (float | None, optional): The call price; None for a straight bond.
            Defaults to 100.
        years_to_first_call (float, optional): The years until the bond can first be called.
            Defaults to 0.
        frequency (int, optional): The coupons per year. Defaults to 2.
        face_value (float, optional): The face value. Defaults to 100.

    Returns:
        dict[str, float]: The Option-Adjusted Spread, the Straight Bond Price (clean) and
        Option Value at that spread, and the Effective Duration and Effective Convexity.
    """
    periods = years_to_maturity * frequency
    coupons = int(np.ceil(periods - 1e-9))
    if coupons < 1:
        raise ValueError("The bond must have at least one coupon left.")
    first_coupon_step = max(1, round((periods - (coupons - 1)) * STEPS_PER_PERIOD))
    steps = first_coupon_step + (coupons - 1) * STEPS_PER_PERIOD
    dt = 1 / (frequency * STEPS_PER_PERIOD)
    accrued = (
        face_value
        * coupon_rate
        / frequency
        * (1 - first_coupon_step / STEPS_PER_PERIOD)
    )
    dirty_price = price + accrued
    times = np.arange(1, steps + 1) * dt
    continuous = np.log1p(np.interp(times, maturities, zero_rates))
    first_call_step = int(np.ceil(years_to_first_call / dt - 1e-9))

    base_tree = _ho_lee_tree(np.exp(-continuous * times), volatility, dt)

    def callable_price(tree: list[np.ndarray], spread: float) -> float:
        return price_callable_bond(
            tree,
            coupon_rate,
            frequency,
            face_value,
            call_price,
            first_call_step,
            spread,
            STEPS_PER_PERIOD,
            first_coupon_step,
        )

    oas = brentq(
        lambda spread: callable_price(base_tree, spread) - dirty_price, -0.2, 1.0
    )
    straight = (
        price_callable_bond(
            base_tree,
            coupon_rate,
            frequency,
            face_value,
            None,
            0,
            oas,
            STEPS_PER_PERIOD,
            first_coupon_step,
        )
        - accrued
    )

    # A 100 basis point shift: at 25 basis points the convexity of a callable bond is mostly
    # lattice noise (it can even turn positive at a 2% volatility). A parallel shift of the
    # continuously compounded curve moves every Ho-Lee node by the same amount, which is the
    # same as moving the spread, so the base tree priced at OAS ± shift gives the shifted prices
    # exactly without building two more trees.
    shift = 0.01
    up = callable_price(base_tree, oas + shift)
    down = callable_price(base_tree, oas - shift)

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
    The default probabilities a credit spread implies with a constant hazard rate (the
    credit triangle): investors are compensated for the expected loss, so

        λ = Spread / (1 - Recovery Rate),  P(default before t) = 1 - e^(-λt)

    Spreads also pay for liquidity and the uncertainty of losses, so these risk-neutral
    probabilities are higher than historical default rates.

    For more information, see the following book:

    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      chapter 24.

    Args:
        spread (float): The credit spread, as a decimal.
        recovery_rate (float, optional): The share recovered on default. Defaults to 0.4.
        horizons (tuple[float, ...], optional): The horizons in years. Defaults to 1, 3, 5
            and 10.

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
