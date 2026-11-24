"""G2++ Two-Factor Gaussian Short-Rate Model

The G2++ model of Brigo and Mercurio (2006, section 4.2), equivalent to the two-factor Hull-White
model, writes the short rate as the sum of two correlated Gaussian factors and a deterministic
shift that fits today's zero curve:

- r(t) = x(t) + y(t) + φ(t), x(0) = y(0) = 0
- dx(t) = -a x(t) dt + σ dW_1(t), dy(t) = -b y(t) dt + η dW_2(t), dW_1 dW_2 = ρ dt

Two factors let the rates of different maturities move less than perfectly together, which a
one-factor model (Vasicek, Hull-White) cannot: there every rate is a function of the short rate.
With η = 0 the model is Hull-White with speed a and volatility σ.

Every function takes the zero-coupon curve as the Toolkit's curves are returned: annually
compounded zero rates by maturity in years, interpolated linearly in between (see
`financetoolkit.fixedincome.helpers.discount_factors`). The model's rates and parameters are
continuously compounded. Times are year fractions from today, without a day count.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from financetoolkit.fixedincome.bonds.bond_model import get_cash_flow_periods
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.utilities.error_model import check_option

# The swaption integral runs over the first factor at expiry within this many standard
# deviations of its mean: the density beyond 12 standard deviations is below 1e-32.
INTEGRATION_WIDTH = 12.0


def _check_parameters(
    speed_x: float,
    speed_y: float,
    volatility_x: float,
    volatility_y: float,
    correlation: float,
) -> None:
    """
    Check that the G2++ parameters describe a model: positive speeds, volatilities that are
    not negative and a correlation between -1 and 1.
    """
    if speed_x <= 0 or speed_y <= 0:
        raise ValueError("The speeds of mean reversion a and b must be positive.")
    if volatility_x < 0 or volatility_y < 0:
        raise ValueError("The volatilities σ and η cannot be negative.")
    if not -1 <= correlation <= 1:
        raise ValueError("The correlation ρ must lie between -1 and 1.")


def _decay(speed: float, time: float | np.ndarray) -> float | np.ndarray:
    """B(z, t, T) = (1 - e^(-z (T - t))) / z for a time T - t, written with expm1."""
    return -np.expm1(-speed * np.asarray(time, dtype=float)) / speed


def _variance(
    time: float | np.ndarray,
    speed_x: float,
    speed_y: float,
    volatility_x: float,
    volatility_y: float,
    correlation: float,
) -> float | np.ndarray:
    """
    The variance V(t, T) of ∫_t^T (x(u) + y(u)) du given the factors at t, a function of
    τ = T - t only (Brigo and Mercurio, 2006, equation 4.10).
    """
    a, b, sigma, eta = speed_x, speed_y, volatility_x, volatility_y
    tau = np.asarray(time, dtype=float)

    def single(speed: float, volatility: float):
        return (
            volatility**2
            / speed**2
            * (tau - 2 * _decay(speed, tau) + _decay(2 * speed, tau))
        )

    cross = (
        2
        * correlation
        * sigma
        * eta
        / (a * b)
        * (tau - _decay(a, tau) - _decay(b, tau) + _decay(a + b, tau))
    )
    return single(a, sigma) + single(b, eta) + cross


def get_g2_bond_price(
    maturities: float | np.ndarray,
    zero_curve: pd.Series,
    speed_x: float,
    volatility_x: float,
    speed_y: float,
    volatility_y: float,
    correlation: float,
    time: float = 0.0,
    factor_x: float | np.ndarray = 0.0,
    factor_y: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """
    Calculate the price at time t of zero-coupon bonds paying 1 at the maturities T in the
    G2++ model, given the factors x(t) and y(t): what the bonds are worth in the two-factor
    model of the short rate.

    The deterministic shift φ is chosen so that the model reproduces the zero curve, so at t
    = 0 the price is the curve's discount factor P(0, T), and at a later t the price follows
    from the factors alone, without φ itself.

    The formula is as follows:

    - P(t, T) = P(0, T) / P(0, t) · exp(½ [V(t, T) - V(0, T) + V(0, t)] - B(a, t, T) x(t)
      - B(b, t, T) y(t))
    - B(z, t, T) = (1 - e^(-z (T - t))) / z
    - V(t, T) = σ²/a² [T - t + 2/a e^(-a(T - t)) - 1/(2a) e^(-2a(T - t)) - 3/(2a)]
      + η²/b² [T - t + 2/b e^(-b(T - t)) - 1/(2b) e^(-2b(T - t)) - 3/(2b)]
      + 2ρση/(ab) [T - t + (e^(-a(T - t)) - 1)/a + (e^(-b(T - t)) - 1)/b
      - (e^(-(a + b)(T - t)) - 1)/(a + b)]

    where a and b are the speeds of mean reversion of the factors, σ and η their
    volatilities and ρ their correlation. V(t, T) is the variance of the integral of
    x + y from t to T. With η = 0 the price is the Hull-White price of the G1++ form.

    The model's rates are continuously compounded; the zero curve is annually compounded
    and read with P(0, t) = (1 + r(t))^(-t).

    For more information, see the following sources:

    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 4.2, equation 4.14.
    - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure
      Models II: Two-Factor Models." The Journal of Derivatives, 2(2), 37-48.

    Referenced against:
    - QuantLib, G2.discountBond: https://www.quantlib.org/reference/class_quant_lib_1_1_g2.html

    Args:
        maturities (float | np.ndarray): The maturities T of the bonds in years from today,
            not before the time t.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        speed_x (float): The speed of mean reversion a of the first factor.
        volatility_x (float): The volatility σ of the first factor.
        speed_y (float): The speed of mean reversion b of the second factor.
        volatility_y (float): The volatility η of the second factor.
        correlation (float): The correlation ρ of the two factors.
        time (float, optional): The time t in years from today at which the bonds are
            priced. Defaults to 0.
        factor_x (float | np.ndarray, optional): The first factor x(t) at that time.
            Defaults to 0.
        factor_y (float | np.ndarray, optional): The second factor y(t) at that time.
            Defaults to 0.

    Returns:
        float | np.ndarray: The bond prices per 1 of face value. Arrays of maturities and
        factors broadcast against each other.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import g2_model

    zero_curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])

    g2_model.get_g2_bond_price(
        maturities=10, zero_curve=zero_curve, speed_x=0.5, volatility_x=0.01,
        speed_y=0.05, volatility_y=0.008, correlation=-0.7, time=2, factor_x=0.005,
        factor_y=-0.002
    )
    ```

    Which returns:

    ```
    0.7193542851904846
    ```
    """
    _check_parameters(speed_x, speed_y, volatility_x, volatility_y, correlation)
    maturities = np.asarray(maturities, dtype=float)
    if time < 0 or np.any(maturities < time):
        raise ValueError(
            "The bonds must mature at or after the time t, itself not before today."
        )
    parameters = (speed_x, speed_y, volatility_x, volatility_y, correlation)

    ratio = discount_factors(zero_curve, maturities) / discount_factors(
        zero_curve, time
    )
    exponent = 0.5 * (
        _variance(maturities - time, *parameters)
        - _variance(maturities, *parameters)
        + _variance(time, *parameters)
    )
    price = ratio * np.exp(
        exponent
        - _decay(speed_x, maturities - time) * np.asarray(factor_x, dtype=float)
        - _decay(speed_y, maturities - time) * np.asarray(factor_y, dtype=float)
    )
    return float(price) if np.ndim(price) == 0 else price


def get_g2_zero_coupon_bond_option(
    option_expiry: float | np.ndarray,
    bond_maturity: float | np.ndarray,
    strike: float | np.ndarray,
    zero_curve: pd.Series,
    speed_x: float,
    volatility_x: float,
    speed_y: float,
    volatility_y: float,
    correlation: float,
    option_type: str = "call",
) -> float | np.ndarray:
    """
    Calculate the price of a European option on a zero-coupon bond in the G2++ model, in
    closed form: what a call or put expiring at T on a bond paying 1 at S > T, struck at K,
    is worth.

    The bond price at T is lognormal, as in Hull-White, with a variance that adds the two
    factors and their covariance, so the price is Black's formula on the forward bond price.
    A caplet is (1 + Kτ) puts on the bond from its reset to its payment date struck at 1 /
    (1 + Kτ).

    The formula is as follows:

    - ZBC = P(0, S) N(h) - K P(0, T) N(h - Σ), h = ln(P(0, S) / (K P(0, T))) / Σ + Σ / 2
    - ZBP = K P(0, T) N(-h + Σ) - P(0, S) N(-h)
    - Σ² = σ²/(2a³) (1 - e^(-a(S - T)))² (1 - e^(-2aT))
      + η²/(2b³) (1 - e^(-b(S - T)))² (1 - e^(-2bT))
      + 2ρση/(ab(a + b)) (1 - e^(-a(S - T))) (1 - e^(-b(S - T))) (1 - e^(-(a + b)T))

    where N is the standard normal distribution, P(0, t) the discount factors of the zero
    curve, a and b the speeds of mean reversion, σ and η the volatilities and ρ the
    correlation of the factors. With η = 0 this is the Hull-White bond option.

    The model's rates are continuously compounded; the zero curve is annually compounded.

    For more information, see the following sources:

    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 4.2.3, equations 4.30 and 4.31 (the caplet).

    Referenced against:
    - QuantLib, G2.discountBondOption: https://www.quantlib.org/reference/class_quant_lib_1_1_g2.html

    Args:
        option_expiry (float | np.ndarray): The expiry T of the option in years.
        bond_maturity (float | np.ndarray): The maturity S of the bond in years, after the
            expiry.
        strike (float | np.ndarray): The strike K, as a price per 1 of face value.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        speed_x (float): The speed of mean reversion a of the first factor.
        volatility_x (float): The volatility σ of the first factor.
        speed_y (float): The speed of mean reversion b of the second factor.
        volatility_y (float): The volatility η of the second factor.
        correlation (float): The correlation ρ of the two factors.
        option_type (str, optional): The option. Options are: "call" (the right to buy the
            bond), "put" (the right to sell it). Defaults to "call".

    Returns:
        float | np.ndarray: The option price per 1 of face value.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import g2_model

    zero_curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])

    g2_model.get_g2_zero_coupon_bond_option(
        option_expiry=2, bond_maturity=5, strike=0.9, zero_curve=zero_curve, speed_x=0.5,
        volatility_x=0.01, speed_y=0.05, volatility_y=0.008, correlation=-0.7,
        option_type="put"
    )
    ```

    Which returns:

    ```
    0.010032203062912748
    ```
    """
    _check_parameters(speed_x, speed_y, volatility_x, volatility_y, correlation)
    check_option("option_type", option_type, ("call", "put"))
    expiry = np.asarray(option_expiry, dtype=float)
    maturity = np.asarray(bond_maturity, dtype=float)
    strike = np.asarray(strike, dtype=float)
    if np.any(expiry <= 0) or np.any(maturity <= expiry):
        raise ValueError(
            "The option must expire after today and before the bond matures."
        )
    a, b, sigma, eta = speed_x, speed_y, volatility_x, volatility_y

    # Σ² written with B(z, T, S) = (1 - e^(-z(S - T))) / z and the variance of each factor
    # at T, so it stays accurate for small speeds.
    b_x, b_y = _decay(a, maturity - expiry), _decay(b, maturity - expiry)
    deviation = np.sqrt(
        sigma**2 * b_x**2 * _decay(2 * a, expiry)
        + eta**2 * b_y**2 * _decay(2 * b, expiry)
        + 2 * correlation * sigma * eta * b_x * b_y * _decay(a + b, expiry)
    )
    p_expiry = discount_factors(zero_curve, expiry)
    p_maturity = discount_factors(zero_curve, maturity)
    h = np.log(p_maturity / (strike * p_expiry)) / deviation + deviation / 2
    sign = 1.0 if option_type == "call" else -1.0
    value = sign * (
        p_maturity * norm.cdf(sign * h)
        - strike * p_expiry * norm.cdf(sign * (h - deviation))
    )
    return float(value) if np.ndim(value) == 0 else value


def get_g2_swaption(
    option_expiry: float,
    tenor: float,
    fixed_rate: float,
    zero_curve: pd.Series,
    speed_x: float,
    volatility_x: float,
    speed_y: float,
    volatility_y: float,
    correlation: float,
    swaption_type: str = "payer",
    fixed_frequency: int = 1,
    notional: float = 1.0,
) -> float:
    """
    Calculate the price of a European swaption in the G2++ model by a one-dimensional
    integral: what the right to enter a swap at the fixed rate is worth in the two-factor
    model of the short rate.

    At expiry T a payer swaption is worth max(1 - CB, 0), with CB the coupon bond paying the
    fixed rate on the fixed leg's dates and 1 at the end. Given the first factor x(T) the
    coupon bond falls as the second factor rises, so there is one value ȳ(x) at which it is
    worth 1, and the expectation over y given x is a sum of normal distribution values. What
    is left is an integral over the normal density of x(T) under the T-forward measure.

    The formula is as follows:

    - Payer Swaption = Notional · P(0, T) ∫ φ(x; μ_x, σ_x) [N(-h_1(x)) - SUM(λ_i(x)
      e^(κ_i(x)) N(-h_2,i(x)))] dx
    - Receiver Swaption = Notional · P(0, T) ∫ φ(x; μ_x, σ_x) [SUM(λ_i(x) e^(κ_i(x))
      N(h_2,i(x))) - N(h_1(x))] dx
    - h_1(x) = (ȳ - μ_y) / (σ_y √(1 - ρ_xy²)) - ρ_xy (x - μ_x) / (σ_x √(1 - ρ_xy²))
    - h_2,i(x) = h_1(x) + B(b, T, t_i) σ_y √(1 - ρ_xy²)
    - λ_i(x) = c_i A(T, t_i) e^(-B(a, T, t_i) x)
    - κ_i(x) = -B(b, T, t_i) [μ_y - ½ (1 - ρ_xy²) σ_y² B(b, T, t_i) + ρ_xy σ_y (x - μ_x) / σ_x]
    - ȳ = ȳ(x) solves SUM(c_i A(T, t_i) e^(-B(a, T, t_i) x - B(b, T, t_i) ȳ)) = 1
    - A(T, t_i) = P(0, t_i) / P(0, T) · exp(½ [V(T, t_i) - V(0, t_i) + V(0, T)])
    - μ_x = -(σ²/a² + ρση/(ab)) (1 - e^(-aT)) + σ²/(2a²) (1 - e^(-2aT))
      + ρση/(b(a + b)) (1 - e^(-(a + b)T)), and μ_y alike with (a, σ) and (b, η) swapped
    - σ_x = σ √((1 - e^(-2aT)) / (2a)), σ_y = η √((1 - e^(-2bT)) / (2b)),
      ρ_xy = ρση / ((a + b) σ_x σ_y) (1 - e^(-(a + b)T))

    with c_i = K τ_i the fixed coupons (plus 1 on the last date t_n), τ_i the accrual of the
    i-th fixed period (1 / fixed_frequency, no day count, a short first period when the
    tenor is not a whole number of periods), B and V those of `get_g2_bond_price`, μ_x, μ_y,
    σ_x, σ_y and ρ_xy the moments of the factors at T under the T-forward measure and
    φ(x; μ, σ) the normal density. The floating leg is worth par at the start of the swap
    (one curve both projects and discounts). The integral is evaluated with adaptive
    quadrature over μ_x ± 12 σ_x.

    The fixed rate is a nominal rate paid fixed_frequency times a year; the model's rates
    are continuously compounded and the zero curve annually compounded.

    For more information, see the following sources:

    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 4.2.5, theorem 4.2.3 (equation 4.31).

    Referenced against:
    - QuantLib, G2SwaptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_g2_swaption_engine.html

    Args:
        option_expiry (float): The expiry T of the swaption, the start of the swap, in years.
        tenor (float): The length of the swap in years.
        fixed_rate (float): The fixed rate K of the swap (decimal).
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        speed_x (float): The speed of mean reversion a of the first factor.
        volatility_x (float): The volatility σ of the first factor, positive.
        speed_y (float): The speed of mean reversion b of the second factor.
        volatility_y (float): The volatility η of the second factor, positive. For η = 0
            the model is Hull-White: use `short_rate_model.get_swaption` with the
            Hull-White model.
        correlation (float): The correlation ρ of the two factors, strictly between -1
            and 1.
        swaption_type (str, optional): The swaption. Options are: "payer" (the right to pay
            the fixed rate), "receiver" (the right to receive it). Defaults to "payer".
        fixed_frequency (int, optional): The fixed payments per year. Defaults to 1.
        notional (float, optional): The notional. Defaults to 1.

    Returns:
        float: The price, in the currency of the notional.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import g2_model

    zero_curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])

    g2_model.get_g2_swaption(
        option_expiry=2, tenor=5, fixed_rate=0.04, zero_curve=zero_curve, speed_x=0.5,
        volatility_x=0.01, speed_y=0.05, volatility_y=0.008, correlation=-0.7,
        notional=1_000_000
    )
    ```

    Which returns:

    ```
    11567.77942017329
    ```
    """
    _check_parameters(speed_x, speed_y, volatility_x, volatility_y, correlation)
    if volatility_x <= 0 or volatility_y <= 0:
        raise ValueError(
            "The swaption integral needs positive volatilities σ and η; for η = 0 the "
            "model is Hull-White (short_rate_model.get_swaption)."
        )
    if not -1 < correlation < 1:
        raise ValueError(
            "The swaption integral needs a correlation strictly within -1 and 1."
        )
    check_option("swaption_type", swaption_type, ("payer", "receiver"))
    if option_expiry <= 0 or tenor <= 0:
        raise ValueError("The expiry and the tenor of the swaption must be positive.")

    a, b, sigma, eta, rho = speed_x, speed_y, volatility_x, volatility_y, correlation
    parameters = (a, b, sigma, eta, rho)
    expiry = float(option_expiry)
    times = expiry + get_cash_flow_periods(tenor, fixed_frequency) / fixed_frequency
    coupons = fixed_rate * np.diff(times, prepend=expiry)
    coupons[-1] += 1

    p_expiry = float(discount_factors(zero_curve, expiry))
    log_a = np.log(discount_factors(zero_curve, times) / p_expiry) + 0.5 * (
        _variance(times - expiry, *parameters)
        - _variance(times, *parameters)
        + _variance(expiry, *parameters)
    )
    b_x, b_y = _decay(a, times - expiry), _decay(b, times - expiry)

    # The moments of x(T) and y(T) under the T-forward measure (Brigo and Mercurio, 2006,
    # equation 4.16).
    cross = rho * sigma * eta
    mean_x = (
        -(sigma**2 / a**2 + cross / (a * b)) * -np.expm1(-a * expiry)
        + sigma**2 / (2 * a**2) * -np.expm1(-2 * a * expiry)
        + cross / (b * (a + b)) * -np.expm1(-(a + b) * expiry)
    )
    mean_y = (
        -(eta**2 / b**2 + cross / (a * b)) * -np.expm1(-b * expiry)
        + eta**2 / (2 * b**2) * -np.expm1(-2 * b * expiry)
        + cross / (a * (a + b)) * -np.expm1(-(a + b) * expiry)
    )
    deviation_x = sigma * np.sqrt(_decay(2 * a, expiry))
    deviation_y = eta * np.sqrt(_decay(2 * b, expiry))
    rho_xy = cross * _decay(a + b, expiry) / (deviation_x * deviation_y)
    residual = np.sqrt(1 - rho_xy**2)
    omega = 1.0 if swaption_type == "payer" else -1.0

    def critical_y(x: float) -> float:
        # The coupon bond less 1 is a decreasing, convex function of y, so Newton's method
        # converges from any start: after at most one step it approaches the root from
        # below, monotonically.
        log_weights = np.log(coupons) + log_a - b_x * x
        y = 0.0
        for _ in range(200):
            terms = np.exp(log_weights - b_y * y)
            step = (terms.sum() - 1) / (b_y @ terms)
            y += step
            if abs(step) <= 1e-15 * max(1.0, abs(y)):
                break
        return float(y)

    def integrand(x: float) -> float:
        standardised = (x - mean_x) / deviation_x
        h_1 = (critical_y(x) - mean_y) / (
            deviation_y * residual
        ) - rho_xy * standardised / residual
        h_2 = h_1 + b_y * deviation_y * residual
        log_lambda = np.log(coupons) + log_a - b_x * x
        kappa = -b_y * (
            mean_y
            - 0.5 * residual**2 * deviation_y**2 * b_y
            + rho_xy * deviation_y * standardised
        )
        value = ndtr(-omega * h_1) - np.sum(
            np.exp(log_lambda + kappa) * ndtr(-omega * h_2)
        )
        return float(
            np.exp(-0.5 * standardised**2) / (deviation_x * np.sqrt(2 * np.pi)) * value
        )

    integral, _ = quad(
        integrand,
        mean_x - INTEGRATION_WIDTH * deviation_x,
        mean_x + INTEGRATION_WIDTH * deviation_x,
        points=[mean_x],
        limit=200,
        epsabs=1e-15,
        epsrel=1e-12,
    )
    return float(notional * omega * p_expiry * integral)
