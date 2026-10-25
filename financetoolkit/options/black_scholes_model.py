"""Black Scholes Model"""

__docformat__ = "google"


import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

# The volatility range the implied volatility is solved within.
MINIMUM_IMPLIED_VOLATILITY = 1e-6
MAXIMUM_IMPLIED_VOLATILITY = 5.0

# The negative halves of the 6, 12 and 20 point Gauss-Legendre rules used by the bivariate
# normal distribution, computed once since the rules are symmetric around zero.
_GAUSS_LEGENDRE = {
    points: tuple(
        array[:points] for array in np.polynomial.legendre.leggauss(2 * points)
    )
    for points in (3, 6, 10)
}

# pylint: disable=too-many-arguments,too-many-locals


def _validate_numeric_inputs(**kwargs) -> None:
    """
    Validate that the provided keyword arguments are numeric (int, float,
    np.ndarray or pd.Series) and raise a TypeError otherwise.

    Args:
        **kwargs: the arguments to validate, keyed by argument name.

    Raises:
        TypeError: if any of the provided values is not a float, int,
        np.ndarray or pd.Series.
    """
    for name, value in kwargs.items():
        if value is None:
            continue
        if not isinstance(value, (int, float, np.ndarray, pd.Series)) or isinstance(
            value, bool
        ):
            raise TypeError(
                f"{name} must be a float, int, np.ndarray or pd.Series, "
                f"received {type(value).__name__}."
            )


@np.errstate(divide="ignore", invalid="ignore")
def get_d1(
    stock_price: float | pd.Series | np.ndarray,
    strike_price: float | pd.Series | np.ndarray,
    risk_free_rate: float | pd.Series | np.ndarray,
    volatility: float | pd.Series | np.ndarray,
    time_to_expiration: float | pd.Series | np.ndarray,
    dividend_yield: float | pd.Series | np.ndarray = 0,
):
    """
    Calculate d1 in the Black-Scholes model for option pricing.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.
        dividend_yield (float or pd.Series): The dividend yield of the stock. Defaults to 0.

    Returns:
        float | pd.Series: The d1 value.
    """

    return (
        np.log(stock_price / strike_price)
        + (risk_free_rate - dividend_yield + (volatility**2) / 2) * time_to_expiration
    ) / (volatility * np.sqrt(time_to_expiration))


@np.errstate(divide="ignore", invalid="ignore")
def get_d2(
    d1: float | pd.Series | np.ndarray,
    volatility: float | pd.Series | np.ndarray,
    time_to_expiration: float | pd.Series | np.ndarray,
):
    """
    Calculate d2 in the Black-Scholes model for option pricing.

    Args:
        d1 (float or pd.Series): The d1 value, as calculated by get_d1.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.

    Returns:
        float | pd.Series: The d2 value.
    """

    return d1 - volatility * np.sqrt(time_to_expiration)


def get_black_scholes(
    stock_price: float | pd.Series | np.ndarray,
    strike_price: float | pd.Series | np.ndarray,
    risk_free_rate: float | pd.Series | np.ndarray,
    volatility: float | pd.Series | np.ndarray,
    time_to_expiration: float | pd.Series | np.ndarray,
    dividend_yield: float | pd.Series | np.ndarray = 0,
    put_option: bool = False,
):
    """
    Calculate the Black-Scholes model for option pricing.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.
        dividend_yield (float or pd.Series): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.

    Returns:
        float | pd.Series: The Black-Scholes value.
    """
    d1 = get_d1(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    )
    d2 = get_d2(d1, volatility, time_to_expiration)

    if put_option:
        value = strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(
            -d2
        ) - stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1)
    else:
        value = stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(
            d1
        ) - strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)

    # At expiration the option is worth its intrinsic value. The formula gets there for
    # every price except the strike itself, where d1 is 0 / 0.
    at_expiration = np.asarray(time_to_expiration) <= 0

    if np.any(at_expiration):
        intrinsic_value = np.maximum(
            (
                (strike_price - stock_price)
                if put_option
                else (stock_price - strike_price)
            ),
            0,
        )
        value = np.where(at_expiration, intrinsic_value, value)

        if np.ndim(value) == 0:
            value = float(value)

    return value


def get_implied_volatility(
    market_price: float,
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
    dividend_yield: float = 0,
    put_option: bool = False,
    initial_guess: float = 0.3,
) -> float:
    """
    Numerically solve for the Black-Scholes implied volatility that reprices a
    single observed market option price: the root of Black-Scholes price(σ) — market
    price, found with Brent's method (scipy.optimize.brentq) on the volatility
    interval [1e-6, 5].

    Args:
        market_price (float): The observed market price of the option.
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        time_to_expiration (float): The time to expiration of the option, in years.
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.
        initial_guess (float): Not used: the solver brackets the root on [1e-6, 5]
            rather than starting from a guess. Defaults to 0.3.

    Returns:
        float: The implied volatility, or NaN when the observed price admits no solution.

    Notes:
        - The Black-Scholes price is strictly increasing in volatility, so the root is
          bracketed on [1e-6, 5] rather than minimized. An unbounded minimization can
          settle on a negative volatility, which then squares into a plausible looking
          variance further downstream.
        - Quotes outside the no-arbitrage bounds, stale marks and options that never
          traded are common in a real option chain. These return NaN instead of a
          fabricated number.
    """
    if time_to_expiration <= 0 or market_price <= 0:
        return np.nan

    discounted_stock = stock_price * np.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strike_price * np.exp(-risk_free_rate * time_to_expiration)

    # Outside these bounds no volatility can reproduce the observed price.
    if put_option:
        lower_bound, upper_bound = (
            max(discounted_strike - discounted_stock, 0.0),
            discounted_strike,
        )
    else:
        lower_bound, upper_bound = (
            max(discounted_stock - discounted_strike, 0.0),
            discounted_stock,
        )

    if not lower_bound < market_price < upper_bound:
        return np.nan

    def objective(volatility: float) -> float:
        theoretical_price = get_black_scholes(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            time_to_expiration=time_to_expiration,
            volatility=volatility,
            dividend_yield=dividend_yield,
            put_option=put_option,
        )

        return theoretical_price - market_price

    if (
        objective(MINIMUM_IMPLIED_VOLATILITY) > 0
        or objective(MAXIMUM_IMPLIED_VOLATILITY) < 0
    ):
        return np.nan

    return brentq(objective, MINIMUM_IMPLIED_VOLATILITY, MAXIMUM_IMPLIED_VOLATILITY)


def get_put_call_parity(
    stock_price: float | pd.Series,
    strike_price: float | pd.Series,
    risk_free_rate: float | pd.Series,
    time_to_expiration: float | pd.Series,
    dividend_yield: float | pd.Series = 0,
    call_price: float | pd.Series | None = None,
    put_price: float | pd.Series | None = None,
) -> float | pd.Series:
    """
    Check or derive the no-arbitrage Put-Call Parity relationship for European options.

    Also known as: Put-Call Parity, PCP, the no-arbitrage relationship between calls and puts.

    Put-Call Parity states that, for European options sharing the same strike price and
    time to expiration, the following relationship must hold in order to prevent arbitrage:

    - C - P = S * e^(-q * t) - K * e^(-r * t)

    Where C is the call option price, P is the put option price, S is the stock price,
    K is the strike price, r is the risk-free rate, q is the dividend yield and t is the
    time to expiration.

    This function is deliberately flexible so it can serve two related purposes:

    1. Derive the theoretical price of the missing leg. Provide either ``call_price`` or
       ``put_price`` (not both) and the function returns the parity-implied price of the
       other option.
    2. Validate an observed pair of option prices. Provide both ``call_price`` and
       ``put_price`` and the function returns the parity gap, i.e. the amount by which the
       observed prices deviate from the no-arbitrage relationship. A gap of (approximately)
       zero confirms no-arbitrage; a non-zero gap signals a potential arbitrage opportunity
       (ignoring transaction costs).

    If neither ``call_price`` nor ``put_price`` is provided, the function returns the
    forward-implied differential S * e^(-q * t) - K * e^(-r * t) itself, i.e. the value that
    C - P should equal under parity.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        time_to_expiration (float or pd.Series): The time to expiration of the option.
        dividend_yield (float or pd.Series): The dividend yield of the stock. Defaults to 0.
        call_price (float or pd.Series, optional): The observed or theoretical call option
            price. Defaults to None.
        put_price (float or pd.Series, optional): The observed or theoretical put option
            price. Defaults to None.

    Returns:
        float | pd.Series: depending on the inputs, either the parity-implied put price,
        the parity-implied call price, the parity gap (call_price and put_price both
        provided) or the forward-implied differential (neither provided).

    Raises:
        TypeError: if any of the numeric inputs is not a float, int, np.ndarray or
        pd.Series.

    Notes:
        Reference: Stoll, H. R. (1969). "The Relationship Between Put and Call Option
        Prices." Journal of Finance, 24(5), 801-824.
    """
    _validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        call_price=call_price,
        put_price=put_price,
    )

    forward_differential = stock_price * np.exp(
        -dividend_yield * time_to_expiration
    ) - strike_price * np.exp(-risk_free_rate * time_to_expiration)

    if call_price is not None and put_price is not None:
        # The parity gap. Should be (approximately) zero if no-arbitrage holds.
        return (call_price - put_price) - forward_differential

    if call_price is not None:
        # Parity-implied put price: P = C - (S * e^(-qt) - K * e^(-rt))
        return call_price - forward_differential

    if put_price is not None:
        # Parity-implied call price: C = P + (S * e^(-qt) - K * e^(-rt))
        return put_price + forward_differential

    # Neither price was provided, return the forward-implied differential itself.
    return forward_differential


def get_garman_kohlhagen(
    stock_price: float | pd.Series,
    strike_price: float | pd.Series,
    risk_free_rate: float | pd.Series,
    foreign_risk_free_rate: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
    put_option: bool = False,
) -> float | pd.Series:
    """
    Calculate the Garman-Kohlhagen model for foreign exchange (FX) option pricing.

    Also known as: the Black-Scholes model for currency options, FX option pricing model.

    The Garman-Kohlhagen model is a variant of the Black-Scholes model that is used to
    price European-style options on foreign exchange rates. Because holding foreign
    currency earns the foreign risk-free rate (analogous to a continuous dividend yield
    on a stock), the domestic risk-free rate used for discounting is complemented by a
    foreign risk-free rate that plays the same role as the dividend yield does for equity
    options.

    The formulas are as follows:

    - d1 = (ln(S / K) + (r — r_f + (σ^2) / 2) * t) / (σ * sqrt(t))
    - d2 = d1 — σ * sqrt(t)
    - Call Option Price = S * e^(—r_f * t) * N(d1) — K * e^(—r * t) * N(d2)
    - Put Option Price = K * e^(—r * t) * N(—d2) — S * e^(—r_f * t) * N(—d1)

    Where S is the spot exchange rate, K is the strike price, r is the domestic risk-free
    rate, r_f is the foreign risk-free rate, σ is the volatility, t is the time to
    expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
    cumulative normal distribution of d2.

    Args:
        stock_price (float or pd.Series): The current spot exchange rate (domestic currency
            per unit of foreign currency).
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The domestic risk-free interest rate.
        foreign_risk_free_rate (float or pd.Series): The foreign risk-free interest rate,
            which plays the same role as the dividend yield in the standard Black-Scholes
            model.
        volatility (float or pd.Series): The volatility of the exchange rate.
        time_to_expiration (float or pd.Series): The time to expiration of the option.
        put_option (bool): Whether the option is a put option or not.

    Returns:
        float | pd.Series: The Garman-Kohlhagen value.

    Raises:
        TypeError: if any of the numeric inputs is not a float, int, np.ndarray or
        pd.Series.

    Notes:
        Reference: Garman, M. B., & Kohlhagen, S. W. (1983). "Foreign Currency Option
        Values." Journal of International Money and Finance, 2(3), 231-237.
    """
    _validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        foreign_risk_free_rate=foreign_risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )

    # Black-Scholes with the foreign rate as the dividend yield, which d1 discounts.
    d1 = get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=foreign_risk_free_rate,
    )
    d2 = get_d2(d1=d1, volatility=volatility, time_to_expiration=time_to_expiration)

    if put_option:
        return strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(
            -d2
        ) - stock_price * np.exp(
            -foreign_risk_free_rate * time_to_expiration
        ) * norm.cdf(
            -d1
        )

    return stock_price * np.exp(
        -foreign_risk_free_rate * time_to_expiration
    ) * norm.cdf(d1) - strike_price * np.exp(
        -risk_free_rate * time_to_expiration
    ) * norm.cdf(
        d2
    )


def get_binary_option(
    stock_price: float | pd.Series,
    strike_price: float | pd.Series,
    risk_free_rate: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
    dividend_yield: float | pd.Series = 0,
    put_option: bool = False,
    option_type: str = "cash-or-nothing",
    cash_payout: float = 1.0,
) -> float | pd.Series:
    """
    Calculate the price of a Binary (Digital) Option using the Black-Scholes framework.

    Also known as: digital option, all-or-nothing option, cash-or-nothing option,
    asset-or-nothing option.

    A binary option pays out a fixed amount if the option expires in-the-money and
    nothing otherwise. Two variants are supported through the ``option_type`` parameter:

    - "cash-or-nothing": pays a fixed cash amount if the option expires in-the-money.

        - Call = cash_payout * e^(—r * t) * N(d2)
        - Put = cash_payout * e^(—r * t) * N(—d2)

    - "asset-or-nothing": pays the value of the underlying asset if the option expires
      in-the-money.

        - Call = S * e^(—q * t) * N(d1)
        - Put = S * e^(—q * t) * N(—d1)

    Where S is the stock price, r is the risk-free rate, q is the dividend yield, t is the
    time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
    cumulative normal distribution of d2.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.
        dividend_yield (float or pd.Series): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.
        option_type (str): Either "cash-or-nothing" or "asset-or-nothing". Defaults to
            "cash-or-nothing".
        cash_payout (float): The fixed cash amount paid out by a cash-or-nothing option
            when it expires in-the-money. Ignored for asset-or-nothing options. Defaults
            to 1.0.

    Returns:
        float | pd.Series: The Binary Option value.

    Raises:
        TypeError: if any of the numeric inputs is not a float, int, np.ndarray or
        pd.Series.
        ValueError: if option_type is not "cash-or-nothing" or "asset-or-nothing".

    Notes:
        Reference: Reiner, E., & Rubinstein, M. (1991). "Unscrambling the Binary Code."
        Risk Magazine, 4(9), 75-83.
    """
    _validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        cash_payout=cash_payout,
    )

    if option_type not in ("cash-or-nothing", "asset-or-nothing"):
        raise ValueError(
            "option_type must be either 'cash-or-nothing' or 'asset-or-nothing', "
            f"received {option_type!r}."
        )

    d1 = get_d1(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    d2 = get_d2(d1=d1, volatility=volatility, time_to_expiration=time_to_expiration)

    if option_type == "cash-or-nothing":
        if put_option:
            return (
                cash_payout
                * np.exp(-risk_free_rate * time_to_expiration)
                * norm.cdf(-d2)
            )

        return cash_payout * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)

    # asset-or-nothing
    if put_option:
        return (
            stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1)
        )

    return stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(d1)


def _bivariate_normal_cdf(x: float, y: float, correlation: float) -> float:
    """
    Calculates the cumulative bivariate standard normal distribution M(x, y; rho), the
    probability that two standard normal variables with correlation rho are below x and y.

    It uses the Gauss-Legendre quadrature of Genz (2004), as written out by West (2005),
    which is accurate to about 1e-15. A general multivariate normal routine is far slower,
    and the Bjerksund-Stensland (2002) approximation needs sixteen evaluations per price.
    The univariate normal distribution is evaluated with scipy.special.ndtr, which equals
    norm.cdf without its per-call overhead.

    See: Genz, A. (2004), "Numerical Computation of Rectangular Bivariate and Trivariate
    Normal and t Probabilities", Statistics and Computing, 14, 251-260, and West, G.
    (2005), "Better Approximations to Cumulative Normal Functions", Wilmott Magazine, 70-76.

    Args:
        x (float): The upper limit of the first variable.
        y (float): The upper limit of the second variable.
        correlation (float): The correlation between the two variables, from -1 to 1.

    Returns:
        float: The probability P(X < x, Y < y).
    """
    absolute_correlation = abs(correlation)

    # More quadrature points as the correlation, and so the curvature of the integrand, grows.
    if absolute_correlation < 0.3:  # noqa: PLR2004
        number_of_points = 3
    elif absolute_correlation < 0.75:  # noqa: PLR2004
        number_of_points = 6
    else:
        number_of_points = 10

    nodes, weights = _GAUSS_LEGENDRE[number_of_points]

    h, k = -x, -y
    hk = h * k
    probability = 0.0

    if absolute_correlation < 0.925:  # noqa: PLR2004
        if absolute_correlation > 0:
            hs = (h * h + k * k) / 2
            arcsine = np.arcsin(correlation)
            for sign in (-1, 1):
                sine = np.sin(arcsine * (sign * nodes + 1) / 2)
                probability += np.sum(
                    weights * np.exp((sine * hk - hs) / (1 - sine * sine))
                )
            probability = probability * arcsine / (4 * np.pi)

        return float(probability + ndtr(-h) * ndtr(-k))

    # A correlation close to one in absolute value is integrated in a transformed variable,
    # since the integrand above becomes singular as rho approaches +1 or -1.
    if correlation < 0:
        k, hk = -k, -hk

    if absolute_correlation < 1:
        one_minus_rho_squared = (1 - correlation) * (1 + correlation)
        a = np.sqrt(one_minus_rho_squared)
        bs = (h - k) ** 2
        c = (4 - hk) / 8
        d = (12 - hk) / 16
        exponent = -(bs / one_minus_rho_squared + hk) / 2

        if exponent > -100:  # noqa: PLR2004
            probability = (
                a
                * np.exp(exponent)
                * (
                    1
                    - c * (bs - one_minus_rho_squared) * (1 - d * bs / 5) / 3
                    + c * d * one_minus_rho_squared**2 / 5
                )
            )
        if -hk < 100:  # noqa: PLR2004
            b = np.sqrt(bs)
            probability -= (
                np.exp(-hk / 2)
                * np.sqrt(2 * np.pi)
                * ndtr(-b / a)
                * b
                * (1 - c * bs * (1 - d * bs / 5) / 3)
            )

        a = a / 2
        for sign in (-1, 1):
            xs = (a * (sign * nodes + 1)) ** 2
            rs = np.sqrt(1 - xs)
            exponent = -(bs / xs + hk) / 2
            probability += np.sum(
                np.where(
                    exponent > -100,  # noqa: PLR2004
                    a
                    * weights
                    * np.exp(exponent)
                    * (
                        np.exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs
                        - (1 + c * xs * (1 + d * xs))
                    ),
                    0.0,
                )
            )
        probability = -probability / (2 * np.pi)

    if correlation > 0:
        return float(probability + ndtr(-max(h, k)))

    probability = -probability
    if k > h:
        probability += ndtr(k) - ndtr(h)

    return float(probability)


def _power_times_probability(base: float, exponent: float, probability: float) -> float:
    """
    Helper function that returns base^exponent * probability in log space. In the
    Bjerksund-Stensland approximation the exponent grows like 1 / σ², so at low volatility
    the power overflows while the probability it multiplies underflows to zero; their
    product is finite and small, which the separate factors (inf * 0) would lose.

    Args:
        base (float): The positive base of the power.
        exponent (float): The exponent of the power.
        probability (float): The (non-negative) probability the power multiplies.

    Returns:
        float: The product base^exponent * probability.
    """
    if probability <= 0:
        return 0.0

    return float(np.exp(exponent * np.log(base) + np.log(probability)))


def _phi(
    stock_price: float,
    time_to_expiration: float,
    gamma: float,
    trigger_price: float,
    boundary_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    scale_price: float = 1.0,
) -> float:
    """
    Helper function, the "phi" function of the Bjerksund-Stensland approximation: the
    value of a claim paying S^gamma at time t that is knocked out when the stock price
    reaches the boundary I before t, and only pays when S_t is below H.

    - φ(S, t, γ, H, I) = e^λ * S^γ * [N(d) — (I / S)^κ * N(d — 2 ln(I / S) / (σ√t))]
    - d = —[ln(S / H) + (b + (γ — 0.5)σ²) * t] / (σ√t)
    - λ = (—r + γb + 0.5γ(γ — 1)σ²) * t
    - κ = 2b / σ² + (2γ — 1)

    Args:
        stock_price (float): The current stock price.
        time_to_expiration (float): The time t to evaluate the claim at, in years.
        gamma (float): The gamma exponent used within the formula.
        trigger_price (float): The price level ("H" in the reference formula) used
            within the payoff condition.
        boundary_price (float): The early-exercise boundary price ("I" in the reference
            formula).
        risk_free_rate (float): The risk-free interest rate.
        cost_of_carry (float): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (float): The volatility of the stock.
        scale_price (float): The price P the result is expressed in, returning
            φ / P^γ. The γ = β terms of the approximation are multiplied by α = (I — K) * I^(—β),
            which together with S^β overflows at low volatility; passing P = I lets the caller
            compute (I — K) * (S / I)^β directly. Defaults to 1.0 (the plain φ).

    Returns:
        float: The value of the phi function divided by scale_price^gamma.
    """
    volatility_time = volatility * np.sqrt(time_to_expiration)

    kappa = 2 * cost_of_carry / (volatility**2) + (2 * gamma - 1)

    lambda_ = (
        -risk_free_rate
        + gamma * cost_of_carry
        + 0.5 * gamma * (gamma - 1) * volatility**2
    ) * time_to_expiration

    d1 = (
        -(
            np.log(stock_price / trigger_price)
            + (cost_of_carry + (gamma - 0.5) * volatility**2) * time_to_expiration
        )
        / volatility_time
    )

    d2 = d1 - 2 * np.log(boundary_price / stock_price) / volatility_time

    # Both powers are taken in log space: S^γ and (I / S)^κ overflow at low volatility (γ and
    # κ grow like 1 / σ²) while the normal probabilities they multiply underflow.  # noqa: E501
    return float(
        np.exp(lambda_ + gamma * np.log(stock_price / scale_price))
        * (
            ndtr(d1)
            - np.exp(kappa * np.log(boundary_price / stock_price) + log_ndtr(d2))
        )
    )


def _psi(
    stock_price: float,
    time_to_expiration: float,
    gamma: float,
    trigger_price: float,
    second_boundary_price: float,
    first_boundary_price: float,
    first_time: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    scale_price: float = 1.0,
) -> float:
    """
    Helper function, the "psi" function of the Bjerksund-Stensland (2002) approximation:
    the two-period counterpart of phi, for a claim paying S^gamma at expiration T that is
    knocked out when the stock price reaches the boundary I2 = I(T) before t1 or the
    boundary I1 = I(t1) between t1 and T, and only pays when S_T is below H. The boundary
    before t1 is the higher one, since the early-exercise boundary of a call falls towards
    the strike as expiration approaches.

    - ψ = e^(λT) * S^γ * [M(—e1, —f1, ρ) — (I2 / S)^κ * M(—e2, —f2, ρ)
      — (I1 / S)^κ * M(—e3, —f3, —ρ) + (I1 / I2)^κ * M(—e4, —f4, —ρ)]

    With M the cumulative bivariate normal distribution, ρ = sqrt(t1 / T), μ = b + (γ —
    0.5)σ², λ and κ as in phi and:

    - e1 = [ln(S / I1) + μ * t1] / (σ√t1),  e2 = [ln(I2² / (S * I1)) + μ * t1] / (σ√t1)
    - e3 = [ln(S / I1) — μ * t1] / (σ√t1),  e4 = [ln(I2² / (S * I1)) — μ * t1] / (σ√t1)
    - f1 = [ln(S / H) + μ * T] / (σ√T),  f2 = [ln(I2² / (S * H)) + μ * T] / (σ√T)
    - f3 = [ln(I1² / (S * H)) + μ * T] / (σ√T),  f4 = [ln(S * I1² / (H * I2²)) + μ * T] / (σ√T)

    Args:
        stock_price (float): The current stock price.
        time_to_expiration (float): The time to expiration T, in years.
        gamma (float): The gamma exponent used within the formula.
        trigger_price (float): The price level ("H" in the reference formula) used
            within the payoff condition.
        second_boundary_price (float): The flat boundary I2 = I(T) that applies before t1.
        first_boundary_price (float): The flat boundary I1 = I(t1) that applies between
            t1 and T.
        first_time (float): The time t1 at which the boundary steps from I2 to I1.
        risk_free_rate (float): The risk-free interest rate.
        cost_of_carry (float): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (float): The volatility of the stock.
        scale_price (float): The price P the result is expressed in, returning
            ψ / P^γ, see ``_phi``. Defaults to 1.0 (the plain ψ).

    Returns:
        float: The value of the psi function divided by scale_price^gamma.
    """
    drift = cost_of_carry + (gamma - 0.5) * volatility**2
    volatility_first_time = volatility * np.sqrt(first_time)
    volatility_time = volatility * np.sqrt(time_to_expiration)

    i1, i2 = first_boundary_price, second_boundary_price

    e1 = (np.log(stock_price / i1) + drift * first_time) / volatility_first_time
    e2 = (
        np.log(i2**2 / (stock_price * i1)) + drift * first_time
    ) / volatility_first_time
    e3 = (np.log(stock_price / i1) - drift * first_time) / volatility_first_time
    e4 = (
        np.log(i2**2 / (stock_price * i1)) - drift * first_time
    ) / volatility_first_time

    f1 = (
        np.log(stock_price / trigger_price) + drift * time_to_expiration
    ) / volatility_time
    f2 = (
        np.log(i2**2 / (stock_price * trigger_price)) + drift * time_to_expiration
    ) / volatility_time
    f3 = (
        np.log(i1**2 / (stock_price * trigger_price)) + drift * time_to_expiration
    ) / volatility_time
    f4 = (
        np.log(stock_price * i1**2 / (trigger_price * i2**2))
        + drift * time_to_expiration
    ) / volatility_time

    correlation = np.sqrt(first_time / time_to_expiration)
    lambda_ = (
        -risk_free_rate
        + gamma * cost_of_carry
        + 0.5 * gamma * (gamma - 1) * volatility**2
    )
    kappa = 2 * cost_of_carry / volatility**2 + (2 * gamma - 1)

    # The powers are combined with their probabilities in log space, see ``_phi``.
    return float(
        np.exp(lambda_ * time_to_expiration + gamma * np.log(stock_price / scale_price))
        * (
            _bivariate_normal_cdf(-e1, -f1, correlation)
            - _power_times_probability(
                i2 / stock_price, kappa, _bivariate_normal_cdf(-e2, -f2, correlation)
            )
            - _power_times_probability(
                i1 / stock_price, kappa, _bivariate_normal_cdf(-e3, -f3, -correlation)
            )
            + _power_times_probability(
                i1 / i2, kappa, _bivariate_normal_cdf(-e4, -f4, -correlation)
            )
        )
    )


def _american_call_binomial_value(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
    steps: int = 1000,
) -> float:
    """
    Helper function that prices an American call with a recombining binomial tree, the
    fallback of the Bjerksund-Stensland approximation where its boundary is undefined.

    The tree centres its moves on the cost of carry, u = e^(b * Δt + σ√Δt) and
    d = e^(b * Δt — σ√Δt), with p = (e^(b * Δt) — d) / (u — d). Unlike the Cox-Ross-Rubinstein
    moves u = e^(σ√Δt), this keeps p between 0 and 1 at any volatility and cost of carry.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        cost_of_carry (float): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option, in years.
        steps (int): The number of time steps of the tree. Defaults to 1000.

    Returns:
        float: The value of the American call option.
    """
    time_delta = time_to_expiration / steps
    up = np.exp(cost_of_carry * time_delta + volatility * np.sqrt(time_delta))
    down = np.exp(cost_of_carry * time_delta - volatility * np.sqrt(time_delta))
    probability = (np.exp(cost_of_carry * time_delta) - down) / (up - down)
    discount = np.exp(-risk_free_rate * time_delta)

    # The prices at the last step, from all down moves to all up moves.
    prices = stock_price * down**steps * (up / down) ** np.arange(steps + 1)
    values = np.maximum(prices - strike_price, 0.0)

    for _ in range(steps):
        prices = prices[:-1] / down
        values = discount * (probability * values[1:] + (1 - probability) * values[:-1])
        values = np.maximum(values, prices - strike_price)

    return float(values[0])


def _bjerksund_stensland_call(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
) -> float:
    """
    Prices an American call option using the Bjerksund-Stensland (2002) closed-form
    approximation with a two-step flat early-exercise boundary. Puts are priced through
    the put-call transformation applied by get_bjerksund_stensland. The value is floored
    at the European price and the exercise value, and a binomial tree takes over where the
    boundary falls to or below the strike.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        cost_of_carry (float): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option.

    Returns:
        float: The value of the American call option.
    """
    if time_to_expiration <= 0:
        # At expiration the option is worth its exercise value; the formulas divide by σ√t.
        return max(stock_price - strike_price, 0.0)

    exercise_value = max(stock_price - strike_price, 0.0)

    # The generalized Black-Scholes price of the European call with the same cost of carry,
    # through the dividend yield q = r — b. The American value is never below it.
    european_value = float(
        get_black_scholes(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=risk_free_rate - cost_of_carry,
        )
    )

    if cost_of_carry >= risk_free_rate:
        # Early exercise is not optimal when the cost of carry is at least the risk-free rate,
        # except with a negative risk-free rate, where paying the strike later costs more
        # than paying it now and deep in-the-money calls are exercised; the exercise value
        # floors the price there.  # noqa: E501
        return max(european_value, exercise_value)

    # The boundary steps at the golden-ratio point of the option's life, the split that
    # Bjerksund and Stensland (2002) found to give the most accurate prices.
    first_time = 0.5 * (np.sqrt(5) - 1) * time_to_expiration

    beta = (0.5 - cost_of_carry / volatility**2) + np.sqrt(
        (cost_of_carry / volatility**2 - 0.5) ** 2 + 2 * risk_free_rate / volatility**2
    )
    b_infinity = beta / (beta - 1) * strike_price
    b_zero = max(
        strike_price, risk_free_rate / (risk_free_rate - cost_of_carry) * strike_price
    )

    def boundary(time: float) -> float:
        h_time = (
            -(cost_of_carry * time + 2 * volatility * np.sqrt(time))
            * strike_price**2
            / ((b_infinity - b_zero) * b_zero)
        )
        return b_zero + (b_infinity - b_zero) * (1 - np.exp(h_time))

    first_boundary = boundary(first_time)
    second_boundary = boundary(time_to_expiration)

    if not np.isfinite(second_boundary) or second_boundary <= strike_price:
        # The boundary function is built for I > K. With a negative cost of carry and low
        # volatility (b * T + 2σ√T < 0) or a non-positive risk-free rate it falls to or
        # below the strike, where the approximation would exercise options that are out of
        # the money; a binomial tree prices the option instead.  # noqa: E501
        return max(
            _american_call_binomial_value(
                stock_price=stock_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                cost_of_carry=cost_of_carry,
                volatility=volatility,
                time_to_expiration=time_to_expiration,
            ),
            european_value,
            exercise_value,
        )

    if stock_price >= second_boundary:
        # Immediate exercise is optimal.
        return max(exercise_value, european_value)

    def phi(
        gamma: float,
        trigger_price: float,
        boundary_price: float,
        scale_price: float = 1.0,
    ) -> float:
        return _phi(
            stock_price=stock_price,
            time_to_expiration=first_time,
            gamma=gamma,
            trigger_price=trigger_price,
            boundary_price=boundary_price,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
            scale_price=scale_price,
        )

    def psi(gamma: float, trigger_price: float, scale_price: float = 1.0) -> float:
        return _psi(
            stock_price=stock_price,
            time_to_expiration=time_to_expiration,
            gamma=gamma,
            trigger_price=trigger_price,
            second_boundary_price=second_boundary,
            first_boundary_price=first_boundary,
            first_time=first_time,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
            scale_price=scale_price,
        )

    # α * S^β and α * φ(β) are written as (I — K) * (S / I)^β and (I — K) * φ(β) / I^β: at low
    # volatility β is large and S^β and I^(—β) overflow and underflow on their own.  # noqa: E501
    first_premium = first_boundary - strike_price
    second_premium = second_boundary - strike_price

    value = (
        second_premium * (stock_price / second_boundary) ** beta
        - second_premium
        * phi(beta, second_boundary, second_boundary, scale_price=second_boundary)
        + phi(1, second_boundary, second_boundary)
        - phi(1, first_boundary, second_boundary)
        - strike_price * phi(0, second_boundary, second_boundary)
        + strike_price * phi(0, first_boundary, second_boundary)
        + first_premium
        * phi(beta, first_boundary, second_boundary, scale_price=first_boundary)
        - first_premium * psi(beta, first_boundary, scale_price=first_boundary)
        + psi(1, first_boundary)
        - psi(1, strike_price)
        - strike_price * psi(0, first_boundary)
        + strike_price * psi(0, strike_price)
    )

    if not np.isfinite(value):
        # Extreme inputs can still leave the approximation undefined; the European price is
        # then the closest lower bound of the American value.  # noqa: E501
        value = european_value

    # The flat-boundary approximation is a lower bound that can fall below the European
    # price (high volatility, long maturities) or the exercise value; as in QuantLib the
    # result is floored at both.
    return float(max(value, european_value, exercise_value))


@np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
def get_bjerksund_stensland(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the price of an American option using the Bjerksund-Stensland (2002)
    closed-form analytical approximation.

    Also known as: BS2002, Bjerksund-Stensland approximation, American option approximation.

    Unlike European options, American options can be exercised at any time up to and
    including expiration, which means their fair value generally needs to be found
    numerically (e.g. through a binomial tree, see ``binomial_trees_model.py``). The
    Bjerksund-Stensland model instead derives a closed-form approximation by assuming
    the early-exercise boundary is flat in steps: once the stock price rises above (for
    calls) or falls below (for puts) a computed trigger price, immediate exercise is
    assumed optimal. The 2002 version splits the option's life at
    t1 = ½(√5 — 1) * T into two periods, each with its own flat boundary, which brings
    the price closer to the exact American value than the single flat boundary of the
    1993 version.

    For a call option with cost of carry b = r - q smaller than the risk-free rate r,
    the boundaries are:

    - β = (0.5 — b / σ²) + sqrt((b / σ² — 0.5)² + 2r / σ²)
    - B∞ = β / (β — 1) * K
    - B0 = max(K, r / (r — b) * K)
    - h(t) = —(b * t + 2σ√t) * K² / ((B∞ — B0) * B0)
    - I(t) = B0 + (B∞ — B0) * (1 — e^h(t)),  with I1 = I(t1) and I2 = I(T)
    - α1 = (I1 — K) * I1^(—β),  α2 = (I2 — K) * I2^(—β)

    If S ≥ I2 the option is exercised immediately and is worth S — K. Otherwise:

    - C = α2 * S^β — α2 * φ(S, t1, β, I2, I2) + φ(S, t1, 1, I2, I2) — φ(S, t1, 1, I1, I2)
      — K * φ(S, t1, 0, I2, I2) + K * φ(S, t1, 0, I1, I2) + α1 * φ(S, t1, β, I1, I2)
      — α1 * ψ(S, T, β, I1, I2, I1, t1) + ψ(S, T, 1, I1, I2, I1, t1) — ψ(S, T, 1, K, I2, I1, t1)
      — K * ψ(S, T, 0, I1, I2, I1, t1) + K * ψ(S, T, 0, K, I2, I1, t1)

    Where φ is a knock-out claim over one period (a univariate normal expression) and ψ
    its two-period counterpart (a bivariate normal expression), see ``_phi`` and ``_psi``.

    When b ≥ r it is never optimal to exercise an American call early, so the value
    collapses to the standard European Black-Scholes price. American puts are priced
    through the put-call transformation:

    - AmericanPut(S, K, T, r, b, σ) = AmericanCall(K, S, T, r — b, —b, σ)

    An American option is worth at least its European counterpart and its exercise value,
    while the flat-boundary approximation can fall below the European price at high
    volatility and long maturities. The result is therefore floored at both, as QuantLib
    does. Where the boundary I2 falls to or below the strike (a negative cost of carry at low
    volatility, or a non-positive risk-free rate), the approximation would exercise options
    that are out of the money, and a 1,000-step binomial tree prices the option instead.

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option, in years.
            At zero the option is worth its exercise value.
        dividend_yield (float): The continuous dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float: The Bjerksund-Stensland (2002) approximation of the American option value.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.

    Notes:
        Reference: Bjerksund, P., & Stensland, G. (2002). "Closed Form Valuation of
        American Options." Working paper, Norwegian School of Economics and Business
        Administration (NHH). As presented in Haug, E. G. (2007). "The Complete Guide to
        Option Pricing Formulas" (2nd ed.), section 3.1.4. Compared with a finite-difference
        American price, the approximation is slightly low; for S = K = 100, r = 5%,
        q = 3%, σ = 25% and T = 1 the put is 8.8439 against 8.8823.
    """
    for name, value in {
        "stock_price": stock_price,
        "strike_price": strike_price,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "time_to_expiration": time_to_expiration,
        "dividend_yield": dividend_yield,
    }.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a float or int, received {type(value).__name__}."
            )

    cost_of_carry = risk_free_rate - dividend_yield

    if put_option:
        # American put via P(S, K, T, r, b, v) = C(K, S, T, r - b, -b, v).
        return _bjerksund_stensland_call(
            stock_price=strike_price,
            strike_price=stock_price,
            risk_free_rate=risk_free_rate - cost_of_carry,
            cost_of_carry=-cost_of_carry,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
        )

    return _bjerksund_stensland_call(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        cost_of_carry=cost_of_carry,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )
