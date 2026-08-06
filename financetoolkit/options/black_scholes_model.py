"""Black Scholes Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

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


def get_d1(
    stock_price: float | pd.Series,
    strike_price: float | pd.Series,
    risk_free_rate: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
    dividend_yield: float | pd.Series = 0,
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
    np.seterr(divide="ignore", invalid="ignore")

    return (
        np.log(stock_price / strike_price)
        + (risk_free_rate - dividend_yield + (volatility**2) / 2) * time_to_expiration
    ) / (volatility * np.sqrt(time_to_expiration))


def get_d2(
    d1: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
):
    """
    Calculate d2 in the Black-Scholes model for option pricing.

    Args:
        stock_price (float or pd.Series): The current stock price.
        strike_price (float or pd.Series): The option's strike price.
        risk_free_rate (float or pd.Series): The risk-free interest rate.
        volatility (float or pd.Series): The volatility of the stock.
        time_to_expiration (float or pd.Series): The time to expiration of the option.

    Returns:
        float | pd.Series: The d2 value.
    """
    np.seterr(divide="ignore", invalid="ignore")

    return d1 - volatility * np.sqrt(time_to_expiration)


def get_black_scholes(
    stock_price: float | pd.Series,
    strike_price: float | pd.Series,
    risk_free_rate: float | pd.Series,
    volatility: float | pd.Series,
    time_to_expiration: float | pd.Series,
    dividend_yield: float | pd.Series = 0,
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
        return strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(
            -d2
        ) - stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(-d1)

    return stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(
        d1
    ) - strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)


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
    single observed market option price, by minimizing the squared difference
    between the Black-Scholes theoretical price and the market price.

    Args:
        market_price (float): The observed market price of the option.
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        time_to_expiration (float): The time to expiration of the option, in years.
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.
        initial_guess (float): The starting volatility guess for the numerical
            solver. Defaults to 0.3.

    Returns:
        float: The implied volatility.
    """

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

        return (theoretical_price - market_price) ** 2

    return minimize(objective, x0=initial_guess).x[0]


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


def _phi(
    stock_price: float,
    time_to_expiration: float,
    gamma: float,
    trigger_price: float,
    boundary_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
) -> float:
    """
    Helper function, the "phi" function used within the Bjerksund-Stensland (1993)
    closed-form approximation for American option pricing.

    Args:
        stock_price (float): The current stock price.
        time_to_expiration (float): The time to expiration of the option.
        gamma (float): The gamma exponent used within the formula.
        trigger_price (float): The price level ("H" in the reference formula) used
            within the boundary condition.
        boundary_price (float): The early-exercise boundary price ("I" in the reference
            formula).
        risk_free_rate (float): The risk-free interest rate.
        cost_of_carry (float): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (float): The volatility of the stock.

    Returns:
        float: The value of the phi function.
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

    return (
        np.exp(lambda_)
        * stock_price**gamma
        * (norm.cdf(d1) - (boundary_price / stock_price) ** kappa * norm.cdf(d2))
    )


def _bjerksund_stensland_call(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    cost_of_carry: float,
    volatility: float,
    time_to_expiration: float,
) -> float:
    """
    Prices an American call option using the Bjerksund-Stensland (1993) closed-form
    approximation. Puts are priced through the put-call transformation applied by
    get_bjerksund_stensland.

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
    if cost_of_carry >= risk_free_rate:
        # Never optimal to exercise early when cost of carry >= the risk-free rate.
        dividend_yield = risk_free_rate - cost_of_carry
        d1 = get_d1(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=dividend_yield,
        )
        d2 = get_d2(d1=d1, volatility=volatility, time_to_expiration=time_to_expiration)
        return stock_price * np.exp(-dividend_yield * time_to_expiration) * norm.cdf(
            d1
        ) - strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)

    beta = (0.5 - cost_of_carry / volatility**2) + np.sqrt(
        (cost_of_carry / volatility**2 - 0.5) ** 2 + 2 * risk_free_rate / volatility**2
    )
    b_infinity = beta / (beta - 1) * strike_price
    b_zero = max(
        strike_price, risk_free_rate / (risk_free_rate - cost_of_carry) * strike_price
    )

    h_time = -(
        cost_of_carry * time_to_expiration
        + 2 * volatility * np.sqrt(time_to_expiration)
    ) * (b_zero / (b_infinity - b_zero))
    trigger_price = b_zero + (b_infinity - b_zero) * (1 - np.exp(h_time))

    if stock_price >= trigger_price:
        # Immediate exercise is optimal.
        return stock_price - strike_price

    alpha = (trigger_price - strike_price) * trigger_price ** (-beta)

    return (
        alpha * stock_price**beta
        - alpha
        * _phi(
            stock_price=stock_price,
            time_to_expiration=time_to_expiration,
            gamma=beta,
            trigger_price=trigger_price,
            boundary_price=trigger_price,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
        )
        + _phi(
            stock_price=stock_price,
            time_to_expiration=time_to_expiration,
            gamma=1,
            trigger_price=trigger_price,
            boundary_price=trigger_price,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
        )
        - _phi(
            stock_price=stock_price,
            time_to_expiration=time_to_expiration,
            gamma=1,
            trigger_price=strike_price,
            boundary_price=trigger_price,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
        )
        - strike_price
        * _phi(
            stock_price=stock_price,
            time_to_expiration=time_to_expiration,
            gamma=0,
            trigger_price=trigger_price,
            boundary_price=trigger_price,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
        )
        + strike_price
        * _phi(
            stock_price=stock_price,
            time_to_expiration=time_to_expiration,
            gamma=0,
            trigger_price=strike_price,
            boundary_price=trigger_price,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
        )
    )


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
    Calculate the price of an American option using the Bjerksund-Stensland (1993)
    closed-form analytical approximation.

    Also known as: BS93, Bjerksund-Stensland approximation, American option approximation.

    Unlike European options, American options can be exercised at any time up to and
    including expiration, which means their fair value generally needs to be found
    numerically (e.g. through a binomial tree, see ``binomial_trees_model.py``). The
    Bjerksund-Stensland model instead derives a closed-form approximation by assuming
    the early-exercise boundary is a flat barrier: once the stock price rises above (for
    calls) or falls below (for puts) a computed trigger price, immediate exercise is
    assumed optimal.

    For a call option with cost of carry b = r - q smaller than the risk-free rate r,
    the approximation is:

    - β = (0.5 — b / σ²) + sqrt((b / σ² — 0.5)² + 2r / σ²)
    - B∞ = β / (β — 1) * K
    - B0 = max(K, r / (r — b) * K)
    - h(T) = —(b * T + 2σ√T) * (B0 / (B∞ — B0))
    - I = B0 + (B∞ — B0) * (1 — e^h(T))     (the early-exercise trigger price)

    If S ≥ I the option is exercised immediately and is worth S — K. Otherwise the value
    is a combination of power/binary "phi" terms (see reference below) that reproduces
    the value of a European option plus the value of the early-exercise premium.

    When b ≥ r it is never optimal to exercise an American call early, so the value
    collapses to the standard European Black-Scholes price. American puts are priced
    through the well-known put-call transformation:

    - AmericanPut(S, K, T, r, b, σ) = AmericanCall(K, S, T, r — b, —b, σ)

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option.
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.

    Returns:
        float: The Bjerksund-Stensland (1993) approximation of the American option value.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.

    Notes:
        Reference: Bjerksund, P., & Stensland, G. (1993). "Closed-Form Approximation of
        American Options." Scandinavian Journal of Management, 9, S87-S99. As presented
        in Haug, E. G. (2007). "The Complete Guide to Option Pricing Formulas" (2nd ed.).
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

    np.seterr(divide="ignore", invalid="ignore")

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
