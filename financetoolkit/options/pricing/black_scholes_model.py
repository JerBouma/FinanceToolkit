"""Black Scholes Model"""

__docformat__ = "google"


import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from financetoolkit.options.helpers import (
    validate_numeric_inputs,
    validate_positive_inputs,
)
from financetoolkit.utilities.error_model import check_option

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
    Calculate d1 of the Black-Scholes model: the standardized distance between the log of the
    stock price and the strike price, which sets the probabilities of the formula.

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
    # A zero strike gives ln(S / 0) = +inf and so d1 = d2 = +inf: the call is worth the
    # discounted forward and the put nothing, rather than a division by zero.
    with np.errstate(divide="ignore"):
        log_forward_moneyness = (
            np.log(np.divide(stock_price, strike_price))
            + (risk_free_rate - dividend_yield) * time_to_expiration
        )
    spread = volatility * np.sqrt(time_to_expiration)
    d1 = (log_forward_moneyness + spread**2 / 2) / spread

    # Without any uncertainty left (no volatility or no time) the price ends up exactly at
    # the forward, so d1 is the limit +inf or -inf on either side of the strike and 0 at it,
    # which prices every option at its discounted intrinsic value instead of 0 / 0 = NaN.
    riskless = np.asarray(spread) == 0
    if np.any(riskless):
        moneyness = np.asarray(log_forward_moneyness, dtype=float)
        limit = np.where(moneyness > 0, np.inf, np.where(moneyness < 0, -np.inf, 0.0))
        values = np.where(riskless, limit, np.asarray(d1, dtype=float))
        if isinstance(d1, pd.Series):
            return pd.Series(values, index=d1.index, name=d1.name)
        if isinstance(d1, pd.DataFrame):
            return pd.DataFrame(values, index=d1.index, columns=d1.columns)
        return float(values) if np.ndim(values) == 0 else values

    return d1


@np.errstate(divide="ignore", invalid="ignore")
def get_d2(
    d1: float | pd.Series | np.ndarray,
    volatility: float | pd.Series | np.ndarray,
    time_to_expiration: float | pd.Series | np.ndarray,
):
    """
    Calculate d2 of the Black-Scholes model: d1 less the volatility over the life of the option,
    whose normal distribution N(d2) is the risk-neutral probability that a call expires in the
    money.

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
    Calculate the Black-Scholes value of a European option: what a call or put is worth today
    when the stock follows a lognormal process with a constant volatility.

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

    Raises:
        ValueError: If the stock price, strike price, volatility or time to expiration is
            negative.
    """
    # A negative volatility or price gives a number (a negative option value) rather than an
    # error, so it is refused here for every formula built on this one.
    validate_positive_inputs(
        allow_zero=True,
        stock_price=stock_price,
        strike_price=strike_price,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )
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
        value = strike_price * np.exp(-risk_free_rate * time_to_expiration) * ndtr(
            -d2
        ) - stock_price * np.exp(-dividend_yield * time_to_expiration) * ndtr(-d1)
    else:
        value = stock_price * np.exp(-dividend_yield * time_to_expiration) * ndtr(
            d1
        ) - strike_price * np.exp(-risk_free_rate * time_to_expiration) * ndtr(d2)

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
) -> float:
    """
    Calculate the Black-Scholes implied volatility: the volatility at which the
    Black-Scholes price equals an observed market price of an option.

    It is the root of Black-Scholes price(σ) — market price, found with Brent's method
    (scipy.optimize.brentq) on the volatility interval [1e-6, 5].

    Args:
        market_price (float): The observed market price of the option.
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        time_to_expiration (float): The time to expiration of the option, in years.
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.

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
    Calculate the Put-Call Parity of European options: the price of a call or put that rules
    out arbitrage given the other, or the gap by which observed prices deviate from it.

    Also known as: Put-Call Parity, PCP, the no-arbitrage relationship between calls and puts.

    Put-Call Parity states that, for European options sharing the same strike price and
    time to expiration, the following relationship must hold in order to prevent arbitrage:

    - C - P = S * e^(-q * t) - K * e^(-r * t)

    Where C is the call option price, P is the put option price, S is the stock price,
    K is the strike price, r is the risk-free rate, q is the dividend yield and t is the
    time to expiration.

    This function is deliberately flexible so it can serve two related purposes:

    1. Derive the theoretical price of the missing leg. Provide either the call price or
       the put price (not both) and the function returns the parity-implied price of the
       other option.
    2. Validate an observed pair of option prices. Provide both the call price and the put
       price and the function returns the parity gap, i.e. the amount by which the
       observed prices deviate from the no-arbitrage relationship. A gap of (approximately)
       zero confirms no-arbitrage; a non-zero gap signals a potential arbitrage opportunity
       (ignoring transaction costs).

    If neither the call price nor the put price is provided, the function returns the
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
        the parity-implied call price, the parity gap (the call price and the put price
        both provided) or the forward-implied differential (neither provided).

    Raises:
        TypeError: if any of the numeric inputs is not a float, int, np.ndarray or
        pd.Series.

    Notes:
        Reference: Stoll, H. R. (1969). "The Relationship Between Put and Call Option
        Prices." Journal of Finance, 24(5), 801-824.
    """
    validate_numeric_inputs(
        arrays=True,
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


def get_put_call_parity_gap(
    strike_price: float | np.ndarray,
    time_to_expiration: float | np.ndarray,
    stock_price: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0,
) -> float | np.ndarray:
    """
    Calculate the put-call parity gap of the Black-Scholes call and put prices: by how much
    the call minus the put differs from the stock price minus the discounted strike price.

    Args:
        strike_price (float | np.ndarray): The strike price(s).
        time_to_expiration (float | np.ndarray): The time(s) to expiration in years.
        stock_price (float): The current stock price.
        risk_free_rate (float): The continuously compounded risk free rate.
        volatility (float): The volatility of the stock.
        dividend_yield (float, optional): The continuously compounded dividend yield.
            Defaults to 0.

    Returns:
        float | np.ndarray: The parity gap per strike price and time to expiration.
    """
    # The call and put of every strike price and expiration on arrays at once.
    call_price, put_price = (
        get_black_scholes(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=dividend_yield,
            put_option=put,
        )
        for put in (False, True)
    )
    return get_put_call_parity(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        call_price=call_price,
        put_price=put_price,
    )


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
    Calculate the Garman-Kohlhagen value of a European currency option: what a call or put
    on an exchange rate is worth, with the foreign interest rate in the role of a dividend
    yield.

    Also known as: the Black-Scholes model for currency options, FX option pricing model.

    The Garman-Kohlhagen model is a variant of the Black-Scholes model that is used to
    price European-style options on foreign exchange rates. Because holding foreign
    currency earns the foreign risk-free rate (analogous to a continuous dividend yield
    on a stock), the domestic risk-free rate used for discounting is complemented by a
    foreign risk-free rate that plays the same role as the dividend yield does for equity
    options. The value is therefore get_black_scholes with the foreign risk-free rate
    as the dividend yield, and at expiration the exercise value.

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
        ValueError: if the exchange rate, strike price, volatility or time to expiration is
        negative.

    Notes:
        Reference: Garman, M. B., & Kohlhagen, S. W. (1983). "Foreign Currency Option
        Values." Journal of International Money and Finance, 2(3), 231-237.
    """
    validate_numeric_inputs(
        arrays=True,
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        foreign_risk_free_rate=foreign_risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )
    validate_positive_inputs(
        allow_zero=True,
        stock_price=stock_price,
        strike_price=strike_price,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )

    # Holding the foreign currency earns the foreign rate, as a stock earns its dividend
    # yield, so the value is the Black-Scholes value with the foreign rate as that yield.
    return get_black_scholes(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=foreign_risk_free_rate,
        put_option=put_option,
    )


def get_black_76(
    forward_price: float | pd.Series | np.ndarray,
    strike_price: float | pd.Series | np.ndarray,
    risk_free_rate: float | pd.Series | np.ndarray,
    volatility: float | pd.Series | np.ndarray,
    time_to_expiration: float | pd.Series | np.ndarray,
    put_option: bool = False,
) -> float | pd.Series | np.ndarray:
    """
    Calculate the Black (1976) value of a European option on a futures or forward contract:
    what a call or put on the futures price is worth, discounted at the risk-free rate.

    Also known as: Black-76, Black's model, the Black model for options on futures.

    A futures or forward price F already contains the cost of carry of the underlying
    (interest, storage, dividends or convenience yield), so it grows at no rate under the
    risk-neutral measure and the option only discounts its payoff at the risk-free rate.
    That is the Black-Scholes model with S * e^(—qT) replaced by F * e^(—rT), which is how
    it is calculated here (a dividend yield equal to the risk-free rate).

    The formulas are as follows:

    - d1 = (ln(F / K) + (σ^2 / 2) * t) / (σ * sqrt(t))
    - d2 = d1 — σ * sqrt(t)
    - Call Option Price = e^(—r * t) * (F * N(d1) — K * N(d2))
    - Put Option Price = e^(—r * t) * (K * N(—d2) — F * N(—d1))

    Where F is the futures or forward price for the option's expiration, K is the strike
    price, r is the continuously compounded risk-free rate, σ is the volatility of the
    futures price, t is the time to expiration in years and N is the standard normal
    distribution. At expiration the option is worth max(F — K, 0) or max(K — F, 0).

    Referenced against:
    - QuantLib, blackFormula: https://www.quantlib.org/reference/namespace_quant_lib.html#adfe71f9b0b157cae6e4bd53faaa676af

    Shares similar logic:
    - FinancePy, Black: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/black.py#L18

    Args:
        forward_price (float | pd.Series | np.ndarray): The futures or forward price.
        strike_price (float | pd.Series | np.ndarray): The option's strike price.
        risk_free_rate (float | pd.Series | np.ndarray): The continuously compounded
            risk-free interest rate.
        volatility (float | pd.Series | np.ndarray): The volatility of the futures price.
        time_to_expiration (float | pd.Series | np.ndarray): The time to expiration of
            the option, in years.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float | pd.Series | np.ndarray: The Black-76 value.

    Raises:
        TypeError: if any of the numeric inputs is not a float, int, np.ndarray or
        pd.Series.
        ValueError: if the forward price, strike price, volatility or time to expiration is
        negative.

    Notes:
        Reference: Black, F. (1976). "The Pricing of Commodity Contracts." Journal of
        Financial Economics, 3(1-2), 167-179.
        - The value agrees with QuantLib's blackFormula within 1e-12. FinancePy's Black
          model approximates the normal distribution and so differs by up to about 1e-4.

    As an example:

    ```python
    from financetoolkit.options.pricing import black_scholes_model

    round(
        black_scholes_model.get_black_76(
            forward_price=70, strike_price=75, risk_free_rate=0.04, volatility=0.3,
            time_to_expiration=0.5, put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    8.7667
    ```
    """
    validate_numeric_inputs(
        arrays=True,
        forward_price=forward_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )
    validate_positive_inputs(
        allow_zero=True,
        forward_price=forward_price,
        strike_price=strike_price,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
    )

    return get_black_scholes(
        stock_price=forward_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=risk_free_rate,
        put_option=put_option,
    )


def get_black_76_implied_volatility(
    market_price: float,
    forward_price: float,
    strike_price: float,
    risk_free_rate: float,
    time_to_expiration: float,
    put_option: bool = False,
) -> float:
    """
    Calculate the Black-76 implied volatility of an option on a futures or forward contract:
    the volatility at which the Black-76 price equals an observed market price.

    It is the root of Black-76 price(σ) — market price, found with Brent's method on the
    volatility interval [1e-6, 5].

    Also known as: implied volatility of options on futures, Black implied volatility.

    The Black-76 price is the Black-Scholes price with a dividend yield equal to the
    risk-free rate (see get_black_76), so this is the Black-Scholes implied volatility of
    get_implied_volatility with that yield and the same no-arbitrage bounds:
    e^(—rt) * max(F — K, 0) < call < e^(—rt) * F and e^(—rt) * max(K — F, 0) < put < e^(—rt) * K.

    Shares similar logic:
    - QuantLib, blackFormulaImpliedStdDev: https://www.quantlib.org/reference/namespace_quant_lib.html#ae5515dd7810c45e5a7c61df4a00a8166
    - FinancePy, Black.implied_volatility: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/black.py#L426

    Args:
        market_price (float): The observed market price of the option.
        forward_price (float): The futures or forward price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        time_to_expiration (float): The time to expiration of the option, in years.
        put_option (bool): Whether the option is a put option or not. Defaults to False.

    Returns:
        float: The implied volatility, or NaN when the observed price admits no solution
            (outside the bounds above, or at or after expiration).

    Notes:
        QuantLib's blackFormulaImpliedStdDev gives the standard deviation, the implied
        volatility times √t.

    As an example:

    ```python
    from financetoolkit.options.pricing import black_scholes_model

    round(
        black_scholes_model.get_black_76_implied_volatility(
            market_price=8.7667, forward_price=70, strike_price=75, risk_free_rate=0.04,
            time_to_expiration=0.5, put_option=True,
        ),
        4,
    )
    ```

    Which returns:

    ```
    0.3
    ```
    """
    return get_implied_volatility(
        market_price=market_price,
        stock_price=forward_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        time_to_expiration=time_to_expiration,
        dividend_yield=risk_free_rate,
        put_option=put_option,
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
    Calculate the price of a Binary (Digital) Option: what a payoff of all or nothing at
    expiration is worth, a fixed cash amount or the stock itself when the option ends in the
    money.

    Also known as: digital option, all-or-nothing option, cash-or-nothing option,
    asset-or-nothing option.

    A binary option pays out a fixed amount if the option expires in-the-money and
    nothing otherwise. Two variants are supported:

    - Cash-or-nothing: pays a fixed cash amount if the option expires in-the-money.

        - Call = cash_payout * e^(—r * t) * N(d2)
        - Put = cash_payout * e^(—r * t) * N(—d2)

    - Asset-or-nothing: pays the value of the underlying asset if the option expires
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
        option_type (str): The payoff. Options are: "cash-or-nothing" (a fixed cash amount),
            "asset-or-nothing" (the value of the stock). Defaults to "cash-or-nothing".
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
    validate_numeric_inputs(
        arrays=True,
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        cash_payout=cash_payout,
    )

    check_option("option_type", option_type, ("cash-or-nothing", "asset-or-nothing"))

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
                cash_payout * np.exp(-risk_free_rate * time_to_expiration) * ndtr(-d2)
            )

        return cash_payout * np.exp(-risk_free_rate * time_to_expiration) * ndtr(d2)

    # asset-or-nothing
    if put_option:
        return stock_price * np.exp(-dividend_yield * time_to_expiration) * ndtr(-d1)

    return stock_price * np.exp(-dividend_yield * time_to_expiration) * ndtr(d1)


@np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
def _bivariate_normal_cdf(x, y, correlation):
    """
    Calculates the cumulative bivariate standard normal distribution M(x, y; rho), the
    probability that two standard normal variables with correlation rho are below x and y.

    It uses the Gauss-Legendre quadrature of Genz (2004), as written out by West (2005),
    which is accurate to about 1e-15. A general multivariate normal routine is far slower,
    and the Bjerksund-Stensland (2002) approximation needs sixteen evaluations per price.
    The inputs may be arrays: every element takes the branch and number of points its own
    correlation calls for, so a grid of strikes and expirations is evaluated in one call
    with the same values as one call per element.

    See: Genz, A. (2004), "Numerical Computation of Rectangular Bivariate and Trivariate
    Normal and t Probabilities", Statistics and Computing, 14, 251-260, and West, G.
    (2005), "Better Approximations to Cumulative Normal Functions", Wilmott Magazine, 70-76.

    Args:
        x (float | np.ndarray): The upper limit of the first variable.
        y (float | np.ndarray): The upper limit of the second variable.
        correlation (float | np.ndarray): The correlation between the two variables, from
            -1 to 1.

    Returns:
        float | np.ndarray: The probability P(X < x, Y < y).
    """
    x, y, correlation = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(correlation, dtype=float),
    )
    h, k, rho = -x.ravel(), -y.ravel(), correlation.ravel()
    absolute_correlation = np.abs(rho)
    probability = np.empty(h.shape)

    # More quadrature points as the correlation, and so the curvature of the integrand, grows.
    number_of_points = np.where(
        absolute_correlation < 0.3,  # noqa: PLR2004
        3,
        np.where(absolute_correlation < 0.75, 6, 10),  # noqa: PLR2004
    )
    # A correlation close to one in absolute value is integrated in a transformed variable,
    # since the integrand of the other branch becomes singular as rho approaches +1 or -1.
    near_one = absolute_correlation >= 0.925  # noqa: PLR2004

    for points in (3, 6, 10):
        nodes, weights = _GAUSS_LEGENDRE[points]

        if (low := (number_of_points == points) & ~near_one).any():
            h_low, k_low, rho_low = h[low], k[low], rho[low]
            hk = (h_low * k_low)[:, None]
            hs = ((h_low * h_low + k_low * k_low) / 2)[:, None]
            arcsine = np.arcsin(rho_low)
            integral = np.zeros(h_low.shape)
            for sign in (-1, 1):
                sine = np.sin(arcsine[:, None] * (sign * nodes + 1) / 2)
                integral = integral + np.sum(
                    weights * np.exp((sine * hk - hs) / (1 - sine * sine)), axis=1
                )
            integral = np.where(rho_low != 0, integral * arcsine / (4 * np.pi), 0.0)
            probability[low] = integral + ndtr(-h_low) * ndtr(-k_low)

        if not (high := (number_of_points == points) & near_one).any():
            continue

        h_high, rho_high = h[high], rho[high]
        k_high = np.where(rho_high < 0, -k[high], k[high])
        hk = h_high * k_high
        integral = np.zeros(h_high.shape)
        inside = np.abs(rho_high) < 1

        one_minus_rho_squared = (1 - rho_high) * (1 + rho_high)
        a = np.sqrt(one_minus_rho_squared)
        bs = (h_high - k_high) ** 2
        c = (4 - hk) / 8
        d = (12 - hk) / 16
        exponent = -(bs / one_minus_rho_squared + hk) / 2
        integral = np.where(
            inside & (exponent > -100),  # noqa: PLR2004
            a
            * np.exp(exponent)
            * (
                1
                - c * (bs - one_minus_rho_squared) * (1 - d * bs / 5) / 3
                + c * d * one_minus_rho_squared**2 / 5
            ),
            integral,
        )
        b = np.sqrt(bs)
        integral = np.where(
            inside & (-hk < 100),  # noqa: PLR2004
            integral
            - np.exp(-hk / 2)
            * np.sqrt(2 * np.pi)
            * ndtr(-b / a)
            * b
            * (1 - c * bs * (1 - d * bs / 5) / 3),
            integral,
        )
        half = (a / 2)[:, None]
        for sign in (-1, 1):
            xs = (half * (sign * nodes + 1)) ** 2
            rs = np.sqrt(1 - xs)
            exponent = -(bs[:, None] / xs + hk[:, None]) / 2
            integral = integral + np.where(
                inside,
                np.sum(
                    np.where(
                        exponent > -100,  # noqa: PLR2004
                        half
                        * weights
                        * np.exp(exponent)
                        * (
                            np.exp(-hk[:, None] * (1 - rs) / (2 * (1 + rs))) / rs
                            - (1 + c[:, None] * xs * (1 + d[:, None] * xs))
                        ),
                        0.0,
                    ),
                    axis=1,
                ),
                0.0,
            )
        integral = np.where(inside, -integral / (2 * np.pi), 0.0)

        probability[high] = np.where(
            rho_high > 0,
            integral + ndtr(-np.maximum(h_high, k_high)),
            -integral + np.where(k_high > h_high, ndtr(k_high) - ndtr(h_high), 0.0),
        )

    probability = probability.reshape(x.shape)

    return probability if probability.ndim else float(probability)


def _power_times_probability(base, exponent, probability):
    """
    Helper function that returns base^exponent * probability in log space. In the
    Bjerksund-Stensland approximation the exponent grows like 1 / σ², so at low volatility
    the power overflows while the probability it multiplies underflows to zero; their
    product is finite and small, which the separate factors (inf * 0) would lose.

    Args:
        base (float | np.ndarray): The positive base of the power.
        exponent (float | np.ndarray): The exponent of the power.
        probability (float | np.ndarray): The (non-negative) probability the power
            multiplies.

    Returns:
        float | np.ndarray: The product base^exponent * probability.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(
            probability <= 0,
            0.0,
            np.exp(exponent * np.log(base) + np.log(probability)),
        )


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
    return np.exp(lambda_ + gamma * np.log(stock_price / scale_price)) * (
        ndtr(d1) - np.exp(kappa * np.log(boundary_price / stock_price) + log_ndtr(d2))
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

    - ψ = e^(λT) * S^γ * [M(—e1, —f1, ρ) — (I2 / S)^κ * M(—e2, —f2, ρ) —
      (I1 / S)^κ * M(—e3, —f3, —ρ) + (I1 / I2)^κ * M(—e4, —f4, —ρ)]

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
    return np.exp(
        lambda_ * time_to_expiration + gamma * np.log(stock_price / scale_price)
    ) * (
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


def _bjerksund_stensland_bounds(
    stock_price: np.ndarray,
    strike_price: np.ndarray,
    risk_free_rate: np.ndarray,
    cost_of_carry: np.ndarray,
    volatility: np.ndarray,
    time_to_expiration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function that returns the inputs of the Bjerksund-Stensland call approximations
    as arrays of one shape, the exercise value max(S — K, 0), the generalized Black-Scholes
    price of the European call with the same cost of carry (through the dividend yield
    q = r — b), and the elements where early exercise can pay: before expiration and with a
    cost of carry below the risk-free rate. Both values are lower bounds of the American
    call, which elsewhere equals the larger of the two (the exercise value at expiration).
    Early exercise is not optimal when the cost of carry is at least the risk-free rate,
    except with a negative risk-free rate, where paying the strike later costs more than
    paying it now and deep in-the-money calls are exercised; the exercise value floors the
    price there.
    """
    exercise_value = np.maximum(stock_price - strike_price, 0.0)
    european_value = np.asarray(
        get_black_scholes(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=risk_free_rate - cost_of_carry,
        ),
        dtype=float,
    )
    early = (time_to_expiration > 0) & (cost_of_carry < risk_free_rate)

    return exercise_value, european_value, early


def _bjerksund_stensland_floor(
    value: np.ndarray,
    boundary: np.ndarray,
    stock_price: np.ndarray,
    strike_price: np.ndarray,
    risk_free_rate: np.ndarray,
    cost_of_carry: np.ndarray,
    volatility: np.ndarray,
    time_to_expiration: np.ndarray,
    european_value: np.ndarray,
    exercise_value: np.ndarray,
) -> np.ndarray:
    """
    Helper function that turns the formula values of a Bjerksund-Stensland call into prices.
    At or above the (last) boundary immediate exercise is optimal, and extreme inputs can
    leave the approximation undefined, where the European price is the closest lower bound.
    The flat-boundary approximation is a lower bound that can fall below the European price
    (high volatility, long maturities) or the exercise value, both lower bounds of an
    American option, so the result is floored at both. The boundary function is built for
    I > K: with a negative cost of carry and low volatility (b * T + 2σ√T < 0) or a
    non-positive risk-free rate it falls to or below the strike, where the approximation
    would exercise options that are out of the money, and a binomial tree prices those
    options instead.
    """
    value = np.where(
        np.isfinite(value) & (stock_price < boundary), value, european_value
    )
    value = np.maximum(np.maximum(value, european_value), exercise_value)

    for index in np.flatnonzero(~np.isfinite(boundary) | (boundary <= strike_price)):
        value[index] = max(
            _american_call_binomial_value(
                stock_price=stock_price[index],
                strike_price=strike_price[index],
                risk_free_rate=risk_free_rate[index],
                cost_of_carry=cost_of_carry[index],
                volatility=volatility[index],
                time_to_expiration=time_to_expiration[index],
            ),
            european_value[index],
            exercise_value[index],
        )

    return value


def _bjerksund_stensland_call(
    stock_price: np.ndarray,
    strike_price: np.ndarray,
    risk_free_rate: np.ndarray,
    cost_of_carry: np.ndarray,
    volatility: np.ndarray,
    time_to_expiration: np.ndarray,
) -> np.ndarray:
    """
    Prices American call options using the Bjerksund-Stensland (2002) closed-form
    approximation with a two-step flat early-exercise boundary. Puts are priced through
    the put-call transformation applied by get_bjerksund_stensland. The value is floored
    at the European price and the exercise value, and a binomial tree takes over where the
    boundary falls to or below the strike. Every input is an array of the same shape, so
    a grid of strikes and expirations is priced at once.

    Args:
        stock_price (np.ndarray): The current stock price.
        strike_price (np.ndarray): The option's strike price.
        risk_free_rate (np.ndarray): The risk-free interest rate.
        cost_of_carry (np.ndarray): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (np.ndarray): The volatility of the stock.
        time_to_expiration (np.ndarray): The time to expiration of the option.

    Returns:
        np.ndarray: The value of the American call options.
    """
    inputs = (
        stock_price,
        strike_price,
        risk_free_rate,
        cost_of_carry,
        volatility,
        time_to_expiration,
    )
    exercise_value, european_value, early = _bjerksund_stensland_bounds(*inputs)
    value = np.maximum(european_value, exercise_value)

    if not early.any():
        return value

    # The elements where early exercise can pay, priced by the approximation.
    (
        stock_price,
        strike_price,
        risk_free_rate,
        cost_of_carry,
        volatility,
        time_to_expiration,
    ) = (array[early] for array in inputs)

    # The boundary steps at the golden-ratio point of the option's life, the split that
    # Bjerksund and Stensland (2002) found to give the most accurate prices.
    first_time = 0.5 * (np.sqrt(5) - 1) * time_to_expiration
    beta = (0.5 - cost_of_carry / volatility**2) + np.sqrt(
        (cost_of_carry / volatility**2 - 0.5) ** 2 + 2 * risk_free_rate / volatility**2
    )
    b_infinity = beta / (beta - 1) * strike_price
    b_zero = np.maximum(
        strike_price, risk_free_rate / (risk_free_rate - cost_of_carry) * strike_price
    )

    def boundary(time: np.ndarray) -> np.ndarray:
        h_time = (
            -(cost_of_carry * time + 2 * volatility * np.sqrt(time))
            * strike_price**2
            / ((b_infinity - b_zero) * b_zero)
        )
        return b_zero + (b_infinity - b_zero) * (1 - np.exp(h_time))

    first_boundary = boundary(first_time)
    second_boundary = boundary(time_to_expiration)

    def phi(gamma, trigger_price, boundary_price, scale_price=1.0) -> np.ndarray:
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

    def psi(gamma, trigger_price, scale_price=1.0) -> np.ndarray:
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
    approximation = (
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

    value[early] = _bjerksund_stensland_floor(
        approximation,
        second_boundary,
        stock_price,
        strike_price,
        risk_free_rate,
        cost_of_carry,
        volatility,
        time_to_expiration,
        european_value[early],
        exercise_value[early],
    )

    return value


def _bjerksund_stensland_1993_call(
    stock_price: np.ndarray,
    strike_price: np.ndarray,
    risk_free_rate: np.ndarray,
    cost_of_carry: np.ndarray,
    volatility: np.ndarray,
    time_to_expiration: np.ndarray,
) -> np.ndarray:
    """
    Prices American call options using the Bjerksund-Stensland (1993) closed-form
    approximation with one flat early-exercise boundary over the option's life. Puts are
    priced through the put-call transformation applied by get_bjerksund_stensland. The
    value is floored at the European price and the exercise value, and a binomial tree takes
    over where the boundary falls to or below the strike, as in the 2002 version. Every
    input is an array of the same shape.

    Args:
        stock_price (np.ndarray): The current stock price.
        strike_price (np.ndarray): The option's strike price.
        risk_free_rate (np.ndarray): The risk-free interest rate.
        cost_of_carry (np.ndarray): The cost of carry, defined as risk_free_rate minus
            dividend_yield.
        volatility (np.ndarray): The volatility of the stock.
        time_to_expiration (np.ndarray): The time to expiration of the option.

    Returns:
        np.ndarray: The value of the American call options.
    """
    inputs = (
        stock_price,
        strike_price,
        risk_free_rate,
        cost_of_carry,
        volatility,
        time_to_expiration,
    )
    exercise_value, european_value, early = _bjerksund_stensland_bounds(*inputs)
    value = np.maximum(european_value, exercise_value)

    if not early.any():
        return value

    (
        stock_price,
        strike_price,
        risk_free_rate,
        cost_of_carry,
        volatility,
        time_to_expiration,
    ) = (array[early] for array in inputs)

    beta = (0.5 - cost_of_carry / volatility**2) + np.sqrt(
        (cost_of_carry / volatility**2 - 0.5) ** 2 + 2 * risk_free_rate / volatility**2
    )
    b_infinity = beta / (beta - 1) * strike_price
    b_zero = np.maximum(
        strike_price, risk_free_rate / (risk_free_rate - cost_of_carry) * strike_price
    )
    # The single flat boundary I = B0 + (B∞ — B0) * (1 — e^h(T)), with the 1993 h(T).
    h_time = (
        -(
            cost_of_carry * time_to_expiration
            + 2 * volatility * np.sqrt(time_to_expiration)
        )
        * b_zero
        / (b_infinity - b_zero)
    )
    boundary = b_zero + (b_infinity - b_zero) * (1 - np.exp(h_time))

    def phi(gamma, trigger_price, scale_price=1.0) -> np.ndarray:
        return _phi(
            stock_price=stock_price,
            time_to_expiration=time_to_expiration,
            gamma=gamma,
            trigger_price=trigger_price,
            boundary_price=boundary,
            risk_free_rate=risk_free_rate,
            cost_of_carry=cost_of_carry,
            volatility=volatility,
            scale_price=scale_price,
        )

    # α * S^β and α * φ(β) written as (I — K) * (S / I)^β and (I — K) * φ(β) / I^β, which
    # stay finite at low volatility, see _bjerksund_stensland_call.
    premium = boundary - strike_price
    approximation = (
        premium * (stock_price / boundary) ** beta
        - premium * phi(beta, boundary, scale_price=boundary)
        + phi(1, boundary)
        - phi(1, strike_price)
        - strike_price * phi(0, boundary)
        + strike_price * phi(0, strike_price)
    )

    value[early] = _bjerksund_stensland_floor(
        approximation,
        boundary,
        stock_price,
        strike_price,
        risk_free_rate,
        cost_of_carry,
        volatility,
        time_to_expiration,
        european_value[early],
        exercise_value[early],
    )

    return value


@np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
def get_bjerksund_stensland(
    stock_price: float | np.ndarray,
    strike_price: float | np.ndarray,
    risk_free_rate: float | np.ndarray,
    volatility: float | np.ndarray,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float | np.ndarray = 0.0,
    put_option: bool = False,
    method: str = "2002",
) -> float | np.ndarray:
    """
    Calculate the price of an American option, which may be exercised at any time up to
    expiration, with the closed-form approximation of Bjerksund and Stensland (2002), or
    their original version of 1993 with one flat exercise boundary.

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

    - C = α2 * S^β — α2 * φ(S, t1, β, I2, I2) + φ(S, t1, 1, I2, I2) — φ(S, t1, 1, I1, I2) —
      K * φ(S, t1, 0, I2, I2) + K * φ(S, t1, 0, I1, I2) + α1 * φ(S, t1, β, I1, I2) —
      α1 * ψ(S, T, β, I1, I2, I1, t1) + ψ(S, T, 1, I1, I2, I1, t1) — ψ(S, T, 1, K, I2, I1, t1) —
      K * ψ(S, T, 0, I1, I2, I1, t1) + K * ψ(S, T, 0, K, I2, I1, t1)

    Where φ is a knock-out claim over one period (a univariate normal expression) and ψ
    its two-period counterpart (a bivariate normal expression), see ``_phi`` and ``_psi``.

    The 1993 version keeps one flat boundary over the whole life of the option:

    - h(T) = —(b * T + 2σ√T) * B0 / (B∞ — B0)
    - I = B0 + (B∞ — B0) * (1 — e^h(T)),  α = (I — K) * I^(—β)
    - C = α * S^β — α * φ(S, T, β, I, I) + φ(S, T, 1, I, I) — φ(S, T, 1, K, I) —
      K * φ(S, T, 0, I, I) + K * φ(S, T, 0, K, I)

    When b ≥ r it is never optimal to exercise an American call early, so the value
    collapses to the standard European Black-Scholes price. American puts are priced
    through the put-call transformation:

    - AmericanPut(S, K, T, r, b, σ) = AmericanCall(K, S, T, r — b, —b, σ)

    An American option is worth at least its European counterpart and its exercise value,
    while the flat-boundary approximation can fall below the European price at high
    volatility and long maturities. The result is therefore floored at both. Where the
    boundary I2 falls to or below the strike (a negative cost of carry at low volatility, or
    a non-positive risk-free rate), the approximation would exercise options that are out
    of the money, and a 1,000-step binomial tree prices the option instead.

    Referenced against:
    - QuantLib, BjerksundStenslandApproximationEngine (1993): https://www.quantlib.org/reference/class_quant_lib_1_1_bjerksund_stensland_approximation_engine.html
    - QuantLib, FdBlackScholesVanillaEngine (2002): https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_vanilla_engine.html

    Args:
        stock_price (float | np.ndarray): The current stock price.
        strike_price (float | np.ndarray): The option's strike price.
        risk_free_rate (float | np.ndarray): The continuously compounded risk-free interest
            rate.
        volatility (float | np.ndarray): The volatility of the stock.
        time_to_expiration (float | np.ndarray): The time to expiration of the option, in
            years. At zero the option is worth its exercise value.
        dividend_yield (float | np.ndarray): The continuous dividend yield of the stock.
            Defaults to 0.
        put_option (bool): Whether the option is a put option or not. Defaults to False.
        method (str): The version of the approximation. Options are: "2002" (two flat
            boundaries), "1993" (one flat boundary). Defaults to "2002".

    Returns:
        float | np.ndarray: The Bjerksund-Stensland approximation of the American option
        value, an array in the shape of the inputs when any of them is an array (a grid of
        strikes and expirations is priced in one call).

    Raises:
        TypeError: if any of the numeric inputs is not a float, int or np.ndarray.
        ValueError: if volatility is not positive or method is not "2002" or "1993".

    Notes:
        Reference for the 1993 version: Bjerksund, P., & Stensland, G. (1993). "Closed-Form
        Approximation of American Options." Scandinavian Journal of Management, 9, S87-S99,
        as presented in Haug (2007), section 3.1.3. QuantLib's
        BjerksundStenslandApproximationEngine also searches the boundary that maximizes the
        price where the cost of carry is negative and the 1993 boundary falls short, where
        this version uses the binomial tree instead. For S = K = 100, r = 5%,
        q = 3%, σ = 25% and T = 1 the 1993 put is 8.8152, against 8.8439 for the 2002
        version.

        Reference for the 2002 version: Bjerksund, P., & Stensland, G. (2002). "Closed Form Valuation of
        American Options." Working paper, Norwegian School of Economics and Business
        Administration (NHH). As presented in Haug, E. G. (2007). "The Complete Guide to
        Option Pricing Formulas" (2nd ed.), section 3.1.4. Compared with a finite-difference
        American price, the approximation is slightly low; for S = K = 100, r = 5%,
        q = 3%, σ = 25% and T = 1 the put is 8.8439 against 8.8823. QuantLib's
        BjerksundStenslandApproximationEngine implements the 1993 model with its single flat
        boundary, so its prices are close to but not a reference for the 2002 version; the
        prices of Haug (2007) and a finite-difference American price are.

        Both versions are derived for a positive risk-free rate and cost of carry. Where either is
        negative the flat boundaries can fall back to the European value while early exercise is
        worth more: for S = K = 100, r = -2%, q = -3%, σ = 1% and T = 1 both give 0.0854 for the
        put, against 0.1669 for a 10,000-step binomial tree. american_model.get_qd_plus prices
        these options accurately.
    """
    validate_numeric_inputs(
        arrays=True,
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )

    # The exercise boundary divides by the variance, which a constant price does not have.
    if np.any(np.asarray(volatility) <= 0):
        raise ValueError(f"volatility must be positive, received {volatility!r}.")

    check_option("method", method, ("2002", "1993"))

    call = (
        _bjerksund_stensland_call
        if method == "2002"
        else _bjerksund_stensland_1993_call
    )
    (
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
    ) = np.broadcast_arrays(
        *(
            np.asarray(value, dtype=float)
            for value in (
                stock_price,
                strike_price,
                risk_free_rate,
                volatility,
                time_to_expiration,
                dividend_yield,
            )
        )
    )
    shape = stock_price.shape
    cost_of_carry = (risk_free_rate - dividend_yield).ravel()

    # American put via P(S, K, T, r, b, v) = C(K, S, T, r - b, -b, v).
    value = call(
        stock_price=(strike_price if put_option else stock_price).ravel(),
        strike_price=(stock_price if put_option else strike_price).ravel(),
        risk_free_rate=risk_free_rate.ravel() - (cost_of_carry if put_option else 0.0),
        cost_of_carry=-cost_of_carry if put_option else cost_of_carry,
        volatility=volatility.ravel(),
        time_to_expiration=time_to_expiration.ravel(),
    ).reshape(shape)

    return value if value.ndim else float(value)


# The points of the log price grid and of the Gauss-Hermite rule with which the Haug, Haug
# and Lewis method carries the option value back from one dividend date to the one before.
_DIVIDEND_GRID_POINTS = 401
_DIVIDEND_GRID_WIDTH = 10.0
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite_e.hermegauss(96)
_HERMITE_WEIGHTS = _HERMITE_WEIGHTS / np.sqrt(2 * np.pi)


def _dividends_before_expiration(
    dividend_times: list[float] | np.ndarray,
    dividend_amounts: list[float] | np.ndarray,
    time_to_expiration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function with the cash dividends that go ex-dividend after today and no later
    than the expiration, in date order.
    """
    times = np.asarray(dividend_times, dtype=float)
    amounts = np.asarray(dividend_amounts, dtype=float)
    if times.shape != amounts.shape:
        raise ValueError(
            "dividend_times and dividend_amounts must have the same length, received "
            f"{times.size} and {amounts.size}."
        )
    keep = (times > 0) & (times <= time_to_expiration) & (amounts != 0)
    order = np.argsort(times[keep])
    return times[keep][order], amounts[keep][order]


def _haug_haug_lewis(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_times: np.ndarray,
    dividend_amounts: np.ndarray,
    put_option: bool,
) -> float:
    """
    Helper function with the European option price when the stock drops by each cash
    dividend on its ex-date, found by integrating the value after each dividend over the
    lognormal distribution of the stock price before it (Haug, Haug and Lewis, 2003).

    After the last dividend the option is a Black-Scholes option on the stock net of that
    dividend. Before it, the value at each earlier dividend date is the discounted
    expectation of the value at the next one, taken with a Gauss-Hermite rule and kept on a
    grid of log prices (interpolated with a cubic spline), so the work grows linearly with
    the number of dividends. A dividend larger than the stock price leaves a stock worth 0.
    """

    def after_dividend(price: np.ndarray, index: int, table) -> np.ndarray:
        """The value just after dividend index, at the stock price net of the dividend."""
        if index == len(dividend_times) - 1:
            return np.asarray(
                get_black_scholes(
                    stock_price=np.maximum(price, 0.0),
                    strike_price=strike_price,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    time_to_expiration=time_to_expiration - dividend_times[index],
                    put_option=put_option,
                ),
                dtype=float,
            )
        return table(price)

    def tabulate(index: int, table):
        """The value just after dividend index as a function of the stock price."""
        if index == len(dividend_times) - 1:
            return None
        remaining = time_to_expiration - dividend_times[index]
        spread = volatility * np.sqrt(remaining)
        log_prices = np.log(stock_price) + np.linspace(
            -_DIVIDEND_GRID_WIDTH * spread - 1.0,
            _DIVIDEND_GRID_WIDTH * spread + 1.0,
            _DIVIDEND_GRID_POINTS,
        )
        grid = np.exp(log_prices)
        step = dividend_times[index + 1] - dividend_times[index]
        future = grid[:, None] * np.exp(
            (risk_free_rate - 0.5 * volatility**2) * step
            + volatility * np.sqrt(step) * _HERMITE_NODES[None, :]
        )
        values = np.exp(-risk_free_rate * step) * (
            after_dividend(future - dividend_amounts[index + 1], index + 1, table)
            @ _HERMITE_WEIGHTS
        )
        spline = CubicSpline(log_prices, values)
        # The value of a worthless stock: a call is worth 0, a put the discounted strike.
        floor_value = (
            strike_price * np.exp(-risk_free_rate * remaining) if put_option else 0.0
        )
        slope = (values[-1] - values[-2]) / (grid[-1] - grid[-2])

        def table_function(price: np.ndarray) -> np.ndarray:
            price = np.asarray(price, dtype=float)
            inside = np.clip(price, grid[0], grid[-1])
            result = spline(np.log(inside))
            # Below the grid, linear towards the value of a worthless stock; above it,
            # linear along the last slope, as the option value is linear far from the strike.
            below = price < grid[0]
            result = np.where(
                below,
                floor_value
                + (values[0] - floor_value) * np.maximum(price, 0.0) / grid[0],
                result,
            )
            return np.where(
                price > grid[-1], values[-1] + slope * (price - grid[-1]), result
            )

        return table_function

    # Backwards from the last dividend: tables[i] is the value just after dividend i.
    tables: list = [None] * len(dividend_times)
    for index in range(len(dividend_times) - 2, -1, -1):
        tables[index] = tabulate(index, tables[index + 1])

    first = dividend_times[0]
    future = stock_price * np.exp(
        (risk_free_rate - 0.5 * volatility**2) * first
        + volatility * np.sqrt(first) * _HERMITE_NODES
    )
    return float(
        np.exp(-risk_free_rate * first)
        * (
            after_dividend(future - dividend_amounts[0], 0, tables[0])
            @ _HERMITE_WEIGHTS
        )
    )


@np.errstate(divide="ignore", invalid="ignore")
def get_black_scholes_cash_dividends(
    stock_price: float | np.ndarray,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float | np.ndarray,
    dividend_times: list[float] | np.ndarray,
    dividend_amounts: list[float] | np.ndarray,
    put_option: bool = False,
    method: str = "escrowed",
):
    """
    Calculate the price of a European option on a stock that pays known cash dividends
    before expiration: what a call or put is worth when the stock drops by each dividend on
    its ex-dividend date, instead of a continuous dividend yield.

    A cash dividend lowers the stock price by its amount on the ex-dividend date, which a
    dividend yield only spreads over the year. Two methods are available:

    - The escrowed dividend model (Hull, 2022, chapter 17): the stock price net of the present value of
      the dividends up to expiration follows the lognormal process, so the Black-Scholes
      formula applies to S* = S — Σ D_i * e^(—r * t_i) with no dividend yield. It is the
      market convention and exact for that process, but the volatility then applies to
      S* rather than S, which lowers the volatility of the stock itself.
    - The Haug-Haug-Lewis model (Haug, Haug and Lewis, 2003): the stock price itself follows the
      lognormal process and drops by D_i at t_i (to 0 when the dividend exceeds it). After
      the last dividend the option is a Black-Scholes option; before it, its value at each
      dividend date is the discounted expectation of the value just after the next one:

      - V_i(x) = e^(—r * (t_(i+1) — t_i)) * E[V_(i+1)(S_(t_(i+1)) — D_(i+1)) | S_(t_i) = x]
      - V = e^(—r * t_1) * E[V_1(S_(t_1) — D_1)]

      The expectations are one-dimensional integrals over the lognormal distribution, taken
      with a 96 point Gauss-Hermite rule, with the value between dividends kept on a grid of
      401 log prices.

    Where S is the stock price, K the strike price, r the continuously compounded risk-free
    rate, σ the volatility, D_i the cash dividend with ex-date t_i years from now and T the
    time to expiration. Dividends after T or not after today are left out.

    Referenced against:
    - QuantLib, AnalyticDividendEuropeanEngine (escrowed): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_dividend_european_engine.html
    - QuantLib, CashDividendEuropeanEngine (Haug-Haug-Lewis): https://www.quantlib.org/reference/class_quant_lib_1_1_cash_dividend_european_engine.html

    Args:
        stock_price (float | np.ndarray): The current stock price.
        strike_price (float | np.ndarray): The option's strike price.
        risk_free_rate (float): The continuously compounded risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float | np.ndarray): The time to expiration of the option, in
            years.
        dividend_times (list[float] | np.ndarray): The times of the ex-dividend dates, in
            years from now.
        dividend_amounts (list[float] | np.ndarray): The cash dividend per share of each
            ex-dividend date.
        put_option (bool): Whether the option is a put option or not. Defaults to False.
        method (str): The dividend model. Options are: "escrowed" (escrowed dividend
            model), "haug_haug_lewis" (Haug-Haug-Lewis). Defaults to "escrowed". The
            Haug-Haug-Lewis method takes a single stock price, strike price and time to
            expiration; the escrowed method also takes arrays of them.

    Returns:
        float | np.ndarray: The value of the European option.

    Raises:
        ValueError: if method is not "escrowed" or "haug_haug_lewis", the dividend times
            and amounts differ in length or the volatility is not positive.

    Notes:
        References: Hull, J. C. (2022). "Options, Futures, and Other Derivatives" (11th
        ed.), chapter 17; Haug, E. G., Haug, J., & Lewis, A. (2003). "Back to Basics: a New
        Approach to the Discrete Dividend Problem." Wilmott Magazine, September, 37-47.
        - QuantLib's CashDividendEuropeanEngine is used with the Spot model.
        - For S = K = 100, r = 5%, σ = 30%, T = 1 and a dividend of 2 after 30 days, the
          escrowed call is 13.0133 and the Haug-Haug-Lewis call 13.0325.
    """
    check_option("method", method, ("escrowed", "haug_haug_lewis"))
    if volatility <= 0:
        raise ValueError(f"volatility must be positive, received {volatility!r}.")

    if method == "escrowed":
        times = np.asarray(dividend_times, dtype=float)
        amounts = np.asarray(dividend_amounts, dtype=float)
        if times.shape != amounts.shape:
            raise ValueError(
                "dividend_times and dividend_amounts must have the same length, received "
                f"{times.size} and {amounts.size}."
            )
        expirations = np.asarray(time_to_expiration, dtype=float)
        # The present value of the dividends up to each expiration.
        present_value = (
            (amounts * np.exp(-risk_free_rate * times))[None, :]
            * ((times > 0)[None, :] & (times[None, :] <= expirations.reshape(-1, 1)))
        ).sum(axis=1)
        present_value = (
            float(present_value[0])
            if expirations.ndim == 0
            else present_value.reshape(expirations.shape)
        )
        return get_black_scholes(
            stock_price=np.maximum(stock_price - present_value, 0.0),
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            put_option=put_option,
        )

    times, amounts = _dividends_before_expiration(
        dividend_times, dividend_amounts, float(time_to_expiration)
    )
    if times.size == 0 or time_to_expiration <= 0:
        return get_black_scholes(
            stock_price=stock_price,
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            put_option=put_option,
        )

    return _haug_haug_lewis(
        stock_price=float(stock_price),
        strike_price=float(strike_price),
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=float(time_to_expiration),
        dividend_times=times,
        dividend_amounts=amounts,
        put_option=put_option,
    )
