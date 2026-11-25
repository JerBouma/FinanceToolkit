"""Multi-Asset Option Model"""

__docformat__ = "google"

from typing import Literal

import numpy as np
from scipy.special import ndtr

from financetoolkit.options.helpers import (
    bivariate_normal_cdf,
    validate_numeric_inputs,
    validate_positive_inputs,
)
from financetoolkit.options.options_model import (
    MAX_SIMULATED_VALUES,
    SIMULATION_CHUNK_SIZE,
)
from financetoolkit.options.pricing.american_model import get_qd_plus
from financetoolkit.utilities.error_model import check_option

# Below this volatility (of the ratio of two prices, or of a spread) the two-asset formulas, which
# divide by it, have no meaning: the prices move as one.
_NO_VOLATILITY = 1e-12


def _validate_two_assets(
    first_volatility: float,
    second_volatility: float,
    correlation: float,
    time_to_expiration: float,
) -> None:
    """
    Raises a ValueError for a volatility that is not positive, a correlation outside -1 to 1
    or a negative time to expiration, which the two-asset formulas cannot take.
    """
    if first_volatility <= 0 or second_volatility <= 0:
        raise ValueError(
            "first_volatility and second_volatility must be positive, received "
            f"{first_volatility!r} and {second_volatility!r}."
        )
    if not -1 <= correlation <= 1:
        raise ValueError(
            f"correlation must be between -1 and 1, received {correlation!r}."
        )
    if time_to_expiration < 0:
        raise ValueError(
            f"time_to_expiration must be at least 0, received {time_to_expiration!r}."
        )


def _ratio_volatility(
    first_volatility: float, second_volatility: float, correlation: float
) -> float:
    """
    The volatility of the ratio of two prices, √(σ1² + σ2² - 2ρσ1σ2), which the exchange and
    rainbow formulas divide by. Two prices that move as one (ρ = 1 and σ1 = σ2) have a ratio
    that never changes, for which the formulas have no meaning.
    """
    volatility = float(
        np.sqrt(
            max(
                first_volatility**2
                + second_volatility**2
                - 2 * correlation * first_volatility * second_volatility,
                0.0,
            )
        )
    )
    if volatility <= _NO_VOLATILITY:
        raise ValueError(
            "The two prices move as one (correlation 1 and equal volatilities), so the "
            "volatility of their ratio is zero and the option has no two-asset value."
        )

    return volatility


def get_exchange_option(
    first_stock_price: float,
    second_stock_price: float,
    first_volatility: float,
    second_volatility: float,
    correlation: float,
    time_to_expiration: float,
    first_dividend_yield: float = 0.0,
    second_dividend_yield: float = 0.0,
    american_option: bool = False,
) -> float:
    """
    Calculate the price of an option to exchange the second asset for the first: what the
    right to receive the first asset for the second at expiration is worth, max(S1 - S2, 0),
    with the formula of Margrabe (1978).

    Measured in units of the second asset, the payoff is a call on the ratio S1 / S2 with a
    strike of 1, so the risk-free rate drops out and the volatility is that of the ratio:

    - Exchange = S1 * e^(-q1 * T) * N(d1) - S2 * e^(-q2 * T) * N(d2)
    - σ = √(σ1² + σ2² - 2ρσ1σ2)
    - d1 = (ln(S1 / S2) + (q2 - q1 + σ² / 2) * T) / (σ√T), d2 = d1 - σ√T

    An American exchange option may be exercised at any time. In units of the second asset it
    is an American call on S1 with strike S2, a risk-free rate of q2 and a dividend yield of
    q1 (Bjerksund and Stensland, 1993), which is priced with the QD+ exercise boundary and
    its integral (american_model.get_qd_plus), within about 0.1% of a 999-step Leisen-Reimer
    tree, where the Bjerksund-Stensland approximations, lower bounds, lie about 0.6% (2002)
    and 1% (1993) below it. Without a dividend yield on the first asset early exercise is
    never optimal and the American option is worth the European one.

    Where S1 and S2 are the prices of the asset received and the asset delivered (times their
    quantities, for an option on Q1 units against Q2 units), σ1 and σ2 their volatilities, ρ
    the correlation of their returns, q1 and q2 their continuous dividend yields (or, for
    futures, the continuously compounded risk-free rate), T the time to expiration in years
    and N the standard normal distribution function.

    Referenced against:
    - QuantLib, AnalyticEuropeanMargrabeEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_european_margrabe_engine.html

    Shares similar logic:
    - QuantLib, AnalyticAmericanMargrabeEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_american_margrabe_engine.html

    Args:
        first_stock_price (float): The price of the asset received, S1.
        second_stock_price (float): The price of the asset delivered, S2.
        first_volatility (float): The volatility of the first asset.
        second_volatility (float): The volatility of the second asset.
        correlation (float): The correlation of the returns of the two assets, between -1
            and 1.
        time_to_expiration (float): The time to expiration in years. At 0 the option is worth
            max(S1 - S2, 0).
        first_dividend_yield (float, optional): The continuous dividend yield of the first
            asset. Defaults to 0.
        second_dividend_yield (float, optional): The continuous dividend yield of the second
            asset. Defaults to 0.
        american_option (bool, optional): Whether the option can be exercised before
            expiration. Defaults to False.

    Returns:
        float: The exchange option value.

    Raises:
        TypeError: If any of the inputs is not a float or int.
        ValueError: If a volatility is not positive, the correlation is outside -1 to 1, the
            time to expiration is negative or the two prices move as one (correlation 1 and
            equal volatilities).

    Notes:
        References: Margrabe, W. (1978). "The Value of an Option to Exchange One Asset for
        Another." Journal of Finance, 33(1), 177-186. Bjerksund, P., & Stensland, G. (1993).
        "American Exchange Options and a Put-Call Transformation: A Note." Journal of
        Business Finance & Accounting, 20(5), 761-764. Haug, E. G. (2007). The Complete Guide
        to Option Pricing Formulas, 2nd ed., chapter 5.
        - The American value is checked against a Leisen-Reimer tree of 999 steps on the call
          it equals. QuantLib's AnalyticAmericanMargrabeEngine takes the Bjerksund-Stensland
          (1993) approximation, a lower bound about 1% below that value.

    As an example:

    ```python
    from financetoolkit.options.multiasset import multi_asset_model

    european = multi_asset_model.get_exchange_option(
        first_stock_price=22,
        second_stock_price=20,
        first_volatility=0.2,
        second_volatility=0.25,
        correlation=-0.5,
        time_to_expiration=0.1,
        first_dividend_yield=0.06,
        second_dividend_yield=0.04,
    )
    american = multi_asset_model.get_exchange_option(
        first_stock_price=22,
        second_stock_price=20,
        first_volatility=0.2,
        second_volatility=0.25,
        correlation=-0.5,
        time_to_expiration=0.1,
        first_dividend_yield=0.06,
        second_dividend_yield=0.04,
        american_option=True,
    )

    round(european, 4), round(american, 4)
    ```

    Which returns:

    ```
    (2.2827, 2.2939)
    ```
    """
    validate_numeric_inputs(
        first_stock_price=first_stock_price,
        second_stock_price=second_stock_price,
        first_volatility=first_volatility,
        second_volatility=second_volatility,
        correlation=correlation,
        time_to_expiration=time_to_expiration,
        first_dividend_yield=first_dividend_yield,
        second_dividend_yield=second_dividend_yield,
    )
    _validate_two_assets(
        first_volatility, second_volatility, correlation, time_to_expiration
    )
    validate_positive_inputs(
        first_stock_price=first_stock_price, second_stock_price=second_stock_price
    )
    volatility = _ratio_volatility(first_volatility, second_volatility, correlation)

    if time_to_expiration == 0:
        return max(first_stock_price - second_stock_price, 0.0)

    if american_option:
        # In units of the second asset the option is an American call on S1 with strike S2,
        # where holding the second asset earns its yield q2 as the "risk-free rate".
        return get_qd_plus(
            stock_price=float(first_stock_price),
            strike_price=float(second_stock_price),
            risk_free_rate=float(second_dividend_yield),
            volatility=float(volatility),
            time_to_expiration=float(time_to_expiration),
            dividend_yield=float(first_dividend_yield),
        )

    deviation = volatility * np.sqrt(time_to_expiration)
    d1 = (
        np.log(first_stock_price / second_stock_price)
        + (second_dividend_yield - first_dividend_yield + 0.5 * volatility**2)
        * time_to_expiration
    ) / deviation

    return float(
        first_stock_price
        * np.exp(-first_dividend_yield * time_to_expiration)
        * ndtr(d1)
        - second_stock_price
        * np.exp(-second_dividend_yield * time_to_expiration)
        * ndtr(d1 - deviation)
    )


def get_spread_option(
    first_stock_price: float,
    second_stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    first_volatility: float,
    second_volatility: float,
    correlation: float,
    time_to_expiration: float,
    first_dividend_yield: float = 0.0,
    second_dividend_yield: float = 0.0,
    put_option: bool = False,
    method: Literal["bjerksund_stensland", "kirk"] = "bjerksund_stensland",
) -> float:
    """
    Calculate the price of a European spread option: what an option on the difference of two
    asset prices is worth, max(S1 - S2 - K, 0) for a call and max(K - S1 + S2, 0) for a put
    at expiration, with the closed-form approximation of Bjerksund and Stensland (2011) or
    of Kirk (1995).

    The spread of two lognormal prices has no closed-form distribution; both approximations
    work on the forward prices F1 = S1 * e^((r - q1) * T) and F2 = S2 * e^((r - q2) * T).

    Kirk (1995) treats F2 + K as one lognormal price and prices an exchange option on it:

    - Call = e^(-rT) * (F1 * N(d1) - (F2 + K) * N(d2))
    - σK² = σ1² - 2ρσ1σ2 * F2 / (F2 + K) + σ2² * (F2 / (F2 + K))²
    - d1 = (ln(F1 / (F2 + K)) + σK² * T / 2) / (σK√T), d2 = d1 - σK√T

    Bjerksund and Stensland (2011) exercise where F1(T) exceeds a boundary fitted to the
    exercise region, which gives a lower bound that is closer to the exact price than Kirk's,
    with b = F2 / (F2 + K):

    - Call = e^(-rT) * (F1 * N(d1) - F2 * N(d2) - K * N(d3))
    - σ² = σ1² - 2bρσ1σ2 + b²σ2²
    - d1 = (ln(F1 / (F2 + K)) + (σ1² / 2 - bρσ1σ2 + b²σ2² / 2) * T) / (σ√T)
    - d2 = (ln(F1 / (F2 + K)) + (-σ1² / 2 + ρσ1σ2 + b²σ2² / 2 - bσ2²) * T) / (σ√T)
    - d3 = (ln(F1 / (F2 + K)) + (-σ1² / 2 + b²σ2² / 2) * T) / (σ√T)

    The put follows from put-call parity, Put = Call - e^(-rT) * (F1 - F2 - K). Where S1 and
    S2 are the prices of the two assets, K the strike price (negative for a strike below a
    spread of zero), r the continuously compounded risk-free rate, q1 and q2 the continuous
    dividend yields (for futures, q = r makes F the futures price), σ1 and σ2 the
    volatilities, ρ the correlation of the returns, T the time to expiration in years and N
    the standard normal distribution function. At a strike of 0 both are Margrabe's exact
    exchange option. Both approximations lose accuracy as F2 + K approaches zero (a strike
    near -F2), where F2 + K is far from lognormal: Kirk's put can then be worth several
    tenths where the exact value is close to 0 (0.42 against 0.00001 by Monte Carlo at
    F2 = 96.4 and K = -96), while Bjerksund and Stensland stay close (0.0).

    Referenced against:
    - QuantLib, BjerksundStenslandSpreadEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_bjerksund_stensland_spread_engine.html
    - QuantLib, KirkEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_kirk_engine.html

    Args:
        first_stock_price (float): The price of the first asset, S1.
        second_stock_price (float): The price of the second asset, S2.
        strike_price (float): The strike price of the spread, K. It may be negative, as long
            as F2 + K stays positive.
        risk_free_rate (float): The continuously compounded risk-free rate.
        first_volatility (float): The volatility of the first asset.
        second_volatility (float): The volatility of the second asset.
        correlation (float): The correlation of the returns of the two assets, between -1
            and 1.
        time_to_expiration (float): The time to expiration in years. At 0 the option is worth
            its payoff.
        first_dividend_yield (float, optional): The continuous dividend yield of the first
            asset. Defaults to 0.
        second_dividend_yield (float, optional): The continuous dividend yield of the second
            asset. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        method (str, optional): The approximation. Options are: "bjerksund_stensland"
            (Bjerksund and Stensland, 2011, the more accurate), "kirk" (Kirk, 1995). Defaults
            to "bjerksund_stensland".

    Returns:
        float: The spread option value.

    Raises:
        TypeError: If any of the inputs is not a float or int.
        ValueError: If a volatility is not positive, the correlation is outside -1 to 1, the
            time to expiration is negative, the method is not one of the two, or F2 + K is
            not positive (both approximations take F2 + K as a lognormal price).

    Notes:
        References: Kirk, E. (1995). "Correlation in the Energy Markets." In Managing Energy
        Price Risk, Risk Publications, 71-78. Bjerksund, P., & Stensland, G. (2014). "Closed
        Form Spread Option Valuation." Quantitative Finance, 14(10), 1785-1794 (working paper
        of 2006, revised 2011). Haug, E. G. (2007). The Complete Guide to Option Pricing
        Formulas, 2nd ed., chapter 5.

    As an example:

    ```python
    from financetoolkit.options.multiasset import multi_asset_model

    # A spread of two futures prices, which grow at the risk-free rate like a dividend yield of r.
    inputs = {
        "first_stock_price": 28,
        "second_stock_price": 20,
        "strike_price": 7,
        "risk_free_rate": 0.05,
        "first_volatility": 0.29,
        "second_volatility": 0.36,
        "correlation": 0.42,
        "time_to_expiration": 0.25,
        "first_dividend_yield": 0.05,
        "second_dividend_yield": 0.05,
    }

    bjerksund_stensland = multi_asset_model.get_spread_option(**inputs)
    kirk = multi_asset_model.get_spread_option(**inputs, method="kirk")

    round(bjerksund_stensland, 4), round(kirk, 4)
    ```

    Which returns:

    ```
    (2.1684, 2.167)
    ```
    """
    validate_numeric_inputs(
        first_stock_price=first_stock_price,
        second_stock_price=second_stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        first_volatility=first_volatility,
        second_volatility=second_volatility,
        correlation=correlation,
        time_to_expiration=time_to_expiration,
        first_dividend_yield=first_dividend_yield,
        second_dividend_yield=second_dividend_yield,
    )
    _validate_two_assets(
        first_volatility, second_volatility, correlation, time_to_expiration
    )
    check_option("method", method, ("bjerksund_stensland", "kirk"))

    sign = -1 if put_option else 1

    if time_to_expiration == 0:
        return max(sign * (first_stock_price - second_stock_price - strike_price), 0.0)

    first_forward = first_stock_price * np.exp(
        (risk_free_rate - first_dividend_yield) * time_to_expiration
    )
    second_forward = second_stock_price * np.exp(
        (risk_free_rate - second_dividend_yield) * time_to_expiration
    )
    boundary = second_forward + strike_price

    if boundary <= 0:
        raise ValueError(
            "The second asset's forward price plus the strike price must be positive, "
            f"received {boundary!r}; both approximations take it as a lognormal price."
        )

    share = second_forward / boundary
    covariance = correlation * first_volatility * second_volatility
    volatility = float(
        np.sqrt(
            max(
                first_volatility**2
                - 2 * share * covariance
                + share**2 * second_volatility**2,
                0.0,
            )
        )
    )
    if volatility <= _NO_VOLATILITY:
        raise ValueError(
            "The spread has no volatility under these inputs, so the approximation has no "
            "meaning."
        )
    deviation = volatility * np.sqrt(time_to_expiration)
    log_moneyness = np.log(first_forward / boundary)
    discount_factor = np.exp(-risk_free_rate * time_to_expiration)

    if method == "kirk":
        d1 = log_moneyness / deviation + 0.5 * deviation
        call = discount_factor * (
            first_forward * ndtr(d1) - boundary * ndtr(d1 - deviation)
        )
    else:
        d1 = (
            log_moneyness
            + (
                0.5 * first_volatility**2
                - share * covariance
                + 0.5 * share**2 * second_volatility**2
            )
            * time_to_expiration
        ) / deviation
        d2 = (
            log_moneyness
            + (
                -0.5 * first_volatility**2
                + covariance
                + 0.5 * share**2 * second_volatility**2
                - share * second_volatility**2
            )
            * time_to_expiration
        ) / deviation
        d3 = (
            log_moneyness
            + (-0.5 * first_volatility**2 + 0.5 * share**2 * second_volatility**2)
            * time_to_expiration
        ) / deviation
        call = discount_factor * (
            first_forward * ndtr(d1)
            - second_forward * ndtr(d2)
            - strike_price * ndtr(d3)
        )

    if put_option:
        return float(
            call - discount_factor * (first_forward - second_forward - strike_price)
        )

    return float(call)


def get_two_asset_correlation_option(
    first_stock_price: float,
    second_stock_price: float,
    first_strike_price: float,
    second_strike_price: float,
    risk_free_rate: float,
    first_volatility: float,
    second_volatility: float,
    correlation: float,
    time_to_expiration: float,
    first_dividend_yield: float = 0.0,
    second_dividend_yield: float = 0.0,
    put_option: bool = False,
) -> float:
    """
    Calculate the price of a two-asset correlation option (Zhang, 1995): what the second
    asset's call payoff S2 - K2 is worth when it is only paid where the first asset ends
    above its own strike K1 as well, and the put payoff K2 - S2 only where both end below
    their strikes:

    - Call = S2 * e^(-q2 * T) * M(y2 + σ2√T, y1 + ρσ2√T; ρ) - K2 * e^(-rT) * M(y2, y1; ρ)
    - Put = K2 * e^(-rT) * M(-y2, -y1; ρ) - S2 * e^(-q2 * T) * M(-y2 - σ2√T, -y1 - ρσ2√T; ρ)
    - y1 = (ln(S1 / K1) + (r - q1 - σ1² / 2) * T) / (σ1√T)
    - y2 = (ln(S2 / K2) + (r - q2 - σ2² / 2) * T) / (σ2√T)

    Where S1 and S2 are the prices of the two assets, K1 and K2 their strike prices, r the
    continuously compounded risk-free rate, q1 and q2 the continuous dividend yields, σ1 and
    σ2 the volatilities, ρ the correlation of the returns, T the time to expiration in years
    and M the bivariate standard normal distribution function. y1 and y2 are the d2 terms of
    Black-Scholes, so N(y) is the risk-neutral probability that an asset ends beyond its
    strike, and the option is a vanilla option on S2 that is also knocked out by S1.

    Referenced against:
    - QuantLib, AnalyticTwoAssetCorrelationEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_two_asset_correlation_engine.html

    Args:
        first_stock_price (float): The price of the first asset, S1, that decides whether
            the option pays.
        second_stock_price (float): The price of the second asset, S2, that sets the payoff.
        first_strike_price (float): The strike price of the first asset, K1.
        second_strike_price (float): The strike price of the second asset, K2.
        risk_free_rate (float): The continuously compounded risk-free rate.
        first_volatility (float): The volatility of the first asset.
        second_volatility (float): The volatility of the second asset.
        correlation (float): The correlation of the returns of the two assets, between -1
            and 1.
        time_to_expiration (float): The time to expiration in years. At 0 the option is worth
            its payoff.
        first_dividend_yield (float, optional): The continuous dividend yield of the first
            asset. Defaults to 0.
        second_dividend_yield (float, optional): The continuous dividend yield of the second
            asset. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.

    Returns:
        float: The two-asset correlation option value.

    Raises:
        TypeError: If any of the inputs is not a float or int.
        ValueError: If a price or strike price is not positive, a volatility is not
            positive, the correlation is outside -1 to 1 or the time to expiration is
            negative.

    Notes:
        References: Zhang, P. G. (1995). "Correlation Digital Options." Journal of Financial
        Engineering, 3(1), 75-96. Haug, E. G. (2007). The Complete Guide to Option Pricing
        Formulas, 2nd ed., chapter 5.

    As an example:

    ```python
    from financetoolkit.options.multiasset import multi_asset_model

    call = multi_asset_model.get_two_asset_correlation_option(
        first_stock_price=52,
        second_stock_price=65,
        first_strike_price=50,
        second_strike_price=70,
        risk_free_rate=0.1,
        first_volatility=0.2,
        second_volatility=0.3,
        correlation=0.75,
        time_to_expiration=0.5,
    )

    round(call, 4)
    ```

    Which returns:

    ```
    4.7073
    ```
    """
    validate_numeric_inputs(
        first_stock_price=first_stock_price,
        second_stock_price=second_stock_price,
        first_strike_price=first_strike_price,
        second_strike_price=second_strike_price,
        risk_free_rate=risk_free_rate,
        first_volatility=first_volatility,
        second_volatility=second_volatility,
        correlation=correlation,
        time_to_expiration=time_to_expiration,
        first_dividend_yield=first_dividend_yield,
        second_dividend_yield=second_dividend_yield,
    )
    _validate_two_assets(
        first_volatility, second_volatility, correlation, time_to_expiration
    )
    if (
        min(
            first_stock_price,
            second_stock_price,
            first_strike_price,
            second_strike_price,
        )
        <= 0
    ):
        raise ValueError("The prices and strike prices must be positive.")

    sign = -1 if put_option else 1

    if time_to_expiration == 0:
        pays = sign * (first_stock_price - first_strike_price) > 0
        return (
            max(sign * (second_stock_price - second_strike_price), 0.0) if pays else 0.0
        )

    first_deviation = first_volatility * np.sqrt(time_to_expiration)
    second_deviation = second_volatility * np.sqrt(time_to_expiration)
    first_term = (
        np.log(first_stock_price / first_strike_price)
        + (risk_free_rate - first_dividend_yield - 0.5 * first_volatility**2)
        * time_to_expiration
    ) / first_deviation
    second_term = (
        np.log(second_stock_price / second_strike_price)
        + (risk_free_rate - second_dividend_yield - 0.5 * second_volatility**2)
        * time_to_expiration
    ) / second_deviation

    return float(
        sign
        * (
            second_stock_price
            * np.exp(-second_dividend_yield * time_to_expiration)
            * bivariate_normal_cdf(
                sign * (second_term + second_deviation),
                sign * (first_term + correlation * second_deviation),
                correlation,
            )
            - second_strike_price
            * np.exp(-risk_free_rate * time_to_expiration)
            * bivariate_normal_cdf(sign * second_term, sign * first_term, correlation)
        )
    )


def get_rainbow_option(
    first_stock_price: float,
    second_stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    first_volatility: float,
    second_volatility: float,
    correlation: float,
    time_to_expiration: float,
    first_dividend_yield: float = 0.0,
    second_dividend_yield: float = 0.0,
    option_type: Literal[
        "call_on_max", "call_on_min", "put_on_max", "put_on_min"
    ] = "call_on_max",
) -> float:
    """
    Calculate the price of a rainbow option on the better or the worse of two assets: what a
    best-of or worst-of call or put is worth, with the formulas of Stulz (1982) and Johnson
    (1987).

    A call on the maximum (best-of call) pays max(max(S1, S2) - K, 0), a call on the minimum
    (worst-of call) max(min(S1, S2) - K, 0), and the puts max(K - max(S1, S2), 0) and max(K
    - min(S1, S2), 0):

    - Call on Max = S1' * M(y1, d; ρ1) + S2' * M(y2, -d + σ√T; ρ2) - K' * (1 - M(-y1 + σ1√T, -y2 + σ2√T; ρ))
    - Call on Min = S1' * M(y1, -d; -ρ1) + S2' * M(y2, d - σ√T; -ρ2) - K' * M(y1 - σ1√T, y2 - σ2√T; ρ)
    - S1' = S1 * e^(-q1 * T), S2' = S2 * e^(-q2 * T), K' = K * e^(-rT)
    - σ = √(σ1² + σ2² - 2ρσ1σ2), ρ1 = (σ1 - ρσ2) / σ, ρ2 = (σ2 - ρσ1) / σ
    - d = (ln(S1 / S2) + (q2 - q1 + σ² / 2) * T) / (σ√T)
    - y1 = (ln(S1 / K) + (r - q1 + σ1² / 2) * T) / (σ1√T), y2 = (ln(S2 / K) + (r - q2 + σ2² / 2) * T) / (σ2√T)

    The puts follow from parity with the value of the better and the worse asset itself,
    which Margrabe's exchange option E(S1, S2) gives:

    - Put on Max = K' - S2' - E(S1, S2) + Call on Max
    - Put on Min = K' - S1' + E(S1, S2) + Call on Min

    Where S1 and S2 are the prices of the two assets, K the strike price, r the continuously
    compounded risk-free rate, q1 and q2 the continuous dividend yields, σ1 and σ2 the
    volatilities, ρ the correlation of the returns, T the time to expiration in years and M
    the bivariate standard normal distribution function.

    Referenced against:
    - QuantLib, StulzEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_stulz_engine.html

    Shares similar logic:
    - FinancePy, EquityRainbowOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_rainbow_option.py#L124

    Args:
        first_stock_price (float): The price of the first asset, S1.
        second_stock_price (float): The price of the second asset, S2.
        strike_price (float): The strike price, K.
        risk_free_rate (float): The continuously compounded risk-free rate.
        first_volatility (float): The volatility of the first asset.
        second_volatility (float): The volatility of the second asset.
        correlation (float): The correlation of the returns of the two assets, between -1
            and 1.
        time_to_expiration (float): The time to expiration in years. At 0 the option is worth
            its payoff.
        first_dividend_yield (float, optional): The continuous dividend yield of the first
            asset. Defaults to 0.
        second_dividend_yield (float, optional): The continuous dividend yield of the second
            asset. Defaults to 0.
        option_type (str, optional): The payoff. Options are: "call_on_max" (best-of call),
            "call_on_min" (worst-of call), "put_on_max" (best-of put), "put_on_min" (worst-of
            put). Defaults to "call_on_max".

    Returns:
        float: The rainbow option value.

    Raises:
        TypeError: If any of the inputs is not a float or int.
        ValueError: If a price or the strike price is not positive, a volatility is not
            positive, the correlation is outside -1 to 1, the time to expiration is negative,
            the two prices move as one or the option type is not one of the four.

    Notes:
        References: Stulz, R. M. (1982). "Options on the Minimum or the Maximum of Two Risky
        Assets: Analysis and Applications." Journal of Financial Economics, 10(2), 161-185.
        Johnson, H. (1987). "Options on the Maximum or the Minimum of Several Assets." Journal
        of Financial and Quantitative Analysis, 22(3), 277-283. Haug, E. G. (2007). The
        Complete Guide to Option Pricing Formulas, 2nd ed., chapter 5.
        - FinancePy's EquityRainbowOption reads the zero rates of its curves annually
          compounded and uses them as continuous rates, so it agrees with these values
        for rates and yields of e^r - 1 rather than r.

    As an example:

    ```python
    from financetoolkit.options.multiasset import multi_asset_model

    inputs = {
        "first_stock_price": 100,
        "second_stock_price": 105,
        "strike_price": 98,
        "risk_free_rate": 0.05,
        "first_volatility": 0.11,
        "second_volatility": 0.16,
        "correlation": 0.63,
        "time_to_expiration": 0.5,
        "first_dividend_yield": 0.05,
        "second_dividend_yield": 0.05,
    }

    {
        option_type: round(
            multi_asset_model.get_rainbow_option(**inputs, option_type=option_type), 4
        )
        for option_type in ("call_on_max", "call_on_min", "put_on_max", "put_on_min")
    }
    ```

    Which returns:

    ```
    {'call_on_max': 9.3984, 'call_on_min': 3.3563, 'put_on_max': 0.9722, 'put_on_min': 3.0048}
    ```
    """
    validate_numeric_inputs(
        first_stock_price=first_stock_price,
        second_stock_price=second_stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        first_volatility=first_volatility,
        second_volatility=second_volatility,
        correlation=correlation,
        time_to_expiration=time_to_expiration,
        first_dividend_yield=first_dividend_yield,
        second_dividend_yield=second_dividend_yield,
    )
    _validate_two_assets(
        first_volatility, second_volatility, correlation, time_to_expiration
    )
    check_option(
        "option_type",
        option_type,
        ("call_on_max", "call_on_min", "put_on_max", "put_on_min"),
    )
    if min(first_stock_price, second_stock_price, strike_price) <= 0:
        raise ValueError("The prices and the strike price must be positive.")
    volatility = _ratio_volatility(first_volatility, second_volatility, correlation)

    on_max = option_type.endswith("max")

    if time_to_expiration == 0:
        extreme = (max if on_max else min)(first_stock_price, second_stock_price)
        sign = -1 if option_type.startswith("put") else 1
        return max(sign * (extreme - strike_price), 0.0)

    root = np.sqrt(time_to_expiration)
    first_deviation = first_volatility * root
    second_deviation = second_volatility * root
    deviation = volatility * root
    first_present = first_stock_price * np.exp(
        -first_dividend_yield * time_to_expiration
    )
    second_present = second_stock_price * np.exp(
        -second_dividend_yield * time_to_expiration
    )
    discounted_strike = strike_price * np.exp(-risk_free_rate * time_to_expiration)
    first_correlation = (
        first_volatility - correlation * second_volatility
    ) / volatility
    second_correlation = (
        second_volatility - correlation * first_volatility
    ) / volatility

    d = (
        np.log(first_stock_price / second_stock_price)
        + (second_dividend_yield - first_dividend_yield) * time_to_expiration
    ) / deviation + 0.5 * deviation
    first_term = (
        np.log(first_stock_price / strike_price)
        + (risk_free_rate - first_dividend_yield) * time_to_expiration
    ) / first_deviation + 0.5 * first_deviation
    second_term = (
        np.log(second_stock_price / strike_price)
        + (risk_free_rate - second_dividend_yield) * time_to_expiration
    ) / second_deviation + 0.5 * second_deviation

    # The exchange option E(S1, S2) = S1 * e^(-q1 T) * N(d) - S2 * e^(-q2 T) * N(d - σ√T)
    # values max(S1 - S2, 0), so max(S1, S2) is worth S2 * e^(-q2 T) + E and min(S1, S2)
    # is worth S1 * e^(-q1 T) - E.
    exchange = first_present * ndtr(d) - second_present * ndtr(d - deviation)

    if on_max:
        call = (
            first_present * bivariate_normal_cdf(first_term, d, first_correlation)
            + second_present
            * bivariate_normal_cdf(second_term, deviation - d, second_correlation)
            - discounted_strike
            * (
                1
                - bivariate_normal_cdf(
                    first_deviation - first_term,
                    second_deviation - second_term,
                    correlation,
                )
            )
        )
        extreme_value = second_present + exchange
    else:
        call = (
            first_present * bivariate_normal_cdf(first_term, -d, -first_correlation)
            + second_present
            * bivariate_normal_cdf(second_term, d - deviation, -second_correlation)
            - discounted_strike
            * bivariate_normal_cdf(
                first_term - first_deviation,
                second_term - second_deviation,
                correlation,
            )
        )
        extreme_value = first_present - exchange

    if option_type.startswith("put"):
        return float(discounted_strike - extreme_value + call)

    return float(call)


def _validate_basket(
    stock_prices: np.ndarray,
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    dividend_yields: np.ndarray,
    time_to_expiration: float,
) -> None:
    """
    Raises a ValueError for basket inputs of different lengths, prices or volatilities that are
    not positive, a correlation matrix that is not a symmetric matrix of correlations with ones
    on its diagonal, or a negative time to expiration.
    """
    assets = stock_prices.size
    if assets == 0:
        raise ValueError("The basket needs at least one asset.")
    if not weights.size == volatilities.size == dividend_yields.size == assets:
        raise ValueError(
            "stock_prices, weights, volatilities and dividend_yields must have the same "
            "length."
        )
    if correlation_matrix.shape != (assets, assets):
        raise ValueError(
            f"correlation_matrix must be {assets} by {assets}, received the shape "
            f"{correlation_matrix.shape}."
        )
    if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-12):
        raise ValueError("correlation_matrix must be symmetric.")
    if not np.allclose(np.diag(correlation_matrix), 1.0, atol=1e-12):
        raise ValueError("correlation_matrix must have ones on its diagonal.")
    if np.abs(correlation_matrix).max() > 1 + 1e-12:
        raise ValueError("correlation_matrix must hold correlations between -1 and 1.")
    if (stock_prices <= 0).any() or (volatilities <= 0).any():
        raise ValueError("The stock prices and volatilities must be positive.")
    if not np.isfinite(
        np.concatenate([stock_prices, weights, volatilities, dividend_yields])
    ).all():
        raise ValueError("The basket inputs must be finite numbers.")
    if time_to_expiration < 0:
        raise ValueError(
            f"time_to_expiration must be at least 0, received {time_to_expiration!r}."
        )


def _basket_inputs(
    stock_prices,
    weights,
    volatilities,
    correlation_matrix,
    dividend_yields,
    risk_free_rate: float,
    time_to_expiration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validates the basket inputs and returns the forward prices, the weights, the volatilities
    and the correlation matrix as arrays.
    """
    stock_prices = np.atleast_1d(np.asarray(stock_prices, dtype=float))
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    volatilities = np.atleast_1d(np.asarray(volatilities, dtype=float))
    correlation_matrix = np.atleast_2d(np.asarray(correlation_matrix, dtype=float))
    dividend_yields = (
        np.zeros(stock_prices.size)
        if dividend_yields is None
        else np.atleast_1d(np.asarray(dividend_yields, dtype=float))
    )
    validate_numeric_inputs(
        risk_free_rate=risk_free_rate, time_to_expiration=time_to_expiration
    )
    _validate_basket(
        stock_prices,
        weights,
        volatilities,
        correlation_matrix,
        dividend_yields,
        time_to_expiration,
    )
    forwards = stock_prices * np.exp(
        (risk_free_rate - dividend_yields) * time_to_expiration
    )

    return forwards, weights, volatilities, correlation_matrix


def _lognormal_call(
    forward: float, strike_price: float, total_deviation: float
) -> float:
    """
    The undiscounted Black value E[max(X - K, 0)] of a lognormal X with mean forward and a
    standard deviation of ln X of total_deviation, max(forward - K, 0) without deviation.
    """
    if strike_price <= 0:
        return forward - strike_price
    if total_deviation <= 0:
        return max(forward - strike_price, 0.0)

    d1 = np.log(forward / strike_price) / total_deviation + 0.5 * total_deviation

    return float(forward * ndtr(d1) - strike_price * ndtr(d1 - total_deviation))


def _geometric_basket(
    forwards: np.ndarray,
    weights: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    time_to_expiration: float,
) -> tuple[np.ndarray, float, float]:
    """
    The value shares a_i = w_i * F_i / Σ w_j * F_j of the basket and the mean and the standard
    deviation of ln G, with G = Π (S_i(T) / F_i)^a_i the geometric basket of Gentle (1993):

    - E[ln G] = -Σ a_i * σ_i² * T / 2
    - Var[ln G] = Σ_i Σ_j a_i * a_j * ρ_ij * σ_i * σ_j * T
    """
    shares = weights * forwards / np.sum(weights * forwards)
    covariance = (
        correlation_matrix * np.outer(volatilities, volatilities) * time_to_expiration
    )
    log_mean = float(-0.5 * np.sum(shares * volatilities**2) * time_to_expiration)
    log_deviation = float(np.sqrt(max(shares @ covariance @ shares, 0.0)))

    return shares, log_mean, log_deviation


def get_basket_option(
    stock_prices: np.ndarray | list[float],
    weights: np.ndarray | list[float],
    strike_price: float,
    risk_free_rate: float,
    volatilities: np.ndarray | list[float],
    correlation_matrix: np.ndarray | list[list[float]],
    time_to_expiration: float,
    dividend_yields: np.ndarray | list[float] | None = None,
    put_option: bool = False,
    method: Literal["levy", "gentle"] = "levy",
) -> float:
    """
    Calculate the closed-form approximation of a European basket option: what an option on a
    weighted sum of lognormal stock prices is worth, max(B - K, 0) for a call and max(K - B,
    0) for a put on the basket B = Σ w_i * S_i(T) at expiration.

    The sum of lognormal prices has no closed-form distribution, so it is approximated by a
    lognormal one.

    Levy (1992) matches the first two moments of the basket at expiration:

    - M1 = Σ w_i * F_i, M2 = Σ_i Σ_j w_i * w_j * F_i * F_j * e^(ρ_ij * σ_i * σ_j * T)
    - v² = ln(M2 / M1²)
    - Call = e^(-rT) * (M1 * N(d1) - K * N(d1 - v)), d1 = (ln(M1 / K) + v² / 2) / v

    Gentle (1993) replaces the arithmetic basket by the geometric one, G = Π (S_i(T) / F_i)^a_i
    with the value shares a_i = w_i * F_i / M1, which is lognormal, and moves the strike by the
    difference of their means:

    - E[ln G] = -Σ a_i * σ_i² * T / 2, vG² = Σ_i Σ_j a_i * a_j * ρ_ij * σ_i * σ_j * T
    - K* = K - M1 * (1 - E[G]), with E[G] = e^(E[ln G] + vG² / 2)
    - Call = e^(-rT) * (M1 * E[G] * N(d1) - K* * N(d1 - vG)), d1 = (ln(M1 * E[G] / K*) + vG² / 2) / vG

    The put follows from put-call parity, Put = Call - e^(-rT) * (M1 - K), for both. Where
    S_i, w_i, σ_i and q_i are the price, weight, volatility and continuous dividend yield of
    asset i, F_i = S_i * e^((r - q_i) * T) its forward price, ρ_ij the correlation of the
    returns of assets i and j, K the strike price, r the continuously compounded risk-free
    rate, T the time to expiration in years and N the standard normal distribution
    function. A strike (or Gentle's moved strike) of 0 or below leaves a call that is always
    exercised, worth e^(-rT) * (M1 - K).

    Both approximations are close for low volatilities and short expirations. As the variance
    of the basket grows, Levy's lognormal is off by up to about 2% of the price (at
    volatilities of 30% to 60% over two years) and Gentle's geometric basket underprices calls
    by up to about 12%, so Levy is the default. Monte Carlo prices of the same basket, which the
    approximations can be checked against, come from get_monte_carlo_basket_option.

    Referenced against:
    - QuantLib, ChoiBasketEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_choi_basket_engine.html

    Shares similar logic:
    - QuantLib, MCEuropeanBasketEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_european_basket_engine.html
    - FinancePy, EquityBasketOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_basket_option.py#L30

    Args:
        stock_prices (np.ndarray | list[float]): The prices of the assets in the basket.
        weights (np.ndarray | list[float]): The number of units of each asset in the basket.
            For Levy the basket's forward value must be positive; for Gentle every weight must
            be at least 0.
        strike_price (float): The strike price of the basket, K.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatilities (np.ndarray | list[float]): The volatility of each asset.
        correlation_matrix (np.ndarray | list[list[float]]): The correlations of the returns
            of the assets, a symmetric matrix with ones on its diagonal.
        time_to_expiration (float): The time to expiration in years. At 0 the option is worth
            its payoff.
        dividend_yields (np.ndarray | list[float] | None, optional): The continuous dividend
            yield of each asset. Defaults to None, which uses 0 for every asset.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        method (str, optional): The approximation. Options are: "levy" (Levy, 1992, moment
            matching), "gentle" (Gentle, 1993, geometric basket). Defaults to "levy".

    Returns:
        float: The basket option value.

    Raises:
        TypeError: If the risk-free rate, strike price or time to expiration is not a float or
            int.
        ValueError: If the inputs have different lengths, a price or volatility is not
            positive, the correlation matrix is not a symmetric matrix of correlations with
            ones on its diagonal, the time to expiration is negative, the method is not one of
            the two, the basket's forward value is not positive (Levy) or a weight is negative
            (Gentle).

    Notes:
        References: Levy, E. (1992). "Pricing European Average Rate Currency Options."
        Journal of International Money and Finance, 11(5), 474-491. Gentle, D. (1993).
        "Basket Weaving." Risk, 6(6), 51-52. Haug, E. G. (2007). The Complete Guide to Option
        Pricing Formulas, 2nd ed.
        - FinancePy's EquityBasketOption matches the same two moments as Levy; QuantLib's
          ChoiBasketEngine and MCEuropeanBasketEngine show the accuracy of the
          approximations.

    As an example:

    ```python
    from financetoolkit.options.multiasset import multi_asset_model

    inputs = {
        "stock_prices": [100, 100, 100],
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "strike_price": 100,
        "risk_free_rate": 0.05,
        "volatilities": [0.2, 0.3, 0.4],
        "correlation_matrix": [[1, 0.5, 0.3], [0.5, 1, 0.6], [0.3, 0.6, 1]],
        "time_to_expiration": 1,
    }

    levy = multi_asset_model.get_basket_option(**inputs)
    gentle = multi_asset_model.get_basket_option(**inputs, method="gentle")

    round(levy, 4), round(gentle, 4)
    ```

    Which returns:

    ```
    (12.2315, 11.9894)
    ```
    """
    validate_numeric_inputs(strike_price=strike_price)
    check_option("method", method, ("levy", "gentle"))
    forwards, weights, volatilities, correlation_matrix = _basket_inputs(
        stock_prices,
        weights,
        volatilities,
        correlation_matrix,
        dividend_yields,
        risk_free_rate,
        time_to_expiration,
    )
    sign = -1 if put_option else 1

    if time_to_expiration == 0:
        return max(sign * (float(np.sum(weights * forwards)) - strike_price), 0.0)

    first_moment = float(np.sum(weights * forwards))
    discount_factor = float(np.exp(-risk_free_rate * time_to_expiration))

    if first_moment <= 0:
        raise ValueError(
            "The basket's forward value must be positive for a lognormal approximation, "
            f"received {first_moment!r}."
        )

    if method == "levy":
        weighted = weights * forwards
        second_moment = float(
            weighted
            @ np.exp(
                correlation_matrix
                * np.outer(volatilities, volatilities)
                * time_to_expiration
            )
            @ weighted
        )
        total_deviation = float(
            np.sqrt(max(np.log(second_moment / first_moment**2), 0.0))
        )
        call = discount_factor * _lognormal_call(
            first_moment, float(strike_price), total_deviation
        )
    else:
        if (weights < 0).any():
            raise ValueError(
                'method="gentle" needs weights of at least 0, as the geometric basket has '
                "no meaning for a short position."
            )
        _, log_mean, log_deviation = _geometric_basket(
            forwards, weights, volatilities, correlation_matrix, time_to_expiration
        )
        geometric_mean = first_moment * np.exp(log_mean + 0.5 * log_deviation**2)
        moved_strike = float(strike_price) - (first_moment - geometric_mean)
        call = discount_factor * _lognormal_call(
            float(geometric_mean), moved_strike, log_deviation
        )

    if put_option:
        return float(call - discount_factor * (first_moment - strike_price))

    return float(call)


def get_monte_carlo_basket_option(
    stock_prices: np.ndarray | list[float],
    weights: np.ndarray | list[float],
    strike_price: float,
    risk_free_rate: float,
    volatilities: np.ndarray | list[float],
    correlation_matrix: np.ndarray | list[list[float]],
    time_to_expiration: float,
    dividend_yields: np.ndarray | list[float] | None = None,
    put_option: bool = False,
    simulations: int = 100_000,
    seed: int | None = 0,
) -> tuple[float, float]:
    """
    Calculate the Monte Carlo price of a European basket option: what an option on a
    weighted sum of correlated lognormal stock prices is worth, max(B - K, 0) for a call and
    max(K - B, 0) for a put on the basket B = Σ w_i * S_i(T) at expiration, with the
    geometric basket of Gentle (1993) as control variate.

    The prices at expiration are drawn in one step, correlated through the Cholesky factor L
    of the correlation matrix (L * Lᵀ = ρ). The geometric basket option on the same draws has
    a closed-form price and moves closely with the basket option, so its simulation error is
    subtracted:

    - S_i(T) = F_i * e^(-σ_i² * T / 2 + σ_i√T * X_i), X = L * Z with Z independent standard normal draws
    - Y = e^(-rT) * max(φ * (B - K), 0), X_G = e^(-rT) * max(φ * (M1 * G - K), 0)
    - G = Π (S_i(T) / F_i)^a_i, a_i = w_i * F_i / M1, M1 = Σ w_i * F_i
    - Price = MEAN(Y) - β * (MEAN(X_G) - E[X_G]), β = Cov(Y, X_G) / Var(X_G)
    - Standard Error = √(Var(Y - β * X_G) / Simulations)

    Where F_i = S_i * e^((r - q_i) * T) is the forward price of asset i, σ_i its volatility,
    q_i its continuous dividend yield, w_i its weight, K the strike price, r the continuously
    compounded risk-free rate, T the time to expiration in years, φ = 1 for a call and -1 for
    a put, and E[X_G] the Black value of the lognormal M1 * G. Var and Cov use ddof=1. The
    control variate needs weights of at least 0; with a negative weight the plain average
    of Y is returned.

    The draws come from numpy's default generator (PCG64) seeded with seed, so a seed gives
    the same price on every platform up to the last digits of floating point arithmetic.

    Referenced against:
    - QuantLib, ChoiBasketEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_choi_basket_engine.html

    Shares similar logic:
    - QuantLib, MCEuropeanBasketEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_european_basket_engine.html
    - FinancePy, EquityBasketOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_basket_option.py#L30

    Args:
        stock_prices (np.ndarray | list[float]): The prices of the assets in the basket.
        weights (np.ndarray | list[float]): The number of units of each asset in the basket.
        strike_price (float): The strike price of the basket, K.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatilities (np.ndarray | list[float]): The volatility of each asset.
        correlation_matrix (np.ndarray | list[list[float]]): The correlations of the returns
            of the assets, a symmetric and positive definite matrix with ones on its diagonal.
        time_to_expiration (float): The time to expiration in years. At 0 the option is worth
            its payoff with a standard error of 0.
        dividend_yields (np.ndarray | list[float] | None, optional): The continuous dividend
            yield of each asset. Defaults to None, which uses 0 for every asset.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        simulations (int, optional): The number of simulated baskets. Defaults to 100,000.
        seed (int | None, optional): The seed of the random number generator. Defaults to 0;
            None draws different prices on every call.

    Returns:
        tuple[float, float]: The price and its standard error.

    Raises:
        TypeError: If the risk-free rate, strike price, time to expiration or number of
            simulations is not a number.
        ValueError: If the inputs are not a valid basket (see get_basket_option), the
            correlation matrix is not positive definite, simulations is below 2 or
            simulations times the number of assets exceeds MAX_SIMULATED_VALUES.

    Notes:
        References: Boyle, P., Broadie, M., & Glasserman, P. (1997). "Monte Carlo Methods for
        Security Pricing." Journal of Economic Dynamics and Control, 21(8-9), 1267-1321.
        Gentle, D. (1993). "Basket Weaving." Risk, 6(6), 51-52.
        - QuantLib's MCEuropeanBasketEngine and FinancePy's EquityBasketOption draw their own
          random numbers, which give other paths.

    As an example:

    ```python
    from financetoolkit.options.multiasset import multi_asset_model

    price, standard_error = multi_asset_model.get_monte_carlo_basket_option(
        stock_prices=[100, 100, 100],
        weights=[1 / 3, 1 / 3, 1 / 3],
        strike_price=100,
        risk_free_rate=0.05,
        volatilities=[0.2, 0.3, 0.4],
        correlation_matrix=[[1, 0.5, 0.3], [0.5, 1, 0.6], [0.3, 0.6, 1]],
        time_to_expiration=1,
        seed=42,
    )

    round(price, 2), round(standard_error, 3)
    ```

    Which returns:

    ```
    (12.14, 0.006)
    ```
    """
    validate_numeric_inputs(strike_price=strike_price, simulations=simulations)
    forwards, weights, volatilities, correlation_matrix = _basket_inputs(
        stock_prices,
        weights,
        volatilities,
        correlation_matrix,
        dividend_yields,
        risk_free_rate,
        time_to_expiration,
    )
    if simulations <= 1:
        raise ValueError(f"simulations must be at least 2, received {simulations!r}.")
    if simulations * forwards.size > MAX_SIMULATED_VALUES:
        raise ValueError(
            f"{simulations:,} simulations of {forwards.size:,} assets is too large; keep "
            f"simulations times assets under {MAX_SIMULATED_VALUES:,}."
        )

    sign = -1 if put_option else 1

    if time_to_expiration == 0:
        return max(sign * (float(np.sum(weights * forwards)) - strike_price), 0.0), 0.0

    # The Cholesky factor is unique, unlike an eigenvector decomposition whose signs depend
    # on the linear algebra library, so a seed gives the same draws on every platform.
    try:
        cholesky = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("correlation_matrix must be positive definite.") from error

    discount_factor = float(np.exp(-risk_free_rate * time_to_expiration))
    deviations = volatilities * np.sqrt(time_to_expiration)
    random_number_generator = np.random.default_rng(seed)
    baskets = np.empty(simulations)
    log_geometric = np.empty(simulations)
    first_moment = float(np.sum(weights * forwards))
    controlled = first_moment > 0 and not (weights < 0).any()
    shares, log_mean, log_deviation = (
        _geometric_basket(
            forwards, weights, volatilities, correlation_matrix, time_to_expiration
        )
        if controlled
        else (np.zeros(forwards.size), 0.0, 0.0)
    )

    # The generator fills the draws row by row, so drawing them in chunks of rows gives the
    # same baskets as one draw while holding only a chunk in memory.
    for first in range(0, simulations, SIMULATION_CHUNK_SIZE):
        rows = min(SIMULATION_CHUNK_SIZE, simulations - first)
        log_returns = (
            random_number_generator.standard_normal((rows, forwards.size)) @ cholesky.T
        ) * deviations - 0.5 * deviations**2
        baskets[first : first + rows] = np.exp(log_returns) @ (weights * forwards)
        log_geometric[first : first + rows] = log_returns @ shares

    payoffs = discount_factor * np.maximum(sign * (baskets - strike_price), 0.0)

    if controlled:
        geometric = discount_factor * np.maximum(
            sign * (first_moment * np.exp(log_geometric) - strike_price), 0.0
        )
        geometric_mean = first_moment * np.exp(log_mean + 0.5 * log_deviation**2)
        geometric_call = discount_factor * _lognormal_call(
            float(geometric_mean), float(strike_price), log_deviation
        )
        geometric_price = (
            geometric_call - discount_factor * (geometric_mean - strike_price)
            if put_option
            else geometric_call
        )
        geometric_variance = geometric.var(ddof=1)
        # Without variation in the control (every basket out of the money) there is
        # nothing to correct and the estimate is the plain average.
        beta = (
            np.cov(payoffs, geometric, ddof=1)[0, 1] / geometric_variance
            if geometric_variance > 0
            else 0.0
        )
        payoffs = payoffs - beta * (geometric - geometric_price)

    return float(payoffs.mean()), float(payoffs.std(ddof=1) / np.sqrt(simulations))
