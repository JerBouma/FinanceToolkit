"""Derivative Models

Interest rate options on a forward rate: European swaptions (physically and cash settled),
caps, floors and collars, year-on-year inflation caps and floors, their Greeks and implied
volatilities, with Black-76 (lognormal), shifted Black (displaced lognormal) and the Bachelier
(normal) model. Bond options are in the short-rate models.

Times are year fractions from today. Rates are simple rates on their accrual periods (the
forward rate of a caplet) or the par rate of a swap paid `payment_frequency` times a year;
discounting is continuous at a flat risk-free rate or on the annually compounded zero rates of
a curve (see `financetoolkit.fixedincome.helpers.discount_factors`).
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.fixedincome.inflation import inflation_swap_model
from financetoolkit.utilities.error_model import check_option

# The standard normal distribution, computed as scipy.stats.norm computes it, without
# importing scipy.stats, which takes around a second and is only needed elsewhere.
NORMAL_DENSITY_SCALE = np.sqrt(2 * np.pi)


def _normal_cdf(value):
    """
    The standard normal cumulative distribution function, as scipy.stats.norm.cdf.

    Args:
        value (float | np.ndarray): The value(s).

    Returns:
        float | np.ndarray: The probability of a value at most this large.
    """
    return ndtr(value)


def _normal_pdf(value):
    """
    The standard normal probability density function, as scipy.stats.norm.pdf.

    Args:
        value (float | np.ndarray): The value(s).

    Returns:
        float | np.ndarray: The density.
    """
    return np.exp(-(value**2) / 2.0) / NORMAL_DENSITY_SCALE


def _black_formula(
    forward_rate,
    strike_rate,
    volatility,
    years_to_expiry,
    is_call,
    shift: float = 0.0,
):
    """
    The undiscounted Black-76 value of a call or put on a forward rate, displaced by a shift
    (Black, 1976; Rubinstein, 1983).

    - Call = F N(d_1) - K N(d_2), Put = K N(-d_2) - F N(-d_1)
    - d_1 = (ln(F / K) + σ² t / 2) / (σ √t), d_2 = d_1 - σ √t

    with F and K the forward and strike rate plus the shift. Without volatility or time
    left the value is the intrinsic value. Works element-wise on arrays.

    Args:
        forward_rate (float | np.ndarray): The forward rate(s).
        strike_rate (float | np.ndarray): The strike rate(s).
        volatility (float | np.ndarray): The lognormal volatility (of the shifted rate).
        years_to_expiry (float | np.ndarray): The years until the rate is fixed.
        is_call (bool | np.ndarray): Whether the option pays F - K (a caplet or payer
            swaption) rather than K - F.
        shift (float): The displacement added to the forward and strike rate. Defaults to 0.

    Returns:
        float | np.ndarray: The value per unit of annuity (discount factor times accrual).
    """
    forward = np.asarray(forward_rate, dtype=float) + shift
    strike = np.asarray(strike_rate, dtype=float) + shift
    deviation = np.asarray(volatility, dtype=float) * np.sqrt(
        np.maximum(np.asarray(years_to_expiry, dtype=float), 0.0)
    )
    sign = np.where(is_call, 1.0, -1.0)
    intrinsic = np.maximum(sign * (forward - strike), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(forward / strike) + 0.5 * deviation**2) / deviation
        d2 = d1 - deviation
        value = sign * (
            forward * _normal_cdf(sign * d1) - strike * _normal_cdf(sign * d2)
        )

    value = np.where(deviation > 0, value, intrinsic)
    return float(value) if value.ndim == 0 else value


def _bachelier_formula(forward_rate, strike_rate, volatility, years_to_expiry, is_call):
    """
    The undiscounted Bachelier (normal) value of a call or put on a forward rate
    (Bachelier, 1900).

    - Call = (F - K) N(d) + σ √t φ(d), Put = (K - F) N(-d) + σ √t φ(d)
    - d = (F - K) / (σ √t)

    Without volatility or time left the value is the intrinsic value. Works element-wise on
    arrays.

    Args:
        forward_rate (float | np.ndarray): The forward rate(s).
        strike_rate (float | np.ndarray): The strike rate(s).
        volatility (float | np.ndarray): The normal (absolute) volatility.
        years_to_expiry (float | np.ndarray): The years until the rate is fixed.
        is_call (bool | np.ndarray): Whether the option pays F - K rather than K - F.

    Returns:
        float | np.ndarray: The value per unit of annuity (discount factor times accrual).
    """
    deviation = np.asarray(volatility, dtype=float) * np.sqrt(
        np.maximum(np.asarray(years_to_expiry, dtype=float), 0.0)
    )
    gap = np.where(is_call, 1.0, -1.0) * (
        np.asarray(forward_rate, dtype=float) - np.asarray(strike_rate, dtype=float)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        d = gap / deviation
        value = gap * _normal_cdf(d) + deviation * _normal_pdf(d)

    value = np.where(deviation > 0, value, np.maximum(gap, 0.0))
    return float(value) if value.ndim == 0 else value


MODELS = ("black", "bachelier")


def _option_value(
    method: str,
    forward_rate,
    strike_rate,
    volatility,
    years_to_expiry,
    is_call,
    shift: float = 0.0,
):
    """
    The undiscounted value of an option on a forward rate in the Black (shifted when the
    shift is not 0) or the Bachelier model, see _black_formula and _bachelier_formula.

    Args:
        method (str): "black" or "bachelier".
        forward_rate (float | np.ndarray): The forward rate(s).
        strike_rate (float | np.ndarray): The strike rate(s).
        volatility (float | np.ndarray): The volatility, lognormal or normal.
        years_to_expiry (float | np.ndarray): The years until the rate is fixed.
        is_call (bool | np.ndarray): Whether the option pays F - K rather than K - F.
        shift (float): The displacement of the Black model. Defaults to 0.

    Returns:
        float | np.ndarray: The value per unit of annuity.
    """
    if method == "black":
        return _black_formula(
            forward_rate, strike_rate, volatility, years_to_expiry, is_call, shift
        )
    return _bachelier_formula(
        forward_rate, strike_rate, volatility, years_to_expiry, is_call
    )


def _check_method(method: str, forward_rate, strike_rate, shift: float) -> None:
    """
    Raise a clear error for an unknown model, or for Black's model on a shifted forward or
    strike rate that is not positive, where its logarithm is undefined.

    Args:
        method (str): "black" or "bachelier".
        forward_rate (float | np.ndarray): The forward rate(s).
        strike_rate (float | np.ndarray): The strike rate(s).
        shift (float): The displacement of the Black model.
    """
    check_option("method", method, MODELS)
    if method == "black" and (
        np.any(np.asarray(forward_rate, dtype=float) + shift <= 0)
        or np.any(np.asarray(strike_rate, dtype=float) + shift <= 0)
    ):
        raise ValueError(
            "Black's model assumes a lognormally distributed (shifted) rate and is "
            "undefined when the forward or strike rate plus the shift "
            f"({shift}) is not positive. Raise the shift or use the Bachelier model, "
            "which admits zero and negative rates."
        )


VOLATILITY_TYPES = ("lognormal", "normal")

SETTLEMENTS = ("physical", "cash")


def _convert_volatility(
    volatility: float,
    volatility_type: str,
    target_type: str,
    forward_rate: float,
) -> float:
    """
    Convert a volatility quote between the lognormal and normal conventions.

    Black-76 and Bachelier do not speak the same language about volatility, and the two
    numbers differ by roughly two orders of magnitude at the rate levels typical of the
    interest rate market. Black's model assumes the forward rate is lognormally
    distributed, so its volatility is a **relative** (percentage-of-rate) quantity: a
    20% lognormal volatility on a 3.25% forward means the rate moves by about 20% of
    3.25%. Bachelier assumes the forward rate is normally distributed, so its volatility
    is an **absolute** quantity quoted in rate units, most often basis points: the same
    market is quoted as roughly 65 basis points of normal volatility. Feeding one number
    into the model expecting the other silently misprices the option by a factor of
    approximately 1 / forward_rate.

    The two conventions are reconciled at the money by matching the two models' at-the-
    money prices, which to first order gives:

        sigma_normal ≈ sigma_lognormal * forward_rate

    This is an approximation, exact only in the limit of small volatility and only at the
    money — away from the money the smile is not preserved by it. It exists so that a
    volatility quoted on one basis can be used with the other model deliberately and
    visibly, not so that the distinction can be ignored.

    Args:
        volatility (float): The volatility as quoted, on the `volatility_type` basis.
        volatility_type (str): The convention `volatility` is quoted on, either
            'lognormal' (relative, a percentage of the forward rate) or 'normal'
            (absolute, in the same units as the forward rate).
        target_type (str): The convention the model consuming the volatility requires,
            either 'lognormal' or 'normal'.
        forward_rate (float): The forward rate the conversion is centred on.

    Returns:
        float: The volatility restated on the `target_type` basis.

    Raises:
        ValueError: If volatility_type or target_type is not 'lognormal' or 'normal', or
            if a conversion is requested around a forward rate of zero.
    """
    for name, value in (
        ("volatility_type", volatility_type),
        ("target_type", target_type),
    ):
        if value not in VOLATILITY_TYPES:
            raise ValueError(
                f"Expected {name} to be one of {VOLATILITY_TYPES}, received "
                f"'{value}' instead."
            )

    if volatility_type == target_type:
        return volatility

    if forward_rate == 0:
        raise ValueError(
            "A volatility cannot be converted between the lognormal and normal "
            "conventions around a forward rate of zero. Quote the volatility on the "
            f"'{target_type}' basis directly instead."
        )

    if target_type == "normal":
        return volatility * forward_rate

    return volatility / forward_rate


def _get_annuity_factor(
    risk_free_rate: float | None,
    years_to_maturity: float,
    tenor: float,
    payment_frequency: int,
    discount_curve: pd.Series | None = None,
    settlement: str = "physical",
    forward_rate: float | None = None,
) -> float:
    """
    Calculate the annuity factor (present value of a basis point, PVBP) of the swap
    underlying a swaption.

    Exercising a swaption does not pay out a single lump sum at expiration — it grants
    the right to enter a swap that exchanges cash flows at every payment date over the
    swap's tenor. The annuity factor is the sum of the discount factors to each of
    those payment dates, weighted by the accrual fraction between them, and is what
    converts a per-period rate differential into the present value of the full stream
    of swap cash flows.

    The formula is as follows:

        Annuity = SUM_(k=1)^(n) [ accrual * DiscountFactor(years_to_maturity + k * accrual) ]

    Where accrual = 1 / payment_frequency and n = tenor * payment_frequency, rounded up, is
    the number of payments made over the life of the underlying swap. A tenor that is not a
    whole number of periods ends with a short period: its last payment falls at the end of the
    tenor and accrues tenor - (n - 1) * accrual. With a discount curve, the discount factors
    are those of its annually compounded zero rates, DiscountFactor(t) = (1 + r(t))^(-t) with
    r(t) interpolated linearly between its maturities (as `swap_model.get_swap_rate` discounts
    the fixed leg); without one they are continuously compounded on a flat curve,
    DiscountFactor(t) = exp(-risk_free_rate * t).

    A cash-settled swaption pays the value of the swap at expiry computed with the cash
    annuity: its payments discounted at the swap rate itself, the par yield, rather than on
    the curve (the ISDA par yield curve method of EUR swaptions). The cash annuity is
    discounted to today from the expiry:

        Cash Annuity = DiscountFactor(years_to_maturity)
                       * SUM_(k=1)^(n) [ accrual_k / PRODUCT_(j=1)^(k) (1 + accrual_j * S) ]

    with S the forward swap rate (Mercurio, 2008). Treating it as the annuity of a
    physically settled swaption is the market's annuity mapping, which is not arbitrage
    free but is how cash-settled swaptions are quoted.

    Args:
        risk_free_rate (float | None): The risk-free interest rate used to discount each
        payment date, continuously compounded and assumed flat across the curve. Not used
        when a discount curve is given.
        years_to_maturity (float): Years to expiration of the swaption, i.e. the time
        at which the underlying swap would start if exercised.
        tenor (float): The tenor (length in years) of the underlying swap.
        payment_frequency (int): The number of fixed-leg payments per year on the
        underlying swap (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly).
        discount_curve (pd.Series | None): The annually compounded zero-coupon rates by
        maturity in years to discount with, such as a curve of `get_zero_coupon_curve` or
        `get_eiopa_risk_free_rate`. Defaults to None, which discounts at the flat risk-free rate.
        settlement (str): "physical" for the annuity on the curve or "cash" for the cash
        annuity at the forward swap rate. Defaults to "physical".
        forward_rate (float | None): The forward swap rate the cash annuity discounts at.
        Defaults to None, which is only allowed for a physically settled swaption.

    Returns:
        float: The annuity factor (PVBP) of the underlying swap.

    Raises:
        ValueError: If neither a risk-free rate nor a discount curve is given, or the
            settlement is unknown.
    """
    check_option("settlement", settlement, SETTLEMENTS)
    # A whole number of periods in floating point noise (2.9999999999999996) is not a short period.  # noqa: E501
    periods = tenor * payment_frequency
    number_of_payments = max(
        round(periods) if math.isclose(periods, round(periods)) else math.ceil(periods),
        1,
    )
    accrual = 1 / payment_frequency

    # Every period is a full one except the last, which ends at the end of the tenor.
    accruals = np.full(number_of_payments, accrual)
    accruals[-1] = tenor - (number_of_payments - 1) * accrual
    payment_times = years_to_maturity + np.cumsum(accruals)

    if discount_curve is None and risk_free_rate is None:
        raise ValueError(
            "A swaption's annuity is discounted with a risk-free rate or a discount curve; "
            "give either one."
        )

    flat_rate = 0.0 if risk_free_rate is None else risk_free_rate

    def discount(times: np.ndarray) -> np.ndarray:
        if discount_curve is not None:
            return discount_factors(discount_curve, times)
        return np.exp(-flat_rate * times)

    if settlement == "cash":
        if forward_rate is None:
            raise ValueError("The cash annuity needs the forward swap rate.")
        growth = np.cumprod(1 + accruals * forward_rate)
        return float(
            discount(np.array([years_to_maturity]))[0] * np.sum(accruals / growth)
        )

    return float(np.sum(accruals * discount(payment_times)))


def _swaption_terms(
    method: str,
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float | None,
    tenor: float | None,
    payment_frequency: int,
    volatility_type: str,
    discount_curve: pd.Series | None,
    shift: float = 0.0,
    settlement: str = "physical",
) -> tuple[float, float]:
    """
    The volatility on the basis of the model and the annuity of a swaption, after checking
    that the model is defined for its rates.

    Args:
        method (str): "black" or "bachelier".
        forward_rate (float): The forward swap rate.
        strike_rate (float): The strike rate.
        volatility (float): The volatility as quoted.
        years_to_maturity (float): The years until the swaption expires.
        risk_free_rate (float | None): The flat, continuously compounded risk-free rate.
        tenor (float | None): The tenor of the swap, None for the years to maturity.
        payment_frequency (int): The fixed payments per year.
        volatility_type (str): "lognormal" or "normal", the basis of the quote.
        discount_curve (pd.Series | None): The annually compounded zero-coupon rates.
        shift (float): The displacement of the Black model. Defaults to 0.
        settlement (str): "physical" or "cash". Defaults to "physical".

    Returns:
        tuple[float, float]: The volatility on the model's basis and the annuity.
    """
    _check_method(method, forward_rate, strike_rate, shift)
    volatility = _convert_volatility(
        volatility=volatility,
        volatility_type=volatility_type,
        target_type="lognormal" if method == "black" else "normal",
        forward_rate=forward_rate + shift if method == "black" else forward_rate,
    )
    annuity = _get_annuity_factor(
        risk_free_rate=risk_free_rate,
        years_to_maturity=years_to_maturity,
        tenor=years_to_maturity if tenor is None else tenor,
        payment_frequency=payment_frequency,
        discount_curve=discount_curve,
        settlement=settlement,
        forward_rate=forward_rate,
    )

    return volatility, annuity


def get_black_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float | None = None,
    notional: float = 10_000_000,
    tenor: float | None = None,
    payment_frequency: int = 2,
    is_receiver: bool = True,
    volatility_type: str = "lognormal",
    discount_curve: pd.Series | None = None,
    shift: float = 0.0,
    settlement: str = "physical",
) -> tuple[float, float]:
    """
    Calculate the price of a European swaption with Black's Model: what the right to enter an interest rate swap at
    the strike rate is worth when the forward swap rate is lognormally distributed.

    Black's Model is a mathematical model used for pricing financial derivatives, its primary applications are for
    pricing options on future contracts, bond options, interest rate caps and floors, and swaptions. This function
    implements the swaption case only.

    Black's Model is a mathematical model used for pricing financial derivatives, its primary applications are for
    pricing options on future contracts, bond options, interest rate caps and floors, and swaptions. This function
    implements the swaption case only.

    Swaption is an option on an interest rate swap that gives the holder the right, but not the obligation,
    to enter into the swap at a predetermined fixed rate (strike rate) at a specified future time (maturity).

    Exercising a swaption is not a single payment at expiration but the right to enter a swap that exchanges
    cash flows at every payment date over the underlying swap's tenor. The Black-76 swaption price therefore
    discounts the option payoff by the swap's annuity (present value of a basis point) rather than a single
    discount factor to expiration — see `_get_annuity_factor`.

    Black's model assumes the forward rate is **lognormally** distributed, so `volatility`
    is a relative quantity expressed as a fraction of the forward rate: 0.20 means a 20%
    lognormal volatility, which on a 3.25% forward corresponds to about 65 basis points of
    rate movement per year. This is a different number from the absolute, basis-point
    volatility that `get_bachelier_price` consumes, and passing one where the other is
    expected misprices the swaption by a factor of roughly 1 / forward_rate. An absolute
    volatility can be handed to this function with its volatility type, which converts it
    explicitly — see `_convert_volatility` for the approximation used.

    Because the lognormal assumption takes the logarithm of the forward rate, the model is
    undefined for a zero or negative forward or strike rate. Euro area rates spent years
    below zero, so this is a real rather than theoretical restriction; use
    `get_bachelier_price`, whose normal distribution admits negative rates, or a shift.

    The shifted (displaced) Black model applies Black's formula to the forward and strike
    rate plus a fixed shift s, so that rates down to -s are allowed (Rubinstein, 1983). The
    volatility is then the lognormal volatility of F + s, as EUR and CHF swaption
    volatilities are quoted (with shifts of 1% to 3%).

    The formula is as follows:

    - Payer = Notional * Annuity * [(F + s) N(d_1) - (K + s) N(d_2)]
    - Receiver = Notional * Annuity * [(K + s) N(-d_2) - (F + s) N(-d_1)]
    - d_1 = (ln((F + s) / (K + s)) + σ² T / 2) / (σ √T), d_2 = d_1 - σ √T

    where F is the forward swap rate, K the strike rate, s the shift (0 for Black's model),
    T the years to expiry and the annuity that of `_get_annuity_factor`: on the curve for a
    physically settled swaption, or the cash annuity at the forward swap rate for a
    cash-settled one.

    For more information, see the following sources:

    - Black, F. (1976). "The Pricing of Commodity Contracts." Journal of Financial
      Economics, 3(1-2), 167-179.
    - Rubinstein, M. (1983). "Displaced Diffusion Option Pricing." Journal of Finance,
      38(1), 213-217.
    - Mercurio, F. (2008). "Cash-Settled Swaptions and No-Arbitrage." Risk, 21(2), 96-98.
    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      ch. 29.

    Referenced against:
    - QuantLib, BlackSwaptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_black_swaption_engine.html

    Args:
        forward_rate (float): Forward rate of the underlying swap. Must be positive.
        strike_rate (float): Strike rate of the swaption. Must be positive.
        volatility (float): Volatility of the underlying swap, quoted on the basis given by
        `volatility_type` and defaulting to the lognormal (relative) convention this model
        is defined in.
        years_to_maturity (float): years to maturity of the swaption.
        risk_free_rate (float | None, optional): The risk-free interest rate, continuously
            compounded and assumed flat over the swap's payment dates, which it discounts.
            Defaults to None, which requires a discount curve.
        notional (float, optional): Notional amount of the swap. Default is 10,000,000.
        tenor (float | None, optional): Tenor of the underlying swap. Defaults to None, which uses the years to
            maturity.
        payment_frequency (int, optional): Number of fixed-leg payments per year on the underlying swap
        (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly). Defaults to 2 (semi-annual).
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.
        volatility_type (str, optional): The convention `volatility` is quoted on. Options
        are: "lognormal" (relative, this model's native basis), "normal" (absolute, in rate
        units, converted before use). Defaults to "lognormal".
        discount_curve (pd.Series | None, optional): The annually compounded zero-coupon rates
        by maturity in years that discount the swap's payment dates instead of the flat
        risk-free rate, such as a curve of `get_zero_coupon_curve`; the forward swap rate and
        annuity on such a curve follow from `swap_model.get_swap_rate` with a forward start.
        Defaults to None, which discounts at the risk-free rate.
        shift (float, optional): The shift s added to the forward and strike rate, which
        makes it the shifted Black model. Defaults to 0, Black's model.
        settlement (str, optional): How the swaption settles. Options are: "physical" (it
        enters the swap), "cash" (it pays the swap's value at expiry with the cash annuity).
        Defaults to "physical".

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.

    Raises:
        ValueError: If volatility_type is not 'lognormal' or 'normal', if the forward
            rate or strike rate plus the shift is not positive, or the settlement is unknown.

    As an example:

    ```python
    from financetoolkit.fixedincome.bonds import derivative_model

    derivative_model.get_black_price(
        forward_rate=0.005, strike_rate=-0.002, volatility=0.25, years_to_maturity=2,
        risk_free_rate=0.01, tenor=5, payment_frequency=1, is_receiver=False, shift=0.02,
        settlement="cash",
    )
    ```

    Which returns:

    ```
    (372117.340059703, 0.007706961246807125)
    ```
    """
    volatility, annuity = _swaption_terms(
        "black",
        forward_rate,
        strike_rate,
        volatility,
        years_to_maturity,
        risk_free_rate,
        tenor,
        payment_frequency,
        volatility_type,
        discount_curve,
        shift,
        settlement,
    )
    payoff = _black_formula(
        forward_rate,
        strike_rate,
        volatility,
        years_to_maturity,
        not is_receiver,
        shift,
    )

    return notional * annuity * payoff, payoff


def get_bachelier_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float | None = None,
    notional: float = 10_000_000,
    tenor: float | None = None,
    payment_frequency: int = 2,
    is_receiver: bool = True,
    volatility_type: str = "normal",
    discount_curve: pd.Series | None = None,
    settlement: str = "physical",
) -> tuple[float, float]:
    """
    Calculate the price of a European swaption with the Bachelier Model: what the right to enter an interest rate
    swap at the strike rate is worth when the forward swap rate is normally distributed, which admits zero and
    negative rates.

    The Bachelier Model is an alternative to Black's Model for pricing options on futures contracts and interest
    rates. It assumes that the distribution of the underlying rate follows a normal distribution with constant
    volatility. This function implements the swaption case.

    The Bachelier Model is an alternative to Black's Model for pricing options on futures contracts and interest
    rates. It assumes that the distribution of the underlying rate follows a normal distribution with constant
    volatility. This function implements the swaption case.

    Exercising a swaption is not a single payment at expiration but the right to enter a swap that exchanges
    cash flows at every payment date over the underlying swap's tenor. The swaption price therefore discounts
    the option payoff by the swap's annuity (present value of a basis point) rather than a single discount
    factor to expiration — see `_get_annuity_factor`.

    Because the forward rate is assumed to be **normally** distributed, `volatility` is an
    absolute quantity in the same units as the rates themselves: 0.0065 means 65 basis
    points of annual rate movement, not 0.65%. This is the opposite convention to
    `get_black_price`, which takes a relative lognormal volatility, and the two numbers
    differ by a factor of roughly the forward rate — reusing a 20% lognormal quote here
    would price the swaption around thirty times too high on a 3.25% forward. Pass
    A relative volatility can be handed to this function with its volatility type, which
    converts it explicitly — see `_convert_volatility` for the approximation used.

    The normal distribution places positive probability on negative rates, which is
    precisely why this model came back into use for euro area and yen rates and why it,
    unlike Black's model, remains well defined at a zero or negative forward rate.

    At the money the price collapses to the closed form
    Annuity * Notional * volatility * SQRT(years_to_maturity / (2 * pi)).

    The formula is as follows:

    - Payer = Notional * Annuity * [(F - K) N(d) + σ √T φ(d)]
    - Receiver = Notional * Annuity * [(K - F) N(-d) + σ √T φ(d)]
    - d = (F - K) / (σ √T)

    where F is the forward swap rate, K the strike rate, σ the normal volatility, T the years
    to expiry and the annuity that of `_get_annuity_factor`: on the curve for a physically
    settled swaption, or the cash annuity at the forward swap rate for a cash-settled one.

    For more information, see the following sources:

    - Bachelier, L. (1900). "Théorie de la spéculation." Annales scientifiques de l'ENS, 17,
      21-86.
    - Mercurio, F. (2008). "Cash-Settled Swaptions and No-Arbitrage." Risk, 21(2), 96-98.
    - https://en.wikipedia.org/wiki/Bachelier_model

    Referenced against:
    - QuantLib, BachelierSwaptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_bachelier_swaption_engine.html

    Args:
        forward_rate (float): Forward rate of the underlying swap. May be zero or negative.
        strike_rate (float): Strike rate of the swaption. May be zero or negative.
        volatility (float): Volatility of the underlying swap, quoted on the basis given by
        `volatility_type` and defaulting to the normal (absolute, basis point) convention
        this model is defined in.
        years_to_maturity (float): years to maturity of the swaption.
        risk_free_rate (float | None, optional): The risk-free interest rate, continuously
            compounded and assumed flat over the swap's payment dates, which it discounts.
            Defaults to None, which requires a discount curve.
        notional (float, optional): Notional amount of the swap. Default is 10,000,000.
        tenor (float | None, optional): Tenor of the underlying swap. Defaults to None, which uses the years to
            maturity.
        payment_frequency (int, optional): Number of fixed-leg payments per year on the underlying swap
        (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly). Defaults to 2 (semi-annual).
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.
        volatility_type (str, optional): The convention `volatility` is quoted on. Options
        are: "normal" (absolute, this model's native basis), "lognormal" (relative to the
        forward rate, converted before use). Defaults to "normal".
        discount_curve (pd.Series | None, optional): The annually compounded zero-coupon rates
        by maturity in years that discount the swap's payment dates instead of the flat
        risk-free rate, such as a curve of `get_zero_coupon_curve`; the forward swap rate and
        annuity on such a curve follow from `swap_model.get_swap_rate` with a forward start.
        Defaults to None, which discounts at the risk-free rate.
        settlement (str, optional): How the swaption settles. Options are: "physical" (it
        enters the swap), "cash" (it pays the swap's value at expiry with the cash annuity).
        Defaults to "physical".

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.

    Raises:
        ValueError: If volatility_type is not 'lognormal' or 'normal', if a lognormal
            volatility is supplied around a forward rate of zero, or the settlement is
            unknown.

    As an example:

    ```python
    from financetoolkit.fixedincome.bonds import derivative_model

    derivative_model.get_bachelier_price(
        forward_rate=0.035, strike_rate=0.03, volatility=0.009, years_to_maturity=1,
        risk_free_rate=0.03, tenor=5, payment_frequency=2,
    )
    ```

    Which returns:

    ```
    (72929.06171049128, 0.0016307438437253015)
    ```
    """
    volatility, annuity = _swaption_terms(
        "bachelier",
        forward_rate,
        strike_rate,
        volatility,
        years_to_maturity,
        risk_free_rate,
        tenor,
        payment_frequency,
        volatility_type,
        discount_curve,
        settlement=settlement,
    )
    payoff = _bachelier_formula(
        forward_rate, strike_rate, volatility, years_to_maturity, not is_receiver
    )

    return notional * annuity * payoff, payoff


def get_swaption_greeks(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float | None = None,
    notional: float = 10_000_000,
    tenor: float | None = None,
    payment_frequency: int = 2,
    is_receiver: bool = True,
    method: str = "black",
    shift: float = 0.0,
    discount_curve: pd.Series | None = None,
    settlement: str = "physical",
) -> dict[str, float]:
    """
    Calculate the price, delta and vega of a European swaption in the Black, shifted Black
    or Bachelier model: how much the swaption's value changes with the forward swap rate and
    with its volatility.

    The delta is the change in value per unit change of the forward swap rate and the vega
    the change per unit change of the volatility, both with the annuity held fixed, as the
    market quotes them: a forward rate or volatility one basis point (0.0001) higher changes
    the value by about 0.0001 times either. The volatility is on the model's own basis:
    lognormal (of the shifted rate) for Black, normal for Bachelier.

    The formula is as follows:

    - Black: Delta = N A N(d_1) (payer), -N A N(-d_1) (receiver); Vega = N A (F + s) φ(d_1) √T
    - d_1 = (ln((F + s) / (K + s)) + σ² T / 2) / (σ √T)
    - Bachelier: Delta = N A N(d) (payer), -N A N(-d) (receiver); Vega = N A φ(d) √T
    - d = (F - K) / (σ √T)

    where N is the notional, A the annuity of `_get_annuity_factor` (the cash annuity for a
    cash-settled swaption), F the forward swap rate, K the strike rate, s the shift and T the
    years to expiry. The vega is the same for a payer and a receiver swaption.

    For more information, see the following sources:

    - Black, F. (1976). "The Pricing of Commodity Contracts." Journal of Financial
      Economics, 3(1-2), 167-179.
    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      ch. 19 and 29.

    QuantLib's BlackSwaptionEngine and BachelierSwaptionEngine report these as their delta and
    vega results.

    Referenced against:
    - QuantLib, BlackSwaptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_black_swaption_engine.html
    - QuantLib, BachelierSwaptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_bachelier_swaption_engine.html

    Args:
        forward_rate (float): The forward swap rate.
        strike_rate (float): The strike rate of the swaption.
        volatility (float): The volatility: lognormal for the Black model (of the forward
            rate plus the shift), normal (absolute) for the Bachelier model.
        years_to_maturity (float): The years until the swaption expires.
        risk_free_rate (float | None, optional): The risk-free interest rate, continuously
            compounded and assumed flat over the swap's payment dates, which it discounts.
            Defaults to None, which requires a discount curve.
        notional (float, optional): The notional of the swap. Defaults to 10,000,000.
        tenor (float | None, optional): The tenor of the underlying swap. Defaults to None,
            which uses the years to maturity.
        payment_frequency (int, optional): The fixed payments per year of the swap.
            Defaults to 2 (semi-annual).
        is_receiver (bool, optional): Whether the swaption receives the fixed rate.
            Defaults to True.
        method (str, optional): The model. Options are: "black" (Black's model, shifted by
            `shift`), "bachelier" (the Bachelier model). Defaults to "black".
        shift (float, optional): The shift of the Black model. Defaults to 0.
        discount_curve (pd.Series | None, optional): The annually compounded zero-coupon
            rates by maturity in years that discount the swap's payment dates. Defaults to
            None, which discounts at the risk-free rate.
        settlement (str, optional): How the swaption settles. Options are: "physical" (it
            enters the swap), "cash" (it pays the swap's value with the cash annuity).
            Defaults to "physical".

    Returns:
        dict[str, float]: The Price, the Delta (per unit of the forward rate) and the Vega
        (per unit of volatility), in the currency of the notional, and the Annuity.

    Raises:
        ValueError: If the method or settlement is unknown, or the Black model gets a
            forward or strike rate plus shift that is not positive.

    As an example:

    ```python
    from financetoolkit.fixedincome.bonds import derivative_model

    derivative_model.get_swaption_greeks(
        forward_rate=0.035, strike_rate=0.03, volatility=0.2, years_to_maturity=1,
        risk_free_rate=0.03, tenor=5, payment_frequency=2, is_receiver=False,
    )
    ```

    Which returns:

    ```
    {'Price': 260171.37136987157, 'Delta': 36137334.751143426, 'Vega': 427414.7024167665, 'Annuity': 4.472134724966416}
    ```
    """
    volatility, annuity = _swaption_terms(
        method,
        forward_rate,
        strike_rate,
        volatility,
        years_to_maturity,
        risk_free_rate,
        tenor,
        payment_frequency,
        "lognormal" if method == "black" else "normal",
        discount_curve,
        shift if method == "black" else 0.0,
        settlement,
    )
    is_call = not is_receiver
    sign = 1.0 if is_call else -1.0
    root_time = np.sqrt(years_to_maturity)
    deviation = volatility * root_time

    if method == "black":
        price = _black_formula(
            forward_rate, strike_rate, volatility, years_to_maturity, is_call, shift
        )
        forward, strike = forward_rate + shift, strike_rate + shift
        if deviation > 0:
            d1 = (np.log(forward / strike) + 0.5 * deviation**2) / deviation
            delta = sign * _normal_cdf(sign * d1)
            vega = forward * _normal_pdf(d1) * root_time
        else:
            delta, vega = float(sign * (forward - strike) > 0) * sign, 0.0
    else:
        price = _bachelier_formula(
            forward_rate, strike_rate, volatility, years_to_maturity, is_call
        )
        if deviation > 0:
            d = (forward_rate - strike_rate) / deviation
            delta = sign * _normal_cdf(sign * d)
            vega = _normal_pdf(d) * root_time
        else:
            delta, vega = float(sign * (forward_rate - strike_rate) > 0) * sign, 0.0

    scale = notional * annuity

    return {
        "Price": scale * price,
        "Delta": float(scale * delta),
        "Vega": float(scale * vega),
        "Annuity": annuity,
    }


OPTION_TYPES = ("call", "put")

# The widest bracket the implied volatility solvers search, in total standard deviation σ √T
# (Black, Bachelier) or volatility (caps): far beyond any traded option.
MAXIMUM_DEVIATION = 1e3


def get_implied_volatility(
    price: float,
    forward_rate: float,
    strike_rate: float,
    years_to_expiry: float,
    annuity: float = 1.0,
    option_type: str = "call",
    method: str = "black",
    shift: float = 0.0,
) -> float:
    """
    Calculate the Black (lognormal, optionally shifted) or Bachelier (normal) implied
    volatility of an option on a forward rate, such as a caplet, floorlet or swaption: the
    volatility at which the model returns the option's price.

    The price is divided by the annuity, the factor the model's formula is multiplied by:
    the notional times the swap annuity for a swaption, or the notional times the accrual
    and the discount factor of the payment date for a caplet. What is left is solved for the
    volatility with Brent's method on a bracket that is widened until it contains the root
    (Brent, 1973), as both formulas increase strictly with the volatility. An in-the-money
    option is first turned into the out-of-the-money option of the same strike by put-call
    parity, Call - Put = F - K, where the volatility is far better determined (Jäckel, 2015).

    The no-arbitrage bounds of the undiscounted price V are:

    - Black: max(F - K, 0) ≤ V < F + s for a call and max(K - F, 0) ≤ V < K + s for a put,
      with F + s and K + s positive
    - Bachelier: V ≥ max(F - K, 0) for a call and V ≥ max(K - F, 0) for a put

    A price on the lower bound has a volatility of 0; a price outside the bounds has no
    implied volatility and returns NaN.

    For more information, see the following sources:

    - Jäckel, P. (2015). "Let's Be Rational." Wilmott, 2015(75), 40-53.
    - Jäckel, P. (2017). "Implied Normal Volatility." Wilmott, 2017(90), 54-57.
    - Choi, J., Kim, K. and Kwak, M. (2009). "Numerical Approximation of the Implied
      Volatility Under Arithmetic Brownian Motion." Applied Mathematical Finance, 16(3),
      261-268.
    - Brent, R.P. (1973). "Algorithms for Minimization Without Derivatives." Prentice-Hall,
      ch. 4.

    Referenced against:
    - QuantLib, blackFormulaImpliedStdDev: https://www.quantlib.org/reference/namespace_quant_lib.html#ae5515dd7810c45e5a7c61df4a00a8166
    - QuantLib, bachelierBlackFormulaImpliedVol: https://www.quantlib.org/reference/namespace_quant_lib.html#ae7f634488a3e089f97d4d1e64625dda8

    Args:
        price (float): The price of the option.
        forward_rate (float): The forward rate (or forward swap rate).
        strike_rate (float): The strike rate.
        years_to_expiry (float): The years until the rate is fixed (the swaption expires).
        annuity (float, optional): The factor the formula is multiplied by, so that the
            price over the annuity is the undiscounted value. Defaults to 1, an undiscounted
            price per unit of notional.
        option_type (str, optional): The option. Options are: "call" (an option on a rising
            rate, a caplet or payer swaption), "put" (an option on a falling rate, a floorlet
            or receiver swaption). Defaults to "call".
        method (str, optional): The model. Options are: "black" (Black's model, shifted by
            `shift`), "bachelier" (the Bachelier model). Defaults to "black".
        shift (float, optional): The shift of the Black model. Defaults to 0.

    Returns:
        float: The implied volatility per year: lognormal (of the rate plus the shift) for
        the Black model, normal (absolute, in rate units) for the Bachelier model. NaN when the price lies
        outside the no-arbitrage bounds.

    Raises:
        ValueError: If the method or option type is unknown, the years to expiry or the
            annuity are not positive, or the Black model gets a forward or strike rate plus
            shift that is not positive.

    As an example:

    ```python
    from financetoolkit.fixedincome.bonds import derivative_model

    price, _ = derivative_model.get_bachelier_price(
        forward_rate=0.035, strike_rate=0.03, volatility=0.009, years_to_maturity=2,
        risk_free_rate=0.03, tenor=5, payment_frequency=2, is_receiver=False,
    )
    annuity = derivative_model.get_swaption_greeks(
        forward_rate=0.035, strike_rate=0.03, volatility=0.009, years_to_maturity=2,
        risk_free_rate=0.03, tenor=5, payment_frequency=2, method="bachelier",
    )["Annuity"]

    derivative_model.get_implied_volatility(
        price, forward_rate=0.035, strike_rate=0.03, years_to_expiry=2,
        annuity=10_000_000 * annuity, method="bachelier",
    )
    ```

    Which returns:

    ```
    0.009
    ```
    """
    check_option("option_type", option_type, OPTION_TYPES)
    _check_method(method, forward_rate, strike_rate, shift)
    if years_to_expiry <= 0 or annuity <= 0:
        raise ValueError("The years to expiry and the annuity must be positive.")

    is_call = option_type == "call"
    value = price / annuity
    gap = forward_rate - strike_rate
    intrinsic = max(gap if is_call else -gap, 0.0)
    # The out-of-the-money option of the same strike, by put-call parity.
    out_of_the_money = value - intrinsic
    otm_is_call = gap < 0 if intrinsic > 0 else is_call
    upper = (
        (forward_rate if otm_is_call else strike_rate) + shift
        if method == "black"
        else np.inf
    )
    tolerance = 1e-14 * max(1.0, abs(forward_rate) + abs(shift))

    if (
        not np.isfinite(value)
        or out_of_the_money < -tolerance
        or out_of_the_money >= upper
    ):
        return np.nan
    if out_of_the_money <= tolerance:
        return 0.0

    def error(deviation: float) -> float:
        return (
            _option_value(
                method, forward_rate, strike_rate, deviation, 1.0, otm_is_call, shift
            )
            - out_of_the_money
        )

    # The root is the total standard deviation σ √T; double the bracket until it holds it.
    high = max(abs(gap), out_of_the_money, 0.01 if method == "black" else 1e-4)
    while error(high) < 0:
        high *= 2
        if high > MAXIMUM_DEVIATION:
            return np.nan

    deviation = brentq(error, 0.0, high, xtol=1e-16, rtol=1e-14, maxiter=500)

    return float(deviation / np.sqrt(years_to_expiry))


CAP_FLOOR_TYPES = ("cap", "floor", "collar")


def _cap_floor_table(
    forward_rates: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    expiries: np.ndarray,
    discounts: np.ndarray,
    strike_rate: float,
    volatility: float | np.ndarray,
    notional: float,
    option_type: str,
    floor_strike_rate: float | None,
    method: str,
    shift: float,
) -> pd.DataFrame:
    """
    The caplets or floorlets of a cap, floor or collar on given forward rates, see
    get_cap_floor.

    Args:
        forward_rates (np.ndarray): The forward rate of each period.
        starts (np.ndarray): The start of each period in years.
        ends (np.ndarray): The end (payment) of each period in years.
        expiries (np.ndarray): The years until each rate is fixed.
        discounts (np.ndarray): The discount factor of each payment.
        strike_rate (float): The strike rate (of the cap for a collar).
        volatility (float | np.ndarray): One volatility or one per period.
        notional (float): The notional.
        option_type (str): "cap", "floor" or "collar".
        floor_strike_rate (float | None): The strike rate of the floor of a collar.
        method (str): "black" or "bachelier".
        shift (float): The shift of the Black model.

    Returns:
        pd.DataFrame: One row per period with its terms and price.
    """
    check_option("option_type", option_type, CAP_FLOOR_TYPES)
    if option_type == "collar" and floor_strike_rate is None:
        raise ValueError("A collar needs the strike rate of its floor.")
    volatilities = np.broadcast_to(
        np.asarray(volatility, dtype=float), forward_rates.shape
    )
    strikes = [strike_rate] + ([floor_strike_rate] if option_type == "collar" else [])
    for strike in strikes:
        _check_method(method, forward_rates, strike, shift)

    accruals = ends - starts
    scale = notional * accruals * discounts
    value = _option_value(
        method,
        forward_rates,
        strike_rate,
        volatilities,
        expiries,
        option_type != "floor",
        shift,
    )
    if option_type == "collar":
        # A long cap and a short floor: the rate paid is kept between the two strikes.
        value = value - _option_value(
            method,
            forward_rates,
            floor_strike_rate,
            volatilities,
            expiries,
            False,
            shift,
        )

    return pd.DataFrame(
        {
            "Start": starts,
            "End": ends,
            "Forward Rate": forward_rates,
            "Discount Factor": discounts,
            "Volatility": volatilities,
            "Price": scale * value,
        }
    )


def _cap_periods(
    zero_curve: pd.Series, maturity: float, frequency: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The periods of a cap that fix after today, with their simple forward rates and the
    discount factors of their payments.

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates by maturity.
        maturity (float): The years until the last payment.
        frequency (int): The periods per year.

    Returns:
        tuple: The starts, ends, forward rates and discount factors.
    """
    if maturity <= 0:
        raise ValueError(f"The maturity of a cap must be positive, not {maturity}.")
    ends = bond_model.get_cash_flow_periods(maturity, frequency) / frequency
    starts = np.maximum(ends - 1 / frequency, 0.0)
    # The first rate is fixed today, so its period is known and not an option.
    later = ~np.isclose(starts, 0.0, rtol=0.0, atol=1e-12)
    starts, ends = starts[later], ends[later]
    if len(ends) == 0:
        raise ValueError(
            "The cap has no period that fixes after today: lengthen the maturity."
        )
    start_discounts = discount_factors(zero_curve, starts)
    discounts = discount_factors(zero_curve, ends)
    forward_rates = (start_discounts / discounts - 1) / (ends - starts)

    return starts, ends, forward_rates, discounts


def get_cap_floor(
    zero_curve: pd.Series,
    strike_rate: float,
    volatility: float | np.ndarray | list,
    maturity: float,
    frequency: int = 4,
    notional: float = 10_000_000,
    option_type: str = "cap",
    floor_strike_rate: float | None = None,
    method: str = "black",
    shift: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    """
    Calculate the price of an interest rate cap, floor or collar, and of each of its caplets
    or floorlets, in the Black, shifted Black or Bachelier model: what the protection
    against a rising (cap) or falling (floor) floating rate is worth.

    A cap pays, at the end of every period, the amount by which the floating rate fixed at
    its start exceeds the strike rate, on the period's accrual. Each payment is a caplet: a
    call on the forward rate of its period, priced with the model's formula and discounted
    from its payment date (Black, 1976; Hull, 2022, ch. 29). A floor is a series of puts
    (floorlets) and a collar is a long cap and a short floor, which keeps the rate paid
    between the floor and the cap strike.

    The formula is as follows:

    - F_i = (P(t_(i-1)) / P(t_i) - 1) / τ_i, the simple forward rate of period i
    - Caplet_i = N τ_i P(t_i) * Black(F_i, K, σ_i, t_(i-1)) or Bachelier(F_i, K, σ_i, t_(i-1))
    - Black call = (F + s) N(d_1) - (K + s) N(d_2), d_1 = (ln((F + s) / (K + s)) + σ² t / 2) / (σ √t)
    - Bachelier call = (F - K) N(d) + σ √t φ(d), d = (F - K) / (σ √t)
    - Cap = SUM(Caplet_i), Floor = SUM(Floorlet_i), Collar = Cap(K) - Floor(K_floor)

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero rate
    r(t), τ_i = t_i - t_(i-1) the accrual of the period in years, t_(i-1) its fixing time and
    s the shift (0 for Black). The forward rates are simple on that accrual, from a single
    curve that both projects and discounts. The periods are counted back from the maturity
    in steps of 1 / frequency years; the first period, whose rate is fixed today, is left
    out, as the market does.

    For more information, see the following sources:

    - Black, F. (1976). "The Pricing of Commodity Contracts." Journal of Financial
      Economics, 3(1-2), 167-179.
    - Bachelier, L. (1900). "Théorie de la spéculation." Annales scientifiques de l'ENS, 17,
      21-86.
    - Rubinstein, M. (1983). "Displaced Diffusion Option Pricing." Journal of Finance,
      38(1), 213-217.
    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      ch. 29.

    Referenced against:
    - QuantLib, Cap: https://www.quantlib.org/reference/class_quant_lib_1_1_cap.html
    - QuantLib, Floor: https://www.quantlib.org/reference/class_quant_lib_1_1_floor.html
    - QuantLib, Collar: https://www.quantlib.org/reference/class_quant_lib_1_1_collar.html
    - QuantLib, BlackCapFloorEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_black_cap_floor_engine.html
    - QuantLib, BachelierCapFloorEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_bachelier_cap_floor_engine.html

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        strike_rate (float): The strike rate (of the cap, for a collar).
        volatility (float | np.ndarray | list): One (flat) volatility for every caplet or one
            per caplet: lognormal for the Black model (of the rate plus the shift), normal
            (absolute) for the Bachelier model.
        maturity (float): The years until the last payment.
        frequency (int, optional): The periods per year. Defaults to 4 (quarterly).
        notional (float, optional): The notional. Defaults to 10,000,000.
        option_type (str, optional): The option. Options are: "cap" (caplets on a rising
            rate), "floor" (floorlets on a falling rate), "collar" (a cap bought and a floor
            sold). Defaults to "cap".
        floor_strike_rate (float | None, optional): The strike rate of the floor of a collar.
            Defaults to None.
        method (str, optional): The model. Options are: "black" (Black's model, shifted by
            `shift`), "bachelier" (the Bachelier model). Defaults to "black".
        shift (float, optional): The shift of the Black model. Defaults to 0.

    Returns:
        tuple[float, pd.DataFrame]: The price and, per caplet, its Start and End (years),
        Forward Rate, Discount Factor, Volatility and Price, in the currency of the notional.

    Raises:
        ValueError: If the option type or method is unknown, a collar has no floor strike,
            the cap has no period that fixes after today, or the Black model gets a forward
            or strike rate plus shift that is not positive.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.bonds import derivative_model

    zero_curve = pd.Series([0.035, 0.037, 0.040], index=[1, 2, 5])

    price, caplets = derivative_model.get_cap_floor(
        zero_curve, strike_rate=0.04, volatility=0.2, maturity=2, frequency=2
    )

    caplets.round(4)
    ```

    Which returns:

    ```
       Start  End  Forward Rate  Discount Factor  Volatility       Price
    0    0.5  1.0        0.0347           0.9662         0.2   2096.0107
    1    1.0  1.5        0.0376           0.9483         0.2   9754.2696
    2    1.5  2.0        0.0396           0.9299         0.2  17156.9434
    ```
    """
    starts, ends, forward_rates, discounts = _cap_periods(
        zero_curve, maturity, frequency
    )
    caplets = _cap_floor_table(
        forward_rates,
        starts,
        ends,
        starts,
        discounts,
        strike_rate,
        volatility,
        notional,
        option_type,
        floor_strike_rate,
        method,
        shift,
    )

    return float(caplets["Price"].sum()), caplets


def get_cap_floor_implied_volatility(
    price: float,
    zero_curve: pd.Series,
    strike_rate: float,
    maturity: float,
    frequency: int = 4,
    notional: float = 10_000_000,
    option_type: str = "cap",
    method: str = "black",
    shift: float = 0.0,
) -> float:
    """
    Calculate the flat implied volatility of a cap or floor: the one volatility that, used
    for every caplet, returns the price (Hull, 2022, ch. 29).

    Cap volatilities are quoted flat per maturity, so this is the inverse of `get_cap_floor`
    with a single volatility. The price increases strictly with the volatility, so it is
    solved with Brent's method on a bracket that is widened until it holds the root.

    The formula is as follows:

    - Solve σ: SUM(Caplet_i(σ)) = Price

    where Caplet_i is the caplet (or floorlet) of `get_cap_floor`. The price lies between the
    value at zero volatility, the sum of the discounted intrinsic values, and, in the Black
    model, the value at infinite volatility, SUM(N τ_i P(t_i) (F_i + s)) for a cap and
    SUM(N τ_i P(t_i) (K + s)) for a floor. A price outside these bounds has no implied
    volatility and returns NaN.

    For more information, see the following sources:

    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      ch. 29.
    - Brent, R.P. (1973). "Algorithms for Minimization Without Derivatives." Prentice-Hall,
      ch. 4.

    Referenced against:
    - QuantLib, CapFloor.impliedVolatility: https://www.quantlib.org/reference/class_quant_lib_1_1_cap_floor.html

    Args:
        price (float): The price of the cap or floor.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        strike_rate (float): The strike rate.
        maturity (float): The years until the last payment.
        frequency (int, optional): The periods per year. Defaults to 4 (quarterly).
        notional (float, optional): The notional. Defaults to 10,000,000.
        option_type (str, optional): The option. Options are: "cap" (caplets on a rising
            rate), "floor" (floorlets on a falling rate). Defaults to "cap".
        method (str, optional): The model. Options are: "black" (Black's model, shifted by
            `shift`), "bachelier" (the Bachelier model). Defaults to "black".
        shift (float, optional): The shift of the Black model. Defaults to 0.

    Returns:
        float: The flat implied volatility per year: lognormal (of the rate plus the shift)
        for the Black model, normal (absolute) for the Bachelier model. NaN when the price lies outside the
        no-arbitrage bounds.

    Raises:
        ValueError: If the option type is not "cap" or "floor", or see `get_cap_floor`.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.bonds import derivative_model

    zero_curve = pd.Series([0.035, 0.037, 0.040], index=[1, 2, 5])

    price, _ = derivative_model.get_cap_floor(
        zero_curve, strike_rate=0.04, volatility=0.22, maturity=5, frequency=2
    )

    derivative_model.get_cap_floor_implied_volatility(
        price, zero_curve, strike_rate=0.04, maturity=5, frequency=2
    )
    ```

    Which returns:

    ```
    0.21999999999999997
    ```
    """
    check_option("option_type", option_type, ("cap", "floor"))
    starts, ends, forward_rates, discounts = _cap_periods(
        zero_curve, maturity, frequency
    )

    def cap_price(volatility: float) -> float:
        return float(
            _cap_floor_table(
                forward_rates,
                starts,
                ends,
                starts,
                discounts,
                strike_rate,
                volatility,
                notional,
                option_type,
                None,
                method,
                shift,
            )["Price"].sum()
        )

    scale = notional * (ends - starts) * discounts
    lower = cap_price(0.0)
    if method == "black":
        cap_bound = (
            forward_rates + shift if option_type == "cap" else strike_rate + shift
        )
        upper = float(np.sum(scale * cap_bound))
    else:
        upper = np.inf
    tolerance = 1e-12 * max(abs(price), notional * 1e-8)

    if not np.isfinite(price) or price < lower - tolerance or price >= upper:
        return np.nan
    if price <= lower + tolerance:
        return 0.0

    high = 0.5 if method == "black" else 0.01
    while cap_price(high) < price:
        high *= 2
        if high > MAXIMUM_DEVIATION:
            return np.nan

    return float(
        brentq(lambda vol: cap_price(vol) - price, 0.0, high, xtol=1e-14, maxiter=500)
    )


def get_year_on_year_inflation_cap_floor(
    zero_curve: pd.Series,
    zero_inflation_curve: pd.Series,
    strike_rate: float,
    volatility: float | np.ndarray | list,
    maturity: int,
    notional: float = 10_000_000,
    option_type: str = "cap",
    floor_strike_rate: float | None = None,
    method: str = "black",
    shift: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    """
    Calculate the price of a year-on-year inflation cap, floor or collar, and of each of its
    caplets or floorlets: what the protection against high (cap) or low (floor) yearly
    inflation is worth, in the Black, shifted Black (with a shift of 1, the unit-displaced
    Black model of the inflation market) or Bachelier model.

    A year-on-year inflation caplet pays, at the end of year i, the amount by which that
    year's inflation I(i) / I(i - 1) - 1 exceeds the strike rate. It is a call on the
    forward year-on-year rate, fixed when the index of the end of the year is known
    (Kerkhof, 2005; Mercurio, 2005). Every year, the first included, is an option, as the
    index at its end is not known today. A floor is a series of puts on the same rates and a
    collar a long cap and a short floor.

    The formula is as follows:

    - f_i = (1 + z(i))^i / (1 + z(i - 1))^(i - 1) - 1, the forward year-on-year rate
    - Caplet_i = N P(i) * Black(f_i, K, σ_i, i) or Bachelier(f_i, K, σ_i, i)
    - Black call = (f + s) N(d_1) - (K + s) N(d_2), d_1 = (ln((f + s) / (K + s)) + σ² t / 2) / (σ √t)
    - Bachelier call = (f - K) N(d) + σ √t φ(d), d = (f - K) / (σ √t)
    - Cap = SUM(Caplet_i), Collar = Cap(K) - Floor(K_floor)

    where P(i) = (1 + r(i))^(-i) is the discount factor of the annually compounded nominal
    zero rate, z(i) the annually compounded zero-coupon inflation swap rate and s the shift.
    Every period accrues a full year. The forward rates are the ratios of the forward
    indices, without the convexity adjustment of a rate fixed and paid at the end of its
    year (see `inflation_swap_model.get_forward_year_on_year_inflation_rates`); with a
    year-on-year curve of your own, the convexity is in the curve. The time to expiry is the
    end of the year: the observation lag of the index is part of the times of the curves.

    For more information, see the following sources:

    - Kerkhof, J. (2005). "Inflation Derivatives Explained." Lehman Brothers Fixed Income
      Quantitative Research.
    - Mercurio, F. (2005). "Pricing Inflation-Indexed Derivatives." Quantitative Finance,
      5(3), 289-302.
    - Black, F. (1976). "The Pricing of Commodity Contracts." Journal of Financial
      Economics, 3(1-2), 167-179.

    Referenced against:
    - QuantLib, YoYInflationCap: https://www.quantlib.org/reference/class_quant_lib_1_1_yo_y_inflation_cap.html
    - QuantLib, YoYInflationFloor: https://www.quantlib.org/reference/class_quant_lib_1_1_yo_y_inflation_floor.html
    - QuantLib, YoYInflationCollar: https://www.quantlib.org/reference/class_quant_lib_1_1_yo_y_inflation_collar.html
    - QuantLib, YoYInflationBlackCapFloorEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_yo_y_inflation_black_cap_floor_engine.html
    - QuantLib, YoYInflationUnitDisplacedBlackCapFloorEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_yo_y_inflation_unit_displaced_black_cap_floor_engine.html
    - QuantLib, YoYInflationBachelierCapFloorEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_yo_y_inflation_bachelier_cap_floor_engine.html

    Args:
        zero_curve (pd.Series): The annually compounded nominal zero-coupon rates
            (decimals), indexed by maturity in years.
        zero_inflation_curve (pd.Series): The annually compounded zero-coupon inflation swap
            rates (decimals), indexed by maturity in years.
        strike_rate (float): The strike rate (of the cap, for a collar).
        volatility (float | np.ndarray | list): One volatility for every caplet or one per
            caplet: lognormal for the Black model (of the rate plus the shift), normal
            (absolute) for the Bachelier model.
        maturity (int): The years until the last payment, a whole number.
        notional (float, optional): The notional. Defaults to 10,000,000.
        option_type (str, optional): The option. Options are: "cap" (caplets on high
            inflation), "floor" (floorlets on low inflation), "collar" (a cap bought and a
            floor sold). Defaults to "cap".
        floor_strike_rate (float | None, optional): The strike rate of the floor of a collar.
            Defaults to None.
        method (str, optional): The model. Options are: "black" (Black's model, shifted by
            `shift`), "bachelier" (the Bachelier model). Defaults to "black".
        shift (float, optional): The shift of the Black model, 1 for the unit-displaced
            model. Defaults to 0.

    Returns:
        tuple[float, pd.DataFrame]: The price and, per caplet, its Start and End (years),
        Forward Rate, Discount Factor, Volatility and Price, in the currency of the notional.

    Raises:
        ValueError: If the maturity is not a positive whole number of years, the option type
            or method is unknown, a collar has no floor strike, or the Black model gets a
            forward or strike rate plus shift that is not positive.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.bonds import derivative_model

    nominal = pd.Series([0.040, 0.042, 0.045], index=[1, 5, 10])
    zero_inflation = pd.Series([0.030, 0.026, 0.024], index=[1, 3, 5])

    price, caplets = derivative_model.get_year_on_year_inflation_cap_floor(
        nominal, zero_inflation, strike_rate=0.03, volatility=0.01, maturity=5,
        method="bachelier",
    )

    caplets.round(5)
    ```

    Which returns:

    ```
       Start  End  Forward Rate  Discount Factor  Volatility        Price
    0    0.0  1.0       0.03000          0.96154        0.01  38359.83465
    1    1.0  2.0       0.02600          0.92367        0.01  35723.67306
    2    2.0  3.0       0.02201          0.88644        0.01  32247.39426
    3    3.0  4.0       0.02201          0.84989        0.01  39186.66668
    4    4.0  5.0       0.02001          0.81407        0.01  39085.89943
    ```
    """
    if maturity < 1 or float(maturity) != int(maturity):
        raise ValueError(
            "A year-on-year inflation cap pays once a year: its maturity must be a "
            "positive whole number of years."
        )
    forward_rates = inflation_swap_model.get_forward_year_on_year_inflation_rates(
        zero_inflation_curve, int(maturity)
    ).to_numpy()
    ends = np.arange(1, int(maturity) + 1, dtype=float)

    caplets = _cap_floor_table(
        forward_rates,
        ends - 1,
        ends,
        ends,
        discount_factors(zero_curve, ends),
        strike_rate,
        volatility,
        notional,
        option_type,
        floor_strike_rate,
        method,
        shift,
    )

    return float(caplets["Price"].sum()), caplets
