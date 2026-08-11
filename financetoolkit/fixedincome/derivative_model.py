"""Derivative Models"""

import numpy as np
from scipy.stats import norm

VOLATILITY_TYPES = ("lognormal", "normal")


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
    risk_free_rate: float,
    years_to_maturity: float,
    tenor: float,
    payment_frequency: int,
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

    Where accrual = 1 / payment_frequency and n = tenor * payment_frequency is the
    number of payments made over the life of the underlying swap.

    Args:
        risk_free_rate (float): The risk-free interest rate used to discount each
        payment date, assumed flat across the curve.
        years_to_maturity (float): Years to expiration of the swaption, i.e. the time
        at which the underlying swap would start if exercised.
        tenor (float): The tenor (length in years) of the underlying swap.
        payment_frequency (int): The number of fixed-leg payments per year on the
        underlying swap (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly).

    Returns:
        float: The annuity factor (PVBP) of the underlying swap.
    """
    number_of_payments = max(round(tenor * payment_frequency), 1)
    accrual = 1 / payment_frequency
    payment_times = years_to_maturity + accrual * np.arange(1, number_of_payments + 1)

    return accrual * np.sum(np.exp(-risk_free_rate * payment_times))


def get_black_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float,
    notional: float = 10_000_000,
    tenor: float | None = None,
    payment_frequency: int = 2,
    is_receiver: bool = True,
    volatility_type: str = "lognormal",
) -> tuple[float, float]:
    """
    Black's Model for pricing financial derivatives.

    Black's Model is a mathematical model used for pricing financial derivatives, its primary applications are for
    pricing options on future contracts, bond options, interest rate cap and floors, and swaptions.

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
    expected misprices the swaption by a factor of roughly 1 / forward_rate. Pass
    `volatility_type='normal'` to hand this function an absolute volatility and have it
    converted explicitly — see `_convert_volatility` for the approximation used.

    Because the lognormal assumption takes the logarithm of the forward rate, the model is
    undefined for a zero or negative forward or strike rate. Euro area rates spent years
    below zero, so this is a real rather than theoretical restriction; use
    `get_bachelier_price`, whose normal distribution admits negative rates, in that case.

    For more information, see: https://en.wikipedia.org/wiki/Black_model

    Args:
        forward_rate (float): Forward rate of the underlying swap. Must be positive.
        strike_rate (float): Strike rate of the swaption. Must be positive.
        volatility (float): Volatility of the underlying swap, quoted on the basis given by
        `volatility_type` and defaulting to the lognormal (relative) convention this model
        is defined in.
        years_to_maturity (float): years to maturity of the swaption.
        risk_free_rate (float): The risk-free interest rate.
        notional (float, optional): Notional amount of the swap. Default is 10,000,000.
        tenor (float | None, optional): Tenor of the underlying swap. Defaults to being equal to years to maturity.
        payment_frequency (int, optional): Number of fixed-leg payments per year on the underlying swap
        (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly). Defaults to 2 (semi-annual).
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.
        volatility_type (str, optional): The convention `volatility` is quoted on, either
        'lognormal' (relative, this model's native basis) or 'normal' (absolute, in rate
        units), in which case it is converted before use. Defaults to 'lognormal'.

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.

    Raises:
        ValueError: If volatility_type is not 'lognormal' or 'normal', or if the forward
            rate or strike rate is not positive.
    """
    if forward_rate <= 0 or strike_rate <= 0:
        raise ValueError(
            "Black's model assumes a lognormally distributed forward rate and is therefore "
            f"undefined for the given forward rate ({forward_rate}) and strike rate "
            f"({strike_rate}), both of which must be positive. Use the Bachelier model "
            "instead, which admits zero and negative rates."
        )

    volatility = _convert_volatility(
        volatility=volatility,
        volatility_type=volatility_type,
        target_type="lognormal",
        forward_rate=forward_rate,
    )

    tenor = years_to_maturity if tenor is None else tenor

    d1 = (
        np.log(forward_rate / strike_rate) + 0.5 * volatility**2 * years_to_maturity
    ) / (volatility * np.sqrt(years_to_maturity))
    d2 = d1 - volatility * np.sqrt(years_to_maturity)

    if is_receiver:
        payoff = -forward_rate * norm.cdf(-d1) + strike_rate * norm.cdf(-d2)
    else:
        payoff = forward_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2)

    annuity = _get_annuity_factor(
        risk_free_rate=risk_free_rate,
        years_to_maturity=years_to_maturity,
        tenor=tenor,
        payment_frequency=payment_frequency,
    )
    swaption_price = notional * annuity * payoff

    return swaption_price, payoff


def get_bachelier_price(
    forward_rate: float,
    strike_rate: float,
    volatility: float,
    years_to_maturity: float,
    risk_free_rate: float,
    notional: float = 10_000_000,
    tenor: float | None = None,
    payment_frequency: int = 2,
    is_receiver: bool = True,
    volatility_type: str = "normal",
) -> tuple[float, float]:
    """
    Bachelier Model for pricing future contracts.

    The Bachelier Model is an alternative to Black's Model for pricing options on futures contracts.
    It assumes that the distribution of the underlying asset follows a normal distribution with constant volatility.

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
    `volatility_type='lognormal'` to hand this function a relative volatility and have it
    converted explicitly — see `_convert_volatility` for the approximation used.

    The normal distribution places positive probability on negative rates, which is
    precisely why this model came back into use for euro area and yen rates and why it,
    unlike Black's model, remains well defined at a zero or negative forward rate.

    At the money the price collapses to the closed form
    Annuity * Notional * volatility * SQRT(years_to_maturity / (2 * pi)).

    For more information, see: https://en.wikipedia.org/wiki/Bachelier_model

    Args:
        forward_rate (float): Forward rate of the underlying swap. May be zero or negative.
        strike_rate (float): Strike rate of the swaption. May be zero or negative.
        volatility (float): Volatility of the underlying swap, quoted on the basis given by
        `volatility_type` and defaulting to the normal (absolute, basis point) convention
        this model is defined in.
        years_to_maturity (float): years to maturity of the swaption.
        risk_free_rate (float): The risk-free interest rate.
        notional (float, optional): Notional amount of the swap. Default is 10,000,000.
        tenor (float | None, optional): Tenor of the underlying swap. Defaults to being equal to years to maturity.
        payment_frequency (int, optional): Number of fixed-leg payments per year on the underlying swap
        (e.g. 1 for annual, 2 for semi-annual, 4 for quarterly). Defaults to 2 (semi-annual).
        is_receiver (bool, optional): Boolean indicating if the swaption holder is receiver. Default is True.
        volatility_type (str, optional): The convention `volatility` is quoted on, either
        'normal' (absolute, this model's native basis) or 'lognormal' (relative to the
        forward rate), in which case it is converted before use. Defaults to 'normal'.

    Returns:
        tuple[float, float]: A tuple containing the price of the swaption and the payoff of the underlying option.

    Raises:
        ValueError: If volatility_type is not 'lognormal' or 'normal', or if a lognormal
            volatility is supplied around a forward rate of zero.
    """
    volatility = _convert_volatility(
        volatility=volatility,
        volatility_type=volatility_type,
        target_type="normal",
        forward_rate=forward_rate,
    )

    tenor = years_to_maturity if tenor is None else tenor

    d = (forward_rate - strike_rate) / (volatility * np.sqrt(years_to_maturity))

    if is_receiver:
        payoff = (strike_rate - forward_rate) * norm.cdf(-d) + volatility * np.sqrt(
            years_to_maturity
        ) * norm.pdf(-d)
    else:
        payoff = (forward_rate - strike_rate) * norm.cdf(d) + volatility * np.sqrt(
            years_to_maturity
        ) * norm.pdf(d)

    annuity = _get_annuity_factor(
        risk_free_rate=risk_free_rate,
        years_to_maturity=years_to_maturity,
        tenor=tenor,
        payment_frequency=payment_frequency,
    )
    swaption_price = notional * annuity * payoff

    return swaption_price, payoff
