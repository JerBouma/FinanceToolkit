"""Convertible Bond Model

A convertible bond with credit risk, valued on a trinomial stock price tree with the split of
Tsiveriotis and Fernandes (1998): the part of the value that ends up paid in shares is
discounted at the risk-free rate, the part paid in cash by the issuer (coupons, redemption,
call and put prices) at the risk-free rate plus the issuer's credit spread.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.bonds import bond_model

# The spacing of the log stock price over the standard deviation of a step without a call (the
# λ of Kamrad and Ritchken, 1991): √(3/2) gives a middle probability of about 1/3.
STRETCH = np.sqrt(1.5)


def get_convertible_bond_value(
    stock_price: float,
    conversion_ratio: float,
    par_value: float,
    coupon_rate: float,
    years_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    credit_spread: float,
    dividend_yield: float = 0.0,
    frequency: int = 2,
    call_price: float | None = None,
    call_start: float = 0.0,
    put_price: float | None = None,
    put_dates: list[float] | None = None,
    steps: int = 500,
) -> pd.Series:
    """
    Calculate the value of a convertible bond with credit risk: what the bond with its
    option to convert into shares is worth, on a trinomial tree of the log stock price
    (Kamrad and Ritchken, 1991), with the Tsiveriotis-Fernandes split into an equity and a
    debt component.

    The holder can convert into `conversion_ratio` shares at any time, the issuer can call
    the bond from `call_start` (the holder then converts or takes the call price) and the
    holder can put it back on the put dates. Cash the issuer owes carries its credit risk,
    shares don't, so the two components are discounted at different rates and the value lies
    between the straight bond (the bond floor) and the conversion value plus the option to
    wait.

    The formula is as follows:

    - ν = r - q - σ² / 2, s = sqrt(σ² * Δt + ν² * Δt²), Δx = λ * s, nodes S * exp(j * Δx)
    - p_up = (s² / Δx² + ν * Δt / Δx) / 2, p_down = (s² / Δx² - ν * Δt / Δx) / 2,
      p_middle = 1 - s² / Δx²
    - At maturity: E = κ * S and D = 0 when κ * S > Redemption, else E = 0 and D = Redemption
    - Holding: E = exp(-r * Δt) * (p_up * E_up + p_middle * E_middle + p_down * E_down)
    - Holding: D = exp(-(r + cs) * Δt) * (p_up * D_up + p_middle * D_middle + p_down * D_down)
      + coupons in the step
    - Called (after call_start, when E + D > Call Price): E = 0 and D = Call Price
    - Put (on a put date, when E + D < Put Price): E = 0 and D = Put Price
    - Converted (when κ * S > E + D): E = κ * S and D = 0
    - Convertible Value = E + D

    where S is the stock price at the node, κ the conversion ratio, σ the volatility, r the
    risk-free rate, q the dividend yield, cs the credit spread, Δt = T / steps and the
    Redemption the par value plus the last coupon. The probabilities match the mean and
    variance of ln S over a step. λ is √(3/2) without a call. A call caps the value at
    max(Call Price, κ * S), a kink at the stock price Call Price / κ: between two nodes it
    makes the value swing with the number of steps, by up to 0.8 per 100 between 200 and
    1,000 steps on a binomial tree, as the barrier of a barrier option does. With a call, λ
    is therefore the smallest value of at least 1 that puts that stock price on a node, with
    ln(Call Price / (κ * S)) a whole number of Δx (Ritchken, 1995), after which the value
    moves by a few hundredths per 100 at most between 200 and 4,000 steps. The rates, the dividend yield and the
    credit spread are continuously compounded. A coupon paid within a step is discounted at
    r + cs to the start of that step and belongs to the holder who holds the bond over it, so
    converting gives up the coupons still to come, the last one included. Calls and puts are
    decided on the full value, so the call and put prices are the amounts paid, accrued
    coupon included, and a put date falls on the nearest step of the tree.

    The value is the full (dirty) value per bond. The Delta and Gamma are those of the three
    nodes of the tree's first step, per unit of the stock price. The Bond Floor is the straight bond, every
    coupon and the par value discounted at r + cs, and the Conversion Premium the value over
    the conversion value κ * S, minus one.

    For more information, see Tsiveriotis and Fernandes (1998), "Valuing convertible bonds
    with credit risk", The Journal of Fixed Income 8(2), 95-102, Kamrad and Ritchken (1991),
    "Multinomial approximating models for options with k state variables", Management Science
    37(12), 1640-1652, Ritchken (1995), "On pricing barrier options", The Journal of
    Derivatives 3(2), 19-28, and Hull (2022), Options, Futures, and Other Derivatives, 11th
    edition, chapter 26.

    QuantLib's engine and this tree agree without a credit spread and without a call, where
    both trees value the same split, within 0.01 per 100. QuantLib calls on its call dates
    only (daily in the cases checked), this tree on every step; a binomial tree called on
    every step approaches this tree's value as its steps grow (104.71 at 24,000 steps
    against 104.71 at 500). With a spread QuantLib's engine applies it in another way and
    its values differ by about 1% in the cases checked. With a spread the split is checked
    against its closed form for a zero-coupon convertible held to maturity, κ * S * N(d1) +
    Par Value * exp(-(r + cs) * T) * N(-d2) with the strike Par Value / κ.

    Referenced against:
    - QuantLib, ConvertibleFixedCouponBond: https://www.quantlib.org/reference/class_quant_lib_1_1_convertible_fixed_coupon_bond.html
    - QuantLib, ConvertibleZeroCouponBond: https://www.quantlib.org/reference/class_quant_lib_1_1_convertible_zero_coupon_bond.html
    - QuantLib, BinomialConvertibleEngine (CRR): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_convertible_engine.html

    Args:
        stock_price (float): The price of the stock today.
        conversion_ratio (float): The number of shares each bond converts into.
        par_value (float): The face value of the bond, repaid at maturity.
        coupon_rate (float): The annual coupon rate (in decimal).
        years_to_maturity (float): The number of years until the bond matures.
        risk_free_rate (float): The risk-free rate (decimal, continuously compounded).
        volatility (float): The volatility of the stock (decimal, annualised).
        credit_spread (float): The issuer's credit spread over the risk-free rate (decimal,
            continuously compounded).
        dividend_yield (float, optional): The dividend yield of the stock (decimal,
            continuously compounded). Defaults to 0.0.
        frequency (int, optional): The number of coupon payments per year. Defaults to 2.
        call_price (float | None, optional): The price at which the issuer can call the bond.
            Defaults to None, not callable.
        call_start (float, optional): The years from today after which the bond can be called.
            Defaults to 0.0.
        put_price (float | None, optional): The price at which the holder can put the bond.
            Defaults to None, not puttable.
        put_dates (list[float] | None, optional): The years from today on which the bond can be
            put. Defaults to None, at maturity only when a put price is given.
        steps (int, optional): The number of time steps of the tree. Defaults to 500.

    Returns:
        pd.Series: The Convertible Value, Equity Component, Debt Component, Conversion Value,
        Bond Floor, Conversion Premium, Delta and Gamma.

    Raises:
        ValueError: If the volatility, the years to maturity or the number of steps is not
            positive, or a probability of the tree is negative.
    """
    if volatility <= 0 or steps < 2:  # noqa: PLR2004
        raise ValueError("The volatility must be positive and the steps at least 2.")
    if years_to_maturity <= 0:
        raise ValueError(
            f"years_to_maturity must be positive, received {years_to_maturity!r}."
        )

    step = years_to_maturity / steps
    drift = risk_free_rate - dividend_yield - 0.5 * volatility**2
    # The standard deviation of ln S over a step around zero, with the drift's square, so that
    # the middle probability is never negative for a spacing at least this wide.
    deviation = np.sqrt(volatility**2 * step + (drift * step) ** 2)
    spacing = STRETCH * deviation
    if call_price is not None:
        # The call caps the value at max(Call Price, κ * S), a kink at κ * S = Call Price that
        # the tree resolves only on a node: off a node the value swings with the steps.
        call_distance = abs(np.log(call_price / (conversion_ratio * stock_price)))
        if (nodes := int(call_distance // deviation)) >= 1:
            spacing = call_distance / nodes

    second_moment = deviation**2 / spacing**2
    up_probability = 0.5 * (second_moment + drift * step / spacing)
    down_probability = 0.5 * (second_moment - drift * step / spacing)
    middle_probability = 1 - second_moment
    if min(up_probability, down_probability) < 0:
        raise ValueError(
            "The tree has no risk-neutral probabilities for these inputs: use more steps."
        )
    equity_discount = np.exp(-risk_free_rate * step)
    debt_rate = risk_free_rate + credit_spread
    debt_discount = np.exp(-debt_rate * step)

    coupon_times = (
        bond_model.get_cash_flow_periods(years_to_maturity, frequency) / frequency
    )
    coupon = par_value * coupon_rate / frequency
    put_steps = set()
    if put_price is not None:
        put_steps = {
            int(round(float(years) / step))
            for years in (put_dates if put_dates is not None else [years_to_maturity])
        }

    def stock_prices(level: int) -> np.ndarray:
        return stock_price * np.exp(spacing * np.arange(level, -level - 1, -1))

    def roll_back(values: np.ndarray, discount: float) -> np.ndarray:
        return discount * (
            up_probability * values[:-2]
            + middle_probability * values[1:-1]
            + down_probability * values[2:]
        )

    def exercise(level: int, equity: np.ndarray, debt: np.ndarray) -> None:
        conversion = conversion_ratio * stock_prices(level)
        if call_price is not None and level * step >= call_start - 1e-12:
            called = equity + debt > call_price
            equity[called], debt[called] = 0.0, call_price
        if level in put_steps:
            put = equity + debt < put_price
            equity[put], debt[put] = 0.0, put_price
        converted = conversion > equity + debt
        equity[converted], debt[converted] = conversion[converted], 0.0

    conversion = conversion_ratio * stock_prices(steps)
    redemption = par_value + coupon
    equity = np.where(conversion > redemption, conversion, 0.0)
    debt = np.where(conversion > redemption, 0.0, float(redemption))
    if put_price is not None and steps in put_steps:
        put = equity + debt < put_price
        equity[put], debt[put] = 0.0, put_price

    first_step = np.full(3, np.nan)
    for level in range(steps - 1, -1, -1):
        equity = roll_back(equity, equity_discount)
        start = level * step
        # The last coupon is part of the redemption, which conversion at maturity gives up.
        in_step = coupon_times[
            (coupon_times > start + 1e-12)
            & (coupon_times <= start + step + 1e-12)
            & (coupon_times < years_to_maturity - 1e-9)
        ]
        debt = (
            roll_back(debt, debt_discount)
            + coupon * np.exp(-debt_rate * (in_step - start)).sum()
        )
        exercise(level, equity, debt)
        if level == 1:
            first_step = equity + debt

    # The three nodes of the first step, up, middle and down, give the Delta and the Gamma.
    prices = stock_prices(1)
    delta_up = (first_step[0] - first_step[1]) / (prices[0] - prices[1])
    delta_down = (first_step[1] - first_step[2]) / (prices[1] - prices[2])

    conversion_value = conversion_ratio * stock_price
    convertible_value = float(equity[0] + debt[0])

    return pd.Series(
        {
            "Convertible Value": convertible_value,
            "Equity Component": float(equity[0]),
            "Debt Component": float(debt[0]),
            "Conversion Value": conversion_value,
            "Bond Floor": float(
                coupon * np.exp(-debt_rate * coupon_times).sum()
                + par_value * np.exp(-debt_rate * years_to_maturity)
            ),
            "Conversion Premium": convertible_value / conversion_value - 1,
            "Delta": float((first_step[0] - first_step[2]) / (prices[0] - prices[2])),
            "Gamma": float((delta_up - delta_down) / ((prices[0] - prices[2]) / 2)),
        }
    )
