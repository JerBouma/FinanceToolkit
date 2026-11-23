"""Convertible Bond Model

A convertible bond with credit risk, valued on a binomial stock price tree with the split of
Tsiveriotis and Fernandes (1998): the part of the value that ends up paid in shares is
discounted at the risk-free rate, the part paid in cash by the issuer (coupons, redemption,
call and put prices) at the risk-free rate plus the issuer's credit spread.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.bonds import bond_model


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
    The value of a convertible bond with credit risk on a Cox-Ross-Rubinstein binomial tree of
    the stock price, with the Tsiveriotis-Fernandes split into an equity and a debt component.
    The holder can convert into `conversion_ratio` shares at any time, the issuer can call the
    bond from `call_start` (the holder then converts or takes the call price) and the holder
    can put it back on the put dates. Cash the issuer owes carries its credit risk, shares
    don't, so the two components are discounted at different rates and the value lies between
    the straight bond (the bond floor) and the conversion value plus the option to wait.

    The formula is as follows:

    - u = exp(σ * sqrt(Δt)), d = 1 / u, p = (exp((r - q) * Δt) - d) / (u - d)
    - At maturity: E = κ * S and D = 0 when κ * S > Redemption, else E = 0 and D = Redemption
    - Holding: E = exp(-r * Δt) * (p * E_up + (1 - p) * E_down)
    - Holding: D = exp(-(r + cs) * Δt) * (p * D_up + (1 - p) * D_down) + coupons in the step
    - Called (after call_start, when E + D > Call Price): E = 0 and D = Call Price
    - Put (on a put date, when E + D < Put Price): E = 0 and D = Put Price
    - Converted (when κ * S > E + D): E = κ * S and D = 0
    - Convertible Value = E + D

    where S is the stock price at the node, κ the conversion ratio, σ the volatility, r the
    risk-free rate, q the dividend yield, cs the credit spread, Δt = T / steps and the
    Redemption the par value plus the last coupon. The rates, the dividend yield and the
    credit spread are continuously compounded. A coupon paid within a step is discounted at
    r + cs to the start of that step and belongs to the holder who holds the bond over it, so
    converting gives up the coupons still to come, the last one included. Calls and puts are
    decided on the full value, so the call and put prices are the amounts paid, accrued
    coupon included, and a put date falls on the nearest step of the tree.

    The value is the full (dirty) value per bond. The Delta and Gamma are those of the tree's
    first two steps, per unit of the stock price. The Bond Floor is the straight bond, every
    coupon and the par value discounted at r + cs, and the Conversion Premium the value over
    the conversion value κ * S, minus one.

    For more information, see Tsiveriotis and Fernandes (1998), "Valuing convertible bonds
    with credit risk", The Journal of Fixed Income 8(2), 95-102, and Hull (2022), Options,
    Futures, and Other Derivatives, 11th edition, chapter 26. Referenced against QuantLib's
    ConvertibleFixedCouponBond and ConvertibleZeroCouponBond with the
    BinomialCRRConvertibleEngine (https://www.quantlib.org) without a credit spread, where the
    two trees are the same; with a spread QuantLib's engine applies it in another way and its
    values differ by about 1% in the cases checked. With a spread the split is checked against
    its closed form for a zero-coupon convertible held to maturity,
    κ * S * N(d1) + Par Value * exp(-(r + cs) * T) * N(-d2) with the strike Par Value / κ.

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
        steps (int, optional): The number of steps of the tree. Defaults to 500.

    Returns:
        pd.Series: The Convertible Value, Equity Component, Debt Component, Conversion Value,
        Bond Floor, Conversion Premium, Delta and Gamma.

    Raises:
        ValueError: If the volatility or the number of steps is not positive, or the risk-
            neutral probability of the tree falls outside 0 and 1.
    """
    if volatility <= 0 or steps < 2:  # noqa: PLR2004
        raise ValueError("The volatility must be positive and the steps at least 2.")

    step = years_to_maturity / steps
    up = np.exp(volatility * np.sqrt(step))
    down = 1 / up
    probability = (np.exp((risk_free_rate - dividend_yield) * step) - down) / (
        up - down
    )
    if not 0 < probability < 1:
        raise ValueError(
            "The tree has no risk-neutral probability for these inputs: use more steps."
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
        return stock_price * up ** np.arange(level, -level - 1, -2, dtype=float)

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

    values = {}
    for level in range(steps - 1, -1, -1):
        equity = equity_discount * (
            probability * equity[:-1] + (1 - probability) * equity[1:]
        )
        start = level * step
        # The last coupon is part of the redemption, which conversion at maturity gives up.
        in_step = coupon_times[
            (coupon_times > start + 1e-12)
            & (coupon_times <= start + step + 1e-12)
            & (coupon_times < years_to_maturity - 1e-9)
        ]
        debt = (
            debt_discount * (probability * debt[:-1] + (1 - probability) * debt[1:])
            + coupon * np.exp(-debt_rate * (in_step - start)).sum()
        )
        exercise(level, equity, debt)
        if level <= 2:  # noqa: PLR2004
            values[level] = equity + debt

    level_one, level_two = stock_prices(1), stock_prices(2)
    delta_up = (values[2][0] - values[2][1]) / (level_two[0] - level_two[1])
    delta_down = (values[2][1] - values[2][2]) / (level_two[1] - level_two[2])

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
            "Delta": float(
                (values[1][0] - values[1][1]) / (level_one[0] - level_one[1])
            ),
            "Gamma": float(
                (delta_up - delta_down) / ((level_two[0] - level_two[2]) / 2)
            ),
        }
    )
