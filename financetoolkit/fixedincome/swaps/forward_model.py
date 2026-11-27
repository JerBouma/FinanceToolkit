"""Forward Model

Forward contracts priced by no-arbitrage: forward rate agreements on a zero-coupon curve,
the convexity adjustment between short-rate futures and forward rates, FX forwards by
covered interest parity and equity forwards (Hull, 2022, ch. 4 to 7).
"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.utilities.error_model import check_option

# The days in a year of the day counts a forward rate agreement can accrue on. The curve's
# times are years of 365 days, so an ACT/360 accrual is 365 / 360 times as long.
DAY_COUNTS = {"act/365": 365, "act/360": 360}


def get_forward_rate_agreement(
    zero_curve: pd.Series,
    start: float,
    end: float,
    fixed_rate: float | None = None,
    notional: float = 10_000_000,
    day_count: str = "act/360",
) -> tuple[float, float]:
    """
    Calculate the rate and the value of a forward rate agreement (FRA) on a zero-coupon
    curve: the simple interest rate fixed today for a loan from `start` to `end`, and what
    an existing FRA is worth.

    A forward rate agreement fixes today the simple interest rate of a loan from `start` to
    `end`. At no arbitrage the FRA rate is the simple forward rate implied by the discount
    factors of the two dates, and a FRA that pays a fixed rate K and receives the floating
    rate is worth the difference of the two rates on the accrual, discounted from the end of
    the period (Hull, 2022, ch. 4.7).

    The formula is as follows:

    - τ = (end - start) * 365 / Days in Year, the accrual fraction of the day count
    - FRA Rate = (P(start) / P(end) - 1) / τ
    - Value = Notional * τ * (FRA Rate - K) * P(end)

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero rate
    r(t) of the curve, interpolated linearly between its maturities. The times are years of
    365 days, and Days in Year is 360 on ACT/360, the money market day count of USD and EUR
    interbank and overnight rates, or 365 on ACT/365. The FRA rate is simple interest on
    that day count. The value is that of a long FRA (paying the fixed rate); a short FRA is
    worth its negative.

    Referenced against:
    - QuantLib, ForwardRateAgreement: https://www.quantlib.org/reference/class_quant_lib_1_1_forward_rate_agreement.html

    Shares similar logic:
    - FinancePy, IborFRA: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/rates/ibor_fra.py#L23
    - rateslib, FRA: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.FRA.html

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        start (float): The start of the forward period, in years from today.
        end (float): The end of the forward period, in years from today.
        fixed_rate (float | None, optional): The fixed rate K of an existing FRA (decimal,
            simple interest on the day count). Defaults to None, which returns no value.
        notional (float, optional): The notional of the FRA. Defaults to 10,000,000.
        day_count (str, optional): The day count of the FRA rate. Options are: "act/360"
            (actual/360), "act/365" (actual/365). Defaults to "act/360".

    Returns:
        tuple[float, float]: The FRA rate (decimal) and the value of the long FRA in the
        currency of the notional (NaN without a fixed rate).

    Raises:
        ValueError: If the end is not after the start, or the day count is unknown.
    """
    check_option("day_count", day_count, DAY_COUNTS)
    if end <= start or start < 0:
        raise ValueError(
            f"A forward rate agreement runs from a start ({start}) of at least 0 to a later "
            f"end ({end})."
        )

    accrual = (end - start) * 365 / DAY_COUNTS[day_count]
    start_discount, end_discount = discount_factors(zero_curve, np.array([start, end]))
    fra_rate = float((start_discount / end_discount - 1) / accrual)
    value = (
        np.nan
        if fixed_rate is None
        else float(notional * accrual * (fra_rate - fixed_rate) * end_discount)
    )

    return fra_rate, value


def get_futures_convexity_adjustment(
    volatility: float,
    start: float,
    end: float,
    mean_reversion: float = 0.0,
) -> float:
    """
    Calculate the convexity adjustment between the rate of a short-rate futures contract
    (such as SOFR or Euribor futures) and the forward rate of the same period: how much the
    futures rate exceeds the forward rate because futures settle daily.

    A futures contract is settled every day while a forward rate agreement is settled once,
    so the futures rate is higher than the forward rate: gains on a short futures position
    are earned when rates are high and can be reinvested at those rates. The difference
    grows with the volatility of rates and roughly with the square of the time to the
    contract. Under the Ho-Lee model it is ½σ²t₁t₂; under the Hull-White model, with mean
    reversion a, it is smaller for long contracts (Hull, 2022, ch. 6.4; Kirikos and Novak,
    1997, "Convexity conundrums", Risk).

    The formula is as follows:

    - Ho-Lee (a = 0): Adjustment = σ² * t1 * t2 / 2
    - Hull-White: Adjustment = B(t1, t2) / (t2 - t1) * (B(t1, t2) * (1 - e^(-2 * a * t1))
      + 2 * a * B(0, t1)²) * σ² / (4 * a)
    - B(t, T) = (1 - e^(-a * (T - t))) / a
    - Forward Rate = Futures Rate - Adjustment

    where t1 is the start of the futures period (its last trading day), t2 its end in years,
    σ the absolute (normal) volatility of the short rate and a its mean reversion. The
    adjustment and both rates are continuously compounded: convert the futures rate
    (100 minus the price, simple interest on ACT/360) to continuous compounding before
    subtracting it.

    QuantLib's HullWhite.convexityBias is a different formula: it states the adjustment in
    simple interest and adds a term for the futures period itself (σ² * t1 * (t2 - t1) / 2
    in Ho-Lee), so it is larger, by about an eighth for a three-month contract two years
    out.

    Referenced against:
    - FinancePy, IborFuture.convexity: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/rates/ibor_future.py#L106

    Shares similar logic:
    - QuantLib, HullWhite.convexityBias: https://www.quantlib.org/reference/class_quant_lib_1_1_hull_white.html

    Args:
        volatility (float): The absolute volatility of the short rate (decimal a year,
            0.01 for 100 basis points).
        start (float): The start of the futures period, in years from today.
        end (float): The end of the futures period, in years from today.
        mean_reversion (float, optional): The mean reversion speed of the Hull-White model.
            Defaults to 0, the Ho-Lee model.

    Returns:
        float: The convexity adjustment (decimal, continuously compounded) to subtract from
        the futures rate.

    Raises:
        ValueError: If the end is not after the start.
    """
    if end <= start or start < 0:
        raise ValueError(
            f"A futures period runs from a start ({start}) of at least 0 to a later end ({end})."
        )

    if abs(mean_reversion) < 1e-10:  # noqa: PLR2004
        return volatility**2 * start * end / 2

    def b_function(length: float) -> float:
        return (1 - np.exp(-mean_reversion * length)) / mean_reversion

    period = b_function(end - start)

    return float(
        period
        / (end - start)
        * (
            period * (1 - np.exp(-2 * mean_reversion * start))
            + 2 * mean_reversion * b_function(start) ** 2
        )
        * volatility**2
        / (4 * mean_reversion)
    )


def get_fx_forward_rate(
    spot_rate: float,
    domestic_rate: float,
    foreign_rate: float,
    years_to_maturity: float,
    strike_rate: float | None = None,
    notional: float = 1_000_000,
) -> tuple[float, float]:
    """
    Calculate the forward exchange rate by covered interest parity, and the value of an
    existing FX forward: the exchange rate at which a currency can be bought for later
    delivery today, set by the interest rates of the two currencies.

    Exchanging currency today and depositing it, or depositing at home and exchanging at the
    forward rate, must give the same amount, so the forward rate is the spot rate carried at
    the difference between the two currencies' interest rates. The currency with the higher
    rate trades at a forward discount (Hull, 2022, ch. 5.10).

    The formula is as follows:

    - Forward Rate = Spot Rate * ((1 + r_d) / (1 + r_f))^T
    - Forward Points = (Forward Rate - Spot Rate) * 10,000
    - Value = Notional * (Forward Rate - K) / (1 + r_d)^T

    where the spot rate is in units of the domestic (quote) currency per unit of the foreign
    (base) currency, r_d and r_f are the annually compounded zero rates of the domestic and
    the foreign currency for T years, and K the strike of a forward that buys the notional in
    foreign currency. With continuously compounded rates this is S * e^((r_d - r_f) * T).
    The value is in the domestic currency.

    Referenced against:
    - rateslib, forward_fx: https://rateslib.com/py/en/3.0.x/api/rateslib.fx.forward_fx.html

    Shares similar logic:
    - QuantLib, FxForward: https://www.quantlib.org/reference/class_quant_lib_1_1_fx_forward.html
    - QuantLib, DiscountingFxForwardEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_discounting_fx_forward_engine.html
    - FinancePy, FXForward: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/fx/fx_forward.py#L20

    Args:
        spot_rate (float): The spot exchange rate, domestic currency per unit of foreign.
        domestic_rate (float): The annually compounded zero rate of the domestic currency.
        foreign_rate (float): The annually compounded zero rate of the foreign currency.
        years_to_maturity (float): The years until delivery.
        strike_rate (float | None, optional): The agreed rate K of an existing forward.
            Defaults to None, which returns no value.
        notional (float, optional): The amount of foreign currency bought forward.
            Defaults to 1,000,000.

    Returns:
        tuple[float, float]: The forward exchange rate and the value of the forward in the
        domestic currency (NaN without a strike).
    """
    domestic_discount = (1 + domestic_rate) ** -years_to_maturity
    forward_rate = (
        spot_rate
        * (1 + domestic_rate) ** years_to_maturity
        / (1 + foreign_rate) ** years_to_maturity
    )
    value = (
        np.nan
        if strike_rate is None
        else notional * (forward_rate - strike_rate) * domestic_discount
    )

    return float(forward_rate), float(value)


def get_equity_forward(
    spot_price: float,
    risk_free_rate: float,
    years_to_maturity: float,
    dividend_yield: float = 0.0,
    dividends: list[tuple[float, float]] | None = None,
    strike_price: float | None = None,
) -> tuple[float, float]:
    """
    Calculate the forward price of a stock or index and the value of an existing forward:
    the price agreed today for delivery later, the spot price carried at the risk-free rate
    less the dividends.

    Buying the stock today and holding it, financed at the risk-free rate, replicates a
    long forward; the dividends received along the way lower the forward price, either as
    a continuous yield (an index) or as known cash dividends (a stock) (Hull, 2022, ch. 5).

    The formula is as follows:

    - Forward Price = (S - PV(D)) * e^((r - q) * T)
    - PV(D) = SUM(D_i * e^(-r * t_i)), over the cash dividends paid before T
    - Value = (Forward Price - K) * e^(-r * T)

    where S is the spot price, r the continuously compounded risk-free rate, q the
    continuously compounded dividend yield (ln(1 + q) of an annual yield), D_i a cash
    dividend paid after t_i years and K the delivery price of a long forward. The value of
    a short forward is its negative.

    Referenced against:
    - FinancePy, EquityForward: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_forward.py#L19

    Shares similar logic:
    - QuantLib, EquityTotalReturnSwap: https://www.quantlib.org/reference/class_quant_lib_1_1_equity_total_return_swap.html

    Args:
        spot_price (float): The price of the stock or index today.
        risk_free_rate (float): The risk-free rate (decimal, continuously compounded).
        years_to_maturity (float): The years until delivery.
        dividend_yield (float, optional): The dividend yield (decimal, continuously
            compounded). Defaults to 0.
        dividends (list[tuple[float, float]] | None, optional): The cash dividends as
            (years from today, amount); those after delivery are ignored. Defaults to None.
        strike_price (float | None, optional): The delivery price K of an existing forward.
            Defaults to None, which returns no value.

    Returns:
        tuple[float, float]: The forward price and the value of a long forward per share
        (NaN without a strike price).
    """
    present_dividends = sum(
        amount * np.exp(-risk_free_rate * years)
        for years, amount in (dividends or [])
        if 0 <= years <= years_to_maturity
    )
    forward_price = (spot_price - present_dividends) * np.exp(
        (risk_free_rate - dividend_yield) * years_to_maturity
    )
    value = (
        np.nan
        if strike_price is None
        else (forward_price - strike_price)
        * np.exp(-risk_free_rate * years_to_maturity)
    )

    return float(forward_price), float(value)
