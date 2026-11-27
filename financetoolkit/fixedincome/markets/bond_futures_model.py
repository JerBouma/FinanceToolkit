"""Bond Forward and Futures Model

Bond forward prices and implied repo rates financed in the repo market, and the US Treasury
futures conversion factor, basis, implied repo rate and cheapest to deliver (Tuckman and
Serrat, 2022, Fixed Income Securities, ch. 13 and 14; Burghardt and Belton, 2005, The
Treasury Bond Basis; Hull, 2022, Options, Futures and Other Derivatives, ch. 6.2).
"""

__docformat__ = "google"

import numpy as np

from financetoolkit.utilities.error_model import check_option

# Repo rates are simple interest on actual/360, the US money market convention.
DAYS_IN_YEAR = 360

# The yield of CME's conversion factors: 6% a year, compounded semi-annually.
CONVERSION_FACTOR_YIELD = 0.06

# How CME rounds the time from the first day of the delivery month to the maturity of a
# deliverable when calculating its conversion factor: to whole months for the 2-, 3- and
# 5-year note futures, to whole quarters for the 10-year notes and the bonds.
MONTH_ROUNDING = {"month": 1, "quarter": 3}


def get_bond_forward_price(
    clean_price: float,
    accrued_interest: float,
    forward_accrued_interest: float,
    repo_rate: float,
    days: int,
    coupons: list[tuple[float, int]] | None = None,
) -> float:
    """
    Calculate the forward price of a bond financed in the repo market: the clean price at
    which buying the bond today with borrowed cash and selling it forward breaks even
    (Tuckman and Serrat, 2022, ch. 13).

    Buying the bond costs its dirty price, borrowed at the repo rate until the forward date.
    The coupons paid in between are received and earn the repo rate until the forward date,
    and the buyer on the forward date pays the accrued interest of that date on top of the
    forward clean price.

    The formula is as follows:

    - Forward Price = (P + AI_0) * (1 + r * d / 360) - Σ_k c_k * (1 + r * (d - d_k) / 360) - AI_T

    where P is the clean price, AI_0 and AI_T the accrued interest today and on the forward
    date, r the repo rate (simple interest, actual/360), d the days to the forward date and
    c_k the coupons paid d_k days from today. The difference P - Forward Price is the carry
    (the drop): positive when the coupon income exceeds the financing cost.

    Referenced against:
    - QuantLib, BondForward: https://www.quantlib.org/reference/class_quant_lib_1_1_bond_forward.html

    Shares similar logic:
    - rateslib, FixedRateBond.fwd_from_repo: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.FixedRateBond.html#rateslib.instruments.FixedRateBond.fwd_from_repo

    Args:
        clean_price (float): The clean price per 100 of par today.
        accrued_interest (float): The accrued interest per 100 of par today.
        forward_accrued_interest (float): The accrued interest per 100 of par on the forward
            date.
        repo_rate (float): The repo rate (decimal, simple interest on actual/360).
        days (int): The days from today to the forward date.
        coupons (list[tuple[float, int]] | None, optional): The coupons per 100 of par paid
            after today up to and including the forward date, each with the days from today
            to its payment. Defaults to None, no coupon in between.

    Returns:
        float: The forward clean price per 100 of par.
    """
    coupons = coupons or []
    financed = (clean_price + accrued_interest) * (1 + repo_rate * days / DAYS_IN_YEAR)
    income = sum(
        coupon * (1 + repo_rate * (days - paid) / DAYS_IN_YEAR)
        for coupon, paid in coupons
    )

    return financed - income - forward_accrued_interest


def get_implied_repo_rate(
    clean_price: float,
    accrued_interest: float,
    forward_price: float,
    forward_accrued_interest: float,
    days: int,
    coupons: list[tuple[float, int]] | None = None,
) -> float:
    """
    Calculate the implied repo rate: the repo rate at which buying a bond today and selling
    it forward at a given forward (or futures invoice) price breaks even. It is the return,
    simple interest on actual/360, of the cash-and-carry trade (Tuckman and Serrat, 2022,
    ch. 13; Burghardt and Belton, 2005, ch. 2).

    The formula is as follows:

    - Implied Repo Rate = (F + AI_T + Σ_k c_k - (P + AI_0)) / ((P + AI_0) * d / 360 - Σ_k c_k * (d - d_k) / 360)

    which inverts get_bond_forward_price: the coupons received in between earn the implied
    repo rate until the forward date. Without coupons in between it is
    ((F + AI_T) / (P + AI_0) - 1) * 360 / d.

    Shares similar logic:
    - rateslib, FixedRateBond.repo_from_fwd: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.FixedRateBond.html#rateslib.instruments.FixedRateBond.repo_from_fwd
    - FinancePy, BondFuture.implied_repo_rate: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_future.py#L288

    Args:
        clean_price (float): The clean price per 100 of par today.
        accrued_interest (float): The accrued interest per 100 of par today.
        forward_price (float): The forward clean price per 100 of par, or for a futures
            contract the futures price times the conversion factor.
        forward_accrued_interest (float): The accrued interest per 100 of par on the forward
            date.
        days (int): The days from today to the forward date.
        coupons (list[tuple[float, int]] | None, optional): The coupons per 100 of par paid
            after today up to and including the forward date, each with the days from today
            to its payment. Defaults to None.

    Returns:
        float: The implied repo rate (decimal, simple interest on actual/360).
    """
    coupons = coupons or []
    dirty = clean_price + accrued_interest
    income = sum(coupon for coupon, _ in coupons)
    reinvested = sum(coupon * (days - paid) for coupon, paid in coupons) / DAYS_IN_YEAR
    denominator = dirty * days / DAYS_IN_YEAR - reinvested
    if days <= 0 or denominator == 0:
        return np.nan

    return (forward_price + forward_accrued_interest + income - dirty) / denominator


def get_conversion_factor(
    coupon: float, years: int, months: int, maturity_rounding: str = "quarter"
) -> float:
    """
    Calculate the conversion factor of a deliverable into a US Treasury futures contract
    with CME Group's formula: how many futures contracts' worth one deliverable bond is, the
    price per 1 of par of the deliverable at a yield of 6%, compounded semi-annually, on the
    first day of the delivery month, with its time to maturity rounded down (CME Group,
    2022, "Calculating U.S. Treasury Futures Conversion Factors"; Hull, 2022, ch. 6.2).

    The invoice the short receives is the futures price times the conversion factor plus the
    accrued interest, so the factor puts deliverables of different coupons and maturities on
    one scale.

    The formula is as follows:

    - v = z when z < 7, otherwise 3 for quarters (z - 6 for months)
    - a = 1 / 1.03^(v / 6)
    - b = (c / 2) * (6 - v) / 6
    - C = 1 / 1.03^(2n) when z < 7, otherwise 1 / 1.03^(2n + 1)
    - d = (c / 0.06) * (1 - C)
    - Conversion Factor = a * (c / 2 + C + d) - b, rounded to four decimals

    where c is the coupon (decimal), n the whole years and z the months from the first day
    of the delivery month to maturity, rounded down to whole quarters for the 10-year
    (ZN), Ultra 10-year (TN), bond (ZB) and Ultra bond (UB) futures and to whole months for
    the 2-, 3- and 5-year note futures (ZT, Z3N, ZF). Hull's examples: a 10% coupon with 20
    years and 2 months has a conversion factor of 1.4623, an 8% coupon with 18 years and 4
    months 1.2199.

    Referenced against:
    - rateslib, BondFuture.cfs: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.BondFuture.html#rateslib.instruments.BondFuture.cfs

    Shares similar logic:
    - FinancePy, BondFuture.conversion_factor: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_future.py#L68

    Args:
        coupon (float): The annual coupon rate of the deliverable (decimal).
        years (int): The whole years from the first day of the delivery month to maturity.
        months (int): The whole months beyond those years (0 to 11), before rounding.
        maturity_rounding (str, optional): How the time to maturity is rounded down. Options
            are: "quarter" (whole quarters), "month" (whole months), see above. Defaults to
            "quarter".

    Returns:
        float: The conversion factor, rounded to four decimals.

    Raises:
        InvalidArgumentError: If the maturity rounding is not "quarter" or "month".
    """
    check_option("maturity_rounding", maturity_rounding, MONTH_ROUNDING)

    step = MONTH_ROUNDING[maturity_rounding]
    z = months // step * step
    # From seven months on, the first coupon falls in the second half year.
    second_half = z >= 7  # noqa: PLR2004
    v = (3 if maturity_rounding == "quarter" else z - 6) if second_half else z
    half_yield = 1 + CONVERSION_FACTOR_YIELD / 2

    a = 1 / half_yield ** (v / 6)
    b = coupon / 2 * (6 - v) / 6
    discount = 1 / half_yield ** (2 * years + 1 if second_half else 2 * years)
    d = coupon / CONVERSION_FACTOR_YIELD * (1 - discount)

    return round(a * (coupon / 2 + discount + d) - b, 4)


def get_futures_basis(
    clean_price: float,
    accrued_interest: float,
    futures_price: float,
    conversion_factor: float,
    delivery_accrued_interest: float,
    repo_rate: float,
    days: int,
    coupons: list[tuple[float, int]] | None = None,
) -> dict[str, float]:
    """
    Calculate the basis of a deliverable into a bond futures contract: what delivering it
    costs and earns against the futures price (Burghardt and Belton, 2005, ch. 1 and 2;
    Tuckman and Serrat, 2022, ch. 14).

    The formula is as follows:

    - Invoice Price = F * CF + AI_T, per 100 of par
    - Gross Basis = P - F * CF
    - Carry = P - Forward Price, the coupon income less the financing cost to delivery
      (see get_bond_forward_price)
    - Net Basis = Gross Basis - Carry = Forward Price - F * CF
    - Implied Repo Rate = the repo rate at which buying the deliverable and delivering it
      against the futures breaks even (see get_implied_repo_rate with F * CF as the forward
      price)

    where P is the clean price, F the futures price, CF the conversion factor (see
    get_conversion_factor) and AI_T the accrued interest on the delivery date. The cheapest
    to deliver is the deliverable with the highest implied repo rate, which is the one with
    the lowest net basis. The net basis is the value of the short's options on what and
    when to deliver, which this leaves out.

    Shares similar logic:
    - FinancePy, BondFuture.gross_basis: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_future.py#L209
    - FinancePy, BondFuture.net_basis: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_future.py#L237
    - rateslib, BondFuture.gross_basis: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.BondFuture.html#rateslib.instruments.BondFuture.gross_basis
    - rateslib, BondFuture.net_basis: https://rateslib.com/py/en/3.0.x/api/rateslib.instruments.BondFuture.html#rateslib.instruments.BondFuture.net_basis

    Args:
        clean_price (float): The clean price of the deliverable per 100 of par.
        accrued_interest (float): Its accrued interest per 100 of par today.
        futures_price (float): The futures price per 100 of par.
        conversion_factor (float): The conversion factor of the deliverable.
        delivery_accrued_interest (float): Its accrued interest per 100 of par on the
            delivery date.
        repo_rate (float): The repo rate to delivery (decimal, simple interest on
            actual/360).
        days (int): The days from today to the delivery date.
        coupons (list[tuple[float, int]] | None, optional): The coupons per 100 of par paid
            before delivery, each with the days from today to its payment. Defaults to None.

    Returns:
        dict[str, float]: The Invoice Price, Gross Basis, Carry, Net Basis (per 100 of par)
        and the Implied Repo Rate (decimal).
    """
    converted = futures_price * conversion_factor
    forward_price = get_bond_forward_price(
        clean_price,
        accrued_interest,
        delivery_accrued_interest,
        repo_rate,
        days,
        coupons,
    )

    return {
        "Invoice Price": converted + delivery_accrued_interest,
        "Gross Basis": clean_price - converted,
        "Carry": clean_price - forward_price,
        "Net Basis": forward_price - converted,
        "Implied Repo Rate": get_implied_repo_rate(
            clean_price,
            accrued_interest,
            converted,
            delivery_accrued_interest,
            days,
            coupons,
        ),
    }
