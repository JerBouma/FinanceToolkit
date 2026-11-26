"""Bond Data Model"""

__docformat__ = "google"

import re
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from financetoolkit.fixedincome.bonds import bond_model
from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.utilities.error_model import check_option

# US Treasuries and corporate bonds pay their coupon twice a year.
FREQUENCY = 2

# The day count conventions of get_years_to_maturity: actual/actual for US Treasuries and
# 30/360 for US corporate bonds.
DAY_COUNTS = ("act/act", "30/360")

# A bond fund names a holding "ISSUER [144A] COUPON% MM/DD/YYYY", sometimes followed by a
# reference in brackets ("(JPM)", "(BBG02444V9S2)"). The coupon is printed to two decimals,
# so 2.125% reads 2.13%.
BOND_NAME = re.compile(
    r"^(?P<issuer>.*?)\s+(?P<coupon>\d+(?:\.\d+)?)\s*%\s+"
    r"(?P<maturity>\d{1,2}/\d{1,2}/\d{4})\s*(?:\([^)]*\))?\s*$"
)
RULE_144A = re.compile(r"\s+144A\b", re.IGNORECASE)
# Programme and coupon markers some funds append to the issuer, such as "MTN" (a medium
# term note) or "(FXD" (a fixed coupon), so a bond is listed under its issuer's name.
ISSUER_MARKERS = re.compile(r"\s+(MTN\b.*|\((FXD|FIXED)\b[^)]*\)?)$", re.IGNORECASE)

# Holdings whose yield to maturity the coupon and price do not determine are left out, by
# the reason given: a floating rate note resets its coupon, as does a bank's fixed-to-floating
# note ("FXD-FRN") from its switch date, an inflation-linked bond is priced on its
# inflation-adjusted principal and a mortgage pool repays its principal early.
EXCLUDED = {
    "floating rate": re.compile(
        r"\b(FRN|FLOATING|FLTG?|VAR|VRN|SOFR|SOFRRATE|FXD-FRN|FX-FRN|FIX-TO-FLOAT)\b",
        re.IGNORECASE,
    ),
    "inflation-linked": re.compile(r"\b(TIPS|CPI|INFLATION|INFL|I/L)\b", re.IGNORECASE),
    "mortgage pool": re.compile(
        r"\b(FNMA|FHLMC|GNMA|UMBS|FGOLD|FNCL|FNCI|G2SF|TBA)\b|FEDERAL NATIONAL MORTGAGE|"
        r"FEDERAL HOME LOAN MORTGAGE|GOVERNMENT NATIONAL MORTGAGE|"
        r"\b(FANNIE MAE|FREDDIE MAC|GINNIE MAE)\b.*\bPOOL\b",
        re.IGNORECASE,
    ),
    "perpetual": re.compile(r"\bPERP", re.IGNORECASE),
}

# A price outside this range (per 100 of par) is a cash line or a data error, not a bond.
PRICE_RANGE = (1.0, 300.0)


def parse_bond_name(name: str) -> dict | None:
    """
    Reads the issuer, coupon and maturity from the name a bond fund gives a holding.

    Args:
        name (str): The holding's name, e.g. "ANHEUSER-BUSCH COMPANIES LLC 4.90% 02/01/2046".

    Returns:
        dict | None: The Issuer, Coupon (decimal), Maturity (date) and whether it is a Rule
        144A bond, or None when the name is not that of a fixed-coupon bond.
    """
    match = BOND_NAME.match(str(name).strip())
    if not match:
        return None

    try:
        maturity = pd.to_datetime(match["maturity"], format="%m/%d/%Y").date()
    except ValueError:
        return None

    issuer = match["issuer"].strip()
    rule_144a = bool(RULE_144A.search(issuer))

    return {
        "Issuer": ISSUER_MARKERS.sub("", RULE_144A.sub("", issuer).strip()).strip(),
        "Coupon": float(match["coupon"]) / 100,
        "Maturity": maturity,
        "Rule 144A": rule_144a,
    }


def exclusion_reason(name: str, maturity: date | None = None) -> str | None:
    """
    The reason a holding is left out of the yield calculations, if any.

    Args:
        name (str): The holding's name.
        maturity (date | None): The parsed maturity; a date centuries away is a placeholder
            for a perpetual bond (a century bond, maturing in 2115, is kept).

    Returns:
        str | None: The reason, or None when the bond is kept.
    """
    for reason, pattern in EXCLUDED.items():
        if pattern.search(str(name)):
            return reason

    if maturity is not None and maturity.year >= 2200:  # noqa: PLR2004
        return "perpetual"

    return None


def _days_30_360(start: date, end: date) -> int:
    """
    The days between two dates on the 30/360 Bond Basis (ISDA 2006, 4.16(f)): every month
    counts 30 days, the 31st counts as the 30th, and an end date on the 31st counts as the
    30th only when the start date is the 30th or 31st.

    Args:
        start (date): The first date.
        end (date): The last date.

    Returns:
        int: The days between them.
    """
    start_day = min(start.day, 30)
    end_day = 30 if end.day == 31 and start_day == 30 else end.day  # noqa: PLR2004

    return (
        360 * (end.year - start.year)
        + 30 * (end.month - start.month)
        + (end_day - start_day)
    )


def get_years_to_maturity(
    settlement: date,
    maturity: date,
    frequency: int = FREQUENCY,
    day_count: str = "act/act",
) -> float:
    """
    The time to maturity in coupon periods, counted on the bond's actual coupon dates.

    The coupon dates run back from maturity in steps of 12 / frequency months. A bond that
    matures on the last day of a month pays every coupon on the last day of its month (the
    end-of-month rule): a bond maturing on 30 April pays on 31 October, one maturing on 31
    August on the last day of February. The whole periods after the next coupon date count
    as whole periods, and the current period counts the share of its days that is still to
    come. Dividing by the frequency gives the years `bond_model` takes, so its coupon
    schedule falls on the bond's real coupon dates and its accrued interest is the elapsed
    share of the current period.

    The days are counted with the bond's day count convention:

    - "act/act": the actual days, as a share of the actual days of the coupon period
      (actual/actual ICMA), the convention of US Treasuries.
    - "30/360": 30 days to every month (the 30/360 Bond Basis), the elapsed days as a
      share of the 360 / frequency days of a coupon period. US corporate bonds mostly use
      30/360 US, which also counts the last day of February as the 30th; the two differ by
      at most a few days of accrual around February.

    Referenced against:
    - QuantLib, FixedRateBond: https://www.quantlib.org/reference/class_quant_lib_1_1_fixed_rate_bond.html

    Args:
        settlement (date): The date the price applies to.
        maturity (date): The maturity date.
        frequency (int): The number of coupons per year: 1, 2, 3, 4, 6 or 12. Defaults to 2.
        day_count (str): The day count convention. Options are: "act/act" (actual/actual, the
            Treasury's), "30/360" (the 30/360 Bond Basis of corporate bonds). Defaults to
            "act/act".

    Returns:
        float: The years to maturity, above 0.

    Raises:
        ValueError: If the day count convention is not "act/act" or "30/360", the
            settlement is on or after maturity (a matured bond has no years left) or the
            frequency is not one of 1, 2, 3, 4, 6 or 12.
    """
    check_option("day_count", day_count, DAY_COUNTS)
    if maturity <= settlement:
        raise ValueError(
            f"The settlement date {settlement} is on or after the maturity date {maturity}: "
            "a matured bond has no years to maturity left."
        )

    previous_coupon, coupons = get_coupon_dates(settlement, maturity, frequency)
    periods = len(coupons)
    next_coupon = coupons[0]
    if day_count == "30/360":
        # Counted from the previous coupon date, so the accrued interest is the 30/360
        # accrued interest also where a period does not count 180 days (31 August to 28
        # February counts 178).
        remaining = 1 - _days_30_360(previous_coupon, settlement) / (360 / frequency)
    else:
        remaining = (next_coupon - settlement).days / (
            next_coupon - previous_coupon
        ).days

    return (periods - 1 + remaining) / frequency


def get_coupon_dates(
    settlement: date, maturity: date, frequency: int = FREQUENCY
) -> tuple[date, list[date]]:
    """
    The coupon date on or before the settlement date and the coupon dates after it, up to
    and including maturity.

    The coupon dates run back from maturity in steps of 12 / frequency months. A bond that
    matures on the last day of a month pays every coupon on the last day of its month (the
    end-of-month rule): a note maturing on 30 April pays on 31 October, a floating rate note
    maturing on 31 July pays on 31 October, 31 January and 30 April. A settlement on a coupon
    date has that date as its previous coupon date, as the coupon is paid to the seller.

    Args:
        settlement (date): The settlement date, before maturity.
        maturity (date): The maturity date.
        frequency (int): The number of coupons per year: 1, 2, 3, 4, 6 or 12. Defaults to 2.

    Returns:
        tuple[date, list[date]]: The previous coupon date and the remaining coupon dates in
        order, the last one being the maturity date.

    Raises:
        ValueError: If the frequency does not divide the year into whole months (1, 2, 3,
            4, 6 or 12).
    """
    if frequency not in (1, 2, 3, 4, 6, 12):
        raise ValueError(
            f"The frequency must be 1, 2, 3, 4, 6 or 12 coupons a year, got {frequency}."
        )
    months = 12 // frequency
    end_of_month = maturity == maturity + relativedelta(day=31)

    def coupon_date(periods: int) -> date:
        # relativedelta keeps the day where the month has it (30 April less six months is
        # 30 October); the end-of-month rule moves it to the month's last day (31 October).
        coupon = maturity - relativedelta(months=months * periods)
        return coupon + relativedelta(day=31) if end_of_month else coupon

    periods = 1
    while coupon_date(periods) > settlement:
        periods += 1

    return coupon_date(periods), [
        coupon_date(remaining) for remaining in range(periods - 1, -1, -1)
    ]


def get_treasury_accrued_interest(
    coupon: float, settlement: date, maturity: date
) -> float:
    """
    The interest accrued per 100 of par on a semi-annual Treasury note or bond on its actual
    coupon dates, actual/actual: the half coupon times the elapsed share of the days of the
    current coupon period (31 CFR 356, Appendix B, I.D). It is `bond_model.get_accrued_interest`
    on the years to maturity of get_years_to_maturity.

    Args:
        coupon (float): The annual coupon rate (decimal).
        settlement (date): The settlement date.
        maturity (date): The maturity date.

    Returns:
        float: The accrued interest per 100 of par, 0 on a coupon date.

    Raises:
        ValueError: If the settlement is on or after maturity.
    """
    years = get_years_to_maturity(settlement, maturity)

    return bond_model.get_accrued_interest(100, coupon, years, FREQUENCY)


def get_bond_analytics(
    coupon: float,
    years_to_maturity: float,
    price: float,
    frequency: int = FREQUENCY,
) -> tuple[float, float]:
    """
    The yield to maturity and modified duration of a fixed-coupon bond from its clean price,
    with `bond_model`.

    Args:
        coupon (float): The annual coupon rate (decimal).
        years_to_maturity (float): The years to maturity, see get_years_to_maturity.
        price (float): The clean price per 100 of par.
        frequency (int): The number of coupons per year, at least 1. Defaults to 2.

    Returns:
        tuple[float, float]: The yield to maturity (nominal, compounded `frequency` times a
        year) and the modified duration, NaN when no yield prices the bond.
    """
    if not np.isfinite(years_to_maturity) or not np.isfinite(price):
        return np.nan, np.nan

    yield_to_maturity = bond_model.get_yield_to_maturity(
        100, coupon, years_to_maturity, price, frequency, guess=max(coupon, 0.03)
    )
    if not np.isfinite(yield_to_maturity):
        return np.nan, np.nan

    duration = bond_model.get_modified_duration(
        100, coupon, years_to_maturity, yield_to_maturity, frequency
    )
    return yield_to_maturity, duration


def get_bill_price(years_to_maturity: float, yield_to_maturity: float) -> float:
    """
    The price of a Treasury bill per 100 of face value at a bond-equivalent yield, the
    pricing that `bond_model.get_bond_equivalent_yield` inverts (31 CFR 356, Appendix B).

    The formula is as follows:

    - Price = 100 / (1 + y * t), for half a year (182.5 days) or less
    - Price = 100 / ((1 + y / 2) * (1 + y * (t - 0.5))), beyond half a year

    where y is the bond-equivalent yield and t the days to maturity divided by 365. Up to
    half a year the yield is simple interest; beyond it an equivalent note would have paid
    a coupon after half a year, so the yield compounds once.

    Referenced against:
    - QuantLib, ZeroCouponBond: https://www.quantlib.org/reference/class_quant_lib_1_1_zero_coupon_bond.html

    Args:
        years_to_maturity (float): The days to maturity divided by 365.
        yield_to_maturity (float): The bond-equivalent yield (decimal).

    Returns:
        float: The price per 100 of face value.
    """
    if years_to_maturity <= 0.5:  # noqa: PLR2004
        return 100 / (1 + yield_to_maturity * years_to_maturity)

    return 100 / (
        (1 + yield_to_maturity / 2)
        * (1 + yield_to_maturity * (years_to_maturity - 0.5))
    )


def get_bill_measures(years_to_maturity: float, yield_to_maturity: float) -> dict:
    """
    The duration and convexity measures of a Treasury bill at its bond-equivalent yield,
    on the bill's own pricing (see get_bill_price) rather than that of a semi-annual bond.

    A bill pays once, at maturity, so its Macaulay duration is its maturity. Up to half a
    year the bond-equivalent yield is simple interest, so the modified duration is
    t / (1 + y t) rather than the t / (1 + y / 2) of a semi-annual bond (0.2468 instead of
    0.2444 for a 91-day bill at 4%).

    The formula is as follows:

    - Macaulay's Duration = t
    - Modified Duration = -dP/dy / P = t / (1 + y t) up to half a year, and
      0.5 / (1 + y / 2) + (t - 0.5) / (1 + y (t - 0.5)) beyond it
    - Effective Duration = (P(y - 0.01) - P(y + 0.01)) / (2 * P * 0.01)
    - Dollar Duration = Modified Duration * P / 100
    - DV01 = (P(y - 0.0001) - P(y + 0.0001)) / 2
    - Convexity = (P(y + 0.0001) + P(y - 0.0001) - 2 * P) / (P * 0.0001^2)

    where P is get_bill_price, y the bond-equivalent yield and t the days to maturity
    divided by 365.

    Referenced against:
    - QuantLib, ZeroCouponBond: https://www.quantlib.org/reference/class_quant_lib_1_1_zero_coupon_bond.html

    Args:
        years_to_maturity (float): The days to maturity divided by 365.
        yield_to_maturity (float): The bond-equivalent yield (decimal).

    Returns:
        dict: The Macaulay's Duration, Modified Duration, Effective Duration, Dollar
        Duration, DV01 and Convexity.
    """
    years, bey = years_to_maturity, yield_to_maturity
    price = get_bill_price(years, bey)

    if years <= 0.5:  # noqa: PLR2004
        modified_duration = years / (1 + bey * years)
    else:
        modified_duration = 0.5 / (1 + bey / 2) + (years - 0.5) / (
            1 + bey * (years - 0.5)
        )

    def shifted(change: float) -> tuple[float, float]:
        return get_bill_price(years, bey - change), get_bill_price(years, bey + change)

    effective_down, effective_up = shifted(0.01)
    basis_point_down, basis_point_up = shifted(0.0001)

    return {
        "Macaulay's Duration": years,
        "Modified Duration": modified_duration,
        "Effective Duration": (effective_down - effective_up) / (2 * price * 0.01),
        "Dollar Duration": modified_duration * price / 100,
        "DV01": (basis_point_down - basis_point_up) / 2,
        "Convexity": (basis_point_up + basis_point_down - 2 * price)
        / (price * 0.0001**2),
    }


def get_bill_analytics(days_to_maturity: int, price: float) -> tuple[float, float]:
    """
    The bond-equivalent yield and modified duration of a Treasury bill from its price.

    The price gives the discount yield, (100 - Price) / 100 * 360 / days, which
    `bond_model.get_bond_equivalent_yield` restates as the Treasury's bond-equivalent yield.
    The modified duration follows from the same pricing, see get_bill_measures.

    Args:
        days_to_maturity (int): The calendar days until the bill matures.
        price (float): The price per 100 of face value.

    Returns:
        tuple[float, float]: The bond-equivalent yield and the modified duration.
    """
    if days_to_maturity <= 0 or not np.isfinite(price):
        return np.nan, np.nan

    discount_yield = (100 - price) / 100 * 360 / days_to_maturity
    yield_to_maturity = float(
        bond_model.get_bond_equivalent_yield(discount_yield, days_to_maturity)
    )
    duration = get_bill_measures(days_to_maturity / 365, yield_to_maturity)[
        "Modified Duration"
    ]
    return yield_to_maturity, duration


def get_security_analytics(
    security: pd.Series, price_date: date
) -> tuple[float, float]:
    """
    Calculate the yield to maturity and modified duration of a Treasury security from its
    price: a bill as a discount security, a note or bond from its coupons, and nothing for
    a floating rate note, whose yield follows its index rate.

    Args:
        security (pd.Series): The Security Type, Coupon, Maturity (a timestamp) and Price
            (per 100 face value) of the security.
        price_date (date): The date of the price.

    Returns:
        tuple[float, float]: The yield to maturity and the modified duration, NaN for a
        floating rate note.
    """
    maturity = security["Maturity"].date()
    if security["Security Type"] == "FRN":
        return np.nan, np.nan
    if security["Security Type"] == "Bill":
        return get_bill_analytics((maturity - price_date).days, security["Price"])
    return get_bond_analytics(
        security["Coupon"],
        get_years_to_maturity(price_date, maturity),
        security["Price"],
    )


def interpolate_curve(curve: pd.Series, years: pd.Series) -> pd.Series:
    """
    The yield of a curve at each maturity, interpolated linearly between its maturities
    and held flat beyond the shortest and longest one.

    Args:
        curve (pd.Series): The yields by maturity label ("3M", "10Y").
        years (pd.Series): The maturities in years.

    Returns:
        pd.Series: The interpolated yields, with the index of years.
    """
    maturities = np.array(
        [curve_model.maturity_in_years(label) for label in curve.index]
    )
    order = np.argsort(maturities)

    return pd.Series(
        np.interp(
            years.to_numpy(dtype=float),
            maturities[order],
            curve.to_numpy(dtype=float)[order],
        ),
        index=years.index,
    )


def get_spreads_at_maturities(
    bonds: pd.DataFrame, maturity_years: list[float]
) -> pd.DataFrame:
    """
    The G-Spread of each issuer at the same maturities, interpolated linearly between the
    issuer's own bonds by their years to maturity, so issuers compare at one point of their
    spread curves rather than on whichever bonds they happen to have.

    The formula is as follows:

    - Spread (issuer, T) = s(a) + (s(b) - s(a)) * (T - a) / (b - a)

    Where a and b are the years to maturity of the issuer's bonds closest below and above
    T and s their G-Spreads; bonds of the same maturity count with their average spread.
    A maturity outside the range of the issuer's bonds has no spread, as a spread curve is
    not extrapolated.

    Args:
        bonds (pd.DataFrame): The bonds, with their Issuer, years to maturity (Years) and
            G-Spread (see FixedIncome.get_bond_universe).
        maturity_years (list[float]): The maturities, in years.

    Returns:
        pd.DataFrame: Per issuer the spread at each maturity (columns such as "5Y"), the
        number of Bonds and the Shortest and Longest years to maturity they cover.
    """
    rows = {}

    for issuer, group in bonds.dropna(subset=["Years", "G-Spread"]).groupby("Issuer"):
        curve = group.groupby("Years")["G-Spread"].mean().sort_index()
        years = curve.index.to_numpy(dtype=float)
        spreads = curve.to_numpy(dtype=float)
        row = {
            f"{maturity:g}Y": (
                float(np.interp(maturity, years, spreads))
                if years[0] <= maturity <= years[-1]
                else np.nan
            )
            for maturity in maturity_years
        }
        row.update({"Bonds": len(group), "Shortest": years[0], "Longest": years[-1]})
        rows[issuer] = row

    spreads_table = pd.DataFrame.from_dict(rows, orient="index")
    spreads_table.index.name = "Issuer"

    return spreads_table
