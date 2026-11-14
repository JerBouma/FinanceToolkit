"""Bond Data Model"""

__docformat__ = "google"

import re
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from financetoolkit.fixedincome import bond_model, curve_model

# US Treasuries and corporate bonds pay their coupon twice a year.
FREQUENCY = 2

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


def cusip_to_isin(cusip: str) -> str:
    """
    The ISIN of a US security from its CUSIP: "US", the CUSIP and a check digit (ISO 6166).

    Args:
        cusip (str): The nine-character CUSIP.

    Returns:
        str: The ISIN, or an empty string when the CUSIP is not nine characters.
    """
    cusip = str(cusip or "").strip().upper()
    if len(cusip) != 9:  # noqa: PLR2004
        return ""

    # Letters count as 10 to 35; the digits are then doubled from the right, every other
    # one, and the digits of the result summed (the Luhn algorithm).
    digits = "".join(str(int(character, 36)) for character in f"US{cusip}")
    total = 0
    for position, digit in enumerate(reversed(digits)):
        value = int(digit) * (2 if position % 2 == 0 else 1)
        total += value // 10 + value % 10

    return f"US{cusip}{(10 - total % 10) % 10}"


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


def get_years_to_maturity(
    settlement: date, maturity: date, frequency: int = FREQUENCY
) -> float:
    """
    The time to maturity in coupon periods, counted on the bond's actual coupon dates.

    The coupon dates run back from maturity in steps of 12 / frequency months (a coupon of
    a bond maturing on 31 August falls on the last day of February). The whole periods
    after the next coupon date count as whole periods, and the current period counts the
    share of its actual days that is still to come (actual/actual, the Treasury
    convention). Dividing by the frequency gives the years `bond_model` takes, so its
    coupon schedule falls on the bond's real coupon dates and its accrued interest is the
    elapsed share of the current period.

    Args:
        settlement (date): The date the price applies to.
        maturity (date): The maturity date.
        frequency (int): The number of coupons per year. Defaults to 2.

    Returns:
        float: The years to maturity, or NaN when the bond has matured.
    """
    if maturity <= settlement:
        return np.nan

    months = 12 // frequency
    periods = 1
    while maturity - relativedelta(months=months * periods) > settlement:
        periods += 1

    previous_coupon = maturity - relativedelta(months=months * periods)
    next_coupon = maturity - relativedelta(months=months * (periods - 1))
    remaining = (next_coupon - settlement).days / (next_coupon - previous_coupon).days

    return (periods - 1 + remaining) / frequency


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
        frequency (int): The number of coupons per year. Defaults to 2.

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


def get_bill_analytics(days_to_maturity: int, price: float) -> tuple[float, float]:
    """
    The bond-equivalent yield and modified duration of a Treasury bill from its price.

    The price gives the discount yield, (100 - Price) / 100 * 360 / days, which
    `bond_model.get_bond_equivalent_yield` restates as the Treasury's bond-equivalent yield.

    Args:
        days_to_maturity (int): The calendar days until the bill matures.
        price (float): The price per 100 of face value.

    Returns:
        tuple[float, float]: The bond-equivalent yield and the modified duration.
    """
    if days_to_maturity <= 0 or not np.isfinite(price):
        return np.nan, np.nan

    discount_yield = (100 - price) / 100 * 360 / days_to_maturity
    yield_to_maturity = bond_model.get_bond_equivalent_yield(
        discount_yield, days_to_maturity
    )
    years = days_to_maturity / 365
    duration = bond_model.get_modified_duration(
        100, 0.0, years, yield_to_maturity, FREQUENCY
    )
    return float(yield_to_maturity), duration


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
