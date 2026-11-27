"""Floating Rate Note Model

US Treasury floating rate notes: the index rate from the 13-week bill auctions, the daily
interest accrual with its two-business-day lockout, and the price, discount margin and
durations of the Treasury's own pricing formula (31 CFR 356, Appendix B, sections I.C and
IV).
"""

__docformat__ = "google"

from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# The days of a year in the Treasury's money market convention for floating rate notes.
DAYS_IN_YEAR = 360

# The discount margins the solver searches between (decimals): -50% to 50%.
MARGIN_BOUNDS = (-0.5, 0.5)

# The change in the index rate or discount margin of the durations: one basis point.
BASIS_POINT = 0.0001


def get_index_rate(high_rate: float, days_to_maturity: float) -> float:
    """
    Calculate the index rate of US Treasury floating rate notes from the high rate of a
    13-week bill auction: the rate the notes pay interest at, before their spread.

    The index rate is the auction's high (discount) rate restated as a simple-interest money
    market yield on an actual/360 basis, rounded to nine decimal places of a percent (31 CFR
    356.2 and Appendix B, I.C.1): the return of buying the bill at its discount price and
    holding it to maturity.

    The formula is as follows:

    - Index Rate = d / (1 - d * t / 360)

    where d is the high rate (decimal) and t the days from the bill's issue to its maturity
    (91 for most 13-week bills). The Treasury's example: a high rate of 0.095% on a 91-day
    bill is an index rate of 0.095022819%.

    Args:
        high_rate (float): The high rate of the 13-week bill auction (decimal).
        days_to_maturity (float): The days from the bill's issue date to its maturity.

    Returns:
        float: The index rate (decimal, simple interest on actual/360).
    """
    return round(high_rate / (1 - high_rate * days_to_maturity / DAYS_IN_YEAR), 11)


def _business_days_before(day: date, count: int) -> date:
    """The day that lies count business days (weekdays) before day."""
    while count:
        day -= timedelta(days=1)
        if day.weekday() < 5:  # noqa: PLR2004
            count -= 1
    return day


def get_daily_index_rates(
    index_rates: pd.Series,
    start: date,
    end: date,
    lockout_dates: list[date] | None = None,
) -> pd.Series:
    """
    Calculate the index rate that applies to each day of a floating rate note's accrual: the
    rate of the latest 13-week bill auction held before each day.

    Each day accrues at the index rate of the most recent 13-week bill auction held before
    that day, so a Monday auction applies from Tuesday (31 CFR 356, Appendix B, I.C.3).
    An auction held in the two business days before an issue date or an interest payment
    date is left out of the accrual up to and including that date, and applies from the day
    after it (the lockout). The lockout "applies only to the issue date and interest payment
    dates" (31 CFR 356, Appendix B, IV), so a secondary-market settlement between them takes
    the latest auction.

    Args:
        index_rates (pd.Series): The index rate of each 13-week bill auction (see
            get_index_rate), indexed by auction date.
        start (date): The first day of the accrual (the dated date or the last interest
            payment date), included.
        end (date): The day the accrual runs up to, excluded (the settlement date).
        lockout_dates (list[date] | None, optional): The issue and interest payment dates
            whose two business days before them are locked out. Defaults to None, which
            locks out the days before start and end, as for an accrual from an interest
            payment date to the issue date of a reopening.

    Returns:
        pd.Series: The index rate of each day from start up to end, indexed by date.

    Raises:
        ValueError: If no auction was held before start.
    """
    auctions = index_rates.dropna().sort_index()
    auction_dates = np.array(list(auctions.index), dtype="datetime64[D]")
    lockouts = [
        (_business_days_before(locked, 2), locked)
        for locked in (lockout_dates if lockout_dates is not None else [start, end])
    ]

    days, rates = [], []
    day = start
    while day < end:
        # The latest auction before the day, not one inside a lockout that covers the day.
        cutoff = day
        for lockout_start, locked in lockouts:
            if lockout_start <= day <= locked:
                cutoff = min(cutoff, lockout_start)
        position = int(np.searchsorted(auction_dates, np.datetime64(cutoff))) - 1
        if position < 0:
            raise ValueError(f"No 13-week bill auction was held before {day}.")
        days.append(day)
        rates.append(float(auctions.iloc[position]))
        day += timedelta(days=1)

    return pd.Series(rates, index=pd.Index(days, name="Date"), dtype=float)


def get_floating_rate_note_accrued_interest(
    daily_index_rates: pd.Series, spread: float
) -> float:
    """
    Calculate the interest a US Treasury floating rate note has accrued per 100 of par: the
    sum of each day's interest at the index rate plus the spread, over 360, with a minimum
    daily accrual of zero (31 CFR 356, Appendix B, IV.C).

    The formula is as follows:

    - Accrued Interest = Σ_j 100 * max(r_j + s, 0) / 360

    where r_j is the index rate of day j (see get_daily_index_rates) from the dated date or
    the last interest payment date up to the settlement date and s the spread.

    Args:
        daily_index_rates (pd.Series): The index rate of each day of the accrual (decimals).
        spread (float): The spread of the note (decimal).

    Returns:
        float: The accrued interest per 100 of par.
    """
    return float(
        (100 * np.maximum(daily_index_rates.to_numpy(dtype=float) + spread, 0)).sum()
        / DAYS_IN_YEAR
    )


def get_frn_price(
    index_rate: float,
    spread: float,
    discount_margin: float,
    period_days: list[int] | np.ndarray,
    accrued_interest: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate the price of a US Treasury floating rate note from its discount margin with the
    Treasury's pricing formula (31 CFR 356, Appendix B, IV).

    The cash flows are projected at the index rate that applies to the settlement date, held
    for every later day, plus the spread, and discounted at that index rate plus the discount
    margin, compounded at each quarterly interest payment on actual/360. The interest accrued
    before the settlement date is paid with the first payment, so it is discounted with it.

    The formula is as follows:

    - a = 100 * max(r + s, 0) / 360, the projected daily interest
    - A_1 = a * (T_1 - T_0) + AI and A_i = a * (T_i - T_(i-1)) for later payments, plus 100 at
      maturity
    - B_i = 1 + (r + m) * (T_i - T_(i-1)) / 360
    - Dirty Price = Σ_i A_i / Π_(k<=i) B_k
    - Clean Price = Dirty Price - AI

    where r is the index rate, s the spread, m the discount margin, T_0 the settlement date,
    T_i the interest payment dates and AI the accrued interest (see get_floating_rate_note_accrued_interest).
    For a new note the discount margin of the auction is its spread. The Treasury's auction
    of 91282CRD5 reopened on 25 September 2026 at a discount margin of 0.040% prices at
    100.015499, as the Treasury announced.

    Referenced against:
    - FinancePy, BondFRN.dirty_price_from_dm: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_frn.py#L107

    Args:
        index_rate (float): The index rate that applies to the settlement date (decimal).
        spread (float): The spread of the note (decimal).
        discount_margin (float): The discount margin (decimal).
        period_days (list[int] | np.ndarray): The days of each remaining interest period, the
            first from the settlement date to the next interest payment date.
        accrued_interest (float, optional): The interest accrued per 100 of par up to the
            settlement date. Defaults to 0.0, a settlement on an interest payment date.

    Returns:
        tuple[float, float]: The dirty and the clean price per 100 of par.
    """
    days = np.asarray(period_days, dtype=float)
    cash_flows = 100 * max(index_rate + spread, 0) / DAYS_IN_YEAR * days
    cash_flows[0] += accrued_interest
    cash_flows[-1] += 100
    discount = np.cumprod(1 + (index_rate + discount_margin) * days / DAYS_IN_YEAR)
    dirty_price = float((cash_flows / discount).sum())

    return dirty_price, dirty_price - accrued_interest


def get_discount_margin(
    clean_price: float,
    index_rate: float,
    spread: float,
    period_days: list[int] | np.ndarray,
    accrued_interest: float = 0.0,
) -> float:
    """
    Calculate the discount margin of a US Treasury floating rate note: the margin over the
    index rate at which the Treasury's pricing formula (see get_frn_price) gives the clean
    price. It is the floating rate note's counterpart of a yield spread: a note priced
    above par has a discount margin below its spread.

    Shares similar logic:
    - FinancePy, BondFRN.discount_margin: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_frn.py#L375

    Args:
        clean_price (float): The clean price per 100 of par.
        index_rate (float): The index rate that applies to the settlement date (decimal).
        spread (float): The spread of the note (decimal).
        period_days (list[int] | np.ndarray): The days of each remaining interest period, the
            first from the settlement date to the next interest payment date.
        accrued_interest (float, optional): The interest accrued per 100 of par up to the
            settlement date. Defaults to 0.0.

    Returns:
        float: The discount margin (decimal), NaN when no margin between -50% and 50%
        gives the price.
    """
    if not np.isfinite(clean_price) or not np.isfinite(index_rate):
        return np.nan

    def price_difference(margin: float) -> float:
        return (
            get_frn_price(index_rate, spread, margin, period_days, accrued_interest)[1]
            - clean_price
        )

    low, high = MARGIN_BOUNDS
    if price_difference(low) * price_difference(high) > 0:
        return np.nan

    return float(brentq(price_difference, low, high, xtol=1e-12))


def get_frn_durations(
    index_rate: float,
    spread: float,
    discount_margin: float,
    period_days: list[int] | np.ndarray,
    accrued_interest: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate the index (rate) duration and the spread duration of a US Treasury floating
    rate note: how much its price changes with the index rate and with the discount margin,
    on the Treasury's pricing formula (see get_frn_price), by a change of one basis point
    either way (Fabozzi, 2012, The Handbook of Fixed Income Securities, ch. 17).

    The coupons follow the index, so a change in the index rate moves both the projected
    cash flows and their discounting and barely moves the price: the index duration is
    about the time to the next interest payment. A change in the discount margin moves only
    the discounting, so the spread duration is about the time to maturity.

    The formula is as follows:

    - Index Duration = (P(r - 0.0001) - P(r + 0.0001)) / (2 * P * 0.0001)
    - Spread Duration = (P(m - 0.0001) - P(m + 0.0001)) / (2 * P * 0.0001)

    where P is the dirty price at the index rate r and discount margin m.

    Referenced against:
    - FinancePy, BondFRN.modified_credit_duration: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_frn.py#L283

    Shares similar logic:
    - FinancePy, BondFRN.dollar_duration: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/bonds/bond_frn.py#L178

    Args:
        index_rate (float): The index rate that applies to the settlement date (decimal).
        spread (float): The spread of the note (decimal).
        discount_margin (float): The discount margin (decimal).
        period_days (list[int] | np.ndarray): The days of each remaining interest period.
        accrued_interest (float, optional): The interest accrued per 100 of par up to the
            settlement date. Defaults to 0.0.

    Returns:
        tuple[float, float]: The index duration and the spread duration, in years.
    """
    if not np.isfinite(discount_margin):
        return np.nan, np.nan

    def dirty(rate: float, margin: float) -> float:
        return get_frn_price(rate, spread, margin, period_days, accrued_interest)[0]

    price = dirty(index_rate, discount_margin)
    index_duration = (
        dirty(index_rate - BASIS_POINT, discount_margin)
        - dirty(index_rate + BASIS_POINT, discount_margin)
    ) / (2 * price * BASIS_POINT)
    spread_duration = (
        dirty(index_rate, discount_margin - BASIS_POINT)
        - dirty(index_rate, discount_margin + BASIS_POINT)
    ) / (2 * price * BASIS_POINT)

    return float(index_duration), float(spread_duration)
