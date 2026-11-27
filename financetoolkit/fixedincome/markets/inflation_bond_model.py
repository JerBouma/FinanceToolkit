"""Inflation-Protected Bond Model

US Treasury Inflation-Protected Securities (TIPS): the reference CPI of a date, the index
ratio and the inflation-adjusted price, following the Treasury's rules (31 CFR 356,
Appendix B, I.B and III).
"""

__docformat__ = "google"

import calendar
import math
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# The reference CPI of the first day of a month is the CPI of the third month before it.
REFERENCE_LAG_MONTHS = 3


def _truncate_and_round(value: float) -> float:
    """Truncated to six decimal places and rounded to five, as the Treasury does."""
    return round(math.floor(value * 1e6 + 1e-6) / 1e6, 5)


def fill_unreported_months(consumer_price_index: pd.Series) -> pd.Series:
    """
    Fill the months the Bureau of Labor Statistics did not report the CPI for with the index
    number the Treasury announces in their place, based on the last twelve-month change of
    the CPI (31 CFR 356, Appendix B, I.B.4(iv)).

    The formula is as follows:

    - CPI_M = CPI_(M-N) * (CPI_(M-N) / CPI_(M-N-12))^(N / 12)

    where CPI_(M-N) is the last CPI reported, N months before the month M without one,
    rounded to three decimals as the CPI is published. The
    Treasury keeps the announced number for that month even when the CPI is reported later,
    so a month without a CPI between reported months (October 2025) takes it as well.

    Args:
        consumer_price_index (pd.Series): The CPI-U, not seasonally adjusted, indexed by
            monthly period.

    Returns:
        pd.Series: The CPI of every month from the first to the last one given, with the
        unreported months filled.
    """
    reported = consumer_price_index.dropna().sort_index()
    if reported.empty:
        return reported

    months = pd.period_range(reported.index[0], reported.index[-1], freq="M")
    filled = reported.reindex(months)
    for month in months[filled.isna().to_numpy()]:
        last = filled.loc[: month - 1].dropna()
        if last.empty:
            continue
        latest_month = last.index[-1]
        year_before = filled.get(latest_month - 12, np.nan)
        lag = (month - latest_month).n
        # Rounded to three decimals, as the BLS publishes the CPI and the Treasury the
        # number it announces (325.604 for October 2025).
        filled[month] = round(
            last.iloc[-1] * (last.iloc[-1] / year_before) ** (lag / 12), 3
        )

    return filled


def get_reference_cpi(consumer_price_index: pd.Series, day: date) -> float:
    """
    Calculate the reference CPI of a day for US Treasury Inflation-Protected Securities: the
    CPI of the third month before, interpolated linearly over the days of the month towards
    the CPI of the second month before (31 CFR 356, Appendix B, I.B.3).

    The formula is as follows:

    - Ref CPI = Ref CPI_M + (t - 1) / D * (Ref CPI_(M+1) - Ref CPI_M)

    where Ref CPI_M is the CPI of the third month before the day's month M, Ref CPI_(M+1) that
    of the second month before, t the day of the month and D the days in the month. The
    result is truncated to six decimal places and rounded to five. The Treasury's example:
    the reference CPI of 15 April 1996, from the CPI of January (154.40) and February 1996
    (154.90), is 154.63333.

    Referenced against:
    - QuantLib, CPI.laggedFixing: https://github.com/lballabio/QuantLib/blob/v1.44/ql/indexes/inflationindex.hpp#L39

    Args:
        consumer_price_index (pd.Series): The CPI-U, not seasonally adjusted, indexed by
            monthly period (see fill_unreported_months).
        day (date): The day.

    Returns:
        float: The reference CPI, NaN when the CPI of either month is missing.
    """
    first_month = day.replace(day=1) - relativedelta(months=REFERENCE_LAG_MONTHS)
    first = consumer_price_index.get(pd.Period(first_month, freq="M"), np.nan)
    second = consumer_price_index.get(
        pd.Period(first_month + relativedelta(months=1), freq="M"), np.nan
    )
    if not (np.isfinite(first) and np.isfinite(second)):
        return np.nan

    days_in_month = calendar.monthrange(day.year, day.month)[1]
    return _truncate_and_round(first + (day.day - 1) / days_in_month * (second - first))


def get_index_ratio(reference_cpi: float, reference_cpi_dated_date: float) -> float:
    """
    Calculate the index ratio of a US Treasury Inflation-Protected Security: the reference
    CPI of the day over the reference CPI of the security's dated date (its original issue
    date, unless the dated date differs from it), truncated to six decimal places and
    rounded to five (31 CFR 356.2 and Appendix B, I.B.2). The inflation-adjusted principal
    is the par amount times the index ratio.

    The formula is as follows:

    - Index Ratio = Ref CPI (Date) / Ref CPI (Dated Date)

    The Treasury's example: 154.65000 / 154.63333 = 1.000107803, which is an index ratio of
    1.00011.

    Shares similar logic:
    - QuantLib, CPIBond: https://www.quantlib.org/reference/class_quant_lib_1_1_c_p_i_bond.html
    - FinancePy, InflationBond.inflation_principal: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/inflation/InflationBond.py#L92

    Args:
        reference_cpi (float): The reference CPI of the day (see get_reference_cpi).
        reference_cpi_dated_date (float): The reference CPI of the dated date.

    Returns:
        float: The index ratio.
    """
    if not (np.isfinite(reference_cpi) and reference_cpi_dated_date > 0):
        return np.nan

    return _truncate_and_round(reference_cpi / reference_cpi_dated_date)


def get_inflation_adjusted_price(
    real_price: float, real_accrued_interest: float, index_ratio: float
) -> tuple[float, float, float]:
    """
    Calculate the inflation-adjusted price and accrued interest of a US Treasury
    Inflation-Protected Security: TIPS are quoted per 100 of inflation-adjusted principal,
    so what a buyer pays per 100 of par is the quoted (real) price and accrued interest
    times the index ratio (31 CFR 356, Appendix B, III).

    The formula is as follows:

    - Adjusted Price = P * Index Ratio
    - Adjusted Accrued Interest = A * Index Ratio
    - Settlement Price = (P + A) * Index Ratio

    where P is the real clean price and A the real accrued interest per 100 of par.

    Shares similar logic:
    - FinancePy, InflationBond.flat_price_from_yield_to_maturity: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/inflation/InflationBond.py#L112

    Args:
        real_price (float): The real (quoted) clean price per 100 of par.
        real_accrued_interest (float): The real accrued interest per 100 of par.
        index_ratio (float): The index ratio of the settlement date.

    Returns:
        tuple[float, float, float]: The adjusted price, the adjusted accrued interest and
        the settlement price (adjusted price plus adjusted accrued interest) per 100 of par.
    """
    adjusted_price = real_price * index_ratio
    adjusted_accrued = real_accrued_interest * index_ratio

    return adjusted_price, adjusted_accrued, adjusted_price + adjusted_accrued
