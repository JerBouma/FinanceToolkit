"""Employees Model: the employee counts the figures per employee divide by."""

from collections.abc import Callable

import pandas as pd

from financetoolkit.ratios.helpers import align_employee_count


def get_employee_count(
    employee_count: pd.DataFrame | None,
    period_end_dates: Callable[[], pd.DataFrame],
    quarterly: bool,
) -> pd.DataFrame:
    """
    Match the employee counts of the companies to the statement periods.

    The counts are matched on the fiscal period end dates, as they are labelled with the year
    of the fiscal year end rather than the statement period. They include the year before the
    start date, which the first quarters use.

    Args:
        employee_count (pd.DataFrame | None): The employee counts as retrieved, None when
            they could not be.
        period_end_dates (Callable[[], pd.DataFrame]): Returns the fiscal period end date of
            each statement period, called only when there are counts to match.
        quarterly (bool): Whether the statements are quarterly.

    Returns:
        pd.DataFrame: The employee count per ticker and statement period, empty without counts.
    """
    if employee_count is None or employee_count.empty:
        return pd.DataFrame()

    return align_employee_count(employee_count, period_end_dates(), quarterly=quarterly)
