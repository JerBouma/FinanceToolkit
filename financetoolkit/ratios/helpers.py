"""Ratios Helper Module"""

import pandas as pd


def map_period_data_to_daily_data(
    period_data: pd.DataFrame,
    daily_dates: pd.DatetimeIndex,
    quarterly: bool = False,
    period_end_dates: pd.DataFrame | None = None,
):
    """
    Map period data, such as quarterly or yearly financial statement figures, to daily dates, so each day carries the
    figures of the last period that closed on or before it.

    Each period's value applies from the last day of its fiscal period onwards, until the
    next period ends. A day therefore only sees the figures of periods that have closed on
    or before it, never those of the period it falls in. By default the figures apply from
    the period end, not from the (later) date they were filed, so a day shortly after a
    period end already uses figures the market only learns of weeks later; a point-in-time
    Toolkit passes the filing days as the period end dates, so each day only uses filed
    figures. The period ratios price each period at the close on the same date (or the last
    trading day before it), so on a trading day that a period's figures start to apply, the
    daily ratio equals the period ratio. Days before the end of the first period have no
    value (NaN): the figures are not filled backwards.

    The statement columns are calendar periods, which for a fiscal year ending in January
    through May is the calendar year before the one it ends in (and for an off-cycle quarter
    the calendar quarter before), so the end of the label can lie before the end of the
    fiscal period. The period end dates give the actual end of each ticker's fiscal periods;
    without it, or for a ticker or period it does not cover, the end of the labelled
    calendar period is used.

    Args:
        period_data (pandas.DataFrame): The period data to map to daily dates, with the
            tickers as index and the periods as columns.
        daily_dates (pandas.DatetimeIndex): The daily dates to map the period data to.
        quarterly (bool, optional): Whether the period data is quarterly or yearly. Defaults to False.
        period_end_dates (pandas.DataFrame | None, optional): The last day of the fiscal period
            behind each column, with the tickers as index and the periods as columns. Defaults
            to None, which uses the end of each labelled calendar period.

    Returns:
        pandas.DataFrame: Daily values with the daily dates as index and the tickers as columns.
    """
    period_values = period_data.T.astype(float)
    periods = pd.PeriodIndex(period_values.index, freq="Q" if quarterly else "Y")

    # The last calendar day of each labelled period, used where no fiscal end date is known.
    label_ends = pd.Series(periods.to_timestamp(how="end").normalize(), index=periods)

    daily_index = pd.DatetimeIndex(
        daily_dates.to_timestamp()
        if isinstance(daily_dates, pd.PeriodIndex)
        else daily_dates
    ).normalize()

    daily_columns = {}

    for column in period_values.columns:
        if period_end_dates is not None and column in period_end_dates.index:
            ends = (
                pd.to_datetime(period_end_dates.loc[column].reindex(periods))
                .fillna(label_ends)
                .dt.normalize()
            )
        else:
            ends = label_ends

        # The first day each period's figures apply to is the end of its fiscal period.
        values = pd.Series(
            period_values[column].to_numpy(), index=pd.DatetimeIndex(ends)
        )
        values = values[~values.index.duplicated(keep="last")]

        # Only forward fill: a period's figures carry over the days after its end, and days
        # before the first period end stay empty instead of borrowing later figures.
        daily_columns[column] = (
            values.reindex(values.index.union(daily_index))
            .sort_index()
            .ffill()
            .reindex(daily_index)
            .to_numpy()
        )

    daily_period = pd.DataFrame(
        daily_columns, index=daily_dates, columns=period_values.columns
    )

    return daily_period


def align_employee_count(
    employee_count: pd.DataFrame,
    period_end_dates: pd.DataFrame,
    quarterly: bool = False,
) -> pd.DataFrame:
    """
    Place the employee counts of the annual filings on the statement periods, so a figure per employee divides each
    period's figures by that period's headcount.

    The employee counts are labelled with the calendar year of the fiscal year end of the
    annual filing they come from (the period of report; a 52/53-week year ending in the first
    week of January counts as the December before), which for a fiscal year ending in
    January through May is the year after the statement label (NVIDIA's fiscal year ending
    January 2024 is the statement column 2023). They are therefore matched on the period
    end dates of the statements, not on the labels:

    - Yearly: a period gets the count of the filing whose fiscal year ends in the same
      calendar year as the period (with the same first week of January rule), the
      headcount the annual report of that period states.
    - Quarterly: a quarter gets the count of the latest fiscal year that ended in a calendar
      year before the one the quarter ends in, or in the same one for a quarter ending in
      December (a period ending in the first days of January counts as December). A count
      of a fiscal year ending within the calendar year of the quarter is not yet used, as
      the quarter can end before that fiscal year does; with a fiscal year ending in
      September, the quarters ending in March, June and September use the count of the
      previous fiscal year.

    Args:
        employee_count (pd.DataFrame): The employee counts with the calendar years of the
            fiscal year ends as index (a PeriodIndex or years) and the tickers as columns,
            as `Toolkit.get_employee_count` returns them.
        period_end_dates (pd.DataFrame): The fiscal period end date of every statement
            column, with the tickers as index and the periods as columns.
        quarterly (bool, optional): Whether the statement periods are quarters. Defaults
            to False.

    Returns:
        pd.DataFrame: The employee counts with the tickers as index and the statement
        periods as columns, NaN for a period without a count.
    """
    years = pd.Index(
        [
            year.year if isinstance(year, pd.Period) else int(year)
            for year in employee_count.index
        ]
    )
    counts = employee_count.set_axis(years).groupby(level=0).last()

    end_dates = period_end_dates.apply(pd.to_datetime)

    if quarterly:
        # The calendar year the quarter ends in, minus one unless it ends in December, so a
        # 52/53-week year ending on 24 December or in the first days of January takes the
        # count of the December it belongs to.
        count_years = end_dates.apply(
            lambda dates: dates.dt.year - (dates.dt.month != 12)  # noqa: PLR2004
        )
    else:
        # A fiscal year ending in the first week of January belongs to the year before, as
        # the counts are labelled (get_employee_count).
        count_years = end_dates.apply(
            lambda dates: (dates - pd.Timedelta(days=7)).dt.year
        )

    aligned = pd.DataFrame(
        float("nan"), index=period_end_dates.index, columns=period_end_dates.columns
    )

    for ticker in aligned.index.intersection(counts.columns):
        aligned.loc[ticker] = (
            counts[ticker].reindex(count_years.loc[ticker].to_numpy()).to_numpy()
        )

    return aligned
