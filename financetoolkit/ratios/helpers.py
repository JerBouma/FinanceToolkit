"""Ratios Helper Module"""

import pandas as pd


def map_period_data_to_daily_data(
    period_data: pd.DataFrame, daily_dates: pd.DatetimeIndex, quarterly: bool = False
):
    """
    Maps period data (e.g. quarterly or yearly financial statement figures) to daily dates.

    Each period's value applies from the last calendar day of that period onwards, until
    the next period ends. A day therefore only sees the figures of periods that have closed
    on or before it, never those of the period it falls in, so daily ratios do not look
    ahead. When a period ends on a trading day, the daily ratio on that day equals the
    period ratio, which is based on the same period-end close. Days before the end of the
    first period have no value (NaN): the figures are not filled backwards.

    Args:
        period_data (pandas.DataFrame): The period data to map to daily dates, with the
            tickers as index and the periods as columns.
        daily_dates (pandas.DatetimeIndex): The daily dates to map the period data to.
        quarterly (bool, optional): Whether the period data is quarterly or yearly. Defaults to False.

    Returns:
        pandas.DataFrame: Daily values with the daily dates as index and the tickers as columns.
    """
    period_values = period_data.T.astype(float)

    # The last calendar day of each period is the first day its figures apply to.
    period_ends = (
        pd.PeriodIndex(period_values.index, freq="Q" if quarterly else "Y")
        .to_timestamp(how="end")
        .normalize()
    )
    period_values.index = period_ends
    period_values = period_values[~period_values.index.duplicated(keep="last")]

    daily_index = pd.DatetimeIndex(
        daily_dates.to_timestamp()
        if isinstance(daily_dates, pd.PeriodIndex)
        else daily_dates
    ).normalize()

    # Only forward fill: a period's figures carry over the days after its end, and days
    # before the first period end stay empty instead of borrowing later figures.
    daily_period = (
        period_values.reindex(period_values.index.union(daily_index))
        .sort_index()
        .ffill()
        .reindex(daily_index)
    )

    # Restore original daily date index
    daily_period.index = daily_dates

    return daily_period
