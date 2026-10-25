"""Ratios Helper Module"""

import pandas as pd


def map_period_data_to_daily_data(
    period_data: pd.DataFrame,
    daily_dates: pd.DatetimeIndex,
    quarterly: bool = False,
    period_end_dates: pd.DataFrame | None = None,
):
    """
    Maps period data (e.g. quarterly or yearly financial statement figures) to daily dates.

    Each period's value applies from the last day of its fiscal period onwards, until the
    next period ends. A day therefore only sees the figures of periods that have closed on
    or before it, never those of the period it falls in. The figures apply from the period
    end, not from the (later) date they were filed, so a day shortly after a period end
    already uses figures the market only learns of weeks later. When a period ends on a
    trading day, the daily ratio on that day equals the period ratio, which is based on the
    same period-end close. Days before the end of the first period have no value (NaN): the
    figures are not filled backwards.

    The statement columns are calendar periods, which for a fiscal year ending in January
    through May is the calendar year before the one it ends in (and for an off-cycle quarter
    the calendar quarter before), so the end of the label can lie before the end of the
    fiscal period. period_end_dates gives the actual end of each ticker's fiscal periods;
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
