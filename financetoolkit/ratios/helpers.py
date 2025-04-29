"""Ratios Helper Module"""

import pandas as pd


def map_period_data_to_daily_data(
    period_data: pd.DataFrame, daily_dates: pd.PeriodIndex, quarterly: bool = False
):
    """
    Maps period data to daily dates and fills missing values.

    Args:
        period_data (pandas.DataFrame): The period data to map to daily dates.
        daily_dates (pandas.DatetimeIndex): The daily dates to map the period data to.
        quarterly (bool, optional): Whether the period data is quarterly or yearly. Defaults to False.

    Returns:
        pandas.DataFrame: Daily values aligned with daily dates.
    """
    daily_period = pd.DataFrame(columns=period_data.index, index=daily_dates)

    # Convert the dates to Quarterly or Yearly periods
    daily_period.index = pd.PeriodIndex(
        daily_period.index, freq="Q" if quarterly else "Y"
    )

    # Fill the results into the daily_period DataFrame
    for period in period_data.columns:
        if period in daily_period.index:
            daily_period.loc[period, :] = period_data[period].to_numpy()

    # Fill missing values using backward and forward fill
    daily_period = daily_period.astype(float).bfill().ffill()

    # Restore original daily date index
    daily_period.index = daily_dates

    return daily_period
