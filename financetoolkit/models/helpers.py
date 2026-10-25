"""Models Helpers Module"""

__docformat__ = "google"


import pandas as pd

# pylint: disable=protected-access

PERIOD_TRANSLATION: dict[str, str] = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def determine_within_historical_data(
    period: str, daily_historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Nests the daily historical data under the week, month, quarter or year it falls in, the
    "within period" data the models estimate a beta per period from.

    Args:
        period (str): The period (weekly, monthly, quarterly or yearly).
        daily_historical_data (pd.DataFrame): The daily historical data, with a PeriodIndex.

    Raises:
        KeyError: If the period is not weekly, monthly, quarterly or yearly.

    Returns:
        pd.DataFrame: The daily data with the period and the day as the index levels.
    """
    symbol = PERIOD_TRANSLATION[period]
    dates = pd.DatetimeIndex(daily_historical_data.to_timestamp().index)

    period_data = daily_historical_data.copy()
    period_data.index = pd.MultiIndex.from_arrays(
        [dates.to_period(symbol), dates.to_period("D")]
    )

    return period_data
