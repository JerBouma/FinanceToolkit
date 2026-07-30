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
    daily_historical_data: pd.DataFrame,
):
    """
    This function is a specific function solely related to the Ratios controller. It
    therefore also requires a self instance to exists with specific parameters.

    Args:
        period (str): the period to return the data for.
        within_period (bool): whether to return the data within the period or the
        entire period.

    Raises:
        ValueError: if the period is not daily, monthly, weekly, quarterly, or yearly.

    Returns:
        pd.Series: the returns for the period.
    """
    within_historical_data = {}

    for period, symbol in PERIOD_TRANSLATION.items():
        period_data = daily_historical_data.copy()
        period_data.index = pd.MultiIndex.from_arrays(
            [
                daily_historical_data.index.to_period(symbol),
                daily_historical_data.index.to_period("D"),
            ]
        )

        within_historical_data[period] = period_data

    return within_historical_data
