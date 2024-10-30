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
        within_historical_data[period] = daily_historical_data.groupby(
            pd.Grouper(freq=f"{symbol}E" if symbol in ["M", "Q", "Y"] else symbol)
        ).apply(lambda x: x)

        within_historical_data[period].index = within_historical_data[
            period
        ].index.set_levels(
            [
                pd.PeriodIndex(
                    within_historical_data[period].index.levels[0],
                    freq=symbol,
                ),
                pd.PeriodIndex(
                    within_historical_data[period].index.levels[1], freq="D"
                ),
            ],
        )

    return within_historical_data
