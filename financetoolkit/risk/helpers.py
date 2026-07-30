"""Risk Helpers Module"""

__docformat__ = "google"

import pandas as pd

# pylint: disable=protected-access

PERIOD_TRANSLATION: dict[str, str | dict[str, str]] = {
    "intraday": {
        "1min": "h",
        "5min": "h",
        "15min": "D",
        "30min": "D",
        "1hour": "D",
    },
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def determine_within_historical_data(
    daily_historical_data: pd.DataFrame,
    intraday_historical_data: pd.DataFrame,
    intraday_period: str | None,
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
        if not intraday_period and period == "intraday":
            continue

        period_symbol = (
            symbol[intraday_period] if period == "intraday" else symbol  # type: ignore
        )

        if not intraday_historical_data.empty and period in [
            "intraday",
            "daily",
        ]:
            source_data = intraday_historical_data
        else:
            source_data = daily_historical_data

        inner_freq = "D" if period != "intraday" else "min"
        period_data = source_data.copy()
        period_data.index = pd.MultiIndex.from_arrays(
            [
                source_data.index.to_period(period_symbol),
                source_data.index.to_period(inner_freq),
            ]
        )

        within_historical_data[period] = period_data

    return within_historical_data
