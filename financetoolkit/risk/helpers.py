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
    This function is a specific function solely related to the Risk controller. It
    reshapes the historical data into a multi-index (period, observation) format for
    each period defined in PERIOD_TRANSLATION, which is what the "within period"
    calculations in the Risk controller operate on.

    Args:
        daily_historical_data (pd.DataFrame): the daily historical data used for the
        weekly, monthly, quarterly and yearly periods.
        intraday_historical_data (pd.DataFrame): the intraday historical data used for
        the intraday period. When empty, the daily historical data is used instead.
        intraday_period (str | None): the intraday frequency (e.g. "1min", "1hour")
        used to look up the outer resampling symbol. When None, the intraday period is
        skipped entirely.

    Returns:
        dict[str, pd.DataFrame]: a dictionary with the period name as key and the
        historical data with a (period, observation) multi-index as value.
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
