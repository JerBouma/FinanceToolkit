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
    # The daily period nests the intraday observations of a single day, so it is only available when intraday data was fetched -- exactly the condition the Risk controller guards its period="daily" branches with, and the Econometrics controller its period="daily" branches that ask for within_period=True.  # noqa: E501
    "daily": "D",
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
        the intraday and daily periods. When empty, the daily period is skipped
        entirely, since there would be nothing to nest inside a single day.
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
        # Without intraday observations there is nothing to nest inside a single day, so the daily period would collapse to one observation per group.  # noqa: E501
        if intraday_historical_data.empty and period == "daily":
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

        # The daily period nests intraday observations inside each day, so like the intraday period it needs a minute-level inner index rather than a daily one.  # noqa: E501
        inner_freq = "min" if period in ("intraday", "daily") else "D"
        period_data = source_data.copy()
        period_data.index = pd.MultiIndex.from_arrays(
            [
                source_data.index.to_period(period_symbol),
                source_data.index.to_period(inner_freq),
            ]
        )

        within_historical_data[period] = period_data

    return within_historical_data
