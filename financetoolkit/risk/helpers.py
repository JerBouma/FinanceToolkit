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
            within_historical_data[period] = (
                intraday_historical_data.groupby(pd.Grouper(freq=period_symbol))
                .apply(lambda x: x)
                .dropna(how="all", axis=0)
            )
        else:
            within_historical_data[period] = daily_historical_data.groupby(
                pd.Grouper(
                    freq=(
                        f"{period_symbol}E"
                        if period_symbol in ["M", "Q", "Y"]
                        else period_symbol
                    )
                )
            ).apply(lambda x: x)

        within_historical_data[period].index = within_historical_data[
            period
        ].index.set_levels(
            [
                pd.PeriodIndex(
                    within_historical_data[period].index.levels[0],
                    freq=period_symbol,
                ),
                pd.PeriodIndex(
                    within_historical_data[period].index.levels[1],
                    freq="D" if period != "intraday" else "min",
                ),
            ],
        )

    return within_historical_data
