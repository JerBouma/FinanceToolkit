"""Performance Helpers Module"""

__docformat__ = "google"


import pandas as pd

from financetoolkit.utilities import logger_model
from financetoolkit.utilities.statistics_model import to_datetime_index

logger = logger_model.get_logger()

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

        # The intraday entry maps each intraday_period to a symbol; every other
        # entry is the symbol itself, and the intraday loop iteration is skipped
        # above whenever no intraday_period was given.
        period_symbol = (
            symbol[intraday_period]
            if isinstance(symbol, dict) and intraday_period is not None
            else symbol
        )
        if not isinstance(period_symbol, str):
            raise TypeError(f"No period symbol resolved for {period}.")

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
                to_datetime_index(source_data.index).to_period(period_symbol),
                to_datetime_index(source_data.index).to_period(inner_freq),
            ]
        )

        within_historical_data[period] = period_data

    return within_historical_data


def determine_within_dataset(
    dataset: pd.DataFrame, period: str, correlation: bool = False
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
    dataset_new = dataset.copy()
    dataset_new.index = pd.DatetimeIndex(dataset_new.to_timestamp().index)
    period_symbol = PERIOD_TRANSLATION[period]

    if correlation:
        within_historical_data = dataset_new.groupby(
            pd.Grouper(
                freq=(
                    f"{period_symbol}E"
                    if period_symbol in ["M", "Q", "Y"]
                    else period_symbol
                )
            )
        ).apply(lambda x: x.corr())

        # The grouped correlation matrices are indexed by (period, ticker) and the
        # outer level is relabelled from timestamps to periods.
        grouped_index = within_historical_data.index
        if not isinstance(grouped_index, pd.MultiIndex):
            raise TypeError("Expected the grouped correlations to be MultiIndexed.")

        within_historical_data.index = grouped_index.set_levels(
            [
                pd.PeriodIndex(grouped_index.levels[0], freq=period_symbol),
                grouped_index.levels[1],
            ],
        )
    else:
        within_historical_data = dataset_new.copy()
        within_historical_data.index = pd.MultiIndex.from_arrays(
            [
                to_datetime_index(dataset_new.index).to_period(period_symbol),
                to_datetime_index(dataset_new.index).to_period("D"),
            ]
        )

    return within_historical_data
