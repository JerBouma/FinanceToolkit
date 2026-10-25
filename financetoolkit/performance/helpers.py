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
    # The intraday bars within each trading day; only available with intraday data.
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


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
