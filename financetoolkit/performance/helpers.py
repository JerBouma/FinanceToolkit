"""Performance Helpers Module"""

__docformat__ = "google"


import pandas as pd

from financetoolkit.utilities.statistics_model import (
    WITHIN_PERIOD_TRANSLATION,
    to_datetime_index,
)


def determine_within_dataset(
    dataset: pd.DataFrame, period: str, correlation: bool = False
) -> pd.DataFrame:
    """
    Groups a daily dataset, such as the Fama-French factors, by the period each day falls
    in, the "within period" format the factor models of the Performance controller are
    estimated on.

    Args:
        dataset (pd.DataFrame): The daily dataset, with a PeriodIndex.
        period (str): The period to group by (weekly, monthly, quarterly or yearly).
        correlation (bool, optional): Whether to return the correlation matrix of the
            columns per period instead of the daily observations. Defaults to False.

    Raises:
        KeyError: If the period is not one of the keys of WITHIN_PERIOD_TRANSLATION.

    Returns:
        pd.DataFrame: The daily observations with the period and the day as the index
        levels, or with correlation=True the correlation matrix per period with the period
        and the column as the index levels.
    """
    dataset_new = dataset.copy()
    dataset_new.index = pd.DatetimeIndex(dataset_new.to_timestamp().index)
    period_symbol = WITHIN_PERIOD_TRANSLATION[period]

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
