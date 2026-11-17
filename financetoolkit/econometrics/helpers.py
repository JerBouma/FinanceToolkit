"""Econometrics Helpers"""

__docformat__ = "google"

import pandas as pd

# The synthetic benchmark series, excluded unless `include_benchmark=True`.
BENCHMARK_TICKER = "Benchmark"

# The observations per year of a series by the median number of days between its observations:
# up to 3 days is daily (252 trading days), up to 9 weekly, up to 44 monthly, up to 119
# quarterly and anything longer yearly.
SPACING_OBSERVATIONS_PER_YEAR = ((4, 252), (10, 52), (45, 12), (120, 4))


def get_observations_per_year(index: pd.Index) -> int:
    """
    The number of observations per year of a series, from the median spacing of its dates,
    so a monthly series given with `Econometrics(data=...)` and held at its own dates (the
    "daily" period) counts 12 observations a year rather than 252.

    Args:
        index (pd.Index): The dates of the observations, a DatetimeIndex or PeriodIndex.

    Returns:
        int: 252, 52, 12, 4 or 1 for a daily, weekly, monthly, quarterly or yearly spacing;
        252 when there are fewer than two dates.
    """
    dates = (
        index.to_timestamp()
        if isinstance(index, pd.PeriodIndex)
        else pd.DatetimeIndex(index)
    )

    if len(dates) < 2:  # noqa: PLR2004
        return SPACING_OBSERVATIONS_PER_YEAR[0][1]

    spacing = pd.Series(dates).diff().dt.days.median()

    for days, observations in SPACING_OBSERVATIONS_PER_YEAR:
        if spacing < days:
            return observations

    return 1
