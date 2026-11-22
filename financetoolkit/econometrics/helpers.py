"""Econometrics Helpers"""

__docformat__ = "google"

import importlib
from collections.abc import Callable
from types import ModuleType

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
        int: 252, 52, 12, 4 or 1 for a daily, weekly, monthly, quarterly or yearly spacing
        (365 for a daily series with regular Saturday and Sunday observations); 252 when
        there are fewer than two dates.
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
            # A daily series that also trades on weekends, such as a cryptocurrency, has
            # 365 observations a year; a stock with an odd weekend session keeps 252.
            weekdays = dates.dayofweek
            if (
                observations == SPACING_OBSERVATIONS_PER_YEAR[0][1]
                and (weekdays == 5).mean() > 0.1  # noqa: PLR2004
                and (weekdays == 6).mean() > 0.1  # noqa: PLR2004
            ):
                return 365
            return observations

    return 1


def calculate_per_series(
    data: pd.DataFrame,
    function: Callable[..., pd.Series | pd.DataFrame],
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculates a rolling statistic per series over the dates the series has a value, and
    places the result back on every date of the data, NaN on the other dates.

    A window counts observations, so the dates on which a ticker did not trade (NaN in the
    prices the Econometrics module reads, see `mask_non_trading_days`) or a series observed
    at other dates than the others are left out rather than counted as periods without a
    move.

    Args:
        data (pd.DataFrame): The series, a column per series.
        function (Callable[..., pd.Series | pd.DataFrame]): The statistic, which takes a
            Series followed by `args` and `kwargs`.
        *args: The further positional arguments of the function, such as the window.
        **kwargs: The further keyword arguments of the function.

    Returns:
        pd.DataFrame: The statistic with a column per series when the function returns a
        Series, and (statistic column, series) columns when it returns a DataFrame.
    """
    results = {
        series: function(data[series].dropna(), *args, **kwargs).reindex(data.index)
        for series in data.columns
    }

    if all(isinstance(result, pd.Series) for result in results.values()):
        return pd.DataFrame(results, index=data.index)

    return pd.concat(results, axis=1).swaplevel(1, 0, axis=1).sort_index(axis=1)


def import_optional(module_name: str, feature: str) -> ModuleType:
    """
    Imports a module of statsmodels or linearmodels when a method first needs it, so the
    Econometrics module itself works without the optional `financetoolkit[econometrics]`
    extra and only the methods built on these libraries ask for it.

    Args:
        module_name (str): The module, e.g. "statsmodels.tsa.arima.model".
        feature (str): The method or feature that needs it, named in the error.

    Returns:
        ModuleType: The imported module.

    Raises:
        ImportError: When the library is not installed, saying how to install it.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"{feature} needs {module_name.partition('.')[0]}. Install it with: "
            "pip install financetoolkit[econometrics]"
        ) from error
