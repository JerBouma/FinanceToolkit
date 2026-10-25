"""Econometrics Data Preparation Model"""

__docformat__ = "google"

import datetime

import numpy as np
import pandas as pd

# The periods the Econometrics methods take, with the pandas frequency of each.
PERIODS = {"weekly": "W-SUN", "monthly": "M", "quarterly": "Q-DEC", "yearly": "Y-DEC"}

# How the change of a series from one observation to the next is calculated.
CHANGES = ("percentage", "log", "difference", None)

# The columns the methods read: the series as given (named like the price columns of the
# Toolkit's historical data, so every method's default works) and its changes (the returns).
LEVEL_COLUMNS = ("Adj Close", "Close", "Level")
CHANGE_COLUMNS = ("Return", "Change")


def _to_daily_period(value: object) -> pd.Period:
    """
    Converts one date, period, string or number into a daily period, see _to_daily_periods.

    Args:
        value (object): The date.

    Returns:
        pd.Period: The date as a daily period.
    """
    if isinstance(value, pd.Period):
        return value.asfreq("D", how="end")
    if isinstance(value, datetime.date | np.datetime64):
        return pd.Period(pd.Timestamp(value), "D")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    # A string or number names the period it is written as: "2020" (or 2020) a year,
    # "2020Q1" a quarter, "2020-01" a month and "2020-01-15" a day. Read as that period
    # and taken at its last day, a year written as text meets the same year as a yearly
    # period (December 31) instead of landing on January 1 next to it.
    try:
        return pd.Period(str(value)).asfreq("D", how="end")
    except (ValueError, TypeError):
        return pd.Period(pd.to_datetime(str(value), format="mixed"), "D")


def _to_daily_periods(index: pd.Index) -> pd.PeriodIndex:
    """
    Converts dates, periods, date strings or years into daily periods. A monthly,
    quarterly or yearly period becomes its last day, so that the period it covers stays
    the same when the data is resampled; a year, quarter or month written as text or a
    number ("2020", 2020, "2020Q1", "2020-01") is read as that period and becomes its last
    day as well. An index that mixes them, such as yearly ratios combined with daily
    prices, is converted value by value.

    Args:
        index (pd.Index): The dates.

    Returns:
        pd.PeriodIndex: The dates as daily periods.
    """
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("D", how="end")
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period("D")
    return pd.PeriodIndex([_to_daily_period(value) for value in index], freq="D")


def looks_like_dates(index: pd.Index) -> bool:
    """Whether an index holds dates or periods (or strings and numbers that read as years or dates)."""
    if isinstance(index, pd.PeriodIndex | pd.DatetimeIndex):
        return True
    if isinstance(index, pd.MultiIndex) or len(index) == 0:
        return False
    try:
        pd.to_datetime(index.astype(str), format="mixed")
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def to_series_frame(data: pd.DataFrame | pd.Series | dict) -> pd.DataFrame:
    """
    Arranges any metric into a DataFrame with a column per series and a row per date:

    - A Series becomes one column, named after the Series (or "Value").
    - A dictionary of Series (or DataFrames) is combined, with the keys as names.
    - Data with the dates in the columns, such as the ratios of the Toolkit (tickers by
      year), is transposed.
    - Columns with several levels, such as a yield curve (country and maturity), are named
      by their levels joined with a space ("United States 10Y").

    Args:
        data (pd.DataFrame | pd.Series | dict): The data.

    Returns:
        pd.DataFrame: The series as numbers, indexed by daily periods in date order.

    Raises:
        ValueError: When the data has no dates in its rows or columns, or no numbers.
    """
    if isinstance(data, dict):
        frames = [
            (
                to_series_frame(part.rename(str(name)))
                if isinstance(part, pd.Series)
                else to_series_frame(part).add_prefix(f"{name} ")
            )
            for name, part in data.items()
        ]
        frame = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    elif isinstance(data, pd.Series):
        frame = data.to_frame(str(data.name) if data.name is not None else "Value")
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        raise TypeError(
            "The data must be a DataFrame, a Series or a dictionary of them, not a "
            f"{type(data).__name__}."
        )

    if not looks_like_dates(frame.index) and looks_like_dates(frame.columns):
        frame = frame.T
    if not looks_like_dates(frame.index):
        raise ValueError(
            "The data needs dates (or periods such as years) in its index or columns, so "
            "that it can be analyzed over time."
        )

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            " ".join(str(level) for level in column if str(level) != "")
            for column in frame.columns
        ]
    frame.columns = [str(column) for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicated = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError(
            f"The series {', '.join(duplicated)} appear more than once; give every series "
            "its own name."
        )

    frame.index = _to_daily_periods(frame.index)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[:, frame.notna().any()]
    # Rows that land on the same day are merged per series (the last value each has), as a
    # yearly ratio on December 31 next to a daily price of that day are two rows with one
    # value each; keeping only the last row would drop the ratio.
    frame = frame.groupby(level=0, sort=True).last()
    frame.index.name = None

    if frame.empty:
        raise ValueError("The data has no numbers to analyze.")

    return frame


def get_changes(levels: pd.DataFrame, changes: str | None) -> pd.DataFrame:
    """
    The change of every series from one observation to the next. Every series is taken
    at its own observations, so a monthly series next to a daily one gets its monthly
    changes, and a series that is not observed on a holiday gets the change over the gap
    rather than none.

    Args:
        levels (pd.DataFrame): The series.
        changes (str | None): "percentage" (the relative change, like a return), "log" (the log
            change), "difference" (the first difference, for rates and ratios that can be
            zero or negative) or None (the series are changes already). The percentage
            change is (x - previous) / |previous|, so a rise from -10 to -5 is +50%
            rather than -50%; a change from zero is NaN.

    Returns:
        pd.DataFrame: The changes at the dates of the levels; the first observation of each
        series, and every date it is not observed, is NaN.
    """
    if changes not in CHANGES:
        raise ValueError(
            "The changes must be 'percentage', 'log', 'difference' or None, "
            f"not {changes!r}."
        )
    if changes is None:
        return levels

    def change(column: pd.Series) -> pd.Series:
        # Row by row on the shared dates, every change next to a missing value is NaN: a
        # monthly series among daily ones would have no changes at all.
        observed = column.dropna()
        if changes == "percentage":
            # Dividing by the absolute previous value keeps the sign of the change when
            # the base is negative (a loss shrinking from -10 to -5 is an increase).
            return (observed.diff() / observed.shift().abs()).replace(
                [np.inf, -np.inf], np.nan
            )
        if changes == "log":
            return np.log(observed.where(observed > 0)).diff()
        return observed.diff()

    return levels.apply(change).reindex(levels.index)


def build_historical_data(
    data: pd.DataFrame | pd.Series | dict,
    changes: str | None = "percentage",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Builds the historical data the Econometrics methods read from any metric: per period
    a DataFrame with the series as given under "Adj Close", "Close" and "Level" and their
    changes under "Return" and "Change", with a column per series below each.

    The "daily" period holds the series at the dates they are observed, whatever their
    frequency, so a monthly series is analyzed monthly by default. The weekly, monthly,
    quarterly and yearly periods take the last observation of each period, and the
    changes are calculated per period.

    Args:
        data (pd.DataFrame | pd.Series | dict): The data, see to_series_frame.
        changes (str | None, optional): How the changes are calculated, see get_changes.
            Defaults to "percentage".
        start_date (str | None, optional): The first date to keep. Defaults to None.
        end_date (str | None, optional): The last date to keep. Defaults to None.

    Returns:
        tuple[dict[str, pd.DataFrame], list[str]]: The historical data per period
        (including an empty "intraday" period) and the names of the series.
    """
    if changes not in CHANGES:
        raise ValueError(
            f"The changes must be one of 'percentage', 'log', 'difference' or None, not {changes!r}."
        )

    levels = to_series_frame(data)
    if start_date:
        levels = levels.loc[levels.index >= pd.Period(start_date, "D")]
    if end_date:
        levels = levels.loc[levels.index <= pd.Period(end_date, "D")]
    if levels.empty:
        raise ValueError(
            f"The data has no observations between {start_date} and {end_date}."
        )

    def combine(period_levels: pd.DataFrame) -> pd.DataFrame:
        period_changes = get_changes(period_levels, changes)
        return pd.concat(
            {
                **{column: period_levels for column in LEVEL_COLUMNS},
                **{column: period_changes for column in CHANGE_COLUMNS},
            },
            axis=1,
        )

    historical_data = {"intraday": pd.DataFrame(), "daily": combine(levels)}
    for period, frequency in PERIODS.items():
        resampled = levels.groupby(levels.index.asfreq(frequency)).last()
        resampled.index.name = None
        historical_data[period] = combine(resampled.dropna(how="all"))

    return historical_data, list(levels.columns)
