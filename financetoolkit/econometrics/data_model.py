"""Econometrics Data Preparation Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# The periods the Econometrics methods take, with the pandas frequency of each.
PERIODS = {"weekly": "W-SUN", "monthly": "M", "quarterly": "Q-DEC", "yearly": "Y-DEC"}

# How the change of a series from one observation to the next is calculated.
CHANGES = ("percentage", "log", "difference", "none")

# The columns the methods read: the series as given (named like the price columns of the
# Toolkit's historical data, so every method's default works) and its changes (the returns).
LEVEL_COLUMNS = ("Adj Close", "Close", "Level")
CHANGE_COLUMNS = ("Return", "Change")


def _to_daily_periods(index: pd.Index) -> pd.PeriodIndex:
    """
    Converts dates, periods or date strings into daily periods. A monthly, quarterly or
    yearly period becomes its last day, so that the period it covers stays the same when
    the data is resampled.

    Args:
        index (pd.Index): The dates.

    Returns:
        pd.PeriodIndex: The dates as daily periods.
    """
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("D", how="end")
    if isinstance(index, pd.DatetimeIndex):
        return index.to_period("D")
    return pd.DatetimeIndex(pd.to_datetime(index.astype(str))).to_period("D")


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
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.index.name = None

    if frame.empty:
        raise ValueError("The data has no numbers to analyze.")

    return frame


def get_changes(levels: pd.DataFrame, changes: str) -> pd.DataFrame:
    """
    The change of every series from one observation to the next.

    Args:
        levels (pd.DataFrame): The series.
        changes (str): "percentage" (the relative change, like a return), "log" (the log
            change), "difference" (the first difference, for rates and ratios that can be
            zero or negative) or "none" (the series are changes already).

    Returns:
        pd.DataFrame: The changes; the first observation of each series is NaN.
    """
    if changes == "percentage":
        return levels.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if changes == "log":
        return np.log(levels.where(levels > 0)).diff()
    if changes == "difference":
        return levels.diff()
    if changes == "none":
        return levels
    raise ValueError(
        f"The changes must be one of {', '.join(CHANGES)}, not {changes!r}."
    )


def build_historical_data(
    data: pd.DataFrame | pd.Series | dict,
    changes: str = "percentage",
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
        changes (str, optional): How the changes are calculated, see get_changes.
            Defaults to "percentage".
        start_date (str | None, optional): The first date to keep. Defaults to None.
        end_date (str | None, optional): The last date to keep. Defaults to None.

    Returns:
        tuple[dict[str, pd.DataFrame], list[str]]: The historical data per period
        (including an empty "intraday" period) and the names of the series.
    """
    if changes not in CHANGES:
        raise ValueError(
            f"The changes must be one of {', '.join(CHANGES)}, not {changes!r}."
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
