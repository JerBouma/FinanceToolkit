"""Frame Module"""

__docformat__ = "google"

import contextlib
from datetime import date

import numpy as np
import pandas as pd

from financetoolkit.cache import serialization_model
from financetoolkit.cache.coverage_model import Interval, normalize_date
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


def get_date_axis(data: pd.DataFrame | pd.Series, date_axis: int = 0) -> pd.Index:
    """
    Return the axis of a frame that carries the dates.

    Most toolkit data is indexed by date, but financial statements are laid out
    with line items as rows and reporting periods as columns. Both layouts have
    to be cached, so the date carrying axis is explicit rather than assumed.

    Args:
        data (pd.DataFrame | pd.Series): The frame to inspect.
        date_axis (int): 0 when the index holds the dates, 1 when the columns do.

    Returns:
        pd.Index: The axis holding the dates.

    Raises:
        ValueError: If ``date_axis`` is not 0 or 1.
    """
    if date_axis == 0:
        return data.index
    if date_axis == 1:
        return data.columns

    raise ValueError(f"The date_axis must be 0 or 1, received {date_axis}.")


def axis_as_dates(axis: pd.Index) -> pd.DatetimeIndex | None:
    """
    Convert a date or period axis to midnight timestamps in one vectorised step.

    Converting label by label with normalize_date costs a pandas Period conversion
    per label, which dominated reading ten years of daily prices back from the cache.
    Periods map to their start, as normalize_date does, and timezone-aware dates keep
    their local calendar date.

    Args:
        axis (pd.Index): The axis holding the dates.

    Returns:
        pd.DatetimeIndex | None: The dates, NaT where a label is missing, or None when
            the axis is not a date or period axis and has to be converted label by label.
    """
    if isinstance(axis, pd.PeriodIndex):
        return axis.to_timestamp(how="start").normalize()
    if isinstance(axis, pd.DatetimeIndex):
        if axis.tz is not None:
            axis = axis.tz_localize(None)

        return axis.normalize()

    return None


def get_date_bounds(
    data: pd.DataFrame | pd.Series, date_axis: int = 0
) -> Interval | None:
    """
    Determine the first and last date present in a frame.

    Args:
        data (pd.DataFrame | pd.Series): The frame to inspect.
        date_axis (int): 0 when the index holds the dates, 1 when the columns do.

    Returns:
        Interval | None: The inclusive (first, last) date pair, or None when the
            frame is empty or its axis holds no parseable dates.
    """
    if data is None or len(data) == 0:
        return None

    axis = get_date_axis(data, date_axis)

    if len(axis) == 0:
        return None

    vectorised = axis_as_dates(axis)

    if vectorised is not None:
        vectorised = vectorised.dropna()

        if vectorised.empty:
            return None

        return (vectorised.min().date(), vectorised.max().date())

    dates = []

    for label in axis:
        try:
            dates.append(normalize_date(label))
        except (TypeError, ValueError):
            continue

    if not dates:
        return None

    return (min(dates), max(dates))


def slice_frame(
    data: pd.DataFrame | pd.Series,
    start: str | date | None = None,
    end: str | date | None = None,
    date_axis: int = 0,
) -> pd.DataFrame | pd.Series:
    """
    Restrict a frame to the requested date range.

    The cache stores a superset of what any single call asked for, so the stored
    frame is narrowed back down to the requested window before being handed to
    the caller. Labels that cannot be parsed as dates are kept, which matters for
    statement frames that carry non-date columns alongside the reporting periods.

    A period is kept when it overlaps the window, as a fresh request truncates it: a
    start date of 31 December 2014 keeps the year 2014. A cached frame is then the same
    as the one a fresh request returns, so no answer depends on what the cache holds.

    Args:
        data (pd.DataFrame | pd.Series): The frame to slice.
        start (str | date | None): Inclusive start of the window. None leaves it open.
        end (str | date | None): Inclusive end of the window. None leaves it open.
        date_axis (int): 0 when the index holds the dates, 1 when the columns do.

    Returns:
        pd.DataFrame | pd.Series: The frame restricted to the requested window.
    """
    if data is None or len(data) == 0 or (start is None and end is None):
        return data

    axis = get_date_axis(data, date_axis)
    start_date = normalize_date(start) if start else None
    end_date = normalize_date(end) if end else None

    vectorised = axis_as_dates(axis)

    if vectorised is not None:
        # Missing labels are kept, as the label-by-label path below keeps anything that
        # is not a date.
        keep = vectorised.isna()
        within = np.ones(len(axis), dtype=bool)
        last_days = (
            axis.to_timestamp(how="end").normalize()
            if isinstance(axis, pd.PeriodIndex)
            else vectorised
        )

        if start_date is not None:
            within &= last_days >= pd.Timestamp(start_date)
        if end_date is not None:
            within &= vectorised <= pd.Timestamp(end_date)

        mask = np.asarray(keep) | within

        return data.loc[mask] if date_axis == 0 else data.loc[:, mask]

    mask = []

    for label in axis:
        try:
            label_date = normalize_date(label)
        except (TypeError, ValueError):
            # Non-date labels are structural, so a date window never filters them out.
            mask.append(True)
            continue

        last_day = label.end_time.date() if isinstance(label, pd.Period) else label_date
        within_start = start_date is None or last_day >= start_date
        within_end = end_date is None or label_date <= end_date
        mask.append(within_start and within_end)

    return data.loc[mask] if date_axis == 0 else data.loc[:, mask]


def _drop_non_date_rows(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Keep only the rows labelled with a date when dates and other labels are mixed.

    Rows hold dates, so a label that is no date (a row labelled 0 that an empty
    response once carried) makes the index unorderable and every date slice of it
    fail. An index without any date is left as it is.

    Args:
        data (pd.DataFrame | pd.Series): The frame whose index is checked.

    Returns:
        pd.DataFrame | pd.Series: The frame with only its dated rows, as a
            PeriodIndex or DatetimeIndex when the dates allow it.
    """
    is_date = np.array(
        [isinstance(label, (pd.Period, pd.Timestamp)) for label in data.index]
    )

    if not is_date.any() or (is_date.all() and data.index.dtype != object):
        return data

    data = data.loc[is_date]
    labels = list(data.index)

    with contextlib.suppress(TypeError, ValueError):
        if all(isinstance(label, pd.Period) for label in labels):
            data.index = pd.PeriodIndex(labels, name=data.index.name)
        elif all(isinstance(label, pd.Timestamp) for label in labels):
            data.index = pd.DatetimeIndex(labels, name=data.index.name)

    return data


def tidy_date_axis(
    data: pd.DataFrame | pd.Series, date_axis: int = 0
) -> pd.DataFrame | pd.Series:
    """
    Put a date or period axis in date order, without missing or repeated dates.

    Every date slice (`.loc[start:end]`) needs an ordered axis, and a missing date
    (NaT) makes pandas treat the axis as unordered. A stored frame is tidied both when
    it is written and when it is read, so a copy stored in another shape (an older
    version, or a source that returned a row without a date) is repaired on its next
    read rather than breaking every call that reads it. Axes that hold labels rather
    than dates, such as statement line items, are returned as they are.

    Args:
        data (pd.DataFrame | pd.Series): The frame to tidy.
        date_axis (int): 0 when the index holds the dates, 1 when the columns do.

    Returns:
        pd.DataFrame | pd.Series: The frame with an ordered date axis.
    """
    if data is None or len(data) == 0:
        return data

    axis = get_date_axis(data, date_axis)

    if axis_as_dates(axis) is None and date_axis == 0:
        data = _drop_non_date_rows(data)
        axis = data.index

    if axis_as_dates(axis) is None:
        return data

    keep = np.asarray(axis.notna() & ~axis.duplicated(keep="last"))

    if not keep.all():
        data = data.loc[keep] if date_axis == 0 else data.loc[:, keep]

    if not get_date_axis(data, date_axis).is_monotonic_increasing:
        data = data.sort_index(axis=date_axis)

    return data


def merge_frames(
    existing: pd.DataFrame | pd.Series | None,
    incoming: pd.DataFrame | pd.Series,
    date_axis: int = 0,
    replaced: Interval | None = None,
) -> pd.DataFrame | pd.Series:
    """
    Combine a newly fetched frame with what the cache already holds.

    Where the two overlap the incoming values win, because they come from a more
    recent call to the source and therefore reflect any revision or restatement.
    Within the replaced range the incoming frame is the whole answer of the source,
    so a date it no longer holds (a removed bar, a corrected date) is dropped from
    the existing frame rather than kept beside the new data. An empty incoming frame
    leaves the existing one as it is, since an empty answer can be a rate limit.
    The result is sorted along the date axis so the merged frame is
    indistinguishable from one fetched in a single request.

    Args:
        existing (pd.DataFrame | pd.Series | None): The frame already cached, if any.
        incoming (pd.DataFrame | pd.Series): The newly fetched frame.
        date_axis (int): 0 when the index holds the dates, 1 when the columns do.
        replaced (Interval | None): The inclusive range the incoming frame answers in
            full, whose existing dates are dropped first. Labels that are not dates are
            kept. Defaults to None, which drops nothing.

    Returns:
        pd.DataFrame | pd.Series: The merged frame.
    """
    if existing is None or len(existing) == 0:
        return tidy_date_axis(incoming, date_axis)
    if incoming is None or len(incoming) == 0:
        return tidy_date_axis(existing, date_axis)

    if replaced is not None:
        axis = get_date_axis(existing, date_axis)
        dates = axis_as_dates(axis)
        start, end = pd.Timestamp(replaced[0]), pd.Timestamp(replaced[1])

        if dates is not None:
            within = np.asarray((dates >= start) & (dates <= end))
        else:
            within = []
            for label in axis:
                try:
                    within.append(start <= pd.Timestamp(normalize_date(label)) <= end)
                except (TypeError, ValueError):
                    within.append(False)
            within = np.asarray(within, dtype=bool)

        existing = existing.loc[~within] if date_axis == 0 else existing.loc[:, ~within]

    if date_axis == 1:
        merged = pd.concat([existing, incoming], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
        sort_axis = 1
    else:
        merged = pd.concat([existing, incoming], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
        sort_axis = 0

    # Axes mixing dates and labels are not orderable; keep concatenation order.
    with contextlib.suppress(TypeError):
        merged = merged.sort_index(axis=sort_axis)

    return tidy_date_axis(merged, date_axis)


def merge_payload(
    existing_payload: bytes | None,
    data: pd.DataFrame | pd.Series,
    date_axis: int,
    coverage: Interval | None,
    source: str,
    dataset: str,
    entity: str,
) -> bytes:
    """
    Merge the incoming frame into whatever the cache holds right now.

    Args:
        existing_payload (bytes | None): The payload stored under the write lock, or None
            when the entity has never been cached.
        data (pd.DataFrame | pd.Series): The incoming frame.
        date_axis (int): The axis that holds the dates, 0 for rows and 1 for columns.
        coverage (Interval | None): The range the incoming frame covers, see merge_frames.
        source (str): The source of the data, for the log message.
        dataset (str): The dataset of the data, for the log message.
        entity (str): The entity of the data, for the log message.

    Returns:
        bytes: The payload to store in its place.
    """
    # A payload that can't be read (another format, such as a pickle) or merged
    # (a frame of another layout) is replaced by the incoming frame, rather than
    # failing the write and leaving the broken payload in place.
    if existing_payload is not None:
        try:
            existing = serialization_model.decode_dataframe(existing_payload)

            return serialization_model.encode_dataframe(
                merge_frames(existing, data, date_axis, coverage)
            )
        except Exception as error:  # noqa: BLE001
            logger.debug(
                "Replacing the cached %s.%s for %s that could not be merged: %s",
                source,
                dataset,
                entity,
                error,
            )

    return serialization_model.encode_dataframe(merge_frames(None, data, date_axis))
