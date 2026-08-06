"""Frame Module"""

__docformat__ = "google"

import contextlib

import pandas as pd

from financetoolkit.cache.coverage_model import Interval, normalize_date


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
    start: str | None = None,
    end: str | None = None,
    date_axis: int = 0,
) -> pd.DataFrame | pd.Series:
    """
    Restrict a frame to the requested date range.

    The cache stores a superset of what any single call asked for, so the stored
    frame is narrowed back down to the requested window before being handed to
    the caller. Labels that cannot be parsed as dates are kept, which matters for
    statement frames that carry non-date columns alongside the reporting periods.

    Args:
        data (pd.DataFrame | pd.Series): The frame to slice.
        start (str | None): Inclusive start of the window. None leaves it open.
        end (str | None): Inclusive end of the window. None leaves it open.
        date_axis (int): 0 when the index holds the dates, 1 when the columns do.

    Returns:
        pd.DataFrame | pd.Series: The frame restricted to the requested window.
    """
    if data is None or len(data) == 0 or (start is None and end is None):
        return data

    axis = get_date_axis(data, date_axis)
    start_date = normalize_date(start) if start else None
    end_date = normalize_date(end) if end else None

    mask = []

    for label in axis:
        try:
            label_date = normalize_date(label)
        except (TypeError, ValueError):
            # Non-date labels are structural rather than temporal, so they are
            # never filtered out by a date window.
            mask.append(True)
            continue

        within_start = start_date is None or label_date >= start_date
        within_end = end_date is None or label_date <= end_date
        mask.append(within_start and within_end)

    return data.loc[mask] if date_axis == 0 else data.loc[:, mask]


def merge_frames(
    existing: pd.DataFrame | pd.Series | None,
    incoming: pd.DataFrame | pd.Series,
    date_axis: int = 0,
) -> pd.DataFrame | pd.Series:
    """
    Combine a newly fetched frame with what the cache already holds.

    Where the two overlap the incoming values win, because they come from a more
    recent call to the source and therefore reflect any revision or restatement.
    The result is sorted along the date axis so the merged frame is
    indistinguishable from one fetched in a single request.

    Args:
        existing (pd.DataFrame | pd.Series | None): The frame already cached, if any.
        incoming (pd.DataFrame | pd.Series): The newly fetched frame.
        date_axis (int): 0 when the index holds the dates, 1 when the columns do.

    Returns:
        pd.DataFrame | pd.Series: The merged frame.
    """
    if existing is None or len(existing) == 0:
        return incoming
    if incoming is None or len(incoming) == 0:
        return existing

    if date_axis == 1:
        merged = pd.concat([existing, incoming], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated(keep="last")]
        sort_axis = 1
    else:
        merged = pd.concat([existing, incoming], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
        sort_axis = 0

    # Axes holding a mix of dates and structural labels are not orderable; leaving
    # them in concatenation order is preferable to failing the merge outright.
    with contextlib.suppress(TypeError):
        merged = merged.sort_index(axis=sort_axis)

    return merged
