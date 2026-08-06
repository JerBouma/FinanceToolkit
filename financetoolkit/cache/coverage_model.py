"""Coverage Module"""

__docformat__ = "google"

from datetime import date, datetime, timedelta

import pandas as pd

# An interval is an inclusive (start, end) pair of dates. The cache records which
# intervals it has already fetched from a source so that a later request only has
# to ask the API for the parts it is missing, rather than the entire range again.
Interval = tuple[date, date]

ONE_DAY = timedelta(days=1)


def normalize_date(value: str | date | datetime | pd.Timestamp | pd.Period) -> date:
    """
    Convert any of the date representations used throughout the toolkit into a
    plain ``datetime.date``.

    The toolkit passes dates around as ISO strings ("2020-01-01"), pandas
    Timestamps and, for historical data, pandas Periods. Interval arithmetic is
    only well defined on a single representation, so everything is funnelled
    through this function first.

    Args:
        value (str | date | datetime | pd.Timestamp | pd.Period): The date to convert.

    Returns:
        date: The equivalent ``datetime.date``.

    Raises:
        TypeError: If the value is not one of the supported date representations.
    """
    if isinstance(value, pd.Period):
        return value.to_timestamp().date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value[:10], "%Y-%m-%d").date()

    raise TypeError(
        f"Unsupported date type ({type(value)}), expected a string, date, "
        "datetime, Timestamp or Period."
    )


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """
    Collapse a list of intervals into the smallest equivalent list.

    Overlapping and directly adjacent intervals are merged, so that
    [(2020-01-01, 2020-06-30), (2020-07-01, 2020-12-31)] becomes a single
    interval covering the whole year. Adjacency counts as overlap because a
    one day gap between two fetched ranges is not a real gap in the data.

    Args:
        intervals (list[Interval]): The intervals to merge, in any order.

    Returns:
        list[Interval]: Disjoint, non-adjacent intervals sorted by start date.
    """
    valid_intervals = [
        (start, end) for start, end in intervals if start is not None and start <= end
    ]

    if not valid_intervals:
        return []

    merged: list[Interval] = []

    for start, end in sorted(valid_intervals):
        if merged and start <= merged[-1][1] + ONE_DAY:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    return merged


def subtract_intervals(target: Interval, covered: list[Interval]) -> list[Interval]:
    """
    Determine which parts of a requested interval are not yet covered.

    This is the core of the incremental cache: given the range the user asked
    for and the ranges already stored, it returns only the gaps that still have
    to be requested from the external API.

    Args:
        target (Interval): The inclusive interval that was requested.
        covered (list[Interval]): The intervals already present in the cache.

    Returns:
        list[Interval]: The parts of ``target`` not covered by ``covered``, sorted
            by start date. An empty list means the request is fully cached.
    """
    target_start, target_end = target

    if target_start > target_end:
        return []

    missing: list[Interval] = []
    cursor = target_start

    for start, end in merge_intervals(covered):
        if end < cursor:
            continue
        if start > target_end:
            break
        if start > cursor:
            missing.append((cursor, min(start - ONE_DAY, target_end)))
        cursor = max(cursor, end + ONE_DAY)
        if cursor > target_end:
            break

    if cursor <= target_end:
        missing.append((cursor, target_end))

    return missing


def intersect_intervals(first: Interval, second: Interval) -> Interval | None:
    """
    Compute the overlap between two intervals.

    Args:
        first (Interval): The first inclusive interval.
        second (Interval): The second inclusive interval.

    Returns:
        Interval | None: The overlapping interval, or None when they are disjoint.
    """
    start = max(first[0], second[0])
    end = min(first[1], second[1])

    return (start, end) if start <= end else None


def total_days(intervals: list[Interval]) -> int:
    """
    Count the number of days spanned by a list of intervals.

    Used for logging and for deciding whether a partial refetch is actually
    cheaper than simply requesting the full range again.

    Args:
        intervals (list[Interval]): The intervals to measure.

    Returns:
        int: The total number of inclusive days covered, after merging overlaps.
    """
    return sum((end - start).days + 1 for start, end in merge_intervals(intervals))
