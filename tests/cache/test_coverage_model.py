"""Coverage Model Tests"""

from datetime import date

import pandas as pd
import pytest

from financetoolkit.cache import coverage_model


def test_normalize_date_accepts_every_representation():
    """Test that all date representations used in the toolkit are supported."""
    expected = date(2020, 1, 15)

    assert coverage_model.normalize_date("2020-01-15") == expected
    assert coverage_model.normalize_date("2020-01-15 00:00:00") == expected
    assert coverage_model.normalize_date(expected) == expected
    assert coverage_model.normalize_date(pd.Timestamp("2020-01-15")) == expected
    assert coverage_model.normalize_date(pd.Period("2020-01-15", freq="D")) == expected


def test_normalize_date_rejects_unsupported_types():
    """Test that an unsupported type raises a TypeError."""
    with pytest.raises(TypeError):
        coverage_model.normalize_date(12345)


def test_merge_intervals_merges_overlapping_and_adjacent():
    """Test that overlapping and directly adjacent intervals collapse into one."""
    merged = coverage_model.merge_intervals(
        [
            (date(2020, 7, 1), date(2020, 12, 31)),
            (date(2020, 1, 1), date(2020, 6, 30)),
            (date(2022, 1, 1), date(2022, 12, 31)),
        ]
    )

    assert merged == [
        (date(2020, 1, 1), date(2020, 12, 31)),
        (date(2022, 1, 1), date(2022, 12, 31)),
    ]


def test_merge_intervals_drops_invalid_intervals():
    """Test that intervals whose start is after their end are discarded."""
    assert coverage_model.merge_intervals([(date(2020, 5, 1), date(2020, 1, 1))]) == []


def test_subtract_intervals_returns_nothing_when_fully_covered():
    """Test that a fully covered request produces no gaps."""
    missing = coverage_model.subtract_intervals(
        (date(2020, 1, 1), date(2020, 12, 31)),
        [(date(2019, 1, 1), date(2021, 12, 31))],
    )

    assert missing == []


def test_subtract_intervals_returns_widened_edges_only():
    """Test that widening a date range only asks for the new years."""
    missing = coverage_model.subtract_intervals(
        (date(2015, 1, 1), date(2025, 1, 1)),
        [(date(2020, 1, 1), date(2023, 1, 1))],
    )

    assert missing == [
        (date(2015, 1, 1), date(2019, 12, 31)),
        (date(2023, 1, 2), date(2025, 1, 1)),
    ]


def test_subtract_intervals_returns_interior_gap():
    """Test that a hole between two cached ranges is detected."""
    missing = coverage_model.subtract_intervals(
        (date(2020, 1, 1), date(2020, 12, 31)),
        [
            (date(2020, 1, 1), date(2020, 3, 31)),
            (date(2020, 7, 1), date(2020, 12, 31)),
        ],
    )

    assert missing == [(date(2020, 4, 1), date(2020, 6, 30))]


def test_subtract_intervals_with_no_coverage_returns_full_range():
    """Test that an empty cache produces the whole requested range."""
    target = (date(2020, 1, 1), date(2020, 12, 31))

    assert coverage_model.subtract_intervals(target, []) == [target]


def test_intersect_intervals():
    """Test the overlap of two intervals, including the disjoint case."""
    overlap = coverage_model.intersect_intervals(
        (date(2020, 1, 1), date(2020, 6, 30)),
        (date(2020, 4, 1), date(2020, 12, 31)),
    )

    assert overlap == (date(2020, 4, 1), date(2020, 6, 30))
    assert (
        coverage_model.intersect_intervals(
            (date(2020, 1, 1), date(2020, 2, 1)), (date(2021, 1, 1), date(2021, 2, 1))
        )
        is None
    )


def test_total_days_counts_inclusive_days():
    """Test that the day count is inclusive and deduplicates overlaps."""
    assert coverage_model.total_days([(date(2020, 1, 1), date(2020, 1, 10))]) == 10
    assert (
        coverage_model.total_days(
            [
                (date(2020, 1, 1), date(2020, 1, 10)),
                (date(2020, 1, 5), date(2020, 1, 10)),
            ]
        )
        == 10
    )
