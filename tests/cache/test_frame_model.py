"""Frame Model Tests"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache import frame_model
from financetoolkit.cache.coverage_model import normalize_date


def slice_label_by_label(data, start, end, date_axis=0):
    """Slice the way the cache did before slicing was vectorised, as the reference."""
    axis = frame_model.get_date_axis(data, date_axis)
    start_date, end_date = normalize_date(start), normalize_date(end)
    mask = []

    for label in axis:
        if pd.isna(label):
            mask.append(True)
            continue

        label_date = normalize_date(label)
        mask.append(start_date <= label_date <= end_date)

    return data.loc[mask] if date_axis == 0 else data.loc[:, mask]


@pytest.mark.parametrize(
    "axis",
    [
        pd.period_range("2019-01-01", "2021-12-31", freq="D"),
        pd.period_range("2019-01", "2021-12", freq="M"),
        pd.period_range("2019Q1", "2021Q4", freq="Q"),
        pd.period_range("2015", "2025", freq="Y"),
        pd.date_range("2019-01-01", "2021-12-31", freq="6h"),
        # A late evening New York timestamp keeps its local date, not the UTC one.
        pd.date_range(
            "2019-01-01 23:00", "2021-12-31 23:00", freq="D", tz="America/New_York"
        ),
    ],
)
@pytest.mark.parametrize("date_axis", [0, 1])
def test_slice_frame_matches_label_by_label_slicing(axis, date_axis):
    """Test that vectorised slicing keeps exactly the labels the per-label rules keep."""
    data = pd.DataFrame({"value": np.arange(len(axis))}, index=axis)
    data = data if date_axis == 0 else data.T

    result = frame_model.slice_frame(
        data, "2020-02-29", "2020-12-31", date_axis=date_axis
    )

    pd.testing.assert_frame_equal(
        result,
        slice_label_by_label(data, "2020-02-29", "2020-12-31", date_axis=date_axis),
    )


def test_slice_frame_keeps_missing_dates():
    """Test that a missing date is kept like any other non-date label, not a crash."""
    data = pd.Series(
        [1, 2, 3], index=pd.DatetimeIndex(["2019-01-01", None, "2020-06-01"])
    )

    result = frame_model.slice_frame(data, "2020-01-01", "2020-12-31")

    assert result.tolist() == [2, 3]


def test_get_date_bounds_ignores_missing_dates():
    """Test that the bounds of a period axis are the start of its first and last period."""
    data = pd.Series(
        [1, 2, 3], index=pd.PeriodIndex(["2020-03", None, "2019-01"], freq="M")
    )

    assert frame_model.get_date_bounds(data) == (date(2019, 1, 1), date(2020, 3, 1))
