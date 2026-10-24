"""Econometrics Data Preparation Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics import data_model

# pylint: disable=missing-function-docstring


def test_series_and_monthly_periods():
    series = pd.Series(
        [0.02, 0.025, 0.03],
        index=pd.period_range("2024-01", periods=3, freq="M"),
        name="Inflation",
    )
    frame = data_model.to_series_frame(series)

    assert list(frame.columns) == ["Inflation"]
    # A month becomes its last day, so resampling keeps it in the same month.
    assert str(frame.index[0]) == "2024-01-31"


def test_dates_in_the_columns_are_transposed():
    ratios = pd.DataFrame(
        [[30.1, 28.2, 25.0], [35.0, 33.3, 31.2]],
        index=["AAPL", "MSFT"],
        columns=pd.period_range("2021", periods=3, freq="Y"),
    )
    frame = data_model.to_series_frame(ratios)

    assert list(frame.columns) == ["AAPL", "MSFT"]
    assert len(frame) == 3  # noqa: PLR2004


def test_multiple_column_levels_and_dictionaries():
    curve = pd.DataFrame(
        [[0.04, 0.045], [0.041, 0.046]],
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        columns=pd.MultiIndex.from_tuples(
            [("United States", "2Y"), ("United States", "10Y")]
        ),
    )
    assert list(data_model.to_series_frame(curve).columns) == [
        "United States 2Y",
        "United States 10Y",
    ]

    combined = data_model.to_series_frame(
        {"Spread": curve[("United States", "10Y")] - curve[("United States", "2Y")]}
    )
    assert list(combined.columns) == ["Spread"]


def test_data_without_dates_or_with_duplicate_names():
    with pytest.raises(ValueError, match="dates"):
        data_model.to_series_frame(pd.DataFrame({"A": [1, 2]}, index=["x", "y"]))

    duplicated = pd.DataFrame(
        [[1.0, 2.0]], index=pd.to_datetime(["2026-01-02"]), columns=["A", "A"]
    )
    with pytest.raises(ValueError, match="more than once"):
        data_model.to_series_frame(duplicated)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ("percentage", 0.1),
        ("log", np.log(1.1)),
        ("difference", 10.0),
        ("none", 110.0),
    ],
)
def test_changes(changes, expected):
    levels = pd.DataFrame({"A": [100.0, 110.0]})
    assert data_model.get_changes(levels, changes)["A"].iloc[1] == pytest.approx(
        expected
    )


def test_historical_data_per_period():
    index = pd.bdate_range("2024-01-01", "2025-12-31")
    data = pd.DataFrame(
        {"A": np.linspace(100, 120, len(index)), "B": np.linspace(50, 40, len(index))},
        index=index,
    )
    historical_data, names = data_model.build_historical_data(
        data, start_date="2024-06-01"
    )

    assert names == ["A", "B"]
    assert historical_data["intraday"].empty
    daily = historical_data["daily"]
    assert {"Adj Close", "Level", "Return", "Change"} <= set(
        daily.columns.get_level_values(0)
    )
    assert str(daily.index[0]) >= "2024-06-01"
    monthly = historical_data["monthly"]
    assert monthly.index.freqstr == "M"
    # A monthly level is the last observation of the month, its change the change of those.
    assert monthly["Adj Close"]["A"].iloc[1] == pytest.approx(
        data.loc["2024-07", "A"].iloc[-1]
    )
    assert monthly["Return"]["A"].iloc[1] == pytest.approx(
        monthly["Adj Close"]["A"].iloc[1] / monthly["Adj Close"]["A"].iloc[0] - 1
    )
    with pytest.raises(ValueError, match="changes"):
        data_model.build_historical_data(data, changes="ratio")
