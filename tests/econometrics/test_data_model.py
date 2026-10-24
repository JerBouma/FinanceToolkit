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


def test_years_written_as_text_or_numbers_end_on_december_31():
    yearly = pd.Series(
        [1.0, 2.0], index=pd.period_range("2020", periods=2, freq="Y"), name="P"
    )
    as_text = pd.Series([1.0, 2.0], index=["2020", "2021"], name="T")
    as_numbers = pd.Series([1.0, 2.0], index=[2020, 2021], name="N")

    frame = data_model.to_series_frame({"P": yearly, "T": as_text, "N": as_numbers})

    # All three name the same years, so they meet on the same dates.
    assert [str(date) for date in frame.index] == ["2020-12-31", "2021-12-31"]
    assert frame.notna().all().all()
    quarter_and_month = data_model.to_series_frame(
        pd.Series([1.0, 2.0], index=["2020Q1", "2020-05"])
    )
    assert [str(date) for date in quarter_and_month.index] == [
        "2020-03-31",
        "2020-05-31",
    ]


def test_yearly_and_daily_series_combined_in_one_frame():
    ratio = pd.DataFrame(
        {"PE": [20.0, 25.0]}, index=pd.period_range("2020", periods=2, freq="Y")
    )
    price = pd.DataFrame(
        {"Price": np.arange(1.0, 801.0)},
        index=pd.period_range("2020-01-01", periods=800, freq="D"),
    )
    # Concatenating periods of different frequencies gives an index of mixed objects.
    frame = data_model.to_series_frame(pd.concat([ratio, price], axis=1))

    assert frame["PE"].dropna().to_dict() == {
        pd.Period("2020-12-31", "D"): 20.0,
        pd.Period("2021-12-31", "D"): 25.0,
    }
    assert frame["Price"].notna().sum() == 800  # noqa: PLR2004


def test_changes_per_series_with_mixed_frequencies_and_calendars():
    days = pd.bdate_range("2023-01-02", "2025-12-31")
    rng = np.random.default_rng(0)
    daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), days)
    months = pd.date_range("2023-01-31", periods=36, freq="ME")
    monthly = pd.Series(np.linspace(100, 135, 36), months)
    # The second daily series skips a holiday the first one trades on.
    holiday = pd.Timestamp("2024-07-04")
    other = daily.drop(holiday) * 2

    historical_data, _ = data_model.build_historical_data(
        {"Daily": daily, "Monthly": monthly, "Other": other}
    )
    changes = historical_data["daily"]["Return"]

    # Every monthly observation after the first has its change, at its own dates.
    assert changes["Monthly"].notna().sum() == 35  # noqa: PLR2004
    assert changes["Monthly"].dropna().iloc[0] == pytest.approx(1 / 100)
    # Across the holiday the change covers the gap instead of being missing.
    after = pd.Period("2024-07-05", "D")
    before = pd.Period("2024-07-03", "D")
    assert changes.loc[after, "Other"] == pytest.approx(
        daily[after.to_timestamp()] / daily[before.to_timestamp()] - 1
    )
    assert np.isnan(changes.loc[pd.Period(holiday, "D"), "Other"])
    differences = data_model.get_changes(
        historical_data["daily"]["Level"], "difference"
    )
    assert differences["Monthly"].dropna().iloc[0] == pytest.approx(1.0)


def test_percentage_changes_divide_by_the_absolute_previous_value():
    # A negative series that rises has a positive change, as in the package's growth rates.
    levels = pd.DataFrame({"A": [-10.0, -5.0, 5.0, 0.0, 2.0]})
    changes = data_model.get_changes(levels, "percentage")["A"]

    assert changes.iloc[1] == pytest.approx(0.5)
    assert changes.iloc[2] == pytest.approx(2.0)
    assert changes.iloc[3] == pytest.approx(-1.0)
    # A change from zero has no relative size.
    assert np.isnan(changes.iloc[4])
