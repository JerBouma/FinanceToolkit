"""Models Helpers Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models import helpers
from financetoolkit.models.distress import chs_model


def test_rates_per_period_broadcast_given_rates_and_fill_the_rest_by_period():
    periods = pd.period_range("2021", "2023", freq="Y")
    default = pd.DataFrame(
        [[0.05, 0.06, 0.07], [0.08, 0.09, 0.10]], index=["A", "B"], columns=periods
    )

    rates = helpers.rates_per_period(
        {"A": 0.04}, ["A", "B"], periods, "rate", lambda: default
    )

    assert rates.loc["A"].tolist() == [0.04, 0.04, 0.04]
    assert rates.loc["B"].tolist() == [0.08, 0.09, 0.10]
    assert (
        helpers.rates_per_period(0.03, ["A"], periods, "rate").loc["A"].tolist()
        == [0.03] * 3
    )

    with pytest.raises(ValueError, match="No rate given"):
        helpers.rates_per_period(None, ["A"], periods, "rate")


def test_exceeds_growth_per_period_leaves_out_only_those_periods():
    rates = pd.DataFrame([[0.05, 0.02, float("nan")]], index=["A"])
    growth = pd.DataFrame([[0.02, 0.03, 0.02]], index=["A"])

    assert helpers.exceeds_growth_per_period(rates, growth).loc["A"].tolist() == [
        True,
        False,
        True,
    ]


def test_trailing_within_periods_covers_the_trailing_periods():
    days = pd.period_range("2024-01-01", "2024-12-31", freq="D")
    within = pd.Series(
        range(len(days)),
        index=pd.MultiIndex.from_arrays([days.asfreq("Q"), days]),
        dtype=float,
    )

    trailing = helpers.trailing_within_periods(within, 2)

    # The first quarter has no earlier quarter; the others hold two quarters of days.
    assert len(trailing.loc[pd.Period("2024Q1")]) == len(
        within.loc[pd.Period("2024Q1")]
    )
    assert len(trailing.loc[pd.Period("2024Q3")]) == len(
        within.loc[pd.Period("2024Q2")]
    ) + len(within.loc[pd.Period("2024Q3")])


def test_a_narrow_spread_between_rate_and_growth_is_valued_with_a_warning(caplog):
    rates = pd.DataFrame([[0.08, 0.045, 0.03]], index=["A"])
    growth = pd.DataFrame([[0.03, 0.03, 0.03]], index=["A"])

    with caplog.at_level("WARNING"):
        exceeds = helpers.exceeds_growth_per_period(rates, growth)

    # 1.5 percentage points is valued but flagged, 5 points is not flagged.
    assert exceeds.loc["A"].tolist() == [True, True, False]
    assert "less than 2 percentage points" in caplog.text
    assert "1 period(s)" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        assert helpers.exceeds_growth("B", 0.04, 0.025)
    assert "less than 2 percentage points" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        assert helpers.exceeds_growth("C", 0.08, 0.025)
    assert caplog.text == ""


def test_a_spread_of_exactly_two_points_is_not_flagged(caplog):
    # 0.06 - 0.04 is 0.019999999999999997 in floating point.
    with caplog.at_level("WARNING"):
        assert helpers.exceeds_growth("A", 0.06, 0.04)
        helpers.exceeds_growth_per_period(
            pd.DataFrame([[0.06]], index=["A"]), pd.DataFrame([[0.04]], index=["A"])
        )
    assert caplog.text == ""


def test_values_on_or_before():
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    daily = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0], "BBB": [5.0, np.nan, 7.0]}, index=days
    )
    dates = pd.DataFrame(
        {
            "first": pd.to_datetime(["2024-01-04", "2024-01-04"]),
            "second": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        },
        index=["AAA", "BBB"],
    )
    values = helpers.values_on_or_before(daily, dates)
    # The last value on or before the date from each ticker's own history; none before the
    # first day, and none more than ten days old.
    assert values.loc["AAA", "first"] == 2.0
    assert values.loc["BBB", "first"] == 5.0
    assert np.isnan(values.loc["AAA", "second"])
    assert np.isnan(values.loc["BBB", "second"])


@pytest.mark.parametrize(
    "order",
    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
)
@pytest.mark.parametrize("tolerance", [0, 10])
def test_values_on_or_before_handles_unsorted_history(order, tolerance):
    daily = pd.DataFrame(
        {"AAA": [10.0, 20.0, 30.0], "BBB": [1.0, np.nan, 3.0], "EMPTY": np.nan},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-05"]),
    ).iloc[list(order)]
    dates = pd.DataFrame(
        {
            "latest": pd.Timestamp("2024-01-05"),
            "between": pd.Timestamp("2024-01-04"),
            "first": pd.Timestamp("2024-01-02"),
            "before": pd.Timestamp("2024-01-01"),
            "stale": pd.Timestamp("2024-01-16"),
            "boundary": pd.Timestamp("2024-01-15"),
            "missing": pd.NaT,
        },
        index=pd.Index(["BBB", "AAA", "MISSING", "EMPTY"], name="Ticker"),
    ).rename_axis("Snapshot", axis=1)
    original_daily, original_dates = daily.copy(deep=True), dates.copy(deep=True)

    result = helpers.values_on_or_before(daily, dates, tolerance=tolerance)

    expected = pd.DataFrame(np.nan, index=dates.index, columns=dates.columns)
    expected.loc[["BBB", "AAA"], "latest"] = [3.0, 30.0]
    expected.loc[["BBB", "AAA"], "first"] = [1.0, 10.0]
    if tolerance == 10:
        expected.loc[["BBB", "AAA"], "between"] = [1.0, 20.0]
        expected.loc[["BBB", "AAA"], "boundary"] = [3.0, 30.0]
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(daily, original_daily)
    pd.testing.assert_frame_equal(dates, original_dates)


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (2, 0, 1, 3)])
def test_values_on_or_before_keeps_last_observation_at_duplicate_dates(order):
    daily = pd.DataFrame(
        {"AAA": [10.0, 20.0, 30.0, 40.0]},
        index=pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-05"]
        ),
    ).iloc[list(order)]
    dates = pd.DataFrame({"asof": [pd.Timestamp("2024-01-03")]}, index=["AAA"])

    result = helpers.values_on_or_before(daily, dates)

    # Equal dates keep their original order; right-sided lookup chooses the last.
    expected_value = 30.0 if order == (0, 1, 2, 3) else 20.0
    assert result.loc["AAA", "asof"] == expected_value


def test_monthly_credit_returns_handle_independently_unsorted_histories():
    days = pd.DatetimeIndex(["2023-12-31", "2024-01-30", "2024-01-31", "2024-02-29"])
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 120.0, 140.0]}, index=days)
    index_levels = pd.Series([50.0, 53.0, 55.0, 60.0], index=days)
    dates = pd.DataFrame({"current": [pd.Timestamp("2024-01-31")]}, index=["AAA"])

    result = chs_model.get_monthly_log_excess_returns(
        prices.iloc[[2, 0, 3, 1]], index_levels.iloc[[3, 1, 0, 2]], dates, months=1
    )

    # January ends at 120 and 55, not at the last physical rows or February values.
    assert len(result) == 1
    expected = pd.DataFrame({"current": [np.log(1.2) - np.log(1.1)]}, index=["AAA"])
    pd.testing.assert_frame_equal(result[0], expected)
