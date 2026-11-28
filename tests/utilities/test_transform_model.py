"""Transform Model Tests"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.filters.hp_filter import hpfilter

from financetoolkit.utilities import statistics_model, transform_model
from financetoolkit.utilities.error_model import InvalidArgumentError

# ruff: noqa: PLR2004


def monthly(values, columns=("United States",)):
    index = pd.period_range("2020-01", periods=len(values), freq="M")
    return pd.DataFrame(
        np.array(values, dtype=float).reshape(len(values), -1),
        index=index,
        columns=list(columns),
    )


def test_the_rolling_steps_match_their_definitions():
    rng = np.random.default_rng(1)
    data = monthly(rng.normal(100, 5, size=(40, 2)), ["A", "B"])
    window = data.rolling(6)

    pd.testing.assert_frame_equal(
        transform_model.apply_transform(data, "sma:6"), window.mean()
    )
    pd.testing.assert_frame_equal(
        transform_model.apply_transform(data, "std:6"), window.std(ddof=1)
    )
    pd.testing.assert_frame_equal(
        transform_model.apply_transform(data, "zscore:6"),
        (data - window.mean()) / window.std(ddof=1),
    )

    last = transform_model.apply_transform(data, "percentile:6").iloc[-1, 0]
    assert last == pytest.approx((data["A"].iloc[-6:] <= data["A"].iloc[-1]).mean())


def test_the_ema_is_seeded_with_the_simple_average():
    data = monthly([1, 2, 3, 4, 5, 6])
    ema = transform_model.apply_transform(data, "ema:3")["United States"]

    assert ema.iloc[:2].isna().all()
    assert ema.iloc[2] == pytest.approx(2.0)
    assert ema.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)


def test_the_changes_follow_the_package_conventions():
    data = monthly([-100, -50, 0, 25, 50, 60, 70, 80, 90, 100, 110, 120, 132])

    change = transform_model.apply_transform(data, "pct_change")["United States"]
    # Growth from a negative base is (x - previous) / |previous|; a zero base has none.
    assert change.iloc[1] == pytest.approx(0.5)
    assert np.isnan(change.iloc[3])

    yoy = transform_model.apply_transform(data, "yoy")["United States"]
    assert yoy.iloc[12] == pytest.approx((132 - -100) / 100)

    log_diff = transform_model.apply_transform(data, "log_diff")["United States"]
    assert np.isnan(log_diff.iloc[1])
    assert log_diff.iloc[5] == pytest.approx(np.log(60 / 50))


def test_steps_run_in_the_order_given():
    data = monthly(
        [100, 101, 102.01, 103.0301, 104.060401, 105.10100501, 106.1520150601]
    )

    quarterly_rate = transform_model.apply_transform(data, "pct_change:3,annualize:3")[
        "United States"
    ]

    assert quarterly_rate.iloc[-1] == pytest.approx(1.01**12 - 1)

    smoothed_change = transform_model.apply_transform(data, ["sma:2", "diff"])
    assert smoothed_change.iloc[-1, 0] == pytest.approx(
        (data.iloc[-1, 0] + data.iloc[-2, 0]) / 2
        - (data.iloc[-2, 0] + data.iloc[-3, 0]) / 2
    )


def test_the_hodrick_prescott_filter_matches_statsmodels():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.normal(size=(80, 2)).cumsum(axis=0),
        index=pd.period_range("2000Q1", periods=80, freq="Q"),
        columns=["A", "B"],
    )
    cycle, trend = hpfilter(data["A"], 1600)

    np.testing.assert_allclose(
        transform_model.apply_transform(data, "hp_trend")["A"], trend, atol=1e-10
    )
    np.testing.assert_allclose(
        transform_model.apply_transform(data, "hp_cycle:1600")["A"], cycle, atol=1e-10
    )

    monthly_cycle, _ = hpfilter(data["B"], 129600)
    as_monthly = data.set_axis(pd.period_range("2000-01", periods=80, freq="M"))
    np.testing.assert_allclose(
        transform_model.apply_transform(as_monthly, "hp_cycle")["B"],
        monthly_cycle,
        atol=1e-8,
    )


def test_each_series_keeps_its_own_history():
    data = monthly([[1, np.nan], [2, np.nan], [3, 10], [4, 20], [5, 30]], ["A", "B"])
    smoothed = transform_model.apply_transform(data, "sma:2")

    assert smoothed["A"].iloc[1] == 1.5
    assert np.isnan(smoothed["B"].iloc[2])
    assert smoothed["B"].iloc[3] == 15


def test_periods_as_columns_are_transformed_per_row():
    ratios = pd.DataFrame(
        [[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]],
        index=["AAPL", "MSFT"],
        columns=pd.period_range("2022", periods=3, freq="Y"),
    )

    changes = transform_model.apply_transform(ratios, "pct_change", axis="columns")

    assert changes.loc["MSFT"].tolist()[1:] == [1.0, 1.0]
    assert changes.shape == ratios.shape


def test_a_series_stays_a_series():
    series = monthly([1, 2, 3])["United States"]

    result = transform_model.apply_transform(series, "cumsum")

    assert isinstance(result, pd.Series)
    assert result.tolist() == [1.0, 3.0, 6.0]


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        ("smooth:3", "Unknown transform step 'smooth'"),
        ("sma", "needs a window"),
        ("log:2", "takes no window"),
        ("sma:2.5", "positive whole number"),
        ("ema:x", "must be a number"),
    ],
)
def test_a_wrong_step_names_what_to_pass(transform, message):
    with pytest.raises(InvalidArgumentError, match=message):
        transform_model.apply_transform(monthly([1, 2, 3]), transform)


def test_a_yearly_step_on_daily_data_asks_for_a_window():
    daily = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]}, index=pd.period_range("2024-01-01", periods=3, freq="D")
    )

    with pytest.raises(InvalidArgumentError, match="pct_change:252"):
        transform_model.apply_transform(daily, "yoy")


def test_finalize_dataset_uses_the_history_before_the_start_date():
    data = monthly(list(range(1, 25)))

    result = statistics_model.finalize_dataset(
        data,
        start_date="2021-01",
        end_date=None,
        default_rounding=None,
        axis="rows",
        row_slice=True,
        transform="sma:12",
    )

    assert result.index[0] == pd.Period("2021-01", freq="M")
    assert result.iloc[0, 0] == pytest.approx(np.mean(range(2, 14)))


def test_a_gap_inside_a_series_is_named_for_rolling_steps(caplog):
    data = monthly([1, 2, np.nan, 4, 5])

    smoothed = transform_model.apply_transform(data, "sma:2")

    assert smoothed["United States"].iloc[2:4].isna().all()
    assert "United States has no value for 2020-03" in caplog.text


def test_a_result_transforms_along_its_dates_wherever_they_are():
    periods = pd.period_range("2024-01", periods=4, freq="M")
    by_ticker = pd.DataFrame(
        {"Value": [1.0, 10, 2, 20, 3, 30, 4, 40], "Name": list("abababab")},
        index=pd.MultiIndex.from_product([periods, ["A", "B"]]),
    )
    transformed = transform_model.transform_time_series(by_ticker, "diff")
    assert transformed["Value"].tolist()[2:] == [1, 10, 1, 10, 1, 10]
    assert transformed["Name"].equals(by_ticker["Name"])

    ticker_first = by_ticker.swaplevel().sort_index()
    swapped = transform_model.transform_time_series(ticker_first, "diff")
    assert swapped.loc["B", "Value"].tolist()[1:] == [10, 10, 10]

    statement = pd.DataFrame(
        [[1.0, 2, 4], [3, 6, 9]],
        index=["Revenue", "Cost"],
        columns=["2022", "2023", "2024"],
    )
    growth = transform_model.transform_time_series(statement, "pct_change")
    assert growth.loc["Revenue", "2024"] == 1
    assert growth.loc["Cost", "2024"] == 0.5


def test_a_result_without_dates_refuses_a_transform():
    with pytest.raises(InvalidArgumentError, match="runs along dates"):
        transform_model.transform_time_series(
            pd.DataFrame({"Value": [1.0, 2.0]}, index=["AAPL", "MSFT"]), "diff"
        )
    with pytest.raises(InvalidArgumentError, match="runs along dates"):
        statistics_model.finalize_dataset(
            pd.Series([0.1, 0.2], index=["AAPL", "MSFT"]),
            start_date=None,
            end_date=None,
            default_rounding=None,
            axis="rows",
            transform="sma:2",
        )


def test_the_transform_decorator_rounds_like_the_method():
    class Owner:
        _rounding = 2

        @transform_model.transform_result
        def get_values(self, rounding=None, transform=None):
            return pd.DataFrame(
                {"A": [1.0, 3.0, 4.0]},
                index=pd.period_range("2024", periods=3, freq="Y"),
            )

    assert Owner().get_values(transform="pct_change")["A"].tolist()[1:] == [2.0, 0.33]
    assert Owner().get_values(rounding=4, transform="pct_change")["A"].iloc[2] == 0.3333
    assert Owner().get_values()["A"].tolist() == [1.0, 3.0, 4.0]
