"""Statistics Model Tests"""

# ruff: noqa: PLR2004

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from financetoolkit.utilities import statistics_model


def test_calculate_growth_basic():
    """Test basic growth calculation with default parameters."""
    data = pd.DataFrame(
        {"2020": [100, 200], "2021": [110, 220], "2022": [121, 242]},
        index=["Revenue", "Expenses"],
    )

    result = statistics_model.calculate_growth(data)

    expected_values = np.array([[np.nan, 0.1, 0.1], [np.nan, 0.1, 0.1]])

    # Check shape
    assert result.shape == data.shape

    # Check that first column is NaN (no previous period)
    assert result.iloc[:, 0].isna().all()

    # Check growth values
    np.testing.assert_array_almost_equal(
        result.iloc[:, 1:].values, expected_values[:, 1:]
    )


def test_calculate_growth_with_lag():
    """Test growth calculation with custom lag."""
    data = pd.DataFrame(
        {"2020": [100], "2021": [110], "2022": [121], "2023": [133]}, index=["Revenue"]
    )

    result = statistics_model.calculate_growth(data, lag=2)

    # First two values should be NaN
    assert result.iloc[0, 0:2].isna().all()

    # Third value should be (121-100)/100 = 0.21
    assert abs(result.iloc[0, 2] - 0.21) < 0.01


def test_calculate_growth_with_list_lag():
    """Test growth calculation with list of lags."""
    data = pd.DataFrame(
        {"2020": [100], "2021": [110], "2022": [121], "2023": [133]}, index=["Revenue"]
    )

    result = statistics_model.calculate_growth(data, lag=[1, 2])

    # Check that result has MultiIndex
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == [None, None]

    # Check lag structure
    assert ("Revenue", "Lag 1") in result.index
    assert ("Revenue", "Lag 2") in result.index


def test_calculate_growth_axis_parameter():
    """Test growth calculation with axis parameter."""
    data = pd.DataFrame(
        {"2020": [100, 200], "2021": [110, 220], "2022": [121, 242]},
        index=["Revenue", "Expenses"],
    )

    result = statistics_model.calculate_growth(data, axis="index")

    # Check shape
    assert result.shape == data.shape

    # When axis="index", growth is calculated along rows
    assert result.columns.equals(data.columns)


def test_calculate_growth_with_rounding():
    """Test growth calculation with custom rounding."""
    data = pd.DataFrame({"2020": [100], "2021": [110.12345]}, index=["Revenue"])

    result = statistics_model.calculate_growth(data, rounding=2)

    # Check rounding
    assert result.iloc[0, 1] == 0.10


def test_calculate_growth_with_missing_data():
    """Test growth calculation with missing data."""
    data = pd.DataFrame(
        {"2020": [100, np.nan], "2021": [110, 220], "2022": [121, 242]},
        index=["Revenue", "Expenses"],
    )

    result = statistics_model.calculate_growth(data)

    # Should handle NaN values properly
    assert not result.isna().all().all()


def test_calculate_growth_warnings_suppression():
    """Test that FutureWarning is suppressed."""
    data = pd.DataFrame({"2020": [100], "2021": [110]}, index=["Revenue"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        statistics_model.calculate_growth(data)

        # Check that no FutureWarnings were raised
        future_warnings = [
            warning for warning in w if issubclass(warning.category, FutureWarning)
        ]
        assert len(future_warnings) == 0


def test_calculate_standardization_series():
    """Test Z-Score standardization on a plain Series."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = statistics_model.calculate_standardization(data)

    assert abs(result.mean()) < 1e-3
    assert abs(result.std() - 1) < 1e-3


def test_calculate_standardization_axis_columns():
    """Test standardization per row (axis='columns'), e.g. ratios indexed by ticker."""
    data = pd.DataFrame(
        {"2020": [1.0, 10.0], "2021": [2.0, 20.0], "2022": [3.0, 30.0]},
        index=["AAPL", "MSFT"],
    )

    result = statistics_model.calculate_standardization(data, axis="columns")

    # Each row should be standardized independently (mean ~0 across its own history)
    row_means = result.mean(axis=1)
    assert abs(row_means["AAPL"]) < 1e-6
    assert abs(row_means["MSFT"]) < 1e-6


def test_calculate_standardization_axis_rows():
    """Test standardization per column (axis='rows'), e.g. date-indexed metrics."""
    data = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0], "MSFT": [10.0, 20.0, 30.0]},
        index=["2020", "2021", "2022"],
    )

    result = statistics_model.calculate_standardization(data, axis="rows")

    # Each column should be standardized independently (mean ~0 across its own history)
    col_means = result.mean(axis=0)
    assert abs(col_means["AAPL"]) < 1e-6
    assert abs(col_means["MSFT"]) < 1e-6


def test_calculate_standardization_rounding():
    """Test standardization respects the rounding parameter."""
    data = pd.DataFrame({"2020": [1.0, 2.0], "2021": [3.0, 4.0]}, index=["A", "B"])

    result = statistics_model.calculate_standardization(data, rounding=2)

    for value in result.to_numpy().flatten():
        assert round(value, 2) == value


def test_finalize_dataset_growth_and_round():
    """Test finalize_dataset applies growth and rounding correctly."""
    data = pd.DataFrame(
        {"2020": [100, 200], "2021": [110, 220], "2022": [121, 242]},
        index=["AAPL", "MSFT"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        growth=True,
        lag=1,
    )

    assert result.iloc[:, 0].isna().all()
    assert abs(result.iloc[0, 1] - 0.1) < 0.01


def test_finalize_dataset_standardize_only():
    """Test finalize_dataset applies standardization without growth."""
    data = pd.DataFrame(
        {"2020": [1.0, 10.0], "2021": [2.0, 20.0], "2022": [3.0, 30.0]},
        index=["AAPL", "MSFT"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        standardize=True,
        axis="columns",
    )

    row_means = result.mean(axis=1)
    assert abs(row_means["AAPL"]) < 1e-6
    assert abs(row_means["MSFT"]) < 1e-6


def test_finalize_dataset_growth_then_standardize():
    """Test finalize_dataset standardizes the growth values when both are set."""
    data = pd.DataFrame(
        {"2020": [100, 200], "2021": [110, 240], "2022": [121, 220]},
        index=["AAPL", "MSFT"],
    )

    growth_only = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        growth=True,
    )
    combined = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        growth=True,
        standardize=True,
    )

    # Combined result should be the standardization of the growth-only result
    expected = statistics_model.calculate_standardization(growth_only, rounding=4)
    pd.testing.assert_frame_equal(combined, expected)


def test_finalize_dataset_default_rounding_fallback():
    """Test finalize_dataset falls back to default_rounding when rounding is None."""
    data = pd.DataFrame({"2020": [1.23456], "2021": [2.34567]}, index=["AAPL"])

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=2,
        rounding=None,
    )

    assert result.iloc[0, 0] == 1.23
    assert result.iloc[0, 1] == 2.35


def test_finalize_dataset_column_slice():
    """Test finalize_dataset slices by column (default) for ratios-shaped data."""
    data = pd.DataFrame(
        {"2020": [1.0], "2021": [2.0], "2022": [3.0]},
        index=["AAPL"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date="2021",
        end_date="2022",
        default_rounding=4,
    )

    assert list(result.columns) == ["2021", "2022"]


def test_finalize_dataset_row_slice():
    """Test finalize_dataset slices by row when row_slice=True."""
    data = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]},
        index=["2020", "2021", "2022"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date="2021",
        end_date="2022",
        default_rounding=4,
        row_slice=True,
    )

    assert list(result.index) == ["2021", "2022"]


def test_finalize_dataset_apply_slice_false():
    """Test finalize_dataset skips slicing entirely when apply_slice=False."""
    data = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]},
        index=["2020", "2021", "2022"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date="2021",
        end_date="2021",
        default_rounding=4,
        row_slice=True,
        apply_slice=False,
    )

    # All rows should still be present since slicing was skipped
    assert list(result.index) == ["2020", "2021", "2022"]


def test_finalize_dataset_rolling():
    """Test finalize_dataset applies rolling-window smoothing before growth."""
    # rolling/trailing operate down the rows, matching the date-indexed shape.
    data = pd.DataFrame(
        {"AAPL": [1.0, 3.0, 5.0]},
        index=["2020", "2021", "2022"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        rolling=2,
    )

    # First value has no prior period to average with -> NaN
    assert pd.isna(result.iloc[0, 0])
    assert result.iloc[1, 0] == 2.0
    assert result.iloc[2, 0] == 4.0


def test_finalize_dataset_trailing():
    """Test finalize_dataset applies trailing-window summation before growth."""
    data = pd.DataFrame(
        {"AAPL": [1.0, 2.0, 3.0]},
        index=["2020", "2021", "2022"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        trailing=2,
    )

    assert pd.isna(result.iloc[0, 0])
    assert result.iloc[1, 0] == 3.0
    assert result.iloc[2, 0] == 5.0


def test_finalize_dataset_dropna():
    """Test finalize_dataset drops all-NaN rows when dropna=True."""
    data = pd.DataFrame(
        {"AAPL": [np.nan, 2.0], "MSFT": [np.nan, 4.0]},
        index=["2020", "2021"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        row_slice=True,
        dropna=True,
    )

    assert list(result.index) == ["2021"]


def test_finalize_dataset_countries_filter():
    """Test finalize_dataset filters to the requested countries."""
    data = pd.DataFrame(
        {"United States": [1.0, 2.0], "Netherlands": [3.0, 4.0], "Japan": [5.0, 6.0]},
        index=["2020", "2021"],
    )

    result = statistics_model.finalize_dataset(
        dataset=data,
        start_date=None,
        end_date=None,
        default_rounding=4,
        row_slice=True,
        countries=["United States", "Netherlands"],
    )

    assert set(result.columns) == {"United States", "Netherlands"}


def test_finalize_dataset_countries_missing_logs_warning():
    """Test finalize_dataset logs a warning for countries not present in the data."""
    data = pd.DataFrame({"United States": [1.0, 2.0]}, index=["2020", "2021"])

    with patch("financetoolkit.utilities.statistics_model.logger") as mock_logger:
        result = statistics_model.finalize_dataset(
            dataset=data,
            start_date=None,
            end_date=None,
            default_rounding=4,
            row_slice=True,
            countries=["United States", "Atlantis"],
            indicator_name="Test Indicator",
        )

        mock_logger.warning.assert_called_once()

    assert "United States" in result.columns
    assert "Atlantis" not in result.columns


def test_calculate_growth_from_zero_is_missing_not_infinite():
    """Growth from a value of 0 is undefined, for a single lag and a list of lags."""
    dataset = pd.DataFrame(
        [[0.0, 5.0, 10.0]], index=["Dividends"], columns=[2021, 2022, 2023]
    )

    single = statistics_model.calculate_growth(dataset, lag=1, rounding=None)
    multiple = statistics_model.calculate_growth(dataset, lag=[1, 2], rounding=None)

    assert np.isnan(single.loc["Dividends", 2022])
    assert single.loc["Dividends", 2023] == 1.0
    assert np.isnan(multiple.loc[("Dividends", "Lag 2"), 2023])
    assert list(multiple.index) == [("Dividends", "Lag 1"), ("Dividends", "Lag 2")]


def test_country_short_forms_and_casing_resolve_to_the_source_names():
    from financetoolkit.utilities.statistics_model import normalize_country_name

    known = ["United States", "United Kingdom", "Euro Area", "Germany"]

    assert normalize_country_name("USA", known) == "United States"
    assert normalize_country_name("uk", known) == "United Kingdom"
    assert normalize_country_name("eurozone", known) == "Euro Area"
    assert normalize_country_name("germany", known) == "Germany"
    assert normalize_country_name("Atlantis", known) == "Atlantis"


def test_a_country_without_values_in_the_window_is_explained(caplog):
    import pandas as pd

    from financetoolkit.utilities.statistics_model import finalize_dataset

    crises = pd.DataFrame(
        {"Germany": [1.0, None, None], "Euro Area": [None, 0.0, 0.0]},
        index=pd.period_range("2017", periods=3, freq="Y"),
    )

    result = finalize_dataset(
        dataset=crises,
        start_date="2018",
        end_date="2019",
        default_rounding=4,
        axis="rows",
        row_slice=True,
        countries=["Germany"],
        indicator_name="Sovereign Debt Crisis",
    )

    assert result.isna().all().all()
    assert "The last one is from 2017" in caplog.text
    assert "use 'Euro Area'" in caplog.text


def test_calculate_growth_divides_by_the_absolute_previous_value():
    """A negative base keeps the sign of the change: a narrowing loss is growth."""
    data = pd.DataFrame(
        [[-100.0, -50.0, 20.0, -10.0, 0.0, 5.0]],
        index=["X"],
        columns=pd.period_range("2020", periods=6, freq="Y"),
    )

    result = statistics_model.calculate_growth(data, lag=1, rounding=4)

    # -100 to -50 is +0.5, -50 to 20 is +1.4, 20 to -10 is -1.5, -10 to 0 is +1 and
    # growth from 0 is undefined.
    assert result.loc["X"].tolist()[1:5] == [0.5, 1.4, -1.5, 1.0]
    assert pd.isna(result.loc["X"].iloc[5])

    lagged = statistics_model.calculate_growth(data, lag=[1, 2], rounding=4)
    assert lagged.loc[("X", "Lag 1")].tolist()[1:5] == [0.5, 1.4, -1.5, 1.0]
    # -100 to 20 over two periods is +1.2.
    assert lagged.loc[("X", "Lag 2")].iloc[2] == 1.2


def test_calculate_growth_equals_the_percentage_change_for_positive_values():
    data = pd.Series([100.0, 110.0, 99.0, 120.0])

    pd.testing.assert_series_equal(
        statistics_model.calculate_growth(data, rounding=None), data.pct_change()
    )


def test_apply_rounding_prints_no_negative_zero():
    # -0.00001 rounds to -0.0, which a table prints as "-0".
    frame = pd.DataFrame(
        {"Statistic": [-0.00001, -1.23456], "Significant": [True, False]},
        index=["A", "B"],
    )

    rounded = statistics_model.apply_rounding(frame, 4)

    assert str(rounded.loc["A", "Statistic"]) == "0.0"
    assert rounded.loc["B", "Statistic"] == -1.2346
    assert rounded["Significant"].tolist() == [True, False]
    assert (
        str(statistics_model.apply_rounding(pd.Series([-0.00001]), 4).iloc[0]) == "0.0"
    )


def test_growth_is_missing_for_a_missing_period_and_the_period_after_it():
    periods = pd.period_range("2019", periods=5, freq="Y")
    data = pd.DataFrame([[100, 110, np.nan, 130, 143.0]], index=["A"], columns=periods)

    growth = statistics_model.calculate_growth(data)
    lagged = statistics_model.calculate_growth(data, lag=[1, 2])

    assert np.isnan(growth.iloc[0, 2]) and np.isnan(growth.iloc[0, 3])
    assert growth.iloc[0, 1] == 0.1
    assert growth.iloc[0, 4] == 0.1
    # Over two periods, 2022 compares with 2020, both of which are known.
    assert lagged.loc[("A", "Lag 2")].iloc[3] == round(130 / 110 - 1, 4)


def test_standardization_describes_the_window_shown():
    periods = pd.period_range("2015", periods=10, freq="Y")
    data = pd.DataFrame({"US": np.arange(10, 20, dtype=float)}, index=periods)

    standardized = statistics_model.finalize_dataset(
        data, "2015", "2016", 4, standardize=True, axis="rows", row_slice=True
    )

    assert standardized["US"].tolist() == [-0.7071, 0.7071]


def test_rounding_leaves_text_and_dates_and_rounds_numbers_among_them():
    data = pd.DataFrame(
        {
            "number": [1.23456, -0.00001],
            "mixed": pd.Series([1.23456, "n/a"], dtype=object),
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "text": ["x", "y"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rounded = statistics_model.apply_rounding(data, 2)
        mixed = statistics_model.apply_rounding(
            pd.Series([1.23456, "x"], dtype=object), 2
        )

    assert rounded["number"].tolist() == [1.23, 0.0]
    assert rounded["mixed"].tolist() == [1.23, "n/a"]
    assert rounded["date"].equals(data["date"])
    assert rounded["text"].tolist() == ["x", "y"]
    assert mixed.tolist() == [1.23, "x"]


def test_rounding_keeps_small_statistics_and_rounds_integers():
    statistics = pd.DataFrame(
        {"value": pd.Series([0.000012345, 0.123456, -0.000004, 0.0], dtype=object)}
    )
    integers = pd.DataFrame({"count": [1234, 5678], "value": [1234.5, 5678.9]})

    rounded = statistics_model.apply_rounding(statistics, 4)

    # A p-value of 0.000012345 keeps its first four significant digits instead of showing as 0.
    assert rounded["value"].tolist() == [0.00001234, 0.1235, -0.000004, 0.0]
    assert statistics_model.apply_rounding(integers, -2)["count"].tolist() == [
        1200,
        5700,
    ]
    assert statistics_model.apply_rounding(integers["count"], -2).tolist() == [
        1200,
        5700,
    ]


@pytest.mark.parametrize("lag", [0, -1, 1.5, True, [], [1, 0]])
def test_calculate_growth_refuses_lags_that_are_not_positive(lag):
    data = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["2020", "2021", "2022"])

    with pytest.raises(ValueError):
        statistics_model.calculate_growth(data, lag=lag)


def test_official_country_names_resolve_to_the_global_macro_database_names():
    from financetoolkit.utilities.statistics_model import normalize_country_name

    known = ["Macedonia", "Macau", "Cape Verde", "Sao Tome and Principe"]

    assert normalize_country_name("North Macedonia", known) == "Macedonia"
    assert normalize_country_name("Macao", known) == "Macau"
    assert normalize_country_name("Cabo Verde", known) == "Cape Verde"
    assert (
        normalize_country_name("São Tomé and Príncipe", known)
        == "Sao Tome and Principe"
    )


def test_compound_growth_is_the_yearly_rate_over_the_lag():
    dataset = pd.DataFrame(
        [[100.0, 110.0, 121.0, 133.1], [100.0, -5.0, 50.0, 200.0]],
        index=["A", "B"],
        columns=pd.period_range("2020", periods=4, freq="Y"),
    )

    growth = statistics_model.calculate_growth(
        dataset, lag=2, compound=True, rounding=None
    )

    # 10% a year, and (50 / 100) ^ (1 / 2) - 1 for B: a loss in between does not matter.
    assert growth.loc["A", pd.Period("2023", "Y")] == pytest.approx(0.1)
    assert growth.loc["B", pd.Period("2022", "Y")] == pytest.approx(0.5**0.5 - 1)
    # From a loss there is no compound rate (the total change convention would give 41).
    assert np.isnan(growth.loc["B", pd.Period("2023", "Y")])


def test_compound_growth_of_quarterly_data_counts_the_lag_in_quarters():
    dataset = pd.DataFrame(
        [np.arange(1.0, 10.0)],
        index=["A"],
        columns=pd.period_range("2020Q1", periods=9, freq="Q"),
    )

    growth = statistics_model.calculate_growth(
        dataset, lag=[4, 8], compound=True, rounding=None
    )

    # Eight quarters are two years: (9 / 1) ^ (1 / 2) - 1.
    assert growth.loc[("A", "Lag 8"), pd.Period("2022Q1", "Q")] == pytest.approx(2.0)
    assert growth.loc[("A", "Lag 4"), pd.Period("2022Q1", "Q")] == pytest.approx(
        9 / 5 - 1
    )


def test_compound_growth_over_monthly_rows_through_finalize_dataset():
    monthly = pd.DataFrame(
        {"A": [100.0] + [None] * 11 + [121.0]},
        index=pd.period_range("2020-01", periods=13, freq="M"),
    )

    growth = statistics_model.finalize_dataset(
        monthly,
        start_date=None,
        end_date=None,
        default_rounding=None,
        growth=True,
        lag=12,
        compound=True,
        axis="rows",
        row_slice=True,
    )

    assert growth.loc[pd.Period("2021-01", "M"), "A"] == pytest.approx(0.21)


def test_compound_growth_needs_periods_with_a_yearly_count():
    daily = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]}, index=pd.period_range("2020-01-01", periods=3, freq="D")
    )

    with pytest.raises(ValueError, match="yearly, quarterly, monthly or weekly"):
        statistics_model.calculate_growth(daily, lag=1, compound=True, axis="rows")


def test_by_own_trading_days_counts_each_columns_own_rows():
    """A row-counting measure counts the rows each column has a value on."""
    from financetoolkit.utilities.statistics_model import by_own_trading_days

    @by_own_trading_days
    def rows(returns):
        return returns.apply(len)

    returns = pd.DataFrame(
        {"A": [0.01, np.nan, 0.02, 0.03], "B": [0.01, 0.02, 0.03, 0.04]}
    )

    # A did not trade on the second row, which only B did.
    assert rows(returns).to_dict() == {"A": 3, "B": 4}


def test_by_own_trading_days_compounds_returns_between_shared_dates():
    """Against a benchmark, the returns run between the dates both traded."""
    from financetoolkit.utilities.statistics_model import by_own_trading_days

    @by_own_trading_days(compound=True)
    def paired(returns, benchmark_returns):
        return returns.assign(Benchmark=benchmark_returns)

    index = pd.RangeIndex(4)
    returns = pd.DataFrame({"A": [np.nan, 0.10, np.nan, 0.10]}, index=index)
    benchmark = pd.Series([np.nan, 0.05, 0.05, 0.05], index=index)

    result = paired(returns, benchmark)

    # On the last row, the benchmark's two days since the previous shared date.
    assert np.isclose(result.loc[3, "Benchmark"], 1.05**2 - 1)
    assert np.isclose(result.loc[3, "A"], 0.10)
    assert pd.isna(result.loc[2, "A"])


def test_drop_non_finite_replaces_infinities_with_nan():
    assert np.isnan(statistics_model.drop_non_finite(np.inf))
    assert statistics_model.drop_non_finite(1.5) == 1.5
    assert statistics_model.drop_non_finite(
        pd.Series([1.0, np.inf, -np.inf])
    ).isna().tolist() == [False, True, True]


def test_finalize_dataset_reports_a_zero_denominator_as_nan():
    ratio = pd.DataFrame(
        [[1.9, np.inf, 2.6, 1.8]], index=["AAPL"], columns=[2019, 2020, 2021, 2022]
    )

    finalized = statistics_model.finalize_dataset(ratio, None, None, 4)
    standardized = statistics_model.finalize_dataset(
        ratio, None, None, 4, standardize=True
    )

    assert finalized.loc["AAPL"].isna().tolist() == [False, True, False, False]
    # One undefined period does not turn the whole Z-Score row into NaN.
    assert standardized.loc["AAPL"].isna().tolist() == [False, True, False, False]
