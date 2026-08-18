"""Statistics Model Tests"""

# ruff: noqa: PLR2004

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

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
