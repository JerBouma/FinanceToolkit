# ruff: noqa
"""Cache Model Tests"""

import os
import pickle
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from financetoolkit.utilities import cache_model


def test_load_cached_data_pandas_method():
    """Test loading cached data using pandas method."""
    # Create test data
    test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_data.pkl")
        test_data.to_pickle(file_path)

        result = cache_model.load_cached_data(
            cached_data_location=temp_dir, file_name="test_data.pkl", method="pandas"
        )

        pd.testing.assert_frame_equal(result, test_data)


def test_load_cached_data_pickle_method():
    """Test loading cached data using pickle method."""
    test_data = {"key1": "value1", "key2": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_data.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(test_data, f)

        result = cache_model.load_cached_data(
            cached_data_location=temp_dir,
            file_name="test_data.pkl",
            method="pickle",
            return_empty_type={},
        )

        assert result == test_data


def test_load_cached_data_invalid_method():
    """Test loading cached data with invalid method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(
            ValueError, match="The method should be either 'pandas' or 'pickle'"
        ):
            cache_model.load_cached_data(
                cached_data_location=temp_dir,
                file_name="test.pkl",
                method="invalid_method",
            )


def test_load_cached_data_file_not_found():
    """Test loading cached data when file doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = cache_model.load_cached_data(
            cached_data_location=temp_dir, file_name="nonexistent.pkl", method="pandas"
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty


def test_load_cached_data_file_not_found_custom_return_type():
    """Test loading cached data when file doesn't exist with custom return type."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = cache_model.load_cached_data(
            cached_data_location=temp_dir,
            file_name="nonexistent.pkl",
            method="pickle",
            return_empty_type={},
        )

        assert result == {}


def test_save_cached_data_pandas_method():
    """Test saving cached data using pandas method."""
    test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("financetoolkit.utilities.cache_model.logger") as mock_logger:
            cache_model.save_cached_data(
                cached_data=test_data,
                cached_data_location=temp_dir,
                file_name="test_data.pkl",
                method="pandas",
            )

            # Check file was created
            assert os.path.exists(os.path.join(temp_dir, "test_data.pkl"))

            # Check log message
            mock_logger.info.assert_called_once()


def test_save_cached_data_pickle_method():
    """Test saving cached data using pickle method."""
    test_data = {"key1": "value1", "key2": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("financetoolkit.utilities.cache_model.logger") as mock_logger:
            cache_model.save_cached_data(
                cached_data=test_data,
                cached_data_location=temp_dir,
                file_name="test_data.pkl",
                method="pickle",
            )

            # Check file was created
            assert os.path.exists(os.path.join(temp_dir, "test_data.pkl"))

            # Load and verify data
            with open(os.path.join(temp_dir, "test_data.pkl"), "rb") as f:
                loaded_data = pickle.load(f)

            assert loaded_data == test_data
            mock_logger.info.assert_called_once()


def test_save_cached_data_file_already_exists():
    """Test saving cached data when file already exists."""
    test_data = pd.DataFrame({"A": [1, 2, 3]})

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "existing_file.pkl")

        # Create existing file
        test_data.to_pickle(file_path)
        original_mtime = os.path.getmtime(file_path)

        with patch("financetoolkit.utilities.cache_model.logger") as mock_logger:
            cache_model.save_cached_data(
                cached_data=test_data,
                cached_data_location=temp_dir,
                file_name="existing_file.pkl",
                method="pandas",
            )

            # File should not be overwritten
            assert os.path.getmtime(file_path) == original_mtime

            # No log message should be shown
            mock_logger.info.assert_not_called()


def test_save_cached_data_no_message():
    """Test saving cached data without log message."""
    test_data = pd.DataFrame({"A": [1, 2, 3]})

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("financetoolkit.utilities.cache_model.logger") as mock_logger:
            cache_model.save_cached_data(
                cached_data=test_data,
                cached_data_location=temp_dir,
                file_name="test_data.pkl",
                method="pandas",
                include_message=False,
            )

            # Check file was created
            assert os.path.exists(os.path.join(temp_dir, "test_data.pkl"))

            # No log message should be shown
            mock_logger.info.assert_not_called()


def test_save_cached_data_creates_directory():
    """Test that save_cached_data creates the directory if it doesn't exist."""
    test_data = pd.DataFrame({"A": [1, 2, 3]})

    with tempfile.TemporaryDirectory() as temp_dir:
        nested_dir = os.path.join(temp_dir, "nested", "directory")

        cache_model.save_cached_data(
            cached_data=test_data,
            cached_data_location=nested_dir,
            file_name="test_data.pkl",
            method="pandas",
        )

        # Check directory was created
        assert os.path.exists(nested_dir)

        # Check file was created
        assert os.path.exists(os.path.join(nested_dir, "test_data.pkl"))


def test_save_cached_data_error_handling():
    """Test error handling in save_cached_data."""
    test_data = pd.DataFrame({"A": [1, 2, 3]})

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("financetoolkit.utilities.cache_model.logger") as mock_logger:
            # Mock pandas to_pickle to raise an exception
            with patch.object(
                test_data, "to_pickle", side_effect=Exception("Test error")
            ):
                cache_model.save_cached_data(
                    cached_data=test_data,
                    cached_data_location=temp_dir,
                    file_name="test_data.pkl",
                    method="pandas",
                )

                # Check error was logged
                mock_logger.error.assert_called_once()
                assert "An error occurred while saving the data" in str(
                    mock_logger.error.call_args
                )


def test_save_cached_data_empty_dataframe():
    """Test saving an empty DataFrame."""
    test_data = pd.DataFrame()

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_model.save_cached_data(
            cached_data=test_data,
            cached_data_location=temp_dir,
            file_name="empty_data.pkl",
            method="pandas",
        )

        # Check file was created
        assert os.path.exists(os.path.join(temp_dir, "empty_data.pkl"))

        # Load and verify it's empty
        loaded_data = pd.read_pickle(os.path.join(temp_dir, "empty_data.pkl"))
        assert loaded_data.empty


def test_save_cached_data_with_multiindex():
    """Test saving DataFrame with MultiIndex."""
    arrays = [["A", "A", "B", "B"], [1, 2, 1, 2]]
    index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
    test_data = pd.DataFrame({"values": [10, 20, 30, 40]}, index=index)

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_model.save_cached_data(
            cached_data=test_data,
            cached_data_location=temp_dir,
            file_name="multiindex_data.pkl",
            method="pandas",
        )

        # Load and verify structure is preserved
        loaded_data = pd.read_pickle(os.path.join(temp_dir, "multiindex_data.pkl"))
        pd.testing.assert_frame_equal(loaded_data, test_data)


def test_load_cached_data_with_complex_pickle_data():
    """Test loading complex pickle data with nested structures."""
    test_data = {
        "dataframe": pd.DataFrame({"A": [1, 2, 3]}),
        "list": [1, 2, 3],
        "nested": {"inner": {"value": 42}},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "complex_data.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(test_data, f)

        result = cache_model.load_cached_data(
            cached_data_location=temp_dir,
            file_name="complex_data.pkl",
            method="pickle",
            return_empty_type={},
        )

        assert result["list"] == test_data["list"]
        assert result["nested"] == test_data["nested"]
        pd.testing.assert_frame_equal(result["dataframe"], test_data["dataframe"])
