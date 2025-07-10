"""Technical Helpers Tests"""

# ruff: noqa: PLR2004, F841

import inspect
from unittest.mock import patch

import pandas as pd

from financetoolkit.technicals import helpers


def test_handle_errors_decorator_preserves_function_attributes():
    """Test that handle_errors decorator preserves function attributes."""

    def test_function(x, y=5):
        """Test function docstring"""
        return x + y

    decorated_function = helpers.handle_errors(test_function)

    assert decorated_function.__name__ == test_function.__name__
    assert decorated_function.__doc__ == test_function.__doc__
    assert decorated_function.__signature__ == inspect.signature(test_function)
    assert decorated_function.__module__ == test_function.__module__


def test_handle_errors_decorator_success():
    """Test that handle_errors decorator works with successful function execution."""

    @helpers.handle_errors
    def test_function(x, y):
        return x + y

    result = test_function(2, 3)
    assert result == 5


def test_handle_errors_decorator_key_error():
    """Test handle_errors decorator with KeyError."""

    @helpers.handle_errors
    def test_function():
        raise KeyError("missing_column")

    with patch("financetoolkit.technicals.helpers.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "index name missing" in error_message


def test_handle_errors_decorator_value_error():
    """Test handle_errors decorator with ValueError."""

    @helpers.handle_errors
    def test_function():
        raise ValueError("Invalid value")

    with patch("financetoolkit.technicals.helpers.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "error occurred while trying to run" in error_message


def test_handle_errors_decorator_attribute_error():
    """Test handle_errors decorator with AttributeError."""

    @helpers.handle_errors
    def test_function():
        raise AttributeError("'NoneType' object has no attribute 'value'")

    with patch("financetoolkit.technicals.helpers.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "error occurred while trying to run" in error_message


def test_handle_errors_decorator_zero_division_error():
    """Test handle_errors decorator with ZeroDivisionError."""

    @helpers.handle_errors
    def test_function():
        return 1 / 0

    with patch("financetoolkit.technicals.helpers.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "error occurred while trying to run" in error_message


def test_handle_errors_decorator_with_return_value():
    """Test handle_errors decorator with function that returns specific value."""

    @helpers.handle_errors
    def test_function():
        return pd.Series([1, 2, 3, 4, 5])

    result = test_function()

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    assert result.tolist() == [1, 2, 3, 4, 5]


def test_handle_errors_decorator_with_dataframe_return():
    """Test handle_errors decorator with function that returns DataFrame."""

    @helpers.handle_errors
    def test_function():
        return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    result = test_function()

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)


def test_handle_errors_decorator_with_parameters():
    """Test handle_errors decorator with function that has parameters."""

    @helpers.handle_errors
    def test_function(x, y, z=10):
        return x + y + z

    result = test_function(1, 2, z=3)

    assert result == 6


def test_handle_errors_decorator_with_kwargs():
    """Test handle_errors decorator with function that uses kwargs."""

    @helpers.handle_errors
    def test_function(**kwargs):
        return sum(kwargs.values())

    result = test_function(a=1, b=2, c=3)

    assert result == 6


def test_handle_errors_decorator_error_logging_details():
    """Test that handle_errors decorator logs proper error details."""

    @helpers.handle_errors
    def test_function_with_specific_name():
        raise ValueError("Specific error message")

    with patch("financetoolkit.technicals.helpers.logger") as mock_logger:
        result = test_function_with_specific_name()

        # Check that the function name is included in the error message
        mock_logger.error.assert_called_once()
        error_args = mock_logger.error.call_args[0]
        assert "test_function_with_specific_name" in error_args[1]
        assert "Specific error message" in str(error_args[2])


def test_handle_errors_decorator_maintains_original_behavior():
    """Test that handle_errors decorator maintains original function behavior when no error occurs."""

    def original_function(a, b, c=None):
        """Original function docstring"""
        if c is None:
            return a + b
        return a + b + c

    decorated_function = helpers.handle_errors(original_function)

    # Test that decorated function works the same as original
    assert decorated_function(1, 2) == original_function(1, 2)
    assert decorated_function(1, 2, c=3) == original_function(1, 2, c=3)

    # Test that attributes are preserved
    assert decorated_function.__name__ == original_function.__name__
    assert decorated_function.__doc__ == original_function.__doc__


def test_handle_errors_decorator_with_none_return():
    """Test handle_errors decorator with function that returns None."""

    @helpers.handle_errors
    def test_function():
        return None

    result = test_function()

    assert result is None


def test_handle_errors_decorator_with_empty_series():
    """Test handle_errors decorator with function that returns empty Series."""

    @helpers.handle_errors
    def test_function():
        return pd.Series([], dtype="float64")

    result = test_function()

    assert isinstance(result, pd.Series)
    assert result.empty
    assert result.dtype == "float64"


def test_handle_errors_decorator_complex_function():
    """Test handle_errors decorator with complex function that performs calculations."""

    @helpers.handle_errors
    def complex_calculation(data):
        """Perform complex calculation on data"""
        return data.rolling(window=5).mean().dropna()

    # Test with valid data
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = complex_calculation(data)

    assert isinstance(result, pd.Series)
    assert len(result) == 6  # 10 - 5 + 1

    # Test with invalid data (will cause error)
    with patch("financetoolkit.technicals.helpers.logger") as mock_logger:
        result = complex_calculation("invalid_data")

        assert isinstance(result, pd.Series)
        assert result.empty
        mock_logger.error.assert_called_once()
