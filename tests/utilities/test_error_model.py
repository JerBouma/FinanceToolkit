"""Error Model Tests"""

# ruff: noqa: PLR2004

import inspect
from unittest.mock import patch

import pandas as pd

from financetoolkit.utilities import error_model


def test_handle_errors_decorator_preserves_function_attributes():
    """Test that handle_errors decorator preserves function attributes."""

    def test_function(x, y=5):
        """Test function docstring"""
        return x + y

    decorated_function = error_model.handle_errors(test_function)

    assert decorated_function.__name__ == test_function.__name__
    assert decorated_function.__doc__ == test_function.__doc__
    assert decorated_function.__signature__ == inspect.signature(test_function)
    assert decorated_function.__module__ == test_function.__module__


def test_handle_errors_decorator_success():
    """Test that handle_errors decorator works with successful function execution."""

    @error_model.handle_errors
    def test_function(x, y):
        return x + y

    result = test_function(2, 3)
    assert result == 5


def test_handle_errors_decorator_key_error():
    """Test handle_errors decorator with KeyError."""

    @error_model.handle_errors
    def test_function():
        raise KeyError("missing_column")

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "index name missing" in error_message
        assert "missing_column" in str(mock_logger.error.call_args[0][1])


def test_handle_errors_decorator_value_error():
    """Test handle_errors decorator with ValueError."""

    @error_model.handle_errors
    def test_function():
        raise ValueError("Invalid value")

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "error occurred while trying to run" in error_message


def test_handle_errors_decorator_attribute_error():
    """Test handle_errors decorator with AttributeError."""

    @error_model.handle_errors
    def test_function():
        raise AttributeError("'NoneType' object has no attribute 'value'")

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "error occurred while trying to run" in error_message


def test_handle_errors_decorator_zero_division_error():
    """Test handle_errors decorator with ZeroDivisionError."""

    @error_model.handle_errors
    def test_function():
        return 1 / 0

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "division by zero" in error_message


def test_handle_errors_decorator_index_error():
    """Test handle_errors decorator with IndexError."""

    @error_model.handle_errors
    def test_function():
        lst = [1, 2, 3]
        return lst[10]

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = test_function()

        assert isinstance(result, pd.Series)
        assert result.dtype == "object"
        assert result.empty

        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "missing data" in error_message


def test_check_for_error_messages_premium_query_parameter():
    """Test check_for_error_messages with premium query parameter error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"PREMIUM QUERY PARAMETER": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "premium query parameter" in error_message.lower()
        assert "AAPL" not in result  # Should be removed
        assert "MSFT" in result  # Should remain


def test_check_for_error_messages_exclusive_endpoint():
    """Test check_for_error_messages with exclusive endpoint error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"EXCLUSIVE ENDPOINT": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "exclusive endpoint" in error_message.lower()
        assert "AAPL" not in result


def test_check_for_error_messages_special_endpoint():
    """Test check_for_error_messages with special endpoint error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"SPECIAL ENDPOINT": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "special endpoint" in error_message.lower()
        assert "AAPL" not in result


def test_check_for_error_messages_bandwidth_limit():
    """Test check_for_error_messages with bandwidth limit error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"BANDWIDTH LIMIT REACH": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "bandwidth limit" in error_message.lower()
        assert "AAPL" not in result


def test_check_for_error_messages_limit_reach():
    """Test check_for_error_messages with limit reach error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"LIMIT REACH": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "limit from Financial Modeling Prep" in error_message
        assert "AAPL" not in result


def test_check_for_error_messages_yfinance_rate_limit():
    """Test check_for_error_messages with yfinance rate limit error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"YFINANCE RATE LIMIT REACHED": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "rate limit from Yahoo Finance" in error_message
        assert "AAPL" not in result


def test_check_for_error_messages_yfinance_rate_limit_fallback():
    """Test check_for_error_messages with yfinance rate limit fallback error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"YFINANCE RATE LIMIT REACHED FALLBACK": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "rate limit from Yahoo Finance" in error_message
        assert "previous attempt to use FinancialModelingPrep" in error_message
        assert "AAPL" not in result


def test_check_for_error_messages_yfinance_no_data():
    """Test check_for_error_messages with yfinance no data error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"YFINANCE RATE LIMIT OR NO DATA FOUND": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "rate limit from Yahoo Finance" in error_message
        assert "no data could be found" in error_message
        assert "AAPL" not in result


def test_check_for_error_messages_no_data_free_plan():
    """Test check_for_error_messages with no data error on free plan."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"NO DATA": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        # Should have two error calls - one for no data, one for free plan
        assert mock_logger.error.call_count == 2

        # Check first error message
        first_error = mock_logger.error.call_args_list[0][0][0]
        assert "no data" in first_error.lower()

        # Check second error message
        second_error = mock_logger.error.call_args_list[1][0][0]
        assert "Free plan" in second_error
        assert "API limit" in second_error

        assert "AAPL" not in result


def test_check_for_error_messages_no_data_premium_plan():
    """Test check_for_error_messages with no data error on premium plan."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"NO DATA": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Premium")

        # Should have only one error call
        assert mock_logger.error.call_count == 1

        error_message = mock_logger.error.call_args[0][0]
        assert "no data" in error_message.lower()
        assert "AAPL" not in result


def test_check_for_error_messages_us_stocks_only():
    """Test check_for_error_messages with US stocks only error."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"US STOCKS ONLY": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "US stocks only" in error_message
        assert "AAPL" not in result


def test_check_for_error_messages_no_errors():
    """Test check_for_error_messages with no errors column."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"NO ERRORS": ["success"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        # Should not call logger.error
        mock_logger.error.assert_not_called()

        # NO ERRORS ticker should still be removed
        assert "AAPL" not in result
        assert "MSFT" in result


def test_check_for_error_messages_dont_delete_tickers():
    """Test check_for_error_messages with delete_tickers=False."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"PREMIUM QUERY PARAMETER": ["error"]}),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(
            dataset_dict, "Free", delete_tickers=False
        )

        mock_logger.error.assert_called()

        # Both tickers should remain
        assert "AAPL" in result
        assert "MSFT" in result


def test_check_for_error_messages_multiple_errors():
    """Test check_for_error_messages with multiple types of errors."""
    dataset_dict = {
        "AAPL": pd.DataFrame({"PREMIUM QUERY PARAMETER": ["error"]}),
        "MSFT": pd.DataFrame({"BANDWIDTH LIMIT REACH": ["error"]}),
        "GOOGL": pd.DataFrame({"NO DATA": ["error"]}),
        "TSLA": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        # Should have multiple error calls
        assert mock_logger.error.call_count >= 3

        # Only TSLA should remain
        assert "AAPL" not in result
        assert "MSFT" not in result
        assert "GOOGL" not in result
        assert "TSLA" in result


def test_check_for_error_messages_empty_dataset():
    """Test check_for_error_messages with empty dataset."""
    dataset_dict = {}

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        # Should not call logger.error
        mock_logger.error.assert_not_called()

        # Result should be empty
        assert result == {}


def test_check_for_error_messages_error_priority():
    """Test check_for_error_messages with error priority (premium query parameter takes precedence)."""
    dataset_dict = {
        "AAPL": pd.DataFrame(
            {
                "PREMIUM QUERY PARAMETER": ["error"],
                "BANDWIDTH LIMIT REACH": ["error"],
                "NO DATA": ["error"],
            }
        ),
        "MSFT": pd.DataFrame({"data": [1, 2, 3]}),
    }

    with patch("financetoolkit.utilities.error_model.logger") as mock_logger:
        result = error_model.check_for_error_messages(dataset_dict, "Free")

        # Should identify AAPL as premium query parameter error (first in the if-elif chain)
        mock_logger.error.assert_called()

        # Check that the first error call is for premium query parameter
        first_error_call = mock_logger.error.call_args_list[0]
        assert "premium query parameter" in first_error_call[0][0].lower()

        assert "AAPL" not in result
        assert "MSFT" in result
