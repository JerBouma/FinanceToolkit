"""Currencies Model Tests"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Suppress FutureWarning from pandas in currency conversion
warnings.filterwarnings(
    "ignore",
    message="Setting an item of incompatible dtype is deprecated",
    category=FutureWarning,
)
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

from financetoolkit import currencies_model


def test_determine_currencies_basic():
    """Test basic currency determination."""
    statement_currencies = pd.DataFrame(
        {
            "2020": ["EUR", "USD", "GBP"],
            "2021": ["EUR", "USD", "GBP"],
            "2022": ["EUR", "USD", "GBP"],
        },
        index=["ASML", "AAPL", "LLOY"],
    )

    historical_currencies = pd.Series(
        ["USD", "USD", "USD"], index=["ASML", "AAPL", "LLOY"]
    )

    result_currencies, currencies_list = currencies_model.determine_currencies(
        statement_currencies, historical_currencies
    )

    # Check result structure
    assert isinstance(result_currencies, pd.Series)
    assert isinstance(currencies_list, list)

    # Check values
    assert result_currencies.loc["ASML"] == "EURUSD=X"
    assert result_currencies.loc["AAPL"] == "USDUSD=X"
    assert result_currencies.loc["LLOY"] == "GBPUSD=X"

    # Check currencies list
    assert "EURUSD=X" in currencies_list
    assert "USDUSD=X" in currencies_list
    assert "GBPUSD=X" in currencies_list


def test_determine_currencies_with_nans():
    """Test currency determination with NaN values."""
    statement_currencies = pd.DataFrame(
        {
            "2020": ["EUR", np.nan, "GBP"],
            "2021": ["EUR", "USD", "GBP"],
            "2022": ["EUR", "USD", "GBP"],
        },
        index=["ASML", "AAPL", "LLOY"],
    )

    historical_currencies = pd.Series(
        ["USD", "USD", "USD"], index=["ASML", "AAPL", "LLOY"]
    )

    result_currencies, currencies_list = currencies_model.determine_currencies(
        statement_currencies, historical_currencies
    )

    # Should forward fill and backward fill NaN values
    assert result_currencies.loc["AAPL"] == "USDUSD=X"

    # NaN values should not be in currencies list
    assert len([c for c in currencies_list if pd.isna(c)]) == 0


def test_determine_currencies_missing_data():
    """Test currency determination with missing data handling."""
    statement_currencies = pd.DataFrame(
        {"2020": ["EUR", "USD"], "2021": [np.nan, "USD"], "2022": ["EUR", np.nan]},
        index=["ASML", "AAPL"],
    )

    historical_currencies = pd.Series(["USD", "USD"], index=["ASML", "AAPL"])

    result_currencies, currencies_list = currencies_model.determine_currencies(
        statement_currencies, historical_currencies
    )

    # Should use last available column
    assert result_currencies.loc["ASML"] == "EURUSD=X"
    assert result_currencies.loc["AAPL"] == "USDUSD=X"


def test_determine_currencies_same_currency():
    """Test currency determination when statement and historical currencies are same."""
    statement_currencies = pd.DataFrame(
        {"2020": ["USD", "USD"], "2021": ["USD", "USD"]}, index=["AAPL", "MSFT"]
    )

    historical_currencies = pd.Series(["USD", "USD"], index=["AAPL", "MSFT"])

    result_currencies, currencies_list = currencies_model.determine_currencies(
        statement_currencies, historical_currencies
    )

    # Should create USDUSD=X for both
    assert result_currencies.loc["AAPL"] == "USDUSD=X"
    assert result_currencies.loc["MSFT"] == "USDUSD=X"
    assert "USDUSD=X" in currencies_list


def test_convert_currencies_basic():
    """Test basic currency conversion."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {
            "2020": [1000, 500, 2000, 1000],
            "2021": [1100, 550, 2200, 1100],
            "2022": [1200, 600, 2400, 1200],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("ASML", "Revenue"),
                ("ASML", "Profit"),
                ("AAPL", "Revenue"),
                ("AAPL", "Profit"),
            ]
        ),
    )

    # Create currency mapping
    currencies = pd.Series(["EURUSD=X", "USDUSD=X"], index=["ASML", "AAPL"])

    # Create exchange rate data
    exchange_rates = pd.DataFrame(
        {"EURUSD=X": [1.2, 1.15, 1.1], "USDUSD=X": [1.0, 1.0, 1.0]},
        index=["2020", "2021", "2022"],
    )

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data, currencies, exchange_rates
        )

        # Check ASML values are converted (multiplied by exchange rate)
        assert result.loc[("ASML", "Revenue"), "2020"] == 1000 * 1.2
        assert result.loc[("ASML", "Revenue"), "2021"] == 1100 * 1.15

        # Check AAPL values are unchanged (same currency)
        assert result.loc[("AAPL", "Revenue"), "2020"] == 2000
        assert result.loc[("AAPL", "Revenue"), "2021"] == 2200

        # Check logger was called
        mock_logger.info.assert_called_once()
        ticker_info = mock_logger.info.call_args[0][2]
        assert "ASML (EUR to USD)" in ticker_info


def test_convert_currencies_with_items_not_to_adjust():
    """Test currency conversion excluding certain items."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {"2020": [1000, 500, 100], "2021": [1100, 550, 110]},
        index=pd.MultiIndex.from_tuples(
            [("ASML", "Revenue"), ("ASML", "Profit"), ("ASML", "Shares Outstanding")]
        ),
    )

    # Create currency mapping
    currencies = pd.Series(["EURUSD=X"], index=["ASML"])

    # Create exchange rate data
    exchange_rates = pd.DataFrame({"EURUSD=X": [1.2, 1.15]}, index=["2020", "2021"])

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data,
            currencies,
            exchange_rates,
            items_not_to_adjust=["Shares Outstanding"],
        )

        # Check revenue and profit are converted
        assert result.loc[("ASML", "Revenue"), "2020"] == 1000 * 1.2
        assert result.loc[("ASML", "Profit"), "2020"] == 500 * 1.2

        # Check shares outstanding is not converted
        assert result.loc[("ASML", "Shares Outstanding"), "2020"] == 100


def test_convert_currencies_with_financial_statement_name():
    """Test currency conversion with custom financial statement name."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {"2020": [1000], "2021": [1100]},
        index=pd.MultiIndex.from_tuples([("ASML", "Revenue")]),
    )

    # Create currency mapping
    currencies = pd.Series(["EURUSD=X"], index=["ASML"])

    # Create exchange rate data
    exchange_rates = pd.DataFrame({"EURUSD=X": [1.2, 1.15]}, index=["2020", "2021"])

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data,
            currencies,
            exchange_rates,
            financial_statement_name="income statement",
        )

        # Check logger message includes custom name
        mock_logger.info.assert_called_once()
        assert "income statement" in mock_logger.info.call_args[0][1]


def test_convert_currencies_no_conversion_needed():
    """Test currency conversion when no conversion is needed."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {"2020": [1000], "2021": [1100]},
        index=pd.MultiIndex.from_tuples([("AAPL", "Revenue")]),
    )

    # Create currency mapping (same currency)
    currencies = pd.Series(["USDUSD=X"], index=["AAPL"])

    # Create exchange rate data
    exchange_rates = pd.DataFrame({"USDUSD=X": [1.0, 1.0]}, index=["2020", "2021"])

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data, currencies, exchange_rates
        )

        # Check values are unchanged
        assert result.loc[("AAPL", "Revenue"), "2020"] == 1000
        assert result.loc[("AAPL", "Revenue"), "2021"] == 1100

        # Check no conversion message
        mock_logger.info.assert_not_called()


def test_convert_currencies_missing_currency_data():
    """Test currency conversion with missing currency data."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {"2020": [1000], "2021": [1100]},
        index=pd.MultiIndex.from_tuples([("ASML", "Revenue")]),
    )

    # Create currency mapping with NaN
    currencies = pd.Series([np.nan], index=["ASML"])

    # Create exchange rate data
    exchange_rates = pd.DataFrame({"EURUSD=X": [1.2, 1.15]}, index=["2020", "2021"])

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data, currencies, exchange_rates
        )

        # Check warning is logged
        mock_logger.warning.assert_called_once()
        assert "ASML" in mock_logger.warning.call_args[0][1]

        # Check values are unchanged
        assert result.loc[("ASML", "Revenue"), "2020"] == 1000


def test_convert_currencies_key_error():
    """Test currency conversion with KeyError (missing exchange rate)."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {"2020": [1000], "2021": [1100]},
        index=pd.MultiIndex.from_tuples([("ASML", "Revenue")]),
    )

    # Create currency mapping
    currencies = pd.Series(["EURUSD=X"], index=["ASML"])

    # Create exchange rate data without the required currency
    exchange_rates = pd.DataFrame({"GBPUSD=X": [1.3, 1.25]}, index=["2020", "2021"])

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data, currencies, exchange_rates
        )

        # Check warning is logged
        mock_logger.warning.assert_called_once()
        assert "ASML" in mock_logger.warning.call_args[0][1]

        # Check values are unchanged
        assert result.loc[("ASML", "Revenue"), "2020"] == 1000


def test_convert_currencies_value_error():
    """Test currency conversion with ValueError."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {"2020": [1000], "2021": [1100]},
        index=pd.MultiIndex.from_tuples([("ASML", "Revenue")]),
    )

    # Create currency mapping with invalid format
    currencies = pd.Series(["INVALID"], index=["ASML"])

    # Create exchange rate data
    exchange_rates = pd.DataFrame({"EURUSD=X": [1.2, 1.15]}, index=["2020", "2021"])

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data, currencies, exchange_rates
        )

        # Check warning is logged
        mock_logger.warning.assert_called_once()
        assert "ASML" in mock_logger.warning.call_args[0][1]


def test_convert_currencies_multiple_tickers():
    """Test currency conversion with multiple tickers."""
    # Create financial statement data
    financial_data = pd.DataFrame(
        {
            "2020": [1000, 500, 2000, 1000, 3000, 1500],
            "2021": [1100, 550, 2200, 1100, 3300, 1650],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("ASML", "Revenue"),
                ("ASML", "Profit"),
                ("AAPL", "Revenue"),
                ("AAPL", "Profit"),
                ("LLOY", "Revenue"),
                ("LLOY", "Profit"),
            ]
        ),
    )

    # Create currency mapping
    currencies = pd.Series(
        ["EURUSD=X", "USDUSD=X", "GBPUSD=X"], index=["ASML", "AAPL", "LLOY"]
    )

    # Create exchange rate data
    exchange_rates = pd.DataFrame(
        {"EURUSD=X": [1.2, 1.15], "USDUSD=X": [1.0, 1.0], "GBPUSD=X": [1.3, 1.25]},
        index=["2020", "2021"],
    )

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data, currencies, exchange_rates
        )

        # Check ASML values are converted
        assert result.loc[("ASML", "Revenue"), "2020"] == 1000 * 1.2

        # Check AAPL values are unchanged
        assert result.loc[("AAPL", "Revenue"), "2020"] == 2000

        # Check LLOY values are converted
        assert result.loc[("LLOY", "Revenue"), "2020"] == 3000 * 1.3

        # Check logger message includes all converted tickers
        mock_logger.info.assert_called_once()
        ticker_info = mock_logger.info.call_args[0][2]
        assert "ASML (EUR to USD)" in ticker_info
        assert "LLOY (GBP to USD)" in ticker_info


def test_convert_currencies_empty_dataframe():
    """Test currency conversion with empty DataFrame."""
    # Create empty financial statement data
    financial_data = pd.DataFrame()

    # Create currency mapping
    currencies = pd.Series([], dtype=object)

    # Create exchange rate data
    exchange_rates = pd.DataFrame()

    with patch("financetoolkit.currencies_model.logger") as mock_logger:
        result = currencies_model.convert_currencies(
            financial_data, currencies, exchange_rates
        )

        # Check result is empty
        assert result.empty

        # Check no logger calls
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()


def test_determine_currencies_single_period():
    """Test determine_currencies with single period."""
    statement_currencies = pd.DataFrame(
        {"2020": ["EUR", "USD"]}, index=["ASML", "AAPL"]
    )

    historical_currencies = pd.Series(["USD", "USD"], index=["ASML", "AAPL"])

    result_currencies, currencies_list = currencies_model.determine_currencies(
        statement_currencies, historical_currencies
    )

    # Check results
    assert result_currencies.loc["ASML"] == "EURUSD=X"
    assert result_currencies.loc["AAPL"] == "USDUSD=X"
    assert len(currencies_list) == 2


# ruff: noqa
