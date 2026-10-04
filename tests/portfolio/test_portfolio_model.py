"""Portfolio Model Tests"""

import os
import tempfile

import pandas as pd
import pytest

from financetoolkit.portfolio.portfolio_model import (
    format_portfolio_dataset,
    read_portfolio_dataset,
)


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Name": ["Apple Inc", "Microsoft Corp", "Amazon.com Inc"],
            "Ticker": ["AAPL", "MSFT", "AMZN"],
            "Price": [150.0, 250.0, 100.0],
            "Volume": [100, 50, 200],
            "Currency": ["USD", "USD", "USD"],
            "Costs": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def sample_excel_file(sample_portfolio_data):
    """Create a temporary Excel file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        sample_portfolio_data.to_excel(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_csv_file(sample_portfolio_data):
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        sample_portfolio_data.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)


def test_read_portfolio_dataset_missing_file():
    """Test reading portfolio dataset with missing file"""
    with pytest.raises(FileNotFoundError):
        read_portfolio_dataset(
            excel_location=["non_existent_file.xlsx"],
            adjust_duplicates=False,
            date_column=["Date"],
            date_format_options=["%Y-%m-%d"],
            name_columns=["Name"],
            ticker_columns=["Ticker"],
            price_columns=["Price"],
            volume_columns=["Volume"],
            currency_columns=["Currency"],
            costs_columns=["Costs"],
            column_mapping={},
        )


def test_format_portfolio_dataset_with_invalid_date():
    """Test formatting portfolio dataset with invalid date format"""
    data = pd.DataFrame(
        {
            "Date": ["invalid-date"],
            "Name": ["Apple Inc"],
            "Ticker": ["AAPL"],
            "Price": [150.0],
            "Volume": [100],
            "Currency": ["USD"],
            "Costs": [1.0],
        }
    )

    with pytest.raises(ValueError):
        format_portfolio_dataset(
            dataset=data,
            date_columns=["Date"],
            date_format_options=["%Y-%m-%d"],
            name_columns=["Name"],
            tickers_columns=["Ticker"],
            price_columns=["Price"],
            volume_columns=["Volume"],
            currency_columns=["Currency"],
            costs_columns=["Costs"],
            column_mapping={
                "date": "Date",
                "name": "Name",
                "identifier": "Ticker",
                "price": "Price",
                "volume": "Volume",
                "currency": "Currency",
                "costs": "Costs",
            },
        )


def test_format_portfolio_dataset_with_missing_columns():
    """Test formatting portfolio dataset with missing required columns"""
    data = pd.DataFrame(
        {
            "Date": ["2023-01-01"],
            "Name": ["Apple Inc"],
            # Missing Ticker column
            "Price": [150.0],
            "Volume": [100],
            "Currency": ["USD"],
            "Costs": [1.0],
        }
    )

    with pytest.raises(ValueError):
        format_portfolio_dataset(
            dataset=data,
            date_columns=["Date"],
            date_format_options=["%Y-%m-%d"],
            name_columns=["Name"],
            tickers_columns=["Ticker"],  # Column doesn't exist
            price_columns=["Price"],
            volume_columns=["Volume"],
            currency_columns=["Currency"],
            costs_columns=["Costs"],
            column_mapping={
                "date": "Date",
                "name": "Name",
                "identifier": "Ticker",
                "price": "Price",
                "volume": "Volume",
                "currency": "Currency",
                "costs": "Costs",
            },
        )


def test_format_portfolio_dataset_with_mixed_length_currency_codes():
    """Test that an invalid-length currency code is rejected even when it shares a
    column with valid 3-letter codes.

    Regression test for a bug where only the *maximum* string length in the currency
    column was checked against CURRENCY_CODE_LENGTH, so a single valid 3-letter code
    (e.g. "USD") masked an invalid code elsewhere in the same column (e.g. "EU"),
    letting it silently pass validation.
    """
    data = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02"],
            "Name": ["Apple Inc", "Microsoft Corp"],
            "Ticker": ["AAPL", "MSFT"],
            "Price": [150.0, 250.0],
            "Volume": [100, 50],
            "Currency": ["USD", "EU"],
            "Costs": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="3-letter currency codes"):
        format_portfolio_dataset(
            dataset=data,
            date_columns=["Date"],
            date_format_options=["%Y-%m-%d"],
            name_columns=["Name"],
            tickers_columns=["Ticker"],
            price_columns=["Price"],
            volume_columns=["Volume"],
            currency_columns=["Currency"],
            costs_columns=["Costs"],
            column_mapping={
                "date": "Date",
                "name": "Name",
                "identifier": "Ticker",
                "price": "Price",
                "volume": "Volume",
                "currency": "Currency",
                "costs": "Costs",
            },
        )


def test_read_portfolio_dataset_edge_cases():
    """Test edge cases for read_portfolio_dataset"""
    # Empty file
    empty_data = pd.DataFrame()

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        empty_data.to_excel(tmp.name, index=False)

        with pytest.raises(Exception):  # Should raise some exception for empty data
            read_portfolio_dataset(
                excel_location=[tmp.name],
                adjust_duplicates=False,
                date_column=["Date"],
                date_format_options=["%Y-%m-%d"],
                name_columns=["Name"],
                ticker_columns=["Ticker"],
                price_columns=["Price"],
                volume_columns=["Volume"],
                currency_columns=["Currency"],
                costs_columns=["Costs"],
                column_mapping={},
            )

    os.unlink(tmp.name)
