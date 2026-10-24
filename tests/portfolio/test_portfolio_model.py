"""Portfolio Model Tests"""

import os
import tempfile

import pandas as pd
import pytest

from financetoolkit.portfolio.portfolio_model import (
    convert_historical_data,
    convert_prices,
    format_portfolio_dataset,
    get_base_currency,
    get_exchange_rate_ticker,
    get_price_unit,
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


def test_currency_codes_keep_the_case_of_fractional_units():
    """GBp is pence and GBP pounds, so only the major currencies are written in capitals."""
    data = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Name": ["Vodafone", "Apple Inc", "Naspers"],
            "Ticker": ["VOD.L", "AAPL", "NPN.JO"],
            "Price": [70.0, 150.0, 3000.0],
            "Volume": [100, 50, 10],
            "Currency": ["GBp", "usd", "ZAc"],
            "Costs": [1.0, 2.0, 3.0],
        }
    )

    dataset = format_portfolio_dataset(
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
    )[0]

    assert dataset["Currency"].tolist() == ["GBp", "USD", "ZAc"]


def test_price_units_and_exchange_rates():
    assert get_price_unit("GBp") == ("GBP", 100)
    assert get_price_unit("GBX") == ("GBP", 100)
    assert get_price_unit("ILA") == ("ILS", 100)
    assert get_price_unit("eur") == ("EUR", 1)

    assert get_exchange_rate_ticker("USD", "GBp") == "USDGBP=X"
    assert get_exchange_rate_ticker("USD", "EUR") == "USDEUR=X"
    # Pence and pounds are one currency: a factor of 100, no exchange rate.
    assert get_exchange_rate_ticker("GBP", "GBp") is None
    assert get_exchange_rate_ticker("USD", "USD") is None

    assert (
        get_base_currency(
            {"VTI": "USD", "IEGA.AS": "EUR", "IGLT.L": "GBP"},
            {"VTI": 0.1, "IEGA.AS": 0.5, "IGLT.L": 0.4},
        )
        == "EUR"
    )
    assert get_base_currency({}, {"VTI": 1.0}) == "USD"


def test_convert_prices_divides_by_the_units_and_the_rate_of_the_day():
    days = pd.period_range("2023-01-02", "2023-01-06", freq="D")
    prices = pd.DataFrame(
        {"AAPL": 150.0, "VOD.L": 70.0, "IGLT.L": 10.0, "BARC.L": 200.0}, index=days
    )
    # No rate on the third day: the rate of the day before applies.
    rates = pd.DataFrame(
        {"USDGBP=X": [0.8, 0.8, None, 0.5, 0.5]}, index=days, dtype=float
    )

    converted = convert_prices(
        prices,
        {"AAPL": "USD", "VOD.L": "GBp", "IGLT.L": "GBP", "BARC.L": "GBX"},
        "USD",
        rates,
    )

    assert converted["AAPL"].tolist() == pytest.approx([150.0] * 5)
    assert converted["VOD.L"].tolist() == pytest.approx([0.875, 0.875, 0.875, 1.4, 1.4])
    assert converted["IGLT.L"].tolist() == pytest.approx([12.5, 12.5, 12.5, 20, 20])
    assert converted["BARC.L"].tolist() == pytest.approx([2.5, 2.5, 2.5, 4, 4])

    # A price in pence valued in pounds needs no exchange rate.
    assert convert_prices(prices[["VOD.L"]], {"VOD.L": "GBp"}, "GBP")[
        "VOD.L"
    ].tolist() == pytest.approx([0.7] * 5)

    with pytest.raises(ValueError, match="USDEUR=X"):
        convert_prices(prices[["AAPL"]], {"AAPL": "EUR"}, "USD", rates)


def test_convert_historical_data_recalculates_the_returns():
    days = pd.period_range("2023-01-02", "2023-01-04", freq="D")
    columns = pd.MultiIndex.from_product(
        [["Close", "Adj Close", "Dividends", "Return", "Cumulative Return"], ["VOD.L"]]
    )
    historical = pd.DataFrame(
        [
            [100.0, 100.0, 0.0, 0.0, 1.0],
            [100.0, 100.0, 5.0, 0.0, 1.0],
            [110.0, 110.0, 0.0, 0.1, 1.1],
        ],
        index=days,
        columns=columns,
    )
    rates = pd.DataFrame({"USDGBP=X": [0.8, 0.5, 0.5]}, index=days)

    converted = convert_historical_data(
        historical,
        {"VOD.L": ("GBp", "USD")},
        rates,
        ["Close", "Adj Close", "Dividends"],
    )

    assert converted[("Adj Close", "VOD.L")].tolist() == pytest.approx([1.25, 2, 2.2])
    assert converted[("Dividends", "VOD.L")].tolist() == pytest.approx([0, 0.1, 0])
    # In dollars the stock rose with the pound as well as in price.
    assert converted[("Return", "VOD.L")].iloc[1:].tolist() == pytest.approx([0.6, 0.1])
    assert converted[("Cumulative Return", "VOD.L")].tolist() == pytest.approx(
        [1, 1.6, 1.76]
    )
