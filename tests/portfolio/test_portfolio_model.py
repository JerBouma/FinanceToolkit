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


def test_benchmark_data_is_aligned_with_carried_forward_prices():
    """A missing benchmark date carries the previous quote with a zero return; a dividend on a date
    the portfolio lacks moves to the next date, so nothing is counted twice or lost."""
    from financetoolkit.portfolio.portfolio_model import (  # noqa: PLC0415
        align_benchmark_data,
    )

    benchmark = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0],
            "Adj Close": [10.0, 11.0, 12.0, 13.0],
            "Dividends": [0.0, 0.5, 0.0, 0.0],
            "Return": [0.0, 0.1, 0.0909, 0.0833],
        },
        index=pd.PeriodIndex(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08"], freq="D"
        ),
    )
    dates = pd.PeriodIndex(
        ["2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08"], freq="D"
    )

    aligned = align_benchmark_data(benchmark, dates)

    assert list(aligned.columns) == list(benchmark.columns)
    assert aligned["Close"].tolist() == [10.0, 12.0, 12.0, 13.0]
    assert aligned["Dividends"].tolist() == [0.0, 0.5, 0.0, 0.0]
    assert aligned["Return"].tolist() == pytest.approx([0.0, 0.2, 0.0, 13 / 12 - 1])


def _format(data: pd.DataFrame) -> pd.DataFrame:
    """The transactions formatted with the column names of the template."""
    return format_portfolio_dataset(
        dataset=data,
        date_columns=["Date"],
        date_format_options=["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"],
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


@pytest.mark.parametrize(
    "dates",
    [
        ["05-06-2024", "07-06-2024"],
        ["2024/06/05", "2024/06/07"],
        ["2024-06-05 00:00:00", "2024-06-07 00:00:00"],
    ],
)
def test_transaction_dates_are_parsed_with_one_format(dates):
    data = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["AAPL", "AAPL"],
            "Price": [100.0, 101.0],
            "Volume": [1, 1],
            "Currency": ["USD", "USD"],
            "Costs": [0.0, 0.0],
        }
    )

    formatted = _format(data)

    assert [str(date) for date in formatted["Date"]] == ["2024-06-05", "2024-06-07"]


def _one_transaction(price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-06-05"],
            "Ticker": ["AAPL"],
            "Price": [price],
            "Volume": [1],
            "Currency": ["USD"],
            "Costs": [0.0],
        }
    )


def test_a_negative_price_is_refused():
    with pytest.raises(ValueError, match="negative price"):
        _format(_one_transaction(-100.0))


def test_a_price_of_zero_is_accepted():
    # A gift, spin-off or stock dividend is booked at zero cost.
    assert _format(_one_transaction(0.0))["Price"].tolist() == [0.0]


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("-4.587,24", -4587.24),
        ("-4,587.24", -4587.24),
        ("1,000", 1.0),
        ("2,500", 2.5),
        ("-1,500", -1.5),
        ("1,234,567", 1234567.0),
        ("1.234.567", 1234567.0),
        ("1,234,567.89", 1234567.89),
        ("1 234,56", 1234.56),
        ("1'234.56", 1234.56),
        ("1234,56", 1234.56),
        ("0,125", 0.125),
        ("4,5", 4.5),
        ("1.234", 1.234),
    ],
)
def test_numbers_with_thousands_separators_are_read(text, number):
    from financetoolkit.portfolio.helpers import convert_to_float  # noqa: PLC0415

    assert convert_to_float(text) == pytest.approx(number)


def test_duplicates_in_another_case_are_merged(tmp_path):
    data = pd.DataFrame(
        {
            "Date": ["2023-01-03", "2023-01-03"],
            "Name": ["Apple Inc", "Apple Inc"],
            "Ticker": ["aapl", "AAPL"],
            "Price": [150.0, 150.0],
            "Volume": [10, 10],
            "Currency": ["USD", "USD"],
            "Costs": [1.0, 1.0],
        }
    )
    location = tmp_path / "transactions.csv"
    data.to_csv(location, index=False)

    dataset = read_portfolio_dataset(
        excel_location=[str(location)],
        adjust_duplicates=True,
        date_column=["Date"],
        date_format_options=["%Y-%m-%d"],
        name_columns=["Name"],
        ticker_columns=["Ticker"],
        price_columns=["Price"],
        volume_columns=["Volume"],
        currency_columns=["Currency"],
        costs_columns=["Costs"],
        column_mapping={
            "date": "Date",
            "name": "Name",
            "identifier": "Identifier",
            "price": "Price",
            "volume": "Volume",
            "currency": "Currency",
            "costs": "Costs",
        },
    )[0]

    assert len(dataset) == 1
    assert dataset["Identifier"].iloc[0] == "AAPL"
    assert dataset["Volume"].iloc[0] == 20


@pytest.mark.parametrize(
    ("column", "numbers"),
    [
        (["1,500", "2,500", "1,234.50"], [1500.0, 2500.0, 1234.5]),
        (["1,500", "1,234,567"], [1500.0, 1234567.0]),
        (["1,500", "2,5"], [1.5, 2.5]),
        (["1.234,56", "2,500"], [1234.56, 2.5]),
        (["2,500", float("nan"), 3.0], [2.5, float("nan"), 3.0]),
    ],
)
def test_a_column_reads_its_separators_once(column, numbers):
    from financetoolkit.portfolio.helpers import (
        convert_column_to_float,
    )  # noqa: PLC0415

    converted = convert_column_to_float(pd.Series(column, dtype=object))

    assert converted.tolist() == pytest.approx(numbers, nan_ok=True)
