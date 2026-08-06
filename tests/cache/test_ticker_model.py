"""Ticker Model Tests"""

import pandas as pd
import pytest

from financetoolkit.cache import ticker_model


def create_column_oriented_frame(tickers: list[str]) -> pd.DataFrame:
    """Build a frame laid out like a profile: fields as rows, tickers as columns."""
    return pd.DataFrame(
        {ticker: [f"{ticker}-sector", f"{ticker}-country"] for ticker in tickers},
        index=["Sector", "Country"],
    )


def create_index_oriented_frame(tickers: list[str]) -> pd.DataFrame:
    """Build a frame laid out like a calendar: (ticker, date) rows, fields as columns."""
    return pd.concat(
        {
            ticker: pd.DataFrame(
                {"EPS": [1.0, 2.0]}, index=["2020-01-01", "2020-04-01"]
            )
            for ticker in tickers
        },
        axis=0,
    )


def test_select_ticker_on_columns():
    """Test that a column-oriented frame is sliced down to one ticker."""
    data = create_column_oriented_frame(["AAPL", "MSFT"])

    selection = ticker_model.select_ticker(data, "AAPL", ticker_model.TICKER_ON_COLUMNS)

    assert list(selection.columns) == ["AAPL"]
    assert list(selection.index) == ["Sector", "Country"]


def test_select_ticker_on_index():
    """Test that an index-oriented frame is sliced down to one ticker."""
    data = create_index_oriented_frame(["AAPL", "MSFT"])

    selection = ticker_model.select_ticker(data, "MSFT", ticker_model.TICKER_ON_INDEX)

    assert set(selection.index.get_level_values(0)) == {"MSFT"}
    assert len(selection) == 2


def test_select_ticker_returns_none_when_absent():
    """Test that a ticker with no data yields None rather than an empty frame."""
    data = create_column_oriented_frame(["AAPL"])

    assert (
        ticker_model.select_ticker(data, "MSFT", ticker_model.TICKER_ON_COLUMNS) is None
    )


def test_select_ticker_rejects_an_unknown_axis():
    """Test that an unsupported axis raises rather than silently misbehaving."""
    with pytest.raises(ValueError, match="Unsupported ticker axis"):
        ticker_model.select_ticker(
            create_column_oriented_frame(["AAPL"]), "AAPL", "diagonal"
        )


def test_split_and_combine_round_trips_a_column_oriented_frame():
    """Test that splitting per ticker and recombining reproduces the original."""
    tickers = ["AAPL", "MSFT", "ASML"]
    original = create_column_oriented_frame(tickers)

    pieces = [
        ticker_model.select_ticker(original, ticker, ticker_model.TICKER_ON_COLUMNS)
        for ticker in tickers
    ]
    combined = ticker_model.combine_tickers(
        pieces, tickers, ticker_model.TICKER_ON_COLUMNS
    )

    pd.testing.assert_frame_equal(combined, original)


def test_split_and_combine_round_trips_an_index_oriented_frame():
    """Test the same round trip for the layout that keeps tickers on the index."""
    tickers = ["AAPL", "MSFT"]
    original = create_index_oriented_frame(tickers)

    pieces = [
        ticker_model.select_ticker(original, ticker, ticker_model.TICKER_ON_INDEX)
        for ticker in tickers
    ]
    combined = ticker_model.combine_tickers(
        pieces, tickers, ticker_model.TICKER_ON_INDEX
    )

    pd.testing.assert_frame_equal(combined, original)


def test_combine_orders_by_the_requested_tickers():
    """Test that the output order follows the request, not the order of the pieces."""
    tickers = ["AAPL", "MSFT"]
    original = create_column_oriented_frame(tickers)

    pieces = [
        ticker_model.select_ticker(original, "MSFT", ticker_model.TICKER_ON_COLUMNS),
        ticker_model.select_ticker(original, "AAPL", ticker_model.TICKER_ON_COLUMNS),
    ]
    combined = ticker_model.combine_tickers(
        pieces, tickers, ticker_model.TICKER_ON_COLUMNS
    )

    assert list(combined.columns) == ["AAPL", "MSFT"]


def test_combine_handles_multiindex_columns():
    """Test the layout where tickers end up as the last of several column levels."""
    tickers = ["AAPL", "MSFT"]
    original = pd.concat(
        {
            ticker: pd.DataFrame({"Long": [1.0], "Short": [2.0]}, index=["2020-01-01"])
            for ticker in tickers
        },
        axis=0,
    ).unstack(level=0)

    pieces = [
        ticker_model.select_ticker(original, ticker, ticker_model.TICKER_ON_COLUMNS)
        for ticker in tickers
    ]
    combined = ticker_model.combine_tickers(
        pieces, tickers, ticker_model.TICKER_ON_COLUMNS
    )

    assert set(combined.columns.get_level_values(-1)) == {"AAPL", "MSFT"}
    assert combined.shape == original.shape


def test_combine_with_nothing_returns_an_empty_frame():
    """Test that combining no pieces yields an empty frame rather than raising."""
    assert ticker_model.combine_tickers(
        [], ["AAPL"], ticker_model.TICKER_ON_COLUMNS
    ).empty
