"""Discovery Controller Tests"""

from unittest.mock import patch

import pandas as pd

from financetoolkit.discovery import discovery_controller

# pylint: disable=missing-function-docstring


def test_discovery_controller_initialization(recorder):
    """Test that Discovery controller initializes correctly."""
    discovery = discovery_controller.Discovery(api_key="test_key")

    recorder.capture(discovery._api_key == "test_key")
    recorder.capture(discovery._fmp_plan == "Premium")


def test_search_instruments(recorder):
    """Test search_instruments method."""
    with patch(
        "financetoolkit.discovery.discovery_model.get_instruments"
    ) as mock_get_instruments:
        mock_data = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "GOOGL"],
                "Name": ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc."],
                "Currency": ["USD", "USD", "USD"],
            }
        )
        mock_get_instruments.return_value = mock_data

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.search_instruments(query="Apple")

        recorder.capture(len(result))
        recorder.capture(result.iloc[0]["Symbol"] == "AAPL")


def test_get_stock_list(recorder):
    """Test get_stock_list method."""
    with patch(
        "financetoolkit.discovery.discovery_model.get_stock_list"
    ) as mock_get_stock_list:
        mock_data = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "GOOGL"],
                "Name": ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc."],
                "Price": [150.0, 300.0, 2800.0],
                "Exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
            }
        )
        mock_get_stock_list.return_value = mock_data

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_stock_list()

        recorder.capture(len(result))
        recorder.capture(result.iloc[0]["Symbol"] == "AAPL")


def test_get_stock_quotes(recorder):
    """Test get_stock_quotes method."""
    with patch(
        "financetoolkit.discovery.discovery_model.get_stock_quotes"
    ) as mock_get_stock_quotes:
        mock_data = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "GOOGL"],
                "Price": [150.0, 300.0, 2800.0],
                "Change": [1.5, -2.3, 15.2],
                "ChangesPercentage": [1.0, -0.76, 0.55],
            }
        )
        mock_get_stock_quotes.return_value = mock_data

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_stock_quotes()

        recorder.capture(len(result))
        recorder.capture(result.iloc[0]["Symbol"] == "AAPL")


def test_get_stock_screener(recorder):
    """Test get_stock_screener method."""
    with patch(
        "financetoolkit.discovery.discovery_model.get_stock_screener"
    ) as mock_get_stock_screener:
        mock_data = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT"],
                "Name": ["Apple Inc.", "Microsoft Corporation"],
                "MarketCap": [3000000000000, 2800000000000],
                "Sector": ["Technology", "Technology"],
            }
        )
        mock_get_stock_screener.return_value = mock_data

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_stock_screener(market_cap_higher=1000000000000)

        recorder.capture(len(result))
        recorder.capture(result.iloc[0]["Symbol"] == "AAPL")
