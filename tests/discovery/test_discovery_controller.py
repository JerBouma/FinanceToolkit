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


def _news_mock_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT"],
            "Publisher": ["Benzinga", "Zacks"],
            "Title": ["Apple news headline", "Microsoft news headline"],
        }
    )


def test_get_stock_news(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_stock_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_stock_news(limit=2)

        recorder.capture(len(result))
        recorder.capture(result.iloc[0]["Symbol"] == "AAPL")


def test_get_general_news(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_general_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_general_news(limit=2)

        recorder.capture(len(result))


def test_get_press_releases(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_press_releases"
    ) as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_press_releases(limit=2)

        recorder.capture(len(result))


def test_get_crypto_news(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_crypto_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_crypto_news(limit=2)

        recorder.capture(len(result))


def test_get_forex_news(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_forex_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_forex_news(limit=2)

        recorder.capture(len(result))


def test_search_stock_news(recorder):
    with patch("financetoolkit.discovery.discovery_model.search_stock_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.search_stock_news(symbols="AAPL", limit=2)

        recorder.capture(len(result))


def test_search_press_releases(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.search_press_releases"
    ) as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.search_press_releases(symbols="AAPL", limit=2)

        recorder.capture(len(result))


def test_search_crypto_news(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.search_crypto_news"
    ) as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.search_crypto_news(symbols="BTCUSD", limit=2)

        recorder.capture(len(result))


def test_search_forex_news(recorder):
    with patch("financetoolkit.discovery.discovery_model.search_forex_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.search_forex_news(symbols="EURUSD", limit=2)

        recorder.capture(len(result))


def test_get_ipo_calendar(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_ipo_calendar") as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {
                "Date": ["2024-05-31"],
                "Company": ["SmartKem, Inc."],
                "Exchange": ["NASDAQ"],
            },
            index=pd.Index(["SMTK"], name="Symbol"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_ipo_calendar(
            start_date="2024-01-01", end_date="2024-06-01"
        )

        recorder.capture(len(result))
        recorder.capture(result.iloc[0]["Company"] == "SmartKem, Inc.")


def test_get_ipo_disclosures(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_ipo_disclosures"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Form": ["CERT"], "CIK": ["0001406234"]},
            index=pd.Index(["BIPH"], name="Symbol"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_ipo_disclosures(
            start_date="2024-01-01", end_date="2024-06-01"
        )

        recorder.capture(len(result))


def test_get_ipo_prospectuses(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_ipo_prospectuses"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Public Price Per Share": [73], "Form": ["S-1"]},
            index=pd.Index(["LUCYW"], name="Symbol"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_ipo_prospectuses(
            start_date="2024-01-01", end_date="2024-06-01"
        )

        recorder.capture(len(result))


def test_get_stock_splits_calendar(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_stock_splits_calendar"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Numerator": [617], "Denominator": [500], "Split Type": ["stock-split"]},
            index=pd.Index(["ALZ.ST"], name="Symbol"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_stock_splits_calendar(
            start_date="2024-01-01", end_date="2024-06-01"
        )

        recorder.capture(len(result))


def test_get_sector_performance_snapshot(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_sector_performance"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Average Change": [-0.31481]},
            index=pd.Index(["Basic Materials"], name="Sector"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_sector_performance(date="2024-02-01")

        recorder.capture(len(result))


def test_get_sector_performance_history(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_sector_performance"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Sector": ["Energy"], "Average Change": [1.399]},
            index=pd.to_datetime(["2024-03-01"]),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_sector_performance(sector="Energy")

        recorder.capture(len(result))


def test_get_industry_performance_snapshot(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_industry_performance"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Average Change": [3.866]},
            index=pd.Index(["Advertising Agencies"], name="Industry"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_industry_performance(date="2024-02-01")

        recorder.capture(len(result))


def test_get_industry_performance_history(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_industry_performance"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Industry": ["Biotechnology"], "Average Change": [2.614]},
            index=pd.to_datetime(["2024-03-01"]),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_industry_performance(industry="Biotechnology")

        recorder.capture(len(result))


def test_get_sector_pe_snapshot(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_sector_pe") as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"PE Ratio": [15.6877]}, index=pd.Index(["Basic Materials"], name="Sector")
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_sector_pe(date="2024-02-01")

        recorder.capture(len(result))


def test_get_sector_pe_history(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_sector_pe") as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Sector": ["Energy"], "PE Ratio": [5.4166]},
            index=pd.to_datetime(["2024-03-01"]),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_sector_pe(sector="Energy")

        recorder.capture(len(result))


def test_get_industry_pe_snapshot(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_industry_pe") as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"PE Ratio": [71.096]},
            index=pd.Index(["Advertising Agencies"], name="Industry"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_industry_pe(date="2024-02-01")

        recorder.capture(len(result))


def test_get_industry_pe_history(recorder):
    with patch("financetoolkit.discovery.discovery_model.get_industry_pe") as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {"Industry": ["Biotechnology"], "PE Ratio": [8.129]},
            index=pd.to_datetime(["2024-03-01"]),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_industry_pe(industry="Biotechnology")

        recorder.capture(len(result))


def test_get_mergers_acquisitions_latest(recorder):
    with patch(
        "financetoolkit.discovery.discovery_model.get_mergers_acquisitions_latest"
    ) as mock_fn:
        mock_fn.return_value = pd.DataFrame(
            {
                "Company Name": ["GENTHERM Inc"],
                "Targeted Company Name": ["Modine Manufacturing Company"],
            },
            index=pd.Index(["THRM"], name="Symbol"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_mergers_acquisitions_latest(limit=5)

        recorder.capture(len(result))
