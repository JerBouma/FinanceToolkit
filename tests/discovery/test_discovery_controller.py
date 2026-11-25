"""Discovery Controller Tests"""

from unittest.mock import patch

import pandas as pd
import pytest

from financetoolkit.discovery import discovery_controller

# Every Discovery is built with a placeholder key, whose plan probe is answered offline.
pytestmark = pytest.mark.usefixtures("known_fmp_plan")


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


def test_get_industry_statistics_reads_the_dataset_of_the_region():
    with patch(
        "financetoolkit.models.valuation.damodaran_model.get_industry_statistics"
    ) as mock_statistics:
        mock_statistics.return_value = pd.DataFrame(
            {"Number of Firms": [66], "Cost of Capital": [0.0983]},
            index=pd.Index(["Semiconductor"], name="Industry"),
        )

        discovery = discovery_controller.Discovery(api_key="test_key")
        result = discovery.get_industry_statistics(
            dataset="cost_of_capital", region="Europe"
        )

        mock_statistics.assert_called_once_with(
            dataset="cost_of_capital", region="Europe"
        )
        assert result.loc["Semiconductor", "Cost of Capital"] == 0.0983


def test_search_news_without_symbols():
    discovery = discovery_controller.Discovery(api_key="test_key")

    # Without symbols, stock news and press releases are the latest of every company.
    with patch("financetoolkit.discovery.discovery_model.get_stock_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()
        assert len(discovery.search_stock_news(limit=2)) == len(_news_mock_data())
        mock_fn.assert_called_once()
    with patch(
        "financetoolkit.discovery.discovery_model.get_press_releases"
    ) as mock_fn:
        mock_fn.return_value = _news_mock_data()
        discovery.search_press_releases(limit=2)
        mock_fn.assert_called_once()

    # Crypto and forex news cover the largest coins and the major pairs.
    with patch(
        "financetoolkit.discovery.discovery_model.search_crypto_news"
    ) as mock_fn:
        mock_fn.return_value = _news_mock_data()
        discovery.search_crypto_news(limit=2)
        assert mock_fn.call_args.kwargs["symbols"] == ["BTCUSD", "ETHUSD"]
    with patch("financetoolkit.discovery.discovery_model.search_forex_news") as mock_fn:
        mock_fn.return_value = _news_mock_data()
        discovery.search_forex_news(limit=2)
        assert "EURUSD" in mock_fn.call_args.kwargs["symbols"]


@pytest.mark.parametrize(
    "method, name",
    [
        ("get_sector_performance", "sector"),
        ("get_industry_performance", "industry"),
        ("get_sector_pe", "sector"),
        ("get_industry_pe", "industry"),
    ],
)
def test_the_sector_and_industry_histories_check_their_dates(method, name):
    discovery = discovery_controller.Discovery(api_key="test_key")

    with patch(f"financetoolkit.discovery.discovery_model.{method}") as mock_fn:
        with pytest.raises(ValueError, match="start_date"):
            getattr(discovery, method)(**{name: "Energy", "start_date": "2024-13-01"})
        with pytest.raises(ValueError, match="must be on or before"):
            getattr(discovery, method)(
                **{name: "Energy", "start_date": "2025-01-01", "end_date": "2024-01-01"}
            )

    mock_fn.assert_not_called()


@pytest.mark.parametrize(
    "method, name, value",
    [
        ("get_sector_performance", "sector", "Energy"),
        ("get_industry_performance", "industry", "Biotechnology"),
        ("get_sector_pe", "sector", "Energy"),
        ("get_industry_pe", "industry", "Biotechnology"),
    ],
)
def test_the_sector_and_industry_snapshots_work_without_arguments(method, name, value):
    from financetoolkit.utilities.error_model import InvalidArgumentError

    discovery = discovery_controller.Discovery(api_key="test_key")

    with patch(f"financetoolkit.discovery.discovery_model.{method}") as mock_fn:
        mock_fn.return_value = pd.DataFrame({"Sector": ["Energy"]})
        getattr(discovery, method)()

        # The snapshot of the latest business day.
        date = pd.Timestamp(mock_fn.call_args.kwargs["date"])
        assert date.dayofweek < 5 and date <= pd.Timestamp.today()
        assert mock_fn.call_args.kwargs[name] is None

        with pytest.raises(InvalidArgumentError, match="not both"):
            getattr(discovery, method)(**{"date": "2025-01-02", name: value})
        with pytest.raises(InvalidArgumentError, match="exchange must be one of"):
            getattr(discovery, method)(exchange="INVALID")

    assert mock_fn.call_count == 1


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("search_instruments", {"query": "Apple", "search_method": "invalid"}),
        ("search_instruments", {"query": ""}),
        ("search_instruments", {}),
        ("get_sector_performance", {"sector": "Nonsense"}),
        ("get_sector_pe", {"sector": "Nonsense"}),
        ("get_sectors_performance", {"exchange": "INVALID"}),
        ("get_stock_screener", {"market_cap_higher": 10, "market_cap_lower": 5}),
        ("get_stock_screener", {"sector": "Nonsense"}),
        ("search_stock_news", {"symbols": ""}),
        ("search_stock_news", {"symbols": []}),
        ("search_press_releases", {"symbols": " , "}),
        ("search_crypto_news", {"symbols": []}),
        ("search_forex_news", {"symbols": ""}),
        ("get_crypto_news", {"limit": 0}),
        ("get_ipo_calendar", {"start_date": "2025-02-01", "end_date": "2025-01-01"}),
    ],
)
def test_invalid_discovery_arguments_are_refused_before_any_request(method, arguments):
    from financetoolkit.utilities.error_model import InvalidArgumentError

    discovery = discovery_controller.Discovery(api_key="test_key")

    with (
        patch(
            "financetoolkit.discovery.discovery_model.get_cached_financial_data"
        ) as mock_fetch,
        pytest.raises(InvalidArgumentError),
    ):
        getattr(discovery, method)(**arguments)

    mock_fetch.assert_not_called()


def test_get_industry_statistics_reports_an_error_page_as_no_data(monkeypatch, caplog):
    from financetoolkit.cache.cache_controller import clear_active_cache
    from financetoolkit.models.valuation import damodaran_model

    class ErrorPage:
        content = b"<!DOCTYPE html><html><body>Unavailable</body></html>"

    clear_active_cache()
    monkeypatch.setattr(
        damodaran_model, "get_request", lambda url, timeout: ErrorPage()
    )

    discovery = discovery_controller.Discovery(api_key="test_key")
    result = discovery.get_industry_statistics(
        dataset="cost_of_capital", region="Europe"
    )

    assert result.empty
    assert "not an Excel workbook" in caplog.text
    assert caplog.text.count("Damodaran") == 1


@pytest.mark.parametrize(
    "method",
    [
        "get_sector_performance",
        "get_industry_performance",
        "get_sector_pe",
        "get_industry_pe",
    ],
)
def test_a_default_snapshot_steps_back_over_a_market_holiday(method, caplog):
    discovery = discovery_controller.Discovery(api_key="test_key")
    snapshot = pd.DataFrame({"Sector": ["Energy"]})

    with patch(f"financetoolkit.discovery.discovery_model.{method}") as mock_fn:
        # The latest business day (a holiday) has no snapshot; the one before it has.
        mock_fn.side_effect = [pd.DataFrame(), snapshot]
        result = getattr(discovery, method)()

        first, second = (
            pd.Timestamp(call.kwargs["date"]) for call in mock_fn.call_args_list
        )
        assert second == first - pd.offsets.BDay()
        assert result.equals(snapshot)

        # An explicit date stays as given.
        mock_fn.reset_mock(side_effect=True)
        mock_fn.return_value = pd.DataFrame()
        assert getattr(discovery, method)(date="2026-11-26").empty
        assert mock_fn.call_count == 1

        # Without any snapshot in the last business days, one warning says so.
        mock_fn.reset_mock()
        assert getattr(discovery, method)().empty
        assert mock_fn.call_count == 1 + discovery_controller.SNAPSHOT_LOOKBACK_DAYS
    assert caplog.text.count("has no") == 1


def test_a_failing_default_snapshot_does_not_step_back():
    from financetoolkit.utilities import error_model

    discovery = discovery_controller.Discovery(api_key="test_key")

    def unreachable(**kwargs):  # noqa: ARG001
        error_model.report_request_failure()
        return pd.DataFrame()

    with patch(
        "financetoolkit.discovery.discovery_model.get_sector_performance",
        side_effect=unreachable,
    ) as mock_fn:
        assert discovery.get_sector_performance().empty

    assert mock_fn.call_count == 1


def test_a_page_count_that_is_not_a_whole_number_is_refused():
    import numpy as np

    from financetoolkit.utilities.error_model import InvalidArgumentError

    discovery = discovery_controller.Discovery(api_key="test_key")

    with patch("financetoolkit.discovery.discovery_model.get_stock_news") as mock_fn:
        with pytest.raises(InvalidArgumentError, match="pages must be a whole number"):
            discovery.get_stock_news(pages="2")
        mock_fn.assert_not_called()

        # A numpy whole number, as read from a DataFrame, is a page count.
        mock_fn.return_value = pd.DataFrame()
        discovery.get_stock_news(pages=np.int64(2))


def test_a_concurrent_failed_request_does_not_stop_the_step_back(caplog):
    import contextvars

    from financetoolkit.utilities import error_model

    discovery = discovery_controller.Discovery(api_key="test_key")
    snapshot = pd.DataFrame({"Sector": ["Energy"]})
    answers = iter([pd.DataFrame(), snapshot])

    def holiday_while_another_call_fails(**kwargs):  # noqa: ARG001
        # Another MCP request, in its own context, fails meanwhile.
        contextvars.Context().run(error_model.report_request_failure)
        return next(answers)

    with patch(
        "financetoolkit.discovery.discovery_model.get_sector_performance",
        side_effect=holiday_while_another_call_fails,
    ) as mock_fn:
        assert discovery.get_sector_performance().equals(snapshot)

    assert mock_fn.call_count == 2  # noqa: PLR2004
