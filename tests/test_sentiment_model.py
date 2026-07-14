"""Sentiment Model Tests"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from financetoolkit import sentiment_model

EXPECTED_SOURCE_CALLS = 2


def test_get_sentiment_returns_source_ticker_dataframe():
    """Adanos trend history should be aligned to the requested date range."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "stocks": [
            {"ticker": "AAPL", "trend_history": [41.0, 43.5, 46.2]},
            {"ticker": "MSFT", "trend_history": [55.1, 53.4, 54.0]},
        ]
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        "financetoolkit.sentiment_model.requests.get",
        return_value=mock_response,
    ) as mock_get:
        result = sentiment_model.get_sentiment(
            ["AAPL", "MSFT"],
            api_key="test-key",
            source="reddit",
            start_date="2024-01-01",
            end_date="2024-01-03",
        )

    assert isinstance(result.index, pd.PeriodIndex)
    assert result.index.freqstr == "D"
    assert list(result.columns) == [("reddit", "AAPL"), ("reddit", "MSFT")]
    assert result.loc[pd.Period("2024-01-03", freq="D"), ("reddit", "AAPL")] == pytest.approx(46.2)

    _, kwargs = mock_get.call_args
    assert kwargs["params"] == {"tickers": "AAPL,MSFT", "days": 3}
    assert kwargs["headers"]["X-API-Key"] == "test-key"


def test_get_sentiment_supports_multiple_sources():
    """Multiple Adanos sources should be returned as source-level columns."""
    mock_response = Mock()
    mock_response.json.return_value = {"stocks": [{"ticker": "AAPL", "trend_history": [41.0]}]}
    mock_response.raise_for_status.return_value = None

    with patch(
        "financetoolkit.sentiment_model.requests.get",
        return_value=mock_response,
    ) as mock_get:
        result = sentiment_model.get_sentiment(
            "AAPL",
            api_key="test-key",
            source=["reddit", "news"],
            start_date="2024-01-01",
            end_date="2024-01-01",
        )

    assert list(result.columns) == [("news", "AAPL"), ("reddit", "AAPL")]
    assert mock_get.call_count == EXPECTED_SOURCE_CALLS


def test_get_sentiment_supports_all_sources():
    """The all source shortcut should call every supported Adanos stock source."""
    mock_response = Mock()
    mock_response.json.return_value = {"stocks": []}
    mock_response.raise_for_status.return_value = None

    with patch(
        "financetoolkit.sentiment_model.requests.get",
        return_value=mock_response,
    ) as mock_get:
        sentiment_model.get_sentiment(
            "AAPL",
            api_key="test-key",
            source="all",
            start_date="2024-01-01",
            end_date="2024-01-01",
        )

    assert mock_get.call_count == len(sentiment_model.ADANOS_SOURCE_ORDER)


def test_get_sentiment_preserves_zero_values_from_dated_payloads():
    """A zero sentiment value is valid and should not be dropped."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "stocks": [
            {
                "ticker": "AAPL",
                "trend_history": [{"date": "2024-01-01", "sentiment": 0.0}],
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        "financetoolkit.sentiment_model.requests.get",
        return_value=mock_response,
    ):
        result = sentiment_model.get_sentiment(
            "AAPL",
            api_key="test-key",
            source="reddit",
            start_date="2024-01-01",
            end_date="2024-01-01",
        )

    assert result.loc[pd.Period("2024-01-01", freq="D"), ("reddit", "AAPL")] == 0.0


def test_get_sentiment_rejects_invalid_source():
    """Only the documented Adanos stock sources should be accepted."""
    with pytest.raises(ValueError, match="Please select a valid Adanos source"):
        sentiment_model.get_sentiment(
            ["AAPL"],
            api_key="test-key",
            source="invalid",
            start_date="2024-01-01",
            end_date="2024-01-01",
        )


def test_get_sentiment_uses_environment_api_key(monkeypatch):
    """The model should use ADANOS_API_KEY when no key is passed directly."""
    mock_response = Mock()
    mock_response.json.return_value = {"stocks": []}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setenv("ADANOS_API_KEY", "env-key")

    with patch(
        "financetoolkit.sentiment_model.requests.get",
        return_value=mock_response,
    ) as mock_get:
        sentiment_model.get_sentiment(
            ["AAPL"],
            source="reddit",
            start_date="2024-01-01",
            end_date="2024-01-01",
        )

    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["X-API-Key"] == "env-key"


def test_get_sentiment_requires_api_key(monkeypatch):
    """The model should fail clearly when no Adanos API key is configured."""
    monkeypatch.delenv("ADANOS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Please provide an Adanos API key"):
        sentiment_model.get_sentiment(
            ["AAPL"],
            source="reddit",
            start_date="2024-01-01",
            end_date="2024-01-01",
        )
