"""External dataset helper tests."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from financetoolkit.utilities import external_dataset_model


def test_get_adanos_sentiment_returns_period_dataframe():
    """Adanos trend_history should become a daily PeriodIndex DataFrame."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "stocks": [
            {"ticker": "AAPL", "trend_history": [41.0, 43.5, 46.2]},
            {"ticker": "MSFT", "trend_history": [55.1, 53.4, 54.0]},
        ]
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        "financetoolkit.utilities.external_dataset_model.requests.get",
        return_value=mock_response,
    ) as mock_get:
        result = external_dataset_model.get_adanos_sentiment(
            ["AAPL", "MSFT"],
            api_key="test-key",
            source="reddit",
            days=3,
        )

    assert isinstance(result.index, pd.PeriodIndex)
    assert result.index.freqstr == "D"
    assert list(result.columns) == ["AAPL", "MSFT"]
    assert result.iloc[-1]["AAPL"] == pytest.approx(46.2)
    assert result.iloc[-1]["MSFT"] == pytest.approx(54.0)

    _, kwargs = mock_get.call_args
    assert kwargs["params"] == {"tickers": "AAPL,MSFT", "days": 3}
    assert kwargs["headers"]["X-API-Key"] == "test-key"


def test_get_adanos_sentiment_rejects_invalid_source():
    """Only the documented Adanos stock sources should be accepted."""
    with pytest.raises(
        ValueError, match="Please select a valid Adanos source"
    ):
        external_dataset_model.get_adanos_sentiment(
            ["AAPL"], api_key="test-key", source="invalid"
        )


def test_combine_external_dataset_adds_new_metric_level():
    """External factor data should join under its own metric group."""
    historical_data = pd.DataFrame(
        {
            ("Adj Close", "AAPL"): [100.0, 101.0],
            ("Adj Close", "MSFT"): [200.0, 201.0],
            ("Return", "AAPL"): [0.01, 0.02],
            ("Return", "MSFT"): [0.03, 0.04],
        },
        index=pd.period_range("2024-01-01", periods=2, freq="D"),
    )
    historical_data.columns = pd.MultiIndex.from_tuples(historical_data.columns)

    sentiment_data = pd.DataFrame(
        {"AAPL": [41.0, 42.0], "MSFT": [55.0, 54.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    result = external_dataset_model.combine_external_dataset(
        historical_data,
        sentiment_data,
        dataset_name="Adanos Sentiment",
    )

    assert ("Adanos Sentiment", "AAPL") in result.columns
    assert ("Adanos Sentiment", "MSFT") in result.columns
    assert result.loc[pd.Period("2024-01-02", freq="D"), ("Adanos Sentiment", "AAPL")] == pytest.approx(42.0)


def test_combine_external_dataset_requires_financetoolkit_history_shape():
    """A flat-column dataset should fail with a clear error."""
    with pytest.raises(
        ValueError,
        match="The historical dataset should contain MultiIndex columns",
    ):
        external_dataset_model.combine_external_dataset(
            pd.DataFrame({"AAPL": [100.0]}),
            pd.DataFrame({"AAPL": [42.0]}),
            dataset_name="Adanos Sentiment",
        )
