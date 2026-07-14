"""Toolkit Sentiment Tests"""

from unittest.mock import patch

import pandas as pd

from financetoolkit import Toolkit

EXPECTED_ROUNDED_SENTIMENT = 41.23


def test_toolkit_get_sentiment_uses_tickers_and_dates():
    """Toolkit.get_sentiment should delegate its configured tickers and dates."""
    expected_sentiment = pd.DataFrame(
        {("reddit", "AAPL"): [41.23456], ("reddit", "MSFT"): [55.98765]},
        index=pd.period_range("2024-01-01", periods=1, freq="D"),
    )
    expected_sentiment.columns = pd.MultiIndex.from_tuples(expected_sentiment.columns, names=["Source", "Ticker"])

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-01-01",
        sleep_timer=False,
    )

    with patch(
        "financetoolkit.toolkit_controller.sentiment_model.get_sentiment",
        return_value=expected_sentiment,
    ) as mock_get_sentiment:
        result = toolkit.get_sentiment(
            adanos_api_key="test-key",
            source=["reddit"],
            rounding=2,
        )

    mock_get_sentiment.assert_called_once_with(
        tickers=["AAPL", "MSFT"],
        api_key="test-key",
        source=["reddit"],
        start_date="2024-01-01",
        end_date="2024-01-01",
        base_url="https://api.adanos.org",
        timeout=30,
    )
    assert result.loc[pd.Period("2024-01-01", freq="D"), ("reddit", "AAPL")] == EXPECTED_ROUNDED_SENTIMENT
