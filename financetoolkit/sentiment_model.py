"""Sentiment Model"""

__docformat__ = "google"

import os
from collections.abc import Sequence

import pandas as pd
import requests

ADANOS_SOURCE_ORDER = ["reddit", "x", "news", "polymarket"]
ADANOS_SOURCES = set(ADANOS_SOURCE_ORDER)


def _get_api_key(api_key: str | None) -> str:
    """Return the provided API key or the ADANOS_API_KEY environment variable."""
    adanos_api_key = api_key or os.getenv("ADANOS_API_KEY", "")

    if not adanos_api_key:
        raise ValueError(
            "Please provide an Adanos API key via the adanos_api_key parameter "
            "or the ADANOS_API_KEY environment variable."
        )

    return adanos_api_key


def _normalize_sources(source: str | Sequence[str]) -> list[str]:
    """Normalize and validate one or more Adanos source names."""
    sources = [source] if isinstance(source, str) else list(source)
    sources = [source_name.lower() for source_name in sources]

    if sources == ["all"]:
        return ADANOS_SOURCE_ORDER

    invalid_sources = [source_name for source_name in sources if source_name not in ADANOS_SOURCES]

    if invalid_sources:
        raise ValueError("Please select a valid Adanos source: reddit, x, news or polymarket.")

    if not sources:
        raise ValueError("Please provide at least one Adanos source.")

    return sources


def _extract_history(
    trend_history: list,
    date_index: pd.PeriodIndex,
) -> pd.Series:
    """Convert an Adanos trend history payload to a daily series."""
    if not trend_history:
        return pd.Series(index=date_index, dtype="float64")

    if all(isinstance(item, dict) for item in trend_history):
        values: dict[pd.Period, float] = {}

        for item in trend_history:
            date_value = item.get("date") or item.get("day") or item.get("period") or item.get("timestamp")
            sentiment_value = next(
                (
                    item[key]
                    for key in ["sentiment", "sentiment_score", "buzz_score", "value"]
                    if key in item and item[key] is not None
                ),
                None,
            )

            if date_value is not None and sentiment_value is not None:
                values[pd.Timestamp(date_value).to_period("D")] = float(sentiment_value)

        return pd.Series(values, dtype="float64").reindex(date_index)

    history = pd.Series([float(value) for value in trend_history], dtype="float64")
    aligned_index = date_index[-len(history) :]

    return pd.Series(history.to_numpy(), index=aligned_index).reindex(date_index)


def get_sentiment(
    tickers: str | Sequence[str],
    api_key: str | None = None,
    source: str | Sequence[str] = "reddit",
    start_date: str | None = None,
    end_date: str | None = None,
    base_url: str = "https://api.adanos.org",
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Retrieve daily Adanos market sentiment for one or more stock tickers.

    Args:
        tickers (str | Sequence[str]): One ticker or a sequence of tickers.
        api_key (str | None): Adanos API key. If omitted, ADANOS_API_KEY is used.
        source (str | Sequence[str]): One or more of reddit, x, news or polymarket.
        start_date (str | None): Start date formatted as YYYY-MM-DD.
        end_date (str | None): End date formatted as YYYY-MM-DD.
        base_url (str): Base Adanos API URL.
        timeout (int): Requests timeout in seconds.

    Returns:
        pd.DataFrame: Daily sentiment indexed by date with multi-index columns
            containing the source and ticker.
    """
    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
    ticker_list = [ticker for ticker in ticker_list if ticker]

    if not ticker_list:
        raise ValueError("Please provide at least one ticker.")

    end_period = pd.Timestamp(end_date).to_period("D") if end_date else pd.Timestamp.now().normalize().to_period("D")
    start_period = pd.Timestamp(start_date).to_period("D") if start_date else end_period - 29

    if start_period > end_period:
        raise ValueError("Please ensure the start date is before the end date.")
    date_index = pd.period_range(start=start_period, end=end_period, freq="D")

    sources = _normalize_sources(source)
    headers = {"X-API-Key": _get_api_key(api_key), "Accept": "application/json"}
    sentiment_data: dict[tuple[str, str], pd.Series] = {}

    for source_name in sources:
        response = requests.get(
            f"{base_url.rstrip('/')}/{source_name}/stocks/v1/compare",
            params={"tickers": ",".join(ticker_list), "days": len(date_index)},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        for stock in response.json().get("stocks", []):
            ticker = stock.get("ticker")
            trend_history = stock.get("trend_history", [])

            if ticker in ticker_list:
                sentiment_data[(source_name, ticker)] = _extract_history(
                    trend_history=trend_history,
                    date_index=date_index,
                )

    sentiment = pd.DataFrame(sentiment_data, index=date_index)

    if not sentiment.empty:
        sentiment.columns = pd.MultiIndex.from_tuples(sentiment.columns, names=["Source", "Ticker"])
        sentiment = sentiment.sort_index(axis=1)

    return sentiment
