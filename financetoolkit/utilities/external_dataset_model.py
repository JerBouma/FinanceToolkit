"""Helpers for integrating optional external factor datasets."""

__docformat__ = "google"

from collections.abc import Sequence

import pandas as pd
import requests

ADANOS_SOURCES = {"reddit", "x", "news", "polymarket"}


def _ensure_period_index(dataset: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return a DataFrame indexed by a daily PeriodIndex."""
    frame = dataset.to_frame() if isinstance(dataset, pd.Series) else dataset.copy()

    if not isinstance(frame.index, pd.PeriodIndex):
        frame.index = pd.to_datetime(frame.index).to_period("D")
    elif frame.index.freqstr != "D":
        frame.index = frame.index.asfreq("D")

    return frame.sort_index()


def get_adanos_sentiment(
    tickers: str | Sequence[str],
    api_key: str,
    source: str = "reddit",
    days: int = 30,
    base_url: str = "https://api.adanos.org",
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Retrieve daily Adanos buzz history for one or more stock tickers.

    The Adanos compare endpoints return `trend_history` arrays ordered from the
    oldest to the newest day. This helper converts those arrays into a DataFrame
    indexed by daily periods so the result can be joined with FinanceToolkit
    historical datasets or used as an external factor series.

    Args:
        tickers (str | Sequence[str]): One ticker or a sequence of tickers.
        api_key (str): Adanos API key sent via the ``X-API-Key`` header.
        source (str): One of ``reddit``, ``x``, ``news``, or ``polymarket``.
        days (int): Number of days to request from the compare endpoint.
        base_url (str): Base Adanos API URL.
        timeout (int): Requests timeout in seconds.

    Returns:
        pd.DataFrame: Daily buzz history indexed by PeriodIndex("D"), with one
            column per ticker.
    """
    if source not in ADANOS_SOURCES:
        raise ValueError(
            "Please select a valid Adanos source: reddit, x, news or polymarket."
        )

    ticker_list = (
        [tickers]
        if isinstance(tickers, str)
        else [ticker for ticker in tickers if ticker]
    )

    if not ticker_list:
        raise ValueError("Please provide at least one ticker.")

    response = requests.get(
        f"{base_url.rstrip('/')}/{source}/stocks/v1/compare",
        params={"tickers": ",".join(ticker_list), "days": days},
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    stocks = payload.get("stocks", [])

    histories: dict[str, list[float]] = {}

    for stock in stocks:
        ticker = stock.get("ticker")
        trend_history = stock.get("trend_history")

        if ticker and isinstance(trend_history, list) and trend_history:
            histories[ticker] = [float(value) for value in trend_history]

    if not histories:
        return pd.DataFrame()

    max_history_length = max(len(history) for history in histories.values())
    end_period = pd.Timestamp.now().normalize().to_period("D")
    index = pd.period_range(end=end_period, periods=max_history_length, freq="D")

    sentiment_dataset = pd.DataFrame(index=index)

    for ticker, history in histories.items():
        aligned_series = pd.Series(history, index=index[-len(history) :], dtype="float64")
        sentiment_dataset[ticker] = aligned_series.reindex(index)

    return sentiment_dataset


def combine_external_dataset(
    historical_data: pd.DataFrame,
    external_dataset: pd.DataFrame | pd.Series,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Add an external factor dataset to a FinanceToolkit historical dataset.

    Args:
        historical_data (pd.DataFrame): FinanceToolkit historical dataset with a
            MultiIndex column layout of ``(metric, ticker)``.
        external_dataset (pd.DataFrame | pd.Series): External data indexed by
            dates or daily periods, with columns named as tickers.
        dataset_name (str): Column-group name used for the external factor.

    Returns:
        pd.DataFrame: Historical dataset enriched with the external factor.
    """
    if not isinstance(historical_data.columns, pd.MultiIndex):
        raise ValueError(
            "The historical dataset should contain MultiIndex columns in the FinanceToolkit format."
        )

    external_frame = _ensure_period_index(external_dataset)
    historical_frame = historical_data.copy()

    if not isinstance(historical_frame.index, pd.PeriodIndex):
        historical_frame.index = pd.to_datetime(historical_frame.index).to_period("D")

    external_frame.columns = pd.MultiIndex.from_product(
        [[dataset_name], external_frame.columns]
    )

    combined_dataset = historical_frame.join(external_frame, how="left")

    return combined_dataset.sort_index(axis=1)
