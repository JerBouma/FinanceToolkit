"""Yahoo Finance Model Tests"""

# pylint: disable=missing-function-docstring

import time

import pytest
import requests

from financetoolkit import yfinance_model


class FakeResponse:
    """A Yahoo Finance chart response holding only the metadata."""

    status_code = 200

    def __init__(self, meta: dict):
        self.meta = meta

    def json(self) -> dict:
        return {"chart": {"result": [{"meta": self.meta}]}}


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(yfinance_model, "get_active_cache", lambda: None)


@pytest.mark.parametrize("timezone", ["Asia/Tokyo", "America/Los_Angeles", "UTC"])
def test_statistics_dates_are_the_exchange_dates_on_any_machine(monkeypatch, timezone):
    # 21:00 in New York on 14 November 2023 is already 15 November in Tokyo.
    meta = {
        "currency": "USD",
        "firstTradeDate": 1700010000,
        "regularMarketTime": 1700010000,
        "gmtoffset": -18000,
    }
    monkeypatch.setattr(
        yfinance_model, "get_request", lambda *_, **__: FakeResponse(meta)
    )
    monkeypatch.setenv("TZ", timezone)
    time.tzset()

    try:
        statistics = yfinance_model.get_historical_statistics("AAPL")
    finally:
        monkeypatch.undo()
        time.tzset()

    assert statistics["First Trade Date"] == "2023-11-14"
    assert statistics["Regular Market Time"] == "2023-11-14"


def test_statistics_of_a_ticker_that_times_out_are_empty(monkeypatch):
    def timeout(*_, **__):
        raise requests.exceptions.ReadTimeout("Read timed out.")

    monkeypatch.setattr(yfinance_model, "get_request", timeout)

    assert yfinance_model.get_historical_statistics("AAPL").empty
