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


def test_a_curl_timeout_from_yfinance_is_no_data(monkeypatch):
    """yfinance's curl_cffi transport raises its own errors, which are OSErrors."""
    from curl_cffi.requests.exceptions import Timeout

    class SlowTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_balance_sheet(self, freq):  # noqa: ARG002
            raise Timeout("Operation timed out")

    monkeypatch.setattr(yfinance_model.yf, "Ticker", SlowTicker)

    assert yfinance_model.get_financial_statement("AAPL", statement="balance").empty


def test_a_currency_pair_is_requested_as_yahoo_finance_writes_it(monkeypatch):
    requested = []

    class FakeTicker:
        def __init__(self, ticker):
            requested.append(ticker)

        def history(self, **kwargs):  # noqa: ARG002
            raise OSError("offline")

    monkeypatch.setattr(yfinance_model.yf, "Ticker", FakeTicker)

    yfinance_model.get_historical_data("EURUSD", start="2025-01-01", end="2025-01-31")

    assert requested == ["EURUSD=X"]


def apple_balance_sheet():
    """A Yahoo Finance balance sheet with Apple's September fiscal year ends."""
    import pandas as pd

    return pd.DataFrame(
        {
            pd.Timestamp("2023-09-30"): [352583e6, 29965e6],
            pd.Timestamp("2022-09-24"): [352755e6, 23646e6],
            pd.Timestamp("2021-09-25"): [351002e6, 34940e6],
        },
        index=["TotalAssets", "CashAndCashEquivalents"],
    )


@pytest.mark.parametrize("quarter", [False, True])
def test_the_yahoo_period_end_dates_are_the_fiscal_period_ends(monkeypatch, quarter):
    import pandas as pd

    from financetoolkit import normalization_model
    from financetoolkit.helpers import determine_period_end_dates

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_balance_sheet(self, freq):  # noqa: ARG002
            return apple_balance_sheet()

    monkeypatch.setattr(yfinance_model.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yfinance_model, "get_reported_currency", lambda _: "USD")

    statement = yfinance_model.get_financial_statement(
        "AAPL", statement="balance", quarter=quarter
    )

    # Labelled with calendar periods, while the row keeps the dates Yahoo Finance reported.
    expected_labels = (
        ["2023Q3", "2022Q3", "2021Q3"] if quarter else ["2023", "2022", "2021"]
    )
    assert [str(period) for period in statement.columns] == expected_labels
    assert statement.loc["periodEndDate"].tolist() == [
        "2023-09-30",
        "2022-09-24",
        "2021-09-25",
    ]

    statistics = yfinance_model.get_statistics_statement(
        "AAPL", periods=statement.columns, financial_statement=statement
    )

    assert statistics.loc["financialCurrency"].tolist() == ["USD"] * 3

    # Normalized as the Toolkit normalizes the Yahoo Finance statistics.
    statistics_format = normalization_model.read_normalization_file("statistics_yf")
    normalized = normalization_model.convert_financial_statements(
        financial_statements=pd.concat({"AAPL": statistics}, axis=0),
        statement_format=statistics_format,
        adjust_financial_statements=False,
        reverse_dates=True,
    )

    period_end_dates = determine_period_end_dates(
        periods=normalized.columns,
        tickers=["AAPL"],
        quarter=quarter,
        statistics_statement=normalized,
    )

    assert period_end_dates.loc["AAPL"].dt.strftime("%Y-%m-%d").tolist() == [
        "2021-09-25",
        "2022-09-24",
        "2023-09-30",
    ]


def test_a_yahoo_statement_without_period_end_dates_leaves_the_statistic_out(
    monkeypatch,
):
    import pandas as pd

    monkeypatch.setattr(yfinance_model, "get_reported_currency", lambda _: "USD")
    periods = pd.PeriodIndex(["2023", "2022"], freq="Y")

    statistics = yfinance_model.get_statistics_statement(
        "AAPL",
        periods=periods,
        financial_statement=pd.DataFrame(
            {period: [1.0] for period in periods}, index=["TotalAssets"]
        ),
    )

    assert statistics.index.tolist() == ["financialCurrency"]

    # Without a reporting currency the period end dates still make a statistics statement.
    monkeypatch.setattr(yfinance_model, "get_reported_currency", lambda _: "")
    statement = pd.DataFrame(
        {periods[0]: ["2023-09-30"], periods[1]: ["2022-09-24"]},
        index=["periodEndDate"],
    )

    statistics = yfinance_model.get_statistics_statement(
        "AAPL", periods=periods, financial_statement=statement
    )

    assert statistics.loc["periodEndDate"].tolist() == ["2023-09-30", "2022-09-24"]
