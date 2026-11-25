"""Yahoo Finance Model Tests"""

import logging
import time

import numpy as np
import pandas as pd
import pytest

from financetoolkit import yfinance_model
from financetoolkit.utilities import error_model


def metadata_ticker(meta: dict | Exception):
    """A stand-in for yfinance.Ticker whose chart metadata is the given dict or raises the given error."""

    class MetadataTicker:
        requested: list[str] = []

        def __init__(self, ticker):
            MetadataTicker.requested.append(ticker)

        def get_history_metadata(self, repair):
            assert repair is False
            if isinstance(meta, Exception):
                raise meta
            return dict(meta)

    return MetadataTicker


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
    monkeypatch.setattr(yfinance_model.yf, "Ticker", metadata_ticker(meta))
    monkeypatch.setenv("TZ", timezone)
    time.tzset()

    try:
        statistics = yfinance_model.get_historical_statistics("AAPL")
    finally:
        monkeypatch.undo()
        time.tzset()

    assert statistics["First Trade Date"] == "2023-11-14"
    assert statistics["Regular Market Time"] == "2023-11-14"


class WindowsDatetime(yfinance_model.datetime):
    """A datetime that, as on Windows, cannot convert a timestamp before 1970."""

    @classmethod
    def fromtimestamp(cls, timestamp, tz=None):
        if timestamp < 0:
            raise OSError(22, "Invalid argument")
        return super().fromtimestamp(timestamp, tz=tz)


def test_statistics_of_a_stock_first_traded_before_1970(monkeypatch):
    # KO's first trade date as Yahoo Finance reports it: 09:30 in New York on 2 January 1962.
    meta = {
        "currency": "USD",
        "firstTradeDate": -252322200,
        "regularMarketTime": 1700010000,
        "gmtoffset": -18000,
    }
    monkeypatch.setattr(yfinance_model.yf, "Ticker", metadata_ticker(meta))
    monkeypatch.setattr(yfinance_model, "datetime", WindowsDatetime)

    statistics = yfinance_model.get_historical_statistics("KO")

    assert statistics["First Trade Date"] == "1962-01-02"
    assert statistics["Regular Market Time"] == "2023-11-14"


def test_statistics_of_a_ticker_that_times_out_are_empty(monkeypatch):
    from curl_cffi.requests.exceptions import Timeout

    monkeypatch.setattr(
        yfinance_model.yf, "Ticker", metadata_ticker(Timeout("Operation timed out"))
    )
    failures = error_model.get_request_failures()

    assert yfinance_model.get_historical_statistics("AAPL").empty
    assert error_model.get_request_failures() == failures + 1


def test_statistics_keep_their_labels_and_types(monkeypatch):
    # The metadata of AAPL as Yahoo Finance reports it, with fields the statistics leave out.
    meta = {
        "currency": "USD",
        "symbol": "AAPL",
        "exchangeName": "NMS",
        "fullExchangeName": "NasdaqGS",
        "instrumentType": "EQUITY",
        "firstTradeDate": 345479400,
        "regularMarketTime": 1795554001,
        "hasPrePostMarketData": True,
        "gmtoffset": -18000,
        "timezone": "EST",
        "exchangeTimezoneName": "America/New_York",
        "regularMarketPrice": 276.97,
    }
    monkeypatch.setattr(yfinance_model.yf, "Ticker", metadata_ticker(meta))

    statistics = yfinance_model.get_historical_statistics("AAPL")

    assert statistics.to_dict() == {
        "Currency": "USD",
        "Symbol": "AAPL",
        "Exchange Name": "NMS",
        "Instrument Type": "EQUITY",
        "First Trade Date": "1980-12-12",
        "Regular Market Time": "2026-11-24",
        "GMT Offset": -18000,
        "Timezone": "EST",
        "Exchange Timezone Name": "America/New_York",
    }
    assert statistics.dtype == object


def test_statistics_of_a_currency_pair_are_requested_as_yahoo_finance_writes_it(
    monkeypatch,
):
    ticker = metadata_ticker({"currency": "USD", "symbol": "EURUSD=X"})
    monkeypatch.setattr(yfinance_model.yf, "Ticker", ticker)

    yfinance_model.get_historical_statistics("EURUSD")

    assert ticker.requested == ["EURUSD=X"]


@pytest.mark.parametrize(
    "error",
    [
        # yfinance's answer to a symbol Yahoo Finance does not know, such as "Portfolio".
        KeyError("exchangeTimezoneName"),
        yfinance_model.yf.exceptions.YFRateLimitError(),
    ],
    ids=["unknown symbol", "rate limit"],
)
def test_statistics_yahoo_finance_cannot_give_are_a_failed_request(monkeypatch, error):
    monkeypatch.setattr(yfinance_model.yf, "Ticker", metadata_ticker(error))
    failures = error_model.get_request_failures()

    assert yfinance_model.get_historical_statistics("Portfolio").empty
    assert error_model.get_request_failures() == failures + 1


def test_statistics_with_a_broken_cache_database_are_a_failed_request(
    monkeypatch, caplog
):
    import peewee

    monkeypatch.setattr(
        yfinance_model.yf,
        "Ticker",
        metadata_ticker(peewee.OperationalError("no such table: _kv")),
    )
    failures = error_model.get_request_failures()

    with caplog.at_level("WARNING", logger="financetoolkit"):
        assert yfinance_model.get_historical_statistics("AAPL").empty

    assert error_model.get_request_failures() == failures + 1
    assert "no such table: _kv" in caplog.text


def test_statistics_mute_the_log_of_yfinance(monkeypatch, caplog):
    class LoggingTicker:
        def __init__(self, ticker):
            pass

        def get_history_metadata(self, repair):  # noqa: ARG002
            logging.getLogger("yfinance").error("$PORTFOLIO: possibly delisted")
            raise KeyError("exchangeTimezoneName")

    monkeypatch.setattr(yfinance_model.yf, "Ticker", LoggingTicker)

    with caplog.at_level("DEBUG"):
        assert yfinance_model.get_historical_statistics("Portfolio").empty

    assert "possibly delisted" not in caplog.text

    # Outside the call the log of yfinance is written again.
    with caplog.at_level("DEBUG"):
        logging.getLogger("yfinance").error("$XYZ: possibly delisted")
    assert "$XYZ: possibly delisted" in caplog.text


def test_statistics_are_cached(monkeypatch):
    stored = {}

    class Cache:
        def get(self, **kwargs):
            return stored.get(kwargs["entity"])

        def set(self, **kwargs):
            stored[kwargs["entity"]] = kwargs["data"]

    ticker = metadata_ticker({"currency": "EUR", "symbol": "ASML.AS"})
    monkeypatch.setattr(yfinance_model, "get_active_cache", Cache)
    monkeypatch.setattr(yfinance_model.yf, "Ticker", ticker)

    first = yfinance_model.get_historical_statistics("ASML.AS")
    second = yfinance_model.get_historical_statistics("ASML.AS")

    assert first.equals(second)
    assert ticker.requested == ["ASML.AS"]


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
    monkeypatch.setattr(
        yfinance_model,
        "get_company_information",
        lambda _: {"financialCurrency": "USD", "lastFiscalYearEnd": ""},
    )

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

    monkeypatch.setattr(
        yfinance_model,
        "get_company_information",
        lambda _: {"financialCurrency": "USD", "lastFiscalYearEnd": ""},
    )
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
    monkeypatch.setattr(
        yfinance_model,
        "get_company_information",
        lambda _: {"financialCurrency": "", "lastFiscalYearEnd": ""},
    )
    statement = pd.DataFrame(
        {periods[0]: ["2023-09-30"], periods[1]: ["2022-09-24"]},
        index=["periodEndDate"],
    )

    statistics = yfinance_model.get_statistics_statement(
        "AAPL", periods=periods, financial_statement=statement
    )

    assert statistics.loc["periodEndDate"].tolist() == ["2023-09-30", "2022-09-24"]


@pytest.mark.parametrize("meta", [{}, {"chart": None, "gmtoffset": "x"}])
def test_statistics_from_an_unexpected_answer_are_empty(monkeypatch, meta):
    monkeypatch.setattr(yfinance_model.yf, "Ticker", metadata_ticker(meta))

    assert yfinance_model.get_historical_statistics("AAPL").empty


def test_prices_come_unrepaired_without_the_repair_extra(monkeypatch):
    import pandas as pd

    repairs = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, repair, **kwargs):  # noqa: ARG002
            repairs.append(repair)
            if repair:
                # Recent yfinance versions import scikit-learn for the repair.
                raise ModuleNotFoundError("No module named 'sklearn'")
            return pd.DataFrame(
                {
                    "Open": 1.0,
                    "High": 1.0,
                    "Low": 1.0,
                    "Close": [1.0, 2.0],
                    "Volume": 1,
                },
                index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
            )

    monkeypatch.setattr(yfinance_model.yf, "Ticker", FakeTicker)

    data = yfinance_model.get_historical_data(
        "ASML.AS", start="2025-01-02", end="2025-01-03"
    )

    assert repairs == [True, False]
    assert not data.empty


def test_a_52_53_week_company_is_labelled_and_dated_as_from_fmp(monkeypatch):
    import pandas as pd

    from financetoolkit import normalization_model
    from financetoolkit.helpers import (
        convert_period_end_dates_to_calendar_periods,
        determine_availability_dates,
        determine_period_end_dates,
    )

    # Yahoo Finance reports Home Depot's quarters ending 4 May, 3 August, 2 November 2025
    # and 1 February 2026 at the end of the month before.
    yahoo_dates = ["2026-01-31", "2025-10-31", "2025-07-31", "2025-04-30"]
    fmp_dates = ["2026-02-01", "2025-11-02", "2025-08-03", "2025-05-04"]

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_income_stmt(self, freq):  # noqa: ARG002
            return pd.DataFrame(
                {pd.Timestamp(date): [1.0] for date in yahoo_dates},
                index=["TotalRevenue"],
            )

    monkeypatch.setattr(yfinance_model.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(
        yfinance_model,
        "get_company_information",
        lambda _: {"financialCurrency": "USD", "lastFiscalYearEnd": "2026-02-01"},
    )

    statement = yfinance_model.get_financial_statement(
        "HD", statement="income", quarter=True
    )
    fmp_labels = convert_period_end_dates_to_calendar_periods(
        pd.to_datetime(fmp_dates), quarter=True
    )

    assert list(statement.columns) == list(fmp_labels)
    assert [str(period) for period in statement.columns] == [
        "2025Q4",
        "2025Q3",
        "2025Q2",
        "2025Q1",
    ]

    statistics = yfinance_model.get_statistics_statement(
        "HD", periods=statement.columns, financial_statement=statement
    )

    # The quarter ending in the month the fiscal year ends closes the fiscal year.
    assert statistics.loc["period"].tolist() == ["Q4", "Q3", "Q2", "Q1"]

    normalized = normalization_model.convert_financial_statements(
        financial_statements=pd.concat({"HD": statistics}, axis=0),
        statement_format=normalization_model.read_normalization_file("statistics_yf"),
        adjust_financial_statements=False,
    )
    period_ends = determine_period_end_dates(
        periods=normalized.columns,
        tickers=["HD"],
        quarter=True,
        statistics_statement=normalized,
    )
    availability = determine_availability_dates(
        period_ends, statistics_statement=normalized, quarter=True
    )

    # The fourth quarter gets the annual reporting deadline, as with FMP's Q4.
    assert (availability - period_ends).loc["HD"].dt.days.tolist() == [90, 90, 90, 120]


def test_a_yearly_yahoo_statement_is_a_full_fiscal_year(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        yfinance_model,
        "get_company_information",
        lambda _: {"financialCurrency": "USD", "lastFiscalYearEnd": "2025-09-27"},
    )
    periods = pd.PeriodIndex(["2025", "2024"], freq="Y")
    statement = pd.DataFrame(
        {periods[0]: ["2025-09-30"], periods[1]: ["2024-09-30"]},
        index=["periodEndDate"],
    )

    statistics = yfinance_model.get_statistics_statement(
        "AAPL", periods=periods, financial_statement=statement
    )

    assert statistics.loc["period"].tolist() == ["FY", "FY"]


def test_futures_without_daily_bars_are_built_from_hourly_bars(monkeypatch):
    import pandas as pd

    intervals = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, interval, **kwargs):  # noqa: ARG002
            intervals.append(interval)
            if interval == "1d":
                # Only the latest bar, as Yahoo Finance answers for TTF=F.
                return pd.DataFrame(
                    {
                        "Open": 87.0,
                        "High": 87.0,
                        "Low": 81.0,
                        "Close": [82.33],
                        "Volume": 1,
                    },
                    index=pd.DatetimeIndex(["2026-11-09"], tz="America/New_York"),
                )
            return pd.DataFrame(
                {
                    "Open": [31.0, 31.5, 32.0, 43.0, 44.0],
                    "High": [31.6, 32.1, 32.2, 44.0, 45.0],
                    "Low": [30.9, 31.4, 31.9, 42.9, 43.5],
                    "Close": [31.5, 32.0, 31.975, 44.0, 43.45],
                    "Volume": [10, 20, 30, 40, 50],
                },
                index=pd.DatetimeIndex(
                    [
                        "2026-02-27 03:00",
                        "2026-02-27 09:00",
                        "2026-02-27 11:00",
                        "2026-03-02 03:00",
                        "2026-03-02 11:00",
                    ],
                    tz="America/New_York",
                ),
            )

    monkeypatch.setattr(yfinance_model.yf, "Ticker", FakeTicker)

    data = yfinance_model.get_historical_data(
        "TTF=F", start="2026-02-27", end="2026-03-02"
    )

    assert intervals == ["1d", "1h"]
    # A bar per exchange day: the first open, the highest high, the lowest low, the last
    # trade as the close and the volume summed.
    assert list(data.index.astype(str)) == ["2026-02-27", "2026-03-02"]
    assert list(data["Close"]) == [31.975, 43.45]
    assert list(data["Open"]) == [31.0, 43.0]
    assert list(data["High"]) == [32.2, 45.0]
    assert list(data["Low"]) == [30.9, 42.9]
    assert list(data["Volume"]) == [60, 90]


@pytest.mark.parametrize("locks", [1, 2])
def test_a_locked_cache_database_is_retried_once_then_a_failed_download(
    monkeypatch, locks
):
    """
    yfinance's SQLite caches are shared by the download threads, so one of them can find a
    database locked by another: the download is tried once more and a second lock counts as
    a failed download rather than an error that ends the calculation.
    """
    import sqlite3

    import pandas as pd
    import peewee

    attempts = []

    class LockedTicker:
        def __init__(self, ticker):
            pass

        def history(self, **kwargs):  # noqa: ARG002
            attempts.append(1)
            if len(attempts) <= locks:
                error = (
                    peewee.OperationalError if locks == 1 else sqlite3.OperationalError
                )
                raise error("database is locked")
            return pd.DataFrame(
                {
                    "Open": 1.0,
                    "High": 1.0,
                    "Low": 1.0,
                    "Close": [1.0, 2.0],
                    "Volume": 1,
                },
                index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
            )

    monkeypatch.setattr(yfinance_model.yf, "Ticker", LockedTicker)

    data = yfinance_model.get_historical_data(
        "ASML.AS", start="2025-01-02", end="2025-01-03"
    )

    assert len(attempts) == 2
    assert data.empty == (locks == 2)


def test_a_locked_cache_database_leaves_a_statement_empty(monkeypatch):
    import peewee

    class LockedTicker:
        def __init__(self, ticker):
            pass

        def get_balance_sheet(self, freq):  # noqa: ARG002
            raise peewee.OperationalError("database is locked")

    monkeypatch.setattr(yfinance_model.yf, "Ticker", LockedTicker)

    assert yfinance_model.get_financial_statement("AAPL", statement="balance").empty


@pytest.mark.parametrize(
    ("message", "retried"),
    [
        ("database is busy", True),
        ("no such table: _kv", False),
        ("attempt to write a readonly database", False),
        ("disk I/O error", False),
    ],
)
def test_only_a_locked_cache_database_is_retried(monkeypatch, caplog, message, retried):
    """
    A lock or busy database is passing and tried once more; a missing table, a read-only file
    or a disk error is lasting, so it is reported with yfinance's own text and not retried.
    """
    import peewee

    attempts = []

    class BrokenTicker:
        def __init__(self, ticker):
            pass

        def history(self, **kwargs):  # noqa: ARG002
            attempts.append(1)
            raise peewee.OperationalError(message)

    monkeypatch.setattr(yfinance_model.yf, "Ticker", BrokenTicker)

    with caplog.at_level("WARNING", logger="financetoolkit"):
        data = yfinance_model.get_historical_data(
            "ASML.AS", start="2025-01-02", end="2025-01-03"
        )

    assert data.empty
    assert len(attempts) == (2 if retried else 1)
    assert ("yfinance's cache database could not be used" in caplog.text) != retried
    if not retried:
        assert message in caplog.text


def test_a_broken_cache_database_is_reported_once_per_call(monkeypatch, caplog):
    import peewee

    from financetoolkit import historical_model

    class BrokenTicker:
        def __init__(self, ticker):
            pass

        def history(self, **kwargs):  # noqa: ARG002
            raise peewee.OperationalError("no such table: _kv")

        def get_balance_sheet(self, freq):  # noqa: ARG002
            raise peewee.OperationalError("no such table: _kv")

    monkeypatch.setattr(yfinance_model.yf, "Ticker", BrokenTicker)

    with caplog.at_level("WARNING", logger="financetoolkit"):
        historical_model.get_historical_data(
            ["ASML.AS", "SAP.DE", "AIR.PA"],
            enforce_source="YahooFinance",
            start="2025-01-02",
            end="2025-01-03",
            sleep_timer=False,
        )

    assert caplog.text.count("yfinance's cache database could not be used") == 1

    # A statement reports the same fault.
    caplog.clear()
    with caplog.at_level("WARNING", logger="financetoolkit"):
        assert yfinance_model.get_financial_statement("AAPL", statement="balance").empty
    assert "no such table: _kv" in caplog.text


def search_returning(quotes: list | Exception):
    """A stand-in for yfinance.Search whose quotes are the given list or which raises the given error."""

    class Search:
        queries: list[dict] = []

        def __init__(self, query, **kwargs):
            Search.queries.append({"query": query, **kwargs})
            if isinstance(quotes, Exception):
                raise quotes
            self.quotes = quotes

    return Search


def test_an_isin_is_converted_to_the_ticker_of_the_best_match(monkeypatch, caplog):
    search = search_returning([{"symbol": "AAPL", "exchange": "NMS"}])
    monkeypatch.setattr(yfinance_model.yf, "Search", search)

    with caplog.at_level("INFO", logger="financetoolkit"):
        assert yfinance_model.convert_isin_to_ticker("US0378331005") == "AAPL"

    assert search.queries[0]["query"] == "US0378331005"
    assert search.queries[0]["max_results"] == 1
    assert "Converted ISIN US0378331005 to ticker AAPL" in caplog.text


def test_a_ticker_is_not_searched_as_an_isin(monkeypatch):
    search = search_returning(AssertionError("A ticker is not searched."))
    monkeypatch.setattr(yfinance_model.yf, "Search", search)

    assert yfinance_model.convert_isin_to_ticker("AAPL") == "AAPL"
    assert yfinance_model.convert_isin_to_ticker("INVALID") == "INVALID"


def test_an_isin_without_a_match_is_returned_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(yfinance_model.yf, "Search", search_returning([]))

    with caplog.at_level("WARNING", logger="financetoolkit"):
        assert yfinance_model.convert_isin_to_ticker("XS0000000000") == "XS0000000000"

    assert "Could not find a ticker for ISIN XS0000000000" in caplog.text


@pytest.mark.parametrize(
    ("error", "message", "failed_request"),
    [
        (OSError("Could not resolve host"), "Request failed for ISIN", True),
        (
            yfinance_model.yf.exceptions.YFRateLimitError(),
            "Request failed for ISIN",
            True,
        ),
        (KeyError("symbol"), "Could not parse response for ISIN", False),
    ],
    ids=["unreachable", "rate limit", "unexpected answer"],
)
def test_an_isin_that_cannot_be_searched_is_returned_with_a_warning(
    monkeypatch, caplog, error, message, failed_request
):
    monkeypatch.setattr(yfinance_model.yf, "Search", search_returning(error))
    failures = error_model.get_request_failures()

    with caplog.at_level("WARNING", logger="financetoolkit"):
        assert yfinance_model.convert_isin_to_ticker("US0378331005") == "US0378331005"

    assert message in caplog.text
    assert error_model.get_request_failures() == failures + failed_request


def test_tickers_given_as_isins_are_converted_when_validated(monkeypatch):
    from financetoolkit.utilities import validation_model

    monkeypatch.setattr(
        yfinance_model.yf, "Search", search_returning([{"symbol": "ASML.AS"}])
    )

    tickers = validation_model.validate_toolkit_parameters(
        tickers=["NL0010273215", "AAPL"],
        start_date=None,
        end_date=None,
        risk_free_rate="10y",
        enforce_source=None,
        api_key="",
        intraday_period=None,
        benchmark_ticker=None,
    )

    assert tickers == ["ASML.AS", "AAPL"]


class OptionsTicker:
    """A stand-in for yfinance.Ticker with the options of one expiry date."""

    def __init__(self, ticker):
        self.ticker = ticker

    @property
    def options(self):
        return ("2026-12-18",)

    def option_chain(self, expiration_date):
        chain = pd.DataFrame({"contractSymbol": ["X"], "strike": [100.0]})
        return type(
            "Chain",
            (),
            {
                "calls": chain,
                "puts": chain.assign(strike=90.0),
                "underlying": {"regularMarketPrice": 101.5},
            },
        )()


def test_option_expiry_dates_and_chains_come_from_yahoo_finance(monkeypatch):
    monkeypatch.setattr(yfinance_model.yf, "Ticker", OptionsTicker)

    assert yfinance_model.get_option_expiry_dates("AAPL") == ["2026-12-18"]

    calls, underlying_price = yfinance_model.get_option_chain("AAPL", "2026-12-18")
    puts, _ = yfinance_model.get_option_chain("AAPL", "2026-12-18", put_option=True)

    assert calls["strike"].tolist() == [100.0]
    assert puts["strike"].tolist() == [90.0]
    assert underlying_price == 101.5


@pytest.mark.parametrize(
    ("error", "failed_request"),
    [
        (OSError("Could not resolve host"), True),
        (ValueError("Expiration `2026-12-18` cannot be found."), False),
    ],
    ids=["unreachable", "no such expiry"],
)
def test_options_yahoo_finance_cannot_give_are_left_out(
    monkeypatch, caplog, error, failed_request
):
    class FailingTicker:
        def __init__(self, ticker):
            pass

        @property
        def options(self):
            raise error

        def option_chain(self, expiration_date):  # noqa: ARG002
            raise error

    monkeypatch.setattr(yfinance_model.yf, "Ticker", FailingTicker)
    failures = error_model.get_request_failures()

    with caplog.at_level("INFO", logger="financetoolkit"):
        assert yfinance_model.get_option_expiry_dates("AAPL") == []
        chain, underlying_price = yfinance_model.get_option_chain("AAPL", "2026-12-18")

    assert chain.empty
    assert np.isnan(underlying_price)
    assert "Yahoo Finance has no options of AAPL" in caplog.text
    assert (
        "No options data is available for AAPL expiring on 2026-12-18, so it is left out"
        in caplog.text
    )
    assert error_model.get_request_failures() == failures + 2 * failed_request


def test_dividends_come_as_daily_periods(monkeypatch):
    dividends = pd.Series(
        [0.2, 0.25],
        index=pd.DatetimeIndex(["2023-05-01", "2023-08-01"], tz="America/New_York"),
    )
    requested = []

    class DividendTicker:
        def __init__(self, ticker):
            requested.append(ticker)
            self.dividends = dividends

    monkeypatch.setattr(yfinance_model.yf, "Ticker", DividendTicker)

    result = yfinance_model.get_dividends("KO")

    assert requested == ["KO"]
    assert result.index.astype(str).tolist() == ["2023-05-01", "2023-08-01"]
    assert result.tolist() == [0.2, 0.25]


def test_dividends_yahoo_finance_cannot_give_are_empty(monkeypatch):
    class RateLimitedTicker:
        def __init__(self, ticker):
            pass

        @property
        def dividends(self):
            raise yfinance_model.yf.exceptions.YFRateLimitError()

    monkeypatch.setattr(yfinance_model.yf, "Ticker", RateLimitedTicker)
    failures = error_model.get_request_failures()

    assert yfinance_model.get_dividends("KO").empty
    assert error_model.get_request_failures() == failures + 1


def test_a_muted_call_does_not_mute_a_concurrent_one(caplog):
    import threading

    muted, written = threading.Event(), threading.Event()

    def muted_call():
        with yfinance_model.quiet_yfinance():
            logging.getLogger("yfinance").error("muted call")
            muted.set()
            written.wait(timeout=5)

    thread = threading.Thread(target=muted_call)

    with caplog.at_level("DEBUG"):
        thread.start()
        muted.wait(timeout=5)
        logging.getLogger("yfinance").error("concurrent call")
        written.set()
        thread.join()

    assert "concurrent call" in caplog.text
    assert "muted call" not in caplog.text
