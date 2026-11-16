"""Yahoo Finance Model Tests"""

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


@pytest.mark.parametrize(
    "payload",
    [
        ValueError("Expecting value"),
        {"chart": {"result": None}},
        {"chart": {"result": []}},
    ],
)
def test_statistics_from_an_unexpected_answer_are_empty(monkeypatch, payload):
    class Response:
        status_code = 200

        def json(self):
            # A consent page instead of JSON raises when it is read.
            if isinstance(payload, Exception):
                raise payload
            return payload

    monkeypatch.setattr(yfinance_model, "get_request", lambda *_, **__: Response())

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
