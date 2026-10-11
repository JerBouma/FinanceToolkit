"""FinancialModelingPrep Endpoint Tests"""

# The company, fund, transcript and news endpoints added in 2.2.2. Every response is
# faked by replacing get_financial_data, so these run offline and check what the
# Finance Toolkit does with a response: the shaping, the date filtering, the conversion
# of percentages to decimals and the routing between endpoints.

import pandas as pd
import pytest
import requests

from financetoolkit import currencies_model, fmp_model, toolkit_controller
from financetoolkit.cache import cache_controller


@pytest.fixture(name="responses")
def fixture_responses(monkeypatch):
    """Serve canned responses by endpoint and record every requested url."""
    canned: dict[str, list[dict]] = {}
    requested: list[str] = []

    def fake_get_financial_data(
        url, sleep_timer=True, raw=False, user_subscription="Free"
    ):  # noqa: ARG001
        requested.append(url)
        endpoint = url.split("/stable/")[1].split("?")[0]
        records = canned.get(endpoint, [])

        return records if raw else pd.DataFrame(records)

    monkeypatch.setattr(fmp_model, "get_financial_data", fake_get_financial_data)

    return canned, requested


@pytest.fixture(name="cache_location")
def fixture_cache_location(tmp_path):
    """Point every Toolkit created in a test at an isolated cache database."""
    cache_controller.reset_cache_registry()

    yield str(tmp_path)

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()


def test_employee_count_keeps_the_latest_filing_per_period(responses):
    canned, requested = responses
    canned["historical-employee-count"] = [
        {
            "periodOfReport": "2024-09-28",
            "filingDate": "2024-11-01",
            "employeeCount": 164000,
        },
        {
            "periodOfReport": "2024-09-28",
            "filingDate": "2024-12-01",
            "employeeCount": 164500,
        },
        {
            "periodOfReport": "2023-09-30",
            "filingDate": "2023-11-03",
            "employeeCount": 161000,
        },
        {
            "periodOfReport": "2015-09-26",
            "filingDate": "2015-10-28",
            "employeeCount": 110000,
        },
    ]

    employees, missing = fmp_model.get_employee_count(
        ["AAPL"], "key", start_date="2020-01-01", user_subscription="Free"
    )

    # The amended filing wins, the period before the start date is left out.
    assert employees["AAPL"].to_dict() == {
        pd.Period("2023", "Y"): 161000,
        pd.Period("2024", "Y"): 164500,
    }
    assert missing == []
    # The Free plan is capped at five records, like the other endpoints.
    assert "limit=5" in requested[0]


def test_shares_float_and_etf_weights_are_decimals(responses):
    canned, _ = responses
    canned["shares-float"] = [
        {
            "date": "2026-10-04",
            "freeFloat": 99.5,
            "floatShares": 995,
            "outstandingShares": 1000,
            "source": "url",
        }
    ]
    canned["etf/country-weightings"] = [
        {"country": "United States", "weightPercentage": "97.31%"}
    ]
    canned["etf/sector-weightings"] = [
        {"symbol": "QQQ", "sector": "Technology", "weightPercentage": 52.5}
    ]

    shares_float, _ = fmp_model.get_shares_float(["AAPL"], "key")
    countries, _ = fmp_model.get_etf_country_weightings(["QQQ"], "key")
    sectors, _ = fmp_model.get_etf_sector_weightings(["QQQ"], "key")

    assert shares_float.loc["Free Float", "AAPL"] == pytest.approx(0.995)
    assert countries.loc["United States", "QQQ"] == pytest.approx(0.9731)
    assert sectors.loc["Technology", "QQQ"] == pytest.approx(0.525)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Microsoft Corporation", "Microsoft"),
        ("Apple Inc.", "Apple"),
        ("ASML Holding N.V.", "ASML"),
        ("Shell plc", "Shell"),
        ("Coca-Cola Co", "Coca-Cola"),
    ],
)
def test_company_name_is_shortened_for_the_search(name, expected):
    assert fmp_model._search_name(name) == expected


def test_mergers_acquisitions_keeps_only_the_tickers_own_deals(responses):
    canned, requested = responses
    canned["mergers-acquisitions-search"] = [
        {
            "symbol": "MSFT",
            "companyName": "MICROSOFT",
            "targetedSymbol": None,
            "targetedCompanyName": "ChipSoft",
            "transactionDate": "1995-02-09",
            "acceptedDate": "1995-02-09",
            "link": "a",
        },
        {
            "symbol": "ORCL",
            "companyName": "ORACLE",
            "targetedSymbol": "MSFT",
            "targetedCompanyName": "Microsoft",
            "transactionDate": "2001-01-01",
            "acceptedDate": "2001-01-01",
            "link": "b",
        },
        {
            "symbol": "MSFTX",
            "companyName": "Microsoftish Ltd",
            "targetedSymbol": None,
            "targetedCompanyName": "Other",
            "transactionDate": "2010-01-01",
            "acceptedDate": "2010-01-01",
            "link": "c",
        },
    ]

    deals, _ = fmp_model.get_mergers_acquisitions(
        ["MSFT"], {"MSFT": "Microsoft Corporation"}, "key"
    )

    assert "name=Microsoft&" in requested[0]
    assert deals.loc["MSFT", "Role"].tolist() == ["Acquirer", "Target"]
    assert "MSFTX" not in deals["Acquirer Symbol"].tolist()


def test_insider_statistics_are_indexed_by_quarter(responses):
    canned, _ = responses
    canned["insider-trading/statistics"] = [
        {
            "year": 2026,
            "quarter": quarter,
            "acquiredTransactions": quarter,
            "disposedTransactions": 1,
            "acquiredDisposedRatio": quarter,
            "totalAcquired": 1,
            "totalDisposed": 1,
            "averageAcquired": 1,
            "averageDisposed": 1,
            "totalPurchases": 0,
            "totalSales": 0,
        }
        for quarter in (3, 2, 1)
    ]

    statistics, _ = fmp_model.get_insider_trade_statistics(
        ["AAPL"], "key", start_date="2026-04-01"
    )

    assert statistics.loc["AAPL"].index.tolist() == [
        pd.Period("2026Q2"),
        pd.Period("2026Q3"),
    ]


def test_stock_grades_are_indexed_by_date_and_grading_company(responses):
    canned, _ = responses
    canned["grades"] = [
        {
            "date": "2026-10-01",
            "gradingCompany": "Needham",
            "previousGrade": "Hold",
            "newGrade": "Buy",
            "action": "upgrade",
        },
        {
            "date": "2026-01-01",
            "gradingCompany": "Needham",
            "previousGrade": "Hold",
            "newGrade": "Hold",
            "action": "maintain",
        },
    ]

    grades, _ = fmp_model.get_stock_grades(["AAPL"], "key", start_date="2026-06-01")

    assert grades.index.names[1:] == ["Date", "Grading Company"]
    assert grades["Action"].tolist() == ["Upgrade"]


def test_etf_identifiers_stay_text_and_a_stock_returns_no_data(responses):
    canned, _ = responses
    canned["etf/info"] = [
        {
            "name": "Invesco QQQ Trust",
            "isin": "US46090E1038",
            "securityCusip": "46090E103",
            "assetClass": "Equity",
            "expenseRatio": 0.18,
            "holdingsCount": 102,
        }
    ]

    information, missing = fmp_model.get_etf_information(["QQQ"], "key")

    # Parsed as JSON by pandas, "46090E103" would become the number 4.609e+107.
    assert information.loc["CUSIP", "QQQ"] == "46090E103"
    assert information.loc["Expense Ratio", "QQQ"] == pytest.approx(0.0018)
    assert missing == []

    canned["etf/info"] = []
    information, missing = fmp_model.get_etf_information(["AAPL"], "key")

    assert information.empty
    assert missing == ["AAPL"]


def test_transcripts_only_retrieve_what_is_needed_and_cache_each_one(
    responses, cache_location
):
    canned, requested = responses
    canned["earning-call-transcript-dates"] = [
        {"fiscalYear": 2026, "quarter": quarter, "date": date}
        for quarter, date in [(3, "2026-07-30"), (2, "2026-04-30"), (1, "2026-01-29")]
    ]
    canned["earning-call-transcript"] = [
        {"symbol": "AAPL", "content": "Good afternoon."}
    ]
    cache_controller.set_active_cache(
        cache_controller.get_cache(location=cache_location, enabled=True)
    )

    latest, _ = fmp_model.get_earnings_call_transcripts(["AAPL"], "key")

    assert latest.loc["AAPL"].index.tolist() == ["2026Q3"]
    assert sum("earning-call-transcript?" in url for url in requested) == 1

    requested.clear()
    history, _ = fmp_model.get_earnings_call_transcripts(
        ["AAPL"], "key", start_date="2026-04-01", latest=False
    )

    # 2026Q2 and 2026Q3 are in the range; 2026Q3 was already retrieved, so only 2026Q2 is requested.
    assert history.loc["AAPL"].index.tolist() == ["2026Q2", "2026Q3"]
    assert [url for url in requested if "earning-call-transcript?" in url] == [
        "https://financialmodelingprep.com/stable/earning-call-transcript?symbol=AAPL&year=2026&quarter=2&apikey=key"
    ]


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("EURUSD", True),
        ("EURUSD=X", True),
        ("usdjpy=x", True),
        ("BTCUSD", False),
        ("TSLA", False),
        ("MSFT.AS", False),
    ],
)
def test_currency_pairs_are_recognised(ticker, expected):
    assert currencies_model.is_currency_pair(ticker) is expected


def test_news_is_routed_by_ticker_type(cache_location, monkeypatch):
    calls: dict[str, list[str]] = {}

    def fake_search(feed):
        def search(symbols, **kwargs):  # noqa: ARG001
            calls[feed] = list(symbols)
            rows = [s for s in symbols if feed != "crypto" or s.endswith("USD")]
            return pd.DataFrame(
                {"Symbol": rows, "URL": [f"{feed}/{s}" for s in rows]},
                index=pd.DatetimeIndex(
                    ["2026-10-01"] * len(rows), name="Published Date"
                ),
            )

        return search

    for feed in ("stock", "crypto", "forex"):
        monkeypatch.setattr(
            toolkit_controller, f"_search_{feed}_news", fake_search(feed)
        )

    toolkit = toolkit_controller.Toolkit(
        ["AAPL", "BTCUSD", "EURUSD=X"],
        api_key="test-key",
        sleep_timer=False,
        use_cached_data=cache_location,
        benchmark_ticker=None,
    )
    news = toolkit.get_stock_news()

    # The pair goes to the forex feed without "=X"; the crypto feed claims BTCUSD, so
    # only AAPL is left for the stock feed.
    assert calls == {
        "crypto": ["AAPL", "BTCUSD"],
        "stock": ["AAPL"],
        "forex": ["EURUSD"],
    }
    assert sorted(news["Symbol"]) == ["AAPL", "BTCUSD", "EURUSD"]


def test_etf_methods_do_not_remove_tickers_that_are_not_funds(
    cache_location, monkeypatch
):
    monkeypatch.setattr(
        toolkit_controller,
        "_get_etf_holdings",
        lambda tickers, **kwargs: (pd.DataFrame(), list(tickers)),
    )

    toolkit = toolkit_controller.Toolkit(
        ["AAPL", "MSFT"],
        api_key="test-key",
        sleep_timer=False,
        use_cached_data=cache_location,
    )
    holdings = toolkit.get_etf_holdings()

    assert holdings.empty
    assert toolkit._tickers == ["AAPL", "MSFT"]


@pytest.fixture(name="calendar_requests")
def fixture_calendar_requests(monkeypatch):
    """Serve a small economic calendar and record the requested urls."""
    from financetoolkit.economics import fmp_model as economics_fmp_model

    requested: list[str] = []
    releases = [
        {
            "date": "2026-09-04 12:30:00",
            "country": "US",
            "event": "Non Farm Payrolls",
            "currency": "USD",
            "previous": 21,
            "actual": 162,
            "changePercentage": 671.43,
            "impact": "High",
            "unit": "K",
        },
        {
            "date": "2026-09-04 08:00:00",
            "country": "EU",
            "event": "ECB Decision",
            "currency": "EUR",
            "previous": 2.4,
            "actual": 2.15,
            "changePercentage": -10.42,
            "impact": "High",
            "unit": "%",
        },
        {
            "date": "2026-09-05 09:00:00",
            "country": "UK",
            "event": "Retail Sales",
            "currency": "GBP",
            "impact": "Medium",
        },
        {
            "date": "2026-09-05 02:00:00",
            "country": "VN",
            "event": "Inflation",
            "currency": "VND",
            "impact": "Low",
        },
    ]

    def fake_get_cached_financial_data(
        url, user_subscription="Free", sleep_timer=True
    ):  # noqa: ARG001
        requested.append(url)
        return pd.DataFrame(releases)

    monkeypatch.setattr(
        economics_fmp_model, "get_cached_financial_data", fake_get_cached_financial_data
    )

    return economics_fmp_model, requested


def test_economic_calendar_filters_by_country_name_or_code(calendar_requests):
    economics_fmp_model, _ = calendar_requests

    by_name = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-09-30", countries="United States"
    )
    by_code = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-09-30", countries="us"
    )
    combined = economics_fmp_model.get_economic_calendar(
        "key",
        "2026-09-01",
        "2026-09-30",
        countries=["Euro Area", "United Kingdom"],
        currencies="EUR,GBP",
        impact="High",
    )

    assert by_name.equals(by_code)
    assert by_name["Event"].tolist() == ["Non Farm Payrolls"]
    assert by_name[["Country", "Country Code"]].iloc[0].tolist() == [
        "United States",
        "US",
    ]
    # The UK release is Medium impact, so only the ECB decision is left.
    assert combined["Event"].tolist() == ["ECB Decision"]


def test_economic_calendar_converts_percentages_to_decimals(calendar_requests):
    economics_fmp_model, _ = calendar_requests

    calendar = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-09-30"
    ).set_index("Event")

    # The ECB rate is quoted in percent, the payrolls in thousands of jobs.
    assert calendar.loc[
        "ECB Decision", ["Previous", "Actual"]
    ].tolist() == pytest.approx([0.024, 0.0215])
    assert calendar.loc["Non Farm Payrolls", ["Previous", "Actual"]].tolist() == [
        21,
        162,
    ]
    assert calendar.loc[
        ["ECB Decision", "Non Farm Payrolls"], "Change %"
    ].tolist() == pytest.approx([-0.1042, 6.7143])


def test_economic_calendar_impact_all_keeps_every_release(calendar_requests):
    economics_fmp_model, _ = calendar_requests

    everything = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-09-30", impact="All"
    )
    high = economics_fmp_model.get_economic_calendar(
        "key", "2026-09-01", "2026-09-30", impact="High"
    )

    assert len(everything) == 4
    assert set(high["Impact"]) == {"High"}


def test_economic_calendar_splits_long_ranges_into_windows(calendar_requests):
    economics_fmp_model, requested = calendar_requests

    calendar = economics_fmp_model.get_economic_calendar(
        "key", "2026-01-01", "2026-09-30"
    )

    # Requested in parallel, so in any order.
    assert sorted(url.split("&from=")[1] for url in requested) == [
        "2026-01-01&to=2026-03-31",
        "2026-04-01&to=2026-06-29",
        "2026-06-30&to=2026-09-27",
        "2026-09-28&to=2026-09-30",
    ]
    # Every window returned the same releases; they are kept once.
    assert len(calendar) == 4


def test_economic_calendar_only_falls_back_to_requested_dates(
    calendar_requests, monkeypatch
):
    from financetoolkit import Economics
    from financetoolkit.economics import gmdb_model

    _, requested = calendar_requests
    monkeypatch.setattr(
        gmdb_model,
        "collect_global_macro_database_dataset",
        lambda cache=None: pd.DataFrame(),
    )

    Economics(
        api_key="key", start_date="2026-09-20", end_date="2026-10-03"
    ).get_economic_calendar()
    Economics(api_key="key").get_economic_calendar()

    # The first uses the dates it was created with, the second the endpoint's default
    # rather than a hundred years of 90-day windows.
    assert requested[0].endswith("&from=2026-09-20&to=2026-10-03")
    assert requested[1].endswith("economic-calendar?apikey=key")
    assert len(requested) == 2


def test_market_risk_premium_is_returned_as_decimals(monkeypatch):
    from financetoolkit import fmp_model

    monkeypatch.setattr(
        fmp_model,
        "get_financial_data",
        lambda url: pd.DataFrame(  # noqa: ARG005
            [
                {
                    "country": "United States",
                    "continent": "North America",
                    "countryRiskPremium": 0.23,
                    "totalEquityRiskPremium": 4.46,
                }
            ]
        ),
    )

    premium = fmp_model.get_market_risk_premium(api_key="key")

    assert premium.loc[
        "United States", ["Country Risk Premium", "Total Equity Risk Premium"]
    ].tolist() == (pytest.approx([0.0023, 0.0446]))


class _Response:
    """A response with a status code and a body, as requests returns it."""

    def __init__(self, status_code: int, text: str = "[]"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:  # noqa: PLR2004
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return []


def test_a_server_error_is_retried(monkeypatch):
    responses = [_Response(503, "Service Unavailable"), _Response(200, '[{"a": 1}]')]
    monkeypatch.setattr(fmp_model, "get_request", lambda *_, **__: responses.pop(0))
    monkeypatch.setattr(fmp_model.time, "sleep", lambda _: None)

    assert fmp_model.get_financial_data("https://example.com").to_dict("list") == {
        "a": [1]
    }


def test_a_timeout_is_reported_instead_of_raised(monkeypatch):
    attempts = []

    def timeout(*_, **__):
        attempts.append(1)
        raise requests.exceptions.ReadTimeout("Read timed out.")

    monkeypatch.setattr(fmp_model, "get_request", timeout)
    monkeypatch.setattr(fmp_model.time, "sleep", lambda _: None)

    result = fmp_model.get_financial_data("https://example.com")

    assert list(result.columns) == ["REQUEST FAILED"]
    assert len(attempts) == fmp_model.TRANSIENT_RETRY_LIMIT + 1


def test_an_error_message_instead_of_prices_is_no_data(monkeypatch):
    monkeypatch.setattr(
        fmp_model,
        "get_financial_data",
        lambda *_, **__: {"Error Message": "Something went wrong."},
    )

    result = fmp_model.get_historical_data(
        ticker="AAPL", api_key="KEY", start="2024-01-01", end="2024-02-01"
    )

    assert result.empty


def test_identifiers_keep_their_leading_zeros(monkeypatch):
    body = (
        '[{"symbol": "AAPL", "cik": "0000320193", "cusip": "037833100", "price": 1.5}]'
    )
    monkeypatch.setattr(fmp_model, "get_request", lambda *_, **__: _Response(200, body))

    result = fmp_model.get_financial_data("https://example.com").iloc[0]

    assert result["cik"] == "0000320193"
    assert result["cusip"] == "037833100"
    assert result["price"] == 1.5  # noqa: PLR2004


def test_a_rating_with_a_missing_score_is_kept(responses):
    canned, _ = responses
    canned["ratings-historical"] = [
        {
            "symbol": "AAPL",
            "date": "2024-01-02",
            "rating": "A",
            "overallScore": 4,
            "priceToBookScore": None,
        },
        {
            "symbol": "AAPL",
            "date": "2024-01-03",
            "rating": "A-",
            "overallScore": 3,
            "priceToBookScore": 2,
        },
    ]

    ratings, _ = fmp_model.get_rating(tickers="AAPL", api_key="KEY")

    assert ratings["Rating"].tolist() == ["A", "A-"]
