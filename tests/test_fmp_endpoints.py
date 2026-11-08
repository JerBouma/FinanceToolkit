"""FinancialModelingPrep Endpoint Tests"""

# The company, fund, transcript and news endpoints added in 2.2.2. Every response is
# faked by replacing get_financial_data, so these run offline and check what the
# Finance Toolkit does with a response: the shaping, the date filtering, the conversion
# of percentages to decimals and the routing between endpoints.

import json
from urllib.parse import parse_qs, urlsplit

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


def test_a_transcript_is_retrieved_for_one_fiscal_period(responses, caplog):
    canned, requested = responses
    canned["earning-call-transcript-dates"] = [
        {"fiscalYear": 2026, "quarter": quarter, "date": date}
        for quarter, date in [(3, "2026-07-30"), (2, "2026-04-30"), (1, "2026-01-29")]
    ]
    canned["earning-call-transcript"] = [
        {"symbol": "AAPL", "content": "Good afternoon."}
    ]

    transcripts, _ = fmp_model.get_earnings_call_transcripts(
        ["AAPL"], "key", fiscal_period="Q2 2026"
    )

    assert transcripts.loc["AAPL"].index.tolist() == ["2026Q2"]
    assert [url for url in requested if "earning-call-transcript?" in url] == [
        "https://financialmodelingprep.com/stable/earning-call-transcript?symbol=AAPL&year=2026&quarter=2&apikey=key"
    ]

    # A period without a transcript names the periods that do have one.
    transcripts, missing = fmp_model.get_earnings_call_transcripts(
        ["AAPL"], "key", fiscal_period="2030Q1"
    )

    assert transcripts.empty
    assert missing == ["AAPL"]
    assert "2026Q3, 2026Q2, 2026Q1" in caplog.text


@pytest.mark.parametrize(
    ("period", "expected"),
    [("2025Q4", (2025, 4)), ("2025-q1", (2025, 1)), ("Q3 2024", (2024, 3))],
)
def test_fiscal_periods_are_read_in_several_forms(period, expected):
    assert fmp_model.parse_fiscal_period(period) == expected


def test_an_unreadable_fiscal_period_is_an_error():
    with pytest.raises(ValueError, match="2025Q4"):
        fmp_model.parse_fiscal_period("last quarter")


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
    holdings = toolkit.etf.get_holdings()

    assert holdings.empty
    assert toolkit._tickers == ["AAPL", "MSFT"]


def test_executives_list_the_best_paid_first(responses):
    canned, _ = responses
    canned["key-executives"] = [
        {"name": "Undisclosed", "title": "Vice President", "pay": None},
        {"name": "Chief", "title": "Chief Executive Officer", "pay": 16759518},
        {"name": "Counsel", "title": "General Counsel", "pay": 5022482},
    ]

    executives, _ = fmp_model.get_executives(["AAPL"], "key")

    assert list(executives.loc["AAPL"].index) == ["Chief", "Counsel", "Undisclosed"]


def test_a_sector_a_fund_does_not_hold_weighs_zero(cache_location, monkeypatch):
    weightings = pd.DataFrame(
        {"QQQ": [0.6, None], "VTI": [0.4, 0.02]},
        index=pd.Index(["Technology", "Real Estate"], name="Sector"),
    )
    monkeypatch.setattr(
        toolkit_controller,
        "_get_etf_sector_weightings",
        lambda tickers, **kwargs: (weightings, []),
    )

    toolkit = toolkit_controller.Toolkit(
        ["QQQ", "VTI"],
        api_key="test-key",
        sleep_timer=False,
        use_cached_data=cache_location,
    )
    sectors = toolkit.etf.get_sector_weightings()

    assert sectors.loc["Real Estate", "QQQ"] == 0
    assert list(sectors.index) == ["Technology", "Real Estate"]


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


def test_the_rating_of_a_single_ticker_is_cached_and_filtered_by_date(
    responses, cache_location
):
    canned, requested = responses
    canned["ratings-historical"] = [
        {"symbol": "AMZN", "date": date, "rating": rating, "overallScore": score}
        for date, rating, score in [
            ("2026-01-05", "A-", 4),
            ("2025-12-31", "B+", 3),
            ("2020-12-31", "B", 3),
        ]
    ]

    # The ticker level stays on the model's answer, which the cache splits on.
    ratings, _ = fmp_model.get_rating(tickers="AMZN", api_key="KEY")
    assert ratings.index.get_level_values(0).unique().tolist() == ["AMZN"]

    toolkit = toolkit_controller.Toolkit(
        ["AMZN"],
        api_key="test-key",
        sleep_timer=False,
        use_cached_data=cache_location,
        start_date="2021-01-01",
        end_date="2025-12-31",
    )
    rating = toolkit.get_rating()

    # One ticker returns its dates only, limited to the start and end date.
    assert rating.index.tolist() == ["2025-12-31"]
    assert rating["Rating"].tolist() == ["B+"]

    # A second Toolkit is answered from the cache rather than finding it marked as missing.
    calls = len(requested)
    again = toolkit_controller.Toolkit(
        ["AMZN"],
        api_key="test-key",
        sleep_timer=False,
        use_cached_data=cache_location,
    ).get_rating()

    # Only the probe of the key's plan may go out, which the canned answers leave undecided.
    assert [url for url in requested[calls:] if "ratings-historical" in url] == []
    assert again["Rating"].tolist() == ["B+", "A-"]


def test_yahoo_finance_symbols_are_requested_as_financial_modeling_prep_writes_them(
    monkeypatch,
):
    requested = []

    def fake_get_request(url, timeout):  # noqa: ARG001
        requested.append(url)
        return _Response(200, "[]")

    monkeypatch.setattr(fmp_model, "get_request", fake_get_request)

    fmp_model.get_financial_data(
        "https://financialmodelingprep.com/stable/batch-quote?symbols=AAPL,EURUSD=X,BTC-USD&apikey=K"
    )

    assert "symbols=AAPL,EURUSD,BTCUSD&" in requested[0]


def test_dividends_are_split_adjusted_like_the_prices(responses):
    canned, _ = responses
    dates = ["2020-08-06", "2020-08-07", "2020-11-05", "2020-11-06"]
    canned["historical-price-eod/full"] = [
        {
            "date": date,
            "open": 110.0,
            "high": 111.0,
            "low": 109.0,
            "close": 110.0,
            "volume": 1,
        }
        for date in dates
    ]
    canned["historical-price-eod/dividend-adjusted"] = [
        {"date": date, "adjClose": 110.0} for date in dates
    ]
    # Apple's 2020 dividends: 0.82 as paid before the 4-for-1 split, 0.205 per share today.
    canned["dividends"] = [
        {"date": "2020-11-06", "dividend": 0.205, "adjDividend": 0.205},
        {"date": "2020-08-07", "dividend": 0.82, "adjDividend": 0.205},
    ]

    historical = fmp_model.get_historical_data(
        "AAPL", "key", start="2020-08-06", end="2020-11-06", sleep_timer=False
    )

    assert historical.loc["2020-08-07", "Dividends"] == pytest.approx(0.205)
    assert historical.loc["2020-11-06", "Dividends"] == pytest.approx(0.205)


def test_profile_and_quote_changes_are_decimals(responses):
    canned, _ = responses
    canned["profile"] = [
        {
            "symbol": "AAPL",
            "price": 250.0,
            "changePercentage": 1.25,
            "companyName": "Apple Inc.",
            "image": "",
            "defaultImage": False,
            "isEtf": False,
            "isActivelyTrading": True,
            "isAdr": False,
            "isFund": False,
        }
    ]
    canned["quote"] = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "price": 250.0,
            "changePercentage": -0.5,
        }
    ]

    profile, _ = fmp_model.get_profile("AAPL", api_key="K")
    quote, _ = fmp_model.get_quote("AAPL", api_key="K")

    # Published in percent, returned as decimals like every other change.
    assert profile.loc["Change %", "AAPL"] == pytest.approx(0.0125)
    assert quote.loc["Change %", "AAPL"] == pytest.approx(-0.005)
    assert profile.loc["Company Name", "AAPL"] == "Apple Inc."


def test_quote_change_has_no_float_residue(responses):
    canned, _ = responses
    canned["quote"] = [{"symbol": "AAPL", "price": 250.0, "changePercentage": -0.75391}]
    canned["profile"] = [
        {
            "symbol": "AAPL",
            "price": 250.0,
            "changePercentage": -0.75391,
            "image": "",
            "defaultImage": False,
            "isEtf": False,
            "isActivelyTrading": True,
            "isAdr": False,
            "isFund": False,
        }
    ]

    quote, _ = fmp_model.get_quote("AAPL", api_key="K")
    profile, _ = fmp_model.get_profile("AAPL", api_key="K")

    # -0.75391 / 100 is -0.007539100000000001 in floating point.
    assert repr(quote.loc["Change %", "AAPL"]) == "-0.0075391"
    assert repr(profile.loc["Change %", "AAPL"]) == "-0.0075391"


def test_two_fiscal_periods_in_one_calendar_period_are_reported(responses, caplog):
    """After a change of fiscal year end, the dropped period is named in a warning."""
    canned, _ = responses
    canned["income-statement"] = [
        {"date": "2023-12-31", "symbol": "XYZ", "revenue": 300},
        {"date": "2023-06-30", "symbol": "XYZ", "revenue": 200},
        {"date": "2022-06-30", "symbol": "XYZ", "revenue": 100},
    ]

    with caplog.at_level("WARNING"):
        statement = fmp_model.get_financial_statement(
            "XYZ", statement="income", api_key="key"
        )

    assert statement.loc["revenue"].tolist() == [300, 100]
    warning = " ".join(record.getMessage() for record in caplog.records)
    assert "XYZ" in warning
    assert "2023-12-31, 2023-06-30" in warning
    assert "(2023)" in warning


def test_statements_keep_the_fiscal_period_end_date(responses):
    canned, _ = responses
    canned["income-statement"] = [
        {
            "date": "2024-01-28",
            "symbol": "NVDA",
            "reportedCurrency": "USD",
            "revenue": 2.0,
        },
        {
            "date": "2023-01-29",
            "symbol": "NVDA",
            "reportedCurrency": "USD",
            "revenue": 1.0,
        },
    ]

    statement = fmp_model.get_financial_statement(
        ticker="NVDA", statement="income", api_key="key", quarter=False
    )

    # The columns are the calendar years holding most of each fiscal year, while the
    # actual last day of the fiscal year is kept as a statistic.
    assert [str(period) for period in statement.columns] == ["2023", "2022"]
    assert statement.loc["periodEndDate"].tolist() == ["2024-01-28", "2023-01-29"]


def test_a_certificate_error_is_reported_without_retrying(monkeypatch):
    attempts = []

    def certificate_error(*_, **__):
        attempts.append(1)
        raise requests.exceptions.SSLError("certificate verify failed")

    monkeypatch.setattr(fmp_model, "get_request", certificate_error)
    monkeypatch.setattr(fmp_model.time, "sleep", lambda _: None)

    assert fmp_model.get_financial_data("https://example.com").empty
    assert len(attempts) == 1


def test_a_refused_connection_is_retried_a_few_times(monkeypatch):
    attempts = []

    def refused(*_, **__):
        attempts.append(1)
        raise requests.exceptions.ConnectionError("Name or service not known")

    monkeypatch.setattr(fmp_model, "get_request", refused)
    monkeypatch.setattr(fmp_model.time, "sleep", lambda _: None)

    assert fmp_model.get_financial_data("https://example.com").empty
    assert len(attempts) == fmp_model.TRANSIENT_RETRY_LIMIT + 1


def test_an_unrecognised_error_is_logged_without_the_api_key(monkeypatch, caplog):
    url = (
        "https://financialmodelingprep.com/stable/profile?symbol=X&apikey=SECRETKEY123"
    )

    def forbidden(*_, **__):
        response = requests.models.Response()
        response.status_code = 403
        response.url = url
        response.raise_for_status()

    monkeypatch.setattr(fmp_model, "get_request", forbidden)

    with caplog.at_level("ERROR"):
        result = fmp_model.get_financial_data(url)

    assert list(result.columns) == ["REQUEST FAILED"]
    assert "unrecognised error" in caplog.text
    assert "SECRETKEY123" not in caplog.text


@pytest.mark.parametrize(
    ("adjusted_error", "incomplete"),
    [("LIMIT REACH", True), ("PREMIUM QUERY PARAMETER", False)],
)
def test_prices_are_incomplete_only_after_a_temporary_failure(
    monkeypatch, adjusted_error, incomplete
):
    prices = [
        {"date": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ]

    def fake(url, **kwargs):  # noqa: ARG001
        if "dividend-adjusted" in url:
            return pd.DataFrame(columns=[adjusted_error])
        if "/dividends" in url:
            return []
        return prices

    monkeypatch.setattr(fmp_model, "get_financial_data", fake)

    data = fmp_model.get_historical_data(
        "AAPL", api_key="KEY", start="2024-01-01", end="2024-01-05"
    )

    assert data.attrs.get("incomplete", False) is incomplete
    assert data["Adj Close"].tolist() == data["Close"].tolist()


def test_free_plan_dividends_are_completed_with_older_yahoo_payments(monkeypatch):
    import yfinance as yf

    from financetoolkit import fmp_model

    latest = pd.Series(
        [0.25] * 5,
        index=pd.period_range("2024-02-01", periods=5, freq="Q").asfreq("D", "start"),
    )
    yahoo = pd.Series(
        [0.2, 0.2, 0.25],
        index=pd.DatetimeIndex(
            ["2023-05-01", "2023-08-01", "2024-02-01"], tz="America/New_York"
        ),
    )
    monkeypatch.setattr(
        yf, "Ticker", lambda symbol: type("T", (), {"dividends": yahoo})()
    )

    completed = fmp_model._complete_older_dividends("AAPL", latest)

    # The two earlier payments come first; the overlapping one is kept from FMP.
    assert completed.index[:2].astype(str).tolist() == ["2023-05-01", "2023-08-01"]
    assert completed.iloc[2:].equals(latest)


def test_free_plan_dividends_without_yahoo_history_warn(monkeypatch):
    import yfinance as yf

    from financetoolkit import fmp_model

    latest = pd.Series(
        [0.25] * 5, index=pd.period_range("2024-01-01", periods=5, freq="D")
    )
    monkeypatch.setattr(
        yf,
        "Ticker",
        lambda symbol: type("T", (), {"dividends": pd.Series(dtype=float)})(),
    )

    from unittest.mock import patch

    with patch.object(fmp_model.logger, "warning") as warning:
        assert fmp_model._complete_older_dividends("XYZ", latest).equals(latest)

    assert warning.call_args[0][0].startswith("No dividend data is available for")


def _raising_ticker(symbol):
    raise RuntimeError("Too Many Requests")


@pytest.mark.parametrize(
    "ticker",
    [lambda symbol: type("T", (), {"dividends": None})(), _raising_ticker],
    ids=["none", "raises"],
)
def test_free_plan_dividends_when_yahoo_fails(monkeypatch, ticker):
    import yfinance as yf

    from financetoolkit import fmp_model

    latest = pd.Series(
        [0.25] * 5, index=pd.period_range("2025-01-01", periods=5, freq="D")
    )
    # yfinance returns None for the dividends when Yahoo answers with an error, and the
    # request itself can raise; both count as no older payments.
    monkeypatch.setattr(yf, "Ticker", ticker)

    from unittest.mock import patch

    with patch.object(fmp_model.logger, "warning") as warning:
        assert fmp_model._complete_older_dividends("AAPL", latest).equals(latest)

    assert warning.call_args[0][0].startswith("No dividend data is available for")


def test_institutional_ownership_is_quarterly_in_decimals_and_skips_open_quarters(
    monkeypatch,
):
    requested: list[str] = []

    def fake_get_financial_data(
        url, sleep_timer=True, raw=False, user_subscription="Free"
    ):  # noqa: ARG001
        requested.append(url)
        quarter = int(url.split("quarter=")[1].split("&")[0])
        if "symbol-positions-summary" in url:
            return pd.DataFrame(
                [
                    {
                        "investorsHolding": 100 * quarter,
                        "investorsHoldingChange": 10,
                        "numberOf13Fshares": 1_000,
                        "numberOf13FsharesChange": 5,
                        "totalInvested": 2_000,
                        "ownershipPercent": 60.0 + quarter,
                        "ownershipPercentChange": 1.0,
                        "newPositions": 3,
                        "increasedPositions": 4,
                        "reducedPositions": 5,
                        "closedPositions": 6,
                        "putCallRatio": 0.9,
                    }
                ]
            )
        # The ten largest holders of the shares; an option position must not count.
        holders = [{"ownership": 2.0, "putCallShare": "Share"} for _ in range(12)]
        holders.insert(1, {"ownership": 5.0, "putCallShare": "Call"})
        return pd.DataFrame(holders)

    monkeypatch.setattr(fmp_model, "get_financial_data", fake_get_financial_data)

    # On 2026-11-05 the third quarter is still within its 45-day filing window.
    class FrozenDatetime(fmp_model.datetime):
        @classmethod
        def today(cls):
            return cls(2026, 11, 5)

    monkeypatch.setattr(fmp_model, "datetime", FrozenDatetime)

    ownership, no_data = fmp_model.get_institutional_ownership(
        ["AAPL"], "key", start_date="2026-01-01", end_date="2026-12-31"
    )

    assert no_data == []
    assert ownership.loc["AAPL"].index.tolist() == [
        pd.Period("2026Q1"),
        pd.Period("2026Q2"),
    ]
    assert ownership.loc[("AAPL", pd.Period("2026Q2")), "Ownership"] == pytest.approx(
        0.62
    )
    assert ownership.loc[
        ("AAPL", pd.Period("2026Q2")), "Ownership Change"
    ] == pytest.approx(0.01)
    assert ownership.loc[
        ("AAPL", pd.Period("2026Q2")), "Top 10 Ownership"
    ] == pytest.approx(0.2)
    assert all("quarter=3" not in url for url in requested)


def test_institutional_ownership_reports_a_ticker_without_filings(monkeypatch):
    monkeypatch.setattr(
        fmp_model,
        "get_financial_data",
        lambda url, sleep_timer=True, raw=False, user_subscription="Free": pd.DataFrame(),  # noqa: ARG005
    )

    ownership, no_data = fmp_model.get_institutional_ownership(
        ["ASML.AS"], "key", start_date="2025-01-01", end_date="2025-06-30"
    )

    assert ownership.empty
    assert no_data == ["ASML.AS"]


def test_price_targets_leave_the_average_of_an_empty_window_out(responses):
    canned, _ = responses
    canned["price-target-consensus"] = [
        {
            "symbol": "NVDA",
            "targetHigh": 300,
            "targetLow": 150,
            "targetConsensus": 250,
            "targetMedian": 240,
        }
    ]
    canned["price-target-summary"] = [
        {
            "symbol": "NVDA",
            "lastMonthCount": 0,
            "lastMonthAvgPriceTarget": 0,
            "lastQuarterCount": 4,
            "lastQuarterAvgPriceTarget": 260,
            "lastYearCount": 20,
            "lastYearAvgPriceTarget": 200,
            "allTimeCount": 80,
            "allTimeAvgPriceTarget": 120,
        }
    ]

    targets, missing = fmp_model.get_price_targets(["NVDA"], "key")

    assert targets.loc["Target Consensus", "NVDA"] == 250
    # The endpoint reports the average of a window without targets as 0.
    assert pd.isna(targets.loc["Last Month Average", "NVDA"])
    assert targets.loc["Last Quarter Average", "NVDA"] == 260
    assert missing == []


def test_price_target_measures():
    targets = pd.DataFrame(
        {"NVDA": [250.0, 240.0, 300.0, 150.0, 260.0, 200.0]},
        index=[
            "Target Consensus",
            "Target Median",
            "Target High",
            "Target Low",
            "Last Quarter Average",
            "Last Year Average",
        ],
    )

    measures = fmp_model.add_price_target_metrics(targets, pd.Series({"NVDA": 200.0}))

    assert measures.loc["Price", "NVDA"] == 200
    assert measures.loc["Upside", "NVDA"] == pytest.approx(0.25)
    assert measures.loc["Dispersion", "NVDA"] == pytest.approx(0.6)
    assert measures.loc["Revision Momentum", "NVDA"] == pytest.approx(0.3)
    # The measures follow the targets they are calculated from.
    assert list(measures.index[:5]) == [
        "Target Consensus",
        "Target Median",
        "Target High",
        "Target Low",
        "Price",
    ]


def test_a_price_target_without_a_quote_has_no_upside():
    targets = pd.DataFrame(
        {"NVDA": [250.0, 240.0, 300.0, 150.0]},
        index=["Target Consensus", "Target Median", "Target High", "Target Low"],
    )

    measures = fmp_model.add_price_target_metrics(targets, pd.Series(dtype=float))

    assert pd.isna(measures.loc["Upside", "NVDA"])
    assert measures.loc["Dispersion", "NVDA"] == pytest.approx(0.6)
    assert "Revision Momentum" not in measures.index


def test_price_target_history_pages_back_to_the_start_date(monkeypatch):
    requested = []

    def target(day, company, price_target, price):
        return {
            "publishedDate": f"{day}T12:00:00.000Z",
            "analystCompany": company,
            "priceTarget": price_target,
            "adjPriceTarget": price_target,
            "priceWhenPosted": price,
        }

    recent = [target("2025-06-30", "Firm", 220, 200)] * 99 + [
        target("2025-03-03", "Bank", 150, 120)
    ]
    older = [
        target("2025-02-03", "Bank", 140, 140),
        target("2025-01-06", "Broker", 100, 125),
        target("2024-12-02", "Broker", 90, 100),
    ]

    def fake_get_financial_data(
        url, sleep_timer=True, raw=False, user_subscription="Free"
    ):  # noqa: ARG001
        requested.append(url)
        page = int(url.split("page=")[1].split("&")[0])

        return pd.DataFrame({0: recent, 1: older}.get(page, []))

    monkeypatch.setattr(fmp_model, "get_financial_data", fake_get_financial_data)

    history, missing = fmp_model.get_price_target_history(
        ["NVDA"], "key", start_date="2025-01-01", end_date="2025-03-31"
    )

    # The second page reaches back before the start date, so no third page is requested.
    assert len(requested) == 2
    assert "from=2025-01-01" in requested[0] and "to=2025-03-31" in requested[0]
    nvda = history.loc["NVDA"]
    assert nvda.index.get_level_values("Analyst Company").tolist() == [
        "Broker",
        "Bank",
        "Bank",
    ]
    assert nvda["Implied Upside"].tolist() == pytest.approx([-0.2, 0.0, 0.25])
    assert missing == []


def test_analyst_recommendations_count_the_analysts_and_score_them(responses):
    canned, requested = responses
    canned["grades-historical"] = [
        {
            "symbol": "AAPL",
            "date": "2025-02-01",
            "analystRatingsStrongBuy": 2,
            "analystRatingsBuy": 4,
            "analystRatingsHold": 3,
            "analystRatingsSell": 1,
            "analystRatingsStrongSell": 0,
        },
        {
            "symbol": "AAPL",
            "date": "2024-01-01",
            "analystRatingsStrongBuy": 1,
            "analystRatingsBuy": 1,
            "analystRatingsHold": 1,
            "analystRatingsSell": 1,
            "analystRatingsStrongSell": 1,
        },
    ]

    recommendations, _ = fmp_model.get_analyst_recommendations(
        ["AAPL"], "key", start_date="2025-01-01", end_date="2025-12-31"
    )

    assert "from=2025-01-01" in requested[0]
    aapl = recommendations.loc["AAPL"]
    # The endpoint ignores the date range, so the 2024 month is filtered out here.
    assert aapl.index.tolist() == [pd.Timestamp("2025-02-01")]
    assert aapl["Analysts"].iloc[0] == 10
    assert aapl["Buy Share"].iloc[0] == pytest.approx(0.6)
    # (1 × 2 + 2 × 4 + 3 × 3 + 4 × 1) / 10
    assert aapl["Mean Recommendation"].iloc[0] == pytest.approx(2.3)


def test_the_toolkit_measures_the_upside_against_the_quote(responses, cache_location):
    canned, _ = responses
    canned["price-target-consensus"] = [
        {
            "symbol": "AAPL",
            "targetHigh": 300,
            "targetLow": 200,
            "targetConsensus": 250,
            "targetMedian": 250,
        }
    ]
    canned["quote"] = [{"symbol": "AAPL", "price": 200.0}]

    toolkit = toolkit_controller.Toolkit(
        ["AAPL"],
        api_key="test-key",
        sleep_timer=False,
        use_cached_data=cache_location,
        benchmark_ticker=None,
    )
    targets = toolkit.get_price_targets()

    assert targets.loc["Upside", "AAPL"] == pytest.approx(0.25)
    assert targets.loc["Dispersion", "AAPL"] == pytest.approx(0.4)


class _JSONResponse(_Response):
    """A response whose body is JSON, as requests returns it."""

    def json(self):
        return json.loads(self.text)


PROFILE_FIELDS = {
    "price": 1.0,
    "image": "",
    "defaultImage": False,
    "isEtf": False,
    "isActivelyTrading": True,
    "isAdr": False,
    "isFund": False,
}


def test_each_symbol_is_requested_as_itself(monkeypatch):
    # Answered as a server reads the request: an unencoded M&M.NS would be read as M, Macy's.
    names = {
        "M": "Macy's",
        "M&M.NS": "Mahindra & Mahindra",
        "BRK.B": "Berkshire Hathaway",
        "^N225": "Nikkei 225",
    }
    requested = []

    def fake_get_request(url, timeout):  # noqa: ARG001
        sent = requests.Request("GET", url).prepare().url
        requested.append(sent)
        symbol = parse_qs(urlsplit(sent).query)["symbol"][0]

        return _JSONResponse(
            200,
            json.dumps(
                [{"symbol": symbol, "companyName": names[symbol], **PROFILE_FIELDS}]
            ),
        )

    monkeypatch.setattr(fmp_model, "get_request", fake_get_request)

    profile, _ = fmp_model.get_profile(["M&M.NS", "BRK.B", "^N225"], api_key="K")

    assert profile.loc["Company Name"].to_dict() == {
        "M&M.NS": "Mahindra & Mahindra",
        "BRK.B": "Berkshire Hathaway",
        "^N225": "Nikkei 225",
    }
    # Requests are sent by concurrent workers; completion order is not guaranteed.
    assert sorted(url.split("?")[1] for url in requested) == sorted(
        [
            "symbol=M%26M.NS&apikey=K",
            "symbol=BRK.B&apikey=K",
            "symbol=%5EN225&apikey=K",
        ]
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("/profile?symbol=M&M.NS&apikey=K", "/profile?symbol=M%26M.NS&apikey=K"),
        (
            "/income-statement?symbol=J&KBANK.NS&period=annual&apikey=K&limit=5",
            "/income-statement?symbol=J%26KBANK.NS&period=annual&apikey=K&limit=5",
        ),
        (
            "/news?apikey=K&symbols=M&M.NS,BRK.B",
            "/news?apikey=K&symbols=M%26M.NS,BRK.B",
        ),
        # An encoded symbol is not encoded twice.
        (
            "/light?symbol=%5EGSPC&from=2024-01-01",
            "/light?symbol=%5EGSPC&from=2024-01-01",
        ),
        ("/quote?symbol=A+B#C&apikey=K", "/quote?symbol=A%2BB%23C&apikey=K"),
    ],
)
def test_symbols_are_encoded_up_to_the_next_parameter(url, expected):
    assert fmp_model.to_financial_modeling_prep_symbols(url) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Invalid API KEY. Feel free to create a Free API Key.", "INVALID API KEY"),
        ("Limit Reach . Please upgrade your plan.", "LIMIT REACH"),
        (
            "Premium Query Parameter: this value is not available.",
            "PREMIUM QUERY PARAMETER",
        ),
        ("Exclusive Endpoint : This endpoint is not available.", "EXCLUSIVE ENDPOINT"),
        ("Symbol not found.", "NO DATA"),
        ("Something unexpected happened.", "REQUEST FAILED"),
    ],
)
@pytest.mark.parametrize("raw", [False, True])
def test_an_error_message_sent_with_status_200_is_classified(
    monkeypatch, message, expected, raw
):
    body = json.dumps({"Error Message": message})
    monkeypatch.setattr(
        fmp_model, "get_request", lambda *_, **__: _JSONResponse(200, body)
    )
    failures = fmp_model.error_model.get_request_failures()

    result = fmp_model.get_financial_data(
        "https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey=K", raw=raw
    )

    assert list(result.columns) == [expected]
    # A failed request is counted, which keeps its answer out of the cache.
    assert fmp_model.error_model.get_request_failures() == failures + 1


def test_a_rate_limit_sent_with_status_200_is_retried(monkeypatch):
    responses = [
        _JSONResponse(200, json.dumps({"Error Message": "Limit Reach ."})),
        _JSONResponse(200, '[{"a": 1}]'),
    ]
    monkeypatch.setattr(fmp_model, "get_request", lambda *_, **__: responses.pop(0))
    monkeypatch.setattr(fmp_model.time, "sleep", lambda _: None)

    result = fmp_model.get_financial_data(
        "https://example.com", user_subscription="Premium"
    )

    assert result.to_dict("list") == {"a": [1]}


def test_an_error_message_sent_with_status_200_is_not_cached(
    cache_location, monkeypatch
):
    from financetoolkit.discovery import discovery_model  # noqa: PLC0415

    cache_controller.set_active_cache(
        cache_controller.get_cache(location=cache_location)
    )
    responses = [
        _JSONResponse(200, json.dumps({"Error Message": "Limit Reach ."})),
        _JSONResponse(200, '[{"symbol": "M&M.NS", "price": 1.0}]'),
    ]
    monkeypatch.setattr(fmp_model, "get_request", lambda *_, **__: responses.pop(0))
    url = "https://financialmodelingprep.com/stable/quote?symbol=M&M.NS&apikey=K"

    first = discovery_model.get_cached_financial_data(url)
    second = discovery_model.get_cached_financial_data(url)

    assert list(first.columns) == ["LIMIT REACH"]
    assert second["symbol"].tolist() == ["M&M.NS"]


def test_price_targets_say_when_the_latest_target_was_published(responses):
    canned, requested = responses
    canned["price-target-consensus"] = [
        {
            "symbol": "NVDA",
            "targetHigh": 300,
            "targetLow": 150,
            "targetConsensus": 250,
            "targetMedian": 240,
        }
    ]
    canned["price-target-news"] = [
        {
            "symbol": "NVDA",
            "publishedDate": "2026-10-05T15:32:19.000Z",
            "analystCompany": "BNP Paribas",
            "adjPriceTarget": 345,
        }
    ]

    targets, _ = fmp_model.get_price_targets(["NVDA"], "key")
    measures = fmp_model.add_price_target_metrics(
        targets, pd.Series({"NVDA": 200.0}), today=pd.Timestamp("2026-11-07")
    )

    news = next(url for url in requested if "price-target-news" in url)
    assert "from=" in news and "to=" in news
    assert measures.loc["Latest Target Date", "NVDA"] == "2026-10-05"
    assert measures.loc["Days Since Latest Target", "NVDA"] == 33
    assert measures.loc["Upside", "NVDA"] == pytest.approx(0.25)


def test_price_targets_without_any_published_target_have_no_latest_date(responses):
    canned, _ = responses
    canned["price-target-consensus"] = [
        {
            "symbol": "NVDA",
            "targetHigh": 300,
            "targetLow": 150,
            "targetConsensus": 250,
            "targetMedian": 240,
        }
    ]

    targets, _ = fmp_model.get_price_targets(["NVDA"], "key")
    measures = fmp_model.add_price_target_metrics(targets, pd.Series({"NVDA": 200.0}))

    assert pd.isna(measures.loc["Latest Target Date", "NVDA"])
    assert pd.isna(measures.loc["Days Since Latest Target", "NVDA"])
