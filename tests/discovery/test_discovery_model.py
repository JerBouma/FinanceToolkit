"""Discovery Model Tests"""

# pylint: disable=missing-function-docstring

import pandas as pd
import pytest

from financetoolkit.discovery import discovery_model


@pytest.fixture(name="responses")
def fixture_responses(monkeypatch):
    """Answer the search and quote endpoints offline, recording every URL."""
    requested: list[str] = []

    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        requested.append(url)

        if "/search-isin" in url:
            # Ranked by market capitalisation in each listing's own currency.
            return pd.DataFrame(
                {
                    "symbol": ["AAPL.MX", "APC.DE", "AAPL"],
                    "name": "Apple Inc.",
                    "isin": "US0378331005",
                    "marketCap": [91e12, 4.4e12, 4.9e12],
                }
            )
        if "/batch-quote" in url:
            return pd.DataFrame(
                {
                    "symbol": ["AAPL.MX", "APC.DE", "AAPL"],
                    "exchange": ["MEX", "XETRA", "NASDAQ"],
                    "volume": [8298, 46309, 37881199],
                }
            )

        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    return requested


def test_an_isin_is_searched_as_an_isin_and_its_primary_listing_comes_first(responses):
    listings = discovery_model.get_instruments(
        api_key="KEY", query="US0378331005", search_method="isin"
    )

    # Asked with `query`, the endpoint answers with Apple whatever the ISIN.
    assert "search-isin?isin=US0378331005&" in responses[0]
    assert list(listings.index) == ["AAPL", "APC.DE", "AAPL.MX"]
    assert listings.loc["AAPL", "Exchange Code"] == "NASDAQ"


def test_an_unknown_search_method_is_refused(responses):
    with pytest.raises(ValueError, match="not valid"):
        discovery_model.get_instruments(
            api_key="KEY", query="X", search_method="ticker"
        )

    assert responses == []


@pytest.fixture(name="screener")
def fixture_screener(monkeypatch):
    """Answer the screener, exchange and exchange rate endpoints offline."""

    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        if "/company-screener" in url:
            return pd.DataFrame(
                {
                    "symbol": ["INGS.TO", "INGA.AS", "ING", "AEH", "SMALL.T"],
                    "companyName": [
                        "ING Groep N.V.",
                        "ING Groep N.V.",
                        "ING Groep N.V.",
                        "Aegon N.V. PRP CP SEC 6.375",
                        "Small Japanese Co.",
                    ],
                    "marketCap": [117e9, 84.6e9, 95.3e9, 52e9, 1.5e12],
                    "exchangeShortName": ["TSX", "AMS", "NYSE", "NYSE", "JPX"],
                    "volume": [10_000, 500_000, 100_000, 1_000, 5_000],
                }
            )
        if "/available-exchanges" in url:
            return pd.DataFrame(
                {
                    "exchange": ["TSX", "AMS", "NYSE", "JPX"],
                    "countryCode": ["CA", "NL", "US", "JP"],
                }
            )
        if "/batch-quote" in url:
            return pd.DataFrame(
                {
                    "symbol": ["CADUSD", "EURUSD", "JPYUSD"],
                    "price": [0.70, 1.12, 0.0065],
                }
            )
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)


def test_the_screener_ranks_and_bounds_market_caps_in_dollars(screener):  # noqa: ARG001
    listings = discovery_model.get_stock_screener(api_key="KEY", market_cap_higher=50e9)

    # 117 billion Canadian dollars is less than 95 billion US dollars, 1.5 trillion yen is
    # below the bound and the perpetual has no market cap of its own.
    assert list(listings.index) == ["ING", "INGA.AS", "INGS.TO"]
    assert listings.loc["INGS.TO", "Currency"] == "CAD"
    assert listings.loc["INGS.TO", "Market Cap (USD)"] == pytest.approx(81.9e9)


def test_the_screener_can_keep_one_listing_per_company(screener):  # noqa: ARG001
    listings = discovery_model.get_stock_screener(api_key="KEY", primary_only=True)

    assert "INGA.AS" in listings.index
    assert not {"ING", "INGS.TO"} & set(listings.index)


def test_a_company_keeps_the_market_cap_and_listing_of_its_home_exchange(monkeypatch):
    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        if "/company-screener" in url:
            # A CEDEAR in Buenos Aires trades in fractions of a share: more volume, and a
            # market cap far too large.
            return pd.DataFrame(
                {
                    "symbol": ["AMD.BA", "AMD"],
                    "companyName": ["Advanced Micro Devices, Inc."] * 2,
                    "marketCap": [1.0e15, 400e9],
                    "exchangeShortName": ["BUE", "NASDAQ"],
                    "country": ["US", "US"],
                    "volume": [9_000_000, 40_000],
                }
            )
        if "/available-exchanges" in url:
            return pd.DataFrame(
                {"exchange": ["BUE", "NASDAQ"], "countryCode": ["AR", "US"]}
            )
        if "/batch-quote" in url:
            return pd.DataFrame({"symbol": ["ARSUSD"], "price": [0.001]})
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    every_listing = discovery_model.get_stock_screener(api_key="KEY")
    primary = discovery_model.get_stock_screener(api_key="KEY", primary_only=True)

    assert (every_listing["Market Cap (USD)"] == 400e9).all()
    assert list(primary.index) == ["AMD"]
    assert "_home_listing" not in primary.columns


def test_the_screener_takes_country_names_and_screens_stocks_only(monkeypatch):
    urls = []

    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        urls.append(url)
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    discovery_model.get_stock_screener(api_key="KEY", country="Netherlands")
    discovery_model.get_stock_screener(api_key="KEY", country="UK", is_etf=True)

    assert (
        "country=NL" in urls[0]
        and "isEtf=false" in urls[0]
        and "isFund=false" in urls[0]
    )
    assert "country=GB" in urls[1] and "isEtf=true" in urls[1]


def test_insider_trades_of_a_date_are_filtered_locally(monkeypatch):
    pages = {
        0: [("2026-10-23", "A"), ("2026-10-22", "B")],
        1: [("2026-10-21", "C"), ("2026-10-20", "D")],
    }
    requested = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        page = int(url.split("page=")[1].split("&")[0])
        rows = pages.get(page, [])
        return pd.DataFrame(
            {"symbol": [row[1] for row in rows], "filingDate": [row[0] for row in rows]}
        )

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    trades = discovery_model.get_insider_trading_latest(
        api_key="KEY", date="2026-10-21"
    )

    # The endpoint ignores a date, so pages are read until it has passed.
    assert list(trades.index) == ["C"]
    assert len(requested) == 2  # noqa: PLR2004


def test_sector_changes_are_decimals(monkeypatch):
    def fake(url, user_subscription="Free"):  # noqa: ARG001
        assert "exchange=NYSE" in url
        return pd.DataFrame(
            {
                "date": ["2026-10-23"],
                "sector": ["Energy"],
                "exchange": ["NYSE"],
                "averageChange": [1.25],
            }
        )

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    snapshot = discovery_model.get_sector_performance(
        api_key="KEY", date="2026-10-23", exchange="NYSE"
    )

    assert snapshot.loc["Energy", "Average Change"] == pytest.approx(0.0125)


@pytest.mark.parametrize(
    ("function", "ascending"),
    [
        (discovery_model.get_biggest_gainers, False),
        (discovery_model.get_biggest_losers, True),
        (discovery_model.get_most_active_stocks, None),
    ],
)
def test_movers_changes_are_decimals(monkeypatch, function, ascending):
    def fake(url, user_subscription="Free"):  # noqa: ARG001
        return pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "name": ["A Inc.", "B Inc."],
                "price": [10.0, 20.0],
                "change": [1.0, -2.0],
                "changesPercentage": [11.1111, -9.0909],
                "exchange": ["NASDAQ", "NYSE"],
            }
        )

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    movers = function(api_key="KEY")

    assert movers.loc["A", "Change %"] == pytest.approx(0.111111)
    assert movers.loc["B", "Change %"] == pytest.approx(-0.090909)

    if ascending is not None:
        assert movers["Change %"].is_monotonic_increasing == ascending


def test_free_float_of_all_stocks_is_a_decimal(monkeypatch):
    def fake(url, user_subscription="Free"):  # noqa: ARG001
        return pd.DataFrame(
            {
                "symbol": ["INDI"],
                "date": ["2026-10-07 22:37:55"],
                "freeFloat": [98.543],
                "floatShares": [2.0821e08],
                "outstandingShares": [2.11289e08],
            }
        )

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    shares_float = discovery_model.get_stock_shares_float(api_key="KEY")

    assert shares_float.loc["INDI", "Free Float"] == pytest.approx(0.98543)


def test_index_constituents(monkeypatch):
    requested = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        return pd.DataFrame(
            {
                "symbol": ["MSFT", "AAPL"],
                "name": ["Microsoft", "Apple"],
                "sector": ["Technology"] * 2,
            }
        )

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    constituents = discovery_model.get_index_constituents(
        api_key="KEY", index="Dow Jones"
    )

    assert "dowjones-constituent" in requested[0]
    assert list(constituents.index) == ["AAPL", "MSFT"]
    with pytest.raises(ValueError, match="not available"):
        discovery_model.get_index_constituents(api_key="KEY", index="DAX")


def test_the_screener_sends_fractional_bounds(monkeypatch):
    urls = []

    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        urls.append(url)
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)
    discovery_model.get_stock_screener(
        api_key="KEY", beta_higher=0.8, beta_lower=1.5, dividend_higher=0.25
    )

    assert "betaMoreThan=0.8" in urls[0]
    assert "betaLowerThan=1.5" in urls[0]
    assert "dividendMoreThan=0.25" in urls[0]


def test_free_float_of_all_stocks_collects_every_page(monkeypatch):
    requested = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        page = int(url.split("page=")[1].split("&")[0])
        # Two full pages, a short one and then an empty one.
        size = {0: 5000, 1: 5000, 2: 1}.get(page, 0)
        if size == 0:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "symbol": [f"T{page}_{number}" for number in range(size)],
                "freeFloat": 50.0,
            }
        )

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    shares_float = discovery_model.get_stock_shares_float(api_key="KEY")

    assert len(requested) == 4
    assert all("/stable/shares-float-all" in url for url in requested)
    assert len(shares_float) == 2 * 5000 + 1


def test_free_float_of_all_stocks_stops_when_the_page_is_ignored(monkeypatch):
    requested = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        # The same page whatever page is asked for.
        return pd.DataFrame({"symbol": ["AAA", "BBB"], "freeFloat": 50.0})

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    shares_float = discovery_model.get_stock_shares_float(api_key="KEY")

    assert len(requested) == 2
    assert list(shares_float.index) == ["AAA", "BBB"]


def test_free_float_of_all_stocks_requests_at_most_the_page_cap(monkeypatch):
    requested = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        return pd.DataFrame({"symbol": [f"T{len(requested)}"], "freeFloat": 50.0})

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    discovery_model.get_stock_shares_float(api_key="KEY")

    assert len(requested) == discovery_model.SHARES_FLOAT_MAX_PAGES


def test_names_with_ampersands_are_sent_whole(monkeypatch):
    requested = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    discovery_model.get_instruments(api_key="KEY", query="AT&T")
    discovery_model.get_sector_performance(api_key="KEY", sector="Aerospace & Defense")

    assert "query=AT%26T&apikey=KEY" in requested[0]
    assert "sector=Aerospace%20%26%20Defense&" in requested[1]


def test_the_upper_market_cap_bound_is_sent_in_the_currency_of_the_listings(
    monkeypatch,
):
    urls = []

    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        urls.append(url)
        if "/available-exchanges" in url:
            return pd.DataFrame(
                {"exchange": ["JPX", "NYSE"], "countryCode": ["JP", "US"]}
            )
        if "/batch-quote" in url:
            return pd.DataFrame(
                {"symbol": ["CADUSD", "JPYUSD"], "price": [0.70, 0.005]}
            )
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    def screener_url():
        return next(url for url in reversed(urls) if "/company-screener" in url)

    # One exchange: its currency, widened by the margin (1 billion dollars is 200 billion yen).
    discovery_model.get_stock_screener(
        api_key="KEY", market_cap_lower=1e9, exchange="JPX"
    )
    assert "marketCapLowerThan=400000000000" in screener_url()

    discovery_model.get_stock_screener(
        api_key="KEY", market_cap_lower=1e9, exchange="NYSE"
    )
    assert "marketCapLowerThan=2000000000" in screener_url()

    # The home listings of a country trade in its currency.
    discovery_model.get_stock_screener(
        api_key="KEY", market_cap_lower=1e9, country="Canada", primary_only=True
    )
    assert "marketCapLowerThan=2857142857" in screener_url()

    # Listings anywhere can trade in any currency, so the bound is applied afterwards only.
    discovery_model.get_stock_screener(
        api_key="KEY", market_cap_lower=1e9, country="Canada"
    )
    assert "marketCapLowerThan" not in screener_url()


def test_a_listing_in_the_united_kingdom_is_its_home_listing(monkeypatch):
    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        if "/company-screener" in url:
            return pd.DataFrame(
                {
                    "symbol": ["SHEL", "SHEL.L"],
                    "companyName": ["Shell plc"] * 2,
                    "marketCap": [210e9, 155e9],
                    "exchangeShortName": ["NYSE", "LSE"],
                    "country": ["GB", "GB"],
                    "volume": [4_000_000, 1_000_000],
                }
            )
        if "/available-exchanges" in url:
            # The exchanges write the United Kingdom as UK.
            return pd.DataFrame(
                {"exchange": ["NYSE", "LSE"], "countryCode": ["US", "UK"]}
            )
        if "/batch-quote" in url:
            return pd.DataFrame({"symbol": ["GBPUSD"], "price": [1.26]})
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    every_listing = discovery_model.get_stock_screener(api_key="KEY")
    primary = discovery_model.get_stock_screener(api_key="KEY", primary_only=True)

    assert list(primary.index) == ["SHEL.L"]
    assert every_listing["Market Cap (USD)"].tolist() == pytest.approx([195.3e9] * 2)


def index_with_history(monkeypatch):
    """Today's two members (AAPL, NEW) and a log in which NEW replaced OLD and GONE left."""
    requested = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        if "historical-" in url:
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2010-01-04", "2020-06-01", "2022-03-01"]),
                    "addedTicker": ["AAPL", "NEW", ""],
                    "addedSecurity": ["Apple", "New Co", ""],
                    "removedTicker": ["", "OLD", "GONE"],
                    "removedSecurity": ["", "Old Co", "Gone Co"],
                    "changeType": ["addition", "replacement", "removal"],
                    "reason": ["", "Acquired.", "Market capitalization change."],
                }
            )
        return pd.DataFrame(
            {
                "symbol": ["NEW", "AAPL"],
                "name": ["New Co", "Apple"],
                "sector": ["Technology"] * 2,
                "dateFirstAdded": ["2020-06-01", "1982-11-30"],
            }
        )

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)
    return requested


def test_index_constituent_changes_are_dated_and_complete(monkeypatch):
    requested = index_with_history(monkeypatch)

    changes = discovery_model.get_index_constituent_changes(
        api_key="KEY", index="Dow Jones", start_date="2015-01-01"
    )

    # The endpoint is always asked for a range, as several fall back to old data without one.
    assert "historical-dowjones-constituent" in requested[0]
    assert "from=2015-01-01" in requested[0] and "&to=" in requested[0]
    assert list(changes.index) == ["2020-06-01", "2022-03-01"]
    assert changes.loc["2022-03-01", "Change Type"] == "Removal"
    assert pd.isna(changes.loc["2022-03-01", "Added Ticker"])


def test_the_members_on_a_past_date_undo_every_later_change(monkeypatch):
    index_with_history(monkeypatch)

    before = discovery_model.get_index_constituents(
        api_key="KEY", index="S&P 500", date="2015-01-01"
    )
    # A change takes effect on its own date.
    on_change = discovery_model.get_index_constituents(
        api_key="KEY", index="S&P 500", date="2020-06-01"
    )

    assert list(before.index) == ["AAPL", "GONE", "OLD"]
    assert before.loc["OLD", "Name"] == "Old Co"
    assert before.loc["OLD", "Date Removed"] == "2020-06-01"
    assert before.loc["GONE", "Date Removed"] == "2022-03-01"
    assert pd.isna(before.loc["AAPL", "Date Removed"])
    assert before.loc["AAPL", "Date Added"] == "2010-01-04"
    assert list(on_change.index) == ["AAPL", "GONE", "NEW"]
    assert on_change.loc["NEW", "Sector"] == "Technology"


def test_the_members_before_the_log_starts_are_not_guessed(monkeypatch):
    index_with_history(monkeypatch)

    with pytest.raises(ValueError, match="recorded from 2010-01-04"):
        discovery_model.get_index_constituents(
            api_key="KEY", index="S&P 500", date="2000-01-01"
        )


def test_get_sectors_performance_returns_decimals_by_day(monkeypatch):
    requested: list[str] = []

    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        requested.append(url)
        if "sector=Technology" in url:
            return pd.DataFrame(
                {
                    "date": ["2024-01-03", "2024-01-02"],
                    "sector": "Technology",
                    "averageChange": [-1.5, 2.0],
                }
            )
        if "sector=Energy" in url:
            return pd.DataFrame(
                {"date": ["2024-01-02"], "sector": "Energy", "averageChange": [0.25]}
            )
        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    result = discovery_model.get_sectors_performance(
        "key", start_date="2024-01-01", end_date="2024-01-31"
    )

    # One request per sector, each over the requested range (some endpoints answer
    # with old data without one), and the changes are percentages turned into decimals.
    assert len(requested) == len(discovery_model.FMP_SECTORS)
    assert all("&from=2024-01-01&to=2024-01-31" in url for url in requested)
    assert list(result.columns) == ["Energy", "Technology"]
    assert [str(day) for day in result.index] == ["2024-01-02", "2024-01-03"]
    assert result.loc["2024-01-02", "Technology"] == pytest.approx(0.02)
    assert result.loc["2024-01-03", "Technology"] == pytest.approx(-0.015)
    assert result.loc["2024-01-02", "Energy"] == pytest.approx(0.0025)
    assert pd.isna(result.loc["2024-01-03", "Energy"])


def test_get_sectors_performance_without_any_data_is_empty(monkeypatch):
    monkeypatch.setattr(
        discovery_model,
        "get_cached_financial_data",
        lambda url, user_subscription="Free": pd.DataFrame(),  # noqa: ARG005
    )

    assert discovery_model.get_sectors_performance("key").empty


@pytest.mark.parametrize(
    ("function", "endpoint", "row"),
    [
        (
            "get_crypto_list",
            "cryptocurrency-list",
            {"symbol": "BTCUSD", "name": "Bitcoin", "exchange": "CCC"},
        ),
        (
            "get_index_list",
            "index-list",
            {
                "symbol": "^GSPC",
                "name": "S&P 500",
                "currency": "USD",
                "exchange": "SNP",
            },
        ),
    ],
)
def test_instrument_lists_are_indexed_by_symbol(monkeypatch, function, endpoint, row):
    requested: list[str] = []

    def fake(url, user_subscription="Free"):  # noqa: ARG001
        requested.append(url)
        return pd.DataFrame([row, {**row, "symbol": "AAA", "name": "First"}])

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    result = getattr(discovery_model, function)("key")

    assert f"/stable/{endpoint}?" in requested[0]
    assert result.index.name == "Symbol"
    assert list(result.index) == sorted(["AAA", row["symbol"]])
    assert result.loc[row["symbol"], "Name"] == row["name"]
    assert result.loc[row["symbol"], "Exchange"] == row["exchange"]


def _calendar_responses(monkeypatch, releases_per_range=None):
    """Answer the earnings calendar, quote, exchange and screener endpoints offline."""
    requested: list[str] = []
    releases = pd.DataFrame(
        {
            "symbol": ["CSCO", "CIS.DE", "SIE.DE", "SIEGY", "TINY"],
            "date": [
                "2026-11-12",
                "2026-11-12",
                "2026-11-12",
                "2026-11-12",
                "2026-11-11",
            ],
            "epsEstimated": [1.32, 1.19, 2.86, 1.43, 0.1],
            "revenueEstimated": [18.1e9, 16.2e9, 23.2e9, 26.1e9, 1e6],
        }
    )

    def fake(url, user_subscription="Free", sleep_timer=True):  # noqa: ARG001
        requested.append(url)

        if "/earnings-calendar" in url:
            if releases_per_range is not None:
                return releases_per_range(url)
            return releases
        if "/batch-quote" in url and "EURUSD" in url:
            return pd.DataFrame({"symbol": ["EURUSD"], "price": [1.1]})
        if "/batch-quote" in url:
            return pd.DataFrame(
                {
                    "symbol": ["CSCO", "CIS.DE", "SIE.DE", "SIEGY", "TINY"],
                    "name": [
                        "Cisco Systems, Inc.",
                        "Cisco Systems, Inc.",
                        "Siemens Aktiengesellschaft",
                        "Siemens AG Sponsored ADR",
                        "Tiny Corp.",
                    ],
                    "exchange": ["NASDAQ", "XETRA", "XETRA", "OTC", "NASDAQ"],
                    # The depositary receipt reports a market cap far off the company's.
                    "marketCap": [370e9, 340e9, 200e9, 900e9, 50e6],
                    "price": [93.0, 80.0, 250.0, 125.0, 2.0],
                    "volume": [20e6, 1e4, 1e6, 1e5, 1e5],
                }
            )
        if "/available-exchanges" in url:
            return pd.DataFrame(
                {
                    "exchange": ["NASDAQ", "XETRA", "OTC"],
                    "countryCode": ["US", "DE", "US"],
                }
            )
        if "/company-screener" in url:
            return pd.DataFrame(
                {
                    "symbol": ["SIE.DE", "SIEGY"],
                    "companyName": "Siemens AG",
                    "marketCap": [200e9, 220e9],
                    "exchangeShortName": ["XETRA", "OTC"],
                    "country": "DE",
                }
            )

        return pd.DataFrame()

    monkeypatch.setattr(discovery_model, "get_cached_financial_data", fake)

    return requested


def test_the_earnings_calendar_ranks_companies_by_their_primary_listing(monkeypatch):
    _calendar_responses(monkeypatch)

    calendar = discovery_model.get_earnings_calendar(
        "KEY", "2026-11-09", "2026-11-13", primary_only=True, sort_by="market_cap"
    )

    # One row per company, its most traded listing, largest first.
    assert list(calendar.index) == ["CSCO", "SIE.DE", "TINY"]
    # The depositary receipt's market cap is not the company's.
    assert calendar.loc["SIE.DE", "Market Cap (USD)"] == pytest.approx(220e9)


def test_every_listing_has_the_market_cap_of_the_primary_listing(monkeypatch):
    _calendar_responses(monkeypatch)

    calendar = discovery_model.get_earnings_calendar("KEY", "2026-11-09", "2026-11-13")

    assert list(calendar.index) == ["TINY", "CIS.DE", "CSCO", "SIE.DE", "SIEGY"]
    assert calendar.loc["CIS.DE", "Market Cap (USD)"] == pytest.approx(370e9)
    assert calendar.loc["SIEGY", "Market Cap (USD)"] == pytest.approx(220e9)


def test_the_earnings_calendar_filters_by_size_country_and_exchange(monkeypatch):
    _calendar_responses(monkeypatch)

    def symbols(**filters):
        return set(
            discovery_model.get_earnings_calendar(
                "KEY", "2026-11-09", "2026-11-13", **filters
            ).index
        )

    assert symbols(min_market_cap=1e9) == {"CSCO", "CIS.DE", "SIE.DE", "SIEGY"}
    # The country of the company, so a German company's listing abroad is kept.
    assert symbols(country="Germany") == {"SIE.DE", "SIEGY"}
    assert symbols(exchange="xetra") == {"CIS.DE", "SIE.DE"}


def test_the_earnings_calendar_splits_a_range_that_reaches_the_limit(monkeypatch):
    def releases(url):
        start = url.split("from=")[1][:10]
        end = url.split("to=")[1][:10]
        days = pd.date_range(start, end)
        if len(days) == 5:
            # The whole week reaches the limit and keeps only its latest releases.
            return pd.DataFrame(
                {
                    "symbol": [
                        f"X{i}" for i in range(discovery_model.EARNINGS_CALENDAR_LIMIT)
                    ],
                    "date": end,
                }
            )
        return pd.DataFrame(
            {
                "symbol": [f"S{day:%d}" for day in days],
                "date": [f"{day:%Y-%m-%d}" for day in days],
            }
        )

    requested = _calendar_responses(monkeypatch, releases)

    calendar = discovery_model.get_earnings_calendar("KEY", "2026-11-09", "2026-11-13")

    ranges = [url for url in requested if "/earnings-calendar" in url]
    assert len(ranges) == 3
    assert sorted(calendar.index) == ["S09", "S10", "S11", "S12", "S13"]


def test_the_earnings_calendar_always_asks_for_a_range(monkeypatch):
    requested = _calendar_responses(monkeypatch)

    discovery_model.get_earnings_calendar("KEY", end_date="2026-11-13")

    assert "from=2026-11-06&to=2026-11-13" in requested[0]
