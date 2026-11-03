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
