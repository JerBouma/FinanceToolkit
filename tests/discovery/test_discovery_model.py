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
