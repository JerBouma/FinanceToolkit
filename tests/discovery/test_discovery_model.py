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
