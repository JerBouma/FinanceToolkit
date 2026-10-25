"""US Treasury Fiscal Data Auction Model Tests"""

from datetime import datetime

import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.fixedincome import fiscaldata_model

# pylint: disable=missing-function-docstring


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    """Start every test without an active cache, so the fake responses are used."""
    clear_active_cache()
    yield
    clear_active_cache()


class FakeResponse:
    """The parts of a requests.Response the model uses."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def auction(date, security_type="Note", term="10-Year", **fields):
    record = {
        "auction_date": date,
        "security_type": security_type,
        "security_term": term,
        "cusip": f"CUSIP{date}",
        "issue_date": date,
        "maturity_date": "2036-08-15",
        "reopening": "No",
        "int_rate": "4.250",
        "high_yield": "4.312",
        "high_investment_rate": "null",
        "high_discnt_rate": "null",
        "bid_to_cover_ratio": "2.45",
        "offering_amt": "42000000000",
        "total_tendered": "102900000000",
        "total_accepted": "42000000000",
        "primary_dealer_accepted": "5000000000",
        "direct_bidder_accepted": "8000000000",
        "indirect_bidder_accepted": "27000000000",
        "inflation_index_security": "No",
        "floating_rate": "No",
        "cash_management_bill_cmb": "No",
    }
    record.update(fields)
    return record


def test_auctions_are_converted_and_classified(monkeypatch):
    year = datetime.now().year
    records = [
        auction(f"{year}-01-08"),
        auction(f"{year}-01-15", "Note", "10-Year", inflation_index_security="Yes"),
        auction(
            f"{year}-01-22", "Note", "2-Year", floating_rate="Yes", high_yield="null"
        ),
        auction(
            f"{year}-01-29",
            "Bill",
            "6-Week",
            cash_management_bill_cmb="Yes",
            int_rate="null",
            high_yield="null",
            high_investment_rate="4.132",
            high_discnt_rate="4.050",
        ),
        auction(f"{year}-02-05", bid_to_cover_ratio="null", total_accepted="null"),
    ]
    requests = []

    def fake_request(url, timeout):
        requests.append(url)
        page = 1 if "page%5Bnumber%5D=1" in url else 2
        data = records[:3] if page == 1 else records[3:]
        return FakeResponse({"data": data, "meta": {"total-pages": 2}})

    monkeypatch.setattr(fiscaldata_model, "get_request", fake_request)

    auctions = fiscaldata_model.get_treasury_auctions(f"{year}-01-01", f"{year}-01-31")

    # Two pages were requested, and the auction after the end date is left out.
    assert len(requests) == 2
    assert list(auctions["Auction Date"]) == [
        f"{year}-01-08",
        f"{year}-01-15",
        f"{year}-01-22",
        f"{year}-01-29",
    ]
    assert list(auctions["Security Type"]) == [
        "Note",
        "TIPS",
        "FRN",
        "Cash Management Bill",
    ]
    assert auctions.loc[0, "High Yield"] == pytest.approx(0.04312)
    assert auctions.loc[0, "Coupon Rate"] == pytest.approx(0.0425)
    assert auctions.loc[0, "Offering Amount"] == 42_000_000_000
    assert pd.isna(auctions.loc[2, "High Yield"])
    assert auctions.loc[3, "High Investment Rate"] == pytest.approx(0.04132)


def test_years_without_auctions(monkeypatch):
    monkeypatch.setattr(
        fiscaldata_model,
        "get_request",
        lambda url, timeout: FakeResponse({"data": [], "meta": {"total-pages": 0}}),
    )

    assert fiscaldata_model.get_treasury_auctions("1950-01-01", "1960-12-31").empty
    assert fiscaldata_model.get_treasury_auctions("1979-01-01", "1979-12-31").empty
