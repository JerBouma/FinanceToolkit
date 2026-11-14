"""US Treasury Fiscal Data Auction Model"""

__docformat__ = "google"

from datetime import datetime
from urllib.parse import urlencode

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

# The US Treasury's Fiscal Data API publishes the announcement and results of every
# Treasury auction from 1979 on, without an API key.
AUCTIONS_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/"
    "auctions_query"
)
FIRST_YEAR = 1979
PAGE_SIZE = 1000

# The fields kept, with the names the Finance Toolkit gives them. Rates are published in
# percent and amounts in dollars.
FIELDS = {
    "auction_date": "Auction Date",
    "security_type": "Security Type",
    "security_term": "Security Term",
    "cusip": "CUSIP",
    "issue_date": "Issue Date",
    "maturity_date": "Maturity Date",
    "reopening": "Reopening",
    "int_rate": "Coupon Rate",
    "high_yield": "High Yield",
    "high_investment_rate": "High Investment Rate",
    "high_discnt_rate": "High Discount Rate",
    "bid_to_cover_ratio": "Bid-to-Cover Ratio",
    "offering_amt": "Offering Amount",
    "total_tendered": "Total Tendered",
    "total_accepted": "Total Accepted",
    "primary_dealer_accepted": "Primary Dealers Accepted",
    "direct_bidder_accepted": "Direct Bidders Accepted",
    "indirect_bidder_accepted": "Indirect Bidders Accepted",
}
PERCENT_FIELDS = [
    "Coupon Rate",
    "High Yield",
    "High Investment Rate",
    "High Discount Rate",
]
AMOUNT_FIELDS = [
    "Offering Amount",
    "Total Tendered",
    "Total Accepted",
    "Primary Dealers Accepted",
    "Direct Bidders Accepted",
    "Indirect Bidders Accepted",
]


def _security_type(record: pd.DataFrame) -> pd.Series:
    """
    The kind of security: Bill, Cash Management Bill, Note, Bond, TIPS or FRN. The Treasury
    files inflation-protected securities and floating rate notes under Note and Bond, with a
    flag for each.
    """
    security_type = record["security_type"].astype("string")
    flags = {
        "cash_management_bill_cmb": "Cash Management Bill",
        "inflation_index_security": "TIPS",
        "floating_rate": "FRN",
    }
    for field, name in flags.items():
        if field in record:
            security_type = security_type.mask(record[field].eq("Yes"), name)
    return security_type


def _get_year(year: int) -> pd.DataFrame:
    """
    Retrieves the auctions of one year. A past year no longer changes once its results are
    published, so it is cached under its own entry and only the current year is requested
    again.

    Args:
        year (int): The year to retrieve.

    Returns:
        pd.DataFrame: One row per auction with the fields in FIELDS.
    """
    description = f"{year} Treasury auction results"

    def fetch() -> pd.DataFrame:
        records: list[dict] = []
        page = 1
        while True:
            query = urlencode(
                {
                    "filter": f"auction_date:gte:{year}-01-01,auction_date:lte:{year}-12-31",
                    "sort": "auction_date",
                    "page[size]": PAGE_SIZE,
                    "page[number]": page,
                }
            )
            response = get_request(f"{AUCTIONS_URL}?{query}", timeout=60).json()
            data = response.get("data", [])
            records.extend(data)
            total_pages = int(response.get("meta", {}).get("total-pages", 1) or 1)
            if page >= total_pages or not data:
                break
            page += 1

        if not records:
            return pd.DataFrame()

        frame = pd.DataFrame(records).replace("null", pd.NA)
        require_columns(frame, {"auction_date", "security_type", "cusip"}, description)
        frame["security_type"] = _security_type(frame)

        frame = frame.reindex(columns=list(FIELDS)).rename(columns=FIELDS)
        for column in PERCENT_FIELDS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100
        for column in [*AMOUNT_FIELDS, "Bid-to-Cover Ratio"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.reset_index(drop=True)

    return collect_cached_data(
        source=policy_model.US_TREASURY,
        dataset="auctions" if year == datetime.now().year else "auctions_year",
        entity=str(year),
        fetch=fetch,
        description=description,
        expect_empty=True,
    )


def get_treasury_auctions(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the US Treasury auctions (bills, notes, bonds, TIPS and floating rate notes)
    held between two dates from the Treasury's Fiscal Data API, one request per year.
    Auctions that are announced but not yet held have no results.

    Args:
        start_date (str): The first auction date (YYYY-MM-DD).
        end_date (str): The last auction date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: One row per auction with the fields in FIELDS, sorted by auction date;
        rates as decimals and amounts in dollars.
    """
    current_year = datetime.now().year
    years = list(
        range(
            max(FIRST_YEAR, int(start_date[:4])),
            min(current_year, int(end_date[:4])) + 1,
        )
    )
    if not years:
        return pd.DataFrame(columns=list(FIELDS.values()))

    frames = [
        frame
        for frame in helpers.run_in_parallel(_get_year, [(year,) for year in years])
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=list(FIELDS.values()))

    auctions = pd.concat(frames, ignore_index=True)
    dates = auctions["Auction Date"].astype(str)
    auctions = auctions[(dates >= start_date) & (dates <= end_date)]
    return auctions.sort_values("Auction Date", kind="stable").reset_index(drop=True)
