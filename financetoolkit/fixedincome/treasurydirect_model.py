"""TreasuryDirect FedInvest Price Model"""

__docformat__ = "google"

import re
from datetime import date, datetime, timedelta
from html import unescape

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS
from financetoolkit.utilities.sources_model import record_url

logger = logger_model.get_logger()

# The Bureau of the Fiscal Service publishes a price for every marketable Treasury each
# business day on its FedInvest page, without an API key: one form per date, which answers
# with a table of the securities.
PRICES_URL = "https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate"

# The columns of the price table, with the names the Finance Toolkit gives them.
COLUMNS = [
    "CUSIP",
    "Security Type",
    "Coupon",
    "Maturity",
    "Call Date",
    "Buy",
    "Sell",
    "End of Day",
]
SECURITY_TYPES = {
    "MARKET BASED BILL": "Bill",
    "MARKET BASED NOTE": "Note",
    "MARKET BASED BOND": "Bond",
    "MARKET BASED FRN": "FRN",
    "TIPS": "TIPS",
}

# The prices of a day are published in the afternoon and completed with the end-of-day
# price the next business day, so a weekend or holiday is skipped by trying the days
# before it.
LOOKBACK_DAYS = 7

ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
TOKEN = re.compile(r'name="_csrf"\s+value="([^"]+)"')


def _parse_prices(html: str) -> pd.DataFrame:
    """
    Reads the price table of the FedInvest page.

    Args:
        html (str): The page.

    Returns:
        pd.DataFrame: One row per security with the columns in COLUMNS, empty for a day
        without prices.
    """
    rows = []
    for row in ROW.findall(html):
        cells = [
            unescape(re.sub(r"<[^>]+>", "", cell)).strip() for cell in CELL.findall(row)
        ]
        if len(cells) == len(COLUMNS):
            rows.append(cells)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    prices = pd.DataFrame(rows, columns=COLUMNS)
    require_columns(prices, set(COLUMNS), "FedInvest Treasury prices")

    prices["Security Type"] = prices["Security Type"].map(
        lambda value: SECURITY_TYPES.get(value.upper(), value.title())
    )
    prices["Coupon"] = (
        pd.to_numeric(prices["Coupon"].str.rstrip("%"), errors="coerce") / 100
    )
    for column in ["Maturity", "Call Date"]:
        prices[column] = pd.to_datetime(
            prices[column], format="%m/%d/%Y", errors="coerce"
        )
    for column in ["Buy", "Sell", "End of Day"]:
        # The page shows 0 for a price that is not (yet) published.
        prices[column] = pd.to_numeric(prices[column], errors="coerce").where(
            lambda values: values > 0
        )

    return prices


def _get_day(price_date: date) -> pd.DataFrame:
    """
    Retrieves the prices of one day. A day whose end-of-day prices are published no longer
    changes, so it is cached for good; a day still waiting for them is asked for again
    after an hour.

    Args:
        price_date (date): The day.

    Returns:
        pd.DataFrame: The prices of the day, see _parse_prices.
    """

    def fetch() -> pd.DataFrame:
        # The form carries a token tied to the session, so the page is opened first and
        # the form sent with the token, as a browser does. The Finance Toolkit names
        # itself in the User-Agent.
        with requests.Session() as session:
            page = session.get(PRICES_URL, headers=TOOLKIT_HEADERS, timeout=60)
            page.raise_for_status()
            token = TOKEN.search(page.text)
            if not token:
                raise ValueError(
                    "The FedInvest price page no longer has the form the Treasury prices are "
                    "requested with."
                )
            response = session.post(
                PRICES_URL,
                data={
                    "priceDate": price_date.isoformat(),
                    "_csrf": token.group(1),
                    "submit": "Show Prices",
                },
                headers=TOOLKIT_HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            record_url(PRICES_URL)
        return _parse_prices(response.text)

    cache = get_active_cache()
    entity = price_date.isoformat()

    if cache is not None:
        final = cache.get(
            source=policy_model.US_TREASURY,
            dataset="fedinvest_prices_final",
            entity=entity,
        )
        if final is not None:
            return final

    prices = collect_cached_data(
        source=policy_model.US_TREASURY,
        dataset="fedinvest_prices",
        entity=entity,
        fetch=fetch,
        description=f"Treasury prices of {price_date}",
        expect_empty=True,
    )

    # Once the end-of-day prices are in, the day no longer changes and is kept for good.
    if cache is not None and not prices.empty and prices["End of Day"].notna().any():
        cache.set(
            source=policy_model.US_TREASURY,
            dataset="fedinvest_prices_final",
            entity=entity,
            data=prices,
        )

    return prices


def get_treasury_prices(price_date: str | None = None) -> tuple[pd.DataFrame, date]:
    """
    Retrieves the prices of every marketable Treasury (bills, notes, bonds, TIPS and floating
    rate notes) on a day, or on the last business day before it that has prices.

    The price is the end-of-day price, or before that is published (the next business day)
    the Sell price, the bid. Both are clean prices per 100 of par; TIPS are priced per 100 of
    inflation-adjusted principal.

    Args:
        price_date (str | None): The day (YYYY-MM-DD). Defaults to None, today.

    Returns:
        tuple[pd.DataFrame, date]: One row per security with its CUSIP, Security Type,
        Coupon (decimal), Maturity, Call Date and Price, and the day the prices are from.

    Raises:
        ValueError: When no day in the week up to price_date has prices.
    """
    day = (
        datetime.strptime(price_date, "%Y-%m-%d").date()
        if price_date
        else datetime.now().date()
    )

    for offset in range(LOOKBACK_DAYS + 1):
        candidate = day - timedelta(days=offset)
        if candidate.weekday() >= 5:  # noqa: PLR2004
            continue

        prices = _get_day(candidate)
        if prices.empty:
            continue

        prices = prices.assign(Price=prices["End of Day"].fillna(prices["Sell"]))
        return (
            prices.drop(columns=["Buy", "Sell", "End of Day"]).dropna(subset=["Price"]),
            candidate,
        )

    raise ValueError(
        f"FedInvest has no Treasury prices for {day} or the {LOOKBACK_DAYS} days before it."
    )
