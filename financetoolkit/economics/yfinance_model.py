"""Yahoo Finance Model"""

__docformat__ = "google"

from datetime import datetime

import pandas as pd
import yfinance as yf

# CME futures month codes, one letter per calendar month.
_MONTH_CODES: dict[int, str] = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}

# (root symbol, exchange suffix) for each supported commodity, matching Yahoo
# Finance's dated futures contract ticker convention (e.g. "CLZ26.NYM" for the
# December 2026 Crude Oil contract).
COMMODITY_TICKERS: dict[str, tuple[str, str]] = {
    "Crude Oil": ("CL", "NYM"),
    "Natural Gas": ("NG", "NYM"),
    "Gold": ("GC", "CMX"),
    "Silver": ("SI", "CMX"),
    "Copper": ("HG", "CMX"),
    "Corn": ("ZC", "CBT"),
    "Wheat": ("ZW", "CBT"),
    "Soybeans": ("ZS", "CBT"),
}


def get_commodity_forward_curve(
    commodity: str,
    start_date: str,
    end_date: str,
    contracts: int = 12,
) -> pd.DataFrame:
    """
    Retrieves the forward/futures curve for a commodity from Yahoo Finance -- the
    historical daily closing price of each dated futures contract over the next
    `contracts` calendar months (e.g. Crude Oil's December 2026, January 2027, ...
    contracts), rather than a single flat continuous/spot price such as `CL=F`.

    This is what a Schwartz-Smith (2000) two-factor commodity price model needs to
    back out the convenience-yield term structure under the risk-neutral (Q)
    measure -- the curve's shape (contango or backwardation) at each point in time
    is exactly what a single spot price series cannot reveal.

    Not every commodity has a listed contract for every calendar month (grains in
    particular only trade specific delivery months), so months with no listed
    contract are silently skipped -- the number of columns returned can be fewer
    than `contracts`.

    Args:
        commodity (str): The commodity to retrieve the curve for. One of "Crude Oil",
            "Natural Gas", "Gold", "Silver", "Copper", "Corn", "Wheat" or "Soybeans".
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        contracts (int, optional): The number of sequential monthly contracts ahead
            of today to attempt to fetch. Defaults to 12.

    Raises:
        ValueError: If `commodity` is not one of the supported names.

    Returns:
        pd.DataFrame: A DataFrame indexed by date, with one column per contract
            labeled by its delivery month (e.g. "2026-12"), containing that
            contract's daily closing price over its trading life. Columns are NaN
            outside the date range the contract actually traded in.
    """
    if commodity not in COMMODITY_TICKERS:
        raise ValueError(
            f"Commodity '{commodity}' is not supported. Choose one of: "
            f"{', '.join(COMMODITY_TICKERS)}."
        )

    root, exchange = COMMODITY_TICKERS[commodity]
    today = datetime.today()

    curve: dict[str, pd.Series] = {}
    for months_ahead in range(contracts):
        month_index = today.month - 1 + months_ahead
        year = today.year + month_index // 12
        month = month_index % 12 + 1

        ticker = f"{root}{_MONTH_CODES[month]}{year % 100:02d}.{exchange}"
        contract_data = yf.Ticker(ticker).history(start=start_date, end=end_date)

        if contract_data.empty:
            continue

        curve[f"{year}-{month:02d}"] = contract_data["Close"]

    if not curve:
        return pd.DataFrame()

    forward_curve = pd.DataFrame(curve)
    forward_curve.index = pd.to_datetime(forward_curve.index).tz_localize(None)
    forward_curve.index = forward_curve.index.to_period(freq="D")
    forward_curve.index.name = "Date"
    forward_curve = forward_curve.sort_index()

    return forward_curve
