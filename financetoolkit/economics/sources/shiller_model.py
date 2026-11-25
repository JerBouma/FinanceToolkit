"""Robert Shiller US Stock Market Data Model"""

__docformat__ = "google"

import re

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.excel_model import read_excel
from financetoolkit.utilities.requests_model import get_request

# The site where Robert Shiller publishes the monthly data behind "Irrational Exuberance".
# The file's link carries a version identifier that changes with every update, so it is
# read from the page.
PAGE_URL = "https://shillerdata.com/"
FILE_PATTERN = re.compile(r'href="((?:https?:)?//[^"]+/ie_data\.xls[^"]*)"')

# The columns of the "Data" worksheet, by position, with the names the Finance Toolkit
# gives them, and the first row with an observation.
COLUMNS = {
    0: "Date",
    1: "Price",
    2: "Dividend",
    3: "Earnings",
    4: "Consumer Price Index",
    6: "Long Interest Rate",
    7: "Real Price",
    8: "Real Dividend",
    9: "Real Total Return Price",
    10: "Real Earnings",
    12: "CAPE",
    14: "Total Return CAPE",
}
FIRST_ROW = 8
HEADER_TEXT = "Date"


def _to_month(value: float) -> pd.Period:
    """
    Converts a date as the data writes it, the year and month as a decimal such as 1871.01
    for January and 2023.1 for October, into a month.

    Args:
        value (float): The date.

    Returns:
        pd.Period: The month.
    """
    year = int(value)
    month = round((value - year) * 100)

    return pd.Period(year=year, month=month, freq="M")


def get_stock_market_data() -> pd.DataFrame:
    """
    Retrieve Robert Shiller's monthly US stock market data from 1871: the S&P Composite price, dividends, earnings,
    interest rates and the CAPE ratio, nominal and real.

    Next to the price, dividends, earnings, the consumer price index and the long-term interest rate come the trailing
    price-earnings ratio (P/E) and the cyclically adjusted price-earnings ratio (CAPE).

    Returns:
        pd.DataFrame: One column per measure, indexed by month. The long interest rate is a
        decimal; the other columns are levels as published.

    Raises:
        ValueError: When the page links no data file or the file's layout changed.
    """
    description = "Shiller stock market data"

    def fetch() -> pd.DataFrame:
        page = get_request(PAGE_URL, timeout=60).text
        link = FILE_PATTERN.search(page)

        if link is None:
            raise ValueError(
                f"The {description} page links no ie_data.xls file, which means the site changed."
            )

        url = (
            link.group(1)
            if link.group(1).startswith("http")
            else f"https:{link.group(1)}"
        )
        sheet = read_excel(
            get_request(url, timeout=120).content,
            description,
            sheet_name="Data",
            header=None,
        )

        if str(sheet.iloc[FIRST_ROW - 1, 0]).strip() != HEADER_TEXT:
            raise ValueError(
                f"The {description} file has no '{HEADER_TEXT}' header where it is expected, "
                "which means its layout changed."
            )

        rows = sheet.iloc[FIRST_ROW:]
        rows = rows[pd.to_numeric(rows[0], errors="coerce").notna()]
        data = (
            rows[list(COLUMNS)]
            .rename(columns=COLUMNS)
            .apply(pd.to_numeric, errors="coerce")
        )

        data.index = pd.PeriodIndex(
            [_to_month(value) for value in data.pop("Date")], freq="M"
        )
        data.index.name = None

        # The long interest rate is published in percent.
        data["Long Interest Rate"] = data["Long Interest Rate"] / 100

        return data.dropna(how="all")

    data = collect_cached_data(
        source=policy_model.SHILLER,
        dataset="stock_market_data",
        entity="ie_data",
        fetch=fetch,
        description=description,
    )

    if data.empty:
        return data

    # The trailing P/E: the month's price over the earnings of the twelve months up to it,
    # next to the CAPE's ten-year average; without positive earnings it has no value. It is
    # derived after the cache, so a cached copy of the file gets it as well.
    data.insert(
        data.columns.get_loc("CAPE"),
        "P/E",
        data["Price"] / data["Earnings"].where(data["Earnings"] > 0),
    )

    return data
