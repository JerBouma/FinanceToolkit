"""World Bank Commodity Price Data (The Pink Sheet) Model"""

__docformat__ = "google"

import re

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.excel_model import read_excel
from financetoolkit.utilities.requests_model import get_request

# The World Bank publishes the monthly commodity prices of "The Pink Sheet" as a workbook
# whose link changes every year, so it is read from the commodity markets page.
PAGE_URL = "https://www.worldbank.org/en/research/commodity-markets"
FILE_PATTERN = re.compile(
    r'(https://thedocs\.worldbank\.org/[^"\']+/CMO-Historical-Data-Monthly\.xlsx)'
)

# The worksheet with the prices: the names are in the fifth row, the units (described in
# Economics.get_commodity_prices) in the sixth and the months, written as 1960M01, in the
# first column from the seventh row on.
SHEET_NAME = "Monthly Prices"
NAME_ROW = 4
FIRST_ROW = 6


def clean_commodity_name(name: object) -> str:
    """
    Returns the name of a commodity as the Finance Toolkit gives it: the World Bank's name
    without its commas and footnote marks, so that names can be listed comma-separated,
    e.g. "Crude oil, Brent" becomes "Crude oil Brent" and "Beef **" becomes "Beef".

    Args:
        name (object): The name as the World Bank publishes it.

    Returns:
        str: The cleaned name.
    """
    return " ".join(str(name).replace("*", "").replace(",", " ").split())


def get_commodity_prices() -> pd.DataFrame:
    """
    Retrieves the World Bank's monthly commodity prices (The Pink Sheet) from 1960: the
    average price of each month in nominal US dollars of around seventy commodities,
    among which crude oil (Brent, WTI, Dubai), natural gas, coal, metals (gold, silver,
    platinum, copper, aluminum, iron ore, nickel, zinc), grains, softs and fertilizers.

    Returns:
        pd.DataFrame: One column per commodity, named as in clean_commodity_name, indexed
        by month.

    Raises:
        ValueError: When the page links no data file or the file's layout changed.
    """
    description = "World Bank commodity prices"

    def fetch() -> pd.DataFrame:
        page = get_request(PAGE_URL, timeout=60).text
        link = FILE_PATTERN.search(page)

        if link is None:
            raise ValueError(
                f"The {description} page links no CMO-Historical-Data-Monthly.xlsx file, "
                "which means the site changed."
            )

        sheet = read_excel(
            get_request(link.group(1), timeout=120).content,
            sheet_name=SHEET_NAME,
            header=None,
        )
        months = sheet.iloc[FIRST_ROW:, 0].astype(str).str.strip()

        if not months.str.fullmatch(r"\d{4}M\d{2}").all():
            raise ValueError(
                f"The {description} file has no months such as 1960M01 in its first column "
                "where they are expected, which means its layout changed."
            )

        data = sheet.iloc[FIRST_ROW:, 1:].apply(pd.to_numeric, errors="coerce")
        data.columns = [clean_commodity_name(name) for name in sheet.iloc[NAME_ROW, 1:]]
        data.index = pd.PeriodIndex(months.str.replace("M", "-"), freq="M")
        data.index.name = None

        return data.dropna(how="all")

    return collect_cached_data(
        source=policy_model.WORLD_BANK,
        dataset="commodity_prices",
        entity="pink_sheet",
        fetch=fetch,
        description=description,
    )
