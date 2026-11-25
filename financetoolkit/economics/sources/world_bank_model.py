"""World Bank Model: Commodity Price Data (The Pink Sheet) and World Development Indicators"""

__docformat__ = "google"

import re

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import COUNTRY_CODES, collect_cached_data
from financetoolkit.economics.sources.oecd_model import CODE_TO_COUNTRY
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

# The columns of the worksheet that are an index (2010 = 100) rather than a price in US
# dollars (or US cents, for plywood), so they are not converted into another currency.
INDEX_COMMODITIES = ["Natural gas index"]


def clean_commodity_name(name: object) -> str:
    """
    Return the name of a commodity as the Finance Toolkit gives it: the World Bank's name without its commas and
    footnote marks, so names can be listed comma-separated.

    For example, "Crude oil, Brent" becomes "Crude oil Brent" and "Beef **" becomes "Beef".

    Args:
        name (object): The name as the World Bank publishes it.

    Returns:
        str: The cleaned name.
    """
    return " ".join(str(name).replace("*", "").replace(",", " ").split())


def get_commodity_prices() -> pd.DataFrame:
    """
    Retrieve the World Bank's monthly commodity prices (The Pink Sheet) from 1960: the average price of each month in
    nominal US dollars of around seventy commodities.

    Among them are crude oil (Brent, WTI, Dubai), natural gas, coal, metals (gold, silver, platinum, copper, aluminum,
    iron ore, nickel, zinc), grains, softs and fertilizers.

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
            description,
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


# The World Development Indicators API, which answers every economy and aggregate of an
# indicator in pages of JSON, and the list of economies, in which the aggregates (regions,
# income groups and lending groups) have the region "Aggregates".
INDICATOR_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    "?format=json&per_page=20000&page={page}"
)
COUNTRIES_URL = (
    "https://api.worldbank.org/v2/country?format=json&per_page=1000&page={page}"
)
AGGREGATES_REGION = "Aggregates"

# The aggregates kept, named as the other sources name them; the World Bank's own name
# for the rest of the world is "World", as the OECD's.
KEPT_AGGREGATES = {"EMU": "Euro Area", "EUU": "European Union", "WLD": "World"}

# The economies that neither the OECD nor the two letter code lists of the Finance Toolkit
# know, and that the World Bank names differently from the Global Macro Database ("St.
# Lucia" against "Saint Lucia"), named as the Global Macro Database does so an economy is
# one column across the Economics module. Saint Martin is not in its data; it is named as
# its list of countries does.
WORLD_BANK_TO_COUNTRY_NAMES = {
    "CUW": "Curaçao",
    "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia",
    "MAF": "Saint Martin",
    "NRU": "Nauru",
    "PRI": "Puerto Rico",
    "SXM": "Sint Maarten",
    "TCA": "Turks and Caicos",
    "VCT": "St-Vincent",
    "VIR": "US Virgin Islands",
}


def _read_pages(url: str, description: str) -> list[dict]:
    """
    Reads every page of a World Bank API answer, which is a list of the page information
    followed by the records of that page.

    Args:
        url (str): The URL with a {page} field.
        description (str): What is retrieved, used in the error message.

    Returns:
        list[dict]: The records of all pages.

    Raises:
        ValueError: When an answer is an error message or not in the expected format.
    """
    records: list[dict] = []
    page, pages = 1, 1

    while page <= pages:
        answer = get_request(url.format(page=page), timeout=120).json()

        # An unknown indicator is answered with a single message instead of the records.
        if (
            not isinstance(answer, list)
            or len(answer) != 2  # noqa: PLR2004
            or "pages" not in answer[0]
        ):
            raise ValueError(
                f"The World Bank answer for the {description} is not a page of records: "
                f"{str(answer)[:200]}"
            )

        pages = int(answer[0]["pages"])
        records.extend(answer[1] or [])
        page += 1

    return records


def get_economy_names() -> pd.DataFrame:
    """
    List the economies of the World Development Indicators, named as elsewhere in the Finance Toolkit where it knows
    the country, otherwise as the World Bank names it.

    A country is known by its three or two letter ISO 3166 code or as the Global Macro Database names it (see
    WORLD_BANK_TO_COUNTRY_NAMES); other names lose the qualifier after a comma ("Bahamas, The" becomes "Bahamas"). Of
    the aggregates, the euro area, the European Union and the world are kept.

    Returns:
        pd.DataFrame: The name ("Country") per three letter code.
    """
    description = "World Bank economies"

    def fetch() -> pd.DataFrame:
        names = {}

        for economy in _read_pages(COUNTRIES_URL, description):
            code = economy["id"]

            if code in KEPT_AGGREGATES:
                names[code] = KEPT_AGGREGATES[code]
            elif economy["region"]["value"] != AGGREGATES_REGION:
                names[code] = CODE_TO_COUNTRY.get(
                    code,
                    COUNTRY_CODES.get(
                        economy["iso2Code"],
                        WORLD_BANK_TO_COUNTRY_NAMES.get(
                            code, economy["name"].split(",", maxsplit=1)[0].strip()
                        ),
                    ),
                )

        if not names:
            raise ValueError(f"The list of {description} is empty.")

        return pd.DataFrame({"Country": list(names.values())}, index=list(names))

    return collect_cached_data(
        source=policy_model.WORLD_BANK,
        dataset="economies",
        entity="countries",
        fetch=fetch,
        description=description,
    )


def get_indicator(indicator: str, description: str) -> pd.DataFrame:
    """
    Retrieves a yearly World Development Indicator for every economy, such as total
    reserves (FI.RES.TOTL.CD) or total reserves in months of imports (FI.RES.TOTL.MO).

    Args:
        indicator (str): The indicator code.
        description (str): What is retrieved, used in the log and error messages.

    Returns:
        pd.DataFrame: The values as published, indexed by year with a column per economy
        (see get_economy_names); the other aggregates are left out.

    Raises:
        ValueError: When the answer is not in the expected format.
    """

    def fetch() -> pd.DataFrame:
        records = _read_pages(
            INDICATOR_URL.replace("{indicator}", indicator), description
        )

        try:
            observations = pd.DataFrame(
                {
                    "code": [record["countryiso3code"] for record in records],
                    "year": [record["date"] for record in records],
                    "value": [record["value"] for record in records],
                }
            )
        except KeyError as error:
            raise ValueError(
                f"The World Bank records of the {description} miss the field {error}, "
                "which means the API changed."
            ) from error

        # The income groups are published without a three letter code; they are
        # aggregates, which are left out.
        observations = observations[observations["code"] != ""].dropna(subset=["value"])

        if observations.empty:
            return pd.DataFrame()

        data = observations.pivot(index="year", columns="code", values="value")
        data.index = pd.PeriodIndex(data.index.astype(str), freq="Y")
        data.index.name = None
        data.columns.name = None

        return data.sort_index().astype(float)

    data = collect_cached_data(
        source=policy_model.WORLD_BANK,
        dataset="indicator",
        entity=indicator,
        fetch=fetch,
        description=description,
    )

    economies = get_economy_names()

    # Without the list of economies (already logged), the aggregates cannot be told apart.
    if data.empty or economies.empty:
        return pd.DataFrame()

    names = economies["Country"]
    data = data[[code for code in data.columns if code in names.index]]

    return data.rename(columns=names.to_dict())
