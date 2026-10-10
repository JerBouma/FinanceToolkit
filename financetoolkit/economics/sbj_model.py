"""Statistics Bureau of Japan Model"""

__docformat__ = "google"

import io
import re

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://www.stat.go.jp/data/cpi/"

# The page that links the CPI files of the current base years.
INDEX_PAGE = "1.html"

# The CPI is rebased every five years and each base has its own file. The base years
# linked from the index page are used, together with these, so a new base is picked up
# without a change here and the older ones remain available after the page drops them.
KNOWN_BASE_YEARS = [2015, 2020, 2025]

FILE_PATTERN = re.compile(r"zmi(\d{4})aa\.csv")

NOT_FOUND = 404

# Rows with observations start with the month, written as 202608.
MONTH_PATTERN = re.compile(r"^\d{6}$")

COUNTRY = "Japan"


def _read_base_file(base_year: int) -> pd.Series | None:
    """
    Reads the all-items index of one CPI base from its CSV file.

    Args:
        base_year (int): The base year of the file, e.g. 2020.

    Returns:
        pd.Series | None: The monthly all-items index, or None when there is no such file.

    Raises:
        ValueError: When the file has no "All items" column, which means its layout changed.
    """
    try:
        response = get_request(
            f"{BASE_URL}{base_year}/csv/zmi{base_year}aa.csv", timeout=120
        )
    except requests.exceptions.HTTPError as error:
        if error.response is not None and error.response.status_code == NOT_FOUND:
            return None
        raise

    rows = list(
        pd.read_csv(
            io.StringIO(response.content.decode("cp932", "replace")),
            header=None,
            dtype=str,
        ).itertuples(index=False)
    )

    # The second row holds the English names, the first data column being "All items".
    english_names = next(
        (row for row in rows if str(row[0]).startswith("Group/Item")), None
    )

    if english_names is None or str(english_names[1]).strip() != "All items":
        raise ValueError(
            f"The Japanese CPI file for base year {base_year} has no 'All items' column where "
            "it is expected, which means the Statistics Bureau changed its layout."
        )

    observations = {
        str(row[0]): pd.to_numeric(row[1], errors="coerce")
        for row in rows
        if MONTH_PATTERN.match(str(row[0]))
    }

    index = pd.Series(observations, dtype=float)
    index.index = pd.PeriodIndex(pd.to_datetime(index.index, format="%Y%m"), freq="M")

    return index.dropna().sort_index()


def _chain(indices: list[pd.Series]) -> pd.Series:
    """
    Links CPI bases into one series on the scale of the latest base.

    Every older base is rescaled by the ratio of the two indices in the months they
    overlap, which keeps its monthly changes intact, the standard way to chain indices.

    Args:
        indices (list[pd.Series]): The index of every base, oldest first.

    Returns:
        pd.Series: The chained index.
    """
    chained = indices[-1]

    for older in reversed(indices[:-1]):
        overlap = chained.index.intersection(older.index)

        if overlap.empty:
            continue

        factor = (chained.loc[overlap] / older.loc[overlap]).mean()
        earlier = older.loc[older.index < chained.index[0]] * factor
        chained = pd.concat([earlier, chained])

    return chained


def get_consumer_price_index() -> pd.DataFrame:
    """
    Retrieves the monthly Japanese Consumer Price Index for all items, chained across the
    base years from 2015 onwards and expressed on the latest base (2025 = 100 since the
    2026 rebase).

    Returns:
        pd.DataFrame: A single "Japan" column indexed by month.
    """
    description = "Japanese consumer price index"

    def fetch() -> pd.DataFrame:
        base_years = set(KNOWN_BASE_YEARS)

        # The page is only used to find new base years, so a change to it does no harm.
        try:
            page = get_request(f"{BASE_URL}{INDEX_PAGE}", timeout=60).text
            base_years.update(int(year) for year in FILE_PATTERN.findall(page))
        except requests.exceptions.RequestException:
            pass

        indices = [
            index
            for base_year in sorted(base_years)
            if (index := _read_base_file(base_year)) is not None and not index.empty
        ]

        if not indices:
            return pd.DataFrame()

        return _chain(indices).round(3).to_frame(COUNTRY)

    return collect_cached_data(
        source=policy_model.STATISTICS_BUREAU_OF_JAPAN,
        dataset="series",
        entity="zmi",
        fetch=fetch,
        description=description,
    )


def get_inflation_rate() -> pd.DataFrame:
    """
    Computes the Japanese annual inflation rate from the all-items Consumer Price Index:
    the change of the index over twelve months.

    Returns:
        pd.DataFrame: A single "Japan" column with the rate as a decimal (0.029 for
        2.9%), indexed by month.
    """
    consumer_price_index = get_consumer_price_index()

    return consumer_price_index.pct_change(12, fill_method=None).dropna()
