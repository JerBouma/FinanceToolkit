"""Reserve Bank of Australia (RBA) Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

# The RBA publishes each statistical table as one CSV file with its full history, a block
# of metadata rows ending with the series identifiers, and then a row per date.
TABLE_URL = "https://www.rba.gov.au/statistics/tables/csv/{table}-data.csv"

# Table F3: the yields of Australian non-financial corporate bonds rated A and BBB, by
# target tenor, monthly from 2005.
CORPORATE_BOND_YIELDS = {
    f"FNFY{rating}{tenor}M": f"{rating} {tenor}Y"
    for rating in ("A", "BBB")
    for tenor in (3, 5, 7, 10)
}

# Table F2: the yields of Australian government bonds, interpolated to 2, 3, 5 and 10
# years, daily from 2013.
GOVERNMENT_BOND_YIELDS = {
    "FCMYGBAG2D": "2Y",
    "FCMYGBAG3D": "3Y",
    "FCMYGBAG5D": "5Y",
    "FCMYGBAG10D": "10Y",
}


def get_table(table: str) -> pd.DataFrame:
    """
    Retrieves an RBA statistical table with a column per series identifier.

    Args:
        table (str): The table, e.g. "f3".

    Returns:
        pd.DataFrame: The values as published, indexed by date.

    Raises:
        ValueError: When the file has no row of series identifiers.
    """
    description = f"RBA table {table.upper()}"

    def fetch() -> pd.DataFrame:
        text = get_request(
            TABLE_URL.format(table=table), timeout=60, extra_headers=TOOLKIT_HEADERS
        ).content.decode("utf-8-sig", errors="replace")
        lines = text.splitlines()
        header = next(
            (
                number
                for number, line in enumerate(lines)
                if line.startswith("Series ID")
            ),
            None,
        )

        if header is None:
            raise ValueError(
                f"The {description} has no row of series identifiers, which means its layout changed."
            )

        data = pd.read_csv(io.StringIO("\n".join(lines[header:])), index_col=0)
        data.index = pd.to_datetime(
            data.index, format="mixed", dayfirst=True, errors="coerce"
        )
        data = data[data.index.notna()].apply(pd.to_numeric, errors="coerce")
        data.index = pd.PeriodIndex(data.index, freq="D")
        data.index.name = None

        return data.dropna(how="all").sort_index()

    return collect_cached_data(
        source=policy_model.RESERVE_BANK_OF_AUSTRALIA,
        dataset="table",
        entity=table,
        fetch=fetch,
        description=description,
    )


def _select(table: str, series: dict[str, str]) -> pd.DataFrame:
    """
    Selects and renames series of a table and converts them from percent to decimals.

    Args:
        table (str): The table, e.g. "f3".
        series (dict[str, str]): The column name per series identifier.

    Returns:
        pd.DataFrame: The series as decimals, indexed by date.
    """
    data = get_table(table)

    if data.empty:
        return data

    available = {code: name for code, name in series.items() if code in data.columns}

    return data[list(available)].rename(columns=available).dropna(how="all") / 100


def get_corporate_bond_yields() -> pd.DataFrame:
    """
    Retrieves the yields of Australian non-financial corporate bonds rated A and BBB with a
    target tenor of 3, 5, 7 and 10 years, monthly from 2005.

    Returns:
        pd.DataFrame: The yields as decimals, indexed by month with a column per rating and
        tenor, e.g. "BBB 10Y".
    """
    yields = _select("f3", CORPORATE_BOND_YIELDS)

    if not yields.empty:
        yields.index = yields.index.asfreq("M")

    return yields


def get_yield_curve() -> pd.DataFrame:
    """
    Retrieves the yields of Australian government bonds interpolated to 2, 3, 5 and 10
    years, daily from 2013.

    Returns:
        pd.DataFrame: The yields as decimals, indexed by day with a column per maturity.
    """
    return _select("f2", GOVERNMENT_BOND_YIELDS)
