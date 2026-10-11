"""Japan Ministry of Finance Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"

# The history up to the end of the previous month and the current month are separate files.
FILES = ["historical/jgbcme_all.csv", "jgbcme.csv"]

COUNTRY = "Japan"


def get_government_bond_yields() -> pd.DataFrame:
    """
    Retrieves the daily yields of Japanese government bonds (JGBs) for every maturity
    from 1 to 40 years, as published by the Ministry of Finance since 1974.

    Returns:
        pd.DataFrame: The yields as decimals, indexed by day with a column per maturity
        ("1Y" to "40Y"). Maturities that were not issued yet are NaN.
    """
    description = "Japanese government bond yields"

    def fetch() -> pd.DataFrame:
        frames = []

        for file in FILES:
            response = get_request(f"{BASE_URL}{file}", timeout=120)
            # The first line is a title; the data ends at the first empty row.
            data = pd.read_csv(
                io.StringIO(response.content.decode("utf-8", "replace")), skiprows=1
            )
            require_columns(data, {"Date", "10Y"}, description)

            data = data[data["Date"].astype(str).str.match(r"^\d{4}/\d{1,2}/\d{1,2}$")]
            data = data.set_index("Date")
            data.index = pd.PeriodIndex(
                pd.to_datetime(data.index, format="%Y/%m/%d"), freq="D"
            )
            frames.append(data.apply(pd.to_numeric, errors="coerce"))

        yields = pd.concat(frames)
        yields = yields[~yields.index.duplicated(keep="last")].sort_index()
        yields = yields.loc[
            :, [column for column in yields.columns if column.endswith("Y")]
        ]
        yields.index.name = None

        # The Ministry of Finance publishes the yields in percent; a "-" is no issue.
        return yields / 100

    return collect_cached_data(
        source=policy_model.JAPAN_MINISTRY_OF_FINANCE,
        dataset="yields",
        entity="jgbcme",
        fetch=fetch,
        description=description,
    )


def get_long_term_interest_rate() -> pd.DataFrame:
    """
    Retrieves the daily yield of a 10-year Japanese government bond.

    Returns:
        pd.DataFrame: A single "Japan" column with the yield as a decimal, indexed by day.
    """
    yields = get_government_bond_yields()

    return (
        yields[["10Y"]].rename(columns={"10Y": COUNTRY}) if not yields.empty else yields
    )
