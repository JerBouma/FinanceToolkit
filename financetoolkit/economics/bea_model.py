"""U.S. Bureau of Economic Analysis (BEA) Model"""

__docformat__ = "google"

import re
from decimal import Decimal

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, to_us_series
from financetoolkit.helpers import round_half_up
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

# Every monthly series of the National Income and Product Accounts in one text file, one
# row per series and month: W875RX,2026M09,"17,015,175". The BEA's own API needs a key.
URL = "https://apps.bea.gov/national/Release/TXT/NipaDataM.txt"

# Real personal income excluding current transfer receipts (NIPA table 2.6, line 36), in
# millions of chained dollars, the source of FRED's W875RX1 series.
REAL_PERSONAL_INCOME = "W875RX"


def get_real_personal_income() -> pd.DataFrame:
    """
    Retrieves the monthly real personal income excluding current transfer receipts, in
    billions of chained dollars at a seasonally adjusted annual rate, from the U.S. Bureau
    of Economic Analysis, from 1959.

    The BEA's file holds the series in millions. It is converted to billions rounded to
    one decimal, rounding a half up, the figures the BEA publishes in its tables.

    Returns:
        pd.DataFrame: A single "United States" column indexed by the first day of each
        month ("Date").

    Raises:
        ValueError: When the file has no rows for the series, which means its layout
            changed.
    """
    description = "real personal income"
    row_pattern = re.compile(
        rf'^{REAL_PERSONAL_INCOME},(\d{{4}})M(\d{{2}}),"?([\d,.\-]+)"?\s*$', re.M
    )

    def fetch() -> pd.DataFrame:
        # The file holds every NIPA series (tens of megabytes), so only the rows of this
        # series are kept from the text.
        text = get_request(URL, timeout=300, extra_headers=TOOLKIT_HEADERS).text
        observations = {
            pd.Period(f"{year}-{month}-01", freq="D"): round_half_up(
                Decimal(value.replace(",", "")) / 1000, 1
            )
            for year, month, value in row_pattern.findall(text)
        }

        return to_us_series(
            observations, f"BEA's NIPA file ({REAL_PERSONAL_INCOME})", description
        )

    return collect_cached_data(
        source=policy_model.BUREAU_OF_ECONOMIC_ANALYSIS,
        dataset="series",
        entity=REAL_PERSONAL_INCOME,
        fetch=fetch,
        description=description,
    )
