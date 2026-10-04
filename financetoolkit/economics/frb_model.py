"""Federal Reserve Board Model"""

__docformat__ = "google"

import re

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

# The seasonally adjusted industrial production indices of the G.17 release, the source
# of FRED's INDPRO series. Every index is one row per year with the twelve months.
URL = "https://www.federalreserve.gov/releases/g17/ipdisk/ip_sa.txt"

# The total index.
TOTAL_INDEX = "B50001"

# A data row holds the series code, the year and the months: "B50001" 2026 101.926 ...
ROW_PATTERN = re.compile(rf'"{TOTAL_INDEX}"\s+(\d{{4}})\s+(.*)$')

COUNTRY = "United States"


def get_industrial_production_index() -> pd.DataFrame:
    """
    Retrieves the monthly Industrial Production Index (2017 = 100), seasonally adjusted,
    from the Federal Reserve's G.17 release, from 1919.

    Returns:
        pd.DataFrame: A single "United States" column, indexed by the first day of each
        month ("Date") as FRED dates it.

    Raises:
        ValueError: When the file has no rows for the total index, which means its layout
            changed.
    """
    description = "industrial production index"

    def fetch() -> pd.DataFrame:
        response = get_request(URL, timeout=120)
        observations = {}

        # Months not published yet are absent or zero.
        for line in response.text.splitlines():
            # The first row of a series also carries its label in front of the code.
            match = ROW_PATTERN.search(line)
            if match is None:
                continue

            year, values = match.group(1), match.group(2).split()

            for month, value in enumerate(values[:12], start=1):
                number = pd.to_numeric(value, errors="coerce")
                if pd.notna(number) and number > 0:
                    observations[pd.Period(f"{year}-{month:02d}-01", freq="D")] = number

        if not observations:
            raise ValueError(
                f"The G.17 file has no rows for the total {description} ({TOTAL_INDEX}), which "
                "means the Federal Reserve changed its layout."
            )

        index = pd.PeriodIndex(list(observations), freq="D", name="Date")

        return pd.DataFrame(
            {COUNTRY: list(observations.values())}, index=index
        ).sort_index()

    return collect_cached_data(
        source=policy_model.FEDERAL_RESERVE_BOARD,
        dataset="series",
        entity=TOTAL_INDEX,
        fetch=fetch,
        description=description,
    )
