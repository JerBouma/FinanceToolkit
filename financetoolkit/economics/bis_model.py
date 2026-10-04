"""Bank for International Settlements (BIS) Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import (
    COUNTRY_CODES,
    collect_ranged_data,
    require_columns,
)
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://stats.bis.org/api/v1/data/"


def get_central_bank_policy_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily policy rates of the central banks the BIS covers, which includes
    the Federal Reserve, the European Central Bank, the Bank of England, the Bank of
    Japan and around forty more, with history back to the 1940s for some. Only the days
    that are not cached yet are requested.

    The euro area is published as "Euro Area"; the euro area members keep a column for
    the years before they adopted the euro, which ends when they did.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The policy rates as decimals (0.0425 for 4.25%), indexed by day with
        a column per country.
    """
    description = "central bank policy rates"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{BASE_URL}WS_CBPOL/D/all?format=csv&detail=dataonly"
            f"&startPeriod={fetch_start}&endPeriod={fetch_end}",
            timeout=300,
        )

        # A range without observations is answered with an empty body.
        if not response.text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(response.text))
        require_columns(data, {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}, description)

        data = data[data["REF_AREA"].isin(COUNTRY_CODES.keys())]
        data = data.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
        data = data.rename(columns=COUNTRY_CODES)
        data.index = pd.PeriodIndex(data.index, freq="D")
        data.index.name = None
        data.columns.name = None

        # The BIS publishes the rates in percent.
        return data.sort_index() / 100

    return collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="dataset",
        entity="WS_CBPOL/D",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
