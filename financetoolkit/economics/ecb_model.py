"""European Central Bank (ECB) Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_ranged_data, require_columns
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://data-api.ecb.europa.eu/service/data/"

SERIES = {
    # 10-year spot rate of the yield curve of all euro area central government bonds.
    "long_term_interest_rate": "YC/B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y",
    # Euro short-term rate (€STR), the overnight rate of the euro area.
    "overnight_rate": "EST/B.EU000A2X2A25.WT",
}

COUNTRY = "Euro Area"


def collect_ecb_series(
    series_key: str, description: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Retrieves the days between two dates of an ECB Data Portal series. Only the days that
    are not cached yet are requested.

    Args:
        series_key (str): The dataset and series key, e.g. "EST/B.EU000A2X2A25.WT".
        description (str): What is retrieved, used in the log and error messages.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A single "Euro Area" column with the values as decimals, indexed by day.
    """

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        # Without detail=dataonly the €STR request runs into the portal's 30-second gateway
        # timeout, since every observation then carries its attributes.
        response = get_request(
            f"{BASE_URL}{series_key}?format=csvdata&detail=dataonly"
            f"&startPeriod={fetch_start}&endPeriod={fetch_end}",
            timeout=120,
        )

        # A range without observations is answered with an empty body.
        if not response.text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(response.text))
        require_columns(data, {"TIME_PERIOD", "OBS_VALUE"}, description)

        index = pd.PeriodIndex(pd.to_datetime(data["TIME_PERIOD"]), freq="D")
        values = pd.to_numeric(data["OBS_VALUE"], errors="coerce").to_numpy()

        # The ECB publishes the rates in percent.
        return pd.DataFrame({COUNTRY: values / 100}, index=index).sort_index()

    return collect_ranged_data(
        source=policy_model.EUROPEAN_CENTRAL_BANK,
        dataset="economics_series",
        entity=series_key,
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


def get_long_term_interest_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily 10-year spot rate of the euro area government bond yield curve,
    estimated by the ECB from the bonds of all euro area central governments.

    Returns:
        pd.DataFrame: The yield as a decimal, indexed by business day.
    """
    return collect_ecb_series(
        SERIES["long_term_interest_rate"],
        "euro area 10-year yield",
        start_date,
        end_date,
    )


def get_overnight_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily euro short-term rate (€STR).

    Returns:
        pd.DataFrame: The rate as a decimal, indexed by business day.
    """
    return collect_ecb_series(
        SERIES["overnight_rate"], "euro short-term rate", start_date, end_date
    )
