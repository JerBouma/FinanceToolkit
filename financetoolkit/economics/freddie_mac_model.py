"""Freddie Mac Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

# The full weekly history of the Primary Mortgage Market Survey, the source of FRED's
# MORTGAGE30US series, published by Freddie Mac every Thursday.
URL = "https://www.freddiemac.com/pmms/docs/PMMS_history.csv"

COUNTRY = "United States"


def get_mortgage_rate_30_year() -> pd.DataFrame:
    """
    Retrieves the weekly average 30-year fixed mortgage rate from Freddie Mac's Primary
    Mortgage Market Survey, from April 1971.

    Returns:
        pd.DataFrame: A single "United States" column with the rate as a decimal (0.0728
        for 7.28%), indexed by the survey day ("Date").
    """
    description = "30-year mortgage rate"

    def fetch() -> pd.DataFrame:
        response = get_request(URL, timeout=60)
        data = pd.read_csv(io.StringIO(response.text))
        require_columns(data, {"date", "pmms30"}, description)

        index = pd.PeriodIndex(
            pd.to_datetime(data["date"], format="%m/%d/%Y"), freq="D", name="Date"
        )
        values = pd.to_numeric(data["pmms30"], errors="coerce").to_numpy()

        # Freddie Mac publishes the rate in percent.
        return pd.DataFrame({COUNTRY: values / 100}, index=index).dropna().sort_index()

    return collect_cached_data(
        source=policy_model.FREDDIE_MAC,
        dataset="series",
        entity="pmms30",
        fetch=fetch,
        description=description,
    )
