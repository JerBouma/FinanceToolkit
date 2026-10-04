"""U.S. Bureau of Labor Statistics (BLS) Model"""

__docformat__ = "google"

from datetime import datetime

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

# The keyless version of the API returns at most three years per request and accepts 25
# requests a day per IP address. It is therefore only asked for the latest releases,
# once a day, and never for a full history.
RECENT_YEARS = 2

SERIES = {
    # Unemployment rate, 16 years and over, seasonally adjusted (Current Population Survey).
    "unemployment_rate": "LNS14000000",
}

COUNTRY = "United States"


class BLSRequestRefusedError(requests.exceptions.RequestException):
    """The BLS refused the request, e.g. because the daily request limit was reached."""


def collect_recent_bls_series(series_id: str, description: str) -> pd.DataFrame:
    """
    Retrieves the monthly observations of a BLS series for the current and the previous
    year, in a single request that is cached for a day.

    Args:
        series_id (str): The BLS series identifier, e.g. "LNS14000000".
        description (str): What is retrieved, used in the log and error messages.

    Returns:
        pd.DataFrame: A single "United States" column indexed by month.

    Raises:
        ValueError: When the response has no observations in the expected format.
    """

    def fetch() -> pd.DataFrame:
        current_year = datetime.now().year
        response = get_request(
            f"{BASE_URL}{series_id}?startyear={current_year - RECENT_YEARS + 1}"
            f"&endyear={current_year}",
            timeout=60,
        ).json()

        # A refused request, such as one over the daily limit, is answered with status 200.
        if response.get("status") != "REQUEST_SUCCEEDED":
            raise BLSRequestRefusedError(
                f"The BLS did not process the request: {' '.join(response.get('message', []))}"
            )

        try:
            observations = pd.DataFrame(response["Results"]["series"][0]["data"])
            observations = observations[
                observations["period"].str.match(r"^M(0[1-9]|1[0-2])$")
            ]
            index = pd.PeriodIndex(
                (
                    observations["year"] + "-" + observations["period"].str[1:]
                ).to_numpy(),
                freq="M",
            )
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f"The BLS response for the {description} has no observations in the expected "
                "format, which means the API changed."
            ) from error

        values = pd.to_numeric(observations["value"], errors="coerce").to_numpy()

        return pd.DataFrame({COUNTRY: values}, index=index).sort_index()

    return collect_cached_data(
        source=policy_model.BUREAU_OF_LABOR_STATISTICS,
        dataset="series",
        entity=f"{series_id}/recent",
        fetch=fetch,
        description=description,
    )


def get_unemployment_rate() -> pd.DataFrame:
    """
    Retrieves the latest months of the US unemployment rate, seasonally adjusted (series
    LNS14000000), the headline figure of the monthly Employment Situation report.

    Returns:
        pd.DataFrame: The unemployment rate as a decimal fraction of the labour force
        (0.042 for 4.2%), indexed by month, for the current and the previous year.
    """
    # The BLS publishes the rate in percent of the labour force.
    return (
        collect_recent_bls_series(SERIES["unemployment_rate"], "US unemployment rate")
        / 100
    )
