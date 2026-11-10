"""U.S. Bureau of Labor Statistics (BLS) Model"""

__docformat__ = "google"

import json
from datetime import datetime

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

BASE_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

# The keyless version of the API returns at most three years per request and accepts 25
# requests a day per IP address. It is therefore only asked for the latest releases,
# once a day, and never for a full history.
RECENT_YEARS = 3

SERIES = {
    # Unemployment rate, 16 years and over, seasonally adjusted (Current Population Survey).
    "unemployment_rate": "LNS14000000",
    # Total nonfarm employment in thousands, seasonally adjusted (Current Employment
    # Statistics), the source of FRED's PAYEMS series.
    "nonfarm_payrolls": "CES0000000001",
    # Producer Price Index for final demand, not seasonally adjusted, November 2009 = 100:
    # the headline PPI.
    "producer_price_index": "WPUFD4",
    # Export and import price indices of all commodities, not seasonally adjusted,
    # 2000 = 100 (International Price Program).
    "export_price_index": "EIUIQ",
    "import_price_index": "EIUIR",
}

# The Consumer Price Index for All Urban Consumers (CPI-U), U.S. city average, not
# seasonally adjusted, 1982-84 = 100: all items, the core index (all items less food and
# energy) and the main components.
CONSUMER_PRICE_SERIES = {
    "all": "CUUR0000SA0",
    "core": "CUUR0000SA0L1E",
    "energy": "CUUR0000SA0E",
    "food": "CUUR0000SAF1",
    "goods": "CUUR0000SAC",
    "services": "CUUR0000SAS",
    "shelter": "CUUR0000SAH1",
}

COUNTRY = "United States"


class BLSRequestRefusedError(requests.exceptions.RequestException):
    """The BLS refused the request, e.g. because the daily request limit was reached."""


def collect_recent_bls_series(
    series_id: str,
    description: str,
    fallback: str | None = None,
    columns: dict[str, str] | None = None,
    dataset: str = "series",
) -> pd.DataFrame:
    """
    Retrieves the monthly observations of one or more BLS series for the current and the
    two previous years, in a single request that is cached (for a day by default).

    Args:
        series_id (str): The BLS series identifier, e.g. "LNS14000000".
        description (str): What is retrieved, used in the log and error messages.
        fallback (str | None): What is used instead when the BLS cannot be reached, logged
            as information rather than as an error. Defaults to None.
        columns (dict[str, str] | None): The column name per series identifier, to retrieve
            several series in the same request. Defaults to None, which retrieves
            series_id alone as a "United States" column.
        dataset (str): The cache dataset, which sets how long the answer is kept.
            Defaults to "series".

    Returns:
        pd.DataFrame: One column per series (a single "United States" column by default),
        indexed by month.

    Raises:
        ValueError: When the response has no observations in the expected format.
    """
    columns = columns or {series_id: COUNTRY}

    def fetch() -> pd.DataFrame:
        current_year = datetime.now().year
        response = get_request(
            BASE_URL,
            timeout=60,
            extra_headers={**TOOLKIT_HEADERS, "Content-Type": "application/json"},
            data=json.dumps(
                {
                    "seriesid": list(columns),
                    "startyear": str(current_year - RECENT_YEARS + 1),
                    "endyear": str(current_year),
                }
            ),
        ).json()

        # A refused request, such as one over the daily limit, is answered with status 200.
        if response.get("status") != "REQUEST_SUCCEEDED":
            raise BLSRequestRefusedError(
                f"The BLS did not process the request: {' '.join(response.get('message', []))}"
            )

        series_values = {}

        try:
            for series in response["Results"]["series"]:
                observations = pd.DataFrame(series["data"])
                observations = observations[
                    observations["period"].str.match(r"^M(0[1-9]|1[0-2])$")
                ]
                index = pd.PeriodIndex(
                    (
                        observations["year"] + "-" + observations["period"].str[1:]
                    ).to_numpy(),
                    freq="M",
                )
                series_values[columns[series["seriesID"]]] = pd.Series(
                    pd.to_numeric(observations["value"], errors="coerce").to_numpy(),
                    index=index,
                )
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f"The BLS response for the {description} has no observations in the expected "
                "format, which means the API changed."
            ) from error

        if not series_values:
            raise ValueError(f"The BLS response for the {description} has no series.")

        return pd.DataFrame(series_values).sort_index()

    return collect_cached_data(
        source=policy_model.BUREAU_OF_LABOR_STATISTICS,
        dataset=dataset,
        entity=f"{','.join(columns)}/recent",
        fetch=fetch,
        description=description,
        fallback=fallback,
    )


def get_unemployment_rate() -> pd.DataFrame:
    """
    Retrieves the latest months of the US unemployment rate, seasonally adjusted (series
    LNS14000000), the headline figure of the monthly Employment Situation report.

    Returns:
        pd.DataFrame: The unemployment rate as a decimal fraction of the labour force
        (0.042 for 4.2%), indexed by month, for the current and the two previous years.
    """
    # The BLS publishes the rate in percent of the labour force.
    return (
        collect_recent_bls_series(
            SERIES["unemployment_rate"],
            "US unemployment rate",
            fallback="the US series is the OECD's, without the months the BLS publishes first",
        )
        / 100
    )


def get_nonfarm_payrolls() -> pd.DataFrame:
    """
    Retrieves the latest months of total nonfarm payroll employment, in thousands of
    persons and seasonally adjusted (series CES0000000001), the headline figure of the
    monthly Employment Situation report.

    The keyless BLS API returns about the last three years, so the full history, from
    1939, requires a FRED API key.

    Returns:
        pd.DataFrame: A single "United States" column indexed by month.
    """
    payrolls = collect_recent_bls_series(
        SERIES["nonfarm_payrolls"], "nonfarm payrolls"
    ).copy()

    # Dated on the first day of each month, as FRED dates the series.
    payrolls.index = payrolls.index.asfreq("D", how="start")
    payrolls.index.name = "Date"

    return payrolls


def get_price_index(measure: str) -> pd.DataFrame:
    """
    Retrieves the latest months of a US price index, not seasonally adjusted: the
    Producer Price Index for final demand (series WPUFD4), or the export or import price
    index of all commodities (series EIUIQ and EIUIR).

    The keyless BLS API returns about the last three years, so the full history requires
    a FRED API key.

    Args:
        measure (str): "producer", "export" or "import".

    Returns:
        pd.DataFrame: A single "United States" column indexed by month.
    """
    return collect_recent_bls_series(
        SERIES[f"{measure}_price_index"], f"US {measure} price index"
    )


def get_consumer_price_index() -> pd.DataFrame:
    """
    Retrieves the latest months of the US Consumer Price Index for All Urban Consumers
    (CPI-U), not seasonally adjusted, 1982-84 = 100: all items, the core index (all items
    less food and energy), energy, food, goods (commodities), services and shelter, in a
    single request. The BLS publishes a month at 8:30 ET on its release day, before any
    other source.

    The keyless BLS API returns about the last three years, so the full history requires
    a FRED API key.

    Returns:
        pd.DataFrame: One column per measure ("all", "core", "energy", "food", "goods",
        "services" and "shelter"), indexed by month.
    """
    return collect_recent_bls_series(
        CONSUMER_PRICE_SERIES["all"],
        "US consumer price index",
        fallback="the US series ends with the months the other sources have",
        columns={series: measure for measure, series in CONSUMER_PRICE_SERIES.items()},
        dataset="consumer_prices",
    )
