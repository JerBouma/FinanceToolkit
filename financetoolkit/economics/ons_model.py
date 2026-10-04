"""Office for National Statistics (ONS) Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://www.ons.gov.uk/"

# The ONS has no API for its time series. Every series page does offer its data as JSON
# at "<page>/data", the format the ONS website itself uses, so that is what is read here.
SERIES = {
    "inflation_rate": "economy/inflationandpriceindices/timeseries/d7g7/mm23",
    "consumer_price_index": "economy/inflationandpriceindices/timeseries/d7bt/mm23",
    "unemployment_rate": "employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms",
}

COUNTRY = "United Kingdom"


def collect_ons_series(series: str, description: str) -> pd.DataFrame:
    """
    Retrieves the monthly observations of an ONS time series.

    Args:
        series (str): The path of the series page, e.g.
            "economy/inflationandpriceindices/timeseries/d7g7/mm23".
        description (str): What is retrieved, used in the log and error messages.

    Returns:
        pd.DataFrame: A single "United Kingdom" column indexed by month.

    Raises:
        ValueError: When the response has no monthly observations in the expected format.
    """

    def fetch() -> pd.DataFrame:
        response = get_request(f"{BASE_URL}{series}/data", timeout=60).json()

        months = response.get("months") if isinstance(response, dict) else None

        if not months or not {"date", "value"}.issubset(months[0]):
            raise ValueError(
                f"The ONS response for the {description} has no monthly observations in the "
                "expected format, which means the ONS changed its website and the data "
                "cannot be interpreted."
            )

        observations = pd.DataFrame(months)
        # Months are written as "2026 AUG".
        index = pd.PeriodIndex(
            pd.to_datetime(observations["date"], format="%Y %b"), freq="M"
        )
        values = pd.to_numeric(observations["value"], errors="coerce").to_numpy()

        return pd.DataFrame({COUNTRY: values}, index=index).sort_index()

    return collect_cached_data(
        source=policy_model.OFFICE_FOR_NATIONAL_STATISTICS,
        dataset="series",
        entity=series,
        fetch=fetch,
        description=description,
    )


def get_inflation_rate() -> pd.DataFrame:
    """
    Retrieves the UK Consumer Prices Index (CPI) annual rate, the headline inflation
    figure the Bank of England targets (series D7G7).

    Returns:
        pd.DataFrame: The annual inflation rate as a decimal (0.031 for 3.1%), indexed by
        month.
    """
    # The ONS publishes the rate in percent.
    return collect_ons_series(SERIES["inflation_rate"], "UK CPI annual rate") / 100


def get_consumer_price_index() -> pd.DataFrame:
    """
    Retrieves the UK Consumer Prices Index (CPI) for all items, with 2015 = 100
    (series D7BT).

    Returns:
        pd.DataFrame: The index, indexed by month.
    """
    return collect_ons_series(SERIES["consumer_price_index"], "UK CPI index")


def get_unemployment_rate() -> pd.DataFrame:
    """
    Retrieves the UK unemployment rate for people aged 16 and over, seasonally adjusted
    (series MGSX). Each month is the average of the three months ending in it, as the
    Labour Force Survey reports it.

    Returns:
        pd.DataFrame: The unemployment rate as a decimal fraction of the labour force
        (0.049 for 4.9%), indexed by month.
    """
    # The ONS publishes the rate in percent of the labour force.
    return collect_ons_series(SERIES["unemployment_rate"], "UK unemployment rate") / 100
