"""Helpers"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://data-api.ecb.europa.eu/service/data/"
EXTENSIONS = "?format=csvdata"


def collect_ecb_data(
    ecb_data_string: str, dataset: str, frequency: str = "D"
) -> pd.DataFrame:
    """
    Collect the data from the ECB API and return it as a DataFrame.

    The ECB endpoint returns the full history of a series in one response and takes
    no date range, so there is nothing to request incrementally. The response is
    cached whole instead, keyed on the series being asked for.

    Args:
        ecb_data_string (str): The string that is appended to the base URL to
            get the data from the ECB API.
        dataset (str): The ECB dataset (dataflow) to query, e.g. 'FM' for the
            Financial Markets dataset.
        frequency (str): The frequency of the returned observations, used to build
            the PeriodIndex, e.g. 'D' for daily or 'M' for monthly. Defaults to 'D'.

    Returns:
       pd.DataFrame: A DataFrame containing the data from the ECB API.
    """
    cache = get_active_cache()
    cache_entity = f"{dataset}/{ecb_data_string}"
    cache_parameters = {"frequency": frequency}

    if cache is not None:
        cached_data = cache.get(
            source=policy_model.EUROPEAN_CENTRAL_BANK,
            dataset="series",
            entity=cache_entity,
            parameters=cache_parameters,
        )

        if cached_data is not None:
            return cached_data

    url = f"{BASE_URL}{dataset}/{ecb_data_string}{EXTENSIONS}"
    response = get_request(url)
    ecb_data = pd.read_csv(io.StringIO(response.text))

    ecb_data = ecb_data.set_index("TIME_PERIOD")

    ecb_data.index = pd.PeriodIndex(data=ecb_data.index, freq=frequency)

    ecb_data.index.name = None

    ecb_data = ecb_data["OBS_VALUE"]

    if cache is not None and not ecb_data.empty:
        cache.set(
            source=policy_model.EUROPEAN_CENTRAL_BANK,
            dataset="series",
            entity=cache_entity,
            data=ecb_data,
            parameters=cache_parameters,
        )

    return ecb_data
