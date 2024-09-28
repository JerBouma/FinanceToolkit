"""Helpers"""

__docformat__ = "google"

import pandas as pd

BASE_URL = "https://data-api.ecb.europa.eu/service/data/"
EXTENSIONS = "?format=csvdata"


def collect_ecb_data(
    ecb_data_string: str, dataset: str, frequency: str = "D"
) -> pd.DataFrame:
    """
    Collect the data from the ECB API and return it as a DataFrame.

    Args:
        ecb_data_string (str): The string that is appended to the base URL to
            get the data from the ECB API.

    Returns:
       pd.DataFrame: A DataFrame containing the data from the ECB API.
    """
    ecb_data = pd.read_csv(f"{BASE_URL}{dataset}/{ecb_data_string}{EXTENSIONS}")

    ecb_data = ecb_data.set_index("TIME_PERIOD")

    ecb_data.index = pd.PeriodIndex(data=ecb_data.index, freq=frequency)

    ecb_data.index.name = None

    ecb_data = ecb_data["OBS_VALUE"]

    return ecb_data
