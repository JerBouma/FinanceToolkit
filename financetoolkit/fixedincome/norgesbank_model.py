"""Norges Bank Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_ranged_data, require_columns
from financetoolkit.utilities.requests_model import get_request

# The generic yields of Norwegian government bonds, for every tenor Norges Bank publishes.
URL = "https://data.norges-bank.no/api/data/GOVT_GENERIC_RATES/B..GBON."


def get_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily generic yields of Norwegian government bonds from Norges Bank, for
    maturities of 3 to 10 years, in a single request. Only the days that are not cached
    yet are requested.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yields as decimals, indexed by business day with a column per
        maturity (e.g. "3Y" to "10Y").
    """
    description = "Norwegian government bond yields"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{URL}?format=csv&locale=en&startPeriod={fetch_start}&endPeriod={fetch_end}",
            timeout=60,
        )

        if not response.text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(response.text), sep=";")
        require_columns(data, {"TENOR", "TIME_PERIOD", "OBS_VALUE"}, description)

        data["OBS_VALUE"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
        yield_curve = data.pivot(
            index="TIME_PERIOD", columns="TENOR", values="OBS_VALUE"
        )
        yield_curve.index = pd.PeriodIndex(yield_curve.index, freq="D")
        yield_curve.index.name = None
        yield_curve.columns.name = None

        # Norges Bank publishes the yields in percent.
        return yield_curve.sort_index() / 100

    return collect_ranged_data(
        source=policy_model.NORGES_BANK,
        dataset="series",
        entity="GOVT_GENERIC_RATES",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
