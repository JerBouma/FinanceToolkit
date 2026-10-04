"""Bank of Canada Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_ranged_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://www.bankofcanada.ca/valet/observations/"

# Treasury bill and benchmark bond yields of the Government of Canada, with their maturity.
# The long-term benchmark is the 30-year bond.
YIELD_CURVE_SERIES = {
    "V80691342": "1M",
    "V80691344": "3M",
    "V80691345": "6M",
    "V80691346": "1Y",
    "BD.CDN.2YR.DQ.YLD": "2Y",
    "BD.CDN.3YR.DQ.YLD": "3Y",
    "BD.CDN.5YR.DQ.YLD": "5Y",
    "BD.CDN.7YR.DQ.YLD": "7Y",
    "BD.CDN.10YR.DQ.YLD": "10Y",
    "BD.CDN.LONG.DQ.YLD": "30Y",
}


def get_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily Government of Canada yield curve from the Bank of Canada's Valet
    API, treasury bills from 1 month to 1 year and benchmark bonds from 2 to 30 years,
    in a single request. Only the days that are not cached yet are requested.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yields as decimals, indexed by business day with a column per
        maturity ("1M" to "30Y").

    Raises:
        ValueError: When the response has no observations in the expected format.
    """
    description = "Government of Canada yield curve"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{BASE_URL}{','.join(YIELD_CURVE_SERIES)}/json"
            f"?start_date={fetch_start}&end_date={fetch_end}",
            timeout=120,
        ).json()

        observations = (
            response.get("observations") if isinstance(response, dict) else None
        )

        if observations is None:
            raise ValueError(
                f"The Bank of Canada response for the {description} has no observations, "
                "which means the Valet API changed."
            )

        if not observations:
            return pd.DataFrame()

        records = {
            observation["d"]: {
                label: pd.to_numeric(
                    observation.get(code, {}).get("v"), errors="coerce"
                )
                for code, label in YIELD_CURVE_SERIES.items()
            }
            for observation in observations
        }
        yield_curve = pd.DataFrame.from_dict(records, orient="index")
        yield_curve.index = pd.PeriodIndex(yield_curve.index, freq="D")

        # The Bank of Canada publishes the yields in percent.
        return yield_curve.dropna(how="all").sort_index() / 100

    return collect_ranged_data(
        source=policy_model.BANK_OF_CANADA,
        dataset="series",
        entity="yield_curve",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
