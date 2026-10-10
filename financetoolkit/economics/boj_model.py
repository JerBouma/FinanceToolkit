"""Bank of Japan Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_ranged_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://www.stat-search.boj.or.jp/api/v1/getDataCode"

# The status the API reports for a successful request.
SUCCESS_STATUS = 200

# The daily call rate is available from 1985.
EARLIEST_DATE = "1985-01-01"

COUNTRY = "Japan"


def collect_boj_series(
    database: str, series_code: str, description: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Retrieves the days between two dates of a series from the Bank of Japan's statistics
    API. The API takes whole months, so only the months that are not cached yet are
    requested.

    Args:
        database (str): The database the series is in, e.g. "FM01".
        series_code (str): The series code, e.g. "STRDCLUCON".
        description (str): What is retrieved, used in the log and error messages.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A single "Japan" column with the values as decimals, indexed by day.

    Raises:
        ValueError: When the API reports an error or the response has no observations.
    """

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        # The API takes months written as 202609.
        start_month = max(fetch_start, EARLIEST_DATE)[:7].replace("-", "")
        end_month = fetch_end[:7].replace("-", "")

        response = get_request(
            f"{BASE_URL}?format=json&lang=en&db={database}&code={series_code}"
            f"&startDate={start_month}&endDate={end_month}",
            timeout=120,
        ).json()

        if response.get("STATUS") != SUCCESS_STATUS or not response.get("RESULTSET"):
            raise ValueError(
                f"The Bank of Japan API returned no data for the {description}: "
                f"{response.get('MESSAGE', 'no message')}"
            )

        values = response["RESULTSET"][0].get("VALUES", {})

        if not {"SURVEY_DATES", "VALUES"}.issubset(values):
            raise ValueError(
                f"The Bank of Japan response for the {description} has no observations in "
                "the expected format, which means the API changed."
            )

        # Dates are written as 20260930.
        index = pd.PeriodIndex(
            pd.to_datetime(
                [str(date) for date in values["SURVEY_DATES"]], format="%Y%m%d"
            ),
            freq="D",
        )
        observations = pd.to_numeric(
            pd.Series(values["VALUES"]), errors="coerce"
        ).to_numpy()

        # The Bank of Japan publishes the rates in percent per annum.
        rate = pd.DataFrame({COUNTRY: observations / 100}, index=index).sort_index()

        # Every calendar day is listed, with no value on weekends and holidays.
        return rate.dropna()

    return collect_ranged_data(
        source=policy_model.BANK_OF_JAPAN,
        dataset="series",
        entity=f"{database}/{series_code}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


def get_overnight_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily average uncollateralized overnight call rate, the Tokyo Overnight
    Average Rate (TONA) the Bank of Japan steers its policy with.

    Returns:
        pd.DataFrame: The rate as a decimal, indexed by day.
    """
    return collect_boj_series(
        "FM01",
        "STRDCLUCON",
        "uncollateralized overnight call rate",
        start_date,
        end_date,
    )
