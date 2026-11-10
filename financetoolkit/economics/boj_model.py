"""Bank of Japan Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, collect_ranged_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://www.stat-search.boj.or.jp/api/v1/getDataCode"

# The status the API reports for a successful request.
SUCCESS_STATUS = 200

# The daily call rate is available from 1985.
EARLIEST_DATE = "1985-01-01"

COUNTRY = "Japan"

# The Corporate Goods Price Index (CGPI, 2020 base) of the Price Indexes database: the
# producer price index of goods sold in Japan, and the export and import price indices
# in yen. All three are published monthly from January 1960, with 2020 = 100.
PRICE_INDEX_DATABASE = "PR01"
PRICE_INDEX_SERIES = {
    "producer": "PRCG20_2200000000",
    "export": "PRCG20_2400000000",
    "import": "PRCG20_2600000000",
}
PRICE_INDEX_FIRST_MONTH = "196001"


def request_boj_series(
    database: str,
    series_code: str,
    description: str,
    start_month: str,
    end_month: str,
) -> pd.Series:
    """
    Requests the observations of a series between two months from the Bank of Japan's
    statistics API.

    Args:
        database (str): The database the series is in, e.g. "FM01".
        series_code (str): The series code, e.g. "STRDCLUCON".
        description (str): What is retrieved, used in the error messages.
        start_month (str): The first month, written as 202609.
        end_month (str): The last month, written as 202609.

    Returns:
        pd.Series: The values as published, indexed by the dates as the API writes them:
        20260930 for a daily and 202609 for a monthly series. Dates without a value are
        NaN.

    Raises:
        ValueError: When the API reports an error or the response has no observations.
    """
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

    return pd.Series(
        pd.to_numeric(pd.Series(values["VALUES"]), errors="coerce").to_numpy(),
        index=[str(date) for date in values["SURVEY_DATES"]],
    )


def collect_boj_series(
    database: str, series_code: str, description: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Retrieves the days between two dates of a daily series from the Bank of Japan's
    statistics API. The API takes whole months, so only the months that are not cached
    yet are requested.

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
        observations = request_boj_series(
            database,
            series_code,
            description,
            max(fetch_start, EARLIEST_DATE)[:7].replace("-", ""),
            fetch_end[:7].replace("-", ""),
        )

        # Dates are written as 20260930.
        index = pd.PeriodIndex(
            pd.to_datetime(observations.index, format="%Y%m%d"), freq="D"
        )

        # The Bank of Japan publishes the rates in percent per annum.
        rate = pd.DataFrame(
            {COUNTRY: observations.to_numpy() / 100}, index=index
        ).sort_index()

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


def collect_price_index(measure: str) -> pd.DataFrame:
    """
    Retrieves the full monthly history of an index of the Corporate Goods Price Index,
    not seasonally adjusted, with 2020 = 100. The series is short enough to request whole,
    and is cached as a whole because the Bank of Japan revises the recent months.

    Args:
        measure (str): "producer" for the producer price index (the domestic CGPI),
            "export" for the export price index or "import" for the import price index,
            both in yen.

    Returns:
        pd.DataFrame: A single "Japan" column indexed by month.
    """
    series_code = PRICE_INDEX_SERIES[measure]
    description = f"Japanese {measure} price index"

    def fetch() -> pd.DataFrame:
        observations = request_boj_series(
            PRICE_INDEX_DATABASE,
            series_code,
            description,
            PRICE_INDEX_FIRST_MONTH,
            pd.Timestamp.today().strftime("%Y%m"),
        )

        # Months are written as 202609.
        index = pd.PeriodIndex(
            pd.to_datetime(observations.index, format="%Y%m"), freq="M"
        )

        # Months not published yet are listed without a value.
        return (
            pd.DataFrame({COUNTRY: observations.to_numpy()}, index=index)
            .sort_index()
            .dropna()
        )

    return collect_cached_data(
        source=policy_model.BANK_OF_JAPAN,
        dataset="price_index",
        entity=f"{PRICE_INDEX_DATABASE}/{series_code}",
        fetch=fetch,
        description=description,
    )
