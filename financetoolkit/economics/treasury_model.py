"""U.S. Department of the Treasury Model"""

__docformat__ = "google"

import io
from datetime import datetime

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, require_columns
from financetoolkit.utilities.requests_model import get_request

BASE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)

# The daily par yield curve is published from 1990, one file per year.
FIRST_YEAR = 1990

COUNTRY = "United States"


def _get_year(year: int) -> pd.DataFrame:
    """
    Retrieves the daily par yield curve of one year. A past year no longer changes, so it
    is cached under its own entry and only the current year is requested again.

    Args:
        year (int): The year to retrieve.

    Returns:
        pd.DataFrame: The 10-year yield in a "United States" column, indexed by day.
    """
    description = f"{year} Treasury par yield curve"

    def fetch() -> pd.DataFrame:
        response = get_request(BASE_URL.format(year=year), timeout=120)
        data = pd.read_csv(io.StringIO(response.text))
        require_columns(data, {"Date", "10 Yr"}, description)

        index = pd.PeriodIndex(
            pd.to_datetime(data["Date"], format="%m/%d/%Y"), freq="D"
        )
        values = pd.to_numeric(data["10 Yr"], errors="coerce").to_numpy()

        # The Treasury publishes the yields in percent.
        return pd.DataFrame({COUNTRY: values / 100}, index=index).sort_index()

    current_year = datetime.now().year

    return collect_cached_data(
        source=policy_model.US_TREASURY,
        dataset="par_yield_curve" if year == current_year else "par_yield_curve_year",
        entity=str(year),
        fetch=fetch,
        description=description,
    )


def get_long_term_interest_rate(
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """
    Retrieves the daily 10-year par yield of US Treasuries as published by the U.S.
    Department of the Treasury, from 1990 onwards.

    Args:
        start_date (str | None): Only the years from this date (YYYY-MM-DD) onwards are
            requested, one file per year. Defaults to None, which starts in 1990.
        end_date (str | None): Only the years up to this date (YYYY-MM-DD) are requested.
            Defaults to None, which ends in the current year.

    Returns:
        pd.DataFrame: A single "United States" column with the yield as a decimal,
        indexed by business day.
    """
    first_year = max(FIRST_YEAR, int(start_date[:4]) if start_date else FIRST_YEAR)
    last_year = min(
        datetime.now().year, int(end_date[:4]) if end_date else datetime.now().year
    )
    years = list(range(first_year, last_year + 1))

    frames = helpers.run_in_parallel(_get_year, [(year,) for year in years])
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames).sort_index().dropna()
