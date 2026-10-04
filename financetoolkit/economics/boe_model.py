"""Bank of England Model"""

__docformat__ = "google"

import io

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_ranged_data, require_columns
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"

# The database answers requests without a browser User-Agent with an error page.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}

# The nominal par yields of gilts the database publishes daily, with their maturity.
YIELD_CURVE_SERIES = {"IUDSNPY": "5Y", "IUDMNPY": "10Y", "IUDLNPY": "20Y"}

SERIES = {
    # Sterling Overnight Index Average (SONIA), daily.
    "overnight_rate": "IUDSOIA",
}

COUNTRY = "United Kingdom"

# The series start in 1993 (gilt yield) and 1997 (SONIA). The database rejects some
# start dates long before that, so a request never starts before this date.
EARLIEST_DATE = "1990-01-01"


def collect_boe_series(
    series_code: str,
    description: str,
    start_date: str,
    end_date: str,
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Retrieves the days between two dates of one or more Bank of England database series
    in a single request. Only the days that are not cached yet are requested.

    Args:
        series_code (str): The series code, e.g. "IUDMNPY", or several separated by commas.
        description (str): What is retrieved, used in the log and error messages.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        columns (dict[str, str] | None): The column name per series code. Defaults to None,
            which names the single series "United Kingdom".

    Returns:
        pd.DataFrame: The values as decimals, indexed by day.
    """
    columns = columns or {series_code: COUNTRY}

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        # The database takes dates written as "01/Jan/2026".
        date_from = pd.Timestamp(max(fetch_start, EARLIEST_DATE)).strftime("%d/%b/%Y")
        date_to = pd.Timestamp(fetch_end).strftime("%d/%b/%Y")

        response = get_request(
            f"{BASE_URL}?csv.x=yes&Datefrom={date_from}&Dateto={date_to}&SeriesCodes={series_code}"
            "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N",
            timeout=120,
            extra_headers=HEADERS,
        )

        # The database answers a request it cannot serve with an HTML page and status 200.
        if "csv" not in response.headers.get("Content-Type", ""):
            raise ValueError(
                f"The Bank of England database returned a web page instead of CSV for the "
                f"{description}, which means it rejected the request or changed its format."
            )

        try:
            data = pd.read_csv(io.StringIO(response.text))
        except pd.errors.ParserError as error:
            raise ValueError(
                f"The Bank of England response for the {description} is not readable as CSV "
                f"({error}), which means the database returned something else."
            ) from error

        require_columns(data, {"DATE", *columns}, description)

        # Dates are written as "01 Sep 2026".
        index = pd.PeriodIndex(
            pd.to_datetime(data["DATE"], format="%d %b %Y"), freq="D", name=None
        )
        values = {
            name: pd.to_numeric(data[code], errors="coerce").to_numpy() / 100
            for code, name in columns.items()
        }

        # The Bank of England publishes the rates in percent.
        return pd.DataFrame(values, index=index).sort_index()

    return collect_ranged_data(
        source=policy_model.BANK_OF_ENGLAND,
        dataset="series",
        entity=series_code,
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


def get_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily nominal par yields of 5, 10 and 20-year UK government bonds
    (gilts) in a single request.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yields as decimals, indexed by day with a column per maturity.
    """
    return collect_boe_series(
        ",".join(YIELD_CURVE_SERIES),
        "gilt par yields",
        start_date,
        end_date,
        columns=YIELD_CURVE_SERIES,
    )


def get_long_term_interest_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily nominal par yield of a 10-year UK government bond (gilt). It is
    taken from the yield curve request, so both share one query.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yield as a decimal, indexed by day.
    """
    yield_curve = get_yield_curve(start_date, end_date)

    if yield_curve.empty:
        return yield_curve

    return yield_curve[["10Y"]].rename(columns={"10Y": COUNTRY}).dropna()


def get_overnight_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily Sterling Overnight Index Average (SONIA).

    Returns:
        pd.DataFrame: The rate as a decimal, indexed by day.
    """
    return collect_boe_series(SERIES["overnight_rate"], "SONIA", start_date, end_date)
