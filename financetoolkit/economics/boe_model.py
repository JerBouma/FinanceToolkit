"""Bank of England Model"""

__docformat__ = "google"

import io
import re
import zipfile

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.economics.helpers import (
    collect_cached_data,
    collect_ranged_data,
    require_columns,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.requests_model import get_request

logger = get_logger()

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


YIELD_CURVE_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
)

# The archives of the daily yield curves the Bank of England estimates, one workbook per
# span of years, and the name of each curve's workbook in the file of the current month.
CURVE_ARCHIVES = {
    "nominal": ("glcnominalddata.zip", "GLC Nominal daily data current month.xlsx"),
    "real": ("glcrealddata.zip", "GLC Real daily data current month.xlsx"),
    "inflation": (
        "glcinflationddata.zip",
        "GLC Inflation daily data current month.xlsx",
    ),
    "ois": ("oisddata.zip", "OIS daily data current month.xlsx"),
}
CURRENT_MONTH_ARCHIVE = "latest-yield-curve-data.zip"
SPOT_CURVE_SHEET_PATTERN = re.compile(r"^4\.\s+.*spot curve\s*$", re.IGNORECASE)

# A workbook that runs to the present has no last year; this stands in for it.
OPEN_ENDED_YEAR = 9999

# The span of years a workbook in the archive covers, from its name.
SPAN_PATTERN = re.compile(r"(\d{4}) to (\d{4}|present)", re.IGNORECASE)


def _parse_spot_curve(workbook: bytes, description: str) -> pd.DataFrame:
    """
    Reads the spot curve worksheet of a Bank of England yield curve workbook.

    Args:
        workbook (bytes): The workbook.
        description (str): What is read, used in the error message.

    Returns:
        pd.DataFrame: The rates as decimals, indexed by day with a column per maturity in
        years ("0.5Y" to "40Y").

    Raises:
        ValueError: When the worksheet or its row of maturities is missing.
    """
    # The worksheet is "4. spot curve" in recent workbooks and "4.  real spot curve" (with
    # two spaces) in older ones, so it is found by its number and name.
    excel = pd.ExcelFile(io.BytesIO(workbook))
    sheet_name = next(
        (name for name in excel.sheet_names if SPOT_CURVE_SHEET_PATTERN.match(name)),
        None,
    )

    if sheet_name is None:
        raise ValueError(
            f"The {description} has no spot curve worksheet, which means the Bank of England "
            "changed its layout."
        )

    sheet = excel.parse(sheet_name=sheet_name, header=None)

    labels = sheet.iloc[:, 0].astype(str).str.strip().str.lower()
    years_rows = sheet.index[labels == "years:"]

    if len(years_rows) == 0:
        raise ValueError(
            f"The {description} has no row of maturities, which means the Bank of England "
            "changed its layout."
        )

    maturities = pd.to_numeric(sheet.iloc[years_rows[0], 1:], errors="coerce")
    dates = pd.to_datetime(sheet.iloc[:, 0], errors="coerce", format="mixed")
    rows = sheet[dates.notna()]

    curve = rows.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    curve.columns = [
        f"{maturity:g}Y" if pd.notna(maturity) else None for maturity in maturities
    ]
    curve = curve.loc[:, [column for column in curve.columns if column is not None]]
    curve.index = pd.PeriodIndex(dates[dates.notna()], freq="D")
    curve.index.name = None

    # The Bank of England publishes the rates in percent.
    return curve.dropna(how="all") / 100


def _get_archive_workbooks(
    curve: str, start_year: int, end_year: int
) -> list[pd.DataFrame]:
    """
    Retrieves the workbooks of a curve's archive that cover the requested years. A workbook
    of a closed span of years no longer changes and is cached for a year; the one that runs
    to the present is cached for a week. The archive is only downloaded when a workbook that
    is needed is not cached.

    Args:
        curve (str): "nominal", "real", "inflation" or "ois".
        start_year (int): The first year needed.
        end_year (int): The last year needed.

    Returns:
        list[pd.DataFrame]: The spot curves of the workbooks covering the years.
    """
    archive_name, _ = CURVE_ARCHIVES[curve]
    description = f"Bank of England {curve} yield curve archive"
    cache = get_active_cache()
    listing_entity = f"{curve}/workbooks"

    names = (
        cache.get(
            source=policy_model.BANK_OF_ENGLAND,
            dataset="curve_listing",
            entity=listing_entity,
        )
        if cache
        else None
    )
    archive = None

    def download() -> zipfile.ZipFile:
        response = get_request(
            f"{YIELD_CURVE_URL}{archive_name}", timeout=300, extra_headers=HEADERS
        )
        return zipfile.ZipFile(io.BytesIO(response.content))

    if names is None:
        archive = download()
        names = pd.Series(archive.namelist())
        if cache:
            cache.set(
                source=policy_model.BANK_OF_ENGLAND,
                dataset="curve_listing",
                entity=listing_entity,
                data=names,
            )

    frames = []
    for name in names:
        span = SPAN_PATTERN.search(name)
        if span is None:
            continue

        first = int(span.group(1))
        last = (
            OPEN_ENDED_YEAR
            if span.group(2).lower() == "present"
            else int(span.group(2))
        )

        if last < start_year or first > end_year:
            continue

        dataset = (
            "curve_workbook_current" if last == OPEN_ENDED_YEAR else "curve_workbook"
        )
        entity = f"{curve}/{name}"
        cached = (
            cache.get(
                source=policy_model.BANK_OF_ENGLAND, dataset=dataset, entity=entity
            )
            if cache
            else None
        )

        if cached is None:
            if archive is None:
                archive = download()
            cached = _parse_spot_curve(archive.read(name), f"{description} ({name})")
            if cache:
                cache.set(
                    source=policy_model.BANK_OF_ENGLAND,
                    dataset=dataset,
                    entity=entity,
                    data=cached,
                )

        frames.append(cached)

    return frames


def get_spot_curve(curve: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily spot curve the Bank of England estimates: the nominal and real
    curves of UK government bonds (gilts), the implied inflation curve, the difference of
    the two and so the UK breakeven inflation, and the curve of overnight index swaps (OIS)
    on SONIA. The curves run from 1985 (OIS from 2009) for maturities of up to 40 years.

    Args:
        curve (str): "nominal", "real", "inflation" or "ois".
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The spot rates as decimals, indexed by day with a column per maturity.
    """
    description = f"Bank of England {curve} yield curve"
    _, current_name = CURVE_ARCHIVES[curve]

    def fetch_current() -> pd.DataFrame:
        response = get_request(
            f"{YIELD_CURVE_URL}{CURRENT_MONTH_ARCHIVE}",
            timeout=120,
            extra_headers=HEADERS,
        )
        archive = zipfile.ZipFile(io.BytesIO(response.content))

        if current_name not in archive.namelist():
            raise ValueError(
                f"The current month of the {description} has no workbook '{current_name}', "
                "which means the Bank of England changed the file."
            )

        return _parse_spot_curve(archive.read(current_name), description)

    try:
        frames = _get_archive_workbooks(curve, int(start_date[:4]), int(end_date[:4]))
    except requests.exceptions.RequestException as error:
        logger.warning(
            "Could not reach the Bank of England for the %s archive (%s).",
            description,
            error,
        )
        frames = []

    current = collect_cached_data(
        source=policy_model.BANK_OF_ENGLAND,
        dataset="curve_current_month",
        entity=curve,
        fetch=fetch_current,
        description=description,
    )
    frames.append(current)

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    spot_curve = pd.concat(frames).sort_index()
    spot_curve = spot_curve[~spot_curve.index.duplicated(keep="last")]

    return spot_curve.loc[pd.Period(start_date, "D") : pd.Period(end_date, "D")]
