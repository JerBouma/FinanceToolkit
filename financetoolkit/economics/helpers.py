"""Economics Helpers"""

__docformat__ = "google"

import io
import time
import zipfile
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar

import pandas as pd
import requests
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

from financetoolkit.cache import frame_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.cache.request_model import redact_credentials
from financetoolkit.helpers import split_values
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    join_options,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import record_source
from financetoolkit.utilities.statistics_model import normalize_country_name

logger = get_logger()

# A rise in prices of more than tenfold (1,000%) in a year, the hyperinflations of the long
# run price indices.
HYPERINFLATION_RISE = 9.0

# Two-letter country codes (ISO 3166, as used by Eurostat, the BIS and the economic
# calendar) named as elsewhere in the Economics module, so a country is one column no
# matter which source it came from. "EU" is the euro area in the economic calendar.
COUNTRY_CODES: dict[str, str] = {
    "AE": "United Arab Emirates", "AL": "Albania", "AM": "Armenia", "AO": "Angola",
    "AR": "Argentina", "AT": "Austria", "AU": "Australia", "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BD": "Bangladesh", "BE": "Belgium", "BG": "Bulgaria",
    "BH": "Bahrain", "BR": "Brazil", "BW": "Botswana", "BY": "Belarus", "CA": "Canada",
    "CH": "Switzerland", "CL": "Chile", "CN": "China", "CO": "Colombia", "CR": "Costa Rica",
    "CV": "Cape Verde", "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark", "DO": "Dominican Republic", "EC": "Ecuador", "EE": "Estonia",
    "EG": "Egypt", "ER": "Eritrea", "ES": "Spain", "ET": "Ethiopia", "EU": "Euro Area",
    "FI": "Finland", "FJ": "Fiji", "FR": "France", "GB": "United Kingdom", "GE": "Georgia",
    "GH": "Ghana", "GN": "Guinea", "GR": "Greece", "GT": "Guatemala", "HK": "Hong Kong",
    "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel",
    "IN": "India", "IQ": "Iraq", "IS": "Iceland", "IT": "Italy", "JM": "Jamaica",
    "JO": "Jordan", "JP": "Japan", "KE": "Kenya", "KG": "Kyrgyzstan", "KH": "Cambodia",
    "KR": "South Korea", "KW": "Kuwait", "KZ": "Kazakhstan", "LB": "Lebanon",
    "LK": "Sri Lanka", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya",
    "MA": "Morocco", "MD": "Moldova", "ME": "Montenegro", "MK": "Macedonia",
    "ML": "Mali", "MN": "Mongolia", "MO": "Macau", "MT": "Malta", "MU": "Mauritius",
    "MV": "Maldives", "MW": "Malawi", "MX": "Mexico", "MY": "Malaysia", "MZ": "Mozambique",
    "NA": "Namibia", "NG": "Nigeria", "NL": "Netherlands", "NO": "Norway", "NP": "Nepal",
    "NZ": "New Zealand", "OM": "Oman", "PA": "Panama", "PE": "Peru",
    "PG": "Papua New Guinea", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland",
    "PS": "Palestine", "PT": "Portugal", "PY": "Paraguay", "QA": "Qatar", "RO": "Romania",
    "RS": "Serbia", "RU": "Russia", "RW": "Rwanda", "SA": "Saudi Arabia",
    "SC": "Seychelles", "SE": "Sweden", "SG": "Singapore", "SI": "Slovenia",
    "SK": "Slovakia", "SL": "Sierra Leone", "SN": "Senegal", "ST": "Sao Tome and Principe",
    "SV": "El Salvador", "TH": "Thailand", "TJ": "Tajikistan", "TM": "Turkmenistan",
    "TN": "Tunisia", "TR": "Turkey", "TW": "Taiwan", "TZ": "Tanzania", "UA": "Ukraine",
    "UG": "Uganda", "UK": "United Kingdom", "US": "United States", "UY": "Uruguay",
    "UZ": "Uzbekistan", "VN": "Vietnam", "WL": "World", "XK": "Kosovo",
    "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
    # Aggregates and the codes that differ from ISO 3166 in Eurostat and BIS data.
    "EL": "Greece", "LI": "Liechtenstein", "XM": "Euro Area", "EA": "Euro Area",
    "EU27_2020": "European Union",
}  # fmt: skip

# The periods a high-frequency series can be returned in, with the PeriodIndex frequency
# each one maps to. Weekly periods end on Friday, the last trading day of the week.
PERIOD_FREQUENCIES = {
    "daily": "D",
    "weekly": "W-FRI",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}

# History requested before the start date, so growth, lag and rolling windows can be
# computed for the first periods shown (finalize_dataset slices the result afterwards).
START_BUFFER_DAYS = {"daily": 400, "weekly": 400, "monthly": 731, "quarterly": 1096}

# Gateway errors, timeouts and dropped connections the public statistics services answer
# with when briefly overloaded. The ECB Data Portal answers around a third of its requests with a 504 at
# random, so these are retried twice, after a growing pause. A rate limit (429) is not
# retried, so a busy source is not asked again.
RETRY_STATUS_CODES = {502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

# Long enough that an entry is never too old to serve when the source cannot be reached.
STALE_TTL_SECONDS = 100 * 365 * 86400

# The sources that could not be reached during a call that retrieves many series (see
# stop_at_unreachable_sources), shared with the worker threads it starts.
_UNREACHABLE_SOURCES: ContextVar[set[str] | None] = ContextVar(
    "financetoolkit_unreachable_sources", default=None
)


class SourceUnreachableError(requests.exceptions.ConnectionError):
    """A request not sent, as its source could not be reached earlier in the same call."""


@contextmanager
def stop_at_unreachable_sources() -> Generator[None]:
    """
    Stops requesting a source for the rest of the block once it could not be reached (no
    connection, a timeout, a server error or a rate limit, after the retries), for a method
    that retrieves many series of the same sources: each further series would otherwise
    wait through the same retries before failing alike. The series not requested are served
    from the cache where it has them and are otherwise empty, logged once with the first.
    A block inside another shares its record. It can also decorate a method.

    Yields:
        None: Nothing; the sources are tracked until the block ends.
    """
    if _UNREACHABLE_SOURCES.get() is not None:
        yield
        return

    token = _UNREACHABLE_SOURCES.set(set())
    try:
        yield
    finally:
        _UNREACHABLE_SOURCES.reset(token)


def is_unreachable(source: str) -> bool:
    """
    Whether a source could not be reached earlier in the current stop_at_unreachable_sources
    block, for a caller that reports the parts it could not retrieve in one line.

    Args:
        source (str): The source, as the cache names it (see policy_model).

    Returns:
        bool: True when the source could not be reached in this block, False otherwise and
        outside such a block.
    """
    unreachable = _UNREACHABLE_SOURCES.get()
    return unreachable is not None and source in unreachable


def _fetch_with_retry(
    fetch: Callable[..., pd.DataFrame], *args: str, source: str = "The source"
) -> pd.DataFrame:
    """
    Calls a fetch function, retrying when the source is briefly unavailable.

    Args:
        fetch (Callable[..., pd.DataFrame]): Retrieves the data from the source.
        *args (str): The arguments to pass to the fetch function.
        source (str, optional): The source, named when its answer is not JSON. Defaults
            to "The source".

    Returns:
        pd.DataFrame: The data.

    Raises:
        requests.exceptions.RequestException: When the last attempt fails as well, or an
            attempt fails for any other reason than a gateway error, timeout or dropped
            connection.
        ValueError: When the source answers with something other than the JSON asked for.
    """
    unreachable = _UNREACHABLE_SOURCES.get()
    if unreachable is not None and source in unreachable:
        raise SourceUnreachableError(
            f"{source} could not be reached for an earlier request of this call, so it is "
            "not asked again"
        )

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fetch(*args)
        except requests.exceptions.JSONDecodeError as error:
            # requests counts an unreadable body as a request error, but the source was
            # reached: it answered in another format (an HTML error page, an empty body).
            raise ValueError(
                f"{source} answered with something other than JSON ({error})."
            ) from error
        except (
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as error:
            response = getattr(error, "response", None)
            status_code = response.status_code if response is not None else None
            retryable = (
                not isinstance(error, requests.exceptions.HTTPError)
                or status_code in RETRY_STATUS_CODES
            )

            if not retryable or attempt == RETRY_ATTEMPTS:
                # A missing series (404) is that series only; anything else is the source.
                if unreachable is not None and (
                    status_code is None
                    or status_code >= 500  # noqa: PLR2004
                    or status_code == requests.codes.too_many_requests
                ):
                    unreachable.add(source)
                raise

            time.sleep(RETRY_DELAY_SECONDS * attempt)

    # Unreachable: the last attempt either returns or raises.
    return pd.DataFrame()


def collect_cached_data(
    source: str,
    dataset: str,
    entity: str,
    fetch: Callable[[], pd.DataFrame],
    description: str,
    expect_empty: bool = False,
    fallback: str | None = None,
    quiet: bool = False,
    failures: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Retrieves a dataset through the cache, serving the last stored copy when the source
    cannot be reached.

    The statistical offices and central banks used here publish a series whole and
    revise its recent past, so a series is cached as a whole and refreshed once its
    time-to-live has passed. A network failure is not a reason to return nothing: the
    most recent copy is served instead, with a warning, since a few days old figure is
    more useful than none. A response that does not have the expected shape is a
    different matter. That means the source changed its format, so the ValueError the
    fetch function raises is passed on rather than answered with stale data.

    Args:
        source (str): The cache source, e.g. "Eurostat".
        dataset (str): The cache dataset within that source.
        entity (str): The series or query that identifies the data.
        fetch (Callable[[], pd.DataFrame]): Retrieves the data from the source.
        description (str): What is retrieved, used in the log messages.
        expect_empty (bool): Whether an empty answer is expected, such as for a period
            before a source starts, so it is not warned about. Defaults to False.
        fallback (str | None): What is used instead when the source cannot be reached and
            nothing is cached, for data that only adds to another source's series. The
            failure is then logged as information with this note rather than as an error.
            Defaults to None.
        quiet (bool): Whether a source that cannot be reached, with nothing cached, is
            logged at debug level only, for a caller that retrieves many parts (such as one
            file per year) and reports the missing ones in a single line. Defaults to False.
        failures (dict[str, str] | None): Where a failed request with nothing cached is
            recorded instead of logged, as the reason per entity (such as "EEX answered HTTP
            403"), for a caller that reports the missing parts with their reasons in a single
            line. A request not sent because the source could not be reached earlier in the
            call is not recorded: the caller checks is_unreachable. Defaults to None.

    Returns:
        pd.DataFrame: The data, or an empty DataFrame when the source cannot be reached
        and nothing is cached.
    """
    cache = get_active_cache()

    if cache is not None:
        cached_data = cache.get(source=source, dataset=dataset, entity=entity)

        if cached_data is not None:
            return cached_data

    try:
        data = _fetch_with_retry(fetch, source=source)
    except requests.exceptions.RequestException as request_error:
        # A request error names its URL, so a key in it is removed before it is logged.
        error = redact_credentials(str(request_error))
        stale_data = (
            cache.get(
                source=source, dataset=dataset, entity=entity, ttl=STALE_TTL_SECONDS
            )
            if cache is not None
            else None
        )

        if stale_data is not None:
            logger.warning(
                "Could not reach %s for the %s (%s). Serving the most recently cached "
                "copy instead, which may not include the latest releases.",
                source,
                description,
                error,
            )
            return stale_data

        if fallback:
            logger.info(
                "Could not reach %s for the %s (%s), so %s.",
                source,
                description,
                error,
                fallback,
            )
            return pd.DataFrame()

        if failures is not None:
            if not isinstance(request_error, SourceUnreachableError):
                response = getattr(request_error, "response", None)
                # The status says why; the error itself names the URL, which differs per entity.  # noqa: E501
                failures[entity] = (
                    f"{source} answered HTTP {response.status_code}"
                    if response is not None
                    else f"{source} could not be reached ({error})"
                )
            return pd.DataFrame()

        (
            logger.debug
            if quiet or isinstance(request_error, SourceUnreachableError)
            else logger.error
        )(
            "Could not reach %s for the %s (%s). No data is returned for it.",
            source,
            description,
            error,
        )
        return pd.DataFrame()

    if cache is not None and not data.empty:
        cache.set(source=source, dataset=dataset, entity=entity, data=data)

    if data.empty and not expect_empty:
        logger.warning("%s returned no %s.", source, description)

    return data


def collect_ranged_data(
    source: str,
    dataset: str,
    entity: str,
    fetch: Callable[[str, str], pd.DataFrame],
    start_date: str,
    end_date: str,
    description: str,
) -> pd.DataFrame:
    """
    Retrieves a date range of a series through the cache, requesting only what is not
    cached yet.

    For sources that accept a date range. The cache works out which part of the range
    was never retrieved or has passed its time-to-live, narrowed to the dataset's
    revision window, so a rerun asks for the recent tail rather than the whole range
    again and no request is made at all within the time-to-live. That keeps the number
    and size of the requests to these public services small. When the source cannot be
    reached, what is cached for the range is served instead, with a warning.

    Args:
        source (str): The cache source, e.g. "BIS".
        dataset (str): The cache dataset within that source.
        entity (str): The series or query that identifies the data.
        fetch (Callable[[str, str], pd.DataFrame]): Retrieves the data between a start and
            end date (YYYY-MM-DD) from the source.
        start_date (str): The start date of the range (YYYY-MM-DD).
        end_date (str): The end date of the range (YYYY-MM-DD).
        description (str): What is retrieved, used in the log messages.

    Returns:
        pd.DataFrame: The data within the range, or an empty DataFrame when the source
        cannot be reached and nothing is cached.
    """
    # A range that starts after today has nothing to retrieve yet, so the source is not asked.
    if pd.Timestamp(start_date) > pd.Timestamp.today().normalize():
        logger.warning(
            "The %s requested from %s lies in the future, so there is no data for it yet.",
            description,
            start_date,
        )
        return pd.DataFrame()

    cache = get_active_cache()
    cached_data = None
    fetch_start, fetch_end = start_date, end_date

    if cache is not None:
        plan = cache.plan(
            source=source,
            dataset=dataset,
            entities=[entity],
            start=start_date,
            end=end_date,
        )
        cached_data = plan.cached_frame(entity)
        fetch_span = plan.get_fetch_span(entity)

        if fetch_span is None:
            return cached_data if cached_data is not None else pd.DataFrame()

        fetch_start = fetch_span[0].strftime("%Y-%m-%d")
        fetch_end = fetch_span[1].strftime("%Y-%m-%d")

    try:
        data = _fetch_with_retry(fetch, fetch_start, fetch_end, source=source)
    except requests.exceptions.RequestException as request_error:
        # A request error names its URL, so a key in it is removed before it is logged.
        error = redact_credentials(str(request_error))
        if cached_data is not None and not cached_data.empty:
            logger.warning(
                "Could not reach %s for the %s (%s). Serving the cached copy instead, which "
                "may not include the latest releases.",
                source,
                description,
                error,
            )
            return cached_data

        (
            logger.debug
            if isinstance(request_error, SourceUnreachableError)
            else logger.error
        )(
            "Could not reach %s for the %s (%s). No data is returned for it.",
            source,
            description,
            error,
        )
        return pd.DataFrame()
    except ValueError as error:
        # A response that cannot be read for the missing part (a format change, or an
        # empty span a source answers in another shape) leaves the cached history intact.
        if cached_data is None or cached_data.empty:
            raise

        logger.warning(
            "Could not read the %s from %s between %s and %s (%s). Serving the cached "
            "copy instead, which may not include the latest releases.",
            description,
            source,
            fetch_start,
            fetch_end,
            error,
        )
        return cached_data

    if cache is not None and not data.empty:
        cache.store(
            source=source,
            dataset=dataset,
            entity=entity,
            data=data,
            start=fetch_start,
            end=fetch_end,
        )

    if data.empty and (cached_data is None or cached_data.empty):
        logger.warning(
            "%s returned no %s between %s and %s.",
            source,
            description,
            fetch_start,
            fetch_end,
        )

    if cached_data is not None and not cached_data.empty:
        data = frame_model.merge_frames(cached_data, data)
        data = frame_model.slice_frame(data, start_date, end_date)

    return data


class USExchangeHolidays(AbstractHolidayCalendar):
    """
    The full-day holidays of the US futures and stock exchanges (CME Group, NYSE): New
    Year's Day (moved to Monday from a Sunday, not to Friday from a Saturday), Martin Luther
    King Jr. Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth (from 2022),
    Independence Day, Labor Day, Thanksgiving and Christmas, a Saturday holiday moving to
    the Friday before and a Sunday holiday to the Monday after.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2022-01-01",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def get_exchange_business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Returns the business days of the US exchanges between two dates: the weekdays that are
    not a holiday of USExchangeHolidays.

    Args:
        start_date (str): The first day (YYYY-MM-DD).
        end_date (str): The last day (YYYY-MM-DD).

    Returns:
        pd.DatetimeIndex: The business days from the start date up to and including the end date.
    """
    holidays = USExchangeHolidays().holidays(start=start_date, end=end_date)

    return pd.bdate_range(start_date, end_date, freq="C", holidays=holidays)


def buffered_start_date(start_date: str, period: str) -> str:
    """
    Moves a start date back far enough for growth, lag and rolling windows.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        period (str): The frequency of the data. Options are: "daily" (daily data), "weekly" (weekly data), "monthly"
            (monthly data), "quarterly" (quarterly data).

    Returns:
        str: The earlier start date (YYYY-MM-DD).
    """
    return (
        pd.Timestamp(start_date) - pd.Timedelta(days=START_BUFFER_DAYS[period])
    ).strftime("%Y-%m-%d")


def normalize_requested_countries(
    countries: list[str] | str | None, known: Iterable[str]
) -> list[str] | None:
    """
    Turn the requested countries into a list of the names the sources use, so a short form or another casing selects
    the same sources as the name itself.

    A string may hold several countries separated by commas; blank entries (a trailing comma in "Germany,") are
    skipped and a country spelled twice is listed once. A country that matches nothing known is kept as given, so it
    is reported as unavailable later on.

    Args:
        countries (list[str] | str | None): The countries as passed to a method.
        known (Iterable[str]): The country names the sources use.

    Returns:
        list[str] | None: The normalized countries, or None when every country is requested.
    """
    if countries is None:
        return None

    requested = countries.split(",") if isinstance(countries, str) else countries
    known = list(known)

    return list(
        dict.fromkeys(
            normalize_country_name(str(country).strip(), known)
            for country in requested
            if str(country).strip()
        )
    )


def require_columns(data: pd.DataFrame, columns: set[str], description: str) -> None:
    """
    Checks that a response has the columns the parsing depends on.

    Args:
        data (pd.DataFrame): The parsed response.
        columns (set[str]): The columns that must be present.
        description (str): What was retrieved, used in the error message.

    Raises:
        ValueError: When a column is missing, which means the source changed its format.
    """
    if missing := columns - set(data.columns):
        raise ValueError(
            f"The response for the {description} is missing the {', '.join(sorted(missing))} "
            "column(s), which means the source changed its format and the data cannot be "
            "interpreted."
        )


def open_zip_archive(response: requests.Response, description: str) -> zipfile.ZipFile:
    """
    Opens a downloaded zip archive, such as a workbook or a set of CSV files.

    Args:
        response (requests.Response): The response holding the archive.
        description (str): What was retrieved, used in the error message.

    Returns:
        zipfile.ZipFile: The archive.

    Raises:
        ValueError: When the answer is not a zip archive, such as an HTML error page.
    """
    try:
        return zipfile.ZipFile(io.BytesIO(response.content))
    except zipfile.BadZipFile as error:
        content_type = response.headers.get("Content-Type") or "no content type"
        raise ValueError(
            f"The {description} is not a zip archive: the source answered with "
            f"{content_type} instead."
        ) from error


def warn_about_long_run_prices(
    consumer_price_index: pd.DataFrame,
    countries: list[str] | str | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """
    Warn about the years of the Global Macro Database's yearly consumer price index whose inflation rate does not
    describe the price change as it reads.

    Warns about the years of the Global Macro Database's yearly consumer price index, in the date range, that its
    inflation rate does not describe as it reads:

    - A value repeated in consecutive years, which the source carried forward rather
      than measured, so that the inflation of those years is 0 by construction. Germany's
      index repeats its 1912 value in 1913 and 1914, where the Jordà-Schularick-Taylor
      Macrohistory Database has prices rising 0.7% and 2.8%.
    - A hyperinflation (prices rising more than tenfold in a year). Yearly figures are
      averages over the year, so the year after one still shows a large rise from that
      average when a currency reform ended it within the year, as Germany's Rentenmark
      did in November 1923 (1924: +723%). The index is chained across currency reforms
      and redenominations, in today's currency, so it has no break at the reform.

    Args:
        consumer_price_index (pd.DataFrame): The yearly index, a column per country.
        countries (list[str] | str | None): The requested countries, all when None.
        start_date (str | None): The start of the date range.
        end_date (str | None): The end of the date range.
    """
    selected = (
        consumer_price_index.columns
        if countries is None
        else [
            country
            for country in split_values(countries)
            if country in consumer_price_index.columns
        ]
    )
    window = consumer_price_index.loc[start_date:end_date, selected]

    def years(index: pd.Index) -> str:
        names = [str(year) for year in index]
        return " and ".join(
            [", ".join(names[:-1]), names[-1]] if len(names) > 1 else names
        )

    for country in window.columns:
        index = window[country].dropna()
        repeated = index.index[1:][(index.to_numpy()[1:] == index.to_numpy()[:-1])]

        if len(repeated):
            logger.warning(
                "The Global Macro Database repeats the %s consumer price index of the "
                "year before in %s: a value carried forward rather than measured, so the "
                "inflation rate of those years is 0 by construction.",
                country,
                years(repeated),
            )

        hyperinflation = index.index[index.pct_change() > HYPERINFLATION_RISE]

        if len(hyperinflation):
            logger.warning(
                "Prices in %s rose more than tenfold in %s. Yearly figures are averages "
                "over the year, so the year after shows a large rise from that average "
                "even when a currency reform ended the hyperinflation within the year; "
                "the index is chained across currency reforms and redenominations, in "
                "today's currency.",
                country,
                years(hyperinflation),
            )


def to_us_series(observations: dict, source: str, description: str) -> pd.DataFrame:
    """
    Turns the dated observations of a US-only series into the single "United States"
    column the Economics module returns for it, as FRED shapes its series.

    Args:
        observations (dict): The value per day (pd.Period with a daily frequency), such as
            the first day of each month or the day each week ends.
        source (str): Where the observations came from, used in the error message.
        description (str): What was retrieved, used in the error message.

    Returns:
        pd.DataFrame: A single "United States" column indexed by day ("Date").

    Raises:
        ValueError: When there are no observations, which means the source changed its
            format.
    """
    if not observations:
        raise ValueError(
            f"The {source} has no {description}, which means its layout changed."
        )

    index = pd.PeriodIndex(list(observations), freq="D", name="Date")

    return pd.DataFrame(
        {"United States": list(observations.values())}, index=index
    ).sort_index()


def check_period_type(period: object) -> None:
    """
    Checks that a period is given as text, so that a wrong type (such as a list) is reported
    as a mistake in the call rather than as an error from deep inside the retrieval.

    Args:
        period (object): The period as passed by the caller.

    Raises:
        InvalidArgumentError: When the period is not None and not a string.
    """
    if period is not None and not isinstance(period, str):
        raise InvalidArgumentError(
            f"The period must be text such as 'monthly' or 'daily', not a {type(period).__name__} "
            f"({period!r}). Please pass one period as text."
        )


def validate_period(period: str, supported: list[str], indicator: str) -> str:
    """
    Normalises a period and checks that the indicator is published at that frequency.

    Args:
        period (str): The requested period, e.g. "monthly".
        supported (list[str]): The periods the indicator supports.
        indicator (str): The name of the indicator, used in the error message.

    Returns:
        str: The period in lower case.

    Raises:
        InvalidArgumentError: When the indicator is not published at that frequency,
            naming the periods it is.
    """
    check_period_type(period)

    # "annual" is accepted as well, the word the MCP server's parameter description uses.
    period = "yearly" if period.lower() == "annual" else period.lower()

    if period not in supported:
        raise InvalidArgumentError(
            f"The {indicator} is not available with period={period!r}: period must be one "
            f"of {join_options(supported)}. Please pass one of these as period."
        )

    return period


def change_over_periods(
    data: pd.DataFrame | pd.Series, periods: int
) -> pd.DataFrame | pd.Series:
    """
    Calculate the relative change over a number of periods on consecutive periods, so a period a source skipped does
    not shift the comparison.

    Args:
        data (pd.DataFrame | pd.Series): The values, indexed by a monthly, quarterly or
            yearly PeriodIndex.
        periods (int): The number of periods to compare with, e.g. 12 for monthly data.

    Returns:
        pd.DataFrame | pd.Series: The change as a decimal, on consecutive periods.
    """
    if data.empty:
        return data

    consecutive = data.reindex(
        pd.period_range(
            data.index.min(),
            data.index.max(),
            freq=data.index.freq,
            name=data.index.name,
        )
    )

    return consecutive.pct_change(periods, fill_method=None)


CONSUMER_PRICE_MEASURES = [
    "all",
    "core",
    "energy",
    "food",
    "goods",
    "services",
    "shelter",
]


def validate_consumer_price_measure(
    measure: str, period: str, oecd_source: bool = False
) -> str:
    """
    Normalises the measure of a consumer price indicator and checks that it exists for the
    requested data: the core index and the components are monthly and not from the OECD.

    Args:
        measure (str): The requested measure, e.g. "core".
        period (str): The period of the request.
        oecd_source (bool): Whether the OECD is the source. Defaults to False.

    Returns:
        str: The measure in lower case.

    Raises:
        InvalidArgumentError: When the measure does not exist, or not at that period or
            source.
    """
    measure = measure.lower()

    check_option("measure", measure, CONSUMER_PRICE_MEASURES)

    if measure != "all" and (period.lower() != "monthly" or oecd_source):
        raise InvalidArgumentError(
            f"The measure '{measure}' is monthly only, without oecd_source. Use "
            "period='monthly'."
        )

    return measure


def resample_to_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Converts a daily series to a weekly, monthly, quarterly or yearly one by taking the last observation of every
    period, the end-of-period convention central banks and the BIS use for rates.

    Args:
        data (pd.DataFrame): The daily series, indexed by a daily PeriodIndex.
        period (str): The frequency to resample to. Options are: "daily" (daily data), "weekly" (weekly data),
            "monthly" (monthly data), "quarterly" (quarterly data), "yearly" (yearly data).

    Returns:
        pd.DataFrame: The series at the requested frequency.
    """
    if data.empty or period == "daily":
        return data

    resampled = data.copy()
    resampled.index = resampled.index.to_timestamp()

    resampled = resampled.groupby(
        resampled.index.to_period(PERIOD_FREQUENCIES[period])
    ).last()
    resampled.index.name = None

    return resampled


# How the weeks of a weekly series can be combined into a month, quarter or year.
WEEKLY_AGGREGATIONS = ["mean", "max", "min", "last"]


def aggregate_weekly_series(
    data: pd.DataFrame, period: str, aggregation: str = "mean"
) -> pd.DataFrame:
    """
    Convert a weekly series, indexed by the day each week ends, to a monthly, quarterly or yearly one: the average,
    highest, lowest or last week of every period.

    A week belongs to the period of the day it ends, as FRED aggregates its weekly series. A period is only returned
    once all of its weeks are published, so a month that is still running is not mistaken for a full one (nor the
    first period when the data starts within it).

    Args:
        data (pd.DataFrame): The weekly series, indexed by a daily PeriodIndex of the days
            the weeks end (all on the same weekday).
        period (str): The frequency to combine the weeks into. Options are: "weekly" (the data as it is), "monthly"
            (monthly data), "quarterly" (quarterly data), "yearly" (yearly data).
        aggregation (str, optional): How the weeks are combined. Options are: "mean" (the average of its weeks), "max"
            (the highest week), "min" (the lowest week), "last" (the last week). Defaults to "mean".

    Returns:
        pd.DataFrame: The series at the requested frequency.

    Raises:
        ValueError: When the aggregation is not one of the options.
    """
    if aggregation not in WEEKLY_AGGREGATIONS:
        raise ValueError(
            f"The aggregation '{aggregation}' does not exist. Choose from "
            f"{', '.join(repr(option) for option in WEEKLY_AGGREGATIONS)}."
        )

    if data.empty or period == "weekly":
        return data

    days = data.index.to_timestamp()
    periods = days.to_period(PERIOD_FREQUENCIES[period])
    grouped = data.groupby(periods)
    aggregated = grouped.agg(aggregation)

    # The weeks every period has: its days on the weekday the weeks end.
    weekday = days[0].strftime("%a").upper()
    expected = pd.Series(
        [
            len(pd.date_range(period.start_time, period.end_time, freq=f"W-{weekday}"))
            for period in aggregated.index
        ],
        index=aggregated.index,
    )

    aggregated = aggregated.where(grouped.count().eq(expected, axis=0))
    aggregated.index.name = data.index.name

    return aggregated.dropna(how="all")


def average_monthly_index(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Converts a monthly index to a quarterly or yearly one by averaging the months of every period, the convention
    statistical offices use for the quarterly and annual values of a price index.

    A period is only averaged once all of its months are published, so a partial quarter or year is not mistaken for a
    full one.

    Args:
        data (pd.DataFrame): The monthly index, indexed by a monthly PeriodIndex.
        period (str): The frequency to average to. Options are: "monthly" (monthly data), "quarterly" (quarterly
            data), "yearly" (yearly data).

    Returns:
        pd.DataFrame: The index at the requested frequency, NaN for incomplete periods.
    """
    if data.empty or period == "monthly":
        return data

    periods = data.index.asfreq("Q" if period == "quarterly" else "Y")
    grouped = data.groupby(periods)

    return grouped.mean().where(grouped.count() == (3 if period == "quarterly" else 12))


# The consecutive quarters a monthly series has to repeat its value in before those quarters
# are taken as a quarterly series spread over its months. A monthly series repeats one value
# over a quarter now and then (about 2% of the quarters of the BIS consumer price indices),
# never over several quarters in a row.
MINIMUM_REPEATED_QUARTERS = 2


def keep_quarter_end_of_repeated_quarters(data: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the last month of each quarter in which a monthly series repeats the quarter's value over its months,
    such as the BIS consumer prices of Australia and New Zealand, which publish their CPI quarterly.

    Repeated months would otherwise show a monthly rate of change of 0 twice a quarter and count each quarter three
    times in averages and rolling windows.

    The decision is made per country and per quarter, so a country that moves from
    quarterly to monthly publication keeps its monthly months while its earlier quarters
    are thinned out. A complete quarter counts as repeated when its three months hold a
    single value, and repeated quarters are thinned out only in runs of at least
    MINIMUM_REPEATED_QUARTERS consecutive quarters, since a monthly series repeats a
    value over a single quarter now and then. A quarter cut off by the start or end of
    the range (two months with a single value) is thinned out as well when it adjoins
    such a run, keeping its last month in the range. A range that holds fewer complete
    repeated quarters in a row than MINIMUM_REPEATED_QUARTERS, such as a range of a few
    months, cannot be told apart from a monthly series and is returned unchanged.

    Args:
        data (pd.DataFrame): The series, indexed by a monthly PeriodIndex with a column
            per country.

    Returns:
        pd.DataFrame: The series with every month but the last of each repeated quarter
        empty.
    """
    if data.empty:
        return data

    quarter_of_month = data.index.asfreq("Q")
    quarters = data.groupby(quarter_of_month)
    # Every quarter of the range, so that a quarter without any month in the index
    # interrupts a run instead of joining the quarters around it.
    all_quarters = pd.period_range(
        quarter_of_month.min(), quarter_of_month.max(), freq="Q"
    )
    count = quarters.count().reindex(all_quarters, fill_value=0)
    single_value = (quarters.max() == quarters.min()).reindex(
        all_quarters, fill_value=False
    )
    repeated = single_value & (count == 3)  # noqa: PLR2004
    cut_off = single_value & (count == 2)  # noqa: PLR2004

    # The length of the run of consecutive repeated quarters each quarter belongs to.
    run_id = (~repeated).cumsum()
    run_length = repeated.apply(
        lambda column: column.groupby(run_id[column.name]).transform("sum")
    )
    thinned_quarters = repeated & (run_length >= MINIMUM_REPEATED_QUARTERS)
    thinned_quarters |= cut_off & (
        thinned_quarters.shift(1, fill_value=False)
        | thinned_quarters.shift(-1, fill_value=False)
    )

    if not thinned_quarters.to_numpy().any():
        return data

    thinned_months = thinned_quarters.reindex(quarter_of_month).set_axis(data.index)

    # The last observed month of each quarter, which is the third month of a complete one.
    observed = data.notna()
    observed_from_here = (
        observed.iloc[::-1].groupby(quarter_of_month[::-1]).cumsum().iloc[::-1]
    )
    last_observed = observed & (observed_from_here == 1)

    return data.mask(thinned_months & ~last_observed)


def choose_sources(sources: list[pd.DataFrame]) -> dict[str, int]:
    """
    Per country, the position of the source with the most recent observation and, when
    sources are equally recent, the first in the list (see combine_sources).

    Args:
        sources (list[pd.DataFrame]): The sources, most preferred first, each indexed by
            period with one column per country.

    Returns:
        dict[str, int]: Per country, the position of its source in the list.
    """
    chosen: dict[str, tuple[pd.Period, int]] = {}

    for position, source in enumerate(sources):
        if source is None or source.empty:
            continue

        for country in source.columns:
            last_observation = source[country].last_valid_index()

            if last_observation is None:
                continue

            if country not in chosen or last_observation > chosen[country][0]:
                chosen[country] = (last_observation, position)

    return {country: position for country, (_, position) in chosen.items()}


def combine_sources(sources: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines country columns from several sources, taking each country from the source with the most recent
    observation and, when sources are equally recent, from the first in the list.

    A source can keep a country it no longer updates, such as the United Kingdom in
    Eurostat data since Brexit, so the most recent observation decides rather than the
    order alone. Values are never mixed within a column, since sources use different
    base years and seasonal adjustment, so splicing them would introduce breaks that
    look like real movements.

    Args:
        sources (list[pd.DataFrame]): The sources, most preferred first, each indexed by
            period with one column per country.

    Returns:
        pd.DataFrame: One column per country, indexed by period.
    """
    chosen = choose_sources(sources)

    if not chosen:
        return pd.DataFrame()

    combined = pd.concat(
        [sources[position][country] for country, position in sorted(chosen.items())],
        axis=1,
    ).sort_index()

    return combined.dropna(how="all")


# The largest difference, as a decimal, two publications of the same series may show for a
# month they share. The figures are rounded to one decimal in percent, so 0.0005 allows
# rounding while catching a different definition.
EXTENSION_TOLERANCE = 0.0005


def extend_with_recent(
    history: pd.DataFrame,
    recent: pd.DataFrame,
    country: str,
    recent_sources: set[str] | None = None,
) -> pd.DataFrame:
    """
    Extends a country's series with the months a faster publisher of the same figures
    already has, such as the BLS for the US unemployment rate the OECD republishes.

    Only observations after the last one in the history are added, and only when both
    publishers agree on every month they share. When they do not, the recent source is
    not the same series after all and the history is returned unchanged, so a change in
    definition can never introduce a break.

    Args:
        history (pd.DataFrame): The full series, one column per country.
        recent (pd.DataFrame): The faster publisher's recent observations for the country.
        country (str): The column to extend.
        recent_sources (set[str] | None): The sources of the recent observations, held
            back with sources_model.collect_sources, which are recorded only when they
            extend the history. Defaults to None.

    Returns:
        pd.DataFrame: The history with the newer observations added to that column.
    """
    if (
        history.empty
        or recent.empty
        or country not in history.columns
        or country not in recent.columns
    ):
        return history

    existing = history[country].dropna()
    newer = recent[country].dropna()

    if existing.empty:
        return history

    shared = existing.index.intersection(newer.index)

    if (
        shared.empty
        or (existing[shared] - newer[shared]).abs().max() > EXTENSION_TOLERANCE
    ):
        # Logged as information rather than a warning: the history is complete as it
        # is, it only lacks the days or months the faster publisher has since.
        logger.info(
            "The recent %s figures do not match the history on the periods both cover, so "
            "the history is not extended with them.",
            country,
        )
        return history

    additions = newer[newer.index > existing.index[-1]]

    if additions.empty:
        return history

    extended = history.reindex(history.index.union(additions.index))
    extended.loc[additions.index, country] = additions

    for source in recent_sources or ():
        record_source(source)

    return extended


# The day count of each overnight rate in its money market: interest accrues over 360
# days a year for the euro and the US dollar and over 365 for sterling and the yen.
OVERNIGHT_RATE_DAY_COUNTS = {
    "Euro Area": 360,
    "United States": 360,
    "United Kingdom": 365,
    "Japan": 365,
}


def compound_overnight_rate(
    rates: pd.DataFrame, day_counts: dict[str, int]
) -> pd.DataFrame:
    """
    Compound daily overnight rates into the growth of 1 deposited on the first day, what a deposit at the overnight
    rate grows to.

    Compounds daily overnight rates into the growth of 1 deposited on the first day, the method of the compounded
    indices the European Central Bank and the New York Fed publish for €STR and SOFR:

    - Growth (t) = Growth (t-1) × (1 + Rate (t-1) × Days (t-1) / Day Count)

    where Days (t-1) is the number of calendar days the rate of the previous publication
    day applies to, three over a weekend, and the Day Count is 360 or 365. The value on a
    day includes the interest up to that day, so the first day is 1. Each country is
    compounded over its own publication days.

    Args:
        rates (pd.DataFrame): The daily rates as decimals, indexed by a daily PeriodIndex
            with a column per country.
        day_counts (dict[str, int]): The day count of every country's rate.

    Returns:
        pd.DataFrame: The growth of 1 per country, NaN on days without a publication.
    """
    growth = {}

    for country in rates.columns:
        rate = rates[country].dropna()
        days = (
            pd.Series(rate.index.to_timestamp(), index=rate.index)
            .diff()
            .shift(-1)
            .dt.days
        )
        growth[country] = (
            (1 + rate * days / day_counts[country]).shift(1, fill_value=1.0).cumprod()
        )

    return pd.DataFrame(growth, index=rates.index, columns=rates.columns)


def rebase_to_first(data: pd.DataFrame) -> pd.DataFrame:
    """
    Divides every column by its first value, so each series is the growth of 1 since its
    first observation (a country that starts later starts from its own first value).

    Args:
        data (pd.DataFrame): The series, a column per country.

    Returns:
        pd.DataFrame: The growth of 1 per column.
    """
    return data if data.empty else data / data.bfill().iloc[0]
