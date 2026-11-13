"""Economics Helpers"""

__docformat__ = "google"

import time
from collections.abc import Callable

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
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.sources_model import record_source

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

# The countries the OECD names differently from the Global Macro Database. An OECD series
# combined with a GMDB series (the real interest rate and the misery index, whose inflation
# leg is always the GMDB's) is renamed to the GMDB names first, as columns that differ only
# in name would otherwise not line up and give a column of NaN for each spelling.
OECD_TO_GMDB_COUNTRY_NAMES: dict[str, str] = {
    "Russia": "Russian Federation",
    "Swaziland": "Eswatini",
    "Democratic Republic of the Congo": "Congo DR",
}


def to_gmdb_country_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the country columns of an OECD series to the names the Global Macro Database
    uses, so the two can be combined column by column.

    Args:
        data (pd.DataFrame): The OECD series with a column per country.

    Returns:
        pd.DataFrame: The same series with the GMDB country names.
    """
    return data.rename(columns=OECD_TO_GMDB_COUNTRY_NAMES)


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
START_BUFFER_DAYS = {"daily": 400, "weekly": 400, "monthly": 731}

# Gateway errors, timeouts and dropped connections the public statistics services answer
# with when briefly overloaded. The ECB Data Portal answers around a third of its requests with a 504 at
# random, so these are retried twice, after a growing pause. A rate limit (429) is not
# retried, so a busy source is not asked again.
RETRY_STATUS_CODES = {502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

# Long enough that an entry is never too old to serve when the source cannot be reached.
STALE_TTL_SECONDS = 100 * 365 * 86400


def _fetch_with_retry(fetch: Callable[..., pd.DataFrame], *args: str) -> pd.DataFrame:
    """
    Calls a fetch function, retrying when the source is briefly unavailable.

    Args:
        fetch (Callable[..., pd.DataFrame]): Retrieves the data from the source.
        *args (str): The arguments to pass to the fetch function.

    Returns:
        pd.DataFrame: The data.

    Raises:
        requests.exceptions.RequestException: When the last attempt fails as well, or an
            attempt fails for any other reason than a gateway error, timeout or dropped
            connection.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fetch(*args)
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
        data = _fetch_with_retry(fetch)
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

        logger.error(
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
        data = _fetch_with_retry(fetch, fetch_start, fetch_end)
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

        logger.error(
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
        pd.DatetimeIndex: The business days from start_date up to and including end_date.
    """
    holidays = USExchangeHolidays().holidays(start=start_date, end=end_date)

    return pd.bdate_range(start_date, end_date, freq="C", holidays=holidays)


def buffered_start_date(start_date: str, period: str) -> str:
    """
    Moves a start date back far enough for growth, lag and rolling windows.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        period (str): "daily", "weekly" or "monthly".

    Returns:
        str: The earlier start date (YYYY-MM-DD).
    """
    return (
        pd.Timestamp(start_date) - pd.Timedelta(days=START_BUFFER_DAYS[period])
    ).strftime("%Y-%m-%d")


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


def warn_about_long_run_prices(
    consumer_price_index: pd.DataFrame,
    countries: list[str] | str | None,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """
    Warns about the years of the Global Macro Database's yearly consumer price index, in
    the date range, that its inflation rate does not describe as it reads:

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
    Checks that a period is given as text, so that a wrong type is reported as such
    rather than as an AttributeError from deep inside the retrieval.

    Args:
        period (object): The period as passed by the caller.

    Raises:
        TypeError: When the period is not None and not a string.
    """
    if period is not None and not isinstance(period, str):
        raise TypeError(
            f"The period must be text such as 'monthly' or 'daily', not a {type(period).__name__} "
            f"({period!r})."
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
        ValueError: When the indicator is not published at that frequency.
    """
    check_period_type(period)

    # "annual" is accepted as well, the word the MCP server's parameter description uses.
    period = "yearly" if period.lower() == "annual" else period.lower()

    if period not in supported:
        raise ValueError(
            f"The {indicator} is not available with period='{period}'. Choose from "
            f"{', '.join(repr(option) for option in supported)}."
        )

    return period


def change_over_periods(
    data: pd.DataFrame | pd.Series, periods: int
) -> pd.DataFrame | pd.Series:
    """
    The relative change over a number of periods (12 months, 4 quarters) on consecutive
    periods: a period a source skipped becomes empty first, so the change compares the
    same period a year earlier rather than the row that many places back, and the skipped
    period and the one a year after it are empty.

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
        ValueError: When the measure does not exist, or not at that period or source.
    """
    measure = measure.lower()

    if measure not in CONSUMER_PRICE_MEASURES:
        raise ValueError(
            f"The measure '{measure}' does not exist. Choose from "
            f"{', '.join(repr(option) for option in CONSUMER_PRICE_MEASURES)}."
        )

    if measure != "all" and (period.lower() != "monthly" or oecd_source):
        raise ValueError(
            f"The measure '{measure}' is monthly only, without oecd_source. Use "
            "period='monthly'."
        )

    return measure


def resample_to_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Converts a daily series to a weekly, monthly, quarterly or yearly one by taking the
    last observation of every period, the end-of-period convention central banks and the
    BIS use for rates.

    Args:
        data (pd.DataFrame): The daily series, indexed by a daily PeriodIndex.
        period (str): "daily", "weekly", "monthly", "quarterly" or "yearly".

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


def average_monthly_index(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Converts a monthly index to a quarterly or yearly one by averaging the months of
    every period, the convention statistical offices use for the quarterly and annual
    values of a price index. A period is only averaged once all of its months are
    published, so a partial quarter or year is not mistaken for a full one.

    Args:
        data (pd.DataFrame): The monthly index, indexed by a monthly PeriodIndex.
        period (str): "monthly", "quarterly" or "yearly".

    Returns:
        pd.DataFrame: The index at the requested frequency, NaN for incomplete periods.
    """
    if data.empty or period == "monthly":
        return data

    periods = data.index.asfreq("Q" if period == "quarterly" else "Y")
    grouped = data.groupby(periods)

    return grouped.mean().where(grouped.count() == (3 if period == "quarterly" else 12))


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
    Combines country columns from several sources, taking each country from the source
    with the most recent observation and, when sources are equally recent, from the
    first in the list.

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
        logger.debug(
            "The recent %s figures do not match the history on the months both cover, so "
            "the history is not extended.",
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
    Compounds daily overnight rates into the growth of 1 deposited on the first day, the
    method of the compounded indices the European Central Bank and the New York Fed
    publish for €STR and SOFR:

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
