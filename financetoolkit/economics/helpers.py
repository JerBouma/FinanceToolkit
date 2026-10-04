"""Economics Helpers"""

__docformat__ = "google"

import time
from collections.abc import Callable

import pandas as pd
import requests

from financetoolkit.cache import frame_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

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
    "MA": "Morocco", "MD": "Moldova", "ME": "Montenegro", "MK": "North Macedonia",
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
PERIOD_FREQUENCIES = {"daily": "D", "weekly": "W-FRI", "monthly": "M"}

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
    except requests.exceptions.RequestException as error:
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

        logger.error(
            "Could not reach %s for the %s (%s). No data is returned for it.",
            source,
            description,
            error,
        )
        return pd.DataFrame()

    if cache is not None and not data.empty:
        cache.set(source=source, dataset=dataset, entity=entity, data=data)

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
    except requests.exceptions.RequestException as error:
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

    if cache is not None and not data.empty:
        cache.store(
            source=source,
            dataset=dataset,
            entity=entity,
            data=data,
            start=fetch_start,
            end=fetch_end,
        )

    if cached_data is not None and not cached_data.empty:
        data = frame_model.merge_frames(cached_data, data)
        data = frame_model.slice_frame(data, start_date, end_date)

    return data


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
    # "annual" is accepted as well, the word the MCP server's parameter description uses.
    period = "yearly" if period.lower() == "annual" else period.lower()

    if period not in supported:
        raise ValueError(
            f"The {indicator} is not available with period='{period}'. Choose from "
            f"{', '.join(repr(option) for option in supported)}."
        )

    return period


def resample_to_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Converts a daily series to a weekly or monthly one by taking the last observation of
    every period, the end-of-period convention central banks and the BIS use for rates.

    Args:
        data (pd.DataFrame): The daily series, indexed by a daily PeriodIndex.
        period (str): "daily", "weekly" or "monthly".

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

    if not chosen:
        return pd.DataFrame()

    combined = pd.concat(
        [
            sources[position][country]
            for country, (_, position) in sorted(chosen.items())
        ],
        axis=1,
    ).sort_index()

    return combined.dropna(how="all")


# The largest difference, as a decimal, two publications of the same series may show for a
# month they share. The figures are rounded to one decimal in percent, so 0.0005 allows
# rounding while catching a different definition.
EXTENSION_TOLERANCE = 0.0005


def extend_with_recent(
    history: pd.DataFrame, recent: pd.DataFrame, country: str
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

    return extended
