"""FinancialModelingPrep Model"""

__docformat__ = "google"

import re
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit import helpers
from financetoolkit.discovery.discovery_model import get_cached_financial_data
from financetoolkit.economics.helpers import COUNTRY_CODES
from financetoolkit.utilities import validation_model
from financetoolkit.utilities.logger_model import get_logger

# The endpoint returns at most this many days per request, so longer ranges are split.
ECONOMIC_CALENDAR_WINDOW_DAYS = 90

# Releases that are levels of a survey or an index rather than percentages, even where the
# calendar labels their unit "%", unless the name says it is their rate of change.
SURVEY_LEVEL_PATTERN = re.compile(
    r"\bPMI\b|Confidence|Sentiment|Optimism|Climate|Expectations|Outlook|\bIndex\b(?!-)",
    flags=re.IGNORECASE,
)
RATE_OF_CHANGE_PATTERN = re.compile(r"\b(?:YoY|MoM|QoQ)\b", flags=re.IGNORECASE)

logger = get_logger()

# The economic calendar uses two-letter country codes, named like the rest of the module.
ECONOMIC_CALENDAR_COUNTRIES = COUNTRY_CODES


def _windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """
    Splits a date range into consecutive windows the endpoint accepts in one request.

    Args:
        start_date (str): The start date, formatted as YYYY-MM-DD.
        end_date (str): The end date, formatted as YYYY-MM-DD.

    Returns:
        list[tuple[str, str]]: The (start, end) date of every window.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    windows = []

    while start <= end:
        window_end = min(start + timedelta(days=ECONOMIC_CALENDAR_WINDOW_DAYS - 1), end)
        windows.append((start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        start = window_end + timedelta(days=1)

    return windows


def get_economic_calendar(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    countries: str | list[str] | None = None,
    currencies: str | list[str] | None = None,
    impact: str | list[str] | None = None,
    events: str | list[str] | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the scheduled releases of economic data (such as inflation, employment and GDP
    figures) with the previous value, the consensus estimate and, once published, the
    actual value, optionally narrowed down to specific countries, currencies and impact.

    The endpoint returns at most 90 days per request, so a longer range is retrieved in
    90-day windows. Each window is cached the same way as the discovery calendars, since
    the calendar is updated as releases come in. The filters are applied afterwards.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with. Defaults to None, which
            together with no end date returns the endpoint's default of the last 90 days.
        end_date (str, optional): The end date to filter data with. Defaults to None, which is
            today when a start date is given.
        countries (str | list[str], optional): The countries to keep, as names ("United States")
            or codes ("US"), in a list or comma-separated. Defaults to None, which keeps all.
        currencies (str | list[str], optional): The currencies to keep, e.g. "USD" or
            ["USD", "EUR"]. Defaults to None, which keeps all.
        impact (str | list[str], optional): The market impact to keep: "Low", "Medium" and/or
            "High", or "All" for every release. Defaults to None, which keeps all.
        events (str | list[str], optional): Parts of event names to keep, matched without
            regard to case, e.g. "PMI" or ["CPI", "Unemployment"]. Defaults to None, which
            keeps every event.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the economic data releases, indexed by date. Values of
        releases quoted in percent and the "Change %" column are decimals.
    """
    base_url = (
        f"https://financialmodelingprep.com/stable/economic-calendar?apikey={api_key}"
    )

    for name, value in (("start_date", start_date), ("end_date", end_date)):
        if value is not None and not validation_model.is_valid_date(value):
            raise ValueError(
                f"The {name} must be a date written as YYYY-MM-DD, such as '2026-09-01', not '{value}'."
            )
    if start_date and end_date and start_date > end_date:
        raise ValueError(
            f"The start_date {start_date} must be on or before the end_date {end_date}."
        )

    impact_levels = {level.lower() for level in helpers.split_values(impact)}
    if unknown_levels := impact_levels - {"low", "medium", "high", "all"}:
        raise ValueError(
            f"The impact must be 'Low', 'Medium', 'High' or 'All', not {', '.join(sorted(unknown_levels))}."
        )

    if start_date is None and end_date is None:
        urls = [base_url]
    else:
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = start_date or (
            datetime.strptime(end_date, "%Y-%m-%d")
            - timedelta(days=ECONOMIC_CALENDAR_WINDOW_DAYS - 1)
        ).strftime("%Y-%m-%d")
        urls = [
            f"{base_url}&from={window_start}&to={window_end}"
            for window_start, window_end in _windows(start_date, end_date)
        ]

    frames = helpers.run_in_parallel(
        lambda url: get_cached_financial_data(
            url=url, user_subscription=user_subscription
        ),
        [(url,) for url in urls],
    )
    frames = [frame for frame in frames if "event" in frame.columns]

    if not frames:
        return pd.DataFrame()

    economic_calendar = pd.concat(frames, ignore_index=True).drop_duplicates()

    country_codes = economic_calendar["country"].fillna("").str.upper()

    if requested_countries := helpers.split_values(countries):
        wanted = {country.upper() for country in requested_countries}
        names = country_codes.map(ECONOMIC_CALENDAR_COUNTRIES).fillna("").str.upper()
        economic_calendar = economic_calendar[
            country_codes.isin(wanted) | names.isin(wanted)
        ]
        country_codes = country_codes[economic_calendar.index]

    if requested_currencies := helpers.split_values(currencies):
        wanted = {currency.upper() for currency in requested_currencies}
        economic_calendar = economic_calendar[
            economic_calendar["currency"].fillna("").str.upper().isin(wanted)
        ]
        country_codes = country_codes[economic_calendar.index]

    # "All" asks for every release, which is how an MCP caller widens its High default.
    requested_impact = helpers.split_values(impact)
    if requested_impact and "all" not in {level.lower() for level in requested_impact}:
        wanted = {level.lower() for level in requested_impact}
        economic_calendar = economic_calendar[
            economic_calendar["impact"].fillna("").str.lower().isin(wanted)
        ]
        country_codes = country_codes[economic_calendar.index]

    if requested_events := helpers.split_values(events):
        # Matched as plain text, so a name like "S&P Global PMI" needs no escaping, and
        # without spaces or hyphens, as the source writes "Non Farm Payrolls" for the
        # headline and "Nonfarm Payrolls Private" for its private part.
        matches = pd.Series(False, index=economic_calendar.index)
        names = (
            economic_calendar["event"]
            .fillna("")
            .str.lower()
            .str.replace(r"[\s-]", "", regex=True)
        )
        for event in requested_events:
            matches |= names.str.contains(
                re.sub(r"[\s-]", "", event.lower()), regex=False
            )
        economic_calendar = economic_calendar[matches]
        country_codes = country_codes[economic_calendar.index]

    if economic_calendar.empty:
        logger.warning(
            "No economic releases match the filters (countries=%s, currencies=%s, impact=%s, "
            "events=%s) in the requested period.",
            countries,
            currencies,
            impact,
            events,
        )

    economic_calendar = economic_calendar.rename(
        columns={
            "date": "Date",
            "country": "Country Code",
            "event": "Event",
            "currency": "Currency",
            "previous": "Previous",
            "estimate": "Estimate",
            "actual": "Actual",
            "change": "Change",
            "changePercentage": "Change %",
            "impact": "Impact",
            "unit": "Unit",
        }
    )

    # Percentages are decimals throughout the toolkit, so a release quoted in percent (an
    # inflation or unemployment rate, a policy rate) is divided by 100, as is the relative
    # change. Releases in other units (thousands of jobs, index points) keep their values.
    # FMP also labels many survey levels "%", such as a PMI of 52.4 or a confidence index,
    # so those are recognised by name and left as they are.
    survey_level = economic_calendar["Event"].fillna("").str.contains(
        SURVEY_LEVEL_PATTERN
    ) & ~economic_calendar["Event"].fillna("").str.contains(RATE_OF_CHANGE_PATTERN)
    in_percent = (
        economic_calendar.get("Unit", pd.Series("", index=economic_calendar.index))
        .fillna("")
        .str.strip()
        == "%"
    ) & ~survey_level
    for column in ["Previous", "Estimate", "Actual", "Change"]:
        if column in economic_calendar.columns:
            economic_calendar.loc[in_percent, column] = (
                economic_calendar.loc[in_percent, column] / 100
            )
    if "Change %" in economic_calendar.columns:
        economic_calendar["Change %"] = economic_calendar["Change %"] / 100

    # Named as elsewhere in the Economics module, with the code kept next to it.
    economic_calendar.insert(
        1,
        "Country",
        country_codes.map(ECONOMIC_CALENDAR_COUNTRIES).fillna(country_codes),
    )
    economic_calendar["Date"] = pd.to_datetime(economic_calendar["Date"])

    # Releases at the same time are ordered by country and event, as the API returns them
    # in no fixed order.
    tie_breakers = [
        column for column in ("Country", "Event") if column in economic_calendar.columns
    ]

    return economic_calendar.sort_values(
        ["Date", *tie_breakers], kind="stable"
    ).set_index("Date")


# The period a release covers is the last bracket of its event name: a month "(Oct)", a
# quarter "(Q3)", a week ending on a day "(Oct/10)" or a year "(2026)".
REFERENCE_PERIOD_PATTERN = re.compile(
    r"^(?P<event>.*?)\s*\((?:(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"(?:/(?P<day>\d{1,2}))?|Q(?P<quarter>[1-4])|(?P<year>(?:19|20)\d{2}))\)\s*$"
)
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def get_reference_period(event: str, release: pd.Timestamp) -> tuple[str, pd.Period]:
    """
    Reads the event and the period a release covers from the event name of the economic
    calendar, e.g. "Producer Price Index YoY (Oct)" released on 2026-11-09 is the
    Producer Price Index YoY of October 2026.

    The name only gives the month or quarter, so the year is the year of the release, or
    the year before when the period starts after the release month: a December figure
    released in January belongs to the previous year, as does a fourth quarter released
    in February. A name without a period (an interest rate decision) covers the day of
    its release.

    Args:
        event (str): The event name, e.g. "GDP Growth Rate QoQ (Q3)".
        release (pd.Timestamp): The release date.

    Returns:
        tuple[str, pd.Period]: The event without its period and the period it covers:
        monthly, quarterly, yearly or daily (the week of a weekly release ends on its day).
    """
    match = REFERENCE_PERIOD_PATTERN.match(event)

    if match is None:
        return event.strip(), pd.Period(release, freq="D")

    name = match["event"]

    if match["year"]:
        return name, pd.Period(int(match["year"]), freq="Y")

    if match["quarter"]:
        first_month = 3 * int(match["quarter"]) - 2
        year = release.year - (first_month > release.month)
        return name, pd.Period(year=year, quarter=int(match["quarter"]), freq="Q")

    month = MONTHS.index(match["month"]) + 1

    if match["day"]:
        day = int(match["day"])
        year = release.year - ((month, day) > (release.month, release.day))
        return name, pd.Period(year=year, month=month, day=day, freq="D")

    return name, pd.Period(
        year=release.year - (month > release.month), month=month, freq="M"
    )


IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}


def _merge_joint_releases(releases: pd.DataFrame) -> pd.DataFrame:
    """
    Merges a release the calendar lists for two consecutive periods on the same day into
    one, under the later period: China publishes its January and February figures
    together and the calendar shows them as a January and a February row, sometimes with
    the same value twice and sometimes with the yuan figure under a lower impact. Two
    periods published the same day with different values and the same impact (two months
    caught up after a delay) are kept apart.

    Args:
        releases (pd.DataFrame): The releases with Country, Event, Period, Day, Impact and
            Actual columns, in order of release.

    Returns:
        pd.DataFrame: The releases with each joint release kept once.
    """
    keep = pd.Series(True, index=releases.index)
    periods = releases["Period"].copy()

    for _, same_day in releases.groupby(["Country", "Event", "Day"], sort=False):
        distinct = sorted(set(same_day["Period"]))
        if len(distinct) != 2 or distinct[1] != distinct[0] + 1:  # noqa: PLR2004
            continue

        actual = same_day["Actual"].dropna().to_numpy()
        impact = same_day["Impact"].to_numpy()
        if actual.size and (actual != actual[0]).any() and (impact == impact[0]).all():
            continue

        chosen = same_day["Impact"][::-1].idxmax()
        keep[same_day.index] = False
        keep[chosen] = True
        periods[chosen] = distinct[1]

    return releases.assign(Period=periods)[keep]


def get_economic_calendar_series(economic_calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Arranges the releases of the economic calendar as time series: the Actual value of
    every event by the period it covers, with a column per country and event, and the
    consensus Estimate next to it ("<event> Estimate").

    The period is read from the event name (see get_reference_period). Several releases
    of the same period, such as a flash estimate followed by the final figure, keep the
    latest value that was published; a release still to come has its Estimate and no
    Actual yet.

    The calendar repeats some releases, which would otherwise end up as separate values:

    - A figure published for two consecutive periods on the same day, with the same value
      or under a lower impact, is one release of both periods together, such as China's
      January and February trade figures. It is kept once, under the later period
      (February), from its highest-impact row; the earlier period stays empty.
    - A release without a period in its name, of an event that otherwise has one, is left
      out, as the period it covers is unknown.

    Args:
        economic_calendar (pd.DataFrame): The releases as get_economic_calendar returns
            them, indexed by release date with Country, Event, Estimate and Actual columns.

    Returns:
        pd.DataFrame: The values by period, with (Country, Event) columns. The index is a
        PeriodIndex at the frequency of the events (monthly for most), or daily, at the
        last day of each period, when events of different frequencies are combined.
    """
    if economic_calendar.empty:
        return pd.DataFrame()

    references = [
        get_reference_period(str(event), release)
        for event, release in zip(
            economic_calendar["Event"], economic_calendar.index, strict=True
        )
    ]

    releases = pd.DataFrame(
        {
            "Country": economic_calendar["Country"].to_numpy(),
            "Event": [event for event, _ in references],
            "Period": [period for _, period in references],
            "Named": [
                REFERENCE_PERIOD_PATTERN.match(str(event)) is not None
                for event in economic_calendar["Event"]
            ],
            "Day": economic_calendar.index.normalize(),
            "Impact": (
                economic_calendar["Impact"]
                .fillna("")
                .str.lower()
                .map(IMPACT_RANK)
                .fillna(0)
                .to_numpy()
                if "Impact" in economic_calendar.columns
                else 0
            ),
            "Actual": pd.to_numeric(economic_calendar["Actual"]).to_numpy(),
            "Estimate": pd.to_numeric(economic_calendar["Estimate"]).to_numpy(),
        }
    )

    # A release without a period, of an event that otherwise has one, covers an
    # unknown period.
    has_named = releases.groupby(["Country", "Event"])["Named"].transform("any")
    releases = releases[releases["Named"] | ~has_named]

    releases = _merge_joint_releases(releases)

    periods = list(releases["Period"])
    # Monthly, quarterly and yearly periods cannot share an index, so a mix of them is
    # placed on the last day of each period.
    if len({period.freqstr for period in periods}) > 1:
        periods = [period.asfreq("D", how="end") for period in periods]
    releases = releases.assign(Period=pd.PeriodIndex(periods))

    # The calendar is in order of release, so the last value of a period is its latest
    # (the last one published: a release still to come has no Actual yet).
    latest = releases.groupby(["Country", "Event", "Period"], sort=False).last()

    actual = latest["Actual"].unstack(["Country", "Event"])
    estimate = latest["Estimate"].unstack(["Country", "Event"])
    estimate.columns = pd.MultiIndex.from_tuples(
        [(country, f"{event} Estimate") for country, event in estimate.columns]
    )

    series = pd.concat([actual, estimate.dropna(axis=1, how="all")], axis=1)
    series = series[sorted(series.columns)].sort_index()
    series.index.name = None
    series.columns.names = ["Country", "Event"]

    return series


# The end-of-day endpoint returns at most 5,000 rows per request, so an index history is
# requested in windows of fifteen years (around 3,800 trading days).
INDEX_HISTORY_WINDOW_YEARS = 15


def get_index_history(
    symbol: str,
    api_key: str,
    start_date: str,
    end_date: str,
    user_subscription: str = "Free",
) -> pd.Series:
    """
    Retrieves the daily closing values of an index, such as "^MOVE" (the ICE BofA MOVE
    index), between two dates.

    Args:
        symbol (str): The index symbol, e.g. "^MOVE".
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        pd.Series: The closing values as published, indexed by day.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []

    while start <= end:
        window_end = min(
            start + pd.DateOffset(years=INDEX_HISTORY_WINDOW_YEARS, days=-1), end
        )
        windows.append((start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        start = window_end + pd.DateOffset(days=1)

    urls = [
        "https://financialmodelingprep.com/stable/historical-price-eod/light"
        f"?symbol={symbol}&from={window_start}&to={window_end}&apikey={api_key}"
        for window_start, window_end in windows
    ]
    frames = helpers.run_in_parallel(
        lambda url: get_cached_financial_data(
            url=url, user_subscription=user_subscription
        ),
        [(url,) for url in urls],
    )
    frames = [
        frame
        for frame in frames
        if isinstance(frame, pd.DataFrame) and {"date", "price"} <= set(frame.columns)
    ]

    if not frames:
        return pd.Series(dtype=float)

    data = pd.concat(frames).drop_duplicates(subset="date")
    values = pd.to_numeric(data["price"], errors="coerce")
    values.index = pd.PeriodIndex(pd.to_datetime(data["date"]), freq="D")

    return values.dropna().sort_index().rename(symbol)
