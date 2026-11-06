"""SEC EDGAR Module: figures as first reported and their restatements from XBRL company facts"""

__docformat__ = "google"

import os
import threading
import time

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.requests_model import get_request

logger = logger_model.get_logger()

USER_AGENT_ENVIRONMENT_VARIABLE = "FINANCETOOLKIT_SEC_USER_AGENT"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# The SEC allows at most ten requests a second per user, so every request waits until
# a tenth of a second (plus a margin) has passed since the previous one, across threads.
MINIMUM_REQUEST_INTERVAL = 0.11
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST: list[float] = [0.0]

# Each item takes the first of its US-GAAP concepts that reports the period: companies
# moved between concepts over the years (SalesRevenueNet until ASC 606 in 2018, then
# RevenueFromContractWithCustomerExcludingAssessedTax) and differ in which one they use.
CONCEPTS: dict[str, list[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "Net Income": ["NetIncomeLoss"],
    "EPS": ["EarningsPerShareBasic"],
    "EPS Diluted": ["EarningsPerShareDiluted"],
    "Total Assets": ["Assets"],
    "Total Liabilities": ["Liabilities"],
    "Total Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "Capital Expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}

# Per share figures cannot be derived by subtracting year-to-date figures, because the
# share count differs per period, so a quarter that is only reported inside a year-to-date
# figure (usually the fourth) is left empty for them.
PER_SHARE_ITEMS = {"EPS", "EPS Diluted"}

# The length in days of a fiscal quarter and a fiscal year: 52-53 week years and
# changed fiscal year ends make them vary around 91 and 365.
QUARTER_DAYS = (80, 100)
YEAR_DAYS = (350, 380)
# A year-to-date figure from which a quarter is derived spans at least two quarters.
YEAR_TO_DATE_MINIMUM_DAYS = 170

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}
QUARTERLY_FORMS = ANNUAL_FORMS | {"10-Q", "10-Q/A"}

COLUMNS = [
    "First Reported",
    "First Filed",
    "Latest",
    "Latest Filed",
    "Restatement",
]


def get_user_agent() -> str:
    """
    Return the User-Agent the SEC requests are made with.

    The SEC's fair access policy asks every automated request to declare who makes it with
    a name and a contact email address, such as "Jane Doe jane.doe@example.com", and refuses
    requests that do not. The Toolkit has no address of its own to send, so the user's own
    is read from the FINANCETOOLKIT_SEC_USER_AGENT environment variable.

    Returns:
        str: The User-Agent.

    Raises:
        ValueError: If FINANCETOOLKIT_SEC_USER_AGENT is not set or holds no email address.
    """
    user_agent = os.environ.get(USER_AGENT_ENVIRONMENT_VARIABLE, "").strip()

    if "@" not in user_agent:
        raise ValueError(
            f"The SEC asks every request to name who makes it. Set {USER_AGENT_ENVIRONMENT_VARIABLE} "
            "to your name and email address, such as 'Jane Doe jane.doe@example.com'."
        )

    return user_agent


def get_sec_json(url: str) -> dict:
    """
    Request a JSON document from the SEC, at most ten times a second.

    Args:
        url (str): The URL to request.

    Returns:
        dict: The parsed document.
    """
    user_agent = get_user_agent()

    with _REQUEST_LOCK:
        wait = MINIMUM_REQUEST_INTERVAL - (time.monotonic() - _LAST_REQUEST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST[0] = time.monotonic()

    response = get_request(url, extra_headers={"User-Agent": user_agent})

    return response.json()


def get_cik_numbers() -> dict[str, int]:
    """
    Get the SEC's Central Index Key (CIK) of every ticker it lists.

    The list holds the companies that file with the SEC, so foreign listings such as ASML.AS
    are not in it (their US listing, ASML, is when the company files a 20-F or 10-K).

    Returns:
        dict[str, int]: The CIK per ticker, in upper case with dashes as the SEC writes them
            (BRK-B).
    """
    cache = get_active_cache()

    if cache is not None:
        stored = cache.get(
            source=policy_model.SEC, dataset="company_tickers", entity="all"
        )
        if stored is not None:
            return stored

    listing = get_sec_json(TICKERS_URL)
    cik_numbers = {
        str(entry["ticker"]).upper(): int(entry["cik_str"])
        for entry in listing.values()
    }

    if cache is not None and cik_numbers:
        cache.set(
            source=policy_model.SEC,
            dataset="company_tickers",
            entity="all",
            data=cik_numbers,
        )

    return cik_numbers


def get_company_facts(cik: int) -> dict:
    """
    Get every XBRL fact a company has filed with the SEC.

    Args:
        cik (int): The company's Central Index Key.

    Returns:
        dict: The company facts document, with every value of every concept, each with the
            filing (accession number, form and filing date) that reported it.
    """
    cache = get_active_cache()
    entity = str(cik)

    if cache is not None:
        stored = cache.get(
            source=policy_model.SEC, dataset="company_facts", entity=entity
        )
        if stored is not None:
            return stored

    facts = get_sec_json(COMPANY_FACTS_URL.format(cik=cik))

    if cache is not None and facts.get("facts"):
        cache.set(
            source=policy_model.SEC,
            dataset="company_facts",
            entity=entity,
            data=facts,
        )

    return facts


def _get_concept_facts(facts: dict, concept: str) -> pd.DataFrame:
    """
    Collect the 10-K and 10-Q values of one US-GAAP concept.

    Args:
        facts (dict): The company facts document.
        concept (str): The US-GAAP concept.

    Returns:
        pd.DataFrame: One row per value with its start (empty for a balance at a point in
            time), end, value, form and filing date.
    """
    units = facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {})

    # Monetary concepts are in USD and per share figures in USD/shares; a company that
    # reports in another currency is left out rather than mixed with dollar figures.
    values = units.get("USD", units.get("USD/shares", []))

    if not values:
        return pd.DataFrame(columns=["start", "end", "val", "form", "filed"])

    frame = pd.DataFrame(values)
    frame = frame[frame["form"].isin(QUARTERLY_FORMS)]

    if "start" not in frame:
        frame["start"] = None

    frame = frame[["start", "end", "val", "form", "filed"]].copy()
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])

    return frame


def _first_and_latest(values: pd.DataFrame) -> pd.Series:
    """
    Pick the first and the latest reported value from all filings of one period.

    The same figure comes back as a comparative in later filings, so the earliest filing
    is the one that first made it public. Within one filing day an original form comes
    before its amendment.

    Args:
        values (pd.DataFrame): The values of one period, with their form and filing date.

    Returns:
        pd.Series: First Reported, First Filed, Latest and Latest Filed.
    """
    ordered = values.assign(amendment=values["form"].str.endswith("/A")).sort_values(
        ["filed", "amendment"], kind="stable"
    )
    first, latest = ordered.iloc[0], ordered.iloc[-1]

    return pd.Series(
        {
            "First Reported": float(first["val"]),
            "First Filed": first["filed"],
            "Latest": float(latest["val"]),
            "Latest Filed": latest["filed"],
        }
    )


def _value_on(values: pd.DataFrame, date: pd.Timestamp) -> float:
    """
    Return the value of one period as it stood on a date: the latest filed on or before it.

    Args:
        values (pd.DataFrame): The values of one period, with their filing date.
        date (pd.Timestamp): The date.

    Returns:
        float: The value, or NaN if none was filed by then.
    """
    known = values[values["filed"] <= date].sort_values("filed", kind="stable")

    return float(known["val"].iloc[-1]) if not known.empty else float("nan")


def get_reported_periods(
    concept_facts: pd.DataFrame, quarterly: bool, per_share: bool = False
) -> pd.DataFrame:
    """
    Turn the values of one concept into one row per fiscal period, as first reported and
    as reported today.

    Balances at a point in time (instant facts) are taken at each period end: from annual
    reports for annual periods, from annual and quarterly reports for quarters. Flows over
    a period (duration facts) are taken from durations of about a year, or about a quarter.
    A 10-Q reports its cash flows year to date and a 10-K reports the fourth quarter only
    within the year, so a quarter without a figure of its own is the year-to-date figure
    less the one of the quarter before (Q4 = FY − 9M, Q3 = 9M − 6M). Its first reported
    value is that difference on the day the later figure was first filed, with the earlier
    figure as it stood that day.

    Args:
        concept_facts (pd.DataFrame): The values of one concept, from _get_concept_facts.
        quarterly (bool): Whether to return quarters instead of fiscal years.
        per_share (bool): Whether the concept is a per share figure, which cannot be
            derived from year-to-date figures. Defaults to False.

    Returns:
        pd.DataFrame: First Reported, First Filed, Latest and Latest Filed per period end.
    """
    if concept_facts.empty:
        return pd.DataFrame(columns=COLUMNS[:-1])

    instant = concept_facts[concept_facts["start"].isna()]
    duration = concept_facts[concept_facts["start"].notna()].copy()

    if not instant.empty:
        if not quarterly:
            instant = instant[instant["form"].isin(ANNUAL_FORMS)]

        return instant.groupby("end").apply(_first_and_latest).sort_index()

    duration["days"] = (duration["end"] - duration["start"]).dt.days

    if not quarterly:
        annual = duration[duration["days"].between(*YEAR_DAYS)]
        return annual.groupby("end").apply(_first_and_latest).sort_index()

    periods = {}

    # Year to date durations, keyed by their start (the fiscal year start) and end.
    year_to_date = {
        key: group for key, group in duration.groupby(["start", "end"], sort=True)
    }

    quarter_ends = sorted(
        {
            end
            for (start, end), group in year_to_date.items()
            if QUARTER_DAYS[0] <= (end - start).days <= YEAR_DAYS[1]
        }
    )

    for end in quarter_ends:
        own_quarter = duration[
            (duration["end"] == end) & duration["days"].between(*QUARTER_DAYS)
        ]

        if not own_quarter.empty:
            periods[end] = _first_and_latest(own_quarter)
            continue

        if per_share:
            continue

        # The longest year-to-date figure ending this quarter, less the year-to-date figure
        # of the same fiscal year that ends one quarter earlier.
        candidates = [
            (start, group)
            for (start, period_end), group in year_to_date.items()
            if period_end == end
            and YEAR_TO_DATE_MINIMUM_DAYS <= (end - start).days <= YEAR_DAYS[1]
        ]

        for start, current in sorted(candidates, key=lambda item: item[0]):
            earlier = [
                group
                for (earlier_start, earlier_end), group in year_to_date.items()
                if earlier_start == start
                and QUARTER_DAYS[0] <= (end - earlier_end).days <= QUARTER_DAYS[1]
            ]

            if not earlier:
                continue

            previous = earlier[0]
            first = _first_and_latest(current)
            first_previous = _value_on(previous, first["First Filed"])
            latest_previous = _first_and_latest(previous)["Latest"]

            periods[end] = pd.Series(
                {
                    "First Reported": first["First Reported"] - first_previous,
                    "First Filed": first["First Filed"],
                    "Latest": first["Latest"] - latest_previous,
                    "Latest Filed": max(
                        first["Latest Filed"],
                        _first_and_latest(previous)["Latest Filed"],
                    ),
                }
            )
            break

    if not periods:
        return pd.DataFrame(columns=COLUMNS[:-1])

    return pd.DataFrame(periods).T.sort_index()


def get_first_reported_figures(
    facts: dict, quarterly: bool = False, as_of: str | None = None
) -> pd.DataFrame:
    """
    Calculate the key figures of one company as first reported and as reported today.

    Args:
        facts (dict): The company facts document.
        quarterly (bool): Whether to return quarters instead of fiscal years. Defaults to False.
        as_of (str | None): Only use the filings made on or before this date (YYYY-MM-DD), so
            "Latest" is the figure as it stood that day and later periods are left out.
            Defaults to None, all filings.

    Returns:
        pd.DataFrame: First Reported, First Filed, Latest, Latest Filed and Restatement (the
            latest value relative to the first, less one, as a decimal; 0 when unchanged) per
            item and period. Periods are labelled by the calendar year or quarter in which
            they end, as the Toolkit's financial statements are.
    """
    items = {}

    for item, concepts in CONCEPTS.items():
        combined = pd.DataFrame(columns=COLUMNS[:-1])

        # Concepts in order of preference: a later concept only fills periods the earlier
        # ones do not report.
        for concept in concepts:
            concept_facts = _get_concept_facts(facts, concept)

            if as_of is not None:
                concept_facts = concept_facts[
                    concept_facts["filed"] <= pd.Timestamp(as_of)
                ]

            periods = get_reported_periods(
                concept_facts,
                quarterly=quarterly,
                per_share=item in PER_SHARE_ITEMS,
            )
            periods = periods[~periods.index.isin(combined.index)]

            if not periods.empty:
                combined = periods if combined.empty else pd.concat([combined, periods])

        if not combined.empty:
            items[item] = combined.sort_index()

    if not items:
        return pd.DataFrame(columns=COLUMNS)

    figures = pd.concat(items, names=["Item", "Date"])

    for column in ["First Reported", "Latest"]:
        figures[column] = figures[column].astype(float)
    for column in ["First Filed", "Latest Filed"]:
        figures[column] = pd.to_datetime(figures[column])

    # Growth from a negative base is measured against its absolute value, so a loss that
    # was restated to a larger loss shows as a negative restatement.
    figures["Restatement"] = (figures["Latest"] - figures["First Reported"]) / figures[
        "First Reported"
    ].abs()

    dates = figures.index.get_level_values("Date")
    periods = pd.PeriodIndex(dates, freq="Q" if quarterly else "Y")
    figures.index = pd.MultiIndex.from_arrays(
        [figures.index.get_level_values("Item"), periods], names=["Item", "Period"]
    )

    # Two period ends within one calendar period (a changed fiscal year end) keep the later.
    figures = figures[~figures.index.duplicated(keep="last")]

    return figures[COLUMNS]
