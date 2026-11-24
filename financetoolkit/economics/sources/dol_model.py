"""U.S. Department of Labor Model"""

__docformat__ = "google"

import re
import time
from datetime import datetime

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

# The national weekly claims report of the Employment and Training Administration (report
# 539), the source of FRED's ICSA, CCSA and IURSA series. It only answers its submitted
# form.
URL = "https://oui.doleta.gov/unemploy/wkclaims/report.asp"

FIRST_YEAR = 1967

# How often, and how far apart, a request is repeated while the report's database cannot
# be reached.
OUTAGE_ATTEMPTS = 6
OUTAGE_DELAY_SECONDS = 3

# A week of the XML report: the Saturday it ends, then a block per measure with its
# seasonally adjusted figure ("SA") next to the unadjusted one. Weeks not published yet
# carry no number (a non-breaking space).
WEEK_PATTERN = re.compile(r"<week>(.*?)</week>", re.S)
WEEK_ENDED_PATTERN = re.compile(r"<weekEnded>([^<]+)</weekEnded>")

# The measures of the report, by the column they are returned in. Continued claims
# (insured unemployment) and the insured unemployment rate are dated by the week of
# unemployment they count, as FRED dates CCSA and IURSA: the news release reports them a
# week behind the initial claims.
MEASURES = {
    "Initial Claims": "InitialClaims",
    "Continued Claims": "ContinuedClaims",
    "Insured Unemployment Rate": "IUR",
}
MEASURE_PATTERNS = {
    column: re.compile(rf"<{tag}>.*?<SA>([^<]*)</SA>.*?</{tag}>", re.S)
    for column, tag in MEASURES.items()
}


def parse_claims_report(text: str) -> pd.DataFrame:
    """
    Reads the weeks of the national weekly claims report (XML).

    Args:
        text (str): The XML report.

    Returns:
        pd.DataFrame: The Initial Claims and Continued Claims (counts) and the Insured
        Unemployment Rate (in percent, as published), seasonally adjusted, indexed by the
        Saturday each week ends ("Date"); weeks without any figure are left out.

    Raises:
        ValueError: When the report has no weeks with initial claims, which means its
            layout changed.
    """
    rows = {}

    for week in WEEK_PATTERN.findall(text):
        week_ended = WEEK_ENDED_PATTERN.search(week)

        if week_ended is None:
            continue

        figures = {}
        for column, pattern in MEASURE_PATTERNS.items():
            match = pattern.search(week)
            figures[column] = (
                pd.to_numeric(match.group(1).replace(",", ""), errors="coerce")
                if match
                else float("nan")
            )

        rows[pd.Period(pd.to_datetime(week_ended.group(1), format="%m/%d/%Y"), "D")] = (
            figures
        )

    report = pd.DataFrame.from_dict(rows, orient="index", columns=list(MEASURES))
    report = report.astype(float).dropna(how="all").sort_index()

    if report.empty or report["Initial Claims"].isna().all():
        raise ValueError(
            "The Department of Labor's claims report has no initial jobless claims, "
            "which means its layout changed."
        )

    report.index = pd.PeriodIndex(report.index, freq="D", name="Date")

    return report


def get_claims_report() -> pd.DataFrame:
    """
    Retrieve the national weekly claims report of the U.S. Department of Labor from January 1967: the initial claims,
    continued claims and insured unemployment rate.

    The figures are seasonally adjusted and come from one request that is cached as a whole. When the report's
    database cannot be reached, which happens at times, the request is retried and otherwise the most recently cached
    copy is served.

    Returns:
        pd.DataFrame: The Initial Claims, Continued Claims and Insured Unemployment Rate
        (in percent), indexed by the Saturday each week ends ("Date").

    Raises:
        ValueError: When the report has no weeks with claims, which means its layout
            changed.
    """

    def fetch() -> pd.DataFrame:
        current_year = datetime.now().year

        # The report's database is often unreachable for a few seconds, up to about half
        # of the requests, and the report then answers with a short HTML page ("Unable to
        # connect!") instead of the XML. That is an outage, retried here and otherwise
        # answered from the cache, not a changed layout.
        for attempt in range(1, OUTAGE_ATTEMPTS + 1):
            response = get_request(
                URL,
                timeout=120,
                extra_headers=TOOLKIT_HEADERS,
                data={
                    "level": "us",
                    "final_yr": str(current_year + 1),
                    "strtdate": str(FIRST_YEAR),
                    "enddate": str(current_year),
                    "filetype": "xml",
                },
            )

            if "Unable to connect" not in response.text:
                break

            if attempt == OUTAGE_ATTEMPTS:
                raise requests.exceptions.ConnectionError(
                    "The Department of Labor's claims report answered without data: "
                    f"{response.text[:200].strip()}"
                )

            time.sleep(OUTAGE_DELAY_SECONDS)

        return parse_claims_report(response.text)

    return collect_cached_data(
        source=policy_model.DEPARTMENT_OF_LABOR,
        dataset="series",
        entity="claims_report",
        fetch=fetch,
        description="weekly claims report",
    )


def get_initial_jobless_claims() -> pd.DataFrame:
    """
    Retrieves the weekly initial claims for unemployment insurance, seasonally adjusted,
    from the U.S. Department of Labor, from January 1967 (see get_claims_report).

    Returns:
        pd.DataFrame: A single "United States" column with the number of claims, indexed
        by the Saturday each week ends ("Date").
    """
    report = get_claims_report()

    if report.empty:
        return report

    return report[["Initial Claims"]].dropna().set_axis(["United States"], axis=1)


def get_continuing_jobless_claims() -> pd.DataFrame:
    """
    Retrieves the weekly continued claims for unemployment insurance (insured unemployment), seasonally adjusted, from
    the U.S. Department of Labor, from January 1967 (see get_claims_report).

    A week is the week of unemployment the claims count, which the news release reports a week after the initial
    claims of that week.

    Returns:
        pd.DataFrame: A single "United States" column with the number of continued claims,
        indexed by the Saturday each week ends ("Date").
    """
    report = get_claims_report()

    if report.empty:
        return report

    return report[["Continued Claims"]].dropna().set_axis(["United States"], axis=1)


def get_insured_unemployment_rate() -> pd.DataFrame:
    """
    Retrieve the weekly insured unemployment rate of the U.S. Department of Labor from January 1971: the continued
    claims as a share of the employment covered by unemployment insurance.

    The rate is seasonally adjusted (see get_claims_report) and dated like the continued claims, by the week of
    unemployment.

    Returns:
        pd.DataFrame: A single "United States" column with the rate as a decimal (0.012
        for 1.2%), indexed by the Saturday each week ends ("Date").
    """
    report = get_claims_report()

    if report.empty:
        return report

    # The report publishes the rate in percent, to one decimal.
    return (
        report[["Insured Unemployment Rate"]]
        .dropna()
        .set_axis(["United States"], axis=1)
        / 100
    )
