"""U.S. Department of Labor Model"""

__docformat__ = "google"

import re
import time
from datetime import datetime

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, to_us_series
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request

# The national weekly claims report of the Employment and Training Administration (report
# 539), the source of FRED's ICSA series. It only answers its submitted form.
URL = "https://oui.doleta.gov/unemploy/wkclaims/report.asp"

FIRST_YEAR = 1967

# How often, and how far apart, a request is repeated while the report's database cannot
# be reached.
OUTAGE_ATTEMPTS = 6
OUTAGE_DELAY_SECONDS = 3

# A week of the XML report: the Saturday it ends and its initial claims, of which the
# seasonally adjusted count ("SA") is used. Weeks not published yet carry no number.
WEEK_PATTERN = re.compile(
    r"<weekEnded>([^<]+)</weekEnded>\s*<InitialClaims>.*?<SA>([^<]*)</SA>", re.S
)


def get_initial_jobless_claims() -> pd.DataFrame:
    """
    Retrieves the weekly initial claims for unemployment insurance, seasonally adjusted,
    from the U.S. Department of Labor, from January 1967. When the report's database
    cannot be reached, which happens at times, the request is retried and otherwise the
    most recently cached copy is served.

    Returns:
        pd.DataFrame: A single "United States" column with the number of claims, indexed
        by the Saturday each week ends ("Date").

    Raises:
        ValueError: When the report has no weeks with claims, which means its layout
            changed.
    """
    description = "initial jobless claims"

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

        observations = {
            pd.Period(pd.to_datetime(week, format="%m/%d/%Y"), freq="D"): number
            for week, claims in WEEK_PATTERN.findall(response.text)
            if pd.notna(
                number := pd.to_numeric(claims.replace(",", ""), errors="coerce")
            )
        }

        return to_us_series(
            observations, "Department of Labor's claims report", description
        )

    return collect_cached_data(
        source=policy_model.DEPARTMENT_OF_LABOR,
        dataset="series",
        entity="initial_claims",
        fetch=fetch,
        description=description,
    )
