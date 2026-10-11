"""Sveriges Riksbank Model"""

__docformat__ = "google"

import time

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_ranged_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://api.riksbank.se/swea/v1/Observations/"

# The groups with Swedish treasury bills (6) and government bonds (7), and the maturity of
# each series in them. The API allows few requests without a subscription key (around
# five a minute), so a range of up to a year is retrieved per group, in two requests, and
# only a longer range, which the group endpoint refuses, per series.
GROUPS = [6, 7]
MAXIMUM_GROUP_DAYS = 365
YIELD_CURVE_SERIES = {
    "SETB1MBENCHC": "1M",
    "SETB3MBENCH": "3M",
    "SETB6MBENCH": "6M",
    "SETB12MBENCH": "1Y",
    "SEGVB2YC": "2Y",
    "SEGVB5YC": "5Y",
    "SEGVB7YC": "7Y",
    "SEGVB10YC": "10Y",
}

# The API negotiates the format and answers the default browser Accept header with XML.
JSON_HEADERS = {"Accept": "application/json"}

# The status the API answers with when the rate limit is reached, and the longest wait it
# may ask for that is accepted before giving up. The limit resets every minute, so a range
# longer than a year, eight requests, takes about a minute the first time.
TOO_MANY_REQUESTS = 429
MAXIMUM_RETRY_AFTER_SECONDS = 65
MAXIMUM_RATE_LIMIT_WAITS = 4


def _get_respecting_rate_limit(url: str) -> list:
    """
    Requests a Riksbank URL, waiting for the rate limit to clear whenever the API asks for
    that with a 429 response and a short Retry-After, up to a few times.

    Args:
        url (str): The URL to request.

    Returns:
        list: The observations, empty when the series has none in the range.
    """
    for attempt in range(MAXIMUM_RATE_LIMIT_WAITS + 1):
        try:
            response = get_request(url, timeout=60, extra_headers=JSON_HEADERS)
            break
        except requests.exceptions.HTTPError as error:
            if (
                error.response is None
                or error.response.status_code != TOO_MANY_REQUESTS
                or attempt == MAXIMUM_RATE_LIMIT_WAITS
            ):
                raise

            retry_after = pd.to_numeric(
                error.response.headers.get("Retry-After"), errors="coerce"
            )

            if pd.isna(retry_after) or retry_after > MAXIMUM_RETRY_AFTER_SECONDS:
                raise

            time.sleep(float(retry_after))

    # A series without observations in the range, such as a discontinued maturity, is
    # answered with an empty body.
    return response.json() if response.content.strip() else []


def get_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily Swedish government yield curve from the Riksbank, treasury bills
    from 1 month to 1 year and government bonds from 2 to 10 years. Only the days that are
    not cached yet are requested. Without a subscription key the API accepts around five
    requests a minute, so the first retrieval of more than a year waits for the limit to
    reset once, after which a daily refresh takes two requests.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The yields as decimals, indexed by business day with a column per
        maturity ("1M" to "10Y").

    Raises:
        ValueError: When the response is not a list of observations.
    """
    description = "Swedish government yield curve"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        days = (pd.Timestamp(fetch_end) - pd.Timestamp(fetch_start)).days

        if days <= MAXIMUM_GROUP_DAYS:
            requests_to_make = [
                (f"{BASE_URL}ByGroup/{group}/{fetch_start}/{fetch_end}", None)
                for group in GROUPS
            ]
        else:
            requests_to_make = [
                (f"{BASE_URL}{series}/{fetch_start}/{fetch_end}", series)
                for series in YIELD_CURVE_SERIES
            ]

        observations = []

        for url, series in requests_to_make:
            response = _get_respecting_rate_limit(url)

            if not isinstance(response, list) or not all(
                isinstance(row, dict) and {"date", "value"}.issubset(row)
                for row in response
            ):
                raise ValueError(
                    f"The Riksbank response for the {description} is not a list of "
                    "observations, which means the API changed."
                )

            # The group endpoint names the series of every row; the series endpoint does not.
            observations.extend(
                row if series is None else {**row, "seriesId": series}
                for row in response
            )

        if not observations:
            return pd.DataFrame()

        data = pd.DataFrame(observations)
        data = data[data["seriesId"].isin(YIELD_CURVE_SERIES)].drop_duplicates(
            subset=["date", "seriesId"], keep="last"
        )
        yield_curve = data.pivot(
            index="date", columns="seriesId", values="value"
        ).rename(columns=YIELD_CURVE_SERIES)
        yield_curve = yield_curve[
            [label for label in YIELD_CURVE_SERIES.values() if label in yield_curve]
        ]
        yield_curve.index = pd.PeriodIndex(yield_curve.index, freq="D")
        yield_curve.index.name = None
        yield_curve.columns.name = None

        # The Riksbank publishes the yields in percent.
        return yield_curve.apply(pd.to_numeric, errors="coerce").sort_index() / 100

    return collect_ranged_data(
        source=policy_model.RIKSBANK,
        dataset="series",
        entity="yield_curve",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
