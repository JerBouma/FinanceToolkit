"""FRED Model"""

import numpy as np
import pandas as pd
import requests

from financetoolkit import helpers
from financetoolkit.cache import frame_model, policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.cache.request_model import redact_credentials
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.requests_model import get_request

logger = logger_model.get_logger()

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_single_series(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: str,
    failures: dict[str, str] | None = None,
    unreachable: set[str] | None = None,
) -> pd.DataFrame:
    """
    Fetches observations for a single FRED series and returns them as a DataFrame.

    When a cache is active, only the part of the requested range that is not already
    stored is requested from FRED. The API key is deliberately excluded from the cache
    key: the observations are identical regardless of who asks for them, and a
    credential has no place in a cache key.

    Args:
        series_id (str): The FRED series identifier (e.g. "DGS10" for the 10-Year Treasury yield).
        start_date (str): Start date of the observation range in YYYY-MM-DD format.
        end_date (str): End date of the observation range in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.
        failures (dict[str, str] | None, optional): Where a failed request is recorded, as the
            reason per series ID, for a caller that retrieves several series and reports the
            failures in one line. Defaults to None, which logs the failure here.
        unreachable (set[str] | None, optional): Where a series is recorded whose request found
            FRED itself down (no answer, a server error or a rate limit) rather than the series
            missing, so a caller can stop requesting the others. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame indexed by a daily PeriodIndex with the series ID as the column name.
            Missing values (reported as "." by FRED) are converted to NaN. Returns an empty DataFrame
            if no observations are found for the given date range, FRED cannot be reached or FRED
            answers with something other than observations (which is logged or recorded in failures).
    """
    cache = get_active_cache()
    cached_data = None
    fetch_start, fetch_end = start_date, end_date

    if cache is not None:
        plan = cache.plan(
            source=policy_model.FRED,
            dataset="series",
            entities=[series_id],
            start=start_date,
            end=end_date,
        )
        cached_data = plan.cached_frame(series_id)
        fetch_span = plan.get_fetch_span(series_id)

        if fetch_span is None:
            return cached_data if cached_data is not None else pd.DataFrame()

        fetch_start = fetch_span[0].strftime("%Y-%m-%d")
        fetch_end = fetch_span[1].strftime("%Y-%m-%d")

    data = _request_series(
        series_id, fetch_start, fetch_end, api_key, failures, unreachable
    )

    if cache is not None and not data.empty:
        cache.store(
            source=policy_model.FRED,
            dataset="series",
            entity=series_id,
            data=data,
            start=fetch_start,
            end=fetch_end,
        )

    if cached_data is not None and not cached_data.empty:
        data = frame_model.merge_frames(cached_data, data)
        data = frame_model.slice_frame(data, start_date, end_date)

    return data


def _request_series(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: str,
    failures: dict[str, str] | None = None,
    unreachable: set[str] | None = None,
) -> pd.DataFrame:
    """
    Perform the actual FRED observations request for a single series.

    Args:
        series_id (str): The FRED series identifier.
        start_date (str): Start date of the observation range in YYYY-MM-DD format.
        end_date (str): End date of the observation range in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.
        failures (dict[str, str] | None, optional): Where a failed request is recorded, as the
            reason per series ID. Defaults to None, which logs the failure here.
        unreachable (set[str] | None, optional): Where the series ID is recorded when FRED
            itself is down: no answer, a server error (5xx) or a rate limit (429). A rejected
            request (another 4xx, such as a series that does not exist) concerns that series
            only and is not recorded here. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame indexed by a daily PeriodIndex with the series ID as
            the column name, or an empty DataFrame when there are no observations, FRED
            cannot be reached, rejects the request or answers with something other than
            observations, such as an HTML error page (which is logged or recorded in failures).
    """

    def report(reason: str) -> pd.DataFrame:
        # FRED's own messages end in a full stop, which the sentences around the reason add.
        reason = reason.rstrip(". ")
        if failures is None:
            logger.error(
                "No data is returned for the FRED series %s: %s.", series_id, reason
            )
        else:
            failures[series_id] = reason
        return pd.DataFrame()

    url = (
        f"{FRED_API_BASE_URL}"
        f"?series_id={series_id}"
        f"&observation_start={start_date}"
        f"&observation_end={end_date}"
        f"&api_key={api_key}"
        "&file_type=json"
    )

    try:
        response = get_request(url, timeout=30)
    except requests.exceptions.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        if unreachable is not None and (
            status_code is None
            or status_code >= 500  # noqa: PLR2004
            or status_code == requests.codes.too_many_requests
        ):
            unreachable.add(series_id)

        # FRED explains a rejected request (an invalid key, a series that does not exist) in
        # the error_message of a JSON body, which says more than the status line.
        try:
            message = error.response.json().get("error_message")
        except (AttributeError, ValueError):
            message = None

        if message:
            return report(
                f"FRED answered HTTP {status_code}: {redact_credentials(str(message))}"
            )

        # The request error names the URL, which carries the API key, so it is redacted.
        return report(f"FRED answered with an error ({redact_credentials(str(error))})")
    except requests.exceptions.RequestException as error:
        if unreachable is not None:
            unreachable.add(series_id)
        # The request error names the URL, which carries the API key, so it is redacted.
        return report(f"FRED could not be reached ({redact_credentials(str(error))})")

    try:
        payload = response.json()
    except ValueError:
        # An HTML error page or an empty or cut-off body: FRED was reached but sent no data.
        return report("FRED answered with something other than JSON")

    if not isinstance(payload, dict):
        return report(
            f"FRED answered with a {type(payload).__name__} instead of observations"
        )

    if "error_message" in payload:
        # FRED answers an invalid key or series with its own message.
        return report(
            f"FRED answered {redact_credentials(str(payload['error_message']))}"
        )

    observations = payload.get("observations", [])
    if not observations:
        return pd.DataFrame()

    data = pd.DataFrame(observations)[["date", "value"]]
    data = data.set_index("date")
    data.index = pd.PeriodIndex(data.index, freq="D")
    data.index.name = "Date"
    data.columns = [series_id]
    data = data.replace(".", np.nan).astype(float)

    return data


def get_fred_data(
    fred_series_id: str | list,
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Retrieves data from the Federal Reserve Economic Data (FRED) API for the specified series ID(s).

    Uses the FRED API (https://fred.stlouisfed.org/docs/api/fred/series_observations.html).
    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument or set the `FRED_API_KEY` environment variable.

    The first series is requested on its own: when FRED is down (no answer, a server error or
    a rate limit), the others would fail alike, each after its own retries, so they are not
    requested. A series FRED rejects (such as a discontinued series) concerns that series
    only, so the others are still requested. A series that fails while the others come back
    is left out, and the failures are reported in a single line.

    Args:
        fred_series_id (str or list): The series ID(s) of the data to retrieve.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        fred_data (pandas.DataFrame): The series that came back, in the order given, with the
            series ID as the column name and the date as the index. Empty when none came back.
    """
    series_ids = (
        [fred_series_id] if isinstance(fred_series_id, str) else list(fred_series_id)
    )
    failures: dict[str, str] = {}
    unreachable: set[str] = set()

    frames = [
        fetch_single_series(
            series_ids[0], start_date, end_date, api_key, failures, unreachable
        )
    ]

    if unreachable:
        logger.error(
            "No data is returned for the FRED series %s: %s.%s",
            series_ids[0],
            failures[series_ids[0]],
            (
                f" The other {len(series_ids) - 1} series are not requested, since they "
                "would fail alike."
                if len(series_ids) > 1
                else ""
            ),
        )
        return pd.DataFrame()

    frames += helpers.run_in_parallel(
        lambda series_id: fetch_single_series(
            series_id, start_date, end_date, api_key, failures, unreachable
        ),
        [(series_id,) for series_id in series_ids[1:]],
    )

    if failures:
        # Series that failed for the same reason (an invalid key rejects each of them) are
        # named together, so the reason is not repeated per series.
        series_per_reason: dict[str, list[str]] = {}
        for series_id, reason in failures.items():
            series_per_reason.setdefault(reason, []).append(series_id)
        logger.error(
            "No data is returned for the FRED series %s, so %s left out.",
            "; ".join(
                f"{', '.join(failed)} ({reason})"
                for reason, failed in series_per_reason.items()
            ),
            "it is" if len(failures) == 1 else "they are",
        )

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        if not failures:
            logger.warning(
                "FRED has no observations of the series %s between %s and %s.",
                ", ".join(series_ids),
                start_date,
                end_date,
            )
        return pd.DataFrame()

    fred_data = pd.concat(frames, axis=1)
    fred_data = fred_data.interpolate(limit_area="inside")

    return fred_data


def _name_columns(data: pd.DataFrame, series: dict[str, str]) -> pd.DataFrame:
    """
    Names the columns of FRED series by their series ID rather than by position, so a series
    that did not come back leaves its column out instead of shifting the names of the others.

    Args:
        data (pd.DataFrame): The series, with the series ID as the column name.
        series (dict[str, str]): The column name per FRED series identifier.

    Returns:
        pd.DataFrame: The series that came back, named and in the order of series.
    """
    return data.rename(columns=series)[
        [name for series_id, name in series.items() if series_id in data.columns]
    ]


def get_maturity_option_adjusted_spread(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the maturity option-adjusted spread (OAS) data from FRED as decimal fractions.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the OAS data for different maturity ranges.
            The columns represent the maturity ranges in years, and the values are decimal
            fractions (0.0085 for 85 basis points).
    """
    year_indices = {
        "BAMLC1A0C13Y": "1-3 Years",
        "BAMLC2A0C35Y": "3-5 Years",
        "BAMLC3A0C57Y": "5-7 Years",
        "BAMLC4A0C710Y": "7-10 Years",
        "BAMLC7A0C1015Y": "10-15 Years",
        "BAMLC8A0C15PY": "15+ Years",
    }

    oas_data = _name_columns(
        get_fred_data(list(year_indices), start_date, end_date, api_key), year_indices
    )

    # FRED quotes the OAS in percent, so divide by 100 for the decimal every other rate uses.
    oas_data = oas_data / 100

    return oas_data


def get_maturity_effective_yield(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the maturity effective yield data from FRED.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the maturity effective yield data.
            The columns represent different maturity ranges, and the values are
            the corresponding yield rates.
    """
    year_indices = {
        "BAMLC1A0C13YEY": "1-3 Years",
        "BAMLC2A0C35YEY": "3-5 Years",
        "BAMLC3A0C57YEY": "5-7 Years",
        "BAMLC4A0C710YEY": "7-10 Years",
        "BAMLC7A0C1015YEY": "10-15 Years",
        "BAMLC8A0C15PYEY": "15+ Years",
    }

    yield_data = _name_columns(
        get_fred_data(list(year_indices), start_date, end_date, api_key), year_indices
    )

    # Yield data is a percentage, so divide by 100 for the numeric value.
    yield_data = yield_data / 100

    return yield_data


def get_maturity_total_return(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the total return data for different maturity periods.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the total return data for different maturity periods.
            The columns represent the maturity periods in years.
    """
    year_indices = {
        "BAMLCC1A013YTRIV": "1-3 Years",
        "BAMLCC2A035YTRIV": "3-5 Years",
        "BAMLCC3A057YTRIV": "5-7 Years",
        "BAMLCC4A0710YTRIV": "7-10 Years",
        "BAMLCC7A01015YTRIV": "10-15 Years",
        "BAMLCC8A015PYTRIV": "15+ Years",
    }

    yield_data = _name_columns(
        get_fred_data(list(year_indices), start_date, end_date, api_key), year_indices
    )

    return yield_data


def get_maturity_yield_to_worst(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the yield to worst data for different maturity periods.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the yield to worst data for different maturity periods.
            The columns represent the maturity periods in years, and the values represent the yield to worst
            as a numeric value.
    """
    year_indices = {
        "BAMLC1A0C13YSYTW": "1-3 Years",
        "BAMLC2A0C35YSYTW": "3-5 Years",
        "BAMLC3A0C57YSYTW": "5-7 Years",
        "BAMLC4A0C710YSYTW": "7-10 Years",
        "BAMLC7A0C1015YSYTW": "10-15 Years",
        "BAMLC8A0C15PYSYTW": "15+ Years",
    }

    yield_data = _name_columns(
        get_fred_data(list(year_indices), start_date, end_date, api_key), year_indices
    )

    # Yield data is a percentage, so divide by 100 for the numeric value.
    yield_data = yield_data / 100

    return yield_data


def get_rating_option_adjusted_spread(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the option-adjusted spread (OAS) data for different credit ratings.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the OAS data for different credit ratings.
            The columns represent the credit ratings: "AAA", "AA", "A", "BBB", "BB", "B", "CCC".
            The values are decimal fractions (0.0085 for 85 basis points).
    """
    rating_indices = {
        "BAMLC0A1CAAA": "AAA",
        "BAMLC0A2CAA": "AA",
        "BAMLC0A3CA": "A",
        "BAMLC0A4CBBB": "BBB",
        "BAMLH0A1HYBB": "BB",
        "BAMLH0A2HYB": "B",
        "BAMLH0A3HYC": "CCC",
    }

    oas_data = _name_columns(
        get_fred_data(list(rating_indices), start_date, end_date, api_key),
        rating_indices,
    )

    # FRED quotes the OAS in percent, so divide by 100 for the decimal every other rate uses.
    oas_data = oas_data / 100

    return oas_data


def get_rating_effective_yield(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the effective yield data for different credit ratings.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the effective yield data for different credit ratings.
            The columns represent the credit ratings: "AAA", "AA", "A", "BBB", "BB", "B", "CCC".
            The values in the DataFrame are the effective yields as numeric values.
    """
    rating_indices = {
        "BAMLC0A1CAAAEY": "AAA",
        "BAMLC0A2CAAEY": "AA",
        "BAMLC0A3CAEY": "A",
        "BAMLC0A4CBBBEY": "BBB",
        "BAMLH0A1HYBBEY": "BB",
        "BAMLH0A2HYBEY": "B",
        "BAMLH0A3HYCEY": "CCC",
    }

    yield_data = _name_columns(
        get_fred_data(list(rating_indices), start_date, end_date, api_key),
        rating_indices,
    )

    # Yield data is a percentage, so divide by 100 for the numeric value.
    yield_data = yield_data / 100

    return yield_data


def get_rating_total_return(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the total return data for different rating indices.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the total return data for different rating indices.
            The columns of the DataFrame represent the ratings: "AAA", "AA", "A", "BBB", "BB", "B", "CCC".
    """
    rating_indices = {
        "BAMLCC0A1AAATRIV": "AAA",
        "BAMLCC0A2AATRIV": "AA",
        "BAMLCC0A3ATRIV": "A",
        "BAMLCC0A4BBBTRIV": "BBB",
        "BAMLHYH0A1BBTRIV": "BB",
        "BAMLHYH0A2BTRIV": "B",
        "BAMLHYH0A3CMTRIV": "CCC",
    }

    yield_data = _name_columns(
        get_fred_data(list(rating_indices), start_date, end_date, api_key),
        rating_indices,
    )

    return yield_data


def get_rating_yield_to_worst(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the yield to worst data for different credit ratings.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pandas.DataFrame: A DataFrame containing the yield to worst data for different credit ratings.
            The columns represent the credit ratings: AAA, AA, A, BBB, BB, B, CCC.
            The values in the DataFrame are the yield to worst percentages.
    """
    rating_indices = {
        "BAMLC0A1CAAASYTW": "AAA",
        "BAMLC0A2CAASYTW": "AA",
        "BAMLC0A3CASYTW": "A",
        "BAMLC0A4CBBBSYTW": "BBB",
        "BAMLH0A1HYBBSYTW": "BB",
        "BAMLH0A2HYBSYTW": "B",
        "BAMLH0A3HYCSYTW": "CCC",
    }

    yield_data = _name_columns(
        get_fred_data(list(rating_indices), start_date, end_date, api_key),
        rating_indices,
    )

    # Yield data is a percentage, so divide by 100 for the numeric value.
    yield_data = yield_data / 100

    return yield_data


# The High Quality Market (HQM) corporate bond yield curve of the U.S. Treasury: monthly
# average spot rates for these maturities and par yields for a few of them.
HQM_SPOT_SERIES: dict[str, str] = {
    "HQMCB6MT": "6M",
    "HQMCB1YR": "1Y",
    "HQMCB2YR": "2Y",
    "HQMCB3YR": "3Y",
    "HQMCB4YR": "4Y",
    "HQMCB5YR": "5Y",
    "HQMCB7YR": "7Y",
    "HQMCB10YR": "10Y",
    "HQMCB15YR": "15Y",
    "HQMCB20YR": "20Y",
    "HQMCB25YR": "25Y",
    "HQMCB30YR": "30Y",
    "HQMCB40YR": "40Y",
    "HQMCB50YR": "50Y",
    "HQMCB60YR": "60Y",
    "HQMCB75YR": "75Y",
    "HQMCB100YR": "100Y",
}
HQM_PAR_SERIES: dict[str, str] = {
    "HQMCB2YRP": "2Y",
    "HQMCB5YRP": "5Y",
    "HQMCB10YRP": "10Y",
    "HQMCB30YRP": "30Y",
}

# Monthly averages of the Treasury constant maturity (par) yields at the maturities the HQM
# par yields are published for, on the same monthly-average basis.
TREASURY_MONTHLY_SERIES: dict[str, str] = {
    "GS2": "2Y",
    "GS5": "5Y",
    "GS10": "10Y",
    "GS30": "30Y",
}

# Moody's seasoned corporate bond yields, daily since 1986 and monthly since 1919, and the
# daily spreads over the 10-year Treasury FRED derives from them.
MOODYS_DAILY_SERIES: dict[str, str] = {"DAAA": "Aaa", "DBAA": "Baa"}
MOODYS_MONTHLY_SERIES: dict[str, str] = {"AAA": "Aaa", "BAA": "Baa"}
MOODYS_DAILY_SPREAD_SERIES: dict[str, str] = {"AAA10Y": "Aaa", "BAA10Y": "Baa"}


def _get_named_series(
    series: dict[str, str],
    start_date: str,
    end_date: str,
    api_key: str,
    monthly: bool = False,
) -> pd.DataFrame:
    """
    Retrieves FRED series, names their columns and converts them from percent to decimals.

    Args:
        series (dict[str, str]): The column name per FRED series identifier.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.
        monthly (bool): Whether the series are monthly, in which case the index (FRED dates
            a month on its first day) becomes a monthly one. Defaults to False.

    Returns:
        pd.DataFrame: One column per series, as decimals, in the order given.
    """
    data = get_fred_data(list(series), start_date, end_date, api_key)

    if data.empty:
        return data

    data = _name_columns(data, series)

    if monthly:
        data.index = data.index.asfreq("M")

    data.index.name = None

    # FRED quotes these in percent, so divide by 100 for the decimal.
    return data.dropna(how="all") / 100


def get_hqm_corporate_bond_yield_curve(
    rate: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the High Quality Market (HQM) corporate bond yield curve the U.S. Treasury
    publishes monthly from 1984, through FRED.

    Args:
        rate (str): The rates. Options are: "spot" (the spot, zero-coupon, rates of 17
            maturities from 6 months to 100 years), "par" (the par yields of 2, 5, 10
            and 30 years).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: The monthly average rates as decimals, indexed by month with a column
            per maturity.
    """
    series = HQM_SPOT_SERIES if rate == "spot" else HQM_PAR_SERIES

    return _get_named_series(series, start_date, end_date, api_key, monthly=True)


def get_hqm_corporate_bond_spread(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Computes the spread of the HQM corporate par yields over the Treasury constant maturity
    yields at the same maturities (2, 5, 10 and 30 years), both monthly averages.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pd.DataFrame: The spreads as decimals (0.012 for 1.2 percentage points), indexed by
            month with a column per maturity.
    """
    corporate = _get_named_series(
        HQM_PAR_SERIES, start_date, end_date, api_key, monthly=True
    )
    treasury = _get_named_series(
        TREASURY_MONTHLY_SERIES, start_date, end_date, api_key, monthly=True
    )

    if corporate.empty or treasury.empty:
        return pd.DataFrame()

    return (corporate - treasury).dropna(how="all")


def get_moodys_corporate_bond_yields(
    frequency: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves Moody's seasoned Aaa and Baa corporate bond yields from FRED.

    Args:
        frequency (str): The frequency. Options are: "daily" (from 1986), "monthly"
            (monthly averages, from 1919).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: The "Aaa" and "Baa" yields as decimals.
    """
    if frequency == "monthly":
        return _get_named_series(
            MOODYS_MONTHLY_SERIES, start_date, end_date, api_key, monthly=True
        )

    return _get_named_series(MOODYS_DAILY_SERIES, start_date, end_date, api_key)


def get_moodys_corporate_bond_spreads(
    frequency: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the spreads of Moody's Aaa and Baa corporate bond yields over the 10-year
    Treasury yield, plus the Baa minus Aaa spread, the classic default spread.

    Args:
        frequency (str): The frequency. Options are: "daily" (from 1986), "monthly"
            (monthly averages, from 1953).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: The "Aaa", "Baa" and "Baa - Aaa" spreads as decimals.
    """
    if frequency == "monthly":
        yields = _get_named_series(
            MOODYS_MONTHLY_SERIES, start_date, end_date, api_key, monthly=True
        )
        treasury = _get_named_series(
            {"GS10": "10Y"}, start_date, end_date, api_key, monthly=True
        )

        if yields.empty or treasury.empty:
            return pd.DataFrame()

        spreads = yields.sub(treasury["10Y"], axis=0)
    else:
        spreads = _get_named_series(
            MOODYS_DAILY_SPREAD_SERIES, start_date, end_date, api_key
        )

        if spreads.empty:
            return spreads

    spreads["Baa - Aaa"] = spreads["Baa"] - spreads["Aaa"]

    return spreads.dropna(how="all")


def get_moodys_aaa_baa_corporate_bonds(
    frequency: str, start_date: str, end_date: str, api_key: str, spread: bool = False
) -> pd.DataFrame:
    """
    Retrieves Moody's seasoned Aaa and Baa corporate bond yields from FRED, or their spreads
    over the 10-year Treasury yield.

    Args:
        frequency (str): The frequency. Options are: "daily", "monthly".
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.
        spread (bool, optional): Whether to return the spreads. Defaults to False.

    Returns:
        pd.DataFrame: The "Aaa" and "Baa" yields or spreads as decimals.
    """
    moodys = (
        get_moodys_corporate_bond_spreads(frequency, start_date, end_date, api_key)
        if spread
        else get_moodys_corporate_bond_yields(frequency, start_date, end_date, api_key)
    )

    return moodys[[column for column in ("Aaa", "Baa") if column in moodys]]
