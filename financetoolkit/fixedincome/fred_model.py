"""FRED Model"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from financetoolkit.utilities.requests_model import get_request

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred/v2/series/observations"


def fetch_single_series(
    series_id: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Fetches observations for a single FRED series and returns them as a DataFrame.

    Args:
        series_id (str): The FRED series identifier (e.g. "DGS10" for the 10-Year Treasury yield).
        start_date (str): Start date of the observation range in YYYY-MM-DD format.
        end_date (str): End date of the observation range in YYYY-MM-DD format.
        api_key (str): FRED API key used for Bearer token authentication.

    Returns:
        pd.DataFrame: A DataFrame indexed by a daily PeriodIndex with the series ID as the column name.
            Missing values (reported as "." by FRED) are converted to NaN. Returns an empty DataFrame
            if no observations are found for the given date range.

    Raises:
        RuntimeError: If the HTTP request to the FRED API fails or returns an unexpected response.
    """
    url = (
        f"{FRED_API_BASE_URL}"
        f"?series_id={series_id}"
        f"&observation_start={start_date}"
        f"&observation_end={end_date}"
    )

    try:
        response = get_request(
            url, timeout=30, extra_headers={"Authorization": f"Bearer {api_key}"}
        )
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"Error fetching data from FRED: {e}") from e

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

    Uses the FRED API v2 (https://fred.stlouisfed.org/docs/api/fred/v2/index.html).
    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument or set the `FRED_API_KEY` environment variable.

    Args:
        fred_series_id (str or list): The series ID(s) of the data to retrieve.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key. Sent as a Bearer token per the v2 authentication scheme.

    Returns:
        fred_data (pandas.DataFrame): The retrieved data as a pandas DataFrame, with the date as the index.
    """
    if isinstance(fred_series_id, str):
        fred_series_id = [fred_series_id]

    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=len(fred_series_id)) as executor:
        future_to_sid = {
            executor.submit(
                fetch_single_series, sid, start_date, end_date, api_key
            ): sid
            for sid in fred_series_id
        }
        for future in as_completed(future_to_sid):
            sid = future_to_sid[future]
            frames[sid] = future.result()

    fred_data = pd.concat([frames[sid] for sid in fred_series_id], axis=1)
    fred_data = fred_data.interpolate(limit_area="inside")

    return fred_data


def get_maturity_option_adjusted_spread(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the maturity option-adjusted spread (OAS) data from FRED and converts it to basis points.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Returns:
        pandas.DataFrame: A DataFrame containing the OAS data for different maturity ranges.
            The columns represent the maturity ranges in years, and the values are in basis points.
    """
    year_indices = [
        "BAMLC1A0C13Y",
        "BAMLC2A0C35Y",
        "BAMLC3A0C57Y",
        "BAMLC4A0C710Y",
        "BAMLC7A0C1015Y",
        "BAMLC8A0C15PY",
    ]

    oas_data = get_fred_data(year_indices, start_date, end_date, api_key)

    oas_data.columns = [
        "1-3 Years",
        "3-5 Years",
        "5-7 Years",
        "7-10 Years",
        "10-15 Years",
        "15+ Years",
    ]

    # Convert OAS data to basis points. Given that the
    # value is in percentages, the value is multiplied by
    # 100 to convert to basis points.
    oas_data = oas_data * 100

    return oas_data


def get_maturity_effective_yield(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the maturity effective yield data from FRED.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Returns:
        pandas.DataFrame: A DataFrame containing the maturity effective yield data.
            The columns represent different maturity ranges, and the values are
            the corresponding yield rates.
    """
    year_indices = [
        "BAMLC1A0C13YEY",
        "BAMLC2A0C35YEY",
        "BAMLC3A0C57YEY",
        "BAMLC4A0C710YEY",
        "BAMLC7A0C1015YEY",
        "BAMLC8A0C15PYEY",
    ]

    yield_data = get_fred_data(year_indices, start_date, end_date, api_key)

    yield_data.columns = [
        "1-3 Years",
        "3-5 Years",
        "5-7 Years",
        "7-10 Years",
        "10-15 Years",
        "15+ Years",
    ]

    # Yield data is a percentage, so it is divided by 100
    # to get the actual numeric value
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

    Returns:
        pandas.DataFrame: A DataFrame containing the total return data for different maturity periods.
            The columns represent the maturity periods in years.
    """
    year_indices = [
        "BAMLCC1A013YTRIV",
        "BAMLCC2A035YTRIV",
        "BAMLCC3A057YTRIV",
        "BAMLCC4A0710YTRIV",
        "BAMLCC7A01015YTRIV",
        "BAMLCC8A015PYTRIV",
    ]

    yield_data = get_fred_data(year_indices, start_date, end_date, api_key)

    yield_data.columns = [
        "1-3 Years",
        "3-5 Years",
        "5-7 Years",
        "7-10 Years",
        "10-15 Years",
        "15+ Years",
    ]

    return yield_data


def get_maturity_yield_to_worst(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the yield to worst data for different maturity periods.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Returns:
        pandas.DataFrame: A DataFrame containing the yield to worst data for different maturity periods.
            The columns represent the maturity periods in years, and the values represent the yield to worst
            as a numeric value.
    """
    year_indices = [
        "BAMLC1A0C13YSYTW",
        "BAMLC2A0C35YSYTW",
        "BAMLC3A0C57YSYTW",
        "BAMLC4A0C710YSYTW",
        "BAMLC7A0C1015YSYTW",
        "BAMLC8A0C15PYSYTW",
    ]

    yield_data = get_fred_data(year_indices, start_date, end_date, api_key)

    yield_data.columns = [
        "1-3 Years",
        "3-5 Years",
        "5-7 Years",
        "7-10 Years",
        "10-15 Years",
        "15+ Years",
    ]

    # Yield data is a percentage, so it is divided by 100
    # to get the actual numeric value
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

    Returns:
        pandas.DataFrame: A DataFrame containing the OAS data for different credit ratings.
            The columns represent the credit ratings: "AAA", "AA", "A", "BBB", "BB", "B", "CCC".
            The values are in basis points.
    """
    rating_indices = [
        "BAMLC0A1CAAA",
        "BAMLC0A2CAA",
        "BAMLC0A3CA",
        "BAMLC0A4CBBB",
        "BAMLH0A1HYBB",
        "BAMLH0A2HYB",
        "BAMLH0A3HYC",
    ]

    oas_data = get_fred_data(rating_indices, start_date, end_date, api_key)

    oas_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    # Convert OAS data to basis points. Given that the
    # value is in percentages, the value is multiplied by
    # 100 to convert to basis points.
    oas_data = oas_data * 100

    return oas_data


def get_rating_effective_yield(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the effective yield data for different credit ratings.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Returns:
        pandas.DataFrame: A DataFrame containing the effective yield data for different credit ratings.
            The columns represent the credit ratings: "AAA", "AA", "A", "BBB", "BB", "B", "CCC".
            The values in the DataFrame are the effective yields as numeric values.
    """
    rating_indices = [
        "BAMLC0A1CAAAEY",
        "BAMLC0A2CAAEY",
        "BAMLC0A3CAEY",
        "BAMLC0A4CBBBEY",
        "BAMLH0A1HYBBEY",
        "BAMLH0A2HYBEY",
        "BAMLH0A3HYCEY",
    ]

    yield_data = get_fred_data(rating_indices, start_date, end_date, api_key)

    yield_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    # Yield data is a percentage, so it is divided by 100
    # to get the actual numeric value
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

    Returns:
        pandas.DataFrame: A DataFrame containing the total return data for different rating indices.
            The columns of the DataFrame represent the ratings: "AAA", "AA", "A", "BBB", "BB", "B", "CCC".
    """
    rating_indices = [
        "BAMLCC0A1AAATRIV",
        "BAMLCC0A2AATRIV",
        "BAMLCC0A3ATRIV",
        "BAMLCC0A4BBBTRIV",
        "BAMLHYH0A1BBTRIV",
        "BAMLHYH0A2BTRIV",
        "BAMLHYH0A3CMTRIV",
    ]

    yield_data = get_fred_data(rating_indices, start_date, end_date, api_key)

    yield_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    return yield_data


def get_rating_yield_to_worst(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the yield to worst data for different credit ratings.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument on FixedIncome or set the `FRED_API_KEY`
    environment variable.

    Returns:
        pandas.DataFrame: A DataFrame containing the yield to worst data for different credit ratings.
            The columns represent the credit ratings: AAA, AA, A, BBB, BB, B, CCC.
            The values in the DataFrame are the yield to worst percentages.
    """
    rating_indices = [
        "BAMLC0A1CAAASYTW",
        "BAMLC0A2CAASYTW",
        "BAMLC0A3CASYTW",
        "BAMLC0A4CBBBSYTW",
        "BAMLH0A1HYBBSYTW",
        "BAMLH0A2HYBSYTW",
        "BAMLH0A3HYCSYTW",
    ]

    yield_data = get_fred_data(rating_indices, start_date, end_date, api_key)

    yield_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    # Yield data is a percentage, so it is divided by 100
    # to get the actual numeric value
    yield_data = yield_data / 100

    return yield_data
