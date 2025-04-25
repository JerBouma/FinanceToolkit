"""FRED Model"""

import io

import numpy as np
import pandas as pd
import requests


def get_fred_data(fred_series_id: str | list):
    """
    Retrieves data from the Federal Reserve Economic Data (FRED) API for the specified series ID(s).

    Args:
        fred_series_id (str or list): The series ID(s) of the data to retrieve. Can be a single ID or a list of IDs.

    Returns:
        fred_data (pandas.DataFrame): The retrieved data as a pandas DataFrame, with the date as the index.
    """
    if isinstance(fred_series_id, list):
        fred_series_id = ",".join(fred_series_id)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    }

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_series_id}"

    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        fred_data = pd.read_csv(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        # Handle exceptions during the request (e.g., connection error, timeout, invalid URL)
        # Consider logging the error or raising a more specific exception
        raise RuntimeError(f"Error fetching data from FRED: {e}") from e

    # Fall back system in case the column name changes, the first column is assumed
    # to be the date column
    fred_data = (
        fred_data.set_index(fred_data.columns[0])
        if "observation_date" not in fred_data.columns
        else fred_data.set_index("observation_date")
    )

    fred_data.index = pd.PeriodIndex(fred_data.index, freq="D")
    fred_data.index.name = "Date"

    fred_data = fred_data.replace(".", np.nan)
    fred_data = fred_data.astype(float)

    fred_data = fred_data.interpolate(limit_area="inside")

    return fred_data


def get_maturity_option_adjusted_spread():
    """
    Retrieves the maturity option-adjusted spread (OAS) data from FRED and converts it to basis points.

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

    oas_data = get_fred_data(year_indices)

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


def get_maturity_effective_yield():
    """
    Retrieves the maturity effective yield data from FRED.

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

    yield_data = get_fred_data(year_indices)

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


def get_maturity_total_return():
    """
    Retrieves the total return data for different maturity periods.

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

    yield_data = get_fred_data(year_indices)

    yield_data.columns = [
        "1-3 Years",
        "3-5 Years",
        "5-7 Years",
        "7-10 Years",
        "10-15 Years",
        "15+ Years",
    ]

    return yield_data


def get_maturity_yield_to_worst():
    """
    Retrieves the yield to worst data for different maturity periods.

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

    yield_data = get_fred_data(year_indices)

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


def get_rating_option_adjusted_spread():
    """
    Retrieves the option-adjusted spread (OAS) data for different credit ratings.

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

    oas_data = get_fred_data(rating_indices)

    oas_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    # Convert OAS data to basis points. Given that the
    # value is in percentages, the value is multiplied by
    # 100 to convert to basis points.
    oas_data = oas_data * 100

    return oas_data


def get_rating_effective_yield():
    """
    Retrieves the effective yield data for different credit ratings.

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

    yield_data = get_fred_data(rating_indices)

    yield_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    # Yield data is a percentage, so it is divided by 100
    # to get the actual numeric value
    yield_data = yield_data / 100

    return yield_data


def get_rating_total_return():
    """
    Retrieves the total return data for different rating indices.

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

    yield_data = get_fred_data(rating_indices)

    yield_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    return yield_data


def get_rating_yield_to_worst():
    """
    Retrieves the yield to worst data for different credit ratings.

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

    yield_data = get_fred_data(rating_indices)

    yield_data.columns = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

    # Yield data is a percentage, so it is divided by 100
    # to get the actual numeric value
    yield_data = yield_data / 100

    return yield_data
