"""FRED Model"""

import numpy as np
import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import frame_model, policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.utilities.requests_model import get_request

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_single_series(
    series_id: str, start_date: str, end_date: str, api_key: str
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

    Returns:
        pd.DataFrame: A DataFrame indexed by a daily PeriodIndex with the series ID as the column name.
            Missing values (reported as "." by FRED) are converted to NaN. Returns an empty DataFrame
            if no observations are found for the given date range.

    Raises:
        RuntimeError: If the HTTP request to the FRED API fails or returns an unexpected response.
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

    data = _request_series(series_id, fetch_start, fetch_end, api_key)

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
    series_id: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Perform the actual FRED observations request for a single series.

    Args:
        series_id (str): The FRED series identifier.
        start_date (str): Start date of the observation range in YYYY-MM-DD format.
        end_date (str): End date of the observation range in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        pd.DataFrame: A DataFrame indexed by a daily PeriodIndex with the series ID as
            the column name, or an empty DataFrame when there are no observations.

    Raises:
        RuntimeError: If the HTTP request to the FRED API fails or returns an unexpected response.
    """
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

    Uses the FRED API (https://fred.stlouisfed.org/docs/api/fred/series_observations.html).
    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    and pass the key via the `fred_api_key` argument or set the `FRED_API_KEY` environment variable.

    Args:
        fred_series_id (str or list): The series ID(s) of the data to retrieve.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key, sent as the `api_key` query parameter.

    Returns:
        fred_data (pandas.DataFrame): The retrieved data as a pandas DataFrame, with the date as the index.
    """
    if isinstance(fred_series_id, str):
        fred_series_id = [fred_series_id]

    frames = helpers.run_in_parallel(
        fetch_single_series,
        [(sid, start_date, end_date, api_key) for sid in fred_series_id],
    )

    fred_data = pd.concat(frames, axis=1)
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

    # OAS is a percentage, so multiply by 100 to get basis points.
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

    # OAS is a percentage, so multiply by 100 to get basis points.
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

    data = data.rename(columns=series)[
        [name for name in series.values() if name in data.rename(columns=series)]
    ]

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
        rate (str): "spot" for the spot (zero-coupon) rates of 17 maturities from 6 months
            to 100 years, or "par" for the par yields of 2, 5, 10 and 30 years.
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
        frequency (str): "daily" (from 1986) or "monthly" (monthly averages, from 1919).
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
        frequency (str): "daily" (from 1986) or "monthly" (monthly averages, from 1953).
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
