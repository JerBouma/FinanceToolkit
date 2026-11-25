"""Helpers"""

__docformat__ = "google"

import io

import numpy as np
import pandas as pd
import requests

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.requests_model import get_request

logger = logger_model.get_logger()

BASE_URL = "https://data-api.ecb.europa.eu/service/data/"
EXTENSIONS = "?format=csvdata"


def collect_ecb_data(
    ecb_data_string: str, dataset: str, frequency: str = "D"
) -> pd.DataFrame:
    """
    Collect the data from the ECB API and return it as a DataFrame.

    The ECB endpoint returns the full history of a series in one response and takes
    no date range, so there is nothing to request incrementally. The response is
    cached whole instead, keyed on the series being asked for.

    Args:
        ecb_data_string (str): The string that is appended to the base URL to
            get the data from the ECB API.
        dataset (str): The ECB dataset (dataflow) to query, e.g. 'FM' for the
            Financial Markets dataset.
        frequency (str): The frequency of the returned observations, used to build
            the PeriodIndex, e.g. 'D' for daily or 'M' for monthly. Defaults to 'D'.

    Returns:
       pd.DataFrame: A DataFrame containing the data from the ECB API, empty (with an error
       logged) when the ECB cannot be reached.
    """
    cache = get_active_cache()
    cache_entity = f"{dataset}/{ecb_data_string}"
    cache_parameters = {"frequency": frequency}

    if cache is not None:
        cached_data = cache.get(
            source=policy_model.EUROPEAN_CENTRAL_BANK,
            dataset="series",
            entity=cache_entity,
            parameters=cache_parameters,
        )

        if cached_data is not None:
            return cached_data

    url = f"{BASE_URL}{dataset}/{ecb_data_string}{EXTENSIONS}"
    try:
        response = get_request(url)
    except requests.exceptions.RequestException as error:
        logger.error(
            "Could not reach the European Central Bank for the %s series %s (%s). No data is "
            "returned for it.",
            dataset,
            ecb_data_string,
            error,
        )
        return pd.Series(dtype=float)
    ecb_data = pd.read_csv(io.StringIO(response.text))

    ecb_data = ecb_data.set_index("TIME_PERIOD")

    ecb_data.index = pd.PeriodIndex(data=ecb_data.index, freq=frequency)

    ecb_data.index.name = None

    ecb_data = ecb_data["OBS_VALUE"]

    if cache is not None and not ecb_data.empty:
        cache.set(
            source=policy_model.EUROPEAN_CENTRAL_BANK,
            dataset="series",
            entity=cache_entity,
            data=ecb_data,
            parameters=cache_parameters,
        )

    return ecb_data


def treasury_cusip(identifier: str) -> str | None:
    """
    The CUSIP of a Treasury security, from its CUSIP or ISIN.

    Args:
        identifier (str): A CUSIP or ISIN, in capitals.

    Returns:
        str | None: The CUSIP when the security is a Treasury (CUSIPs starting with 912),
        otherwise None.
    """
    if len(identifier) == 12 and identifier.startswith("US912"):  # noqa: PLR2004
        return identifier[2:11]
    if len(identifier) == 9 and identifier.startswith("912"):  # noqa: PLR2004
        return identifier
    return None


def maturity_label(years: float) -> str:
    """
    The label of a maturity in years, such as '3M' or '10Y'.

    Args:
        years (float): The maturity in years.

    Returns:
        str: The label, in months below a year and in years otherwise.
    """
    if years < 1 or not float(years).is_integer():
        months = round(years * 12, 1)
        return f"{int(months) if float(months).is_integer() else months}M"

    return f"{int(years)}Y"


MATURITY_UNITS: dict[str, tuple[int, int]] = {
    "D": (1, 365),
    "W": (7, 365),
    "M": (1, 12),
    "Y": (1, 1),
}


def maturity_in_years(maturity) -> float:
    """
    Converts a maturity, a number of years or a label such as "1W", "6M" or "10Y", into years.

    Days and weeks count in calendar days (a year of 365 days), so "1D" is 1/365 and "1W"
    is 7/365 of a year, while months are twelfths of a year.

    Args:
        maturity (float | int | str): The maturity in years, or a label ending in "D"
            (days), "W" (weeks), "M" (months) or "Y" (years) as the yield curve methods
            and the MCP server pass them.

    Returns:
        float: The maturity in years.

    Raises:
        ValueError: If the maturity is neither a number nor such a label.
    """
    if isinstance(maturity, int | float | np.number):
        return float(maturity)

    label = str(maturity).strip().upper()

    try:
        return float(label)
    except ValueError:
        pass

    if label[-1:] in MATURITY_UNITS:
        length, per_year = MATURITY_UNITS[label[-1]]

        try:
            return float(label[:-1]) * length / per_year
        except ValueError:
            pass

    raise ValueError(
        f"The maturity {maturity!r} is not a number of years or a label such as '1W', '6M' or '10Y'."
    )


def maturity_key(maturity) -> float | int:
    """
    Converts a maturity into years, kept as an integer when it is a whole number of years.

    Whole years stay integers so that curve maturities and the key rate maturities read from
    them label the output as 5 rather than 5.0.

    Args:
        maturity (float | int | str): The maturity in years, or a label such as "6M" or "10Y".

    Returns:
        float | int: The maturity in years.
    """
    years = maturity_in_years(maturity)

    return int(years) if years.is_integer() else years


def parse_maturities(
    maturities: float | str | list | range | None, default: list | range
) -> list[float | int]:
    """
    Maturities in years, from years or labels such as '6M' (a list or a comma-separated string).

    Args:
        maturities (float | str | list | range | None): The maturities, or None for the default.
        default (list | range): The maturities to use when none are given.

    Returns:
        list[float | int]: The maturities in years.
    """
    if maturities is None:
        maturities = default
    elif isinstance(maturities, str):
        maturities = helpers.split_values(maturities)
    elif isinstance(maturities, int | float):
        maturities = [maturities]

    return [maturity_key(maturity) for maturity in maturities]


def discount_factors(zero_curve: pd.Series, years: float | np.ndarray) -> np.ndarray:
    """
    The discount factors of a zero-coupon curve at any times, as the Toolkit's curves define them.

    The curve holds annually compounded zero-coupon rates by maturity in years, as
    `get_zero_coupon_curve` and `get_eiopa_risk_free_rate` return them, so a payment in t
    years is discounted with P(t) = (1 + r(t))^(-t). The zero rate r(t) between two
    maturities of the curve is interpolated linearly, and held at the first and last rate
    outside them, as `get_forward_rate` and `get_par_yield` read the curve. P(0) is 1.

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        years (float | np.ndarray): The times of the payments, in years from today.

    Returns:
        np.ndarray: The discount factor of each time.
    """
    zero_curve = zero_curve.dropna().sort_index()
    years = np.asarray(years, dtype=float)
    zero_rates = np.interp(
        years,
        zero_curve.index.to_numpy(dtype=float),
        zero_curve.to_numpy(dtype=float),
    )

    return (1 + zero_rates) ** -years
