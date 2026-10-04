"""Eurostat Model"""

__docformat__ = "google"

import re

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import COUNTRY_CODES, collect_ranged_data
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

# Eurostat moved the HICP to the ECOICOP version 2 classification in February 2026. The
# earlier datasets (prc_hicp_manr, prc_hicp_midx) stop at December 2025.
HICP_DATASET = "prc_hicp_minr"
UNEMPLOYMENT_DATASET = "une_rt_m"
GDP_DATASET = "namq_10_gdp"

# The euro area is published as "EA" (changing composition) in some datasets and only
# under its current membership, e.g. "EA21" since Bulgaria joined in 2026, in others.
EURO_AREA_PATTERN = re.compile(r"^EA(\d+)$")


def _resolve_euro_area(codes: list[str]) -> str | None:
    """
    Picks the euro area code to use: "EA" when the dataset has it, otherwise the one
    with the most members, which is the current composition.

    Args:
        codes (list[str]): The geo codes in the response.

    Returns:
        str | None: The euro area code, or None when the dataset has none.
    """
    if "EA" in codes:
        return "EA"

    numbered = [code for code in codes if EURO_AREA_PATTERN.match(code)]

    return (
        max(numbered, key=lambda code: int(EURO_AREA_PATTERN.match(code).group(1)))  # type: ignore[union-attr]
        if numbered
        else None
    )


def parse_json_stat(response: dict, description: str) -> pd.DataFrame:
    """
    Converts a Eurostat JSON-stat response into a DataFrame with one column per country.

    Every dimension other than geo and time must be pinned to a single value by the
    query. Otherwise several series would land in one country column and whichever came
    last would win, so that is treated as an error rather than resolved silently.

    Args:
        response (dict): The decoded JSON-stat response.
        description (str): What was retrieved, used in the error messages.

    Returns:
        pd.DataFrame: The values indexed by month or quarter, with a column per country.

    Raises:
        ValueError: When the response is not JSON-stat, misses the geo or time dimension,
            or holds more than one series per country.
    """
    if "error" in response:
        raise ValueError(
            f"Eurostat rejected the query for the {description}: {response['error']}"
        )

    if not {"id", "size", "dimension", "value"}.issubset(response):
        raise ValueError(
            f"The Eurostat response for the {description} is not in the JSON-stat format, "
            "which means the API changed and the response cannot be interpreted."
        )

    dimensions = response["id"]
    sizes = response["size"]

    if not {"geo", "time"}.issubset(dimensions):
        raise ValueError(
            f"The Eurostat response for the {description} has no geo or time dimension, "
            "which means the dataset changed and the response cannot be interpreted."
        )

    if unpinned := [
        dimension
        for dimension, size in zip(dimensions, sizes, strict=True)
        if dimension not in ("geo", "time") and size > 1
    ]:
        raise ValueError(
            f"The Eurostat query for the {description} matches more than one series per "
            f"country (the {', '.join(unpinned)} dimension is not pinned), so the values "
            "cannot be assigned to a single column."
        )

    categories = [
        sorted(
            response["dimension"][dimension]["category"]["index"].items(),
            key=lambda item: item[1],
        )
        for dimension in dimensions
    ]
    geo_position = dimensions.index("geo")
    time_position = dimensions.index("time")

    # Each dimension has size 1 apart from geo and time, so the flat index of a value
    # decodes into a geo and a time position.
    records = []
    for flat_index, value in response["value"].items():
        remainder = int(flat_index)
        positions = [0] * len(sizes)

        for axis in reversed(range(len(sizes))):
            positions[axis] = remainder % sizes[axis]
            remainder //= sizes[axis]

        records.append(
            (
                categories[geo_position][positions[geo_position]][0],
                categories[time_position][positions[time_position]][0],
                value,
            )
        )

    if not records:
        return pd.DataFrame()

    data = pd.DataFrame(records, columns=["geo", "time", "value"]).pivot(
        index="time", columns="geo", values="value"
    )

    # "EU" is the European Union in Eurostat data but the euro area in the economic
    # calendar the country names are shared with, so the union comes from EU27_2020.
    euro_area = _resolve_euro_area(list(data.columns))
    countries = {
        code: COUNTRY_CODES[code]
        for code in data.columns
        if code in COUNTRY_CODES and code != "EU" and not code.startswith("EA")
    }
    if euro_area is not None:
        countries[euro_area] = "Euro Area"

    data = data[list(countries)].rename(columns=countries)
    # Monthly periods are written as "2026-09" and quarters as "2026-Q2".
    frequency = "Q" if str(data.index[0])[-2] == "Q" else "M"
    data.index = pd.PeriodIndex(data.index, freq=frequency)
    data.index.name = None
    data.columns.name = None

    return data.sort_index().astype(float)


def collect_eurostat_data(
    dataset: str,
    filters: dict[str, str],
    description: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Retrieves the months or quarters between two dates of a Eurostat dataset for every
    country it covers. Only the periods that are not cached yet are requested.

    Args:
        dataset (str): The Eurostat dataset code, e.g. "prc_hicp_minr".
        filters (dict[str, str]): The value to pin every dimension other than geo and time to.
        description (str): What is retrieved, used in the log and error messages.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The values indexed by month or quarter, with a column per country.
    """
    # Eurostat takes "2026-09" for a monthly and "2026-Q3" for a quarterly dataset.
    quarterly = filters.get("freq") == "Q"

    def to_period(date: str) -> str:
        return (
            str(pd.Period(date, freq="Q")).replace("Q", "-Q") if quarterly else date[:7]
        )

    query = "&".join(
        f"{dimension}={value}" for dimension, value in sorted(filters.items())
    )

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{BASE_URL}{dataset}?{query}&sinceTimePeriod={to_period(fetch_start)}"
            f"&untilTimePeriod={to_period(fetch_end)}",
            timeout=120,
        )

        return parse_json_stat(response.json(), description)

    return collect_ranged_data(
        source=policy_model.EUROSTAT,
        dataset="dataset",
        entity=f"{dataset}?{query}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


def get_inflation_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly annual rate of change of the Harmonised Index of Consumer
    Prices (HICP) for the euro area and the countries of the European Economic Area.

    The latest month of the euro area and its members is usually the flash estimate,
    published at the end of the month itself and replaced by the final figure about two
    weeks later.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The annual inflation rate as a decimal (0.038 for 3.8%), indexed by
        month with a column per country.
    """
    inflation_rate = collect_eurostat_data(
        HICP_DATASET,
        {"freq": "M", "unit": "RCH_A", "coicop18": "TOTAL"},
        "HICP annual inflation rate",
        start_date,
        end_date,
    )

    # Eurostat publishes the rate in percent.
    return inflation_rate / 100


def get_consumer_price_index(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly Harmonised Index of Consumer Prices (HICP) for the euro area
    and the countries of the European Economic Area, with 2015 = 100.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The all-items index, indexed by month with a column per country.
    """
    return collect_eurostat_data(
        HICP_DATASET,
        {"freq": "M", "unit": "I15", "coicop18": "TOTAL"},
        "HICP index",
        start_date,
        end_date,
    )


def get_unemployment_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly unemployment rate, seasonally adjusted, for the euro area and
    the countries Eurostat covers.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The unemployment rate as a decimal fraction of the labour force
        (0.04 for 4%), indexed by month with a column per country.
    """
    unemployment_rate = collect_eurostat_data(
        UNEMPLOYMENT_DATASET,
        {"freq": "M", "s_adj": "SA", "age": "TOTAL", "sex": "T", "unit": "PC_ACT"},
        "unemployment rate",
        start_date,
        end_date,
    )

    # Eurostat publishes the rate in percent of the labour force.
    return unemployment_rate / 100


def get_gross_domestic_product_growth(
    start_date: str, end_date: str, year_over_year: bool = False
) -> pd.DataFrame:
    """
    Retrieves the growth of gross domestic product in chain-linked volumes, seasonally
    and calendar adjusted, for the euro area and the countries Eurostat covers. The
    latest quarter of the euro area is the preliminary flash estimate, published about
    thirty days after the quarter ends.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        year_over_year (bool): Whether to return the change on the same quarter a year
            earlier instead of on the previous quarter. Defaults to False.

    Returns:
        pd.DataFrame: The growth as a decimal (0.006 for 0.6%), indexed by quarter with a
        column per country.
    """
    growth = collect_eurostat_data(
        GDP_DATASET,
        {
            "freq": "Q",
            "unit": "CLV_PCH_SM" if year_over_year else "CLV_PCH_PRE",
            "s_adj": "SCA",
            "na_item": "B1GQ",
        },
        "GDP growth",
        start_date,
        end_date,
    )

    # Eurostat publishes the growth in percent.
    return growth / 100
