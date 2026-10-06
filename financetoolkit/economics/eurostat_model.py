"""Eurostat Model"""

__docformat__ = "google"

import re
from collections.abc import Callable

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
# A yearly period is written as its four-digit year.
YEAR_LENGTH = 4

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


def parse_json_stat(
    response: dict,
    description: str,
    extra_dimension: str | None = None,
    extra_labels: Callable[[str], object] | None = None,
) -> pd.DataFrame:
    """
    Converts a Eurostat JSON-stat response into a DataFrame with one column per country.

    Every dimension other than geo, time and the extra dimension must be pinned to a
    single value by the query. Otherwise several series would land in one column and
    whichever came last would win, so that is treated as an error rather than resolved
    silently.

    Args:
        response (dict): The decoded JSON-stat response.
        description (str): What was retrieved, used in the error messages.
        extra_dimension (str | None): A dimension that may hold several values, such as
            "age" for a life table, which becomes the second level of the columns.
            Defaults to None.
        extra_labels (Callable[[str], object] | None): Converts a code of the extra
            dimension into its label, e.g. "Y_LT1" into 0. Defaults to None, which keeps
            the codes.

    Returns:
        pd.DataFrame: The values indexed by year, quarter or month, with a column per
        country, or per country and value of the extra dimension.

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
        if dimension not in ("geo", "time", extra_dimension) and size > 1
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
    extra_position = (
        dimensions.index(extra_dimension) if extra_dimension in dimensions else None
    )

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
                (
                    categories[extra_position][positions[extra_position]][0]
                    if extra_position is not None
                    else None
                ),
                value,
            )
        )

    if not records:
        return pd.DataFrame()

    long_format = pd.DataFrame(records, columns=["geo", "time", "extra", "value"])
    data = long_format.pivot(
        index="time",
        columns=["geo", "extra"] if extra_position is not None else "geo",
        values="value",
    )

    # "EU" is the European Union in Eurostat data but the euro area in the economic
    # calendar the country names are shared with, so the union comes from EU27_2020.
    geo_codes = list(
        dict.fromkeys(
            data.columns.get_level_values(0)
            if extra_position is not None
            else data.columns
        )
    )
    euro_area = _resolve_euro_area(geo_codes)
    countries = {
        code: COUNTRY_CODES[code]
        for code in geo_codes
        if code in COUNTRY_CODES and code != "EU" and not code.startswith("EA")
    }
    if euro_area is not None:
        countries[euro_area] = "Euro Area"

    if extra_position is None:
        data = data[list(countries)].rename(columns=countries)
        data.columns.name = None
    else:
        data = data.loc[:, data.columns.get_level_values(0).isin(list(countries))]
        data.columns = pd.MultiIndex.from_tuples(
            [
                (countries[code], extra_labels(extra) if extra_labels else extra)
                for code, extra in data.columns
            ],
            names=["Country", extra_dimension.title()],
        )
        data = data.sort_index(axis=1)

    # Years are written as "2024", quarters as "2026-Q2" and months as "2026-09".
    first = str(data.index[0])
    frequency = "Y" if len(first) == YEAR_LENGTH else "Q" if first[-2] == "Q" else "M"
    data.index = pd.PeriodIndex(data.index, freq=frequency)
    data.index.name = None

    return data.sort_index().astype(float)


def collect_eurostat_data(
    dataset: str,
    filters: dict[str, str],
    description: str,
    start_date: str,
    end_date: str,
    extra_dimension: str | None = None,
    extra_labels: Callable[[str], object] | None = None,
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
        extra_dimension (str | None): A dimension that may hold several values, see
            parse_json_stat. Defaults to None.
        extra_labels (Callable[[str], object] | None): Converts its codes into labels.
            Defaults to None.

    Returns:
        pd.DataFrame: The values indexed by year, quarter or month, with a column per
        country, or per country and value of the extra dimension.
    """
    # Eurostat takes "2024" for a yearly, "2026-Q3" for a quarterly and "2026-09" for a
    # monthly dataset.
    frequency = filters.get("freq")

    def to_period(date: str) -> str:
        if frequency == "A":
            return date[:4]
        if frequency == "Q":
            return str(pd.Period(date, freq="Q")).replace("Q", "-Q")
        return date[:7]

    query = "&".join(
        f"{dimension}={value}" for dimension, value in sorted(filters.items())
    )

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{BASE_URL}{dataset}?{query}&sinceTimePeriod={to_period(fetch_start)}"
            f"&untilTimePeriod={to_period(fetch_end)}",
            timeout=120,
        )

        return parse_json_stat(
            response.json(), description, extra_dimension, extra_labels
        )

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


# The life table measures, by the name the Finance Toolkit uses for them.
LIFE_TABLE_MEASURES = {
    "death_rate": "DEATHRATE",
    "death_probability": "PROBDEATH",
    "survival_probability": "PROBSURV",
    "survivors": "SURVIVORS",
    "deaths": "NUMBERDYING",
    "person_years": "PYLIVED",
    "total_person_years": "TOTPYLIVED",
    "life_expectancy": "LIFEXP",
}
LIFE_TABLE_SEXES = {"total": "T", "male": "M", "female": "F"}


# Older life tables end at an open-ended group of 85 and over, published next to the
# single age of 85 of the later tables.
OPEN_ENDED_85 = "Y_GE85"


def _age_label(code: str) -> int | str:
    """
    Converts an age code of the Eurostat life tables into the age in years: "Y_LT1" is 0,
    "Y45" is 45 and "Y_GE95", the open-ended last age group, is 95. "Y_GE85", the last age
    group of older tables, keeps its code so get_life_table can tell it from age 85.

    Args:
        code (str): The age code.

    Returns:
        int | str: The age in years, or the code of the 85 and over group.
    """
    if code == "Y_LT1":
        return 0

    if code == OPEN_ENDED_85:
        return code

    return int(re.sub(r"\D", "", code))


def get_life_table(
    measure: str,
    sex: str,
    start_date: str,
    end_date: str,
    country_codes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Retrieves the life tables Eurostat compiles yearly for the countries of the European
    Economic Area, by single year of age from 0 to 95 and over, from 1960.

    Args:
        measure (str): A key of LIFE_TABLE_MEASURES, e.g. "death_probability".
        sex (str): "total", "male" or "female".
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        country_codes (list[str] | None): The Eurostat codes of the countries to retrieve.
            Defaults to None, which retrieves every country.

    Returns:
        pd.DataFrame: The measure, indexed by year with a column per country and age. Age
        85 is the 85 and over group where a table ends there, as 95 is 95 and over.
    """
    filters = {
        "freq": "A",
        "indic_de": LIFE_TABLE_MEASURES[measure],
        "sex": LIFE_TABLE_SEXES[sex],
    }

    if country_codes:
        filters["geo"] = "&geo=".join(country_codes)

    life_table = collect_eurostat_data(
        "demo_mlifetable",
        filters,
        f"{measure.replace('_', ' ')} life table",
        start_date,
        end_date,
        extra_dimension="age",
        extra_labels=_age_label,
    )

    if life_table.empty:
        return life_table

    # The single age of 85 is kept, and the 85 and over group fills it only where a table
    # ends at 85 and so has no single age. A table cached before the group kept its code
    # labels both 85, the single age first.
    ages = life_table.columns.get_level_values(1)
    single = life_table.loc[:, ages != OPEN_ENDED_85]
    single = single.T.groupby(level=[0, 1], sort=False).first().T
    open_ended = life_table.loc[:, ages == OPEN_ENDED_85]

    for country, _ in open_ended.columns:
        column = (country, 85)
        group = open_ended[(country, OPEN_ENDED_85)]
        single[column] = (
            single[column].fillna(group) if column in single.columns else group
        )

    single.columns = pd.MultiIndex.from_tuples(
        single.columns, names=life_table.columns.names
    )

    return single.sort_index(axis=1, level=[0, 1], sort_remaining=False)
