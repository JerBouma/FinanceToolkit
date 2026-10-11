"""International Monetary Fund (IMF) Model"""

__docformat__ = "google"

import re

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, collect_ranged_data
from financetoolkit.economics.oecd_model import CODE_TO_COUNTRY
from financetoolkit.utilities.requests_model import get_request

BASE_URL = "https://api.imf.org/external/sdmx/2.1/"

# The consumer price index dataset, its country codelist and the series of the annual
# rate of change of the all-items index, monthly.
STRUCTURE_URL = f"{BASE_URL}dataflow/IMF.STA/CPI/latest?references=descendants"
INFLATION_SERIES = "CPI._T.YOY_PCH_PA_PT.M"

# The API refuses a request for every country at once, so countries are requested in
# batches of this size.
BATCH_SIZE = 40

CODELIST_PATTERN = re.compile(
    r'<str:Codelist[^>]*id="CL_COUNTRY"[^>]*>(.*?)</str:Codelist>', re.S
)
CODE_PATTERN = re.compile(r'<str:Code id="([A-Z]{3})"[^>]*>(.*?)</str:Code>', re.S)
NAME_PATTERN = re.compile(r'<com:Name xml:lang="en">([^<]+)</com:Name>')
SERIES_PATTERN = re.compile(r"<Series\b([^>]*)>(.*?)</Series>", re.S)
OBSERVATION_PATTERN = re.compile(r"<Obs\b([^>]*)/?>")
COUNTRY_ATTRIBUTE = re.compile(r'\bCOUNTRY="([A-Z]{3})"')
PERIOD_ATTRIBUTE = re.compile(r'\bTIME_PERIOD="([^"]+)"')
VALUE_ATTRIBUTE = re.compile(r'\bOBS_VALUE="([^"]+)"')


def _short_name(imf_name: str) -> str:
    """
    Shortens an IMF country name to its common form: "Bahamas, The" becomes "Bahamas" and
    "Macao Special Administrative Region, People's Republic of China" becomes "Macao".

    Args:
        imf_name (str): The name as the IMF writes it.

    Returns:
        str: The common name.
    """
    return (
        imf_name.split(",", maxsplit=1)[0]
        .replace(" Special Administrative Region", "")
        .strip()
    )


def get_countries() -> pd.DataFrame:
    """
    Lists the countries of the IMF consumer price index dataset, named as elsewhere in the
    Finance Toolkit where the Finance Toolkit knows the country, otherwise as the IMF names it.

    Returns:
        pd.DataFrame: The name per ISO 3166 alpha-3 code.

    Raises:
        ValueError: When the structure has no country codelist, which means it changed.
    """
    description = "IMF consumer price index countries"

    def fetch() -> pd.DataFrame:
        structure = get_request(STRUCTURE_URL, timeout=120).text
        codelist = CODELIST_PATTERN.search(structure)

        if codelist is None:
            raise ValueError(
                f"The {description} structure has no country codelist, which means the API changed."
            )

        imf_names = {
            code: (name.group(1) if (name := NAME_PATTERN.search(block)) else code)
            for code, block in CODE_PATTERN.findall(codelist.group(1))
        }
        names = {
            code: CODE_TO_COUNTRY.get(code, _short_name(imf_name))
            for code, imf_name in imf_names.items()
        }

        # A historical country can shorten to the name of its successor, such as North
        # Vietnam (VDR) to "Vietnam", so a name used twice keeps the IMF's full name.
        used = {code: name for code, name in names.items() if code in CODE_TO_COUNTRY}
        for code, name in names.items():
            if code not in CODE_TO_COUNTRY and name in used.values():
                names[code] = imf_names[code]
            used[code] = names[code]

        return pd.DataFrame({"Country": list(names.values())}, index=list(names))

    return collect_cached_data(
        source=policy_model.INTERNATIONAL_MONETARY_FUND,
        dataset="codelist",
        entity="CPI/CL_COUNTRY",
        fetch=fetch,
        description=description,
    )


def _parse_observations(text: str) -> list[tuple[str, str, float]]:
    """
    Reads the (country, month, value) observations of an SDMX structure-specific response.
    The format is simple and fixed, so it is read with patterns rather than an XML parser,
    which keeps external XML features such as entity expansion out of the picture.

    Args:
        text (str): The response.

    Returns:
        list[tuple[str, str, float]]: The observations.
    """
    observations = []

    for attributes, body in SERIES_PATTERN.findall(text):
        country = COUNTRY_ATTRIBUTE.search(attributes)

        if country is None:
            continue

        for observation in OBSERVATION_PATTERN.finditer(body):
            period = PERIOD_ATTRIBUTE.search(observation.group(1))
            value = VALUE_ATTRIBUTE.search(observation.group(1))
            number = pd.to_numeric(value.group(1), errors="coerce") if value else None

            if period and number is not None and pd.notna(number):
                observations.append((country.group(1), period.group(1), float(number)))

    return observations


def get_inflation_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly annual rate of change of the consumer price index the IMF
    collects from national statistical offices, for around 160 countries, including many
    that no other source in the Finance Toolkit covers. Only the months that are not cached
    yet are requested.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The rate as a decimal (0.048 for 4.8%), indexed by month with a column
        per country.
    """
    countries = get_countries()
    description = "IMF consumer prices"

    if countries.empty:
        return pd.DataFrame()

    codes = list(countries.index)

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        def batch(chunk: list[str]) -> list[tuple[str, str, float]]:
            response = get_request(
                f"{BASE_URL}data/IMF.STA,CPI/{'+'.join(chunk)}.{INFLATION_SERIES}"
                f"?startPeriod={fetch_start[:7]}&endPeriod={fetch_end[:7]}",
                timeout=180,
            )
            return _parse_observations(response.text) if response.text.strip() else []

        batches = helpers.run_in_parallel(
            batch,
            [(codes[i : i + BATCH_SIZE],) for i in range(0, len(codes), BATCH_SIZE)],
            max_workers=4,
        )
        observations = [observation for result in batches for observation in result]

        if not observations:
            return pd.DataFrame()

        data = pd.DataFrame(observations, columns=["country", "period", "value"])
        rates = data.pivot_table(
            index="period", columns="country", values="value", aggfunc="last"
        )
        rates = rates.rename(columns=countries["Country"].to_dict())
        # Months are written as "2026-M07".
        rates.index = pd.PeriodIndex(rates.index.str.replace("M", ""), freq="M")
        rates.index.name = None
        rates.columns.name = None

        # The IMF publishes the rate in percent.
        return rates.sort_index() / 100

    return collect_ranged_data(
        source=policy_model.INTERNATIONAL_MONETARY_FUND,
        dataset="consumer_prices",
        entity=INFLATION_SERIES,
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
