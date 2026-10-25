"""Bank for International Settlements (BIS) Model"""

__docformat__ = "google"

import io

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import (
    COUNTRY_CODES,
    collect_ranged_data,
    require_columns,
)
from financetoolkit.utilities.requests_model import get_request

# Version 2 of the BIS SDMX API, asking for the latest version of every dataflow (~), so
# that a new version of a dataset does not break the request.
BASE_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/"

# The HTTP status the BIS answers for a period without published data.
NOT_FOUND = 404


def request_csv(url: str, timeout: int) -> str:
    """
    Request a BIS dataset as CSV. The BIS answers a period it has not published yet (such as
    the current month) with a 404 rather than an empty body, so that counts as no data and
    the cached history is served without an error.
    """
    try:
        return get_request(url, timeout=timeout).text
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == NOT_FOUND:
            return ""
        raise


def get_central_bank_policy_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the daily policy rates of the central banks the BIS covers, which includes
    the Federal Reserve, the European Central Bank, the Bank of England, the Bank of
    Japan and around forty more, with history back to the 1940s for some. Only the days
    that are not cached yet are requested.

    The euro area is published as "Euro Area"; the euro area members keep a column for
    the years before they adopted the euro, which ends when they did.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The policy rates as decimals (0.0425 for 4.25%), indexed by day with
        a column per country.
    """
    description = "central bank policy rates"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        text = request_csv(
            f"{BASE_URL}WS_CBPOL/~/D.?format=csv&detail=dataonly"
            f"&startPeriod={fetch_start}&endPeriod={fetch_end}",
            timeout=300,
        )

        # A range without observations is answered with an empty body.
        if not text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(text))
        require_columns(data, {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}, description)

        data = data[data["REF_AREA"].isin(COUNTRY_CODES.keys())]
        data = data.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
        data = data.rename(columns=COUNTRY_CODES)
        data.index = pd.PeriodIndex(data.index, freq="D")
        data.index.name = None
        data.columns.name = None

        # The BIS publishes the rates in percent.
        return data.sort_index() / 100

    return collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="dataset",
        entity="WS_CBPOL/D",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


# The units the BIS publishes its consumer prices in.
CONSUMER_PRICE_UNITS = {"inflation_rate": "771", "consumer_price_index": "628"}


def get_consumer_prices(measure: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly consumer prices the BIS compiles from national sources for
    around sixty countries, including China, India, Brazil, the United States and the
    other G20 economies. Only the months that are not cached yet are requested.

    Args:
        measure (str): "inflation_rate" for the change on a year earlier or
            "consumer_price_index" for the index (2010 = 100).
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The rate as a decimal (0.034 for 3.4%) or the index, indexed by month
        with a column per country.
    """
    unit = CONSUMER_PRICE_UNITS[measure]
    description = "BIS consumer prices"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        text = request_csv(
            f"{BASE_URL}WS_LONG_CPI/~/M..{unit}?format=csv&detail=dataonly"
            f"&startPeriod={fetch_start[:7]}&endPeriod={fetch_end[:7]}",
            timeout=120,
        )

        if not text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(text))
        require_columns(data, {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}, description)

        data = data[data["REF_AREA"].isin(COUNTRY_CODES.keys())]
        data = data.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
        data = data.rename(columns=COUNTRY_CODES)
        data.index = pd.PeriodIndex(data.index, freq="M")
        data.index.name = None
        data.columns.name = None

        # The rate of change is published in percent; the index is a level.
        return data.sort_index() / (100 if measure == "inflation_rate" else 1)

    return collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="consumer_prices",
        entity=f"WS_LONG_CPI/{unit}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


# The exchange rates against the US dollar the BIS publishes, by frequency, as the average
# over each period (A) or the rate at its end (E).
EXCHANGE_RATE_FREQUENCIES = {"daily": "D", "monthly": "M"}


def get_exchange_rates(
    frequency: str, start_date: str, end_date: str, end_of_period: bool = True
) -> pd.DataFrame:
    """
    Retrieves the exchange rates of national currencies against the US dollar the BIS
    compiles from central banks: daily for around 80 currencies, with history back to 1945
    for some, and monthly for around 190, back to 1940. Only the periods that are not
    cached yet are requested.

    Args:
        frequency (str): "daily" or "monthly".
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        end_of_period (bool): For monthly data, whether to return the rate at the end of
            each month instead of its average. Defaults to True.

    Returns:
        pd.DataFrame: Units of national currency per US dollar, indexed by day or month with
        a column per country.
    """
    code = EXCHANGE_RATE_FREQUENCIES[frequency]
    collection = "E" if end_of_period and frequency == "monthly" else "A"
    description = f"{frequency} exchange rates"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        period_start = fetch_start if code == "D" else fetch_start[:7]
        period_end = fetch_end if code == "D" else fetch_end[:7]
        text = request_csv(
            f"{BASE_URL}WS_XRU/~/{code}...{collection if code == 'M' else ''}?format=csv"
            f"&detail=dataonly&startPeriod={period_start}&endPeriod={period_end}",
            timeout=300,
        )

        if not text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(text))
        require_columns(data, {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}, description)

        data = data[data["REF_AREA"].isin(COUNTRY_CODES.keys())]
        data = data.drop_duplicates(subset=["TIME_PERIOD", "REF_AREA"], keep="last")
        rates = data.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
        rates = rates.rename(columns=COUNTRY_CODES)
        rates.index = pd.PeriodIndex(rates.index, freq=code)
        rates.index.name = None
        rates.columns.name = None

        return rates.sort_index()

    return collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="exchange_rates",
        entity=f"WS_XRU/{code}/{collection}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )


# The commercial types of the BIS property price dataset (RE_TYPE): all commercial
# property, offices, retail, office and retail, and industrial property.
COMMERCIAL_PROPERTY_TYPES = {"A", "B", "C", "D", "G"}

# The PRICED_UNIT of an index in real terms, "real price per square meter".
REAL_PRICE_UNIT = "8"


def get_commercial_property_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the quarterly commercial property price indices the BIS collects from
    national sources, for around twenty economies including the euro area, from 1945 for
    the United States and from the 1990s and 2000s for most others.

    Per country one nominal index of commercial property is taken: the one for all
    commercial property in the whole country where there is one, otherwise the longest
    index of a commercial type (offices, retail, office and retail, or industrial
    property), preferring the whole country. Indices of dwellings, land, agricultural
    property and real prices are left out, so a country without a nominal commercial
    property index has no column.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The price index, indexed by quarter with a column per country.
    """
    description = "commercial property prices"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        text = request_csv(
            f"{BASE_URL}WS_CPP/~/Q.......?format=csv&detail=dataonly"
            # Quarters are written as "2026-Q2".
            f"&startPeriod={str(pd.Period(fetch_start, 'Q')).replace('Q', '-Q')}"
            f"&endPeriod={str(pd.Period(fetch_end, 'Q')).replace('Q', '-Q')}",
            timeout=120,
        )

        if not text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(text))
        require_columns(
            data,
            {"REF_AREA", "COVERED_AREA", "RE_TYPE", "TIME_PERIOD", "OBS_VALUE"},
            description,
        )

        # The dataset also holds dwellings (rented ones for Switzerland), land and real
        # prices (PRICED_UNIT 8, Iceland), which are not commercial property prices and
        # would otherwise be picked for a country where they are the longest series.
        data = data[data["RE_TYPE"].astype(str).isin(COMMERCIAL_PROPERTY_TYPES)]

        if "PRICED_UNIT" in data.columns:
            data = data[data["PRICED_UNIT"].astype(str) != REAL_PRICE_UNIT]

        if data.empty:
            return pd.DataFrame()

        series_columns = [
            column
            for column in data.columns
            if column not in ("FREQ", "TIME_PERIOD", "OBS_VALUE")
        ]
        lengths = data.groupby(series_columns, as_index=False).size()
        lengths["whole_country"] = lengths["COVERED_AREA"].astype(str) == "0"
        lengths["preferred"] = lengths["whole_country"] & (
            lengths["RE_TYPE"].astype(str) == "A"
        )
        chosen = (
            lengths.sort_values(
                ["preferred", "whole_country", "size"], ascending=False, kind="stable"
            )
            .drop_duplicates(subset="REF_AREA")
            .drop(columns=["preferred", "whole_country", "size"])
        )

        data = data.merge(chosen, on=series_columns)
        data = data[data["REF_AREA"].isin(COUNTRY_CODES.keys())]
        prices = data.pivot(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
        prices = prices.rename(columns=COUNTRY_CODES)
        prices.index = pd.PeriodIndex(prices.index.str.replace("-", ""), freq="Q")
        prices.index.name = None
        prices.columns.name = None

        return prices.sort_index()

    return collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="commercial_property",
        entity="WS_CPP",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
