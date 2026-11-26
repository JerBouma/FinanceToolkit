"""Bank for International Settlements (BIS) Model"""

__docformat__ = "google"

import io

import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import (
    COUNTRY_CODES,
    collect_ranged_data,
    keep_quarter_end_of_repeated_quarters,
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
    Request a BIS dataset as CSV.

    The BIS answers a period it has not published yet (such as the current month) with a 404 rather than an empty
    body, so that counts as no data and the cached history is served without an error.

    Args:
        url (str): The URL of the BIS dataset request.
        timeout (int): The number of seconds to wait for an answer.

    Returns:
        str: The CSV text, or an empty string when the BIS has no data for the period.
    """
    try:
        return get_request(url, timeout=timeout).text
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == NOT_FOUND:
            return ""
        raise


def get_central_bank_policy_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve the daily policy rates of the central banks the BIS covers: the Federal Reserve, the European Central
    Bank, the Bank of England, the Bank of Japan and around forty more.

    The history reaches back to the 1940s for some. Only the days that are not cached yet are requested.

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
    Retrieves the monthly consumer prices the BIS compiles from national sources for around sixty countries, including
    China, India, Brazil, the United States and the other G20 economies.

    Only the months that are not cached yet are requested.

    Countries that publish their consumer prices quarterly, such as Australia and New
    Zealand, are spread over the three months of each quarter in the BIS data. Their
    value is kept in the last month of each quarter only (the other months are NaN), so a
    quarter is not counted three times and the change on the month before is not 0.

    Args:
        measure (str): The measure. Options are: "inflation_rate" (the change on a year earlier),
            "consumer_price_index" (the index, 2010 = 100).
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

    return keep_quarter_end_of_repeated_quarters(
        collect_ranged_data(
            source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
            dataset="consumer_prices",
            entity=f"WS_LONG_CPI/{unit}",
            fetch=fetch,
            start_date=start_date,
            end_date=end_date,
            description=description,
        )
    )


# The exchange rates against the US dollar the BIS publishes, by frequency, as the average
# over each period (A) or the rate at its end (E).
EXCHANGE_RATE_FREQUENCIES = {"daily": "D", "monthly": "M"}


def get_exchange_rates(
    frequency: str, start_date: str, end_date: str, end_of_period: bool = True
) -> pd.DataFrame:
    """
    Retrieves the exchange rates of national currencies against the US dollar the BIS compiles from central banks:
    daily for around 80 currencies, with history back to 1945 for some, and monthly for around 190, back to 1940.

    Only the periods that are not cached yet are requested.

    Args:
        frequency (str): The frequency of the data. Options are: "daily" (daily data), "monthly" (monthly data).
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


# The euro is published for the euro area and for every member; the members' series run
# back to before 1999 in their former currency converted at the irrevocable rate, which
# is not the euro's history, so only the euro area series is taken.
EURO_AREA_CODE = "XM"


def get_monthly_average_exchange_rate(
    currency: str, start_date: str, end_date: str
) -> pd.Series:
    """
    Retrieves the average of each month of one currency against the US dollar, the BIS exchange rate series of the
    currency's economy (WS_XRU, collection A).

    The euro is the euro area's series, which starts in June 1974; a currency several economies share, such as the CFA
    franc, has the same rate in each, so the series with the longest history is taken and the others fill its gaps.
    Only the months that are not cached yet are requested.

    Args:
        currency (str): The ISO 4217 code of the currency, e.g. "EUR" or "JPY".
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.Series: Units of the currency per US dollar, indexed by month and named after
        the currency, or an empty Series when the BIS has no rate for it.
    """
    currency = currency.upper()
    description = f"monthly average {currency} exchange rates"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        text = request_csv(
            f"{BASE_URL}WS_XRU/~/M..{currency}.A?format=csv&detail=dataonly"
            f"&startPeriod={fetch_start[:7]}&endPeriod={fetch_end[:7]}",
            timeout=120,
        )

        if not text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(text))
        require_columns(data, {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}, description)

        if currency == "EUR":
            data = data[data["REF_AREA"] == EURO_AREA_CODE]

        rates = data.pivot_table(
            index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="last"
        )
        rates = rates[rates.count().sort_values(ascending=False).index]
        rates = rates.bfill(axis=1).iloc[:, [0]].set_axis([currency], axis=1)
        rates.index = pd.PeriodIndex(rates.index, freq="M")
        rates.index.name = None

        return rates.sort_index()

    rates = collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="exchange_rates",
        entity=f"WS_XRU/M/A/{currency}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )

    return (
        rates[currency].dropna()
        if currency in rates.columns
        else pd.Series(name=currency, dtype=float)
    )


def get_residential_property_prices(
    country_code: str, inflation_adjusted: bool, start_date: str, end_date: str
) -> pd.Series:
    """
    Retrieve the quarterly residential property price index of one economy from the BIS selected series: how house
    prices move, as an index with 2010 = 100.

    The BIS takes the index from the national source, for China the National Bureau of Statistics' prices of newly
    built homes in 70 cities.

    Args:
        country_code (str): The two-letter code of the economy, e.g. "CN".
        inflation_adjusted (bool): Whether to return the real index (deflated by consumer
            prices) instead of the nominal one.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.Series: The price index, indexed by quarter and named after the country, or an
        empty Series when the BIS has no data.
    """
    value = "R" if inflation_adjusted else "N"
    country = COUNTRY_CODES[country_code]
    description = (
        f"{'real' if inflation_adjusted else 'nominal'} house prices of {country}"
    )

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        text = request_csv(
            f"{BASE_URL}WS_SPP/~/Q.{country_code}.{value}.628?format=csv&detail=dataonly"
            f"&startPeriod={str(pd.Period(fetch_start, 'Q')).replace('Q', '-Q')}"
            f"&endPeriod={str(pd.Period(fetch_end, 'Q')).replace('Q', '-Q')}",
            timeout=120,
        )

        if not text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(text))
        require_columns(data, {"TIME_PERIOD", "OBS_VALUE"}, description)
        prices = data.set_index("TIME_PERIOD")[["OBS_VALUE"]].rename(
            columns={"OBS_VALUE": country}
        )
        prices.index = pd.PeriodIndex(prices.index.str.replace("-", ""), freq="Q")
        prices.index.name = None

        return prices.sort_index()

    prices = collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="residential_property",
        entity=f"WS_SPP/{country_code}/{value}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )

    return (
        prices[country]
        if country in prices.columns
        else pd.Series(name=country, dtype=float)
    )


# The commercial types of the BIS property price dataset (RE_TYPE), in the order one is
# preferred over the next: all commercial property, offices, retail, office and retail,
# and industrial property.
COMMERCIAL_PROPERTY_TYPES = ["A", "B", "C", "D", "G"]

# The PRICED_UNIT of an index in real terms, "real price per square meter".
REAL_PRICE_UNIT = "8"


def get_commercial_property_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve the quarterly commercial property price indices the BIS collects from national sources for around twenty
    economies: how the prices of offices, shops and other commercial property move.

    They include the euro area and start in 1945 for the United States and in the 1990s and 2000s for most others.

    Per country one nominal index of commercial property is taken, chosen from every
    series the BIS publishes rather than from those with observations in the date range,
    so the same series is returned whatever the range: the whole country before a region
    or city, and all commercial property (A) before offices (B), retail (C), office and
    retail (D) and industrial property (G), in that order. Among regions the lowest BIS
    area code is taken. Indices of dwellings, land, agricultural property and real prices
    are left out, so a country without a nominal commercial property index has no column.

    Args:
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The price index, indexed by quarter with a column per country.
    """
    description = "commercial property prices"

    def commercial_series(data: pd.DataFrame) -> pd.DataFrame:
        # The dataset also holds dwellings (rented ones for Switzerland), land and real
        # prices (PRICED_UNIT 8, Iceland), which are not commercial property prices.
        data = data[data["RE_TYPE"].astype(str).isin(COMMERCIAL_PROPERTY_TYPES)]

        if "PRICED_UNIT" in data.columns:
            data = data[data["PRICED_UNIT"].astype(str) != REAL_PRICE_UNIT]

        return data

    def choose_series(series_keys: pd.DataFrame) -> pd.DataFrame:
        # One series per country from the keys of every published series, so the choice
        # does not depend on which series have observations in the range fetched: the
        # cache joins ranges, which would otherwise join different series (such as a
        # region of Saudi Arabia before 2021 and the whole country after).
        keys = commercial_series(series_keys).copy()
        keys["area"] = pd.to_numeric(keys["COVERED_AREA"], errors="coerce")
        keys["type"] = keys["RE_TYPE"].astype(str).map(COMMERCIAL_PROPERTY_TYPES.index)
        keys["whole_country"] = keys["area"] == 0
        keys["preferred"] = keys["whole_country"] & (keys["type"] == 0)
        chosen = keys.sort_values(
            ["preferred", "whole_country", "type", "area"],
            ascending=[False, False, True, True],
            kind="stable",
        ).drop_duplicates(subset="REF_AREA")

        return chosen.drop(columns=["area", "type", "whole_country", "preferred"])

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        text = request_csv(
            f"{BASE_URL}WS_CPP/~/Q.......?format=csv&detail=dataonly"
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

        data = commercial_series(data)

        if data.empty:
            return pd.DataFrame()

        series_keys = pd.read_csv(
            io.StringIO(
                request_csv(
                    f"{BASE_URL}WS_CPP/~/Q.......?format=csv&detail=serieskeysonly",
                    timeout=120,
                )
            )
        )
        series_columns = [
            column
            for column in data.columns
            if column not in ("FREQ", "TIME_PERIOD", "OBS_VALUE")
        ]
        require_columns(series_keys, set(series_columns), description)
        chosen = choose_series(
            series_keys[series_columns].drop_duplicates().astype(str)
        )

        data = data.astype({column: str for column in series_columns}).merge(
            chosen, on=series_columns
        )
        data = data[data["REF_AREA"].isin(COUNTRY_CODES.keys())]

        if data.empty:
            return pd.DataFrame()

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


def get_credit_statistics(
    dataflow: str, key: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Retrieves one quarterly series per country from the BIS credit statistics: the credit-to-GDP gaps (WS_CREDIT_GAP),
    the debt service ratios (WS_DSR) or the total credit to the non-financial sector (WS_TC).

    The key selects the series within the dataflow, such as the borrowing sector and the unit, and leaves the country
    open, so that each country gives exactly one series. Aggregates such as "all reporting economies" have no country
    and are left out.

    Args:
        dataflow (str): The BIS dataflow, e.g. "WS_DSR".
        key (str): The series key with the country left empty, e.g. "Q..P".
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The series as decimals (the BIS publishes percentages, of GDP or of
        income), indexed by quarter with a column per country.
    """
    description = f"BIS credit statistics ({dataflow})"

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        text = request_csv(
            f"{BASE_URL}{dataflow}/~/{key}?format=csv&detail=dataonly"
            f"&startPeriod={str(pd.Period(fetch_start, 'Q')).replace('Q', '-Q')}"
            f"&endPeriod={str(pd.Period(fetch_end, 'Q')).replace('Q', '-Q')}",
            timeout=120,
        )

        if not text.strip():
            return pd.DataFrame()

        data = pd.read_csv(io.StringIO(text))
        require_columns(
            data, {"BORROWERS_CTY", "TIME_PERIOD", "OBS_VALUE"}, description
        )

        data = data[data["BORROWERS_CTY"].isin(COUNTRY_CODES.keys())]
        if data.empty:
            return pd.DataFrame()

        statistics = data.pivot(
            index="TIME_PERIOD", columns="BORROWERS_CTY", values="OBS_VALUE"
        )
        statistics = statistics.rename(columns=COUNTRY_CODES)
        statistics.index = pd.PeriodIndex(
            statistics.index.str.replace("-", ""), freq="Q"
        )
        statistics.index.name = None
        statistics.columns.name = None

        return statistics.sort_index() / 100

    return collect_ranged_data(
        source=policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
        dataset="credit",
        entity=f"{dataflow}/{key}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )
