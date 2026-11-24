"""International Monetary Fund (IMF) Model"""

__docformat__ = "google"

import re

import pandas as pd

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.economics.helpers import collect_cached_data, collect_ranged_data
from financetoolkit.economics.sources.oecd_model import CODE_TO_COUNTRY
from financetoolkit.utilities.requests_model import get_request
from financetoolkit.utilities.statistics_model import EURO_AREA_MEMBERS

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
# A series without observations is written as an empty element, <Series .../>, which must
# not take the observations of the series after it.
SERIES_PATTERN = re.compile(r"<Series\b([^>]*?)(?:/>|>(.*?)</Series>)", re.S)
OBSERVATION_PATTERN = re.compile(r"<Obs\b([^>]*)/?>")
COUNTRY_ATTRIBUTE = re.compile(r'\bCOUNTRY="([A-Z0-9]+)"')
INDICATOR_ATTRIBUTE = re.compile(r'\bINDICATOR="([^"]+)"')
PERIOD_ATTRIBUTE = re.compile(r'\bTIME_PERIOD="([^"]+)"')
VALUE_ATTRIBUTE = re.compile(r'\bOBS_VALUE="([^"]+)"')


# The IMF countries whose common name differs from the one the other sources (Eurostat, the
# BIS and the Global Macro Database) use, so a country is one column no matter which source
# it came from and the IMF methods name it as the gross domestic product does. The IMF's
# "Curaçao and Sint Maarten" (CWX) is their monetary union, not Curaçao alone, so it
# keeps the IMF's name.
IMF_TO_COUNTRY_NAMES = {
    "CPV": "Cape Verde",
    "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia",
    "MAC": "Macau",
    "NRU": "Nauru",
    "STP": "Sao Tome and Principe",
    "TCA": "Turks and Caicos",
    "VCT": "St-Vincent",
    "WBG": "Palestine",
}


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

    # The IMF's own names are cached and the common names applied to them on every read,
    # so a change of IMF_TO_COUNTRY_NAMES applies at once rather than when the codelist
    # expires. A common name read back maps to itself.
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

        return pd.DataFrame(
            {"Country": list(imf_names.values())}, index=list(imf_names)
        )

    # The IMF's own names, under a key apart from "codelist", which holds the common names, so a
    # cache shared with code that reads "codelist" as it is never mixes the two.
    codelist = collect_cached_data(
        source=policy_model.INTERNATIONAL_MONETARY_FUND,
        dataset="codelist_raw",
        entity="CPI/CL_COUNTRY",
        fetch=fetch,
        description=description,
    )

    if codelist.empty:
        return codelist

    imf_names = codelist["Country"].to_dict()
    names = {
        code: CODE_TO_COUNTRY.get(
            code, IMF_TO_COUNTRY_NAMES.get(code, _short_name(imf_name))
        )
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


def _parse_observations(text: str) -> list[tuple[str, str, str, float]]:
    """
    Reads the (country, indicator, period, value) observations of an SDMX
    structure-specific response. The format is simple and fixed, so it is read with
    patterns rather than an XML parser, which keeps external XML features such as entity
    expansion out of the picture.

    Args:
        text (str): The response.

    Returns:
        list[tuple[str, str, str, float]]: The observations, with an empty indicator for a
        series without one.
    """
    observations = []

    for attributes, body in SERIES_PATTERN.findall(text):
        country = COUNTRY_ATTRIBUTE.search(attributes)
        indicator = INDICATOR_ATTRIBUTE.search(attributes)

        if country is None:
            continue

        for observation in OBSERVATION_PATTERN.finditer(body):
            period = PERIOD_ATTRIBUTE.search(observation.group(1))
            value = VALUE_ATTRIBUTE.search(observation.group(1))
            number = pd.to_numeric(value.group(1), errors="coerce") if value else None

            if period and number is not None and pd.notna(number):
                observations.append(
                    (
                        country.group(1),
                        indicator.group(1) if indicator else "",
                        period.group(1),
                        float(number),
                    )
                )

    return observations


def get_inflation_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves the monthly annual rate of change of the consumer price index the IMF collects from national statistical
    offices, for around 160 countries, including many that no other source in the Finance Toolkit covers.

    Only the months that are not cached yet are requested.

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

        data = pd.DataFrame(
            observations, columns=["country", "indicator", "period", "value"]
        )
        rates = data.pivot_table(
            index="period", columns="country", values="value", aggfunc="last"
        )
        # Months are written as "2026-M07".
        rates.index = pd.PeriodIndex(rates.index.str.replace("M", ""), freq="M")
        rates.index.name = None
        rates.columns.name = None

        # The IMF publishes the rate in percent.
        return rates.sort_index() / 100

    # Cached by IMF code and named on every read, so a change of a country's name never
    # splits it into two columns. Not "consumer_prices", which holds named columns.
    rates = collect_ranged_data(
        source=policy_model.INTERNATIONAL_MONETARY_FUND,
        dataset="consumer_prices_by_code",
        entity=INFLATION_SERIES,
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )

    return rates.rename(columns=countries["Country"].to_dict())


# The Government Finance Statistics and national accounts the IMF compiles from the
# national statistical offices and finance ministries, for general government (S13): the
# central, state and local governments and the social security funds together. The
# interest and revenue come from the Quarterly Government Finance Statistics (QGFS), the
# expenditure by function from the Classification of the Functions of Government (COFOG)
# and the GDP from the National Economic Accounts, quarterly (QNEA, not seasonally
# adjusted, like the government flows) and annual (ANEA).
GOVERNMENT_INTEREST = "G24_T_XDC"
GOVERNMENT_REVENUE = "G1_T_XDC"
GOVERNMENT_FINANCE_KEYS = {
    "Q": ".SOO.S13.{indicators}.Q",
    "Y": ".SOO.S13.{indicators}.A",
}
GROSS_DOMESTIC_PRODUCT_KEYS = {
    "Q": ("QNEA", ".B1GQ.V.NSA.XDC.Q"),
    "Y": ("ANEA", ".B1GQ.V.XDC.A"),
}

# The ten COFOG divisions, plus the public debt transactions (group 01.7, mostly the
# interest on government debt, within general public services) and the total.
EXPENDITURE_FUNCTIONS = {
    "General Public Services": "GF01_T",
    "Defence": "GF02_T",
    "Public Order and Safety": "GF03_T",
    "Economic Affairs": "GF04_T",
    "Environmental Protection": "GF05_T",
    "Housing and Community Amenities": "GF06_T",
    "Health": "GF07_T",
    "Recreation, Culture and Religion": "GF08_T",
    "Education": "GF09_T",
    "Social Protection": "GF10_T",
    "Public Debt Transactions": "GF0170_T",
    "Total": "TOTOUT_T",
}
DIVISIONS = list(EXPENDITURE_FUNCTIONS)[:10]

# The aggregates that are not in the country codelist.
AGGREGATE_NAMES = {"G163": "Euro Area", "G309": "Eastern Caribbean Currency Union"}

PERIOD_CODES = {"quarterly": "Q", "yearly": "Y"}


def collect_government_finance(
    dataflow: str, key: str, start_date: str, end_date: str, description: str
) -> pd.DataFrame:
    """
    Retrieves an IMF dataset for every country it covers, with a column per country and indicator.

    Only the periods that are not cached yet are requested.

    Args:
        dataflow (str): The dataflow, e.g. "QGFS".
        key (str): The series key without the country, e.g. ".SOO.S13.G24_T_XDC.Q". It
            ends with the frequency, "A" for yearly or "Q" for quarterly.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        description (str): What is retrieved, used in the log messages.

    Returns:
        pd.DataFrame: The values in units of national currency (or as published), indexed
        by year or quarter with a (Country, Indicator) column.
    """
    yearly = key.endswith(".A")

    def fetch(fetch_start: str, fetch_end: str) -> pd.DataFrame:
        response = get_request(
            f"{BASE_URL}data/IMF.STA,{dataflow}/{key}"
            f"?startPeriod={fetch_start[:4]}&endPeriod={fetch_end[:4]}",
            timeout=180,
        )
        observations = _parse_observations(response.text)

        if not observations:
            return pd.DataFrame()

        data = pd.DataFrame(
            observations, columns=["country", "indicator", "period", "value"]
        ).pivot_table(
            index="period",
            columns=["country", "indicator"],
            values="value",
            aggfunc="last",
        )
        data.columns.names = ["Country", "Indicator"]
        # Quarters are written as "2026-Q2" and years as "2026".
        data.index = pd.PeriodIndex(
            data.index.str.replace("-", ""), freq="Y" if yearly else "Q"
        )
        data.index.name = None

        return data.sort_index().sort_index(axis=1)

    # Cached by IMF code and named on every read, so a change of a country's name never
    # splits it into two columns. Not "government_finance": see the policy of "series_by_code".
    data = collect_ranged_data(
        source=policy_model.INTERNATIONAL_MONETARY_FUND,
        dataset="series_by_code",
        entity=f"{dataflow}/{key}",
        fetch=fetch,
        start_date=start_date,
        end_date=end_date,
        description=description,
    )

    if data.empty:
        return data

    names = {**get_countries()["Country"].to_dict(), **AGGREGATE_NAMES}
    data.columns = pd.MultiIndex.from_tuples(
        [(names.get(code, code), indicator) for code, indicator in data.columns],
        names=["Country", "Indicator"],
    )

    return data


def get_government_interest(
    period: str, start_date: str, end_date: str, denominator: str | None = None
) -> pd.DataFrame:
    """
    Retrieve the interest payable by the general government (GFS item 24, D.41 in the national accounts) for around 40
    economies: what governments pay on their debt.

    The interest is on an accrual basis and before interest received is netted; the economies include the United
    States, the euro area and its members, the United Kingdom, Canada, Australia, Brazil, Turkey and Indonesia.

    As a ratio, the interest is divided by the GDP or the government revenue of the same
    period. Quarterly ratios cover the four quarters up to and including each quarter, so
    they are annual figures without the seasonal swings of tax receipts and GDP.

    Args:
        period (str): The frequency of the data. Options are: "quarterly" (quarterly data), "yearly" (yearly data).
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        denominator (str | None): What the interest is divided by. Options are: None (the interest itself in millions
            of national currency), "gdp" (the share of GDP), "revenue" (the share of government revenue). Defaults to
            None.

    Returns:
        pd.DataFrame: The interest or its ratio (a decimal, 0.04 for 4%), indexed by quarter
        or year with a column per country.
    """
    frequency = PERIOD_CODES[period]
    # The year before the start date, for the four quarters of the first quarterly ratio.
    fetch_start = f"{int(start_date[:4]) - 1}{start_date[4:]}"
    indicators = (
        GOVERNMENT_INTEREST
        if denominator != "revenue"
        else f"{GOVERNMENT_INTEREST}+{GOVERNMENT_REVENUE}"
    )
    operations = collect_government_finance(
        "QGFS",
        GOVERNMENT_FINANCE_KEYS[frequency].format(indicators=indicators),
        fetch_start,
        end_date,
        "IMF government finance statistics",
    )

    if operations.empty:
        return pd.DataFrame()

    interest = operations.xs(GOVERNMENT_INTEREST, axis=1, level="Indicator")
    interest.columns.name = None

    if denominator is None:
        return interest / 1_000_000

    if denominator == "revenue":
        divisor = operations.xs(GOVERNMENT_REVENUE, axis=1, level="Indicator")
    else:
        dataflow, key = GROSS_DOMESTIC_PRODUCT_KEYS[frequency]
        gross_domestic_product = collect_government_finance(
            dataflow, key, fetch_start, end_date, "IMF national accounts"
        )

        if gross_domestic_product.empty:
            return pd.DataFrame()

        # A GDP of zero or less is a placeholder (the QNEA states Uzbekistan's 2024
        # quarters as 0), so it is missing rather than a divisor.
        divisor = gross_domestic_product.droplevel("Indicator", axis=1)
        divisor = divisor.where(divisor > 0)

    divisor = divisor.reindex(index=interest.index, columns=interest.columns)

    if frequency == "Q":
        # A missing quarter leaves the trailing year empty rather than three quarters long.
        interest = interest.rolling(4).sum()
        divisor = divisor.rolling(4).sum()

    return interest / divisor


def get_expenditure_by_function(
    functions: list[str], start_date: str, end_date: str, to_gdp: bool = False
) -> pd.DataFrame:
    """
    Retrieves the yearly general government expenditure by function, classified by the United Nations' Classification
    of the Functions of Government (COFOG), for around 180 countries.

    Args:
        functions (list[str]): The functions, keys of EXPENDITURE_FUNCTIONS.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        to_gdp (bool): Whether to return the expenditure as a share of GDP (the IMF's own
            ratio, a decimal) instead of in millions of national currency. Defaults to False.

    Returns:
        pd.DataFrame: The expenditure, indexed by year with a (Country, Function) column in
        the order of the functions.
    """
    codes = {EXPENDITURE_FUNCTIONS[function]: function for function in functions}
    expenditure = collect_government_finance(
        "GFS_COFOG",
        f".S13.G2MF.{'+'.join(codes)}.{'POGDP_PT' if to_gdp else 'XDC'}.A",
        start_date,
        end_date,
        "IMF government expenditure by function",
    )

    if expenditure.empty:
        return pd.DataFrame()

    expenditure = expenditure.rename(columns=codes, level="Indicator")
    expenditure.columns = expenditure.columns.set_names(["Country", "Function"])
    countries = list(dict.fromkeys(expenditure.columns.get_level_values("Country")))

    expenditure = expenditure.reindex(
        columns=pd.MultiIndex.from_product(
            [countries, functions], names=["Country", "Function"]
        )
    ).dropna(axis=1, how="all")

    # The ratio is published in percent of GDP and the level in units of national currency.
    return expenditure / 100 if to_gdp else expenditure / 1_000_000


# The net international investment position of the IMF's International Investment Position
# statistics (BPM6): the external financial assets less the liabilities of all resident
# sectors at the end of each period. The IMF publishes it in US dollars only (the national
# currency series are empty), converted at the exchange rate at the end of the period. That
# rate, national currency per US dollar, comes from the IMF exchange rates dataset, where
# the euro members use the euro area's.
NET_INTERNATIONAL_INVESTMENT_POSITION_KEYS = {
    "Q": ".NETAL_P.NIIP.USD.Q",
    "Y": ".NETAL_P.NIIP.USD.A",
}
END_OF_PERIOD_EXCHANGE_RATE_KEYS = {
    "Q": ".XDC_USD.EOP_RT.Q",
    "Y": ".XDC_USD.EOP_RT.A",
}


def get_net_international_investment_position(
    period: str, start_date: str, end_date: str, to_gdp: bool = False
) -> pd.DataFrame:
    """
    Retrieves the net international investment position (external financial assets less liabilities) at the end of
    each quarter or year, for about 130 economies quarterly and 175 yearly, and the euro area.

    As a share of GDP, the position is converted to national currency at the IMF's
    end-of-period exchange rate and divided by the nominal GDP in national currency of the
    same year: the four quarters up to and including each quarter (not seasonally
    adjusted) or the calendar year. Both are then in one currency, so the ratio does not
    move with the gap between the end-of-period and the average exchange rate, the
    convention of Eurostat's macroeconomic imbalance procedure scoreboard.

    Args:
        period (str): The frequency of the data. Options are: "quarterly" (quarterly data), "yearly" (yearly data).
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).
        to_gdp (bool): Whether to return the position as a share of GDP (a decimal, -0.35
            for -35%) instead of in millions of US dollars. Defaults to False.

    Returns:
        pd.DataFrame: The position, indexed by quarter or year with a column per country.
    """
    frequency = PERIOD_CODES[period]
    # The year before the start date, for the four quarters of GDP of the first quarter.
    fetch_start = f"{int(start_date[:4]) - 1}{start_date[4:]}" if to_gdp else start_date
    position = collect_government_finance(
        "IIP",
        NET_INTERNATIONAL_INVESTMENT_POSITION_KEYS[frequency],
        fetch_start,
        end_date,
        "IMF international investment position",
    )

    if position.empty:
        return pd.DataFrame()

    position = position.droplevel("Indicator", axis=1)
    position.columns.name = None

    if not to_gdp:
        # Published in US dollars.
        return position / 1_000_000

    exchange_rate = collect_government_finance(
        "ER",
        END_OF_PERIOD_EXCHANGE_RATE_KEYS[frequency],
        fetch_start,
        end_date,
        "IMF exchange rates",
    )
    dataflow, key = GROSS_DOMESTIC_PRODUCT_KEYS[frequency]
    gross_domestic_product = collect_government_finance(
        dataflow, key, fetch_start, end_date, "IMF national accounts"
    )

    if exchange_rate.empty or gross_domestic_product.empty:
        return pd.DataFrame()

    exchange_rate = exchange_rate.droplevel("Indicator", axis=1)

    # The Marshall Islands use the US dollar as their currency, and the IMF publishes no
    # exchange rate for them (the other dollarized economies have a series of ones).
    if "Marshall Islands" not in exchange_rate.columns:
        exchange_rate["Marshall Islands"] = 1.0

    if "Euro Area" in exchange_rate.columns:
        # The IMF states the national accounts of a euro member in euro for every year,
        # also before it joined, while its exchange rate series, where there is one, ends
        # with its former currency (the litas for Lithuania until 2014).
        for member in EURO_AREA_MEMBERS:
            exchange_rate[member] = exchange_rate["Euro Area"]

    exchange_rate = exchange_rate.reindex(
        index=position.index, columns=position.columns
    )
    gross_domestic_product = gross_domestic_product.droplevel("Indicator", axis=1)
    # A GDP of zero or less is a placeholder (the QNEA states Uzbekistan's 2024 quarters
    # as 0), so it is missing and leaves the trailing year of its quarters empty.
    gross_domestic_product = gross_domestic_product.where(gross_domestic_product > 0)

    if frequency == "Q":
        # Summed over every quarter, also those without a position, and a missing quarter
        # leaves the trailing year empty rather than three quarters long.
        gross_domestic_product = (
            gross_domestic_product.reindex(
                pd.period_range(
                    gross_domestic_product.index.min(),
                    gross_domestic_product.index.max(),
                    freq="Q",
                )
            )
            .rolling(4)
            .sum()
        )

    gross_domestic_product = gross_domestic_product.reindex(
        index=position.index, columns=position.columns
    )

    return position * exchange_rate / gross_domestic_product
