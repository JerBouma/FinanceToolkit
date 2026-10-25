"""Discovery Model"""

__docformat__ = "google"

import re

import numpy as np
import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.cache.request_model import redact_credentials
from financetoolkit.economics.helpers import COUNTRY_CODES
from financetoolkit.fmp_model import get_financial_data
from financetoolkit.utilities import error_model, logger_model
from financetoolkit.utilities.statistics_model import normalize_country_name

logger = logger_model.get_logger()

# The sectors the Financial Modeling Prep sector endpoints classify companies into.
FMP_SECTORS = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]


def get_cached_financial_data(
    url: str,
    user_subscription: str = "Free",
    sleep_timer: bool = True,
) -> pd.DataFrame:
    """
    Retrieve a discovery endpoint, serving it from the cache when it is still fresh.

    The discovery endpoints return listings, screeners, calendars and news rather
    than data for a specific company, so there is no ticker to split them by and no
    date range to extend: the whole response is the unit. The request URL, with its
    credentials stripped, is what identifies it.

    Args:
        url (str): The full request URL, including the API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".
        sleep_timer (bool): Whether to wait and retry when the rate limit is reached.

    Returns:
        pd.DataFrame: The endpoint's response, either freshly retrieved or cached.
    """
    cache = get_active_cache()
    entity = redact_credentials(url)

    if cache is not None:
        cached_response = cache.get(
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset="discovery",
            entity=entity,
        )

        if cached_response is not None:
            return cached_response

    failures_before = error_model.get_request_failures()
    response = get_financial_data(
        url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
    )

    # An empty frame is a genuine result unless the request failed (a rate limit or a
    # plan restriction), in which case it is asked again next time.
    if (
        cache is not None
        and isinstance(response, pd.DataFrame)
        and (
            not response.empty
            or (
                error_model.get_request_failures() == failures_before
                and not error_model.is_error_response(response)
            )
        )
    ):
        cache.set(
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset="discovery",
            entity=entity,
            data=response,
        )

    return response


# The parameter each search of Financial Modeling Prep takes the query as.
SEARCH_PARAMETERS = {
    "name": "query",
    "symbol": "query",
    "cik": "cik",
    "cusip": "cusip",
    "isin": "isin",
}


def get_instruments(
    api_key: str,
    query: str,
    search_method: str = "name",
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get a list of instruments based on a query.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        query (str): The query to search for.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of instruments.
    """
    if search_method not in SEARCH_PARAMETERS:
        raise ValueError(
            f"The search method {search_method!r} is not valid. Choose from "
            f"{', '.join(SEARCH_PARAMETERS)}."
        )

    # The identifier searches take the identifier as their own parameter: given `query`,
    # FinancialModelingPrep ignores it and answers with Apple for every ISIN or CUSIP.
    url = (
        f"https://financialmodelingprep.com/stable/search-{search_method}"
        f"?{SEARCH_PARAMETERS[search_method]}={query}&apikey={api_key}"
    )

    instruments_query = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    error_model.check_for_error_messages(
        {"SEARCH_INSTRUMENTS": instruments_query}, user_subscription=user_subscription
    )

    if not instruments_query.empty:
        instruments_query = instruments_query.rename(
            columns={
                "symbol": "Symbol",
                "name": "Name",
                "companyName": "Name",
                "currency": "Currency",
                "exchangeFullName": "Exchange",
                "exchange": "Exchange Code",
                "cusip": "CUSIP",
                "cik": "CIK",
                "isin": "ISIN",
                "marketCap": "Market Cap",
            }
        )

        instruments_query = instruments_query.set_index("Symbol")

        if search_method in ("isin", "cusip"):
            instruments_query = _rank_listings_by_volume(
                instruments_query, api_key, user_subscription
            )

    return instruments_query


def _rank_listings_by_volume(
    listings: pd.DataFrame, api_key: str, user_subscription: str
) -> pd.DataFrame:
    """
    Puts the primary listing of an ISIN or CUSIP first. The search ranks the listings by
    their market capitalisation in their own currency, so a listing in Mexican pesos came
    before the one in US dollars; trading volume counts shares on every exchange alike,
    and the primary listing trades by far the most.

    Args:
        listings (pd.DataFrame): The listings, indexed by symbol.
        api_key (str): The API key from Financial Modeling Prep.
        user_subscription (str): The user subscription level.

    Returns:
        pd.DataFrame: The listings with their exchange and volume, most traded first.
    """
    quotes = get_cached_financial_data(
        url=(
            "https://financialmodelingprep.com/stable/batch-quote"
            f"?symbols={','.join(listings.index)}&apikey={api_key}"
        ),
        user_subscription=user_subscription,
    )

    if quotes.empty or not {"symbol", "volume"}.issubset(quotes.columns):
        return listings

    quotes = quotes.drop_duplicates("symbol").set_index("symbol")

    if "exchange" in quotes.columns and "Exchange Code" not in listings.columns:
        listings["Exchange Code"] = quotes["exchange"].reindex(listings.index)

    listings["Volume"] = quotes["volume"].reindex(listings.index)

    return listings.sort_values("Volume", ascending=False, na_position="last")


# The abbreviations in the names of preferred shares and hybrid securities, such as "Aegon
# N.V. PRP CP SEC 6.375", case-sensitive so that a company such as Preferred Bank is not.
HYBRID_SECURITY_PATTERN = re.compile(
    r"\b(?:PRP|PERP|PFD|PRF|CAP SECS?|CP SECS?|FLTG RT|DEP SHS?|SUB NTS?)\b"  # codespell:ignore shs
    r"|\s\d+\.\d+%?$|\d+\.\d+%"
)

# The company screener returns at most this many results.
MAXIMUM_SCREENER_LIMIT = 10000

# The share of the US dollar lower market cap bound sent to the screener, low enough that
# a listing in a currency worth up to twice a dollar is not left out before converting.
MARKET_CAP_BOUND_MARGIN = 0.5

# The currency of the exchanges of each country Financial Modeling Prep lists (UK, not GB).
# A helper column of the screener: whether a listing trades in the company's own country.
HOME_LISTING = "_home_listing"

COUNTRY_CURRENCIES = {
    "US": "USD",
    "CA": "CAD",
    "MX": "MXN",
    "BR": "BRL",
    "AR": "ARS",
    "CL": "CLP",
    "UK": "GBP",
    "GB": "GBP",
    "IE": "EUR",
    "NL": "EUR",
    "BE": "EUR",
    "FR": "EUR",
    "DE": "EUR",
    "AT": "EUR",
    "IT": "EUR",
    "ES": "EUR",
    "PT": "EUR",
    "FI": "EUR",
    "GR": "EUR",
    "CH": "CHF",
    "SE": "SEK",
    "NO": "NOK",
    "DK": "DKK",
    "IS": "ISK",
    "PL": "PLN",
    "CZ": "CZK",
    "TR": "TRY",
    "RU": "RUB",
    "IL": "ILS",
    "SA": "SAR",
    "AE": "AED",
    "QA": "QAR",
    "ZA": "ZAR",
    "IN": "INR",
    "CN": "CNY",
    "HK": "HKD",
    "TW": "TWD",
    "JP": "JPY",
    "KR": "KRW",
    "SG": "SGD",
    "MY": "MYR",
    "TH": "THB",
    "ID": "IDR",
    "AU": "AUD",
    "NZ": "NZD",
}


def _to_country_code(country: str | None) -> str | None:
    """
    The two-letter code the screener takes for a country written as a code or a name, such
    as "Netherlands", "netherlands" or "UK".

    Args:
        country (str | None): The country.

    Returns:
        str | None: The code, or the country as given when it is not known.
    """
    if country is None or len(country.strip()) == 2:  # noqa: PLR2004
        # The ISO code of the United Kingdom is GB, which is what the screener uses.
        code = country.strip().upper() if country else country
        return "GB" if code == "UK" else code

    name = normalize_country_name(country, COUNTRY_CODES.values())
    # The first code of a country wins: GB rather than UK for the United Kingdom.
    codes: dict[str, str] = {}
    for code, value in COUNTRY_CODES.items():
        codes.setdefault(value, code)

    return codes.get(str(name), country)


def get_stock_screener(
    api_key: str,
    market_cap_higher: int | None = None,
    market_cap_lower: int | None = None,
    price_higher: float | None = None,
    price_lower: float | None = None,
    beta_higher: float | None = None,
    beta_lower: float | None = None,
    volume_higher: int | None = None,
    volume_lower: int | None = None,
    dividend_higher: float | None = None,
    dividend_lower: float | None = None,
    sector: str | None = None,
    industry: str | None = None,
    country: str | None = None,
    exchange: str | None = None,
    is_etf: bool | None = None,
    limit: int = 1000,
    user_subscription: str = "Free",
    primary_only: bool = False,
) -> pd.DataFrame:
    """
    Get a list of instruments based on the screening criteria provided. It defaults
    to all stocks.

    The market capitalisations of the listings are in their own currency, so they are
    converted to US dollars, which the market cap bounds and the ranking use: a listing
    in Canadian dollars does not rank above the same company's listing in euros.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        market_cap_higher (int, optional): The market cap higher than. Defaults to None.
        market_cap_lower (int, optional): The market cap lower than. Defaults to None.
        price_higher (float, optional): The price higher than. Defaults to None.
        price_lower (float, optional): The price lower than. Defaults to None.
        beta_higher (float, optional): The beta higher than. Defaults to None.
        beta_lower (float, optional): The beta lower than. Defaults to None.
        volume_higher (int, optional): The volume higher than. Defaults to None.
        volume_lower (int, optional): The volume lower than. Defaults to None.
        dividend_higher (float, optional): The minimum last annual dividend per share, an amount
            in the listing's currency rather than a yield. Defaults to None.
        dividend_lower (float, optional): The maximum last annual dividend per share, an amount
            in the listing's currency rather than a yield. Defaults to None.
        sector (str, optional): The sector to restrict the screen to, e.g. "Energy".
        industry (str, optional): The industry to restrict the screen to, e.g. "Biotechnology".
        country (str, optional): The country to restrict the screen to, as a two-letter code
            ("NL") or a name ("Netherlands"). Defaults to None.
        exchange (str, optional): The exchange code to restrict the screen to, e.g. "NASDAQ".
        is_etf (bool, optional): True for ETFs only. Defaults to None, which screens stocks
            only (no ETFs or funds), as does False.
        limit (int, optional): The maximum number of results the API returns. Defaults to 1000,
            which is also the API's own default, so hitting exactly this number means the result
            was truncated.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".
        primary_only (bool, optional): Whether to keep only the home listing (on an exchange in
            the company's country, the most traded one among several) of each
            company, rather than every listing (e.g. ING in Amsterdam, New York, Toronto and
            Frankfurt). Defaults to False.

    Returns:
        pd.DataFrame: DataFrame of instruments matching the query, with each listing's currency
            and its market cap in US dollars, largest first. An empty DataFrame is returned when
            nothing matches.

    Notes:
        - Every bound is compared against None rather than truthiness, so a bound of exactly zero
          (a perfectly valid beta, dividend or volume floor) is sent to the API instead of dropped.
    """
    url = f"https://financialmodelingprep.com/stable/company-screener?apikey={api_key}"
    widened = (
        market_cap_higher is not None or market_cap_lower is not None or primary_only
    )
    # One listing per company needs room for the companies listed many times over (ASML
    # has six listings), whose foreign listings can also rank above the home one.
    requested_limit = (
        min(limit * (10 if primary_only else 4), MAXIMUM_SCREENER_LIMIT)
        if widened
        else limit
    )

    # The endpoint compares market caps in each listing's own currency, while the bounds
    # are in US dollars: the lower bound is widened for currencies worth more than a dollar
    # and the upper bound is applied after converting, so no listing is wrongly left out.
    # More results are asked for, since the converted bounds then narrow them down.
    query_parameters = {
        "marketCapMoreThan": (
            int(market_cap_higher * MARKET_CAP_BOUND_MARGIN)
            if market_cap_higher is not None
            else None
        ),
        "priceMoreThan": price_higher,
        "priceLowerThan": price_lower,
        "betaMoreThan": beta_higher,
        "betaLowerThan": beta_lower,
        "volumeMoreThan": volume_higher,
        "volumeLowerThan": volume_lower,
        "dividendMoreThan": dividend_higher,
        "dividendLowerThan": dividend_lower,
        "sector": sector,
        "industry": industry,
        "country": _to_country_code(country),
        "exchange": exchange,
        # Stocks unless ETFs are asked for: the endpoint otherwise mixes in ETFs and mutual
        # funds, which have no market capitalisation of their own.
        "isEtf": str(bool(is_etf)).lower(),
        "isFund": "false",
        "limit": requested_limit,
    }

    for parameter, value in query_parameters.items():
        if value is not None:
            url += f"&{parameter}={value}"

    stock_screener = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if stock_screener.empty:
        return stock_screener

    stock_screener = stock_screener.rename(
        columns={
            "symbol": "Symbol",
            "companyName": "Name",
            "marketCap": "Market Cap",
            "sector": "Sector",
            "industry": "Industry",
            "beta": "Beta",
            "price": "Price",
            "lastAnnualDividend": "Dividend",
            "volume": "Volume",
            "exchange": "Exchange",
            "exchangeShortName": "Exchange Code",
            "country": "Country",
            "currency": "Currency",
            "stockExchange": "Exchange",
        }
    )

    # These are internal flags of the endpoint rather than screening results, they are dropped defensively so that a new or removed flag does not raise.  # noqa: E501
    stock_screener = stock_screener.drop(
        columns=["isEtf", "isFund", "isActivelyTrading"], errors="ignore"
    )
    stock_screener = stock_screener.set_index("Symbol")
    truncated = len(stock_screener) >= requested_limit

    # A preferred share or hybrid security has no market capitalisation of its own, while
    # the screener reports one in the tens of billions for such a listing.
    if {"Name", "Market Cap"}.issubset(stock_screener.columns):
        hybrids = (
            stock_screener["Name"].astype(str).str.contains(HYBRID_SECURITY_PATTERN)
        )
        stock_screener.loc[hybrids, "Market Cap"] = np.nan

    stock_screener = _add_market_cap_in_dollars(
        stock_screener, api_key, user_subscription
    )
    dollar_market_cap = (
        "Market Cap (USD)"
        if "Market Cap (USD)" in stock_screener.columns
        else "Market Cap"
    )

    if market_cap_higher is not None:
        stock_screener = stock_screener[
            stock_screener[dollar_market_cap] >= market_cap_higher
        ]
    if market_cap_lower is not None:
        stock_screener = stock_screener[
            stock_screener[dollar_market_cap] <= market_cap_lower
        ]

    stock_screener = stock_screener.sort_values(
        dollar_market_cap, ascending=False, na_position="last"
    )

    if primary_only and "Name" in stock_screener.columns:
        # The primary listing of a company is the one in its own country, and among those
        # (or without one) the most traded; volume alone favours depositary receipts that
        # trade in fractions of a share, such as CEDEARs.
        order = [
            column
            for column in (HOME_LISTING, "Volume")
            if column in stock_screener.columns
        ]
        most_traded = (
            stock_screener.sort_values(order, ascending=False, kind="stable")
            if order
            else stock_screener
        )
        primary_listings = most_traded[~most_traded["Name"].duplicated()].index
        stock_screener = stock_screener[stock_screener.index.isin(primary_listings)]

    stock_screener = stock_screener.drop(columns=HOME_LISTING, errors="ignore").head(
        limit
    )

    if truncated:
        logger.warning(
            "The screener returned the maximum of %s results, so the list is truncated. "
            "Narrow the criteria or raise the limit to see the remainder.",
            requested_limit,
        )

    return stock_screener


def _add_market_cap_in_dollars(
    listings: pd.DataFrame, api_key: str, user_subscription: str
) -> pd.DataFrame:
    """
    Adds each listing's currency and its market cap in US dollars. The currency follows
    from the country of the listing's exchange, and the exchange rates are the latest
    quotes. Without them, the listings are returned as they are, with a warning.

    Args:
        listings (pd.DataFrame): The listings, with an "Exchange Code" and "Market Cap".
        api_key (str): The API key from Financial Modeling Prep.
        user_subscription (str): The user subscription level.

    Returns:
        pd.DataFrame: The listings with a "Currency" and a "Market Cap (USD)" column.
    """
    if not {"Exchange Code", "Market Cap"}.issubset(listings.columns):
        return listings

    exchanges = get_cached_financial_data(
        url=f"https://financialmodelingprep.com/stable/available-exchanges?apikey={api_key}",
        user_subscription=user_subscription,
    )

    if exchanges.empty or not {"exchange", "countryCode"}.issubset(exchanges.columns):
        logger.warning(
            "The currencies of the exchanges could not be retrieved, so market caps are "
            "in each listing's own currency."
        )
        return listings

    exchange_currencies = {
        exchange: COUNTRY_CURRENCIES.get(country_code)
        for exchange, country_code in zip(
            exchanges["exchange"], exchanges["countryCode"], strict=False
        )
    }
    listings["Currency"] = listings["Exchange Code"].map(exchange_currencies)
    # Whether a listing trades in the company's own country, which marks its home listing.
    exchange_countries = dict(
        zip(exchanges["exchange"], exchanges["countryCode"], strict=False)
    )
    if "Country" in listings.columns:
        listings[HOME_LISTING] = listings["Exchange Code"].map(
            exchange_countries
        ) == listings["Country"].astype(str).str.upper().replace({"UK": "GB"})

    foreign_currencies = sorted(
        {currency for currency in listings["Currency"].dropna() if currency != "USD"}
    )
    rates = {"USD": 1.0}

    if foreign_currencies:
        quotes = get_cached_financial_data(
            url=(
                "https://financialmodelingprep.com/stable/batch-quote?symbols="
                f"{','.join(f'{currency}USD' for currency in foreign_currencies)}"
                f"&apikey={api_key}"
            ),
            user_subscription=user_subscription,
        )

        if not quotes.empty and {"symbol", "price"}.issubset(quotes.columns):
            rates.update(
                {
                    symbol.removesuffix("USD"): price
                    for symbol, price in zip(
                        quotes["symbol"], quotes["price"], strict=False
                    )
                }
            )

    listings["Market Cap (USD)"] = listings["Market Cap"] * listings["Currency"].map(
        rates
    )

    # A market cap belongs to the company, not to one of its listings: a depositary receipt
    # abroad (such as a CEDEAR in Buenos Aires) can report one many times too large, so every
    # listing of a company takes the one of its home listing.
    if HOME_LISTING in listings.columns and "Name" in listings.columns:
        home_caps = (
            listings[listings[HOME_LISTING] & listings["Market Cap (USD)"].notna()]
            .groupby("Name")["Market Cap (USD)"]
            .first()
        )
        listings["Market Cap (USD)"] = (
            listings["Name"].map(home_caps).fillna(listings["Market Cap (USD)"])
        )

    unconverted = listings["Market Cap"].notna() & listings["Market Cap (USD)"].isna()

    if missing := sorted(set(listings.loc[unconverted, "Exchange Code"].dropna())):
        logger.warning(
            "The market caps of the listings on %s could not be converted to US dollars.",
            ", ".join(missing),
        )

    return listings


def get_stock_list(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get a list of stocks.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of stocks.
    """
    url = f"https://financialmodelingprep.com/stable/stock-list?apikey={api_key}"

    stock_list = get_cached_financial_data(url=url, user_subscription=user_subscription)

    if stock_list.empty:
        return stock_list

    stock_list = stock_list.rename(
        columns={
            "symbol": "Symbol",
            "companyName": "Name",
        }
    )

    stock_list = stock_list.set_index("Symbol").sort_index()

    return stock_list


def get_stock_shares_float(
    api_key: str, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the shares float for all stocks.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of shares float.
    """
    url = f"https://financialmodelingprep.com/api/v4/shares_float/all?apikey={api_key}"

    stock_shares_float = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if stock_shares_float.empty:
        return stock_shares_float

    stock_shares_float = stock_shares_float.rename(
        columns={
            "symbol": "Symbol",
            "date": "Date",
            "freeFloat": "Free Float",
            "floatShares": "Float Shares",
            "outstandingShares": "Outstanding Shares",
        }
    )

    # The free float is published in percent and returned as a decimal (0.85 is 85%), the
    # same as the per-company shares float.
    if "Free Float" in stock_shares_float.columns:
        stock_shares_float["Free Float"] = (
            pd.to_numeric(stock_shares_float["Free Float"], errors="coerce") / 100
        )

    stock_shares_float = stock_shares_float.set_index("Symbol").sort_index()

    return stock_shares_float


def get_sectors_performance(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
    exchange: str = "NASDAQ",
) -> pd.DataFrame:
    """
    Get the historical performance of every sector, one column per sector.

    The wide layout is assembled from the per-sector historical endpoint, which is queried once
    for each of the sectors Financial Modeling Prep classifies companies into.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with, e.g. "2024-01-01".
            Defaults to a year ago.
        end_date (str, optional): The end date to filter data with, e.g. "2024-12-31".
            Defaults to today.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".
        exchange (str, optional): The exchange whose companies are averaged, "NASDAQ", "NYSE"
            or "AMEX". Defaults to "NASDAQ".

    Returns:
        pd.DataFrame: DataFrame of sector performance indexed by date, with one column per sector
            holding the average change of that sector on that date, as a decimal (0.0125 is
            +1.25%).
    """
    sector_changes = {}

    for sector in FMP_SECTORS:
        # A year up to today unless a range is given (see _history_range).
        url = (
            "https://financialmodelingprep.com/stable/historical-sector-performance"
            f"?sector={sector}{_history_range(start_date, end_date)}&exchange={exchange}"
            f"&apikey={api_key}"
        )

        sector_performance = get_cached_financial_data(
            url=url, user_subscription=user_subscription
        )

        if sector_performance.empty:
            continue

        sector_performance = sector_performance.rename(
            columns={"date": "Date", "averageChange": sector}
        )
        sector_changes[sector] = sector_performance.set_index("Date")[sector]

    if not sector_changes:
        return pd.DataFrame()

    # The changes are published in percent and returned as decimals, like every return.
    sectors_performance = (
        pd.concat(sector_changes, axis=1).apply(pd.to_numeric, errors="coerce") / 100
    )

    sectors_performance.index = pd.PeriodIndex(sectors_performance.index, freq="D")

    sectors_performance = sectors_performance.sort_index()

    return sectors_performance


def get_biggest_gainers(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get the biggest gainers.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of biggest gainers.
    """
    url = f"https://financialmodelingprep.com/stable/biggest-gainers?apikey={api_key}"

    biggest_gainers = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if biggest_gainers.empty:
        return biggest_gainers

    biggest_gainers = biggest_gainers.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "change": "Change",
            "price": "Price",
            "changesPercentage": "Change %",
            "exchange": "Exchange",
        }
    )

    # The change is published in percent and returned as a decimal (0.0125 is +1.25%).
    if "Change %" in biggest_gainers.columns:
        biggest_gainers["Change %"] = (
            pd.to_numeric(biggest_gainers["Change %"], errors="coerce") / 100
        )

    # The largest rise first, so the head of the list is what its name promises.
    biggest_gainers = biggest_gainers.set_index("Symbol").sort_values(
        "Change %", ascending=False, kind="stable"
    )

    return biggest_gainers


def get_biggest_losers(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get the biggest losers.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of biggest losers.
    """
    url = f"https://financialmodelingprep.com/stable/biggest-losers?apikey={api_key}"

    biggest_losers = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if biggest_losers.empty:
        return biggest_losers

    biggest_losers = biggest_losers.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "change": "Change",
            "price": "Price",
            "changesPercentage": "Change %",
            "exchange": "Exchange",
        }
    )

    # The change is published in percent and returned as a decimal (0.0125 is +1.25%).
    if "Change %" in biggest_losers.columns:
        biggest_losers["Change %"] = (
            pd.to_numeric(biggest_losers["Change %"], errors="coerce") / 100
        )

    # The largest fall first, so the head of the list is what its name promises.
    biggest_losers = biggest_losers.set_index("Symbol").sort_values(
        "Change %", ascending=True, kind="stable"
    )

    return biggest_losers


def get_most_active_stocks(
    api_key: str, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the most active stocks.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of most active stocks.
    """
    url = f"https://financialmodelingprep.com/stable/most-actives?apikey={api_key}"

    most_active = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if most_active.empty:
        return most_active

    most_active = most_active.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "change": "Change",
            "price": "Price",
            "changesPercentage": "Change %",
            "exchange": "Exchange",
        }
    )

    # The change is published in percent and returned as a decimal (0.0125 is +1.25%).
    if "Change %" in most_active.columns:
        most_active["Change %"] = (
            pd.to_numeric(most_active["Change %"], errors="coerce") / 100
        )

    # Kept in the source's order, the most traded first, as there is no volume to sort on.
    most_active = most_active.set_index("Symbol")

    return most_active


def get_crypto_list(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get a list of cryptocurrencies.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of cryptocurrencies.
    """
    url = (
        f"https://financialmodelingprep.com/stable/cryptocurrency-list?apikey={api_key}"
    )

    crypto_list = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if crypto_list.empty:
        return crypto_list

    crypto_list = crypto_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            # The endpoint's "exchange" field holds an exchange code such as CCC, not a currency, so labelling the column Currency mislabelled the data.  # noqa: E501
            "exchange": "Exchange",
            "icoDate": "ICO Date",
            "circulatingSupply": "Circulating Supply",
            "totalSupply": "Total Supply",
        }
    )

    crypto_list = crypto_list.set_index("Symbol").sort_index()

    return crypto_list


def get_delisted_stocks(
    api_key: str, page: int = 0, limit: int = 100, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get a list of delisted companies.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        page (int, optional): The page of results to retrieve, starting at 0. Defaults to 0.
        limit (int, optional): The number of results per page, capped at 100 by the API.
            Defaults to 100.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of delisted companies, one page at a time. The endpoint returns
            at most 100 rows per call, so a full list requires walking `page` upwards until an
            empty frame comes back.
    """
    url = (
        "https://financialmodelingprep.com/stable/delisted-companies"
        f"?page={page}&limit={limit}&apikey={api_key}"
    )

    delisted_companies = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if delisted_companies.empty:
        return delisted_companies

    delisted_companies = delisted_companies.rename(
        columns={
            "symbol": "Symbol",
            "companyName": "Name",
            "exchange": "Exchange",
            "ipoDate": "IPO Date",
            "delistedDate": "Delisted Date",
        }
    )

    delisted_companies = delisted_companies.set_index("Symbol").sort_index()

    return delisted_companies


def get_forex_list(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get a list of forex pairs.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of forex pairs.
    """
    # The "symbol/available-forex-currency-pairs" path was retired and answers with an empty list, which then raised a KeyError on the missing columns further down.  # noqa: E501
    url = f"https://financialmodelingprep.com/stable/forex-list?apikey={api_key}"

    forex_list = get_cached_financial_data(url=url, user_subscription=user_subscription)

    if forex_list.empty:
        return forex_list

    forex_list = forex_list.rename(
        columns={
            "symbol": "Symbol",
            "fromCurrency": "From Currency",
            "toCurrency": "To Currency",
            "fromName": "From Name",
            "toName": "To Name",
        }
    )

    forex_list = forex_list.set_index("Symbol").sort_index()

    return forex_list


def get_commodity_list(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get a list of commodities.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of commodities.
    """
    # The "symbol/available-commodities" path was retired and answers with an empty list, which then raised a KeyError on the missing columns further down.  # noqa: E501
    url = f"https://financialmodelingprep.com/stable/commodities-list?apikey={api_key}"

    commody_list = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if commody_list.empty:
        return commody_list

    commody_list = commody_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "currency": "Currency",
            "exchange": "Exchange",
            "tradeMonth": "Trade Month",
        }
    )

    commody_list = commody_list.set_index("Symbol").sort_index()

    return commody_list


def get_etf_list(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get a list of ETFs.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of ETFs.
    """
    # The "etf/list" path was retired and answers with an empty list, which then raised a KeyError on the missing columns further down.  # noqa: E501
    url = f"https://financialmodelingprep.com/stable/etf-list?apikey={api_key}"

    etf_list = get_cached_financial_data(url=url, user_subscription=user_subscription)

    if etf_list.empty:
        return etf_list

    etf_list = etf_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
        }
    )

    etf_list = etf_list.set_index("Symbol").sort_index()

    return etf_list


# The indices whose constituents Financial Modeling Prep publishes, with their endpoint.
INDEX_CONSTITUENTS = {
    "S&P 500": "sp500-constituent",
    "NASDAQ 100": "nasdaq-constituent",
    "Dow Jones": "dowjones-constituent",
}


def get_index_constituents(
    api_key: str, index: str = "S&P 500", user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the current constituents of an index. Past figures computed from today's members
    leave out the companies removed from the index since (survivorship bias).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        index (str, optional): "S&P 500", "NASDAQ 100" or "Dow Jones". Defaults to "S&P 500".
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: The constituents, indexed by symbol, with their name, sector, sub-sector,
        headquarters, the date they were added, CIK and founding year.
    """
    if index not in INDEX_CONSTITUENTS:
        raise ValueError(
            f"The constituents of {index!r} are not available. Choose from "
            f"{', '.join(INDEX_CONSTITUENTS)}."
        )

    url = (
        f"https://financialmodelingprep.com/stable/{INDEX_CONSTITUENTS[index]}"
        f"?apikey={api_key}"
    )
    constituents = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if constituents.empty:
        return constituents

    constituents = constituents.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "sector": "Sector",
            "subSector": "Sub-Sector",
            "headQuarter": "Headquarters",
            "dateFirstAdded": "Date Added",
            "cik": "CIK",
            "founded": "Founded",
        }
    )

    return constituents.set_index("Symbol").sort_index()


def get_index_list(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get a list of indexes.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of indexes.
    """
    # The "symbol/available-indexes" path was retired and answers with an empty list, which then raised a KeyError on the missing columns further down.  # noqa: E501
    url = f"https://financialmodelingprep.com/stable/index-list?apikey={api_key}"

    index_list = get_cached_financial_data(url=url, user_subscription=user_subscription)

    if index_list.empty:
        return index_list

    index_list = index_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "currency": "Currency",
            "exchange": "Exchange",
        }
    )

    index_list = index_list.set_index("Symbol").sort_index()

    return index_list


def _format_news(news_data: pd.DataFrame) -> pd.DataFrame:
    """
    Shared column renaming and indexing logic for all news-related endpoints.

    Args:
        news_data (pd.DataFrame): the raw news DataFrame as returned by the API.

    Returns:
        pd.DataFrame: DataFrame of news articles, indexed by Published Date (most recent first).
    """
    if news_data.empty:
        return news_data

    news_data = news_data.rename(
        columns={
            "symbol": "Symbol",
            "publishedDate": "Published Date",
            "publisher": "Publisher",
            "title": "Title",
            "image": "Image",
            "site": "Site",
            "text": "Text",
            "url": "URL",
        }
    )

    news_data["Published Date"] = pd.to_datetime(news_data["Published Date"])
    news_data = news_data.set_index("Published Date").sort_index(ascending=False)

    return news_data


def _normalize_symbols(symbols: str | list[str]) -> str:
    """
    Normalizes a single symbol, comma-separated string, or list of symbols into
    the comma-separated string format the FMP API expects.

    Args:
        symbols (str | list[str]): a single symbol, comma-separated string of
            symbols, or a list of symbols.

    Returns:
        str: a comma-separated string of symbols.
    """
    if isinstance(symbols, list):
        return ",".join(symbols)

    return symbols


def _get_news_pages(
    base_url: str, pages: int, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Fetches and concatenates one or more pages of a paginated news endpoint. Each
    page is a separate API call, e.g. pages=5 makes 5 calls (page 0 through 4).

    Args:
        base_url (str): the endpoint url without the "page" query parameter.
        pages (int): the number of pages to collect.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: the concatenated, unformatted news DataFrame across all pages.
    """
    page_frames = []

    for page in range(pages):
        page_data = get_cached_financial_data(
            url=f"{base_url}&page={page}", user_subscription=user_subscription
        )
        if not page_data.empty:
            page_frames.append(page_data)

    if not page_frames:
        return pd.DataFrame()

    return pd.concat(page_frames, ignore_index=True)


def get_stock_news(
    api_key: str,
    limit: int = 100,
    pages: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the latest stock market news articles.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the latest stock market news articles.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/stock-latest"
        f"?limit={limit}&apikey={api_key}"
    )
    if start_date:
        base_url += f"&from={start_date}"
    if end_date:
        base_url += f"&to={end_date}"

    stock_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(stock_news)


def get_general_news(
    api_key: str, limit: int = 100, pages: int = 1, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the latest general news articles.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the latest general news articles.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/general-latest"
        f"?limit={limit}&apikey={api_key}"
    )

    general_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(general_news)


def get_press_releases(
    api_key: str,
    limit: int = 100,
    pages: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the latest company press releases.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the latest company press releases.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/press-releases-latest"
        f"?limit={limit}&apikey={api_key}"
    )
    if start_date:
        base_url += f"&from={start_date}"
    if end_date:
        base_url += f"&to={end_date}"

    press_releases = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(press_releases)


def get_crypto_news(
    api_key: str, limit: int = 100, pages: int = 1, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the latest cryptocurrency news articles.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the latest cryptocurrency news articles.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/crypto-latest"
        f"?limit={limit}&apikey={api_key}"
    )

    crypto_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(crypto_news)


def get_forex_news(
    api_key: str, limit: int = 100, pages: int = 1, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the latest forex news articles.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the latest forex news articles.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/forex-latest"
        f"?limit={limit}&apikey={api_key}"
    )

    forex_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(forex_news)


def search_stock_news(
    api_key: str,
    symbols: str | list[str],
    limit: int = 100,
    pages: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Search stock market news articles by ticker symbol.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        symbols (str | list[str]): One or more ticker symbols, e.g. "AAPL" or
            ["AAPL", "MSFT"].
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of stock news articles matching the given symbols.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/stock"
        f"?symbols={_normalize_symbols(symbols)}&limit={limit}&apikey={api_key}"
    )
    if start_date:
        base_url += f"&from={start_date}"
    if end_date:
        base_url += f"&to={end_date}"

    stock_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(stock_news)


def search_press_releases(
    api_key: str,
    symbols: str | list[str],
    limit: int = 100,
    pages: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Search company press releases by ticker symbol.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        symbols (str | list[str]): One or more ticker symbols, e.g. "AAPL" or
            ["AAPL", "MSFT"].
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of press releases matching the given symbols.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/press-releases"
        f"?symbols={_normalize_symbols(symbols)}&limit={limit}&apikey={api_key}"
    )
    if start_date:
        base_url += f"&from={start_date}"
    if end_date:
        base_url += f"&to={end_date}"

    press_releases = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(press_releases)


def search_crypto_news(
    api_key: str,
    symbols: str | list[str],
    limit: int = 100,
    pages: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Search cryptocurrency news articles by symbol.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        symbols (str | list[str]): One or more crypto symbols, e.g. "BTCUSD" or
            ["BTCUSD", "ETHUSD"].
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of crypto news articles matching the given symbols.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/crypto"
        f"?symbols={_normalize_symbols(symbols)}&limit={limit}&apikey={api_key}"
    )

    if start_date:
        base_url += f"&from={start_date}"
    if end_date:
        base_url += f"&to={end_date}"

    crypto_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(crypto_news)


def search_forex_news(
    api_key: str,
    symbols: str | list[str],
    limit: int = 100,
    pages: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Search forex news articles by currency pair symbol.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        symbols (str | list[str]): One or more forex pairs, e.g. "EURUSD" or
            ["EURUSD", "GBPUSD"].
        limit (int, optional): The number of articles to return per page. Defaults to 100.
        pages (int, optional): The number of pages to collect, each page is a separate
            API call, e.g. pages=5 makes 5 calls. Defaults to 1.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of forex news articles matching the given symbols.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/forex"
        f"?symbols={_normalize_symbols(symbols)}&limit={limit}&apikey={api_key}"
    )

    if start_date:
        base_url += f"&from={start_date}"
    if end_date:
        base_url += f"&to={end_date}"

    forex_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(forex_news)


def get_ipo_calendar(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the calendar of upcoming and recent initial public offerings (IPOs).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with (max 90-day range).
        end_date (str, optional): The end date to filter data with (max 90-day range).
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of upcoming and recent IPOs.
    """
    url = f"https://financialmodelingprep.com/stable/ipos-calendar?apikey={api_key}"

    if start_date:
        url += f"&from={start_date}"
    if end_date:
        url += f"&to={end_date}"

    ipo_calendar = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if ipo_calendar.empty:
        return ipo_calendar

    ipo_calendar = ipo_calendar.rename(
        columns={
            "symbol": "Symbol",
            "date": "Date",
            "daa": "Date (ISO)",
            "company": "Company",
            "exchange": "Exchange",
            "actions": "Status",
            "shares": "Shares",
            "priceRange": "Price Range",
            "marketCap": "Market Cap",
            "dealSize": "Deal Size",
        }
    )

    # In date order, as a calendar is read.
    ipo_calendar = ipo_calendar.sort_values("Date", kind="stable").set_index("Symbol")

    return ipo_calendar


def get_ipo_disclosures(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get IPO disclosure filings (regulatory filings ahead of an IPO).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of IPO disclosure filings.
    """
    url = f"https://financialmodelingprep.com/stable/ipos-disclosure?apikey={api_key}"

    if start_date:
        url += f"&from={start_date}"
    if end_date:
        url += f"&to={end_date}"

    ipo_disclosures = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if ipo_disclosures.empty:
        return ipo_disclosures

    ipo_disclosures = ipo_disclosures.rename(
        columns={
            "symbol": "Symbol",
            "filingDate": "Filing Date",
            "acceptedDate": "Accepted Date",
            "effectivenessDate": "Effectiveness Date",
            "cik": "CIK",
            "form": "Form",
            "url": "URL",
        }
    )

    ipo_disclosures = ipo_disclosures.set_index("Symbol").sort_index()

    return ipo_disclosures


def get_ipo_prospectuses(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get IPO prospectus filings, including public offering pricing details.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of IPO prospectus filings.
    """
    url = f"https://financialmodelingprep.com/stable/ipos-prospectus?apikey={api_key}"

    if start_date:
        url += f"&from={start_date}"
    if end_date:
        url += f"&to={end_date}"

    ipo_prospectuses = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if ipo_prospectuses.empty:
        return ipo_prospectuses

    ipo_prospectuses = ipo_prospectuses.rename(
        columns={
            "symbol": "Symbol",
            "acceptedDate": "Accepted Date",
            "filingDate": "Filing Date",
            "ipoDate": "IPO Date",
            "cik": "CIK",
            "pricePublicPerShare": "Public Price Per Share",
            "pricePublicTotal": "Public Price Total",
            "discountsAndCommissionsPerShare": "Discounts and Commissions Per Share",
            "discountsAndCommissionsTotal": "Discounts and Commissions Total",
            "proceedsBeforeExpensesPerShare": "Proceeds Before Expenses Per Share",
            "proceedsBeforeExpensesTotal": "Proceeds Before Expenses Total",
            "form": "Form",
            "url": "URL",
        }
    )

    ipo_prospectuses = ipo_prospectuses.set_index("Symbol").sort_index()

    return ipo_prospectuses


def get_stock_splits_calendar(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the calendar of upcoming and recent stock splits across all companies.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with (max 90-day range).
        end_date (str, optional): The end date to filter data with (max 90-day range).
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of upcoming and recent stock splits.
    """
    url = f"https://financialmodelingprep.com/stable/splits-calendar?apikey={api_key}"

    if start_date:
        url += f"&from={start_date}"
    if end_date:
        url += f"&to={end_date}"

    splits_calendar = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if splits_calendar.empty:
        return splits_calendar

    splits_calendar = splits_calendar.rename(
        columns={
            "symbol": "Symbol",
            "date": "Date",
            "numerator": "Numerator",
            "denominator": "Denominator",
            "splitType": "Split Type",
        }
    )

    # In date order, as a calendar is read.
    splits_calendar = splits_calendar.sort_values("Date", kind="stable").set_index(
        "Symbol"
    )

    return splits_calendar


def _history_range(start_date: str | None, end_date: str | None) -> str:
    """
    The from and to parameters of a sector or industry history: without them the
    endpoints answer with February 2024 only, so a year up to today is the default.

    Args:
        start_date (str | None): The start date, a year ago when None.
        end_date (str | None): The end date, today when None.

    Returns:
        str: The query string part, e.g. "&from=2025-10-11&to=2026-10-11".
    """
    today = pd.Timestamp.today().normalize()
    start = pd.Timestamp(start_date) if start_date else today - pd.DateOffset(years=1)
    end = pd.Timestamp(end_date) if end_date else today

    return f"&from={start:%Y-%m-%d}&to={end:%Y-%m-%d}"


def get_sector_performance(
    api_key: str,
    date: str | None = None,
    sector: str | None = None,
    user_subscription: str = "Free",
    start_date: str | None = None,
    end_date: str | None = None,
    exchange: str = "NASDAQ",
) -> pd.DataFrame:
    """
    Get sector performance. Provide exactly one of `date` (a snapshot across all
    sectors on that date) or `sector` (the historical time series for one sector).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        date (str, optional): The date to retrieve a snapshot for, e.g. "2024-02-01".
        sector (str, optional): The sector to retrieve the history for, e.g. "Energy".
        user_subscription (str, optional): The user subscription level. Defaults to "Free".
        start_date (str, optional): The start of the history of one sector. Defaults to a year ago.
        end_date (str, optional): The end of the history of one sector. Defaults to today.
        exchange (str, optional): The exchange whose companies are averaged, "NASDAQ", "NYSE"
            or "AMEX". Defaults to "NASDAQ".

    Returns:
        pd.DataFrame: DataFrame of sector performance, indexed by Sector (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(sector):
        raise ValueError("Please provide exactly one of `date` or `sector`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/sector-performance-snapshot"
            f"?date={date}&exchange={exchange}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-sector-performance"
            f"?sector={sector}{_history_range(start_date, end_date)}&exchange={exchange}"
            f"&apikey={api_key}"
        )

    sector_performance = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if sector_performance.empty:
        return sector_performance

    sector_performance = sector_performance.rename(
        columns={
            "date": "Date",
            "sector": "Sector",
            "exchange": "Exchange",
            "averageChange": "Average Change",
        }
    )
    # The change is published in percent and returned as a decimal, like every return.
    sector_performance["Average Change"] = (
        pd.to_numeric(sector_performance["Average Change"], errors="coerce") / 100
    )

    if date:
        sector_performance = sector_performance.set_index("Sector").sort_index()
    else:
        sector_performance["Date"] = pd.to_datetime(sector_performance["Date"])
        sector_performance = sector_performance.set_index("Date").sort_index()

    return sector_performance


def get_industry_performance(
    api_key: str,
    date: str | None = None,
    industry: str | None = None,
    user_subscription: str = "Free",
    start_date: str | None = None,
    end_date: str | None = None,
    exchange: str = "NASDAQ",
) -> pd.DataFrame:
    """
    Get industry performance. Provide exactly one of `date` (a snapshot across all
    industries on that date) or `industry` (the historical time series for one
    industry).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        date (str, optional): The date to retrieve a snapshot for, e.g. "2024-02-01".
        industry (str, optional): The industry to retrieve the history for, e.g. "Biotechnology".
        user_subscription (str, optional): The user subscription level. Defaults to "Free".
        start_date (str, optional): The start of the history of one industry. Defaults to a year ago.
        end_date (str, optional): The end of the history of one industry. Defaults to today.
        exchange (str, optional): The exchange whose companies are averaged, "NASDAQ", "NYSE"
            or "AMEX". Defaults to "NASDAQ".

    Returns:
        pd.DataFrame: DataFrame of industry performance, indexed by Industry (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(industry):
        raise ValueError("Please provide exactly one of `date` or `industry`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/industry-performance-snapshot"
            f"?date={date}&exchange={exchange}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-industry-performance"
            f"?industry={industry}{_history_range(start_date, end_date)}&exchange={exchange}"
            f"&apikey={api_key}"
        )

    industry_performance = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if industry_performance.empty:
        return industry_performance

    industry_performance = industry_performance.rename(
        columns={
            "date": "Date",
            "industry": "Industry",
            "exchange": "Exchange",
            "averageChange": "Average Change",
        }
    )
    # The change is published in percent and returned as a decimal, like every return.
    industry_performance["Average Change"] = (
        pd.to_numeric(industry_performance["Average Change"], errors="coerce") / 100
    )

    if date:
        industry_performance = industry_performance.set_index("Industry").sort_index()
    else:
        industry_performance["Date"] = pd.to_datetime(industry_performance["Date"])
        industry_performance = industry_performance.set_index("Date").sort_index()

    return industry_performance


def get_sector_pe(
    api_key: str,
    date: str | None = None,
    sector: str | None = None,
    user_subscription: str = "Free",
    start_date: str | None = None,
    end_date: str | None = None,
    exchange: str = "NASDAQ",
) -> pd.DataFrame:
    """
    Get sector price-to-earnings (P/E) ratios. Provide exactly one of `date` (a
    snapshot across all sectors on that date) or `sector` (the historical time
    series for one sector).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        date (str, optional): The date to retrieve a snapshot for, e.g. "2024-02-01".
        sector (str, optional): The sector to retrieve the history for, e.g. "Energy".
        user_subscription (str, optional): The user subscription level. Defaults to "Free".
        start_date (str, optional): The start of the history of one sector. Defaults to a year ago.
        end_date (str, optional): The end of the history of one sector. Defaults to today.
        exchange (str, optional): The exchange whose companies are averaged, "NASDAQ", "NYSE"
            or "AMEX". Defaults to "NASDAQ".

    Returns:
        pd.DataFrame: DataFrame of sector P/E ratios, indexed by Sector (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(sector):
        raise ValueError("Please provide exactly one of `date` or `sector`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/sector-pe-snapshot"
            f"?date={date}&exchange={exchange}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-sector-pe"
            f"?sector={sector}{_history_range(start_date, end_date)}&exchange={exchange}"
            f"&apikey={api_key}"
        )

    sector_pe = get_cached_financial_data(url=url, user_subscription=user_subscription)

    if sector_pe.empty:
        return sector_pe

    sector_pe = sector_pe.rename(
        columns={
            "date": "Date",
            "sector": "Sector",
            "exchange": "Exchange",
            "pe": "PE Ratio",
        }
    )

    if date:
        sector_pe = sector_pe.set_index("Sector").sort_index()
    else:
        sector_pe["Date"] = pd.to_datetime(sector_pe["Date"])
        sector_pe = sector_pe.set_index("Date").sort_index()

    return sector_pe


def get_industry_pe(
    api_key: str,
    date: str | None = None,
    industry: str | None = None,
    user_subscription: str = "Free",
    start_date: str | None = None,
    end_date: str | None = None,
    exchange: str = "NASDAQ",
) -> pd.DataFrame:
    """
    Get industry price-to-earnings (P/E) ratios. Provide exactly one of `date` (a
    snapshot across all industries on that date) or `industry` (the historical time
    series for one industry).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        date (str, optional): The date to retrieve a snapshot for, e.g. "2024-02-01".
        industry (str, optional): The industry to retrieve the history for, e.g. "Biotechnology".
        user_subscription (str, optional): The user subscription level. Defaults to "Free".
        start_date (str, optional): The start of the history of one industry. Defaults to a year ago.
        end_date (str, optional): The end of the history of one industry. Defaults to today.
        exchange (str, optional): The exchange whose companies are averaged, "NASDAQ", "NYSE"
            or "AMEX". Defaults to "NASDAQ".

    Returns:
        pd.DataFrame: DataFrame of industry P/E ratios, indexed by Industry (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(industry):
        raise ValueError("Please provide exactly one of `date` or `industry`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/industry-pe-snapshot"
            f"?date={date}&exchange={exchange}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-industry-pe"
            f"?industry={industry}{_history_range(start_date, end_date)}&exchange={exchange}"
            f"&apikey={api_key}"
        )

    industry_pe = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if industry_pe.empty:
        return industry_pe

    industry_pe = industry_pe.rename(
        columns={
            "date": "Date",
            "industry": "Industry",
            "exchange": "Exchange",
            "pe": "PE Ratio",
        }
    )

    if date:
        industry_pe = industry_pe.set_index("Industry").sort_index()
    else:
        industry_pe["Date"] = pd.to_datetime(industry_pe["Date"])
        industry_pe = industry_pe.set_index("Date").sort_index()

    return industry_pe


def get_mergers_acquisitions_latest(
    api_key: str, limit: int = 100, page: int = 0, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the most recent mergers and acquisitions deal announcements.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        limit (int, optional): The number of results to return. Defaults to 100.
        page (int, optional): The page number to retrieve. Defaults to 0.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the latest mergers and acquisitions.
    """
    url = (
        "https://financialmodelingprep.com/stable/mergers-acquisitions-latest"
        f"?page={page}&limit={limit}&apikey={api_key}"
    )

    mergers_acquisitions = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    return _format_mergers_acquisitions(mergers_acquisitions)


def _format_mergers_acquisitions(mergers_acquisitions: pd.DataFrame) -> pd.DataFrame:
    """
    Shared column renaming and indexing logic for the mergers and acquisitions endpoints.

    Args:
        mergers_acquisitions (pd.DataFrame): the raw M&A DataFrame as returned by the API.

    Returns:
        pd.DataFrame: DataFrame of M&A deals, indexed by Symbol.
    """
    if mergers_acquisitions.empty:
        return mergers_acquisitions

    mergers_acquisitions = mergers_acquisitions.rename(
        columns={
            "symbol": "Symbol",
            "companyName": "Company Name",
            "cik": "CIK",
            "targetedCompanyName": "Targeted Company Name",
            "targetedCik": "Targeted CIK",
            "targetedSymbol": "Targeted Symbol",
            "transactionDate": "Transaction Date",
            "acceptedDate": "Accepted Date",
            "link": "URL",
        }
    )

    mergers_acquisitions = mergers_acquisitions.set_index("Symbol").sort_index()

    return mergers_acquisitions


def get_earnings_calendar(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the earnings releases of all companies in a date range, with the reported and
    estimated earnings per share and revenue.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the earnings releases, indexed by symbol.
    """
    url = f"https://financialmodelingprep.com/stable/earnings-calendar?apikey={api_key}"

    if start_date:
        url += f"&from={start_date}"
    if end_date:
        url += f"&to={end_date}"

    earnings_calendar = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if earnings_calendar.empty:
        return earnings_calendar

    earnings_calendar = earnings_calendar.rename(
        columns={
            "symbol": "Symbol",
            "date": "Date",
            "epsActual": "EPS",
            "epsEstimated": "Estimated EPS",
            "revenueActual": "Revenue",
            "revenueEstimated": "Estimated Revenue",
            "lastUpdated": "Last Updated",
        }
    )

    earnings_calendar = earnings_calendar.sort_values(["Date", "Symbol"])

    return earnings_calendar.set_index("Symbol")


def get_sec_filings_8k(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    page: int = 0,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the most recent 8-K filings with the SEC: the reports companies file to announce
    a material event, such as an acquisition, a change of management or results.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with. Defaults to 30 days ago.
        end_date (str, optional): The end date to filter data with. Defaults to today.
        limit (int, optional): The number of results to return. Defaults to 100.
        page (int, optional): The page number to retrieve. Defaults to 0.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the 8-K filings, indexed by symbol.
    """
    url = (
        "https://financialmodelingprep.com/stable/sec-filings-8k"
        f"?page={page}&limit={limit}&apikey={api_key}"
    )

    # Without a range the endpoint answers with filings from March 2024, so the most
    # recent filings are the last 30 days unless a range is given.
    today = pd.Timestamp.today().normalize()
    start_date = start_date or f"{today - pd.Timedelta(days=30):%Y-%m-%d}"
    end_date = end_date or f"{today:%Y-%m-%d}"
    url += f"&from={start_date}&to={end_date}"

    filings = get_cached_financial_data(url=url, user_subscription=user_subscription)

    if filings.empty:
        return filings

    filings = filings.rename(
        columns={
            "symbol": "Symbol",
            "cik": "CIK",
            "filingDate": "Filing Date",
            "acceptedDate": "Accepted Date",
            "formType": "Form Type",
            "hasFinancials": "Has Financials",
            "link": "Link",
            "finalLink": "Final Link",
        }
    )

    return filings.set_index("Symbol")


# The most pages of the latest insider trades read to find the filings of one date.
INSIDER_TRADING_MAXIMUM_PAGES = 20


def get_insider_trading_latest(
    api_key: str,
    date: str | None = None,
    limit: int = 100,
    page: int = 0,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the most recent trades by company insiders (officers, directors and large
    shareholders) as reported in their Form 4 filings.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        date (str, optional): Only return the trades filed on this date.
        limit (int, optional): The number of results to return. Defaults to 100.
        page (int, optional): The page number to retrieve. Defaults to 0.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the latest insider trades, indexed by symbol.
    """

    def fetch(number: int) -> pd.DataFrame:
        return get_cached_financial_data(
            url=(
                "https://financialmodelingprep.com/stable/insider-trading/latest"
                f"?page={number}&limit={limit}&apikey={api_key}"
            ),
            user_subscription=user_subscription,
        )

    if not date:
        insider_trading = fetch(page)
    else:
        # The endpoint ignores a date, so the latest filings are paged through until
        # that date has passed and only its filings are kept.
        pages = []
        for number in range(page, page + INSIDER_TRADING_MAXIMUM_PAGES):
            batch = fetch(number)
            if batch.empty or "filingDate" not in batch:
                break
            filed = batch["filingDate"].astype(str).str[:10]
            pages.append(batch[filed == date])
            if (filed < date).any():
                break
        insider_trading = (
            pd.concat(pages, ignore_index=True).head(limit) if pages else pd.DataFrame()
        )

    if insider_trading.empty:
        return insider_trading

    insider_trading = insider_trading.rename(
        columns={
            "symbol": "Symbol",
            "filingDate": "Filing Date",
            "transactionDate": "Transaction Date",
            "reportingName": "Reporting Name",
            "typeOfOwner": "Type of Owner",
            "transactionType": "Transaction Type",
            "acquisitionOrDisposition": "Acquisition or Disposition",
            "directOrIndirect": "Direct or Indirect",
            "securitiesTransacted": "Securities Transacted",
            "price": "Price",
            "securitiesOwned": "Securities Owned",
            "securityName": "Security Name",
            "formType": "Form Type",
            "reportingCik": "Reporting CIK",
            "companyCik": "Company CIK",
            "url": "URL",
        }
    )

    return insider_trading.set_index("Symbol")
