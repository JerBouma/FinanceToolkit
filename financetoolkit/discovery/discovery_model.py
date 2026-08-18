"""Discovery Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.cache.request_model import redact_credentials
from financetoolkit.fmp_model import get_financial_data
from financetoolkit.utilities import error_model, logger_model

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

    response = get_financial_data(
        url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
    )

    # An empty frame may be a rate limit or plan error, not a genuine result.
    if cache is not None and isinstance(response, pd.DataFrame) and not response.empty:
        cache.set(
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset="discovery",
            entity=entity,
            data=response,
        )

    return response


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
    url = f"https://financialmodelingprep.com/stable/search-{search_method}?query={query}&apikey={api_key}"

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

    return instruments_query


def get_stock_screener(
    api_key: str,
    market_cap_higher: int | None = None,
    market_cap_lower: int | None = None,
    price_higher: int | None = None,
    price_lower: int | None = None,
    beta_higher: int | None = None,
    beta_lower: int | None = None,
    volume_higher: int | None = None,
    volume_lower: int | None = None,
    dividend_higher: int | None = None,
    dividend_lower: int | None = None,
    sector: str | None = None,
    industry: str | None = None,
    country: str | None = None,
    exchange: str | None = None,
    is_etf: bool | None = None,
    limit: int = 1000,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get a list of instruments based on the screening criteria provided. It defaults
    to all stocks.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        market_cap_higher (int, optional): The market cap higher than. Defaults to None.
        market_cap_lower (int, optional): The market cap lower than. Defaults to None.
        price_higher (int, optional): The price higher than. Defaults to None.
        price_lower (int, optional): The price lower than. Defaults to None.
        beta_higher (int, optional): The beta higher than. Defaults to None.
        beta_lower (int, optional): The beta lower than. Defaults to None.
        volume_higher (int, optional): The volume higher than. Defaults to None.
        volume_lower (int, optional): The volume lower than. Defaults to None.
        dividend_higher (int, optional): The dividend higher than. Defaults to None.
        dividend_lower (int, optional): The dividend lower than. Defaults to None.
        sector (str, optional): The sector to restrict the screen to, e.g. "Energy".
        industry (str, optional): The industry to restrict the screen to, e.g. "Biotechnology".
        country (str, optional): The two-letter country code to restrict the screen to, e.g. "US".
        exchange (str, optional): The exchange code to restrict the screen to, e.g. "NASDAQ".
        is_etf (bool, optional): Whether the instrument is an ETF. Defaults to None.
        limit (int, optional): The maximum number of results the API returns. Defaults to 1000,
            which is also the API's own default, so hitting exactly this number means the result
            was truncated.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of instruments matching the query. An empty DataFrame is returned
            when nothing matches.

    Notes:
        - Every bound is compared against None rather than truthiness, so a bound of exactly zero
          (a perfectly valid beta, dividend or volume floor) is sent to the API instead of dropped.
    """
    url = f"https://financialmodelingprep.com/stable/company-screener?apikey={api_key}"

    query_parameters = {
        "marketCapMoreThan": market_cap_higher,
        "marketCapLowerThan": market_cap_lower,
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
        "country": country,
        "exchange": exchange,
        "isEtf": is_etf,
        "limit": limit,
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

    if len(stock_screener) >= limit:
        logger.warning(
            "The screener returned the maximum of %s results, so the list is truncated. "
            "Narrow the criteria or raise the limit to see the remainder.",
            limit,
        )

    return stock_screener


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

    stock_shares_float = stock_shares_float.set_index("Symbol").sort_index()

    return stock_shares_float


def get_sectors_performance(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the historical performance of every sector, one column per sector.

    The wide layout is assembled from the per-sector historical endpoint, which is queried once
    for each of the sectors Financial Modeling Prep classifies companies into. The single
    combined endpoint this function used to call was retired and returns an empty response.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with, e.g. "2024-01-01".
        end_date (str, optional): The end date to filter data with, e.g. "2024-12-31".
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of sector performance indexed by date, with one column per sector
            holding the average percentage change of that sector on that date. The values are
            percentages, i.e. 1.25 means +1.25%.
    """
    sector_changes = {}

    for sector in FMP_SECTORS:
        url = (
            "https://financialmodelingprep.com/stable/historical-sector-performance"
            f"?sector={sector}&apikey={api_key}"
        )

        if start_date:
            url += f"&from={start_date}"
        if end_date:
            url += f"&to={end_date}"

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

    sectors_performance = pd.concat(sector_changes, axis=1)

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

    biggest_gainers = biggest_gainers.set_index("Symbol").sort_index()

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

    biggest_losers = biggest_losers.set_index("Symbol").sort_index()

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

    most_active = most_active.set_index("Symbol").sort_index()

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
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of crypto news articles matching the given symbols.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/crypto"
        f"?symbols={_normalize_symbols(symbols)}&limit={limit}&apikey={api_key}"
    )

    crypto_news = _get_news_pages(
        base_url=base_url, pages=pages, user_subscription=user_subscription
    )

    return _format_news(crypto_news)


def search_forex_news(
    api_key: str,
    symbols: str | list[str],
    limit: int = 100,
    pages: int = 1,
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
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of forex news articles matching the given symbols.
    """
    base_url = (
        "https://financialmodelingprep.com/stable/news/forex"
        f"?symbols={_normalize_symbols(symbols)}&limit={limit}&apikey={api_key}"
    )

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
        }
    )

    ipo_calendar = ipo_calendar.set_index("Symbol").sort_index()

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

    splits_calendar = splits_calendar.set_index("Symbol").sort_index()

    return splits_calendar


def get_sector_performance(
    api_key: str,
    date: str | None = None,
    sector: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get sector performance. Provide exactly one of `date` (a snapshot across all
    sectors on that date) or `sector` (the historical time series for one sector).

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        date (str, optional): The date to retrieve a snapshot for, e.g. "2024-02-01".
        sector (str, optional): The sector to retrieve the history for, e.g. "Energy".
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of sector performance, indexed by Sector (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(sector):
        raise ValueError("Please provide exactly one of `date` or `sector`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/sector-performance-snapshot"
            f"?date={date}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-sector-performance"
            f"?sector={sector}&apikey={api_key}"
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

    Returns:
        pd.DataFrame: DataFrame of industry performance, indexed by Industry (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(industry):
        raise ValueError("Please provide exactly one of `date` or `industry`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/industry-performance-snapshot"
            f"?date={date}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-industry-performance"
            f"?industry={industry}&apikey={api_key}"
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

    Returns:
        pd.DataFrame: DataFrame of sector P/E ratios, indexed by Sector (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(sector):
        raise ValueError("Please provide exactly one of `date` or `sector`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/sector-pe-snapshot"
            f"?date={date}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-sector-pe"
            f"?sector={sector}&apikey={api_key}"
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

    Returns:
        pd.DataFrame: DataFrame of industry P/E ratios, indexed by Industry (snapshot)
            or Date (historical).
    """
    if bool(date) == bool(industry):
        raise ValueError("Please provide exactly one of `date` or `industry`.")

    if date:
        url = (
            "https://financialmodelingprep.com/stable/industry-pe-snapshot"
            f"?date={date}&apikey={api_key}"
        )
    else:
        url = (
            "https://financialmodelingprep.com/stable/historical-industry-pe"
            f"?industry={industry}&apikey={api_key}"
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
