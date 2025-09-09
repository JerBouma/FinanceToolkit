"""Discovery Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fmp_model import get_financial_data
from financetoolkit.utilities import error_model


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

    instruments_query = get_financial_data(url=url, user_subscription=user_subscription)

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
    is_etf: bool | None = None,
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
        is_etf (bool, optional): Whether the instrument is an ETF. Defaults to None.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of instruments matching the query.
    """
    url = f"https://financialmodelingprep.com/stable/company-screener?apikey={api_key}"

    if market_cap_higher:
        url += f"&marketCapMoreThan={market_cap_higher}"
    if market_cap_lower:
        url += f"&marketCapLowerThan={market_cap_lower}"
    if price_higher:
        url += f"&priceMoreThan={price_higher}"
    if price_lower:
        url += f"&priceLowerThan={price_lower}"
    if beta_higher:
        url += f"&betaMoreThan={beta_higher}"
    if beta_lower:
        url += f"&betaLowerThan={beta_lower}"
    if volume_higher:
        url += f"&volumeMoreThan={volume_higher}"
    if volume_lower:
        url += f"&volumeLowerThan={volume_lower}"
    if dividend_higher:
        url += f"&dividendMoreThan={dividend_higher}"
    if dividend_lower:
        url += f"&dividendLowerThan={dividend_lower}"
    if is_etf is not None:
        url += f"&isEtf={str(is_etf)}"

    stock_screener = get_financial_data(url=url, user_subscription=user_subscription)

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

    if stock_screener.empty:
        raise ValueError("No stocks found matching the query.")

    stock_screener = stock_screener.drop(columns=["isEtf", "isActivelyTrading"])
    stock_screener = stock_screener.set_index("Symbol")

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

    stock_list = get_financial_data(url=url, user_subscription=user_subscription)

    stock_list = stock_list.rename(
        columns={
            "symbol": "Symbol",
            "companyName": "Name",
        }
    )

    stock_list = stock_list.set_index("Symbol").sort_index()

    return stock_list


def get_stock_quotes(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get the quotes for all stocks.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of stock quotes.
    """
    url = f"https://financialmodelingprep.com/stable/stock/full/real-time-price?apikey={api_key}"

    stock_quotes = get_financial_data(url=url, user_subscription=user_subscription)

    stock_quotes = stock_quotes.rename(
        columns={
            "symbol": "Symbol",
            "bidSize": "Bid Size",
            "askPrice": "Ask Price",
            "volume": "Volume",
            "askSize": "Ask Size",
            "bidPrice": "Bid Price",
            "lastSalePrice": "Last Sale Price",
            "lastSaleSize": "Last Sale Size",
            "lastSaleTime": "Last Sale Time",
        }
    )

    stock_quotes = stock_quotes.drop(columns=["fmpLast", "lastUpdated"])

    stock_quotes = stock_quotes.set_index("Symbol").sort_index()

    return stock_quotes


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

    stock_shares_float = get_financial_data(
        url=url, user_subscription=user_subscription
    )

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
    api_key: str, user_subscription: str = "Free"
) -> pd.DataFrame:
    """
    Get the sectors performance.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of sectors performance.
    """
    url = f"https://financialmodelingprep.com/stable/historical-sectors-performance?apikey={api_key}"

    sectors_performance = get_financial_data(
        url=url, user_subscription=user_subscription
    )

    sectors_performance = sectors_performance.rename(
        columns={
            "date": "Date",
            "utilitiesChangesPercentage": "Utilities",
            "basicMaterialsChangesPercentage": "Basic Materials",
            "communicationServicesChangesPercentage": "Communication Services",
            "conglomeratesChangesPercentage": "Conglomerates",
            "consumerCyclicalChangesPercentage": "Consumer Cyclical",
            "consumerDefensiveChangesPercentage": "Consumer Defensive",
            "energyChangesPercentage": "Energy",
            "financialChangesPercentage": "Financial",
            "financialServicesChangesPercentage": "Financial Services",
            "healthcareChangesPercentage": "Healthcare",
            "industrialsChangesPercentage": "Industrials",
            "realEstateChangesPercentage": "Real Estate",
            "servicesChangesPercentage": "Services",
            "technologyChangesPercentage": "Technology",
        }
    )

    sectors_performance = sectors_performance.set_index("Date")

    sectors_performance.index = pd.PeriodIndex(sectors_performance.index, freq="D")

    sectors_performance = sectors_performance.sort_index()

    sectors_performance = sectors_performance.dropna(how="all", axis=1)

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

    biggest_gainers = get_financial_data(url=url, user_subscription=user_subscription)

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

    biggest_losers = get_financial_data(url=url, user_subscription=user_subscription)

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

    most_active = get_financial_data(url=url, user_subscription=user_subscription)

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

    crypto_list = get_financial_data(url=url, user_subscription=user_subscription)

    crypto_list = crypto_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "exchange": "Currency",
            "icoDate": "ICO Date",
            "circulatingSupply": "Circulating Supply",
            "totalSupply": "Total Supply",
        }
    )

    crypto_list = crypto_list.set_index("Symbol").sort_index()

    return crypto_list


def get_delisted_stocks(api_key: str, user_subscription: str = "Free") -> pd.DataFrame:
    """
    Get a list of delisted companies.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of delisted companies.
    """
    url = (
        f"https://financialmodelingprep.com/stable/delisted-companies?apikey={api_key}"
    )

    delisted_companies = get_financial_data(
        url=url, user_subscription=user_subscription
    )

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
    url = f"https://financialmodelingprep.com/stable/symbol/available-forex-currency-pairs?apikey={api_key}"

    forex_list = get_financial_data(url=url, user_subscription=user_subscription)

    forex_list = forex_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "currency": "Currency",
            "stockExchange": "Exchange",
            "exchangeShortName": "Exchange Code",
        }
    )

    forex_list = forex_list.drop(columns=["Exchange Code"])

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
    url = f"https://financialmodelingprep.com/stable/symbol/available-commodities?apikey={api_key}"

    commody_list = get_financial_data(url=url, user_subscription=user_subscription)

    commody_list = commody_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "currency": "Currency",
            "stockExchange": "Exchange",
            "exchangeShortName": "Exchange Code",
        }
    )

    commody_list = commody_list.drop(columns=["Exchange Code"])

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
    url = f"https://financialmodelingprep.com/stable/etf/list?apikey={api_key}"

    etf_list = get_financial_data(url=url, user_subscription=user_subscription)

    etf_list = etf_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "price": "Price",
            "exchange": "Exchange",
            "exchangeShortName": "Exchange Code",
            "type": "Type",
        }
    )

    etf_list = etf_list.drop(columns=["Type"])

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
    url = f"https://financialmodelingprep.com/stable/symbol/available-indexes?apikey={api_key}"

    index_list = get_financial_data(url=url, user_subscription=user_subscription)

    index_list = index_list.rename(
        columns={
            "symbol": "Symbol",
            "name": "Name",
            "currency": "Currency",
            "stockExchange": "Exchange",
            "exchangeShortName": "Exchange Code",
        }
    )

    index_list = index_list.drop(columns=["Exchange Code"])

    index_list = index_list.set_index("Symbol").sort_index()

    return index_list
