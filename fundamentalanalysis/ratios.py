from urllib.request import urlopen
from urllib.error import HTTPError
import json
import pandas as pd


def key_metrics(ticker, api_key, period="annual", TTM=False, limit=0):
    """
    Description
    ----
    Gives information about key metrics of a company overtime which includes
    i.a. PE ratio, Debt to Equity, Dividend Yield and Average Inventory.

    Input
    ----
    ticker (string)
        The company ticker (for example: "NFLX")
    api_key (string)
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    period (string)
        Data period, this can be "annual" or "quarter".
    TTM (boolean)
        Obtain the trailing twelve months (TTM) key metrics.
    limit (integer)
        The limit for the years of data

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    if TTM:
        URL = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?limit={limit}&apikey={api_key}"
    else:
        URL = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period={period}&limit={limit}&apikey={api_key}"

    try:
        response = urlopen(URL)
        data = json.loads(response.read().decode("utf-8"))
    except HTTPError:
        raise ValueError("This endpoint is only for premium members. Please visit the subscription page to upgrade the "
                         "plan (Starter or higher) at https://financialmodelingprep.com/developer/docs/pricing")

    if 'Error Message' in data:
        raise ValueError(data['Error Message'])

    if TTM:
        data_formatted = pd.Series(data[0])
    else:
        data_formatted = {}
        for value in data:
            if period == "quarter":
                date = value['date'][:7]
            else:
                date = value['date'][:4]
            del value['date']
            del value['symbol']

            data_formatted[date] = value
        data_formatted = pd.DataFrame(data_formatted)

    return data_formatted


def financial_ratios(ticker, api_key, period="annual", TTM=False, limit=0):
    """
    Description
    ----
    Gives information about the financial ratios of a company overtime
    which includes i.a. investment, liquidity, profitability and debt ratios.

    Input
    ----
    ticker (string)
        The company ticker (for example: "LYFT")
    api_key (string)
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    period (string)
        Data period, this can be "annual" or "quarter".
    TTM (boolean)
        Obtain the trailing twelve months (TTM) ratios.
    limit (integer)
        The limit for the years of data

    Output
    ----
    data (dataframe or series)
        Data with variables in rows and the period in columns.
    """
    if TTM:
        URL = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?limit={limit}&apikey={api_key}"
    else:
        URL = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?period={period}&limit={limit}&apikey={api_key}"

    try:
        response = urlopen(URL)
        data = json.loads(response.read().decode("utf-8"))
    except HTTPError:
        raise ValueError("This endpoint is only for premium members. Please visit the subscription page to upgrade the "
                         "plan (Starter or higher) at https://financialmodelingprep.com/developer/docs/pricing")

    if 'Error Message' in data:
        raise ValueError(data['Error Message'])

    if TTM:
        data_formatted = pd.Series(data[0])
    else:
        data_formatted = {}
        for value in data:
            if period == "quarter":
                date = value['date'][:7]
            else:
                date = value['date'][:4]
            del value['date']
            del value['symbol']

            data_formatted[date] = value
        data_formatted = pd.DataFrame(data_formatted)

    return data_formatted


def financial_statement_growth(ticker, api_key, period="annual", limit=0):
    """
    Description
    ----
    Gives information about the financial statement growth of a company overtime
    which includes i.a. EBIT growth (%) and shareholder equity growth (% per 3, 5
    and 10 years)

    Input
    ----
    ticker (string)
        The company ticker (for example: "WMT")
    api_key (string)
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    period (string)
        Data period, this can be "annual" or "quarter".
    limit (integer)
        The limit for the years of data

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    try:
        response = urlopen(f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}"
                           f"?period={period}&limit={limit}&apikey={api_key}")
        data = json.loads(response.read().decode("utf-8"))
    except HTTPError:
        raise ValueError("This endpoint is only for premium members. Please visit the subscription page to upgrade the "
                         "plan (Starter or higher) at https://financialmodelingprep.com/developer/docs/pricing")

    if 'Error Message' in data:
        raise ValueError(data['Error Message'])

    data_formatted = {}
    for value in data:
        if period == "quarter":
            date = value['date'][:7]
        else:
            date = value['date'][:4]
        del value['date']
        del value['symbol']

        data_formatted[date] = value

    return pd.DataFrame(data_formatted)
