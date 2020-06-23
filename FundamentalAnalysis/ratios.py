from urllib.request import urlopen
import json
import pandas as pd


def key_metrics(ticker, api_key, period="annual"):
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

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/key-metrics/" +
                       ticker + "?period=" + period + "&apikey=" + api_key)
    data = json.loads(response.read().decode("utf-8"))

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


def financial_ratios(ticker, api_key, period="annual"):
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

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/ratios/" +
                       ticker + "?period=" + period + "&apikey=" + api_key)
    data = json.loads(response.read().decode("utf-8"))

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


def financial_statement_growth(ticker, api_key, period="annual"):
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

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/financial-growth/" +
                       ticker + "?period=" + period + "&apikey=" + api_key)
    data = json.loads(response.read().decode("utf-8"))

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
