from urllib.request import urlopen
import json
import pandas as pd


def income_statement(ticker, api_key, period="annual"):
    """
    Description
    ----
    Gives information about the income statement of a company overtime
    which includes i.a. revenue, operating expenses, profit margin and ETBIDA.

    Input
    ----
    ticker (string)
        The company ticker (for example: "GOOGL")
    api_key (string)
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/income-statement/" +
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


def balance_sheet_statement(ticker, api_key, period="annual"):
    """
    Description
    ----
    Gives information about the balance sheet statement of a company  overtime
    which includes i.a. total assets, payables, tax liabilities and investments.

    Input
    ----
    ticker (string)
        The company ticker (for example: "RDS-B")
    api_key (string)
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/balance-sheet-statement/" +
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


def cash_flow_statement(ticker, api_key, period="annual"):
    """
    Description
    ----
    Gives information about the cash flow statement of a company  overtime
    which includes i.a. operating cash flow, dividend payments and capital expenditure.

    Input
    ----
    ticker (string)
        The company ticker (for example: "NKE")
    api_key (string)
        The API Key obtained from https://financialmodelingprep.com/developer/docs/
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/cash-flow-statement/" +
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