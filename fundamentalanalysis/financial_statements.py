from urllib.request import urlopen
from urllib.error import HTTPError
import json
import pandas as pd


def income_statement(ticker, api_key, period="annual", as_reported=False, limit=0):
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
    as_reported (boolean)
        Raw data without modifications.
    limit (integer)
        The limit for the years of data

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    if as_reported:
        URL = (f"https://financialmodelingprep.com/api/v3/income-statement-as-reported/{ticker}"
               f"?period={period}&limit={limit}&apikey={api_key}")
    else:
        URL = (f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
               f"?period={period}&limit={limit}&apikey={api_key}")

    try:
        response = urlopen(URL)
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


def balance_sheet_statement(ticker, api_key, period="annual", as_reported=False, limit=0):
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
    as_reported (boolean)
        Raw data without modifications.
    limit (integer)
        The limit for the years of data

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    if as_reported:
        URL = (f"https://financialmodelingprep.com/api/v3/balance-sheet-statement-as-reported/{ticker}"
               f"?period={period}&limit={limit}&apikey={api_key}")
    else:
        URL = (f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}"
               f"?period={period}&limit={limit}&apikey={api_key}")

    try:
        response = urlopen(URL)
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


def cash_flow_statement(ticker, api_key, period="annual", as_reported=False, limit=0):
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
    as_reported (boolean)
        Raw data without modifications.
    limit (integer)
        The limit for the years of data

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    if as_reported:
        URL = (f"https://financialmodelingprep.com/api/v3/cash-flow-statement-as-reported/{ticker}"
               f"?period={period}&limit={limit}&apikey={api_key}")
    else:
        URL = (f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}"
               f"?period={period}&limit={limit}&apikey={api_key}")

    try:
        response = urlopen(URL)
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