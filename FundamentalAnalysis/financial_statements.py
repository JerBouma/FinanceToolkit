from urllib.request import urlopen
import json
import pandas as pd


def income_statement(ticker, period="annual"):
    """
    Description
    ----
    Gives information about the income statement of a company overtime
    which includes i.a. revenue, operating expenses, profit margin and ETBIDA.

    Input
    ----
    ticker (string)
        The company ticker (for example: "GOOGL")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/financials/income-statement/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['financials']

    data_formatted = {}
    for data in data_json:
        if period == "quarter":
            date = data['date'][:7]
        else:
            date = data['date'][:4]
        del data['date']
        data_formatted[date] = data

    return pd.DataFrame(data_formatted)


def balance_sheet_statement(ticker, period="annual"):
    """
    Description
    ----
    Gives information about the balance sheet statement of a company  overtime
    which includes i.a. total assets, payables, tax liabilities and investments.

    Input
    ----
    ticker (string)
        The company ticker (for example: "RDS-B")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['financials']

    data_formatted = {}
    for data in data_json:
        if period == "quarter":
            date = data['date'][:7]
        else:
            date = data['date'][:4]
        del data['date']
        data_formatted[date] = data

    return pd.DataFrame(data_formatted)


def cash_flow_statement(ticker, period="annual"):
    """
    Description
    ----
    Gives information about the cash flow statement of a company  overtime
    which includes i.a. operating cash flow, dividend payments and capital expenditure.

    Input
    ----
    ticker (string)
        The company ticker (for example: "NKE")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['financials']

    data_formatted = {}
    for data in data_json:
        if period == "quarter":
            date = data['date'][:7]
        else:
            date = data['date'][:4]
        del data['date']
        data_formatted[date] = data

    return pd.DataFrame(data_formatted)