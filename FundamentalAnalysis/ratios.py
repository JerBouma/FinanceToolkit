from urllib.request import urlopen
import json
import pandas as pd


def key_metrics(ticker, period="annual"):
    """
    Description
    ----
    Gives information about key metrics of a company overtime which includes
    i.a. PE ratio, Debt to Equity, Dividend Yield and Average Inventory.

    Input
    ----
    ticker (string)
        The company ticker (for example: "NFLX")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/company-key-metrics/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['metrics']

    data_formatted = {}
    for data in data_json:
        if period == "quarter":
            date = data['date'][:7]
        else:
            date = data['date'][:4]
        del data['date']
        data_formatted[date] = data

    return pd.DataFrame(data_formatted)


def financial_ratios(ticker):
    """
    Description
    ----
    Gives information about the financial ratios of a company overtime
    which includes i.a. investment, liquidity, profitability and debt ratios.

    Input
    ----
    ticker (string)
        The company ticker (for example: "LYFT")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/financial-ratios/" +
                       ticker)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['ratios']

    data_formatted = {}
    for data in data_json:
        date = data['date'][:4]
        del data['date']
        ratio_data = {}

        for key in data.keys():
            ratio_data.update(data[key])

        data_formatted[date] = ratio_data

    return pd.DataFrame(data_formatted)


def financial_statement_growth(ticker, period="annual"):
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
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/financial-statement-growth/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['growth']

    data_formatted = {}
    for data in data_json:
        if period == "quarter":
            date = data['date'][:7]
        else:
            date = data['date'][:4]
        del data['date']
        data_formatted[date] = data

    return pd.DataFrame(data_formatted)