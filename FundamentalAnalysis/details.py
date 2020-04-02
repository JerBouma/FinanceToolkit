from urllib.request import urlopen
import json
import pandas as pd


def available_companies():
    """
    Description
    ----
    Gives all company names and tickers that are available for retrieval
    of financial statements, ratios and extended stock data. General stock
    data can be retrieved for any company or financial instrument.

    Output
    ----
    data (dataframe)
        Data with the ticker as the index and the company name in the column.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/company/stock/list")
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['symbolsList']

    ticker_list = {}
    for ticker in data_json:
        try:
            ticker_list[ticker["symbol"]] = ticker['name']
        except KeyError:
            ticker_list[ticker["symbol"]] = ticker['symbol']
    data_formatted = pd.DataFrame(ticker_list,
                                  index=["name"]).T.sort_index()

    return data_formatted


def profile(ticker):
    """
    Description
    ----
    Gives information about the profile of a company which includes
    i.a. beta, company description, industry and sector.

    Input
    ----
    ticker (string)
        The company ticker (for example: "AAPL")

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/company/profile/" + ticker)
    data = response.read().decode("utf-8")
    data_formatted = pd.DataFrame(json.loads(data)['profile'], index=['profile']).T

    return data_formatted


def quote(ticker):
    """
    Description
    ----
    Gives information about the quote of a company which includes i.a.
    high/low close prices, price-to-earning ratio and shares outstanding.

    Input
    ----
    ticker (string)
        The company ticker (for example: "AMD")

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/quote/" + ticker)
    data = response.read().decode("utf-8")
    data_formatted = pd.DataFrame(json.loads(data)[0], index=["quote"]).T

    return data_formatted


def enterprise(ticker, period="annual"):
    """
    Description
    ----
    Gives information about the enterprise value of a company which includes
    i.a. market capitalisation, Cash & Cash Equivalents, total debt and enterprise value.

    Input
    ----
    ticker (string)
        The company ticker (for example: "TSLA")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/enterprise-value/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['enterpriseValues']

    data_formatted = {}
    for data in data_json:
        if period == "quarter":
            date = data['date'][:7]
        else:
            date = data['date'][:4]
        del data['date']
        data_formatted[date] = data

    return pd.DataFrame(data_formatted)


def rating(ticker):
    """
     Description
     ----
     Gives information about the rating of a company which includes i.a. the company
     rating and recommendation as well as ratings based on a variety of ratios.

     Input
     ----
     ticker (string)
         The company ticker (for example: "MSFT")

     Output
     ----
     data (dataframe)
        Data with variables in rows and the period in columns..
     """
    response = urlopen("https://financialmodelingprep.com/api/v3/company/rating/" +
                       ticker)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)
    data_formatted = pd.DataFrame(data_json["ratingDetails"]).T

    return data_formatted


def discounted_cash_flow(ticker, period="annual"):
    """
    Description
    ----
    Gives information about the discounted cash flow (DCF) of a company which includes
    i.a. the (current) stock price and DCF and over time.

    Input
    ----
    ticker (string)
        The company ticker (for example: "UBER")
    period (string)
        Data period, this can be "annual" or "quarter".

    Output
    ----
    data (dataframe)
        Data with variables in rows and the period in columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/company/discounted-cash-flow/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json_current = json.loads(data)

    try:
        del data_json_current['symbol']
        data_json_current['DCF'] = data_json_current.pop('dcf')
    except KeyError:
        pass

    response = urlopen("https://financialmodelingprep.com/api/v3/company/historical-discounted-cash-flow/" +
                       ticker + "?period=" + period)
    data = response.read().decode("utf-8")
    data_json = json.loads(data)['historicalDCF']

    data_formatted = {}

    if period == "quarter":
        current_year = data_json_current['date'][:7]
    else:
        current_year = data_json_current['date'][:4]
    data_formatted[current_year] = data_json_current

    for data in data_json:
        if period == "quarter":
            date = data['date'][:7]
        else:
            date = data['date'][:4]
        data_formatted[date] = data

    return pd.DataFrame(data_formatted)
