import pandas as pd
from datetime import datetime
import requests
from urllib.request import urlopen
import json


def stock_data(ticker, period="max", interval="1d", start=None, end=None):
    """
    Description
    ----
    Gives information about the profile of a company which includes
    i.a. beta, company description, industry and sector.

    Input
    ----
    ticker (string)
        The company ticker (for example: "URTH")
    period (string)
        The range of the data (default = "max"), this can hold the following
        periods: "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
    interval (string)
        The frequency of the data (default = "1d"), this can hold the
        following intervals: "1m", "2m", "5m", "15m","30m","60m",
        "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo".
        Note that an interval less than 1h can only hold a period 1mo or less."
    start (string)
        The start date in the format %Y-%m-%d. Choose either start/end
        or period.
    end (string)
        The end date in the format %Y-%m-%d. Choose either start/end
        or period.

    Output
    ----
    data (dataframe)
        Data with dates in rows and the quotes in columns.
    """
    parameters = {"interval": interval}

    if start is None or end is None:
        parameters['range'] = period
    else:
        start_timestamp = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end, '%Y-%m-%d').timestamp())
        parameters["period1"] = start_timestamp
        parameters["period2"] = end_timestamp

    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker

    try:
        data = requests.get(url=url, params=parameters)
        data_json = data.json()['chart']['result'][0]
    except TypeError:
        raise TypeError("No data available. Please check if you have a "
                        "valid period and/or interval." + '\n' +
                        "Note that an interval less than 1h can only "
                        "hold a period of 1mo or less.")

    timestamp = data_json['timestamp']
    dates = []
    for ts in timestamp:
        if interval in ['1m', '2m', '5m', '15m',
                        '30m', '60m', '1h']:
            dates.append(datetime.fromtimestamp(int(ts)))
        else:
            dates.append(datetime.fromtimestamp(int(ts)).date())

    indicators = data_json['indicators']['quote'][0]
    try:
        indicators.update(data_json['indicators']['adjclose'][0])
    except Exception as e:
        print("Data for " + str(e) + " could not be included.")

    return pd.DataFrame(indicators, index=dates)


def stock_data_detailed(ticker, begin="1792-05-17", end=None):
    """
    Description
    ----
    Gives complete information about the company's stock which includes open, high,
    low, close, adjusted close, volume, unadjusted volume, change, change percent,
    volume weighted average price, label and change over time.

    This function only allows company tickers and is limited to the companies found
    by calling available_companies() from the details module.

    Input
    ----
    ticker (string)
        The company ticker (for example: "FIZZ")
    begin (string)
        Begin date in the format %Y-%m-%d.
        Default is the beginning of the Stock Market: 1792-05-17.
    end (string)
        End date in the format %Y-%m-%d.
        Default is the current date.

    Output
    ----
    data (dataframe)
        Data with the date in the rows and the variables in the columns.
    """
    response = urlopen("https://financialmodelingprep.com/api/v3/historical-price-full/" +
                       ticker + "?from=" + str(begin) + "&to=" + str(end))
    data = response.read().decode("utf-8")

    try:
        data_json = json.loads(data)['historical']
    except KeyError:
        raise ValueError("No data available. Please note this function "
                         "only takes a specific selection of companies." + '\n' +
                         "See: FundamentalAnalysis.available_companies()")

    data_formatted = {}
    for data in data_json:
        date = data['date']
        del data['date']
        data_formatted[date] = data
    data_formatted = pd.DataFrame(data_formatted).T

    return data_formatted
