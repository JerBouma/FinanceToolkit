"""Historical Module"""
__docformat__ = "numpy"

from datetime import datetime, timedelta

import pandas as pd


def get_historical_data(
    tickers: list[str] | str,
    start: str = None,
    end: str = None,
    interval: str = "1d",
):
    """
    Retrieves historical stock data for the given ticker(s) from Yahoo! Finance API for a specified period.
    If start and/or end date are not provided, it defaults to 10 years from the current date.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if end is not None:
        end_date = datetime.strptime(end, "%Y-%m-%d")
        end_timestamp = int(end_date.timestamp())
    else:
        end_date = datetime.today()
        end_timestamp = int(end_date.timestamp())

    if start is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d")

        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date}))"
            )

        start_timestamp = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    else:
        start_date = datetime.now() - timedelta(days=10 * 365)

        if start_date > end_date:
            start_date = end_date - timedelta(days=10 * 365)

        start_timestamp = int(start_date.timestamp())

    historical_data_dict: dict = {}

    if interval == "yearly":
        yearly = True
        interval = "1d"
    else:
        yearly = False

    corrupt_tickers = []
    for ticker in ticker_list:
        try:
            url = (
                f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?"
                f"interval={interval}&period1={start_timestamp}&period2={end_timestamp}"
                "&events=history&includeAdjustedClose=true"
            )
            historical_data_dict[ticker] = pd.read_csv(url, index_col="Date")
        except:
            corrupt_tickers.append(ticker)
            continue
    
    print(f"Removed {corrupt_tickers} from the tickers list.")

    historical_data = pd.concat(historical_data_dict, axis=1)

    historical_data.columns = historical_data.columns.swaplevel(0, 1)
    historical_data = historical_data.sort_index(axis=1)

    if yearly:
        historical_data = convert_daily_to_yearly(historical_data)

    return historical_data, corrupt_tickers


def convert_daily_to_yearly(daily_historical_data: pd.DataFrame):
    """
    Converts daily historical data to yearly historical data.

    Args:
        daily_historical_data (pd.DataFrame): A DataFrame containing daily historical data.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the yearly historical stock data.
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    daily_historical_data.index.name = "Date"
    daily_historical_data = daily_historical_data.reset_index()
    dates = pd.to_datetime(daily_historical_data.Date).dt.to_period("Y")
    yearly_historical_data = daily_historical_data.groupby(dates).transform("last")
    yearly_historical_data["Date"] = yearly_historical_data["Date"].str[:4]
    yearly_historical_data = yearly_historical_data.drop_duplicates().set_index("Date")

    return yearly_historical_data
