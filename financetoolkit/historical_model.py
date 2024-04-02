"""Historical Module"""
__docformat__ = "google"

import contextlib
import threading
import time
from datetime import datetime, timedelta
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import requests

from financetoolkit import fundamentals_model, helpers

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False

# pylint: disable=too-many-locals,unsubscriptable-object,too-many-lines

TREASURY_LIMIT = 90


def get_historical_data(
    tickers: list[str] | str,
    api_key: str | None = None,
    source: str = "FinancialModelingPrep",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
    include_dividends: bool = True,
    progress_bar: bool = True,
    fill_nan: bool = True,
    divide_ohlc_by: int | float | None = None,
    rounding: int | None = None,
    sleep_timer: bool = True,
    show_ticker_seperation: bool = True,
    show_errors: bool = False,
    tqdm_message: str = "Obtaining historical data",
):
    """
    Retrieves historical stock data for the given ticker(s) from Financial Modeling Prep or/and Yahoo Finance
    for a specified period. If start and/or end date are not provided, it defaults to 10 years from
    the current date.

    It intelligently uses both Financial Modeling Prep and Yahoo Finance to retrieve the data. If the data
    is not available from Financial Modeling Prep, it will use Yahoo Finance instead. This is because
    opposes limits to Free plans (e.g. no tickers from outside the American exchanges) and in some cases
    Yahoo Finance has a broader universe.

    By using threading, multiple API calls can be made at the same time, which speeds up the process
    significantly. For example, collecting historical data of 100 tickers takes around 10 seconds.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        interval (str, optional): A string representing the interval to retrieve data for.
        return_column (str, optional): A string representing the column to use for return calculations.
        risk_free_rate (pd.Series, optional): A pandas Series object containing the risk free rate data.
        This is used to calculate the excess return and excess volatility. Defaults to pd.Series().
        include_dividends (bool, optional): A boolean representing whether to include dividends in the
        historical data. Defaults to True.
        progress_bar (bool, optional): A boolean representing whether to show a progress bar. Defaults to True.
        fill_nan (bool, optional): A boolean representing whether to fill NaN values with the previous
        value. Defaults to True.
        divide_ohlc_by (int, optional): An integer representing the value to divide the OHLC data by.
        This is useful if the OHLC data is presented in percentages or similar. Defaults to None.
        rounding (int, optional): The number of decimal places to round the data to. Defaults to None.
        sleep_timer (bool, optional): A boolean representing whether to introduce a sleep timer to prevent
        rate limit errors. Defaults to True.
        show_ticker_seperation (bool, optional): A boolean representing whether to show which tickers
        acquired data from FinancialModelingPrep and which tickers acquired data from YahooFinance.
        show_errors (bool, optional): A boolean representing whether to show errors. Defaults to True.
        tqdm_message (str, optional): A string representing the message to show in the progress bar.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    empty_historical_data = pd.DataFrame(
        data=0,
        index=pd.PeriodIndex(pd.date_range(start, end), freq="D"),
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Return",
            "Volatility",
            "Cumulative Return",
        ],
    )

    def worker(ticker, historical_data_dict):
        historical_data = pd.DataFrame()

        if api_key and interval in ["1min", "5min", "15min", "30min", "1hour", "4hour"]:
            historical_data = get_intraday_data_from_financial_modeling_prep(
                ticker=ticker,
                api_key=api_key,
                start=start,
                end=end,
                interval=interval,
                return_column=return_column,
                sleep_timer=sleep_timer,
            )
        elif not api_key and interval in [
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
            "4hour",
        ]:
            raise ValueError(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
        else:
            if api_key and source == "FinancialModelingPrep":
                historical_data = get_historical_data_from_financial_modeling_prep(
                    ticker=ticker,
                    api_key=api_key,
                    start=start,
                    end=end,
                    interval=interval,
                    return_column=return_column,
                    risk_free_rate=risk_free_rate,
                    include_dividends=include_dividends,
                    divide_ohlc_by=divide_ohlc_by,
                    sleep_timer=sleep_timer,
                )

                if not historical_data.empty:
                    fmp_tickers.append(ticker)

            if source == "YahooFinance" or historical_data.empty:
                historical_data = get_historical_data_from_yahoo_finance(
                    ticker=ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    return_column=return_column,
                    risk_free_rate=risk_free_rate,
                    include_dividends=include_dividends,
                    divide_ohlc_by=divide_ohlc_by,
                )

                if not historical_data.empty:
                    yf_tickers.append(ticker)

        if historical_data.empty:
            no_data.append(ticker)
            historical_data_dict[ticker] = empty_historical_data
        if not historical_data.empty:
            historical_data_dict[ticker] = historical_data

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    ticker_list_iterator = (
        tqdm(ticker_list, desc=tqdm_message)
        if (ENABLE_TQDM & progress_bar)
        else ticker_list
    )

    historical_data_dict: dict[str, pd.DataFrame] = {}
    fmp_tickers: list[str] = []
    yf_tickers: list[str] = []
    no_data: list[str] = []
    threads = []

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, historical_data_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    historical_data_dict = (
        helpers.check_for_error_messages(historical_data_dict)
        if show_errors
        else historical_data_dict
    )

    if fmp_tickers and yf_tickers and show_ticker_seperation:
        print(
            f"The following tickers acquired historical data from FinancialModelingPrep: {', '.join(fmp_tickers)}"
        )
        print(
            f"The following tickers acquired historical data from YahooFinance: {', '.join(yf_tickers)}"
        )

    if no_data and show_errors:
        print(f"No data found for the following tickers: {', '.join(no_data)}")

    if len(historical_data_dict) == 0:
        # Fill the DataFrame with zeros to ensure the DataFrame is returned
        # even if no data is found. This is mostly applicable when nothing
        # can be found at all.
        for ticker in tickers:
            historical_data_dict[ticker] = empty_historical_data

    reorder_tickers = [ticker for ticker in tickers if ticker in historical_data_dict]

    if not historical_data_dict:
        raise ValueError("No data found for the given tickers.")

    historical_data = pd.concat(historical_data_dict).unstack(level=0)
    historical_data = historical_data.reindex(reorder_tickers, level=1, axis=1)

    if "Dividends" in historical_data.columns:
        historical_data["Dividends"] = historical_data["Dividends"].fillna(0)

    if fill_nan:
        # Interpolation is done when there are NaN values in the DataFrame
        # while technically, that specific date doesn't have a value, it
        # smoothens the result with limited impact on any metric.
        historical_data = historical_data.interpolate(limit_area="inside")

    if rounding:
        historical_data = historical_data.round(rounding)

    return historical_data, no_data


def get_historical_data_from_financial_modeling_prep(
    ticker: str,
    api_key: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
    include_dividends: bool = True,
    divide_ohlc_by: int | float | None = None,
    sleep_timer: bool = True,
):
    """
    Retrieves historical stock data for the given ticker from Financial Modeling Prep for a specified period.
    If start and/or end date are not provided, it defaults to 10 years from the current date.

    Note that when using a Free API key from FinancialModelingPrep it will be limited to 5 years.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        interval (str, optional): A string representing the interval to retrieve data for.
        return_column (str, optional): A string representing the column to use for return calculations.
        risk_free_rate (pd.Series, optional): A pandas Series object containing the risk free rate data.
        This is used to calculate the excess return and excess volatility. Defaults to pd.Series().
        include_dividends (bool, optional): A boolean representing whether to include dividends in the
        historical data. Defaults to True.
        ohlc_divide_by (int, optional): An integer representing the value to divide the OHLC data by.
        This is useful if the OHLC data is presented in percentages or similar. Defaults to None.
        show_errors (bool, optional): A boolean representing whether to show errors. Defaults to True.
        sleep_timer (bool, optional): A boolean representing whether to introduce a sleep timer to prevent
        rate limit errors. Defaults to True.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    # Additional data is collected to ensure return calculations are correct
    end_date_value = (
        datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1 * 365)
        if end is not None
        else datetime.today()
    )

    if start is not None:
        # Additional data is collected to ensure return calculations are correct
        start_date_value = datetime.strptime(start, "%Y-%m-%d") - timedelta(
            days=1 * 365
        )

        if start_date_value > end_date_value:
            raise ValueError(
                f"Start date ({start_date_value}) must be before end date ({end_date_value}))"
            )
    else:
        start_date_value = datetime.now() - timedelta(days=10 * 365)

        if start_date_value > end_date_value:
            start_date_value = end_date_value - timedelta(days=10 * 365)

    end_date_string = end_date_value.strftime("%Y-%m-%d")
    start_date_string = start_date_value.strftime("%Y-%m-%d")

    if interval in ["yearly", "quarterly"]:
        interval = "1d"

    historical_data_url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?"
        f"apikey={api_key}&from={start_date_string}&to={end_date_string}"
    )

    dividend_url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?"
        f"apikey={api_key}&from={start_date_string}&to={end_date_string}"
    )

    try:
        historical_data = helpers.get_financial_data(
            url=historical_data_url, sleep_timer=sleep_timer, raw=True
        )

        historical_data = pd.DataFrame(historical_data["historical"]).set_index("date")
    except (HTTPError, KeyError, ValueError, URLError, RemoteDisconnected):
        return pd.DataFrame(historical_data)

    historical_data = historical_data.sort_index()

    if historical_data.loc[start_date_string:end_date_string].empty:
        print(f"The given start and end date result in no data found for {ticker}")
        return pd.DataFrame()

    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data.index = historical_data.index.to_period(freq="D")

    historical_data = historical_data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjClose": "Adj Close",
            "volume": "Volume",
        }
    )

    historical_data = historical_data[
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    ]

    if divide_ohlc_by:
        # Set divide by zero and invalid value warnings to ignore as it is fine that
        # dividing NaN by divide_ohlc_by results in NaN
        np.seterr(divide="ignore", invalid="ignore")
        # In case tickers are presented in percentages or similar
        historical_data = historical_data.div(divide_ohlc_by)

    if include_dividends:
        try:
            dividends = helpers.get_financial_data(
                url=dividend_url, sleep_timer=sleep_timer, raw=True
            )

            try:
                dividends_df = pd.DataFrame(dividends["historical"]).set_index("date")

                if not dividends_df.empty:
                    dividends_df.index = pd.to_datetime(dividends_df.index)
                    dividends_df.index = dividends_df.index.to_period(freq="D")

                    historical_data["Dividends"] = dividends_df["dividend"]
                else:
                    historical_data["Dividends"] = 0
            except KeyError:
                historical_data["Dividends"] = 0
        except (HTTPError, URLError, RemoteDisconnected):
            historical_data["Dividends"] = 0

    historical_data = historical_data.loc[
        ~historical_data.index.duplicated(keep="first")
    ]

    historical_data = enrich_historical_data(
        historical_data=historical_data,
        start=start,
        end=end,
        return_column=return_column,
        risk_free_rate=risk_free_rate,
    )

    return historical_data


def get_historical_data_from_yahoo_finance(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
    include_dividends: bool = True,
    divide_ohlc_by: int | float | None = None,
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
        interval (str, optional): A string representing the interval to retrieve data for.
        return_column (str, optional): A string representing the column to use for return calculations.
        risk_free_rate (pd.Series, optional): A pandas Series object containing the risk free rate data.
        include_dividends (bool, optional): A boolean representing whether to include dividends in the
        historical data. Defaults to True.
        This is used to calculate the excess return and excess volatility. Defaults to pd.Series().
        divide_ohlc_by (int or float, optional): A number to divide the OHLC data by. Defaults to None.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    if end is not None:
        # Additional data is collected to ensure return calculations are correct
        end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1 * 365)
        end_timestamp = int(end_date.timestamp())
    else:
        end_date = datetime.today()
        end_timestamp = int(end_date.timestamp())
        end = end_date.strftime("%Y-%m-%d")

    if start is not None:
        # Additional data is collected to ensure return calculations are correct
        start_date = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1 * 365)

        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date}))"
            )

        start_timestamp = int(start_date.timestamp())
    else:
        start_date = datetime.now() - timedelta(days=10 * 365)
        start = start_date.strftime("%Y-%m-%d")

        if start_date > end_date:
            start_date = end_date - timedelta(days=10 * 365)

        start_timestamp = int(start_date.timestamp())

    if interval in ["yearly", "quarterly"]:
        interval = "1d"

    historical_data_url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?"
        f"interval={interval}&period1={start_timestamp}&period2={end_timestamp}"
        "&events=history&includeAdjustedClose=true"
    )

    dividend_url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?"
        f"interval={interval}&period1={start_timestamp}&period2={end_timestamp}"
        "&events=div&includeAdjustedClose=true"
    )

    try:
        historical_data = pd.read_csv(historical_data_url, index_col="Date")
    except (HTTPError, URLError, RemoteDisconnected):
        return pd.DataFrame()

    if historical_data.loc[start:end].empty:
        print(f"The given start and end date result in no data found for {ticker}")
        return pd.DataFrame()

    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data.index = historical_data.index.to_period(freq="D")

    if divide_ohlc_by:
        # Set divide by zero and invalid value warnings to ignore as it is fine that
        # dividing NaN by divide_ohlc_by results in NaN
        np.seterr(divide="ignore", invalid="ignore")
        # In case tickers are presented in percentages or similar
        historical_data = historical_data.div(divide_ohlc_by)

    if include_dividends:
        try:
            dividends = pd.read_csv(dividend_url, index_col="Date")["Dividends"]

            dividends.index = pd.to_datetime(dividends.index)
            dividends.index = dividends.index.to_period(freq="D")

            historical_data["Dividends"] = dividends

            historical_data["Dividends"] = historical_data["Dividends"].infer_objects(
                copy=False
            )

            historical_data["Dividends"] = historical_data["Dividends"].fillna(0)
        except (HTTPError, URLError, RemoteDisconnected):
            historical_data["Dividends"] = 0

    historical_data = historical_data.loc[
        ~historical_data.index.duplicated(keep="first")
    ]

    historical_data = enrich_historical_data(
        historical_data=historical_data,
        start=start,
        end=end,
        return_column=return_column,
        risk_free_rate=risk_free_rate,
    )

    return historical_data


def get_intraday_data_from_financial_modeling_prep(
    ticker: str,
    api_key: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1hour",
    return_column: str = "Close",
    sleep_timer: bool = True,
):
    """
    Retrieves intraday stock data for the given ticker from Financial Modeling Prep for a specified period.
    If start and/or end date are not provided, it defaults to 10 years from the current date.

    Note that when using a Free API key from FinancialModelingPrep it will be limited to 5 years.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        interval (str, optional): A string representing the interval to retrieve data for.
        return_column (str, optional): A string representing the column to use for return calculations.
        risk_free_rate (pd.Series, optional): A pandas Series object containing the risk free rate data.
        This is used to calculate the excess return and excess volatility. Defaults to pd.Series().
        include_dividends (bool, optional): A boolean representing whether to include dividends in the
        historical data. Defaults to True.
        ohlc_divide_by (int, optional): An integer representing the value to divide the OHLC data by.
        This is useful if the OHLC data is presented in percentages or similar. Defaults to None.
        show_errors (bool, optional): A boolean representing whether to show errors. Defaults to True.
        sleep_timer (bool, optional): A boolean representing whether to introduce a sleep timer to prevent
        rate limit errors. Defaults to True.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    # Additional data is collected to ensure return calculations are correct
    end_date_value = (
        datetime.strptime(end, "%Y-%m-%d") if end is not None else datetime.today()
    )

    if start is not None:
        # Additional data is collected to ensure return calculations are correct
        start_date_value = datetime.strptime(start, "%Y-%m-%d")

        if start_date_value > end_date_value:
            raise ValueError(
                f"Start date ({start_date_value}) must be before end date ({end_date_value}))"
            )
    else:
        start_date_value = datetime.now() - timedelta(days=5)

        if start_date_value > end_date_value:
            start_date_value = end_date_value - timedelta(days=5)

    end_date_string = end_date_value.strftime("%Y-%m-%d")
    start_date_string = start_date_value.strftime("%Y-%m-%d")

    historical_data_url = (
        f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{ticker}?"
        f"from={start_date_string}&to={end_date_string}&apikey={api_key}"
    )

    try:
        historical_data = helpers.get_financial_data(
            url=historical_data_url, sleep_timer=sleep_timer, raw=True
        )

        historical_data = pd.DataFrame(historical_data).set_index("date")
    except (HTTPError, KeyError, ValueError, URLError, RemoteDisconnected):
        return pd.DataFrame(historical_data)

    historical_data = historical_data.sort_index()

    if historical_data.loc[start_date_string:end_date_string].empty:
        print(f"The given start and end date result in no data found for {ticker}")
        return pd.DataFrame()

    if interval in ["1min", "5min", "15min", "30min"]:
        frequency = "min"
    elif interval in ["1hour", "4hour"]:
        frequency = "H"
    else:
        raise ValueError(
            f"Interval {interval} is not valid. It should be either 1min, 5min, 15min, 30min, 1hour or 4hour."
        )

    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data.index = historical_data.index.to_period(freq=frequency)

    historical_data = historical_data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    historical_data = historical_data[["Open", "High", "Low", "Close", "Volume"]]

    historical_data = historical_data.loc[
        ~historical_data.index.duplicated(keep="first")
    ]

    historical_data = enrich_historical_data(
        historical_data=historical_data,
        start=start,
        end=end,
        return_column=return_column,
        risk_free_rate=pd.DataFrame(),
    )

    return historical_data


def enrich_historical_data(
    historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
):
    """
    Retrieves enriched historical stock data for the given ticker(s) from Yahoo! Finance API for
    a specified period. It calculates the following:

        - Return: The return for the given period.
        - Volatility: The volatility for the given period.
        - Excess Return: The excess return for the given period.
        - Excess Volatility: The excess volatility for the given period.
        - Cumulative Return: The cumulative return for the given period.

    The return is calculated as the percentage change in the given return column and the excess return
    is calculated as the percentage change in the given return column minus the risk free rate.

    The volatility is calculated as the standard deviation of the daily returns and the excess volatility
    is calculated as the standard deviation of the excess returns.

    The cumulative return is calculated as the cumulative product of the percentage change in the given
    return column.

    Args:
        historical_data (pd.DataFrame): A pandas DataFrame object containing the historical stock data
        for the given ticker(s).
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        return_column (str, optional): A string representing the column to use for return calculations.
        risk_free_rate (pd.Series, optional): A pandas Series object containing the risk free rate data.
        This is used to calculate the excess return and excess volatility. Defaults to pd.Series().


    Returns:
        pd.DataFrame: A pandas DataFrame object containing the enriched historical stock data for the given ticker(s).
    """

    historical_data["Return"] = historical_data[return_column].ffill().pct_change()

    historical_data["Volatility"] = historical_data.loc[start:end, "Return"].std()

    if not risk_free_rate.empty:
        try:
            historical_data["Excess Return"] = historical_data["Return"].sub(
                risk_free_rate["Adj Close"]
            )

            historical_data["Excess Volatility"] = historical_data.loc[
                start:end, "Excess Return"
            ].std()
        except ValueError as error:
            print(
                f"Not able to calculate excess return and excess volatility due to {error}."
            )
            historical_data["Excess Return"] = 0
            historical_data["Excess Volatility"] = 0

    historical_data["Cumulative Return"] = 1

    adjusted_return = historical_data.loc[start:end, "Return"].copy()

    with contextlib.suppress(IndexError):
        adjusted_return.iloc[0] = 0

    historical_data["Cumulative Return"] = pd.Series(np.nan).astype(float)

    historical_data.loc[start:end, "Cumulative Return"] = (
        1.0 + adjusted_return
    ).cumprod()

    return historical_data


def convert_daily_to_other_period(
    period: str,
    daily_historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    risk_free_rate: pd.Series = pd.DataFrame(),
    rounding: int | None = None,
):
    """
    Converts daily historical data to another period which can be:
        - Weekly
        - Monthly
        - Quarterly
        - Yearly

    It calculates the following:
        - Return: The return for the given period.
        - Volatility: The volatility for the given period.
        - Excess Return: The excess return for the given period.
        - Excess Volatility: The excess volatility for the given period.
        - Cumulative Return: The cumulative return for the given period.

    Args:
        daily_historical_data (pd.DataFrame): A DataFrame containing daily historical data.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the yearly historical stock data.
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    period_translation = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }
    volatility_window_translation = {
        "weekly": 252 / 52,
        "monthly": 252 / 12,
        "quarterly": 252 / 4,
        "yearly": 252,
    }

    if period not in ["weekly", "monthly", "quarterly", "yearly"]:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    period_str = period_translation[period]
    volatility_window = volatility_window_translation[period]

    daily_historical_data.index.name = "Date"
    dates = daily_historical_data.index.asfreq(period_str)
    daily_historical_data = daily_historical_data.reset_index()
    period_historical_data = daily_historical_data.groupby(dates).transform("last")

    if "Dividends" in period_historical_data:
        period_historical_data["Dividends"] = (
            daily_historical_data["Dividends"].groupby(dates).transform("sum")
        )

    period_historical_data["Date"] = period_historical_data["Date"]
    period_historical_data = period_historical_data.drop_duplicates().set_index("Date")
    period_historical_data.index = pd.PeriodIndex(
        period_historical_data.index, freq=period_str
    )

    if "Return" in period_historical_data:
        period_historical_data["Return"] = (
            period_historical_data["Adj Close"]
            / period_historical_data["Adj Close"].shift()
            - 1
        )

        # Volatility is calculated as the daily volatility multiplied by the
        # square root of the number of trading days (252)
        period_historical_data["Volatility"] = daily_historical_data["Return"].groupby(
            dates
        ).std() * np.sqrt(volatility_window)

        if not risk_free_rate.empty:
            period_historical_data["Excess Return"] = period_historical_data[
                "Return"
            ].sub(risk_free_rate["Adj Close"], axis=0)

            period_historical_data["Excess Volatility"] = daily_historical_data[
                "Excess Return"
            ].groupby(dates).std() * np.sqrt(volatility_window)

    if "Cumulative Return" in period_historical_data:
        if start:
            start = pd.Period(start).asfreq(period_str)

            if start < period_historical_data.index[0]:
                start = period_historical_data.index[0]
        if end:
            end = pd.Period(end).asfreq(period_str)

            if end > period_historical_data.index[-1]:
                end = period_historical_data.index[-1]

        adjusted_return = period_historical_data.loc[start:end, "Return"].copy()
        adjusted_return.iloc[0] = 0

        period_historical_data["Cumulative Return"] = (1 + adjusted_return).cumprod()
        period_historical_data["Cumulative Return"] = period_historical_data[
            "Cumulative Return"
        ].fillna(1)

    period_historical_data = period_historical_data.sort_index()

    if rounding:
        period_historical_data = period_historical_data.round(rounding)

    return period_historical_data.fillna(0)


def get_historical_statistics(
    tickers: list[str] | str,
    api_key: str | None = None,
    progress_bar: bool = True,
    show_errors: bool = False,
    tqdm_message: str = "Obtaining historical statistics",
):
    """
    Retrieves statistics about each ticker's historical data. This is useful to understand why certain
    tickers might fluctuate more than others as it could be due to local regulations or the currency the instrument
    is denoted in. It returns:

        - Currency: The currency the instrument is denoted in.
        - Symbol: The symbol of the instrument.
        - Exchange Name: The name of the exchange the instrument is listed on.
        - Instrument Type: The type of instrument.
        - First Trade Date: The date the instrument was first traded.
        - Regular Market Time: The time the instrument is traded.
        - GMT Offset: The GMT offset.
        - Timezone: The timezone the instrument is traded in.
        - Exchange Timezone Name: The name of the timezone the instrument is traded in.

    It attempts to collect data from Yahoo Finance first and if that fails it will attempt to collect data from
    FinancialModelingPrep. If both fail it will return an empty DataFrame for the ticker. Generally Yahoo Finance
    should be sufficient but for delisted companies, only FinancialModelingPrep will offer data.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        api_key (str, optional): The API key to use to retrieve the data from FinancialModelingPrep.
        progress_bar (bool, optional): A boolean representing whether to show a progress bar. Defaults to True.
        show_errors (bool, optional): A boolean representing whether to show errors. Defaults to True.
        tqdm_message (str, optional): A string representing the message to show in the progress bar.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the statistics for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the statistics as the second level.
    """

    def worker(ticker, historical_statistics_dict):
        historical_statistics = pd.DataFrame()

        if historical_statistics.empty:
            historical_statistics = get_historical_statistics_from_yahoo_finance(
                ticker=ticker
            )

        if api_key and historical_statistics.empty:
            historical_statistics = (
                get_historical_statistics_from_financial_modeling_prep(
                    ticker=ticker,
                    api_key=api_key,
                )
            )

        if historical_statistics.empty:
            no_data.append(ticker)
        if not historical_statistics.empty:
            historical_statistics_dict[ticker] = historical_statistics

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    ticker_list_iterator = (
        tqdm(ticker_list, desc=tqdm_message)
        if (ENABLE_TQDM & progress_bar)
        else ticker_list
    )

    historical_statistics_dict: dict[str, pd.DataFrame] = {}
    no_data: list[str] = []
    threads = []

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, historical_statistics_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    historical_statistics_dict = (
        helpers.check_for_error_messages(historical_statistics_dict)
        if show_errors
        else historical_statistics_dict
    )

    historical_statistics = pd.concat(historical_statistics_dict, axis=1)

    return historical_statistics, no_data


def get_historical_statistics_from_financial_modeling_prep(
    ticker: str, api_key: str
) -> pd.Series:
    """
    Retrieve statistics about each ticker's historical data. This is especially useful to understand why certain
    tickers might fluctuate more than others as it could be due to local regulations or the currency the instrument
    is denoted in. It returns:

        - Currency: The currency the instrument is denoted in.
        - Symbol: The symbol of the instrument.
        - Exchange Name: The name of the exchange the instrument is listed on.
        - Instrument Type: The type of instrument.
        - First Trade Date: The date the instrument was first traded.
        - Regular Market Time: The time the instrument is traded.
        - GMT Offset: The GMT offset.
        - Timezone: The timezone the instrument is traded in.
        - Exchange Timezone Name: The name of the timezone the instrument is traded in.

    Please note that it follows the same format as the statistics retrieved from Yahoo Finance however not
    all information is available from FinancialModelingPrep and therefore some values will be NaN.

    Args:
        ticker (str): the ticker to retrieve statistics for.
        api_key (str): the API key to use to retrieve the data.

    Returns:
        pd.Series: A Sries containing the statistics for the given ticker.
    """
    profile, _ = fundamentals_model.get_profile(
        tickers=ticker, api_key=api_key, progress_bar=False, report_missing=False
    )

    profile_df = pd.Series(
        [np.nan] * 9,
        index=[
            "Currency",
            "Symbol",
            "Exchange Name",
            "Instrument Type",
            "First Trade Date",
            "Regular Market Time",
            "GMT Offset",
            "Timezone",
            "Exchange Timezone Name",
        ],
        dtype=object,
    )

    if not profile.empty:
        profile = profile.loc[:, ticker]

        profile_df.loc["Currency"] = profile.loc["Currency"]
        profile_df.loc["Symbol"] = profile.loc["Symbol"]
        profile_df.loc["Exchange Name"] = profile.loc["Exchange Short Name"]
        profile_df.loc["IPO Date"] = profile.loc["IPO Date"]

    return profile_df


def get_historical_statistics_from_yahoo_finance(ticker: str) -> pd.Series:
    """
    Retrieve statistics about each ticker's historical data. This is especially useful to understand why certain
    tickers might fluctuate more than others as it could be due to local regulations or the currency the instrument
    is denoted in. It returns:

        - Currency: The currency the instrument is denoted in.
        - Symbol: The symbol of the instrument.
        - Exchange Name: The name of the exchange the instrument is listed on.
        - Instrument Type: The type of instrument.
        - First Trade Date: The date the instrument was first traded.
        - Regular Market Time: The time the instrument is traded.
        - GMT Offset: The GMT offset.
        - Timezone: The timezone the instrument is traded in.
        - Exchange Timezone Name: The name of the timezone the instrument is traded in.

    Args:
        ticker (str): the ticker to retrieve statistics for.

    Returns:
        pd.Series: A Sries containing the statistics for the given ticker.
    """
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=None",
        timeout=60,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit"
            "/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        },
    )

    if response.status_code == 200:  # noqa
        data = response.json()

        try:
            statistics = data["chart"]["result"][0]["meta"]

            for timestamp_data in ["firstTradeDate", "regularMarketTime"]:
                if timestamp_data in statistics and statistics[timestamp_data]:
                    timestamp = (
                        datetime.fromtimestamp(0)
                        + timedelta(seconds=statistics[timestamp_data])
                    ).strftime("%Y-%m-%d")
                    statistics[timestamp_data] = timestamp

        except (KeyError, ValueError):
            return pd.DataFrame()

        columns = {
            "currency": "Currency",
            "symbol": "Symbol",
            "exchangeName": "Exchange Name",
            "instrumentType": "Instrument Type",
            "firstTradeDate": "First Trade Date",
            "regularMarketTime": "Regular Market Time",
            "gmtoffset": "GMT Offset",
            "timezone": "Timezone",
            "exchangeTimezoneName": "Exchange Timezone Name",
        }

        stats_df = pd.Series(statistics)
        stats_df = stats_df.rename(index=columns)
        stats_df = stats_df.loc[
            [column for column in columns.values() if column in stats_df.index]
        ]

        return stats_df

    return pd.DataFrame()
