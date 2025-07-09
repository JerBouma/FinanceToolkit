"""Historical Module"""

__docformat__ = "google"

import importlib.util
import threading
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from financetoolkit import fmp_model, yfinance_model
from financetoolkit.utilities import error_model, logger_model

logger = logger_model.get_logger()

# Check if yfinance is installed
yf_spec = importlib.util.find_spec("yfinance")
ENABLE_YFINANCE = yf_spec is not None

# pylint: disable=too-many-locals,unsubscriptable-object,too-many-lines

TREASURY_LIMIT = 90

INTERVAL_STR = {
    "1min": "min",
    "5min": "min",
    "15min": "min",
    "30min": "min",
    "1hour": "h",
    "4hour": "h",
    "1d": "D",
}


def get_historical_data(
    tickers: list[str] | str,
    api_key: str | None = None,
    enforce_source: str | None = None,
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
    user_subscription: str = "Free",
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
        index=pd.PeriodIndex(pd.date_range(start, end), freq=INTERVAL_STR[interval]),
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

    def worker(ticker, historical_data_dict, historical_data_error_dict):
        historical_data = pd.DataFrame()
        attempted_fmp = False

        if api_key and interval in ["1min", "5min", "15min", "30min", "1hour", "4hour"]:
            historical_data = fmp_model.get_intraday_data(
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
            if api_key and enforce_source in [None, "FinancialModelingPrep"]:
                historical_data = fmp_model.get_historical_data(
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

                attempted_fmp = True

            if (
                enforce_source != "FinancialModelingPrep"
                and historical_data.empty
                and ENABLE_YFINANCE
            ):
                historical_data = yfinance_model.get_historical_data(
                    ticker=ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    return_column=return_column,
                    risk_free_rate=risk_free_rate,
                    divide_ohlc_by=divide_ohlc_by,
                    fallback=attempted_fmp,
                )

                if not historical_data.empty:
                    yf_tickers.append(ticker)

        if historical_data.empty:
            no_data.append(ticker)
            historical_data_error_dict[ticker] = historical_data
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
        tqdm(ticker_list, desc=tqdm_message) if progress_bar else ticker_list
    )

    historical_data_dict: dict[str, pd.DataFrame] = {}
    historical_data_error_dict: dict[str, pd.DataFrame] = {}
    fmp_tickers: list[str] = []
    yf_tickers: list[str] = []
    no_data: list[str] = []
    threads = []

    for ticker in ticker_list_iterator:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, historical_data_dict, historical_data_error_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if show_errors:
        error_model.check_for_error_messages(
            dataset_dictionary=historical_data_error_dict,
            user_subscription=user_subscription,
        )

    if fmp_tickers and yf_tickers and show_ticker_seperation:
        logger.info(
            "The following tickers acquired historical data from FinancialModelingPrep: %s",
            ", ".join(fmp_tickers),
        )
        logger.info(
            "The following tickers acquired historical data from YahooFinance: %s",
            ", ".join(yf_tickers),
        )

    if (
        yf_tickers
        and not fmp_tickers
        and enforce_source == "FinancialModelingPrep"
        and show_errors
    ):
        logger.warning(
            "No data found using FinancialModelingPrep, this is usually due to Bandwidth "
            "API limits or usage of the Free plan.\n"
            "Therefore data was retrieved from YahooFinance instead for: %s",
            ", ".join(yf_tickers),
        )

    if no_data and show_errors:
        if not ENABLE_YFINANCE:
            logger.info(
                "Due to a missing optional dependency (yfinance) and your current FinancialModelingPrep plan, "
                "data for the following tickers could not be acquired: %s\n"
                "Enable this functionality by using:\033[1m pip install yfinance\033[0m",
                ", ".join(no_data),
            )
        else:
            logger.warning(
                "No data found for the following tickers: %s", ", ".join(no_data)
            )

    if len(historical_data_dict) == 0:
        # Fill the DataFrame with zeros to ensure the DataFrame is returned
        # even if no data is found. This is mostly applicable when nothing
        # can be found at all.
        for ticker in tickers:
            historical_data_dict[ticker] = empty_historical_data

    reorder_tickers = [ticker for ticker in tickers if ticker in historical_data_dict]

    if historical_data_dict and len(no_data) != len(tickers):
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

    return pd.DataFrame(), no_data


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

        if not risk_free_rate.empty and risk_free_rate["Adj Close"].sum() != 0:
            period_historical_data["Excess Return"] = period_historical_data[
                "Return"
            ].sub(risk_free_rate["Adj Close"], axis=0)

            period_historical_data["Excess Volatility"] = daily_historical_data[
                "Excess Return"
            ].groupby(dates).std() * np.sqrt(volatility_window)

    if "Cumulative Return" in period_historical_data:
        if start:
            start = max(
                pd.Period(start).asfreq(period_str), period_historical_data.index[0]
            )
        if end:
            end = min(
                pd.Period(end).asfreq(period_str), period_historical_data.index[-1]
            )

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
    user_subscription: str = "Free",
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
            historical_statistics = yfinance_model.get_historical_statistics(
                ticker=ticker
            )

        if api_key and historical_statistics.empty:
            historical_statistics = fmp_model.get_historical_statistics(
                ticker=ticker,
                api_key=api_key,
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
        tqdm(ticker_list, desc=tqdm_message) if progress_bar else ticker_list
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
        error_model.check_for_error_messages(
            dataset_dictionary=historical_statistics_dict,
            user_subscription=user_subscription,
        )
        if show_errors
        else historical_statistics_dict
    )

    if historical_statistics_dict:
        historical_statistics = pd.concat(historical_statistics_dict, axis=1)

        return historical_statistics, no_data

    return pd.DataFrame(), no_data
