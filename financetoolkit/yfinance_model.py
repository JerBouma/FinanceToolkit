"""YFinance Module"""

__docformat__ = "google"

import warnings
from datetime import datetime, timedelta
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from financetoolkit import helpers
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


def get_financial_statement(
    ticker: str, statement: str, quarter: bool = False, fallback: bool = False
):
    """
    Retrieves a specific financial statement (balance sheet, income statement, or cash flow statement)
    for a given stock ticker using the yfinance library.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL" for Apple).
        statement (str): The type of financial statement to retrieve.
                         Must be one of 'balance', 'income', or 'cashflow'.
        quarter (bool, optional): If True, retrieves quarterly data.
                                  If False, retrieves yearly data. Defaults to False.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the requested financial statement.
                      The columns are periods (yearly or quarterly), and the rows are financial items.
                      Returns an empty DataFrame if the data cannot be retrieved or if the
                      ticker is invalid.
    """
    period = "quarterly" if quarter else "yearly"

    if statement not in ["balance", "income", "cashflow"]:
        raise ValueError(
            "Please choose either 'balance', 'income', or "
            "cashflow' for the statement parameter."
        )

    # Create a ticker object from yfinance
    ticker_info = yf.Ticker(ticker)

    # Get the requested financial statement
    try:
        if statement == "balance":
            # Get balance sheet
            financial_statement = ticker_info.get_balance_sheet(freq=period)
        elif statement == "income":
            # Get income statement
            financial_statement = ticker_info.get_income_stmt(freq=period)
        elif statement == "cashflow":
            # Get cash flow statement
            financial_statement = ticker_info.get_cash_flow(freq=period)
        else:
            raise ValueError(
                "Please choose either 'balance', 'income', or "
                "cashflow' for the statement parameter."
            )
    except (
        HTTPError,
        URLError,
        RemoteDisconnected,
        IndexError,
        AttributeError,
    ):
        return pd.DataFrame()
    except yf.exceptions.YFRateLimitError:
        error_code = (
            "YFINANCE RATE LIMIT REACHED FALLBACK"
            if fallback
            else "YFINANCE RATE LIMIT REACHED"
        )
        return pd.DataFrame(columns=[error_code])

    if financial_statement.empty:
        error_code = (
            "YFINANCE RATE LIMIT OR NO DATA FOUND FALLBACK"
            if fallback
            else "YFINANCE RATE LIMIT OR NO DATA FOUND"
        )
        return pd.DataFrame(columns=[error_code])

    # yfinance returns the statements with dates as columns and items as rows
    # Convert dates to period format
    financial_statement.columns = pd.PeriodIndex(
        financial_statement.columns, freq="Q" if quarter else "Y"
    )

    if financial_statement.columns.duplicated().any():
        financial_statement = financial_statement.loc[
            :, ~financial_statement.columns.duplicated()
        ]

    # Check for NaN values and fill them with 0
    if financial_statement.isna().to_numpy().any():
        financial_statement = financial_statement.infer_objects(copy=False).fillna(0)

    return financial_statement


def get_historical_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
    divide_ohlc_by: int | float | None = None,
    fallback: bool = False,
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
    else:
        end_date = datetime.today()
        end = end_date.strftime("%Y-%m-%d")

    if start is not None:
        # Additional data is collected to ensure return calculations are correct
        start_date = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1 * 365)

        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date}))"
            )
    else:
        start_date = datetime.now() - timedelta(days=10 * 365)
        start = start_date.strftime("%Y-%m-%d")

        if start_date > end_date:
            start_date = end_date - timedelta(days=10 * 365)

    if interval in ["yearly", "quarterly"]:
        interval = "1d"

    try:

        historical_data = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval=interval,
            actions=True,
            auto_adjust=True,
            repair=True,
        )

        # Due to an odd error, it can sometimes occur that the columns are duplicated
        # which is why a check is performed here to ensure these don't stay in the DataFrame
        historical_data = historical_data.loc[:, ~historical_data.columns.duplicated()]

        if "Adj Close" not in historical_data and historical_data.columns.nlevels == 1:
            historical_data.loc[:, "Adj Close"] = historical_data.loc[
                :, "Close"
            ].to_numpy()

    except (HTTPError, URLError, RemoteDisconnected, IndexError):
        return pd.DataFrame()
    except yf.exceptions.YFRateLimitError:
        error_code = "YFINANCE RATE LIMIT REACHED" + " FALLBACK" if fallback else ""
        return pd.DataFrame(columns=[error_code])

    if not historical_data.empty and historical_data.loc[start:end].empty:
        logger.warning(
            "The given start and end date result in no data found for %s", ticker
        )
        return pd.DataFrame()

    historical_data.index = pd.to_datetime(historical_data.index)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        historical_data.index = historical_data.index.to_period(freq="D")

    if divide_ohlc_by:
        # Set divide by zero and invalid value warnings to ignore as it is fine that
        # dividing NaN by divide_ohlc_by results in NaN
        np.seterr(divide="ignore", invalid="ignore")
        # In case tickers are presented in percentages or similar
        historical_data = historical_data.div(divide_ohlc_by)

    historical_data = historical_data.loc[
        ~historical_data.index.duplicated(keep="first")
    ]

    if "Stock Splits" in historical_data and "Capital Gains" in historical_data:
        historical_data = historical_data.drop(
            columns=["Stock Splits", "Capital Gains"]
        )
    elif "Stock Splits" in historical_data:
        historical_data = historical_data.drop(columns=["Stock Splits"])
    elif "Capital Gains" in historical_data:
        historical_data = historical_data.drop(columns=["Capital Gains"])

    if "Dividends" not in historical_data:
        # If there are no dividends, create a column with 0.0 values
        historical_data["Dividends"] = 0.0

    historical_data = historical_data[
        ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends"]
    ]

    historical_data = helpers.enrich_historical_data(
        historical_data=historical_data,
        start=start,
        end=end,
        return_column=return_column,
        risk_free_rate=risk_free_rate,
    )

    return historical_data


def get_historical_statistics(ticker: str) -> pd.Series:
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
