"""YFinance Module"""

__docformat__ = "google"

import warnings
from datetime import datetime, timedelta
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import yfinance as yf

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.requests_model import get_request

logger = logger_model.get_logger()

# The reporting currency of a company changes almost never, so it is resolved once per
# ticker per process rather than on every statement request.
REPORTED_CURRENCY_CACHE: dict[str, str] = {}


def get_financial_statement(
    ticker: str,
    statement: str,
    quarter: bool = False,
    fallback: bool = False,
    fiscal_year_adjustments: dict | None = None,
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
        fallback (bool, optional): Whether this call follows an unsuccessful attempt at
                                   FinancialModelingPrep, which changes the error reported.
                                   Defaults to False.
        fiscal_year_adjustments (dict | None, optional): A registry that collects every reporting
                                   period whose calendar label differs from its fiscal label,
                                   keyed by ticker. Defaults to None.

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

    # yfinance returns dates as columns and items as rows; convert to periods. A fiscal
    # period is labelled with the calendar period holding most of it, at both
    # frequencies and identically to the FinancialModelingPrep path.
    financial_statement.columns = helpers.convert_period_end_dates_to_calendar_periods(
        period_end_dates=pd.DatetimeIndex(financial_statement.columns),
        quarter=quarter,
        ticker=ticker,
        fiscal_year_adjustments=fiscal_year_adjustments,
    )

    if financial_statement.columns.duplicated().any():
        financial_statement = financial_statement.loc[
            :, ~financial_statement.columns.duplicated()
        ]

    # Left as NaN, not filled with 0, matching the Toolkit-wide convention for unreported line items.
    if financial_statement.isna().to_numpy().any():
        financial_statement = financial_statement.infer_objects(copy=False)

    return financial_statement


def get_statistics_statement(
    ticker: str,
    periods: pd.Index,
) -> pd.DataFrame:
    """
    Builds the statistics statement for a Yahoo Finance sourced ticker, mirroring the
    statistics that FinancialModelingPrep returns alongside its financial statements.

    The only statistic Yahoo Finance publishes that the Toolkit depends on is the
    currency the financial statements are reported in, exposed as ``financialCurrency``
    on the quote summary. Without it a Yahoo sourced ticker has no `Reported Currency`,
    which is what the currency conversion compares against the currency the instrument
    trades in, so those tickers were never converted at all and their statements were
    left in a different currency from their own price history.

    The reporting currency is a property of the company rather than of a period, so the
    same value is repeated across every period, matching the shape FinancialModelingPrep
    returns.

    Args:
        ticker (str): the ticker to retrieve the statistics for.
        periods (pd.Index): the reporting periods to report the statistics against,
            normally the columns of the financial statement of the same ticker.

    Returns:
        pd.DataFrame: A DataFrame with the Yahoo Finance statistics names as index and
        the given periods as columns. Empty when Yahoo Finance does not report a
        reporting currency for the ticker.
    """
    reported_currency = get_reported_currency(ticker)

    if not reported_currency or len(periods) == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        [[reported_currency] * len(periods)],
        index=["financialCurrency"],
        columns=periods,
    )


def get_reported_currency(ticker: str) -> str:
    """
    Retrieves the currency a company reports its financial statements in from Yahoo
    Finance. This is not necessarily the currency the instrument trades in: Shell
    reports in USD while its London listing trades in GBp, and comparing a statement to
    a price without accounting for that compares two different currencies.

    The reporting currency only changes when a company changes its reporting currency,
    so it is cached per ticker both in memory and in the incremental cache.

    Args:
        ticker (str): the ticker to retrieve the reporting currency for.

    Returns:
        str: The ISO currency code the statements are reported in, or an empty string
        when Yahoo Finance does not report one.
    """
    if ticker in REPORTED_CURRENCY_CACHE:
        return REPORTED_CURRENCY_CACHE[ticker]

    cache = get_active_cache()

    if cache is not None:
        cached_currency = cache.get(
            source=policy_model.YAHOO_FINANCE,
            dataset="reported_currency",
            entity=ticker,
        )

        if cached_currency is not None:
            REPORTED_CURRENCY_CACHE[ticker] = cached_currency
            return cached_currency

    try:
        information = yf.Ticker(ticker).get_info() or {}
    except (
        HTTPError,
        URLError,
        RemoteDisconnected,
        IndexError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        yf.exceptions.YFRateLimitError,
    ):
        return ""

    # financialCurrency is the statement currency. currency is the trading currency and
    # is only a fallback, since for most listings the two are in fact the same.
    reported_currency = str(
        information.get("financialCurrency") or information.get("currency") or ""
    )

    REPORTED_CURRENCY_CACHE[ticker] = reported_currency

    if cache is not None and reported_currency:
        cache.set(
            source=policy_model.YAHOO_FINANCE,
            dataset="reported_currency",
            entity=ticker,
            data=reported_currency,
        )

    return reported_currency


def get_historical_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    divide_ohlc_by: int | float | None = None,
    fallback: bool = False,
):
    """
    Retrieves historical stock data for the given ticker(s) from Yahoo! Finance API for a specified period.
    If start and/or end date are not provided, it defaults to 10 years from the current date.

    Args:
        ticker (str): The ticker symbol to retrieve data for.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        interval (str, optional): A string representing the interval to retrieve data for.
        return_column (str, optional): A string representing the column to use for return calculations.
        divide_ohlc_by (int or float, optional): A number to divide the OHLC data by. Defaults to None.
        fallback (bool, optional): Whether this call follows an unsuccessful attempt at
            FinancialModelingPrep, which changes the error reported. Defaults to False.

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

    # The widened window is what is actually requested, matching the FinancialModelingPrep
    # path. Fetching only the requested window would leave the first return in range NaN
    # because it has no preceding close to compare against.
    start_date_string = start_date.strftime("%Y-%m-%d")
    end_date_string = end_date.strftime("%Y-%m-%d")

    try:

        # auto_adjust=False matches FMP; True made Close disagree by the dividend history (1.8% median, AAPL 2022-2023).
        # yfinance's split-repair misfires on synthetic instruments (^IRX 100x, CL=F 10,000x after oil went negative).
        repair_splits = not (ticker.startswith("^") or "=" in ticker)

        historical_data = yf.Ticker(ticker).history(
            start=start_date_string,
            end=end_date_string,
            interval=interval,
            actions=True,
            auto_adjust=False,
            repair=repair_splits,
        )

        # Columns can occasionally be duplicated, so they are checked and dropped.
        historical_data = historical_data.loc[:, ~historical_data.columns.duplicated()]

        if "Adj Close" not in historical_data and historical_data.columns.nlevels == 1:
            historical_data.loc[:, "Adj Close"] = historical_data.loc[
                :, "Close"
            ].to_numpy()

    except (HTTPError, URLError, RemoteDisconnected, IndexError):
        return pd.DataFrame()
    except yf.exceptions.YFRateLimitError:
        error_code = "YFINANCE RATE LIMIT REACHED" + (" FALLBACK" if fallback else "")
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
        # NaN divided by divide_ohlc_by is fine, so those warnings are ignored.
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

    These describe the instrument itself (its currency, exchange and listing date)
    rather than its price, so they change very rarely and are cached per ticker.

    Args:
        ticker (str): the ticker to retrieve statistics for.

    Returns:
        pd.Series: A Sries containing the statistics for the given ticker.
    """
    cache = get_active_cache()

    if cache is not None:
        cached_statistics = cache.get(
            source=policy_model.YAHOO_FINANCE,
            dataset="historical_statistics",
            entity=ticker,
        )

        if cached_statistics is not None:
            return cached_statistics

    response = get_request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=None",
        timeout=60,
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

        if cache is not None and not stats_df.empty:
            cache.set(
                source=policy_model.YAHOO_FINANCE,
                dataset="historical_statistics",
                entity=ticker,
                data=stats_df,
            )

        return stats_df

    return pd.DataFrame()
