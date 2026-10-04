"""FMP Module"""

__docformat__ = "google"


import hashlib
import importlib.util
import random
import time
from datetime import datetime, timedelta
from http.client import RemoteDisconnected
from io import StringIO
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import requests
from urllib3.exceptions import MaxRetryError

from financetoolkit import helpers
from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.utilities import error_model, logger_model
from financetoolkit.utilities.dataframe_model import to_dataframe
from financetoolkit.utilities.requests_model import get_request

logger = logger_model.get_logger()


# Check if yfinance is installed
yf_spec = importlib.util.find_spec("yfinance")
ENABLE_YFINANCE = yf_spec is not None

logger = logger_model.get_logger()

# pylint: disable=no-member,too-many-locals,too-many-lines

RETRY_LIMIT = 12

# Base delay for the exponential backoff on rate limits and refused connections. The
# delay doubles per attempt and carries random jitter: with 10 worker threads hitting
# the limit at the same time, a fixed sleep would wake them all together and re-collide
# on the very next attempt.
RETRY_BASE_DELAY_SECONDS = 2.5
RETRY_MAX_DELAY_SECONDS = 60.0


def determine_retry_delay(attempt: int) -> float:
    """
    Determines how long to wait before retry `attempt` (starting at 0), doubling the
    base delay per attempt up to a cap, with up to 50% random jitter added so that
    concurrent workers spread out instead of retrying in lockstep.

    Args:
        attempt (int): The zero-based retry attempt number.

    Returns:
        float: The number of seconds to sleep before the retry.
    """
    delay = min(RETRY_BASE_DELAY_SECONDS * 2**attempt, RETRY_MAX_DELAY_SECONDS)

    return delay * (1 + random.uniform(0, 0.5))  # noqa: S311


def get_financial_data(
    url: str,
    sleep_timer: bool = True,
    raw: bool = False,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Collects the financial data from the FinancialModelingPrep API. This is a
    separate function to properly segregate the different types of errors that can occur.

    Args:
        url (str): The url to retrieve the data from.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
            if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to True.
        raw (bool): Whether to return the raw JSON data. Defaults to False.
        user_subscription (str): The subscription type of the user. Defaults to "Free". Used to determine retry logic
            on rate limits.

    Returns:
        pd.DataFrame or dict: A DataFrame containing the financial data, or a dictionary if raw=True.
            Returns an empty DataFrame with specific column names indicating errors like 'LIMIT REACH',
            'INVALID API KEY', etc., in case of API issues.
    """
    error_retry_counter = 0
    limit_retry_counter = 0

    while True:
        try:
            response = get_request(url, timeout=60)
            response.raise_for_status()

            if raw:
                return response.json()

            json_io = StringIO(response.text)

            financial_data = pd.read_json(json_io)

            return financial_data

        except (requests.exceptions.HTTPError, ValueError) as e:
            error_message = (
                e.response.text
                if isinstance(e, requests.exceptions.HTTPError)
                and e.response is not None
                else ""
            )

            if "Premium Query Parameter" in error_message:
                return pd.DataFrame(columns=["PREMIUM QUERY PARAMETER"])
            if "Exclusive Endpoint" in error_message:
                return pd.DataFrame(columns=["EXCLUSIVE ENDPOINT"])
            if "Special Endpoint" in error_message:
                return pd.DataFrame(columns=["SPECIAL ENDPOINT"])
            if "Premium Endpoint" in error_message:
                return pd.DataFrame(columns=["SPECIAL ENDPOINT"])
            if "Bandwidth Limit Reach" in error_message:
                return pd.DataFrame(columns=["BANDWIDTH LIMIT REACH"])
            if "Limit Reach" in error_message:
                if (
                    sleep_timer
                    and limit_retry_counter < RETRY_LIMIT
                    and user_subscription != "Free"
                ):
                    time.sleep(determine_retry_delay(limit_retry_counter))
                    limit_retry_counter += 1
                    continue

                return pd.DataFrame(columns=["LIMIT REACH"])
            if "US stocks only" in error_message:
                return pd.DataFrame(columns=["US STOCKS ONLY"])

            if "Invalid API KEY." in error_message:
                return pd.DataFrame(columns=["INVALID API KEY"])

            # Anything that is not one of the messages above is not retryable, without this the loop would fall through and hammer the endpoint indefinitely.  # noqa: E501
            logger.error(
                "The request to Financial Modeling Prep failed with an unrecognised error: %s",
                error_message or e,
            )

            return pd.DataFrame(columns=["REQUEST FAILED"])

        except (
            MaxRetryError,
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
        ):
            # Retry a refused connection up to RETRY_LIMIT times, then return empty.
            if error_retry_counter == RETRY_LIMIT:
                return pd.DataFrame(columns=["NO ERRORS"])

            time.sleep(determine_retry_delay(error_retry_counter))
            error_retry_counter += 1


PLAN_RESTRICTION_MESSAGES = (
    "PREMIUM QUERY PARAMETER",
    "EXCLUSIVE ENDPOINT",
    "NO DATA",
    "BANDWIDTH LIMIT REACH",
    "INVALID API KEY",
    "LIMIT REACH",
)


def determine_subscription_plan(api_key: str) -> tuple[str, bool]:
    """
    Probe the API key to establish which FinancialModelingPrep plan it belongs to.

    The plan governs the sleep timer and several other decisions, so it is needed
    before any real request is made. Because the answer only changes when a
    subscription changes, it is cached briefly rather than probed on every single
    Toolkit or Discovery construction. The key itself is never stored: the cache
    entry is identified by a digest of it, so two processes using the same key share
    the answer while the key stays out of the database.

    Args:
        api_key (str): The FinancialModelingPrep API key to probe.

    Returns:
        tuple[str, bool]: The plan ("Premium" or "Free"), and whether the key was
            rejected as invalid.
    """
    cache = get_active_cache()
    entity = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "no_key"

    if cache is not None:
        cached_plan = cache.get(
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset="subscription_plan",
            entity=entity,
        )

        if cached_plan is not None:
            return cached_plan["plan"], cached_plan["invalid_key"]

    determine_plan = get_financial_data(
        url=f"https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey={api_key}&limit=10",
        sleep_timer=False,
        user_subscription="Free",
    )

    plan = "Premium"
    invalid_key = False

    for option in PLAN_RESTRICTION_MESSAGES:
        if option in determine_plan:
            invalid_key = option == "INVALID API KEY"
            plan = "Free"
            break

    # A rate limited probe says nothing about the plan, so that answer is not stored.
    if cache is not None and "LIMIT REACH" not in determine_plan:
        cache.set(
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset="subscription_plan",
            entity=entity,
            data={"plan": plan, "invalid_key": invalid_key},
        )

    return plan, invalid_key


def get_financial_statement(
    ticker: str,
    statement: str = "",
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    sleep_timer: bool = True,
    user_subscription: str = "Free",
    fiscal_year_adjustments: dict | None = None,
) -> pd.DataFrame:
    """
    Retrieves financial statements (balance, income, or cash flow statements) for a single company ticker.

    Args:
        ticker (str): The company ticker.
        statement (str): The type of financial statement to retrieve. Can be "balance", "income", or "cash-flow".
        api_key (str): API key for the financial data provider.
        quarter (bool): Whether to retrieve quarterly data. Defaults to False (annual data).
        start_date (str | None): The start date to filter data with. Defaults to None.
        fiscal_year_adjustments (dict | None): A registry that collects every reporting period whose
            calendar label differs from its fiscal label, keyed by ticker. Defaults to None.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
            if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to True.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        pd.DataFrame: A DataFrame containing the financial statement data for the specified ticker.
                      The index represents the financial statement items, and the columns represent the dates/periods.
                      Returns an empty DataFrame if data retrieval fails or no data is found for the given parameters.
    """
    if statement == "balance":
        location = "balance-sheet-statement"
    elif statement == "income":
        location = "income-statement"
    elif statement == "cashflow":
        location = "cash-flow-statement"
    else:
        raise ValueError(
            "Please choose either 'balance', 'income', or "
            "cashflow' for the statement parameter."
        )

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    period = "quarter" if quarter else "annual"

    periods_to_fetch = 5  # Default limit

    if start_date and user_subscription != "Free":
        if quarter:
            # Convert dates to period objects
            start_period = pd.Period(pd.to_datetime(start_date), freq="Q")
            end_period = pd.Period(pd.to_datetime(datetime.today()), freq="Q")

            # Calculate number of quarters between dates
            periods_to_fetch = (
                (end_period.year - start_period.year) * 4
                + (end_period.quarter - start_period.quarter)
                + 1
            )
        else:
            # Calculate number of years between dates
            start_year = pd.to_datetime(start_date).year
            end_year = pd.to_datetime(datetime.today()).year
            periods_to_fetch = end_year - start_year + 1

    # Ensure we don't exceed the API's limit
    periods_to_fetch = min(periods_to_fetch, 9999) if user_subscription != "Free" else 5

    url = (
        f"https://financialmodelingprep.com/stable/{location}"
        f"?symbol={ticker}&period={period}&apikey={api_key}&"
        f"limit={periods_to_fetch}"
    )

    financial_statement = get_financial_data(
        url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
    )

    if not financial_statement.empty:
        financial_statement = financial_statement.drop("symbol", axis=1)

        # One day is deducted: a period reported as 2023-07-01 is really 2023Q2.
        financial_statement["date"] = pd.to_datetime(
            financial_statement["date"]
        ) - pd.offsets.Day(1)

        # A fiscal period is labelled with the calendar period holding most of it, at both frequencies, so yearly and quarterly output stay consistent.  # noqa: E501
        financial_statement["date"] = (
            helpers.convert_period_end_dates_to_calendar_periods(
                period_end_dates=financial_statement["date"],
                quarter=quarter,
                ticker=ticker,
                fiscal_year_adjustments=fiscal_year_adjustments,
            )
        )

        financial_statement = financial_statement.set_index("date").T

        if financial_statement.columns.duplicated().any():
            # Duplicate statements for one period are equal, so one copy can be dropped.
            financial_statement = financial_statement.loc[
                :, ~financial_statement.columns.duplicated()
            ]

    return financial_statement


def get_historical_data(
    ticker: str,
    api_key: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    include_dividends: bool = True,
    divide_ohlc_by: int | float | None = None,
    sleep_timer: bool = True,
    user_subscription: str = "Free",
):
    """
    Retrieves historical stock data for the given ticker from Financial Modeling Prep for a specified period.
    If start and/or end date are not provided, it defaults to 10 years from the current date.

    Note that when using a Free API key from FinancialModelingPrep it will be limited to 5 years.

    Args:
        ticker (str): The ticker symbol to retrieve data for.
        api_key (str): API key for the financial data provider.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        interval (str, optional): A string representing the interval to retrieve data for (e.g., '1d', '1wk').
            Defaults to '1d'. Note: FMP only supports daily data for this endpoint, 'yearly' and 'quarterly'
            will be converted to '1d'.
        return_column (str, optional): A string representing the column to use for return calculations.
            Defaults to 'Adj Close'.
        include_dividends (bool, optional): A boolean representing whether to include dividends in the
            historical data. Defaults to True.
        divide_ohlc_by (int | float | None, optional): A value to divide the OHLC data by.
            This is useful if the OHLC data is presented in percentages or similar. Defaults to None.
        sleep_timer (bool, optional): Whether to set a sleep timer when the rate limit is reached. Note that this
            only works if you have a Premium subscription (Starter or higher) from FinancialModelingPrep.
            Defaults to True.
        user_subscription (str): The subscription type of the user. Defaults to "Free". Used to determine retry logic
            on rate limits.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker.
                      The index of the DataFrame is the date of the data and the columns include OHLC, Volume,
                      Dividends (if requested), Return, and Cumulative Return.
                      Returns an empty DataFrame if data retrieval fails or no data is found for the given parameters.
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
        f"https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={ticker}&apikey={api_key}&from={start_date_string}&to={end_date_string}"
    )

    # The stable EOD endpoint no longer returns an adjusted close, so the dividend adjusted variant is queried separately to obtain it.  # noqa: E501
    adjusted_data_url = (
        f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
        f"?symbol={ticker}&apikey={api_key}&from={start_date_string}&to={end_date_string}"
    )

    dividend_url = (
        f"https://financialmodelingprep.com/stable/dividends"
        f"?symbol={ticker}&apikey={api_key}&limit={'99999' if user_subscription != 'Free' else '5'}"
    )

    try:
        historical_data = get_financial_data(
            url=historical_data_url,
            sleep_timer=sleep_timer,
            raw=True,
            user_subscription=user_subscription,
        )

        historical_data = pd.DataFrame(historical_data).set_index("date")
    except (HTTPError, KeyError, ValueError, URLError, RemoteDisconnected):
        return pd.DataFrame(historical_data)

    historical_data = historical_data.sort_index()

    if (
        not historical_data.empty
        and historical_data.loc[start_date_string:end_date_string].empty
    ):
        logger.warning(
            "The given start and end date result in no data found for %s", ticker
        )
        return pd.DataFrame()

    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data.index = pd.DatetimeIndex(historical_data.index).to_period(freq="D")

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

    try:
        adjusted_data = get_financial_data(
            url=adjusted_data_url,
            sleep_timer=sleep_timer,
            raw=True,
            user_subscription=user_subscription,
        )

        adjusted_data = pd.DataFrame(adjusted_data).set_index("date")
        adjusted_data.index = pd.to_datetime(adjusted_data.index).to_period(freq="D")
        adjusted_data = adjusted_data[~adjusted_data.index.duplicated(keep="first")]

        historical_data["Adj Close"] = adjusted_data["adjClose"]
    except (HTTPError, KeyError, ValueError, URLError, RemoteDisconnected):
        # Without the adjusted close, returns would be based on raw prices and would therefore exclude dividends, so this is reported rather than silently accepted.  # noqa: E501
        logger.warning(
            "No adjusted close data found for %s, falling back to the unadjusted close. "
            "Returns will not include dividends.",
            ticker,
        )
        historical_data["Adj Close"] = historical_data["Close"]

    historical_data = historical_data[
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    ]

    if divide_ohlc_by:
        # NaN divided by divide_ohlc_by is fine, so those warnings are ignored.
        np.seterr(divide="ignore", invalid="ignore")
        # In case tickers are presented in percentages or similar
        historical_data = historical_data.div(divide_ohlc_by)

    if include_dividends:
        try:
            dividends = get_financial_data(
                url=dividend_url,
                sleep_timer=sleep_timer,
                raw=True,
                user_subscription=user_subscription,
            )

            try:
                dividends_df = pd.DataFrame(dividends).set_index("date")

                if not dividends_df.empty:
                    dividends_df.index = pd.to_datetime(dividends_df.index)
                    dividends_df.index = pd.DatetimeIndex(dividends_df.index).to_period(
                        freq="D"
                    )
                    dividends_df = dividends_df[
                        ~dividends_df.index.duplicated(keep="first")
                    ]

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

    historical_data = helpers.enrich_historical_data(
        historical_data=historical_data,
        start=start,
        end=end,
        return_column=return_column,
    )

    return historical_data


def get_intraday_data(
    ticker: str,
    api_key: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1hour",
    return_column: str = "Close",
    sleep_timer: bool = True,
    user_subscription: str = "Free",
):
    """
    Retrieves intraday stock data for the given ticker from Financial Modeling Prep for a specified period.
    If start and/or end date are not provided, it defaults to 5 days from the current date.

    Note that when using a Free API key from FinancialModelingPrep it will be limited to 5 days.

    Args:
        ticker (str): The ticker symbol to retrieve data for.
        api_key (str): API key for the financial data provider.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None (5 days ago).
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None (today).
        interval (str, optional): A string representing the interval to retrieve data for (e.g., '1min', '1hour').
            Defaults to '1hour'. Valid intervals are '1min', '5min', '15min', '30min', '1hour', '4hour'.
        return_column (str, optional): A string representing the column to use for return calculations.
            Defaults to 'Close'.
        sleep_timer (bool, optional): Whether to set a sleep timer when the rate limit is reached. Note that
            this only works if you have a Premium subscription (Starter or higher) from FinancialModelingPrep.
                Defaults to True.
        user_subscription (str): The subscription type of the user. Defaults to "Free". Used to determine retry logic
            on rate limits.

    Raises:
        ValueError: If the start date is after the end date or if the interval is invalid.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the intraday stock data for the given ticker.
                      The index of the DataFrame is the timestamp of the data (as a Period object) and the columns include
                      OHLC, Volume, Log Return, and Cumulative Return.
                      Returns an empty DataFrame if data retrieval fails or no data is found for the given parameters.
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
        f"https://financialmodelingprep.com/stable/historical-chart/{interval}"
        f"?symbol={ticker}&from={start_date_string}&to={end_date_string}&apikey={api_key}"
    )

    try:
        historical_data = get_financial_data(
            url=historical_data_url,
            sleep_timer=sleep_timer,
            raw=True,
            user_subscription=user_subscription,
        )

        historical_data = pd.DataFrame(historical_data).set_index("date")
    except (HTTPError, KeyError, ValueError, URLError, RemoteDisconnected):
        return pd.DataFrame(historical_data)

    historical_data = historical_data.sort_index()

    if (
        not historical_data.empty
        and historical_data.loc[start_date_string:end_date_string].empty
    ):
        logger.warning(
            "The given start and end date result in no data found for %s", ticker
        )
        return pd.DataFrame()

    if interval in ["1min", "5min", "15min", "30min"]:
        frequency = "min"
    elif interval in ["1hour", "4hour"]:
        frequency = "h"
    else:
        raise ValueError(
            f"Interval {interval} is not valid. It should be either 1min, 5min, 15min, 30min, 1hour or 4hour."
        )

    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data.index = pd.DatetimeIndex(historical_data.index).to_period(
        freq=frequency
    )

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

    historical_data = helpers.enrich_historical_data(
        historical_data=historical_data,
        start=start,
        end=end,
        return_column=return_column,
    )

    return historical_data


def get_historical_statistics(ticker: str, api_key: str) -> pd.Series:
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
    profile, _ = get_profile(tickers=ticker, api_key=api_key)

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
        profile_df.loc["Exchange Name"] = profile.loc["Exchange"]
        profile_df.loc["IPO Date"] = profile.loc["IPO Date"]

    return profile_df


def get_revenue_segmentation(
    tickers: str | list[str],
    method: str = "",
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = False,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves revenue segmentation data (geographic or product) for one or multiple companies,
    and returns a DataFrame containing the data.

    Args:
        tickers (list[str] | str): A single ticker or a list of company tickers.
        method (str): The segmentation to retrieve, either "geographic" or "product".
        api_key (str): API key for the financial data provider.
        quarter (bool): Whether to retrieve quarterly data. Defaults to False (annual data).
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: A DataFrame containing the financial statement data and the list of
                      tickers without data. If only one ticker is provided, the
                      returned DataFrame will have a single column containing the data for that ticker. If multiple
                      tickers are provided, the returned DataFrame will have multiple columns, one for each ticker,
                      with the ticker symbol as the column name.
    """

    def worker(ticker):
        url = (
            f"https://financialmodelingprep.com/stable/{location}"
            f"?symbol={ticker}&period={period}&structure=flat&apikey={api_key}"
        )
        revenue_segmentation_json = get_financial_data(
            url=url,
            sleep_timer=sleep_timer,
            raw=True,
            user_subscription=user_subscription,
        )

        revenue_segmentation = pd.DataFrame()

        if not isinstance(revenue_segmentation_json, pd.DataFrame):
            try:
                for period_data in revenue_segmentation_json:
                    period_data_dict = {period_data["date"]: period_data["data"]}
                    revenue_segmentation = pd.concat(
                        [revenue_segmentation, pd.DataFrame(period_data_dict)], axis=1
                    )

                if quarter:
                    revenue_segmentation.columns = pd.PeriodIndex(
                        revenue_segmentation.columns, freq="Q"
                    )
                else:
                    revenue_segmentation.columns = pd.PeriodIndex(
                        revenue_segmentation.columns, freq="Y"
                    )

                if revenue_segmentation.columns.duplicated().any():
                    # Duplicate statements for one period are equal, so one copy can be dropped.
                    revenue_segmentation = revenue_segmentation.loc[
                        :, ~revenue_segmentation.columns.duplicated()
                    ]

                revenue_segmentation = revenue_segmentation.rename(index=naming)
                revenue_segmentation.index = [
                    index.lower().title() for index in revenue_segmentation.index
                ]

                # This groups items that have the same naming convention
                revenue_segmentation = revenue_segmentation.groupby(level=0).sum()

                return ticker, revenue_segmentation, True
            except (KeyError, ValueError):
                return ticker, revenue_segmentation, False

        # An error frame rather than JSON; kept so the error reporting can inspect it.
        return ticker, revenue_segmentation_json, False

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if method == "geographic":
        location = "revenue-geographic-segmentation"
        naming = {
            "U S": "United States",
            "U": "United States",
            "C N": "China",
            "N O": "North America",
            "Non Us": "Non-US",
            "Americas Segment": "Americas",
            "Europe Segment": "Europe",
            "Greater China Segment": "China",
            "Japan Segment": "Japan",
            "Rest of Asia Pacific Segment": "Asia Pacific",
            "Asia-Pacific": "Asia Pacific",
            "J P": "Japan",
            "North America Segment": "North America",
            "TAIWAN, PROVINCE OF CHINA": "Taiwan",
            "Segment, Geographical, Rest of the World, Excluding United States and United Kingdom": "Rest Of The World",
            "D E": "Germany",
        }
    elif method == "product":
        location = "revenue-product-segmentation"
        naming = {}
    else:
        raise ValueError(
            "Please choose either 'balance', 'income', or "
            "cashflow' for the statement parameter."
        )

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    period = "quarter" if quarter else "annual"

    logger.info(
        "Obtaining %s segmentation data for %d ticker(s)", method, len(ticker_list)
    )
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    revenue_segmentation_dict: dict = {ticker: data for ticker, data, _ in results}
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    revenue_segmentation_dict = error_model.check_for_error_messages(
        dataset_dictionary=revenue_segmentation_dict,
        required_subscription="Professional or Enterprise",
        user_subscription=user_subscription,
    )

    if revenue_segmentation_dict:
        revenue_segmentation_total = pd.concat(revenue_segmentation_dict, axis=0)

        try:
            revenue_segmentation_total = revenue_segmentation_total.astype(np.float64)
        except ValueError as error:
            logger.error(
                "Not able to convert DataFrame to float64 due to %s. This could result in"
                "issues when values are zero and is predominantly relevant for "
                "ratio calculations.",
                error,
            )

        revenue_segmentation_total = revenue_segmentation_total.sort_index(
            axis=1
        ).truncate(before=start_date, after=end_date, axis=1)

        if quarter:
            revenue_segmentation_total.columns = pd.PeriodIndex(
                revenue_segmentation_total.columns, freq="Q"
            )
        else:
            revenue_segmentation_total.columns = pd.PeriodIndex(
                revenue_segmentation_total.columns, freq="Y"
            )

        # Rows summing to zero have no data in this window, so drop them.
        revenue_segmentation_total = revenue_segmentation_total[
            revenue_segmentation_total.sum(axis=1) != 0
        ]

        return (
            revenue_segmentation_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_analyst_estimates(
    tickers: str | list[str],
    api_key: str = "",
    quarter: bool = False,
    start_date: str | None = None,
    rounding: int | None = 4,
    sleep_timer: bool = False,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves analyst estimates for one or multiple companies, and returns a DataFrame containing the data.

    This data contains the following estimates:
        - Estimated Revenue Low
        - Estimated Revenue High
        - Estimated Revenue Average
        - Estimated EBITDA Low
        - Estimated EBITDA High
        - Estimated EBITDA Average
        - Estimated EBIT Low
        - Estimated EBIT High
        - Estimated EBIT Average
        - Estimated Net Income Low
        - Estimated Net Income High
        - Estimated Net Income Average
        - Estimated SGA Expense Low
        - Estimated SGA Expense High
        - Estimated SGA Expense Average
        - Estimated EPS Low
        - Estimated EPS High
        - Estimated EPS Average
        - Number of Analysts

    Args:
        tickers (list[str] | str): A single ticker or a list of company tickers.
        api_key (str): API key for the financial data provider.
        quarter (bool): Whether to retrieve quarterly data. Defaults to False (annual data).
        start_date (str): The start date to filter data with.
        rounding (int | None): The number of decimals to round the results to, or None for no
            rounding. Defaults to 4.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: A DataFrame containing the analyst estimates for all provided
            tickers and the list of tickers without data.
    """

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/analyst-estimates"
            f"?symbol={ticker}&period={period}&apikey={api_key}"
        )
        analyst_estimates = get_financial_data(
            url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
        )

        try:
            analyst_estimates = analyst_estimates.drop("symbol", axis=1)

            # One day is deducted: a period reported as 2023-07-01 is really 2023Q2.
            analyst_estimates["date"] = pd.to_datetime(
                analyst_estimates["date"]
            ) - pd.offsets.Day(1)

            if quarter:
                analyst_estimates["date"] = pd.to_datetime(
                    analyst_estimates["date"]
                ).dt.to_period("Q")
            else:
                analyst_estimates["date"] = pd.to_datetime(
                    analyst_estimates["date"].astype(str)
                ).dt.to_period("Y")

            analyst_estimates = analyst_estimates.set_index("date").T

            if analyst_estimates.columns.duplicated().any():
                # Duplicate statements for one period are equal, so one copy can be dropped.
                analyst_estimates = analyst_estimates.loc[
                    :, ~analyst_estimates.columns.duplicated()
                ]

            analyst_estimates.loc["Number of Analysts", :] = (
                analyst_estimates.loc["numAnalystsRevenue", :]
                + analyst_estimates.loc["numAnalystsEps", :]
            ) // 2
            analyst_estimates = analyst_estimates.drop(
                ["numAnalystsRevenue", "numAnalystsEps"], axis=0
            )

            return ticker, analyst_estimates.rename(index=naming), True
        except KeyError:
            return ticker, analyst_estimates, False

    naming: dict = {
        "revenueLow": "Estimated Revenue Low",
        "revenueHigh": "Estimated Revenue High",
        "revenueAvg": "Estimated Revenue Average",
        "ebitdaLow": "Estimated EBITDA Low",
        "ebitdaHigh": "Estimated EBITDA High",
        "ebitdaAvg": "Estimated EBITDA Average",
        "ebitLow": "Estimated EBIT Low",
        "ebitHigh": "Estimated EBIT High",
        "ebitAvg": "Estimated EBIT Average",
        "netIncomeLow": "Estimated Net Income Low",
        "netIncomeHigh": "Estimated Net Income High",
        "netIncomeAvg": "Estimated Net Income Average",
        "sgaExpenseLow": "Estimated SGA Expense Low",
        "sgaExpenseHigh": "Estimated SGA Expense High",
        "sgaExpenseAvg": "Estimated SGA Expense Average",
        "epsLow": "Estimated EPS Low",
        "epsHigh": "Estimated EPS High",
        "epsAvg": "Estimated EPS Average",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    period = "quarter" if quarter else "annual"

    logger.info("Obtaining analyst estimates for %d ticker(s)", len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    analyst_estimates_dict: dict = {ticker: data for ticker, data, _ in results}
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    analyst_estimates_dict = error_model.check_for_error_messages(
        dataset_dictionary=analyst_estimates_dict, user_subscription=user_subscription
    )

    if analyst_estimates_dict and len(no_data) != len(ticker_list):
        analyst_estimates_total = pd.concat(analyst_estimates_dict, axis=0)

        try:
            # "Number of Analysts" stays float64 (not int) so an unreported count stays NaN, not 0.
            analyst_estimates_total = analyst_estimates_total.astype(np.float64)
        except ValueError as error:
            logger.error(
                "Not able to convert DataFrame to float64 due to %s. This could result in"
                "issues when values are zero and is predominantly relevant for "
                "ratio calculations.",
                error,
            )

        analyst_estimates_total.columns = pd.PeriodIndex(
            analyst_estimates_total.columns, freq="Q" if quarter else "Y"
        )

        analyst_estimates_total = analyst_estimates_total.sort_index(axis=1).truncate(
            before=start_date, axis=1
        )

        analyst_estimates_total = analyst_estimates_total.round(rounding)

        if quarter:
            analyst_estimates_total.columns = pd.PeriodIndex(
                analyst_estimates_total.columns, freq="Q"
            )
        else:
            analyst_estimates_total.columns = pd.PeriodIndex(
                analyst_estimates_total.columns, freq="Y"
            )

        return (
            analyst_estimates_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_profile(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Gives information about the profile of a company which includes i.a. beta, company description, industry and sector.

    Args:
        tickers (list[str] | str): the company ticker or tickers (for example: "AAPL")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: the profile data and the tickers without data.
    """

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
        profile_data = get_financial_data(url=url, user_subscription=user_subscription)

        if profile_data.empty:
            return ticker, profile_data, False

        return ticker, profile_data.T, True

    naming: dict = {
        "symbol": "Symbol",
        "price": "Price",
        "beta": "Beta",
        "marketCap": "Market Capitalization",
        "volume": "Volume",
        "averageVolume": "Average Volume",
        "lastDividend": "Last Dividend",
        "range": "Range",
        "change": "Change",
        "changePercentage": "Change %",
        "companyName": "Company Name",
        "currency": "Currency",
        "cik": "CIK",
        "isin": "ISIN",
        "cusip": "CUSIP",
        "exchange": "Exchange",
        "exchangeFullName": "Exchange Full Name",
        "industry": "Industry",
        "website": "Website",
        "description": "Description",
        "ceo": "CEO",
        "sector": "Sector",
        "country": "Country",
        "fullTimeEmployees": "Full Time Employees",
        "phone": "Phone",
        "address": "Address",
        "city": "City",
        "state": "State",
        "zip": "ZIP Code",
        "dcfDiff": "DCF Difference",
        "dcf": "DCF",
        "ipoDate": "IPO Date",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    logger.info("Obtaining company profiles for %d ticker(s)", len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    profile_dict: dict[str, pd.DataFrame] = {
        ticker: data for ticker, data, _ in results
    }
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    profile_dict = error_model.check_for_error_messages(
        dataset_dictionary=profile_dict, user_subscription=user_subscription
    )

    if profile_dict:
        try:
            profile_dataframe = to_dataframe(
                pd.concat(profile_dict)[0].unstack(level=0)
            )
            profile_dataframe = profile_dataframe.rename(index=naming)
            profile_dataframe = profile_dataframe.drop(
                [
                    "image",
                    "defaultImage",
                    "isEtf",
                    "isActivelyTrading",
                    "isAdr",
                    "isFund",
                ],
                axis=0,
            )
        except (ValueError, KeyError):
            return pd.DataFrame(), no_data

        return profile_dataframe, no_data

    return pd.DataFrame(), no_data


def get_quote(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Gives information about the quote of a company which includes i.a. high/low close prices,
    price-to-earning ratio and shares outstanding.

    Args:
        tickers (list[str] | str): the company ticker or tickers (for example: "AMD")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: the quote data and the tickers without data.
    """

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/quote?symbol={ticker}&apikey={api_key}"
        quote_data = get_financial_data(url=url, user_subscription=user_subscription)

        if quote_data.empty:
            return ticker, quote_data, False

        return ticker, quote_data.T, True

    naming: dict = {
        "symbol": "Symbol",
        "name": "Name",
        "price": "Price",
        "change": "Change",
        "changePercentage": "Change %",
        "dayLow": "Day Low",
        "dayHigh": "Day High",
        "yearHigh": "Year High",
        "yearLow": "Year Low",
        "marketCap": "Market Capitalization",
        "priceAvg50": "Price Average 50 Days",
        "priceAvg200": "Price Average 200 Days",
        "exchange": "Exchange",
        "volume": "Volume",
        "avgVolume": "Average Volume",
        "open": "Open",
        "previousClose": "Previous Close",
        "eps": "EPS",
        "pe": "PE",
        "earningsAnnouncement": "Earnings Announcement",
        "sharesOutstanding": "Shares Outstanding",
        "timestamp": "Timestamp",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    logger.info("Obtaining company quotes for %d ticker(s)", len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    quote_dict: dict[str, pd.DataFrame] = {ticker: data for ticker, data, _ in results}
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    quote_dict = error_model.check_for_error_messages(
        dataset_dictionary=quote_dict, user_subscription=user_subscription
    )

    if quote_dict:
        quote_dataframe = to_dataframe(pd.concat(quote_dict)[0].unstack(level=0))
        quote_dataframe = quote_dataframe.rename(index=naming)

    return quote_dataframe, no_data


def get_rating(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Gives information about the rating of a company which includes i.a. the company rating and
    recommendation as well as ratings based on a variety of ratios.

    Args:
        tickers (list[str] | str): the company ticker or tickers (for example: "MSFT")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: the rating data and the tickers without data.
    """

    def worker(ticker):
        url = (
            f"https://financialmodelingprep.com/stable/ratings-historical?symbol={ticker}&"
            f"apikey={api_key}&limit={'99999' if user_subscription != 'Free' else '1'}"
        )
        ratings = get_financial_data(url=url, user_subscription=user_subscription)

        try:
            ratings = ratings.drop("symbol", axis=1).sort_values(
                by="date", ascending=True
            )

            ratings = ratings.set_index("date")

            ratings = ratings.rename(
                columns={
                    "rating": "Rating",
                    "overallScore": "Rating Score",
                    "discountedCashFlowScore": "DCF Score",
                    "returnOnEquityScore": "ROE Score",
                    "returnOnAssetsScore": "ROA Score",
                    "debtToEquityScore": "DE Score",
                    "priceToEarningsScore": "PE Score",
                    "priceToBookScore": "PB Score",
                }
            )

            return ticker, ratings, True
        except (KeyError, ValueError):
            return ticker, ratings, False

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    logger.info("Obtaining company ratings for %d ticker(s)", len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    ratings_dict: dict[str, pd.DataFrame] = {
        ticker: data for ticker, data, _ in results
    }
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    ratings_dict = error_model.check_for_error_messages(
        dataset_dictionary=ratings_dict, user_subscription=user_subscription
    )

    if ratings_dict:
        ratings_dataframe = pd.concat(ratings_dict, axis=0).dropna()

        if len(ticker_list) == 1:
            ratings_dataframe = to_dataframe(ratings_dataframe.loc[ticker_list[0]])

        return ratings_dataframe, no_data

    return pd.DataFrame(), no_data


def get_earnings_calendar(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    actual_dates: bool = True,
    sleep_timer: bool = False,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Obtains Earnings Calendar which shows the expected earnings and EPS for a company.

    Args:
        tickers (list[str] | str): the company ticker or tickers (for example: "MSFT")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        actual_dates (bool): Whether to retrieve actual dates. Defaults to False (converted to quarterly). This is the
        default because the actual date refers to the corresponding quarter.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: the earnings calendar data and the tickers without data.
    """

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/earnings"
            f"?symbol={ticker}&apikey={api_key}&limit={'99999' if user_subscription != 'Free' else '5'}"
        )
        earnings_calendar = get_financial_data(
            url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
        )

        try:
            earnings_calendar["date"] = (
                pd.to_datetime(earnings_calendar["date"])
                if actual_dates
                else pd.to_datetime(earnings_calendar["date"]).dt.to_period("Q")
            )

            earnings_calendar = earnings_calendar.set_index("date").sort_index()

            if earnings_calendar.columns.duplicated().any():
                # Duplicate statements for one period are equal, so one copy can be dropped.
                earnings_calendar = earnings_calendar.loc[
                    :, ~earnings_calendar.columns.duplicated()
                ]

            earnings_calendar = earnings_calendar.rename(columns=naming)

            earnings_calendar = earnings_calendar.sort_index(axis=0).truncate(
                before=start_date, after=end_date, axis=0
            )

            return ticker, earnings_calendar[naming.values()], True
        except KeyError:
            return ticker, earnings_calendar, False

    naming: dict = {
        "epsActual": "EPS",
        "epsEstimated": "Estimated EPS",
        "revenueActual": "Revenue",
        "revenueEstimated": "Estimated Revenue",
        "lastUpdated": "Last Updated",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    logger.info("Obtaining earnings calendars for %d ticker(s)", len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    earnings_calendar_dict: dict = {ticker: data for ticker, data, _ in results}
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    earnings_calendar_dict = error_model.check_for_error_messages(
        dataset_dictionary=earnings_calendar_dict, user_subscription=user_subscription
    )

    if earnings_calendar_dict:
        earnings_calendar_total = pd.concat(earnings_calendar_dict, axis=0)

        return (
            earnings_calendar_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_dividend_calendar(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = False,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Obtains Dividend Calendar which shows the dividends and related dates.

    Args:
        tickers (list[str] | str): the company ticker or tickers (for example: "MSFT")
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: the earnings calendar data and the tickers without data.
    """

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/dividends"
            f"?symbol={ticker}&apikey={api_key}&limit={'99999' if user_subscription != 'Free' else '5'}"
        )
        dividend_calendar = get_financial_data(
            url=url,
            sleep_timer=sleep_timer,
            raw=True,
            user_subscription=user_subscription,
        )

        try:
            dividend_calendar = pd.DataFrame(dividend_calendar)

            if "date" not in dividend_calendar.columns:
                # Nothing to store: without a date column there is no calendar and no
                # error frame worth reporting either.
                return ticker, None, False

            dividend_calendar = dividend_calendar.set_index("date")

            dividend_calendar.index = pd.to_datetime(dividend_calendar.index)
            dividend_calendar.index = pd.DatetimeIndex(
                dividend_calendar.index
            ).to_period(freq="D")

            dividend_calendar = dividend_calendar.sort_index()

            dividend_calendar = dividend_calendar.rename(columns=naming)

            dividend_calendar = dividend_calendar.sort_index(axis=0).truncate(
                before=start_date, after=end_date, axis=0
            )

            return ticker, dividend_calendar[naming.values()], True
        except KeyError:
            return ticker, dividend_calendar, False

    naming: dict = {
        "adjDividend": "Adj Dividend",
        "dividend": "Dividend",
        "yield": "Yield",
        "recordDate": "Record Date",
        "paymentDate": "Payment Date",
        "declarationDate": "Declaration Date",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    logger.info("Obtaining dividend calendars for %d ticker(s)", len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    dividend_calendar_dict: dict = {
        ticker: data for ticker, data, _ in results if data is not None
    }
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    dividend_calendar_dict = error_model.check_for_error_messages(
        dataset_dictionary=dividend_calendar_dict, user_subscription=user_subscription
    )

    if dividend_calendar_dict:
        dividend_calendar_total = pd.concat(dividend_calendar_dict, axis=0)

        return (
            dividend_calendar_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_esg_scores(
    tickers: list[str] | str,
    api_key: str,
    quarter: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = False,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Obtains the ESG Scores for a selection of companies.

    Args:
        tickers (list[str] | str): the company ticker or tickers (for example: "MSFT")
        quarter (bool): whether to retrieve quarterly data. Defaults to False.
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: the earnings calendar data and the tickers without data.
    """

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/esg-disclosures?"
            f"symbol={ticker}&apikey={api_key}"
        )
        esg_scores = get_financial_data(
            url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
        )

        try:
            if "date" not in esg_scores.columns:
                return ticker, esg_scores, False

            # One day is deducted: a period reported as 2023-07-01 is really 2023Q2.
            esg_scores["date"] = pd.to_datetime(esg_scores["date"]) - pd.offsets.Day(1)

            esg_scores = esg_scores.set_index("date")
            esg_scores.index = pd.DatetimeIndex(esg_scores.index).to_period(
                freq="Q" if quarter else "Y"
            )

            esg_scores = esg_scores.sort_index()
            esg_scores = esg_scores.rename(columns=naming)

            esg_scores = esg_scores.sort_index(axis=0).truncate(
                before=start_date, after=end_date, axis=0
            )

            esg_scores = esg_scores[~esg_scores.index.duplicated()]

            return ticker, esg_scores[naming.values()], True
        except KeyError:
            return ticker, esg_scores, False

    naming: dict = {
        "environmentalScore": "Environmental Score",
        "socialScore": "Social Score",
        "governanceScore": "Governance Score",
        "ESGScore": "ESG Score",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    logger.info("Obtaining ESG scores for %d ticker(s)", len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    esg_scores_dict: dict = {ticker: data for ticker, data, _ in results}
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    esg_scores_dict = error_model.check_for_error_messages(
        dataset_dictionary=esg_scores_dict, user_subscription=user_subscription
    )

    if esg_scores_dict:
        esg_scores_total = pd.concat(esg_scores_dict, axis=0).unstack(level=0)

        return (
            esg_scores_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def get_market_risk_premium(
    api_key: str,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Obtains the equity market risk premium by country -- the country default spread plus the
    equity risk premium, following the approach popularized by Aswath Damodaran -- which is
    widely used to calibrate country-specific costs of equity and discount rates in a
    multi-country setting.

    Also known as: country risk premium, Damodaran equity risk premium.

    Args:
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        pd.DataFrame: the market risk premium by country, indexed by country and including
        the continent, Country Risk Premium and Total Equity Risk Premium (both in
        percentage points).
    """
    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    url = (
        f"https://financialmodelingprep.com/stable/market-risk-premium?apikey={api_key}"
    )

    market_risk_premium = get_financial_data(url=url)

    if "country" not in market_risk_premium.columns:
        return market_risk_premium

    market_risk_premium = market_risk_premium.rename(
        columns={
            "country": "Country",
            "continent": "Continent",
            "countryRiskPremium": "Country Risk Premium",
            "totalEquityRiskPremium": "Total Equity Risk Premium",
        }
    )

    market_risk_premium = market_risk_premium.set_index("Country").sort_index()

    return market_risk_premium


def get_commitment_of_traders(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Obtains the CFTC Commitment of Traders (COT) report for a selection of tickers. Published
    weekly by the U.S. Commodity Futures Trading Commission, it breaks down open interest in
    futures markets by trader type -- Non-Commercial (large speculators), Commercial (hedgers)
    and Non-Reportable (small traders) -- and is widely used to gauge positioning and sentiment
    in commodity, currency, interest rate and stock index futures markets.

    Note that this data is only available for CFTC-tracked futures markets. Tickers without a
    corresponding futures contract (e.g. most individual equities) return no data for that ticker.

    Also known as: COT report, CFTC positioning data, speculator/hedger positioning.

    Args:
        tickers (list[str] | str): the ticker or tickers (for example: "NG" for Natural Gas futures)
        api_key (string): the API Key obtained from
        https://www.jeroenbouma.com/fmp
        start_date (str): The start date to filter data with.
        end_date (str): The end date to filter data with.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: the Commitment of Traders report data and the tickers without data.
    """

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/commitment-of-traders-report?"
            f"symbol={ticker}&apikey={api_key}"
        )
        cot_report = get_financial_data(
            url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
        )

        try:
            if "date" not in cot_report.columns:
                return ticker, cot_report, False

            cot_report["date"] = pd.to_datetime(cot_report["date"])
            cot_report = cot_report.set_index("date").sort_index()

            cot_report = cot_report.rename(columns=naming)

            cot_report = cot_report.truncate(before=start_date, after=end_date, axis=0)

            cot_report = cot_report[~cot_report.index.duplicated()]

            columns = [
                column for column in naming.values() if column in cot_report.columns
            ]

            return ticker, cot_report[columns], True
        except KeyError:
            return ticker, cot_report, False

    naming: dict = {
        "name": "Name",
        "sector": "Sector",
        "openInterestAll": "Open Interest",
        "noncommPositionsLongAll": "Non-Commercial Long",
        "noncommPositionsShortAll": "Non-Commercial Short",
        "noncommPositionsSpreadAll": "Non-Commercial Spread",
        "commPositionsLongAll": "Commercial Long",
        "commPositionsShortAll": "Commercial Short",
        "totReptPositionsLongAll": "Total Reportable Long",
        "totReptPositionsShortAll": "Total Reportable Short",
        "nonreptPositionsLongAll": "Non-Reportable Long",
        "nonreptPositionsShortAll": "Non-Reportable Short",
        "changeInOpenInterestAll": "Change in Open Interest",
        "changeInNoncommLongAll": "Change in Non-Commercial Long",
        "changeInNoncommShortAll": "Change in Non-Commercial Short",
        "changeInCommLongAll": "Change in Commercial Long",
        "changeInCommShortAll": "Change in Commercial Short",
        "pctOfOiNoncommLongAll": "% OI Non-Commercial Long",
        "pctOfOiNoncommShortAll": "% OI Non-Commercial Short",
        "pctOfOiCommLongAll": "% OI Commercial Long",
        "pctOfOiCommShortAll": "% OI Commercial Short",
    }

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    sleep_timer = user_subscription != "Free"

    logger.info(
        "Obtaining Commitment of Traders data for %d ticker(s)", len(ticker_list)
    )
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    cot_dict: dict = {ticker: data for ticker, data, _ in results}
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them
    cot_dict = error_model.check_for_error_messages(
        dataset_dictionary=cot_dict, user_subscription=user_subscription
    )

    if cot_dict:
        cot_total = pd.concat(cot_dict, axis=0).unstack(level=0)

        return (
            cot_total,
            no_data,
        )

    return pd.DataFrame(), no_data


def _to_ticker_list(tickers: list[str] | str) -> list[str]:
    """
    Normalises the tickers argument of the per-ticker functions below into a list.

    Args:
        tickers (list[str] | str): A single ticker or a list of tickers.

    Returns:
        list[str]: The tickers as a list.

    Raises:
        ValueError: If tickers is neither a string nor a list.
    """
    if isinstance(tickers, str):
        return [tickers]
    if isinstance(tickers, list):
        return tickers

    raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")


def _require_fmp_api_key(api_key: str) -> None:
    """
    Raises the standard error when no FinancialModelingPrep API key is available.

    Args:
        api_key (str): The FinancialModelingPrep API key.

    Raises:
        ValueError: If the API key is empty.
    """
    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )


def _record_limit(user_subscription: str) -> str:
    """
    The record limit to request, following the same rule as the other endpoints: the
    Free plan is capped at five records, any paid plan gets the full history.

    Args:
        user_subscription (str): The subscription type of the user.

    Returns:
        str: The limit to pass to the endpoint.
    """
    return "99999" if user_subscription != "Free" else "5"


def _collect_per_ticker_frames(
    worker,
    ticker_list: list[str],
    description: str,
    user_subscription: str,
    unstack: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Runs a per-ticker worker in parallel and combines the results, exactly as the
    other per-ticker functions in this module do: tickers without data are reported
    separately and FinancialModelingPrep error messages are surfaced.

    Args:
        worker (Callable): A function taking a ticker and returning (ticker, data, has_data).
        ticker_list (list[str]): The tickers to collect.
        description (str): What is collected, used in the log message.
        user_subscription (str): The subscription type of the user.
        unstack (bool): When True, each ticker's data is a Series or single column that
            becomes one column per ticker. When False, the tickers become the first
            index level. Defaults to False.

    Returns:
        tuple[pd.DataFrame, list[str]]: The combined data and the tickers without data.
    """
    logger.info("Obtaining %s for %d ticker(s)", description, len(ticker_list))
    results = helpers.run_in_parallel(worker, [(ticker,) for ticker in ticker_list])

    data_dict: dict = {ticker: data for ticker, data, _ in results}
    no_data: list[str] = [ticker for ticker, _, has_data in results if not has_data]

    # Checks if any errors are in the dataset and if this is the case, reports them. A
    # failed response stays in so its error is reported, and the check reads columns,
    # so a single-ticker Series is looked at as a one-column frame.
    checked = error_model.check_for_error_messages(
        dataset_dictionary={
            ticker: data if isinstance(data, pd.DataFrame) else data.to_frame()
            for ticker, data in data_dict.items()
        },
        user_subscription=user_subscription,
    )
    data_dict = {
        ticker: data
        for ticker, data in data_dict.items()
        if ticker in checked and ticker not in no_data
    }

    if not data_dict:
        return pd.DataFrame(), no_data

    if unstack:
        return pd.concat(data_dict, axis=1), no_data

    return pd.concat(data_dict, axis=0), no_data


def get_executives(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the key executives of each ticker: their title, pay, gender and year of birth.

    Args:
        tickers (list[str] | str): The tickers to retrieve the executives for.
        api_key (str): The FinancialModelingPrep API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The executives indexed by ticker and name, and the
            tickers for which no data could be found.
    """
    naming: dict = {
        "title": "Title",
        "pay": "Pay",
        "currencyPay": "Currency",
        "gender": "Gender",
        "yearBorn": "Year Born",
        "titleSince": "Title Since",
        "active": "Active",
    }

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/key-executives?symbol={ticker}&apikey={api_key}"
        executives = get_financial_data(url=url, user_subscription=user_subscription)

        if "name" not in executives.columns:
            return ticker, executives, False

        executives = executives.set_index("name").rename(columns=naming)
        executives.index.name = "Name"

        return (
            ticker,
            executives[[column for column in naming.values() if column in executives]],
            True,
        )

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "executives", user_subscription
    )


def get_executive_compensation(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the compensation of each ticker's executives per year as reported in the
    proxy statements: salary, bonus, stock and option awards and the total.

    Args:
        tickers (list[str] | str): The tickers to retrieve the compensation for.
        api_key (str): The FinancialModelingPrep API key.
        start_date (str | None): Only keep years from the year of this date onwards.
        end_date (str | None): Only keep years up to the year of this date.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The compensation indexed by ticker, year and
            executive, and the tickers for which no data could be found.
    """
    naming: dict = {
        "filingDate": "Filing Date",
        "acceptedDate": "Accepted Date",
        "salary": "Salary",
        "bonus": "Bonus",
        "stockAward": "Stock Award",
        "optionAward": "Option Award",
        "incentivePlanCompensation": "Incentive Plan Compensation",
        "allOtherCompensation": "All Other Compensation",
        "total": "Total",
        "link": "Link",
    }

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/governance-executive-compensation"
            f"?symbol={ticker}&apikey={api_key}"
        )
        compensation = get_financial_data(url=url, user_subscription=user_subscription)

        if "nameAndPosition" not in compensation.columns:
            return ticker, compensation, False

        # Reported per fiscal year rather than per date, so the years are compared directly.
        if start_date:
            compensation = compensation[compensation["year"] >= int(start_date[:4])]
        if end_date:
            compensation = compensation[compensation["year"] <= int(end_date[:4])]

        compensation = compensation.rename(
            columns={"year": "Year", "nameAndPosition": "Name and Position"}
        )
        compensation = compensation.set_index(
            ["Year", "Name and Position"]
        ).sort_index()
        compensation = compensation.rename(columns=naming)

        return ticker, compensation[list(naming.values())], not compensation.empty

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "executive compensation", user_subscription
    )


def get_company_notes(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the notes (debt securities) each ticker has listed, e.g. "1.625% Notes due 2026".

    Args:
        tickers (list[str] | str): The tickers to retrieve the notes for.
        api_key (str): The FinancialModelingPrep API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The notes indexed by ticker and title, and the
            tickers for which no data could be found.
    """

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/company-notes?symbol={ticker}&apikey={api_key}"
        notes = get_financial_data(url=url, user_subscription=user_subscription)

        if "title" not in notes.columns:
            return ticker, notes, False

        notes = notes.rename(
            columns={"title": "Title", "exchange": "Exchange", "cik": "CIK"}
        )

        return ticker, notes.set_index("Title")[["Exchange", "CIK"]], True

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "company notes", user_subscription
    )


def get_employee_count(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = False,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the number of employees each ticker reported in its annual filings over
    time. The latest count is simply the most recent row, so this covers both the
    employee count and the historical employee count endpoints.

    Args:
        tickers (list[str] | str): The tickers to retrieve the employee count for.
        api_key (str): The FinancialModelingPrep API key.
        start_date (str | None): The start date to filter the reporting periods with.
        end_date (str | None): The end date to filter the reporting periods with.
        sleep_timer (bool): Whether to wait and retry when the rate limit is reached.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The employee count per reporting year (rows) and
            ticker (columns), and the tickers for which no data could be found.
    """

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/historical-employee-count"
            f"?symbol={ticker}&limit={_record_limit(user_subscription)}&apikey={api_key}"
        )
        employees = get_financial_data(
            url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
        )

        if "employeeCount" not in employees.columns:
            return ticker, employees, False

        # An amended filing reports the same period again; the latest filing is kept.
        employees = employees.sort_values("filingDate").drop_duplicates(
            "periodOfReport", keep="last"
        )
        employees["periodOfReport"] = pd.to_datetime(employees["periodOfReport"])
        employee_count = employees.set_index("periodOfReport")[
            "employeeCount"
        ].sort_index()
        employee_count = employee_count.truncate(before=start_date, after=end_date)
        employee_count.index = pd.DatetimeIndex(employee_count.index).to_period(
            freq="Y"
        )
        employee_count = employee_count[~employee_count.index.duplicated(keep="last")]
        employee_count.index.name = "Period"

        return ticker, employee_count, not employee_count.empty

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "employee counts", user_subscription, unstack=True
    )


def get_shares_float(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves each ticker's free float: the share of outstanding shares that is available
    for public trading, as a decimal.

    Args:
        tickers (list[str] | str): The tickers to retrieve the share float for.
        api_key (str): The FinancialModelingPrep API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The share float figures (rows) per ticker (columns),
            and the tickers for which no data could be found.
    """
    naming: dict = {
        "date": "Date",
        "freeFloat": "Free Float",
        "floatShares": "Float Shares",
        "outstandingShares": "Outstanding Shares",
        "source": "Source",
    }

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/shares-float?symbol={ticker}&apikey={api_key}"
        shares_float = get_financial_data(url=url, user_subscription=user_subscription)

        if "freeFloat" not in shares_float.columns:
            return ticker, shares_float, False

        # Published as a percentage; every other ratio in the Finance Toolkit is a decimal.
        shares_float["freeFloat"] = shares_float["freeFloat"] / 100

        return (
            ticker,
            shares_float.iloc[0].rename(index=naming)[list(naming.values())],
            True,
        )

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "share floats", user_subscription, unstack=True
    )


# Legal suffixes the mergers and acquisitions search does not match on: it finds
# "Microsoft" but nothing for "Microsoft Corporation".
COMPANY_NAME_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "plc",
    "llc",
    "lp",
    "nv",
    "sa",
    "ag",
    "se",
    "asa",
    "ab",
    "oyj",
    "spa",
    "bv",
    "holding",
    "holdings",
    "group",
}


def _search_name(company_name: str) -> str:
    """
    Shortens a company name to the part the mergers and acquisitions search matches on,
    by dropping trailing legal suffixes, e.g. "Microsoft Corporation" becomes "Microsoft".

    Args:
        company_name (str): The company name as listed in the profile.

    Returns:
        str: The name without its trailing legal suffixes.
    """
    words = company_name.replace(",", " ").split()

    while (
        len(words) > 1 and words[-1].lower().replace(".", "") in COMPANY_NAME_SUFFIXES
    ):
        words.pop()

    return " ".join(words)


def get_mergers_acquisitions(
    tickers: list[str] | str,
    company_names: dict[str, str],
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the mergers and acquisitions each ticker took part in, as acquirer or target.

    The endpoint searches by company name rather than ticker, and a name search also
    returns companies with a similar name ("Apple" matches "Apple Hospitality REIT"), so
    only the deals in which the ticker itself is the acquirer or the target are kept.

    Args:
        tickers (list[str] | str): The tickers to retrieve the deals for.
        company_names (dict[str, str]): The company name per ticker, as listed in the profile; its
            legal suffix is dropped before searching.
        api_key (str): The FinancialModelingPrep API key.
        start_date (str | None): The start date to filter the transactions with.
        end_date (str | None): The end date to filter the transactions with.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The deals indexed by ticker and transaction date,
            and the tickers for which no data could be found.
    """
    naming: dict = {
        "symbol": "Acquirer Symbol",
        "companyName": "Acquirer Name",
        "targetedSymbol": "Target Symbol",
        "targetedCompanyName": "Target Name",
        "acceptedDate": "Accepted Date",
        "link": "Link",
    }

    def worker(ticker):
        name = company_names.get(ticker)

        if not name:
            return ticker, pd.DataFrame(), False

        url = (
            "https://financialmodelingprep.com/stable/mergers-acquisitions-search"
            f"?name={requests.utils.quote(_search_name(name))}&apikey={api_key}"
        )
        deals = get_financial_data(url=url, user_subscription=user_subscription)

        if "transactionDate" not in deals.columns:
            return ticker, deals, False

        deals = deals[
            (deals["symbol"] == ticker) | (deals["targetedSymbol"] == ticker)
        ].copy()
        deals["Role"] = (
            deals["symbol"].eq(ticker).map({True: "Acquirer", False: "Target"})
        )
        deals["transactionDate"] = pd.to_datetime(deals["transactionDate"])
        deals = (
            deals.set_index("transactionDate")
            .sort_index()
            .truncate(before=start_date, after=end_date)
        )
        deals.index.name = "Transaction Date"
        deals = deals.rename(columns=naming)

        return ticker, deals[["Role", *naming.values()]], not deals.empty

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "mergers and acquisitions", user_subscription
    )


def get_stock_splits(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the stock splits of each ticker, e.g. a 4-for-1 split as numerator 4 and
    denominator 1.

    Args:
        tickers (list[str] | str): The tickers to retrieve the splits for.
        api_key (str): The FinancialModelingPrep API key.
        start_date (str | None): The start date to filter the splits with.
        end_date (str | None): The end date to filter the splits with.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The splits indexed by ticker and date, and the
            tickers for which no data could be found.
    """
    naming: dict = {
        "numerator": "Numerator",
        "denominator": "Denominator",
        "splitType": "Split Type",
    }

    def worker(ticker):
        url = (
            "https://financialmodelingprep.com/stable/splits"
            f"?symbol={ticker}&limit={_record_limit(user_subscription)}&apikey={api_key}"
        )
        splits = get_financial_data(url=url, user_subscription=user_subscription)

        if "date" not in splits.columns:
            return ticker, splits, False

        splits["date"] = pd.to_datetime(splits["date"])
        splits = (
            splits.set_index("date")
            .sort_index()
            .truncate(before=start_date, after=end_date)
        )
        splits.index.name = "Date"
        splits = splits.rename(columns=naming)

        return (
            ticker,
            splits[[column for column in naming.values() if column in splits]],
            not splits.empty,
        )

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "stock splits", user_subscription
    )


def get_insider_trade_statistics(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves quarterly statistics on the insider transactions of each ticker: the number
    and size of acquisitions and disposals, and the number of purchases and sales.

    Args:
        tickers (list[str] | str): The tickers to retrieve the statistics for.
        api_key (str): The FinancialModelingPrep API key.
        start_date (str | None): The start date to filter the quarters with.
        end_date (str | None): The end date to filter the quarters with.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The statistics indexed by ticker and quarter, and the
            tickers for which no data could be found.
    """
    naming: dict = {
        "acquiredTransactions": "Acquired Transactions",
        "disposedTransactions": "Disposed Transactions",
        "acquiredDisposedRatio": "Acquired/Disposed Ratio",
        "totalAcquired": "Total Acquired",
        "totalDisposed": "Total Disposed",
        "averageAcquired": "Average Acquired",
        "averageDisposed": "Average Disposed",
        "totalPurchases": "Total Purchases",
        "totalSales": "Total Sales",
    }

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/insider-trading/statistics?symbol={ticker}&apikey={api_key}"
        statistics = get_financial_data(url=url, user_subscription=user_subscription)

        if "quarter" not in statistics.columns:
            return ticker, statistics, False

        statistics.index = pd.PeriodIndex(
            [
                pd.Period(year=year, quarter=quarter, freq="Q")
                for year, quarter in zip(statistics["year"], statistics["quarter"])
            ],
            name="Period",
        )
        statistics = statistics.sort_index().rename(columns=naming)
        statistics = statistics.loc[
            (
                pd.Period(start_date, freq="Q")
                if start_date
                else statistics.index.min()
            ) : (pd.Period(end_date, freq="Q") if end_date else statistics.index.max())
        ]

        return ticker, statistics[list(naming.values())], not statistics.empty

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "insider trade statistics", user_subscription
    )


def get_stock_grades(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the analyst grades each ticker received: per date and grading company the
    previous and the new grade, and whether it was upgraded, downgraded or maintained.

    Args:
        tickers (list[str] | str): The tickers to retrieve the grades for.
        api_key (str): The FinancialModelingPrep API key.
        start_date (str | None): The start date to filter the grades with.
        end_date (str | None): The end date to filter the grades with.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The grades indexed by ticker, date and grading
            company, and the tickers for which no data could be found.
    """
    naming: dict = {
        "previousGrade": "Previous Grade",
        "newGrade": "New Grade",
        "action": "Action",
    }

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/grades?symbol={ticker}&apikey={api_key}"
        grades = get_financial_data(url=url, user_subscription=user_subscription)

        if "gradingCompany" not in grades.columns:
            return ticker, grades, False

        grades["date"] = pd.to_datetime(grades["date"])
        grades = grades.rename(
            columns={"date": "Date", "gradingCompany": "Grading Company", **naming}
        )
        grades = grades.set_index(["Date", "Grading Company"]).sort_index()
        grades = grades.loc[
            pd.Timestamp(
                start_date or grades.index.get_level_values(0).min()
            ) : pd.Timestamp(end_date or grades.index.get_level_values(0).max())
        ]
        grades["Action"] = grades["Action"].str.capitalize()

        return ticker, grades[list(naming.values())], not grades.empty

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "stock grades", user_subscription
    )


def get_etf_holdings(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the holdings of each ticker that is an ETF or fund, with their weight as a
    decimal. Tickers that are not an ETF or fund return no data.

    Args:
        tickers (list[str] | str): The tickers to retrieve the holdings for.
        api_key (str): The FinancialModelingPrep API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The holdings indexed by ticker and asset, and the
            tickers for which no data could be found.
    """
    naming: dict = {
        "name": "Name",
        "isin": "ISIN",
        "securityCusip": "CUSIP",
        "sharesNumber": "Shares",
        "weightPercentage": "Weight",
        "marketValue": "Market Value",
        "updatedAt": "Updated At",
    }

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/etf/holdings?symbol={ticker}&apikey={api_key}"
        holdings = get_financial_data(url=url, user_subscription=user_subscription)

        if "asset" not in holdings.columns:
            return ticker, holdings, False

        # Published as a percentage; every other ratio in the Finance Toolkit is a decimal.
        holdings["weightPercentage"] = holdings["weightPercentage"] / 100
        holdings = holdings.set_index("asset").rename(columns=naming)
        holdings.index.name = "Asset"

        return ticker, holdings[list(naming.values())], True

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "ETF holdings", user_subscription
    )


def get_etf_information(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the profile of each ticker that is an ETF or fund: issuer, asset class,
    expense ratio, assets under management and more. Tickers that are not an ETF or fund
    return no data.

    Args:
        tickers (list[str] | str): The tickers to retrieve the information for.
        api_key (str): The FinancialModelingPrep API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The information (rows) per ticker (columns), and the
            tickers for which no data could be found.
    """
    naming: dict = {
        "name": "Name",
        "description": "Description",
        "isin": "ISIN",
        "securityCusip": "CUSIP",
        "assetClass": "Asset Class",
        "domicile": "Domicile",
        "etfCompany": "ETF Company",
        "website": "Website",
        "inceptionDate": "Inception Date",
        "expenseRatio": "Expense Ratio",
        "assetsUnderManagement": "Assets Under Management",
        "avgVolume": "Average Volume",
        "nav": "NAV",
        "navCurrency": "NAV Currency",
        "holdingsCount": "Holdings Count",
        "isActivelyTrading": "Actively Trading",
        "updatedAt": "Updated At",
    }

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/etf/info?symbol={ticker}&apikey={api_key}"
        information = get_financial_data(url=url, user_subscription=user_subscription)

        if "assetClass" not in information.columns:
            return ticker, information, False

        # Published as a percentage; every other ratio in the Finance Toolkit is a decimal.
        information["expenseRatio"] = information["expenseRatio"] / 100
        information = information.iloc[0].rename(index=naming)

        return (
            ticker,
            information[
                [field for field in naming.values() if field in information.index]
            ],
            True,
        )

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "ETF information", user_subscription, unstack=True
    )


def get_etf_country_weightings(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves how each ETF or fund is allocated across countries, as decimals. Tickers that
    are not an ETF or fund return no data.

    Args:
        tickers (list[str] | str): The tickers to retrieve the country weightings for.
        api_key (str): The FinancialModelingPrep API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The weight per country (rows) and ticker (columns),
            and the tickers for which no data could be found.
    """

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/etf/country-weightings?symbol={ticker}&apikey={api_key}"
        weightings = get_financial_data(url=url, user_subscription=user_subscription)

        if "country" not in weightings.columns:
            return ticker, weightings, False

        # Published as text such as "97.31%" rather than as a number.
        weights = (
            pd.to_numeric(
                weightings["weightPercentage"].astype(str).str.rstrip("%"),
                errors="coerce",
            )
            / 100
        )
        weights.index = pd.Index(weightings["country"], name="Country")

        return ticker, weights, True

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "ETF country weightings", user_subscription, unstack=True
    )


def get_etf_sector_weightings(
    tickers: list[str] | str,
    api_key: str,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves how each ETF or fund is allocated across sectors, as decimals. Tickers that
    are not an ETF or fund return no data.

    Args:
        tickers (list[str] | str): The tickers to retrieve the sector weightings for.
        api_key (str): The FinancialModelingPrep API key.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The weight per sector (rows) and ticker (columns), and
            the tickers for which no data could be found.
    """

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/etf/sector-weightings?symbol={ticker}&apikey={api_key}"
        weightings = get_financial_data(url=url, user_subscription=user_subscription)

        if "sector" not in weightings.columns:
            return ticker, weightings, False

        # Published as a percentage; every other ratio in the Finance Toolkit is a decimal.
        weights = pd.to_numeric(weightings["weightPercentage"], errors="coerce") / 100
        weights.index = pd.Index(weightings["sector"], name="Sector")

        return ticker, weights, True

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "ETF sector weightings", user_subscription, unstack=True
    )


def get_earnings_call_transcripts(
    tickers: list[str] | str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    latest: bool = True,
    user_subscription: str = "Free",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieves the earnings call transcripts of each ticker.

    The endpoint returns one transcript per call, so collecting a history means one call
    per quarter. To keep that affordable, the quarters that have a transcript are listed
    first (one call per ticker) and only the transcripts that are needed are retrieved:
    the most recent one by default, or every one between the start and end date. Each
    transcript is cached on its own, because a published transcript does not change, so
    a quarter is only ever retrieved once no matter how often a history is requested.

    Args:
        tickers (list[str] | str): The tickers to retrieve the transcripts for.
        api_key (str): The FinancialModelingPrep API key.
        start_date (str | None): The start date of the calls to retrieve when latest is False.
        end_date (str | None): The end date of the calls to retrieve when latest is False.
        latest (bool): Whether to only retrieve the most recent transcript. Defaults to True.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Returns:
        tuple[pd.DataFrame, list[str]]: The transcripts indexed by ticker and fiscal
            period, and the tickers for which no data could be found.
    """
    cache = get_active_cache()

    def transcript(ticker: str, fiscal_year: int, quarter: int) -> pd.DataFrame:
        entity = f"{ticker}|{fiscal_year}Q{quarter}"

        if cache is not None:
            stored = cache.get(
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="earnings_call_transcripts",
                entity=entity,
            )
            if stored is not None:
                return stored

        url = (
            "https://financialmodelingprep.com/stable/earning-call-transcript"
            f"?symbol={ticker}&year={fiscal_year}&quarter={quarter}&apikey={api_key}"
        )
        result = get_financial_data(url=url, user_subscription=user_subscription)

        if "content" in result.columns and cache is not None:
            cache.set(
                source=policy_model.FINANCIAL_MODELING_PREP,
                dataset="earnings_call_transcripts",
                entity=entity,
                data=result,
            )

        return result

    def worker(ticker):
        url = f"https://financialmodelingprep.com/stable/earning-call-transcript-dates?symbol={ticker}&apikey={api_key}"
        calls = get_financial_data(url=url, user_subscription=user_subscription)

        if "fiscalYear" not in calls.columns:
            return ticker, calls, False

        calls["date"] = pd.to_datetime(calls["date"])
        calls = calls.sort_values("date")

        if latest:
            calls = calls.tail(1)
        else:
            calls = (
                calls.set_index("date")
                .truncate(before=start_date, after=end_date)
                .reset_index()
            )

        rows = {}
        for fiscal_year, quarter, call_date in zip(
            calls["fiscalYear"], calls["quarter"], calls["date"]
        ):
            result = transcript(ticker, int(fiscal_year), int(quarter))
            if "content" in result.columns and not result.empty:
                rows[f"{int(fiscal_year)}Q{int(quarter)}"] = {
                    "Date": call_date,
                    "Transcript": result["content"].iloc[0],
                }

        transcripts = pd.DataFrame.from_dict(rows, orient="index")
        transcripts.index.name = "Fiscal Period"

        return ticker, transcripts, not transcripts.empty

    ticker_list = _to_ticker_list(tickers)
    _require_fmp_api_key(api_key)

    return _collect_per_ticker_frames(
        worker, ticker_list, "earnings call transcripts", user_subscription
    )
