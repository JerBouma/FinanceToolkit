"""YFinance Module"""

__docformat__ = "google"

import logging
import re
import sqlite3
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import peewee
import yfinance as yf
import yfinance.exceptions

from financetoolkit import helpers
from financetoolkit.cache import frame_model, policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.currencies_model import to_yahoo_finance_symbol
from financetoolkit.utilities import error_model, logger_model
from financetoolkit.utilities.error_model import check_option
from financetoolkit.utilities.sources_model import record_source

logger = logger_model.get_logger()

# The raw name of the row holding each fiscal period's last day, mapped to Period End Date by statistics_yf.csv as FinancialModelingPrep's periodEndDate is by statistics.csv.  # noqa: E501
PERIOD_END_DATE_ITEM = "periodEndDate"

# The fiscal quarter of each period (Q1 to Q4, FY for a year), mapped to Period by statistics_yf.csv as FinancialModelingPrep's period is by statistics.csv.  # noqa: E501
FISCAL_PERIOD_ITEM = "period"

# The reporting currency and fiscal year end of a company change almost never, so they are resolved once per ticker per process rather than on every statement request.  # noqa: E501
COMPANY_INFORMATION_CACHE: dict[str, dict[str, str]] = {}


# yfinance keeps its caches of time zones, cookies and ISINs in SQLite databases shared by every thread. A thread that writes to one while another checkpoints it gets "database is locked" (an OperationalError of peewee, or of sqlite3 underneath), which says nothing about the ticker, so a download is tried once more before it counts as failed.  # noqa: E501
CACHE_DATABASE_ERRORS = (peewee.OperationalError, sqlite3.OperationalError)


# Whether yfinance's own log is muted for the current call. A context variable rather than the logger's level, so concurrent calls cannot leave the logger muted for good (one restoring the level another muted), and the worker threads of helpers.run_in_parallel, which copy the context, are muted with their caller.  # noqa: E501
YFINANCE_QUIET: ContextVar[bool] = ContextVar("yfinance_quiet", default=False)

logging.getLogger("yfinance").addFilter(lambda _record: not YFINANCE_QUIET.get())


@contextmanager
def quiet_yfinance() -> Generator[None]:
    """
    Silences yfinance's own log while a caller retrieves data it reports on itself: yfinance
    logs every ticker without prices ("possibly delisted") and every answer of Yahoo Finance
    with an error, which the caller reports in one line or counts as a failed request.

    Yields:
        None: The block runs with the yfinance log muted for this call.
    """
    token = YFINANCE_QUIET.set(True)
    try:
        yield
    finally:
        YFINANCE_QUIET.reset(token)


def is_cache_database_locked(error: Exception) -> bool:
    """
    Whether an error of yfinance's cache database is a lock held by another thread, which a
    second attempt can get past, rather than a lasting fault such as a missing table, a
    read-only file or a disk error.

    Args:
        error (Exception): The OperationalError raised by the cache database.

    Returns:
        bool: Whether the database was locked or busy.
    """
    text = str(error).lower()
    return "locked" in text or "busy" in text


def report_cache_database_error(error: Exception) -> None:
    """
    Report a cache database error of yfinance that is not a passing lock, since a missing table,
    a read-only file or a disk error fails every download until it is fixed and says nothing
    about the ticker. The message is the same for every ticker, so a caller running inside
    logger_model.log_each_message_once writes it once.

    Args:
        error (Exception): The OperationalError raised by the cache database.
    """
    if not is_cache_database_locked(error):
        logger.warning(
            "yfinance's cache database could not be used (%s), so the data from Yahoo Finance "
            "could not be retrieved.",
            error,
        )


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
    ticker_info = yf.Ticker(to_yahoo_finance_symbol(ticker))

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
    except (OSError, IndexError, AttributeError):
        # OSError covers the network errors of both urllib and curl_cffi, which yfinance
        # uses in recent versions, such as a timeout or a refused connection.
        error_model.report_request_failure()
        return pd.DataFrame()
    except CACHE_DATABASE_ERRORS as error:
        report_cache_database_error(error)
        error_model.report_request_failure()
        return pd.DataFrame()
    except yf.exceptions.YFRateLimitError:
        error_model.report_request_failure()
        error_code = (
            "YFINANCE RATE LIMIT REACHED FALLBACK"
            if fallback
            else "YFINANCE RATE LIMIT REACHED"
        )
        return pd.DataFrame(columns=[error_code])

    if not isinstance(financial_statement, pd.DataFrame):
        # yfinance only hands back a dict when asked with as_dict=True, which never
        # happens here; treating anything else as "no data" keeps the fallback path.
        return pd.DataFrame()

    if not financial_statement.empty:
        record_source(policy_model.YAHOO_FINANCE)

    if financial_statement.empty:
        error_code = (
            "YFINANCE RATE LIMIT OR NO DATA FOUND FALLBACK"
            if fallback
            else "YFINANCE RATE LIMIT OR NO DATA FOUND"
        )
        return pd.DataFrame(columns=[error_code])

    period_end_dates = pd.DatetimeIndex(financial_statement.columns)

    # yfinance returns dates as columns and items as rows; convert to periods. A fiscal period is labelled with the calendar period holding most of it, at both frequencies and identically to the FinancialModelingPrep path.  # noqa: E501
    financial_statement.columns = helpers.convert_period_end_dates_to_calendar_periods(
        period_end_dates=period_end_dates,
        quarter=quarter,
        ticker=ticker,
        fiscal_year_adjustments=fiscal_year_adjustments,
    )

    # The label above is a calendar period, so the date Yahoo Finance reports, the actual last day of each fiscal period, is kept as a row (picked up as the Period End Date statistic by get_statistics_statement): daily ratios apply a period's figures from that day on and currency translation takes its closing and average rates from the fiscal period it ends. Added before the duplicates are dropped so the date stays with the period that is kept.  # noqa: E501
    financial_statement.loc[PERIOD_END_DATE_ITEM] = period_end_dates.strftime(
        "%Y-%m-%d"
    )

    if financial_statement.columns.duplicated().any():
        financial_statement = financial_statement.loc[
            :, ~financial_statement.columns.duplicated()
        ]

    # Left as NaN, not filled with 0, matching the Toolkit-wide convention for unreported line items.
    if financial_statement.isna().to_numpy().any():
        financial_statement = financial_statement.infer_objects()

    return financial_statement


def get_statistics_statement(
    ticker: str,
    periods: pd.Index,
    financial_statement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Builds the statistics statement for a Yahoo Finance sourced ticker, mirroring the
    statistics that FinancialModelingPrep returns alongside its financial statements.

    Three statistics Yahoo Finance publishes are needed by the Toolkit:

        - The currency the financial statements are reported in, exposed as
          ``financialCurrency`` on the quote summary. It is what the currency conversion
          compares against the currency the instrument trades in; without it a Yahoo
          sourced ticker's statements would stay in a different currency from its own
          price history. It is a property of the company rather than of a period, so the
          same value is repeated across every period, matching the shape
          FinancialModelingPrep returns.
        - The last day of each fiscal period (``periodEndDate``, the Period End Date), the
          column date Yahoo Finance reports before get_financial_statement relabels it to a
          calendar period. Apple's fiscal year labelled 2023 ends on 30 September 2023,
          which is the day from which daily ratios apply its figures and over which
          currency translation takes its closing and average rates. It is read from the
          ``periodEndDate`` row get_financial_statement adds to the statement; when the
          statement does not hold it (such as a statement cached without it), the row is
          left out and the Toolkit falls back to the latest date the period can end on.
          Yahoo Finance reports the end of the month a period stands for, so a 52/53-week
          period ending a few days into the next month (Home Depot's quarter ending 3 August
          2025) carries the month end before it (31 July 2025).
        - The fiscal quarter of each quarterly period (``period``, Q1 to Q4, and FY for a
          year), as FinancialModelingPrep reports it. It is derived from the last fiscal year
          end on the quote summary (``lastFiscalYearEnd``): the quarter ending in the month
          the fiscal year ends is Q4, which is published with the annual report and so gets
          the annual reporting deadline when no filing date is known. Left out when Yahoo
          Finance does not report a fiscal year end.

    Args:
        ticker (str): the ticker to retrieve the statistics for.
        periods (pd.Index): the reporting periods to report the statistics against,
            normally the columns of the financial statement of the same ticker.
        financial_statement (pd.DataFrame | None, optional): the financial statement as
            returned by get_financial_statement, whose ``periodEndDate`` row holds the
            fiscal period end dates. Defaults to None, which leaves the Period End Date out.

    Returns:
        pd.DataFrame: A DataFrame with the Yahoo Finance statistics names as index and
        the given periods as columns. Empty when Yahoo Finance reports none of these
        statistics for the ticker.
    """
    if len(periods) == 0:
        return pd.DataFrame()

    statistics = {}

    information = get_company_information(ticker)
    reported_currency = information["financialCurrency"]

    if reported_currency:
        statistics["financialCurrency"] = pd.Series(
            [reported_currency] * len(periods), index=periods, dtype=object
        )

    if (
        financial_statement is not None
        and PERIOD_END_DATE_ITEM in financial_statement.index
    ):
        statistics[PERIOD_END_DATE_ITEM] = (
            financial_statement.loc[PERIOD_END_DATE_ITEM]
            .reindex(periods)
            .astype(object)
        )

        if information["lastFiscalYearEnd"] and isinstance(periods, pd.PeriodIndex):
            if periods.freqstr.startswith("Q"):
                # Months after the fiscal year end month, counted as the labels count them (a period ending in the first five days of a month ends in the month before), so Home Depot's quarter ending 3 August 2025 (fiscal year ending 1 February) is Q2.  # noqa: E501
                end_months = helpers.determine_period_end_months(
                    pd.to_datetime(statistics[PERIOD_END_DATE_ITEM])
                )
                year_end_month = helpers.determine_period_end_months(
                    pd.Series(pd.Timestamp(information["lastFiscalYearEnd"]))
                )[0]
                fiscal_quarters = [
                    (
                        None
                        if pd.isna(end_month)
                        else f"Q{(end_month.month - year_end_month.month - 1) % 12 // 3 + 1}"
                    )
                    for end_month in end_months
                ]
            else:
                fiscal_quarters = ["FY"] * len(periods)

            statistics[FISCAL_PERIOD_ITEM] = pd.Series(
                fiscal_quarters, index=periods, dtype=object
            )

    if not statistics:
        return pd.DataFrame()

    return pd.DataFrame(statistics).T.reindex(columns=periods)


def get_company_information(ticker: str) -> dict[str, str]:
    """
    Retrieves the company information from Yahoo Finance that the statistics statement
    needs, with one request to the quote summary:

        - ``financialCurrency``: the currency a company reports its financial statements
          in. This is not necessarily the currency the instrument trades in: Shell reports
          in USD while its London listing trades in GBp, and comparing a statement to a
          price without accounting for that compares two different currencies.
        - ``lastFiscalYearEnd``: the last day of the latest fiscal year (YYYY-MM-DD), which
          says which quarter closes the fiscal year.

    Both only change when the company changes its reporting currency or fiscal year, so
    they are cached per ticker both in memory and in the incremental cache.

    Args:
        ticker (str): the ticker to retrieve the information for.

    Returns:
        dict[str, str]: The ISO currency code the statements are reported in and the last
        fiscal year end, each an empty string when Yahoo Finance does not report it.
    """
    if ticker in COMPANY_INFORMATION_CACHE:
        record_source(policy_model.YAHOO_FINANCE)
        return COMPANY_INFORMATION_CACHE[ticker]

    cache = get_active_cache()

    if cache is not None:
        cached_information = cache.get(
            source=policy_model.YAHOO_FINANCE,
            dataset="company_information",
            entity=ticker,
        )

        if cached_information is not None:
            COMPANY_INFORMATION_CACHE[ticker] = cached_information
            return cached_information

    try:
        information = yf.Ticker(to_yahoo_finance_symbol(ticker)).get_info() or {}
    except (
        OSError,
        IndexError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        yf.exceptions.YFException,
    ):
        error_model.report_request_failure()
        return {"financialCurrency": "", "lastFiscalYearEnd": ""}
    except CACHE_DATABASE_ERRORS as error:
        report_cache_database_error(error)
        error_model.report_request_failure()
        return {"financialCurrency": "", "lastFiscalYearEnd": ""}

    record_source(policy_model.YAHOO_FINANCE)
    fiscal_year_end = information.get("lastFiscalYearEnd")

    company_information = {
        # financialCurrency is the statement currency. currency is the trading currency and is only a fallback, since for most listings the two are in fact the same.  # noqa: E501
        "financialCurrency": str(
            information.get("financialCurrency") or information.get("currency") or ""
        ),
        # Reported as a Unix timestamp at midnight UTC of the fiscal year end.
        "lastFiscalYearEnd": (
            pd.Timestamp(fiscal_year_end, unit="s").strftime("%Y-%m-%d")
            if isinstance(fiscal_year_end, int | float)
            else ""
        ),
    }

    COMPANY_INFORMATION_CACHE[ticker] = company_information

    if cache is not None and any(company_information.values()):
        cache.set(
            source=policy_model.YAHOO_FINANCE,
            dataset="company_information",
            entity=ticker,
            data=company_information,
        )

    return company_information


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

    # The widened window is what is actually requested, matching the FinancialModelingPrep path. Fetching only the requested window would leave the first return in range NaN because it has no preceding close to compare against.  # noqa: E501
    start_date_string = start_date.strftime("%Y-%m-%d")
    end_date_string = end_date.strftime("%Y-%m-%d")

    try:

        # auto_adjust=False matches FMP (True made Close disagree by the dividend history, 1.8% median, AAPL 2022-2023); repair is disabled for synthetic instruments since yfinance's split-repair misfires on them (^IRX 100x, CL=F 10,000x after oil went negative).  # noqa: E501
        repair_splits = not (ticker.startswith("^") or "=" in ticker)

        history_parameters = {
            "start": start_date_string,
            "end": end_date_string,
            "interval": interval,
            "actions": True,
            "auto_adjust": False,
        }

        for attempt in range(2):
            try:
                try:
                    historical_data = yf.Ticker(
                        to_yahoo_finance_symbol(ticker)
                    ).history(**history_parameters, repair=repair_splits)
                except ModuleNotFoundError:
                    # The repair needs yfinance's optional "repair" extra (scikit-learn in recent versions), which a plain install lacks; the prices are then taken unrepaired rather than lost.  # noqa: E501
                    logger.info(
                        "yfinance's price repair needs its 'repair' extra (pip install 'yfinance[repair]'), "
                        "so the prices of %s are not repaired.",
                        ticker,
                    )
                    historical_data = yf.Ticker(
                        to_yahoo_finance_symbol(ticker)
                    ).history(**history_parameters, repair=False)
                break
            except CACHE_DATABASE_ERRORS as error:
                # A cache database locked by another thread's download is retried once; a second lock, or any other database error (which a retry would hit again), counts as a failed download.  # noqa: E501
                if attempt == 1 or not is_cache_database_locked(error):
                    raise

        if (
            interval == "1d"
            and ticker.upper().endswith("=F")
            and len(historical_data) <= 1
            and (end_date - start_date).days > 1
        ):
            # Yahoo Finance answers the daily (and weekly, monthly) history of some continuous futures, such as Dutch TTF gas (TTF=F), with the latest bar only, while it keeps their hourly bars for the last 730 days. The days are rebuilt from those bars, in the exchange's time zone, so the close is the last trade of the day rather than the settlement price.  # noqa: E501
            hourly_data = yf.Ticker(ticker).history(
                start=max(start_date, datetime.now() - timedelta(days=729)).strftime(
                    "%Y-%m-%d"
                ),
                end=end_date_string,
                interval="1h",
                auto_adjust=False,
                repair=False,
            )

            if len(hourly_data) > len(historical_data):
                logger.warning(
                    "Yahoo Finance has no daily history of %s, so its days are built from the "
                    "hourly bars of the last 730 days: the close is the last trade of each day "
                    "rather than the settlement price.",
                    ticker,
                )
                historical_data = (
                    hourly_data.groupby(hourly_data.index.date)
                    .agg(
                        {
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum",
                        }
                    )
                    .dropna(subset=["Close"])
                )
                historical_data.index = pd.DatetimeIndex(historical_data.index)

        # Columns can occasionally be duplicated, so they are checked and dropped.
        historical_data = historical_data.loc[:, ~historical_data.columns.duplicated()]

        if "Adj Close" not in historical_data and historical_data.columns.nlevels == 1:
            historical_data.loc[:, "Adj Close"] = historical_data.loc[
                :, "Close"
            ].to_numpy()

    except (OSError, IndexError):
        error_model.report_request_failure()
        return pd.DataFrame()
    except CACHE_DATABASE_ERRORS as error:
        report_cache_database_error(error)
        error_model.report_request_failure()
        return pd.DataFrame()
    except yf.exceptions.YFRateLimitError:
        error_model.report_request_failure()
        error_code = "YFINANCE RATE LIMIT REACHED" + (" FALLBACK" if fallback else "")
        return pd.DataFrame(columns=[error_code])
    except yf.exceptions.YFException:
        # Such as a delisted ticker or one without prices in the range.
        return pd.DataFrame()

    if historical_data.empty:
        # No prices in the range, such as before the listing.
        return pd.DataFrame()

    record_source(policy_model.YAHOO_FINANCE)

    if historical_data.loc[start:end].empty:
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
        # In case tickers are presented in percentages or similar
        with np.errstate(divide="ignore", invalid="ignore"):
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

    These describe the instrument itself (its currency, exchange and listing date)
    rather than its price, so they change very rarely and are cached per ticker.

    Args:
        ticker (str): the ticker to retrieve statistics for.

    Returns:
        pd.Series: A Series containing the statistics for the given ticker.
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

    try:
        # The metadata of the chart, which yfinance requests with the last five days of hourly bars (and, for a ticker it has not seen before, a first request for its exchange time zone). Without price repair the currency is the one Yahoo Finance quotes the prices in (GBp for a London listing), as in the historical data. yfinance logs a symbol it does not know, which is counted as a failed request below instead.  # noqa: E501
        with quiet_yfinance():
            statistics = dict(
                yf.Ticker(to_yahoo_finance_symbol(ticker)).get_history_metadata(
                    repair=False
                )
            )

        # The dates are the exchange's calendar dates: a timestamp read in the machine's own
        # timezone would move a day for users east or west of it.
        exchange_offset = timedelta(seconds=statistics.get("gmtoffset") or 0)

        # Counted from the epoch rather than with datetime.fromtimestamp, which raises an
        # OSError on Windows for the negative timestamps of stocks first traded before 1970
        # (such as KO, IBM and XOM).
        epoch = datetime(1970, 1, 1, tzinfo=UTC)

        for timestamp_data in ["firstTradeDate", "regularMarketTime"]:
            if statistics.get(timestamp_data):
                statistics[timestamp_data] = (
                    epoch
                    + timedelta(seconds=statistics[timestamp_data])
                    + exchange_offset
                ).strftime("%Y-%m-%d")
    except (
        OSError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        yf.exceptions.YFException,
    ):
        # Yahoo Finance knows no chart of a symbol such as the "Portfolio" a Portfolio's Toolkit includes, which yfinance answers with metadata lacking the exchange time zone (a KeyError). A timeout, a rate limit or an unexpected answer is reported the same way, so one ticker does not fail the rest.  # noqa: E501
        error_model.report_request_failure()
        return pd.Series()
    except CACHE_DATABASE_ERRORS as error:
        report_cache_database_error(error)
        error_model.report_request_failure()
        return pd.Series()

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

    statistics = pd.Series(
        {name: statistics[key] for key, name in columns.items() if key in statistics},
        dtype=object,
    )

    if statistics.empty:
        error_model.report_request_failure()
        return statistics

    record_source(policy_model.YAHOO_FINANCE)

    if cache is not None:
        cache.set(
            source=policy_model.YAHOO_FINANCE,
            dataset="historical_statistics",
            entity=ticker,
            data=statistics,
        )

    return statistics


def convert_isin_to_ticker(isin_code: str) -> str:
    """
    Converts an ISIN code to a ticker symbol with the search of Yahoo Finance, which ranks
    the primary listing first.

    Args:
        isin_code (str): The ISIN code to convert.

    Returns:
        str: The corresponding ticker symbol if found, otherwise the original ISIN code (or
        the input itself when it is not an ISIN).
    """
    if not re.match("^([A-Z]{2})([A-Z0-9]{9})([0-9])$", isin_code):
        return isin_code

    try:
        # Only the best match is needed, so no news, lists or other parts of the search are requested.  # noqa: E501
        quotes = yf.Search(
            isin_code,
            max_results=1,
            news_count=0,
            lists_count=0,
            include_cb=False,
            recommended=0,
        ).quotes
        symbol = quotes[0]["symbol"] if quotes else None
    except (OSError, yf.exceptions.YFException) as error:
        error_model.report_request_failure()
        logger.warning(
            "Request failed for ISIN %s: %s. Returning ISIN.", isin_code, error
        )
        return isin_code
    except CACHE_DATABASE_ERRORS as error:
        report_cache_database_error(error)
        error_model.report_request_failure()
        logger.warning(
            "Request failed for ISIN %s: %s. Returning ISIN.", isin_code, error
        )
        return isin_code
    except (KeyError, ValueError, IndexError, TypeError):
        logger.warning(
            "Could not parse response for ISIN %s. Returning ISIN.", isin_code
        )
        return isin_code

    if not symbol:
        logger.warning(
            "Could not find a ticker for ISIN %s. Returning ISIN.", isin_code
        )
        return isin_code

    record_source(policy_model.YAHOO_FINANCE)
    logger.info("Converted ISIN %s to ticker %s", isin_code, symbol)

    return symbol


def report_failed_request(error: Exception) -> None:
    """
    Reports an error of yfinance as a failed request when it is one (no connection, a timeout,
    a rate limit or a broken cache database), rather than Yahoo Finance having no such data.

    Args:
        error (Exception): The error yfinance raised.
    """
    if isinstance(error, CACHE_DATABASE_ERRORS):
        report_cache_database_error(error)
        error_model.report_request_failure()
    elif isinstance(error, OSError | yf.exceptions.YFRateLimitError):
        error_model.report_request_failure()


def get_option_expiry_dates(ticker: str) -> list[str]:
    """
    Retrieves the dates on which the listed options of a ticker expire.

    Args:
        ticker (str): The ticker symbol to retrieve the option expiry dates for.

    Returns:
        list[str]: The option expiry dates in 'YYYY-MM-DD' format, empty when Yahoo Finance
        lists no options for the ticker or cannot be reached.
    """
    try:
        expiry_dates = list(yf.Ticker(ticker).options)
    except Exception as error:  # noqa: BLE001 - Yahoo Finance fails in many ways
        report_failed_request(error)
        logger.info("Yahoo Finance has no options of %s (%s).", ticker, error)
        return []

    if expiry_dates:
        record_source(policy_model.YAHOO_FINANCE)

    return expiry_dates


def get_option_chain(
    ticker: str, expiration_date: str, put_option: bool = False
) -> tuple[pd.DataFrame, float]:
    """
    Retrieves the option chain (calls or puts) of a ticker for one expiration date, as
    Yahoo Finance reports it, with the price of the underlying quoted with the chain.

    Args:
        ticker (str): The ticker symbol to retrieve the option chain for.
        expiration_date (str): The expiration date of the options (format: 'YYYY-MM-DD').
        put_option (bool, optional): If True, the puts; otherwise, the calls. Defaults to False.

    Returns:
        tuple[pd.DataFrame, float]: The option chain with Yahoo Finance's column names and the
        price of the underlying (NaN when Yahoo Finance does not report it). The chain is
        empty, with a warning, when it cannot be retrieved.
    """
    try:
        option_chain = yf.Ticker(ticker).option_chain(expiration_date)
        options = option_chain.puts if put_option else option_chain.calls
        # The price of the underlying quoted with the chain, so metrics that combine
        # the two use prices of the same moment rather than the last historical close.
        underlying_price = (option_chain.underlying or {}).get(
            "regularMarketPrice", np.nan
        )
    except Exception as error:  # noqa: BLE001 - Yahoo Finance fails in many ways
        report_failed_request(error)
        # A warning, so the answer says which ticker is left out and why.
        logger.warning(
            "No options data is available for %s expiring on %s, so it is left out (%s).",
            ticker,
            expiration_date,
            error,
        )
        return pd.DataFrame(), np.nan

    record_source(policy_model.YAHOO_FINANCE)

    return options, underlying_price


def get_dividends(ticker: str) -> pd.Series:
    """
    Retrieves every dividend Yahoo Finance reports for a ticker, split-adjusted as
    FinancialModelingPrep's adjDividend is.

    Args:
        ticker (str): The ticker symbol to retrieve the dividends for.

    Returns:
        pd.Series: The dividend per share indexed by a daily PeriodIndex of the payment
        dates, empty when Yahoo Finance has none or cannot be reached.
    """
    # When Yahoo Finance answers with an error (a rate limit, for example), yfinance logs it and returns None instead of raising, so anything but a Series counts as no dividends.  # noqa: E501
    try:
        dividends = yf.Ticker(to_yahoo_finance_symbol(ticker)).dividends
    except Exception as error:  # noqa: BLE001 - Yahoo Finance fails in many ways
        report_failed_request(error)
        logger.info(
            "Yahoo Finance dividends of %s could not be retrieved: %s", ticker, error
        )
        return pd.Series(dtype=float)

    if not isinstance(dividends, pd.Series) or dividends.empty:
        return pd.Series(dtype=float)

    record_source(policy_model.YAHOO_FINANCE)
    dividends.index = pd.DatetimeIndex(dividends.index).tz_localize(None).to_period("D")

    return dividends


# CME futures month codes, one letter per calendar month.
_MONTH_CODES: dict[int, str] = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}

# (root, exchange suffix) in Yahoo's dated futures notation, e.g. CLZ26.NYM.
COMMODITY_TICKERS: dict[str, tuple[str, str]] = {
    "Crude Oil": ("CL", "NYM"),
    "Natural Gas": ("NG", "NYM"),
    "Gold": ("GC", "CMX"),
    "Silver": ("SI", "CMX"),
    "Copper": ("HG", "CMX"),
    "Corn": ("ZC", "CBT"),
    "Wheat": ("ZW", "CBT"),
    "Soybeans": ("ZS", "CBT"),
}


def get_commodity_forward_curve(
    commodity: str,
    start_date: str,
    end_date: str,
    contracts: int = 12,
) -> pd.DataFrame:
    """
    Retrieve the futures curve of a commodity from Yahoo Finance: the daily closing price of each dated futures
    contract over the coming calendar months.

    These are the contracts of the coming calendar months, such as Crude Oil's December 2026 and January 2027
    contracts, rather than a single continuous or spot price.

    This is what a Schwartz-Smith (2000) two-factor commodity price model needs to
    back out the convenience-yield term structure under the risk-neutral (Q)
    measure -- the curve's shape (contango or backwardation) at each point in time
    is exactly what a single spot price series cannot reveal.

    Not every commodity has a listed contract for every calendar month (grains in
    particular only trade specific delivery months), so months with no listed (or an
    expired) contract are skipped and later months are tried instead, until the requested
    number of contracts is found.

    Args:
        commodity (str): The commodity to retrieve the curve for. Options are: "Crude Oil" (WTI crude oil), "Natural
            Gas" (Henry Hub natural gas), "Gold" (gold), "Silver" (silver), "Copper" (copper), "Corn" (corn), "Wheat"
            (wheat), "Soybeans" (soybeans).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        contracts (int, optional): The number of contracts to return, the nearest
            listed ones from the current month on. Defaults to 12.

    Raises:
        InvalidArgumentError: If the commodity is not one of the supported names.

    Returns:
        pd.DataFrame: A DataFrame indexed by date, with one column per contract
            labeled by its delivery month (e.g. "2026-12"), containing that
            contract's daily closing price over its trading life. Columns are NaN
            outside the date range the contract actually traded in.
    """
    check_option("commodity", commodity, tuple(COMMODITY_TICKERS))

    root, exchange = COMMODITY_TICKERS[commodity]
    today = datetime.today()
    cache = get_active_cache()

    curve: dict[str, pd.Series] = {}
    tried, reached = 0, True
    # The months are scanned until `contracts` contracts are found, since the nearest ones
    # may have expired and some commodities only list a few delivery months a year.
    for months_ahead in range(contracts * 3 + 3):
        if len(curve) >= contracts:
            break
        month_index = today.month - 1 + months_ahead
        year = today.year + month_index // 12
        month = month_index % 12 + 1

        ticker = f"{root}{_MONTH_CODES[month]}{year % 100:02d}.{exchange}"

        # Each contract is cached separately, so a rebuild refetches only stale months.
        cached_contract = None
        fetch_start, fetch_end = start_date, end_date

        if cache is not None:
            plan = cache.plan(
                source=policy_model.YAHOO_FINANCE,
                dataset="futures",
                entities=[ticker],
                start=start_date,
                end=end_date,
            )
            cached_contract = plan.cached.get(ticker)
            fetch_span = plan.get_fetch_span(ticker)

            if fetch_span is None:
                if cached_contract is not None and not cached_contract.empty:
                    curve[f"{year}-{month:02d}"] = cached_contract["Close"]

                continue

            fetch_start = fetch_span[0].strftime("%Y-%m-%d")
            fetch_end = fetch_span[1].strftime("%Y-%m-%d")

        # Yahoo Finance's end date is exclusive, so the day after includes the end date.
        # yfinance logs every month without a listed contract as "possibly delisted",
        # which is expected while scanning, so its logger is quiet for the request.
        tried += 1
        try:
            with quiet_yfinance():
                contract_data = yf.Ticker(ticker).history(
                    start=fetch_start,
                    end=(pd.Timestamp(fetch_end) + pd.Timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    ),
                )
        except (
            OSError,
            yf.exceptions.YFRateLimitError,
            *CACHE_DATABASE_ERRORS,
        ) as error:
            # curl's connection errors are OSErrors; the other months would fail alike, as they would after a rate limit or with a broken cache database.  # noqa: E501
            report_failed_request(error)
            logger.error(
                "Could not reach Yahoo Finance for the %s futures (%s).",
                commodity,
                error,
            )
            reached = False
            break

        if not contract_data.empty:
            record_source(policy_model.YAHOO_FINANCE)
            contract_data.index = pd.to_datetime(contract_data.index).tz_localize(None)
            contract_data = frame_model.slice_frame(
                contract_data, fetch_start, fetch_end
            )

            if cache is not None:
                cache.store(
                    source=policy_model.YAHOO_FINANCE,
                    dataset="futures",
                    entity=ticker,
                    data=contract_data,
                    start=fetch_start,
                    end=fetch_end,
                )

        if cached_contract is not None and not cached_contract.empty:
            contract_data = frame_model.merge_frames(cached_contract, contract_data)
            contract_data = frame_model.slice_frame(contract_data, start_date, end_date)

        if contract_data.empty:
            continue

        curve[f"{year}-{month:02d}"] = contract_data["Close"]

    if not curve:
        if tried and reached:
            logger.warning(
                "Yahoo Finance returned no data for any of the %s %s futures contracts "
                "tried, so no futures curve is returned.",
                tried,
                commodity,
            )
        return pd.DataFrame()

    forward_curve = pd.DataFrame(curve)
    expiration_dates = pd.to_datetime(forward_curve.index)

    # Cached contracts are already tz-naive, and tz_localize(None) rejects those.
    if expiration_dates.tz is not None:
        expiration_dates = expiration_dates.tz_localize(None)

    forward_curve.index = expiration_dates.to_period(freq="D")
    forward_curve.index.name = "Date"
    forward_curve = forward_curve.sort_index()

    return forward_curve
