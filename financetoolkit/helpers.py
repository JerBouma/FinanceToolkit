"""Helpers Module"""

__docformat__ = "google"

import contextlib
import contextvars
import importlib
import inspect
import numbers
import os
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from decimal import ROUND_HALF_UP, Decimal
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from financetoolkit.cache.request_model import redact_credentials
from financetoolkit.utilities import error_model, logger_model
from financetoolkit.utilities.dataframe_model import FilteredClass
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    bounded_ffill,
    drop_non_finite,
)

logger = logger_model.get_logger()


def import_functions_when_used(
    package: str, modules: dict[str, list[str]]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """
    Make the functions of a package's model modules importable from the package itself
    (`from financetoolkit.models import get_altman_z_score`), importing each module only
    when one of its functions is first used (PEP 562). The package's `__init__.py` assigns
    the returned functions to its `__getattr__` and `__dir__`.

    Args:
        package (str): The name of the package, its `__name__`.
        modules (dict[str, list[str]]): The full name of each model module with the
            names of the functions it provides.

    Returns:
        tuple[Callable[[str], Any], Callable[[], list[str]]]: The package's `__getattr__`,
        which imports and returns a function by name, and its `__dir__`.
    """
    module_of = {name: module for module, names in modules.items() for name in names}

    def __getattr__(name: str) -> Any:  # noqa: N807
        if name not in module_of:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(module_of[name]), name)
        # Stored on the package, so later lookups don't come back here.
        setattr(sys.modules[package], name, value)
        return value

    def __dir__() -> list[str]:  # noqa: N807
        return sorted(set(vars(sys.modules[package])) | set(module_of))

    return __getattr__, __dir__


# Set FINANCETOOLKIT_MAX_WORKERS to raise or lower the number of API calls made at the same time; the default is a deliberate middle ground between fetching a large universe quickly and not tripping the data providers' rate limits (which the retry logic in fmp_model.get_financial_data absorbs when they are hit anyway).  # noqa: E501
MAX_WORKERS_ENVIRONMENT_VARIABLE = "FINANCETOOLKIT_MAX_WORKERS"

DEFAULT_MAX_WORKERS = 10


def determine_max_workers() -> int:
    """
    Determines the number of worker threads used for concurrent API calls, taking the
    FINANCETOOLKIT_MAX_WORKERS environment variable when it is set to a positive integer
    and falling back to DEFAULT_MAX_WORKERS otherwise.

    Returns:
        int: The number of worker threads to use.
    """
    max_workers = os.environ.get(MAX_WORKERS_ENVIRONMENT_VARIABLE, "").strip()

    if max_workers:
        try:
            if int(max_workers) > 0:
                return int(max_workers)

            raise ValueError("The number of workers must be positive.")
        except ValueError:
            logger.warning(
                "The %s environment variable is set to %s which is not a positive "
                "integer. Using the default of %d instead.",
                MAX_WORKERS_ENVIRONMENT_VARIABLE,
                max_workers,
                DEFAULT_MAX_WORKERS,
            )

    return DEFAULT_MAX_WORKERS


def run_in_parallel(
    worker: Callable,
    worker_args: Iterable[tuple],
    max_workers: int | None = None,
) -> list:
    """
    Runs a worker function over a collection of argument tuples with a bounded pool of
    threads, which is what every data collection function uses to make multiple API
    calls at the same time. For example, collecting historical data of 100 tickers
    this way takes seconds rather than minutes.

    A bounded pool (rather than one thread per ticker) keeps the number of concurrent
    API calls predictable regardless of how large the ticker universe is, which is
    kinder to both the data providers' rate limits and the local machine.

    A worker that fails on its data, with a KeyError, IndexError, ValueError or
    ZeroDivisionError (as handle_errors reports them), is logged and left out of the
    result, so one ticker without usable data does not discard the others' results. Any
    other exception, such as an AttributeError or TypeError that points to a defect, is
    re-raised here instead of dying silently inside its thread, as is every exception
    when FINANCETOOLKIT_STRICT_ERRORS is set.

    Args:
        worker (Callable): The function to execute for each tuple of arguments.
        worker_args (Iterable[tuple]): The argument tuples, one per call, e.g. one per
        ticker. Each tuple is unpacked into the worker.
        max_workers (int | None, optional): The maximum number of threads to use.
        Defaults to None, which uses the FINANCETOOLKIT_MAX_WORKERS environment
        variable when set and DEFAULT_MAX_WORKERS otherwise.

    Returns:
        list: The results of the workers that succeeded, in the order of worker_args,
        which is how the collection functions hand per-ticker results back without
        sharing state. A failed worker has no entry, so a caller that needs to know which
        call a result belongs to has the worker return that, e.g. the ticker.
    """
    worker_args = list(worker_args)

    if not worker_args:
        return []

    if max_workers is None:
        max_workers = determine_max_workers()

    with ThreadPoolExecutor(max_workers=min(max_workers, len(worker_args))) as executor:
        # Every worker runs in a copy of the caller's context, so context-bound state,
        # such as the messages an MCP tool call collects, also covers the workers.
        futures = [
            executor.submit(contextvars.copy_context().run, worker, *args)
            for args in worker_args
        ]

        results = []

        # Collecting in submission order keeps the result in the order of worker_args;
        # the calls themselves still run concurrently.
        for args, future in zip(worker_args, futures, strict=True):
            try:
                results.append(future.result())
            except error_model.REPORTED_ERRORS as error:
                if error_model.use_strict_errors():
                    raise

                # The first argument names the call (a ticker, a year, a series or a
                # URL); the others can hold an API key, and so can a URL or a message.
                # A function (one fetch per source) has no readable name to show.
                logger.error(
                    "No data could be retrieved for %s: %s: %s",
                    (
                        redact_credentials(str(args[0]))
                        if args and not callable(args[0])
                        else "one of the requests"
                    ),
                    type(error).__name__,
                    redact_credentials(str(error)),
                )

    return results


# A 52/53-week fiscal period ends on a fixed weekday, at most five days past the month end it stands for (Home Depot's quarter ending 5 May 2019 is its February to April quarter), so a period ending in the first five days of a month belongs to the month before. Periods ending later in a month are left in it: Costco's 12-week quarter ending 7 May 2023 follows one ending 12 February 2023.  # noqa: E501
MONTH_END_OVERRUN_DAYS = 5


def determine_period_end_months(
    period_end_dates: pd.DatetimeIndex | pd.Series,
) -> pd.PeriodIndex:
    """
    Determines the month each fiscal period ends in. A period ending in the first five days
    of a month counts as ending in the month before: a 52/53-week fiscal period ends on a fixed
    weekday a few days past the month end it stands for (Home Depot's quarter ending 3
    August 2025 is its May to July quarter), and Yahoo Finance reports such a period at the
    end of that month (31 July 2025).

    Args:
        period_end_dates (pd.DatetimeIndex | pd.Series): the final day of each fiscal
            reporting period, as reported by the data provider.

    Returns:
        pd.PeriodIndex: the month each period ends in, at monthly frequency.
    """
    end_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(period_end_dates).to_numpy()))

    return end_dates.to_period("M") - (end_dates.day <= MONTH_END_OVERRUN_DAYS).astype(
        int
    )


def convert_period_end_dates_to_calendar_periods(
    period_end_dates: pd.DatetimeIndex | pd.Series,
    quarter: bool,
    ticker: str | None = None,
    fiscal_year_adjustments: dict | None = None,
) -> pd.PeriodIndex:
    """
    Labels fiscal reporting periods with the calendar period in which the majority of
    the period falls, so that a company with a non-December fiscal year end lines up
    with calendar time and with every other company in the same output.

    A fiscal period is a span of time, not a point, and the date a provider reports is
    only its final day. Labelling by that final day alone puts NVIDIA's fiscal year
    February 2023 - January 2024 in calendar 2024, even though eleven of its twelve
    months are 2023. The convention applied here is therefore the majority rule:

        - Yearly: a fiscal year ending in January through May has at least seven of its
          twelve months in the preceding calendar year, so it is labelled with that
          preceding year. A June through December year end keeps its own year.
        - Quarterly: a fiscal quarter ending in month m spans months m-2, m-1 and m, so
          at least two of its three months fall in the calendar quarter that contains
          month m-1. The quarter is therefore labelled with the calendar quarter of the
          month before its end. For the calendar aligned quarter ends (March, June,
          September and December) this is the quarter it already sat in, so only
          companies reporting on an off-calendar quarter cycle are relabelled.

    Both rules are the same rule at two frequencies: NVIDIA's fiscal 2024 is labelled 2023
    and its four quarters are labelled 2023Q1 through 2023Q4. A year ending in November,
    December or January keeps its four quarters within the label of its year. Any other
    year end splits them over two calendar years: Oracle's fiscal 2024 (June 2023 - May
    2024) is labelled 2023 with quarters 2023Q3 through 2024Q2, and Apple's fiscal 2024
    (October 2023 - September 2024) is labelled 2024 with quarters 2023Q4 through 2024Q3.
    A yearly figure is therefore not the sum of the four quarters with the same year label
    for such a company.

    A period ending in the first five days of a month is treated as ending in the month
    before (see determine_period_end_months). Companies with a 52/53-week fiscal year end
    their periods on a fixed weekday, up to a few days past a month end: Home Depot's
    quarter ending 3 August 2025 covers May to July and Western Digital's fiscal year
    ending 1 July 2022 covers July 2021 to June 2022. Yahoo Finance reports such periods at
    the end of the month before (31 July 2025 and 30 June 2022), FinancialModelingPrep at
    their actual last day, and both are labelled alike. Without this, two consecutive
    quarters (ending 2 August and 1 November) would share one calendar quarter.

    Args:
        period_end_dates (pd.DatetimeIndex | pd.Series): the final day of each fiscal
            reporting period, as reported by the data provider.
        quarter (bool): whether the periods are quarters. Yearly periods when False.
        ticker (str | None, optional): the ticker the periods belong to, used as the key
            in fiscal_year_adjustments. Defaults to None.
        fiscal_year_adjustments (dict | None, optional): a registry that collects, per
            ticker, every period whose label differs from its fiscal label. This is what
            the MCP server reports back to the user as a note. Defaults to None.

    Returns:
        pd.PeriodIndex: the calendar periods to label the reporting periods with, at
        yearly frequency when quarter is False and quarterly frequency when True.
    """
    end_months = determine_period_end_months(period_end_dates)

    if quarter:
        fiscal_periods = end_months.asfreq("Q")
        # The calendar quarter holding the middle month of the fiscal quarter.
        calendar_periods = (end_months - 1).asfreq("Q")
    else:
        fiscal_periods = end_months.asfreq("Y")
        # A fiscal year ending before June is labelled with the previous calendar
        # year; shifting back five months lands January to May in that year and
        # leaves June to December where they are.
        calendar_periods = (end_months - 5).asfreq("Y")

    shifted_mask = calendar_periods != fiscal_periods

    if (
        shifted_mask.any()
        and fiscal_year_adjustments is not None
        and ticker is not None
    ):
        # Reported as plain integers for years so the value stays readable, and as period labels for quarters where a bare integer would lose the quarter.  # noqa: E501
        adjustments = [
            {
                "fiscal_year": str(fiscal) if quarter else fiscal.year,
                "calendar_year": str(calendar) if quarter else calendar.year,
            }
            for fiscal, calendar in zip(
                fiscal_periods[shifted_mask], calendar_periods[shifted_mask]
            )
        ]

        # Called once per statement type (income, balance, cash, statistics), so the entries are merged rather than assigned -- assigning would let the last statement fetched silently discard the periods the earlier ones relabelled.  # noqa: E501
        existing = fiscal_year_adjustments.setdefault(ticker, [])
        existing.extend(
            adjustment for adjustment in adjustments if adjustment not in existing
        )

    return calendar_periods


PERIOD_END_DATE = "Period End Date"


def determine_period_end_dates(
    periods: pd.PeriodIndex,
    tickers: list[str],
    quarter: bool,
    statistics_statement: pd.DataFrame | None = None,
    fiscal_year_adjustments: dict | None = None,
) -> pd.DataFrame:
    """
    Determines the last day of the fiscal period behind every statement column per ticker.

    Statements are labelled with calendar periods (see
    convert_period_end_dates_to_calendar_periods), but a fiscal period ends on its own date:
    Microsoft's fiscal year labelled 2023 ends on 30 June 2023 and NVIDIA's fiscal year
    labelled 2023 on 28 January 2024. That date is what daily ratios need (the figures of a
    period apply from its end) and what currency translation needs (the closing rate on the
    period end and the average rate over the fiscal period, IAS 21.39-40).

    The date is taken from the Period End Date statistic, which FinancialModelingPrep and
    Yahoo Finance statements carry. When it is not available (an older cache or a dataset
    supplied to the Toolkit), the end of the last month the period can end in given its label
    is used, so that figures are not applied before the period ended (a 52/53-week period
    ending in the first five days of the next month counts as ending in that month, see
    convert_period_end_dates_to_calendar_periods):

        - A label that was not moved from its fiscal period ends at the end of the labelled
          period (31 December for a year, the last day of the quarter for a quarter).
        - A label that was moved back (listed in fiscal_year_adjustments) belongs to a fiscal
          year ending in January through May of the next year, so it ends at the latest on
          31 May of that year, or to a fiscal quarter ending in the month after the labelled
          quarter, so it ends at the latest on the last day of that month.

    Args:
        periods (pd.PeriodIndex): The statement periods (the columns of the statements).
        tickers (list[str]): The tickers to determine the end dates for.
        quarter (bool): Whether the periods are quarters. Years when False.
        statistics_statement (pd.DataFrame | None, optional): The statistics statement with a
            (ticker, item) index and the periods as columns, holding the Period End Date row
            when the source reports it. Defaults to None.
        fiscal_year_adjustments (dict | None, optional): The registry of relabelled periods
            per ticker, as filled by convert_period_end_dates_to_calendar_periods. Defaults
            to None.

    Returns:
        pd.DataFrame: The period end dates (Timestamps) with the tickers as index and the
        periods as columns.
    """
    periods = pd.PeriodIndex(periods)
    label_ends = pd.Series(
        periods.to_timestamp(how="end").normalize(),
        index=periods,
        dtype="datetime64[ns]",
    )

    if quarter:
        # The last day of the month after the labelled quarter.
        shifted_ends = pd.Series(
            ((periods.asfreq("M", how="end") + 1).to_timestamp(how="end")).normalize(),
            index=periods,
            dtype="datetime64[ns]",
        )
    else:
        # 31 May of the year after the labelled year.
        shifted_ends = pd.Series(
            pd.to_datetime([f"{period.year + 1}-05-31" for period in periods]),
            index=periods,
            dtype="datetime64[ns]",
        )

    reported = pd.DataFrame()

    if (
        statistics_statement is not None
        and not statistics_statement.empty
        and isinstance(statistics_statement.index, pd.MultiIndex)
        and PERIOD_END_DATE in statistics_statement.index.get_level_values(1)
    ):
        reported = statistics_statement.xs(PERIOD_END_DATE, level=1).apply(
            pd.to_datetime, errors="coerce"
        )
        reported.columns = pd.PeriodIndex(reported.columns)

    period_end_dates = {}

    for ticker in tickers:
        shifted_labels = {
            str(adjustment["calendar_year"])
            for adjustment in (fiscal_year_adjustments or {}).get(ticker, [])
        }
        fallback = label_ends.where(
            ~pd.Index([str(period) for period in periods]).isin(shifted_labels),
            shifted_ends,
        )

        if ticker in reported.index:
            ticker_dates = (
                reported.loc[ticker].reindex(periods).astype("datetime64[ns]")
            )
            period_end_dates[ticker] = ticker_dates.fillna(fallback)
        else:
            period_end_dates[ticker] = fallback

    return pd.DataFrame(period_end_dates, index=periods).T


ACCEPTED_DATE = "Accepted Date"
FILING_DATE = "Filing Date"
FISCAL_PERIOD = "Period"
DECEMBER = 12

# The latest a period's figures are expected to be public when no filing date is known: four
# months after a fiscal year end (the annual report deadline of the EU Transparency Directive
# and of a Form 20-F) and three months after an interim period end (the EU half-yearly report
# deadline, which also covers the 40 and 45 days of a Form 10-Q).
FALLBACK_FILING_DAYS = {"annual": 120, "interim": 90}

# No company files its statements within a week of the period end, so a date that close is the
# period end FinancialModelingPrep reports for companies that do not file with the SEC.
MINIMUM_FILING_DAYS = 7

# SEC filings accepted at or after the 16:00 New York close are first reflected in the next
# day's prices.
MARKET_CLOSE_HOUR = 16


def determine_availability_dates(
    period_end_dates: pd.DataFrame,
    statistics_statement: pd.DataFrame | None,
    quarter: bool,
) -> pd.DataFrame:
    """
    Determines the first day the figures of every statement period were public per ticker.

    A period's figures are only known once the company files them, weeks after the period
    ends, so a point-in-time analysis applies them from the filing instead of from the
    period end. The day is taken from:

        - The Accepted Date (the moment the SEC accepted the filing): that day, or the next
          day when the filing was accepted at or after the 16:00 close.
        - Otherwise the Filing Date, from the day after it, since its time is not known.
        - Otherwise the period end plus the longest common reporting deadline: 120 days for a
          fiscal year (and the quarter that closes it) and 90 days for another quarter (see
          FALLBACK_FILING_DAYS).

    A filing date within a week of the period end is not a real filing date
    (FinancialModelingPrep reports about the period end for companies that do not file with
    the SEC), so the deadline is used for those periods.

    Args:
        period_end_dates (pd.DataFrame): The last day of the fiscal period behind every
            statement column, with the tickers as index and the periods as columns, as
            returned by determine_period_end_dates.
        statistics_statement (pd.DataFrame | None): The statistics statement with a
            (ticker, item) index and the periods as columns, holding the Accepted Date and
            Filing Date rows when the source reports them.
        quarter (bool): Whether the periods are quarters. Years when False.

    Returns:
        pd.DataFrame: The availability dates (Timestamps) with the tickers as index and the
        periods as columns.
    """
    period_ends = period_end_dates.apply(pd.to_datetime).astype("datetime64[ns]")
    deadline = pd.Timedelta(
        days=FALLBACK_FILING_DAYS["interim" if quarter else "annual"]
    )
    availability = period_ends + deadline
    has_statistics = (
        statistics_statement is not None
        and not statistics_statement.empty
        and isinstance(statistics_statement.index, pd.MultiIndex)
    )

    if quarter:
        # The quarter that closes a fiscal year is published with the annual report, so it
        # gets the annual deadline: the quarter the source labels Q4, or December without one.
        fiscal_period = pd.DataFrame(
            index=period_ends.index, columns=period_ends.columns
        )

        if (
            has_statistics
            and FISCAL_PERIOD in statistics_statement.index.get_level_values(1)
        ):
            fiscal_period = statistics_statement.xs(FISCAL_PERIOD, level=1)
            fiscal_period.columns = pd.PeriodIndex(
                fiscal_period.columns, freq=period_ends.columns.freq
            )
            fiscal_period = fiscal_period.reindex(
                index=period_ends.index, columns=period_ends.columns
            )

        december = period_ends.apply(lambda dates: dates.dt.month == DECEMBER)
        fiscal_year_end = (fiscal_period == "Q4") | (fiscal_period.isna() & december)
        availability = availability.where(
            ~fiscal_year_end,
            period_ends + pd.Timedelta(days=FALLBACK_FILING_DAYS["annual"]),
        )

    if not has_statistics:
        return availability

    items = statistics_statement.index.get_level_values(1)

    def reported(item: str) -> pd.DataFrame:
        if item not in items:
            return pd.DataFrame(index=period_ends.index, columns=period_ends.columns)

        dates = statistics_statement.xs(item, level=1).apply(
            pd.to_datetime, errors="coerce", format="mixed"
        )
        dates.columns = pd.PeriodIndex(dates.columns, freq=period_ends.columns.freq)

        return dates.reindex(index=period_ends.index, columns=period_ends.columns)

    accepted = reported(ACCEPTED_DATE).astype("datetime64[ns]")
    after_close = accepted.apply(lambda column: column.dt.hour >= MARKET_CLOSE_HOUR)
    accepted_day = accepted.apply(lambda column: column.dt.normalize()).where(
        ~after_close,
        accepted.apply(lambda column: column.dt.normalize()) + pd.Timedelta(days=1),
    )
    filed_day = reported(FILING_DATE).astype("datetime64[ns]").apply(
        lambda column: column.dt.normalize()
    ) + pd.Timedelta(days=1)

    # Only dates well after the period end are filings; the rest falls back to the deadline.
    earliest_filing = period_ends + pd.Timedelta(days=MINIMUM_FILING_DAYS)
    accepted_day = accepted_day.where(accepted_day > earliest_filing)
    filed_day = filed_day.where(filed_day > earliest_filing + pd.Timedelta(days=1))

    return accepted_day.fillna(filed_day).fillna(availability)


def enrich_historical_data(
    historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    return_column: str = "Adj Close",
):
    """
    Enriches the historical data of a ticker, as retrieved from any data provider, with the
    following columns:

        - Return: The return for the given period.
        - Cumulative Return: The growth of 1 invested at the start, the product of (1 + Return):
          1.0 on the first date and 1.25 after a 25% gain.

    The return is calculated as the percentage change in the given return column.

    The cumulative return is calculated as the cumulative product of one plus the return,
    starting at 1 on the first date of the start to end window.

    Volatility, Excess Return and Excess Volatility are not calculated here. These are
    available as dedicated, reusable calculations in the Risk module (e.g.
    risk_model.get_volatility, risk_model.get_excess_volatility) and the Performance module
    (e.g. performance_model.get_excess_return).

    Args:
        historical_data (pd.DataFrame): A pandas DataFrame object containing the historical stock data
        for the given ticker(s).
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        return_column (str, optional): A string representing the column to use for return calculations.
            Defaults to "Adj Close".

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the enriched historical stock data for the given ticker(s).
    """

    # A price of 0, such as a faulty bar or a futures contract settling at 0, has no
    # return from or to it, and an infinite return would carry into every cumulative one.
    historical_data["Return"] = drop_non_finite(
        bounded_ffill(historical_data[return_column]).pct_change()
    )

    historical_data["Cumulative Return"] = 1

    adjusted_return = historical_data.loc[start:end, "Return"]

    with contextlib.suppress(IndexError):
        adjusted_return.iloc[0] = 0

    # A scalar rather than pd.Series(np.nan): assigned to a frame without rows, a Series
    # brings its own index along, adding a row labelled 0 to an otherwise empty response
    # that was then cached and merged into the dates.
    historical_data["Cumulative Return"] = np.nan

    historical_data.loc[start:end, "Cumulative Return"] = (
        1.0 + adjusted_return.fillna(0)
    ).cumprod()

    return historical_data


def prepare_supplied_historical_data(
    historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Checks historical data passed to the Toolkit and brings it into the shape the
    retrieved data has, so that every calculation can rely on it:

        - The columns have two levels: the field (Open, High, Low, Close, Adj Close, Volume,
          Dividends, Return, Cumulative Return) and the ticker.
        - The dates become a daily PeriodIndex named Date, sorted from old to new. Dates or
          date strings are converted, a PeriodIndex is kept as it is.
        - A date that occurs more than once keeps its last row, with a warning, as a repeated
          day would otherwise count twice in every average, sum and volatility.
        - Without a Return or Cumulative Return column, both are calculated per ticker from
          Adj Close, as for retrieved data (see enrich_historical_data). Supplied returns are
          used as given.

    The supplied frame itself is not changed.

    Args:
        historical_data (pd.DataFrame): The historical data passed to the Toolkit.
        start (str, optional): The start date of the Toolkit, from which the Cumulative
            Return starts at 1. Defaults to None.
        end (str, optional): The end date of the Toolkit. Defaults to None.

    Returns:
        pd.DataFrame: The checked historical data.

    Raises:
        ValueError: If the columns do not have the field and ticker levels, the index does not
            hold dates, or the returns cannot be calculated because Adj Close is missing.
    """
    if (
        not isinstance(historical_data.columns, pd.MultiIndex)
        or historical_data.columns.nlevels != 2  # noqa: PLR2004
    ):
        raise ValueError(
            "The historical data needs two column levels: the field (such as Close and Adj Close) "
            "and the ticker, e.g. ('Adj Close', 'AAPL'). For one ticker, pd.concat({'AAPL': data}, "
            "axis=1).swaplevel(axis=1) adds the ticker level."
        )

    index = historical_data.index

    if not isinstance(index, pd.PeriodIndex):
        # Integer positions would be read as nanoseconds since 1970 rather than rejected.
        if pd.api.types.is_numeric_dtype(index):
            raise ValueError(
                "The historical data needs dates as its index, found numbers instead. Set the "
                "date column as the index, e.g. data.set_index('Date')."
            )
        try:
            index = pd.DatetimeIndex(pd.to_datetime(index)).to_period("D")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"The index of the historical data could not be read as dates: {error}"
            ) from error

    historical_data = historical_data.set_axis(index.rename("Date"), axis=0)

    if not historical_data.index.is_monotonic_increasing:
        historical_data = historical_data.sort_index()

    duplicated = historical_data.index.duplicated(keep="last")

    if duplicated.any():
        logger.warning(
            "The historical data holds %s dates more than once (first: %s). The last row of each "
            "is used.",
            historical_data.index[duplicated].nunique(),
            historical_data.index[duplicated][0],
        )
        historical_data = historical_data[~duplicated]

    fields = historical_data.columns.unique(level=0)

    if "Return" not in fields or "Cumulative Return" not in fields:
        if "Adj Close" not in fields:
            raise ValueError(
                "The historical data needs an Adj Close column to calculate the Return and "
                "Cumulative Return from, or both of those columns."
            )

        tickers = historical_data.columns.unique(level=1)
        enriched = pd.concat(
            {
                ticker: enrich_historical_data(
                    historical_data.xs(ticker, level=1, axis=1).copy(), start, end
                )
                for ticker in tickers
            },
            axis=1,
        ).swaplevel(axis=1)
        historical_data = enriched[
            [
                (field, ticker)
                for field in enriched.columns.unique(level=0)
                for ticker in tickers
                if (field, ticker) in enriched.columns
            ]
        ]

    return historical_data


PERIODS_BY_FREQUENCY = {
    "D": "daily",
    "W": "weekly",
    "M": "monthly",
    "Q": "quarterly",
    "Y": "yearly",
    "A": "yearly",
}


def period_of_result(result: pd.DataFrame) -> str | None:
    """
    Determine the period of a result from the frequency of the dates on either axis.

    Args:
        result (pd.DataFrame): The result, with dates as index or as columns.

    Returns:
        str | None: "daily", "weekly", "monthly", "quarterly" or "yearly", or None when
            neither axis holds periods.
    """
    for axis in (result.index, result.columns):
        if isinstance(axis, pd.PeriodIndex):
            return PERIODS_BY_FREQUENCY.get(axis.freq.rule_code[0])

    return None


class LazyDict(dict):
    """
    A dictionary that retrieves an entry through `loader` when it is first read and keeps
    it, so a module loads only the data a calculation asks for (e.g. the prices of a single
    period). A plain dictionary of the same data behaves the same, without the loading.

    Args:
        loader (Callable[[str], Any]): Returns the entry of a key, and raises a KeyError for a
            key without an entry as a plain dictionary does.
    """

    def __init__(self, loader: Callable[[str], Any]):
        super().__init__()
        self._loader = loader

    def __missing__(self, key: str) -> Any:
        self[key] = self._loader(key)
        return self[key]

    def get(
        self, key: str, default: Any = None
    ) -> Any:  # ty: ignore[incompatible-method-override]
        try:
            return self[key]
        except KeyError:
            return default


def round_half_up(value: float | str | Decimal, decimals: int) -> float:
    """
    Rounds a number to a number of decimals with a half rounded up (away from zero), as
    statistical agencies round their published figures. Python's round() rounds a half to
    even and works on the binary float, so 0.125 and 2.675 would not round up.

    Args:
        value (float | str | Decimal): The number, a missing value is returned as is and
            an infinity as a float.
        decimals (int): The number of decimals to round to.

    Returns:
        float: The rounded number.
    """
    if not isinstance(value, str | Decimal) and pd.isna(value):
        return value

    # str gives the shortest decimal that is the float, so 2.675 rounds as 2.675.
    number = value if isinstance(value, Decimal) else Decimal(str(value))

    # An infinity (or NaN given as text) has no decimals to round, and quantize raises on it.
    if not number.is_finite():
        return float(number)

    return float(number.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def _has_own_portfolio_values(result: pd.DataFrame) -> bool:
    """
    Whether a result already holds values for the Portfolio, calculated from its own prices
    and returns (the Toolkit of a Portfolio has them as the "Portfolio" ticker).
    """
    if "Portfolio" in result.columns and not isinstance(result.columns, pd.MultiIndex):
        return bool(result["Portfolio"].notna().to_numpy().any())
    if "Portfolio" in result.index and not isinstance(result.index, pd.MultiIndex):
        return bool(result.loc[["Portfolio"]].notna().to_numpy().any())
    return False


def handle_portfolio(func):
    """
    A decorator that processes the result of a function to handle portfolio data.
    This decorator checks if "Portfolio" is in the `self._tickers` attribute and, if so,
    calculates the weighted average of the result DataFrame using `self._portfolio_weights`
    and appends it as a new row or column named "Portfolio".

    A result that already has values for the Portfolio keeps them: a measure calculated on
    the Portfolio's own prices and returns (a volatility, a Value at Risk, a drawdown)
    includes how the holdings move together, which a weighted average of the holdings'
    measures leaves out. The weighted average is for the measures that have no values
    for the Portfolio, such as the ratios of the financial statements.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The wrapped function with additional portfolio handling logic.

    Notes:
        - The decorated function should have a `self` parameter as the first argument.
        - The decorated function should return a DataFrame.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original function
        result = func(self, *args, **kwargs)

        # Check if "Portfolio" is in self._tickers, only for a result to add it to: a module
        # retrieves its tickers when first needed, which a failed calculation must not do.
        if (
            isinstance(result, pd.DataFrame)
            and isinstance(self._tickers, list | str)
            and "Portfolio" in self._tickers
            and not _has_own_portfolio_values(result)
        ):
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            # Merge defaults with kwargs, without overriding explicitly passed values
            for key, value in bound_args.arguments.items():
                if key not in kwargs:
                    kwargs[key] = value

            # Get the rounding parameter from kwargs or use a default value
            rounding = kwargs.get("rounding", self._rounding)
            lag = kwargs.get("lag", 1)
            growth = kwargs.get("growth", False)
            period = kwargs.get("period")

            if rounding is None:
                rounding = self._rounding

            # The weights belong to the periods of the result, which the frequency of its
            # dates tells most reliably: a method's default period differs per module
            # (e.g. daily for most risk metrics, yearly for the ratios).
            period = period_of_result(result) or period

            if period is None:
                period = "quarterly" if getattr(self, "_quarterly", False) else "yearly"

            # Select the appropriate portfolio weights
            weights = self._portfolio_weights.get(period, pd.DataFrame())

            # Exclude "Benchmark" from the weighted average calculation
            result_without_benchmark = (
                result.drop(columns=["Benchmark"])
                if "Benchmark" in result.columns
                else result
            )

            # Calculate the weighted average for each column
            if isinstance(result.columns, pd.PeriodIndex) and not isinstance(
                result.columns, pd.MultiIndex
            ):
                # reindex fills periods missing from weights with NaN rather than raising.
                weights = weights.reindex(result_without_benchmark.columns).T

                # Only the weights of the tickers with a value count, so a missing value
                # does not pull the portfolio towards zero.
                weighted_averages = apply_rounding(
                    (result_without_benchmark * weights).sum(axis=0)
                    / weights.where(result_without_benchmark.notna()).sum(axis=0),
                    rounding,
                )

                # Append the weighted averages as a new row
                result.loc["Portfolio"] = weighted_averages
            elif isinstance(result.index, pd.PeriodIndex) and not isinstance(
                result.columns, pd.MultiIndex
            ):
                weights = weights.reindex(result.index)

                weighted_averages = apply_rounding(
                    (result_without_benchmark * weights).sum(axis=1)
                    / weights.where(result_without_benchmark.notna()).sum(axis=1),
                    rounding,
                )

                # Append the weighted averages as a new row
                result["Portfolio"] = weighted_averages

            if growth and isinstance(lag, list):
                logger.warning(
                    "Calculating multiple lags for the portfolio data is not currently available. \n"
                    "If desired, please reach out via https://github.com/JerBouma/FinanceToolkit/issues"
                )

        return result

    return wrapper


def split_values(values: Iterable | str | float | None) -> list:
    """
    Splits values given as one comma-separated string (as the MCP server passes a list),
    as a list or as a single number into a list. Text is stripped of surrounding spaces,
    and empty text and None are left out; other values, such as numbers, are kept as they
    are.

    Args:
        values (Iterable | str | float | None): The values, a comma-separated string, a
            single value or None.

    Returns:
        list: The values, empty when none are given.
    """
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    elif not isinstance(values, Iterable):
        values = [values]

    return [
        value.strip() if isinstance(value, str) else value
        for value in values
        if value is not None and (not isinstance(value, str) or value.strip())
    ]


def require_api_key(api_key: str | None, data: str) -> None:
    """
    Warns and raises when data from FinancialModelingPrep is requested without an API key.

    Args:
        api_key (str | None): The FinancialModelingPrep API key.
        data (str): The data that needs the key, completing "required to retrieve ...".

    Raises:
        ValueError: When no key is given.
    """
    if not api_key:
        logger.warning(
            "No FinancialModelingPrep API key found. Retrieving %s requires a key, obtain "
            "one (with 15%% off) at https://www.jeroenbouma.com/fmp and pass it via the "
            "api_key argument.",
            data,
        )
        raise ValueError(
            f"A FinancialModelingPrep API key is required to retrieve {data}. Obtain one at "
            "https://www.jeroenbouma.com/fmp and pass it via the api_key argument."
        )


def require_input(data: pd.Series | pd.DataFrame, name: str) -> None:
    """
    Raises when a metric another one builds on could not be calculated, so that the
    composite reports which input is missing instead of failing on the empty result.

    Args:
        data (pd.Series | pd.DataFrame): The input metric, an empty Series when it failed
            (the reason is logged where it failed).
        name (str): The name of the input, completing "... is not available".

    Raises:
        ValueError: When the input is empty.
    """
    if data.empty:
        raise ValueError(
            f"The {name} is not available, so this metric cannot be calculated from it."
        )


def require_fred_api_key(fred_api_key: str | None, data: str) -> None:
    """
    Warns and raises when data from FRED (Federal Reserve Economic Data) is requested
    without an API key.

    Args:
        fred_api_key (str | None): The FRED API key.
        data (str): The data that needs the key, completing "required to retrieve ...".

    Raises:
        ValueError: When no key is given.
    """
    if not fred_api_key:
        logger.warning(
            "No FRED API key found. %s is sourced from FRED (Federal Reserve Economic Data) "
            "and requires a key to access — registration is entirely free and takes about a "
            "minute at https://fred.stlouisfed.org/docs/api/api_key.html. Once you have one, "
            "pass it via the fred_api_key argument or set the FRED_API_KEY environment "
            "variable.",
            data[0].upper() + data[1:],
        )
        raise ValueError(
            f"A FRED API key is required to retrieve {data}. Obtain a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and pass it via the "
            "fred_api_key argument or set the FRED_API_KEY environment variable."
        )


def fill_diluted_shares(income_statement: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the Weighted Average Shares where an income statement lacks the Weighted Average
    Shares Diluted, so that the per-share figures, the market cap and everything built on
    them can still be calculated.

    Some data providers report only the basic share count (or leave the diluted one out in
    some periods). Without dilution the two are equal, and otherwise the basic count is
    lower by the options and convertibles not yet exercised, so it is the closest figure
    reported. The tickers this applies to are logged once.

    Args:
        income_statement (pd.DataFrame): The income statement, with the tickers and the line
            items as its index and the periods as its columns.

    Returns:
        pd.DataFrame: The income statement with the Weighted Average Shares Diluted filled
        in, or the statement itself when nothing is missing.
    """
    basic_name, diluted_name = (
        "Weighted Average Shares",
        "Weighted Average Shares Diluted",
    )

    if income_statement.empty or basic_name not in income_statement.index.unique(
        level=1
    ):
        return income_statement

    basic = income_statement.xs(basic_name, level=1)
    diluted = (
        income_statement.xs(diluted_name, level=1)
        if diluted_name in income_statement.index.unique(level=1)
        else pd.DataFrame(index=[], columns=income_statement.columns, dtype=float)
    ).reindex(basic.index)

    missing = diluted.isna() & basic.notna()
    tickers = list(missing.index[missing.any(axis=1)])

    if not tickers:
        return income_statement

    logger.warning(
        "The income statement has no Weighted Average Shares Diluted for %s (in all or "
        "some periods), so the Weighted Average Shares are used there.",
        ", ".join(map(str, tickers)),
    )

    filled = diluted.loc[tickers].fillna(basic.loc[tickers]).astype(float)
    filled.index = pd.MultiIndex.from_product([tickers, [diluted_name]])

    statement = pd.concat(
        [income_statement.drop(index=filled.index, errors="ignore"), filled]
    )

    # The rows of each ticker together again, in the order of the statement.
    return statement.loc[income_statement.index.unique(level=0)]


def get_goodwill_and_intangible_assets(balance_sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the goodwill and other intangible assets on a balance sheet, the assets the
    tangible measures (tangible assets, tangible book value, asset coverage) deduct.

    The row that combines them is used where the data provider reports it. Where it is
    missing, the separate Goodwill and Intangible Assets rows are added up, with a missing
    one counting as zero: a company that reports goodwill but no other intangibles (or the
    reverse) still gets a value. The result is NaN only when none of the three is reported.

    Args:
        balance_sheet (pd.DataFrame): The balance sheet statement, with the tickers and the
            line items as its index and the periods as its columns.

    Returns:
        pd.DataFrame: The goodwill and intangible assets, one row per ticker.
    """
    tickers = balance_sheet.index.unique(level=0)
    items = balance_sheet.index.get_level_values(1)

    # A supplied balance sheet can hold the combined row without the separate ones (or the
    # reverse), so a row that isn't there counts as not reported for every ticker.
    def select(item: str) -> pd.DataFrame:
        if item not in items:
            return pd.DataFrame(np.nan, index=tickers, columns=balance_sheet.columns)

        return balance_sheet.loc[:, item, :].reindex(tickers)

    separate = select("Goodwill").add(select("Intangible Assets"), fill_value=0)

    return select("Goodwill and Intangible Assets").fillna(separate)


def calendar_periods(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Puts a frame with the periods as columns on every period from the first to the last.

    Statements only hold the periods a company reports: a company reporting half-yearly
    has no first and third quarter, and a missing filing leaves out a quarter or year.
    A period that is not reported becomes a NaN column, so that rolling windows and
    shifts count calendar periods rather than reported ones. Columns that are not
    periods (or not unique) are left as they are.

    Args:
        frame (pd.DataFrame): The data, with the tickers (and line items) as index and the
            periods as columns.

    Returns:
        pd.DataFrame: The data on every period from the first to the last column.
    """
    columns = frame.columns

    if (
        not isinstance(columns, pd.PeriodIndex)
        or columns.empty
        or not columns.is_unique
        or columns.hasnans
    ):
        return frame

    return frame.reindex(
        columns=pd.period_range(columns.min(), columns.max(), freq=columns.freq)
    )


def trailing_sum(frame: pd.DataFrame, trailing: int | None) -> pd.DataFrame:
    """
    Sums a flow (an income or cash flow statement item) over the trailing calendar periods,
    e.g. the trailing twelve months with `trailing=4` on quarterly data.

    Every window must be complete: a period that is not reported (such as the first and
    third quarter of a company reporting half-yearly) leaves the sums that include it NaN,
    rather than letting them reach back over more time than `trailing` periods.

    Args:
        frame (pd.DataFrame): The data, with the periods as columns.
        trailing (int | None): The number of periods to sum over, or None for no sum.

    Returns:
        pd.DataFrame: The trailing sums on the columns of the data, or the data itself
        without trailing.
    """
    if not trailing:
        return frame

    return (
        calendar_periods(frame)
        .T.rolling(trailing)
        .sum()
        .T.reindex(columns=frame.columns)
    )


def trailing_mean(frame: pd.DataFrame, trailing: int | None) -> pd.DataFrame:
    """
    Averages a balance (a balance sheet item) over the trailing calendar period ends, e.g.
    the opening and closing balance of a period with `trailing=2`.

    As with `trailing_sum`, a period that is not reported leaves the averages that include
    it NaN.

    Args:
        frame (pd.DataFrame): The data, with the periods as columns.
        trailing (int | None): The number of period ends to average, or None for none.

    Returns:
        pd.DataFrame: The trailing averages on the columns of the data, or the data itself
        without trailing.
    """
    if not trailing:
        return frame

    return (
        calendar_periods(frame)
        .T.rolling(trailing)
        .mean()
        .T.reindex(columns=frame.columns)
    )


def check_trailing(cls: FilteredClass) -> FilteredClass:
    """
    Class decorator that checks the trailing window of every method of the Ratios and
    Models classes that has one: a whole number of at least one period, else an
    InvalidArgumentError. A window longer than the periods of the financial statements
    leaves the values NaN, which is logged once per call (see `report_long_trailing`),
    also when a collection calls the methods of its ratios.

    Args:
        cls (type): The class.

    Returns:
        type: The same class.
    """
    for name, member in list(vars(cls).items()):
        if name.startswith("_") or not callable(member):
            continue

        signature = inspect.signature(member)
        if "trailing" in signature.parameters:
            setattr(cls, name, _trailing_checked(member, signature))

    return cls


def _trailing_checked(function: Callable, signature: inspect.Signature) -> Callable:
    """Wraps a method with a trailing window for `check_trailing`."""

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        trailing = signature.bind_partial(self, *args, **kwargs).arguments.get(
            "trailing"
        )
        error_model.validate_window("trailing", trailing, 1)

        depth = self.__dict__.get("_trailing_depth", 0)
        self._trailing_depth = depth + 1
        try:
            result = function(self, *args, **kwargs)
        finally:
            self._trailing_depth = depth

        # The statements are only counted with a window, as reading them raises when they could not be retrieved.  # noqa: E501
        if trailing and not depth:
            report_long_trailing(trailing, len(self._income_statement.columns), result)

        return result

    return wrapper


def report_long_trailing(
    trailing: int | None, periods: int, result: pd.Series | pd.DataFrame
) -> None:
    """
    Logs a trailing window that leaves values NaN for want of periods: one longer than the
    periods of the financial statements, or one as long as them when the result has no
    values at all, as a ratio of averaged balances also needs the period end that opens
    its window.

    Args:
        trailing (int | None): The trailing window, None for none.
        periods (int): The number of periods of the financial statements.
        result (pd.Series | pd.DataFrame): The result calculated with the window.
    """
    if not trailing or not periods or not isinstance(trailing, numbers.Integral):
        return

    if isinstance(result, pd.DataFrame):
        values = result.select_dtypes("number")
    elif isinstance(result, pd.Series) and pd.api.types.is_numeric_dtype(result):
        values = result
    else:
        values = pd.Series(dtype="float64")
    without_values = values.size > 0 and not values.notna().to_numpy().any()

    if trailing > periods or (trailing == periods and without_values):
        logger.warning(
            "The trailing window of %s periods is %s the %s periods of the financial "
            "statements, so the values are NaN where no period has a complete window (a ratio "
            "of averaged balances also needs the period end before the window). Pass a smaller "
            "trailing window or an earlier start_date.",
            trailing,
            "longer than" if trailing > periods else "as long as",
            periods,
        )


def add_basket_total(
    statement: pd.DataFrame, tickers: list[str], not_summed: list[str]
) -> pd.DataFrame:
    """
    Adds the tickers up into one company named "Total", so that a group of companies (a
    basket) can be analysed as if it were one: its financial statements are the sums of
    theirs, and every ratio calculated from them is the ratio of those sums.

    A figure of the Total is missing for a period when a ticker has no value for it, or no
    statement at all, as the sum of the others would understate the basket. Figures that do not add up across
    companies (per-share amounts and share counts) are left empty.

    Notes: All the figures must be in the same currency.

    Args:
        statement (pd.DataFrame): The financial statement, with the tickers and the line
            items as its index and the periods as its columns.
        tickers (list[str]): The tickers to add up.
        not_summed (list[str]): The line items that are left empty for the Total.

    Returns:
        pd.DataFrame: The financial statement with the Total added after the tickers.
    """
    figures = statement.loc[statement.index.get_level_values(0).isin(tickers)]
    total = figures.groupby(level=1, sort=False).sum(min_count=len(tickers))
    total.loc[total.index.isin(not_summed)] = np.nan
    total.index = pd.MultiIndex.from_product(
        [["Total"], total.index], names=statement.index.names
    )

    return pd.concat([statement, total])


def deflate_with_price_index(
    values: pd.DataFrame, price_index: pd.Series
) -> pd.DataFrame:
    """
    Expresses monthly or yearly values in the money of one period, the latest period of a
    price index such as the consumer price index, so that values from decades apart can be
    compared in real terms ("in September 2026 dollars").

    The formula is as follows:

    - Real Value (t) = Nominal Value (t) * Price Index (latest month) / Price Index (t)

    A month after the latest month of the price index, which statistics offices publish a few
    weeks late, keeps its nominal value: it is in the money of that latest month already but
    for the inflation since. A month before the first month of the price index has no real
    value.

    Args:
        values (pd.DataFrame): The nominal values, with periods (a PeriodIndex) as index.
        price_index (pd.Series): The price index, with the same periods as index.

    Returns:
        pd.DataFrame: The values in the money of the latest period of the price index.
    """
    price_index = price_index.dropna()
    monthly = price_index.reindex(values.index)
    # Months after the latest month of the price index take that month's level.
    monthly = monthly.where(values.index <= price_index.index[-1], price_index.iloc[-1])

    return values.mul(price_index.iloc[-1] / monthly, axis=0)


def add_score_zone(
    results: pd.DataFrame, score: str, cut_offs: list[float], zones: list[str]
) -> pd.DataFrame:
    """
    Adds a "Zone" row after each ticker's rows that names the zone its score falls in, such
    as "Distress", "Grey" or "Safe" for the Altman Z-Score, so a score does not have to be
    compared with the cut-offs of its model by hand.

    A score equal to a cut-off falls in the zone above it. A period without a score has no
    zone.

    Args:
        results (pd.DataFrame): The model's results, with (ticker, item) rows and periods as
            columns.
        score (str): The row with the score, e.g. "Altman Z-Score".
        cut_offs (list[float]): The cut-offs between the zones, in increasing order.
        zones (list[str]): The zones from the lowest scores to the highest, one more than
            there are cut-offs.

    Returns:
        pd.DataFrame: The results with a "Zone" row per ticker.
    """
    scores = results.xs(score, level=1)
    zone = scores.apply(
        lambda row: pd.cut(
            row, bins=[-np.inf, *cut_offs, np.inf], labels=zones, right=False
        ).astype(object),
        axis=1,
    ).where(scores.notna())
    zone.index = pd.MultiIndex.from_product([zone.index, ["Zone"]])

    tickers = list(dict.fromkeys(results.index.get_level_values(0)))

    return pd.concat([results.astype(object), zone]).reindex(tickers, level=0)


def calendar_end_date(end_date: str | None) -> str | None:
    """
    Returns the date a calendar of scheduled events (earnings releases, dividends) is cut
    off at. A window that reaches today also keeps the events scheduled after it: the
    upcoming release is what a calendar is read for, and the Toolkit's end date defaults
    to today. A window that ends in the past is cut off at its end date.

    Args:
        end_date (str | None): The end date of the window (YYYY-MM-DD), or None.

    Returns:
        str | None: The end date, or None when the window reaches today and nothing is cut
            off.
    """
    if end_date is None or pd.Timestamp(end_date) >= pd.Timestamp.today().normalize():
        return None

    return end_date


def rank_across_tickers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks the tickers within every period, from the highest value (rank 1) to the lowest,
    and compares each ticker with its neighbour in the ranking toward the middle: the
    ticker ranked just below it in the top half and the one ranked just above it in the
    bottom half. The first and last rank therefore show how far the best and the worst
    stand apart from the rest, e.g. an index that fell more than twice as far as the next
    worst.

    The Difference is the ticker's value minus its neighbour's, in the units of the values
    (a difference of returns in decimals is in points). The Multiple is the ticker's value
    divided by its neighbour's, only when both have the same sign (a multiple of a loss and
    a gain says nothing). A ticker without a value in a period is left out of its ranking,
    and tickers with the same value share the higher rank.

    Args:
        data (pd.DataFrame): The values with the periods as rows and a column per ticker.

    Returns:
        pd.DataFrame: A row per period and ticker, ordered by rank, with the Rank, Value,
            Compared With, Difference and Multiple.
    """
    rankings = {}
    for period, row in data.iterrows():
        values = (
            pd.to_numeric(row, errors="coerce")
            .dropna()
            .sort_values(ascending=False, kind="stable")
        )
        count = len(values)
        if not count:
            continue
        middle = (count - 1) / 2
        neighbours = [
            position + 1 if position <= middle else position - 1
            for position in range(count)
        ]
        compared = values.iloc[neighbours] if count > 1 else pd.Series([np.nan])
        compared_values = compared.to_numpy(dtype=float)
        same_sign = np.sign(values.to_numpy()) == np.sign(compared_values)
        rankings[period] = pd.DataFrame(
            {
                "Rank": values.rank(ascending=False, method="min").astype(int),
                "Value": values,
                "Compared With": list(compared.index) if count > 1 else [None],
                "Difference": values.to_numpy() - compared_values,
                "Multiple": np.where(
                    same_sign & (compared_values != 0),
                    values.to_numpy()
                    / np.where(compared_values == 0, np.nan, compared_values),
                    np.nan,
                ),
            },
            index=pd.Index(values.index, name=data.columns.name or "Ticker"),
        )

    if not rankings:
        return pd.DataFrame(
            columns=["Rank", "Value", "Compared With", "Difference", "Multiple"]
        )

    return pd.concat(
        rankings, names=[data.index.name or "Date", data.columns.name or "Ticker"]
    )


def get_non_trading_days(historical_data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    The dates on which a ticker did not trade while another ticker or the Benchmark did.

    A Toolkit with tickers on different calendars, such as a stock and a cryptocurrency or
    stocks of two exchanges, holds the dates of all of them. On a date only another ticker
    traded, a ticker's last price is carried forward as a flat bar (Open, High, Low and
    Close at the last close, a Volume of 0) without a Return (see
    `historical_model.fill_non_trading_days`). Those dates are recognised by the missing
    Return after the ticker's first price, up to its last Return. A ticker's first price has
    no Return by construction and is a trading day.

    Args:
        historical_data (pd.DataFrame): The historical data of one period with (column,
            ticker) columns, including "Return" and `column`.
        column (str): The price column whose tickers are checked, e.g. "Close".

    Returns:
        pd.DataFrame: True on the dates a ticker did not trade, a column per ticker.
    """
    prices = historical_data[column]
    if "Return" not in historical_data.columns.get_level_values(0):
        return pd.DataFrame(False, index=prices.index, columns=prices.columns)

    has_return = historical_data["Return"].reindex(columns=prices.columns).notna()
    after_first_price = prices.notna().cumsum() > 1
    up_to_last_return = has_return[::-1].cummax()[::-1]

    return ~has_return & after_first_price & up_to_last_return


def delay_embedding(values: np.ndarray, dimension: int, delay: int = 1) -> np.ndarray:
    """
    The delay embedding of a series (Takens, 1981): every row holds `dimension` values of
    the series, `delay` observations apart, X_i = [x_i, x_(i + delay), ...,
    x_(i + (dimension - 1) * delay)], for every i the series is long enough for.

    Args:
        values (np.ndarray): The series.
        dimension (int): The number of values per row, the embedding dimension.
        delay (int, optional): The number of observations between the values of a row.
            Defaults to 1.

    Returns:
        np.ndarray: The embedded vectors, a row per vector, empty when the series is too short.
    """
    span = (dimension - 1) * delay + 1
    if dimension < 1 or delay < 1:
        raise ValueError("The dimension and the delay should be at least 1.")
    if len(values) < span:
        return np.empty((0, dimension))

    return np.lib.stride_tricks.sliding_window_view(values, span)[:, ::delay]


def calculate_per_window(
    data: pd.DataFrame,
    function: Callable[..., Any],
    window: int | None = None,
    rolling: Callable[..., Any] | None = None,
    **kwargs,
) -> pd.Series | pd.DataFrame:
    """
    Applies a measure of a whole series, such as an entropy or a Hurst exponent, to every
    series of `data` over the dates the series has a value: once over its full history, or
    over every trailing window of `window` observations.

    Args:
        data (pd.DataFrame): The series, a column per series.
        function (Callable[..., Any]): The measure, which takes the values of a series as a
            numpy array followed by `kwargs` and returns a number or a Series of numbers.
        window (int | None, optional): The number of observations per window. Defaults to
            None, which measures the full history of every series once.
        rolling (Callable[..., Any] | None, optional): The same measure of every trailing
            window at once, which takes the values of a series, the window and `kwargs` and
            returns one number per window (such as an entropy whose comparisons the windows
            share). Defaults to None, which applies `function` to every window.
        **kwargs: The further keyword arguments of the measure.

    Returns:
        pd.Series | pd.DataFrame: Without a window, the value per series (a row per series
        with a column per output when the measure returns a Series). With a window, the value
        on the last date of every window, a column per series (or (output, series) columns),
        NaN before the first full window and on the dates a series has no value.
    """
    if window is not None and window < 2:  # noqa: PLR2004
        raise ValueError("The window should hold at least 2 observations.")

    results = {}
    for series in data.columns:
        column = data[series].dropna()
        values = column.to_numpy(dtype=float)

        if window is None:
            results[series] = function(values, **kwargs)
            continue

        measures = (
            list(rolling(values, window, **kwargs))
            if rolling is not None
            else [
                function(values[end - window : end], **kwargs)
                for end in range(window, len(values) + 1)
            ]
        )
        dates = column.index[window - 1 :]
        results[series] = (
            pd.DataFrame(measures, index=dates)
            if measures and isinstance(measures[0], pd.Series)
            else pd.Series(measures, index=dates, dtype=float)
        ).reindex(data.index)

    if window is None:
        if any(isinstance(result, pd.Series) for result in results.values()):
            return pd.DataFrame(results).T

        return pd.Series(results, dtype=float)

    if all(isinstance(result, pd.Series) for result in results.values()):
        return pd.DataFrame(results, index=data.index)

    combined = pd.concat(results, axis=1).swaplevel(1, 0, axis=1)
    outputs = list(dict.fromkeys(combined.columns.get_level_values(0)))

    return combined.reindex(
        columns=pd.MultiIndex.from_product([outputs, list(results)])
    )
