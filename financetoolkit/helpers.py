"""Helpers Module"""

__docformat__ = "google"

import contextlib
import contextvars
import inspect
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from financetoolkit.utilities import logger_model
from financetoolkit.utilities.statistics_model import apply_rounding, bounded_ffill

logger = logger_model.get_logger()


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
    kinder to both the data providers' rate limits and the local machine. Exceptions
    raised inside a worker are re-raised here instead of dying silently inside their
    thread, so a genuine defect surfaces as an immediate, traceable failure.

    Args:
        worker (Callable): The function to execute for each tuple of arguments.
        worker_args (Iterable[tuple]): The argument tuples, one per call, e.g. one per
        ticker. Each tuple is unpacked into the worker.
        max_workers (int | None, optional): The maximum number of threads to use.
        Defaults to None, which uses the FINANCETOOLKIT_MAX_WORKERS environment
        variable when set and DEFAULT_MAX_WORKERS otherwise.

    Returns:
        list: The worker results in the same order as worker_args, which is how the
        collection functions hand per-ticker results back without sharing state.
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

        # Collecting in submission order keeps the result aligned with worker_args;
        # the calls themselves still run concurrently.
        return [future.result() for future in futures]


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

    Both rules are the same rule at two frequencies, which keeps the yearly and
    quarterly outputs of the same company consistent with one another: NVIDIA's fiscal
    2024 is labelled 2023 and its four quarters are labelled 2023Q1 through 2023Q4.

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
    end_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(period_end_dates).to_numpy()))

    if quarter:
        fiscal_periods = end_dates.to_period("Q")
        # The calendar quarter holding the middle month of the fiscal quarter.
        calendar_periods = (end_dates.to_period("M") - 1).asfreq("Q")
    else:
        fiscal_periods = end_dates.to_period("Y")
        # A fiscal year ending before June is labelled with the previous calendar
        # year; shifting back five months lands January to May in that year and
        # leaves June to December where they are.
        calendar_periods = (end_dates.to_period("M") - 5).asfreq("Y")

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

    The date is taken from the Period End Date statistic, which FinancialModelingPrep
    statements carry. When it is not available (another source, an older cache or a dataset
    supplied to the Toolkit), the latest date the period can end on given its label is used,
    so that figures are never applied before the period ended:

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
        - Cumulative Return: The cumulative return for the given period.

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
    historical_data["Return"] = (
        bounded_ffill(historical_data[return_column])
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
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
            return PERIODS_BY_FREQUENCY.get(axis.freqstr[0])

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
    ) -> Any:  # ty: ignore[invalid-method-override]
        try:
            return self[key]
        except KeyError:
            return default


def handle_portfolio(func):
    """
    A decorator that processes the result of a function to handle portfolio data.
    This decorator checks if "Portfolio" is in the `self._tickers` attribute and, if so,
    calculates the weighted average of the result DataFrame using `self._portfolio_weights`
    and appends it as a new row or column named "Portfolio".

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
