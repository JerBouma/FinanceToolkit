"""Helpers Module"""

__docformat__ = "google"

import contextlib
import inspect
from functools import wraps

import numpy as np
import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


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
        calendar_periods = pd.PeriodIndex(
            (end_dates.year - (end_dates.month < 6).astype(int)).astype(str),  # noqa
            freq="Y",
        )

    shifted_mask = calendar_periods != fiscal_periods

    if (
        shifted_mask.any()
        and fiscal_year_adjustments is not None
        and ticker is not None
    ):
        # Reported as plain integers for years so the value stays readable, and as
        # period labels for quarters where a bare integer would lose the quarter.
        adjustments = [
            {
                "fiscal_year": str(fiscal) if quarter else fiscal.year,
                "calendar_year": str(calendar) if quarter else calendar.year,
            }
            for fiscal, calendar in zip(
                fiscal_periods[shifted_mask], calendar_periods[shifted_mask]
            )
        ]

        # Called once per statement type (income, balance, cash, statistics), so the
        # entries are merged rather than assigned -- assigning would let the last
        # statement fetched silently discard the periods the earlier ones relabelled.
        existing = fiscal_year_adjustments.setdefault(ticker, [])
        existing.extend(
            adjustment for adjustment in adjustments if adjustment not in existing
        )

    return calendar_periods


def enrich_historical_data(
    historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    return_column: str = "Adj Close",
):
    """
    Retrieves enriched historical stock data for the given ticker(s) from Yahoo! Finance API for
    a specified period. It calculates the following:

        - Return: The return for the given period.
        - Cumulative Return: The cumulative return for the given period.

    The return is calculated as the percentage change in the given return column.

    The cumulative return is calculated as the cumulative product of the percentage change in the given
    return column.

    Note that Volatility, Excess Return and Excess Volatility are intentionally not calculated here
    anymore. These are available as dedicated, reusable calculations in the Risk module
    (e.g. risk_model.get_volatility, risk_model.get_excess_volatility) and the Performance module
    (e.g. performance_model.get_excess_return) instead.

    Args:
        historical_data (pd.DataFrame): A pandas DataFrame object containing the historical stock data
        for the given ticker(s).
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        return_column (str, optional): A string representing the column to use for return calculations.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the enriched historical stock data for the given ticker(s).
    """

    historical_data["Return"] = historical_data[return_column].ffill().pct_change()

    historical_data["Cumulative Return"] = 1

    adjusted_return = historical_data.loc[start:end, "Return"]

    with contextlib.suppress(IndexError):
        adjusted_return.iloc[0] = 0

    historical_data["Cumulative Return"] = pd.Series(np.nan).astype(float)

    historical_data.loc[start:end, "Cumulative Return"] = (
        1.0 + adjusted_return.fillna(0)
    ).cumprod()

    return historical_data


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

        # Check if "Portfolio" is in self._tickers
        if (
            isinstance(self._tickers, list | str)
            and "Portfolio" in self._tickers
            and isinstance(result, pd.DataFrame)
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

                weighted_averages = round(
                    (result_without_benchmark * weights).sum(axis=0)
                    / weights.sum(axis=0),
                    rounding,
                )

                # Append the weighted averages as a new row
                result.loc["Portfolio"] = weighted_averages
            elif isinstance(result.index, pd.PeriodIndex) and not isinstance(
                result.columns, pd.MultiIndex
            ):
                weights = weights.reindex(result.index)

                weighted_averages = round(
                    (result_without_benchmark * weights).sum(axis=1)
                    / weights.sum(axis=1),
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
