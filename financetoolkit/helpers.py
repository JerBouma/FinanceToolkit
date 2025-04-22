"""Helpers Module"""

__docformat__ = "google"

import inspect
import time
import warnings
from functools import wraps
from io import StringIO

import numpy as np
import pandas as pd
import requests
from urllib3.exceptions import MaxRetryError

from financetoolkit import logger_model

logger = logger_model.get_logger()

RETRY_LIMIT = 12

# pylint: disable=comparison-with-itself,too-many-locals,protected-access


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
        ticker (str): The company ticker (for example: "AAPL")
        url (str): The url to retrieve the data from.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
        if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to False.
        raw (bool): Whether to return the raw JSON data. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the financial data.
    """
    error_retry_counter = 0
    limit_retry_counter = 0

    while True:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            if raw:
                return response.json()

            json_io = StringIO(response.text)

            financial_data = pd.read_json(json_io)

            return financial_data

        except (requests.exceptions.HTTPError, ValueError):
            error_message = response.text

            if "Exclusive Endpoint" in error_message:
                return pd.DataFrame(columns=["EXCLUSIVE ENDPOINT"])
            if "Special Endpoint" in error_message:
                return pd.DataFrame(columns=["SPECIAL ENDPOINT"])
            if "Bandwidth Limit Reach" in error_message:
                return pd.DataFrame(columns=["BANDWIDTH LIMIT REACH"])
            if "Limit Reach" in error_message:
                if (
                    sleep_timer
                    and limit_retry_counter < RETRY_LIMIT
                    and user_subscription != "Free"
                ):
                    time.sleep(5.01)
                    limit_retry_counter += 1
                else:
                    return pd.DataFrame(columns=["LIMIT REACH"])
            if "US stocks only" in error_message:
                return pd.DataFrame(columns=["US STOCKS ONLY"])

            if "Invalid API KEY." in error_message:
                return pd.DataFrame(columns=["INVALID API KEY"])

        except (
            MaxRetryError,
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
        ):
            # When the connection is refused, retry the request 12 times
            # and if it doesn't work, then return an empty dataframe
            if error_retry_counter == RETRY_LIMIT:
                return pd.DataFrame(columns=["NO ERRORS"])

            error_retry_counter += 1
            time.sleep(5)


def calculate_growth(
    dataset: pd.Series | pd.DataFrame,
    lag: int | list[int] = 1,
    rounding: int | None = 4,
    axis: str = "columns",
) -> pd.Series | pd.DataFrame:
    """
    Calculates growth for a given dataset. Defaults to a lag of 1 (i.e. 1 year or 1 quarter).

    Args:
        dataset (pd.Series | pd.DataFrame): the dataset to calculate the growth values for.
        lag (int | str): the lag to use for the calculation. Defaults to 1.

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    # With Pandas 2.1, pct_change will no longer automatically forward fill
    # given that this has been solved within the code already but the warning
    # still appears, this is a temporary fix to ignore the warning
    warnings.simplefilter(action="ignore", category=FutureWarning)

    if isinstance(lag, list):
        new_index = []
        lag_dict = {f"Lag {lag_value}": lag_value for lag_value in lag}

        if axis == "columns":
            for old_index in dataset.index:
                for lag_value in lag_dict:
                    new_index.append(
                        (*old_index, lag_value)
                        if isinstance(old_index, tuple)
                        else (old_index, lag_value)
                    )

            dataset_lag = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(new_index),
                columns=dataset.columns,
                dtype=np.float64,
            )

            for new_index in dataset_lag.index:
                lag_key = new_index[-1]
                other_indices = new_index[:-1]

                dataset_lag.loc[new_index] = (
                    dataset.loc[other_indices]
                    .ffill()
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
                )
        else:
            for old_index in dataset.columns:
                for lag_value in lag_dict:
                    new_index.append(
                        (*old_index, lag_value)
                        if isinstance(old_index, tuple)
                        else (old_index, lag_value)
                    )

            dataset_lag = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(new_index),
                index=dataset.index,
                dtype=np.float64,
            )

            for new_index in dataset_lag.columns:
                lag_key = new_index[-1]
                other_indices = new_index[:-1]

                dataset_lag.loc[:, new_index] = (
                    dataset.loc[:, other_indices]
                    .ffill()
                    .pct_change(periods=lag_dict[lag_key])  # type: ignore
                    .to_numpy()
                )

        return dataset_lag.round(rounding)

    return dataset.ffill().pct_change(periods=lag, axis=axis).round(rounding)


def combine_dataframes(dataset_dictionary: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine the dataframes from different companies of the same financial statement,
    e.g. the balance sheet statement, into a single dataframe.

    Args:
        dataset_dictionary (dict[str, pd.DataFrame]): A dictionary containing the
        dataframes for each company. It should have the structure key: ticker,
        value: dataframe.

    Returns:
        pd.DataFrame: A pandas DataFrame with the combined financial statements.
    """
    combined_df = pd.concat(dict(dataset_dictionary), axis=0)

    return combined_df.sort_index(level=0, sort_remaining=False)


def equal_length(dataset1: pd.Series, dataset2: pd.Series) -> pd.Series:
    """
    Equalize the length of two datasets by adding zeros to the beginning of the shorter dataset.

    Args:
        dataset1 (pd.Series): The first dataset to be equalized.
        dataset2 (pd.Series): The second dataset to be equalized.

    Returns:
        pd.Series, pd.Series: The equalized datasets.
    """
    if int(dataset1.columns[0]) > int(dataset2.columns[0]):
        for value in range(
            int(dataset1.columns[0]) - 1, int(dataset2.columns[0]) - 1, -1
        ):
            dataset1.insert(0, value, 0.0)
        dataset1 = dataset1.sort_index()
    elif int(dataset1.columns[0]) < int(dataset2.columns[0]):
        for value in range(
            int(dataset2.columns[0]) - 1, int(dataset1.columns[0]) - 1, -1
        ):
            dataset2.insert(0, value, 0.0)
        dataset2 = dataset2.sort_index()

    return dataset1, dataset2


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
            isinstance(self._tickers, (list, str))
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
                weights = weights.loc[result_without_benchmark.columns, :].T

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
                weights = weights.loc[result.index, :]

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
