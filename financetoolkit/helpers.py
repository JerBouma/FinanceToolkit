"""Helpers Module"""

__docformat__ = "google"

import inspect
import os
import pickle
import time
import warnings
from functools import wraps
from io import StringIO

import numpy as np
import pandas as pd
import requests
from urllib3.exceptions import MaxRetryError

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
            if (
                "not available under your current subscription"
                in response.json()["Error Message"]
            ):
                return pd.DataFrame(columns=["NOT AVAILABLE"])

            if "Bandwidth Limit Reach" in response.json()["Error Message"]:
                return pd.DataFrame(columns=["BANDWIDTH LIMIT REACH"])
            if "Limit Reach" in response.json()["Error Message"]:
                if (
                    sleep_timer
                    and limit_retry_counter < RETRY_LIMIT
                    and user_subscription != "Free"
                ):
                    time.sleep(5.01)
                    limit_retry_counter += 1
                else:
                    return pd.DataFrame(columns=["NO DATA"])
            if (
                "Free plan is limited to US stocks only"
                in response.json()["Error Message"]
            ):
                return pd.DataFrame(columns=["US STOCKS ONLY"])

            if "Invalid API KEY." in response.json()["Error Message"]:
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


def determine_currencies(
    statement_currencies: pd.DataFrame, historical_currencies: pd.DataFrame
):
    """
    Based on the statement currencies and the historical currencies, determine the
    currencies that are used in the financial statements and the historical datasets.

    This is relevant to prevent mismatches between the perceived price of the instrument
    and the numbers as found in the financial statements. If there is a mismatch, then
    the currency conversion needs to be applied.

    Args:
        statement_currencies (pd.DataFrame): A DataFrame containing the statement currencies.
        historical_currencies (pd.DataFrame): A DataFrame containing the historical currencies.

    Returns:
        pd.Series, list: a Series containing the currency symbols per ticker
        and a list containing the currencies.
    """
    currencies = []

    for period in statement_currencies.columns:
        statement_currencies.loc[:, period] = (
            statement_currencies[period] + historical_currencies + "=X"
        )

        for currency in statement_currencies[period].unique():
            # Identify the currencies that are not in the list yet
            # and that are not NaN (the currency == currency check)
            if currency not in currencies and currency == currency:  # noqa
                currencies.append(currency)

    statement_currencies = statement_currencies.bfill(axis=1).ffill(axis=1)

    statement_currencies = statement_currencies[statement_currencies.columns[-1]]

    return statement_currencies, currencies


def convert_currencies(
    financial_statement_data: pd.DataFrame,
    financial_statement_currencies: pd.Series,
    exchange_rate_data: pd.DataFrame,
    items_not_to_adjust: list[str] | None = None,
    financial_statement_name: str | None = None,
):
    """
    Based on the retrieved currency definitions (e.g. EURUSD=X) for each ticker, obtained
    through using the determine_currencies function, convert the financial statement data
    to the historical currency.

    The function reports the tickers that are converted and the currencies that they are
    converted from and to. If the currency is the same, then no conversion is applied.

    The function will also report the tickers that could not be converted. This is usually
    due to the fact that the currency is not available in the historical data.

    Args:
        financial_statement_data (pd.DataFrame): A DataFrame containing the financial statement data.
        financial_statement_currencies (pd.Series): A Series containing the currency symbols per ticker.
        exchange_rate_data (pd.DataFrame): A DataFrame containing the exchange rate data.
        items_not_to_adjust (list[str]): A list containing the items that should not be adjusted. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the converted financial statement data.
    """
    no_data = []

    periods = financial_statement_data.columns
    tickers = financial_statement_data.index.get_level_values(0).unique()
    currencies: dict[str, list[str]] = {}

    for ticker in tickers:
        try:
            currency = financial_statement_currencies.loc[ticker]

            # Only proceed if the currency is not NaN
            if currency == currency:  # noqa
                base_currency, quote_currency = currency[:3], currency[3:6]

                if base_currency != quote_currency:
                    if currency not in currencies:
                        currencies[currency] = []

                    if items_not_to_adjust is not None:
                        items_to_adjust = [
                            item
                            for item in financial_statement_data.index.get_level_values(
                                level=1
                            )
                            if item not in items_not_to_adjust
                        ]
                    else:
                        items_to_adjust = (
                            financial_statement_data.index.get_level_values(level=1)
                        )

                    financial_statement_data.loc[(ticker, items_to_adjust), :] = (
                        financial_statement_data.loc[(ticker, items_to_adjust), :].mul(
                            exchange_rate_data.loc[periods, currency], axis=1
                        )
                    ).to_numpy()

                    currencies[currency].append(ticker)
            else:
                no_data.append(ticker)
        except (KeyError, ValueError):
            no_data.append(ticker)
            continue

    if no_data:
        print(
            "The following tickers could not be converted to the historical data currency: "
            f"{', '.join(no_data)}"
        )

    currencies_text = []
    for currency, ticker_match in currencies.items():
        base_currency, quote_currency = currency[:3], currency[3:6]

        if base_currency != quote_currency:
            for ticker in ticker_match:
                currencies_text.append(
                    f"{ticker} ({base_currency} to {quote_currency})"
                )

    if currencies_text:
        print(
            f"The {financial_statement_name if financial_statement_name else 'financial statement'} "
            f"from the following tickers are converted: {', '.join(currencies_text)}"
        )

    return financial_statement_data


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


def handle_errors(func):
    """
    Decorator to handle specific errors that may occur in a function and provide informative messages.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    Raises:
        KeyError: If an index name is missing in the provided financial statements.
        ValueError: If an error occurs while running the function, typically due to incomplete financial statements.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            function_name = func.__name__
            print(
                "There is an index name missing in the provided financial statements. "
                f"This is {e}. This is required for the function ({function_name}) "
                "to run. Please fill this column to be able to calculate the ratios."
            )
            return pd.Series(dtype="object")
        except ValueError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e}"
            )
            return pd.Series(dtype="object")
        except AttributeError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e}"
            )
            return pd.Series(dtype="object")
        except ZeroDivisionError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e} This is due to a division by zero."
            )
            return pd.Series(dtype="object")
        except IndexError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e} This is due to missing data."
            )
            return pd.Series(dtype="object")

    # These steps are there to ensure the docstring of the function remains intact
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__module__ = func.__module__

    return wrapper


def check_for_error_messages(
    dataset_dictionary: dict[str, pd.DataFrame],
    user_subscription: str,
    required_subscription: str = "Premium",
    delete_tickers: bool = True,
):
    """
    This functionality checks whether any of the defined errors are found in the
    dataset and if they are, report them accordingly. This function is written
    to prevent spamming the command line with error messages.

    Args:
        dataset_dictionary (dict[str, pd.DataFrame]): a dictionary with the ticker
        as key and the dataframe as value.
        user_subscription (str): the subscription type of the user.
        subscription_type (str): the subscription type of the user. Defaults to "Premium".
        delete_tickers (bool): whether to delete the tickers that have an error from the
        dataset dictionary. Defaults to True.
    """

    not_available = []
    bandwidth_limit_reach = []
    no_data = []
    us_stocks_only = []
    invalid_api_key = []
    no_errors = []

    for ticker, dataframe in dataset_dictionary.items():
        if "NOT AVAILABLE" in dataframe.columns:
            not_available.append(ticker)
        if "BANDWIDTH LIMIT REACH" in dataframe.columns:
            bandwidth_limit_reach.append(ticker)
        if "NO DATA" in dataframe.columns:
            no_data.append(ticker)
        if "US STOCKS ONLY" in dataframe.columns:
            us_stocks_only.append(ticker)
        if "INVALID API KEY" in dataframe.columns:
            invalid_api_key.append(ticker)
        if "NO ERRORS" in dataframe.columns:
            no_errors.append(ticker)

    if not_available:
        print(
            f"The requested data is part of the {required_subscription} Subscription from "
            f"FinancialModelingPrep: {', '.join(not_available)}.\nIf you wish to access "
            "this data, consider upgrading your plan. You can get 15% off by using the "
            "following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp"
        )

    if bandwidth_limit_reach:
        print(
            f"The bandwidth limit has been reached for the following tickers: {', '.join(bandwidth_limit_reach)}.\n"
            "Consider upgrading your plan to a higher plan to increase your bandwidth limit. You can get 15% "
            "off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp"
        )

    if no_data:
        print(
            "Some tickers have no data, verify if the ticker has any data to begin with. "
            "If it does, please open an issue here: https://github.com/JerBouma/FinanceToolkit/issues. "
            f"These tickers are: {', '.join(no_data)}"
        )

        if user_subscription == "Free":
            print(
                "Given that you are using the Free plan, it could be due to reaching the API "
                "limit of the day, consider upgrading your plan. You can get 15% off by "
                "using the following affiliate link which also supports the project: "
                "https://www.jeroenbouma.com/fmp"
            )

    if us_stocks_only:
        print(
            "The Free plan is limited to US stocks only. Therefore the following tickers are not "
            f"available: {', '.join(us_stocks_only)}\nConsider upgrading your plan to Starter or "
            "higher. You can get 15% off by using the following affiliate link which also "
            "supports the project: https://www.jeroenbouma.com/fmp"
        )

    if invalid_api_key:
        print(
            "You have entered an invalid API key from FinancialModelingPrep. Obtain your API key for free "
            "and get 15% off the Premium plans by using the following affiliate link.\nThis also supports "
            "the project: https://www.jeroenbouma.com/fmp"
        )

    if delete_tickers:
        removed_tickers = set(
            not_available + no_data + us_stocks_only + invalid_api_key + no_errors
        )

        for ticker in removed_tickers:
            del dataset_dictionary[ticker]

    return dataset_dictionary


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
                print(
                    "Calculating multiple lags for the portfolio data is not currently available. \n"
                    "If desired, please reach out via https://github.com/JerBouma/FinanceToolkit/issues"
                )

        return result

    return wrapper


def load_cached_data(cached_data_location: str, file_name: str, method: str = "pandas"):
    """
    Load the cached data from the specified location and file name.

    Args:
        cached_data_location (str): The location of the cached data.
        file_name (str): The name of the file to load.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        if method == "pandas":
            cached_data = pd.read_pickle(f"{cached_data_location}/{file_name}")
        elif method == "pickle":
            with open(f"{cached_data_location}/{file_name}", "rb") as file:
                cached_data = pickle.load(file)
        else:
            raise ValueError("The method should be either 'pandas' or 'pickle'.")

        return cached_data

    except FileNotFoundError:
        return pd.DataFrame()


def save_cached_data(
    cached_data: pd.DataFrame,
    cached_data_location: str,
    file_name: str,
    method: str = "pandas",
    include_message: bool = True,
):
    """
    Save the cached data to the specified location and file name.

    Args:
        cached_data_location (str): The location to save the cached data.
        file_name (str): The name of the file to save.
    """
    os.makedirs(cached_data_location, exist_ok=True)

    if file_name in os.listdir(cached_data_location):
        # When the file already exists do nothing.
        pass
    else:
        try:
            if method == "pandas":
                cached_data.to_pickle(f"{cached_data_location}/{file_name}")
            elif method == "pickle":
                with open(f"{cached_data_location}/{file_name}", "wb") as file:
                    pickle.dump(cached_data, file, protocol=pickle.HIGHEST_PROTOCOL)

            if include_message:
                print(f"The data has been saved to {cached_data_location}/{file_name}")
        except Exception as error:  # pylint: disable=broad-except
            print(f"An error occurred while saving the data: {error}")
