"""Performance Helpers Module"""
__docformat__ = "google"

import inspect

import pandas as pd

# pylint: disable=protected-access


def handle_return_data_periods(self, period: str, within_period: bool):
    """
    This function is a specific function solely related to the Ratios controller. It
    therefore also requires a self instance to exists with specific parameters.

    Args:
        period (str): the period to return the data for.
        within_period (bool): whether to return the data within the period or the
        entire period.

    Raises:
        ValueError: if the period is not daily, monthly, weekly, quarterly, or yearly.

    Returns:
        pd.Series: the returns for the period.
    """
    if period == "daily":
        return self._daily_historical_data
    if period == "weekly":
        return (
            self._weekly_within_historical_data
            if within_period
            else self._weekly_historical_data
        )
    if period == "monthly":
        return (
            self._monthly_within_historical_data
            if within_period
            else self._monthly_historical_data
        )
    if period == "quarterly":
        return (
            self._quarterly_within_historical_data
            if within_period
            else self._quarterly_historical_data
        )
    if period == "yearly":
        return (
            self._yearly_within_historical_data
            if within_period
            else self._yearly_historical_data
        )

    raise ValueError("Period must be weekly, monthly, quarterly, or yearly.")


def handle_risk_free_data_periods(self, period: str):
    """
    This function is a specific function solely related to the Ratios controller. It
    therefore also requires a self instance to exists with specific parameters.

    Args:
        period (str): the period to return the data for.
        within_period (bool): whether to return the data within the period or the
        entire period.

    Raises:
        ValueError: if the period is not daily, monthly, weekly, quarterly, or yearly.

    Returns:
        pd.Series: the returns for the period.
    """
    if period == "daily":
        return self._daily_risk_free_rate_data
    if period == "weekly":
        return self._weekly_risk_free_rate_data
    if period == "monthly":
        return self._monthly_risk_free_rate_data
    if period == "quarterly":
        return self._quarterly_risk_free_rate_data
    if period == "yearly":
        return self._yearly_risk_free_rate_data

    raise ValueError("Period must be daily, monthly, weekly, quarterly, or yearly.")


def handle_errors(func):
    """
    Decorator to handle specific performance errors that may occur in a function and provide informative messages.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    Raises:
        KeyError: If an index name is missing in the provided historical data.
        ValueError: If an error occurs while running the function, typically due to incomplete historical data.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            function_name = func.__name__
            if "Benchmark" in str(e):
                print(
                    f"Please set a benchmark_ticker in the Toolkit class to calculate {function_name}."
                    "\nFor example: toolkit = Toolkit(['TSLA', 'AAPL', 'MSFT'], benchmark_ticker='^GSPC')"
                )
            else:
                print(
                    "There is an index name missing in the provided historical dataset. "
                    f"This is {e}. This is required for the function ({function_name}) "
                    "to run. Please fill this column to be able to calculate the ratios."
                )
            return pd.Series(dtype="object")
        except ValueError as e:
            function_name = func.__name__
            if "Benchmark" in str(e):
                print(
                    f"Please set a benchmark_ticker in the Toolkit class to calculate {function_name}."
                    "\nFor example: toolkit = Toolkit(['TSLA', 'AAPL', 'MSFT'], benchmark_ticker='^GSPC')"
                )
            else:
                print(
                    f"An error occurred while trying to run the function "
                    f"{function_name}. This is {e}. Usually this is due to incomplete "
                    "historical data. "
                )
            return pd.Series(dtype="object")
        except AttributeError as e:
            function_name = func.__name__
            if "Benchmark" in str(e):
                print(
                    f"Please set a benchmark_ticker in the Toolkit class to calculate {function_name}."
                    "\nFor example: toolkit = Toolkit(['TSLA', 'AAPL', 'MSFT'], benchmark_ticker='^GSPC')"
                )
            else:
                print(
                    f"An error occurred while trying to run the function "
                    f"{function_name}. This is {e}. Usually this is due to incomplete "
                    "historical data. "
                )
            return pd.Series(dtype="object")
        except ZeroDivisionError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. This is {e}. This is due to a division by zero."
            )
            return pd.Series(dtype="object")

    # These steps are there to ensure the docstring of the function remains intact
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__module__ = func.__module__

    return wrapper
