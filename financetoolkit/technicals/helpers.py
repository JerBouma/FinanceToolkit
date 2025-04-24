"""Technicals Helpers Module"""

__docformat__ = "google"

import inspect

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


def handle_errors(func):
    """
    Decorator to handle specific technical errors that may occur in a function and provide informative messages.

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
            logger.error(
                "There is an index name missing in the provided historical dataset. "
                "%s",
                e,
            )
            return pd.Series(dtype="object")
        except ValueError as e:
            function_name = func.__name__
            logger.error(
                "An error occurred while trying to run the function %s. %s",
                function_name,
                e,
            )
            return pd.Series(dtype="object")
        except AttributeError as e:
            function_name = func.__name__
            logger.error(
                "An error occurred while trying to run the function %s. %s",
                function_name,
                e,
            )
            return pd.Series(dtype="object")
        except ZeroDivisionError as e:
            function_name = func.__name__
            logger.error(
                "An error occurred while trying to run the function %s. %s",
                function_name,
                e,
            )
            return pd.Series(dtype="object")

    # These steps are there to ensure the docstring of the function remains intact
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__module__ = func.__module__

    return wrapper
