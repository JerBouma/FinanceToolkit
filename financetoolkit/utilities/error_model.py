"""Error Module"""

__docformat__ = "google"

import inspect
import os

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


# pylint: disable=comparison-with-itself,too-many-locals,protected-access

# Set FINANCETOOLKIT_STRICT_ERRORS to 1 (or true/yes/on) to make every failure inside a
# metric raise instead of being reported and returned as an empty Series. This is what
# a test suite or a scheduled job should run with, since it turns a quietly missing
# number into an immediate, traceable failure.
STRICT_ERRORS_ENVIRONMENT_VARIABLE = "FINANCETOOLKIT_STRICT_ERRORS"

# AttributeError and TypeError cannot be produced by financial data that is merely
# incomplete; they mean the code asked an object for something it does not have. There
# is no value that can be returned for them that is not a lie, so they always raise.
ALWAYS_RAISED_ERRORS = (AttributeError, TypeError)


def use_strict_errors() -> bool:
    """
    Reports whether strict error handling is enabled, in which case every failure inside
    a metric is raised rather than reported and replaced by an empty Series.

    Returns:
        bool: whether strict error handling is enabled.
    """
    return os.environ.get(STRICT_ERRORS_ENVIRONMENT_VARIABLE, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_tickers_from_arguments(args: tuple) -> str:
    """
    Recovers the tickers a failing metric was calculating for, so that the error message
    names the companies involved rather than only the metric.

    Args:
        args (tuple): the positional arguments the decorated function was called with.
            The first is the controller instance for every method this decorator is
            applied to.

    Returns:
        str: a comma separated list of tickers, or "unknown" when they cannot be
        recovered from the arguments.
    """
    tickers = getattr(args[0], "_tickers", None) if args else None

    if isinstance(tickers, str):
        return tickers
    if isinstance(tickers, list) and tickers:
        return ", ".join(str(ticker) for ticker in tickers)

    return "unknown"


def handle_errors(func):
    """
    Decorator that reports failures inside a metric calculation instead of letting them
    propagate as a raw traceback, so that one unavailable line item does not abort an
    entire analysis.

    Silently returning an empty Series where a number was expected is the worst outcome
    a financial library can produce, so the behaviour is deliberately split by what the
    exception actually says about the data:

        - KeyError and IndexError mean a line item the calculation needs is not in the
          statements, which is a genuine and common gap between data providers rather
          than a defect. These are logged as an error naming the metric, the tickers and
          the missing item, and an empty Series is returned.
        - ValueError and ZeroDivisionError mean the data is present but could not be
          computed with. These are logged as an error, with the traceback attached so
          the failing line is identifiable, and an empty Series is returned.
        - AttributeError and TypeError cannot be caused by incomplete financial data at
          all; they mean the code is wrong. These are always raised.

    Setting the FINANCETOOLKIT_STRICT_ERRORS environment variable to 1 raises every
    exception instead, which is the appropriate setting for a test suite or a scheduled
    job where an empty result must not pass unnoticed.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function, which returns an empty Series of dtype object
        in place of a result whenever a reported failure occurs.

    Raises:
        AttributeError: If the calculation asks an object for an attribute it lacks.
        TypeError: If the calculation is performed on an unsupported type.
        Exception: Any exception at all when strict error handling is enabled.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ALWAYS_RAISED_ERRORS as error:
            logger.error(
                "%s failed for %s with a %s (%s), which indicates a defect rather than "
                "missing data.",
                func.__name__,
                get_tickers_from_arguments(args),
                type(error).__name__,
                error,
            )
            raise
        except KeyError as error:
            if use_strict_errors():
                raise
            logger.error(
                "%s could not be calculated for %s because the item %s is missing from "
                "the provided financial statements. Fill this row to obtain the metric.",
                func.__name__,
                get_tickers_from_arguments(args),
                error,
            )
            return pd.Series(dtype="object")
        except IndexError as error:
            if use_strict_errors():
                raise
            logger.error(
                "%s could not be calculated for %s due to missing data. %s: %s",
                func.__name__,
                get_tickers_from_arguments(args),
                type(error).__name__,
                error,
            )
            return pd.Series(dtype="object")
        except ZeroDivisionError as error:
            if use_strict_errors():
                raise
            logger.error(
                "%s could not be calculated for %s due to a division by zero. %s: %s",
                func.__name__,
                get_tickers_from_arguments(args),
                type(error).__name__,
                error,
            )
            return pd.Series(dtype="object")
        except ValueError as error:
            if use_strict_errors():
                raise
            logger.error(
                "%s could not be calculated for %s. %s: %s",
                func.__name__,
                get_tickers_from_arguments(args),
                type(error).__name__,
                error,
                exc_info=True,
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
        required_subscription (str): the subscription the requested data needs. Defaults to "Premium".
        delete_tickers (bool): whether to delete the tickers that have an error from the
        dataset dictionary. Defaults to True.
    """

    not_available = []
    premium_query_parameter = []
    exclusive_endpoint = []
    special_endpoint = []
    bandwidth_limit_reach = []
    limit_reach = []
    yfinance_rate_limit_reached = []
    yfinance_rate_limit_reached_fallback = []
    yfinance_rate_limit_or_no_data_found = []
    yfinance_rate_limit_or_no_data_found_fallback = []
    no_data = []
    us_stocks_only = []
    invalid_api_key = []
    no_errors = []
    request_failed = []

    for ticker, dataframe in dataset_dictionary.items():
        if "PREMIUM QUERY PARAMETER" in dataframe.columns:
            premium_query_parameter.append(ticker)
        if "EXCLUSIVE ENDPOINT" in dataframe.columns:
            exclusive_endpoint.append(ticker)
        elif "SPECIAL ENDPOINT" in dataframe.columns:
            special_endpoint.append(ticker)
        elif "NOT AVAILABLE" in dataframe.columns:
            not_available.append(ticker)
        elif "BANDWIDTH LIMIT REACH" in dataframe.columns:
            bandwidth_limit_reach.append(ticker)
        elif "LIMIT REACH" in dataframe.columns:
            limit_reach.append(ticker)
        elif "YFINANCE RATE LIMIT OR NO DATA FOUND FALLBACK" in dataframe.columns:
            yfinance_rate_limit_or_no_data_found_fallback.append(ticker)
        elif "YFINANCE RATE LIMIT OR NO DATA FOUND" in dataframe.columns:
            yfinance_rate_limit_or_no_data_found.append(ticker)
        elif "YFINANCE RATE LIMIT REACHED FALLBACK" in dataframe.columns:
            yfinance_rate_limit_reached_fallback.append(ticker)
        elif "YFINANCE RATE LIMIT REACHED" in dataframe.columns:
            yfinance_rate_limit_reached.append(ticker)
        elif "NO DATA" in dataframe.columns:
            no_data.append(ticker)
        elif "US STOCKS ONLY" in dataframe.columns:
            us_stocks_only.append(ticker)
        elif "INVALID API KEY" in dataframe.columns:
            invalid_api_key.append(ticker)
        elif "REQUEST FAILED" in dataframe.columns:
            request_failed.append(ticker)
        elif "NO ERRORS" in dataframe.columns:
            no_errors.append(ticker)

    if premium_query_parameter:
        logger.error(
            "The following tickers are using a premium query parameter from Financial Modeling Prep: %s.\n"
            "This is not available in your current plan. Consider upgrading your plan to a higher plan. "
            "You can get 15%% off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            ", ".join(premium_query_parameter),
        )

    if exclusive_endpoint:
        logger.error(
            "The following tickers are using an exclusive endpoint from Financial Modeling Prep: %s.\n"
            "This is not available in the Free plan. Consider upgrading your plan to a higher plan. "
            "You can get 15%% off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            ", ".join(exclusive_endpoint),
        )
    if special_endpoint:
        logger.error(
            "The following tickers are using a special endpoint from Financial Modeling Prep: %s.\n"
            "This is not available in the Free plan. Consider upgrading your plan to a higher plan. "
            "You can get 15%% off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            ", ".join(special_endpoint),
        )
    if not_available:
        logger.error(
            "The requested data is part of the %s Subscription from "
            "Financial Modeling Prep: %s.\nIf you wish to access "
            "this data, consider upgrading your plan. You can get 15%% off by using the "
            "following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            required_subscription,
            ", ".join(not_available),
        )

    if bandwidth_limit_reach:
        logger.error(
            "The bandwidth limit from Financial Modeling Prep has been reached for the following tickers: %s.\n"
            "Consider upgrading your plan to a higher plan to increase your bandwidth limit. You can get 15%% "
            "off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            ", ".join(bandwidth_limit_reach),
        )

    if limit_reach:
        logger.error(
            "The limit from Financial Modeling Prep has been reached for the following tickers: %s.\n"
            "Consider upgrading your plan to a higher plan to increase your limit. You can get 15%% "
            "off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            ", ".join(limit_reach),
        )

    if yfinance_rate_limit_or_no_data_found_fallback:
        ticker_text = (
            "tickers"
            if len(yfinance_rate_limit_or_no_data_found_fallback) > 1
            else "ticker"
        )
        logger.error(
            "The rate limit from Yahoo Finance has been reached or no data "
            "could be found from this source for the following %s: %s.\n"
            "This occurred after a previous attempt to use FinancialModelingPrep was unsuccessful "
            "and is likely due to no data being available for the %s.",
            ticker_text,
            ", ".join(yfinance_rate_limit_or_no_data_found_fallback),
            ticker_text,
        )
    if yfinance_rate_limit_or_no_data_found:
        logger.error(
            "The rate limit from Yahoo Finance has been reached or no data could be found "
            "from this source for the following tickers: %s.\n"
            "Consider obtaining an API key from FinancialModelingPrep to potentially "
            "avoid this issue. You can get 15%% "
            "off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            ", ".join(yfinance_rate_limit_or_no_data_found),
        )
    if yfinance_rate_limit_reached_fallback:
        logger.error(
            "The rate limit from Yahoo Finance has been reached for the following tickers: %s.\n"
            "This occurred after a previous attempt to use FinancialModelingPrep was unsuccessful.",
            ", ".join(yfinance_rate_limit_reached_fallback),
        )
    if yfinance_rate_limit_reached:
        logger.error(
            "The rate limit from Yahoo Finance has been reached for the following tickers: %s.\n"
            "Consider obtaining an API key from FinancialModelingPrep to potentially avoid this issue. "
            "You can get 15%% off by using the following affiliate link which also supports the project: "
            "https://www.jeroenbouma.com/fmp",
            ", ".join(yfinance_rate_limit_reached),
        )
    if no_data:
        logger.error(
            "Some tickers from Financial Modeling Prep have no data, verify if the ticker has any data to "
            "begin with. If it does, please open an issue here: https://github.com/JerBouma/FinanceToolkit/issues. "
            "These tickers are: %s",
            ", ".join(no_data),
        )

        if user_subscription == "Free":
            logger.error(
                "Given that you are using the Free plan, it could be due to reaching the API "
                "limit of the day, consider upgrading your plan. You can get 15% off by "
                "using the following affiliate link which also supports the project: "
                "https://www.jeroenbouma.com/fmp"
            )

    if us_stocks_only:
        logger.error(
            "The Free plan of Financial Modeling Prep is limited to US stocks only. "
            "Therefore the following tickers are not available: %s\nConsider upgrading your plan to Starter or "
            "higher. You can get 15%% off by using the following affiliate link which also "
            "supports the project: https://www.jeroenbouma.com/fmp",
            ", ".join(us_stocks_only),
        )

    if invalid_api_key:
        logger.error(
            "You have entered an invalid API key from Financial Modeling Prep. Obtain an API key for free "
            "or get 15% off the Premium plans by using the following affiliate link.\nThis also supports "
            "the project: https://www.jeroenbouma.com/fmp"
        )

    if request_failed:
        logger.error(
            "The request to Financial Modeling Prep failed with an unrecognised error for the "
            "following tickers: %s.\nNo data could be collected for them.",
            ", ".join(request_failed),
        )

    if no_errors:
        # These exhausted every connection retry, so this is a network failure rather
        # than the absence of an error it is named after.
        logger.error(
            "The connection to Financial Modeling Prep could not be established for the "
            "following tickers: %s.\nNo data could be collected for them.",
            ", ".join(no_errors),
        )

    if delete_tickers:
        # Tickers that errored are removed so the rest of the program continues.
        removed_tickers = set(
            premium_query_parameter
            + exclusive_endpoint
            + special_endpoint
            + not_available
            + bandwidth_limit_reach
            + limit_reach
            + yfinance_rate_limit_or_no_data_found
            + yfinance_rate_limit_or_no_data_found_fallback
            + yfinance_rate_limit_reached
            + yfinance_rate_limit_reached_fallback
            + us_stocks_only
            + no_data
            + invalid_api_key
            + request_failed
            + no_errors
        )

        for ticker in removed_tickers:
            del dataset_dictionary[ticker]

    return dataset_dictionary
