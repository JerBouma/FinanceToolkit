"""Error Module"""

__docformat__ = "google"

import inspect

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


# pylint: disable=comparison-with-itself,too-many-locals,protected-access


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
            logger.error(
                "There is an index name missing in the provided financial statements. "
                "This is %s. This is required for the function (%s) "
                "to run. Please fill this column to be able to calculate the ratios.",
                e,
                function_name,
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
                "An error occurred while trying to run the function "
                "%s. %s This is due to a division by zero.",
                function_name,
                e,
            )
            return pd.Series(dtype="object")
        except IndexError as e:
            function_name = func.__name__
            logger.error(
                "An error occurred while trying to run the function "
                "%s. %s This is due to missing data.",
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
                "limit of the day, consider upgrading your plan. You can get 15%% off by "
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
            "or get 15%% off the Premium plans by using the following affiliate link.\nThis also supports "
            "the project: https://www.jeroenbouma.com/fmp"
        )

    if delete_tickers:
        # If any of the errors are found, remove the tickers from the dataset dictionary so that
        # the user can continue using the program without having to worry about the errors.
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
            + us_stocks_only
            + invalid_api_key
            + no_errors
        )

        for ticker in removed_tickers:
            del dataset_dictionary[ticker]

    return dataset_dictionary
