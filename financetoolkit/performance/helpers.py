"""Performance Helpers Module"""

__docformat__ = "google"

import inspect

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=protected-access

PERIOD_TRANSLATION: dict[str, str | dict[str, str]] = {
    "intraday": {
        "1min": "h",
        "5min": "h",
        "15min": "D",
        "30min": "D",
        "1hour": "D",
    },
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def determine_within_historical_data(
    daily_historical_data: pd.DataFrame,
    intraday_historical_data: pd.DataFrame,
    intraday_period: str | None,
):
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
    within_historical_data = {}

    for period, symbol in PERIOD_TRANSLATION.items():
        if not intraday_period and period == "intraday":
            continue

        period_symbol = (
            symbol[intraday_period] if period == "intraday" else symbol  # type: ignore
        )

        if not intraday_historical_data.empty and period in [
            "intraday",
            "daily",
        ]:
            within_historical_data[period] = (
                intraday_historical_data.groupby(pd.Grouper(freq=period_symbol))
                .apply(lambda x: x)
                .dropna(how="all", axis=0)
            )
        else:
            within_historical_data[period] = daily_historical_data.groupby(
                pd.Grouper(
                    freq=(
                        f"{period_symbol}E"
                        if period_symbol in ["M", "Q", "Y"]
                        else period_symbol
                    )
                )
            ).apply(lambda x: x)

        within_historical_data[period].index = within_historical_data[
            period
        ].index.set_levels(
            [
                pd.PeriodIndex(
                    within_historical_data[period].index.levels[0],
                    freq=period_symbol,
                ),
                pd.PeriodIndex(
                    within_historical_data[period].index.levels[1],
                    freq="D" if period != "intraday" else "min",
                ),
            ],
        )

    return within_historical_data


def determine_within_dataset(
    dataset: pd.DataFrame, period: str, correlation: bool = False
):
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
    dataset_new = dataset.copy()
    dataset_new.index = pd.DatetimeIndex(dataset_new.to_timestamp().index)
    period_symbol = PERIOD_TRANSLATION[period]

    within_historical_data = dataset_new.groupby(
        pd.Grouper(
            freq=(
                f"{period_symbol}E"
                if period_symbol in ["M", "Q", "Y"]
                else period_symbol
            )
        )
    ).apply(lambda x: x.corr() if correlation else x)

    within_historical_data.index = within_historical_data.index.set_levels(
        [
            pd.PeriodIndex(
                within_historical_data.index.levels[0],
                freq=period_symbol,
            ),
            (
                within_historical_data.index.levels[1]
                if correlation
                else pd.PeriodIndex(within_historical_data.index.levels[1], freq="D")
            ),
        ],
    )

    return within_historical_data


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
                logger.error(
                    "Please set a benchmark_ticker in the Toolkit class to calculate %s. "
                    "For example: toolkit = Toolkit(['TSLA', 'AAPL', 'MSFT'], benchmark_ticker='SPY')",
                    function_name,
                )
            else:
                logger.error(
                    "There is an index name missing in the provided historical dataset. "
                    "This is %s. This is required for the function (%s) "
                    "to run. Please fill this column to be able to calculate the metrics.",
                    e,
                    function_name,
                )
            return pd.Series(dtype="object")
        except ValueError as e:
            function_name = func.__name__
            if "Benchmark" in str(e):
                logger.error(
                    "Please set a benchmark_ticker in the Toolkit class to calculate %s. "
                    "For example: toolkit = Toolkit(['TSLA', 'AAPL', 'MSFT'], benchmark_ticker='SPY')",
                    function_name,
                )
            else:
                logger.error(
                    "An error occurred while trying to run the function " + "%s. %s",
                    function_name,
                    e,
                )
            return pd.Series(dtype="object")
        except AttributeError as e:
            function_name = func.__name__
            if "Benchmark" in str(e):
                logger.error(
                    "Please set a benchmark_ticker in the Toolkit class to calculate %s. "
                    + "For example: toolkit = Toolkit(['TSLA', 'AAPL', 'MSFT'], benchmark_ticker='SPY')",
                    function_name,
                )
            else:
                logger.error(
                    "An error occurred while trying to run the function " + "%s. %s",
                    function_name,
                    e,
                )
            return pd.Series(dtype="object")
        except ZeroDivisionError as e:
            function_name = func.__name__
            logger.error(
                "An error occurred while trying to run the function " + "%s. %s",
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
