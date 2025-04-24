"""Option Helpers Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=too-many-locals


def define_strike_prices(
    tickers: list[str],
    stock_price: pd.DataFrame,
    strike_step_size: int,
    strike_price_range: float,
):
    """
    Define strike prices for each ticker in the tickers list. The strike prices are defined as a range from
    strike_price_range% below the current stock price to strike_price_range% above the current stock price, with a
    step size of strike_step_size. The numbers are rounded to the nearest strike_step_size.


    Args:
        tickers (list[str]): the list of tickers.
        stock_price (pd.DataFrame): the current stock price for each ticker.
        strike_step_size (int): the strike step size.
        strike_price_range (float): the strike price range.

    Returns:
        dict: a dictionary of strike prices for each ticker.
    """
    strike_prices_per_ticker = {}

    for ticker in tickers:
        if strike_price_range < 0 or strike_step_size < 0:
            raise ValueError(
                "The strike price range and the strike step size must be positive."
            )
        if strike_price_range > 1:
            raise ValueError("The strike price range must be between 0 and 1.")
        if strike_step_size == 0:
            raise ValueError("The strike step size must be greater than 0.")

        if strike_price_range == 0:
            strike_prices_per_ticker[ticker] = [
                strike_step_size
                * round(int(stock_price.loc[ticker]) / strike_step_size)
            ]
        else:
            strike_prices_per_ticker[ticker] = list(
                range(
                    strike_step_size
                    * round(
                        int(stock_price.loc[ticker] * (1 - strike_price_range))
                        / strike_step_size
                    ),
                    strike_step_size
                    * round(
                        int(stock_price.loc[ticker] * (1 + strike_price_range))
                        / strike_step_size
                    ),
                    strike_step_size,
                )
            )

    return strike_prices_per_ticker


def create_greek_dataframe(
    greek_dictionary: dict[str, dict[float, dict[float, float]]], start_date: str
):
    """
    Creates the DataFrame that correctly displays the greeks for each ticker and strike price.
    over time (the time of expiration).

    Args:
        greek_dictionary (dict[str, dict[float, dict[float, float]]]): a dictionary
        containing the greeks for each ticker, strike price and expiration date.
        start_date (str): the start date that should be excluded out of the DataFrame.

    Returns:
        pd.DataFrame: the DataFrame that correctly displays the greeks for each ticker
        and strike price.
    """
    greek_dataframe = pd.DataFrame.from_dict(
        {
            (ticker, strike_price): values
            for ticker, strike_prices in greek_dictionary.items()
            for strike_price, values in strike_prices.items()
        },
        orient="index",
    )

    greek_dataframe.columns = pd.period_range(
        start=start_date,
        periods=len(greek_dataframe.columns),
        freq="D",
    )

    greek_dataframe = greek_dataframe.drop(start_date, axis=1)

    greek_dataframe.index.names = ["Ticker", "Strike Price"]

    return greek_dataframe


def create_binomial_tree_dataframe(
    binomial_tree_dictionary: dict[str, dict[float, pd.DataFrame]],
    start_date: pd.PeriodIndex,
    time_to_expiration: int,
):
    """
    Creates the DataFrame that correctly displays the binomial tree for each ticker and strike price.
    over time (the time of expiration).

    Args:
        binomial_tree_dictionary (dict[str, dict[float, dict[float, float]]]): a dictionary
        containing the binomial tree for each ticker, strike price and expiration date.
        start_date (str): the start date that should be excluded out of the DataFrame.

    Returns:
        pd.DataFrame: the DataFrame that correctly displays the binomial tree for each ticker
        and strike price.
    """
    binomial_tree_dataframe = pd.DataFrame.from_dict(
        {
            (ticker, strike_price, movement): values
            for ticker, strike_prices in binomial_tree_dictionary.items()
            for strike_price, movements in strike_prices.items()
            for movement, values in movements.T.items()
        },
        orient="index",
    )

    start_date_string = pd.to_datetime(str(start_date))
    end_date_string = pd.to_datetime(
        str(start_date_string + pd.Timedelta(days=time_to_expiration * 365))
    )

    binomial_tree_dataframe.columns = pd.date_range(
        start=start_date_string,
        end=end_date_string,
        periods=len(binomial_tree_dataframe.columns),
    )

    binomial_tree_dataframe.columns = pd.PeriodIndex(
        binomial_tree_dataframe.columns, freq="D"
    )

    binomial_tree_dataframe.index.names = ["Ticker", "Strike Price", "Movement"]

    return binomial_tree_dataframe


def create_stock_simulation_dataframe(
    stock_simulation_dictonary: dict[str, dict[float, dict[float, float]]],
    start_date: pd.PeriodIndex,
    time_to_expiration: int,
):
    """
    Creates the DataFrame that correctly displays the binomial tree for each ticker and strike price.
    over time (the time of expiration).

    Args:
        binomial_tree_dictionary (dict[str, dict[float, dict[float, float]]]): a dictionary
        containing the binomial tree for each ticker, strike price and expiration date.
        start_date (str): the start date that should be excluded out of the DataFrame.

    Returns:
        pd.DataFrame: the DataFrame that correctly displays the binomial tree for each ticker
        and strike price.
    """
    stock_simulation_dataframe = pd.concat(stock_simulation_dictonary)

    start_date_string = pd.to_datetime(str(start_date))
    end_date_string = pd.to_datetime(
        str(start_date_string + pd.Timedelta(days=time_to_expiration * 365))
    )

    stock_simulation_dataframe.columns = pd.date_range(
        start=start_date_string,
        end=end_date_string,
        periods=len(stock_simulation_dataframe.columns),
    )

    stock_simulation_dataframe.columns = pd.PeriodIndex(
        stock_simulation_dataframe.columns, freq="D"
    )

    stock_simulation_dataframe.index.names = ["Ticker", "Movement"]

    return stock_simulation_dataframe


def show_input_info(
    start_date: str,
    end_date: str,
    stock_prices: pd.Series,
    volatility: pd.Series,
    risk_free_rate: float,
    dividend_yield: dict[str, float] | None = None,
    up_movement_dict: dict | None = None,
    down_movement_dict: dict | None = None,
    risk_neutral_probability_dict: dict | None = None,
):
    """
    Based on the input parameters, print the input information.

    Args:
        start_date (str): the start date.
        end_date (str): the end date.
        stock_prices (pd.Series): the stock price for each ticker.
        volatility (pd.Series): the volatility for each ticker.
        risk_free_rate (float): the risk free rate.
        dividend_yield (dict[str, float]): the dividend yield for each ticker.

    Returns:
        a print statement with the input information.
    """
    stock_price_list = [
        (f"{ticker} ({round(stock_price_value, 2)})")
        for ticker, stock_price_value in stock_prices.items()
    ]
    volatility_list = [
        (f"{ticker} ({round(volatility_value * 100, 2)}%)")
        for ticker, volatility_value in volatility.items()
    ]

    stock_price_str = ", ".join(stock_price_list)
    volatility_str = ", ".join(volatility_list)

    logger.info(
        "Based on the period %s to %s the following parameters were used:\n"
        "Stock Price: %s\n"
        "Volatility: %s",
        start_date,
        end_date,
        stock_price_str,
        volatility_str,
    )

    if dividend_yield is not None:
        dividend_yield_list = [
            (f"{ticker} ({round(dividend_yield_value * 100, 2)}%)")
            for ticker, dividend_yield_value in dividend_yield.items()
        ]
        logger.info("Dividend Yield: %s", ", ".join(dividend_yield_list))

    if up_movement_dict is not None:
        up_movement_list = [
            (f"{ticker} ({round(up_movement * 100, 2)}%)")
            for ticker, up_movement in up_movement_dict.items()
        ]
        logger.info("Up Movement: %s", ", ".join(up_movement_list))

    if down_movement_dict is not None:
        down_movement_list = [
            (f"{ticker} ({round(down_movement * 100, 2)}%)")
            for ticker, down_movement in down_movement_dict.items()
        ]
        logger.info("Down Movement: %s", ", ".join(down_movement_list))

    if risk_neutral_probability_dict is not None:
        risk_neutral_probability_list = [
            (f"{ticker} ({round(risk_neutral_probability * 100, 2)}%)")
            for ticker, risk_neutral_probability in risk_neutral_probability_dict.items()
        ]
        logger.info(
            "Risk Neutral Probability: %s", ", ".join(risk_neutral_probability_list)
        )

    logger.info("Risk Free Rate: %s%%", round(risk_free_rate * 100, 2))
