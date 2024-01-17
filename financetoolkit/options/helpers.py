"""Option Helpers Module"""
__docformat__ = "google"

import pandas as pd


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


def show_input_info(
    start_date: str,
    end_date: str,
    stock_prices: pd.Series,
    volatility: pd.Series,
    risk_free_rate: float,
    dividend_yield: dict[str, float],
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

    dividend_yield_list = [
        (f"{ticker} ({round(dividend_yield_value * 100, 2)}%)")
        for ticker, dividend_yield_value in dividend_yield.items()
    ]

    print(
        f"Based on the period {start_date} to {end_date} "
        "the following parameters were used:\n"
        f"Stock Price: {', '.join(stock_price_list)}\n"
        f"Volatility: {', '.join(volatility_list)}\n"
        f"Dividend Yield: {', '.join(dividend_yield_list)}\n"
        f"Risk Free Rate: {round(risk_free_rate * 100, 2)}%\n"
    )
