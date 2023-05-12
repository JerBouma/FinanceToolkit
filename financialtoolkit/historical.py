"""Historical Module"""
__docformat__ = "numpy"

from datetime import datetime, timedelta

import pandas as pd


def get_historical_data(
    tickers: list[str] | str,
    start: str = None,
    end: str = None,
    interval: str = "1d",
):
    """
    Retrieves historical stock data for the given ticker(s) from Yahoo! Finance API for a specified period.
    If start and/or end date are not provided, it defaults to 10 years from the current date.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    if end is not None:
        end_date = datetime.strptime(end, "%Y-%m-%d")
        end_timestamp = int(end_date.timestamp())
    else:
        end_date = datetime.today()
        end_timestamp = int(end_date.timestamp())

    if start is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d")

        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date}))"
            )

        start_timestamp = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    else:
        start_date = datetime.now() - timedelta(days=10 * 365)

        if start_date > end_date:
            start_date = end_date - timedelta(days=10 * 365)

        start_timestamp = int(start_date.timestamp())

    historical_data_dict: dict = {}

    if interval == "yearly":
        yearly = True
        interval = "1d"
    else:
        yearly = False

    for ticker in ticker_list:
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?"
            f"interval={interval}&period1={start_timestamp}&period2={end_timestamp}"
            "&events=history&includeAdjustedClose=true"
        )

        historical_data_dict[ticker] = pd.read_csv(url, index_col="Date")

    historical_data = pd.concat(historical_data_dict, axis=1)

    if len(ticker_list) == 1:
        historical_data = historical_data[ticker_list[0]]
    else:
        historical_data.columns = historical_data.columns.swaplevel(0, 1)
        historical_data = historical_data.sort_index(axis=1)

    if yearly:
        historical_data = convert_daily_to_yearly(historical_data)

    return historical_data


def get_returns(historical_data: pd.Series) -> pd.Series:
    """
    Calculate the returns of historical data for a given Series, with an option to include a rolling period.

    Args:
        historical_data (pd.Series): Series containing historical data.

    Returns:
        pd.Series: Series containing the calculated returns.
    """
    return historical_data.pct_change()


def get_volatility(returns: pd.Series) -> pd.Series:
    """
    Calculate the volatility of returns over a rolling window.

    Args:
        returns (pd.Series): A Series of returns with time as index
        rolling (int): The size of the rolling window.

    Returns:
        pd.Series: A Series of volatility with time as index and assets as columns.
    """
    return returns.std()


def get_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float | pd.Series
) -> pd.Series:
    """
    Calculate the Sharpe ratio of returns over a rolling window.

    Args:
        returns (pd.Series): A Series of returns with time as index
        risk_free_rate (float or pd.Series): The annualized risk-free rate.

    Returns:
        pd.Series: A Series of Sharpe ratios with time as index and assets as columns.
    """
    excess_returns = returns - risk_free_rate

    return excess_returns / returns.std()


def get_sortino_ratio(
    returns: pd.Series, risk_free_rate: float | pd.Series
) -> pd.Series:
    """
    Calculate the Sortino ratio of returns over a rolling window.

    Args:
        returns (pd.Series): A Series of returns with time as index
        risk_free_rate (float or pd.Series): The annualized risk-free rate.
        rolling_window (int): The size of the rolling window.

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """
    excess_returns = returns - risk_free_rate
    downside_volatility = excess_returns[excess_returns < 0].std()

    return excess_returns / downside_volatility


def get_beta(returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculate the beta of returns with respect to a benchmark over a rolling window.

    Args:
        returns (pd.DataFrame): A DataFrame of returns with time as index and assets as columns.
        benchmark_returns (pd.DataFrame): A DataFrame of benchmark returns with time as index and a single column.
        rolling_window (int): The size of the rolling window.

    Returns:
        pd.Series: A Series of beta values with time as index and assets as columns.
    """
    excess_returns = returns - returns.mean()
    excess_benchmark_returns = benchmark_returns - benchmark_returns.mean()
    cov = excess_returns.cov(excess_benchmark_returns)
    var = excess_benchmark_returns.var()

    return cov / var


def convert_daily_to_yearly(daily_historical_data: pd.DataFrame):
    """
    Converts daily historical data to yearly historical data.

    Args:
        daily_historical_data (pd.DataFrame): A DataFrame containing daily historical data.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the yearly historical stock data.
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    daily_historical_data = daily_historical_data.reset_index()
    dates = pd.to_datetime(daily_historical_data.Date).dt.to_period("Y")
    yearly_historical_data = daily_historical_data.groupby(dates).transform("last")
    yearly_historical_data["Date"] = yearly_historical_data["Date"].str[:4]
    yearly_historical_data = yearly_historical_data.drop_duplicates().set_index("Date")

    return yearly_historical_data
