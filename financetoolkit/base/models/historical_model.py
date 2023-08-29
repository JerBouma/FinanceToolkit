"""Historical Module"""
__docformat__ = "google"

from datetime import datetime, timedelta
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from financetoolkit.base.models import fundamentals_model

# pylint: disable=too-many-locals


def get_historical_data(
    tickers: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    risk_free_rate: pd.Series | int = 0,
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
        interval (str, optional): A string representing the interval to retrieve data for.
        return_column (str, optional): A string representing the column to use for return calculations.

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
        # Additional data is collected to ensure return calculations are correct
        end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1 * 365)
        end_timestamp = int(end_date.timestamp())
    else:
        end_date = datetime.today()
        end_timestamp = int(end_date.timestamp())
        end = end_date.strftime("%Y-%m-%d")

    if start is not None:
        # Additional data is collected to ensure return calculations are correct
        start_date = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1 * 365)

        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date}))"
            )

        start_timestamp = int(start_date.timestamp())
    else:
        start_date = datetime.now() - timedelta(days=10 * 365)
        start = start_date.strftime("%Y-%m-%d")

        if start_date > end_date:
            start_date = end_date - timedelta(days=10 * 365)

        start_timestamp = int(start_date.timestamp())

    historical_data_dict: dict = {}

    if interval in ["yearly", "quarterly"]:
        interval = "1d"

    invalid_tickers = []
    for ticker in ticker_list:
        historical_data_url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?"
            f"interval={interval}&period1={start_timestamp}&period2={end_timestamp}"
            "&events=history&includeAdjustedClose=true"
        )

        dividend_url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?"
            f"interval={interval}&period1={start_timestamp}&period2={end_timestamp}"
            "&events=div&includeAdjustedClose=true"
        )

        try:
            historical_data_dict[ticker] = pd.read_csv(
                historical_data_url, index_col="Date"
            )
        except HTTPError:
            print(f"No historical data found for {ticker}")
            invalid_tickers.append(ticker)
            continue

        try:
            historical_data_dict[ticker]["Dividends"] = pd.read_csv(
                dividend_url, index_col="Date"
            )["Dividends"]
            historical_data_dict[ticker]["Dividends"] = historical_data_dict[ticker][
                "Dividends"
            ].fillna(0)
        except HTTPError:
            print(f"No dividend data found for {ticker}")
            continue

        historical_data_dict[ticker]["Return"] = historical_data_dict[ticker][
            return_column
        ].pct_change()

        historical_data_dict[ticker]["Volatility"] = (
            historical_data_dict[ticker].loc[start:end, "Return"].std()  # type: ignore
        )

        historical_data_dict[ticker]["Excess Return"] = historical_data_dict[ticker][
            "Return"
        ].sub(risk_free_rate, axis=0)

        historical_data_dict[ticker]["Excess Volatility"] = (
            historical_data_dict[ticker].loc[start:end, "Excess Return"].std()  # type: ignore
        )

        historical_data_dict[ticker]["Cumulative Return"] = 1

        adjusted_return = historical_data_dict[ticker].loc[start:end, "Return"].copy()  # type: ignore
        adjusted_return.iloc[0] = 0

        historical_data_dict[ticker].loc[start:end, "Cumulative Return"] = (  # type: ignore
            1 + adjusted_return
        ).cumprod()

    if historical_data_dict:
        historical_data = pd.concat(historical_data_dict, axis=1)
        historical_data.columns = historical_data.columns.swaplevel(0, 1)

        # Sort the DataFrame with respect to the original column order
        column_order = historical_data.columns.get_level_values(0).unique()
        historical_data = historical_data.sort_index(
            level=0, axis=1, sort_remaining=False
        ).reindex(column_order, level=0, axis=1)

        return historical_data, invalid_tickers

    return pd.DataFrame(), invalid_tickers


def get_treasury_rates(
    api_key: str,
    start: str | None = None,
    end: str | None = None,
    rounding: int | None = 4,
):
    """
    Retrieves treasury rates for a specified period. Please note that the maximum period is 3 months.

    Args:
        api_key (str): A string representing the API key to use for the request.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        rounding (int, optional): The number of decimal places to round the data to. Defaults to 4.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the treasury rates for the given period.
    """
    naming = {
        "month1": "1 Month",
        "month2": "2 Month",
        "month3": "3 Month",
        "month6": "6 Month",
        "year1": "1 Year",
        "year2": "2 Year",
        "year3": "3 Year",
        "year5": "5 Year",
        "year7": "7 Year",
        "year10": "10 Year",
        "year20": "20 Year",
        "year30": "30 Year",
    }

    end_date = (
        datetime.strptime(end, "%Y-%m-%d") + timedelta(days=30)
        if end is not None
        else datetime.today()
    )

    if start is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=30)

        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date}))"
            )
    else:
        start_date = datetime.now() - timedelta(days=100)

        if start_date > end_date:
            start_date = end_date - timedelta(days=100)

    start_date_string = start_date.strftime("%Y-%m-%d")
    end_date_string = end_date.strftime("%Y-%m-%d")

    url = (
        f"https://financialmodelingprep.com/api/v4/treasury?from={start_date_string}"
        f"&to={end_date_string}&apikey={api_key}"
    )

    treasury_rates = fundamentals_model.get_financial_data(ticker="TREASURY", url=url)

    if "ERROR_MESSAGE" in treasury_rates:
        return pd.DataFrame()

    treasury_rates = treasury_rates.set_index("date")

    try:
        treasury_rates = treasury_rates.astype(np.float64)
    except ValueError as error:
        print(
            f"Not able to convert DataFrame to float64 due to {error}. This could result in"
            "issues when values are zero and is predominently relevant for "
            "ratio calculations."
        )

    treasury_rates = treasury_rates.rename(columns=naming)
    treasury_rates = treasury_rates.round(rounding)

    return treasury_rates


def convert_daily_to_yearly(
    daily_historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    yearly_risk_free_rate: pd.Series | int = 0,
):
    """
    Converts daily historical data to yearly historical data.

    Args:
        daily_historical_data (pd.DataFrame): A DataFrame containing daily historical data.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the yearly historical stock data.
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    daily_historical_data.index.name = "Date"
    daily_historical_data = daily_historical_data.reset_index()
    dates = pd.to_datetime(daily_historical_data.Date).dt.to_period("Y")
    yearly_historical_data = daily_historical_data.groupby(dates).transform("last")

    if "Dividends" in yearly_historical_data:
        yearly_historical_data["Dividends"] = (
            daily_historical_data["Dividends"].groupby(dates).transform("sum")
        )

    yearly_historical_data["Date"] = yearly_historical_data["Date"].str[:4]
    yearly_historical_data = yearly_historical_data.drop_duplicates().set_index("Date")
    yearly_historical_data.index = pd.PeriodIndex(
        yearly_historical_data.index, freq="Y"
    )

    if "Return" in yearly_historical_data:
        yearly_historical_data["Return"] = (
            yearly_historical_data["Adj Close"]
            / yearly_historical_data["Adj Close"].shift()
            - 1
        )

        # Volatility is calculated as the daily volatility multiplied by the
        # square root of the number of trading days (252)
        yearly_historical_data["Volatility"] = daily_historical_data["Return"].groupby(
            dates
        ).agg(np.std) * np.sqrt(252)

        yearly_historical_data["Excess Return"] = yearly_historical_data["Return"].sub(
            yearly_risk_free_rate, axis=0
        )

        yearly_historical_data["Excess Volatility"] = daily_historical_data[
            "Excess Return"
        ].groupby(dates).agg(np.std) * np.sqrt(252)

    if "Cumulative Return" in yearly_historical_data:
        if start:
            start = pd.Period(start).asfreq("Y")

            if start < yearly_historical_data.index[0]:
                start = yearly_historical_data.index[0]
        if end:
            end = pd.Period(end).asfreq("Y")

            if end > yearly_historical_data.index[-1]:
                end = yearly_historical_data.index[-1]

        adjusted_return = yearly_historical_data.loc[start:end, "Return"].copy()  # type: ignore
        adjusted_return.iloc[0] = 0

        yearly_historical_data["Cumulative Return"] = (1 + adjusted_return).cumprod()  # type: ignore
        yearly_historical_data["Cumulative Return"] = yearly_historical_data[
            "Cumulative Return"
        ].fillna(1)

    return yearly_historical_data.fillna(0)


def convert_daily_to_quarterly(
    daily_historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    quarterly_risk_free_rate: pd.Series | int = 0,
):
    """
    Converts daily historical data to quarterly historical data.

    Args:
        daily_historical_data (pd.DataFrame): A DataFrame containing daily historical data.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the yearly historical stock data.
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    daily_historical_data.index.name = "Date"
    daily_historical_data = daily_historical_data.reset_index()
    dates = pd.to_datetime(daily_historical_data.Date).dt.to_period("Q")
    quarterly_historical_data = daily_historical_data.groupby(dates).transform("last")

    if "Dividends" in quarterly_historical_data:
        quarterly_historical_data["Dividends"] = (
            daily_historical_data["Dividends"].groupby(dates).transform("sum")
        )

    quarterly_historical_data["Date"] = quarterly_historical_data["Date"].str[:7]
    quarterly_historical_data = quarterly_historical_data.drop_duplicates().set_index(
        "Date"
    )
    quarterly_historical_data.index = pd.PeriodIndex(
        quarterly_historical_data.index, freq="Q"
    )

    if "Return" in quarterly_historical_data:
        quarterly_historical_data["Return"] = (
            quarterly_historical_data["Adj Close"]
            / quarterly_historical_data["Adj Close"].shift()
            - 1
        )

        # Volatility is calculated as the daily volatility multiplied by the
        # square root of the number of trading days divided by 4 (252 / 4)
        quarterly_historical_data["Volatility"] = daily_historical_data[
            "Return"
        ].groupby(dates).agg(np.std) * np.sqrt(63)

        quarterly_historical_data["Excess Return"] = quarterly_historical_data[
            "Return"
        ].sub(quarterly_risk_free_rate, axis=0)

        quarterly_historical_data["Excess Volatility"] = daily_historical_data[
            "Excess Return"
        ].groupby(dates).agg(np.std) * np.sqrt(63)

    if "Cumulative Return" in quarterly_historical_data:
        if start:
            start = pd.Period(start).asfreq("Q")

            if start < quarterly_historical_data.index[0]:
                start = quarterly_historical_data.index[0]
        if end:
            end = pd.Period(end).asfreq("Q")

            if end > quarterly_historical_data.index[-1]:
                end = quarterly_historical_data.index[-1]

        adjusted_return = quarterly_historical_data.loc[start:end, "Return"].copy()  # type: ignore
        adjusted_return.iloc[0] = 0

        quarterly_historical_data["Cumulative Return"] = (1 + adjusted_return).cumprod()
        quarterly_historical_data["Cumulative Return"] = quarterly_historical_data[
            "Cumulative Return"
        ].fillna(1)

    return quarterly_historical_data.fillna(0)
