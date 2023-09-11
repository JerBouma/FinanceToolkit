"""Historical Module"""
__docformat__ = "google"

from datetime import datetime, timedelta
from urllib.error import HTTPError

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False

from financetoolkit.base import fundamentals_model

# pylint: disable=too-many-locals

TREASURY_LIMIT = 90


def get_historical_data(
    tickers: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
    progress_bar: bool = True,
    fill_nan: bool = True,
    divide_ohlc_by: int | float | None = None,
    rounding: int | None = None,
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
        risk_free_rate (pd.Series, optional): A pandas Series object containing the risk free rate data.
        This is used to calculate the excess return and excess volatility. Defaults to pd.Series().
        progress_bar (bool, optional): A boolean representing whether to show a progress bar or not.
        fill_nan (bool, optional): A boolean representing whether to fill NaN values with the previous value.
        divide_ohlc_by (int or float, optional): A number to divide the OHLC data by. Defaults to None.
        rounding (int, optional): The number of decimal places to round the data to. Defaults to None.

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

    ticker_list_iterator = (
        tqdm(ticker_list, desc="Obtaining historical data")
        if (ENABLE_TQDM & progress_bar)
        else ticker_list
    )

    for ticker in ticker_list_iterator:
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

        if historical_data_dict[ticker].loc[start:end].empty:
            print(f"The given start and end date result in no data found for {ticker}")
            del historical_data_dict[ticker]
            invalid_tickers.append(ticker)
            continue

        historical_data_dict[ticker].index = pd.to_datetime(
            historical_data_dict[ticker].index
        )
        historical_data_dict[ticker].index = historical_data_dict[
            ticker
        ].index.to_period(freq="D")

        if divide_ohlc_by:
            # Set divide by zero and invalid value warnings to ignore as it is fine that
            # dividing NaN by divide_ohlc_by results in NaN
            np.seterr(divide="ignore", invalid="ignore")
            # In case tickers are presented in percentages or similar
            historical_data_dict[ticker] = historical_data_dict[ticker].div(
                divide_ohlc_by
            )

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

        historical_data_dict[ticker] = enrich_historical_data(
            historical_data=historical_data_dict[ticker],
            start=start,
            end=end,
            return_column=return_column,
            risk_free_rate=risk_free_rate,
        )

    if historical_data_dict:
        historical_data = pd.concat(historical_data_dict, axis=1)
        historical_data.columns = historical_data.columns.swaplevel(0, 1)

        # Sort the DataFrame with respect to the original column order
        column_order = historical_data.columns.get_level_values(0).unique()
        historical_data = historical_data.sort_index(
            level=0, axis=1, sort_remaining=False
        ).reindex(column_order, level=0, axis=1)

        historical_data = historical_data.sort_index()

        if fill_nan:
            historical_data = historical_data.ffill()

        if rounding:
            historical_data = historical_data.round(rounding)

        return historical_data, invalid_tickers

    return pd.DataFrame(), invalid_tickers


def enrich_historical_data(
    historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    return_column: str = "Adj Close",
    risk_free_rate: pd.DataFrame = pd.DataFrame(),
):
    """
    Retrieves enriched historical stock data for the given ticker(s) from Yahoo! Finance API for
    a specified period. It calculates the following:

        - Return: The return for the given period.
        - Volatility: The volatility for the given period.
        - Excess Return: The excess return for the given period.
        - Excess Volatility: The excess volatility for the given period.
        - Cumulative Return: The cumulative return for the given period.

    The return is calculated as the percentage change in the given return column and the excess return
    is calculated as the percentage change in the given return column minus the risk free rate.

    The volatility is calculated as the standard deviation of the daily returns and the excess volatility
    is calculated as the standard deviation of the excess returns.

    The cumulative return is calculated as the cumulative product of the percentage change in the given
    return column.

    Args:
        historical_data (pd.DataFrame): A pandas DataFrame object containing the historical stock data
        for the given ticker(s).
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        return_column (str, optional): A string representing the column to use for return calculations.
        risk_free_rate (pd.Series, optional): A pandas Series object containing the risk free rate data.
        This is used to calculate the excess return and excess volatility. Defaults to pd.Series().


    Returns:
        pd.DataFrame: A pandas DataFrame object containing the enriched historical stock data for the given ticker(s).
    """

    historical_data["Return"] = historical_data[return_column].ffill().pct_change()

    historical_data["Volatility"] = historical_data.loc[start:end, "Return"].std()

    if not risk_free_rate.empty:
        historical_data["Excess Return"] = historical_data["Return"].sub(
            risk_free_rate["Adj Close"]
        )

        historical_data["Excess Volatility"] = historical_data.loc[
            start:end, "Excess Return"
        ].std()

    historical_data["Cumulative Return"] = 1

    adjusted_return = historical_data.loc[start:end, "Return"].copy()
    adjusted_return.iloc[0] = 0

    historical_data["Cumulative Return"] = pd.Series(np.nan).astype(float)

    historical_data.loc[start:end, "Cumulative Return"] = (
        1.0 + adjusted_return
    ).cumprod()

    return historical_data


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
        datetime.strptime(end, "%Y-%m-%d")
        if end is not None
        else datetime.today() + timedelta(days=1 * 365)
    )

    if start is not None:
        start_date = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1 * 365)

        if start_date > end_date:
            raise ValueError(
                f"Start date ({start_date}) must be before end date ({end_date}))"
            )
    else:
        start_date = datetime.now() - timedelta(days=10 * 365)

        if start_date > end_date:
            start_date = end_date - timedelta(days=10 * 365)

    if (end_date - start_date).days > TREASURY_LIMIT:
        # Given that the limit is set to 90 days, the requests need to be split to
        # be able to collect enough data to match the period requested
        groups = np.ceil((end_date - start_date).days / (TREASURY_LIMIT))
        treasury_rates = pd.DataFrame()

        for group in range(0, int(groups) + 1):
            start_date_group = start_date + timedelta(days=TREASURY_LIMIT * group)
            end_date_group = start_date + timedelta(days=TREASURY_LIMIT * (group + 1))
            start_date_string = start_date_group.strftime("%Y-%m-%d")
            end_date_string = end_date_group.strftime("%Y-%m-%d")

            url = (
                f"https://financialmodelingprep.com/api/v4/treasury?from={start_date_string}"
                f"&to={end_date_string}&apikey={api_key}"
            )

            treasury_rates_subset = fundamentals_model.get_financial_data(
                ticker="TREASURY", url=url
            )

            if "ERROR_MESSAGE" in treasury_rates_subset:
                return pd.DataFrame()

            treasury_rates = pd.concat([treasury_rates, treasury_rates_subset], axis=0)
    else:
        start_date_string = start_date.strftime("%Y-%m-%d")
        end_date_string = end_date.strftime("%Y-%m-%d")

        url = (
            f"https://financialmodelingprep.com/api/v4/treasury?from={start_date_string}"
            f"&to={end_date_string}&apikey={api_key}"
        )
        treasury_rates = fundamentals_model.get_financial_data(
            ticker="TREASURY", url=url
        )

        if "ERROR_MESSAGE" in treasury_rates:
            return pd.DataFrame()

    treasury_rates = treasury_rates.set_index("date")

    # Division is done by 100 to make the numbers represent actual values
    # by default, they are represented as percentages
    treasury_rates = treasury_rates / 100
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
    treasury_rates = treasury_rates.sort_index()
    treasury_rates.index = treasury_rates.index.astype(str)

    return treasury_rates


def convert_daily_to_other_period(
    period: str,
    daily_historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    risk_free_rate: pd.Series = pd.DataFrame(),
    rounding: int | None = None,
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
    period_translation = {
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }
    volatility_window_translation = {
        "weekly": 252 / 52,
        "monthly": 252 / 12,
        "quarterly": 252 / 4,
        "yearly": 252,
    }

    if period not in ["weekly", "monthly", "quarterly", "yearly"]:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    period_str = period_translation[period]
    volatility_window = volatility_window_translation[period]

    daily_historical_data.index.name = "Date"
    dates = daily_historical_data.index.asfreq(period_str)
    daily_historical_data = daily_historical_data.reset_index()
    period_historical_data = daily_historical_data.groupby(dates).transform("last")

    if "Dividends" in period_historical_data:
        period_historical_data["Dividends"] = (
            daily_historical_data["Dividends"].groupby(dates).transform("sum")
        )

    period_historical_data["Date"] = period_historical_data["Date"]
    period_historical_data = period_historical_data.drop_duplicates().set_index("Date")
    period_historical_data.index = pd.PeriodIndex(
        period_historical_data.index, freq=period_str
    )

    if "Return" in period_historical_data:
        period_historical_data["Return"] = (
            period_historical_data["Adj Close"]
            / period_historical_data["Adj Close"].shift()
            - 1
        )

        # Volatility is calculated as the daily volatility multiplied by the
        # square root of the number of trading days (252)
        period_historical_data["Volatility"] = daily_historical_data["Return"].groupby(
            dates
        ).std() * np.sqrt(volatility_window)

        if not risk_free_rate.empty:
            period_historical_data["Excess Return"] = period_historical_data[
                "Return"
            ].sub(risk_free_rate["Adj Close"], axis=0)

            period_historical_data["Excess Volatility"] = daily_historical_data[
                "Excess Return"
            ].groupby(dates).std() * np.sqrt(volatility_window)

    if "Cumulative Return" in period_historical_data:
        if start:
            start = pd.Period(start).asfreq(period_str)

            if start < period_historical_data.index[0]:
                start = period_historical_data.index[0]
        if end:
            end = pd.Period(end).asfreq(period_str)

            if end > period_historical_data.index[-1]:
                end = period_historical_data.index[-1]

        adjusted_return = period_historical_data.loc[start:end, "Return"].copy()
        adjusted_return.iloc[0] = 0

        period_historical_data["Cumulative Return"] = (1 + adjusted_return).cumprod()
        period_historical_data["Cumulative Return"] = period_historical_data[
            "Cumulative Return"
        ].fillna(1)

    period_historical_data = period_historical_data.sort_index()

    if rounding:
        period_historical_data = period_historical_data.round(rounding)

    return period_historical_data.fillna(0)
