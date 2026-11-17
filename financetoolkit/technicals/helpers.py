"""Technicals Helpers Module"""

__docformat__ = "google"

from collections.abc import Callable

import pandas as pd

# The observations in a year of a ticker that trades on an exchange's trading days, and of
# one that also trades on weekends, such as a cryptocurrency.
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


def get_non_trading_days(historical_data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    The dates on which a ticker did not trade while another ticker or the Benchmark did.

    A Toolkit with tickers on different calendars, such as a stock and a cryptocurrency or
    stocks of two exchanges, holds the dates of all of them. On a date only another ticker
    traded, a ticker's last price is carried forward as a flat bar (Open, High, Low and
    Close at the last close, a Volume of 0) without a Return (see
    `historical_model.fill_non_trading_days`). Those dates are recognised by the missing
    Return after the ticker's first price, up to its last Return. A ticker's first price has
    no Return by construction and is a trading day.

    Args:
        historical_data (pd.DataFrame): The historical data of one period with (column,
            ticker) columns, including "Return" and `column`.
        column (str): The price column whose tickers are checked, e.g. "Close".

    Returns:
        pd.DataFrame: True on the dates a ticker did not trade, a column per ticker.
    """
    prices = historical_data[column]
    if "Return" not in historical_data.columns.get_level_values(0):
        return pd.DataFrame(False, index=prices.index, columns=prices.columns)

    has_return = historical_data["Return"].reindex(columns=prices.columns).notna()
    after_first_price = prices.notna().cumsum() > 1
    up_to_last_return = has_return[::-1].cummax()[::-1]

    return ~has_return & after_first_price & up_to_last_return


def calculate_per_ticker(
    historical_data: pd.DataFrame,
    columns: list[str],
    function: Callable[..., pd.Series | pd.DataFrame],
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculates an indicator per ticker over the dates the ticker traded.

    An indicator over a number of periods counts bars, so the flat bars a ticker gets on the
    dates only another ticker traded (see `get_non_trading_days`) would count as periods
    without a move: they lower an Average True Range or a Bollinger Band width, lengthen
    every window in calendar time and turn a ratio of ranges into 0 / 0. Each ticker is
    therefore calculated on its own trading days, and a result per date is placed back on
    every date of the data, NaN on the dates the ticker did not trade. A ticker that traded
    on every date is calculated on its data as it is, so a Toolkit whose tickers share one
    calendar gets the same results as a calculation over all tickers at once. A result that
    is not per date, such as the Volatility Cone, is returned as it is.

    Args:
        historical_data (pd.DataFrame): The historical data of one period with (column,
            ticker) columns.
        columns (list[str]): The columns the function takes, in its order, e.g.
            ["High", "Low", "Close"]. The tickers are those of the first column.
        function (Callable[..., pd.Series | pd.DataFrame]): The indicator, which takes a
            Series per column followed by `args` and `kwargs`.
        *args: The further positional arguments of the function, such as the window.
        **kwargs: The further keyword arguments of the function.

    Returns:
        pd.DataFrame: The indicator with a column per ticker when the function returns a
        Series, and (indicator column, ticker) columns when it returns a DataFrame.
    """
    non_trading_days = get_non_trading_days(historical_data, columns[0])

    results = {}
    for ticker in historical_data[columns[0]].columns:
        inputs = [historical_data[column][ticker] for column in columns]
        traded = ~non_trading_days[ticker]

        if traded.all():
            results[ticker] = function(*inputs, *args, **kwargs)
            continue

        inputs = [values[traded] for values in inputs]
        result = function(*inputs, *args, **kwargs)
        results[ticker] = (
            result.reindex(historical_data.index)
            if result.index.equals(inputs[0].index)
            else result
        )

    if all(isinstance(result, pd.Series) for result in results.values()):
        return pd.DataFrame(results)

    return pd.concat(results, axis=1).swaplevel(1, 0, axis=1).sort_index(axis=1)


def get_observations_per_year(prices: pd.Series) -> int:
    """
    The number of daily observations in a year of a ticker, to annualize a daily measure or
    to express a year in periods: 252 for a ticker that trades on weekdays only, and 365 for
    one with prices on a Saturday or Sunday, such as a cryptocurrency that trades every
    calendar day. Counting the prices of each year instead would give a stock between 250
    and 253 depending on its holidays, so a ticker on an exchange's trading days keeps the
    252 convention.

    Args:
        prices (pd.Series): The daily (or intraday) prices with a PeriodIndex or
            DatetimeIndex, NaN on the dates without a price.

    Returns:
        int: The observations per year, 252 or 365.
    """
    dates = prices.index[prices.notna().to_numpy()]

    return (
        CALENDAR_DAYS_PER_YEAR
        if bool((dates.dayofweek >= 5).any())  # noqa: PLR2004
        else TRADING_DAYS_PER_YEAR
    )
