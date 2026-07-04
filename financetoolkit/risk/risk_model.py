"""Risk Model"""

import numpy as np
import pandas as pd

from financetoolkit.helpers import PERIOD_TRANSLATION, VOLATILITY_WINDOW_TRANSLATION

ALPHA_CONSTRAINT = 0.5

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2


def get_max_drawdown(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Maximum Drawdown (MDD) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame | float: MDD values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_max_drawdown(returns.loc[sub_period])
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        max_drawdown = pd.concat(period_data_list, axis=1)

        return max_drawdown.T

    cum_returns = (1 + returns.fillna(0)).cumprod()  # type: ignore

    return (cum_returns / cum_returns.cummax() - 1).min()


def get_ui(
    returns: pd.Series | pd.DataFrame, rolling: int = 14
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Ulcer Index (UI), a measure of downside volatility.

    For more information see:
     - http://www.tangotools.com/ui/ui.htm
     - https://en.wikipedia.org/wiki/Ulcer_index

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        rolling (int, optional): The rolling period to use for the calculation.
        If you select period = 'monthly' and set rolling to 12 you obtain the rolling
        12-month Ulcer Index. If no value is given, then it calculates it for the
        entire period. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: UI values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index, if.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(get_ui)
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            ulcer_index = pd.concat(period_data_list, axis=1)

            return ulcer_index.T

        return returns.aggregate(get_ui)

    if isinstance(returns, pd.Series):
        cumulative_returns = (1 + returns.fillna(0)).cumprod()
        cumulative_max = cumulative_returns.rolling(window=rolling).max()
        drawdowns = (cumulative_returns - cumulative_max) / cumulative_max

        ulcer_index_value = np.sqrt((drawdowns**2).mean())

        return ulcer_index_value

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_skewness(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Computes the skewness of dataset.

    Args:
        dataset (pd.Series | pd.Dataframe): A single index dataframe or series

    Returns:
        pd.Series | pd.Dataframe: Skewness of the dataset
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(get_skewness)
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            skewness = pd.concat(period_data_list, axis=1)

            return skewness.T
        return returns.aggregate(get_skewness)
    if isinstance(returns, pd.Series):
        return returns.skew()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_kurtosis(
    returns: pd.Series | pd.DataFrame, fisher: bool = True
) -> pd.Series | pd.DataFrame:
    """
    Computes the kurtosis of dataset.

    Args:
        dataset (pd.Series | pd.Dataframe): A single index dataframe or series

    Returns:
        pd.Series | pd.Dataframe: Kurtosis of the dataset
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_kurtosis, fisher=fisher
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            kurtosis = pd.concat(period_data_list, axis=1)

            return kurtosis.T
        return returns.aggregate(get_kurtosis, fisher=fisher)
    if isinstance(returns, pd.Series):
        if fisher:
            return returns.kurtosis()
        return (((returns - returns.mean()) / returns.std(ddof=0)) ** 4).mean()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_variance(
    returns: pd.Series | pd.DataFrame, period: str
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Variance of returns for a given period (weekly, monthly,
    quarterly or yearly) based on daily historical returns.

    The daily Variance is scaled to the given period by multiplying it with the
    number of trading days within that period (e.g. 252 / 52 for weekly).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the Variance for. Can be weekly,
        monthly, quarterly or yearly.

    Returns:
        pd.Series | pd.DataFrame: Variance values with time as the index, resampled
        to the given period.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    period_str = PERIOD_TRANSLATION[period]
    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]
    dates = returns.index.asfreq(period_str)

    return returns.groupby(dates).var() * volatility_window


def get_volatility(
    returns: pd.Series | pd.DataFrame, period: str
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) based on daily historical returns.

    The daily Volatility is scaled to the given period by multiplying it with the
    square root of the number of trading days within that period
    (e.g. SQRT(252 / 52) for weekly).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the Volatility for. Can be weekly,
        monthly, quarterly or yearly.

    Returns:
        pd.Series | pd.DataFrame: Volatility values with time as the index, resampled
        to the given period.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    period_str = PERIOD_TRANSLATION[period]
    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]
    dates = returns.index.asfreq(period_str)

    return returns.groupby(dates).std() * np.sqrt(volatility_window)


def get_excess_volatility(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    period: str,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Excess Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) based on daily historical returns.

    The Excess Volatility is the Volatility of the Excess Return, i.e. the daily return
    minus the risk free rate, scaled to the given period in the same way as the Volatility
    (see get_volatility).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        risk_free_rate (pd.Series): A Series of the daily risk free rate.
        period (str): The period to calculate the Excess Volatility for. Can be weekly,
        monthly, quarterly or yearly.

    Returns:
        pd.Series | pd.DataFrame: Excess Volatility values with time as the index, resampled
        to the given period.
    """
    excess_returns = returns.sub(risk_free_rate, axis=0)

    return get_volatility(excess_returns, period)
