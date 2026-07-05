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


def get_rolling_variance(
    returns: pd.Series | pd.DataFrame, period: str, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculates the rolling Variance of returns for a given period (weekly, monthly,
    quarterly or yearly) based on period-frequency historical returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns for the given period.
        period (str): The period the returns are given in and to scale the Variance for. Can be
        weekly, monthly, quarterly or yearly.
        window_size (int): The size of the rolling window, in number of periods.

    Returns:
        pd.Series | pd.DataFrame: Rolling Variance values with time as the index.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]

    return returns.rolling(window=window_size).var() * volatility_window


def get_rolling_volatility(
    returns: pd.Series | pd.DataFrame, period: str, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculates the rolling Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) based on period-frequency historical returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns for the given period.
        period (str): The period the returns are given in and to scale the Volatility for. Can be
        weekly, monthly, quarterly or yearly.
        window_size (int): The size of the rolling window, in number of periods.

    Returns:
        pd.Series | pd.DataFrame: Rolling Volatility values with time as the index.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]

    return returns.rolling(window=window_size).std() * np.sqrt(volatility_window)


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


def get_conditional_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of returns.

    CDaR extends the concept of Value at Risk and Conditional Value at Risk to the drawdown
    series instead of the return series. The Drawdown at Risk (DaR) is the alpha-quantile of
    the drawdown distribution and CDaR is the average of the drawdowns that are at least as
    severe as the DaR.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CDaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_conditional_drawdown_at_risk(
                returns.loc[sub_period], alpha
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        conditional_drawdown_at_risk = pd.concat(period_data_list, axis=1)

        return conditional_drawdown_at_risk.T

    cum_returns = (1 + returns.fillna(0)).cumprod()  # type: ignore
    drawdowns = cum_returns / cum_returns.cummax() - 1

    drawdown_at_risk = drawdowns.quantile(alpha)

    return drawdowns[drawdowns <= drawdown_at_risk].mean()


def get_tail_ratio(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Tail Ratio of returns.

    The Tail Ratio compares the size of the right (gain) tail to the left (loss) tail of the
    return distribution. It is calculated as the absolute value of the (1 - alpha)-th percentile
    of returns divided by the absolute value of the alpha-th percentile of returns. A Tail Ratio
    above 1 indicates that best-case gains outsize worst-case losses.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The percentile used to define each tail (e.g., 0.05 uses the 5th and
        95th percentile).

    Returns:
        pd.Series | pd.DataFrame: Tail Ratio values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_tail_ratio, alpha=alpha
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            tail_ratio = pd.concat(period_data_list, axis=1)

            return tail_ratio.T

        return returns.aggregate(get_tail_ratio, alpha=alpha)
    if isinstance(returns, pd.Series):
        right_tail = np.percentile(returns, (1 - alpha) * 100)
        left_tail = np.percentile(returns, alpha * 100)

        return abs(right_tail) / abs(left_tail)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_tail_ratio(
    returns: pd.Series | pd.DataFrame, alpha: float, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Tail Ratio of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The percentile used to define each tail (e.g., 0.05 uses the 5th and
        95th percentile).
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Tail Ratio values with time as index.
    """

    def _tail_ratio(window):
        right_tail = np.percentile(window, (1 - alpha) * 100)
        left_tail = np.percentile(window, alpha * 100)

        return abs(right_tail) / abs(left_tail)

    return returns.rolling(window=window_size).apply(_tail_ratio, raw=True)


def get_rolling_conditional_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame, alpha: float, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Conditional Drawdown at Risk (CDaR) of returns.

    Within each rolling window, the cumulative return path is rebuilt from scratch (rebased to 1
    at the start of the window) so that the CDaR reflects only the drawdowns that occurred within
    that window.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling CDaR values with time as index.
    """

    def _cdar(window):
        cum_returns = np.cumprod(1 + np.nan_to_num(window))
        drawdowns = cum_returns / np.maximum.accumulate(cum_returns) - 1

        drawdown_at_risk = np.percentile(drawdowns, alpha * 100)
        tail_drawdowns = drawdowns[drawdowns <= drawdown_at_risk]

        return tail_drawdowns.mean() if len(tail_drawdowns) else np.nan

    return returns.rolling(window=window_size).apply(_cdar, raw=True)


def get_max_drawdown_duration(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the duration of the Maximum Drawdown, i.e. the number of periods between the
    peak and the lowest point of the largest drawdown.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: Maximum Drawdown Duration values, in number of periods, as
        float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_max_drawdown_duration
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            max_drawdown_duration = pd.concat(period_data_list, axis=1)

            return max_drawdown_duration.T

        return returns.aggregate(get_max_drawdown_duration)
    if isinstance(returns, pd.Series):
        cum_returns = (1 + returns.fillna(0)).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = cum_returns / running_max - 1

        trough_position = drawdowns.to_numpy().argmin()
        peak_position = np.flatnonzero(
            cum_returns.to_numpy()[: trough_position + 1]
            == running_max.to_numpy()[trough_position]
        )[-1]

        return float(trough_position - peak_position)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_max_drawdown_recovery_time(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Recovery Time of the Maximum Drawdown, i.e. the number of periods it takes
    for the cumulative return to reach a new high after the lowest point of the largest drawdown. If
    the drawdown has not yet been recovered from, this returns NaN.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: Maximum Drawdown Recovery Time values, in number of periods,
        as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_max_drawdown_recovery_time
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            max_drawdown_recovery_time = pd.concat(period_data_list, axis=1)

            return max_drawdown_recovery_time.T

        return returns.aggregate(get_max_drawdown_recovery_time)
    if isinstance(returns, pd.Series):
        cum_returns = (1 + returns.fillna(0)).cumprod()
        running_max = cum_returns.cummax()

        trough_position = (cum_returns / running_max - 1).to_numpy().argmin()
        peak_value = running_max.to_numpy()[trough_position]

        post_trough = cum_returns.to_numpy()[trough_position:]
        recovered = np.flatnonzero(post_trough >= peak_value)

        if recovered.size == 0:
            return np.nan

        return float(recovered[0])

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_skewness(
    returns: pd.Series | pd.DataFrame, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Skewness of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Skewness values with time as index.
    """
    return returns.rolling(window=window_size).skew()


def get_rolling_kurtosis(
    returns: pd.Series | pd.DataFrame, window_size: int, fisher: bool = True
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Kurtosis of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        fisher (bool, optional): Whether to use Fisher's definition of kurtosis (kurtosis = 0.0
        for a normal distribution). Defaults to True.

    Returns:
        pd.Series | pd.DataFrame: Rolling Kurtosis values with time as index.
    """
    if fisher:
        return returns.rolling(window=window_size).kurt()

    def _pearson_kurtosis(window):
        return (((window - window.mean()) / window.std(ddof=0)) ** 4).mean()

    return returns.rolling(window=window_size).apply(_pearson_kurtosis)


def get_downside_deviation(
    returns: pd.Series | pd.DataFrame, minimum_acceptable_return: float = 0.0
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Downside Deviation of returns, i.e. the standard deviation of only the returns
    that fall below a minimum acceptable return (MAR).

    Also known as: semi-deviation, downside risk.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as
        the threshold below which returns are considered downside. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Downside Deviation values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_downside_deviation,
                    minimum_acceptable_return=minimum_acceptable_return,
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            downside_deviation = pd.concat(period_data_list, axis=1)

            return downside_deviation.T

        return returns.aggregate(
            get_downside_deviation,
            minimum_acceptable_return=minimum_acceptable_return,
        )
    if isinstance(returns, pd.Series):
        downside_returns = (
            returns[returns < minimum_acceptable_return] - minimum_acceptable_return
        )

        return downside_returns.std()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_downside_deviation(
    returns: pd.Series | pd.DataFrame,
    window_size: int,
    minimum_acceptable_return: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Downside Deviation of returns, i.e. the standard deviation of only the
    returns that fall below a minimum acceptable return (MAR), within a rolling window.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as
        the threshold below which returns are considered downside. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Rolling Downside Deviation values with time as index.
    """

    def _downside_deviation(window):
        downside_returns = window[window < minimum_acceptable_return]

        return downside_returns.std() if len(downside_returns) > 1 else np.nan

    return returns.rolling(window=window_size).apply(_downside_deviation, raw=True)


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


def get_rolling_excess_volatility(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    period: str,
    window_size: int,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the rolling Excess Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) based on period-frequency historical returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns for the given period.
        risk_free_rate (pd.Series): A Series of the risk free rate for the given period.
        period (str): The period the returns are given in and to scale the Excess Volatility for.
        Can be weekly, monthly, quarterly or yearly.
        window_size (int): The size of the rolling window, in number of periods.

    Returns:
        pd.Series | pd.DataFrame: Rolling Excess Volatility values with time as index.
    """
    excess_returns = returns.sub(risk_free_rate, axis=0)

    return get_rolling_volatility(excess_returns, period, window_size)
