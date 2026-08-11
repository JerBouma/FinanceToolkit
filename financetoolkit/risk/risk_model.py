"""Risk Model"""

import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    VOLATILITY_WINDOW_TRANSLATION,
)

ALPHA_CONSTRAINT = 0.5

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2


def get_max_drawdown(
    returns: pd.Series | pd.DataFrame,
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Maximum Drawdown (MDD) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level"), e.g. prices, portfolio
        value, or any other level series for which a percentage return is not
        meaningful (interest rates, or any series that can be zero or negative).
        method (str, optional): Either "return" (default), which compounds `returns`
        via `(1 + returns).cumprod()` before measuring the percentage decline from
        the running peak, or "level", which treats `returns` as already being a
        level series and measures the absolute (same units as the input) decline
        from the running peak directly -- well-defined even when the series can be
        zero or negative, where a percentage decline is not. Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame | float: MDD values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index. In "return" mode
        this is a percentage (e.g. -0.5 for a 50% decline); in "level" mode it is in
        the same units as the input.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_max_drawdown(returns.loc[sub_period], method=method)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        max_drawdown = pd.concat(period_data_list, axis=1)

        return max_drawdown.T

    if method == "level":
        return (returns - returns.cummax()).min()

    cum_returns = (1 + returns.fillna(0)).cumprod()  # type: ignore

    return (cum_returns / cum_returns.cummax() - 1).min()


def get_ui(
    returns: pd.Series | pd.DataFrame,
    rolling: int | None = 14,
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Ulcer Index (UI), a measure of downside volatility.

    For more information see:
     - http://www.tangotools.com/ui/ui.htm
     - https://en.wikipedia.org/wiki/Ulcer_index

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        rolling (int | None, optional): The trailing lookback window used as the
        high-water mark reference for each day's drawdown. If you select
        period='monthly' and set rolling to 12 you obtain the rolling 12-month
        Ulcer Index. Pass None for an expanding (since-inception) high-water mark
        instead of a fixed trailing window -- this matches the running-peak
        convention used by `get_max_drawdown`/`get_conditional_drawdown_at_risk`/etc.
        Note a fixed int window is NOT a substitute for "the entire period": pandas'
        `.rolling(window=N)` only produces a value once N observations exist, so
        passing `rolling=len(returns)` degenerates to just the final row's drawdown
        rather than a true full-history calculation -- pass None instead. Defaults
        to 14.
        method (str, optional): Either "return" (default), the textbook Ulcer Index
        computed on percentage drawdowns of the compounded return series (a
        dimensionless, cross-asset-comparable figure), or "level", computed on
        absolute drawdowns of the raw level series directly -- use this when
        `returns` is not a genuine percentage return (e.g. a series that can be zero
        or negative). Note that in "level" mode the result is in squared input units,
        not the dimensionless index the name implies, so it is not comparable across
        assets/series with different scales. Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame: UI values as a float if returns is a pd.Series, as
        a pd.Series indexed by column if returns is a single index pd.DataFrame, or as
        a pd.DataFrame with the sub-periods as index if returns has a multi-index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = returns.loc[sub_period].aggregate(
                get_ui, rolling=rolling, method=method
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        ulcer_index = pd.concat(period_data_list, axis=1)

        return ulcer_index.T

    if isinstance(returns, pd.DataFrame | pd.Series):
        # expanding/rolling/cumprod/mean are column-vectorized on a DataFrame already, no per-column loop needed.
        if method == "level":
            reference_max = (
                returns.expanding().max()
                if rolling is None
                else returns.rolling(window=rolling).max()
            )
            drawdowns = returns - reference_max
        else:
            cumulative_returns = (1 + returns.fillna(0)).cumprod()
            reference_max = (
                cumulative_returns.expanding().max()
                if rolling is None
                else cumulative_returns.rolling(window=rolling).max()
            )
            drawdowns = (cumulative_returns - reference_max) / reference_max

        return np.sqrt((drawdowns**2).mean())

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_skewness(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Computes the skewness of dataset.

    Args:
        returns (pd.Series | pd.Dataframe): A single index dataframe or series

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
        returns (pd.Series | pd.Dataframe): A single index dataframe or series
        fisher (bool, optional): Whether to return Fisher's definition of kurtosis
        (excess kurtosis, i.e. normal distribution equals 0.0) instead of Pearson's
        definition (normal distribution equals 3.0). The two differ by exactly 3,
        since both are computed from the same bias-corrected (sample) estimator that
        `get_skewness` also uses. Defaults to True.

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
        # Pearson is Fisher shifted by 3 by definition, so both branches must come
        # from the same estimator -- the population moment ratio used previously is
        # a different (biased) estimator, leaving the two definitions inconsistent.
        return returns.kurtosis() if fisher else returns.kurtosis() + 3

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_variance(
    returns: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Variance of returns for a given period (weekly, monthly,
    quarterly or yearly) based on daily historical returns.

    The daily Variance is scaled to the given period by multiplying it with the
    number of trading days within that period (e.g. 252 / 52 for weekly).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the Variance for. Can be weekly,
        monthly, quarterly or yearly. Only used to look up the scaling multiplier
        when `groups` is not provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `returns`, to group by instead of deriving calendar periods from
        `returns.index` via `.asfreq()`. Use this when `returns` does not have a
        DatetimeIndex/PeriodIndex (e.g. a plain Series of simulated outcomes).
        Defaults to None, which requires a DatetimeIndex/PeriodIndex on `returns`.

    Returns:
        pd.Series | pd.DataFrame: Variance values with time (or `groups`) as the
        index.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]
    dates = (
        groups
        if groups is not None
        else returns.index.asfreq(PERIOD_TRANSLATION[period])
    )

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
    returns: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
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
        monthly, quarterly or yearly. Only used to look up the scaling multiplier
        when `groups` is not provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `returns`, to group by instead of deriving calendar periods from
        `returns.index` via `.asfreq()`. Use this when `returns` does not have a
        DatetimeIndex/PeriodIndex (e.g. a plain Series of simulated outcomes).
        Defaults to None, which requires a DatetimeIndex/PeriodIndex on `returns`.

    Returns:
        pd.Series | pd.DataFrame: Volatility values with time (or `groups`) as the
        index.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]
    dates = (
        groups
        if groups is not None
        else returns.index.asfreq(PERIOD_TRANSLATION[period])
    )

    return returns.groupby(dates).std() * np.sqrt(volatility_window)


def get_conditional_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame, alpha: float, method: str = "return"
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Drawdown at Risk (CDaR) of returns.

    CDaR extends the concept of Value at Risk and Conditional Value at Risk to the drawdown
    series instead of the return series. The Drawdown at Risk (DaR) is the alpha-quantile of
    the drawdown distribution and CDaR is the average of the drawdowns that are at least as
    severe as the DaR.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        method (str, optional): Either "return" (default), which measures percentage
        drawdowns of the compounded return series, or "level", which measures
        absolute drawdowns of the raw level series directly -- use this when
        `returns` is not a genuine percentage return (e.g. a series that can be zero
        or negative). Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame: CDaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_conditional_drawdown_at_risk(
                returns.loc[sub_period], alpha, method=method
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        conditional_drawdown_at_risk = pd.concat(period_data_list, axis=1)

        return conditional_drawdown_at_risk.T

    if method == "level":
        drawdowns = returns - returns.cummax()
    else:
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
    returns: pd.Series | pd.DataFrame,
    alpha: float,
    window_size: int,
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Conditional Drawdown at Risk (CDaR) of returns.

    Within each rolling window, the cumulative return path is rebuilt from scratch (rebased to 1
    at the start of the window) so that the CDaR reflects only the drawdowns that occurred within
    that window.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        window_size (int): The size of the rolling window.
        method (str, optional): Either "return" (default), which rebuilds the
        cumulative return path within each window, or "level", which uses the raw
        level values within each window directly -- use this when `returns` is not
        a genuine percentage return (e.g. a series that can be zero or negative).
        Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame: Rolling CDaR values with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    def _cdar(window):
        if method == "level":
            drawdowns = window - np.maximum.accumulate(window)
        else:
            cum_returns = np.cumprod(1 + np.nan_to_num(window))
            drawdowns = cum_returns / np.maximum.accumulate(cum_returns) - 1

        drawdown_at_risk = np.percentile(drawdowns, alpha * 100)
        tail_drawdowns = drawdowns[drawdowns <= drawdown_at_risk]

        return tail_drawdowns.mean() if len(tail_drawdowns) else np.nan

    return returns.rolling(window=window_size).apply(_cdar, raw=True)


def _drawdown_trough(
    values: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-column running max, the row index of each column's single worst
    drawdown, and which columns are entirely NaN -- shared vectorized core of
    get_max_drawdown_duration/get_max_drawdown_recovery_time's DataFrame path
    (values shape (n_rows, n_cols)).
    """
    computed = (
        values
        if method == "level"
        else (1 + np.nan_to_num(values, nan=0.0)).cumprod(axis=0)
    )
    # np.maximum.accumulate propagates NaN forever; -inf substitution matches pandas cummax()'s skip-NaN behavior.
    running_max = np.maximum.accumulate(
        np.where(np.isnan(computed), -np.inf, computed), axis=0
    )
    drawdowns = (
        computed - running_max if method == "level" else computed / running_max - 1
    )
    all_nan = np.all(np.isnan(drawdowns), axis=0)
    trough_position = np.argmin(
        np.where(np.isnan(drawdowns), np.inf, drawdowns), axis=0
    )
    return computed, running_max, trough_position, all_nan


def get_max_drawdown_duration(
    returns: pd.Series | pd.DataFrame,
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculates the duration of the Maximum Drawdown, i.e. the number of periods between the
    peak and the lowest point of the largest drawdown.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        method (str, optional): Either "return" (default), which finds the trough via
        the percentage decline of the compounded return series, or "level", which
        finds it via the absolute decline of the raw level series directly -- use
        this when `returns` is not a genuine percentage return (e.g. a series that
        can be zero or negative, where the percentage-decline ratio can pick the
        wrong trough entirely). Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame: Maximum Drawdown Duration values, in number of periods, as
        float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_max_drawdown_duration, method=method
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            max_drawdown_duration = pd.concat(period_data_list, axis=1)

            return max_drawdown_duration.T

        # Vectorized across every column at once -- was a per-column .aggregate() loop, see _drawdown_trough().
        values = returns.to_numpy(dtype=float)
        computed, running_max, trough_position, all_nan = _drawdown_trough(
            values, method
        )
        n_rows, n_cols = computed.shape
        row_index = np.arange(n_rows)[:, None]
        column_index = np.arange(n_cols)
        peak_value = running_max[trough_position, column_index]
        is_peak = (computed == peak_value[None, :]) & (
            row_index <= trough_position[None, :]
        )
        peak_position = np.where(is_peak, row_index, -1).max(axis=0)
        duration = (trough_position - peak_position).astype(float)
        duration[all_nan] = np.nan
        return pd.Series(duration, index=returns.columns)
    if isinstance(returns, pd.Series):
        series = returns if method == "level" else (1 + returns.fillna(0)).cumprod()
        running_max = series.cummax()
        drawdowns = (
            series - running_max if method == "level" else series / running_max - 1
        )

        if drawdowns.isna().all():
            return np.nan

        # nanargmin so a NaN in a 'level' series, never fillna(0)'d, does not crash.
        trough_position = np.nanargmin(drawdowns.to_numpy())
        peak_position = np.flatnonzero(
            series.to_numpy()[: trough_position + 1]
            == running_max.to_numpy()[trough_position]
        )[-1]

        return float(trough_position - peak_position)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_max_drawdown_recovery_time(
    returns: pd.Series | pd.DataFrame,
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Recovery Time of the Maximum Drawdown, i.e. the number of periods it takes
    for the cumulative return to reach a new high after the lowest point of the largest drawdown. If
    the drawdown has not yet been recovered from, this returns NaN.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        method (str, optional): Either "return" (default), which finds the trough via
        the percentage decline of the compounded return series, or "level", which
        finds it via the absolute decline of the raw level series directly -- use
        this when `returns` is not a genuine percentage return (e.g. a series that
        can be zero or negative, where the percentage-decline ratio can pick the
        wrong trough entirely). Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame: Maximum Drawdown Recovery Time values, in number of periods,
        as float if returns is a pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_max_drawdown_recovery_time, method=method
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            max_drawdown_recovery_time = pd.concat(period_data_list, axis=1)

            return max_drawdown_recovery_time.T

        # Vectorized across every column at once -- was a per-column .aggregate() loop, see _drawdown_trough().
        values = returns.to_numpy(dtype=float)
        computed, running_max, trough_position, all_nan = _drawdown_trough(
            values, method
        )
        n_rows, n_cols = computed.shape
        row_index = np.arange(n_rows)[:, None]
        column_index = np.arange(n_cols)
        peak_value = running_max[trough_position, column_index]
        recovered = (computed >= peak_value[None, :]) & (
            row_index >= trough_position[None, :]
        )
        has_recovered = recovered.any(axis=0)
        first_recovery_row = np.where(recovered, row_index, n_rows).min(axis=0)
        recovery_time = (first_recovery_row - trough_position).astype(float)
        recovery_time[~has_recovered] = np.nan
        recovery_time[all_nan] = np.nan
        return pd.Series(recovery_time, index=returns.columns)
    if isinstance(returns, pd.Series):
        series = returns if method == "level" else (1 + returns.fillna(0)).cumprod()
        running_max = series.cummax()
        drawdowns = (
            series - running_max if method == "level" else series / running_max - 1
        )

        if drawdowns.isna().all():
            return np.nan

        # nanargmin so a NaN in a 'level' series, never fillna(0)'d, does not crash.
        trough_position = np.nanargmin(drawdowns.to_numpy())
        peak_value = running_max.to_numpy()[trough_position]

        post_trough = series.to_numpy()[trough_position:]
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
        for a normal distribution) instead of Pearson's (kurtosis = 3.0 for a normal
        distribution). The two differ by exactly 3, as in `get_kurtosis`. Defaults to True.

    Returns:
        pd.Series | pd.DataFrame: Rolling Kurtosis values with time as index.
    """
    rolling_kurtosis = returns.rolling(window=window_size).kurt()

    return rolling_kurtosis if fisher else rolling_kurtosis + 3


def get_downside_deviation(
    returns: pd.Series | pd.DataFrame, minimum_acceptable_return: float = 0.0
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Downside Deviation of returns, i.e. the square root of the average squared
    shortfall below a minimum acceptable return (MAR). This is the second order lower partial
    moment used as the denominator of the Sortino ratio.

    Returns above the MAR contribute a shortfall of zero but still count towards the average,
    so the measure reflects how often shortfalls occur as well as how large they are.

    The formula is as follows:

        Downside Deviation = sqrt((1 / N) * sum(min(Return - MAR, 0)^2))

    Also known as: semi-deviation, downside risk, second order lower partial moment.

    References:
        Sortino, F. A. and Price, L. N. (1994). Performance Measurement in a Downside Risk
        Framework. The Journal of Investing, 3(3), 59-64.

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
        # The second order lower partial moment: shortfalls are squared and averaged over
        # every observation, not only the ones below the threshold, and they are measured
        # from the minimum acceptable return rather than from their own mean. Taking the
        # standard deviation of the shortfalls alone understates the risk materially.
        shortfalls = (returns - minimum_acceptable_return).clip(upper=0)

        return np.sqrt((shortfalls**2).mean())

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_downside_deviation(
    returns: pd.Series | pd.DataFrame,
    window_size: int,
    minimum_acceptable_return: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Downside Deviation of returns, i.e. the square root of the average
    squared shortfall below a minimum acceptable return (MAR), within a rolling window.

    The formula is as follows:

        Downside Deviation = sqrt((1 / N) * sum(min(Return - MAR, 0)^2))

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used as
        the threshold below which returns are considered downside. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Rolling Downside Deviation values with time as index.
    """

    def _downside_deviation(window):
        shortfalls = np.minimum(window - minimum_acceptable_return, 0)

        return np.sqrt(np.mean(shortfalls**2))

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


def get_mean_absolute_deviation(
    returns: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Mean Absolute Deviation (MAD) of returns for a given period (weekly,
    monthly, quarterly or yearly) based on daily historical returns.

    MAD measures the average absolute distance of each return from the mean return. Unlike
    Variance and Volatility, it does not square the deviations, making it less sensitive to
    outliers.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the MAD for. Can be weekly,
        monthly, quarterly or yearly. Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `returns`, to group by instead of deriving calendar periods from
        `returns.index` via `.asfreq()`. Use this when `returns` does not have a
        DatetimeIndex/PeriodIndex (e.g. a plain Series of simulated outcomes).
        Defaults to None, which requires a DatetimeIndex/PeriodIndex on `returns`.

    Returns:
        pd.Series | pd.DataFrame: MAD values with time (or `groups`) as the index.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    dates = (
        groups
        if groups is not None
        else returns.index.asfreq(PERIOD_TRANSLATION[period])
    )

    return returns.groupby(dates).apply(lambda x: (x - x.mean()).abs().mean())


def get_coefficient_of_variation(
    returns: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Coefficient of Variation (CV) of returns for a given period (weekly,
    monthly, quarterly or yearly) based on daily historical returns.

    The Coefficient of Variation is the ratio of the standard deviation to the mean of
    returns, which normalizes dispersion relative to the average return. This makes it
    useful for comparing the relative volatility of assets with different average returns,
    which a raw standard deviation cannot do.

    Also known as: relative standard deviation.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the CV for. Can be weekly,
        monthly, quarterly or yearly. Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `returns`, to group by instead of deriving calendar periods from
        `returns.index` via `.asfreq()`. Use this when `returns` does not have a
        DatetimeIndex/PeriodIndex (e.g. a plain Series of simulated outcomes).
        Defaults to None, which requires a DatetimeIndex/PeriodIndex on `returns`.

    Returns:
        pd.Series | pd.DataFrame: Coefficient of Variation values with time (or
        `groups`) as the index.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    dates = (
        groups
        if groups is not None
        else returns.index.asfreq(PERIOD_TRANSLATION[period])
    )

    grouped = returns.groupby(dates)

    return grouped.std() / grouped.mean()


def get_ewma_volatility(
    returns: pd.Series | pd.DataFrame, lambda_: float = 0.94
) -> pd.Series | pd.DataFrame:
    """
    Calculates the exponentially weighted moving average (EWMA) Volatility of daily
    returns, following the RiskMetrics methodology.

    Unlike a fixed-window rolling Volatility, EWMA Volatility weights recent observations
    more heavily than older ones, so it reacts faster to changes in the underlying
    volatility regime. It is a simpler, more interpretable alternative to a full GARCH fit.

    The formula is as follows:

    - EWMA Variance(t) = lambda * EWMA Variance(t-1) + (1 - lambda) * Return(t-1) ** 2

    Note that, per RiskMetrics' original methodology, this recursion assumes a zero
    mean return (i.e. it is built directly from squared, non-demeaned returns) and
    uses the *lagged* return to forecast the current period's Variance -- it is
    therefore computed here directly from that recursion rather than via a generic
    `pandas.Series.ewm(...).std()`, which would instead subtract each point's own
    exponentially weighted mean and use the *contemporaneous* (not lagged) return,
    neither of which matches the RiskMetrics definition above.

    For more information about the method, see the following paper:

    - J.P. Morgan/Reuters (1996). "RiskMetrics -- Technical Document." 4th ed.

    Also known as: RiskMetrics volatility, exponentially weighted volatility.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        lambda_ (float, optional): The decay factor. Higher values weight the past more
        heavily (slower to react), lower values weight recent returns more heavily
        (faster to react). RiskMetrics uses 0.94 for daily data. Defaults to 0.94.

    Returns:
        pd.Series | pd.DataFrame: Daily EWMA Volatility values with time as the index.
        The first value is NaN, since the recursion has no prior period to seed from.
    """
    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    ewma_variance = (returns**2).shift(1).ewm(alpha=1 - lambda_, adjust=False).mean()

    return np.sqrt(ewma_variance)


def get_autocorrelation(data: pd.Series, lags: int = 10) -> pd.Series:
    """
    Calculate the Autocorrelation Function (ACF) of a series for a range of lags.

    The ACF measures the correlation between a series and a lagged version of itself.
    It is a natural sibling to the AR/MA model-fitting utilities in this module, since
    the ACF is typically the first diagnostic used to decide how many AR or MA terms
    a series needs.

    Args:
        data (pd.Series): A Series of values (e.g. returns or prices) to calculate the
        ACF for.
        lags (int, optional): The number of lags to calculate the ACF for. Defaults to 10.

    Returns:
        pd.Series: The ACF value for each lag from 1 up to and including `lags`, indexed
        by lag number.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Expects pd.Series, no other value.")

    values = data.dropna().to_numpy()
    mean = values.mean()
    variance = np.sum((values - mean) ** 2)

    acf_values = {}
    for lag in range(1, lags + 1):
        covariance = np.sum((values[:-lag] - mean) * (values[lag:] - mean))
        acf_values[lag] = covariance / variance

    return pd.Series(acf_values, name="Autocorrelation")


def get_hill_estimator(
    returns: pd.Series | pd.DataFrame,
    k: int | float = 0.1,
    tail: str = "left",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Hill Estimator of the tail index of returns, over the `k` most
    extreme order statistics.

    Unlike the (finite-sample) skewness and kurtosis in `get_skewness` and
    `get_kurtosis`, the Hill Estimator is a semi-parametric estimate of how heavy the
    tail of the return distribution actually is, under the assumption that the tail
    follows a Pareto-type power law P(X > x) ~ x^(-alpha) as x becomes large. Smaller
    values of the tail index `alpha` indicate a heavier tail (more extreme outliers
    are likely) -- as a rule of thumb, `alpha` < 4 implies the kurtosis is theoretically
    infinite, and `alpha` <= 2 implies the Variance itself is theoretically infinite.

    The estimator sorts the (loss-side, by default) values in descending order and
    averages the log-ratio of the `k` largest values to the (k+1)-th largest value:

    - xi_hill = (1 / k) * SUM_{i=1}^{k} [ln(X_(i)) - ln(X_(k+1))]
    - alpha_hill = 1 / xi_hill

    Where `X_(i)` is the i-th largest (strictly positive) value. The estimator is
    only defined on strictly positive values, since it operates on logs -- for
    `tail="left"` (the default) this module treats the losses (the negated returns)
    as the variable of interest, so only days with a negative return contribute; for
    `tail="right"` the raw (positive) returns are used instead, so only days with a
    positive return contribute.

    The choice of `k` trades off bias against variance: too large a `k` pulls in
    observations from the center of the distribution (biasing `alpha` upward, i.e.
    understating tail heaviness), while too small a `k` leaves too few observations
    for a stable estimate (inflating the Standard Error). The (large-sample) Standard
    Error of `alpha_hill` is `alpha_hill / sqrt(k)`.

    Also known as: Hill tail index estimator, Hill's estimator.

    For more information about the method, see the following paper:

    - Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of a
    Distribution." The Annals of Statistics, 3(5), 1163-1174.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        k (int | float, optional): The number of upper order statistics to use. If a
        float in (0, 1) it is interpreted as the fraction of the strictly positive
        (loss- or gain-side, depending on `tail`) observations to use, rounded down
        to the nearest integer with a minimum of 1. Defaults to 0.1 (the top 10%).
        tail (str, optional): Which tail to estimate, one of "left" (the loss tail,
        i.e. the negated returns) or "right" (the gain tail, i.e. the raw returns).
        Defaults to "left".

    Returns:
        pd.Series | pd.DataFrame: The Hill tail index (alpha), the Hill shape
        parameter (xi, its reciprocal), its Standard Error and the number of order
        statistics `k` actually used, as a pd.Series if returns is a pd.Series,
        otherwise as a pd.DataFrame with one column per asset.

    Raises:
        ValueError: If `tail` is not one of "left" or "right".
    """
    if tail not in ("left", "right"):
        raise ValueError("tail must be 'left' (loss tail) or 'right' (gain tail).")

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []
            valid_periods = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_hill_estimator, k=k, tail=tail
                )

                if not period_data.empty:
                    period_data_list.append(period_data)
                    valid_periods.append(sub_period)

            # Keyed on the sub-period, since every sub-period contributes a whole
            # frame of statistics rather than the single row the other functions
            # in this module return -- concatenating without the keys would drop
            # the period labels entirely.
            return pd.concat(period_data_list, keys=valid_periods, axis=0)

        return pd.DataFrame(
            {
                column: get_hill_estimator(returns[column], k=k, tail=tail)
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        tail_values = -returns.dropna() if tail == "left" else returns.dropna()
        tail_values = tail_values[tail_values > 0].sort_values(ascending=False)

        n = len(tail_values)
        number_of_order_statistics = int(k * n) if isinstance(k, float) else k
        number_of_order_statistics = max(number_of_order_statistics, 1)

        if number_of_order_statistics >= n:
            return pd.Series(
                {
                    "Hill Tail Index": np.nan,
                    "Hill Shape (xi)": np.nan,
                    "Standard Error": np.nan,
                    "Observations Used (k)": number_of_order_statistics,
                }
            )

        order_statistics = tail_values.to_numpy()[:number_of_order_statistics]
        threshold = tail_values.to_numpy()[number_of_order_statistics]

        xi_hill = np.mean(np.log(order_statistics) - np.log(threshold))
        alpha_hill = 1 / xi_hill
        standard_error = alpha_hill / np.sqrt(number_of_order_statistics)

        return pd.Series(
            {
                "Hill Tail Index": alpha_hill,
                "Hill Shape (xi)": xi_hill,
                "Standard Error": standard_error,
                "Observations Used (k)": number_of_order_statistics,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_hurst_exponent(data: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate the Hurst Exponent of a series, a measure of long-term memory that
    indicates whether a series is mean-reverting, trending, or a random walk.

    The Hurst Exponent (H) is interpreted as follows:

    - H < 0.5: the series is mean-reverting (anti-persistent).
    - H = 0.5: the series is a random walk (no memory).
    - H > 0.5: the series is trending (persistent).

    It is estimated here via the generalized Hurst exponent (structure function)
    method: for a self-affine process (e.g. fractional Brownian motion), the standard
    deviation of the lagged differences scales as a power law of the lag,
    Std(X_(t+lag) - X_t) ~ lag^H, so H is recovered directly as the slope of a linear
    regression of the log of that standard deviation against the log of the lag --
    no further rescaling of the slope is needed, since the square root in the standard
    deviation already converts the lag^(2H) scaling of the underlying Variance into
    lag^H.

    Also known as: generalized Hurst exponent, structure-function Hurst estimator.

    For more information about the method, see the following paper:

    - Weron, R. (2002). "Estimating Long-Range Dependence: Finite Sample Properties
    and Confidence Intervals." Physica A, 312(1-2), 285-299.

    Args:
        data (pd.Series): A Series of values (e.g. prices) to calculate the Hurst
        Exponent for.
        max_lag (int, optional): The maximum lag to use when estimating the exponent.
        Defaults to 20.

    Returns:
        float: The estimated Hurst Exponent.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Expects pd.Series, no other value.")

    values = data.dropna().to_numpy()
    lags = range(2, max_lag)

    standard_deviations = [
        np.std(np.subtract(values[lag:], values[:-lag])) for lag in lags
    ]

    poly = np.polyfit(np.log(list(lags)), np.log(standard_deviations), 1)

    return poly[0]
