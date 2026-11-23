"""Risk Model"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from financetoolkit.risk.helpers import get_days_per_period, get_observed_span
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    by_own_trading_days,
    drop_non_finite,
    to_period_index,
    validate_period,
)

ALPHA_CONSTRAINT = 0.5

# Calendar days per year, including leap years, for a length in years.
DAYS_PER_YEAR = 365.25

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# The returns a period needs for a quantile of their distribution.
MINIMUM_QUANTILE_OBSERVATIONS = 2

# A wealth path that comes back to exactly its old high lands within rounding of it rather than
# on it, as it is compounded from the returns (closes of 100, 99, 95 and 100 give 0.9999999999999999),
# so a wealth within this relative distance of its peak is at the peak.  # noqa: E501
PEAK_TOLERANCE = 1e-12


def get_quantile(
    returns: pd.Series | pd.DataFrame, quantile: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate a quantile of returns per column and, for a within period index, per period.
    A period with fewer than two returns gives NaN rather than its single return.

    The quantile is interpolated linearly, as np.percentile does, but skips missing
    returns: a single missing return would otherwise make the quantile missing.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns, optionally
            with a (period, date) index.
        quantile (float): The quantile, between 0 and 1.

    Returns:
        pd.Series | pd.DataFrame | float: The quantile as a float for a Series, per column
        for a DataFrame and per period and column for a within period index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            grouped = returns.groupby(level=0, sort=False)
            # A period with a single return (a period cut by the start or end date) has no distribution to take a quantile of, as the parametric estimates are NaN for it.  # noqa: E501
            result = grouped.quantile(quantile).where(
                grouped.count() >= MINIMUM_QUANTILE_OBSERVATIONS
            )
            result.index.name = None

            return result

        return returns.quantile(quantile).rename(None)
    if isinstance(returns, pd.Series):
        return float(returns.quantile(quantile))

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_wealth_and_peak(
    returns: pd.Series | pd.DataFrame,
    compounded: bool = True,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Compound returns into a wealth path and the running peak a drawdown is measured from.

    The path starts from a wealth of 1 before the first return, so the peak is never
    below 1: a loss on the first day of the window is a drawdown from the starting
    wealth, as in Magdon-Ismail & Atiya (2004), rather than
    the peak it would otherwise set. This matters most within a period, where every
    period's first return is a real one.

    Without compounding the wealth is 1 plus the sum of the returns, so the difference
    between it and its peak is the absolute (uncompounded) drawdown of Riskfolio-Lib and
    skfolio.

    A wealth within a relative 1e-12 of its peak (`PEAK_TOLERANCE`) is set to the peak, so
    a path that returns exactly to its old high counts as back at it despite the rounding
    of the compounding.

    Args:
        returns (pd.Series | pd.DataFrame): The returns, missing returns counting as 0.
        compounded (bool, optional): Whether to compound the returns. Defaults to True.

    Returns:
        tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]: The wealth path and
            its running peak.
    """
    wealth = (
        (1 + returns.fillna(0)).cumprod()
        if compounded
        else 1 + returns.fillna(0).cumsum()
    )

    peak = wealth.cummax().clip(lower=1)

    return wealth.mask(wealth >= peak * (1 - PEAK_TOLERANCE), peak), peak


def apply_to_stacked_windows(
    returns: pd.Series | pd.DataFrame,
    window_size: int,
    function: Callable[[np.ndarray], np.ndarray],
) -> pd.Series | pd.DataFrame:
    """
    Apply a calculation to every full rolling window of returns at once.

    The same result as `returns.rolling(window_size).apply(function, raw=True)`, but the
    function receives all windows of a column as the rows of one array (a view, see
    numpy's sliding_window_view) and returns one value per row, instead of being called
    once per window. As in a rolling apply, a window with a missing return gives NaN and
    so do the first window_size - 1 observations.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        function (Callable[[np.ndarray], np.ndarray]): Takes an array with one window per
            row and returns one value per row.

    Returns:
        pd.Series | pd.DataFrame: The rolling values with time as index.
    """
    values = returns.to_numpy(dtype=float).reshape(len(returns), -1)
    rolling_values = np.full(values.shape, np.nan)

    # Column by column, so only one column's windows are held in memory at a time.
    for column in range(values.shape[1] if len(returns) >= window_size else 0):
        windows = sliding_window_view(values[:, column], window_size)
        complete = ~np.isnan(windows).any(axis=1)

        if complete.any():
            rolling_values[window_size - 1 :, column][complete] = function(
                windows[complete]
            )

    if isinstance(returns, pd.Series):
        return pd.Series(rolling_values[:, 0], index=returns.index, name=returns.name)

    return pd.DataFrame(rolling_values, index=returns.index, columns=returns.columns)


def apply_to_observations(
    returns: pd.Series | pd.DataFrame,
    function: Callable[[np.ndarray], np.ndarray],
    window_size: int | None = None,
    minimum_observations: int = MINIMUM_QUANTILE_OBSERVATIONS,
) -> pd.Series | pd.DataFrame | float:
    """
    Apply a calculation to the observed returns of each column: over the whole series, per
    period for a "within period" (period, date) Multi Index, or over every rolling window.

    The function receives the samples as the rows of a two-dimensional array without missing
    values (a single row for a column or period, one row per window when rolling) and returns
    one value per row, so a measure is written once for all three cases. The missing returns of
    a column (the dates only another ticker traded, the dates before a later listing) are left
    out, so each ticker is measured on its own history; rolling windows run over each ticker's
    own trading days (see `by_own_trading_days`).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with a
            (period, date) index.
        function (Callable[[np.ndarray], np.ndarray]): Takes an array with one sample per row
            and returns one value per row.
        window_size (int | None, optional): The size of the rolling window. Defaults to None,
            which calculates one value per column (or per period and column).
        minimum_observations (int, optional): The returns a column or period needs, fewer give
            NaN. Defaults to 2.

    Returns:
        pd.Series | pd.DataFrame | float: A float for a Series, a value per column for a
        DataFrame, per period and column for a within period index and per date and column
        when rolling.
    """
    if window_size:
        if window_size < minimum_observations:
            raise ValueError(
                f"The rolling window needs at least {minimum_observations} observations."
            )

        return by_own_trading_days(apply_to_stacked_windows)(
            returns, window_size, function
        )
    if isinstance(returns, pd.Series):
        values = returns.dropna().to_numpy(dtype=float)

        if len(values) < minimum_observations:
            return np.nan

        return np.float64(function(values[np.newaxis, :])[0])
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            return pd.DataFrame(
                {
                    period: apply_to_observations(
                        period_returns.droplevel(0),
                        function,
                        minimum_observations=minimum_observations,
                    )
                    for period, period_returns in returns.groupby(level=0, sort=False)
                }
            ).T

        return returns.apply(
            apply_to_observations,
            function=function,
            minimum_observations=minimum_observations,
        ).astype(float)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def mask_without_observations(
    result: pd.Series | float, returns: pd.Series | pd.DataFrame
) -> pd.Series | float:
    """
    Set the result of every column without a single observation to NaN.

    The drawdown measures compound the returns with missing returns counting as 0 (see
    `get_wealth_and_peak`), which is right for a gap within the data but would report a
    drawdown of exactly 0 for a column (or a period) that has no data at all.

    Args:
        result (pd.Series | float): The result per column (DataFrame input) or a float
            (Series input).
        returns (pd.Series | pd.DataFrame): The returns the result was calculated from.

    Returns:
        pd.Series | float: The result, NaN where `returns` holds no observation.
    """
    if isinstance(returns, pd.DataFrame) and isinstance(result, pd.Series):
        return result.where(returns.count() > 0)

    return result if returns.count() > 0 else np.nan


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

    wealth, peak = get_wealth_and_peak(returns)

    return mask_without_observations((wealth / peak - 1).min(), returns)


@by_own_trading_days
def get_ui(
    returns: pd.Series | pd.DataFrame,
    rolling: int | None = 14,
    method: str = "return",
    compounded: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Ulcer Index (UI), a measure of downside volatility.

    The formula is as follows:

    - Drawdown_t = (Wealth_t - Highest High_t) / Highest High_t
    - UI = SQRT(MEAN(Drawdown_t^2))

    Martin & McCann (1989) measure every drawdown from the highest value so far
    (`rolling=None`). A trailing high over the last `rolling` observations (14 by
    default) is the variant StockCharts popularised. Until `rolling` observations are
    available the high is taken over the observations so far, so short sub-periods
    (e.g. a week of daily returns) still give a value. The result is a decimal (0.05 is
    a root mean squared drawdown of 5%), where Martin quotes percentages.

    For more information see:
     - Martin, P.G., & McCann, B.B. (1989). "The Investor's Guide to Fidelity Funds."
       John Wiley & Sons.
     - http://www.tangotools.com/ui/ui.htm
     - https://en.wikipedia.org/wiki/Ulcer_index

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        rolling (int | None, optional): The trailing lookback window used as the
        high-water mark reference for each day's drawdown. If you select
        period='monthly' and set rolling to 12 the drawdowns are measured from the
        trailing 12-month high. Pass None for an expanding (since-inception)
        high-water mark instead of a fixed trailing window -- Martin's definition and
        the running-peak convention used by
        `get_max_drawdown`/`get_conditional_drawdown_at_risk`/etc. Defaults to 14.
        method (str, optional): Either "return" (default), the textbook Ulcer Index
        computed on percentage drawdowns of the compounded return series (a
        dimensionless, cross-asset-comparable figure), or "level", computed on
        absolute drawdowns of the raw level series directly -- use this when
        `returns` is not a genuine percentage return (e.g. a series that can be zero
        or negative). Note that in "level" mode the result is in the units of the
        input (the root mean squared drawdown of the levels), not the dimensionless
        index the name implies, so it is not comparable across assets/series with
        different scales. Defaults to "return".
        compounded (bool, optional): Only for method="return". Whether to compound the
        returns into the wealth path; without compounding the drawdowns are the absolute
        declines of the summed returns from their peak (Riskfolio-Lib's UCI_Abs with
        rolling=None). Defaults to True.

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
        # The same calculation as below, restarted in every sub-period by grouping on it
        # instead of a call per sub-period and column.
        def per_period(data):
            return data.groupby(level=0, sort=False)

        if method == "level":
            reference_max = (
                per_period(returns).cummax()
                if rolling is None
                else per_period(returns)
                .rolling(window=rolling, min_periods=1)
                .max()
                .droplevel(0)
                .reindex(returns.index)
            )
            drawdowns = returns - reference_max
        else:
            cumulative_returns = (
                per_period(1 + returns.fillna(0)).cumprod()
                if compounded
                else 1 + per_period(returns.fillna(0)).cumsum()
            )

            if rolling is None:
                reference_max = per_period(cumulative_returns).cummax().clip(lower=1)
            else:
                reference_max = (
                    per_period(cumulative_returns)
                    .rolling(window=rolling, min_periods=1)
                    .max()
                    .droplevel(0)
                    # A grouped rolling window lists the rows per sub-period; back in the
                    # original order, the positional start mask below lines up with them.
                    .reindex(returns.index)
                )
                # The windows that reach back to the start of each sub-period also hold
                # the starting wealth of 1, see get_wealth_and_peak.
                at_start = (per_period(returns).cumcount() < rolling - 1).to_numpy()
                reference_max.iloc[at_start] = reference_max.iloc[at_start].clip(
                    lower=1
                )

            drawdowns = (cumulative_returns - reference_max) / (
                reference_max if compounded else 1
            )

        # The rows before a ticker's first and after its last return in a period are no observations of it.  # noqa: E501
        drawdowns = drawdowns.where(get_observed_span(returns, by_period=True))
        ulcer_index = np.sqrt(per_period(drawdowns**2).mean())
        ulcer_index = ulcer_index.where(per_period(returns).count() > 0)
        ulcer_index.index.name = None

        return ulcer_index

    if isinstance(returns, pd.DataFrame | pd.Series):
        # expanding/rolling/cumprod/mean are column-vectorized on a DataFrame already, no per-column loop needed.
        if method == "level":
            reference_max = (
                returns.expanding().max()
                if rolling is None
                else returns.rolling(window=rolling, min_periods=1).max()
            )
            drawdowns = returns - reference_max
        else:
            cumulative_returns, peak = get_wealth_and_peak(returns, compounded)

            if rolling is None:
                reference_max = peak
            else:
                # min_periods=1: a window that is not yet full uses the high so far, otherwise every period shorter than the window (a week of daily returns) would be NaN.  # noqa: E501
                reference_max = cumulative_returns.rolling(
                    window=rolling, min_periods=1
                ).max()
                # The windows that reach back to the start also hold the starting wealth of 1, see get_wealth_and_peak.  # noqa: E501
                reference_max.iloc[: max(rolling - 1, 0)] = reference_max.iloc[
                    : max(rolling - 1, 0)
                ].clip(lower=1)

            drawdowns = (cumulative_returns - reference_max) / (
                reference_max if compounded else 1
            )

        # The rows before a ticker's first and after its last return (a later listing, a delisting) are no observations of it, see get_observed_span.  # noqa: E501
        drawdowns = drawdowns.where(get_observed_span(returns))

        return mask_without_observations(np.sqrt((drawdowns**2).mean()), returns)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


@by_own_trading_days
def get_pain_index(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Pain Index of returns, i.e. the mean depth of the drawdowns from the
    running peak over every observation.

    The formula is as follows:

        - Pain Index = (1/N) * sum( |Drawdown_t| )
        - Drawdown_t = Wealth_t / Highest Wealth up to t - 1

    Where N is the number of observations, those at a new peak counting as a drawdown of 0,
    and the wealth starts from 1 before the first return (see `get_wealth_and_peak`). Like
    the Ulcer Index (`get_ui` with rolling=None) it grows with both the depth and the
    duration of the drawdowns, but it averages their absolute size where the Ulcer Index
    takes the root mean square, so deep drawdowns weigh less in it. A "within period"
    (period, date) Multi Index restarts the wealth and its peak in every period. It is
    the denominator of the Pain Ratio
    (`financetoolkit.performance.performance_model.get_pain_ratio`).

    For more information about the method, see the following sources:

    - Bacon, C.R. (2013). "Practical Risk-Adjusted Performance Measurement." John Wiley &
    Sons, chapter 5 (the Pain Index of Zephyr Associates).
    - PerformanceAnalytics (R), `PainIndex`.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns.

    Returns:
        pd.Series | pd.DataFrame | float: Pain Index values as positive decimals, one per
        period with a "within period" Multi Index and otherwise one per column (or one for a
        Series). The rows before a ticker's first and after its last return are no
        observations of it.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):

        def per_period(data):
            return data.groupby(level=0, sort=False)

        wealth = per_period(1 + returns.fillna(0)).cumprod()
        peak = per_period(wealth).cummax().clip(lower=1)
        drawdowns = (wealth / peak - 1).where(
            get_observed_span(returns, by_period=True)
        )
        pain_index = per_period(drawdowns.abs()).mean()
        pain_index.index.name = None

        return pain_index

    wealth, peak = get_wealth_and_peak(returns)
    drawdowns = (wealth / peak - 1).where(get_observed_span(returns))

    return drawdowns.abs().mean()


def get_skewness(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Computes the skewness of dataset.

    Args:
        returns (pd.Series | pd.Dataframe): A single index dataframe or series

    Returns:
        pd.Series | pd.Dataframe: Skewness of the dataset, NaN for a series without
        variation (e.g. a constant price), whose shape is undefined, or with fewer than
        three returns.
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
        # pandas reports 0 for a series that does not vary, which would describe a
        # constant price (a suspended stock, a cash fund) as perfectly symmetric.
        return np.nan if returns.max() == returns.min() else returns.skew()

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
        pd.Series | pd.Dataframe: Kurtosis of the dataset, NaN for a series without
        variation (e.g. a constant price), whose shape is undefined, or with fewer than
        four returns.
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
        # Pearson is Fisher shifted by 3 by definition, so both branches come from the same bias-corrected estimator; the population moment ratio is a different (biased) estimator and would leave the two definitions inconsistent.  # noqa: E501
        # pandas reports 0 (Fisher) for a series that does not vary, which would describe
        # a constant price (a suspended stock, a cash fund) as normally distributed.
        if returns.max() == returns.min():
            return np.nan

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
    number of daily returns within that period: 252 a year for a ticker that trades on
    weekdays (e.g. 252 / 52 for weekly) and 365 a year for one that also trades on
    weekends, such as a cryptocurrency (e.g. 365 / 52 for weekly), see
    `helpers.get_days_per_period`.

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
    validate_period(period)

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(returns, period, groups)
    dates = (
        groups
        if groups is not None
        else to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period])
    )

    return returns.groupby(dates).var() * days_per_period


@by_own_trading_days
def get_rolling_variance(
    returns: pd.Series | pd.DataFrame, period: str, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculates the rolling Variance of returns for a given period (weekly, monthly,
    quarterly or yearly) based on period-frequency historical returns.

    The variance of returns of the given period is already the Variance per period, the
    unit get_variance reports as well, so it is not scaled any further.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns for the given period.
        period (str): The period the returns are given in and to scale the Variance for. Can be
        weekly, monthly, quarterly or yearly.
        window_size (int): The size of the rolling window, in number of periods.

    Returns:
        pd.Series | pd.DataFrame: Rolling Variance values with time as the index.
    """
    validate_period(period)

    return returns.rolling(window=window_size).var()


@by_own_trading_days
def get_rolling_volatility(
    returns: pd.Series | pd.DataFrame, period: str, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculates the rolling Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) based on period-frequency historical returns.

    The standard deviation of returns of the given period is already the Volatility per
    period, the unit get_volatility reports as well, so it is not scaled any further.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns for the given period.
        period (str): The period the returns are given in and to scale the Volatility for. Can be
        weekly, monthly, quarterly or yearly.
        window_size (int): The size of the rolling window, in number of periods.

    Returns:
        pd.Series | pd.DataFrame: Rolling Volatility values with time as the index.
    """
    validate_period(period)

    return returns.rolling(window=window_size).std()


def get_volatility(
    returns: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) based on daily historical returns.

    The daily Volatility is scaled to the given period by multiplying it with the
    square root of the number of daily returns within that period: 252 a year for a
    ticker that trades on weekdays (e.g. SQRT(252 / 52) for weekly) and 365 a year for
    one that also trades on weekends, such as a cryptocurrency (e.g. SQRT(365 / 52) for
    weekly), see `helpers.get_days_per_period`.

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
    validate_period(period)

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(returns, period, groups)
    dates = (
        groups
        if groups is not None
        else to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period])
    )

    return returns.groupby(dates).std() * np.sqrt(days_per_period)


def get_compounded_volatility(
    returns: pd.Series | pd.DataFrame, period: str
) -> pd.Series | pd.DataFrame:
    """
    Calculates the compounded Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) from the daily returns within it: the standard deviation of the
    return over the period when the daily returns compound, rather than the daily standard
    deviation scaled with the square root of time.

    The formula is as follows:

    - Volatility = SQRT((Var(r) + (1 + g)²)^P - (1 + g)^(2P))
    - g = PRODUCT(1 + r)^(1 / n) - 1, the geometric mean of the n daily returns r

    Where Var is the sample variance (ddof=1) and P the number of daily returns in the period
    (see `helpers.get_days_per_period`). It is the exact standard deviation of the product of P
    independent returns with a mean of g and a variance of Var(r), and exceeds the square-root of
    time scaling when the returns are large.

    For more information, see the following source:

    - Referenced against backtesting.py's "Volatility (Ann.) [%]" (Zach Lûster,
      https://github.com/kernc/backtesting.py), which applies it to a year of daily returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the Volatility for. Can be weekly, monthly,
        quarterly or yearly.

    Returns:
        pd.Series | pd.DataFrame: Volatility values with time as the index.
    """
    validate_period(period)

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(returns, period)
    groups = to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period])
    grouped = returns.groupby(groups)

    variance = grouped.var()
    # The geometric mean through the average log return; a period without returns stays NaN.
    growth = np.exp(np.log1p(returns).groupby(groups).mean())

    return np.sqrt(
        (variance + growth**2) ** days_per_period - growth ** (2 * days_per_period)
    )


@by_own_trading_days
def get_conditional_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame,
    alpha: float,
    method: str = "return",
    compounded: bool = True,
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
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        method (str, optional): Either "return" (default), which measures percentage
        drawdowns of the compounded return series, or "level", which measures
        absolute drawdowns of the raw level series directly -- use this when
        `returns` is not a genuine percentage return (e.g. a series that can be zero
        or negative). Defaults to "return".
        compounded (bool, optional): Only for method="return". Whether to compound the
        returns; without compounding the drawdowns are the absolute declines of the summed
        returns from their peak (Riskfolio-Lib's CDaR_Abs). Defaults to True.

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
                returns.loc[sub_period], alpha, method=method, compounded=compounded
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        conditional_drawdown_at_risk = pd.concat(period_data_list, axis=1)

        return conditional_drawdown_at_risk.T

    if method == "level":
        drawdowns = returns - returns.cummax()
    else:
        wealth, peak = get_wealth_and_peak(returns, compounded)
        drawdowns = wealth / peak - 1 if compounded else wealth - peak

    # The rows before a ticker's first and after its last return (a later listing, a delisting) are no observations of it, see get_observed_span.  # noqa: E501
    drawdowns = drawdowns.where(get_observed_span(returns))
    drawdown_at_risk = drawdowns.quantile(alpha)

    return mask_without_observations(
        drawdowns[drawdowns <= drawdown_at_risk].mean(), returns
    )


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
    if not isinstance(returns, pd.DataFrame | pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    right_tail = get_quantile(returns, 1 - alpha)
    left_tail = get_quantile(returns, alpha)

    # A left tail of exactly zero has no meaningful ratio.
    if isinstance(returns, pd.Series):
        return abs(right_tail) / abs(left_tail) if left_tail != 0 else np.nan

    return drop_non_finite(right_tail.abs() / left_tail.abs())


@by_own_trading_days
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

    rolling = returns.rolling(window=window_size)
    right_tail = rolling.quantile(1 - alpha, interpolation="linear")
    left_tail = rolling.quantile(alpha, interpolation="linear")

    return drop_non_finite(right_tail.abs() / left_tail.abs())


@by_own_trading_days
def get_rolling_conditional_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame,
    alpha: float,
    window_size: int,
    method: str = "return",
    compounded: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Conditional Drawdown at Risk (CDaR) of returns.

    Within each rolling window, the cumulative return path is rebuilt from scratch (rebased to 1
    at the start of the window) so that the CDaR reflects only the drawdowns that occurred within
    that window.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int): The size of the rolling window.
        method (str, optional): Either "return" (default), which rebuilds the
        cumulative return path within each window, or "level", which uses the raw
        level values within each window directly -- use this when `returns` is not
        a genuine percentage return (e.g. a series that can be zero or negative).
        Defaults to "return".
        compounded (bool, optional): Only for method="return". Whether to compound the
        returns within each window. Defaults to True.

    Returns:
        pd.Series | pd.DataFrame: Rolling CDaR values with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    def _cdar(windows):
        if method == "level":
            drawdowns = windows - np.maximum.accumulate(windows, axis=1)
        elif not compounded:
            cum_returns = np.cumsum(windows, axis=1)
            # Each window starts from a sum of 0, see get_wealth_and_peak.
            drawdowns = cum_returns - np.maximum(
                np.maximum.accumulate(cum_returns, axis=1), 0
            )
        else:
            cum_returns = np.cumprod(1 + windows, axis=1)
            # Each window starts from a wealth of 1, see get_wealth_and_peak.
            drawdowns = (
                cum_returns / np.maximum(np.maximum.accumulate(cum_returns, axis=1), 1)
                - 1
            )

        drawdown_at_risk = np.percentile(drawdowns, alpha * 100, axis=1, keepdims=True)

        # The value at the percentile is always in the tail, so no window has an empty one.
        return np.nanmean(
            np.where(drawdowns <= drawdown_at_risk, drawdowns, np.nan), axis=1
        )

    return apply_to_stacked_windows(returns, window_size, _cdar)


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

    if method != "level":
        # The wealth starts at 1 before the first return and is at its peak within
        # PEAK_TOLERANCE, see get_wealth_and_peak.
        running_max = np.maximum(running_max, 1.0)
        computed = np.where(
            computed >= running_max * (1 - PEAK_TOLERANCE), running_max, computed
        )
    drawdowns = (
        computed - running_max if method == "level" else computed / running_max - 1
    )
    all_nan = np.all(np.isnan(drawdowns), axis=0)
    trough_position = np.argmin(
        np.where(np.isnan(drawdowns), np.inf, drawdowns), axis=0
    )
    return computed, running_max, trough_position, all_nan


@by_own_trading_days
def get_max_drawdown_duration(
    returns: pd.Series | pd.DataFrame,
    method: str = "return",
) -> pd.Series | pd.DataFrame | float:
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

        # Vectorized across every column at once, see _drawdown_trough().
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
        duration[all_nan | (returns.count().to_numpy() == 0)] = np.nan
        return pd.Series(duration, index=returns.columns)
    if isinstance(returns, pd.Series):
        # Without observations there is no drawdown, rather than one of zero periods.
        if returns.count() == 0:
            return np.nan

        if method == "level":
            series, running_max = returns, returns.cummax()
        else:
            series, running_max = get_wealth_and_peak(returns)
        drawdowns = (
            series - running_max if method == "level" else series / running_max - 1
        )

        if drawdowns.isna().all():
            return np.nan

        # nanargmin so a NaN in a 'level' series, never fillna(0)'d, does not crash.
        trough_position = np.nanargmin(drawdowns.to_numpy())
        peak_positions = np.flatnonzero(
            series.to_numpy()[: trough_position + 1]
            == running_max.to_numpy()[trough_position]
        )
        # No row holds the peak when it is the starting wealth before the first row.
        peak_position = peak_positions[-1] if len(peak_positions) else -1

        return float(trough_position - peak_position)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


@by_own_trading_days
def get_max_drawdown_recovery_time(
    returns: pd.Series | pd.DataFrame,
    method: str = "return",
) -> pd.Series | pd.DataFrame | float:
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

        # Vectorized across every column at once, see _drawdown_trough().
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
        recovery_time[all_nan | (returns.count().to_numpy() == 0)] = np.nan
        return pd.Series(recovery_time, index=returns.columns)
    if isinstance(returns, pd.Series):
        # Without observations there is no drawdown, rather than one of zero periods.
        if returns.count() == 0:
            return np.nan

        if method == "level":
            series, running_max = returns, returns.cummax()
        else:
            series, running_max = get_wealth_and_peak(returns)
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


@by_own_trading_days
def get_drawdown_series(returns: pd.DataFrame, recovery: bool = False) -> pd.DataFrame:
    """
    Calculate the drawdown of every period: the decline of the wealth path from its
    running peak, also known as the underwater curve, or the return needed to get back
    to that peak.

    Each column starts from a wealth of 1 at the row before its first return (the first
    close a return is measured from) or, when its first return is on the first row, before
    the data (see `get_wealth_and_peak`). Rows before that start are NaN, so a ticker that
    listed later keeps its own history; missing returns after it count as 0.

    The formula is as follows:

    - Drawdown(t) = Wealth(t) / Peak(t) - 1
    - Recovery Needed(t) = Peak(t) / Wealth(t) - 1 = -Drawdown(t) / (1 + Drawdown(t))

    Args:
        returns (pd.DataFrame): The returns, one column per asset.
        recovery (bool, optional): Whether to return the gain needed to get back to the
            running peak instead of the drawdown. Defaults to False.

    Returns:
        pd.DataFrame: The drawdown per period as a negative decimal, or with recovery=True
            the gain needed as a positive decimal (0 at a new high), NaN before each
            column's start.
    """
    values = returns.to_numpy(dtype=float)
    has_return = ~np.isnan(values)
    # The row before the first return holds the close that return is measured from.
    start = np.maximum(has_return.argmax(axis=0) - 1, 0).astype(float)
    start[~has_return.any(axis=0)] = np.inf

    wealth, peak = get_wealth_and_peak(returns)
    # Peak / Wealth - 1 is the same as -Drawdown / (1 + Drawdown): a 50% fall needs a 100% gain.
    drawdowns = peak / wealth - 1 if recovery else wealth / peak - 1

    before_start = np.arange(len(returns))[:, None] < start[None, :]

    return drawdowns.mask(before_start)


def get_deepest_since(drawdowns: pd.DataFrame) -> pd.DataFrame:
    """
    Find, for every period below the peak, the last earlier period with a drawdown at
    least as deep: the "deepest since" date of the drawdown, as `get_move_significance`
    gives the "largest since" date of a return.

    The formula is as follows:

    - Deepest Since(t) = the last date s < t with Drawdown(s) <= Drawdown(t), for Drawdown(t) < 0

    Args:
        drawdowns (pd.DataFrame): The drawdowns, one column per asset (see
            `get_drawdown_series`).

    Returns:
        pd.DataFrame: The date as text (YYYY-MM-DD for daily data) per period and asset,
            NaN at a high, before an asset's start and when no earlier period was as deep,
            which makes the drawdown the deepest since the start of the data.
    """
    deepest_since = pd.DataFrame(
        np.nan, index=drawdowns.index, columns=drawdowns.columns, dtype=object
    )

    for column in drawdowns.columns:
        observed = drawdowns[column].dropna()
        values = observed.to_numpy(dtype=float)
        labels = observed.index.astype(str)
        since = np.full(len(values), None, dtype=object)
        # Earlier positions whose drawdowns rise from the bottom of the stack to the top: a
        # position is left out once a later one is at least as deep, as that one is then
        # the later answer for every drawdown it would answer.
        stack: list[int] = []

        for position, value in enumerate(values):
            while stack and values[stack[-1]] > value:
                stack.pop()
            if value < 0 and stack:
                since[position] = labels[stack[-1]]
            stack.append(position)

        deepest_since.loc[observed.index, column] = since

    return deepest_since


@by_own_trading_days
def get_drawdown_table(returns: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """
    List the largest drawdowns of every column with their peak, trough and recovery.

    A drawdown runs from the last row at a high (the peak) until wealth is back at that
    high (the recovery). Its depth is the lowest drawdown in between (the trough), and its
    lengths count periods (rows), so the deepest drawdown matches `get_max_drawdown`,
    `get_max_drawdown_duration` (To Trough) and `get_max_drawdown_recovery_time`
    (To Recovery). The same lengths in calendar days are the differences between the
    dates (the last day of each period), and in years those days divided by 365.25.
    Each drawdown is dated by its peak, valley and recovery, so it starts on the peak
    rather than on the first day below it.

    Args:
        returns (pd.DataFrame): The returns, one column per asset.
        top (int, optional): The number of drawdowns per column, deepest first. Defaults
            to 5.

    Returns:
        pd.DataFrame: One row per drawdown, indexed by column and rank, with the dates of
            the Peak, Trough and Recovery (NaN when not yet recovered), the Drawdown as a
            negative decimal, the gain needed from the trough back to the peak (Recovery
            Needed = -Drawdown / (1 + Drawdown)), in periods the time from peak to trough (To Trough),
            from trough to recovery (To Recovery) and from peak to recovery (Length), the same
            in calendar days (To Trough (Days), To Recovery (Days), Length (Days)) and Length
            (Years), To Recovery / To Trough (Recovery Ratio) and the return from the trough to
            the close of the recovery (Recovery Return, at least the Recovery Needed as that
            close may lie above the peak), both NaN when not yet recovered.
            The Peak, and the lengths from it, are NaN when it is the starting wealth before
            the first row.
    """
    drawdown_series = get_drawdown_series(returns)
    wealth_series = get_wealth_and_peak(returns)[0]
    dates = drawdown_series.index
    # The last day of each period, so the calendar days between two months run from month end
    # to month end.
    days = (
        dates.end_time.normalize()
        if isinstance(dates, pd.PeriodIndex)
        else pd.DatetimeIndex(dates).normalize()
    )
    rows = []

    def calendar_days(start: int | None, end: int | None) -> float:
        if start is None or end is None or start < 0:
            return np.nan
        return float((days[end] - days[start]).days)

    for column in drawdown_series.columns:
        drawdowns = drawdown_series[column].to_numpy()
        wealth = wealth_series[column].to_numpy()
        below_peak = np.nan_to_num(drawdowns, nan=0.0) < 0
        # Edges of each run of rows below the peak.
        changes = np.diff(np.concatenate([[0], below_peak.astype(int), [0]]))
        run_starts = np.flatnonzero(changes == 1)
        run_ends = np.flatnonzero(changes == -1) - 1

        episodes = []

        for run_start, run_end in zip(run_starts, run_ends, strict=True):
            trough = run_start + int(np.argmin(drawdowns[run_start : run_end + 1]))
            peak = run_start - 1
            recovery = run_end + 1 if run_end + 1 < len(drawdowns) else None
            episodes.append(
                {
                    "Peak": dates[peak] if peak >= 0 else np.nan,
                    "Trough": dates[trough],
                    "Recovery": dates[recovery] if recovery is not None else np.nan,
                    "Drawdown": drawdowns[trough],
                    "Recovery Needed": -drawdowns[trough] / (1 + drawdowns[trough]),
                    "To Trough": float(trough - peak),
                    "To Recovery": (
                        float(recovery - trough) if recovery is not None else np.nan
                    ),
                    "Length": (
                        float(recovery - peak) if recovery is not None else np.nan
                    ),
                    "Recovery Ratio": (
                        (recovery - trough) / (trough - peak)
                        if recovery is not None
                        else np.nan
                    ),
                    "Recovery Return": (
                        wealth[recovery] / wealth[trough] - 1
                        if recovery is not None
                        else np.nan
                    ),
                    "To Trough (Days)": calendar_days(peak, trough),
                    "To Recovery (Days)": calendar_days(trough, recovery),
                    "Length (Days)": calendar_days(peak, recovery),
                    "Length (Years)": calendar_days(peak, recovery) / DAYS_PER_YEAR,
                }
            )

        episodes = sorted(episodes, key=lambda episode: episode["Drawdown"])[:top]

        for rank, episode in enumerate(episodes, start=1):
            rows.append({"Ticker": column, "Rank": rank, **episode})

    columns = [
        "Ticker",
        "Rank",
        "Peak",
        "Trough",
        "Recovery",
        "Drawdown",
        "Recovery Needed",
        "To Trough",
        "To Recovery",
        "Length",
        "Recovery Ratio",
        "Recovery Return",
        "To Trough (Days)",
        "To Recovery (Days)",
        "Length (Days)",
        "Length (Years)",
    ]

    return pd.DataFrame(rows, columns=columns).set_index(["Ticker", "Rank"])


@by_own_trading_days
def get_rolling_skewness(
    returns: pd.Series | pd.DataFrame, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Skewness of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Skewness values with time as index, NaN for a
        window without variation, as in `get_skewness`.
    """
    windows = returns.rolling(window=window_size)

    return windows.skew().mask(windows.max() == windows.min())


@by_own_trading_days
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
        pd.Series | pd.DataFrame: Rolling Kurtosis values with time as index, NaN for a
        window without variation, as in `get_kurtosis`.
    """
    windows = returns.rolling(window=window_size)
    rolling_kurtosis = windows.kurt().mask(windows.max() == windows.min())

    return rolling_kurtosis if fisher else rolling_kurtosis + 3


def get_downside_deviation(
    returns: pd.Series | pd.DataFrame, minimum_acceptable_return: float | str = 0.0
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Downside Deviation of returns, i.e. the square root of the average squared
    shortfall below a minimum acceptable return (MAR). This is the second order lower partial
    moment used as the denominator of the Sortino ratio.

    Returns above the MAR contribute a shortfall of zero but still count towards the average,
    so the measure reflects how often shortfalls occur as well as how large they are.

    The formula is as follows:

        Downside Deviation = sqrt((1 / N) * sum(min(Return - MAR, 0)^2))

    With minimum_acceptable_return="mean" it is the semi-deviation below each sample's own
    mean (Markowitz 1959), divided by N - 1 as the sample variance is, since the mean is
    estimated from the same returns:

        Semi-Deviation = sqrt((1 / (N - 1)) * sum(min(Return - Mean, 0)^2))

    Also known as: semi-deviation, downside risk, second order lower partial moment.

    References:
        Sortino, F. A. and Price, L. N. (1994). Performance Measurement in a Downside Risk
        Framework. The Journal of Investing, 3(3), 59-64.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        minimum_acceptable_return (float | str, optional): The minimum acceptable return (MAR) used as
        the threshold below which returns are considered downside, or "mean" for the semi-deviation
        below the mean. Defaults to 0.0.

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
        # The second order lower partial moment: shortfalls are squared and averaged over every observation, not only the ones below the threshold, and they are measured from the minimum acceptable return rather than from their own mean -- taking the standard deviation of the shortfalls alone understates the risk materially.  # noqa: E501
        if minimum_acceptable_return == "mean":
            # Estimating the mean from the same returns costs a degree of freedom, as in the sample variance (Riskfolio-Lib and skfolio divide by N - 1 as well).  # noqa: E501
            shortfalls = (returns - returns.mean()).clip(upper=0)
            observations = returns.count()

            return (
                np.sqrt((shortfalls**2).sum() / (observations - 1))
                if observations > 1
                else np.nan
            )

        shortfalls = (returns - minimum_acceptable_return).clip(upper=0)

        return np.sqrt((shortfalls**2).mean())

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


@by_own_trading_days
def get_rolling_downside_deviation(
    returns: pd.Series | pd.DataFrame,
    window_size: int,
    minimum_acceptable_return: float | str = 0.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Downside Deviation of returns, i.e. the square root of the average
    squared shortfall below a minimum acceptable return (MAR), within a rolling window.

    The formula is as follows:

        Downside Deviation = sqrt((1 / N) * sum(min(Return - MAR, 0)^2))

    With minimum_acceptable_return="mean" it is the semi-deviation below the mean of each
    window, divided by N - 1 (see `get_downside_deviation`).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        minimum_acceptable_return (float | str, optional): The minimum acceptable return (MAR) used as
        the threshold below which returns are considered downside, or "mean" for the semi-deviation
        below the mean. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Rolling Downside Deviation values with time as index.
    """
    if minimum_acceptable_return == "mean":
        return apply_to_stacked_windows(
            returns,
            window_size,
            lambda windows: np.sqrt(
                (np.minimum(windows - windows.mean(axis=1, keepdims=True), 0) ** 2).sum(
                    axis=1
                )
                / (window_size - 1)
            ),
        )

    shortfalls = (returns - minimum_acceptable_return).clip(upper=0)

    return np.sqrt((shortfalls**2).rolling(window=window_size).mean())


def get_partial_moment(
    returns: pd.Series | pd.DataFrame,
    order: float = 1,
    minimum_acceptable_return: float | str = 0.0,
    side: str = "lower",
    root: bool = False,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the lower or upper partial moment of returns around a minimum acceptable
    return (MAR), as in Bawa (1975) and Fishburn (1977).

    The formula is as follows:

        Lower Partial Moment = (1 / N) * sum(max(MAR - Return, 0)^order)
        Upper Partial Moment = (1 / N) * sum(max(Return - MAR, 0)^order)

    Where N counts every return, so the returns on the other side of the MAR count as a
    deviation of zero. An order of 0 counts the returns strictly below (or above) the MAR,
    the downside (upside) frequency, as 0^0 is taken as 0 there. With root=True the moment
    is raised to the power 1 / order, e.g. the Downside Deviation for order 2.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        order (float, optional): The order of the moment, 0 or higher. Defaults to 1.
        minimum_acceptable_return (float | str, optional): The threshold, or "mean" for the
            mean of each sample. Defaults to 0.0.
        side (str, optional): "lower" for the deviations below the MAR, "upper" for those
            above. Defaults to "lower".
        root (bool, optional): Whether to take the order-th root. Defaults to False.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The partial moment, see `apply_to_observations`.
    """
    if side not in ["lower", "upper"]:
        raise ValueError("Side must be lower or upper.")
    if order < 0 or (root and order == 0):
        raise ValueError(
            "The order must be 0 or higher, and higher than 0 to take its root."
        )
    if (
        isinstance(minimum_acceptable_return, str)
        and minimum_acceptable_return != "mean"
    ):
        raise ValueError("The minimum acceptable return must be a number or 'mean'.")

    def partial_moment(samples: np.ndarray) -> np.ndarray:
        target = (
            samples.mean(axis=1, keepdims=True)
            if minimum_acceptable_return == "mean"
            else minimum_acceptable_return
        )
        deviations = target - samples if side == "lower" else samples - target
        # Multiplying by the indicator keeps 0^0 at 0 for the returns on the other side of the MAR, which makes order 0 the frequency of the deviations.  # noqa: E501
        moment = (np.clip(deviations, 0, None) ** order * (deviations > 0)).mean(axis=1)

        return moment ** (1 / order) if root else moment

    return apply_to_observations(returns, partial_moment, window_size)


def get_volatility_skewness(
    returns: pd.Series | pd.DataFrame,
    minimum_acceptable_return: float = 0.0,
    method: str = "volatility",
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Volatility Skewness or Variability Skewness of returns (Bacon 2008): the
    upside variance over the downside variance around a minimum acceptable return (MAR), or
    the square root of that ratio, the upside risk over the downside risk.

    The formula is as follows:

        Volatility Skewness = sum(max(Return - MAR, 0)^2) / sum(max(MAR - Return, 0)^2)
        Variability Skewness = sqrt(Volatility Skewness)

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        minimum_acceptable_return (float, optional): The MAR. Defaults to 0.0.
        method (str, optional): "volatility" or "variability". Defaults to "volatility".
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The ratio, NaN without a return below the MAR.
    """
    if method not in ["volatility", "variability"]:
        raise ValueError("Method must be volatility or variability.")

    def volatility_skewness(samples: np.ndarray) -> np.ndarray:
        deviations = samples - minimum_acceptable_return
        upside = (np.clip(deviations, 0, None) ** 2).sum(axis=1)
        downside = (np.clip(-deviations, 0, None) ** 2).sum(axis=1)
        ratio = np.divide(
            upside, downside, out=np.full(len(samples), np.nan), where=downside > 0
        )

        return ratio if method == "volatility" else np.sqrt(ratio)

    return apply_to_observations(returns, volatility_skewness, window_size)


def get_excess_volatility(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame,
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
        risk_free_rate (pd.Series | pd.DataFrame): The daily risk free rate, a Series
        subtracted from every column or a DataFrame with the rate per column (the rate
        each ticker earns per return of its own calendar).
        period (str): The period to calculate the Excess Volatility for. Can be weekly,
        monthly, quarterly or yearly.

    Returns:
        pd.Series | pd.DataFrame: Excess Volatility values with time as the index, resampled
        to the given period.
    """
    excess_returns = (
        returns - risk_free_rate
        if isinstance(risk_free_rate, pd.DataFrame)
        else returns.sub(risk_free_rate, axis=0)
    )

    return get_volatility(excess_returns, period)


@by_own_trading_days
def get_rolling_excess_volatility(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame,
    period: str,
    window_size: int,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the rolling Excess Volatility of returns for a given period (weekly, monthly,
    quarterly or yearly) based on period-frequency historical returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns for the given period.
        risk_free_rate (pd.Series | pd.DataFrame): The risk free rate for the given period, a
        Series subtracted from every column or a DataFrame with the rate per column (the rate
        each ticker earns over its own dates in each period).
        period (str): The period the returns are given in and to scale the Excess Volatility for.
        Can be weekly, monthly, quarterly or yearly.
        window_size (int): The size of the rolling window, in number of periods.

    Returns:
        pd.Series | pd.DataFrame: Rolling Excess Volatility values with time as index.
    """
    excess_returns = (
        returns - risk_free_rate
        if isinstance(risk_free_rate, pd.DataFrame)
        else returns.sub(risk_free_rate, axis=0)
    )

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
    validate_period(period)

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    dates = (
        groups
        if groups is not None
        else to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period])
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
    validate_period(period)

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    dates = (
        groups
        if groups is not None
        else to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period])
    )

    grouped = returns.groupby(dates)

    return grouped.std() / grouped.mean()


@by_own_trading_days
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


@by_own_trading_days
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
        by lag number, NaN for a lag at or beyond the number of values.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Expects pd.Series, no other value.")

    values = data.dropna().to_numpy()
    mean = values.mean()
    variance = np.sum((values - mean) ** 2)

    acf_values = {}
    for lag in range(1, lags + 1):
        # A lag at or beyond the number of values has no pairs to correlate.
        if lag >= len(values):
            acf_values[lag] = np.nan
            continue
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
        to the nearest integer with a minimum of 1 (0 when there are no such
        observations). Defaults to 0.1 (the top 10%).
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

            # Keyed on the sub-period, since every sub-period contributes a whole frame of statistics rather than the single row the other functions in this module return -- concatenating without the keys would drop the period labels entirely.  # noqa: E501
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
        # At least one order statistic when there is data, and none when there is not (a period before a ticker listed, or without a single loss or gain), so the count reports what was actually used.  # noqa: E501
        number_of_order_statistics = max(number_of_order_statistics, 1) if n else 0

        if number_of_order_statistics < n:
            order_statistics = tail_values.to_numpy()[:number_of_order_statistics]
            threshold = tail_values.to_numpy()[number_of_order_statistics]
            xi_hill = np.mean(np.log(order_statistics) - np.log(threshold))

        # Without more observations than order statistics, or with a tail of equal values
        # (a constant series, whose shape is 0 and index infinite), there is no estimate.
        if number_of_order_statistics >= n or xi_hill <= 0:
            return pd.Series(
                {
                    "Hill Tail Index": np.nan,
                    "Hill Shape (xi)": np.nan,
                    "Standard Error": np.nan,
                    "Observations Used (k)": number_of_order_statistics,
                }
            )

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


@by_own_trading_days
@by_own_trading_days
def get_move_significance(
    returns: pd.Series, window: int = 252, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Calculate how unusual each return is compared with the returns before it: in units
    of the trailing volatility and of the trailing Value at Risk, as a percentile of the
    earlier moves, and the date of the last move at least as large in the same direction.

    The formula is as follows:

    - Volatility Multiple(t) = Return(t) / σ(Return(t - window), ..., Return(t - 1)), with ddof=1
    - VaR Multiple(t) = Return(t) / |VaR(t)|, with VaR(t) the historic VaR: the alpha quantile
      (linear interpolation) of Return(t - window), ..., Return(t - 1)
    - Percentile(t) = (Count(|Return(s)| < |Return(t)|) + 0.5 * Count(|Return(s)| = |Return(t)|))
      / Count(s), over s < t
    - Largest Since(t) = the last date s < t with Return(s) <= Return(t) for a fall, or
      Return(s) >= Return(t) for a rise

    Every measure only uses the returns before the day itself, so a day is never compared
    with the move it is measuring. The volatility and the percentile need `window` earlier
    returns and are NaN before that. The VaR Multiple is NaN when the trailing alpha
    quantile is not a loss (zero or positive), since a move cannot be scaled by a Value at
    Risk that is no risk. An earlier move of the same size counts as half in the percentile
    (the mid-rank), so a series of many equal moves, such as a rate that is unchanged most
    days, puts an unchanged day in the middle of its ties rather than at zero. Largest Since
    is NaN when no earlier return in the series was as large, i.e. the move is the largest
    of the history available, and for a flat day (no direction).

    Args:
        returns (pd.Series): The returns of a single asset.
        window (int, optional): The number of earlier returns the volatility and the
            percentile need at least. The volatility uses exactly the last `window`
            returns; the percentile uses all earlier returns. Defaults to 252.
        alpha (float, optional): The tail probability of the trailing Value at Risk.
            Defaults to 0.05.

    Returns:
        pd.DataFrame: Return, Volatility Multiple, VaR Multiple, Percentile and Largest
        Since (the date as text, YYYY-MM-DD for daily data) per date.
    """
    observed = returns.dropna()
    values = observed.to_numpy(dtype=float)
    labels = observed.index.astype(str)
    count = len(values)

    # Shifted by one so the volatility on a day only covers the returns before it.
    trailing_volatility = (
        observed.rolling(window, min_periods=window).std(ddof=1).shift(1)
    )
    # The historic VaR of the same earlier returns, as var_model.get_var_historic takes it (a return,
    # negative for a loss); its size scales the move, so the sign stays the move's.
    trailing_var = (
        observed.rolling(window, min_periods=window)
        .quantile(alpha, interpolation="linear")
        .shift(1)
    )

    percentile = np.full(count, np.nan)
    # NaN (not None) as the missing marker, so it matches the NaN rows of the reindex below.
    largest_since = np.full(count, np.nan, dtype=object)
    absolute = np.abs(values)

    for position in range(1, count):
        earlier = values[:position]

        # Moves of rates quoted to a few decimals are differences of rounded numbers, so
        # equal moves differ in their last bits (0.001 as 0.000999...994 or 0.001...009):
        # compare with a relative tolerance so they count as ties in every run.
        same_size = np.isclose(
            absolute[:position], absolute[position], rtol=1e-9, atol=0
        )
        same_move = np.isclose(earlier, values[position], rtol=1e-9, atol=0)

        if position >= window:
            # The mid-rank of the move among the earlier ones, as get_level_significance
            # ranks a level: ties count as half, so a run of unchanged days is not at 0.
            percentile[position] = np.mean(
                (absolute[:position] < absolute[position]) & ~same_size
            ) + 0.5 * np.mean(same_size)

        # The same direction, so a fall is compared with falls ("the largest drop since")
        # and a rise with rises; a flat day has no direction and gets no date.
        if values[position] < 0:
            as_large = np.flatnonzero((earlier <= values[position]) | same_move)
        elif values[position] > 0:
            as_large = np.flatnonzero((earlier >= values[position]) | same_move)
        else:
            continue

        if as_large.size:
            largest_since[position] = labels[as_large[-1]]

    significance = drop_non_finite(
        pd.DataFrame(
            {
                "Return": values,
                "Volatility Multiple": values / trailing_volatility.to_numpy(),
                # A quantile of zero or above is no loss to scale the move by: NaN rather than
                # an infinite or sign-flipped multiple.
                "VaR Multiple": values
                / np.abs(trailing_var.where(trailing_var < 0).to_numpy()),
                "Percentile": percentile,
                "Largest Since": largest_since,
            },
            index=observed.index,
        )
    )

    # Back on the full index, so a gap in the history stays a NaN row instead of vanishing.
    return significance.reindex(returns.index)
