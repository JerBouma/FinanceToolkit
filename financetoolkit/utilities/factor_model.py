"""Factor Model"""

import ast
import inspect
import re
import typing
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.ratios.helpers import map_period_data_to_daily_data
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    check_period,
)
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR

logger = get_logger()

# The fewest tickers a period needs for its values to be ranked against each other.
MINIMUM_TICKERS = 3

# The fewest periods a statistic over time (the shape of the IC, a regression) needs.
MINIMUM_PERIODS = 3

# The labels of the forward return horizons, per period frequency.
HORIZON_UNITS = {
    "daily": "Day",
    "weekly": "Week",
    "monthly": "Month",
    "quarterly": "Quarter",
    "yearly": "Year",
}

# The forward return horizons per period frequency when none are given: about a period, a
# quarter and a year ahead.
DEFAULT_HORIZONS = {
    "daily": [1, 5, 21],
    "weekly": [1, 4, 13],
    "monthly": [1, 3, 12],
    "quarterly": [1, 2, 4],
    "yearly": [1, 2, 3],
}

# The parts of the analysis that detail= returns for a single metric.
DETAILS = (
    "information_coefficient",
    "information_coefficient_by_month",
    "information_coefficient_by_year",
    "group_information_coefficient",
    "quantile_returns",
    "quantile_returns_by_date",
    "quantile_return_statistics",
    "quantile_statistics",
    "quantile_turnover",
    "rank_autocorrelation",
    "factor_weights",
    "factor_returns",
    "cumulative_returns",
    "event_returns",
)

# The ways the tickers of a period are sorted into quantiles (binning=).
BINNINGS = ("quantile", "equal_width", "zero_aware", "group")

# The calendar periods the IC is averaged over, and the evaluation periods that fit in them.
CALENDAR_PERIODS = {
    "information_coefficient_by_month": ("M", "Month", ("daily", "weekly", "monthly")),
    "information_coefficient_by_year": (
        "Y",
        "Year",
        ("daily", "weekly", "monthly", "quarterly", "yearly"),
    ),
}

# The price columns some indicators return next to their own values, never a factor.
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

# The lines an oscillator derives from its main line (the MACD, PPO and PVO), which sort
# before it but are not the indicator itself, so the default is the main line.
DERIVED_LINES = ("Histogram", "Signal Line")

# Pandas period frequencies of the evaluation periods.
PERIOD_FREQUENCIES = {
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def get_horizon_label(horizon: int, period: str) -> str:
    """
    Returns the label of a forward return horizon, e.g. "1 Month" or "3 Months".

    Args:
        horizon (int): The number of periods the forward return covers.
        period (str): The period frequency: daily, weekly, monthly, quarterly or yearly.

    Returns:
        str: The label of the horizon.
    """
    unit = HORIZON_UNITS[period]

    return f"{horizon} {unit}" if horizon == 1 else f"{horizon} {unit}s"


def sample_by_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Takes the last value of each period per column, e.g. the factor or the price on the last
    trading day of each month. A column without a value in a period stays empty (NaN) for that
    period, so a ticker that is not yet listed or no longer listed is left out of it rather
    than carrying an older value.

    Args:
        data (pd.DataFrame): Daily values with the dates as index and the tickers as columns.
        period (str): The period frequency: daily, weekly, monthly, quarterly or yearly.

    Returns:
        pd.DataFrame: The value at the end of each period with a PeriodIndex.
    """
    index = data.index
    dates = (
        index.to_timestamp()
        if isinstance(index, pd.PeriodIndex)
        else pd.DatetimeIndex(index)
    )
    data = data.set_axis(dates.normalize(), axis=0)

    periods = data.index.to_period(PERIOD_FREQUENCIES[period])

    # groupby().last() takes the last non-missing value of each column within the period.
    return data.groupby(periods).last()


def get_forward_returns(
    prices: pd.DataFrame, horizon: int, period: str
) -> pd.DataFrame:
    """
    Calculates the return from the close of each period to the close horizon periods later.
    The return of a period therefore starts after the factor of that period is known, so it
    can be compared with the factor without look-ahead.

    Daily returns count each ticker's own trading days: a day another ticker trades in but
    it does not (a holiday on its exchange, NaN in the prices) is skipped rather than
    cutting its return short. Weekly, monthly, quarterly and yearly returns count calendar
    periods, so a ticker without a price in a period has no return over it, and a return
    over horizon periods never stretches across a longer span.

    Args:
        prices (pd.DataFrame): The (dividend and split adjusted) prices with the periods as
            index and the tickers as columns, NaN where a ticker did not trade.
        horizon (int): The number of periods the return covers. A negative horizon gives
            the change from the period back to that many periods before, P(t - k) / P(t) - 1,
            the part of an event window before the event (see get_event_returns).
        period (str): The period frequency: daily, weekly, monthly, quarterly or yearly.

    Returns:
        pd.DataFrame: The forward returns on the index of the prices, NaN for the last
        horizon periods of each ticker.
    """
    if period != "daily":
        return prices.shift(-horizon) / prices - 1

    forward_returns = {}

    for ticker in prices.columns:
        price = prices[ticker].dropna()
        forward_returns[ticker] = (price.shift(-horizon) / price - 1).reindex(
            prices.index
        )

    return pd.DataFrame(forward_returns, index=prices.index, columns=prices.columns)


def mask_non_trading_days(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the prices on the days a ticker did not trade to NaN. On a date only another
    ticker or the benchmark traded, the historical data carries a ticker's last price
    forward without a Return (see `historical_model.fill_non_trading_days`), which a
    forward return would count as a trading day without a move. A day without a Return
    whose previous day has a price is such a filled day; a ticker's first price has no
    Return by construction and is kept.

    Args:
        prices (pd.DataFrame): The daily prices with the dates as index and tickers as columns.
        returns (pd.DataFrame): The daily returns of the historical data on the same dates.
            Tickers without a column here keep all their prices.

    Returns:
        pd.DataFrame: The prices, NaN on the days the ticker did not trade.
    """
    without_return = returns.reindex(index=prices.index, columns=prices.columns).isna()
    without_return.loc[:, ~prices.columns.isin(returns.columns)] = False

    return prices.mask(without_return & prices.shift().notna())


def get_row_correlation(
    first: pd.DataFrame, second: pd.DataFrame, minimum_observations: int = 3
) -> pd.Series:
    """
    Calculates the Pearson correlation between two DataFrames per row, over the columns
    where both have a value.

    Args:
        first (pd.DataFrame): The first values, with the dates as index.
        second (pd.DataFrame): The second values on the same index and columns.
        minimum_observations (int, optional): The fewest columns with both values for a row to
            get a correlation. Defaults to 3.

    Returns:
        pd.Series: The correlation per row, NaN for a row with too few values or without
        variation.
    """
    # On the numpy arrays, a row at a time for all rows at once: the analyses call this for
    # every factor and horizon, where the pandas operations per row cost more than the sums.
    first_values = first.to_numpy(dtype=float)
    second_values = second.reindex(index=first.index, columns=first.columns).to_numpy(
        dtype=float
    )
    both = ~np.isnan(first_values) & ~np.isnan(second_values)
    observations = both.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        first_values = np.where(both, first_values, 0.0)
        second_values = np.where(both, second_values, 0.0)
        first_deviation = np.where(
            both,
            first_values - first_values.sum(axis=1)[:, None] / observations[:, None],
            0.0,
        )
        second_deviation = np.where(
            both,
            second_values - second_values.sum(axis=1)[:, None] / observations[:, None],
            0.0,
        )

        covariance = (first_deviation * second_deviation).sum(axis=1)
        scale = np.sqrt(
            (first_deviation**2).sum(axis=1) * (second_deviation**2).sum(axis=1)
        )
        correlation = covariance / np.where(scale == 0, np.nan, scale)

    correlation[observations < max(minimum_observations, 1)] = np.nan

    return pd.Series(correlation, index=first.index)


def get_information_coefficient(
    factor: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.Series:
    """
    Calculates the information coefficient (IC) per date: the Spearman rank correlation
    between the factor of the tickers on that date and their forward returns (Grinold and
    Kahn, Active Portfolio Management, 2000). Only tickers with both a factor and a forward
    return on a date are ranked, as in alphalens' factor_information_coefficient.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        forward_returns (pd.DataFrame): The forward returns on the same index and columns.

    Returns:
        pd.Series: The information coefficient per date, NaN for a date with fewer than three
        tickers with both values.
    """
    both = factor.notna() & forward_returns.notna()

    # The Spearman correlation is the Pearson correlation of the ranks.
    factor_ranks = factor.where(both).rank(axis=1)
    return_ranks = forward_returns.where(both).rank(axis=1)

    return get_row_correlation(factor_ranks, return_ranks)


def summarize_information_coefficient(
    information_coefficient: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarizes the information coefficients per horizon as alphalens' information table
    does: the mean, the standard deviation (ddof=1), their ratio (the IC information ratio,
    the risk-adjusted IC), the t-statistic of the mean and its two-sided p-value, the
    skewness and excess (Fisher) kurtosis of the IC (the population moments, as scipy's
    skew and kurtosis), the share of dates with a positive IC and the number of dates.

    Args:
        information_coefficient (pd.DataFrame): The IC per date (index) and horizon (columns).

    Returns:
        pd.DataFrame: The statistics (index) per horizon (columns).
    """
    observations = information_coefficient.count()
    mean = information_coefficient.mean()
    standard_deviation = information_coefficient.std(ddof=1)
    # A constant IC (a perfect factor on every date) has no deviation, and no ratio.
    standard_deviation = standard_deviation.replace(0, np.nan)
    t_statistic = mean / (standard_deviation / np.sqrt(observations))
    p_value = pd.Series(
        2 * stats.t.sf(t_statistic.abs(), observations - 1),
        index=information_coefficient.columns,
    )
    positive_share = (information_coefficient > 0).sum() / observations.replace(
        0, np.nan
    )
    # The population moments of the dates with an IC; too few dates or a constant IC has
    # no shape.
    skewness, kurtosis = (
        pd.Series(
            {
                horizon: (
                    moment(values.dropna())
                    if values.count() >= MINIMUM_PERIODS and values.std() > 0
                    else np.nan
                )
                for horizon, values in information_coefficient.items()
            },
            index=information_coefficient.columns,
            dtype=float,
        )
        for moment in (stats.skew, stats.kurtosis)
    )

    return pd.DataFrame(
        {
            "Mean IC": mean,
            "IC Standard Deviation": standard_deviation,
            "IC Information Ratio": mean / standard_deviation,
            "t-Statistic": t_statistic,
            "P-Value": p_value,
            "IC Skewness": skewness,
            "IC Kurtosis": kurtosis,
            "Positive Share": positive_share,
            "Observations": observations.astype(float),
        }
    ).T


def assign_quantiles(factor: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """
    Sorts the tickers of each date into equally sized quantiles of the factor, 1 for the
    lowest values and quantiles for the highest. Equal factor values are split by the order
    of the tickers, so each quantile holds as close to the same number of tickers as
    possible. A date with fewer tickers with a factor than quantiles gets no quantiles.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        quantiles (int): The number of quantiles.

    Returns:
        pd.DataFrame: The quantile (1 to quantiles) of each ticker per date, NaN where the
        ticker has no factor or the date too few tickers.
    """
    ranks = factor.rank(axis=1, method="first")
    count = factor.notna().sum(axis=1)

    assignment = np.ceil(ranks.mul(quantiles).div(count.replace(0, np.nan), axis=0))

    return assignment.where(count >= quantiles, axis=0)


def get_quantile_returns_by_date(
    quantile_assignment: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates the equally weighted forward return of the tickers in each quantile per date,
    and the top minus bottom spread, the highest quantile's return minus the lowest
    quantile's return on that date (alphalens' mean_return_by_quantile with by_date=True).

    Args:
        quantile_assignment (pd.DataFrame): The quantile of each ticker per date.
        forward_returns (pd.DataFrame): The forward returns on the same index and columns.

    Returns:
        pd.DataFrame: The return of "Quantile 1" to "Quantile n" and the "Top Minus Bottom"
        spread (columns) per date (index).
    """
    quantile_numbers = sorted(
        int(value) for value in pd.unique(quantile_assignment.stack().dropna())
    )

    returns_per_date = pd.DataFrame(
        {
            f"Quantile {number}": forward_returns.where(
                quantile_assignment == number
            ).mean(axis=1)
            for number in quantile_numbers
        },
        index=quantile_assignment.index,
    )

    if quantile_numbers:
        returns_per_date["Top Minus Bottom"] = (
            returns_per_date[f"Quantile {quantile_numbers[-1]}"]
            - returns_per_date[f"Quantile {quantile_numbers[0]}"]
        )

    return returns_per_date


def get_quantile_returns(
    quantile_assignment: pd.DataFrame, forward_returns: pd.DataFrame
) -> pd.Series:
    """
    Calculates the mean forward return of each quantile, as the average over the dates of
    the equally weighted return of the tickers in the quantile, and the top minus bottom
    spread, the average over the dates of the highest quantile's return minus the lowest
    quantile's return (alphalens' mean_return_by_quantile and the long-short spread).

    Args:
        quantile_assignment (pd.DataFrame): The quantile of each ticker per date.
        forward_returns (pd.DataFrame): The forward returns on the same index and columns.

    Returns:
        pd.Series: The mean forward return of "Quantile 1" to "Quantile n" and the
        "Top Minus Bottom" spread.
    """
    return get_quantile_returns_by_date(quantile_assignment, forward_returns).mean()


def get_quantile_turnover(
    quantile_assignment: pd.DataFrame, lag: int = 1
) -> pd.DataFrame:
    """
    Calculates the turnover of each quantile per date: the share of the tickers in the
    quantile that were not in it lag dates before (alphalens' quantile_turnover). A date
    whose earlier date had no quantiles, such as the first lag dates, has no turnover.

    Args:
        quantile_assignment (pd.DataFrame): The quantile of each ticker per date.
        lag (int, optional): The number of dates to compare with. Defaults to 1.

    Returns:
        pd.DataFrame: The turnover per date (index) and quantile (columns).
    """
    quantile_numbers = sorted(
        int(value) for value in pd.unique(quantile_assignment.stack().dropna())
    )
    earlier_assigned = (
        quantile_assignment.notna().any(axis=1).shift(lag, fill_value=False)
    )

    turnover = {}

    for number in quantile_numbers:
        members = quantile_assignment == number
        earlier_members = members.shift(lag, fill_value=False)

        size = members.sum(axis=1)
        stayed = (members & earlier_members).sum(axis=1)

        turnover[f"Quantile {number}"] = (1 - stayed / size.replace(0, np.nan)).where(
            earlier_assigned
        )

    return pd.DataFrame(turnover, index=quantile_assignment.index)


def get_factor_rank_autocorrelation(factor: pd.DataFrame, lag: int = 1) -> pd.Series:
    """
    Calculates the factor rank autocorrelation per date: the correlation between the ranks
    of the tickers' factor on that date and lag dates before, over the tickers with a factor
    on both (alphalens' factor_rank_autocorrelation). A value close to one means a stable
    ranking and so a low turnover of a portfolio built on it.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        lag (int, optional): The number of dates to compare with. Defaults to 1.

    Returns:
        pd.Series: The rank autocorrelation per date, NaN for the first lag dates.
    """
    ranks = factor.rank(axis=1)

    return get_row_correlation(ranks, ranks.shift(lag))


def get_groups(
    group: str | dict | pd.Series | None,
    profile: pd.DataFrame | Callable[[], pd.DataFrame] | None,
    tickers: list[str],
) -> pd.Series | None:
    """
    Returns the group of each ticker, such as its sector or industry, for the group options
    of the factor analysis. A text names a company profile item ("sector", "industry",
    "country", matched regardless of case) and takes it from the company profiles; a
    mapping (dict or Series) gives the group of each ticker directly. A ticker without a
    group is left out of every group calculation.

    Args:
        group (str | dict | pd.Series | None): The profile item or the group per ticker.
        profile (pd.DataFrame | Callable | None): The company profiles with the profile items
            as index and the tickers as columns, or a function that retrieves them. Only
            used when group is a text.
        tickers (list[str]): The tickers of the analysis.

    Returns:
        pd.Series | None: The group per ticker, NaN for a ticker without one, or None when
        no group is given.
    """
    if group is None:
        return None

    if isinstance(group, str):
        profile = profile() if callable(profile) else profile
        if profile is None or profile.empty:
            raise ValueError(
                f"group='{group}' takes the groups from the company profiles, which need a "
                "Financial Modeling Prep API key. Alternatively, pass the group of each "
                "ticker, e.g. group={'AAPL': 'Technology', 'JPM': 'Financial Services'}."
            )
        items = {str(item).lower(): item for item in profile.index}
        if group.lower() not in items:
            raise InvalidArgumentError(
                f"group must be a company profile item such as 'sector' or 'industry', or "
                f"the group of each ticker as a dictionary, not '{group}'."
            )
        values = profile.loc[items[group.lower()]]
    else:
        values = pd.Series(group, dtype=object)

    values = values.reindex(tickers)

    return values.where(values.notna() & (values.astype(str).str.strip() != ""))


def demean_forward_returns(
    forward_returns: pd.DataFrame, groups: pd.Series | None = None
) -> pd.DataFrame:
    """
    Subtracts the mean forward return of the tickers on each date, or with groups the mean
    of the ticker's own group on that date, so the returns are relative to the universe or
    the group (alphalens' demean_forward_returns):

    - R̃(i, t) = R(i, t) − mean of R(j, t) over the tickers j in the group of i on date t

    Args:
        forward_returns (pd.DataFrame): The forward returns with the dates as index and the
            tickers as columns, NaN for a ticker outside the analysis on that date.
        groups (pd.Series | None, optional): The group per ticker. Defaults to None, which
            demeans by the mean of all tickers.

    Returns:
        pd.DataFrame: The demeaned forward returns, NaN for a ticker without a group when
        groups are given.
    """
    if groups is None:
        return forward_returns.sub(forward_returns.mean(axis=1), axis=0)

    group_of = groups.reindex(forward_returns.columns)
    group_means = forward_returns.T.groupby(group_of).transform("mean").T

    return forward_returns - group_means


def get_group_information_coefficient(
    factor: pd.DataFrame, forward_returns: pd.DataFrame, groups: pd.Series
) -> dict[str, pd.Series]:
    """
    Calculates the information coefficient per date within each group: the Spearman rank
    correlation between the factor and the forward returns of the tickers of that group only
    (alphalens' factor_information_coefficient with by_group=True). A group needs at least
    three tickers with both values on a date for an IC on that date.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        forward_returns (pd.DataFrame): The forward returns on the same index and columns.
        groups (pd.Series): The group per ticker.

    Returns:
        dict[str, pd.Series]: The IC per date of each group.
    """
    group_of = groups.reindex(factor.columns)

    return {
        str(name): get_information_coefficient(
            factor.loc[:, group_of == name], forward_returns.loc[:, group_of == name]
        )
        for name in sorted(group_of.dropna().unique(), key=str)
    }


def assign_bins(
    factor: pd.DataFrame,
    quantiles: int,
    binning: str = "quantile",
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Sorts the tickers of each date into bins of the factor, 1 for the lowest values:

    - "quantile": equally sized quantiles (see assign_quantiles).
    - "equal_width": bins of equal width between the lowest and highest factor of the date,
      each closed on the right, so a value on an edge falls in the lower bin. A date whose
      factor does not vary has no bins.
    - "zero_aware": quantiles / 2 equally sized quantiles of the negative values (1 to
      quantiles / 2) and of the zero and positive values (quantiles / 2 + 1 to quantiles),
      so the sign of the factor decides the side, as for a signal centred on zero.
    - "group": equally sized quantiles within each group, so each quantile holds the same
      share of every group.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        quantiles (int): The number of bins.
        binning (str, optional): The way to bin, see above. Defaults to "quantile".
        groups (pd.Series | None, optional): The group per ticker, for "group". Defaults to
            None.

    Returns:
        pd.DataFrame: The bin (1 to quantiles) of each ticker per date, NaN where the ticker
        has no factor or its date (or group) too few tickers.
    """
    if binning == "equal_width":
        lowest = factor.min(axis=1)
        width = (factor.max(axis=1) - lowest) / quantiles
        position = factor.sub(lowest, axis=0).div(width.replace(0, np.nan), axis=0)

        return np.ceil(position).clip(lower=1, upper=quantiles)

    if binning == "zero_aware":
        half = quantiles // 2
        negative = assign_quantiles(factor.where(factor < 0), half)
        positive = assign_quantiles(factor.where(factor >= 0), half) + half

        return negative.fillna(positive)

    if binning == "group" and groups is not None:
        group_of = groups.reindex(factor.columns)
        assignment = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
        for name in group_of.dropna().unique():
            members = factor.columns[group_of == name]
            assignment[members] = assign_quantiles(factor[members], quantiles)

        return assignment

    return assign_quantiles(factor, quantiles)


def summarize_quantile_returns(
    returns_per_date: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """
    Summarizes the quantile returns per date of one horizon: the mean over the dates, its
    standard error (the standard deviation over the dates, ddof=1, divided by the square
    root of the number of dates) and both converted to a single period (alphalens'
    mean_return_by_quantile, rate_of_return and std_conversion):

    - Mean Return Per Period = (1 + Mean Return) ^ (1 / horizon) − 1
    - Standard Error Per Period = Standard Error / √horizon

    The top minus bottom spread takes the difference of the two quantiles' means with the
    standard error √(SE top² + SE bottom²), which treats the two legs as independent
    (compute_mean_returns_spread).

    Args:
        returns_per_date (pd.DataFrame): The return per date of each quantile and the spread
            (see get_quantile_returns_by_date).
        horizon (int): The number of periods the returns cover.

    Returns:
        pd.DataFrame: The Mean Return, Standard Error, Mean Return Per Period and Standard
        Error Per Period (columns) per quantile and the spread (index).
    """
    quantile_columns = [
        column for column in returns_per_date.columns if column != "Top Minus Bottom"
    ]
    quantile_returns = returns_per_date[quantile_columns]

    mean = quantile_returns.mean()
    standard_error = quantile_returns.std(ddof=1) / np.sqrt(quantile_returns.count())
    statistics = pd.DataFrame(
        {
            "Mean Return": mean,
            "Standard Error": standard_error,
            "Mean Return Per Period": (1 + mean) ** (1 / horizon) - 1,
            "Standard Error Per Period": standard_error / np.sqrt(horizon),
        }
    )

    if quantile_columns:
        top, bottom = statistics.iloc[-1], statistics.iloc[0]
        statistics.loc["Top Minus Bottom"] = [
            top["Mean Return"] - bottom["Mean Return"],
            np.sqrt(top["Standard Error"] ** 2 + bottom["Standard Error"] ** 2),
            top["Mean Return Per Period"] - bottom["Mean Return Per Period"],
            np.sqrt(
                top["Standard Error Per Period"] ** 2
                + bottom["Standard Error Per Period"] ** 2
            ),
        ]

    return statistics


def get_quantile_statistics(
    factor: pd.DataFrame, quantile_assignment: pd.DataFrame
) -> pd.DataFrame:
    """
    Describes the factor values in each quantile over all dates and tickers: the lowest,
    highest and mean value, the standard deviation (ddof=1), the number of values and their
    share of all values in a quantile (alphalens' quantile statistics table). Overlapping
    ranges between quantiles show that the factor's level shifts over time, as quantiles
    are formed per date.

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        quantile_assignment (pd.DataFrame): The quantile of each ticker per date.

    Returns:
        pd.DataFrame: The Minimum, Maximum, Mean, Standard Deviation, Count and Count Share
        (columns) per quantile (index).
    """
    values = pd.DataFrame(
        {"Factor": factor.stack(), "Quantile": quantile_assignment.stack()}
    ).dropna()

    statistics = values.groupby("Quantile")["Factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    statistics.columns = ["Minimum", "Maximum", "Mean", "Standard Deviation", "Count"]
    statistics["Count"] = statistics["Count"].astype(float)
    statistics["Count Share"] = statistics["Count"] / statistics["Count"].sum()
    statistics.index = [f"Quantile {int(number)}" for number in statistics.index]

    return statistics


def get_factor_weights(
    factor: pd.DataFrame, groups: pd.Series | None = None
) -> pd.DataFrame:
    """
    Calculates the weights of the factor-weighted long-short portfolio per date: each
    ticker's factor minus the mean factor of the date, divided by the sum of the absolute
    deviations, so the long and short sides each hold half of a gross exposure of one
    (alphalens' factor_weights with demeaned=True). With groups, the factor is demeaned and
    scaled within each group and the weights then rescaled to a gross exposure of one, so
    every group gets the same gross weight and the portfolio is neutral within each group
    (group_adjust=True):

    - w(i, t) = (f(i, t) − mean f(t)) / Σ_j |f(j, t) − mean f(t)|

    Args:
        factor (pd.DataFrame): The factor with the dates as index and the tickers as columns.
        groups (pd.Series | None, optional): The group per ticker. Defaults to None.

    Returns:
        pd.DataFrame: The weight per date and ticker, NaN for a ticker without a factor (or
        group) and for a date whose factor does not vary.
    """
    if groups is None:
        demeaned = factor.sub(factor.mean(axis=1), axis=0)

        return demeaned.div(demeaned.abs().sum(axis=1).replace(0, np.nan), axis=0)

    group_of = groups.reindex(factor.columns)
    demeaned = factor - factor.T.groupby(group_of).transform("mean").T
    gross = demeaned.abs().T.groupby(group_of).transform("sum").T
    weights = demeaned / gross.replace(0, np.nan)

    return weights.div(weights.abs().sum(axis=1).replace(0, np.nan), axis=0)


def get_alpha_beta(
    factor_returns: pd.Series, universe_returns: pd.Series, periods_per_year: float
) -> tuple[float, float]:
    """
    Calculates the alpha and beta of the factor-weighted portfolio against the equally
    weighted universe by ordinary least squares over the dates with both returns (Jensen,
    1968, as in alphalens' factor_alpha_beta). The alpha is annualized by compounding it
    over the number of horizons in a year:

    - R_F(t) = α + β × R_U(t) + ε(t)
    - β = Cov(R_F, R_U) / Var(R_U), α = mean R_F − β × mean R_U
    - Annualized Alpha = (1 + α) ^ periods_per_year − 1

    Args:
        factor_returns (pd.Series): The return of the factor-weighted portfolio per date.
        universe_returns (pd.Series): The mean return of the tickers per date.
        periods_per_year (float): The number of horizons in a year, e.g. 12 / 3 for a three
            month horizon.

    Returns:
        tuple[float, float]: The annualized alpha and the beta, NaN with fewer than three
        dates or a universe return that does not vary.
    """
    both = factor_returns.notna() & universe_returns.notna()
    dependent, independent = factor_returns[both], universe_returns[both]

    if both.sum() < MINIMUM_PERIODS or independent.var(ddof=1) == 0:
        return np.nan, np.nan

    beta = dependent.cov(independent) / independent.var(ddof=1)
    alpha = dependent.mean() - beta * independent.mean()

    return (1 + alpha) ** periods_per_year - 1, beta


def get_cumulative_returns(
    returns_per_date: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """
    Calculates the growth of 1 invested in portfolios rebalanced every horizon periods.
    With a horizon of more than one period the returns of consecutive dates overlap, so the
    capital is split over horizon sub-portfolios started on consecutive dates, each
    rebalanced every horizon periods and compounding its own returns; their mean is the
    value of the whole (as alphalens describes for its multi-period returns). A value is
    known once its holding period has ended, so the return from date t over the horizon
    shows on the date horizon periods later and the series starts at 1. A date without a
    return (too few tickers) is held in cash.

    - V_k(s) = Π (1 + R(t)) over the dates t ≡ k (mod horizon) with t + horizon ≤ s
    - V(s) = mean of V_k(s) over k = 0 to horizon − 1

    Args:
        returns_per_date (pd.DataFrame): The return over the horizon from each date
            (index) per portfolio (columns).
        horizon (int): The number of periods the returns cover.

    Returns:
        pd.DataFrame: The value per date and portfolio, starting at 1.
    """
    sub_portfolios = []

    for start in range(horizon):
        rebalancing = returns_per_date.iloc[start::horizon]
        growth = (1 + rebalancing).cumprod()
        # The value after each holding period, on the date the period ends.
        positions = np.arange(start, len(returns_per_date), horizon) + horizon
        ended = positions < len(returns_per_date)
        value = pd.DataFrame(
            np.nan, index=returns_per_date.index, columns=returns_per_date.columns
        )
        value.iloc[positions[ended]] = growth.iloc[ended].to_numpy()
        sub_portfolios.append(value.ffill().fillna(1.0))

    return sum(sub_portfolios) / horizon


def get_event_returns(
    quantile_assignment: pd.DataFrame,
    prices: pd.DataFrame,
    period: str,
    periods_before: int,
    periods_after: int,
    demeaned: bool = False,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Calculates the average price path of the tickers of each quantile around the dates they
    are in it, an event study (MacKinlay, 1997, "Event Studies in Economics and Finance")
    with each date as the event. The path is rebased to 0 on the date itself, so the values
    before it show how the tickers got into the quantile and the values after it what
    followed. As for the quantile returns, the tickers of a quantile are averaged per date
    first, so the standard deviation is that of the quantile's path across dates:

    - path(i, t, k) = P(i, t + k) / P(i, t) − 1 for k = −periods_before to periods_after
    - demeaned: path(i, t, k) − mean of path(j, t, k) over the tickers j with a quantile on
      date t (or in the group of i)
    - path(q, t, k) = mean of path(i, t, k) over the tickers i in quantile q on date t
    - mean and standard deviation (ddof=1) of path(q, t, k) over the dates t per q and k

    Args:
        quantile_assignment (pd.DataFrame): The quantile of each ticker per date.
        prices (pd.DataFrame): The prices on the same index, which may hold the periods
            before and after the assignment.
        period (str): The period frequency, see get_forward_returns.
        periods_before (int): The number of periods before each date.
        periods_after (int): The number of periods after each date.
        demeaned (bool, optional): Whether to subtract the mean path of the tickers with a
            quantile on the date, or with groups that of the ticker's group. Defaults to
            False.
        groups (pd.Series | None, optional): The group per ticker. Defaults to None.

    Returns:
        pd.DataFrame: The Mean and Standard Deviation per quantile (columns) and offset k
        (index).
    """
    quantile_numbers = sorted(
        int(value) for value in pd.unique(quantile_assignment.stack().dropna())
    )
    in_analysis = quantile_assignment.notna()

    paths = {}
    for offset in range(-periods_before, periods_after + 1):
        path = (
            pd.DataFrame(0.0, index=prices.index, columns=prices.columns).where(
                prices.notna()
            )
            if offset == 0
            else get_forward_returns(prices, offset, period)
        )
        path = path.reindex(quantile_assignment.index).where(in_analysis)
        if demeaned:
            path = demean_forward_returns(path, groups)
        paths[offset] = path

    statistics = {}
    for number in quantile_numbers:
        members = quantile_assignment == number
        values = pd.DataFrame(
            {offset: path.where(members).mean(axis=1) for offset, path in paths.items()}
        )
        statistics[f"Quantile {number}"] = pd.DataFrame(
            {"Mean": values.mean(), "Standard Deviation": values.std(ddof=1)}
        )

    result = pd.concat(statistics, axis=1)
    result.index.name = "Offset"

    return result


def get_horizons(period: str, horizons: list[int] | int | None) -> list[int]:
    """
    Returns the forward return horizons as a list of positive numbers of periods.

    Args:
        period (str): The period frequency, which sets the default horizons.
        horizons (list[int] | int | None): The horizons given, if any.

    Returns:
        list[int]: The horizons.
    """
    check_period(period, DEFAULT_HORIZONS)

    if horizons is None:
        return DEFAULT_HORIZONS[period]

    horizon_list = [horizons] if isinstance(horizons, int) else list(horizons)

    if not horizon_list or any(int(horizon) < 1 for horizon in horizon_list):
        raise ValueError("Horizons must be whole numbers of periods of at least 1.")

    return [int(horizon) for horizon in horizon_list]


def get_metric_label(metric: str | tuple[str, dict]) -> str:
    """
    Returns the name of a metric in the factor analysis: the method name, or for a variant of
    a method the method with its parameters, e.g. "get_moving_average(method='t3')".

    Args:
        metric (str | tuple[str, dict]): The method name, or the method name with the
            parameters of the variant.

    Returns:
        str: The label of the metric.
    """
    if isinstance(metric, str):
        return metric

    name, parameters = metric
    arguments = ", ".join(f"{key}={value!r}" for key, value in parameters.items())

    return f"{name}({arguments})"


def parse_metric_label(label: str) -> str | tuple[str, dict]:
    """
    Returns the metric a label names: a method name as it is, and a method with parameters,
    such as "get_moving_average(method='hull')" or 'get_moving_average(method="hull",
    window=10)', as the method name with its parameters, ("get_moving_average", {"method":
    "hull"}). The parameters are keyword arguments with literal values (text in single or
    double quotes, numbers, booleans or None). A label that is not of this form is returned
    as it is, so that it is reported as an unknown metric.

    Args:
        label (str): The method name or label.

    Returns:
        str | tuple[str, dict]: The method name, or the method name with the parameters.
    """
    try:
        call = ast.parse(label.strip(), mode="eval").body
    except SyntaxError:
        return label

    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.args
        or any(keyword.arg is None for keyword in call.keywords)
    ):
        return label

    try:
        parameters = {
            str(keyword.arg): ast.literal_eval(keyword.value)
            for keyword in call.keywords
        }
    except ValueError:
        return label

    return (call.func.id, parameters) if parameters else call.func.id


def select_metrics(
    metrics: str | list[str], collections: dict[str, list]
) -> list[str | tuple[str, dict]]:
    """
    Returns the metrics to analyse: a collection is replaced by its metrics, and a text may
    name several metrics or collections separated by commas. A metric is a method name or a
    method name with the parameters of one of its variants, such as ("get_moving_average",
    {"method": "t3"}), which can also be named by its label, "get_moving_average(method='t3')"
    (see parse_metric_label).

    Args:
        metrics (str | list[str]): The metric names, labels and collections.
        collections (dict[str, list]): The metrics of each collection of the module.

    Returns:
        list[str | tuple[str, dict]]: The metrics, each once, in the order given.
    """
    if isinstance(metrics, str):
        # The commas between the parameters of a label do not separate metrics.
        metrics = [
            metric.strip()
            for metric in re.split(r",(?![^()]*\))", metrics)
            if metric.strip()
        ]

    names: list[str | tuple[str, dict]] = []
    for metric in metrics:
        for name in collections.get(metric, [parse_metric_label(metric)]):
            if name not in names:
                names.append(name)

    if not names:
        raise InvalidArgumentError("Please name a metric or a collection to analyse.")

    return names


def get_daily_factors(
    module: object,
    names: list[str] | list[str | tuple[str, dict]],
    parameters: dict,
    defaults: dict,
    tickers: list[str],
    column: str | None = None,
    period_end_dates: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculates each named metric of a module and returns it per day and ticker, as the
    factor to analyse.

    Each metric is called unrounded with the parameters it takes: the module's defaults
    (such as show_daily for the ratios) and the parameters given, which must be taken by at least one
    of the metrics so that none is silently ignored. A variant of a metric, a method name with
    parameters, is called with those parameters in place of the ones given. A metric that
    returns no data, such as one missing a statement line item, is left out with a warning in
    the log of its own.

    Args:
        module (object): The module (Ratios, Technicals, Models, Risk or Performance).
        names (list[str] | list[str | tuple[str, dict]]): The metric (method) names, or a
            method name with the parameters of a variant (see select_metrics).
        parameters (dict): The parameters given for the metrics, e.g. window=14.
        defaults (dict): The module's own parameters, passed to the metrics that take them.
        tickers (list[str]): The tickers to compare.
        column (str | None, optional): The output of a metric with several, see
            to_daily_factor. Defaults to None.
        period_end_dates (pd.DataFrame | None, optional): The day from which each statement
            period's figures apply per ticker. Defaults to None.

    Returns:
        dict[str, pd.DataFrame]: The daily factor of each metric that returned data, by the
        label of the metric (see get_metric_label).
    """
    methods = {}
    unused = set(parameters)
    for metric in names:
        name, variant = (metric, {}) if isinstance(metric, str) else metric
        method = getattr(module, name, None)
        if (
            method is None
            or not name.startswith("get_")
            or name in ("get_factor_analysis", "get_indicator_summary")
        ):
            raise InvalidArgumentError(
                f"{name} is not a metric of the {type(module).__name__} module. Please pass "
                "the name of one of its get_ methods or a collection."
            )
        accepted = inspect.signature(method).parameters
        for key, value in variant.items():
            if key not in accepted:
                raise InvalidArgumentError(f"{key} is not a parameter of {name}.")
            # The options of a parameter typed as a Literal (or a Literal or None).
            annotation = accepted[key].annotation
            options = [
                option
                for member in (annotation, *typing.get_args(annotation))
                if typing.get_origin(member) is typing.Literal
                for option in typing.get_args(member)
            ]
            if options and value not in options:
                raise InvalidArgumentError(
                    f"{key}={value!r} is not an option of {name}, which takes "
                    f"{', '.join(map(repr, options))}."
                )
        methods[get_metric_label(metric)] = (method, variant)
        unused -= set(accepted)

    if unused:
        raise InvalidArgumentError(
            f"{', '.join(sorted(unused))} is not a parameter of {', '.join(methods)}."
        )

    # Unrounded values, as rounding ties small values (an Amihud illiquidity of 1e-10 rounds
    # to 0) and so loses the ranking the analysis is about.
    arguments = {"rounding": 15, **defaults, **parameters}

    factors = {}
    for name, (method, variant) in methods.items():
        accepted = inspect.signature(method).parameters
        values = method(
            **{
                key: value
                for key, value in {**arguments, **variant}.items()
                if key in accepted
            }
        )
        if values.empty:
            continue
        factors[name] = to_daily_factor(values, tickers, column, period_end_dates)

    return factors


def to_daily_factor(
    values: pd.DataFrame | pd.Series,
    tickers: list[str],
    column: str | None = None,
    period_end_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Turns the output of a metric into a factor per day (index) and ticker (columns), without
    look-ahead.

    A metric with the tickers as rows and statement periods as columns (a ratio or a model)
    applies from the day each period's figures are known: the fiscal period end, or the
    filing day with point_in_time (period_end_dates). A metric per period and ticker (a risk
    or performance metric per month) applies from the last day of its period, and a daily
    metric (a technical indicator) from its own day.

    A metric with several outputs gives one: column names it, e.g. "Lower Band" of the
    Bollinger Bands or "Altman Z-Score" of the Altman Z-Score model. Without it, an indicator
    gives its first output other than a price, a histogram or a signal line (e.g. "MACD
    Line") and a model its last row, which is its score.

    Args:
        values (pd.DataFrame | pd.Series): The output of the metric.
        tickers (list[str]): The tickers to compare; the Benchmark and Portfolio never are.
        column (str | None, optional): The output to use of a metric with several. Defaults
            to None.
        period_end_dates (pd.DataFrame | None, optional): The day from which each statement
            period's figures apply per ticker. Defaults to None, which uses the end of each
            labelled period.

    Returns:
        pd.DataFrame: The factor per day and ticker.
    """
    if isinstance(values, pd.Series):
        raise InvalidArgumentError(
            "The metric gives one value per ticker for the whole range, which cannot be "
            "followed over time; choose a metric with a value per period."
        )

    if values.columns.nlevels > 1:
        outputs = list(dict.fromkeys(values.columns.get_level_values(0)))
        candidates = [output for output in outputs if output not in PRICE_COLUMNS]
        choice = column or next(
            (output for output in candidates if output not in DERIVED_LINES),
            candidates[0] if candidates else outputs[0],
        )
        if choice not in outputs:
            raise InvalidArgumentError(
                f"column must be one of {', '.join(map(str, outputs))}."
            )
        values = values[choice]
    elif values.index.nlevels > 1:
        outputs = list(dict.fromkeys(values.index.get_level_values(1)))
        choice = column or outputs[-1]
        if choice not in outputs:
            raise InvalidArgumentError(
                f"column must be one of {', '.join(map(str, outputs))}."
            )
        values = values.xs(choice, level=1)

    tickers = [ticker for ticker in tickers if ticker not in ("Benchmark", "Portfolio")]

    if values.index.isin(tickers).any() and not values.columns.isin(tickers).any():
        periods = values.columns
        quarterly = isinstance(periods, pd.PeriodIndex) and periods.freqstr.startswith(
            "Q"
        )
        statement = values.loc[values.index.isin(tickers)]
        start = (
            periods.min().to_timestamp()
            if isinstance(periods, pd.PeriodIndex)
            else pd.Timestamp(periods.min())
        )
        values = map_period_data_to_daily_data(
            period_data=statement,
            daily_dates=pd.date_range(start, pd.Timestamp.today().normalize()),
            quarterly=quarterly,
            period_end_dates=period_end_dates,
        )
    elif isinstance(values.index, pd.PeriodIndex):
        # A value per period is known once its period has ended.
        values = values.set_axis(
            values.index.to_timestamp(how="end").normalize(), axis=0
        )

    values = values.loc[:, values.columns.isin(tickers)]
    values.index = pd.DatetimeIndex(values.index)

    return values.astype(float)


def analyse_factor(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    period: str,
    horizons: list[int],
    quantiles: int,
    periods_apart: int,
    detail: str | None = None,
    binning: str = "quantile",
    demeaned: bool = False,
    groups: pd.Series | None = None,
) -> dict[str, pd.DataFrame | pd.Series] | None:
    """
    Analyses one factor: the information coefficient per period and horizon, the mean
    forward return per quantile, the quantile turnover, the rank autocorrelation and the
    alpha and beta of the factor-weighted portfolio, which the summary uses, or the one
    part detail names. A factor that no period has for at least three tickers can't be
    ranked and gives None.

    Args:
        factor (pd.DataFrame): The daily factor with the dates as index and tickers as columns.
        prices (pd.DataFrame): The daily (dividend and split adjusted) prices of the tickers.
        period (str): The frequency at which the factor is evaluated.
        horizons (list[int]): The number of periods the forward returns cover.
        quantiles (int): The number of quantiles.
        periods_apart (int): The periods between the dates the turnover and the rank
            autocorrelation compare.
        detail (str | None, optional): The part of DETAILS to calculate. Defaults to None,
            which calculates the parts of the summary.
        binning (str, optional): The way the tickers are sorted into quantiles, one of
            BINNINGS (see assign_bins). Defaults to "quantile".
        demeaned (bool, optional): Whether to compare the forward returns with the mean of
            the tickers on each date, or with groups with the mean of their group, which also
            makes the factor-weighted portfolio neutral within each group. Defaults to False.
        groups (pd.Series | None, optional): The group per ticker. Defaults to None.

    Returns:
        dict[str, pd.DataFrame | pd.Series] | None: The parts, keyed as in DETAILS, or None
            when too few tickers have a value.
    """
    prices = prices.loc[:, prices.columns.isin(factor.columns)]
    sampled_prices = sample_by_period(prices, period)
    sampled_factor = sample_by_period(factor[prices.columns], period).reindex(
        sampled_prices.index
    )
    # Periods without any factor (before the start date) only serve as prices.
    sampled_factor = sampled_factor.loc[sampled_factor.notna().any(axis=1)]

    if sampled_factor.notna().sum(axis=1).max() < MINIMUM_TICKERS:
        return None

    # The forward returns of the tickers with a factor; demeaned against those tickers (or
    # their group) where asked, which shifts the quantile returns but not the ranks of the
    # IC unless the mean is taken per group. Demeaned with groups is a group-neutral
    # analysis, so the factor-weighted portfolio is neutral within each group as well.
    forward_returns = {
        horizon: get_forward_returns(sampled_prices, horizon, period)
        .reindex(sampled_factor.index)
        .where(sampled_factor.notna())
        for horizon in horizons
    }
    compared_returns = {
        horizon: (demean_forward_returns(returns, groups) if demeaned else returns)
        for horizon, returns in forward_returns.items()
    }
    portfolio_groups = groups if demeaned else None
    quantile_assignment = assign_bins(sampled_factor, quantiles, binning, groups)

    labels = {horizon: get_horizon_label(horizon, period) for horizon in horizons}

    parts: dict[str, pd.DataFrame | pd.Series] = {
        "quantile_assignment": quantile_assignment
    }

    if detail in (
        None,
        "information_coefficient",
        "information_coefficient_by_month",
        "information_coefficient_by_year",
    ):
        parts["information_coefficient"] = pd.DataFrame(
            {
                labels[horizon]: get_information_coefficient(
                    sampled_factor, compared_returns[horizon]
                )
                for horizon in horizons
            },
            index=sampled_factor.index,
        )

    if detail in CALENDAR_PERIODS:
        frequency, name, fitting_periods = CALENDAR_PERIODS[detail]
        if period not in fitting_periods:
            raise InvalidArgumentError(
                f"{detail} averages the IC per {name.lower()}, which needs a period of a "
                f"{name.lower()} or shorter instead of {period}."
            )
        information_coefficient = parts["information_coefficient"]
        calendar = information_coefficient.index.asfreq(frequency, how="end")
        parts[detail] = (
            information_coefficient.groupby(calendar).mean().dropna(how="all")
        )
        parts[detail].index.name = name

    if detail == "group_information_coefficient" and groups is not None:
        per_group = {
            labels[horizon]: get_group_information_coefficient(
                sampled_factor, compared_returns[horizon], groups
            )
            for horizon in horizons
        }
        summaries = {
            name: summarize_information_coefficient(
                pd.DataFrame(
                    {label: per_group[label][name] for label in per_group},
                    index=sampled_factor.index,
                )
            ).T
            for name in next(iter(per_group.values()))
        }
        result = pd.concat(summaries)
        result.index.names = ["Group", "Horizon"]
        parts[detail] = result

    if detail in (
        None,
        "quantile_returns",
        "quantile_returns_by_date",
        "quantile_return_statistics",
        "cumulative_returns",
    ):
        returns_by_date = {
            labels[horizon]: get_quantile_returns_by_date(
                quantile_assignment, compared_returns[horizon]
            )
            for horizon in horizons
        }
        parts["quantile_returns"] = pd.DataFrame(
            {label: returns.mean() for label, returns in returns_by_date.items()}
        )
        parts["quantile_returns_by_date"] = pd.concat(returns_by_date, axis=1)
        parts["quantile_return_statistics"] = pd.concat(
            {
                labels[horizon]: summarize_quantile_returns(
                    returns_by_date[labels[horizon]], horizon
                )
                for horizon in horizons
            }
        )

    if detail == "quantile_statistics":
        parts[detail] = get_quantile_statistics(sampled_factor, quantile_assignment)

    if detail in (None, "quantile_turnover"):
        parts["quantile_turnover"] = get_quantile_turnover(
            quantile_assignment, periods_apart
        )

    if detail in (None, "rank_autocorrelation"):
        parts["rank_autocorrelation"] = get_factor_rank_autocorrelation(
            sampled_factor, periods_apart
        )

    if detail == "factor_weights":
        parts[detail] = get_factor_weights(sampled_factor, portfolio_groups)

    if detail in (None, "factor_returns", "cumulative_returns"):
        # The weights of the tickers with a forward return over the horizon, so the
        # portfolio stays fully invested when a ticker has no return.
        factor_returns = {
            horizon: (
                get_factor_weights(
                    sampled_factor.where(forward_returns[horizon].notna()),
                    portfolio_groups,
                )
                * forward_returns[horizon]
            ).sum(axis=1, min_count=1)
            for horizon in horizons
        }
        parts["factor_returns"] = pd.DataFrame(
            {labels[horizon]: factor_returns[horizon] for horizon in horizons}
        )
        # The alpha against the equally weighted tickers with a factor, compounded over
        # the number of horizons in a year.
        parts["alpha_beta"] = pd.DataFrame(
            {
                labels[horizon]: get_alpha_beta(
                    factor_returns[horizon],
                    forward_returns[horizon].mean(axis=1),
                    PERIODS_PER_YEAR[period] / horizon,
                )
                for horizon in horizons
            },
            index=["Annualized Alpha", "Beta"],
        ).T

    if detail == "cumulative_returns":
        # The periods after the last factor show the value of the last holding periods.
        calendar = sampled_prices.index[sampled_prices.index >= sampled_factor.index[0]]
        parts[detail] = pd.concat(
            {
                labels[horizon]: get_cumulative_returns(
                    pd.concat(
                        [
                            factor_returns[horizon].rename("Factor"),
                            returns_by_date[labels[horizon]].drop(
                                columns="Top Minus Bottom", errors="ignore"
                            ),
                        ],
                        axis=1,
                    ).reindex(calendar),
                    horizon,
                )
                for horizon in horizons
            },
            axis=1,
        )

    if detail == "event_returns":
        window = max(horizons)
        parts[detail] = get_event_returns(
            quantile_assignment,
            sampled_prices,
            period,
            window,
            window,
            demeaned,
            groups,
        )

    return parts


def get_factor_analysis(
    factors: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    period: str = "monthly",
    horizons: list[int] | int | None = None,
    quantiles: int = 5,
    periods_apart: int = 1,
    detail: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    returns: pd.DataFrame | None = None,
    binning: str = "quantile",
    demeaned: bool = False,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Analyses the predictive power of each factor: a summary row per factor and horizon, or
    with detail one part of the analysis of a single factor.

    The summary has the mean information coefficient (IC), its standard deviation (ddof=1),
    the IC information ratio, the t-statistic of the mean with its p-value, the skewness and
    excess kurtosis of the IC, the share of periods with a positive IC, the number of
    periods, the top minus bottom quantile spread, the annualized alpha and the beta of the
    factor-weighted long-short portfolio against the equally weighted tickers, the mean
    turnover of the top and bottom quantiles and the mean rank autocorrelation. The turnover
    and the rank autocorrelation do not depend on the horizon.

    Args:
        factors (dict[str, pd.DataFrame]): The daily factor (dates × tickers) per name.
        prices (pd.DataFrame): The daily (dividend and split adjusted) prices of the tickers.
        period (str, optional): The frequency at which the factors are evaluated. Defaults
            to "monthly".
        horizons (list[int] | int | None, optional): The number of periods the forward
            returns cover. Defaults to None, which uses DEFAULT_HORIZONS.
        quantiles (int, optional): The number of quantiles. Defaults to 5.
        periods_apart (int, optional): The periods between the dates the turnover and the
            rank autocorrelation compare. Defaults to 1.
        detail (str | None, optional): One of DETAILS to return that part for a single
            factor. Defaults to None, which returns the summary.
        start_date (str | None, optional): The first date a factor is evaluated on. Defaults
            to None.
        end_date (str | None, optional): The last date a factor is evaluated on. Defaults
            to None.
        returns (pd.DataFrame | None, optional): The daily returns of the historical data,
            which mark the days a ticker did not trade (see `mask_non_trading_days`), so
            daily forward returns count trading days only. Defaults to None, which treats
            every price as traded.
        binning (str, optional): The way the tickers are sorted into quantiles, one of
            BINNINGS (see assign_bins). Defaults to "quantile".
        demeaned (bool, optional): Whether to compare the forward returns with the mean of
            the tickers on each date, or with groups with the mean of their group (a
            group-neutral analysis). Defaults to False.
        groups (pd.Series | None, optional): The group per ticker (see get_groups), needed
            for binning="group" and the group IC. Defaults to None.

    Returns:
        pd.DataFrame: The summary per factor and horizon (index), or the detail.
    """
    horizon_list = get_horizons(period, horizons)

    if returns is not None:
        prices = mask_non_trading_days(prices, returns)

    if detail is not None and detail not in DETAILS:
        raise InvalidArgumentError(f"detail must be one of {', '.join(DETAILS)}.")
    check_option("binning", binning, BINNINGS)
    if binning == "zero_aware" and (not quantiles or quantiles % 2):
        raise ValueError(
            "binning='zero_aware' splits the quantiles evenly over the negative and the "
            f"positive values, so it needs an even number of quantiles instead of {quantiles}."
        )
    if groups is None and (
        binning == "group" or detail == "group_information_coefficient"
    ):
        raise ValueError(
            "binning='group' and the group_information_coefficient need the groups: pass "
            "group='sector', group='industry' or the group of each ticker."
        )
    if detail is not None and len(factors) != 1:
        raise ValueError(
            "detail shows one metric at a time; name a single metric instead of "
            f"{', '.join(factors) or 'none'}."
        )
    if not factors:
        raise ValueError("None of the metrics returned data to analyse.")

    analyses = {
        name: analyse_factor(
            factor.loc[start_date:end_date],
            prices,
            period,
            horizon_list,
            quantiles,
            periods_apart,
            detail,
            binning,
            demeaned,
            groups,
        )
        for name, factor in factors.items()
    }

    # A metric most tickers lack (e.g. inventory turnover for companies without inventory)
    # can't be ranked; it is left out of a collection rather than failing the others.
    thin = [name for name, analysis in analyses.items() if analysis is None]
    analyses = {
        name: analysis for name, analysis in analyses.items() if analysis is not None
    }

    # A collection can leave dozens of metrics without a ranking; a few names say enough.
    thin_names = (
        ", ".join(thin)
        if len(thin) <= 3  # noqa: PLR2004
        else f"{len(thin)} metrics ({', '.join(thin[:3])} and others)"
    )
    if not analyses:
        raise ValueError(
            f"{thin_names} {'has' if len(thin) == 1 else 'have'} a value for fewer than "
            f"{MINIMUM_TICKERS} tickers in every period, so there is nothing to rank; twenty "
            "or more tickers give meaningful quantiles."
        )
    if thin:
        logger.warning(
            "%s left out of the factor analysis: fewer than %s tickers have a value in any "
            "period.",
            thin_names,
            MINIMUM_TICKERS,
        )

    # Fewer tickers than quantiles leave every period without quantiles to compare.
    if (
        detail is None or detail.startswith("quantile") or detail == "event_returns"
    ) and all(
        analysis["quantile_assignment"].isna().all(axis=None)
        for analysis in analyses.values()
    ):
        raise ValueError(
            f"{quantiles} quantiles need at least {quantiles} tickers with a value in a "
            "period (per group with binning='group', per sign with binning='zero_aware'); "
            "add tickers or lower quantiles."
        )

    if detail is not None:
        result = next(iter(analyses.values()))[detail]
        if detail == "rank_autocorrelation":
            result = result.to_frame("Rank Autocorrelation")
        if detail in (
            "information_coefficient",
            "quantile_returns_by_date",
            "quantile_turnover",
            "rank_autocorrelation",
            "factor_weights",
            "factor_returns",
        ):
            result = result.dropna(how="all")
        if detail in (
            "information_coefficient",
            "quantile_returns_by_date",
            "quantile_turnover",
            "rank_autocorrelation",
            "factor_weights",
            "factor_returns",
            "cumulative_returns",
        ):
            result.index.name = "Date"
        if detail == "quantile_return_statistics":
            result.index.names = ["Horizon", "Quantile"]
        return result

    summaries = {}
    for name, analysis in analyses.items():
        summary = summarize_information_coefficient(
            analysis["information_coefficient"]
        ).T
        summary["Top Minus Bottom"] = analysis["quantile_returns"].T.get(
            "Top Minus Bottom", np.nan
        )
        summary = summary.join(analysis["alpha_beta"])
        turnover = analysis["quantile_turnover"]
        summary["Quantile Turnover"] = (
            turnover.iloc[:, [0, -1]].stack().mean() if not turnover.empty else np.nan
        )
        summary["Rank Autocorrelation"] = analysis["rank_autocorrelation"].mean()
        summaries[name] = summary

    result = pd.concat(summaries)
    result.index.names = ["Metric", "Horizon"]

    return result
