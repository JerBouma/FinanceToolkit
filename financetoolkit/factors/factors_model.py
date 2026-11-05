"""Factors Model"""

import numpy as np
import pandas as pd
from scipy import stats

# The labels of the forward return horizons, per period frequency.
HORIZON_UNITS = {
    "daily": "Day",
    "weekly": "Week",
    "monthly": "Month",
    "quarterly": "Quarter",
    "yearly": "Year",
}

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


def get_forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Calculates the return from the close of each period to the close horizon periods later,
    per ticker over its own periods with a price. The return of a period therefore starts
    after the factor of that period is known, so it can be compared with the factor without
    look-ahead.

    Args:
        prices (pd.DataFrame): The (dividend and split adjusted) prices with the periods as
            index and the tickers as columns.
        horizon (int): The number of periods the return covers.

    Returns:
        pd.DataFrame: The forward returns on the index of the prices, NaN for the last
        horizon periods of each ticker.
    """
    forward_returns = {}

    for ticker in prices.columns:
        # Each ticker counts its own periods with a price, so a period another ticker trades
        # in but it does not (a holiday on its exchange) does not cut its return short.
        price = prices[ticker].dropna()
        forward_returns[ticker] = (price.shift(-horizon) / price - 1).reindex(
            prices.index
        )

    return pd.DataFrame(forward_returns, index=prices.index, columns=prices.columns)


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
    both = first.notna() & second.notna()
    first, second = first.where(both), second.where(both)

    first_deviation = first.sub(first.mean(axis=1), axis=0)
    second_deviation = second.sub(second.mean(axis=1), axis=0)

    covariance = (first_deviation * second_deviation).sum(axis=1, min_count=1)
    scale = np.sqrt(
        (first_deviation**2).sum(axis=1, min_count=1)
        * (second_deviation**2).sum(axis=1, min_count=1)
    )

    correlation = covariance / scale.replace(0, np.nan)

    return correlation.where(both.sum(axis=1) >= minimum_observations)


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
    the risk-adjusted IC), the t-statistic of the mean and its two-sided p-value, the share
    of dates with a positive IC and the number of dates.

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

    return pd.DataFrame(
        {
            "Mean IC": mean,
            "IC Standard Deviation": standard_deviation,
            "IC Information Ratio": mean / standard_deviation,
            "t-Statistic": t_statistic,
            "p-Value": p_value,
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
    quantile_numbers = sorted(
        int(value) for value in pd.unique(quantile_assignment.stack().dropna())
    )

    returns_per_date = pd.DataFrame(
        {
            f"Quantile {number}": forward_returns.where(
                quantile_assignment == number
            ).mean(axis=1)
            for number in quantile_numbers
        }
    )

    mean_returns = returns_per_date.mean()

    if quantile_numbers:
        spread = (
            returns_per_date[f"Quantile {quantile_numbers[-1]}"]
            - returns_per_date[f"Quantile {quantile_numbers[0]}"]
        )
        mean_returns["Top Minus Bottom"] = spread.mean()

    return mean_returns


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
