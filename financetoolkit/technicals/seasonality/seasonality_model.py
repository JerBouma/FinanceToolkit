"""Seasonality and Relative Strength Model"""

__docformat__ = "google"

import calendar

import numpy as np
import pandas as pd

from financetoolkit.technicals.helpers import get_observations_per_year
from financetoolkit.utilities.error_model import InvalidArgumentError
from financetoolkit.utilities.statistics_model import PERIODS_PER_YEAR

# The weights of the 3, 6, 9 and 12-month returns in the relative strength rating, so the
# latest quarter weighs most. Investor's Business Daily keeps its own rating's formula to itself; this weighting
# is the common public approximation of it.
RATING_QUARTERS = {63: 0.4, 126: 0.2, 189: 0.2, 252: 0.2}


def get_seasonality(
    returns: pd.Series,
    frequency: str = "month",
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Calculate the seasonality of returns: their average per calendar month (from monthly returns) or per day of the
    week (from daily returns).

    Each comes with the median, the share of positive returns (hit rate), the t-statistic of the average and the
    number of observations. A pattern only holds up with a t-statistic well above 2 over many years, and even then
    it can disappear once it is widely known.

    For more information, see the following paper:

    - Heston, S.L. and Sadka, R. (2008). "Seasonality in the Cross-Section of Stock Returns."
      Journal of Financial Economics, 87(2), 418-445.

    Args:
        returns (pd.Series): Monthly returns for frequency="month", daily returns for
            frequency="weekday", indexed by date.
        frequency (str, optional): Options are: "month" (per calendar month), "weekday" (per day of the week).
            Defaults to "month".
        end_date (str | pd.Timestamp | None, optional): The date the returns run up to. A
            monthly return is measured from one month's last close to the next, so with an
            end date before the last business day of its month the final return covers only
            part of that month; it is then left out rather than averaged as a full month.
            The last business day ignores exchange holidays, so an end date on the last
            trading day before a month-end holiday (such as Good Friday) also leaves that
            month out; pass the calendar month end to keep it. Defaults to None, which takes
            every return as complete.

    Returns:
        pd.DataFrame: The Average Return, Median Return, Hit Rate, t-Statistic and
        Observations per month or weekday.
    """
    values = returns.dropna()
    index = values.index
    dates = (
        index.to_timestamp()
        if isinstance(index, pd.PeriodIndex)
        else pd.DatetimeIndex(index)
    )

    if frequency == "month":
        if end_date is not None and len(values):
            month_start = pd.Period(dates[-1], freq="M").start_time
            last_business_day = pd.offsets.BMonthEnd().rollforward(month_start)
            if pd.Timestamp(end_date).normalize() < last_business_day:
                values, dates = values.iloc[:-1], dates[:-1]
        groups = dates.month
        labels = {month: calendar.month_name[month] for month in range(1, 13)}
    elif frequency == "weekday":
        groups = dates.weekday
        labels = {day: calendar.day_name[day] for day in range(7)}
    else:
        raise InvalidArgumentError(
            f"The frequency must be 'month' or 'weekday', not {frequency!r}."
        )

    groups = np.asarray(groups)
    grouped = values.groupby(groups)
    count = grouped.count()
    mean = grouped.mean()
    deviation = grouped.std(ddof=1)
    statistics = pd.DataFrame(
        {
            "Average Return": mean,
            "Median Return": grouped.median(),
            "Hit Rate": (values > 0).groupby(groups).mean(),
            "t-Statistic": mean / (deviation / np.sqrt(count)),
            "Observations": count,
        }
    )
    statistics.index = pd.Index(
        [labels[group] for group in statistics.index],
        name="Month" if frequency == "month" else "Weekday",
    )
    return statistics


def get_relative_strength(
    prices: pd.Series | pd.DataFrame, benchmark_prices: pd.Series, window: int = 252
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Calculate the relative strength of every ticker against the Benchmark and against the other tickers: which
    tickers outperform and by how much.

    Weinstein reads it from weekly charts against a 52-week average, a year.

    For more information, see the following book:

    - Weinstein, S. (1988). "Stan Weinstein's Secrets for Profiting in Bull and Bear
      Markets." Dow Jones-Irwin.

    Args:
        prices (pd.Series | pd.DataFrame): The prices, a column each.
        benchmark_prices (pd.Series): The benchmark's prices.
        window (int, optional): The moving average window of the Mansfield relative strength,
            in periods of the prices. Defaults to 252 (a year of daily prices).

    Returns:
        tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]: The price relative and the
        Mansfield relative strength, in the shape of the prices.
    """
    relative = prices.div(benchmark_prices, axis=0)
    # The first value of each column, as the backfilled first row holds it.
    relative = relative / relative.bfill().iloc[0]
    mansfield = relative / relative.rolling(window).mean() - 1
    return relative, mansfield


def get_period_relative_strength(
    prices: pd.Series,
    benchmark_prices: pd.Series,
    period: str,
    window: int | None = None,
) -> pd.DataFrame:
    """
    Calculate the relative strength and the Mansfield relative strength of one ticker
    against the Benchmark, see get_relative_strength, over a year of periods unless another
    window is given.

    Args:
        prices (pd.Series): The prices of the ticker.
        benchmark_prices (pd.Series): The prices of the Benchmark.
        period (str): The period of the prices, such as "daily" or "weekly".
        window (int | None, optional): The window of the Mansfield relative strength.
            Defaults to None, a year of periods.

    Returns:
        pd.DataFrame: The Relative Strength and Mansfield Relative Strength per date.
    """
    # A year of periods whatever the period: 252 periods would be almost five years of weekly and 21 years of monthly prices, leaving most histories without a value. A ticker that trades on weekends has 365 daily periods a year.  # noqa: E501
    relative, mansfield = get_relative_strength(
        prices,
        benchmark_prices.reindex(prices.index),
        (
            window
            if window is not None
            else (
                get_observations_per_year(prices)
                if period == "daily"
                else PERIODS_PER_YEAR[period]
            )
        ),
    )
    return pd.concat(
        {
            "Relative Strength": relative,
            "Mansfield Relative Strength": mansfield,
        },
        axis=1,
    )


def get_relative_strength_score(
    prices: pd.Series | pd.DataFrame, observations_per_year: float = 252
) -> pd.Series | pd.DataFrame:
    """
    Calculate the relative strength score of every asset: its performance over the last year, the latest quarter
    weighing most, which `get_relative_strength_rating` ranks.

    The formula is as follows:

    - Score = 0.4 * Return over 3 Months + 0.2 * Return over 6 Months + 0.2 * Return over
      9 Months + 0.2 * Return over 12 Months

    Where a quarter is 63 observations of an asset that trades on an exchange's trading
    days (`observations_per_year` of 252), and 91 of one that trades every calendar day
    (365), so it covers three months in both cases.

    For more information, see the following book:

    - O'Neil, W.J. (1988). "How to Make Money in Stocks." McGraw-Hill.

    Args:
        prices (pd.Series | pd.DataFrame): The daily prices, a column each.
        observations_per_year (float, optional): The daily observations in a year of the
            prices. Defaults to 252.

    Returns:
        pd.Series | pd.DataFrame: The score per date, in the shape of the prices.
    """
    return sum(
        weight * (prices / prices.shift(round(days * observations_per_year / 252)) - 1)
        for days, weight in RATING_QUARTERS.items()
    )


def get_relative_strength_rating(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the relative strength rating of every asset: its performance over the last year, ranked against the
    other assets from 1 (weakest) to 99 (strongest).

    The latest quarter weighs most: the 3, 6, 9 and 12-month returns are weighted 0.4, 0.2, 0.2 and 0.2 (see
    `get_relative_strength_score`). The rating is only meaningful across many assets, such as the constituents of an
    index.

    The rating follows the one Investor's Business Daily publishes, whose exact formula is
    proprietary; the 0.4, 0.2, 0.2 and 0.2 weighting is its common public approximation.
    Assets with the same score share the highest rank among them, so a rating reads as the
    share of the other assets it outperformed or matched, and tied leaders both rate 99.

    For more information, see the following book:

    - O'Neil, W.J. (1988). "How to Make Money in Stocks." McGraw-Hill.

    Args:
        scores (pd.DataFrame): The relative strength scores per date, a column each.

    Returns:
        pd.DataFrame: The rating per date and asset.
    """
    ranks = scores.rank(axis=1, method="max")
    count = scores.notna().sum(axis=1)
    # The weakest asset of a date rates 1 and the strongest 99.
    rating = 1 + (ranks - 1).div((count - 1).replace(0, np.nan), axis=0) * 98
    return rating.round().where(scores.notna())
