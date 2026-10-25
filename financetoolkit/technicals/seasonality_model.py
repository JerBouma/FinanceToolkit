"""Seasonality and Relative Strength Model"""

__docformat__ = "google"

import calendar

import numpy as np
import pandas as pd

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
    Calculate the seasonality of returns: their average per calendar month (from monthly
    returns) or per day of the week (from daily returns), with the median, the share of
    positive returns (hit rate), the t-statistic of the average and the number of
    observations. A pattern only holds up with a t-statistic well above 2 over many years,
    and even then it can disappear once it is widely known.

    For more information, see the following paper:

    - Heston, S.L. and Sadka, R. (2008). "Seasonality in the Cross-Section of Stock Returns."
      Journal of Financial Economics, 87(2), 418-445.

    Args:
        returns (pd.Series): Monthly returns for frequency="month", daily returns for
            frequency="weekday", indexed by date.
        frequency (str, optional): "month" or "weekday". Defaults to "month".
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
        raise ValueError(
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
    prices: pd.DataFrame, benchmark_prices: pd.Series, window: int = 252
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the relative strength of prices against a benchmark: the price relative (the
    price divided by the benchmark's, set to 1 on the first date), which rises when the
    asset outperforms, and the Mansfield relative strength, the price relative against its
    own moving average minus 1, which is positive while the asset is gaining on the
    benchmark. Weinstein reads it from weekly charts against a 52-week average, a year.

    For more information, see the following book:

    - Weinstein, S. (1988). "Stan Weinstein's Secrets for Profiting in Bull and Bear
      Markets." McGraw-Hill.

    Args:
        prices (pd.DataFrame): The prices, a column each.
        benchmark_prices (pd.Series): The benchmark's prices.
        window (int, optional): The moving average window of the Mansfield relative strength,
            in periods of the prices. Defaults to 252 (a year of daily prices).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The price relative and the Mansfield relative
        strength.
    """
    relative = prices.div(benchmark_prices, axis=0)
    relative = relative / relative.apply(
        lambda column: column.dropna().iloc[0] if column.notna().any() else np.nan
    )
    mansfield = relative / relative.rolling(window).mean() - 1
    return relative, mansfield


def get_relative_strength_rating(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the relative strength rating of every asset: its performance over the last
    year, the latest quarter weighing most (the 3, 6, 9 and 12-month returns weighted
    0.4, 0.2, 0.2 and 0.2), ranked against the other assets from 1 (weakest) to 99
    (strongest). The rating is only meaningful across many assets, such as the constituents
    of an index.

    The rating follows the one Investor's Business Daily publishes, whose exact formula is
    proprietary; the 0.4, 0.2, 0.2 and 0.2 weighting is its common public approximation.
    Assets with the same score share the highest rank among them, so a rating reads as the
    share of the other assets it outperformed or matched, and tied leaders both rate 99.

    For more information, see the following book:

    - O'Neil, W.J. (1988). "How to Make Money in Stocks." McGraw-Hill.

    Args:
        prices (pd.DataFrame): The daily prices, a column each.

    Returns:
        pd.DataFrame: The rating per date and asset.
    """
    score = sum(
        weight * (prices / prices.shift(days) - 1)
        for days, weight in RATING_QUARTERS.items()
    )
    ranks = score.rank(axis=1, method="max")
    count = score.notna().sum(axis=1)
    # The weakest asset of a date rates 1 and the strongest 99.
    rating = 1 + (ranks - 1).div((count - 1).replace(0, np.nan), axis=0) * 98
    return rating.round().where(score.notna())
