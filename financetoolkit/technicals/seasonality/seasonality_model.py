"""Seasonality and Relative Strength Model"""

__docformat__ = "google"

import calendar

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.econometrics.least_squares_model import (
    fit_least_squares,
    get_wald_statistic,
)
from financetoolkit.performance.helpers import get_newey_west_lags
from financetoolkit.risk.helpers import WEEKEND_DAY_SHARE, trades_on_weekends
from financetoolkit.technicals.helpers import get_observations_per_year, to_dates
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
    dates = to_dates(values.index)

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


def get_day_of_week_effect(returns: pd.Series, lags: int | None = None) -> pd.DataFrame:
    """
    Calculate the day-of-week effect: the average return of every weekday, and whether the weekdays differ, with
    standard errors that allow for heteroskedastic and autocorrelated returns.

    The returns are regressed on a dummy per weekday without a constant, so each coefficient is the average return
    of its weekday. The Newey-West standard errors keep the tests valid when the volatility clusters or the
    returns are autocorrelated, which the plain t-statistics of `get_seasonality` assume away. The joint test asks
    whether all weekdays share one average (French, 1980; Gibbons and Hess, 1981). The Monday effect, a lower
    return over the weekend, was strong in US stocks until the 1980s and has largely gone since.

    The formula is as follows:

    - Return_t = b_Monday * D_Monday,t + ... + b_Friday * D_Friday,t + e_t
    - Test Statistic of a weekday = b_d / SE_NW(b_d), with a two-sided p-value from the standard normal
    - Equal Means = W / q, with W the Wald statistic of b_Monday = b_Tuesday = ... = b_Friday under the
      Newey-West covariance and q the number of restrictions (weekdays - 1), against the F(q, n - k) distribution

    Where D_d,t is 1 when date t falls on weekday d, SE_NW the Newey-West (1987) standard error with Bartlett
    weights over `lags` lags, n the number of returns and k the number of weekdays.

    For more information, see the following papers:

    - French, K.R. (1980). "Stock Returns and the Weekend Effect." Journal of Financial Economics, 8(1), 55-69.
    - Gibbons, M.R. and Hess, P. (1981). "Day of the Week Effects and Asset Returns." The Journal of
      Business, 54(4), 579-596.
    - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and
      Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703-708.

    Referenced against:
    - statsmodels, OLS (HAC covariance and f_test): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html

    Args:
        returns (pd.Series): The daily returns, indexed by date (a PeriodIndex or DatetimeIndex).
        lags (int | None, optional): The number of lags of the Newey-West covariance. Defaults to None, which
            uses the Newey and West (1994) rule floor(4 * (n / 100)^(2/9)).

    Returns:
        pd.DataFrame: The Average Return, Standard Error, Test Statistic, p-Value and Observations per weekday, and
        an Equal Means row with the F-statistic of the joint test as its Test Statistic.

    Notes:
    - Only the weekdays with returns get a row: Monday to Friday for a stock, every day for a cryptocurrency.
    - The p-values of the weekdays test an average of zero, not a difference from the other days; the Equal
      Means row tests the differences.

    As an example:

    ```python
    import numpy as np
    import pandas as pd

    from financetoolkit.technicals.seasonality import seasonality_model

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", "2025-12-31")
    returns = pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)

    seasonality_model.get_day_of_week_effect(returns).round(4)
    ```

    Which returns:

    |             |   Average Return |   Standard Error |   Test Statistic |   p-Value |   Observations |
    |:------------|-----------------:|-----------------:|-----------------:|----------:|---------------:|
    | Monday      |          -0.0005 |           0.0006 |          -0.8868 |    0.3752 |            261 |
    | Tuesday     |           0.0005 |           0.0006 |           0.8829 |    0.3773 |            261 |
    | Wednesday   |          -0.0006 |           0.0006 |          -0.9249 |    0.355  |            261 |
    | Thursday    |           0.0014 |           0.0006 |           2.2817 |    0.0225 |            260 |
    | Friday      |          -0.0006 |           0.0006 |          -0.9686 |    0.3327 |            261 |
    | Equal Means |         nan      |         nan      |           2.3667 |    0.051  |           1304 |
    """
    values = returns.dropna()
    weekdays = to_dates(values.index).weekday.to_numpy()
    present = np.unique(weekdays)
    design = (weekdays[:, None] == present[None, :]).astype(float)
    observations = len(values)
    lags = get_newey_west_lags(observations) if lags is None else lags

    fit = fit_least_squares(
        values.to_numpy(dtype=float), design, cov_type="HAC", lags=lags
    )

    equal_means = p_value = np.nan
    if len(present) > 1:
        # Every weekday's average against Monday's (or the first weekday present).
        restrictions = np.column_stack(
            [-np.ones(len(present) - 1), np.eye(len(present) - 1)]
        )
        wald, number_of_restrictions = get_wald_statistic(
            fit["coefficients"], fit["covariance_matrix"], restrictions
        )
        equal_means = wald / number_of_restrictions
        p_value = stats.f.sf(
            equal_means, number_of_restrictions, fit["degrees_of_freedom"]
        )

    table = pd.DataFrame(
        {
            "Average Return": fit["coefficients"],
            "Standard Error": fit["standard_errors"],
            "Test Statistic": fit["t_statistics"],
            "p-Value": fit["p_values"],
            "Observations": design.sum(axis=0).astype(int),
        },
        index=[calendar.day_name[day] for day in present],
    )
    table.loc["Equal Means"] = [np.nan, np.nan, equal_means, p_value, observations]
    table["Observations"] = table["Observations"].astype(int)

    return table


def get_turn_of_month_days(
    dates: pd.Index, days_before: int = 1, days_after: int = 3
) -> pd.Series:
    """
    Mark the turn-of-the-month days among trading dates: the last trading days of every month and the first
    trading days of the next.

    Ariel (1987) found that US stock returns came almost entirely from the last trading day of a month and the
    first nine of the next; Lakonishok and Smidt (1988) narrowed the window to the last trading day and the first
    three, the default here.

    The rule is as follows:

    - Turn of Month_t = 1 when date t is one of the last `days_before` trading dates of its month or one of the
      first `days_after`, else 0

    For more information, see the following papers:

    - Ariel, R.A. (1987). "A Monthly Effect in Stock Returns." Journal of Financial Economics, 18(1), 161-174.
    - Lakonishok, J. and Smidt, S. (1988). "Are Seasonal Anomalies Real? A Ninety-Year Perspective." The
      Review of Financial Studies, 1(4), 403-425.

    Args:
        dates (pd.Index): The trading dates (a PeriodIndex or DatetimeIndex), such as the dates of a ticker's
            daily returns.
        days_before (int, optional): The last trading days of a month that count. Defaults to 1.
        days_after (int, optional): The first trading days of a month that count. Defaults to 3.

    Returns:
        pd.Series: 1.0 on the turn-of-the-month dates and 0.0 on the others, NaN in a first or last month the
        dates do not cover in full.

    Notes:
    - The trading days of a month are counted on the dates given. A first month whose dates begin after its first
      business day, or a last month whose dates end before its last business day, is NaN, as its positions are
      unknown. Business days ignore exchange holidays, so a holiday on that business day (such as New Year's
      Day) also leaves the month out.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.technicals.seasonality import seasonality_model

    dates = pd.bdate_range("2025-01-01", "2025-02-28").to_period("D")

    seasonality_model.get_turn_of_month_days(dates).loc["2025-01-29":"2025-02-06"]
    ```

    Which returns:

    |            |   Value |
    |:-----------|--------:|
    | 2025-01-29 |       0 |
    | 2025-01-30 |       0 |
    | 2025-01-31 |       1 |
    | 2025-02-03 |       1 |
    | 2025-02-04 |       1 |
    | 2025-02-05 |       1 |
    | 2025-02-06 |       0 |
    """
    days = to_dates(dates)
    months = days.to_period("M")
    position = pd.Series(months, index=days).groupby(months)
    from_start = position.cumcount().to_numpy() + 1
    from_end = position.cumcount(ascending=False).to_numpy() + 1

    marked = pd.Series(
        ((from_end <= days_before) | (from_start <= days_after)).astype(float),
        index=dates,
    )

    if len(days):
        first_business_day = pd.offsets.BMonthBegin().rollforward(months[0].start_time)
        last_business_day = pd.offsets.BMonthEnd().rollforward(months[-1].start_time)
        incomplete = ((months == months[0]) & (days[0] > first_business_day)) | (
            (months == months[-1]) & (days[-1] < last_business_day)
        )
        marked[np.asarray(incomplete)] = np.nan

    return marked


def get_pre_holiday_days(dates: pd.Index) -> pd.Series:
    """
    Mark the pre-holiday days among trading dates: the trading days followed by a regular trading weekday without
    trading.

    Lakonishok and Smidt (1988) and Ariel (1990) found returns on the trading day before a holiday to be many
    times those of other days. The holidays are read from the dates themselves: a day of the week the asset
    regularly trades on (more than 10% of its dates fall on it) without trading is taken as an exchange holiday.

    The rule is as follows:

    - Pre-Holiday_t = 1 when the next regular trading weekday after date t is not among the dates, else 0

    For more information, see the following papers:

    - Lakonishok, J. and Smidt, S. (1988). "Are Seasonal Anomalies Real? A Ninety-Year Perspective." The
      Review of Financial Studies, 1(4), 403-425.
    - Ariel, R.A. (1990). "High Stock Returns before Holidays: Existence and Evidence on Possible Causes."
      The Journal of Finance, 45(5), 1611-1626.

    Args:
        dates (pd.Index): The trading dates (a PeriodIndex or DatetimeIndex), such as the dates of a ticker's
            daily returns.

    Returns:
        pd.Series: 1.0 on the pre-holiday dates and 0.0 on the others, NaN on the last date (whose next trading
        day is not known) and on every date of an asset that trades on Saturdays and Sundays, such as a
        cryptocurrency, which has no holidays to read.

    Notes:
    - A weekday missing from the data for another reason (a trading halt or a gap in the data) also counts as a
      holiday, so the dates should be those of a complete price history.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.technicals.seasonality import seasonality_model

    dates = pd.bdate_range("2025-12-01", "2025-12-31").drop(pd.Timestamp("2025-12-25"))

    seasonality_model.get_pre_holiday_days(dates.to_period("D")).loc["2025-12-19":]
    ```

    Which returns:

    |            |   Value |
    |:-----------|--------:|
    | 2025-12-19 |       0 |
    | 2025-12-22 |       0 |
    | 2025-12-23 |       0 |
    | 2025-12-24 |       1 |
    | 2025-12-26 |       0 |
    | 2025-12-29 |       0 |
    | 2025-12-30 |       0 |
    | 2025-12-31 |     nan |
    """
    days = to_dates(dates)
    marked = pd.Series(np.nan, index=dates)

    # A cryptocurrency trades on every day, so a missing day is a gap in its data, not a holiday.
    if len(days) == 0 or trades_on_weekends(days.to_period("D")):
        return marked

    # The days of the week the asset trades on (each more than 10% of its dates): Monday to Friday on most
    # exchanges, Sunday to Thursday on some, so a weekly closing day is not taken for a holiday.
    trading_weekdays = (
        np.bincount(days.weekday, minlength=7) / len(days) > WEEKEND_DAY_SHARE
    )
    # The calendar days from each day of the week to the next day of the week the asset trades on.
    days_ahead = np.array(
        [
            next(step for step in range(1, 8) if trading_weekdays[(weekday + step) % 7])
            for weekday in range(7)
        ]
    )
    next_trading_day = days + pd.to_timedelta(days_ahead[days.weekday], unit="D")

    marked[:] = (~next_trading_day.isin(days)).astype(float)
    marked.iloc[-1] = np.nan

    return marked


def get_calendar_effect(
    returns: pd.Series,
    marked_days: pd.Series,
    name: str = "Marked Days",
    lags: int | None = None,
) -> pd.DataFrame:
    """
    Calculate a calendar effect: the average return on the marked days (such as the turn of the month or the days
    before a holiday), on the other days and the difference, with standard errors that allow for
    heteroskedastic and autocorrelated returns.

    The returns are regressed on a constant and a dummy that is 1 on the marked days, so the constant is the
    average return of the other days and the slope the difference (Lakonishok and Smidt, 1988). The Newey-West
    standard errors keep the test valid when the volatility clusters or the returns are autocorrelated.

    The formula is as follows:

    - Return_t = a + b * D_t + e_t, with D_t = 1 on the marked days and 0 on the others
    - Other Days = a, the marked days (`name`) = a + b and Difference = b
    - Test Statistic = estimate / SE_NW(estimate), with a two-sided p-value from the standard normal

    Where SE_NW is the Newey-West (1987) standard error with Bartlett weights over `lags` lags; the standard
    error of a + b includes the covariance of a and b.

    For more information, see the following papers:

    - Lakonishok, J. and Smidt, S. (1988). "Are Seasonal Anomalies Real? A Ninety-Year Perspective." The
      Review of Financial Studies, 1(4), 403-425.
    - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and
      Autocorrelation Consistent Covariance Matrix." Econometrica, 55(3), 703-708.

    Referenced against:
    - statsmodels, OLS (HAC covariance): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html

    Args:
        returns (pd.Series): The daily returns, indexed by date.
        marked_days (pd.Series): 1 on the marked days and 0 on the others, on the dates of the returns, such as
            those of `get_turn_of_month_days` or `get_pre_holiday_days`. Dates where it is NaN are left out.
        name (str, optional): The name of the marked days in the result. Defaults to "Marked Days".
        lags (int | None, optional): The number of lags of the Newey-West covariance. Defaults to None, which
            uses the Newey and West (1994) rule floor(4 * (n / 100)^(2/9)).

    Returns:
        pd.DataFrame: The Average Return, Standard Error, Test Statistic, p-Value and Observations of the marked
        days, the Other Days and the Difference. NaN when the returns hold no marked or no other days.

    As an example:

    ```python
    import numpy as np
    import pandas as pd

    from financetoolkit.technicals.seasonality import seasonality_model

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", "2025-12-31")
    returns = pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates)

    seasonality_model.get_calendar_effect(
        returns, seasonality_model.get_turn_of_month_days(returns.index), "Turn of Month"
    ).round(4)
    ```

    Which returns:

    |               |   Average Return |   Standard Error |   Test Statistic |   p-Value |   Observations |
    |:--------------|-----------------:|-----------------:|-----------------:|----------:|---------------:|
    | Turn of Month |          -0.0006 |           0.0006 |          -0.9881 |    0.3231 |            240 |
    | Other Days    |           0.0002 |           0.0003 |           0.7237 |    0.4693 |           1064 |
    | Difference    |          -0.0009 |           0.0007 |          -1.2143 |    0.2246 |           1304 |
    """
    data = pd.concat([returns, marked_days.reindex(returns.index)], axis=1).dropna()
    values = data.iloc[:, 0].to_numpy(dtype=float)
    marked = data.iloc[:, 1].to_numpy(dtype=float)
    marked_count = int(marked.sum())
    observations = len(values)
    index = [name, "Other Days", "Difference"]
    counts = [marked_count, observations - marked_count, observations]

    if marked_count in (0, observations):
        return pd.DataFrame(
            {
                "Average Return": np.nan,
                "Standard Error": np.nan,
                "Test Statistic": np.nan,
                "p-Value": np.nan,
                "Observations": counts,
            },
            index=index,
        )

    lags = get_newey_west_lags(observations) if lags is None else lags
    fit = fit_least_squares(
        values,
        np.column_stack([np.ones(observations), marked]),
        cov_type="HAC",
        lags=lags,
    )
    covariance = fit["covariance_matrix"]
    # The marked days' average a + b and its variance Var(a) + Var(b) + 2 Cov(a, b).
    estimates = np.array(
        [fit["coefficients"].sum(), fit["coefficients"][0], fit["coefficients"][1]]
    )
    standard_errors = np.sqrt([covariance.sum(), covariance[0, 0], covariance[1, 1]])
    test_statistics = estimates / standard_errors

    return pd.DataFrame(
        {
            "Average Return": estimates,
            "Standard Error": standard_errors,
            "Test Statistic": test_statistics,
            "p-Value": 2 * stats.norm.sf(np.abs(test_statistics)),
            "Observations": counts,
        },
        index=index,
    )
