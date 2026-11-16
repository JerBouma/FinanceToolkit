"""Performance Helpers Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    WITHIN_PERIOD_TRANSLATION,
    to_datetime_index,
)


def determine_within_dataset(
    dataset: pd.DataFrame, period: str, correlation: bool = False
) -> pd.DataFrame:
    """
    Groups a daily dataset, such as the Fama-French factors, by the period each day falls
    in, the "within period" format the factor models of the Performance controller are
    estimated on.

    Args:
        dataset (pd.DataFrame): The daily dataset, with a PeriodIndex.
        period (str): The period to group by (weekly, monthly, quarterly or yearly).
        correlation (bool, optional): Whether to return the correlation matrix of the
            columns per period instead of the daily observations. Defaults to False.

    Raises:
        KeyError: If the period is not one of the keys of WITHIN_PERIOD_TRANSLATION.

    Returns:
        pd.DataFrame: The daily observations with the period and the day as the index
        levels, or with correlation=True the correlation matrix per period with the period
        and the column as the index levels.
    """
    dataset_new = dataset.copy()
    dataset_new.index = pd.DatetimeIndex(dataset_new.to_timestamp().index)
    period_symbol = WITHIN_PERIOD_TRANSLATION[period]

    if correlation:
        within_historical_data = dataset_new.groupby(
            pd.Grouper(
                freq=(
                    f"{period_symbol}E"
                    if period_symbol in ["M", "Q", "Y"]
                    else period_symbol
                )
            )
        ).apply(lambda x: x.corr())

        # The grouped correlation matrices are indexed by (period, ticker) and the
        # outer level is relabelled from timestamps to periods.
        grouped_index = within_historical_data.index
        if not isinstance(grouped_index, pd.MultiIndex):
            raise TypeError("Expected the grouped correlations to be MultiIndexed.")

        within_historical_data.index = grouped_index.set_levels(
            [
                pd.PeriodIndex(grouped_index.levels[0], freq=period_symbol),
                grouped_index.levels[1],
            ],
        )
    else:
        within_historical_data = dataset_new.copy()
        within_historical_data.index = pd.MultiIndex.from_arrays(
            [
                to_datetime_index(dataset_new.index).to_period(period_symbol),
                to_datetime_index(dataset_new.index).to_period("D"),
            ]
        )

    return within_historical_data


def get_newey_west_lags(n_observations: int) -> int:
    """
    The Newey and West (1994) rule of thumb for the number of autocorrelation lags to
    include: floor(4 * (n / 100)^(2/9)), e.g. 4 lags for a year of 252 daily returns and 3
    for five years of monthly returns.

    For more information about the rule, see the following paper:

    - Newey, W.K., & West, K.D. (1994). "Automatic Lag Selection in Covariance Matrix
    Estimation." Review of Economic Studies, 61(4), 631-653.

    Args:
        n_observations (int): The number of observations.

    Returns:
        int: The number of lags.
    """
    return int(np.floor(4 * (n_observations / 100) ** (2 / 9)))


def align_annualized_rate(
    annualized_rate: pd.Series, dates: pd.PeriodIndex
) -> pd.Series:
    """
    The annualized risk free rate on each of the dates: a date without a quote (a weekend or
    a holiday of the Treasury market) takes the most recent earlier quote, and the dates
    before the first quote take the first quote.

    Args:
        annualized_rate (pd.Series): The annualized risk free rate with a daily PeriodIndex.
        dates (pd.PeriodIndex): The daily dates to align the rate to.

    Returns:
        pd.Series: The annualized risk free rate per date.
    """
    rate = annualized_rate.sort_index()
    rate = rate[~rate.index.duplicated(keep="last")]

    return rate.reindex(rate.index.union(dates)).ffill().bfill().reindex(dates)


def trades_on_weekends(dates: pd.PeriodIndex) -> bool:
    """
    Whether the daily dates include a Saturday or a Sunday, as those of a cryptocurrency
    do, so the dates are calendar days rather than trading days.

    Args:
        dates (pd.PeriodIndex): The daily dates with an observation.

    Returns:
        bool: True when a date falls on a weekend.
    """
    return bool((dates.dayofweek >= 5).any())  # noqa: PLR2004


def get_calendar_days_between(dates: pd.PeriodIndex) -> np.ndarray:
    """
    The number of calendar days since the date before, 1 for the first date.

    Args:
        dates (pd.PeriodIndex): The daily dates, in order.

    Returns:
        np.ndarray: The calendar days since the date before, per date.
    """
    ordinals = dates.asfreq("D").asi8

    return (
        np.diff(ordinals, prepend=ordinals[0] - 1).astype(float)
        if len(dates)
        else np.array([])
    )


def get_risk_free_rate_per_date(
    annualized_rate: pd.Series, returns: pd.DataFrame
) -> pd.Series:
    """
    The risk free return earned on each date of the daily returns since the date before, from
    the annualized risk free rate (such as a Treasury yield) quoted on that date. It is the
    rate the returns of a period are compared with once compounded over the period (see
    `compound_risk_free_rate`).

    Dates that are trading days (no returns on a Saturday or Sunday) follow the trading day
    convention of 252 days a year, so a year of trading days earns the annualized rate.
    Once a ticker trades on weekends as well, such as a cryptocurrency, the dates are
    calendar days and the rate accrues over the calendar days since the date before, on
    365 days a year.

    The formula is as follows:

        - Trading Days: Risk Free Rate per Date = (1 + Annualized Rate)^(1 / 252) - 1
        - Calendar Days: Risk Free Rate per Date = (1 + Annualized Rate)^(Days Since the Date Before / 365) - 1

    Args:
        annualized_rate (pd.Series): The annualized risk free rate with a daily PeriodIndex, see
            `align_annualized_rate`.
        returns (pd.DataFrame): The daily returns with a daily PeriodIndex, one column per ticker.

    Returns:
        pd.Series: The risk free return per date of the returns, as a decimal.
    """
    dates = returns.index
    rate = align_annualized_rate(annualized_rate, dates)

    if trades_on_weekends(dates[returns.notna().any(axis=1).to_numpy()]):
        return (1 + rate) ** (get_calendar_days_between(dates) / 365) - 1

    return (1 + rate) ** (1 / 252) - 1


def get_risk_free_rate_per_observation(
    annualized_rate: pd.Series, returns: pd.DataFrame
) -> pd.DataFrame:
    """
    The risk free return each ticker earns over the period of each of its returns, from the
    annualized risk free rate quoted on the date of the return.

    A ticker that trades on trading days earns a trading day of the rate with every return,
    on 252 trading days a year, whichever other tickers the data holds: its measures use
    its own trading days. A ticker that also trades on weekends, such as a cryptocurrency,
    earns the rate over the calendar days since its previous return, on 365 days a year,
    so it earns the annualized rate over a year rather than about 365 / 252 times it.

    The formula is as follows:

        - Trading Days: Risk Free Rate per Observation = (1 + Annualized Rate)^(1 / 252) - 1
        - Weekends Included: Risk Free Rate per Observation = (1 + Annualized Rate)^(Days Since the Previous
        Observation / 365) - 1

    Args:
        annualized_rate (pd.Series): The annualized risk free rate with a daily PeriodIndex, see
            `align_annualized_rate`.
        returns (pd.DataFrame): The daily returns with a daily PeriodIndex, one column per ticker,
            NaN on the dates a ticker has no return.

    Returns:
        pd.DataFrame: The risk free return per observation, shaped as the returns and NaN where
        the returns are NaN. The first observation of a ticker trading on weekends earns one
        calendar day.
    """
    rate = align_annualized_rate(annualized_rate, returns.index)
    risk_free_rate = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for position, column in enumerate(returns.columns):
        observed = returns.iloc[:, position].notna().to_numpy()
        dates = returns.index[observed]
        exponent = (
            get_calendar_days_between(dates) / 365
            if trades_on_weekends(dates)
            else 1 / 252
        )
        risk_free_rate.iloc[observed, position] = (
            (1 + rate[observed]) ** exponent - 1
        ).to_numpy()

    return risk_free_rate


def compound_risk_free_rate(
    risk_free_rate_per_date: pd.Series, returns: pd.DataFrame, period: str
) -> pd.Series:
    """
    The risk free return of each period: the risk free return per date compounded over the
    dates in the period that hold a return, as the return of a period compounds the daily
    returns within it. A period the data covers in part, such as the current year, earns the
    risk free rate of the dates it covers rather than of a full period.

    The formula is as follows:

        - Period Risk Free Rate = PRODUCT(1 + Risk Free Rate per Date) - 1, over the dates in the period

    Args:
        risk_free_rate_per_date (pd.Series): The risk free return per date, see
            `get_risk_free_rate_per_date`.
        returns (pd.DataFrame): The daily returns, one column per ticker.
        period (str): The period to compound to (weekly, monthly, quarterly or yearly).

    Returns:
        pd.Series: The risk free return per period, as a decimal.
    """
    rate = risk_free_rate_per_date.reindex(returns.index)[
        returns.notna().any(axis=1).to_numpy()
    ]

    return (1 + rate).groupby(rate.index.asfreq(PERIOD_TRANSLATION[period])).prod() - 1


def get_observations_per_year(returns: pd.DataFrame) -> pd.Series:
    """
    The number of daily returns in a year per ticker, to annualize a ratio computed from
    them: 252 for a ticker that trades on trading days only, and 365 for a ticker that also
    trades on weekends, such as a cryptocurrency with a return on every calendar day.

    Args:
        returns (pd.DataFrame): The daily returns with a daily PeriodIndex, one column per ticker.

    Returns:
        pd.Series: The observations per year, indexed by ticker.
    """
    return pd.Series(
        {
            column: (
                365.0
                if trades_on_weekends(returns.index[returns[column].notna().to_numpy()])
                else 252.0
            )
            for column in returns.columns
        },
        dtype=float,
    )
