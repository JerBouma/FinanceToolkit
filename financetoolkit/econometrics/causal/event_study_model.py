"""Event Study Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.econometrics.regression import regression_model


def get_event_study(
    returns: pd.Series,
    market_returns: pd.Series,
    event_date: str,
    estimation_window: int = 250,
    gap_days: int = 30,
    pre_event_days: int = 10,
    post_event_days: int = 10,
) -> dict:
    """
    Perform a market-model event study around a single event date: the return of a stock
    around an event beyond what its relation with the market explains (the abnormal return).

    The method follows MacKinlay, A.C. (1997), "Event Studies in Economics and Finance,"
    Journal of Economic Literature, 35(1), 13-39 -- the canonical reference and still
    the dominant approach used to measure the stock-price impact of corporate events
    (earnings announcements, M&A deals, index additions/deletions, dividend changes,
    regulatory actions, etc.).

    The method has two steps:

    1. Estimate a market model (`Return_t = alpha + beta * Market_Return_t + e_t`) via
       OLS over a clean "estimation window" that ends `gap_days` before the event, so
       the event itself cannot contaminate the estimate of what a "normal" return
       would have looked like.
    2. Compute the Abnormal Return (AR) for each day in the "event window" around the
       event date as the actual return minus the market-model-predicted expected
       return, and cumulate them into the Cumulative Abnormal Return (CAR) -- the
       total return attributable to the event, after stripping out the return that
       would have been expected from general market movements alone.

    The formula is as follows:

    - Market model (fit over the estimation window): Return_t = alpha + beta *
      Market_Return_t + e_t
    - Abnormal Return: AR_t = Return_t - (alpha + beta * Market_Return_t)
    - Cumulative Abnormal Return: CAR = SUM(AR_t) over the event window
    - Var(CAR) = L2 * sigma_e^2, where L2 is the number of days in the event window
      and sigma_e^2 is the market model's estimation-window residual variance
    - t-statistic: CAR / sqrt(Var(CAR)), tested against a Student-T distribution with
      the estimation window's residual degrees of freedom

    Also known as: CAR analysis, abnormal returns analysis, market model event study.

    Args:
        returns (pd.Series): Daily returns of the security being studied, indexed by
            date. Must cover both the estimation and event windows.
        market_returns (pd.Series): Daily returns of the market/benchmark used to
            estimate normal returns. Reindexed to `returns`'s index.
        event_date (str): The date of the event, between the first and the last date of
            `returns`. A date without a return (a weekend or holiday) moves to the next
            date with one, the first trading day the event can be priced.
        estimation_window (int): Number of trading days used to estimate the market
            model. Defaults to 250 (approximately one trading year).
        gap_days (int): Number of trading days between the end of the estimation
            window and the event date, so pre-event drift or information leakage
            doesn't contaminate the market-model estimate. Defaults to 30.
        pre_event_days (int): Number of trading days before the event date included
            in the event window. Defaults to 10.
        post_event_days (int): Number of trading days after the event date included
            in the event window. Defaults to 10.

    Returns:
        dict: The abnormal returns, CAR and its significance test, plus the
            underlying market-model parameters -- keys `abnormal_returns`,
            `cumulative_abnormal_return`, `car_t_statistic`, `car_p_value`, `alpha`,
            `beta`, `estimation_window_observations`.

    Raises:
        TypeError: If `returns` or `market_returns` is not a pd.Series.
        ValueError: If `event_date` is outside the dates of `returns`, there is not enough
            history before/after the event date for the requested windows, or
            `gap_days` is too small to keep the estimation window from overlapping
            the event window (i.e. `pre_event_days > gap_days`).
    """
    if not isinstance(returns, pd.Series) or not isinstance(market_returns, pd.Series):
        raise TypeError("returns and market_returns must be pd.Series.")

    returns = returns.sort_index()
    market_returns = market_returns.reindex(returns.index)

    # An event on a weekend or holiday (a release on a Saturday) is first priced on the
    # next trading day, so the event day is the first date on or after event_date.
    timestamps = (
        returns.index.to_timestamp()
        if isinstance(returns.index, pd.PeriodIndex)
        else pd.DatetimeIndex(returns.index)
    )
    event = pd.Timestamp(event_date)
    # A date without a time zone is read in the time zone of the returns.
    if timestamps.tz is not None and event.tzinfo is None:
        event = event.tz_localize(timestamps.tz)
    if returns.empty:
        raise ValueError("There are no returns to study the event with.")
    if not timestamps[0] <= event <= timestamps[-1]:
        raise ValueError(
            f"event_date {event_date} is outside the returns, which run from "
            f"{returns.index[0]} to {returns.index[-1]}."
        )

    if pre_event_days > gap_days:
        raise ValueError(
            f"pre_event_days ({pre_event_days}) must not exceed gap_days "
            f"({gap_days}) -- otherwise the estimation window (which ends "
            "gap_days before the event) would overlap the event window (which "
            "starts pre_event_days before the event), contaminating the market "
            "model's 'normal return' estimate with event-window observations."
        )

    event_position = int(timestamps.searchsorted(event, side="left"))

    estimation_end = event_position - gap_days
    estimation_start = estimation_end - estimation_window
    event_start = event_position - pre_event_days
    event_end = event_position + post_event_days + 1

    if estimation_start < 0:
        raise ValueError(
            "Not enough history before the event date for the requested "
            "estimation_window and gap_days."
        )
    if event_end > len(returns):
        raise ValueError(
            "Not enough history after the event date for the requested "
            "post_event_days."
        )

    estimation_returns = returns.iloc[estimation_start:estimation_end]
    estimation_market = market_returns.iloc[estimation_start:estimation_end]

    market_model = regression_model.get_ols(
        y=estimation_returns.reset_index(drop=True),
        x=estimation_market.reset_index(drop=True),
    )
    alpha = market_model["coefficients"][0]
    beta = market_model["coefficients"][1]

    event_returns = returns.iloc[event_start:event_end]
    event_market = market_returns.iloc[event_start:event_end]

    expected_returns = alpha + beta * event_market
    abnormal_returns = event_returns - expected_returns

    car = float(abnormal_returns.sum())
    car_variance = len(abnormal_returns) * market_model["residual_variance"]
    car_t_statistic = car / np.sqrt(car_variance) if car_variance > 0 else np.nan
    car_p_value = (
        2 * (1 - stats.t.cdf(abs(car_t_statistic), market_model["degrees_of_freedom"]))
        if not np.isnan(car_t_statistic)
        else np.nan
    )

    return {
        "abnormal_returns": abnormal_returns,
        "cumulative_abnormal_return": car,
        "car_t_statistic": float(car_t_statistic),
        "car_p_value": float(car_p_value),
        "alpha": float(alpha),
        "beta": float(beta),
        "estimation_window_observations": len(estimation_returns),
    }
