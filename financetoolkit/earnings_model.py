"""Earnings Model: the surprise of reported earnings and the share price reaction to them."""

import numpy as np
import pandas as pd


def get_earnings_surprise(
    earnings: pd.DataFrame,
    prices: pd.Series,
    benchmark_prices: pd.Series | None = None,
    window: int = 1,
    sue_window: int = 8,
) -> pd.DataFrame:
    """
    Calculate the surprise of each earnings announcement of one company and the share price
    reaction to it.

    The EPS and revenue surprise are the reported figure against the analyst consensus, as a
    fraction of the absolute consensus. The standardized unexpected earnings (SUE) divide the
    EPS surprise by the standard deviation (ddof=1) of the EPS surprises of the sue_window
    announcements before it, so only information known at the time is used (Livnat and
    Mendenhall, 2006, with the analyst consensus as the expectation; Bernard and Thomas, 1989,
    scale a time-series surprise the same way).

    The reaction compounds the share's returns over the trading days from window days before
    to window days after the announcement (day 0 is the announcement date or, when that is not
    a trading day, the next one). With the default window of 1 these are days -1, 0 and +1,
    which covers an announcement before the open (reaction on day 0) and after the close
    (reaction on day +1) alike, as the source doesn't say which it was. The abnormal reaction
    subtracts the benchmark's return between the same dates (a market-adjusted return).

    Args:
        earnings (pd.DataFrame): The announcements, indexed by date, with the columns "EPS",
            "Estimated EPS", "Revenue" and "Estimated Revenue".
        prices (pd.Series): The company's daily (adjusted) prices on its trading days.
        benchmark_prices (pd.Series | None, optional): The benchmark's daily prices. Defaults
            to None, which leaves the benchmark and abnormal reaction out.
        window (int, optional): The trading days before and after the announcement the
            reaction covers. Defaults to 1.
        sue_window (int, optional): The number of earlier announcements whose EPS surprises
            scale the SUE. Defaults to 8 (two years of quarterly reports).

    Returns:
        pd.DataFrame: Per reported announcement the EPS, Estimated EPS, EPS Surprise, Revenue,
            Estimated Revenue, Revenue Surprise, SUE and Reaction, plus Benchmark Reaction and
            Abnormal Reaction when benchmark prices are given.
    """
    reported = earnings[earnings["EPS"].notna()].sort_index()

    eps_difference = reported["EPS"] - reported["Estimated EPS"]
    eps_surprise = eps_difference / reported["Estimated EPS"].abs().replace(0, np.nan)
    revenue_surprise = (reported["Revenue"] - reported["Estimated Revenue"]) / reported[
        "Estimated Revenue"
    ].abs().replace(0, np.nan)
    # The surprises before each announcement, so the scale holds no look-ahead.
    surprise_scale = (
        eps_difference.shift(1).rolling(sue_window, min_periods=sue_window).std(ddof=1)
    )

    result = pd.DataFrame(
        {
            "EPS": reported["EPS"],
            "Estimated EPS": reported["Estimated EPS"],
            "EPS Surprise": eps_surprise,
            "Revenue": reported["Revenue"],
            "Estimated Revenue": reported["Estimated Revenue"],
            "Revenue Surprise": revenue_surprise,
            "SUE": eps_difference / surprise_scale.replace(0, np.nan),
        },
        index=reported.index,
    )

    prices = prices.dropna()
    dates = pd.DatetimeIndex(
        prices.index.to_timestamp()
        if isinstance(prices.index, pd.PeriodIndex)
        else prices.index
    )
    announcement_dates = pd.DatetimeIndex(
        reported.index.to_timestamp()
        if isinstance(reported.index, pd.PeriodIndex)
        else reported.index
    )
    # Day 0 is the first trading day on or after the announcement; the reaction runs from
    # the close before day -window to the close of day +window.
    day_zero = dates.searchsorted(announcement_dates)
    before = day_zero - window - 1
    after = day_zero + window
    available = (before >= 0) & (after < len(dates))

    values = prices.to_numpy()
    reaction = np.full(len(reported), np.nan)
    reaction[available] = values[after[available]] / values[before[available]] - 1
    result["Reaction"] = reaction

    if benchmark_prices is not None:
        benchmark_prices = benchmark_prices.dropna()
        benchmark_prices.index = pd.DatetimeIndex(
            benchmark_prices.index.to_timestamp()
            if isinstance(benchmark_prices.index, pd.PeriodIndex)
            else benchmark_prices.index
        )
        benchmark_reaction = np.full(len(reported), np.nan)
        if not benchmark_prices.empty and available.any():
            # The benchmark's last close on or before each date, as its exchange may have
            # been closed on a day the company traded.
            start = benchmark_prices.asof(dates[before[available]]).to_numpy()
            end = benchmark_prices.asof(dates[after[available]]).to_numpy()
            benchmark_reaction[available] = end / start - 1
        result["Benchmark Reaction"] = benchmark_reaction
        result["Abnormal Reaction"] = result["Reaction"] - result["Benchmark Reaction"]

    return result
