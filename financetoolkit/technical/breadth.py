"""Breadth Module"""
__docformat__ = "google"

import pandas as pd


def get_mcclellan_oscillator(
    prices_close: pd.Series, short_ema_window: int, long_ema_window: int
) -> pd.Series:
    """
    Calculate the McClellan Oscillator for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_ema_window (int): Number of periods for the short-term EMA.
        long_ema_window (int): Number of periods for the long-term EMA.

    Returns:
        pd.Series: McClellan Oscillator values.
    """
    advancers_decliners = get_advancers_decliners(prices_close)
    short_ema = advancers_decliners.ewm(
        span=short_ema_window, min_periods=1, adjust=False
    ).mean()
    long_ema = advancers_decliners.ewm(
        span=long_ema_window, min_periods=1, adjust=False
    ).mean()

    return short_ema - long_ema


def get_advancers_decliners(prices_close: pd.Series) -> pd.Series:
    """
    Calculate the difference between advancers and decliners for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: Advancers - Decliners values.
    """
    advancers = prices_close.where(prices_close > prices_close.shift(1), 0)
    decliners = -prices_close.where(prices_close < prices_close.shift(1), 0)

    return advancers - decliners
