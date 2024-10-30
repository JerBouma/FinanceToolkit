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


def get_on_balance_volume(prices_close: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV) of a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.

    Returns:
        pd.Series: OBV values.
    """
    price_diff = prices_close.diff(1)
    obv = (price_diff / abs(price_diff)) * volumes

    return obv.cumsum()


def get_accumulation_distribution_line(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
) -> pd.Series:
    """
    Calculate the Accumulation/Distribution Line for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.

    Returns:
        pd.Series: Accumulation/Distribution Line values.
    """
    money_flow_multiplier = (
        (prices_close - prices_low) - (prices_high - prices_close)
    ) / (prices_high - prices_low)
    money_flow_volume = money_flow_multiplier * volumes

    return money_flow_volume.cumsum()


def get_chaikin_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Calculate the Chaikin Oscillator for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        short_window (int): Number of periods for the short-term EMA.
        long_window (int): Number of periods for the long-term EMA.

    Returns:
        pd.Series: Chaikin Oscillator values.
    """
    adl = get_accumulation_distribution_line(
        prices_high, prices_low, prices_close, volumes
    )
    short_ema = adl.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = adl.ewm(span=long_window, min_periods=1, adjust=False).mean()

    return short_ema - long_ema
