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


def get_trin(prices_close: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series:
    """
    Calculate the TRIN (Arms Index) for a universe of tickers.

    TRIN compares the ratio of advancing to declining issues against the ratio of
    volume in advancing issues to volume in declining issues. It is a cross-sectional
    market breadth indicator, meaning it is calculated across the columns (tickers) of
    the provided data rather than per ticker over time.

    The formula is a follows:

    - TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)

    Also known as: Arms Index, TRIN.

    Args:
        prices_close (pd.DataFrame): DataFrame of closing prices with tickers as columns.
        volumes (pd.DataFrame): DataFrame of trading volumes with tickers as columns.

    Returns:
        pd.Series: TRIN values.
    """
    price_change = prices_close.diff()

    advancing = price_change > 0
    declining = price_change < 0

    advancing_issues = advancing.sum(axis=1)
    declining_issues = declining.sum(axis=1)

    advancing_volume = volumes.where(advancing, 0).sum(axis=1)
    declining_volume = volumes.where(declining, 0).sum(axis=1)

    return (advancing_issues / declining_issues) / (advancing_volume / declining_volume)


def get_new_highs_new_lows(prices_close: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the New Highs — New Lows for a universe of tickers.

    New Highs — New Lows measures the number of tickers reaching a new high over the
    specified window minus the number of tickers reaching a new low over the same
    window. It is a cross-sectional market breadth indicator, meaning it is calculated
    across the columns (tickers) of the provided data rather than per ticker over time.

    The formula is a follows:

    - New Highs — New Lows = (Number of tickers at a window-period high) — (Number of tickers at a window-period low)

    Also known as: new highs minus new lows, record high percent.

    Args:
        prices_close (pd.DataFrame): DataFrame of closing prices with tickers as columns.
        window (int): Number of periods to consider for the new high / new low lookback.

    Returns:
        pd.Series: New Highs — New Lows values.
    """
    rolling_high = prices_close.rolling(window=window, min_periods=1).max()
    rolling_low = prices_close.rolling(window=window, min_periods=1).min()

    new_highs = (prices_close >= rolling_high).sum(axis=1)
    new_lows = (prices_close <= rolling_low).sum(axis=1)

    return new_highs - new_lows


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
