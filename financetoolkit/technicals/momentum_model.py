"""Momentum Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.technicals.overlap_model import (
    get_exponential_moving_average,
    get_moving_average,
)


def get_money_flow_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate the Money Flow Index (MFI) indicator for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for MFI calculation.

    Returns:
        pd.Series: MFI values.
    """
    typical_prices = (prices_high + prices_low + prices_close) / 3
    raw_money_flow = typical_prices * volumes

    positive_money_flow = (
        raw_money_flow.where(typical_prices > typical_prices.shift(1), 0)
        .rolling(window=window)
        .sum()
    )
    negative_money_flow = (
        raw_money_flow.where(typical_prices < typical_prices.shift(1), 0)
        .rolling(window=window)
        .sum()
    )

    money_ratio = positive_money_flow / negative_money_flow
    mfi = 100 - (100 / (1 + money_ratio))

    return mfi


def get_williams_percent_r(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Williams %R indicator for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for %R calculation.

    Returns:
        pd.Series: Williams %R values.
    """
    highest_high = prices_high.rolling(window=window).max()
    lowest_low = prices_low.rolling(window=window).min()

    percent_r = -((highest_high - prices_close) / (highest_high - lowest_low)) * 100
    return percent_r


def get_aroon_indicator(
    prices_high: pd.Series, prices_low: pd.Series, window: int
) -> pd.DataFrame:
    """
    Calculate the Aroon Indicator for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods for Aroon calculation.

    Returns:
        pd.DataFrame: Aroon Up and Aroon Down values.
    """
    aroon_up = (
        (
            window
            - prices_high.rolling(window=window).apply(lambda x: x.argmax(), raw=True)
        )
        / window
        * 100
    )
    aroon_down = (
        (
            window
            - prices_low.rolling(window=window).apply(lambda x: x.argmin(), raw=True)
        )
        / window
        * 100
    )

    return pd.concat([aroon_up, aroon_down], keys=["Aroon Up", "Aroon Down"], axis=1)


def get_commodity_channel_index(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    constant: float = 0.015,
) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI) for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for CCI calculation.
        constant (float): Constant multiplier for CCI.

    Returns:
        pd.Series: CCI values.
    """
    typical_prices = (prices_high + prices_low + prices_close) / 3
    sma_typical_prices = typical_prices.rolling(window=window).mean()

    mean_deviation = (
        (typical_prices - sma_typical_prices).abs().rolling(window=window).mean()
    )

    cci_values = (typical_prices - sma_typical_prices) / (constant * mean_deviation)

    return cci_values


def get_relative_vigor_index(
    prices_open: pd.Series, prices_close: pd.Series, volumes: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Relative Vigor Index (RVI) for a given price series.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for RVI calculation.

    Returns:
        pd.Series: RVI values.
    """
    close_open_diff = prices_close - prices_open

    up_close_open = close_open_diff.where(close_open_diff > 0, 0)
    down_close_open = -close_open_diff.where(close_open_diff < 0, 0)

    up_sum = up_close_open.rolling(window=window).sum()
    down_sum = down_close_open.rolling(window=window).sum()

    volume_sum = volumes.rolling(window=window).sum()

    rvi = (up_sum / volume_sum) / (down_sum / volume_sum)

    return rvi


def get_force_index(
    prices_close: pd.Series, volumes: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Force Index for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods for Force Index calculation.

    Returns:
        pd.Series: Force Index values.
    """
    return prices_close.diff(1) * volumes.rolling(window=window).sum()


def get_ultimate_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window_1: int,
    window_2: int,
    window_3: int,
) -> pd.Series:
    """
    Calculate the Ultimate Oscillator for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window_1 (int): Number of periods for the first time period.
        window_2 (int): Number of periods for the second time period.
        window_3 (int): Number of periods for the third time period.

    Returns:
        pd.Series: Ultimate Oscillator values.
    """
    true_range = pd.concat(
        [
            prices_high - prices_low,
            abs(prices_high - prices_close.shift(1)),
            abs(prices_low - prices_close.shift(1)),
        ],
        axis=1,
    ).max(axis=1)

    avg_true_range_1 = true_range.rolling(window=window_1).mean()
    avg_true_range_2 = true_range.rolling(window=window_2).mean()
    avg_true_range_3 = true_range.rolling(window=window_3).mean()

    buying_pressure = prices_close - pd.concat(
        [prices_low.shift(1), prices_close.shift(1)], axis=1
    ).min(axis=1)

    avg_buying_pressure_1 = buying_pressure.rolling(window=window_1).sum()
    avg_buying_pressure_2 = buying_pressure.rolling(window=window_2).sum()
    avg_buying_pressure_3 = buying_pressure.rolling(window=window_3).sum()

    ultimate_oscillator = (
        (avg_buying_pressure_1 / avg_true_range_1 * 4)
        + (avg_buying_pressure_2 / avg_true_range_2 * 2)
        + (avg_buying_pressure_3 / avg_true_range_3)
    ) / (4 + 2 + 1)

    return ultimate_oscillator


def get_percentage_price_oscillator(
    prices_close: pd.Series, short_window: int, long_window: int
) -> pd.Series:
    """
    Calculate the Percentage Price Oscillator (PPO) for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        short_window (int): Number of periods for the short-term EMA.
        long_window (int): Number of periods for the long-term EMA.

    Returns:
        pd.Series: PPO values.
    """
    short_ema = get_exponential_moving_average(prices_close, short_window)
    long_ema = get_exponential_moving_average(prices_close, long_window)

    ppo = ((short_ema - long_ema) / long_ema) * 100
    return ppo


def get_detrended_price_oscillator(
    prices_close: pd.Series,
    window: int,
    moving_average_type: str = "sma",
) -> pd.Series:
    """
    Calculate the Detrended Price Oscillator (DPO) for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for DPO calculation.

    Returns:
        pd.Series: DPO values.
    """
    if moving_average_type == "sma":
        moving_average = get_moving_average(prices_close, window)
    elif moving_average_type == "ema":
        moving_average = get_exponential_moving_average(prices_close, window)
    else:
        raise ValueError("Invalid moving average type. Choose either 'sma' or 'ema'.")

    dpo = prices_close.shift(int(window / 2)) - moving_average.shift(int(window / 2))

    return dpo


def get_average_directional_index(
    prices_high: pd.Series, prices_low: pd.Series, prices_close: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Average Directional Movement Index (ADX) of a given price series.

    The Average Directional Movement Index (ADX) is a technical indicator used to quantify
    the strength of a trend. It combines the information from the Plus Directional Indicator (+DI)
    and Minus Directional Indicator (-DI) to provide a single value that represents the trend's strength.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods to consider for ADX calculation.

    Returns:
        pd.Series: ADX values.
    """
    high_low = prices_high - prices_low
    high_close_prev = abs(prices_high - prices_close.shift(1))
    low_close_prev = abs(prices_low - prices_close.shift(1))

    tr = pd.DataFrame(
        {
            "high_low": high_low,
            "high_close_prev": high_close_prev,
            "low_close_prev": low_close_prev,
        },
        index=prices_high.index,
    )

    tr["true_range"] = tr[["high_low", "high_close_prev", "low_close_prev"]].max(axis=1)

    plus_dm = prices_high.diff().apply(lambda x: x if x > 0 else 0)
    minus_dm = -prices_low.diff().apply(lambda x: x if x < 0 else 0)

    atr = tr["true_range"].rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

    adx = dx.rolling(window=window).mean()

    return adx


def get_chande_momentum_oscillator(prices_close: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Chande Momentum Oscillator (CMO) for a given price series.

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for CMO calculation.

    Returns:
        pd.Series: CMO values.
    """
    price_diff = prices_close.diff(1)

    up_sum = price_diff.where(price_diff > 0, 0).rolling(window=window).sum()
    down_sum = abs(price_diff.where(price_diff < 0, 0)).rolling(window=window).sum()

    cmo = ((up_sum - down_sum) / (up_sum + down_sum)) * 100
    return cmo


def get_ichimoku_cloud(
    prices_high: pd.Series,
    prices_low: pd.Series,
    conversion_window: int,
    base_window: int,
    lead_span_b_window: int,
) -> pd.DataFrame:
    """
    Calculate the Ichimoku Cloud components for a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        conversion_window (int): Number of periods for the conversion line.
        base_window (int): Number of periods for the base line.
        lead_span_b_window (int): Number of periods for the lead span B.
        lead_span_b_shift (int): Number of periods to shift lead span B.

    Returns:
        pd.DataFrame: Ichimoku Cloud components (Conversion Line, Base Line, Lead Span A, Lead Span B).
    """
    conversion_line = (
        prices_high.rolling(window=conversion_window).max()
        + prices_low.rolling(window=conversion_window).min()
    ) / 2
    base_line = (
        prices_high.rolling(window=base_window).max()
        + prices_low.rolling(window=base_window).min()
    ) / 2
    lead_span_a = ((conversion_line + base_line) / 2).shift(conversion_window)
    lead_span_b = (
        (
            prices_high.rolling(window=lead_span_b_window).max()
            + prices_low.rolling(window=lead_span_b_window).min()
        )
        / 2
    ).shift(conversion_window)

    return pd.concat(
        [conversion_line, base_line, lead_span_a, lead_span_b],
        keys=["Conversion Line", "Base Line", "Leading Span A", "Leading Span B"],
        axis=1,
    )


def get_stochastic_oscillator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    window: int,
    smooth_window: int,
) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator of a given price series.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for the stochastic calculation.
        smooth_window (int): Number of periods for smoothing the %K values.

    Returns:
        pd.DataFrame: Stochastic Oscillator (%K and %D).
    """
    lowest_low = prices_low.rolling(window=window).min()
    highest_high = prices_high.rolling(window=window).max()

    stochastic_k = ((prices_close - lowest_low) / (highest_high - lowest_low)) * 100
    stochastic_d = stochastic_k.rolling(window=smooth_window).mean()

    return pd.concat(
        [stochastic_k, stochastic_d], keys=["Stochastic %K", "Stochastic %D"], axis=1
    )


def get_moving_average_convergence_divergence(
    prices: pd.Series, short_window: int, long_window: int, signal_window: int
) -> pd.Series:
    """
    Calculate the Moving Average Convergence Divergence (MACD) of a given price series.

    Args:
        prices (pd.Series): Series of prices.
        short_window (int): Number of periods for the short-term moving average.
        long_window (int): Number of periods for the long-term moving average.
        signal_window (int): Number of periods for the signal line moving average.

    Returns:
        pd.Series: MACD values.
    """
    short_ema = prices.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()

    return pd.concat(
        [macd_line, signal_line], keys=["MACD Line", "Signal Line"], axis=1
    )


def get_relative_strength_index(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) of a given price series.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    # Calculate price changes
    price_diff = prices.diff(1)

    # Calculate upward and downward price changes
    up_changes = price_diff.where(price_diff > 0, 0)
    down_changes = -price_diff.where(price_diff < 0, 0)

    # Calculate average gains and losses over the specified window
    avg_gain = up_changes.rolling(window=window).mean()
    avg_loss = down_changes.rolling(window=window).mean()

    # Calculate the relative strength index
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_balance_of_power(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
) -> pd.Series:
    """
    Calculate the Balance of Power (BOP) indicator for a given price series.

    The Balance of Power (BOP) indicator measures the strength of buyers and sellers in the market.
    It considers the relationship between the closing price and the trading range (high - low) of each period.
    BOP values above zero suggest bullish buying pressure, while values below zero suggest bearish selling pressure.

    Args:
        prices_open (pd.Series): Series of opening prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.

    Returns:
        pd.Series: BOP values.
    """
    bop_values = ((prices_close - prices_open) / (prices_high - prices_low)).fillna(0)

    return bop_values
