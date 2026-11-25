"""Overlap Module"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import gammaln

from financetoolkit.technicals.helpers import apply_variable_smoothing
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option


def get_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA): the average price over a window, which smooths the prices to show
    their trend.

    The Simple Moving Average is the unweighted arithmetic mean of price over a trailing
    window, used both as a standalone trend-following signal (e.g. price crossing above or
    below the average) and as a building block for many other indicators in this module.

    The formula is as follows:

    - SMA = Mean(Close, window)

    Also known as: SMA, moving average.

    Reference: The academic literature testing moving-average trading rules (rather than
    defining the SMA itself, which predates any single paper) includes Brock, W., Lakonishok,
    J., & LeBaron, B. (1992). "Simple Technical Trading Rules and the Stochastic Properties of
    Stock Returns." Journal of Finance, 47(5), 1731-1764.

    Referenced against:
    - TA-Lib, SMA: https://ta-lib.org/functions/sma.html

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for the moving average.

    Returns:
        pd.Series: Moving Average values.
    """
    return prices.rolling(window=window).mean()


def get_exponential_moving_average(
    prices: pd.Series | pd.DataFrame, window: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Exponential Moving Average (EMA): an average that weights recent prices more, so it follows the
    trend of the prices with less lag.

    The Exponential Moving Average weights recent observations more heavily than older
    ones, with the weight decaying exponentially the further back in time an observation
    is, so that it reacts to new price information faster than a Simple Moving Average of
    the same window while still smoothing out noise.

    The formula is as follows:

    - Smoothing Factor (alpha) = 2 / (window + 1)
    - First EMA = SMA(Close, window), on the window-th period
    - EMA(t) = alpha * Close(t) + (1 - alpha) * EMA(t-1)

    Also known as: EMA.

    Reference: The exponential smoothing technique underlying the EMA originates in Brown,
    R.G. (1956). "Exponential Smoothing for Predicting Demand." Arthur D. Little Inc.; the
    standard technical-analysis treatment is Murphy, J.J. (1999). "Technical Analysis of the
    Financial Markets." New York Institute of Finance.

    Referenced against:
    - TA-Lib, EMA: https://ta-lib.org/functions/ema.html

    Notes:
        - The recursion is seeded with the Simple Moving Average of the first `window`
          values, placed on the last of them, and the periods before it are missing. This
          is the convention of TA-Lib's EMA and StockCharts. Seeding with the first
          price instead gives every early value a weight on that single price that a
          `window`-period average does not have.
        - A missing value has no average: the period is missing, and the recursion
          continues from the last average on the next value, as with Wilder's smoothing.
          A series that ends early, such as a delisted ticker's, therefore stops where
          its prices stop instead of repeating its last average.
        - A DataFrame is smoothed per column, each column seeded on its own first
          `window` values, so tickers with a later first price get their own warm-up.

    Args:
        prices (pd.Series | pd.DataFrame): Series (or DataFrame with one column per
            ticker) of prices.
        window (int): Number of periods to consider for EMA calculation.

    Returns:
        pd.Series | pd.DataFrame: EMA values, missing for the first `window` - 1 periods.
    """
    # The first full window marks the seed; every period before it stays missing.
    simple_moving_average = prices.rolling(window=window).mean()
    has_seed = simple_moving_average.notna().astype(bool).cummax()
    seed_period = has_seed & ~has_seed.shift(1, fill_value=False).astype(bool)

    # The seed replaces the price on its own period, after which ewm with adjust=False
    # runs the plain recursion EMA(t) = alpha * Price(t) + (1 - alpha) * EMA(t-1).
    seeded_prices = prices.where(has_seed).mask(seed_period, simple_moving_average)
    exponential_moving_average = seeded_prices.ewm(
        span=window, adjust=False, ignore_na=True
    ).mean()

    return exponential_moving_average.where(has_seed & prices.notna())


def get_double_exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Double Exponential Moving Average (DEMA): two exponential moving averages combined to remove most
    of their lag.

    DEMA combines a single and a double-smoothed EMA to reduce the lag inherent in moving
    averages while retaining most of their smoothing benefit. Because the second EMA lags
    the first, subtracting it out (after doubling the first) removes much of the delay a
    plain EMA of the same window would have.

    The formula is as follows:

    - EMA1 = EMA(Close, window)
    - EMA2 = EMA(EMA1, window)
    - DEMA = 2 * EMA1 - EMA2

    Also known as: DEMA.

    Reference: Mulloy, P.G. (1994). "Smoothing Data with Faster Moving Averages." Technical
    Analysis of Stocks & Commodities, 12(1).

    Referenced against:
    - TA-Lib, DEMA: https://ta-lib.org/functions/dema.html

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for DEMA calculation.

    Returns:
        pd.Series: DEMA values.
    """
    # The second EMA is seeded on the first `window` values of the first, so DEMA starts
    # at period 2 * (window - 1), as TA-Lib's DEMA does.
    ema_first = get_exponential_moving_average(prices, window)
    ema_second = get_exponential_moving_average(ema_first, window)
    dema = 2 * ema_first - ema_second

    return dema


def get_trix(
    prices_close: pd.Series, window: int, signal_window: int | None = None
) -> pd.Series | pd.DataFrame:
    """
    Calculate the TRIX: the rate of change of a triple exponential moving average, which shows momentum without the
    noise of the shortest moves.

    Trix applies an Exponential Moving Average three times in succession (triple smoothing)
    to filter out short-term price fluctuations and insignificant cycles, then plots the
    percentage rate of change of that triple-smoothed series. Because insignificant
    fluctuations have already been smoothed away, the result oscillates around zero mainly
    in response to genuine trend changes. Hutson read crossings of the TRIX and its signal
    line, a moving average of the TRIX, as signals; the histogram is their difference.

    The formula is as follows:

    - EMA1 = EMA(Close, window)
    - EMA2 = EMA(EMA1, window)
    - EMA3 = EMA(EMA2, window)
    - Trix = (EMA3(t) - EMA3(t-1)) / EMA3(t-1)
    - Signal Line = EMA(Trix, signal_window)
    - Histogram = Trix - Signal Line

    Also known as: TRIX, Triple Exponential Average, TRIXH, TRIX histogram.

    Reference: Hutson, J.K. (1983). "Good TRIX." Technical Analysis of Stocks & Commodities,
    1(5), 105-108.

    Referenced against:
    - TA-Lib, TRIX: https://ta-lib.org/functions/trix.html

    Shares similar logic:
    - pandas-ta-classic, trix: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/trix.py#L18
    - pandas-ta-classic, trixh: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/trixh.py#L12

    Notes:
        - The signal line is an EMA seeded with the simple average of its first
          `signal_window` values, as every EMA of the Toolkit. pandas-ta-classic's trixh
          takes a simple moving average of the TRIX as its signal line.

    Args:
        prices_close (pd.Series): Series of closing prices.
        window (int): Number of periods for Trix calculation.
        signal_window (int | None): Number of periods of the signal line. Defaults to
            None, which returns only the Trix.

    Returns:
        pd.Series | pd.DataFrame: Trix Indicator values as a decimal (0.001 is 0.1%), or
        with a `signal_window` the TRIX, Signal Line and Histogram as columns.
    """
    ema_1 = get_exponential_moving_average(prices_close, window)
    ema_2 = get_exponential_moving_average(ema_1, window)
    ema_3 = get_exponential_moving_average(ema_2, window)

    trix = (ema_3 - ema_3.shift(1)) / ema_3.shift(1)

    if signal_window is None:
        return trix

    signal_line = get_exponential_moving_average(trix, signal_window)

    return pd.concat(
        [trix, signal_line, trix - signal_line],
        keys=["TRIX", "Signal Line", "Histogram"],
        axis=1,
    )


def get_triangular_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Triangular Moving Average (TRIMA): an average of an average, which weights the middle of the
    window most and gives a smoother trend.

    The Triangular Moving Average is a doubly-smoothed Simple Moving Average: taking an SMA
    of an SMA produces a triangular weighting scheme where the middle observations of the
    combined window carry the most weight and the oldest/newest observations carry the
    least, which reduces the impact of outliers and short-term fluctuations more than a
    single SMA of the same overall length.

    The formula is as follows:

    - For an odd window: Sub-window Length = (window + 1) / 2, applied for both passes.
    - For an even window: the two passes use different sub-window lengths, window / 2
      and window / 2 + 1 (matching TA-Lib's TRIMA convention).
    - TRIMA = SMA(SMA(Close, Sub-window Length 1), Sub-window Length 2)

    Also known as: TRIMA.

    Reference: The standard textbook treatment is Colby, R.W. (2003). "The Encyclopedia of
    Technical Market Indicators." 2nd ed. McGraw-Hill.

    Referenced against:
    - TA-Lib, TRIMA: https://ta-lib.org/functions/trima.html
    - pandas-ta-classic, swma: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/swma.py#L18

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for TRIMA calculation.

    Returns:
        pd.Series: TRIMA values.
    """
    if window % 2 == 1:
        sub_window_first = sub_window_second = max((window + 1) // 2, 1)
    else:
        sub_window_first = max(window // 2, 1)
        sub_window_second = sub_window_first + 1

    # Both passes need full windows, so the first value is the full triangular average at
    # period window - 1 (as in TA-Lib) rather than an average over fewer prices.
    first_pass = prices.rolling(window=sub_window_first).mean()
    tri_ma = first_pass.rolling(window=sub_window_second).mean()

    return tri_ma


def get_weighted_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Weighted Moving Average (WMA): an average that weights the prices linearly, the latest most, so it
    follows the trend with less lag.

    The Weighted Moving Average assigns a linearly increasing weight to more recent
    prices within the window, making it more responsive to recent price changes than
    a Simple Moving Average while remaining smoother than an Exponential Moving Average.

    The formula is as follows:

    - WMA = Sum(Price(i) * i, i = 1..window) / Sum(i, i = 1..window), where i = 1 is the
      oldest price in the window and i = window is the most recent

    Also known as: WMA, linearly weighted moving average.

    Reference: The standard textbook treatment is Colby, R.W. (2003). "The Encyclopedia of
    Technical Market Indicators." 2nd ed. McGraw-Hill.

    Referenced against:
    - TA-Lib, WMA: https://ta-lib.org/functions/wma.html

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for the WMA calculation.

    Returns:
        pd.Series: WMA values.
    """
    weights = np.arange(1, window + 1, dtype=float)

    # One matrix product over every window at once instead of a Python call per window.
    # A window holding a missing price is missing, as with rolling().apply().
    return apply_to_rolling_windows(
        prices, window, lambda windows: windows @ weights / weights.sum()
    )


def apply_to_rolling_windows(
    data: pd.Series | pd.DataFrame, window: int, calculation
) -> pd.Series | pd.DataFrame:
    """
    Apply a calculation to every rolling window of a Series or of each DataFrame column: the building block of the
    moving averages with their own weights.

    Args:
        data (pd.Series | pd.DataFrame): The values, with time as index.
        window (int): The number of values in each window.
        calculation (Callable): Takes an array of windows, with the values of each window
            along the last axis, and returns one value per window.

    Returns:
        pd.Series | pd.DataFrame: The result, missing for the first window - 1 rows.
    """
    values = data.to_numpy(dtype=float)
    result = np.full(values.shape, np.nan)

    if window >= 1 and len(values) >= window:
        result[window - 1 :] = calculation(sliding_window_view(values, window, axis=0))

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(result, index=data.index, columns=data.columns)

    return pd.Series(result, index=data.index, name=data.name)


def get_kaufman_adaptive_moving_average(
    prices: pd.Series,
    window: int = 10,
    short_window: int = 2,
    long_window: int = 30,
) -> pd.Series:
    """
    Calculate the Kaufman Adaptive Moving Average (KAMA): an average that moves fast in a trend and slowly in noise,
    by its Efficiency Ratio.

    The Kaufman Adaptive Moving Average adjusts its own responsiveness to price changes
    based on how "efficiently" price is moving. It compares the net directional move over
    the window to the total (sum of absolute) movement over that same window — the
    Efficiency Ratio. When price trends strongly in one direction (an efficient move), the
    Efficiency Ratio is close to 1 and KAMA tracks price closely, behaving like a fast EMA.
    When price whipsaws sideways (an inefficient move), the Efficiency Ratio is close to 0
    and KAMA flattens out, behaving like a slow EMA — reducing whipsaw signals in choppy
    markets while still reacting quickly during strong trends.

    The formula is as follows:

    - Change = |Close(t) - Close(t - window)|
    - Volatility = Sum(|Close(i) - Close(i - 1)|, window)
    - Efficiency Ratio (ER) = Change / Volatility, and 1 when Volatility ≤ Change (flat prices)
    - Fastest SC = 2 / (short_window + 1), Slowest SC = 2 / (long_window + 1)
    - Smoothing Constant (SC) = [ER * (Fastest SC - Slowest SC) + Slowest SC]^2
    - KAMA(t) = KAMA(t-1) + SC * (Close(t) - KAMA(t-1))

    Also known as: KAMA, Kaufman's Adaptive Moving Average.

    Reference: Kaufman, P.J. (1998). "Trading Systems and Methods." 3rd ed. Wiley.

    Referenced against:
    - TA-Lib, KAMA: https://ta-lib.org/functions/kama.html

    Notes:
        - The recursion is seeded with the closing price of the period before the first
          period where the Efficiency Ratio is defined, and the first KAMA value is one
          update of that seed, the convention of TA-Lib's KAMA. StockCharts instead describes seeding with a Simple Moving
          Average; the two conventions differ only in a transient that decays geometrically
          over the periods that follow, and neither affects the steady-state values.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods over which the Efficiency Ratio is calculated.
            Defaults to 10.
        short_window (int): The number of periods that corresponds to the fastest EMA
            constant used when the Efficiency Ratio is at its maximum (1.0). Defaults to 2.
        long_window (int): The number of periods that corresponds to the slowest EMA
            constant used when the Efficiency Ratio is at its minimum (0.0). Defaults to 30.

    Returns:
        pd.Series: KAMA values.

    Raises:
        TypeError: If `prices` is not a pandas Series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")

    change = (prices - prices.shift(window)).abs()
    volatility = prices.diff().abs().rolling(window=window).sum()
    # When the path is no longer than the net move (only possible with equal values, as on
    # flat prices where both are 0) the move is fully efficient and the ratio is 1, as in
    # TA-Lib's KAMA. Dividing would give 0 / 0 on flat prices and
    # stall the average; the ratio stays NaN only where prices are missing.  # noqa: E501
    efficiency_ratio = (change / volatility).where(volatility > change, 1.0)
    efficiency_ratio = efficiency_ratio.where(change.notna() & volatility.notna())

    fastest_smoothing_constant = 2 / (short_window + 1)
    slowest_smoothing_constant = 2 / (long_window + 1)

    smoothing_constant = (
        efficiency_ratio * (fastest_smoothing_constant - slowest_smoothing_constant)
        + slowest_smoothing_constant
    ) ** 2

    kama = pd.Series(index=prices.index, dtype="float64")

    length = len(prices)
    if length == 0:
        return kama

    first_valid_index = smoothing_constant.first_valid_index()
    if first_valid_index is None:
        return kama

    start_position = prices.index.get_loc(first_valid_index)

    # The recursion runs on plain Python lists, since writing a pandas Series one element
    # at a time costs tens of microseconds per element.
    price_values = prices.to_numpy(dtype="float64").tolist()
    constants = smoothing_constant.to_numpy(dtype="float64").tolist()
    averages = [math.nan] * length

    # The Efficiency Ratio needs the previous `window` changes, so the price before the first defined ratio always exists and seeds the recursion; the first output is already one update of it, as in TA-Lib.  # noqa: E501
    previous_average = price_values[start_position - 1]

    for i in range(start_position, length):
        smoothing_constant_value = constants[i]
        if not math.isnan(smoothing_constant_value):
            previous_average = previous_average + smoothing_constant_value * (
                price_values[i] - previous_average
            )

        averages[i] = previous_average

    # A period without a price has no average, while the recursion carries on from the
    # last one, so a series that ends early stops instead of repeating its last value.
    return pd.Series(averages, index=prices.index, dtype="float64").where(
        prices.notna()
    )


def get_hull_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Hull Moving Average (HMA): weighted moving averages combined so the trend line is both smooth and
    quick to respond.

    The Hull Moving Average reduces the lag typically associated with moving averages
    while improving smoothing, by combining a WMA of half the window length, a WMA of
    the full window length, and a further WMA over the square root of the window length.

    The formula is as follows:

    - Raw HMA = 2 * WMA(Close, window / 2) - WMA(Close, window)
    - HMA = WMA(Raw HMA, sqrt(window))

    Also known as: HMA.

    Reference: Hull, A. (2005). "Hull Moving Average."

    Referenced against:
    - TA-Lib, HMA: https://ta-lib.org/functions/hma.html

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider for the HMA calculation.

    Returns:
        pd.Series: HMA values.
    """
    half_window_wma = get_weighted_moving_average(prices, max(window // 2, 1))
    full_window_wma = get_weighted_moving_average(prices, window)

    raw_hma = 2 * half_window_wma - full_window_wma

    return get_weighted_moving_average(raw_hma, max(int(np.sqrt(window)), 1))


def get_triple_exponential_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Triple Exponential Moving Average (TEMA): three exponential moving averages combined to remove
    most of their lag.

    TEMA takes the lag out of an Exponential Moving Average further than DEMA does: it
    combines a single, a double and a triple smoothed EMA so that the lag of the single
    EMA is cancelled by the lag of the smoother ones, while the result stays smooth.

    The formula is as follows:

    - EMA1 = EMA(Close, window)
    - EMA2 = EMA(EMA1, window)
    - EMA3 = EMA(EMA2, window)
    - TEMA = 3 * EMA1 - 3 * EMA2 + EMA3

    Also known as: TEMA.

    Reference: Mulloy, P.G. (1994). "Smoothing Data with Less Lag." Technical Analysis of
    Stocks & Commodities, 12(2).

    Referenced against:
    - TA-Lib, TEMA: https://ta-lib.org/functions/tema.html

    Notes:
        - Each EMA is seeded with the simple average of its first `window` values, so TEMA
          starts at period 3 * (window - 1), as in TA-Lib's TEMA.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of each of the three EMAs.

    Returns:
        pd.Series: TEMA values.
    """
    ema_first = get_exponential_moving_average(prices, window)
    ema_second = get_exponential_moving_average(ema_first, window)
    ema_third = get_exponential_moving_average(ema_second, window)

    return 3 * ema_first - 3 * ema_second + ema_third


def get_t3_moving_average(
    prices: pd.Series, window: int = 5, volume_factor: float = 0.7
) -> pd.Series:
    """
    Calculate the Tillson T3 Moving Average: six exponential moving averages combined into a smooth trend line with
    little lag.

    The T3 applies Tillson's Generalized DEMA, GD(x) = (1 + a) * EMA(x) - a * EMA(EMA(x)),
    three times in a row. The volume factor a sets how much of the lag correction of DEMA is
    used: 0 gives a plain EMA of an EMA of an EMA, 1 gives three DEMAs in a row. The result
    is a smooth average with little lag and little overshoot.

    The formula is as follows:

    - E1 = EMA(Close, window), E2 = EMA(E1, window), ..., E6 = EMA(E5, window)
    - c1 = -a^3, c2 = 3a^2 + 3a^3, c3 = -6a^2 - 3a - 3a^3, c4 = 1 + 3a + a^3 + 3a^2
    - T3 = c1 * E6 + c2 * E5 + c3 * E4 + c4 * E3

    Also known as: T3, Tillson T3.

    Reference: Tillson, T. (1998). "Smoothing Techniques for More Accurate Signals."
    Technical Analysis of Stocks & Commodities, 16(1).

    Referenced against:
    - TA-Lib, T3: https://ta-lib.org/functions/t3.html
    - pandas-ta-classic, t3: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/t3.py#L14

    Notes:
        - Each of the six EMAs is seeded with the simple average of its first `window`
          values, so T3 starts at period 6 * (window - 1), as in TA-Lib's T3.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of each of the six EMAs. Defaults to 5.
        volume_factor (float): The volume factor a, between 0 and 1. Defaults to 0.7.

    Returns:
        pd.Series: T3 values.
    """
    emas = [prices]
    for _ in range(6):
        emas.append(get_exponential_moving_average(emas[-1], window))

    a = volume_factor
    c1 = -(a**3)
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = 1 + 3 * a + a**3 + 3 * a**2

    return c1 * emas[6] + c2 * emas[5] + c3 * emas[4] + c4 * emas[3]


def get_zero_lag_exponential_moving_average(
    prices: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Zero Lag Exponential Moving Average (ZLEMA): an exponential moving average of the prices corrected
    for their lag, so it follows the trend more closely.

    An EMA lags the price by about (window - 1) / 2 periods. ZLEMA adds the change over
    that lag to the price before averaging, so the average is taken of a price that is
    extrapolated by its own lag, which takes most of the lag out of the EMA.

    The formula is as follows:

    - Lag = (window - 1) / 2, rounded down
    - De-lagged Close = Close + (Close - Close(t - Lag)) = 2 * Close - Close(t - Lag)
    - ZLEMA = EMA(De-lagged Close, window)

    Also known as: ZLEMA, ZLMA, zero lag moving average.

    Reference: Ehlers, J.F. & Way, R. (2010). "Zero Lag (Well, Almost)." Technical Analysis
    of Stocks & Commodities, November 2010.

    Referenced against:
    - TA-Lib, ZLEMA: https://ta-lib.org/functions/zlema.html

    Notes:
        - The EMA is seeded with the simple average of the first `window` de-lagged prices,
          so ZLEMA starts at period Lag + window - 1, as in TA-Lib.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the EMA.

    Returns:
        pd.Series: ZLEMA values.
    """
    lag = (window - 1) // 2

    return get_exponential_moving_average(2 * prices - prices.shift(lag), window)


def get_arnaud_legoux_moving_average(
    prices: pd.Series, window: int = 9, sigma: float = 6.0, offset: float = 0.85
) -> pd.Series:
    """
    Calculate the Arnaud Legoux Moving Average (ALMA): an average with Gaussian weights centred near the latest
    prices, which is both smooth and responsive.

    ALMA weights the prices in the window with a Gaussian curve whose peak sits at `offset`
    of the way from the oldest to the latest price. An offset close to 1 puts the weight on
    recent prices (less lag), close to 0 on older ones (more smoothing), and `sigma` sets
    how narrow the curve is.

    The formula is as follows:

    - m = floor(offset * (window - 1)), s = window / sigma
    - Weight(i) = exp(-(i - m)^2 / (2 * s^2)), for i = 0 (the oldest price) to window - 1
    - ALMA = Sum(Weight(i) * Price(i)) / Sum(Weight(i))

    Also known as: ALMA.

    Reference: Legoux, A. & Kouzis-Loukas, D. (2009). "ALMA: Arnaud Legoux Moving
    Average." arnaudlegoux.com.

    Referenced against:
    - TA-Lib, ALMA: https://ta-lib.org/functions/alma.html
    - pandas-ta-classic, alma: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/alma.py#L12

    Notes:
        - The peak of the curve m is rounded down to a whole period, as in TA-Lib.
          pandas-ta-classic's alma (and TradingView's ta.alma by default) keep the
          unrounded m, which moves the weights by a fraction of a period.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider. Defaults to 9.
        sigma (float): The width of the Gaussian curve, as the number of standard
            deviations that fit in the window. Defaults to 6.0.
        offset (float): The position of the peak of the curve in the window, from 0
            (the oldest price) to 1 (the latest price). Defaults to 0.85.

    Returns:
        pd.Series: ALMA values.
    """
    peak = math.floor(offset * (window - 1))
    width = window / sigma
    weights = np.exp(-((np.arange(window) - peak) ** 2) / (2 * width**2))

    return apply_to_rolling_windows(
        prices, window, lambda windows: windows @ weights / weights.sum()
    )


def get_volume_weighted_moving_average(
    prices: pd.Series, volumes: pd.Series, window: int
) -> pd.Series:
    """
    Calculate the Volume Weighted Moving Average (VWMA): the average price over a window weighted by volume, so the
    prices of busy periods count more.

    The VWMA weights every closing price in the window by the volume traded in its period,
    so prices at which much was traded count more than those of quiet periods. It differs
    from the rolling VWAP only in the price that is averaged: the close instead of the
    typical price.

    The formula is as follows:

    - VWMA = Sum(Close * Volume, window) / Sum(Volume, window)

    Also known as: VWMA.

    Reference: Dormeier, B.P. (2011). "Investing with Volume Analysis." FT Press.

    Referenced against:
    - TA-Lib, VWMA: https://ta-lib.org/functions/vwma.html
    - pandas-ta-classic, vwma: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/vwma.py#L13

    Shares similar logic:
    - Tulip Indicators, vwma: https://tulipindicators.org/vwma

    Args:
        prices (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods to consider.

    Returns:
        pd.Series: VWMA values, missing for a window without volume.
    """
    total_volume = volumes.rolling(window=window).sum()

    return (prices * volumes).rolling(window=window).sum() / total_volume.where(
        total_volume != 0
    )


def get_mcginley_dynamic(
    prices: pd.Series, window: int = 14, constant: float = 1.0
) -> pd.Series:
    """
    Calculate the McGinley Dynamic: an average that speeds up when the price moves away from it and slows down when
    it does not, so it follows the price with fewer whipsaws.

    The McGinley Dynamic is a moving average that adjusts its own speed to the market: the
    step towards the price is divided by (Price / MD)^4, so the average moves slowly when
    the price runs above it (rising markets) and quickly when the price falls below it,
    hugging the price more closely than an average with a fixed speed.

    The formula is as follows:

    - First MD = SMA(Close, window), on the window-th period
    - MD(t) = MD(t-1) + (Close(t) - MD(t-1)) / (constant * window * (Close(t) / MD(t-1))^4)

    Also known as: MCGD, McGinley Dynamic Indicator.

    Reference: McGinley, J.R. (1997). "The McGinley Dynamic." Journal of Technical Analysis
    (Market Technicians Association).

    Referenced against:
    - TA-Lib, MCGD: https://ta-lib.org/functions/mcgd.html
    - pandas-ta-classic, mcgd: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/mcgd.py#L28

    Notes:
        - The recursion is seeded with the simple average of the first `window` prices, as
          the Toolkit seeds its exponential moving averages. TA-Lib's MCGD and
          pandas-ta-classic's mcgd start the recursion from the first price instead; the
          difference fades as the seed's weight decays, by about a factor 1 - 1 / window
          per period.
        - McGinley suggested a constant of 0.6 to make the average follow the price more
          closely; TA-Lib and pandas-ta-classic use 1.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods, the N of the formula. Defaults to 14.
        constant (float): The constant k that multiplies the window. Defaults to 1.0.

    Returns:
        pd.Series: McGinley Dynamic values.
    """
    simple_moving_average = prices.rolling(window=window).mean()
    dynamic = pd.Series(index=prices.index, dtype="float64")

    first_valid_index = simple_moving_average.first_valid_index()
    if first_valid_index is None:
        return dynamic

    start_position = prices.index.get_loc(first_valid_index)
    previous_dynamic = float(simple_moving_average.iloc[start_position])
    price_values = prices.to_numpy(dtype="float64").tolist()
    averages = [math.nan] * len(prices)
    averages[start_position] = previous_dynamic

    # The recursion runs on plain Python lists, since writing a pandas Series one element
    # at a time costs tens of microseconds per element.
    for i in range(start_position + 1, len(prices)):
        price = price_values[i]
        if not math.isnan(price) and previous_dynamic != 0:
            previous_dynamic = previous_dynamic + (price - previous_dynamic) / (
                constant * window * (price / previous_dynamic) ** 4
            )
        averages[i] = previous_dynamic

    # A period without a price has no average, while the recursion carries on from the
    # last one, so a series that ends early stops instead of repeating its last value.
    return pd.Series(averages, index=prices.index, dtype="float64").where(
        prices.notna()
    )


def get_variable_index_dynamic_average(
    prices: pd.Series, window: int = 12, cmo_window: int = 9
) -> pd.Series:
    """
    Calculate the Variable Index Dynamic Average (VIDYA): an exponential moving average whose speed follows the
    momentum of the prices.

    VIDYA is an exponential moving average whose smoothing factor is scaled by the
    strength of the trend, measured by the absolute Chande Momentum Oscillator: in a strong
    trend (|CMO| near 1) it moves like an EMA of `window` periods, in a market that goes
    nowhere (|CMO| near 0) it hardly moves at all.

    The formula is as follows:

    - Gains = Sum(max(Close(t) - Close(t-1), 0), cmo_window)
    - Losses = Sum(max(Close(t-1) - Close(t), 0), cmo_window)
    - |CMO| = |Gains - Losses| / (Gains + Losses), and 0 when prices did not move
    - Alpha = 2 / (window + 1)
    - VIDYA(t) = Alpha * |CMO(t)| * Close(t) + (1 - Alpha * |CMO(t)|) * VIDYA(t-1)

    Also known as: VIDYA, Chande's Variable Index Dynamic Average.

    For more information, see the following sources:

    - Chande, T.S. (1992). "Adapting Moving Averages to Market Volatility." Technical
      Analysis of Stocks & Commodities, 10(3).
    - Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    Referenced against:
    - TA-Lib, VIDYA: https://ta-lib.org/functions/vidya.html
    - pandas-ta-classic, vidya: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/vidya.py#L27

    Notes:
        - The volatility index is the absolute Chande Momentum Oscillator over unsmoothed
          sums (Chande & Kroll, 1994), as in TA-Lib and pandas-ta-classic. Chande's 1992
          article and Tulip Indicators' vidya use the ratio of a short to a long standard
          deviation instead.
        - The recursion is seeded with the simple average of `window` prices on the period
          before the first oscillator value that follows a full window, as the Toolkit
          seeds its exponential moving averages and as pandas-ta-classic does. TA-Lib starts
          the recursion from the first price, with the oscillator over the changes available
          so far, and shows it from period `cmo_window` on; the difference fades by a factor
          1 - Alpha * |CMO| per period.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods of the exponential moving average. Defaults to 12.
        cmo_window (int): Number of periods of the Chande Momentum Oscillator. Defaults
            to 9.

    Returns:
        pd.Series: VIDYA values.
    """
    change = prices.diff()
    gains = change.clip(lower=0).rolling(window=cmo_window).sum()
    losses = (-change).clip(lower=0).rolling(window=cmo_window).sum()
    movement = gains + losses

    # Without any movement the oscillator is 0 / 0, and an average that does not move
    # follows a market that does not move either, so the factor is 0 there.
    absolute_cmo = ((gains - losses) / movement.where(movement != 0)).abs()
    absolute_cmo = absolute_cmo.where(movement != 0, 0.0).where(movement.notna())

    return apply_variable_smoothing(prices, 2 / (window + 1) * absolute_cmo, window)


def get_fractal_adaptive_moving_average(
    prices_high: pd.Series, prices_low: pd.Series, window: int = 16
) -> pd.Series:
    """
    Calculate the Fractal Adaptive Moving Average (FRAMA): an average whose speed follows the fractal dimension of
    the prices, fast in a trend and slow in a range.

    FRAMA is an exponential moving average of the median price whose smoothing factor
    follows the fractal dimension of the prices over the window. The dimension compares the
    ranges of the two halves of the window with the range of the whole: in a trend the
    ranges of the halves add up to the range of the whole (dimension 1) and FRAMA follows
    the price closely, in a range-bound market each half covers most of the range
    (dimension towards 2) and FRAMA barely moves.

    The formula is as follows:

    - Price = (High + Low) / 2
    - N1 = (Highest High - Lowest Low) / (window / 2) over the latest window / 2 periods
    - N2 = (Highest High - Lowest Low) / (window / 2) over the window / 2 periods before
    - N3 = (Highest High - Lowest Low) / window over the whole window
    - D = (log(N1 + N2) - log(N3)) / log(2)
    - Alpha = exp(-4.6 * (D - 1)), limited to between 0.01 and 1
    - FRAMA(t) = Alpha(t) * Price(t) + (1 - Alpha(t)) * FRAMA(t-1)

    Also known as: FRAMA.

    Reference: Ehlers, J.F. (2005). "Fractal Adaptive Moving Averages." Technical Analysis of
    Stocks & Commodities, 23(10), 81-82.

    Referenced against:
    - TA-Lib, FRAMA: https://ta-lib.org/functions/frama.html
    - finta, FRAMA: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L463

    Notes:
        - The recursion is seeded with the simple average of the first `window` median
          prices, on the window-th period, as the Toolkit seeds its exponential moving
          averages. Ehlers and TA-Lib start from the median price of that period, which
          TA-Lib does not show, so TA-Lib's FRAMA starts one period later; the
          difference fades by a factor 1 - Alpha per period.
        - A half without a range still gives a dimension, as in TA-Lib: a flat latest half
          next to a moving earlier one is a dimension of 1, so FRAMA moves to the price.
          Ehlers' formula keeps the last dimension whenever one of the ranges is zero. Without
          any range in the window, or in both halves, the last dimension is kept.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        window (int): Number of periods over which the fractal dimension is measured, an
            even number. Defaults to 16.

    Returns:
        pd.Series: FRAMA values.

    Raises:
        ValueError: If `window` is not an even number of at least 2.
    """
    if window < 2 or window % 2:  # noqa: PLR2004
        raise ValueError("The window of FRAMA must be an even number of at least 2.")

    half = window // 2
    half_range = (
        prices_high.rolling(window=half).max() - prices_low.rolling(window=half).min()
    )
    first_half = half_range.shift(half) / half
    second_half = half_range / half
    whole = (
        prices_high.rolling(window=window).max()
        - prices_low.rolling(window=window).min()
    ) / window

    # A flat half still measures the dimension (a flat latest half next to a moving one is
    # a dimension of 1, so FRAMA jumps to the price, as in TA-Lib); only without any range
    # in the window or in both halves is it undefined and the last dimension kept.
    defined = (first_half + second_half > 0) & (whole > 0)
    dimension = (
        (
            np.log((first_half + second_half).where(defined))
            - np.log(whole.where(defined))
        )
        / np.log(2)
    ).ffill()
    alpha = np.exp(-4.6 * (dimension - 1)).clip(lower=0.01, upper=1)

    return apply_variable_smoothing((prices_high + prices_low) / 2, alpha, window)


def get_sine_weighted_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Sine Weighted Moving Average (SINWMA): an average with weights on a sine curve, which weights the
    middle of the window most.

    The weights follow the first half of a sine wave over the window, so the middle of the
    window counts most and both ends least, a rounder version of the triangular weights of
    the Triangular Moving Average.

    The formula is as follows:

    - Weight(i) = sin(i * pi / (window + 1)), for i = 1 to window
    - SINWMA = Sum(Weight(i) * Price(i)) / Sum(Weight(i))

    Also known as: SINWMA.

    Reference: Everget (2018). "Sine Weighted Moving Average." TradingView script.

    Referenced against:
    - pandas-ta-classic, sinwma: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/sinwma.py#L12

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider.

    Returns:
        pd.Series: SINWMA values.
    """
    weights = np.sin(np.arange(1, window + 1) * np.pi / (window + 1))

    return apply_to_rolling_windows(
        prices, window, lambda windows: windows @ weights / weights.sum()
    )


# The Fibonacci weight above which the weights so far are scaled down, well below the largest float.
LARGEST_FIBONACCI_WEIGHT = 1e300


def get_fibonacci_weighted_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Fibonacci Weighted Moving Average (FWMA): an average with Fibonacci numbers as weights, which
    weights the latest prices most.

    The weights are the Fibonacci numbers 1, 1, 2, 3, 5, ... from the oldest to the latest
    price, so the weight grows by about the golden ratio (1.618) per period and the latest
    prices dominate, more so than with the linear weights of the WMA.

    The formula is as follows:

    - Weight(1) = Weight(2) = 1, Weight(i) = Weight(i-1) + Weight(i-2), from the oldest
      (i = 1) to the latest price (i = window)
    - FWMA = Sum(Weight(i) * Price(i)) / Sum(Weight(i))

    Also known as: FWMA.

    Reference: defined in pandas-ta by Kevin Johnson, without a published primary source.

    Referenced against:
    - pandas-ta-classic, fwma: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/fwma.py#L17

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider.

    Returns:
        pd.Series: FWMA values.
    """
    weights = np.ones(window)
    for i in range(2, window):
        weights[i] = weights[i - 1] + weights[i - 2]
        # The numbers pass the largest float from a window of about 1,475; the recurrence is
        # linear, so scaling every weight so far keeps their ratios.
        if weights[i] > LARGEST_FIBONACCI_WEIGHT:
            weights[: i + 1] /= weights[i]

    return apply_to_rolling_windows(
        prices, window, lambda windows: windows @ weights / weights.sum()
    )


def get_pascal_weighted_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Pascal's Weighted Moving Average (PWMA): an average with the numbers of a row of Pascal's triangle as
    weights, which weights the middle of the window most.

    The weights are row window - 1 of Pascal's triangle (the binomial coefficients), a
    bell-shaped set of weights around the middle of the window that approaches a Gaussian
    curve as the window grows.

    The formula is as follows:

    - Weight(i) = (window - 1)! / (i! * (window - 1 - i)!), for i = 0 to window - 1
    - PWMA = Sum(Weight(i) * Price(i)) / Sum(Weight(i))

    Also known as: PWMA.

    Reference: defined in pandas-ta by Kevin Johnson, without a published primary source.

    Referenced against:
    - pandas-ta-classic, pwma: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/pwma.py#L17

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods to consider.

    Returns:
        pd.Series: PWMA values.
    """
    # The binomial coefficients exceed the float range from a window of about 1,030, so they are built as
    # logarithms (ln C(n, i) = ln n! - ln i! - ln (n - i)!) and scaled by the largest before leaving log space;
    # the average divides by the sum, so the scale drops out.  # noqa: E501
    positions = np.arange(window)
    log_weights = gammaln(window) - gammaln(positions + 1) - gammaln(window - positions)
    weights = np.exp(log_weights - log_weights.max())

    return apply_to_rolling_windows(
        prices, window, lambda windows: windows @ weights / weights.sum()
    )


def get_price_transform(
    prices_open: pd.Series,
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    method: str = "typical",
) -> pd.Series:
    """
    Calculate a Price Transform of a given price series: the average, median, typical or weighted close price of
    each period.

    A price transform condenses the open, high, low and close of a period into one price.
    Indicators that should reflect the whole bar rather than only the close are built on
    them: the Commodity Channel Index, Money Flow Index and VWAP on the typical price, the
    Awesome Oscillator and FRAMA on the median price.

    The formula is as follows, per method:

    - Average price: Average Price = (Open + High + Low + Close) / 4
    - Median price: Median Price = (High + Low) / 2
    - Typical price: Typical Price = (High + Low + Close) / 3
    - Weighted close price: Weighted Close Price = (High + Low + 2 * Close) / 4

    Also known as: AVGPRICE, OHLC4, MEDPRICE, HL2, TYPPRICE, HLC3, WCLPRICE, HLCC4.

    Reference: Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic
    Trends." Commodities, October 1980, the typical price of the CCI.

    Referenced against:
    - TA-Lib, AVGPRICE: https://ta-lib.org/functions/avgprice.html
    - TA-Lib, MEDPRICE: https://ta-lib.org/functions/medprice.html
    - TA-Lib, TYPPRICE: https://ta-lib.org/functions/typprice.html
    - TA-Lib, WCLPRICE: https://ta-lib.org/functions/wclprice.html

    Shares similar logic:
    - pandas-ta-classic, ohlc4: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/ohlc4.py#L12
    - pandas-ta-classic, hl2: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/hl2.py#L12
    - pandas-ta-classic, hlc3: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/hlc3.py#L12
    - pandas-ta-classic, wcp: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/wcp.py#L12
    - finta, TP: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1405
    - Tulip Indicators, avgprice: https://tulipindicators.org/avgprice
    - Tulip Indicators, medprice: https://tulipindicators.org/medprice
    - Tulip Indicators, typprice: https://tulipindicators.org/typprice
    - Tulip Indicators, wcprice: https://tulipindicators.org/wcprice

    Args:
        prices_open (pd.Series): Series of open prices.
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        method (str): The price. Options are: "average" (the average of the open, high, low and close),
            "median" (the average of the high and low), "typical" (the average of the high, low and close),
            "weighted_close" (the weighted close, (high + low + 2 × close) / 4). Defaults to "typical".

    Returns:
        pd.Series: The price of each period.

    Raises:
        ValueError: If `method` is not one of the four transforms.
    """
    if method == "average":
        return (prices_open + prices_high + prices_low + prices_close) / 4
    if method == "median":
        return (prices_high + prices_low) / 2
    if method == "typical":
        return (prices_high + prices_low + prices_close) / 3
    if method == "weighted_close":
        return (prices_high + prices_low + 2 * prices_close) / 4

    raise ValueError(
        f"method must be average, median, typical or weighted_close, not {method}."
    )


def get_midpoint(
    prices_high: pd.Series, prices_low: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculate the Midpoint of a given price series: the middle of the highest high and the lowest low over a window.

    The midpoint is the centre of the range the price covered over the window, the middle
    line of a Donchian Channel that includes the current period. Passing the closing prices
    as both the highs and the lows gives the midpoint of the closes (TA-Lib's MIDPOINT),
    the highs and lows the midpoint price (TA-Lib's MIDPRICE).

    The formula is as follows:

    - Midpoint = (Highest High(window) + Lowest Low(window)) / 2

    Also known as: MIDPOINT, MIDPRICE, midpoint price, Donchian middle line.

    Reference: Donchian, R.D. (1960). "Commodities: High Finance in Copper." Financial
    Analysts Journal, 16(6), 133-142, the channel of the highest high and lowest low.

    Referenced against:
    - TA-Lib, MIDPOINT: https://ta-lib.org/functions/midpoint.html
    - TA-Lib, MIDPRICE: https://ta-lib.org/functions/midprice.html

    Shares similar logic:
    - pandas-ta-classic, midpoint: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/midpoint.py#L12
    - pandas-ta-classic, midprice: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/midprice.py#L12

    Args:
        prices_high (pd.Series): Series of high prices (or closing prices).
        prices_low (pd.Series): Series of low prices (or closing prices).
        window (int): Number of periods, including the current one. Defaults to 14.

    Returns:
        pd.Series: Midpoint values.
    """
    return (
        prices_high.rolling(window=window).max()
        + prices_low.rolling(window=window).min()
    ) / 2


def get_holt_winters_moving_average(
    prices: pd.Series,
    level_smoothing: float = 0.2,
    velocity_smoothing: float = 0.1,
    acceleration_smoothing: float = 0.1,
) -> pd.Series:
    """
    Calculate the Holt-Winters Moving Average (HWMA): a smoothing of the level, the trend and the acceleration of
    the prices, which follows a trend with little lag.

    The HWMA applies Holt's exponential smoothing with a trend, extended with an
    acceleration term, to the price: it smooths the level of the price, its velocity (the
    change of the level) and its acceleration (the change of the velocity), and adds the
    expected velocity and acceleration of one period to the level. The average therefore
    follows a trend with little lag, at the cost of overshooting when the trend turns.

    The formula is as follows:

    - F(t) = (1 - level_smoothing) * (F(t-1) + V(t-1) + 0.5 * A(t-1)) + level_smoothing * Price(t)
    - V(t) = (1 - velocity_smoothing) * (V(t-1) + A(t-1)) + velocity_smoothing * (F(t) - F(t-1))
    - A(t) = (1 - acceleration_smoothing) * A(t-1) + acceleration_smoothing * (V(t) - V(t-1))
    - HWMA(t) = F(t) + V(t) + 0.5 * A(t)

    with F starting at the first price and V and A at 0.

    Also known as: HWMA, Holt-Winters moving average.

    For more information, see the following sources:

    - Holt, C.C. (1957). "Forecasting Seasonals and Trends by Exponentially Weighted Moving
      Averages." ONR Memorandum 52, Carnegie Institute of Technology (reprinted in the
      International Journal of Forecasting, 20(1), 2004).
    - Winters, P.R. (1960). "Forecasting Sales by Exponentially Weighted Moving Averages."
      Management Science, 6(3), 324-342.
    - MQL5 Code Base: HWMA (https://www.mql5.com/en/code/20856).

    Referenced against:
    - pandas-ta-classic, hwma: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/hwma.py#L31

    Notes:
        - The recursion starts on the first price, as in the MQL5 indicator and
          pandas-ta-classic, so there is no warm-up: the first values lean on that price
          and the weight of the start fades by a factor 1 - level_smoothing per period.
        - A period without a price is missing, and the recursion continues from the last
          values on the next price.

    Args:
        prices (pd.Series): Series of prices.
        level_smoothing (float): The smoothing factor of the level, between 0 and 1.
            Defaults to 0.2.
        velocity_smoothing (float): The smoothing factor of the velocity, between 0 and 1.
            Defaults to 0.1.
        acceleration_smoothing (float): The smoothing factor of the acceleration, between 0
            and 1. Defaults to 0.1.

    Returns:
        pd.Series: HWMA values.
    """
    valid_prices = prices.dropna()
    averages = []

    # The recursion runs on plain Python floats, since writing a pandas Series one element
    # at a time costs tens of microseconds per element.
    price_values = valid_prices.to_numpy(dtype=float).tolist()
    level = price_values[0] if price_values else math.nan
    velocity = acceleration = 0.0
    for price in price_values:
        new_level = (1 - level_smoothing) * (
            level + velocity + 0.5 * acceleration
        ) + level_smoothing * price
        new_velocity = (1 - velocity_smoothing) * (
            velocity + acceleration
        ) + velocity_smoothing * (new_level - level)
        acceleration = (
            1 - acceleration_smoothing
        ) * acceleration + acceleration_smoothing * (new_velocity - velocity)
        level, velocity = new_level, new_velocity
        averages.append(level + velocity + 0.5 * acceleration)

    return pd.Series(averages, index=valid_prices.index, dtype="float64").reindex(
        prices.index
    )


def get_holt_winters_channel(
    prices: pd.Series,
    level_smoothing: float = 0.2,
    velocity_smoothing: float = 0.1,
    acceleration_smoothing: float = 0.1,
    deviation_smoothing: float = 0.1,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate the Holt-Winters Channel: bands a multiple of the smoothed standard deviation around the Holt-Winters
    Moving Average, which widen when the price strays from its trend.

    The channel draws bands around the Holt-Winters Moving Average at a multiple of an
    exponentially smoothed standard deviation of the price from the average, so it widens
    when the price strays from its trend and narrows when it follows it.

    The formula is as follows:

    - HWMA = Holt-Winters Moving Average (see `get_holt_winters_moving_average`)
    - Var(t) = (1 - deviation_smoothing) * Var(t-1) + deviation_smoothing *
      (Price(t-1) - HWMA(t-1))^2, starting at 0
    - Upper Band = HWMA(t) + multiplier * sqrt(Var(t-1))
    - Lower Band = HWMA(t) - multiplier * sqrt(Var(t-1))

    Also known as: HWC, Holt-Winters channel.

    For more information, see the following sources:

    - Holt, C.C. (1957). "Forecasting Seasonals and Trends by Exponentially Weighted Moving
      Averages." ONR Memorandum 52, Carnegie Institute of Technology.
    - Winters, P.R. (1960). "Forecasting Sales by Exponentially Weighted Moving Averages."
      Management Science, 6(3), 324-342.
    - MQL5 Code Base: HWC (https://www.mql5.com/en/code/20857).

    Referenced against:
    - pandas-ta-classic, hwc: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/volatility/hwc.py#L75

    Notes:
        - The band of period t uses the variance up to period t-1, which holds the
          deviations up to period t-2, as in the MQL5 indicator and pandas-ta-classic. Since
          the average starts on the first price, the bands open from the fourth price on.

    Args:
        prices (pd.Series): Series of prices.
        level_smoothing (float): The smoothing factor of the level. Defaults to 0.2.
        velocity_smoothing (float): The smoothing factor of the velocity. Defaults to 0.1.
        acceleration_smoothing (float): The smoothing factor of the acceleration.
            Defaults to 0.1.
        deviation_smoothing (float): The smoothing factor of the variance. Defaults to 0.1.
        multiplier (float): The number of standard deviations of the bands. Defaults to 1.

    Returns:
        pd.DataFrame: The Upper Band, Middle Band (the HWMA) and Lower Band.
    """
    valid_prices = prices.dropna()
    average = get_holt_winters_moving_average(
        valid_prices, level_smoothing, velocity_smoothing, acceleration_smoothing
    )

    # The squared deviation of the period before, 0 on the first period, smoothed into
    # the variance; the band of a period takes the variance of the period before.
    squared_deviation = ((valid_prices - average) ** 2).shift(1, fill_value=0.0)
    variance = squared_deviation.ewm(alpha=deviation_smoothing, adjust=False).mean()
    deviation = np.sqrt(variance.shift(1, fill_value=0.0))

    return pd.concat(
        [
            average + multiplier * deviation,
            average,
            average - multiplier * deviation,
        ],
        keys=["Upper Band", "Middle Band", "Lower Band"],
        axis=1,
    ).reindex(prices.index)


def get_elastic_volume_weighted_moving_average(
    prices: pd.Series, volumes: pd.Series, window: int = 20
) -> pd.Series:
    """
    Calculate the Elastic Volume Weighted Moving Average (eVWMA): an average that moves towards each price by the
    share of the volume traded, the average price of the shares held.

    The eVWMA approximates the average price paid per share over the latest `window`
    periods. Each period the shares traded replace the same number of shares at the
    previous average, out of the shares traded over the window, so the average moves
    towards the price by the share of the window's volume traded in that period: fast on
    heavy volume, hardly at all on light volume.

    The formula is as follows:

    - Volume Sum = Sum(Volume, window)
    - eVWMA(t) = ((Volume Sum - Volume(t)) * eVWMA(t-1) + Volume(t) * Close(t)) / Volume Sum
    - which is eVWMA(t) = eVWMA(t-1) + Volume(t) / Volume Sum * (Close(t) - eVWMA(t-1))

    Also known as: eVWMA, EVWMA, elastic volume weighted moving average.

    Reference: Fries, C.P. (2001). "Elastic Moving Averages." Technical Analysis of Stocks &
    Commodities, 19(6), 52-56 (https://www.christian-fries.de/evwma/), which takes the number of shares outstanding
    (the float) as the volume sum; the sum of the volume over a window stands in for it.

    Referenced against:
    - finta, EVWMA: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L381

    Notes:
        - The recursion is seeded with the simple average of the first `window` prices, as
          the Toolkit seeds its exponential moving averages. finta starts the recursion
          from 0 on the first full window, so its eVWMA climbs from 0 to the price and
          converges by a factor 1 - Volume / Volume Sum per period.
        - A window without any volume keeps the last average, and a period without a price
          is missing while the recursion continues on the next price.

    Args:
        prices (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods of the volume sum. Defaults to 20.

    Returns:
        pd.Series: eVWMA values.
    """
    volume_sum = volumes.rolling(window=window).sum()

    return apply_variable_smoothing(
        prices, volumes / volume_sum.where(volume_sum != 0), window
    )


def get_volume_adjusted_moving_average(
    prices: pd.Series, volumes: pd.Series, window: int = 8
) -> pd.Series:
    """
    Calculate the Volume Adjusted Moving Average (VAMA): an average that weights each price by its volume relative
    to the average volume.

    The VAMA weights every price in the window by its traded value relative to the average
    volume of the window ending on that period, so a period that trades more than usual
    counts more than a quiet one.

    The formula is as follows:

    - Weight(t) = Close(t) * Volume(t) / SMA(Volume, window)(t)
    - VAMA = Sum(Weight * Close, window) / Sum(Weight, window)

    Also known as: VAMA, volume adjusted moving average.

    Reference: Arms, R.W. Jr. (1983). "Volume Cycles in the Stock Market: Market Timing
    Through Equivolume Charting." Dow Jones-Irwin, on adjusting prices for the volume
    traded; the VAMA itself is defined in finta (Peerchemist), without a published source.

    Referenced against:
    - finta, VAMA: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L226

    Notes:
        - The weights are finta's, which multiplies the volume ratio by the price as well,
          so the weight is the traded value (in currency) relative to the average volume.

    Args:
        prices (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods. Defaults to 8.

    Returns:
        pd.Series: VAMA values, missing for a window without volume.
    """
    average_volume = volumes.rolling(window=window).mean()
    weights = prices * volumes / average_volume.where(average_volume != 0)
    total_weight = weights.rolling(window=window).sum()

    return (weights * prices).rolling(window=window).sum() / total_weight.where(
        total_weight != 0
    )


def get_gann_hilo_activator(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    high_window: int = 13,
    low_window: int = 21,
) -> pd.DataFrame:
    """
    Calculate the Gann HiLo Activator: a trailing stop drawn from a moving average of the lows in an uptrend and of
    the highs in a downtrend, which flips when the close crosses the other average.

    The HiLo Activator is a trailing stop drawn from a moving average of the highs and one
    of the lows. A close above the previous average of the highs turns the trend up, after
    which the line follows the average of the lows beneath the price; a close below the
    previous average of the lows turns it down, after which the line follows the average of
    the highs above the price. A close between the two averages keeps the trend.

    The formula is as follows:

    - High Average = SMA(High, high_window), Low Average = SMA(Low, low_window)
    - Trend(t) = 1 if Close(t) > High Average(t-1), -1 if Close(t) < Low Average(t-1),
      otherwise Trend(t-1)
    - HiLo Activator(t) = Low Average(t) when the trend is 1, High Average(t) when it is -1

    Also known as: HiLo, Gann HiLo, HiLo Activator, Gann swing.

    Reference: Krausz, R. (1998). "The New Gann Swing Chartist." Technical Analysis of
    Stocks & Commodities, 16(2), with the HiLo Activator on three-period averages of the
    highs and lows.

    Shares similar logic:
    - pandas-ta-classic, hilo: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/hilo.py#L40

    Notes:
        - The line follows the average of the side of the trend every period, as in
          Krausz's formula. pandas-ta-classic's hilo keeps the previous value of the line on
          a close between the two averages instead, so the two are equal on every period
          with a close beyond an average and differ on the others.
        - The defaults are those of pandas-ta-classic and TradingView (13 and 21 periods);
          Krausz used 3 periods for both.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        high_window (int): Number of periods of the average of the highs. Defaults to 13.
        low_window (int): Number of periods of the average of the lows. Defaults to 21.

    Returns:
        pd.DataFrame: The Gann HiLo Activator and the Trend Direction (1 up, -1 down).
    """
    high_average = prices_high.rolling(window=high_window).mean()
    low_average = prices_low.rolling(window=low_window).mean()

    # A close between the averages (or before the averages exist) keeps the trend.
    trend = pd.Series(
        np.select(
            [
                prices_close > high_average.shift(1),
                prices_close < low_average.shift(1),
            ],
            [1.0, -1.0],
            np.nan,
        ),
        index=prices_close.index,
    ).ffill()
    trend = trend.where(prices_close.notna())

    activator = low_average.where(trend == 1, high_average.where(trend == -1))

    return pd.concat(
        [activator, trend],
        keys=["Gann HiLo Activator", "Trend Direction"],
        axis=1,
    )


def get_zigzag(
    prices_high: pd.Series,
    prices_low: pd.Series,
    sensitivity: float = 0.05,
    minimum_trend_length: int = 1,
) -> pd.DataFrame:
    """
    Calculate the Zig Zag: the swing highs and lows of the prices, ignoring moves smaller than a threshold, which
    shows the trend's turning points.

    The Zig Zag filters out moves smaller than `sensitivity`: it follows the highest high of
    a rising leg until the low falls `sensitivity` below it, which confirms that high as a
    swing high and starts a falling leg from that low, and the lowest low of a falling leg
    until the high rises `sensitivity` above it. Drawn on a chart, the lines between the
    confirmed swings are the zig zag.

    The formula is as follows:

    - In a rising leg: a new trend down when Low(t) <= Extreme * (1 - sensitivity), else
      Extreme = High(t) when High(t) >= Extreme
    - In a falling leg: a new trend up when High(t) >= Extreme * (1 + sensitivity), else
      Extreme = Low(t) when Low(t) <= Extreme
    - A leg only turns once at least `minimum_trend_length` periods have passed since its
      extreme

    Also known as: ZIGZAG, Zig Zag, swing filter, filtered waves.

    Reference: Merrill, A.A. (1977). "Filtered Waves: Basic Theory." Analysis Press.

    Referenced against:
    - TA-Lib, ZIGZAG: https://ta-lib.org/functions/zigzag.html

    Notes:
        - Every period shows what is known at its close: the extreme of the current leg,
          its direction and how many periods ago it was set. This is the series TA-Lib
          returns and it never changes once a period has passed. The zig zag line drawn on
          a chart instead connects the swings, and its last leg is redrawn as long as the
          leg runs: a swing is only confirmed by the reversal that follows it, so a
          backtest must read a swing from the period of its reversal, as here.
        - The first leg starts falling from the first low, as in TA-Lib, and the first
          `minimum_trend_length` periods are missing. A price equal to the extreme moves the
          pivot to the later period, as in TA-Lib.
        - `sensitivity` is a decimal (0.05 is 5%); TA-Lib takes it in percent.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        sensitivity (float): The reversal from the extreme that turns a leg, as a decimal.
            Defaults to 0.05.
        minimum_trend_length (int): The number of periods since the extreme of a leg
            before it can turn. Defaults to 1.

    Returns:
        pd.DataFrame: The Zig Zag (the extreme of the current leg), the Trend Direction (1
        rising, -1 falling) and the Periods Since Pivot (since that extreme).
    """
    valid = prices_high.notna() & prices_low.notna()
    high = prices_high[valid].to_numpy(dtype=float).tolist()
    low = prices_low[valid].to_numpy(dtype=float).tolist()
    length = len(high)

    extremes = [math.nan] * length
    trends = [math.nan] * length
    periods_since = [math.nan] * length

    if length:
        trend, extreme, extreme_position = -1, low[0], 0

        # The legs run on plain Python lists, since reading and writing pandas Series one
        # element at a time costs tens of microseconds per element. A reversal is checked
        # against the extreme before a new extreme, as in TA-Lib.
        for i in range(1, length):
            turns = i - extreme_position >= minimum_trend_length
            if trend == 1:
                if turns and low[i] <= extreme * (1 - sensitivity):
                    trend, extreme, extreme_position = -1, low[i], i
                elif high[i] >= extreme:
                    extreme, extreme_position = high[i], i
            elif turns and high[i] >= extreme * (1 + sensitivity):
                trend, extreme, extreme_position = 1, high[i], i
            elif low[i] <= extreme:
                extreme, extreme_position = low[i], i

            if i >= minimum_trend_length:
                extremes[i] = extreme
                trends[i] = trend
                periods_since[i] = i - extreme_position

    return pd.DataFrame(
        {
            "Zig Zag": extremes,
            "Trend Direction": trends,
            "Periods Since Pivot": periods_since,
        },
        index=prices_high.index[valid.to_numpy()],
        dtype="float64",
    ).reindex(prices_high.index)


def get_volume_weighted_average_price(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    volumes: pd.Series,
    window: int,
    anchor: str | None = None,
) -> pd.Series:
    """
    Calculate the Volume Weighted Average Price (VWAP): the average price weighted by volume since an anchor date,
    the price at which most volume traded.

    VWAP weighs the typical price of each period by its traded volume over a rolling
    window, giving a more volume-informed view of the average price than a plain
    moving average. It is widely used as an institutional execution benchmark: a trade
    filled at or better than VWAP is generally considered to have avoided excess market
    impact.

    The formula is as follows:

    - Typical Price = (High + Low + Close) / 3
    - VWAP = Sum(Typical Price * Volume, window) / Sum(Volume, window)
    - Anchored VWAP = Cumulative Sum(Typical Price * Volume) / Cumulative Sum(Volume), both
      summed from the anchor on

    Also known as: VWAP, anchored VWAP, session VWAP.

    For more information, see the following sources:

    - Berkowitz, S.A., Logue, D.E., & Noser, E.A. Jr. (1988). "The Total Cost of
      Transactions on the NYSE." Journal of Finance, 43(1), 97-112 — the paper that
      introduced the volume-weighted average price as a transaction-cost benchmark.
    - Shannon, B. (2023). "Maximum Trading Gains With Anchored VWAP." Alphatrends
      Publishing.

    Referenced against:
    - TA-Lib, VWAP: https://ta-lib.org/functions/vwap.html

    Shares similar logic:
    - pandas-ta-classic, vwap: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/vwap.py#L18
    - finta, VWAP: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L411

    Notes:
        - TA-Lib's VWAP accumulates from the first period, the anchor of all periods; pandas-ta-classic's vwap
          starts again at every calendar period.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        volumes (pd.Series): Series of trading volumes.
        window (int): Number of periods to consider for the VWAP calculation, when there
            is no anchor.
        anchor (str | None): Where the sums start. Options are: "all" (the first period), "daily" (the start of
            every day, a session VWAP on intraday data), "weekly" (the start of every week), "monthly" (the
            start of every month), "quarterly" (the start of every quarter), "yearly" (the start of every year),
            or a date such as "2024-08-05" for the VWAP from that date on, missing before it. Defaults to None,
            which takes the rolling `window`.

    Returns:
        pd.Series: VWAP values.

    Raises:
        ValueError: If `anchor` is not one of the options or a date.
    """
    typical_price = (prices_high + prices_low + prices_close) / 3
    price_volume = typical_price * volumes

    if anchor is None:
        return (
            price_volume.rolling(window=window).sum()
            / volumes.rolling(window=window).sum()
        )

    frequencies = {
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }
    dates = (
        prices_close.index.to_timestamp()
        if isinstance(prices_close.index, pd.PeriodIndex)
        else pd.DatetimeIndex(prices_close.index)
    )

    if anchor == "all":
        groups = np.zeros(len(dates))
    elif anchor in frequencies:
        groups = dates.to_period(frequencies[anchor]).asi8
    else:
        try:
            start = pd.Timestamp(anchor)
        except ValueError as error:
            raise ValueError(
                "anchor must be all, daily, weekly, monthly, quarterly, yearly or a "
                f"date, not {anchor}."
            ) from error
        groups = np.where(dates >= start, 0.0, np.nan)

    # A period without a price or volume adds nothing to the sums, and is itself missing.
    traded = price_volume.notna() & volumes.notna()
    cumulative_price_volume = price_volume.where(traded, 0).groupby(groups).cumsum()
    cumulative_volume = volumes.where(traded, 0).groupby(groups).cumsum()

    # The periods before a date anchor are in no group and come back as missing.
    vwap = cumulative_price_volume / cumulative_volume.where(cumulative_volume != 0)

    return vwap.reindex(prices_close.index).where(traded)


def get_parabolic_sar(
    prices_high: pd.Series,
    prices_low: pd.Series,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
    af_start_short: float | None = None,
    af_increment_short: float | None = None,
    af_max_short: float | None = None,
    start_value: float = 0.0,
    offset_on_reverse: float = 0.0,
) -> pd.Series:
    """
    Calculate the Parabolic Stop and Reverse (SAR): a trailing stop that speeds up as the trend extends and flips
    side when the price crosses it.

    The Parabolic SAR is a trend-following indicator that trails price action, flipping
    from below to above price (and vice versa) whenever the trend reverses. The
    acceleration factor increases as the trend extends, causing the SAR to converge
    towards price over time.

    The formula is as follows:

    - SAR(t + 1) = SAR(t) + AF * (EP - SAR(t)), where EP (the extreme point) is the highest
      high of a long trade or the lowest low of a short one
    - AF starts at `af_start` and rises by `af_increment` with every new extreme point, up
      to `af_max`
    - In a long trade the SAR is at most the lows of the current and prior bar, in a short
      one at least their highs
    - A long reverses to a short when the low reaches the SAR (Low <= SAR), a short to a
      long when the high reaches it (High >= SAR); the SAR of that bar is the extreme
      point of the trade that ends, moved away from the price by `offset_on_reverse`
      (SAR * (1 + offset) for a new short, SAR * (1 - offset) for a new long), and AF
      starts again

    Also known as: SAR, Stop and Reverse, SAREXT, extended parabolic SAR.

    Reference: Wilder, J.W. Jr. (1978). "New Concepts in Technical Trading Systems." Trend
    Research.

    Referenced against:
    - TA-Lib, SAR: https://ta-lib.org/functions/sar.html
    - TA-Lib, SAREXT: https://ta-lib.org/functions/sarext.html

    Shares similar logic:
    - pandas-ta-classic, psar: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/psar.py#L73
    - pandas-ta-classic, sarext: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/sarext.py#L156

    Notes:
        - The start is TA-Lib's convention: the first bar only sets the direction
          and the starting SAR, so the first value is on the second bar. The trade starts
          short when the second bar's low fell by more than its high rose (a positive -DM),
          otherwise long, with the SAR at the first bar's low (long) or high (short) and
          the extreme point at the second bar's high (long) or low (short).
        - A `start_value` sets the start instead: a positive value starts a long with that
          SAR, a negative one a short with its absolute value as the SAR, as in TA-Lib's
          SAREXT.
        - The SAR is positive in a short trade too; TA-Lib's SAREXT returns it negated
          there to mark the direction.
        - Bars without a high or low are left out, so a ticker that lists later starts on
          its own first bars and a missing day is NaN.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        af_start (float): Initial acceleration factor. Defaults to 0.02.
        af_increment (float): Amount by which the acceleration factor increases every
            time a new extreme point is reached. Defaults to 0.02.
        af_max (float): Maximum value the acceleration factor can reach. Defaults to 0.2.
        af_start_short (float | None): Initial acceleration factor of a short trade.
            Defaults to None, which takes `af_start`.
        af_increment_short (float | None): Increment of the acceleration factor of a short
            trade. Defaults to None, which takes `af_increment`.
        af_max_short (float | None): Maximum acceleration factor of a short trade. Defaults
            to None, which takes `af_max`.
        start_value (float): The SAR of the first trade, positive for a long and negative
            for a short. Defaults to 0, which derives the start from the first two bars.
        offset_on_reverse (float): The share by which the SAR moves away from the price on
            a reversal, as a decimal. Defaults to 0.

    Returns:
        pd.Series: Parabolic SAR values.
    """
    af_start_short = af_start if af_start_short is None else af_start_short
    af_increment_short = (
        af_increment if af_increment_short is None else af_increment_short
    )
    af_max_short = af_max if af_max_short is None else af_max_short

    sar = pd.Series(index=prices_high.index, dtype="float64")

    # The recursion runs over the bars with both a high and a low, so a ticker that lists
    # later starts from its own first bar and a missing day is NaN instead of a carried value.
    valid = prices_high.notna() & prices_low.notna()
    prices_high, prices_low = prices_high[valid], prices_low[valid]
    length = len(prices_high)

    # The first bar only sets the starting direction and SAR, so a value needs two bars.
    if length < 2:  # noqa: PLR2004
        return sar

    # The recursion runs on plain Python lists, since reading and writing pandas Series
    # one element at a time costs tens of microseconds per element.
    high = prices_high.to_numpy(dtype="float64").tolist()
    low = prices_low.to_numpy(dtype="float64").tolist()

    # The direction comes from the directional movement of the second bar, TA-Lib's SAR
    # convention: a short when the low fell by more than the high rose, otherwise a long. The
    # SAR starts at the first bar's extreme on the other side of the trade.
    up_move = high[1] - high[0]
    down_move = low[0] - low[1]
    uptrend = (
        start_value > 0
        if start_value != 0
        else not (down_move > 0 and down_move > up_move)
    )
    af = af_start if uptrend else af_start_short
    extreme_point = high[1] if uptrend else low[1]
    current_sar = (
        abs(start_value) if start_value != 0 else (low[0] if uptrend else high[0])
    )

    sar_values = [math.nan] * length
    # On the second bar the previous bar's range is taken as its own, as in TA-Lib.
    new_high, new_low = high[1], low[1]

    for i in range(1, length):
        previous_high, previous_low = new_high, new_low
        new_high, new_low = high[i], low[i]

        if uptrend:
            if new_low <= current_sar:
                # The low reaches the SAR: the trade reverses to a short with the SAR at the
                # extreme point of the long, at least the highs of this and the prior bar.
                uptrend = False
                current_sar = max(extreme_point, previous_high, new_high)
                current_sar += current_sar * offset_on_reverse
                sar_values[i] = current_sar
                af = af_start_short
                extreme_point = new_low
                current_sar = current_sar + af * (extreme_point - current_sar)
                current_sar = max(current_sar, previous_high, new_high)
            else:
                sar_values[i] = current_sar
                if new_high > extreme_point:
                    extreme_point = new_high
                    af = min(af + af_increment, af_max)
                current_sar = current_sar + af * (extreme_point - current_sar)
                current_sar = min(current_sar, previous_low, new_low)
        elif new_high >= current_sar:
            # The high reaches the SAR: the trade reverses to a long with the SAR at the
            # extreme point of the short, at most the lows of this and the prior bar.
            uptrend = True
            current_sar = min(extreme_point, previous_low, new_low)
            current_sar -= current_sar * offset_on_reverse
            sar_values[i] = current_sar
            af = af_start
            extreme_point = new_high
            current_sar = current_sar + af * (extreme_point - current_sar)
            current_sar = min(current_sar, previous_low, new_low)
        else:
            sar_values[i] = current_sar
            if new_low < extreme_point:
                extreme_point = new_low
                af = min(af + af_increment_short, af_max_short)
            current_sar = current_sar + af * (extreme_point - current_sar)
            current_sar = max(current_sar, previous_high, new_high)

    return pd.Series(sar_values, index=prices_high.index, dtype="float64").reindex(
        sar.index
    )


def get_pivot_points(
    prices_high: pd.Series,
    prices_low: pd.Series,
    prices_close: pd.Series,
    prices_open: pd.Series | None = None,
    method: str = "classic",
    central_pivot_range: bool = False,
) -> pd.DataFrame:
    """
    Calculate the Pivot Points: the levels of support and resistance derived from the previous period's high, low
    and close.

    Pivot Points are calculated from the previous period's high, low and close prices
    and are used to identify potential support and resistance levels for the current
    period. The "classic" (floor trader) variant is the default; the Fibonacci, Camarilla,
    Woodie and DeMark variants place the levels differently.

    The formula is as follows, with H, L, C and O the previous period's high, low, close
    and open:

    - Classic: P = (H + L + C) / 3, Resistance 1 = 2 * P - L, Support 1 = 2 * P - H,
      Resistance 2 = P + (H - L), Support 2 = P - (H - L), Resistance 3 = H + 2 * (P - L),
      Support 3 = L - 2 * (H - P)
    - Fibonacci: P = (H + L + C) / 3, Resistance n = P + Ratio * (H - L) and Support n =
      P - Ratio * (H - L), with the ratios 0.382, 0.618 and 1.0
    - Camarilla: P = (H + L + C) / 3, Resistance n = C + (H - L) * 1.1 / Divisor and
      Support n = C - (H - L) * 1.1 / Divisor, with the divisors 12, 6, 4 and 2
    - Woodie: P = (H + L + 2 * C) / 4, Resistance 1 = 2 * P - L, Support 1 = 2 * P - H,
      Resistance 2 = P + (H - L), Support 2 = P - (H - L)
    - DeMark: X = H + 2 * L + C when C < O, 2 * H + L + C when C > O and H + L + 2 * C
      otherwise; P = X / 4, Resistance 1 = X / 2 - L, Support 1 = X / 2 - H
    - Central Pivot Range: Bottom Central Pivot = (H + L) / 2, Top Central Pivot =
      2 * P - Bottom Central Pivot

    Also known as: floor pivots, classic pivot points, standard pivot points, Fibonacci
    pivots, Camarilla pivots, Woodie pivots, DeMark pivots, central pivot range, CPR.

    For more information, see the following sources:

    - Person, J.L. (2004). "A Complete Guide to Technical Trading Tactics." Wiley, the
      standard treatment of the floor trader pivots, which originate in floor-trading
      practice rather than a published paper.
    - Scott, N. (1989). The Camarilla equation, as published by Camarilla Levels.
    - DeMark, T.R. (1994). "The New Science of Technical Analysis." Wiley.
    - Ochoa, F. (2010). "Secrets of a Pivot Boss: Revealing Proven Methods for Profiting in
      the Market." 311 Publishing, the central pivot range.

    Referenced against:
    - pandas-ta-classic, cpr: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/trend/cpr.py#L76

    Shares similar logic:
    - finta, PIVOT: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1123
    - finta, PIVOT_FIB: https://github.com/peerchemist/finta/blob/1.3/finta/finta.py#L1162

    Notes:
        - Every level is derived exclusively from the *previous* period's prices, so the
          whole set is known at the open of the current period and carries no look-ahead.
        - The Woodie pivot weights the previous close twice, as in pandas-ta-classic; some
          charting packages take the current open instead, which is only known at the
          open of the period.
        - finta's PIVOT and PIVOT_FIB add a fourth level (classic, and 1.382 times the
          range for Fibonacci); the levels they share are equal.

    Args:
        prices_high (pd.Series): Series of high prices.
        prices_low (pd.Series): Series of low prices.
        prices_close (pd.Series): Series of closing prices.
        prices_open (pd.Series | None): Series of open prices, needed for "demark".
            Defaults to None.
        method (str): The pivots. Options are: "classic" (classic floor pivots), "fibonacci" (Fibonacci pivots),
            "camarilla" (Camarilla pivots), "woodie" (Woodie's pivots), "demark" (DeMark's pivots). Defaults to
            "classic".
        central_pivot_range (bool): Whether to add the Top and Bottom Central Pivot.
            Defaults to False.

    Returns:
        pd.DataFrame: The Pivot Point with the resistance and support levels of the method.

    Raises:
        ValueError: If `method` is not one of the variants, or "demark" has no open prices.
    """
    previous_high = prices_high.shift(1)
    previous_low = prices_low.shift(1)
    previous_close = prices_close.shift(1)
    price_range = previous_high - previous_low

    levels: dict[str, pd.Series] = {}

    if method in ("classic", "fibonacci", "camarilla"):
        pivot = (previous_high + previous_low + previous_close) / 3
    elif method == "woodie":
        pivot = (previous_high + previous_low + 2 * previous_close) / 4
    elif method == "demark":
        if prices_open is None:
            raise ValueError("The DeMark pivot points need the open prices.")
        previous_open = prices_open.shift(1)
        extended_close = (
            previous_close.where(previous_close == previous_open)
            .add(previous_high.where(previous_close > previous_open), fill_value=0)
            .add(previous_low.where(previous_close < previous_open), fill_value=0)
        )
        demark_sum = previous_high + previous_low + previous_close + extended_close
        pivot = demark_sum / 4
        levels["Resistance 1"] = demark_sum / 2 - previous_low
        levels["Support 1"] = demark_sum / 2 - previous_high
    else:
        raise InvalidArgumentError(
            "method must be classic, fibonacci, camarilla, woodie or demark, "
            f"not {method}."
        )

    if method in ("classic", "woodie"):
        levels["Resistance 1"] = (2 * pivot) - previous_low
        levels["Support 1"] = (2 * pivot) - previous_high
        levels["Resistance 2"] = pivot + price_range
        levels["Support 2"] = pivot - price_range

    if method == "classic":
        levels["Resistance 3"] = previous_high + 2 * (pivot - previous_low)
        levels["Support 3"] = previous_low - 2 * (previous_high - pivot)
    elif method == "fibonacci":
        for number, ratio in enumerate((0.382, 0.618, 1.0), start=1):
            levels[f"Resistance {number}"] = pivot + ratio * price_range
            levels[f"Support {number}"] = pivot - ratio * price_range
    elif method == "camarilla":
        for number, divisor in enumerate((12, 6, 4, 2), start=1):
            levels[f"Resistance {number}"] = (
                previous_close + price_range * 1.1 / divisor
            )
            levels[f"Support {number}"] = previous_close - price_range * 1.1 / divisor

    if central_pivot_range:
        bottom_central_pivot = (previous_high + previous_low) / 2
        levels["Bottom Central Pivot"] = bottom_central_pivot
        levels["Top Central Pivot"] = 2 * pivot - bottom_central_pivot

    levels = {"Pivot Point": pivot, **dict(sorted(levels.items()))}

    return pd.concat(levels.values(), keys=levels.keys(), axis=1)


def _get_confirmed_levels(
    prices: pd.Series, extreme_indices: np.ndarray, window: int, sensitivity: float
) -> pd.Series:
    """
    Turn a set of centred local extremes into a strictly causal series of price levels.

    A local extreme found by a centred pivot window at position `i` is not knowable until
    `window` further periods have printed and confirmed that no higher high (or lower low)
    followed. Each extreme is therefore published at position `i + window` — the first
    period on which an observer could actually have identified it — and extremes whose
    confirmation period falls beyond the end of the sample are dropped entirely rather
    than published early.

    Extremes within `sensitivity` (a fractional distance) of an already-established level
    are treated as a retest of that level and blend into it as a running average. The
    blended value is published at the *new* extreme's confirmation date; the value
    already published at earlier dates is never rewritten, so the series is append-only.

    Args:
        prices (pd.Series): Series of prices the extremes were detected on.
        extreme_indices (np.ndarray): Positional indices of the centred local extremes.
        window (int): The half-width of the centred pivot window, and therefore the
            number of periods by which confirmation of each extreme lags its occurrence.
        sensitivity (float): Fractional distance below which a new extreme is treated as
            a retest of an existing level rather than a new level.

    Returns:
        pd.Series: Levels indexed by the date each level was confirmed on.
    """
    length = len(prices)
    levels: list[float] = []
    published: dict = {}

    for position in extreme_indices:
        confirmation_position = int(position) + window

        # An extreme is only knowable once the centred window that identified it has fully printed; anything not yet confirmed within the sample is discarded.  # noqa: E501
        if confirmation_position >= length:
            break

        price = float(prices.iloc[int(position)])
        confirmation_date = prices.index[confirmation_position]

        for level_position, level in enumerate(levels):
            if level and abs(price - level) / abs(level) < sensitivity:
                levels[level_position] = (level + price) / 2
                published[confirmation_date] = levels[level_position]
                break
        else:
            levels.append(price)
            published[confirmation_date] = price

    return pd.Series(published, dtype="float64")


def get_support_resistance_levels(
    prices: pd.Series, window: int = 5, sensitivity: float = 0.05
) -> pd.DataFrame:
    """
    Calculate the support and resistance levels: the prices at which the prices repeatedly stopped falling (support)
    or rising (resistance).

    Support levels are the valleys where price has repeatedly stopped falling, and
    resistance levels are the peaks where it has repeatedly stopped rising. Both are
    identified here as centred local extremes — a price that is higher (or lower) than
    every price within `window` periods on either side of it — and extremes that sit
    within `sensitivity` of one another are blended into a single level, on the reasoning
    that price stalling repeatedly at nearly the same value describes one level rather
    than several.

    Also known as: support levels, resistance levels, swing highs and swing lows.

    Reference: The standard textbook treatment is Murphy, J.J. (1999). "Technical Analysis
    of the Financial Markets." New York Institute of Finance.

    Notes:
        - A centred pivot window cannot identify an extreme until `window` further periods
          have printed without exceeding it. Every level is therefore published with a
          confirmation lag of exactly `window` periods: the level formed at period `t` first
          appears at period `t + window`, which is the first period on which an observer
          could have known about it. Extremes that have not yet been confirmed by the end
          of the sample are not published at all.
        - The series is append-only. A retest that blends into an existing level publishes
          the blended value from its own confirmation date onward and never rewrites a value
          that was already published, so a value read at any date is exactly the value that
          was available at that date and the result is safe to use in a backtest.

    Args:
        prices (pd.Series): Series of prices to identify levels on (typically closing prices).
        window (int): Half-width of the centred window used to identify local maxima and
            minima, and therefore also the confirmation lag in periods. Defaults to 5.
        sensitivity (float): Fractional distance below which a new extreme is treated as a
            retest of an existing level rather than a new one. A higher value blends more
            extremes together and therefore yields fewer distinct levels. Defaults to 0.05.

    Returns:
        pd.DataFrame: Support and resistance levels with two columns, "Resistance" and
            "Support", reindexed to `prices` and forward-filled, so that every date carries
            the most recently confirmed level (NaN before the first level is confirmed).
    """
    # scipy.signal takes around a second to import, so only where it is needed.
    from scipy.signal import argrelextrema  # noqa: PLC0415

    # The extremes are found on the observed prices only, so a ticker that lists later or
    # misses a day is judged on its own closes, and only over full windows: an extreme in
    # the first `window` closes has no full window before it.
    observed = prices.dropna()
    values = observed.to_numpy()
    local_maxima_indices = argrelextrema(values, np.greater, order=window)[0]
    local_minima_indices = argrelextrema(values, np.less, order=window)[0]
    local_maxima_indices = local_maxima_indices[local_maxima_indices >= window]
    local_minima_indices = local_minima_indices[local_minima_indices >= window]

    resistance_levels = _get_confirmed_levels(
        observed, local_maxima_indices, window, sensitivity
    )
    support_levels = _get_confirmed_levels(
        observed, local_minima_indices, window, sensitivity
    )

    support_resistance_levels = pd.DataFrame(
        {"Resistance": resistance_levels, "Support": support_levels}
    ).sort_index()

    return support_resistance_levels.reindex(prices.index).ffill()


def get_fibonacci_retracement_levels(
    high_prices: pd.Series,
    low_prices: pd.Series,
    levels: list[float] | None = None,
    trend: str = "uptrend",
) -> pd.DataFrame:
    """
    Calculate the Fibonacci Retracement Levels: the prices at the Fibonacci ratios of the range between the high and
    the low, where a pullback may find support or resistance.

    Fibonacci Retracement Levels are horizontal price levels, derived from ratios found in the
    Fibonacci sequence, that traders watch as potential support (during a pullback within an
    uptrend) or resistance (during a bounce within a downtrend) zones. `high_prices` and
    `low_prices` are expected to already represent the swing high and swing low over the
    lookback window of interest — e.g. a rolling maximum and rolling minimum computed by the
    caller — so that a full set of retracement levels is produced for every date rather than
    for a single, hand-picked swing.

    The formula is as follows:

    - Uptrend (retracing down from the high): Level = High - Ratio * (High - Low)
    - Downtrend (retracing up from the low): Level = Low + Ratio * (High - Low)

    Also known as: Fibonacci retracement, Fib levels, retracement levels.

    Notes:
        - The 50% level is not actually a Fibonacci ratio. It is included purely by long-standing
          market convention, based on the Dow Theory observation that markets often retrace about
          half of a prior move.
        - The 78.6% level is the square root of 0.618, not a ratio drawn directly from the
          Fibonacci sequence itself (unlike 23.6%, 38.2% and 61.8%, which are).
        - There is no single canonical academic paper behind Fibonacci Retracement Levels — the
          indicator is a practitioner tool derived from the Fibonacci sequence's ratios rather
          than a published financial model. The standard textbook treatment is Murphy, J.J.
          (1999). "Technical Analysis of the Financial Markets." New York Institute of Finance.

    Args:
        high_prices (pd.Series): Series of high prices (e.g. a rolling maximum over the
            desired lookback window).
        low_prices (pd.Series): Series of low prices (e.g. a rolling minimum over the
            desired lookback window).
        levels (list[float] | None): The Fibonacci ratios to calculate levels for. Defaults to
            the standard [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0].
        trend (str): The direction of the prior move. Options are: "uptrend" (levels measured down from the high
            — the conventional direction, used when a prior move was up and price is now pulling back),
            "downtrend" (levels measured up from the low, used when a prior move was down and price is now
            bouncing). Defaults to "uptrend".

    Returns:
        pd.DataFrame: Fibonacci Retracement Levels, one column per ratio in `levels`, labelled
            by the ratio expressed as a percentage (e.g. "23.6%").

    Raises:
        ValueError: If `trend` is not "uptrend" or "downtrend".
    """
    if levels is None:
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    check_option("trend", trend, ("uptrend", "downtrend"))

    price_range = high_prices - low_prices

    retracement_levels = {}
    for level in levels:
        column_name = f"{level * 100:.1f}%"
        if trend == "uptrend":
            retracement_levels[column_name] = high_prices - level * price_range
        else:
            retracement_levels[column_name] = low_prices + level * price_range

    return pd.concat(
        retracement_levels.values(), keys=retracement_levels.keys(), axis=1
    )


# Guppy's windows of the short-term (traders) and long-term (investors) groups.
GUPPY_SHORT_WINDOWS = [3, 5, 8, 10, 12, 15]
GUPPY_LONG_WINDOWS = [30, 35, 40, 45, 50, 60]


def get_guppy_multiple_moving_averages(
    prices: pd.Series,
    short_windows: list[int] | None = None,
    long_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Calculate Daryl Guppy's Multiple Moving Averages (GMMA): a group of short and a group of long exponential moving
    averages, whose spread and order show the strength of the traders' and the investors' trend.

    The Guppy Multiple Moving Averages are two groups of Exponential Moving Averages: a short
    group of 3 to 15 periods that follows the traders, and a long group of 30 to 60 periods
    that follows the investors. A trend is up when the short group lies above the long group
    and down when it lies below, and a change of trend shows where the groups cross. How far
    the averages of a group spread apart says how strong the conviction of that group is: a
    long group that widens confirms the trend, one that compresses warns that the investors
    are changing their minds.

    The formula is as follows:

    - Short EMA(n) = EMA(Close, n), for n in 3, 5, 8, 10, 12 and 15
    - Long EMA(n) = EMA(Close, n), for n in 30, 35, 40, 45, 50 and 60
    - Short Group Spread = Max(Short EMAs) - Min(Short EMAs)
    - Long Group Spread = Max(Long EMAs) - Min(Long EMAs)
    - Group Spread = Mean(Short EMAs) - Mean(Long EMAs)

    Also known as: GMMA, Guppy multiple moving average, Guppy ribbons.

    Reference: Guppy, D. (2004). "Trend Trading: A Seven-Step Approach to Success." Wiley.

    Referenced against:
    - TTR (R), GMMA: https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/GMMA.R#L55

    Notes:
        - The averages are `get_exponential_moving_average`, seeded with the simple average
          of their first `n` prices and missing before it, as TTR's EMA is.
        - The spreads are in price units and start once every average of their groups has
          started. TTR's GMMA returns the averages only.

    Args:
        prices (pd.Series): Series of prices.
        short_windows (list[int] | None): The windows of the short group. Defaults to None,
            which uses Guppy's 3, 5, 8, 10, 12 and 15 (as does an empty list).
        long_windows (list[int] | None): The windows of the long group. Defaults to None,
            which uses Guppy's 30, 35, 40, 45, 50 and 60 (as does an empty list).

    Returns:
        pd.DataFrame: The short and long EMAs and the group spreads, in price units.
    """
    short_windows = short_windows or GUPPY_SHORT_WINDOWS
    long_windows = long_windows or GUPPY_LONG_WINDOWS

    short_group = pd.concat(
        [get_exponential_moving_average(prices, window) for window in short_windows],
        axis=1,
        keys=[f"Short EMA {window}" for window in short_windows],
    )
    long_group = pd.concat(
        [get_exponential_moving_average(prices, window) for window in long_windows],
        axis=1,
        keys=[f"Long EMA {window}" for window in long_windows],
    )

    spreads = pd.DataFrame(
        {
            "Short Group Spread": short_group.max(axis=1, skipna=False)
            - short_group.min(axis=1, skipna=False),
            "Long Group Spread": long_group.max(axis=1, skipna=False)
            - long_group.min(axis=1, skipna=False),
            "Group Spread": short_group.mean(axis=1, skipna=False)
            - long_group.mean(axis=1, skipna=False),
        }
    )

    return pd.concat([short_group, long_group, spreads], axis=1)
