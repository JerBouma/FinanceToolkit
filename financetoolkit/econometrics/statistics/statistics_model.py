"""Statistics Model: the rolling linear regression and the rolling statistics of a series"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from financetoolkit.utilities.error_model import check_option


def get_linear_regression(prices: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Calculate the rolling least-squares Linear Regression of a price series: the end point of
    the regression line, its slope, intercept and angle, and the Time Series Forecast.

    Over every window of `window` periods, a straight line is fitted through the prices by
    ordinary least squares, with the time x running from 0 for the oldest price of the window
    to window - 1 for the latest. The end point of that line is a moving average that lags
    less than a simple average, as it follows the trend of the window rather than its middle,
    and the Time Series Forecast extends the line by one period.

    The formula is as follows:

    - Slope (b1) = Sum((x - Mean(x)) * (Close - Mean(Close))) / Sum((x - Mean(x))^2), x = 0, ..., window - 1
    - Intercept (b0) = Mean(Close) - b1 * Mean(x), the line at the oldest period of the window
    - Linear Regression = b0 + b1 * (window - 1), the line at the latest period
    - Time Series Forecast = b0 + b1 * window, the line one period ahead
    - Angle = arctan(b1) in degrees

    Also known as: LINEARREG, linear regression indicator, end-point moving average, least
    squares moving average, LSMA, TSF, time series forecast, regression slope.

    Reference: Chande, T.S. & Kroll, S. (1994). "The New Technical Trader." Wiley.

    Referenced against TA-Lib's LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE
    and TSF (https://ta-lib.org/functions/linearreg.html).

    Notes:
        - Equals TA-Lib's LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE
          and TSF, Tulip Indicators' linreg, linregslope, linregintercept and tsf and
          pandas-ta-classic's linreg.
        - The slope is in price per period and the angle depends on the scale of the prices,
          so neither compares across securities with different share prices.
        - A window with a missing price has no line.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods the line is fitted over, at least 2. Defaults to 14.

    Returns:
        pd.DataFrame: Linear Regression, Slope, Intercept, Angle and Time Series Forecast.
    """
    if window < 2:  # noqa: PLR2004
        raise ValueError("The window of a linear regression is at least 2 periods.")

    values = prices.to_numpy(dtype=float)
    slope = np.full(len(values), np.nan)
    intercept = np.full(len(values), np.nan)

    if len(values) >= window:
        windows = sliding_window_view(values, window)
        time = np.arange(window) - (window - 1) / 2
        slope[window - 1 :] = windows @ time / (time @ time)
        intercept[window - 1 :] = windows.mean(axis=1) - slope[window - 1 :] * (
            (window - 1) / 2
        )

    return pd.DataFrame(
        {
            "Linear Regression": intercept + slope * (window - 1),
            "Slope": slope,
            "Intercept": intercept,
            "Angle": np.degrees(np.arctan(slope)),
            "Time Series Forecast": intercept + slope * window,
        },
        index=prices.index,
    )


def get_slope(prices: pd.Series, window: int = 1, as_angle: bool = False) -> pd.Series:
    """
    Calculate the Slope of a price series: the average change per period over a window,
    the rise over the run of the line between the price `window` periods ago and the latest.

    The formula is as follows:

    - Slope = (Close - Close(t - window)) / window
    - Angle = arctan(Slope) in degrees

    Also known as: slope, rise over run, average change per period.

    Reference: the slope of a straight line between two points (rise over run), a standard
    measure; pandas-ta-classic defines the same indicator.

    Referenced against pandas-ta-classic's slope
    (https://github.com/xgboosted/pandas-ta-classic).

    Notes:
        - Equals pandas-ta-classic's slope, and with `as_angle` its slope with
          `as_angle=True, to_degrees=True`; pandas-ta-classic gives the angle in radians
          unless asked for degrees.
        - The least-squares slope over a window, which uses every price of the window
          rather than its two ends, is the Slope of `get_linear_regression`.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods between the two prices. Defaults to 1.
        as_angle (bool): Whether to return the angle of the slope in degrees instead of the
            slope. Defaults to False.

    Returns:
        pd.Series: Slope values, in price per period or in degrees.
    """
    slope = prices.diff(window) / window

    return np.degrees(np.arctan(slope)) if as_angle else slope


def get_rolling_correlation(
    prices: pd.Series, other_prices: pd.Series, window: int = 30
) -> pd.Series:
    """
    Calculate the rolling Pearson Correlation between two price series over a window.

    The correlation is +1 when the two series move up and down together in a straight-line
    relation, -1 when they move in opposite directions and around 0 when they are unrelated.
    On price levels, two series that both trend have a high correlation even when their
    daily moves are unrelated, so it shows whether two prices follow the same course over
    the window rather than whether their returns are related.

    The formula is as follows:

    - Correlation = Covariance(X, Y, window) / (StdDev(X, window) * StdDev(Y, window))

    Also known as: CORREL, Pearson correlation coefficient, rolling correlation.

    Reference: Pearson, K. (1895). "Notes on Regression and Inheritance in the Case of Two
    Parents." Proceedings of the Royal Society of London, 58, 240-242.

    Referenced against TA-Lib's CORREL (https://ta-lib.org/functions/correl.html).

    Notes:
        - Equals TA-Lib's CORREL and pandas-ta-classic's correl.
        - A window in which either series is flat has no correlation and is missing, where
          TA-Lib and pandas-ta-classic give 0.

    Args:
        prices (pd.Series): Series of prices.
        other_prices (pd.Series): The series of prices to correlate with, on the same dates.
        window (int): Number of periods. Defaults to 30.

    Returns:
        pd.Series: Correlation values between -1 and 1.
    """
    correlation = prices.rolling(window=window).corr(other_prices)
    flat = (prices.rolling(window=window).std() == 0) | (
        other_prices.rolling(window=window).std() == 0
    )

    return correlation.mask(flat).where(np.isfinite(correlation))


def get_rolling_median(prices: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate the rolling Median of a price series: the middle price of the window.

    The median is a moving average that is not pulled by a single outlier, such as one day
    with a spike, as it only depends on the order of the prices.

    The formula is as follows:

    - Median = the middle value of the sorted prices of the window, or the mean of the two
      middle values for an even window

    Also known as: MEDIAN, moving median, simple moving median, SMM.

    Reference: Tukey, J.W. (1977). "Exploratory Data Analysis." Addison-Wesley, for running
    medians; TA-Lib's MEDIAN gives the same definition.

    Referenced against TA-Lib's MEDIAN (https://ta-lib.org/functions/median.html).

    Notes:
        - Equals TA-Lib's MEDIAN, pandas-ta-classic's median and finta's SMM (whose default
          window is 9).

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods. Defaults to 30.

    Returns:
        pd.Series: Median values.
    """
    return prices.rolling(window=window).median()


def get_rolling_percentile(
    prices: pd.Series,
    window: int = 100,
    percentile: float = 0.5,
    method: str = "nearest_rank",
) -> pd.Series:
    """
    Calculate the rolling Percentile of a price series: the price below which a given share
    of the prices of the window lies.

    The formula is as follows, per method:

    - "nearest_rank": the k-th smallest price of the window, k = ceil(percentile * window)
      (Hyndman and Fan's definition 1, the inverse of the empirical distribution function)
    - "linear": interpolated between the two prices around position percentile * (window - 1)
      of the sorted window, with positions from 0 (Hyndman and Fan's definition 7)

    Also known as: PERCENTILE, quantile, rolling quantile, moving percentile.

    Reference: Hyndman, R.J. & Fan, Y. (1996). "Sample Quantiles in Statistical Packages."
    The American Statistician, 50(4), 361-365.

    Referenced against TA-Lib's PERCENTILE
    (https://ta-lib.org/functions/percentile.html).

    Notes:
        - The nearest rank agrees with TA-Lib's PERCENTILE, and the linear interpolation equals
          pandas-ta-classic's quantile (window 30) and numpy's default quantile.
        - The percentile is a decimal (0.5 is the median) where TA-Lib takes a percentage.
        - A window with a missing price has no percentile.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods. Defaults to 100.
        percentile (float): The percentile as a decimal between 0 and 1. Defaults to 0.5.
        method (str): "nearest_rank" or "linear". Defaults to "nearest_rank".

    Returns:
        pd.Series: Percentile values.
    """
    methods = {"nearest_rank": "inverted_cdf", "linear": "linear"}
    check_option("method", method, methods)
    if not 0 <= percentile <= 1:
        raise ValueError("The percentile is a decimal between 0 and 1.")

    values = prices.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)

    if len(values) >= window:
        windows = sliding_window_view(values, window)
        result[window - 1 :] = np.where(
            np.isnan(windows).any(axis=1),
            np.nan,
            np.quantile(windows, percentile, axis=1, method=methods[method]),
        )

    return pd.Series(result, index=prices.index)


def get_percent_rank(prices: pd.Series, window: int = 100) -> pd.Series:
    """
    Calculate the Percent Rank of a price series: the share of the previous `window` prices
    that are lower than the latest price.

    A Percent Rank of 1 means the price is above every price of the window, a new high for
    the window, and 0 that it is at or below all of them. It ranks the price within its
    recent history the way a percentile does, and is a component of Connors' RSI.

    The formula is as follows:

    - Percent Rank = Count(Close(t - i) < Close(t), i = 1, ..., window) / window

    Also known as: PERCENTRANK, percentile rank, percent rank.

    Reference: Connors, L., Alvarez, C. & Radtke, M. (2012). "An Introduction to ConnorsRSI."
    Connors Research Trading Strategy Series.

    Referenced against TA-Lib's PERCENTRANK
    (https://ta-lib.org/functions/percentrank.html).

    Notes:
        - Equals TA-Lib's PERCENTRANK divided by 100: the Toolkit gives a decimal between
          0 and 1 where TA-Lib gives a percentage.
        - The latest price is compared with the `window` prices before it, so the first
          value needs `window` + 1 prices; a window with a missing price has no rank.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of previous periods. Defaults to 100.

    Returns:
        pd.Series: Percent Rank values as decimals.
    """
    values = prices.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)

    if len(values) > window:
        windows = sliding_window_view(values, window + 1)
        rank = (windows[:, :-1] < windows[:, -1:]).sum(axis=1) / window
        result[window:] = np.where(np.isnan(windows).any(axis=1), np.nan, rank)

    return pd.Series(result, index=prices.index)


def get_rolling_z_score(prices: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate the rolling Z-Score of a price series: the number of standard deviations the
    price lies above or below its moving average.

    The formula is as follows:

    - Z-Score = (Close - SMA(Close, window)) / StdDev(Close, window)

    with the sample standard deviation (ddof=1).

    Also known as: z-score, standard score, ZSCORE, normalized price.

    Reference: the standard score of the price within its window, a standard statistic;
    pandas-ta-classic's zscore gives the same rolling form.

    Referenced against pandas-ta-classic's zscore
    (https://github.com/xgboosted/pandas-ta-classic).

    Notes:
        - Equals pandas-ta-classic's zscore (window 30, ddof=1). Its cdl_z applies the
          same score to the open, high, low and close.
        - A window of flat prices has no standard deviation and is missing, where
          pandas-ta-classic gives 0.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods. Defaults to 30.

    Returns:
        pd.Series: Z-Score values.
    """
    standard_deviation = prices.rolling(window=window).std(ddof=1)

    return (prices - prices.rolling(window=window).mean()) / standard_deviation.where(
        standard_deviation > 0
    )


def get_rolling_mean_absolute_deviation(
    prices: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculate the rolling Mean Absolute Deviation of a price series: the average distance
    of the prices of the window from their mean.

    It measures the dispersion of the prices like the standard deviation, but without
    squaring the deviations, so a single large deviation weighs less. Lambert's Commodity
    Channel Index divides by it.

    The formula is as follows:

    - Mean Absolute Deviation = Mean(|Close(t - i) - SMA(Close, window)|, i = 0, ..., window - 1)

    with the average of the window itself for every price of the window.

    Also known as: AVGDEV, average deviation, mean deviation, MAD, MD.

    Reference: Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic
    Trends." Commodities, October 1980.

    Referenced against TA-Lib's AVGDEV (https://ta-lib.org/functions/avgdev.html).

    Notes:
        - Equals TA-Lib's AVGDEV, pandas-ta-classic's mad and md (window 30) and Tulip
          Indicators' md.
        - The Mean Absolute Deviation of `toolkit.risk` measures the daily returns within
          each period instead of the prices over a rolling window.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods. Defaults to 14.

    Returns:
        pd.Series: Mean Absolute Deviation values, in price units.
    """
    values = prices.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)

    if len(values) >= window:
        windows = sliding_window_view(values, window)
        result[window - 1 :] = np.abs(
            windows - windows.mean(axis=1, keepdims=True)
        ).mean(axis=1)

    return pd.Series(result, index=prices.index)


def get_rolling_standard_error(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the rolling Standard Error of the mean of a price series: the standard
    deviation of the window divided by the square root of the number of periods.

    It is the uncertainty of the moving average as an estimate of the mean price of the
    window: a moving average plus and minus two standard errors spans roughly where the
    mean lies.

    The formula is as follows:

    - Standard Error = StdDev(Close, window) / sqrt(window)

    with the sample standard deviation (ddof=1).

    Also known as: STDERR, standard error of the mean, SEM.

    Reference: the standard error of the sample mean, a standard statistic;
    pandas-ta-classic's stderr gives the same rolling form.

    Referenced against pandas-ta-classic's stderr
    (https://github.com/xgboosted/pandas-ta-classic).

    Notes:
        - Equals pandas-ta-classic's stderr (ddof=1). Tulip Indicators' stderr uses the
          population standard deviation (ddof=0), which is sqrt((window - 1) / window)
          times this value.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods. Defaults to 14.

    Returns:
        pd.Series: Standard Error values, in price units.
    """
    return prices.rolling(window=window).std(ddof=1) / np.sqrt(window)


def get_rolling_entropy(
    prices: pd.Series, window: int = 10, base: float = 2.0
) -> pd.Series:
    """
    Calculate the rolling Shannon Entropy of a price series: the entropy of the prices of the
    window taken as shares of their sum.

    The entropy is at its maximum, log(window), when all prices of the window are equal,
    and falls as the prices spread out, so a lower entropy points to a larger move within
    the window. In base 2 it is measured in bits.

    The formula is as follows:

    - P(t - i) = Close(t - i) / Sum(Close, window), i = 0, ..., window - 1
    - Entropy = -Sum(P * log_base(P))

    Also known as: ENTP, Shannon entropy, information entropy.

    Reference: Shannon, C.E. (1948). "A Mathematical Theory of Communication." Bell System
    Technical Journal, 27(3), 379-423.

    Referenced against pandas-ta-classic's entropy
    (https://github.com/xgboosted/pandas-ta-classic).

    Notes:
        - Equals pandas-ta-classic's entropy (window 10, base 2).
        - The shares are only probabilities for positive prices: a window with a negative
          price or a sum of 0 has no entropy.

    Args:
        prices (pd.Series): Series of prices.
        window (int): Number of periods. Defaults to 10.
        base (float): The base of the logarithm. Defaults to 2.

    Returns:
        pd.Series: Entropy values.
    """
    values = prices.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)

    if len(values) >= window:
        windows = sliding_window_view(values, window)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = windows / windows.sum(axis=1, keepdims=True)
            terms = np.where(shares == 0, 0.0, -shares * np.log(shares))
        entropy = terms.sum(axis=1) / np.log(base)
        result[window - 1 :] = np.where(
            (shares < 0).any(axis=1) | ~np.isfinite(entropy), np.nan, entropy
        )

    return pd.Series(result, index=prices.index)
