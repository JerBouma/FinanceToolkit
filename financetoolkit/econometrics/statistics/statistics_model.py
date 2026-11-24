"""Statistics Model: the linear regression and slope, and statistics over the whole sample or a rolling window"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from financetoolkit.econometrics.helpers import latest_unless_rolling
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

    Referenced against:
    - TA-Lib, LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE, TSF: https://ta-lib.org/functions/linearreg.html
    - Tulip Indicators, linreg: https://tulipindicators.org/linreg
    - pandas-ta-classic, linreg: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/overlap/linreg.py#L92

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
    Calculate the Slope of a price series: the average change per period over a window.

    The slope is the rise over the run of the line between the price a window ago and the
    latest.

    The formula is as follows:

    - Slope = (Close - Close(t - window)) / window
    - Angle = arctan(Slope) in degrees

    Also known as: slope, rise over run, average change per period.

    Reference: the slope of a straight line between two points (rise over run), a standard
    measure; pandas-ta-classic defines the same indicator.

    Referenced against:
    - pandas-ta-classic, slope: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/momentum/slope.py#L12

    Notes:
        - Equals pandas-ta-classic's slope, also as an angle in degrees; pandas-ta-classic
          gives the angle in radians unless asked for degrees.
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


def get_correlation(
    prices: pd.Series, other_prices: pd.Series, window: int | None = None
) -> pd.Series | float:
    """
    Calculate the Pearson Correlation between two series: how closely they move together in
    a straight-line relation, over the whole sample or a rolling window.

    The correlation is +1 when the two series move up and down together in a straight-line
    relation, -1 when they move in opposite directions and around 0 when they are unrelated.
    On price levels, two series that both trend have a high correlation even when their
    daily moves are unrelated, so it shows whether two prices follow the same course rather
    than whether their returns are related.

    The formula is as follows:

    - Correlation = Covariance(X, Y) / (StdDev(X) * StdDev(Y)), over the window

    Also known as: CORREL, Pearson correlation coefficient, rolling correlation.

    Reference: Pearson, K. (1895). "Notes on Regression and Inheritance in the Case of Two
    Parents." Proceedings of the Royal Society of London, 58, 240-242.

    Referenced against:
    - TA-Lib, CORREL: https://ta-lib.org/functions/correl.html
    - pandas-ta-classic, correl: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/correl.py#L11

    Notes:
        - The rolling correlation equals TA-Lib's CORREL and pandas-ta-classic's correl.
        - A window in which either series is flat has no correlation and is missing, where
          TA-Lib and pandas-ta-classic give 0.

    Args:
        prices (pd.Series): Series of prices.
        other_prices (pd.Series): The series of prices to correlate with, on the same dates.
        window (int | None): Number of periods of a rolling window. Defaults to None, which
            correlates every observation and returns one number.

    Returns:
        pd.Series | float: Correlation values between -1 and 1 per date, or one value over
        the whole sample without a window.
    """
    length = window if window is not None else len(prices)
    correlation = prices.rolling(window=length).corr(other_prices)
    flat = (prices.rolling(window=length).std() == 0) | (
        other_prices.rolling(window=length).std() == 0
    )

    return latest_unless_rolling(
        correlation.mask(flat).where(np.isfinite(correlation)), window
    )


def get_percentile(
    prices: pd.Series,
    window: int | None = None,
    percentile: float = 0.5,
    method: str = "linear",
) -> pd.Series | float:
    """
    Calculate the Percentile of a series: the value below which a given share of the
    observations lies, such as the median (0.5), over the whole sample or a rolling window.

    The median of a rolling window is a moving average that a single outlier, such as one
    day with a spike, does not pull along, as it only depends on the order of the prices.

    The formula is as follows, per method, with n the number of observations:

    - Linear: interpolated between the two values around position percentile * (n - 1) of
      the sorted values, with positions from 0 (Hyndman and Fan's definition 7); the 0.5
      percentile is the median, the middle value or the mean of the two middle values
    - Nearest rank: the k-th smallest value, k = ceil(percentile * n) (Hyndman and Fan's
      definition 1, the inverse of the empirical distribution function)

    Also known as: PERCENTILE, MEDIAN, quantile, rolling quantile, moving percentile, moving
    median, simple moving median, SMM.

    Reference:
    - Hyndman, R.J. & Fan, Y. (1996). "Sample Quantiles in Statistical Packages." The
      American Statistician, 50(4), 361-365.
    - Tukey, J.W. (1977). "Exploratory Data Analysis." Addison-Wesley, for running medians.

    Referenced against:
    - TA-Lib, PERCENTILE: https://ta-lib.org/functions/percentile.html
    - TA-Lib, MEDIAN: https://ta-lib.org/functions/median.html
    - pandas-ta-classic, quantile: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/quantile.py#L12
    - pandas-ta-classic, median: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/median.py#L12
    - finta, SMM: https://github.com/peerchemist/finta/blob/v0.25/finta/finta.py#L13

    Notes:
        - The nearest rank agrees with TA-Lib's PERCENTILE, and the linear interpolation
          equals pandas-ta-classic's quantile (window 30) and numpy's default quantile.
        - The linear 0.5 percentile equals TA-Lib's MEDIAN, pandas-ta-classic's median and
          finta's SMM (whose default window is 9).
        - The percentile is a decimal (0.5 is the median) where TA-Lib takes a percentage.
        - A window with a missing value has no percentile.

    Args:
        prices (pd.Series): Series of prices.
        window (int | None): Number of periods of a rolling window. Defaults to None, which
            takes every observation and returns one number.
        percentile (float): The percentile as a decimal between 0 and 1. Defaults to 0.5,
            the median.
        method (str): The percentile definition. Options are: "linear" (Hyndman and Fan's
            definition 7), "nearest_rank" (Hyndman and Fan's definition 1). Defaults to
            "linear".

    Returns:
        pd.Series | float: Percentile values per date, or one value over the whole sample
        without a window.
    """
    check_option("method", method, ["linear", "nearest_rank"])
    if not 0 <= percentile <= 1:
        raise ValueError("The percentile is a decimal between 0 and 1.")

    values = prices.to_numpy(dtype=float)
    length = window if window is not None else len(values)
    result = np.full(len(values), np.nan)

    if 0 < length <= len(values):
        windows = sliding_window_view(values, length)
        position = percentile * (length - 1)
        if method == "linear" and position % 1 == 0.5:  # noqa: PLR2004
            # Halfway between two values, such as the median of an even number of them, the
            # percentile is their mean, (a + b) / 2, where numpy's interpolation, b - (b - a) / 2,
            # can differ from it in the last digit.
            ordered = np.sort(windows, axis=1)
            middle = int(position)
            quantiles = (ordered[:, middle] + ordered[:, middle + 1]) / 2
        else:
            quantiles = np.quantile(
                windows,
                percentile,
                axis=1,
                method="linear" if method == "linear" else "inverted_cdf",
            )
        result[length - 1 :] = np.where(
            np.isnan(windows).any(axis=1), np.nan, quantiles
        )

    return latest_unless_rolling(pd.Series(result, index=prices.index), window)


def get_percent_rank(prices: pd.Series, window: int | None = None) -> pd.Series | float:
    """
    Calculate the Percent Rank of a price series: the share of the earlier prices that are
    lower than the latest price, over the whole sample or a rolling window.

    A Percent Rank of 1 means the price is above every earlier price of the window, a new
    high for the window, and 0 that it is at or below all of them. It ranks the price within
    its history the way a percentile does, and is a component of Connors' RSI.

    The formula is as follows:

    - Percent Rank = Count(Close(t - i) < Close(t), i = 1, ..., window) / window

    with window = n - 1, every earlier observation, over the whole sample.

    Also known as: PERCENTRANK, percentile rank, percent rank.

    Reference: Connors, L., Alvarez, C. & Radtke, M. (2012). "An Introduction to ConnorsRSI."
    Connors Research Trading Strategy Series.

    Referenced against:
    - TA-Lib, PERCENTRANK: https://ta-lib.org/functions/percentrank.html

    Notes:
        - Equals TA-Lib's PERCENTRANK divided by 100: the Toolkit gives a decimal between
          0 and 1 where TA-Lib gives a percentage.
        - The latest price is compared with the `window` prices before it, so the first
          value needs `window` + 1 prices; a window with a missing price has no rank.

    Args:
        prices (pd.Series): Series of prices.
        window (int | None): Number of previous periods of a rolling window. Defaults to
            None, which ranks the latest price among every earlier one and returns one number.

    Returns:
        pd.Series | float: Percent Rank values as decimals per date, or the rank of the
        latest price without a window.
    """
    values = prices.to_numpy(dtype=float)
    length = window if window is not None else len(values) - 1
    result = np.full(len(values), np.nan)

    if 0 < length < len(values):
        windows = sliding_window_view(values, length + 1)
        rank = (windows[:, :-1] < windows[:, -1:]).sum(axis=1) / length
        result[length:] = np.where(np.isnan(windows).any(axis=1), np.nan, rank)

    return latest_unless_rolling(pd.Series(result, index=prices.index), window)


def get_z_score(prices: pd.Series, window: int | None = None) -> pd.Series | float:
    """
    Calculate the Z-Score of a price series: the number of standard deviations the latest
    price lies above or below the average, over the whole sample or a rolling window.

    The formula is as follows:

    - Z-Score = (Close - Mean(Close)) / StdDev(Close), over the window

    with the sample standard deviation (ddof=1).

    Also known as: z-score, standard score, ZSCORE, normalized price.

    Reference: the standard score of the price within its window, a standard statistic;
    pandas-ta-classic's zscore gives the same rolling form.

    Referenced against:
    - pandas-ta-classic, zscore: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/zscore.py#L12

    Notes:
        - The rolling Z-Score equals pandas-ta-classic's zscore (window 30, ddof=1). Its
          cdl_z applies the same score to the open, high, low and close.
        - A window of flat prices has no standard deviation and is missing, where
          pandas-ta-classic gives 0.

    Args:
        prices (pd.Series): Series of prices.
        window (int | None): Number of periods of a rolling window. Defaults to None, which
            scores the latest price against the mean and standard deviation of every
            observation and returns one number.

    Returns:
        pd.Series | float: Z-Score values per date, or the Z-Score of the latest price
        without a window.
    """
    length = window if window is not None else len(prices)
    standard_deviation = prices.rolling(window=length).std(ddof=1)

    return latest_unless_rolling(
        (prices - prices.rolling(window=length).mean())
        / standard_deviation.where(standard_deviation > 0),
        window,
    )


def get_mean_absolute_deviation(
    prices: pd.Series, window: int | None = None
) -> pd.Series | float:
    """
    Calculate the Mean Absolute Deviation of a price series: the average distance of the
    prices from their mean, over the whole sample or a rolling window.

    It measures the dispersion of the prices like the standard deviation, but without
    squaring the deviations, so a single large deviation weighs less. Lambert's Commodity
    Channel Index divides by it.

    The formula is as follows:

    - Mean Absolute Deviation = Mean(|Close(t - i) - Mean(Close)|, i = 0, ..., window - 1)

    with the average of the window itself for every price of the window.

    Also known as: AVGDEV, average deviation, mean deviation, MAD, MD.

    Reference: Lambert, D.R. (1980). "Commodity Channel Index: Tool for Trading Cyclic
    Trends." Commodities, October 1980.

    Referenced against:
    - TA-Lib, AVGDEV: https://ta-lib.org/functions/avgdev.html
    - pandas-ta-classic, mad: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/mad.py#L12
    - pandas-ta-classic, md: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/md.py#L12
    - Tulip Indicators, md: https://tulipindicators.org/md

    Notes:
        - The rolling deviation equals TA-Lib's AVGDEV, pandas-ta-classic's mad and md
          (window 30) and Tulip Indicators' md.
        - The Mean Absolute Deviation of `toolkit.risk` measures the daily returns within
          each period instead of the prices.

    Args:
        prices (pd.Series): Series of prices.
        window (int | None): Number of periods of a rolling window. Defaults to None, which
            takes every observation and returns one number.

    Returns:
        pd.Series | float: Mean Absolute Deviation values per date, in price units, or one
        value over the whole sample without a window.
    """
    values = prices.to_numpy(dtype=float)
    length = window if window is not None else len(values)
    result = np.full(len(values), np.nan)

    if 0 < length <= len(values):
        windows = sliding_window_view(values, length)
        result[length - 1 :] = np.abs(
            windows - windows.mean(axis=1, keepdims=True)
        ).mean(axis=1)

    return latest_unless_rolling(pd.Series(result, index=prices.index), window)


def get_standard_error(
    prices: pd.Series, window: int | None = None
) -> pd.Series | float:
    """
    Calculate the Standard Error of the mean of a price series: how uncertain the average
    price is as an estimate of the mean, over the whole sample or a rolling window.

    The standard error is the standard deviation divided by the square root of the number
    of observations. For a rolling window it is the uncertainty of the moving average as an
    estimate of the mean price of the window: a moving average plus and minus two standard
    errors spans roughly where the mean lies.

    The formula is as follows:

    - Standard Error = StdDev(Close) / sqrt(window), over the window

    with the sample standard deviation (ddof=1).

    Also known as: STDERR, standard error of the mean, SEM.

    Reference: the standard error of the sample mean, a standard statistic;
    pandas-ta-classic's stderr gives the same rolling form.

    Referenced against:
    - pandas-ta-classic, stderr: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/stderr.py#L12

    Shares similar logic:
    - Tulip Indicators, stderr: https://tulipindicators.org/stderr

    Notes:
        - The rolling standard error equals pandas-ta-classic's stderr (ddof=1). Unlike
          Tulip Indicators' stderr, which uses the population standard deviation (ddof=0),
          this is the sample one: Tulip's value is sqrt((window - 1) / window) times this
          value.

    Args:
        prices (pd.Series): Series of prices.
        window (int | None): Number of periods of a rolling window. Defaults to None, which
            takes every observation and returns one number.

    Returns:
        pd.Series | float: Standard Error values per date, in price units, or one value
        over the whole sample without a window.
    """
    length = window if window is not None else len(prices)

    return latest_unless_rolling(
        prices.rolling(window=length).std(ddof=1) / np.sqrt(length), window
    )
