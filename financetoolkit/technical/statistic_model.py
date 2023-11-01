"""Statistic Module"""
__docformat__ = "google"

import numpy as np
import pandas as pd


def get_beta(returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """
    Calculate the beta of returns with respect to a benchmark over a rolling window.

    Beta measures the sensitivity of an asset's returns to changes in the returns of a benchmark.
    It indicates the asset's risk in relation to the benchmark. A beta greater than 1 suggests
    the asset is more volatile than the benchmark, while a beta less than 1 indicates lower volatility.

    Args:
        returns (pd.Series): A Series of returns.
        benchmark_returns (pd.Series): A Series of benchmark returns.

    Returns:
        pd.Series: A Series of beta values with assets as index.
    """
    excess_returns = returns - returns.mean()
    excess_benchmark_returns = benchmark_returns - benchmark_returns.mean()

    cov = excess_returns.cov(excess_benchmark_returns)

    var = excess_benchmark_returns.var()

    beta_values = cov / var

    return beta_values


def get_pearsons_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate Pearson's Correlation Coefficient between two given series.

    Pearson's Correlation Coefficient measures the linear relationship between two sets of data points.
    It ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating
    no linear correlation. Positive values suggest a positive linear relationship, while negative values
    suggest a negative linear relationship.

    Args:
        series1 (pd.Series): First series of data points.
        series2 (pd.Series): Second series of data points.

    Returns:
        float: Pearson's Correlation Coefficient.
    """
    correlation_matrix = np.corrcoef(series1, series2)
    correlation_coefficient = correlation_matrix[0, 1]

    return correlation_coefficient


def get_spearman_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calculate the Spearman Correlation Coefficient between two given data series.

    Spearman Correlation Coefficient measures the strength and direction of monotonic relationship
    between two sets of data points. It is a non-parametric measure, meaning it assesses the similarity
    in the ranks of data points rather than their actual values.

    Args:
        series1 (pd.Series): First series of data points.
        series2 (pd.Series): Second series of data points.

    Returns:
        float: Spearman Correlation Coefficient.
    """
    rank_series1 = series1.rank()
    rank_series2 = series2.rank()

    d = rank_series1 - rank_series2
    d_squared = d**2

    n = len(series1)
    spearman_corr = 1 - (6 * d_squared.sum()) / (n * (n**2 - 1))

    return spearman_corr


def get_variance(prices: pd.Series) -> float:
    """
    Calculate the Variance of a given data series.

    Variance measures the spread or dispersion of data points around the mean.
    A higher variance indicates more variability in the data, while a lower variance
    suggests that the data points are closer to the mean.

    Args:
        data (pd.Series): Series of data points.

    Returns:
        float: Variance value.
    """
    return prices.var()


def get_standard_deviation(prices: pd.Series) -> float:
    """
    Calculate the Standard Deviation of a given data series.

    Standard Deviation measures the amount of dispersion or variability in a set of data points.
    It is the square root of the Variance. A higher standard deviation indicates greater variability,
    while a lower standard deviation suggests that data points are closer to the mean.

    Args:
        data (pd.Series): Series of data points.

    Returns:
        float: Standard Deviation value.
    """
    return prices.std()
