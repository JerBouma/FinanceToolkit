"""Technical Statistics Model Tests"""

# ruff: noqa: PLR2004

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals import statistic_model


def test_get_beta_basic():
    """Test basic beta calculation."""
    returns = pd.Series([0.05, 0.02, -0.01, 0.03, 0.01])
    benchmark_returns = pd.Series([0.04, 0.01, -0.02, 0.02, 0.005])

    beta = statistic_model.get_beta(returns, benchmark_returns)

    assert isinstance(beta, float)
    assert not np.isnan(beta)
    assert beta > 0  # Should be positive for positively correlated returns


def test_get_beta_zero_benchmark_variance():
    """Test beta calculation with zero benchmark variance."""
    returns = pd.Series([0.05, 0.02, -0.01, 0.03, 0.01])
    benchmark_returns = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02])  # No variance

    beta = statistic_model.get_beta(returns, benchmark_returns)

    # Should return inf or handle division by zero
    assert np.isinf(beta) or np.isnan(beta)


def test_get_beta_perfect_correlation():
    """Test beta calculation with perfect correlation."""
    returns = pd.Series([0.05, 0.02, -0.01, 0.03, 0.01])
    benchmark_returns = returns  # Perfect correlation

    beta = statistic_model.get_beta(returns, benchmark_returns)

    assert abs(beta - 1.0) < 0.001  # Should be close to 1


def test_get_pearsons_correlation_basic():
    """Test basic Pearson's correlation calculation."""
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([2, 4, 6, 8, 10])

    correlation = statistic_model.get_pearsons_correlation(series1, series2)

    assert isinstance(correlation, float)
    assert abs(correlation - 1.0) < 0.001  # Perfect positive correlation


def test_get_pearsons_correlation_negative():
    """Test Pearson's correlation with negative correlation."""
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([10, 8, 6, 4, 2])

    correlation = statistic_model.get_pearsons_correlation(series1, series2)

    assert correlation < 0  # Negative correlation
    assert abs(correlation - (-1.0)) < 0.001  # Perfect negative correlation


def test_get_pearsons_correlation_no_correlation():
    """Test Pearson's correlation with no correlation."""
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([2, 1, 4, 3, 5])

    correlation = statistic_model.get_pearsons_correlation(series1, series2)

    assert isinstance(correlation, float)
    assert abs(correlation) < 1.0  # Should be between -1 and 1


def test_get_spearman_correlation_basic():
    """Test basic Spearman correlation calculation."""
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([2, 4, 6, 8, 10])

    correlation = statistic_model.get_spearman_correlation(series1, series2)

    assert isinstance(correlation, float)
    assert abs(correlation - 1.0) < 0.001  # Perfect positive correlation


def test_get_spearman_correlation_negative():
    """Test Spearman correlation with negative correlation."""
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([10, 8, 6, 4, 2])

    correlation = statistic_model.get_spearman_correlation(series1, series2)

    assert correlation < 0  # Negative correlation
    assert abs(correlation - (-1.0)) < 0.001  # Perfect negative correlation


def test_get_spearman_correlation_tied_ranks():
    """Test Spearman correlation with tied ranks."""
    series1 = pd.Series([1, 2, 2, 4, 5])
    series2 = pd.Series([2, 4, 4, 8, 10])

    correlation = statistic_model.get_spearman_correlation(series1, series2)

    assert isinstance(correlation, float)
    assert -1 <= correlation <= 1  # Should be between -1 and 1


def test_get_variance_basic():
    """Test basic variance calculation."""
    data = pd.Series([1, 2, 3, 4, 5])

    variance = statistic_model.get_variance(data)

    assert isinstance(variance, float)
    assert variance > 0  # Variance should be positive for non-constant data
    assert abs(variance - 2.5) < 0.001  # Expected variance


def test_get_variance_constant_data():
    """Test variance calculation with constant data."""
    data = pd.Series([5, 5, 5, 5, 5])

    variance = statistic_model.get_variance(data)

    assert variance == 0  # Variance should be zero for constant data


def test_get_standard_deviation_basic():
    """Test basic standard deviation calculation."""
    data = pd.Series([1, 2, 3, 4, 5])

    std_dev = statistic_model.get_standard_deviation(data)

    assert isinstance(std_dev, float)
    assert std_dev > 0  # Standard deviation should be positive
    assert abs(std_dev - np.sqrt(2.5)) < 0.001  # Should be sqrt of variance


def test_get_standard_deviation_constant_data():
    """Test standard deviation calculation with constant data."""
    data = pd.Series([5, 5, 5, 5, 5])

    std_dev = statistic_model.get_standard_deviation(data)

    assert std_dev == 0  # Standard deviation should be zero for constant data


def test_get_ar_weights_lsm_basic():
    """Test AR weights calculation using LSM method."""
    # Create AR(1) process: y_t = 0.5 * y_{t-1} + 0.1 + noise
    np.random.seed(42)
    series = np.random.randn(100)
    for i in range(1, 100):
        series[i] = 0.5 * series[i - 1] + 0.1 + 0.1 * np.random.randn()

    phi, c, sigma2 = statistic_model.get_ar_weights_lsm(series, p=1)

    assert isinstance(phi, np.ndarray)
    assert len(phi) == 1
    assert isinstance(c, float)
    assert isinstance(sigma2, float)
    assert sigma2 > 0  # Variance should be positive


def test_get_ar_weights_lsm_higher_order():
    """Test AR weights calculation with higher order."""
    np.random.seed(42)
    series = np.random.randn(100)

    phi, c, sigma2 = statistic_model.get_ar_weights_lsm(series, p=3)

    assert len(phi) == 3
    assert isinstance(c, float)
    assert isinstance(sigma2, float)
    assert sigma2 > 0


def test_estimate_ar_weights_yule_walker_basic():
    """Test AR weights estimation using Yule-Walker method."""
    # Create AR(1) process
    np.random.seed(42)
    series = pd.Series(np.random.randn(100))
    for i in range(1, 100):
        series.iloc[i] = 0.5 * series.iloc[i - 1] + 0.1 * np.random.randn()

    phi, c, sigma2 = statistic_model.estimate_ar_weights_yule_walker(series, p=1)

    assert isinstance(phi, np.ndarray)
    assert len(phi) == 1
    assert isinstance(c, float)
    assert isinstance(sigma2, float)
    assert sigma2 > 0


def test_get_ar_with_numpy_array():
    """Test AR prediction with numpy array input."""
    np.random.seed(42)
    data = np.random.randn(50)

    predictions = statistic_model.get_ar(data, p=1, steps=3)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3
    assert not np.isnan(predictions).any()


def test_get_ar_with_pandas_series():
    """Test AR prediction with pandas Series input."""
    np.random.seed(42)
    data = pd.Series(np.random.randn(50))

    predictions = statistic_model.get_ar(data, p=1, steps=3)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3


def test_get_ar_with_provided_parameters():
    """Test AR prediction with provided parameters."""
    np.random.seed(42)
    data = np.random.randn(50)
    phi = np.array([0.5])
    c = 0.1

    predictions = statistic_model.get_ar(data, p=1, steps=3, phi=phi, c=c)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3


def test_get_ar_yule_walker_method():
    """Test AR prediction using Yule-Walker method."""
    np.random.seed(42)
    data = pd.Series(np.random.randn(50))

    predictions = statistic_model.get_ar(data, p=1, steps=3, method="yw")

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3


def test_get_ar_invalid_method():
    """Test AR prediction with invalid method."""
    data = np.random.randn(50)

    with pytest.raises(ValueError, match="Method must be 'lsm' or 'yw'"):
        statistic_model.get_ar(data, p=1, steps=3, method="invalid")


def test_ma_likelihood_basic():
    """Test MA likelihood calculation."""
    np.random.seed(42)
    data = np.random.randn(100)
    params = np.array([0.5, 1.0])  # theta=0.5, sigma2=1.0

    neg_likelihood, errors = statistic_model.ma_likelihood(params, data)

    assert isinstance(neg_likelihood, float)
    assert isinstance(errors, np.ndarray)
    assert len(errors) == len(data)
    assert neg_likelihood > 0  # Negative log-likelihood should be positive


def test_fit_ma_model_basic():
    """Test MA model fitting."""
    np.random.seed(42)
    data = np.random.randn(100)

    theta, sigma2, errors = statistic_model.fit_ma_model(data, q=1)

    assert isinstance(theta, np.ndarray)
    assert len(theta) == 1
    assert isinstance(sigma2, float)
    assert sigma2 > 0
    assert isinstance(errors, np.ndarray)
    assert len(errors) == len(data) - 1


def test_fit_ma_model_higher_order():
    """Test MA model fitting with higher order."""
    np.random.seed(42)
    data = np.random.randn(100)

    theta, sigma2, errors = statistic_model.fit_ma_model(data, q=3)

    assert len(theta) == 3
    assert isinstance(sigma2, float)
    assert sigma2 > 0
    assert len(errors) == len(data) - 3


def test_fit_ma_model_optimization_failure():
    """Test MA model fitting with optimization failure."""
    # Create problematic data that might cause optimization to fail
    data = np.full(10, 1.0)  # Constant data

    with patch("scipy.optimize.minimize") as mock_minimize:
        mock_result = MagicMock()
        mock_result.success = False
        mock_minimize.return_value = mock_result

        with pytest.raises(RuntimeError, match="Optimization failed to converge"):
            statistic_model.fit_ma_model(data, q=1)


def test_get_ma_with_numpy_array():
    """Test MA prediction with numpy array input."""
    np.random.seed(42)
    data = np.random.randn(50)

    predictions = statistic_model.get_ma(data, q=1, steps=3)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3


def test_get_ma_with_pandas_series():
    """Test MA prediction with pandas Series input."""
    np.random.seed(42)
    data = pd.Series(np.random.randn(50))

    predictions = statistic_model.get_ma(data, q=1, steps=3)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3


def test_get_ma_with_provided_parameters():
    """Test MA prediction with provided parameters."""
    np.random.seed(42)
    data = np.random.randn(50)
    theta = np.array([0.5])
    errors = np.random.randn(49)

    predictions = statistic_model.get_ma(data, q=1, steps=3, theta=theta, errors=errors)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3


def test_get_ma_zero_steps():
    """Test MA prediction with zero steps."""
    np.random.seed(42)
    data = np.random.randn(50)

    predictions = statistic_model.get_ma(data, q=1, steps=0)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 0


def test_get_ar_zero_steps():
    """Test AR prediction with zero steps."""
    np.random.seed(42)
    data = np.random.randn(50)

    predictions = statistic_model.get_ar(data, p=1, steps=0)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 0


def test_correlation_functions_with_nans():
    """Test correlation functions with NaN values."""
    series1 = pd.Series([1, 2, np.nan, 4, 5])
    series2 = pd.Series([2, 4, 6, np.nan, 10])

    # Should handle NaN values gracefully
    pearson_corr = statistic_model.get_pearsons_correlation(series1, series2)
    spearman_corr = statistic_model.get_spearman_correlation(series1, series2)

    # Results may be NaN due to NaN values in input
    assert isinstance(pearson_corr, float)
    assert isinstance(spearman_corr, float)


def test_variance_and_std_with_single_value():
    """Test variance and standard deviation with single value."""
    data = pd.Series([5])

    variance = statistic_model.get_variance(data)
    std_dev = statistic_model.get_standard_deviation(data)

    # Single value should have zero variance and std dev
    assert pd.isna(variance) or variance == 0
    assert pd.isna(std_dev) or std_dev == 0


def test_ar_weights_lsm_minimum_data():
    """Test AR weights LSM with minimum required data."""
    series = np.array([1, 2, 3])  # Minimum for p=1

    phi, c, sigma2 = statistic_model.get_ar_weights_lsm(series, p=1)

    assert isinstance(phi, np.ndarray)
    assert len(phi) == 1
    assert isinstance(c, float)
    assert isinstance(sigma2, float)


def test_ar_weights_insufficient_data():
    """Test AR weights with insufficient data."""
    series = np.array([1, 2])  # Insufficient for p=2

    # Should still work but might give unreliable results
    phi, c, sigma2 = statistic_model.get_ar_weights_lsm(series, p=2)

    assert isinstance(phi, np.ndarray)
    assert len(phi) == 2
