"""Statistic Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd
import scipy


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


def get_variance(data: pd.Series) -> float:
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
    return data.var()


def get_standard_deviation(data: pd.Series) -> float:
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
    return data.std()


def get_ar_weights_lsm(series: np.ndarray, p: int) -> tuple:
    """
    Fit an AR(p) model to a time series.

    The LSM finds the coefficients (parameters) that minimize the sum of the squared residuals
    between the actual and predicted values in a time series, providing a straightforward
    and often efficient way to estimate the parameters of an Autoregressive (AR) model.

    Upsides:
    - Doesn't require stationarity.
    - Consistent and unbiased for large datasets.

    Downsides:
    - Computationally expensive for large datasets.
    - Sensible to outliers.

    Args:
        series (np.ndarry): The time series data to model.
        p (int): The order of the autoregressive model, indicating how many past values to consider.

    Returns:
        tuple: A tuple containing three elements:
            - numpy array of estimated parameters phi,
            - float representing the estimated constant c,
            - float representing the sigma squared of the white noise.
    """
    X = np.column_stack(
        [np.ones(len(series) - p)] + [series[i : len(series) - p + i] for i in range(p)]
    )
    Y = series[p:]

    # Solving for AR coefficients and constant term using the Least Squares Method
    params = np.linalg.lstsq(X, Y, rcond=None)[0]
    phi = params[1:]
    c = params[0]

    residuals = Y - X @ params
    sigma2 = residuals @ residuals / len(Y)

    return phi, c, sigma2


def estimate_ar_weights_yule_walker(series: pd.Series, p: int) -> tuple:
    """
    Estimate the weights (parameters) of an Autoregressive (AR) model using the Yule-Walker Method.

    This method computes the Yule-Walker equations which are a set of linear equations
    to estimate the parameters of an AR model based on the autocorrelation function of
    the input series. It is specifically designed for AR models and is highly efficient
    for stationary time series data. Make sure the series is stationary before using this method.

    Args:
        series (pd.Series): The time series data to model.
        p (int): The order of the autoregressive model.

    Returns:
        tuple: A tuple containing three elements:
            - numpy array of estimated parameters phi,
            - float representing the estimated constant c,
            - float representing the sigma squared of the white noise.
    """
    autocov = [
        np.correlate(series, series, mode="full")[len(series) - 1 - i] / len(series)
        for i in range(p + 1)
    ]

    # Create the Yule-Walker matrices
    R = scipy.linalg.toeplitz(autocov[:p])
    r = autocov[1 : p + 1]

    # Solve the Yule-Walker equations
    phi = np.linalg.solve(R, r)
    mu = series.mean()
    c = mu * (1 - np.sum(phi))
    sigma2 = autocov[0] - phi @ r

    return phi, c, sigma2


def get_ar(
    data: np.ndarray | pd.Series | pd.DataFrame,
    p: int = 1,
    steps: int = 1,
    phi: np.ndarray | None = None,
    c: float | None = None,
    method: str = "lsm",
) -> np.ndarray | pd.Series | pd.DataFrame:
    """
    Predict values using an AR(p) model.

    Generates future values of a time series based on an Autoregressive (AR) model.
    This function uses recent observations and the AR model coefficients, forecasted,
    to forecast the next 'steps' values. The predictions are made iteratively, where
    each subsequent prediction becomes a new observation.

    Often the following is used to estimate the order, p, of the AR model:
    1. Plot the Partial Autocorrelation Function (PACF): Plot the PACF of the time series.
    2. Identify Significant Lags: Look for the lag after which most partial autocorrelations
    are not significantly different from zero. A common rule of thumb is that the PACF
    'cuts off' after lag p.

    Args:
        data (np.ndarray | pd.Series | pd.DataFrame):
            The data to predict values for with AR(p). The number of observations should be at
            least equal to the order of the AR model. Note that it calculates the AR model
            for each column of the DataFrame separately.
        p (int): The order of the autoregressive model, indicating how many past values
            to consider. It is only used if c or phi isn't provided. Defaults to 1.
        steps (int, optional): The number of future time steps to predict. Defaults to 1.
        phi (np.ndarray | None): Estimated parameters of the AR model.
        c (float | None): The constant term of the AR model.
        method (str, optional): The method to use to estimate the AR parameters. Can be
            'lsm' (Least Squares Method) or 'yw' (Yule-Walker Method). Defaults to 'lsm'.
            See the weight calculation functions documentation for more details.

    Returns:
        np.ndarray | pd.Series | pd.DataFrame: Predicted values for the specified
        number of future steps.
    """
    if isinstance(data, pd.DataFrame):
        if data.index.nlevels != 1:
            raise ValueError("Expects single index DataFrame, no other value.")
        return data.aggregate(
            lambda x: get_ar(x, steps=steps, phi=phi, p=p, method=method)
        )
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    if phi is None:
        if method == "lsm":
            phi, c, _ = get_ar_weights_lsm(data, p)
        elif method == "yw":
            phi, c, _ = estimate_ar_weights_yule_walker(data, p)
        else:
            raise ValueError("Method must be 'lsm' or 'yw'.")

    predictions = np.zeros(steps)
    recent_values = list(data[-p:])

    for i in range(steps):
        next_value = c + phi @ recent_values
        predictions[i] = next_value

        recent_values.append(next_value)
        recent_values.pop(0)

    return predictions


def ma_likelihood(params, data: np.ndarray) -> float:
    """
    Calculate the negative log-likelihood for an MA(q) model and the residuals/errors.

    Args:
        params (np.ndarray): Model parameters where the last element is the variance of the
            white noise and the others are the MA parameters (theta_1, ..., theta_q).
        data (np.ndarray): Observed time series data.

    Returns:
        tuple:
            - float: The negative log-likelihood of the MA model.
            - np.ndarray: Residuals/errors from the MA model.
    """
    q = len(params) - 1
    theta = params[:-1]
    sigma2 = params[-1]
    n = len(data)

    errors = np.zeros(n)

    for t in range(q, n):
        errors[t] = data[t] - theta @ errors[t - q : t][::-1]
    likelihood = -n / 2 * np.log(2 * np.pi * sigma2) - np.sum(errors[q:] ** 2) / (
        2 * sigma2
    )

    return -likelihood, errors


def fit_ma_model(data: np.ndarray, q: int) -> tuple:
    """
    Fit an MA(q) model to the time series data.

    Finds the parameters of the MA model that minimize the negative log-likelihood of
    the observed data and calculate residuals.

    This MLE method is described in:
    @inbook{NBERc12707,
    Crossref = "NBERaesm76-1",
    title = "Maximum Likelihood Estimation of Moving Average Processes",
    author = "Denise R. Osborn",
    BookTitle = "Annals of Economic and Social Measurement, Volume 5, number 1",
    Publisher = "NBER",
    pages = "75-87",
    year = "1976",
    URL = "http://www.nber.org/chapters/c12707",
    }
    @book{NBERaesm76-1,
    title = "Annals of Economic and Social Measurement, Volume 5, number 1",
    author = "Sanford V. Berg",
    institution = "National Bureau of Economic Research",
    type = "Book",
    publisher = "NBER",
    year = "1976",
    URL = "https://www.nber.org/books-and-chapters/annals-economic-and-social-measurement-volume-5-number-1",
    }

    Args:
        data (np.ndarray): Observed time series data.
        q (int): The order of the MA model.

    Returns:
        tuple:
            - np.ndarray: The parameters theta of the fitted MA model.
            - float: The variance sigma of the fitted MA model.
            - np.ndarray: The residuals/errors from the fitted MA model.
    """
    initial_params = np.zeros(q + 1)
    initial_params[-1] = np.var(data)

    # Minimize the negative log-likelihood
    result = scipy.optimize.minimize(
        lambda params, data: ma_likelihood(params, data)[0],
        initial_params,
        args=(data,),
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError("Optimization failed to converge.")
    _, errors = ma_likelihood(result.x, data)

    return result.x[:-1], result.x[-1], errors[q:]


def get_ma(
    data: np.ndarray | pd.Series | pd.DataFrame,
    q: int,
    steps: int = 1,
    theta: np.ndarray = None,
    errors: np.ndarray = None,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """
    Predict values using an MA(q) model.

    Generates future values of a time series based on a Moving Average (MA) model.
    This function uses the series of errors (innovations) and the MA model coefficients
    to forecast the next 'steps' values. The predictions are made based on the assumption
    that future errors are expected to be zero, which is a common approach in MA forecasting.

    Args:
        data (np.ndarray | pd.Series | pd.DataFrame): The data to predict values for with MA(q).
            Note that it calculates the AR model for each column of the DataFrame separately.
        q (int): The order of the moving average model, indicating how many past error terms to consider.
            It is only used if theta or errors isn't provided.
        steps (int, optional): The number of future time steps to predict. Defaults to 1.
        theta (np.ndarray | None): Estimated parameters of the MA model.
        errors (np.ndarray | None): Array of past errors (residuals) from the model.

    Returns:
        np.ndarray | pd.Series | pd.DataFrame: Predicted values for the specified number of future steps.
    """
    if isinstance(data, pd.DataFrame):
        if data.index.nlevels != 1:
            raise ValueError("Expects single index DataFrame.")
        return data.aggregate(
            lambda x: get_ma(x, q=q, steps=steps, theta=theta, errors=errors)
        )
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    if theta is None or errors is None:
        theta, _, errors = fit_ma_model(data, q)

    if len(data) < q:
        raise ValueError("Data length must be at least equal to the MA order (q).")

    mu = np.mean(data)
    predictions = np.full(steps, mu)

    for i in range(steps):
        if i < q:
            relevant_errors = (
                errors[-q:]
                if i == 0
                else np.concatenate((errors[-(q - i) :], np.zeros(i)))
            )
            predictions[i] += theta[: len(relevant_errors)] @ relevant_errors[::-1]

    return predictions
