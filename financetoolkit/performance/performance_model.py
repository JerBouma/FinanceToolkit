"""Performance Model"""

import io
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from financetoolkit.helpers import PERIOD_TRANSLATION, get_request

# This is meant for calculations in which a Multi Index exists. This is the case
# when calculating a "within period" in which the first index represents the period
# (e.g. 2020Q1) and the second index the days within that period (January to March)
MULTI_PERIOD_INDEX_LEVELS = 2

# pylint: disable=isinstance-second-argument-not-valid-type


def get_covariance(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the covariance of returns.

    A warnings filter is included given that the following error
    can occur:

    RuntimeWarning: Degrees of freedom <= 0 for slice
        return np.cov(a, b, ddof=ddof)[0, 1]

    Given that this is due to division by zero or NaN values, it does
    not have any impact on the result. The warning is therefore
    ignored.

    Args:
        returns (pd.Series | pd.DataFrame): _description_
        benchmark_returns (pd.Series | pd.DataFrame): _description_

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(returns, pd.DataFrame):
            covariance = pd.Series(index=returns.columns)

            for column in returns.columns:
                covariance.loc[column] = returns[column].cov(benchmark_returns)

            return covariance

        if isinstance(returns, pd.Series | pd.core.window.rolling.Rolling):
            return returns.cov(benchmark_returns)

    return returns.cov(benchmark_returns)


def get_beta(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate beta. Beta represents the slope in the linear regression between
    the asset returns and the benchmark returns.

    Args:
        returns (pd.Series | pd.DataFrame): return series.
        benchmark_returns (pd.Series | pd.DataFrame): benchmark return series.

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            combination = pd.concat([returns, benchmark_returns], axis=1)

            # Calculate Sharpe ratio for each asset (ticker) in the DataFrame
            covariance = combination.groupby(level=0).apply(
                lambda x: get_covariance(x[returns.columns], x[benchmark_returns.name])
            )
            variance = benchmark_returns.groupby(level=0).apply(lambda x: x.var())

            return covariance.div(variance, axis=0)

        return get_covariance(returns, benchmark_returns) / benchmark_returns.var()

    if isinstance(returns, pd.Series):
        # Calculate Sharpe ratio for a single asset (ticker)
        return get_covariance(returns, benchmark_returns) / benchmark_returns.var()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_beta(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate rolling beta. Beta represents the slope in the linear regression between
    the asset returns and the benchmark returns.

    Rolling beta is calculated by calculating the covariance between the asset returns
    and the benchmark returns over a rolling window and dividing this by the variance

    Args:
        returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        window_size (int): Rolling window size.

    Returns:
        pd.Series | pd.DataFrame: Rolling beta values.
    """
    returns = pd.DataFrame(returns) if isinstance(returns, pd.Series) else returns
    rolling_cov = pd.DataFrame(
        index=returns.index, columns=returns.columns, dtype=np.float64
    )

    for column in returns.columns:
        rolling_cov.loc[:, column] = get_covariance(
            returns[column].rolling(window=window_size), benchmark_returns
        )

    rolling_var = benchmark_returns.rolling(window=window_size).var()

    rolling_beta = rolling_cov.div(rolling_var, axis=0)

    return rolling_beta


def get_capital_asset_pricing_model(
    risk_free_rate: pd.Series | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
) -> pd.Series | pd.DataFrame:
    """
    CAPM, or the Capital Asset Pricing Model, is a financial model used to estimate the expected return on an investment,
    such as a stock or portfolio of stocks. It provides a framework for evaluating the risk and return trade-off of
    an asset or portfolio in relation to the overall market. CAPM is based on the following key components:

        - Risk-Free Rate (Rf): This is the theoretical return an investor could earn from an investment with no
        risk of financial loss. It is typically based on the yield of a government bond.
        - Market Risk Premium (Rm - Rf): This represents the additional return that investors expect to earn
        for taking on the risk of investing in the overall market as opposed to a risk-free asset. It
        is calculated as the difference between the expected return of the market (Rm) and the risk-free rate (Rf).
        - Beta (β): Beta is a measure of an asset's or portfolio's sensitivity to market movements. It
        quantifies how much an asset's returns are expected to move in relation to changes in the
        overall market. A beta of 1 indicates that the asset moves in line with the market, while a
        beta greater than 1 suggests higher volatility, and a beta less than 1 indicates lower volatility.

    The formula is as follows:

    - Expected Return (ER) = Rf + β * (Rm — Rf)

    Args:
        risk_free_rate (pd.Series | float): the risk free rate.
        beta (pd.Series | pd.DataFrame | float): the beta.
        benchmark_returns (pd.Series | float): the benchmark returns.

    Returns:
        pd.Series | pd.DataFrame | float: the capital asset pricing model.
    """
    if isinstance(beta, pd.DataFrame):
        capital_asset_pricing_model = pd.DataFrame(
            index=beta.index, columns=beta.columns, dtype=np.float64
        )
        for column in capital_asset_pricing_model.columns:
            capital_asset_pricing_model.loc[:, column] = risk_free_rate + beta[
                column
            ] * (benchmark_returns - risk_free_rate)
    if isinstance(beta, (pd.Series | float)):
        capital_asset_pricing_model = risk_free_rate + beta * (
            benchmark_returns - risk_free_rate
        )

    return capital_asset_pricing_model


def obtain_fama_and_french_dataset(fama_and_french_url: str | None = None):
    """
    This functionality returns the Fama and French 5 Factor Model dataset. It is a dataset that contains the
    excess returns of the 5 factors that are used in the Fama and French 5 Factor Model. The factors are:

        - Market (MKT): The excess return of the market.
        - Size (SMB): Small companies tend to outperform large companies.
        - Value (HML): Value stocks tend to outperform growth stocks.
        - Investment (CMA): Companies that invest conservatively tend to outperform companies that invest aggressively.
        - Profitability (RMW): Companies with high operating profitability tend to outperform companies with low
        operating profitability.

    Next to that, it also includes the Risk Free Rate. The dataset is available on the website of Dartmouth College
    and is updated on a monthly basis. The dataset is packaged in a ZIP file, so it needs to be extracted first.
    The ZIP file contains a CSV file with the dataset.

    It is also possible to read other datasets from Fama and French with this functionality.

    Args:
        fama_and_french_url (str): the URL of the ZIP file that contains the dataset. If no URL is provided, the
        default URL (Fama and French 5 Factor) is used.

    Returns:
        pd.DataFrame: the Fama and French 5 Factor Model dataset.
    """
    # Define the URL of the ZIP file
    fama_and_french_url = (
        fama_and_french_url
        if fama_and_french_url
        else "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )

    response = get_request(fama_and_french_url, timeout=10)
    zip_data = response.content

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
        # The dataset is packaged in a ZIP file, so it needs to be extracted first
        zip_file_contents = zip_file.namelist()

        csv_file_name = zip_file_contents[0]

        with zip_file.open(csv_file_name) as csv_file:
            # Skip the first 3 rows which are header information
            fama_and_french_dataset = pd.read_csv(csv_file, skiprows=3, index_col=0)

        fama_and_french_dataset = fama_and_french_dataset.dropna(axis="index")

        fama_and_french_dataset.index = pd.to_datetime(
            fama_and_french_dataset.index, format="%Y%m%d"
        ).to_period(freq="D")
        fama_and_french_dataset.index.name = "Date"

    return fama_and_french_dataset


def get_factor_asset_correlations(
    factors: pd.DataFrame,
    excess_return: pd.Series,
) -> pd.DataFrame:
    """
    Calculates factor exposures for each asset.

    The major difference between the Fama and French Model here is that the correlation
    is taken as opposed to a Linear Regression in which the R-squared or Slope can be used to
    understand the exposure to each factor.

    For assessing the exposure or influence of a stock to external factors, it's often preferable
    to use R-squared (R²) or Beta because it explicitly measures how well the factors explain the stock's
    returns. A higher R² indicates that the stock's returns are more closely related to the factors,
    and thus, the factors have a greater influence on the stock's performance.

    However, since the results are closely related and tend to point into the same direction it could
    be fine to use correlations as well depending on the level of accuracy required.

    Args:
        factors (pd.DataFrame): the factor dataset with each factor in a column.
        excess_returns (pd.Series): the excess returns.

    Returns:
        pd.DataFrame: the factor asset correlations.
    """
    correlations = factors.corrwith(excess_return)

    return correlations


def get_fama_and_french_model_multi(
    excess_returns: pd.Series,
    factor_dataset: pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Fama and French 5 Factor Model is an extension of the CAPM model. It adds four additional factors to the
    regression analysis to better describe asset returns:
        - Size (SMB): Small companies tend to outperform large companies.
        - Value (HML): Value stocks tend to outperform growth stocks.
        - Investment (CMA): Companies that invest conservatively tend to outperform companies that invest aggressively.
        - Profitability (RMW): Companies with high operating profitability tend to outperform companies with low
        operating profitability.

    This functionality performs the regression analysis and returns the regression parameters for a given return
    and factor series.

    The formula is as follows:

        - Excess Return = Intercept + Beta1 * Mkt-RF + Beta2 * SMB + Beta3 * HML +
            Beta4 * RMW + Beta5 * CMA + Residuals

    Args:
        excess_returns (pd.Series): the excess returns.
        factor_dataset (pd.DataFrame): the factor dataset with each factor in a column.

    Returns:
        dict: the regression results.
        pd.Series: the residuals.
    """
    error_message = None

    if excess_returns.isna().any():
        excess_returns = excess_returns.bfill(limit=None)
        excess_returns = excess_returns.ffill(limit=None)

    if factor_dataset.isna().any().any():
        excess_returns = excess_returns.bfill(limit=None, axis=1)
        factor_dataset = factor_dataset.ffill(limit=None, axis=1)

    model = LinearRegression()
    model.fit(factor_dataset, excess_returns)

    # Check for sufficient samples before calculating R^2
    if factor_dataset.shape[0] < 2:  # noqa
        error_message = (
            "R2 score is not well-defined with less than two samples. "
            "Setting value to NaN"
        )
        r_squared = np.nan
    else:
        # Calculate R^2 using the model's score method
        r_squared = model.score(factor_dataset, excess_returns)

    y_pred = model.predict(factor_dataset)

    residuals = excess_returns - y_pred

    mse = mean_squared_error(excess_returns, y_pred)

    regression_results = {"Intercept": model.intercept_}

    for factor in factor_dataset.columns:
        regression_results[f"{factor} Slope"] = model.coef_[
            factor_dataset.columns.get_loc(factor)
        ]

    regression_results["Mean Squared Error (MSE)"] = mse
    regression_results["R Squared"] = r_squared

    return regression_results, residuals, error_message


def get_fama_and_french_model_single(
    excess_returns: pd.Series,
    factor: pd.Series,
) -> pd.Series | pd.DataFrame:
    """
    The Fama and French 5 Factor Model is an extension of the CAPM model. It adds four additional factors to the
    regression analysis to better describe asset returns:
        - Size (SMB): Small companies tend to outperform large companies.
        - Value (HML): Value stocks tend to outperform growth stocks.
        - Investment (CMA): Companies that invest conservatively tend to outperform companies that invest aggressively.
        - Profitability (RMW): Companies with high operating profitability tend to outperform companies with low
        operating profitability.

    This functionality performs the regression analysis and returns the regression parameters for a given return
    and factor series.

    The formula is as follows:

        - Excess Return = Intercept + Slope * Factor Value + Residuals

    Args:
        excess_returns (pd.Series): the excess returns.
        factor (pd.Series): the factor series.

    Returns:
        dict: the regression results.
        pd.Series: the residuals.
    """
    if len(excess_returns) < 2:  # noqa
        # Robust handling of insufficient data points for regression method
        regression_results = {
            "Intercept": np.nan,
            "Slope": np.nan,
            "R Squared": np.nan,
            "P Value": np.nan,
            "Standard Error": np.nan,
        }
        return regression_results, excess_returns * np.nan

    result = linregress(excess_returns, factor)

    regression_results = {
        "Intercept": result.intercept,
        "Slope": result.slope,
        "R Squared": result.rvalue**2,
        "P Value": result.pvalue,
        "Standard Error": result.stderr,
    }

    residuals = excess_returns - (result.slope * factor + result.intercept)

    return regression_results, residuals


def get_alpha(
    asset_returns: pd.Series | float,
    benchmark_returns: pd.Series | float,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Alpha.

    Args:
        excess_returns (pd.Series): A Series of returns with risk-free rate subtracted.

    Returns:
        pd.Series: A Series of Sharpe ratios with time as index and assets as columns.
    """
    if isinstance(asset_returns, pd.DataFrame):
        alpha = pd.DataFrame(
            index=asset_returns.index, columns=asset_returns.columns, dtype=np.float64
        )
        for column in alpha.columns:
            alpha.loc[:, column] = asset_returns[column] - benchmark_returns
    if isinstance(asset_returns, (pd.Series | float)):
        alpha = asset_returns - benchmark_returns

    return alpha


def get_rolling_alpha(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window_size: int,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Alpha of returns.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Alpha values with time as index.
    """
    return (
        asset_returns.sub(benchmark_returns, axis=0).rolling(window=window_size).mean()
    )


def get_jensens_alpha(
    asset_returns: pd.Series | float,
    risk_free_rate: pd.Series | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Jensen's Alpha.

    Args:
        excess_returns (pd.Series): A Series of returns with risk-free rate subtracted.

    Returns:
        pd.Series: A Series of Sharpe ratios with time as index and assets as columns.
    """
    if isinstance(beta, pd.DataFrame) and isinstance(asset_returns, pd.DataFrame):
        jensens_alpha = pd.DataFrame(
            index=beta.index, columns=beta.columns, dtype=np.float64
        )
        for column in jensens_alpha.columns:
            jensens_alpha.loc[:, column] = asset_returns[column] - (
                risk_free_rate + beta[column] * (benchmark_returns - risk_free_rate)
            )
    elif isinstance(beta, (pd.Series | float)) and isinstance(
        beta, (pd.Series | float)
    ):
        jensens_alpha = asset_returns - (
            risk_free_rate + beta * (benchmark_returns - risk_free_rate)
        )
    else:
        raise TypeError(
            "Expects pd.DataFrame for both Asset Returns and Beta or pd.Series / Float "
            "for both Asset Returns and Beta"
        )

    return jensens_alpha


def get_treynor_ratio(
    asset_returns: pd.Series | float,
    risk_free_rate: pd.Series | float,
    beta: pd.Series | pd.DataFrame | float,
) -> pd.Series:
    """
    Calculate the Treynor ratio of returns.

    Args:
        excess_returns (pd.Series): A Series of returns with risk-free rate subtracted.
        beta (float): The portfolio's beta (systematic risk).

    Returns:
        pd.Series: A Series of Treynor ratios with time as index and assets as columns.
    """
    if isinstance(beta, pd.DataFrame) and isinstance(asset_returns, pd.DataFrame):
        treynor_ratio = pd.DataFrame(
            index=beta.index, columns=beta.columns, dtype=np.float64
        )
        for column in treynor_ratio.columns:
            treynor_ratio.loc[:, column] = (
                asset_returns[column] - risk_free_rate
            ) / beta[column]
    elif isinstance(beta, (pd.Series | float)) and isinstance(
        asset_returns, (pd.Series | float)
    ):
        treynor_ratio = (asset_returns - risk_free_rate) / beta
    else:
        raise TypeError(
            "Expects pd.DataFrame for both Asset Returns and Beta or pd.Series / Float "
            "for both Asset Returns and Beta"
        )

    return treynor_ratio


def get_sharpe_ratio(excess_returns: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Calculate the Sharpe ratio of returns.

    Args:
        excess_returns (pd.Series): A Series of returns with risk-free rate subtracted.

    Returns:
        pd.Series: A Series of Sharpe ratios with time as index and assets as columns.
    """
    if isinstance(excess_returns, pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # Calculate Sharpe ratio for each asset (ticker) in the DataFrame
            sharpe_ratios = excess_returns.groupby(level=0).apply(
                lambda x: x.mean() / x.std()
            )
            return sharpe_ratios

        return excess_returns / excess_returns.std()

    if isinstance(excess_returns, pd.Series):
        # Calculate Sharpe ratio for a single asset (ticker)
        return excess_returns / excess_returns.std()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_sharpe_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    window_size: int,
) -> pd.Series:
    """
    Calculate the rolling Sharpe ratio of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free rate subtracted.
        window_size (int): The size of the rolling window in months. Default is 12.

    Returns:
        pd.Series: A Series of rolling Sharpe ratios with time as index and assets as columns.
    """
    sharpe_ratio = (
        excess_returns.rolling(window=window_size).mean()
        / excess_returns.rolling(window=window_size).std()
    )

    return sharpe_ratio


def get_sortino_ratio(excess_returns: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Calculate the Sortino ratio of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free
        rate already subtracted.

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """
    if isinstance(excess_returns, pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # Calculate Sortino ratio for each asset (ticker) in the DataFrame
            sortino_ratios = excess_returns.groupby(level=0).apply(
                lambda x: x.mean() / x[x < 0].std()
            )
            return sortino_ratios

        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = downside_returns.std()

        return excess_returns.mean() / downside_volatility

    if isinstance(excess_returns, pd.Series):
        # Calculate Sortino ratio for a single asset (ticker)
        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = downside_returns.std()
        return excess_returns.mean() / downside_volatility

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


def get_rolling_sortino_ratio(
    excess_returns: pd.Series | pd.DataFrame, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Sortino ratio of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free rate subtracted.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Sortino ratio values with time as index.
    """

    def _downside_volatility(window):
        downside_returns = window[window < 0]

        return downside_returns.std() if len(downside_returns) > 1 else np.nan

    rolling_mean = excess_returns.rolling(window=window_size).mean()
    rolling_downside_volatility = excess_returns.rolling(window=window_size).apply(
        _downside_volatility, raw=True
    )

    return rolling_mean / rolling_downside_volatility


def get_ulcer_performance_index(
    excess_returns: pd.Series | pd.DataFrame, ulcer_index: pd.Series | pd.DataFrame
) -> pd.Series:
    """
    Calculate the Ulcer Performance Index (UPI) of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series of returns with risk-free rate subtracted.
        ulcer_index (pd.Series | pd.DataFrame): The corresponding

    Returns:
        pd.Series: A Series of Ulcer Performance Index values with time as index and assets as columns.
    """
    return (excess_returns / ulcer_index).dropna()


def get_m2_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    asset_standard_deviation: pd.Series | pd.DataFrame,
) -> pd.Series:
    """
    Calculate the M2 Ratio (Modigliani-Modigliani Measure) of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free
        rate already subtracted.

    Returns:
        pd.Series: A Series of M2 ratios with time as index and assets as columns.
    """
    if isinstance(asset_returns, pd.DataFrame) and isinstance(
        asset_standard_deviation, pd.DataFrame
    ):
        m2_ratio = pd.DataFrame(
            index=asset_returns.index, columns=asset_returns.columns, dtype=np.float64
        )
        for column in m2_ratio.columns:
            m2_ratio.loc[:, column] = (
                asset_returns[column] - risk_free_rate
            ) / asset_standard_deviation[column]
    elif isinstance(asset_returns, (pd.Series | float)) and isinstance(
        asset_standard_deviation, (pd.Series | float)
    ):
        m2_ratio = (asset_returns - risk_free_rate) / asset_standard_deviation
    else:
        raise TypeError(
            "Expects pd.DataFrame for both Asset Returns and Asset Standard Deviations or pd.Series / Float "
            "for both Asset Returns and Asset Standed Deviations."
        )

    return m2_ratio


def get_rolling_m2_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    window_size: int,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling M2 Ratio (Modigliani-Modigliani Measure) of returns.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        risk_free_rate (pd.Series): The risk free rate, aligned to the same period as the returns.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling M2 Ratio values with time as index.
    """
    excess_returns = asset_returns.sub(risk_free_rate, axis=0)

    rolling_mean = excess_returns.rolling(window=window_size).mean()
    rolling_std = asset_returns.rolling(window=window_size).std()

    return rolling_mean / rolling_std


def get_tracking_error(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate the Tracking Error of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free
        rate already subtracted.

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """
    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            tracking_error = asset_returns.groupby(level=0).apply(
                lambda x: (x.sub(benchmark_returns, axis=0)).std()
            )

            return tracking_error

        tracking_error = pd.DataFrame(
            index=asset_returns.index, columns=asset_returns.columns, dtype=np.float64
        )
        for column in tracking_error.columns:
            tracking_error.loc[:, column] = (asset_returns - benchmark_returns).std()

    if isinstance(asset_returns, (pd.Series | float)):
        tracking_error = (asset_returns - benchmark_returns).std()

    return tracking_error


def get_rolling_tracking_error(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window_size: int,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Tracking Error of returns.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Tracking Error values with time as index.
    """
    return (
        asset_returns.sub(benchmark_returns, axis=0).rolling(window=window_size).std()
    )


def get_information_ratio(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate the Information Ratio of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free
        rate already subtracted.

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """
    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            information_ratio = asset_returns.groupby(level=0).apply(
                lambda x: (
                    x.sub(benchmark_returns, axis=0).mean()
                    / (x.sub(benchmark_returns, axis=0)).std()
                )
            )

            return information_ratio

        information_ratio = pd.DataFrame(
            index=asset_returns.index, columns=asset_returns.columns, dtype=np.float64
        )
        for column in information_ratio.columns:
            difference = asset_returns[column] - benchmark_returns
            information_ratio.loc[:, column] = difference.mean() / difference.std()

    if isinstance(asset_returns, (pd.Series | float)):
        difference = asset_returns - benchmark_returns
        information_ratio = difference.mean() / difference.std()

    return information_ratio


def get_rolling_information_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window_size: int,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Information Ratio of returns.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Information Ratio values with time as index.
    """
    difference = asset_returns.sub(benchmark_returns, axis=0)

    rolling_mean = difference.rolling(window=window_size).mean()
    rolling_std = difference.rolling(window=window_size).std()

    return rolling_mean / rolling_std


def get_calmar_ratio(
    returns: pd.Series | pd.DataFrame, maximum_drawdown: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Calmar Ratio of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of period returns.
        maximum_drawdown (pd.Series | pd.DataFrame): The corresponding Maximum Drawdown values.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Calmar ratios with time as index
        and assets as columns.
    """
    return (returns / maximum_drawdown.abs()).dropna()


def get_average_drawdown(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Average Drawdown of returns, i.e. the mean of all pointwise negative
    drawdowns in the cumulative return series.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: Average Drawdown values as float if returns is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_average_drawdown(returns.loc[sub_period])
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        average_drawdown = pd.concat(period_data_list, axis=1)

        return average_drawdown.T

    cum_returns = (1 + returns.fillna(0)).cumprod()
    drawdowns = cum_returns / cum_returns.cummax() - 1

    return drawdowns[drawdowns < 0].mean()


def get_sterling_ratio(
    returns: pd.Series | pd.DataFrame,
    average_drawdown: pd.Series | pd.DataFrame,
    adjustment: float = 0.1,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Sterling Ratio of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of period returns.
        average_drawdown (pd.Series | pd.DataFrame): The corresponding Average Drawdown values.
        adjustment (float, optional): The fixed adjustment added to the average drawdown,
        conventionally 0.1 (10%). Defaults to 0.1.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Sterling ratios with time as index
        and assets as columns.
    """
    return (returns / (average_drawdown.abs() + adjustment)).dropna()


def get_burke_drawdown_measure(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Burke Drawdown Measure of returns, i.e. the square root of the sum of
    squared pointwise negative drawdowns in the cumulative return series.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: Burke Drawdown Measure values as float if returns is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_burke_drawdown_measure(returns.loc[sub_period])
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        burke_drawdown_measure = pd.concat(period_data_list, axis=1)

        return burke_drawdown_measure.T

    cum_returns = (1 + returns.fillna(0)).cumprod()
    drawdowns = cum_returns / cum_returns.cummax() - 1

    return np.sqrt((drawdowns[drawdowns < 0] ** 2).sum())


def get_burke_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    burke_drawdown_measure: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Burke Ratio of returns.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        risk-free rate subtracted.
        burke_drawdown_measure (pd.Series | pd.DataFrame): The corresponding Burke Drawdown
        Measure values.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Burke ratios with time as index
        and assets as columns.
    """
    return (excess_returns / burke_drawdown_measure).dropna()


def _get_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    upside: bool,
) -> pd.Series | pd.DataFrame:
    """
    Shared implementation for the Upside and Downside Capture Ratios.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        upside (bool): Whether to compute the Upside (True) or Downside (False) Capture Ratio.

    Returns:
        pd.Series | pd.DataFrame: Capture Ratio values.
    """
    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            combination = pd.concat([asset_returns, benchmark_returns], axis=1)

            def _capture(frame):
                mask = (
                    frame[benchmark_returns.name] > 0
                    if upside
                    else frame[benchmark_returns.name] < 0
                )
                return (
                    frame[asset_returns.columns][mask].mean()
                    / frame[benchmark_returns.name][mask].mean()
                )

            return combination.groupby(level=0).apply(_capture)

        mask = benchmark_returns > 0 if upside else benchmark_returns < 0

        return asset_returns[mask].mean() / benchmark_returns[mask].mean()

    if isinstance(asset_returns, pd.Series):
        mask = benchmark_returns > 0 if upside else benchmark_returns < 0

        return asset_returns[mask].mean() / benchmark_returns[mask].mean()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_upside_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Upside Capture Ratio, i.e. the ratio of the asset's average return to the
    benchmark's average return during periods in which the benchmark's return is positive.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.

    Returns:
        pd.Series | pd.DataFrame: Upside Capture Ratio values.
    """
    return _get_capture_ratio(asset_returns, benchmark_returns, upside=True)


def get_downside_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Downside Capture Ratio, i.e. the ratio of the asset's average return to the
    benchmark's average return during periods in which the benchmark's return is negative.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.

    Returns:
        pd.Series | pd.DataFrame: Downside Capture Ratio values.
    """
    return _get_capture_ratio(asset_returns, benchmark_returns, upside=False)


def get_win_rate(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Win Rate, i.e. the percentage of periods in which the asset's return
    exceeds the benchmark's return.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.

    Returns:
        pd.Series | pd.DataFrame: Win Rate values.
    """
    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            combination = pd.concat([asset_returns, benchmark_returns], axis=1)

            return combination.groupby(level=0).apply(
                lambda x: x[asset_returns.columns]
                .gt(x[benchmark_returns.name], axis=0)
                .mean()
            )

        return asset_returns.gt(benchmark_returns, axis=0).mean()

    if isinstance(asset_returns, pd.Series):
        return (asset_returns > benchmark_returns).mean()

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_kappa_ratio(
    excess_returns: pd.Series | pd.DataFrame, order: int = 3
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Kappa Ratio of returns. The Sortino Ratio is the special case of the Kappa
    Ratio with order=2.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        risk-free rate subtracted.
        order (int, optional): The order of the lower partial moment used in the denominator.
        Defaults to 3.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Kappa ratios with time as index
        and assets as columns.
    """
    if isinstance(excess_returns, pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            return excess_returns.groupby(level=0).apply(
                lambda x: x.mean() / (x[x < 0].abs() ** order).mean() ** (1 / order)
            )

        downside_returns = excess_returns[excess_returns < 0]
        lower_partial_moment = (downside_returns.abs() ** order).mean()

        return excess_returns.mean() / lower_partial_moment ** (1 / order)

    if isinstance(excess_returns, pd.Series):
        downside_returns = excess_returns[excess_returns < 0]
        lower_partial_moment = (downside_returns.abs() ** order).mean()

        return excess_returns.mean() / lower_partial_moment ** (1 / order)

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


def get_omega_ratio(
    returns: pd.Series | pd.DataFrame, minimum_acceptable_return: float = 0.0
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Omega Ratio of returns, i.e. the sum of gains above the minimum acceptable
    return (MAR) divided by the sum of losses below the MAR.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used
        as the threshold between gains and losses. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Omega Ratio values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_omega_ratio, minimum_acceptable_return=minimum_acceptable_return
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            omega_ratio = pd.concat(period_data_list, axis=1)

            return omega_ratio.T

        return returns.aggregate(
            get_omega_ratio, minimum_acceptable_return=minimum_acceptable_return
        )

    if isinstance(returns, pd.Series):
        excess_returns = returns - minimum_acceptable_return
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        return gains / losses

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_omega_ratio(
    returns: pd.Series | pd.DataFrame,
    window_size: int,
    minimum_acceptable_return: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Omega Ratio of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used
        as the threshold between gains and losses. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Rolling Omega Ratio values with time as index.
    """

    def _omega(window):
        excess_returns = window - minimum_acceptable_return
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        return gains / losses if losses else np.nan

    return returns.rolling(window=window_size).apply(_omega, raw=True)


def get_gain_to_pain_ratio(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Gain-to-Pain Ratio of returns, i.e. the sum of all returns divided by the
    sum of the absolute value of all losses.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        pd.Series | pd.DataFrame: Gain-to-Pain Ratio values as float if returns is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(get_gain_to_pain_ratio)
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            gain_to_pain_ratio = pd.concat(period_data_list, axis=1)

            return gain_to_pain_ratio.T

        return returns.aggregate(get_gain_to_pain_ratio)

    if isinstance(returns, pd.Series):
        losses = returns[returns < 0].abs().sum()

        return returns.sum() / losses

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_compound_growth_rate(
    prices: pd.Series | pd.DataFrame, periods: int
) -> float | pd.Series:
    """
    Calculate the Compound Growth Rate of a given data series.

    CGR is the mean growth rate of an investment over a specified period. Typically,
    the Compound Annual Growth Rate (CAGR) is used.

    Args:
        prices (pd.Series): Series of data points.

    Returns:
        float: CGR value.
    """
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / periods) - 1


def get_returns(
    returns: pd.Series | pd.DataFrame, period: str, cumulative: bool = False
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Return for a given period (weekly, monthly, quarterly or
    yearly) based on daily historical returns.

    The period Return is obtained by compounding the daily returns within each
    period, following the formula:

        - Period Return = ((1 + Return 1) * (1 + Return 2) * ... * (1 + Return N)) - 1

    If cumulative is set to True, the period returns are compounded further into
    a cumulative return, following the formula:

        - Cumulative Return = (1 + Period Return 1) * (1 + Period Return 2) * ... * (1 + Period Return N)

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the Return for. Can be weekly,
        monthly, quarterly or yearly.
        cumulative (bool, optional): Whether to return the cumulative return over
        time instead of the discrete return per period. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: Return values with time as the index, resampled
        to the given period. If cumulative is True, the cumulative return is
        returned instead.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    period_str = PERIOD_TRANSLATION[period]
    dates = returns.index.asfreq(period_str)

    period_returns = (1 + returns).groupby(dates).prod() - 1

    if cumulative:
        return (1 + period_returns).cumprod()

    return period_returns


def get_excess_return(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    cumulative: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Calculates the Excess Return, i.e. the return minus the risk free rate.

    If cumulative is set to True, the excess returns are compounded into a
    cumulative excess return, following the formula:

        - Cumulative Excess Return = (1 + Excess Return 1) * (1 + Excess Return 2) * ... * (1 + Excess Return N)

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        risk_free_rate (pd.Series): A Series of the risk free rate, aligned to the
        same period as the returns.
        cumulative (bool, optional): Whether to return the cumulative excess return
        over time instead of the discrete excess return per period. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: Excess Return values with time as the index. If
        cumulative is True, the cumulative excess return is returned instead.
    """
    excess_return = returns.sub(risk_free_rate, axis=0)

    if cumulative:
        return (1 + excess_return).cumprod()

    return excess_return
