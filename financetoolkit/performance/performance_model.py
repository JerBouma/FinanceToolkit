"""Performance Model"""

import io
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.risk import cvar_model
from financetoolkit.utilities.requests_model import get_request
from financetoolkit.utilities.statistics_model import PERIOD_TRANSLATION

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# The Euler-Mascheroni constant, used by the Deflated Sharpe Ratio.
EULER_MASCHERONI_CONSTANT = 0.5772156649015329

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

    For more information about the method, see the following papers:

    - Sharpe, W.F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under
    Conditions of Risk." Journal of Finance, 19(3), 425-442.
    - Lintner, J. (1965). "The Valuation of Risk Assets and the Selection of Risky
    Investments in Stock Portfolios and Capital Budgets." Review of Economics and
    Statistics, 47(1), 13-37.

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
    elif isinstance(beta, (pd.Series | float)):
        capital_asset_pricing_model = risk_free_rate + beta * (
            benchmark_returns - risk_free_rate
        )
    else:
        raise TypeError("Expects pd.DataFrame, pd.Series or float, no other value.")

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

    cache = get_active_cache()

    if cache is not None:
        cached_dataset = cache.get(
            source=policy_model.KEN_FRENCH,
            dataset="factors_decimal",
            entity=fama_and_french_url,
        )

        if cached_dataset is not None:
            return cached_dataset

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

        # Ken French publishes the factors in percent while returns are decimals.
        fama_and_french_dataset = fama_and_french_dataset / 100

        fama_and_french_dataset.index = pd.to_datetime(
            fama_and_french_dataset.index, format="%Y%m%d"
        ).to_period(freq="D")
        fama_and_french_dataset.index.name = "Date"

    if cache is not None and not fama_and_french_dataset.empty:
        cache.set(
            source=policy_model.KEN_FRENCH,
            dataset="factors_decimal",
            entity=fama_and_french_url,
            data=fama_and_french_dataset,
        )

    return fama_and_french_dataset


def obtain_fama_and_french_three_factor_dataset(
    fama_and_french_url: str | None = None,
) -> pd.DataFrame:
    """
    This functionality returns the Fama and French 3 Factor Model dataset:

        - Market (MKT): The excess return of the market.
        - Size (SMB): Small companies tend to outperform large companies.
        - Value (HML): Value stocks tend to outperform growth stocks.

    Next to that, it also includes the Risk Free Rate.

    This is a separate dataset from the 5 Factor Model (see
    `obtain_fama_and_french_dataset`) and is *not* interchangeable with it: `Mkt-RF`,
    `HML` and `RF` are identical across the two files, but `SMB` is not. In the 3 Factor
    file SMB is built from the 2x3 size/book-to-market sort alone, whereas in the 5
    Factor file it is the average of three separate SMB legs (size/book-to-market,
    size/operating-profitability and size/investment). The Carhart (1997) Four Factor
    Model extends the *three* factor model, so it must be estimated against this file.

    Args:
        fama_and_french_url (str): the URL of the ZIP file that contains the dataset. If no URL is
        provided, the default URL (Fama and French 3 Factor) is used.

    Returns:
        pd.DataFrame: the Fama and French 3 Factor Model dataset.
    """
    return obtain_fama_and_french_dataset(
        fama_and_french_url
        if fama_and_french_url
        else "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    )


def obtain_carhart_momentum_dataset(momentum_url: str | None = None) -> pd.DataFrame:
    """
    This functionality returns the Carhart Momentum (MOM) factor dataset, used to extend
    the Fama and French 3 Factor Model into the Carhart 4 Factor Model:

        - Momentum (MOM): Stocks with high prior returns (winners) tend to keep outperforming
        stocks with low prior returns (losers) over the medium term.

    The dataset is available on the website of Dartmouth College and is updated on a monthly
    basis. The dataset is packaged in a ZIP file, so it needs to be extracted first.

    Args:
        momentum_url (str): the URL of the ZIP file that contains the dataset. If no URL is
        provided, the default Ken French Momentum Factor URL is used.

    Returns:
        pd.DataFrame: the Carhart Momentum Factor dataset, with a single "Mom" column.
    """
    momentum_url = (
        momentum_url
        if momentum_url
        else "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    )

    cache = get_active_cache()

    if cache is not None:
        cached_dataset = cache.get(
            source=policy_model.KEN_FRENCH,
            dataset="factors_decimal",
            entity=momentum_url,
        )

        if cached_dataset is not None:
            return cached_dataset

    response = get_request(momentum_url, timeout=10)
    zip_data = response.content

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
        zip_file_contents = zip_file.namelist()

        csv_file_name = zip_file_contents[0]

        with zip_file.open(csv_file_name) as csv_file:
            # Skip the descriptive header rows; the "Mom" data starts right after
            momentum_dataset = pd.read_csv(csv_file, skiprows=13, index_col=0)

        # A trailing comma makes an all-NaN column that would drop every valid row.
        momentum_dataset = momentum_dataset.loc[
            :, ~momentum_dataset.columns.str.contains("Unnamed")
        ]
        momentum_dataset = momentum_dataset.dropna(axis="index")

        # Ken French publishes the factor in percent while returns are decimals.
        momentum_dataset = momentum_dataset / 100

        momentum_dataset.index = pd.to_datetime(
            momentum_dataset.index, format="%Y%m%d"
        ).to_period(freq="D")
        momentum_dataset.index.name = "Date"
        momentum_dataset.columns = momentum_dataset.columns.str.strip()

    if cache is not None and not momentum_dataset.empty:
        cache.set(
            source=policy_model.KEN_FRENCH,
            dataset="factors_decimal",
            entity=momentum_url,
            data=momentum_dataset,
        )

    return momentum_dataset


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

    Notes:
    - Observations for which either the excess return or any factor is NaN are dropped
    (listwise deletion), which is the standard treatment for a missing observation in an
    OLS regression. Every return series starts with at least one NaN (the first
    observation has no prior close to compare against), and market holidays that differ
    between the return and factor calendars add more. Back- or forward-filling them
    instead would duplicate a neighbouring day's return into the regression, and for a
    leading NaN would use a future observation to explain a past one.
    """
    error_message = None

    factor_dataset = factor_dataset.reindex(excess_returns.index)
    valid = excess_returns.notna() & factor_dataset.notna().all(axis=1)
    excess_returns = excess_returns[valid]
    factor_dataset = factor_dataset[valid]

    if factor_dataset.empty:
        error_message = (
            "No overlapping non-missing observations between the returns and the "
            "factors. Setting values to NaN"
        )
        regression_results = {"Intercept": np.nan}

        for factor in factor_dataset.columns:
            regression_results[f"{factor} Slope"] = np.nan

        regression_results["Mean Squared Error (MSE)"] = np.nan
        regression_results["R Squared"] = np.nan

        return regression_results, excess_returns * np.nan, error_message

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

    Note that `scipy.stats.linregress(x, y)` fits `y = intercept + slope * x`, i.e. `x`
    is the independent/predictor variable and `y` is the dependent/response variable.
    Given the formula above, `factor` is `x` and `excess_returns` is `y`, so it must be
    called as `linregress(factor, excess_returns)` -- calling it the other way around
    (as an earlier version of this function did) fits the reverse regression (`factor`
    as a function of `excess_returns`), which generally yields a different slope and
    intercept (since OLS is not symmetric in `x` and `y`) and silently produces the
    wrong regression parameters and residuals for this model.

    Args:
        excess_returns (pd.Series): the excess returns.
        factor (pd.Series): the factor series.

    Returns:
        dict: the regression results.
        pd.Series: the residuals.

    Notes:
    - Observations for which either the excess return or the factor is NaN are dropped
    (listwise deletion), mirroring `get_fama_and_french_model_multi`. `scipy.stats.linregress`
    propagates NaN, so a single missing observation -- which the first period of any dataset
    always has, the first return having no prior close -- would otherwise turn every
    regression parameter and every residual for that period into NaN.
    """
    factor = factor.reindex(excess_returns.index)
    valid = excess_returns.notna() & factor.notna()
    excess_returns = excess_returns[valid]
    factor = factor[valid]

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

    result = linregress(factor, excess_returns)

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
    Calculate the (arithmetic) Alpha, i.e. the asset's return in excess of the
    benchmark's return.

    This is the simple, model-free notion of Alpha (sometimes called "active return"):
    it does not risk-adjust for market exposure at all. For the risk-adjusted version
    that nets out the return attributable to Beta, see `get_jensens_alpha`.

    The formula is as follows:

        - Alpha = Asset Return - Benchmark Return

    Args:
        asset_returns (pd.Series | float): The asset's or portfolio's return.
        benchmark_returns (pd.Series | float): The benchmark's return over the same period.

    Returns:
        pd.Series | pd.DataFrame: Alpha values with time as index and assets as columns.
    """
    if isinstance(asset_returns, pd.DataFrame):
        alpha = pd.DataFrame(
            index=asset_returns.index, columns=asset_returns.columns, dtype=np.float64
        )
        for column in alpha.columns:
            alpha.loc[:, column] = asset_returns[column] - benchmark_returns
    elif isinstance(asset_returns, (pd.Series | float)):
        alpha = asset_returns - benchmark_returns
    else:
        raise TypeError("Expects pd.DataFrame, pd.Series or float, no other value.")

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
    Calculate Jensen's Alpha, i.e. the return earned above what CAPM would predict
    given the asset's Beta (systematic risk exposure).

    The formula is as follows:

        - Jensen's Alpha = Asset Return - [Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate)]

    Where the bracketed term is the CAPM-predicted return (see
    `get_capital_asset_pricing_model`). A positive Jensen's Alpha indicates the asset
    outperformed what its market risk alone would justify (potential stock-picking or
    timing skill); a negative value indicates underperformance relative to that
    risk-adjusted benchmark.

    For more information about the method, see the following paper:

    - Jensen, M.C. (1968). "The Performance of Mutual Funds in the Period 1945-1964."
    Journal of Finance, 23(2), 389-416.

    Args:
        asset_returns (pd.Series | float): The asset's or portfolio's return.
        risk_free_rate (pd.Series | float): The risk-free rate over the same period.
        beta (pd.Series | pd.DataFrame | float): The asset's or portfolio's Beta, see
        `get_beta`.
        benchmark_returns (pd.Series | float): The benchmark's return over the same period.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Jensen's Alpha values with
        time as index and assets as columns.
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
    Calculate the Treynor ratio of returns, i.e. the excess return earned per unit of
    systematic (market) risk, as measured by Beta.

    Unlike the Sharpe ratio, which divides by total volatility (standard deviation),
    the Treynor ratio divides by Beta -- appropriate for an investor who holds this
    asset as one position within an already-diversified portfolio, and therefore only
    cares about compensation for market (non-diversifiable) risk, not idiosyncratic risk.

    The formula is as follows:

        - Treynor Ratio = (Asset Return - Risk-Free Rate) / Beta

    For more information about the method, see the following paper:

    - Treynor, J.L. (1965). "How to Rate Management of Investment Funds." Harvard
    Business Review, 43(1), 63-75.

    Args:
        asset_returns (pd.Series | float): The asset's or portfolio's return.
        risk_free_rate (pd.Series | float): The risk-free rate over the same period.
        beta (pd.Series | pd.DataFrame | float): The asset's or portfolio's Beta
        (systematic risk), see `get_beta`.

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
    Calculate the Sharpe ratio of returns, i.e. the mean excess return per unit of total
    (upside and downside) volatility.

    The formula is as follows:

        - Sharpe Ratio = Mean(Excess Return) / Standard Deviation(Excess Return)

    Where "Excess Return" is the return with the risk-free rate already subtracted (see
    `get_excess_return`). When `excess_returns` has a "within period" (period, date) Multi
    Index, the mean and standard deviation are computed from the daily observations within
    each period, producing one Sharpe ratio per period; otherwise both are computed once
    over the entire series (or once per column, for a plain DataFrame), producing a single
    ratio (per column).

    For more information about the method, see the following papers:

    - Sharpe, W.F. (1966). "Mutual Fund Performance." Journal of Business, 39(1), 119-138.
    - Sharpe, W.F. (1994). "The Sharpe Ratio." Journal of Portfolio Management, 21(1), 49-58.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        the risk-free rate subtracted.

    Returns:
        pd.Series | pd.DataFrame: Sharpe Ratio values. If `excess_returns` has a "within
        period" Multi Index, one value is returned per period (with time as index);
        otherwise a single value is returned per column (or overall, for a Series).
    """
    if isinstance(excess_returns, pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # Calculate Sharpe ratio for each asset (ticker) in the DataFrame
            sharpe_ratios = excess_returns.groupby(level=0).apply(
                lambda x: x.mean() / x.std()
            )
            return sharpe_ratios

        return excess_returns.mean() / excess_returns.std()

    if isinstance(excess_returns, pd.Series):
        # Calculate Sharpe ratio for a single asset (ticker)
        return excess_returns.mean() / excess_returns.std()

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


def get_probabilistic_sharpe_ratio(
    sharpe_ratio: pd.Series | pd.DataFrame | float,
    benchmark_sharpe_ratio: pd.Series | pd.DataFrame | float,
    skewness: pd.Series | pd.DataFrame | float,
    kurtosis: pd.Series | pd.DataFrame | float,
    n_observations: pd.Series | pd.DataFrame | float | int,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    The naive Sharpe ratio significance test (e.g. a t-test on SR̂) implicitly assumes
    that returns are normally (i.i.d. Gaussian) distributed. Real financial returns are
    typically skewed and fat-tailed, which means the standard error of the Sharpe ratio
    is understated and the naive test overstates confidence. The PSR corrects for this by
    explicitly folding the skewness and kurtosis of the return distribution into the
    standard error of the Sharpe ratio estimator.

    The formula is as follows:

        - PSR(SR*) = Φ( (SR̂ − SR*) · sqrt(n − 1) / sqrt(1 − γ₃·SR̂ + ((γ₄ − 1) / 4)·SR̂²) )

    Where SR̂ is the observed (realized) Sharpe ratio, SR* is the benchmark/hypothesized
    Sharpe ratio being tested against (often 0, i.e. "is there any skill at all"), γ₃ is
    the skewness of the returns, γ₄ is the kurtosis of the returns, n is the number of
    return observations used to estimate SR̂, and Φ is the standard normal CDF.

    PSR(SR*) is the probability that the true, population Sharpe ratio exceeds SR*, given
    the observed sample. It is bounded between 0 and 1: a PSR close to 1 means it is very
    likely the strategy has genuine skill above SR*, a PSR near 0.5 means the evidence is
    inconclusive, and a low PSR means the observed Sharpe ratio could plausibly be noise.

    Note that this formula requires **non-excess (raw) kurtosis**, i.e. a Normal
    distribution has a kurtosis of 3, not 0. This codebase's `risk_model.get_kurtosis`
    defaults to `fisher=True` (excess kurtosis, Normal = 0); either call it with
    `fisher=False` before passing the result in here, or add 3 to a Fisher/excess
    kurtosis value. Getting this convention backwards silently shifts every PSR value
    (via the ((γ₄ − 1) / 4) term), which is exactly the kind of sign/convention bug this
    codebase has been bitten by before, so double check which convention is being fed in.

    For more information about the method, see the following paper:

    - Bailey, D.H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
    Journal of Risk, 15(2), 3-44.

    Args:
        sharpe_ratio (pd.Series | pd.DataFrame | float): The observed (realized) Sharpe
        ratio (SR̂), computed over the same return frequency as `n_observations`.
        benchmark_sharpe_ratio (pd.Series | pd.DataFrame | float): The hypothesized or
        benchmark Sharpe ratio (SR*) to test the observed Sharpe ratio against. Often 0.
        skewness (pd.Series | pd.DataFrame | float): The skewness (γ₃) of the same
        returns used to compute `sharpe_ratio`.
        kurtosis (pd.Series | pd.DataFrame | float): The non-excess (raw) kurtosis (γ₄)
        of the same returns used to compute `sharpe_ratio`. See the Notes above — a
        Normal distribution has a kurtosis of 3 under this convention, not 0.
        n_observations (pd.Series | pd.DataFrame | float | int): The number of return
        observations used to compute `sharpe_ratio`.

    Returns:
        pd.Series | pd.DataFrame | float: The probability, between 0 and 1, that the true
        Sharpe ratio exceeds `benchmark_sharpe_ratio`.
    """
    numerator = (sharpe_ratio - benchmark_sharpe_ratio) * np.sqrt(n_observations - 1)
    denominator = np.sqrt(
        1 - skewness * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2
    )

    z_score = numerator / denominator
    probabilistic_sharpe_ratio = stats.norm.cdf(z_score)

    if isinstance(z_score, pd.DataFrame):
        return pd.DataFrame(
            probabilistic_sharpe_ratio, index=z_score.index, columns=z_score.columns
        )
    if isinstance(z_score, pd.Series):
        return pd.Series(probabilistic_sharpe_ratio, index=z_score.index)

    return probabilistic_sharpe_ratio


def get_deflated_sharpe_ratio(
    sharpe_ratio: pd.Series | pd.DataFrame | float,
    sharpe_ratio_variance: pd.Series | pd.DataFrame | float,
    n_trials: pd.Series | pd.DataFrame | float | int,
    n_observations: pd.Series | pd.DataFrame | float | int,
    skewness: pd.Series | pd.DataFrame | float,
    kurtosis: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).

    The Probabilistic Sharpe Ratio corrects the naive Sharpe ratio test for skewed,
    fat-tailed returns, but it still assumes the reported Sharpe ratio is the *only*
    one that was ever computed. In practice a reported Sharpe ratio is often the best
    of many strategy variations, lookback windows, or parameter combinations tried
    during a backtest (the "backtest overfitting" / multiple-testing / selection-bias
    problem) — the more trials attempted, the more likely it is that at least one of
    them shows an impressive Sharpe ratio purely by chance, even with zero true skill.

    The DSR corrects for this by first computing the Sharpe ratio one would *expect* to
    observe, purely by chance, as the maximum of `n_trials` independent trials under the
    null hypothesis of no skill, and then uses that expected maximum as the benchmark
    (SR*) plugged into the Probabilistic Sharpe Ratio formula (see
    `get_probabilistic_sharpe_ratio`), instead of a naive benchmark such as 0.

    The formula for the expected maximum Sharpe ratio benchmark is as follows:

        - SR* = sqrt(Var[SR_trials]) · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    Where `N` is `n_trials`, `Var[SR_trials]` is the variance of the Sharpe ratios
    observed across those N trials (`sharpe_ratio_variance`), γ ≈ 0.5772 is the
    Euler-Mascheroni constant, and Φ⁻¹ is the inverse standard normal CDF. This
    asymptotic approximation is only meaningful for `N >= 2` trials (with N = 1 there is
    no order-statistic/selection effect at all, so the benchmark falls back to 0, which
    is also the true limiting value of the expected maximum of a single draw).

    DSR = PSR(SR*), i.e. the Deflated Sharpe Ratio is the Probabilistic Sharpe Ratio
    computed with this trial-adjusted SR* as the benchmark rather than a fixed value
    such as 0. Because SR* only grows (or stays flat) as `n_trials` increases, the DSR is
    always less than or equal to the PSR computed with the same underlying Sharpe ratio,
    skewness, kurtosis and number of observations against a benchmark of 0.

    For more information about the method, see the following paper:

    - Bailey, D.H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting, and Non-Normality." Journal of Portfolio
    Management, 40(5), 94-107.

    Args:
        sharpe_ratio (pd.Series | pd.DataFrame | float): The observed (realized) Sharpe
        ratio (SR̂) being tested, i.e. the one actually reported after the trials.
        sharpe_ratio_variance (pd.Series | pd.DataFrame | float): The variance of the
        Sharpe ratios observed across the `n_trials` trials. When the literal N distinct
        strategy trials are not available, a reasonable proxy is the variance of a
        rolling Sharpe ratio series computed over the same returns (see
        `get_rolling_sharpe_ratio`), used as an approximation of how dispersed the
        Sharpe ratio could plausibly have been under different choices.
        n_trials (pd.Series | pd.DataFrame | float | int): The number of independent (or
        effectively independent) strategy variations, parameter combinations, or
        lookback windows tried before arriving at `sharpe_ratio`. Must be >= 1.
        n_observations (pd.Series | pd.DataFrame | float | int): The number of return
        observations used to compute `sharpe_ratio`.
        skewness (pd.Series | pd.DataFrame | float): The skewness (γ₃) of the same
        returns used to compute `sharpe_ratio`.
        kurtosis (pd.Series | pd.DataFrame | float): The non-excess (raw) kurtosis (γ₄)
        of the same returns used to compute `sharpe_ratio`. See the Notes in
        `get_probabilistic_sharpe_ratio` — a Normal distribution has a kurtosis of 3
        under this convention, not 0.

    Returns:
        pd.Series | pd.DataFrame | float: The probability, between 0 and 1, that the
        true Sharpe ratio exceeds the trial-adjusted expected maximum Sharpe ratio one
        would observe purely by chance given `n_trials` attempts.
    """
    n_trials_array = np.asarray(n_trials, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_max_z_score = (1 - EULER_MASCHERONI_CONSTANT) * stats.norm.ppf(
            1 - 1 / n_trials_array
        ) + EULER_MASCHERONI_CONSTANT * stats.norm.ppf(1 - 1 / (n_trials_array * np.e))

    # The expected maximum of a single draw is 0, so N = 1 gets no deflation.
    expected_max_z_score = np.where(n_trials_array <= 1, 0.0, expected_max_z_score)

    benchmark_sharpe_ratio = np.sqrt(sharpe_ratio_variance) * expected_max_z_score

    return get_probabilistic_sharpe_ratio(
        sharpe_ratio=sharpe_ratio,
        benchmark_sharpe_ratio=benchmark_sharpe_ratio,
        skewness=skewness,
        kurtosis=kurtosis,
        n_observations=n_observations,
    )


def get_sortino_ratio(excess_returns: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Calculate the Sortino ratio of returns, i.e. the mean excess return per unit of
    downside deviation.

    Unlike the Sharpe ratio, which penalizes upside and downside volatility equally via
    the standard deviation, the Sortino ratio only penalizes volatility below a Minimum
    Acceptable Return (here, a MAR of 0, since `excess_returns` already has the risk-free
    rate subtracted). This is achieved by replacing the standard deviation in the
    denominator with the "downside deviation" (the target semi-deviation).

    The formula is as follows:

        - Sortino Ratio = Mean(Excess Return) / Downside Deviation
        - Downside Deviation = sqrt( (1/N) * sum( min(Excess Return, 0)^2 ) )

    Where N is the *total* number of observations (not just the number of negative
    ones) -- periods with a non-negative excess return contribute 0 to the sum, but
    still count towards N, exactly as in Sortino & Price (1994) and the equivalent
    empyrical/pyfolio `downside_risk` implementation. This is deliberately not the same
    as `excess_returns[excess_returns < 0].std()`: that alternative would divide by
    (number of negative observations - 1) and measure the dispersion of the losses
    around *their own mean*, rather than their root-mean-square distance below the MAR
    of 0 -- a materially different (and non-standard) quantity that understates how
    volatile the downside really is whenever there are few, large losses.

    For more information about the method, see the following paper:

    - Sortino, F.A., & Price, L.N. (1994). "Performance Measurement in a Downside Risk
    Framework." Journal of Investing, 3(3), 59-64.

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
                lambda x: x.mean() / np.sqrt((x.clip(upper=0) ** 2).mean())
            )
            return sortino_ratios

        downside_deviation = np.sqrt((excess_returns.clip(upper=0) ** 2).mean())

        return excess_returns.mean() / downside_deviation

    if isinstance(excess_returns, pd.Series):
        # Calculate Sortino ratio for a single asset (ticker)
        downside_deviation = np.sqrt((excess_returns.clip(upper=0) ** 2).mean())
        return excess_returns.mean() / downside_deviation

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


def get_rolling_sortino_ratio(
    excess_returns: pd.Series | pd.DataFrame, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Sortino ratio of returns. See `get_sortino_ratio` for the
    formula and the Notes on why the downside deviation is computed as the root-mean-
    square of the negative part of returns over the *entire* window (Sortino & Price,
    1994), rather than as the standard deviation of only the negative observations
    within that window.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free rate subtracted.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling Sortino ratio values with time as index.
    """

    def _downside_deviation(window):
        downside = np.minimum(window, 0.0)

        return np.sqrt(np.mean(downside**2))

    rolling_mean = excess_returns.rolling(window=window_size).mean()
    rolling_downside_deviation = excess_returns.rolling(window=window_size).apply(
        _downside_deviation, raw=True
    )

    return rolling_mean / rolling_downside_deviation


def get_ulcer_performance_index(
    excess_returns: pd.Series | pd.DataFrame, ulcer_index: pd.Series | pd.DataFrame
) -> pd.Series:
    """
    Calculate the Ulcer Performance Index (UPI) of returns, i.e. the mean excess return
    per unit of Ulcer Index (a depth- and duration-weighted measure of drawdown pain).

    The formula is as follows:

        - Ulcer Performance Index = Excess Return / Ulcer Index

    Also known as: Martin Ratio.

    For more information about the method, see the following source:

    - Martin, P.G., & McCann, B.B. (1989). "The Investor's Guide to Fidelity Funds:
    Winning Strategies for Mutual Fund Investors." John Wiley & Sons. (Peter Martin's
    Ulcer Index, on which this ratio is based, was originally introduced in 1987.)

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series of returns with risk-free rate subtracted.
        ulcer_index (pd.Series | pd.DataFrame): The corresponding Ulcer Index values (see
        `financetoolkit.risk.risk_model.get_ulcer_index`).

    Returns:
        pd.Series: A Series of Ulcer Performance Index values with time as index and assets as columns.
    """
    return (excess_returns / ulcer_index).dropna()


def get_m2_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    asset_standard_deviation: pd.Series | pd.DataFrame,
    benchmark_standard_deviation: pd.Series | float,
) -> pd.Series:
    """
    Calculate the M2 Ratio (Modigliani-Modigliani Measure) of returns.

    The Sharpe ratio is dimensionless (a ratio, not a return), which makes it awkward to
    communicate intuitively -- a Sharpe ratio of 0.5 does not by itself say how much
    return an investor gave up or gained. The M2 measure rescales the Sharpe ratio back
    into return units by asking: what return would this portfolio have earned if it had
    been leveraged (or de-leveraged) with risk-free borrowing/lending to match the
    benchmark's volatility exactly? This yields a return-space number that can be
    compared directly against the benchmark's actual return.

    The formula is as follows:

        - M2 Ratio = Risk-Free Rate + Sharpe Ratio * Benchmark Standard Deviation
        - Sharpe Ratio = (Asset Return - Risk-Free Rate) / Asset Standard Deviation

    Note that this requires the *benchmark's* standard deviation, not just the asset's
    own -- without it, this collapses to the plain Sharpe ratio (dimensionless) rather
    than M2 (a return, expressed in the same units as `asset_returns`). A previous
    version of this function omitted `benchmark_standard_deviation` entirely and
    returned the Sharpe ratio under the "M2 Ratio" name.

    For more information about the method, see the following paper:

    - Modigliani, F., & Modigliani, L. (1997). "Risk-Adjusted Performance." Journal of
    Portfolio Management, 23(2), 45-54.

    Args:
        asset_returns (pd.Series | pd.DataFrame): A Series or DataFrame of period returns.
        risk_free_rate (pd.Series): The risk-free rate, aligned to the same period as the returns.
        asset_standard_deviation (pd.Series | pd.DataFrame): The asset's (or portfolio's)
        standard deviation, over the same window as `asset_returns`.
        benchmark_standard_deviation (pd.Series | float): The benchmark's standard
        deviation, over the same window as `asset_returns`.

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
            sharpe_ratio = (
                asset_returns[column] - risk_free_rate
            ) / asset_standard_deviation[column]
            m2_ratio.loc[:, column] = (
                risk_free_rate + sharpe_ratio * benchmark_standard_deviation
            )
    elif isinstance(asset_returns, (pd.Series | float)) and isinstance(
        asset_standard_deviation, (pd.Series | float)
    ):
        sharpe_ratio = (asset_returns - risk_free_rate) / asset_standard_deviation
        m2_ratio = risk_free_rate + sharpe_ratio * benchmark_standard_deviation
    else:
        raise TypeError(
            "Expects pd.DataFrame for both Asset Returns and Asset Standard Deviations or pd.Series / Float "
            "for both Asset Returns and Asset Standed Deviations."
        )

    return m2_ratio


def get_rolling_m2_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    benchmark_returns: pd.Series,
    window_size: int,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling M2 Ratio (Modigliani-Modigliani Measure) of returns. See
    `get_m2_ratio` for the formula.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        risk_free_rate (pd.Series): The risk free rate, aligned to the same period as the returns.
        benchmark_returns (pd.Series): Benchmark returns, aligned to the same period as
        the returns, from which the rolling benchmark standard deviation is derived.
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling M2 Ratio values with time as index.
    """
    excess_returns = asset_returns.sub(risk_free_rate, axis=0)

    rolling_mean = excess_returns.rolling(window=window_size).mean()
    rolling_std = asset_returns.rolling(window=window_size).std()
    rolling_sharpe_ratio = rolling_mean / rolling_std

    rolling_risk_free_rate = risk_free_rate.rolling(window=window_size).mean()
    rolling_benchmark_std = benchmark_returns.rolling(window=window_size).std()

    return rolling_sharpe_ratio.mul(rolling_benchmark_std, axis=0).add(
        rolling_risk_free_rate, axis=0
    )


def get_tracking_error(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series:
    """
    Calculate the Tracking Error of returns, i.e. the standard deviation of the
    difference between the asset's and the benchmark's return (the "active return").

    The formula is as follows:

        - Tracking Error = Standard Deviation(Asset Return - Benchmark Return)

    Args:
        asset_returns (pd.Series | pd.DataFrame): A Series or DataFrame of asset returns.
        benchmark_returns (pd.Series): The benchmark's returns over the same period.

    Returns:
        pd.Series: A Series of Tracking Error values with time as index and assets as columns.
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
            tracking_error.loc[:, column] = (
                asset_returns[column] - benchmark_returns
            ).std()

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
    Calculate the Information Ratio of returns, i.e. the mean active return (asset
    return minus benchmark return) per unit of Tracking Error.

    The formula is as follows:

        - Information Ratio = Mean(Asset Return - Benchmark Return) / Tracking Error

    Where Tracking Error is the standard deviation of that same difference (see
    `get_tracking_error`). The Information Ratio is the Sharpe ratio of "active"
    (benchmark-relative) returns, and is closely tied to Grinold's "Fundamental Law of
    Active Management" as a way of scoring active managers on a risk-adjusted basis.

    For more information about the method, see the following paper:

    - Grinold, R.C. (1989). "The Fundamental Law of Active Management." Journal of
    Portfolio Management, 15(3), 30-37.

    Args:
        asset_returns (pd.Series | pd.DataFrame): A Series or DataFrame of asset returns.
        benchmark_returns (pd.Series): The benchmark's returns over the same period.

    Returns:
        pd.Series: A Series of Information Ratio values with time as index and assets as columns.
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
    Calculate the Calmar Ratio of returns, i.e. the (typically annualized) return per
    unit of Maximum Drawdown.

    The formula is as follows:

        - Calmar Ratio = Return / |Maximum Drawdown|

    Named after "California Managed Accounts Reports," the newsletter in which Terry W.
    Young originally proposed it.

    For more information about the method, see the following source:

    - Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." Futures, 20(1), 40.

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
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Average Drawdown of returns, i.e. the mean of all pointwise negative
    drawdowns in the cumulative return series.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        method (str, optional): Either "return" (default), which measures percentage
        drawdowns of the compounded return series, or "level", which measures
        absolute drawdowns of the raw level series directly -- use this when
        `returns` is not a genuine percentage return (e.g. a series that can be zero
        or negative). Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame: Average Drawdown values as float if returns is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_average_drawdown(returns.loc[sub_period], method=method)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        average_drawdown = pd.concat(period_data_list, axis=1)

        return average_drawdown.T

    if method == "level":
        drawdowns = returns - returns.cummax()
    else:
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
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Burke Drawdown Measure of returns, i.e. the square root of the sum of
    squared pointwise negative drawdowns in the cumulative return series.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns
        (method="return") or of raw levels (method="level").
        method (str, optional): Either "return" (default), which measures percentage
        drawdowns of the compounded return series, or "level", which measures
        absolute drawdowns of the raw level series directly -- use this when
        `returns` is not a genuine percentage return (e.g. a series that can be zero
        or negative). Note that in "level" mode the result is in squared input units,
        not a percentage. Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame: Burke Drawdown Measure values as float if returns is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    if method not in ("return", "level"):
        raise ValueError("method must be 'return' or 'level'.")

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_burke_drawdown_measure(
                returns.loc[sub_period], method=method
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        burke_drawdown_measure = pd.concat(period_data_list, axis=1)

        return burke_drawdown_measure.T

    if method == "level":
        drawdowns = returns - returns.cummax()
    else:
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
    Calculate the Kappa Ratio of returns, i.e. the mean excess return per unit of the
    n-th order Lower Partial Moment (LPM). The Sortino Ratio (see `get_sortino_ratio`)
    is the special case of the Kappa Ratio with order=2.

    The formula is as follows:

        - Kappa Ratio = Mean(Excess Return) / LPM_order^(1/order)
        - LPM_order = (1/N) * sum( max(-Excess Return, 0)^order )

    Where N is the *total* number of observations (not just the number of negative
    ones) -- periods with a non-negative excess return contribute 0 to the sum, but
    still count towards N. This mirrors the same convention used in `get_sortino_ratio`
    (order=2): dividing only by the count of negative observations instead of the total
    N, or centering the moment on the downside subset's own mean instead of on the MAR
    of 0, would compute a different (non-standard) quantity than the Lower Partial
    Moment this ratio is defined against.

    For more information about the method, see the following paper:

    - Kaplan, P.D., & Knowles, J.A. (2004). "Kappa: A Generalized Downside Risk-Adjusted
    Performance Measure." Journal of Performance Measurement, 8(3), 42-54.

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
                lambda x: x.mean()
                / (x.clip(upper=0).abs() ** order).mean() ** (1 / order)
            )

        lower_partial_moment = (excess_returns.clip(upper=0).abs() ** order).mean()

        return excess_returns.mean() / lower_partial_moment ** (1 / order)

    if isinstance(excess_returns, pd.Series):
        lower_partial_moment = (excess_returns.clip(upper=0).abs() ** order).mean()

        return excess_returns.mean() / lower_partial_moment ** (1 / order)

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


def get_omega_ratio(
    returns: pd.Series | pd.DataFrame, minimum_acceptable_return: float = 0.0
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Omega Ratio of returns, i.e. the sum of gains above the minimum acceptable
    return (MAR) divided by the sum of losses below the MAR.

    Unlike the Sharpe/Sortino ratios, which only use the first two (Sharpe) or a partial
    (Sortino) moment of the return distribution, the Omega Ratio uses the entire
    distribution: it is equivalent to the ratio of the areas above and below the MAR
    under the return distribution's cumulative distribution function, so it implicitly
    captures all higher moments (skewness, kurtosis, etc.) without needing to estimate
    them explicitly.

    The formula is as follows:

        - Omega Ratio = sum(max(Return - MAR, 0)) / sum(max(MAR - Return, 0))

    For more information about the method, see the following paper:

    - Keating, C., & Shadwick, W.F. (2002). "A Universal Performance Measure." Journal
    of Performance Measurement, 6(3), 59-84.

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

    The formula is as follows:

        - Gain-to-Pain Ratio = sum(Return) / sum(|Return| where Return < 0)

    A value of 1 means gains exactly offset losses; a value below (above) 1 means losses
    outweigh (are outweighed by) gains in aggregate.

    For more information about the method, see the following source:

    - Schwager, J.D. (2012). "Hedge Fund Market Wizards: How Winning Traders Win." John
    Wiley & Sons. (Popularized this simple ratio as a robustness-focused alternative to
    the Sharpe ratio for evaluating trading strategies.)

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

    The formula is as follows:

        - CGR = (Final Value / Initial Value) ^ (1 / Number of Periods) - 1

    Note that `periods` is the number of compounding *intervals* between the first and
    the last observation, which for a series of N observations is N - 1, not N. Passing
    N instead of N - 1 systematically understates the growth rate (e.g. ten year-end
    prices span nine years of compounding, not ten).

    Args:
        prices (pd.Series | pd.DataFrame): Series or DataFrame of data points, ordered
        from oldest to newest.
        periods (int): The number of compounding intervals between the first and the
        last observation. Must be greater than zero.

    Returns:
        float | pd.Series: CGR value, as a float if `prices` is a Series and as a
        Series (one value per column) if `prices` is a DataFrame.

    Raises:
        ValueError: If `periods` is not greater than zero.
    """
    if periods <= 0:
        raise ValueError("The number of periods must be greater than zero.")

    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / periods) - 1


def get_returns(
    returns: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
    cumulative: bool = False,
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
        monthly, quarterly or yearly. Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `returns`, to group by instead of deriving calendar periods from
        `returns.index` via `.asfreq()`. Use this when `returns` does not have a
        DatetimeIndex/PeriodIndex (e.g. a plain Series of simulated outcomes).
        Defaults to None, which requires a DatetimeIndex/PeriodIndex on `returns`.
        cumulative (bool, optional): Whether to return the cumulative return over
        time instead of the discrete return per period. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame: Return values with time (or `groups`) as the
        index. If cumulative is True, the cumulative return is returned instead.
    """
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    dates = (
        groups
        if groups is not None
        else returns.index.asfreq(PERIOD_TRANSLATION[period])
    )

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


def get_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the full pairwise correlation matrix across N assets.

    Unlike `get_beta` or `get_covariance`, which relate a single asset to a single
    benchmark, this computes the correlation between every pair of columns in the
    provided returns DataFrame at once. This is a prerequisite for portfolio variance
    calculations and any mean-variance optimization work.

    Args:
        returns (pd.DataFrame): A Dataframe of returns with one column per asset.

    Returns:
        pd.DataFrame: The N x N correlation matrix, with assets as both the index
        and the columns.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Expects pd.DataFrame, no other value.")

    return returns.corr()


def get_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the full pairwise covariance matrix across N assets.

    Unlike `get_beta` or `get_covariance`, which relate a single asset to a single
    benchmark, this computes the covariance between every pair of columns in the
    provided returns DataFrame at once. This is a prerequisite for portfolio variance
    calculations and any mean-variance optimization work.

    Args:
        returns (pd.DataFrame): A Dataframe of returns with one column per asset.

    Returns:
        pd.DataFrame: The N x N covariance matrix, with assets as both the index
        and the columns.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Expects pd.DataFrame, no other value.")

    return returns.cov()


def get_capm_residuals(
    excess_returns: pd.DataFrame,
    beta: pd.DataFrame,
    benchmark_excess_returns: pd.Series,
) -> pd.DataFrame:
    """
    Calculate the pointwise (e.g. daily) CAPM regression residuals within each period of
    a "within period" (period, date) Multi Index dataset, given a Beta estimated once per
    period.

    The formula is as follows:

        - CAPM Residual = Excess Return − Beta * Benchmark Excess Return

    This is exactly the same formula as `get_jensens_alpha` evaluated on already-excess
    returns (i.e. with the risk-free rate set to 0). `get_jensens_alpha` itself cannot be
    called directly on a full "within period" DataFrame together with a period-indexed
    Beta, because it shapes its output after `beta.index` (one row per period) rather than
    after the (period, date) index of the pointwise returns. This function instead calls
    `get_jensens_alpha` once per (period, ticker) pair — where both the returns and the
    Beta reduce to plain, aligned Series/scalars — so that the period-level Beta is
    correctly broadcast across every day within that period, and reuses the exact same
    CAPM regression formula rather than reimplementing it.

    Args:
        excess_returns (pd.DataFrame): A "within period" (period, date) Multi Index
        DataFrame of asset excess returns, one column per ticker.
        beta (pd.DataFrame): The asset's Beta per period, indexed by period with one
        column per ticker (see `get_beta`).
        benchmark_excess_returns (pd.Series): A "within period" (period, date) Multi Index
        Series of benchmark excess returns.

    Returns:
        pd.DataFrame: The pointwise CAPM regression residuals, indexed the same way as
        `excess_returns`.
    """
    periods = excess_returns.index.get_level_values(0).unique()
    period_residuals_list = []

    for sub_period in periods:
        period_excess_returns = excess_returns.loc[sub_period]
        period_benchmark_excess_return = benchmark_excess_returns.loc[sub_period]
        period_beta = beta.loc[sub_period]

        period_residuals = pd.DataFrame(
            index=period_excess_returns.index,
            columns=period_excess_returns.columns,
            dtype=np.float64,
        )

        for column in period_excess_returns.columns:
            period_residuals[column] = get_jensens_alpha(
                period_excess_returns[column],
                0.0,
                period_beta[column],
                period_benchmark_excess_return,
            )

        period_residuals_list.append(period_residuals)

    # pd.concat rebuilds the Multi Index; partial .loc is unreliable under CoW.
    return pd.concat(
        period_residuals_list, keys=periods, names=excess_returns.index.names
    )


def get_appraisal_ratio(
    jensens_alpha: pd.Series | pd.DataFrame,
    capm_residuals: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Appraisal Ratio, i.e. Jensen's Alpha divided by the idiosyncratic
    (residual, unsystematic) standard deviation left over from the CAPM regression that
    produced that Alpha.

    Jensen's Alpha (see `get_jensens_alpha`) measures how much return a manager generated
    above what CAPM would predict given the asset's Beta. However, a large Alpha achieved
    with wildly noisy, unpredictable residual returns is far less attractive than the same
    Alpha achieved consistently. The Appraisal Ratio normalizes Alpha by that noise (the
    "specific risk" not explained by market exposure), giving a Sharpe-ratio-like measure
    of stock-picking or timing skill per unit of idiosyncratic risk taken.

    The formula is as follows:

        - Appraisal Ratio = Jensen's Alpha / Residual Standard Deviation

    Where the residual standard deviation is the standard deviation of the pointwise CAPM
    regression residuals, i.e. of `capm_residuals`, computed as:

        - CAPM Residual = Asset Excess Return − Beta * Benchmark Excess Return

    which is exactly the same formula as `get_jensens_alpha`, evaluated on already-excess
    returns with the risk-free rate set to 0 (see the Notes below and `get_capm_residuals`
    for the regression machinery this reuses).

    Also known as: Treynor-Black Appraisal Ratio, information ratio of a CAPM regression.

    For more information about the method, see the following paper:

    - Treynor, J.L., & Black, F. (1973). "How to Use Security Analysis to Improve
    Portfolio Selection." Journal of Business, 46(1), 66-86.

    Args:
        jensens_alpha (pd.Series | pd.DataFrame): Jensen's Alpha, see `get_jensens_alpha`.
        capm_residuals (pd.Series | pd.DataFrame): The pointwise CAPM regression residuals
        over the estimation window used for `jensens_alpha`, from which the idiosyncratic
        (residual) standard deviation is derived. When this has a "within period" Multi
        Index (period, date), the standard deviation is computed separately within each
        period; otherwise it is computed over the entire series at once.

    Returns:
        pd.Series | pd.DataFrame: Appraisal Ratio values.

    Notes:
    - `capm_residuals` is typically obtained via `get_capm_residuals`, which calls
    `get_jensens_alpha` under the hood on already-excess returns (i.e. with
    `risk_free_rate` set to 0), so it reuses the exact same CAPM regression machinery as
    `jensens_alpha` itself rather than reimplementing it.
    """
    if capm_residuals.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        residual_standard_deviation = capm_residuals.groupby(level=0).std()
    else:
        residual_standard_deviation = capm_residuals.std()

    return jensens_alpha / residual_standard_deviation


def get_fama_decomposition(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
    asset_standard_deviation: pd.Series | pd.DataFrame,
    benchmark_standard_deviation: pd.Series | float,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Calculate the Fama (1972) decomposition of total excess return into Selectivity and
    Diversification.

    Jensen's Alpha alone conflates two very different sources of excess return: genuine
    stock/timing selection skill, and simply carrying more total risk than the market by
    holding an under-diversified portfolio (which, in a CAPM world, should be compensated
    with extra return even absent any skill). Fama's decomposition separates the two by
    comparing the portfolio's actual return against two different CAPM-implied return
    benchmarks:

        - one using the portfolio's actual Beta (systematic risk only), and
        - one using the portfolio's actual *total* risk ratio Sigma_Portfolio / Sigma_Market
        in place of Beta, i.e. the return CAPM would imply if the portfolio's entire
        standard deviation (not just its market-correlated slice) were systematic.

    The formulas are as follows:

        - Selectivity = (Asset Return − Risk-Free Rate) − (Sigma_Portfolio / Sigma_Market)
            * (Benchmark Return − Risk-Free Rate)
        - Diversification = [Risk-Free Rate + (Sigma_Portfolio / Sigma_Market)
            * (Benchmark Return − Risk-Free Rate)] − [Risk-Free Rate + Beta * (Benchmark Return − Risk-Free Rate)]

    Selectivity is the return earned above what would be required for a *fully
    diversified* portfolio carrying the same total risk, i.e. genuine security selection
    or timing skill. Diversification is the extra return the manager left on the table (if
    positive, it is a cost) by taking on unsystematic risk that a fully diversified
    portfolio of the same total risk would not have: Sigma_Portfolio / Sigma_Market >= Beta
    always holds (since Beta only captures the market-correlated fraction of total risk),
    so Diversification >= 0 whenever Beta < Sigma_Portfolio / Sigma_Market, i.e. whenever
    the portfolio is not perfectly diversified relative to the benchmark.

    Selectivity + Diversification + Beta * (Benchmark Return − Risk-Free Rate) reconstructs
    the portfolio's total excess return exactly; equivalently, Jensen's Alpha (see
    `get_jensens_alpha`) equals Selectivity + Diversification.

    Also known as: Fama's Net Selectivity, Fama performance decomposition.

    For more information about the method, see the following paper:

    - Fama, E.F. (1972). "Components of Investment Performance." Journal of Finance,
    27(3), 551-567.

    Notes:
    - Terminology: in Fama's own (1972) notation, "Selectivity" refers specifically to
    (Asset Return − Risk-Free Rate) − Beta * (Benchmark Return − Risk-Free Rate), which
    is exactly Jensen's Alpha (see `get_jensens_alpha`). Fama then splits that quantity
    into "Diversification" and "Net Selectivity", where Net Selectivity is what this
    function returns as `selectivity`: (Asset Return − Risk-Free Rate) − (Sigma_Portfolio
    / Sigma_Market) * (Benchmark Return − Risk-Free Rate). The `diversification` return
    value matches Fama's "Diversification" term exactly. The reconstruction identity
    (Net Selectivity + Diversification = Selectivity = Jensen's Alpha) holds regardless
    of which of these two labels is used for the first term, but readers cross-checking
    against the original paper's exact terminology should substitute "Net Selectivity"
    wherever this function and its caller (`get_fama_decomposition` in
    `performance_controller.py`) say "Selectivity".

    Args:
        asset_returns (pd.Series | pd.DataFrame | float): The asset's or portfolio's
        return over the period.
        risk_free_rate (pd.Series | float): The risk-free rate over the same period.
        beta (pd.Series | pd.DataFrame | float): The asset's or portfolio's Beta, see
        `get_beta`.
        benchmark_returns (pd.Series | float): The benchmark's return over the same
        period.
        asset_standard_deviation (pd.Series | pd.DataFrame): The asset's or portfolio's
        total return standard deviation (Sigma_Portfolio) over the same window used to
        estimate `beta`.
        benchmark_standard_deviation (pd.Series | float): The benchmark's total return
        standard deviation (Sigma_Market) over the same window used to estimate `beta`.

    Returns:
        tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]: The Selectivity and
        Diversification components, in that order. Their sum equals Jensen's Alpha.
    """
    if isinstance(beta, pd.DataFrame) and isinstance(asset_returns, pd.DataFrame):
        total_risk_ratio = asset_standard_deviation.div(
            benchmark_standard_deviation, axis=0
        )

        selectivity = pd.DataFrame(
            index=beta.index, columns=beta.columns, dtype=np.float64
        )
        diversification = pd.DataFrame(
            index=beta.index, columns=beta.columns, dtype=np.float64
        )

        for column in beta.columns:
            capm_actual_beta = risk_free_rate + beta[column] * (
                benchmark_returns - risk_free_rate
            )
            capm_total_risk = risk_free_rate + total_risk_ratio[column] * (
                benchmark_returns - risk_free_rate
            )

            selectivity.loc[:, column] = asset_returns[column] - capm_total_risk
            diversification.loc[:, column] = capm_total_risk - capm_actual_beta
    elif isinstance(beta, (pd.Series | float)) and isinstance(
        asset_returns, (pd.Series | float)
    ):
        total_risk_ratio = asset_standard_deviation / benchmark_standard_deviation

        capm_actual_beta = risk_free_rate + beta * (benchmark_returns - risk_free_rate)
        capm_total_risk = risk_free_rate + total_risk_ratio * (
            benchmark_returns - risk_free_rate
        )

        selectivity = asset_returns - capm_total_risk
        diversification = capm_total_risk - capm_actual_beta
    else:
        raise TypeError(
            "Expects pd.DataFrame for both Asset Returns and Beta or pd.Series / Float "
            "for both Asset Returns and Beta"
        )

    return selectivity, diversification


def get_adjusted_sharpe_ratio(
    sharpe_ratio: pd.Series | pd.DataFrame | float,
    skewness: pd.Series | pd.DataFrame | float,
    kurtosis: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Adjusted Sharpe Ratio (ASR).

    The Sharpe ratio only looks at the mean and standard deviation of returns, implicitly
    assuming a Normal distribution. The Adjusted Sharpe Ratio (Pezier & White, 2006)
    penalizes (or rewards) the Sharpe ratio for negative skewness and excess kurtosis using
    a Cornish-Fisher-style expansion, so that two strategies with the same Sharpe ratio but
    different tail shapes are no longer scored identically: a strategy with negative
    skewness (large, infrequent losses) or fat tails (high kurtosis) is penalized relative
    to one with positive skewness or thin tails.

    The formula is as follows:

        - ASR = SR * [1 + (S / 6) * SR − ((K − 3) / 24) * SR^2]

    Where SR is the (ordinary, period) Sharpe ratio, S is the skewness (γ₃) of the same
    returns, and K is the non-excess (raw) kurtosis (γ₄) of the same returns.

    Note that this formula requires **non-excess (raw) kurtosis**, i.e. a Normal
    distribution has a kurtosis of 3, not 0 — exactly the same convention documented in
    `get_probabilistic_sharpe_ratio` and used by `get_deflated_sharpe_ratio`. This
    codebase's `risk_model.get_kurtosis` defaults to `fisher=True` (excess kurtosis, Normal
    = 0); either call it with `fisher=False` before passing the result in here, or add 3 to
    a Fisher/excess kurtosis value first. Getting this convention backwards flips the sign
    of the ((K - 3) / 24) * SR^2 term for any return series that is actually close to
    Normal (where raw kurtosis is close to 3 but excess/Fisher kurtosis is close to 0),
    silently turning a near-zero adjustment into a large, wrong one — exactly the kind of
    sign/convention bug this codebase has been bitten by before, so double check which
    convention is being fed in.

    For more information about the method, see the following paper:

    - Pezier, J., & White, A. (2006). "The Relative Merits of Investable Hedge Fund
    Indices and of Funds of Hedge Funds in Optimal Passive Portfolios." ICMA Centre
    Discussion Papers in Finance, DP2006-10.

    Args:
        sharpe_ratio (pd.Series | pd.DataFrame | float): The (ordinary) Sharpe ratio, see
        `get_sharpe_ratio`.
        skewness (pd.Series | pd.DataFrame | float): The skewness (γ₃) of the same returns
        used to compute `sharpe_ratio`.
        kurtosis (pd.Series | pd.DataFrame | float): The non-excess (raw) kurtosis (γ₄) of
        the same returns used to compute `sharpe_ratio`. See the Notes above — a Normal
        distribution has a kurtosis of 3 under this convention, not 0.

    Returns:
        pd.Series | pd.DataFrame | float: The Adjusted Sharpe Ratio values.
    """
    return sharpe_ratio * (
        1 + (skewness / 6) * sharpe_ratio - ((kurtosis - 3) / 24) * sharpe_ratio**2
    )


def get_starr_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the STARR (Stable Tail Adjusted Return Ratio) of returns.

    The Sharpe ratio penalizes upside and downside volatility equally via the standard
    deviation. The STARR ratio instead scales the mean excess return by the Conditional
    Value at Risk (CVaR / Expected Shortfall), a coherent tail-risk measure that only looks
    at the average magnitude of losses beyond the `alpha` quantile. This makes STARR more
    appropriate than the Sharpe ratio for return distributions with fat left tails, where
    the standard deviation understates the risk that actually matters to an investor.

    The formula is as follows:

        - STARR Ratio = Excess Return / |CVaR(alpha)|

    Where CVaR is computed on the raw (non-excess) `returns` using the historical method
    (see `financetoolkit.risk.cvar_model.get_cvar_historic`), which this function calls
    directly rather than reimplementing the CVaR calculation. `excess_returns` is divided
    by CVaR elementwise (e.g. per period), so pass already-aggregated excess returns (e.g.
    one value per period) rather than a raw daily series if a mean excess return per period
    is intended — this function does not average `excess_returns` itself.

    Also known as: Stable Tail Adjusted Return Ratio, Conditional Sharpe Ratio.

    For more information about the method, see the following paper:

    - Martin, R.D., Rachev, S.T., & Siboulet, F. (2003). "Phi-Alpha Optimal Portfolios and
    Extreme Risk Management." Wilmott Magazine of Finance, 6, 70-83.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        the risk-free rate subtracted.
        returns (pd.Series | pd.DataFrame): The corresponding raw (non-excess) returns,
        from which the CVaR denominator is computed.
        alpha (float, optional): The confidence level used for the CVaR calculation (e.g.
        0.05 for the worst 5% of outcomes). Defaults to 0.05.

    Returns:
        pd.Series | pd.DataFrame: STARR Ratio values.
    """
    conditional_value_at_risk = cvar_model.get_cvar_historic(returns, alpha)

    starr_ratio = excess_returns / abs(conditional_value_at_risk)

    if isinstance(starr_ratio, pd.Series | pd.DataFrame):
        return starr_ratio.dropna()

    return starr_ratio


def get_rachev_ratio(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Rachev Ratio (R-Ratio) of returns.

    The Rachev ratio compares the "quality" of the best outcomes to the "quality" of the
    worst outcomes by taking the ratio of the right-tail Expected Shortfall (the average of
    the best `alpha` fraction of returns) to the left-tail Expected Shortfall (the average
    magnitude of the worst `alpha` fraction of returns). A ratio above 1 indicates that the
    average size of extreme gains outweighs the average size of extreme losses, i.e. the
    return distribution is favorably (right-)skewed in its tails; a ratio below 1 indicates
    the opposite.

    The formula is as follows:

        - Rachev Ratio = ES_right(alpha) / ES_left(alpha)
        - ES_right(alpha) = E[Return | Return >= Quantile(1 - alpha)]
        - ES_left(alpha) = -E[Return | Return <= Quantile(alpha)]

    Both tail expected shortfalls are computed by calling the existing
    `financetoolkit.risk.cvar_model.get_cvar_historic` function — the left tail directly on
    `returns`, and the right tail on `-returns` (which turns the upper tail of `returns`
    into the lower tail of the negated series, letting the same CVaR machinery compute it
    without reimplementing any tail-averaging logic) — rather than duplicating the CVaR
    calculation.

    Also known as: R-Ratio.

    For more information about the method, see the following paper:

    - Biglova, A., Ortobelli, S., Rachev, S.T., & Stoyanov, S. (2004). "Different Approaches
    to Risk Estimation in Portfolio Theory." Journal of Portfolio Management, 31(1), 103-112.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns.
        alpha (float, optional): The confidence level used for both tails (e.g. 0.05 for
        the best/worst 5% of outcomes). Defaults to 0.05.

    Returns:
        pd.Series | pd.DataFrame: Rachev Ratio values.
    """
    right_tail_expected_shortfall = -cvar_model.get_cvar_historic(-returns, alpha)
    left_tail_expected_shortfall = -cvar_model.get_cvar_historic(returns, alpha)

    rachev_ratio = right_tail_expected_shortfall / left_tail_expected_shortfall

    if isinstance(rachev_ratio, pd.Series | pd.DataFrame):
        return rachev_ratio.dropna()

    return rachev_ratio


def _get_market_timing_regression(
    excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    second_regressor: pd.Series,
    second_regressor_name: str,
) -> tuple[dict, pd.Series]:
    """
    Shared regression machinery for the Treynor-Mazuy and Henriksson-Merton market timing
    models, which both regress excess returns on the benchmark excess return plus one
    additional engineered regressor that captures convexity (Treynor-Mazuy) or a piecewise
    up-market slope (Henriksson-Merton).

    Args:
        excess_returns (pd.Series): The asset's excess returns.
        benchmark_excess_returns (pd.Series): The benchmark's excess returns.
        second_regressor (pd.Series): The additional engineered regressor.
        second_regressor_name (str): The name to use for the additional regressor's
        coefficient in the returned regression results (e.g. "Gamma" or "Up Market Beta").

    Returns:
        dict: the regression results.
        pd.Series: the residuals.

    Notes:
    - Rows in which the asset's or the benchmark's excess return is NaN are dropped
    before fitting. Every return series starts with at least one NaN (the first
    observation has no prior close to compare against), and `LinearRegression` raises
    on NaN inputs, so without this the regression fails for the very first period of
    any dataset.
    """
    regression_results_nan = {
        "Alpha": np.nan,
        "Beta": np.nan,
        second_regressor_name: np.nan,
        "R Squared": np.nan,
    }

    factor_dataset = pd.DataFrame(
        {
            "Benchmark Excess Return": benchmark_excess_returns,
            second_regressor_name: second_regressor,
        }
    ).reindex(excess_returns.index)

    valid = excess_returns.notna() & factor_dataset.notna().all(axis=1)
    excess_returns_valid = excess_returns[valid]
    factor_dataset = factor_dataset[valid]

    if len(excess_returns_valid) < 3:  # noqa
        # Robust handling of insufficient data points for regression method
        return regression_results_nan, excess_returns * np.nan

    model = LinearRegression()
    model.fit(factor_dataset, excess_returns_valid)

    y_pred = model.predict(factor_dataset)
    residuals = excess_returns_valid - y_pred

    r_squared = model.score(factor_dataset, excess_returns_valid)

    regression_results = {
        "Alpha": model.intercept_,
        "Beta": model.coef_[0],
        second_regressor_name: model.coef_[1],
        "R Squared": r_squared,
    }

    return regression_results, residuals


def get_treynor_mazuy_model(
    excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
) -> tuple[dict, pd.Series]:
    """
    Calculate the Treynor-Mazuy market timing model.

    Jensen's Alpha and Beta from a plain CAPM regression cannot distinguish stock-picking
    skill (selectivity) from market-timing skill (shifting exposure ahead of market moves).
    The Treynor-Mazuy model adds a quadratic term in the benchmark excess return to the
    regression: a manager who successfully increases (decreases) market exposure ahead of
    up (down) markets will show a return profile that curves upward as a function of the
    benchmark return, captured by a positive quadratic coefficient (Gamma).

    The formula is as follows:

        - Excess Return = Alpha + Beta * Benchmark Excess Return + Gamma * Benchmark Excess Return^2 + Residuals

    Gamma > 0 indicates positive market-timing ability (the portfolio's beta effectively
    rises in up markets and falls in down markets); Gamma <= 0 indicates no timing ability.
    Alpha in this regression captures selectivity net of any timing effect already absorbed
    by Gamma.

    Also known as: Treynor-Mazuy quadratic timing model, TM model.

    For more information about the method, see the following paper:

    - Treynor, J., & Mazuy, K. (1966). "Can Mutual Funds Outguess the Market?" Harvard
    Business Review, 44(4), 131-136.

    Args:
        excess_returns (pd.Series): The asset's excess returns.
        benchmark_excess_returns (pd.Series): The benchmark's excess returns, aligned to
        the same index as `excess_returns`.

    Returns:
        dict: The regression results (Alpha, Beta, Gamma, R Squared).
        pd.Series: The regression residuals.
    """
    if not isinstance(excess_returns, pd.Series) or not isinstance(
        benchmark_excess_returns, pd.Series
    ):
        raise TypeError(
            "Expects pd.Series for both excess_returns and benchmark_excess_returns."
        )

    return _get_market_timing_regression(
        excess_returns,
        benchmark_excess_returns,
        benchmark_excess_returns**2,
        "Gamma",
    )


def get_henriksson_merton_model(
    excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
) -> tuple[dict, pd.Series]:
    """
    Calculate the Henriksson-Merton market timing model.

    Like the Treynor-Mazuy model, this separates market-timing skill from selectivity, but
    models timing as a piecewise (rather than quadratic) change in Beta: an "up-market"
    Beta and a "down-market" Beta. This maps naturally onto a manager choosing between two
    discrete exposure levels (e.g. based on a market call) rather than continuously scaling
    exposure with the magnitude of the expected move.

    The formula is as follows:

        - Excess Return = Alpha + Beta1 * Benchmark Excess Return + Beta2 * max(Benchmark Excess Return, 0) + Residuals

    Beta1 is the "down-market" Beta (the portfolio's market exposure when the benchmark
    excess return is negative), and Beta1 + Beta2 is the "up-market" Beta (exposure when
    the benchmark excess return is positive). Beta2 > 0 indicates positive market-timing
    ability (higher exposure in up markets than down markets); Beta2 <= 0 indicates no
    timing ability.

    Also known as: Henriksson-Merton piecewise timing model, HM model.

    For more information about the method, see the following paper:

    - Henriksson, R.D., & Merton, R.C. (1981). "On Market Timing and Investment
    Performance II: Statistical Procedures for Evaluating Forecasting Skills." Journal of
    Business, 54(4), 513-533.

    Args:
        excess_returns (pd.Series): The asset's excess returns.
        benchmark_excess_returns (pd.Series): The benchmark's excess returns, aligned to
        the same index as `excess_returns`.

    Returns:
        dict: The regression results (Alpha, Beta, Up Market Beta, R Squared), where
        "Beta" is the down-market Beta and "Beta" + "Up Market Beta" is the up-market Beta.
        pd.Series: The regression residuals.
    """
    if not isinstance(excess_returns, pd.Series) or not isinstance(
        benchmark_excess_returns, pd.Series
    ):
        raise TypeError(
            "Expects pd.Series for both excess_returns and benchmark_excess_returns."
        )

    up_market_excess_returns = benchmark_excess_returns.clip(lower=0)

    return _get_market_timing_regression(
        excess_returns,
        benchmark_excess_returns,
        up_market_excess_returns,
        "Up Market Beta",
    )
