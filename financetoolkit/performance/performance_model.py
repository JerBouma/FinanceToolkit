"""Performance Model"""

import io
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from scipy.stats import linregress

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.performance.helpers import get_newey_west_lags
from financetoolkit.risk import cvar_model
from financetoolkit.risk.risk_model import (
    apply_to_stacked_windows,
    get_tail_ratio,
    get_wealth_and_peak,
)
from financetoolkit.utilities.requests_model import get_request
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    by_own_trading_days,
    drop_non_finite,
    to_period_index,
    validate_period,
)

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# A Sortino ratio of a single observation only reports its sign, so two are required.
MIN_SORTINO_OBSERVATIONS = 2

# The Euler-Mascheroni constant, used by the Deflated Sharpe Ratio.
EULER_MASCHERONI_CONSTANT = 0.5772156649015329


@by_own_trading_days(compound=True)
def get_covariance(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame | float:
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
        returns (pd.Series | pd.DataFrame): The asset returns, one column per asset, or a
            rolling window object of them.
        benchmark_returns (pd.Series | pd.DataFrame): The benchmark returns, aligned on the
            same index as `returns`.

    Returns:
        pd.Series | pd.DataFrame | float: The sample covariance (ddof=1) of each asset with
        the benchmark: a Series per column for a DataFrame, a float for a Series and the
        rolling covariance for a rolling window object.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(returns, pd.DataFrame):
            covariance = pd.Series(index=returns.columns)

            for column in returns.columns:
                covariance.loc[column] = returns[column].cov(benchmark_returns)

            return covariance

        if isinstance(returns, pd.Series):
            if not isinstance(benchmark_returns, pd.Series):
                raise TypeError(
                    "A Series of returns needs a Series of benchmark returns."
                )
            return returns.cov(benchmark_returns)

    # A rolling window object, which pairs with either a Series or a DataFrame.
    return returns.cov(benchmark_returns)


@by_own_trading_days(compound=True)
def get_beta(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate beta. Beta represents the slope in the linear regression between
    the asset returns and the benchmark returns:

        - Beta = Covariance(Asset Returns, Benchmark Returns) / Variance(Benchmark Returns)

    Beta is estimated on the raw returns, not on returns in excess of the risk free rate.
    With a risk free rate that barely moves within a period the two slopes are nearly
    identical, and raw returns keep beta independent of the chosen risk free rate.

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns, either a flat index or a
            'within period' MultiIndex (period, date) to get one beta per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.

    Returns:
        pd.Series | pd.DataFrame | float: A DataFrame of betas per period (rows) and asset
        (columns) for a 'within period' index, otherwise a Series of betas per asset for a
        DataFrame or a float for a Series. Assets with fewer than two dates shared with
        the benchmark get NaN.
    """
    if not isinstance(returns, pd.DataFrame | pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns

    # The covariance and the benchmark variance are taken over the same dates, those on
    # which both have a return. A benchmark variance over every date would bias the beta
    # of an asset with a shorter history (e.g. a later listing) towards the benchmark's
    # volatility in a period the asset was not part of.
    benchmark = pd.DataFrame(
        np.repeat(benchmark_returns.to_numpy()[:, None], frame.shape[1], axis=1),
        index=frame.index,
        columns=frame.columns,
    )
    both = frame.notna() & benchmark.notna()
    asset, benchmark = frame.where(both), benchmark.where(both)

    within_period = frame.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    groups = frame.index.get_level_values(0) if within_period else np.zeros(len(frame))

    asset_deviation = asset - asset.groupby(groups).transform("mean")
    benchmark_deviation = benchmark - benchmark.groupby(groups).transform("mean")

    # The degrees of freedom of the covariance and the variance cancel out, and a beta
    # needs at least two shared dates (min_count).
    beta = (asset_deviation * benchmark_deviation).groupby(groups).sum(min_count=2) / (
        benchmark_deviation**2
    ).groupby(groups).sum(min_count=2)

    if within_period:
        return drop_non_finite(beta)

    beta = drop_non_finite(beta.iloc[0])

    return beta if isinstance(returns, pd.DataFrame) else float(beta.iloc[0])


@by_own_trading_days(compound=True)
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

    rolling_beta = pd.DataFrame(
        index=returns.index, columns=returns.columns, dtype=np.float64
    )

    for column in returns.columns:
        # The benchmark variance is taken over the same dates as the covariance, those
        # on which the asset has a return: over every date it would mix in benchmark
        # volatility from days the asset did not trade (e.g. before a later listing).
        benchmark = benchmark_returns.where(returns[column].notna())
        rolling_cov.loc[:, column] = get_covariance(
            returns[column].rolling(window=window_size), benchmark
        )
        rolling_beta.loc[:, column] = (
            rolling_cov[column] / benchmark.rolling(window=window_size).var()
        )

    return rolling_beta


@by_own_trading_days(compound=True)
def get_beta_implied_return(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    period: str,
    beta_window: int,
) -> pd.DataFrame:
    """
    Split each period's return into the part the market explains and the asset's own part:

        - Implied Return = Beta × Benchmark Return
        - Residual = Actual Return − Implied Return

    The beta of a period is the rolling beta of the daily returns over the `beta_window`
    trading days that end on the day before the period starts, so a period never helps
    estimate the beta that explains it. The period returns compound the daily returns,
    as `get_returns` does.

    Args:
        returns (pd.DataFrame): The daily asset returns, a daily PeriodIndex.
        benchmark_returns (pd.Series): The daily benchmark returns on the same index.
        period (str): The period to explain: daily, weekly, monthly, quarterly or yearly.
        beta_window (int): The number of daily returns the beta is estimated on.

    Returns:
        pd.DataFrame: The periods as rows and a MultiIndex of (measure, ticker) as columns,
        with the measures Beta, Benchmark Return, Implied Return, Actual Return and Residual.
    """
    # The window ends on the day before: the shift keeps a day out of its own beta.
    daily_beta = get_rolling_beta(returns, benchmark_returns, beta_window).shift(1)

    if period == "daily":
        beta, actual_return, benchmark_return = daily_beta, returns, benchmark_returns
    else:
        actual_return = get_returns(returns, period)
        benchmark_return = get_returns(benchmark_returns, period)

        # Each period takes the beta of its first trading day, the one known when it began.
        labels = to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period])
        beta = daily_beta.groupby(labels).nth(0)
        beta.index = labels[~labels.duplicated()]

    benchmark = pd.DataFrame(
        np.repeat(benchmark_return.to_numpy()[:, None], returns.shape[1], axis=1),
        index=actual_return.index,
        columns=returns.columns,
    )
    implied_return = beta.reindex(actual_return.index) * benchmark

    return pd.concat(
        {
            "Beta": beta.reindex(actual_return.index),
            "Benchmark Return": benchmark,
            "Implied Return": implied_return,
            "Actual Return": actual_return,
            "Residual": actual_return - implied_return,
        },
        axis=1,
    )


BETA_METHODS = (
    "ols",
    "blume",
    "vasicek",
    "dimson",
    "scholes_williams",
    "upside",
    "downside",
)

# Blume (1971, 1975) found betas regress towards 1 by about a third between estimation
# windows; Bloomberg's adjusted beta applies the rounded weights 2/3 and 1/3.
BLUME_WEIGHT = 0.67

# The Vasicek prior for a single asset, which has no cross-section to estimate it from:
# the market's own beta as the mean and a standard deviation of 0.5 around it.
VASICEK_SINGLE_ASSET_PRIOR = (1.0, 0.25)


def _slope(asset: np.ndarray, market: np.ndarray) -> tuple[float, float]:
    """
    The least squares slope of the asset on the market and the squared standard
    error of that slope, over the observations both have. Fewer than three shared
    observations (or a market without variance) give NaN.
    """
    shared = ~(np.isnan(asset) | np.isnan(market))
    asset, market = asset[shared], market[shared]

    if len(asset) < 3:  # noqa: PLR2004
        return np.nan, np.nan

    market_deviation = market - market.mean()
    sum_of_squares = (market_deviation**2).sum()

    if sum_of_squares == 0:
        return np.nan, np.nan

    slope = (market_deviation * (asset - asset.mean())).sum() / sum_of_squares
    residuals = asset - asset.mean() - slope * market_deviation

    return slope, (residuals**2).sum() / (len(asset) - 2) / sum_of_squares


def _beta_of_sample(
    asset_returns: pd.DataFrame, market_returns: pd.DataFrame, method: str, lags: int
) -> pd.Series:
    """
    The beta of every asset over one sample of returns.

    Args:
        asset_returns (pd.DataFrame): The asset returns of the sample (one column per asset).
        market_returns (pd.DataFrame): The benchmark returns on the same rows, one column per
            shift k from -lags to lags holding the benchmark return of k periods later.
        method (str): One of BETA_METHODS.
        lags (int): The number of leads and lags of the Dimson beta.

    Returns:
        pd.Series: The beta per asset.
    """
    market = market_returns[0].to_numpy(dtype=float)
    betas, variances = {}, {}

    for column in asset_returns.columns:
        asset = asset_returns[column].to_numpy(dtype=float)

        if method in ("ols", "blume", "vasicek"):
            betas[column], variances[column] = _slope(asset, market)
        elif method in ("upside", "downside"):
            # The benchmark's direction picks the observations; a flat benchmark falls in
            # neither.
            direction = market > 0 if method == "upside" else market < 0
            betas[column] = _slope(asset[direction], market[direction])[0]
        elif method == "dimson":
            shifts = market_returns.loc[:, list(range(-lags, lags + 1))].to_numpy(
                dtype=float
            )
            complete = ~(np.isnan(asset) | np.isnan(shifts).any(axis=1))

            if complete.sum() <= shifts.shape[1] + 1:
                betas[column] = np.nan
                continue

            design = np.column_stack([np.ones(complete.sum()), shifts[complete]])
            coefficients = np.linalg.lstsq(design, asset[complete], rcond=None)[0]
            # The intercept is left out: the beta is the sum of the slopes on the
            # lagged, contemporaneous and leading benchmark returns.
            betas[column] = coefficients[1:].sum()
        else:
            # Scholes and Williams (1977): the three simple slopes and the benchmark's
            # first-order autocorrelation over the same observations.
            shifts = market_returns.loc[:, [-1, 0, 1]].to_numpy(dtype=float)
            complete = ~(np.isnan(asset) | np.isnan(shifts).any(axis=1))
            sample, shifted = asset[complete], shifts[complete]

            if len(sample) < 3:  # noqa: PLR2004
                betas[column] = np.nan
                continue

            slopes = sum(_slope(sample, shifted[:, index])[0] for index in range(3))
            autocorrelation = np.corrcoef(shifted[:, 1], shifted[:, 0])[0, 1]
            betas[column] = slopes / (1 + 2 * autocorrelation)

    beta = pd.Series(betas, dtype=float)

    if method == "blume":
        return BLUME_WEIGHT * beta + (1 - BLUME_WEIGHT)

    if method == "vasicek":
        variance = pd.Series(variances, dtype=float)
        estimated = beta.dropna()

        if len(estimated) >= 2:  # noqa: PLR2004
            prior_mean, prior_variance = estimated.mean(), estimated.var(ddof=1)
        else:
            prior_mean, prior_variance = VASICEK_SINGLE_ASSET_PRIOR

        # The posterior weighs the asset's own beta against the prior by their
        # precisions: a noisy estimate (a large standard error) moves towards the
        # prior mean, a precise one keeps close to its own value.
        return (prior_variance * beta + variance * prior_mean) / (
            prior_variance + variance
        )

    return beta


@by_own_trading_days(compound=True)
def get_beta_estimate(
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    method: str = "blume",
    lags: int = 1,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate beta with an estimator other than ordinary least squares:

        - Blume: 0.67 x OLS Beta + 0.33, the adjusted beta of Bloomberg (Blume, 1971, 1975).
        - Vasicek: the OLS beta shrunk towards the cross-sectional mean beta of the assets,
          weighted by the precisions: (σ²_prior x β + SE² x β_prior) / (σ²_prior + SE²)
          (Vasicek, 1973). The prior is the mean and sample variance (ddof=1) of the OLS
          betas of the assets in the same period; one asset alone gets a prior mean of 1
          with a variance of 0.25.
        - Dimson: the sum of the slopes of a regression of the asset returns on the
          benchmark returns from `lags` periods before to `lags` periods after (Dimson, 1979).
        - Scholes-Williams: (β₋₁ + β₀ + β₊₁) / (1 + 2ρ), the simple slopes on the previous,
          same and next benchmark return over the benchmark's first-order autocorrelation
          ρ (Scholes and Williams, 1977).
        - Upside and downside: the OLS beta over the observations on which the benchmark
          rose or fell.

    Dimson and Scholes-Williams correct the downward bias of the OLS beta of assets that
    trade infrequently or react late to the market. The leads and lags come from the full
    history, so the first observation of a period uses the benchmark return before it.

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns, either a flat index or a
            'within period' MultiIndex (period, date) to get one beta per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        method (str, optional): One of "blume", "vasicek", "dimson", "scholes_williams",
            "upside", "downside" or "ols". Defaults to "blume".
        lags (int, optional): The leads and lags of the Dimson beta. Defaults to 1.
        window_size (int | None, optional): A rolling window over a flat index instead of
            one beta per period. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The betas per period (rows) and asset (columns) for a
        'within period' index or a rolling window, otherwise a Series of betas per asset.
        Assets with fewer than three shared observations get NaN.
    """
    if method not in BETA_METHODS:
        raise ValueError(
            f"The method must be one of {', '.join(BETA_METHODS)}, not {method}."
        )

    if lags < 1:
        raise ValueError("The number of lags must be at least 1.")

    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    market = pd.concat(
        {shift: benchmark_returns.shift(-shift) for shift in range(-lags, lags + 1)},
        axis=1,
    )

    if window_size:
        betas = {
            frame.index[end - 1]: _beta_of_sample(
                frame.iloc[end - window_size : end],
                market.iloc[end - window_size : end],
                method,
                lags,
            )
            for end in range(window_size, len(frame) + 1)
        }

        return pd.DataFrame(betas, index=frame.columns).T.reindex(frame.index)

    if frame.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        periods = frame.index.get_level_values(0)

        return pd.DataFrame(
            {
                period: _beta_of_sample(
                    frame[periods == period], market[periods == period], method, lags
                )
                for period in periods.unique()
            },
            index=frame.columns,
        ).T.rename_axis(frame.index.names[0])

    return _beta_of_sample(frame, market, method, lags)


def get_capital_asset_pricing_model(
    risk_free_rate: pd.Series | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame | float:
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
        benchmark_returns (pd.Series | pd.DataFrame | float): the benchmark returns, or with a
            DataFrame beta, a column of market returns per column of beta.

    Returns:
        pd.Series | pd.DataFrame | float: the capital asset pricing model.
    """
    if isinstance(beta, pd.DataFrame):
        capital_asset_pricing_model = pd.DataFrame(
            index=beta.index, columns=beta.columns, dtype=np.float64
        )
        for column in capital_asset_pricing_model.columns:
            # A market return per column (e.g. a country risk premium per company) or one for all.
            market_returns = (
                benchmark_returns[column]
                if isinstance(benchmark_returns, pd.DataFrame)
                else benchmark_returns
            )
            capital_asset_pricing_model.loc[:, column] = risk_free_rate + beta[
                column
            ] * (market_returns - risk_free_rate)
    elif isinstance(beta, (pd.Series | float)):
        capital_asset_pricing_model = risk_free_rate + beta * (
            benchmark_returns - risk_free_rate
        )
    else:
        raise TypeError("Expects pd.DataFrame, pd.Series or float, no other value.")

    return capital_asset_pricing_model


# Ken French updates the factor files monthly, so a file downloaded in this session is
# kept for an hour: every factor model call would otherwise download it again when the
# Toolkit's cache is not enabled.
FACTOR_FILE_LIFETIME_SECONDS = 3600
_factor_files: dict[str, tuple[float, bytes]] = {}


KEN_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# The regions Ken French publishes daily factors for, with their files spelled as the data
# library lists them. The server matches the names regardless of case, but the published
# spelling differs per region (only Asia Pacific ex Japan spells its momentum file "MOM").
# Every region's factors are U.S. dollar returns with the U.S. one-month T-bill as RF.
FACTOR_REGIONS = {
    "United States": {
        "five": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "three": "F-F_Research_Data_Factors_daily_CSV.zip",
        "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
    },
    "Developed": {
        "five": "Developed_5_Factors_Daily_CSV.zip",
        "three": "Developed_3_Factors_Daily_CSV.zip",
        "momentum": "Developed_Mom_Factor_Daily_CSV.zip",
    },
    "Developed ex US": {
        "five": "Developed_ex_US_5_Factors_Daily_CSV.zip",
        "three": "Developed_ex_US_3_Factors_Daily_CSV.zip",
        "momentum": "Developed_ex_US_Mom_Factor_Daily_CSV.zip",
    },
    "Europe": {
        "five": "Europe_5_Factors_Daily_CSV.zip",
        "three": "Europe_3_Factors_Daily_CSV.zip",
        "momentum": "Europe_Mom_Factor_Daily_CSV.zip",
    },
    "Japan": {
        "five": "Japan_5_Factors_Daily_CSV.zip",
        "three": "Japan_3_Factors_Daily_CSV.zip",
        "momentum": "Japan_Mom_Factor_Daily_CSV.zip",
    },
    "Asia Pacific ex Japan": {
        "five": "Asia_Pacific_ex_Japan_5_Factors_Daily_CSV.zip",
        "three": "Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip",
        "momentum": "Asia_Pacific_ex_Japan_MOM_Factor_Daily_CSV.zip",
    },
    "North America": {
        "five": "North_America_5_Factors_Daily_CSV.zip",
        "three": "North_America_3_Factors_Daily_CSV.zip",
        "momentum": "North_America_Mom_Factor_Daily_CSV.zip",
    },
}

# Ken French marks a missing value with -99.99 (or -999 in some files).
MISSING_FACTOR_VALUES = (-99.99, -999.0)


def factor_url(region: str = "United States", kind: str = "five") -> str:
    """
    The URL of Ken French's daily factor file of a region.

    Args:
        region (str, optional): One of FACTOR_REGIONS. Defaults to "United States".
        kind (str, optional): "five" (5 factors), "three" (3 factors) or "momentum".
            Defaults to "five".

    Returns:
        str: The URL of the ZIP file.

    Raises:
        ValueError: When the region is not published.
    """
    if region not in FACTOR_REGIONS:
        raise ValueError(
            f"The factors of {region!r} are not published. Choose from "
            f"{', '.join(FACTOR_REGIONS)}."
        )
    return f"{KEN_FRENCH_URL}{FACTOR_REGIONS[region][kind]}"


def read_factor_file(zip_data: bytes) -> pd.DataFrame:
    """
    Reads one of Ken French's daily factor files: the header is the first line that starts
    with a comma (the number of description lines above it differs per file), the dates are
    YYYYMMDD (padded with spaces in the regional files) and the values are in percent.

    Args:
        zip_data (bytes): The ZIP file.

    Returns:
        pd.DataFrame: The factors as decimals, indexed by day.
    """
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
        text = zip_file.open(zip_file.namelist()[0]).read().decode("latin-1")

    lines = text.splitlines()
    header = next(
        number for number, line in enumerate(lines) if line.lstrip().startswith(",")
    )
    dataset = pd.read_csv(
        io.StringIO("\n".join(lines[header:])), index_col=0, skipinitialspace=True
    )
    dataset.index = dataset.index.astype(str).str.strip()
    dataset = dataset[dataset.index.str.fullmatch(r"\d{8}")]
    dataset = dataset.loc[:, ~dataset.columns.str.contains("Unnamed")]
    dataset.columns = dataset.columns.str.strip()
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset = dataset.mask(dataset.isin(MISSING_FACTOR_VALUES)).dropna(how="all")

    # Ken French publishes the factors in percent while returns are decimals.
    dataset = dataset / 100
    dataset.index = pd.to_datetime(dataset.index, format="%Y%m%d").to_period(freq="D")
    dataset.index.name = "Date"
    return dataset


def download_factor_file(url: str) -> bytes:
    """
    Download one of Ken French's factor files, or reuse the copy downloaded within the hour.

    Args:
        url (str): The URL of the ZIP file.

    Returns:
        bytes: The ZIP file.
    """
    downloaded = _factor_files.get(url)

    if downloaded and time.monotonic() - downloaded[0] < FACTOR_FILE_LIFETIME_SECONDS:
        return downloaded[1]

    content = get_request(url, timeout=10).content
    _factor_files[url] = (time.monotonic(), content)

    return content


def obtain_fama_and_french_dataset(
    fama_and_french_url: str | None = None, region: str = "United States"
):
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
        5 factor file of the region is used.
        region (str, optional): The region whose factors to use, one of FACTOR_REGIONS: "United States",
        "Developed", "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan" or "North America".
        Every region's factors are U.S. dollar returns with the U.S. one-month Treasury bill as the
        risk-free rate (RF). Defaults to "United States".

    Returns:
        pd.DataFrame: the Fama and French 5 Factor Model dataset.
    """
    # Define the URL of the ZIP file
    fama_and_french_url = fama_and_french_url or factor_url(region, "five")

    cache = get_active_cache()

    if cache is not None:
        cached_dataset = cache.get(
            source=policy_model.KEN_FRENCH,
            dataset="factors_decimal",
            entity=fama_and_french_url,
        )

        if cached_dataset is not None:
            return cached_dataset

    fama_and_french_dataset = read_factor_file(
        download_factor_file(fama_and_french_url)
    )

    if cache is not None and not fama_and_french_dataset.empty:
        cache.set(
            source=policy_model.KEN_FRENCH,
            dataset="factors_decimal",
            entity=fama_and_french_url,
            data=fama_and_french_dataset,
        )

    return fama_and_french_dataset


def obtain_fama_and_french_three_factor_dataset(
    fama_and_french_url: str | None = None, region: str = "United States"
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
        provided, the 3 factor file of the region is used.
        region (str, optional): The region whose factors to use, see
        `obtain_fama_and_french_dataset`. Defaults to "United States".

    Returns:
        pd.DataFrame: the Fama and French 3 Factor Model dataset.
    """
    return obtain_fama_and_french_dataset(
        fama_and_french_url or factor_url(region, "three")
    )


def obtain_carhart_momentum_dataset(
    momentum_url: str | None = None, region: str = "United States"
) -> pd.DataFrame:
    """
    This functionality returns the Carhart Momentum (MOM) factor dataset, used to extend
    the Fama and French 3 Factor Model into the Carhart 4 Factor Model:

        - Momentum (MOM): Stocks with high prior returns (winners) tend to keep outperforming
        stocks with low prior returns (losers) over the medium term.

    The dataset is available on the website of Dartmouth College and is updated on a monthly
    basis. The dataset is packaged in a ZIP file, so it needs to be extracted first.

    Args:
        momentum_url (str): the URL of the ZIP file that contains the dataset. If no URL is
        provided, the momentum file of the region is used.
        region (str, optional): The region whose factor to use, see
        `obtain_fama_and_french_dataset`. Defaults to "United States".

    Returns:
        pd.DataFrame: the Carhart Momentum Factor dataset, with a single "Mom" column.
    """
    momentum_url = momentum_url or factor_url(region, "momentum")

    cache = get_active_cache()

    if cache is not None:
        cached_dataset = cache.get(
            source=policy_model.KEN_FRENCH,
            dataset="factors_decimal",
            entity=momentum_url,
        )

        if cached_dataset is not None:
            return cached_dataset

    momentum_dataset = read_factor_file(download_factor_file(momentum_url))
    # The regional files name the factor WML (winners minus losers).
    momentum_dataset = momentum_dataset.rename(columns={"WML": "Mom"})

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
) -> pd.Series:
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
        pd.Series: the correlation of each factor with the excess return.
    """
    correlations = factors.corrwith(excess_return)

    return correlations


def _ordinary_least_squares(
    regressors: pd.DataFrame, response: pd.Series
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Fits an ordinary least squares regression with an intercept, the way scikit-learn's
    LinearRegression does: the data is centered, the slopes are solved with LAPACK's
    gelsd and the intercept follows from the means.

    Args:
        regressors (pd.DataFrame): The independent variables.
        response (pd.Series): The dependent variable.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: The intercept, the slope per regressor and the
        fitted values.

    Raises:
        ValueError: When the data contains NaN or infinite values.
    """
    regressor_values = np.asarray(regressors, dtype=float)
    response_values = np.asarray(response, dtype=float)

    if not (np.isfinite(regressor_values).all() and np.isfinite(response_values).all()):
        raise ValueError(
            "Input contains NaN or infinity, which a regression cannot fit."
        )

    regressor_mean = regressor_values.mean(axis=0)
    response_mean = response_values.mean()
    coefficients, *_ = linalg.lstsq(
        regressor_values - regressor_mean,
        response_values - response_mean,
        lapack_driver="gelsd",
    )
    intercept = float(response_mean - regressor_mean @ coefficients)

    return intercept, coefficients, regressor_values @ coefficients + intercept


def _r_squared(response: pd.Series, fitted: np.ndarray) -> float:
    """
    Calculates the coefficient of determination, with scikit-learn's convention for a
    response without variance: 1 for a perfect fit and 0 otherwise.

    Args:
        response (pd.Series): The dependent variable.
        fitted (np.ndarray): The fitted values.

    Returns:
        float: The R squared.
    """
    response_values = np.asarray(response, dtype=float)
    residual_sum_of_squares = float(((response_values - fitted) ** 2).sum())
    total_sum_of_squares = float(
        ((response_values - response_values.mean()) ** 2).sum()
    )

    if total_sum_of_squares == 0:
        return 1.0 if residual_sum_of_squares == 0 else 0.0

    return 1 - residual_sum_of_squares / total_sum_of_squares


def get_fama_and_french_model_multi(
    excess_returns: pd.Series,
    factor_dataset: pd.DataFrame,
) -> tuple[dict, pd.Series, str | None]:
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

    intercept, coefficients, y_pred = _ordinary_least_squares(
        factor_dataset, excess_returns
    )

    # Check for sufficient samples before calculating R^2
    if factor_dataset.shape[0] < 2:  # noqa
        error_message = (
            "R2 score is not well-defined with less than two samples. "
            "Setting value to NaN"
        )
        r_squared = np.nan
    else:
        r_squared = _r_squared(excess_returns, y_pred)

    residuals = excess_returns - y_pred

    mse = float(np.mean((np.asarray(excess_returns, dtype=float) - y_pred) ** 2))

    regression_results = {"Intercept": intercept}

    for factor in factor_dataset.columns:
        regression_results[f"{factor} Slope"] = coefficients[
            factor_dataset.columns.get_loc(factor)
        ]

    regression_results["Mean Squared Error (MSE)"] = mse
    regression_results["R Squared"] = r_squared

    return regression_results, residuals, error_message


def get_fama_and_french_model_single(
    excess_returns: pd.Series,
    factor: pd.Series,
) -> tuple[dict, pd.Series]:
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
    fits the reverse regression (`factor` as a function of `excess_returns`), which generally yields a different slope and
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


@by_own_trading_days(compound=True)
def get_alpha(
    asset_returns: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
) -> pd.Series | pd.DataFrame | float:
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


@by_own_trading_days(compound=True)
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
    asset_returns: pd.Series | pd.DataFrame | float,
    risk_free_rate: pd.Series | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
) -> pd.Series | pd.DataFrame | float:
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
    asset_returns: pd.Series | pd.DataFrame | float,
    risk_free_rate: pd.Series | float,
    beta: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame | float:
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


def get_sharpe_ratio(
    excess_returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame | float:
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
        window_size (int): The size of the rolling window, in observations of `excess_returns`
            (e.g. 12 for a year of monthly returns).

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


def get_sortino_ratio(
    excess_returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame | float:
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
    still count towards N, exactly as in Sortino & Price (1994). This is deliberately not the same
    as `excess_returns[excess_returns < 0].std()`: that alternative would divide by
    (number of negative observations - 1) and measure the dispersion of the losses
    around *their own mean*, rather than their root-mean-square distance below the MAR
    of 0 -- a materially different (and non-standard) quantity that understates how
    volatile the downside really is whenever there are few, large losses.

    For more information about the method, see the following paper:

    - Sortino, F.A., & Price, L.N. (1994). "Performance Measurement in a Downside Risk
    Framework." Journal of Investing, 3(3), 59-64.

    A Sortino ratio needs at least two observations: with a single one the mean and the
    downside deviation are the same number, so the ratio would only report its sign (-1)
    or be infinite. Such periods (e.g. a period that holds only one trading day of the
    date range) are returned as NaN.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free
        rate already subtracted.

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """

    def _sortino(returns: pd.Series | pd.DataFrame):
        downside_deviation = np.sqrt((returns.clip(upper=0) ** 2).mean())
        ratio = returns.mean() / downside_deviation
        # A single observation only yields its own sign (see above).
        return (
            ratio.where(returns.count() >= MIN_SORTINO_OBSERVATIONS)
            if isinstance(ratio, pd.Series)
            else (ratio if returns.count() >= MIN_SORTINO_OBSERVATIONS else np.nan)
        )

    if isinstance(excess_returns, pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # Calculate Sortino ratio for each asset (ticker) in the DataFrame
            return excess_returns.groupby(level=0).apply(_sortino)

        return _sortino(excess_returns)

    if isinstance(excess_returns, pd.Series):
        # Calculate Sortino ratio for a single asset (ticker)
        return _sortino(excess_returns)

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
    # A window of a single observation only yields its sign, see `get_sortino_ratio`.
    if window_size < MIN_SORTINO_OBSERVATIONS:
        return excess_returns * np.nan

    rolling_mean = excess_returns.rolling(window=window_size).mean()
    rolling_downside_deviation = np.sqrt(
        (excess_returns.clip(upper=0) ** 2).rolling(window=window_size).mean()
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
    return (excess_returns / ulcer_index).dropna(how="all")


def get_m2_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    asset_standard_deviation: pd.Series | pd.DataFrame,
    benchmark_standard_deviation: pd.Series | float,
) -> pd.Series | pd.DataFrame:
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
    than M2 (a return, expressed in the same units as `asset_returns`).

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


@by_own_trading_days(compound=True)
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


@by_own_trading_days(compound=True)
def get_tracking_error(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Tracking Error of returns, i.e. the standard deviation of the
    difference between the asset's and the benchmark's return (the "active return").

    The formula is as follows:

        - Tracking Error = Standard Deviation(Asset Return - Benchmark Return)

    Args:
        asset_returns (pd.Series | pd.DataFrame): A Series or DataFrame of asset returns.
        benchmark_returns (pd.Series): The benchmark's returns over the same period.

    Returns:
        pd.Series | pd.DataFrame | float: Tracking Error values. With a "within period"
        (period, date) MultiIndex one value per period (rows) and asset (columns), for a
        plain DataFrame one value per asset (a Series) and for a Series a float.
    """
    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            tracking_error = asset_returns.groupby(level=0).apply(
                lambda x: (x.sub(benchmark_returns, axis=0)).std()
            )

            return tracking_error

        # One value per asset over the whole series.
        return asset_returns.sub(benchmark_returns, axis=0).std()

    if isinstance(asset_returns, (pd.Series | float)):
        tracking_error = (asset_returns - benchmark_returns).std()

    return tracking_error


@by_own_trading_days(compound=True)
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


@by_own_trading_days(compound=True)
def get_information_ratio(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame | float:
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
        pd.Series | pd.DataFrame | float: Information Ratio values. With a "within period"
        (period, date) MultiIndex one value per period (rows) and asset (columns), for a
        plain DataFrame one value per asset (a Series) and for a Series a float.
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

        # One value per asset over the whole series.
        difference = asset_returns.sub(benchmark_returns, axis=0)

        return difference.mean() / difference.std()

    if isinstance(asset_returns, (pd.Series | float)):
        difference = asset_returns - benchmark_returns
        information_ratio = difference.mean() / difference.std()

    return information_ratio


@by_own_trading_days(compound=True)
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
    return (returns / maximum_drawdown.abs()).dropna(how="all")


@by_own_trading_days
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
        wealth, peak = get_wealth_and_peak(returns)
        drawdowns = wealth / peak - 1

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
    return (returns / (average_drawdown.abs() + adjustment)).dropna(how="all")


@by_own_trading_days
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
        wealth, peak = get_wealth_and_peak(returns)
        drawdowns = wealth / peak - 1

    burke_drawdown_measure = np.sqrt((drawdowns[drawdowns < 0] ** 2).sum())

    # A series without a single return has no drawdowns to measure, rather than none.
    if isinstance(returns, pd.DataFrame):
        return burke_drawdown_measure.where(returns.notna().any())
    return burke_drawdown_measure if returns.notna().any() else np.nan


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
    return (excess_returns / burke_drawdown_measure).dropna(how="all")


def _geometric_mean_return(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    """
    The geometric average return per observation, (PRODUCT(1 + r))^(1 / n) - 1, over the
    non-missing returns of each column.

    Args:
        returns (pd.Series | pd.DataFrame): The returns to average.

    Returns:
        pd.Series | float: The geometric average return per column (or overall, for a Series).
    """
    # The mean of log(1 + r) skips missing values, so n is the count of each column.
    return np.expm1(np.log1p(returns).mean())


def _get_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    upside: bool,
) -> pd.Series | pd.DataFrame:
    """
    Shared implementation for the Upside and Downside Capture Ratios, following
    Morningstar's definition: the geometric average return of the asset over the
    observations in which the benchmark rose (fell), divided by the geometric average
    return of the benchmark over those same observations.

        - Capture Ratio = ((PRODUCT(1 + Asset Return))^(1 / n) - 1)
          / ((PRODUCT(1 + Benchmark Return))^(1 / n) - 1)

    Where n is the number of observations with a positive (negative) benchmark return
    on which the asset also has a return, so that both averages cover the same dates.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        upside (bool): Whether to compute the Upside (True) or Downside (False) Capture Ratio.

    Returns:
        pd.Series | pd.DataFrame: Capture Ratio values.
    """

    def _capture(assets: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
        mask = benchmark > 0 if upside else benchmark < 0
        selected = assets[mask]
        # The benchmark is averaged per asset over the dates that asset has a return,
        # so a shorter asset history is compared with the matching benchmark window.
        benchmark_per_asset = pd.DataFrame(
            np.repeat(benchmark[mask].to_numpy()[:, None], assets.shape[1], axis=1),
            index=selected.index,
            columns=assets.columns,
        ).where(selected.notna())

        return _geometric_mean_return(selected) / _geometric_mean_return(
            benchmark_per_asset
        )

    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            benchmark_name = benchmark_returns.name
            combination = pd.concat([asset_returns, benchmark_returns], axis=1)

            return combination.groupby(level=0).apply(
                lambda frame: _capture(
                    frame[asset_returns.columns], frame[benchmark_name]
                )
            )

        return _capture(asset_returns, benchmark_returns)

    if isinstance(asset_returns, pd.Series):
        return float(_capture(asset_returns.to_frame(), benchmark_returns).iloc[0])

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


@by_own_trading_days(compound=True)
def get_upside_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Upside Capture Ratio, i.e. the ratio of the asset's geometric average
    return to the benchmark's geometric average return during the observations in which
    the benchmark's return is positive (Morningstar's definition):

        - Upside Capture Ratio = ((PRODUCT(1 + Asset Return | Benchmark > 0))^(1 / n) - 1)
          / ((PRODUCT(1 + Benchmark Return | Benchmark > 0))^(1 / n) - 1)

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.

    Returns:
        pd.Series | pd.DataFrame: Upside Capture Ratio values.
    """
    return _get_capture_ratio(asset_returns, benchmark_returns, upside=True)


@by_own_trading_days(compound=True)
def get_downside_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Downside Capture Ratio, i.e. the ratio of the asset's geometric average
    return to the benchmark's geometric average return during the observations in which
    the benchmark's return is negative (Morningstar's definition):

        - Downside Capture Ratio = ((PRODUCT(1 + Asset Return | Benchmark < 0))^(1 / n) - 1)
          / ((PRODUCT(1 + Benchmark Return | Benchmark < 0))^(1 / n) - 1)

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.

    Returns:
        pd.Series | pd.DataFrame: Downside Capture Ratio values.
    """
    return _get_capture_ratio(asset_returns, benchmark_returns, upside=False)


@by_own_trading_days(compound=True)
def get_win_rate(
    asset_returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Win Rate, i.e. the percentage of periods in which the asset's return
    exceeds the benchmark's return. Periods in which either return is missing are left out
    of both the wins and the number of periods.

    The formula is as follows:

        - Win Rate = Number of Periods with Asset Return > Benchmark Return / Number of Periods
          with both an Asset and a Benchmark Return

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.

    Returns:
        pd.Series | pd.DataFrame: Win Rate values.
    """

    def win_share(
        asset: pd.Series | pd.DataFrame, benchmark: pd.Series
    ) -> pd.Series | float:
        # A comparison with a missing return is False, which would count a day without a quote as a loss; mask those days so the mean only covers days on which both returns exist.  # noqa: E501
        wins = asset.gt(benchmark, axis=0).astype(float)

        return wins.where(asset.notna()).where(benchmark.notna(), axis=0).mean()

    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            combination = pd.concat([asset_returns, benchmark_returns], axis=1)

            return combination.groupby(level=0).apply(
                lambda x: win_share(x[asset_returns.columns], x[benchmark_returns.name])
            )

        return win_share(asset_returns, benchmark_returns)

    if isinstance(asset_returns, pd.Series):
        return win_share(asset_returns, benchmark_returns)

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
        otherwise as pd.Series or pd.DataFrame with time as index. Without any return below
        the minimum acceptable return there are no losses to divide by and the ratio is
        undefined (NaN).
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

        return _divide_gains_by_losses(gains, losses)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def _divide_gains_by_losses(gains: float, losses: float) -> float:
    """
    The Omega ratio from the summed gains and losses, NaN without losses (the sum of no
    losses is -0.0, which would otherwise give minus infinity).
    """
    return gains / losses if losses > 0 else np.nan


def get_rolling_omega_ratio(
    returns: pd.Series | pd.DataFrame,
    window_size: int,
    minimum_acceptable_return: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Omega Ratio of returns.

    A window without losses (no return below the minimum acceptable return) has no ratio
    (NaN), like get_omega_ratio over the whole period.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used
        as the threshold between gains and losses. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Rolling Omega Ratio values with time as index.
    """

    def _omega(windows):
        excess_returns = windows - minimum_acceptable_return
        gains = excess_returns.clip(min=0).sum(axis=1)
        losses = (-excess_returns).clip(min=0).sum(axis=1)

        # A window without losses has no ratio.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(losses > 0, gains / losses, np.nan)

    return apply_to_stacked_windows(returns, window_size, _omega)


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

        # A period without losses (or without returns) divides by zero, which is inf or NaN.
        with np.errstate(divide="ignore", invalid="ignore"):
            return returns.sum() / losses

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_return_statistics(returns: pd.Series) -> pd.Series:
    """
    Calculate the statistics traders report about a series of returns: how often it wins,
    how large its wins and losses are, how long its streaks run and how it recovers from
    its drawdown.

    The formulas are as follows:

        - Win Rate = Number of Positive Returns / Number of Non-Zero Returns
        - Average Win = Average of the Positive Returns
        - Average Loss = Average of the Negative Returns
        - Payoff Ratio = Average Win / |Average Loss|
        - Profit Factor = Sum of the Positive Returns / |Sum of the Negative Returns|
        - Expectancy = Win Rate x Average Win + (1 - Win Rate) x Average Loss
        - Maximum Consecutive Wins (Losses) = The longest run of positive (negative) returns
        - Recovery Factor = Cumulative Return / |Maximum Drawdown|
        - Common Sense Ratio = Profit Factor x Tail Ratio (95th / 5th percentile)

    A return of exactly zero is neither a win nor a loss: it is left out of the win rate
    and ends a streak. The Recovery Factor compounds the returns into the cumulative
    return, so that it relates the same wealth path to the drawdown measured on it.

    For more information, see the following sources:

    - Tharp, V.K. (2006). "Trade Your Way to Financial Freedom." 2nd ed., McGraw-Hill.
    (Expectancy and the payoff ratio as the basis of evaluating a trading system.)

    Args:
        returns (pd.Series): A Series of returns; missing returns are left out.

    Returns:
        pd.Series: The statistics, indexed by their name. A statistic without the returns
        it needs (e.g. the Profit Factor or the Payoff Ratio without losses) is NaN, as
        for the Omega Ratio.
    """
    returns = returns.dropna()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    non_zero = len(wins) + len(losses)

    win_rate = len(wins) / non_zero if non_zero else np.nan
    average_win = wins.mean() if len(wins) else np.nan
    average_loss = losses.mean() if len(losses) else np.nan

    # Without losses there is nothing to divide by, as for the Omega Ratio.
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) else np.nan

    # Streaks: every change of sign (a zero return included) starts a new run.
    signs = np.sign(returns)
    runs = (signs != signs.shift()).cumsum()
    run_lengths = signs.groupby(runs).agg(["first", "size"])
    longest = run_lengths.groupby("first")["size"].max()

    wealth, peak = get_wealth_and_peak(returns)
    maximum_drawdown = (wealth / peak - 1).min() if len(returns) else np.nan
    cumulative_return = wealth.iloc[-1] - 1 if len(returns) else np.nan

    tail_ratio = get_tail_ratio(returns, 0.05) if len(returns) else np.nan

    return pd.Series(
        {
            "Win Rate": win_rate,
            "Average Win": average_win,
            "Average Loss": average_loss,
            "Payoff Ratio": (
                average_win / abs(average_loss) if len(losses) else np.nan
            ),
            "Profit Factor": profit_factor,
            # Equal to the average of the non-zero returns.
            "Expectancy": (
                (wins.sum() + losses.sum()) / non_zero if non_zero else np.nan
            ),
            "Max Consecutive Wins": (
                float(longest.get(1.0, 0)) if len(returns) else np.nan
            ),
            "Max Consecutive Losses": (
                float(longest.get(-1.0, 0)) if len(returns) else np.nan
            ),
            "Recovery Factor": (
                cumulative_return / abs(maximum_drawdown)
                if maximum_drawdown < 0
                else np.nan
            ),
            "Common Sense Ratio": profit_factor * tail_ratio,
        }
    )


def get_return_statistics_table(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the return statistics of `get_return_statistics` for every column, and for a
    within period index for every period.

    Args:
        returns (pd.DataFrame): A DataFrame of returns, one column per asset, optionally
            with a (period, date) index.

    Returns:
        pd.DataFrame: Over the whole history, the statistics as the index and the assets as
        the columns. Within periods, the periods as the index and (statistic, asset) as the
        columns.
    """
    if returns.index.nlevels != MULTI_PERIOD_INDEX_LEVELS:
        return returns.apply(get_return_statistics)

    statistics = {
        sub_period: returns.loc[sub_period].apply(get_return_statistics).stack()
        for sub_period in returns.index.get_level_values(0).unique()
    }

    return pd.DataFrame(statistics).T


def get_kelly_criterion(
    returns: pd.Series | pd.DataFrame,
    method: str = "discrete",
    fraction: float = 1.0,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Kelly Criterion, the share of capital to put at risk that maximizes the
    expected logarithmic growth of wealth.

    The formulas are as follows:

        - Discrete: Kelly = Win Rate - (1 - Win Rate) / Payoff Ratio
        - Continuous: Kelly = Average Excess Return / Variance of the Excess Returns

    The discrete form is Kelly's (1956) bet with a probability of winning and a ratio of the
    average win to the average loss, calculated from the returns (a zero return counting
    as neither). The continuous form is Thorp's (2006) for returns
    that are (close to) normally distributed, on excess returns over the risk free rate,
    with the sample variance (ddof=1); its value does not depend on the return frequency,
    since the average and the variance scale alike. A value above 1 means borrowing to
    invest, a negative value means the edge is negative (short or stay out).

    For more information, see the following sources:

    - Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical
    Journal, 35(4), 917-926.
    - Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock
    Market." Handbook of Asset and Liability Management, Volume 1, 385-428.
    - MacLean, L.C., Thorp, E.O. & Ziemba, W.T. (2011). "The Kelly Capital Growth Investment
    Criterion." World Scientific. (Fractional Kelly to reduce the volatility of growth.)

    Args:
        returns (pd.Series | pd.DataFrame): Returns for the discrete form, excess returns
            for the continuous form, optionally with a (period, date) index.
        method (str, optional): "discrete" or "continuous". Defaults to "discrete".
        fraction (float, optional): The share of the full Kelly bet, e.g. 0.5 for half
            Kelly. Defaults to 1.0.

    Returns:
        pd.Series | pd.DataFrame | float: The Kelly Criterion as a float for a Series, per
        column for a DataFrame and per period and column for a within period index. NaN
        when it is undefined: the discrete form without wins or losses, the continuous form
        without variance.
    """
    if method not in ("discrete", "continuous"):
        raise ValueError("Please choose a method of 'discrete' or 'continuous'.")

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            return pd.DataFrame(
                {
                    sub_period: get_kelly_criterion(
                        returns.loc[sub_period], method, fraction
                    )
                    for sub_period in returns.index.get_level_values(0).unique()
                }
            ).T

        return returns.apply(get_kelly_criterion, method=method, fraction=fraction)

    if not isinstance(returns, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    kelly = kelly_of_windows(returns.dropna().to_numpy(dtype=float)[None, :], method)

    return float(kelly[0]) * fraction


def get_rolling_kelly_criterion(
    returns: pd.Series | pd.DataFrame,
    window_size: int,
    method: str = "discrete",
    fraction: float = 1.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling Kelly Criterion of returns (discrete) or excess returns
    (continuous), as `get_kelly_criterion` calculates it.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        window_size (int): The size of the rolling window.
        method (str, optional): "discrete" or "continuous". Defaults to "discrete".
        fraction (float, optional): The share of the full Kelly bet. Defaults to 1.0.

    Returns:
        pd.Series | pd.DataFrame: Rolling Kelly Criterion values with time as index.
    """
    if method not in ("discrete", "continuous"):
        raise ValueError("Please choose a method of 'discrete' or 'continuous'.")

    return (
        apply_to_stacked_windows(
            returns, window_size, lambda windows: kelly_of_windows(windows, method)
        )
        * fraction
    )


def kelly_of_windows(windows: np.ndarray, method: str) -> np.ndarray:
    """
    Calculate the full Kelly Criterion of every row of an array of returns.

    Args:
        windows (np.ndarray): One set of returns per row, without missing values.
        method (str): "discrete" or "continuous".

    Returns:
        np.ndarray: The Kelly Criterion per row, NaN where it is undefined.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "continuous":
            variance = (
                windows.var(axis=1, ddof=1)
                if windows.shape[1] > 1
                else np.full(len(windows), np.nan)
            )

            return np.where(variance > 0, windows.mean(axis=1) / variance, np.nan)

        wins = windows > 0
        losses = windows < 0
        win_count = wins.sum(axis=1)
        loss_count = losses.sum(axis=1)

        win_rate = win_count / (win_count + loss_count)
        payoff_ratio = (np.where(wins, windows, 0).sum(axis=1) / win_count) / (
            np.abs(np.where(losses, windows, 0).sum(axis=1)) / loss_count
        )

        return np.where(
            (win_count > 0) & (loss_count > 0),
            win_rate - (1 - win_rate) / payoff_ratio,
            np.nan,
        )


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


# The average length of a year in days, leap years included, to express calendar time in years.
DAYS_PER_YEAR = 365.25


def get_compound_annual_growth_rate(
    prices: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    Calculate the Compound Annual Growth Rate (CAGR) of a price series on calendar time.

    The CAGR is the constant annual rate that grows the first price into the last one over
    the time between them. That time is measured in calendar days rather than in rows, so
    the rate does not depend on how often a price is observed: an asset that trades every
    day (such as a cryptocurrency) and one that trades on weekdays only, observed over the
    same dates, are measured over the same number of years.

    The formula is as follows:

        - Years = (Last Date - First Date) / 365.25 days
        - CAGR = (Last Price / First Price) ^ (1 / Years) - 1

    Each column is measured on its own first and last observation, so a ticker without a
    price on the first or last date of a shared calendar (a weekend for a stock next to a
    cryptocurrency, or a later listing) still gets the rate over the dates it has.

    Args:
        prices (pd.Series | pd.DataFrame): Series or DataFrame of prices with a
        DatetimeIndex or PeriodIndex, ordered from oldest to newest.

    Returns:
        float | pd.Series: The CAGR, as a float if `prices` is a Series and as a Series
        (one value per column) if `prices` is a DataFrame. A series with fewer than two
        prices, or whose prices fall on a single date, returns NaN.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.apply(get_compound_annual_growth_rate).astype(float)

    observed = prices.dropna()

    if len(observed) < 2:  # noqa: PLR2004
        return float("nan")

    dates = observed.index
    if isinstance(dates, pd.PeriodIndex):
        dates = dates.to_timestamp()

    years = (dates[-1] - dates[0]).days / DAYS_PER_YEAR

    if years <= 0:
        return float("nan")

    return float((observed.iloc[-1] / observed.iloc[0]) ** (1 / years) - 1)


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

    A period without any return (e.g. before a ticker listed) is NaN rather than 0.

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
    validate_period(period)

    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    dates = (
        groups
        if groups is not None
        else to_period_index(returns.index).asfreq(PERIOD_TRANSLATION[period])
    )

    # min_count=1 keeps a period without any return (a ticker that listed later or a gap in its history) NaN; the plain product of an empty group is 1, which would report a return of exactly 0 for it.  # noqa: E501
    period_returns = (1 + returns).groupby(dates).prod(min_count=1) - 1

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


@by_own_trading_days(compound=True)
def get_capm_residuals(
    excess_returns: pd.Series | pd.DataFrame,
    beta: pd.Series | pd.DataFrame,
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
    if not isinstance(excess_returns, pd.DataFrame) or not isinstance(
        beta, pd.DataFrame
    ):
        raise TypeError(
            "The residuals are computed per ticker, which needs a DataFrame of excess "
            "returns and a DataFrame of betas."
        )

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
        period and rescaled to the period frequency (see Notes below); otherwise it is
        computed over the entire series at once.

    Returns:
        pd.Series | pd.DataFrame: Appraisal Ratio values.

    Notes:
    - `capm_residuals` is typically obtained via `get_capm_residuals`, which calls
    `get_jensens_alpha` under the hood on already-excess returns (i.e. with
    `risk_free_rate` set to 0), so it reuses the exact same CAPM regression machinery as
    `jensens_alpha` itself rather than reimplementing it.
    - When `capm_residuals` has a "within period" (period, date) Multi Index, the residuals
    themselves are pointwise (e.g. daily), so their per-period standard deviation is a
    daily-scale figure. `jensens_alpha`, however, is always period-scale (e.g. yearly).
    Dividing a period-scale Alpha by a daily-scale standard deviation would silently
    inflate the ratio by roughly the square root of the number of observations in the
    period — the same timeframe-mixing mistake the Sharpe ratio guards against by
    de-annualizing the risk-free rate first. To keep both sides on the same frequency, the
    per-period residual standard deviation is scaled up by multiplying it with the square
    root of the number of residual observations in that period, mirroring how
    `financetoolkit.risk.risk_model.get_volatility` scales daily volatility to a period.
    """
    if capm_residuals.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        residual_standard_deviation = capm_residuals.groupby(level=0).std()
        # Each asset's own number of residuals in the period.
        observations_per_period = capm_residuals.groupby(level=0).count()
        residual_standard_deviation = residual_standard_deviation * np.sqrt(
            observations_per_period
        )
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

    When `returns` has a "within period" (period, date) Multi Index, CVaR is computed from
    the daily observations within each period (see `get_cvar_historic`), which makes it a
    daily-scale tail-risk measure. Dividing a period-scale `excess_returns` (e.g. a yearly
    excess return) by that daily-scale CVaR would silently inflate the ratio by roughly the
    square root of the number of trading days in the period — the same timeframe-mixing
    mistake the Sharpe ratio guards against by de-annualizing the risk-free rate before
    computing daily excess returns. To keep both sides of the ratio on the same frequency,
    the CVaR is instead scaled up to the period frequency by multiplying it with the square
    root of the number of return observations within that period, mirroring how
    `financetoolkit.risk.risk_model.get_volatility` scales daily volatility to a period.

    Also known as: Stable Tail Adjusted Return Ratio, Conditional Sharpe Ratio.

    For more information about the method, see the following paper:

    - Martin, R.D., Rachev, S.T., & Siboulet, F. (2003). "Phi-Alpha Optimal Portfolios and
    Extreme Risk Management." Wilmott Magazine of Finance, 6, 70-83.

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        the risk-free rate subtracted, already aggregated to one value per period (see
        Notes above).
        returns (pd.Series | pd.DataFrame): The corresponding raw (non-excess) returns,
        from which the CVaR denominator is computed. When this has a "within period" Multi
        Index (period, date), the resulting per-period CVaR is rescaled to the period
        frequency (see above) so it stays comparable to `excess_returns`.
        alpha (float, optional): The confidence level used for the CVaR calculation (e.g.
        0.05 for the worst 5% of outcomes). Defaults to 0.05.

    Returns:
        pd.Series | pd.DataFrame: STARR Ratio values.
    """
    conditional_value_at_risk = cvar_model.get_cvar_historic(returns, alpha)

    if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS and isinstance(
        conditional_value_at_risk, pd.Series | pd.DataFrame
    ):
        # Each asset's own number of returns in the period: a date only other assets
        # traded on is no observation of it.
        observations_per_period = returns.groupby(level=0).count()
        conditional_value_at_risk = conditional_value_at_risk * np.sqrt(
            observations_per_period
        )

    starr_ratio = excess_returns / abs(conditional_value_at_risk)

    if isinstance(starr_ratio, pd.Series | pd.DataFrame):
        return starr_ratio.dropna(how="all")

    return starr_ratio


def get_rachev_ratio(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series | pd.DataFrame | float:
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
        pd.Series | pd.DataFrame | float: Rachev Ratio values, a single float for a Series.
    """
    right_tail_expected_shortfall = -cvar_model.get_cvar_historic(-returns, alpha)
    left_tail_expected_shortfall = -cvar_model.get_cvar_historic(returns, alpha)

    rachev_ratio = right_tail_expected_shortfall / left_tail_expected_shortfall

    if isinstance(rachev_ratio, pd.Series | pd.DataFrame):
        return rachev_ratio.dropna(how="all")

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
    observation has no prior close to compare against), and the regression raises
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

    intercept, coefficients, y_pred = _ordinary_least_squares(
        factor_dataset, excess_returns_valid
    )
    residuals = excess_returns_valid - y_pred

    r_squared = _r_squared(excess_returns_valid, y_pred)

    regression_results = {
        "Alpha": intercept,
        "Beta": coefficients[0],
        second_regressor_name: coefficients[1],
        "R Squared": r_squared,
    }

    return regression_results, residuals


@by_own_trading_days(compound=True)
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


@by_own_trading_days(compound=True)
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


def get_style_weights(returns: pd.Series, style_returns: pd.DataFrame) -> pd.Series:
    """
    The weights of returns-based style analysis (Sharpe, 1992): the mix of style indices,
    every weight between 0 and 1 and the weights adding up to 1, that minimizes the variance
    of the difference between the fund's return and the weighted style returns. As

        - Var(Fund - Styles @ w) = Var(Fund) - 2 * w' Cov(Styles, Fund) + w' Cov(Styles) w

    the weights solve a quadratic programme in the covariances. It is scaled by the fund's
    variance and solved with its exact gradients: on daily returns the tracking variance is
    around 1e-6, so small that the optimizer's tolerance would otherwise stop it before the
    weights settle in the fourth decimal.

    For more information, see the following paper:

    - Sharpe, W.F. (1992). "Asset Allocation: Management Style and Performance Measurement."
      The Journal of Portfolio Management, 18(2), 7-19.

    Args:
        returns (pd.Series): The fund's returns.
        style_returns (pd.DataFrame): The returns of the style indices, a column each, on the
            same dates as `returns` without missing values.

    Returns:
        pd.Series: The weight of every style.
    """
    y = returns.to_numpy(dtype=float)
    x = style_returns.to_numpy(dtype=float)
    styles = x.shape[1]

    covariance = np.cov(np.column_stack([y, x]), rowvar=False, ddof=1)
    scale = covariance[0, 0] if covariance[0, 0] > 0 else 1.0
    style_covariance = covariance[1:, 1:] / scale
    fund_covariance = covariance[1:, 0] / scale

    fit = optimize.minimize(
        lambda weights: weights @ style_covariance @ weights
        - 2 * weights @ fund_covariance,
        np.full(styles, 1 / styles),
        jac=lambda weights: 2 * (style_covariance @ weights - fund_covariance),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * styles,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1,
                "jac": np.ones_like,
            }
        ],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    # SLSQP can end a hair outside the bounds, which would let a weight read as -0.0000.
    weights = np.clip(fit.x, 0, 1)

    return pd.Series(weights / weights.sum(), index=style_returns.columns)


def get_style_analysis(
    returns: pd.Series, style_returns: pd.DataFrame, periods_per_year: int = 12
) -> pd.Series:
    """
    Returns-based style analysis (Sharpe, 1992): the mix of style indices whose returns track
    a fund's returns most closely. The weights are found by minimizing the variance of the
    difference between the fund's return and the weighted style returns, with every weight
    between 0 and 1 and the weights adding up to 1, so they read as the fund's effective
    asset mix (see `get_style_weights`).

    The other figures are in-sample, measured over the same returns the weights are fitted
    to: the part of the fund's return the styles do not explain is the Selection Return, its
    volatility the Tracking Error and the share of the fund's variance the styles explain the
    R Squared. Sharpe measures selection out of sample instead, against the style estimated
    over the periods before (see `get_rolling_style_analysis`); the in-sample figures flatter
    the fit, as the weights are chosen to explain these very returns.

    For more information, see the following paper:

    - Sharpe, W.F. (1992). "Asset Allocation: Management Style and Performance Measurement."
      The Journal of Portfolio Management, 18(2), 7-19.

    Args:
        returns (pd.Series): The fund's returns.
        style_returns (pd.DataFrame): The returns of the style indices, a column each.
        periods_per_year (int, optional): The returns per year, to annualize the selection
            return and the tracking error. Defaults to 12.

    Returns:
        pd.Series: The weight of every style, the R Squared, the (annualized) Selection Return
        and the (annualized) Tracking Error. Dates on which the fund or a style has no return
        are left out; with no more complete dates than styles the weights are undetermined and
        every value is NaN.
    """
    styles = list(style_returns.columns)
    columns = [*styles, "R Squared", "Selection Return", "Tracking Error"]
    data = pd.concat([returns, style_returns], axis=1).dropna()

    if len(data) <= len(styles):
        return pd.Series(np.nan, index=columns)

    # By position, so a fund that carries the name of a style does not clash with it.
    weights = get_style_weights(data.iloc[:, 0], data.iloc[:, 1:]).to_numpy()
    y = data.iloc[:, 0].to_numpy(dtype=float)
    residual = y - data.iloc[:, 1:].to_numpy(dtype=float) @ weights
    fund_variance = np.var(y, ddof=1)

    return pd.Series(
        [
            *weights,
            (
                # A constant fund has no variance to explain; in floating point its
                # variance is a tiny positive number rather than zero.
                1 - np.var(residual, ddof=1) / fund_variance
                if np.ptp(y) > 0
                else np.nan
            ),
            residual.mean() * periods_per_year,
            residual.std(ddof=1) * np.sqrt(periods_per_year),
        ],
        index=columns,
    )


def get_rolling_style_analysis(
    returns: pd.Series,
    style_returns: pd.DataFrame,
    window: int,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Returns-based style analysis (Sharpe, 1992) over a rolling window: per date the style
    weights, R Squared, Selection Return and Tracking Error of the `window` periods up to and
    including that date (see `get_style_analysis`, the figures are in-sample within each
    window), which shows how a fund's style drifts over time.

    Next to them is Sharpe's out-of-sample measure of selection. The Out-of-Sample Selection
    Return of a date is the fund's return of that period minus the return of that period of
    the style mix estimated over the window that ends one period earlier: the return the
    manager added beyond a passive mix of the styles the manager was known to follow. Sharpe
    estimates the style over the previous 60 months. It is a return per period (not
    annualized), so its average and cumulative sum show whether the selection paid off.

    For more information, see the following paper:

    - Sharpe, W.F. (1992). "Asset Allocation: Management Style and Performance Measurement."
      The Journal of Portfolio Management, 18(2), 7-19.

    Args:
        returns (pd.Series): The fund's returns.
        style_returns (pd.DataFrame): The returns of the style indices, a column each, on the
            same dates as `returns`.
        window (int): The number of periods every estimate uses.
        periods_per_year (int, optional): The returns per year, to annualize the in-sample
            selection return and tracking error. Defaults to 12.

    Returns:
        pd.DataFrame: Per date (the last date of a window) the weight of every style, the R
        Squared, the (annualized) Selection Return, the (annualized) Tracking Error and the
        Out-of-Sample Selection Return. A window with no more complete dates than styles is
        NaN, as is the Out-of-Sample Selection Return of the first window, which has no
        earlier estimate.
    """
    styles = list(style_returns.columns)
    columns = [
        *styles,
        "R Squared",
        "Selection Return",
        "Tracking Error",
        "Out-of-Sample Selection Return",
    ]
    # By position from here on, so a fund that carries the name of a style does not clash.
    data = pd.concat([returns, style_returns], axis=1)
    fund_returns, style_data = data.iloc[:, 0], data.iloc[:, 1:]

    if len(data) < window:
        return pd.DataFrame(columns=columns, dtype=float)

    result = pd.DataFrame(
        [
            get_style_analysis(
                fund_returns.iloc[end - window : end],
                style_data.iloc[end - window : end],
                periods_per_year,
            ).to_numpy()
            for end in range(window, len(data) + 1)
        ],
        index=data.index[window - 1 :],
        columns=columns[:-1],
        dtype=float,
    )

    # The weights of the previous window applied to this period's style returns. A style
    # without a return leaves the period's figure unknown rather than treating it as zero.
    previous_weights = result.iloc[:, : len(styles)].shift(1).to_numpy()
    style_mix_return = np.sum(
        style_data.iloc[window - 1 :].to_numpy(dtype=float) * previous_weights, axis=1
    )
    result["Out-of-Sample Selection Return"] = (
        fund_returns.iloc[window - 1 :].to_numpy(dtype=float) - style_mix_return
    )

    return result


def get_geometric_excess_return(
    returns: pd.Series | pd.DataFrame, risk_free_rate: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the geometric excess return, the return relative to the risk-free return of the
    same period instead of minus it, as Morningstar defines it for its risk-adjusted return.

    The formula is as follows:

        - Geometric Excess Return = (1 + Return) / (1 + Risk Free Rate) - 1

    Args:
        returns (pd.Series | pd.DataFrame): The returns per period, as decimals.
        risk_free_rate (pd.Series): The risk-free return of the same periods, as decimals
            per period (not annualized).

    Returns:
        pd.Series | pd.DataFrame: The geometric excess returns, with the index of the returns.
    """
    return (1 + returns).div(1 + risk_free_rate, axis=0) - 1


def _morningstar_risk_adjusted_return(
    excess_returns: np.ndarray, gamma: float, periods_per_year: int
) -> float:
    """
    The Morningstar Risk-Adjusted Return of one series of geometric excess returns, without
    its missing values; NaN without any return.
    """
    excess_returns = excess_returns[~np.isnan(excess_returns)]
    if excess_returns.size == 0:
        return np.nan

    if gamma == 0:
        # The limit of the formula for a gamma of zero: the annualized geometric mean.
        return float(
            np.prod(1 + excess_returns) ** (periods_per_year / excess_returns.size) - 1
        )

    return float(
        np.mean((1 + excess_returns) ** -gamma) ** (-periods_per_year / gamma) - 1
    )


def get_morningstar_risk_adjusted_return(
    excess_returns: pd.Series | pd.DataFrame,
    gamma: float = 2.0,
    periods_per_year: int = 12,
) -> float | pd.Series:
    """
    Calculate the Morningstar Risk-Adjusted Return (MRAR), the annualized certainty
    equivalent of a series of geometric excess returns for an investor with risk aversion
    gamma. It is the return a risk-averse investor would accept with certainty instead of the
    uncertain returns: the more the returns vary, the further it falls below their geometric
    mean. Morningstar ranks funds within their category on it, with a gamma of 2, for its
    star rating.

    The formula is as follows:

        - MRAR(gamma) = [1/T * SUM((1 + Geometric Excess Return t)^-gamma)]^(-12/gamma) - 1
        - MRAR(0) = [PRODUCT(1 + Geometric Excess Return t)]^(12/T) - 1

    with T the number of months. MRAR(0), the annualized geometric mean excess return, is
    Morningstar Return.

    For more information, see the following:

    - Morningstar (2023). "The Morningstar Rating for Funds." Morningstar Methodology, April 2023.
      https://s25.q4cdn.com/634000362/files/doc_downloads/1/1157231.pdf
    - Morningstar (2009). "MRAR Illustrated."
      https://www.morningstar.com/content/dam/marketing/apac/au/pdfs/Legal/MRARIllustrated_Factsheet.pdf

    Args:
        excess_returns (pd.Series | pd.DataFrame): The geometric excess returns per month (see
            get_geometric_excess_return), as decimals.
        gamma (float, optional): The risk aversion. Defaults to 2.0, Morningstar's choice.
        periods_per_year (int, optional): The number of periods in a year, to annualize.
            Defaults to 12 for monthly returns.

    Returns:
        float | pd.Series: The MRAR as a decimal, a float for a Series and one value per
        column for a DataFrame. Each column uses its own returns, without its missing values.
    """
    if gamma < 0:
        raise ValueError("The risk aversion gamma cannot be negative.")

    if isinstance(excess_returns, pd.Series):
        return _morningstar_risk_adjusted_return(
            excess_returns.to_numpy(dtype=float), gamma, periods_per_year
        )

    return excess_returns.apply(
        lambda column: _morningstar_risk_adjusted_return(
            column.to_numpy(dtype=float), gamma, periods_per_year
        )
    )


def get_rolling_morningstar_risk_adjusted_return(
    excess_returns: pd.Series | pd.DataFrame,
    window_size: int,
    gamma: float = 2.0,
    periods_per_year: int = 12,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Morningstar Risk-Adjusted Return over a rolling window, e.g. 36, 60 or 120
    months for the three, five and ten year figures Morningstar rates funds on.

    Args:
        excess_returns (pd.Series | pd.DataFrame): The geometric excess returns per month, as
            decimals.
        window_size (int): The number of periods in each window.
        gamma (float, optional): The risk aversion. Defaults to 2.0.
        periods_per_year (int, optional): The number of periods in a year. Defaults to 12.

    Returns:
        pd.Series | pd.DataFrame: The MRAR at the end of every window, NaN until a window is
        full of returns.
    """
    if gamma < 0:
        raise ValueError("The risk aversion gamma cannot be negative.")

    # A window with a missing return is not a full window of the fund's history.
    return excess_returns.rolling(window_size, min_periods=window_size).apply(
        lambda window: _morningstar_risk_adjusted_return(
            window, gamma, periods_per_year
        ),
        raw=True,
    )


def get_lo_annualization_factor(
    autocorrelations: pd.Series | np.ndarray | list[float],
    periods_per_year: int,
) -> float:
    """
    Calculate Lo's (2002) factor η(q) that turns a Sharpe ratio of one period into a Sharpe
    ratio of q periods (e.g. a year) when the returns are serially correlated.

    The formula is as follows:

        - η(q) = q / SQRT( q + 2 * SUM_{k=1}^{q-1} (q - k) * ρ_k )

    Where q is `periods_per_year` and ρ_k the autocorrelation of the returns at lag k. The
    autocorrelations beyond those passed in are taken as 0. Without autocorrelation η(q)
    equals SQRT(q), the usual annualization factor; positive autocorrelation (e.g. smoothed
    returns of illiquid assets or hedge funds) makes it smaller, negative autocorrelation
    larger. When the term under the square root is not positive, the factor is undefined
    and NaN is returned.

    For more information about the method, see the following paper:

    - Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal,
    58(4), 36-52, equation 17.

    Args:
        autocorrelations (pd.Series | np.ndarray | list[float]): The autocorrelations
        ρ_1, ρ_2, ... of the returns, starting at lag 1.
        periods_per_year (int): The number of periods q the Sharpe ratio is scaled to.

    Returns:
        float: The factor η(q).
    """
    q = int(periods_per_year)
    rho = np.asarray(autocorrelations, dtype=float)[: max(q - 1, 0)]
    k = np.arange(1, len(rho) + 1)

    variance_ratio = q + 2 * np.sum((q - k) * rho)

    return float(q / np.sqrt(variance_ratio)) if variance_ratio > 0 else np.nan


def get_lo_sharpe_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    periods_per_year: float,
    lags: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the annualized Sharpe ratio of Lo (2002), which corrects the annualization for
    the serial correlation of the returns.

    The usual annualized Sharpe ratio, SQRT(q) * SR, assumes the returns are independent.
    When returns are autocorrelated, the variance of a q-period return is no longer q times
    the variance of a one-period return, and Lo shows the q-period Sharpe ratio becomes:

        - SR(q) = η(q) * SR
        - η(q) = q / SQRT( q + 2 * SUM_{k=1}^{q-1} (q - k) * ρ_k )

    Where SR is the Sharpe ratio of one period (mean excess return divided by its sample
    standard deviation, ddof=1), q is `periods_per_year` and ρ_k the sample autocorrelation
    at lag k (Box-Jenkins estimator). The sample autocorrelations at lags close to the
    sample size rest on very few pairs, so the sum includes `lags` lags and takes the
    autocorrelations beyond them as 0. By default `lags` follows the Newey and West (1994)
    rule floor(4 * (n / 100)^(2/9)), capped at q - 1 (Lo's formula itself).

    The sample autocorrelation is biased towards a negative value: for independent returns
    its expectation is −(n − k) / (n · (n − 1)) (Moran, 1948), since the deviations from the
    sample mean sum to zero. Weighted by (q − k) these small biases add up and would raise
    η(q) above SQRT(q) on short samples (by about 7% for a quarter of daily returns), so each
    ρ_k is bias corrected with ρ_k + (n − k) / (n · (n − 1)) first. Below 30 observations the
    standard error of each ρ_k (about 1 / SQRT(n)) is too large for a meaningful η(q), since
    the (q − k) weights magnify it, and NaN is returned.

    For more information about the method, see the following papers:

    - Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal,
    58(4), 36-52.
    - Moran, P.A.P. (1948). "Some Theorems on Time Series: I." Biometrika, 35(3/4),
    255-260.
    - Newey, W.K., & West, K.D. (1994). "Automatic Lag Selection in Covariance Matrix
    Estimation." Review of Economic Studies, 61(4), 631-653.

    Args:
        excess_returns (pd.Series | pd.DataFrame): The excess returns. With a "within
        period" (period, date) Multi Index, one ratio is calculated per period.
        periods_per_year (float): The number of return observations in a year (q), e.g.
        252 for daily and 12 for monthly returns. Rounded to a whole number.
        lags (int | None, optional): The number of autocorrelation lags in the sum.
        Defaults to None, the Newey and West (1994) rule.

    Returns:
        pd.Series | pd.DataFrame | float: The annualized Sharpe ratio per period (Multi
        Index), per column (DataFrame) or a single value (Series), NaN with fewer than 30
        observations.
    """

    if isinstance(excess_returns, pd.DataFrame):
        aggregate = excess_returns
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            aggregate = excess_returns.groupby(level=0)

        return aggregate.agg(
            lambda series: get_lo_sharpe_ratio(series, periods_per_year, lags)
        )

    if not isinstance(excess_returns, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    values = excess_returns.dropna().to_numpy(dtype=float)
    n = len(values)

    # With fewer observations each ρ_k has a standard error of about 1 / SQRT(n) > 0.18,
    # which the (q − k) weights magnify into an η(q) that is mostly noise.
    if n < 30:  # noqa: PLR2004
        return np.nan

    q = int(round(periods_per_year))
    lags = get_newey_west_lags(n) if lags is None else lags
    lags = max(0, min(lags, q - 1, n - 2))

    deviation = values - values.mean()
    variance = np.sum(deviation**2)
    # The sample autocorrelation of Box and Jenkins: autocovariances over n with the
    # mean of the whole sample, as in `risk_model.get_autocorrelation`, plus Moran's (1948)
    # correction of its expectation −(n − k) / (n · (n − 1)) under independence, which
    # (q − k)-weighted would otherwise push η(q) above SQRT(q) on independent returns.
    rho = [
        np.sum(deviation[:-lag] * deviation[lag:]) / variance
        + (n - lag) / (n * (n - 1))
        for lag in range(1, lags + 1)
    ]

    return (values.mean() / values.std(ddof=1)) * get_lo_annualization_factor(rho, q)


def get_rolling_lo_sharpe_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    window_size: int,
    periods_per_year: float,
    lags: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the annualized Sharpe ratio of Lo (2002) over a rolling window, see
    `get_lo_sharpe_ratio`.

    Args:
        excess_returns (pd.Series | pd.DataFrame): The excess returns.
        window_size (int): The number of observations in each window.
        periods_per_year (float): The number of return observations in a year (q).
        lags (int | None, optional): The number of autocorrelation lags in the sum.
        Defaults to None, the Newey and West (1994) rule on the window size.

    Returns:
        pd.Series | pd.DataFrame: The annualized Sharpe ratio at the end of every full window.
    """
    return excess_returns.rolling(window_size, min_periods=window_size).apply(
        lambda window: get_lo_sharpe_ratio(window, periods_per_year, lags)
    )


def get_sharpe_ratio_standard_error(
    excess_returns: pd.Series | pd.DataFrame,
    method: str = "iid",
    lags: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the standard error of the Sharpe ratio estimate, i.e. how far the measured
    Sharpe ratio may lie from the true one.

    The formulas are as follows, with SR the Sharpe ratio of one period and n the number of
    observations:

        - "iid": SE = SQRT( (1 + SR^2 / 2) / n ) (Lo, 2002, equation 9, normally and
          independently distributed returns)
        - "non_normal": SE = SQRT( (1 − γ₃·SR + ((γ₄ − 1) / 4)·SR^2) / (n − 1) ) (Mertens,
          2002; Bailey & López de Prado, 2012), independent but skewed and fat-tailed
          returns, with γ₃ the skewness and γ₄ the non-excess (raw) kurtosis. This is the
          standard error inside the Probabilistic Sharpe Ratio.
        - "gmm": SE = SQRT( ∂g/∂θ · Σ · ∂g/∂θ' / n ) (Lo, 2002, Appendix), returns that may
          also be serially correlated and heteroskedastic. θ = (μ, σ²), g(θ) = μ / σ, so
          ∂g/∂θ = (1 / σ, −μ / (2σ³)), and Σ is the Newey and West (1987) long-run covariance
          of the moment conditions (x_t − μ, (x_t − μ)² − σ²) with Bartlett weights over
          `lags` lags (σ² with 1 / n, as in GMM).

    With independent normal returns the three agree as n grows; with lags=0 the "gmm" and
    "non_normal" errors differ only by the finite-sample corrections of the moments.

    For more information about the method, see the following papers:

    - Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal,
    58(4), 36-52.
    - Mertens, E. (2002). "Comments on Variance of the IID Estimator in Lo (2002)."
    Working Paper, University of Basel.
    - Bailey, D.H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
    Journal of Risk, 15(2), 3-44.
    - Newey, W.K., & West, K.D. (1987). "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica,
    55(3), 703-708.

    Args:
        excess_returns (pd.Series | pd.DataFrame): The excess returns. With a "within
        period" (period, date) Multi Index, one standard error is calculated per period.
        method (str, optional): "iid", "non_normal" or "gmm". Defaults to "iid".
        lags (int | None, optional): The number of lags in the Newey-West covariance, only
        for method="gmm". Defaults to None, the Newey and West (1994) rule
        floor(4 * (n / 100)^(2/9)).

    Returns:
        pd.Series | pd.DataFrame | float: The standard error, on the scale of the Sharpe
        ratio of one period (not annualized).
    """
    if method not in ("iid", "non_normal", "gmm"):
        raise ValueError("Method must be iid, non_normal or gmm.")

    if isinstance(excess_returns, pd.DataFrame):
        aggregate = excess_returns
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            aggregate = excess_returns.groupby(level=0)

        return aggregate.agg(
            lambda series: get_sharpe_ratio_standard_error(series, method, lags)
        )

    if not isinstance(excess_returns, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    values = excess_returns.dropna().to_numpy(dtype=float)
    n = len(values)

    if n < 4:  # noqa: PLR2004
        return np.nan

    deviation = values - values.mean()

    if method == "iid":
        sharpe_ratio = values.mean() / values.std(ddof=1)

        return float(np.sqrt((1 + sharpe_ratio**2 / 2) / n))

    if method == "non_normal":
        sharpe_ratio = values.mean() / values.std(ddof=1)
        series = pd.Series(values)
        skewness, kurtosis = series.skew(), series.kurt() + 3

        variance = 1 - skewness * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio**2

        return float(np.sqrt(variance / (n - 1)))

    # GMM: the moment conditions of the mean and the variance, φ_t = (x_t − μ,
    # (x_t − μ)² − σ²), with their long-run covariance by Newey and West (1987).
    variance = np.mean(deviation**2)
    sharpe_ratio = values.mean() / np.sqrt(variance)
    moments = np.column_stack([deviation, deviation**2 - variance])

    lags = get_newey_west_lags(n) if lags is None else lags
    lags = max(0, min(lags, n - 1))

    covariance = moments.T @ moments / n
    for lag in range(1, lags + 1):
        autocovariance = moments[lag:].T @ moments[:-lag] / n
        covariance += (1 - lag / (lags + 1)) * (autocovariance + autocovariance.T)

    gradient = np.array([1 / np.sqrt(variance), -values.mean() / (2 * variance**1.5)])

    return float(np.sqrt(gradient @ covariance @ gradient / n))


def get_minimum_track_record_length(
    sharpe_ratio: pd.Series | pd.DataFrame | float,
    benchmark_sharpe_ratio: float,
    skewness: pd.Series | pd.DataFrame | float,
    kurtosis: pd.Series | pd.DataFrame | float,
    confidence_level: float = 0.95,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Minimum Track Record Length (MinTRL), the number of return observations
    needed before the measured Sharpe ratio is significantly above a benchmark Sharpe ratio.

    The formula is as follows:

        - MinTRL = 1 + (1 − γ₃·SR + ((γ₄ − 1) / 4)·SR^2) * (Z_α / (SR − SR*))^2

    Where SR is the observed Sharpe ratio of one period, SR* the benchmark Sharpe ratio on
    the same scale, γ₃ the skewness and γ₄ the non-excess (raw) kurtosis of the returns
    (3 for a normal distribution) and Z_α the standard normal quantile at the confidence
    level. It is the n at which the Probabilistic Sharpe Ratio reaches the confidence
    level: with n = MinTRL observations, PSR(SR*) = confidence_level. When SR does not exceed
    SR*, no track record is long enough and the result is NaN.

    For more information about the method, see the following paper:

    - Bailey, D.H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
    Journal of Risk, 15(2), 3-44, equation 11.

    Args:
        sharpe_ratio (pd.Series | pd.DataFrame | float): The observed Sharpe ratio of one
        period (not annualized).
        benchmark_sharpe_ratio (float): The benchmark Sharpe ratio SR* on the same scale.
        skewness (pd.Series | pd.DataFrame | float): The skewness of the returns.
        kurtosis (pd.Series | pd.DataFrame | float): The non-excess (raw) kurtosis of the
        returns, 3 for a normal distribution.
        confidence_level (float, optional): The confidence level. Defaults to 0.95.

    Returns:
        pd.Series | pd.DataFrame | float: The minimum number of return observations.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must be between 0 and 1.")

    z_score = stats.norm.ppf(confidence_level)
    difference = sharpe_ratio - benchmark_sharpe_ratio

    minimum_track_record_length = (
        1
        + (1 - skewness * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2)
        * (z_score / difference) ** 2
    )

    if isinstance(minimum_track_record_length, pd.Series | pd.DataFrame):
        minimum_track_record_length = minimum_track_record_length.where(difference > 0)

        # The skewness and kurtosis carry no index name, which drops the dates' name.
        if isinstance(sharpe_ratio, pd.Series | pd.DataFrame):
            minimum_track_record_length.index.name = sharpe_ratio.index.name

        return minimum_track_record_length

    return minimum_track_record_length if difference > 0 else np.nan


def get_co_moment(
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    power: int,
    method: str,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the standardized co-moment of each asset with the benchmark, the shared part of
    `get_coskewness` (power 2) and `get_cokurtosis` (power 3):

        - Co-Moment = E[a · b^power] / (SQRT(E[a^2]) · E[b^2]^(power / 2))

    Where b is the demeaned benchmark return and a the demeaned asset return
    ("standardized") or the residual of the asset's regression on the benchmark
    ("residual"), over the dates both have a return, with population moments (1 / n).

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns. With a "within period"
        (period, date) Multi Index, one value is calculated per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        power (int): The power of the benchmark deviation, 2 or 3.
        method (str): "standardized" or "residual".

    Returns:
        pd.Series | pd.DataFrame | float: The co-moment per period (Multi Index), per
        column (DataFrame) or a single value (Series).
    """
    if method not in ("standardized", "residual"):
        raise ValueError("Method must be standardized or residual.")

    if not isinstance(returns, pd.DataFrame | pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns

    benchmark = pd.DataFrame(
        np.repeat(benchmark_returns.to_numpy()[:, None], frame.shape[1], axis=1),
        index=frame.index,
        columns=frame.columns,
    )
    both = frame.notna() & benchmark.notna()
    asset, benchmark = frame.where(both), benchmark.where(both)

    within_period = frame.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    groups = frame.index.get_level_values(0) if within_period else np.zeros(len(frame))

    asset = asset - asset.groupby(groups).transform("mean")
    benchmark = benchmark - benchmark.groupby(groups).transform("mean")

    def expectation(values: pd.DataFrame) -> pd.DataFrame:
        # Population moments (1 / n) throughout, so the ratio is free of scale; three
        # shared dates are the least for a third moment.
        return values.groupby(groups).sum(min_count=3) / values.groupby(groups).count()

    if method == "residual":
        beta = expectation(asset * benchmark) / expectation(benchmark**2)
        asset = asset - benchmark * beta.reindex(groups).to_numpy()

    co_moment = expectation(asset * benchmark**power) / (
        np.sqrt(expectation(asset**2)) * expectation(benchmark**2) ** (power / 2)
    )
    co_moment = co_moment.replace([np.inf, -np.inf], np.nan)

    if within_period:
        return co_moment

    co_moment = co_moment.iloc[0]

    return co_moment if isinstance(returns, pd.DataFrame) else float(co_moment.iloc[0])


def get_coskewness(
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    method: str = "standardized",
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the co-skewness of each asset with the benchmark: how the asset's return moves
    with the squared (large, either way) moves of the market. A negative co-skewness means
    the asset does worse when the market makes large moves, which adds to the market's
    downside skewness, and investors demand a premium for holding it.

    The formulas are as follows:

        - "standardized": CoSkewness = E[(r_i − μ_i)(r_m − μ_m)^2] / (σ_i · σ_m^2)
        - "residual": β_SKD = E[ε_i · ε_m^2] / (SQRT(E[ε_i^2]) · E[ε_m^2]) (Harvey and
          Siddique, 2000, equation 11)

    Where r_i is the asset return, r_m the benchmark return, ε_m = r_m − μ_m and ε_i the
    residual of the regression r_i = α + β r_m + ε_i. The residual version removes the part
    of the asset's return the market explains, so it does not carry the market's own
    skewness through the beta. The expectations are population moments (1 / n) over the
    dates both have a return, so σ_i and σ_m use 1 / n as well and the measure does not
    depend on the scale of the returns.

    For more information about the method, see the following papers:

    - Harvey, C.R., & Siddique, A. (2000). "Conditional Skewness in Asset Pricing Tests."
    Journal of Finance, 55(3), 1263-1295.
    - Kraus, A., & Litzenberger, R.H. (1976). "Skewness Preference and the Valuation of
    Risk Assets." Journal of Finance, 31(4), 1085-1100.

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns. With a "within period"
        (period, date) Multi Index, one value is calculated per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        method (str, optional): "standardized" or "residual". Defaults to "standardized".

    Returns:
        pd.Series | pd.DataFrame | float: The co-skewness per period (Multi Index), per
        column (DataFrame) or a single value (Series).
    """
    return get_co_moment(returns, benchmark_returns, power=2, method=method)


def get_cokurtosis(
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    method: str = "standardized",
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the co-kurtosis of each asset with the benchmark: how the asset's return moves
    with the cubed moves of the market, i.e. whether it shares the market's extreme moves.
    A high co-kurtosis means the asset adds to the fat tails of a market portfolio, which
    Dittmar (2002) shows investors with a preference against kurtosis demand a premium for.

    The formulas are as follows:

        - "standardized": CoKurtosis = E[(r_i − μ_i)(r_m − μ_m)^3] / (σ_i · σ_m^3)
        - "residual": E[ε_i · ε_m^3] / (SQRT(E[ε_i^2]) · E[ε_m^2]^(3/2))

    Where r_i is the asset return, r_m the benchmark return, ε_m = r_m − μ_m and ε_i the
    residual of the regression r_i = α + β r_m + ε_i, the residual form Harvey and Siddique
    (2000) use for co-skewness. For an asset equal to the market the standardized
    co-kurtosis is the market's (non-excess) kurtosis, 3 for normal returns. The
    expectations are population moments (1 / n) over the dates both have a return.

    For more information about the method, see the following papers:

    - Dittmar, R.F. (2002). "Nonlinear Pricing Kernels, Kurtosis Preference, and Evidence
    from the Cross Section of Equity Returns." Journal of Finance, 57(1), 369-403.
    - Harvey, C.R., & Siddique, A. (2000). "Conditional Skewness in Asset Pricing Tests."
    Journal of Finance, 55(3), 1263-1295.

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns. With a "within period"
        (period, date) Multi Index, one value is calculated per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        method (str, optional): "standardized" or "residual". Defaults to "standardized".

    Returns:
        pd.Series | pd.DataFrame | float: The co-kurtosis per period (Multi Index), per
        column (DataFrame) or a single value (Series).
    """
    return get_co_moment(returns, benchmark_returns, power=3, method=method)
