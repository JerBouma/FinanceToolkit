"""Performance Model"""

import io
import itertools
import time
import warnings
import zipfile
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from scipy.stats import linregress

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.performance.helpers import (
    get_newey_west_lags,
    remove_undefined_ratios,
)
from financetoolkit.risk.distribution import dispersion_model
from financetoolkit.risk.drawdown import drawdown_model
from financetoolkit.risk.risk_model import (
    apply_to_stacked_windows,
    get_autocorrelation,
    get_conditional_drawdown_at_risk,
    get_mean_absolute_deviation,
    get_move_significance,
    get_partial_moment,
    get_tail_ratio,
    get_ui,
    get_wealth_and_peak,
)
from financetoolkit.risk.value_at_risk import cvar_model, evar_model, var_model
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    validate_window,
)
from financetoolkit.utilities.requests_model import TOOLKIT_HEADERS, get_request
from financetoolkit.utilities.sources_model import record_url
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    PERIODS_PER_YEAR,
    by_own_trading_days,
    drop_non_finite,
    to_period_index,
    validate_period,
)

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# A Sortino ratio of a single observation only reports its sign, so two are required.
MIN_SORTINO_OBSERVATIONS = 2

# A regression of the returns on the benchmark needs three returns to leave a residual.
MINIMUM_REGRESSION_WINDOW = 3

# The Euler-Mascheroni constant, used by the Deflated Sharpe Ratio.
EULER_MASCHERONI_CONSTANT = 0.5772156649015329


@by_own_trading_days(compound=True)
def get_covariance(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the covariance of returns: how far every asset's returns move together with the
    benchmark's.

    A window without enough data raises NumPy's RuntimeWarning "Degrees of freedom <= 0 for
    slice". It comes from a division by zero or NaN values and has no impact on the result, so
    it is ignored.

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns, one column per asset, or a
            rolling window object of them.
        benchmark_returns (pd.Series | pd.DataFrame): The benchmark returns, aligned on the
            same index as `returns`.

    Returns:
        pd.Series | pd.DataFrame | float: The sample covariance (with n - 1 in the denominator) of each asset with
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
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate Beta: how strongly the asset's returns move with the benchmark's, the slope of the
    linear regression of the asset returns on the benchmark returns, over the whole sample or
    within every rolling window.

    The formula is as follows:

        - Beta = Covariance(Asset Returns, Benchmark Returns) / Variance(Benchmark Returns)

    Beta is estimated on the raw returns, not on returns in excess of the risk free rate.
    With a risk free rate that barely moves within a period the two slopes are nearly
    identical, and raw returns keep beta independent of the chosen risk free rate. With a
    window, the covariance and the benchmark variance are taken over the last `window`
    returns at every date.

    Shares similar logic:
    - PerformanceAnalytics (R), CAPM.beta: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L89
    - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314
    - empyrical, roll_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1058

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns, either a flat index or a
            'within period' MultiIndex (period, date) to get one beta per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which estimates one beta over all returns (per period for a 'within period'
            index).

    Returns:
        pd.Series | pd.DataFrame | float: A DataFrame of betas per period (rows) and asset
        (columns) for a 'within period' index, otherwise a Series of betas per asset for a
        DataFrame or a float for a Series. With a window, a DataFrame of the beta at the end
        of every window with time as index. Assets with fewer than two dates shared with
        the benchmark get NaN.
    """
    validate_window("window", window, 2)
    if not isinstance(returns, pd.DataFrame | pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns

    if window is not None:
        rolling_beta = pd.DataFrame(
            index=frame.index, columns=frame.columns, dtype=np.float64
        )

        for column in frame.columns:
            # The benchmark variance is taken over the same dates as the covariance, those
            # on which the asset has a return: over every date it would mix in benchmark
            # volatility from days the asset did not trade (e.g. before a later listing).
            benchmark = benchmark_returns.where(frame[column].notna())
            # A window in which the benchmark did not move (two equal returns) has no beta.
            variance = benchmark.rolling(window=window).var()
            rolling_beta.loc[:, column] = get_covariance(
                frame[column].rolling(window=window), benchmark
            ) / variance.where(variance > 0)

        return rolling_beta

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
def get_standard_deviation(
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the standard deviation of returns: how much the returns vary, over the whole
    sample or within every rolling window, on the dates the benchmark has a return as well.

    The formula is as follows:

        - Standard Deviation = SQRT(SUM((Return - Mean Return)^2) / (N - 1))

    The standard deviation is the sample one (with n - 1 in the denominator), over each
    column's own dates, so the dates only another ticker has a return on (the weekends of a
    cryptocurrency next to a stock) do not leave every window of a column incomplete.

    With `benchmark_returns`, the returns are those of the dates both the column and the
    benchmark have a return on, with the returns of the dates in between compounded into the
    next such date: the same sample as `get_beta`, so a total risk ratio (the standard
    deviation of the asset over that of the benchmark) is at least the absolute Beta.

    Args:
        returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series | None, optional): Benchmark returns whose dates the
            returns are taken on. Defaults to None, which uses every date of the column.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the standard deviation over all returns.

    Returns:
        pd.Series | pd.DataFrame | float: The standard deviation per column (a float for a
        Series), or with a window the standard deviation at the end of every window with time
        as index.
    """
    validate_window("window", window, 2)
    if benchmark_returns is not None:
        returns = returns.where(benchmark_returns.notna(), axis=0)

    if window is not None:
        return returns.rolling(window=window).std()

    return returns.std()


@by_own_trading_days(compound=True)
def get_beta_implied_return(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    period: str,
    beta_window: int,
) -> pd.DataFrame:
    """
    Calculate the Beta-implied return: how much of each period's return the market explains, and
    the Residual, the asset's own part of it.

    The formula is as follows:

        - Implied Return = Beta × Benchmark Return
        - Residual = Actual Return − Implied Return

    The beta of a period is the rolling beta of the daily returns over the `beta_window`
    trading days that end on the day before the period starts, so a period never helps
    estimate the beta that explains it. The period returns compound the daily returns,
    as `get_returns` does.

    Each Residual is also set against the residuals before it, as `get_move_significance`
    sets a return against the earlier returns: the Residual Percentile is the share of the
    earlier residuals with a smaller size (once a year of them, at least 20, is available)
    and the Residual Largest Since the date of the last residual at least as large in the
    same direction, NaN when no earlier residual was.

    Args:
        returns (pd.DataFrame): The daily asset returns, a daily PeriodIndex.
        benchmark_returns (pd.Series): The daily benchmark returns on the same index.
        period (str): The period to explain. Options are: "daily", "weekly", "monthly",
            "quarterly", "yearly".
        beta_window (int): The number of daily returns the beta is estimated on.

    Returns:
        pd.DataFrame: The periods as rows and a MultiIndex of (measure, ticker) as columns,
        with the measures Beta, Benchmark Return, Implied Return, Actual Return, Residual,
        Residual Percentile and Residual Largest Since (the date as text).
    """
    # The window ends on the day before: the shift keeps a day out of its own beta.
    daily_beta = get_beta(returns, benchmark_returns, window=beta_window).shift(1)

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
    residual = actual_return - implied_return

    # The residual against its own history, with the window get_move_significance uses by
    # default: the percentile needs a year of earlier residuals (at least 20).
    significance = {
        ticker: get_move_significance(
            residual[ticker], window=max(PERIODS_PER_YEAR[period], 20)
        )
        for ticker in residual.columns
    }

    return pd.concat(
        {
            "Beta": beta.reindex(actual_return.index),
            "Benchmark Return": benchmark,
            "Implied Return": implied_return,
            "Actual Return": actual_return,
            "Residual": residual,
            **{
                f"Residual {measure}": pd.DataFrame(
                    {
                        ticker: measures[measure]
                        for ticker, measures in significance.items()
                    }
                )
                for measure in ("Percentile", "Largest Since")
            },
        },
        axis=1,
    )


def get_residual_value(
    residual: pd.DataFrame, market_cap: pd.DataFrame, period: str
) -> pd.DataFrame:
    """
    Calculate the Residual Value: what the asset's own move of each period added to or took from
    the company's market value, the residual return times the market capitalization at the close
    before the period started.

        - Residual Value = Residual × Market Cap at the previous close

    The previous close is the last close of the period before, so for daily periods the
    close of the trading day before; the value is what the asset's own move added to or
    took from the company's market value, in the currency of the market cap.

    Args:
        residual (pd.DataFrame): The residual return per period (rows) and ticker (columns),
            with the PeriodIndex of `period`.
        market_cap (pd.DataFrame): The daily market capitalization per ticker, a daily
            PeriodIndex. NaN where it is unknown, which keeps the value NaN.
        period (str): The period of the residuals. Options are: "daily", "weekly", "monthly",
            "quarterly", "yearly".

    Returns:
        pd.DataFrame: The Residual Value per period and ticker, NaN without a market cap.
    """
    labels = to_period_index(market_cap.index)

    if period != "daily":
        labels = labels.asfreq(PERIOD_TRANSLATION[period])

    # The last market cap of every period (the latest close with known shares), carried to
    # the period after it.
    previous_market_cap = market_cap.groupby(labels).last().shift(1)

    return residual * previous_market_cap.reindex(
        index=residual.index, columns=residual.columns
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
    Calculate Beta with an estimator other than ordinary least squares: how strongly the asset's
    returns move with the benchmark's, corrected for estimation noise or for a late reaction to
    the market.

    The estimators are:

        - Blume: 0.67 x OLS Beta + 0.33, the adjusted beta of Bloomberg (Blume, 1971, 1975).
        - Vasicek: the OLS beta shrunk towards the cross-sectional mean beta of the assets,
          weighted by the precisions: (σ²_prior x β + SE² x β_prior) / (σ²_prior + SE²)
          (Vasicek, 1973). The prior is the mean and sample variance (with n - 1 in the denominator) of the OLS
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
        method (str, optional): The beta estimator. Options are: "ols" (ordinary least squares),
            "blume" (Blume), "vasicek" (Vasicek), "dimson" (Dimson), "scholes_williams"
            (Scholes-Williams), "upside" (upside beta), "downside" (downside beta). Defaults to
            "blume".
        lags (int, optional): The leads and lags of the Dimson beta. Defaults to 1.
        window_size (int | None, optional): A rolling window over a flat index instead of
            one beta per period. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The betas per period (rows) and asset (columns) for a
        'within period' index or a rolling window, otherwise a Series of betas per asset.
        Assets with fewer than three shared observations get NaN.
    """
    check_option("method", method, BETA_METHODS)

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
    risk_free_rate: pd.Series | pd.DataFrame | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Capital Asset Pricing Model (CAPM): the return an asset is expected to earn
    for its exposure to the market, the risk-free rate plus Beta times the market risk premium.

    CAPM is a financial model used to estimate the expected return on an investment, such as a
    stock or portfolio of stocks. It provides a framework for evaluating the risk and return
    trade-off of an asset or portfolio in relation to the overall market. CAPM is based on the
    following key components:

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
        risk_free_rate (pd.Series | pd.DataFrame | float): the risk free rate, or a DataFrame with
            the rate of each column of beta over the periods of its own calendar.
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
            # A risk free rate per column, over the periods of each ticker's own calendar.
            column_risk_free_rate = (
                risk_free_rate[column]
                if isinstance(risk_free_rate, pd.DataFrame)
                else risk_free_rate
            )
            capital_asset_pricing_model.loc[:, column] = column_risk_free_rate + beta[
                column
            ] * (market_returns - column_risk_free_rate)
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
    Give the URL of Ken French's daily factor file of a region, the source of the factor returns
    of the factor models.

    Args:
        region (str, optional): The region of the factors. Options are: "United States", "Developed",
            "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan", "North America".
            Defaults to "United States".
        kind (str, optional): The factor file. Options are: "five" (5 factors), "three"
            (3 factors), "momentum" (momentum factor). Defaults to "five".

    Returns:
        str: The URL of the ZIP file.

    Raises:
        ValueError: When the region is not published.
    """
    check_option("region", region, FACTOR_REGIONS)
    return f"{KEN_FRENCH_URL}{FACTOR_REGIONS[region][kind]}"


def read_factor_file(zip_data: bytes) -> pd.DataFrame:
    """
    Read one of Ken French's daily factor files into daily factor returns as decimals.

    The header is the first line that starts with a comma (the number of description lines above
    it differs per file), the dates are YYYYMMDD (padded with spaces in the regional files) and
    the values are in percent.

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
        record_url(url)
        return downloaded[1]

    # Dartmouth answers a request that presents itself as a browser without being one with
    # a 404, so the Finance Toolkit names itself.
    content = get_request(url, timeout=10, extra_headers=TOOLKIT_HEADERS).content
    _factor_files[url] = (time.monotonic(), content)

    return content


def obtain_fama_and_french_dataset(
    fama_and_french_url: str | None = None, region: str = "United States"
):
    """
    Obtain the Fama and French 5 Factor Model dataset: the daily returns of the market, size,
    value, investment and profitability factors and the risk-free rate.

    It is a dataset that contains the excess returns of the 5 factors that are used in the Fama
    and French 5 Factor Model. The factors are:

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
        fama_and_french_url (str | None, optional): the URL of the ZIP file that contains the dataset.
        Defaults to None, which uses the 5 factor file of the region.
        region (str, optional): The region whose factors to use. Options are: "United States", "Developed",
            "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan", "North America".
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
    Obtain the Fama and French 3 Factor Model dataset: the daily returns of the market, size and
    value factors and the risk-free rate.

    The factors are:

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
        fama_and_french_url (str | None, optional): the URL of the ZIP file that contains the dataset.
        Defaults to None, which uses the 3 factor file of the region.
        region (str, optional): The region whose factors to use. Options are: "United States", "Developed",
            "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan", "North America".
        Defaults to "United States".

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
    Obtain the Carhart Momentum (MOM) factor dataset: the daily return of stocks with high prior
    returns over stocks with low prior returns.

    It extends the Fama and French 3 Factor Model into the Carhart 4 Factor Model:

        - Momentum (MOM): Stocks with high prior returns (winners) tend to keep outperforming
        stocks with low prior returns (losers) over the medium term.

    The dataset is available on the website of Dartmouth College and is updated on a monthly
    basis. The dataset is packaged in a ZIP file, so it needs to be extracted first.

    Args:
        momentum_url (str | None, optional): the URL of the ZIP file that contains the dataset.
        Defaults to None, which uses the momentum file of the region.
        region (str, optional): The region whose factor to use. Options are: "United States", "Developed",
            "Developed ex US", "Europe", "Japan", "Asia Pacific ex Japan", "North America".
        Defaults to "United States".

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
    Calculate the correlation of every factor with the asset's excess return: how closely the
    asset moves with each factor.

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
        excess_return (pd.Series): the excess return of the asset.

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
    Calculate the Fama and French 5 Factor Model regression: how much of the asset's excess
    return the market, size, value, investment and profitability factors explain, and the
    intercept left over.

    The model is an extension of the CAPM model. It adds four additional factors to the
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
    Calculate the regression of the asset's excess return on a single factor: how much of the
    return one factor of the Fama and French model explains, with its slope and intercept.

    The Fama and French 5 Factor Model is an extension of the CAPM model. It adds four
    additional factors to the regression analysis to better describe asset returns:

        - Size (SMB): Small companies tend to outperform large companies.
        - Value (HML): Value stocks tend to outperform growth stocks.
        - Investment (CMA): Companies that invest conservatively tend to outperform companies that invest aggressively.
        - Profitability (RMW): Companies with high operating profitability tend to outperform companies with low
        operating profitability.

    This functionality performs the regression analysis and returns the regression parameters for a given return
    and factor series.

    The formula is as follows:

        - Excess Return = Intercept + Slope * Factor Value + Residuals

    The factor is the independent (predictor) variable and the excess return the dependent
    (response) variable. OLS is not symmetric in the two, so the reverse regression, the
    factor as a function of the excess return, would give a different slope and intercept
    and the wrong residuals for this model.

    Args:
        excess_returns (pd.Series): the excess returns.
        factor (pd.Series): the factor series.

    Returns:
        dict: the regression results.
        pd.Series: the residuals.

    Notes:
    - Observations for which either the excess return or the factor is NaN are dropped
    (listwise deletion), mirroring `get_fama_and_french_model_multi`. A linear regression
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
            "P-Value": np.nan,
            "Standard Error": np.nan,
        }
        return regression_results, excess_returns * np.nan

    result = linregress(factor, excess_returns)

    regression_results = {
        "Intercept": result.intercept,
        "Slope": result.slope,
        "R Squared": result.rvalue**2,
        "P-Value": result.pvalue,
        "Standard Error": result.stderr,
    }

    residuals = excess_returns - (result.slope * factor + result.intercept)

    return regression_results, residuals


@by_own_trading_days(compound=True)
def get_alpha(
    asset_returns: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the (arithmetic) Alpha: how much the asset's return exceeded the benchmark's
    return, per observation or on average within every rolling window.

    This is the simple, model-free notion of Alpha (sometimes called "active return"):
    it does not risk-adjust for market exposure at all. For the risk-adjusted version
    that nets out the return attributable to Beta, see `get_jensens_alpha`.

    The formula is as follows:

        - Alpha = Asset Return - Benchmark Return
        - Rolling Alpha = Mean(Asset Return - Benchmark Return) over the last window of returns

    Args:
        asset_returns (pd.Series | float): The asset's or portfolio's return.
        benchmark_returns (pd.Series | float): The benchmark's return over the same period.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which gives the difference of every return.

    Returns:
        pd.Series | pd.DataFrame: Alpha values with time as index and assets as columns.
    """
    validate_window("window", window, 2)
    if window is not None:
        return (
            asset_returns.sub(benchmark_returns, axis=0).rolling(window=window).mean()
        )

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


def get_jensens_alpha(
    asset_returns: pd.Series | pd.DataFrame | float,
    risk_free_rate: pd.Series | pd.DataFrame | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate Jensen's Alpha: the return earned above what the CAPM predicts for the asset's
    Beta (systematic risk exposure).

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

    Shares similar logic:
    - PerformanceAnalytics (R), CAPM.jensenAlpha: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.jensenAlpha.R#L34
    - empyrical, alpha: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1154

    Args:
        asset_returns (pd.Series | float): The asset's or portfolio's return.
        risk_free_rate (pd.Series | pd.DataFrame | float): The risk-free rate over the same period,
        or a DataFrame with the rate of each asset over the periods of its own calendar.
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
            column_risk_free_rate = (
                risk_free_rate[column]
                if isinstance(risk_free_rate, pd.DataFrame)
                else risk_free_rate
            )
            jensens_alpha.loc[:, column] = asset_returns[column] - (
                column_risk_free_rate
                + beta[column] * (benchmark_returns - column_risk_free_rate)
            )
    elif isinstance(beta, (pd.Series | float)) and isinstance(
        asset_returns, (pd.Series | float)
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
    risk_free_rate: pd.Series | pd.DataFrame | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_standard_deviation: pd.Series | float | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Treynor ratio: the excess return earned per unit of systematic (market) risk,
    as measured by Beta.

    Unlike the Sharpe ratio, which divides by total volatility (standard deviation),
    the Treynor ratio divides by Beta -- appropriate for an investor who holds this
    asset as one position within an already-diversified portfolio, and therefore only
    cares about compensation for market (non-diversifiable) risk, not idiosyncratic risk.

    The formula is as follows:

        - Treynor Ratio = (Asset Return - Risk-Free Rate) / Beta

    For more information about the method, see the following paper:

    - Treynor, J.L. (1965). "How to Rate Management of Investment Funds." Harvard
    Business Review, 43(1), 63-75.

    With `benchmark_standard_deviation` the denominator is the systematic risk, Beta times the
    standard deviation of the benchmark, which gives the modified Treynor ratio of Bacon
    (2008). It is expressed per unit of return volatility rather than per unit of Beta:

        - Modified Treynor Ratio = (Asset Return - Risk-Free Rate) / (Beta * Benchmark Standard Deviation)

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, p. 77.

    A beta of 0 (or of rounding noise) leaves the ratio undefined: NaN, not infinite.

    Referenced against:
    - PerformanceAnalytics (R), TreynorRatio (modified): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/TreynorRatio.R#L47

    Shares similar logic:
    - quantstats, treynor_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1469

    Args:
        asset_returns (pd.Series | float): The asset's or portfolio's return.
        risk_free_rate (pd.Series | pd.DataFrame | float): The risk-free rate over the same period,
        or a DataFrame with the rate of each asset over the periods of its own calendar.
        beta (pd.Series | pd.DataFrame | float): The asset's or portfolio's Beta
        (systematic risk), see `get_beta`.
        benchmark_standard_deviation (pd.Series | float | None, optional): The standard deviation
        of the benchmark over the same periods as `asset_returns` (one value per period for a
        DataFrame). When given, the modified Treynor ratio is returned. Defaults to None.

    Returns:
        pd.Series: A Series of Treynor ratios with time as index and assets as columns.
    """
    if benchmark_standard_deviation is not None:
        # The systematic risk, Beta times the benchmark volatility, replaces Beta.
        beta = (
            beta.mul(benchmark_standard_deviation, axis=0)
            if isinstance(beta, pd.DataFrame)
            else beta * benchmark_standard_deviation
        )

    if isinstance(beta, pd.DataFrame) and isinstance(asset_returns, pd.DataFrame):
        treynor_ratio = pd.DataFrame(
            index=beta.index, columns=beta.columns, dtype=np.float64
        )
        for column in treynor_ratio.columns:
            column_risk_free_rate = (
                risk_free_rate[column]
                if isinstance(risk_free_rate, pd.DataFrame)
                else risk_free_rate
            )
            treynor_ratio.loc[:, column] = (
                asset_returns[column] - column_risk_free_rate
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

    # A beta of 0 (or of rounding noise) leaves the ratio undefined.
    return remove_undefined_ratios(treynor_ratio)


@by_own_trading_days(rolling_only=True)
def get_sharpe_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Sharpe ratio: the mean excess return per unit of total (upside and downside)
    volatility, over the whole sample or within every rolling window.

    The formula is as follows:

        - Sharpe Ratio = Mean(Excess Return) / Standard Deviation(Excess Return)

    Where "Excess Return" is the return with the risk-free rate already subtracted (see
    `get_excess_return`). When `excess_returns` has a "within period" (period, date) Multi
    Index, the mean and standard deviation are computed from the daily observations within
    each period, producing one Sharpe ratio per period; otherwise both are computed once
    over the entire series (or once per column, for a plain DataFrame), producing a single
    ratio (per column). With a window, both are computed over the last `window`
    observations at every date.

    Returns without variation (a standard deviation of 0, or of rounding noise) have no
    Sharpe ratio and give NaN rather than an infinite value (see
    `financetoolkit.performance.helpers.remove_undefined_ratios`).

    For more information about the method, see the following papers:

    - Sharpe, W.F. (1966). "Mutual Fund Performance." Journal of Business, 39(1), 119-138.
    - Sharpe, W.F. (1994). "The Sharpe Ratio." Journal of Portfolio Management, 21(1), 49-58.

    Referenced against:
    - quantstats, smart_sharpe (with get_autocorrelation_penalty): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1000

    Shares similar logic:
    - PerformanceAnalytics (R), SharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SharpeRatio.R#L80
    - empyrical, sharpe_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L652
    - quantstats, sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L938
    - quantstats, rolling_sharpe: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1030

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        the risk-free rate subtracted.
        window (int | None, optional): The number of observations of a rolling window, at least 2,
            in observations of `excess_returns` (e.g. 12 for a year of monthly returns). Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame: Sharpe Ratio values. If `excess_returns` has a "within
        period" Multi Index, one value is returned per period (with time as index), for a
        Series as well as a DataFrame; otherwise a single value is returned per column (or
        one overall, for a Series). With a window, the ratio at the end of every window with
        time as index.
    """
    validate_window("window", window, 2)
    if window is not None:
        return remove_undefined_ratios(
            excess_returns.rolling(window=window).mean()
            / excess_returns.rolling(window=window).std()
        )

    if isinstance(excess_returns, pd.Series | pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            return remove_undefined_ratios(
                excess_returns.groupby(level=0).apply(
                    lambda period: period.mean() / period.std()
                )
            )

        return remove_undefined_ratios(excess_returns.mean() / excess_returns.std())

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_probabilistic_sharpe_ratio(
    sharpe_ratio: pd.Series | pd.DataFrame | float,
    benchmark_sharpe_ratio: pd.Series | pd.DataFrame | float,
    skewness: pd.Series | pd.DataFrame | float,
    kurtosis: pd.Series | pd.DataFrame | float,
    n_observations: pd.Series | pd.DataFrame | float | int,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR): the probability that the true Sharpe ratio
    exceeds a benchmark Sharpe ratio, given the skewness, kurtosis and length of the returns.

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
    distribution has a kurtosis of 3, not 0. `risk_model.get_kurtosis` returns the excess
    (Fisher) kurtosis by default (Normal = 0); either ask it for the raw (Pearson) kurtosis
    before passing the result in here, or add 3 to an excess kurtosis value. Getting this
    convention backwards silently shifts every PSR value (via the ((γ₄ − 1) / 4) term), so double
    check which convention is being fed in.

    The probability does not depend on the time scale of the ratio: annualizing SR̂, SR* and
    the standard error of SR̂ by the same SQRT(q) leaves the z-score unchanged, so SR̂ and SR*
    only need to be on the same (here: per-observation) scale.

    The same formula serves as the generic probabilistic ratio: quantstats applies it to the
    Sortino ratio and to the adjusted Sortino ratio (Sortino / SQRT(2)) as well, by passing
    that ratio as `sharpe_ratio`. The standard error was derived for the Sharpe ratio only,
    so for other ratios it is an approximation without a formal derivation.

    For more information about the method, see the following sources:

    - Bailey, D.H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
    Journal of Risk, 15(2), 3-44.

    Referenced against:
    - quantstats, probabilistic_sortino_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1404
    - quantstats, probabilistic_adjusted_sortino_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1434

    Shares similar logic:
    - quantstats, probabilistic_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1297
    - quantstats, probabilistic_sharpe_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1369
    - PerformanceAnalytics (R), ProbSharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ProbSharpeRatio.R#L47
    - RiskLabAI, probabilistic_sharpe_ratio: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/probabilistic_sharpe_ratio.py#L16

    Args:
        sharpe_ratio (pd.Series | pd.DataFrame | float): The observed (realized) Sharpe
        ratio (SR̂), computed over the same return frequency as `n_observations`.
        benchmark_sharpe_ratio (pd.Series | pd.DataFrame | float): The hypothesized or
        benchmark Sharpe ratio (SR*) to test the observed Sharpe ratio against. Often 0.
        skewness (pd.Series | pd.DataFrame | float): The skewness (γ₃) of the same
        returns used to compute `sharpe_ratio`.
        kurtosis (pd.Series | pd.DataFrame | float): The non-excess (raw) kurtosis (γ₄)
        of the same returns used to compute `sharpe_ratio`. See the note above — a
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
    Calculate the Deflated Sharpe Ratio (DSR): the probability that the true Sharpe ratio
    exceeds the best Sharpe ratio the number of strategies tried would give by chance.

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

    Shares similar logic:
    - ffn, calc_deflated_sharpe_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3077
    - ffn, calc_expected_max_sharpe: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3048

    Args:
        sharpe_ratio (pd.Series | pd.DataFrame | float): The observed (realized) Sharpe
        ratio (SR̂) being tested, i.e. the one actually reported after the trials.
        sharpe_ratio_variance (pd.Series | pd.DataFrame | float): The variance of the
        Sharpe ratios observed across the `n_trials` trials. When the literal N distinct
        strategy trials are not available, a reasonable proxy is the variance of a
        rolling Sharpe ratio series computed over the same returns (see
        `get_sharpe_ratio` with a window), used as an approximation of how dispersed the
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


@by_own_trading_days(rolling_only=True)
def get_sortino_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Sortino ratio: the mean excess return per unit of downside deviation, so only
    volatility below the target counts as risk, over the whole sample or within every rolling
    window.

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
    as the standard deviation of the negative excess returns alone: that alternative would divide by
    (number of negative observations - 1) and measure the dispersion of the losses
    around *their own mean*, rather than their root-mean-square distance below the MAR
    of 0 -- a materially different (and non-standard) quantity that understates how
    volatile the downside really is whenever there are few, large losses. With a window,
    N is the number of observations in the window.

    For more information about the method, see the following paper:

    - Sortino, F.A., & Price, L.N. (1994). "Performance Measurement in a Downside Risk
    Framework." Journal of Investing, 3(3), 59-64.

    A Sortino ratio needs at least two observations: with a single one the mean and the
    downside deviation are the same number, so the ratio would only report its sign (-1)
    or be infinite. Such periods (e.g. a period that holds only one trading day of the
    date range) and windows of a single observation are returned as NaN.

    Without a return below 0 the downside deviation is 0 and the ratio is undefined: NaN, not
    infinite.

    Referenced against:
    - quantstats, smart_sortino (with get_autocorrelation_penalty): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1158
    - quantstats, adjusted_sortino: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1258

    Shares similar logic:
    - PerformanceAnalytics (R), SortinoRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SortinoRatio.R#L48
    - empyrical, sortino_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L727
    - quantstats, sortino: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1085
    - quantstats, rolling_sortino: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1188

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with risk-free
        rate already subtracted.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series: A Series of Sortino ratios with time as index and assets as columns.
    """
    validate_window("window", window, 2)
    if window is not None:
        # A window of a single observation only yields its sign, see above.
        if window < MIN_SORTINO_OBSERVATIONS:
            return excess_returns * np.nan

        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_downside_deviation = np.sqrt(
            (excess_returns.clip(upper=0) ** 2).rolling(window=window).mean()
        )

        return remove_undefined_ratios(rolling_mean / rolling_downside_deviation)

    def _sortino(returns: pd.Series | pd.DataFrame):
        downside_deviation = np.sqrt((returns.clip(upper=0) ** 2).mean())
        # Without a loss the downside deviation is 0 and the ratio undefined.
        ratio = remove_undefined_ratios(returns.mean() / downside_deviation)
        # A single observation only yields its own sign (see above).
        return (
            ratio.where(returns.count() >= MIN_SORTINO_OBSERVATIONS)
            if isinstance(ratio, pd.Series)
            else (ratio if returns.count() >= MIN_SORTINO_OBSERVATIONS else np.nan)
        )

    if isinstance(excess_returns, pd.Series | pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # The mean, downside deviation and count per period of all assets at once.
            periods = excess_returns.groupby(level=0)
            downside_deviation = np.sqrt(
                (excess_returns.clip(upper=0) ** 2).groupby(level=0).mean()
            )
            return remove_undefined_ratios(periods.mean() / downside_deviation).where(
                periods.count() >= MIN_SORTINO_OBSERVATIONS
            )

        return _sortino(excess_returns)

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


@by_own_trading_days(rolling_only=True)
def get_autocorrelation_penalty(
    returns: np.ndarray | pd.Series | pd.DataFrame,
    window: int | None = None,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculate the autocorrelation penalty: how much serially correlated returns understate the
    risk of holding the asset over several periods, the factor by which the "smart" Sharpe and
    Sortino ratios divide the plain ratio, over the whole sample or within every rolling window.

    Positively autocorrelated returns (smoothed prices of illiquid assets, momentum) make the
    standard deviation of one return understate the risk of holding the asset over several
    periods, which flatters the Sharpe and Sortino ratios. The penalty scales the risk
    measure up by the variance inflation of a sum of n returns whose autocorrelation at lag k
    is ρ^k, with ρ the absolute first order autocorrelation:

        - Penalty = SQRT( 1 + 2 · Σ_{k=1}^{n−1} ((n − k) / n) · |ρ|^k )
        - Smart Ratio = Ratio / Penalty

    Where n is the number of returns (of the window) and ρ the Pearson correlation between
    each return and the next. Returns without autocorrelation give a penalty of 1. Taking the
    absolute value of ρ penalizes negative autocorrelation as well, which would in fact lower
    the variance of a sum of returns; quantstats takes it the same way. Lo (2002), on which the
    weights are based, uses the sample autocorrelation at every lag instead of powers of the
    first one (see `get_lo_sharpe_ratio`).

    The penalty, and the "smart" Sharpe and Sortino ratios that divide by it, are defined in
    quantstats (Ran Aroussi).

    For more information, see the following sources:

    - Lo, A.W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal, 58(4),
    36-52.

    Fewer than two returns, or returns without variation, have no autocorrelation and give
    NaN (quantstats returns a penalty of 1 for them).

    Referenced against:
    - quantstats, autocorr_penalty: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L875

    Args:
        returns (np.ndarray | pd.Series | pd.DataFrame): The (excess) returns. A "within
        period" (period, date) Multi Index gives one penalty per period and column.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the penalty over all returns (per period for a 'within
            period' index).

    Returns:
        float | pd.Series | pd.DataFrame: The penalty, a float for a Series or array, one per
        column for a DataFrame and one per period and column for a "within period" index. With
        a window, the penalty at the end of every full window.
    """
    validate_window("window", window, 2)
    if window is not None:
        return returns.rolling(window, min_periods=window).apply(
            get_autocorrelation_penalty, raw=True
        )

    if (
        isinstance(returns, pd.Series | pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        return returns.groupby(level=0).agg(
            lambda values: get_autocorrelation_penalty(values.to_numpy())
        )

    if isinstance(returns, pd.DataFrame):
        return returns.apply(get_autocorrelation_penalty).astype(float)

    values = np.asarray(returns, dtype=float)
    values = values[~np.isnan(values)]
    n_observations = len(values)

    if n_observations < 2 or values.max() == values.min():  # noqa: PLR2004
        return np.nan

    # Two returns give one pair, whose correlation numpy reports as a warning per window.
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        autocorrelation = abs(np.corrcoef(values[:-1], values[1:])[0, 1])

    if np.isnan(autocorrelation):
        return np.nan

    lags = np.arange(1, n_observations)

    return float(
        np.sqrt(
            1
            + 2
            * np.sum((n_observations - lags) / n_observations * autocorrelation**lags)
        )
    )


def get_annualized_return(
    returns: pd.Series | pd.DataFrame,
    observations_per_year: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate the annualized (geometric) return: the constant yearly return that compounds into
    the same growth over the number of years the returns span.

    The formula is as follows:

        - Annualized Return = PRODUCT(1 + Return) ^ (Observations per Year / N) − 1

    Where N is the number of returns of each column. The years are counted in observations
    (N / Observations per Year), as PerformanceAnalytics and empyrical do, rather than in
    calendar days as `get_compound_annual_growth_rate` does; both agree for a full history of trading days.

    Referenced against:
    - PerformanceAnalytics (R), Return.annualized: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.annualized.R#L39

    Shares similar logic:
    - empyrical, annual_return: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L408

    Args:
        returns (pd.Series | pd.DataFrame): The returns per observation, each column on its
        own history (missing returns are left out).
        observations_per_year (float | pd.Series): The number of returns in a year, such as
        252 for daily returns of a stock and 365 for those of a cryptocurrency, or one value
        per column.

    Returns:
        float | pd.Series: The annualized return, a float for a Series and one per column for
        a DataFrame. A column without returns gives NaN.
    """
    growth = (1 + returns).prod(min_count=1)

    # A column without returns has a growth of NaN, whatever the exponent.
    with np.errstate(divide="ignore"):
        return growth ** (observations_per_year / returns.count()) - 1


def get_safety_first_ratio(
    returns: pd.Series | pd.DataFrame,
    minimum_acceptable_return: float | pd.Series = 0.0,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate Roy's Safety-First Ratio: the mean return above a minimum acceptable (shortfall)
    return per unit of total volatility, over the whole sample or within every rolling window.

    The formula is as follows:

        - Safety-First Ratio = (Mean(Return) - Threshold) / Standard Deviation(Return)

    Roy (1952) chooses the portfolio that minimizes the probability of a return below the
    minimum acceptable return (the "disaster level"); for normally distributed returns that is the
    portfolio with the highest Safety-First Ratio, as the probability is N(-Ratio). With
    the risk-free rate as the threshold the ratio is the Sharpe Ratio (see
    `get_sharpe_ratio`). The standard deviation is the sample one (with n - 1 in the denominator), and the
    threshold is in the units of the returns: per observation, so a daily threshold for
    daily returns. A "within period" (period, date) Multi Index gives one ratio per period,
    and with a window each column is taken over its own trading days.

    For more information about the method, see the following paper:

    - Roy, A.D. (1952). "Safety First and the Holding of Assets." Econometrica, 20(3),
    431-449.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns.
        minimum_acceptable_return (float | pd.Series, optional): The threshold return per observation, a
        number or one per column. Defaults to 0.0.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame | float: Safety-First Ratio values, one per period with a
        "within period" Multi Index, for a Series as well as a DataFrame, and otherwise one
        per column (or one for a Series). With a window, the ratio at the end of every window
        with time as index.
    """
    validate_window("window", window, 2)
    # Subtracting a constant leaves the standard deviation as it is, so this is the
    # Sharpe ratio of the returns in excess of the threshold.
    return get_sharpe_ratio(returns - minimum_acceptable_return, window=window)


@by_own_trading_days(rolling_only=True)
def get_upside_potential_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Upside Potential Ratio: the expected gain above the minimum acceptable return
    (the first order upper partial moment) per unit of downside deviation (the square root of
    the second order lower partial moment), over the whole sample or within every rolling
    window.

    The formula is as follows:

        - Upside Potential Ratio = UPM_1 / SQRT(LPM_2)
        - UPM_1 = (1/N) * sum( max(Excess Return, 0) )
        - LPM_2 = (1/N) * sum( min(Excess Return, 0)^2 )

    Where the minimum acceptable return is already subtracted from `excess_returns` (the
    risk-free rate, as in `get_sortino_ratio`) and N is the *total* number of observations
    (of the window) in both moments, as Sortino, van der Meer & Plantinga (1999) weigh each
    outcome by its probability: observations on the other side of the minimum acceptable
    return add 0 but still count. The denominator is the downside deviation of
    `get_sortino_ratio`, so the two ratios differ only in their numerator: the Sortino Ratio
    rewards the mean excess return, which losses lower, and the Upside Potential Ratio only
    the gains.

    For more information about the method, see the following paper:

    - Sortino, F.A., van der Meer, R., & Plantinga, A. (1999). "The Dutch Triangle."
    Journal of Portfolio Management, 26(1), 50-57.

    A ratio needs at least two observations (with a single one either moment is 0), and
    without a return below the minimum acceptable return there is no downside to divide by:
    both are returned as NaN.

    Shares similar logic:
    - PerformanceAnalytics (R), UpsidePotentialRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpsidePotentialRatio.R#L72

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with the
        minimum acceptable return (the risk-free rate) subtracted.
        window (int | None, optional): The number of observations of a rolling window, at least 1. Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame | float: Upside Potential Ratio values, one per period with a
        "within period" (period, date) Multi Index, for a Series as well as a DataFrame, and
        otherwise one per column (or one for a Series). With a window, the ratio at the end of
        every window with time as index.
    """
    validate_window("window", window, 1)
    if window is not None:
        # A window of a single observation has one of the two moments at 0.
        if window < MIN_SORTINO_OBSERVATIONS:
            return excess_returns * np.nan

        upside_potential = excess_returns.clip(lower=0).rolling(window=window).mean()
        downside_deviation = np.sqrt(
            (excess_returns.clip(upper=0) ** 2).rolling(window=window).mean()
        )

        return remove_undefined_ratios(upside_potential / downside_deviation)

    def _upside_potential(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
        frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
        upside_potential = frame.clip(lower=0).mean()
        downside_deviation = np.sqrt((frame.clip(upper=0) ** 2).mean())
        ratio = remove_undefined_ratios(upside_potential / downside_deviation).where(
            frame.count() >= MIN_SORTINO_OBSERVATIONS
        )

        return ratio.iloc[0] if isinstance(returns, pd.Series) else ratio

    if isinstance(excess_returns, pd.Series | pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            return excess_returns.groupby(level=0).apply(_upside_potential)

        return _upside_potential(excess_returns)

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


def get_ulcer_performance_index(
    excess_returns: pd.Series | pd.DataFrame, ulcer_index: pd.Series | pd.DataFrame
) -> pd.Series:
    """
    Calculate the Ulcer Performance Index (UPI): the mean excess return per unit of Ulcer Index,
    a depth- and duration-weighted measure of drawdown pain.

    The formula is as follows:

        - Ulcer Performance Index = Excess Return / Ulcer Index

    Also known as: Martin Ratio.

    For more information about the method, see the following source:

    - Martin, P.G., & McCann, B.B. (1989). "The Investor's Guide to Fidelity Funds:
    Winning Strategies for Mutual Fund Investors." John Wiley & Sons. (Peter Martin's
    Ulcer Index, on which this ratio is based, was originally introduced in 1987.)

    Without a drawdown the Ulcer Index is 0 and the ratio is undefined (NaN).

    Shares similar logic:
    - PerformanceAnalytics (R), MartinRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MartinRatio.R#L33
    - quantstats, ulcer_performance_index: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1855
    - ffn, to_ulcer_performance_index: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L2989

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series of returns with risk-free rate subtracted.
        ulcer_index (pd.Series | pd.DataFrame): The corresponding Ulcer Index values (see
        `financetoolkit.risk.risk_model.get_ui`).

    Returns:
        pd.Series: A Series of Ulcer Performance Index values with time as index and assets as columns.
    """
    return remove_undefined_ratios(excess_returns / ulcer_index).dropna(how="all")


@by_own_trading_days(compound=True, rolling_only=True)
def get_m2_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame,
    asset_standard_deviation: pd.Series | pd.DataFrame | None = None,
    benchmark_standard_deviation: pd.Series | float | None = None,
    benchmark_returns: pd.Series | None = None,
    window: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the M2 Ratio (Modigliani-Modigliani Measure): the return the asset would have
    earned with the benchmark's volatility, the Sharpe ratio expressed as a return, per period
    or within every rolling window.

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
    than M2 (a return, expressed in the same units as `asset_returns`). With a window, the
    returns and the risk-free rate are the means over the last `window` periods and both
    standard deviations those of the same windows of `asset_returns` and `benchmark_returns`.

    For more information about the method, see the following paper:

    - Modigliani, F., & Modigliani, L. (1997). "Risk-Adjusted Performance." Journal of
    Portfolio Management, 23(2), 45-54.

    Returns without variation have no Sharpe ratio, which leaves the M2 Ratio NaN.

    Referenced against:
    - PerformanceAnalytics (R), MSquared: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MSquared.R#L33

    Shares similar logic:
    - PerformanceAnalytics (R), Modigliani: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Modigliani.R#L36

    Args:
        asset_returns (pd.Series | pd.DataFrame): A Series or DataFrame of period returns.
        risk_free_rate (pd.Series | pd.DataFrame): The risk-free rate, aligned to the same period as
        the returns, or a DataFrame with the rate of each asset over the periods of its own calendar.
        asset_standard_deviation (pd.Series | pd.DataFrame | None, optional): The asset's (or
        portfolio's) standard deviation, over the same window as `asset_returns`. Required
        without a window, not used with one. Defaults to None.
        benchmark_standard_deviation (pd.Series | float | None, optional): The benchmark's standard
        deviation, over the same window as `asset_returns`. Required without a window, not used
        with one. Defaults to None.
        benchmark_returns (pd.Series | None, optional): The benchmark returns, aligned to the same
        period as the returns, from which the rolling benchmark standard deviation is derived.
        Required with a window, not used without one. Defaults to None.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the ratio of every period from the given standard
            deviations.

    Returns:
        pd.Series: A Series of M2 ratios with time as index and assets as columns.
    """
    validate_window("window", window, 2)
    if window is not None:
        if benchmark_returns is None:
            raise ValueError("A rolling M2 Ratio needs the benchmark_returns.")

        excess_returns = asset_returns.sub(risk_free_rate, axis=0)

        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = asset_returns.rolling(window=window).std()
        rolling_sharpe_ratio = remove_undefined_ratios(rolling_mean / rolling_std)

        rolling_risk_free_rate = risk_free_rate.rolling(window=window).mean()
        rolling_benchmark_std = benchmark_returns.rolling(window=window).std()

        return rolling_sharpe_ratio.mul(rolling_benchmark_std, axis=0).add(
            rolling_risk_free_rate, axis=0
        )

    if asset_standard_deviation is None or benchmark_standard_deviation is None:
        raise ValueError(
            "The M2 Ratio needs the asset_standard_deviation and the "
            "benchmark_standard_deviation, or a window with the benchmark_returns."
        )

    if isinstance(asset_returns, pd.DataFrame) and isinstance(
        asset_standard_deviation, pd.DataFrame
    ):
        m2_ratio = pd.DataFrame(
            index=asset_returns.index, columns=asset_returns.columns, dtype=np.float64
        )
        for column in m2_ratio.columns:
            column_risk_free_rate = (
                risk_free_rate[column]
                if isinstance(risk_free_rate, pd.DataFrame)
                else risk_free_rate
            )
            sharpe_ratio = remove_undefined_ratios(
                (asset_returns[column] - column_risk_free_rate)
                / asset_standard_deviation[column]
            )
            m2_ratio.loc[:, column] = (
                column_risk_free_rate + sharpe_ratio * benchmark_standard_deviation
            )
    elif isinstance(asset_returns, (pd.Series | float)) and isinstance(
        asset_standard_deviation, (pd.Series | float)
    ):
        sharpe_ratio = remove_undefined_ratios(
            (asset_returns - risk_free_rate) / asset_standard_deviation
        )
        m2_ratio = risk_free_rate + sharpe_ratio * benchmark_standard_deviation
    else:
        raise TypeError(
            "Expects pd.DataFrame for both Asset Returns and Asset Standard Deviations or pd.Series / Float "
            "for both Asset Returns and Asset Standed Deviations."
        )

    return m2_ratio


def get_m2_excess_return(
    m2_ratio: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series | float,
    method: str = "arithmetic",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the M2 excess return: the return the risk-matched portfolio of the M2 measure (see
    `get_m2_ratio`) earned beyond the benchmark.

    The formula is as follows:

        - Arithmetic M2 Excess Return = M2 − Benchmark Return
        - Geometric M2 Excess Return = (1 + M2) / (1 + Benchmark Return) − 1

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, p. 68.

    Referenced against:
    - PerformanceAnalytics (R), MSquaredExcess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MSquaredExcess.R#L35

    Args:
        m2_ratio (pd.Series | pd.DataFrame): The M2 measure per period (columns per asset).
        benchmark_returns (pd.Series | float): The benchmark return over the same periods.
        method (str, optional): The kind of excess return. Options are: "arithmetic"
            (arithmetic M2 excess return), "geometric" (geometric M2 excess return). Defaults to
            "arithmetic".

    Returns:
        pd.Series | pd.DataFrame: The M2 excess return.

    Raises:
        ValueError: If `method` is not "arithmetic" or "geometric".
    """
    if method == "arithmetic":
        return (
            m2_ratio.sub(benchmark_returns, axis=0)
            if isinstance(m2_ratio, pd.DataFrame)
            else m2_ratio - benchmark_returns
        )
    if method == "geometric":
        return (
            (1 + m2_ratio).div(1 + benchmark_returns, axis=0)
            if isinstance(m2_ratio, pd.DataFrame)
            else (1 + m2_ratio) / (1 + benchmark_returns)
        ) - 1

    raise InvalidArgumentError(
        f"method must be one of 'arithmetic' or 'geometric', "
        f"not {method!r}. Please pass one of these as method."
    )


@by_own_trading_days(compound=True)
def get_tracking_error(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Tracking Error: how far the asset's return strays from the benchmark's, the
    standard deviation of their difference (the "active return"), over the whole sample or
    within every rolling window.

    The formula is as follows:

        - Tracking Error = Standard Deviation(Asset Return - Benchmark Return)

    Shares similar logic:
    - PerformanceAnalytics (R), TrackingError: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/TrackingError.R#L32

    Args:
        asset_returns (pd.Series | pd.DataFrame): A Series or DataFrame of asset returns.
        benchmark_returns (pd.Series): The benchmark's returns over the same period.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the Tracking Error over all returns (per period for a
            'within period' index).

    Returns:
        pd.Series | pd.DataFrame | float: Tracking Error values. With a "within period"
        (period, date) MultiIndex one value per period (rows) and asset (columns), for a
        plain DataFrame one value per asset (a Series) and for a Series a float. With a
        window, the Tracking Error at the end of every window with time as index.
    """
    validate_window("window", window, 2)
    if window is not None:
        return asset_returns.sub(benchmark_returns, axis=0).rolling(window=window).std()

    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # One standard deviation per period of all assets at once, rather than one per period.  # noqa: E501
            return (
                asset_returns.sub(benchmark_returns, axis=0)
                .reindex(asset_returns.index)
                .groupby(level=0)
                .std()
            )

        # One value per asset over the whole series.
        return asset_returns.sub(benchmark_returns, axis=0).std()

    if isinstance(asset_returns, (pd.Series | float)):
        tracking_error = (asset_returns - benchmark_returns).std()

    return tracking_error


@by_own_trading_days(compound=True)
def get_information_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Information Ratio: the mean active return (asset return minus benchmark
    return) per unit of Tracking Error, over the whole sample or within every rolling window.

    The formula is as follows:

        - Information Ratio = Mean(Asset Return - Benchmark Return) / Tracking Error

    Where Tracking Error is the standard deviation of that same difference (see
    `get_tracking_error`). The Information Ratio is the Sharpe ratio of "active"
    (benchmark-relative) returns, and is closely tied to Grinold's "Fundamental Law of
    Active Management" as a way of scoring active managers on a risk-adjusted basis.

    For more information about the method, see the following paper:

    - Grinold, R.C. (1989). "The Fundamental Law of Active Management." Journal of
    Portfolio Management, 15(3), 30-37.

    Returns that track the benchmark exactly have a tracking error of 0 and no ratio (NaN).

    Shares similar logic:
    - PerformanceAnalytics (R), InformationRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/InformationRatio.R#L32
    - quantstats, information_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2884
    - ffn, calc_information_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1691

    Args:
        asset_returns (pd.Series | pd.DataFrame): A Series or DataFrame of asset returns.
        benchmark_returns (pd.Series): The benchmark's returns over the same period.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame | float: Information Ratio values. With a "within period"
        (period, date) MultiIndex one value per period (rows) and asset (columns), for a
        plain DataFrame one value per asset (a Series) and for a Series a float. With a
        window, the ratio at the end of every window with time as index.
    """
    validate_window("window", window, 2)
    if window is not None:
        difference = asset_returns.sub(benchmark_returns, axis=0)

        rolling_mean = difference.rolling(window=window).mean()
        rolling_std = difference.rolling(window=window).std()

        return remove_undefined_ratios(rolling_mean / rolling_std)

    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # The mean and standard deviation per period of all assets at once, rather than one period at a time.  # noqa: E501
            difference = (
                asset_returns.sub(benchmark_returns, axis=0)
                .reindex(asset_returns.index)
                .groupby(level=0)
            )

            return remove_undefined_ratios(difference.mean() / difference.std())

        # One value per asset over the whole series.
        difference = asset_returns.sub(benchmark_returns, axis=0)

        return remove_undefined_ratios(difference.mean() / difference.std())

    if isinstance(asset_returns, (pd.Series | float)):
        difference = asset_returns - benchmark_returns
        information_ratio = remove_undefined_ratios(
            difference.mean() / difference.std()
        )

    return information_ratio


def get_calmar_ratio(
    returns: pd.Series | pd.DataFrame, maximum_drawdown: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Calmar Ratio: the (typically annualized) return per unit of Maximum Drawdown.

    The formula is as follows:

        - Calmar Ratio = Return / |Maximum Drawdown|

    Named after "California Managed Accounts Reports," the newsletter in which Terry W.
    Young originally proposed it.

    For more information about the method, see the following source:

    - Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." Futures, 20(1), 40.

    The reference libraries divide the annualized return over the whole sample (see
    `get_annualized_return`) by the Maximum Drawdown over the whole sample; pass those to get
    their ratio. ffn counts the years in calendar days rather than in observations, which
    gives a slightly different annualized return.

    Without a drawdown the ratio is undefined (NaN).

    Shares similar logic:
    - PerformanceAnalytics (R), CalmarRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CalmarRatio.R#L50
    - empyrical, calmar_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L548
    - quantstats, calmar: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1788
    - ffn, calc_calmar_ratio: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L2945

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of period returns.
        maximum_drawdown (pd.Series | pd.DataFrame): The corresponding Maximum Drawdown values.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Calmar ratios with time as index
        and assets as columns.
    """
    return remove_undefined_ratios(returns / maximum_drawdown.abs()).dropna(how="all")


@by_own_trading_days
def get_average_largest_drawdown(
    returns: pd.Series | pd.DataFrame,
    largest_drawdowns: int = 3,
    method: str = "return",
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Average Largest Drawdown D_Lar (Bacon, 2008): the mean depth of the d deepest
    drawdowns, each measured from its peak to its trough.

    The formula is as follows:

        - D_Lar = Σ_(j=1..d) Trough_j / d

    Where Trough_1 <= Trough_2 <= ... are the depths of the drawdown episodes (runs of
    observations below the running peak, an episode that has not recovered by the end
    included), deepest first, as in `risk_model.get_drawdown_table`, and d is the number
    of largest drawdowns, or the number of episodes when there are fewer. d = 1 gives the
    Maximum Drawdown, and a d at least the number of episodes the average depth of all of
    them (Bacon's Average Drawdown).

    For more information, see the following source:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns, or of raw levels
        for level drawdowns, optionally with a (period, date) index for a value per period.
        largest_drawdowns (int, optional): The number d of deepest drawdowns averaged.
        Defaults to 3.
        method (str, optional): The kind of drawdown. Options are: "return" (percentage drawdowns
        of the compounded return series), "level" (absolute drawdowns of the raw level series,
        for a series that is not a genuine percentage return, such as one that can be zero or
        negative). Defaults to "return".

    Returns:
        pd.Series | pd.DataFrame | float: The Average Largest Drawdown, negative (0 without a
        drawdown, NaN without returns), as a float for a Series, per column for a DataFrame and
        per period and column with a (period, date) index.

    Raises:
        ValueError: If `method` is not one of "return" or "level", or `largest_drawdowns` is
        below 1.
    """
    check_option("method", method, ("return", "level"))
    if largest_drawdowns < 1:
        raise ValueError(
            f"largest_drawdowns must be at least 1, received {largest_drawdowns}."
        )

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()

            return pd.DataFrame(
                {
                    sub_period: get_average_largest_drawdown(
                        returns.loc[sub_period], largest_drawdowns, method
                    )
                    for sub_period in periods
                }
            ).T

        return pd.Series(
            {
                column: get_average_largest_drawdown(
                    returns[column], largest_drawdowns, method
                )
                for column in returns.columns
            },
            dtype=float,
        )

    values = returns.dropna()

    if values.empty:
        return np.nan

    if method == "level":
        drawdowns = values - values.cummax()
    else:
        wealth, peak = get_wealth_and_peak(values)
        drawdowns = wealth / peak - 1

    troughs = np.sort(drawdown_model.get_drawdown_troughs(drawdowns.to_numpy()))

    return float(troughs[:largest_drawdowns].mean()) if len(troughs) else 0.0


def get_sterling_ratio(
    returns: pd.Series | pd.DataFrame,
    average_largest_drawdown: pd.Series | pd.DataFrame,
    adjustment: float = 0.1,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Sterling Ratio as Bacon (2008) defines it: the return per unit of the Average
    Largest Drawdown plus a fixed adjustment.

    The formula is as follows:

        - Sterling Ratio = Return / (|D_Lar| + Adjustment)

    Where D_Lar is the Average Largest Drawdown, the mean depth of the d deepest drawdowns
    (peak to trough, see `get_average_largest_drawdown`), and the Adjustment conventionally
    10%. With d = 1, D_Lar is the Maximum Drawdown, which is the Sterling Ratio of
    PerformanceAnalytics (with the annualized return over the whole sample).
    The original form of Deane Sterling Jones, popularised by Kestner (1996), divides the
    compound annual return of the last three years by the average of the maximum drawdown of
    each of those years plus 10% instead: an average of annual maxima, whereas D_Lar averages
    the deepest drawdowns wherever they fall.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons.
    - Kestner, L.N. (1996). "Getting a Handle on True Performance." Futures, 25(1), 44-46.

    Shares similar logic:
    - PerformanceAnalytics (R), SterlingRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CalmarRatio.R#L85

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of period returns.
        average_largest_drawdown (pd.Series | pd.DataFrame): The corresponding Average Largest
        Drawdown values (or Maximum Drawdown values for d = 1).
        adjustment (float, optional): The fixed adjustment added to the drawdown,
        conventionally 0.1 (10%). Defaults to 0.1.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Sterling ratios with time as index
        and assets as columns.
    """
    return (returns / (average_largest_drawdown.abs() + adjustment)).dropna(how="all")


@by_own_trading_days
def get_burke_drawdown_measure(
    returns: pd.Series | pd.DataFrame,
    method: str = "return",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Burke Drawdown Measure: how deep and how long the cumulative returns stayed
    below their peak, the square root of the sum of the squared pointwise drawdowns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns, or of raw levels
        for level drawdowns.
        method (str, optional): The kind of drawdown. Options are: "return" (percentage drawdowns
        of the compounded return series), "level" (absolute drawdowns of the raw level series,
        for a series that is not a genuine percentage return, such as one that can be zero or
        negative; the result is then in squared input units, not a percentage). Defaults to
        "return".

    Returns:
        pd.Series | pd.DataFrame: Burke Drawdown Measure values as float if returns is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `method` is not one of "return" or "level".
    """
    check_option("method", method, ("return", "level"))

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
    n_observations: pd.Series | pd.DataFrame | float | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Burke Ratio: the excess return per unit of drawdown risk, the root of the sum
    of the squared drawdowns.

    The formula is as follows:

        - Burke Ratio = Excess Return / SQRT(SUM(Drawdown^2))
        - Modified Burke Ratio = Excess Return / SQRT(SUM(Drawdown^2) / N) = Burke Ratio * SQRT(N)

    Where N is the number of observations the drawdowns are measured over. The modified
    ratio divides by the root mean square drawdown instead of the root of the sum, so it no
    longer shrinks as the history gets longer.

    For more information, see the following sources:

    - Burke, G. (1994). "A Sharper Sharpe Ratio." Futures, 23(3), 56.
    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, pp. 90-91.

    Without a drawdown the ratio is undefined (NaN).

    Shares similar logic:
    - PerformanceAnalytics (R), BurkeRatio (modified): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/BurkeRatio.R#L37

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        risk-free rate subtracted.
        burke_drawdown_measure (pd.Series | pd.DataFrame): The corresponding Burke Drawdown
        Measure values.
        n_observations (pd.Series | pd.DataFrame | float | None, optional): The number of
        observations the drawdowns were measured over. When given, the modified Burke ratio is
        returned. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Burke ratios with time as index
        and assets as columns.
    """
    if n_observations is not None:
        # The root mean square drawdown instead of the root of the sum of squares.
        burke_drawdown_measure = burke_drawdown_measure / np.sqrt(n_observations)

    return remove_undefined_ratios(excess_returns / burke_drawdown_measure).dropna(
        how="all"
    )


def get_pain_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    pain_index: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Pain Ratio: the excess return per unit of Pain Index, the average depth of the
    drawdowns.

    The formula is as follows:

        - Pain Ratio = (Return - Risk-Free Rate) / Pain Index

    Without a drawdown the Pain Index is 0 and the ratio is undefined (NaN).

    For more information about the method, see the following source:

    - Bacon, C.R. (2013). "Practical Risk-Adjusted Performance Measurement." John Wiley &
    Sons, chapter 5.

    Shares similar logic:
    - PerformanceAnalytics (R), PainRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/PainRatio.R#L33

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with the
        risk-free rate subtracted.
        pain_index (pd.Series | pd.DataFrame): The corresponding Pain Index values (see
        `financetoolkit.risk.risk_model.get_pain_index`).

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Pain Ratios with time as index
        and assets as columns.
    """
    return (excess_returns / pain_index.where(pain_index > 0)).dropna(how="all")


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
    upside: bool | None,
    method: str = "geometric",
    periods_per_year: float | pd.Series = 252.0,
) -> pd.Series | pd.DataFrame:
    """
    Shared implementation for the Upside, Downside and overall Capture Ratios: the average
    return of the asset over the observations in which the benchmark rose (fell, or all of
    them), divided by the average return of the benchmark over those same observations.

        - geometric (Morningstar): ((PRODUCT(1 + Asset Return))^(1 / n) - 1)
          / ((PRODUCT(1 + Benchmark Return))^(1 / n) - 1)
        - annualized (empyrical): ((PRODUCT(1 + Asset Return))^(P / n) - 1)
          / ((PRODUCT(1 + Benchmark Return))^(P / n) - 1)
        - arithmetic (PerformanceAnalytics): SUM(Asset Return) / SUM(Benchmark Return)

    Where n is the number of observations with a positive (negative) benchmark return
    on which the asset also has a return, so that both averages cover the same dates, and
    P the number of observations in a year.

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        upside (bool | None): Whether to compute the Upside (True), Downside (False) or the
            overall (None) Capture Ratio.
        method (str, optional): "geometric", "annualized" or "arithmetic". Defaults to
            "geometric".
        periods_per_year (float | pd.Series, optional): The observations in a year of the
            annualized method, or one value per asset. Defaults to 252.

    Returns:
        pd.Series | pd.DataFrame: Capture Ratio values.
    """
    check_option("method", method, ("geometric", "annualized", "arithmetic"))

    def _capture(assets: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
        # The overall ratio takes every date, the upside (downside) one those the
        # benchmark rose (fell).
        mask = (
            benchmark.notna()
            if upside is None
            else benchmark > 0 if upside else benchmark < 0
        )
        selected = assets[mask]
        # The benchmark is averaged per asset over the dates that asset has a return,
        # so a shorter asset history is compared with the matching benchmark window.
        benchmark_per_asset = pd.DataFrame(
            np.repeat(benchmark[mask].to_numpy()[:, None], assets.shape[1], axis=1),
            index=selected.index,
            columns=assets.columns,
        ).where(selected.notna())

        if method == "arithmetic":
            # The sums cover the same dates, so their ratio is that of the means.
            return selected.mean() / benchmark_per_asset.mean()

        asset_average = _geometric_mean_return(selected)
        benchmark_average = _geometric_mean_return(benchmark_per_asset)

        if method == "annualized":
            observations = (
                periods_per_year.reindex(assets.columns)
                if isinstance(periods_per_year, pd.Series)
                else periods_per_year
            )
            return ((1 + asset_average) ** observations - 1) / (
                (1 + benchmark_average) ** observations - 1
            )

        return asset_average / benchmark_average

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
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    method: str = "geometric",
    periods_per_year: float | pd.Series = 252.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Upside Capture Ratio: how much of the benchmark's gains the asset captured,
    its average return over the benchmark's in the periods in which the benchmark rose
    (Morningstar's definition).

    The formula is as follows:

        - Upside Capture Ratio = ((PRODUCT(1 + Asset Return | Benchmark > 0))^(1 / n) - 1)
          / ((PRODUCT(1 + Benchmark Return | Benchmark > 0))^(1 / n) - 1)

    The annualized method annualizes both averages, (1 + average)^P - 1, as empyrical does; the
    arithmetic method takes the ratio of the sums of the returns, as PerformanceAnalytics'
    UpDownRatios does.

    Referenced against:
    - empyrical, up_capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1887
    - PerformanceAnalytics (R), UpDownRatios (arithmetic): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        method (str, optional): The average of the returns. Options are: "geometric" (geometric
            averages), "annualized" (annualized geometric averages), "arithmetic" (the ratio of the
            sums). Defaults to "geometric".
        periods_per_year (float | pd.Series, optional): The observations in a year of the
            annualized method, or one value per asset. Defaults to 252.

    Returns:
        pd.Series | pd.DataFrame: Upside Capture Ratio values.
    """
    return _get_capture_ratio(
        asset_returns, benchmark_returns, True, method, periods_per_year
    )


@by_own_trading_days(compound=True)
def get_downside_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    method: str = "geometric",
    periods_per_year: float | pd.Series = 252.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Downside Capture Ratio: how much of the benchmark's losses the asset took part
    in, its average return over the benchmark's in the periods in which the benchmark fell
    (Morningstar's definition).

    The formula is as follows:

        - Downside Capture Ratio = ((PRODUCT(1 + Asset Return | Benchmark < 0))^(1 / n) - 1)
          / ((PRODUCT(1 + Benchmark Return | Benchmark < 0))^(1 / n) - 1)

    The annualized method annualizes both averages, (1 + average)^P - 1, as empyrical does; the
    arithmetic method takes the ratio of the sums of the returns, as PerformanceAnalytics'
    UpDownRatios does. PerformanceAnalytics' Down Capture also counts the periods in which the benchmark
    returned exactly 0, which are left out here.

    Referenced against:
    - empyrical, down_capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1921

    Shares similar logic:
    - PerformanceAnalytics (R), UpDownRatios (arithmetic): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        method (str, optional): The average of the returns. Options are: "geometric" (geometric
            averages), "annualized" (annualized geometric averages), "arithmetic" (the ratio of the
            sums). Defaults to "geometric".
        periods_per_year (float | pd.Series, optional): The observations in a year of the
            annualized method, or one value per asset. Defaults to 252.

    Returns:
        pd.Series | pd.DataFrame: Downside Capture Ratio values.
    """
    return _get_capture_ratio(
        asset_returns, benchmark_returns, False, method, periods_per_year
    )


@by_own_trading_days(compound=True)
def get_capture_ratio(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    method: str = "geometric",
    periods_per_year: float | pd.Series = 252.0,
    window: int | None = None,
    side: str = "all",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the (overall) Capture Ratio: the asset's average return over the benchmark's on
    every observation both have, the measure empyrical calls capture, over the whole sample or
    within every rolling window.

    The formula is as follows:

        - Capture Ratio = ((PRODUCT(1 + Asset Return))^(1 / n) - 1)
          / ((PRODUCT(1 + Benchmark Return))^(1 / n) - 1)

    The annualized method annualizes both averages, (1 + average)^P - 1, which is
    empyrical's capture (the ratio of the annual returns); the arithmetic method takes the
    ratio of the sums of the returns.

    With a window, each ratio covers the last `window` observations, such as 36 monthly
    returns for Morningstar's three year capture ratios, and the ratio can also be the
    Upside or Downside Capture Ratio (see `get_upside_capture_ratio`) or the upside over the
    downside capture. empyrical's rolling capture ratios are the annualized method on daily
    returns.

    Referenced against:
    - empyrical, capture (annualized): https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1531

    Shares similar logic:
    - empyrical, roll_up_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2020
    - empyrical, roll_down_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2045
    - empyrical, roll_up_down_capture: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2070

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns, one column per asset.
        benchmark_returns (pd.Series): Benchmark returns on the same dates.
        method (str, optional): The average of the returns. Options are: "geometric" (geometric
            averages), "annualized" (annualized geometric averages), "arithmetic" (the ratio of the
            sums). Defaults to "geometric".
        periods_per_year (float | pd.Series, optional): The observations in a year of the
            annualized method, or one value per asset. Defaults to 252.
        window (int | None, optional): The number of observations of a rolling window, at least 1. Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).
        side (str, optional): The capture ratio of every window; only applies with a window.
            Options are: "all" (overall Capture Ratio), "up" (Upside Capture Ratio), "down"
            (Downside Capture Ratio), "up_down" (the upside over the downside capture). Defaults
            to "all".

    Returns:
        pd.Series | pd.DataFrame: Capture Ratio values. With a window, the capture ratio at the
        end of every window, NaN before the first full window.
    """
    validate_window("window", window, 1)
    if window is None:
        return _get_capture_ratio(
            asset_returns, benchmark_returns, None, method, periods_per_year
        )

    # A Series is calculated as a one-column frame and returned as a Series again.
    if isinstance(asset_returns, pd.Series):
        return get_capture_ratio(
            asset_returns.to_frame(),
            benchmark_returns,
            method=method,
            periods_per_year=periods_per_year,
            window=window,
            side=side,
        ).iloc[:, 0]

    # A name of its own, as the window is no longer None inside the function below.
    window_size = window
    sides = {"up": [True], "down": [False], "all": [None], "up_down": [True, False]}

    check_option("side", side, sides)

    def window_capture(end: int) -> pd.Series:
        ratios = [
            _get_capture_ratio(
                asset_returns.iloc[end - window_size : end],
                benchmark_returns.iloc[end - window_size : end],
                upside,
                method,
                periods_per_year,
            )
            for upside in sides[side]
        ]

        return ratios[0] / ratios[1] if side == "up_down" else ratios[0]

    capture_ratios = {
        asset_returns.index[end - 1]: window_capture(end)
        for end in range(window_size, len(asset_returns) + 1)
    }

    return (
        pd.DataFrame(capture_ratios, index=asset_returns.columns)
        .T.reindex(asset_returns.index)
        .replace([np.inf, -np.inf], np.nan)
    )


@by_own_trading_days(compound=True)
def get_up_down_number(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    market: str = "up",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Up Number or Down Number of Bacon (2008): how often the asset rose (fell) when
    the benchmark rose (fell), as a share of those periods.

    The formula is as follows:

        - Up Number = Number of Periods with Asset Return > 0 and Benchmark Return > 0
          / Number of Periods with Benchmark Return > 0
        - Down Number = Number of Periods with Asset Return < 0 and Benchmark Return < 0
          / Number of Periods with Benchmark Return < 0

    Periods in which the asset has no return are left out of both counts.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons.

    Referenced against:
    - PerformanceAnalytics (R), UpDownRatios (Up Number, Down Number): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns, a "within period" (period,
            date) Multi Index gives one value per period.
        benchmark_returns (pd.Series): Benchmark returns.
        market (str, optional): The market. Options are: "up" (Up Number), "down" (Down
            Number). Defaults to "up".

    Returns:
        pd.Series | pd.DataFrame: The Up or Down Number.
    """
    check_option("market", market, ("up", "down"))

    def share(asset: pd.Series | pd.DataFrame, benchmark: pd.Series):
        counted = benchmark > 0 if market == "up" else benchmark < 0
        same_direction = (asset > 0 if market == "up" else asset < 0).astype(float)

        return same_direction.where(asset.notna()).where(counted, axis=0).mean()

    if isinstance(asset_returns, pd.DataFrame | pd.Series):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            frame = (
                asset_returns.to_frame()
                if isinstance(asset_returns, pd.Series)
                else asset_returns
            )
            combination = pd.concat([frame, benchmark_returns], axis=1)

            return combination.groupby(level=0).apply(
                lambda x: share(x[frame.columns], x[benchmark_returns.name])
            )

        return share(asset_returns, benchmark_returns)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


@by_own_trading_days(compound=True)
def get_win_rate(
    asset_returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    market: str = "all",
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Win Rate: the share of periods in which the asset's return exceeded the
    benchmark's.

    Periods in which either return is missing are left out of both the wins and the number of
    periods.

    The formula is as follows:

        - Win Rate = Number of Periods with Asset Return > Benchmark Return / Number of Periods
          with both an Asset and a Benchmark Return

    In up or down markets only the periods in which the benchmark rose (Benchmark Return > 0)
    or fell (Benchmark Return < 0) are counted, the up and down market batting averages of
    empyrical and the Up and Down Percent of PerformanceAnalytics' UpDownRatios.

    Referenced against:
    - PerformanceAnalytics (R), UpDownRatios (Up Percent, Down Percent): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/UpDownRatios.R#L44

    Shares similar logic:
    - quantstats, win_rate: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L564

    Args:
        asset_returns (pd.Series | pd.DataFrame): Asset returns.
        benchmark_returns (pd.Series): Benchmark returns.
        market (str, optional): The periods counted. Options are: "all" (every period), "up"
            (periods in which the benchmark rose), "down" (periods in which the benchmark fell).
            Defaults to "all".

    Returns:
        pd.Series | pd.DataFrame: Win Rate values.
    """
    check_option("market", market, ("all", "up", "down"))

    def win_share(
        asset: pd.Series | pd.DataFrame, benchmark: pd.Series, by_period: bool = False
    ) -> pd.Series | pd.DataFrame | float:
        # A comparison with a missing return is False, which would count a day without a quote as a loss; mask those days so the mean only covers days on which both returns exist.  # noqa: E501
        benchmark = benchmark.reindex(asset.index)
        wins = asset.gt(benchmark, axis=0).astype(float)
        counted = benchmark.notna()

        if market != "all":
            # A day the benchmark did not move is neither an up nor a down market.
            counted = benchmark > 0 if market == "up" else benchmark < 0

        shares = wins.where(asset.notna()).where(counted, axis=0)

        return shares.groupby(level=0).mean() if by_period else shares.mean()

    if isinstance(asset_returns, pd.DataFrame):
        if asset_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # The wins of every day are known without the period, so the shares per period are means over the counted days, all periods at once.  # noqa: E501
            return win_share(asset_returns, benchmark_returns, by_period=True)

        return win_share(asset_returns, benchmark_returns)

    if isinstance(asset_returns, pd.Series):
        return win_share(asset_returns, benchmark_returns)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_kappa_ratio(
    excess_returns: pd.Series | pd.DataFrame, order: int = 3
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Kappa Ratio: the mean excess return per unit of the n-th order Lower Partial
    Moment (LPM), a downside risk that weighs large losses more as the order rises.

    The Sortino Ratio (see `get_sortino_ratio`) is the special case of the Kappa Ratio of order 2.

    The formula is as follows:

        - Kappa Ratio = Mean(Excess Return) / LPM_order^(1/order)
        - LPM_order = (1/N) * sum( max(-Excess Return, 0)^order )

    Where N is the *total* number of observations (not just the number of negative
    ones) -- periods with a non-negative excess return contribute 0 to the sum, but
    still count towards N. This mirrors the same convention used in `get_sortino_ratio`
    (order 2): dividing only by the count of negative observations instead of the total
    N, or centering the moment on the downside subset's own mean instead of on the MAR
    of 0, would compute a different (non-standard) quantity than the Lower Partial
    Moment this ratio is defined against.

    For more information about the method, see the following paper:

    - Kaplan, P.D., & Knowles, J.A. (2004). "Kappa: A Generalized Downside Risk-Adjusted
    Performance Measure." Journal of Performance Measurement, 8(3), 42-54.

    Without a return below 0 the lower partial moment is 0 and the ratio is undefined (NaN).

    Shares similar logic:
    - PerformanceAnalytics (R), Kappa: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Kappa.R#L52

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        risk-free rate subtracted.
        order (int, optional): The order of the lower partial moment used in the denominator.
        Defaults to 3.

    Returns:
        pd.Series | pd.DataFrame: A Series or DataFrame of Kappa ratios with time as index
        and assets as columns.
    """
    if isinstance(excess_returns, pd.Series | pd.DataFrame):
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            return remove_undefined_ratios(
                excess_returns.groupby(level=0).apply(
                    lambda x: x.mean()
                    / (x.clip(upper=0).abs() ** order).mean() ** (1 / order)
                )
            )

        lower_partial_moment = (excess_returns.clip(upper=0).abs() ** order).mean()

        return remove_undefined_ratios(
            excess_returns.mean() / lower_partial_moment ** (1 / order)
        )

    raise TypeError("Expects pd.DataFrame, pd.Series inputs, no other value.")


@by_own_trading_days(rolling_only=True)
def get_omega_ratio(
    returns: pd.Series | pd.DataFrame,
    minimum_acceptable_return: float = 0.0,
    window: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Omega Ratio: the sum of gains above the minimum acceptable return (MAR) over
    the sum of losses below it, over the whole sample or within every rolling window.

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

    Shares similar logic:
    - PerformanceAnalytics (R), Omega: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Omega.R#L72
    - empyrical, omega_ratio: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L599
    - quantstats, omega: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1508

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) used
        as the threshold between gains and losses. Defaults to 0.0.
        window (int | None, optional): The number of observations of a rolling window, at least 1. Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame: Omega Ratio values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index. Without any return below
        the minimum acceptable return (in the sample or the window) there are no losses to
        divide by and the ratio is undefined (NaN).
    """
    validate_window("window", window, 1)
    if window is not None:

        def _omega(windows):
            excess_returns = windows - minimum_acceptable_return
            gains = excess_returns.clip(min=0).sum(axis=1)
            losses = (-excess_returns).clip(min=0).sum(axis=1)

            # A window without losses has no ratio.
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(losses > 0, gains / losses, np.nan)

        return apply_to_stacked_windows(returns, window, _omega)

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            # The gains and losses per period of all assets at once, rather than one period and asset at a time.  # noqa: E501
            excess_returns = returns - minimum_acceptable_return
            gains = excess_returns.where(excess_returns > 0).groupby(level=0).sum()
            losses = -excess_returns.where(excess_returns < 0).groupby(level=0).sum()

            return (gains / losses).where(losses > 0)

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


def get_gain_to_pain_ratio(
    returns: pd.Series | pd.DataFrame,
    minimum_acceptable_return: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Gain-to-Pain Ratio: the sum of all returns over the sum of the absolute value
    of all losses.

    The formula is as follows:

        - Gain-to-Pain Ratio = sum(Return) / sum(|Return| where Return < 0)

    A value of 0 means gains exactly offset losses; a value below (above) 0 means losses
    outweigh (are outweighed by) gains in aggregate.

    Measured against a minimum acceptable return (MAR) other than 0, the ratio is the
    Omega-Sharpe ratio of Kazemi, Schneeweis and Gupta (2004), the Omega ratio minus 1:

        - Omega-Sharpe Ratio = sum(Return - MAR) / sum(max(MAR - Return, 0)) = Omega Ratio - 1

    which equals the mean return above the MAR divided by the downside potential (the mean
    shortfall below the MAR), as PerformanceAnalytics defines it.

    For more information about the method, see the following source:

    - Schwager, J.D. (2012). "Hedge Fund Market Wizards: How Winning Traders Win." John
    Wiley & Sons. (Popularized this simple ratio as a robustness-focused alternative to
    the Sharpe ratio for evaluating trading strategies.)
    - Kazemi, H., Schneeweis, T., & Gupta, R. (2004). "Omega as a Performance Measure."
    Journal of Performance Measurement, 8(3), 16-25.

    Without a loss the ratio is undefined (NaN).

    Referenced against:
    - PerformanceAnalytics (R), OmegaSharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/OmegaSharpeRatio.R#L37
    - quantstats, gain_to_pain_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1584

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) per
        observation that gains and losses are measured from. Defaults to 0.0.

    Returns:
        pd.Series | pd.DataFrame: Gain-to-Pain Ratio values as float if returns is a
        pd.Series, otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_gain_to_pain_ratio,
                    minimum_acceptable_return=minimum_acceptable_return,
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            gain_to_pain_ratio = pd.concat(period_data_list, axis=1)

            return gain_to_pain_ratio.T

        return returns.aggregate(
            get_gain_to_pain_ratio, minimum_acceptable_return=minimum_acceptable_return
        )

    if isinstance(returns, pd.Series):
        excess_returns = returns - minimum_acceptable_return
        losses = excess_returns[excess_returns < 0].abs().sum()

        # A period without losses (or without returns) divides by zero: no ratio.
        with np.errstate(divide="ignore", invalid="ignore"):
            return remove_undefined_ratios(excess_returns.sum() / losses)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_d_ratio(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the d ratio: the losses relative to the gains, each weighted by how often they
    occur.

    The formula is as follows:

        - d Ratio = (n_d * SUM(|Return| where Return < 0)) / (n_u * SUM(Return where Return > 0))

    Where n_d is the number of negative returns and n_u the number of positive returns. It is
    the inverse of the Bernardo-Ledoit (gain-loss) ratio weighted by the frequencies of the
    losses and gains. Lower is better: 0 means there were no losses at all.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, p. 95.

    Without a positive return the ratio is undefined: NaN where PerformanceAnalytics reports
    infinity.

    Referenced against:
    - PerformanceAnalytics (R), DRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/DRatio.R#L38

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns. A "within period"
        (period, date) Multi Index gives one ratio per period and column.

    Returns:
        pd.Series | pd.DataFrame | float: The d ratio, a float for a Series, one per column for
        a DataFrame and one per period and column for a "within period" index.
    """
    if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        return returns.groupby(level=0).agg(
            lambda values: get_d_ratio(values.droplevel(0))
        )

    if isinstance(returns, pd.DataFrame):
        return returns.apply(get_d_ratio).astype(float)

    losses = returns[returns < 0]
    gains = returns[returns > 0]

    if gains.empty:
        return np.nan

    return float(len(losses) * -losses.sum() / (len(gains) * gains.sum()))


def get_prospect_ratio(
    returns: pd.Series | pd.DataFrame,
    minimum_acceptable_return: float = 0.0,
    loss_aversion: float = 2.25,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Prospect ratio: a Sortino-type ratio in which losses weigh more heavily than
    gains, following the loss aversion of prospect theory.

    The formula is as follows:

        - Prospect Ratio = ((1 / N) * SUM(MAX(Return, 0) + λ * MIN(Return, 0)) − MAR) / Downside Deviation
        - Downside Deviation = SQRT( (1 / N) * SUM( MIN(Return − MAR, 0)^2 ) )

    Where N is the number of returns, MAR the minimum acceptable return per observation and
    λ the loss aversion, 2.25 as estimated by Tversky and Kahneman (1992).

    For more information, see the following sources:

    - Watanabe, Y. (2006). "Is Sharpe Ratio Still Effective?" Journal of Performance
    Measurement, 11(1), 55-66.
    - Tversky, A., & Kahneman, D. (1992). "Advances in Prospect Theory: Cumulative
    Representation of Uncertainty." Journal of Risk and Uncertainty, 5(4), 297-323.
    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, p. 100.

    This is the formula of Bacon (2008) and the PerformanceAnalytics documentation.
    PerformanceAnalytics' ProspectRatio subtracts MAR / N instead of the MAR from the average, so both
    only agree for a MAR of 0, the default here.

    Without a return below the MAR the downside deviation is 0 and the ratio undefined (NaN).

    Referenced against:
    - PerformanceAnalytics (R), ProspectRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ProspectRatio.R#L32

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns. A "within period"
        (period, date) Multi Index gives one ratio per period and column.
        minimum_acceptable_return (float, optional): The minimum acceptable return (MAR) per
        observation. Defaults to 0.0.
        loss_aversion (float, optional): The weight λ of the losses relative to the gains.
        Defaults to 2.25.

    Returns:
        pd.Series | pd.DataFrame | float: The Prospect ratio, a float for a Series, one per
        column for a DataFrame and one per period and column for a "within period" index.
    """
    if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        return returns.groupby(level=0).agg(
            lambda values: get_prospect_ratio(
                values.droplevel(0), minimum_acceptable_return, loss_aversion
            )
        )

    if isinstance(returns, pd.DataFrame):
        return returns.apply(
            get_prospect_ratio,
            minimum_acceptable_return=minimum_acceptable_return,
            loss_aversion=loss_aversion,
        ).astype(float)

    returns = returns.dropna()

    if returns.empty:
        return np.nan

    weighted_return = (
        returns.clip(lower=0) + loss_aversion * returns.clip(upper=0)
    ).mean()
    downside_deviation = np.sqrt(
        ((returns - minimum_acceptable_return).clip(upper=0) ** 2).mean()
    )

    return remove_undefined_ratios(
        float((weighted_return - minimum_acceptable_return) / downside_deviation)
        if downside_deviation > 0
        else np.nan
    )


def get_omega_excess_return(
    returns: pd.Series | pd.DataFrame,
    downside_deviation: pd.Series | pd.DataFrame,
    benchmark_downside_deviation: pd.Series | float,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Omega excess return: the return minus a charge for downside risk that is
    priced at the downside risk of the benchmark.

    The formula is as follows:

        - Omega Excess Return = Return − 3 * β_S * σ_DB^2 = Return − 3 * σ_D * σ_DB
        - β_S = σ_D / σ_DB

    Where σ_D and σ_DB are the downside deviations of the asset and the benchmark below the
    minimum acceptable return, over the same horizon as the return, and β_S the style beta.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, p. 103.

    Referenced against:
    - PerformanceAnalytics (R), OmegaExcessReturn: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/OmegaExcessReturn.R#L36

    Args:
        returns (pd.Series | pd.DataFrame): The return of each period (columns per asset).
        downside_deviation (pd.Series | pd.DataFrame): The downside deviation of the asset in
        each period, scaled to the period.
        benchmark_downside_deviation (pd.Series | float): The downside deviation of the
        benchmark in each period, scaled to the period.

    Returns:
        pd.Series | pd.DataFrame: The Omega excess return per period and asset.
    """
    downside_risk_charge = (
        downside_deviation.mul(benchmark_downside_deviation, axis=0)
        if isinstance(downside_deviation, pd.DataFrame)
        else downside_deviation * benchmark_downside_deviation
    )

    return (returns - 3 * downside_risk_charge).dropna(how="all")


def get_return_statistics(
    returns: pd.Series,
    periods_per_year: float = 252.0,
    confidence_level: float = 0.95,
) -> pd.Series:
    """
    Calculate the statistics traders report about a series of returns: how often it wins,
    how large its wins and losses are, how long its streaks run, how it recovers from
    its drawdown, its extremes and outliers, and how precisely its mean return is known.

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
        - Best (Worst) Return = The highest (lowest) return
        - Exposure = Number of Non-Zero Returns / Number of Returns
        - CPC Index = Profit Factor x Win Rate x Payoff Ratio
        - Profit Ratio = (Average Win / Number of Positive Returns) / (|Average Loss| / Number of
          Negative Returns)
        - Outlier Win Ratio = 99th Percentile of the Returns / Average Win
        - Outlier Loss Ratio = 1st Percentile of the Returns / Average Loss
        - Risk of Ruin = ((1 - Win Rate) / (1 + Win Rate))^n
        - Risk-Adjusted Return = Annualized Return / Exposure
        - Mean Return = Average of all Returns
        - Standard Error = Standard Deviation of the Returns / SQRT(n)
        - Lower (Upper) Confidence Limit = Mean Return -(+) t(1 - (1 - c) / 2, n - 1) x Standard Error

    Where n is the number of returns, c the confidence level and t the quantile of Student's t
    distribution. The Annualized Return compounds the returns, PRODUCT(1 + Return)^(P / n) - 1
    with P the returns in a year (see `get_annualized_return`).

    A return of exactly zero is neither a win nor a loss: it is left out of the win rate,
    the average win and loss and the profit ratio, and ends a streak. The Expectancy equals
    the average of the non-zero returns. The Recovery Factor compounds the returns into the
    cumulative return, so that it relates the same wealth path to the drawdown measured on it.

    The Expectancy and the Payoff Ratio follow Tharp (2006). The Common Sense Ratio, the CPC
    Index, the Profit Ratio, the Outlier Win and Loss Ratios, the Risk of Ruin, the Exposure
    and the Risk-Adjusted Return are defined in quantstats (Ran Aroussi).

    For more information, see the following sources:

    - Tharp, V.K. (2006). "Trade Your Way to Financial Freedom." 2nd ed., McGraw-Hill.
    (Expectancy and the payoff ratio as the basis of evaluating a trading system.)

    Referenced against:
    - quantstats, avg_return (Expectancy): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L640
    - quantstats, best: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L362
    - quantstats, worst: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L395
    - quantstats, exposure: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L514
    - quantstats, cpc_index: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2417
    - quantstats, profit_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2312
    - quantstats, outlier_win_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2484
    - quantstats, outlier_loss_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2523
    - quantstats, risk_of_ruin: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1968
    - PerformanceAnalytics (R), mean.stderr: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/mean.utils.R#L137
    - PerformanceAnalytics (R), mean.LCL: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/mean.utils.R#L169
    - PerformanceAnalytics (R), mean.UCL: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/mean.utils.R#L208

    Shares similar logic:
    - quantstats, win_rate: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L564
    - quantstats, payoff_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2242
    - quantstats, profit_factor: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2370
    - quantstats, consecutive_wins: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L428
    - quantstats, consecutive_losses: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L471
    - quantstats, recovery_factor: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2562
    - quantstats, common_sense_ratio: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2451
    - quantstats, rar: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1698

    Args:
        returns (pd.Series): A Series of returns; missing returns are left out.
        periods_per_year (float, optional): The number of returns in a year, to annualize the
            Risk-Adjusted Return with. Defaults to 252.
        confidence_level (float, optional): The confidence level of the confidence limits
            of the mean return. Defaults to 0.95.

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
    payoff_ratio = average_win / abs(average_loss) if len(losses) else np.nan

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

    exposure = non_zero / len(returns) if len(returns) else np.nan
    standard_error = returns.std() / np.sqrt(len(returns)) if len(returns) else np.nan
    # The confidence limits need a standard deviation, so two returns at least.
    t_value = (
        stats.t.ppf(1 - (1 - confidence_level) / 2, len(returns) - 1)
        if len(returns) > 1
        else np.nan
    )

    return pd.Series(
        {
            "Win Rate": win_rate,
            "Average Win": average_win,
            "Average Loss": average_loss,
            "Payoff Ratio": payoff_ratio,
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
            "Best Return": returns.max() if len(returns) else np.nan,
            "Worst Return": returns.min() if len(returns) else np.nan,
            "Exposure": exposure,
            "CPC Index": profit_factor * win_rate * payoff_ratio,
            "Profit Ratio": (
                (average_win / len(wins)) / (abs(average_loss) / len(losses))
                if len(wins) and len(losses)
                else np.nan
            ),
            "Outlier Win Ratio": returns.quantile(0.99) / average_win,
            "Outlier Loss Ratio": returns.quantile(0.01) / average_loss,
            "Risk of Ruin": ((1 - win_rate) / (1 + win_rate)) ** len(returns),
            "Risk-Adjusted Return": (
                get_annualized_return(returns, periods_per_year) / exposure
                if exposure
                else np.nan
            ),
            "Mean Return": returns.mean() if len(returns) else np.nan,
            "Standard Error": standard_error,
            "Lower Confidence Limit": returns.mean() - t_value * standard_error,
            "Upper Confidence Limit": returns.mean() + t_value * standard_error,
        }
    )


def get_return_statistics_table(
    returns: pd.DataFrame,
    periods_per_year: float | pd.Series = 252.0,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Calculate the return statistics of every column: how often each asset wins, how large its
    wins and losses are and how its streaks, drawdown recovery and extremes look, see
    `get_return_statistics`.

    A within period index gives the statistics of every period.

    Args:
        returns (pd.DataFrame): A DataFrame of returns, one column per asset, optionally
            with a (period, date) index.
        periods_per_year (float | pd.Series, optional): The number of returns in a year, or
            one value per column. Defaults to 252.
        confidence_level (float, optional): The confidence level of the confidence limits
            of the mean return. Defaults to 0.95.

    Returns:
        pd.DataFrame: Over the whole history, the statistics as the index and the assets as
        the columns. Within periods, the periods as the index and (statistic, asset) as the
        columns.
    """

    def statistics_of(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: get_return_statistics(
                    frame[column],
                    (
                        periods_per_year[column]
                        if isinstance(periods_per_year, pd.Series)
                        else periods_per_year
                    ),
                    confidence_level,
                )
                for column in frame.columns
            }
        )

    if returns.index.nlevels != MULTI_PERIOD_INDEX_LEVELS:
        return statistics_of(returns)

    statistics = {
        sub_period: statistics_of(returns.loc[sub_period]).stack()
        for sub_period in returns.index.get_level_values(0).unique()
    }

    return pd.DataFrame(statistics).T


@by_own_trading_days(rolling_only=True)
def get_kelly_criterion(
    returns: pd.Series | pd.DataFrame,
    method: str = "discrete",
    fraction: float = 1.0,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Kelly Criterion: the share of capital to put at risk that maximizes the
    expected logarithmic growth of wealth, over the whole sample or within every rolling
    window.

    The formulas are as follows:

        - Discrete: Kelly = Win Rate - (1 - Win Rate) / Payoff Ratio
        - Continuous: Kelly = Average Excess Return / Variance of the Excess Returns

    The discrete form is Kelly's (1956) bet with a probability of winning and a ratio of the
    average win to the average loss, calculated from the returns (a zero return counting
    as neither). The continuous form is Thorp's (2006) for returns
    that are (close to) normally distributed, on excess returns over the risk free rate,
    with the sample variance (with n - 1 in the denominator); its value does not depend on the return frequency,
    since the average and the variance scale alike. A value above 1 means borrowing to
    invest, a negative value means the edge is negative (short or stay out).

    For more information, see the following sources:

    - Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical
    Journal, 35(4), 917-926.
    - Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock
    Market." Handbook of Asset and Liability Management, Volume 1, 385-428.
    - MacLean, L.C., Thorp, E.O. & Ziemba, W.T. (2011). "The Kelly Capital Growth Investment
    Criterion." World Scientific. (Fractional Kelly to reduce the volatility of growth.)

    Shares similar logic:
    - quantstats, kelly_criterion (discrete): https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2786
    - PerformanceAnalytics (R), KellyRatio (continuous): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/KellyRatio.R#L39

    Args:
        returns (pd.Series | pd.DataFrame): Returns for the discrete form, excess returns
            for the continuous form, optionally with a (period, date) index.
        method (str, optional): The form of the Kelly Criterion. Options are: "discrete"
            (Kelly's discrete bet), "continuous" (Thorp's continuous form). Defaults to "discrete".
        fraction (float, optional): The share of the full Kelly bet, e.g. 0.5 for half
            Kelly. Defaults to 1.0.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the Kelly Criterion over all returns (per period for a
            'within period' index).

    Returns:
        pd.Series | pd.DataFrame | float: The Kelly Criterion as a float for a Series, per
        column for a DataFrame and per period and column for a within period index, or with
        a window at the end of every window with time as index. NaN when it is undefined: the
        discrete form without wins or losses, the continuous form without variance.
    """
    validate_window("window", window, 2)
    check_option("method", method, ("discrete", "continuous"))

    if window is not None:
        return (
            apply_to_stacked_windows(
                returns, window, lambda windows: kelly_of_windows(windows, method)
            )
            * fraction
        )

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


def kelly_of_windows(windows: np.ndarray, method: str) -> np.ndarray:
    """
    Calculate the full Kelly Criterion of every row of an array of returns: the share of capital
    to put at risk that maximizes the expected logarithmic growth of each set of returns.

    Args:
        windows (np.ndarray): One set of returns per row, without missing values.
        method (str): The form of the Kelly Criterion. Options are: "discrete" (Kelly's discrete
            bet), "continuous" (Thorp's continuous form).

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
    Calculate the Compound Growth Rate (CGR): the constant rate per period that grows the first
    value of a series into its last.

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
    Calculate the Compound Annual Growth Rate (CAGR): the constant yearly rate that grows the
    first price into the last one, on calendar time.

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

    Shares similar logic:
    - quantstats, cagr: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1636
    - ffn, calc_cagr: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1563

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
    Calculate the Return per period: how much an investment gained or lost each week, month,
    quarter or year, compounding the daily returns.

    The period Return is obtained by compounding the daily returns within each
    period, following the formula:

        - Period Return = ((1 + Return 1) * (1 + Return 2) * ... * (1 + Return N)) - 1

    A period without any return (e.g. before a ticker listed) is NaN rather than 0.

    The cumulative return compounds the period returns further, following the formula:

        - Cumulative Return = (1 + Period Return 1) * (1 + Period Return 2) * ... * (1 + Period Return N)

    Shares similar logic:
    - empyrical, aggregate_returns: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L316
    - PerformanceAnalytics (R), Return.cumulative: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.cumulative.R#L27

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of daily returns.
        period (str): The period to calculate the Return for. Options are: "weekly", "monthly",
        "quarterly", "yearly". Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `returns`, to group by instead of deriving calendar periods from
        the index of `returns`. Use this when `returns` does not have a
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
    risk_free_rate: pd.Series | pd.DataFrame,
    cumulative: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Excess Return: how much the return exceeded the risk free rate.

    The cumulative excess return compounds the excess returns, following the formula:

        - Cumulative Excess Return = (1 + Excess Return 1) * (1 + Excess Return 2) * ... * (1 + Excess Return N)

    Shares similar logic:
    - PerformanceAnalytics (R), Return.excess: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.excess.R#L42

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        risk_free_rate (pd.Series | pd.DataFrame): A Series of the risk free rate, aligned to the
        same period as the returns, or a DataFrame with the rate of each column of the returns.
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
    Calculate the correlation matrix: how closely every pair of assets moves together.

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
    Calculate the covariance matrix: how far every pair of assets moves together, in the units
    of the returns squared.

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
    Calculate the CAPM residuals: the part of each (e.g. daily) excess return that the asset's
    Beta to the benchmark does not explain, within each period of a "within period" (period,
    date) Multi Index dataset with a Beta estimated once per period.

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
    capm_residuals: pd.Series | pd.DataFrame | None = None,
    excess_returns: pd.DataFrame | None = None,
    benchmark_excess_returns: pd.Series | None = None,
    window: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Appraisal Ratio: Jensen's Alpha per unit of the idiosyncratic (residual,
    unsystematic) risk left over from the CAPM regression that produced that Alpha, per period
    or within every rolling window.

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
    for the regression machinery this reuses). With a window, the rolling Jensen's Alpha is
    divided by the standard deviation of the CAPM regression residuals of the same window
    (see `get_residual_standard_deviation`), so each ratio only uses the returns of its own
    window and the first one follows the first full window.

    Also known as: Treynor-Black Appraisal Ratio, information ratio of a CAPM regression.

    For more information about the method, see the following paper:

    - Treynor, J.L., & Black, F. (1973). "How to Use Security Analysis to Improve
    Portfolio Selection." Journal of Business, 46(1), 66-86.

    Without residual risk the ratio is undefined (NaN).

    Shares similar logic:
    - PerformanceAnalytics (R), AppraisalRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/AppraisalRatio.R#L45

    Args:
        jensens_alpha (pd.Series | pd.DataFrame): Jensen's Alpha, see `get_jensens_alpha`; with
        a window the rolling Jensen's Alpha over the same window.
        capm_residuals (pd.Series | pd.DataFrame | None, optional): The pointwise CAPM regression
        residuals over the estimation window used for `jensens_alpha`, from which the idiosyncratic
        (residual) standard deviation is derived. When this has a "within period" Multi
        Index (period, date), the standard deviation is computed separately within each
        period and rescaled to the period frequency (see Notes below); otherwise it is
        computed over the entire series at once. Required without a window, not used with
        one. Defaults to None.
        excess_returns (pd.DataFrame | None, optional): The excess returns of the assets.
        Required with a window, not used without one. Defaults to None.
        benchmark_excess_returns (pd.Series | None, optional): The excess returns of the
        benchmark. Required with a window, not used without one. Defaults to None.
        window (int | None, optional): The number of observations of a rolling window, the
            window of `jensens_alpha`. A window of fewer than 3 observations leaves the
            regression no residual degrees of freedom and gives NaN. Defaults to None, which
            divides by the standard deviation of `capm_residuals`.

    Returns:
        pd.Series | pd.DataFrame: Appraisal Ratio values.

    Notes:
    - `capm_residuals` is typically obtained via `get_capm_residuals`, which calls
    `get_jensens_alpha` under the hood on already-excess returns (i.e. with a
    risk-free rate of 0), so it reuses the exact same CAPM regression machinery as
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
    validate_window("window", window, 2)
    if window is not None:
        if excess_returns is None or benchmark_excess_returns is None:
            raise ValueError(
                "A rolling Appraisal Ratio needs the excess_returns and the "
                "benchmark_excess_returns."
            )

        # A regression line through two returns fits them exactly: without residual
        # degrees of freedom the residual risk is 0 or rounding noise, so there is no ratio.
        if window < MINIMUM_REGRESSION_WINDOW:
            return jensens_alpha * np.nan

        return remove_undefined_ratios(
            jensens_alpha
            / get_residual_standard_deviation(
                excess_returns, benchmark_excess_returns, window=window
            )
        )

    if capm_residuals is None:
        raise ValueError(
            "The Appraisal Ratio needs the capm_residuals, or a window with the "
            "excess_returns and the benchmark_excess_returns."
        )

    if capm_residuals.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        residual_standard_deviation = capm_residuals.groupby(level=0).std()
        # Each asset's own number of residuals in the period.
        observations_per_period = capm_residuals.groupby(level=0).count()
        residual_standard_deviation = residual_standard_deviation * np.sqrt(
            observations_per_period
        )
    else:
        residual_standard_deviation = capm_residuals.std()

    return remove_undefined_ratios(jensens_alpha / residual_standard_deviation)


def get_modified_jensens_alpha(
    jensens_alpha: pd.Series | pd.DataFrame,
    beta: pd.Series | pd.DataFrame,
    benchmark_standard_deviation: pd.Series | float | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the modified Jensen's alpha, Jensen's Alpha per unit of Beta, or the alternative
    Jensen's alpha, Jensen's Alpha per unit of systematic risk: how much risk-adjusted return
    the asset earned for each unit of market risk it took.

    The formula is as follows:

        - Modified Jensen's Alpha = Jensen's Alpha / Beta
        - Alternative Jensen's Alpha = Jensen's Alpha / (Beta * Benchmark Standard Deviation)

    Next to the Appraisal Ratio (Jensen's Alpha per unit of specific risk, see
    `get_appraisal_ratio`) these are the two other ways Bacon (2008) scales Jensen's Alpha
    by risk.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, p. 77.

    A Beta of 0 (or of rounding noise) leaves the ratio undefined (NaN).

    Referenced against:
    - PerformanceAnalytics (R), AppraisalRatio (modified, alternative): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/AppraisalRatio.R#L45

    Args:
        jensens_alpha (pd.Series | pd.DataFrame): Jensen's Alpha, see `get_jensens_alpha`.
        beta (pd.Series | pd.DataFrame): The Beta that Jensen's Alpha was calculated with.
        benchmark_standard_deviation (pd.Series | float | None, optional): The standard deviation
        of the benchmark over the same periods (one value per period for a DataFrame). When
        given, the alternative Jensen's alpha is returned. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The modified (or alternative) Jensen's alpha.
    """
    risk = beta
    if benchmark_standard_deviation is not None:
        risk = (
            beta.mul(benchmark_standard_deviation, axis=0)
            if isinstance(beta, pd.DataFrame)
            else beta * benchmark_standard_deviation
        )

    return remove_undefined_ratios(jensens_alpha / risk)


@by_own_trading_days(compound=True)
def get_residual_standard_deviation(
    excess_returns: pd.Series | pd.DataFrame,
    benchmark_excess_returns: pd.Series,
    window: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the residual standard deviation: how much of each asset's excess return its Beta
    to the benchmark leaves unexplained, over the whole sample or within every rolling window,
    each window with the Beta of its own regression.

    The residuals are those of the regression of the asset's excess returns on the
    benchmark's (over the sample or the window), so they follow from the variances and the
    covariance without computing them one by one:

        - Beta = Cov(Asset Excess Return, Benchmark Excess Return) / Var(Benchmark Excess Return)
        - Residual Variance = Var(Asset Excess Return) - Beta^2 * Var(Benchmark Excess Return)

    All three moments are sample moments (with n - 1 in the denominator), so the residual variance is the sum of
    the squared residuals divided by the number of returns minus 1, the sample variance of the
    residuals. Taking the standard deviation of residuals that each come from the Beta of
    a different window instead would mix regressions and need two windows of returns
    before its first value.

    Args:
        excess_returns (pd.Series | pd.DataFrame): The excess returns of the assets.
        benchmark_excess_returns (pd.Series): The excess returns of the benchmark, on the
            same dates.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the residual standard deviation over all returns.

    Returns:
        pd.Series | pd.DataFrame: The residual standard deviation per asset, or with a window
        at the end of every full window, one column per asset.
    """
    validate_window("window", window, 2)
    excess_returns = (
        excess_returns.to_frame()
        if isinstance(excess_returns, pd.Series)
        else excess_returns
    )
    residual_standard_deviation = (
        pd.Series(index=excess_returns.columns, dtype=np.float64)
        if window is None
        else pd.DataFrame(
            index=excess_returns.index, columns=excess_returns.columns, dtype=np.float64
        )
    )

    for column in excess_returns.columns:
        # Both moments run over the dates on which the asset and the benchmark have a
        # return, as in `get_beta`.
        benchmark = benchmark_excess_returns.where(excess_returns[column].notna())
        asset = excess_returns[column].where(benchmark.notna())

        # A perfect fit can leave a residual variance a rounding error below 0.
        if window is None:
            residual_standard_deviation[column] = np.sqrt(
                max(asset.var() - asset.cov(benchmark) ** 2 / benchmark.var(), 0)
            )
        else:
            covariance = asset.rolling(window).cov(benchmark)
            benchmark_variance = benchmark.rolling(window).var()
            residual_variance = (
                asset.rolling(window).var() - covariance**2 / benchmark_variance
            )
            residual_standard_deviation[column] = np.sqrt(
                residual_variance.clip(lower=0)
            )

    return residual_standard_deviation


def get_fama_decomposition(
    asset_returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame | float,
    beta: pd.Series | pd.DataFrame | float,
    benchmark_returns: pd.Series | float,
    asset_standard_deviation: pd.Series | pd.DataFrame,
    benchmark_standard_deviation: pd.Series | float,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Calculate the Fama (1972) decomposition: how much of the excess return came from security
    selection (Selectivity) and how much from carrying risk that diversification would have
    removed (Diversification).

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

        - Selectivity = (Asset Return − Risk-Free Rate) − (Sigma_Portfolio / Sigma_Market) *
          (Benchmark Return − Risk-Free Rate)
        - Diversification = [Risk-Free Rate + (Sigma_Portfolio / Sigma_Market) * (Benchmark Return − Risk-Free Rate)] −
          [Risk-Free Rate + Beta * (Benchmark Return − Risk-Free Rate)]

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

    Shares similar logic:
    - PerformanceAnalytics (R), NetSelectivity: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/NetSelectivity.R#L35
    - PerformanceAnalytics (R), Selectivity: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Selectivity.R#L32

    Args:
        asset_returns (pd.Series | pd.DataFrame | float): The asset's or portfolio's
        return over the period.
        risk_free_rate (pd.Series | pd.DataFrame | float): The risk-free rate over the same period,
        or a DataFrame with the rate of each asset over the periods of its own calendar.
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
            column_risk_free_rate = (
                risk_free_rate[column]
                if isinstance(risk_free_rate, pd.DataFrame)
                else risk_free_rate
            )
            capm_actual_beta = column_risk_free_rate + beta[column] * (
                benchmark_returns - column_risk_free_rate
            )
            capm_total_risk = column_risk_free_rate + total_risk_ratio[column] * (
                benchmark_returns - column_risk_free_rate
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
    Calculate the Adjusted Sharpe Ratio (ASR): the Sharpe ratio penalized for negative skewness
    and fat tails of the returns.

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
    `get_probabilistic_sharpe_ratio` and used by `get_deflated_sharpe_ratio`.
    `risk_model.get_kurtosis` returns the excess (Fisher) kurtosis by default (Normal = 0);
    either ask it for the raw (Pearson) kurtosis before passing the result in here, or add 3
    to an excess kurtosis value first. Getting this convention backwards flips the sign
    of the ((K - 3) / 24) * SR^2 term for any return series that is actually close to
    Normal (where raw kurtosis is close to 3 but excess/Fisher kurtosis is close to 0),
    silently turning a near-zero adjustment into a large, wrong one, so double check which
    convention is being fed in.

    For more information about the method, see the following paper:

    - Pezier, J., & White, A. (2006). "The Relative Merits of Investable Hedge Fund
    Indices and of Funds of Hedge Funds in Optimal Passive Portfolios." ICMA Centre
    Discussion Papers in Finance, DP2006-10.

    Shares similar logic:
    - PerformanceAnalytics (R), AdjustedSharpeRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/AdjustedSharpeRatio.R#L35

    Args:
        sharpe_ratio (pd.Series | pd.DataFrame | float): The (ordinary) Sharpe ratio, see
        `get_sharpe_ratio`.
        skewness (pd.Series | pd.DataFrame | float): The skewness (γ₃) of the same returns
        used to compute `sharpe_ratio`.
        kurtosis (pd.Series | pd.DataFrame | float): The non-excess (raw) kurtosis (γ₄) of
        the same returns used to compute `sharpe_ratio`. See the note above — a Normal
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
    Calculate the STARR (Stable Tail Adjusted Return Ratio): the excess return per unit of
    Conditional Value at Risk, the average loss in the worst tail.

    The Sharpe ratio penalizes upside and downside volatility equally via the standard
    deviation. The STARR ratio instead scales the mean excess return by the Conditional
    Value at Risk (CVaR / Expected Shortfall), a coherent tail-risk measure that only looks
    at the average magnitude of losses beyond the `alpha` quantile. This makes STARR more
    appropriate than the Sharpe ratio for return distributions with fat left tails, where
    the standard deviation understates the risk that actually matters to an investor.

    The formula is as follows:

        - STARR Ratio = Excess Return / |CVaR(alpha)|

    Where CVaR is computed on the raw (non-excess) `returns` using the historical method
    (see `financetoolkit.risk.value_at_risk.cvar_model.get_cvar_historic`), which this function calls
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

    A CVaR of 0 leaves the ratio undefined (NaN).

    Args:
        excess_returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns with
        the risk-free rate subtracted, already aggregated to one value per period (see
        above).
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

    starr_ratio = remove_undefined_ratios(
        excess_returns / abs(conditional_value_at_risk)
    )

    if isinstance(starr_ratio, pd.Series | pd.DataFrame):
        return starr_ratio.dropna(how="all")

    return starr_ratio


def get_rachev_ratio(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Rachev Ratio (R-Ratio): the average size of the extreme gains over that of the
    extreme losses.

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
    `financetoolkit.risk.value_at_risk.cvar_model.get_cvar_historic` function — the left tail directly on
    `returns`, and the right tail on `-returns` (which turns the upper tail of `returns`
    into the lower tail of the negated series, letting the same CVaR machinery compute it
    without reimplementing any tail-averaging logic) — rather than duplicating the CVaR
    calculation.

    Also known as: R-Ratio.

    For more information about the method, see the following paper:

    - Biglova, A., Ortobelli, S., Rachev, S.T., & Stoyanov, S. (2004). "Different Approaches
    to Risk Estimation in Portfolio Theory." Journal of Portfolio Management, 31(1), 103-112.

    A left tail expected shortfall of 0 leaves the ratio undefined (NaN).

    Shares similar logic:
    - PerformanceAnalytics (R), RachevRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/RachevRatio.R#L30

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns.
        alpha (float, optional): The confidence level used for both tails (e.g. 0.05 for
        the best/worst 5% of outcomes). Defaults to 0.05.

    Returns:
        pd.Series | pd.DataFrame | float: Rachev Ratio values, a single float for a Series.
    """
    right_tail_expected_shortfall = -cvar_model.get_cvar_historic(-returns, alpha)
    left_tail_expected_shortfall = -cvar_model.get_cvar_historic(returns, alpha)

    rachev_ratio = remove_undefined_ratios(
        right_tail_expected_shortfall / left_tail_expected_shortfall
    )

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
    Calculate the Treynor-Mazuy market timing model: whether the asset raised its market
    exposure ahead of rising markets, a quadratic term in the benchmark excess return next to
    Alpha and Beta.

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

    Shares similar logic:
    - PerformanceAnalytics (R), MarketTiming (TM): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MarketTiming.R#L52

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
    Calculate the Henriksson-Merton market timing model: whether the asset had a higher Beta in
    rising markets than in falling ones, next to its Alpha.

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

    Shares similar logic:
    - PerformanceAnalytics (R), MarketTiming (HM): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MarketTiming.R#L52

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
    Calculate the weights of returns-based style analysis (Sharpe, 1992): the mix of style
    indices that tracks a fund's returns most closely, every weight between 0 and 1 and the
    weights adding up to 1.

    The weights minimize the variance of the difference between the fund's return and the
    weighted style returns. As

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
    returns: pd.Series,
    style_returns: pd.DataFrame,
    periods_per_year: int = 12,
    window: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate returns-based style analysis (Sharpe, 1992): the mix of style indices whose
    returns track a fund's returns most closely, the fund's effective asset mix, over the whole
    sample or over a rolling window to show how the style drifts over time.

    The weights are found by minimizing the variance of the difference between the fund's return
    and the weighted style returns, with every weight between 0 and 1 and the weights adding up
    to 1 (see `get_style_weights`):

        - Minimize Var(Fund Return - SUM(Weight × Style Return)), with 0 <= Weight <= 1 and
          SUM(Weight) = 1

    The other figures are in-sample, measured over the same returns the weights are fitted
    to: the part of the fund's return the styles do not explain is the Selection Return, its
    volatility the Tracking Error and the share of the fund's variance the styles explain the
    R Squared. The in-sample figures flatter the fit, as the weights are chosen to explain
    these very returns.

    With a window, these figures are those of the `window` periods up to and including each
    date, and next to them is Sharpe's out-of-sample measure of selection. The Out-of-Sample
    Selection Return of a date is the fund's return of that period minus the return of that
    period of the style mix estimated over the window that ends one period earlier: the
    return the manager added beyond a passive mix of the styles the manager was known to
    follow. Sharpe estimates the style over the previous 60 months. It is a return per
    period (not annualized), so its average and cumulative sum show whether the selection
    paid off.

    For more information, see the following paper:

    - Sharpe, W.F. (1992). "Asset Allocation: Management Style and Performance Measurement."
      The Journal of Portfolio Management, 18(2), 7-19.

    Args:
        returns (pd.Series): The fund's returns.
        style_returns (pd.DataFrame): The returns of the style indices, a column each, on the
            same dates as `returns`.
        periods_per_year (int, optional): The returns per year, to annualize the (in-sample)
            selection return and the tracking error. Defaults to 12.
        window (int | None, optional): The number of observations of a rolling window, at least 2,
            the periods every estimate uses. Defaults to None, which estimates the style mix over
            all returns.

    Returns:
        pd.Series | pd.DataFrame: The weight of every style, the R Squared, the (annualized)
        Selection Return and the (annualized) Tracking Error. Dates on which the fund or a
        style has no return are left out; with no more complete dates than styles the weights
        are undetermined and every value is NaN. With a window, a DataFrame with these per date
        (the last date of a window) and the Out-of-Sample Selection Return, which is NaN for
        the first window, as it has no earlier estimate.
    """
    validate_window("window", window, 2)
    styles = list(style_returns.columns)
    columns = [*styles, "R Squared", "Selection Return", "Tracking Error"]

    if window is not None:
        # By position from here on, so a fund that carries the name of a style does not clash.
        data = pd.concat([returns, style_returns], axis=1)
        fund_returns, style_data = data.iloc[:, 0], data.iloc[:, 1:]

        if len(data) < window:
            return pd.DataFrame(
                columns=[*columns, "Out-of-Sample Selection Return"], dtype=float
            )

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
            columns=columns,
            dtype=float,
        )

        # The weights of the previous window applied to this period's style returns. A style
        # without a return leaves the period's figure unknown rather than treating it as zero.
        previous_weights = result.iloc[:, : len(styles)].shift(1).to_numpy()
        style_mix_return = np.sum(
            style_data.iloc[window - 1 :].to_numpy(dtype=float) * previous_weights,
            axis=1,
        )
        result["Out-of-Sample Selection Return"] = (
            fund_returns.iloc[window - 1 :].to_numpy(dtype=float) - style_mix_return
        )

        return result

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


def get_geometric_excess_return(
    returns: pd.Series | pd.DataFrame, risk_free_rate: pd.Series
) -> pd.Series | pd.DataFrame:
    """
    Calculate the geometric excess return: the return relative to the risk-free return of the
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
    window: int | None = None,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculate the Morningstar Risk-Adjusted Return (MRAR): the return a risk-averse investor
    would accept with certainty instead of the uncertain returns, the annualized certainty
    equivalent of the geometric excess returns for a risk aversion gamma, over the whole sample
    or within every rolling window.

    The more the returns vary, the further it falls below their geometric mean. Morningstar
    ranks funds within their category on it, with a gamma of 2, for its star rating, over
    windows of 36, 60 and 120 months for the three, five and ten year figures.

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
        window (int | None, optional): The number of observations of a rolling window, at least 1. Defaults
            to None, which calculates the MRAR over all returns.

    Returns:
        float | pd.Series | pd.DataFrame: The MRAR as a decimal, a float for a Series and one
        value per column for a DataFrame. Each column uses its own returns, without its missing
        values. With a window, the MRAR at the end of every window with time as index, NaN
        until a window is full of returns.
    """
    validate_window("window", window, 1)
    if gamma < 0:
        raise ValueError("The risk aversion gamma cannot be negative.")

    if window is not None:
        # A window with a missing return is not a full window of the fund's history.
        return excess_returns.rolling(window, min_periods=window).apply(
            lambda values: _morningstar_risk_adjusted_return(
                values, gamma, periods_per_year
            ),
            raw=True,
        )

    if isinstance(excess_returns, pd.Series):
        return _morningstar_risk_adjusted_return(
            excess_returns.to_numpy(dtype=float), gamma, periods_per_year
        )

    return excess_returns.apply(
        lambda column: _morningstar_risk_adjusted_return(
            column.to_numpy(dtype=float), gamma, periods_per_year
        )
    )


def get_lo_annualization_factor(
    autocorrelations: pd.Series | np.ndarray | list[float],
    periods_per_year: int,
) -> float:
    """
    Calculate Lo's (2002) annualization factor η(q): how much a Sharpe ratio grows from one
    period to q periods (e.g. a year) when the returns are serially correlated.

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


@by_own_trading_days(rolling_only=True)
def get_lo_sharpe_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    periods_per_year: float,
    lags: int | None = None,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the annualized Sharpe ratio of Lo (2002): the mean excess return per unit of
    volatility over a year, with the annualization corrected for the serial correlation of the
    returns, over the whole sample or within every rolling window.

    The usual annualized Sharpe ratio, SQRT(q) * SR, assumes the returns are independent.
    When returns are autocorrelated, the variance of a q-period return is no longer q times
    the variance of a one-period return, and Lo shows the q-period Sharpe ratio becomes:

        - SR(q) = η(q) * SR
        - η(q) = q / SQRT( q + 2 * SUM_{k=1}^{q-1} (q - k) * ρ_k )

    Where SR is the Sharpe ratio of one period (mean excess return divided by its sample
    standard deviation with n - 1 in the denominator), q is `periods_per_year` and ρ_k the sample autocorrelation
    at lag k (Box-Jenkins estimator). The sample autocorrelations at lags close to the
    sample size rest on very few pairs, so the sum includes `lags` lags and takes the
    autocorrelations beyond them as 0. By default `lags` follows the Newey and West (1994)
    rule floor(4 * (n / 100)^(2/9)), capped at q - 1 (Lo's formula itself), with n the
    number of returns (of the window).

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
    - Moran, P.A.P. (1948). "Some Theorems on Time Series: II. The Significance of the Serial
    Correlation Coefficient." Biometrika, 35(3/4), 255-260.
    - Newey, W.K., & West, K.D. (1994). "Automatic Lag Selection in Covariance Matrix
    Estimation." Review of Economic Studies, 61(4), 631-653.

    Returns without variation have no ratio (NaN).

    Args:
        excess_returns (pd.Series | pd.DataFrame): The excess returns. With a "within
        period" (period, date) Multi Index, one ratio is calculated per period.
        periods_per_year (float): The number of return observations in a year (q), e.g.
        252 for daily and 12 for monthly returns. Rounded to a whole number.
        lags (int | None, optional): The number of autocorrelation lags in the sum.
        Defaults to None, the Newey and West (1994) rule on the number of returns.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the ratio over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame | float: The annualized Sharpe ratio per period (Multi
        Index), per column (DataFrame) or a single value (Series), or with a window at the end
        of every full window. NaN with fewer than 30 observations.
    """
    validate_window("window", window, 2)
    if window is not None:
        return excess_returns.rolling(window, min_periods=window).apply(
            lambda values: get_lo_sharpe_ratio(values, periods_per_year, lags)
        )

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

    return remove_undefined_ratios(
        (values.mean() / values.std(ddof=1)) * get_lo_annualization_factor(rho, q)
    )


def get_sharpe_ratio_standard_error(
    excess_returns: pd.Series | pd.DataFrame,
    method: str = "iid",
    lags: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the standard error of the Sharpe ratio: how far the measured Sharpe ratio may lie
    from the true one.

    The formulas are as follows, with SR the Sharpe ratio of one period and n the number of
    observations:

        - Independent normal returns: SE = SQRT( (1 + SR^2 / 2) / n ) (Lo, 2002, equation 9, normally and
          independently distributed returns)
        - Non-normal returns: SE = SQRT( (1 − γ₃·SR + ((γ₄ − 1) / 4)·SR^2) / (n − 1) ) (Mertens,
          2002; Bailey & López de Prado, 2012), independent but skewed and fat-tailed
          returns, with γ₃ the skewness and γ₄ the non-excess (raw) kurtosis. This is the
          standard error inside the Probabilistic Sharpe Ratio.
        - GMM: SE = SQRT( ∂g/∂θ · Σ · ∂g/∂θ' / n ) (Lo, 2002, Appendix), returns that may
          also be serially correlated and heteroskedastic. θ = (μ, σ²), g(θ) = μ / σ, so
          ∂g/∂θ = (1 / σ, −μ / (2σ³)), and Σ is the Newey and West (1987) long-run covariance
          of the moment conditions (x_t − μ, (x_t − μ)² − σ²) with Bartlett weights over
          `lags` lags (σ² with 1 / n, as in GMM).

    With independent normal returns the three agree as n grows; without lags the GMM and
    non-normal errors differ only by the finite-sample corrections of the moments.

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
        method (str, optional): The standard error. Options are: "iid" (independent normal
        returns), "non_normal" (independent, skewed and fat-tailed returns), "gmm" (GMM, serially
        correlated and heteroskedastic returns). Defaults to "iid".
        lags (int | None, optional): The number of lags in the Newey-West covariance, only
        for the GMM error. Defaults to None, the Newey and West (1994) rule
        floor(4 * (n / 100)^(2/9)).

    Returns:
        pd.Series | pd.DataFrame | float: The standard error, on the scale of the Sharpe
        ratio of one period (not annualized).
    """
    check_option("method", method, ("iid", "non_normal", "gmm"))

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


SHARPE_RATIO_DIFFERENCE_METHODS = ("memmel", "ledoit_wolf")


def get_sharpe_ratio_difference(
    excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    method: str = "memmel",
    lags: int | None = None,
) -> pd.Series:
    """
    Calculate the Sharpe ratio difference test: whether two Sharpe ratios differ (H0: SR_a =
    SR_b), on the dates both have an excess return, with a two-sided p-value from the standard
    normal distribution.

    The formulas are as follows, with n the number of paired observations:

        - Memmel: z = (SR_a − SR_b) / SQRT( (2 − 2ρ + (SR_a² + SR_b² − 2·SR_a·SR_b·ρ²) / 2) / n ),
          the Jobson and Korkie (1981) test as corrected by Memmel (2003), for independent and
          jointly normal returns, with SR = mean / standard deviation (n - 1 in the denominator) and ρ the
          correlation of the two excess returns.
        - Ledoit-Wolf: z = Δ / SQRT( ∇f' Ψ ∇f / n ), the HAC test of Ledoit and Wolf (2008,
          section 3.1), robust to fat tails and serial correlation. With v = (μ_a, μ_b, γ_a, γ_b)
          the means and the mean squares (γ = mean(r²)), Δ = f(v) = μ_a / SQRT(γ_a − μ_a²) −
          μ_b / SQRT(γ_b − μ_b²), ∇f = (γ_a / c_a, −γ_b / c_b, −μ_a / (2·c_a), μ_b / (2·c_b))
          with c = (γ − μ²)^(3/2), and Ψ = n / (n − 4) · (Γ_0 + Σ_{j=1..L} (1 − j / (L + 1))·(Γ_j +
          Γ_j')), the Bartlett (Newey and West 1987) kernel estimate of the long-run covariance
          of y_t = (r_a − μ_a, r_b − μ_b, r_a² − γ_a, r_b² − γ_b), with Γ_j = Σ_t y_t·y_{t−j}' / n.

    Args:
        excess_returns (pd.Series): The excess returns of the first asset.
        benchmark_excess_returns (pd.Series): The excess returns of the second asset.
        method (str, optional): The test. Options are: "memmel" (Jobson-Korkie test with
            Memmel's correction), "ledoit_wolf" (Ledoit-Wolf HAC test). Defaults to "memmel".
        lags (int | None, optional): The number of lags L of the Ledoit-Wolf covariance.
            Defaults to None, the Newey and West (1994) rule floor(4 * (n / 100)^(2/9)).

    Returns:
        pd.Series: The Sharpe Difference (SR_a − SR_b, of one period), its Standard Error, the
            Z-Statistic and the P-Value, NaN with fewer than five paired observations.
    """
    check_option("method", method, SHARPE_RATIO_DIFFERENCE_METHODS)

    paired = pd.concat([excess_returns, benchmark_excess_returns], axis=1).dropna()
    values = paired.to_numpy(dtype=float)
    n = len(values)
    result = pd.Series(
        np.nan, index=["Sharpe Difference", "Standard Error", "Z-Statistic", "P-Value"]
    )

    # The Ledoit-Wolf covariance scales by n / (n - 4), the number of moments estimated. A
    # series without variation (up to rounding, 1e-14 per observation) has no Sharpe ratio:
    # its standard deviation would be rounding noise.
    if n < 5 or np.any(  # noqa: PLR2004
        np.ptp(values, axis=0) <= n * 1e-14 * np.abs(values).max(axis=0)
    ):
        return result

    if method == "memmel":
        sharpe_ratios = values.mean(axis=0) / values.std(axis=0, ddof=1)
        correlation = np.corrcoef(values, rowvar=False)[0, 1]
        difference = sharpe_ratios[0] - sharpe_ratios[1]
        variance = (
            2
            - 2 * correlation
            + (
                sharpe_ratios[0] ** 2
                + sharpe_ratios[1] ** 2
                - 2 * sharpe_ratios[0] * sharpe_ratios[1] * correlation**2
            )
            / 2
        ) / n
    else:
        means = values.mean(axis=0)
        mean_squares = (values**2).mean(axis=0)
        variances = mean_squares - means**2
        difference = means[0] / np.sqrt(variances[0]) - means[1] / np.sqrt(variances[1])
        cubed = variances**1.5
        gradient = np.array(
            [
                mean_squares[0] / cubed[0],
                -mean_squares[1] / cubed[1],
                -means[0] / (2 * cubed[0]),
                means[1] / (2 * cubed[1]),
            ]
        )
        moments = np.column_stack([values - means, values**2 - mean_squares])

        lags = get_newey_west_lags(n) if lags is None else lags
        lags = max(0, min(lags, n - 1))

        covariance = moments.T @ moments / n
        for lag in range(1, lags + 1):
            autocovariance = moments[lag:].T @ moments[:-lag] / n
            covariance += (1 - lag / (lags + 1)) * (autocovariance + autocovariance.T)

        variance = gradient @ (n / (n - 4) * covariance) @ gradient / n

    standard_error = np.sqrt(variance)
    z_statistic = difference / standard_error

    result[:] = [
        difference,
        standard_error,
        z_statistic,
        2 * stats.norm.sf(abs(z_statistic)),
    ]

    return result


def get_sharpe_ratio_difference_test(
    excess_returns: pd.DataFrame,
    benchmark_excess_returns: pd.Series,
    method: str = "memmel",
    lags: int | None = None,
) -> pd.DataFrame:
    """
    Calculate the Sharpe ratio difference test of every column: whether its Sharpe ratio differs
    from the benchmark's, with `get_sharpe_ratio_difference`, over the whole sample or, with a
    "within period" (period, date) Multi Index, per period.

    Args:
        excess_returns (pd.DataFrame): The excess returns, one column per asset.
        benchmark_excess_returns (pd.Series): The excess returns of the benchmark, on the
            same index.
        method (str, optional): The test. Options are: "memmel" (Jobson-Korkie test with
            Memmel's correction), "ledoit_wolf" (Ledoit-Wolf HAC test). Defaults to "memmel".
        lags (int | None, optional): The number of lags of the Ledoit-Wolf covariance.
            Defaults to None, the Newey and West (1994) rule.

    Returns:
        pd.DataFrame: The Sharpe Difference, Standard Error, Z-Statistic and P-Value per
            asset (columns), per period (rows) with a within period index and as a single
            row otherwise.
    """
    check_option("method", method, SHARPE_RATIO_DIFFERENCE_METHODS)

    def test(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
        return pd.concat(
            {
                column: get_sharpe_ratio_difference(
                    returns[column], benchmark, method, lags
                )
                for column in returns.columns
            }
        )

    if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        return pd.DataFrame(
            {
                period: test(
                    period_returns, benchmark_excess_returns.loc[period_returns.index]
                )
                for period, period_returns in excess_returns.groupby(
                    level=0, sort=False
                )
            }
        ).T

    return test(excess_returns, benchmark_excess_returns).to_frame().T


def get_minimum_track_record_length(
    sharpe_ratio: pd.Series | pd.DataFrame | float,
    benchmark_sharpe_ratio: float,
    skewness: pd.Series | pd.DataFrame | float,
    kurtosis: pd.Series | pd.DataFrame | float,
    confidence_level: float = 0.95,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Minimum Track Record Length (MinTRL): the number of return observations needed
    before the measured Sharpe ratio is significantly above a benchmark Sharpe ratio.

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

    Shares similar logic:
    - PerformanceAnalytics (R), MinTrackRecord: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/MinTRL.R#L49

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
    Calculate the standardized co-moment of each asset with the benchmark: how the asset's
    return moves with a power of the market's moves, the shared part of `get_coskewness` (power
    2) and `get_cokurtosis` (power 3).

    The formula is as follows:

        - Co-Moment = E[a · b^power] / (SQRT(E[a^2]) · E[b^2]^(power / 2))

    Where b is the demeaned benchmark return and a the demeaned asset return (standardized
    co-moment) or the residual of the asset's regression on the benchmark (residual
    co-moment), over the dates both have a return, with population moments (1 / n).

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns. With a "within period"
        (period, date) Multi Index, one value is calculated per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        power (int): The power of the benchmark deviation, 2 or 3.
        method (str): The co-moment. Options are: "standardized" (demeaned asset returns),
            "residual" (residuals of the asset's regression on the benchmark).

    Returns:
        pd.Series | pd.DataFrame | float: The co-moment per period (Multi Index), per
        column (DataFrame) or a single value (Series).
    """
    check_option("method", method, ("standardized", "residual"))

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
    with the squared (large, either way) moves of the market.

    A negative co-skewness means the asset does worse when the market makes large moves, which
    adds to the market's downside skewness, and investors demand a premium for holding it.

    The formulas are as follows:

        - Standardized: CoSkewness = E[(r_i − μ_i)(r_m − μ_m)^2] / (σ_i · σ_m^2)
        - Residual: β_SKD = E[ε_i · ε_m^2] / (SQRT(E[ε_i^2]) · E[ε_m^2]) (Harvey and
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

    Shares similar logic:
    - PerformanceAnalytics (R), CoSkewness: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CoMoments.R#L391

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns. With a "within period"
        (period, date) Multi Index, one value is calculated per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        method (str, optional): The co-skewness. Options are: "standardized" (standardized
            co-skewness), "residual" (Harvey and Siddique's residual co-skewness). Defaults to
            "standardized".

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

        - Standardized: CoKurtosis = E[(r_i − μ_i)(r_m − μ_m)^3] / (σ_i · σ_m^3)
        - Residual: E[ε_i · ε_m^3] / (SQRT(E[ε_i^2]) · E[ε_m^2]^(3/2))

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

    Shares similar logic:
    - PerformanceAnalytics (R), CoKurtosis: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CoMoments.R#L466

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns. With a "within period"
        (period, date) Multi Index, one value is calculated per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        method (str, optional): The co-kurtosis. Options are: "standardized" (standardized
            co-kurtosis), "residual" (residual co-kurtosis). Defaults to "standardized".

    Returns:
        pd.Series | pd.DataFrame | float: The co-kurtosis per period (Multi Index), per
        column (DataFrame) or a single value (Series).
    """
    return get_co_moment(returns, benchmark_returns, power=3, method=method)


@by_own_trading_days(compound=True)
def get_r_squared(
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the R-squared of each asset against the benchmark: the share of the variance of
    its returns that the single factor (CAPM) regression on the benchmark returns explains,
    over the whole sample or within every rolling window.

    With one regressor it is the squared correlation:

        - R Squared = Covariance(Asset, Benchmark)^2 / (Variance(Asset) x Variance(Benchmark))

    Each asset is paired with the benchmark on the dates both have a return.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons.

    Referenced against:
    - quantstats, r_squared: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2828

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns, either a flat index or a
            'within period' (period, date) Multi Index to get one value per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.
        window (int | None, optional): The number of observations of a rolling window, at least 2. Defaults
            to None, which calculates the R-squared over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame | float: The R-squared per period (Multi Index), per column
        (DataFrame) or a single value (Series), or with a window at the end of every window.
        Fewer than three shared dates, or returns without variation, give NaN.
    """
    validate_window("window", window, 2)
    if window is not None:
        return (returns.rolling(window).corr(benchmark_returns) ** 2).replace(
            [np.inf, -np.inf], np.nan
        )

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

    # The degrees of freedom cancel out in the ratio.
    r_squared = (asset * benchmark).groupby(groups).sum(min_count=3) ** 2 / (
        (asset**2).groupby(groups).sum(min_count=3)
        * (benchmark**2).groupby(groups).sum(min_count=3)
    )
    r_squared = r_squared.replace([np.inf, -np.inf], np.nan)

    if within_period:
        return r_squared

    r_squared = r_squared.iloc[0]

    return r_squared if isinstance(returns, pd.DataFrame) else float(r_squared.iloc[0])


def get_outperformance_probability(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Calculate the probability of outperformance: how often the asset's compounded return beat
    the benchmark's over every window of a given number of consecutive periods, as defined in
    PerformanceAnalytics (Peterson and Carl).

    Without a benchmark the asset's compounded return is compared with 0, which for a window of
    twelve monthly returns is ffn's twelve month win percentage.

    The formula is as follows:

        - Window Return = PRODUCT(1 + Return over the window) - 1
        - Probability of Outperformance = Number of Windows with Window Return > Benchmark Window
          Return / Number of Windows in which they differ

    The windows end at every period from the first full window onwards, so consecutive
    windows overlap. A window in which both returns are equal counts as neither.

    Referenced against:
    - PerformanceAnalytics (R), table.ProbOutPerformance: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.ProbOutperformance.R#L32

    Shares similar logic:
    - ffn, PerformanceStats (twelve_month_win_perc): https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L123

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, each on its own
            history.
        benchmark_returns (pd.Series | None, optional): The benchmark returns on the same
            index, or None to compare with a return of 0. Defaults to None.
        windows (list[int] | None, optional): The window lengths in periods. Defaults to
            None, PerformanceAnalytics' 1, 3, 6, 9, 12, 18 and 36.

    Returns:
        pd.DataFrame: The probability per window length (index) and asset (columns), NaN
        for a window longer than the history.
    """
    windows = windows if windows is not None else [1, 3, 6, 9, 12, 18, 36]

    probabilities = {}

    for column in returns.columns:
        asset = returns[column]
        benchmark = (
            benchmark_returns
            if benchmark_returns is not None
            else pd.Series(0.0, index=returns.index)
        )
        # The periods both have a return, each ticker on its own history.
        shared = asset.notna() & benchmark.notna()
        asset_growth, benchmark_growth = np.log1p(asset[shared]), np.log1p(
            benchmark[shared]
        )

        column_probabilities = {}
        for window in windows:
            difference = (
                np.expm1(asset_growth.rolling(window).sum())
                - np.expm1(benchmark_growth.rolling(window).sum())
            ).dropna()
            wins, losses = (difference > 0).sum(), (difference < 0).sum()
            column_probabilities[window] = (
                wins / (wins + losses) if wins + losses else np.nan
            )

        probabilities[column] = column_probabilities

    return pd.DataFrame(probabilities).rename_axis("Window")


@by_own_trading_days(compound=True)
def get_bull_bear_beta(
    excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    market_returns: pd.Series | None = None,
    periods_per_year: float = 252.0,
    minimum_observations: int = 3,
) -> dict:
    """
    Calculate the bull and bear market Betas and Alphas and the Timing Ratio: how strongly the
    asset moved with the benchmark in rising and in falling markets.

    They come from the single factor (CAPM) regression of the asset's excess returns on the
    benchmark's excess returns, fitted separately over the observations in which the market rose
    (bull) and fell (bear).

    The formulas are as follows:

        - Excess Return = Bull Alpha + Bull Beta x Benchmark Excess Return, over Market > 0
        - Excess Return = Bear Alpha + Bear Beta x Benchmark Excess Return, over Market < 0
        - Timing Ratio = Bull Beta / Bear Beta
        - Annualized Alpha = (1 + Alpha)^P - 1

    Where the market is the benchmark excess return by default (as in PerformanceAnalytics
    and in empyrical on excess returns) or, passed as `market_returns`, another series such
    as the benchmark return itself, and P the observations in a year. A Timing Ratio above 1
    means the asset moved more with the market when it rose than when it fell, the asymmetry
    a successful market timer shows (Henriksson and Merton, 1981).

    For more information, see the following sources:

    - Henriksson, R.D., & Merton, R.C. (1981). "On Market Timing and Investment
    Performance II: Statistical Procedures for Evaluating Forecasting Skills." Journal of
    Business, 54(4), 513-533.
    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons.

    Referenced against:
    - PerformanceAnalytics (R), CAPM.beta.bull: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L133
    - PerformanceAnalytics (R), CAPM.beta.bear: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L184
    - PerformanceAnalytics (R), TimingRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.beta.R#L236
    - empyrical, up_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1986
    - empyrical, down_alpha_beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2004

    Args:
        excess_returns (pd.Series): The asset's excess returns.
        benchmark_excess_returns (pd.Series): The benchmark's excess returns on the same
            index.
        market_returns (pd.Series | None, optional): The returns whose sign splits the bull
            and bear observations. Defaults to None, the benchmark excess returns.
        periods_per_year (float, optional): The observations in a year, to annualize the
            Alphas with. Defaults to 252.
        minimum_observations (int, optional): The fewest observations a side is fitted on;
            a regression needs at least 3. Defaults to 3.

    Returns:
        dict: The Bull Alpha, Bull Beta, Bear Alpha, Bear Beta and Timing Ratio. A side with
        fewer than minimum_observations observations, or without variation, gives NaN.

    Notes:
    - The Alpha is annualized by compounding it P times, so the Alpha of a few observations
      is amplified to extremes: three daily returns in a week give an annualized Alpha in the
      trillions. A larger minimum, such as the 20 days the Toolkit uses, avoids that.
    """
    validate_window("minimum_observations", minimum_observations, 3)
    market = benchmark_excess_returns if market_returns is None else market_returns
    pairs = pd.concat(
        [excess_returns, benchmark_excess_returns, market], axis=1, keys=[0, 1, 2]
    ).dropna()

    results = {}
    for side, observations in (("Bull", pairs[2] > 0), ("Bear", pairs[2] < 0)):
        asset, benchmark = pairs.loc[observations, 0], pairs.loc[observations, 1]
        variance = ((benchmark - benchmark.mean()) ** 2).sum()

        if len(asset) < minimum_observations or variance == 0:
            results[f"{side} Alpha"], results[f"{side} Beta"] = np.nan, np.nan
            continue

        beta = (
            (benchmark - benchmark.mean()) * (asset - asset.mean())
        ).sum() / variance
        alpha = asset.mean() - beta * benchmark.mean()

        results[f"{side} Alpha"] = (1 + alpha) ** periods_per_year - 1
        results[f"{side} Beta"] = beta

    # A bear Beta of 0 (or NaN) leaves the ratio undefined.
    results["Timing Ratio"] = (
        results["Bull Beta"] / results["Bear Beta"]
        if results["Bear Beta"] != 0
        else np.nan
    )

    return results


@by_own_trading_days(compound=True)
def get_beta_fragility_heuristic(
    returns: pd.Series | pd.DataFrame, benchmark_returns: pd.Series
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the beta fragility heuristic of Taleb, Canetti, Kinda, Loukoianova and Schmieder
    (2012): how much the asset's response to the benchmark bends at its extremes.

    The returns are sorted by the benchmark return, and the asset's return at the median
    benchmark return is compared with the line through its returns at the lowest and highest
    benchmark returns:

        - H = (w_low x Return_low + w_high x Return_high) - Return_middle
        - w_low = (Benchmark_high - Benchmark_middle) / (Benchmark_high - Benchmark_low)
        - w_high = (Benchmark_middle - Benchmark_low) / (Benchmark_high - Benchmark_low)

    Where low, middle (position n // 2) and high are the observations with the lowest,
    median and highest benchmark return (both weights 0.5 when the benchmark does not move).
    A negative H means the asset's returns are concave in the market's (it loses more in
    large falls than a linear Beta suggests), a sign of fragility.

    For more information, see the following sources:

    - Taleb, N.N., Canetti, E., Kinda, T., Loukoianova, E., & Schmieder, C. (2012). "A New
    Heuristic Measure of Fragility and Tail Risks: Application to Stress Testing." IMF
    Working Paper WP/12/216.

    Referenced against:
    - empyrical, beta_fragility_heuristic: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1565

    Args:
        returns (pd.Series | pd.DataFrame): The asset returns, either a flat index or a
            'within period' (period, date) Multi Index to get one value per period.
        benchmark_returns (pd.Series): The benchmark returns on the same index.

    Returns:
        pd.Series | pd.DataFrame | float: The heuristic per period (Multi Index), per column
        (DataFrame) or a single value (Series). Fewer than three shared dates give NaN.
    """
    if not isinstance(returns, pd.DataFrame | pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        frame = returns.to_frame() if isinstance(returns, pd.Series) else returns

        return pd.DataFrame(
            {
                period: get_beta_fragility_heuristic(
                    frame.loc[period], benchmark_returns.loc[period]
                )
                for period in frame.index.get_level_values(0).unique()
            }
        ).T.rename_axis(frame.index.names[0])

    if isinstance(returns, pd.DataFrame):
        return pd.Series(
            {
                column: get_beta_fragility_heuristic(returns[column], benchmark_returns)
                for column in returns.columns
            },
            dtype=float,
        )

    pairs = pd.concat([returns, benchmark_returns], axis=1, keys=[0, 1]).dropna()

    if len(pairs) < 3:  # noqa: PLR2004
        return np.nan

    pairs = pairs.sort_values(1, kind="mergesort")
    (low_return, low), (middle_return, middle), (high_return, high) = (
        pairs.iloc[position] for position in (0, len(pairs) // 2, len(pairs) - 1)
    )
    benchmark_range = high - low
    low_weight, high_weight = (
        ((high - middle) / benchmark_range, (middle - low) / benchmark_range)
        if benchmark_range != 0
        else (0.5, 0.5)
    )

    return float(low_weight * low_return + high_weight * high_return - middle_return)


def get_stability_of_returns(
    returns: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the stability of returns: how steadily the asset grew, the R-squared of a linear
    fit of the cumulative log returns on time.

    A value close to 1 means the asset grew at a steady rate; a value close to 0 means its
    growth path was erratic. The measure is defined in empyrical (Quantopian).

    The formula is as follows:

        - Cumulative Log Return_t = SUM(LN(1 + Return)) up to t
        - Stability = R Squared of the regression of Cumulative Log Return_t on t

    Where t counts the returns of each asset (0, 1, 2, ...), missing returns left out.

    Referenced against:
    - empyrical, stability_of_timeseries: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1471

    Args:
        returns (pd.Series | pd.DataFrame): The returns, either a flat index or a 'within
            period' (period, date) Multi Index to get one value per period.

    Returns:
        pd.Series | pd.DataFrame | float: The stability per period (Multi Index), per column
        (DataFrame) or a single value (Series). Fewer than two returns give NaN.
    """
    if not isinstance(returns, pd.DataFrame | pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        return returns.groupby(level=0).agg(
            lambda values: get_stability_of_returns(values.droplevel(0))
        )

    if isinstance(returns, pd.DataFrame):
        return returns.apply(get_stability_of_returns).astype(float)

    cumulative_log_return = np.log1p(returns.dropna()).cumsum()

    if len(cumulative_log_return) < 2:  # noqa: PLR2004
        return np.nan

    return float(
        np.corrcoef(np.arange(len(cumulative_log_return)), cumulative_log_return)[0, 1]
        ** 2
    )


@by_own_trading_days(compound=True)
def get_risk_decomposition(
    excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    observations: float = 1.0,
) -> dict:
    """
    Calculate the systematic, specific and total risk of an asset: its volatility split into
    the part the single factor (CAPM) regression on the benchmark explains and the part it
    leaves in the residuals.

    The formulas are as follows:

        - Systematic Risk = Beta x Standard Deviation(Benchmark Excess Return) x SQRT(N)
        - Specific Risk = Standard Deviation(Residuals) x SQRT(N)
        - Total Risk = SQRT(Systematic Risk^2 + Specific Risk^2)

    Where the Residuals are those of the regression Excess Return = Alpha + Beta x Benchmark
    Excess Return, the standard deviations are sample ones (with n - 1 in the denominator) over the dates both have
    a return, and N the number of observations to scale them to (such as 252 for a year of
    daily returns). With the same degrees of freedom the Total Risk equals the volatility of
    the excess returns, as the regression splits the variance exactly.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, pp. 71-72.

    Referenced against:
    - PerformanceAnalytics (R), SystematicRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SystematicRisk.R#L36
    - PerformanceAnalytics (R), SpecificRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SpecificRisk.R#L27

    Shares similar logic:
    - PerformanceAnalytics (R), TotalRisk: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/TotalRisk.R#L30

    Args:
        excess_returns (pd.Series): The asset's excess returns.
        benchmark_excess_returns (pd.Series): The benchmark's excess returns on the same
            index.
        observations (float, optional): The number of observations to scale the risks to.
            Defaults to 1, the risk per observation.

    Returns:
        dict: The Systematic Risk, Specific Risk and Total Risk. Fewer than three shared
        dates, or a benchmark without variation, give NaN.

    Notes:
    - PerformanceAnalytics' SpecificRisk divides the residual variance by n where the standard
    deviations here divide by n - 1.
    """
    pairs = pd.concat(
        [excess_returns, benchmark_excess_returns], axis=1, keys=[0, 1]
    ).dropna()
    asset, benchmark = pairs[0], pairs[1]
    benchmark_variance = benchmark.var()

    if len(pairs) < 3 or not benchmark_variance > 0:  # noqa: PLR2004
        return {
            "Systematic Risk": np.nan,
            "Specific Risk": np.nan,
            "Total Risk": np.nan,
        }

    beta = asset.cov(benchmark) / benchmark_variance
    residuals = asset - beta * benchmark
    systematic_risk = beta * np.sqrt(benchmark_variance * observations)
    specific_risk = np.sqrt(residuals.var() * observations)

    return {
        "Systematic Risk": systematic_risk,
        "Specific Risk": specific_risk,
        "Total Risk": np.sqrt(systematic_risk**2 + specific_risk**2),
    }


@by_own_trading_days(compound=True)
def get_capital_market_line(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_returns: pd.Series,
) -> dict:
    """
    Calculate the Capital Market Line and the Security Market Line of the benchmark: the return
    an efficient portfolio earns for the asset's total risk, and the return the CAPM gives per
    unit of Beta (Sharpe, 1964).

    The formulas are as follows:

        - Capital Market Line Slope = Mean(Benchmark Excess Return) / Standard Deviation(Benchmark Return)
        - Capital Market Line Return = Mean(Risk Free Rate) + Capital Market Line Slope x
          Standard Deviation(Asset Return)
        - Security Market Line Slope = Mean(Benchmark Excess Return)

    All per observation (such as per day), over the dates the asset and the benchmark both
    have a return. The CML slope is the benchmark's Sharpe ratio; the SML slope is the market
    risk premium, the return the CAPM adds per unit of Beta. An asset whose mean return lies
    below its CML return earned less than a mix of the benchmark and the risk free asset with
    the same volatility.

    For more information, see the following sources:

    - Sharpe, W.F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under
    Conditions of Risk." Journal of Finance, 19(3), 425-442.

    Referenced against:
    - PerformanceAnalytics (R), CAPM.CML.slope: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.utils.R#L6
    - PerformanceAnalytics (R), CAPM.SML.slope: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.utils.R#L152

    Args:
        returns (pd.Series): The asset's returns.
        benchmark_returns (pd.Series): The benchmark's returns on the same index.
        risk_free_returns (pd.Series): The risk free return per observation on the same
            index.

    Returns:
        dict: The Capital Market Line Slope, Capital Market Line Return and Security Market
        Line Slope. Fewer than two shared dates, or a benchmark without variation, give NaN.

    Notes:
    - The Security Market Line Slope is the reciprocal of PerformanceAnalytics' CAPM.SML.slope,
    which gives the Beta per unit of return.
    """
    pairs = pd.concat(
        [returns, benchmark_returns, risk_free_returns], axis=1, keys=[0, 1, 2]
    ).dropna()
    benchmark_volatility = pairs[1].std()
    market_risk_premium = (pairs[1] - pairs[2]).mean()

    if len(pairs) < 2 or not benchmark_volatility > 0:  # noqa: PLR2004
        return {
            "Capital Market Line Slope": np.nan,
            "Capital Market Line Return": np.nan,
            "Security Market Line Slope": np.nan,
        }

    slope = market_risk_premium / benchmark_volatility

    return {
        "Capital Market Line Slope": slope,
        "Capital Market Line Return": pairs[2].mean() + slope * pairs[0].std(),
        "Security Market Line Slope": market_risk_premium,
    }


@by_own_trading_days(compound=True)
def get_regression_epsilon(
    excess_returns: pd.Series,
    benchmark_excess_returns: pd.Series,
    period_excess_return: float,
    benchmark_period_excess_return: float,
) -> float:
    """
    Calculate the regression epsilon of Bacon (2008): the part of the asset's return over a
    period that its single factor (CAPM) regression does not predict.

    The formula is as follows:

        - Regression Epsilon = Period Excess Return - Alpha - Beta x Benchmark Period Excess Return

    Where Alpha and Beta come from the regression of the asset's excess returns per
    observation (such as per day) on those of the benchmark, and the period excess returns
    are compounded over the period (such as the year), as PerformanceAnalytics combines the
    annualized returns with the regression Alpha per observation.

    For more information, see the following sources:

    - Bacon, C.R. (2008). "Practical Portfolio Performance Measurement and Attribution."
    2nd edition, John Wiley & Sons, p. 71.

    Referenced against:
    - PerformanceAnalytics (R), CAPM.epsilon: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/CAPM.epsilon.R#L33

    Args:
        excess_returns (pd.Series): The asset's excess returns per observation.
        benchmark_excess_returns (pd.Series): The benchmark's excess returns per observation
            on the same index.
        period_excess_return (float): The asset's excess return over the period.
        benchmark_period_excess_return (float): The benchmark's excess return over the period.

    Returns:
        float: The regression epsilon. Fewer than three shared observations, or a benchmark
        without variation, give NaN.
    """
    pairs = pd.concat(
        [excess_returns, benchmark_excess_returns], axis=1, keys=[0, 1]
    ).dropna()
    benchmark_variance = pairs[1].var()

    if len(pairs) < 3 or not benchmark_variance > 0:  # noqa: PLR2004
        return np.nan

    beta = pairs[0].cov(pairs[1]) / benchmark_variance
    alpha = pairs[0].mean() - beta * pairs[1].mean()

    return float(period_excess_return - alpha - beta * benchmark_period_excess_return)


def get_log_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculate log (continuously compounded) returns: the simple returns converted so that they
    add up over time.

    The formula is as follows:

        - Log Return = ln(1 + Return) = ln(Price_t / Price_t-1)

    Log returns add up over time where simple returns compound: the sum of the log returns of
    the days in a year is the log return of the year.

    For more information, see the following sources:

    - Campbell, J.Y., Lo, A.W., & MacKinlay, A.C. (1997). "The Econometrics of Financial
    Markets." Princeton University Press, chapter 1.4.

    Shares similar logic:
    - ffn, to_log_returns: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1223
    - pandas-ta-classic, log_return: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/performance/log_return.py#L12

    Args:
        returns (pd.Series | pd.DataFrame): The simple returns.

    Returns:
        pd.Series | pd.DataFrame: The log returns, NaN where the return is missing.
    """
    return np.log1p(returns)


def get_geltner_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculate Geltner's (1991, 1993) unsmoothed returns: the returns of an appraisal-based or
    illiquid asset with the smoothing of its first order autocorrelation removed.

    Appraisal-based and illiquid assets (real estate, private equity, many hedge funds)
    report returns that are smoothed: each return carries part of the earlier ones, which
    shows up as positive autocorrelation and understates the volatility. Geltner reverses a
    first order smoothing:

        - Unsmoothed Return_t = (Return_t - ρ1 x Return_t-1) / (1 - ρ1)

    Where ρ1 is the first order autocorrelation of each column's own returns (the sample
    autocorrelation of `financetoolkit.risk.risk_model.get_autocorrelation`, deviations from
    the mean divided by the sum of squared deviations). The first return has no predecessor
    and is NaN.

    For more information, see the following sources:

    - Geltner, D. (1991). "Smoothing in Appraisal-Based Returns." Journal of Real Estate
    Finance and Economics, 4(3), 327-345.
    - Geltner, D. (1993). "Estimating Market Values from Appraised Values without Assuming an
    Efficient Market." Journal of Real Estate Research, 8(3), 325-345.

    Referenced against:
    - PerformanceAnalytics (R), Return.Geltner: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Return.Geltner.R#L43

    Args:
        returns (pd.Series | pd.DataFrame): The returns, each column on its own history
            (missing returns are left out).

    Returns:
        pd.Series | pd.DataFrame: The unsmoothed returns on the same index.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(get_geltner_returns)

    observed = returns.dropna()

    if len(observed) < 3:  # noqa: PLR2004
        return returns * np.nan

    autocorrelation = get_autocorrelation(observed, lags=1).iloc[0]
    unsmoothed = (observed - autocorrelation * observed.shift(1)) / (
        1 - autocorrelation
    )

    return unsmoothed.reindex(returns.index)


def get_smoothing_index(returns: pd.Series, order: int = 2) -> pd.Series:
    """
    Calculate the Smoothing Index of Getmansky, Lo and Makarov (2004): how much reported returns
    are smoothed over past periods, with the smoothing coefficients.

    Getmansky, Lo and Makarov model an observed return as a weighted average of the true
    returns of the current and the `order` previous periods, which is what reported returns of
    illiquid assets look like:

        - Observed Return_t = θ0 x True Return_t + θ1 x True Return_t-1 + ... + θk x True Return_t-k
        - θ0 + θ1 + ... + θk = 1
        - Smoothing Index ξ = θ0² + θ1² + ... + θk²

    The weights are those of a moving average model of order k fitted by exact maximum
    likelihood (a Gaussian MA(k) without a mean, invertible), divided by 1 + MA1 + ... + MAk
    so they add up to 1. A Smoothing Index of 1 means no smoothing (θ0 = 1); the lower it
    is, the more the returns are smoothed, down to 1 / (k + 1) when every period has the same
    weight.

    For more information, see the following sources:

    - Getmansky, M., Lo, A.W., & Makarov, I. (2004). "An Econometric Model of Serial
    Correlation and Illiquidity in Hedge Fund Returns." Journal of Financial Economics, 74(3),
    529-609.

    The likelihood is that of the returns as given: Getmansky, Lo and Makarov fit the model to
    de-meaned returns, so subtract the mean first to follow them. Negative moving average
    coefficients are set to 0 before the weights are normalized, as PerformanceAnalytics does
    by default, since a negative weight on a past return is no smoothing.

    Referenced against:
    - statsmodels, ARIMA (order (0, 0, k), without a mean): https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html

    Shares similar logic:
    - PerformanceAnalytics (R), SmoothingIndex: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SmoothingIndex.R#L96

    Args:
        returns (pd.Series): The returns, usually monthly (missing returns are left out).
        order (int, optional): The number of lagged periods k. Defaults to 2.

    Returns:
        pd.Series: The weights θ0 to θk ("Theta 0", "Theta 1", ...) and the "Smoothing
        Index". Fewer than 2(k + 1) returns, or returns without variation, give NaN.
    """
    if order < 1 or int(order) != order:
        raise ValueError("The order must be a positive integer.")

    labels = [f"Theta {lag}" for lag in range(order + 1)] + ["Smoothing Index"]
    values = returns.dropna().to_numpy(dtype=float)
    observations = len(values)

    if observations < 2 * (order + 1) or not np.ptp(values) > 0:
        return pd.Series(np.nan, index=labels)

    def negative_log_likelihood(coefficients: np.ndarray) -> float:
        # The autocovariances of an MA(k) process with unit innovation variance, as the
        # banded (upper) form of its covariance matrix.
        weights = np.concatenate([[1.0], coefficients])
        bands = np.zeros((order + 1, observations))

        for lag in range(order + 1):
            bands[order - lag, lag:] = weights[: order + 1 - lag] @ weights[lag:]

        try:
            cholesky = linalg.cholesky_banded(bands)
        except linalg.LinAlgError:
            return np.inf

        quadratic_form = values @ linalg.cho_solve_banded((cholesky, False), values)
        log_determinant = 2 * np.log(cholesky[-1]).sum()

        # The innovation variance is concentrated out: its estimate is the quadratic form / n.
        return 0.5 * (
            observations * np.log(quadratic_form / observations) + log_determinant
        )

    fit = optimize.minimize(
        negative_log_likelihood,
        np.zeros(order),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000 * order},
    )

    # The likelihood is the same for a non-invertible MA polynomial and the invertible one
    # with its roots inside the unit circle mirrored outside, which is the one reported.
    roots = np.roots(np.concatenate([[1.0], fit.x])[::-1])
    roots = np.where(np.abs(roots) < 1, 1 / roots, roots)
    polynomial = np.real(np.poly(roots))
    coefficients = (polynomial / polynomial[-1])[::-1][1:]

    weights = np.concatenate([[1.0], np.clip(coefficients, 0, None)])
    weights = weights / weights.sum()

    return pd.Series(np.append(weights, (weights**2).sum()), index=labels)


def get_serenity_index(
    excess_returns: pd.Series | pd.DataFrame,
    ulcer_index: pd.Series | pd.DataFrame,
    conditional_drawdown_at_risk: pd.Series | pd.DataFrame,
    volatility: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Serenity Index (Serenity Ratio) of KeyQuant: the excess return per unit of
    Ulcer Index, penalized further when the drawdowns have a fat tail compared to the volatility
    of the returns.

    The formula is as follows:

        - Pitfall Indicator = |Conditional Drawdown at Risk| / Volatility
        - Serenity Index = Excess Return / (Ulcer Index x Pitfall Indicator)

    The Ulcer Index measures the depth and duration of the drawdowns, the Conditional Drawdown
    at Risk the average of the worst drawdowns and the Volatility the standard deviation of
    the returns over the same horizon as the Excess Return, so the Pitfall Indicator is the
    number of standard deviations the worst drawdowns reach. It equals the Ulcer Performance
    Index divided by the Pitfall Indicator.

    For more information, see the following sources:

    - KeyQuant (2017). "Assessing Performance Through Risk: Part 1." KeyQuant white paper
    (the Serenity Ratio and the Pitfall Indicator).
    - Martin, P.G., & McCann, B.B. (1989). "The Investor's Guide to Fidelity Funds." John
    Wiley & Sons (the Ulcer Index).

    A period without a drawdown leaves the index undefined (NaN).

    Shares similar logic:
    - quantstats, serenity_index: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L1906

    Args:
        excess_returns (pd.Series | pd.DataFrame): The excess returns, one per period.
        ulcer_index (pd.Series | pd.DataFrame): The Ulcer Index of the same periods (see
            `financetoolkit.risk.risk_model.get_ui`).
        conditional_drawdown_at_risk (pd.Series | pd.DataFrame): The Conditional Drawdown at
            Risk of the same periods (see
            `financetoolkit.risk.risk_model.get_conditional_drawdown_at_risk`).
        volatility (pd.Series | pd.DataFrame): The volatility of the returns of the same
            periods, on the horizon of the excess returns.

    Returns:
        pd.Series | pd.DataFrame | float: The Serenity Index values.
    """
    pitfall_indicator = abs(conditional_drawdown_at_risk) / volatility
    serenity_index = remove_undefined_ratios(
        excess_returns / (ulcer_index * pitfall_indicator)
    )

    if isinstance(serenity_index, pd.Series | pd.DataFrame):
        return serenity_index.dropna(how="all")

    return serenity_index


# The risk measures of get_risk_measure, each a loss or dispersion of the returns.
RISK_MEASURES = [
    "var",
    "cvar",
    "evar",
    "erm",
    "worst",
    "mad",
    "lpm",
    "gmd",
    "max_drawdown",
    "average_drawdown",
    "dar",
    "cdar",
    "edar",
    "ulcer",
]


def get_risk_measure(
    returns: pd.Series | pd.DataFrame, risk_measure: str = "cvar", alpha: float = 0.05
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate a risk measure as a positive number: how much the returns are at risk by one of
    the measures of the Toolkit's risk module, the denominator of `get_risk_measure_ratio`.

    The risk measures are:

    - Value at Risk: the historical Value at Risk, the alpha quantile of the returns.
    - Conditional Value at Risk: the historical Conditional Value at Risk, the mean of the
      returns at or below it.
    - Entropic Value at Risk: the historical Entropic Value at Risk.
    - Entropic Risk Measure: the Entropic Risk Measure with a risk aversion of 1.
    - Worst Realization: the lowest return.
    - Mean Absolute Deviation: the Mean Absolute Deviation from the mean.
    - Lower Partial Moment: the first Lower Partial Moment below the mean,
      mean(max(Mean - Return, 0)).
    - Gini Mean Difference: the Gini Mean Difference.
    - Maximum Drawdown: the Maximum Drawdown.
    - Average Drawdown: the Average Drawdown over every observation.
    - Drawdown at Risk: the alpha quantile of the drawdowns.
    - Conditional Drawdown at Risk: the Conditional Drawdown at Risk.
    - Entropic Drawdown at Risk: the Entropic Drawdown at Risk.
    - Ulcer Index: the root mean square drawdown.

    The drawdowns compound the returns and start from a wealth of 1 before the first return.

    Args:
        returns (pd.Series | pd.DataFrame): The returns, optionally with a (period, date)
            index for one value per period.
        risk_measure (str, optional): The risk measure. Options are: "var" (historical Value at
            Risk), "cvar" (historical Conditional Value at Risk), "evar" (historical Entropic Value
            at Risk), "erm" (Entropic Risk Measure), "worst" (Worst Realization), "mad" (Mean
            Absolute Deviation), "lpm" (first Lower Partial Moment), "gmd" (Gini Mean Difference),
            "max_drawdown" (Maximum Drawdown), "average_drawdown" (Average Drawdown), "dar"
            (Drawdown at Risk), "cdar" (Conditional Drawdown at Risk), "edar" (Entropic Drawdown
            at Risk), "ulcer" (Ulcer Index). Defaults to "cvar".
        alpha (float, optional): The tail probability of the VaR, CVaR, EVaR, ERM, DaR,
            CDaR and EDaR. Defaults to 0.05.

    Returns:
        pd.Series | pd.DataFrame | float: The risk measure as a positive number, one per
        column, or per period and column for a (period, date) index.
    """
    check_option("risk_measure", risk_measure, RISK_MEASURES)

    if risk_measure == "mad":
        groups = (
            returns.index.get_level_values(0)
            if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
            else np.zeros(len(returns))
        )
        result = get_mean_absolute_deviation(returns, "yearly", groups=groups)

        return (
            result
            if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
            else result.iloc[0]
        )

    measures = {
        "var": lambda: var_model.get_var_historic(returns, alpha),
        "cvar": lambda: cvar_model.get_cvar_historic(returns, alpha),
        "evar": lambda: evar_model.get_evar_historic(returns, alpha),
        "erm": lambda: evar_model.get_entropic_risk_measure(returns, 1.0, alpha),
        "worst": lambda: dispersion_model.get_worst_realization(returns),
        "lpm": lambda: get_partial_moment(returns, 1, "mean"),
        "gmd": lambda: dispersion_model.get_gini_mean_difference(returns),
        "max_drawdown": lambda: drawdown_model.get_maximum_drawdown(returns),
        "average_drawdown": lambda: drawdown_model.get_average_drawdown(
            returns, "observation"
        ),
        "dar": lambda: drawdown_model.get_drawdown_at_risk(returns, alpha),
        "cdar": lambda: get_conditional_drawdown_at_risk(returns, alpha),
        "edar": lambda: drawdown_model.get_entropic_drawdown_at_risk(returns, alpha),
        "ulcer": lambda: get_ui(returns, rolling=None),
    }

    return abs(measures[risk_measure]())


def get_risk_measure_ratio(
    excess_returns: pd.Series | pd.DataFrame,
    returns: pd.Series | pd.DataFrame,
    risk_measure: str = "cvar",
    alpha: float = 0.05,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the risk measure ratio: the mean excess return per unit of a risk measure, the
    Sharpe ratio with any risk measure of `get_risk_measure` in place of the standard deviation.

    The formula is as follows:

        - Risk Measure Ratio = Mean(Return - Risk Free Rate) / Risk Measure(Return)

    With the standard deviation this is the Sharpe ratio, with the Conditional Value at Risk
    the Conditional Sharpe (STARR) ratio and with the Maximum Drawdown a Calmar ratio. Both
    the mean and the risk measure are per observation (e.g. per day), not annualized.

    For more information, see the following sources:

    - Cajas, D. (2021). "Riskfolio-Lib: Portfolio Optimization in Python", the risk-adjusted
    return ratios.
    - Gregoriou, G.N., & Gueyie, J.P. (2003). "Risk-Adjusted Performance of Funds of Hedge Funds
    Using a Modified Sharpe Ratio." Journal of Wealth Management, 6(3), 77-83.

    A risk measure of 0 leaves the ratio undefined (NaN).

    Referenced against:
    - skfolio, RatioMeasure: https://skfolio.org/generated/skfolio.measures.RatioMeasure.html

    Shares similar logic:
    - Riskfolio-Lib, Sharpe: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Sharpe
    - PerformanceAnalytics (R), SharpeRatio (VaR and ES): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SharpeRatio.R#L80

    Args:
        excess_returns (pd.Series | pd.DataFrame): The returns minus the risk free rate, on the
            same index as `returns`.
        returns (pd.Series | pd.DataFrame): The returns the risk measure is taken of, optionally
            with a (period, date) index for one ratio per period.
        risk_measure (str, optional): The risk measure. Options are: "var" (historical Value at
            Risk), "cvar" (historical Conditional Value at Risk), "evar" (historical Entropic Value
            at Risk), "erm" (Entropic Risk Measure), "worst" (Worst Realization), "mad" (Mean
            Absolute Deviation), "lpm" (first Lower Partial Moment), "gmd" (Gini Mean Difference),
            "max_drawdown" (Maximum Drawdown), "average_drawdown" (Average Drawdown), "dar"
            (Drawdown at Risk), "cdar" (Conditional Drawdown at Risk), "edar" (Entropic Drawdown
            at Risk), "ulcer" (Ulcer Index). Defaults to "cvar".
        alpha (float, optional): The tail probability of the tail and drawdown measures.
            Defaults to 0.05.

    Returns:
        pd.Series | pd.DataFrame | float: The ratio, one per column, or per period and column
        for a (period, date) index.
    """
    mean_excess_return = (
        excess_returns.groupby(level=0, sort=False).mean()
        if excess_returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
        else excess_returns.mean()
    )

    if isinstance(mean_excess_return, pd.DataFrame):
        mean_excess_return.index.name = None

    return remove_undefined_ratios(
        mean_excess_return / get_risk_measure(returns, risk_measure, alpha)
    )


# The trailing windows of get_trailing_returns in months, None for the periods to date.
TRAILING_WINDOWS = {
    "Month to Date": None,
    "3 Months": 3,
    "6 Months": 6,
    "Year to Date": None,
    "1 Year": 12,
    "3 Years (Annualized)": 36,
    "5 Years (Annualized)": 60,
    "10 Years (Annualized)": 120,
}

# Windows of at least this many months are annualized.
ANNUALIZED_WINDOW_MONTHS = 36


def get_trailing_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Calculate the trailing returns: how much the prices gained up to their last close over the
    month to date, the last 3 and 6 months, year to date, the last year, the last 3, 5 and 10
    years (annualized) and the whole history (annualized).

    The formula is as follows:

        - Trailing Return = Last Close / Close at the Start of the Window - 1
        - Annualized Trailing Return = (1 + Trailing Return) ^ (365.25 / Calendar Days) - 1

    The close at the start of a window is the last close on or before the date that lies the
    window before the last close (e.g. 3 months before). When the last close falls on the last
    business day of its month, the windows run from month end to month end instead (a history
    up to 30 June has its 3 month return from the close of 31 March), as ffn does. Month to
    date and year to date start from the last close of the previous month and year. The
    returns of 3 years and longer are annualized over the calendar days of the window, and the
    return since inception is the Compound Annual Growth Rate of the whole history (see
    `get_compound_annual_growth_rate`).

    A history that does not reach back to the start of a window gives NaN for it.

    Referenced against:
    - ffn, PerformanceStats: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L123

    Shares similar logic:
    - ffn, calc_mtd: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1426
    - ffn, calc_ytd: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1437
    - PerformanceAnalytics (R), table.TrailingPeriods: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.RollingPeriods.R#L36

    Args:
        prices (pd.Series | pd.DataFrame): The prices (or total return index) with a
            DatetimeIndex or PeriodIndex, each column on its own history.

    Returns:
        pd.Series | pd.DataFrame: The trailing returns as the index, one column per column of
        `prices` (a Series for a Series).
    """
    if isinstance(prices, pd.DataFrame):
        return pd.DataFrame(
            {column: get_trailing_returns(prices[column]) for column in prices.columns}
        )

    observed = prices.dropna()
    labels = [*TRAILING_WINDOWS, "Since Inception (Annualized)"]

    if len(observed) < 2:  # noqa: PLR2004
        return pd.Series(np.nan, index=labels, name=prices.name)

    if isinstance(observed.index, pd.PeriodIndex):
        observed.index = observed.index.to_timestamp()

    end = observed.index[-1]
    last_close = observed.iloc[-1]
    at_month_end = (
        end.normalize()
        >= pd.offsets.BMonthEnd().rollback(end + pd.offsets.MonthEnd(0)).normalize()
    )
    month_start = end.to_period("M").to_timestamp()

    trailing_returns = {}

    for label, months in TRAILING_WINDOWS.items():
        if months is None:
            cutoff = (
                month_start
                if label == "Month to Date"
                else pd.Timestamp(end.year, 1, 1)
            ) - pd.Timedelta(days=1)
        elif at_month_end:
            cutoff = (
                month_start - pd.DateOffset(months=months - 1) - pd.Timedelta(days=1)
            )
        else:
            cutoff = end - pd.DateOffset(months=months)

        start = observed.loc[:cutoff]

        if start.empty:
            trailing_returns[label] = np.nan
            continue

        trailing_return = last_close / start.iloc[-1] - 1

        if months is not None and months >= ANNUALIZED_WINDOW_MONTHS:
            years = (end - start.index[-1]).days / DAYS_PER_YEAR
            trailing_return = (1 + trailing_return) ** (1 / years) - 1

        trailing_returns[label] = trailing_return

    trailing_returns["Since Inception (Annualized)"] = get_compound_annual_growth_rate(
        observed
    )

    return pd.Series(trailing_returns, name=prices.name)


# The month names of the columns of get_calendar_returns.
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def get_calendar_returns(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the calendar returns: the compounded return of every month, in a table with the
    years as the rows and the months as the columns, and the compounded return of the year.

    The formula is as follows:

        - Month Return = ((1 + Return 1) x (1 + Return 2) x ... x (1 + Return N)) - 1
        - Year Return = ((1 + Month Return 1) x ... x (1 + Month Return 12)) - 1

    A month without any return (before a listing or after the end of the data) is NaN, and
    the year compounds the months it has.

    Shares similar logic:
    - quantstats, monthly_returns: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L3190
    - ffn, PerformanceStats (return_table): https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L123
    - PerformanceAnalytics (R), table.CalendarReturns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/table.CalendarReturns.R#L41

    Args:
        returns (pd.Series | pd.DataFrame): The daily (or any higher frequency than monthly)
            returns with a DatetimeIndex or PeriodIndex.

    Returns:
        pd.DataFrame: The years as the index and the months and "Year" as the columns, with
        the column of `returns` as the first column level for a DataFrame.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.concat(
            {
                column: get_calendar_returns(returns[column])
                for column in returns.columns
            },
            axis=1,
        )

    if isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.set_axis(returns.index.to_period("D"))

    monthly_returns = get_returns(returns, "monthly")
    months = to_period_index(monthly_returns.index)

    calendar = pd.DataFrame(
        {
            "Year": months.year,
            "Month": pd.Categorical(
                [MONTH_NAMES[month - 1] for month in months.month],
                categories=MONTH_NAMES,
            ),
            "Return": monthly_returns.to_numpy(),
        }
    ).pivot_table(
        index="Year", columns="Month", values="Return", aggfunc="first", observed=False
    )
    calendar = calendar.reindex(columns=MONTH_NAMES)
    calendar.columns = pd.Index(MONTH_NAMES)
    calendar["Year"] = (1 + calendar[MONTH_NAMES]).prod(axis=1, min_count=1) - 1
    calendar.index = pd.PeriodIndex(calendar.index, freq="Y")
    calendar.index.name = None

    return calendar.dropna(how="all")


# The ways get_bootstrap_samples draws the returns of a simulation.
BOOTSTRAP_METHODS = ["resample", "shuffle"]


def get_bootstrap_samples(
    observations: int,
    simulations: int = 1000,
    method: str = "resample",
    length: int | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Draw the positions of the returns of every simulation of a bootstrap: the histories that
    could have happened, by resampling or shuffling the returns.

    The bootstraps are:

    - Resampling draws `length` positions with replacement (Efron's bootstrap), so a return
    can appear several times or not at all and every statistic varies across simulations.
    - Shuffling puts the positions in a random order without replacement (a permutation), so
    every simulation holds the same returns: the cumulative return, the volatility and the
    Sharpe ratio are those of the history, and only the statistics that depend on the order,
    such as the drawdowns, vary.

    Args:
        observations (int): The number of returns to draw from.
        simulations (int, optional): The number of simulations. Defaults to 1000.
        method (str, optional): The bootstrap. Options are: "resample" (draws with replacement),
            "shuffle" (a random order without replacement). Defaults to "resample".
        length (int | None, optional): The number of returns of each simulation, only for
            resampling. Defaults to None, the number of observations.
        seed (int | None, optional): The seed of the random numbers, for reproducible results.
            Defaults to 0.

    Returns:
        np.ndarray: The positions, one simulation per row.
    """
    check_option("method", method, BOOTSTRAP_METHODS)
    if simulations < 1:
        raise ValueError("The number of simulations must be at least 1.")

    rng = np.random.default_rng(seed)

    if method == "shuffle":
        return rng.permuted(np.tile(np.arange(observations), (simulations, 1)), axis=1)

    return rng.integers(0, observations, size=(simulations, length or observations))


# The number of simulations get_bootstrap_statistics and get_forecast_cone hold at once: the
# paths of 20,000 simulations of ten years of daily returns take 400 MB per array otherwise.
BOOTSTRAP_CHUNK = 1000


def _get_bootstrap_chunks(
    observations: int,
    simulations: int,
    method: str = "resample",
    length: int | None = None,
    seed: int | None = 0,
):
    """
    Yield the positions of `get_bootstrap_samples` in blocks of at most BOOTSTRAP_CHUNK
    simulations. The blocks are drawn one after the other from one generator, which gives
    the same rows as drawing all simulations at once (numpy draws the rows of `integers` and
    `permuted` in order), so the results do not depend on the block size.
    """
    check_option("method", method, BOOTSTRAP_METHODS)
    if simulations < 1:
        raise ValueError("The number of simulations must be at least 1.")

    rng = np.random.default_rng(seed)

    for first in range(0, simulations, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, simulations - first)
        if method == "shuffle":
            yield rng.permuted(np.tile(np.arange(observations), (size, 1)), axis=1)
        else:
            yield rng.integers(0, observations, size=(size, length or observations))


# The percentiles of the bootstrap distributions in get_bootstrap_statistics.
BOOTSTRAP_PERCENTILES = [5, 25, 75, 95]


def get_bootstrap_statistics(
    returns: pd.Series,
    excess_returns: pd.Series,
    observations_per_year: float,
    n_bootstrap: int = 1000,
    method: str = "resample",
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Calculate bootstrap statistics: how much the cumulative return, annualized return,
    volatility, Sharpe ratio and maximum drawdown of one history could have differed by chance,
    from bootstrapped histories of the returns.

    Every simulation draws a history of the same length from the returns (see
    `get_bootstrap_samples`, with the excess return of the same date drawn with each return)
    and calculates:

        - Cumulative Return = PRODUCT(1 + Return) - 1
        - Annualized Return = (1 + Cumulative Return) ^ (P / N) - 1
        - Annualized Volatility = STDEV(Return) x SQRT(P)
        - Sharpe Ratio = MEAN(Excess Return) / STDEV(Excess Return) x SQRT(P)
        - Maximum Drawdown = MIN(Wealth / Running Peak of Wealth - 1), the wealth starting at 1

    Where N is the number of returns and P the returns in a year. Each statistic is summarized
    by the mean, the median, the standard deviation (with n - 1 in the denominator) and the 5th, 25th, 75th and 95th
    percentiles over the resamples.

    For more information, see the following sources:

    - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." The Annals of
    Statistics, 7(1), 1-26.

    pyfolio resamples with replacement, quantstats shuffles the returns.

    Shares similar logic:
    - pyfolio, perf_stats_bootstrap: https://github.com/quantopian/pyfolio/blob/0.9.2/pyfolio/timeseries.py#L742
    - quantstats, montecarlo: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L3428
    - ffn, resample_returns: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3293

    Args:
        returns (pd.Series): The returns (missing returns are left out).
        excess_returns (pd.Series): The excess returns on the same index.
        observations_per_year (float): The number of returns in a year, such as 252.
        n_bootstrap (int, optional): The number of bootstrap resamples. Defaults to 1000.
        method (str, optional): The bootstrap, see `get_bootstrap_samples`. Options are:
            "resample" (draws with replacement), "shuffle" (a random order without replacement).
            Defaults to "resample".
        seed (int | None, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        pd.DataFrame: The statistics as the index and the summaries (Mean, Median, Std, 5%,
        25%, 75%, 95%) as the columns. Fewer than two returns give NaN.
    """
    statistics = [
        "Cumulative Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
    ]
    summaries = ["Mean", "Median", "Std"] + [f"{p}%" for p in BOOTSTRAP_PERCENTILES]

    pairs = pd.concat([returns, excess_returns], axis=1).dropna()

    if len(pairs) < 2:  # noqa: PLR2004
        return pd.DataFrame(np.nan, index=statistics, columns=summaries)

    return_values = pairs.iloc[:, 0].to_numpy()
    excess_values = pairs.iloc[:, 1].to_numpy()
    chunks: dict[str, list[np.ndarray]] = {statistic: [] for statistic in statistics}

    # Every statistic is one number per simulation, so the resamples are calculated a
    # block at a time and only those numbers are kept.
    for positions in _get_bootstrap_chunks(len(pairs), n_bootstrap, method, seed=seed):
        samples = return_values[positions]
        excess_samples = excess_values[positions]

        wealth = np.cumprod(1 + samples, axis=1)
        cumulative_return = wealth[:, -1] - 1

        with np.errstate(divide="ignore", invalid="ignore"):
            chunks["Cumulative Return"].append(cumulative_return)
            chunks["Annualized Return"].append(
                (1 + cumulative_return) ** (observations_per_year / samples.shape[1])
                - 1
            )
            chunks["Annualized Volatility"].append(
                samples.std(axis=1, ddof=1) * np.sqrt(observations_per_year)
            )
            chunks["Sharpe Ratio"].append(
                excess_samples.mean(axis=1)
                / excess_samples.std(axis=1, ddof=1)
                * np.sqrt(observations_per_year)
            )
            chunks["Maximum Drawdown"].append(
                (wealth / np.maximum(np.maximum.accumulate(wealth, axis=1), 1) - 1).min(
                    axis=1
                )
            )

    simulated = {
        statistic: np.concatenate(chunks[statistic]) for statistic in statistics
    }

    summary = {}

    for statistic, simulated_values in simulated.items():
        values = (
            remove_undefined_ratios(pd.Series(simulated_values)).dropna().to_numpy()
        )
        summary[statistic] = (
            [
                values.mean(),
                np.median(values),
                values.std(ddof=1) if len(values) > 1 else np.nan,
                *np.percentile(values, BOOTSTRAP_PERCENTILES),
            ]
            if len(values)
            else [np.nan] * len(summaries)
        )

    return pd.DataFrame(summary, index=summaries).T


def get_forecast_cone(
    returns: pd.Series,
    horizon: int = 252,
    standard_deviations: list[float] | None = None,
    simulations: int = 1000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Calculate a forecast cone: the range the growth of 1 may take over the next `horizon`
    periods, the mean of bootstrapped paths of the returns and the bounds a number of standard
    deviations above and below it.

    Every simulation draws `horizon` returns with replacement from the history (see
    `get_bootstrap_samples`) and compounds them into a path of wealth starting at 1:

        - Wealth_t = PRODUCT(1 + Simulated Return_1 ... Simulated Return_t)
        - Bound_t = MEAN(Wealth_t) ± k x STDEV(Wealth_t)

    The cone does not assume normally distributed returns, but it does assume that the coming
    returns are drawn from the same distribution as the past ones, independently. The
    bootstrap forecast cone is the one pyfolio (Quantopian) draws in its tear sheets.

    For more information, see the following source:

    - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." The Annals of
    Statistics, 7(1), 1-26.

    Referenced against:
    - pyfolio, summarize_paths: https://github.com/quantopian/pyfolio/blob/0.9.2/pyfolio/timeseries.py#L1111

    Shares similar logic:
    - pyfolio, forecast_cone_bootstrap: https://github.com/quantopian/pyfolio/blob/0.9.2/pyfolio/timeseries.py#L1147

    Args:
        returns (pd.Series): The returns to draw from (missing returns are left out).
        horizon (int, optional): The number of periods to forecast. Defaults to 252.
        standard_deviations (list[float] | None, optional): The numbers of standard deviations
            k of the bounds. Defaults to None, which uses 1, 1.5 and 2 as pyfolio does.
        simulations (int, optional): The number of simulated paths. Defaults to 1000.
        seed (int | None, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        pd.DataFrame: The periods ahead (1 to horizon) as the index and the bounds ("Lower 2.0",
        ..., "Mean", ..., "Upper 2.0") as the columns. Fewer than two returns give NaN.
    """
    standard_deviations = sorted(standard_deviations or [1.0, 1.5, 2.0])
    labels = (
        [f"Lower {k:g}" for k in reversed(standard_deviations)]
        + ["Mean"]
        + [f"Upper {k:g}" for k in standard_deviations]
    )
    values = returns.dropna().to_numpy(dtype=float)

    if horizon < 1:
        raise ValueError("The horizon must be at least 1.")
    if len(values) < 2:  # noqa: PLR2004
        return pd.DataFrame(np.nan, index=range(1, horizon + 1), columns=labels)

    # The paths are drawn a block of simulations at a time, twice with the same seed: first
    # for the sum of the wealth per period, then for the squared deviations from its mean.
    # Summing the blocks row after row adds in the order numpy sums all paths at once (and
    # the mean and standard deviation are those sums divided, starting from 0), so the cone
    # is the same as from holding every path.
    def wealth_chunks():
        for positions in _get_bootstrap_chunks(
            len(values), simulations, "resample", length=horizon, seed=seed
        ):
            yield np.cumprod(1 + values[positions], axis=1)

    total = np.zeros(horizon)
    for wealth in wealth_chunks():
        for row in wealth:
            total += row
    mean = total / simulations

    deviation = np.zeros(horizon)
    if simulations > 1:
        squares = np.zeros(horizon)
        for wealth in wealth_chunks():
            np.subtract(wealth, mean, out=wealth)
            np.multiply(wealth, wealth, out=wealth)
            for row in wealth:
                squares += row
        deviation = np.sqrt(squares / (simulations - 1))

    cone = (
        [mean - k * deviation for k in reversed(standard_deviations)]
        + [mean]
        + [mean + k * deviation for k in standard_deviations]
    )

    return pd.DataFrame(
        np.column_stack(cone), index=range(1, horizon + 1), columns=labels
    )


def get_probability_of_backtest_overfitting(
    returns: pd.DataFrame, blocks: int = 16
) -> pd.Series:
    """
    Calculate the Probability of Backtest Overfitting (PBO): how likely the strategy (or asset)
    with the best past Sharpe ratio does no better than the median out of sample, with the
    Combinatorially Symmetric Cross-Validation (CSCV) of Bailey, Borwein, López de Prado and Zhu
    (2017).

    The history is split into S blocks of equal length. For every way to choose S / 2 blocks
    as the in-sample period, with the other half as the out-of-sample period:

        - The strategy with the highest in-sample Sharpe ratio (mean / standard deviation of the
          returns) is selected.
        - ω = Rank of its out-of-sample Sharpe ratio among the N strategies / (N + 1)
        - λ = ln(ω / (1 - ω))
        - PBO = Share of the combinations with λ <= 0

    The PBO is the probability that the strategy that looked best in-sample performs at or
    below the median out of sample. Close to 0, the selection carries over out of sample; at
    0.5 or higher, picking the in-sample best adds nothing. Ties in the out-of-sample Sharpe
    ratios share their average rank, a tie in-sample selects the first strategy.

    The table also reports:

        - Average Out-of-Sample Rank = Mean of ω (0.5 when the selection adds nothing)
        - Probability of Loss = Share of the combinations in which the selected strategy has an
          out-of-sample Sharpe ratio below 0
        - Performance Degradation = The slope of the regression of the selected strategy's
          out-of-sample Sharpe ratio on its in-sample Sharpe ratio (negative when a high
          in-sample Sharpe ratio is followed by a low one)
        - Combinations = The number of in-sample and out-of-sample splits, C(S, S / 2)

    For more information, see the following sources:

    - Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J. (2017). "The Probability of
    Backtest Overfitting." Journal of Computational Finance, 20(4), 39-69.

    The returns beyond a multiple of S rows are left out from the end, as in ffn. The Sharpe
    ratios skip missing returns. A combination in which a strategy has no Sharpe ratio (too
    few returns, or returns without variation) leaves the statistics undefined (NaN).

    Referenced against:
    - ffn, calc_prob_backtest_overfitting: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L3165

    Shares similar logic:
    - RiskLabAI, probability_of_backtest_overfitting: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/probability_of_backtest_overfitting.py#L80

    Args:
        returns (pd.DataFrame): The returns of the strategies, one column each, in
            chronological order.
        blocks (int, optional): The number of blocks S, even and at least 2. Defaults to 16.

    Returns:
        pd.Series: The Probability of Backtest Overfitting, Average Out-of-Sample Rank,
        Probability of Loss, Performance Degradation and Combinations.
    """
    if blocks < 2 or blocks % 2 or int(blocks) != blocks:  # noqa: PLR2004
        raise ValueError("The number of blocks must be an even integer of 2 or more.")

    rows = (len(returns) // blocks) * blocks

    if rows < blocks or returns.shape[1] < 2:  # noqa: PLR2004
        raise ValueError(
            "The probability of backtest overfitting needs at least two strategies and a "
            "return per block."
        )

    values = returns.iloc[:rows].to_numpy(dtype=float)
    observed = ~np.isnan(values)

    # The count, sum and sum of squared deviations from the strategy's overall mean per block,
    # from which the Sharpe ratio of any union of blocks follows without revisiting the rows.
    center = np.nanmean(values, axis=0)
    deviations = np.where(observed, values - center, 0.0)
    counts = np.add.reduceat(observed.astype(float), np.arange(0, rows, rows // blocks))
    sums = np.add.reduceat(deviations, np.arange(0, rows, rows // blocks))
    squares = np.add.reduceat(deviations**2, np.arange(0, rows, rows // blocks))

    combinations = np.array(
        list(itertools.combinations(range(blocks), blocks // 2)), dtype=int
    )
    in_sample = np.zeros((len(combinations), blocks))
    in_sample[np.arange(len(combinations))[:, np.newaxis], combinations] = 1

    def sharpe_ratios(selection: np.ndarray) -> np.ndarray:
        count = selection @ counts
        mean = (selection @ sums) / count
        variance = (selection @ squares - count * mean**2) / (count - 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe_ratio = (mean + center) / np.sqrt(variance)

        return np.where((count > 1) & (variance > 0), sharpe_ratio, np.nan)

    in_sample_sharpe = sharpe_ratios(in_sample)
    out_of_sample_sharpe = sharpe_ratios(1 - in_sample)

    if np.isnan(in_sample_sharpe).any() or np.isnan(out_of_sample_sharpe).any():
        return pd.Series(
            {
                "Probability of Backtest Overfitting": np.nan,
                "Average Out-of-Sample Rank": np.nan,
                "Probability of Loss": np.nan,
                "Performance Degradation": np.nan,
                "Combinations": float(len(combinations)),
            }
        )

    selected = in_sample_sharpe.argmax(axis=1)
    rows_index = np.arange(len(combinations))
    selected_out_of_sample = out_of_sample_sharpe[rows_index, selected]

    # The average rank of the selected strategy among ties, as scipy's rankdata "average".
    lower = (out_of_sample_sharpe < selected_out_of_sample[:, np.newaxis]).sum(axis=1)
    equal = (out_of_sample_sharpe == selected_out_of_sample[:, np.newaxis]).sum(axis=1)
    relative_rank = (lower + (equal + 1) / 2) / (returns.shape[1] + 1)
    logits = np.log(relative_rank / (1 - relative_rank))

    degradation = linregress(
        in_sample_sharpe[rows_index, selected], selected_out_of_sample
    ).slope

    return pd.Series(
        {
            "Probability of Backtest Overfitting": float((logits <= 0).mean()),
            "Average Out-of-Sample Rank": float(relative_rank.mean()),
            "Probability of Loss": float((selected_out_of_sample < 0).mean()),
            "Performance Degradation": float(degradation),
            "Combinations": float(len(combinations)),
        }
    )


def get_probabilistic_momentum(
    information_ratio: pd.Series | pd.DataFrame | float,
    observations: pd.Series | pd.DataFrame | float,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Probabilistic Momentum of Varadi (2014): the confidence that one asset
    outperforms another, the probability of Student's t distribution that the mean of the
    differences between their returns is above 0.

    The formula is as follows:

        - t = Information Ratio x SQRT(N)
        - Probabilistic Momentum = T(t, N - 1)

    Where the Information Ratio is the mean of the return differences over their standard
    deviation, N the number of return differences and T the cumulative Student's t
    distribution with N - 1 degrees of freedom. A value above 0.5 means the asset has done
    better; Varadi switches between two assets only when the confidence passes a threshold
    such as 0.6, rather than on the sign of the difference alone.

    For more information, see the following sources:

    - Varadi, D. (2014). "Are Simple Momentum Strategies Too Dumb? Introducing Probabilistic
    Momentum." CSS Analytics, 28 January 2014.

    Referenced against:
    - ffn, calc_prob_mom: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L1739

    Args:
        information_ratio (pd.Series | pd.DataFrame | float): The information ratio of the
            return differences, per observation.
        observations (pd.Series | pd.DataFrame | float): The number of return differences it
            was calculated from.

    Returns:
        pd.Series | pd.DataFrame | float: The probability, between 0 and 1. Fewer than two
        observations give NaN.
    """
    t_statistic = information_ratio * np.sqrt(observations)
    degrees_of_freedom = np.where(
        np.asarray(observations, dtype=float) > 1,
        np.asarray(observations, dtype=float) - 1,
        np.nan,
    )
    probability = stats.t.cdf(np.asarray(t_statistic, dtype=float), degrees_of_freedom)

    if isinstance(t_statistic, pd.DataFrame):
        return pd.DataFrame(
            probability, index=t_statistic.index, columns=t_statistic.columns
        )
    if isinstance(t_statistic, pd.Series):
        return pd.Series(probability, index=t_statistic.index, name=t_statistic.name)

    return float(probability)


def get_factor_regression(
    excess_returns: pd.Series, factor_dataset: pd.DataFrame, window: int | None = None
) -> pd.DataFrame:
    """
    Calculate the factor regression: how strongly the excess returns move with each factor, from
    the regression fitted over all observations or over every rolling window to show how the
    factor exposures move over time.

    The formula is as follows:

        - Excess Return = Intercept + Slope1 x Factor1 + ... + SlopeK x FactorK + Residual

    fitted by ordinary least squares on all observations, or on the last `window` observations
    at every date, so the slopes show how the factor exposures move over time. Observations
    where the excess return or a factor is missing are left out (listwise deletion), and the
    windows run over the observations that remain.

    For more information, see the following sources:

    - Fama, E.F., & French, K.R. (2015). "A Five-Factor Asset Pricing Model." Journal of
    Financial Economics, 116(1), 1-22.

    Referenced against:
    - pyfolio, rolling_regression: https://github.com/stefan-jansen/pyfolio-reloaded/blob/0.10.2/src/pyfolio/timeseries.py#L550

    Shares similar logic:
    - statsmodels, RollingOLS: https://www.statsmodels.org/stable/generated/statsmodels.regression.rolling.RollingOLS.html

    Args:
        excess_returns (pd.Series): The excess returns.
        factor_dataset (pd.DataFrame): The factors, one per column, on the same index.
        window (int | None, optional): The number of observations of a rolling window, more
            than the number of factors plus 1. Defaults to None, which fits one regression on
            all observations.

    Returns:
        pd.DataFrame: The Intercept, the Slope of every factor ("<factor> Slope"), the Mean
        Squared Error (MSE) and the R Squared of each window, indexed by its last date. Without
        a window a single row, indexed by the last date of the observations.
    """
    validate_window("window", window, 2)
    factors = list(factor_dataset.columns)
    columns = (
        ["Intercept"]
        + [f"{factor} Slope" for factor in factors]
        + ["Mean Squared Error (MSE)", "R Squared"]
    )

    if window is not None and window <= len(factors) + 1:
        raise ValueError(
            "The rolling window needs more observations than the factors plus one."
        )

    data = pd.concat(
        [
            excess_returns.rename("__response__"),
            factor_dataset.reindex(excess_returns.index),
        ],
        axis=1,
    ).dropna()
    # Without a window, a single window that holds every observation.
    window_size = len(data) if window is None else window

    if len(data) < window_size or len(data) <= len(factors) + 1:
        return pd.DataFrame(columns=columns, dtype=float)

    response = np.lib.stride_tricks.sliding_window_view(
        data.iloc[:, 0].to_numpy(dtype=float), window_size
    )
    regressors = np.lib.stride_tricks.sliding_window_view(
        data.iloc[:, 1:].to_numpy(dtype=float), window_size, axis=0
    ).transpose(0, 2, 1)

    response_mean = response.mean(axis=1)
    regressor_mean = regressors.mean(axis=1)
    centered_response = response - response_mean[:, np.newaxis]
    centered_regressors = regressors - regressor_mean[:, np.newaxis, :]

    # The normal equations of every window at once; the pseudo-inverse gives the minimum norm
    # solution where a factor does not vary within a window, as the least squares solver would.
    slopes = np.einsum(
        "wkl,wl->wk",
        np.linalg.pinv(
            np.einsum("wik,wil->wkl", centered_regressors, centered_regressors)
        ),
        np.einsum("wik,wi->wk", centered_regressors, centered_response),
    )
    residuals = centered_response - np.einsum("wik,wk->wi", centered_regressors, slopes)
    residual_sum_of_squares = (residuals**2).sum(axis=1)
    total_sum_of_squares = (centered_response**2).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.where(
            total_sum_of_squares > 0,
            1 - residual_sum_of_squares / total_sum_of_squares,
            np.where(residual_sum_of_squares == 0, 1.0, 0.0),
        )

    return pd.DataFrame(
        np.column_stack(
            [
                response_mean - np.einsum("wk,wk->w", regressor_mean, slopes),
                slopes,
                residual_sum_of_squares / window_size,
                r_squared,
            ]
        ),
        index=data.index[window_size - 1 :],
        columns=columns,
    )


def get_variability_weighted_return(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: float | pd.Series = 252.0,
    tau: float = 2.0,
    maximum_deviation: float = 0.2,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Variability-Weighted Return (VWR): the annualized return reduced by how far
    the value strays from a steady exponential path with the same overall growth.

    A return earned along a smooth path keeps most of its value; one earned with large swings
    loses part of it, and all of it once the deviations reach `maximum_deviation`.

    The formula is as follows:

    - r̄ = Average of ln(1 + Return), the average log return per period
    - Annualized Return = exp(r̄ x P) - 1
    - Deviation_i = V_i / (V_0 x exp(r̄ x i)) - 1, for i = 1..N, with V the value (growth of 1)
    - VWR = Annualized Return x (1 - (Std(Deviation) / maximum_deviation) ^ tau)

    Where P is the number of returns in a year, N the number of returns and Std the sample
    standard deviation (with n - 1 in the denominator).

    The measure comes from backtrader's VWR analyzer (Daniel Rodriguez), after the "Sharpe
    Ratio: Better with Log Returns" article of crystalbull.com.

    Shares similar logic:
    - backtrader, VWR: https://github.com/backtrader/backtrader/blob/1.9.74.123/backtrader/analyzers/vwr.py#L32

    Args:
        returns (pd.Series | pd.DataFrame): The returns, one column per asset, optionally with a
            (period, date) index; each column on its own returns, missing returns left out.
        periods_per_year (float | pd.Series, optional): The number of returns in a year, or one
            value per column. Defaults to 252.
        tau (float, optional): The exponent of the penalty for variability. Defaults to 2.0.
        maximum_deviation (float, optional): The standard deviation of the deviations at which
            the return is reduced to zero. Defaults to 0.2.

    Returns:
        pd.Series | pd.DataFrame | float: The Variability-Weighted Return as a decimal, a float
        for a Series, per column for a DataFrame and per period and column for a within period
        index. NaN with fewer than two returns.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            return pd.DataFrame(
                {
                    sub_period: get_variability_weighted_return(
                        returns.loc[sub_period],
                        periods_per_year,
                        tau,
                        maximum_deviation,
                    )
                    for sub_period in returns.index.get_level_values(0).unique()
                }
            ).T

        return pd.Series(
            {
                column: get_variability_weighted_return(
                    returns[column],
                    (
                        periods_per_year[column]
                        if isinstance(periods_per_year, pd.Series)
                        else periods_per_year
                    ),
                    tau,
                    maximum_deviation,
                )
                for column in returns.columns
            },
            dtype=float,
        )

    values = returns.dropna().to_numpy(dtype=float)
    if values.size < 2:  # noqa: PLR2004
        return np.nan

    log_returns = np.log1p(values)
    average = log_returns.mean()
    steps = np.arange(1, values.size + 1)
    deviations = np.exp(np.cumsum(log_returns) - average * steps) - 1

    return float(
        np.expm1(average * periods_per_year)
        * (1 - (deviations.std(ddof=1) / maximum_deviation) ** tau)
    )


def _get_herfindahl_hirschman_index(values: np.ndarray) -> float:
    """
    The normalized Herfindahl-Hirschman Index of the shares of values in their sum: 0 when they
    are equal, 1 when one holds everything. NaN with fewer than three values (López de Prado).

    Args:
        values (np.ndarray): The values, all of one sign.

    Returns:
        float: The concentration.
    """
    if values.size <= 2 or values.sum() == 0:  # noqa: PLR2004
        return np.nan

    weights = values / values.sum()
    count = values.size

    return float(((weights**2).sum() - 1 / count) / (1 - 1 / count))


def get_return_concentration(returns: pd.Series) -> pd.Series:
    """
    Calculate the return concentration of López de Prado: how much of the gains (and of the
    losses) comes from a few returns, and how much the bets cluster in a few months, each as a
    normalized Herfindahl-Hirschman Index between 0 (spread evenly) and 1 (all in one).

    The formula is as follows:

    - h(w) = (SUM w_j² - 1 / N) / (1 - 1 / N), w_j = x_j / SUM x, over N values
    - Positive Concentration = h of the returns of 0 or more
    - Negative Concentration = h of the returns below 0
    - Time Concentration = h of the number of returns in each calendar month

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 14.5.1
      (Snippet 14.3).

    Args:
        returns (pd.Series): The returns of one asset with a date (or period) index, missing
            returns left out.

    Returns:
        pd.Series: The Positive, Negative and Time Concentration, each NaN with fewer than three
        values.
    """
    returns = returns.dropna()
    months = (
        returns.index.to_period("M")
        if isinstance(returns.index, pd.DatetimeIndex)
        else to_period_index(returns.index).asfreq("M")
    )
    values = returns.to_numpy(dtype=float)

    return pd.Series(
        {
            "Positive Concentration": _get_herfindahl_hirschman_index(
                values[values >= 0]
            ),
            "Negative Concentration": _get_herfindahl_hirschman_index(
                values[values < 0]
            ),
            "Time Concentration": _get_herfindahl_hirschman_index(
                returns.groupby(months).count().to_numpy(dtype=float)
            ),
        }
    )


def get_binomial_sharpe_ratio(
    profit_taking: float, stop_loss: float, precision: float, frequency: float
) -> float:
    """
    Calculate the binomial Sharpe ratio: the annualized Sharpe ratio of a strategy whose bets
    either gain `profit_taking` or lose `stop_loss`, with a probability `precision` of a gain
    and `frequency` bets a year.

    The formula is as follows:

    - SR = ((π+ - π-) x p + π-) / ((π+ - π-) x SQRT(p x (1 - p))) x SQRT(n)

    Where π+ is the gain, π- the loss (a negative number), p the precision and n the frequency.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 15.3
      (Snippets 15.1 to 15.4).

    Shares similar logic:
    - RiskLabAI, binomial_sharpe_ratio: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L193

    Args:
        profit_taking (float): The return of a winning bet π+.
        stop_loss (float): The return of a losing bet π-, negative.
        precision (float): The probability of a winning bet p.
        frequency (float): The number of bets a year n.

    Returns:
        float: The annualized Sharpe ratio.
    """
    spread = profit_taking - stop_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            (spread * precision + stop_loss)
            / (spread * np.sqrt(precision * (1 - precision)))
            * np.sqrt(frequency)
        )


def get_implied_precision(
    profit_taking: float, stop_loss: float, frequency: float, target_sharpe_ratio: float
) -> float:
    """
    Calculate the implied precision: the probability of a winning bet a strategy needs to reach
    a target Sharpe ratio with its gains, losses and number of bets a year.

    The formula is as follows:

    - a = (n + θ²) x (π+ - π-)²
    - b = (2 x n x π- - θ² x (π+ - π-)) x (π+ - π-)
    - c = n x π-²
    - p* = (-b + SQRT(b² - 4ac)) / (2a)

    Where θ is the target Sharpe ratio, n the bets a year, π+ the gain and π- the loss.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 15.4
      (Snippet 15.3).

    Shares similar logic:
    - RiskLabAI, implied_precision: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L82

    Args:
        profit_taking (float): The return of a winning bet π+.
        stop_loss (float): The return of a losing bet π-, negative.
        frequency (float): The number of bets a year n.
        target_sharpe_ratio (float): The annualized Sharpe ratio to reach θ.

    Returns:
        float: The implied precision, NaN when no precision between 0 and 1 reaches the target.
    """
    spread = profit_taking - stop_loss
    a = (frequency + target_sharpe_ratio**2) * spread**2
    b = (2 * frequency * stop_loss - target_sharpe_ratio**2 * spread) * spread
    c = frequency * stop_loss**2
    discriminant = b**2 - 4 * a * c
    if not discriminant >= 0 or a == 0:
        return np.nan

    precision = (-b + np.sqrt(discriminant)) / (2 * a)

    return float(precision) if 0 <= precision <= 1 else np.nan


def get_implied_frequency(
    profit_taking: float, stop_loss: float, precision: float, target_sharpe_ratio: float
) -> float:
    """
    Calculate the implied frequency: the number of bets a year a strategy needs to reach a target
    Sharpe ratio with its gains, losses and precision.

    The formula is as follows:

    - n* = θ² x (π+ - π-)² x p x (1 - p) / ((π+ - π-) x p + π-)²

    Where θ is the target Sharpe ratio, p the precision, π+ the gain and π- the loss. The squared
    equation also solves a negative target, so a frequency is only implied when the expected
    return of a bet has the sign of the target.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 15.4
      (Snippet 15.4).

    Shares similar logic:
    - RiskLabAI, bin_frequency: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L134

    Args:
        profit_taking (float): The return of a winning bet π+.
        stop_loss (float): The return of a losing bet π-, negative.
        precision (float): The probability of a winning bet p.
        target_sharpe_ratio (float): The annualized Sharpe ratio to reach θ.

    Returns:
        float: The implied number of bets a year, NaN when no frequency reaches the target.
    """
    spread = profit_taking - stop_loss
    expected = spread * precision + stop_loss
    if expected == 0 or np.sign(expected) != np.sign(target_sharpe_ratio):
        return np.nan

    return float(
        target_sharpe_ratio**2 * spread**2 * precision * (1 - precision) / expected**2
    )


def get_strategy_risk(
    returns: pd.Series, frequency: float = 252.0, target_sharpe_ratio: float = 2.0
) -> pd.Series:
    """
    Calculate the strategy risk of López de Prado: how likely a strategy is to miss a target
    Sharpe ratio, reading the returns as bets that either win the average gain or lose the
    average loss.

    It gives the Sharpe ratio such a strategy has, the precision and the number of bets a year
    it needs to reach the target, and the probability that its true precision is below the one
    needed (the probability of failure).

    The formula is as follows:

    - p = Number of Returns above 0 / Number of Returns, the precision
    - π+ = Average of the Returns above 0, π- = Average of the Returns of 0 or below
    - Binomial Sharpe Ratio = ((π+ - π-) x p + π-) / ((π+ - π-) x SQRT(p x (1 - p))) x SQRT(n)
    - Implied Precision p* and Implied Frequency n* that give the target Sharpe ratio θ
    - Probability of Failure = N((p* - p) / SQRT(p x (1 - p) / N))

    Where n is the number of bets a year, N the number of returns and N(.) the standard normal
    distribution function: the precision measured over N bets has a standard error of
    SQRT(p x (1 - p) / N), the spread of its bootstrap distribution.

    For more information, see the following book and paper:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 15
      (Snippets 15.1 to 15.5).
    - Bailey, D.H. and López de Prado, M. (2014). "The Sharpe Ratio Efficient Frontier." Journal
      of Risk, 15(2), 3-44.

    Shares similar logic:
    - RiskLabAI, implied_precision: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L82
    - RiskLabAI, bin_frequency: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L134
    - RiskLabAI, binomial_sharpe_ratio: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L193
    - RiskLabAI, failure_probability: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/backtest/strategy_risk.py#L281

    Args:
        returns (pd.Series): The returns of the bets, missing returns left out.
        frequency (float, optional): The number of bets a year. Defaults to 252.
        target_sharpe_ratio (float, optional): The annualized Sharpe ratio to reach. Defaults
            to 2.0.

    Returns:
        pd.Series: The Precision, Average Gain, Average Loss, Bets per Year, Binomial Sharpe
        Ratio, Implied Precision, Implied Frequency and Probability of Failure.
    """
    values = returns.dropna().to_numpy(dtype=float)
    gains, losses = values[values > 0], values[values <= 0]
    names = [
        "Precision",
        "Average Gain",
        "Average Loss",
        "Bets per Year",
        "Binomial Sharpe Ratio",
        "Implied Precision",
        "Implied Frequency",
        "Probability of Failure",
    ]
    if gains.size == 0 or losses.size == 0:
        return pd.Series(np.nan, index=names)

    precision = gains.size / values.size
    profit_taking, stop_loss = gains.mean(), losses.mean()
    implied_precision = get_implied_precision(
        profit_taking, stop_loss, frequency, target_sharpe_ratio
    )
    standard_error = np.sqrt(precision * (1 - precision) / values.size)

    return pd.Series(
        [
            precision,
            profit_taking,
            stop_loss,
            frequency,
            get_binomial_sharpe_ratio(profit_taking, stop_loss, precision, frequency),
            implied_precision,
            get_implied_frequency(
                profit_taking, stop_loss, precision, target_sharpe_ratio
            ),
            stats.norm.cdf((implied_precision - precision) / standard_error),
        ],
        index=names,
    )


def get_backtest_statistics_table(
    returns: pd.DataFrame,
    statistic: Callable[..., pd.Series],
    **kwargs: float | pd.Series,
) -> pd.DataFrame:
    """
    Calculate a backtest statistic of every column: a statistic that returns several values (a
    Series) per asset, and for a within period index per period, with keyword arguments given
    per column as a Series.

    Args:
        returns (pd.DataFrame): A DataFrame of returns, one column per asset, optionally with a
            (period, date) index.
        statistic (Callable[..., pd.Series]): The statistic of one column of returns.
        **kwargs (float | pd.Series): The other arguments of the statistic, a Series holding
            one value per column.

    Returns:
        pd.DataFrame: Over the whole history, the statistics as the index and the assets as the
        columns. Within periods, the periods as the index and (statistic, asset) as the columns.
    """

    def statistics_of(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: statistic(
                    frame[column],
                    **{
                        name: value[column] if isinstance(value, pd.Series) else value
                        for name, value in kwargs.items()
                    },
                )
                for column in frame.columns
            }
        )

    if returns.index.nlevels != MULTI_PERIOD_INDEX_LEVELS:
        return statistics_of(returns)

    return pd.DataFrame(
        {
            sub_period: statistics_of(returns.loc[sub_period]).stack()
            for sub_period in returns.index.get_level_values(0).unique()
        }
    ).T
