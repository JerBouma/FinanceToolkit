"""Value at Risk Model"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from financetoolkit.risk import risk_model
from financetoolkit.risk.helpers import as_float
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.error_model import InvalidArgumentError, validate_window
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import by_own_trading_days

logger = get_logger()

ALPHA_CONSTRAINT = 0.5

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# A Student-T fit has three parameters, so fewer observations cannot identify it.
MINIMUM_STUDENT_T_OBSERVATIONS = 3

# Floating point slack when comparing alpha with the GPD exceedance probability.
ALPHA_TOLERANCE = 1e-12

# The losses above the threshold a GPD fit needs: its shape and scale are unstable on a
# handful of points, so with fewer the EVT VaR and CVaR are NaN rather than a random value.
MINIMUM_NUMBER_OF_EXCEEDANCES = 10

# The standard deviation of the spreads needs two of them (ddof=1).
MINIMUM_SPREAD_OBSERVATIONS = 2

# A GARCH fit needs a long sample: as rugarch's ugarchfit, at least 100 returns per fit.
MINIMUM_GARCH_OBSERVATIONS = 100


@by_own_trading_days(rolling_only=True)
def get_var_historic(
    returns: pd.Series | pd.DataFrame, alpha: float, window: int | None = None
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the historical Value at Risk (VaR) of returns: the return that is only beaten
    on the downside in a fraction alpha of the observed periods, over the whole sample or
    within every rolling window.

    The formula is as follows:

        - VaR = Quantile(Return, alpha), with linear interpolation between observations

    Shares similar logic:
    - PerformanceAnalytics (R), VaR: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/VaR.R#L329
    - empyrical, value_at_risk: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2095
    - Riskfolio-Lib, VaR_Hist: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.VaR_Hist
    - skfolio, value_at_risk: https://skfolio.org/generated/skfolio.measures.value_at_risk.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window (int | None, optional): The number of observations of a rolling window, at least 1. Defaults
            to None, which calculates the VaR over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.

    Notes:
    - The VaR is the alpha quantile of the returns with linear interpolation; Riskfolio-Lib and
      skfolio take an observed return instead and report the loss as a positive number.
    """
    validate_window("window", window, 1)
    if window is not None:
        # Linear interpolation, as np.percentile, in one pass instead of a call per window.
        return returns.rolling(window=window).quantile(alpha, interpolation="linear")

    return risk_model.get_quantile(returns, alpha)


def get_var_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the gaussian Value at Risk (VaR) of returns: the alpha quantile of a normal
    distribution with the mean and standard deviation of the returns.

    The formula is as follows:

    - VaR = mean + z_alpha * std

    Where `z_alpha` is the standard normal quantile at `alpha` and `std` is the sample
    standard deviation (with n - 1 in the denominator), the estimator every parametric VaR
    and CVaR in this module uses. See `get_var_cornish_fisher` for the version that corrects the
    quantile for the skewness and kurtosis of the returns.

    More material/resources:
     - "Finance Compact Plus" by Zimmerman; Part 1, page 130-131;
     - "Risk Management and Financial Institutions" by John C. Hull

    Shares similar logic:
    - PerformanceAnalytics (R), VaR: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/VaR.R#L329
    - quantstats, value_at_risk: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2014

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns. A rolling
            window of them (`returns.rolling(window_size)`) works too and gives the VaR of
            every window, as the out-of-sample estimate of the risk module uses.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_gaussian(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    za = stats.norm.ppf(alpha, 0, 1)

    return returns.mean() + za * returns.std(ddof=1)


def get_kernel_bandwidth(samples: np.ndarray) -> np.ndarray:
    """
    Calculate the bandwidth of a Gaussian kernel density of every row of returns: how far
    each return is smoothed out, by Silverman's (1986) normal reference rule.

    The formula is as follows:

    - h = Std * (4 / (3 * n))^(1/5)

    Where Std is the sample standard deviation (with n - 1 in the denominator) and n the
    number of returns.

    Shares similar logic:
    - scipy, gaussian_kde: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Args:
        samples (np.ndarray): The returns, one sample per row.

    Returns:
        np.ndarray: The bandwidth of every row.
    """
    return samples.std(axis=1, ddof=1) * (4 / (3 * samples.shape[1])) ** 0.2


def get_var_kernel_samples(samples: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate the kernel Value at Risk of every row of returns without missing values: the
    alpha quantile of a smoothed (Gaussian kernel) density of the returns.

    The VaR is the root of the following, with h from `get_kernel_bandwidth`:

    - mean(Φ((VaR - Return_i) / h)) = alpha

    Referenced against:
    - scipy, gaussian_kde: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Args:
        samples (np.ndarray): The returns, one sample per row.
        alpha (float): The tail probability, between 0 and 1.

    Returns:
        np.ndarray: The VaR of every row as a return, negative for a loss.
    """
    bandwidth = get_kernel_bandwidth(samples)
    result = samples[:, 0].astype(float)

    for row in np.flatnonzero(bandwidth > 0):
        values, width = samples[row], bandwidth[row]
        # The kernel distribution function is below alpha ten bandwidths under the lowest
        # return and above it ten bandwidths over the highest.
        result[row] = brentq(
            lambda quantile, values=values, width=width: stats.norm.cdf(
                (quantile - values) / width
            ).mean()
            - alpha,
            values.min() - 10 * width,
            values.max() + 10 * width,
            xtol=1e-14,
        )

    return result


def get_var_kernel(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the kernel Value at Risk (VaR) of returns: the alpha quantile of a smoothed
    estimate of the return distribution, a less jumpy version of the historical VaR.

    The distribution is a Gaussian kernel density estimate with Silverman's bandwidth
    (Gourieroux, Laurent and Scaillet 2000).

    The formula is as follows:

    - VaR solves mean(Φ((VaR - Return_i) / h)) = alpha
    - h = Std * (4 / (3 * n))^(1/5)

    For more information see:

    - Gourieroux, C., Laurent, J.P. and Scaillet, O. (2000). "Sensitivity Analysis of Values
    at Risk." Journal of Empirical Finance, 7(3-4), 225-245.
    - Silverman, B.W. (1986). "Density Estimation for Statistics and Data Analysis." Chapman
    and Hall.

    Referenced against:
    - scipy, gaussian_kde: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Shares similar logic:
    - PerformanceAnalytics (R), VaR: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/VaR.R#L329

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

    Returns:
        pd.Series | pd.DataFrame | float: The VaR as a return, negative for a loss.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie between 0 and 1.")

    return risk_model.apply_to_observations(
        returns, lambda samples: get_var_kernel_samples(samples, alpha)
    )


def fit_gpd_tail(
    returns: pd.DataFrame, threshold_percentile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Generalized Pareto Distribution (GPD) to the losses of each column of `returns`
    that exceed a threshold: the shape and scale of the extreme losses (Peak-over-Threshold).

    This is the shared fitting routine behind both `get_var_evt` (below) and
    `cvar_model.get_cvar_evt`, since both need the same shape, scale and threshold
    parameters of the fitted GPD tail.

    Args:
        returns (pd.DataFrame): A Dataframe of returns, one column per asset.
        threshold_percentile (float): The percentile of losses above which the GPD is
        fitted (e.g. 0.95 fits the GPD on the worst 5% of losses).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The fitted shape (xi), scale
        (sigma), threshold (u) and exceedance probability, one value per column of `returns`,
        in the same order as `returns.columns`. The GPD fit needs at least 10 losses above
        the threshold (MINIMUM_NUMBER_OF_EXCEEDANCES), so about 200 returns at the default
        threshold percentile of 0.95; with fewer (e.g. a short sub-period) NaN is returned
        for the shape, scale and exceedance probability of that column, see
        `warn_too_few_exceedances`.
    """
    shape, scale, threshold, exceedance_probability = [], [], [], []
    for column in returns.columns:
        losses = -returns[column].dropna()

        # A column without observations (e.g. a period before the listing) has no tail to fit.
        if losses.empty:
            shape.append(np.nan)
            scale.append(np.nan)
            threshold.append(np.nan)
            exceedance_probability.append(np.nan)
            continue

        column_threshold = np.percentile(losses, threshold_percentile * 100)
        exceedances = losses[losses > column_threshold] - column_threshold

        if len(exceedances) < MINIMUM_NUMBER_OF_EXCEEDANCES:
            shape.append(np.nan)
            scale.append(np.nan)
            threshold.append(column_threshold)
            exceedance_probability.append(np.nan)
            continue

        column_shape, _, column_scale = stats.genpareto.fit(exceedances, floc=0)

        shape.append(column_shape)
        scale.append(column_scale)
        threshold.append(column_threshold)
        exceedance_probability.append(len(exceedances) / len(losses))

    return (
        np.array(shape),
        np.array(scale),
        np.array(threshold),
        np.array(exceedance_probability),
    )


def get_number_of_exceedances(
    returns: pd.DataFrame, threshold_percentile: float
) -> pd.Series:
    """
    Calculate the number of losses above the `threshold_percentile` percentile of the losses
    per column: how many observations `fit_gpd_tail` fits the GPD to.

    Args:
        returns (pd.DataFrame): A Dataframe of returns, one column per asset.
        threshold_percentile (float): The percentile of losses above which the GPD is fitted.

    Returns:
        pd.Series: The number of exceedances per column, NaN for a column without returns.
    """
    losses = -returns

    return (
        (losses > losses.quantile(threshold_percentile)).sum().where(losses.count() > 0)
    )


def warn_too_few_exceedances(
    returns: pd.Series | pd.DataFrame, threshold_percentile: float
) -> None:
    """
    Log, once for all columns and periods, where the EVT VaR or CVaR is NaN: where fewer than
    MINIMUM_NUMBER_OF_EXCEEDANCES losses lie above the threshold (see `fit_gpd_tail`).

    Args:
        returns (pd.Series | pd.DataFrame): The returns, optionally with a (period, date)
            index.
        threshold_percentile (float): The percentile of losses above which the GPD is fitted.
    """
    returns = pd.DataFrame(returns)

    if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
        counts = returns.groupby(level=0, sort=False).apply(
            lambda period_returns: get_number_of_exceedances(
                period_returns, threshold_percentile
            )
        )
    else:
        counts = get_number_of_exceedances(returns, threshold_percentile).to_frame().T

    too_few = (counts < MINIMUM_NUMBER_OF_EXCEEDANCES).sum()
    too_few = too_few[too_few > 0]

    if not too_few.empty:
        logger.warning(
            "The Extreme Value Theory VaR and CVaR fit the Generalized Pareto Distribution to at least %s losses "  # noqa: E501
            "above the %s percentile, which %s do not have in up to %s of %s periods, so those are NaN. Use more returns "  # noqa: E501
            "per period (a longer period, within_period=False or daily data), about %s for this threshold.",  # noqa: E501
            MINIMUM_NUMBER_OF_EXCEEDANCES,
            threshold_percentile,
            ", ".join(map(str, too_few.index)),
            int(too_few.max()),
            len(counts),
            int(np.ceil(MINIMUM_NUMBER_OF_EXCEEDANCES / (1 - threshold_percentile))),
        )


def validate_evt_alpha(alpha: float, threshold_percentile: float) -> None:
    """
    Check that the Peak-over-Threshold VaR and CVaR are evaluated in the fitted tail, where
    the GPD describes the losses.

    The GPD only describes the losses beyond the threshold, so the quantile at `alpha`
    must lie beyond it: `alpha` can be at most the exceedance probability
    1 - `threshold_percentile`. Shared by `get_var_evt` and `cvar_model.get_cvar_evt`.

    Args:
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        threshold_percentile (float): The percentile of losses above which the GPD is fitted.

    Raises:
        ValueError: If `alpha` is larger than 1 - `threshold_percentile`.
    """
    if alpha > 1 - threshold_percentile + ALPHA_TOLERANCE:
        raise ValueError(
            f"An alpha of {alpha} lies outside the tail fitted above the "
            f"{threshold_percentile} threshold percentile. Use an alpha of at most "
            f"{1 - threshold_percentile:.4g} or lower the threshold_percentile."
        )


def get_var_evt(
    returns: pd.Series | pd.DataFrame,
    alpha: float,
    threshold_percentile: float = 0.95,
    warn: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Value at Risk (VaR) of returns based on Extreme Value Theory (EVT): the
    alpha quantile of a distribution fitted to the extreme losses alone.

    It uses a Peak-over-Threshold approach in which a Generalized Pareto Distribution (GPD)
    is fitted to the losses that exceed a given threshold.

    The formula is as follows:

    - VaR = -(u + (sigma / xi) * ((alpha / p)^(-xi) - 1))

    Where `u` is the loss threshold, `xi` and `sigma` the fitted GPD shape and scale and
    `p` the fraction of losses above the threshold. The formula extrapolates the tail
    beyond the threshold, so it only holds for `alpha <= 1 - threshold_percentile`.

    For more information see:

    - McNeil, A.J., & Frey, R. (2000). "Estimation of Tail-Related Risk Measures for
    Heteroscedastic Financial Time Series: An Extreme Value Approach." Journal of Empirical
    Finance, 7(3-4), 271-300.
    - https://en.wikipedia.org/wiki/Extreme_value_theory

    Shares similar logic:
    - empyrical, gpd_risk_estimates: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1675

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level). It needs
        to be at most 1 - `threshold_percentile`.
        threshold_percentile (float, optional): The percentile of losses above which the GPD is
        fitted (e.g. 0.95 fits the GPD on the worst 5% of losses). Defaults to 0.95.
        warn (bool, optional): Whether to log the columns and periods with fewer than 10 losses
        above the threshold, for which the VaR is NaN (see `fit_gpd_tail`). Defaults to True.

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `alpha` is larger than 1 - `threshold_percentile`, where the VaR
        would lie below the threshold the GPD is fitted above.
    """
    validate_evt_alpha(alpha, threshold_percentile)

    if warn:
        warn_too_few_exceedances(returns, threshold_percentile)

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_evt(
                returns.loc[sub_period], alpha, threshold_percentile, warn=False
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    returns = pd.DataFrame(returns)

    # Fitting a Generalized Pareto Distribution to the exceedances over the threshold.
    shape, scale, threshold, exceedance_probability = fit_gpd_tail(
        returns, threshold_percentile
    )

    value_at_risk = -(
        threshold + (scale / shape) * ((alpha / exceedance_probability) ** (-shape) - 1)
    )

    return pd.Series(value_at_risk, index=returns.columns)


def fit_student_t(
    returns: pd.Series | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Student-T distribution to each column of `returns` by maximum likelihood: the
    degrees of freedom, location and scale that describe their fat tails.

    Shared by `get_var_studentt` and `cvar_model.get_cvar_studentt`, so the VaR and CVaR
    come from the same fitted distribution. Missing observations are dropped per column,
    since scipy's fit raises on any non-finite value.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The degrees of freedom, location and
        scale, one value per column (NaN for a column with fewer than three observations).
    """
    columns = (
        [returns.dropna()]
        if isinstance(returns, pd.Series)
        else [returns[column].dropna() for column in returns.columns]
    )

    fitted = [
        (
            stats.t.fit(column)
            if len(column) >= MINIMUM_STUDENT_T_OBSERVATIONS
            else (np.nan, np.nan, np.nan)
        )
        for column in columns
    ]

    return (
        np.array([parameters[0] for parameters in fitted], dtype=float),
        np.array([parameters[1] for parameters in fitted], dtype=float),
        np.array([parameters[2] for parameters in fitted], dtype=float),
    )


def get_var_studentt(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Student-T Value at Risk (VaR) of returns: the alpha quantile of a fat-tailed
    Student-T distribution fitted to the returns.

    The degrees of freedom `v`, the location and the scale are estimated jointly by
    maximum likelihood (scipy's t.fit), and the VaR is the corresponding quantile
    of the fitted distribution:

    - VaR = location + scale * t_v^-1(alpha)

    The fitted location is used rather than the sample mean so that the quantile
    belongs to one and the same fitted distribution (with fat tails the two can differ).
    Columns with fewer than three observations return NaN, since the three parameters
    cannot be identified.

    Note that the scale of a Student-T distribution is *not* its standard deviation:
    the two are related by std = scale * SQRT(v / (v - 2)), and only for v > 2,
    since a Student-T with two or fewer degrees of freedom has no finite Variance at
    all. Using the fitted scale rather than rescaling the sample standard deviation
    therefore keeps the estimate well defined for every fitted v (heavily fat-tailed
    return samples routinely fit below v = 2), and matches the convention used by
    `cvar_model.get_cvar_studentt`. It also avoids the classic overstatement of risk
    that results from multiplying the sample standard deviation by the *unscaled*
    `t_v^-1(alpha)` quantile.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_studentt(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    degrees_of_freedom, location, scale = fit_student_t(returns)
    value_at_risk = location + scale * stats.t.ppf(alpha, degrees_of_freedom)

    # A Series gives a length one array, the type `as_float` (risk helpers) reduces.
    if isinstance(returns, pd.Series):
        return value_at_risk

    return pd.Series(value_at_risk, index=returns.columns)


def get_var_cornish_fisher(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Cornish-Fisher (modified) Value at Risk (VaR) of returns: the gaussian VaR
    corrected for the skewness and fat tails of the returns.

    The Cornish-Fisher expansion adjusts the standard normal quantile for the sample
    skewness and (excess) kurtosis of the return distribution, which makes the resulting
    VaR more accurate than the plain gaussian VaR (`get_var_gaussian`) for returns that
    are not normally distributed, i.e. that are skewed and/or fat-tailed (as most asset
    returns are, see `econometrics.diagnostics_model.get_jarque_bera_test`).

    The formula is as follows:

    - z_cf = z + (z^2 - 1) * S / 6 + (z^3 - 3z) * K / 24 - (2z^3 - 5z) * S^2 / 36
    - VaR = mean + z_cf * std

    Where `z` is the gaussian quantile at `alpha`, `S` is the skewness, `K` is the
    excess (Fisher) kurtosis and `std` the sample standard deviation (with n - 1 in the
    denominator) of `returns`.

    The expansion is only a valid quantile function while it is monotone in `z`, which
    holds for moderate skewness and excess kurtosis only (Maillard, 2012). For strongly
    skewed or very fat-tailed returns the adjusted quantile can bend back, so a deeper
    tail gives a less negative VaR; compare with `get_var_historic` or `get_var_evt`
    in that case.

    Also known as: modified VaR, mVaR.

    For more information about the method, see the following papers:

    - Cornish, E.A., & Fisher, R.A. (1938). "Moments and Cumulants in the Specification
    of Distributions." Revue de l'Institut International de Statistique, 5(4), 307-320.
    - Favre, L., & Galeano, J.A. (2002). "Mean-Modified Value-at-Risk Optimization with
    Hedge Funds." Journal of Alternative Investments, 5(2), 21-25.
    - Maillard, D. (2012). "A User's Guide to the Cornish Fisher Expansion." SSRN
    Working Paper 1997178.

    Shares similar logic:
    - PerformanceAnalytics (R), VaR: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/VaR.R#L329

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_cornish_fisher(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    za = stats.norm.ppf(alpha, 0, 1)
    skewness = risk_model.get_skewness(returns)
    excess_kurtosis = risk_model.get_kurtosis(returns, fisher=True)

    z_cornish_fisher = (
        za
        + (za**2 - 1) * skewness / 6
        + (za**3 - 3 * za) * excess_kurtosis / 24
        - (2 * za**3 - 5 * za) * (skewness**2) / 36
    )

    return returns.mean() + z_cornish_fisher * returns.std(ddof=1)


def _get_portfolio_var(
    portfolio_returns: pd.Series, alpha: float, method: str
) -> float:
    """
    Dispatch to the requested single-asset VaR calculation for a portfolio return
    series. Shared by `get_marginal_var` and, transitively, `get_component_var`.

    Args:
        portfolio_returns (pd.Series): The weighted portfolio return series.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        method (str): One of "historic", "gaussian", "cornish-fisher" or
            "studentt".

    Returns:
        float: The portfolio VaR.

    Raises:
        ValueError: If `method` is not one of the supported options.
    """
    if method == "historic":
        return get_var_historic(portfolio_returns, alpha)
    if method == "gaussian":
        return get_var_gaussian(portfolio_returns, alpha)
    if method == "cornish-fisher":
        return get_var_cornish_fisher(portfolio_returns, alpha)
    if method == "studentt":
        return get_var_studentt(portfolio_returns, alpha)

    raise InvalidArgumentError(
        f"method must be one of 'historic', 'gaussian', 'cornish-fisher' or 'studentt', "
        f"not {method!r}. Please pass one of these as method."
    )


def get_marginal_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float,
    method: str = "historic",
) -> pd.Series:
    """
    Calculate the Marginal Value at Risk (Marginal VaR) of each asset in a portfolio: how
    much the portfolio's VaR changes when the weight of that asset rises a little.

    Ordinary VaR (see `get_var_historic`) treats each asset in isolation. Marginal VaR
    instead measures how much the *portfolio's* VaR would change for an infinitesimal
    increase in a given asset's portfolio weight — i.e. the sensitivity of portfolio
    risk to each holding, not the risk of the holding on its own. VaR is homogeneous of
    degree one in the portfolio weights, so these sensitivities are the Euler
    allocation of the portfolio VaR.

    For the gaussian distribution the derivative has an exact closed form, which also
    accounts for the mean returns:

    - Portfolio Return = SUM(weight_i * Return_i)
    - Marginal VaR_i = Mean_i + z_alpha * Cov(Return_i, Portfolio Return) / Std(Portfolio Return)

    For the other distributions (historic, cornish-fisher and studentt) there is no
    such closed form, and the sensitivity is approximated with each asset's beta
    against the portfolio, which is exact for a zero-mean elliptical distribution:

    - Beta_i = Cov(Return_i, Portfolio Return) / Var(Portfolio Return)
    - Marginal VaR_i = Beta_i * Portfolio VaR

    The covariances and standard deviations are sample estimates (with n - 1 in the
    denominator), the same estimator as the portfolio VaR. An asset with Beta_i > 1 contributes
    disproportionately to portfolio risk (adding to it increases portfolio VaR by more
    than the asset's own weight would suggest), while Beta_i < 1 (and especially
    Beta_i < 0) indicates a diversifying holding. Returns are expected on common dates;
    rows with a missing return are dropped.

    For more information about the method, see the following sources:

    - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
    - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
    Series.
    - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
    Risk." 3rd ed., McGraw-Hill, Chapter 7.

    Also known as: Marginal VaR, MVaR.

    Shares similar logic:
    - PerformanceAnalytics (R), VaR.Marginal: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/VaR.Marginal.R#L1

    Args:
        returns (pd.DataFrame): A DataFrame of asset returns, one column per asset.
        weights (pd.Series): Portfolio weights, indexed by the same asset labels as
            `returns`'s columns. Normalized internally to sum to 1.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        method (str, optional): The estimator of the underlying portfolio VaR. Options
            are: "historic" (historical), "gaussian" (gaussian), "cornish-fisher"
            (Cornish-Fisher), "studentt" (Student-T). The gaussian decomposition is exact;
            the Beta-based decomposition of the others is an approximation. Defaults to
            "historic".

    Returns:
        pd.Series: The Marginal VaR of each asset, indexed the same as `returns`'s
            columns.

    Raises:
        TypeError: If `returns` is not a pd.DataFrame or `weights` is not a pd.Series.
        ValueError: If `weights` does not cover every column in `returns`, or if
            `method` is not one of the supported options.
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(weights, pd.Series):
        raise TypeError(
            "returns must be a pd.DataFrame and weights must be a pd.Series."
        )

    weights = weights.reindex(returns.columns)

    if weights.isna().any():
        raise ValueError("weights must be provided for every column in returns.")

    weights = weights / weights.sum()

    # The portfolio return is only defined on dates where every constituent has a return.
    returns = returns.dropna()

    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    portfolio_var = as_float(_get_portfolio_var(portfolio_returns, alpha, method))
    portfolio_variance = portfolio_returns.var(ddof=1)

    if portfolio_variance == 0 or np.isnan(portfolio_variance):
        return pd.Series(np.nan, index=returns.columns)

    covariance_with_portfolio = returns.apply(
        lambda column: column.cov(portfolio_returns, ddof=1)
    )

    if method == "gaussian":
        # The exact derivative of mean_p + z * std_p with respect to weight_i; the weighted sum of these is the portfolio VaR again, also with a non-zero mean (the beta form drops the mean).  # noqa: E501
        return returns.mean() + stats.norm.ppf(
            alpha
        ) * covariance_with_portfolio / np.sqrt(portfolio_variance)

    beta = covariance_with_portfolio / portfolio_variance

    return beta * portfolio_var


def get_component_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float,
    method: str = "historic",
) -> pd.Series:
    """
    Calculate the Component Value at Risk (Component VaR) of each asset in a portfolio: the
    part of the portfolio's VaR that each holding accounts for.

    Component VaR allocates total portfolio VaR across its constituent assets, such
    that the allocations sum exactly back to the portfolio VaR — an "Euler" (or
    "fully consistent") risk decomposition, following from the fact that VaR is
    homogeneous of degree one in the portfolio weights:

    - Component VaR_i = weight_i * Marginal VaR_i    (see `get_marginal_var`)
    - SUM(Component VaR_i) = Portfolio VaR

    Where Marginal VaR (see `get_marginal_var`) measures each asset's risk
    *sensitivity*, Component VaR measures its actual *contribution* in the same units
    as portfolio VaR, making it directly usable to identify which holdings account for
    the largest share of portfolio risk.

    For more information about the method, see the following sources:

    - Garman, M.B. (1997). "Taking VaR to Pieces." Risk, 10(10), 70-71.
    - Litterman, R. (1996). "Hot Spots and Hedges." Goldman Sachs Risk Management
    Series.
    - Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial
    Risk." 3rd ed., McGraw-Hill, Chapter 7.

    Also known as: Component VaR, CVaR (portfolio-decomposition sense, unrelated to
    Conditional VaR / Expected Shortfall), risk contribution.

    Shares similar logic:
    - PerformanceAnalytics (R), VaR: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/VaR.R#L329
    - Riskfolio-Lib, Risk_Contribution: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Risk_Contribution

    Args:
        returns (pd.DataFrame): A DataFrame of asset returns, one column per asset.
        weights (pd.Series): Portfolio weights, indexed by the same asset labels as
            `returns`'s columns. Normalized internally to sum to 1.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        method (str, optional): The estimator of the underlying portfolio VaR. Options
            are: "historic" (historical), "gaussian" (gaussian), "cornish-fisher"
            (Cornish-Fisher), "studentt" (Student-T). Defaults to "historic".

    Returns:
        pd.Series: The Component VaR of each asset, indexed the same as `returns`'s
            columns. Sums to the total portfolio VaR.

    Raises:
        TypeError: If `returns` is not a pd.DataFrame or `weights` is not a pd.Series.
        ValueError: If `weights` does not cover every column in `returns`, or if
            `method` is not one of the supported options.
    """
    marginal_var = get_marginal_var(returns, weights, alpha, method)

    weights = weights.reindex(returns.columns)
    weights = weights / weights.sum()

    return weights * marginal_var


def get_liquidity_cost(
    spreads: pd.Series | pd.DataFrame, alpha: float, spread_scaling: float | None = None
) -> pd.Series | float:
    """
    Calculate the exogenous liquidity cost of Bangia, Diebold, Schuermann and Stroughair
    (1999): what closing a position at the bid (or the ask) instead of the mid price costs
    when spreads are wide, half of a bad-case relative bid-ask spread.

    The formula is as follows:

    - Liquidity Cost = 0.5 * (Mean(S) + a * Std(S))

    Where S are the relative spreads (the spread divided by the mid price) and Std the
    sample standard deviation (with n - 1 in the denominator). Without a `spread_scaling` a, the paper's choice
    applies: a is set so that Mean(S) + a * Std(S) is the empirical (1 - alpha) percentile
    of the spreads, which makes the cost 0.5 * Quantile(S, 1 - alpha) (linear
    interpolation, as the historic VaR), since spreads are far from normally distributed.

    The spreads are best quoted spreads, such as the average relative spread of each day
    from quote data. Spreads estimated from daily prices (`get_corwin_schultz_spread`,
    `get_abdi_ranaldo_spread`, `get_edge_spread`) work too, but for liquid large caps they
    mostly measure volatility rather than the spread (Jahan-Parvar and Zikes, 2023): their
    monthly Corwin-Schultz spread puts a stock with a quoted spread of 1 to 2 basis points
    at a cost of about 40 basis points, while for small and illiquid stocks they are of
    the right order.

    Args:
        spreads (pd.Series | pd.DataFrame): The relative spreads as decimals, one column
            per asset for a DataFrame.
        alpha (float): The tail probability of the Value at Risk the cost is added to
            (e.g., 0.05 for the 95% confidence level).
        spread_scaling (float | None, optional): The scaling factor a of the spread
            volatility, such as 2.33 for the normal 99% quantile. Defaults to None, the
            empirical percentile.

    Returns:
        pd.Series | float: The liquidity cost as a decimal of the price, a float for a
        Series and per column for a DataFrame. NaN with fewer than two spreads.
    """
    if isinstance(spreads, pd.DataFrame):
        return pd.Series(
            {
                column: get_liquidity_cost(spreads[column], alpha, spread_scaling)
                for column in spreads.columns
            },
            dtype=float,
        )

    spreads = spreads.dropna()

    if len(spreads) < MINIMUM_SPREAD_OBSERVATIONS:
        return np.nan

    if spread_scaling is None:
        return 0.5 * spreads.quantile(1 - alpha, interpolation="linear")

    return 0.5 * (spreads.mean() + spread_scaling * spreads.std(ddof=1))


def get_liquidity_adjusted_value_at_risk(
    value_at_risk: pd.Series | float,
    spreads: pd.Series | pd.DataFrame,
    alpha: float,
    spread_scaling: float | None = None,
) -> pd.Series | float:
    """
    Calculate the liquidity-adjusted Value at Risk of Bangia, Diebold, Schuermann and
    Stroughair (1999): the bad-case loss of a position including the cost of closing it at
    a wide bid-ask spread (see `get_liquidity_cost`).

    The formula is as follows:

    - Liquidity-Adjusted VaR = VaR - Liquidity Cost

    The VaR is a return, negative for a loss, as the other Value at Risk functions return
    it, and the liquidity cost a positive decimal of the price, so the liquidity-adjusted
    VaR is the larger loss. The paper writes the same as a positive loss in price terms.
    The spreads and the returns of the VaR should cover the same history, and the spread
    is not scaled with the horizon of the VaR.

    Args:
        value_at_risk (pd.Series | float): The Value at Risk as a return, per asset for a
            Series (such as the result of `get_var_historic`).
        spreads (pd.Series | pd.DataFrame): The relative spreads as decimals, one column
            per asset for a DataFrame, matching the assets of `value_at_risk`.
        alpha (float): The tail probability of the Value at Risk (e.g., 0.05 for the 95%
            confidence level).
        spread_scaling (float | None, optional): The scaling factor of the spread
            volatility. Defaults to None, the empirical percentile (see
            `get_liquidity_cost`).

    Returns:
        pd.Series | float: The liquidity-adjusted Value at Risk as a return.
    """
    return value_at_risk - get_liquidity_cost(spreads, alpha, spread_scaling)


def get_var_garch(
    returns: pd.Series | pd.DataFrame,
    alpha: float,
    window_size: int | None = None,
    refit_every: int = 21,
    method: str = "garch",
    distribution: str = "normal",
    expected_shortfall: bool = False,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the conditional Value at Risk (VaR) of returns from a GARCH-family model: the
    alpha quantile of the model's distribution of the next return, given today's volatility.

    The formula is as follows:

    - VaR_(t+1) = Mu + Volatility_(t+1) * F^-1(alpha)
    - CVaR_(t+1) = Mu + Volatility_(t+1) * ES_F(alpha), for the Conditional Value at Risk

    Where Mu and Volatility_(t+1) are the constant mean and the one-step volatility forecast of
    the model fitted by maximum likelihood (see
    `garch_model.get_garch_family_rolling_forecast`), F^-1 the quantile of its standardized
    innovation distribution and ES_F(alpha) = E[z | z <= F^-1(alpha)] its expected shortfall
    (see `garch_model.get_standardized_quantile` and
    `garch_model.get_standardized_expected_shortfall`), both with the fitted Shape and Skew.

    Without a window size the model is fitted on all returns of every column (or of every
    period of a 'within period' index) and the result is the VaR of the period after the last
    return. With a window size the result is the path of the VaR of the next period at every
    date, from a model refitted every `refit_every` returns on the `window_size` returns
    before it, so each value only uses the returns up to its date.

    For more information, see the following sources:

    - Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
    Journal of Econometrics, 31(3), 307-327.
    - McNeil, A.J. and Frey, R. (2000). "Estimation of Tail-Related Risk Measures for
    Heteroscedastic Financial Time Series: An Extreme Value Approach." Journal of Empirical
    Finance, 7(3-4), 271-300.

    Referenced against:
    - arch, GARCH: https://bashtage.github.io/arch/univariate/generated/arch.univariate.GARCH.html

    Shares similar logic:
    - rugarch (R), ugarchroll: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L1141

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int | None, optional): The number of returns of every fit of a rolling
            path, at least 100. Defaults to None, which fits once on all returns.
        refit_every (int, optional): The number of returns between two fits of a rolling path.
            Defaults to 21.
        method (str, optional): The variance model of `garch_model.get_garch_family`. Options
            are: "garch" (GARCH), "gjr" (GJR-GARCH), "egarch" (EGARCH), "igarch" (IGARCH),
            "aparch" (APARCH), "tgarch" (threshold GARCH), "ngarch" (nonlinear GARCH),
            "avgarch" (absolute value GARCH), "nagarch" (nonlinear asymmetric GARCH),
            "component" (component GARCH), "figarch" (FIGARCH). Defaults to "garch".
        distribution (str, optional): The innovation distribution. Options are: "normal"
            (normal), "t" (Student-t), "skewt" (Hansen's skewed Student-t),
            "skewt_fernandez_steel" (Fernández and Steel's skewed Student-t), "ged"
            (generalized error distribution), "johnson_su" (Johnson's SU). Defaults to
            "normal".
        expected_shortfall (bool, optional): Whether to return the Conditional Value at Risk
            (expected shortfall) instead. Defaults to False.

    Returns:
        pd.Series | pd.DataFrame | float: The VaR (or CVaR) of the next period as a float for a
        Series, per column for a DataFrame and per period for a 'within period' index, or the
        rolling path with `window_size`. A column with fewer than 100 returns is NaN.
    """
    arguments = {
        "window_size": window_size,
        "refit_every": refit_every,
        "method": method,
        "distribution": distribution,
        "expected_shortfall": expected_shortfall,
    }

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            return pd.DataFrame(
                {
                    sub_period: get_var_garch(
                        returns.loc[sub_period], alpha, **arguments
                    )
                    for sub_period in periods
                }
            ).T

        values = {
            column: get_var_garch(returns[column], alpha, **arguments)
            for column in returns.columns
        }

        if window_size is None:
            return pd.Series(values, index=returns.columns, dtype=float)

        return pd.DataFrame(values, columns=returns.columns)

    if not isinstance(returns, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    returns = returns.dropna()
    window = len(returns) if window_size is None else window_size

    if window < MINIMUM_GARCH_OBSERVATIONS or len(returns) < window:
        if window_size is not None and window_size < MINIMUM_GARCH_OBSERVATIONS:
            raise ValueError(
                f"A GARCH fit needs a window of at least {MINIMUM_GARCH_OBSERVATIONS} returns, "
                f"received window_size={window_size}."
            )
        logger.warning(
            "The GARCH VaR needs at least %s returns, %s has %s, so it is NaN.",
            max(window, MINIMUM_GARCH_OBSERVATIONS),
            returns.name,
            len(returns),
        )
        return float("nan") if window_size is None else pd.Series(dtype=float)

    forecast = garch_model.get_garch_family_rolling_forecast(
        returns, window_size, refit_every, method, distribution, "constant"
    )
    parameter_names = list(forecast.columns[2:])
    tail = (
        garch_model.get_standardized_expected_shortfall
        if expected_shortfall
        else garch_model.get_standardized_quantile
    )

    # The distribution parameters change at a refit only, so the tail of each set is computed once.  # noqa: E501
    parameters = (
        list(forecast[parameter_names].itertuples(index=False, name=None))
        if parameter_names
        else [()] * len(forecast)
    )
    tails = {
        values: float(tail(alpha, distribution, values))
        for values in set(parameters)
        if np.all(np.isfinite(values))
    }
    standardized = pd.Series(
        [tails.get(values, np.nan) for values in parameters], index=forecast.index
    )

    value_at_risk = forecast["Mean"] + forecast["Volatility"] * standardized

    if window_size is None:
        return float(value_at_risk.iloc[-1])

    return value_at_risk
