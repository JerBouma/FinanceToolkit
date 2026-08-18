"""Conditional Value at Risk Model"""

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk import risk_model, var_model
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

ALPHA_CONSTRAINT = 0.5

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2


def get_cvar_historic(returns: pd.Series | pd.DataFrame, alpha: float) -> pd.Series:
    """
    Calculate the historical Conditional Value at Risk (CVaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_cvar_historic, alpha=alpha
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            value_at_risk = pd.concat(period_data_list, axis=1)

            return value_at_risk.T
        return returns.aggregate(get_cvar_historic, alpha=alpha)
    if isinstance(returns, pd.Series):
        return returns[
            returns <= var_model.get_var_historic(returns, alpha)
        ].mean()  # The actual calculation without data wrangling

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_rolling_cvar_historic(
    returns: pd.Series | pd.DataFrame, alpha: float, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling historical Conditional Value at Risk (CVaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling CVaR values with time as index.
    """

    def _cvar(window):
        value_at_risk = np.percentile(window, alpha * 100)
        tail_losses = window[window <= value_at_risk]

        return tail_losses.mean() if len(tail_losses) > 0 else np.nan

    return returns.rolling(window=window_size).apply(_cvar, raw=True)


def get_cvar_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the gaussian distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_gaussian(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    za = stats.norm.ppf(alpha, 0, 1)
    return returns.std(ddof=0) * -stats.norm.pdf(za) / alpha + returns.mean()


def get_cvar_studentt(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the Student-T distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_studentt(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    returns = pd.DataFrame(returns)

    # Fitting student-t parameters to the data; missing observations are dropped per column, since scipy's fit raises on any non-finite value while the mean below simply skips them.  # noqa: E501
    v, scale = np.array([]), np.array([])
    for col in returns.columns:
        col_v, _, col_scale = stats.t.fit(returns[col].dropna())
        v = np.append(v, col_v)
        scale = np.append(scale, col_scale)
    za = stats.t.ppf(1 - alpha, v)

    return -scale * (v + za**2) / (v - 1) * stats.t.pdf(za, v) / alpha + returns.mean()


def get_cvar_laplace(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the Laplace distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence). Note that `alpha` needs to be
        0.5 or below, since the closed-form Laplace Expected Shortfall below is only defined on the lower
        half of the distribution. NaN is returned (with a warning) for any higher `alpha`.

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_laplace(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # For formula see: https://en.wikipedia.org/wiki/Expected_shortfall#Laplace_distribution

    # Laplace variance is 2*b**2, so b is fitted to the data's variance.
    b = np.sqrt(returns.std(ddof=0) ** 2 / 2)

    if alpha > ALPHA_CONSTRAINT:
        # NaN rather than 0, which would read as "no tail risk at all" downstream.
        logger.warning(
            "The Laplace Conditional Value at Risk is only defined for an alpha of "
            "0.5 or below, %s was given. Returning NaN instead.",
            alpha,
        )

        return returns.mean() * np.nan

    return -b * (1 - np.log(2 * alpha)) + returns.mean()


def get_cvar_logistic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the logistic distribution.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_logistic(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # For formula see: https://en.wikipedia.org/wiki/Expected_shortfall#Logistic_distribution

    # Logistic variance is b**2*pi**2/3, so the scale is fitted to it.
    scale = np.sqrt(3 * returns.std(ddof=0) ** 2 / np.pi**2)

    return -scale * np.log(((1 - alpha) ** (1 - 1 / alpha)) / alpha) + returns.mean()


def get_cvar_cornish_fisher(
    returns: pd.Series | pd.DataFrame, alpha: float, number_of_quantiles: int = 1000
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on the Cornish-Fisher
    (modified Gaussian) expansion.

    There is no clean closed-form Expected Shortfall for the Cornish-Fisher expansion (unlike
    for e.g. the gaussian or Student-T distributions), since the expansion is a quantile
    transformation rather than a full probability density. Instead, this is computed by
    numerically integrating the Cornish-Fisher quantile function over the tail:

    - CVaR = (1 / alpha) * INTEGRAL from 0 to alpha of VaR_cf(p) dp

    Where `VaR_cf(p)` is the Cornish-Fisher VaR (see `get_var_cornish_fisher`) evaluated at
    quantile `p` instead of at `alpha`. The integral is approximated with a fine, evenly
    spaced grid of `number_of_quantiles` probabilities between 0 and `alpha`.

    Since the Cornish-Fisher expansion adjusts for skewness and kurtosis, this CVaR moves in
    the correct direction relative to the gaussian CVaR (`get_cvar_gaussian`): for negatively
    skewed, fat-tailed returns it shows more tail risk (a more negative CVaR), and for
    close-to-normal returns it converges to the gaussian CVaR.

    Also known as: modified CVaR, modified Expected Shortfall, mES.

    For more information about the method, see the following papers:

    - Cornish, E.A., & Fisher, R.A. (1938). "Moments and Cumulants in the Specification
    of Distributions." Revue de l'Institut International de Statistique, 5(4), 307-320.
    - Favre, L., & Galeano, J.A. (2002). "Mean-Modified Value-at-Risk Optimization with
    Hedge Funds." Journal of Alternative Investments, 5(2), 21-25.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        number_of_quantiles (int, optional): The number of points used to numerically
        integrate the Cornish-Fisher quantile function over the (0, alpha] tail. Defaults
        to 1000.

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_cornish_fisher(
                returns.loc[sub_period], alpha, number_of_quantiles
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    mean = returns.mean()
    std = returns.std(ddof=0)
    skewness = risk_model.get_skewness(returns)
    excess_kurtosis = risk_model.get_kurtosis(returns, fisher=True)

    # A fine grid in (0, alpha] to integrate the Cornish-Fisher quantile function.
    probabilities = np.linspace(alpha / number_of_quantiles, alpha, number_of_quantiles)
    za = stats.norm.ppf(probabilities)

    if isinstance(returns, pd.DataFrame):
        za = za[:, None]
        mean_values = mean.to_numpy()
        std_values = std.to_numpy()
        skewness_values = skewness.to_numpy()
        excess_kurtosis_values = excess_kurtosis.to_numpy()
    else:
        mean_values = mean
        std_values = std
        skewness_values = skewness
        excess_kurtosis_values = excess_kurtosis

    z_cornish_fisher = (
        za
        + (za**2 - 1) * skewness_values / 6
        + (za**3 - 3 * za) * excess_kurtosis_values / 24
        - (2 * za**3 - 5 * za) * (skewness_values**2) / 36
    )

    quantile_values = mean_values + z_cornish_fisher * std_values
    conditional_value_at_risk = quantile_values.mean(axis=0)

    if isinstance(returns, pd.DataFrame):
        return pd.Series(conditional_value_at_risk, index=returns.columns)

    return float(conditional_value_at_risk)


def get_cvar_evt(
    returns: pd.Series | pd.DataFrame,
    alpha: float,
    threshold_percentile: float = 0.95,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on Extreme Value Theory
    (EVT), using the same Generalized Pareto Distribution (GPD) Peak-over-Threshold fit as
    `var_model.get_var_evt`.

    Because the GPD has a closed-form mean above the threshold, the Expected Shortfall can be
    derived analytically from the same fitted shape (xi), scale (sigma) and threshold (u) used
    for the EVT VaR, rather than needing a separate simulation or numerical integration step:

    - ES = VaR / (1 - xi) + (sigma - xi * u) / (1 - xi)

    This is only valid for a shape parameter `xi < 1`; for `xi >= 1` the GPD's mean (and hence
    its Expected Shortfall) is undefined/infinite, so NaN is returned for that column instead.
    Since the Expected Shortfall averages over all losses beyond the VaR threshold, `abs(CVaR)`
    is always at least as large as `abs(VaR)` for the same `alpha`.

    For more information about the method, see the following paper:

    - McNeil, A.J., & Frey, R. (2000). "Estimation of Tail-Related Risk Measures for
    Heteroscedastic Financial Time Series: An Extreme Value Approach." Journal of Empirical
    Finance, 7(3-4), 271-300.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).
        threshold_percentile (float, optional): The percentile of losses above which the GPD is
        fitted (e.g. 0.95 fits the GPD on the worst 5% of losses). Defaults to 0.95.

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_evt(
                returns.loc[sub_period], alpha, threshold_percentile
            )
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    returns = pd.DataFrame(returns)

    shape, scale, threshold, exceedance_probability = var_model.fit_gpd_tail(
        returns, threshold_percentile
    )

    value_at_risk_loss = threshold + (scale / shape) * (
        (alpha / exceedance_probability) ** (-shape) - 1
    )

    # Closed-form GPD Expected Shortfall (McNeil & Frey, 2000), undefined at shape >= 1.
    expected_shortfall_loss = np.where(
        shape < 1,
        value_at_risk_loss / (1 - shape) + (scale - shape * threshold) / (1 - shape),
        np.nan,
    )

    conditional_value_at_risk = -expected_shortfall_loss

    return pd.Series(conditional_value_at_risk, index=returns.columns)
