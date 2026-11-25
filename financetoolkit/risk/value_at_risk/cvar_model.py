"""Conditional Value at Risk Model"""

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk import risk_model
from financetoolkit.risk.value_at_risk import var_model
from financetoolkit.utilities.error_model import validate_window
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.statistics_model import by_own_trading_days

logger = get_logger()

ALPHA_CONSTRAINT = 0.5

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2


@by_own_trading_days(rolling_only=True)
def get_cvar_historic(
    returns: pd.Series | pd.DataFrame, alpha: float, window: int | None = None
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the historical Conditional Value at Risk (CVaR) of returns: the average return
    in the worst alpha share of the observed periods, over the whole sample or within every
    rolling window.

    The formula is as follows:

        - CVaR = Mean(Return | Return <= VaR), with VaR the alpha quantile of the returns

    Shares similar logic:
    - PerformanceAnalytics (R), ES: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ES.R#L245
    - empyrical, conditional_value_at_risk: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L2115
    - quantstats, conditional_value_at_risk: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2086
    - Riskfolio-Lib, CVaR_Hist: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.CVaR_Hist
    - skfolio, cvar: https://skfolio.org/generated/skfolio.measures.cvar.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window (int | None, optional): The number of observations of a rolling window, at least 1. Defaults
            to None, which calculates the CVaR over all returns (per period for a 'within
            period' index).

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    validate_window("window", window, 1)
    if window is not None:

        def _cvar(windows):
            value_at_risk = np.percentile(windows, alpha * 100, axis=1, keepdims=True)

            # The value at the percentile is always in the tail, so no window has an empty one.
            return np.nanmean(
                np.where(windows <= value_at_risk, windows, np.nan), axis=1
            )

        return risk_model.apply_to_stacked_windows(returns, window, _cvar)

    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            value_at_risk = returns.groupby(level=0, sort=False).quantile(alpha)
            value_at_risk = value_at_risk.reindex(
                returns.index.get_level_values(0)
            ).set_axis(returns.index)
            # A period with a single return has no tail to average, see risk_model.get_quantile.
            result = (
                returns.where(returns <= value_at_risk)
                .groupby(level=0, sort=False)
                .mean()
                .where(
                    returns.groupby(level=0, sort=False).count()
                    >= risk_model.MINIMUM_QUANTILE_OBSERVATIONS
                )
            )
            result.index.name = None

            return result

        value_at_risk = returns.quantile(alpha)

        return returns.where(returns <= value_at_risk).mean()
    if isinstance(returns, pd.Series):
        return returns[
            returns <= var_model.get_var_historic(returns, alpha)
        ].mean()  # The actual calculation without data wrangling

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_cvar_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the gaussian Conditional Value at Risk (CVaR) of returns: the average return
    below the alpha quantile of a normal distribution with the mean and standard deviation of
    the returns.

    The formula is as follows:

    - CVaR = mean - std * phi(z_alpha) / alpha

    Where `phi` is the standard normal density, `z_alpha` the standard normal quantile at
    `alpha` and `std` the sample standard deviation (with n - 1 in the denominator).

    Shares similar logic:
    - PerformanceAnalytics (R), ES: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ES.R#L245

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns. A rolling
            window of them (`returns.rolling(window_size)`) works too and gives the CVaR of
            every window, as the out-of-sample estimate of the risk module uses.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

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
    return returns.std(ddof=1) * -stats.norm.pdf(za) / alpha + returns.mean()


def get_cvar_kernel(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the kernel Conditional Value at Risk (CVaR) of returns: the average return below
    the kernel VaR under a smoothed estimate of the return distribution.

    The distribution is a Gaussian kernel density estimate with Silverman's bandwidth
    (Scaillet 2004), which gives a closed form given the VaR:

    - CVaR = mean(Return_i * Φ(u_i) - h * φ(u_i)) / alpha
    - u_i = (VaR - Return_i) / h

    Where VaR is the kernel VaR of `var_model.get_var_kernel`, h the bandwidth of
    `var_model.get_kernel_bandwidth` and Φ and φ the standard normal distribution and density.

    For more information see:

    - Scaillet, O. (2004). "Nonparametric Estimation and Sensitivity Analysis of Expected
    Shortfall." Mathematical Finance, 14(1), 115-129.

    Referenced against:
    - scipy, gaussian_kde: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

    Returns:
        pd.Series | pd.DataFrame | float: The CVaR as a return, negative for a loss.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie between 0 and 1.")

    def kernel_cvar(samples: np.ndarray) -> np.ndarray:
        value_at_risk = var_model.get_var_kernel_samples(samples, alpha)[:, np.newaxis]
        bandwidth = var_model.get_kernel_bandwidth(samples)[:, np.newaxis]
        # A sample of equal returns has no spread to smooth: its CVaR is that return.
        spread = np.where(bandwidth > 0, bandwidth, 1.0)
        standardized = (value_at_risk - samples) / spread
        tail = (
            samples * stats.norm.cdf(standardized)
            - spread * stats.norm.pdf(standardized)
        ).mean(axis=1) / alpha

        return np.where(bandwidth[:, 0] > 0, tail, samples[:, 0])

    return risk_model.apply_to_observations(returns, kernel_cvar)


def get_weighted_cvar(
    conditional_values_at_risk: list[pd.Series | pd.DataFrame], weights: list[float]
) -> pd.Series | pd.DataFrame:
    """
    Combine the Conditional Value at Risk at several tail probabilities into the Weighted
    (spectral) Conditional Value at Risk: one tail loss that weighs the deeper tails as chosen.

    It is the weighted sum of the CVaRs (Acerbi 2002, Cajas 2021):

    - Weighted CVaR = Σ_k Weight_k * CVaR(alpha_k)

    With non-negative weights that sum to 1 it is a spectral risk measure: a weighted average
    of the returns that weighs every return at least as much as any better return.

    Args:
        conditional_values_at_risk (list[pd.Series | pd.DataFrame]): The CVaR at every tail
            probability, on the same index.
        weights (list[float]): The weight of each, non-negative and summing to 1.

    Returns:
        pd.Series | pd.DataFrame: The Weighted CVaR as a return, negative for a loss.
    """
    if len(weights) != len(conditional_values_at_risk):
        raise ValueError("Give one weight per alpha.")
    if min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("The weights must be non-negative and sum to 1.")

    return sum(
        weight * conditional_value_at_risk
        for weight, conditional_value_at_risk in zip(
            weights, conditional_values_at_risk, strict=True
        )
    )


def get_cvar_studentt(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Student-T Conditional Value at Risk (CVaR) of returns: the average return
    below the alpha quantile of a fat-tailed Student-T distribution fitted to the returns.

    The degrees of freedom `v`, the location and the scale are fitted by maximum likelihood
    (see `var_model.fit_student_t`, the same fit as `var_model.get_var_studentt`), and the
    Expected Shortfall of the fitted distribution is:

    - CVaR = location - scale * (v + t^2) / (v - 1) * f_v(t) / alpha

    Where t = t_v^-1(1 - alpha) and `f_v` is the Student-T density. The Expected
    Shortfall only exists when the mean does, i.e. for v > 1; NaN is returned for a
    column whose fitted degrees of freedom are 1 or lower (or that has fewer than three
    observations).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

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

    v, location, scale = var_model.fit_student_t(returns)
    za = stats.t.ppf(1 - alpha, v)

    # The tail mean diverges for v <= 1 (no finite mean), where the formula would flip sign.
    conditional_value_at_risk = np.where(
        v > 1,
        location - scale * (v + za**2) / (v - 1) * stats.t.pdf(za, v) / alpha,
        np.nan,
    )

    return pd.Series(conditional_value_at_risk, index=returns.columns)


def get_cvar_laplace(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Laplace Conditional Value at Risk (CVaR) of returns: the average return below
    the alpha quantile of a Laplace (double exponential) distribution with the mean and spread
    of the returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level). Note that `alpha` needs to be
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

    # Laplace variance is 2*b**2, so b is fitted to the data's (sample) variance.
    b = np.sqrt(returns.std(ddof=1) ** 2 / 2)

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
    Calculate the logistic Conditional Value at Risk (CVaR) of returns: the average return below
    the alpha quantile of a logistic distribution with the mean and standard deviation of the
    returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

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

    # Logistic variance is b**2*pi**2/3, so the scale is fitted to the (sample) variance.
    scale = np.sqrt(3 * returns.std(ddof=1) ** 2 / np.pi**2)

    return -scale * np.log(((1 - alpha) ** (1 - 1 / alpha)) / alpha) + returns.mean()


def get_cvar_cornish_fisher(
    returns: pd.Series | pd.DataFrame, alpha: float, number_of_quantiles: int = 1000
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Cornish-Fisher (modified) Conditional Value at Risk (CVaR) of returns: the
    average return below the gaussian quantile corrected for the skewness and fat tails of
    the returns.

    There is no clean closed-form Expected Shortfall for the Cornish-Fisher expansion (unlike
    for e.g. the gaussian or Student-T distributions), since the expansion is a quantile
    transformation rather than a full probability density. Instead, this is computed by
    numerically integrating the Cornish-Fisher quantile function over the tail:

    - CVaR = (1 / alpha) * INTEGRAL from 0 to alpha of VaR_cf(p) dp

    Where `VaR_cf(p)` is the Cornish-Fisher VaR (see `get_var_cornish_fisher`) evaluated at
    quantile `p` instead of at `alpha`. The integral is approximated with the midpoint rule
    on `number_of_quantiles` equally wide intervals between 0 and `alpha`, i.e. at the
    probabilities (i + 0.5) * alpha / number_of_quantiles. The midpoints avoid both p = 0,
    where the quantile is infinite, and the bias of evaluating every interval at its right
    edge.

    The Cornish-Fisher quantile is only monotone for moderate skewness and excess kurtosis
    (Maillard, 2012). For large excess kurtosis or strong skewness it bends back in the far
    tail, and the integral then understates the tail loss; compare with
    `get_cvar_historic` or `get_cvar_evt` in that case.

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
    - Maillard, D. (2012). "A User's Guide to the Cornish Fisher Expansion." SSRN
    Working Paper 1997178.

    Shares similar logic:
    - PerformanceAnalytics (R), ES: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/ES.R#L245

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        number_of_quantiles (int, optional): The number of midpoints used to numerically
        integrate the Cornish-Fisher quantile function over the (0, alpha) tail. Defaults
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
    std = returns.std(ddof=1)
    skewness = risk_model.get_skewness(returns)
    excess_kurtosis = risk_model.get_kurtosis(returns, fisher=True)

    # Midpoint rule over (0, alpha): the interval midpoints, never 0 where the quantile is infinite.  # noqa: E501
    probabilities = (np.arange(number_of_quantiles) + 0.5) * alpha / number_of_quantiles
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
    warn: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Conditional Value at Risk (CVaR) of returns based on Extreme Value Theory
    (EVT): the average loss beyond the VaR under a distribution fitted to the extreme losses.

    It uses the same Generalized Pareto Distribution (GPD) Peak-over-Threshold fit as
    `var_model.get_var_evt`.

    Because the GPD has a closed-form mean above the threshold, the Expected Shortfall can be
    derived analytically from the same fitted shape (xi), scale (sigma) and threshold (u) used
    for the EVT VaR, rather than needing a separate simulation or numerical integration step:

    - ES = VaR / (1 - xi) + (sigma - xi * u) / (1 - xi)

    This is only valid for a shape parameter xi < 1; for xi >= 1 the GPD's mean (and hence
    its Expected Shortfall) is undefined/infinite, so NaN is returned for that column instead.
    Since the Expected Shortfall averages over all losses beyond the VaR threshold, `abs(CVaR)`
    is always at least as large as `abs(VaR)` for the same `alpha`.

    For more information about the method, see the following paper:

    - McNeil, A.J., & Frey, R. (2000). "Estimation of Tail-Related Risk Measures for
    Heteroscedastic Financial Time Series: An Extreme Value Approach." Journal of Empirical
    Finance, 7(3-4), 271-300.

    Shares similar logic:
    - empyrical, gpd_risk_estimates: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1675

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level). It needs
        to be at most 1 - `threshold_percentile`.
        threshold_percentile (float, optional): The percentile of losses above which the GPD is
        fitted (e.g. 0.95 fits the GPD on the worst 5% of losses). Defaults to 0.95.
        warn (bool, optional): Whether to log the columns and periods with fewer than 10 losses
        above the threshold, for which the CVaR is NaN (see `var_model.fit_gpd_tail`).
        Defaults to True.

    Returns:
        pd.Series | pd.DataFrame: CVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `alpha` is larger than 1 - `threshold_percentile`, where the VaR
        would lie below the threshold the GPD is fitted above.
    """
    var_model.validate_evt_alpha(alpha, threshold_percentile)

    if warn:
        var_model.warn_too_few_exceedances(returns, threshold_percentile)

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_cvar_evt(
                returns.loc[sub_period], alpha, threshold_percentile, warn=False
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


def get_cvar_garch(
    returns: pd.Series | pd.DataFrame,
    alpha: float,
    window_size: int | None = None,
    refit_every: int = 21,
    method: str = "garch",
    distribution: str = "normal",
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the conditional Conditional Value at Risk (CVaR, expected shortfall) of returns
    from a GARCH-family model: the average of the model's distribution of the next return
    below its alpha quantile, given today's volatility.

    The formula is as follows:

    - CVaR_(t+1) = Mu + Volatility_(t+1) * ES_F(alpha), with ES_F(alpha) = E[z | z <= F^-1(alpha)]

    Where Mu and Volatility_(t+1) are the constant mean and the one-step volatility forecast of
    the fitted model and F the distribution function of its standardized innovations (see
    `var_model.get_var_garch`, which also describes `window_size` and `refit_every`).

    For more information, see the following paper:

    - McNeil, A.J. and Frey, R. (2000). "Estimation of Tail-Related Risk Measures for
    Heteroscedastic Financial Time Series: An Extreme Value Approach." Journal of Empirical
    Finance, 7(3-4), 271-300.

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

    Returns:
        pd.Series | pd.DataFrame | float: The CVaR of the next period, shaped as the VaR of
        `var_model.get_var_garch`.
    """
    return var_model.get_var_garch(
        returns,
        alpha,
        window_size,
        refit_every,
        method,
        distribution,
        expected_shortfall=True,
    )


# The halvings of the bisection of the higher moment coherent risk: the bracket shrinks below
# the spacing of floats long before, see get_higher_moment_coherent_risk_samples.
HIGHER_MOMENT_BISECTIONS = 200


def get_higher_moment_coherent_risk_samples(
    samples: np.ndarray, order: float, alpha: float
) -> np.ndarray:
    """
    Calculate the Higher Moment Coherent Risk (HMCR) of every row of an array of returns
    without missing values: the tail loss that weighs the deepest losses more, as
    `get_higher_moment_coherent_risk` reports it.

    With the losses L = -Return, X = L - ζ and the mean over the n returns of a row:

    - HMCR = min over ζ of ζ + (1 / alpha) * (mean(max(X, 0)^order))^(1 / order)
    - dHMCR / dζ = 1 - (1 / alpha) * mean(max(X, 0)^(order - 1)) / mean(max(X, 0)^order)^((order - 1) / order)

    The objective is convex in ζ, so its derivative rises with ζ and the minimum is where it
    changes sign, found by bisection on all rows at once. Above the largest loss the objective is
    ζ itself, so the minimum is at most the largest loss. As the norm is at least the mean
    (Lyapunov), the objective lies above mean(L) / alpha + ζ * (1 - 1 / alpha), which exceeds the
    largest loss below ζ = (mean(L) - alpha * max(L)) / (1 - alpha): the bracket starts at the
    lower of that and the smallest loss. Just below the largest loss (a single one) the derivative
    is 1 - n^(-1 / order) / alpha, so with n^(-1 / order) >= alpha the minimum is the largest loss.

    Args:
        samples (np.ndarray): The returns, one sample per row.
        order (float): The order p of the moment, at least 1 (1 gives the Conditional Value at
            Risk of Rockafellar and Uryasev, 2 the coherent second moment).
        alpha (float): The tail probability, between 0 and 1.

    Returns:
        np.ndarray: The HMCR of every row as a return, negative for a loss.
    """
    losses = -samples
    largest = losses.max(axis=1)
    lower = np.minimum(
        losses.min(axis=1), (losses.mean(axis=1) - alpha * largest) / (1 - alpha)
    )
    upper = largest.astype(float)

    def objective(threshold: np.ndarray) -> np.ndarray:
        excess = np.maximum(losses - threshold[:, None], 0)

        return threshold + np.mean(excess**order, axis=1) ** (1 / order) / alpha

    for _ in range(HIGHER_MOMENT_BISECTIONS):
        middle = (lower + upper) / 2

        # Every bracket is down to neighbouring floats.
        if np.all((middle == lower) | (middle == upper)):
            break

        excess = np.maximum(losses - middle[:, None], 0)
        moment = np.mean(excess**order, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            lower_moment = (
                np.mean(excess > 0, axis=1)
                if order == 1
                else np.mean(excess ** (order - 1), axis=1)
            )
            slope = 1 - lower_moment / moment ** ((order - 1) / order) / alpha

        # No loss above the threshold: the objective is the threshold itself, rising.
        slope = np.where(moment > 0, slope, 1.0)
        falling = slope < 0
        lower = np.where(falling, middle, lower)
        upper = np.where(falling, upper, middle)

    return -np.minimum(objective(lower), objective(upper))


def get_higher_moment_coherent_risk(
    returns: pd.Series | pd.DataFrame,
    order: float = 2,
    alpha: float = 0.05,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Higher Moment Coherent Risk (HMCR) of returns (Krokhmal 2007): a tail loss
    like the Conditional Value at Risk that weighs the deepest losses more.

    The coherent risk measure extends the Conditional Value at Risk from the first to a higher
    moment of the losses beyond a threshold:

    - HMCR = min over ζ of ζ + (1 / alpha) * (mean(max(-Return - ζ, 0)^order))^(1 / order)

    With an order of 1 it is the Conditional Value at Risk of Rockafellar and Uryasev (2002),
    with an order of 2 the coherent second moment (CSM). A higher order weighs the deepest losses more, so
    the HMCR rises with the order towards the worst loss. The mean runs over the n returns, so
    the measure does not depend on the length of the sample, see
    `get_higher_moment_coherent_risk_samples` for the minimization.

    For more information see:

    - Krokhmal, P.A. (2007). "Higher Moment Coherent Risk Measures." Quantitative Finance,
    7(4), 373-387.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        order (float, optional): The order p of the moment, at least 1. Defaults to 2.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The HMCR as a return, negative for a loss.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie between 0 and 1.")
    if order < 1:
        raise ValueError("The order must be at least 1.")

    return risk_model.apply_to_observations(
        returns,
        lambda samples: get_higher_moment_coherent_risk_samples(samples, order, alpha),
        window_size,
    )


def get_expected_quadratic_shortfall_samples(
    samples: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Calculate the Expected Quadratic Shortfall (EQS) of every row of an array of returns
    without missing values: the average squared loss in the tail, as
    `get_expected_quadratic_shortfall` reports it.

    The minimum of the Rockafellar and Uryasev (2002) form over ζ is the average of the alpha
    share of the largest squared losses, counting the squared loss at the boundary with the
    fraction of it inside the tail:

    - EQS = (Σ_{i <= k} S_(i) + (n * alpha - k) * S_(k + 1)) / (n * alpha)

    Where S_(1) >= S_(2) >= ... are the squared losses max(-Return, 0)^2 in descending order
    and k = floor(n * alpha).

    Args:
        samples (np.ndarray): The returns, one sample per row.
        alpha (float): The tail probability, between 0 and 1.

    Returns:
        np.ndarray: The EQS of every row, in squared returns and positive.
    """
    squared_losses = -np.sort(-np.maximum(-samples, 0) ** 2, axis=1)
    tail = samples.shape[1] * alpha
    whole = int(np.floor(tail))
    boundary = squared_losses[:, min(whole, samples.shape[1] - 1)]

    return (squared_losses[:, :whole].sum(axis=1) + (tail - whole) * boundary) / tail


def get_expected_quadratic_shortfall(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Expected Quadratic Shortfall (EQS) of returns: the average squared loss in
    the worst alpha share of the periods, the Conditional Value at Risk of the squared losses.

    It takes the Rockafellar and Uryasev (2002) form:

    - EQS = min over ζ of ζ + (1 / (n * alpha)) * Σ max(max(-Return, 0)^2 - ζ, 0)

    It is the average of the alpha share of the largest squared losses, so it weighs a loss by
    its square, as a semi-variance does, but only in the tail. Gains count as no loss. It is
    convex and monotone in the losses, so it suits a portfolio objective, but it is not a
    coherent measure: it scales with the square of the position and does not shift with cash.
    Its square root is a tail root mean square loss on the scale of the returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The EQS in squared returns, positive.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must lie between 0 and 1.")

    return risk_model.apply_to_observations(
        returns,
        lambda samples: get_expected_quadratic_shortfall_samples(samples, alpha),
        window_size,
    )
