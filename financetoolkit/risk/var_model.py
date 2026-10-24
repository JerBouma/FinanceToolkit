"""Value at Risk Model"""

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk import risk_model

ALPHA_CONSTRAINT = 0.5

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2

# A Student-T fit has three parameters, so fewer observations cannot identify it.
MINIMUM_STUDENT_T_OBSERVATIONS = 3

# Floating point slack when comparing alpha with the GPD exceedance probability.
ALPHA_TOLERANCE = 1e-12


def get_var_historic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the historical Value at Risk (VaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    return risk_model.get_quantile(returns, alpha)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_var_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Value at Risk (VaR) of returns based on the gaussian distribution.

    The formula is as follows:

    - VaR = mean + z_alpha * std

    Where `z_alpha` is the standard normal quantile at `alpha` and `std` is the sample
    standard deviation (ddof=1), the estimator every parametric VaR and CVaR in this
    module uses. See `get_var_cornish_fisher` for the version that corrects the
    quantile for the skewness and kurtosis of the returns.

    More material/resources:
     - "Finance Compact Plus" by Zimmerman; Part 1, page 130-131;
     - "Risk Management and Financial Institutions" by John C. Hull

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
            period_data = get_var_gaussian(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    za = stats.norm.ppf(alpha, 0, 1)

    return returns.mean() + za * returns.std(ddof=1)


def get_rolling_var_historic(
    returns: pd.Series | pd.DataFrame, alpha: float, window_size: int
) -> pd.Series | pd.DataFrame:
    """
    Calculate the rolling historical Value at Risk (VaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int): The size of the rolling window.

    Returns:
        pd.Series | pd.DataFrame: Rolling VaR values with time as index.
    """
    # Linear interpolation, as np.percentile, in one pass instead of a call per window.
    return returns.rolling(window=window_size).quantile(alpha, interpolation="linear")


def fit_gpd_tail(
    returns: pd.DataFrame, threshold_percentile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Generalized Pareto Distribution (GPD) to the losses of each column of `returns`
    that exceed a given threshold, following the Peak-over-Threshold approach.

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
        in the same order as `returns.columns`. A minimum number of exceedances is required
        for the GPD fit to be meaningful; if there are too few (e.g. a short sub-period), NaN
        is returned for the shape, scale and exceedance probability of that column.
    """
    minimum_number_of_exceedances = 2

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

        if len(exceedances) < minimum_number_of_exceedances:
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


def validate_evt_alpha(alpha: float, threshold_percentile: float) -> None:
    """
    Check that the Peak-over-Threshold VaR and CVaR are evaluated in the fitted tail.

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
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Value at Risk (VaR) of returns based on Extreme Value Theory (EVT), using a
    Peak-over-Threshold approach in which a Generalized Pareto Distribution (GPD) is fitted to
    the losses that exceed a given threshold.

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

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level). It needs
        to be at most 1 - `threshold_percentile`.
        threshold_percentile (float, optional): The percentile of losses above which the GPD is
        fitted (e.g. 0.95 fits the GPD on the worst 5% of losses). Defaults to 0.95.

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.

    Raises:
        ValueError: If `alpha` is larger than 1 - `threshold_percentile`, where the VaR
        would lie below the threshold the GPD is fitted above.
    """
    validate_evt_alpha(alpha, threshold_percentile)

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_var_evt(
                returns.loc[sub_period], alpha, threshold_percentile
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
    Fit a Student-T distribution to each column of `returns` by maximum likelihood.

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
    Calculate the Value at Risk (VaR) of returns based on the Student-T distribution.

    The degrees of freedom `v`, the location and the scale are estimated jointly by
    maximum likelihood (`scipy.stats.t.fit`), and the VaR is the corresponding quantile
    of the fitted distribution:

    - VaR = location + scale * t_v^-1(alpha)

    The fitted location is used rather than the sample mean so that the quantile
    belongs to one and the same fitted distribution (with fat tails the two can differ).
    Columns with fewer than three observations return NaN, since the three parameters
    cannot be identified.

    Note that the scale of a Student-T distribution is *not* its standard deviation:
    the two are related by `std = scale * SQRT(v / (v - 2))`, and only for `v > 2`,
    since a Student-T with two or fewer degrees of freedom has no finite Variance at
    all. Using the fitted scale rather than rescaling the sample standard deviation
    therefore keeps the estimate well defined for every fitted `v` (heavily fat-tailed
    return samples routinely fit below `v = 2`), and matches the convention used by
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

    # A Series gives a length one array, the type `_as_scalar` in the controller reduces.
    if isinstance(returns, pd.Series):
        return value_at_risk

    return pd.Series(value_at_risk, index=returns.columns)


def get_var_cornish_fisher(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Value at Risk (VaR) of returns based on the Cornish-Fisher (modified
    Gaussian) expansion.

    The Cornish-Fisher expansion adjusts the standard normal quantile for the sample
    skewness and (excess) kurtosis of the return distribution, which makes the resulting
    VaR more accurate than the plain gaussian VaR (`get_var_gaussian`) for returns that
    are not normally distributed, i.e. that are skewed and/or fat-tailed (as most asset
    returns are, see `econometrics.diagnostics_model.get_jarque_bera_test`).

    The formula is as follows:

    - z_cf = z + (z^2 - 1) * S / 6 + (z^3 - 3z) * K / 24 - (2z^3 - 5z) * S^2 / 36
    - VaR = mean + z_cf * std

    Where `z` is the gaussian quantile at `alpha`, `S` is the skewness, `K` is the
    excess (Fisher) kurtosis and `std` the sample standard deviation (ddof=1) of
    `returns`.

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


def _as_float(value: float | np.ndarray | pd.Series) -> float:
    """
    Reduce a single-value VaR (a float, or the length one array the Student-T VaR
    returns for a Series) to a float.

    Args:
        value (float | np.ndarray | pd.Series): The VaR.

    Returns:
        float: The VaR as a float.
    """
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _get_portfolio_var(
    portfolio_returns: pd.Series, alpha: float, distribution: str
) -> float:
    """
    Dispatch to the requested single-asset VaR calculation for a portfolio return
    series. Shared by `get_marginal_var` and, transitively, `get_component_var`.

    Args:
        portfolio_returns (pd.Series): The weighted portfolio return series.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        distribution (str): One of "historic", "gaussian", "cornish-fisher" or
            "studentt".

    Returns:
        float: The portfolio VaR.

    Raises:
        ValueError: If `distribution` is not one of the supported options.
    """
    if distribution == "historic":
        return get_var_historic(portfolio_returns, alpha)
    if distribution == "gaussian":
        return get_var_gaussian(portfolio_returns, alpha)
    if distribution == "cornish-fisher":
        return get_var_cornish_fisher(portfolio_returns, alpha)
    if distribution == "studentt":
        return get_var_studentt(portfolio_returns, alpha)

    raise ValueError(
        "distribution must be historic, gaussian, cornish-fisher or studentt."
    )


def get_marginal_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float,
    distribution: str = "historic",
) -> pd.Series:
    """
    Calculate the Marginal Value at Risk (Marginal VaR) of each asset in a portfolio.

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

    The covariances and standard deviations are sample estimates (ddof=1), the same
    estimator as the portfolio VaR. An asset with Beta_i > 1 contributes
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

    Args:
        returns (pd.DataFrame): A DataFrame of asset returns, one column per asset.
        weights (pd.Series): Portfolio weights, indexed by the same asset labels as
            `returns`'s columns. Normalized internally to sum to 1.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        distribution (str): The distribution to use for the underlying portfolio VaR
            calculation (historic, gaussian, cornish-fisher or studentt). The gaussian
            decomposition is exact; the Beta-based decomposition of the others is an
            approximation. Defaults to "historic".

    Returns:
        pd.Series: The Marginal VaR of each asset, indexed the same as `returns`'s
            columns.

    Raises:
        TypeError: If `returns` is not a pd.DataFrame or `weights` is not a pd.Series.
        ValueError: If `weights` does not cover every column in `returns`, or if
            `distribution` is not one of the supported options.
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
    portfolio_var = _as_float(
        _get_portfolio_var(portfolio_returns, alpha, distribution)
    )
    portfolio_variance = portfolio_returns.var(ddof=1)

    if portfolio_variance == 0 or np.isnan(portfolio_variance):
        return pd.Series(np.nan, index=returns.columns)

    covariance_with_portfolio = returns.apply(
        lambda column: column.cov(portfolio_returns, ddof=1)
    )

    if distribution == "gaussian":
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
    distribution: str = "historic",
) -> pd.Series:
    """
    Calculate the Component Value at Risk (Component VaR) of each asset in a portfolio.

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

    Args:
        returns (pd.DataFrame): A DataFrame of asset returns, one column per asset.
        weights (pd.Series): Portfolio weights, indexed by the same asset labels as
            `returns`'s columns. Normalized internally to sum to 1.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        distribution (str): The distribution to use for the underlying portfolio VaR
            calculation (historic, gaussian, cornish-fisher or studentt). Defaults to
            "historic".

    Returns:
        pd.Series: The Component VaR of each asset, indexed the same as `returns`'s
            columns. Sums to the total portfolio VaR.

    Raises:
        TypeError: If `returns` is not a pd.DataFrame or `weights` is not a pd.Series.
        ValueError: If `weights` does not cover every column in `returns`, or if
            `distribution` is not one of the supported options.
    """
    marginal_var = get_marginal_var(returns, weights, alpha, distribution)

    weights = weights.reindex(returns.columns)
    weights = weights / weights.sum()

    return weights * marginal_var
