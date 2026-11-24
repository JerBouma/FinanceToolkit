"""CoVaR, Marginal Expected Shortfall and SRISK (Systemic Risk) Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from financetoolkit.risk.helpers import get_paired_windows
from financetoolkit.utilities.statistics_model import validate_period

MINIMUM_COVAR_OBSERVATIONS = 10

# The fewest daily returns a beta is estimated from, as for the regressions of the
# Idiosyncratic Volatility (Ang, Hodrick, Xing and Zhang (2006) require 15 within a month).
MINIMUM_BETA_OBSERVATIONS = 15
MEDIAN_QUANTILE = 0.5


def _quantile_regression(
    y: np.ndarray, x: np.ndarray, tau: float
) -> tuple[float, float]:
    """
    Fit a simple (one-predictor) linear Quantile Regression of `y` on `x` at quantile
    `tau`, via the standard Koenker & Bassett (1978) Linear Program formulation:

    - minimize: tau * SUM(u_plus) + (1 - tau) * SUM(u_minus)
    - subject to: intercept + slope * x_t + u_plus_t - u_minus_t = y_t,
    u_plus_t >= 0, u_minus_t >= 0

    Unlike Ordinary Least Squares (which minimizes squared residuals and therefore
    estimates the conditional mean), this minimizes an asymmetric ("pinball") loss
    that is minimized at the conditional `tau`-quantile of `y` given `x`. Shared by
    `covar_model.get_covar` (the CoVaR quantile regression is fit at `tau = alpha`,
    the tail quantile, and again at `tau = 0.5`, the median, to isolate the tail-
    specific spillover from the "normal state" relationship).

    Args:
        y (np.ndarray): The dependent variable.
        x (np.ndarray): The (single) predictor.
        tau (float): The quantile to fit, in (0, 1).

    Returns:
        tuple[float, float]: The fitted intercept and slope.
    """
    n = len(y)
    number_of_variables = 2 + 2 * n

    # No cost on intercept/slope, asymmetric cost on the positive/negative parts.
    c = np.concatenate([[0.0, 0.0], np.full(n, tau), np.full(n, 1 - tau)])

    rows = np.repeat(np.arange(n), 4)
    cols = np.column_stack(
        [
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
            2 + np.arange(n),
            2 + n + np.arange(n),
        ]
    ).ravel()
    data = np.column_stack([np.ones(n), x, np.ones(n), -np.ones(n)]).ravel()

    a_eq = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n, number_of_variables)
    ).tocsr()

    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)

    result = optimize.linprog(c, A_eq=a_eq, b_eq=y, bounds=bounds, method="highs")

    if not result.success:
        return np.nan, np.nan

    return float(result.x[0]), float(result.x[1])


def get_covar(
    returns: pd.Series,
    conditioning_returns: pd.Series,
    alpha: float,
) -> pd.Series:
    """
    Calculate the (Delta-)CoVaR of `returns`: how much worse their Value at Risk gets when
    `conditioning_returns` is in its own distress state, the spillover of tail risk.

    Ordinary Value at Risk (see `get_var_historic`) treats each asset in isolation,
    which misses systemic risk -- the fact that one institution's or asset's distress
    can spill over and worsen another's risk. CoVaR (short for "Conditional VaR",
    unrelated to `cvar_model`'s Conditional Value at Risk / Expected Shortfall)
    directly measures that spillover: it is the VaR of `returns`, conditional on
    `conditioning_returns` itself being at its own `alpha`-VaR:

    - P(returns <= CoVaR | conditioning_returns = VaR_alpha(conditioning_returns)) = alpha

    This is estimated by fitting a linear Quantile Regression of `returns` on
    `conditioning_returns` at quantile `alpha`, then evaluating the fitted line at
    `conditioning_returns`'s own historical VaR:

    - returns = a_alpha + b_alpha * conditioning_returns + e   (fit at quantile alpha)
    - CoVaR = a_alpha + b_alpha * VaR_alpha(conditioning_returns)

    The Delta-CoVaR isolates the marginal, distress-specific contribution by holding
    the *same* (single) alpha-quantile regression fixed and instead comparing what it
    predicts when `conditioning_returns` is at its own distress VaR versus at its
    median -- i.e. the same `a_alpha`/`b_alpha` used above, evaluated at two different
    values of `conditioning_returns` (this is the construction in Adrian & Brunnermeier
    (2016), Section II.B: they do not re-fit a separate median-quantile regression):

    - Delta-CoVaR = CoVaR - (a_alpha + b_alpha * median(conditioning_returns))
    - = b_alpha * (VaR_alpha(conditioning_returns) - median(conditioning_returns))

    A large (negative) Delta-CoVaR means `conditioning_returns` being in distress
    meaningfully worsens the tail risk of `returns` beyond their typical (median-state)
    relationship -- i.e. there is systemic spillover, not just an unconditionally
    higher/lower level of risk.

    For more information about the method, see the following paper:

    - Adrian, T., & Brunnermeier, M.K. (2016). "CoVaR." American Economic Review,
    106(7), 1705-1741.

    Also known as: Conditional Value at Risk (systemic risk sense), Delta-CoVaR.

    Args:
        returns (pd.Series): The returns of the asset/institution whose conditional
        VaR is being measured.
        conditioning_returns (pd.Series): The returns of the asset/institution (or
        e.g. a financial-system index) whose distress `returns` is conditioned on.
        alpha (float): The tail probability for both the tail quantile regression and
        the VaR of `conditioning_returns` (e.g., 0.05 for the 95% confidence level).

    Returns:
        pd.Series: The CoVaR, the Delta-CoVaR, the tail (alpha-quantile) Quantile
        Regression slope and intercept, and the number of (paired, non-missing)
        observations used.

    Raises:
        TypeError: If `returns` or `conditioning_returns` is not a pd.Series.
    """
    if not isinstance(returns, pd.Series) or not isinstance(
        conditioning_returns, pd.Series
    ):
        raise TypeError("Expects pd.Series, no other value.")

    aligned = pd.concat([returns, conditioning_returns], axis=1, join="inner").dropna()
    n = len(aligned)

    if n <= MINIMUM_COVAR_OBSERVATIONS:
        return pd.Series(
            {
                "CoVaR": np.nan,
                "Delta-CoVaR": np.nan,
                "Quantile Regression Slope": np.nan,
                "Quantile Regression Intercept": np.nan,
                "Observations": n,
            }
        )

    y = aligned.iloc[:, 0].to_numpy()
    x = aligned.iloc[:, 1].to_numpy()

    intercept_alpha, slope_alpha = _quantile_regression(y, x, alpha)

    var_conditioning_alpha = np.percentile(x, alpha * 100)
    median_conditioning = np.percentile(x, MEDIAN_QUANTILE * 100)

    covar = intercept_alpha + slope_alpha * var_conditioning_alpha
    # The same alpha-quantile regression at the median, not a refit median one.
    delta_covar = slope_alpha * (var_conditioning_alpha - median_conditioning)

    return pd.Series(
        {
            "CoVaR": covar,
            "Delta-CoVaR": delta_covar,
            "Quantile Regression Slope": slope_alpha,
            "Quantile Regression Intercept": intercept_alpha,
            "Observations": n,
        }
    )


def get_marginal_expected_shortfall(
    returns: pd.Series | pd.DataFrame,
    market_returns: pd.Series,
    alpha: float = 0.05,
    period: str = "yearly",
    rolling: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Marginal Expected Shortfall (MES) per period: the average return of an
    asset on the days the market return is in its worst `alpha` tail.

    Acharya, Pedersen, Philippon and Richardson (2017) show that the contribution of a
    financial institution to the Expected Shortfall of the system is its MES, the return it
    is expected to make when the system as a whole is in its tail. They measure it as the
    equally weighted average return of a firm on the 5% worst days of the market over a
    year of daily returns:

    - VaR_alpha(Market) = Quantile(Market Returns, alpha)
    - MES = MEAN(Returns_t | Market Return_t <= VaR_alpha(Market))

    The MES is reported as a return, negative for a loss, as the Value at Risk and the
    Conditional Value at Risk of the Risk module are (the paper reports the loss, -MES).
    The quantile interpolates linearly between the two nearest returns. The market return
    on the days the asset did not trade is compounded into the next shared date, and a
    period with fewer than 1 / alpha shared returns (so that the tail holds less than one
    day on average) is NaN.

    For more information about the method, see the following paper:

    - Acharya, V.V., Pedersen, L.H., Philippon, T., & Richardson, M. (2017). "Measuring
    Systemic Risk." The Review of Financial Studies, 30(1), 2-47.

    Args:
        returns (pd.Series | pd.DataFrame): The daily returns of the asset(s).
        market_returns (pd.Series): The daily returns of the market (the system).
        alpha (float, optional): The tail probability of the market returns. Defaults to
        0.05, the paper's 5% worst days.
        period (str, optional): The period of each estimate. Options are: "weekly", "monthly", "quarterly", "yearly".
        Defaults to
        "yearly", as in the paper.
        rolling (int | None, optional): Use the days of the last `rolling` periods instead
        of a single period. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The MES with the periods as the index (and the assets as
        columns for a DataFrame).

    Raises:
        TypeError: If `returns` is not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_marginal_expected_shortfall(
                    returns[column], market_returns, alpha, period, rolling
                )
                for column in returns.columns
            }
        )
    if not isinstance(returns, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    minimum = int(np.ceil(1 / alpha))
    marginal_expected_shortfall = {}

    for end, asset, market in get_paired_windows(
        returns, market_returns, period, rolling
    ):
        if len(market) < minimum:
            marginal_expected_shortfall[end] = np.nan
            continue

        tail = market <= np.percentile(market, alpha * 100)
        marginal_expected_shortfall[end] = asset[tail].mean()

    return pd.Series(
        marginal_expected_shortfall, name=returns.name, dtype=float
    ).rename_axis(returns.index.name)


def get_market_beta(
    returns: pd.Series | pd.DataFrame,
    market_returns: pd.Series,
    period: str = "yearly",
    rolling: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the market beta per period from the daily returns within it: how strongly the
    asset moves with the market.

    It uses the dates the asset and the market both have a return (see
    `get_marginal_expected_shortfall` for the alignment):

    - Beta = Covariance(Returns, Market Returns) / Variance(Market Returns)

    A period with fewer than 15 shared returns is NaN.

    Shares similar logic:
    - empyrical, beta: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L1314

    Args:
        returns (pd.Series | pd.DataFrame): The daily returns of the asset(s).
        market_returns (pd.Series): The daily returns of the market.
        period (str, optional): The period of each estimate. Options are: "weekly", "monthly", "quarterly", "yearly".
        Defaults to
        "yearly".
        rolling (int | None, optional): Use the days of the last `rolling` periods instead
        of a single period. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame: The beta with the periods as the index (and the assets as
        columns for a DataFrame).

    Raises:
        TypeError: If `returns` is not a pd.Series or pd.DataFrame.
    """
    validate_period(period)

    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_market_beta(
                    returns[column], market_returns, period, rolling
                )
                for column in returns.columns
            }
        )
    if not isinstance(returns, pd.Series):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    beta = {}

    for end, asset, market in get_paired_windows(
        returns, market_returns, period, rolling
    ):
        market_variance = market.var(ddof=1) if len(market) > 1 else 0.0

        beta[end] = (
            np.cov(asset, market, ddof=1)[0, 1] / market_variance
            if len(market) >= MINIMUM_BETA_OBSERVATIONS and market_variance > 0
            else np.nan
        )

    return pd.Series(beta, name=returns.name, dtype=float).rename_axis(
        returns.index.name
    )


def get_long_run_marginal_expected_shortfall(
    beta: pd.Series | pd.DataFrame | float, decline: float = 0.4
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Long-Run Marginal Expected Shortfall (LRMES): the return of an asset's
    equity in a crisis, a fall of the market by `decline` over six months.

    The approximation of NYU Stern's V-Lab maps the market beta of the asset onto the
    crisis: an asset with a beta of 1 falls by as much as the market, one with a higher beta
    by more.

    - LRMES = (1 - decline) ^ Beta - 1 = exp(log(1 - decline) * Beta) - 1

    It is reported as a return, negative for a loss (V-Lab and Brownlees and Engle (2017)
    report the loss, 1 - exp(log(1 - decline) * Beta)).

    For more information about the method, see the following papers:

    - Brownlees, C., & Engle, R.F. (2017). "SRISK: A Conditional Capital Shortfall Measure
    of Systemic Risk." The Review of Financial Studies, 30(1), 48-79.
    - Acharya, V.V., Engle, R.F., & Richardson, M. (2012). "Capital Shortfall: A New
    Approach to Ranking and Regulating Systemic Risks." American Economic Review, 102(3),
    59-64.

    Args:
        beta (pd.Series | pd.DataFrame | float): The market beta of the asset(s).
        decline (float, optional): The fall of the market over six months that defines the
        crisis, as a decimal. Defaults to 0.4.

    Returns:
        pd.Series | pd.DataFrame | float: The LRMES as a return.
    """
    return np.exp(np.log(1 - decline) * beta) - 1


def get_srisk(
    total_liabilities: pd.Series | pd.DataFrame | float,
    market_cap: pd.Series | pd.DataFrame | float,
    long_run_marginal_expected_shortfall: pd.Series | pd.DataFrame | float,
    capital_ratio: float = 0.08,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate SRISK, the capital a firm is expected to need in a crisis: the prudential
    share `capital_ratio` of its assets it has to hold as equity, less the equity it has
    left after the crisis.

    - SRISK = k * Debt - (1 - k) * (1 + LRMES) * Equity

    Where k is the capital ratio, Debt the book value of the liabilities, Equity the market
    capitalization and LRMES the Long-Run Marginal Expected Shortfall as a return (negative
    for a loss, so 1 + LRMES is the share of the equity left), which is the formula of
    Brownlees and Engle (2017) with their LRMES as a loss written as 1 - LRMES. A negative
    SRISK is a capital surplus. SRISK is in the currency of `total_liabilities` and
    `market_cap`.

    For more information about the method, see the following papers:

    - Brownlees, C., & Engle, R.F. (2017). "SRISK: A Conditional Capital Shortfall Measure
    of Systemic Risk." The Review of Financial Studies, 30(1), 48-79.
    - Acharya, V.V., Engle, R.F., & Richardson, M. (2012). "Capital Shortfall: A New
    Approach to Ranking and Regulating Systemic Risks." American Economic Review, 102(3),
    59-64.

    Args:
        total_liabilities (pd.Series | pd.DataFrame | float): The book value of the
        liabilities.
        market_cap (pd.Series | pd.DataFrame | float): The market capitalization.
        long_run_marginal_expected_shortfall (pd.Series | pd.DataFrame | float): The LRMES as
        a return (see `get_long_run_marginal_expected_shortfall`).
        capital_ratio (float, optional): The prudential capital ratio k. Defaults to 0.08,
        V-Lab's ratio for U.S. firms.

    Returns:
        pd.Series | pd.DataFrame | float: The SRISK, positive for a capital shortfall.
    """
    return (
        capital_ratio * total_liabilities
        - (1 - capital_ratio) * (1 + long_run_marginal_expected_shortfall) * market_cap
    )


def get_srisk_share(srisk: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate each firm's share of the total SRISK of the firms: how much of the system's
    capital shortfall in a crisis it accounts for (SRISK% in Brownlees and Engle (2017)).

    A capital surplus does not offset the shortfall of another firm in a
    crisis, so the total only adds the positive SRISK and a firm with a surplus has a share
    of 0:

    - SRISK Share = MAX(SRISK, 0) / SUM(MAX(SRISK, 0) of every firm)

    A period without any shortfall (or without any SRISK) is NaN.

    Args:
        srisk (pd.DataFrame): The SRISK with the periods as the index and the firms as
        columns.

    Returns:
        pd.DataFrame: The share of each firm as a decimal, summing to 1 per period.
    """
    shortfall = srisk.clip(lower=0)
    total = shortfall.sum(axis=1, min_count=1)

    return shortfall.div(total.where(total > 0), axis=0)
