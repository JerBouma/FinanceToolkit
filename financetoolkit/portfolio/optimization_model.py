"""Portfolio Optimization Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import pdist

# pylint: disable=too-many-arguments,too-many-locals

# Tolerances for the numerical optimizers: tight enough that the weights are stable to the
# fourth decimal, which is the default rounding of the Finance Toolkit.
OPTIMIZER_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}

# Weights closer to zero than this are numerical noise of the optimizer.
ZERO_WEIGHT = 1e-10

# How far a solution may lie outside a constraint or bound and still be accepted.
FEASIBILITY_TOLERANCE = 1e-8

# SLSQP's "Positive directional derivative for linesearch": the line search found no step
# that improves the objective, which also happens at an optimum that is flat to machine
# precision.
SLSQP_LINE_SEARCH_STATUS = 8

# How closely the risk contributions of a risk parity portfolio must match the budget.
RISK_BUDGET_TOLERANCE = 1e-6


def _bounds(
    assets: int, minimum_weight: float, maximum_weight: float
) -> list[tuple[float, float]]:
    """
    The weight bounds per asset, checked for feasibility: the weights have to be able to
    sum to one within them.
    """
    if minimum_weight > maximum_weight:
        raise ValueError("minimum_weight cannot be larger than maximum_weight.")
    if assets * maximum_weight < 1 - 1e-9 or assets * minimum_weight > 1 + 1e-9:
        raise ValueError(
            f"Weights between {minimum_weight} and {maximum_weight} cannot sum to one "
            f"for {assets} assets."
        )
    return [(minimum_weight, maximum_weight)] * assets


def _fully_invested() -> dict:
    """The constraint that the weights sum to one."""
    return {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
        "jac": np.ones_like,
    }


def _target_return(mu: np.ndarray, target: float) -> dict:
    """The constraint that the expected return of the weights equals the target."""
    return {"type": "eq", "fun": lambda w: w @ mu - target, "jac": lambda w: mu}


def _is_feasible(
    weights: np.ndarray, bounds: list[tuple[float, float]], constraints: list[dict]
) -> bool:
    """Whether weights lie within the bounds and meet every constraint."""
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    if not np.all(np.isfinite(weights)):
        return False
    if np.any(weights < lower - FEASIBILITY_TOLERANCE) or np.any(
        weights > upper + FEASIBILITY_TOLERANCE
    ):
        return False
    for constraint in constraints:
        value = np.atleast_1d(constraint["fun"](weights))
        if constraint["type"] == "eq" and np.any(np.abs(value) > FEASIBILITY_TOLERANCE):
            return False
        if constraint["type"] == "ineq" and np.any(value < -FEASIBILITY_TOLERANCE):
            return False
    return True


def _solve(
    objective,
    gradient,
    assets: int,
    bounds: list[tuple[float, float]],
    constraints: list[dict],
) -> np.ndarray:
    """
    Run SLSQP from the equally weighted portfolio, clipped into the bounds. A line search
    that stops without a better step (status 8) is accepted when the weights meet every
    constraint and bound: SLSQP reports it at optima whose objective no longer changes at
    machine precision, e.g. for the maximum Sharpe ratio.
    """
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    start = np.clip(np.full(assets, 1 / assets), lower, upper)

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=OPTIMIZER_OPTIONS,
    )

    if not result.success and not (
        result.status == SLSQP_LINE_SEARCH_STATUS
        and _is_feasible(result.x, bounds, constraints)
    ):
        raise ValueError(f"The optimization did not converge: {result.message}")

    return result.x


def _return_range(
    mu: np.ndarray, bounds: list[tuple[float, float]]
) -> tuple[float, float, np.ndarray]:
    """
    The lowest and highest expected return a fully invested portfolio can have, and the
    weights of the portfolio with the highest return.
    """
    assets = len(mu)
    extremes = [
        linprog(
            sign * mu,
            A_eq=np.ones((1, assets)),
            b_eq=[1],
            bounds=bounds,
            method="highs",
        )
        for sign in (1, -1)
    ]
    return float(extremes[0].fun), float(-extremes[1].fun), extremes[1].x


def _as_series(weights: np.ndarray, assets: pd.Index) -> pd.Series:
    """Weights as a Series, with numerical noise around zero removed."""
    weights = np.where(np.abs(weights) < ZERO_WEIGHT, 0.0, weights)
    return pd.Series(weights, index=assets, name="Weight")


def get_expected_returns_and_covariance(
    returns: pd.DataFrame, periods_per_year: int
) -> tuple[pd.Series, pd.DataFrame]:
    """
    The annualized expected returns (the mean return per period times the number of
    periods in a year) and the annualized covariance matrix of a set of returns.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.

    Returns:
        tuple[pd.Series, pd.DataFrame]: The expected returns and the covariance matrix.
    """
    returns = returns.dropna(how="any")

    if len(returns) < 2:  # noqa: PLR2004
        raise ValueError("At least two periods of returns are needed for every asset.")

    return returns.mean() * periods_per_year, returns.cov() * periods_per_year


def get_portfolio_statistics(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    The expected return, volatility and Sharpe ratio of a portfolio.

    Args:
        weights (pd.Series): The weight per asset.
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        risk_free_rate (float, optional): The annualized risk free rate. Defaults to 0.

    Returns:
        pd.Series: The Expected Return, Volatility and Sharpe Ratio.
    """
    weights = weights.reindex(expected_returns.index).fillna(0)
    portfolio_return = float(weights @ expected_returns)
    portfolio_volatility = float(np.sqrt(weights @ covariance @ weights))

    return pd.Series(
        {
            "Expected Return": portfolio_return,
            "Volatility": portfolio_volatility,
            "Sharpe Ratio": (
                (portfolio_return - risk_free_rate) / portfolio_volatility
                if portfolio_volatility > 0
                else np.nan
            ),
        }
    )


def get_minimum_variance_weights(
    covariance: pd.DataFrame,
    minimum_weight: float = 0.0,
    maximum_weight: float = 1.0,
) -> pd.Series:
    """
    The fully invested portfolio with the lowest variance (Markowitz, 1952).

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        minimum_weight (float, optional): The lowest weight an asset may have; a negative
            value allows short positions. Defaults to 0.
        maximum_weight (float, optional): The highest weight an asset may have. Defaults to 1.

    Returns:
        pd.Series: The weight per asset.
    """
    sigma = covariance.to_numpy()
    assets = len(sigma)

    weights = _solve(
        lambda w: w @ sigma @ w,
        lambda w: 2 * sigma @ w,
        assets,
        _bounds(assets, minimum_weight, maximum_weight),
        [_fully_invested()],
    )

    return _as_series(weights, covariance.index)


def get_maximum_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    minimum_weight: float = 0.0,
    maximum_weight: float = 1.0,
) -> pd.Series:
    """
    The fully invested portfolio with the highest Sharpe ratio, the tangency portfolio of
    the efficient frontier (Sharpe, 1966; Tobin, 1958).

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        risk_free_rate (float, optional): The annualized risk free rate. Defaults to 0.
        minimum_weight (float, optional): The lowest weight an asset may have. Defaults to 0.
        maximum_weight (float, optional): The highest weight an asset may have. Defaults to 1.

    Returns:
        pd.Series: The weight per asset.
    """
    mu = expected_returns.to_numpy() - risk_free_rate
    sigma = covariance.to_numpy()
    assets = len(mu)

    if np.all(mu <= 0):
        raise ValueError(
            "No asset has an expected return above the risk free rate, so no portfolio "
            "has a positive Sharpe ratio."
        )

    def negative_sharpe(weights):
        return -(weights @ mu) / np.sqrt(weights @ sigma @ weights)

    def gradient(weights):
        variance = weights @ sigma @ weights
        volatility = np.sqrt(variance)
        excess = weights @ mu
        return -(mu * volatility - excess * (sigma @ weights) / volatility) / variance

    weights = _solve(
        negative_sharpe,
        gradient,
        assets,
        _bounds(assets, minimum_weight, maximum_weight),
        [_fully_invested()],
    )

    return _as_series(weights, covariance.index)


def get_mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float | None = None,
    risk_aversion: float = 2.5,
    minimum_weight: float = 0.0,
    maximum_weight: float = 1.0,
) -> pd.Series:
    """
    A mean-variance efficient portfolio (Markowitz, 1952). With a target return, the
    portfolio with the lowest variance that expects that return; otherwise the portfolio
    that maximizes the expected return minus half the risk aversion times the variance.

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        target_return (float | None, optional): The annualized return the portfolio must
            expect. Defaults to None, which uses the risk aversion instead.
        risk_aversion (float, optional): The trade-off between return and variance; higher
            values give less risky portfolios. Defaults to 2.5.
        minimum_weight (float, optional): The lowest weight an asset may have. Defaults to 0.
        maximum_weight (float, optional): The highest weight an asset may have. Defaults to 1.

    Returns:
        pd.Series: The weight per asset.
    """
    mu = expected_returns.to_numpy()
    sigma = covariance.to_numpy()
    assets = len(mu)
    bounds = _bounds(assets, minimum_weight, maximum_weight)

    if target_return is not None:
        lowest, highest, _ = _return_range(mu, bounds)
        if not lowest - 1e-9 <= target_return <= highest + 1e-9:
            raise ValueError(
                f"A target return of {target_return:.4f} cannot be reached within the "
                f"weight bounds ({lowest:.4f} to {highest:.4f})."
            )
        weights = _solve(
            lambda w: w @ sigma @ w,
            lambda w: 2 * sigma @ w,
            assets,
            bounds,
            [_fully_invested(), _target_return(mu, target_return)],
        )
    else:
        weights = _solve(
            lambda w: -(w @ mu) + risk_aversion / 2 * (w @ sigma @ w),
            lambda w: -mu + risk_aversion * (sigma @ w),
            assets,
            bounds,
            [_fully_invested()],
        )

    return _as_series(weights, covariance.index)


def get_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    points: int = 20,
    minimum_weight: float = 0.0,
    maximum_weight: float = 1.0,
) -> pd.DataFrame:
    """
    The efficient frontier: for evenly spaced target returns between the minimum variance
    portfolio and the highest return the constraints allow, the portfolio with the lowest
    variance (Markowitz, 1952).

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        risk_free_rate (float, optional): The annualized risk free rate, for the Sharpe
            ratio of each portfolio. Defaults to 0.
        points (int, optional): The number of portfolios on the frontier. Defaults to 20.
        minimum_weight (float, optional): The lowest weight an asset may have. Defaults to 0.
        maximum_weight (float, optional): The highest weight an asset may have. Defaults to 1.

    Returns:
        pd.DataFrame: Per portfolio its Expected Return, Volatility, Sharpe Ratio and the
        weight per asset.
    """
    if points < 2:  # noqa: PLR2004
        raise ValueError("The efficient frontier needs at least two points.")

    minimum_variance = get_minimum_variance_weights(
        covariance, minimum_weight, maximum_weight
    )
    lowest = float(minimum_variance @ expected_returns)

    mu = expected_returns.to_numpy()
    sigma = covariance.to_numpy()
    assets = len(mu)
    bounds = _bounds(assets, minimum_weight, maximum_weight)
    # The frontier runs up to the highest return the bounds allow.
    _, highest, highest_weights = _return_range(mu, bounds)
    targets = np.linspace(lowest, highest, points)

    rows = []
    for number, target in enumerate(targets, start=1):
        try:
            weights = _solve(
                lambda w: w @ sigma @ w,
                lambda w: 2 * sigma @ w,
                assets,
                bounds,
                [_fully_invested(), _target_return(mu, target)],
            )
        except ValueError:
            # The highest return is a corner of the weight bounds, where SLSQP can report
            # the constraints as incompatible although the portfolio exists: it is the one
            # the linear programme of the return range found. Other points still fail.
            if number != len(targets):
                raise
            weights = highest_weights
        weights = _as_series(weights, covariance.index)
        statistics = get_portfolio_statistics(
            weights, expected_returns, covariance, risk_free_rate
        )
        rows.append(pd.concat([statistics, weights]))

    frontier = pd.DataFrame(rows)
    frontier.index = pd.RangeIndex(1, len(frontier) + 1, name="Portfolio")

    return frontier


def get_inverse_volatility_weights(covariance: pd.DataFrame) -> pd.Series:
    """
    Weights proportional to one over each asset's volatility, a simple risk-based
    allocation that ignores correlations.

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.

    Returns:
        pd.Series: The weight per asset.
    """
    inverse_volatility = 1 / np.sqrt(np.diag(covariance.to_numpy()))

    return _as_series(inverse_volatility / inverse_volatility.sum(), covariance.index)


def get_risk_parity_weights(
    covariance: pd.DataFrame, risk_budget: pd.Series | None = None
) -> pd.Series:
    """
    The equal risk contribution (risk parity) portfolio: every asset contributes the same
    share of the portfolio's volatility, or the share given by a risk budget (Maillard,
    Roncalli and Teïletche, 2010). It is found by minimizing the convex function
    1/2 w'Σw - Σ b_i ln(w_i) and scaling the result to sum to one (Spinu, 2013).

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        risk_budget (pd.Series | None, optional): The share of the risk each asset should
            contribute; it is scaled to sum to one. Defaults to None, an equal share each.

    Returns:
        pd.Series: The weight per asset.
    """
    sigma = covariance.to_numpy()
    assets = len(sigma)

    budget = (
        np.full(assets, 1 / assets)
        if risk_budget is None
        else risk_budget.reindex(covariance.index).to_numpy(dtype=float)
    )
    if np.any(np.isnan(budget)) or np.any(budget <= 0):
        raise ValueError("Every asset needs a positive risk budget.")
    budget = budget / budget.sum()

    result = minimize(
        lambda y: 0.5 * y @ sigma @ y - budget @ np.log(y),
        np.full(assets, 1 / np.sqrt(np.trace(sigma))),
        jac=lambda y: sigma @ y - budget / y,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * assets,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000},
    )

    # The solution is judged by what it has to achieve, risk contributions in proportion
    # to the budget, rather than by the solver's status: L-BFGS-B stops with ABNORMAL when
    # its line search cannot improve an objective that is already flat to machine
    # precision, which happens at the optimum for some covariance matrices.
    weights = result.x / result.x.sum()
    contributions = weights * (sigma @ weights)
    if not np.all(np.isfinite(contributions)) or not np.allclose(
        contributions / contributions.sum(), budget, atol=RISK_BUDGET_TOLERANCE
    ):
        raise ValueError(f"The optimization did not converge: {result.message}")

    return _as_series(weights, covariance.index)


def get_hierarchical_risk_parity_weights(
    covariance: pd.DataFrame, linkage_method: str = "single"
) -> pd.Series:
    """
    Hierarchical Risk Parity (López de Prado, 2016): the assets are clustered on their
    correlations, ordered so that similar assets sit next to each other, and the weight is
    split top-down between the two halves of each cluster in inverse proportion to their
    variance. It needs no matrix inversion, so it stays stable when assets are highly
    correlated.

    As in the paper (Stage 1, step 2), the clustering uses the Euclidean distance between
    the columns of the correlation distance matrix d = √((1 - ρ) / 2), a distance of
    distances that compares how each pair of assets relates to all the others. Some
    libraries cluster on d itself, which can order the assets differently.

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering, e.g.
            "single", "complete", "average" or "ward". Defaults to "single".

    Returns:
        pd.Series: The weight per asset.
    """
    volatility = np.sqrt(np.diag(covariance.to_numpy()))
    correlation = covariance / np.outer(volatility, volatility)

    # The correlation distance of López de Prado, sqrt((1 - rho) / 2), and the Euclidean
    # distance between its columns that the paper clusters on (its code passes the square
    # matrix to linkage, which reads the rows as observations and does exactly this).
    distance = np.sqrt(np.clip((1 - correlation.to_numpy()) / 2, 0, None))
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(pdist(distance), method=linkage_method))
    ordered = covariance.index[order].tolist()

    def cluster_variance(items: list) -> float:
        sub = covariance.loc[items, items].to_numpy()
        inverse_variance = 1 / np.diag(sub)
        weights = inverse_variance / inverse_variance.sum()
        return float(weights @ sub @ weights)

    weights = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        clusters = [
            half
            for cluster in clusters
            if len(cluster) > 1
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            left_variance, right_variance = cluster_variance(left), cluster_variance(
                right
            )
            alpha = 1 - left_variance / (left_variance + right_variance)
            weights[left] *= alpha
            weights[right] *= 1 - alpha

    return _as_series(weights.reindex(covariance.index).to_numpy(), covariance.index)


def get_minimum_cvar_weights(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
    minimum_weight: float = 0.0,
    maximum_weight: float = 1.0,
) -> pd.Series:
    """
    The fully invested portfolio with the lowest Conditional Value at Risk (expected
    shortfall) over the historical returns, solved as a linear programme (Rockafellar and
    Uryasev, 2000).

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        confidence_level (float, optional): The confidence level of the CVaR, e.g. 0.95 for
            the average of the worst 5% of periods. Defaults to 0.95.
        minimum_weight (float, optional): The lowest weight an asset may have. Defaults to 0.
        maximum_weight (float, optional): The highest weight an asset may have. Defaults to 1.

    Returns:
        pd.Series: The weight per asset.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")

    data = returns.dropna(how="any").to_numpy()
    periods, assets = data.shape
    bounds = _bounds(assets, minimum_weight, maximum_weight)

    # Variables: the weights (assets), the Value at Risk (1) and the shortfall per period.
    cost = np.concatenate(
        [
            np.zeros(assets),
            [1.0],
            np.full(periods, 1 / ((1 - confidence_level) * periods)),
        ]
    )
    # Shortfall_t >= -r_t'w - VaR, written as -r_t'w - VaR - shortfall_t <= 0.
    inequality = np.hstack([-data, -np.ones((periods, 1)), -np.eye(periods)])
    equality = np.concatenate([np.ones(assets), [0.0], np.zeros(periods)])[None, :]

    result = linprog(
        cost,
        A_ub=inequality,
        b_ub=np.zeros(periods),
        A_eq=equality,
        b_eq=[1.0],
        bounds=bounds + [(None, None)] + [(0, None)] * periods,
        method="highs",
    )

    if not result.success:
        raise ValueError(f"The optimization did not converge: {result.message}")

    return _as_series(result.x[:assets], returns.columns)


def get_black_litterman_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | None = None,
    view_confidences: dict[str, float] | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    The Black-Litterman expected returns (Black and Litterman, 1992), as total returns.
    The prior is the return the market implies through its weights: the excess return
    Π = δΣw at which holding the market is optimal (He and Litterman, 1999), plus the risk
    free rate. Views move it towards the investor's expectations in proportion to their
    confidence. A view is absolute ("AAPL": 0.10, AAPL returns 10% a year in total) or
    relative ("AAPL-MSFT": 0.02, AAPL beats MSFT by 2%, which the risk free rate does not
    affect). Without a confidence, a view's uncertainty is the variance of the view
    portfolio scaled by tau, Ω_k = τ p_k'Σp_k (He and Litterman, 1999). A confidence c
    between 0 and 1 scales it to Ω_k = (1 - c) / c × τ p_k'Σp_k, the closed form of
    Idzorek's (2005) confidence method derived by Walters (2014): a confidence of 0.5 keeps
    the uncertainty and one close to 1 moves the return all the way to the view.

    References:
        - Black, F. and Litterman, R. (1992). "Global Portfolio Optimization." Financial
          Analysts Journal, 48(5), 28-43.
        - He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman Model
          Portfolios." Goldman Sachs Investment Management Research.
        - Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model:
          Incorporating User-Specified Confidence Levels." Working paper.
        - Walters, J. (2014). "The Black-Litterman Model in Detail."
          https://ssrn.com/abstract=1314585

    Args:
        covariance (pd.DataFrame): The annualized covariance matrix.
        market_weights (pd.Series): The market capitalization weight per asset.
        views (dict[str, float] | None, optional): The annualized views, absolute views as
            total returns, as described above. Defaults to None, which returns the market
            implied returns.
        view_confidences (dict[str, float] | None, optional): The confidence per view,
            between 0 and 1. Defaults to None.
        risk_aversion (float, optional): The risk aversion of the market, δ. Defaults to 2.5.
        tau (float, optional): The uncertainty of the prior relative to the covariance.
            Defaults to 0.05.
        risk_free_rate (float, optional): The annualized risk free rate, added to the
            implied excess returns so that they are total returns like the views. Defaults
            to 0.

    Returns:
        pd.DataFrame: The Implied Return and the Black-Litterman Return per asset, both
        annualized total returns.
    """
    assets = covariance.index
    sigma = covariance.to_numpy()
    weights = market_weights.reindex(assets).fillna(0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        raise ValueError("The market weights must sum to a positive number.")
    weights = weights / weights.sum()

    # Π = δΣw is an excess return; with the risk free rate added it is a total return,
    # comparable to absolute views. A relative view compares two total returns, so the
    # risk free rate cancels out of it.
    implied = risk_aversion * sigma @ weights + risk_free_rate
    posterior = implied

    if views:
        pick, targets, uncertainty = [], [], []
        for view, value in views.items():
            row = np.zeros(len(assets))
            if view in assets:
                row[assets.get_loc(view)] = 1
            elif "-" in view and all(part in assets for part in view.split("-", 1)):
                long_leg, short_leg = view.split("-", 1)
                row[assets.get_loc(long_leg)] = 1
                row[assets.get_loc(short_leg)] = -1
            else:
                raise ValueError(
                    f"The view {view!r} names no asset; use 'TICKER' or 'TICKER-TICKER'."
                )
            variance = tau * row @ sigma @ row
            confidence = (view_confidences or {}).get(view)
            if confidence is not None:
                if not 0 < confidence <= 1:
                    raise ValueError("A view confidence must lie between 0 and 1.")
                variance *= max(1 - confidence, 1e-8) / confidence
            pick.append(row)
            targets.append(value)
            uncertainty.append(variance)

        p = np.array(pick)
        q = np.array(targets)
        omega = np.diag(uncertainty)
        tau_sigma_inverse = np.linalg.inv(tau * sigma)
        posterior = np.linalg.solve(
            tau_sigma_inverse + p.T @ np.linalg.inv(omega) @ p,
            tau_sigma_inverse @ implied + p.T @ np.linalg.inv(omega) @ q,
        )

    return pd.DataFrame(
        {"Implied Return": implied, "Black-Litterman Return": posterior}, index=assets
    )


def get_risk_contributions(
    weights: pd.Series, covariance: pd.DataFrame
) -> pd.DataFrame:
    """
    How much each asset adds to the volatility of a portfolio (Euler allocation): the
    marginal contribution ∂σ/∂w_i = (Σw)_i / σ, the risk contribution w_i times that, and
    its share of the total, which sums to one.

    Args:
        weights (pd.Series): The weight per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.

    Returns:
        pd.DataFrame: The Weight, Marginal Contribution, Risk Contribution and Percentage
        Contribution per asset.
    """
    weights = weights.reindex(covariance.index).fillna(0)
    sigma = covariance.to_numpy()
    volatility = float(np.sqrt(weights @ sigma @ weights))
    if volatility == 0:
        raise ValueError("The portfolio has no volatility.")

    marginal = sigma @ weights.to_numpy() / volatility
    contribution = weights.to_numpy() * marginal

    return pd.DataFrame(
        {
            "Weight": weights.to_numpy(),
            "Marginal Contribution": marginal,
            "Risk Contribution": contribution,
            "Percentage Contribution": contribution / volatility,
        },
        index=covariance.index,
    )
