"""Portfolio Optimization Model"""

__docformat__ = "google"

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse
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

# The volatility below which a portfolio or asset counts as riskless: rounding leaves a
# constant return series with a volatility of about 1e-17 instead of 0, which would give
# a Sharpe ratio of 1e17.
ZERO_VOLATILITY_TOLERANCE = 1e-12


def _validate_covariance(covariance: pd.DataFrame, positive: bool = True) -> None:
    """
    Raises a clear error for a covariance matrix the optimizers cannot use: one with missing
    or infinite values (e.g. an asset without returns) or, when positive, an asset without
    any variance (a constant return), which a risk-based weighting divides by.
    """
    values = covariance.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        missing = covariance.index[~np.isfinite(values).all(axis=1)]
        raise ValueError(
            f"The covariance of {', '.join(map(str, missing))} has missing values; these "
            "assets have no returns over the range."
        )
    if positive:
        riskless = covariance.index[np.diag(values) <= ZERO_VOLATILITY_TOLERANCE**2]
        if len(riskless):
            raise ValueError(
                f"{', '.join(map(str, riskless))} has no variance (a constant return), "
                "which this weighting cannot handle."
            )


class WeightConstraints(NamedTuple):
    """
    The constraints on the weights as a linear system over the variables x: the weights
    w, followed by one variable t_i ≥ |w_i| per asset when the gross exposure is limited
    and short positions are allowed, and by one variable d_i ≥ |w_i - w0_i| per asset
    when there are current weights w0 to trade from. Every optimizer solves over x and
    keeps its first `assets` entries.
    """

    assets: int
    bounds: list[tuple[float, float]]
    inequality: np.ndarray
    upper: np.ndarray
    equality: np.ndarray
    target: np.ndarray
    current: np.ndarray | None = None


def _per_asset(
    bound: float | dict[str, float], assets: pd.Index, default: float, name: str
) -> np.ndarray:
    """A bound per asset from one number or from {asset: bound} with a default."""
    if not isinstance(bound, dict):
        return np.full(len(assets), float(bound))
    unknown = [asset for asset in bound if asset not in assets]
    if unknown:
        raise ValueError(
            f"{name} names assets that are not in the portfolio: {', '.join(map(str, unknown))}."
        )
    return (
        pd.Series(bound, dtype=float)
        .reindex(assets)
        .fillna(default)
        .to_numpy(dtype=float)
    )


def get_weight_constraints(
    assets: pd.Index,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
) -> WeightConstraints:
    """
    The constraints on the weights of a portfolio as a linear system, checked for
    feasibility so that an impossible combination fails with a clear message before any
    optimization runs.

    - Per asset, minimum_weight ≤ w_i ≤ maximum_weight; a bound given as {asset: bound}
      applies to the assets it names, the others keep 0 (minimum) or 1 (maximum).
    - Per group (e.g. a sector or country), the weights of its assets sum to between the
      group's minimum and maximum, Σ_{i ∈ g} w_i, as in PyPortfolioOpt's
      add_sector_constraints and skfolio's groups with linear constraints.
    - The net exposure Σ w_i equals net_exposure (1 for a fully invested portfolio, 0 for
      a market neutral one).
    - The gross exposure Σ |w_i| is at most gross_exposure, e.g. 1.6 for a 130/30
      portfolio. |w_i| is linearized with one variable t_i per asset: t_i ≥ w_i,
      t_i ≥ -w_i and Σ t_i ≤ gross_exposure, which is exact because a smaller Σ t_i is
      always allowed (the same polyhedron as the split w = w⁺ - w⁻ with Σ (w⁺ + w⁻) ≤ L).
      Without short positions Σ |w_i| equals the net exposure, so no variables are added.
    - The turnover Σ |w_i - w0_i| from the current weights w0 is at most
      maximum_turnover, linearized the same way with one variable d_i per asset:
      d_i ≥ w_i - w0_i, d_i ≥ w0_i - w_i and Σ d_i ≤ maximum_turnover (Lobo, Fazel and
      Boyd, 2007). The variables are added whenever current weights are given, so that
      an optimizer can also charge a transaction cost on them.

    Args:
        assets (pd.Index): The assets of the portfolio.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset. A negative value allows a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset. Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector" and "Country"); an asset without a group is in none. Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification, the minimum and maximum total weight per group, e.g.
            {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights.
            Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from; an asset it leaves out is not held. Defaults to
            None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|,
            the sum of the weights bought and sold, e.g. 0.2 to trade at most 10% of the
            portfolio each way. Needs current weights. Defaults to None, no limit.

    Returns:
        WeightConstraints: The bounds and the linear inequality (A x ≤ b) and equality
            (A x = b) constraints over the weights and, with a gross exposure limit and
            short positions, the absolute weights, and with current weights, the
            absolute trades.
    """
    count = len(assets)
    lower = _per_asset(minimum_weight, assets, 0.0, "minimum_weight")
    upper = _per_asset(maximum_weight, assets, 1.0, "maximum_weight")

    crossed = assets[lower > upper]
    if len(crossed):
        raise ValueError(
            f"The minimum weight is above the maximum weight for {', '.join(map(str, crossed))}."
        )
    if lower.sum() > net_exposure + 1e-9 or upper.sum() < net_exposure - 1e-9:
        bounds_text = (
            f"between {minimum_weight} and {maximum_weight}"
            if not isinstance(minimum_weight, dict)
            and not isinstance(maximum_weight, dict)
            else f"between {lower.sum():.4f} and {upper.sum():.4f} in total"
        )
        raise ValueError(
            f"Weights {bounds_text} cannot sum to "
            f"{'one' if net_exposure == 1 else net_exposure} for {count} assets."
        )

    rows, limits = [], []
    for classification, bounds_per_group in (group_bounds or {}).items():
        if groups is None or classification not in groups.columns:
            raise ValueError(f"The assets have no {classification} to group them by.")
        membership = groups[classification].reindex(assets)
        known = sorted(membership.dropna().unique().astype(str))
        for group, group_bound in bounds_per_group.items():
            if (
                not isinstance(group_bound, list | tuple)
                or len(group_bound) != 2  # noqa: PLR2004
            ):
                raise ValueError(
                    f"The bounds of {classification} {group!r} must be a (minimum, "
                    f"maximum) pair, not {group_bound!r}."
                )
            low, high = (float(value) for value in group_bound)
            if group not in known:
                raise ValueError(
                    f"No asset has the {classification} {group!r}; the assets have "
                    f"{', '.join(known) or 'none'}."
                )
            if low > high:
                raise ValueError(
                    f"The minimum of {classification} {group!r} is above its maximum."
                )
            member = (membership == group).to_numpy(dtype=float)
            rows.extend([member, -member])
            limits.extend([high, -low])

    bounds = list(zip(lower, upper, strict=True))
    inequality = np.array(rows).reshape(-1, count)
    limit = np.array(limits, dtype=float)
    equality = np.ones((1, count))

    if gross_exposure is not None:
        if gross_exposure < abs(net_exposure) - 1e-9:
            raise ValueError(
                f"A gross exposure of {gross_exposure} cannot hold a net exposure of "
                f"{net_exposure}."
            )
        if np.any(lower < 0):
            identity = np.eye(count)
            inequality = np.vstack(
                [
                    np.hstack([inequality, np.zeros((len(inequality), count))]),
                    np.hstack([identity, -identity]),
                    np.hstack([-identity, -identity]),
                    np.concatenate([np.zeros(count), np.ones(count)])[None, :],
                ]
            )
            limit = np.concatenate([limit, np.zeros(2 * count), [gross_exposure]])
            equality = np.hstack([equality, np.zeros((1, count))])
            bounds += [(0.0, max(abs(low), abs(high))) for low, high in bounds]

    current = None
    if maximum_turnover is not None and current_weights is None:
        raise ValueError("A maximum turnover needs the current weights to trade from.")
    if current_weights is not None:
        current = _per_asset(current_weights.to_dict(), assets, 0.0, "current_weights")
        if maximum_turnover is not None and maximum_turnover < 0:
            raise ValueError("The maximum turnover cannot be negative.")
        # d_i ≥ |w_i - w0_i| as w_i - d_i ≤ w0_i and -w_i - d_i ≤ -w0_i, the trades
        # after any absolute weights of a gross exposure limit.
        variables = len(bounds)
        identity = np.eye(count)
        padding = np.zeros((count, variables - count))
        rows = [
            np.hstack([inequality, np.zeros((len(inequality), count))]),
            np.hstack([identity, padding, -identity]),
            np.hstack([-identity, padding, -identity]),
        ]
        limits = [limit, current, -current]
        if maximum_turnover is not None:
            rows.append(np.concatenate([np.zeros(variables), np.ones(count)])[None, :])
            limits.append([maximum_turnover])
        inequality = np.vstack(rows)
        limit = np.concatenate(limits)
        equality = np.hstack([equality, np.zeros((1, count))])
        bounds += [
            (0.0, max(abs(low - held), abs(high - held)))
            for (low, high), held in zip(bounds[:count], current, strict=True)
        ]

    constraints = WeightConstraints(
        count, bounds, inequality, limit, equality, np.array([net_exposure]), current
    )

    # The individual checks above name the usual mistakes; this finds the combinations of
    # bounds that cannot hold together, e.g. a sector minimum above its assets' maxima.
    if (
        len(inequality)
        and not _linear_programme(np.zeros(len(bounds)), constraints).success
    ):
        raise ValueError(
            "No weights meet all the constraints together: the weight bounds, the group "
            "bounds, the net and gross exposure and the maximum turnover contradict each "
            "other."
        )

    return constraints


def _linear_programme(cost: np.ndarray, constraints: WeightConstraints):
    """A linear programme over the variables of the constraints, solved with HiGHS."""
    return linprog(
        cost,
        A_ub=constraints.inequality if len(constraints.inequality) else None,
        b_ub=constraints.upper if len(constraints.inequality) else None,
        A_eq=constraints.equality,
        b_eq=constraints.target,
        bounds=constraints.bounds,
        method="highs",
    )


def _linear_constraints(constraints: WeightConstraints) -> list[dict]:
    """The linear constraints in the form SLSQP takes."""
    linear = [
        {
            "type": "eq",
            "fun": lambda x: constraints.equality @ x - constraints.target,
            "jac": lambda x: constraints.equality,
        }
    ]
    if len(constraints.inequality):
        linear.append(
            {
                "type": "ineq",
                "fun": lambda x: constraints.upper - constraints.inequality @ x,
                "jac": lambda x: -constraints.inequality,
            }
        )
    return linear


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
    constraints: WeightConstraints,
    extra: list[dict] | None = None,
    all_variables: bool = False,
) -> np.ndarray:
    """
    Run SLSQP from the portfolio with equal weights summing to the net exposure, clipped
    into the bounds. The objective, its gradient and the extra constraints take the
    weights; the absolute weight and trade variables of a gross exposure or turnover
    limit only enter the linear constraints, unless all_variables is set, when the
    objective and its gradient take all of them (to charge a cost on the trades). A line
    search that stops without a better step (status 8) is accepted when the weights meet
    every constraint and bound: SLSQP reports it at optima whose objective no longer
    changes at machine precision, e.g. for the maximum Sharpe ratio.
    """
    assets = constraints.assets
    lower = np.array([bound[0] for bound in constraints.bounds])
    upper = np.array([bound[1] for bound in constraints.bounds])
    weights = np.clip(
        np.full(assets, constraints.target[0] / assets), lower[:assets], upper[:assets]
    )
    # The absolute weights of a gross exposure limit, then the trades from the current
    # weights, each at the value the start weights give them.
    trades = (
        [np.abs(weights - constraints.current)]
        if constraints.current is not None
        else []
    )
    absolute = [np.abs(weights)] if len(lower) - assets * (1 + len(trades)) > 0 else []
    start = np.clip(np.concatenate([weights, *absolute, *trades]), lower, upper)

    padding = np.zeros(len(lower) - assets)
    slsqp_constraints = _linear_constraints(constraints) + [
        {
            "type": constraint["type"],
            "fun": lambda x, function=constraint["fun"]: function(x[:assets]),
            "jac": lambda x, jacobian=constraint["jac"]: np.concatenate(
                [jacobian(x[:assets]), padding]
            ),
        }
        for constraint in extra or []
    ]

    result = minimize(
        objective if all_variables else lambda x: objective(x[:assets]),
        start,
        jac=(
            gradient
            if all_variables
            else lambda x: np.concatenate([gradient(x[:assets]), padding])
        ),
        method="SLSQP",
        bounds=constraints.bounds,
        constraints=slsqp_constraints,
        options=OPTIMIZER_OPTIONS,
    )

    if not result.success and not (
        result.status == SLSQP_LINE_SEARCH_STATUS
        and _is_feasible(result.x, constraints.bounds, slsqp_constraints)
    ):
        raise ValueError(f"The optimization did not converge: {result.message}")

    return result.x[:assets]


def _trading_costs(
    constraints: WeightConstraints, transaction_cost: float
) -> np.ndarray:
    """
    The cost of each variable of the constraints: the transaction cost on the trades d_i,
    the last `assets` variables, and nothing on the others.
    """
    if transaction_cost < 0:
        raise ValueError("The transaction cost cannot be negative.")
    costs = np.zeros(len(constraints.bounds))
    if transaction_cost:
        if constraints.current is None:
            raise ValueError(
                "A transaction cost needs the current weights to trade from."
            )
        costs[-constraints.assets :] = transaction_cost
    return costs


def _return_range(
    mu: np.ndarray, constraints: WeightConstraints
) -> tuple[float, float, np.ndarray]:
    """
    The lowest and highest expected return a portfolio within the constraints can have,
    and the weights of the portfolio with the highest return.
    """
    padding = np.zeros(len(constraints.bounds) - constraints.assets)
    extremes = [
        _linear_programme(np.concatenate([sign * mu, padding]), constraints)
        for sign in (1, -1)
    ]
    return (
        float(extremes[0].fun),
        float(-extremes[1].fun),
        extremes[1].x[: constraints.assets],
    )


def _as_series(weights: np.ndarray, assets: pd.Index) -> pd.Series:
    """Weights as a Series, with numerical noise around zero removed."""
    weights = np.where(np.abs(weights) < ZERO_WEIGHT, 0.0, weights)
    return pd.Series(weights, index=assets, name="Weight")


def _shrink_to_identity(
    sample: np.ndarray, shrinkage: float, assets: pd.Index
) -> pd.DataFrame:
    """The sample covariance shrunk towards the identity scaled by its average variance."""
    target = np.trace(sample) / len(sample) * np.eye(len(sample))
    return pd.DataFrame(
        (1 - shrinkage) * sample + shrinkage * target, index=assets, columns=assets
    )


def get_ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    The Ledoit-Wolf shrinkage estimator of the covariance matrix: the sample covariance
    pulled towards the identity matrix scaled by the average variance, with the shrinkage
    intensity that minimizes the expected squared (Frobenius) error. This keeps the matrix
    well-conditioned when there are many assets compared to the number of periods, which
    is where the sample covariance makes optimizers chase estimation noise.

    The formula and the maximum likelihood sample covariance it shrinks (divided by the
    number of periods, not one less) follow scikit-learn's LedoitWolf, so the results
    agree with it.

    See Ledoit, O. and Wolf, M. (2004), "A well-conditioned estimator for large-dimensional
    covariance matrices", Journal of Multivariate Analysis, 88(2), 365-411.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.DataFrame, float]: The covariance matrix per period and the shrinkage
            intensity between 0 (the sample covariance) and 1 (the scaled identity).
    """
    values = returns.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    periods, assets = centered.shape
    sample = centered.T @ centered / periods

    squared = centered**2
    average_variance = squared.sum() / periods / assets
    # beta is the estimated variance of the sample covariance's entries, delta the squared
    # distance between the sample covariance and the target; their ratio is the intensity.
    beta = (
        np.sum(squared.T @ squared) / periods
        - np.sum((centered.T @ centered) ** 2) / periods**2
    ) / (assets * periods)
    delta = (
        np.sum((centered.T @ centered) ** 2) / periods**2
        - 2 * average_variance * squared.sum() / periods
        + assets * average_variance**2
    ) / assets
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta

    return _shrink_to_identity(sample, shrinkage, returns.columns), shrinkage


def get_oracle_approximating_shrinkage_covariance(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    The Oracle Approximating Shrinkage (OAS) estimator of the covariance matrix: like
    Ledoit-Wolf, the sample covariance shrunk towards the identity matrix scaled by the
    average variance, but with an intensity derived for normally distributed returns,
    which converges faster and shrinks more when there are few periods.

    The formula and the maximum likelihood sample covariance it shrinks follow
    scikit-learn's OAS, so the results agree with it.

    See Chen, Y., Wiesel, A., Eldar, Y. C. and Hero, A. O. (2010), "Shrinkage Algorithms
    for MMSE Covariance Estimation", IEEE Transactions on Signal Processing, 58(10),
    5016-5029.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.DataFrame, float]: The covariance matrix per period and the shrinkage
            intensity between 0 (the sample covariance) and 1 (the scaled identity).
    """
    values = returns.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    periods, assets = centered.shape
    sample = centered.T @ centered / periods

    mean_squared_entry = np.mean(sample**2)
    squared_average_variance = (np.trace(sample) / assets) ** 2
    numerator = mean_squared_entry + squared_average_variance
    denominator = (periods + 1) * (
        mean_squared_entry - squared_average_variance / assets
    )
    shrinkage = 1.0 if denominator == 0 else min(numerator / denominator, 1.0)

    return _shrink_to_identity(sample, shrinkage, returns.columns), shrinkage


def get_semicovariance(returns: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    The semicovariance matrix: the co-movement of the returns below a threshold, each
    return above it counting as zero. It measures only the downside risk, which is what
    an investor who doesn't mind upside surprises wants to minimize.

    Every period counts in the average, the ones above the threshold as zero, as in
    Estrada and in PyPortfolioOpt's semicovariance.

    See Estrada, J. (2007), "Mean-Semivariance Optimization: A Heuristic Approach",
    Journal of Applied Finance, 18(1), 57-72.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        threshold (float, optional): The return per period below which a return counts
            as downside. Defaults to 0.

    Returns:
        pd.DataFrame: The semicovariance matrix per period.
    """
    downside = (returns - threshold).clip(upper=0)
    return downside.T @ downside / len(downside)


def get_exponential_covariance(returns: pd.DataFrame, span: int = 180) -> pd.DataFrame:
    """
    The exponentially weighted covariance matrix: the products of the deviations from
    the mean return weighted by (1 − α)^age with α = 2 / (span + 1), so that recent
    periods count most and the matrix follows changes in volatility and correlation.

    The deviations are taken from the mean over the whole range and the weights are
    normalized to sum to one, as in PyPortfolioOpt's exp_cov.

    See RiskMetrics Technical Document (J.P. Morgan/Reuters, 1996), chapter 5.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        span (int, optional): The span of the weights in periods; the weight halves
            after about span / 2.9 periods. Defaults to 180.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    if span < 1:
        raise ValueError("The span of the exponential covariance must be at least 1.")
    deviations = (returns - returns.mean()).to_numpy(dtype=float)
    decay = 1 - 2 / (span + 1)
    weights = decay ** np.arange(len(deviations) - 1, -1, -1)
    weights /= weights.sum()
    covariance = (deviations * weights[:, None]).T @ deviations
    return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)


def get_denoised_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    The denoised covariance matrix: the sample covariance with the eigenvalues of its
    correlation matrix that random returns could produce replaced by their average. Under
    random matrix theory, the eigenvalues of the correlation matrix of T periods of N
    unrelated returns fall below the Marčenko-Pastur upper edge λ+ = σ²(1 + √(N/T))², so
    the eigenvalues below it carry noise rather than structure. Averaging them keeps the
    trace (the total variance) and the eigenvectors of the signal, and removes the small
    eigenvalues that make the inverse of the matrix, and with it optimized weights, chase
    estimation error.

    The variance σ² of the noise is the one whose Marčenko-Pastur density fits the
    eigenvalues best: the sum of squared differences with a Gaussian kernel density of
    the eigenvalues (bandwidth 0.01), at 1,000 points between the edges of the
    Marčenko-Pastur distribution, minimized for σ² between 0.00001 and 0.99999 starting
    at 0.5. This is López de Prado's "constant residual eigenvalue" method, as in
    skfolio's DenoiseCovariance, so the results agree with it. The correlation matrix is
    rescaled to a unit diagonal after the replacement and turned back into a covariance
    with the sample standard deviations (ddof=1).

    See Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (2000), "Random Matrix
    Approach to Cross Correlations in Financial Data", Physical Review E, 63(3), 037101,
    and López de Prado, M. (2020), "Machine Learning for Asset Managers", Cambridge
    University Press, chapter 2.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.DataFrame, int]: The covariance matrix per period and the number of
            eigenvalues kept as signal (those above the Marčenko-Pastur upper edge).
    """
    sample = returns.cov().to_numpy(dtype=float)
    periods, assets = returns.shape
    ratio = periods / assets
    volatility = np.sqrt(np.diag(sample))
    correlation = sample / np.outer(volatility, volatility)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    def density_error(variance: np.ndarray) -> float:
        # The Marčenko-Pastur density at 1,000 points between its edges against a Gaussian
        # kernel density of the eigenvalues at the same points.
        lower = variance[0] * (1 - (1 / ratio) ** 0.5) ** 2
        upper = variance[0] * (1 + (1 / ratio) ** 0.5) ** 2
        points = np.linspace(lower, upper, 1000)
        theoretical = (
            ratio
            / (2 * np.pi * variance[0] * points)
            * np.sqrt(np.clip((upper - points) * (points - lower), 0, None))
        )
        bandwidth = 0.01
        empirical = np.exp(
            -0.5 * ((points[:, None] - eigenvalues[None, :]) / bandwidth) ** 2
        ).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
        return float(np.sum((empirical - theoretical) ** 2))

    variance = minimize(density_error, x0=0.5, bounds=((1e-5, 1 - 1e-5),)).x[0]
    upper_edge = variance * (1 + (1 / ratio) ** 0.5) ** 2
    signals = int(assets - np.searchsorted(eigenvalues[::-1], upper_edge))

    if signals < assets:
        eigenvalues = eigenvalues.copy()
        eigenvalues[signals:] = eigenvalues[signals:].mean()
        correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        scale = np.sqrt(np.diag(correlation))
        correlation = correlation / np.outer(scale, scale)

    return (
        pd.DataFrame(
            correlation * np.outer(volatility, volatility),
            index=returns.columns,
            columns=returns.columns,
        ),
        signals,
    )


COVARIANCE_METHODS = (
    "sample",
    "ledoit_wolf",
    "oas",
    "semicovariance",
    "exponential",
    "denoised",
)

EXPECTED_RETURN_METHODS = ("mean", "exponential", "capm", "james_stein")


def get_expected_returns(
    returns: pd.DataFrame,
    periods_per_year: int,
    method: str = "mean",
    span: int = 180,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    The annualized expected return of each asset, estimated from its returns per period.

    - "mean": the average return per period times the number of periods in a year, the
      arithmetic historical mean (PyPortfolioOpt's mean_historical_return with
      compounding=False).
    - "exponential": the exponentially weighted average return, with the weight of a
      period (1 - α)^age, α = 2 / (span + 1), normalized to sum to one, so recent
      periods count most (PyPortfolioOpt's ema_historical_return with
      compounding=False, pandas' ewm with adjust=True).
    - "capm": the return the Capital Asset Pricing Model implies, r_f + β_i (E[r_m] -
      r_f), with β_i = Cov(r_i, r_m) / Var(r_m) of the returns per period and E[r_m]
      the annualized average return of the benchmark (Sharpe, 1964; PyPortfolioOpt's
      capm_return with compounding=False).
    - "james_stein": the Bayes-Stein shrinkage estimator of Jorion (1986), which pulls
      every average return towards the average return of the minimum variance portfolio,
      μ_0 = 1'Σ⁻¹μ̂ / 1'Σ⁻¹1, by φ = (N + 2) / ((N + 2) + T (μ̂ - μ_0 1)'Σ⁻¹(μ̂ - μ_0 1)),
      with μ̂ the average of the T returns per period of the N assets and Σ = S (T - 1) /
      (T - N - 2) Jorion's estimate of the covariance matrix from the sample covariance
      S (ddof=1): μ = (1 - φ) μ̂ + φ μ_0 1. Riskfolio-Lib's mean_vector with method="BS"
      uses S itself, which shrinks slightly less. The average returns are the noisiest input of a mean-variance
      optimization, and the shrinkage is strongest when they differ little compared to
      their estimation error.

    See Sharpe, W. F. (1964), "Capital Asset Prices: A Theory of Market Equilibrium under
    Conditions of Risk", The Journal of Finance, 19(3), 425-442, and Jorion, P. (1986),
    "Bayes-Stein Estimation for Portfolio Analysis", Journal of Financial and
    Quantitative Analysis, 21(3), 279-292.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.
        method (str, optional): "mean", "exponential", "capm" or "james_stein". Defaults
            to "mean".
        span (int, optional): The span in periods of the "exponential" average. Defaults
            to 180.
        benchmark_returns (pd.Series | None, optional): The returns per period of the
            market, for "capm". Defaults to None.
        risk_free_rate (float, optional): The annualized risk free rate, for "capm".
            Defaults to 0.

    Returns:
        pd.Series: The annualized expected return per asset.
    """
    if method == "mean":
        mean = returns.mean()
    elif method == "exponential":
        if span < 1:
            raise ValueError("The span of the exponential average must be at least 1.")
        mean = returns.ewm(span=span).mean().iloc[-1]
    elif method == "capm":
        if benchmark_returns is None:
            raise ValueError("The CAPM returns need the returns of a benchmark.")
        # The betas and the market return are estimated on the periods both have.
        market = benchmark_returns.reindex(returns.index).dropna()
        if len(market) < 2:  # noqa: PLR2004
            raise ValueError("The benchmark shares too few periods with the assets.")
        variance = market.var()
        if variance <= 0:
            raise ValueError("The benchmark's returns have no variance.")
        beta = (
            returns.loc[market.index].apply(lambda column: column.cov(market))
            / variance
        )
        return risk_free_rate + beta * (
            market.mean() * periods_per_year - risk_free_rate
        )
    elif method == "james_stein":
        periods, assets = returns.shape
        if periods <= assets + 2:
            raise ValueError(
                "The James-Stein expected returns need more periods than the number of "
                "assets plus two."
            )
        average = returns.mean().to_numpy()
        # Jorion's estimate of the covariance matrix, S (T - 1) / (T - N - 2).
        inverse = np.linalg.pinv(
            returns.cov().to_numpy() * (periods - 1) / (periods - assets - 2)
        )
        ones = np.ones(assets)
        target = float(ones @ inverse @ average / (ones @ inverse @ ones))
        deviation = average - target
        shrinkage = (assets + 2) / (
            (assets + 2) + periods * float(deviation @ inverse @ deviation)
        )
        mean = pd.Series(
            (1 - shrinkage) * average + shrinkage * target, index=returns.columns
        )
    else:
        raise ValueError(
            f"The expected returns must be one of {', '.join(EXPECTED_RETURN_METHODS)}, "
            f"not '{method}'."
        )
    return mean * periods_per_year


def get_expected_returns_and_covariance(
    returns: pd.DataFrame,
    periods_per_year: int,
    covariance: str = "sample",
    covariance_span: int = 180,
    expected_returns: str = "mean",
    expected_returns_span: int = 180,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    The annualized expected returns (see get_expected_returns) and the annualized
    covariance matrix (the covariance per period times the number of periods in a year)
    of a set of returns.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.
        covariance (str, optional): The estimator of the covariance matrix: "sample"
            (ddof=1), "ledoit_wolf", "oas", "semicovariance" (below a zero return),
            "exponential" or "denoised" (Marčenko-Pastur). Defaults to "sample".
        covariance_span (int, optional): The span in periods of the "exponential"
            covariance. Defaults to 180.
        expected_returns (str, optional): The estimator of the expected returns: "mean",
            "exponential", "capm" or "james_stein". Defaults to "mean".
        expected_returns_span (int, optional): The span in periods of the "exponential"
            expected returns. Defaults to 180.
        benchmark_returns (pd.Series | None, optional): The returns per period of the
            market, for the "capm" expected returns. Defaults to None.
        risk_free_rate (float, optional): The annualized risk free rate, for the "capm"
            expected returns. Defaults to 0.

    Returns:
        tuple[pd.Series, pd.DataFrame]: The expected returns and the covariance matrix.
    """
    returns = returns.dropna(how="any")

    if len(returns) < 2:  # noqa: PLR2004
        raise ValueError("At least two periods of returns are needed for every asset.")

    if covariance == "sample":
        matrix = returns.cov()
    elif covariance == "ledoit_wolf":
        matrix, _ = get_ledoit_wolf_covariance(returns)
    elif covariance == "oas":
        matrix, _ = get_oracle_approximating_shrinkage_covariance(returns)
    elif covariance == "semicovariance":
        matrix = get_semicovariance(returns)
    elif covariance == "exponential":
        matrix = get_exponential_covariance(returns, covariance_span)
    elif covariance == "denoised":
        matrix, _ = get_denoised_covariance(returns)
    else:
        raise ValueError(
            f"The covariance must be one of {', '.join(COVARIANCE_METHODS)}, not "
            f"'{covariance}'."
        )

    mean = get_expected_returns(
        returns,
        periods_per_year,
        expected_returns,
        expected_returns_span,
        benchmark_returns,
        risk_free_rate,
    )
    return mean, matrix * periods_per_year


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
                if portfolio_volatility > ZERO_VOLATILITY_TOLERANCE
                else np.nan
            ),
        }
    )


def get_minimum_variance_weights(
    covariance: pd.DataFrame,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
) -> pd.Series:
    """
    The fully invested portfolio with the lowest variance (Markowitz, 1952).

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance, positive=False)
    sigma = covariance.to_numpy()
    constraints = get_weight_constraints(
        covariance.index,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )

    weights = _solve(lambda w: w @ sigma @ w, lambda w: 2 * sigma @ w, constraints)

    return _as_series(weights, covariance.index)


def get_maximum_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """
    The fully invested portfolio with the highest Sharpe ratio, the tangency portfolio of
    the efficient frontier (Sharpe, 1966; Tobin, 1958). A transaction cost c lowers the
    expected return by c Σ |w_i - w0_i|, so the ratio maximized is
    (w'μ - r_f - c Σ d_i) / √(w'Σw) with d_i ≥ |w_i - w0_i| (skfolio's MeanRisk with
    transaction_costs does the same).

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        risk_free_rate (float, optional): The annualized risk free rate. Defaults to 0.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.
        transaction_cost (float, optional): The cost per unit of weight traded from the
            current weights, e.g. 0.001 for 10 basis points, subtracted once from the
            annualized expected return, which spreads it over a year of holding the
            portfolio. Needs current weights. Defaults to 0.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance)
    mu = expected_returns.to_numpy() - risk_free_rate
    sigma = covariance.to_numpy()
    constraints = get_weight_constraints(
        covariance.index,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )

    # Without short positions no portfolio beats the risk free rate when no asset does.
    if np.all(mu <= 0) and all(bound[0] >= 0 for bound in constraints.bounds):
        raise ValueError(
            "No asset has an expected return above the risk free rate, so no portfolio "
            "has a positive Sharpe ratio."
        )

    costs = _trading_costs(constraints, transaction_cost)
    count = constraints.assets

    def negative_sharpe(x):
        weights = x[:count]
        return -(weights @ mu - costs @ x) / np.sqrt(weights @ sigma @ weights)

    def gradient(x):
        weights = x[:count]
        variance = weights @ sigma @ weights
        volatility = np.sqrt(variance)
        excess = weights @ mu - costs @ x
        slope = costs / volatility
        slope[:count] -= (mu * volatility - excess * (sigma @ weights) / volatility) / (
            variance
        )
        return slope

    weights = _solve(negative_sharpe, gradient, constraints, all_variables=True)

    return _as_series(weights, covariance.index)


def get_mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float | None = None,
    risk_aversion: float = 2.5,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """
    A mean-variance efficient portfolio (Markowitz, 1952). With a target return, the
    portfolio with the lowest variance that expects that return; otherwise the portfolio
    that maximizes the expected return minus half the risk aversion times the variance.
    A transaction cost c lowers the expected return by c Σ |w_i - w0_i|, so the utility
    maximized is w'μ - c Σ d_i - (λ/2) w'Σw with d_i ≥ |w_i - w0_i|, as PyPortfolioOpt's
    transaction_cost objective adds to its quadratic utility.

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        target_return (float | None, optional): The annualized return the portfolio must
            expect. Defaults to None, which uses the risk aversion instead.
        risk_aversion (float, optional): The trade-off between return and variance; higher
            values give less risky portfolios. Defaults to 2.5.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.
        transaction_cost (float, optional): The cost per unit of weight traded from the
            current weights, e.g. 0.001 for 10 basis points, subtracted once from the
            annualized expected return, which spreads it over a year of holding the
            portfolio. Needs current weights. Defaults to 0.

    Returns:
        pd.Series: The weight per asset.
    """
    mu = expected_returns.to_numpy()
    sigma = covariance.to_numpy()
    constraints = get_weight_constraints(
        covariance.index,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )

    if target_return is not None:
        if transaction_cost:
            raise ValueError(
                "A transaction cost applies to the risk aversion form; it cannot be "
                "combined with a target return."
            )
        lowest, highest, _ = _return_range(mu, constraints)
        if not lowest - 1e-9 <= target_return <= highest + 1e-9:
            raise ValueError(
                f"A target return of {target_return:.4f} cannot be reached within the "
                f"constraints ({lowest:.4f} to {highest:.4f})."
            )
        weights = _solve(
            lambda w: w @ sigma @ w,
            lambda w: 2 * sigma @ w,
            constraints,
            [_target_return(mu, target_return)],
        )
    else:
        costs = _trading_costs(constraints, transaction_cost)
        count = constraints.assets

        def gradient(x):
            slope = costs.copy()
            slope[:count] += -mu + risk_aversion * (sigma @ x[:count])
            return slope

        weights = _solve(
            lambda x: -(x[:count] @ mu)
            + costs @ x
            + risk_aversion / 2 * (x[:count] @ sigma @ x[:count]),
            gradient,
            constraints,
            all_variables=True,
        )

    return _as_series(weights, covariance.index)


def get_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    points: int = 20,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
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
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.

    Returns:
        pd.DataFrame: Per portfolio its Expected Return, Volatility, Sharpe Ratio and the
        weight per asset.
    """
    _validate_covariance(covariance)
    if len(covariance) < 2:  # noqa: PLR2004
        raise ValueError("At least two assets are needed.")
    if points < 2:  # noqa: PLR2004
        raise ValueError("The efficient frontier needs at least two points.")

    minimum_variance = get_minimum_variance_weights(
        covariance,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )
    lowest = float(minimum_variance @ expected_returns)

    mu = expected_returns.to_numpy()
    sigma = covariance.to_numpy()
    constraints = get_weight_constraints(
        covariance.index,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )
    # The frontier runs up to the highest return the constraints allow.
    _, highest, highest_weights = _return_range(mu, constraints)
    targets = np.linspace(lowest, highest, points)

    rows = []
    for number, target in enumerate(targets, start=1):
        try:
            weights = _solve(
                lambda w: w @ sigma @ w,
                lambda w: 2 * sigma @ w,
                constraints,
                [_target_return(mu, target)],
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
    _validate_covariance(covariance)
    inverse_volatility = 1 / np.sqrt(np.diag(covariance.to_numpy()))

    return _as_series(inverse_volatility / inverse_volatility.sum(), covariance.index)


def get_risk_parity_weights(
    covariance: pd.DataFrame,
    risk_budget: pd.Series | None = None,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> pd.Series:
    """
    The equal risk contribution (risk parity) portfolio: every asset contributes the same
    share of the portfolio's volatility, or the share given by a risk budget (Maillard,
    Roncalli and Teïletche, 2010). It is found by minimizing the convex function
    1/2 y'Σy - Σ b_i ln(y_i) over y > 0 and scaling the result to sum to one, w = y / 1'y
    (Spinu, 2013).

    Weight and group bounds are constraints on w = y / 1'y, which are linear and
    homogeneous in y (w_i ≤ u becomes y_i - u 1'y ≤ 0), so the problem stays convex and its
    solution does not depend on the scale of y. Within bounds that bind, the risk
    contributions can no longer all match the budget: the result is the constrained risk
    budgeting portfolio of Richard and Roncalli (2019), the same problem skfolio's
    RiskBudgeting solves. Risk parity holds long positions only, so a negative minimum
    weight is refused, and the weights always sum to one.

    See Richard, J.-C. and Roncalli, T. (2019), "Constrained Risk Budgeting Portfolios:
    Theory, Algorithms, Applications & Puzzles", arXiv:1902.05710.

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        risk_budget (pd.Series | None, optional): The share of the risk each asset should
            contribute; it is scaled to sum to one. Defaults to None, an equal share each.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0). Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance)
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

    def objective(y):
        return 0.5 * y @ sigma @ y - budget @ np.log(y)

    def gradient(y):
        return sigma @ y - budget / y

    positive = [(1e-12, None)] * assets
    result = minimize(
        objective,
        np.full(assets, 1 / np.sqrt(np.trace(sigma))),
        jac=gradient,
        method="L-BFGS-B",
        bounds=positive,
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

    constraints = get_weight_constraints(
        covariance.index, minimum_weight, maximum_weight, groups, group_bounds
    )
    lower = np.array([bound[0] for bound in constraints.bounds])
    upper = np.array([bound[1] for bound in constraints.bounds])
    if np.any(lower < 0):
        raise ValueError(
            "Risk parity holds long positions only; the minimum weight cannot be negative."
        )

    # Every constraint a'w ≤ b on the weights becomes (a - b 1)'y ≤ 0 on y, with w = y / 1'y.
    ones = np.ones(assets)
    rows = np.vstack(
        [
            constraints.inequality - constraints.upper[:, None] * ones,
            np.eye(assets) - upper[:, None] * ones,
            lower[:, None] * ones - np.eye(assets),
        ]
    )
    # The unconstrained solution is the answer whenever it meets the constraints.
    if np.all(rows @ weights <= FEASIBILITY_TOLERANCE):
        return _as_series(weights, covariance.index)

    homogeneous = [
        {"type": "ineq", "fun": lambda y: -(rows @ y), "jac": lambda y: -rows}
    ]
    result = minimize(
        objective,
        result.x,
        jac=gradient,
        method="SLSQP",
        bounds=positive,
        constraints=homogeneous,
        options=OPTIMIZER_OPTIONS,
    )
    if not result.success and not (
        result.status == SLSQP_LINE_SEARCH_STATUS
        and _is_feasible(result.x, positive, homogeneous)
    ):
        raise ValueError(f"The optimization did not converge: {result.message}")

    return _as_series(result.x / result.x.sum(), covariance.index)


def get_hierarchical_risk_parity_weights(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
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

    Weight bounds move each split just enough to keep both halves within the sum of
    their assets' bounds, as skfolio and Riskfolio-Lib do: the share α of the left half
    is clipped to [Σ min_left / W, Σ max_left / W], with W the weight of the cluster
    being split, and then 1 - α to the same range of the right half. A split can only
    move weight between the two halves of a cluster, so group bounds, which cut across
    the clusters, cannot be applied.

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering, e.g.
            "single", "complete", "average" or "ward". Defaults to "single".
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0). Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance)
    if len(covariance) < 2:  # noqa: PLR2004
        raise ValueError("At least two assets are needed.")
    constraints = get_weight_constraints(
        covariance.index, minimum_weight, maximum_weight
    )
    lower = pd.Series(
        [bound[0] for bound in constraints.bounds], index=covariance.index
    )
    upper = pd.Series(
        [bound[1] for bound in constraints.bounds], index=covariance.index
    )
    if (lower < 0).any():
        raise ValueError(
            "Hierarchical Risk Parity holds long positions only; the minimum weight "
            "cannot be negative."
        )
    volatility = np.sqrt(np.diag(covariance.to_numpy()))
    correlation = covariance / np.outer(volatility, volatility)

    # The correlation distance of López de Prado, sqrt((1 - rho) / 2), and the Euclidean
    # distance between its columns that the paper clusters on (its code passes the square
    # matrix to linkage, which reads the rows as observations and does exactly this).
    distance = np.sqrt(np.clip((1 - correlation.to_numpy()) / 2, 0, None))
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(pdist(distance), method=linkage_method))
    # The bisection works on positions in numpy: label lookups in pandas took most of
    # the time, which counts in a backtest that runs it on every rebalancing date.
    sigma = covariance.to_numpy()
    lower, upper = lower.to_numpy(), upper.to_numpy()

    def cluster_variance(items: list) -> float:
        sub = sigma[np.ix_(items, items)]
        inverse_variance = 1 / np.diag(sub)
        weights = inverse_variance / inverse_variance.sum()
        return float(weights @ sub @ weights)

    weights = np.ones(len(sigma))
    clusters = [order.tolist()]
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
            # The weight of the cluster being split, which all its assets still share.
            parent = weights[left[0]]
            alpha = min(
                upper[left].sum() / parent, max(lower[left].sum() / parent, alpha)
            )
            alpha = 1 - min(
                upper[right].sum() / parent,
                max(lower[right].sum() / parent, 1 - alpha),
            )
            weights[left] *= alpha
            weights[right] *= 1 - alpha

    return _as_series(weights, covariance.index)


def get_minimum_cvar_weights(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
) -> pd.Series:
    """
    The fully invested portfolio with the lowest Conditional Value at Risk (expected
    shortfall) over the historical returns, solved as a linear programme (Rockafellar and
    Uryasev, 2000).

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        confidence_level (float, optional): The confidence level of the CVaR, e.g. 0.95 for
            the average of the worst 5% of periods. Defaults to 0.95.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.

    Returns:
        pd.Series: The weight per asset.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")

    data = returns.dropna(how="any").to_numpy()
    periods, assets = data.shape
    if periods == 0:
        raise ValueError("No period has a return for every asset.")
    constraints = get_weight_constraints(
        returns.columns,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )
    # The weights and, with a gross exposure limit and short positions, their absolute
    # values, followed by the Value at Risk (1) and the shortfall per period.
    variables = len(constraints.bounds)

    cost = np.concatenate(
        [
            np.zeros(variables),
            [1.0],
            np.full(periods, 1 / ((1 - confidence_level) * periods)),
        ]
    )
    # Shortfall_t >= -r_t'w - VaR, written as -r_t'w - VaR - shortfall_t <= 0. Sparse, since
    # the shortfall block is an identity matrix of periods by periods (a dense one of 25 years
    # of daily returns holds 40 million zeros). The constraints on the weights follow, with
    # zeros for the Value at Risk and the shortfalls.
    inequality = sparse.vstack(
        [
            sparse.hstack(
                [
                    sparse.csr_matrix(-data),
                    sparse.csr_matrix((periods, variables - assets)),
                    sparse.csr_matrix(-np.ones((periods, 1))),
                    -sparse.eye(periods, format="csr"),
                ]
            ),
            sparse.hstack(
                [
                    sparse.csr_matrix(constraints.inequality),
                    sparse.csr_matrix((len(constraints.inequality), 1 + periods)),
                ]
            ),
        ],
        format="csr",
    )
    equality = np.hstack([constraints.equality, np.zeros((1, 1 + periods))])

    result = linprog(
        cost,
        A_ub=inequality,
        b_ub=np.concatenate([np.zeros(periods), constraints.upper]),
        A_eq=equality,
        b_eq=constraints.target,
        bounds=constraints.bounds + [(None, None)] + [(0, None)] * periods,
        method="highs",
    )

    if not result.success:
        raise ValueError(f"The optimization did not converge: {result.message}")

    return _as_series(result.x[:assets], returns.columns)


def get_diversification_ratio(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """
    The diversification ratio of a portfolio (Choueifaty and Coignard, 2008): the weighted
    average volatility of its assets divided by its volatility, w'σ / √(w'Σw). It is one
    for a single asset or perfectly correlated assets and grows as the correlations fall;
    its square is the number of independent risk factors a long-only portfolio holds
    (Choueifaty, Froidure and Reynier, 2013).

    Args:
        weights (pd.Series): The weight per asset.
        covariance (pd.DataFrame): The covariance matrix of the assets.

    Returns:
        float: The diversification ratio.
    """
    weights = weights.reindex(covariance.index).fillna(0).to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    return float(weights @ np.sqrt(np.diag(sigma)) / np.sqrt(weights @ sigma @ weights))


def get_maximum_diversification_weights(
    covariance: pd.DataFrame,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
) -> pd.Series:
    """
    The fully invested portfolio with the highest diversification ratio w'σ / √(w'Σw), the
    most diversified portfolio of Choueifaty and Coignard (2008). The ratio is maximized
    directly with SLSQP within the weight bounds, with its gradient
    σ / √(w'Σw) - (w'σ) Σw / (w'Σw)^(3/2).

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance)
    sigma = covariance.to_numpy()
    volatility = np.sqrt(np.diag(sigma))
    constraints = get_weight_constraints(
        covariance.index,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )

    def negative_ratio(weights):
        return -(weights @ volatility) / np.sqrt(weights @ sigma @ weights)

    def gradient(weights):
        variance = weights @ sigma @ weights
        return -(
            volatility / np.sqrt(variance)
            - (weights @ volatility) * (sigma @ weights) / variance**1.5
        )

    weights = _solve(negative_ratio, gradient, constraints)

    return _as_series(weights, covariance.index)


def get_minimum_tracking_error_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    benchmark: str,
    objective: str = "tracking_error",
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
) -> pd.Series:
    """
    The fully invested portfolio closest to a benchmark (Roll, 1992): with the objective
    "tracking_error", the one with the lowest tracking error, the volatility of its return
    minus the benchmark's, TE²(w) = Var(R w - r_b) = w'Σ_AA w - 2 w'Σ_Ab + σ_b²; with
    "information_ratio", the one with the highest information ratio, the expected active
    return per unit of tracking error, (w'μ - μ_b) / TE(w).

    Args:
        expected_returns (pd.Series): The annualized expected return of the assets and the
            benchmark.
        covariance (pd.DataFrame): The annualized covariance matrix of the assets and the
            benchmark.
        benchmark (str): The name of the benchmark in the expected returns and covariance
            matrix; it is not an asset of the portfolio.
        objective (str, optional): "tracking_error" or "information_ratio". Defaults to
            "tracking_error".
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.

    Returns:
        pd.Series: The weight per asset, without the benchmark.
    """
    if objective not in ("tracking_error", "information_ratio"):
        raise ValueError(
            "The objective must be 'tracking_error' or 'information_ratio', not "
            f"'{objective}'."
        )
    if benchmark not in covariance.index:
        raise ValueError(f"The covariance matrix holds no benchmark '{benchmark}'.")
    _validate_covariance(covariance)
    assets = covariance.index.drop(benchmark)
    if len(assets) < 2:  # noqa: PLR2004
        raise ValueError("At least two assets besides the benchmark are needed.")
    sigma = covariance.loc[assets, assets].to_numpy()
    with_benchmark = covariance.loc[assets, benchmark].to_numpy()
    benchmark_variance = float(covariance.loc[benchmark, benchmark])
    active = expected_returns.reindex(assets).to_numpy() - expected_returns[benchmark]
    constraints = get_weight_constraints(
        assets,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
        current_weights,
        maximum_turnover,
    )

    def tracking_variance(weights):
        return (
            weights @ sigma @ weights
            - 2 * weights @ with_benchmark
            + benchmark_variance
        )

    if objective == "tracking_error":
        weights = _solve(
            tracking_variance,
            lambda w: 2 * sigma @ w - 2 * with_benchmark,
            constraints,
        )
    else:
        if np.all(active <= 0) and all(bound[0] >= 0 for bound in constraints.bounds):
            raise ValueError(
                "No asset has an expected return above the benchmark's, so no portfolio "
                "has a positive information ratio."
            )

        def gradient(weights):
            variance = tracking_variance(weights)
            excess = weights @ active
            return -(
                active / np.sqrt(variance)
                - excess * (sigma @ weights - with_benchmark) / variance**1.5
            )

        weights = _solve(
            lambda w: -(w @ active) / np.sqrt(tracking_variance(w)),
            gradient,
            constraints,
        )

    return _as_series(weights, assets)


def get_tracking_error(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    benchmark: str,
) -> pd.Series:
    """
    The expected active return, tracking error and information ratio of a portfolio
    against a benchmark: w'μ - μ_b, √(w'Σ_AA w - 2 w'Σ_Ab + σ_b²) and their ratio
    (Roll, 1992; Grinold and Kahn, 2000).

    Args:
        weights (pd.Series): The weight per asset.
        expected_returns (pd.Series): The annualized expected return of the assets and the
            benchmark.
        covariance (pd.DataFrame): The annualized covariance matrix of the assets and the
            benchmark.
        benchmark (str): The name of the benchmark in the expected returns and covariance
            matrix.

    Returns:
        pd.Series: The Active Return, Tracking Error and Information Ratio.
    """
    # The active position: the weights of the assets against a short position in the
    # benchmark, whose variance is the tracking variance.
    active = weights.reindex(covariance.index).fillna(0)
    active[benchmark] -= 1
    active_return = float(active @ expected_returns.reindex(covariance.index))
    tracking_error = float(np.sqrt(max(active @ covariance @ active, 0)))

    return pd.Series(
        {
            "Active Return": active_return,
            "Tracking Error": tracking_error,
            "Information Ratio": (
                active_return / tracking_error
                if tracking_error > ZERO_VOLATILITY_TOLERANCE
                else np.nan
            ),
        }
    )


def get_growth_optimal_weights(
    returns: pd.DataFrame,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    method: str = "exact",
    fraction: float = 1.0,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
) -> pd.Series:
    """
    The growth optimal (Kelly) portfolio: the fully invested portfolio with the highest
    expected logarithmic growth of wealth (Kelly, 1956; Breiman, 1961; Thorp, 2006). The
    "exact" method maximizes the average log return over the historical returns,
    (1/T) Σ ln(1 + r_t'w), with gradient (1/T) Σ r_t / (1 + r_t'w). The "approximate"
    method maximizes its second-order approximation w'μ - ½ w'Σw, a mean-variance
    portfolio with a risk aversion of one. Fractional Kelly scales the weights by the
    fraction; the remainder (1 - fraction) is cash, or borrowed when the fraction is
    above one.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, for the
            "exact" method.
        expected_returns (pd.Series): The expected return per asset, for the
            "approximate" method, on the same scale (per period or annualized) as the
            covariance.
        covariance (pd.DataFrame): The covariance matrix of the assets, for the
            "approximate" method.
        method (str, optional): "exact" or "approximate". Defaults to "exact".
        fraction (float, optional): The share of the growth optimal weights to hold, e.g.
            0.5 for half Kelly. Defaults to 1.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification, one row per asset and one column per classification (e.g.
            "Sector"). Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group,
            e.g. {"Sector": {"Technology": (0, 0.4)}}. Defaults to None.
        net_exposure (float, optional): The sum of the weights, 1 for a fully invested
            portfolio. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.

    The constraints apply to the full Kelly portfolio, before the fraction scales it.

    Returns:
        pd.Series: The weight per asset, summing to the fraction times the net exposure.
    """
    if method not in ("exact", "approximate"):
        raise ValueError(
            f"The method must be 'exact' or 'approximate', not '{method}'."
        )
    if fraction <= 0:
        raise ValueError("The fraction must be positive.")

    if method == "approximate":
        weights = get_mean_variance_weights(
            expected_returns,
            covariance,
            risk_aversion=1.0,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
            groups=groups,
            group_bounds=group_bounds,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
        )
        return _as_series(weights.to_numpy() * fraction, weights.index)

    data = returns.dropna(how="any").to_numpy(dtype=float)
    if len(data) == 0:
        raise ValueError("No period has a return for every asset.")
    constraints = get_weight_constraints(
        returns.columns,
        minimum_weight,
        maximum_weight,
        groups,
        group_bounds,
        net_exposure,
        gross_exposure,
    )

    # The sum rather than the average of the log returns keeps the objective of daily
    # returns large enough for the optimizer's tolerance. A wealth of zero or less in a
    # period (possible with short positions) is ruin, an infinitely bad outcome.
    def negative_growth(weights):
        wealth = 1 + data @ weights
        if np.any(wealth <= 0):
            return np.inf
        return -np.sum(np.log(wealth))

    weights = _solve(
        negative_growth,
        lambda w: -(data / (1 + data @ w)[:, None]).sum(axis=0),
        constraints,
    )

    return _as_series(weights * fraction, returns.columns)


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

    unmatched = set(view_confidences or {}) - set(views or {})
    if unmatched:
        raise ValueError(
            f"The view confidences name views that are not given: {', '.join(sorted(unmatched))}."
        )

    if views:
        pick, targets, uncertainty = [], [], []
        for view, value in views.items():
            row = np.zeros(len(assets))
            # Tickers can hold a hyphen themselves (BRK-B), so every hyphen is tried as the
            # split between the two legs of a relative view.
            legs = [
                (view[:position], view[position + 1 :])
                for position, character in enumerate(view)
                if character == "-"
                and view[:position] in assets
                and view[position + 1 :] in assets
            ]
            if view in assets:
                row[assets.get_loc(view)] = 1
            elif len(legs) == 1:
                long_leg, short_leg = legs[0]
                row[assets.get_loc(long_leg)] = 1
                row[assets.get_loc(short_leg)] = -1
            elif len(legs) > 1:
                raise ValueError(
                    f"The view {view!r} can be read as more than one pair of assets."
                )
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
        # The posterior Π + τΣP'(PτΣP' + Ω)^-1 (Q - PΠ), the same as the textbook
        # ((τΣ)^-1 + P'Ω^-1 P)^-1 ((τΣ)^-1 Π + P'Ω^-1 Q) but without inverting Σ or Ω, so
        # duplicate or collinear assets (a singular covariance) and certain views work.
        tau_sigma = tau * sigma
        try:
            adjustment = np.linalg.solve(p @ tau_sigma @ p.T + omega, q - p @ implied)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "The views contradict or repeat each other, so they cannot be combined."
            ) from error
        posterior = implied + tau_sigma @ p.T @ adjustment

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


def get_expected_shortfall_contributions(
    weights: pd.Series, returns: pd.DataFrame, confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    How much each asset adds to the historical expected shortfall (CVaR) of a portfolio
    (Euler allocation). The expected shortfall is the average loss in the periods in
    which the portfolio's return is at or below its Value at Risk, the
    (1 - confidence level) quantile of its returns, ES = -E[r_p | r_p ≤ VaR]. Because it
    is homogeneous of degree one in the weights, its derivative per asset,
    ∂ES/∂w_i = -E[r_i | r_p ≤ VaR], times the weight gives contributions that add up to
    the expected shortfall (Tasche, 2000; Scaillet, 2004). The tail periods are those of
    the portfolio, so an asset that tends to gain when the portfolio loses contributes
    less than its own expected shortfall, or a negative amount when it hedges.

    The expected shortfall is the average of the returns at or below the quantile, as in
    the Finance Toolkit's historical CVaR and the CVaR column of the minimum CVaR
    weights; it equals the Rockafellar-Uryasev CVaR of the minimum CVaR optimization
    when the tail holds a whole number of periods.

    See Tasche, D. (2000), "Conditional Expectations as Quantile Derivatives", arXiv
    math/0104190, and Scaillet, O. (2004), "Nonparametric Estimation and Sensitivity
    Analysis of Expected Shortfall", Mathematical Finance, 14(1), 115-129.

    Args:
        weights (pd.Series): The weight per asset.
        returns (pd.DataFrame): The returns per period, one column per asset.
        confidence_level (float, optional): The confidence level, e.g. 0.95 for the
            worst 5% of periods. Defaults to 0.95.

    Returns:
        pd.DataFrame: The Weight, Marginal Contribution (-E[r_i | tail]), Risk
        Contribution (the weight times the marginal contribution) and Percentage
        Contribution per asset, as losses per period.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("No period has a return for every asset.")
    weights = weights.reindex(returns.columns).fillna(0)
    portfolio = returns @ weights
    tail = portfolio <= portfolio.quantile(1 - confidence_level)

    marginal = -returns[tail].mean().to_numpy()
    contribution = weights.to_numpy() * marginal
    shortfall = contribution.sum()
    if shortfall == 0:
        raise ValueError("The portfolio has no expected shortfall.")

    return pd.DataFrame(
        {
            "Weight": weights.to_numpy(),
            "Marginal Contribution": marginal,
            "Risk Contribution": contribution,
            "Percentage Contribution": contribution / shortfall,
        },
        index=returns.columns,
    )


def get_diversification_metrics(
    weights: pd.Series, covariance: pd.DataFrame
) -> pd.Series:
    """
    How diversified a portfolio is, by its weights and by its risk.

    - Herfindahl-Hirschman Index: Σ w_i², the concentration of the weights, from 1/N for
      equal weights to 1 for a single asset.
    - Effective Number of Assets: 1 / Σ w_i², the number of equally weighted assets with
      the same concentration.
    - Diversification Ratio: w'σ / √(w'Σw), the weighted average volatility of the assets
      over the portfolio's volatility (Choueifaty and Coignard, 2008).
    - Effective Number of Bets: exp(-Σ p_k ln p_k), with p_k = λ_k (e_k'w)² / w'Σw the
      share of the portfolio's variance from principal component k (eigenvalue λ_k,
      eigenvector e_k) of the covariance matrix, the exponential of the entropy of the
      uncorrelated risk sources: 1 when all risk comes from one source and N when it is
      spread evenly over all of them (Meucci, 2009).

    See Meucci, A. (2009), "Managing Diversification", Risk, 22(5), 74-79, and
    Choueifaty, Y. and Coignard, Y. (2008), "Toward Maximum Diversification", The Journal
    of Portfolio Management, 35(1), 40-51.

    Args:
        weights (pd.Series): The weight per asset.
        covariance (pd.DataFrame): The covariance matrix of the assets.

    Returns:
        pd.Series: The Herfindahl-Hirschman Index, Effective Number of Assets,
        Diversification Ratio and Effective Number of Bets.
    """
    _validate_covariance(covariance, positive=False)
    vector = weights.reindex(covariance.index).fillna(0).to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    variance = float(vector @ sigma @ vector)
    if variance <= ZERO_VOLATILITY_TOLERANCE**2:
        raise ValueError("The portfolio has no volatility.")

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    shares = np.clip(eigenvalues, 0, None) * (eigenvectors.T @ vector) ** 2
    # Normalized by their sum, which is the variance when the covariance matrix is
    # positive semi-definite and keeps the shares a distribution when it is not.
    shares = shares[shares > 0] / shares.sum()
    concentration = float(np.sum(vector**2))

    return pd.Series(
        {
            "Herfindahl-Hirschman Index": concentration,
            "Effective Number of Assets": 1 / concentration,
            "Diversification Ratio": get_diversification_ratio(weights, covariance),
            "Effective Number of Bets": float(np.exp(-np.sum(shares * np.log(shares)))),
        }
    )
