"""Portfolio Optimization Model"""

__docformat__ = "google"

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import cut_tree, fcluster, leaves_list, linkage, to_tree
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import pdist, squareform

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
      a market neutral one). A net exposure of 0 or below needs a negative minimum_weight
      for at least one asset.
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
    # Without short positions the only portfolio with a net exposure of 0 holds nothing and
    # a negative net exposure holds less than nothing.
    if net_exposure <= 0 and np.all(lower >= 0):
        raise ValueError(
            f"A net exposure of {net_exposure} needs short positions: set a negative "
            "minimum_weight for at least one asset."
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
    direction: np.ndarray | None = None,
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

    With a net exposure of 0 the equal weights are all 0, where a ratio such as the
    Sharpe or diversification ratio is 0 / 0. When the objective is not finite there,
    SLSQP starts halfway between them and the portfolio within the constraints with the
    highest weights @ direction (a corner found by linear programming), e.g. the
    expected excess returns for the Sharpe ratio.
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
    with np.errstate(invalid="ignore", divide="ignore"):
        undefined = not np.isfinite(
            objective(start) if all_variables else objective(start[:assets])
        )
    if direction is not None and undefined:
        corner = _linear_programme(
            np.concatenate([-direction, np.zeros(len(lower) - assets)]), constraints
        ).x
        start = (start + corner) / 2

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


def _homogeneous_constraints(
    constraints: WeightConstraints,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The constraints on x rewritten for the Charnes-Cooper transformation of a ratio,
    over the variables (y, κ) with y = κx and κ > 0: A x ≤ b becomes A y - b κ ≤ 0, the
    equality E x = n becomes E y - n κ = 0 and every bound l ≤ x_i ≤ u becomes the rows
    y_i - u κ ≤ 0 and l κ - y_i ≤ 0. All of them are homogeneous, so the scale of y is
    free to normalize the denominator of the ratio.

    Returns:
        tuple[np.ndarray, np.ndarray]: The inequality (≤ 0) and equality (= 0) rows over
            the variables of the constraints followed by κ.
    """
    variables = len(constraints.bounds)
    identity = np.eye(variables)
    rows = [np.hstack([constraints.inequality, -constraints.upper[:, None]])]
    for i, (low, high) in enumerate(constraints.bounds):
        if high is not None and np.isfinite(high):
            rows.append(np.append(identity[i], -high)[None, :])
        if low is not None and np.isfinite(low):
            rows.append(np.append(-identity[i], low)[None, :])
    equality = np.hstack([constraints.equality, -constraints.target[:, None]])
    return np.vstack(rows), equality


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


def get_ledoit_wolf_constant_correlation_covariance(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    The Ledoit-Wolf shrinkage estimator of the covariance matrix with the constant
    correlation target: the sample covariance S pulled towards the matrix F that keeps
    every sample variance and sets every correlation to the average sample correlation,
    f_ii = s_ii and f_ij = r̄ √(s_ii s_jj) with r̄ = 2 / (N (N - 1)) Σ_(i<j) r_ij. Unlike
    the scaled identity, the target keeps each asset's own variance and the general level
    of co-movement between assets, so it shrinks only what is noisy: the differences
    between the correlations.

    The shrinkage intensity is the one of the paper that minimizes the expected squared
    (Frobenius) distance to the true covariance matrix, δ = max(0, min(κ / T, 1)) with
    κ = (π - ρ) / γ, where, for the T demeaned returns y_t,

    - π = Σ_ij (1 / T) Σ_t (y_it y_jt - s_ij)², the variance of the sample covariances,
    - ρ = Σ_i π_ii + Σ_(i≠j) (r̄ / 2) (√(s_jj / s_ii) ϑ_ii,ij + √(s_ii / s_jj) ϑ_jj,ij), with
      ϑ_ii,ij = (1 / T) Σ_t (y_it² - s_ii)(y_it y_jt - s_ij), their covariance with the
      target, and
    - γ = Σ_ij (f_ij - s_ij)², the squared distance between the target and S.

    The estimate is (1 - δ) S + δ F, with S the maximum likelihood sample covariance
    (divided by T, as in the paper).

    See Ledoit, O. and Wolf, M. (2004), "Honey, I Shrunk the Sample Covariance Matrix",
    The Journal of Portfolio Management, 30(4), 110-119.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.DataFrame, float]: The covariance matrix per period and the shrinkage
            intensity between 0 (the sample covariance) and 1 (the constant correlation
            target).
    """
    values = returns.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    periods, assets = centered.shape
    if assets < 2:  # noqa: PLR2004
        raise ValueError("The constant correlation target needs at least two assets.")
    sample = centered.T @ centered / periods
    variance = np.diag(sample)
    if np.any(variance <= 0):
        raise ValueError("Every asset needs returns that vary.")
    volatility = np.sqrt(variance)
    correlation = sample / np.outer(volatility, volatility)
    average_correlation = (correlation.sum() - assets) / (assets * (assets - 1))
    target = average_correlation * np.outer(volatility, volatility)
    np.fill_diagonal(target, variance)

    squared = centered**2
    # pi_ij, the variance of each sample covariance, and theta_ii,ij, the covariance of
    # the sample variance of i with the sample covariance of i and j.
    pi = squared.T @ squared / periods - sample**2
    theta = (centered**3).T @ centered / periods - variance[:, None] * sample
    np.fill_diagonal(theta, 0)
    rho = np.trace(pi) + average_correlation * np.sum(
        np.outer(1 / volatility, volatility) * theta
    )
    gamma = np.sum((target - sample) ** 2)
    shrinkage = (
        0.0 if gamma == 0 else max(0.0, min((pi.sum() - rho) / gamma / periods, 1.0))
    )

    return (
        pd.DataFrame(
            (1 - shrinkage) * sample + shrinkage * target,
            index=returns.columns,
            columns=returns.columns,
        ),
        shrinkage,
    )


def get_ledoit_wolf_single_factor_covariance(
    returns: pd.DataFrame, benchmark_returns: pd.Series
) -> tuple[pd.DataFrame, float]:
    """
    The Ledoit-Wolf shrinkage estimator of the covariance matrix with the single factor
    target: the sample covariance S pulled towards the covariance matrix F that the market
    model implies, with the benchmark as the market. Each asset's return is β_i r_m plus
    a residual that is unrelated to the other assets, so f_ij = s_im s_jm / s_mm off the
    diagonal (β_i β_j σ²_m, with s_im the sample covariance of asset i and the benchmark and
    s_mm the sample variance of the benchmark) and f_ii = s_ii on it. The target holds only
    2N + 1 parameters, so it is estimated precisely, and S adds what the factor leaves out.

    The shrinkage intensity is the one of the paper that minimizes the expected squared
    (Frobenius) distance to the true covariance matrix, δ = max(0, min(κ / T, 1)) with
    κ = (π - ρ) / γ, where, for the T demeaned returns y_t of the assets and m_t of the
    benchmark,

    - π = Σ_ij (1 / T) Σ_t (y_it y_jt - s_ij)², the variance of the sample covariances,
    - ρ = Σ_i π_ii + Σ_(i≠j) (1 / T) Σ_t [(s_jm s_mm y_it + s_im s_mm y_jt - s_im s_jm m_t)
      m_t y_it y_jt / s_mm² - f_ij s_ij], their covariance with the target, and
    - γ = Σ_ij (f_ij - s_ij)², the squared distance between the target and S.

    The estimate is (1 - δ) S + δ F, with S the maximum likelihood sample covariance
    (divided by T, as in the paper). The paper uses the value-weighted market of its
    stocks; here the factor is the benchmark, on the periods the benchmark and the assets
    share.

    See Ledoit, O. and Wolf, M. (2003), "Improved Estimation of the Covariance Matrix of
    Stock Returns with an Application to Portfolio Selection", Journal of Empirical
    Finance, 10(5), 603-621.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        benchmark_returns (pd.Series): The returns per period of the benchmark, the
            factor of the target.

    Returns:
        tuple[pd.DataFrame, float]: The covariance matrix per period and the shrinkage
            intensity between 0 (the sample covariance) and 1 (the single factor target).
    """
    market = benchmark_returns.reindex(returns.index).dropna()
    if len(market) < 2:  # noqa: PLR2004
        raise ValueError(
            "The single factor target needs the returns of a benchmark on the same "
            "periods as the assets."
        )
    values = returns.loc[market.index].to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    factor = market.to_numpy(dtype=float) - market.mean()
    periods = len(centered)
    sample = centered.T @ centered / periods
    covariance_market = centered.T @ factor / periods
    variance_market = float(factor @ factor / periods)
    if variance_market <= 0:
        raise ValueError("The benchmark's returns have no variance.")
    target = np.outer(covariance_market, covariance_market) / variance_market
    np.fill_diagonal(target, np.diag(sample))

    squared = centered**2
    pi = squared.T @ squared / periods - sample**2
    # The covariance of the target with S off the diagonal, the paper's r_ij averaged
    # over the periods: two terms in which one asset's covariance with the factor is
    # estimated with the other asset, and one in which both are.
    scaled = centered * factor[:, None]
    first = (squared.T @ scaled / periods - covariance_market[:, None] * sample) * (
        covariance_market[None, :] / variance_market
    )
    third = (scaled.T @ scaled / periods - variance_market * sample) * (
        np.outer(covariance_market, covariance_market) / variance_market**2
    )
    off_diagonal = first + first.T - third
    np.fill_diagonal(off_diagonal, 0)
    rho = np.trace(pi) + off_diagonal.sum()
    gamma = np.sum((target - sample) ** 2)
    shrinkage = (
        0.0 if gamma == 0 else max(0.0, min((pi.sum() - rho) / gamma / periods, 1.0))
    )

    return (
        pd.DataFrame(
            (1 - shrinkage) * sample + shrinkage * target,
            index=returns.columns,
            columns=returns.columns,
        ),
        shrinkage,
    )


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
    "ledoit_wolf_constant_correlation",
    "ledoit_wolf_single_factor",
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
            (ddof=1), "ledoit_wolf" (towards the scaled identity),
            "ledoit_wolf_constant_correlation", "ledoit_wolf_single_factor" (towards the
            covariance the benchmark as a single factor implies), "oas", "semicovariance"
            (below a zero return), "exponential" or "denoised" (Marčenko-Pastur).
            Defaults to "sample".
        covariance_span (int, optional): The span in periods of the "exponential"
            covariance. Defaults to 180.
        expected_returns (str, optional): The estimator of the expected returns: "mean",
            "exponential", "capm" or "james_stein". Defaults to "mean".
        expected_returns_span (int, optional): The span in periods of the "exponential"
            expected returns. Defaults to 180.
        benchmark_returns (pd.Series | None, optional): The returns per period of the
            market, for the "capm" expected returns and the "ledoit_wolf_single_factor"
            covariance. Defaults to None.
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
    elif covariance == "ledoit_wolf_constant_correlation":
        matrix, _ = get_ledoit_wolf_constant_correlation_covariance(returns)
    elif covariance == "ledoit_wolf_single_factor":
        if benchmark_returns is None:
            raise ValueError(
                "The single factor covariance needs the returns of a benchmark."
            )
        matrix, _ = get_ledoit_wolf_single_factor_covariance(returns, benchmark_returns)
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
            portfolio. At 0 (market neutral) the lowest variance is that of holding
            nothing, so all weights are 0. Defaults to 1.
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

    weights = _solve(
        negative_sharpe, gradient, constraints, all_variables=True, direction=mu
    )

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
    target_volatility: float | None = None,
) -> pd.Series:
    """
    A mean-variance efficient portfolio (Markowitz, 1952). With a target return, the
    portfolio with the lowest variance that expects that return. With a target
    volatility σ*, the portfolio with the highest expected return whose volatility is at
    most σ*: max w'μ subject to w'Σw ≤ σ*², the same efficient frontier approached from
    the risk side (Merton, 1972). Otherwise the portfolio that maximizes the expected
    return minus half the risk aversion times the variance. A transaction cost c lowers
    the expected return by c Σ |w_i - w0_i|, so the utility maximized is
    w'μ - c Σ d_i - (λ/2) w'Σw with d_i ≥ |w_i - w0_i|, and with a target volatility the
    return maximized is w'μ - c Σ d_i.

    See Markowitz, H. (1952), "Portfolio Selection", The Journal of Finance, 7(1), 77-91,
    and Merton, R. C. (1972), "An Analytic Derivation of the Efficient Portfolio
    Frontier", Journal of Financial and Quantitative Analysis, 7(4), 1851-1872.

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        target_return (float | None, optional): The annualized return the portfolio must
            expect. Defaults to None, which uses the target volatility or the risk
            aversion instead.
        risk_aversion (float, optional): The trade-off between return and variance; higher
            values give less risky portfolios. Not used with a target return or target
            volatility. Defaults to 2.5.
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
        target_volatility (float | None, optional): The highest annualized volatility
            √(w'Σw) of the portfolio, e.g. 0.15 for 15%; it must lie at or above the
            volatility of the minimum variance portfolio within the constraints. Cannot
            be combined with a target return. Defaults to None, which uses the target
            return or the risk aversion instead.

    Returns:
        pd.Series: The weight per asset.
    """
    if target_return is not None and target_volatility is not None:
        raise ValueError(
            "A portfolio takes a target return or a target volatility, not both: the "
            "one fixes the other on the efficient frontier."
        )
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
    elif target_volatility is not None:
        if target_volatility <= 0:
            raise ValueError("The target volatility must be positive.")
        lowest = _solve(lambda w: w @ sigma @ w, lambda w: 2 * sigma @ w, constraints)
        lowest_volatility = float(np.sqrt(lowest @ sigma @ lowest))
        if target_volatility < lowest_volatility - 1e-9:
            raise ValueError(
                f"A target volatility of {target_volatility:.4f} is below the lowest "
                f"volatility a portfolio within the constraints can have "
                f"({lowest_volatility:.4f})."
            )
        costs = _trading_costs(constraints, transaction_cost)
        count = constraints.assets

        def return_gradient(x):
            slope = costs.copy()
            slope[:count] -= mu
            return slope

        weights = _solve(
            lambda x: -(x[:count] @ mu) + costs @ x,
            return_gradient,
            constraints,
            # The variance budget sigma*^2 - w'Sigma w >= 0, with a target that falls
            # short of the lowest volatility only by rounding raised to it.
            [
                {
                    "type": "ineq",
                    "fun": lambda w: max(target_volatility, lowest_volatility) ** 2
                    - w @ sigma @ w,
                    "jac": lambda w: -2 * sigma @ w,
                }
            ],
            all_variables=True,
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
            portfolio. At 0 (market neutral) the minimum variance portfolio holds nothing,
            so the first portfolio has all weights 0 and no Sharpe ratio (NaN). Defaults
            to 1.
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


CODEPENDENCE_METHODS = ("pearson", "spearman", "kendall")


def get_codependence(
    returns: pd.DataFrame, covariance: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """
    The correlation matrix the hierarchical clustering of the assets works on, through
    the correlation distance d = √((1 - ρ) / 2) (Mantegna, 1999; López de Prado, 2016).

    - "pearson": the linear correlation ρ_ij = σ_ij / (σ_i σ_j) of the covariance matrix,
      so a shrinkage or denoised covariance estimator also shapes the clusters.
    - "spearman": the rank correlation of Spearman (1904), the Pearson correlation of the
      ranks of the returns, 1 - 6 Σ d_t² / (T (T² - 1)) without ties, with d_t the
      difference between the ranks of the two assets' returns in period t.
    - "kendall": the rank correlation τ_b of Kendall (1938, 1945), (n_c - n_d) /
      √((n_0 - n_1)(n_0 - n_2)), with n_c and n_d the concordant and discordant pairs of
      periods, n_0 = T (T - 1) / 2 and n_1, n_2 the pairs tied in either asset.

    The rank correlations depend only on the order of the returns, so a few extreme
    returns do not dominate them, and they capture any monotonic co-movement rather than
    a linear one only. They are taken from the returns themselves, whatever the
    covariance estimator.

    See Spearman, C. (1904), "The Proof and Measurement of Association between Two
    Things", The American Journal of Psychology, 15(1), 72-101, Kendall, M. G. (1938),
    "A New Measure of Rank Correlation", Biometrika, 30(1/2), 81-93, and Mantegna, R. N.
    (1999), "Hierarchical Structure in Financial Markets", The European Physical Journal
    B, 11(1), 193-197.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        covariance (pd.DataFrame): The covariance matrix of the assets, for "pearson".
        method (str, optional): "pearson", "spearman" or "kendall". Defaults to
            "pearson".

    Returns:
        pd.DataFrame: The correlation matrix, in the order of the covariance matrix.
    """
    if method not in CODEPENDENCE_METHODS:
        raise ValueError(
            f"The codependence must be one of {', '.join(CODEPENDENCE_METHODS)}, not "
            f"'{method}'."
        )
    if method == "pearson":
        volatility = np.sqrt(np.diag(covariance.to_numpy()))
        return covariance / np.outer(volatility, volatility)
    return returns[covariance.index].corr(method=method)


def get_hierarchical_risk_parity_weights(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    correlation: pd.DataFrame | None = None,
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
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on, e.g. a rank correlation from get_codependence. Defaults to
            None, the correlation matrix of the covariance matrix.

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
    if correlation is None:
        volatility = np.sqrt(np.diag(covariance.to_numpy()))
        correlation = covariance / np.outer(volatility, volatility)

    # The correlation distance of López de Prado, sqrt((1 - rho) / 2), and the Euclidean
    # distance between its columns that the paper clusters on (its code passes the square
    # matrix to linkage, which reads the rows as observations and does exactly this).
    distance = np.sqrt(
        np.clip(
            (1 - correlation.loc[covariance.index, covariance.index].to_numpy()) / 2,
            0,
            None,
        )
    )
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


LINKAGE_METHODS = ("single", "complete", "average", "ward")


def get_hierarchical_clusters(
    covariance: pd.DataFrame,
    linkage_method: str = "ward",
    clusters: int | None = None,
    correlation: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster the assets on their correlations: a hierarchical clustering on the
    correlation distance d = √((1 - ρ) / 2) (Mantegna, 1999), cut into a number of
    clusters.

    Without a number of clusters, it is chosen by the two-order difference of the gap
    statistic of Yue, Wang and Wei (2008): with W_k the pooled within-cluster dispersion
    of the cut into k clusters (Tibshirani, Walther and Hastie, 2001), the sum over the
    clusters of D_r / (2 n_r), with D_r the sum of the squared distances between the n_r
    assets of cluster r, the number of clusters is the k + 1 that maximizes
    W_(k+2) + W_k - 2 W_(k+1), the sharpest bend in the dispersion, with k up to
    max(8, √N) clusters for N assets. skfolio and Riskfolio-Lib sum the average distance
    within each cluster instead, which does not fall as clusters split and tends to pick
    more clusters than the correlation blocks hold.

    See Yue, S., Wang, X. and Wei, M. (2008), "Application of Two-Order Difference to Gap
    Statistic", Transactions of Tianjin University, 14(3), 217-221, and Tibshirani, R.,
    Walther, G. and Hastie, T. (2001), "Estimating the Number of Clusters in a Data Set
    via the Gap Statistic", Journal of the Royal Statistical Society B, 63(2), 411-423.

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering:
            "single", "complete", "average" or "ward". Defaults to "ward".
        clusters (int | None, optional): The number of clusters. Defaults to None, chosen
            by the two-order difference of the gap statistic.
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on, e.g. a rank correlation from get_codependence. Defaults to
            None, the correlation matrix of the covariance matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The linkage matrix of the clustering and the
            cluster of every asset, numbered from 0.
    """
    if linkage_method not in LINKAGE_METHODS:
        raise ValueError(
            f"The linkage method must be one of {', '.join(LINKAGE_METHODS)}, not "
            f"'{linkage_method}'."
        )
    count = len(covariance)
    if count < 2:  # noqa: PLR2004
        raise ValueError("At least two assets are needed.")
    if clusters is not None and not 1 <= clusters <= count:
        raise ValueError(f"The number of clusters must lie between 1 and {count}.")

    if correlation is None:
        volatility = np.sqrt(np.diag(covariance.to_numpy()))
        correlation = covariance / np.outer(volatility, volatility)
    distance = np.sqrt(
        np.clip(
            (1 - correlation.loc[covariance.index, covariance.index].to_numpy()) / 2,
            0,
            None,
        )
    )
    np.fill_diagonal(distance, 0)
    matrix = linkage(squareform(distance, checks=False), method=linkage_method)

    if clusters is None:
        if count == 2:  # noqa: PLR2004
            # Two assets have no second difference; each is its own cluster.
            clusters = 2
        else:
            # Column j of the cut tree holds the clusters of the cut into N - j clusters.
            levels = cut_tree(matrix)
            dispersion = []
            for k in range(1, min(count, max(8, round(np.sqrt(count)))) + 1):
                level = levels[:, count - k]
                # Tibshirani, Walther and Hastie's pooled dispersion: per cluster the sum
                # of the squared distances between its assets over twice its size. The
                # distance is Euclidean between standardized returns, so this is the
                # within-cluster sum of squares and falls as clusters split.
                dispersion.append(
                    sum(
                        (distance[np.ix_(members, members)] ** 2).sum()
                        / (2 * len(members))
                        for label in np.unique(level)
                        if len(members := np.flatnonzero(level == label)) > 1
                    )
                )
            dispersion = np.array(dispersion)
            gaps = dispersion[2:] + dispersion[:-2] - 2 * dispersion[1:-1]
            clusters = int(np.argmax(gaps)) + 2

    return matrix, fcluster(matrix, t=clusters, criterion="maxclust") - 1


def get_hierarchical_equal_risk_contribution_weights(
    covariance: pd.DataFrame,
    linkage_method: str = "ward",
    clusters: int | None = None,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    correlation: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Hierarchical Equal Risk Contribution (Raffinot, 2018): the assets are clustered on
    their correlations and the tree is cut into clusters. Within a cluster the weights
    are inverse variance (naive risk parity). Between clusters, the weight is split
    top-down along the dendrogram: at every node the two branches get weights in
    inverse proportion to their risk, the sum of the variances of the inverse variance
    portfolios of the clusters in each branch. Unlike Hierarchical Risk Parity, which
    bisects the ordered list of assets, the splits follow the tree, so similar assets
    stay together.

    The weight and group bounds are applied afterwards as skfolio does: when the
    weights fall outside them, they move to the weights within the bounds with the
    smallest relative deviation, Σ (w_i / w_HERC,i - 1)².

    See Raffinot, T. (2018), "The Hierarchical Equal Risk Contribution Portfolio", SSRN
    3237540.

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering:
            "single", "complete", "average" or "ward". Defaults to "ward", as in the paper.
        clusters (int | None, optional): The number of clusters. Defaults to None, chosen
            by the two-order difference of the gap statistic.
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
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on, e.g. a rank correlation from get_codependence. Defaults to
            None, the correlation matrix of the covariance matrix.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance)
    constraints = get_weight_constraints(
        covariance.index, minimum_weight, maximum_weight, groups, group_bounds
    )
    if any(bound[0] < 0 for bound in constraints.bounds):
        raise ValueError(
            "Hierarchical Equal Risk Contribution holds long positions only; the minimum "
            "weight cannot be negative."
        )
    matrix, labels = get_hierarchical_clusters(
        covariance, linkage_method, clusters, correlation
    )
    sigma = covariance.to_numpy()

    members = [np.flatnonzero(labels == label) for label in range(labels.max() + 1)]
    weights = np.zeros(len(sigma))
    risk = np.zeros(len(members))
    for label, assets in enumerate(members):
        inverse_variance = 1 / np.diag(sigma)[assets]
        weights[assets] = inverse_variance / inverse_variance.sum()
        risk[label] = weights[assets] @ sigma[np.ix_(assets, assets)] @ weights[assets]

    # Walk down the dendrogram until every node is one of the clusters, splitting the
    # weight of each node between the clusters under its two branches.
    cluster_weights = np.ones(len(members))
    leaves = [set(assets) for assets in members]
    nodes = [to_tree(matrix)]
    while nodes:
        node = nodes.pop()
        if set(node.pre_order()) in leaves:
            continue
        branches = [set(node.get_left().pre_order()), set(node.get_right().pre_order())]
        left, right = (
            [label for label, assets in enumerate(leaves) if assets <= branch]
            for branch in branches
        )
        alpha = 1 - risk[left].sum() / (risk[left].sum() + risk[right].sum())
        cluster_weights[left] *= alpha
        cluster_weights[right] *= 1 - alpha
        nodes.extend([node.get_left(), node.get_right()])

    weights = weights * cluster_weights[labels]

    if not _is_feasible(weights, constraints.bounds, _linear_constraints(constraints)):
        # Solved over the relative weights v = w / w_HERC, which keeps the objective
        # Σ (v_i - 1)² well scaled when some weights are small.
        target = weights
        relative = constraints._replace(
            bounds=[
                (low / scale, high / scale)
                for (low, high), scale in zip(constraints.bounds, target, strict=True)
            ],
            inequality=constraints.inequality * target,
            equality=constraints.equality * target,
        )
        weights = target * _solve(
            lambda v: np.sum((v - 1) ** 2), lambda v: 2 * (v - 1), relative
        )

    return _as_series(weights, covariance.index)


NESTED_CLUSTERED_OBJECTIVES = ("minimum_variance", "maximum_sharpe", "risk_parity")


def get_nested_clustered_optimization_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    objective: str = "minimum_variance",
    risk_free_rate: float = 0.0,
    linkage_method: str = "ward",
    clusters: int | None = None,
    correlation: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Nested Clustered Optimization (López de Prado, 2019): the assets are clustered on
    their correlations, the objective is optimized within every cluster, and then again
    across the clusters, each treated as one asset with the returns of its portfolio.
    The noise of the covariance matrix mostly lives within clusters of correlated assets,
    so containing each optimization to one level keeps the weights stable.

    With W the N × K matrix whose column k holds the weights within cluster k, the
    clusters have the expected returns W'μ and the covariance matrix W'ΣW, the weights
    across the clusters v are optimal for those, and the weight of every asset is
    w = W v. A cluster of one asset holds that asset only. With "maximum_sharpe", a
    cluster without an asset above the risk free rate holds its minimum variance
    portfolio. Both levels are long only and fully invested, as in Riskfolio-Lib and
    skfolio by default. The paper clusters with k-means and solves both levels without
    bounds, allowing short positions; here the clustering is hierarchical, as in HERC.

    See López de Prado, M. (2019), "A Robust Estimator of the Efficient Frontier", SSRN
    3469961, and López de Prado, M. (2020), Machine Learning for Asset Managers,
    Cambridge University Press, chapter 7.

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        objective (str, optional): The objective at both levels: "minimum_variance",
            "maximum_sharpe" or "risk_parity". Defaults to "minimum_variance".
        risk_free_rate (float, optional): The annualized risk free rate of the maximum
            Sharpe ratio. Defaults to 0.
        linkage_method (str, optional): The linkage of the hierarchical clustering:
            "single", "complete", "average" or "ward". Defaults to "ward".
        clusters (int | None, optional): The number of clusters. Defaults to None, chosen
            by the two-order difference of the gap statistic.
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on, e.g. a rank correlation from get_codependence. Defaults to
            None, the correlation matrix of the covariance matrix.

    Returns:
        pd.Series: The weight per asset.
    """
    if objective not in NESTED_CLUSTERED_OBJECTIVES:
        raise ValueError(
            f"The objective must be one of {', '.join(NESTED_CLUSTERED_OBJECTIVES)}, not "
            f"'{objective}'."
        )
    _validate_covariance(covariance)

    def optimize(mu: pd.Series, sigma: pd.DataFrame) -> pd.Series:
        if len(sigma) == 1:
            return pd.Series(1.0, index=sigma.index)
        if objective == "minimum_variance":
            return get_minimum_variance_weights(sigma)
        if objective == "maximum_sharpe":
            return get_maximum_sharpe_weights(mu, sigma, risk_free_rate)
        return get_risk_parity_weights(sigma)

    if objective == "maximum_sharpe" and (expected_returns <= risk_free_rate).all():
        raise ValueError(
            "No asset has an expected return above the risk free rate, so there is no "
            "maximum Sharpe ratio portfolio."
        )
    _, labels = get_hierarchical_clusters(
        covariance, linkage_method, clusters, correlation
    )
    within = pd.DataFrame(0.0, index=covariance.index, columns=np.unique(labels))
    for label in within.columns:
        assets = covariance.index[labels == label]
        if (
            objective == "maximum_sharpe"
            and (expected_returns[assets] <= risk_free_rate).all()
        ):
            # No asset of the cluster beats the risk free rate, so it has no maximum
            # Sharpe portfolio; it holds its least risky one and the step across the
            # clusters decides how much it gets.
            within.loc[assets, label] = (
                get_minimum_variance_weights(covariance.loc[assets, assets])
                if len(assets) > 1
                else 1.0
            )
            continue
        within.loc[assets, label] = optimize(
            expected_returns[assets], covariance.loc[assets, assets]
        )

    across = optimize(
        within.T @ expected_returns.reindex(covariance.index),
        within.T @ covariance @ within,
    )

    return _as_series((within @ across).to_numpy(), covariance.index)


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
            portfolio. At 0 (market neutral) the lowest CVaR is that of holding nothing,
            so all weights are 0. Defaults to 1.
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


def get_tail_probabilities(
    losses: np.ndarray | pd.Series, confidence_level: float
) -> np.ndarray:
    """
    The weight of every period in the expected tail loss of Rockafellar and Uryasev
    (2002): 1 / ((1 - α)T) for each of the ⌊(1 - α)T⌋ largest losses, the remaining
    fraction of the tail for the next largest and 0 for the others, so that the weights
    sum to one. The expected tail loss is the losses weighted by them; the returns
    weighted by them give each asset's share of it (Euler allocation).

    Args:
        losses (np.ndarray | pd.Series): The losses per period, a loss being positive.
        confidence_level (float): The confidence level α, e.g. 0.95 for the worst 5%.

    Returns:
        np.ndarray: The weight of every period, in the order of the losses.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")
    losses = np.asarray(losses, dtype=float)
    if len(losses) == 0:
        raise ValueError("There are no losses to take the tail of.")
    tail = (1 - confidence_level) * len(losses)
    # A tail of 62.75 periods holds the 62 largest losses in full and three quarters of
    # the 63rd; the stable sort keeps the earliest of tied losses first.
    order = np.argsort(-losses, kind="stable")
    full = min(math.floor(tail), len(losses))
    probabilities = np.zeros(len(losses))
    probabilities[order[:full]] = 1 / tail
    if full < len(losses):
        probabilities[order[full]] = (tail - full) / tail
    return probabilities


def get_expected_tail_loss(
    losses: np.ndarray | pd.Series, confidence_level: float
) -> float:
    """
    The average of the worst 1 - α share of the losses, the minimum over ζ of
    ζ + Σ max(0, L_t - ζ) / ((1 - α)T) (Rockafellar and Uryasev, 2002). It counts a
    fraction of the loss on the boundary when (1 - α)T is not a whole number, and it is
    the value the CVaR and CDaR linear programmes minimize. Every CVaR, CDaR and expected
    shortfall of the portfolio construction uses it.

    Args:
        losses (np.ndarray | pd.Series): The losses per period, a loss being positive.
        confidence_level (float): The confidence level α, e.g. 0.95 for the worst 5%.

    Returns:
        float: The expected tail loss.
    """
    losses = np.asarray(losses, dtype=float)
    return float(get_tail_probabilities(losses, confidence_level) @ losses)


def get_drawdowns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    The drawdown in every period on uncompounded returns (Chekhlov, Uryasev and
    Zabarankin, 2005): the highest cumulative sum of the returns so far, starting from
    zero, minus the current cumulative sum. These are absolute drawdowns of a portfolio
    that keeps its invested amount constant, which keeps the drawdown optimizers linear;
    compounded drawdowns are relative to the peak value.

    Args:
        returns (pd.Series | pd.DataFrame): The returns per period.

    Returns:
        pd.Series | pd.DataFrame: The drawdown per period, a drawdown being positive.
    """
    cumulative = returns.cumsum()
    return cumulative.cummax().clip(lower=0) - cumulative


DRAWDOWN_MEASURES = ("cdar", "maximum", "average")


def get_minimum_drawdown_weights(
    returns: pd.DataFrame,
    measure: str = "cdar",
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
    The portfolio with the lowest drawdown risk over the historical returns, solved as a
    linear programme (Chekhlov, Uryasev and Zabarankin, 2005). With y_t = Σ_(s≤t) r_s'w
    the uncompounded cumulative return and u_t ≥ y_t, u_t ≥ u_(t-1), u_0 = 0 its running
    peak, the drawdown is u_t - y_t and the programme minimizes:

    - "cdar": the Conditional Drawdown at Risk, the average of the worst 1 - α share of
      the drawdowns, ζ + Σ z_t / ((1 - α)T) with z_t ≥ u_t - y_t - ζ and z_t ≥ 0.
    - "maximum": the maximum drawdown m, with u_t - y_t ≤ m.
    - "average": the average drawdown Σ (u_t - y_t) / T.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        measure (str, optional): The drawdown measure: "cdar", "maximum" or "average".
            Defaults to "cdar".
        confidence_level (float, optional): The confidence level of the CDaR, e.g. 0.95 for
            the average of the worst 5% of drawdowns. Defaults to 0.95.
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
    if measure not in DRAWDOWN_MEASURES:
        raise ValueError(
            f"The measure must be one of {', '.join(DRAWDOWN_MEASURES)}, not '{measure}'."
        )
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")

    data = returns.dropna(how="any").to_numpy()
    periods, assets = data.shape
    if periods < 2:  # noqa: PLR2004
        raise ValueError(
            "At least two periods with a return for every asset are needed."
        )
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
    # The variables are the weights (and, with a gross exposure limit and short
    # positions, their absolute values), the running peaks u_t and then, for the CDaR,
    # ζ and the excess drawdowns z_t or, for the maximum drawdown, m.
    variables = len(constraints.bounds)
    cumulative = sparse.hstack(
        [
            sparse.csr_matrix(np.cumsum(data, axis=0)),
            sparse.csr_matrix((periods, variables - assets)),
        ]
    )
    identity = sparse.eye(periods, format="csr")
    column = sparse.csr_matrix(-np.ones((periods, 1)))
    # u_t - u_(t+1) ≤ 0: the peak never falls; u_t ≥ 0 starts it from u_0 = 0.
    rising = sparse.eye(periods - 1, periods) - sparse.eye(periods - 1, periods, k=1)

    if measure == "cdar":
        blocks = [
            [cumulative, -identity, None, None],
            [None, rising, None, None],
            [-cumulative, identity, column, -identity],
        ]
        cost = np.concatenate(
            [
                np.zeros(variables + periods),
                [1.0],
                np.full(periods, 1 / ((1 - confidence_level) * periods)),
            ]
        )
        extra = [(None, None)] + [(0, None)] * periods
    elif measure == "maximum":
        blocks = [
            [cumulative, -identity, None],
            [None, rising, None],
            [-cumulative, identity, column],
        ]
        cost = np.concatenate([np.zeros(variables + periods), [1.0]])
        extra = [(0, None)]
    else:
        blocks = [[cumulative, -identity], [None, rising]]
        cost = (
            np.concatenate(
                [-np.asarray(cumulative.sum(axis=0)).ravel(), np.ones(periods)]
            )
            / periods
        )
        extra = []

    width = len(cost)
    inequality = sparse.bmat(blocks, format="csr")
    if len(constraints.inequality):
        inequality = sparse.vstack(
            [
                inequality,
                sparse.hstack(
                    [
                        sparse.csr_matrix(constraints.inequality),
                        sparse.csr_matrix(
                            (len(constraints.inequality), width - variables)
                        ),
                    ]
                ),
            ],
            format="csr",
        )

    result = linprog(
        cost,
        A_ub=inequality,
        b_ub=np.concatenate(
            [np.zeros(inequality.shape[0] - len(constraints.upper)), constraints.upper]
        ),
        A_eq=np.hstack([constraints.equality, np.zeros((1, width - variables))]),
        b_eq=constraints.target,
        bounds=constraints.bounds + [(0, None)] * periods + extra,
        method="highs",
    )

    if not result.success:
        raise ValueError(f"The optimization did not converge: {result.message}")
    if np.allclose(result.x[:assets], 0, atol=ZERO_WEIGHT):
        raise ValueError(
            "Holding nothing has no drawdown, so the minimum is the empty portfolio. This "
            "happens with a net exposure of 0; use a net exposure other than 0."
        )

    return _as_series(result.x[:assets], returns.columns)


def get_maximum_omega_weights(
    returns: pd.DataFrame,
    minimum_acceptable_return: float = 0.0,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
    expected_returns: pd.Series | None = None,
) -> pd.Series:
    """
    The portfolio with the highest Omega ratio over the historical returns,
    Ω(w) = 1 + (r̄'w - τ) / E[max(0, τ - r_t'w)] with τ the minimum acceptable return
    (Keating and Shadwick, 2002). The ratio of a linear function to a convex
    piecewise-linear one is maximized as a linear programme with the Charnes-Cooper
    transformation y = κw, κ > 0 (Kapsos, Christofides and Rustem, 2014):

    maximize r̄'y - τκ subject to Σ d_t / T = 1, d_t ≥ τκ - r_t'y, d_t ≥ 0 and the
    weight constraints in homogeneous form; the weights are w = y / κ.

    See Kapsos, M., Christofides, N. and Rustem, B. (2014), "Worst-Case Robust Omega
    Ratio", European Journal of Operational Research, 234(2), 499-507, and Mausser, H.,
    Saunders, D. and Seco, L. (2006), "Optimising Omega", Risk, 19(11), 88-92.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        minimum_acceptable_return (float, optional): The threshold τ between gains and
            losses, a return per period. Defaults to 0.
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
        expected_returns (pd.Series | None, optional): The expected return per period of
            every asset in the numerator, e.g. from a shrinkage estimator; the losses in
            the denominator stay those of the historical returns. Defaults to None, the
            average of the returns.

    Returns:
        pd.Series: The weight per asset.
    """
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
    mean = (
        data.mean(axis=0)
        if expected_returns is None
        else expected_returns.reindex(returns.columns).to_numpy(dtype=float)
    )
    if np.isnan(mean).any():
        raise ValueError("The expected returns must cover every asset.")
    # The transformation needs a positive numerator: with κ = 0 every ratio below zero
    # would be traded for the value 0 of y = 0.
    if _return_range(mean, constraints)[1] <= minimum_acceptable_return:
        raise ValueError(
            "No portfolio within the constraints has an average return above the minimum "
            "acceptable return, so none has an Omega ratio above one."
        )

    # The variables are y (the scaled weights and absolute weights), κ and d_t.
    variables = len(constraints.bounds)
    homogeneous, equality = _homogeneous_constraints(constraints)
    inequality = sparse.vstack(
        [
            sparse.hstack(
                [
                    sparse.csr_matrix(-data),
                    sparse.csr_matrix((periods, variables - assets)),
                    sparse.csr_matrix(np.full((periods, 1), minimum_acceptable_return)),
                    -sparse.eye(periods, format="csr"),
                ]
            ),
            sparse.hstack(
                [
                    sparse.csr_matrix(homogeneous),
                    sparse.csr_matrix((len(homogeneous), periods)),
                ]
            ),
        ],
        format="csr",
    )
    result = linprog(
        np.concatenate(
            [
                -mean,
                np.zeros(variables - assets),
                [minimum_acceptable_return],
                np.zeros(periods),
            ]
        ),
        A_ub=inequality,
        b_ub=np.zeros(inequality.shape[0]),
        A_eq=np.vstack(
            [
                np.hstack([equality, np.zeros((1, periods))]),
                np.concatenate(
                    [np.zeros(variables + 1), np.full(periods, 1 / periods)]
                ),
            ]
        ),
        b_eq=[0.0, 1.0],
        bounds=[(None, None)] * variables + [(0, None)] * (1 + periods),
        method="highs",
    )

    if not result.success:
        raise ValueError(
            f"The optimization did not converge: {result.message} A portfolio that never "
            "returns less than the minimum acceptable return has an unbounded Omega ratio."
            if result.status == 3  # noqa: PLR2004
            else f"The optimization did not converge: {result.message}"
        )

    return _as_series(result.x[:assets] / result.x[variables], returns.columns)


def get_maximum_cvar_ratio_weights(
    returns: pd.DataFrame,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.0,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    net_exposure: float = 1.0,
    gross_exposure: float | None = None,
    current_weights: pd.Series | None = None,
    maximum_turnover: float | None = None,
    expected_returns: pd.Series | None = None,
) -> pd.Series:
    """
    The portfolio with the highest ratio of expected excess return to Conditional Value
    at Risk over the historical returns, (r̄ - r_f)'w / CVaR_α(w), the STARR ratio of
    Martin, Rachev and Siboulet (2003). With the CVaR as the linear programme of
    Rockafellar and Uryasev (2000) and the Charnes-Cooper transformation y = κw, κ > 0,
    it is the linear programme:

    maximize (r̄ - r_f)'y subject to ζ + Σ s_t / ((1 - α)T) ≤ 1, s_t ≥ -r_t'y - ζ,
    s_t ≥ 0 and the weight constraints in homogeneous form; the weights are w = y / κ.

    See Martin, R. D., Rachev, S. T. and Siboulet, F. (2003), "Phi-Alpha Optimal
    Portfolios and Extreme Risk Management", Wilmott Magazine, November, 70-83.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        confidence_level (float, optional): The confidence level of the CVaR, e.g. 0.95 for
            the average of the worst 5% of periods. Defaults to 0.95.
        risk_free_rate (float, optional): The risk free rate per period. Defaults to 0.
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
        expected_returns (pd.Series | None, optional): The expected return per period of
            every asset in the numerator, e.g. from a shrinkage estimator; the losses in
            the denominator stay those of the historical returns. Defaults to None, the
            average of the returns.

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
    excess = (
        data.mean(axis=0)
        if expected_returns is None
        else expected_returns.reindex(returns.columns).to_numpy(dtype=float)
    ) - risk_free_rate
    if np.isnan(excess).any():
        raise ValueError("The expected returns must cover every asset.")
    # The transformation needs a positive numerator: with κ = 0 every ratio below zero
    # would be traded for the value 0 of y = 0.
    if _return_range(excess, constraints)[1] <= 0:
        raise ValueError(
            "No portfolio within the constraints has an average return above the risk "
            "free rate, so none has a positive CVaR ratio."
        )

    # The variables are y (the scaled weights and absolute weights), κ, ζ and s_t.
    variables = len(constraints.bounds)
    homogeneous, equality = _homogeneous_constraints(constraints)
    inequality = sparse.vstack(
        [
            sparse.hstack(
                [
                    sparse.csr_matrix(-data),
                    sparse.csr_matrix((periods, variables - assets + 1)),
                    sparse.csr_matrix(-np.ones((periods, 1))),
                    -sparse.eye(periods, format="csr"),
                ]
            ),
            sparse.csr_matrix(
                np.concatenate(
                    [
                        np.zeros(variables + 1),
                        [1.0],
                        np.full(periods, 1 / ((1 - confidence_level) * periods)),
                    ]
                )[None, :]
            ),
            sparse.hstack(
                [
                    sparse.csr_matrix(homogeneous),
                    sparse.csr_matrix((len(homogeneous), 1 + periods)),
                ]
            ),
        ],
        format="csr",
    )
    b_ub = np.zeros(inequality.shape[0])
    b_ub[periods] = 1.0
    result = linprog(
        np.concatenate([-excess, np.zeros(variables - assets + 2 + periods)]),
        A_ub=inequality,
        b_ub=b_ub,
        A_eq=np.hstack([equality, np.zeros((1, 1 + periods))]),
        b_eq=[0.0],
        bounds=[(None, None)] * variables
        + [(0, None), (None, None)]
        + [(0, None)] * periods,
        method="highs",
    )

    if not result.success:
        raise ValueError(
            f"The optimization did not converge: {result.message} A portfolio whose "
            "worst returns are gains has no tail risk and an unbounded CVaR ratio."
            if result.status == 3  # noqa: PLR2004
            else f"The optimization did not converge: {result.message}"
        )

    return _as_series(result.x[:assets] / result.x[variables], returns.columns)


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

    weights = _solve(negative_ratio, gradient, constraints, direction=volatility)

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


RESAMPLED_OBJECTIVES = ("maximum_sharpe", "minimum_variance", "mean_variance")
RESAMPLING_METHODS = ("parametric", "bootstrap")


def get_resampled_efficient_weights(
    returns: pd.DataFrame,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    periods_per_year: int,
    objective: str = "maximum_sharpe",
    resamples: int = 500,
    resampling: str = "parametric",
    seed: int | None = 0,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 2.5,
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
    The weights of the resampled efficient portfolio (Michaud, 1998): the optimal weights
    for many return histories that could have been drawn instead of the one observed,
    whose average is the resampled portfolio. Each draw has as many periods as the
    returns and comes from a normal distribution with the expected returns and
    covariance ("parametric", as in Michaud), or is drawn with replacement from the
    periods of the returns ("bootstrap", which keeps their fat tails and skew). The
    expected returns and covariance of every draw are its sample mean and sample
    covariance (ddof=1), annualized, and the objective is optimized on them within the
    constraints.

    The average of the draws hedges against the estimation error a single optimization
    takes at face value: an asset only gets a large weight when it does so in most of
    the histories the data could have produced. As every constraint is linear (or, for
    the gross exposure and turnover, convex), the average of weights that meet them
    meets them too.

    See Michaud, R. O. (1998), "Efficient Asset Management: A Practical Guide to Stock
    Portfolio Optimization and Asset Allocation", Harvard Business School Press, and
    Scherer, B. (2002), "Portfolio Resampling: Review and Critique", Financial Analysts
    Journal, 58(6), 98-109.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset; their
            number of periods sets that of every draw and, for "bootstrap", the draws
            come from them.
        expected_returns (pd.Series): The annualized expected returns the "parametric"
            draws are centred on.
        covariance (pd.DataFrame): The annualized covariance matrix of the "parametric"
            draws.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily
            returns.
        objective (str, optional): The portfolio of every draw: "maximum_sharpe",
            "minimum_variance" or "mean_variance" (with the risk aversion). Defaults to
            "maximum_sharpe".
        resamples (int, optional): The number of draws. Defaults to 500.
        resampling (str, optional): How the draws are made: "parametric" or
            "bootstrap". Defaults to "parametric".
        seed (int | None, optional): The seed of the random draws, so that the same seed
            gives the same weights. Defaults to 0; None draws anew every time.
        risk_free_rate (float, optional): The annualized risk free rate of the maximum
            Sharpe ratio. Defaults to 0.
        risk_aversion (float, optional): The risk aversion of "mean_variance".
            Defaults to 2.5.
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
        pd.DataFrame: The weights of every draw that has a solution, one row per draw
            (numbered from 0) and one column per asset; their average is the resampled
            efficient portfolio. A draw without a solution, e.g. one in which no asset
            beats the risk free rate for the maximum Sharpe ratio, is left out.
    """
    if objective not in RESAMPLED_OBJECTIVES:
        raise ValueError(
            f"The objective must be one of {', '.join(RESAMPLED_OBJECTIVES)}, not "
            f"'{objective}'."
        )
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(
            f"The resampling must be one of {', '.join(RESAMPLING_METHODS)}, not "
            f"'{resampling}'."
        )
    if resamples < 1:
        raise ValueError("At least one resample is needed.")
    data = returns.dropna(how="any")
    periods, assets = data.shape
    if periods < assets + 1:
        raise ValueError(
            f"{periods} periods are too few to estimate the covariance of {assets} "
            "assets in every draw."
        )
    columns = data.columns
    mean = expected_returns.reindex(columns).to_numpy(dtype=float) / periods_per_year
    sigma = (
        covariance.reindex(index=columns, columns=columns).to_numpy(dtype=float)
        / periods_per_year
    )
    if np.isnan(mean).any() or np.isnan(sigma).any():
        raise ValueError("The expected returns and covariance must cover every asset.")

    # The constraints are checked once, so that infeasible ones fail with their own
    # message instead of leaving every draw without a solution.
    constraints = {
        "minimum_weight": minimum_weight,
        "maximum_weight": maximum_weight,
        "groups": groups,
        "group_bounds": group_bounds,
        "net_exposure": net_exposure,
        "gross_exposure": gross_exposure,
        "current_weights": current_weights,
        "maximum_turnover": maximum_turnover,
    }
    get_weight_constraints(columns, **constraints)

    # The normal draws are mean + z F' with F = U √Λ from the eigendecomposition of the
    # covariance, decomposed once rather than per draw, which gives the draws of
    # numpy's multivariate_normal(method="eigh") for the same seed.
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))
    generator = np.random.default_rng(seed)
    history = data.to_numpy()
    draws = {}
    for draw in range(resamples):
        sample = (
            mean + generator.standard_normal((periods, assets)) @ factor.T
            if resampling == "parametric"
            else history[generator.integers(0, periods, size=periods)]
        )
        centred = sample - sample.mean(axis=0)
        mu = pd.Series(sample.mean(axis=0) * periods_per_year, index=columns)
        cov = pd.DataFrame(
            centred.T @ centred / (periods - 1) * periods_per_year,
            index=columns,
            columns=columns,
        )
        try:
            if objective == "maximum_sharpe":
                weights = get_maximum_sharpe_weights(
                    mu, cov, risk_free_rate, **constraints
                )
            elif objective == "minimum_variance":
                weights = get_minimum_variance_weights(cov, **constraints)
            else:
                weights = get_mean_variance_weights(
                    mu, cov, risk_aversion=risk_aversion, **constraints
                )
        except ValueError:
            continue
        draws[draw] = weights

    if not draws:
        raise ValueError(
            f"None of the {resamples} draws has a {objective.replace('_', ' ')} "
            "portfolio within the constraints."
        )
    return pd.DataFrame(draws).T


def get_black_litterman_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    views: dict[str, float] | None = None,
    view_confidences: dict[str, float] | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The Black-Litterman expected returns (Black and Litterman, 1992), as total returns,
    and their posterior covariance matrix.
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

    The posterior covariance matrix is Σ + M (He and Litterman, 1999): the covariance of
    the returns plus M = ((τΣ)⁻¹ + P'Ω⁻¹P)⁻¹, the uncertainty that remains in the
    Black-Litterman returns, computed as τΣ - τΣP'(PτΣP' + Ω)⁻¹PτΣ so that neither Σ nor
    Ω is inverted. Without views M = τΣ and the posterior covariance is (1 + τ) Σ; every
    view lowers M in its direction, a certain view to zero.

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
        tuple[pd.DataFrame, pd.DataFrame]: The Implied Return and the Black-Litterman
            Return per asset, both annualized total returns, and the annualized posterior
            covariance matrix Σ + M.
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
    tau_sigma = tau * sigma
    uncertainty_left = tau_sigma

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
        try:
            solved = np.linalg.solve(
                p @ tau_sigma @ p.T + omega,
                np.column_stack([q - p @ implied, p @ tau_sigma]),
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "The views contradict or repeat each other, so they cannot be combined."
            ) from error
        posterior = implied + tau_sigma @ p.T @ solved[:, 0]
        uncertainty_left = tau_sigma - tau_sigma @ p.T @ solved[:, 1:]

    return (
        pd.DataFrame(
            {"Implied Return": implied, "Black-Litterman Return": posterior},
            index=assets,
        ),
        pd.DataFrame(sigma + uncertainty_left, index=assets, columns=assets),
    )


def get_market_implied_risk_aversion(
    benchmark_returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """
    The risk aversion of the market, the excess return it earns per unit of variance:
    δ = (E[r_m] - r_f) / σ²_m. A representative investor who holds the market is
    indifferent to the next unit of risk at exactly this δ, which makes it the risk
    aversion that the market implied returns Π = δΣw_mkt of Black and Litterman
    (1992) are consistent with.

    E[r_m] is the average return per period of the benchmark times the number of periods
    in a year (arithmetic, not compounded) and σ²_m its variance per period (ddof=1)
    times the same number, so δ is the same at every frequency up to estimation error.

    See Black, F. and Litterman, R. (1992), "Global Portfolio Optimization", Financial
    Analysts Journal, 48(5), 28-43, and Walters, J. (2014), "The Black-Litterman Model in
    Detail", https://ssrn.com/abstract=1314585.

    Args:
        benchmark_returns (pd.Series): The returns per period of the benchmark.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.
        risk_free_rate (float, optional): The annualized risk free rate. Defaults to 0.

    Returns:
        float: The risk aversion δ of the market.
    """
    market = benchmark_returns.dropna()
    if len(market) < 2:  # noqa: PLR2004
        raise ValueError(
            "The market risk aversion needs at least two benchmark returns."
        )
    variance = float(market.var()) * periods_per_year
    if variance <= 0:
        raise ValueError("The benchmark's returns have no variance.")
    risk_aversion = (
        float(market.mean()) * periods_per_year - risk_free_rate
    ) / variance
    if risk_aversion <= 0:
        raise ValueError(
            f"The benchmark earned {float(market.mean()) * periods_per_year:.4f} a year, "
            f"no more than the risk free rate of {risk_free_rate:.4f}, so it implies no "
            "positive risk aversion; pass a number instead."
        )
    return risk_aversion


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
    (Euler allocation). The expected shortfall is the average loss in the worst
    1 - α share of the periods, ES = -Σ p_t r_p,t, with p_t the tail probabilities of
    Rockafellar and Uryasev (2002): 1 / ((1 - α)T) for each of the ⌊(1 - α)T⌋ worst
    periods of the portfolio and the remaining fraction of the tail for the next one.
    Because it is homogeneous of degree one in the weights, its derivative per asset,
    ∂ES/∂w_i = -Σ p_t r_i,t, times the weight gives contributions that add up to the
    expected shortfall (Tasche, 2000; Scaillet, 2004). The tail periods are those of the
    portfolio, so an asset that tends to gain when the portfolio loses contributes less
    than its own expected shortfall, or a negative amount when it hedges.

    The expected shortfall is the one the minimum CVaR and maximum CVaR ratio
    optimizations minimize and report, the average of the tail with a fraction of the
    boundary period when (1 - α)T is not a whole number (Riskfolio-Lib's CVaR).

    See Tasche, D. (2000), "Conditional Expectations as Quantile Derivatives", arXiv
    math/0104190, and Scaillet, O. (2004), "Nonparametric Estimation and Sensitivity
    Analysis of Expected Shortfall", Mathematical Finance, 14(1), 115-129.

    Args:
        weights (pd.Series): The weight per asset.
        returns (pd.DataFrame): The returns per period, one column per asset.
        confidence_level (float, optional): The confidence level, e.g. 0.95 for the
            worst 5% of periods. Defaults to 0.95.

    Returns:
        pd.DataFrame: The Weight, Marginal Contribution (-Σ p_t r_i,t), Risk
        Contribution (the weight times the marginal contribution) and Percentage
        Contribution per asset, as losses per period.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("No period has a return for every asset.")
    weights = weights.reindex(returns.columns).fillna(0)
    probabilities = get_tail_probabilities(-(returns @ weights), confidence_level)

    marginal = -(probabilities @ returns.to_numpy())
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
