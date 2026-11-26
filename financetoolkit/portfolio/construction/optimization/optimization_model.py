"""Portfolio Optimization Model"""

__docformat__ = "google"

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.cluster.hierarchy import (
    cut_tree,
    fcluster,
    leaves_list,
    linkage,
    optimal_leaf_ordering,
    to_tree,
)
from scipy.linalg import null_space
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize
from scipy.spatial.distance import pdist, squareform

from financetoolkit.performance import performance_model
from financetoolkit.portfolio.construction.optimization import estimation_model
from financetoolkit.portfolio.helpers import (
    get_cutting_plane_solution,
    get_regularized_covariance,
    get_risk_budget,
    get_view_vector,
)
from financetoolkit.risk import risk_model
from financetoolkit.risk.distribution import dispersion_model
from financetoolkit.risk.drawdown import drawdown_model
from financetoolkit.risk.value_at_risk import cvar_model, evar_model, var_model
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    validate_alpha,
)

# pylint: disable=too-many-arguments,too-many-locals

# Tolerances for the numerical optimizers: tight enough that the weights are stable to the
# fourth decimal, which is the default rounding of the Finance Toolkit. The ftol is absolute,
# so objectives in variances are expressed in units of the average variance of the assets.
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
    The constraints on the weights as a linear system over the variables x.

    The variables are the weights w, followed by one variable t_i ≥ |w_i| per asset when the
    gross exposure is limited and short positions are allowed, and by one variable d_i ≥
    |w_i - w0_i| per asset when there are current weights w0 to trade from. Every optimizer
    solves over x and keeps its first `assets` entries.
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
    The constraints on the weights of a portfolio as a linear system.

    They are checked for feasibility so that an impossible combination fails with a clear
    message before any optimization runs.

    - Per asset, minimum_weight ≤ w_i ≤ maximum_weight; a bound given as {asset: bound}
      applies to the assets it names, the others keep 0 (minimum) or 1 (maximum).
    - Per group (e.g. a sector or country), the weights of its assets sum to between the
      group's minimum and maximum, Σ_{i ∈ g} w_i.
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
    The Ledoit-Wolf shrinkage estimator of the covariance matrix.

    It is the sample covariance pulled towards the identity matrix scaled by the average
    variance, with the shrinkage intensity that minimizes the expected squared (Frobenius)
    error. This keeps the matrix well-conditioned when there are many assets compared to the
    number of periods, which is where the sample covariance makes optimizers chase
    estimation noise.

    It shrinks the maximum likelihood sample covariance, divided by the number of periods
    rather than one less.

    See Ledoit, O. and Wolf, M. (2004), "A well-conditioned estimator for large-dimensional
    covariance matrices", Journal of Multivariate Analysis, 88(2), 365-411.

    Referenced against:
    - scikit-learn, LedoitWolf: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html

    Shares similar logic:
    - PyPortfolioOpt, CovarianceShrinkage.ledoit_wolf: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf

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
    The Oracle Approximating Shrinkage (OAS) estimator of the covariance matrix.

    Like Ledoit-Wolf, the sample covariance shrunk towards the identity matrix scaled by the
    average variance, but with an intensity derived for normally distributed returns, which
    converges faster and shrinks more when there are few periods. It shrinks the maximum
    likelihood sample covariance, divided by the number of periods rather than one less.

    See Chen, Y., Wiesel, A., Eldar, Y. C. and Hero, A. O. (2010), "Shrinkage Algorithms
    for MMSE Covariance Estimation", IEEE Transactions on Signal Processing, 58(10),
    5016-5029.

    Referenced against:
    - scikit-learn, OAS: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html

    Shares similar logic:
    - PyPortfolioOpt, CovarianceShrinkage.oracle_approximating: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.oracle_approximating

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
    correlation target.

    It is the sample covariance S pulled towards the matrix F that keeps every sample
    variance and sets every correlation to the average sample correlation, f_ii = s_ii and
    f_ij = r̄ √(s_ii s_jj) with r̄ = 2 / (N (N - 1)) Σ_(i<j) r_ij. Unlike the scaled
    identity, the target keeps each asset's own variance and the general level of
    co-movement between assets, so it shrinks only what is noisy: the differences between
    the correlations.

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

    Shares similar logic:
    - PyPortfolioOpt, CovarianceShrinkage.ledoit_wolf: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf

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
    target.

    It is the sample covariance S pulled towards the covariance matrix F that the market
    model implies, with the benchmark as the market. Each asset's return is β_i r_m plus a
    residual that is unrelated to the other assets, so f_ij = s_im s_jm / s_mm off the
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

    Shares similar logic:
    - PyPortfolioOpt, CovarianceShrinkage.ledoit_wolf: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        benchmark_returns (pd.Series): The returns per period of the benchmark, the
            factor of the target.

    Returns:
        tuple[pd.DataFrame, float]: The covariance matrix per period and the shrinkage
            intensity between 0 (the sample covariance) and 1 (the single factor target).

    Notes:
    - Unlike PyPortfolioOpt, whose single factor is the equal-weighted average of the
      assets, the factor is the benchmark.
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
    The semicovariance matrix: the co-movement of the returns below a threshold, each return
    above it counting as zero.

    It measures only the downside risk, which is what an investor who doesn't mind upside
    surprises wants to minimize.

    Every period counts in the average, the ones above the threshold as zero, as in
    Estrada.

    See Estrada, J. (2008), "Mean-Semivariance Optimization: A Heuristic Approach",
    Journal of Applied Finance, 18(1), 57-72.

    Referenced against:
    - PyPortfolioOpt, semicovariance: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.semicovariance

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
    The exponentially weighted covariance matrix.

    Its entries are the products of the deviations from the mean return weighted by (1 −
    α)^age with α = 2 / (span + 1), so that recent periods count most and the matrix follows
    changes in volatility and correlation. The deviations are taken from the mean over the
    whole range and the weights are normalized to sum to one, as PyPortfolioOpt's exp_cov
    does. The weighting is the one of RiskMetrics with decay factor λ = 1 − α, but the
    matrix differs from the RiskMetrics estimate in two ways: RiskMetrics assumes a mean
    return of zero rather than demeaning by the sample mean, and its daily decay factor λ =
    0.94 is a span of about 32 periods, while the default span of 180 is λ ≈ 0.989 (a
    half-life of about 62 periods). A span of 32 gives the RiskMetrics daily weighting.

    See RiskMetrics Technical Document (J.P. Morgan/Reuters, 1996), chapter 5, for the
    exponential weighting.

    The formula is as follows:

    - α = 2 / (span + 1), λ = 1 − α
    - w_t = λ^(T − t) / Σ λ^(T − s), for the periods t = 1, ..., T
    - Covariance(i, j) = Σ w_t × (r_i,t − mean(r_i)) × (r_j,t − mean(r_j))

    Referenced against:
    - PyPortfolioOpt, exp_cov: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.exp_cov

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        span (int, optional): The span of the weights in periods; the weight halves
            after about span / 2.9 periods. Defaults to 180 (λ ≈ 0.989).

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


DENOISE_METHODS = ("fixed", "spectral", "shrink")


def get_marcenko_pastur_fit(
    eigenvalues: np.ndarray, ratio: float
) -> tuple[float, float, float]:
    """
    The Marčenko-Pastur distribution that fits the eigenvalues of a correlation matrix best.

    It separates the eigenvalues that random returns alone could produce from those that
    carry signal. The eigenvalues of the correlation matrix of T periods of N unrelated
    returns with variance σ² follow, as T and N grow with T / N = q, the density

    - f(λ) = q / (2π σ² λ) √((λ+ - λ)(λ - λ-)) for λ- ≤ λ ≤ λ+
    - λ± = σ² (1 ± √(1 / q))²

    The variance σ² is the one whose density is closest to a Gaussian kernel density of
    the eigenvalues (bandwidth 0.01) in the sum of squared differences at 1,000 points
    between the edges λ- and λ+, minimized with SciPy's L-BFGS-B for σ² between 0.00001
    and 0.99999 from 0.5 (López de Prado, 2020, section 2.4, which fits σ² this way). The
    eigenvalues above λ+ are the signal; those below it could come from noise alone.

    See Marčenko, V. A. and Pastur, L. A. (1967), "Distribution of Eigenvalues for Some
    Sets of Random Matrices", Mathematics of the USSR-Sbornik, 1(4), 457-483, and López de
    Prado, M. (2020), Machine Learning for Asset Managers, Cambridge University Press,
    chapter 2.

    Shares similar logic:
    - RiskLabAI, find_max_eval: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/denoising.py#L138
    - RiskLabAI, marcenko_pastur_pdf: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/denoising.py#L26

    Args:
        eigenvalues (np.ndarray): The eigenvalues of the correlation matrix.
        ratio (float): The number of periods over the number of assets, T / N.

    Returns:
        tuple[float, float, float]: The variance σ² of the noise and the lower and upper
            edges λ- and λ+ of its eigenvalues.

    Notes:
    - RiskLabAI fits σ² the same way with scikit-learn's KernelDensity, but evaluates the
      densities at as many points as there are eigenvalues rather than at the book's 1,000
      points, which moves σ² by a few percent when there are few.
    """

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

    variance = float(minimize(density_error, x0=0.5, bounds=((1e-5, 1 - 1e-5),)).x[0])
    return (
        variance,
        variance * (1 - (1 / ratio) ** 0.5) ** 2,
        variance * (1 + (1 / ratio) ** 0.5) ** 2,
    )


def get_denoised_covariance(
    returns: pd.DataFrame, method: str = "fixed", alpha: float = 0.0
) -> tuple[pd.DataFrame, int]:
    """
    The denoised covariance matrix: the sample covariance with the eigenvalues of its
    correlation matrix that random returns could produce replaced by their average.

    Under random matrix theory, the eigenvalues of the correlation matrix of T periods of N
    unrelated returns fall below the Marčenko-Pastur upper edge λ+ = σ²(1 + √(N/T))², so the
    eigenvalues below it carry noise rather than structure. Averaging them keeps the trace
    (the total variance) and the eigenvectors of the signal, and removes the small
    eigenvalues that make the inverse of the matrix, and with it optimized weights, chase
    estimation error.

    The variance σ² of the noise is the one whose Marčenko-Pastur density fits the
    eigenvalues best: the sum of squared differences with a Gaussian kernel density of
    the eigenvalues (bandwidth 0.01), at 1,000 points between the edges of the
    Marčenko-Pastur distribution, minimized for σ² between 0.00001 and 0.99999 starting
    at 0.5. The noise eigenvalues are then treated by one of the methods of López de
    Prado (2020):

    - The constant residual eigenvalue method: the noise eigenvalues are replaced by their
      average, which keeps the trace.
    - The spectral method: the noise eigenvalues are set to zero, so only the signal
      remains.
    - Targeted shrinkage: the noise part of the correlation matrix shrunk towards its own
      diagonal, C = C_signal + α C_noise + (1 - α) diag(C_noise), with C_signal and C_noise
      the parts of the signal and noise eigenvalues; α = 0 keeps only the diagonal of the
      noise, α = 1 the sample correlation.

    The correlation matrix is rescaled to a unit diagonal after the replacement and turned
    back into a covariance with the sample standard deviations (ddof=1).

    See Laloux, L., Cizeau, P., Potters, M. and Bouchaud, J.-P. (2000), "Random Matrix
    Theory and Financial Correlations", International Journal of Theoretical and Applied
    Finance, 3(3), 391-397, and López de Prado, M. (2020), "Machine Learning for Asset Managers", Cambridge
    University Press, chapter 2.

    Referenced against:
    - skfolio, DenoiseCovariance (constant residual eigenvalue): https://skfolio.org/generated/skfolio.moments.DenoiseCovariance.html
    - Riskfolio-Lib, denoiseCov (spectral and targeted shrinkage): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.denoiseCov

    Shares similar logic:
    - RiskLabAI, denoise_cov: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/denoising.py#L376

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): How the noise eigenvalues are treated. Options are: "fixed" (the
            noise eigenvalues replaced by their average), "spectral" (the noise eigenvalues set to
            zero), "shrink" (targeted shrinkage towards the diagonal of the noise). Defaults to
            "fixed".
        alpha (float, optional): The share α of the noise correlations that targeted shrinkage
            keeps, between 0 and 1. Defaults to 0.

    Returns:
        tuple[pd.DataFrame, int]: The covariance matrix per period and the number of
            eigenvalues kept as signal (those above the Marčenko-Pastur upper edge).

    Notes:
    - Riskfolio-Lib's covar_matrix defaults to α = 0.1 for targeted shrinkage; the default
      here is López de Prado's α = 0.
    """
    check_option("method", method, DENOISE_METHODS)
    if not 0 <= alpha <= 1:
        raise ValueError(
            "The share of the noise correlations kept must lie between 0 and 1."
        )
    sample = returns.cov().to_numpy(dtype=float)
    periods, assets = returns.shape
    ratio = periods / assets
    volatility = np.sqrt(np.diag(sample))
    correlation = sample / np.outer(volatility, volatility)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    upper_edge = get_marcenko_pastur_fit(eigenvalues, ratio)[2]
    signals = int(assets - np.searchsorted(eigenvalues[::-1], upper_edge))

    if method == "spectral" and signals == 0:
        raise ValueError(
            "No eigenvalue lies above the Marčenko-Pastur upper edge, so the spectral "
            "denoising leaves no correlation."
        )
    if signals < assets:
        if method == "shrink":
            signal = eigenvectors[:, :signals]
            noise = eigenvectors[:, signals:]
            noise_correlation = noise @ np.diag(eigenvalues[signals:]) @ noise.T
            correlation = (
                signal @ np.diag(eigenvalues[:signals]) @ signal.T
                + alpha * noise_correlation
                + (1 - alpha) * np.diag(np.diag(noise_correlation))
            )
        else:
            eigenvalues = eigenvalues.copy()
            eigenvalues[signals:] = (
                eigenvalues[signals:].mean() if method == "fixed" else 0.0
            )
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
    "shrunk",
    "semicovariance",
    "exponential",
    "denoised",
    "denoised_spectral",
    "denoised_shrink",
    "gerber",
    "factor",
    "minimum_covariance_determinant",
    "jlogo",
    "geodesic",
    "regime_adjusted_ew",
    "nercome",
    "ccc_garch",
    "mm",
    "rocke",
)

EXPECTED_RETURN_METHODS = (
    "mean",
    "geometric",
    "exponential",
    "exponential_geometric",
    "capm",
    "bayes_stein",
    "james_stein",
    "bodnar_okhrin_parolya",
    "centroid",
)


def get_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    span: int = 180,
    benchmark_returns: pd.Series | None = None,
    shrinkage: float = 0.1,
    gerber_threshold: float = 0.5,
    gerber_statistic: int = 1,
    denoise_alpha: float = 0.0,
    factors: pd.DataFrame | None = None,
    half_life: float = 40,
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    The covariance matrix per period of a set of returns, by one of the estimators, which
    shape how much the weights of an optimizer chase estimation noise:

    - The sample covariance (ddof=1).
    - The Ledoit-Wolf and Oracle Approximating Shrinkage estimators: shrinkage with an
      estimated intensity towards the scaled identity, the constant correlation matrix or
      the single factor (benchmark) covariance.
    - Shrinkage with a fixed intensity towards the scaled identity (see
      estimation_model.get_shrunk_covariance).
    - The semicovariance: the co-movement below a zero return.
    - The exponentially weighted covariance: recent periods weigh most.
    - The denoised covariances: the Marčenko-Pastur noise eigenvalues of the correlation
      matrix averaged, set to zero or shrunk towards the diagonal (see
      get_denoised_covariance).
    - The Gerber covariance (see estimation_model.get_gerber_covariance).
    - The factor covariance: the covariance a time series factor model implies, B Σ_f B' +
      D, with the OLS loadings on every factor, estimated on the periods both the returns
      and the factors have (see estimation_model.get_factor_model).
    - The Minimum Covariance Determinant: the reweighted covariance of the periods with the
      smallest covariance determinant, robust to outliers (see
      estimation_model.get_minimum_covariance_determinant).
    - The j-LoGo covariance: the inverse of the sparse LoGo precision matrix on the TMFG of
      the correlations (see estimation_model.get_logo_covariance).
    - Geodesic shrinkage: shrinkage with a fixed intensity towards the scaled identity along
      the geodesic of the positive definite matrices (see
      estimation_model.get_geodesic_shrinkage_covariance).
    - The regime adjusted exponentially weighted covariance: rescaled by the recent ratio of
      realized to predicted risk (see
      estimation_model.get_regime_adjusted_exponential_covariance).
    - NERCOME: the eigenvectors of one part of the periods with the variances along them of
      the other, averaged over random splits (see estimation_model.get_nercome_covariance).
    - The CCC-GARCH covariance: the next-period covariance of the constant conditional
      correlation GARCH(1, 1) model (see estimation_model.get_ccc_garch_covariance).
    - The robust MM-estimator and Rocke's S-estimator (see
      estimation_model.get_robust_covariance).

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): The estimator. Options are: "sample" (sample covariance),
            "ledoit_wolf" (Ledoit-Wolf), "ledoit_wolf_constant_correlation" (Ledoit-Wolf constant
            correlation), "ledoit_wolf_single_factor" (Ledoit-Wolf single factor), "oas" (Oracle
            Approximating Shrinkage), "shrunk" (fixed shrinkage), "semicovariance" (semicovariance),
            "exponential" (exponentially weighted), "denoised" (denoised, constant residual
            eigenvalue), "denoised_spectral" (denoised, spectral), "denoised_shrink" (denoised,
            targeted shrinkage), "gerber" (Gerber), "factor" (factor model),
            "minimum_covariance_determinant" (Minimum Covariance Determinant), "jlogo" (j-LoGo),
            "geodesic" (geodesic shrinkage), "regime_adjusted_ew" (regime adjusted exponentially
            weighted), "nercome" (NERCOME), "ccc_garch" (CCC-GARCH), "mm" (MM-estimator), "rocke"
            (Rocke's S-estimator). Defaults to "sample".
        span (int, optional): The span in periods of the "exponential" covariance.
            Defaults to 180.
        benchmark_returns (pd.Series | None, optional): The returns per period of the
            market, for "ledoit_wolf_single_factor". Defaults to None.
        shrinkage (float, optional): The fixed shrinkage intensity of "shrunk" and
            "geodesic". Defaults to 0.1.
        gerber_threshold (float, optional): The threshold in standard deviations of
            "gerber". Defaults to 0.5.
        gerber_statistic (int, optional): The Gerber statistic (0, 1 or 2) of "gerber".
            Defaults to 1.
        denoise_alpha (float, optional): The share of the noise correlations kept by
            "denoised_shrink". Defaults to 0.
        factors (pd.DataFrame | None, optional): The factor returns per period, for
            "factor". Defaults to None.
        half_life (float, optional): The half-life in periods of "regime_adjusted_ew".
            Defaults to 40.
        seed (int | None, optional): The seed of the random subsets of
            "minimum_covariance_determinant" (also the start of "mm" and "rocke") and of the
            random splits of "nercome". Defaults to 0.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    check_option("covariance", method, COVARIANCE_METHODS)
    if method == "sample":
        return performance_model.get_covariance_matrix(returns)
    if method == "ledoit_wolf":
        return get_ledoit_wolf_covariance(returns)[0]
    if method == "ledoit_wolf_constant_correlation":
        return get_ledoit_wolf_constant_correlation_covariance(returns)[0]
    if method == "ledoit_wolf_single_factor":
        if benchmark_returns is None:
            raise ValueError(
                "The single factor covariance needs the returns of a benchmark."
            )
        return get_ledoit_wolf_single_factor_covariance(returns, benchmark_returns)[0]
    if method == "oas":
        return get_oracle_approximating_shrinkage_covariance(returns)[0]
    if method == "shrunk":
        return estimation_model.get_shrunk_covariance(returns, shrinkage)
    if method == "semicovariance":
        return get_semicovariance(returns)
    if method == "exponential":
        return get_exponential_covariance(returns, span)
    if method in ("denoised", "denoised_spectral", "denoised_shrink"):
        return get_denoised_covariance(
            returns,
            {"denoised": "fixed", "denoised_spectral": "spectral"}.get(
                method, "shrink"
            ),
            denoise_alpha,
        )[0]
    if method == "gerber":
        return estimation_model.get_gerber_covariance(
            returns, gerber_threshold, gerber_statistic
        )
    if method == "factor":
        if factors is None:
            raise ValueError("The factor covariance needs the returns of factors.")
        periods = returns.index.intersection(factors.dropna(how="any").index)
        if len(periods) <= factors.shape[1] + 1:
            raise ValueError(
                f"The returns and the factors share only {len(periods)} periods, too few "
                "to estimate the factor model."
            )
        loadings = estimation_model.get_factor_loadings(
            returns.loc[periods], factors.loc[periods]
        )
        return estimation_model.get_factor_model(
            returns.loc[periods], factors.loc[periods], loadings
        )[1]
    if method == "minimum_covariance_determinant":
        return estimation_model.get_minimum_covariance_determinant(returns, seed=seed)
    if method == "jlogo":
        return estimation_model.get_logo_covariance(returns)
    if method == "geodesic":
        return estimation_model.get_geodesic_shrinkage_covariance(returns, shrinkage)
    if method == "regime_adjusted_ew":
        return estimation_model.get_regime_adjusted_exponential_covariance(
            returns, half_life
        )[0]
    if method == "nercome":
        return estimation_model.get_nercome_covariance(returns, seed=seed)
    if method == "ccc_garch":
        return estimation_model.get_ccc_garch_covariance(returns)[0]
    # The robust estimators, "mm" and "rocke", the options that are left.
    return estimation_model.get_robust_covariance(returns, method, seed=seed)[1]


def get_expected_returns(
    returns: pd.DataFrame,
    periods_per_year: int,
    method: str = "mean",
    span: int = 180,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    adjust: bool = True,
    ranking: list | dict | pd.Series | None = None,
) -> pd.Series:
    """
    The annualized expected return of each asset, estimated from its returns per period, the
    input of a mean-variance optimization whose estimate moves the weights most:

    - The mean: the average return per period times the number of periods in a year, the
      arithmetic historical mean.
    - The geometric average: the compounded average return, (Π (1 + r_t))^(P / T) - 1 for T
      periods with P periods in a year.
    - The exponentially weighted average: with the weight of a period (1 - α)^age, α = 2 /
      (span + 1), normalized to sum to one, so recent periods count most (as pandas' ewm
      does by default), times P. Without that normalization the average is the recursion m_t
      = α r_t + (1 - α) m_(t-1) seeded with the first return.
    - The exponentially weighted geometric average: that average compounded over a year,
      (1 + m)^P - 1.
    - The CAPM return: the return the Capital Asset Pricing Model implies, r_f + β_i
      (E[r_m] - r_f), with β_i = Cov(r_i, r_m) / Var(r_m) of the returns per period and
      E[r_m] the annualized average return of the benchmark (Sharpe, 1964).
    - The Bayes-Stein estimator of Jorion (1986), which pulls every average return towards
      the average return of the minimum variance portfolio, μ_0 = 1'Σ⁻¹μ̂ / 1'Σ⁻¹1, by φ =
      (N + 2) / ((N + 2) + T (μ̂ - μ_0 1)'Σ⁻¹(μ̂ - μ_0 1)), with μ̂ the average of the T
      returns per period of the N assets and Σ = S (T - 1) / (T - N - 2) Jorion's estimate
      of the covariance matrix from the sample covariance S (ddof=1): μ = (1 - φ) μ̂ + φ μ_0
      1.
    - The James-Stein estimator: the averages pulled towards their grand mean (see
      estimation_model.get_james_stein_expected_returns).
    - The shrinkage estimator of Bodnar, Okhrin and Parolya (2019) towards the grand mean
      (see estimation_model.get_bodnar_okhrin_parolya_expected_returns).
    - The centroid: the returns an ordering of the assets implies, the centroid of Almgren
      and Chriss (2006) times the annualized volatility (see
      estimation_model.get_centroid_expected_returns); it needs the ranking.

    The average returns are the noisiest input of a mean-variance optimization; the
    shrinkage estimators are strongest when the averages differ little compared to their
    estimation error. Every estimator except the geometric ones is an arithmetic return per
    period times P, without compounding.

    See Sharpe, W. F. (1964), "Capital Asset Prices: A Theory of Market Equilibrium under
    Conditions of Risk", The Journal of Finance, 19(3), 425-442, and Jorion, P. (1986),
    "Bayes-Stein Estimation for Portfolio Analysis", Journal of Financial and
    Quantitative Analysis, 21(3), 279-292.

    Referenced against:
    - PyPortfolioOpt, mean_historical_return (geometric average): https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.mean_historical_return
    - PyPortfolioOpt, ema_historical_return (exponentially weighted averages): https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.ema_historical_return
    - PyPortfolioOpt, capm_return (CAPM return): https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html#pypfopt.expected_returns.capm_return
    - Riskfolio-Lib, mean_vector (the exponential recursion): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.mean_vector

    Shares similar logic:
    - skfolio, ShrunkMu (Bayes-Stein): https://skfolio.org/generated/skfolio.moments.ShrunkMu.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.
        method (str, optional): Options are: "mean" (the historical average), "geometric" (the
            compounded average), "exponential" (the exponentially weighted average, recent periods
            weigh most), "exponential_geometric" (that average compounded over a year), "capm" (the
            returns the Capital Asset Pricing Model implies from each asset's beta to the benchmark,
            Sharpe, 1964), "bayes_stein" (the averages shrunk towards that of the minimum variance
            portfolio, Jorion, 1986), "james_stein" (the averages shrunk towards their grand mean,
            James and Stein, 1961), "bodnar_okhrin_parolya" (the averages and their grand mean
            combined with optimal weights, Bodnar, Okhrin and Parolya, 2019), "centroid" (the
            returns an ordering of the assets implies, Almgren and Chriss, 2006). Defaults to
            "mean".
        span (int, optional): The span in periods of the exponential averages. Defaults
            to 180.
        benchmark_returns (pd.Series | None, optional): The returns per period of the
            market, for "capm". Defaults to None.
        risk_free_rate (float, optional): The annualized risk free rate, for "capm".
            Defaults to 0.
        adjust (bool, optional): Whether the exponential averages normalize the weights over the
            whole history, as pandas' ewm does by default, rather than follow the recursion.
            Defaults to True.
        ranking (list | dict | pd.Series | None, optional): The ordering of the centroid, from the
            highest expected return to the lowest: a list of assets (an entry can be a list of
            assets ranked together) or scores per asset. Defaults to None.

    Returns:
        pd.Series: The annualized expected return per asset.

    Notes:
    - Unlike skfolio's ShrunkMu and Riskfolio-Lib's mean_vector, which take the sample
      covariance S itself, the Bayes-Stein estimator uses Jorion's rescaled Σ.
    - Riskfolio-Lib's mean_vector holds the exponential average with normalized weights as
      its first variant and the recursion as its second.
    """
    if method in ("exponential", "exponential_geometric") and span < 1:
        raise ValueError("The span of the exponential average must be at least 1.")
    if method == "mean":
        mean = returns.mean()
    elif method == "geometric":
        return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    elif method == "exponential":
        mean = returns.ewm(span=span, adjust=adjust).mean().iloc[-1]
    elif method == "exponential_geometric":
        return (1 + returns.ewm(span=span, adjust=adjust).mean().iloc[-1]) ** (
            periods_per_year
        ) - 1
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
    elif method == "bayes_stein":
        periods, assets = returns.shape
        if periods <= assets + 2:
            raise ValueError(
                "The Bayes-Stein expected returns need more periods than the number of "
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
    elif method == "james_stein":
        mean, _ = estimation_model.get_james_stein_expected_returns(returns)
    elif method == "bodnar_okhrin_parolya":
        mean, _, _ = estimation_model.get_bodnar_okhrin_parolya_expected_returns(
            returns
        )
    elif method == "centroid":
        if ranking is None:
            raise ValueError(
                "The centroid expected returns need a ranking of the assets, from the "
                "highest expected return to the lowest."
            )
        return estimation_model.get_centroid_expected_returns(
            returns, ranking, periods_per_year
        )
    else:
        raise InvalidArgumentError(
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
    factors: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    The annualized expected returns and covariance matrix of a set of returns.

    See get_expected_returns and get_covariance_matrix; the annualized covariance is the
    covariance per period times the number of periods in a year.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.
        covariance (str, optional): The estimator of the covariance matrix. Options are: "sample"
            (ddof=1), "ledoit_wolf" (towards the scaled identity),
            "ledoit_wolf_constant_correlation" (towards the average correlation),
            "ledoit_wolf_single_factor" (towards the covariance the benchmark as a single factor
            implies), "oas" (Oracle Approximating Shrinkage), "shrunk" (a fixed intensity of 0.1
            towards the scaled identity), "semicovariance" (below a zero return), "exponential"
            (exponentially weighted), "denoised" (Marčenko-Pastur noise eigenvalues averaged),
            "denoised_spectral" (set to zero), "denoised_shrink" (shrunk towards the diagonal),
            "gerber" (threshold 0.5, statistic 1), "factor" (a factor model with OLS loadings),
            "minimum_covariance_determinant" (robust, seed 0), "jlogo" (the LoGo sparse inverse on
            the TMFG), "geodesic" (10% towards the scaled identity along the geodesic),
            "regime_adjusted_ew" (half-life 40, rescaled by the recent realized risk), "nercome"
            (split-sample eigenvalues, seed 0), "ccc_garch" (the next-period CCC-GARCH(1, 1)
            covariance), "mm" (robust MM-estimator, seed 0), "rocke" (Rocke's robust S-estimator,
            seed 0). Defaults to "sample".
        covariance_span (int, optional): The span in periods of the "exponential"
            covariance. Defaults to 180.
        expected_returns (str, optional): The estimator of the expected returns; the centroid needs
            a ranking, see get_expected_returns. Options are: "mean" (the historical average),
            "geometric" (the compounded average), "exponential" (the exponentially weighted average,
            recent periods weigh most), "exponential_geometric" (that average compounded over a
            year), "capm" (the returns the Capital Asset Pricing Model implies from each asset's
            beta to the benchmark, Sharpe, 1964), "bayes_stein" (the averages shrunk towards that of
            the minimum variance portfolio, Jorion, 1986), "james_stein" (the averages shrunk
            towards their grand mean, James and Stein, 1961), "bodnar_okhrin_parolya" (the averages
            and their grand mean combined with optimal weights, Bodnar, Okhrin and Parolya, 2019).
            Defaults to "mean".
        expected_returns_span (int, optional): The span in periods of the exponential
            expected returns. Defaults to 180.
        benchmark_returns (pd.Series | None, optional): The returns per period of the
            market, for the "capm" expected returns and the "ledoit_wolf_single_factor"
            covariance. Defaults to None.
        risk_free_rate (float, optional): The annualized risk free rate, for the "capm"
            expected returns. Defaults to 0.
        factors (pd.DataFrame | None, optional): The factor returns per period, for the
            "factor" covariance. Defaults to None.

    Returns:
        tuple[pd.Series, pd.DataFrame]: The expected returns and the covariance matrix.
    """
    returns = returns.dropna(how="any")

    if len(returns) < 2:  # noqa: PLR2004
        raise ValueError("At least two periods of returns are needed for every asset.")

    matrix = get_covariance_matrix(
        returns, covariance, covariance_span, benchmark_returns, factors=factors
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

    The part of the portfolio that is not invested, 1 - Σw, is cash that earns the risk free
    rate (and, above a net exposure of one, is borrowed at it), so the expected return is
    w'μ + (1 - Σw) r_f and the Sharpe ratio w'(μ - r_f) / √(w'Σw), the ratio the maximum
    Sharpe ratio optimizer maximizes, which is the same for every net exposure of the same
    mix of assets (Tobin, 1958).

    Args:
        weights (pd.Series): The weight per asset.
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        risk_free_rate (float, optional): The annualized risk free rate. Defaults to 0.

    Returns:
        pd.Series: The Expected Return, Volatility and Sharpe Ratio.
    """
    weights = weights.reindex(expected_returns.index).fillna(0)
    portfolio_return = (
        float(weights @ expected_returns) + (1 - float(weights.sum())) * risk_free_rate
    )
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
    l2_regularization: float = 0.0,
) -> pd.Series:
    """
    The fully invested portfolio with the lowest variance (Markowitz, 1952).

    An L2 regularization γ adds the penalty γ w'w, so the variance minimized is w'(Σ + γI)w,
    a ridge on the weights: with weights summing to one, w'w is the squared distance from
    the equal weights plus 1/N, so a higher γ moves the portfolio towards equal weights.

    See DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009), "A Generalized
    Approach to Portfolio Optimization: Improving Performance by Constraining Portfolio
    Norms", Management Science, 55(5), 798-812, and Hoerl, A. E. and Kennard, R. W.
    (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
    Technometrics, 12(1), 55-67.

    Referenced against:
    - PyPortfolioOpt, EfficientFrontier.min_volatility: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.min_volatility
    - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg
    - skfolio, MeanRisk: https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

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
        l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
            the weights, in the units of the covariance matrix (annualized variance for
            an annualized one), which pulls the portfolio towards equal weights and
            spreads it over more assets. Defaults to 0, no penalty.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance, positive=False)
    sigma = get_regularized_covariance(covariance.to_numpy(), l2_regularization)
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

    # SLSQP stops on an absolute change of the objective (ftol), which variances of
    # near-cash assets (about 1e-5 a year) reach long before the optimum. In units of the
    # average variance the objective is of order one, with the same minimum.
    scaled = sigma / (np.mean(np.diag(sigma)) or 1.0)
    weights = _solve(lambda w: w @ scaled @ w, lambda w: 2 * scaled @ w, constraints)

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
    l2_regularization: float = 0.0,
) -> pd.Series:
    """
    The fully invested portfolio with the highest Sharpe ratio, the tangency portfolio of
    the efficient frontier (Sharpe, 1966; Tobin, 1958).

    A transaction cost c lowers the expected return by c Σ |w_i - w0_i|, so the ratio
    maximized is (w'(μ - r_f) - c Σ d_i) / √(w'Σw) with d_i ≥ |w_i - w0_i|. The part of the
    portfolio that is not invested, 1 - Σw, earns the risk free rate (see
    get_portfolio_statistics), so, when no bound binds, a net exposure other than one scales
    the weights of the same portfolio, as on the capital market line (Tobin, 1958). An L2
    regularization γ replaces Σ by Σ + γI in the denominator, √(w'Σw + γ w'w), which pulls
    the portfolio towards equal weights. It equals adding γ y'y to the variance y'Σy that
    the homogeneous form of the problem minimizes (y = κw with (μ - r_f)'y = 1).

    See DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009), "A Generalized
    Approach to Portfolio Optimization: Improving Performance by Constraining Portfolio
    Norms", Management Science, 55(5), 798-812, and Hoerl, A. E. and Kennard, R. W.
    (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
    Technometrics, 12(1), 55-67.

    Referenced against:
    - PyPortfolioOpt, EfficientFrontier.max_sharpe: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.max_sharpe
    - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg
    - skfolio, MeanRisk (with transaction costs): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

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
        l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
            the weights, in the units of the covariance matrix (annualized variance for
            an annualized one), which pulls the portfolio towards equal weights and
            spreads it over more assets. Defaults to 0, no penalty.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance)
    mu = expected_returns.to_numpy() - risk_free_rate
    sigma = get_regularized_covariance(covariance.to_numpy(), l2_regularization)
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
    l2_regularization: float = 0.0,
) -> pd.Series:
    """
    A mean-variance efficient portfolio (Markowitz, 1952).

    It is a portfolio on the efficient frontier, the lowest variance for its expected
    return, at a target return, a target volatility or a risk aversion.

    With a target return, the portfolio with the lowest variance that expects that return.
    With a target volatility σ*, the portfolio with the highest expected return whose
    volatility is at most σ*: max w'μ subject to w'Σw ≤ σ*², the same efficient frontier
    approached from the risk side (Merton, 1972). Otherwise the portfolio that maximizes the
    expected return minus half the risk aversion times the variance. A transaction cost c
    lowers the expected return by c Σ |w_i - w0_i|, so the utility maximized is w'μ - c Σ
    d_i - (λ/2) w'Σw with d_i ≥ |w_i - w0_i|, and with a target volatility the return
    maximized is w'μ - c Σ d_i.

    See Markowitz, H. (1952), "Portfolio Selection", The Journal of Finance, 7(1), 77-91,
    and Merton, R. C. (1972), "An Analytic Derivation of the Efficient Portfolio
    Frontier", Journal of Financial and Quantitative Analysis, 7(4), 1851-1872.

    An L2 regularization γ adds the penalty γ w'w, which pulls the portfolio towards equal
    weights: with a target return the variance minimized becomes w'(Σ + γI)w, the utility
    maximized w'μ - c Σ d_i - (λ/2) w'Σw - γ w'w and with a target volatility the
    objective w'μ - c Σ d_i - γ w'w, with the volatility limit still on w'Σw.

    See DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009), "A Generalized
    Approach to Portfolio Optimization: Improving Performance by Constraining Portfolio
    Norms", Management Science, 55(5), 798-812, and Hoerl, A. E. and Kennard, R. W.
    (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
    Technometrics, 12(1), 55-67.

    Referenced against:
    - PyPortfolioOpt, EfficientFrontier.efficient_return (target return): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.efficient_return
    - PyPortfolioOpt, EfficientFrontier.efficient_risk (target volatility): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.efficient_risk
    - PyPortfolioOpt, EfficientFrontier.max_quadratic_utility (risk aversion): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.max_quadratic_utility
    - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg

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
        l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
            the weights, in the units of the covariance matrix (annualized variance for
            an annualized one), which pulls the portfolio towards equal weights and
            spreads it over more assets. Defaults to 0, no penalty.

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
    # SLSQP stops on an absolute change of the objective (ftol), which variances of
    # near-cash assets (about 1e-5 a year) reach long before the optimum. In units of the
    # average variance the objective is of order one, with the same minimum.
    variance_scale = np.mean(np.diag(sigma)) or 1.0
    scaled = sigma / variance_scale
    # The penalty γ w'w in the same units, folded into the variance for a target return.
    penalty = l2_regularization / variance_scale
    regularized = get_regularized_covariance(scaled, penalty)
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
            lambda w: w @ regularized @ w,
            lambda w: 2 * regularized @ w,
            constraints,
            [_target_return(mu, target_return)],
        )
    elif target_volatility is not None:
        if target_volatility <= 0:
            raise ValueError("The target volatility must be positive.")
        lowest = _solve(lambda w: w @ scaled @ w, lambda w: 2 * scaled @ w, constraints)
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
            slope[:count] += -mu + 2 * l2_regularization * x[:count]
            return slope

        weights = _solve(
            lambda x: -(x[:count] @ mu)
            + costs @ x
            + l2_regularization * (x[:count] @ x[:count]),
            return_gradient,
            constraints,
            # The variance budget sigma*^2 - w'Sigma w >= 0, with a target that falls
            # short of the lowest volatility only by rounding raised to it.
            [
                {
                    "type": "ineq",
                    "fun": lambda w: max(target_volatility, lowest_volatility) ** 2
                    / variance_scale
                    - w @ scaled @ w,
                    "jac": lambda w: -2 * scaled @ w,
                }
            ],
            all_variables=True,
        )
    else:
        costs = _trading_costs(constraints, transaction_cost)
        count = constraints.assets

        def gradient(x):
            slope = costs.copy()
            slope[:count] += (
                -mu
                + risk_aversion * (sigma @ x[:count])
                + 2 * l2_regularization * x[:count]
            )
            return slope

        weights = _solve(
            lambda x: -(x[:count] @ mu)
            + costs @ x
            + risk_aversion / 2 * (x[:count] @ sigma @ x[:count])
            + l2_regularization * (x[:count] @ x[:count]),
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
    l2_regularization: float = 0.0,
) -> pd.DataFrame:
    """
    The efficient frontier: the lowest variance for every expected return (Markowitz, 1952).

    For evenly spaced target returns between the minimum variance portfolio and the highest
    return the constraints allow, the portfolio with the lowest variance (Markowitz, 1952).
    An L2 regularization γ minimizes w'(Σ + γI)w instead, which pulls each portfolio towards
    equal weights; the volatility and Sharpe ratio reported are those of w'Σw.

    See DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009), "A Generalized
    Approach to Portfolio Optimization: Improving Performance by Constraining Portfolio
    Norms", Management Science, 55(5), 798-812, and Hoerl, A. E. and Kennard, R. W.
    (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
    Technometrics, 12(1), 55-67.

    Referenced against:
    - PyPortfolioOpt, objective_functions.L2_reg: https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.L2_reg

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
        l2_regularization (float, optional): The weight γ of the L2 penalty γ w'w on
            the weights, in the units of the covariance matrix (annualized variance for
            an annualized one), which pulls the portfolio towards equal weights and
            spreads it over more assets. Defaults to 0, no penalty.

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
        l2_regularization,
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

    # In units of the average variance, as in get_minimum_variance_weights.
    scaled = get_regularized_covariance(sigma, l2_regularization) / (
        np.mean(np.diag(sigma)) or 1.0
    )
    rows = []
    for number, target in enumerate(targets, start=1):
        try:
            weights = _solve(
                lambda w: w @ scaled @ w,
                lambda w: 2 * scaled @ w,
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
    Weights proportional to one over each asset's volatility, a simple risk-based allocation
    that gives the calmer assets more weight and ignores correlations.

    Shares similar logic:
    - skfolio, InverseVolatility: https://skfolio.org/generated/skfolio.optimization.InverseVolatility.html

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
    The equal risk contribution (risk parity) portfolio.

    Every asset contributes the same share of the portfolio's volatility, or the share given
    by a risk budget (Maillard, Roncalli and Teïletche, 2010). It is found by minimizing the
    convex function 1/2 y'Σy - Σ b_i ln(y_i) over y > 0 and scaling the result to sum to
    one, w = y / 1'y (Spinu, 2013).

    Weight and group bounds are constraints on w = y / 1'y, which are linear and
    homogeneous in y (w_i ≤ u becomes y_i - u 1'y ≤ 0), so the problem stays convex and
    its solution does not depend on the scale of y. Within bounds that bind, the risk
    contributions can no longer all match the budget: the result is the constrained risk
    budgeting portfolio of Richard and Roncalli (2019). Risk parity holds long positions
    only, so a negative minimum weight is refused, and the weights always sum to one.

    See Richard, J.-C. and Roncalli, T. (2019), "Constrained Risk Budgeting Portfolios:
    Theory, Algorithms, Applications & Puzzles", arXiv:1902.05710.

    Referenced against:
    - skfolio, RiskBudgeting: https://skfolio.org/generated/skfolio.optimization.RiskBudgeting.html

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        risk_budget (pd.Series | None, optional): The share of the risk each asset should
            contribute, positive for every asset; it is scaled to sum to one. A budget for
            an asset outside the portfolio, or an asset without one, raises a ValueError
            that names them. Defaults to None, an equal share each.
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

    budget = get_risk_budget(risk_budget, covariance.index)

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


CODEPENDENCE_METHODS = (
    "pearson",
    "spearman",
    "kendall",
    "gerber",
    "detoned",
    "distance",
    "mutual_information",
    "lower_tail",
    "absolute_angular",
    "squared_angular",
    "gnpr",
)


def get_codependence(
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    method: str = "pearson",
    gerber_threshold: float = 0.5,
    gerber_statistic: int = 1,
    components: int = 1,
    bins: str | int = "freedman",
    alpha: float = 0.05,
    gnpr_theta: float = 0.5,
    gnpr_bandwidth: float = 0.01,
) -> pd.DataFrame:
    """
    The codependence matrix the hierarchical clustering of the assets works on.

    It is a correlation, or another measure of how much two assets move together, that
    get_codependence_distance turns into the distance the clusters are formed on.

    - The Pearson correlation: the linear correlation ρ_ij = σ_ij / (σ_i σ_j) of the
      covariance matrix, so a shrinkage or denoised covariance estimator also shapes the
      clusters.
    - The Spearman correlation: the rank correlation of Spearman (1904), the Pearson
      correlation of the ranks of the returns, 1 - 6 Σ d_t² / (T (T² - 1)) without ties,
      with d_t the difference between the ranks of the two assets' returns in period t.
    - The Kendall correlation: the rank correlation τ_b of Kendall (1938, 1945), (n_c - n_d)
      / √((n_0 - n_1)(n_0 - n_2)), with n_c and n_d the concordant and discordant pairs of
      periods, n_0 = T (T - 1) / 2 and n_1, n_2 the pairs tied in either asset.
    - The Gerber statistic, which counts only moves beyond a threshold (see
      estimation_model.get_gerber_covariance).
    - The detoned correlation: the correlation with the first principal components, the
      market, removed (see estimation_model.get_detoned_covariance), so the clusters reflect
      what assets share besides the market.
    - The distance correlation of Székely, Rizzo and Bakirov (2007), zero only for
      independent returns (see estimation_model.get_distance_correlation_matrix).
    - The mutual information: the normalized mutual information of histograms of the returns
      (see estimation_model.get_mutual_information_matrix).
    - The lower tail dependence: how often two assets fall into their lowest returns
      together (see estimation_model.get_lower_tail_dependence_matrix).
    - The absolute and squared angular codependences: |ρ| and ρ² of the linear correlation
      of the covariance matrix, which treat a negative correlation as as close as a positive
      one of the same size (López de Prado, 2020, chapter 3).
    - The GNPR codependence: 1 - d², with d the distance of the Generic Non-Parametric
      Representation of Marti et al. (2016), a mix of the rank co-movement and the
      difference between the distributions of the returns (see
      estimation_model.get_gnpr_distance_matrix).

    The rank, Gerber, distance, information and tail measures depend on the returns
    themselves, whatever the covariance estimator; the rank correlations capture any
    monotonic co-movement and the distance correlation and mutual information also
    non-monotonic co-movement.

    See Spearman, C. (1904), "The Proof and Measurement of Association between Two
    Things", The American Journal of Psychology, 15(1), 72-101, Kendall, M. G. (1938),
    "A New Measure of Rank Correlation", Biometrika, 30(1/2), 81-93, Mantegna, R. N.
    (1999), "Hierarchical Structure in Financial Markets", The European Physical Journal
    B, 11(1), 193-197, and López de Prado, M. (2020), Machine Learning for Asset
    Managers, Cambridge University Press, chapter 3.

    Shares similar logic:
    - Riskfolio-Lib, codep_dist: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.codep_dist
    - skfolio, CovarianceDistance (Pearson correlation): https://skfolio.org/generated/skfolio.distance.CovarianceDistance.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        covariance (pd.DataFrame): The covariance matrix of the assets, for the Pearson correlation.
        method (str, optional): The codependence. Options are: "pearson" (Pearson correlation),
            "spearman" (Spearman correlation), "kendall" (Kendall correlation), "gerber" (Gerber
            statistic), "detoned" (detoned correlation), "distance" (distance correlation),
            "mutual_information" (mutual information), "lower_tail" (lower tail dependence),
            "absolute_angular" (absolute angular codependence), "squared_angular" (squared angular
            codependence), "gnpr" (GNPR codependence). Defaults to "pearson".
        gerber_threshold (float, optional): The threshold in standard deviations of
            "gerber". Defaults to 0.5.
        gerber_statistic (int, optional): The Gerber statistic (0, 1 or 2). Defaults
            to 1.
        components (int, optional): The number of principal components "detoned"
            removes. Defaults to 1.
        bins (str | int, optional): The number of histogram bins of the mutual information, or the
            rule that sets it. Options are: "freedman" (the rule of Freedman and Diaconis),
            "hacine_gharbi" (the rule of Hacine-Gharbi et al.). Defaults to "freedman".
        alpha (float, optional): The share of the lowest returns that forms the
            tail of "lower_tail". Defaults to 0.05.
        gnpr_theta (float, optional): The weight θ of the co-movement in "gnpr", between 0
            and 1. Defaults to 0.5.
        gnpr_bandwidth (float, optional): The width of the histogram bins of "gnpr", in the
            units of the returns. Defaults to 0.01.

    Returns:
        pd.DataFrame: The codependence matrix, in the order of the covariance matrix.
    """
    check_option("method", method, CODEPENDENCE_METHODS)
    if method == "gnpr":
        return (
            1
            - estimation_model.get_gnpr_distance_matrix(
                returns[covariance.index], gnpr_theta, gnpr_bandwidth
            )
            ** 2
        )
    if method in (
        "pearson",
        "gerber",
        "detoned",
        "absolute_angular",
        "squared_angular",
    ):
        if method == "gerber":
            covariance = estimation_model.get_gerber_covariance(
                returns[covariance.index], gerber_threshold, gerber_statistic
            )
        elif method == "detoned":
            covariance = estimation_model.get_detoned_covariance(
                returns[covariance.index], components
            )
        volatility = np.sqrt(np.diag(covariance.to_numpy()))
        correlation = covariance / np.outer(volatility, volatility)
        if method == "absolute_angular":
            return correlation.abs()
        if method == "squared_angular":
            return correlation**2
        return correlation
    if method == "distance":
        return estimation_model.get_distance_correlation_matrix(
            returns[covariance.index]
        )
    if method == "mutual_information":
        return estimation_model.get_mutual_information_matrix(
            returns[covariance.index], bins
        )[0]
    if method == "lower_tail":
        return estimation_model.get_lower_tail_dependence_matrix(
            returns[covariance.index], alpha
        )
    return returns[covariance.index].corr(method=method)


def get_codependence_distance(
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    method: str = "pearson",
    gerber_threshold: float = 0.5,
    gerber_statistic: int = 1,
    components: int = 1,
    bins: str | int = "freedman",
    alpha: float = 0.05,
    gnpr_theta: float = 0.5,
    gnpr_bandwidth: float = 0.01,
) -> pd.DataFrame:
    """
    The distance between every pair of assets that the hierarchical clustering works on,
    from the codependence of get_codependence (with the same arguments):

    - The Pearson, Spearman, Kendall, Gerber and detoned correlations: the correlation
      distance d = √((1 - ρ) / 2) of Mantegna (1999) and López de Prado (2016).
    - The distance correlation: d = √(1 - dCor).
    - The mutual information: the normalized variation of information, VI(X, Y) / H(X, Y).
    - The lower tail dependence: d = -ln(λ), with the tail dependence λ raised to at least
      1e-8.
    - The absolute angular codependence: d = √(1 - |ρ|) (López de Prado, 2020, chapter 3),
      and the squared angular one: d = √(1 - ρ²), the same construction on ρ², so strongly
      negatively correlated assets, which hedge each other, end up close. Taking half of 1 -
      |ρ| and 1 - ρ², as the correlation distance takes of 1 - ρ, gives a distance √2 times
      smaller with the same clusters.
    - The GNPR codependence: the GNPR distance d_θ itself (see
      estimation_model.get_gnpr_distance_matrix).

    Shares similar logic:
    - skfolio, DistanceCorrelation: https://skfolio.org/generated/skfolio.distance.DistanceCorrelation.html
    - skfolio, MutualInformation: https://skfolio.org/generated/skfolio.distance.MutualInformation.html
    - Riskfolio-Lib, codep_dist: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.codep_dist

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        covariance (pd.DataFrame): The covariance matrix of the assets, for the Pearson correlation.
        method (str, optional): The codependence. Options are: "pearson" (Pearson correlation),
            "spearman" (Spearman correlation), "kendall" (Kendall correlation), "gerber" (Gerber
            statistic), "detoned" (detoned correlation), "distance" (distance correlation),
            "mutual_information" (mutual information), "lower_tail" (lower tail dependence),
            "absolute_angular" (absolute angular codependence), "squared_angular" (squared angular
            codependence), "gnpr" (GNPR codependence). Defaults to "pearson".
        gerber_threshold (float, optional): The threshold of "gerber". Defaults to 0.5.
        gerber_statistic (int, optional): The Gerber statistic. Defaults to 1.
        components (int, optional): The components "detoned" removes. Defaults to 1.
        bins (str | int, optional): The histogram bins of the mutual information, or the rule that
            sets them. Options are: "freedman" (the rule of Freedman and Diaconis), "hacine_gharbi"
            (the rule of Hacine-Gharbi et al.). Defaults to "freedman".
        alpha (float, optional): The share of the lowest returns that forms the tail
            of the lower tail dependence. Defaults to 0.05.
        gnpr_theta (float, optional): The weight θ of "gnpr". Defaults to 0.5.
        gnpr_bandwidth (float, optional): The histogram bin width of "gnpr". Defaults to
            0.01.

    Returns:
        pd.DataFrame: The distance matrix, with zeros on the diagonal, in the order of
            the covariance matrix.
    """
    if method == "mutual_information":
        distance = estimation_model.get_mutual_information_matrix(
            returns[covariance.index], bins
        )[1].to_numpy(copy=True)
    else:
        codependence = get_codependence(
            returns,
            covariance,
            method,
            gerber_threshold,
            gerber_statistic,
            components,
            bins,
            alpha,
            gnpr_theta,
            gnpr_bandwidth,
        ).to_numpy(dtype=float)
        if method in ("distance", "absolute_angular", "squared_angular", "gnpr"):
            # 1 - dCor, 1 - |ρ|, 1 - ρ² and, for the GNPR, 1 - (1 - d²) = d².
            distance = np.sqrt(np.clip(1 - codependence, 0, None))
        elif method == "lower_tail":
            distance = -np.log(
                np.clip(codependence, estimation_model.MINIMUM_TAIL_DEPENDENCE, None)
            )
        else:
            distance = np.sqrt(np.clip((1 - codependence) / 2, 0, None))
    np.fill_diagonal(distance, 0)
    return pd.DataFrame(distance, index=covariance.index, columns=covariance.index)


def get_codependence_similarity(
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    method: str = "pearson",
    gerber_threshold: float = 0.5,
    gerber_statistic: int = 1,
    components: int = 1,
    bins: str | int = "freedman",
    alpha: float = 0.05,
    gnpr_theta: float = 0.5,
    gnpr_bandwidth: float = 0.01,
) -> pd.DataFrame:
    """
    The non-negative similarity between every pair of assets that the Directed Bubble
    Hierarchical Tree (see get_dbht_linkage) builds its planar graph on.

    It is returned next to the distance of get_codependence_distance, from the codependence
    of get_codependence (with the same arguments):

    - The Pearson, Spearman, Kendall, Gerber and detoned correlations: s = 1 - d² = (1 + ρ)
      / 2, with d = √((1 - ρ) / 2) the correlation distance. Song, Di Matteo and Aste (2012)
      take 1 + ρ, which gives the same graph and clusters.
    - The distance correlation, the mutual information, the lower tail dependence, the
      angular codependences and the GNPR codependence: the codependence itself, the distance
      correlation, the normalized mutual information, the lower tail dependence, |ρ|, ρ² or
      1 - d² of the GNPR distance, all between 0 and 1 (the last three equal to 1 - d² of
      their distance d).

    The choice per codependence is that of Riskfolio-Lib's DBHT clustering.

    Referenced against:
    - Riskfolio-Lib, HCPortfolio (DBHT linkage): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/hcportfolio.html#HCPortfolio.HCPortfolio

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        covariance (pd.DataFrame): The covariance matrix of the assets, for the Pearson correlation.
        method (str, optional): The codependence. Options are: "pearson" (Pearson correlation),
            "spearman" (Spearman correlation), "kendall" (Kendall correlation), "gerber" (Gerber
            statistic), "detoned" (detoned correlation), "distance" (distance correlation),
            "mutual_information" (mutual information), "lower_tail" (lower tail dependence),
            "absolute_angular" (absolute angular codependence), "squared_angular" (squared angular
            codependence), "gnpr" (GNPR codependence). Defaults to "pearson".
        gerber_threshold (float, optional): The threshold of "gerber". Defaults to 0.5.
        gerber_statistic (int, optional): The Gerber statistic. Defaults to 1.
        components (int, optional): The components "detoned" removes. Defaults to 1.
        bins (str | int, optional): The histogram bins of the mutual information, or the rule that
            sets them. Options are: "freedman" (the rule of Freedman and Diaconis), "hacine_gharbi"
            (the rule of Hacine-Gharbi et al.). Defaults to "freedman".
        alpha (float, optional): The share of the lowest returns that forms the tail
            of the lower tail dependence. Defaults to 0.05.
        gnpr_theta (float, optional): The weight θ of "gnpr". Defaults to 0.5.
        gnpr_bandwidth (float, optional): The histogram bin width of "gnpr". Defaults to
            0.01.

    Returns:
        pd.DataFrame: The similarity matrix, in the order of the covariance matrix.
    """
    codependence = get_codependence(
        returns,
        covariance,
        method,
        gerber_threshold,
        gerber_statistic,
        components,
        bins,
        alpha,
        gnpr_theta,
        gnpr_bandwidth,
    )
    if method in (
        "distance",
        "mutual_information",
        "lower_tail",
        "absolute_angular",
        "squared_angular",
        "gnpr",
    ):
        return codependence
    return (1 + codependence.clip(-1, 1)) / 2


def get_hierarchical_risk_parity_weights(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    correlation: pd.DataFrame | None = None,
    distance: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    risk_measure: str = "variance",
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
    similarity: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Hierarchical Risk Parity (López de Prado, 2016): weights that spread the risk over
    clusters of similar assets.

    The assets are clustered on their correlations, ordered so that similar assets sit next
    to each other, and the weight is split top-down between the two halves of each cluster
    in inverse proportion to their variance. It needs no matrix inversion, so it stays
    stable when assets are highly correlated.

    The risk of a half is that of its inverse risk portfolio, w_i ∝ 1 / ρ_i with ρ_i the
    risk of asset i alone: the inverse variance portfolio and its variance by default, or
    another risk measure ρ, such as the volatility, the CVaR or the CDaR, measured on the
    returns (see get_hierarchical_risk).

    As in the paper (Stage 1, step 2), the clustering uses the Euclidean distance between
    the columns of the correlation distance matrix d = √((1 - ρ) / 2), a distance of
    distances that compares how each pair of assets relates to all the others. Some
    libraries cluster on d itself, which can order the assets differently.

    Weight bounds move each split just enough to keep both halves within the sum of
    their assets' bounds: the share α of the left half
    is clipped to [Σ min_left / W, Σ max_left / W], with W the weight of the cluster
    being split, and then 1 - α to the same range of the right half. A split can only
    move weight between the two halves of a cluster, so group bounds, which cut across
    the clusters, cannot be applied.

    Referenced against:
    - Riskfolio-Lib, HCPortfolio (HRP): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/hcportfolio.html#HCPortfolio.HCPortfolio
    - skfolio, HierarchicalRiskParity: https://skfolio.org/generated/skfolio.optimization.HierarchicalRiskParity.html

    Shares similar logic:
    - PyPortfolioOpt, HRPOpt: https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html#pypfopt.hierarchical_portfolio.HRPOpt
    - RiskLabAI, hrp: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/optimization/hrp.py#L169

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering; the Directed
            Bubble Hierarchical Tree works on the distance itself (see get_dbht_linkage). Options
            are: "single" (single linkage), "complete" (complete linkage), "average" (average
            linkage), "ward" (Ward's method), "dbht" (the Directed Bubble Hierarchical Tree of Song,
            Di Matteo and Aste, 2012). Defaults to "single".
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0). Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on, e.g. a rank correlation from get_codependence. Defaults to
            None, the correlation matrix of the covariance matrix.
        distance (pd.DataFrame | None, optional): The distance matrix the assets are
            clustered on, e.g. from get_codependence_distance, in place of the
            correlation distance of the correlation. Defaults to None.
        returns (pd.DataFrame | None, optional): The returns per period, one column per
            asset, which the risk measures besides the variance and volatility need.
            Defaults to None.
        risk_measure (str, optional): The risk measure of the splits. Options are: "variance" (of
            the covariance matrix), "volatility" (of the covariance matrix), "mad" (mean absolute
            deviation, Konno and Yamazaki, 1991), "semideviation" (below the minimum acceptable
            return), "lower_partial_moment" (first order), "worst_realization" (the worst return),
            "gini_mean_difference" (Yitzhaki, 1982), "tail_gini" (Furman et al., 2017), "range" (the
            best minus the worst return), "cvar_range" (the CVaR of the losses and of the gains),
            "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the
            uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar"
            (Chekhlov et al., 2005). Defaults to "variance".
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold of the
            semideviation and lower partial moment per period, or "mean". Defaults to 0.
        similarity (pd.DataFrame | None, optional): The similarity matrix of the Directed Bubble
            Hierarchical Tree, e.g. from get_codependence_similarity. Defaults to None, 1 - d² of
            the distance, (1 + ρ) / 2 for the correlation distance.

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
    order = get_hierarchical_order(
        covariance, linkage_method, correlation, distance, similarity
    )
    # The bisection works on positions in numpy: label lookups in pandas took most of
    # the time, which counts in a backtest that runs it on every rebalancing date.
    sigma = covariance.to_numpy()
    lower, upper = lower.to_numpy(), upper.to_numpy()
    data = None
    if risk_measure not in ("variance", "volatility") and returns is not None:
        data = returns[covariance.index].dropna(how="any").to_numpy(dtype=float)
    measure = (risk_measure, alpha, minimum_acceptable_return)
    asset_risk = get_hierarchical_risk(np.eye(len(sigma)), sigma, data, *measure)
    if not (asset_risk > 0).all():
        raise ValueError(
            f"Every asset needs a positive {risk_measure} to be weighted by its inverse."
        )

    weights = np.ones(len(sigma))
    clusters = [order.tolist()]
    while clusters:
        clusters = [
            half
            for cluster in clusters
            if len(cluster) > 1
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
        ]
        if not clusters:
            break
        # The inverse risk portfolios of all halves of this level, one column each, are
        # measured in one go: a measure on the returns costs mostly its overhead.
        portfolios = np.zeros((len(sigma), len(clusters)))
        for column, half in enumerate(clusters):
            portfolios[half, column] = 1 / asset_risk[half]
        risk = get_hierarchical_risk(
            portfolios / portfolios.sum(axis=0), sigma, data, *measure
        )
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            split = 1 - risk[i] / (risk[i] + risk[i + 1])
            # The weight of the cluster being split, which all its assets still share.
            parent = weights[left[0]]
            split = min(
                upper[left].sum() / parent, max(lower[left].sum() / parent, split)
            )
            split = 1 - min(
                upper[right].sum() / parent,
                max(lower[right].sum() / parent, 1 - split),
            )
            weights[left] *= split
            weights[right] *= 1 - split

    return _as_series(weights, covariance.index)


LINKAGE_METHODS = ("single", "complete", "average", "ward", "dbht")


def get_hierarchical_clusters(
    covariance: pd.DataFrame,
    linkage_method: str = "ward",
    clusters: int | None = None,
    correlation: pd.DataFrame | None = None,
    distance: pd.DataFrame | None = None,
    cluster_criterion: str = "two_difference_gap",
    similarity: pd.DataFrame | None = None,
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
    max(8, √N) clusters for N assets. Summing the average distance within each cluster
    instead does not fall as clusters split and tends to pick more clusters than the
    correlation blocks hold.

    With the standardized silhouette as the criterion, it is the number of clusters, from 2
    up to the same limit, whose cut has the highest standardized silhouette of López de
    Prado (2020), the mean of the silhouette coefficients over their standard deviation
    (see get_standardized_silhouette).

    With the Optimal Number of Clusters criterion, the clusters are those of the Optimal
    Number of Clusters algorithm of López de Prado and Lewis (2019), k-means on the rows of
    the distance matrix with its own choice of their number (see
    estimation_model.get_optimal_number_of_clusters, seed 0), as Nested Clustered
    Optimization clusters in López de Prado (2020, chapter 7); they are not a cut of the
    tree, which is returned as well.

    With the Directed Bubble Hierarchical Tree as the linkage, the tree is the Directed
    Bubble Hierarchical Tree of Song, Di Matteo and Aste (2012) on the distance and the
    similarity (see get_dbht_linkage), and without a number of clusters the clusters are its
    own, the converging bubbles; the cluster criterion then does not apply. Its heights are
    ranks that tie between clusters, so a given number of clusters that falls within a tie
    returns the nearest cut with fewer clusters, as SciPy's fcluster does.

    See Yue, S., Wang, X. and Wei, M. (2008), "Application of Two-Order Difference to Gap
    Statistic", Transactions of Tianjin University, 14(3), 217-221, and Tibshirani, R.,
    Walther, G. and Hastie, T. (2001), "Estimating the Number of Clusters in a Data Set
    via the Gap Statistic", Journal of the Royal Statistical Society B, 63(2), 411-423.

    Referenced against:
    - skfolio, HierarchicalClustering: https://skfolio.org/generated/skfolio.cluster.HierarchicalClustering.html

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering. Options are:
            "single" (single linkage), "complete" (complete linkage), "average" (average linkage),
            "ward" (Ward's method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo
            and Aste, 2012). Defaults to "ward".
        clusters (int | None, optional): The number of clusters. Defaults to None, chosen by the
            two-order difference of the gap statistic (by the Directed Bubble Hierarchical Tree
            itself for that linkage).
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on, e.g. a rank correlation from get_codependence. Defaults to
            None, the correlation matrix of the covariance matrix.
        distance (pd.DataFrame | None, optional): The distance matrix the assets are
            clustered on, e.g. from get_codependence_distance, in place of the
            correlation distance of the correlation. Defaults to None.
        cluster_criterion (str, optional): How the number of clusters is chosen when it is not
            given. Options are: "two_difference_gap" (the two-order difference of the gap
            statistic), "standardized_silhouette" (the highest standardized silhouette), "onc" (the
            Optimal Number of Clusters algorithm). Defaults to "two_difference_gap".
        similarity (pd.DataFrame | None, optional): The similarity matrix of the Directed Bubble
            Hierarchical Tree, e.g. from get_codependence_similarity. Defaults to None, 1 - d² of
            the distance.

    Returns:
        tuple[np.ndarray, np.ndarray]: The linkage matrix of the clustering and the
            cluster of every asset, numbered from 0.
    """
    check_option("cluster_criterion", cluster_criterion, CLUSTER_CRITERIA)
    check_option("linkage_method", linkage_method, LINKAGE_METHODS)
    count = len(covariance)
    if count < 2:  # noqa: PLR2004
        raise ValueError("At least two assets are needed.")
    if clusters is not None and not 1 <= clusters <= count:
        raise ValueError(f"The number of clusters must lie between 1 and {count}.")

    if distance is not None:
        distance = distance.loc[covariance.index, covariance.index].to_numpy()
    else:
        if correlation is None:
            volatility = np.sqrt(np.diag(covariance.to_numpy()))
            correlation = covariance / np.outer(volatility, volatility)
        distance = np.sqrt(
            np.clip(
                (1 - correlation.loc[covariance.index, covariance.index].to_numpy())
                / 2,
                0,
                None,
            )
        )
        np.fill_diagonal(distance, 0)
    if linkage_method == "dbht":
        matrix = get_dbht_linkage(
            distance,
            (
                None
                if similarity is None
                else similarity.loc[covariance.index, covariance.index]
            ),
        )
    else:
        matrix = linkage(squareform(distance, checks=False), method=linkage_method)

    if clusters is None:
        if linkage_method == "dbht":
            # The merges within the DBHT clusters lie at heights up to 1, those between
            # them at 2 and above.
            clusters = 1 + int((matrix[:, 2] > 1).sum())
        elif count == 2:  # noqa: PLR2004
            # Two assets have no second difference; each is its own cluster.
            clusters = 2
        elif cluster_criterion == "onc":
            return matrix, estimation_model.get_optimal_number_of_clusters(
                pd.DataFrame(distance)
            )
        elif cluster_criterion == "standardized_silhouette":
            # Column j of the cut tree holds the clusters of the cut into N - j clusters.
            levels = cut_tree(matrix)
            candidates = range(2, min(count - 1, max(8, round(np.sqrt(count)))) + 1)
            scores = [
                get_standardized_silhouette(distance, levels[:, count - k])
                for k in candidates
            ]
            clusters = candidates[int(np.argmax(scores))] if scores else 2
        else:
            # Column j of the cut tree holds the clusters of the cut into N - j clusters.
            levels = cut_tree(matrix)
            dispersion = []
            for k in range(1, min(count, max(8, round(np.sqrt(count)))) + 1):
                level = levels[:, count - k]
                # Tibshirani, Walther and Hastie's pooled dispersion: per cluster the sum
                # of the squared distances between its assets over twice its size. The
                # correlation distance is Euclidean between standardized returns, so this
                # is the within-cluster sum of squares and falls as clusters split.
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


def get_dbht_linkage(
    distance: pd.DataFrame | np.ndarray,
    similarity: pd.DataFrame | np.ndarray | None = None,
) -> np.ndarray:
    """
    The Directed Bubble Hierarchical Tree (DBHT) of Song, Di Matteo and Aste (2012).

    It is a clustering that needs no number of clusters, found in the topology of a planar
    graph of the strongest links between the assets rather than by merging the closest pair
    over and over. Here the graph is the Triangulated Maximally Filtered Graph (TMFG) of the
    similarities (Massara, Di Matteo and Aste, 2017), whose bubbles, the parts that no
    triangle separates, are its 4-cliques, joined by the triangles that separate them into a
    tree.

    The algorithm is as follows, with S the similarities, D the distances and L the
    shortest path distances over the TMFG with every link as long as its distance:

    - The bubble tree links two 4-cliques that share a separating triangle. Removing the
      triangle splits the graph in two, and the link points towards the side the triangle
      is most strongly attached to, the larger sum of S between the triangle's vertices
      and the other vertices of that side.
    - The converging bubbles, the 4-cliques with no link pointing away, are the centres of
      the clusters. A vertex of one converging bubble belongs to its cluster; a vertex of
      several to the converging bubble with the largest χ = Σ_u S_vu / (3 (n_b - 2)), the
      sum over the bubble's vertices u over its 3 (n_b - 2) links, with n_b = 4 its
      vertices.
    - Every other vertex belongs to the cluster with the smallest mean L to the vertices
      assigned so far, among the converging bubbles that the vertex's bubbles flow to.
    - Within a cluster, every vertex is assigned to one bubble: its only bubble, or the
      bubble with the largest Σ_u S_vu over the total S of the bubble's links.
    - The tree is built by complete linkage on L in three levels: the vertices of every
      bubble, then the bubbles of every cluster, then the clusters. Two groups merge at the
      largest L within their union, the smallest such value first.

    The heights of the tree are ranks, as in Aste's code: the merges within a cluster of n
    assets get the heights 1 / (n - 1), 1 / (n - 2), ..., 1 in the order they happen, and
    a merge between clusters the number of clusters it joins, so a cut between 1 and 2
    returns the DBHT clusters. The leaves are then ordered so that neighbouring leaves are
    close in D (the optimal leaf ordering of Bar-Joseph, Gifford and Jaakkola, 2001),
    which changes the order but not the tree. Up to three assets are all linked and form
    a single bubble and cluster.

    What the paper leaves open follows Aste's MATLAB code (MATLAB Central File Exchange
    46750, the DBHT package):

    - The side of a separating triangle holds all the vertices beyond it, not only those
      of the neighbouring bubble.
    - A vertex of several converging bubbles takes the largest χ over all converging
      bubbles, also those it is only linked to.
    - The bubbles are ordered from the outermost one, which lies on the outside of all
      its separating triangles (the side with fewer vertices is the inside; with as many,
      the side without the lowest vertex outside the triangle), level by level inwards,
      each level in the lexicographic order of the triangles that lead to it. The
      clusters are numbered in the order of their converging bubbles, and within a
      cluster the bubbles merge their vertices in this order, which sets the heights.
    - A tie in the direction of a link points away from the outermost bubble; a tie in χ,
      the attachment to a bubble or the mean distance goes to the first in this order,
      and a tie between pairs of groups to the pair with the lowest numbers.

    The clusters and the linkage matrix equal those of Riskfolio-Lib's DBHTs, with two
    deliberate differences (see Notes).

    See Song, W.-M., Di Matteo, T. and Aste, T. (2012), "Hierarchical Information
    Clustering by Means of Topologically Embedded Graphs", PLoS ONE, 7(3), e31929, and
    Massara, G. P., Di Matteo, T. and Aste, T. (2017), "Network Filtering for Big Data:
    Triangulated Maximally Filtered Graph", Journal of Complex Networks, 5(2), 161-178.

    Referenced against:
    - Riskfolio-Lib, DBHTs: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#DBHT.DBHTs

    Args:
        distance (pd.DataFrame | np.ndarray): The N × N distance matrix D, symmetric with
            zeros on the diagonal, e.g. the correlation distance √((1 - ρ) / 2).
        similarity (pd.DataFrame | np.ndarray | None, optional): The N × N matrix of
            non-negative similarities S the TMFG keeps the largest of. Defaults to None,
            1 - D², which is (1 + ρ) / 2 for the correlation distance and the distance
            correlation for √(1 - dCor).

    Returns:
        np.ndarray: The linkage matrix in SciPy's format, one row per merge with the two
            clusters, the height and the number of assets.

    Notes:
    - Unlike Riskfolio-Lib, which drops a link of the TMFG whose similarity or distance is
      zero from the graph, such a link stays a link.
    - The triangle of the first three assets counts as separating when it separates the
      graph, which Riskfolio-Lib's FindDisjoint never does (it leaves the third vertex in
      the graph it splits), so that there it joins two 4-cliques into one bubble and can
      find other clusters.
    """
    distance = np.asarray(distance, dtype=float)
    count = len(distance)
    if count < 2:  # noqa: PLR2004
        raise ValueError("At least two assets are needed.")
    similarity = (
        np.clip(1 - distance**2, 0, None)
        if similarity is None
        else np.asarray(similarity, dtype=float)
    )
    if count >= 4:  # noqa: PLR2004
        cliques, separators = (
            estimation_model.get_triangulated_maximally_filtered_graph(similarity)
        )
    else:
        # Up to three assets are all linked: one bubble.
        cliques, separators = np.arange(count)[None, :], np.empty((0, 3), dtype=int)
    bubbles = len(cliques)
    member = np.zeros((count, bubbles), dtype=bool)
    member[cliques, np.arange(bubbles)[:, None]] = True
    linked = (member.astype(int) @ member.T.astype(int)) > 0
    np.fill_diagonal(linked, False)
    strength = np.where(linked, similarity, 0.0)
    # The shortest paths over the links; a link of zero length still counts as one.
    paths = sparse.csgraph.shortest_path(
        sparse.csgraph.csgraph_from_dense(
            np.where(linked, distance, np.inf), null_value=np.inf
        ),
        directed=False,
    )

    # The 4-clique k ≥ 1 of the TMFG grew from a face of an earlier one, its parent in the
    # bubble tree, across separator k - 1; the subtree of k holds every later 4-clique
    # that grew from it.
    parent = np.array(
        [-1]
        + [
            int(np.flatnonzero(member[separator].all(axis=0)[:k])[0])
            for k, separator in enumerate(separators, start=1)
        ]
    )
    subtree = np.eye(bubbles, dtype=bool)
    for k in range(bubbles - 1, 0, -1):
        subtree[parent[k]] |= subtree[k]
    # The vertices beyond every separator on the side of 4-clique k (below) and of its
    # parent (above), and the 4-clique on the inside, the side with fewer vertices or,
    # when both have as many, the side without the lowest vertex outside the separator.
    sides, inside = [], []
    for k, separator in enumerate(separators, start=1):
        others = np.ones(count, dtype=bool)
        others[separator] = False
        below = member[:, subtree[k]].any(axis=1) & others
        above = member[:, ~subtree[k]].any(axis=1) & others
        sides.append((below, above))
        if below.sum() == above.sum():
            inside.append(parent[k] if below[np.flatnonzero(others)[0]] else k)
        else:
            inside.append(k if below.sum() < above.sum() else parent[k])
    # The bubbles in Aste's order: from the outermost bubble, on the outside of all its
    # separators, level by level inwards, each level in the lexicographic order of the
    # separators that lead to it. The order breaks ties and sets which merges within a
    # cluster come first.
    outermost = int(np.setdiff1d(np.arange(bubbles), inside)[0])
    order, level = [outermost], {outermost}
    while level:
        steps = sorted(
            (sorted(separators[k - 1].tolist()), k if parent[k] in level else parent[k])
            for k in range(1, bubbles)
            if (parent[k] in level) != (k in level)
        )
        level = {bubble for _, bubble in steps} - set(order)
        order += [bubble for _, bubble in steps if bubble in level]

    flows = np.zeros((bubbles, bubbles), dtype=bool)
    for k, separator in enumerate(separators, start=1):
        # The link points to the side the separator is most strongly attached to; on a
        # tie away from the outermost bubble.
        below, above = sides[k - 1]
        near, far = (k, parent[k]) if subtree[k, outermost] else (parent[k], k)
        near_side, far_side = (
            (below, above) if subtree[k, outermost] else (above, below)
        )
        if (
            strength[np.ix_(separator, near_side)].sum()
            > strength[np.ix_(separator, far_side)].sum()
        ):
            flows[far, near] = True
        else:
            flows[near, far] = True
    cliques, member = cliques[order], member[:, order]
    flows = flows[np.ix_(order, order)]

    converging = np.flatnonzero(~flows.any(axis=1))
    labels = np.zeros(count, dtype=int)
    if len(converging) > 1:
        # The bubbles that flow into every converging bubble, and the vertices they hold.
        basins = np.zeros((bubbles, len(converging)), dtype=bool)
        basins[converging, np.arange(len(converging))] = True
        for _ in range(bubbles):
            grown = basins | (flows.astype(int) @ basins.astype(int) > 0)
            if (grown == basins).all():
                break
            basins = grown
        reach = (member.astype(int) @ basins.astype(int)) > 0
        centre = member[:, converging]
        chi = (strength @ centre) / (3 * (centre.sum(axis=0) - 2))
        assigned = centre.any(axis=1)
        # A vertex of several converging bubbles goes to the largest χ over all of them,
        # including those it is linked to without being their vertex, as in Aste's code.
        core = centre & (centre.sum(axis=1) == 1)[:, None]
        shared = centre.sum(axis=1) > 1
        core[shared, np.argmax(chi[shared], axis=1)] = True
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_path = paths @ core / core.sum(axis=0)
        labels = np.argmax(core, axis=1)
        labels[~assigned] = np.argmin(
            np.where(reach, mean_path, np.inf)[~assigned], axis=1
        )

    # The bubble of every vertex within its cluster.
    internal = np.array(
        [strength[np.ix_(clique, clique)].sum() / 2 for clique in cliques]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        attachment = (strength @ member) / internal
    home = np.argmax(np.where(member, attachment, -np.inf), axis=1)

    rows: list[list[float]] = []
    group = np.arange(count)

    def agglomerate(vertices: np.ndarray, heights: np.ndarray | None) -> None:
        """
        Merges the groups of the vertices by complete linkage on the shortest paths,
        with the given heights, or the number of groups joined when None.
        """
        ids = list(np.unique(group[vertices]))
        sets = [vertices[group[vertices] == i] for i in ids]
        farthest = np.array([paths[members].max(axis=0) for members in sets])
        spread = np.column_stack([farthest[:, members].max(axis=1) for members in sets])
        units = np.ones(len(ids))
        for step in range(len(ids) - 1):
            # The largest distance within the union of every pair of groups; the first
            # smallest in the order of the groups' numbers is merged.
            diameter = np.diag(spread)
            union = np.maximum(spread, np.maximum.outer(diameter, diameter))
            union[np.tril_indices(len(ids))] = np.inf
            first, second = np.unravel_index(int(np.argmin(union)), union.shape)
            height = units[first] + units[second] if heights is None else heights[step]
            joined = np.concatenate([sets[first], sets[second]])
            rows.append([ids[first], ids[second], height, len(joined)])
            group[joined] = count + len(rows) - 1
            merged = np.maximum(spread[first], spread[second])
            merged = np.append(merged, union[first, second])
            keep = [i for i in range(len(ids)) if i not in (first, second)]
            spread = np.vstack(
                [
                    np.column_stack([spread[np.ix_(keep, keep)], merged[keep]]),
                    merged[keep + [len(ids)]],
                ]
            )
            ids = [ids[i] for i in keep] + [count + len(rows) - 1]
            sets = [sets[i] for i in keep] + [joined]
            units = np.append(units[keep], units[first] + units[second])

    for label in np.unique(labels):
        cluster = np.flatnonzero(labels == label)
        heights = 1 / np.arange(len(cluster) - 1, 0, -1)
        done = 0
        for bubble in np.unique(home[cluster]):
            vertices = cluster[home[cluster] == bubble]
            agglomerate(vertices, heights[done : done + len(vertices) - 1])
            done += len(vertices) - 1
        agglomerate(cluster, heights[done:])
    agglomerate(np.arange(count), None)

    matrix = np.array(rows, dtype=float)
    return optimal_leaf_ordering(matrix, squareform(distance, checks=False))


def get_hierarchical_equal_risk_contribution_weights(
    covariance: pd.DataFrame,
    linkage_method: str = "ward",
    clusters: int | None = None,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
    correlation: pd.DataFrame | None = None,
    distance: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    risk_measure: str = "variance",
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
    within_clusters: str = "inverse_risk",
    cluster_criterion: str = "two_difference_gap",
    similarity: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Hierarchical Equal Risk Contribution (Raffinot, 2018): weights that give clusters of
    similar assets an equal share of the risk.

    The assets are clustered on their correlations and the tree is cut into clusters. Within
    a cluster the weights are inverse variance (naive risk parity). Between clusters, the
    weight is split top-down along the dendrogram: at every node the two branches get
    weights in inverse proportion to their risk, the sum of the variances of the inverse
    variance portfolios of the clusters in each branch. Unlike Hierarchical Risk Parity,
    which bisects the ordered list of assets, the splits follow the tree, so similar assets
    stay together.

    The risk of a cluster is that of its inverse risk portfolio, w_i ∝ 1 / ρ_i with ρ_i
    the risk of asset i alone: the variance by default, or another risk measure ρ, such
    as the volatility, the CVaR or the CDaR, measured on the returns (Raffinot, 2018,
    see get_hierarchical_risk). With equal weights within the clusters, the assets of a
    cluster share its weight equally while the clusters are still split by the risk of their
    inverse risk portfolios, the HERC2 of Riskfolio-Lib.

    The weight and group bounds are applied afterwards: when the
    weights fall outside them, they move to the weights within the bounds with the
    smallest relative deviation, Σ (w_i / w_HERC,i - 1)².

    See Raffinot, T. (2018), "The Hierarchical Equal Risk Contribution Portfolio", SSRN
    3237540.

    Referenced against:
    - skfolio, HierarchicalEqualRiskContribution: https://skfolio.org/generated/skfolio.optimization.HierarchicalEqualRiskContribution.html
    - Riskfolio-Lib, HCPortfolio (HERC and HERC2): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/hcportfolio.html#HCPortfolio.HCPortfolio

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering; without a number
            of clusters the Directed Bubble Hierarchical Tree takes its own clusters. Options are:
            "single" (single linkage), "complete" (complete linkage), "average" (average linkage),
            "ward" (Ward's method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo
            and Aste, 2012). Defaults to "ward".
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
        distance (pd.DataFrame | None, optional): The distance matrix the assets are
            clustered on, e.g. from get_codependence_distance, in place of the
            correlation distance of the correlation. Defaults to None.
        returns (pd.DataFrame | None, optional): The returns per period, one column per
            asset, which the risk measures besides the variance and volatility need.
            Defaults to None.
        risk_measure (str, optional): The risk measure of the clusters. Options are: "variance" (of
            the covariance matrix), "volatility" (of the covariance matrix), "mad" (mean absolute
            deviation, Konno and Yamazaki, 1991), "semideviation" (below the minimum acceptable
            return), "lower_partial_moment" (first order), "worst_realization" (the worst return),
            "gini_mean_difference" (Yitzhaki, 1982), "tail_gini" (Furman et al., 2017), "range" (the
            best minus the worst return), "cvar_range" (the CVaR of the losses and of the gains),
            "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the
            uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar"
            (Chekhlov et al., 2005). Defaults to "variance".
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold of the
            semideviation and lower partial moment per period, or "mean". Defaults to 0.
        within_clusters (str, optional): The weights within a cluster. Options are: "inverse_risk"
            (inverse risk weights), "equal" (equal weights, HERC2). Defaults to "inverse_risk".
        cluster_criterion (str, optional): How the number of clusters is chosen when it is not
            given. Options are: "two_difference_gap" (the two-order difference of the gap
            statistic), "standardized_silhouette" (the highest standardized silhouette). Defaults to
            "two_difference_gap".
        similarity (pd.DataFrame | None, optional): The similarity matrix of the Directed Bubble
            Hierarchical Tree, e.g. from get_codependence_similarity. Defaults to None, 1 - d² of
            the distance, (1 + ρ) / 2 for the correlation distance.

    Returns:
        pd.Series: The weight per asset.
    """
    _validate_covariance(covariance)
    if cluster_criterion == "onc" and clusters is None:
        raise ValueError(
            "HERC splits its weight along the tree, which the ONC clusters (k-means) do "
            "not follow; use 'two_difference_gap' or 'standardized_silhouette', or ONC "
            "with get_nested_clustered_optimization_weights."
        )
    constraints = get_weight_constraints(
        covariance.index, minimum_weight, maximum_weight, groups, group_bounds
    )
    if any(bound[0] < 0 for bound in constraints.bounds):
        raise ValueError(
            "Hierarchical Equal Risk Contribution holds long positions only; the minimum "
            "weight cannot be negative."
        )
    check_option("within_clusters", within_clusters, ("inverse_risk", "equal"))
    matrix, labels = get_hierarchical_clusters(
        covariance,
        linkage_method,
        clusters,
        correlation,
        distance,
        cluster_criterion,
        similarity,
    )
    sigma = covariance.to_numpy()
    data = None
    if risk_measure not in ("variance", "volatility") and returns is not None:
        data = returns[covariance.index].dropna(how="any").to_numpy(dtype=float)
    measure = (risk_measure, alpha, minimum_acceptable_return)
    asset_risk = get_hierarchical_risk(np.eye(len(sigma)), sigma, data, *measure)
    if not (asset_risk > 0).all():
        raise ValueError(
            f"Every asset needs a positive {risk_measure} to be weighted by its inverse."
        )

    members = [np.flatnonzero(labels == label) for label in range(labels.max() + 1)]
    # The inverse risk portfolio of every cluster, one column each, and its risk.
    portfolios = np.zeros((len(sigma), len(members)))
    for label, assets in enumerate(members):
        portfolios[assets, label] = 1 / asset_risk[assets]
    portfolios /= portfolios.sum(axis=0)
    risk = get_hierarchical_risk(portfolios, sigma, data, *measure)
    weights = portfolios.sum(axis=1)
    if within_clusters == "equal":
        weights = np.array([1 / np.sum(labels == label) for label in labels])

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
        split = 1 - risk[left].sum() / (risk[left].sum() + risk[right].sum())
        cluster_weights[left] *= split
        cluster_weights[right] *= 1 - split
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
    distance: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    risk_measure: str = "variance",
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
    cluster_criterion: str = "two_difference_gap",
    similarity: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Nested Clustered Optimization (López de Prado, 2019).

    Weights optimized within clusters of similar assets and then across the clusters, which
    keeps the noise of the covariance matrix from spreading over the whole portfolio. The
    assets are clustered on their correlations, the objective is optimized within every
    cluster, and then again across the clusters, each treated as one asset with the returns
    of its portfolio. The noise of the covariance matrix mostly lives within clusters of
    correlated assets, so containing each optimization to one level keeps the weights
    stable.

    With W the N × K matrix whose column k holds the weights within cluster k, the
    clusters have the expected returns W'μ and the covariance matrix W'ΣW, the weights
    across the clusters v are optimal for those, and the weight of every asset is
    w = W v. A cluster of one asset holds that asset only. With the maximum Sharpe ratio
    objective, a cluster without an asset above the risk free rate holds its minimum
    variance portfolio. Both levels are long only and fully invested. The paper clusters
    with k-means and solves both levels without bounds, allowing short positions; here the
    clustering is hierarchical, as in HERC, or the paper's k-means with the Optimal Number
    of Clusters criterion.

    With another risk measure than the variance (or the volatility, whose minimum and equal
    risk contribution portfolios are those of the variance), the minimum risk objective
    minimizes that measure of the returns (get_mean_risk_weights, get_minimum_cvar_weights
    or get_minimum_drawdown_weights) and the risk parity objective equalizes its risk
    contributions (get_risk_budgeting_weights), within the clusters on the returns of their
    assets and across them on the returns of the clusters' portfolios, R W, as
    Riskfolio-Lib's HCPortfolio does for NCO. The maximum Sharpe ratio objective takes the
    variance only.

    See López de Prado, M. (2019), "A Robust Estimator of the Efficient Frontier", SSRN
    3469961, and López de Prado, M. (2020), Machine Learning for Asset Managers,
    Cambridge University Press, chapter 7.

    Referenced against:
    - Riskfolio-Lib, HCPortfolio (NCO): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/hcportfolio.html#HCPortfolio.HCPortfolio

    Shares similar logic:
    - skfolio, NestedClustersOptimization: https://skfolio.org/generated/skfolio.optimization.NestedClustersOptimization.html
    - RiskLabAI, get_optimal_portfolio_weights_nco: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/optimization/nco.py#L49

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        objective (str, optional): The objective at both levels. Options are: "minimum_variance"
            (the lowest risk), "maximum_sharpe" (the highest Sharpe ratio), "risk_parity" (equal
            risk contributions). Defaults to "minimum_variance".
        risk_free_rate (float, optional): The annualized risk free rate of the maximum
            Sharpe ratio. Defaults to 0.
        linkage_method (str, optional): The linkage of the hierarchical clustering; without a number
            of clusters the Directed Bubble Hierarchical Tree takes its own clusters. Options are:
            "single" (single linkage), "complete" (complete linkage), "average" (average linkage),
            "ward" (Ward's method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo
            and Aste, 2012). Defaults to "ward".
        clusters (int | None, optional): The number of clusters. Defaults to None, chosen
            by the two-order difference of the gap statistic.
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on, e.g. a rank correlation from get_codependence. Defaults to
            None, the correlation matrix of the covariance matrix.
        distance (pd.DataFrame | None, optional): The distance matrix the assets are
            clustered on, e.g. from get_codependence_distance, in place of the
            correlation distance of the correlation. Defaults to None.
        returns (pd.DataFrame | None, optional): The returns per period, one column per
            asset, which the risk measures besides the variance and volatility need.
            Defaults to None.
        risk_measure (str, optional): The risk measure of the minimum risk and risk parity
            objectives. Options are: "variance" (of the covariance matrix), "volatility" (of the
            covariance matrix), "mad" (mean absolute deviation, Konno and Yamazaki, 1991),
            "semideviation" (below the minimum acceptable return), "lower_partial_moment" (first
            order), "worst_realization" (the worst return), "gini_mean_difference" (Yitzhaki, 1982),
            "tail_gini" (Furman et al., 2017), "range" (the best minus the worst return),
            "cvar_range" (the CVaR of the losses and of the gains), "tail_gini_range" (the Tail Gini
            of the losses and of the gains), "ulcer_index" (of the uncompounded cumulative return),
            "cvar" (Rockafellar and Uryasev, 2002), "cdar" (Chekhlov et al., 2005). Defaults to
            "variance".
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold of the
            semideviation and lower partial moment per period, or "mean". Defaults to 0.
        cluster_criterion (str, optional): How the number of clusters is chosen when it is not
            given. Options are: "two_difference_gap" (the two-order difference of the gap
            statistic), "standardized_silhouette" (the highest standardized silhouette), "onc" (the
            k-means clusters of the Optimal Number of Clusters algorithm, as in López de Prado,
            2020, chapter 7). Defaults to "two_difference_gap".
        similarity (pd.DataFrame | None, optional): The similarity matrix of the Directed Bubble
            Hierarchical Tree, e.g. from get_codependence_similarity. Defaults to None, 1 - d² of
            the distance, (1 + ρ) / 2 for the correlation distance.

    Returns:
        pd.Series: The weight per asset.
    """
    check_option("objective", objective, NESTED_CLUSTERED_OBJECTIVES)
    _validate_covariance(covariance)
    on_returns = risk_measure not in ("variance", "volatility")
    if on_returns:
        check_option("risk_measure", risk_measure, HIERARCHICAL_RISK_MEASURES)
        if objective == "maximum_sharpe":
            raise ValueError(
                "The maximum Sharpe objective takes the variance only; use "
                "'minimum_variance' or 'risk_parity' with another risk measure."
            )
        if returns is None:
            raise ValueError(f"The risk measure '{risk_measure}' needs the returns.")
    history = (
        returns[covariance.index].dropna(how="any")
        if on_returns and returns is not None
        else None
    )

    def optimize(
        mu: pd.Series, sigma: pd.DataFrame, data: pd.DataFrame | None
    ) -> pd.Series:
        if len(sigma) == 1:
            return pd.Series(1.0, index=sigma.index)
        if data is not None:
            if objective == "risk_parity":
                return get_risk_budgeting_weights(
                    data,
                    risk_measure,
                    alpha=alpha,
                    minimum_acceptable_return=minimum_acceptable_return,
                )
            if risk_measure == "cvar":
                return get_minimum_cvar_weights(data, alpha)
            if risk_measure == "cdar":
                return get_minimum_drawdown_weights(data, "cdar", alpha)
            return get_mean_risk_weights(
                data,
                risk_measure,
                alpha=alpha,
                minimum_acceptable_return=minimum_acceptable_return,
            )
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
        covariance,
        linkage_method,
        clusters,
        correlation,
        distance,
        cluster_criterion,
        similarity,
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
            expected_returns[assets],
            covariance.loc[assets, assets],
            history[assets] if history is not None else None,
        )

    across = optimize(
        within.T @ expected_returns.reindex(covariance.index),
        within.T @ covariance @ within,
        history @ within if history is not None else None,
    )

    return _as_series((within @ across).to_numpy(), covariance.index)


def get_minimum_cvar_weights(
    returns: pd.DataFrame,
    alpha: float = 0.05,
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
    shortfall) over the historical returns.

    It is solved as a linear programme (Rockafellar and Uryasev, 2000).

    Referenced against:
    - skfolio, MeanRisk (CVaR): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

    Shares similar logic:
    - PyPortfolioOpt, EfficientCVaR.min_cvar: https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#pypfopt.efficient_frontier.EfficientCVaR.min_cvar
    - Riskfolio-Lib, Portfolio.optimization: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        alpha (float, optional): The tail probability of the CVaR, e.g. 0.05 for the
            average of the worst 5% of periods. Defaults to 0.05.
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
    validate_alpha(alpha)

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
            np.full(periods, 1 / (alpha * periods)),
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


def get_tail_probabilities(losses: np.ndarray | pd.Series, alpha: float) -> np.ndarray:
    """
    The weight of every period in the expected tail loss of Rockafellar and Uryasev (2002).

    1 / ((1 - α)T) for each of the ⌊(1 - α)T⌋ largest losses, the remaining fraction of the
    tail for the next largest and 0 for the others, so that the weights sum to one. The
    expected tail loss is the losses weighted by them; the returns weighted by them give
    each asset's share of it (Euler allocation).

    Args:
        losses (np.ndarray | pd.Series): The losses per period, a loss being positive.
        alpha (float): The tail probability α, e.g. 0.05 for the worst 5%.

    Returns:
        np.ndarray: The weight of every period, in the order of the losses.
    """
    validate_alpha(alpha)
    losses = np.asarray(losses, dtype=float)
    if len(losses) == 0:
        raise ValueError("There are no losses to take the tail of.")
    tail = alpha * len(losses)
    # A tail of 62.75 periods holds the 62 largest losses in full and three quarters of
    # the 63rd; the stable sort keeps the earliest of tied losses first.
    order = np.argsort(-losses, kind="stable")
    full = min(math.floor(tail), len(losses))
    probabilities = np.zeros(len(losses))
    probabilities[order[:full]] = 1 / tail
    if full < len(losses):
        probabilities[order[full]] = (tail - full) / tail
    return probabilities


def get_expected_tail_loss(losses: np.ndarray | pd.Series, alpha: float) -> float:
    """
    The average of the worst 1 - α share of the losses, the minimum over ζ of ζ + Σ max(0,
    L_t - ζ) / ((1 - α)T) (Rockafellar and Uryasev, 2002).

    It counts a fraction of the loss on the boundary when (1 - α)T is not a whole number,
    and it is the value the CVaR and CDaR linear programmes minimize. Every CVaR, CDaR and
    expected shortfall of the portfolio construction uses it.

    Args:
        losses (np.ndarray | pd.Series): The losses per period, a loss being positive.
        alpha (float): The tail probability α, e.g. 0.05 for the worst 5%.

    Returns:
        float: The expected tail loss.
    """
    losses = np.asarray(losses, dtype=float)
    return float(get_tail_probabilities(losses, alpha) @ losses)


def get_drawdowns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    The drawdown in every period on uncompounded returns (Chekhlov, Uryasev and Zabarankin,
    2005).

    A drawdown is the highest cumulative sum of the returns so far, starting from zero,
    minus the current cumulative sum. These are absolute drawdowns of a portfolio that keeps
    its invested amount constant, which keeps the drawdown optimizers linear; compounded
    drawdowns are relative to the peak value.

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
    alpha: float = 0.05,
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
    linear programme (Chekhlov, Uryasev and Zabarankin, 2005).

    With y_t = Σ_(s≤t) r_s'w the uncompounded cumulative return and u_t ≥ y_t, u_t ≥
    u_(t-1), u_0 = 0 its running peak, the drawdown is u_t - y_t and the programme
    minimizes:

    - The Conditional Drawdown at Risk: the average of the worst 1 - α share of the
      drawdowns, ζ + Σ z_t / ((1 - α)T) with z_t ≥ u_t - y_t - ζ and z_t ≥ 0.
    - The maximum drawdown m, with u_t - y_t ≤ m.
    - The average drawdown Σ (u_t - y_t) / T.

    Referenced against:
    - Riskfolio-Lib, Portfolio.optimization: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization
    - skfolio, MeanRisk: https://skfolio.org/generated/skfolio.optimization.MeanRisk.html
    - PyPortfolioOpt, EfficientCDaR.min_cdar (Conditional Drawdown at Risk): https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#pypfopt.efficient_frontier.EfficientCDaR.min_cdar

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        measure (str, optional): The drawdown measure. Options are: "cdar" (Conditional Drawdown at
            Risk), "maximum" (maximum drawdown), "average" (average drawdown). Defaults to "cdar".
        alpha (float, optional): The tail probability of the CDaR, e.g. 0.05 for the
            average of the worst 5% of drawdowns. Defaults to 0.05.
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
    check_option("measure", measure, DRAWDOWN_MEASURES)
    validate_alpha(alpha)

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
                np.full(periods, 1 / (alpha * periods)),
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
    The portfolio with the highest Omega ratio over the historical returns.

    The ratio is Ω(w) = 1 + (r̄'w - τ) / E[max(0, τ - r_t'w)] with τ the minimum acceptable
    return (Keating and Shadwick, 2002).

    The ratio of a linear function to a convex piecewise-linear one is maximized as a linear
    programme with the Charnes-Cooper transformation y = κw, κ > 0 (Kapsos, Christofides and
    Rustem, 2014):

    maximize r̄'y - τκ subject to Σ d_t / T = 1, d_t ≥ τκ - r_t'y, d_t ≥ 0 and the
    weight constraints in homogeneous form; the weights are w = y / κ.

    See Kapsos, M., Christofides, N. and Rustem, B. (2014), "Worst-Case Robust Omega
    Ratio", European Journal of Operational Research, 234(2), 499-507, and Mausser, H.,
    Saunders, D. and Seco, L. (2006), "Optimising Omega", Risk, 19(11), 88-92.

    Referenced against:
    - Riskfolio-Lib, Portfolio.optimization (first lower partial moment): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization

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
    alpha: float = 0.05,
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
    The portfolio with the highest ratio of expected excess return to Conditional Value at
    Risk over the historical returns.

    The ratio is (r̄ - r_f)'w / CVaR_α(w), the STARR ratio of Martin, Rachev and Siboulet
    (2003).

    With the CVaR as the linear programme of Rockafellar and Uryasev (2000) and the
    Charnes-Cooper transformation y = κw, κ > 0, it is the linear programme:

    maximize (r̄ - r_f)'y subject to ζ + Σ s_t / ((1 - α)T) ≤ 1, s_t ≥ -r_t'y - ζ,
    s_t ≥ 0 and the weight constraints in homogeneous form; the weights are w = y / κ.

    See Martin, R. D., Rachev, S. T. and Siboulet, F. (2003), "Phi-Alpha Optimal
    Portfolios and Extreme Risk Management", Wilmott Magazine, November, 70-83.

    Referenced against:
    - Riskfolio-Lib, Portfolio.optimization (CVaR): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization
    - skfolio, MeanRisk (CVaR): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        alpha (float, optional): The tail probability of the CVaR, e.g. 0.05 for the
            average of the worst 5% of periods. Defaults to 0.05.
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
    validate_alpha(alpha)
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
                        np.full(periods, 1 / (alpha * periods)),
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


# The risk measures of the mean-risk optimizer, with the column they are shown under.
MEAN_RISK_MEASURES = {
    "mad": "MAD",
    "semideviation": "Semideviation",
    "lower_partial_moment": "Lower Partial Moment",
    "worst_realization": "Worst Realization",
    "gini_mean_difference": "Gini Mean Difference",
    "tail_gini": "Tail Gini",
    "range": "Range",
    "cvar_range": "CVaR Range",
    "tail_gini_range": "Tail Gini Range",
    "ulcer_index": "Ulcer Index",
    "cvar": "CVaR",
    "cdar": "CDaR",
    "maximum_drawdown": "Maximum Drawdown",
    "average_drawdown": "Average Drawdown",
}
MEAN_RISK_OBJECTIVES = ("minimum_risk", "maximum_ratio")

# The measures that are the square root of a sum of squares, which the optimizers minimize
# with SLSQP; the others are linear programmes.
QUADRATIC_RISK_MEASURES = ("semideviation", "ulcer_index")

# The measures of the risk budgeting portfolios besides the volatility.
RISK_BUDGETING_MEASURES = {
    measure: name
    for measure, name in MEAN_RISK_MEASURES.items()
    if measure
    not in ("worst_realization", "range", "maximum_drawdown", "average_drawdown")
}

# How often the cap on the scale of the risk budgeting solution of an ordered weighted
# average is raised.
RISK_BUDGETING_CAP_RAISES = 10


LINEAR_RISK_MEASURES = (
    "mad",
    "lower_partial_moment",
    "worst_realization",
    "range",
    "cvar",
    "cvar_range",
    "cdar",
    "maximum_drawdown",
    "average_drawdown",
    "gini_mean_difference",
    "tail_gini",
    "tail_gini_range",
)


# The ordered weighted averages among them, which the solvers meet with cutting planes.
OWA_RISK_MEASURES = ("gini_mean_difference", "tail_gini", "tail_gini_range")


class RiskProgramme(NamedTuple):
    """
    A risk measure of the portfolio's returns as a linear programme over the weights w and
    auxiliary variables z.

    The measure is cost'z at the smallest z that meets A_w w + A_z z ≤ upper and E_w w + E_z
    z = 0 within the bounds on z. The right-hand side upper is only nonzero for a minimum
    acceptable return; it scales with κ in the Charnes-Cooper transformation of a ratio.
    """

    weights: sparse.csr_matrix
    auxiliary: sparse.csr_matrix
    upper: np.ndarray
    equality_weights: sparse.csr_matrix
    equality_auxiliary: sparse.csr_matrix
    cost: np.ndarray
    bounds: list[tuple[float | None, float | None]]


def get_risk_measure_programme(
    data: np.ndarray,
    risk_measure: str,
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
) -> RiskProgramme:
    """
    The linear programme of a risk measure of the portfolio's returns X_t = r_t'w over T
    periods, with α the tail probability:

    - The mean absolute deviation Σ |X_t - X̄| / T, as 2 Σ d_t / T with d_t ≥ -(r_t - r̄)'w
      and d_t ≥ 0, since the deviations below and above the mean are equal in total (Konno
      and Yamazaki, 1991).
    - The first lower partial moment Σ max(τ - X_t, 0) / T with d_t ≥ τ - r_t'w and d_t ≥ 0,
      τ the minimum acceptable return or each asset's mean (Bawa, 1975).
    - The worst realization: the largest loss m ≥ -r_t'w (Young, 1998).
    - The range: the largest gain plus the largest loss, m_G ≥ r_t'w and m_L ≥ -r_t'w.
    - The Conditional Value at Risk ζ + Σ s_t / (αT) with s_t ≥ -r_t'w - ζ and s_t ≥ 0
      (Rockafellar and Uryasev, 2000).
    - The CVaR range: the CVaR of the losses plus that of the gains, the average of the best
      α share of the returns minus the average of the worst (Cajas, 2021).
    - The Conditional Drawdown at Risk of the uncompounded cumulative returns, with running
      peaks u_t ≥ Σ_(s≤t) r_s'w, u_t ≥ u_(t-1), u_t ≥ 0 and the CVaR programme on the
      drawdowns u_t - Σ_(s≤t) r_s'w (Chekhlov, Uryasev and Zabarankin, 2005).
    - The maximum drawdown: the largest drawdown m ≥ u_t - Σ_(s≤t) r_s'w, with the same
      running peaks.
    - The average drawdown: a ≥ Σ (u_t - Σ_(s≤t) r_s'w) / T, with the same running peaks.
    - The Gini mean difference, the Tail Gini and the Tail Gini range: ordered weighted
      averages Σ c_i X_(i) of the sorted returns (see get_owa_risk). Their programme is a
      single variable θ, the measure, with cost one and no rows: the solvers add the cuts θ
      ≥ g'w of get_owa_risk as they go, since writing the ordered weighted average out as a
      linear programme takes T² variables (Ogryczak and Śliwiński, 2003).

    The returns are divided by their largest absolute value first, which leaves the
    weights that minimize the measure or maximize a ratio to it unchanged and keeps the
    programme well scaled for HiGHS: the cost of the programme is the measure of the
    scaled returns.

    See Konno, H. and Yamazaki, H. (1991), "Mean-Absolute Deviation Portfolio
    Optimization Model and Its Applications to Tokyo Stock Market", Management Science,
    37(5), 519-531; Young, M. R. (1998), "A Minimax Portfolio Selection Rule with Linear
    Programming Solution", Management Science, 44(5), 673-683; Cajas, D. (2021), "OWA
    Portfolio Optimization: A Disciplined Convex Programming Framework", SSRN 3988927;
    and Ogryczak, W. and Śliwiński, T. (2003), "On Solving Linear Programs with the
    Ordered Weighted Averaging Objective", European Journal of Operational Research,
    148(1), 80-91.

    Args:
        data (np.ndarray): The T × N returns, without missing values.
        risk_measure (str): The risk measure. Options are: "mad" (mean absolute deviation),
            "lower_partial_moment" (first lower partial moment), "worst_realization" (the largest
            loss), "range" (the largest gain plus the largest loss), "cvar" (Conditional Value at
            Risk), "cvar_range" (the CVaR of the losses plus that of the gains), "cdar" (Conditional
            Drawdown at Risk), "maximum_drawdown" (the largest drawdown), "average_drawdown" (the
            average drawdown), "gini_mean_difference" (Gini mean difference), "tail_gini" (Tail
            Gini), "tail_gini_range" (Tail Gini range).
        alpha (float, optional): The tail probability α of the tail measures, e.g. 0.05 for the worst
            5% of periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold τ of the lower partial
            moment per period, or "mean" for each asset's mean. Defaults to 0.

    Returns:
        RiskProgramme: The rows, cost and bounds of the programme.
    """
    periods, assets = data.shape
    scale = np.max(np.abs(data)) or 1.0
    data = data / scale
    identity = sparse.eye(periods, format="csr")
    ones = sparse.csr_matrix(np.ones((periods, 1)))
    returns = sparse.csr_matrix(data)
    equality_weights = sparse.csr_matrix((0, assets))
    equality_auxiliary = None
    upper = np.zeros(periods)

    if risk_measure == "mad":
        weights = sparse.csr_matrix(-(data - data.mean(axis=0)))
        auxiliary = -identity
        cost = np.full(periods, 2 / periods)
        bounds = [(0.0, None)] * periods
    elif risk_measure == "lower_partial_moment":
        if minimum_acceptable_return == "mean":
            weights = sparse.csr_matrix(-(data - data.mean(axis=0)))
        else:
            weights = -returns
            upper = np.full(periods, -float(minimum_acceptable_return) / scale)
        auxiliary = -identity
        cost = np.full(periods, 1 / periods)
        bounds = [(0.0, None)] * periods
    elif risk_measure == "worst_realization":
        weights, auxiliary = -returns, -ones
        cost, bounds = np.ones(1), [(None, None)]
    elif risk_measure == "range":
        weights = sparse.vstack([-returns, returns])
        auxiliary = sparse.block_diag([-ones, -ones])
        cost, bounds = np.ones(2), [(None, None)] * 2
        upper = np.zeros(2 * periods)
    elif risk_measure in ("cvar", "cvar_range"):
        tail = np.concatenate([[1.0], np.full(periods, 1 / (alpha * periods))])
        block = sparse.hstack([-ones, -identity])
        bounds = [(None, None)] + [(0.0, None)] * periods
        if risk_measure == "cvar":
            weights, auxiliary, cost = -returns, block, tail
        else:
            # The CVaR of the gains is that of the losses of a short position.
            weights = sparse.vstack([-returns, returns])
            auxiliary = sparse.block_diag([block, block])
            cost, bounds = np.tile(tail, 2), bounds * 2
            upper = np.zeros(2 * periods)
    elif risk_measure == "cdar":
        # The peaks u_t, ζ and the excess drawdowns s_t: Y_t - u_t ≤ 0, u_t - u_(t+1) ≤ 0
        # and u_t - Y_t - ζ - s_t ≤ 0 with Y_t the cumulative return.
        cumulative = sparse.csr_matrix(np.cumsum(data, axis=0))
        rising = sparse.eye(periods - 1, periods) - sparse.eye(
            periods - 1, periods, k=1
        )
        weights = sparse.vstack(
            [cumulative, sparse.csr_matrix((periods - 1, assets)), -cumulative]
        )
        auxiliary = sparse.bmat(
            [
                [-identity, None, None],
                [rising, sparse.csr_matrix((periods - 1, 1)), None],
                [identity, -ones, -identity],
            ],
            format="csr",
        )
        cost = np.concatenate(
            [np.zeros(periods), [1.0], np.full(periods, 1 / (alpha * periods))]
        )
        bounds = [(0.0, None)] * periods + [(None, None)] + [(0.0, None)] * periods
        upper = np.zeros(3 * periods - 1)
    elif risk_measure in ("maximum_drawdown", "average_drawdown"):
        # The peaks u_t and the measure m: Y_t - u_t ≤ 0, u_t - u_(t+1) ≤ 0 and then
        # u_t - Y_t - m ≤ 0 for the maximum, or Σ (u_t - Y_t) / T - m ≤ 0 for the average.
        cumulative = sparse.csr_matrix(np.cumsum(data, axis=0))
        rising = sparse.eye(periods - 1, periods) - sparse.eye(
            periods - 1, periods, k=1
        )
        if risk_measure == "maximum_drawdown":
            drawdown_weights, drawdown_auxiliary = -cumulative, [identity, -ones]
        else:
            drawdown_weights = sparse.csr_matrix(-cumulative.sum(axis=0) / periods)
            drawdown_auxiliary = [
                sparse.csr_matrix(np.full((1, periods), 1 / periods)),
                sparse.csr_matrix(-np.ones((1, 1))),
            ]
        weights = sparse.vstack(
            [cumulative, sparse.csr_matrix((periods - 1, assets)), drawdown_weights]
        )
        auxiliary = sparse.bmat(
            [
                [-identity, sparse.csr_matrix((periods, 1))],
                [rising, sparse.csr_matrix((periods - 1, 1))],
                drawdown_auxiliary,
            ],
            format="csr",
        )
        cost = np.concatenate([np.zeros(periods), [1.0]])
        bounds = [(0.0, None)] * periods + [(0.0, None)]
        upper = np.zeros(weights.shape[0])
    elif risk_measure in OWA_RISK_MEASURES:
        weights, auxiliary = sparse.csr_matrix((0, assets)), sparse.csr_matrix((0, 1))
        cost, bounds, upper = np.ones(1), [(None, None)], np.zeros(0)
    else:
        raise InvalidArgumentError(
            f"The risk measure must be one of {', '.join(LINEAR_RISK_MEASURES)}, not "
            f"'{risk_measure}'."
        )

    if equality_auxiliary is None:
        equality_auxiliary = sparse.csr_matrix((0, len(cost)))

    return RiskProgramme(
        sparse.csr_matrix(weights),
        sparse.csr_matrix(auxiliary),
        upper,
        sparse.csr_matrix(equality_weights),
        equality_auxiliary,
        cost,
        bounds,
    )


def get_squared_risk(
    data: np.ndarray,
    weights: np.ndarray,
    risk_measure: str,
    minimum_acceptable_return: float | str = 0.0,
) -> tuple[float, np.ndarray]:
    """
    The square of a quadratic risk measure of the portfolio's returns X_t = r_t'w over T
    periods, and its gradient in the weights:

    - The semideviation: Σ min(X_t - τ, 0)² / T below a minimum acceptable return τ, or Σ
      min(X_t - X̄, 0)² / (T - 1) below the mean, the square of the Risk module's downside
      deviation (Markowitz, 1959; Sortino and Price, 1994). Its gradient, 2 Σ min(X_t - τ,
      0) r_t / T, is continuous, so SLSQP minimizes it directly in the weights as Estrada
      (2008) suggests.
    - The Ulcer Index: Σ D_t² / T with D_t = max_(s≤t) Y_s - Y_t the drawdowns of the
      uncompounded cumulative return Y_t (from Y_0 = 0), the square of the Ulcer Index of
      Martin and McCann (1989), with gradient 2 Σ D_t (C_s*(t) - C_t) / T for the cumulative
      asset returns C and s*(t) the latest peak.

    Args:
        data (np.ndarray): The T × N returns, without missing values.
        weights (np.ndarray): The weight per asset.
        risk_measure (str): The quadratic risk measure. Options are: "semideviation"
            (semideviation), "ulcer_index" (Ulcer Index).
        minimum_acceptable_return (float | str, optional): The threshold τ of the
            semideviation per period, or "mean". Defaults to 0.

    Returns:
        tuple[float, np.ndarray]: The squared measure and its gradient.
    """
    periods = len(data)
    if risk_measure == "semideviation":
        if minimum_acceptable_return == "mean":
            centred = data - data.mean(axis=0)
            shortfall = np.minimum(centred @ weights, 0)
            divisor = periods - 1
        else:
            centred = data
            shortfall = np.minimum(data @ weights - float(minimum_acceptable_return), 0)
            divisor = periods
        return float(shortfall @ shortfall) / divisor, 2 * shortfall @ centred / divisor
    if risk_measure == "ulcer_index":
        cumulative = np.vstack([np.zeros(data.shape[1]), np.cumsum(data, axis=0)])
        wealth = cumulative @ weights
        records = np.where(
            wealth >= np.maximum.accumulate(wealth), np.arange(len(wealth)), 0
        )
        slopes = cumulative[np.maximum.accumulate(records)[1:]] - cumulative[1:]
        drawdowns = slopes @ weights
        return float(drawdowns @ drawdowns) / periods, 2 * drawdowns @ slopes / periods
    raise ValueError(
        f"The risk measure must be one of {', '.join(QUADRATIC_RISK_MEASURES)}, not "
        f"'{risk_measure}'."
    )


def get_owa_risk(
    data: np.ndarray,
    weights: np.ndarray,
    risk_measure: str,
    alpha: float = 0.05,
) -> tuple[float, np.ndarray]:
    """
    An ordered weighted average Σ c_i X_(i) of the portfolio's returns X_t = r_t'w sorted
    from low to high, with a subgradient in the weights.

    The weights c_i are those of the Risk module's dispersion measures: the Gini mean
    difference (Yitzhaki, 1982), minus the Tail Gini of the returns (Furman, Wang and
    Zitikis, 2017), and the Tail Gini range, minus the Tail Gini of the returns plus that of
    the returns sorted from high to low. The weights rise with i, so by the rearrangement
    inequality the average is the largest of Σ c_i X_π(i) over all orderings π: a maximum of
    functions that are linear in the weights, hence convex and positively homogeneous
    (Cajas, 2021). The linear function of the sorted order, g'w with g = Σ c_i r_(i),
    touches it at w and lies below it everywhere else, which makes θ ≥ g'w a cut of the
    cutting plane solvers.

    See Yitzhaki, S. (1982), "Stochastic Dominance, Mean Variance, and Gini's Mean
    Difference", The American Economic Review, 72(1), 178-185; Furman, E., Wang, R. and
    Zitikis, R. (2017), "Gini-Type Measures of Risk and Variability: Gini Shortfall,
    Capital Allocations, and Heavy-Tailed Risks", Journal of Banking & Finance, 83,
    70-84; and Cajas, D. (2021), "OWA Portfolio Optimization: A Disciplined Convex
    Programming Framework", SSRN 3988927.

    Args:
        data (np.ndarray): The T × N returns, without missing values.
        weights (np.ndarray): The weight per asset.
        risk_measure (str): The ordered weighted average. Options are: "gini_mean_difference" (Gini
            mean difference), "tail_gini" (Tail Gini), "tail_gini_range" (Tail Gini range).
        alpha (float, optional): The tail probability of the Tail Gini measures.
            Defaults to 0.05.

    Returns:
        tuple[float, np.ndarray]: The measure and its subgradient.
    """
    periods = len(data)
    if risk_measure == "gini_mean_difference":
        coefficients = dispersion_model.get_owa_gmd_weights(periods)
    elif risk_measure in ("tail_gini", "tail_gini_range"):
        tail = dispersion_model.get_owa_tail_gini_weights(periods, alpha)
        coefficients = -tail + (tail[::-1] if risk_measure == "tail_gini_range" else 0)
    else:
        raise InvalidArgumentError(
            f"The risk measure must be one of {', '.join(OWA_RISK_MEASURES)}, not "
            f"'{risk_measure}'."
        )
    subgradient = coefficients @ data[np.argsort(data @ weights, kind="stable")]
    return float(subgradient @ weights), subgradient


def get_mean_risk_measure(
    returns: pd.DataFrame,
    risk_measure: str,
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
) -> pd.Series:
    """
    A risk measure of the mean-risk and risk budgeting optimizers for every column of
    returns (each an asset or portfolio).

    It is per period and positive for a loss, from the Risk module's measures: the mean
    absolute deviation, the semideviation (the downside deviation below τ, or below the mean
    with T - 1), the first lower partial moment, the worst realization, the Gini mean
    difference, the Tail Gini, the range, the Tail Gini range and the Ulcer Index and the
    maximum and average drawdown of the uncompounded cumulative returns. The CVaR, the CVaR
    range and the CDaR are the averages of the tail with a fraction of the boundary period
    (Rockafellar and Uryasev, 2002), the values the linear programmes minimize, where the
    Risk module's CVaR averages the returns at or below an interpolated quantile.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset or portfolio.
        risk_measure (str): The risk measure. Options are: "mad" (mean absolute deviation, Konno and
            Yamazaki, 1991), "semideviation" (below the minimum acceptable return),
            "lower_partial_moment" (first order), "worst_realization" (the worst return),
            "gini_mean_difference" (Yitzhaki, 1982), "tail_gini" (Furman et al., 2017), "range" (the
            best minus the worst return), "cvar_range" (the CVaR of the losses and of the gains),
            "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the
            uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar"
            (Chekhlov et al., 2005), "maximum_drawdown" (of the uncompounded cumulative return),
            "average_drawdown" (of the uncompounded cumulative return).
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold of the
            semideviation and lower partial moment per period, or "mean". Defaults to 0.

    Returns:
        pd.Series: The risk measure per column.
    """
    returns = returns.dropna(how="any")
    if risk_measure == "mad":
        measure = (returns - returns.mean()).abs().mean()
    elif risk_measure == "semideviation":
        measure = risk_model.get_downside_deviation(returns, minimum_acceptable_return)
    elif risk_measure == "lower_partial_moment":
        measure = risk_model.get_partial_moment(
            returns, order=1, minimum_acceptable_return=minimum_acceptable_return
        )
    elif risk_measure == "worst_realization":
        measure = -dispersion_model.get_worst_realization(returns)
    elif risk_measure == "gini_mean_difference":
        measure = dispersion_model.get_gini_mean_difference(returns)
    elif risk_measure == "tail_gini":
        measure = -dispersion_model.get_tail_gini(returns, alpha)
    elif risk_measure == "range":
        measure = dispersion_model.get_risk_range(returns, "range")
    elif risk_measure == "tail_gini_range":
        measure = dispersion_model.get_risk_range(returns, "tail_gini", alpha)
    elif risk_measure == "cvar":
        measure = (-returns).apply(get_expected_tail_loss, alpha=alpha)
    elif risk_measure == "cvar_range":
        measure = (-returns).apply(get_expected_tail_loss, alpha=alpha) + returns.apply(
            get_expected_tail_loss, alpha=alpha
        )
    elif risk_measure == "cdar":
        measure = get_drawdowns(returns).apply(get_expected_tail_loss, alpha=alpha)
    elif risk_measure == "maximum_drawdown":
        measure = get_drawdowns(returns).max()
    elif risk_measure == "average_drawdown":
        measure = get_drawdowns(returns).mean()
    elif risk_measure == "ulcer_index":
        measure = risk_model.get_ui(returns, rolling=None, compound_returns=False)
    else:
        raise InvalidArgumentError(
            f"The risk measure must be one of {', '.join(MEAN_RISK_MEASURES)}, not "
            f"'{risk_measure}'."
        )
    return pd.Series(measure, index=returns.columns, dtype=float)


def get_mean_risk_contributions(
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_measure: str,
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
) -> pd.Series:
    """
    The share of a mean-risk or risk budgeting measure (see get_mean_risk_measure) that each
    asset contributes.

    It is w_i ∂ρ/∂w_i / ρ(w) (Euler allocation, Tasche, 2000), with the derivative found by
    central finite differences with a step of 1e-7, as in
    get_finite_difference_risk_contributions. The shares add up to one for the measures that
    are positively homogeneous in the weights, which are those of the risk budgeting
    portfolios (with a minimum acceptable return of zero or the mean).

    Args:
        weights (pd.Series): The weight per asset.
        returns (pd.DataFrame): The returns per period, one column per asset.
        risk_measure (str): The risk measure. Options are: "mad" (mean absolute deviation, Konno and
            Yamazaki, 1991), "semideviation" (below the minimum acceptable return),
            "lower_partial_moment" (first order), "worst_realization" (the worst return),
            "gini_mean_difference" (Yitzhaki, 1982), "tail_gini" (Furman et al., 2017), "range" (the
            best minus the worst return), "cvar_range" (the CVaR of the losses and of the gains),
            "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the
            uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar"
            (Chekhlov et al., 2005), "maximum_drawdown" (of the uncompounded cumulative return),
            "average_drawdown" (of the uncompounded cumulative return).
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold of the
            semideviation and lower partial moment per period, or "mean". Defaults to 0.

    Returns:
        pd.Series: The share of the measure per asset.
    """
    returns = returns.dropna(how="any")
    weights = weights.reindex(returns.columns).fillna(0).to_numpy(dtype=float)
    assets = len(weights)
    # Every column holds the weights with one asset's weight moved up or down a step,
    # and the last the weights themselves.
    shifts = np.hstack([np.eye(assets), -np.eye(assets), np.zeros((assets, 1))])
    measures = get_mean_risk_measure(
        pd.DataFrame(
            returns.to_numpy(dtype=float)
            @ (weights[:, None] + shifts * FINITE_DIFFERENCE_STEP),
            index=returns.index,
        ),
        risk_measure,
        alpha,
        minimum_acceptable_return,
    ).to_numpy()
    marginal = (measures[:assets] - measures[assets:-1]) / (2 * FINITE_DIFFERENCE_STEP)
    return pd.Series(weights * marginal / measures[-1], index=returns.columns)


def get_mean_risk_weights(
    returns: pd.DataFrame,
    risk_measure: str = "mad",
    objective: str = "minimum_risk",
    target_return: float | None = None,
    risk_free_rate: float = 0.0,
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
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
    A mean-risk portfolio over the historical returns for a risk measure ρ other than the
    variance (Markowitz, 1959; Konno and Yamazaki, 1991).

    It is the portfolio with the lowest risk, or with the best return per unit of that risk.
    With the minimum risk objective it is the portfolio with the lowest ρ(w), or with a
    target return the lowest ρ(w) that expects that return, and with the maximum ratio
    objective the portfolio with the highest (r̄ - r_f)'w / ρ(w).

    The linear measures (see get_risk_measure_programme) are solved as linear programmes
    with HiGHS, the ratio with the Charnes-Cooper transformation y = κw, κ > 0: maximize
    (r̄ - r_f)'y subject to ρ(y) ≤ 1 and the weight constraints in homogeneous form, with
    a minimum acceptable return τ scaled to τκ, then w = y / κ. The ordered weighted
    averages add the cuts of get_owa_risk with get_cutting_plane_solution, and for the
    ratio κ ≤ 2 / ρ_min with ρ_min the lowest risk within the constraints, as κρ(w) ≤ 1
    bounds κ by 1 / ρ(w) for every feasible w. The semideviation and
    the Ulcer Index (see get_squared_risk) are minimized, or the ratio to them maximized,
    with SLSQP directly in the weights.

    With the CVaR the minimum risk portfolio is that of get_minimum_cvar_weights and the
    maximum ratio portfolio that of get_maximum_cvar_ratio_weights; with the CDaR and the
    maximum and average drawdown the minimum risk portfolio is that of
    get_minimum_drawdown_weights with the same measure.

    Referenced against:
    - skfolio, MeanRisk: https://skfolio.org/generated/skfolio.optimization.MeanRisk.html
    - Riskfolio-Lib, Portfolio.optimization: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization
    - Riskfolio-Lib, Portfolio.owa_optimization (Tail Gini): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.owa_optimization

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        risk_measure (str, optional): The risk measure. Options are: "mad" (mean absolute deviation,
            Konno and Yamazaki, 1991), "semideviation" (below the minimum acceptable return),
            "lower_partial_moment" (first order), "worst_realization" (the worst return),
            "gini_mean_difference" (Yitzhaki, 1982), "tail_gini" (Furman et al., 2017), "range" (the
            best minus the worst return), "cvar_range" (the CVaR of the losses and of the gains),
            "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the
            uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar"
            (Chekhlov et al., 2005), "maximum_drawdown" (of the uncompounded cumulative return),
            "average_drawdown" (of the uncompounded cumulative return). Defaults to "mad".
        objective (str, optional): Options are: "minimum_risk" (the lowest risk), "maximum_ratio"
            (the highest excess return per unit of risk). Defaults to "minimum_risk".
        target_return (float | None, optional): The expected return per period the
            minimum risk portfolio must have. Defaults to None, no target.
        risk_free_rate (float, optional): The risk free rate per period of the ratio.
            Defaults to 0.
        alpha (float, optional): The tail probability of the CVaR, the CVaR range, the
            CDaR and the Tail Gini measures, e.g. 0.05 for the worst 5%. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold per period of
            the semideviation and the lower partial moment, or "mean" for each asset's
            mean. Defaults to 0.
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
            portfolio. Holding nothing has no drawdown, so a net exposure of 0 is refused
            for the drawdown measures. Defaults to 1.
        gross_exposure (float | None, optional): The highest sum of the absolute weights,
            e.g. 1.6 for a 130/30 portfolio. Defaults to None, no limit.
        current_weights (pd.Series | None, optional): The weights held now, which the
            turnover is measured from. Defaults to None.
        maximum_turnover (float | None, optional): The highest turnover Σ |w_i - w0_i|
            from the current weights. Defaults to None, no limit.
        expected_returns (pd.Series | None, optional): The expected return per period of
            every asset, for the ratio and the target return. Defaults to None, the
            average of the returns.

    Returns:
        pd.Series: The weight per asset.
    """
    check_option("risk_measure", risk_measure, MEAN_RISK_MEASURES)
    check_option("objective", objective, MEAN_RISK_OBJECTIVES)
    if target_return is not None and objective != "minimum_risk":
        raise ValueError("A target return applies to the minimum risk objective only.")
    validate_alpha(alpha)
    if (
        isinstance(minimum_acceptable_return, str)
        and minimum_acceptable_return != "mean"
    ):
        raise ValueError("The minimum acceptable return must be a number or 'mean'.")

    data = returns.dropna(how="any").to_numpy(dtype=float)
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
    mean = (
        data.mean(axis=0)
        if expected_returns is None
        else expected_returns.reindex(returns.columns).to_numpy(dtype=float)
    )
    if np.isnan(mean).any():
        raise ValueError("The expected returns must cover every asset.")
    excess = mean - risk_free_rate
    if objective == "maximum_ratio" and _return_range(excess, constraints)[1] <= 0:
        raise ValueError(
            "No portfolio within the constraints has an expected return above the risk "
            "free rate, so none has a positive return to risk ratio."
        )
    if target_return is not None:
        lowest, highest, _ = _return_range(mean, constraints)
        if not lowest - 1e-12 <= target_return <= highest + 1e-12:
            raise ValueError(
                f"A target return of {target_return:.6f} per period cannot be reached "
                f"within the constraints ({lowest:.6f} to {highest:.6f})."
            )

    if risk_measure in QUADRATIC_RISK_MEASURES:
        # The returns divided by their largest absolute value keep the squared measure of
        # order one for SLSQP's absolute ftol; the optimal weights are the same.
        scaled = data / (np.max(np.abs(data)) or 1.0)
        threshold = (
            minimum_acceptable_return
            if minimum_acceptable_return == "mean"
            else float(minimum_acceptable_return) / (np.max(np.abs(data)) or 1.0)
        )

        def squared(weights):
            return get_squared_risk(scaled, weights, risk_measure, threshold)

        if objective == "minimum_risk":
            weights = _solve(
                lambda w: squared(w)[0],
                lambda w: squared(w)[1],
                constraints,
                (
                    [_target_return(mean, target_return)]
                    if target_return is not None
                    else None
                ),
            )
        else:

            def negative_ratio(w):
                return -(w @ excess) / np.sqrt(squared(w)[0])

            def ratio_gradient(w):
                value, slope = squared(w)
                risk = np.sqrt(value)
                return -(excess * risk - (w @ excess) * slope / (2 * risk)) / value

            weights = _solve(
                negative_ratio, ratio_gradient, constraints, direction=excess
            )
        return _as_series(weights, returns.columns)

    programme = get_risk_measure_programme(
        data, risk_measure, alpha, minimum_acceptable_return
    )
    scaled = data / (np.max(np.abs(data)) or 1.0)
    ratio = objective == "maximum_ratio"
    # The variables are the weights (y = κw for the ratio) with the absolute weight and
    # trade variables of the constraints, κ for the ratio, then those of the programme.
    variables = len(constraints.bounds)
    measure = variables + ratio
    width = measure + len(programme.cost)
    equality_rows = programme.equality_weights.shape[0]

    def stack(rows, block, kappa=None):
        """Rows over the weights and their variables, κ and the programme's variables."""
        columns = [sparse.csr_matrix(rows)]
        if ratio:
            columns.append(
                sparse.csr_matrix((rows.shape[0], 1) if kappa is None else kappa)
            )
        return sparse.hstack([*columns, block], format="csr")

    def owa_cut(weights):
        """The cut θ ≥ g'w of an ordered weighted average at the weights."""
        slope = get_owa_risk(scaled, weights, risk_measure, alpha)[1]
        return np.concatenate([slope, np.zeros(width - assets - 1), [-1.0]])

    def cuts(x):
        if risk_measure not in OWA_RISK_MEASURES:
            return []
        return [(owa_cut(x[:assets]), 0.0)]

    def feasible(x):
        """
        The solution with θ the measure itself and, for the ratio, scaled down to
        ρ(y) ≤ 1, which the homogeneous constraints allow.
        """
        risk = get_owa_risk(scaled, x[:assets], risk_measure, alpha)[0]
        if ratio:
            if risk <= 0:
                return None
            factor = min(1.0, 1 / risk)
            x, risk = x * factor, risk * factor
        return np.append(x[:-1], risk)

    stabilization = feasible if risk_measure in OWA_RISK_MEASURES else None

    measure_block = sparse.hstack(
        [
            programme.weights,
            sparse.csr_matrix((len(programme.upper), variables - assets)),
        ]
    )
    starting_cuts = (
        np.array([owa_cut(np.full(assets, 1 / assets))])
        if risk_measure in OWA_RISK_MEASURES
        else np.zeros((0, width))
    )
    programme_equality = sparse.hstack(
        [
            programme.equality_weights,
            sparse.csr_matrix((equality_rows, variables - assets)),
        ]
    )

    if not ratio:
        inequality = sparse.vstack(
            [
                stack(measure_block, programme.auxiliary),
                sparse.csr_matrix(starting_cuts),
                stack(
                    constraints.inequality,
                    sparse.csr_matrix(
                        (len(constraints.inequality), len(programme.cost))
                    ),
                ),
            ],
            format="csr",
        )
        equality = [
            stack(constraints.equality, sparse.csr_matrix((1, len(programme.cost)))),
            stack(programme_equality, programme.equality_auxiliary),
        ]
        targets = [constraints.target, np.zeros(equality_rows)]
        if target_return is not None:
            equality.append(
                sparse.csr_matrix(
                    np.concatenate([mean, np.zeros(width - assets)])[None]
                )
            )
            targets.append(np.array([target_return]))
        solution = get_cutting_plane_solution(
            np.concatenate([np.zeros(variables), programme.cost]),
            inequality,
            np.concatenate(
                [programme.upper, np.zeros(len(starting_cuts)), constraints.upper]
            ),
            sparse.vstack(equality, format="csr"),
            np.concatenate(targets),
            constraints.bounds + programme.bounds,
            cuts,
            stabilization,
        )
        if risk_measure in (
            "cdar",
            "maximum_drawdown",
            "average_drawdown",
        ) and np.allclose(solution[:assets], 0, atol=ZERO_WEIGHT):
            raise ValueError(
                "Holding nothing has no drawdown, so the minimum is the empty portfolio. This "
                "happens with a net exposure of 0; use a net exposure other than 0."
            )
        return _as_series(solution[:assets], returns.columns)

    # κ is at most 1 / ρ(w) for every feasible w, so at most one over the lowest risk. The
    # linear programmes of the other measures are bounded without it, but the cuts of an
    # ordered weighted average only bound the risk once enough of them are found.
    largest_kappa = None
    if risk_measure in OWA_RISK_MEASURES:
        lowest = get_owa_risk(
            scaled,
            get_mean_risk_weights(
                returns,
                risk_measure,
                alpha=alpha,
                minimum_weight=minimum_weight,
                maximum_weight=maximum_weight,
                groups=groups,
                group_bounds=group_bounds,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                current_weights=current_weights,
                maximum_turnover=maximum_turnover,
            ).to_numpy(),
            risk_measure,
            alpha,
        )[0]
        if lowest <= 0:
            raise ValueError(
                "A portfolio whose risk is zero or negative has an unbounded return to "
                "risk ratio."
            )
        largest_kappa = 2 / lowest

    homogeneous, homogeneous_equality = _homogeneous_constraints(constraints)
    inequality = sparse.vstack(
        [
            stack(
                measure_block,
                programme.auxiliary,
                sparse.csr_matrix(-programme.upper[:, None]),
            ),
            sparse.csr_matrix(starting_cuts),
            sparse.hstack(
                [
                    sparse.csr_matrix(homogeneous),
                    sparse.csr_matrix((len(homogeneous), len(programme.cost))),
                ]
            ),
            sparse.csr_matrix(
                np.concatenate([np.zeros(measure), programme.cost])[None]
            ),
        ],
        format="csr",
    )
    upper = np.zeros(inequality.shape[0])
    upper[-1] = 1.0
    equality = sparse.vstack(
        [
            sparse.hstack(
                [
                    sparse.csr_matrix(homogeneous_equality),
                    sparse.csr_matrix((1, len(programme.cost))),
                ]
            ),
            stack(programme_equality, programme.equality_auxiliary),
        ],
        format="csr",
    )
    solution = get_cutting_plane_solution(
        np.concatenate([-excess, np.zeros(width - assets)]),
        inequality,
        upper,
        equality,
        np.zeros(equality.shape[0]),
        [(None, None)] * variables + [(0, largest_kappa)] + programme.bounds,
        cuts,
        stabilization,
        unbounded="A portfolio whose risk is zero or negative has an unbounded return to risk "
        "ratio.",
    )
    return _as_series(solution[:assets] / solution[variables], returns.columns)


def get_risk_budgeting_weights(
    returns: pd.DataFrame,
    risk_measure: str = "cvar",
    risk_budget: pd.Series | None = None,
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    groups: pd.DataFrame | None = None,
    group_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> pd.Series:
    """
    The risk budgeting portfolio for a risk measure ρ of the historical returns that is
    positively homogeneous of degree one and convex.

    Every asset contributes the share b_i of the risk set by the budget, w_i ∂ρ/∂w_i = b_i
    ρ(w) (Roncalli, 2013). It is the solution of min ρ(y) subject to Σ b_i ln y_i ≥ 0 and y
    ≥ 0, scaled to w = y / 1'y: the first-order conditions y_i ∂ρ/∂y_i = λ b_i give the
    budget (Bruder and Roncalli, 2012).

    For the linear measures ρ is the linear programme of get_risk_measure_programme and
    the logarithmic constraint is met by cutting planes (Kelley, 1960): the concave
    Σ b_i ln y_i lies below its tangent at any point p with Σ b_i ln p_i = 0, so
    Σ (b_i / p_i) y_i ≥ 1 holds for every feasible y. The ordered weighted averages
    (the Gini mean difference and the Tail Gini measures) enter as the cuts θ ≥ g'y of
    get_owa_risk as well, with 1'y capped by ρ(1) / μ for μ the lowest risk of a fully
    invested portfolio, since ρ(y) ≥ μ 1'y. See get_cutting_plane_solution: the result is
    the exact optimum of the convex programme. The semideviation and the Ulcer Index are
    smooth enough to minimize ρ(y) - Σ b_i ln y_i with SLSQP, as the risk parity weights
    minimize the volatility (Spinu, 2013). Weight and group bounds are homogeneous
    constraints on y, as in get_risk_parity_weights.

    See Roncalli, T. (2013), "Introduction to Risk Parity and Budgeting", Chapman &
    Hall/CRC; Bruder, B. and Roncalli, T. (2012), "Managing Risk Exposures Using the Risk
    Budgeting Approach", SSRN 2009778; and Kelley, J. E. (1960), "The Cutting-Plane
    Method for Solving Convex Programs", Journal of the Society for Industrial and
    Applied Mathematics, 8(4), 703-712.

    Referenced against:
    - skfolio, RiskBudgeting: https://skfolio.org/generated/skfolio.optimization.RiskBudgeting.html
    - Riskfolio-Lib, Portfolio.rp_optimization: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.rp_optimization

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        risk_measure (str, optional): The risk measure. Options are: "mad" (mean absolute deviation,
            Konno and Yamazaki, 1991), "semideviation" (below the minimum acceptable return),
            "lower_partial_moment" (first order), "gini_mean_difference" (Yitzhaki, 1982),
            "tail_gini" (Furman et al., 2017), "cvar_range" (the CVaR of the losses and of the
            gains), "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index"
            (of the uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar"
            (Chekhlov et al., 2005). Defaults to "cvar".
        risk_budget (pd.Series | None, optional): The share of the risk each asset should
            contribute, positive for every asset; it is scaled to sum to one. A budget for
            an asset outside the portfolio, or an asset without one, raises a ValueError
            that names them. Defaults to None, an equal share each.
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold of the semideviation and
            lower partial moment; any other threshold does not scale with the weights. Options are:
            0 (a return of zero), "mean" (each asset's mean). Defaults to 0.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset. Defaults to 1.
        groups (pd.DataFrame | None, optional): The group of each asset per
            classification. Defaults to None.
        group_bounds (dict[str, dict[str, tuple[float, float]]] | None, optional): Per
            classification in groups, the minimum and maximum total weight per group.
            Defaults to None.

    Returns:
        pd.Series: The weight per asset.
    """
    check_option("risk_measure", risk_measure, RISK_BUDGETING_MEASURES)
    validate_alpha(alpha)
    check_option("minimum_acceptable_return", minimum_acceptable_return, (0, "mean"))
    data = returns.dropna(how="any").to_numpy(dtype=float)
    periods, assets = data.shape
    if periods < 2:  # noqa: PLR2004
        raise ValueError(
            "At least two periods with a return for every asset are needed."
        )
    budget = get_risk_budget(risk_budget, returns.columns)

    constraints = get_weight_constraints(
        returns.columns, minimum_weight, maximum_weight, groups, group_bounds
    )
    lower = np.array([bound[0] for bound in constraints.bounds])
    upper = np.array([bound[1] for bound in constraints.bounds])
    if np.any(lower < 0):
        raise ValueError(
            "Risk budgeting holds long positions only; the minimum weight cannot be "
            "negative."
        )
    # Every constraint a'w ≤ b on the weights becomes (a - b 1)'y ≤ 0 on y, with w = y / 1'y.
    ones = np.ones(assets)
    homogeneous = np.vstack(
        [
            constraints.inequality - constraints.upper[:, None] * ones,
            np.eye(assets) - upper[:, None] * ones,
            lower[:, None] * ones - np.eye(assets),
        ]
    )

    if risk_measure in QUADRATIC_RISK_MEASURES:
        scaled = data / (np.max(np.abs(data)) or 1.0)

        def objective(y):
            return np.sqrt(
                get_squared_risk(scaled, y, risk_measure, minimum_acceptable_return)[0]
            ) - budget @ np.log(y)

        def gradient(y):
            value, slope = get_squared_risk(
                scaled, y, risk_measure, minimum_acceptable_return
            )
            return slope / (2 * np.sqrt(value)) - budget / y

        # At the optimum ρ(y) equals Σ b_i = 1, so equal weights scaled to a risk of one
        # start close to it.
        start = ones / np.sqrt(
            get_squared_risk(scaled, ones, risk_measure, minimum_acceptable_return)[0]
        )
        positive = [(1e-12, None)] * assets
        linear = [
            {
                "type": "ineq",
                "fun": lambda y: -(homogeneous @ y),
                "jac": lambda y: -homogeneous,
            }
        ]
        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=positive,
            constraints=linear,
            options=OPTIMIZER_OPTIONS,
        )
        if not result.success and not (
            result.status == SLSQP_LINE_SEARCH_STATUS
            and _is_feasible(result.x, positive, linear)
        ):
            raise ValueError(f"The optimization did not converge: {result.message}")
        return _as_series(result.x / result.x.sum(), returns.columns)

    programme = get_risk_measure_programme(
        data, risk_measure, alpha, minimum_acceptable_return
    )
    scaled = data / (np.max(np.abs(data)) or 1.0)
    auxiliary = len(programme.cost)
    owa = risk_measure in OWA_RISK_MEASURES

    def tangent(point):
        """
        The tangent Σ (b_i / p_i) y_i ≥ 1 of Σ b_i ln y_i ≥ 0 at the point scaled onto the
        boundary Σ b_i ln p_i = 0, the tightest cut in its direction.

        A weight of zero has no tangent, so it is taken just above zero.
        """
        point = np.maximum(point, ZERO_WEIGHT * point.max())
        point = point * np.exp(-budget @ np.log(point))
        return np.concatenate([-budget / point, np.zeros(auxiliary)]), -1.0

    def owa_cut(point):
        """The cut θ ≥ g'y of an ordered weighted average at the point."""
        slope = get_owa_risk(scaled, point, risk_measure, alpha)[1]
        return np.concatenate([slope, [-1.0]]), 0.0

    def cuts(x):
        return [tangent(x[:assets])] + ([owa_cut(x[:assets])] if owa else [])

    def feasible(x):
        """
        The solution scaled up onto Σ b_i ln y_i ≥ 0, which keeps the programme's
        variables feasible as its rows are homogeneous, with θ the measure itself.
        """
        if np.any(x[:assets] <= 0):
            return None
        x = x * max(1.0, np.exp(-budget @ np.log(x[:assets])))
        if owa:
            x[assets] = get_owa_risk(scaled, x[:assets], risk_measure, alpha)[0]
        return x

    starting = [tangent(ones)] + ([owa_cut(ones)] if owa else [])
    rows = [
        sparse.hstack([programme.weights, programme.auxiliary]),
        sparse.hstack(
            [
                sparse.csr_matrix(homogeneous),
                sparse.csr_matrix((len(homogeneous), auxiliary)),
            ]
        ),
        sparse.csr_matrix(np.array([row for row, _ in starting])),
    ]
    limits = [
        programme.upper,
        np.zeros(len(homogeneous)),
        np.array([limit for _, limit in starting]),
    ]
    equality = sparse.hstack(
        [programme.equality_weights, programme.equality_auxiliary], format="csr"
    )
    unbounded = (
        "Risk budgeting needs a risk measure that is positive for every portfolio."
    )
    if not owa:
        solution = get_cutting_plane_solution(
            np.concatenate([np.zeros(assets), programme.cost]),
            sparse.vstack(rows, format="csr"),
            np.concatenate(limits),
            equality if equality.shape[0] else None,
            np.zeros(equality.shape[0]) if equality.shape[0] else None,
            [(0.0, None)] * assets + programme.bounds,
            cuts,
            feasible,
            unbounded=unbounded,
        )
        return _as_series(solution[:assets] / solution[:assets].sum(), returns.columns)

    # The cuts of an ordered weighted average bound the risk only once enough of them are
    # found, so 1'y is capped: ρ(y) ≥ μ 1'y with μ the lowest risk of a fully invested
    # long-only portfolio, which bounds 1'y at the optimum by its risk over μ. A cap that
    # still binds at the solution is raised tenfold and the programme solved again, since
    # an optimum that does not touch the cap is the optimum without it.
    lowest = get_owa_risk(
        scaled,
        get_mean_risk_weights(returns, risk_measure, alpha=alpha).to_numpy(),
        risk_measure,
        alpha,
    )[0]
    if lowest <= 0:
        raise ValueError(unbounded)
    cap = 2 * get_owa_risk(scaled, ones, risk_measure, alpha)[0] / lowest
    for _ in range(RISK_BUDGETING_CAP_RAISES):
        solution = get_cutting_plane_solution(
            np.concatenate([np.zeros(assets), programme.cost]),
            sparse.vstack(
                [*rows, sparse.csr_matrix(np.append(ones, 0.0)[None])], format="csr"
            ),
            np.concatenate([*limits, [cap]]),
            None,
            None,
            [(0.0, None)] * assets + programme.bounds,
            cuts,
            feasible,
            unbounded=unbounded,
        )
        if solution[:assets].sum() < cap * (1 - FEASIBILITY_TOLERANCE):
            return _as_series(
                solution[:assets] / solution[:assets].sum(), returns.columns
            )
        cap *= 10
    raise ValueError(unbounded)


def get_random_weights(
    assets: pd.Index, seed: int | np.random.Generator | None = None
) -> pd.Series:
    """
    Random weights drawn uniformly from all fully invested long-only portfolios.

    They follow a flat Dirichlet distribution with every concentration parameter 1 (Rubin,
    1981), the baseline an optimizer should beat out of sample. See Rubin, D. B. (1981),
    "The Bayesian Bootstrap", The Annals of Statistics, 9(1), 130-134.

    Shares similar logic:
    - skfolio, Random: https://skfolio.org/generated/skfolio.optimization.Random.html

    Args:
        assets (pd.Index): The assets of the portfolio.
        seed (int | np.random.Generator | None, optional): The seed or generator of the
            draw. Defaults to None, a different draw every time.

    Returns:
        pd.Series: The weight per asset.
    """
    weights = np.random.default_rng(seed).dirichlet(np.ones(len(assets)))
    return _as_series(weights, assets)


# The risk measures of the hierarchical optimizers: the variance and volatility of the
# covariance matrix and the measures of the mean-risk optimizer on the returns.
HIERARCHICAL_RISK_MEASURES = {
    "variance": "Variance",
    "volatility": "Volatility",
    **{
        measure: name
        for measure, name in MEAN_RISK_MEASURES.items()
        if measure not in ("maximum_drawdown", "average_drawdown")
    },
}

# How the number of clusters is chosen when it is not given.
CLUSTER_CRITERIA = ("two_difference_gap", "standardized_silhouette", "onc")


def get_hierarchical_risk(
    weights: np.ndarray,
    covariance: np.ndarray,
    returns: np.ndarray | None = None,
    risk_measure: str = "variance",
    alpha: float = 0.05,
    minimum_acceptable_return: float | str = 0.0,
) -> np.ndarray:
    """
    The risk of portfolios by which the hierarchical optimizers split their weight.

    It is the variance w'Σw or the volatility √(w'Σw) of the covariance matrix, or a measure
    of get_mean_risk_measure on the portfolio's returns X_t = r_t'w (per period, positive
    for a loss), the measures Raffinot (2018) allocates HERC by.

    Args:
        weights (np.ndarray): The N × K weights of K portfolios, one column each.
        covariance (np.ndarray): The N × N covariance matrix.
        returns (np.ndarray | None, optional): The T × N returns, without missing values,
            for the measures besides the variance and volatility. Defaults to None.
        risk_measure (str, optional): The risk measure. Options are: "variance" (of the covariance
            matrix), "volatility" (of the covariance matrix), "mad" (mean absolute deviation, Konno
            and Yamazaki, 1991), "semideviation" (below the minimum acceptable return),
            "lower_partial_moment" (first order), "worst_realization" (the worst return),
            "gini_mean_difference" (Yitzhaki, 1982), "tail_gini" (Furman et al., 2017), "range" (the
            best minus the worst return), "cvar_range" (the CVaR of the losses and of the gains),
            "tail_gini_range" (the Tail Gini of the losses and of the gains), "ulcer_index" (of the
            uncompounded cumulative return), "cvar" (Rockafellar and Uryasev, 2002), "cdar"
            (Chekhlov et al., 2005). Defaults to "variance".
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.
        minimum_acceptable_return (float | str, optional): The threshold of the
            semideviation and lower partial moment per period, or "mean". Defaults to 0.

    Returns:
        np.ndarray: The risk of every portfolio.
    """
    check_option("risk_measure", risk_measure, HIERARCHICAL_RISK_MEASURES)
    if risk_measure in ("variance", "volatility"):
        variance = np.einsum("ik,ij,jk->k", weights, covariance, weights)
        return variance if risk_measure == "variance" else np.sqrt(variance)
    if returns is None:
        raise ValueError(f"The risk measure '{risk_measure}' needs the returns.")
    return get_mean_risk_measure(
        pd.DataFrame(returns @ weights),
        risk_measure,
        alpha,
        minimum_acceptable_return,
    ).to_numpy()


def get_hierarchical_order(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    correlation: pd.DataFrame | None = None,
    distance: pd.DataFrame | None = None,
    similarity: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    The order of the assets that Hierarchical Risk Parity bisects (López de Prado, 2016,
    Stages 1 and 2).

    These are the leaves of the hierarchical clustering on the Euclidean distance between
    the columns of the correlation distance matrix d = √((1 - ρ) / 2), so that similar
    assets sit next to each other. With the Directed Bubble Hierarchical Tree as the
    linkage, the leaves of the Directed Bubble Hierarchical Tree of the distance d itself
    and the similarity (see get_dbht_linkage), as the DBHT works on the graph of the
    distances rather than on a distance between their columns.

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        linkage_method (str, optional): The linkage of the hierarchical clustering. Options are:
            "single" (single linkage), "complete" (complete linkage), "average" (average linkage),
            "ward" (Ward's method), "dbht" (the Directed Bubble Hierarchical Tree of Song, Di Matteo
            and Aste, 2012). Defaults to "single".
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on. Defaults to None, the correlation matrix of the covariance
            matrix.
        distance (pd.DataFrame | None, optional): The distance matrix the assets are
            clustered on, in place of the correlation distance. Defaults to None.
        similarity (pd.DataFrame | None, optional): The similarity matrix of the Directed Bubble
            Hierarchical Tree. Defaults to None, 1 - d² of the distance.

    Returns:
        np.ndarray: The positions of the assets in the order of the leaves.
    """
    if distance is not None:
        matrix = distance.loc[covariance.index, covariance.index].to_numpy()
    else:
        if correlation is None:
            volatility = np.sqrt(np.diag(covariance.to_numpy()))
            correlation = covariance / np.outer(volatility, volatility)
        # The correlation distance of López de Prado, sqrt((1 - rho) / 2).
        matrix = np.sqrt(
            np.clip(
                (1 - correlation.loc[covariance.index, covariance.index].to_numpy())
                / 2,
                0,
                None,
            )
        )
        np.fill_diagonal(matrix, 0)
    check_option("linkage_method", linkage_method, LINKAGE_METHODS)
    if linkage_method == "dbht":
        return leaves_list(
            get_dbht_linkage(
                matrix,
                (
                    None
                    if similarity is None
                    else similarity.loc[covariance.index, covariance.index]
                ),
            )
        )
    # The Euclidean distance between the columns of the distance matrix that the paper
    # clusters on (the paper's listing passes the square matrix to linkage, which reads the
    # rows as observations and does exactly this).
    return leaves_list(linkage(pdist(matrix), method=linkage_method))


def get_standardized_silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    """
    The standardized silhouette of a clustering: how well separated its clusters are.

    It is the quality score q = E[S] / √V[S] of López de Prado (2020, chapter 4) over the
    silhouette coefficients of Rousseeuw (1987), S_i = (b_i - a_i) / max(a_i, b_i), with a_i
    the average distance of asset i to the other assets of its cluster and b_i the smallest
    average distance to the assets of another cluster (S_i = 0 for an asset alone in its
    cluster). As in the book, the assets are the rows of the distance matrix, so the
    distances are the Euclidean distances between those rows, and the variance has no
    degrees of freedom correction.

    See López de Prado, M. (2020), Machine Learning for Asset Managers, Cambridge
    University Press, and Rousseeuw, P. J. (1987), "Silhouettes: A Graphical Aid to the
    Interpretation and Validation of Cluster Analysis", Journal of Computational and
    Applied Mathematics, 20, 53-65.

    Referenced against:
    - scikit-learn, silhouette_samples: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html

    Args:
        distance (np.ndarray): The N × N distance matrix the assets are clustered on.
        labels (np.ndarray): The cluster of every asset.

    Returns:
        float: The mean of the silhouette coefficients over their standard deviation.
    """
    silhouette = estimation_model.get_silhouette_coefficients(
        squareform(pdist(distance)), labels
    )
    return float(silhouette.mean() / silhouette.std())


def get_schur_complementary_weights(
    covariance: pd.DataFrame,
    gamma: float = 0.5,
    linkage_method: str = "single",
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
    correlation: pd.DataFrame | None = None,
    distance: pd.DataFrame | None = None,
    similarity: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Schur Complementary Allocation (Cotton, 2024): Hierarchical Risk Parity whose splits see
    the covariance between the two halves.

    HRP splits every cluster with the covariance blocks A and D of its two halves and
    ignores the block B between them, which the minimum variance portfolio w ∝ Σ⁻¹1 uses
    through the Schur complements of the block inverse. Cotton writes the minimum variance
    weights of each half as a scaled minimum variance portfolio of an augmented matrix and
    shrinks the step from A to it with γ:

    - A_c = A - γ B D⁻¹ B', the Schur complement of D scaled by γ.
    - b = 1 - γ B D⁻¹ M' 1, with M the n_A × n_D step-up matrix that spreads the n_D assets
      of the other half over the n_A of this one (M 1 = 1; the identity for equal halves,
      and for n_A = n_D + 1 the average of the identities with a row of 1 / n_D inserted
      at every position).
    - Ã = (I - γ B D⁻¹ M')⁻¹ A_c, made symmetric, so that Ã⁻¹1 = A_c⁻¹ b, the minimum
      variance weights of the half up to a scale (the collapse of Cotton's appendix A).
    - The split is that of HRP on the augmented blocks, α = 1 - ν(Ã) / (ν(Ã) + ν(D̃)),
      with ν the variance of the inverse variance portfolio of a block, and the halves
      are split further on Ã and D̃.

    With γ = 0 the blocks are not augmented and the weights are those of Hierarchical
    Risk Parity; as γ rises towards 1 they move towards the minimum variance portfolio.
    A split of a cluster with a half of one asset uses no augmentation, as both halves
    are then a single asset or the augmentation has nothing to spread.

    The correction as a matrix, with the step-up matrix for halves of unequal size, is the
    form of Cotton's precise package.

    See Cotton, P. (2024), "Schur Complementary Allocation: A Unification of
    Hierarchical Risk Parity and Minimum Variance Portfolios", arXiv 2411.05807.

    Referenced against:
    - skfolio, SchurComplementary: https://skfolio.org/generated/skfolio.optimization.SchurComplementary.html

    Args:
        covariance (pd.DataFrame): The covariance matrix of the assets.
        gamma (float, optional): How much of the covariance between the halves the splits
            use, from 0 (Hierarchical Risk Parity) to 1. Defaults to 0.5.
        linkage_method (str, optional): The linkage of the hierarchical clustering, as in
            Hierarchical Risk Parity. Options are: "single" (single linkage), "complete" (complete
            linkage), "average" (average linkage), "ward" (Ward's method), "dbht" (the Directed
            Bubble Hierarchical Tree of Song, Di Matteo and Aste, 2012). Defaults to "single".
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0). Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.
        correlation (pd.DataFrame | None, optional): The correlation matrix the assets
            are clustered on. Defaults to None, the correlation matrix of the covariance
            matrix.
        distance (pd.DataFrame | None, optional): The distance matrix the assets are
            clustered on, in place of the correlation distance. Defaults to None.
        similarity (pd.DataFrame | None, optional): The similarity matrix of the Directed Bubble
            Hierarchical Tree, e.g. from get_codependence_similarity. Defaults to None, 1 - d² of
            the distance, (1 + ρ) / 2 for the correlation distance.

    Returns:
        pd.Series: The weight per asset.

    Notes:
    - skfolio's SchurComplementary can also keep the weights monotonic in γ; these weights
      equal its weights without that adjustment.
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma must lie between 0 and 1, not {gamma}.")
    _validate_covariance(covariance)
    if len(covariance) < 2:  # noqa: PLR2004
        raise ValueError("At least two assets are needed.")
    constraints = get_weight_constraints(
        covariance.index, minimum_weight, maximum_weight
    )
    lower = np.array([bound[0] for bound in constraints.bounds])
    upper = np.array([bound[1] for bound in constraints.bounds])
    if (lower < 0).any():
        raise ValueError(
            "Schur Complementary Allocation holds long positions only; the minimum "
            "weight cannot be negative."
        )
    order = get_hierarchical_order(
        covariance, linkage_method, correlation, distance, similarity
    )
    # The blocks are augmented in place, so every later split works on the augmented
    # blocks of the cluster it splits.
    sigma = covariance.to_numpy(dtype=float, copy=True)

    def augment(own: np.ndarray, between: np.ndarray, other: np.ndarray) -> np.ndarray:
        size, other_size = len(own), len(other)
        if size == other_size:
            step_up = np.eye(size)
        else:
            larger, smaller = max(size, other_size), min(size, other_size)
            # Row i of the larger-by-smaller step-up matrix: 1 / smaller everywhere, plus
            # larger - 1 - i on column i and i on column i - 1, all over larger.
            rows = np.arange(larger)[:, None]
            columns = np.arange(smaller)[None, :]
            step_up = (
                1 / smaller
                + (larger - 1 - rows) * (columns == rows)
                + rows * (columns == rows - 1)
            ) / larger
            if size < other_size:
                step_up = step_up.T * size / other_size
        projection = gamma * np.linalg.solve(other, between.T).T
        augmented = np.linalg.solve(
            np.eye(size) - projection @ step_up.T, own - projection @ between.T
        )
        return (augmented + augmented.T) / 2

    def inverse_variance_risk(block: np.ndarray) -> float:
        inverse_variance = 1 / np.diag(block)
        weights = inverse_variance / inverse_variance.sum()
        return float(weights @ block @ weights)

    weights = np.ones(len(sigma))
    clusters = [order]
    while clusters:
        clusters = [
            half
            for cluster in clusters
            if len(cluster) > 1
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            own, other = sigma[np.ix_(left, left)], sigma[np.ix_(right, right)]
            if gamma > 0 and len(left) > 1:
                between = sigma[np.ix_(left, right)]
                own, other = (
                    augment(own, between, other),
                    augment(other, between.T, own),
                )
                for block in (own, other):
                    if np.any(np.linalg.eigvalsh(block) <= 0):
                        raise ValueError(
                            f"With a gamma of {gamma}, an augmented covariance block "
                            "is not positive definite; choose a smaller gamma or a "
                            "shrinkage covariance estimator."
                        )
                sigma[np.ix_(left, left)], sigma[np.ix_(right, right)] = own, other
            left_risk, right_risk = inverse_variance_risk(own), inverse_variance_risk(
                other
            )
            alpha = 1 - left_risk / (left_risk + right_risk)
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


def get_stacking_method_weights(
    window: pd.DataFrame,
    method: str,
    risk_free_rate: Callable[[pd.Index], float],
    periods_per_year: float,
    covariance: str,
    covariance_span: int,
    expected_returns: str,
    expected_returns_span: int,
    benchmark: pd.Series | None,
    factors: pd.DataFrame | None,
) -> pd.Series:
    """
    Set the weights of one optimizer of a stacking portfolio from the returns of a window,
    with the expected returns and covariance matrix estimated from that window.

    Args:
        window (pd.DataFrame): The returns of the window, one column per asset.
        method (str): The optimizer, such as "equal_weight", "minimum_variance" or
            "maximum_sharpe"; any other method sets hierarchical risk parity weights.
        risk_free_rate (Callable[[pd.Index], float]): Gives the annualized risk free rate over
            the dates of a window.
        periods_per_year (float): The number of return periods in a year.
        covariance (str): The covariance estimator.
        covariance_span (int): The span of an exponentially weighted covariance.
        expected_returns (str): The expected returns estimator.
        expected_returns_span (int): The span of exponentially weighted expected returns.
        benchmark (pd.Series | None): The benchmark returns, for the CAPM expected returns and
            the single factor covariance.
        factors (pd.DataFrame | None): The factor returns, for the factor covariance.

    Returns:
        pd.Series: The weight per asset.
    """
    if method == "equal_weight":
        return pd.Series(1 / window.shape[1], index=window.columns)
    if method == "minimum_cvar":
        return get_minimum_cvar_weights(window)
    if method == "minimum_drawdown":
        return get_minimum_drawdown_weights(window)
    if method in ("minimum_mad", "minimum_semideviation"):
        return get_mean_risk_weights(
            window, risk_measure=method.removeprefix("minimum_")
        )
    window_risk_free_rate = risk_free_rate(window.index)
    mu, sigma = get_expected_returns_and_covariance(
        window,
        periods_per_year,
        covariance,
        covariance_span,
        expected_returns,
        expected_returns_span,
        benchmark,
        window_risk_free_rate,
        factors,
    )
    if method == "minimum_variance":
        return get_minimum_variance_weights(sigma)
    if method == "maximum_sharpe":
        return get_maximum_sharpe_weights(mu, sigma, window_risk_free_rate)
    if method == "risk_parity":
        return get_risk_parity_weights(sigma)
    if method == "inverse_volatility":
        return get_inverse_volatility_weights(sigma)
    if method == "maximum_diversification":
        return get_maximum_diversification_weights(sigma)
    if method == "hierarchical_equal_risk_contribution":
        return get_hierarchical_equal_risk_contribution_weights(sigma)
    if method == "nested_clustered_optimization":
        return get_nested_clustered_optimization_weights(
            mu, sigma, risk_free_rate=window_risk_free_rate
        )
    if method == "schur_complementary":
        return get_schur_complementary_weights(sigma)
    return get_hierarchical_risk_parity_weights(sigma)


def get_stacking_weights(
    returns: pd.DataFrame,
    methods: dict[str, Callable[[pd.DataFrame], pd.Series]],
    final_method: Callable[[pd.DataFrame], pd.Series],
    folds: int | None = 5,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Stacking (Wolpert, 1992) of portfolio optimizers.

    Every method gives its weights, a final method is optimized over the returns of the
    methods' portfolios as if they were assets, and the weights are the methods' weights
    combined with the final method's, w = W v, with W the N × K matrix of the K methods'
    weights and v the final weights. The final method should learn how the portfolios do out
    of sample, not how well each fits the periods it was optimized on. With folds, the
    periods are cut into that many consecutive blocks, the methods are optimized on all
    periods outside a block and their portfolios earn the returns of the block, so the final
    method sees returns none of the portfolios was fitted on; this is a K-fold
    cross-validation without shuffling. Without folds, the final method is optimized on the
    in-sample returns R W. The weights of the methods themselves are always optimized on all
    periods.

    See Wolpert, D. H. (1992), "Stacked Generalization", Neural Networks, 5(2), 241-259.

    Referenced against:
    - skfolio, StackingOptimization: https://skfolio.org/generated/skfolio.optimization.StackingOptimization.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        methods (dict[str, Callable[[pd.DataFrame], pd.Series]]): Per name, a function
            that gives the weight per asset from a window of returns.
        final_method (Callable[[pd.DataFrame], pd.Series]): The function that gives the
            weight per method from the returns of the methods' portfolios, one column per
            method.
        folds (int | None, optional): The number of consecutive blocks of the
            cross-validation, at least 2, or None for the in-sample returns. Defaults to 5.

    Returns:
        tuple[pd.Series, pd.DataFrame, pd.Series]: The weight per asset, the weights of
            every method (one column each) and the weight of every method in the final
            portfolio.
    """
    if len(methods) < 2:  # noqa: PLR2004
        raise ValueError("Stacking needs at least two methods.")
    if folds is not None and not 2 <= folds <= len(returns):  # noqa: PLR2004
        raise ValueError(
            f"The number of folds must lie between 2 and {len(returns)}, or be None."
        )
    weights = pd.DataFrame(
        {name: method(returns) for name, method in methods.items()}
    ).reindex(returns.columns)
    if folds is None:
        portfolio_returns = returns @ weights
    else:
        blocks = np.array_split(np.arange(len(returns)), folds)
        portfolio_returns = pd.concat(
            [
                returns.iloc[block]
                @ pd.DataFrame(
                    {
                        name: method(
                            returns.iloc[np.setdiff1d(np.arange(len(returns)), block)]
                        )
                        for name, method in methods.items()
                    }
                ).reindex(returns.columns)
                for block in blocks
            ]
        )
    final = final_method(portfolio_returns).reindex(weights.columns)
    return weights @ final, weights, final


def get_diversification_ratio(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """
    The diversification ratio of a portfolio (Choueifaty and Coignard, 2008): the weighted
    average volatility of its assets divided by its volatility, w'σ / √(w'Σw).

    It is one for a single asset or perfectly correlated assets and grows as the
    correlations fall; its square is the number of independent risk factors a long-only
    portfolio holds (Choueifaty, Froidure and Reynier, 2013).

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
    most diversified portfolio of Choueifaty and Coignard (2008).

    The ratio is maximized directly with SLSQP within the weight bounds, with its gradient σ
    / √(w'Σw) - (w'σ) Σw / (w'Σw)^(3/2).

    Referenced against:
    - skfolio, MaximumDiversification: https://skfolio.org/generated/skfolio.optimization.MaximumDiversification.html

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
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    The fully invested portfolio closest to a benchmark (Roll, 1992).

    With the tracking error objective, the one with the lowest tracking error, the
    volatility of its return minus the benchmark's, TE²(w) = Var(R w - r_b) = w'Σ_AA w - 2
    w'Σ_Ab + σ_b²; with the information ratio objective, the one with the highest
    information ratio, the expected active return per unit of tracking error, (w'μ + (1 -
    Σw) r_f - μ_b) / TE(w). The part of the portfolio that is not invested, 1 - Σw, is cash
    at the risk free rate (borrowed at it above a net exposure of one), as in
    get_tracking_error; it has no risk, so it leaves the tracking error as it is.

    Referenced against:
    - skfolio, BenchmarkTracker (tracking error): https://skfolio.org/generated/skfolio.optimization.BenchmarkTracker.html
    - PyPortfolioOpt, objective_functions.ex_post_tracking_error (tracking error): https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.objective_functions.ex_post_tracking_error
    - skfolio, MeanRisk (information ratio): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

    Args:
        expected_returns (pd.Series): The annualized expected return of the assets and the
            benchmark.
        covariance (pd.DataFrame): The annualized covariance matrix of the assets and the
            benchmark.
        benchmark (str): The name of the benchmark in the expected returns and covariance
            matrix; it is not an asset of the portfolio.
        objective (str, optional): Options are: "tracking_error" (the lowest tracking error),
            "information_ratio" (the highest information ratio). Defaults to "tracking_error".
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
        risk_free_rate (float, optional): The annualized risk free rate the part that is
            not invested earns, for the information ratio. Defaults to 0.

    Returns:
        pd.Series: The weight per asset, without the benchmark.
    """
    check_option("objective", objective, ("tracking_error", "information_ratio"))
    if benchmark not in covariance.index:
        raise ValueError(f"The covariance matrix holds no benchmark '{benchmark}'.")
    _validate_covariance(covariance)
    assets = covariance.index.drop(benchmark)
    if len(assets) < 2:  # noqa: PLR2004
        raise ValueError("At least two assets besides the benchmark are needed.")
    sigma = covariance.loc[assets, assets].to_numpy()
    with_benchmark = covariance.loc[assets, benchmark].to_numpy()
    benchmark_variance = float(covariance.loc[benchmark, benchmark])
    # The active return w'(μ - r_f) - (μ_b - r_f) counts the part not invested at the
    # risk free rate; it differs from w'μ - μ_b by a constant only at a net exposure of one.
    active = expected_returns.reindex(assets).to_numpy() - risk_free_rate
    benchmark_excess = float(expected_returns[benchmark]) - risk_free_rate
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
        # In units of the average variance, as in get_minimum_variance_weights.
        variance_scale = np.mean(np.diag(sigma)) or 1.0
        weights = _solve(
            lambda w: tracking_variance(w) / variance_scale,
            lambda w: (2 * sigma @ w - 2 * with_benchmark) / variance_scale,
            constraints,
        )
    else:
        if _return_range(active, constraints)[1] <= benchmark_excess:
            raise ValueError(
                "No portfolio within the constraints expects a return above the "
                "benchmark's, so none has a positive information ratio."
            )

        def gradient(weights):
            variance = tracking_variance(weights)
            excess = weights @ active - benchmark_excess
            return -(
                active / np.sqrt(variance)
                - excess * (sigma @ weights - with_benchmark) / variance**1.5
            )

        weights = _solve(
            lambda w: -(w @ active - benchmark_excess) / np.sqrt(tracking_variance(w)),
            gradient,
            constraints,
        )

    return _as_series(weights, assets)


def get_tracking_error(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    benchmark: str,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    The expected active return, tracking error and information ratio of a portfolio against
    a benchmark.

    W'μ + (1 - Σw) r_f - μ_b, √(w'Σ_AA w - 2 w'Σ_Ab + σ_b²) and their ratio (Roll, 1992;
    Grinold and Kahn, 2000). The part of the portfolio that is not invested, 1 - Σw, is cash
    at the risk free rate (borrowed at it above a net exposure of one), as in
    get_portfolio_statistics.

    Args:
        weights (pd.Series): The weight per asset.
        expected_returns (pd.Series): The annualized expected return of the assets and the
            benchmark.
        covariance (pd.DataFrame): The annualized covariance matrix of the assets and the
            benchmark.
        benchmark (str): The name of the benchmark in the expected returns and covariance
            matrix.
        risk_free_rate (float, optional): The annualized risk free rate the part that is
            not invested earns. Defaults to 0.

    Returns:
        pd.Series: The Active Return, Tracking Error and Information Ratio.
    """
    # The active position: the weights of the assets against a short position in the
    # benchmark, whose variance is the tracking variance.
    active = weights.reindex(covariance.index).fillna(0)
    cash = 1 - float(active.sum())
    active[benchmark] -= 1
    active_return = (
        float(active @ expected_returns.reindex(covariance.index))
        + cash * risk_free_rate
    )
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
    expected logarithmic growth of wealth (Kelly, 1956; Breiman, 1961; Thorp, 2006).

    The exact method maximizes the average log return over the historical returns, (1/T) Σ
    ln(1 + r_t'w), with gradient (1/T) Σ r_t / (1 + r_t'w). The approximate method maximizes
    its second-order approximation w'μ - ½ w'Σw, a mean-variance portfolio with a risk
    aversion of one. Fractional Kelly scales the weights by the fraction; the remainder (1 -
    fraction) is cash, or borrowed when the fraction is above one. The constraints apply to
    the full Kelly portfolio, before the fraction scales it.

    Referenced against:
    - Riskfolio-Lib, Portfolio.optimization (Kelly criterion): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, for the
            "exact" method.
        expected_returns (pd.Series): The expected return per asset, for the
            "approximate" method, on the same scale (per period or annualized) as the
            covariance.
        covariance (pd.DataFrame): The covariance matrix of the assets, for the
            "approximate" method.
        method (str, optional): Options are: "exact" (the average log return of the historical
            returns), "approximate" (its second-order approximation). Defaults to "exact".
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

    Returns:
        pd.Series: The weight per asset, summing to the fraction times the net exposure.
    """
    check_option("method", method, ("exact", "approximate"))
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
    The weights of the resampled efficient portfolio (Michaud, 1998).

    These are the optimal weights for many return histories that could have been drawn
    instead of the one observed, whose average is the resampled portfolio. Each draw has as
    many periods as the returns and comes from a normal distribution with the expected
    returns and covariance (the parametric draws of Michaud), or is drawn with replacement
    from the periods of the returns (the bootstrap, which keeps their fat tails and skew).
    The expected returns and covariance of every draw are its sample mean and sample
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

    Referenced against:
    - Riskfolio-Lib, Portfolio.optimization (every draw): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.optimization

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset; their number of
            periods sets that of every draw and, for the bootstrap, the draws come from them.
        expected_returns (pd.Series): The annualized expected returns the parametric draws are
            centred on.
        covariance (pd.DataFrame): The annualized covariance matrix of the parametric draws.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily
            returns.
        objective (str, optional): The portfolio of every draw. Options are: "maximum_sharpe" (the
            highest Sharpe ratio), "minimum_variance" (the lowest variance), "mean_variance" (the
            utility with the risk aversion). Defaults to "maximum_sharpe".
        resamples (int, optional): The number of draws. Defaults to 500.
        resampling (str, optional): How the draws are made. Options are: "parametric" (drawn from a
            normal distribution), "bootstrap" (drawn with replacement from the periods). Defaults to
            "parametric".
        seed (int | None, optional): The seed of the random draws, so that the same seed
            gives the same weights. Defaults to 0; None draws anew every time.
        risk_free_rate (float, optional): The annualized risk free rate of the maximum
            Sharpe ratio. Defaults to 0.
        risk_aversion (float, optional): The risk aversion of the mean-variance objective. Defaults
            to 2.5.
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
    check_option("objective", objective, RESAMPLED_OBJECTIVES)
    check_option("resampling", resampling, RESAMPLING_METHODS)
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
    risk_free_rate: float | pd.Series = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The Black-Litterman expected returns (Black and Litterman, 1992), as total returns, and
    their posterior covariance matrix.

    It is the returns the market implies, moved towards the investor's views in proportion
    to the confidence in them.

    The prior is the return the market implies through its weights: the excess return Π =
    δΣw at which holding the market is optimal (He and Litterman, 1999), plus the risk free
    rate. Views move it towards the investor's expectations in proportion to their
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

    Shares similar logic:
    - PyPortfolioOpt, BlackLittermanModel: https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html#pypfopt.black_litterman.BlackLittermanModel
    - Riskfolio-Lib, black_litterman: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.black_litterman

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
        risk_free_rate (float | pd.Series, optional): The annualized risk free rate, added
            to the implied excess returns so that they are total returns like the views, or
            one per name (0 for a name missing from it), e.g. 0 for the factors of the
            augmented model, whose returns are excess returns. Defaults to 0.

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
    implied = risk_aversion * sigma @ weights + (
        risk_free_rate.reindex(assets).fillna(0).to_numpy(dtype=float)
        if isinstance(risk_free_rate, pd.Series)
        else risk_free_rate
    )
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
            row = get_view_vector(view, assets)
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


def get_augmented_black_litterman_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    factor_covariance: pd.DataFrame,
    loadings: pd.DataFrame,
    views: dict[str, float] | None = None,
    view_confidences: dict[str, float] | None = None,
    factor_views: dict[str, float] | None = None,
    factor_view_confidences: dict[str, float] | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The augmented Black-Litterman expected returns of Cheung (2009), which take views on
    factors next to (or instead of) views on assets.

    The assets and the factors are stacked into one vector with the covariance matrix

    - Σ_a = [[Σ, B Σ_f], [Σ_f B', Σ_f]]

    with B the loadings of the assets on the factors and Σ_f the covariance of the factor
    returns, and the market weights extended with a zero weight for every factor, so the
    implied returns are Π_a = δ Σ_a w_a = [δ Σ w; δ Σ_f B' w]: the excess returns of the
    assets and the factor premiums the market portfolio implies. The Black-Litterman
    update of get_black_litterman_returns then runs on the stacked vector, so a view on a
    factor moves every asset through its covariance with that factor. The risk free rate
    is added to the assets only: factor returns are excess returns already, so a factor
    view is a factor premium ("HML": 0.03, value beats growth by 3% a year).

    Without factor views and with the factors left out, this is the Black-Litterman model
    itself. The intercepts of the loadings are left out, as in Cheung (2009), where the
    equilibrium has no alpha.

    See Cheung, W. (2009), "The Augmented Black-Litterman Model: A Ranking-Free Approach to
    Factor-Based Portfolio Construction and Beyond", https://ssrn.com/abstract=1347648.

    Referenced against:
    - Riskfolio-Lib, augmented_black_litterman: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.augmented_black_litterman

    Args:
        covariance (pd.DataFrame): The annualized covariance matrix of the assets.
        market_weights (pd.Series): The market capitalization weight per asset.
        factor_covariance (pd.DataFrame): The annualized covariance matrix of the factors.
        loadings (pd.DataFrame): The loading of every asset (rows) on every factor (columns).
        views (dict[str, float] | None, optional): The annualized views on assets, as in
            get_black_litterman_returns. Defaults to None.
        view_confidences (dict[str, float] | None, optional): The confidence per asset view.
            Defaults to None.
        factor_views (dict[str, float] | None, optional): The annualized views on factors
            ("HML": 0.03) or on the difference of two ("SMB-HML": 0.01). Defaults to None.
        factor_view_confidences (dict[str, float] | None, optional): The confidence per
            factor view. Defaults to None.
        risk_aversion (float, optional): The risk aversion of the market, δ. Defaults to 2.5.
        tau (float, optional): The uncertainty of the prior relative to the covariance.
            Defaults to 0.05.
        risk_free_rate (float, optional): The annualized risk free rate, added to the
            implied excess returns of the assets. Defaults to 0.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The Implied Return and the Black-Litterman
            Return per asset, both annualized total returns, and the annualized posterior
            covariance matrix of the assets.

    Notes:
    - Unlike Riskfolio-Lib's augmented_black_litterman, which adds the intercepts of the
      loadings matrix to the posterior returns when it holds a constant, the intercepts are
      left out.
    """
    assets = covariance.index
    factor_names = factor_covariance.index
    overlap = set(assets) & set(factor_names)
    if overlap:
        raise ValueError(
            f"The factors share names with the assets: {', '.join(sorted(overlap))}."
        )
    beta = loadings.reindex(index=assets, columns=factor_names).fillna(0).to_numpy()
    sigma_factors = factor_covariance.to_numpy()
    cross = beta @ sigma_factors
    names = assets.append(factor_names)
    augmented = pd.DataFrame(
        np.block([[covariance.to_numpy(), cross], [cross.T, sigma_factors]]),
        index=names,
        columns=names,
    )
    table, posterior = get_black_litterman_returns(
        augmented,
        market_weights.reindex(names).fillna(0),
        views={**(views or {}), **(factor_views or {})} or None,
        view_confidences={**(view_confidences or {}), **(factor_view_confidences or {})}
        or None,
        risk_aversion=risk_aversion,
        tau=tau,
        risk_free_rate=pd.Series(risk_free_rate, index=assets),
    )
    return table.loc[assets], posterior.loc[assets, assets]


def get_bayesian_black_litterman_returns(
    loadings: pd.DataFrame,
    factor_returns: pd.Series,
    factor_covariance: pd.DataFrame,
    residual_variance: pd.Series,
    factor_views: dict[str, float],
    factor_view_confidences: dict[str, float] | None = None,
    tau: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The Bayesian Black-Litterman expected returns of Kolm and Ritter (2017), which put the
    views on the factors of a factor model r = α + B f + ε rather than on the assets.

    The factor premiums have the prior N(μ_f, τ Σ_f), with μ_f the historical average factor
    returns, and the views P μ_f = Q the uncertainty Ω_k = τ p_k'Σ_f p_k, scaled by (1 - c)
    / c for a confidence c (as in get_black_litterman_returns). Their posterior is

    - Ŝ = τΣ_f - τΣ_f P'(P τΣ_f P' + Ω)⁻¹ P τΣ_f
    - Π̂ = μ_f + τΣ_f P'(P τΣ_f P' + Ω)⁻¹ (Q - P μ_f)

    and the asset returns follow from the factor model: μ = α + B Π̂ and Σ_post = B Σ_f B' +
    D + B Ŝ B', with D the residual variances.

    Without views the prior is the factor model itself and μ is the average return of
    every asset (the OLS intercept makes the residuals average zero).

    See Kolm, P. and Ritter, G. (2017), "On the Bayesian Interpretation of Black-Litterman",
    European Journal of Operational Research, 258(2), 564-572.

    Referenced against:
    - Riskfolio-Lib, black_litterman_bayesian: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.black_litterman_bayesian

    Args:
        loadings (pd.DataFrame): The Intercept and the factor loadings per asset, annualized
            intercept (see estimation_model.get_factor_loadings).
        factor_returns (pd.Series): The annualized average return of every factor.
        factor_covariance (pd.DataFrame): The annualized covariance matrix of the factors.
        residual_variance (pd.Series): The annualized residual variance of every asset.
        factor_views (dict[str, float]): The annualized views on factors ("HML": 0.03) or on
            the difference of two ("SMB-HML": 0.01).
        factor_view_confidences (dict[str, float] | None, optional): The confidence per
            factor view, between 0 and 1. Defaults to None.
        tau (float, optional): The uncertainty of the prior factor premiums relative to the
            factor covariance. Defaults to 0.05.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The Prior Return (α + B μ_f) and the
            Black-Litterman Return per asset, annualized, and the annualized posterior
            covariance matrix.

    Notes:
    - Riskfolio-Lib's black_litterman_bayesian inverts Σ to the same result: its Π_BLB =
      Σ_BLB S⁻¹ B (Ŝ⁻¹ + B'S⁻¹B)⁻¹ Ŝ⁻¹ Π̂ equals B Π̂ and its Σ_BLB = (S⁻¹ - S⁻¹B(Ŝ⁻¹ +
      B'S⁻¹B)⁻¹B'S⁻¹)⁻¹ equals S + B Ŝ B' (Woodbury). It takes the prior N(μ_f, Σ_f) and Ω
      with τ = 1 / T; a tau of 1 and a confidence of T / (T + 1) for every view give its
      result.
    """
    if not factor_views:
        raise ValueError("The Bayesian Black-Litterman model needs views on factors.")
    unmatched = set(factor_view_confidences or {}) - set(factor_views)
    if unmatched:
        raise ValueError(
            f"The view confidences name views that are not given: {', '.join(sorted(unmatched))}."
        )
    factor_names = factor_covariance.index
    assets = loadings.index
    beta = loadings[factor_names].to_numpy(dtype=float)
    intercept = loadings["Intercept"].to_numpy(dtype=float)
    sigma_factors = factor_covariance.to_numpy(dtype=float)
    prior = factor_returns.reindex(factor_names).to_numpy(dtype=float)
    tau_sigma = tau * sigma_factors

    pick, uncertainty = [], []
    for view in factor_views:
        row = get_view_vector(view, factor_names)
        variance = tau * row @ sigma_factors @ row
        confidence = (factor_view_confidences or {}).get(view)
        if confidence is not None:
            if not 0 < confidence <= 1:
                raise ValueError("A view confidence must lie between 0 and 1.")
            variance *= max(1 - confidence, 1e-8) / confidence
        pick.append(row)
        uncertainty.append(variance)
    p = np.array(pick)
    try:
        solved = np.linalg.solve(
            p @ tau_sigma @ p.T + np.diag(uncertainty),
            np.column_stack(
                [np.array(list(factor_views.values())) - p @ prior, p @ tau_sigma]
            ),
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The views contradict or repeat each other, so they cannot be combined."
        ) from error
    posterior_premiums = prior + tau_sigma @ p.T @ solved[:, 0]
    uncertainty_left = tau_sigma - tau_sigma @ p.T @ solved[:, 1:]

    covariance = beta @ (sigma_factors + uncertainty_left) @ beta.T + np.diag(
        residual_variance.reindex(assets).to_numpy(dtype=float)
    )
    return (
        pd.DataFrame(
            {
                "Prior Return": intercept + beta @ prior,
                "Black-Litterman Return": intercept + beta @ posterior_premiums,
            },
            index=assets,
        ),
        pd.DataFrame(covariance, index=assets, columns=assets),
    )


def get_market_implied_risk_aversion(
    benchmark_returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """
    The risk aversion of the market, the excess return it earns per unit of variance: δ =
    (E[r_m] - r_f) / σ²_m.

    A representative investor who holds the market is indifferent to the next unit of risk
    at exactly this δ, which makes it the risk aversion that the market implied returns Π =
    δΣw_mkt of Black and Litterman (1992) are consistent with.

    E[r_m] is the average return per period of the benchmark times the number of periods
    in a year (arithmetic, not compounded) and σ²_m its variance per period (ddof=1)
    times the same number, so δ is the same at every frequency up to estimation error.

    See Black, F. and Litterman, R. (1992), "Global Portfolio Optimization", Financial
    Analysts Journal, 48(5), 28-43, and Walters, J. (2014), "The Black-Litterman Model in
    Detail", https://ssrn.com/abstract=1314585.

    Shares similar logic:
    - PyPortfolioOpt, market_implied_risk_aversion: https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html#pypfopt.black_litterman.market_implied_risk_aversion

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
    How much each asset adds to the volatility of a portfolio (Euler allocation).

    These are the marginal contribution ∂σ/∂w_i = (Σw)_i / σ, the risk contribution w_i
    times that, and its share of the total, which sums to one.

    Shares similar logic:
    - Riskfolio-Lib, Risk_Contribution: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Risk_Contribution

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
    if volatility <= ZERO_VOLATILITY_TOLERANCE:
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
    weights: pd.Series, returns: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """
    How much each asset adds to the historical expected shortfall (CVaR) of a portfolio
    (Euler allocation).

    The expected shortfall is the average loss in the worst 1 - α share of the periods, ES =
    -Σ p_t r_p,t, with p_t the tail probabilities of Rockafellar and Uryasev (2002): 1 /
    ((1 - α)T) for each of the ⌊(1 - α)T⌋ worst periods of the portfolio and the remaining
    fraction of the tail for the next one. Because it is homogeneous of degree one in the
    weights, its derivative per asset, ∂ES/∂w_i = -Σ p_t r_i,t, times the weight gives
    contributions that add up to the expected shortfall (Tasche, 2000; Scaillet, 2004). The
    tail periods are those of the portfolio, so an asset that tends to gain when the
    portfolio loses contributes less than its own expected shortfall, or a negative amount
    when it hedges.

    The expected shortfall is the one the minimum CVaR and maximum CVaR ratio
    optimizations minimize and report, the average of the tail with a fraction of the
    boundary period when (1 - α)T is not a whole number.

    See Tasche, D. (2000), "Conditional Expectations as Quantile Derivatives", arXiv
    math/0104190, and Scaillet, O. (2004), "Nonparametric Estimation and Sensitivity
    Analysis of Expected Shortfall", Mathematical Finance, 14(1), 115-129.

    Referenced against:
    - Riskfolio-Lib, Risk_Contribution (CVaR): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Risk_Contribution

    Args:
        weights (pd.Series): The weight per asset.
        returns (pd.DataFrame): The returns per period, one column per asset.
        alpha (float, optional): The tail probability, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.

    Returns:
        pd.DataFrame: The Weight, Marginal Contribution (-Σ p_t r_i,t), Risk
        Contribution (the weight times the marginal contribution) and Percentage
        Contribution per asset, as losses per period.
    """
    validate_alpha(alpha)
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("No period has a return for every asset.")
    weights = weights.reindex(returns.columns).fillna(0)
    probabilities = get_tail_probabilities(-(returns @ weights), alpha)

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


HISTORICAL_RISK_MEASURES = ("semideviation", "mad", "cdar")


def get_historical_risk_contributions(
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_measure: str = "semideviation",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    How much each asset adds to a risk measure of a portfolio's historical returns (Euler
    allocation).

    Each measure is homogeneous of degree one in the weights, so the weight times its
    derivative per asset, w_i ∂ρ/∂w_i, adds up to the portfolio's measure (Tasche, 2000).
    With r_p,t = Σ w_i r_i,t the portfolio's return over T periods:

    - The semideviation: the downside deviation below a return of zero, SD⁻ = √(Σ min(r_p,t,
      0)² / T) (Sortino and Price, 1994), with ∂SD⁻/∂w_i = Σ min(r_p,t, 0) r_i,t / (T SD⁻).
      An asset contributes by losing in the periods in which the portfolio loses. The
      threshold stays at zero: a threshold other than zero would not scale with the weights
      and the contributions would not add up.
    - The mean absolute deviation, MAD = Σ |r_p,t - μ_p| / T (Konno and Yamazaki, 1991),
      with ∂MAD/∂w_i = Σ sign(r_p,t - μ_p)(r_i,t - μ_i) / T: an asset contributes by moving
      with the portfolio's deviations from its mean.
    - The Conditional Drawdown at Risk, the average of the worst 1 - α share of the
      drawdowns D_t = max_(s≤t) Y_s - Y_t of the uncompounded cumulative return Y_t =
      Σ_(s≤t) r_p,s (from Y_0 = 0), weighted by the tail probabilities p_t of Rockafellar
      and Uryasev (2002), the drawdowns the minimum drawdown optimization minimizes
      (Chekhlov, Uryasev and Zabarankin, 2005). A drawdown is the sum of the assets'
      drawdowns from the portfolio's peak s*(t), so ∂CDaR/∂w_i = Σ p_t (C_i,s*(t) - C_i,t)
      with C_i the cumulative return of asset i (Goldberg and Mahmoud, 2017).

    The measures are per period (e.g. daily), like the downside deviation and mean
    absolute deviation of the Risk module, and not annualized.

    See Tasche, D. (2000), "Conditional Expectations as Quantile Derivatives", arXiv
    math/0104190; Sortino, F. A. and Price, L. N. (1994), "Performance Measurement in a
    Downside Risk Framework", The Journal of Investing, 3(3), 59-64; Konno, H. and
    Yamazaki, H. (1991), "Mean-Absolute Deviation Portfolio Optimization Model and Its
    Applications to Tokyo Stock Market", Management Science, 37(5), 519-531; and
    Goldberg, L. R. and Mahmoud, O. (2017), "Drawdown: From Practice to Theory and Back
    Again", Mathematics and Financial Economics, 11(3), 275-297.

    Args:
        weights (pd.Series): The weight per asset.
        returns (pd.DataFrame): The returns per period, one column per asset.
        risk_measure (str, optional): The risk measure. Options are: "semideviation" (downside
            deviation below zero), "mad" (mean absolute deviation), "cdar" (Conditional Drawdown at
            Risk). Defaults to "semideviation".
        alpha (float, optional): The tail probability of the CDaR, e.g. 0.05 for the
            average of the worst 5% of drawdowns. Defaults to 0.05.

    Returns:
        pd.DataFrame: The Weight, Marginal Contribution (∂ρ/∂w_i), Risk Contribution (the
        weight times the marginal contribution) and Percentage Contribution per asset.
    """
    check_option("risk_measure", risk_measure, HISTORICAL_RISK_MEASURES)
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("No period has a return for every asset.")
    weights = weights.reindex(returns.columns).fillna(0)
    data = returns.to_numpy(dtype=float)
    portfolio = data @ weights.to_numpy()
    periods = len(data)

    if risk_measure == "semideviation":
        shortfall = np.minimum(portfolio, 0)
        semideviation = np.sqrt(shortfall @ shortfall / periods)
        if semideviation == 0:
            raise ValueError("The portfolio has no return below zero.")
        marginal = shortfall @ data / (periods * semideviation)
    elif risk_measure == "mad":
        marginal = (
            np.sign(portfolio - portfolio.mean()) @ (data - data.mean(axis=0)) / periods
        )
    else:
        # The cumulative returns start from a peak of zero before the first period, as
        # the drawdowns of the minimum drawdown optimization do.
        cumulative = np.vstack([np.zeros(data.shape[1]), np.cumsum(data, axis=0)])
        wealth = cumulative @ weights.to_numpy()
        # The latest period up to each one in which the portfolio stood at its peak.
        records = np.where(
            wealth >= np.maximum.accumulate(wealth), np.arange(len(wealth)), 0
        )
        peaks = np.maximum.accumulate(records)[1:]
        drawdowns = cumulative[peaks] - cumulative[1:]
        probabilities = get_tail_probabilities(drawdowns @ weights.to_numpy(), alpha)
        marginal = probabilities @ drawdowns

    contribution = weights.to_numpy() * marginal
    total = contribution.sum()
    if total == 0:
        raise ValueError(f"The portfolio has no {risk_measure} to decompose.")

    return pd.DataFrame(
        {
            "Weight": weights.to_numpy(),
            "Marginal Contribution": marginal,
            "Risk Contribution": contribution,
            "Percentage Contribution": contribution / total,
        },
        index=returns.columns,
    )


# The risk measures whose contributions are found by finite differences, with the order of
# homogeneity of each in the weights (the square root of the fourth moment doubles when the
# weights double twice over).
FINITE_DIFFERENCE_RISK_MEASURES = {
    "var": 1,
    "evar": 1,
    "worst_realization": 1,
    "gini_mean_difference": 1,
    "tail_gini": 1,
    "lower_partial_moment": 1,
    "kurtosis": 2,
    "maximum_drawdown": 1,
    "average_drawdown": 1,
    "drawdown_at_risk": 1,
    "entropic_drawdown_at_risk": 1,
    "ulcer_index": 1,
}

# The step of the central differences: 1e-7, as in Riskfolio-Lib, and 1e-4 for the entropic
# measures, which are found by a numerical minimization of their own.
FINITE_DIFFERENCE_STEP = 1e-7
ENTROPIC_FINITE_DIFFERENCE_STEP = 1e-4


def get_risk_measure(
    returns: pd.DataFrame, risk_measure: str, alpha: float = 0.05
) -> pd.Series:
    """
    A risk measure of every column of returns (each a portfolio), from the Risk module's
    measures, as a positive loss per period.

    These are the historical Value at Risk and Entropic Value at Risk, the worst
    realization, the Gini mean difference, the Tail Gini, the first lower partial moment
    below zero, the square root of the fourth central moment (the kurtosis measure of
    Riskfolio-Lib), and the maximum and average drawdown, the Drawdown at Risk, the Entropic
    Drawdown at Risk and the Ulcer Index of the uncompounded cumulative returns (the
    absolute drawdowns of Riskfolio-Lib), which scale with the weights.

    Args:
        returns (pd.DataFrame): The returns per period, one column per portfolio.
        risk_measure (str): The risk measure. Options are: "var" (historical Value at Risk), "evar"
            (Entropic Value at Risk), "worst_realization" (the largest loss), "gini_mean_difference"
            (Gini mean difference), "tail_gini" (Tail Gini), "lower_partial_moment" (first lower
            partial moment below zero), "kurtosis" (square root of the fourth central moment),
            "maximum_drawdown" (maximum drawdown), "average_drawdown" (average drawdown),
            "drawdown_at_risk" (Drawdown at Risk), "entropic_drawdown_at_risk" (Entropic Drawdown at
            Risk), "ulcer_index" (Ulcer Index).
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.

    Returns:
        pd.Series: The risk measure per column, positive for a loss.
    """
    if risk_measure == "var":
        measure = -var_model.get_var_historic(returns, alpha)
    elif risk_measure == "evar":
        measure = -evar_model.get_evar_historic(returns, alpha)
    elif risk_measure == "worst_realization":
        measure = -dispersion_model.get_worst_realization(returns)
    elif risk_measure == "gini_mean_difference":
        measure = dispersion_model.get_gini_mean_difference(returns)
    elif risk_measure == "tail_gini":
        measure = -dispersion_model.get_tail_gini(returns, alpha)
    elif risk_measure == "lower_partial_moment":
        measure = risk_model.get_partial_moment(returns, order=1)
    elif risk_measure == "kurtosis":
        measure = dispersion_model.get_central_moment(returns, order=4) ** 0.5
    elif risk_measure == "maximum_drawdown":
        measure = -drawdown_model.get_maximum_drawdown(returns, compound_returns=False)
    elif risk_measure == "average_drawdown":
        measure = -drawdown_model.get_average_drawdown(
            returns, method="observation", compound_returns=False
        )
    elif risk_measure == "drawdown_at_risk":
        measure = -drawdown_model.get_drawdown_at_risk(
            returns, alpha, compound_returns=False
        )
    elif risk_measure == "entropic_drawdown_at_risk":
        measure = -drawdown_model.get_entropic_drawdown_at_risk(
            returns, alpha, compound_returns=False
        )
    elif risk_measure == "ulcer_index":
        measure = risk_model.get_ui(returns, rolling=None, compound_returns=False)
    else:
        raise InvalidArgumentError(
            f"The risk measure must be one of {', '.join(FINITE_DIFFERENCE_RISK_MEASURES)}, "
            f"not '{risk_measure}'."
        )
    return pd.Series(measure, index=returns.columns, dtype=float)


def get_finite_difference_risk_contributions(
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_measure: str = "maximum_drawdown",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    How much each asset adds to a risk measure of a portfolio's historical returns (Euler
    allocation), with the derivative of the measure found by central finite differences.

    ∂ρ/∂w_i ≈ (ρ(w + h e_i) - ρ(w - h e_i)) / (2h), with h = 1e-7 (1e-4 for the entropic
    measures). A measure that is homogeneous of order k in the weights satisfies Σ w_i
    ∂ρ/∂w_i = k ρ (Euler's theorem), so the risk contribution w_i ∂ρ/∂w_i / k adds up to ρ
    (Tasche, 2000). The measures that are not smooth (the Value at Risk, the worst
    realization, the drawdowns) have a derivative almost everywhere: the measure follows the
    returns of a single period, or a few, and the derivative is the return of each asset in
    those periods.

    The measure is taken of the portfolio's returns, see get_risk_measure; every measure is
    per period (e.g. daily), not annualized.

    Notes:
        At an optimum several periods often tie (two drawdowns of the same depth, two
        equally bad realizations), and there the measure has a kink: the central difference
        averages the slopes on either side, and the contributions no longer add up to the
        measure. The contributions are therefore scaled so that they add up to the
        portfolio's measure ρ(w), contribution_i × ρ(w) / Σ contribution_j (whenever that
        sum is not zero), and the marginal contributions are scaled by the same factor, so
        that Σ w_i ∂ρ/∂w_i / k = ρ(w) holds for the reported values. Away from a kink the
        factor is one up to the finite-difference error.

    See Tasche, D. (2000), "Conditional Expectations as Quantile Derivatives", arXiv
    math/0104190.

    Referenced against:
    - Riskfolio-Lib, Risk_Contribution: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Risk_Contribution

    Args:
        weights (pd.Series): The weight per asset.
        returns (pd.DataFrame): The returns per period, one column per asset.
        risk_measure (str, optional): The risk measure. Options are: "var" (historical Value at
            Risk), "evar" (Entropic Value at Risk), "worst_realization" (the largest loss),
            "gini_mean_difference" (Gini mean difference), "tail_gini" (Tail Gini),
            "lower_partial_moment" (first lower partial moment below zero), "kurtosis" (square root
            of the fourth central moment), "maximum_drawdown" (maximum drawdown), "average_drawdown"
            (average drawdown), "drawdown_at_risk" (Drawdown at Risk), "entropic_drawdown_at_risk"
            (Entropic Drawdown at Risk), "ulcer_index" (Ulcer Index). Defaults to
            "maximum_drawdown".
        alpha (float, optional): The tail probability of the tail measures, e.g. 0.05 for the worst 5% of
            periods. Defaults to 0.05.

    Returns:
        pd.DataFrame: The Weight, Marginal Contribution (∂ρ/∂w_i), Risk Contribution and
        Percentage Contribution per asset.
    """
    check_option("risk_measure", risk_measure, FINITE_DIFFERENCE_RISK_MEASURES)
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("No period has a return for every asset.")
    weights = weights.reindex(returns.columns).fillna(0)
    step = (
        ENTROPIC_FINITE_DIFFERENCE_STEP
        if risk_measure in ("evar", "entropic_drawdown_at_risk")
        else FINITE_DIFFERENCE_STEP
    )
    assets = len(weights)
    # Every column holds the weights with one asset's weight moved up or down a step.
    shifts = np.hstack([np.eye(assets), -np.eye(assets)]) * step
    # The last column holds the weights themselves, for the portfolio's own measure.
    shifted = weights.to_numpy(dtype=float)[:, np.newaxis] + np.hstack(
        [shifts, np.zeros((assets, 1))]
    )
    measures = get_risk_measure(
        pd.DataFrame(returns.to_numpy(dtype=float) @ shifted, index=returns.index),
        risk_measure,
        alpha,
    ).to_numpy()
    measure = measures[-1]
    marginal = (measures[:assets] - measures[assets : 2 * assets]) / (2 * step)
    contribution = (
        weights.to_numpy() * marginal / FINITE_DIFFERENCE_RISK_MEASURES[risk_measure]
    )
    # Where periods tie (a kink of the measure) the central difference averages the left and
    # right slopes, so the contributions are scaled to add up to the measure itself.
    if contribution.sum() != 0:
        scale = measure / contribution.sum()
        marginal = marginal * scale
        contribution = contribution * scale
    total = contribution.sum()
    if total == 0:
        raise ValueError(f"The portfolio has no {risk_measure} to decompose.")

    return pd.DataFrame(
        {
            "Weight": weights.to_numpy(),
            "Marginal Contribution": marginal,
            "Risk Contribution": contribution,
            "Percentage Contribution": contribution / total,
        },
        index=returns.columns,
    )


def get_factor_risk_contributions(
    weights: pd.Series, loadings: pd.DataFrame, marginal_contributions: pd.Series
) -> pd.DataFrame:
    """
    How much each factor of a factor model adds to the risk of a portfolio, the factor risk
    contributions of Meucci (2009) and Roncalli and Weisang (2016).

    With B the N × K loadings of the assets on the factors, the portfolio's exposure to the
    factors is y = B'w, and the marginal risk of a factor follows from the marginal risk of
    the assets m = ∂ρ/∂w through the pseudo-inverse, ∂ρ/∂y = B⁺'m with B⁺ = pinv(B'). The
    risk contribution of factor f is y_f (B⁺'m)_f. The part of the weights the factors do
    not span (the null space of B') forms an Other factor, Σ (B̃'w) ∘ (B̃⁺'m) with B̃ =
    pinv(null(B')'), so the factor contributions and Other add up to the risk measure, Σ w_i
    m_i.

    See Meucci, A. (2009), "Managing Diversification", Risk, 22(5), 74-79; Roncalli, T. and
    Weisang, G. (2016), "Risk Parity Portfolios with Risk Factors", Quantitative Finance,
    16(3), 377-388.

    Referenced against:
    - Riskfolio-Lib, Factors_Risk_Contribution: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Factors_Risk_Contribution

    Args:
        weights (pd.Series): The weight per asset.
        loadings (pd.DataFrame): The loading of every asset (rows) on every factor
            (columns), without the intercept.
        marginal_contributions (pd.Series): The marginal contribution ∂ρ/∂w_i of every
            asset to the risk measure.

    Returns:
        pd.DataFrame: Per factor and Other, the Exposure, Marginal Contribution, Risk
        Contribution and Percentage Contribution.
    """
    assets = loadings.index
    beta = loadings.to_numpy(dtype=float)
    weight = weights.reindex(assets).fillna(0).to_numpy(dtype=float)
    marginal = marginal_contributions.reindex(assets).to_numpy(dtype=float)
    exposure = beta.T @ weight
    factor_marginal = np.linalg.pinv(beta.T).T @ marginal
    contribution = exposure * factor_marginal
    complement = null_space(beta.T)
    other = 0.0
    if complement.size:
        other_loadings = np.linalg.pinv(complement.T)
        other = float(
            np.sum(
                (other_loadings.T @ weight)
                * (np.linalg.pinv(other_loadings.T).T @ marginal)
            )
        )
    total = float(weight @ marginal)
    if total == 0:
        raise ValueError("The portfolio has no risk to decompose.")
    table = pd.DataFrame(
        {
            "Exposure": np.append(exposure, np.nan),
            "Marginal Contribution": np.append(factor_marginal, np.nan),
            "Risk Contribution": np.append(contribution, other),
        },
        index=[*loadings.columns, "Other"],
    )
    table["Percentage Contribution"] = table["Risk Contribution"] / total
    return table


def get_diversification_metrics(
    weights: pd.Series, covariance: pd.DataFrame
) -> pd.Series:
    """
    How diversified a portfolio is, by how concentrated its weights are and by how many
    independent sources its risk comes from.

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

    Referenced against:
    - skfolio, Portfolio (diversification and effective_number_assets): https://skfolio.org/generated/skfolio.portfolio.Portfolio.html

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


DISCRETE_ALLOCATION_METHODS = ("integer", "greedy")

# The longest the integer programme of a discrete allocation may search, in seconds; it
# returns the best allocation found by then.
DISCRETE_ALLOCATION_TIME_LIMIT = 30


def get_discrete_allocation(
    weights: pd.Series,
    prices: pd.Series,
    portfolio_value: float = 10000,
    method: str = "integer",
) -> pd.DataFrame:
    """
    The whole number of shares per asset that comes closest to the weights for a portfolio
    value, with the cash that is left over.

    The target amount of asset i is t_i = w_i V, and n_i shares at a price p_i hold n_i p_i.

    - The integer programme that minimizes the absolute deviation from the target amounts
      with the leftover cash counted as a holding with a target of zero, min Σ |t_i - n_i
      p_i| + c subject to Σ n_i p_i + c = V, c ≥ 0 and n_i whole, with the absolute
      deviations written as d_i ≥ ±(t_i - n_i p_i), solved with branch and bound
      (Cornuejols, Peña and Tütüncü, 2018, ch. 11). The amounts are rescaled for the solver,
      and its allocation replaces the greedy one only when it deviates less, so it is never
      further from the targets than the greedy allocation.
    - The greedy allocation: every position is first rounded down to whole shares, ⌊t_i /
      p_i⌋, and the leftover cash then buys one more share at a time of the asset furthest
      below its target amount, while one is below it and its price fits the cash. A share of
      an asset below its target never increases the total deviation, and after the rounding
      each asset is less than one share below it, so this takes at most one round per asset.

    A short weight is held as a negative number of shares: rounding down makes a
    short position larger, so the proceeds never fall short of the long positions they
    fund and the leftover cash is never negative. The integer programme keeps each
    position on the side of its weight. It holds a long position at no more than
    ⌈t_i / p_i⌉ shares, beyond which it only moves further from its target, but lets a
    short position go beyond its target: shorting a share more raises the cash for a long
    share that is further below its own target, which can lower the total deviation. A
    short position is bounded only by |n_i| p_i - |t_i| ≤ D, with D the deviation of the
    greedy allocation, which no better allocation can exceed.

    See Cornuejols, G., Peña, J. and Tütüncü, R. (2018), "Optimization Methods in
    Finance", 2nd edition, Cambridge University Press.

    Shares similar logic:
    - PyPortfolioOpt, DiscreteAllocation.lp_portfolio (integer programme): https://pyportfolioopt.readthedocs.io/en/latest/Postprocessing.html#pypfopt.discrete_allocation.DiscreteAllocation.lp_portfolio
    - PyPortfolioOpt, DiscreteAllocation.greedy_portfolio (greedy allocation): https://pyportfolioopt.readthedocs.io/en/latest/Postprocessing.html#pypfopt.discrete_allocation.DiscreteAllocation.greedy_portfolio

    Args:
        weights (pd.Series): The weight per asset, summing to one.
        prices (pd.Series): The price of one share per asset.
        portfolio_value (float, optional): The amount to invest. Defaults to 10000.
        method (str, optional): Options are: "integer" (the integer programme), "greedy" (the greedy
            allocation). Defaults to "integer".

    Returns:
        pd.DataFrame: Per asset the Price, Shares, Amount (shares times price), Target
        Weight and Weight (the amount over the portfolio value), and a "Leftover Cash" row
        with the amount that is not invested.
    """
    check_option("method", method, DISCRETE_ALLOCATION_METHODS)
    if not portfolio_value > 0:
        raise ValueError("The portfolio value must be above zero.")
    prices = prices.reindex(weights.index).astype(float)
    missing = prices.index[~(prices > 0)].tolist()
    if missing:
        raise ValueError(f"There is no price above zero for {', '.join(missing)}.")

    price = prices.to_numpy()
    target = weights.to_numpy(dtype=float) * portfolio_value

    # The greedy allocation, which also bounds the integer programme below.
    shares = np.floor(target / price)
    cash = portfolio_value - shares @ price
    for _ in range(len(price)):
        shortfall = target - shares * price
        candidates = np.flatnonzero((shortfall > 0) & (price <= cash))
        if not len(candidates):
            break
        asset = candidates[np.argmax(shortfall[candidates])]
        shares[asset] += 1
        cash -= price[asset]

    if method == "integer":
        assets = len(price)
        # Beyond ⌈t_i / p_i⌉ shares a long position only moves further from its target. A short position can go beyond its target, as one more share short raises the cash for a long share further below its own target, but its deviation |n_i| p_i - |t_i| alone can never exceed the total deviation D of the greedy allocation in an optimum, so |n_i| ≤ (|t_i| + D) / p_i bounds it without excluding one; an open bound leaves the branch and bound of millions of shares to the time limit.  # noqa: E501
        limit = np.ceil(np.abs(target) / price)
        greedy_deviation = np.abs(target - shares * price).sum() + cash
        short_limit = np.floor((np.abs(target) + greedy_deviation) / price)
        bounds = Bounds(
            np.concatenate(
                [np.where(target < 0, -short_limit, 0), np.zeros(assets + 1)]
            ),
            np.concatenate(
                [np.where(target > 0, limit, 0), np.full(assets + 1, np.inf)]
            ),
        )
        identity = np.eye(assets)
        no_cash = np.zeros((assets, 1))
        found, message = False, ""
        # The variables are the shares n_i, the absolute deviations d_i and the cash c. An objective of Σ d_i + c itself, rather than one offset by the portfolio value, makes the solver's relative gap a gap in the deviation. The amounts are scaled for the solver, as amounts in the thousands next to share counts leave HiGHS with a solve error on some portfolios: first to multiples of the average price and, should that fail or not improve on the greedy allocation, to fractions of the portfolio value. The greedy allocation stays when the solver finds nothing closer, which it then is by definition.  # noqa: E501
        for scale in (float(price.mean()), portfolio_value):
            scaled_price, scaled_target = price / scale, target / scale
            result = milp(
                np.concatenate([np.zeros(assets), np.ones(assets + 1)]),
                constraints=[
                    LinearConstraint(
                        np.block(
                            [
                                [np.diag(scaled_price), identity, no_cash],
                                [-np.diag(scaled_price), identity, no_cash],
                            ]
                        ),
                        np.concatenate([scaled_target, -scaled_target]),
                        np.inf,
                    ),
                    LinearConstraint(
                        np.concatenate([scaled_price, np.zeros(assets), [1.0]]),
                        portfolio_value / scale,
                        portfolio_value / scale,
                    ),
                ],
                integrality=np.concatenate([np.ones(assets), np.zeros(assets + 1)]),
                bounds=bounds,
                options={"time_limit": DISCRETE_ALLOCATION_TIME_LIMIT},
            )
            message = result.message
            if result.x is None:
                continue
            candidate = np.round(result.x[:assets])
            candidate_cash = portfolio_value - candidate @ price
            if candidate_cash < -FEASIBILITY_TOLERANCE * portfolio_value:
                message = "the allocation costs more than the portfolio value"
                continue
            found = True
            if (
                np.abs(target - candidate * price).sum() + candidate_cash
                <= np.abs(target - shares * price).sum() + cash
            ):
                shares, cash = candidate, candidate_cash
                break
        if not found:
            raise ValueError(f"No allocation was found: {message}")

    table = pd.DataFrame(
        {
            "Price": price,
            "Shares": shares,
            "Amount": shares * price,
            "Target Weight": weights.to_numpy(dtype=float),
            "Weight": shares * price / portfolio_value,
        },
        index=weights.index,
    )
    table.loc["Leftover Cash"] = [np.nan, np.nan, cash, 0.0, cash / portfolio_value]
    return table


def get_portfolio_risk_metrics(
    returns: pd.Series | pd.DataFrame,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    The return and risk of the return series of a portfolio, with the formulas of the
    Performance and Risk modules:

    - Annualized Return: the geometric average, Π(1 + r_t)^(P/T) - 1, with P the periods
      per year and T the number of periods.
    - Annualized Volatility: the standard deviation (ddof=1) times √P.
    - Sharpe Ratio: the average excess return over its standard deviation, times √P
      (Sharpe, 1966), with the annualized risk free rate spread evenly over the periods.
    - Sortino Ratio: the average excess return over its downside deviation below zero,
      times √P (Sortino and Price, 1994).
    - Maximum Drawdown: the largest fall of the compounded value from its peak, as a
      negative return.
    - Value at Risk: the 1 - α quantile of the returns per period (historical).
    - Conditional Value at Risk: the average of the returns at or below the Value at
      Risk per period (historical).

    Args:
        returns (pd.Series | pd.DataFrame): The returns per period, one column per
            portfolio.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.
        risk_free_rate (float, optional): The average annualized risk free rate over the
            periods. Defaults to 0.
        alpha (float, optional): The tail probability of the Value at Risk and the
            Conditional Value at Risk, e.g. 0.05 for the worst 5% of periods.
            Defaults to 0.05.

    Returns:
        pd.DataFrame: The measures as rows, one column per portfolio.
    """
    validate_alpha(alpha)
    returns = returns.to_frame() if isinstance(returns, pd.Series) else returns
    excess = returns - risk_free_rate / periods_per_year
    annualization = np.sqrt(periods_per_year)

    return pd.DataFrame(
        {
            "Annualized Return": (1 + returns).prod(min_count=1)
            ** (periods_per_year / returns.count())
            - 1,
            "Annualized Volatility": returns.std() * annualization,
            "Sharpe Ratio": performance_model.get_sharpe_ratio(excess) * annualization,
            "Sortino Ratio": performance_model.get_sortino_ratio(excess)
            * annualization,
            "Maximum Drawdown": risk_model.get_max_drawdown(returns),
            "Value at Risk": var_model.get_var_historic(returns, alpha),
            "Conditional Value at Risk": cvar_model.get_cvar_historic(returns, alpha),
        }
    ).T


# The uncertainty sets of the worst-case mean-variance optimizer and its objectives.
WORST_CASE_UNCERTAINTY = ("box", "ellipsoid")
WORST_CASE_OBJECTIVES = ("maximum_sharpe", "utility", "minimum_variance")

# How close two values of λ of the critical line algorithm must be to count as the same
# turning point, relative to the larger of them and 1.
CRITICAL_LINE_TOLERANCE = 1e-12


def get_critical_line_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    minimum_weight: float | dict[str, float] = 0.0,
    maximum_weight: float | dict[str, float] = 1.0,
) -> pd.DataFrame:
    """
    The corner portfolios of the efficient frontier with weight bounds and a budget.

    They are computed exactly with the critical line algorithm (Markowitz, 1956), next to
    the minimum variance and maximum Sharpe ratio portfolios on that frontier.

    Every efficient portfolio minimizes ½ w'Σw - λ μ'w subject to Σ w_i = 1 and
    l_i ≤ w_i ≤ u_i for some λ ≥ 0. For a fixed split of the assets into free assets F
    (strictly between their bounds) and bound assets B (at a bound), the first-order
    conditions on the free assets are linear:

    - Σ_FF w_F + γ 1 = λ μ_F - Σ_FB w_B and 1'w_F = 1 - 1'w_B,

    so w_F and the multiplier γ of the budget are affine in λ, and so is the gradient
    g_j = (Σw)_j - λ μ_j + γ of each bound asset, which must stay ≥ 0 at a lower bound
    and ≤ 0 at an upper bound. Starting from the portfolio with the highest expected
    return (λ = ∞), λ is lowered until a free asset reaches a bound or the gradient of a
    bound asset changes sign, which frees it. Each such λ is a corner (turning point)
    where the set of free assets changes; between two corners the weights move along a
    straight line. The last corner, at λ = 0, is the minimum variance portfolio.

    Along the line w(t) = (1 - t) w_k + t w_k+1 between two corners the Sharpe ratio
    (m0 + t m1) / √(v0 + 2 t v1 + t² v2) has a single stationary point,
    t* = (m0 v1 - m1 v0) / (m1 v1 - m0 v2), so the maximum Sharpe ratio portfolio is
    found exactly by comparing the corners with these points.

    See Markowitz, H. (1956), "The Optimization of a Quadratic Function Subject to
    Linear Constraints", Naval Research Logistics Quarterly, 3(1-2), 111-133, and
    Bailey, D. H. and López de Prado, M. (2013), "An Open-Source Implementation of the
    Critical-Line Algorithm for Portfolio Optimization", Algorithms, 6(1), 169-196.

    Referenced against:
    - PyPortfolioOpt, CLA: https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html#pypfopt.cla.CLA

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix, positive definite.
        risk_free_rate (float, optional): The annualized risk free rate, for the Sharpe
            ratios. Defaults to 0.
        minimum_weight (float | dict[str, float], optional): The lowest weight of every
            asset, or per asset ({"AAPL": 0.05}, the others 0); a negative value allows
            a short position. Defaults to 0.
        maximum_weight (float | dict[str, float], optional): The highest weight of every
            asset, or per asset ({"AAPL": 0.3}, the others 1). Defaults to 1.

    Returns:
        pd.DataFrame: One row per corner portfolio ("Corner 1" has the highest expected
        return, the last one is the minimum variance portfolio), followed by the
        "Minimum Variance" and "Maximum Sharpe" portfolios, with the Lambda (λ, the
        risk tolerance at which the portfolio is optimal), the Expected Return,
        Volatility, Sharpe Ratio and the weight per asset.
    """
    _validate_covariance(covariance)
    assets = covariance.index
    if len(assets) < 2:  # noqa: PLR2004
        raise ValueError("At least two assets are needed.")
    mu = expected_returns.reindex(assets).to_numpy(dtype=float)
    if not np.all(np.isfinite(mu)):
        raise ValueError("Every asset needs an expected return.")
    sigma = covariance.to_numpy(dtype=float)
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError(
            "The critical line algorithm needs a positive definite covariance matrix."
        )
    constraints = get_weight_constraints(assets, minimum_weight, maximum_weight)
    lower = np.array([bound[0] for bound in constraints.bounds])
    upper = np.array([bound[1] for bound in constraints.bounds])
    count = len(assets)

    # The portfolio with the highest expected return: every asset at its lower bound,
    # then the budget left goes to the assets with the highest expected returns, each up
    # to its upper bound. The asset that takes the last of the budget is free.
    weights = lower.copy()
    status = np.full(count, -1)  # -1 at the lower bound, 1 at the upper bound, 0 free
    budget = 1 - lower.sum()
    for i in np.argsort(-mu, kind="stable"):
        if upper[i] <= lower[i]:
            continue
        step = min(upper[i] - lower[i], budget)
        weights[i] += step
        budget -= step
        if budget <= ZERO_WEIGHT:
            status[i] = 0
            break
        status[i] = 1

    corners: list[tuple[float, np.ndarray]] = []
    current = np.inf
    moved = -1
    for _ in range(10 * count + 100):
        free = np.flatnonzero(status == 0)
        bound = np.flatnonzero(status != 0)

        # The weights w(λ) = intercept + λ slope and the budget multiplier γ(λ) from the
        # first-order conditions of the free assets, as a bordered linear system.
        system = np.zeros((len(free) + 1, len(free) + 1))
        system[:-1, :-1] = sigma[np.ix_(free, free)]
        system[:-1, -1] = 1
        system[-1, :-1] = 1
        held = weights[bound]
        solution = np.linalg.solve(
            system,
            np.column_stack(
                [
                    np.append(-sigma[np.ix_(free, bound)] @ held, 1 - held.sum()),
                    np.append(mu[free], 0.0),
                ]
            ),
        )
        intercept, slope = np.zeros(count), np.zeros(count)
        intercept[bound] = held
        intercept[free], slope[free] = solution[:-1, 0], solution[:-1, 1]
        gradient_intercept = sigma[bound] @ intercept + solution[-1, 0]
        gradient_slope = sigma[bound] @ slope - mu[bound] + solution[-1, 1]

        # The λ at which each free asset reaches the bound it moves towards as λ falls
        # (only with two or more free assets: a single one holds the rest of the
        # budget), and at which the gradient of each bound asset changes sign.
        events: list[tuple[float, int]] = []
        if len(free) > 1:
            for i in free:
                if slope[i] > 0:
                    events.append(((lower[i] - intercept[i]) / slope[i], i))
                elif slope[i] < 0:
                    events.append(((upper[i] - intercept[i]) / slope[i], i))
        for j, constant, rate in zip(
            bound, gradient_intercept, gradient_slope, strict=True
        ):
            if upper[j] > lower[j] and rate * status[j] < 0:
                events.append((-constant / rate, j))

        # The next corner is the largest λ below the current one; an event at the same
        # λ (several assets changing at once) counts, except for the asset that just
        # changed.
        tolerance = CRITICAL_LINE_TOLERANCE * max(
            1.0, abs(current) if np.isfinite(current) else 1.0
        )
        candidates = [
            (value, asset)
            for value, asset in events
            if value < current - tolerance
            or (value <= current + tolerance and asset != moved)
        ]
        value, asset = max(candidates, default=(-np.inf, -1))
        if value <= 0:
            corners.append((0.0, intercept.copy()))
            break
        weights = intercept + value * slope
        corners.append((value, weights.copy()))
        if status[asset] == 0:
            status[asset] = -1 if slope[asset] > 0 else 1
            weights[asset] = lower[asset] if status[asset] == -1 else upper[asset]
        else:
            status[asset] = 0
        current, moved = value, asset
    else:
        raise ValueError("The critical line algorithm did not reach λ = 0.")

    # Several assets changing at the same λ give the same corner more than once.
    unique = [corners[0]]
    for corner in corners[1:]:
        if np.max(np.abs(corner[1] - unique[-1][1])) > ZERO_WEIGHT:
            unique.append(corner)
        else:
            unique[-1] = corner

    # The maximum Sharpe ratio along every line between two neighbouring corners.
    best = (-np.inf, unique[0][0], unique[0][1])
    for (start_lambda, start), (end_lambda, end) in zip(
        unique, unique[1:] + [unique[-1]], strict=True
    ):
        step = end - start
        m0, m1 = start @ mu - risk_free_rate, step @ mu
        v0, v1, v2 = start @ sigma @ start, start @ sigma @ step, step @ sigma @ step
        denominator = m1 * v1 - m0 * v2
        points = [0.0, 1.0]
        if abs(denominator) > 0:
            stationary = (m0 * v1 - m1 * v0) / denominator
            if 0 < stationary < 1:
                points.append(stationary)
        for t in points:
            variance = v0 + 2 * t * v1 + t * t * v2
            if variance <= ZERO_VOLATILITY_TOLERANCE**2:
                continue
            sharpe = (m0 + t * m1) / np.sqrt(variance)
            if sharpe > best[0]:
                best = (
                    sharpe,
                    (1 - t) * start_lambda + t * end_lambda,
                    start + t * step,
                )

    labels = [f"Corner {number}" for number in range(1, len(unique) + 1)]
    rows = []
    for risk_tolerance, corner in [
        *unique,
        unique[-1],
        (best[1], best[2]),
    ]:
        portfolio = _as_series(corner, assets)
        rows.append(
            pd.concat(
                [
                    pd.Series({"Lambda": risk_tolerance}),
                    get_portfolio_statistics(
                        portfolio, expected_returns, covariance, risk_free_rate
                    ),
                    portfolio,
                ]
            )
        )
    table = pd.DataFrame(rows, index=[*labels, "Minimum Variance", "Maximum Sharpe"])
    table.index.name = "Portfolio"

    return table


def get_worst_case_return(
    weights: pd.Series,
    expected_returns: pd.Series,
    mean_covariance: pd.DataFrame,
    uncertainty: str = "box",
    confidence_level: float = 0.95,
) -> float:
    """
    The lowest expected return of a portfolio over an uncertainty set of the expected
    returns centred on the estimate μ.

    The set follows Tütüncü and Koenig, 2004; Goldfarb and Iyengar, 2003), with Σ_μ the
    covariance of the estimate and α the confidence level:

    - A box: every expected return within μ_i ± δ_i, δ_i = z_(1+α)/2 √(Σ_μ,ii), the
      two-sided confidence interval of each estimate on its own. The worst case is μ'w -
      δ'|w|.
    - An ellipsoid: the expected returns within (m - μ)'Σ_μ⁻¹(m - μ) ≤ κ², with κ² =
      χ²_n(α), the joint confidence region of the n estimates. The worst case is μ'w - κ
      √(w'Σ_μ w).

    Args:
        weights (pd.Series): The weight per asset.
        expected_returns (pd.Series): The expected return per asset.
        mean_covariance (pd.DataFrame): The covariance of the estimate of the expected
            returns, e.g. the covariance of the returns divided by the number of periods
            for the historical average, in the same units as the expected returns.
        uncertainty (str, optional): The uncertainty set. Options are: "box" (a confidence interval
            per estimate), "ellipsoid" (the joint confidence region). Defaults to "box".
        confidence_level (float, optional): The confidence level of the uncertainty set.
            Defaults to 0.95.

    Returns:
        float: The worst-case expected return.
    """
    check_option("uncertainty", uncertainty, WORST_CASE_UNCERTAINTY)
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")
    assets = mean_covariance.index
    w = weights.reindex(assets).fillna(0).to_numpy(dtype=float)
    mu = expected_returns.reindex(assets).to_numpy(dtype=float)
    uncertain = mean_covariance.to_numpy(dtype=float)
    if uncertainty == "box":
        half_width = stats.norm.ppf((1 + confidence_level) / 2) * np.sqrt(
            np.diag(uncertain)
        )
        return float(mu @ w - half_width @ np.abs(w))
    radius = np.sqrt(stats.chi2.ppf(confidence_level, len(assets)))
    return float(mu @ w - radius * np.sqrt(max(w @ uncertain @ w, 0.0)))


def get_worst_case_mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    mean_covariance: pd.DataFrame,
    uncertainty: str = "box",
    confidence_level: float = 0.95,
    objective: str = "maximum_sharpe",
    risk_free_rate: float = 0.0,
    risk_aversion: float = 2.5,
    target_return: float | None = None,
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
    The robust mean-variance portfolio that is optimal for the worst expected returns around
    the estimate.

    The uncertainty set follows Tütüncü and Koenig (2004) and Goldfarb and Iyengar (2003).

    The expected return μ'w of the mean-variance objectives is replaced by its worst case
    R(w) over the set (see get_worst_case_return): μ'w - δ'|w| for a box and μ'w - κ √(w'Σ_μ
    w) for an ellipsoid. Both are concave in w, so the problems stay convex:

    - The maximum Sharpe ratio: maximize (R(w) - n r_f) / √(w'Σw) with n = Σw the net
      exposure, solved with the Charnes-Cooper transformation y = κw, κ ≥ 0 (Charnes and
      Cooper, 1962): maximize R(y) - n r_f κ subject to y'Σy ≤ 1 and the weight constraints
      scaled by κ, then w = y / κ.
    - The utility: maximize R(w) - (λ/2) w'Σw with λ the risk aversion.
    - The minimum variance: minimize w'Σw subject to R(w) + (1 - n) r_f ≥ the target return.

    The part of the portfolio that is not invested, 1 - n, is cash that earns the risk
    free rate without uncertainty (borrowed at it above a net exposure of one), as in
    get_portfolio_statistics, so the worst-case return of the portfolio is
    R(w) + (1 - n) r_f and its worst-case Sharpe ratio (R(w) - n r_f) / √(w'Σw).

    With Σ_μ proportional to Σ (such as Σ / T) the ellipsoid lowers every Sharpe ratio
    by the same κ √(w'Σ_μ w) / √(w'Σw), so its maximum Sharpe portfolio is the ordinary
    one; the box, the utility and the minimum variance objectives do change the weights.
    The covariance Σ is taken as known: uncertainty in Σ needs semidefinite
    programming, which the SciPy solvers do not provide.

    See Tütüncü, R. H. and Koenig, M. (2004), "Robust Asset Allocation", Annals of
    Operations Research, 132, 157-187, and Goldfarb, D. and Iyengar, G. (2003), "Robust
    Portfolio Selection Problems", Mathematics of Operations Research, 28(1), 1-38.

    Referenced against:
    - Riskfolio-Lib, Portfolio.wc_optimization (box): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html#Portfolio.Portfolio.wc_optimization
    - skfolio, MeanRisk (with an EmpiricalMuUncertaintySet): https://skfolio.org/generated/skfolio.optimization.MeanRisk.html

    Args:
        expected_returns (pd.Series): The annualized expected return per asset.
        covariance (pd.DataFrame): The annualized covariance matrix.
        mean_covariance (pd.DataFrame): The covariance of the estimate of the expected
            returns, in the same units; for the annualized historical average it is the
            annualized covariance times the periods per year over the number of periods.
        uncertainty (str, optional): The uncertainty set. Options are: "box" (a confidence interval
            per estimate), "ellipsoid" (the joint confidence region). Defaults to "box".
        confidence_level (float, optional): The confidence level of the uncertainty set.
            Defaults to 0.95.
        objective (str, optional): Options are: "maximum_sharpe" (the highest worst-case Sharpe
            ratio), "utility" (the highest worst-case utility), "minimum_variance" (the lowest
            variance at a worst-case target return). Defaults to "maximum_sharpe".
        risk_free_rate (float, optional): The annualized risk free rate. Defaults to 0.
        risk_aversion (float, optional): The risk aversion λ of the utility. Defaults to
            2.5.
        target_return (float | None, optional): The lowest annualized worst-case
            return of the portfolio, cash included, of the minimum variance objective,
            which needs it. Defaults to None.
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
    check_option("uncertainty", uncertainty, WORST_CASE_UNCERTAINTY)
    check_option("objective", objective, WORST_CASE_OBJECTIVES)
    if (target_return is None) == (objective == "minimum_variance"):
        raise ValueError(
            "A target return goes with the minimum_variance objective, which needs one."
        )
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")
    _validate_covariance(covariance)
    assets = covariance.index
    mu = expected_returns.reindex(assets).to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    uncertain = mean_covariance.reindex(index=assets, columns=assets).to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(uncertain)):
        raise ValueError("The covariance of the expected returns has missing values.")
    count = len(assets)
    scale = np.mean(np.diag(sigma)) or 1.0

    arguments = (
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
    constraints = get_weight_constraints(*arguments)
    short = any(bound[0] < 0 for bound in constraints.bounds[:count])
    # δ'|w| of the box needs the absolute weights t_i ≥ |w_i| when short positions are
    # allowed; a gross exposure no portfolio within the bounds can exceed adds them.
    if uncertainty == "box" and short and gross_exposure is None:
        constraints = get_weight_constraints(
            *arguments[:6],
            float(sum(max(abs(low), abs(high)) for low, high in constraints.bounds)),
            current_weights,
            maximum_turnover,
        )
    variables = len(constraints.bounds)

    # The worst-case return and its gradient over all variables x of the constraints:
    # linear for the box (on the absolute weights t when short positions are allowed),
    # μ'w - κ√(w'Σ_μ w) for the ellipsoid. Both are positively homogeneous.
    if uncertainty == "box":
        half_width = stats.norm.ppf((1 + confidence_level) / 2) * np.sqrt(
            np.diag(uncertain)
        )
        linear = np.zeros(variables)
        linear[:count] = mu
        if short:
            linear[count : 2 * count] = -half_width
        else:
            linear[:count] -= half_width

        def worst(x):
            return linear @ x[: len(linear)]

        def worst_gradient(x):
            return np.concatenate([linear, np.zeros(len(x) - len(linear))])

    else:
        radius = np.sqrt(stats.chi2.ppf(confidence_level, count))

        def worst(x):
            weights = x[:count]
            return mu @ weights - radius * np.sqrt(
                max(weights @ uncertain @ weights, 0)
            )

        def worst_gradient(x):
            weights = x[:count]
            spread = np.sqrt(max(weights @ uncertain @ weights, 0))
            slope = np.zeros(len(x))
            slope[:count] = mu - (
                radius * (uncertain @ weights) / spread if spread > 0 else 0
            )
            return slope

    def variance_gradient(x, factor):
        return np.concatenate([factor * (sigma @ x[:count]), np.zeros(len(x) - count)])

    # SLSQP stops on an absolute change of the objective (ftol), so the returns and
    # variances are expressed in units of the average variance, as in
    # get_mean_variance_weights, which keeps the objectives of order one.
    if objective == "utility":
        weights = _solve(
            lambda x: (-worst(x) + risk_aversion / 2 * (x[:count] @ sigma @ x[:count]))
            / scale,
            lambda x: (-worst_gradient(x) + variance_gradient(x, risk_aversion))
            / scale,
            constraints,
            all_variables=True,
        )
        return _as_series(weights, assets)

    # The highest worst-case return within the constraints bounds the target and must
    # beat the risk free rate for a positive worst-case Sharpe ratio. The point keeps
    # the absolute weights and trades at |w| and |w - w0|, their least values.
    weights = _solve(
        lambda x: -worst(x) / scale,
        lambda x: -worst_gradient(x) / scale,
        constraints,
        all_variables=True,
    )
    trades = (
        [np.abs(weights - constraints.current)]
        if constraints.current is not None
        else []
    )
    absolute = [np.abs(weights)] if variables - count * (1 + len(trades)) > 0 else []
    highest_point = np.concatenate([weights, *absolute, *trades])
    highest = float(worst(highest_point))

    if objective == "minimum_variance" and target_return is not None:
        cash_return = (1 - net_exposure) * risk_free_rate
        if target_return > highest + cash_return + 1e-9:
            raise ValueError(
                f"A worst-case return of {target_return:.4f} cannot be reached within "
                f"the constraints (at most {highest + cash_return:.4f})."
            )
        target = min(target_return - cash_return, highest)
        if uncertainty == "box":
            # The box's worst case is linear, so R(x) ≥ target is one more row A x ≤ b.
            constraints = constraints._replace(
                inequality=np.vstack(
                    [constraints.inequality.reshape(-1, variables), -linear[None, :]]
                ),
                upper=np.append(constraints.upper, -target),
            )
            extra = None
        else:
            extra = [
                {
                    "type": "ineq",
                    "fun": lambda w: worst(w) - target,
                    "jac": worst_gradient,
                }
            ]
        weights = _solve(
            lambda w: w @ sigma @ w / scale,
            lambda w: 2 * sigma @ w / scale,
            constraints,
            extra,
        )
        return _as_series(weights, assets)

    if highest <= risk_free_rate * net_exposure:
        raise ValueError(
            "No portfolio has a worst-case return above the risk free rate, so no "
            "portfolio has a positive worst-case Sharpe ratio; a lower confidence level "
            "narrows the uncertainty set."
        )

    # Charnes-Cooper: over z = (y, κ) maximize R(y) - n r_f κ subject to y'Σy ≤ 1 (in
    # units of the average variance) and the homogeneous weight constraints, starting
    # from the portfolio with the highest worst-case return scaled onto y'Σy = 1.
    scaled = sigma / scale
    homogeneous, homogeneous_equality = _homogeneous_constraints(constraints)
    size = np.sqrt(highest_point[:count] @ scaled @ highest_point[:count])
    if size <= 0:
        raise ValueError(
            "The portfolio with the highest worst-case return has no risk."
        )
    start = np.append(highest_point, 1.0) / size
    # The objective in units of its value at the start, which is positive.
    unit = (highest - risk_free_rate * net_exposure) / size
    result = minimize(
        lambda z: -(worst(z[:-1]) - risk_free_rate * net_exposure * z[-1]) / unit,
        start,
        jac=lambda z: -np.append(worst_gradient(z[:-1]), -risk_free_rate * net_exposure)
        / unit,
        method="SLSQP",
        bounds=[(None, None)] * variables + [(0, None)],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda z: 1 - z[:count] @ scaled @ z[:count],
                "jac": lambda z: np.concatenate(
                    [-2 * scaled @ z[:count], np.zeros(variables + 1 - count)]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda z: -(homogeneous @ z),
                "jac": lambda z: -homogeneous,
            },
            {
                "type": "eq",
                "fun": lambda z: homogeneous_equality @ z,
                "jac": lambda z: homogeneous_equality,
            },
        ],
        options=OPTIMIZER_OPTIONS,
    )
    kappa = result.x[-1]
    if kappa <= 0 or not (
        result.success
        or (
            result.status == SLSQP_LINE_SEARCH_STATUS
            and _is_feasible(
                result.x[:-1] / kappa,
                constraints.bounds,
                _linear_constraints(constraints),
            )
        )
    ):
        raise ValueError(f"The optimization did not converge: {result.message}")

    return _as_series(result.x[:count] / kappa, assets)


class WassersteinProgramme(NamedTuple):
    """
    The rows of the distributionally robust mean-CVaR linear programme over the variables
    (w, τ, λ, s, y).

    The variables are the weights w (n), the CVaR threshold τ, the multiplier λ of the
    Wasserstein ball, the worst-case loss s_t per period (T) and the shifts y_kj of the two
    loss pieces per asset (2n). The rows are A (w, τ, λ, s, y) ≤ 0.
    """

    cost: np.ndarray
    weights: sparse.csr_matrix
    auxiliary: sparse.csr_matrix
    bounds: list[tuple[float | None, float | None]]


def get_wasserstein_cvar_programme(
    returns: np.ndarray,
    alpha: float = 0.05,
    wasserstein_radius: float = 0.0005,
    risk_aversion: float = 1.0,
    mean_weight: float = 1.0,
) -> WassersteinProgramme:
    """
    The linear programme of the worst-case mean-CVaR objective within a Wasserstein ball.

    It is the worst-case expectation of -m X + ρ (τ + max(-X - τ, 0) / α), with X = ξ'w the
    portfolio return, over every distribution of the asset returns ξ within a type-1
    Wasserstein ball of radius ε around the T historical returns (Mohajerin Esfahani and
    Kuhn, 2018). The transport cost is the 1-norm of the change in the returns, and returns
    cannot fall below -100% (ξ ≥ -1). Minimized over τ it is the worst case of -m E[X] + ρ
    CVaR_α(-X).

    The loss is the maximum of two linear pieces ℓ_k(ξ) = c_k w'ξ + b_k τ, with
    c_1 = -m, b_1 = ρ, c_2 = -m - ρ / α and b_2 = ρ (1 - 1 / α), the pieces of
    the paper's Section 7.1. By its duality (Theorem 4.2; Corollary
    5.1 for piecewise affine losses on a polyhedral support) the worst case equals
    min λε + Σ s_t / T over λ ≥ 0 with s_t ≥ sup_(ξ ≥ -1) ℓ_k(ξ) - λ‖ξ - ξ_t‖₁ for
    both pieces. The supremum separates per asset: moving ξ_j up gains c_k w_j - λ per
    unit, which must not be positive (λ ≥ c_k w_j), and moving it down to -1 gains
    (-c_k w_j - λ)(1 + ξ_tj) when that is positive. With y_kj ≥ max(0, -c_k w_j - λ),
    which does not depend on the period, the rows are:

    - c_k ξ_t'w + b_k τ + Σ_j (1 + ξ_tj) y_kj - s_t ≤ 0 per period and piece.
    - -c_k w_j - λ - y_kj ≤ 0 and c_k w_j - λ ≤ 0 per asset and piece.

    The programme has n + 2 + T + 2n variables and 2T + 4n rows.

    See Mohajerin Esfahani, P. and Kuhn, D. (2018), "Data-Driven Distributionally
    Robust Optimization Using the Wasserstein Metric: Performance Guarantees and
    Tractable Reformulations", Mathematical Programming, 171(1-2), 115-166.

    Args:
        returns (np.ndarray): The returns per period (rows) and asset (columns), above
            -100%.
        alpha (float, optional): The tail probability α of the CVaR, e.g. 0.05 for the
            worst 5% of periods. Defaults to 0.05.
        wasserstein_radius (float, optional): The radius ε of the Wasserstein ball, in
            the units of the returns per period. Defaults to 0.0005.
        risk_aversion (float, optional): The weight ρ of the CVaR. Defaults to 1.
        mean_weight (float, optional): The weight m of the expected loss, 1 for the
            mean-CVaR objective and 0 for the CVaR alone. Defaults to 1.

    Returns:
        WassersteinProgramme: The cost of every variable, the rows' coefficients of the
            weights and of (τ, λ, s, y), and the bounds of (τ, λ, s, y).
    """
    validate_alpha(alpha)
    if wasserstein_radius < 0:
        raise ValueError("The Wasserstein radius cannot be negative.")
    if risk_aversion < 0 or mean_weight < 0:
        raise ValueError(
            "The risk aversion and the weight of the mean cannot be negative."
        )
    periods, assets = returns.shape
    if np.any(returns < -1):
        raise ValueError("A return below -100% is not possible.")
    tail = 1 / alpha
    slopes = np.array([-mean_weight, -mean_weight - risk_aversion * tail])
    intercepts = np.array([risk_aversion, risk_aversion * (1 - tail)])
    growth = sparse.csr_matrix(1 + returns)
    identity = sparse.eye(assets, format="csr")
    period_identity = sparse.eye(periods, format="csr")
    assets_column = sparse.csr_matrix(np.ones((assets, 1)))

    weight_rows, auxiliary_rows = [], []
    for piece, (slope, intercept) in enumerate(zip(slopes, intercepts, strict=True)):
        shifts = [sparse.csr_matrix((periods, assets))] * 2
        shifts[piece] = growth
        # Per period: c_k r_t'w + b_k τ + (1 + r_t)'y_k - s_t ≤ 0.
        weight_rows.append(sparse.csr_matrix(slope * returns))
        auxiliary_rows.append(
            sparse.hstack(
                [
                    sparse.csr_matrix(np.full((periods, 1), intercept)),
                    sparse.csr_matrix((periods, 1)),
                    -period_identity,
                    *shifts,
                ]
            )
        )
    for piece, slope in enumerate(slopes):
        shift = [sparse.csr_matrix((assets, assets))] * 2
        shift[piece] = -identity
        # Per asset: -c_k w_j - λ - y_kj ≤ 0, then c_k w_j - λ ≤ 0.
        weight_rows += [-slope * identity, slope * identity]
        auxiliary_rows += [
            sparse.hstack(
                [
                    sparse.csr_matrix((assets, 1)),
                    -assets_column,
                    sparse.csr_matrix((assets, periods)),
                    *shift,
                ]
            ),
            sparse.hstack(
                [
                    sparse.csr_matrix((assets, 1)),
                    -assets_column,
                    sparse.csr_matrix((assets, periods + 2 * assets)),
                ]
            ),
        ]

    return WassersteinProgramme(
        np.concatenate(
            [
                [0.0, wasserstein_radius],
                np.full(periods, 1 / periods),
                np.zeros(2 * assets),
            ]
        ),
        sparse.vstack(weight_rows, format="csr"),
        sparse.vstack(auxiliary_rows, format="csr"),
        [(None, None), (0, None)]
        + [(None, None)] * periods
        + [(0, None)] * (2 * assets),
    )


def get_worst_case_cvar(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
    wasserstein_radius: float = 0.0005,
) -> float:
    """
    The highest CVaR of a portfolio over every distribution of the asset returns within a
    Wasserstein ball of radius ε around the historical returns.

    The transport cost is the 1-norm as the transport cost and returns of at least -100%
    (Mohajerin Esfahani and Kuhn, 2018). It is the programme of
    get_wasserstein_cvar_programme with the weights fixed, without the expected loss (m = 0)
    and with ρ = 1; at a radius of 0 it is the historical CVaR of get_expected_tail_loss.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        weights (pd.Series): The weight per asset.
        alpha (float, optional): The tail probability of the CVaR, e.g. 0.05 for the
            worst 5% of periods. Defaults to 0.05.
        wasserstein_radius (float, optional): The radius ε of the Wasserstein ball, in
            the units of the returns per period. Defaults to 0.0005.

    Returns:
        float: The worst-case CVaR, positive for a loss.
    """
    data = returns.dropna(how="any").to_numpy(dtype=float)
    if len(data) == 0:
        raise ValueError("No period has a return for every asset.")
    programme = get_wasserstein_cvar_programme(
        data, alpha, wasserstein_radius, 1.0, 0.0
    )
    fixed = programme.weights @ weights.reindex(returns.columns).fillna(0).to_numpy(
        dtype=float
    )
    result = linprog(
        programme.cost,
        A_ub=programme.auxiliary,
        b_ub=-fixed,
        bounds=programme.bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"The optimization did not converge: {result.message}")
    return float(result.fun)


def get_distributionally_robust_cvar_weights(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    wasserstein_radius: float = 0.0005,
    risk_aversion: float = 1.0,
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
    The portfolio with the lowest mean-CVaR objective under the least favourable
    distribution near the historical returns.

    It minimizes the worst case of -E[X] + ρ CVaR_α(-X), with X the portfolio return, over
    every distribution within a type-1 Wasserstein ball of radius ε around the historical
    returns (Mohajerin Esfahani and Kuhn, 2018, Section 7.1), solved exactly as one linear
    programme with HiGHS (see get_wasserstein_cvar_programme). At a radius of 0 it is the
    historical mean-CVaR portfolio; a larger radius guards against distributions that put
    more weight on losses than the sample did.

    See Mohajerin Esfahani, P. and Kuhn, D. (2018), "Data-Driven Distributionally
    Robust Optimization Using the Wasserstein Metric: Performance Guarantees and
    Tractable Reformulations", Mathematical Programming, 171(1-2), 115-166.

    Referenced against:
    - skfolio, DistributionallyRobustCVaR: https://skfolio.org/generated/skfolio.optimization.DistributionallyRobustCVaR.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        alpha (float, optional): The tail probability α of the CVaR, e.g. 0.05 for the
            average of the worst 5% of periods. Defaults to 0.05.
        wasserstein_radius (float, optional): The radius ε of the Wasserstein ball: the
            average total change in the returns of the assets (the 1-norm) per period that
            the distributions considered may have. Defaults to 0.0005.
        risk_aversion (float, optional): The weight ρ of the CVaR against the expected
            loss. Defaults to 1.
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
        current_weights,
        maximum_turnover,
    )
    programme = get_wasserstein_cvar_programme(
        data, alpha, wasserstein_radius, risk_aversion
    )
    count = constraints.assets
    variables = len(constraints.bounds)
    extra = programme.auxiliary.shape[1]
    # The variables of the weight constraints (weights first), then (τ, λ, s, y).
    inequality = sparse.vstack(
        [
            sparse.hstack(
                [
                    programme.weights,
                    sparse.csr_matrix((programme.weights.shape[0], variables - count)),
                    programme.auxiliary,
                ]
            ),
            sparse.hstack(
                [
                    sparse.csr_matrix(constraints.inequality.reshape(-1, variables)),
                    sparse.csr_matrix((len(constraints.upper), extra)),
                ]
            ),
        ],
        format="csr",
    )
    result = linprog(
        np.concatenate([np.zeros(variables), programme.cost]),
        A_ub=inequality,
        b_ub=np.concatenate([np.zeros(programme.weights.shape[0]), constraints.upper]),
        A_eq=np.hstack([constraints.equality, np.zeros((1, extra))]),
        b_eq=constraints.target,
        bounds=constraints.bounds + programme.bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"The optimization did not converge: {result.message}")

    return _as_series(result.x[:count], returns.columns)
