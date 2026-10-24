"""Portfolio Optimization Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from financetoolkit.portfolio import optimization_model

# pylint: disable=missing-function-docstring

ASSETS = ["A", "B", "C", "D"]
COVARIANCE = pd.DataFrame(
    [
        [0.04, 0.006, 0.01, 0.002],
        [0.006, 0.09, 0.012, 0.0],
        [0.01, 0.012, 0.0625, 0.004],
        [0.002, 0.0, 0.004, 0.0225],
    ],
    index=ASSETS,
    columns=ASSETS,
)
EXPECTED_RETURNS = pd.Series([0.08, 0.14, 0.10, 0.05], index=ASSETS)


def test_minimum_variance_matches_the_closed_form():
    inverse = np.linalg.inv(COVARIANCE.to_numpy())
    closed_form = inverse @ np.ones(4) / (np.ones(4) @ inverse @ np.ones(4))

    weights = optimization_model.get_minimum_variance_weights(COVARIANCE, -1, 1)

    assert np.allclose(weights, closed_form, atol=1e-5)


def test_maximum_sharpe_beats_every_frontier_portfolio():
    weights = optimization_model.get_maximum_sharpe_weights(
        EXPECTED_RETURNS, COVARIANCE, risk_free_rate=0.02
    )
    sharpe = optimization_model.get_portfolio_statistics(
        weights, EXPECTED_RETURNS, COVARIANCE, 0.02
    )["Sharpe Ratio"]
    frontier = optimization_model.get_efficient_frontier(
        EXPECTED_RETURNS, COVARIANCE, 0.02, points=15
    )

    assert sharpe >= frontier["Sharpe Ratio"].max() - 1e-6
    assert weights.sum() == pytest.approx(1)


def test_mean_variance_reaches_its_target_and_respects_the_bounds():
    weights = optimization_model.get_mean_variance_weights(
        EXPECTED_RETURNS, COVARIANCE, target_return=0.1, maximum_weight=0.4
    )

    assert weights @ EXPECTED_RETURNS == pytest.approx(0.1, abs=1e-6)
    assert weights.max() <= 0.4 + 1e-9

    with pytest.raises(ValueError, match="cannot be reached"):
        optimization_model.get_mean_variance_weights(
            EXPECTED_RETURNS, COVARIANCE, target_return=0.5
        )
    with pytest.raises(ValueError, match="cannot sum to one"):
        optimization_model.get_minimum_variance_weights(COVARIANCE, 0, 0.2)


def test_risk_parity_gives_equal_and_budgeted_contributions():
    weights = optimization_model.get_risk_parity_weights(COVARIANCE)
    shares = optimization_model.get_risk_contributions(weights, COVARIANCE)[
        "Percentage Contribution"
    ]
    assert np.allclose(shares, 0.25, atol=1e-6)

    budget = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
    weights = optimization_model.get_risk_parity_weights(COVARIANCE, budget)
    shares = optimization_model.get_risk_contributions(weights, COVARIANCE)[
        "Percentage Contribution"
    ]
    assert np.allclose(shares, budget, atol=1e-6)


def test_hierarchical_risk_parity_and_inverse_volatility_are_fully_invested():
    hrp = optimization_model.get_hierarchical_risk_parity_weights(COVARIANCE)
    inverse = optimization_model.get_inverse_volatility_weights(COVARIANCE)

    assert hrp.sum() == pytest.approx(1)
    assert (hrp > 0).all()
    assert inverse["D"] == pytest.approx(
        (1 / 0.15) / sum(1 / np.sqrt(np.diag(COVARIANCE)))
    )


def test_minimum_cvar_lowers_the_tail_loss():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        rng.multivariate_normal(EXPECTED_RETURNS / 252, COVARIANCE / 252, 1500),
        columns=ASSETS,
    )

    def cvar(weights):
        portfolio = returns @ weights
        return -portfolio[portfolio <= portfolio.quantile(0.05)].mean()

    weights = optimization_model.get_minimum_cvar_weights(returns)

    assert weights.sum() == pytest.approx(1)
    assert cvar(weights) <= cvar(pd.Series(0.25, index=ASSETS)) + 1e-9


def test_black_litterman_without_views_returns_the_implied_returns():
    market = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
    result = optimization_model.get_black_litterman_returns(COVARIANCE, market)

    assert np.allclose(result["Implied Return"], 2.5 * COVARIANCE @ market)
    assert np.allclose(result["Implied Return"], result["Black-Litterman Return"])

    # A confident view pulls the return almost all the way to it.
    confident = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, views={"D": 0.2}, view_confidences={"D": 0.999}
    )
    assert confident.loc["D", "Black-Litterman Return"] == pytest.approx(0.2, abs=1e-3)

    with pytest.raises(ValueError, match="names no asset"):
        optimization_model.get_black_litterman_returns(
            COVARIANCE, market, views={"Z": 0.1}
        )


def _random_problem(seed: int, assets: int) -> tuple[pd.Series, pd.DataFrame]:
    """A seeded random covariance matrix and expected returns."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(assets, assets + 2))
    covariance = factors @ factors.T / (assets + 2) * rng.uniform(0.01, 0.1)
    names = [f"A{i}" for i in range(assets)]
    return (
        pd.Series(rng.normal(0.08, 0.05, assets), index=names),
        pd.DataFrame(covariance, index=names, columns=names),
    )


def test_risk_parity_is_judged_by_its_risk_contributions(monkeypatch):
    # L-BFGS-B stops with "ABNORMAL" on this matrix while its solution matches the budget.
    _, covariance = _random_problem(425, 4)
    weights = optimization_model.get_risk_parity_weights(covariance)
    shares = optimization_model.get_risk_contributions(weights, covariance)[
        "Percentage Contribution"
    ]
    assert np.allclose(shares, 0.25, atol=1e-6)

    # A solution whose contributions miss the budget is still refused.
    class Unconverged:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        success = False
        message = "ABNORMAL"

    monkeypatch.setattr(
        optimization_model, "minimize", lambda *args, **kwargs: Unconverged()
    )
    with pytest.raises(ValueError, match="did not converge"):
        optimization_model.get_risk_parity_weights(covariance)


def test_maximum_sharpe_accepts_a_line_search_stop_at_the_optimum():
    # SLSQP answers "Positive directional derivative for linesearch" on this problem.
    expected_returns, covariance = _random_problem(453, 8)
    weights = optimization_model.get_maximum_sharpe_weights(
        expected_returns, covariance, risk_free_rate=0.02
    )
    sharpe = optimization_model.get_portfolio_statistics(
        weights, expected_returns, covariance, 0.02
    )["Sharpe Ratio"]

    # The tangency portfolio as a quadratic programme: the lowest variance for one unit
    # of excess return, scaled to sum to one.
    excess = expected_returns.to_numpy() - 0.02
    sigma = covariance.to_numpy()
    reference = minimize(
        lambda y: y @ sigma @ y,
        np.full(len(excess), 1 / len(excess)),
        jac=lambda y: 2 * sigma @ y,
        method="SLSQP",
        bounds=[(0, None)] * len(excess),
        constraints=[{"type": "eq", "fun": lambda y: y @ excess - 1}],
        options={"ftol": 1e-15, "maxiter": 2000},
    ).x
    reference_sharpe = optimization_model.get_portfolio_statistics(
        pd.Series(reference / reference.sum(), index=covariance.index),
        expected_returns,
        covariance,
        0.02,
    )["Sharpe Ratio"]

    assert weights.sum() == pytest.approx(1)
    assert sharpe == pytest.approx(reference_sharpe, abs=1e-8)


def test_a_line_search_stop_outside_the_constraints_is_refused(monkeypatch):
    class Infeasible:
        x = np.array([0.7, 0.7, 0.0, 0.0])
        success = False
        status = 8
        message = "Positive directional derivative for linesearch"

    monkeypatch.setattr(
        optimization_model, "minimize", lambda *args, **kwargs: Infeasible()
    )
    with pytest.raises(ValueError, match="did not converge"):
        optimization_model.get_minimum_variance_weights(COVARIANCE)


def test_efficient_frontier_ends_at_the_highest_return_corner():
    # SLSQP calls the constraints of the last point "incompatible" on this problem.
    expected_returns, covariance = _random_problem(2265, 4)
    frontier = optimization_model.get_efficient_frontier(
        expected_returns, covariance, 0.02, points=20, minimum_weight=-0.2
    )
    weights = frontier[covariance.index]

    assert len(frontier) == 20  # noqa: PLR2004
    assert frontier["Expected Return"].is_monotonic_increasing
    assert np.allclose(weights.sum(axis=1), 1)
    assert (weights >= -0.2 - 1e-8).all().all()
    # The highest return within the bounds: the maximum weight in the best asset, the
    # minimum weight in the two worst and the rest in the second best.
    ranked = expected_returns.sort_values(ascending=False)
    assert frontier["Expected Return"].iloc[-1] == pytest.approx(
        ranked.to_numpy() @ np.array([1.0, 0.4, -0.2, -0.2])
    )
    assert np.allclose(
        weights.iloc[-1][ranked.index], [1.0, 0.4, -0.2, -0.2], atol=1e-6
    )


def test_hierarchical_risk_parity_clusters_on_the_distance_of_distances():
    rng = np.random.default_rng(8)
    factors = rng.normal(size=(6, 3))
    names = list("ABCDEF")
    covariance = pd.DataFrame(
        factors @ factors.T + np.diag(rng.uniform(0.2, 1.0, 6)),
        index=names,
        columns=names,
    )
    volatility = np.sqrt(np.diag(covariance))
    distance = np.sqrt(
        np.clip((1 - covariance / np.outer(volatility, volatility)) / 2, 0, None)
    ).to_numpy(copy=True)
    np.fill_diagonal(distance, 0)

    def recursive_bisection(order: list[str]) -> pd.Series:
        def variance(items):
            sub = covariance.loc[items, items].to_numpy()
            inverse = 1 / np.diag(sub)
            inverse = inverse / inverse.sum()
            return inverse @ sub @ inverse

        weights = pd.Series(1.0, index=order)
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
                alpha = 1 - variance(left) / (variance(left) + variance(right))
                weights[left] *= alpha
                weights[right] *= 1 - alpha
        return weights.reindex(names)

    # López de Prado (2016) passes the square distance matrix to linkage, which clusters
    # on the Euclidean distance between its columns.
    paper = recursive_bisection(
        [names[i] for i in leaves_list(linkage(pdist(distance), "single"))]
    )
    direct = recursive_bisection(
        [
            names[i]
            for i in leaves_list(linkage(squareform(distance, checks=False), "single"))
        ]
    )

    weights = optimization_model.get_hierarchical_risk_parity_weights(covariance)

    assert np.allclose(weights, paper, atol=1e-10)
    assert not np.allclose(weights, direct, atol=1e-2)


def test_black_litterman_returns_are_total_returns():
    market = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
    excess = optimization_model.get_black_litterman_returns(COVARIANCE, market)
    total = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, risk_free_rate=0.03
    )
    assert np.allclose(total["Implied Return"], 2.5 * COVARIANCE @ market + 0.03)
    assert np.allclose(
        total["Implied Return"] - excess["Implied Return"], 0.03, atol=1e-12
    )

    # A relative view compares two total returns, so the risk free rate shifts every
    # posterior return by the same amount and the weights stay the same.
    relative = {"A-B": 0.02, "C-D": -0.01}
    without = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, views=relative
    )
    with_rate = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, views=relative, risk_free_rate=0.03
    )
    assert np.allclose(
        with_rate["Black-Litterman Return"] - without["Black-Litterman Return"],
        0.03,
        atol=1e-12,
    )
    assert np.allclose(
        optimization_model.get_mean_variance_weights(
            with_rate["Black-Litterman Return"], COVARIANCE
        ),
        optimization_model.get_mean_variance_weights(
            without["Black-Litterman Return"], COVARIANCE
        ),
        atol=1e-6,
    )
    # Without views the weights are the market weights, with or without the rate.
    assert np.allclose(
        optimization_model.get_mean_variance_weights(
            total["Black-Litterman Return"], COVARIANCE, minimum_weight=-1
        ),
        market,
        atol=1e-5,
    )

    # An absolute view is a total return: a confident view of 20% lands on 20%.
    confident = optimization_model.get_black_litterman_returns(
        COVARIANCE,
        market,
        views={"D": 0.2},
        view_confidences={"D": 0.999},
        risk_free_rate=0.03,
    )
    assert confident.loc["D", "Black-Litterman Return"] == pytest.approx(0.2, abs=1e-3)
