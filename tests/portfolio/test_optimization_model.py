"""Portfolio Optimization Model Tests"""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from financetoolkit.portfolio.construction.optimization import (
    estimation_model,
    optimization_model,
)
from financetoolkit.utilities.error_model import InvalidArgumentError

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
    result, _ = optimization_model.get_black_litterman_returns(COVARIANCE, market)

    assert np.allclose(result["Implied Return"], 2.5 * COVARIANCE @ market)
    assert np.allclose(result["Implied Return"], result["Black-Litterman Return"])

    # A confident view pulls the return almost all the way to it.
    confident, _ = optimization_model.get_black_litterman_returns(
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
    excess, _ = optimization_model.get_black_litterman_returns(COVARIANCE, market)
    total, _ = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, risk_free_rate=0.03
    )
    assert np.allclose(total["Implied Return"], 2.5 * COVARIANCE @ market + 0.03)
    assert np.allclose(
        total["Implied Return"] - excess["Implied Return"], 0.03, atol=1e-12
    )

    # A relative view compares two total returns, so the risk free rate shifts every
    # posterior return by the same amount and the weights stay the same.
    relative = {"A-B": 0.02, "C-D": -0.01}
    without, _ = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, views=relative
    )
    with_rate, _ = optimization_model.get_black_litterman_returns(
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
    confident, _ = optimization_model.get_black_litterman_returns(
        COVARIANCE,
        market,
        views={"D": 0.2},
        view_confidences={"D": 0.999},
        risk_free_rate=0.03,
    )
    assert confident.loc["D", "Black-Litterman Return"] == pytest.approx(0.2, abs=1e-3)


def test_minimum_cvar_weights_use_sparse_constraints_for_long_histories():
    returns = pd.DataFrame(
        np.random.default_rng(0).normal(0.0005, 0.01, (6000, 4)), columns=list("ABCD")
    )

    weights = optimization_model.get_minimum_cvar_weights(returns)

    assert weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="No period"):
        optimization_model.get_minimum_cvar_weights(returns.iloc[:0])


def test_black_litterman_reads_hyphenated_tickers_and_singular_covariances():
    covariance = pd.DataFrame(
        [[0.04, 0.01, 0.012], [0.01, 0.09, 0.02], [0.012, 0.02, 0.0625]],
        index=["A", "BRK-B", "C"],
        columns=["A", "BRK-B", "C"],
    )
    weights = pd.Series({"A": 0.3, "BRK-B": 0.3, "C": 0.4})

    relative = optimization_model.get_black_litterman_returns(
        covariance, weights, views={"BRK-B-C": 0.02}
    )[0]["Black-Litterman Return"]
    implied = optimization_model.get_black_litterman_returns(covariance, weights)[0][
        "Implied Return"
    ]

    # The view that BRK-B beats C pulls their gap towards 2%.
    assert (relative["BRK-B"] - relative["C"]) > (implied["BRK-B"] - implied["C"])

    duplicate = pd.DataFrame(0.04, index=["X", "Y"], columns=["X", "Y"])
    posterior = optimization_model.get_black_litterman_returns(
        duplicate, pd.Series({"X": 0.5, "Y": 0.5}), views={"X": 0.1}
    )[0]["Black-Litterman Return"]

    assert np.isfinite(posterior).all()
    with pytest.raises(ValueError, match="view confidences"):
        optimization_model.get_black_litterman_returns(
            covariance, weights, views={"A": 0.1}, view_confidences={"C": 0.5}
        )


def test_optimizers_explain_a_covariance_they_cannot_use():
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.0]], index=["A", "CASH"], columns=["A", "CASH"]
    )

    with pytest.raises(ValueError, match="CASH has no variance"):
        optimization_model.get_risk_parity_weights(covariance)

    missing = pd.DataFrame(
        [[0.04, np.nan], [np.nan, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="missing values"):
        optimization_model.get_inverse_volatility_weights(missing)

    single = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    with pytest.raises(ValueError, match="two assets"):
        optimization_model.get_hierarchical_risk_parity_weights(single)

    statistics = optimization_model.get_portfolio_statistics(
        pd.Series({"A": 1.0}),
        pd.Series({"A": 0.05}),
        pd.DataFrame([[1e-34]], index=["A"], columns=["A"]),
    )
    assert np.isnan(statistics["Sharpe Ratio"])


# Eight periods of three assets; the expected values below are scikit-learn's LedoitWolf and
# OAS and PyPortfolioOpt's semicovariance and exp_cov on the same returns.
SHRINKAGE_RETURNS = pd.DataFrame(
    {
        "A": [0.01, -0.02, 0.015, 0.003, -0.01, 0.02, -0.005, 0.007],
        "B": [0.004, -0.01, 0.02, -0.006, 0.001, 0.012, -0.015, 0.009],
        "C": [-0.003, 0.006, 0.001, 0.01, -0.012, 0.004, 0.002, -0.008],
    }
)


def test_ledoit_wolf_matches_scikit_learn():
    covariance, shrinkage = optimization_model.get_ledoit_wolf_covariance(
        SHRINKAGE_RETURNS
    )

    assert shrinkage == pytest.approx(0.3210581082483753, rel=1e-12)
    assert covariance.loc["A", "B"] == pytest.approx(7.234974533978252e-05, rel=1e-12)
    assert covariance.loc["C", "C"] == pytest.approx(6.66137982702211e-05, rel=1e-12)


def test_oracle_approximating_shrinkage_matches_scikit_learn():
    covariance, shrinkage = (
        optimization_model.get_oracle_approximating_shrinkage_covariance(
            SHRINKAGE_RETURNS
        )
    )

    assert shrinkage == pytest.approx(0.6338126146782789, rel=1e-12)
    assert covariance.loc["A", "B"] == pytest.approx(3.902184324834591e-05, rel=1e-12)
    assert covariance.loc["C", "C"] == pytest.approx(8.59638544258504e-05, rel=1e-12)


def test_semicovariance_and_exponential_covariance_match_pyportfolioopt():
    semicovariance = optimization_model.get_semicovariance(SHRINKAGE_RETURNS)
    exponential = optimization_model.get_exponential_covariance(SHRINKAGE_RETURNS, 4)

    assert semicovariance.loc["A", "B"] == pytest.approx(3.4375e-05, rel=1e-12)
    assert semicovariance.loc["C", "C"] == pytest.approx(2.7125e-05, rel=1e-12)
    assert exponential.loc["A", "B"] == pytest.approx(8.306283588151976e-05, rel=1e-12)
    assert exponential.loc["C", "C"] == pytest.approx(4.810124354274288e-05, rel=1e-12)


def test_the_exponential_covariance_weights_deviations_from_the_sample_mean():
    deviations = SHRINKAGE_RETURNS - SHRINKAGE_RETURNS.mean()
    exponential = optimization_model.get_exponential_covariance(SHRINKAGE_RETURNS)

    # Normalized weights (1 - 2 / (span + 1)) ** age on the deviations from the sample mean.
    expected = (
        (deviations["A"] * deviations["B"]).ewm(span=180, adjust=True).mean().iloc[-1]
    )
    assert exponential.loc["A", "B"] == pytest.approx(expected, rel=1e-12)


def test_shrinkage_keeps_the_average_variance_and_pulls_correlations_in():
    sample = SHRINKAGE_RETURNS.cov(ddof=0)
    covariance, _ = optimization_model.get_ledoit_wolf_covariance(SHRINKAGE_RETURNS)

    # The target has the sample's average variance, so the trace is unchanged.
    assert np.trace(covariance) == pytest.approx(np.trace(sample))
    assert abs(covariance.loc["A", "B"]) < abs(sample.loc["A", "B"])


@pytest.mark.parametrize(
    "covariance",
    ["sample", "ledoit_wolf", "oas", "semicovariance", "exponential", "denoised"],
)
def test_every_covariance_estimator_is_annualized(covariance):
    _, annual = optimization_model.get_expected_returns_and_covariance(
        SHRINKAGE_RETURNS, 252, covariance
    )
    _, per_period = optimization_model.get_expected_returns_and_covariance(
        SHRINKAGE_RETURNS, 1, covariance
    )

    pd.testing.assert_frame_equal(annual, per_period * 252)


def test_an_unknown_covariance_estimator_is_refused():
    with pytest.raises(ValueError, match="ledoit_wolf"):
        optimization_model.get_expected_returns_and_covariance(
            SHRINKAGE_RETURNS, 252, "graphical_lasso"
        )


def _reference_returns() -> tuple[pd.DataFrame, pd.Series]:
    """
    Five assets with two common factors and set average returns, and a benchmark close to
    their equally weighted portfolio. The reference values below come from skfolio 1.12.7,
    PyPortfolioOpt 1.6.0 and Riskfolio-Lib 7.4.1 run on these same returns.
    """
    rng = np.random.default_rng(42)
    factors = rng.normal(0.0004, 0.01, size=(500, 2))
    loadings = rng.uniform(0.2, 1.2, size=(2, 5))
    data = factors @ loadings + rng.normal(0.0002, 0.012, size=(500, 5))
    data = data - data.mean(axis=0) + np.array([0.10, 0.12, 0.14, 0.11, 0.13]) / 252
    returns = pd.DataFrame(data, columns=list("ABCDE"))
    benchmark = returns.mean(axis=1) + rng.normal(0, 0.004, size=500)
    return returns, benchmark


def test_denoised_covariance_matches_skfolio():
    returns, _ = _reference_returns()
    covariance, signals = optimization_model.get_denoised_covariance(returns)

    # skfolio's DenoiseCovariance().fit(returns).covariance_
    assert signals == 1
    np.testing.assert_allclose(
        covariance.iloc[0],
        [
            2.0119956392e-04,
            7.6140760647e-05,
            8.1351640371e-05,
            8.1579274082e-05,
            3.9645931816e-05,
        ],
        rtol=1e-8,
    )
    np.testing.assert_allclose(np.diag(covariance), returns.var(), rtol=1e-12)


def test_denoised_covariance_keeps_the_correlations_of_a_strong_signal():
    rng = np.random.default_rng(0)
    common = rng.normal(size=(1000, 1))
    returns = pd.DataFrame(
        common + 0.1 * rng.normal(size=(1000, 2)), columns=["A", "B"]
    )
    covariance, signals = optimization_model.get_denoised_covariance(returns)

    # With two assets the one eigenvalue below the edge is averaged with itself.
    assert signals == 1
    np.testing.assert_allclose(covariance, returns.cov(), rtol=1e-10)


def test_maximum_diversification_matches_skfolio():
    returns, _ = _reference_returns()
    weights = optimization_model.get_maximum_diversification_weights(returns.cov())

    # skfolio's MaximumDiversification().fit(returns).weights_; its interior point solver
    # stops a little earlier, at a diversification ratio of 1.510483907 against 1.510483920.
    np.testing.assert_allclose(
        weights,
        [0.2068061426, 0.1638346123, 0.1424396347, 0.1304583187, 0.3564612917],
        atol=1e-4,
    )
    assert optimization_model.get_diversification_ratio(
        weights, returns.cov()
    ) == pytest.approx(1.510483920, abs=1e-8)


def test_diversification_ratio_is_one_for_a_single_asset_and_root_n_for_independent_ones():
    single = optimization_model.get_diversification_ratio(
        pd.Series({"A": 1.0}), COVARIANCE
    )
    independent = pd.DataFrame(
        np.diag([0.04, 0.09, 0.01]), index=list("XYZ"), columns=list("XYZ")
    )
    weights = optimization_model.get_maximum_diversification_weights(independent)

    assert single == pytest.approx(1)
    # Uncorrelated assets: the most diversified portfolio is inverse volatility weighted
    # and its ratio is the square root of the number of assets.
    np.testing.assert_allclose(
        weights, np.array([5, 10 / 3, 10]) / (5 + 10 / 3 + 10), atol=1e-6
    )
    assert optimization_model.get_diversification_ratio(
        weights, independent
    ) == pytest.approx(np.sqrt(3))


def test_minimum_tracking_error_matches_skfolio_and_pyportfolioopt():
    returns, benchmark = _reference_returns()
    joint = returns.assign(Benchmark=benchmark)
    weights = optimization_model.get_minimum_tracking_error_weights(
        joint.mean() * 252, joint.cov() * 252, "Benchmark"
    )

    # skfolio's BenchmarkTracker().fit(returns, benchmark).weights_; PyPortfolioOpt's
    # ex_post_tracking_error objective agrees with it to 2e-6.
    np.testing.assert_allclose(
        weights,
        [0.2060920221, 0.1957605698, 0.2037438147, 0.2022345485, 0.1921690448],
        atol=1e-5,
    )
    statistics = optimization_model.get_tracking_error(
        weights, joint.mean() * 252, joint.cov() * 252, "Benchmark"
    )
    active = returns @ weights - benchmark
    assert statistics["Tracking Error"] == pytest.approx(active.std() * np.sqrt(252))
    assert statistics["Active Return"] == pytest.approx(active.mean() * 252)


def test_maximum_information_ratio_matches_skfolio():
    returns, benchmark = _reference_returns()
    joint = returns.assign(Benchmark=benchmark)
    weights = optimization_model.get_minimum_tracking_error_weights(
        joint.mean() * 252,
        joint.cov() * 252,
        "Benchmark",
        objective="information_ratio",
    )

    # skfolio's MeanRisk(objective_function=MAXIMIZE_RATIO, risk_measure=STANDARD_DEVIATION)
    # on the active returns R - r_b, whose weights sum to one like the portfolio's.
    np.testing.assert_allclose(
        weights,
        [0.0725260955, 0.1878748205, 0.3300022901, 0.1381993728, 0.271397421],
        atol=1e-5,
    )


def test_tracking_error_needs_the_benchmark_and_a_valid_objective():
    returns, benchmark = _reference_returns()
    joint = returns.assign(Benchmark=benchmark)
    with pytest.raises(ValueError, match="benchmark"):
        optimization_model.get_minimum_tracking_error_weights(
            returns.mean(), returns.cov(), "Benchmark"
        )
    with pytest.raises(ValueError, match="objective"):
        optimization_model.get_minimum_tracking_error_weights(
            joint.mean(), joint.cov(), "Benchmark", objective="sharpe"
        )


def test_growth_optimal_weights_match_riskfolio():
    returns, _ = _reference_returns()
    exact = optimization_model.get_growth_optimal_weights(
        returns, returns.mean(), returns.cov()
    )
    approximate = optimization_model.get_growth_optimal_weights(
        returns, returns.mean() * 252, returns.cov() * 252, method="approximate"
    )

    # Riskfolio-Lib's Portfolio.optimization(obj="MaxRet", kelly="exact" / "approx").
    np.testing.assert_allclose(exact, [0, 0, 0.4255907897, 0, 0.5744092095], atol=1e-7)
    np.testing.assert_allclose(
        approximate, [0, 0, 0.4254029847, 0, 0.5745969921], atol=1e-4
    )

    # The approximation is a mean-variance portfolio with a risk aversion of one, here the
    # closed form of its two-asset interior solution.
    pair = returns[["C", "E"]]
    system = np.block(
        [[pair.cov().to_numpy(), np.ones((2, 1))], [np.ones((1, 2)), np.zeros((1, 1))]]
    )
    closed_form = np.linalg.solve(system, np.r_[pair.mean().to_numpy(), 1])[:2]
    np.testing.assert_allclose(approximate[["C", "E"]], closed_form, atol=1e-8)


def test_fractional_kelly_scales_the_weights():
    returns, _ = _reference_returns()
    full = optimization_model.get_growth_optimal_weights(
        returns, returns.mean(), returns.cov()
    )
    half = optimization_model.get_growth_optimal_weights(
        returns, returns.mean(), returns.cov(), fraction=0.5
    )

    np.testing.assert_allclose(half, full / 2)
    with pytest.raises(ValueError, match="fraction"):
        optimization_model.get_growth_optimal_weights(
            returns, returns.mean(), returns.cov(), fraction=0
        )


SECTORS = pd.DataFrame(
    {"Sector": {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Health", "E": "Health"}}
)


def _long_short_returns() -> pd.DataFrame:
    """
    The reference returns with a sixth asset F, a noisy 1.5 times A, which the minimum
    variance and minimum CVaR portfolios short heavily (to -0.62 and -0.60) to hedge A
    when nothing limits the gross exposure.
    """
    returns, _ = _reference_returns()
    returns["F"] = 1.5 * returns["A"] + np.random.default_rng(7).normal(0, 0.004, 500)
    return returns


def test_per_asset_and_sector_bounds_match_pyportfolioopt_and_skfolio():
    returns, _ = _reference_returns()
    annual_returns, annual_covariance = returns.mean() * 252, returns.cov() * 252

    minimum_variance = optimization_model.get_minimum_variance_weights(
        returns.cov(),
        minimum_weight={"E": 0.25},
        maximum_weight={"A": 0.1},
        groups=SECTORS,
        group_bounds={"Sector": {"Tech": (0, 0.25), "Health": (0.3, 0.5)}},
    )
    maximum_sharpe = optimization_model.get_maximum_sharpe_weights(
        annual_returns,
        annual_covariance,
        maximum_weight=0.35,
        groups=SECTORS,
        group_bounds={"Sector": {"Tech": (0, 0.3)}},
    )
    target = optimization_model.get_mean_variance_weights(
        annual_returns,
        annual_covariance,
        target_return=0.13,
        groups=SECTORS,
        group_bounds={"Sector": {"Energy": (0, 0.2)}},
    )

    # PyPortfolioOpt 1.6.0: EfficientFrontier(weight_bounds=...) with
    # add_sector_constraints and min_volatility / max_sharpe / efficient_return(0.13);
    # skfolio 1.12.7's MeanRisk with groups and linear constraints agrees to 1e-5.
    np.testing.assert_allclose(
        minimum_variance,
        [0.1, 0.15, 0.25, 0.0448636906, 0.4551363094],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        maximum_sharpe,
        [0.1674405078, 0.1325594922, 0.2564393052, 0.0935606948, 0.35],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        target, [0.0076735036, 0.1560877896, 0.2, 0.0104458498, 0.625792857], atol=1e-6
    )


def test_sector_bounds_match_skfolio_for_diversification_cvar_and_tracking_error():
    returns, benchmark = _reference_returns()
    with_benchmark = returns.assign(BM=benchmark)

    diversification = optimization_model.get_maximum_diversification_weights(
        returns.cov(),
        maximum_weight=0.3,
        groups=SECTORS,
        group_bounds={"Sector": {"Tech": (0, 0.3)}},
    )
    cvar = optimization_model.get_minimum_cvar_weights(
        returns,
        minimum_weight=-0.2,
        groups=SECTORS,
        group_bounds={"Sector": {"Tech": (0, 0.5)}},
        gross_exposure=1.4,
    )
    tracking = optimization_model.get_minimum_tracking_error_weights(
        with_benchmark.mean() * 252,
        with_benchmark.cov() * 252,
        "BM",
        minimum_weight={"E": 0.25},
        maximum_weight=0.3,
        groups=SECTORS,
        group_bounds={"Sector": {"Tech": (0, 0.3)}},
    )

    # skfolio 1.12.7's MaximumDiversification and MeanRisk(risk_measure=CVAR) with
    # groups and linear constraints; its interior point solver stops a little earlier for
    # the diversification ratio (1.4914861269 against 1.4914861812 here). The tracking
    # error is the same problem solved with cvxpy.
    np.testing.assert_allclose(
        diversification,
        [0.1749702229, 0.1250296936, 0.2048636223, 0.1951365259, 0.3],
        atol=1e-4,
    )
    assert optimization_model.get_diversification_ratio(
        diversification, returns.cov()
    ) == pytest.approx(1.4914861812, abs=1e-9)
    np.testing.assert_allclose(
        cvar,
        [0.1827515508, 0.1679373687, 0.1443273254, 0.0412642027, 0.4637195524],
        atol=1e-7,
    )
    # The tracking error is nearly flat between the two Tech assets (a shift of 4e-7
    # between A and B changes the tracking variance by about 1e-14), so SLSQP lands within
    # 1e-6 of the cvxpy weights rather than 1e-7.
    np.testing.assert_allclose(
        tracking,
        [0.1497636622, 0.1502363378, 0.2233589641, 0.2266410359, 0.25],
        atol=1e-6,
    )


def test_gross_exposure_limits_long_short_portfolios_as_skfolio_and_cvxpy_do():
    returns = _long_short_returns()
    annual_returns, annual_covariance = returns.mean() * 252, returns.cov() * 252

    unlimited = optimization_model.get_minimum_variance_weights(
        annual_covariance, minimum_weight=-1
    )
    minimum_variance = optimization_model.get_minimum_variance_weights(
        annual_covariance, minimum_weight=-1, gross_exposure=1.3
    )
    cvar = optimization_model.get_minimum_cvar_weights(
        returns, minimum_weight=-1, gross_exposure=1.3
    )
    sharpe = optimization_model.get_maximum_sharpe_weights(
        annual_returns, annual_covariance, minimum_weight=-0.5, gross_exposure=1.3
    )
    neutral = optimization_model.get_mean_variance_weights(
        annual_returns,
        annual_covariance,
        minimum_weight=-1,
        net_exposure=0,
        gross_exposure=1,
    )

    # skfolio 1.12.7's MeanRisk with max_short=0.15 (a gross exposure of 1.3 at a net
    # exposure of 1) for the variance and CVaR; cvxpy with norm1(w) <= 1.3 (and the
    # Charnes-Cooper form of the Sharpe ratio) for the others.
    assert unlimited.abs().sum() == pytest.approx(2.2343162, abs=1e-6)
    np.testing.assert_allclose(
        minimum_variance,
        [0.439570321, 0.1226283763, 0.0838146256, 0.078221562, 0.4257651151, -0.15],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        cvar,
        [0.3975681407, 0.1223904634, 0.1081521135, 0.0645047478, 0.4573845342, -0.15],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        sharpe,
        [0.3619640418, 0.118044775, 0.1571807769, 0.0410120287, 0.4717983776, -0.15],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        neutral, [0.411418575, 0, 0.088581425, 0, 0, -0.5], atol=1e-7
    )
    for weights, net in ((minimum_variance, 1), (cvar, 1), (sharpe, 1), (neutral, 0)):
        assert weights.sum() == pytest.approx(net, abs=1e-8)
        assert weights.abs().sum() <= (1 if net == 0 else 1.3) + 1e-8


def test_the_efficient_frontier_keeps_the_constraints_at_every_point():
    returns = _long_short_returns()
    frontier = optimization_model.get_efficient_frontier(
        returns.mean() * 252,
        returns.cov() * 252,
        points=5,
        minimum_weight=-0.3,
        gross_exposure=1.4,
    )
    sectors = optimization_model.get_efficient_frontier(
        returns.mean() * 252,
        returns.cov() * 252,
        points=5,
        maximum_weight=0.4,
        groups=SECTORS,
        group_bounds={"Sector": {"Health": (0.3, 0.5)}},
    )
    weights = frontier[list("ABCDEF")]

    assert np.allclose(weights.sum(axis=1), 1)
    assert (weights.abs().sum(axis=1) <= 1.4 + 1e-8).all()
    assert weights.abs().sum(axis=1).iloc[-1] == pytest.approx(1.4)
    assert frontier["Expected Return"].is_monotonic_increasing
    health = sectors[["D", "E"]].sum(axis=1)
    assert health.between(0.3 - 1e-8, 0.5 + 1e-8).all()


def test_risk_parity_bounds_match_skfolio_risk_budgeting():
    returns, _ = _reference_returns()
    covariance = returns.cov()
    capped = optimization_model.get_risk_parity_weights(
        covariance,
        maximum_weight=0.22,
        groups=SECTORS,
        group_bounds={"Sector": {"Health": (0, 0.4)}},
    )
    floored = optimization_model.get_risk_parity_weights(
        covariance, minimum_weight={"D": 0.25}
    )
    loose = optimization_model.get_risk_parity_weights(covariance, maximum_weight=0.9)

    # skfolio 1.12.7's RiskBudgeting(risk_measure=VARIANCE) with the same bounds and
    # groups. Its interior point solver stops earlier: with the floor on D, its weights
    # give -b'ln(w) + ln(w'Σw) / 2, the objective at the optimal scale, of -2.9605365172
    # against -2.9605365614 here.
    np.testing.assert_allclose(
        capped,
        [0.2199997309, 0.1964004269, 0.1836015084, 0.179998356, 0.2199999778],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        floored,
        [0.1862257787, 0.1622669699, 0.1517504428, 0.2500000607, 0.2497567478],
        atol=5e-5,
    )
    objective = -0.2 * np.log(floored).sum() + 0.5 * np.log(
        floored @ covariance @ floored
    )
    assert objective == pytest.approx(-2.9605365614, abs=1e-9)
    # Bounds that don't bind leave the equal risk contributions.
    np.testing.assert_allclose(
        loose, optimization_model.get_risk_parity_weights(covariance), atol=1e-6
    )
    with pytest.raises(ValueError, match="long positions only"):
        optimization_model.get_risk_parity_weights(covariance, minimum_weight=-0.1)


def test_hierarchical_risk_parity_bounds_move_the_splits_as_skfolio_does():
    returns, _ = _reference_returns()
    covariance = returns.cov()
    bounded = optimization_model.get_hierarchical_risk_parity_weights(
        covariance, minimum_weight=0.15, maximum_weight=0.3
    )
    unbounded = optimization_model.get_hierarchical_risk_parity_weights(covariance)
    capped = optimization_model.get_hierarchical_risk_parity_weights(
        covariance, maximum_weight={"E": 0.2}
    )

    # skfolio 1.12.7's HierarchicalRiskParity(min_weights=0.15, max_weights=0.3) with
    # single linkage. It clusters on the correlation distance itself, which orders B, C
    # and D differently, so with E capped only A (which takes E's excess in both) and E
    # are compared.
    np.testing.assert_allclose(bounded, [0.25, 0.15, 0.15, 0.15, 0.3], atol=1e-10)
    assert capped["E"] == pytest.approx(0.2)
    assert capped["A"] == pytest.approx(0.4207855551, abs=1e-9)
    np.testing.assert_allclose(capped[["B", "C", "D"]], unbounded[["B", "C", "D"]])
    with pytest.raises(ValueError, match="long positions only"):
        optimization_model.get_hierarchical_risk_parity_weights(
            covariance, minimum_weight=-0.1
        )


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        ({"minimum_weight": 0.3}, "cannot sum to one"),
        (
            {"minimum_weight": {"A": 0.5}, "maximum_weight": {"A": 0.4}},
            "above the maximum",
        ),
        ({"minimum_weight": {"Z": 0.1}}, "not in the portfolio: Z"),
        (
            {
                "maximum_weight": 0.3,
                "groups": SECTORS,
                "group_bounds": {"Sector": {"Tech": (0.7, 1)}},
            },
            "No weights meet all the constraints together",
        ),
        (
            {"groups": SECTORS, "group_bounds": {"Sector": {"Retail": (0, 1)}}},
            "No asset has the Sector 'Retail'; the assets have Energy, Health, Tech",
        ),
        (
            {"groups": SECTORS, "group_bounds": {"Country": {"US": (0, 1)}}},
            "no Country",
        ),
        (
            {"groups": SECTORS, "group_bounds": {"Sector": {"Tech": (0.5, 0.4)}}},
            "above its maximum",
        ),
        (
            {"groups": SECTORS, "group_bounds": {"Sector": {"Tech": 0.4}}},
            "pair",
        ),
        ({"minimum_weight": -1, "gross_exposure": 0.5}, "cannot hold a net exposure"),
    ],
)
def test_constraints_no_portfolio_can_meet_are_refused_clearly(constraints, message):
    returns, _ = _reference_returns()
    with pytest.raises(ValueError, match=message):
        optimization_model.get_minimum_variance_weights(returns.cov(), **constraints)


def test_a_gross_exposure_adds_absolute_weights_only_with_short_positions():
    assets = pd.Index(list("ABC"))
    long_only = optimization_model.get_weight_constraints(assets, gross_exposure=1.5)
    long_short = optimization_model.get_weight_constraints(
        assets, minimum_weight=-0.5, gross_exposure=1.5
    )

    assert len(long_only.bounds) == 3  # noqa: PLR2004
    assert len(long_short.bounds) == 6  # noqa: PLR2004
    # t_i >= |w_i| for each asset and their sum at most the gross exposure.
    assert long_short.inequality.shape == (7, 6)
    assert long_short.upper[-1] == 1.5  # noqa: PLR2004


def test_expected_returns_match_pyportfolioopt_and_riskfolio():
    returns, benchmark = _reference_returns()

    def estimate(method, **kwargs):
        return optimization_model.get_expected_returns(
            returns, 252, method, **kwargs
        ).to_numpy()

    assert np.allclose(estimate("mean"), returns.mean() * 252)
    # PyPortfolioOpt ema_historical_return(span=60, compounding=False).
    assert np.allclose(
        estimate("exponential", span=60),
        [0.4248258365, 0.3784078348, 0.9053789584, 0.4968557113, 0.672913352],
    )
    # PyPortfolioOpt capm_return(risk_free_rate=0.02, compounding=False).
    assert np.allclose(
        estimate("capm", benchmark_returns=benchmark, risk_free_rate=0.02),
        [0.0863760324, 0.1009805469, 0.1085805934, 0.1087109621, 0.060395485],
    )
    # Jorion (1986) with Σ = S (T - 1) / (T - N - 2), written out. Riskfolio-Lib's
    # mean_vector(method="BS", target="b2") uses S itself and gives
    # [0.1208729336, 0.1210211013, 0.121169269, 0.1209470175, 0.1210951851].
    periods, assets = returns.shape
    average = returns.mean().to_numpy()
    inverse = np.linalg.inv(
        returns.cov().to_numpy() * (periods - 1) / (periods - assets - 2)
    )
    ones = np.ones(assets)
    target = ones @ inverse @ average / (ones @ inverse @ ones)
    deviation = average - target
    phi = (assets + 2) / (assets + 2 + periods * deviation @ inverse @ deviation)
    assert np.allclose(
        estimate("bayes_stein"), ((1 - phi) * average + phi * target) * 252
    )
    with pytest.raises(ValueError, match="more periods"):
        optimization_model.get_expected_returns(returns.iloc[:7], 252, "bayes_stein")


def test_bayes_stein_leaves_equal_averages_and_shrinks_dispersed_ones():
    returns, _ = _reference_returns()
    equal = returns - returns.mean() + 0.0004
    assert np.allclose(
        optimization_model.get_expected_returns(equal, 252, "bayes_stein"), 0.0004 * 252
    )

    shrunk = optimization_model.get_expected_returns(returns, 252, "bayes_stein")
    assert shrunk.std() < (returns.mean() * 252).std()


def test_capm_returns_need_a_benchmark_and_unknown_estimators_are_refused():
    returns, _ = _reference_returns()
    with pytest.raises(ValueError, match="benchmark"):
        optimization_model.get_expected_returns(returns, 252, "capm")
    with pytest.raises(ValueError, match="must be one of"):
        optimization_model.get_expected_returns(returns, 252, "median")


CURRENT = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})


def test_a_maximum_turnover_limits_the_trades_from_the_current_weights():
    returns, _ = _reference_returns()
    covariance = returns.cov() * 252
    free = optimization_model.get_minimum_variance_weights(covariance)
    capped = optimization_model.get_minimum_variance_weights(
        covariance, current_weights=CURRENT, maximum_turnover=0.2
    )
    current = CURRENT.reindex(covariance.index).fillna(0)

    assert np.abs(free - current).sum() > 0.2  # noqa: PLR2004
    assert np.abs(capped - current).sum() == pytest.approx(0.2, abs=1e-7)
    assert capped @ covariance @ capped < current @ covariance @ current
    # No turnover keeps the current weights, a wide limit gives the free optimum.
    assert np.allclose(
        optimization_model.get_minimum_variance_weights(
            covariance, current_weights=CURRENT, maximum_turnover=0
        ),
        current,
        atol=1e-7,
    )
    assert np.allclose(
        optimization_model.get_minimum_variance_weights(
            covariance, current_weights=CURRENT, maximum_turnover=2
        ),
        free,
        atol=1e-5,
    )


def test_a_turnover_limit_works_next_to_a_gross_exposure_limit():
    returns, _ = _reference_returns()
    weights = optimization_model.get_minimum_cvar_weights(
        returns,
        minimum_weight=-0.5,
        gross_exposure=1.4,
        current_weights=CURRENT,
        maximum_turnover=0.5,
    )
    current = CURRENT.reindex(returns.columns).fillna(0)

    assert weights.sum() == pytest.approx(1)
    assert np.abs(weights).sum() <= 1.4 + 1e-7  # noqa: PLR2004
    assert np.abs(weights - current).sum() <= 0.5 + 1e-7  # noqa: PLR2004


def test_transaction_costs_match_pyportfolioopt_and_skfolio():
    returns, _ = _reference_returns()
    expected_returns, covariance = (
        optimization_model.get_expected_returns_and_covariance(returns, 252)
    )
    # PyPortfolioOpt max_quadratic_utility(risk_aversion=2.5) with transaction_cost(k=0.01).
    assert np.allclose(
        optimization_model.get_mean_variance_weights(
            expected_returns,
            covariance,
            current_weights=CURRENT,
            transaction_cost=0.01,
        ),
        [0.104088, 0.194714, 0.2, 0.049088, 0.45211],
        atol=1e-4,
    )
    # skfolio MeanRisk(MAXIMIZE_RATIO, transaction_costs=0.01 / 252, risk_free_rate=0.02 / 252)
    # on the daily returns, the same problem per day.
    assert np.allclose(
        optimization_model.get_maximum_sharpe_weights(
            expected_returns,
            covariance,
            0.02,
            current_weights=CURRENT,
            transaction_cost=0.01,
        ),
        [0.148652, 0.156373, 0.2, 0.047757, 0.447217],
        atol=1e-4,
    )


def test_l2_regularization_matches_pyportfolioopt():
    returns, _ = _reference_returns()
    annual_returns, annual_covariance = returns.mean() * 252, returns.cov() * 252
    gamma = 0.05

    # PyPortfolioOpt 1.6.0 (cvxpy 1.9.3): EfficientFrontier(annual_returns, annual_covariance)
    # with add_objective(objective_functions.L2_reg, gamma=0.05) and min_volatility,
    # max_sharpe(risk_free_rate=0.02), max_quadratic_utility(risk_aversion=2.5),
    # efficient_return(0.125), efficient_risk(0.17) and, with weight_bounds=(0, 0.3),
    # min_volatility.
    references = {
        "variance": [0.22216723, 0.16755499, 0.14540449, 0.14219593, 0.32267736],
        "sharpe": [0.1438285, 0.16539811, 0.22202917, 0.10335151, 0.3653927],
        "utility": [0.12147089, 0.16059729, 0.23952789, 0.08493295, 0.39347098],
        "return": [0.12750838, 0.16494878, 0.23799221, 0.09525915, 0.37429149],
        "volatility": [0.02205758, 0.19235481, 0.37081239, 0.09629266, 0.31848255],
        "bounded": [0.22955601, 0.17309792, 0.15039289, 0.14695319, 0.3],
    }
    weights = {
        "variance": optimization_model.get_minimum_variance_weights(
            annual_covariance, l2_regularization=gamma
        ),
        "sharpe": optimization_model.get_maximum_sharpe_weights(
            annual_returns, annual_covariance, 0.02, l2_regularization=gamma
        ),
        "utility": optimization_model.get_mean_variance_weights(
            annual_returns, annual_covariance, l2_regularization=gamma
        ),
        "return": optimization_model.get_mean_variance_weights(
            annual_returns, annual_covariance, 0.125, l2_regularization=gamma
        ),
        "volatility": optimization_model.get_mean_variance_weights(
            annual_returns,
            annual_covariance,
            target_volatility=0.17,
            l2_regularization=gamma,
        ),
        "bounded": optimization_model.get_minimum_variance_weights(
            annual_covariance, 0, 0.3, l2_regularization=gamma
        ),
    }
    for name, reference in references.items():
        np.testing.assert_allclose(weights[name], reference, atol=1e-5, err_msg=name)

    # The first point of the frontier is the regularized minimum variance portfolio,
    # with the volatility of the covariance itself.
    frontier = optimization_model.get_efficient_frontier(
        annual_returns, annual_covariance, points=3, l2_regularization=gamma
    )
    np.testing.assert_allclose(
        frontier.loc[1, list("ABCDE")], references["variance"], atol=1e-6
    )
    statistics = optimization_model.get_portfolio_statistics(
        frontier.loc[1, list("ABCDE")], annual_returns, annual_covariance
    )
    assert frontier.loc[1, "Volatility"] == pytest.approx(statistics["Volatility"])


def test_l2_regularization_moves_towards_equal_weights():
    returns, _ = _reference_returns()
    covariance = returns.cov() * 252
    equal = np.full(5, 0.2)
    distances = [
        np.abs(
            optimization_model.get_minimum_variance_weights(
                covariance, l2_regularization=gamma
            )
            - equal
        ).sum()
        for gamma in (0.0, 0.01, 0.1, 100.0)
    ]

    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < 1e-3
    # The penalty is in the units of the covariance: a per-period covariance with γ / 252
    # gives the same portfolio as the annualized one with γ.
    np.testing.assert_allclose(
        optimization_model.get_minimum_variance_weights(
            returns.cov(), l2_regularization=0.05 / 252
        ),
        optimization_model.get_minimum_variance_weights(
            covariance, l2_regularization=0.05
        ),
        atol=1e-8,
    )
    with pytest.raises(ValueError, match="L2 regularization cannot be negative"):
        optimization_model.get_minimum_variance_weights(
            covariance, l2_regularization=-0.1
        )


def test_a_high_transaction_cost_keeps_the_current_weights():
    expected_returns, covariance = EXPECTED_RETURNS, COVARIANCE
    weights = optimization_model.get_mean_variance_weights(
        expected_returns, covariance, current_weights=CURRENT, transaction_cost=10
    )
    assert np.allclose(weights, CURRENT.reindex(ASSETS), atol=1e-7)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"maximum_turnover": 0.2}, "needs the current weights"),
        ({"transaction_cost": 0.01}, "needs the current weights"),
        (
            {"current_weights": CURRENT, "transaction_cost": -0.01},
            "cannot be negative",
        ),
        ({"current_weights": CURRENT, "maximum_turnover": -1}, "cannot be negative"),
        ({"current_weights": pd.Series({"Z": 1.0})}, "not in the portfolio"),
    ],
)
def test_turnover_and_cost_arguments_are_checked(arguments, message):
    with pytest.raises(ValueError, match=message):
        optimization_model.get_maximum_sharpe_weights(
            EXPECTED_RETURNS, COVARIANCE, **arguments
        )


def test_a_transaction_cost_cannot_be_combined_with_a_target_return():
    with pytest.raises(ValueError, match="target return"):
        optimization_model.get_mean_variance_weights(
            EXPECTED_RETURNS,
            COVARIANCE,
            target_return=0.08,
            current_weights=CURRENT,
            transaction_cost=0.01,
        )


def test_expected_shortfall_contributions_match_riskfolio_and_add_up():
    returns, _ = _reference_returns()
    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=returns.columns)
    contributions = optimization_model.get_expected_shortfall_contributions(
        weights, returns, 0.05
    )
    portfolio = returns @ weights
    shortfall = -portfolio[portfolio <= portfolio.quantile(0.05)].mean()

    # Riskfolio-Lib Risk_Contribution(rm="CVaR", alpha=0.05); 25 tail periods, so its
    # Rockafellar-Uryasev CVaR equals the average of the tail.
    assert np.allclose(
        contributions["Risk Contribution"],
        [0.0069948866, 0.0060393695, 0.0041596864, 0.0036566954, 0.0008148678],
    )
    assert contributions["Risk Contribution"].sum() == pytest.approx(shortfall)
    assert contributions["Percentage Contribution"].sum() == pytest.approx(1)


def test_an_asset_that_gains_in_the_portfolios_worst_periods_contributes_negatively():
    returns, _ = _reference_returns()
    returns["H"] = -returns["A"]
    weights = pd.Series([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], index=returns.columns)
    contributions = optimization_model.get_expected_shortfall_contributions(
        weights, returns
    )
    assert contributions.loc["H", "Risk Contribution"] < 0


def test_diversification_metrics():
    returns, _ = _reference_returns()
    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=returns.columns)
    metrics = optimization_model.get_diversification_metrics(weights, returns.cov())
    # skfolio Portfolio.diversification and effective_number_assets.
    assert metrics["Diversification Ratio"] == pytest.approx(1.4213985072385122)
    assert metrics["Effective Number of Assets"] == pytest.approx(4.444444444444445)
    assert metrics["Herfindahl-Hirschman Index"] == pytest.approx(0.225)

    # Uncorrelated assets with the same variance: N bets and a ratio of √N.
    independent = pd.DataFrame(np.eye(4) * 0.04, index=ASSETS, columns=ASSETS)
    equal = pd.Series(0.25, index=ASSETS)
    metrics = optimization_model.get_diversification_metrics(equal, independent)
    assert metrics["Effective Number of Bets"] == pytest.approx(4)
    assert metrics["Diversification Ratio"] == pytest.approx(2)

    # Perfectly correlated assets are one bet, however many there are.
    common = pd.DataFrame(np.full((4, 4), 0.04), index=ASSETS, columns=ASSETS)
    metrics = optimization_model.get_diversification_metrics(equal, common)
    assert metrics["Effective Number of Bets"] == pytest.approx(1)
    assert metrics["Effective Number of Assets"] == pytest.approx(4)


def test_capm_expected_returns_use_the_periods_the_benchmark_has():
    returns, benchmark = _reference_returns()
    gaps = benchmark.copy()
    gaps.iloc[::40] = np.nan

    estimate = optimization_model.get_expected_returns(
        returns, 252, "capm", benchmark_returns=gaps, risk_free_rate=0.02
    )
    shared = returns.loc[gaps.dropna().index]
    expected = optimization_model.get_expected_returns(
        shared, 252, "capm", benchmark_returns=gaps.dropna(), risk_free_rate=0.02
    )

    pd.testing.assert_series_equal(estimate, expected)


def test_effective_number_of_bets_stays_a_distribution_without_a_positive_covariance():
    covariance = pd.DataFrame(
        [[0.04, 0.05, 0.0], [0.05, 0.04, 0.0], [0.0, 0.0, 0.09]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    weights = pd.Series([0.4, 0.3, 0.3], index=list("ABC"))

    bets = optimization_model.get_diversification_metrics(weights, covariance)[
        "Effective Number of Bets"
    ]

    # The eigenvalue -0.01 is clipped and the shares of the others sum to one.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.to_numpy())
    shares = np.clip(eigenvalues, 0, None) * (eigenvectors.T @ weights.to_numpy()) ** 2
    shares = shares[shares > 0] / shares.sum()
    assert bets == pytest.approx(np.exp(-np.sum(shares * np.log(shares))))
    assert 1 <= bets <= 3  # noqa: PLR2004


def _block_returns() -> pd.DataFrame:
    """
    Ten assets in three blocks that share a factor, so that the clusterings have
    something to find. The reference values below come from skfolio 1.12.7 and
    Riskfolio-Lib 7.4.1 run on these same returns.
    """
    rng = np.random.default_rng(11)
    factors = rng.normal(0, 0.01, size=(750, 3))
    block = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    volatility = [0.008, 0.01, 0.012, 0.009, 0.015, 0.011, 0.013, 0.006, 0.007, 0.02]
    data = np.column_stack(
        [
            factors[:, factor] + rng.normal(0, noise, 750)
            for factor, noise in zip(block, volatility, strict=True)
        ]
    )
    data = data - data.mean(axis=0) + rng.uniform(0.02, 0.2, 10) / 252
    return pd.DataFrame(data, columns=[f"X{i}" for i in range(10)])


def test_hierarchical_clusters_match_skfolio():
    covariance = _block_returns().cov()

    # skfolio's HierarchicalClustering(linkage_method=WARD, max_clusters=k).labels_. On its
    # own, the two-order difference of the pooled dispersion finds the three blocks;
    # skfolio's sum of average distances picks six clusters ([0, 0, 0, 0, 1, 2, 3, 4, 4, 5]).
    _, chosen = optimization_model.get_hierarchical_clusters(covariance)
    _, three = optimization_model.get_hierarchical_clusters(covariance, clusters=3)

    assert three.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert chosen.tolist() == three.tolist()
    with pytest.raises(ValueError, match="linkage"):
        optimization_model.get_hierarchical_clusters(covariance, "centroid")
    with pytest.raises(ValueError, match="clusters"):
        optimization_model.get_hierarchical_clusters(covariance, clusters=11)


def test_hierarchical_equal_risk_contribution_matches_skfolio_and_riskfolio():
    covariance = _block_returns().cov() * 252

    # Six clusters is what skfolio chooses on its own; the pooled dispersion finds the
    # three blocks.
    chosen = optimization_model.get_hierarchical_equal_risk_contribution_weights(
        covariance, clusters=6
    )
    three = optimization_model.get_hierarchical_equal_risk_contribution_weights(
        covariance, clusters=3
    )
    capped = optimization_model.get_hierarchical_equal_risk_contribution_weights(
        covariance, clusters=6, maximum_weight=0.15
    )
    pd.testing.assert_series_equal(
        optimization_model.get_hierarchical_equal_risk_contribution_weights(covariance),
        three,
    )

    # skfolio's HierarchicalEqualRiskContribution(risk_measure=VARIANCE); with three
    # clusters Riskfolio-Lib's HCPortfolio(model="HERC", rm="MV", linkage="ward", k=3)
    # gives the same weights to 1e-16.
    np.testing.assert_allclose(
        chosen,
        [
            0.2518486089,
            0.22575452,
            0.1909375395,
            0.249568167,
            0.0044208511,
            0.0067831161,
            0.023761549,
            0.0200429267,
            0.0170281206,
            0.009854601,
        ],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        three,
        [
            0.1881299712,
            0.1686377841,
            0.1426296296,
            0.1864264896,
            0.0385146146,
            0.0590947521,
            0.0494522318,
            0.0782115486,
            0.0664471664,
            0.0224558121,
        ],
        atol=1e-10,
    )
    # skfolio moves weights outside the bounds to the closest weights in relative terms.
    np.testing.assert_allclose(
        capped,
        [
            0.15,
            0.15,
            0.15,
            0.15,
            0.0088102317,
            0.0171164659,
            0.1499999369,
            0.1102613985,
            0.0821473183,
            0.0316646433,
        ],
        atol=1e-6,
    )


def test_hierarchical_equal_risk_contribution_takes_group_bounds():
    covariance = _block_returns().cov() * 252
    groups = pd.DataFrame({"Sector": ["A"] * 4 + ["B"] * 6}, index=covariance.index)
    weights = optimization_model.get_hierarchical_equal_risk_contribution_weights(
        covariance, groups=groups, group_bounds={"Sector": {"A": (0, 0.5)}}
    )

    assert weights.sum() == pytest.approx(1)
    assert weights.iloc[:4].sum() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="long positions"):
        optimization_model.get_hierarchical_equal_risk_contribution_weights(
            covariance, minimum_weight=-0.1
        )


def test_nested_clustered_optimization_matches_riskfolio():
    returns = _block_returns()
    expected_returns, covariance = returns.mean() * 252, returns.cov() * 252

    # Riskfolio-Lib's HCPortfolio(model="NCO", linkage="ward", k=3) with obj="MinRisk",
    # "Sharpe" and "ERC" (rm="MV"); skfolio's NestedClustersOptimization with MeanRisk
    # inner and outer estimators (cv="ignore") agrees to 3e-5.
    for objective, reference in (
        (
            "minimum_variance",
            [
                0.103786947,
                0.0787445765,
                0.0590616196,
                0.1004225063,
                0.0667527316,
                0.1353947092,
                0.1058806144,
                0.2135579758,
                0.121171711,
                0.0152266085,
            ],
        ),
        (
            "maximum_sharpe",
            [0, 0.3034288499, 0, 0.0569662472, 0, 0.1339948997, 0.1326428203]
            + [0.3729671813, 0, 0],
        ),
        (
            "risk_parity",
            [
                0.08905631,
                0.0852055904,
                0.08090702,
                0.0883902433,
                0.0968755781,
                0.1172192353,
                0.1100548923,
                0.1300465961,
                0.1230500477,
                0.0791944868,
            ],
        ),
    ):
        weights = optimization_model.get_nested_clustered_optimization_weights(
            expected_returns, covariance, objective, clusters=3
        )
        np.testing.assert_allclose(weights, reference, atol=1e-5)

    with pytest.raises(ValueError, match="objective"):
        optimization_model.get_nested_clustered_optimization_weights(
            expected_returns, covariance, "maximum_return"
        )


def test_minimum_drawdown_weights_match_riskfolio_skfolio_and_pyportfolioopt():
    returns, _ = _reference_returns()

    # Riskfolio-Lib's Portfolio.optimization(obj="MinRisk", rm="CDaR" / "MDD" / "ADD");
    # skfolio's MeanRisk with the same risk measures and PyPortfolioOpt's
    # EfficientCDaR.min_cdar() agree to 2e-8.
    for measure, reference in (
        ("cdar", [0.1150271505, 0.0, 0.1997221787, 0.3666460814, 0.3186045893]),
        (
            "maximum",
            [0.0892092877, 0.0163364947, 0.2007031949, 0.3143927637, 0.379358259],
        ),
        ("average", [0.19197741, 0.0, 0.2161228913, 0.2844472116, 0.3074524871]),
    ):
        weights = optimization_model.get_minimum_drawdown_weights(returns, measure)
        np.testing.assert_allclose(weights, reference, atol=1e-6)

    with pytest.raises(ValueError, match="measure"):
        optimization_model.get_minimum_drawdown_weights(returns, "ulcer")


def test_the_cdar_of_the_minimum_cdar_portfolio_is_its_objective():
    returns, _ = _reference_returns()
    weights = optimization_model.get_minimum_drawdown_weights(returns)
    drawdowns = optimization_model.get_drawdowns(returns @ weights)
    cdar = optimization_model.get_expected_tail_loss(drawdowns, 0.05)

    # Every other portfolio, e.g. equal weights, has a higher CDaR.
    equal = optimization_model.get_drawdowns(returns.mean(axis=1))
    assert cdar < optimization_model.get_expected_tail_loss(equal, 0.05)
    # 500 periods at 95%: the average of exactly the 25 largest drawdowns.
    assert cdar == pytest.approx(drawdowns.nlargest(25).mean())


def test_expected_tail_loss_counts_a_fraction_of_the_boundary_loss():
    losses = np.arange(1, 11, dtype=float)

    # The worst 15% of ten losses: 10 in full and half of 9.
    assert optimization_model.get_expected_tail_loss(losses, 0.15) == pytest.approx(
        (10 + 0.5 * 9) / 1.5
    )
    assert optimization_model.get_expected_tail_loss(losses, 0.2) == pytest.approx(9.5)


def test_maximum_omega_and_cvar_ratio_match_riskfolio_and_skfolio():
    returns, _ = _reference_returns()

    # Riskfolio-Lib's Portfolio.optimization(obj="Sharpe", rm="FLPM") maximizes
    # (Ω - 1) and obj="Sharpe", rm="CVaR" the CVaR ratio, with rf as the threshold;
    # skfolio's MeanRisk(MAXIMIZE_RATIO, risk_measure=CVAR) agrees to 3e-8.
    np.testing.assert_allclose(
        optimization_model.get_maximum_omega_weights(returns),
        [0.1456205976, 0.1248614985, 0.2238105684, 0.0084553791, 0.4972519564],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        optimization_model.get_maximum_omega_weights(returns, 1e-4),
        [0.1182706293, 0.0962259967, 0.253212229, 0, 0.5322911102],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        optimization_model.get_maximum_cvar_ratio_weights(returns),
        [0.1228439116, 0.1886678274, 0.2194983203, 0, 0.4689899363],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        optimization_model.get_maximum_cvar_ratio_weights(returns, 0.05, 1e-4),
        [0.1160064356, 0.1761774468, 0.2405297615, 0, 0.4672863516],
        atol=1e-6,
    )


def test_ratio_optimizers_need_a_return_above_the_threshold_and_take_constraints():
    returns, _ = _reference_returns()
    with pytest.raises(ValueError, match="minimum acceptable return"):
        optimization_model.get_maximum_omega_weights(returns, 0.01)
    with pytest.raises(ValueError, match="risk free rate"):
        optimization_model.get_maximum_cvar_ratio_weights(returns, 0.05, 0.01)

    for optimizer in (
        optimization_model.get_maximum_omega_weights,
        optimization_model.get_maximum_cvar_ratio_weights,
        optimization_model.get_minimum_drawdown_weights,
    ):
        weights = optimizer(
            _long_short_returns(),
            minimum_weight=-0.3,
            maximum_weight={"E": 0.25},
            groups=SECTORS.reindex(list("ABCDEF")),
            group_bounds={"Sector": {"Tech": (0, 0.4)}},
            gross_exposure=1.3,
        )
        assert weights.sum() == pytest.approx(1)
        assert weights.abs().sum() <= 1.3 + 1e-7
        assert weights["E"] <= 0.25 + 1e-7
        assert weights[["A", "B"]].sum() <= 0.4 + 1e-7
        assert (weights >= -0.3 - 1e-7).all()


def test_nested_clustered_maximum_sharpe_holds_minimum_variance_in_a_cluster_below_the_risk_free_rate():
    returns = _block_returns()
    covariance = returns.cov() * 252
    expected_returns = returns.mean() * 252
    # The third block earns less than the risk free rate.
    expected_returns.iloc[7:] = 0.001

    # The cluster has no maximum Sharpe portfolio of its own; it holds its minimum
    # variance portfolio and the step across the clusters gives it no weight.
    weights = optimization_model.get_nested_clustered_optimization_weights(
        expected_returns, covariance, "maximum_sharpe", 0.01, clusters=3
    )

    assert weights.sum() == pytest.approx(1)
    assert (weights >= 0).all()
    assert weights.iloc[7:].sum() == pytest.approx(0, abs=1e-8)
    for clusters in (1, 3):
        with pytest.raises(ValueError, match="No asset has an expected return"):
            optimization_model.get_nested_clustered_optimization_weights(
                expected_returns * 0,
                covariance,
                "maximum_sharpe",
                0.01,
                clusters=clusters,
            )


def test_minimum_drawdown_refuses_the_empty_market_neutral_portfolio():
    returns = _block_returns()
    with pytest.raises(ValueError, match="Holding nothing"):
        optimization_model.get_minimum_drawdown_weights(
            returns, minimum_weight=-1.0, net_exposure=0.0, gross_exposure=1.0
        )


CURRENT_FIVE = pd.Series([0.4, 0.3, 0.1, 0.1, 0.1], index=list("ABCDE"))


def test_the_drawdown_and_ratio_optimizers_limit_the_turnover():
    returns, _ = _reference_returns()

    # The same problems written out in cvxpy 1.7 (Clarabel), with the turnover
    # Σ |w_i - w0_i| ≤ L, and for the ratios Σ |y_i - κ w0_i| ≤ L κ; they agree to 3e-7.
    references = {
        (optimization_model.get_minimum_drawdown_weights, 0.1): [
            0.4,
            0.25,
            0.15,
            0.1,
            0.1,
        ],
        (optimization_model.get_maximum_cvar_ratio_weights, 0.1): [
            0.35,
            0.3,
            0.1,
            0.1,
            0.15,
        ],
        (optimization_model.get_maximum_cvar_ratio_weights, 0.3): [
            0.2894037749,
            0.2605962251,
            0.1,
            0.1,
            0.25,
        ],
        (optimization_model.get_maximum_omega_weights, 0.1): [
            0.358543048,
            0.291456952,
            0.1,
            0.1,
            0.15,
        ],
        (optimization_model.get_maximum_omega_weights, 0.3): [
            0.2926110051,
            0.2573889949,
            0.1,
            0.1,
            0.25,
        ],
    }
    for (optimizer, turnover), reference in references.items():
        weights = optimizer(
            returns, current_weights=CURRENT_FIVE, maximum_turnover=turnover
        )
        np.testing.assert_allclose(weights, reference, atol=1e-6)
        assert (weights - CURRENT_FIVE).abs().sum() <= turnover + 1e-8

    maximum = optimization_model.get_minimum_drawdown_weights(
        returns, "maximum", current_weights=CURRENT_FIVE, maximum_turnover=0.1
    )
    np.testing.assert_allclose(maximum, [0.4, 0.25, 0.15, 0.1, 0.1], atol=1e-6)


def test_the_ratio_optimizers_take_expected_returns():
    returns, _ = _reference_returns()
    expected = pd.Series([0.0006, 0.0002, 0.0004, 0.0005, 0.0001], index=list("ABCDE"))

    # Riskfolio-Lib 7.4.1 optimization(obj="Sharpe", rm="FLPM" / "CVaR") with port.mu
    # set to these expected returns.
    np.testing.assert_allclose(
        optimization_model.get_maximum_omega_weights(
            returns, expected_returns=expected
        ),
        [0.6338427083, 0, 0.1340901416, 0.23206715, 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        optimization_model.get_maximum_cvar_ratio_weights(
            returns, expected_returns=expected
        ),
        [0.5991221747, 0, 0.1482425477, 0.2526352776, 0],
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="every asset"):
        optimization_model.get_maximum_omega_weights(
            returns, expected_returns=expected.drop("E")
        )


def test_expected_shortfall_contributions_count_a_fraction_of_the_boundary_period():
    returns, _ = _reference_returns()
    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=returns.columns)

    # Riskfolio-Lib Risk_Contribution(rm="CVaR"), with tails of 22.5 and 13.5 periods.
    for alpha, reference in (
        (0.045, [0.0071414198, 0.006284051, 0.0043148177, 0.0035652219, 0.0007657886]),
        (0.027, [0.0079713688, 0.0068635433, 0.0046414049, 0.0035747537, 0.0009117271]),
    ):
        contributions = optimization_model.get_expected_shortfall_contributions(
            weights, returns, alpha
        )
        np.testing.assert_allclose(
            contributions["Risk Contribution"], reference, atol=1e-9
        )
        assert contributions["Risk Contribution"].sum() == pytest.approx(
            optimization_model.get_expected_tail_loss(-(returns @ weights), alpha)
        )


def test_tail_probabilities_sum_to_one_with_a_fraction_on_the_boundary():
    losses = np.array([3.0, 10.0, 1.0, 9.0, 5.0])
    probabilities = optimization_model.get_tail_probabilities(losses, 0.5)

    # 2.5 periods: 10 and 9 in full and half of 5.
    np.testing.assert_allclose(probabilities, [0, 0.4, 0, 0.4, 0.2])
    assert probabilities.sum() == pytest.approx(1)
    # A tail shorter than a period is the largest loss.
    assert optimization_model.get_expected_tail_loss(losses, 0.01) == pytest.approx(10)
    with pytest.raises(InvalidArgumentError, match="alpha is a tail probability"):
        optimization_model.get_tail_probabilities(losses, 1)


def test_resampled_efficient_weights_average_the_optimizer_over_the_draws():
    returns, _ = _reference_returns()
    expected, covariance = returns.mean() * 252, returns.cov() * 252

    # Every draw solved by Riskfolio-Lib 7.4.1 (MinRisk, MV) or cvxpy (the risk
    # aversion utility) on the same draws, numpy's multivariate_normal(method="eigh")
    # and integers for the bootstrap with default_rng(7), and averaged.
    for objective, resampling, reference in (
        (
            "minimum_variance",
            "parametric",
            [0.229921588, 0.1330852067, 0.0955409025, 0.0806252357, 0.460827067],
        ),
        (
            "minimum_variance",
            "bootstrap",
            [0.2305472472, 0.1338746262, 0.0980794088, 0.0806223961, 0.4568763219],
        ),
        (
            "mean_variance",
            "parametric",
            [0.1483744711, 0.2223718445, 0.1965154474, 0.1280152188, 0.3047230183],
        ),
    ):
        draws = optimization_model.get_resampled_efficient_weights(
            returns, expected, covariance, 252, objective, 200, resampling, 7
        )
        assert len(draws) == 200  # noqa: PLR2004
        np.testing.assert_allclose(draws.mean(), reference, atol=1e-5)

    # The draws match numpy's multivariate normal draws for the same seed, from the
    # same matrix (the sign of an eigenvector, and so the draws, can flip with the last
    # digit).
    generator = np.random.default_rng(3)
    sample = generator.multivariate_normal(
        expected / 252, covariance / 252, size=len(returns), method="eigh"
    )
    first = optimization_model.get_resampled_efficient_weights(
        returns, expected, covariance, 252, "minimum_variance", 1, seed=3
    )
    np.testing.assert_allclose(
        first.iloc[0],
        optimization_model.get_minimum_variance_weights(
            pd.DataFrame(
                np.cov(sample, rowvar=False) * 252,
                index=list("ABCDE"),
                columns=list("ABCDE"),
            )
        ),
        atol=1e-6,
    )


def test_resampled_efficient_weights_leave_out_draws_without_a_solution():
    returns, _ = _reference_returns()
    expected, covariance = returns.mean() * 252, returns.cov() * 252

    # With a risk free rate near the expected returns, some draws have no asset above
    # it and no maximum Sharpe ratio portfolio.
    draws = optimization_model.get_resampled_efficient_weights(
        returns, expected, covariance, 252, resamples=100, risk_free_rate=0.12
    )
    assert 0 < len(draws) < 100  # noqa: PLR2004
    np.testing.assert_allclose(draws.sum(axis=1), 1, atol=1e-8)
    assert (draws >= -1e-10).all().all()

    capped = optimization_model.get_resampled_efficient_weights(
        returns, expected, covariance, 252, resamples=50, maximum_weight=0.3
    )
    assert (capped.mean() <= 0.3 + 1e-8).all()

    with pytest.raises(ValueError, match="None of the"):
        optimization_model.get_resampled_efficient_weights(
            returns, expected, covariance, 252, resamples=5, risk_free_rate=10
        )
    with pytest.raises(ValueError, match="objective"):
        optimization_model.get_resampled_efficient_weights(
            returns, expected, covariance, 252, "maximum_return"
        )
    with pytest.raises(ValueError, match="resampling"):
        optimization_model.get_resampled_efficient_weights(
            returns, expected, covariance, 252, resampling="jackknife"
        )


def test_resampled_efficient_weights_settle_as_the_draws_grow():
    returns, _ = _reference_returns()
    expected, covariance = returns.mean() * 252, returns.cov() * 252

    # The spread of the average over seeds falls with the number of draws, about as
    # 1 / √S.
    def spread(resamples):
        return np.std(
            [
                optimization_model.get_resampled_efficient_weights(
                    returns, expected, covariance, 252, resamples=resamples, seed=seed
                )
                .mean()
                .to_numpy()
                for seed in range(8)
            ],
            axis=0,
        ).max()

    assert spread(100) < spread(10) / 2


def test_market_neutral_ratio_portfolios_start_away_from_zero():
    rng = np.random.default_rng(3)
    volatility = rng.uniform(0.1, 0.4, 6)
    samples = (
        rng.normal(size=(500, 6))
        @ np.linalg.cholesky(np.eye(6) * 0.5 + 0.5).T
        * volatility
        / np.sqrt(252)
    )
    assets = [f"A{number}" for number in range(6)]
    covariance = pd.DataFrame(samples, columns=assets).cov() * 252
    expected_returns = pd.Series(rng.uniform(-0.05, 0.2, 6), index=assets)
    neutral = {
        "minimum_weight": -0.5,
        "maximum_weight": 0.5,
        "net_exposure": 0.0,
        "gross_exposure": 1.0,
    }
    sigma = covariance.to_numpy()

    sharpe = optimization_model.get_maximum_sharpe_weights(
        expected_returns, covariance, **neutral
    )
    diversification = optimization_model.get_maximum_diversification_weights(
        covariance, **neutral
    )

    # An independent SLSQP over the split w = w⁺ − w⁻ with Σ (w⁺ + w⁻) ≤ 1, from several
    # random starts, reaches the same Sharpe ratio of 1.2915.
    assert (sharpe @ expected_returns) / np.sqrt(
        sharpe @ sigma @ sharpe
    ) == pytest.approx(1.2915407, abs=1e-5)
    for weights in (sharpe, diversification):
        assert weights.sum() == pytest.approx(0, abs=1e-8)
        assert weights.abs().sum() <= 1 + 1e-8
        assert weights.abs().sum() > 0.5

    # The lowest risk at a net exposure of 0 is holding nothing.
    assert (
        optimization_model.get_minimum_variance_weights(covariance, **neutral) == 0
    ).all()
    frontier = optimization_model.get_efficient_frontier(
        expected_returns, covariance, points=4, **neutral
    )
    assert (frontier.loc[1, assets] == 0).all()
    assert np.isnan(frontier.loc[1, "Sharpe Ratio"])
    assert frontier.loc[2:, "Sharpe Ratio"].notna().all()


@pytest.mark.parametrize("net_exposure", [0.0, -0.5])
def test_a_net_exposure_of_zero_or_below_needs_short_positions(net_exposure):
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=list("ABC"), columns=list("ABC"))

    with pytest.raises(ValueError, match="needs short positions"):
        optimization_model.get_minimum_variance_weights(
            covariance, net_exposure=net_exposure
        )


@pytest.mark.parametrize("risk_measure", ["semideviation", "mad", "cdar"])
def test_historical_risk_contributions_add_up_to_the_measure(risk_measure):
    returns, _ = _reference_returns()
    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=returns.columns)
    contributions = optimization_model.get_historical_risk_contributions(
        weights, returns, risk_measure
    )
    portfolio = returns @ weights
    measure = {
        "semideviation": np.sqrt((portfolio.clip(upper=0) ** 2).mean()),
        "mad": (portfolio - portfolio.mean()).abs().mean(),
        "cdar": optimization_model.get_expected_tail_loss(
            optimization_model.get_drawdowns(portfolio), 0.05
        ),
    }[risk_measure]
    assert contributions["Risk Contribution"].sum() == pytest.approx(measure)
    assert contributions["Percentage Contribution"].sum() == pytest.approx(1)

    # The marginal contributions are the derivatives of the measure.
    def value(vector):
        return optimization_model.get_historical_risk_contributions(
            pd.Series(vector, index=returns.columns), returns, risk_measure
        )["Risk Contribution"].sum()

    step = 1e-7
    for position, asset in enumerate(returns.columns):
        shift = np.eye(len(weights))[position] * step
        derivative = (value(weights + shift) - value(weights - shift)) / (2 * step)
        assert contributions.loc[asset, "Marginal Contribution"] == pytest.approx(
            derivative, abs=1e-7
        )


def test_historical_risk_contributions_refuse_an_unknown_measure():
    returns, _ = _reference_returns()
    with pytest.raises(InvalidArgumentError, match="risk_measure must be one of"):
        optimization_model.get_historical_risk_contributions(
            pd.Series(0.2, index=returns.columns), returns, "variance"
        )


def test_discrete_allocation_finds_the_closest_whole_shares():
    weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    prices = pd.Series([129.93, 239.82, 57.4], index=["A", "B", "C"])
    integer = optimization_model.get_discrete_allocation(weights, prices, 3000)
    greedy = optimization_model.get_discrete_allocation(
        weights, prices, 3000, method="greedy"
    )

    def deviation(table):
        amounts = table["Amount"]
        return (weights * 3000 - amounts.drop("Leftover Cash")).abs().sum() + amounts[
            "Leftover Cash"
        ]

    # Every allocation within the budget, tried one by one.
    target = weights * 3000
    best = min(
        np.abs(target - np.array(shares) * prices).sum()
        + 3000
        - np.array(shares) @ prices
        for shares in itertools.product(
            *[range(int(amount // price) + 3) for amount, price in zip(target, prices)]
        )
        if np.array(shares) @ prices <= 3000  # noqa: PLR2004
    )
    assert deviation(integer) == pytest.approx(best)
    assert deviation(greedy) >= deviation(integer) - 1e-9
    for table in (integer, greedy):
        assert table["Amount"].sum() == pytest.approx(3000)
        assert table.loc["Leftover Cash", "Amount"] >= 0
        assert (table["Shares"].dropna() % 1 == 0).all()


def test_discrete_allocation_holds_short_weights_as_short_shares():
    weights = pd.Series([1.3, -0.3], index=["A", "B"])
    prices = pd.Series([129.93, 239.82], index=["A", "B"])
    for method in ("integer", "greedy"):
        table = optimization_model.get_discrete_allocation(
            weights, prices, 5000, method=method
        )
        assert table.loc["A", "Shares"] > 0
        assert table.loc["B", "Shares"] < 0
        assert table.loc["Leftover Cash", "Amount"] >= 0
        assert table["Amount"].sum() == pytest.approx(5000)


def test_discrete_allocation_needs_a_price_for_every_asset():
    with pytest.raises(ValueError, match="no price"):
        optimization_model.get_discrete_allocation(
            pd.Series([0.5, 0.5], index=["A", "B"]),
            pd.Series([10.0, np.nan], index=["A", "B"]),
        )


def test_portfolio_risk_metrics_use_the_risk_and_performance_formulas():
    returns, benchmark = _reference_returns()
    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=returns.columns)
    portfolio = returns @ weights
    metrics = optimization_model.get_portfolio_risk_metrics(
        pd.DataFrame({"Portfolio": portfolio, "Benchmark": benchmark}), 252, 0.02
    )["Portfolio"]
    excess = portfolio - 0.02 / 252
    assert metrics["Annualized Return"] == pytest.approx(
        (1 + portfolio).prod() ** (252 / 500) - 1
    )
    assert metrics["Annualized Volatility"] == pytest.approx(
        portfolio.std() * np.sqrt(252)
    )
    assert metrics["Sharpe Ratio"] == pytest.approx(
        excess.mean() / excess.std() * np.sqrt(252)
    )
    assert metrics["Sortino Ratio"] == pytest.approx(
        excess.mean() / np.sqrt((excess.clip(upper=0) ** 2).mean()) * np.sqrt(252)
    )
    wealth = (1 + portfolio).cumprod()
    assert metrics["Maximum Drawdown"] == pytest.approx(
        (wealth / wealth.cummax().clip(lower=1) - 1).min()
    )
    assert metrics["Value at Risk"] == pytest.approx(portfolio.quantile(0.05))
    assert metrics["Conditional Value at Risk"] == pytest.approx(
        portfolio[portfolio <= portfolio.quantile(0.05)].mean()
    )


def _factor_returns(periods: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    """Seeded returns of four assets driven by one market factor."""
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, periods))
    returns = pd.DataFrame(
        market.to_numpy()[:, None] * np.array([0.3, 1.5, 2.5, 0.1])
        + rng.normal(0, 0.01, (periods, 4)),
        columns=ASSETS,
    )
    return returns, market


def test_ledoit_wolf_constant_correlation_follows_the_paper():
    returns, _ = _factor_returns()
    y = returns.to_numpy() - returns.to_numpy().mean(axis=0)
    periods, count = y.shape
    sample = y.T @ y / periods
    sd = np.sqrt(np.diag(sample))
    r_bar = (
        2
        / (count * (count - 1))
        * sum(
            sample[i, j] / (sd[i] * sd[j])
            for i in range(count)
            for j in range(i + 1, count)
        )
    )
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(sample))
    pi = np.array(
        [
            [np.mean((y[:, i] * y[:, j] - sample[i, j]) ** 2) for j in range(count)]
            for i in range(count)
        ]
    )

    def theta(i, j):
        return np.mean(
            (y[:, i] ** 2 - sample[i, i]) * (y[:, i] * y[:, j] - sample[i, j])
        )

    rho = np.trace(pi) + sum(
        r_bar
        / 2
        * (
            np.sqrt(sample[j, j] / sample[i, i]) * theta(i, j)
            + np.sqrt(sample[i, i] / sample[j, j]) * theta(j, i)
        )
        for i in range(count)
        for j in range(count)
        if i != j
    )
    gamma = np.sum((target - sample) ** 2)
    intensity = max(0, min((pi.sum() - rho) / gamma / periods, 1))

    covariance, shrinkage = (
        optimization_model.get_ledoit_wolf_constant_correlation_covariance(returns)
    )

    assert 0 < shrinkage < 1
    assert shrinkage == pytest.approx(intensity, rel=1e-10)
    assert np.allclose(
        covariance, intensity * target + (1 - intensity) * sample, atol=1e-15
    )


def test_ledoit_wolf_single_factor_follows_the_paper():
    returns, market = _factor_returns()
    y = returns.to_numpy() - returns.to_numpy().mean(axis=0)
    m = market.to_numpy() - market.mean()
    periods, count = y.shape
    sample = y.T @ y / periods
    s_m = y.T @ m / periods
    s_mm = m @ m / periods
    target = np.outer(s_m, s_m) / s_mm
    np.fill_diagonal(target, np.diag(sample))
    pi = np.array(
        [
            [np.mean((y[:, i] * y[:, j] - sample[i, j]) ** 2) for j in range(count)]
            for i in range(count)
        ]
    )
    # r_ij of Ledoit and Wolf (2003), averaged over the periods.
    rho = np.trace(pi) + sum(
        np.mean(
            (s_m[j] * s_mm * y[:, i] + s_m[i] * s_mm * y[:, j] - s_m[i] * s_m[j] * m)
            * m
            * y[:, i]
            * y[:, j]
            / s_mm**2
            - target[i, j] * sample[i, j]
        )
        for i in range(count)
        for j in range(count)
        if i != j
    )
    gamma = np.sum((target - sample) ** 2)
    intensity = max(0, min((pi.sum() - rho) / gamma / periods, 1))

    covariance, shrinkage = optimization_model.get_ledoit_wolf_single_factor_covariance(
        returns, market
    )

    assert 0 < shrinkage < 1
    assert shrinkage == pytest.approx(intensity, rel=1e-10)
    assert np.allclose(
        covariance, intensity * target + (1 - intensity) * sample, atol=1e-15
    )

    # The dispatch annualizes it and needs the benchmark.
    _, annual = optimization_model.get_expected_returns_and_covariance(
        returns, 252, "ledoit_wolf_single_factor", benchmark_returns=market
    )
    assert np.allclose(annual, covariance * 252)
    with pytest.raises(ValueError, match="benchmark"):
        optimization_model.get_expected_returns_and_covariance(
            returns, 252, "ledoit_wolf_single_factor"
        )


def test_target_volatility_gives_the_frontier_portfolio_at_that_risk():
    weights = optimization_model.get_mean_variance_weights(
        EXPECTED_RETURNS, COVARIANCE, target_volatility=0.15
    )
    assert np.sqrt(weights @ COVARIANCE @ weights) == pytest.approx(0.15, abs=1e-7)

    # The same portfolio as the lowest variance one at its expected return.
    at_return = optimization_model.get_mean_variance_weights(
        EXPECTED_RETURNS, COVARIANCE, target_return=float(weights @ EXPECTED_RETURNS)
    )
    assert np.allclose(weights, at_return, atol=1e-5)

    # Above the volatility of the highest return portfolio the target does not bind.
    highest = optimization_model.get_mean_variance_weights(
        EXPECTED_RETURNS, COVARIANCE, target_volatility=0.5
    )
    assert highest["B"] == pytest.approx(1)

    lowest = optimization_model.get_minimum_variance_weights(COVARIANCE)
    lowest_volatility = float(np.sqrt(lowest @ COVARIANCE @ lowest))
    with pytest.raises(ValueError, match="below the lowest volatility"):
        optimization_model.get_mean_variance_weights(
            EXPECTED_RETURNS, COVARIANCE, target_volatility=lowest_volatility - 0.01
        )
    with pytest.raises(ValueError, match="not both"):
        optimization_model.get_mean_variance_weights(
            EXPECTED_RETURNS, COVARIANCE, target_return=0.1, target_volatility=0.15
        )


def test_codependence_sets_the_correlation_that_is_clustered():
    returns, _ = _factor_returns(200)
    covariance = returns.cov()

    pearson = optimization_model.get_codependence(returns, covariance)
    spearman = optimization_model.get_codependence(returns, covariance, "spearman")
    kendall = optimization_model.get_codependence(returns, covariance, "kendall")

    assert np.allclose(pearson, returns.corr())
    # Spearman's correlation is the Pearson correlation of the ranks.
    assert np.allclose(spearman, returns.rank().corr())
    # Kendall's tau from its concordant and discordant pairs (no ties here).
    a, b = returns["A"].to_numpy(), returns["B"].to_numpy()
    signs = np.sign(a[:, None] - a[None, :]) * np.sign(b[:, None] - b[None, :])
    assert kendall.loc["A", "B"] == pytest.approx(signs.sum() / (len(a) * (len(a) - 1)))
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        optimization_model.get_codependence(returns, covariance, "euclidean")

    # The Pearson correlation of the covariance is the default of the clustering.
    assert np.allclose(
        optimization_model.get_hierarchical_risk_parity_weights(covariance),
        optimization_model.get_hierarchical_risk_parity_weights(
            covariance, correlation=pearson
        ),
    )
    # A correlation that pairs A with B and C with D, where the returns pair A with D
    # and B with C, changes the clusters and the weights.
    swapped = pd.DataFrame(
        [[1, 0.9, 0, 0], [0.9, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0.9, 1]],
        index=ASSETS,
        columns=ASSETS,
    )
    _, natural = optimization_model.get_hierarchical_clusters(covariance, "ward", 2)
    _, labels = optimization_model.get_hierarchical_clusters(
        covariance, "ward", 2, swapped
    )
    assert natural[0] == natural[3] and natural[1] == natural[2]
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert not np.allclose(
        optimization_model.get_hierarchical_risk_parity_weights(covariance),
        optimization_model.get_hierarchical_risk_parity_weights(
            covariance, correlation=swapped
        ),
    )


def test_black_litterman_posterior_covariance_follows_he_and_litterman():
    market = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
    tau = 0.05
    sigma = COVARIANCE.to_numpy()

    _, without = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, tau=tau
    )
    assert np.allclose(without, (1 + tau) * sigma)

    views = {"A": 0.1, "B-C": 0.02}
    _, posterior = optimization_model.get_black_litterman_returns(
        COVARIANCE, market, views=views, tau=tau
    )
    p = np.array([[1, 0, 0, 0], [0, 1, -1, 0]])
    omega = np.diag(np.diag(tau * p @ sigma @ p.T))
    remaining = np.linalg.inv(
        np.linalg.inv(tau * sigma) + p.T @ np.linalg.inv(omega) @ p
    )
    assert np.allclose(posterior, sigma + remaining)


def test_market_implied_risk_aversion():
    benchmark = pd.Series([0.01, -0.005, 0.02, 0.0, 0.004, -0.01])
    expected = (benchmark.mean() * 12 - 0.02) / (benchmark.var(ddof=1) * 12)

    assert optimization_model.get_market_implied_risk_aversion(
        benchmark, 12, 0.02
    ) == pytest.approx(expected)
    with pytest.raises(ValueError, match="no positive risk aversion"):
        optimization_model.get_market_implied_risk_aversion(benchmark, 12, 0.5)


def test_minimum_variance_of_near_cash_assets_matches_the_closed_form():
    # Annualized variances of about 1e-5, as of Treasury bill funds: an absolute tolerance
    # on the variance would stop the solver long before the optimum.
    rng = np.random.default_rng(0)
    volatility = np.array([0.002, 0.0025, 0.003, 0.0035, 0.004])
    loadings = rng.normal(size=(5, 5))
    correlation = np.corrcoef(loadings @ loadings.T + 5 * np.eye(5))
    assets = list("ABCDE")
    covariance = pd.DataFrame(
        np.outer(volatility, volatility) * correlation, index=assets, columns=assets
    )
    inverse = np.linalg.solve(covariance.to_numpy(), np.ones(5))

    weights = optimization_model.get_minimum_variance_weights(
        covariance, minimum_weight=-1, maximum_weight=2
    )
    np.testing.assert_allclose(weights, inverse / inverse.sum(), atol=1e-8)

    # The same holds for a target return: the Lagrange conditions of
    # min w'Σw s.t. 1'w = 1 and μ'w = target, solved as one linear system.
    mu = pd.Series([0.040, 0.041, 0.042, 0.043, 0.044], index=assets)
    target = float(mu @ (inverse / inverse.sum())) + 0.0005
    system = np.zeros((7, 7))
    system[:5, :5] = 2 * covariance.to_numpy()
    system[:5, 5], system[5, :5] = 1, 1
    system[:5, 6], system[6, :5] = mu, mu
    exact = np.linalg.solve(system, np.r_[np.zeros(5), 1, target])[:5]
    weights = optimization_model.get_mean_variance_weights(
        mu, covariance, target_return=target, minimum_weight=-1, maximum_weight=2
    )
    np.testing.assert_allclose(weights, exact, atol=1e-6)


def test_discrete_allocation_shorts_beyond_the_target_to_fund_a_long_share():
    weights = pd.Series([1.1, -0.1], index=["MSFT", "AAPL"])
    prices = pd.Series([239.82, 129.93], index=["MSFT", "AAPL"])
    table = optimization_model.get_discrete_allocation(weights, prices, 22000)

    # 18 shares short instead of the 17 that reach the target pay for a 101st long share.
    assert table.loc["MSFT", "Shares"] == 101  # noqa: PLR2004
    assert table.loc["AAPL", "Shares"] == -18  # noqa: PLR2004
    deviation = (weights * 22000 - table["Amount"].drop("Leftover Cash")).abs().sum()
    assert deviation + table.loc["Leftover Cash", "Amount"] == pytest.approx(277.48)

    # Every allocation within the budget, with shorts up to twice their target, tried one by one.
    best = min(
        abs(24200 - long * 239.82)
        + abs(-2200 + short * 129.93)
        + 22000
        - long * 239.82
        + short * 129.93
        for long in range(102)
        for short in range(35)
        if long * 239.82 - short * 129.93 <= 22000  # noqa: PLR2004
    )
    assert best == pytest.approx(277.48)


def test_discrete_allocation_solves_a_long_short_portfolio_in_any_currency_amount():
    weights = pd.Series([1.3, -0.3], index=["AAPL", "MSFT"])
    prices = pd.Series([129.93, 239.82], index=["AAPL", "MSFT"])
    for value in (19000, 19000 * 1e4, 190):
        table = optimization_model.get_discrete_allocation(weights, prices, value)
        assert table.loc["Leftover Cash", "Amount"] >= 0
        assert table["Amount"].sum() == pytest.approx(value)
    table = optimization_model.get_discrete_allocation(weights, prices, 19000)
    assert table["Shares"].dropna().tolist() == [190, -24]


def test_shrunk_expected_returns_match_skfolio_and_pyportfolioopt():
    returns, _ = _reference_returns()
    # skfolio 1.14.4 ShrunkMu(method=JAMES_STEIN).fit(returns).mu_
    james_stein, shrinkage = estimation_model.get_james_stein_expected_returns(returns)
    assert np.allclose(
        james_stein,
        [0.0004672095, 0.0004761905, 0.0004851714, 0.0004717, 0.000480681],
        atol=1e-12,
    )
    assert 0 < shrinkage < 1
    # skfolio 1.14.4 ShrunkMu(method=BODNAR_OKHRIN).fit(returns).mu_
    assert np.allclose(
        estimation_model.get_bodnar_okhrin_parolya_expected_returns(returns)[0],
        [0.0084635314, 0.0008708127, -0.0067219059, 0.004667172, -0.0029255466],
        atol=1e-11,
    )
    # PyPortfolioOpt 1.6.0 mean_historical_return and ema_historical_return (span=180),
    # both with compounding=True.
    assert np.allclose(
        optimization_model.get_expected_returns(returns, 252, "geometric"),
        [0.0775548999, 0.0915498493, 0.1105849751, 0.078348333, 0.1178475834],
    )
    assert np.allclose(
        optimization_model.get_expected_returns(returns, 252, "exponential_geometric"),
        [0.2357493687, 0.3052073832, 0.6412725694, 0.2321704521, 0.417909466],
    )
    # Riskfolio-Lib's "ewma2", the recursive average.
    assert np.allclose(
        optimization_model.get_expected_returns(
            returns, 1, "exponential", adjust=False
        ),
        returns.ewm(span=180, adjust=False).mean().iloc[-1],
    )


def test_james_stein_does_not_shrink_two_assets_and_keeps_equal_averages():
    returns, _ = _reference_returns()
    _, shrinkage = estimation_model.get_james_stein_expected_returns(
        returns[["A", "B"]]
    )
    assert shrinkage == 0
    equal = returns - returns.mean() + 0.0004
    mean, shrinkage = estimation_model.get_james_stein_expected_returns(equal)
    assert np.allclose(mean, 0.0004) and shrinkage == 1
    with pytest.raises(ValueError, match="more periods"):
        estimation_model.get_bodnar_okhrin_parolya_expected_returns(returns.iloc[:5])


def test_shrunk_and_gerber_covariances_match_the_libraries():
    returns, _ = _reference_returns()
    # PyPortfolioOpt 1.6.0 CovarianceShrinkage(frequency=1).shrunk_covariance(delta=0.1)
    assert np.allclose(
        estimation_model.get_shrunk_covariance(returns).iloc[0],
        [
            0.000204291923,
            6.703121356e-05,
            6.716798336e-05,
            8.319697091e-05,
            3.349463266e-05,
        ],
        atol=1e-15,
    )
    # skfolio 1.14.4 GerberCovariance (psd_variant True and False) and Riskfolio-Lib's
    # gerber_cov_stat0, with the threshold scaled to standard deviations with ddof=1.
    for statistic, row in (
        (
            1,
            [
                0.0002011995639,
                3.574007167e-05,
                4.320555061e-05,
                4.7359949e-05,
                1.659997325e-05,
            ],
        ),
        (
            2,
            [
                0.0002011995639,
                4.907652668e-05,
                5.808555703e-05,
                6.528850649e-05,
                2.315134173e-05,
            ],
        ),
        (
            0,
            [
                0.0002011995639,
                7.828777604e-05,
                8.859319974e-05,
                0.0001050483975,
                3.822480327e-05,
            ],
        ),
    ):
        assert np.allclose(
            estimation_model.get_gerber_covariance(returns, 0.5, statistic).iloc[0],
            row,
            atol=1e-15,
        )
    with pytest.raises(ValueError, match="statistic"):
        estimation_model.get_gerber_covariance(returns, 0.5, 3)


def test_robust_and_graph_covariances_match_the_libraries():
    returns, _ = _reference_returns()
    covariance = estimation_model.get_minimum_covariance_determinant(returns)
    assert np.allclose(
        covariance.iloc[0],
        [
            0.0001926157068,
            6.162985182e-05,
            5.305321627e-05,
            8.378201482e-05,
            2.812951558e-05,
        ],
        rtol=1e-8,
        atol=1e-15,
    )
    # scikit-learn 1.9.1 MinCovDet(random_state=0).fit(returns).covariance_ draws other
    # random subsets and ends in a subset with a slightly larger determinant, so the two
    # agree to a few percent rather than exactly.
    assert np.allclose(
        covariance.iloc[0],
        [
            0.000190424533,
            5.883712985e-05,
            5.165067873e-05,
            8.27543892e-05,
            2.750130636e-05,
        ],
        rtol=0.05,
    )
    raw = estimation_model.get_minimum_covariance_determinant(returns, reweight=False)
    assert np.allclose(
        raw.iloc[0],
        [
            9.831912815e-05,
            1.61831393e-05,
            9.530881342e-06,
            3.254906473e-05,
            1.836728295e-06,
        ],
        rtol=1e-8,
        atol=1e-15,
    )
    # The log-determinant of the raw covariance of scikit-learn's MinCovDet (and of
    # PyPortfolioOpt 1.6.0 min_cov_determinant, which uses it): the subset found here has
    # a smaller one.
    assert np.linalg.slogdet(raw.to_numpy())[1] < -46.25735239675681
    # Riskfolio-Lib 7.4.1 covar_matrix(method="jlogo")
    assert np.allclose(
        estimation_model.get_logo_covariance(returns).iloc[0],
        [
            0.0002011995639,
            7.447912618e-05,
            7.463109262e-05,
            9.244107878e-05,
            1.799519061e-05,
        ],
        rtol=1e-8,
        atol=1e-15,
    )
    # skfolio 1.14.4 GeodesicShrinkageCovariance() and RegimeAdjustedEWCovariance()
    assert np.allclose(
        estimation_model.get_geodesic_shrinkage_covariance(returns).iloc[0],
        [
            0.0002007115225,
            6.454682054e-05,
            6.396514924e-05,
            8.081259758e-05,
            3.313288717e-05,
        ],
        rtol=1e-8,
        atol=1e-15,
    )
    covariance, multiplier = (
        estimation_model.get_regime_adjusted_exponential_covariance(returns)
    )
    assert multiplier == pytest.approx(1.0317845797524008, rel=1e-10)
    assert np.allclose(
        covariance.iloc[0],
        [
            0.0002000370684,
            8.56906179e-05,
            7.54182817e-05,
            8.877009717e-05,
            3.262295314e-05,
        ],
        rtol=1e-8,
        atol=1e-15,
    )


def test_the_tmfg_is_planar_and_the_logo_covariance_matches_riskfolio():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 0.01, size=(300, 3)) @ rng.uniform(
        -0.5, 1.5, size=(3, 9)
    ) + rng.normal(0, 0.01, size=(300, 9))
    returns = pd.DataFrame(data)
    cliques, separators = estimation_model.get_triangulated_maximally_filtered_graph(
        (1 + returns.corr().to_numpy()) / 2
    )
    links = {
        tuple(sorted(pair))
        for clique in cliques
        for pair in itertools.combinations(clique, 2)
    }
    assert len(cliques) == 6 and len(separators) == 5 and len(links) == 3 * (9 - 2)

    covariance = estimation_model.get_logo_covariance(returns)
    # Riskfolio-Lib 7.4.1 covar_matrix(method="jlogo"), rows 0 and 4
    assert np.allclose(
        covariance.iloc[[0, 4]],
        [
            [
                0.0001317249554,
                7.418686351e-05,
                2.072060258e-05,
                4.365831013e-05,
                5.587314601e-05,
                5.972952688e-05,
                8.486342137e-05,
                7.24960195e-05,
                3.670953614e-05,
            ],
            [
                5.587314601e-05,
                0.0002825002499,
                6.181341499e-05,
                0.0001318769107,
                0.0004437460064,
                0.0002494497729,
                0.0002501185663,
                0.0002157045381,
                0.000109035685,
            ],
        ],
        rtol=1e-8,
        atol=1e-15,
    )
    # The precision matrix is zero for every pair the graph does not link.
    precision = np.linalg.inv(covariance.to_numpy())
    unlinked = [
        pair for pair in itertools.combinations(range(9), 2) if pair not in links
    ]
    assert np.allclose([precision[pair] for pair in unlinked], 0, atol=1e-6)
    # Four or fewer assets are all linked: the sample covariance.
    assert np.allclose(
        estimation_model.get_logo_covariance(returns.iloc[:, :4]),
        returns.iloc[:, :4].cov(),
    )


def test_the_robust_estimators_reject_bad_input():
    returns, _ = _reference_returns()
    with pytest.raises(ValueError, match="two assets"):
        estimation_model.get_minimum_covariance_determinant(returns[["A"]])
    with pytest.raises(ValueError, match="support fraction"):
        estimation_model.get_minimum_covariance_determinant(returns, 1.5)
    with pytest.raises(ValueError, match="target"):
        estimation_model.get_geodesic_shrinkage_covariance(returns, 0.1, "constant")
    with pytest.raises(ValueError, match="at least 40 periods"):
        estimation_model.get_regime_adjusted_exponential_covariance(returns.iloc[:30])
    # The geodesic estimate runs from the sample covariance to the target.
    assert np.allclose(
        estimation_model.get_geodesic_shrinkage_covariance(returns, 0), returns.cov()
    )
    assert np.allclose(
        estimation_model.get_geodesic_shrinkage_covariance(returns, 1, "diagonal"),
        np.diag(returns.var()),
    )
    # The same seed gives the same estimate.
    assert estimation_model.get_minimum_covariance_determinant(returns, seed=5).equals(
        estimation_model.get_minimum_covariance_determinant(returns, seed=5)
    )


def test_the_nearest_correlation_is_positive_definite():
    broken = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    repaired = estimation_model.get_nearest_correlation(broken)
    assert np.allclose(np.diag(repaired), 1)
    assert np.linalg.eigvalsh(repaired).min() > 0


def test_denoised_and_detoned_covariances_match_the_libraries():
    returns, _ = _reference_returns()
    # Riskfolio-Lib 7.4.1 denoiseCov(kind="spectral") and (kind="shrink", alpha=0.1)
    assert np.allclose(
        optimization_model.get_denoised_covariance(returns, "spectral")[0].iloc[0],
        [
            0.0002011995639,
            0.0002276367642,
            0.0002370466696,
            0.0002351083183,
            0.0001724777708,
        ],
        atol=1e-15,
    )
    assert np.allclose(
        optimization_model.get_denoised_covariance(returns, "shrink", 0.1)[0].iloc[0],
        [
            0.0002011995639,
            0.0001058429583,
            0.0001141344841,
            0.0001169234086,
            4.887645446e-05,
        ],
        atol=1e-15,
    )
    # skfolio 1.14.4 DetoneCovariance(n_markets=1)
    assert np.allclose(
        estimation_model.get_detoned_covariance(returns).iloc[0],
        [
            0.0002011995639,
            -6.742798973e-05,
            -8.906452328e-05,
            -5.652099068e-05,
            -1.930956709e-05,
        ],
        atol=1e-15,
    )
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        optimization_model.get_denoised_covariance(returns, "clipped")


def test_codependence_matrices_match_the_libraries():
    returns, _ = _reference_returns()
    # skfolio 1.14.4 DistanceCorrelation and MutualInformation (Freedman-Diaconis bins)
    assert np.allclose(
        estimation_model.get_distance_correlation_matrix(returns).iloc[0],
        [1.0, 0.2777265578, 0.2784778991, 0.3474408113, 0.1969609606],
    )
    information, variation = estimation_model.get_mutual_information_matrix(returns)
    assert np.allclose(
        information.iloc[0], [1.0, 0.1209754278, 0.1232466104, 0.1168981788, 0.10587601]
    )
    assert np.allclose(
        variation.iloc[0], [0.0, 0.936807723, 0.9354425946, 0.9380798469, 0.944779884]
    )
    # Riskfolio-Lib 7.4.1 mutual_info_matrix(bins_info="HGR") and ltdi_matrix(alpha=0.05)
    assert np.allclose(
        estimation_model.get_mutual_information_matrix(returns, "hacine_gharbi")[
            0
        ].iloc[0],
        [1.0, 0.06527586, 0.04542139, 0.07519232, 0.03232228],
        atol=1e-8,
    )
    assert np.allclose(
        estimation_model.get_lower_tail_dependence_matrix(returns).iloc[0],
        [1.0, 0.2, 0.12, 0.24, 0.08],
    )


def test_codependence_distances_feed_the_clustering():
    returns, _ = _reference_returns()
    covariance = returns.cov()
    for method in optimization_model.CODEPENDENCE_METHODS:
        distance = optimization_model.get_codependence_distance(
            returns, covariance, method
        )
        assert np.allclose(np.diag(distance), 0)
        assert (distance.to_numpy() >= 0).all()
        weights = optimization_model.get_hierarchical_risk_parity_weights(
            covariance, distance=distance
        )
        assert weights.sum() == pytest.approx(1)
    # The Pearson distance is the correlation distance the optimizers use without one.
    pearson = optimization_model.get_codependence_distance(returns, covariance)
    pd.testing.assert_series_equal(
        optimization_model.get_hierarchical_risk_parity_weights(covariance),
        optimization_model.get_hierarchical_risk_parity_weights(
            covariance, distance=pearson
        ),
    )
    dcor = optimization_model.get_codependence(returns, covariance, "distance")
    assert np.allclose(
        optimization_model.get_codependence_distance(returns, covariance, "distance"),
        np.sqrt(1 - dcor),
    )


def test_fast_threshold_clusters_match_ffn():
    returns, _ = _reference_returns()
    correlation = returns.corr()
    # ffn 1.4.1 calc_ftca(returns, threshold) gives {D}, {E}, {B}, {A}, {C} at 0.5 and
    # {D, A, B, C}, {E} at 0.2.
    labels = estimation_model.get_fast_threshold_clusters(correlation, 0.5)
    assert list(correlation.index[np.argsort(labels)]) == ["D", "E", "B", "A", "C"]
    labels = estimation_model.get_fast_threshold_clusters(correlation, 0.2)
    assert labels.tolist() == [0, 0, 0, 0, 1]


def test_comoment_matrices_match_performance_analytics():
    returns, _ = _reference_returns()
    # PerformanceAnalytics M3.MM, M4.MM and M3.ewma (λ = 1 - 2 / 181, of the demeaned
    # returns), written out as the sums of Kronecker products of the R functions.
    assert np.allclose(
        estimation_model.get_coskewness_matrix(
            returns, standardized_moments=False
        ).iloc[0, :5],
        [
            2.714127856e-08,
            -2.269730157e-07,
            8.740163754e-09,
            -9.198416238e-08,
            -4.919408914e-08,
        ],
        atol=1e-17,
    )
    assert np.allclose(
        estimation_model.get_cokurtosis_matrix(
            returns, standardized_moments=False
        ).iloc[1, :5],
        [
            5.142156415e-08,
            6.726514486e-08,
            3.971143825e-08,
            3.833282117e-08,
            1.515485499e-08,
        ],
        atol=1e-17,
    )
    assert np.allclose(
        estimation_model.get_coskewness_matrix(
            returns, "exponential", standardized_moments=False
        ).iloc[2, :5],
        [
            4.874726938e-07,
            6.265450241e-07,
            3.675341346e-07,
            -3.041534362e-08,
            5.83275354e-08,
        ],
        atol=1e-17,
    )
    assert np.allclose(
        estimation_model.get_cokurtosis_matrix(
            returns, "semi", standardized_moments=False
        ).iloc[0, :5],
        [
            6.064906238e-08,
            3.378784769e-08,
            2.957372637e-08,
            3.327197593e-08,
            1.642869839e-08,
        ],
        atol=1e-17,
    )
    standardized = estimation_model.get_cokurtosis_matrix(returns)
    centred = returns - returns.mean()
    assert standardized.loc["A", ("A", "A", "A")] == pytest.approx(
        (centred["A"] ** 4).mean() / (centred["A"] ** 2).mean() ** 2
    )


def test_bootstrap_estimates_cover_the_averages():
    returns, _ = _reference_returns()
    indices = estimation_model.get_bootstrap_indices(
        500, "moving", 10, np.random.default_rng(0)
    )
    assert len(indices) == 500 and indices.max() < 500
    # Moving blocks run in consecutive periods.
    assert (np.diff(indices[:10]) == 1).all()
    table, lower, upper = estimation_model.get_bootstrap_estimates(
        returns, n_bootstrap=500
    )
    assert (table["Lower"] < table["Expected Return"]).all()
    assert (table["Upper"] > table["Expected Return"]).all()
    assert (lower.to_numpy() <= upper.to_numpy()).all()
    # The standard error of an average of nearly independent returns is σ / √T; the
    # stationary bootstrap of skfolio 1.14.4 gives the same within sampling error.
    assert np.allclose(
        table["Standard Error"], returns.std() / np.sqrt(len(returns)), rtol=0.15
    )
    normal, _, _ = estimation_model.get_bootstrap_estimates(
        returns, "normal", n_bootstrap=500
    )
    assert np.allclose(
        normal["Standard Error"], returns.std() / np.sqrt(len(returns)), rtol=0.15
    )


def _reference_factors() -> pd.DataFrame:
    """
    The two factors _reference_returns is built from and a third factor of noise, as the
    factor returns of the factor model tests.
    """
    factors = np.random.default_rng(42).normal(0.0004, 0.01, size=(500, 2))
    noise = np.random.default_rng(7).normal(0.0001, 0.006, size=500)
    return pd.DataFrame({"MKT": factors[:, 0], "SMB": factors[:, 1], "NOISE": noise})


def test_factor_loadings_and_covariance_match_riskfolio():
    returns, _ = _reference_returns()
    factors = _reference_factors()
    # Riskfolio-Lib 7.4.1 loadings_matrix (backward, threshold 1: every factor).
    ols = estimation_model.get_factor_loadings(returns, factors)
    assert np.allclose(
        ols.loc["A"], [2.7557241519e-04, 0.66495261593, 0.31542852097, 0.034711426664]
    )
    # The forward selection by p-value and the backward selection by AIC both leave
    # out the noise.
    for selection, criterion in (("forward", "pvalue"), ("backward", "aic")):
        selected = estimation_model.get_factor_loadings(
            returns, factors, selection, criterion
        )
        assert (selected["NOISE"] == 0).all()
        assert np.allclose(
            selected.loc["B"], [3.5394116532e-04, 0.54109093511, 0.76184540587, 0]
        )
    # Riskfolio-Lib PCR with two components.
    assert np.allclose(
        estimation_model.get_factor_loadings(returns, factors, "pcr", components=2).loc[
            "A"
        ],
        [3.5589006336e-04, 0.55884531295, 0.44260245264, 0.12261809151],
    )
    # Riskfolio-Lib risk_factors.
    mean, covariance, _ = estimation_model.get_factor_model(returns, factors, ols)
    assert np.allclose(
        covariance.iloc[0],
        [
            2.0119956392e-04,
            6.8579700403e-05,
            8.1445829507e-05,
            7.7248935555e-05,
            2.81771077e-05,
        ],
        rtol=1e-8,
    )
    # With an intercept the factor model's expected returns are the averages.
    assert np.allclose(mean, returns.mean())


def test_stepwise_selection_keeps_one_factor_and_rejects_unknown_criteria():
    returns, _ = _reference_returns()
    factors = _reference_factors()[["NOISE"]].assign(
        OTHER=np.random.default_rng(1).normal(0, 0.01, 500)
    )
    assert len(estimation_model.get_stepwise_factors(returns["A"], factors)) == 1
    with pytest.raises(ValueError, match="criterion"):
        estimation_model.get_stepwise_factors(returns["A"], factors, criterion="r2")


def test_augmented_and_bayesian_black_litterman_match_riskfolio():
    returns, _ = _reference_returns()
    factors = _reference_factors()
    loadings = estimation_model.get_factor_loadings(returns, factors)
    periods = len(returns)
    # Riskfolio-Lib augmented_black_litterman (τ = 1 / T, δ = 2.5, equal weights, B
    # without a constant) with a view on A and one on SMB.
    table, _ = optimization_model.get_augmented_black_litterman_returns(
        returns.cov(),
        pd.Series(0.2, index=returns.columns),
        factors.cov(),
        loadings.drop(columns="Intercept"),
        views={"A": 0.001},
        factor_views={"SMB": 0.0008},
        risk_aversion=2.5,
        tau=1 / periods,
    )
    assert np.allclose(
        table["Black-Litterman Return"],
        [0.0006785368, 0.0006550672, 0.0007350946, 0.0007361426, 0.0002707868],
        atol=1e-12,
    )
    # Riskfolio-Lib black_litterman_bayesian: its prior N(μ_f, Σ_f) and views with
    # τ = 1 / T are tau=1 and a confidence of T / (T + 1).
    _, _, residual_variance = estimation_model.get_factor_model(
        returns, factors, loadings
    )
    table, covariance = optimization_model.get_bayesian_black_litterman_returns(
        loadings,
        factors.mean(),
        factors.cov(),
        residual_variance,
        {"SMB": 0.0008},
        {"SMB": periods / (periods + 1)},
        tau=1,
    )
    assert np.allclose(
        table["Black-Litterman Return"],
        [0.0007224086, 0.001140982, 0.0013732232, 0.0011761662, 0.0006878033],
        atol=1e-12,
    )
    assert np.allclose(
        covariance.iloc[0],
        [
            2.4374572901e-04,
            1.0333040446e-04,
            1.2128331357e-04,
            1.1685915731e-04,
            4.7605264598e-05,
        ],
        rtol=1e-8,
    )
    assert np.allclose(table["Prior Return"], returns.mean())


def test_entropy_pooling_matches_skfolio_and_meets_the_views():
    returns, _ = _reference_returns()
    probabilities = estimation_model.get_entropy_pooling(
        returns, {"A": 0.001, "B": (">=", 0.0006)}, {"C": 0.0004**0.5}
    )
    # skfolio 1.14.4 EntropyPooling(mean_views=["A == 0.001", "B >= 0.0006"],
    # variance_views=["C == 0.0004"]).
    assert np.allclose(
        probabilities @ returns,
        [0.001, 0.0009506459, 0.0007793088, 0.0006197087, 0.0005345727],
        atol=1e-9,
    )
    assert np.exp(-np.sum(probabilities * np.log(probabilities))) == pytest.approx(
        483.9120721, rel=1e-6
    )
    correlated = estimation_model.get_entropy_pooling(
        returns, correlation_views={"A-B": 0.6}
    )
    # The correlation view holds to the precision of the dual's solver, with the means
    # and variances of A and B kept at those of the prior.
    weighted = np.cov(returns, rowvar=False, aweights=correlated, ddof=0)
    assert weighted[0, 1] / np.sqrt(weighted[0, 0] * weighted[1, 1]) == pytest.approx(
        0.6, abs=1e-4
    )
    assert np.allclose(
        (correlated @ returns)[["A", "B"]],
        returns.mean()[["A", "B"]],
        rtol=0,
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="operator"):
        estimation_model.get_entropy_pooling(returns, {"A": ("!=", 0.001)})


def test_opinion_pooling_matches_skfolio():
    returns, _ = _reference_returns()
    opinions = pd.concat(
        [
            estimation_model.get_entropy_pooling(returns, {"A": 0.001}),
            estimation_model.get_entropy_pooling(returns, {"B": ("<=", 0.0)}),
        ],
        axis=1,
    )
    # skfolio 1.14.4 OpinionPooling(opinion_probabilities=[0.4, 0.4],
    # is_linear_pooling=False, divergence_penalty=1.0).
    pooled = estimation_model.get_opinion_pooling(
        opinions, [0.4, 0.4], "logarithmic", 1.0
    )
    assert np.allclose(
        pooled @ returns,
        [0.0005831489, 0.0003749356, 0.0005577186, 0.0004682266, 0.0005329563],
        atol=1e-10,
    )
    linear = estimation_model.get_opinion_pooling(opinions)
    assert np.allclose(linear, opinions.mean(axis=1))


def test_finite_difference_risk_contributions_match_riskfolio():
    returns, _ = _reference_returns()
    weights = pd.Series([0.3, 0.2, 0.2, 0.2, 0.1], index=returns.columns)
    # Riskfolio-Lib 7.4.1 Risk_Contribution (alpha 0.05).
    references = {
        "evar": [0.008006527, 0.0055135929, 0.0045127416, 0.0056649863, 0.0012344606],
        "maximum_drawdown": [
            0.0377682157,
            0.0428562987,
            0.0099817534,
            0.0361105725,
            0.00371522,
        ],
        "average_drawdown": [
            0.0150599254,
            0.0168259054,
            0.0068143795,
            0.0122606549,
            -0.0008226426,
        ],
        "ulcer_index": [
            0.017279098454,
            0.020111478517,
            0.0066907511406,
            0.014108644968,
            -3.6944107495e-05,
        ],
        "worst_realization": [
            0.0111445088,
            0.0076422823,
            0.0050695402,
            0.0077565192,
            0.0018130024,
        ],
        "gini_mean_difference": [
            0.0034886211,
            0.0025435014,
            0.0027629012,
            0.0028333186,
            0.0004989152,
        ],
        "lower_partial_moment": [
            0.0011502159,
            0.0008066813,
            0.0009023448,
            0.0010039715,
            0.0001521146,
        ],
        "kurtosis": [
            6.0861272849e-05,
            4.3190886333e-05,
            4.6965257678e-05,
            4.5732598942e-05,
            9.0538752575e-06,
        ],
        "entropic_drawdown_at_risk": [
            0.0348477172,
            0.0406381936,
            0.005982999,
            0.0308742672,
            0.0036738177,
        ],
        "tail_gini": [
            0.0075863268,
            0.005239196,
            0.0044990444,
            0.005171983,
            0.0009032992,
        ],
    }
    for measure, reference in references.items():
        table = optimization_model.get_finite_difference_risk_contributions(
            weights, returns, measure
        )
        assert np.allclose(
            table["Risk Contribution"], reference, rtol=1e-6, atol=1e-12
        ), measure
        assert table["Percentage Contribution"].sum() == pytest.approx(1)
    # The contributions add up to the measure itself.
    table = optimization_model.get_finite_difference_risk_contributions(
        weights, returns, "maximum_drawdown"
    )
    measure = optimization_model.get_risk_measure(
        (returns @ weights).to_frame(), "maximum_drawdown"
    )
    assert table["Risk Contribution"].sum() == pytest.approx(measure.iloc[0], rel=1e-6)
    with pytest.raises(InvalidArgumentError, match="risk_measure must be one of"):
        optimization_model.get_finite_difference_risk_contributions(
            weights, returns, "range"
        )


def test_finite_difference_risk_contributions_add_up_at_tied_drawdowns():
    # Three drawdowns of 10% each, driven by A, by B and by all three assets: at this kink
    # the central differences average the one-sided slopes, (0.15, 0.15, 0.05), whose
    # weighted sum of 0.1167 overshoots the measure, so the contributions are scaled to 0.1.
    returns = pd.DataFrame(
        {
            "A": [-0.3, 0.3, 0.0, 0.0, -0.1, 0.1],
            "B": [0.0, 0.0, -0.3, 0.3, -0.1, 0.1],
            "C": [0.0, 0.0, 0.0, 0.0, -0.1, 0.1],
        }
    )
    weights = pd.Series(1 / 3, index=returns.columns)
    table = optimization_model.get_finite_difference_risk_contributions(
        weights, returns, "maximum_drawdown"
    )
    assert table["Risk Contribution"].sum() == pytest.approx(0.1, rel=1e-6)
    scale = 0.1 / (0.35 / 3)
    assert np.allclose(
        table["Marginal Contribution"], np.array([0.15, 0.15, 0.05]) * scale, rtol=1e-5
    )
    assert np.allclose(
        table["Risk Contribution"],
        table["Weight"] * table["Marginal Contribution"],
        rtol=1e-9,
    )


def test_factor_risk_contributions_match_riskfolio():
    returns, _ = _reference_returns()
    factors = _reference_factors()
    weights = pd.Series([0.3, 0.2, 0.2, 0.2, 0.1], index=returns.columns)
    loadings = estimation_model.get_factor_loadings(returns, factors).drop(
        columns="Intercept"
    )
    marginal = optimization_model.get_risk_contributions(weights, returns.cov())[
        "Marginal Contribution"
    ]
    table = optimization_model.get_factor_risk_contributions(
        weights, loadings, marginal
    )
    # Riskfolio-Lib 7.4.1 Factors_Risk_Contribution (rm="MV"): MKT, SMB, NOISE, Other.
    assert np.allclose(
        table["Risk Contribution"],
        [7.5924048242e-03, 3.0548911433e-03, 1.2644958343e-04, 1.7070360056e-05],
        rtol=1e-7,
    )
    assert table["Risk Contribution"].sum() == pytest.approx(
        np.sqrt(weights @ returns.cov() @ weights)
    )


def _heavy_tailed_returns(periods: int = 300) -> pd.DataFrame:
    """
    Six assets with two common factors, Student-t noise with four degrees of freedom and
    rising average returns. The reference weights below come from skfolio 1.12.7,
    PyPortfolioOpt 1.6.0 and Riskfolio-Lib 7.4.1 run on these same returns.
    """
    rng = np.random.default_rng(11)
    factors = rng.normal(0.0005, 0.01, size=(300, 2))
    loadings = rng.uniform(0.2, 1.2, size=(2, 6))
    data = (
        factors @ loadings
        + rng.standard_t(4, size=(300, 6)) * 0.008
        + np.linspace(0.0001, 0.0008, 6)
    )
    return pd.DataFrame(data[:periods], columns=list("ABCDEF"))


@pytest.mark.parametrize(
    ("risk_measure", "objective", "options", "expected", "tolerance"),
    [
        # skfolio's MeanRisk(risk_measure=MEAN_ABSOLUTE_DEVIATION) and Riskfolio-Lib's
        # rm="MAD", which is half the mean absolute deviation, with the same weights.
        (
            "mad",
            "minimum_risk",
            {},
            [0, 0.0347734216, 0.5262817103, 0.3246192576, 0.1096908455, 0.004634765],
            1e-6,
        ),
        # skfolio's GINI_MEAN_DIFFERENCE, solved there as a conic programme.
        (
            "gini_mean_difference",
            "minimum_risk",
            {},
            [
                0.0026359318,
                0.031325571,
                0.5055641101,
                0.3383285235,
                0.1057577612,
                0.0163881023,
            ],
            1e-6,
        ),
        (
            "gini_mean_difference",
            "maximum_ratio",
            {},
            [0, 0, 0, 0.5692902921, 0.104815026, 0.3258946819],
            1e-6,
        ),
        # skfolio's SEMI_DEVIATION (below the mean) and Riskfolio-Lib's rm="SLPM" (below
        # zero, dividing by T - 1, which leaves the weights the same).
        (
            "semideviation",
            "minimum_risk",
            {"minimum_acceptable_return": "mean"},
            [
                0.0847075313,
                0.0031361177,
                0.3971334142,
                0.3678331255,
                0.0935467463,
                0.0536430649,
            ],
            1e-5,
        ),
        (
            "semideviation",
            "minimum_risk",
            {},
            [0.0896392341, 0, 0.2906146879, 0.4204702169, 0.0993671961, 0.099908665],
            1e-5,
        ),
        (
            "worst_realization",
            "maximum_ratio",
            {},
            [0.2659777981, 0.0461838133, 0, 0.4639749503, 0.1833094602, 0.0405539781],
            1e-6,
        ),
        (
            "ulcer_index",
            "minimum_risk",
            {},
            [
                0.0225897069,
                0.0875584094,
                0.0487951834,
                0.4012491176,
                0.1244854761,
                0.3153221065,
            ],
            1e-5,
        ),
        # Riskfolio-Lib's rm="FLPM" (below zero, and below 3 bp with rf = 3 bp), "RG"
        # and "CVRG".
        (
            "lower_partial_moment",
            "minimum_risk",
            {},
            [0, 0.010412575, 0.3725869762, 0.4396263205, 0.0981088711, 0.0792652572],
            1e-6,
        ),
        (
            "lower_partial_moment",
            "maximum_ratio",
            {"minimum_acceptable_return": 0.0003, "risk_free_rate": 0.0003},
            [0, 0, 0, 0.5678425424, 0.1325499623, 0.2996074953],
            1e-6,
        ),
        (
            "range",
            "maximum_ratio",
            {},
            [0, 0, 0, 0.5495244465, 0.1971272058, 0.2533483476],
            1e-6,
        ),
        (
            "cvar_range",
            "minimum_risk",
            {},
            [0.0577358746, 0.0459595966, 0.5635759213, 0.280878097, 0.0518505104, 0],
            1e-6,
        ),
        (
            "mad",
            "minimum_risk",
            {"target_return": 0.0019380920748932906},
            [0, 0, 0.075071018, 0.5298867963, 0.1940962165, 0.2009459692],
            1e-6,
        ),
    ],
)
def test_mean_risk_weights_match_skfolio_and_riskfolio(
    risk_measure, objective, options, expected, tolerance
):
    weights = optimization_model.get_mean_risk_weights(
        _heavy_tailed_returns(), risk_measure, objective, **options
    )
    np.testing.assert_allclose(weights, expected, atol=tolerance)


def test_tail_gini_weights_match_riskfolio_owa_optimization():
    # Riskfolio-Lib's owa_optimization with the exact OWA weights of the Tail Gini (its
    # optimization(rm="TG") approximates them with power cones).
    returns = _heavy_tailed_returns(120)
    np.testing.assert_allclose(
        optimization_model.get_mean_risk_weights(returns, "tail_gini"),
        [0.2846084462, 0.0091859065, 0, 0.4045979515, 0.3016076958, 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        optimization_model.get_mean_risk_weights(
            returns, "tail_gini_range", "maximum_ratio"
        ),
        [0.1755349998, 0, 0, 0.5456848296, 0.2132122843, 0.0655678864],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("risk_measure", "options", "expected"),
    [
        # skfolio's RiskBudgeting and Riskfolio-Lib's rp_optimization.
        (
            "cvar",
            {},
            [
                0.1493162439,
                0.1266643342,
                0.1913236198,
                0.2413999808,
                0.1444466518,
                0.1468491695,
            ],
        ),
        (
            "gini_mean_difference",
            {},
            [
                0.1245107771,
                0.1271034549,
                0.240772619,
                0.2225250969,
                0.1519726552,
                0.1331153969,
            ],
        ),
        (
            "ulcer_index",
            {},
            [
                0.1084087756,
                0.1124421422,
                0.1331280362,
                0.3086170678,
                0.152221135,
                0.1851828433,
            ],
        ),
        (
            "cdar",
            {},
            [
                0.1257794487,
                0.1147119552,
                0.1377592667,
                0.2537011554,
                0.1645675781,
                0.2034805959,
            ],
        ),
        (
            "semideviation",
            {"minimum_acceptable_return": "mean"},
            [
                0.1307015766,
                0.1285613263,
                0.2212932522,
                0.2323165828,
                0.154607222,
                0.1325200401,
            ],
        ),
        (
            "cvar",
            {
                "risk_budget": pd.Series(
                    [0.3, 0.2, 0.2, 0.1, 0.1, 0.1], index=list("ABCDEF")
                )
            },
            [
                0.2264144742,
                0.1285689762,
                0.2159474907,
                0.2302954138,
                0.104429135,
                0.09434451,
            ],
        ),
    ],
)
def test_risk_budgeting_weights_match_skfolio_and_riskfolio(
    risk_measure, options, expected
):
    returns = _heavy_tailed_returns()
    weights = optimization_model.get_risk_budgeting_weights(
        returns, risk_measure, **options
    )
    np.testing.assert_allclose(weights, expected, atol=1e-5)
    if risk_measure not in ("gini_mean_difference", "semideviation"):
        return
    # Every asset contributes its share of the measure; the finite differences of the
    # other measures straddle a kink at the optimum, where they average two slopes.
    budget = options.get("risk_budget", pd.Series(1 / 6, index=returns.columns))
    np.testing.assert_allclose(
        optimization_model.get_mean_risk_contributions(
            weights,
            returns,
            risk_measure,
            minimum_acceptable_return=options.get("minimum_acceptable_return", 0.0),
        ),
        budget / budget.sum(),
        atol=1e-3,
    )


def test_owa_risk_is_the_risk_module_measure_with_its_subgradient():
    returns = _heavy_tailed_returns()
    weights = np.linspace(1, 2, 6) / np.linspace(1, 2, 6).sum()
    portfolio = pd.DataFrame({"P": returns.to_numpy() @ weights})
    for measure in optimization_model.OWA_RISK_MEASURES:
        value, slope = optimization_model.get_owa_risk(
            returns.to_numpy(), weights, measure
        )
        assert value == pytest.approx(
            optimization_model.get_mean_risk_measure(portfolio, measure).iloc[0]
        )
        # Positively homogeneous: the subgradient at w gives the measure back.
        assert slope @ weights == pytest.approx(value)


def test_mean_risk_weights_take_constraints_and_refuse_bad_input():
    returns = _long_short_returns()
    for measure in ("mad", "gini_mean_difference", "semideviation"):
        weights = optimization_model.get_mean_risk_weights(
            returns,
            measure,
            minimum_weight=-0.3,
            maximum_weight={"E": 0.25},
            groups=SECTORS.reindex(list("ABCDEF")),
            group_bounds={"Sector": {"Tech": (0, 0.4)}},
            gross_exposure=1.3,
        )
        assert weights.sum() == pytest.approx(1)
        assert weights.abs().sum() <= 1.3 + 1e-7
        assert weights["E"] <= 0.25 + 1e-7
        assert weights[["A", "B"]].sum() <= 0.4 + 1e-7
    with pytest.raises(InvalidArgumentError, match="risk_measure must be one of"):
        optimization_model.get_mean_risk_weights(returns, "variance")
    with pytest.raises(ValueError, match="target return"):
        optimization_model.get_mean_risk_weights(returns, "mad", target_return=1.0)
    with pytest.raises(ValueError, match="0 or 'mean'"):
        optimization_model.get_risk_budgeting_weights(
            returns, "mad", minimum_acceptable_return=0.001
        )


def test_risk_budgets_name_unknown_and_missing_assets():
    returns = _long_short_returns()
    covariance = returns.cov()
    budgets = {
        "not in the portfolio: FOO": pd.Series(1.0, index=[*"ABCDEF", "FOO"]),
        "without a risk budget: C, D, E, F": pd.Series({"A": 0.5, "B": 0.5}),
        "not positive: F": pd.Series([1, 1, 1, 1, 1, 0.0], index=list("ABCDEF")),
    }
    for message, budget in budgets.items():
        with pytest.raises(ValueError, match=message):
            optimization_model.get_risk_parity_weights(covariance, budget)
        with pytest.raises(ValueError, match=message):
            optimization_model.get_risk_budgeting_weights(returns, "cvar", budget)


def test_critical_line_corners_match_pyportfolioopt():
    # PyPortfolioOpt 1.6.0's CLA on the annualized heavy-tailed returns, long only; it
    # lists the first corner twice (at λ = ∞ and at the first λ).
    returns = _heavy_tailed_returns()
    table = optimization_model.get_critical_line_weights(
        returns.mean() * 252, returns.cov() * 252
    )
    corners = table.loc[table.index.str.startswith("Corner"), list("ABCDEF")]
    np.testing.assert_allclose(
        corners,
        [
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0.557402549, 0, 0.442597451],
            [0, 0, 0, 0.5699268943, 0.1269427932, 0.3031303125],
            [0.0151717076, 0, 0, 0.5656595361, 0.1337849149, 0.2853838413],
            [0.0377655527, 0, 0.3379904172, 0.4036647881, 0.1090234485, 0.1115557936],
            [
                0.0420241622,
                0.0188577777,
                0.4796388873,
                0.3317869026,
                0.0946033341,
                0.0330889361,
            ],
        ],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        table.loc[table.index.str.startswith("Corner"), "Lambda"],
        [0.4797924204, 0.0897615433, 0.034569539, 0.0302708325, 0.0090965035, 0],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        table.loc["Maximum Sharpe", list("ABCDEF")],
        [0, 0, 0, 0.5668398769, 0.0956537644, 0.3375063587],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        table.loc["Minimum Variance"], table.loc["Corner 6"], atol=1e-12
    )


def test_critical_line_corners_lie_on_the_efficient_frontier():
    # With an upper bound of 0.3, PyPortfolioOpt's CLA agrees on the minimum variance
    # and maximum Sharpe portfolios but skips the corners where B leaves and re-enters
    # the portfolio. Every corner here is the SLSQP minimum variance portfolio at its
    # expected return.
    returns = _heavy_tailed_returns()
    mu, sigma = returns.mean() * 252, returns.cov() * 252
    table = optimization_model.get_critical_line_weights(mu, sigma, maximum_weight=0.3)
    np.testing.assert_allclose(
        table.loc["Minimum Variance", list("ABCDEF")],
        [0.0811420496, 0.0652610238, 0.3, 0.3, 0.1616023354, 0.0919945912],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        table.loc["Maximum Sharpe", list("ABCDEF")],
        [0.1, 0, 0, 0.3, 0.3, 0.3],
        atol=1e-9,
    )
    assert len(table) == 10  # noqa: PLR2004
    for label in table.index[1:-2]:
        weights = optimization_model.get_mean_variance_weights(
            mu,
            sigma,
            target_return=table.loc[label, "Expected Return"],
            maximum_weight=0.3,
        )
        np.testing.assert_allclose(weights, table.loc[label, list("ABCDEF")], atol=1e-6)
    # The returns fall and λ falls from corner to corner.
    corners = table.iloc[:-2]
    assert corners["Expected Return"].is_monotonic_decreasing
    assert corners["Lambda"].is_monotonic_decreasing

    # On the four asset example the maximum Sharpe portfolio is that of SLSQP.
    exact = optimization_model.get_critical_line_weights(
        EXPECTED_RETURNS, COVARIANCE, risk_free_rate=0.02
    )
    np.testing.assert_allclose(
        exact.loc["Maximum Sharpe", ASSETS],
        optimization_model.get_maximum_sharpe_weights(
            EXPECTED_RETURNS, COVARIANCE, 0.02
        ),
        atol=1e-6,
    )


def test_critical_line_refuses_a_singular_covariance():
    singular = pd.DataFrame(
        np.ones((2, 2)) * 0.04, index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="positive definite"):
        optimization_model.get_critical_line_weights(
            pd.Series([0.1, 0.2], index=["A", "B"]), singular
        )


def test_worst_case_mean_variance_matches_riskfolio_and_skfolio():
    returns = _heavy_tailed_returns()
    mu, sigma = returns.mean(), returns.cov()
    uncertain = sigma / len(returns)
    # Riskfolio-Lib 7.4.1's wc_optimization(Umu="box", Ucov=None) with the same box
    # half-widths; its utility has l w'Σw, which is λ/2 with λ = 2.5.
    np.testing.assert_allclose(
        optimization_model.get_worst_case_mean_variance_weights(mu, sigma, uncertain),
        [0, 0, 0, 0.6981680941, 0, 0.3018319059],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        optimization_model.get_worst_case_mean_variance_weights(
            mu, sigma, uncertain, "box", objective="utility"
        ),
        [0, 0, 0, 0.717850325, 0, 0.282149675],
        atol=1e-6,
    )
    # skfolio 1.12.7's MeanRisk with EmpiricalMuUncertaintySet(diagonal=False).
    np.testing.assert_allclose(
        optimization_model.get_worst_case_mean_variance_weights(
            mu, sigma, uncertain, "ellipsoid", objective="utility"
        ),
        [0, 0, 0, 0.5677059283, 0.1044260114, 0.3278680602],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        optimization_model.get_worst_case_mean_variance_weights(
            mu,
            sigma,
            uncertain,
            "ellipsoid",
            objective="minimum_variance",
            target_return=-0.000308155090171875,
        ),
        [0.0349189784, 0, 0.2954033788, 0.4240730939, 0.1121447145, 0.1334598333],
        atol=1e-5,
    )
    # With Σ_μ = Σ / T the ellipsoid lowers every Sharpe ratio by the same amount.
    np.testing.assert_allclose(
        optimization_model.get_worst_case_mean_variance_weights(
            mu, sigma, uncertain, "ellipsoid"
        ),
        optimization_model.get_critical_line_weights(mu, sigma).loc[
            "Maximum Sharpe", list("ABCDEF")
        ],
        atol=1e-8,
    )


def test_worst_case_return_and_short_positions():
    mu, sigma = EXPECTED_RETURNS, COVARIANCE
    uncertain = sigma / 60
    weights = pd.Series([0.5, -0.2, 0.4, 0.3], index=ASSETS)
    half_width = 1.959963985 * np.sqrt(np.diag(uncertain))
    assert np.isclose(
        optimization_model.get_worst_case_return(weights, mu, uncertain, "box"),
        weights @ mu - half_width @ weights.abs(),
    )
    assert np.isclose(
        optimization_model.get_worst_case_return(weights, mu, uncertain, "ellipsoid"),
        weights @ mu - np.sqrt(9.487729037) * np.sqrt(weights @ uncertain @ weights),
    )
    # A short position pays the box penalty on its absolute weight, so the worst-case
    # utility is never above that of the long-only optimum by more than the extra room.
    short = optimization_model.get_worst_case_mean_variance_weights(
        mu, sigma, uncertain, "box", objective="utility", minimum_weight=-0.3
    )
    assert np.isclose(short.sum(), 1)
    assert short.min() >= -0.3 - 1e-8
    utility = optimization_model.get_worst_case_return(
        short, mu, uncertain, "box"
    ) - 1.25 * (short @ sigma @ short)
    for candidate in [
        pd.Series([0.25] * 4, index=ASSETS),
        short.clip(lower=0) / short.clip(lower=0).sum(),
    ]:
        assert (
            utility
            >= optimization_model.get_worst_case_return(candidate, mu, uncertain, "box")
            - 1.25 * (candidate @ sigma @ candidate)
            - 1e-9
        )
    with pytest.raises(ValueError, match="uncertainty"):
        optimization_model.get_worst_case_mean_variance_weights(
            mu, sigma, uncertain, "cube"
        )
    with pytest.raises(ValueError, match="target return"):
        optimization_model.get_worst_case_mean_variance_weights(
            mu, sigma, uncertain, objective="minimum_variance"
        )
    with pytest.raises(ValueError, match="cannot be reached"):
        optimization_model.get_worst_case_mean_variance_weights(
            mu, sigma, uncertain, objective="minimum_variance", target_return=0.5
        )


@pytest.mark.parametrize(
    ("radius", "risk_aversion", "expected"),
    [
        # skfolio 1.12.7's DistributionallyRobustCVaR(cvar_beta=0.95).
        (
            0.0,
            1.0,
            [0.1486216398, 0, 0.1600879639, 0.4425478359, 0.0518559898, 0.1968865703],
        ),
        (
            0.0,
            5.0,
            [0.159984991, 0, 0.1907703424, 0.4722754639, 0.059826139, 0.1171430636],
        ),
        (
            0.001,
            1.0,
            [
                0.1844150639,
                0.0616383107,
                0.1884866563,
                0.1884866563,
                0.1884866563,
                0.1884866563,
            ],
        ),
        (
            0.001,
            5.0,
            [
                0.1887159401,
                0.0578295433,
                0.1964406636,
                0.1964406636,
                0.1964406634,
                0.164132526,
            ],
        ),
        (0.02, 1.0, [1 / 6] * 6),
    ],
)
def test_distributionally_robust_cvar_weights_match_skfolio(
    radius, risk_aversion, expected
):
    np.testing.assert_allclose(
        optimization_model.get_distributionally_robust_cvar_weights(
            _heavy_tailed_returns(), 0.05, radius, risk_aversion
        ),
        expected,
        atol=1e-7,
    )


def test_worst_case_cvar_grows_with_the_radius_up_to_a_total_loss():
    returns = _heavy_tailed_returns()
    weights = pd.Series([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], index=list("ABCDEF"))
    historical = optimization_model.get_expected_tail_loss(-(returns @ weights), 0.05)
    assert np.isclose(
        optimization_model.get_worst_case_cvar(returns, weights, 0.05, 0.0), historical
    )
    # For a small radius the worst case moves the asset with the largest weight down in
    # the tail periods: the CVaR rises by ε max w / (1 - β).
    assert np.isclose(
        optimization_model.get_worst_case_cvar(returns, weights, 0.05, 0.01),
        historical + 0.01 * 0.5 / 0.05,
    )
    # Returns cannot fall below -100%, so no distribution loses more than everything.
    assert np.isclose(
        optimization_model.get_worst_case_cvar(returns, weights, 0.05, 2.0), 1
    )
    with pytest.raises(ValueError, match="radius"):
        optimization_model.get_distributionally_robust_cvar_weights(returns, 0.05, -0.1)


def test_hierarchical_risk_parity_with_other_risk_measures_matches_riskfolio():
    returns = _block_returns()
    covariance = returns.cov() * 252

    # Riskfolio-Lib 7.4.1's HCPortfolio(model="HRP", rm=...) with its clustering replaced by
    # the paper's (single linkage on the distance between the columns of the correlation
    # distance, no optimal leaf order); skfolio 1.14.4's HierarchicalRiskParity agrees to
    # 1e-15 on the measures it has (with the lower partial moment below the mean).
    for measure, threshold, reference in (
        (
            "cvar",
            0.0,
            [0.1161672636, 0.1012285676, 0.0940929137, 0.0800697728, 0.0589711385]
            + [0.0771216812, 0.1274498611, 0.1547867123, 0.1422451844, 0.0478669049],
        ),
        (
            "cdar",
            0.0,
            [0.0531439788, 0.0795147121, 0.067648696, 0.0525658726, 0.0555266023]
            + [0.1325611325, 0.1357556776, 0.2477535403, 0.1448670112, 0.0306627766],
        ),
        (
            "semideviation",
            "mean",
            [0.1146711497, 0.1028620888, 0.0962579363, 0.0799957216, 0.0642459939]
            + [0.0796885237, 0.1249699125, 0.1503596939, 0.1378157114, 0.0491332682],
        ),
        (
            "mad",
            0.0,
            [0.1125908443, 0.104902673, 0.0958432631, 0.0799940541, 0.0671899363]
            + [0.0818589711, 0.1215005635, 0.1496912421, 0.137189869, 0.0492385835],
        ),
    ):
        weights = optimization_model.get_hierarchical_risk_parity_weights(
            covariance,
            returns=returns,
            risk_measure=measure,
            minimum_acceptable_return=threshold,
        )
        np.testing.assert_allclose(weights, reference, atol=1e-9)

    # The default is the variance, which needs no returns.
    pd.testing.assert_series_equal(
        optimization_model.get_hierarchical_risk_parity_weights(covariance),
        optimization_model.get_hierarchical_risk_parity_weights(
            covariance, returns=returns, risk_measure="variance"
        ),
    )
    with pytest.raises(ValueError, match="needs the returns"):
        optimization_model.get_hierarchical_risk_parity_weights(
            covariance, risk_measure="cvar"
        )
    with pytest.raises(InvalidArgumentError, match="risk_measure must be one of"):
        optimization_model.get_hierarchical_risk_parity_weights(
            covariance, returns=returns, risk_measure="evar"
        )


def test_hierarchical_equal_risk_contribution_with_other_risk_measures_and_herc2():
    returns = _block_returns()
    covariance = returns.cov() * 252

    # Riskfolio-Lib 7.4.1's HCPortfolio(model="HERC" and "HERC2", rm="TGRG", "UCI",
    # "CVaR", linkage="ward", k=3).
    for measure, within, reference in (
        (
            "tail_gini_range",
            "inverse_risk",
            [0.1815740159, 0.1654591656, 0.1539332793, 0.1804151798, 0.0445363039]
            + [0.0573186006, 0.052637495, 0.0667957452, 0.0616573699, 0.0356728448],
        ),
        (
            "ulcer_index",
            "inverse_risk",
            [0.1128946573, 0.1410532182, 0.1315905827, 0.1269351136, 0.0425877443]
            + [0.0974349796, 0.0767734283, 0.1539009066, 0.0868242067, 0.0300051628],
        ),
        (
            "cvar",
            "equal",
            [0.1710084832] * 4 + [0.0509455136] * 3 + [0.0543765089] * 3,
        ),
    ):
        weights = optimization_model.get_hierarchical_equal_risk_contribution_weights(
            covariance,
            clusters=3,
            returns=returns,
            risk_measure=measure,
            within_clusters=within,
        )
        np.testing.assert_allclose(weights, reference, atol=1e-9)

    with pytest.raises(InvalidArgumentError, match="within_clusters must be one of"):
        optimization_model.get_hierarchical_equal_risk_contribution_weights(
            covariance, within_clusters="minimum_variance"
        )


def test_schur_complementary_allocation_runs_from_hrp_and_matches_skfolio():
    covariance = _block_returns().cov() * 252

    # skfolio 1.14.4's SchurComplementary(gamma=0.5 and 1, keep_monotonic=False) on the
    # same asset order as Hierarchical Risk Parity (single linkage on the distance between
    # the columns of the correlation distance, no optimal leaf order).
    np.testing.assert_allclose(
        optimization_model.get_schur_complementary_weights(covariance, 0.5),
        [0.0911093606, 0.0968271732, 0.072868089, 0.0730544678, 0.073638219]
        + [0.0994546955, 0.1155508147, 0.185364875, 0.1603300792, 0.0318022258],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        optimization_model.get_schur_complementary_weights(covariance, 1),
        [0.0680562228, 0.144592316, 0.0905720389, 0.0398501771, 0.0778828908]
        + [0.0856543794, 0.1056420165, 0.1975842061, 0.1707637921, 0.0194019602],
        atol=1e-9,
    )
    pd.testing.assert_series_equal(
        optimization_model.get_schur_complementary_weights(covariance, 0),
        optimization_model.get_hierarchical_risk_parity_weights(covariance),
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        optimization_model.get_schur_complementary_weights(covariance, 1.5)

    # One factor drives these four assets; past a gamma of about 0.65 the augmented block
    # of a half is no longer positive definite.
    factor = pd.DataFrame(
        [
            [0.6991, 0.4507, 0.3956, 0.8257],
            [0.4507, 0.4822, 0.2698, 0.6075],
            [0.3956, 0.2698, 0.3163, 0.5151],
            [0.8257, 0.6075, 0.5151, 1.3291],
        ],
        index=ASSETS,
        columns=ASSETS,
    )
    path = [
        optimization_model.get_schur_complementary_weights(factor, gamma)
        for gamma in (0, 0.3, 0.6)
    ]
    variances = [weights @ factor @ weights for weights in path]
    assert variances[0] > variances[1] > variances[2]
    with pytest.raises(ValueError, match="not positive definite"):
        optimization_model.get_schur_complementary_weights(factor, 0.9)

    # skfolio's SchurComplementary(gamma=0.5, min_weights=0.05, max_weights=0.15).
    np.testing.assert_allclose(
        optimization_model.get_schur_complementary_weights(
            covariance, 0.5, minimum_weight=0.05, maximum_weight=0.15
        ),
        [0.0911093606, 0.0968271732, 0.072868089, 0.0548566937, 0.0852958257]
        + [0.1151992875, 0.1338435703, 0.15, 0.15, 0.05],
        atol=1e-9,
    )


def test_standardized_silhouette_matches_scikit_learn_and_finds_the_blocks():
    returns = _block_returns()
    covariance = returns.cov()
    distance = np.sqrt(np.clip((1 - returns.corr().to_numpy()) / 2, 0, 1))
    np.fill_diagonal(distance, 0)
    levels = cut_tree(linkage(squareform(distance, checks=False), "ward"))

    # scikit-learn's silhouette_samples on the rows of the distance matrix, mean over
    # standard deviation, as in López de Prado (2020) and Riskfolio-Lib's
    # std_silhouette_score.
    for clusters, reference in (
        (2, 1.9123244657),
        (3, 4.9654630069),
        (4, 2.5600878606),
    ):
        assert optimization_model.get_standardized_silhouette(
            distance, levels[:, 10 - clusters]
        ) == pytest.approx(reference, abs=1e-9)
    _, labels = optimization_model.get_hierarchical_clusters(
        covariance, cluster_criterion="standardized_silhouette"
    )
    assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    with pytest.raises(InvalidArgumentError, match="cluster_criterion must be one of"):
        optimization_model.get_hierarchical_clusters(
            covariance, cluster_criterion="elbow"
        )


def test_nested_clustered_optimization_with_other_risk_measures_matches_riskfolio():
    returns = _block_returns()
    expected_returns, covariance = returns.mean() * 252, returns.cov() * 252

    # Riskfolio-Lib 7.4.1's HCPortfolio(model="NCO", linkage="ward", k=3) with
    # obj="MinRisk" and rm="MAD" and "CVaR", and obj="ERC" and rm="CDaR" (CLARABEL).
    for measure, objective, reference in (
        (
            "mad",
            "minimum_variance",
            [0.09444029, 0.09168186, 0.06958896, 0.09217494, 0.07243633]
            + [0.13847335, 0.10206972, 0.21611476, 0.11053042, 0.01248937],
        ),
        (
            "cvar",
            "minimum_variance",
            [0.13700428, 0.10232395, 0.03583351, 0.06301302, 0.05068709]
            + [0.09974235, 0.12601089, 0.23831106, 0.13874519, 0.00832865],
        ),
        (
            "cdar",
            "risk_parity",
            [0.04393936, 0.06516633, 0.08036446, 0.06888044, 0.06769165]
            + [0.09088553, 0.14653803, 0.23043557, 0.12071156, 0.08538707],
        ),
    ):
        weights = optimization_model.get_nested_clustered_optimization_weights(
            expected_returns,
            covariance,
            objective,
            clusters=3,
            returns=returns,
            risk_measure=measure,
        )
        np.testing.assert_allclose(weights, reference, atol=1e-5)

    with pytest.raises(ValueError, match="variance only"):
        optimization_model.get_nested_clustered_optimization_weights(
            expected_returns,
            covariance,
            "maximum_sharpe",
            returns=returns,
            risk_measure="mad",
        )


def test_stacking_matches_skfolio():
    returns = _block_returns()

    def inverse_volatility(window: pd.DataFrame) -> pd.Series:
        return optimization_model.get_inverse_volatility_weights(window.cov())

    methods = {
        "Inverse Volatility": inverse_volatility,
        "Hierarchical Risk Parity": lambda window: (
            optimization_model.get_hierarchical_risk_parity_weights(window.cov())
        ),
        "Equal Weight": lambda window: pd.Series(
            1 / window.shape[1], index=window.columns
        ),
    }

    # skfolio 1.14.4's StackingOptimization of InverseVolatility, HierarchicalRiskParity
    # (on the same asset order) and EqualWeighted with an InverseVolatility final
    # estimator, cv=KFold(5) and cv="ignore".
    for folds, reference in (
        (
            5,
            [0.1007349654, 0.095329073, 0.0882301414, 0.100167522, 0.0827999411]
            + [0.1016128522, 0.1035304866, 0.1365747254, 0.1239793324, 0.0670409605],
        ),
        (
            None,
            [0.1007148391, 0.0952946832, 0.0881786924, 0.1001456146, 0.0827563179]
            + [0.1016214873, 0.1035703051, 0.1367047698, 0.1240668883, 0.0669464023],
        ),
    ):
        weights, method_weights, final = optimization_model.get_stacking_weights(
            returns, methods, inverse_volatility, folds
        )
        np.testing.assert_allclose(weights, reference, atol=1e-10)
        assert final.sum() == pytest.approx(1)
        pd.testing.assert_series_equal(
            method_weights["Equal Weight"],
            pd.Series(0.1, index=returns.columns),
            check_names=False,
        )

    with pytest.raises(ValueError, match="at least two methods"):
        optimization_model.get_stacking_weights(
            returns, {"Equal Weight": methods["Equal Weight"]}, inverse_volatility
        )
    with pytest.raises(ValueError, match="folds"):
        optimization_model.get_stacking_weights(
            returns, methods, inverse_volatility, folds=1
        )


def _dbht_returns() -> pd.DataFrame:
    """
    Fifteen assets in three blocks with their own factors, which the DBHT finds as its
    clusters. The reference values below come from Riskfolio-Lib 7.4.1 run on these
    same returns.
    """
    rng = np.random.default_rng(0)
    block = np.arange(15) % 3
    factors = rng.normal(0, 0.01, size=(500, 3))
    data = factors[:, block] * rng.uniform(0.6, 1.4, 15) + rng.normal(
        0, 0.008, size=(500, 15)
    )
    return pd.DataFrame(data, columns=[f"D{i:02d}" for i in range(15)])


def test_the_dbht_matches_riskfolio():
    returns = _dbht_returns()
    covariance = returns.cov()
    distance = optimization_model.get_codependence_distance(returns, covariance)
    similarity = optimization_model.get_codependence_similarity(returns, covariance)
    assert np.allclose(similarity, (1 + returns.corr()) / 2)

    # Riskfolio-Lib's DBHTs(D, S) with its default optimal leaf ordering: the same
    # merges, heights and order.
    matrix = optimization_model.get_dbht_linkage(distance, similarity)
    assert np.allclose(
        matrix,
        [
            [10, 1, 0.25, 2],
            [4, 13, 1 / 3, 2],
            [16, 15, 0.5, 4],
            [17, 7, 1.0, 5],
            [8, 2, 0.25, 2],
            [11, 19, 1 / 3, 3],
            [20, 5, 0.5, 4],
            [14, 21, 1.0, 5],
            [3, 6, 0.25, 2],
            [23, 9, 1 / 3, 3],
            [0, 12, 0.5, 2],
            [25, 24, 1.0, 5],
            [22, 18, 2.0, 10],
            [26, 27, 3.0, 15],
        ],
        atol=1e-12,
    )
    # Without a number of clusters, the DBHT's own: the three blocks.
    _, labels = optimization_model.get_hierarchical_clusters(
        covariance, "dbht", distance=distance, similarity=similarity
    )
    assert labels.tolist() == [0, 2, 1] * 5
    # The defaults derive the same distance and similarity from the covariance.
    assert (
        optimization_model.get_hierarchical_clusters(covariance, "dbht")[1].tolist()
        == labels.tolist()
    )

    # Riskfolio-Lib's HCPortfolio(linkage="DBHT", rm="MV", k=3) with model="HRP",
    # "HERC", "HERC2" and "NCO" (obj="MinRisk", solved to about 1e-5 by its solver).
    options = {"distance": distance, "similarity": similarity}
    references = {
        "HRP": optimization_model.get_hierarchical_risk_parity_weights(
            covariance, "dbht", **options
        ),
        "HERC": optimization_model.get_hierarchical_equal_risk_contribution_weights(
            covariance, "dbht", **options
        ),
        "HERC2": optimization_model.get_hierarchical_equal_risk_contribution_weights(
            covariance, "dbht", within_clusters="equal", **options
        ),
        "NCO": optimization_model.get_nested_clustered_optimization_weights(
            returns.mean(), covariance, linkage_method="dbht", **options
        ),
    }
    expected = {
        "HRP": [
            0.0941047028,
            0.0405907212,
            0.0672417116,
            0.053457103,
            0.0973872823,
            0.0990891829,
            0.0542886766,
            0.0609920351,
            0.0329178319,
            0.0843162113,
            0.0303144049,
            0.0959843797,
            0.0585235509,
            0.0268380624,
            0.1039541434,
        ],
        "HERC": [
            0.1613824692,
            0.0236745208,
            0.0381735516,
            0.1384618199,
            0.0309904582,
            0.0355037149,
            0.0960519566,
            0.0355735783,
            0.0186876646,
            0.1491791212,
            0.019705124,
            0.0401270554,
            0.1515846708,
            0.0174454141,
            0.0434588804,
        ],
        "HERC2": [0.1393320075, 0.0254778191, 0.0351901734] * 5,
        "NCO": [
            0.1170371688,
            0.0237126176,
            0.0850734153,
            0.0515686364,
            0.1128046965,
            0.0680497402,
            0.0,
            0.1545980627,
            0.0,
            0.0812008486,
            0.0,
            0.0922174547,
            0.0860588972,
            0.0,
            0.127678411,
        ],
    }
    for model, weights in references.items():
        tolerance = 2e-5 if model == "NCO" else 1e-9
        assert np.allclose(weights, expected[model], atol=tolerance), model


def test_the_dbht_breaks_ties_and_handles_few_assets():
    # Three blocks of exactly equal correlations: every choice is a tie. Riskfolio-Lib's
    # DBHTs gives this tree once its FindDisjoint leaves out all three vertices of the
    # triangle; as published it never separates on the triangle of assets 0, 1 and 2 and
    # finds two clusters.
    correlation = np.full((15, 15), 0.1)
    for start in (0, 5, 10):
        correlation[start : start + 5, start : start + 5] = 0.6
    np.fill_diagonal(correlation, 1.0)
    distance = np.sqrt((1 - correlation) / 2)
    matrix = optimization_model.get_dbht_linkage(distance)
    assert np.allclose(
        matrix,
        [
            [14, 13, 0.25, 2],
            [12, 10, 1 / 3, 2],
            [11, 15, 0.5, 3],
            [17, 16, 1.0, 5],
            [1, 0, 0.25, 2],
            [2, 19, 1 / 3, 3],
            [4, 3, 0.5, 2],
            [20, 21, 1.0, 5],
            [6, 5, 0.25, 2],
            [7, 23, 1 / 3, 3],
            [9, 8, 0.5, 2],
            [24, 25, 1.0, 5],
            [22, 18, 2.0, 10],
            [26, 27, 3.0, 15],
        ],
    )
    names = [f"A{i}" for i in range(15)]
    _, labels = optimization_model.get_hierarchical_clusters(
        pd.DataFrame(correlation, index=names, columns=names), "dbht"
    )
    assert labels.tolist() == [1] * 5 + [0] * 5 + [2] * 5

    # Up to four assets form one bubble and one cluster.
    for assets in (2, 3, 4):
        small = optimization_model.get_dbht_linkage(distance[:assets, :assets])
        assert len(small) == assets - 1 and (small[:, 2] <= 1).all()
        assert small[-1, 3] == assets
    with pytest.raises(ValueError, match="At least two assets"):
        optimization_model.get_dbht_linkage(np.zeros((1, 1)))


def _benchmark_inputs() -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal([0.0009, 0.0006, 0.0003, 0.0007], 0.015, (600, 4)),
        columns=["A", "B", "C", "Benchmark"],
    )
    returns["Benchmark"] += returns[["A", "B", "C"]].mean(axis=1)
    return returns.mean() * 252, returns.cov() * 252


def test_the_information_ratio_counts_the_part_not_invested_at_the_risk_free_rate():
    expected, covariance = _benchmark_inputs()
    weights = optimization_model.get_minimum_tracking_error_weights(
        expected,
        covariance,
        "Benchmark",
        "information_ratio",
        net_exposure=2.0,
        risk_free_rate=0.03,
    )
    statistics = optimization_model.get_tracking_error(
        weights, expected, covariance, "Benchmark", 0.03
    )

    assert weights.sum() == pytest.approx(2)
    assert statistics["Active Return"] == pytest.approx(
        weights @ expected[weights.index] - 0.03 - expected["Benchmark"]
    )
    # No other portfolio within the bounds has a higher information ratio than the one
    # reported for the optimizer's weights.
    rng = np.random.default_rng(0)
    draws = 2 * rng.dirichlet(np.ones(3), 5000)
    best = max(
        optimization_model.get_tracking_error(
            pd.Series(draw, index=weights.index),
            expected,
            covariance,
            "Benchmark",
            0.03,
        )["Information Ratio"]
        for draw in draws[draws.max(axis=1) <= 1]
    )
    assert statistics["Information Ratio"] >= best - 1e-8


def test_the_portfolio_statistics_hold_the_rest_in_cash_at_the_risk_free_rate():
    expected, covariance = _benchmark_inputs()
    expected, covariance = expected[:3], covariance.iloc[:3, :3]
    full = optimization_model.get_maximum_sharpe_weights(
        expected, covariance, 0.03, maximum_weight=2.0
    )
    half = optimization_model.get_maximum_sharpe_weights(
        expected, covariance, 0.03, maximum_weight=2.0, net_exposure=0.5
    )
    statistics = optimization_model.get_portfolio_statistics(
        half, expected, covariance, 0.03
    )

    np.testing.assert_allclose(half, 0.5 * full, atol=1e-6)
    assert statistics["Expected Return"] == pytest.approx(half @ expected + 0.5 * 0.03)
    assert statistics["Sharpe Ratio"] == pytest.approx(
        optimization_model.get_portfolio_statistics(full, expected, covariance, 0.03)[
            "Sharpe Ratio"
        ],
        rel=1e-6,
    )


def test_the_worst_case_sharpe_ratio_holds_the_rest_in_cash_at_the_risk_free_rate():
    expected, covariance = _benchmark_inputs()
    expected, covariance = expected[:3], covariance.iloc[:3, :3]
    weights = {
        net_exposure: optimization_model.get_worst_case_mean_variance_weights(
            expected,
            covariance,
            covariance / 600,
            risk_free_rate=0.03,
            maximum_weight=2.0,
            net_exposure=net_exposure,
        )
        for net_exposure in (1.0, 0.5)
    }

    # The box's worst case is homogeneous, so half the net exposure halves the weights.
    np.testing.assert_allclose(weights[0.5], 0.5 * weights[1.0], atol=1e-5)


def _estimator_returns(
    seed: int = 7, periods: int = 500, sizes=(4, 4, 4)
) -> pd.DataFrame:
    # Returns with a common factor per block of assets and fat-tailed noise.
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    factors = rng.standard_normal((periods, len(sizes)))
    noise = rng.standard_t(5, (periods, labels.size))
    return pd.DataFrame(
        0.01 * (0.8 * factors[:, labels] + 0.6 * noise),
        columns=[f"A{number}" for number in range(labels.size)],
    )


def test_onc_finds_the_blocks_for_every_seed():
    returns = _estimator_returns(sizes=(3, 5, 7))
    distance = optimization_model.get_codependence_distance(returns, returns.cov())
    expected = np.repeat([0, 1, 2], [3, 5, 7])
    for seed in range(5):
        labels = estimation_model.get_optimal_number_of_clusters(distance, seed=seed)
        np.testing.assert_array_equal(labels, expected)
    # Shuffled assets keep their clusters, numbered in the order of their first asset.
    order = np.random.default_rng(1).permutation(15)
    shuffled = distance.iloc[order, order]
    labels = estimation_model.get_optimal_number_of_clusters(shuffled)
    assert len(np.unique(labels)) == 3
    assert labels[0] == 0
    for label in range(3):
        assert len(set(expected[order][labels == label])) == 1


def test_onc_handles_few_assets_and_bad_input():
    returns = _estimator_returns(sizes=(1, 1))
    distance = optimization_model.get_codependence_distance(returns, returns.cov())
    # Two assets leave no number of clusters between 2 and N - 1 to choose from.
    np.testing.assert_array_equal(
        estimation_model.get_optimal_number_of_clusters(distance), [0, 0]
    )
    with pytest.raises(ValueError, match="square"):
        estimation_model.get_optimal_number_of_clusters(distance.iloc[:, :1])
    with pytest.raises(ValueError, match="at least 2"):
        estimation_model.get_optimal_number_of_clusters(distance, maximum_clusters=1)


def test_onc_redoes_the_clusters_of_below_average_quality():
    # Two tight blocks and a loose one that holds two sub-blocks: the clusters of
    # below-average quality are clustered again and split when that raises the quality.
    rng = np.random.default_rng(3)
    periods = 600
    tight = rng.standard_normal((periods, 2))
    loose = rng.standard_normal((periods, 1))
    sub = rng.standard_normal((periods, 2))
    columns = [tight[:, [0]]] * 4 + [tight[:, [1]]] * 4
    returns = np.hstack(
        [0.9 * column + 0.3 * rng.standard_normal((periods, 1)) for column in columns]
        + [
            0.5 * loose
            + 0.6 * sub[:, [k // 3]]
            + 0.4 * rng.standard_normal((periods, 1))
            for k in range(6)
        ]
    )
    frame = pd.DataFrame(returns)
    distance = optimization_model.get_codependence_distance(frame, frame.cov())
    labels = estimation_model.get_optimal_number_of_clusters(distance)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:8])) == 1
    assert labels[0] != labels[4]
    assert set(labels[:8]).isdisjoint(labels[8:])


def test_the_silhouette_coefficients_match_the_definition():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((7, 3))
    labels = np.array([0, 0, 1, 1, 1, 2, 0])
    euclidean = squareform(pdist(points))
    coefficients = estimation_model.get_silhouette_coefficients(euclidean, labels)
    # A single-asset cluster has a coefficient of 0.
    assert coefficients[5] == 0
    for asset in (0, 1, 2, 3, 4, 6):
        same = (labels == labels[asset]) & (np.arange(7) != asset)
        within = euclidean[asset, same].mean()
        between = min(
            euclidean[asset, labels == other].mean()
            for other in set(labels) - {labels[asset]}
        )
        assert coefficients[asset] == pytest.approx(
            (between - within) / max(between, within)
        )


def test_the_angular_distances_follow_the_correlation():
    returns = _estimator_returns(sizes=(2, 2))
    returns["A3"] = -returns["A0"] + 0.002 * np.random.default_rng(2).standard_normal(
        500
    )
    correlation = returns.corr().to_numpy()
    absolute = optimization_model.get_codependence_distance(
        returns, returns.cov(), "absolute_angular"
    )
    squared = optimization_model.get_codependence_distance(
        returns, returns.cov(), "squared_angular"
    )
    expected = np.sqrt(np.clip(1 - np.abs(correlation), 0, None))
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(absolute, expected, atol=1e-12)
    expected = np.sqrt(np.clip(1 - correlation**2, 0, None))
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(squared, expected, atol=1e-12)
    # A hedge is as close as a twin.
    assert absolute.loc["A0", "A3"] < 0.15 < absolute.loc["A0", "A2"]
    similarity = optimization_model.get_codependence_similarity(
        returns, returns.cov(), "absolute_angular"
    )
    np.testing.assert_allclose(similarity, 1 - absolute**2, atol=1e-12)


def test_the_gnpr_distance_mixes_ranks_and_distributions():
    returns = _estimator_returns(sizes=(2, 1))
    # Without ties the rank term is (1 - Spearman) / 2.
    spearman = returns.corr("spearman").to_numpy()
    copula = estimation_model.get_gnpr_distance_matrix(returns, theta=1.0)
    expected = np.sqrt(np.clip((1 - spearman) / 2, 0, None))
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(copula, expected, atol=1e-12)
    # The distribution term is the Hellinger distance of the histograms, by hand.
    first, second = returns["A0"].to_numpy(), returns["A2"].to_numpy()
    low, high = min(first.min(), second.min()), max(first.max(), second.max())
    edges = np.arange(low, high + 0.005, 0.005)
    hellinger = 0.5 * np.sum(
        (
            np.sqrt(np.histogram(first, edges)[0] / 500)
            - np.sqrt(np.histogram(second, edges)[0] / 500)
        )
        ** 2
    )
    marginal = estimation_model.get_gnpr_distance_matrix(returns, 0.0, 0.005)
    assert marginal.loc["A0", "A2"] == pytest.approx(np.sqrt(hellinger))
    mixed = estimation_model.get_gnpr_distance_matrix(returns, 0.3, 0.005)
    np.testing.assert_allclose(
        mixed**2, 0.3 * copula**2 + 0.7 * marginal**2, atol=1e-12
    )
    assert (mixed.to_numpy() <= 1).all()
    # The same returns scaled up differ only in their distribution.
    scaled = returns.assign(A1=returns["A0"] * 3)
    assert estimation_model.get_gnpr_distance_matrix(scaled, 1.0).loc[
        "A0", "A1"
    ] == pytest.approx(0, abs=1e-12)
    assert estimation_model.get_gnpr_distance_matrix(scaled, 0.0).loc["A0", "A1"] > 0.3
    with pytest.raises(ValueError, match="theta"):
        estimation_model.get_gnpr_distance_matrix(returns, 1.5)


def test_the_nercome_covariance_matches_its_definition():
    returns = _estimator_returns(periods=300, sizes=(2, 2))
    values = returns.to_numpy()
    estimate = estimation_model.get_nercome_covariance(returns, splits=4, seed=3)
    # By hand: the same random orders, every candidate split, the one with the smallest
    # Frobenius norm of the averaged difference with the second part's covariance.
    rng = np.random.default_rng(3)
    orders = [rng.permutation(300) for _ in range(4)]
    best, best_error = None, np.inf
    for size in sorted(
        {
            round(x)
            for x in (
                2 * np.sqrt(300),
                60,
                120,
                180,
                240,
                300 - 2.5 * np.sqrt(300),
                300 - 1.5 * np.sqrt(300),
            )
        }
    ):
        total, difference = 0, 0
        for order in orders:
            second = np.cov(values[order[size:]], rowvar=False)
            vectors = np.linalg.eigh(np.cov(values[order[:size]], rowvar=False))[1]
            variances = np.diag(vectors.T @ second @ vectors)
            total = total + vectors @ np.diag(variances) @ vectors.T / 4
            difference = (
                difference + (vectors @ np.diag(variances) @ vectors.T - second) / 4
            )
        if np.sum(difference**2) < best_error:
            best, best_error = total, np.sum(difference**2)
    np.testing.assert_allclose(estimate, best, rtol=1e-10)
    assert np.linalg.eigvalsh(estimate).min() > 0
    assert estimation_model.get_nercome_covariance(returns).equals(
        estimation_model.get_nercome_covariance(returns)
    )
    with pytest.raises(ValueError, match="more periods than assets"):
        estimation_model.get_nercome_covariance(returns.iloc[:6])


def test_the_ccc_garch_covariance_combines_the_garch_volatilities():
    from financetoolkit.risk.volatility import garch_model

    returns = _estimator_returns(periods=400, sizes=(2,))
    covariance, scenarios = estimation_model.get_ccc_garch_covariance(returns)
    residuals = (returns - returns.mean()).to_numpy()
    standardized, volatility = [], []
    for column in range(2):
        weights = garch_model.get_garch_weights(residuals[:, column])
        variance = garch_model.get_garch(residuals[:, column], weights)
        standardized.append(residuals[:, column] / np.sqrt(variance))
        volatility.append(
            np.sqrt(
                weights[0]
                + weights[1] * residuals[-1, column] ** 2
                + weights[2] * variance[-1]
            )
        )
    expected = np.corrcoef(standardized) * np.outer(volatility, volatility)
    np.testing.assert_allclose(covariance, expected, rtol=1e-10)
    # The scenarios carry the next-period covariance.
    np.testing.assert_allclose(scenarios.cov(), covariance, rtol=1e-10)
    with pytest.raises(ValueError, match="100 periods"):
        estimation_model.get_ccc_garch_covariance(returns.iloc[:50])


def test_the_bisquare_constants_match_statsmodels():
    # statsmodels' table of biweight constants for 95% shape efficiency (Lopuhaä, 1989),
    # 1 to 5 and 14 variables.
    for assets, constant in (
        (1, 5.49025),
        (2, 5.81032),
        (3, 6.09627),
        (4, 6.35622),
        (5, 6.59558),
        (14, 8.22785),
    ):
        assert estimation_model.get_bisquare_constants(assets, 0.5, 0.95)[
            1
        ] == pytest.approx(constant, abs=5e-5)
    # The S constant makes E[ρ(√X)] = δ for X ~ χ²_N, checked by quadrature.
    from scipy import integrate, stats

    for assets, delta in ((2, 0.5), (4, 0.45), (10, 0.3)):
        constant = estimation_model.get_bisquare_constants(assets, delta, 0.95)[0]
        average = integrate.quad(
            lambda x: min(1 - (1 - x / constant**2) ** 3, 1)
            * stats.chi2.pdf(x, assets),
            0,
            constant**2,
        )[0] + stats.chi2.sf(constant**2, assets)
        assert average == pytest.approx(delta, abs=1e-9)


def _contaminated_returns():
    rng = np.random.default_rng(9)
    covariance = (
        np.array(
            [
                [1, 0.6, 0.3, 0.2],
                [0.6, 1, 0.4, 0.1],
                [0.3, 0.4, 1, 0.5],
                [0.2, 0.1, 0.5, 1],
            ]
        )
        * 1e-4
    )
    values = rng.multivariate_normal(np.zeros(4), covariance, 500)
    values[:25] = rng.multivariate_normal(np.full(4, 0.05), covariance * 25, 25)
    return pd.DataFrame(values, columns=ASSETS), covariance


@pytest.mark.parametrize("method", ["mm", "rocke"])
def test_the_robust_covariances_withstand_outliers(method):
    returns, covariance = _contaminated_returns()

    def error(matrix) -> float:
        return np.linalg.norm(np.asarray(matrix) - covariance) / np.linalg.norm(
            covariance
        )

    location, robust = estimation_model.get_robust_covariance(returns, method)
    # Five percent of the periods far away inflate the sample covariance several times
    # over; the robust estimates stay close to the covariance of the bulk.
    assert error(returns.cov()) > 3
    assert error(robust) < 0.2
    assert np.abs(location.to_numpy()).max() < 0.002
    # On the clean periods the MM-estimator is close to the sample covariance.
    clean = returns.iloc[25:]
    if method == "mm":
        np.testing.assert_allclose(
            estimation_model.get_robust_covariance(clean, method)[1],
            clean.cov(),
            rtol=0.1,
            atol=2e-6,
        )


@pytest.mark.parametrize("method", ["mm", "rocke"])
def test_the_robust_covariances_are_fixed_points(method):
    # The iterations run to the fixed point, so returns perturbed by 1e-12 give the same
    # estimate to about 1e-10, and the same seed the same estimate.
    returns, _ = _contaminated_returns()
    noise = np.random.default_rng(11).standard_normal(returns.shape) * 1e-12
    first = estimation_model.get_robust_covariance(returns, method)[1]
    second = estimation_model.get_robust_covariance(returns + noise, method)[1]
    np.testing.assert_allclose(first, second, rtol=1e-8)
    assert first.equals(estimation_model.get_robust_covariance(returns, method)[1])
    with pytest.raises(
        InvalidArgumentError, match="method must be one of 'mm' or 'rocke'"
    ):
        estimation_model.get_robust_covariance(returns, "tyler")
    with pytest.raises(ValueError, match="twice as many periods"):
        estimation_model.get_robust_covariance(returns.iloc[:8], method)


def test_the_centroid_matches_the_order_statistics():
    from scipy import stats

    returns = _estimator_returns(sizes=(5,))
    ranking = ["A3", "A0", "A4", "A1", "A2"]
    expected = estimation_model.get_centroid_expected_returns(returns, ranking, 252)
    alpha = 0.4424 - 0.1185 * 5 ** (-0.21)
    centroid = stats.norm.ppf((5 + 1 - np.arange(1, 6) - alpha) / (5 - 2 * alpha + 1))
    volatility = returns.std() * np.sqrt(252)
    np.testing.assert_allclose(
        expected[ranking], centroid * volatility[ranking].to_numpy(), rtol=1e-12
    )
    # The approximation is close to the Monte Carlo mean of sorted normal draws.
    draws = -np.sort(-np.random.default_rng(0).standard_normal((200000, 5)), axis=1)
    np.testing.assert_allclose(centroid, draws.mean(axis=0), atol=0.01)
    # Assets ranked together share the average of their positions; scores rank alike.
    buckets = estimation_model.get_centroid_expected_returns(
        returns, ["A3", ["A0", "A4"], "A1", "A2"], 252
    )
    assert buckets["A0"] / volatility["A0"] == pytest.approx(centroid[1:3].mean())
    scores = estimation_model.get_centroid_expected_returns(
        returns, {"A3": 5, "A0": 4, "A4": 4, "A1": 2, "A2": 1}, 252
    )
    pd.testing.assert_series_equal(scores, buckets)
    with pytest.raises(ValueError, match="every asset once"):
        estimation_model.get_centroid_expected_returns(returns, ranking[:4], 252)
    with pytest.raises(ValueError, match="ranking"):
        optimization_model.get_expected_returns(returns, 252, "centroid")


def test_the_ranking_views_order_the_posterior_means():
    returns = _estimator_returns(sizes=(3,))
    means = returns.mean()
    ranking = list(means.sort_values().index)  # from the lowest to the highest
    probabilities = estimation_model.get_entropy_pooling(returns, ranking=ranking)
    posterior = probabilities @ returns
    assert all(
        posterior[higher] >= posterior[lower] - 1e-10
        for higher, lower in itertools.pairwise(ranking)
    )
    # The closest probabilities in relative entropy, checked with SLSQP on the primal.
    count = len(returns)
    data = returns.to_numpy()
    order = [returns.columns.get_loc(name) for name in ranking]
    solution = minimize(
        lambda p: np.sum(p * np.log(np.maximum(p, 1e-300) * count)),
        np.full(count, 1 / count),
        method="SLSQP",
        bounds=[(0, 1)] * count,
        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}]
        + [
            {"type": "ineq", "fun": lambda p, i=i, j=j: p @ (data[:, i] - data[:, j])}
            for i, j in itertools.pairwise(order)
        ],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    np.testing.assert_allclose(probabilities, solution.x, atol=1e-6)
    # A ranking the history already follows leaves the probabilities equal.
    np.testing.assert_allclose(
        estimation_model.get_entropy_pooling(returns, ranking=ranking[::-1]),
        1 / count,
    )
    with pytest.raises(ValueError, match="at least two different assets"):
        estimation_model.get_entropy_pooling(returns, ranking=["A0"])


def test_the_denoised_detoned_covariance_detones_the_denoised_correlation():
    returns = _estimator_returns(sizes=(3, 3))
    denoised = optimization_model.get_denoised_covariance(returns)[0]
    detoned = estimation_model.get_detoned_covariance(returns, 1, denoised)
    volatility = np.sqrt(np.diag(denoised))
    correlation = denoised.to_numpy() / np.outer(volatility, volatility)
    values, vectors = np.linalg.eigh(correlation)
    expected = correlation - values[-1] * np.outer(vectors[:, -1], vectors[:, -1])
    scale = np.sqrt(np.diag(expected))
    expected = expected / np.outer(scale, scale) * np.outer(volatility, volatility)
    np.testing.assert_allclose(detoned, expected, atol=1e-14)
    np.testing.assert_allclose(np.diag(detoned), returns.var(), rtol=1e-12)


def test_the_marcenko_pastur_fit_of_random_returns():
    # Unrelated returns: every eigenvalue lies near the Marčenko-Pastur bulk with σ² ≈ 1.
    returns = np.random.default_rng(4).standard_normal((2000, 100))
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(returns, rowvar=False))
    variance, lower, upper = optimization_model.get_marcenko_pastur_fit(eigenvalues, 20)
    assert variance == pytest.approx(1, abs=0.1)
    assert upper == pytest.approx(variance * (1 + np.sqrt(1 / 20)) ** 2)
    assert lower == pytest.approx(variance * (1 - np.sqrt(1 / 20)) ** 2)
    assert (eigenvalues < upper * 1.1).all()


def test_onc_clusters_nested_clustered_optimization_but_not_herc():
    returns = _estimator_returns(sizes=(3, 3))
    covariance = returns.cov() * 252
    weights = optimization_model.get_nested_clustered_optimization_weights(
        returns.mean() * 252, covariance, cluster_criterion="onc"
    )
    assert weights.sum() == pytest.approx(1)
    _, labels = optimization_model.get_hierarchical_clusters(
        covariance, cluster_criterion="onc"
    )
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="ONC"):
        optimization_model.get_hierarchical_equal_risk_contribution_weights(
            covariance, cluster_criterion="onc"
        )
