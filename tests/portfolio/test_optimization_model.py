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
    )["Black-Litterman Return"]
    implied = optimization_model.get_black_litterman_returns(covariance, weights)[
        "Implied Return"
    ]

    # The view that BRK-B beats C pulls their gap towards 2%.
    assert (relative["BRK-B"] - relative["C"]) > (implied["BRK-B"] - implied["C"])

    duplicate = pd.DataFrame(0.04, index=["X", "Y"], columns=["X", "Y"])
    posterior = optimization_model.get_black_litterman_returns(
        duplicate, pd.Series({"X": 0.5, "Y": 0.5}), views={"X": 0.1}
    )["Black-Litterman Return"]

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
            SHRINKAGE_RETURNS, 252, "shrunk"
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
    np.testing.assert_allclose(
        tracking,
        [0.1497636622, 0.1502363378, 0.2233589641, 0.2266410359, 0.25],
        atol=1e-7,
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
        estimate("james_stein"), ((1 - phi) * average + phi * target) * 252
    )
    with pytest.raises(ValueError, match="more periods"):
        optimization_model.get_expected_returns(returns.iloc[:7], 252, "james_stein")


def test_james_stein_leaves_equal_averages_and_shrinks_dispersed_ones():
    returns, _ = _reference_returns()
    equal = returns - returns.mean() + 0.0004
    assert np.allclose(
        optimization_model.get_expected_returns(equal, 252, "james_stein"), 0.0004 * 252
    )

    shrunk = optimization_model.get_expected_returns(returns, 252, "james_stein")
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
        weights, returns, 0.95
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
    cdar = optimization_model.get_expected_tail_loss(drawdowns, 0.95)

    # Every other portfolio, e.g. equal weights, has a higher CDaR.
    equal = optimization_model.get_drawdowns(returns.mean(axis=1))
    assert cdar < optimization_model.get_expected_tail_loss(equal, 0.95)
    # 500 periods at 95%: the average of exactly the 25 largest drawdowns.
    assert cdar == pytest.approx(drawdowns.nlargest(25).mean())


def test_expected_tail_loss_counts_a_fraction_of_the_boundary_loss():
    losses = np.arange(1, 11, dtype=float)

    # The worst 15% of ten losses: 10 in full and half of 9.
    assert optimization_model.get_expected_tail_loss(losses, 0.85) == pytest.approx(
        (10 + 0.5 * 9) / 1.5
    )
    assert optimization_model.get_expected_tail_loss(losses, 0.8) == pytest.approx(9.5)


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
        optimization_model.get_maximum_cvar_ratio_weights(returns, 0.95, 1e-4),
        [0.1160064356, 0.1761774468, 0.2405297615, 0, 0.4672863516],
        atol=1e-6,
    )


def test_ratio_optimizers_need_a_return_above_the_threshold_and_take_constraints():
    returns, _ = _reference_returns()
    with pytest.raises(ValueError, match="minimum acceptable return"):
        optimization_model.get_maximum_omega_weights(returns, 0.01)
    with pytest.raises(ValueError, match="risk free rate"):
        optimization_model.get_maximum_cvar_ratio_weights(returns, 0.95, 0.01)

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
    for confidence_level, reference in (
        (0.955, [0.0071414198, 0.006284051, 0.0043148177, 0.0035652219, 0.0007657886]),
        (0.973, [0.0079713688, 0.0068635433, 0.0046414049, 0.0035747537, 0.0009117271]),
    ):
        contributions = optimization_model.get_expected_shortfall_contributions(
            weights, returns, confidence_level
        )
        np.testing.assert_allclose(
            contributions["Risk Contribution"], reference, atol=1e-9
        )
        assert contributions["Risk Contribution"].sum() == pytest.approx(
            optimization_model.get_expected_tail_loss(
                -(returns @ weights), confidence_level
            )
        )


def test_tail_probabilities_sum_to_one_with_a_fraction_on_the_boundary():
    losses = np.array([3.0, 10.0, 1.0, 9.0, 5.0])
    probabilities = optimization_model.get_tail_probabilities(losses, 0.5)

    # 2.5 periods: 10 and 9 in full and half of 5.
    np.testing.assert_allclose(probabilities, [0, 0.4, 0, 0.4, 0.2])
    assert probabilities.sum() == pytest.approx(1)
    # A tail shorter than a period is the largest loss.
    assert optimization_model.get_expected_tail_loss(losses, 0.99) == pytest.approx(10)
    with pytest.raises(ValueError, match="confidence level"):
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
