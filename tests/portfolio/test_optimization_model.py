"""Portfolio Optimization Model Tests"""

import numpy as np
import pandas as pd
import pytest

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
