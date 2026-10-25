"""Portfolio Construction Controller Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.portfolio.construction_controller import PortfolioConstruction

# pylint: disable=missing-function-docstring,redefined-outer-name


@pytest.fixture(scope="module")
def portfolio_module(test_toolkit):
    return test_toolkit.portfolio


def test_get_minimum_variance_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_minimum_variance_weights())
    recorder.capture(
        portfolio_module.get_minimum_variance_weights(
            period="weekly", include_benchmark=True
        )
    )


def test_get_maximum_sharpe_weights(recorder, portfolio_module):
    recorder.capture(
        portfolio_module.get_maximum_sharpe_weights(include_benchmark=True)
    )
    recorder.capture(
        portfolio_module.get_maximum_sharpe_weights(
            include_benchmark=True, maximum_weight=0.5
        )
    )


def test_get_mean_variance_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_mean_variance_weights(include_benchmark=True))
    recorder.capture(
        portfolio_module.get_mean_variance_weights(
            target_return=0.25, include_benchmark=True
        )
    )


def test_get_mean_variance_weights_unreachable_target(portfolio_module):
    assert portfolio_module.get_mean_variance_weights(target_return=5.0).empty


def test_get_efficient_frontier(recorder, portfolio_module):
    frontier = portfolio_module.get_efficient_frontier(points=5, include_benchmark=True)
    recorder.capture(frontier)
    # Every portfolio is fully invested and the returns rise along the frontier.
    assert np.allclose(
        frontier[["AAPL", "MSFT", "Benchmark"]].sum(axis=1), 1, atol=1e-3
    )
    assert frontier["Expected Return"].is_monotonic_increasing


def test_get_risk_parity_weights(recorder, portfolio_module):
    weights = portfolio_module.get_risk_parity_weights(include_benchmark=True)
    recorder.capture(weights)
    assert np.allclose(weights["Risk Contribution"].iloc[:-1], 1 / 3, atol=1e-3)
    recorder.capture(
        portfolio_module.get_risk_parity_weights(
            risk_budget={"AAPL": 0.5, "MSFT": 0.3, "Benchmark": 0.2},
            include_benchmark=True,
        )
    )


def test_get_hierarchical_risk_parity_weights(recorder, portfolio_module):
    recorder.capture(
        portfolio_module.get_hierarchical_risk_parity_weights(include_benchmark=True)
    )
    recorder.capture(
        portfolio_module.get_hierarchical_risk_parity_weights(
            linkage_method="average", include_benchmark=True
        )
    )


def test_get_inverse_volatility_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_inverse_volatility_weights())


def test_get_minimum_cvar_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_minimum_cvar_weights(include_benchmark=True))
    recorder.capture(
        portfolio_module.get_minimum_cvar_weights(
            confidence_level=0.99, include_benchmark=True
        )
    )


def test_get_black_litterman_weights(recorder, portfolio_module):
    market = {"AAPL": 2.0, "MSFT": 1.5}
    recorder.capture(
        portfolio_module.get_black_litterman_weights(market_weights=market)
    )
    recorder.capture(
        portfolio_module.get_black_litterman_weights(
            views={"MSFT": 0.2, "AAPL-MSFT": -0.05},
            view_confidences={"MSFT": 0.8},
            market_weights=market,
        )
    )


def test_get_risk_contributions(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_risk_contributions())
    contributions = portfolio_module.get_risk_contributions(
        weights={"AAPL": 0.7, "MSFT": 0.3}
    )
    recorder.capture(contributions)
    assert contributions.loc["Total", "Percentage Contribution"] == pytest.approx(1)


def test_get_risk_contributions_unknown_asset(portfolio_module):
    assert portfolio_module.get_risk_contributions(weights={"TSLA": 1.0}).empty


def _offline_construction(test_toolkit) -> PortfolioConstruction:
    """A Portfolio Construction module whose outside data comes from fixed frames."""
    days = pd.period_range("2020-02-14", "2020-03-31", freq="D")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(-0.005, 0.02, (len(days), 5)), axis=0),
        index=days,
        columns=["AAPL", "MSFT", "SPY", "XLK", "XLV"],
    )

    profile = pd.DataFrame(
        {
            "AAPL": {
                "Sector": "Technology",
                "Country": "US",
                "Market Capitalization": 3,
            },
            "MSFT": {
                "Sector": "Technology",
                "Country": "US",
                "Market Capitalization": 2,
            },
        }
    )

    def funds(kind, tickers):
        if tickers == ["SPY"]:
            return pd.DataFrame(
                {"SPY": {"Technology": 0.4, "Healthcare": 0.6}}
                if kind == "sector"
                else {"SPY": {"United States": 0.97, "Ireland": 0.03}}
            )
        return pd.DataFrame()

    original = test_toolkit.portfolio
    return PortfolioConstruction(
        tickers=["AAPL", "MSFT"],
        historical_data=original._historical_data,
        risk_free_rate_data=original._risk_free_rate_data,
        benchmark_name="SPY",
        price_function=lambda tickers, start, end: prices.reindex(columns=tickers),
        profile_function=lambda: profile,
        fund_function=funds,
    )


def test_get_stress_test(recorder, test_toolkit):
    construction = _offline_construction(test_toolkit)
    recorder.capture(
        construction.get_stress_test(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            scenarios={"Crash": ("2020-02-19", "2020-03-23")},
        )
    )


def test_get_look_through_exposure(recorder, test_toolkit):
    construction = _offline_construction(test_toolkit)
    recorder.capture(construction.get_look_through_exposure())
    recorder.capture(construction.get_look_through_exposure(by="country"))
    recorder.capture(construction.get_look_through_exposure(by="holdings"))
    assert construction.get_look_through_exposure(by="industry").empty


def test_get_brinson_attribution(recorder, test_toolkit):
    construction = _offline_construction(test_toolkit)
    attribution = construction.get_brinson_attribution()
    recorder.capture(attribution)
    total = attribution.loc["Total"]
    assert total["Total Effect"] == pytest.approx(
        total["Portfolio Return"] - total["Benchmark Return"], abs=1e-3
    )
