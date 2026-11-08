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


def test_get_maximum_diversification_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_maximum_diversification_weights())
    recorder.capture(
        portfolio_module.get_maximum_diversification_weights(
            include_benchmark=True, covariance="denoised"
        )
    )


def test_get_minimum_tracking_error_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_minimum_tracking_error_weights())
    recorder.capture(
        portfolio_module.get_minimum_tracking_error_weights(
            objective="information_ratio", period="weekly"
        )
    )


def test_get_growth_optimal_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_growth_optimal_weights())
    recorder.capture(
        portfolio_module.get_growth_optimal_weights(
            method="approximate", fraction=0.5, include_benchmark=True
        )
    )


def test_the_tracking_error_of_the_portfolio_is_that_of_its_active_returns(
    portfolio_module,
):
    table = portfolio_module.get_minimum_tracking_error_weights(rounding=12)
    returns = portfolio_module._get_returns("daily", True)
    weights = table["Weight"].drop("Portfolio")
    active = returns[weights.index] @ weights - returns["Benchmark"]

    assert table.loc["Portfolio", "Tracking Error"] == pytest.approx(
        active.std() * np.sqrt(252)
    )
    assert table.loc["Portfolio", "Active Return"] == pytest.approx(active.mean() * 252)


def test_fractional_kelly_holds_the_rest_in_cash(portfolio_module):
    full = portfolio_module.get_growth_optimal_weights(rounding=12)
    half = portfolio_module.get_growth_optimal_weights(fraction=0.5, rounding=12)

    assert half.loc["Cash", "Weight"] == pytest.approx(0.5)
    assert half.loc["Portfolio", "Weight"] == pytest.approx(1)
    np.testing.assert_allclose(
        half["Weight"].drop(["Cash", "Portfolio"]),
        full["Weight"].drop("Portfolio") / 2,
    )
    assert half.loc["Portfolio", "Volatility"] == pytest.approx(
        full.loc["Portfolio", "Volatility"] / 2
    )
    assert "Cash" not in full.index


def test_get_minimum_cvar_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_minimum_cvar_weights(include_benchmark=True))
    recorder.capture(
        portfolio_module.get_minimum_cvar_weights(
            confidence_level=0.99, include_benchmark=True
        )
    )


def test_get_hierarchical_equal_risk_contribution_weights(recorder, portfolio_module):
    table = portfolio_module.get_hierarchical_equal_risk_contribution_weights(
        include_benchmark=True
    )
    recorder.capture(table)
    assert table["Weight"].drop("Portfolio").sum() == pytest.approx(1, abs=1e-4)
    recorder.capture(
        portfolio_module.get_hierarchical_equal_risk_contribution_weights(
            clusters=3, linkage_method="average", include_benchmark=True
        )
    )


def test_get_nested_clustered_optimization_weights(recorder, portfolio_module):
    recorder.capture(
        portfolio_module.get_nested_clustered_optimization_weights(
            include_benchmark=True
        )
    )
    recorder.capture(
        portfolio_module.get_nested_clustered_optimization_weights(
            objective="risk_parity", clusters=2, include_benchmark=True
        )
    )


def test_get_minimum_drawdown_weights(recorder, portfolio_module):
    for measure in ("cdar", "maximum", "average"):
        recorder.capture(
            portfolio_module.get_minimum_drawdown_weights(
                measure=measure, include_benchmark=True
            )
        )


def test_the_minimum_drawdown_portfolio_has_the_lowest_drawdown(portfolio_module):
    table = portfolio_module.get_minimum_drawdown_weights(
        measure="maximum", include_benchmark=True, rounding=12
    )
    returns = portfolio_module._get_returns("daily", True)
    weights = table["Weight"].drop("Portfolio")
    cumulative = (returns @ weights).cumsum()
    drawdown = (cumulative.cummax().clip(lower=0) - cumulative).max()

    assert table.loc["Portfolio", "Maximum Drawdown"] == pytest.approx(drawdown)
    assert (
        table.loc["Portfolio", "Maximum Drawdown"]
        <= table["Maximum Drawdown"].drop("Portfolio").min() + 1e-12
    )


def test_get_maximum_omega_weights(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_maximum_omega_weights(include_benchmark=True))
    recorder.capture(
        portfolio_module.get_maximum_omega_weights(
            minimum_acceptable_return=0.0005, period="weekly", include_benchmark=True
        )
    )
    assert portfolio_module.get_maximum_omega_weights(
        minimum_acceptable_return=0.5
    ).empty


def test_get_maximum_cvar_ratio_weights(recorder, portfolio_module):
    table = portfolio_module.get_maximum_cvar_ratio_weights(include_benchmark=True)
    recorder.capture(table)
    assert (
        table.loc["Portfolio", "CVaR Ratio"]
        >= table["CVaR Ratio"].drop("Portfolio").max() - 1e-4
    )
    recorder.capture(
        portfolio_module.get_maximum_cvar_ratio_weights(
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


def test_get_brinson_attribution_looks_through_funds(test_toolkit):
    # MSFT stands in for a fund: its profile names the issuer's sector, while its sector
    # weightings split it over Technology and Healthcare.
    construction = _offline_construction(test_toolkit)
    construction._profile_function = lambda: pd.DataFrame(
        {"AAPL": {"Sector": "Technology"}, "MSFT": {"Sector": "Financial Services"}}
    )
    spy_and_fund = {
        "SPY": {"Technology": 0.4, "Healthcare": 0.6},
        "MSFT": {"Technology": 0.5, "Healthcare": 0.5},
    }
    construction._fund_function = lambda kind, tickers: pd.DataFrame(
        {ticker: spy_and_fund[ticker] for ticker in tickers if ticker in spy_and_fund}
    )
    attribution = construction.get_brinson_attribution(
        weights={"AAPL": 0.5, "MSFT": 0.5}
    )

    assert "Financial Services" not in attribution.index
    assert attribution.loc["Technology", "Portfolio Weight"] == pytest.approx(0.75)
    assert attribution.loc["Healthcare", "Portfolio Weight"] == pytest.approx(0.25)


def test_get_stress_test_notes_a_scenario_before_any_listing(test_toolkit, caplog):
    construction = _offline_construction(test_toolkit)
    with caplog.at_level("WARNING", logger="financetoolkit"):
        stress = construction.get_stress_test(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            scenarios={"Early": ("2019-01-02", "2019-02-01")},
        )

    assert stress.loc["Early", "Coverage"] == 0
    assert np.isnan(stress.loc["Early", "Portfolio Return"])
    assert "No price data is available for the Early" in caplog.text


def test_brinson_attribution_leaves_out_an_asset_without_a_start_price(
    test_toolkit, caplog
):
    construction = _offline_construction(test_toolkit)
    daily = construction._historical_data["daily"].copy()
    daily.loc[daily.index[:10], ("Adj Close", "MSFT")] = np.nan
    construction._historical_data = {"daily": daily}

    with caplog.at_level("WARNING", logger="financetoolkit"):
        attribution = construction.get_brinson_attribution(
            weights={"AAPL": 0.5, "MSFT": 0.5}
        )

    # MSFT would otherwise count with half the weight and a return over a shorter span.
    assert attribution.loc["Technology", "Portfolio Weight"] == pytest.approx(1)
    assert "No price data is available for MSFT" in caplog.text


def test_weights_with_a_negative_sum_are_refused(test_toolkit):
    construction = _offline_construction(test_toolkit)

    with pytest.raises(ValueError, match="positive net weight"):
        construction._resolve_weights({"AAPL": -0.5, "MSFT": -0.5}, ["AAPL", "MSFT"])


def test_brinson_attribution_keeps_an_asset_closed_on_the_first_day(test_toolkit):
    construction = _offline_construction(test_toolkit)
    daily = construction._historical_data["daily"].copy()
    # A holiday on MSFT's exchange: no price on the first two days of the range.
    daily.loc[daily.index[:2], ("Adj Close", "MSFT")] = np.nan
    construction._historical_data = {"daily": daily}

    attribution = construction.get_brinson_attribution(
        weights={"AAPL": 0.5, "MSFT": 0.5}
    )

    # Both assets keep their weight; MSFT starts at its first price.
    assert attribution.loc["Technology", "Portfolio Weight"] == pytest.approx(1)
    prices = daily["Adj Close"]
    expected = 0.5 * (prices["AAPL"].iloc[-1] / prices["AAPL"].iloc[0] - 1) + 0.5 * (
        prices["MSFT"].iloc[-1] / prices["MSFT"].iloc[2] - 1
    )
    assert attribution.loc["Technology", "Portfolio Return"] == pytest.approx(
        expected, abs=1e-4
    )


@pytest.mark.parametrize(
    "method",
    [
        "get_minimum_variance_weights",
        "get_maximum_sharpe_weights",
        "get_risk_parity_weights",
        "get_hierarchical_risk_parity_weights",
        "get_inverse_volatility_weights",
        "get_black_litterman_weights",
        "get_maximum_diversification_weights",
        "get_minimum_tracking_error_weights",
        "get_growth_optimal_weights",
        "get_hierarchical_equal_risk_contribution_weights",
        "get_nested_clustered_optimization_weights",
        "get_resampled_efficient_weights",
    ],
)
@pytest.mark.parametrize(
    "covariance",
    ["sample", "ledoit_wolf", "oas", "semicovariance", "exponential", "denoised"],
)
def test_the_optimizers_take_a_covariance_estimator(
    portfolio_module, method, covariance
):
    # Black-Litterman needs market weights, which come from FMP profiles otherwise.
    extra = (
        {"market_weights": {"AAPL": 2.0, "MSFT": 1.5}} if "litterman" in method else {}
    )
    # Fewer resamples keep the resampled weights quick.
    extra |= {"resamples": 50} if "resampled" in method else {}
    weights = getattr(portfolio_module, method)(covariance=covariance, **extra)[
        "Weight"
    ].drop("Portfolio", errors="ignore")

    assert weights.sum() == pytest.approx(1, abs=1e-4)
    assert (weights >= -1e-8).all()


def test_a_covariance_estimator_changes_the_weights_and_their_risk(portfolio_module):
    sample = portfolio_module.get_minimum_variance_weights(rounding=10)
    exponential = portfolio_module.get_minimum_variance_weights(
        covariance="exponential", covariance_span=20, rounding=10
    )
    contributions = portfolio_module.get_risk_contributions(
        covariance="oas", rounding=10
    )

    assert not np.allclose(sample["Weight"], exponential["Weight"])
    assert not np.allclose(
        sample.loc["Portfolio", "Volatility"],
        exponential.loc["Portfolio", "Volatility"],
    )
    assert contributions.loc["Total", "Percentage Contribution"] == pytest.approx(1)


@pytest.mark.parametrize(
    "method",
    [
        "get_minimum_variance_weights",
        "get_maximum_sharpe_weights",
        "get_mean_variance_weights",
        "get_minimum_cvar_weights",
        "get_maximum_diversification_weights",
        "get_growth_optimal_weights",
        "get_risk_parity_weights",
        "get_hierarchical_equal_risk_contribution_weights",
        "get_minimum_drawdown_weights",
        "get_maximum_omega_weights",
        "get_maximum_cvar_ratio_weights",
        "get_resampled_efficient_weights",
    ],
)
def test_sector_and_country_bounds_come_from_the_profiles(test_toolkit, method):
    construction = _offline_construction(test_toolkit)
    extra = {"resamples": 50} if "resampled" in method else {}
    by_sector = getattr(construction, method)(
        include_benchmark=True,
        sector_bounds={"Technology": (0, 0.6)},
        rounding=10,
        **extra,
    )
    # The country is named by its code in the profile; both the code and the name work.
    by_country = getattr(construction, method)(
        include_benchmark=True, country_bounds={"US": [0.2, 0.5]}, rounding=10, **extra
    )
    by_name = getattr(construction, method)(
        include_benchmark=True,
        country_bounds={"United States": [0.2, 0.5]},
        rounding=10,
        **extra,
    )

    assert by_sector.loc[["AAPL", "MSFT"], "Weight"].sum() <= 0.6 + 1e-8
    assert 0.2 - 1e-8 <= by_country.loc[["AAPL", "MSFT"], "Weight"].sum() <= 0.5 + 1e-8
    pd.testing.assert_frame_equal(by_country, by_name)
    assert by_sector.loc["Portfolio", "Weight"] == pytest.approx(1)


def test_per_asset_bounds_and_long_short_exposures(test_toolkit):
    construction = _offline_construction(test_toolkit)
    capped = construction.get_maximum_sharpe_weights(
        include_benchmark=True, maximum_weight={"AAPL": 0.2}, rounding=10
    )
    long_short = construction.get_minimum_variance_weights(
        include_benchmark=True, minimum_weight=-0.5, gross_exposure=1.2, rounding=10
    )
    frontier = construction.get_efficient_frontier(
        points=3,
        include_benchmark=True,
        minimum_weight=-0.5,
        net_exposure=0.5,
        gross_exposure=1.5,
        rounding=10,
    )
    hierarchical = construction.get_hierarchical_risk_parity_weights(
        include_benchmark=True, maximum_weight={"AAPL": 0.2}, rounding=10
    )

    assert capped.loc["AAPL", "Weight"] == pytest.approx(0.2)
    weights = long_short["Weight"].drop("Portfolio")
    assert weights.sum() == pytest.approx(1)
    assert weights.abs().sum() <= 1.2 + 1e-8
    assets = frontier[["AAPL", "MSFT", "Benchmark"]]
    assert np.allclose(assets.sum(axis=1), 0.5)
    assert (assets.abs().sum(axis=1) <= 1.5 + 1e-8).all()
    assert hierarchical.loc["AAPL", "Weight"] <= 0.2 + 1e-10


def test_unsupported_or_impossible_constraints_are_refused(test_toolkit):
    construction = _offline_construction(test_toolkit)

    # handle_errors turns the ValueError into an empty result and logs the reason.
    assert construction.get_minimum_variance_weights(
        sector_bounds={"Energy": (0, 0.5)}
    ).empty
    assert construction.get_minimum_variance_weights(
        maximum_weight={"AAPL": 0.3, "MSFT": 0.3}
    ).empty
    assert construction.get_risk_parity_weights(minimum_weight=-0.1).empty
    with pytest.raises(TypeError, match="gross_exposure"):
        construction.get_hierarchical_risk_parity_weights(gross_exposure=1.5)
    with pytest.raises(TypeError, match="sector_bounds"):
        construction.get_hierarchical_risk_parity_weights(
            sector_bounds={"Technology": (0, 0.5)}
        )
    assert construction.get_hierarchical_equal_risk_contribution_weights(
        minimum_weight=-0.1
    ).empty
    with pytest.raises(TypeError, match="maximum_weight"):
        construction.get_nested_clustered_optimization_weights(maximum_weight=0.5)


def test_fractional_kelly_with_a_net_exposure_holds_the_rest_in_cash(test_toolkit):
    construction = _offline_construction(test_toolkit)
    table = construction.get_growth_optimal_weights(
        fraction=0.5, net_exposure=0.8, rounding=10
    )

    assert table.loc[["AAPL", "MSFT"], "Weight"].sum() == pytest.approx(0.4)
    assert table.loc["Cash", "Weight"] == pytest.approx(0.6)


def test_get_optimization_backtest(recorder, portfolio_module):
    summary = portfolio_module.get_optimization_backtest(transaction_cost=0.001)
    recorder.capture(summary)
    assert {"Equal Weight", "Benchmark"} <= set(summary.index)
    assert summary.loc["Equal Weight", "Average Turnover"] > 0
    assert np.isnan(summary.loc["Benchmark", "Average Turnover"])

    returns = portfolio_module.get_optimization_backtest(
        methods="minimum_variance,minimum_cvar",
        rebalancing="quarterly",
        output="returns",
    )
    recorder.capture(returns.iloc[-20:])
    assert list(returns.columns) == [
        "Minimum Variance",
        "Minimum CVaR",
        "Equal Weight",
        "Benchmark",
    ]
    weights = portfolio_module.get_optimization_backtest(
        methods=["mean_variance"], expected_returns="james_stein", output="weights"
    )
    recorder.capture(weights)
    assert np.allclose(weights.T.groupby(level=0).sum().T, 1, atol=1e-3)


def test_get_optimization_backtest_refuses_unknown_methods(portfolio_module):
    assert portfolio_module.get_optimization_backtest(methods="best").empty
    assert portfolio_module.get_optimization_backtest(lookback=2).empty


def test_get_risk_contributions_to_the_expected_shortfall(recorder, portfolio_module):
    contributions = portfolio_module.get_risk_contributions(
        weights={"AAPL": 0.7, "MSFT": 0.3}, risk_measure="cvar", rounding=8
    )
    recorder.capture(contributions)
    returns = portfolio_module._get_returns("daily", False)
    portfolio = returns["AAPL"] * 0.7 + returns["MSFT"] * 0.3
    # The tail of 5% of the periods holds a fraction of its boundary period, as in the
    # Rockafellar-Uryasev CVaR the optimizers minimize.
    tail = 0.05 * len(portfolio)
    assert tail != int(tail)
    losses = (-portfolio).sort_values(ascending=False).to_numpy()
    shortfall = (
        losses[: int(tail)].sum() + (tail - int(tail)) * losses[int(tail)]
    ) / tail
    assert contributions.loc["Total", "Risk Contribution"] == pytest.approx(
        shortfall, abs=1e-8
    )


def test_get_diversification_metrics(recorder, portfolio_module):
    metrics = portfolio_module.get_diversification_metrics()
    recorder.capture(metrics)
    assert metrics["Effective Number of Assets"] == pytest.approx(2)
    assert 1 <= metrics["Effective Number of Bets"] <= 2  # noqa: PLR2004


def test_expected_return_estimators_change_the_expected_returns(portfolio_module):
    tables = {
        method: portfolio_module.get_mean_variance_weights(
            expected_returns=method, include_benchmark=True, rounding=8
        )
        for method in ("mean", "exponential", "capm", "james_stein")
    }
    returns = portfolio_module._get_returns("daily", True)
    assert np.allclose(
        tables["mean"]["Expected Return"].drop("Portfolio"), returns.mean() * 252
    )
    # The benchmark's beta to itself is one, so CAPM gives it its own average return.
    assert tables["capm"].loc["Benchmark", "Expected Return"] == pytest.approx(
        tables["mean"].loc["Benchmark", "Expected Return"]
    )
    for method in ("exponential", "capm", "james_stein"):
        assert not np.allclose(
            tables[method]["Expected Return"], tables["mean"]["Expected Return"]
        )


def test_current_weights_add_the_turnover_and_limit_it(portfolio_module):
    current = {"AAPL": 0.2, "MSFT": 0.8}
    free = portfolio_module.get_minimum_variance_weights(current_weights=current)
    capped = portfolio_module.get_minimum_variance_weights(
        current_weights=current, maximum_turnover=0.1, rounding=8
    )

    assert free.loc["Portfolio", "Turnover"] > 0.1  # noqa: PLR2004
    assert capped.loc["Portfolio", "Turnover"] == pytest.approx(0.1)
    assert capped.loc["AAPL", "Turnover"] == pytest.approx(
        abs(capped.loc["AAPL", "Weight"] - 0.2)
    )
    assert "Turnover" not in portfolio_module.get_minimum_variance_weights().columns
    # A turnover limit or cost without current weights or a Portfolio to take them from.
    assert portfolio_module.get_minimum_variance_weights(maximum_turnover=0.1).empty
    assert portfolio_module.get_maximum_sharpe_weights(transaction_cost=0.01).empty


def test_a_transaction_cost_trades_less(portfolio_module):
    current = {"AAPL": 0.2, "MSFT": 0.8}
    free = portfolio_module.get_mean_variance_weights(
        current_weights=current, include_benchmark=True
    )
    costly = portfolio_module.get_mean_variance_weights(
        current_weights=current, transaction_cost=0.02, include_benchmark=True
    )
    assert costly.loc["Portfolio", "Turnover"] < free.loc["Portfolio", "Turnover"]


def test_the_backtest_leaves_out_days_on_which_none_of_the_assets_traded(
    portfolio_module, monkeypatch
):
    daily = portfolio_module._historical_data["daily"].copy()
    # Only the benchmark's market is open on this day.
    closed = daily.index[-30]
    daily.loc[closed, [("Return", "AAPL"), ("Return", "MSFT")]] = np.nan
    monkeypatch.setitem(portfolio_module._historical_data, "daily", daily)

    returns = portfolio_module.get_optimization_backtest(
        methods="minimum_variance", output="returns"
    )

    assert closed not in returns.index
    assert returns.index[-1] == daily.index[-1]


def test_get_resampled_efficient_weights(recorder, portfolio_module):
    table = portfolio_module.get_resampled_efficient_weights(
        resamples=100, include_benchmark=True
    )
    recorder.capture(table)
    recorder.capture(
        portfolio_module.get_resampled_efficient_weights(
            objective="minimum_variance",
            resamples=100,
            resampling="bootstrap",
            include_benchmark=True,
        )
    )
    assert table.loc["Portfolio", "Weight"] == pytest.approx(1)
    assert (table["Weight Standard Deviation"].drop("Portfolio") > 0).all()

    # The same seed gives the same weights, another seed or none other ones.
    same = portfolio_module.get_resampled_efficient_weights(
        resamples=100, include_benchmark=True
    )
    pd.testing.assert_frame_equal(table, same)
    other = portfolio_module.get_resampled_efficient_weights(
        resamples=100, seed=None, include_benchmark=True
    )
    assert not np.allclose(table["Weight"], other["Weight"])

    # The average of weights within the bounds is within them.
    capped = portfolio_module.get_resampled_efficient_weights(
        resamples=50, maximum_weight=0.4, include_benchmark=True, rounding=10
    )
    assert (capped["Weight"].drop("Portfolio") <= 0.4 + 1e-8).all()
    assert portfolio_module.get_resampled_efficient_weights(objective="best").empty
    assert portfolio_module.get_resampled_efficient_weights(
        resampling="jackknife"
    ).empty


def test_the_backtest_runs_the_clustering_drawdown_and_ratio_optimizers(
    recorder, portfolio_module
):
    methods = [
        "hierarchical_equal_risk_contribution",
        "nested_clustered_optimization",
        "minimum_drawdown",
        "maximum_omega",
        "maximum_cvar_ratio",
    ]
    summary = portfolio_module.get_optimization_backtest(
        methods=methods, rebalancing="quarterly", transaction_cost=0.001
    )
    recorder.capture(summary)
    assert list(summary.index) == [
        "Hierarchical Equal Risk Contribution",
        "Nested Clustered Optimization",
        "Minimum Drawdown",
        "Maximum Omega",
        "Maximum CVaR Ratio",
        "Equal Weight",
        "Benchmark",
    ]
    weights = portfolio_module.get_optimization_backtest(
        methods=methods, rebalancing="quarterly", maximum_weight=0.7, output="weights"
    )
    assert np.allclose(weights.T.groupby(level=0).sum().T, 1, atol=1e-6)
    # The ratio optimizers find no weights on a date on which no asset had a return
    # above zero, and keep their drifted weights there.
    bounded = ["Hierarchical Equal Risk Contribution", "Minimum Drawdown"]
    assert (weights[bounded] <= 0.7 + 1e-6).all().all()


@pytest.mark.parametrize(
    "method",
    [
        "get_minimum_drawdown_weights",
        "get_maximum_omega_weights",
        "get_maximum_cvar_ratio_weights",
    ],
)
def test_the_drawdown_and_ratio_optimizers_limit_the_turnover(portfolio_module, method):
    current = {"AAPL": 0.2, "MSFT": 0.6, "Benchmark": 0.2}
    free = getattr(portfolio_module, method)(
        current_weights=current, include_benchmark=True, rounding=10
    )
    capped = getattr(portfolio_module, method)(
        current_weights=current,
        maximum_turnover=0.1,
        include_benchmark=True,
        rounding=10,
    )
    assert free.loc["Portfolio", "Turnover"] > 0.1  # noqa: PLR2004
    assert capped.loc["Portfolio", "Turnover"] == pytest.approx(0.1, abs=1e-7)
    assert capped.loc["Portfolio", "Weight"] == pytest.approx(1)


@pytest.mark.parametrize(
    ("method", "column"),
    [
        ("get_maximum_omega_weights", "Omega Ratio"),
        ("get_maximum_cvar_ratio_weights", "CVaR Ratio"),
        ("get_nested_clustered_optimization_weights", None),
    ],
)
def test_the_ratio_optimizers_take_an_expected_return_estimator(
    portfolio_module, method, column
):
    extra = {"objective": "maximum_sharpe"} if column is None else {}
    mean = getattr(portfolio_module, method)(include_benchmark=True, **extra)
    shrunk = getattr(portfolio_module, method)(
        expected_returns="james_stein", include_benchmark=True, **extra
    )
    assert not np.allclose(mean["Weight"], shrunk["Weight"])
    assert not np.allclose(mean["Expected Return"], shrunk["Expected Return"])
    if column is not None:
        # The ratio reported is the one maximized, on the expected returns used, so no
        # asset alone beats the portfolio.
        assert (
            shrunk[column].drop("Portfolio").max()
            <= shrunk.loc["Portfolio", column] + 1e-8
        )
