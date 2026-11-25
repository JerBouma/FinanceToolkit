"""Portfolio Construction Controller Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.portfolio.construction.construction_controller import (
    PortfolioConstruction,
)
from financetoolkit.portfolio.construction.optimization import optimization_model
from financetoolkit.utilities.error_model import InvalidArgumentError

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


def test_the_minimum_drawdown_portfolio_has_the_lowest_drawdown(portfolio_module):
    table = portfolio_module.get_mean_risk_weights(
        risk_measure="maximum_drawdown", include_benchmark=True, rounding=12
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


def test_get_mean_risk_weights(recorder, portfolio_module):
    table = portfolio_module.get_mean_risk_weights(include_benchmark=True)
    recorder.capture(table)
    # The minimum risk portfolio has no more risk than any asset on its own.
    assert table.loc["Portfolio", "MAD"] <= table["MAD"].drop("Portfolio").min() + 1e-4
    ratio = portfolio_module.get_mean_risk_weights(
        risk_measure="gini_mean_difference",
        objective="maximum_ratio",
        include_benchmark=True,
    )
    recorder.capture(ratio)
    assert (
        ratio.loc["Portfolio", "Gini Mean Difference Ratio"]
        >= ratio["Gini Mean Difference Ratio"].drop("Portfolio").max() - 1e-4
    )
    recorder.capture(
        portfolio_module.get_mean_risk_weights(
            risk_measure="semideviation",
            target_return=0.2,
            minimum_acceptable_return="mean",
            include_benchmark=True,
        )
    )

    # The minimum CVaR portfolio, recorded without its ratio column.
    for alpha in (0.05, 0.01):
        recorder.capture(
            portfolio_module.get_mean_risk_weights(
                risk_measure="cvar",
                alpha=alpha,
                include_benchmark=True,
            ).drop(columns="CVaR Ratio")
        )

    # The maximum CVaR ratio (STARR) portfolio.
    table = portfolio_module.get_mean_risk_weights(
        risk_measure="cvar", objective="maximum_ratio", include_benchmark=True
    )
    recorder.capture(table)
    assert (
        table.loc["Portfolio", "CVaR Ratio"]
        >= table["CVaR Ratio"].drop("Portfolio").max() - 1e-4
    )
    recorder.capture(
        portfolio_module.get_mean_risk_weights(
            risk_measure="cvar",
            objective="maximum_ratio",
            alpha=0.01,
            include_benchmark=True,
        )
    )

    # The minimum drawdown portfolios, recorded without their ratio columns.
    for risk_measure, name in (
        ("cdar", "CDaR"),
        ("maximum_drawdown", "Maximum Drawdown"),
        ("average_drawdown", "Average Drawdown"),
    ):
        recorder.capture(
            portfolio_module.get_mean_risk_weights(
                risk_measure=risk_measure, include_benchmark=True
            ).drop(columns=f"{name} Ratio")
        )

    # Holding nothing has no drawdown, so a net exposure of 0 is refused.
    assert portfolio_module.get_mean_risk_weights(
        risk_measure="cdar", net_exposure=0, minimum_weight=-1, include_benchmark=True
    ).empty
    with pytest.raises(InvalidArgumentError, match="risk_measure must be one of"):
        portfolio_module.get_mean_risk_weights(risk_measure="cvaR")


def test_get_risk_parity_weights_risk_measure(recorder, portfolio_module):
    table = portfolio_module.get_risk_parity_weights(
        risk_measure="cvar", include_benchmark=True
    )
    recorder.capture(table)
    assert np.allclose(table["CVaR Contribution"].iloc[:-1], 1 / 3, atol=1e-3)
    recorder.capture(
        portfolio_module.get_risk_parity_weights(
            risk_measure="mad",
            risk_budget={"AAPL": 0.5, "MSFT": 0.3, "Benchmark": 0.2},
            include_benchmark=True,
        )
    )


def test_get_hierarchical_weights_with_other_risk_measures(recorder, portfolio_module):
    table = portfolio_module.get_hierarchical_risk_parity_weights(
        risk_measure="cvar", include_benchmark=True
    )
    recorder.capture(table)
    assert "CVaR" in table.columns
    recorder.capture(
        portfolio_module.get_hierarchical_equal_risk_contribution_weights(
            risk_measure="cdar",
            within_clusters="equal",
            clusters=2,
            include_benchmark=True,
        )
    )
    recorder.capture(
        portfolio_module.get_nested_clustered_optimization_weights(
            risk_measure="mad",
            objective="risk_parity",
            clusters=2,
            include_benchmark=True,
        )
    )
    recorder.capture(
        portfolio_module.get_hierarchical_equal_risk_contribution_weights(
            cluster_criterion="standardized_silhouette", include_benchmark=True
        )
    )


def test_get_schur_complementary_weights(recorder, portfolio_module):
    table = portfolio_module.get_schur_complementary_weights(include_benchmark=True)
    recorder.capture(table)
    # With three assets one half of the first split is a single asset, which leaves
    # nothing to augment, so the weights are those of Hierarchical Risk Parity.
    pd.testing.assert_frame_equal(
        table,
        portfolio_module.get_hierarchical_risk_parity_weights(include_benchmark=True),
    )


def test_get_stacking_weights(recorder, portfolio_module):
    table = portfolio_module.get_stacking_weights(include_benchmark=True)
    recorder.capture(table)
    assert table.loc["Portfolio", "Weight"] == pytest.approx(1, abs=1e-4)
    recorder.capture(
        portfolio_module.get_stacking_weights(
            methods="minimum_variance,risk_parity,equal_weight",
            final_method="inverse_volatility",
            folds=None,
            include_benchmark=True,
        )
    )
    with pytest.raises(InvalidArgumentError, match="methods must be one of"):
        portfolio_module.get_stacking_weights(methods=["random"])


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
    # The sector funds of the attribution cover the date range of the assets.
    fund_days = pd.bdate_range("2019-12-31", "2022-12-30").to_period("D")
    sector_funds = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.012, (len(fund_days), 2)), axis=0),
        index=fund_days,
        columns=["XLK", "XLV"],
    )

    def price_function(tickers, start, end):
        if set(tickers) <= set(sector_funds.columns):
            return sector_funds.reindex(columns=tickers)
        return prices.reindex(columns=tickers)

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
        price_function=price_function,
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
    with pytest.raises(InvalidArgumentError, match="by must be one of"):
        construction.get_look_through_exposure(by="industry")


def test_get_brinson_attribution(recorder, test_toolkit):
    construction = _offline_construction(test_toolkit)
    attribution = construction.get_brinson_attribution()
    recorder.capture(attribution)
    total = attribution.loc["Total"]
    assert total["Total Effect"] == pytest.approx(
        total["Portfolio Return"] - total["Benchmark Return"], abs=1e-3
    )


def test_get_brinson_attribution_linked(recorder, test_toolkit):
    construction = _offline_construction(test_toolkit)
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    single = construction.get_brinson_attribution(weights=weights, rounding=16)
    effects = ["Allocation Effect", "Selection Effect", "Interaction Effect"]

    for period, linking in [("monthly", "carino"), ("quarterly", "menchero")]:
        linked = construction.get_brinson_attribution(
            weights=weights, period=period, linking=linking, rounding=16
        )
        recorder.capture(linked.round(6))
        total = linked.loc["Total"]
        # The linked effects add up to the compounded excess return, and buy and hold
        # over the sub-periods compounds to the returns of the whole range.
        assert total["Total Effect"] == pytest.approx(
            total["Portfolio Return"] - total["Benchmark Return"], abs=1e-12
        )
        assert total[effects].sum() == pytest.approx(total["Total Effect"], abs=1e-12)
        assert total["Total Effect"] == pytest.approx(
            single.loc["Total", "Total Effect"], abs=1e-12
        )
        pd.testing.assert_frame_equal(
            linked.iloc[:, :4], single.iloc[:, :4], check_exact=False, atol=1e-12
        )
        # The weights drift, so the effects differ from those of one period.
        assert not np.allclose(linked[effects], single[effects])


def test_get_brinson_attribution_invalid_period_and_linking(test_toolkit):
    construction = _offline_construction(test_toolkit)
    with pytest.raises(InvalidArgumentError, match="period must be weekly"):
        construction.get_brinson_attribution(period="hourly")
    with pytest.raises(InvalidArgumentError, match="linking must be one of"):
        construction.get_brinson_attribution(period="monthly", linking="frongello")


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

    # Linked over weeks, the fund's slices drift with its return and the effects still add
    # up to the excess return of the whole range.
    for linking in ("carino", "menchero"):
        linked = construction.get_brinson_attribution(
            weights={"AAPL": 0.5, "MSFT": 0.5},
            period="weekly",
            linking=linking,
            rounding=16,
        )
        pd.testing.assert_frame_equal(
            linked.iloc[:, :4],
            construction.get_brinson_attribution(
                weights={"AAPL": 0.5, "MSFT": 0.5}, rounding=16
            ).iloc[:, :4],
            check_exact=False,
            atol=1e-12,
        )
        total = linked.loc["Total"]
        assert total["Total Effect"] == pytest.approx(
            total["Portfolio Return"] - total["Benchmark Return"], abs=1e-12
        )
        assert linked["Total Effect"].iloc[:-1].sum() == pytest.approx(
            total["Total Effect"], abs=1e-12
        )


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
    [
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
        "jlogo",
        "geodesic",
        "regime_adjusted_ew",
    ],
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
    ("method", "extra"),
    [
        ("get_minimum_variance_weights", {}),
        ("get_maximum_sharpe_weights", {}),
        ("get_mean_variance_weights", {}),
        ("get_mean_risk_weights", {"risk_measure": "cvar"}),
        ("get_maximum_diversification_weights", {}),
        ("get_growth_optimal_weights", {}),
        ("get_risk_parity_weights", {}),
        ("get_hierarchical_equal_risk_contribution_weights", {}),
        ("get_mean_risk_weights", {"risk_measure": "cdar"}),
        ("get_maximum_omega_weights", {}),
        (
            "get_mean_risk_weights",
            {"risk_measure": "cvar", "objective": "maximum_ratio"},
        ),
        ("get_resampled_efficient_weights", {"resamples": 50}),
    ],
)
def test_sector_and_country_bounds_come_from_the_profiles(test_toolkit, method, extra):
    construction = _offline_construction(test_toolkit)
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


def test_l2_regularization(recorder, portfolio_module):
    recorder.capture(
        portfolio_module.get_minimum_variance_weights(l2_regularization=0.05)
    )
    recorder.capture(
        portfolio_module.get_maximum_sharpe_weights(
            maximum_weight=0.8, l2_regularization=0.05
        )
    )
    recorder.capture(portfolio_module.get_mean_variance_weights(l2_regularization=0.05))
    recorder.capture(
        portfolio_module.get_efficient_frontier(points=4, l2_regularization=0.05)
    )

    # A higher γ brings the weights closer to equal weights.
    distances = [
        (
            portfolio_module.get_minimum_variance_weights(
                l2_regularization=gamma, rounding=10
            )["Weight"].drop("Portfolio")
            - 0.5
        )
        .abs()
        .sum()
        for gamma in (0.0, 0.05, 1.0)
    ]
    assert distances == sorted(distances, reverse=True)
    assert portfolio_module.get_minimum_variance_weights(l2_regularization=-1).empty


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
        methods=["mean_variance"], expected_returns="bayes_stein", output="weights"
    )
    recorder.capture(weights)
    assert np.allclose(weights.T.groupby(level=0).sum().T, 1, atol=1e-3)


def test_the_backtest_runs_the_mean_risk_and_random_portfolios(
    recorder, portfolio_module
):
    methods = "minimum_mad,minimum_semideviation,random"
    summary = portfolio_module.get_optimization_backtest(
        methods=methods, rebalancing="quarterly", transaction_cost=0.001
    )
    recorder.capture(summary)
    assert list(summary.index) == [
        "Minimum MAD",
        "Minimum Semideviation",
        "Random",
        "Equal Weight",
        "Benchmark",
    ]
    weights = portfolio_module.get_optimization_backtest(
        methods=methods, rebalancing="quarterly", maximum_weight=0.7, output="weights"
    )
    recorder.capture(weights)
    assert np.allclose(weights.T.groupby(level=0).sum().T, 1, atol=1e-6)
    assert (weights[["Minimum MAD", "Minimum Semideviation"]] <= 0.7 + 1e-6).all().all()

    # One generator seeded at 0 per backtest: the random portfolios are the successive
    # draws of that generator (to the Toolkit's rounding), the same on every run.
    generator = np.random.default_rng(0)
    draws = [
        optimization_model.get_random_weights(weights["Random"].columns, generator)
        for _ in weights.index
    ]
    np.testing.assert_allclose(weights["Random"], draws, atol=1e-4)


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
        for method in ("mean", "exponential", "capm", "bayes_stein")
    }
    returns = portfolio_module._get_returns("daily", True)
    assert np.allclose(
        tables["mean"]["Expected Return"].drop("Portfolio"), returns.mean() * 252
    )
    # The benchmark's beta to itself is one, so CAPM gives it its own average return.
    assert tables["capm"].loc["Benchmark", "Expected Return"] == pytest.approx(
        tables["mean"].loc["Benchmark", "Expected Return"]
    )
    for method in ("exponential", "capm", "bayes_stein"):
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
    with pytest.raises(InvalidArgumentError, match="objective must be one of"):
        portfolio_module.get_resampled_efficient_weights(objective="best")
    with pytest.raises(InvalidArgumentError, match="resampling must be one of"):
        portfolio_module.get_resampled_efficient_weights(resampling="jackknife")


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
    ("method", "extra"),
    [
        ("get_mean_risk_weights", {"risk_measure": "cdar"}),
        ("get_maximum_omega_weights", {}),
        (
            "get_mean_risk_weights",
            {"risk_measure": "cvar", "objective": "maximum_ratio"},
        ),
    ],
)
def test_the_drawdown_and_ratio_optimizers_limit_the_turnover(
    portfolio_module, method, extra
):
    current = {"AAPL": 0.2, "MSFT": 0.6, "Benchmark": 0.2}
    free = getattr(portfolio_module, method)(
        current_weights=current, include_benchmark=True, rounding=10, **extra
    )
    capped = getattr(portfolio_module, method)(
        current_weights=current,
        maximum_turnover=0.1,
        include_benchmark=True,
        rounding=10,
        **extra,
    )
    assert free.loc["Portfolio", "Turnover"] > 0.1  # noqa: PLR2004
    assert capped.loc["Portfolio", "Turnover"] == pytest.approx(0.1, abs=1e-7)
    assert capped.loc["Portfolio", "Weight"] == pytest.approx(1)


@pytest.mark.parametrize(
    ("method", "column", "extra"),
    [
        ("get_maximum_omega_weights", "Omega Ratio", {}),
        (
            "get_mean_risk_weights",
            "CVaR Ratio",
            {"risk_measure": "cvar", "objective": "maximum_ratio"},
        ),
        (
            "get_nested_clustered_optimization_weights",
            None,
            {"objective": "maximum_sharpe"},
        ),
    ],
)
def test_the_ratio_optimizers_take_an_expected_return_estimator(
    portfolio_module, method, column, extra
):
    mean = getattr(portfolio_module, method)(include_benchmark=True, **extra)
    shrunk = getattr(portfolio_module, method)(
        expected_returns="bayes_stein", include_benchmark=True, **extra
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


def test_get_risk_contributions_of_historical_measures(recorder, portfolio_module):
    for risk_measure in ("semideviation", "mad", "cdar"):
        contributions = portfolio_module.get_risk_contributions(
            weights={"AAPL": 0.7, "MSFT": 0.3}, risk_measure=risk_measure, rounding=10
        )
        recorder.capture(contributions)
        assert contributions.loc["Total", "Percentage Contribution"] == pytest.approx(1)
        assert contributions.loc["Total", "Risk Contribution"] == pytest.approx(
            contributions["Risk Contribution"].drop("Total").sum()
        )


def test_get_portfolio_risk_metrics(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_portfolio_risk_metrics())
    metrics = portfolio_module.get_portfolio_risk_metrics(
        weights={"AAPL": 0.7, "MSFT": 0.3}, period="weekly", alpha=0.1
    )
    recorder.capture(metrics)
    assert list(metrics.columns) == ["Portfolio", "Benchmark"]


def test_get_discrete_allocation(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_discrete_allocation())
    allocation = portfolio_module.get_discrete_allocation(
        weights={"AAPL": 0.7, "MSFT": 0.3}, portfolio_value=2000, method="greedy"
    )
    recorder.capture(allocation)
    assert allocation.loc["Total", "Amount"] == pytest.approx(2000)
    # The latest traded closing price, not the dividend-adjusted one.
    closes = portfolio_module._historical_data["daily"]["Close"]
    assert allocation.loc["AAPL", "Price"] == pytest.approx(closes["AAPL"].iloc[-1])


def test_get_discrete_allocation_invalid_method(portfolio_module):
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        portfolio_module.get_discrete_allocation(method="round")


def test_the_ledoit_wolf_targets_shrink_the_sample_covariance(portfolio_module):
    # With two assets the constant correlation target is the sample covariance itself.
    returns = portfolio_module._get_returns("daily", include_benchmark=True)
    benchmark = portfolio_module._get_benchmark_returns("daily")
    sample = returns.cov(ddof=0).to_numpy() * 252

    for covariance in ("ledoit_wolf_constant_correlation", "ledoit_wolf_single_factor"):
        _, _, matrix, _ = portfolio_module._get_inputs(
            "daily", True, covariance=covariance
        )
        # The targets keep every variance, so only the covariances move.
        assert np.allclose(np.diag(matrix), np.diag(sample))
        assert not np.allclose(matrix, sample)

    _, factor = optimization_model.get_ledoit_wolf_single_factor_covariance(
        returns, benchmark
    )
    assert 0 < factor <= 1


def test_get_mean_variance_weights_with_a_target_volatility(portfolio_module):
    table = portfolio_module.get_mean_variance_weights(
        target_volatility=0.35, rounding=10
    )
    assert table.loc["Portfolio", "Volatility"] == pytest.approx(0.35, abs=1e-6)
    assert table.loc["Portfolio", "Weight"] == pytest.approx(1)

    # A target below the lowest volatility, or with a target return, is refused.
    assert portfolio_module.get_mean_variance_weights(target_volatility=0.01).empty
    assert portfolio_module.get_mean_variance_weights(
        target_return=0.25, target_volatility=0.35
    ).empty


@pytest.mark.parametrize(
    "method",
    [
        "get_hierarchical_risk_parity_weights",
        "get_hierarchical_equal_risk_contribution_weights",
        "get_nested_clustered_optimization_weights",
    ],
)
def test_the_clustering_optimizers_take_a_codependence(portfolio_module, method):
    default = getattr(portfolio_module, method)(include_benchmark=True)
    pearson = getattr(portfolio_module, method)(
        codependence="pearson", include_benchmark=True
    )
    pd.testing.assert_frame_equal(default, pearson)

    for codependence in ("spearman", "kendall"):
        weights = getattr(portfolio_module, method)(
            codependence=codependence, include_benchmark=True
        )["Weight"]
        assert weights.drop("Portfolio").sum() == pytest.approx(1, abs=1e-4)

    with pytest.raises(InvalidArgumentError, match="codependence must be one of"):
        getattr(portfolio_module, method)(codependence="euclidean")


def test_black_litterman_with_the_market_risk_aversion(portfolio_module):
    market = {"AAPL": 2.0, "MSFT": 1.5}
    table = portfolio_module.get_black_litterman_weights(
        market_weights=market, risk_aversion="market", rounding=10
    )
    returns, _, covariance, risk_free_rate = portfolio_module._get_inputs(
        "daily", False
    )
    benchmark = portfolio_module._get_benchmark_returns("daily").reindex(returns.index)
    delta = (benchmark.mean() * 252 - risk_free_rate) / (benchmark.var() * 252)
    weights = pd.Series(market) / sum(market.values())

    assert np.allclose(
        table["Implied Return"],
        delta * covariance @ weights + risk_free_rate,
    )
    # Without views the weights are the market weights, and the posterior volatility is
    # that of (1 + tau) times the covariance.
    assert np.allclose(table["Weight"], weights, atol=1e-5)
    assert np.allclose(
        table["Posterior Volatility"], np.sqrt(1.05 * np.diag(covariance))
    )
    assert portfolio_module.get_black_litterman_weights(
        market_weights=market, risk_aversion="equilibrium"
    ).empty


def test_an_asset_without_weight_does_not_shorten_the_sample(test_toolkit):
    # A third asset listed halfway through the range, as a ticker with a short history.
    construction = _offline_construction(test_toolkit)
    data = construction._historical_data["daily"]
    returns = data["Return"]
    late = returns["MSFT"].copy()
    late.iloc[: len(late) // 2] = np.nan
    construction._historical_data = {
        "daily": pd.concat(
            [data, pd.concat({"Return": late.to_frame("LATE")}, axis=1)], axis=1
        )
    }
    construction._tickers = [*construction._tickers, "LATE"]
    weights = {"AAPL": 0.5, "MSFT": 0.5}

    metrics = construction.get_portfolio_risk_metrics(weights=weights, rounding=12)
    portfolio = returns[["AAPL", "MSFT"]].dropna() @ pd.Series(weights)
    assert metrics.loc["Annualized Volatility", "Portfolio"] == pytest.approx(
        portfolio.std() * np.sqrt(252)
    )

    contributions = construction.get_risk_contributions(weights=weights, rounding=12)
    assert contributions.index.tolist() == ["AAPL", "MSFT", "Total"]
    assert contributions.loc["Total", "Risk Contribution"] == pytest.approx(
        portfolio.std() * np.sqrt(252)
    )

    # An optimizer chooses among every loaded asset, so all of them decide the periods.
    assert len(construction._get_returns("daily", False)) == late.notna().sum()


def test_get_brinson_attribution_keeps_a_sector_whose_fund_starts_later(
    test_toolkit, caplog
):
    construction = _offline_construction(test_toolkit)
    prices = construction._price_function

    def late_healthcare_fund(tickers, start, end):
        funds = prices(tickers, start, end)
        funds.loc[: pd.Period("2021-01-04", freq="D"), "XLV"] = np.nan
        return funds

    complete = construction.get_brinson_attribution(rounding=16)
    construction._price_function = late_healthcare_fund
    with caplog.at_level("WARNING", logger="financetoolkit"):
        attribution = construction.get_brinson_attribution(rounding=16)

    assert "Healthcare (XLV, first price on 2021-01-05)" in caplog.text
    # Healthcare keeps its weight of 0.6 and has no return over the range, rather than the
    # return of its fund from 2021-01-05.
    assert attribution.loc["Healthcare", "Benchmark Weight"] == pytest.approx(0.6)
    assert np.isnan(attribution.loc["Healthcare", "Benchmark Return"])
    assert attribution.loc["Technology", "Benchmark Return"] == pytest.approx(
        complete.loc["Technology", "Benchmark Return"]
    )
    assert attribution.loc["Total", ["Benchmark Return", "Total Effect"]].isna().all()
    assert attribution.loc["Total", "Portfolio Return"] == pytest.approx(
        complete.loc["Total", "Portfolio Return"]
    )

    # Linked over years, 2021 and 2022 have a Healthcare return but 2020 does not, so the
    # compounded benchmark return and the linked effects stay unknown.
    linked = construction.get_brinson_attribution(period="yearly", rounding=16)
    assert linked.loc["Healthcare", "Benchmark Weight"] == pytest.approx(0.6)
    assert np.isnan(linked.loc["Healthcare", "Benchmark Return"])
    assert linked.loc["Total", "Allocation Effect":].isna().all()

    # From the fund's first price onwards the attribution is complete.
    construction._start_date = "2021-01-05"
    later = construction.get_brinson_attribution(period="yearly", rounding=16)
    assert later.notna().loc["Total"].all()
    assert later.loc["Total", "Total Effect"] == pytest.approx(
        later.loc["Total", "Portfolio Return"] - later.loc["Total", "Benchmark Return"],
        abs=1e-12,
    )


def test_get_expected_returns_estimate(recorder, portfolio_module):
    table = portfolio_module.get_expected_returns_estimate(include_benchmark=True)
    recorder.capture(table)
    recorder.capture(
        portfolio_module.get_expected_returns_estimate(
            method=["exponential", "exponential_geometric"], span=60, adjust=False
        )
    )
    weights = portfolio_module.get_mean_variance_weights(
        expected_returns="james_stein", include_benchmark=True
    )
    assert np.allclose(
        weights["Expected Return"].drop("Portfolio"), table["James-Stein"], atol=1e-4
    )


def test_get_covariance_estimate(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_covariance_estimate(include_benchmark=True))
    for method in ("shrunk", "gerber", "denoised_shrink", "detoned"):
        recorder.capture(
            portfolio_module.get_covariance_estimate(
                method=method, include_benchmark=True, rounding=6
            )
        )
    correlation = portfolio_module.get_covariance_estimate(
        method="gerber", gerber_statistic=2, correlation=True, include_benchmark=True
    )
    assert np.allclose(np.diag(correlation), 1)


def test_get_codependence_matrix(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_codependence_matrix(include_benchmark=True))
    for method in ("distance", "mutual_information", "lower_tail"):
        recorder.capture(
            portfolio_module.get_codependence_matrix(
                method=method, include_benchmark=True
            )
        )
    distance = portfolio_module.get_codependence_matrix(
        method="mutual_information", distance=True, include_benchmark=True
    )
    recorder.capture(distance)
    assert np.allclose(np.diag(distance), 0)


def test_get_asset_clusters(recorder, portfolio_module):
    recorder.capture(portfolio_module.get_asset_clusters(include_benchmark=True))
    recorder.capture(
        portfolio_module.get_asset_clusters(
            method="ftca", threshold=0.7, include_benchmark=True
        )
    )
    recorder.capture(
        portfolio_module.get_asset_clusters(
            cluster_criterion="standardized_silhouette", include_benchmark=True
        )
    )
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        portfolio_module.get_asset_clusters(method="k_means")


def test_get_coskewness_and_cokurtosis_matrix(recorder, portfolio_module):
    coskewness = portfolio_module.get_coskewness_matrix()
    recorder.capture(coskewness)
    recorder.capture(portfolio_module.get_coskewness_matrix(method="exponential"))
    cokurtosis = portfolio_module.get_cokurtosis_matrix()
    recorder.capture(cokurtosis)
    recorder.capture(portfolio_module.get_cokurtosis_matrix(method="semi"))
    # The diagonal of the standardized matrices is each asset's moment skewness and
    # (non-excess) kurtosis.
    returns = portfolio_module._get_returns("daily", False)
    assert coskewness.loc["AAPL", ("AAPL", "AAPL")] == pytest.approx(
        returns["AAPL"].skew()
        * (len(returns) - 2)
        / np.sqrt(len(returns) * (len(returns) - 1)),
        abs=1e-4,
    )
    assert cokurtosis.shape == (2, 8)


def test_get_estimation_uncertainty(recorder, portfolio_module):
    table = portfolio_module.get_estimation_uncertainty(n_bootstrap=200)
    recorder.capture(table)
    assert (table["Lower"] < table["Expected Return"]).all()
    assert (table["Upper"] > table["Expected Return"]).all()
    recorder.capture(
        portfolio_module.get_estimation_uncertainty(
            method="normal", n_bootstrap=200, output="covariance", rounding=6
        )
    )


@pytest.mark.parametrize(
    "expected_returns",
    ["geometric", "exponential_geometric", "james_stein", "bodnar_okhrin_parolya"],
)
def test_the_optimizers_take_the_shrunk_and_compounded_expected_returns(
    portfolio_module, expected_returns
):
    table = portfolio_module.get_mean_variance_weights(
        expected_returns=expected_returns, include_benchmark=True, rounding=8
    )
    assert table["Weight"].drop("Portfolio").sum() == pytest.approx(1, abs=1e-4)
    assert np.allclose(
        table["Expected Return"].drop("Portfolio"),
        portfolio_module.get_expected_returns_estimate(
            method=expected_returns, include_benchmark=True, rounding=8
        ).iloc[:, 0],
    )


@pytest.fixture
def fama_french(monkeypatch, portfolio_module):
    """
    Offline daily Fama and French five factors (and the risk free rate) on the days of the
    test data: the market factor follows the Benchmark, the others are noise.
    """
    from financetoolkit.performance import performance_model  # noqa: PLC0415

    returns = portfolio_module._historical_data["daily"]["Return"].fillna(0)
    rng = np.random.default_rng(11)
    factors = pd.DataFrame(
        rng.normal(0.0001, 0.005, (len(returns), 4)),
        index=returns.index,
        columns=["SMB", "HML", "RMW", "CMA"],
    )
    factors.insert(
        0, "Mkt-RF", returns["Benchmark"] + rng.normal(0, 0.002, len(returns))
    )
    factors["HML"] += 0.3 * returns["MSFT"]
    factors["RF"] = 0.0001
    monkeypatch.setattr(
        performance_model, "obtain_fama_and_french_dataset", lambda **_: factors
    )
    portfolio_module._factor_datasets.clear()
    yield factors.drop(columns="RF")
    portfolio_module._factor_datasets.clear()


def test_get_factor_loadings(recorder, portfolio_module, fama_french):
    table = portfolio_module.get_factor_loadings(include_benchmark=True, rounding=6)
    recorder.capture(table)
    # The Benchmark is the market factor without the noise.
    assert table.loc["Benchmark", "Mkt-RF"] == pytest.approx(1, abs=0.05)
    recorder.capture(
        portfolio_module.get_factor_loadings(
            selection="forward", period="monthly", rounding=6
        )
    )
    recorder.capture(portfolio_module.get_factor_loadings(selection="pcr", rounding=6))


def test_get_covariance_estimate_factor(recorder, portfolio_module, fama_french):
    covariance = portfolio_module.get_covariance_estimate(method="factor", rounding=8)
    recorder.capture(covariance)
    recorder.capture(
        portfolio_module.get_covariance_estimate(
            method="factor", factor_selection="backward", rounding=8
        )
    )
    weights = portfolio_module.get_minimum_variance_weights(
        covariance="factor", rounding=8
    )
    assert weights["Weight"].drop("Portfolio").sum() == pytest.approx(1, abs=1e-6)


def test_get_covariance_estimate_robust(recorder, portfolio_module):
    for method in (
        "minimum_covariance_determinant",
        "jlogo",
        "geodesic",
        "regime_adjusted_ew",
    ):
        recorder.capture(
            portfolio_module.get_covariance_estimate(
                method=method, include_benchmark=True, rounding=8
            )
        )
    recorder.capture(
        portfolio_module.get_covariance_estimate(
            method="regime_adjusted_ew", half_life=20, rounding=8
        )
    )
    # With three assets the TMFG links every pair, so j-LoGo is the sample covariance.
    assert np.allclose(
        portfolio_module.get_covariance_estimate(
            method="jlogo", include_benchmark=True, rounding=10
        ),
        portfolio_module.get_covariance_estimate(include_benchmark=True, rounding=10),
    )
    # The robust estimate is reproducible and feeds the optimizers.
    weights = portfolio_module.get_minimum_variance_weights(
        covariance="minimum_covariance_determinant", period="weekly", rounding=8
    )
    assert weights["Weight"].drop("Portfolio").sum() == pytest.approx(1, abs=1e-6)


def test_get_risk_contributions_finite_differences(recorder, portfolio_module):
    for measure in ("maximum_drawdown", "kurtosis", "evar"):
        table = portfolio_module.get_risk_contributions(
            weights={"AAPL": 0.7, "MSFT": 0.3}, risk_measure=measure, rounding=8
        )
        recorder.capture(table)
        assert table.loc["Total", "Percentage Contribution"] == pytest.approx(1)


def test_get_factor_risk_contributions(recorder, portfolio_module, fama_french):
    table = portfolio_module.get_factor_risk_contributions(rounding=8)
    recorder.capture(table)
    total = portfolio_module.get_risk_contributions(rounding=8)
    assert table.loc["Total", "Risk Contribution"] == pytest.approx(
        total.loc["Total", "Risk Contribution"], rel=1e-5
    )
    recorder.capture(
        portfolio_module.get_factor_risk_contributions(
            weights={"AAPL": 0.7, "MSFT": 0.3}, risk_measure="cvar", rounding=8
        )
    )


def test_get_factor_attribution(recorder, portfolio_module, fama_french):
    table = portfolio_module.get_factor_attribution(rounding=8)
    recorder.capture(table.iloc[:20])
    assert np.allclose(
        table["Common Return"] + table["Specific Return"], table["Total Return"]
    )
    recorder.capture(
        portfolio_module.get_factor_attribution(
            weights={"AAPL": 0.7, "MSFT": 0.3}, period="monthly", rounding=8
        )
    )


def test_get_entropy_pooling(recorder, portfolio_module):
    table = portfolio_module.get_entropy_pooling(
        views={"AAPL": 0.05}, volatility_views={"MSFT": 0.3}, rounding=6
    )
    recorder.capture(table)
    # The posterior covariance is on the basis the views are imposed on, so the volatility
    # view comes back exactly, on daily as on monthly returns.
    assert table.loc["MSFT", "Posterior Volatility"] == pytest.approx(0.3, abs=1e-6)
    monthly = portfolio_module.get_entropy_pooling(
        volatility_views={"MSFT": 0.3}, period="monthly", rounding=10
    )
    assert monthly.loc["MSFT", "Posterior Volatility"] == pytest.approx(0.3, abs=1e-7)
    recorder.capture(
        portfolio_module.get_entropy_pooling(
            views={"AAPL": 0.05}, output="summary", rounding=6
        )
    )
    recorder.capture(
        portfolio_module.get_entropy_pooling(
            opinions=[{"views": {"AAPL": 0.05}}, {"views": {"MSFT": (">=", 0.4)}}],
            pooling="logarithmic",
            rounding=6,
        )
    )


def test_get_return_contributions(recorder, portfolio_module):
    table = portfolio_module.get_return_contributions(period="yearly", rounding=8)
    recorder.capture(table)
    assert np.allclose(
        table.drop(columns="Portfolio").sum(axis=1), table["Portfolio"], atol=1e-7
    )
    recorder.capture(
        portfolio_module.get_return_contributions(
            weights={"AAPL": 0.7, "MSFT": 0.3}, period="quarterly", rounding=8
        )
    )


def test_get_black_litterman_weights_with_factor_views(
    recorder, portfolio_module, fama_french
):
    market = {"AAPL": 2.0, "MSFT": 1.5}
    recorder.capture(
        portfolio_module.get_black_litterman_weights(
            market_weights=market,
            method="augmented",
            views={"MSFT": 0.2},
            factor_views={"HML": 0.03},
        )
    )
    recorder.capture(
        portfolio_module.get_black_litterman_weights(
            method="bayesian", factor_views={"HML": 0.03, "SMB-HML": 0.01}
        )
    )
    # Factor views need the augmented or Bayesian method.
    assert portfolio_module.get_black_litterman_weights(
        market_weights=market, factor_views={"HML": 0.03}
    ).empty


def test_get_critical_line_weights(recorder, portfolio_module):
    table = portfolio_module.get_critical_line_weights(include_benchmark=True)
    recorder.capture(table)
    # The minimum variance portfolio is the last corner and the least volatile one.
    assert (
        table.loc["Minimum Variance", "Volatility"] <= table["Volatility"].min() + 1e-4
    )
    assert (
        table.loc["Maximum Sharpe", "Sharpe Ratio"]
        >= table["Sharpe Ratio"].max() - 1e-4
    )
    recorder.capture(
        portfolio_module.get_critical_line_weights(
            maximum_weight=0.5, include_benchmark=True
        )
    )


def test_get_worst_case_mean_variance_weights(recorder, portfolio_module):
    table = portfolio_module.get_worst_case_mean_variance_weights(
        confidence_level=0.5, include_benchmark=True
    )
    recorder.capture(table)
    assert (table["Worst-Case Return"] <= table["Expected Return"] + 1e-4).all()
    recorder.capture(
        portfolio_module.get_worst_case_mean_variance_weights(
            uncertainty="ellipsoid", objective="utility", include_benchmark=True
        )
    )
    target = portfolio_module.get_worst_case_mean_variance_weights(
        objective="minimum_variance", target_return=-0.16, include_benchmark=True
    )
    recorder.capture(target)
    assert target.loc["Portfolio", "Worst-Case Return"] >= -0.16 - 1e-4


def test_get_distributionally_robust_cvar_weights(recorder, portfolio_module):
    table = portfolio_module.get_distributionally_robust_cvar_weights(
        include_benchmark=True
    )
    recorder.capture(table)
    assert (table["Worst-Case CVaR"] >= table["CVaR"] - 1e-4).all()
    recorder.capture(
        portfolio_module.get_distributionally_robust_cvar_weights(
            wasserstein_radius=0.0, risk_aversion=5.0, include_benchmark=True
        )
    )


def test_hierarchical_weights_on_the_dbht(recorder, portfolio_module):
    # Three assets form one bubble of the TMFG, so the DBHT holds a single cluster.
    herc = portfolio_module.get_hierarchical_equal_risk_contribution_weights(
        linkage_method="dbht", include_benchmark=True
    )
    recorder.capture(herc)
    assert (herc["Cluster"].drop("Portfolio") == 1).all()
    recorder.capture(
        portfolio_module.get_hierarchical_risk_parity_weights(
            linkage_method="dbht", codependence="spearman", include_benchmark=True
        )
    )
    recorder.capture(
        portfolio_module.get_nested_clustered_optimization_weights(
            linkage_method="dbht", include_benchmark=True
        )
    )
    recorder.capture(
        portfolio_module.get_schur_complementary_weights(
            linkage_method="dbht", include_benchmark=True
        )
    )
    recorder.capture(
        portfolio_module.get_asset_clusters(
            linkage_method="dbht", codependence="distance", include_benchmark=True
        )
    )


def _synthetic_construction(
    index: pd.PeriodIndex, assets: int = 3, end_date: str | None = None
) -> tuple[PortfolioConstruction, pd.DataFrame]:
    """A Portfolio Construction module on random daily returns and a 3% risk free rate."""
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        rng.normal(0.0008, 0.015, (len(index), assets)),
        index=index,
        columns=["A", "B", "C"][:assets],
    )
    construction = PortfolioConstruction(
        tickers=returns.columns.tolist(),
        historical_data={"daily": pd.concat({"Return": returns}, axis=1)},
        risk_free_rate_data={"daily": pd.Series(0.03, index=index)},
        rounding=None,
        end_date=end_date,
    )
    return construction, returns


def test_without_company_profiles_the_error_names_the_api_key(monkeypatch):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    construction, _ = _synthetic_construction(
        pd.bdate_range("2022-01-03", periods=300).to_period("D")
    )
    calls = []
    construction._profile_function = lambda: calls.append(1)
    construction._fund_function = lambda by, assets: None
    cases = [
        (
            lambda: construction.get_black_litterman_weights(views={"A": 0.1}),
            "market_weights",
        ),
        (
            lambda: construction.get_minimum_variance_weights(
                sector_bounds={"Technology": (0, 0.4)}
            ),
            "minimum_weight and maximum_weight",
        ),
        (
            lambda: construction.get_minimum_variance_weights(
                country_bounds={"US": (0, 0.4)}
            ),
            "minimum_weight and maximum_weight",
        ),
        (construction.get_look_through_exposure, 'by="holdings"'),
    ]
    for call, alternative in cases:
        with pytest.raises(ValueError, match="API key") as error:
            call()
        assert alternative in str(error.value)
    # Nothing is cached, so every call asks for the profiles again.
    assert len(calls) == len(cases)
    assert construction._profile is None
    # The alternatives work without the profiles.
    assert not construction.get_black_litterman_weights(
        views={"A": 0.1}, market_weights={"A": 1, "B": 1, "C": 1}
    ).empty
    assert not construction.get_look_through_exposure(by="holdings").empty


def test_return_contributions_start_when_every_asset_held_has_a_return():
    index = pd.bdate_range("2022-01-03", periods=300).to_period("D")
    construction, returns = _synthetic_construction(index)
    # C is listed on the 101st day; B has no trade on one day after that.
    returns.iloc[:100, 2] = np.nan
    returns.iloc[150, 1] = np.nan
    construction._historical_data["daily"] = pd.concat({"Return": returns}, axis=1)
    table = construction.get_return_contributions(
        weights={"A": 0.5, "B": 0.3, "C": 0.2}, period="daily"
    )
    # The days before C's listing are left out instead of counting as returns of 0.
    assert table.index[0] == returns.index[100]
    assert len(table) == len(returns) - 100
    np.testing.assert_allclose(
        table["Portfolio"], returns.iloc[100:].fillna(0) @ [0.5, 0.3, 0.2]
    )
    assert table.loc[returns.index[150], "B"] == 0
    # Without C the sample is the whole range.
    assert (
        len(
            construction.get_return_contributions(
                weights={"A": 0.5, "B": 0.5}, period="daily"
            )
        )
        == len(returns) - 1
    )


def test_a_net_exposure_other_than_one_holds_the_rest_in_cash():
    construction, _ = _synthetic_construction(
        pd.bdate_range("2022-01-03", periods=300).to_period("D")
    )
    assets = ["A", "B", "C"]
    full = construction.get_maximum_sharpe_weights(maximum_weight=2.0)

    for net_exposure in (0.5, 1.5):
        table = construction.get_maximum_sharpe_weights(
            maximum_weight=2.0, net_exposure=net_exposure
        )
        weights = table.loc[assets, "Weight"]
        np.testing.assert_allclose(
            weights, net_exposure * full.loc[assets, "Weight"], atol=1e-6
        )
        assert table.loc["Cash", "Weight"] == pytest.approx(1 - net_exposure)
        assert table.loc["Cash", "Expected Return"] == pytest.approx(0.03)
        assert table.loc["Portfolio", "Weight"] == pytest.approx(1)
        assert table.loc["Portfolio", "Expected Return"] == pytest.approx(
            weights @ table.loc[assets, "Expected Return"] + (1 - net_exposure) * 0.03
        )
        # The same mix of assets has the same Sharpe ratio, the one maximized.
        assert table.loc["Portfolio", "Sharpe Ratio"] == pytest.approx(
            full.loc["Portfolio", "Sharpe Ratio"], rel=1e-6
        )
    assert "Cash" not in full.index


def test_a_market_neutral_minimum_risk_portfolio_holds_nothing():
    construction, _ = _synthetic_construction(
        pd.bdate_range("2022-01-03", periods=300).to_period("D")
    )
    assets = ["A", "B", "C"]
    neutral = {"minimum_weight": -1.0, "net_exposure": 0.0, "gross_exposure": 2.0}
    tables = [
        construction.get_minimum_variance_weights(**neutral),
        construction.get_mean_risk_weights(risk_measure="mad", **neutral),
        construction.get_mean_risk_weights(risk_measure="cvar", **neutral),
        construction.get_distributionally_robust_cvar_weights(**neutral),
    ]
    for table in tables:
        # The least risky market neutral portfolio holds nothing: it has no volatility to
        # share out, so the risk contributions and the Sharpe ratio are NaN.
        np.testing.assert_allclose(table.loc[assets, "Weight"], 0, atol=1e-6)
        assert table.loc["Cash", "Weight"] == pytest.approx(1)
        assert table.loc["Portfolio", "Volatility"] == pytest.approx(0, abs=1e-9)
        assert table["Risk Contribution"].isna().all()
        assert np.isnan(table.loc["Portfolio", "Sharpe Ratio"])


def test_the_transaction_cost_is_shown_and_taken_from_the_expected_return():
    construction, _ = _synthetic_construction(
        pd.bdate_range("2022-01-03", periods=300).to_period("D")
    )
    current = {"A": 0.6, "B": 0.4}
    gross = construction.get_maximum_sharpe_weights(current_weights=current)
    table = construction.get_maximum_sharpe_weights(
        current_weights=current, transaction_cost=0.02
    )
    portfolio = table.loc["Portfolio"]
    assets = ["A", "B", "C"]
    cost = 0.02 * portfolio["Turnover"]
    before_cost = table.loc[assets, "Weight"] @ table.loc[assets, "Expected Return"]

    assert "Transaction Cost" not in gross.columns
    assert cost > 0
    assert portfolio["Transaction Cost"] == pytest.approx(cost)
    np.testing.assert_allclose(
        table.loc[assets, "Transaction Cost"], 0.02 * table.loc[assets, "Turnover"]
    )
    assert portfolio["Expected Return"] == pytest.approx(before_cost - cost)
    assert portfolio["Sharpe Ratio"] == pytest.approx(
        (before_cost - cost - 0.03) / portfolio["Volatility"]
    )


def test_returns_on_every_calendar_day_are_annualized_with_365_days():
    construction, returns = _synthetic_construction(
        pd.period_range("2023-01-01", "2024-12-31", freq="D"), assets=2
    )
    table = construction.get_maximum_sharpe_weights()
    np.testing.assert_allclose(
        table.loc[["A", "B"], "Expected Return"], returns.mean() * 365
    )

    options = {"methods": "equal_weight", "lookback": 20, "rebalancing": "yearly"}
    path = construction.get_optimization_backtest(output="returns", **options)[
        "Equal Weight"
    ].dropna()
    summary = construction.get_optimization_backtest(**options).loc["Equal Weight"]
    assert summary["Annualized Return"] == pytest.approx(
        (1 + path).prod() ** (365 / len(path)) - 1
    )
    assert summary["Annualized Volatility"] == pytest.approx(path.std() * np.sqrt(365))


def test_the_backtest_runs_to_the_end_of_the_data(caplog):
    index = pd.bdate_range("2022-01-03", periods=200).to_period("D")
    construction, returns = _synthetic_construction(index)
    daily = construction._historical_data["daily"]
    # One market is closed on the last day, the others are open.
    daily.loc[index[-1], ("Return", "A")] = np.nan
    options = {"methods": "equal_weight", "lookback": 40, "output": "returns"}

    path = construction.get_optimization_backtest(**options)["Equal Weight"]
    assert path.index[-1] == index[-1]
    # Equal weights from the last month end, drifted, with A at 0% on the last day.
    held = daily["Return"].loc[path.index[path.index > "2022-09-30"]].fillna(0)
    value = (1 + held).cumprod().sum(axis=1)
    assert path.iloc[-1] == pytest.approx(value.iloc[-1] / value.iloc[-2] - 1)
    assert "has no returns after" not in caplog.text

    # An asset whose returns stop months before the end is held at 0% and said so.
    daily.loc[index[-80:], ("Return", "C")] = np.nan
    path = construction.get_optimization_backtest(**options)["Equal Weight"]
    assert path.index[-1] == index[-1]
    assert "C has no returns after" in caplog.text


def test_a_backtest_strategy_whose_bounds_cannot_be_met_has_no_returns(caplog):
    construction, _ = _synthetic_construction(
        pd.bdate_range("2022-01-03", periods=200).to_period("D")
    )
    options = {"methods": "minimum_variance", "lookback": 40, "maximum_weight": 0.3}

    summary = construction.get_optimization_backtest(**options)
    weights = construction.get_optimization_backtest(output="weights", **options)

    assert summary.loc["Minimum Variance"].isna().all()
    assert weights["Minimum Variance"].isna().all().all()
    assert summary.loc["Equal Weight"].notna().any()
    assert "Minimum Variance found no weights on any of the" in caplog.text


def test_get_covariance_estimate_split_and_robust(recorder, portfolio_module):
    for method in ("denoised_detoned", "nercome", "mm", "rocke"):
        covariance = portfolio_module.get_covariance_estimate(
            method=method, include_benchmark=True, rounding=8
        )
        recorder.capture(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance.to_numpy())
        if method == "denoised_detoned":
            # The market component is removed, so the matrix is singular.
            assert eigenvalues.min() == pytest.approx(0, abs=1e-6)
        else:
            assert eigenvalues.min() > 0
    # The robust estimates feed the optimizers.
    weights = portfolio_module.get_minimum_variance_weights(
        covariance="rocke", include_benchmark=True
    )
    assert weights["Weight"].drop("Portfolio").sum() == pytest.approx(1, abs=1e-6)


def test_get_covariance_estimate_ccc_garch(recorder, portfolio_module):
    # Rounded to 6 decimals: the GARCH legs come from a multi-start likelihood fit whose
    # optimum moves by about 1e-8 (relative) under 1e-12 changes of the returns.
    covariance = portfolio_module.get_covariance_estimate(
        method="ccc_garch", rounding=6
    )
    recorder.capture(covariance)
    coskewness = portfolio_module.get_coskewness_matrix(method="ccc_garch", rounding=6)
    recorder.capture(coskewness)
    # The constant correlations are those of the standardized residuals, below the
    # sample correlation of returns that share volatility bursts.
    correlation = covariance.loc["AAPL", "MSFT"] / np.sqrt(
        covariance.loc["AAPL", "AAPL"] * covariance.loc["MSFT", "MSFT"]
    )
    assert 0 < correlation < 1


def test_get_random_matrix_statistics(recorder, portfolio_module):
    summary = portfolio_module.get_random_matrix_statistics(include_benchmark=True)
    recorder.capture(summary)
    eigenvalues = portfolio_module.get_random_matrix_statistics(
        output="eigenvalues", include_benchmark=True
    )
    recorder.capture(eigenvalues)
    assert eigenvalues["Eigenvalue"].sum() == pytest.approx(3, abs=1e-3)
    assert summary["Signal Eigenvalues"].iloc[0] == eigenvalues["Signal"].sum()
    with pytest.raises(InvalidArgumentError, match="output must be one of"):
        portfolio_module.get_random_matrix_statistics(output="pdf")


def test_get_expected_returns_estimate_centroid(recorder, portfolio_module):
    table = portfolio_module.get_expected_returns_estimate(
        ranking=["MSFT", "Benchmark", "AAPL"], include_benchmark=True
    )
    recorder.capture(table)
    assert table["Centroid"]["MSFT"] > 0 > table["Centroid"]["AAPL"]
    assert table["Centroid"]["Benchmark"] == pytest.approx(0)
    recorder.capture(
        portfolio_module.get_expected_returns_estimate(
            method="centroid",
            ranking={"AAPL": 2.0, "MSFT": 2.0, "Benchmark": 1.0},
            include_benchmark=True,
        )
    )
    # Without a ranking every other estimator is shown, and the centroid alone is empty.
    assert "Centroid" not in portfolio_module.get_expected_returns_estimate().columns
    assert portfolio_module.get_expected_returns_estimate(method="centroid").empty


def test_get_codependence_matrix_angular_and_gnpr(recorder, portfolio_module):
    for method in ("absolute_angular", "squared_angular", "gnpr"):
        recorder.capture(
            portfolio_module.get_codependence_matrix(
                method=method, include_benchmark=True
            )
        )
    distance = portfolio_module.get_codependence_matrix(
        method="gnpr", gnpr_theta=1.0, distance=True, include_benchmark=True
    )
    recorder.capture(distance)
    assert np.allclose(np.diag(distance), 0)


def test_get_asset_clusters_onc(recorder, portfolio_module):
    clusters = portfolio_module.get_asset_clusters(method="onc", include_benchmark=True)
    recorder.capture(clusters)
    for seed in (1, 2):
        assert clusters.equals(
            portfolio_module.get_asset_clusters(
                method="onc", seed=seed, include_benchmark=True
            )
        )
    weights = portfolio_module.get_nested_clustered_optimization_weights(
        cluster_criterion="onc", include_benchmark=True
    )
    recorder.capture(weights)
    assert (weights["Cluster"].drop("Portfolio") == clusters["Cluster"]).all()
    assert portfolio_module.get_hierarchical_equal_risk_contribution_weights(
        cluster_criterion="onc"
    ).empty


def test_get_entropy_pooling_ranking(recorder, portfolio_module):
    # The history has AAPL above MSFT; the opposite ranking pulls them level.
    table = portfolio_module.get_entropy_pooling(ranking=["MSFT", "AAPL"], rounding=6)
    recorder.capture(table)
    assert (
        table.loc["MSFT", "Posterior Return"] >= table.loc["AAPL", "Posterior Return"]
    )
    unchanged = portfolio_module.get_entropy_pooling(
        ranking=["AAPL", "MSFT"], output="summary"
    )
    assert unchanged["Relative Entropy"].iloc[0] == pytest.approx(0, abs=1e-10)
    recorder.capture(
        portfolio_module.get_entropy_pooling(
            opinions=[{"ranking": ["MSFT", "AAPL"]}, {"views": {"AAPL": 0.3}}],
            rounding=6,
        )
    )


@pytest.fixture(scope="module")
def constant_price_portfolio():
    """The offline Toolkit's portfolio module with a constant price for MSFT."""
    from financetoolkit import Toolkit

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle").drop(
        columns=["Return", "Cumulative Return"], level=0
    )
    for column in ["Open", "High", "Low", "Close", "Adj Close"]:
        historical[(column, "MSFT")] = 100.0
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")
    return toolkit.portfolio


@pytest.mark.parametrize(
    "method",
    [
        "get_minimum_variance_weights",
        "get_mean_variance_weights",
        "get_mean_risk_weights",
        "get_distributionally_robust_cvar_weights",
        "get_optimization_backtest",
    ],
)
def test_an_asset_without_volatility_has_no_sharpe_ratio(
    constant_price_portfolio, method
):
    table = getattr(constant_price_portfolio, method)()
    numbers = table.select_dtypes("number").to_numpy(float)

    assert not table.empty
    assert not np.isinf(numbers).any()


@pytest.mark.parametrize(
    ("method", "arguments", "name"),
    [
        ("get_minimum_variance_weights", {"covariance": "bogus"}, "covariance"),
        ("get_optimization_backtest", {"covariance": "bogus"}, "covariance"),
        ("get_codependence_matrix", {"covariance": "bogus"}, "covariance"),
        (
            "get_hierarchical_risk_parity_weights",
            {"linkage_method": "bogus"},
            "linkage_method",
        ),
        (
            "get_schur_complementary_weights",
            {"linkage_method": "bogus"},
            "linkage_method",
        ),
        ("get_stacking_weights", {"final_method": "bogus"}, "final_method"),
        ("get_stacking_weights", {"methods": "minimum_variance,bogus"}, "methods"),
        ("get_covariance_estimate", {"factor_selection": "bogus"}, "factor_selection"),
        ("get_entropy_pooling", {"pooling": "bogus"}, "pooling"),
        ("get_codependence_matrix", {"bins": "bogus"}, "bins"),
        ("get_codependence_matrix", {"alpha": 1.5}, "alpha"),
    ],
)
def test_an_unknown_option_raises(portfolio_module, method, arguments, name):
    with pytest.raises(InvalidArgumentError, match=name):
        getattr(portfolio_module, method)(**arguments)


def test_equal_weights_are_not_reported_as_rescaled(portfolio_module, caplog):
    with caplog.at_level("INFO"):
        portfolio_module.get_risk_contributions()

    assert "are scaled to one" not in caplog.text


def test_a_missing_api_key_is_reported_once(portfolio_module, caplog):
    with caplog.at_level("ERROR"):
        assert portfolio_module.get_look_through_exposure().empty

    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(errors) == 1
    assert "Financial Modeling Prep API key" in errors[0].getMessage()
