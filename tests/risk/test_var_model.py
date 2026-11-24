"""Value at Risk Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.risk.value_at_risk import var_model

# How close Cornish-Fisher VaR must sit to gaussian VaR on near-normal data.
CLOSE_TO_GAUSSIAN_TOLERANCE = 0.001


def test_get_var_historic(recorder):
    recorder.capture(
        var_model.get_var_historic(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_var_historic_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.05],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.08, -0.03],
        }
    )
    recorder.capture(var_model.get_var_historic(returns=returns_df, alpha=0.05))


def test_get_var_historic_different_alphas(recorder):
    returns = pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15, -0.05, 0.08, 0.12])

    for alpha in [0.01, 0.05, 0.1, 0.2]:
        result = var_model.get_var_historic(returns=returns, alpha=alpha)
        recorder.capture(round(result, 4))


def test_get_var_gaussian(recorder):
    recorder.capture(
        round(
            var_model.get_var_cornish_fisher(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
                alpha=0.05,
            ),
            4,
        )
    )
    recorder.capture(
        round(
            var_model.get_var_gaussian(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
                alpha=0.03,
            ),
            4,
        )
    )


def test_get_var_gaussian_dataframe(recorder):
    returns_df = pd.DataFrame(
        {"AAPL": [0.3, 0.2, 0.1, 0, 0.06], "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04]}
    )

    result_cf = var_model.get_var_cornish_fisher(returns=returns_df, alpha=0.05)
    recorder.capture(result_cf.round(4))

    result_normal = var_model.get_var_gaussian(returns=returns_df, alpha=0.05)
    recorder.capture(result_normal.round(4))


def test_get_var_gaussian_negative_returns(recorder):
    returns = pd.Series([-0.1, -0.05, -0.02, 0.01, 0.03, -0.08])

    recorder.capture(
        round(
            var_model.get_var_cornish_fisher(returns=returns, alpha=0.05),
            4,
        )
    )


def test_get_var_studentt(recorder):
    recorder.capture_list(
        var_model.get_var_studentt(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.03
        ).round(4)
    )


def test_get_var_studentt_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.05],
        }
    )

    result = var_model.get_var_studentt(returns=returns_df, alpha=0.05)
    recorder.capture(result.round(4))


def test_get_var_historic_rolling(recorder):
    returns = pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15, -0.05, 0.08, 0.12])
    recorder.capture(
        var_model.get_var_historic(returns=returns, alpha=0.05, window=3).round(4)
    )


def test_get_var_historic_rolling_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.05],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.08, -0.03],
        }
    )
    recorder.capture(
        var_model.get_var_historic(returns=returns_df, alpha=0.05, window=3).round(4)
    )


def test_get_var_evt(recorder):
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.15, 0.01, -0.02, 0.04, -0.2, 0.02, -0.01, -0.18, 0.03]
    )
    recorder.capture(
        round(
            var_model.get_var_evt(
                returns=returns, alpha=0.05, threshold_percentile=0.7
            ),
            4,
        )
    )


def test_get_var_evt_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [
                0.02,
                -0.01,
                0.03,
                -0.15,
                0.01,
                -0.02,
                0.04,
                -0.2,
                0.02,
                -0.01,
                -0.18,
                0.03,
            ],
            "MSFT": [
                0.01,
                -0.02,
                0.02,
                -0.12,
                0.02,
                -0.01,
                0.03,
                -0.17,
                0.01,
                -0.02,
                -0.14,
                0.02,
            ],
        }
    )
    recorder.capture(
        var_model.get_var_evt(
            returns=returns_df, alpha=0.05, threshold_percentile=0.7
        ).round(4)
    )


def test_get_var_evt_insufficient_exceedances(recorder):
    # Too few exceedances to fit a GPD, so this returns NaN rather than raising.
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
    recorder.capture(
        var_model.get_var_evt(returns=returns, alpha=0.05, threshold_percentile=0.95)
    )


def test_var_edge_cases(recorder):
    # Test with very small dataset
    small_returns = pd.Series([0.01, 0.02])

    recorder.capture(var_model.get_var_historic(returns=small_returns, alpha=0.05))

    # Test with constant returns
    constant_returns = pd.Series([0.01] * 10)

    recorder.capture(
        round(
            var_model.get_var_gaussian(returns=constant_returns, alpha=0.05),
            4,
        )
    )


def test_get_var_cornish_fisher(recorder):
    recorder.capture(
        round(
            var_model.get_var_cornish_fisher(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
            ),
            4,
        )
    )


def test_get_var_cornish_fisher_dataframe(recorder):
    returns_df = pd.DataFrame(
        {"AAPL": [0.3, 0.2, 0.1, 0, 0.06], "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04]}
    )
    recorder.capture(
        var_model.get_var_cornish_fisher(returns=returns_df, alpha=0.05).round(4)
    )


def test_get_var_cornish_fisher_close_to_gaussian_for_normal_data(recorder):
    # Near-normal data has skewness and excess kurtosis near 0, so the two agree.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(2000) * 0.01)

    gaussian = var_model.get_var_gaussian(returns=returns, alpha=0.05)
    cornish_fisher = var_model.get_var_cornish_fisher(returns=returns, alpha=0.05)

    recorder.capture(
        bool(round(abs(gaussian - cornish_fisher), 4) < CLOSE_TO_GAUSSIAN_TOLERANCE)
    )


def test_get_var_cornish_fisher_more_extreme_for_skewed_fat_tailed_data(recorder):
    # Negatively skewed, fat-tailed returns should give a more negative VaR.
    rng = np.random.default_rng(42)
    returns = pd.Series(
        np.concatenate(
            [
                rng.standard_normal(1900) * 0.005,
                -np.abs(rng.standard_normal(100)) * 0.05,
            ]
        )
    )

    gaussian = var_model.get_var_gaussian(returns=returns, alpha=0.05)
    cornish_fisher = var_model.get_var_cornish_fisher(returns=returns, alpha=0.05)

    recorder.capture(bool(cornish_fisher < gaussian))


def test_get_var_cornish_fisher_multiindex(recorder):
    periods = ["2023Q1", "2023Q2"]
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    multi_index = pd.MultiIndex.from_product(
        [periods[:1], dates[:5]], names=["Period", "Date"]
    )
    returns_multi = pd.DataFrame(
        {"AAPL": np.random.default_rng(1).normal(0.001, 0.02, 5)}, index=multi_index
    )

    result = var_model.get_var_cornish_fisher(returns=returns_multi, alpha=0.05)
    recorder.capture(result.shape)


def _three_asset_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAPL": [0.03, 0.02, 0.01, 0.0, 0.006, -0.01, -0.005, 0.008, 0.012, -0.02],
            "MSFT": [
                0.025,
                0.015,
                0.008,
                -0.002,
                0.004,
                -0.008,
                -0.003,
                0.007,
                0.01,
                -0.015,
            ],
            "GOOG": [-0.01, 0.01, -0.02, 0.015, 0.0, 0.02, -0.005, -0.01, 0.005, 0.01],
        }
    )


def test_get_marginal_var(recorder):
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})

    recorder.capture(
        var_model.get_marginal_var(returns=returns, weights=weights, alpha=0.05).round(
            6
        )
    )


def test_get_marginal_var_equal_weights(recorder):
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 1 / 3, "MSFT": 1 / 3, "GOOG": 1 / 3})

    recorder.capture(
        var_model.get_marginal_var(returns=returns, weights=weights, alpha=0.05).round(
            6
        )
    )


def test_get_marginal_var_gaussian(recorder):
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})

    recorder.capture(
        var_model.get_marginal_var(
            returns=returns, weights=weights, alpha=0.05, method="gaussian"
        ).round(6)
    )


def test_get_marginal_var_invalid_distribution():
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})

    try:
        var_model.get_marginal_var(
            returns=returns, weights=weights, alpha=0.05, method="unknown"
        )
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_marginal_var_missing_weight():
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})

    try:
        var_model.get_marginal_var(returns=returns, weights=weights, alpha=0.05)
        raise AssertionError("Expected a ValueError to be raised.")
    except ValueError:
        pass


def test_get_component_var(recorder):
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})

    recorder.capture(
        var_model.get_component_var(returns=returns, weights=weights, alpha=0.05).round(
            6
        )
    )


def test_get_component_var_sums_to_portfolio_var(recorder):
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})

    component_var = var_model.get_component_var(
        returns=returns, weights=weights, alpha=0.05
    )
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    portfolio_var = var_model.get_var_historic(portfolio_returns, alpha=0.05)

    recorder.capture(bool(np.isclose(component_var.sum(), portfolio_var, atol=1e-8)))


def test_get_var_gaussian_uses_sample_standard_deviation():
    returns = pd.Series([0.03, -0.02, 0.01, -0.04, 0.02, 0.005, -0.01])
    expected = returns.mean() + stats.norm.ppf(0.05) * returns.std(ddof=1)

    assert np.isclose(var_model.get_var_gaussian(returns, 0.05), expected)


def test_get_var_studentt_uses_fitted_location():
    rng = np.random.default_rng(7)
    returns = pd.Series(stats.t.rvs(4, size=1500, random_state=rng) * 0.01 + 0.001)
    dof, location, scale = stats.t.fit(returns)

    result = var_model.get_var_studentt(returns, 0.05)

    assert np.isclose(
        float(result[0]), location + scale * stats.t.ppf(0.05, dof), atol=1e-10
    )


def test_get_var_studentt_too_few_observations_is_nan():
    returns = pd.DataFrame({"A": [0.01, np.nan, np.nan], "B": [0.01, -0.02, 0.03]})

    result = var_model.get_var_studentt(returns, 0.05)

    assert np.isnan(result["A"])
    assert np.isfinite(result["B"])


def test_get_var_evt_alpha_outside_fitted_tail_raises():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 500))

    with pytest.raises(ValueError, match="outside the tail"):
        var_model.get_var_evt(returns, alpha=0.10, threshold_percentile=0.95)

    # The boundary itself is inside the fitted tail.
    assert np.isfinite(var_model.get_var_evt(returns, 0.05, 0.95).iloc[0])


def test_get_marginal_var_gaussian_is_exact_euler_allocation():
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.multivariate_normal(
            [0.002, 0.001, -0.0005],
            [[4e-4, 1e-4, 0], [1e-4, 2e-4, 5e-5], [0, 5e-5, 1e-4]],
            size=1000,
        ),
        columns=["A", "B", "C"],
    )
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    marginal = var_model.get_marginal_var(returns, weights, 0.05, "gaussian")
    component = var_model.get_component_var(returns, weights, 0.05, "gaussian")
    portfolio_var = var_model.get_var_gaussian(returns @ weights, 0.05)

    # The components add up to the portfolio VaR, also with non-zero means.
    assert np.isclose(component.sum(), portfolio_var, atol=1e-12)

    # And each marginal VaR is the derivative of the portfolio VaR to the weight.
    step = 1e-6
    for asset in weights.index:
        bumped = weights.copy()
        bumped[asset] += step
        finite_difference = (
            var_model.get_var_gaussian(returns @ bumped, 0.05) - portfolio_var
        ) / step
        assert np.isclose(marginal[asset], finite_difference, atol=1e-5)


def test_get_marginal_var_drops_dates_with_a_missing_return():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, -0.01, 0.02, np.nan],
            "B": [0.02, -0.01, 0.01, -0.03, 0.01, 0.05],
        }
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    with_gap = var_model.get_component_var(returns, weights, 0.05, "gaussian")
    without_row = var_model.get_component_var(
        returns.iloc[:-1], weights, 0.05, "gaussian"
    )

    pd.testing.assert_series_equal(with_gap, without_row)


def test_get_var_evt_needs_ten_exceedances(caplog):
    rng = np.random.default_rng(11)
    returns = pd.Series(stats.t.rvs(4, size=400, random_state=rng) * 0.01)

    # 400 returns hold 20 losses above the 95th percentile: the fit matches scipy's.
    losses = -returns.to_numpy()
    threshold = np.percentile(losses, 95)
    exceedances = losses[losses > threshold] - threshold
    shape, _, scale = stats.genpareto.fit(exceedances, floc=0)
    probability = len(exceedances) / len(losses)
    expected = -(threshold + scale / shape * ((0.05 / probability) ** (-shape) - 1))
    assert var_model.get_var_evt(returns, 0.05).iloc[0] == pytest.approx(expected)

    # 60 returns hold 3, too few for a stable fit: NaN, with one logged warning.
    with caplog.at_level("WARNING"):
        assert np.isnan(var_model.get_var_evt(returns.iloc[:60], 0.05).iloc[0])
    assert sum("Extreme Value Theory" in m for m in caplog.messages) == 1


def test_get_var_evt_within_period_warns_once(caplog):
    rng = np.random.default_rng(12)
    index = pd.period_range("2020-01-01", periods=730, freq="D")
    returns = pd.DataFrame(
        {"A": stats.t.rvs(4, size=730, random_state=rng) * 0.01}, index=index
    )
    within = returns.set_axis(pd.MultiIndex.from_arrays([index.asfreq("Q"), index]))

    with caplog.at_level("WARNING"):
        result = var_model.get_var_evt(within, 0.05)

    # A quarter of daily returns has about 4 losses above the 95th percentile.
    assert result["A"].isna().all()
    assert sum("Extreme Value Theory" in m for m in caplog.messages) == 1


def test_get_liquidity_cost_constant_spread_is_half_the_spread():
    # With a constant spread its standard deviation is 0, so either scaling gives S / 2.
    spreads = pd.Series(0.004, index=pd.period_range("2024-01", periods=12, freq="M"))

    assert var_model.get_liquidity_cost(spreads, 0.05) == pytest.approx(0.002)
    assert var_model.get_liquidity_cost(spreads, 0.05, 2.33) == pytest.approx(0.002)


def test_get_liquidity_cost_matches_the_formula():
    spreads = pd.Series(np.random.default_rng(3).gamma(2, 0.002, 60))

    assert var_model.get_liquidity_cost(spreads, 0.05) == pytest.approx(
        0.5 * np.percentile(spreads, 95)
    )
    assert var_model.get_liquidity_cost(spreads, 0.01, 1.5) == pytest.approx(
        0.5 * (spreads.mean() + 1.5 * spreads.std(ddof=1))
    )
    assert np.isnan(var_model.get_liquidity_cost(spreads.iloc[:1], 0.05))


def test_get_liquidity_adjusted_value_at_risk_per_asset():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.normal(0, 0.02, (500, 2)), columns=["A", "B"])
    spreads = pd.DataFrame({"A": rng.gamma(2, 0.001, 60), "B": rng.gamma(2, 0.01, 60)})
    value_at_risk = var_model.get_var_historic(returns, 0.05)

    result = var_model.get_liquidity_adjusted_value_at_risk(
        value_at_risk, spreads, 0.05
    )

    for column in ["A", "B"]:
        assert result[column] == pytest.approx(
            value_at_risk[column] - 0.5 * np.percentile(spreads[column], 95)
        )
    # The less liquid asset carries the larger adjustment.
    assert (value_at_risk - result)["B"] > (value_at_risk - result)["A"]


# Forty returns of a Student-t(4) draw; the kernel reference is the quantile of scipy's
# gaussian_kde(bw_method="silverman") on them.
KERNEL_REFERENCE_RETURNS = pd.Series(
    [0.0004, -0.0137, 0.0009, -0.0064, 0.0025, 0.0177, -0.0455, -0.0052, 0.0026, -0.0068,
     -0.0241, -0.007, 0.0001, -0.001, -0.0152, -0.0147, -0.0041, -0.0149, -0.0015, 0.0059,
     0.0036, -0.0195, 0.0084, -0.0079, -0.0059, 0.0107, -0.0016, -0.0031, -0.0039, -0.0035,
     0.0002, 0.0143, -0.0145, -0.0299, 0.0015, -0.0072, -0.0021, 0.0113, 0.0011, -0.0077]
)  # fmt: skip


def test_get_var_kernel_reference_value():
    assert np.isclose(
        var_model.get_var_kernel(KERNEL_REFERENCE_RETURNS, 0.05),
        -0.028146773187185482,
        atol=1e-12,
    )


def test_get_var_kernel_frame(recorder):
    frame = pd.DataFrame(
        {"A": KERNEL_REFERENCE_RETURNS, "B": KERNEL_REFERENCE_RETURNS * 2}
    )
    result = var_model.get_var_kernel(frame, 0.05)
    # The bandwidth scales with the returns, so the quantile does too.
    assert np.isclose(result["B"], 2 * result["A"])
    recorder.capture(result.round(10))
    # A sample without spread is its own quantile.
    assert var_model.get_var_kernel(pd.Series([0.01, 0.01, 0.01]), 0.05) == 0.01


def _garch_returns(n=800, seed=2):
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    variance = 1e-4
    for step in range(n):
        shock = rng.standard_t(6) * np.sqrt(4 / 6)
        returns[step] = 0.0003 + np.sqrt(variance) * shock
        variance = 2e-6 + 0.08 * (returns[step] - 0.0003) ** 2 + 0.9 * variance
    return pd.Series(returns, index=pd.bdate_range("2020-01-01", periods=n))


@pytest.mark.parametrize("distribution", ["normal", "t", "skewt", "ged"])
def test_get_var_garch_matches_the_arch_package(distribution):
    arch = pytest.importorskip("arch")
    from financetoolkit.risk.volatility import garch_model  # noqa: PLC0415

    returns = _garch_returns()
    weights = garch_model.get_garch_family_weights(
        returns.to_numpy(), "garch", distribution, "constant"
    )

    # arch's order is Mu, Omega, Alpha, Beta and the distribution, on returns in percent.
    model = arch.arch_model(
        returns.to_numpy() * 100, mean="Constant", vol="GARCH", dist=distribution
    )
    fixed = model.fix(
        np.r_[weights[3] * 100, weights[0] * 1e4, weights[1:3], weights[4:]]
    )
    forecast = fixed.forecast(horizon=1, reindex=False)
    quantile = np.asarray(
        model.distribution.ppf(0.05, list(weights[4:]) or None)
    ).item()
    expected = (
        forecast.mean.iloc[-1, 0] + np.sqrt(forecast.variance.iloc[-1, 0]) * quantile
    ) / 100

    assert var_model.get_var_garch(
        returns, 0.05, distribution=distribution
    ) == pytest.approx(expected, rel=1e-6)


def test_get_cvar_garch_is_the_mean_below_the_garch_var():
    from financetoolkit.risk.value_at_risk import cvar_model  # noqa: PLC0415
    from financetoolkit.risk.volatility import garch_model  # noqa: PLC0415

    returns = _garch_returns()
    forecast = garch_model.get_garch_family_rolling_forecast(
        returns, None, 21, "garch", "t", "constant"
    ).iloc[-1]

    # The mean of draws from the next period's distribution below their 5% quantile.
    draws = forecast["Mean"] + forecast[
        "Volatility"
    ] * garch_model.get_standardized_draws(
        "t", (forecast["Shape"],), 2_000_000, np.random.default_rng(0)
    )
    var = var_model.get_var_garch(returns, 0.05, distribution="t")
    cvar = cvar_model.get_cvar_garch(returns, 0.05, distribution="t")

    assert np.mean(draws < var) == pytest.approx(0.05, abs=1e-3)
    assert cvar == pytest.approx(draws[draws < var].mean(), rel=5e-3)


def test_get_var_garch_rolling_path_and_frames(recorder):
    returns = _garch_returns(n=400)
    frame = pd.DataFrame({"A": returns, "B": returns.iloc[::-1].to_numpy()})
    frame.iloc[:150, 1] = np.nan

    path = var_model.get_var_garch(frame, 0.05, window_size=200, refit_every=50)

    # Each column is rolled over its own returns: B starts 150 returns later.
    assert path["A"].first_valid_index() == frame.index[199]
    assert path["B"].first_valid_index() == frame.index[349]
    # Four decimals: the fit at the 300th return of A peaks on the Alpha = 0 bound, where Omega and Beta trade
    # off along a flat ridge, so its VaR moves by about 2e-5 under 1e-12 relative noise in the returns.
    recorder.capture(path.dropna(how="all").iloc[::25].round(4))

    with pytest.raises(ValueError, match="at least 100"):
        var_model.get_var_garch(returns, 0.05, window_size=50)
    # Too few returns for a single fit give NaN rather than a fit on a handful of returns.
    assert np.isnan(var_model.get_var_garch(returns.iloc[:50], 0.05))
