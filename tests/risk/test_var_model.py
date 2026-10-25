"""Value at Risk Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.risk import var_model

# pylint: disable=missing-function-docstring

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


def test_get_rolling_var_historic(recorder):
    returns = pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15, -0.05, 0.08, 0.12])
    recorder.capture(
        var_model.get_rolling_var_historic(
            returns=returns, alpha=0.05, window_size=3
        ).round(4)
    )


def test_get_rolling_var_historic_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.05],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.08, -0.03],
        }
    )
    recorder.capture(
        var_model.get_rolling_var_historic(
            returns=returns_df, alpha=0.05, window_size=3
        ).round(4)
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
            returns=returns, weights=weights, alpha=0.05, distribution="gaussian"
        ).round(6)
    )


def test_get_marginal_var_invalid_distribution():
    returns = _three_asset_returns()
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})

    try:
        var_model.get_marginal_var(
            returns=returns, weights=weights, alpha=0.05, distribution="unknown"
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
