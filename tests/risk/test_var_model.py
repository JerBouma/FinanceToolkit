"""Value at Risk Model Tests"""

import numpy as np
import pandas as pd

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
            var_model.get_var_gaussian(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
                alpha=0.05,
                cornish_fisher=True,
            ),
            4,
        )
    )
    recorder.capture(
        round(
            var_model.get_var_gaussian(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
                alpha=0.03,
                cornish_fisher=False,
            ),
            4,
        )
    )


def test_get_var_gaussian_dataframe(recorder):
    returns_df = pd.DataFrame(
        {"AAPL": [0.3, 0.2, 0.1, 0, 0.06], "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04]}
    )

    result_cf = var_model.get_var_gaussian(
        returns=returns_df, alpha=0.05, cornish_fisher=True
    )
    recorder.capture(result_cf.round(4))

    result_normal = var_model.get_var_gaussian(
        returns=returns_df, alpha=0.05, cornish_fisher=False
    )
    recorder.capture(result_normal.round(4))


def test_get_var_gaussian_negative_returns(recorder):
    returns = pd.Series([-0.1, -0.05, -0.02, 0.01, 0.03, -0.08])

    recorder.capture(
        round(
            var_model.get_var_gaussian(
                returns=returns, alpha=0.05, cornish_fisher=True
            ),
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
            var_model.get_var_gaussian(
                returns=constant_returns, alpha=0.05, cornish_fisher=False
            ),
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
