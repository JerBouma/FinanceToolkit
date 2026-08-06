"""Conditional Value at Risk Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.risk import cvar_model, var_model

# pylint: disable=missing-function-docstring


def test_get_cvar_historic(recorder):
    recorder.capture(
        cvar_model.get_cvar_historic(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_gaussian(recorder):
    recorder.capture(
        round(
            cvar_model.get_cvar_gaussian(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
            ),
            4,
        )
    )


def test_get_cvar_studentt(recorder):
    recorder.capture(
        cvar_model.get_cvar_studentt(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_laplace(recorder):
    recorder.capture(
        cvar_model.get_cvar_laplace(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_rolling_cvar_historic(recorder):
    returns = pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15, -0.05, 0.08, 0.12])
    recorder.capture(
        cvar_model.get_rolling_cvar_historic(
            returns=returns, alpha=0.05, window_size=3
        ).round(4)
    )


def test_get_rolling_cvar_historic_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.05],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.08, -0.03],
        }
    )
    recorder.capture(
        cvar_model.get_rolling_cvar_historic(
            returns=returns_df, alpha=0.05, window_size=3
        ).round(4)
    )


def test_get_cvar_logistic(recorder):
    recorder.capture(
        round(
            cvar_model.get_cvar_logistic(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
            ),
            4,
        )
    )


def test_get_cvar_cornish_fisher(recorder):
    recorder.capture(
        round(
            cvar_model.get_cvar_cornish_fisher(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
            ),
            4,
        )
    )


def test_get_cvar_cornish_fisher_dataframe(recorder):
    returns_df = pd.DataFrame(
        {"AAPL": [0.3, 0.2, 0.1, 0, 0.06], "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04]}
    )
    recorder.capture(
        cvar_model.get_cvar_cornish_fisher(returns=returns_df, alpha=0.05).round(4)
    )


def test_get_cvar_cornish_fisher_more_extreme_than_var(recorder):
    # CVaR should always be at least as extreme (more negative) than VaR at the same alpha.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(2000) * 0.01)

    var_cf = var_model.get_var_cornish_fisher(returns=returns, alpha=0.05)
    cvar_cf = cvar_model.get_cvar_cornish_fisher(returns=returns, alpha=0.05)

    recorder.capture(bool(cvar_cf <= var_cf))


def test_get_cvar_cornish_fisher_more_extreme_for_skewed_fat_tailed_data(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(
        np.concatenate(
            [
                rng.standard_normal(1900) * 0.005,
                -np.abs(rng.standard_normal(100)) * 0.05,
            ]
        )
    )

    gaussian = cvar_model.get_cvar_gaussian(returns=returns, alpha=0.05)
    cornish_fisher = cvar_model.get_cvar_cornish_fisher(returns=returns, alpha=0.05)

    recorder.capture(bool(cornish_fisher < gaussian))


def test_get_cvar_cornish_fisher_multiindex(recorder):
    periods = ["2023Q1", "2023Q2"]
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    multi_index = pd.MultiIndex.from_product(
        [periods[:1], dates[:5]], names=["Period", "Date"]
    )
    returns_multi = pd.DataFrame(
        {"AAPL": np.random.default_rng(1).normal(0.001, 0.02, 5)}, index=multi_index
    )

    result = cvar_model.get_cvar_cornish_fisher(returns=returns_multi, alpha=0.05)
    recorder.capture(result.shape)


def test_get_cvar_evt(recorder):
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.15, 0.01, -0.02, 0.04, -0.2, 0.02, -0.01, -0.18, 0.03]
    )
    recorder.capture(
        round(
            cvar_model.get_cvar_evt(
                returns=returns, alpha=0.05, threshold_percentile=0.7
            ),
            4,
        )
    )


def test_get_cvar_evt_dataframe(recorder):
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
        cvar_model.get_cvar_evt(
            returns=returns_df, alpha=0.05, threshold_percentile=0.7
        ).round(4)
    )


def test_get_cvar_evt_at_least_as_extreme_as_var_evt(recorder):
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.standard_t(3, 2000) * 0.01)

    var_evt = var_model.get_var_evt(
        returns=returns, alpha=0.05, threshold_percentile=0.9
    )
    cvar_evt = cvar_model.get_cvar_evt(
        returns=returns, alpha=0.05, threshold_percentile=0.9
    )

    recorder.capture(bool((cvar_evt <= var_evt).all()))


def test_get_cvar_evt_insufficient_exceedances(recorder):
    # Too few exceedances to fit a GPD, so this returns NaN rather than raising.
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
    recorder.capture(
        cvar_model.get_cvar_evt(returns=returns, alpha=0.05, threshold_percentile=0.95)
    )
