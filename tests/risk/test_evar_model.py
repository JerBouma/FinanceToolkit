"""Entropic Value at Risk Model Tests"""

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk import evar_model

# pylint: disable=missing-function-docstring


def test_get_evar_gaussian(recorder):
    recorder.capture(
        evar_model.get_evar_gaussian(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_evar_gaussian_dataframe(recorder):
    returns_df = pd.DataFrame(
        {"AAPL": [0.3, 0.2, 0.1, 0, 0.06], "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04]}
    )
    recorder.capture(evar_model.get_evar_gaussian(returns=returns_df, alpha=0.05))


def test_get_evar_gaussian_different_alphas(recorder):
    returns = pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.05])

    # Test with different alpha levels
    for alpha in [0.01, 0.05, 0.1]:
        result = evar_model.get_evar_gaussian(returns=returns, alpha=alpha)
        recorder.capture(round(result, 4))


def test_get_evar_gaussian_negative_returns(recorder):
    returns = pd.Series([-0.1, -0.05, -0.02, 0.01, 0.03])
    recorder.capture(evar_model.get_evar_gaussian(returns=returns, alpha=0.05))


def test_get_evar_gaussian_multiindex(recorder):
    # Test with MultiIndex DataFrame (period-based)
    periods = ["2023Q1", "2023Q2"]
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    multi_index = pd.MultiIndex.from_product(
        [periods[:1], dates[:5]], names=["Period", "Date"]
    )
    returns_multi = pd.DataFrame(
        {"AAPL": np.random.normal(0.001, 0.02, 5)}, index=multi_index
    )

    result = evar_model.get_evar_gaussian(returns=returns_multi, alpha=0.05)
    recorder.capture(result.shape)


def test_get_evar_gaussian_edge_cases(recorder):
    # Test with very small dataset
    small_returns = pd.Series([0.01, 0.02])
    recorder.capture(evar_model.get_evar_gaussian(returns=small_returns, alpha=0.05))

    # Test with constant returns
    constant_returns = pd.Series([0.01] * 10)
    recorder.capture(evar_model.get_evar_gaussian(returns=constant_returns, alpha=0.05))


def test_get_evar_gaussian_uses_sample_standard_deviation():
    returns = pd.Series([0.03, -0.02, 0.01, -0.04, 0.02, 0.005, -0.01])
    expected = returns.mean() - returns.std(ddof=1) * np.sqrt(-2 * np.log(0.05))

    assert np.isclose(evar_model.get_evar_gaussian(returns, 0.05), expected)
    # The EVaR bounds the gaussian VaR at the same alpha.
    assert expected <= returns.mean() + stats.norm.ppf(0.05) * returns.std(ddof=1)
