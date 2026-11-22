"""Entropic Value at Risk Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.risk.value_at_risk import evar_model


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


# Forty returns of a Student-t(4) draw, with the reference values of skfolio (evar,
# entropic_risk_measure) and riskfolio-lib (EVaR_Hist, Entropic_RM, EVRG), signs flipped
# to the Toolkit's return convention.
REFERENCE_RETURNS = pd.Series(
    [0.0004, -0.0137, 0.0009, -0.0064, 0.0025, 0.0177, -0.0455, -0.0052, 0.0026, -0.0068,
     -0.0241, -0.007, 0.0001, -0.001, -0.0152, -0.0147, -0.0041, -0.0149, -0.0015, 0.0059,
     0.0036, -0.0195, 0.0084, -0.0079, -0.0059, 0.0107, -0.0016, -0.0031, -0.0039, -0.0035,
     0.0002, 0.0143, -0.0145, -0.0299, 0.0015, -0.0072, -0.0021, 0.0113, 0.0011, -0.0077]
)  # fmt: skip


def test_get_evar_historic_reference_values():
    assert np.isclose(
        evar_model.get_evar_historic(REFERENCE_RETURNS, 0.05),
        -0.04227204001870698,
        atol=1e-12,
    )
    assert np.isclose(
        evar_model.get_evar_historic(REFERENCE_RETURNS, 0.2),
        -0.031652995960299385,
        atol=1e-12,
    )


def test_get_evar_historic_bounds():
    # Between the historical CVaR and the worst return, and equal to the worst return when it
    # covers the tail on its own (a single loss is more than 5% of 3 returns).
    returns = REFERENCE_RETURNS
    evar = evar_model.get_evar_historic(returns, 0.05)
    tail = returns[returns <= returns.quantile(0.05)]
    assert returns.min() <= evar <= tail.mean()
    assert evar_model.get_evar_historic(pd.Series([0.01, -0.02, 0.03]), 0.05) == -0.02


def test_get_evar_historic_frame_and_rolling(recorder):
    frame = pd.DataFrame(
        {"A": REFERENCE_RETURNS, "B": REFERENCE_RETURNS[::-1].to_numpy() * 2}
    )
    recorder.capture(evar_model.get_evar_historic(frame, 0.05).round(10))
    recorder.capture(
        evar_model.get_evar_historic(frame, 0.05, window_size=20).round(10)
    )


def test_get_entropic_risk_measure_reference_values():
    assert np.isclose(
        evar_model.get_entropic_risk_measure(REFERENCE_RETURNS, 0.02, 0.05),
        -0.06892219395430653,
        atol=1e-12,
    )
    # Never less severe than the EVaR, which minimizes it over the risk aversion.
    for risk_aversion in [0.001, 0.01, 0.1, 1.0]:
        assert evar_model.get_entropic_risk_measure(
            REFERENCE_RETURNS, risk_aversion, 0.05
        ) <= evar_model.get_evar_historic(REFERENCE_RETURNS, 0.05)


def test_get_entropic_risk_measure_validation():
    with pytest.raises(ValueError):
        evar_model.get_entropic_risk_measure(REFERENCE_RETURNS, 0, 0.05)
    with pytest.raises(ValueError):
        evar_model.get_evar_historic(REFERENCE_RETURNS, 1.5)
