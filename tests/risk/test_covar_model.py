"""CoVaR Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.risk import covar_model

# pylint: disable=missing-function-docstring

QUANTILE_REGRESSION_TOLERANCE = 0.2
CRASH_PROBABILITY = 0.05
TAIL_LINKED_SLOPE_MINIMUM = 0.2


def test_quantile_regression_matches_ols_at_median(recorder):
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.standard_normal(n)
    y = 2 + 3 * x + rng.standard_normal(n)

    intercept, slope = covar_model._quantile_regression(y, x, 0.5)  # noqa: SLF001
    assert abs(intercept - 2) < QUANTILE_REGRESSION_TOLERANCE
    assert abs(slope - 3) < QUANTILE_REGRESSION_TOLERANCE
    recorder.capture((round(intercept, 4), round(slope, 4)))


def test_get_covar_tail_linked(recorder):
    rng = np.random.default_rng(6)
    n = 2000

    crash_factor = rng.standard_normal(n)
    crash_factor[rng.random(n) < CRASH_PROBABILITY] -= 5
    conditioning = pd.Series(crash_factor * 0.01 + rng.standard_normal(n) * 0.01)
    returns = pd.Series(crash_factor * 0.008 + rng.standard_normal(n) * 0.015)

    result = covar_model.get_covar(returns, conditioning, alpha=0.05)
    unconditional_var = np.percentile(returns, 5)

    # CoVaR conditional on distress should be materially worse than the plain
    # (unconditional) VaR when there is genuine tail linkage.
    assert result["CoVaR"] < unconditional_var
    assert result["Quantile Regression Slope"] > TAIL_LINKED_SLOPE_MINIMUM
    recorder.capture(result.round(4))


def test_get_covar_independent(recorder):
    rng = np.random.default_rng(6)
    n = 2000
    returns = pd.Series(rng.standard_normal(n) * 0.02)
    conditioning = pd.Series(rng.standard_normal(n) * 0.02)
    result = covar_model.get_covar(returns, conditioning, alpha=0.05)
    recorder.capture(result.round(4))


def test_get_covar_too_few_observations(recorder):
    returns = pd.Series([0.01, -0.02, 0.03])
    conditioning = pd.Series([0.02, -0.01, 0.02])
    recorder.capture(covar_model.get_covar(returns, conditioning, alpha=0.05))


def test_get_covar_invalid_type():
    try:
        covar_model.get_covar(1, pd.Series([1.0, 2.0]), alpha=0.05)  # type: ignore
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass
