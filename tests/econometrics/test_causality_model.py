"""Granger Causality Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.econometrics import causality_model

# pylint: disable=missing-function-docstring


def test_get_granger_causality_causal(recorder):
    rng = np.random.default_rng(3)
    n = 500
    b = rng.standard_normal(n)
    a = np.zeros(n)
    for i in range(2, n):
        a[i] = 0.6 * a[i - 1] + 0.5 * b[i - 1] + rng.standard_normal() * 0.3
    recorder.capture(
        causality_model.get_granger_causality(
            pd.Series(a), pd.Series(b), max_lag=2
        ).round(4)
    )


def test_get_granger_causality_unrelated(recorder):
    rng = np.random.default_rng(3)
    series_a = pd.Series(rng.standard_normal(500))
    series_b = pd.Series(rng.standard_normal(500))
    recorder.capture(
        causality_model.get_granger_causality(series_a, series_b, max_lag=3).round(4)
    )


def test_get_granger_causality_too_few_observations(recorder):
    series_a = pd.Series([1.0, 2.0, 3.0])
    series_b = pd.Series([1.0, 2.0, 3.0])
    recorder.capture(
        causality_model.get_granger_causality(series_a, series_b, max_lag=5)
    )
