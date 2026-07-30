"""Cointegration Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.timeseries import cointegration_model

# pylint: disable=missing-function-docstring


def test_get_engle_granger_cointegration_independent(recorder):
    rng = np.random.default_rng(2)
    series_a = pd.Series(np.cumsum(rng.standard_normal(500)))
    series_b = pd.Series(np.cumsum(rng.standard_normal(500)))
    recorder.capture(
        cointegration_model.get_engle_granger_cointegration(series_a, series_b).round(4)
    )


def test_get_engle_granger_cointegration_cointegrated(recorder):
    rng = np.random.default_rng(2)
    common_trend = np.cumsum(rng.standard_normal(500))
    series_a = pd.Series(common_trend + rng.standard_normal(500) * 0.1)
    series_b = pd.Series(common_trend * 2 + 5 + rng.standard_normal(500) * 0.1)
    recorder.capture(
        cointegration_model.get_engle_granger_cointegration(series_a, series_b).round(4)
    )
