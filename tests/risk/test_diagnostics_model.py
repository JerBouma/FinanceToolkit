"""Risk Diagnostics Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.risk import diagnostics_model

# pylint: disable=missing-function-docstring


def test_get_arch_lm_test(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(200) * 0.01)
    recorder.capture(diagnostics_model.get_arch_lm_test(returns).round(4))


def test_get_arch_lm_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_arch_lm_test(returns_df).round(4))


def test_get_arch_lm_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_arch_lm_test(returns, lags=5))


def test_get_jarque_bera_test(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(200) * 0.01)
    recorder.capture(diagnostics_model.get_jarque_bera_test(returns).round(4))


def test_get_jarque_bera_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_exponential(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_jarque_bera_test(returns_df).round(4))
