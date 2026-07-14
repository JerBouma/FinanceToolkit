"""Conditional Value at Risk Model Tests"""

import pandas as pd

from financetoolkit.risk import cvar_model

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
