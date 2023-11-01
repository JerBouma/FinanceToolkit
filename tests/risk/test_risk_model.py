"""Performance Model Tests"""
import pandas as pd

from financetoolkit.risk import risk_model

# pylint: disable=missing-function-docstring


def test_get_var_historic(recorder):
    recorder.capture(
        risk_model.get_var_historic(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_var_gaussian(recorder):
    recorder.capture(
        risk_model.get_var_gaussian(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05, cornish_fisher=True
        )
    )
    recorder.capture(
        risk_model.get_var_gaussian(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            alpha=0.03,
            cornish_fisher=False,
        )
    )


def test_get_var_studentt(recorder):
    recorder.capture_list(
        risk_model.get_var_studentt(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.03
        )
    )


def test_get_cvar_historic(recorder):
    recorder.capture(
        risk_model.get_cvar_historic(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_gaussian(recorder):
    recorder.capture(
        risk_model.get_cvar_gaussian(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_studentt(recorder):
    recorder.capture(
        risk_model.get_cvar_studentt(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_laplace(recorder):
    recorder.capture(
        risk_model.get_cvar_laplace(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_logistic(recorder):
    recorder.capture(
        risk_model.get_cvar_logistic(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_evar_gaussian(recorder):
    recorder.capture(
        risk_model.get_evar_gaussian(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_max_drawdown(recorder):
    recorder.capture(
        risk_model.get_max_drawdown(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
        )
    )


def test_get_ui(recorder):
    recorder.capture(
        risk_model.get_ui(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
            rolling=2,
        )
    )


def test_get_skewness(recorder):
    recorder.capture(
        risk_model.get_skewness(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
        )
    )


def test_get_kurtosis(recorder):
    recorder.capture(
        risk_model.get_kurtosis(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), fisher=True
        )
    )

    recorder.capture(
        risk_model.get_kurtosis(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), fisher=False
        )
    )
