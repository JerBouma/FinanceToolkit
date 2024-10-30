"""Entropic Value at Risk Model Tests"""

import pandas as pd

from financetoolkit.risk import evar_model

# pylint: disable=missing-function-docstring


def test_get_evar_gaussian(recorder):
    recorder.capture(
        evar_model.get_evar_gaussian(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )
