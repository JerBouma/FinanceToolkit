"""Value at Risk Model Tests"""

import pandas as pd

from financetoolkit.risk import var_model

# pylint: disable=missing-function-docstring


def test_get_var_historic(recorder):
    recorder.capture(
        var_model.get_var_historic(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_var_gaussian(recorder):
    recorder.capture(
        round(
            var_model.get_var_gaussian(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
                alpha=0.05,
                cornish_fisher=True,
            ),
            4,
        )
    )
    recorder.capture(
        round(
            var_model.get_var_gaussian(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]),
                alpha=0.03,
                cornish_fisher=False,
            ),
            4,
        )
    )


def test_get_var_studentt(recorder):
    recorder.capture_list(
        var_model.get_var_studentt(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.03
        ).round(4)
    )
