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


def test_get_var_historic_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.05],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.08, -0.03],
        }
    )
    recorder.capture(var_model.get_var_historic(returns=returns_df, alpha=0.05))


def test_get_var_historic_different_alphas(recorder):
    returns = pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15, -0.05, 0.08, 0.12])

    for alpha in [0.01, 0.05, 0.1, 0.2]:
        result = var_model.get_var_historic(returns=returns, alpha=alpha)
        recorder.capture(round(result, 4))


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


def test_get_var_gaussian_dataframe(recorder):
    returns_df = pd.DataFrame(
        {"AAPL": [0.3, 0.2, 0.1, 0, 0.06], "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04]}
    )

    result_cf = var_model.get_var_gaussian(
        returns=returns_df, alpha=0.05, cornish_fisher=True
    )
    recorder.capture(result_cf.round(4))

    result_normal = var_model.get_var_gaussian(
        returns=returns_df, alpha=0.05, cornish_fisher=False
    )
    recorder.capture(result_normal.round(4))


def test_get_var_gaussian_negative_returns(recorder):
    returns = pd.Series([-0.1, -0.05, -0.02, 0.01, 0.03, -0.08])

    recorder.capture(
        round(
            var_model.get_var_gaussian(
                returns=returns, alpha=0.05, cornish_fisher=True
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


def test_get_var_studentt_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.05],
        }
    )

    result = var_model.get_var_studentt(returns=returns_df, alpha=0.05)
    recorder.capture(result.round(4))


def test_var_edge_cases(recorder):
    # Test with very small dataset
    small_returns = pd.Series([0.01, 0.02])

    recorder.capture(var_model.get_var_historic(returns=small_returns, alpha=0.05))

    # Test with constant returns
    constant_returns = pd.Series([0.01] * 10)

    recorder.capture(
        round(
            var_model.get_var_gaussian(
                returns=constant_returns, alpha=0.05, cornish_fisher=False
            ),
            4,
        )
    )
