# ruff: noqa

"""Normalization Model Tests"""


import pandas as pd

from financetoolkit import normalization_model

# pylint: disable=missing-function-docstring


def test_convert_financial_statements_empty_format(recorder):
    financial_statements = pd.DataFrame(
        {("AAPL", "2023"): [1000, 2000, 3000], ("AAPL", "2022"): [900, 1800, 2700]}
    )
    financial_statements.index = ["Revenue", "Cost of Revenue", "Gross Profit"]

    empty_format = pd.DataFrame()

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=empty_format,
        reverse_dates=False,
    )

    recorder.capture(result.shape)
