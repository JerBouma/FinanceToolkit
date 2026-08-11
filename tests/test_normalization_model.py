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


def _partial_statement(statement_format, reported: int = 2):
    """A statement holding only the first `reported` line items of the format."""
    index = pd.MultiIndex.from_tuples(
        [("AAPL", name) for name in statement_format.index[:reported]]
    )

    return pd.DataFrame(
        [[100.0, 110.0], [40.0, 45.0]][:reported],
        index=index,
        columns=["2023", "2024"],
    )


def test_adjust_financial_statements_fills_missing_items_with_nan():
    """An item the provider never reported must not read as a genuine zero.

    Filling with zero makes an unreported line item indistinguishable from one
    reported as actually zero, and that fabricated zero then flows into any ratio
    using it as a term or a denominator.
    """
    statement_format = normalization_model.read_normalization_file("balance")
    financial_statements = _partial_statement(statement_format)

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=statement_format,
        adjust_financial_statements=True,
    )

    unreported = result.loc[("AAPL", statement_format.iloc[2])]

    assert unreported.isna().all()
    assert not (unreported == 0).any()


def test_adjust_financial_statements_still_aligns_every_row():
    """Filling with NaN must not cost the row alignment the option exists for."""
    statement_format = normalization_model.read_normalization_file("balance")
    financial_statements = _partial_statement(statement_format)

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=statement_format,
        adjust_financial_statements=True,
    )

    assert len(result.loc["AAPL"]) == len(statement_format)


def test_adjust_financial_statements_leaves_reported_values_untouched():
    statement_format = normalization_model.read_normalization_file("balance")
    financial_statements = _partial_statement(statement_format)

    result = normalization_model.convert_financial_statements(
        financial_statements=financial_statements,
        statement_format=statement_format,
        adjust_financial_statements=True,
    )

    reported = result.loc[("AAPL", statement_format.iloc[0])]

    assert list(reported) == [100.0, 110.0]
