"""Fundamentals Model Tests"""

# ruff: noqa

import pytest

from financetoolkit import fundamentals_model

# pylint: disable=missing-function-docstring


def test_collect_financial_statements_invalid_ticker_type():
    with pytest.raises(ValueError, match="Type for the tickers"):
        fundamentals_model.collect_financial_statements(
            tickers=123, statement="income", api_key="test_key"  # Invalid type
        )


def test_collect_financial_statements_missing_api_key():
    with pytest.raises(ValueError, match="Please enter an API key"):
        fundamentals_model.collect_financial_statements(
            tickers="AAPL",
            statement="income",
            api_key="",
            enforce_source="FinancialModelingPrep",
        )
