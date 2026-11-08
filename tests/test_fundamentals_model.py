"""Fundamentals Model Tests"""

# ruff: noqa

import pytest

from financetoolkit import fundamentals_model


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


def test_a_period_ending_after_the_end_date_leaves_the_statistics_too(monkeypatch):
    import pandas as pd

    from financetoolkit import fmp_model

    def fake_statement(ticker, **kwargs):
        return pd.DataFrame(
            {
                "2021": ["2021-12-31", "USD", 1.0],
                "2022": ["2022-12-31", "USD", 2.0],
                "2023": ["2023-12-31", "USD", 3.0],
            },
            index=["periodEndDate", "reportedCurrency", "totalAssets"],
        )

    monkeypatch.setattr(fmp_model, "get_financial_statement", fake_statement)

    def normalization(name):
        return pd.read_csv(
            f"financetoolkit/normalization/{name}.csv", index_col=0
        ).iloc[:, 0]

    total, statistics, *_ = fundamentals_model.collect_financial_statements(
        tickers=["AAA", "BBB"],
        statement="balance",
        api_key="KEY",
        start_date="2021-01-01",
        end_date="2023-06-30",
        fmp_statement_format=normalization("balance"),
        fmp_statistics_format=normalization("statistics"),
        sleep_timer=False,
        enforce_source="FinancialModelingPrep",
    )

    # Fiscal 2023 ends on 31 December, after the end date, so neither holds it.
    assert [str(period) for period in total.columns] == ["2021", "2022"]
    assert [str(period) for period in statistics.columns] == ["2021", "2022"]
