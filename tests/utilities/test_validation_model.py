"""Validation Model Tests"""

# ruff: noqa: PLR2004

import pytest

from financetoolkit.utilities import validation_model


def validate(**overrides):
    """Call the validator with sensible defaults, overriding what a test cares about."""
    parameters = {
        "tickers": ["AAPL", "MSFT"],
        "start_date": None,
        "end_date": None,
        "risk_free_rate": "10y",
        "enforce_source": None,
        "api_key": "",
        "intraday_period": None,
        "benchmark_ticker": "SPY",
    }
    parameters.update(overrides)

    return validation_model.validate_toolkit_parameters(**parameters)


def test_tickers_are_uppercased_and_string_accepted():
    assert validate(tickers="aapl") == ["AAPL"]
    assert validate(tickers=["aapl", "msft"]) == ["AAPL", "MSFT"]


def test_portfolio_entry_keeps_its_casing():
    assert validate(tickers=["aapl", "Portfolio"]) == ["AAPL", "Portfolio"]


def test_duplicate_tickers_are_removed_in_order():
    assert validate(tickers=["MSFT", "aapl", "AAPL", "MSFT"]) == ["MSFT", "AAPL"]


def test_benchmark_ticker_is_removed_from_tickers():
    assert validate(tickers=["AAPL", "SPY"], benchmark_ticker="SPY") == ["AAPL"]


def test_no_tickers_raises():
    with pytest.raises(ValueError, match="ticker or a list"):
        validate(tickers=None)


def test_invalid_tickers_type_raises():
    with pytest.raises(TypeError, match="string or a list"):
        validate(tickers=123)


def test_invalid_start_date_raises():
    with pytest.raises(ValueError, match="valid start date"):
        validate(start_date="01-01-2020")


def test_invalid_end_date_raises():
    with pytest.raises(ValueError, match="valid end date"):
        validate(end_date="2020/01/01")


def test_start_after_end_raises():
    with pytest.raises(ValueError, match="before the end date"):
        validate(start_date="2022-01-01", end_date="2020-01-01")


def test_invalid_risk_free_rate_raises():
    with pytest.raises(ValueError, match="valid risk free rate"):
        validate(risk_free_rate="7y")


def test_invalid_enforce_source_raises():
    with pytest.raises(ValueError, match="enforced source"):
        validate(enforce_source="Bloomberg")


def test_enforced_fmp_without_api_key_raises():
    with pytest.raises(ValueError, match="API key from FinancialModelingPrep"):
        validate(enforce_source="FinancialModelingPrep", api_key="")


def test_enforced_fmp_with_api_key_passes():
    assert validate(enforce_source="FinancialModelingPrep", api_key="KEY") == [
        "AAPL",
        "MSFT",
    ]


def test_invalid_intraday_period_raises():
    with pytest.raises(ValueError, match="valid intraday period"):
        validate(intraday_period="2min")
