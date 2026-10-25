"""Tests of the indicator references an econometrics call can analyze (the series parameter)."""

import asyncio

import numpy as np
import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller, series_model
from financetoolkit.mcp_server.provider_model import ToolkitProvider

# pylint: disable=missing-function-docstring


def test_references_are_parsed():
    references = series_model.parse_series_references(
        "valuation:get_price_to_earnings_ratio:AAPL, MSFT; get_inflation_rate:Germany\n"
        "rates:government_bond_yield_curve:United States:10Y"
    )

    assert [reference.tool for reference in references] == ["valuation", None, "rates"]
    assert references[0].entities == ["AAPL", "MSFT"]
    assert references[1].indicator == "get_inflation_rate"
    assert references[2].indicator == "get_government_bond_yield_curve"
    assert references[2].column == "10Y"

    with pytest.raises(ValueError, match="tool:indicator:entities"):
        series_model.parse_series_references("inflation")


def test_answers_become_series_by_date():
    years = pd.period_range("2020", periods=3, freq="Y")
    ratios = pd.DataFrame(
        [[30.0, 28.0, 25.0], [35.0, 33.0, 31.0]], index=["AAPL", "MSFT"], columns=years
    )
    reference = series_model.parse_series_references("valuation:get_pe:AAPL")[0]

    # The tickers are rows and the years columns; only the requested ticker is kept.
    assert list(series_model.to_series_columns(ratios, reference).columns) == ["AAPL"]

    days = pd.period_range("2026-01-01", periods=3, freq="D")
    curve = pd.DataFrame(
        np.full((3, 2), 0.04),
        index=days,
        columns=pd.MultiIndex.from_tuples(
            [("United States", "2Y"), ("United States", "10Y")]
        ),
    )
    ten_year = series_model.parse_series_references(
        "rates:get_government_bond_yield_curve:United States:10Y"
    )[0]
    assert list(series_model.to_series_columns(curve, ten_year).columns) == [
        "United States 10Y"
    ]

    missing = ten_year._replace(column="99Y")
    with pytest.raises(ValueError, match="Choose from: 10Y, 2Y"):
        series_model.to_series_columns(curve, missing)


def test_shared_names_get_the_indicator_added():
    days = pd.period_range("2026-01-01", periods=2, freq="D")
    first, second = series_model.parse_series_references(
        "valuation:get_price_to_earnings_ratio:AAPL; valuation:get_price_to_book_ratio:AAPL"
    )
    combined = series_model.combine_series(
        [
            (first, pd.DataFrame({"AAPL": [30.0, 31.0]}, index=days)),
            (second, pd.DataFrame({"AAPL": [40.0, 42.0]}, index=days)),
        ]
    )

    assert list(combined.columns) == [
        "AAPL Price To Earnings Ratio",
        "AAPL Price To Book Ratio",
    ]


@pytest.fixture
def indicator_answers(monkeypatch):
    """Answer every indicator with a monthly series per requested country."""
    calls = []
    rng = np.random.default_rng(0)
    months = pd.period_range("2010-01", periods=180, freq="M")

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append((module_name, method_name, category, kwargs))
        countries = kwargs.get("countries") or []
        return pd.DataFrame(
            {
                country: 0.03 + rng.normal(0, 0.002, len(months)).cumsum() / 10
                for country in countries
            },
            index=months,
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call_econometrics(**arguments):
    result = asyncio.run(mcp_controller.mcp.call_tool("econometrics", arguments))
    return str(result)


def test_an_econometrics_call_analyzes_other_indicators(indicator_answers):
    answer = call_econometrics(
        indicator="get_augmented_dickey_fuller",
        series="macroeconomics:get_inflation_rate:USA,Germany",
        start_date="2010-01-01",
    )

    # The indicator was retrieved through its own tool, with the country names resolved.
    module_name, method_name, category, kwargs = indicator_answers[-1]
    assert (module_name, method_name, category) == (
        "economics",
        "get_inflation_rate",
        "standalone",
    )
    assert kwargs["countries"] == ["United States", "Germany"]
    assert kwargs["start_date"] == "2010-01-01"
    assert "ADF Statistic" in answer and "Germany" in answer


def test_series_errors_are_explained(indicator_answers):
    assert "names no known indicator" in call_econometrics(
        indicator="get_ols", series="get_imaginary_metric:Germany"
    )
    assert "is not an indicator of `rates`" in call_econometrics(
        indicator="get_ols", series="rates:get_inflation_rate:Germany"
    )
    # Without series, the tickers are still required.
    assert "requires a `tickers` parameter" in call_econometrics(indicator="get_ols")
