"""Regression tests for the input validation the stress test added."""

import asyncio
import json
import logging

import pandas as pd
import pytest

from financetoolkit import Discovery, Economics, FixedIncome
from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics import (
    fmp_model as economics_fmp_model,
    helpers,
)
from financetoolkit.mcp_server import mcp_controller
from financetoolkit.utilities import validation_model
from financetoolkit.utilities.statistics_model import finalize_dataset

# mcp_controller is imported while collecting, before a test may replace FastMCP with a
# stub and reload the module, so the tool calls below always reach the real server.


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    """Start every test without an active cache another test may have left behind."""
    clear_active_cache()
    yield
    clear_active_cache()


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("2026-02-28", True),
        ("2026-02-30", False),
        ("2026-13-01", False),
        ("26-01-01", False),
        ("", False),
        (None, False),
    ],
)
def test_dates_must_exist(value, valid):
    assert validation_model.is_valid_date(value) is valid


def test_controllers_refuse_dates_that_do_not_exist():
    with pytest.raises(ValueError, match="2026-02-30"):
        Economics(start_date="2026-02-30", gmdb_source=False)
    with pytest.raises(ValueError, match="2026-13-01"):
        FixedIncome(end_date="2026-13-01")


def test_a_period_of_the_wrong_type_is_a_clear_type_error():
    with pytest.raises(TypeError, match="period must be text"):
        helpers.validate_period(12, ["monthly"], "inflation rate")
    with pytest.raises(TypeError, match="period must be text"):
        Economics(gmdb_source=False).get_unemployment_rate(period=3)


def test_countries_of_the_wrong_type_are_a_clear_type_error():
    data = pd.DataFrame(
        {"Japan": [1.0]}, index=pd.period_range("2026-01", periods=1, freq="M")
    )

    with pytest.raises(TypeError, match="countries must be a country name"):
        finalize_dataset(data, None, None, 4, countries=5, axis="rows", row_slice=True)
    with pytest.raises(TypeError, match="countries must be a country name"):
        FixedIncome().get_government_bond_yield_curve(countries=7)


def test_an_empty_window_explains_itself(caplog):
    data = pd.DataFrame(
        {"Japan": [1.0]}, index=pd.period_range("2026-09-25", periods=1, freq="D")
    )

    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        result = finalize_dataset(
            data,
            "2026-09-26",
            "2026-09-27",
            4,
            axis="rows",
            row_slice=True,
            indicator_name="Overnight Rate",
        )

    assert result.empty
    assert (
        "no Overnight Rate observations between 2026-09-26 and 2026-09-27"
        in caplog.text
    )


def test_a_future_range_is_not_requested(monkeypatch, caplog):
    def never(*args):
        raise AssertionError("no request should be made")

    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        result = helpers.collect_ranged_data(
            "ECB", "series", "x", never, "2999-01-01", "2999-02-01", "test"
        )

    assert result.empty
    assert "lies in the future" in caplog.text


def test_a_source_without_observations_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        helpers.collect_ranged_data(
            "BIS",
            "dataset",
            "x",
            lambda start, end: pd.DataFrame(),
            "2026-01-01",
            "2026-02-01",
            "test",
        )

    assert "BIS returned no test between 2026-01-01 and 2026-02-01" in caplog.text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"start_date": "2026-10-03", "end_date": "2026-09-01"},
            "must be on or before",
        ),
        ({"start_date": "yesterday"}, "must be a date written as YYYY-MM-DD"),
        ({"impact": "Extreme"}, "must be 'Low', 'Medium', 'High' or 'All'"),
    ],
)
def test_the_calendar_refuses_arguments_it_cannot_honour(arguments, message):
    with pytest.raises(ValueError, match=message):
        economics_fmp_model.get_economic_calendar("key", **arguments)


@pytest.mark.parametrize(
    ("method", "arguments", "error", "message"),
    [
        (
            "get_earnings_calendar",
            {"start_date": "2026-10-03", "end_date": "2026-10-01"},
            ValueError,
            "on or before",
        ),
        (
            "get_earnings_calendar",
            {"start_date": "2026-13-01"},
            ValueError,
            "YYYY-MM-DD",
        ),
        ("get_sec_filings_8k", {"limit": 0}, ValueError, "one or more"),
        ("get_sec_filings_8k", {"limit": "ten"}, TypeError, "whole number"),
        ("get_insider_trading_latest", {"page": -1}, ValueError, "zero or more"),
        ("get_sector_performance", {"date": "2026/09/31"}, ValueError, "YYYY-MM-DD"),
        (
            "search_crypto_news",
            {"symbols": "BTCUSD", "pages": 0},
            ValueError,
            "one or more",
        ),
    ],
)
def test_discovery_refuses_arguments_the_endpoints_would_ignore(
    method, arguments, error, message
):
    with pytest.raises(error, match=message):
        getattr(Discovery(api_key="key"), method)(**arguments)


def _call_tool(tool, arguments):
    result = asyncio.run(mcp_controller.mcp.call_tool(tool, arguments))
    content = result[0] if isinstance(result, tuple) else result.content
    return "".join(getattr(item, "text", "") for item in content)


def test_mcp_refuses_a_date_it_cannot_read():
    text = _call_tool(
        "rates",
        {
            "indicator": "get_overnight_rate",
            "countries": "Euro Area",
            "start_date": "last tuesday",
        },
    )

    assert text.startswith("Invalid `start_date`")


def test_mcp_requires_countries_for_country_indicators():
    text = _call_tool(
        "macroeconomics", {"indicator": "get_consumer_price_index", "period": "monthly"}
    )

    assert "requires a `countries` parameter" in text


def test_mcp_accepts_countries_as_a_list(monkeypatch):
    from financetoolkit.mcp_server.provider_model import ToolkitProvider

    received = {}

    def fake(self, module_name, method_name, category, **kwargs):
        received.update(kwargs)
        return pd.DataFrame(
            {"Japan": [0.02]}, index=pd.period_range("2026-08", periods=1, freq="M")
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", fake)

    text = _call_tool(
        "macroeconomics",
        {
            "indicator": "get_inflation_rate",
            "countries": ["Japan", "Germany"],
            "period": "monthly",
        },
    )

    assert received["countries"] == ["Japan", "Germany"]
    assert json.loads(text)[0]["Japan"] == 0.02
