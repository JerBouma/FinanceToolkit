"""Tests for the MCP-only method defaults (config.yaml method_defaults)."""

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.provider_model import ToolkitProvider


@pytest.fixture
def captured(monkeypatch):
    """Record what the tool wrapper passes to the provider instead of fetching data."""
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append(kwargs)
        return pd.DataFrame({"Event": ["CPI"]})

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call_calendar(**arguments):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "macroeconomics", {"indicator": "get_economic_calendar", **arguments}
        )
    )


def test_economic_calendar_defaults_to_high_impact_around_today(captured):
    call_calendar(countries="United States")

    today = datetime.now()
    assert captured[-1]["impact"] == "High"
    assert captured[-1]["start_date"] == (today - timedelta(days=23)).strftime(
        "%Y-%m-%d"
    )
    assert captured[-1]["end_date"] == (today + timedelta(days=7)).strftime("%Y-%m-%d")


def test_economic_calendar_explicit_values_win(captured):
    call_calendar(impact="All", start_date="2026-09-01", end_date="2026-09-10")

    assert captured[-1]["impact"] == "All"
    assert (captured[-1]["start_date"], captured[-1]["end_date"]) == (
        "2026-09-01",
        "2026-09-10",
    )


def call_performance(**arguments):
    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance",
            {"indicator": "get_sharpe_ratio", "tickers": "AAPL", **arguments},
        )
    )

    return str(result)


def test_daily_performance_needs_a_rolling_window(captured):
    """Within-period results need a longer period; a rolling window reads daily data."""
    assert "does not support" in call_performance(period="daily")
    assert not captured

    call_performance(period="daily", rolling=20)

    assert captured[-1]["period"] == "daily"
    assert captured[-1]["rolling"] == 20  # noqa: PLR2004
