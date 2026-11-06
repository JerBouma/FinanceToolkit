"""Tests for get_indicator_summary in the momentum tool."""

import asyncio

import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.provider_model import ToolkitProvider


@pytest.fixture
def answered_by(monkeypatch, technical_module):
    """Answer the tool with the offline Toolkit and record what the tool wrapper passes on."""
    calls = []

    def call_method(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        calls.append({"method": method_name, **kwargs})
        own = {
            key: value
            for key, value in kwargs.items()
            if key in ("metric", "below", "above", "forward_periods", "show_crossings")
        }
        return getattr(technical_module, method_name)(**own)

    monkeypatch.setattr(ToolkitProvider, "call_method", call_method)
    return calls


def call_momentum(**arguments):
    return asyncio.run(
        mcp_controller.mcp.call_tool(
            "momentum",
            {"indicator": "get_indicator_summary", "tickers": "AAPL,MSFT", **arguments},
        )
    )[0][0].text


def test_the_momentum_tool_offers_the_indicator_summary():
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}
    properties = tools["momentum"].inputSchema["properties"]

    assert "get_indicator_summary" in str(properties["indicator"])
    assert {"metric", "below", "above", "forward_periods", "show_crossings"} <= set(
        properties
    )


def test_the_indicator_summary_is_a_small_answer(answered_by):
    text = call_momentum(below=30, forward_periods=[5, 21])

    assert answered_by[-1]["below"] == 30  # noqa: PLR2004
    assert answered_by[-1]["forward_periods"] == [5, 21]
    # A row per ticker instead of the daily series.
    assert len(text) < 3000  # noqa: PLR2004
    for field in ("Percentile", "Minimum Date", "Crossings Below"):
        assert field in text
    assert "AAPL" in text and "MSFT" in text


def test_the_crossings_are_listed_per_ticker(answered_by):
    text = call_momentum(below=30, forward_periods=[5], show_crossings=True)

    assert answered_by[-1]["show_crossings"] is True
    assert "5-Period Return" in text
