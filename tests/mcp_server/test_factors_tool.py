"""Tests for the factors MCP tool."""

import asyncio

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
        calls.append({"module": module_name, "method": method_name, **kwargs})
        return pd.DataFrame({"1 Month": [0.05]}, index=["Mean IC"])

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def test_the_factors_tool_takes_a_factor_by_name():
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    properties = tools["factors"].inputSchema["properties"]

    factor = properties["factor"]
    assert {option.get("type") for option in factor.get("anyOf", [factor])} - {
        "null"
    } == {"string"}
    assert "get_price_to_earnings_ratio" in factor["description"]
    assert {"period", "horizons", "quantiles", "summary", "periods_apart"} <= set(
        properties
    )


def test_a_factor_evaluation_does_not_report_its_tickers_as_missing(captured):
    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "factors",
            {
                "indicator": "get_information_coefficient",
                "tickers": "AAPL,MSFT,JNJ",
                "factor": "get_price_to_earnings_ratio",
                "summary": True,
            },
        )
    )[0][0].text

    assert captured[-1]["module"] == "factors"
    assert captured[-1]["factor"] == "get_price_to_earnings_ratio"
    assert "No data is available" not in result
