"""Tests for get_factor_analysis in the MCP tools of the ratios, technicals, models, risk and performance."""

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
        return pd.DataFrame(
            {"Mean IC": [0.05]},
            index=pd.MultiIndex.from_tuples([("get_return_on_equity", "1 Month")]),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def test_each_module_tool_offers_the_factor_analysis_with_metrics_and_detail():
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    for name in ("valuation", "momentum", "models", "risk", "performance"):
        properties = tools[name].inputSchema["properties"]
        assert "get_factor_analysis" in str(properties["indicator"]), name
        assert {"metrics", "detail", "horizons", "quantiles", "periods_apart"} <= set(
            properties
        ), name
        assert "predictive power" in tools[name].description, name

    assert "factors" not in tools
    # The other ratio and indicator tools point to the module's factor analysis.
    assert "metrics='profitability'" in tools["profitability"].description
    assert "metrics='volatility'" in tools["volatility"].description


def test_the_factor_analysis_passes_the_parameters_of_the_metrics_on(captured):
    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "valuation",
            {
                "indicator": "get_factor_analysis",
                "tickers": "AAPL,MSFT,JNJ",
                "metrics": "profitability",
                "trailing": 4,
            },
        )
    )[0][0].text

    assert captured[-1]["method"] == "get_factor_analysis"
    assert captured[-1]["metrics"] == "profitability"
    # trailing is a parameter of the ratios, not of get_factor_analysis itself.
    assert captured[-1]["trailing"] == 4  # noqa: PLR2004
    # The answer is for the tickers together, so none is reported as missing.
    assert "No data is available" not in result
    assert "AAPL" not in result.split("get_return_on_equity")[-1]


def test_search_metrics_finds_the_factor_analysis_as_predictive_power():
    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "search_metrics", {"query": "predictive power of a ratio"}
        )
    )[0][0].text

    assert "get_factor_analysis" in result.split("\n\n")[0] or (
        "get_factor_analysis" in result[:600]
    )
