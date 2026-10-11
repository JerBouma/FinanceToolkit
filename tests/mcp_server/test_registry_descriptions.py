"""Registry Description Tests"""

# pylint: disable=missing-function-docstring

import asyncio

import yaml

from financetoolkit.mcp_server import mcp_controller


def test_long_tool_descriptions_are_complete():
    """The guidance at the end of a long description reaches the client."""
    with open("financetoolkit/mcp_server/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    configured = {
        group["tool_name"]: group["description"].strip()
        for group in config["tool_groups"]
        if group.get("description")
    }
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    for name in ("macroeconomics", "rates"):
        assert configured[name][-200:] in tools[name].description


def test_the_server_reports_the_finance_toolkit_version():
    """Clients show the server's version, which is the package's, not the MCP SDK's."""
    from importlib import metadata

    options = mcp_controller.mcp._mcp_server.create_initialization_options()

    assert options.server_version == metadata.version("financetoolkit")


def test_tickers_without_data_are_named_in_the_answer():
    from financetoolkit.mcp_server.diagnostics_model import missing_data_notes

    assert missing_data_notes(
        [
            "No quote data is available for EURUSD=X, BTC-USD.",
            "No quote data is available for EURUSD=X, BTC-USD.",
            "Something unrelated",
        ]
    ) == ["No quote data is available for EURUSD=X, BTC-USD."]


def test_the_search_tools_say_they_belong_to_the_finance_toolkit():
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    for name in ("search_metric_categories", "search_by_category", "search_metrics"):
        assert tools[name].description.startswith("Finance Toolkit:")

    # Looking up instruments is left to the Finance Database MCP server, whose tools do
    # not share a name with these.
    assert "search_instruments" not in tools
    assert "search_categories" not in tools


def test_a_category_of_one_tool_is_named_after_it():
    """An agent can go from a category straight to its tool."""
    from collections import defaultdict

    with open("financetoolkit/mcp_server/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tools_per_category = defaultdict(list)

    for group in config["tool_groups"]:
        category = group.get("index_category") or group["module_name"]
        tools_per_category[category].append(group["tool_name"])

    for category, tools in tools_per_category.items():
        if len(tools) == 1:
            assert tools == [category]
        assert category in config["category_descriptions"]


def test_an_unnamed_result_is_named_after_its_method():
    from financetoolkit.mcp_server.provider_model import readable_name

    assert readable_name("get_value_at_risk") == "Value at Risk"
    assert readable_name("get_return_on_equity") == "Return on Equity"
