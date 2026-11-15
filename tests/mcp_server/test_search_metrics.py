"""Tests for the indicator search of the MCP server (search_metrics)."""

import asyncio

import pytest

from financetoolkit.mcp_server import mcp_controller


def search(query):
    result = asyncio.run(
        mcp_controller.mcp.call_tool("search_metrics", {"query": query})
    )
    text = result[0][0].text
    rows = [line.split("|") for line in text.splitlines() if line.startswith("| `")]

    return [(row[1].strip(" `"), row[2].strip(" `")) for row in rows]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("free cash flow yield", ("solvency", "get_free_cash_flow_yield")),
        ("value at risk", ("risk", "get_value_at_risk")),
        ("moving average", ("overlap", "get_moving_average")),
        ("exponential moving average", ("overlap", "get_moving_average")),
        ("DEMA", ("overlap", "get_moving_average")),
        ("real GDP in USD", ("macroeconomics", "get_gross_domestic_product")),
        ("P/E", ("valuation", "get_price_to_earnings_ratio")),
        ("return on equitty", ("profitability", "get_return_on_equity")),
        ("RSI", ("momentum", "get_relative_strength_index")),
        ("bollinger", ("volatility", "get_bollinger_bands")),
        ("how expensive is the stock", ("valuation", "get_price_to_earnings_ratio")),
        ("bankruptcy risk", ("models", "get_altman_z_score")),
        ("correlation between stocks", ("performance", "get_correlation_matrix")),
        ("latest inflation", ("macroeconomics", "get_inflation_rate")),
        ("current ratio", ("liquidity", "get_current_ratio")),
    ],
)
def test_the_best_matching_indicator_comes_first(query, expected):
    assert search(query)[0] == expected


def test_a_close_spelling_does_not_count_when_a_word_matches_exactly():
    # "sharpe" is close to "share", but the Sharpe ratio exists, so no per share metric.
    assert all("per_share" not in indicator for _, indicator in search("sharpe"))


def test_a_search_is_kept_to_a_few_words():
    from financetoolkit.mcp_server.registry import tools_model

    long_query = " ".join(["volatility"] * 100)
    result = asyncio.run(
        mcp_controller.mcp.call_tool("search_metrics", {"query": long_query})
    )
    assert "keep it to a few keywords" in result[0][0].text
    assert len(long_query) > tools_model.MAX_QUERY_CHARACTERS


def test_an_indicator_name_returns_its_description_and_parameters():
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "search_metrics", {"query": "get_yield_decomposition"}
        )
    )[0][0].text

    assert text.startswith("### `get_yield_decomposition` (tool `rates`)")
    assert "Nominal Yield = Real Yield + Breakeven Inflation" in text
    assert "- `maturities`:" in text


def test_a_partial_indicator_name_is_searched_by_its_words():
    assert search("get_maximum_sharpe")[0] == (
        "portfolio",
        "get_maximum_sharpe_weights",
    )


def test_compact_tools_keep_the_first_clause_and_the_indicator_description(
    monkeypatch,
):
    from financetoolkit.mcp_server.registry import registry_controller

    monkeypatch.setenv("FT_MCP_COMPACT_TOOLS", "1")
    assert registry_controller.compact_tools_enabled()

    description = registry_controller._compact_description(
        "Interest rate data (central bank policy rates, ...). More text.",
        "get_central_bank_policy_rate",
    )
    assert description.startswith("Interest rate data. Pass the indicator by name")
    assert "'get_central_bank_policy_rate'" in description

    class Tool:
        parameters = {
            "properties": {
                "indicator": {"description": "Name of the metric."},
                "start_date": {"description": "The first date."},
            }
        }

    class Manager:
        def get_tool(self, name):
            return Tool

    class Server:
        _tool_manager = Manager()

    registry_controller._strip_schema_descriptions(Server(), "rates")
    assert Tool.parameters["properties"] == {
        "indicator": {"description": "Name of the metric."},
        "start_date": {},
    }
