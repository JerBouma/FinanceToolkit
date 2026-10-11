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
        ("P/E", ("valuation", "get_price_to_earnings_ratio")),
        ("return on equitty", ("profitability", "get_return_on_equity")),
        ("RSI", ("momentum", "get_relative_strength_index")),
        ("bollinger", ("volatility", "get_bollinger_bands")),
    ],
)
def test_the_best_matching_indicator_comes_first(query, expected):
    assert search(query)[0] == expected


def test_a_close_spelling_does_not_count_when_a_word_matches_exactly():
    # "sharpe" is close to "share", but the Sharpe ratio exists, so no per share metric.
    assert all("per_share" not in indicator for _, indicator in search("sharpe"))
