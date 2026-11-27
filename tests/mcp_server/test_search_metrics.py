"""Tests for the indicator search of the MCP server (search_metrics)."""

import asyncio
import re

import pytest

from financetoolkit.mcp_server import mcp_controller


def search(query):
    result = asyncio.run(
        mcp_controller.mcp.call_tool("search_metrics", {"query": query})
    )
    text = result[0].text

    # Each result is a call line: "- `tool get_indicator(...)`: summary".
    return [
        (match.group(1), match.group(2))
        for line in text.splitlines()
        if (match := re.match(r"- `(\w+) (\w+)\(", line))
    ]


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
    assert "keep it to a few keywords" in result[0].text
    assert len(long_query) > tools_model.MAX_QUERY_CHARACTERS


def test_an_indicator_name_returns_its_description_and_parameters():
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "search_metrics", {"query": "get_yield_decomposition"}
        )
    )[0].text

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


def test_an_indicator_name_returns_the_whole_description_up_to_its_sections():
    text = asyncio.run(
        mcp_controller.mcp.call_tool("search_metrics", {"query": "get_aroon_indicator"})
    )[0].text
    # The summary sentence, the explanation and the formula, but not the library references or Args.
    assert "how many periods ago the highest high" in text
    assert "- Aroon Oscillator = Aroon Up — Aroon Down" in text
    assert "Referenced against" not in text
    assert "ta-lib.org" not in text


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("mean reversion", "get_unit_root_test"),
        ("mean reversion", "get_variance_ratio_test"),
        ("mean reversion", "get_hurst_exponent"),
        ("is the stock trending or mean reverting", "get_unit_root_test"),
        ("is the stock trending or mean reverting", "get_variance_ratio_test"),
        ("is the stock trending or mean reverting", "get_hurst_exponent"),
        ("golden cross", "get_signals"),
        ("portfolio VaR", "get_component_value_at_risk"),
        ("portfolio VaR", "get_portfolio_risk_metrics"),
        ("Jensen's alpha", "get_alpha"),
        ("Jensen's alpha", "get_capital_asset_pricing_model"),
        ("Jensen's alpha", "get_fama_and_french_model"),
        ("yield curve inverted", "get_yield_curve_slope"),
        ("yield curve inverted", "get_yield_curve_spread"),
        ("2s10s", "get_yield_curve_spread"),
        ("FCF yield", "get_free_cash_flow_yield"),
    ],
)
def test_the_words_users_ask_with_find_the_indicator(query, expected):
    """The phrasings of a usage test, found through the indicators' aliases."""
    assert expected in [indicator for _, indicator in search(query)[:5]]


def test_the_first_result_of_a_phrase_is_the_most_direct_answer():
    assert search("golden cross")[0] == ("momentum", "get_signals")
    assert search("portfolio VaR")[0] == ("risk", "get_component_value_at_risk")
    assert search("Jensen's alpha")[0] == ("performance", "get_alpha")
    assert search("FCF yield")[0] == ("solvency", "get_free_cash_flow_yield")


def test_a_search_result_shows_how_to_call_each_indicator():
    text = asyncio.run(
        mcp_controller.mcp.call_tool("search_metrics", {"query": "Jensen's alpha"})
    )[0].text
    line = next(line for line in text.splitlines() if "get_alpha(" in line)

    # The default leads the options, followed by the one the query names.
    assert "method='standard'|'jensen'" in line
    # The parameters most of the tool's indicators take are named once, not per indicator.
    assert "growth=" not in line
    assert (
        "Most indicators also take period, growth, lag, standardize (performance)"
        in text
    )


def test_a_call_line_marks_required_parameters_and_shortens_long_lists():
    from financetoolkit.mcp_server.registry.tools_model import call_line

    entry = {
        "tool": "risk",
        "indicator": "get_example",
        "common_parameters": ["period", "growth"],
        "call_parameters": [
            {
                "name": "growth",
                "options": [],
                "default": False,
                "required": False,
                "conditional": False,
            },
            {
                "name": "refit_every",
                "options": [],
                "default": 21,
                "required": False,
                "conditional": True,
            },
            {
                "name": "method",
                "options": ["historic", "gaussian", "evt", "garch"],
                "default": "historic",
                "required": False,
                "conditional": False,
            },
            {
                "name": "ticker",
                "options": [],
                "default": None,
                "required": True,
                "conditional": False,
            },
        ],
    }

    assert call_line(entry, ["evt"]) == (
        "risk get_example(ticker (required), method='historic'|'evt'|'gaussian'|..., refit_every=21)"
    )


def test_an_indicator_name_names_the_tool_parameters_its_schema_describes():
    text = asyncio.run(
        mcp_controller.mcp.call_tool("search_metrics", {"query": "get_value_at_risk"})
    )[0].text

    assert "- `alpha`:" in text
    assert "The tool also takes tickers, start_date, end_date" in text
    assert "- `tickers`:" not in text
