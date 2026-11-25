"""Tests that an indicator tool answers several of its indicators in one call, each as a call
of its own with the same arguments, under its name."""

import asyncio
import json

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.analytics import analytics_model
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from financetoolkit.mcp_server.registry.registry_controller import (
    MAX_INDICATORS,
    _indicator_count,
)
from tests.mcp_server.answers import read_answer

VALUATION = [
    "get_price_to_earnings_ratio",
    "get_ev_to_ebitda_ratio",
    "get_price_to_book_ratio",
]


@pytest.fixture
def calls(monkeypatch):
    """Answer every indicator with a small table of its own, and keep the calls."""
    made = []

    def call_method(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        made.append((method_name, kwargs))
        return pd.DataFrame(
            {"2024": [float(len(made))], "2025": [float(len(made)) + 0.5]},
            index=pd.Index(["AMZN"]),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", call_method)
    monkeypatch.setattr(
        ToolkitProvider, "get_transformation_notes", lambda self, **kwargs: []
    )
    return made


def call_text(tool, arguments):
    return asyncio.run(mcp_controller.mcp.call_tool(tool, arguments))[0].text


def call(tool, arguments):
    return read_answer(call_text(tool, arguments))


def test_several_indicators_are_answered_each_under_its_name(calls):
    result = call("valuation", {"indicator": VALUATION, "tickers": "AMZN"})

    assert [name for name, _ in calls] == VALUATION
    assert list(result) == VALUATION
    assert result["get_ev_to_ebitda_ratio"] == [
        {"ticker": "AMZN", "2024": 2.0, "2025": 2.5}
    ]


def test_comma_separated_indicators_are_read_as_several(calls):
    result = call(
        "valuation", {"indicator": " , ".join(VALUATION[:2]), "tickers": "AMZN"}
    )

    assert list(result) == VALUATION[:2]


def test_an_unknown_indicator_is_answered_with_its_error_only(calls):
    result = call(
        "valuation",
        {
            "indicator": ["get_price_to_earnings_ratio", "get_price_to_nothing"],
            "tickers": "AMZN",
        },
    )

    assert result["get_price_to_earnings_ratio"] == [
        {"ticker": "AMZN", "2024": 1.0, "2025": 1.5}
    ]
    assert result["get_price_to_nothing"]["error"].startswith(
        "Unknown indicator `get_price_to_nothing` for `valuation`."
    )


def test_an_indicator_of_another_tool_is_answered_by_that_tool(calls):
    result = call(
        "valuation",
        {
            "indicator": ["get_price_to_book_ratio", "get_current_ratio"],
            "tickers": "AMZN",
        },
    )

    assert [name for name, _ in calls] == [
        "get_price_to_book_ratio",
        "get_current_ratio",
    ]
    assert result["get_current_ratio"]["_notes"] == [
        "`get_current_ratio` is an indicator of `liquidity`, so `liquidity` answered it."
    ]
    assert "_notes" not in result


def test_a_parameter_only_one_indicator_takes_is_noted_on_the_others(calls):
    result = call(
        "valuation",
        {
            "indicator": ["get_price_to_earnings_ratio", "get_ev_to_ebitda_ratio"],
            "tickers": "AMZN",
            "include_dividends": True,
        },
    )

    assert calls[0][1]["include_dividends"] is True
    assert "include_dividends" not in calls[1][1]
    assert "_notes" not in result["get_price_to_earnings_ratio"]
    assert result["get_ev_to_ebitda_ratio"]["_notes"] == [
        "`include_dividends` is not used by `get_ev_to_ebitda_ratio` and was ignored."
    ]


def test_a_note_every_indicator_has_is_given_once(calls):
    result = call(
        "valuation", {"indicator": VALUATION[:2], "tickers": "AMZN", "perid": "x"}
    )

    assert result["_notes"] == [
        "`perid` is not a parameter of `valuation` and was ignored; did you mean `period`?"
    ]
    assert result["get_price_to_earnings_ratio"] == [
        {"ticker": "AMZN", "2024": 1.0, "2025": 1.5}
    ]


def test_more_indicators_than_the_maximum_are_refused(calls):
    names = [
        "get_price_to_earnings_ratio",
        "get_ev_to_ebitda_ratio",
        "get_price_to_book_ratio",
        "get_price_to_sales_ratio",
        "get_earnings_yield",
        "get_dividend_yield",
    ]

    assert (
        len(call("valuation", {"indicator": names[:MAX_INDICATORS], "tickers": "AMZN"}))
        == MAX_INDICATORS
    )
    calls.clear()
    assert call_text("valuation", {"indicator": names, "tickers": "AMZN"}) == (
        f"`valuation` received 6 indicators; at most {MAX_INDICATORS} can be calculated "
        "in one call. Split them over several calls."
    )
    assert not calls


def test_several_indicators_share_the_size_of_a_response(monkeypatch):
    days = pd.period_range("2000-01-03", periods=6000, freq="D")
    monkeypatch.setattr(
        ToolkitProvider,
        "call_method",
        lambda self, module_name, method_name, category, **kwargs: pd.DataFrame(  # noqa: ARG005
            {"AMZN": [123.456789] * len(days), "MSFT": [234.567891] * len(days)},
            index=days,
        ),
    )
    monkeypatch.setattr(
        ToolkitProvider, "get_transformation_notes", lambda self, **kwargs: []
    )

    one = call_text("valuation", {"indicator": VALUATION[0], "tickers": "AMZN,MSFT"})
    several = call_text("valuation", {"indicator": VALUATION, "tickers": "AMZN,MSFT"})

    answer = json.loads(several)
    assert len(several) <= len(one) + 1000  # noqa: PLR2004
    assert all(len(answer[name]["rows"]) < len(days) for name in VALUATION)
    assert len(answer[VALUATION[0]]["rows"]) < len(json.loads(one)["rows"])


def test_the_schema_takes_one_indicator_or_a_list():
    tool = next(
        tool
        for tool in asyncio.run(mcp_controller.mcp.list_tools())
        if tool.name == "valuation"
    )
    schema = tool.inputSchema

    assert "get_price_to_earnings_ratio" in schema["$defs"]["indicator"]["enum"]
    assert schema["properties"]["indicator"]["anyOf"] == [
        {"$ref": "#/$defs/indicator"},
        {
            "type": "array",
            "items": {"$ref": "#/$defs/indicator"},
            "maxItems": MAX_INDICATORS,
        },
    ]


def test_the_usage_statistics_count_each_indicator(tmp_path):
    analytics = analytics_model.UsageAnalytics(
        server_name="Test Server",
        location=tmp_path / "mcp_stats.json",
        secret=b"secret",
        resolve_api_key=lambda: "KEY",
    )

    def tool(indicator: str, tickers: str = "") -> str:  # noqa: ARG001
        return "answer"

    counted = analytics.counted("valuation", tool, _indicator_count)
    counted(indicator="get_price_to_earnings_ratio")
    counted(indicator="get_price_to_earnings_ratio,get_ev_to_ebitda_ratio")
    counted(indicator=["a", "b", "c", "d", "e", "f"])

    assert analytics.calls_per_tool == {"valuation": 4}
    assert analytics.total_calls == 4  # noqa: PLR2004


def test_a_message_every_indicator_gives_is_given_once():
    from financetoolkit.mcp_server.registry.registry_controller import _combine_answers

    message = "A FinancialModelingPrep API key is required for this tool."
    assert (
        _combine_answers(
            {"get_price_to_earnings_ratio": message, "get_price_to_book_ratio": message}
        )
        == message
    )
