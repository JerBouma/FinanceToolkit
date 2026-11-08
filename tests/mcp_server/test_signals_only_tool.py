"""Tests for signals_only on the signal and candlestick pattern methods of the MCP."""

import asyncio
import json

import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.provider_model import ToolkitProvider


@pytest.fixture
def answered_by(monkeypatch, technical_module):
    """Answer the tools with the offline Toolkit and record what the tool wrapper passes on."""
    calls = []

    def call_method(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        calls.append({"method": method_name, **kwargs})
        own = {
            key: value
            for key, value in kwargs.items()
            if key in ("signals_only", "patterns")
        }
        return getattr(technical_module, method_name)(**own)

    monkeypatch.setattr(ToolkitProvider, "call_method", call_method)
    return calls


def call(tool, indicator, **arguments):
    return asyncio.run(
        mcp_controller.mcp.call_tool(
            tool, {"indicator": indicator, "tickers": "AAPL,MSFT", **arguments}
        )
    )[0][0].text


def test_the_tools_offer_signals_only():
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    for name in ("momentum", "overlap"):
        assert "signals_only" in tools[name].inputSchema["properties"]


@pytest.mark.parametrize(
    ("tool", "indicator", "extra"),
    [
        ("momentum", "get_signals", {}),
        ("overlap", "get_candlestick_patterns", {"patterns": "Engulfing"}),
    ],
)
def test_signals_only_is_passed_on(answered_by, tool, indicator, extra):
    call(tool, indicator, signals_only=True, limit=-5, **extra)

    assert answered_by[-1]["method"] == indicator
    assert answered_by[-1]["signals_only"] is True


def test_a_limit_gives_every_ticker_its_last_signals(answered_by, technical_module):
    rows = json.loads(
        call(
            "overlap",
            "get_candlestick_patterns",
            signals_only=True,
            limit=-3,
            patterns="Engulfing",
        )
    )
    engulfing = technical_module.get_candlestick_patterns(patterns="Engulfing")[
        "Engulfing"
    ]

    for ticker in ("AAPL", "MSFT"):
        dates = [row["Date"] for row in rows if row["ticker"] == ticker]
        expected = engulfing[ticker][engulfing[ticker].abs().gt(0)].index[-3:]
        assert dates == [date.strftime("%Y-%m-%d") for date in expected]
    assert all(row["Engulfing"] != 0 for row in rows)


def test_a_ticker_without_signals_is_named(answered_by, technical_module):
    soldiers = technical_module.get_candlestick_patterns(
        patterns="Three White Soldiers"
    )
    quiet = [
        ticker
        for ticker in ("AAPL", "MSFT")
        if not soldiers["Three White Soldiers"][ticker].abs().gt(0).any()
    ]
    if not quiet:
        pytest.skip("Every ticker has the pattern in the offline data.")

    answer = json.loads(
        call(
            "overlap",
            "get_candlestick_patterns",
            signals_only=True,
            patterns="Three White Soldiers",
        )
    )

    assert f"{quiet[0]} has no signal in the period" in " ".join(answer["_notes"])
