"""Usage Analytics Tests"""

# pylint: disable=missing-function-docstring

import asyncio
import hashlib
import json

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from financetoolkit.mcp_server import analytics_model


def make_analytics(tmp_path, key="KEY") -> analytics_model.UsageAnalytics:
    return analytics_model.UsageAnalytics(
        server_name="Test Server",
        location=tmp_path / "mcp_stats.json",
        secret=b"secret",
        resolve_api_key=lambda: key,
    )


def get_ratio(ticker: str, period: str = "yearly", rounding: int | None = 4) -> str:
    """Return a ratio."""
    return f"{ticker} {period} {rounding}"


def test_analytics_are_off_unless_switched_on(tmp_path, monkeypatch):
    monkeypatch.delenv(analytics_model.ENABLED_ENVIRONMENT_VARIABLE, raising=False)

    assert (
        analytics_model.create_from_environment(
            "Test Server", tmp_path / "mcp_stats.json", b"secret", lambda: ""
        )
        is None
    )
    assert not (tmp_path / "mcp_stats.json").exists()


def test_a_counted_tool_keeps_its_input_schema(tmp_path):
    mcp = FastMCP("test")
    analytics = make_analytics(tmp_path)

    mcp.add_tool(get_ratio, name="plain")
    mcp.add_tool(analytics.counted("counted", get_ratio), name="counted")
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}

    assert tools["counted"].inputSchema == tools["plain"].inputSchema
    assert tools["counted"].description == tools["plain"].description


def test_calls_are_counted_per_tool_day_and_user(tmp_path):
    analytics = make_analytics(tmp_path)
    tool = analytics.counted("ratios", get_ratio)

    def failing_tool(ticker: str) -> str:
        raise ValueError(ticker)

    tool("AAPL")
    tool("MSFT")
    with pytest.raises(ValueError, match="AAPL"):
        analytics.counted("failing", failing_tool)("AAPL")

    assert analytics.total_calls == 3  # noqa: PLR2004
    assert analytics.calls_per_tool == {"ratios": 2, "failing": 1}
    assert analytics.failures_per_tool == {"failing": 1}
    # The key itself is never stored, only its HMAC under the server's secret.
    assert list(analytics.calls_per_user) == [analytics.user_id("KEY")]
    assert "KEY" not in json.dumps(analytics.to_dict())
    assert analytics.user_id("KEY") != f"u_{hashlib.sha256(b'KEY').hexdigest()[:12]}"


def test_an_async_tool_is_counted(tmp_path):
    analytics = make_analytics(tmp_path, key="")

    async def get_quote(ticker: str) -> str:
        return ticker

    assert asyncio.run(analytics.counted("quote", get_quote)("AAPL")) == "AAPL"
    assert analytics.calls_per_user == {"anonymous": 1}


def test_counts_survive_a_restart(tmp_path):
    analytics = make_analytics(tmp_path)
    tool = analytics.counted("ratios", get_ratio)

    for _ in range(analytics_model.SAVE_EVERY_CALLS):
        tool("AAPL")

    restarted = make_analytics(tmp_path)
    restarted.load()

    assert restarted.total_calls == analytics_model.SAVE_EVERY_CALLS
    assert restarted.calls_per_tool == {"ratios": analytics_model.SAVE_EVERY_CALLS}


def test_users_counted_with_an_unkeyed_hash_are_not_counted_twice(tmp_path):
    legacy = f"u_{hashlib.sha256(b'KEY').hexdigest()[:12]}"
    (tmp_path / "mcp_stats.json").write_text(
        json.dumps(
            {
                "total_calls": 7,
                "calls_per_user": {legacy: 7},
                "first_seen": {legacy: "2026-10-01"},
            }
        )
    )
    analytics = make_analytics(tmp_path)
    analytics.load()

    analytics.counted("ratios", get_ratio)("AAPL")

    assert analytics.calls_per_user == {analytics.user_id("KEY"): 8}
    assert analytics.first_seen == {analytics.user_id("KEY"): "2026-10-01"}


def test_the_stats_page_shows_totals_only(tmp_path):
    mcp = FastMCP("test")
    analytics = make_analytics(tmp_path)
    analytics.counted("ratios", get_ratio)("AAPL")
    analytics.register_route(mcp)

    stats = TestClient(mcp.streamable_http_app()).get("/stats").json()

    assert stats["total_calls"] == 1
    assert stats["unique_users"] == 1
    assert stats["top_tools"][0]["tool"] == "ratios"
    assert len(stats["daily_calls"]) == analytics_model.DAILY_SERIES_DAYS
    assert "calls_per_user" not in stats
