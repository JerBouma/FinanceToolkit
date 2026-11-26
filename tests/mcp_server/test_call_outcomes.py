"""Tests that the usage statistics count how each call ended, per tool and indicator, without
storing anything the caller typed."""

import asyncio
import json
import threading

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.analytics import analytics_model
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from financetoolkit.mcp_server.registry.registry_controller import _indicator_count
from financetoolkit.utilities.error_model import InvalidArgumentError
from financetoolkit.utilities.logger_model import get_logger


def make_analytics(tmp_path) -> analytics_model.UsageAnalytics:
    return analytics_model.UsageAnalytics(
        server_name="Test Server",
        location=tmp_path / "mcp_stats.json",
        secret=b"secret",
        resolve_api_key=lambda: "KEY",
    )


@pytest.fixture
def answers(monkeypatch):
    """Answer every indicator with what the test puts under its name: a table, or an
    exception to raise."""
    by_indicator: dict[str, object] = {}

    def call_method(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        answer = by_indicator.get(
            method_name,
            pd.DataFrame({"2024": [1.0], "2025": [1.5]}, index=pd.Index(["AMZN"])),
        )
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer()
        return answer

    monkeypatch.setattr(ToolkitProvider, "call_method", call_method)
    monkeypatch.setattr(
        ToolkitProvider, "get_transformation_notes", lambda self, **kwargs: []
    )
    return by_indicator


def counted_tool(analytics, tool_name):
    """The registered tool of the server, counted as a hosted server counts it."""
    function = mcp_controller.mcp._tool_manager.get_tool(tool_name).fn
    tool = analytics.counted(tool_name, function, _indicator_count)

    def call(**arguments):
        return asyncio.run(tool(**arguments))

    return call


def test_an_answer_with_data_is_counted_as_data(tmp_path, answers):  # noqa: ARG001
    analytics = make_analytics(tmp_path)
    counted_tool(analytics, "valuation")(
        indicator="get_price_to_earnings_ratio", tickers="AMZN"
    )

    assert analytics.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"data": 1}
    }


def test_an_answer_cut_to_the_size_limit_is_counted_as_cut(tmp_path, answers):
    days = pd.period_range("2000-01-03", periods=6000, freq="D")
    answers["get_price_to_earnings_ratio"] = pd.DataFrame(
        {"AMZN": [123.456789] * len(days), "MSFT": [234.567891] * len(days)},
        index=days,
    )
    analytics = make_analytics(tmp_path)
    answer = counted_tool(analytics, "valuation")(
        indicator="get_price_to_earnings_ratio", tickers="AMZN,MSFT"
    )

    assert "was cut" in answer
    assert analytics.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"cut": 1}
    }


def test_an_answer_without_data_is_counted_as_empty(tmp_path, answers):
    answers["get_price_to_earnings_ratio"] = pd.DataFrame(
        {"2024": [float("nan")]}, index=pd.Index(["AMZN"])
    )
    answers["get_price_to_book_ratio"] = KeyError("AMZN")
    analytics = make_analytics(tmp_path)
    tool = counted_tool(analytics, "valuation")
    tool(indicator="get_price_to_earnings_ratio", tickers="AMZN")
    tool(indicator="get_price_to_book_ratio", tickers="AMZN")

    assert analytics.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"empty": 1},
        "valuation:get_price_to_book_ratio": {"empty": 1},
    }


def test_a_refused_call_is_counted_as_invalid(tmp_path, answers):
    answers["get_price_to_book_ratio"] = InvalidArgumentError("Unknown method 'x'.")
    analytics = make_analytics(tmp_path)
    tool = counted_tool(analytics, "valuation")
    tool(indicator="get_price_to_earnings_ratio", tickers="AMZN", start_date="2024-13")
    tool(indicator="get_price_to_earnings_ratio")
    tool(indicator="get_price_to_book_ratio", tickers="AMZN")
    tool(indicator="get_price_to_earnings_ratio", tickers="AMZN", limit="five")

    assert analytics.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"invalid": 3},
        "valuation:get_price_to_book_ratio": {"invalid": 1},
    }
    # A refused call still returned, so it is no failed call.
    assert analytics.failed_calls == 0


def test_an_answer_without_data_for_want_of_an_api_key_is_invalid(tmp_path, answers):
    def without_key():
        get_logger().warning("No FinancialModelingPrep API key was provided.")
        return pd.DataFrame()

    answers["get_price_to_earnings_ratio"] = without_key
    analytics = make_analytics(tmp_path)
    counted_tool(analytics, "valuation")(
        indicator="get_price_to_earnings_ratio", tickers="AMZN"
    )

    assert analytics.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"invalid": 1}
    }


def test_an_unknown_indicator_is_stored_without_its_name(
    tmp_path, answers
):  # noqa: ARG001
    analytics = make_analytics(tmp_path)
    counted_tool(analytics, "valuation")(
        indicator="get_my_secret_portfolio_of_jan", tickers="SECRETCO"
    )
    analytics.save()

    assert analytics.outcomes_per_indicator == {"valuation:(unknown)": {"invalid": 1}}
    stored = (tmp_path / "mcp_stats.json").read_text()
    assert "secret_portfolio" not in stored
    assert "SECRETCO" not in stored


def test_an_unexpected_exception_is_counted_as_an_error(tmp_path, answers):
    answers["get_price_to_earnings_ratio"] = RuntimeError("boom")
    analytics = make_analytics(tmp_path)
    counted_tool(analytics, "valuation")(
        indicator="get_price_to_earnings_ratio", tickers="AMZN"
    )

    assert analytics.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"error": 1}
    }


def test_several_indicators_are_counted_each(tmp_path, answers):
    answers["get_price_to_book_ratio"] = pd.DataFrame()
    analytics = make_analytics(tmp_path)
    counted_tool(analytics, "valuation")(
        indicator=[
            "get_price_to_earnings_ratio",
            "get_price_to_book_ratio",
            "get_not_an_indicator",
        ],
        tickers="AMZN",
    )

    assert analytics.calls_per_tool == {"valuation": 3}
    assert analytics.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"data": 1},
        "valuation:get_price_to_book_ratio": {"empty": 1},
        "valuation:(unknown)": {"invalid": 1},
    }


def test_too_many_indicators_count_once_for_the_tool(tmp_path, answers):  # noqa: ARG001
    analytics = make_analytics(tmp_path)
    counted_tool(analytics, "valuation")(
        indicator=["a", "b", "c", "d", "e", "f"], tickers="AMZN"
    )

    assert analytics.outcomes_per_indicator == {"valuation": {"invalid": 1}}


def test_an_indicator_of_another_tool_counts_for_that_tool(
    tmp_path, answers
):  # noqa: ARG001
    analytics = make_analytics(tmp_path)
    counted_tool(analytics, "valuation")(indicator="get_current_ratio", tickers="AMZN")

    assert analytics.calls_per_tool == {"valuation": 1}
    assert analytics.outcomes_per_indicator == {
        "liquidity:get_current_ratio": {"data": 1}
    }


def test_a_tool_that_reports_nothing_counts_by_its_name(tmp_path):
    analytics = make_analytics(tmp_path)

    def search_metrics(query: str) -> str:
        return query

    def failing(query: str) -> str:
        raise ValueError(query)

    analytics.counted("search_metrics", search_metrics)("price to earnings")
    with pytest.raises(ValueError, match="AAPL"):
        analytics.counted("failing", failing)("AAPL")

    assert analytics.outcomes_per_indicator == {
        "search_metrics": {"data": 1},
        "failing": {"error": 1},
    }
    assert "price to earnings" not in json.dumps(analytics.to_dict())


def test_a_call_over_the_time_limit_is_counted_as_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(analytics_model, "_running_calls", {})
    analytics_model.limit_calls(0.2, 1, lambda: "KEY")
    release = threading.Event()
    try:

        def slow() -> str:
            release.wait(5)
            return "done"

        analytics = make_analytics(tmp_path)
        tool = analytics.counted("slow", analytics_model.run_in_thread(slow))

        async def run_calls():
            return await tool(), await tool()

        timed_out, refused = asyncio.run(run_calls())
    finally:
        release.set()
        analytics_model.limit_calls(None, None, str)

    assert "longer than the 0.2 seconds" in timed_out
    assert "You have 1 calls running" in refused
    assert analytics.outcomes_per_indicator == {"slow": {"error": 1, "invalid": 1}}


def test_outcomes_survive_a_restart_and_older_files_load(tmp_path):
    analytics = make_analytics(tmp_path)
    analytics.record_call(
        "valuation", "KEY", outcomes=[(None, "get_price_to_earnings_ratio", "cut")]
    )
    analytics.save()

    restarted = make_analytics(tmp_path)
    restarted.load()
    assert restarted.outcomes_per_indicator == {
        "valuation:get_price_to_earnings_ratio": {"cut": 1}
    }

    (tmp_path / "mcp_stats.json").write_text(json.dumps({"total_calls": 3}))
    older = make_analytics(tmp_path)
    older.load()
    assert older.total_calls == 3  # noqa: PLR2004
    assert older.outcomes_per_indicator == {}


def test_the_stats_page_shows_outcome_totals_only(tmp_path):
    analytics = make_analytics(tmp_path)
    analytics.record_call(
        "valuation",
        "KEY",
        calls=2,
        outcomes=[
            (None, "get_price_to_earnings_ratio", "data"),
            (None, analytics_model.UNKNOWN_INDICATOR, "invalid"),
        ],
    )
    analytics.record_call("search_metrics", "KEY")
    mcp = FastMCP("test")
    analytics.register_route(mcp)

    stats = TestClient(mcp.streamable_http_app()).get("/stats").json()

    assert stats["outcomes"] == {
        "data": 2,
        "cut": 0,
        "empty": 0,
        "invalid": 1,
        "error": 0,
    }
    assert stats["total_calls"] == 3  # noqa: PLR2004
    assert "outcomes_per_indicator" not in stats
    assert "get_price_to_earnings_ratio" not in json.dumps(stats)


def test_outcomes_are_only_recorded_inside_a_counted_call():
    # A local server without analytics: nothing is collected, and nothing fails.
    analytics_model.record_outcome("data", "valuation", "get_price_to_earnings_ratio")

    assert analytics_model._CALL_OUTCOMES.get() is None
