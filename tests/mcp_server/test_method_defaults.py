"""Tests for the MCP-only method defaults (config.yaml method_defaults)."""

import asyncio
from datetime import datetime, timedelta

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
        calls.append(kwargs)
        return pd.DataFrame({"Event": ["CPI"]})

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call_calendar(**arguments):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "macroeconomics", {"indicator": "get_economic_calendar", **arguments}
        )
    )


def test_economic_calendar_defaults_to_high_impact_around_today(captured):
    call_calendar(countries="United States")

    today = datetime.now()
    assert captured[-1]["impact"] == "High"
    assert captured[-1]["start_date"] == (today - timedelta(days=23)).strftime(
        "%Y-%m-%d"
    )
    assert captured[-1]["end_date"] == (today + timedelta(days=7)).strftime("%Y-%m-%d")


def test_economic_calendar_explicit_values_win(captured):
    call_calendar(impact="All", start_date="2026-09-01", end_date="2026-09-10")

    assert captured[-1]["impact"] == "All"
    assert (captured[-1]["start_date"], captured[-1]["end_date"]) == (
        "2026-09-01",
        "2026-09-10",
    )


def call_performance(**arguments):
    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance",
            {"indicator": "get_sharpe_ratio", "tickers": "AAPL", **arguments},
        )
    )

    return str(result)


def test_daily_performance_needs_a_rolling_window(captured):
    """Within-period results need a longer period; a rolling window reads daily data."""
    assert "does not support" in call_performance(period="daily")
    assert not captured

    call_performance(period="daily", rolling=20)

    assert captured[-1]["period"] == "daily"
    assert captured[-1]["rolling"] == 20  # noqa: PLR2004


def test_large_outputs_default_to_a_compact_selection(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "rates",
            {"indicator": "get_eiopa_risk_free_rate", "countries": "Netherlands"},
        )
    )
    assert captured[-1]["maturity_years"] == [1, 2, 5, 10, 20, 30]

    asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data", {"indicator": "get_etf_holdings", "tickers": "IWDA.AS"}
        )
    )
    assert captured[-1]["limit"] == 25  # noqa: PLR2004


def test_the_server_cuts_an_answer_that_is_too_large(monkeypatch):
    import numpy as np

    def huge(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"Close": np.arange(200_000.0)},
            index=pd.period_range("1500-01-01", periods=200_000, freq="D"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", huge)

    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance", {"indicator": "get_sharpe_ratio", "tickers": "AAPL"}
        )
    )
    text = result[0][0].text

    assert len(text) < 62_000  # noqa: PLR2004
    assert "was too large and was cut" in text


def test_limit_selects_the_first_or_last_rows(monkeypatch):
    import json

    def prices(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"Close": [1.0, 2.0, 3.0, 4.0]},
            index=pd.period_range("2026-01-01", periods=4, freq="D"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", prices)

    def closes(limit):
        result = asyncio.run(
            mcp_controller.mcp.call_tool(
                "performance",
                {"indicator": "get_sharpe_ratio", "tickers": "AAPL", "limit": limit},
            )
        )
        return [row["Close"] for row in json.loads(result[0][0].text)]

    assert closes(2) == [1.0, 2.0]
    assert closes(-2) == [3.0, 4.0]


def test_an_indicator_with_its_own_limit_receives_it(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data",
            {"indicator": "get_etf_holdings", "tickers": "IWDA.AS", "limit": -10},
        )
    )

    assert captured[-1]["limit"] == -10  # noqa: PLR2004


def test_a_method_default_replaces_the_indicators_own_default(captured):
    def simulations(**arguments):
        asyncio.run(
            mcp_controller.mcp.call_tool(
                "risk",
                {"indicator": "get_copula_simulation", "tickers": "AAPL", **arguments},
            )
        )
        return captured[-1]["n_simulations"]

    # The tool schema must not carry the indicator's 10,000, or the MCP default of 100
    # could never apply; an explicit value still wins.
    assert simulations() == 100  # noqa: PLR2004
    assert simulations(n_simulations=5000) == 5000  # noqa: PLR2004


def test_a_transcript_is_one_full_call_of_one_ticker(monkeypatch):
    calls = []

    def transcript(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        calls.append(kwargs)
        return pd.DataFrame(
            {"Date": ["2025-10-30"], "Transcript": ["word " * 30_000]},
            index=pd.Index(["2025Q4"], name="Fiscal Period"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", transcript)

    def call(**arguments):
        result = asyncio.run(
            mcp_controller.mcp.call_tool(
                "market_data",
                {"indicator": "get_earnings_call_transcripts", **arguments},
            )
        )
        return result[0][0].text

    # 150,000 characters, well over the response size, and still returned in full.
    text = call(tickers="AAPL", fiscal_period="2025Q4")

    assert len(text) > 150_000  # noqa: PLR2004
    assert "was cut" not in text
    assert calls[-1]["fiscal_period"] == "2025Q4"

    assert "one ticker at a time" in call(tickers="AAPL,MSFT")
    assert "fiscal_period" in call(tickers="AAPL", latest=False)
    assert "e.g. '2025Q4'" in call(tickers="AAPL", fiscal_period="last quarter")
    assert len(calls) == 1


def test_missing_required_inputs_are_named(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "models", {"indicator": "get_intrinsic_valuation", "tickers": "MSFT"}
        )
    )[0][0].text

    assert "requires `growth_rate`" in text
    assert captured == []


def test_an_indicator_of_another_tool_is_pointed_there(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "overlap", {"indicator": "get_bollinger_bands", "tickers": "AAPL"}
        )
    )[0][0].text

    assert "indicator of `volatility`" in text
    assert captured == []


def test_the_benchmark_note_is_left_out_without_a_benchmark(monkeypatch):
    def holdings(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame({"Asset": ["NVDA"], "Weight": [0.08]})

    monkeypatch.setattr(ToolkitProvider, "call_method", holdings)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data", {"indicator": "get_etf_holdings", "tickers": "SPY"}
        )
    )[0][0].text

    assert "benchmark" not in text.lower()
