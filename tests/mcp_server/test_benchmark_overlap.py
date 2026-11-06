"""Tests for a requested ticker that is also the benchmark."""

import asyncio
import json

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.provider_model import ToolkitProvider


@pytest.fixture
def toolkit_calls(monkeypatch):
    """Answer like the Toolkit: a ticker that is also the benchmark is left out."""
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append(kwargs["benchmark_ticker"])
        tickers = [
            ticker
            for ticker in kwargs["tickers"]
            if ticker != kwargs["benchmark_ticker"]
        ]
        if method_name == "get_beta":
            return pd.DataFrame({ticker: [1.5] for ticker in tickers}, index=["2026"])
        return pd.DataFrame({"Price": 100.0}, index=pd.Index(tickers, name="ticker"))

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call(tool, arguments):
    return asyncio.run(mcp_controller.mcp.call_tool(tool, arguments))[0][0].text


def test_a_quote_of_the_benchmark_ticker_is_returned(toolkit_calls):
    answer = call(
        "market_data",
        {"indicator": "get_quote", "tickers": "NVDA,SPY", "benchmark_ticker": "SPY"},
    )

    assert "SPY" in answer
    # Asked once with the benchmark, then once without it.
    assert toolkit_calls == ["SPY", None]
    assert "Benchmark" not in answer


def test_a_benchmark_based_metric_keeps_the_benchmark_and_says_so(toolkit_calls):
    answer = json.loads(
        call(
            "performance",
            {"indicator": "get_beta", "tickers": "NVDA,SPY", "benchmark_ticker": "SPY"},
        )
    )

    assert toolkit_calls == ["SPY"]
    assert "SPY is the benchmark, so it is left out as a ticker" in answer["_notes"][0]
