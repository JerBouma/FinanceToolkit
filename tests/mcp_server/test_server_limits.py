"""Tests for the limits that keep one MCP call to a size a shared server can hold."""

import asyncio
import json
import threading
import time
import types

import numpy as np
import pandas as pd
import pytest
import requests

from financetoolkit.mcp_server import analytics_model, mcp_controller
from financetoolkit.mcp_server.coercion_model import size_limit_problem
from financetoolkit.mcp_server.formatting_model import format_result
from financetoolkit.mcp_server.provider_model import ToolkitProvider
from financetoolkit.utilities import requests_model


@pytest.fixture
def captured(monkeypatch):
    """Record what the tool wrapper passes to the provider instead of fetching data."""
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append((method_name, kwargs))
        return pd.DataFrame({"Value": [1.0]})

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call(tool, **arguments):
    return str(asyncio.run(mcp_controller.mcp.call_tool(tool, arguments)))


def test_a_simulation_larger_than_the_server_allows_is_refused(captured):
    answer = call(
        "risk",
        indicator="get_copula_simulation",
        tickers="AAPL,MSFT",
        n_simulations=3_000_000,
    )

    assert "`n_simulations` may be at most 10,000" in answer
    assert not captured

    call(
        "risk",
        indicator="get_copula_simulation",
        tickers="AAPL,MSFT",
        n_simulations=500,
    )
    assert captured[-1][1]["n_simulations"] == 500  # noqa: PLR2004


def test_parameters_computed_together_are_limited_with_the_indicator_defaults(captured):
    # 100,000 paths of the indicator's default 100 steps is over the product limit,
    # although each value on its own is within its maximum.
    answer = call(
        "options",
        indicator="get_monte_carlo_option_price",
        tickers="AAPL",
        simulations=100_000,
    )

    assert "`simulations` times `time_steps` may be at most 5,000,000" in answer
    assert not captured


def test_the_backtest_lookback_is_limited(captured):
    assert "`lookback` may be at most 1,260" in call(
        "portfolio",
        indicator="get_optimization_backtest",
        tickers="AAPL,MSFT",
        lookback=2520,
    )
    assert not captured


def test_the_short_volume_history_is_limited_and_never_unbounded(captured):
    assert "`history_days` may be at most 366" in call(
        "market_data", indicator="get_short_volume", tickers="AAPL", history_days=6000
    )
    assert not captured

    # Left out (or null) it is the MCP default, rather than the whole date range.
    call("market_data", indicator="get_short_volume", tickers="AAPL", history_days=None)
    assert captured[-1][1]["history_days"] == 92  # noqa: PLR2004


def test_size_limits_check_every_value_of_a_list():
    limits = {"maximums": {"lags": 100}, "products": []}

    assert size_limit_problem({"lags": [1, 5, 500]}, {}, limits)
    assert size_limit_problem({"lags": [1, 5, 50]}, {}, limits) is None
    # A product leaves out an indicator that does not take every parameter of it.
    product = {"maximums": {}, "products": [{"parameters": ["a", "b"], "maximum": 10}]}
    assert size_limit_problem({"a": 100}, {}, product) is None
    assert size_limit_problem({"a": 100}, {"b": 1}, product)


def test_tickers_and_benchmarks_are_validated_before_any_request(captured):
    for tickers in ("AAPL&apikey=X", "../v7/finance", "A" * 30):
        assert "invalid ticker" in call(
            "valuation", indicator="get_price_to_earnings_ratio", tickers=tickers
        )
    assert "Invalid `benchmark_ticker`" in call(
        "performance",
        indicator="get_sharpe_ratio",
        tickers="AAPL",
        benchmark_ticker="SPY?x=1",
    )
    assert not captured

    call(
        "performance",
        indicator="get_sharpe_ratio",
        tickers="ASML.AS,BTC-USD,EURUSD=X",
        benchmark_ticker="^GSPC",
    )
    assert captured


def test_series_references_are_held_to_the_entity_limit(captured):
    tickers = ",".join(f"T{number}" for number in range(30))
    answer = call(
        "econometrics",
        indicator="get_ols",
        series=(
            f"valuation:get_price_to_earnings_ratio:{tickers};"
            f"valuation:get_price_to_book_ratio:{tickers}"
        ),
    )

    assert "60 tickers and countries together; at most 50" in answer
    assert "invalid ticker" in call(
        "econometrics",
        indicator="get_ols",
        series="valuation:get_price_to_earnings_ratio:AAPL&x=1",
    )
    assert not captured


def test_a_large_table_is_cut_before_its_records_are_built(monkeypatch):
    """The records of a table of millions of rows would not fit in memory."""
    from financetoolkit.mcp_server import formatting_model

    built = []
    to_records = formatting_model._df_to_records
    monkeypatch.setattr(
        formatting_model,
        "_df_to_records",
        lambda frame: built.append(len(frame)) or to_records(frame),
    )
    table = pd.DataFrame(
        np.random.default_rng(0).normal(size=(1_000_000, 2)), columns=["A", "B"]
    )

    answer = json.loads(format_result(table, max_characters=60_000))

    assert max(built) <= 60_000 // 4  # noqa: PLR2004
    assert "first" in answer["_notes"][-1] and "1,000,000 rows" in answer["_notes"][-1]
    assert len(json.dumps(answer["data"])) <= 60_000  # noqa: PLR2004


@pytest.fixture
def hosted_limits(monkeypatch):
    """A hosted server's limits: a short time limit and two calls per user."""
    user = {"key": "first"}
    monkeypatch.setattr(analytics_model, "_running_calls", {})
    analytics_model.limit_calls(0.5, 2, lambda: user["key"])
    yield user
    analytics_model.limit_calls(None, None, str)


def test_a_call_over_the_time_limit_is_answered_and_the_user_limit_holds(
    hosted_limits,
):
    release = threading.Event()

    def slow() -> str:
        release.wait(5)
        return "done"

    tool = analytics_model.run_in_thread(slow)

    async def run_calls():
        first = await asyncio.gather(tool(), tool())
        # Both timed out, but their work still runs, so the user has no call left...
        refused = await tool()
        # ...while another user does.
        hosted_limits["key"] = "second"
        other = await tool()
        return first, refused, other

    first, refused, other = asyncio.run(run_calls())
    release.set()

    assert all("longer than the 0.5 seconds" in answer for answer in first)
    assert "You have 2 calls running" in refused
    assert "longer than the 0.5 seconds" in other

    # Once the work ends, the user's calls are counted as finished.
    deadline = time.time() + 5
    while analytics_model._running_calls and time.time() < deadline:
        time.sleep(0.05)
    assert analytics_model._running_calls == {}


def test_a_hosted_server_never_retries_without_verifying_the_certificate(monkeypatch):
    verified = []

    def fake_get(url, headers, timeout, verify):  # noqa: ARG001
        verified.append(verify)
        raise requests.exceptions.SSLError("certificate verify failed")

    no_system_certificates = types.SimpleNamespace(cafile=None)
    monkeypatch.setattr(requests_model.SESSION, "get", fake_get)
    monkeypatch.setattr(
        requests_model.ssl,
        "get_default_verify_paths",
        lambda: no_system_certificates,
    )
    monkeypatch.setenv(requests_model.UNVERIFIED_SSL_ENVIRONMENT_VARIABLE, "0")

    with pytest.raises(requests.exceptions.SSLError):
        requests_model.get_request("https://example.org/data.csv")

    assert False not in verified
