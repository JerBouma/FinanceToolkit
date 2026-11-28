"""Tests for indicator parameters that reach the indicator over MCP as they do in Python."""

import asyncio
import json

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from financetoolkit.utilities.logger_model import get_logger
from tests.mcp_server.answers import read_answer


@pytest.fixture
def captured(monkeypatch):
    """Record what the tool wrapper passes to the provider instead of fetching data."""
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append(kwargs)
        return pd.DataFrame({"Value": [1.0]}, index=["AAPL"])

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call(tool, arguments):
    return asyncio.run(mcp_controller.mcp.call_tool(tool, arguments))[0].text


def test_entropy_pooling_takes_inequality_views(captured):
    call(
        "portfolio",
        {
            "indicator": "get_entropy_pooling",
            "tickers": "XOM,JNJ",
            "views": {"XOM-JNJ": [">=", 0.02], "XOM": 0.08},
            "volatility_views": {"XOM": ["<=", 0.3]},
        },
    )

    # As in Python: a number for an equality, a pair (operator, value) for an inequality.
    assert captured[-1]["views"] == {"XOM-JNJ": (">=", 0.02), "XOM": 0.08}
    assert captured[-1]["volatility_views"] == {"XOM": ("<=", 0.3)}


def test_a_malformed_view_is_refused():
    with pytest.raises(Exception, match="views"):
        call(
            "portfolio",
            {
                "indicator": "get_entropy_pooling",
                "tickers": "XOM,JNJ",
                "views": {"XOM": [">=", 0.02, 0.03]},
            },
        )


def test_treasury_maturities_reach_the_forward_prices(captured):
    answer = call(
        "fixed_income",
        {
            "indicator": "get_treasury_forward_prices",
            "maturity_start": "2030-01-01",
            "maturity_end": "2035-12-31",
        },
    )

    assert captured[-1]["maturity_start"] == "2030-01-01"
    assert captured[-1]["maturity_end"] == "2035-12-31"
    assert "maturity_start" not in answer


def test_the_sale_date_of_a_delta_hedging_backtest_is_its_own(captured):
    call(
        "options",
        {
            "indicator": "get_delta_hedging_backtest",
            "tickers": "AAPL",
            "start_date": "2024-01-01",
            "sale_date": "2025-06-02",
        },
    )

    # The Toolkit's range and the date the option is sold are two parameters.
    assert captured[-1]["start_date"] == "2024-01-01"
    assert captured[-1]["sale_date"] == "2025-06-02"


def test_show_columns_selects_the_archer_on_balance_volume_columns(
    monkeypatch, technical_module
):
    monkeypatch.setattr(
        ToolkitProvider,
        "call_method",
        lambda self, module_name, method_name, category, **kwargs: getattr(  # noqa: ARG005
            technical_module, method_name
        )(),
    )

    answer = read_answer(
        call(
            "breadth",
            {
                "indicator": "get_archer_on_balance_volume",
                "tickers": "AAPL,MSFT",
                "show_columns": "Fast Average,Slow Average",
                "limit": -1,
            },
        )
    )
    rows = answer["data"] if isinstance(answer, dict) else answer

    assert {key for row in rows for key in row} == {
        "Date",
        "ticker",
        "Fast Average",
        "Slow Average",
    }


@pytest.mark.parametrize("tickers", ["KO", "KO,PEP"])
def test_no_bubble_episode_is_an_answer_with_its_columns(monkeypatch, tickers):
    def no_episodes(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        for ticker in kwargs["tickers"]:
            get_logger().warning(
                "No bubble episode is found for %s from 2021-01-01 to 2025-12-31: its BSADF "
                "statistic is not above the critical value (95%% confidence) for 4 "
                "observations in a row.",
                ticker,
            )
        return pd.DataFrame(
            columns=["Start", "End", "Duration", "Peak Date", "Peak Statistic"],
            index=pd.MultiIndex.from_tuples([], names=["Ticker", "Episode"]),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", no_episodes)

    answer = json.loads(
        call("econometrics", {"indicator": "get_bubble_episodes", "tickers": tickers})
    )

    assert "error" not in answer
    assert answer["columns"] == [
        "Ticker",
        "Episode",
        "Start",
        "End",
        "Duration",
        "Peak Date",
        "Peak Statistic",
    ]
    assert answer["rows"] == []
    assert any(
        note.startswith("No bubble episode is found for KO")
        for note in answer["_notes"]
    )
    assert not any("check the symbol" in note for note in answer["_notes"])


def test_an_empty_table_without_an_explaining_note_is_still_no_data(monkeypatch):
    monkeypatch.setattr(
        ToolkitProvider,
        "call_method",
        lambda self, module_name, method_name, category, **kwargs: pd.DataFrame(  # noqa: ARG005
            columns=["Start", "End"]
        ),
    )

    answer = json.loads(
        call("econometrics", {"indicator": "get_bubble_episodes", "tickers": "KO"})
    )

    assert answer["error"] == "No data available."
