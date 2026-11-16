"""Tests for the portfolio_performance tool."""

import asyncio

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from financetoolkit.mcp_server.registry.registry_controller import MAX_TRANSACTIONS


@pytest.fixture
def captured(monkeypatch):
    """Record what the tool wrapper passes to the provider instead of fetching data."""
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append(
            {
                "module": module_name,
                "method": method_name,
                "category": category,
                **kwargs,
            }
        )
        return pd.DataFrame({"Return": [0.1]}, index=["Portfolio"])

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call(indicator="get_portfolio_overview", **arguments):
    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "portfolio_performance", {"indicator": indicator, **arguments}
        )
    )
    return str(result)


def test_both_portfolio_tools_are_registered():
    tools = {tool.name for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    assert {"portfolio", "portfolio_performance"} <= tools


def test_transactions_reach_the_provider_as_validated_trades(captured):
    call(
        transactions=[
            {"Date": "2023-01-03", "ticker": "aapl", "Volume": 10, "price": 125.1},
            {"date": "2024/06/03", "symbol": "AAPL", "shares": -5, "Price": "194"},
        ],
        currency="usd",
    )

    portfolio = captured[-1]["portfolio"]
    assert captured[-1]["category"] == "portfolio"
    assert captured[-1]["module"] == "portfolio_performance"
    assert portfolio["currency"] == "USD"
    assert portfolio["transactions"] == [
        {
            "Date": "2023-01-03",
            "Ticker": "AAPL",
            "Volume": 10.0,
            "Costs": 0.0,
            "Price": 125.1,
        },
        {
            "Date": "2024-06-03",
            "Ticker": "AAPL",
            "Volume": -5.0,
            "Costs": 0.0,
            "Price": 194.0,
        },
    ]
    # Transactions start on their first date, so no default start date is sent.
    assert captured[-1]["start_date"] == ""


def test_transactions_sent_as_json_text_are_read(captured):
    call(transactions='[{"date": "2023-01-03", "ticker": "MSFT", "volume": 2}]')

    assert captured[-1]["portfolio"]["transactions"][0]["Ticker"] == "MSFT"


def test_weights_and_holdings_are_read_from_text_and_objects(captured):
    call(
        weights="VTI:0.6, BND:0.4",
        start_date="2020-01-02",
        rebalancing="Yearly",
        initial_value=5000,
    )
    assert captured[-1]["portfolio"] == {
        "weights": {"VTI": 0.6, "BND": 0.4},
        "rebalancing": "yearly",
        "initial_value": 5000.0,
    }
    assert captured[-1]["start_date"] == "2020-01-02"

    call(holdings={"aapl": 10, "MSFT": "5"})
    assert captured[-1]["portfolio"] == {"holdings": {"AAPL": 10.0, "MSFT": 5.0}}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "exactly one of transactions, holdings, weights or preset"),
        ({"weights": {"VTI": 1}, "preset": "60/40"}, "received preset, weights"),
        ({"transactions": [{"date": "2023-01-03", "ticker": "AAPL"}]}, "has no volume"),
        (
            {"transactions": [{"date": "03-01-2023x", "ticker": "AAPL", "volume": 1}]},
            "invalid date",
        ),
        (
            {
                "transactions": [
                    {"date": "2023-01-03", "ticker": "AAPL", "volume": 1, "note": "x"}
                ]
            },
            "unknown field",
        ),
        (
            {"transactions": [{"date": "2023-01-03", "ticker": "../etc", "volume": 1}]},
            "invalid ticker",
        ),
        (
            {
                "transactions": [
                    {"date": "2023-01-03", "ticker": "AAPL", "volume": "nan"}
                ]
            },
            "must be a number",
        ),
        (
            {
                "transactions": [
                    {"date": "2023-01-03", "ticker": "AAPL", "volume": 1, "price": 100},
                    {"date": "2023-01-04", "ticker": "AAPL", "volume": 1},
                ]
            },
            "Price is given for 1 of the 2 transactions",
        ),
        ({"transactions": "my_trades.csv"}, "must be a list of trades"),
        ({"weights": "VTI 0.6"}, "must be given per ticker"),
        ({"weights": {"VTI": 1}, "rebalancing": "daily"}, "rebalancing must be one of"),
        ({"weights": {"VTI": 1}, "currency": "dollar"}, "three-letter code"),
        ({"weights": {"VTI": 1}, "initial_value": -1}, "more than 0"),
    ],
)
def test_invalid_portfolios_are_explained_without_a_call(captured, arguments, message):
    answer = call(**arguments)

    assert message in answer
    assert not captured


def test_too_many_transactions_or_tickers_are_refused(captured):
    trade = {"date": "2023-01-03", "ticker": "AAPL", "volume": 1}
    assert f"at most {MAX_TRANSACTIONS}" in call(
        transactions=[trade] * (MAX_TRANSACTIONS + 1)
    )

    weights = {f"T{number}": 1 for number in range(51)}
    assert "holds 51 tickers" in call(weights=weights)
    assert not captured


def test_the_presets_need_no_portfolio(captured):
    call(indicator="get_presets")

    assert captured[-1]["method"] == "get_presets"
    assert captured[-1]["portfolio"] == {}


def test_the_provider_builds_the_portfolio_from_its_inputs(monkeypatch, tmp_path):
    built = []

    class FakePortfolio:
        def __init__(self, **kwargs):
            built.append(kwargs)

        @classmethod
        def from_preset(cls, name, **kwargs):
            return cls(preset=name, **kwargs)

        def get_portfolio_overview(self, **kwargs):
            return pd.DataFrame({"Return": [0.1]}, index=["Portfolio"])

    from financetoolkit.mcp_server.registry import provider_model  # noqa: PLC0415

    monkeypatch.setattr(provider_model, "Portfolio", FakePortfolio)
    provider = ToolkitProvider(
        cache_ttl=0, database_location=str(tmp_path), api_key="key", cache_enabled=False
    )

    trades = [{"Date": "2023-01-03", "Ticker": "AAPL", "Volume": 10.0}]
    provider.call_portfolio_functionality(
        "get_portfolio_overview",
        {"transactions": trades},
        start_date="",
        benchmark_ticker="SPY",
    )
    # Transactions become a DataFrame: a text value would be read as a file path.
    assert isinstance(built[-1]["transactions"], pd.DataFrame)
    assert built[-1]["benchmark_ticker"] == "SPY"
    assert "start_date" not in built[-1]

    provider.call_portfolio_functionality(
        "get_portfolio_overview",
        {"preset": "60/40", "initial_value": 500.0},
        "2020-01-02",
        "SPY",
    )
    assert built[-1]["preset"] == "60/40"
    assert built[-1]["start_date"] == "2020-01-02"
    assert built[-1]["initial_value"] == 500.0  # noqa: PLR2004

    # The same portfolio is answered from the instance already built.
    count = len(built)
    provider.call_portfolio_functionality(
        "get_portfolio_overview",
        {"preset": "60/40", "initial_value": 500.0},
        "2020-01-02",
        "SPY",
    )
    assert len(built) == count


def test_the_provider_needs_a_key_for_a_portfolio(tmp_path):
    provider = ToolkitProvider(
        cache_ttl=0, database_location=str(tmp_path), api_key="", cache_enabled=False
    )

    with pytest.raises(ValueError):
        provider.get_portfolio_instance({"holdings": {"AAPL": 1.0}}, "", "SPY")
