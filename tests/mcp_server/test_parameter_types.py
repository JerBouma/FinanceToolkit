"""Tests for the types the MCP tool schemas give parameters and the values they accept."""

import asyncio

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.coercion_model import text_from_parsed_json
from financetoolkit.mcp_server.provider_model import ToolkitProvider


@pytest.fixture
def captured(monkeypatch):
    """Record what the tool wrapper passes to the provider instead of fetching data."""
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append(kwargs)
        return pd.DataFrame({"Value": [1.0]})

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


@pytest.fixture(scope="module")
def schemas():
    tools = asyncio.run(mcp_controller.mcp.list_tools())
    return {tool.name: tool.inputSchema["properties"] for tool in tools}


def types_of(schema):
    return {option.get("type") for option in schema.get("anyOf", [schema])} - {"null"}


def call(tool, arguments):
    return asyncio.run(mcp_controller.mcp.call_tool(tool, arguments))[0][0].text


@pytest.mark.parametrize(
    ("tool", "parameter"),
    [
        ("discovery", "beta_lower"),
        ("discovery", "beta_higher"),
        ("discovery", "price_lower"),
        ("discovery", "price_higher"),
        ("discovery", "dividend_lower"),
        ("discovery", "dividend_higher"),
        ("volatility", "num_std_dev"),
        ("volatility", "atr_multiplier"),
        ("options", "time_to_expiration"),
    ],
)
def test_a_fractional_parameter_is_a_number(schemas, tool, parameter):
    assert types_of(schemas[tool][parameter]) == {"number"}


def test_the_screener_takes_a_fractional_beta(captured):
    call(
        "discovery",
        {"indicator": "get_stock_screener", "beta_higher": 0.8, "beta_lower": 1.5},
    )

    assert (captured[-1]["beta_higher"], captured[-1]["beta_lower"]) == (0.8, 1.5)


def test_a_restriction_matrix_is_an_array_of_arrays(schemas):
    matrix = schemas["econometrics"]["restriction_matrix"]

    assert types_of(matrix) == {"array"}
    assert matrix["anyOf"][0]["items"]["type"] == "array"


@pytest.mark.parametrize("matrix", ["[[0, 1, 0], [0, 0, 1]]", [[0, 1, 0], [0, 0, 1]]])
def test_the_wald_test_takes_a_matrix_as_json_or_as_an_array(captured, matrix):
    call(
        "econometrics",
        {
            "indicator": "get_wald_test",
            "tickers": "AAPL,MSFT,AMZN",
            "restriction_matrix": matrix,
            "restriction_values": "[0, 0.5]",
        },
    )

    assert captured[-1]["restriction_matrix"] == [[0, 1, 0], [0, 0, 1]]
    assert captured[-1]["restriction_values"] == [0.0, 0.5]


def test_a_text_parameter_takes_a_json_list(captured):
    """The MCP SDK parses JSON-looking text into a list before validation."""
    call(
        "momentum",
        {
            "indicator": "get_know_sure_thing",
            "tickers": '["AAPL", "MSFT"]',
            "roc_windows": "[10, 15, 20, 30]",
        },
    )

    assert captured[-1]["tickers"] == ["AAPL", "MSFT"]
    assert captured[-1]["roc_windows"] == [10, 15, 20, 30]


def test_parsed_json_becomes_text_again():
    assert text_from_parsed_json(["AAPL", "MSFT"]) == "AAPL,MSFT"
    assert text_from_parsed_json([[0, 1], [1, 0]]) == "[[0, 1], [1, 0]]"
    assert text_from_parsed_json({"AAPL": 0.5}) == '{"AAPL": 0.5}'
    assert text_from_parsed_json("AAPL") == "AAPL"


def test_wls_and_gls_estimate_their_weights_and_covariance(schemas, captured):
    assert schemas["econometrics"]["weights"]["default"] == "feasible"
    assert schemas["econometrics"]["omega"]["default"] == "ar1"

    for indicator in ("get_wls", "get_gls"):
        text = call("econometrics", {"indicator": indicator, "tickers": "AAPL,MSFT"})
        assert "Required" not in text
    assert captured[0]["weights"] == "feasible"
    assert captured[1]["omega"] == "ar1"
