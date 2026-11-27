"""Tests for rebase_to, which indexes the time series of an MCP answer to a base window."""

import asyncio
import json

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.registry.formatting_model import (
    parse_base_window,
    rebase_result,
)
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from tests.mcp_server.answers import read_answer

YEARLY = pd.DataFrame(
    {
        "Real consumption wages": [50.0, 80.0, 120.0, 100.0, 200.0],
        "Consumer price index": [10.0, 20.0, 30.0, 40.0, 50.0],
    },
    index=pd.period_range("1339", periods=5, freq="Y"),
)


@pytest.fixture
def answer(monkeypatch):
    """Answer every tool call with the table set on it, instead of fetching data."""
    table = {"result": YEARLY}

    def call_method(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return table["result"]

    monkeypatch.setattr(ToolkitProvider, "call_method", call_method)
    return table


def call(**arguments):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "macroeconomics",
            {"indicator": "get_millennium_of_macroeconomic_data", **arguments},
        )
    )[0].text
    try:
        return read_answer(text)
    except json.JSONDecodeError:
        return text


def values(answer_json, column):
    return {row["date"]: row[column] for row in answer_json["data"]}


@pytest.mark.parametrize(
    ("text", "first", "last"),
    [
        ("1340-1347", "1340-01-01", "1347-12-31"),
        ("2020-03", "2020-03-01", "2020-03-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2020-03-31", "2020-03-31", "2020-03-31"),
        ("2020-01:2020-06-15", "2020-01-01", "2020-06-15"),
    ],
)
def test_a_base_window_reads_a_year_range_apart_from_a_month(text, first, last):
    start, end = parse_base_window(text)

    assert (str(start), str(end)) == (first, last)


@pytest.mark.parametrize("text", ["2020-13", "2020-1", "20-01", "1347-1340", "x"])
def test_an_unreadable_base_window_is_refused_before_fetching(answer, text):
    answer["result"] = None  # a fetch would fail on it

    assert call(rebase_to=text).startswith("Invalid `rebase_to`")


def test_a_yearly_series_is_indexed_to_a_year_range(answer):
    result = call(rebase_to="1340-1341")

    # 100 = the average of 1340 and 1341: (80 + 120) / 2 and (20 + 30) / 2
    assert values(result, "Real consumption wages") == {
        "1339": 50.0,
        "1340": 80.0,
        "1341": 120.0,
        "1342": 100.0,
        "1343": 200.0,
    }
    assert values(result, "Consumer price index")["1343"] == 200.0
    assert (
        "its average over 1340-01-01 to 1341-12-31 (2 periods)" in result["_notes"][0]
    )


def test_a_yearly_series_is_indexed_to_a_single_year(answer):
    result = call(rebase_to="1342")

    assert values(result, "Real consumption wages")["1343"] == 200.0
    assert values(result, "Consumer price index")["1339"] == 25.0


def test_a_daily_series_is_indexed_to_a_month(answer):
    answer["result"] = pd.DataFrame(
        {"Price": [10.0, 30.0, 40.0, 60.0]},
        index=pd.to_datetime(["2020-02-28", "2020-03-02", "2020-03-31", "2020-04-01"]),
    )

    result = call(rebase_to="2020-03")

    assert values(result, "Price") == {
        "2020-02-28": 28.5714,
        "2020-03-02": 85.7143,
        "2020-03-31": 114.2857,
        "2020-04-01": 171.4286,
    }


def test_a_window_outside_the_answer_says_how_to_widen_the_dates(answer):
    before = call(rebase_to="1300-1310")
    after = call(rebase_to="1400")

    assert "matches no row of the answer" in before
    assert "set start_date on or before 1300-01-01" in before
    assert "set end_date on or after 1400-01-01" in after


def test_a_column_without_a_base_is_empty_with_a_note(answer):
    answer["result"] = YEARLY.assign(
        Empty=[1.0, None, None, 4.0, 5.0], Zero=[1.0, 0.0, 0.0, 4.0, 5.0]
    )

    result = call(rebase_to="1340-1341")

    assert set(values(result, "Real consumption wages").values()) != {None}
    assert all(
        row.get("Empty") is None and row.get("Zero") is None for row in result["data"]
    )
    assert any(
        note.startswith("Empty has no value in the base window")
        for note in result["_notes"]
    )
    assert any(
        note.startswith("Zero has an average over the base window")
        for note in result["_notes"]
    )


def test_a_column_with_a_negative_base_is_empty_with_a_note():
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    table = pd.DataFrame(
        {"Negative": [-1.0, 1.0, 2.0], "Positive": [2.0, 1.0, 4.0]}, index=days
    )

    rebased, notes = rebase_result(table, parse_base_window("2024-01-01"), "2024-01-01")

    # Indexed to -1, the column would turn every sign around (1 becoming -100).
    assert rebased["Negative"].isna().all()
    assert rebased["Positive"].tolist() == [100.0, 50.0, 200.0]
    assert any(
        note.startswith("Negative has an average over the base window")
        and "not positive" in note
        for note in notes
    )


def test_the_base_is_taken_before_limit_and_show_columns(answer):
    result = call(
        rebase_to="1340-1341", limit=-1, show_columns="Real consumption wages"
    )

    # The base window lies outside the last row, which is still indexed to it.
    assert result["data"] == [{"date": "1343", "Real consumption wages": 200.0}]


def test_growth_rates_are_not_rebased(answer):
    result = call(rebase_to="1340", growth=True)

    assert values(result, "Consumer price index")["1343"] == 50.0
    assert any("rebase_to='1340' was not applied" in note for note in result["_notes"])


def test_a_ratio_with_periods_as_columns_is_indexed_per_row(monkeypatch):
    ratios = pd.DataFrame(
        [[10.0, 20.0, 30.0], [4.0, 2.0, 1.0]],
        index=["AAPL", "MSFT"],
        columns=pd.period_range("2022", periods=3, freq="Y"),
    )
    monkeypatch.setattr(
        ToolkitProvider, "call_method", lambda self, *args, **kwargs: ratios
    )
    monkeypatch.setattr(
        ToolkitProvider, "get_transformation_notes", lambda self, **kwargs: []
    )

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "valuation",
            {
                "indicator": "get_price_to_earnings_ratio",
                "tickers": "AAPL,MSFT",
                "rebase_to": "2022",
            },
        )
    )[0].text

    assert "300.0" in text
    assert "25.0" in text
    assert "Every row is indexed to 100" in text
