"""Tests for sample, which keeps the last row of each calendar period of a daily MCP answer."""

import asyncio
import json

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.registry.formatting_model import sample_rows
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from tests.mcp_server.answers import read_answer

# Trading days from Monday 1 January to Wednesday 13 March 2024: ten whole weeks and a
# partial one, two whole months and a partial one.
DAYS = pd.bdate_range("2024-01-01", "2024-03-13").to_period("D")
DAILY = pd.DataFrame(
    {"AAPL": range(len(DAYS)), "MSFT": range(100, 100 + len(DAYS))},
    index=DAYS,
    dtype=float,
)


@pytest.fixture
def answer(monkeypatch):
    """Answer every tool call with the table set on it, instead of fetching data."""
    table = {"result": DAILY}

    def call_method(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return table["result"]

    monkeypatch.setattr(ToolkitProvider, "call_method", call_method)
    monkeypatch.setattr(
        ToolkitProvider, "get_transformation_notes", lambda self, **kwargs: []
    )
    return table


def call(tool="overlap", indicator="get_moving_average", **arguments):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            tool, {"indicator": indicator, "tickers": "AAPL,MSFT", **arguments}
        )
    )[0].text
    try:
        return read_answer(text)
    except json.JSONDecodeError:
        return text


def dates(answer_json):
    return [row["date"] for row in answer_json["data"]]


def test_every_tool_takes_sample():
    tools = asyncio.run(mcp_controller.mcp.list_tools())
    routers = [tool for tool in tools if "indicator" in tool.inputSchema["properties"]]

    assert routers
    assert all("sample" in tool.inputSchema["properties"] for tool in routers)


def test_weekly_keeps_the_last_day_of_each_week_and_of_the_partial_last_one(answer):
    result = call(sample="weekly")

    assert dates(result) == [
        "2024-01-05",
        "2024-01-12",
        "2024-01-19",
        "2024-01-26",
        "2024-02-02",
        "2024-02-09",
        "2024-02-16",
        "2024-02-23",
        "2024-03-01",
        "2024-03-08",
        "2024-03-13",
    ]
    # The values are those of the days themselves, not of a resampled series.
    assert result["data"][0] == {"date": "2024-01-05", "AAPL": 4.0, "MSFT": 104.0}
    assert "Sampled to the last row of each week (11 of 53 rows)." in result["_notes"]


def test_monthly_keeps_the_last_day_of_each_month(answer):
    result = call(sample="Monthly")

    assert dates(result) == ["2024-01-31", "2024-02-29", "2024-03-13"]
    assert "Sampled to the last row of each month (3 of 53 rows)." in result["_notes"]


def test_a_day_without_any_value_is_not_the_last_reading_of_its_period(answer):
    answer["result"] = DAILY.copy()
    answer["result"].loc[pd.Period("2024-01-31", "D")] = None

    assert dates(call(sample="monthly"))[0] == "2024-01-30"


def test_intraday_rows_keep_their_last_time_of_each_week():
    hours = pd.date_range("2024-01-04 09:00", "2024-01-09 16:00", freq="h")
    sampled, note = sample_rows(pd.Series(1.0, index=hours), "weekly")

    assert list(sampled.index) == [
        pd.Timestamp("2024-01-07 23:00"),
        pd.Timestamp("2024-01-09 16:00"),
    ]
    assert note.startswith("Sampled to the last row of each week (2 of ")


def test_sample_runs_before_limit(answer):
    result = call(sample="weekly", limit=-3)

    assert dates(result) == ["2024-03-01", "2024-03-08", "2024-03-13"]


def test_dates_as_columns_are_sampled_by_column(monkeypatch):
    sideways = DAILY.T
    monkeypatch.setattr(
        ToolkitProvider, "call_method", lambda self, *args, **kwargs: sideways
    )
    monkeypatch.setattr(
        ToolkitProvider, "get_transformation_notes", lambda self, **kwargs: []
    )

    sampled, note = sample_rows(sideways, "monthly")

    assert [str(column) for column in sampled.columns] == [
        "2024-01-31",
        "2024-02-29",
        "2024-03-13",
    ]
    assert list(sampled.index) == ["AAPL", "MSFT"]
    assert note == "Sampled to the last column of each month (3 of 53 columns)."

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "valuation",
            {
                "indicator": "get_price_to_earnings_ratio",
                "tickers": "AAPL,MSFT",
                "sample": "monthly",
            },
        )
    )[0].text
    assert "2024-02-29" in text
    assert "2024-02-28" not in text
    assert "Sampled to the last column of each month" in text


def test_rows_of_a_date_and_an_item_keep_every_row_of_the_chosen_dates():
    rows = pd.MultiIndex.from_product([DAYS[:10], ["AAPL", "MSFT"]])
    sampled, note = sample_rows(pd.Series(1.0, index=rows), "weekly")

    assert [str(day) for day in sampled.index.get_level_values(0).unique()] == [
        "2024-01-05",
        "2024-01-12",
    ]
    assert len(sampled) == 4
    assert note == "Sampled to the last row of each week (2 of 10 rows)."


@pytest.mark.parametrize(
    ("index", "sample", "reason"),
    [
        (pd.period_range("2023-01", periods=12, freq="M"), "weekly", "monthly already"),
        (
            pd.period_range("2023-01", periods=12, freq="M"),
            "monthly",
            "monthly already",
        ),
        (
            pd.period_range("2023-01-06", periods=8, freq="W-FRI"),
            "weekly",
            "weekly already",
        ),
        (
            pd.period_range("2023Q1", periods=8, freq="Q"),
            "monthly",
            "quarterly already",
        ),
    ],
)
def test_an_answer_at_the_sample_or_coarser_is_unchanged_with_a_note(
    answer, index, sample, reason
):
    answer["result"] = pd.DataFrame(
        {"AAPL": range(len(index)), "MSFT": range(len(index))}, index=index, dtype=float
    )

    result = call(sample=sample, limit=0)

    assert len(result["data"]) == len(index)
    assert any(
        note.startswith(f"sample={sample!r} was not applied: the answer is {reason}")
        for note in result["_notes"]
    )


def test_an_answer_without_dates_is_unchanged_with_a_note(answer):
    answer["result"] = pd.DataFrame(
        {"Sharpe Ratio": [1.2, 0.8]}, index=["AAPL", "MSFT"]
    )

    result = call(tool="performance", indicator="get_sharpe_ratio", sample="weekly")

    assert len(result["data"]) == 2  # noqa: PLR2004
    assert (
        "sample='weekly' was not applied: the answer has no dates as its rows or columns."
        in result["_notes"]
    )


@pytest.mark.parametrize("value", ["daily", "week", "1W"])
def test_an_invalid_sample_names_the_valid_values(answer, value):
    answer["result"] = None  # a fetch would fail on it

    text = call(sample=value)

    assert text.startswith("Invalid input for `overlap` (`get_moving_average`)")
    assert "weekly, monthly, quarterly, yearly" in text


def test_weeks_are_sampled_to_the_month_their_friday_falls_in(answer):
    weeks = pd.period_range("2023-01-06", periods=8, freq="W-FRI")
    answer["result"] = pd.DataFrame(
        {"AAPL": range(len(weeks)), "MSFT": range(len(weeks))}, index=weeks, dtype=float
    )

    result = call(sample="monthly", limit=0)

    # The week to Friday 3 February is February's first week, though it starts in January.
    assert dates(result) == ["2023-01-21/2023-01-27", "2023-02-18/2023-02-24"]
    assert "Sampled to the last row of each month (2 of 8 rows)." in result["_notes"]
