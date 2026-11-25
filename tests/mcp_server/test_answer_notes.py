"""Tests that every MCP answer applies the parameters it is given or says why not, with notes
about the data in the answer only."""

import asyncio
import json
import logging

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from tests.mcp_server.answers import read_answer

FISCAL_NOTE = (
    "Fiscal years are shown as the calendar year they mostly fall in, so companies "
    "compare like for like: NVDA (e.g. fiscal 2026 as 2025)"
)


@pytest.fixture
def answer(monkeypatch):
    """Answer every tool call with the table set on it (and its log messages), instead of
    fetching data."""
    table = {"result": None, "log": [], "notes": []}

    def call_method(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        for message in table["log"]:
            logging.getLogger("financetoolkit").warning(message)
        return table["result"]

    monkeypatch.setattr(ToolkitProvider, "call_method", call_method)
    monkeypatch.setattr(
        ToolkitProvider,
        "get_transformation_notes",
        lambda self, **kwargs: list(table["notes"]),
    )
    return table


def call(tool, arguments):
    text = asyncio.run(mcp_controller.mcp.call_tool(tool, arguments))[0].text
    try:
        return read_answer(text)
    except json.JSONDecodeError:
        return text


def test_show_columns_with_the_name_of_the_rows_keeps_only_those(answer):
    answer["result"] = pd.DataFrame(
        {"Name": ["Agilent", "Apple"], "Sector": ["Healthcare", "Technology"]},
        index=pd.Index(["A", "AAPL"], name="Symbol"),
    )

    result = call(
        "discovery",
        {"indicator": "get_index_constituents", "show_columns": "Symbol", "limit": 0},
    )

    assert result == ["A", "AAPL"]


def test_show_columns_that_match_nothing_are_named_first(answer):
    answer["result"] = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.MultiIndex.from_product(
            [["AAPL"], [240, 250]], names=["Ticker", "Strike Price"]
        ),
        columns=pd.period_range("2026-11-26", periods=2, freq="D"),
    )

    result = call(
        "options",
        {
            "indicator": "get_black_scholes_model",
            "tickers": "AAPL",
            "show_columns": "182",
        },
    )

    assert len(result["data"][0]) == 4  # noqa: PLR2004
    assert result["_notes"][0].startswith(
        "show_columns: '182' is not part of this answer"
    )


def test_signals_only_keeps_the_periods_of_the_signal_show_columns_names(answer):
    days = pd.period_range("2026-01-05", periods=4, freq="D")
    signals = pd.DataFrame(
        {
            ("Moving Average Crossover", "NVDA"): [0.0, 1.0, 0.0, 0.0],
            ("RSI Oversold", "NVDA"): [1.0, 0.0, 1.0, 1.0],
        },
        index=days,
    )
    signals.index.name = "Date"
    answer["result"] = signals

    result = call(
        "momentum",
        {
            "indicator": "get_signals",
            "tickers": "NVDA",
            "signals_only": True,
            "show_columns": "Moving Average Crossover",
            "benchmark_ticker": "None",
        },
    )

    assert result == [
        {"ticker": "NVDA", "Date": "2026-01-06", "Moving Average Crossover": 1.0}
    ]


def test_a_portfolio_answer_names_no_ticker_as_missing(answer):
    answer["result"] = pd.DataFrame(
        {"Portfolio": [0.10, -0.02], "Benchmark": [0.09, -0.017]},
        index=pd.Index(["Annualized Return", "Value at Risk"], name="metric"),
    )

    result = call(
        "portfolio",
        {
            "indicator": "get_portfolio_risk_metrics",
            "tickers": "SPY,QQQ",
            "weights": '{"SPY": 0.5, "QQQ": 0.5}',
        },
    )

    assert not any("No data is available" in note for note in result.get("_notes", []))


def test_a_missing_ticker_among_answered_ones_is_still_named(answer):
    answer["result"] = pd.DataFrame(
        {"AAPL": [1.0, 2.0]}, index=pd.period_range("2024", periods=2, freq="Y")
    )

    result = call(
        "profitability",
        {"indicator": "get_return_on_equity", "tickers": "AAPL,NOPE"},
    )

    assert any(
        note.startswith("No data is available for NOPE") for note in result["_notes"]
    )


def test_the_fiscal_year_note_is_left_out_of_a_daily_price_answer(answer):
    answer["notes"] = [FISCAL_NOTE]
    answer["result"] = pd.DataFrame(
        {"NVDA": [180.0, 181.0]},
        index=pd.period_range("2026-11-23", periods=2, freq="D"),
    )

    result = call(
        "overlap",
        {"indicator": "get_moving_average", "tickers": "NVDA", "window": 50},
    )

    assert isinstance(result, list)  # no notes at all


def test_the_fiscal_year_note_stays_on_an_answer_per_fiscal_year(answer):
    answer["notes"] = [FISCAL_NOTE]
    answer["result"] = pd.DataFrame(
        [[0.017, 0.021]],
        index=["NVDA"],
        columns=pd.period_range("2024", periods=2, freq="Y"),
    )

    result = call(
        "solvency", {"indicator": "get_free_cash_flow_yield", "tickers": "NVDA"}
    )

    assert FISCAL_NOTE in result["_notes"]


def test_a_missing_change_of_a_country_not_in_the_answer_is_left_out(answer):
    answer["log"] = [
        "No change for United States in 2026-10: its base period 2025-10 has no value.",
        "No change for France in 2026-09: its base period 2025-09 has no value.",
    ]
    answer["result"] = pd.DataFrame(
        {"Germany": [0.029, 0.033], "France": [None, 0.037]},
        index=pd.period_range("2026-09", periods=2, freq="M"),
    )

    result = call(
        "macroeconomics",
        {"indicator": "get_inflation_rate", "countries": "Germany,France"},
    )

    assert result["_notes"] == [
        "No change for France in 2026-09: its base period 2025-09 has no value."
    ]


QUARTERS = pd.period_range("2024Q1", periods=8, freq="Q")


def test_sample_reads_a_quarterly_level_once_a_year(answer):
    answer["result"] = pd.DataFrame(
        {"United States": range(1, 9)}, index=QUARTERS, dtype=float
    )

    result = call(
        "macroeconomics",
        {
            "indicator": "get_gross_domestic_product",
            "countries": "United States",
            "sample": "yearly",
        },
    )

    assert result["data"] == [
        {"date": "2024Q4", "United States": 4.0},
        {"date": "2025Q4", "United States": 8.0},
    ]
    assert "Sampled to the last row of each year (2 of 8 rows)." in result["_notes"]


@pytest.mark.parametrize("parameter", ["sample", "period"])
def test_quarterly_growth_rates_are_not_read_once_a_year(answer, parameter):
    answer["result"] = pd.DataFrame(
        {"United States": [0.01] * 8}, index=QUARTERS, dtype=float
    )

    result = call(
        "macroeconomics",
        {
            "indicator": "get_gross_domestic_product_growth",
            "countries": "United States",
            parameter: "yearly",
        },
    )

    assert len(result["data"]) == len(QUARTERS)
    assert any(
        "growth rates over each quarter" in note
        and "the last one of a year is not the growth over that year" in note
        and "year_over_year=true gives the growth over the past year" in note
        for note in result["_notes"]
    )
    assert not any("is not used by" in note for note in result["_notes"])


def test_the_tracking_error_says_it_is_a_daily_figure(answer):
    answer["result"] = pd.DataFrame(
        {"TSLA": [0.022]}, index=pd.period_range("2026", periods=1, freq="Y")
    )

    result = call(
        "performance",
        {
            "indicator": "get_tracking_error",
            "tickers": "TSLA",
            "period": "yearly",
            "annualize": True,
        },
    )

    assert (
        "`annualize` is not used by `get_tracking_error` and was ignored."
        in result["_notes"]
    )
    assert any(
        "a daily figure: multiply it by the square root of 252" in note
        for note in result["_notes"]
    )


def test_a_rolling_tracking_error_names_its_period(answer):
    answer["result"] = pd.DataFrame(
        {"TSLA": [0.05, 0.06]},
        index=pd.period_range("2026-09", periods=2, freq="M"),
    )

    result = call(
        "performance",
        {
            "indicator": "get_tracking_error",
            "tickers": "TSLA",
            "period": "monthly",
            "rolling": 12,
        },
    )

    assert any(
        "monthly return differences over 12 months, a monthly figure" in note
        and "square root of 12" in note
        for note in result["_notes"]
    )


def test_the_garch_forecast_says_it_is_a_variance(answer):
    answer["result"] = pd.DataFrame(
        {"AAPL": [0.00027, 0.00028]},
        index=pd.period_range("2026-11-27", periods=2, freq="D"),
    )

    result = call("risk", {"indicator": "get_garch_forecast", "tickers": "AAPL"})

    assert (
        "The forecast is the variance of the daily return at each step ahead; its "
        "square root is the daily volatility." in result["_notes"]
    )


def test_year_over_year_growth_is_read_once_a_year(answer):
    answer["result"] = pd.DataFrame(
        {"United States": [0.02] * 8}, index=QUARTERS, dtype=float
    )

    result = call(
        "macroeconomics",
        {
            "indicator": "get_gross_domestic_product_growth",
            "countries": "United States",
            "year_over_year": True,
            "sample": "yearly",
        },
    )

    assert [row["date"] for row in result["data"]] == ["2024Q4", "2025Q4"]
