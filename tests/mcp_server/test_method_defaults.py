"""Tests for the MCP-only method defaults (config.yaml method_defaults)."""

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from tests.mcp_server.answers import read_answer


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
            "etf", {"indicator": "get_holdings", "tickers": "IWDA.AS"}
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
    text = result[0].text

    assert len(text) < 62_000  # noqa: PLR2004
    assert "was too large and was cut" in text


def test_limit_selects_the_first_or_last_rows(monkeypatch):

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
        return [row["Close"] for row in read_answer(result[0].text)]

    assert closes(2) == [1.0, 2.0]
    assert closes(-2) == [3.0, 4.0]


def test_an_indicator_with_its_own_limit_receives_it(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "etf",
            {"indicator": "get_holdings", "tickers": "IWDA.AS", "limit": -10},
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
        return captured[-1]["simulations"]

    # The tool schema must not carry the indicator's 10,000, or the MCP default of 100
    # could never apply; an explicit value still wins.
    assert simulations() == 100  # noqa: PLR2004
    assert simulations(simulations=5000) == 5000  # noqa: PLR2004


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
        return result[0].text

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
            "econometrics", {"indicator": "get_chow_test", "tickers": "MSFT"}
        )
    )[0].text

    assert "requires `break_date`" in text
    assert captured == []


def test_an_indicator_of_another_tool_is_answered_there(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "overlap", {"indicator": "get_bollinger_bands", "tickers": "AAPL"}
        )
    )[0].text

    assert "indicator of `volatility`, so `volatility` answered it" in text
    assert len(captured) == 1


def test_the_benchmark_note_is_left_out_without_a_benchmark(monkeypatch):
    def holdings(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame({"Asset": ["NVDA"], "Weight": [0.08]})

    monkeypatch.setattr(ToolkitProvider, "call_method", holdings)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "etf", {"indicator": "get_holdings", "tickers": "SPY"}
        )
    )[0].text

    assert "benchmark" not in text.lower()


def test_a_ticker_without_data_and_an_unknown_column_are_named(monkeypatch):
    def rsi(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {
                "AAPL": [55.0, 56.0],
                "NOTATICKER": [None, None],
                "Benchmark": [60.0, 61.0],
            },
            index=pd.period_range("2026-10-08", periods=2, freq="D"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", rsi)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "momentum",
            {
                "indicator": "get_relative_strength_index",
                "tickers": "AAPL,NOTATICKER",
                "show_columns": "AAPL,RSI",
                "limit": -1,
            },
        )
    )[0].text

    assert '"NOTATICKER"' not in text
    assert "No data is available for NOTATICKER" in text
    assert "'RSI' is not part of this answer" in text


def test_limit_does_not_cut_an_answer_of_one_row_per_ticker(monkeypatch):
    def cvar(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"Conditional Value at Risk": [-0.26, -0.28, -0.18]},
            index=pd.Index(["AAPL", "MSFT", "Benchmark"], name="ticker"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", cvar)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "risk",
            {
                "indicator": "get_conditional_value_at_risk",
                "tickers": "AAPL,MSFT",
                "period": "yearly",
                "limit": -2,
            },
        )
    )[0].text

    assert '"AAPL"' in text and "was not applied" in text


def test_a_market_wide_snapshot_needs_no_tickers(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data", {"indicator": "get_market_risk_premium"}
        )
    )

    assert len(captured) == 1


def test_discovery_passes_a_range_only_when_one_is_given(monkeypatch):
    from financetoolkit.discovery.discovery_controller import Discovery

    ranges = []

    def calendar(self, start_date=None, end_date=None):  # noqa: ARG001
        ranges.append((start_date, end_date))
        return pd.DataFrame({"Date": ["2026-10-12"]}, index=["AAPL"])

    monkeypatch.setattr(Discovery, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(Discovery, "get_ipo_calendar", calendar)

    def call(**arguments):
        asyncio.run(
            mcp_controller.mcp.call_tool(
                "discovery", {"indicator": "get_ipo_calendar", **arguments}
            )
        )

    call()
    call(start_date="2026-10-12", end_date="2026-10-31")

    assert ranges == [(None, None), ("2026-10-12", "2026-10-31")]


def test_an_argument_the_indicator_does_not_take_is_named(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "momentum",
            {
                "indicator": "get_relative_strength_index",
                "tickers": "AAPL",
                "short_window": 5,
            },
        )
    )[0].text

    assert "`short_window` is not used by `get_relative_strength_index`" in text


def test_a_tool_specific_default_applies_to_that_tool_only(captured):
    today = datetime.now()

    asyncio.run(
        mcp_controller.mcp.call_tool(
            "discovery", {"indicator": "get_earnings_calendar"}
        )
    )
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data", {"indicator": "get_earnings_calendar", "tickers": "AAPL"}
        )
    )

    # The market-wide calendar looks a week ahead, a company's five years back too.
    assert captured[0]["start_date"] == today.strftime("%Y-%m-%d")
    assert captured[0]["end_date"] == (today + timedelta(days=7)).strftime("%Y-%m-%d")
    assert captured[1]["start_date"] == (today - timedelta(days=1825)).strftime(
        "%Y-%m-%d"
    )


def test_an_unknown_period_is_named_with_the_known_ones(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance",
            {
                "indicator": "get_sharpe_ratio",
                "tickers": "AAPL",
                "period": "fortnightly",
            },
        )
    )[0].text

    assert "Invalid `period`" in text and "monthly" in text
    assert captured == []


def test_a_yearly_answer_to_a_quarterly_question_says_so(monkeypatch):
    def debt(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"United States": [37.0, 38.1]},
            index=pd.period_range("2024", periods=2, freq="Y"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", debt)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "government",
            {
                "indicator": "get_government_debt",
                "countries": "United States",
                "quarterly": True,
            },
        )
    )[0].text

    assert "Quarterly data is not available" in text


def test_a_near_certain_indicator_name_is_read_as_the_known_one(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "models",
            {
                "indicator": "get_gorden_growth_model",
                "tickers": "KO",
                "rate_of_return": 0.08,
                "growth_rate": 0.03,
            },
        )
    )[0].text

    assert "was read as `get_gordon_growth_model`" in text


def test_a_result_and_its_detail_become_named_tables(monkeypatch):
    def model(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        index = pd.period_range("2026-08-28", periods=2, freq="D")
        return (
            pd.DataFrame({"R Squared": [0.39]}, index=["AAPL"]),
            pd.DataFrame({"AAPL": [0.006, -0.005]}, index=index),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", model)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance",
            {
                "indicator": "get_fama_and_french_model",
                "tickers": "AAPL",
                "period": "yearly",
                "include_daily_residuals": True,
                "limit": -1,
            },
        )
    )[0].text

    assert '"Model"' in text and '"Daily Residuals"' in text
    assert "-0.005" in text and "0.006" not in text


def test_show_columns_matches_names_in_any_casing(monkeypatch):
    def quote(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"AAPL": [336.6, 4.9e12], "MSFT": [535.1, 4.0e12]},
            index=["Price", "Market Capitalization"],
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", quote)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data",
            {"indicator": "get_quote", "tickers": "AAPL,MSFT", "show_columns": "price"},
        )
    )[0].text

    assert [row["Price"] for row in read_answer(text)] == [336.6, 535.1]
    assert "Market Capitalization" not in text


def test_limit_leaves_a_table_of_metrics_per_ticker_whole(monkeypatch):
    def float_table(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"AAPL": [0.9988, 14576491739, "https://www.sec.gov"]},
            index=["Free Float", "Float Shares", "Source"],
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", float_table)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data",
            {"indicator": "get_shares_float", "tickers": "AAPL", "limit": -1},
        )
    )[0].text

    assert "Free Float" in text and "one column per ticker" in text


def test_an_empty_answer_over_a_weekend_says_why(monkeypatch):
    monkeypatch.setattr(
        ToolkitProvider, "call_method", lambda self, *args, **kwargs: pd.DataFrame()
    )

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "market_data",
            {
                "indicator": "get_historical_data",
                "tickers": "AAPL",
                "start_date": "2026-10-11",
                "end_date": "2026-10-11",
            },
        )
    )[0].text

    assert "no trading days" in text


def test_a_number_parameter_also_takes_a_list(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "fixed_income",
            {
                "indicator": "get_forward_rate",
                "near_maturity": [1, 2],
                "far_maturity": 10,
            },
        )
    )

    assert captured[-1]["near_maturity"] == [1, 2]
    assert captured[-1]["far_maturity"] == 10


def test_bond_calculations_select_bonds(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "fixed_income",
            {
                "indicator": "collect_bond_statistics",
                "bonds": "912810FT0,91282CPZ8",
                "date": "2026-11-05",
            },
        )
    )

    assert captured[-1]["bonds"] == ["912810FT0", "91282CPZ8"]
    assert captured[-1]["date"] == "2026-11-05"


def test_daily_style_analysis_is_not_blocked(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance",
            {
                "indicator": "get_style_analysis",
                "tickers": "ARKK",
                "style_tickers": "IWF,AGG",
                "period": "daily",
            },
        )
    )

    assert captured[-1]["period"] == "daily"


def test_default_dates_follow_the_day_of_the_call(captured, monkeypatch):
    from financetoolkit.mcp_server.inspection import inspection_controller

    tool = asyncio.run(mcp_controller.mcp.list_tools())
    schema = next(t for t in tool if t.name == "valuation").inputSchema["properties"]
    assert schema["start_date"]["default"] == ""
    assert schema["end_date"]["default"] == ""

    class Tomorrow(datetime):
        @classmethod
        def today(cls):
            return datetime(2031, 3, 15)

    monkeypatch.setattr(inspection_controller, "datetime", Tomorrow)
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "valuation",
            {"indicator": "get_price_to_earnings_ratio", "tickers": "AAPL"},
        )
    )
    assert captured[-1]["end_date"] == "2031-03-15"
    assert captured[-1]["start_date"] == "2026-03-16"


def test_limit_zero_returns_every_row(monkeypatch):

    def prices(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"Close": [1.0, 2.0, 3.0]},
            index=pd.period_range("2026-01-01", periods=3, freq="D"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", prices)
    result = asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance",
            {"indicator": "get_sharpe_ratio", "tickers": "AAPL", "limit": 0},
        )
    )
    assert [row["Close"] for row in read_answer(result[0].text)] == [1.0, 2.0, 3.0]


def test_too_many_tickers_or_countries_are_refused(captured):
    tickers = ",".join(f"T{number}" for number in range(51))
    answer = str(
        asyncio.run(
            mcp_controller.mcp.call_tool(
                "valuation",
                {"indicator": "get_price_to_earnings_ratio", "tickers": tickers},
            )
        )
    )
    assert "received 51 tickers; at most 50" in answer
    assert not captured


@pytest.mark.parametrize("period", ["Yearly", "annual", "YEARLY"])
def test_the_period_reaches_the_indicator_normalised(captured, period):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "performance",
            {"indicator": "get_sharpe_ratio", "tickers": "AAPL", "period": period},
        )
    )

    assert captured[-1]["period"] == "yearly"


def test_a_ticker_left_out_by_show_columns_is_not_missing(monkeypatch):
    def var(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"AAPL": [-0.037], "MSFT": [-0.038], "Benchmark": [-0.026]},
            index=pd.period_range("2022", periods=1, freq="Y"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", var)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "risk",
            {
                "indicator": "get_value_at_risk",
                "tickers": "AAPL,MSFT",
                "period": "yearly",
                "within_period": True,
                "show_columns": "AAPL",
            },
        )
    )[0].text

    assert '"MSFT"' not in text
    assert "No data is available" not in text


def test_limit_zero_lifts_the_mcp_default_limit(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "etf", {"indicator": "get_holdings", "tickers": "SPY", "limit": 0}
        )
    )
    assert "limit" not in captured[-1]

    asyncio.run(
        mcp_controller.mcp.call_tool(
            "etf", {"indicator": "get_holdings", "tickers": "SPY"}
        )
    )
    assert captured[-1]["limit"] == 25  # noqa: PLR2004


def test_an_unknown_parameter_is_named_with_a_near_match(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "risk",
            {
                "indicator": "get_value_at_risk",
                "tickers": "AAPL",
                "alpa": 0.01,
                "colour": "red",
            },
        )
    )[0].text

    assert (
        "`alpa` is not a parameter of `risk` and was ignored; did you mean `alpha`?"
        in text
    )
    assert "`colour` is not a parameter of `risk` and was ignored." in text
    assert "alpa" not in captured[-1] and "colour" not in captured[-1]


def test_the_tool_schema_does_not_take_unknown_parameters():
    tools = asyncio.run(mcp_controller.mcp.list_tools())
    schema = next(tool for tool in tools if tool.name == "risk").inputSchema

    assert schema.get("additionalProperties") in (None, False)


def test_a_series_that_lags_behind_the_last_rows_is_named(monkeypatch):
    import json

    def inflation(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"United States": [0.034, 0.037, None], "Germany": [0.029, 0.033, 0.028]},
            index=pd.period_range("2026-08", periods=3, freq="M"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", inflation)

    def notes(limit):
        text = asyncio.run(
            mcp_controller.mcp.call_tool(
                "macroeconomics",
                {
                    "indicator": "get_inflation_rate",
                    "countries": "United States,Germany",
                    "limit": limit,
                },
            )
        )[0].text
        return json.loads(text).get("_notes", []) if text.startswith("{") else []

    assert any(
        note.startswith("United States (latest 2026-09) has no value for the selected")
        for note in notes(-1)
    )
    # With a value in the selected rows, there is nothing to say.
    assert not any("latest 2026-09" in note for note in notes(-2))


def test_a_monthly_answer_to_a_quarterly_question_says_so(monkeypatch):
    def inflation(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"United States": [0.068, 0.070]},
            index=pd.period_range("2021-11", periods=2, freq="M"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", inflation)

    def answer(**arguments):
        return asyncio.run(
            mcp_controller.mcp.call_tool(
                "macroeconomics",
                {
                    "indicator": "get_inflation_rate",
                    "countries": "United States",
                    "quarterly": True,
                    **arguments,
                },
            )
        )[0].text

    assert (
        "Quarterly data is not available for `get_inflation_rate`, so the monthly figures are shown."
        in answer()
    )
    # A period given explicitly is what was asked for.
    assert "Quarterly data is not available" not in answer(period="monthly")


def test_a_parameter_of_another_tool_is_named_and_not_applied(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "macroeconomics",
            {
                "indicator": "get_economic_calendar",
                "benchmark_ticker": "QQQ",
                "basket": True,
                "risk_free_rate": "3m",
            },
        )
    )[0].text

    for pname in ("benchmark_ticker", "basket", "risk_free_rate"):
        assert (
            f"`{pname}` is not a parameter of `macroeconomics` and was ignored." in text
        )
    assert "QQQ" not in str(captured[-1])
    assert captured[-1].get("basket") in (None, False)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search_metrics", {"query": "volatility", "quer": "x"}),
        ("search_by_category", {"category": "risk", "quer": "x"}),
        ("search_metric_categories", {"quer": "x"}),
    ],
)
def test_the_search_tools_name_an_unknown_parameter(tool, arguments):
    text = asyncio.run(mcp_controller.mcp.call_tool(tool, arguments))[0].text

    assert f"`quer` is not a parameter of `{tool}` and was ignored" in text


def test_limit_does_not_cut_tickers_next_to_a_portfolio_row(monkeypatch):
    def component(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"Component VaR": [-0.0168, -0.0157, -0.0325]},
            index=pd.Index(["AAPL", "MSFT", "Portfolio"], name="ticker"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", component)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "risk",
            {
                "indicator": "get_component_value_at_risk",
                "tickers": "AAPL,MSFT",
                "limit": -1,
            },
        )
    )[0].text

    assert '"AAPL"' in text and '"MSFT"' in text and '"Portfolio"' in text
    assert "was not applied" in text


@pytest.mark.parametrize(
    ("tool", "indicator", "tickers", "rows"),
    [
        ("fixed_income", "get_loan_payment", {}, 1),
        ("fixed_income", "get_affordable_loan", {}, 1),
        ("overlap", "get_support_resistance_levels", {"tickers": "AAPL"}, 20),
        ("econometrics", "get_move_significance", {"tickers": "AAPL"}, 20),
    ],
)
def test_a_default_limit_selects_the_last_rows(
    monkeypatch, tool, indicator, tickers, rows
):

    def series(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"Value": [float(number) for number in range(100)]},
            index=pd.period_range("2026-01-01", periods=100, freq="D"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", series)

    def answer(**arguments):
        return asyncio.run(
            mcp_controller.mcp.call_tool(
                tool, {"indicator": indicator, **tickers, **arguments}
            )
        )[0].text

    text = answer()
    data = read_answer(text)["data"]
    assert [row["Value"] for row in data] == [float(n) for n in range(100 - rows, 100)]
    assert f"limit=-{rows} applies by default" in text

    # An explicit limit, limit=0 and a range given with dates win.
    assert len(read_answer(answer(limit=-3))) == 3  # noqa: PLR2004
    for arguments in ({"limit": 0}, {"start_date": "2026-01-01"}):
        text = answer(**arguments)
        assert "applies by default" not in text
        assert len(read_answer(text)) == 100  # noqa: PLR2004


def test_a_cryptocurrency_in_a_statement_tool_has_no_statements(monkeypatch):
    def ratio(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame({"2025": [30.0]}, index=pd.Index(["AAPL"], name="ticker"))

    monkeypatch.setattr(ToolkitProvider, "call_method", ratio)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "valuation",
            {
                "indicator": "get_price_to_earnings_ratio",
                "tickers": "AAPL,BTC-USD,NOTATICKER",
            },
        )
    )[0].text

    assert (
        "BTC-USD is left out: currencies and cryptocurrencies have no financial" in text
    )
    assert "No data is available for NOTATICKER" in text
    assert "No data is available for BTC-USD" not in text


@pytest.mark.parametrize(
    ("indicator", "currencies", "noted"),
    [
        ("get_earnings_per_share", {"AAPL": "USD", "ASML.AS": "EUR"}, True),
        ("get_earnings_per_share", {"AAPL": "USD", "ASML.AS": "USD"}, False),
        ("get_price_to_earnings_ratio", {"AAPL": "USD", "ASML.AS": "EUR"}, False),
    ],
)
def test_a_ranking_of_amounts_in_different_currencies_says_so(
    monkeypatch, indicator, currencies, noted
):
    def values(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        return pd.DataFrame(
            {"AAPL": [7.5], "ASML.AS": [25.0]},
            index=pd.period_range("2025", periods=1, freq="Y"),
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", values)
    monkeypatch.setattr(
        ToolkitProvider, "get_value_currencies", lambda self, **kwargs: currencies
    )

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "valuation",
            {"indicator": indicator, "tickers": "AAPL,ASML.AS", "rank": True},
        )
    )[0].text

    assert '"Rank"' in text
    assert ("in each ticker's own currency (AAPL: USD, ASML.AS: EUR)" in text) is noted


@pytest.mark.parametrize(
    ("tool", "indicator", "name"),
    [
        # risk has a ticker parameter (of another indicator), performance has none.
        ("risk", "get_value_at_risk", "ticker"),
        ("performance", "get_sharpe_ratio", "ticker"),
        ("performance", "get_sharpe_ratio", "symbol"),
    ],
)
def test_a_ticker_passed_as_ticker_is_pointed_to_tickers(
    captured, tool, indicator, name
):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(tool, {"indicator": indicator, name: "AAPL"})
    )[0].text

    assert (
        f"`{name}` is not a parameter of `{indicator}`; pass the ticker symbols as `tickers`"
        in text
    )
    assert "requires a `tickers` parameter" in text
    assert captured == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            KeyError("AAPL"),
            "No data for `risk` (`get_value_at_risk`): 'AAPL' is missing.",
        ),
        (
            ValueError("alpha is wrong"),
            "Invalid input for `risk` (`get_value_at_risk`)",
        ),
    ],
)
def test_missing_data_is_not_called_invalid_input(monkeypatch, error, expected):
    def failing(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        raise error

    monkeypatch.setattr(ToolkitProvider, "call_method", failing)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "risk", {"indicator": "get_value_at_risk", "tickers": "AAPL"}
        )
    )[0].text

    assert expected in text


def test_an_indicator_of_another_tool_is_answered_by_that_tool(monkeypatch):
    called = []

    def ratio(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        called.append((category, method_name, kwargs.get("tickers")))
        return pd.DataFrame({"2024": [20.2]}, index=["TSLA"])

    monkeypatch.setattr(ToolkitProvider, "call_method", ratio)

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "solvency", {"indicator": "get_interest_coverage_ratio", "tickers": "TSLA"}
        )
    )[0].text

    # The interest coverage ratio itself, not solvency's cash interest coverage ratio.
    assert [method for _, method, _ in called] == ["get_interest_coverage_ratio"]
    assert "is an indicator of `profitability`, so `profitability` answered it" in text
    assert "was ignored" not in text


def test_a_name_with_a_word_more_is_not_read_as_another_measure(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "macroeconomics",
            {"indicator": "get_real_gross_domestic_product", "countries": "Japan"},
        )
    )[0].text

    assert not captured
    assert "Unknown indicator `get_real_gross_domestic_product`" in text
    assert "get_gross_domestic_product" in text


def test_an_unknown_indicator_suggests_close_names_of_other_tools(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "solvency", {"indicator": "get_interest_coverage", "tickers": "TSLA"}
        )
    )[0].text

    assert not captured
    assert "get_interest_coverage_ratio (tool `profitability`)" in text
