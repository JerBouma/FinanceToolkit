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
    )[0][0].text

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
    )[0][0].text

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
    )[0][0].text

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
    )[0][0].text

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
    )[0][0].text

    assert "Quarterly data is not available" in text


def test_a_near_certain_indicator_name_is_read_as_the_known_one(captured):
    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "models",
            {
                "indicator": "get_gordon_growth_model",
                "tickers": "KO",
                "rate_of_return": 0.08,
                "growth_rate": 0.03,
            },
        )
    )[0][0].text

    assert "was read as `get_gorden_growth_model`" in text


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
    )[0][0].text

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
    )[0][0].text

    assert '"Price": 336.6' in text and "Market Capitalization" not in text


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
    )[0][0].text

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
    )[0][0].text

    assert "no trading days" in text


def test_a_curve_is_passed_by_maturity_and_its_absence_is_named(captured):
    def call(**arguments):
        return asyncio.run(
            mcp_controller.mcp.call_tool(
                "fixed_income",
                {
                    "indicator": "get_yield_curve_spread",
                    "short_maturity": 2,
                    "long_maturity": 10,
                    **arguments,
                },
            )
        )[0][0].text

    assert "sample curve" in call()
    assert "sample curve" not in call(spot_rates={"2": 0.048, "10": 0.0524})
    assert captured[-1]["spot_rates"] == {2.0: 0.048, 10.0: 0.0524}


def test_a_number_parameter_also_takes_a_list(captured):
    asyncio.run(
        mcp_controller.mcp.call_tool(
            "fixed_income",
            {
                "indicator": "get_duration",
                "par_value": 1000,
                "coupon_rate": [0.03, 0.05],
                "years_to_maturity": [5, 10],
            },
        )
    )

    assert captured[-1]["coupon_rate"] == [0.03, 0.05]


def test_an_assumed_bond_input_is_named(captured):
    def call(**arguments):
        return asyncio.run(
            mcp_controller.mcp.call_tool(
                "fixed_income",
                {
                    "indicator": "get_present_value",
                    "par_value": 1000,
                    "coupon_rate": 0.04,
                    "years_to_maturity": 5,
                    **arguments,
                },
            )
        )[0][0].text

    assert "Assumed yield_to_maturity=0.08 (the default)" in call()
    assert "Assumed" not in call(yield_to_maturity=0.045)


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
