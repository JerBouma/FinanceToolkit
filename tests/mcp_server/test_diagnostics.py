"""Tests for the reasons an MCP tool call without data returns."""

import json

import pandas as pd

from financetoolkit import helpers
from financetoolkit.mcp_server.registry import diagnostics_model
from financetoolkit.mcp_server.registry.formatting_model import format_result
from financetoolkit.utilities.logger_model import get_logger
from tests.mcp_server.answers import read_answer


def test_a_missing_key_is_answered_with_how_to_pass_one():
    fmp = diagnostics_model.summarize_reasons(
        [
            "No FinancialModelingPrep API key found. Pass it via the api_key argument.",
            "get_economic_calendar could not be calculated. ValueError: ...",
        ]
    )
    fred = diagnostics_model.summarize_reasons(
        ["No FRED API key found. Register for free."]
    )

    assert fmp == diagnostics_model.FMP_KEY_HINT and "X-FMP-API-Key" in fmp
    assert fred == diagnostics_model.FRED_KEY_HINT and "X-FRED-API-Key" in fred


def test_specific_reasons_come_before_the_generic_message_and_credentials_are_redacted():
    reason = diagnostics_model.summarize_reasons(
        [
            "get_x could not be calculated. ValueError: nothing",
            "OECD API rate limit reached (429 Too Many Requests).",
            "Could not reach BIS (404 for url: https://x.org/data?apikey=SECRET&format=csv).",
            "OECD API rate limit reached (429 Too Many Requests).",
        ]
    )

    assert reason == (
        "OECD API rate limit reached (429 Too Many Requests). | "
        "Could not reach BIS (404 for url: https://x.org/data?apikey=***&format=csv)."
    )
    assert diagnostics_model.summarize_reasons(["get_x could not be calculated."]) == (
        "get_x could not be calculated."
    )
    assert diagnostics_model.summarize_reasons([]) is None


def test_messages_are_captured_per_call_including_worker_threads():
    logger = get_logger()

    def worker(number):
        logger.warning("worker %s could not reach its source", number)
        return number

    with diagnostics_model.capture_call_messages() as messages:
        logger.error("the call itself failed")
        logger.info("informational, not a reason")
        helpers.run_in_parallel(worker, [(1,), (2,)])

    logger.warning("logged after the call, not part of it")

    assert sorted(messages) == [
        "the call itself failed",
        "worker 1 could not reach its source",
        "worker 2 could not reach its source",
    ]


def test_a_response_without_data_carries_the_reason():
    assert read_answer(format_result(pd.DataFrame(), reason="A key is required.")) == {
        "error": "No data available.",
        "reason": "A key is required.",
    }
    assert read_answer(format_result(None)) == {"error": "No data available."}


def test_an_empty_result_is_not_cached(tmp_path, monkeypatch):
    from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider

    provider = ToolkitProvider(
        cache_ttl=3600, database_location=str(tmp_path / "cache.db")
    )
    results = [pd.DataFrame(), pd.DataFrame({"United States": [0.04]})]
    monkeypatch.setattr(
        provider,
        "call_standalone_module_functionality",
        lambda **kwargs: results.pop(0),
    )

    arguments = {
        "module_name": "economics",
        "method_name": "get_x",
        "category": "standalone",
        "countries": ["United States"],
        "start_date": "2026-01-01",
        "end_date": "2026-10-01",
    }

    assert provider.call_method(**arguments).empty
    # The empty answer was not stored, so the second call reaches the source again.
    assert provider.call_method(**arguments).iloc[0, 0] == 0.04


def test_the_key_hints_survive_redaction_and_logs_are_redacted():
    import logging

    from financetoolkit.utilities.logger_model import RedactingFilter, redact

    for hint in (diagnostics_model.FMP_KEY_HINT, diagnostics_model.FRED_KEY_HINT):
        assert redact(hint) == hint
        # A key in a URL ends up in proxy logs and browser history, so it isn't advised.
        assert "_api_key=" not in hint

    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        (
            "1.2.3.4",
            "POST",
            "/mcp?fmp_api_key=SECRET123&fred_api_key=FREDSECRET",
            "1.1",
            200,
        ),
        None,
    )
    RedactingFilter().filter(record)
    assert "SECRET" not in record.getMessage()
    assert "fmp_api_key=***" in record.getMessage()


def test_a_long_list_is_cut_to_the_response_size():
    answer = read_answer(format_result(list(range(200_000)), max_characters=1_000))

    assert len(json.dumps(answer["data"])) <= 1_000
    assert answer["data"][:3] == [0, 1, 2]
    assert "of 200000 values" in answer["_notes"][-1]


def test_a_ticker_without_the_expiry_is_noted_with_its_dates_not_as_unknown(
    monkeypatch, options_module
):
    import asyncio

    from financetoolkit.mcp_server import mcp_controller
    from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
    from financetoolkit.options import options_model

    expiry_dates = {"AAPL": ["2026-12-04", "2026-12-18"], "MSFT": ["2026-12-18"]}

    def fake_option_chains(tickers, expiration_date, put_option=False):  # noqa: ARG001
        chain = pd.DataFrame(
            {
                "Bid": [1.0],
                "Ask": [1.1],
                "Change": [0.1],
                "Percent Change": [0.01],
                "Implied Volatility": [0.3],
            },
            index=pd.MultiIndex.from_tuples(
                [(tickers[0], 100.0)], names=["Ticker", "Strike Price"]
            ),
        )
        return chain if expiration_date in expiry_dates[tickers[0]] else pd.DataFrame()

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        return options_module.get_option_chains(
            expiration_date=kwargs.get("expiration_date")
        )

    monkeypatch.setattr(
        options_model, "get_option_expiry_dates", lambda ticker: expiry_dates[ticker]
    )
    monkeypatch.setattr(options_model, "get_option_chains", fake_option_chains)
    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)

    answer = read_answer(
        asyncio.run(
            mcp_controller.mcp.call_tool(
                "options",
                {
                    "indicator": "get_option_chains",
                    "tickers": "AAPL,MSFT",
                    "expiration_date": "2026-12-04",
                },
            )
        )[0].text
    )

    assert {row["Ticker"] for row in answer["data"]} == {"AAPL"}
    assert answer["_notes"] == [
        "No options data is available for MSFT expiring on 2026-12-04, so it is left "
        "out. Its expiration dates are 2026-12-18."
    ]


def test_an_implied_move_from_stale_quotes_is_a_note():
    message = (
        "EZU 2026-12-18: the 65 call has no two-sided quote and is priced at its last trade of "
        "2026-11-09 (open interest 1), so the implied move may not reflect the live market."
    )

    assert diagnostics_model.missing_data_notes([message]) == [message]
