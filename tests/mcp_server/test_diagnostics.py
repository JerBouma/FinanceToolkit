"""Tests for the reasons an MCP tool call without data returns."""

import json

import pandas as pd

from financetoolkit import helpers
from financetoolkit.mcp_server.registry import diagnostics_model
from financetoolkit.mcp_server.registry.formatting_model import format_result
from financetoolkit.utilities.logger_model import get_logger


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
    assert json.loads(format_result(pd.DataFrame(), reason="A key is required.")) == {
        "error": "No data available.",
        "reason": "A key is required.",
    }
    assert json.loads(format_result(None)) == {"error": "No data available."}


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
    answer = json.loads(format_result(list(range(200_000)), max_characters=1_000))

    assert len(json.dumps(answer["data"])) <= 1_000
    assert answer["data"][:3] == [0, 1, 2]
    assert "of 200000 values" in answer["_notes"][-1]
