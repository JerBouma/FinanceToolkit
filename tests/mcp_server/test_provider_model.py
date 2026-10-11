"""MCP Provider Model Tests"""

# Both bugs covered here only showed up through the MCP server, never in the library:
# the server built FixedIncome without the FMP key (so get_treasury_rates came back
# empty) and silently swapped a benchmark that was also a requested ticker (so
# "Benchmark" was QQQ while SPY was asked for). The Toolkit and FixedIncome classes
# are replaced by recorders, so these run offline and check what the server builds.

import pandas as pd
import pytest

from financetoolkit.mcp_server import provider_model
from financetoolkit.mcp_server.provider_model import (
    ToolkitProvider,
    resolve_benchmark_ticker,
)


class RecordingToolkit:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class RecordingFixedIncome:
    instances: list["RecordingFixedIncome"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingFixedIncome.instances.append(self)

    def get_treasury_rates(self):
        return pd.DataFrame({"10 Year": [0.04]})


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_model, "Toolkit", RecordingToolkit)
    monkeypatch.setattr(provider_model, "FixedIncome", RecordingFixedIncome)
    monkeypatch.setattr(provider_model, "resolve_api_key", lambda: "")
    monkeypatch.setattr(provider_model, "resolve_fred_api_key", lambda: "")
    RecordingFixedIncome.instances = []

    return ToolkitProvider(
        cache_ttl=0,
        database_location=str(tmp_path),
        api_key="",
        fred_api_key="",
        cache_enabled=False,
    )


@pytest.mark.parametrize(
    ("tickers", "benchmark", "expected"),
    [
        (["AAPL"], "SPY", "SPY"),
        (["AAPL", "SPY"], "SPY", "SPY"),
        (["aapl", "spy"], "SPY", "SPY"),
        (["SPY"], "SPY", "QQQ"),
        (["QQQ"], "QQQ", "SPY"),
        (["AAPL"], None, None),
    ],
)
def test_resolve_benchmark_ticker(tickers, benchmark, expected):
    resolved, note = resolve_benchmark_ticker(tickers, benchmark)

    assert resolved == expected
    # A note is owed whenever the benchmark overlaps the tickers, and only then.
    overlaps = benchmark is not None and benchmark.upper() in [
        t.upper() for t in tickers
    ]
    assert (note is not None) == overlaps


def test_benchmark_that_is_also_a_ticker_is_kept(provider):
    toolkit = provider.get_toolkit_instance(
        tickers=["AAPL", "SPY"],
        start_date="2026-01-01",
        end_date="2026-09-30",
        quarterly=False,
        benchmark_ticker="SPY",
        api_key="test-key",
    )

    assert toolkit.kwargs["benchmark_ticker"] == "SPY"


def test_benchmark_note_is_returned_with_the_result(provider):
    notes = provider.get_transformation_notes(
        tickers=["AAPL", "SPY"],
        start_date="2026-01-01",
        end_date="2026-09-30",
        quarterly=False,
        benchmark_ticker="SPY",
        api_key="test-key",
    )

    assert any("'Benchmark'" in note for note in notes)


def test_fixedincome_receives_the_fmp_key(provider):
    result = provider.call_standalone_module_functionality(
        module_name="fixedincome",
        method_name="get_treasury_rates",
        start_date="2026-09-01",
        end_date="2026-09-30",
        quarterly=False,
        api_key="test-key",
    )

    assert not result.empty
    assert RecordingFixedIncome.instances[0].kwargs["api_key"] == "test-key"


def test_fixedincome_instances_are_not_shared_between_fmp_keys(provider):
    for key in ["first-key", "second-key"]:
        provider.call_standalone_module_functionality(
            module_name="fixedincome",
            method_name="get_treasury_rates",
            start_date="2026-09-01",
            end_date="2026-09-30",
            quarterly=False,
            api_key=key,
        )

    assert [
        instance.kwargs["api_key"] for instance in RecordingFixedIncome.instances
    ] == [
        "first-key",
        "second-key",
    ]
