"""MCP Provider Model Tests"""

# Both bugs covered here only showed up through the MCP server, never in the library:
# the server built FixedIncome without the FMP key (so get_treasury_rates came back
# empty) and silently swapped a benchmark that was also a requested ticker (so
# "Benchmark" was QQQ while SPY was asked for). The Toolkit and FixedIncome classes
# are replaced by recorders, so these run offline and check what the server builds.

import pandas as pd
import pytest

from financetoolkit.mcp_server.registry import provider_model
from financetoolkit.mcp_server.registry.provider_model import (
    ToolkitProvider,
    resolve_benchmark_ticker,
)
from tests.mcp_server.answers import read_answer


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


def test_the_standalone_instances_are_bounded_least_recently_used_first(
    monkeypatch, provider
):
    class LightEconomics:
        def __init__(self, **kwargs):
            pass

        def get_inflation_rate(self):
            return pd.Series([0.02], name="Germany")

    monkeypatch.setattr(provider_model, "Economics", LightEconomics)

    starts = ["2000-01-01", "2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01"]
    for start in starts:
        provider.call_standalone_module_functionality(
            module_name="economics",
            method_name="get_inflation_rate",
            start_date=start,
            end_date="2026-01-01",
            quarterly=False,
        )

    kept = list(provider._standalone_cache)

    assert len(kept) == provider_model.MAXIMUM_STANDALONE_INSTANCES
    assert not any("2000-01-01" in key for key in kept)


def test_a_single_country_reaches_a_method_of_one_country(monkeypatch, provider):
    class CurveFixedIncome:
        def __init__(self, **kwargs):
            pass

        def get_yield_curve_principal_components(self, country="United States"):
            return pd.DataFrame({"PC1": [0.3]}, index=[country])

    monkeypatch.setattr(provider_model, "FixedIncome", CurveFixedIncome)
    arguments = dict(
        module_name="fixedincome",
        method_name="get_yield_curve_principal_components",
        start_date="2020-01-01",
        end_date="2026-01-01",
        quarterly=False,
    )

    result = provider.call_standalone_module_functionality(
        countries=["Germany"], **arguments
    )
    assert list(result.index) == ["Germany"]

    with pytest.raises(ValueError, match="one country at a time"):
        provider.call_standalone_module_functionality(
            countries=["Germany", "Japan"], **arguments
        )


def test_calls_on_one_cached_toolkit_run_one_at_a_time(monkeypatch, tmp_path):
    import threading
    import time

    from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider

    class SlowToolkit:
        def __init__(self):
            self.active = 0
            self.peak = 0

        def get_profile(self):
            self.active += 1
            self.peak = max(self.peak, self.active)
            time.sleep(0.05)
            self.active -= 1
            return "profile"

    provider = ToolkitProvider(
        cache_ttl=0,
        database_location=str(tmp_path),
        api_key="KEY",
        cache_enabled=False,
    )
    shared = SlowToolkit()
    monkeypatch.setattr(provider, "get_toolkit_instance", lambda **kwargs: shared)

    def call():
        provider.call_toolkit_functionality(
            "get_profile", ["AAPL"], "2024-01-01", "2025-01-01", False, "SPY"
        )

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert shared.peak == 1
    assert provider.instance_lock(shared) is provider.instance_lock(shared)


def test_the_risk_free_rate_is_part_of_the_toolkit(provider):
    arguments = {
        "tickers": ["VOO"],
        "start_date": "2021-01-01",
        "end_date": "2025-12-31",
        "quarterly": False,
        "benchmark_ticker": "SPY",
        "api_key": "test-key",
    }
    default = provider.get_toolkit_instance(**arguments)
    bill = provider.get_toolkit_instance(**arguments, toolkit_risk_free_rate="13w")

    assert default.kwargs["risk_free_rate"] == "10y"
    assert bill.kwargs["risk_free_rate"] == "13w"
    # A Toolkit kept for one rate is never handed out for another.
    assert bill is not default


def test_the_basket_is_part_of_the_toolkit(provider):
    arguments = {
        "tickers": ["MSFT", "AMZN"],
        "start_date": "2021-01-01",
        "end_date": "2025-12-31",
        "quarterly": True,
        "benchmark_ticker": "SPY",
        "api_key": "test-key",
    }
    default = provider.get_toolkit_instance(**arguments)
    basket = provider.get_toolkit_instance(**arguments, toolkit_basket=True)

    assert default.kwargs["basket"] is False
    assert basket.kwargs["basket"] is True
    # A Toolkit kept without the Total is never handed out for a basket.
    assert basket is not default


def test_an_instance_unused_for_the_idle_time_is_dropped(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(provider_model.time, "monotonic", lambda: now[0])
    released = []
    monkeypatch.setattr(provider_model, "release_memory", lambda: released.append(1))
    cache = provider_model.InstanceCache(maximum=3, idle_seconds=60)

    cache.keep("old", "first")
    now[0] += 50
    cache.keep("recent", "second")
    now[0] += 20

    # "old" was last used 70 seconds ago, "recent" 20 seconds ago.
    assert cache.recall("recent") == "second"
    assert list(cache) == ["recent"]
    assert released == [1]

    # Using an instance keeps it, and the least recently used goes beyond the maximum.
    for key in ("a", "b", "c"):
        now[0] += 10
        cache.keep(key, key)
    assert list(cache) == ["a", "b", "c"]
    # A second instance for a kept key leaves the kept one in place.
    assert cache.keep("a", "other") == "a"


def test_the_conversion_note_names_every_currency_a_company_reported_in(provider):
    toolkit = provider.get_toolkit_instance(
        tickers=["SHEL"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        quarterly=False,
        benchmark_ticker="SPY",
        api_key="test-key",
    )
    # Reported in GBP until 2021 and in USD after, while priced in USD.
    latest = pd.Series({"SHEL": "USDUSD=X"})
    latest.attrs["pairs"] = pd.DataFrame(
        {"2021": ["GBPUSD=X"], "2022": ["USDUSD=X"]}, index=["SHEL"]
    )
    toolkit._statement_currencies = latest
    toolkit._convert_currency = True

    notes = provider.get_transformation_notes(
        tickers=["SHEL"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        quarterly=False,
        benchmark_ticker="SPY",
        api_key="test-key",
    )

    assert any("SHEL (GBP to USD)" in note for note in notes)


class OrderedToolkit:
    """A Toolkit whose regression takes its first ticker as the dependent one."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.econometrics = self

    def get_ols(self, **kwargs):
        dependent, *independent = self.kwargs["tickers"]
        return pd.DataFrame(
            {"Coefficient": [0.0, 0.5], "Dependent": [dependent, dependent]},
            index=["Intercept", *independent],
        )


def test_the_toolkit_keeps_the_order_of_the_tickers(provider):
    arguments = {
        "start_date": "2023-01-01",
        "end_date": "2025-10-31",
        "quarterly": False,
        "benchmark_ticker": None,
        "api_key": "test-key",
    }
    first = provider.get_toolkit_instance(tickers=["AAPL", "MSFT"], **arguments)
    second = provider.get_toolkit_instance(tickers=["MSFT", "AAPL"], **arguments)

    # The first ticker is the dependent one of a regression or pair, so the order is part
    # of the question and a Toolkit kept for one order is never handed out for another.
    assert second is not first
    assert second.kwargs["tickers"] == ["MSFT", "AAPL"]
    assert provider.get_toolkit_instance(tickers=["aapl", "msft"], **arguments) is first


def test_a_cached_answer_keeps_the_order_of_the_tickers(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_model, "Toolkit", OrderedToolkit)
    monkeypatch.setattr(provider_model, "resolve_api_key", lambda: "")
    provider = ToolkitProvider(
        cache_ttl=3600,
        database_location=str(tmp_path / "cache.sqlite"),
        api_key="test-key",
    )
    request = {
        "module_name": "econometrics",
        "method_name": "get_ols",
        "category": "ticker",
        "start_date": "2023-01-01",
        "end_date": "2025-10-31",
        "benchmark_ticker": None,
    }

    for tickers in (["AAPL", "MSFT"], ["MSFT", "AAPL"], ["AAPL", "MSFT"]):
        answer = provider.call_method(tickers=tickers, **request)
        assert answer["Dependent"].iloc[0] == tickers[0]
        assert list(answer.index) == ["Intercept", tickers[1]]


def test_the_econometrics_tool_answers_each_order_of_the_tickers(monkeypatch, tmp_path):
    import asyncio

    from financetoolkit.mcp_server import mcp_controller

    monkeypatch.setattr(provider_model, "Toolkit", OrderedToolkit)
    monkeypatch.setattr(provider_model, "resolve_api_key", lambda: "")
    provider = ToolkitProvider(
        cache_ttl=3600,
        database_location=str(tmp_path / "cache.sqlite"),
        api_key="test-key",
    )
    # The server's provider answers from this one, with its own cache of answers.
    call_method = ToolkitProvider.call_method
    get_toolkit_instance = ToolkitProvider.get_toolkit_instance
    monkeypatch.setattr(
        ToolkitProvider,
        "call_method",
        lambda self, *args, **kwargs: call_method(provider, *args, **kwargs),
    )
    monkeypatch.setattr(
        ToolkitProvider,
        "get_toolkit_instance",
        lambda self, **kwargs: get_toolkit_instance(provider, **kwargs),
    )

    answers = [
        asyncio.run(
            mcp_controller.mcp.call_tool(
                "econometrics",
                {
                    "indicator": "get_ols",
                    "tickers": tickers,
                    "start_date": "2023-01-01",
                    "end_date": "2025-10-31",
                },
            )
        )[0].text
        for tickers in ("AAPL,MSFT", "MSFT,AAPL")
    ]

    first, second = (read_answer(answer) for answer in answers)
    assert {row["Dependent"] for row in first} == {"AAPL"}
    assert "MSFT" in {row["metric"] for row in first}
    assert {row["Dependent"] for row in second} == {"MSFT"}
    assert "AAPL" in {row["metric"] for row in second}
