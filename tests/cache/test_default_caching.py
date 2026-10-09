"""Default Caching Tests

Data retrieved from external sources is cached by default in every module, unless
use_cached_data says otherwise or FINANCE_TOOLKIT_CACHE_ENABLED switches the default off.
"""

# pylint: disable=missing-function-docstring

import pandas as pd
import pytest

from financetoolkit import Toolkit
from financetoolkit.cache import cache_controller, ticker_model
from financetoolkit.economics.economics_controller import Economics
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.utilities import error_model


@pytest.fixture(name="shared_location")
def fixture_shared_location(tmp_path, monkeypatch):
    """Point the shared cache to a temporary database and restore the default."""
    location = tmp_path / "shared.db"
    monkeypatch.setenv("FINANCE_TOOLKIT_CACHE_DB", str(location))
    monkeypatch.delenv(cache_controller.CACHE_ENVIRONMENT_VARIABLE, raising=False)
    cache_controller.reset_cache_registry()
    cache_controller.clear_active_cache()

    yield location

    cache_controller.reset_cache_registry()
    cache_controller.clear_active_cache()


def build_toolkit(**kwargs) -> Toolkit:
    return Toolkit(
        tickers=["AAPL"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
        **kwargs,
    )


def test_every_module_caches_by_default(shared_location):
    toolkit = build_toolkit()

    assert toolkit._cache.enabled
    assert toolkit._cache.location == shared_location
    assert Economics()._cache.location == shared_location
    assert FixedIncome()._cache.location == shared_location


def test_the_default_can_be_switched_off_with_the_environment(
    shared_location, monkeypatch
):
    monkeypatch.setenv(cache_controller.CACHE_ENVIRONMENT_VARIABLE, "0")

    assert not build_toolkit()._cache.enabled
    assert Economics()._cache is None
    assert FixedIncome()._cache is None
    # An explicit choice wins over the environment.
    assert build_toolkit(use_cached_data=True)._cache.enabled


def test_caching_can_be_switched_off_per_module(shared_location):
    assert not build_toolkit(use_cached_data=False)._cache.enabled
    assert Economics(use_cached_data=False)._cache is None
    assert FixedIncome(use_cached_data=False)._cache is None


def test_a_dedicated_location(shared_location, tmp_path):
    location = tmp_path / "dedicated.db"

    assert build_toolkit(use_cached_data=str(location))._cache.location == location
    assert Economics(use_cached_data=str(location))._cache.location == location


def collect(cache, collector):
    return ticker_model.collect_per_ticker(
        cache=cache,
        source="FinancialModelingPrep",
        dataset="etf_holdings",
        tickers=["AAPL"],
        ticker_axis=ticker_model.TICKER_ON_INDEX,
        collector=collector,
    )


def test_a_ticker_without_data_is_cached(tmp_path):
    """ETF holdings of a company: the answer is that there are none, asked once."""
    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    calls = []

    def collector(tickers):
        calls.append(tickers)
        return pd.DataFrame()

    collect(cache, collector)
    collect(cache, collector)

    assert len(calls) == 1


def test_a_ticker_whose_request_failed_is_asked_again(tmp_path):
    """A rate limit is not an answer, so it is not remembered as one."""
    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    calls = []

    def collector(tickers):
        calls.append(tickers)
        error_model.report_request_failure()
        return pd.DataFrame()

    collect(cache, collector)
    collect(cache, collector)

    assert len(calls) == 2  # noqa: PLR2004


def test_an_excluded_source_is_never_cached(tmp_path):
    """A hosted MCP server shares every source but FinancialModelingPrep."""
    cache = cache_controller.Cache(
        location=tmp_path / "cache.db", excluded_sources={"FinancialModelingPrep"}
    )
    frame = pd.DataFrame({"value": [1.0]})

    for source in ("FinancialModelingPrep", "OECD"):
        cache.set(source=source, dataset="profile", entity="AAPL", data=frame)
        cache.store(
            source=source,
            dataset="historical",
            entity="AAPL",
            data=frame.set_axis(pd.period_range("2024-01-01", periods=1, freq="D")),
            start="2024-01-01",
            end="2024-01-01",
        )

    assert (
        cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")
        is None
    )
    assert cache.get(source="OECD", dataset="profile", entity="AAPL") is not None
    assert cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities="AAPL",
        start="2024-01-01",
        end="2024-01-01",
    ).entities_to_fetch == ["AAPL"]
    assert not cache.plan(
        source="OECD",
        dataset="historical",
        entities="AAPL",
        start="2024-01-01",
        end="2024-01-01",
    ).entities_to_fetch


def test_a_toolkit_uses_the_cache_it_is_given(shared_location, tmp_path):
    cache = cache_controller.Cache(
        location=tmp_path / "cache.db", excluded_sources={"FinancialModelingPrep"}
    )

    assert build_toolkit(use_cached_data=cache)._cache is cache
