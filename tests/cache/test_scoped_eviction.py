"""Scoped Eviction Tests"""

import pandas as pd
import pytest

from financetoolkit.cache import cache_controller


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Provide a cache holding both source data and computed tool responses."""
    cache_controller.reset_cache_registry()

    cache = cache_controller.Cache(location=tmp_path / "cache.db")

    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=pd.DataFrame(
            {"Close": 1.0},
            index=pd.period_range("2020-01-01", "2020-12-31", freq="D"),
        ),
        start="2020-01-01",
        end="2020-12-31",
    )
    cache.set(
        source="FinancialModelingPrep",
        dataset="profile",
        entity="AAPL",
        data=pd.DataFrame({"a": [1]}),
    )
    cache.set(
        source="MCP",
        dataset="tool",
        entity="ratios.get_current_ratio",
        data=pd.DataFrame({"AAPL": [1.5]}),
    )

    yield cache

    cache_controller.reset_cache_registry()


def test_scoped_eviction_leaves_source_data_alone(cache):
    """Test that evicting one source does not discard the rest of a shared cache.

    The MCP server evicts its own tool responses on a short schedule while sharing a
    database with the library's price history and filings, which have far longer
    lifetimes. An unscoped eviction there would silently throw those away.
    """
    evicted = cache.remove_expired_entries(ttl=0, source="MCP")

    assert evicted == 1
    assert (
        cache.get(source="MCP", dataset="tool", entity="ratios.get_current_ratio")
        is None
    )
    assert (
        cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")
        is not None
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
    )

    assert plan.fully_cached


def test_unscoped_eviction_removes_everything(cache):
    """Test that an eviction without a scope still clears the whole cache."""
    cache.remove_expired_entries(ttl=0)

    assert cache.get_contents() == []
