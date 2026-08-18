"""Cache Removal Tests"""

import pandas as pd
import pytest

from financetoolkit.cache import cache_controller, sqlite_model


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Provide a cache holding a few entries across two sources."""
    cache_controller.reset_cache_registry()

    cache = cache_controller.Cache(location=tmp_path / "cache.db")

    for ticker in ("AAPL", "MSFT"):
        cache.store(
            source="YahooFinance",
            dataset="historical",
            entity=ticker,
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
    cache.store(
        source="OECD",
        dataset="query",
        entity="NLD",
        data=pd.DataFrame(
            {"Value": 1.0}, index=pd.period_range("2020-01-01", "2020-12-31", freq="M")
        ),
        start="2020-01-01",
        end="2020-12-31",
    )

    yield cache

    cache_controller.reset_cache_registry()


def test_remove_by_source_leaves_other_sources_intact(cache):
    """Test that clearing one source does not touch the rest of the cache."""
    removed = cache.remove(source="OECD")

    assert removed == 1

    sources = {entry["source"] for entry in cache.get_contents()}

    assert sources == {"YahooFinance", "FinancialModelingPrep"}


def test_remove_by_entity_leaves_other_entities_intact(cache):
    """Test that clearing a single ticker keeps the others cached."""
    cache.remove(source="YahooFinance", entity="AAPL")

    plan = cache.plan(
        source="YahooFinance",
        dataset="historical",
        entities=["AAPL", "MSFT"],
        start="2020-01-01",
        end="2020-12-31",
    )

    assert plan.entities_to_fetch == ["AAPL"]


def test_remove_by_dataset(cache):
    """Test that clearing one dataset of a source keeps its other datasets."""
    cache.remove(source="FinancialModelingPrep", dataset="profile")

    assert (
        cache.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")
        is None
    )
    assert len(cache.get_contents()) == 2


def test_remove_without_arguments_clears_everything(cache):
    """Test that an unscoped removal empties the cache."""
    cache.remove()

    assert cache.get_contents() == []


def test_get_contents_reports_entities_per_dataset(cache):
    """Test that the overview groups entries by source and dataset."""
    contents = {
        (entry["source"], entry["dataset"]): entry for entry in cache.get_contents()
    }

    assert contents[("YahooFinance", "historical")]["entities"] == ["AAPL", "MSFT"]
    assert contents[("OECD", "query")]["entities"] == ["NLD"]


def test_schema_mismatch_warns_but_never_removes(tmp_path, monkeypatch, caplog):
    """Test that a cache from another schema version is reported, not discarded."""
    cache_controller.reset_cache_registry()

    location = tmp_path / "cache.db"
    cache = cache_controller.Cache(location=location)

    cache.set(
        source="FinancialModelingPrep",
        dataset="profile",
        entity="AAPL",
        data=pd.DataFrame({"a": [1]}),
    )

    monkeypatch.setattr(
        cache_controller, "SCHEMA_VERSION", sqlite_model.SCHEMA_VERSION + 1
    )

    with caplog.at_level("WARNING"):
        reopened = cache_controller.Cache(location=location)

    assert "Nothing has been removed" in caplog.text
    assert (
        reopened.get(source="FinancialModelingPrep", dataset="profile", entity="AAPL")
        is not None
    )

    cache_controller.reset_cache_registry()
