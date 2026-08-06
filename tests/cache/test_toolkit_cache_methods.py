"""Toolkit Cache Method Tests"""

import pandas as pd
import pytest

from financetoolkit import Toolkit
from financetoolkit.cache import cache_controller


@pytest.fixture(name="toolkit")
def fixture_toolkit(tmp_path):
    """Provide a Toolkit pointed at an isolated cache holding two tickers."""
    cache_controller.reset_cache_registry()

    toolkit = Toolkit(
        ["AAPL", "MSFT"],
        api_key="",
        sleep_timer=False,
        use_cached_data=str(tmp_path),
    )

    for ticker in ("AAPL", "MSFT"):
        toolkit._cache.store(  # noqa: SLF001
            source="FinancialModelingPrep",
            dataset="historical",
            entity=ticker,
            data=pd.DataFrame(
                {"Close": 1.0},
                index=pd.period_range("2020-01-01", "2020-12-31", freq="D"),
            ),
            start="2020-01-01",
            end="2020-12-31",
        )

    yield toolkit

    cache_controller.reset_cache_registry()


def test_get_cache_contents_lists_what_is_stored(toolkit):
    """Test that the overview reports the stored source, dataset and entity count."""
    contents = toolkit.get_cache_contents()

    assert list(contents["source"]) == ["FinancialModelingPrep"]
    assert list(contents["dataset"]) == ["historical"]
    assert list(contents["entities"]) == [2]


def test_clear_cache_requires_confirmation_for_a_full_wipe(toolkit):
    """Test that removing everything is refused unless it is explicitly confirmed."""
    with pytest.raises(ValueError, match="entire cache"):
        toolkit.clear_cache()

    assert not toolkit.get_cache_contents().empty


def test_clear_cache_can_remove_a_single_ticker(toolkit):
    """Test that a scoped removal leaves the rest of the cache in place."""
    removed = toolkit.clear_cache(source="FinancialModelingPrep", ticker="AAPL")

    assert removed == 1
    assert list(toolkit.get_cache_contents()["entities"]) == [1]


def test_clear_cache_with_confirmation_removes_everything(toolkit):
    """Test that a confirmed removal empties the cache."""
    toolkit.clear_cache(confirm=True)

    assert toolkit.get_cache_contents().empty


def test_clear_cache_works_without_caching_enabled(tmp_path):
    """Test that a cache filled earlier can still be cleared by a later session."""
    cache_controller.reset_cache_registry()

    cache_controller.Cache(location=tmp_path / "financetoolkit_cache.db").set(
        source="FinancialModelingPrep",
        dataset="profile",
        entity="AAPL",
        data=pd.DataFrame({"a": [1]}),
    )

    toolkit = Toolkit("AAPL", api_key="", sleep_timer=False, use_cached_data=False)
    toolkit._cache_location = str(tmp_path)  # noqa: SLF001

    assert toolkit.clear_cache(source="FinancialModelingPrep") == 1

    cache_controller.reset_cache_registry()
