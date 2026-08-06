"""Disabled MCP Provider Caching Tests"""

import pandas as pd

from financetoolkit.cache import cache_controller
from financetoolkit.mcp_server.provider_model import ToolkitProvider


def build_provider(tmp_path, enabled):
    """Construct a provider against an isolated database path."""
    cache_controller.reset_cache_registry()

    return ToolkitProvider(
        cache_ttl=600,
        database_location=str(tmp_path / "cache.db"),
        api_key="key",
        fred_api_key="fred",
        cache_enabled=enabled,
    )


def test_a_disabled_provider_never_creates_the_database(tmp_path):
    """Test that a hosted server with caching off writes nothing to disk.

    The concern is not only that entries are unused but that the file exists at
    all: a hosted deployment should not accumulate a database it has no eviction
    policy for.
    """
    provider = build_provider(tmp_path, enabled=False)

    assert not (tmp_path / "cache.db").exists()
    assert provider._cache.enabled is False
    assert provider._cache_ttl == 0

    # Writes are accepted and discarded rather than raising, so no call site has
    # to branch on whether caching is on.
    provider._cache.set(
        source="MCP",
        dataset="tool",
        entity="ratios.get_current_ratio",
        data=pd.DataFrame({"AAPL": [1.0]}),
        parameters={},
    )

    assert not (tmp_path / "cache.db").exists()

    cache_controller.reset_cache_registry()


def test_a_disabled_provider_opts_the_library_out_too(tmp_path):
    """Test that the source data underneath a tool response is uncached as well.

    Zeroing the tool-response TTL alone would leave price history and statements
    still being written by Toolkit, which is the layer that actually fills a disk.
    """
    provider = build_provider(tmp_path, enabled=False)

    assert provider._use_cached_data is False
    assert cache_controller.get_active_cache() is None

    cache_controller.reset_cache_registry()


def test_an_enabled_provider_still_caches(tmp_path):
    """Test that the local single-user default is unchanged."""
    provider = build_provider(tmp_path, enabled=True)

    assert provider._cache.enabled is True
    assert provider._cache_ttl == 600
    assert provider._use_cached_data == str(tmp_path / "cache.db")
    assert cache_controller.get_active_cache() is provider._cache

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()
