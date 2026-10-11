"""MCP controller tests."""

import importlib
import sys
import types


def test_build_mcp_app_uses_global_cache_path(monkeypatch, tmp_path):
    """Ensure the MCP bootstrap points SQLite cache storage at the global config dir."""

    # Pre-imported so the real `mcp` is cached before the stub below replaces it.
    import fastmcp.server.dependencies  # noqa: F401

    captured = {}

    fastmcp_module = types.ModuleType("mcp.server.fastmcp")

    class StubFastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def add_tool(self, *args, **kwargs):
            pass

    fastmcp_module.FastMCP = StubFastMCP
    server_module = types.ModuleType("mcp.server")
    server_module.fastmcp = fastmcp_module
    mcp_module = types.ModuleType("mcp")
    mcp_module.server = server_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    for mod in [
        "financetoolkit.mcp_server.mcp_controller",
        "financetoolkit.mcp_server.registry_controller",
        "financetoolkit.mcp_server.tools_model",
    ]:
        sys.modules.pop(mod, None)

    mcp_controller = importlib.import_module("financetoolkit.mcp_server.mcp_controller")

    class DummyProvider:
        def __init__(
            self,
            *,
            api_key,
            fred_api_key,
            cache_ttl,
            database_location,
            cache_enabled,
            excluded_sources,
        ):
            captured["api_key"] = api_key
            captured["fred_api_key"] = fred_api_key
            captured["cache_ttl"] = cache_ttl
            captured["database_location"] = database_location
            captured["cache_enabled"] = cache_enabled
            captured["excluded_sources"] = excluded_sources

    class DummyInspector:
        def __init__(self, *args, **kwargs):
            pass

    class DummyRegistry:
        def __init__(self, *args, **kwargs):
            pass

        def register_all_tools(self):
            return 0

    class DummyUtilityRegistry:
        def __init__(self, *args, **kwargs):
            pass

        def register_all_tools(self):
            return 0

    class DummyMCP:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.delenv("FINANCE_TOOLKIT_CACHE_DB", raising=False)
    monkeypatch.delenv("FINANCE_TOOLKIT_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    monkeypatch.setattr(
        mcp_controller.setup_model,
        "get_global_cache_db_path",
        lambda: tmp_path / "financetoolkit_cache.db",
    )
    monkeypatch.setattr(mcp_controller, "ToolkitProvider", DummyProvider)
    monkeypatch.setattr(mcp_controller, "ControllerInspector", DummyInspector)
    monkeypatch.setattr(mcp_controller, "ToolRegistry", DummyRegistry)
    monkeypatch.setattr(mcp_controller, "UtilityToolRegistry", DummyUtilityRegistry)
    monkeypatch.setattr(mcp_controller, "FastMCP", DummyMCP)

    mcp_controller._build_mcp_app()

    assert captured["database_location"] == str(tmp_path / "financetoolkit_cache.db")
    assert captured["cache_ttl"] > 0

    # No transport set means stdio, which is the local single-user case.
    assert captured["cache_enabled"] is True
    assert captured["excluded_sources"] == frozenset()


def test_a_hosted_server_shares_everything_but_financial_modeling_prep(monkeypatch):
    """Test that an HTTP transport caches every source but FinancialModelingPrep.

    A hosted server serves every user from one process and one database, so the
    public sources are shared while FinancialModelingPrep data, which belongs to each
    subscriber, is always retrieved with the user's own key.
    """
    from financetoolkit.mcp_server.mcp_controller import _resolve_cache_mode

    monkeypatch.delenv("FINANCE_TOOLKIT_CACHE_ENABLED", raising=False)

    for transport in ("sse", "streamable-http"):
        monkeypatch.setenv("MCP_TRANSPORT", transport)
        assert _resolve_cache_mode("auto") == "shared"

    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    assert _resolve_cache_mode("auto") == "full"

    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    assert _resolve_cache_mode("auto") == "full"


def test_cache_enabled_env_overrides_transport(monkeypatch):
    """Test that the explicit environment override wins over both defaults."""
    from financetoolkit.mcp_server.mcp_controller import _resolve_cache_mode

    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("FINANCE_TOOLKIT_CACHE_ENABLED", "true")

    assert _resolve_cache_mode("auto") == "full"

    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    monkeypatch.setenv("FINANCE_TOOLKIT_CACHE_ENABLED", "false")

    assert _resolve_cache_mode("auto") == "off"

    # An explicit boolean in config.yaml still beats the transport heuristic.
    monkeypatch.delenv("FINANCE_TOOLKIT_CACHE_ENABLED", raising=False)
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")

    assert _resolve_cache_mode(True) == "full"
