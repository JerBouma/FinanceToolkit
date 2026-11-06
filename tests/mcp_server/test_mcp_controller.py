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


def test_setup_leaves_a_config_it_cannot_read_unchanged(monkeypatch, tmp_path):
    import json

    from financetoolkit.mcp_server import setup_model

    monkeypatch.setattr(
        setup_model, "get_global_env_path", lambda: tmp_path / "config" / ".env"
    )
    vscode = tmp_path / ".vscode"
    vscode.mkdir()
    text = '{\n  // my servers\n  "servers": {"other": {"command": "x"}}\n}\n'
    (vscode / "mcp.json").write_text(text)

    setup_model.write_vscode_config("KEY", tmp_path)
    setup_model.remove_vscode_config(tmp_path)

    assert (vscode / "mcp.json").read_text() == text

    (vscode / "mcp.json").write_text('{"servers": {"other": {"command": "x"}}}')
    setup_model.write_vscode_config("KEY", tmp_path)
    servers = json.loads((vscode / "mcp.json").read_text())["servers"]
    assert set(servers) == {"other", "finance-toolkit"}


def test_project_configs_leave_out_keys_the_global_env_holds(monkeypatch, tmp_path):
    from financetoolkit.mcp_server import setup_model

    env = tmp_path / ".env"
    env.write_text("FINANCIAL_MODELING_PREP_API_KEY=FMPKEY\n")
    monkeypatch.setattr(setup_model, "get_global_env_path", lambda: env)

    assert setup_model.keys_outside_global_env("FMPKEY", "FREDKEY") == ("", "FREDKEY")
    assert setup_model.keys_outside_global_env("OTHER", "") == ("OTHER", "")


def test_claude_desktop_on_linux_reads_its_capitalised_config_folder(
    monkeypatch, tmp_path
):
    from financetoolkit.mcp_server import setup_model

    monkeypatch.setattr(setup_model.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_model.pathlib.Path, "home", lambda: tmp_path)

    assert setup_model.get_claude_config_path() == (
        tmp_path / ".config" / "Claude" / "claude_desktop_config.json"
    )


def test_devin_desktop_uses_the_windsurf_folder_only_where_it_exists(
    monkeypatch, tmp_path
):
    from financetoolkit.mcp_server import setup_model

    monkeypatch.setattr(setup_model.pathlib.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(setup_model.platform, "system", lambda: "Linux")

    assert setup_model.get_devin_config_path() == (
        tmp_path / ".config" / "devin" / "mcp_config.json"
    )

    monkeypatch.setattr(setup_model.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    assert setup_model.get_devin_config_path() == (
        tmp_path / "AppData" / "devin" / "mcp_config.json"
    )

    (tmp_path / ".codeium" / "windsurf").mkdir(parents=True)
    assert setup_model.get_devin_config_path() == (
        tmp_path / ".codeium" / "windsurf" / "mcp_config.json"
    )


def test_the_global_copilot_config_follows_copilot_home(monkeypatch, tmp_path):
    import json

    from financetoolkit.mcp_server import setup_model

    monkeypatch.setattr(setup_model.pathlib.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        setup_model, "get_global_env_path", lambda: tmp_path / "config" / ".env"
    )
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    assert (
        setup_model.get_copilot_config_path()
        == tmp_path / ".copilot" / "mcp-config.json"
    )

    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "copilot"))
    setup_model.write_copilot_config("KEY")
    written = json.loads((tmp_path / "copilot" / "mcp-config.json").read_text())

    assert written["mcpServers"]["finance-toolkit"]["command"] == "uvx"

    setup_model.remove_copilot_config()
    written = json.loads((tmp_path / "copilot" / "mcp-config.json").read_text())
    assert "finance-toolkit" not in written["mcpServers"]


def test_vscode_entries_name_their_stdio_transport(monkeypatch, tmp_path):
    import json

    from financetoolkit.mcp_server import setup_model

    monkeypatch.setattr(
        setup_model, "get_global_env_path", lambda: tmp_path / "config" / ".env"
    )

    setup_model.write_vscode_config("KEY", tmp_path)
    interactive = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
    assert interactive["servers"]["finance-toolkit"]["type"] == "stdio"

    setup_model.write_client_config_uvx("vscode", tmp_path / "cli", api_key="KEY")
    non_interactive = json.loads(
        (tmp_path / "cli" / ".vscode" / "mcp.json").read_text()
    )
    assert non_interactive["servers"]["finance-toolkit"]["type"] == "stdio"
