"""
Finance Toolkit MCP Server
"""

import argparse
import os
import pathlib
import subprocess
import sys

import anyio
import uvicorn
import yaml
from dotenv import dotenv_values, load_dotenv
from fastmcp.server.dependencies import (  # noqa: PLC0415
    get_http_headers,
    get_http_request,
)
from mcp.server.fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware

from financetoolkit.mcp_server import setup_model
from financetoolkit.mcp_server.auth_model import (
    MCPAuthMiddleware,
    register_auth_routes,
    resolve_api_key,
    resolve_fred_api_key,
)
from financetoolkit.mcp_server.inspection_controller import ControllerInspector
from financetoolkit.mcp_server.provider_model import ToolkitProvider
from financetoolkit.mcp_server.registry_controller import ToolRegistry
from financetoolkit.mcp_server.tools_model import UtilityToolRegistry
from financetoolkit.utilities.logger_model import get_logger, setup_logger

# Attached before any module-level log call and before FastMCP is imported.
setup_logger()


def _load_dotenv_configuration() -> None:
    """Load dotenv configuration unless both API keys are already present.

    MCP clients can inject ``FINANCIAL_MODELING_PREP_API_KEY`` and/or the
    optional ``FRED_API_KEY`` directly into the server process environment
    (the ``env`` block in their config).  When a key is present the server
    uses it immediately without reading any file. Otherwise the server falls
    back to ``FINANCETOOLKIT_ENV_FILE`` (a path to a ``.env`` file) and then
    to the global Finance Toolkit ``.env`` location. Both keys are checked
    (rather than short-circuiting on FMP alone) so that a client which embeds
    the FMP key directly but leaves FRED to the env file still picks up FRED.
    """
    if os.environ.get("FINANCIAL_MODELING_PREP_API_KEY") and os.environ.get(
        "FRED_API_KEY"
    ):
        return

    # Order: cwd .env, then FINANCETOOLKIT_ENV_FILE, then the global config dir.
    local_env = pathlib.Path.cwd() / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=True)
    env_file = os.environ.get("FINANCETOOLKIT_ENV_FILE")
    if env_file and pathlib.Path(env_file).exists():
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)


_load_dotenv_configuration()


def _resolve_cache_enabled(configured: object) -> bool:
    """
    Decide whether this server caches anything at all.

    ``FINANCE_TOOLKIT_CACHE_ENABLED`` wins when set. Otherwise the config.yaml
    value is used, where the default ``auto`` means "on locally, off when
    hosted": a stdio server is one user on their own machine, while an HTTP
    server multiplexes every user through one process and one database. Sharing
    cache entries there would serve one subscriber's paid data to another, and
    downloaded source data has no eviction policy that would keep the disk
    bounded, so a hosted server fetches live unless told otherwise.

    Args:
        configured (object): The ``cache.enabled`` value from config.yaml, either
            a boolean or the string ``"auto"``.

    Returns:
        bool: True when the cache should be opened and used.
    """
    override = os.environ.get("FINANCE_TOOLKIT_CACHE_ENABLED", "").strip().lower()

    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False

    if isinstance(configured, bool):
        return configured

    return os.environ.get("MCP_TRANSPORT", "stdio") not in ("sse", "streamable-http")


def _build_mcp_app() -> FastMCP:
    """
    Bootstrap the MCP application and return the configured FastMCP instance.

    Reads the server configuration from config.yaml, instantiates the
    ToolkitProvider, ControllerInspector, ToolRegistry, and UtilityToolRegistry,
    registers all tools, and returns the ready-to-run FastMCP instance. The
    FINANCIAL_MODELING_PREP_API_KEY environment variable (or .env file via
    python-dotenv) is loaded before any component is initialized.

    Returns:
        FastMCP: A fully configured FastMCP instance with all toolkit and utility
            tools registered and ready to serve requests.
    """
    _load_dotenv_configuration()
    logger = get_logger()

    configuration_path = pathlib.Path(__file__).parent / "config.yaml"

    with open(configuration_path, encoding="utf-8") as _f:
        configuration: dict = yaml.safe_load(_f)

    _cache_db_env = os.environ.get("FINANCE_TOOLKIT_CACHE_DB", "")
    _cache_ttl_env = os.environ.get("FINANCE_TOOLKIT_CACHE_TTL", "")
    _cache_enabled = _resolve_cache_enabled(configuration["cache"].get("enabled", True))

    logger.info(
        "Caching is %s for this server.",
        "enabled" if _cache_enabled else "disabled, every request is fetched live",
    )

    provider = ToolkitProvider(
        api_key=os.environ.get("FINANCIAL_MODELING_PREP_API_KEY", ""),
        fred_api_key=os.environ.get("FRED_API_KEY", ""),
        cache_ttl=(
            int(_cache_ttl_env)
            if _cache_ttl_env.isdigit()
            else configuration["cache"]["ttl_seconds"]
        ),
        database_location=_cache_db_env or str(setup_model.get_global_cache_db_path()),
        cache_enabled=_cache_enabled,
    )

    mcp = FastMCP(
        name="Finance Toolkit Analyst",
        log_level="CRITICAL",
        host="0.0.0.0",  # noqa: S104
    )

    controller_inspector = ControllerInspector(
        categories=configuration["categories"],
        skip_params=configuration["skip_params"],
        init_handled_params=configuration["init_handled_params"],
    )

    toolkit_registry = ToolRegistry(
        mcp=mcp,
        provider=provider,
        inspector=controller_inspector,
        module_class_map=configuration["module_class_map"],
        skip_methods=configuration["skip_methods"],
        direct_methods=configuration["direct_methods"],
        tool_groups=configuration["tool_groups"],
        blocked_periods=configuration.get("blocked_periods", {}),
    )

    utility_registry = UtilityToolRegistry(
        mcp=mcp,
        registry=toolkit_registry,
        provider=provider,
        search_stop_words=configuration["search_stop_words"],
        category_descriptions=configuration["category_descriptions"],
    )

    toolkit_count = toolkit_registry.register_all_tools()
    utility_count = utility_registry.register_all_tools()

    # Diagnostic tool for hosted platforms; never exposes the full key.
    if os.environ.get("FT_MCP_DIAG"):

        def diagnostics() -> str:
            """Report what the server sees for API-key resolution (debug)."""

            report: dict = {}
            try:

                headers = get_http_headers(include={"authorization"})
                report["header_names"] = sorted(headers.keys())
                report["has_x_fmp_api_key"] = "x-fmp-api-key" in headers
                report["mcp_session_id"] = headers.get("mcp-session-id", "")
                try:
                    request = get_http_request()
                    report["query_string"] = str(request.url.query)
                    report["query_keys"] = sorted(request.query_params.keys())
                except RuntimeError as exc:
                    report["request_err"] = repr(exc)
            except Exception as exc:  # pragma: no cover - diagnostic only
                report["error"] = repr(exc)

            resolved = resolve_api_key()
            report["resolved_present"] = bool(resolved)
            report["resolved_len"] = len(resolved)
            report["resolved_tail"] = resolved[-4:] if resolved else ""

            resolved_fred = resolve_fred_api_key()
            report["fred_resolved_present"] = bool(resolved_fred)
            report["fred_resolved_len"] = len(resolved_fred)
            report["fred_resolved_tail"] = resolved_fred[-4:] if resolved_fred else ""
            return str(report)

        mcp.add_tool(
            diagnostics, name="diagnostics", description=diagnostics.__doc__ or ""
        )
        logger.info("Diagnostic tool registered (FT_MCP_DIAG set).")

    logger.info(
        f"Finance Toolkit MCP Server ready. Registered {toolkit_count} "
        f"router tools and {utility_count} utility tools."
    )

    return mcp


mcp = _build_mcp_app()
register_auth_routes(mcp)


def main() -> None:
    """
    Start the Finance Toolkit MCP server.

    Bootstraps the MCP application via _build_mcp_app() and starts the server
    using the transport defined by the MCP_TRANSPORT environment variable.
    Defaults to stdio transport when MCP_TRANSPORT is not set, which is the
    correct setting for use with VS Code and other MCP clients.
    """
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    get_logger().info(f"Starting MCP server on transport {transport}")

    if transport in ("sse", "streamable-http"):
        # Apply environment variable overrides for host/port if specified
        host = os.environ.get("MCP_HOST", "0.0.0.0")  # noqa: S104
        port_env = os.environ.get("MCP_PORT", "8000")
        port = int(port_env) if port_env.isdigit() else 8000

        mcp.settings.host = host
        mcp.settings.port = port

        starlette_app = (
            mcp.streamable_http_app()
            if transport == "streamable-http"
            else mcp.sse_app()
        )

        # MCPAuthMiddleware is innermost (runs after CORS)
        starlette_app.add_middleware(MCPAuthMiddleware)
        # CORSMiddleware is outermost — handles OPTIONS before auth check
        starlette_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["WWW-Authenticate"],
        )

        config = uvicorn.Config(
            starlette_app,
            host=mcp.settings.host,
            port=mcp.settings.port,
            log_level="info",
            access_log=True,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
        server = uvicorn.Server(config)
        anyio.run(server.serve)
    else:
        mcp.run(transport=transport)


def inspector() -> None:
    """
    Launch the MCP Inspector UI for interactive testing of the server.

    Invokes the MCP Inspector via npx, pointing it at this module so that all
    registered tools can be explored and tested interactively in a browser.
    Exits with the same return code as the Inspector process.
    """
    server_path = str(pathlib.Path(__file__).resolve())
    sys.exit(
        subprocess.call(  # noqa
            ["npx", "@modelcontextprotocol/inspector", "python", server_path]  # noqa
        )
    )


def setup() -> None:
    """Entry point for ``financetoolkit-mcp-setup``.

    When called **without** arguments the interactive setup wizard is launched.

    When called **with** ``--client`` the configuration is written
    non-interactively using a uvx-based server invocation so the entry works
    without a pre-installed local package.

    The setup contains the following optional arguments:

    --client {claude-desktop,claude-code,vscode,cursor,gemini,windsurf}
        Configure a single client without opening the interactive menu.
    --overwrite
        Silently overwrite an existing configuration.  Without this flag the
        command exits with a warning if the target already contains a
        ``finance-toolkit`` entry.
    """
    parser = argparse.ArgumentParser(
        prog="financetoolkit-mcp-setup",
        description="Finance Toolkit MCP Setup Wizard",
        add_help=True,
    )
    parser.add_argument(
        "--client",
        choices=[
            "claude-desktop",
            "claude-code",
            "vscode",
            "cursor",
            "gemini",
            "windsurf",
        ],
        metavar="CLIENT",
        help=(
            "Configure a specific client non-interactively. "
            "Choices: claude-desktop, claude-code, vscode, cursor, gemini, windsurf."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing configuration without prompting.",
    )

    args = parser.parse_args()

    if args.client:
        _setup_cli(args.client, args.overwrite)
    else:
        _setup_interactive()


def _setup_cli(client: str, overwrite: bool) -> None:
    """Non-interactive setup: write uvx-based config for *client*."""
    setup_model.print_banner()

    api_key, key_source = setup_model.discover_api_key()
    fred_api_key, fred_key_source = setup_model.discover_fred_api_key()
    global_env = setup_model.get_global_env_path()

    if api_key:
        masked = (
            f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"  # noqa
        )
        setup_model.ok(
            f"API key found  [dim]·[/]  [dim]{key_source}[/]  [dim]·[/]  [dim cyan]{masked}[/]"
        )
        if client != "claude-desktop" and (
            not global_env.exists() or key_source != "global config"
        ):
            global_values = dotenv_values(global_env) if global_env.exists() else {}
            if global_values.get("FINANCIAL_MODELING_PREP_API_KEY") != api_key:
                setup_model.info(f"Syncing key to global config ({global_env})…")
                if not setup_model.write_global_env_key(api_key):
                    setup_model.warn(
                        "Could not write the global config file — "
                        "the API key will be embedded directly in the client config instead."
                    )
    else:
        setup_model.warn(
            "No API key found.  "
            "Set FINANCIAL_MODELING_PREP_API_KEY in your environment or run without "
            "--client to use the interactive wizard."
        )

    if fred_api_key:
        masked_fred = (
            f"{fred_api_key[:4]}...{fred_api_key[-4:]}"
            if len(fred_api_key) > 8  # noqa
            else "****"
        )
        setup_model.ok(
            f"FRED API key found (optional)  [dim]·[/]  [dim]{fred_key_source}[/]"
            f"  [dim]·[/]  [dim cyan]{masked_fred}[/]"
        )
        if client != "claude-desktop" and (
            not global_env.exists() or fred_key_source != "global config"
        ):
            global_values = dotenv_values(global_env) if global_env.exists() else {}
            if global_values.get("FRED_API_KEY") != fred_api_key:
                setup_model.info(f"Syncing FRED key to global config ({global_env})…")
                if not setup_model.write_global_env_fred_key(fred_api_key):
                    setup_model.warn(
                        "Could not write the global config file — "
                        "the FRED API key will be embedded directly in the client config instead."
                    )
    else:
        setup_model.info(
            "No FRED API key found — optional, free, only unlocks a handful of "
            "US-only economic indicators. Set FRED_API_KEY in your environment "
            "to include it, or get one at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    setup_model.console.print()
    setup_model.write_client_config_uvx(
        client,
        pathlib.Path.cwd(),
        overwrite,
        api_key=api_key,
        fred_api_key=fred_api_key,
    )

    setup_model.console.print()


def _setup_interactive() -> None:
    """Launch the full interactive setup wizard."""
    setup_model.print_banner()

    # Discover an existing key from all known sources before prompting.
    api_key, key_source = setup_model.discover_api_key()
    fred_api_key, fred_key_source = setup_model.discover_fred_api_key()
    global_env = setup_model.get_global_env_path()

    if api_key:
        masked_key = (
            f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"  # noqa
        )
        setup_model.ok(
            f"API key found  [dim]·[/]  [dim]{key_source}[/]  [dim]·[/]  [dim cyan]{masked_key}[/]"
        )
    else:
        setup_model.warn(
            "No API key found.  Get one at [cyan]https://www.jeroenbouma.com/fmp[/]  [dim](15% discount)[/]"
        )
        setup_model.info("Press [bold]Enter[/] to skip and configure via .env later.")
        setup_model.console.print()
        api_key = setup_model.console.input(
            "  [bold]API Key[/]  [dim cyan]›[/] "
        ).strip()

        if api_key and not setup_model.write_global_env_key(api_key):
            setup_model.warn(
                "Could not write the global config file — "
                "the API key will be embedded directly in the client config instead."
            )

    setup_model.console.print()

    if fred_api_key:
        masked_fred_key = (
            f"{fred_api_key[:4]}...{fred_api_key[-4:]}"
            if len(fred_api_key) > 8  # noqa
            else "****"
        )
        setup_model.ok(
            f"FRED API key found (optional)  [dim]·[/]  [dim]{fred_key_source}[/]"
            f"  [dim]·[/]  [dim cyan]{masked_fred_key}[/]"
        )
    else:
        setup_model.info(
            "FRED API key — [bold]optional[/], free, unlocks a handful of "
            "US-only economic indicators (nonfarm payrolls, jobless claims, "
            "the real TIPS yield curve, and similar). Get one at "
            "[cyan]https://fred.stlouisfed.org/docs/api/api_key.html[/]"
        )
        setup_model.info("Press [bold]Enter[/] to skip — everything else still works.")
        setup_model.console.print()
        fred_api_key = setup_model.console.input(
            "  [bold]FRED API Key (optional)[/]  [dim cyan]›[/] "
        ).strip()

        if fred_api_key and not setup_model.write_global_env_fred_key(fred_api_key):
            setup_model.warn(
                "Could not write the global config file — "
                "the FRED API key will be embedded directly in the client config instead."
            )

    setup_model.console.print()
    setup_model.print_menu()
    setup_model.console.print()
    choice_str = setup_model.console.input("  [cyan]›[/] ").strip()

    if not choice_str or "0" in choice_str:
        setup_model.console.print()
        setup_model.info("Setup cancelled.")
        setup_model.console.print()
        return

    cwd = pathlib.Path.cwd()

    # Option 7 is handled as a distinct removal flow.
    if "7" in choice_str:
        setup_model.remove_all_configs(cwd)
        return

    # Extract unique valid choices from the input string (e.g., '13' -> ['1', '3'])
    valid_map = {
        "1": ("Claude Desktop", setup_model.write_claude_config),
        "2": ("Claude Code", setup_model.write_claude_code_config),
        "3": (
            "VS Code",
            lambda k, fk: setup_model.write_vscode_config(k, cwd, fk),
        ),
        "4": (
            "Cursor",
            lambda k, fk: setup_model.write_cursor_config(k, cwd, fk),
        ),
        "5": ("Gemini", setup_model.write_gemini_config),
        "6": ("Windsurf", setup_model.write_windsurf_config),
    }

    # Filter only valid numeric choices from input
    to_process = [c for c in dict.fromkeys(choice_str) if c in valid_map]

    if not to_process:
        setup_model.console.print()
        setup_model.err("No valid options selected.")
        return

    needs_global_env = any(c in to_process for c in ("2", "3", "4", "5", "6"))
    if (
        api_key
        and needs_global_env
        and (not global_env.exists() or key_source != "global config")
    ):
        global_values = dotenv_values(global_env) if global_env.exists() else {}
        if global_values.get("FINANCIAL_MODELING_PREP_API_KEY") != api_key:
            setup_model.info(f"Syncing key to global config ({global_env})…")
            if not setup_model.write_global_env_key(api_key):
                setup_model.warn(
                    "Could not write the global config file — "
                    "the API key will be embedded directly in the client config instead."
                )

    if (
        fred_api_key
        and needs_global_env
        and (not global_env.exists() or fred_key_source != "global config")
    ):
        global_values = dotenv_values(global_env) if global_env.exists() else {}
        if global_values.get("FRED_API_KEY") != fred_api_key:
            setup_model.info(f"Syncing FRED key to global config ({global_env})…")
            if not setup_model.write_global_env_fred_key(fred_api_key):
                setup_model.warn(
                    "Could not write the global config file — "
                    "the FRED API key will be embedded directly in the client config instead."
                )

    setup_model.console.print()
    for char in to_process:
        name, func = valid_map[char]
        try:
            func(api_key, fred_api_key)
        except Exception as e:
            setup_model.err(f"Error configuring {name}: {e}")

    # Final summary
    setup_model.console.print()
    setup_model.console.rule("[dim]Done[/]", style="dim")
    setup_model.console.print()
    setup_model.ok("[bold]All selected configurations updated![/]")
    setup_model.info("Restart your client(s) to apply changes.")

    setup_model.console.print()


if __name__ == "__main__":
    main()
