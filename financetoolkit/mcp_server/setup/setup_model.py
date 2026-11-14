"""
Finance Toolkit MCP Setup Model

Helper functions that locate and write the MCP client configuration files for
VS Code, Cursor, and Claude Desktop. Each writer reads any existing file before
writing so that unrelated server entries are never disturbed, and prompts the
user for confirmation when a ``finance-toolkit`` entry is already present.
"""

import json
import os
import pathlib
import platform
import tempfile
from contextlib import suppress

from dotenv import dotenv_values
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

from financetoolkit.cache.cache_controller import get_default_cache_location

console = Console(stderr=True)


def print_banner() -> None:
    """Print the Finance Toolkit MCP setup wizard header."""
    body = Text()
    body.append("\nFinanceToolkit", style="bold cyan")
    body.append("  ·  ", style="dim")
    body.append("MCP Setup Wizard\n", style="bold")
    body.append("Transparent and Efficient Financial Analysis\n", style="dim")
    console.print(Panel(body, border_style="cyan", padding=(0, 2)))
    console.print()


def print_menu() -> None:
    """Print the numbered client-selection menu."""
    text = Text()
    text.append("  1  ", style="bold cyan")
    text.append("Claude Desktop\n")
    text.append("  2  ", style="bold cyan")
    text.append("Claude Code\n")
    text.append("  3  ", style="bold cyan")
    text.append("VS Code\n")
    text.append("  4  ", style="bold cyan")
    text.append("Cursor\n")
    text.append("  5  ", style="bold cyan")
    text.append("Gemini\n")
    text.append("  6  ", style="bold cyan")
    text.append("Devin Desktop (Windsurf)\n")
    text.append("  7  ", style="bold cyan")
    text.append("GitHub Copilot (global)\n\n")
    text.append("  8  ", style="yellow")
    text.append("Remove configuration\n")
    text.append("  0  ", style="dim")
    text.append("Exit", style="dim")
    console.print(
        Panel(
            text,
            title="[bold]Configure Clients[/]",
            subtitle="[dim]e.g. [cyan]13[/] for Claude Desktop + VS Code[/]",
            border_style="dim",
            padding=(1, 2),
        )
    )


def ok(message: str) -> None:
    """Print a success line."""
    console.print(f"  [green]✔[/]  {message}")


def warn(message: str) -> None:
    """Print a warning line."""
    console.print(f"  [yellow]⚠[/]  {message}")


def err(message: str) -> None:
    """Print an error line."""
    console.print(f"  [red]✘[/]  {message}")


def info(message: str) -> None:
    """Print a dim informational line."""
    console.print(f"  [dim]{message}[/]")


def get_claude_config_path() -> pathlib.Path:
    """
    Return the platform-specific path to the Claude Desktop configuration file.

    Returns:
        pathlib.Path: Absolute path to claude_desktop_config.json for the
            current operating system.
    """
    system = platform.system()
    if system == "Darwin":
        return (
            pathlib.Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA") or str(
            pathlib.Path.home() / "AppData" / "Roaming"
        )
        return pathlib.Path(appdata) / "Claude" / "claude_desktop_config.json"
    return pathlib.Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def get_claude_code_config_path() -> pathlib.Path:
    """
    Return the path to the Claude Code user-level MCP configuration file.

    Claude Code stores its user-scoped MCP server registry at
    ``~/.claude.json`` on all platforms.

    Returns:
        pathlib.Path: Absolute path to ``~/.claude.json``.
    """
    return pathlib.Path.home() / ".claude.json"


def get_gemini_config_path() -> pathlib.Path:
    """
    Return the path to the Gemini CLI MCP configuration file.

    Gemini CLI stores its MCP server registry at ``~/.gemini/settings.json``
    on all platforms.

    Returns:
        pathlib.Path: Absolute path to ``~/.gemini/settings.json``.
    """
    return pathlib.Path.home() / ".gemini" / "settings.json"


def get_devin_config_path() -> pathlib.Path:
    """
    Return the path to the Devin Desktop (Windsurf) user MCP configuration file.

    Devin Desktop reads its user-scoped MCP servers from
    ``~/.config/devin/mcp_config.json`` (``%APPDATA%\\devin\\mcp_config.json`` on
    Windows). Installs that still have the ``~/.codeium/windsurf`` folder keep their
    servers in ``~/.codeium/windsurf/mcp_config.json``, so that file is used when the
    folder exists.

    Returns:
        pathlib.Path: Absolute path to the MCP configuration file Devin Desktop reads.
    """
    windsurf_path = pathlib.Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    if windsurf_path.parent.exists():
        return windsurf_path
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA") or str(
            pathlib.Path.home() / "AppData" / "Roaming"
        )
        return pathlib.Path(appdata) / "devin" / "mcp_config.json"
    return pathlib.Path.home() / ".config" / "devin" / "mcp_config.json"


def get_copilot_config_path() -> pathlib.Path:
    """
    Return the path to the global GitHub Copilot MCP configuration file.

    GitHub Copilot (in VS Code and the Copilot CLI) reads servers that apply to every
    workspace from ``$COPILOT_HOME/mcp-config.json``, or ``~/.copilot/mcp-config.json``
    when ``COPILOT_HOME`` is not set. The file uses a top-level ``mcpServers`` key.

    Returns:
        pathlib.Path: Absolute path to the global Copilot MCP configuration file.
    """
    copilot_home = os.environ.get("COPILOT_HOME")
    base = (
        pathlib.Path(copilot_home) if copilot_home else pathlib.Path.home() / ".copilot"
    )
    return base / "mcp-config.json"


def _uvx_server_entry(
    api_key: str | None = None, fred_api_key: str | None = None
) -> dict:
    """Return the MCP server config block that invokes the server via uvx.

    The generated entry is portable: it does not depend on any locally
    installed ``financetoolkit-mcp`` binary.  Clients that do not have the
    package installed can still launch the server because uvx downloads and
    runs it on demand.

    Each key is embedded directly in the ``env`` block when supplied, so the
    server uses it immediately without reading any file. ``FRED_API_KEY`` is
    optional — it only unlocks a handful of US-only economic indicators, and
    the server works fine without it. Whenever either key is missing,
    ``FINANCETOOLKIT_ENV_FILE`` is also included, pointing at the global
    Finance Toolkit ``.env`` file so the missing key(s) can still be picked up
    at runtime without being stored in the client config.

    Args:
        api_key (str | None): FinancialModelingPrep API key.
        fred_api_key (str | None): FRED API key. Optional — enables US labor
            market, macroeconomic and interest rate indicators that have no
            FMP/OECD equivalent.

    Returns:
        dict: Ready-to-serialise MCP server configuration dict.
    """
    env: dict[str, str] = {}
    if api_key:
        env["FINANCIAL_MODELING_PREP_API_KEY"] = api_key
    if fred_api_key:
        env["FRED_API_KEY"] = fred_api_key
    if not api_key or not fred_api_key:
        env["FINANCETOOLKIT_ENV_FILE"] = str(get_global_env_path())

    return {
        "command": "uvx",
        "args": ["--from", "financetoolkit[mcp]", "financetoolkit-mcp"],
        "env": env,
    }


# Maps --client name to (config_path_fn, outer_json_key, display_name).
_CLIENT_CONFIG: dict[str, tuple] = {
    "claude-desktop": (get_claude_config_path, "mcpServers", "Claude Desktop"),
    "claude-code": (get_claude_code_config_path, "mcpServers", "Claude Code"),
    "gemini": (get_gemini_config_path, "mcpServers", "Gemini"),
    "devin": (get_devin_config_path, "mcpServers", "Devin Desktop (Windsurf)"),
    "copilot": (get_copilot_config_path, "mcpServers", "GitHub Copilot (global)"),
    # VS Code and Cursor are workspace-local, so paths resolve at call time.
    "vscode": (None, "servers", "VS Code"),
    "cursor": (None, "mcpServers", "Cursor"),
}


def mask_key(value: str) -> str:
    """
    Shorten an API key to its first and last four characters for display.

    Args:
        value (str): The key.

    Returns:
        str: The masked key, e.g. "abcd...wxyz", or "****" for a short key.
    """
    return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"  # noqa: PLR2004


def mask_entry(entry: object) -> object:
    """
    Copy a client config entry with the API keys in its ``env`` block masked, for display.

    Args:
        entry (object): The server entry from a client config.

    Returns:
        object: The entry with every ``*KEY*`` variable masked.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("env"), dict):
        return entry
    return {
        **entry,
        "env": {
            name: mask_key(str(value)) if "KEY" in name.upper() and value else value
            for name, value in entry["env"].items()
        },
    }


def _replace_file(path: pathlib.Path, content: bytes) -> None:
    """
    Replace a file in one step with a new one only its owner can read and write.

    The content goes to a temporary file in the same folder (created private) that then
    takes the file's place, so an interrupted write or a full disk leaves the old file
    as it was instead of truncated.

    Args:
        path (pathlib.Path): The file to replace or create.
        content (bytes): Its new content.
    """
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise


def write_private_file(path: pathlib.Path, text: str) -> None:
    """
    Write a file that holds API keys (the .env file or a client config): private to its
    owner, replaced in one step, with the previous version kept beside it as
    ``<name>.bak`` (private as well).

    Args:
        path (pathlib.Path): The file to write.
        text (str): Its new content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _replace_file(path.with_name(f"{path.name}.bak"), path.read_bytes())
    _replace_file(path, text.encode("utf-8"))


def get_global_env_path() -> pathlib.Path:
    """
    Return the platform-specific path to the global Finance Toolkit .env file.

    The file lives inside a ``financetoolkit`` subdirectory of the platform's
    standard user-configuration directory:

    * **Windows** — ``%APPDATA%\\financetoolkit\\.env``
    * **macOS / Linux** — ``$XDG_CONFIG_HOME/financetoolkit/.env``
      (falls back to ``~/.config/financetoolkit/.env`` when the variable is
      not set)

    Returns:
        pathlib.Path: Absolute path to the global .env file.
    """
    system = platform.system()
    if system == "Windows":
        base = pathlib.Path(
            os.environ.get("APPDATA")
            or str(pathlib.Path.home() / "AppData" / "Roaming")
        )
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = pathlib.Path(xdg) if xdg else pathlib.Path.home() / ".config"
    return base / "financetoolkit" / ".env"


def keys_outside_global_env(api_key: str, fred_api_key: str) -> tuple[str, str]:
    """
    Leave out the keys the global .env file already holds.

    VS Code and Cursor keep their config in the project (``.vscode``, ``.cursor``), which is
    often committed, so a key the server can read from the global .env file is not written
    there as well.

    Args:
        api_key (str): FinancialModelingPrep API key.
        fred_api_key (str): FRED API key.

    Returns:
        tuple[str, str]: The FMP and FRED keys, each "" when the global .env file holds it.
    """
    global_env = get_global_env_path()
    values = dotenv_values(global_env) if global_env.exists() else {}
    return (
        "" if values.get("FINANCIAL_MODELING_PREP_API_KEY") == api_key else api_key,
        "" if values.get("FRED_API_KEY") == fred_api_key else fred_api_key,
    )


def get_global_cache_db_path() -> pathlib.Path:
    """
    Return the path to the global Finance Toolkit SQLite cache database.

    The location is owned by the cache package rather than by the MCP server,
    since the library writes to the same database. It sits alongside the global
    ``.env`` file so Claude Desktop and other MCP clients write to a user-owned
    configuration directory instead of the current working directory.

    Returns:
        pathlib.Path: Absolute path to the shared cache database.
    """
    return get_default_cache_location()


def _write_global_env_value(env_key: str, value: str, label: str) -> bool:
    """
    Write or update a single key in the global .env file.

    The file (and any missing parent directories) are created automatically if
    they do not already exist. When the key is already present it is updated
    in-place without disturbing any other entries in the file.

    Args:
        env_key (str): The .env variable name to write, e.g.
            ``"FINANCIAL_MODELING_PREP_API_KEY"``.
        value (str): The value to write.
        label (str): Human-readable label used in the success/error message,
            e.g. ``"API key"`` or ``"FRED API key"``.

    Returns:
        bool: ``True`` when the value was written successfully, ``False`` when
            a filesystem error prevented the write (e.g. permission denied).
    """
    env_path = get_global_env_path()
    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)

        env_line = f"{env_key}={value}\n"

        existing_lines: list[str] = []
        existing_index: int | None = None

        if env_path.exists():
            existing_lines = env_path.read_text(encoding="utf-8").splitlines(
                keepends=True
            )
            for idx, line in enumerate(existing_lines):
                if line.strip().startswith(f"{env_key}="):
                    existing_index = idx
                    break

        if existing_index is None:
            if existing_lines and not existing_lines[-1].endswith("\n"):
                existing_lines[-1] += "\n"
            existing_lines.append(env_line)
        else:
            existing_lines[existing_index] = env_line

        write_private_file(env_path, "".join(existing_lines))
        ok(f"{label} saved to [dim cyan]{env_path}[/]")
        return True

    except OSError as exc:
        err(f"Could not write {label} to [dim cyan]{env_path}[/]: {exc}")
        return False


def write_global_env_key(api_key: str) -> bool:
    """
    Write or update ``FINANCIAL_MODELING_PREP_API_KEY`` in the global .env file.

    Args:
        api_key (str): The API key value to write.

    Returns:
        bool: ``True`` when the key was written successfully, ``False`` when a
            filesystem error prevented the write (e.g. permission denied).
    """
    return _write_global_env_value(_ENV_KEY, api_key, "API key")


def write_global_env_fred_key(fred_api_key: str) -> bool:
    """
    Write or update ``FRED_API_KEY`` in the global .env file.

    FRED is optional — this key only unlocks a handful of US-only economic
    indicators (e.g. nonfarm payrolls, the real TIPS yield curve). It's free;
    register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        fred_api_key (str): The FRED API key value to write.

    Returns:
        bool: ``True`` when the key was written successfully, ``False`` when a
            filesystem error prevented the write (e.g. permission denied).
    """
    return _write_global_env_value(_FRED_ENV_KEY, fred_api_key, "FRED API key")


_ENV_KEY = "FINANCIAL_MODELING_PREP_API_KEY"
_FRED_ENV_KEY = "FRED_API_KEY"


def _discover_env_value(env_key: str) -> tuple[str, str]:
    """
    Probe all known key sources and return the first value found together with
    a human-readable description of where it came from.

    The search order is:

    1. Global Finance Toolkit env file (``get_global_env_path()``)
    2. Local ``.env`` in the current working directory
    3. *env_key* already set in the process environment (e.g. exported in the
       shell before running the wizard)

    Args:
        env_key (str): The .env / environment variable name to look up.

    Returns:
        tuple[str, str]: ``(value, source)`` where *source* is a short label
            such as ``"global config"`` or ``"local .env"``.  Both values are
            empty strings when nothing is found.
    """
    # 1. Global env file
    global_env = get_global_env_path()
    if global_env.exists():
        values = dotenv_values(global_env)
        value = values.get(env_key, "")
        if value:
            return value, "global config"

    # 2. Local .env in cwd
    local_env = pathlib.Path.cwd() / ".env"
    if local_env.exists():
        values = dotenv_values(local_env)
        value = values.get(env_key, "")
        if value:
            return value, f"local .env ({local_env})"

    # 3. Process environment (exported shell variable)
    value = os.environ.get(env_key, "")
    if value:
        return value, "shell environment"

    return "", ""


def discover_api_key() -> tuple[str, str]:
    """
    Probe all known key sources for the FinancialModelingPrep API key.

    Returns:
        tuple[str, str]: ``(api_key, source)``. Both values are empty strings
            when the key cannot be found.
    """
    return _discover_env_value(_ENV_KEY)


def discover_fred_api_key() -> tuple[str, str]:
    """
    Probe all known key sources for the (optional) FRED API key.

    FRED unlocks a handful of US-only economic indicators (e.g. nonfarm
    payrolls, the real TIPS yield curve) that have no FMP/OECD equivalent.
    It's free; register at https://fred.stlouisfed.org/docs/api/api_key.html.
    A missing FRED key never blocks setup — every other indicator works
    without it.

    Returns:
        tuple[str, str]: ``(fred_api_key, source)``. Both values are empty
            strings when the key cannot be found.
    """
    return _discover_env_value(_FRED_ENV_KEY)


def remove_global_env_key() -> None:
    """
    Delete the global Finance Toolkit .env file and remove the parent directory
    if it is left empty.
    """
    env_path = get_global_env_path()
    if not env_path.exists():
        info("Global env file not found — nothing to remove.")
        return
    env_path.unlink()
    ok(f"Deleted [dim cyan]{env_path}[/]")
    with suppress(OSError):
        env_path.parent.rmdir()  # only succeeds if the directory is now empty


def _read_json_config(config_path: pathlib.Path) -> dict | None:
    """
    Read a client's JSON config, or return None when it can't be read as JSON.

    Editors allow comments in some of these files (VS Code's mcp.json, Gemini's
    settings.json), which strict JSON can't read. Such a file is left untouched rather than
    rewritten with only the Finance Toolkit entry, which would lose everything else in it.

    Args:
        config_path: The config file.

    Returns:
        dict | None: The config ({} when the file doesn't exist), or None when it isn't
            valid JSON.
    """
    if not config_path.exists():
        return {}
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        err(
            f"[dim cyan]{config_path}[/] is not valid JSON ({error.msg}, line {error.lineno}), "
            "so it was left unchanged. Remove comments or trailing commas, or add the "
            "'finance-toolkit' entry by hand."
        )
        return None
    return existing if isinstance(existing, dict) else None


def _remove_entry_from_config(
    config_path: pathlib.Path,
    outer_key: str,
    entry_key: str = "finance-toolkit",
) -> None:
    """
    Remove *entry_key* from *outer_key* inside *config_path* (a JSON file).

    If the file does not exist or the entry is absent, a short notice is printed
    and the function returns without modifying anything.

    Args:
        config_path: Absolute path to the JSON config file.
        outer_key: Top-level JSON key that contains the server map
            (``"servers"`` for VS Code, ``"mcpServers"`` for others).
        entry_key: The server entry to remove. Defaults to ``"finance-toolkit"``.
    """
    if not config_path.exists():
        info(f"No config found at {config_path} — skipping.")
        return

    existing = _read_json_config(config_path)
    if existing is None:
        return

    if entry_key not in existing.get(outer_key, {}):
        info(f"No '{entry_key}' entry found in {config_path} — skipping.")
        return

    del existing[outer_key][entry_key]
    write_private_file(config_path, json.dumps(existing, indent=2) + "\n")
    ok(f"Removed [bold]'{entry_key}'[/] from [dim cyan]{config_path}[/]")


def remove_vscode_config(target_dir: pathlib.Path) -> None:
    """
    Remove the ``finance-toolkit`` entry from ``.vscode/mcp.json``.

    Args:
        target_dir: Directory that contains the ``.vscode`` subdirectory.
    """
    _remove_entry_from_config(
        target_dir / ".vscode" / "mcp.json",
        outer_key="servers",
    )


def remove_cursor_config(target_dir: pathlib.Path) -> None:
    """
    Remove the ``finance-toolkit`` entry from ``.cursor/mcp.json``.

    Args:
        target_dir: Directory that contains the ``.cursor`` subdirectory.
    """
    _remove_entry_from_config(
        target_dir / ".cursor" / "mcp.json",
        outer_key="mcpServers",
    )


def remove_claude_config() -> None:
    """Remove the ``finance-toolkit`` entry from the Claude Desktop config file."""
    _remove_entry_from_config(
        get_claude_config_path(),
        outer_key="mcpServers",
    )


def remove_all_configs(target_dir: pathlib.Path) -> None:
    """
    Remove the ``finance-toolkit`` entry from every known client config file and
    delete the global Finance Toolkit .env file.

    The function prints a summary of what exists before asking for confirmation
    so the user knows exactly what will be removed.

    Args:
        target_dir: Working directory used to resolve ``.vscode`` and ``.cursor``
            paths (typically ``pathlib.Path.cwd()``).
    """
    candidates = [
        ("Claude Desktop", get_claude_config_path(), "mcpServers"),
        ("Claude Code", get_claude_code_config_path(), "mcpServers"),
        ("VS Code", target_dir / ".vscode" / "mcp.json", "servers"),
        ("Cursor", target_dir / ".cursor" / "mcp.json", "mcpServers"),
        ("Gemini", get_gemini_config_path(), "mcpServers"),
        ("Devin Desktop (Windsurf)", get_devin_config_path(), "mcpServers"),
        ("GitHub Copilot (global)", get_copilot_config_path(), "mcpServers"),
    ]

    found: list[tuple[str, pathlib.Path, str]] = []
    for name, path, key in candidates:
        if not path.exists():
            continue
        existing = _read_json_config(path)
        if existing is not None and "finance-toolkit" in existing.get(key, {}):
            found.append((name, path, key))

    global_env = get_global_env_path()
    has_global_env = global_env.exists()

    if not found and not has_global_env:
        console.print()
        info("No Finance Toolkit configuration found — nothing to remove.")
        console.print()
        return

    summary = Text()
    summary.append("The following will be removed:\n\n", style="bold")
    for name, path, _ in found:
        summary.append("  ·  ", style="yellow")
        summary.append(f"{name}  ", style="bold")
        summary.append(f"{path}\n", style="dim cyan")
    if has_global_env:
        summary.append("  ·  ", style="yellow")
        summary.append("Global env file  ", style="bold")
        summary.append(f"{global_env}\n", style="dim cyan")

    console.print()
    console.print(
        Panel(
            summary,
            title="[bold yellow]Remove Configuration[/]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()
    if not Confirm.ask("  Proceed with removal?", default=False):
        info("Removal cancelled — nothing was changed.")
        console.print()
        return

    console.print()
    for _, path, key in found:
        _remove_entry_from_config(path, outer_key=key)
    if has_global_env:
        remove_global_env_key()

    console.print()
    ok("[bold]Removal complete.[/]  Restart your client(s) to apply.")
    console.print()


def _write_global_client_config(
    config_path: pathlib.Path, display_name: str, api_key: str, fred_api_key: str
) -> None:
    """
    Add the Finance Toolkit MCP server entry to a client's global ``mcpServers`` file.

    The file and its folder are created when missing. An existing ``finance-toolkit``
    entry is shown and only replaced after confirmation; all other entries are kept.
    When *api_key* is provided it is embedded directly as
    ``FINANCIAL_MODELING_PREP_API_KEY``; otherwise the entry uses
    ``FINANCETOOLKIT_ENV_FILE`` pointing at the global ``.env`` file.

    Args:
        config_path (pathlib.Path): The client's configuration file.
        display_name (str): The client's name in messages.
        api_key (str): FinancialModelingPrep API key. When non-empty the key is
            embedded directly in the config (takes priority over the env file).
        fred_api_key (str): Optional FRED API key. When non-empty the key is
            embedded directly in the config; unlocks a handful of US-only
            economic indicators and is free to obtain.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_json_config(config_path)
    if existing is None:
        return

    current_entry = existing.get("mcpServers", {}).get("finance-toolkit")
    if current_entry is not None:
        console.print()
        warn(
            f"Existing [bold]'finance-toolkit'[/] entry found in [dim cyan]{config_path}[/]"
        )
        console.print(f"  [dim]{json.dumps(mask_entry(current_entry), indent=4)}[/]")
        console.print()
        if not Confirm.ask("  Overwrite this entry?", default=False):
            info(f"Skipped — existing {display_name} config left unchanged.")
            return

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["finance-toolkit"] = _uvx_server_entry(
        api_key or None, fred_api_key or None
    )

    write_private_file(config_path, json.dumps(existing, indent=2) + "\n")
    ok(f"{display_name} config written to [dim cyan]{config_path}[/]")


def write_gemini_config(api_key: str, fred_api_key: str = "") -> None:
    """
    Add the Finance Toolkit MCP server entry to the Gemini CLI configuration file
    (``~/.gemini/settings.json``).

    Args:
        api_key (str): FinancialModelingPrep API key, embedded when non-empty.
        fred_api_key (str): Optional FRED API key, embedded when non-empty.
    """
    _write_global_client_config(
        get_gemini_config_path(), "Gemini", api_key, fred_api_key
    )


def remove_gemini_config() -> None:
    """Remove the ``finance-toolkit`` entry from the Gemini CLI config file."""
    _remove_entry_from_config(get_gemini_config_path(), outer_key="mcpServers")


def write_devin_config(api_key: str, fred_api_key: str = "") -> None:
    """
    Add the Finance Toolkit MCP server entry to the Devin Desktop (Windsurf) user
    configuration file (see ``get_devin_config_path``).

    Args:
        api_key (str): FinancialModelingPrep API key, embedded when non-empty.
        fred_api_key (str): Optional FRED API key, embedded when non-empty.
    """
    _write_global_client_config(
        get_devin_config_path(), "Devin Desktop (Windsurf)", api_key, fred_api_key
    )


def remove_devin_config() -> None:
    """Remove the ``finance-toolkit`` entry from the Devin Desktop (Windsurf) config file."""
    _remove_entry_from_config(get_devin_config_path(), outer_key="mcpServers")


def write_copilot_config(api_key: str, fred_api_key: str = "") -> None:
    """
    Add the Finance Toolkit MCP server entry to the global GitHub Copilot configuration
    file (``~/.copilot/mcp-config.json``, see ``get_copilot_config_path``), which applies
    to every workspace.

    Args:
        api_key (str): FinancialModelingPrep API key, embedded when non-empty.
        fred_api_key (str): Optional FRED API key, embedded when non-empty.
    """
    _write_global_client_config(
        get_copilot_config_path(), "GitHub Copilot (global)", api_key, fred_api_key
    )


def remove_copilot_config() -> None:
    """Remove the ``finance-toolkit`` entry from the global GitHub Copilot config file."""
    _remove_entry_from_config(get_copilot_config_path(), outer_key="mcpServers")


def write_client_config_uvx(
    client: str,
    target_dir: pathlib.Path,
    overwrite: bool = False,
    api_key: str = "",
    fred_api_key: str = "",
) -> None:
    """Write an MCP server config block that invokes the server via uvx.

    Used by the non-interactive ``--client`` CLI path.  The generated entry
    uses ``uvx --from financetoolkit[mcp] financetoolkit-mcp`` so it works
    without a pre-installed package.

    If the target config file does not exist it is created (along with any
    missing parent directories).  When *overwrite* is ``False`` and an
    existing ``finance-toolkit`` entry is found, the function exits with a
    warning without modifying anything.  If the config directory for a global
    client (e.g. Gemini, Devin Desktop) cannot be determined, the raw JSON block is
    printed to the terminal as a fallback.

    When *api_key* is supplied it is embedded directly in the config as
    ``FINANCIAL_MODELING_PREP_API_KEY``; otherwise the entry uses
    ``FINANCETOOLKIT_ENV_FILE`` pointing at the global ``.env`` file.

    Args:
        client: Canonical client name — one of ``"claude-desktop"``,
            ``"claude-code"``, ``"vscode"``, ``"cursor"``, ``"gemini"``,
            ``"devin"``, ``"copilot"``.
        target_dir: Workspace root / cwd used to resolve workspace-local paths
            (``.vscode/mcp.json``, ``.cursor/mcp.json``).
        overwrite: When ``True``, silently replace an existing entry.
        api_key: FinancialModelingPrep API key.  When non-empty the key is
            embedded directly in the config (takes priority over the env file).
        fred_api_key: Optional FRED API key. When non-empty the key is
            embedded directly in the config; unlocks a handful of US-only
            economic indicators and is free to obtain.
    """
    entry = _uvx_server_entry(api_key or None, fred_api_key or None)

    if client in ("vscode", "cursor"):
        if client == "vscode":
            config_path = target_dir / ".vscode" / "mcp.json"
            outer_key = "servers"
            entry = {"type": "stdio", **entry}
        else:
            config_path = target_dir / ".cursor" / "mcp.json"
            outer_key = "mcpServers"
        config_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        cfg_fn, outer_key, _ = _CLIENT_CONFIG[client]
        config_path = cfg_fn()
        if not config_path.parent.exists():
            # Parent directory missing means the client is not installed; print the block.
            console.print()
            warn(
                f"Config directory not found for [bold]{_CLIENT_CONFIG[client][2]}[/].  "
                "Printing the block to paste manually:"
            )
            console.print()
            block = {"mcpServers": {"finance-toolkit": mask_entry(entry)}}
            console.print(json.dumps(block, indent=2))
            console.print()
            info("API keys are masked: put your own keys in their place.")
            return
        config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_json_config(config_path)
    if existing is None:
        return

    current_entry = existing.get(outer_key, {}).get("finance-toolkit")
    if current_entry is not None and not overwrite:
        warn(
            f"Existing [bold]'finance-toolkit'[/] entry found in "
            f"[dim cyan]{config_path}[/]  — use --overwrite to replace."
        )
        return

    existing.setdefault(outer_key, {})
    existing[outer_key]["finance-toolkit"] = entry
    write_private_file(config_path, json.dumps(existing, indent=2) + "\n")
    display = _CLIENT_CONFIG.get(client, (None, None, client))[2]
    ok(f"{display} config written to [dim cyan]{config_path}[/]")


def write_claude_code_config(api_key: str, fred_api_key: str = "") -> None:
    """
    Patch the Claude Code user configuration file (``~/.claude.json``) to add
    the Finance Toolkit MCP server entry.

    When *api_key* is provided it is embedded directly as
    ``FINANCIAL_MODELING_PREP_API_KEY``; otherwise the entry uses
    ``FINANCETOOLKIT_ENV_FILE`` pointing at the global ``.env`` file so that
    the key is loaded at runtime from a file outside version control.

    If the configuration file does not exist it is created from scratch. If it
    already exists and the ``finance-toolkit`` entry is present, the user is
    shown the existing entry and asked to confirm before overwriting. All other
    existing entries are always preserved.

    Args:
        api_key (str): FinancialModelingPrep API key.  When non-empty the key
            is embedded directly in the config (takes priority over the env
            file).
        fred_api_key (str): Optional FRED API key. When non-empty the key is
            embedded directly in the config; unlocks a handful of US-only
            economic indicators and is free to obtain.
    """
    config_path = get_claude_code_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_json_config(config_path)
    if existing is None:
        return

    current_entry = existing.get("mcpServers", {}).get("finance-toolkit")
    if current_entry is not None:
        console.print()
        warn(
            f"Existing [bold]'finance-toolkit'[/] entry found in [dim cyan]{config_path}[/]"
        )
        console.print(f"  [dim]{json.dumps(mask_entry(current_entry), indent=4)}[/]")
        console.print()
        if not Confirm.ask("  Overwrite this entry?", default=False):
            info("Skipped — existing Claude Code config left unchanged.")
            return

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["finance-toolkit"] = _uvx_server_entry(
        api_key or None, fred_api_key or None
    )

    write_private_file(config_path, json.dumps(existing, indent=2) + "\n")
    ok(f"Claude Code config written to [dim cyan]{config_path}[/]")


def remove_claude_code_config() -> None:
    """Remove the ``finance-toolkit`` entry from the Claude Code config file."""
    _remove_entry_from_config(
        get_claude_code_config_path(),
        outer_key="mcpServers",
    )


def write_vscode_config(
    api_key: str, target_dir: pathlib.Path, fred_api_key: str = ""
) -> None:
    """
    Write (or merge) a .vscode/mcp.json file in the given directory.

    When *api_key* is provided it is embedded directly as
    ``FINANCIAL_MODELING_PREP_API_KEY``; otherwise the entry uses
    ``FINANCETOOLKIT_ENV_FILE`` pointing at the global ``.env`` file so that
    the key is loaded at runtime from a file outside version control.

    If the file already exists and the ``finance-toolkit`` entry is present, the
    user is shown the existing entry and asked to confirm before overwriting. All
    other existing server entries are always preserved.

    Args:
        api_key (str): FinancialModelingPrep API key.  When non-empty the key
            is embedded directly in the config (takes priority over the env
            file).
        target_dir (pathlib.Path): Directory in which .vscode/mcp.json is created.
        fred_api_key (str): Optional FRED API key. When non-empty the key is
            embedded directly in the config; unlocks a handful of US-only
            economic indicators and is free to obtain.
    """
    vscode_dir = target_dir / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    config_path = vscode_dir / "mcp.json"

    existing = _read_json_config(config_path)
    if existing is None:
        return

    current_entry = existing.get("servers", {}).get("finance-toolkit")
    if current_entry is not None:
        console.print()
        warn(
            f"Existing [bold]'finance-toolkit'[/] entry found in [dim cyan]{config_path}[/]"
        )
        console.print(f"  [dim]{json.dumps(mask_entry(current_entry), indent=4)}[/]")
        console.print()
        if not Confirm.ask("  Overwrite this entry?", default=False):
            info("Skipped — existing VS Code config left unchanged.")
            return

    existing.setdefault("servers", {})
    # VS Code's mcp.json names the transport of each server.
    existing["servers"]["finance-toolkit"] = {
        "type": "stdio",
        **_uvx_server_entry(api_key or None, fred_api_key or None),
    }

    write_private_file(config_path, json.dumps(existing, indent=2) + "\n")
    ok(f"VS Code config written to [dim cyan]{config_path}[/]")


def write_cursor_config(
    api_key: str, target_dir: pathlib.Path, fred_api_key: str = ""
) -> None:
    """
    Write (or merge) a .cursor/mcp.json file in the given directory.

    When *api_key* is provided it is embedded directly as
    ``FINANCIAL_MODELING_PREP_API_KEY``; otherwise the entry uses
    ``FINANCETOOLKIT_ENV_FILE`` pointing at the global ``.env`` file so that
    the key is loaded at runtime from a file outside version control.

    If the file already exists and the ``finance-toolkit`` entry is present, the
    user is shown the existing entry and asked to confirm before overwriting. All
    other existing server entries are always preserved.

    Args:
        api_key (str): FinancialModelingPrep API key.  When non-empty the key
            is embedded directly in the config (takes priority over the env
            file).
        target_dir (pathlib.Path): Directory in which .cursor/mcp.json is created.
        fred_api_key (str): Optional FRED API key. When non-empty the key is
            embedded directly in the config; unlocks a handful of US-only
            economic indicators and is free to obtain.
    """
    cursor_dir = target_dir / ".cursor"
    cursor_dir.mkdir(exist_ok=True)
    config_path = cursor_dir / "mcp.json"

    existing = _read_json_config(config_path)
    if existing is None:
        return

    current_entry = existing.get("mcpServers", {}).get("finance-toolkit")
    if current_entry is not None:
        console.print()
        warn(
            f"Existing [bold]'finance-toolkit'[/] entry found in [dim cyan]{config_path}[/]"
        )
        console.print(f"  [dim]{json.dumps(mask_entry(current_entry), indent=4)}[/]")
        console.print()
        if not Confirm.ask("  Overwrite this entry?", default=False):
            info("Skipped — existing Cursor config left unchanged.")
            return

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["finance-toolkit"] = _uvx_server_entry(
        api_key or None, fred_api_key or None
    )

    write_private_file(config_path, json.dumps(existing, indent=2) + "\n")
    ok(f"Cursor config written to [dim cyan]{config_path}[/]")


def write_claude_config(api_key: str, fred_api_key: str = "") -> None:
    """
    Patch the Claude Desktop configuration file to add the Finance Toolkit server.

    For Claude Desktop, the API key is embedded directly in the MCP server
    environment as ``FINANCIAL_MODELING_PREP_API_KEY`` so Claude does not need
    to point at a separate .env file.

    If the configuration file does not exist it is created from scratch, including
    any missing parent directories. If it already exists and the ``finance-toolkit``
    entry is present, the user is shown the existing entry and asked to confirm
    before overwriting. All other existing server entries are always preserved.

    Args:
        api_key (str): FinancialModelingPrep API key.  When non-empty the key
            is embedded directly in the config; otherwise the entry uses
            ``FINANCETOOLKIT_ENV_FILE`` pointing at the global ``.env`` file.
        fred_api_key (str): Optional FRED API key. When non-empty the key is
            embedded directly in the config; unlocks a handful of US-only
            economic indicators and is free to obtain.
    """
    config_path = get_claude_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_json_config(config_path)
    if existing is None:
        return

    current_entry = existing.get("mcpServers", {}).get("finance-toolkit")
    if current_entry is not None:
        console.print()
        warn(
            f"Existing [bold]'finance-toolkit'[/] entry found in [dim cyan]{config_path}[/]"
        )
        console.print(f"  [dim]{json.dumps(mask_entry(current_entry), indent=4)}[/]")
        console.print()
        if not Confirm.ask("  Overwrite this entry?", default=False):
            info("Skipped — existing Claude Desktop config left unchanged.")
            return

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["finance-toolkit"] = _uvx_server_entry(
        api_key, fred_api_key or None
    )

    write_private_file(config_path, json.dumps(existing, indent=2) + "\n")
    ok(f"Claude Desktop config written to [dim cyan]{config_path}[/]")
