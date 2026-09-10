"""Version Alignment Tests"""

# The package version is declared in three places that all reach users through a
# different channel: pyproject.toml (PyPI), the MCPB manifest (the desktop extension
# bundle) and server.json (the MCP registry). They have drifted apart before, which
# is invisible locally and confusing publicly, so this asserts they move together.

import json
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_versions_are_aligned():
    pyproject_version = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())[
        "project"
    ]["version"]

    manifest_version = json.loads(
        (
            PROJECT_ROOT / "financetoolkit" / "mcp_server" / "mcpb" / "manifest.json"
        ).read_text()
    )["version"]

    server_metadata = json.loads((PROJECT_ROOT / "server.json").read_text())
    server_versions = [server_metadata["version"]] + [
        package["version"] for package in server_metadata["packages"]
    ]

    assert (
        manifest_version == pyproject_version
    ), f"manifest.json ({manifest_version}) does not match pyproject.toml ({pyproject_version})"
    for server_version in server_versions:
        assert (
            server_version == pyproject_version
        ), f"server.json ({server_version}) does not match pyproject.toml ({pyproject_version})"
