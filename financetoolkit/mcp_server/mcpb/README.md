# MCP Bundle

This directory contains the source for `financetoolkit.mcpb`, an [MCP Bundle](https://github.com/modelcontextprotocol/mcpb) that installs the Finance Toolkit MCP Server into any compatible AI client (Claude Desktop, Claude Code, and others) with a single click — no Python environment or manual configuration required.

## Installing

1. Download the latest `financetoolkit.mcpb` from the releases page [here](https://github.com/JerBouma/FinanceToolkit/releases/latest/download/financetoolkit.mcpb)
2. Double-click `financetoolkit.mcpb` which will open Claude Desktop, this will open a prompt whether you want to install the bundle, click "Install".
3. A prompt will appear asking you to confirm the installation, click "Install" again.
4. The bundle will ask you to provide a Financial Modeling Prep API key (obtain one [here](https://www.jeroenbouma.com/fmp)) and click "Save".
5. Enable the bundle by toggling the switch which says "Disabled" to "Enabled".
6. Restart Claude Desktop and the Finance Toolkit MCP Server will be available for use in your conversations.

## Releases

Every release builds `financetoolkit.mcpb` from this directory and attaches it to the GitHub release (see `.github/workflows/release.yml`). The bundle holds no code of its own: it installs the `financetoolkit[mcp]` version pinned in `pyproject.toml` from PyPI. `manifest.json` lists the server's tools with the descriptions from `financetoolkit/mcp_server/config.yaml`, and the tests check that both stay the same.

## Building and Testing Locally

1. Install the [MCPB CLI](https://github.com/modelcontextprotocol/mcpb) with `npm install -g @anthropic-ai/mcpb`.
2. Build the bundle from the repository root with `mcpb pack financetoolkit/mcp_server/mcpb dist/financetoolkit.mcpb`.
3. Double-click `dist/financetoolkit.mcpb` to install it into Claude Desktop.

The bundle installs the pinned version from PyPI, so it runs the latest release rather than local changes. To test local changes, run the server from your checkout with `uv run financetoolkit-mcp` instead.
