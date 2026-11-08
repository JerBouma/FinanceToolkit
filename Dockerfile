# This file is meant for deploying the MCP server using Docker Compose. This, together with
# the Docker Compose file, makes it possible to create a localhost deployment of the MCP server,
# which can be used for testing and development purposes or for running on a server.
#
# The server is currently already live at financetoolkit.jeroenbouma.com/mcp
#
# Both images are pinned by digest (the tag is kept for reading); Dependabot bumps the digests.

FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54

COPY --from=uv /uv /usr/local/bin/uv

# The server runs as this user. The code and the virtual environment stay owned by root, so the
# server can't change them; it only writes to its home directory (the cache, usage counts and the
# yfinance cache).
RUN useradd --system --uid 10001 --user-group --create-home --home-dir /home/financetoolkit \
    --shell /usr/sbin/nologin financetoolkit

# The image's own Python is used; uv keeps no download cache in the image. The packages are compiled
# to bytecode here because the server user can't write __pycache__ (a start without it is several times slower).
ENV UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Dependencies first, so a change to the code alone doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra mcp --no-dev --no-install-project

COPY financetoolkit/ financetoolkit/
RUN uv sync --frozen --extra mcp --no-dev \
    && .venv/bin/python -m compileall -q financetoolkit

# The cache database, mcp_stats.json and a global .env live in $XDG_CONFIG_HOME/financetoolkit,
# which is the volume in docker-compose.yml. The directory is created here so that a new volume
# takes over its owner.
ENV PATH="/app/.venv/bin:$PATH" \
    XDG_CONFIG_HOME=/home/financetoolkit/.config
RUN mkdir -p /home/financetoolkit/.config/financetoolkit \
    && chown -R financetoolkit:financetoolkit /home/financetoolkit/.config

ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

USER financetoolkit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# The virtual environment is complete at build time, so the server starts without `uv run`, which
# would sync (and download) packages on every start.
CMD ["financetoolkit-mcp"]
