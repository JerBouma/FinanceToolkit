# This file is meant for deploying the MCP server using Docker Compose. This, together with
# the Docker Compose file, makes it possible to create a localhost deployment of the MCP server,
# which can be used for testing and development purposes or for running on a server.
#
# The server is currently already live at financetoolkit.jeroenbouma.com/mcp
#
# Both images are pinned by digest (the tag is kept for reading); Dependabot bumps the digests.

FROM ghcr.io/astral-sh/uv:0.13.10@sha256:a3ea1357342c1a4bf9a0c53a3deb1b55379aefa037338d8113d5771efa01774f AS uv

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

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

# One thread per linear algebra library. The server answers requests in parallel, and the
# calculations work on small matrices (a few thousand rows by a few columns), where a pool of
# threads per operation spends more time on starting and syncing them than on the work: an SVD
# of 2,520 by 30 takes 12 ms with one thread against 685 ms with a pool, a unit root test 0.26
# against 6.56 s, an Engle-Granger test 0.68 against 16.7 s, a GARCH fit 3.06 against 6.07 s,
# a Fama-MacBeth regression 7.7 against 14.9 s and a GLS regression 1.2 against 9.3 s.
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

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
