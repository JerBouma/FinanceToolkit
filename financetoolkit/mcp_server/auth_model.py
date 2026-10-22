"""Per-request Financial Modeling Prep and FRED API key resolution and OAuth 2.1 authentication for hosted deployments."""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import pathlib
import secrets
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import fastmcp.server.dependencies as deps
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from financetoolkit.mcp_server.setup_model import (
    get_global_env_path,
)
from financetoolkit.mcp_server.token_store_model import (
    ACCESS_TOKEN_LIFETIME,
    TokenStore,
    is_issued_token,
)
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

_JWT_PART_COUNT = 3

# A hosted server signs and encrypts with FT_MCP_SECRET_KEY; anything shorter is guessable.
_MINIMUM_SECRET_LENGTH = 32

# Schemes a client may not ask to be redirected to.
_UNSAFE_REDIRECT_SCHEMES = {"javascript", "data", "vbscript", "file"}

# Checked in order. FastMCP forwards `x-*`; `authorization` must be opted in.
_HEADER_NAMES: tuple[str, ...] = (
    "x-fmp-api-key",
    "x-financial-modeling-prep-api-key",
)

# Query-string parameter names checked in order.
_QUERY_NAMES: tuple[str, ...] = (
    "fmp_api_key",
    "api_key",
    "fmp_key",
)

# Checked in order for the optional FRED key used by Economics/FixedIncome.
_FRED_HEADER_NAMES: tuple[str, ...] = ("x-fred-api-key",)

# Query-string parameter names checked in order for the FRED API key.
_FRED_QUERY_NAMES: tuple[str, ...] = (
    "fred_api_key",
    "fred_key",
)

# The page a browser gets at the server's address and at /mcp: how to connect, with a link to
# the full documentation. MCP clients never see it; they ask for JSON or an event stream.
ROOT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Finance Toolkit MCP</title>
    <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
    <link rel="icon" type="image/png" sizes="512x512" href="/oauth/icon">
    <link rel="apple-touch-icon" href="/apple-touch-icon.png">
    <style>
        body { margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
               background: linear-gradient(135deg, #0f172a 0%, #020617 100%); color: #f8fafc;
               font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
        main { max-width: 40rem; padding: 2.5rem 1.5rem; }
        .head { display: flex; align-items: center; gap: 1rem; }
        .head img { width: 3.5rem; height: 3.5rem; border-radius: 0.9rem; }
        h1 { font-size: 2rem; margin: 0; }
        h1 span { color: #38bdf8; }
        p { color: #cbd5e1; line-height: 1.6; }
        ol { color: #cbd5e1; line-height: 1.7; padding-left: 1.3rem; }
        li { margin-bottom: 0.6rem; }
        strong { color: #f8fafc; }
        .url { display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0; padding: 0.7rem 0.9rem; border-radius: 0.6rem;
               background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.45);
               font-family: ui-monospace, "JetBrains Mono", monospace; font-size: 0.95rem; color: #f8fafc; word-break: break-all; }
        code { font-family: ui-monospace, "JetBrains Mono", monospace; color: #e2e8f0; }
        .button { display: inline-block; margin-top: 0.8rem; padding: 0.75rem 1.4rem; border-radius: 0.6rem; font-weight: 600;
                  color: #fff; text-decoration: none; background: linear-gradient(135deg, #0284c7, #2563eb); }
        .muted { color: #94a3b8; font-size: 0.9rem; }
    </style>
</head>
<body>
<main>
    <div class="head"><img src="/oauth/icon" alt=""><h1>Finance Toolkit <span>MCP</span></h1></div>
    <p>This address is an MCP server: it gives AI assistants such as Claude, Copilot and Cursor the
    Finance Toolkit's 500+ transparent financial metrics. It is meant to be added to an MCP client
    rather than opened in a browser.</p>
    <ol>
        <li><strong>Add the server</strong> to your MCP client as a remote (HTTP) server, with this URL:
            <div class="url">https://financetoolkit.jeroenbouma.com/mcp</div>
            In Claude: Settings &rarr; Connectors &rarr; Add custom connector.</li>
        <li><strong>Sign in once.</strong> On first use, a page asks for a Financial Modeling Prep API key
            (and optionally a FRED key); your client remembers it.</li>
        <li><strong>Ask away,</strong> for example &ldquo;Compare Apple&rsquo;s and Microsoft&rsquo;s return on equity&rdquo;.</li>
    </ol>
    <p class="muted">Prefer to run it locally? <code>uvx --from "financetoolkit[mcp]" financetoolkit-mcp-setup</code>
    sets up your client and API key.</p>
    <a class="button" href="https://www.jeroenbouma.com/projects/financetoolkit/mcp">Setup guide and documentation</a>
</main>
</body>
</html>
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Finance Toolkit &middot; Authorize</title>
    <link rel="icon" type="image/png" href="/oauth/icon">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        :root {{
            --accent: #38bdf8;
            --accent-dark: #0284c7;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --card-bg: rgba(30, 41, 59, 0.6);
            --card-border: rgba(255, 255, 255, 0.1);
            --input-bg: rgba(15, 23, 42, 0.5);
            --input-border: rgba(255, 255, 255, 0.1);
            --btn-gradient: linear-gradient(135deg, #0284c7, #2563eb);
            --btn-gradient-hover: linear-gradient(135deg, #0369a1, #1d4ed8);
            --btn-shadow: rgba(37, 99, 235, 0.4);
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px 16px;
            position: relative;
            overflow-x: hidden;
        }}

        .glow {{
            position: fixed;
            border-radius: 50%;
            filter: blur(70px);
            pointer-events: none;
            z-index: 0;
        }}

        .glow-1 {{
            width: 500px;
            height: 500px;
            top: -120px;
            left: -100px;
            background: radial-gradient(circle, rgba(2, 132, 199, 0.18) 0%, transparent 70%);
        }}

        .glow-2 {{
            width: 420px;
            height: 420px;
            bottom: -100px;
            right: -80px;
            background: radial-gradient(circle, rgba(37, 99, 235, 0.15) 0%, transparent 70%);
        }}

        .glow-3 {{
            width: 360px;
            height: 360px;
            top: 40%;
            right: 10%;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.1) 0%, transparent 70%);
        }}

        .content {{
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }}

        .wordmark {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 28px;
        }}

        .wordmark-text {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 15px;
            color: var(--text-primary);
            letter-spacing: -0.3px;
        }}

        .wordmark-text span {{
            color: var(--text-muted);
            font-weight: 400;
            margin-left: 8px;
        }}

        .card {{
            position: relative;
            width: 100%;
            max-width: 420px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 36px 32px 28px;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }}

        .card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 15%;
            right: 15%;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent), transparent);
            opacity: 0.55;
            pointer-events: none;
            border-radius: 0 0 2px 2px;
        }}

        .card-header {{
            text-align: center;
            margin-bottom: 28px;
        }}

        .client-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 100px;
            padding: 4px 12px 4px 8px;
            margin-bottom: 16px;
            font-size: 12px;
            color: var(--accent);
            font-weight: 500;
        }}

        .client-badge-dot {{
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            flex-shrink: 0;
        }}

        h1 {{
            font-weight: 700;
            font-size: 22px;
            letter-spacing: -0.5px;
            color: var(--text-primary);
            margin-bottom: 8px;
        }}

        .subtitle {{
            font-size: 13.5px;
            color: var(--text-secondary);
            line-height: 1.55;
        }}

        .divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.15), transparent);
            margin: 24px 0;
        }}

        .form-group {{
            margin-bottom: 20px;
        }}

        label {{
            display: block;
            font-size: 11.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-secondary);
            margin-bottom: 7px;
        }}

        .input-wrapper {{
            position: relative;
        }}

        .key-input {{
            width: 100%;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 8px;
            padding: 12px 44px 12px 14px;
            color: var(--text-primary);
            font-size: 14px;
            font-family: 'Inter', monospace;
            letter-spacing: 0.5px;
            transition: border-color 0.18s, box-shadow 0.18s;
            -webkit-appearance: none;
        }}

        .key-input::placeholder {{
            color: var(--text-muted);
            letter-spacing: 0;
        }}

        .key-input:focus {{
            outline: none;
            border-color: rgba(56, 189, 248, 0.5);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.08);
        }}

        .toggle-btn {{
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: color 0.18s;
            border-radius: 4px;
        }}

        .toggle-btn:hover {{
            color: var(--accent);
        }}

        .toggle-btn .icon-hidden {{ display: none; }}
        .toggle-btn.revealed .icon-visible {{ display: none; }}
        .toggle-btn.revealed .icon-hidden {{ display: block; }}

        .trust-note {{
            display: flex;
            align-items: center;
            gap: 7px;
            background: rgba(56, 189, 248, 0.05);
            border: 1px solid rgba(56, 189, 248, 0.12);
            border-radius: 8px;
            padding: 9px 12px;
            margin-bottom: 20px;
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.4;
        }}

        .trust-note svg {{
            color: var(--accent);
            flex-shrink: 0;
        }}

        .btn-authorize {{
            width: 100%;
            background: var(--btn-gradient);
            border: none;
            border-radius: 8px;
            padding: 13px 20px;
            color: #fff;
            font-size: 15px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 6px -1px var(--btn-shadow), 0 2px 4px -1px var(--btn-shadow);
            letter-spacing: -0.1px;
        }}

        .btn-authorize:hover {{
            background: var(--btn-gradient-hover);
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px var(--btn-shadow);
        }}

        .btn-authorize:active {{
            transform: translateY(0);
        }}

        .card-footer {{
            margin-top: 20px;
            padding-top: 18px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .footer-text {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        .footer-link {{
            font-size: 12px;
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.18s;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        .footer-link:hover {{
            color: var(--accent);
        }}

        .page-footer {{
            margin-top: 24px;
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
        }}

        .page-footer a {{
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.18s;
        }}

        .page-footer a:hover {{
            color: var(--accent);
        }}
    </style>
</head>
<body>
    <div class="glow glow-1"></div>
    <div class="glow glow-2"></div>
    <div class="glow glow-3"></div>

    <div class="content">
    <div class="wordmark">
        <div class="wordmark-text">Finance Toolkit<span>&middot; MCP Server</span></div>
    </div>

    <div class="card">
        <div class="card-header">
            <div class="client-badge">
                <span class="client-badge-dot"></span>
                {client_name}
            </div>
            <h1>Authorize Access</h1>
            <p class="subtitle">
                Enter your Financial Modeling Prep API key to grant this client access to financial data tools.
                Optionally add a FRED API key to also enable US labor market, macroeconomic and interest rate
                indicators.
            </p>
        </div>

        <div class="divider"></div>

        <form action="/oauth/authorize" method="POST">
            <input type="hidden" name="client_id" value="{client_id}">
            <input type="hidden" name="redirect_uri" value="{redirect_uri}">
            <input type="hidden" name="code_challenge" value="{code_challenge}">
            <input type="hidden" name="code_challenge_method" value="{code_challenge_method}">
            <input type="hidden" name="state" value="{state}">

            <div class="form-group">
                <label for="fmp_api_key">FMP API Key</label>
                <div class="input-wrapper">
                    <input
                        class="key-input"
                        type="password"
                        id="fmp_api_key"
                        name="fmp_api_key"
                        placeholder="Enter your FMP API key"
                        required
                        autocomplete="current-password"
                        spellcheck="false"
                    >
                    <button type="button" class="toggle-btn" id="toggleBtn" aria-label="Toggle key visibility">
                        <svg class="icon-visible" xmlns="http://www.w3.org/2000/svg"
                            width="16" height="16" viewBox="0 0 24 24"
                            fill="none" stroke="currentColor" stroke-width="2"
                            stroke-linecap="round" stroke-linejoin="round">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                        <svg class="icon-hidden" xmlns="http://www.w3.org/2000/svg"
                            width="16" height="16" viewBox="0 0 24 24"
                            fill="none" stroke="currentColor" stroke-width="2"
                            stroke-linecap="round" stroke-linejoin="round">
                            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8
a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8
a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                            <line x1="1" y1="1" x2="23" y2="23"></line>
                        </svg>
                    </button>
                </div>
            </div>

            <div class="form-group">
                <label for="fred_api_key">FRED API Key <span style="text-transform:none;font-weight:400;color:var(--text-muted);">(optional)</span></label>
                <div class="input-wrapper">
                    <input
                        class="key-input"
                        type="password"
                        id="fred_api_key"
                        name="fred_api_key"
                        placeholder="Enter your FRED API key (optional)"
                        autocomplete="current-password"
                        spellcheck="false"
                    >
                    <button type="button" class="toggle-btn" id="toggleBtnFred" aria-label="Toggle key visibility">
                        <svg class="icon-visible" xmlns="http://www.w3.org/2000/svg"
                            width="16" height="16" viewBox="0 0 24 24"
                            fill="none" stroke="currentColor" stroke-width="2"
                            stroke-linecap="round" stroke-linejoin="round">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                        <svg class="icon-hidden" xmlns="http://www.w3.org/2000/svg"
                            width="16" height="16" viewBox="0 0 24 24"
                            fill="none" stroke="currentColor" stroke-width="2"
                            stroke-linecap="round" stroke-linejoin="round">
                            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8
a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8
a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                            <line x1="1" y1="1" x2="23" y2="23"></line>
                        </svg>
                    </button>
                </div>
            </div>

            <div class="trust-note">
                <svg xmlns="http://www.w3.org/2000/svg"
                    width="14" height="14" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor" stroke-width="2"
                    stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
                Your keys are embedded in a signed session token and are never stored on the server.
            </div>

            <button type="submit" class="btn-authorize">Authorize Access</button>
        </form>

        <div class="card-footer">
            <span class="footer-text">No FMP key yet?</span>
            <a class="footer-link" href="https://www.jeroenbouma.com/fmp" target="_blank" rel="noopener noreferrer">
                Get one with 15% discount
                <svg xmlns="http://www.w3.org/2000/svg"
                    width="11" height="11" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor" stroke-width="2.5"
                    stroke-linecap="round" stroke-linejoin="round">
                    <line x1="7" y1="17" x2="17" y2="7"></line>
                    <polyline points="7 7 17 7 17 17"></polyline>
                </svg>
            </a>
        </div>
        <div class="card-footer">
            <span class="footer-text">No FRED key yet?</span>
            <a class="footer-link" href="https://fred.stlouisfed.org/docs/api/api_key.html" target="_blank" rel="noopener noreferrer">
                Get one for free
                <svg xmlns="http://www.w3.org/2000/svg"
                    width="11" height="11" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor" stroke-width="2.5"
                    stroke-linecap="round" stroke-linejoin="round">
                    <line x1="7" y1="17" x2="17" y2="7"></line>
                    <polyline points="7 7 17 7 17 17"></polyline>
                </svg>
            </a>
        </div>
    </div>

    <p class="page-footer">
        <a href="https://www.jeroenbouma.com/projects/financetoolkit"
            target="_blank" rel="noopener noreferrer">Finance Toolkit</a>
        &nbsp;&middot;&nbsp;
        <a href="https://github.com/JerBouma/FinanceToolkit" target="_blank" rel="noopener noreferrer">GitHub</a>
        &nbsp;&middot;&nbsp;
        <a href="https://www.linkedin.com/in/boumajeroen/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
    </p>
    </div>

    <script>
        (function () {{
            function wireToggle(buttonId, inputId) {{
                var btn = document.getElementById(buttonId);
                var inp = document.getElementById(inputId);
                btn.addEventListener('click', function () {{
                    var isHidden = inp.type === 'password';
                    inp.type = isHidden ? 'text' : 'password';
                    btn.classList.toggle('revealed', isHidden);
                }});
            }}
            wireToggle('toggleBtn', 'fmp_api_key');
            wireToggle('toggleBtnFred', 'fred_api_key');
        }})();
    </script>
</body>
</html>
"""


def get_secret_key() -> bytes:
    """
    The server secret: FT_MCP_SECRET_KEY when set, otherwise a random key generated once and
    kept in the global configuration directory.

    It derives the key that encrypts the stored API keys and pseudonymises the usage
    statistics. There is no built-in fallback: a secret that cannot be stored stops the
    server instead of leaving it with a value anyone could read in the source.

    Returns:
        bytes: The secret.

    Raises:
        RuntimeError: When FT_MCP_SECRET_KEY is not set and the key file cannot be written.
    """
    env_key = os.environ.get("FT_MCP_SECRET_KEY")
    if env_key:
        return env_key.encode("utf-8")

    config_dir = get_global_env_path().parent
    secret_file = config_dir / ".mcp_secret"
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        if secret_file.exists():
            return secret_file.read_bytes()
        key = secrets.token_bytes(32)
        secret_file.write_bytes(key)
        secret_file.chmod(0o600)
        return key
    except OSError as error:
        raise RuntimeError(
            f"Cannot store the MCP server secret in {secret_file}. Set FT_MCP_SECRET_KEY "
            "to a random value of at least 32 characters instead."
        ) from error


def require_hosted_secret() -> None:
    """
    Refuse to start a hosted server without FT_MCP_SECRET_KEY.

    A hosted server needs a secret that is set deliberately and survives a rebuild of its
    container; a generated key file would sign everyone out when it is lost.

    Raises:
        SystemExit: When FT_MCP_SECRET_KEY is missing or shorter than 32 characters.
    """
    if len(os.environ.get("FT_MCP_SECRET_KEY", "")) < _MINIMUM_SECRET_LENGTH:
        raise SystemExit(
            "A hosted MCP server (sse or streamable-http) needs FT_MCP_SECRET_KEY: a random "
            'value of at least 32 characters, e.g. from `python -c "import secrets; '
            'print(secrets.token_urlsafe(48))"`.'
        )


_token_store: TokenStore | None = None


def get_token_store() -> TokenStore:
    """
    The store of OAuth grants, created on first use.

    It lives in the global configuration directory as mcp_auth.db, or at FT_MCP_AUTH_DB.

    Returns:
        TokenStore: The store.
    """
    global _token_store  # noqa: PLW0603
    path = os.environ.get("FT_MCP_AUTH_DB") or (
        get_global_env_path().parent / "mcp_auth.db"
    )
    if _token_store is None or str(_token_store.path) != str(path):
        _token_store = TokenStore(path, get_secret_key())
    return _token_store


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to a URL-safe base64 string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    """Decode a URL-safe base64 string without padding."""
    padding = "=" * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)


def sign_jwt(payload: dict[str, Any], expires_in: int = 2592000) -> str:
    """
    Create a signed JWT containing payload, expiring in expires_in seconds.

    The OAuth flow issues opaque tokens, not JWTs, because a JWT payload is readable by
    anyone holding it. This only reads legacy JWT access tokens, see `_legacy_keys`.
    """
    header = {"alg": "HS256", "typ": "JWT"}

    payload_copy = payload.copy()
    payload_copy["exp"] = int(time.time()) + expires_in

    header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(payload_copy).encode("utf-8"))

    message = f"{header_b64}.{payload_b64}".encode()
    secret = get_secret_key()
    signature = hmac.new(secret, message, hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_jwt(token: str) -> dict[str, Any] | None:
    """Verify a JWT signature and expiration. Returns payload if valid, else None."""
    try:
        parts = token.split(".")
        if len(parts) != _JWT_PART_COUNT:
            return None

        header_b64, payload_b64, signature_b64 = parts

        message = f"{header_b64}.{payload_b64}".encode()
        secret = get_secret_key()
        expected_signature = hmac.new(secret, message, hashlib.sha256).digest()
        expected_signature_b64 = _base64url_encode(expected_signature)

        if not hmac.compare_digest(signature_b64, expected_signature_b64):
            return None

        payload = json.loads(_base64url_decode(payload_b64).decode("utf-8"))

        if "exp" in payload and int(payload["exp"]) < time.time():
            return None

        return payload
    except Exception:
        return None


def _legacy_keys(token: str) -> dict[str, Any] | None:
    """
    The keys inside a legacy JWT access token, which carries them in its payload.

    Those tokens carry the API keys in readable form, so they are only accepted while
    FT_MCP_ACCEPT_LEGACY_TOKENS is set; otherwise their holders sign in again.
    """
    if os.environ.get("FT_MCP_ACCEPT_LEGACY_TOKENS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None
    return verify_jwt(token)


def _looks_like_token(value: str) -> bool:
    """Whether a value is a token (issued now or before) rather than an API key."""
    return is_issued_token(value) or value.count(".") == _JWT_PART_COUNT - 1


def _key_from_credential(value: str, field: str) -> str:
    """
    The API key behind a credential: an access token of this server, a legacy JWT or the
    API key itself.

    An access token that is invalid or expired gives "", so the request is answered with a
    401 and the client refreshes or signs in again, instead of the token being passed on to
    the data provider as if it were an API key.

    Args:
        value (str): The credential as sent.
        field (str): "fmp_api_key" or "fred_api_key".

    Returns:
        str: The API key, or "" when the token is not valid.
    """
    value = value.strip()
    if not value:
        return ""
    if is_issued_token(value):
        keys = get_token_store().keys_for_access_token(value)
        return str(keys.get(field) or "") if keys else ""
    if _looks_like_token(value):
        payload = _legacy_keys(value)
        return str(payload.get(field) or "") if payload else ""
    return value


def get_api_key_from_headers(headers: Any) -> str:
    """Extract and validate the FMP API key from request headers."""
    # 1. Custom headers
    for name in _HEADER_NAMES:
        value = headers.get(name)
        if value:
            return _key_from_credential(value, "fmp_api_key")

    # 2. Authorization header
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return _key_from_credential(token, "fmp_api_key")

    return ""


def get_api_key_from_request(request: Request) -> str:
    """Extract and validate the FMP API key from request headers, auth header, or query params."""
    # Try headers first
    key = get_api_key_from_headers(request.headers)
    if key:
        return key

    # Try query parameters
    for name in _QUERY_NAMES:
        value = request.query_params.get(name)
        if value:
            return _key_from_credential(value, "fmp_api_key")

    return ""


def resolve_api_key() -> str:
    """Resolve the FMP API key from the active HTTP request, if any.

    Returns:
        str: The resolved API key, or ``""`` when none is present.
    """
    # Try headers via deps (highest priority)
    try:
        headers = deps.get_http_headers(include={"authorization"})
        key = get_api_key_from_headers(headers)
        if key:
            return key
    except Exception:  # noqa: S110
        pass

    # Try query parameters via deps
    try:
        request = deps.get_http_request()
        for name in _QUERY_NAMES:
            value = request.query_params.get(name)
            if value:
                return _key_from_credential(value, "fmp_api_key")
    except RuntimeError:
        pass

    return ""


def get_fred_api_key_from_headers(headers: Any) -> str:
    """Extract and validate the (optional) FRED API key from request headers."""
    for name in _FRED_HEADER_NAMES:
        value = headers.get(name)
        if value:
            return _key_from_credential(value, "fred_api_key")

    # An access token refers to the FRED key alongside the FMP key, if one was entered. A raw
    # bearer value is an FMP key and says nothing about FRED.
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token and _looks_like_token(token):
            return _key_from_credential(token, "fred_api_key")

    return ""


def get_fred_api_key_from_request(request: Request) -> str:
    """Extract and validate the FRED API key from headers, auth header, or query params."""
    key = get_fred_api_key_from_headers(request.headers)
    if key:
        return key

    for name in _FRED_QUERY_NAMES:
        value = request.query_params.get(name)
        if value:
            return _key_from_credential(value, "fred_api_key")

    return ""


def resolve_fred_api_key() -> str:
    """Resolve the (optional) FRED API key from the active HTTP request, if any.

    Mirrors `resolve_api_key`, but for the FRED key used by the Economics and
    FixedIncome modules' US labor/activity/rates indicators. Unlike the FMP key,
    a missing FRED key does not block access — it only disables the subset of
    tools that require it.

    Returns:
        str: The resolved API key, or ``""`` when none is present.
    """
    try:
        headers = deps.get_http_headers(include={"authorization"})
        key = get_fred_api_key_from_headers(headers)
        if key:
            return key
    except Exception:  # noqa: S110
        pass

    try:
        request = deps.get_http_request()
        for name in _FRED_QUERY_NAMES:
            value = request.query_params.get(name)
            if value:
                return _key_from_credential(value, "fred_api_key")
    except RuntimeError:
        pass

    return ""


def _is_browser(request: Request) -> bool:
    """Whether a request comes from a browser rather than an MCP client."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "text/event-stream" not in accept


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware to protect MCP endpoints in hosted deployments."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        path = request.url.path

        # CORS preflight — always allow so CORSMiddleware can respond
        if request.method == "OPTIONS":
            return await call_next(request)

        # Bypass authorization checks for discovery, oauth, health, diagnostic and usage statistics endpoints
        if (
            path.startswith("/.well-known")
            or path.startswith("/oauth")
            or path in {"/health", "/diagnostics", "/stats"}
        ):
            return await call_next(request)

        # Someone who opens the server's URL in a browser gets the setup page instead of a 401.
        # MCP clients never ask for HTML: they send JSON (POST) or ask for an event stream (GET).
        if path == "/mcp" and request.method == "GET" and _is_browser(request):
            return HTMLResponse(ROOT_PAGE)

        # Protect the MCP endpoints: SSE (/sse, /messages) and streamable-HTTP (/mcp).
        if path == "/sse" or path.startswith("/messages") or path == "/mcp":
            # If the server has a global default API key, allow access
            if os.environ.get("FINANCIAL_MODELING_PREP_API_KEY"):
                return await call_next(request)

            api_key = get_api_key_from_request(request)
            if not api_key:
                base_url = str(request.base_url).rstrip("/")
                metadata_url = f"{base_url}/.well-known/oauth-protected-resource"

                # WWW-Authenticate header pointing to our protected resource metadata. A token
                # that was sent but is not valid (expired, revoked) is reported as such, so
                # the client uses its refresh token or signs in again (RFC 6750 section 3).
                challenge = f'Bearer resource_metadata="{metadata_url}"'
                if request.headers.get("authorization"):
                    challenge += ', error="invalid_token"'
                headers = {"WWW-Authenticate": challenge}
                return JSONResponse(
                    {
                        "error": "unauthorized",
                        "error_description": "FMP API Key or Access Token is required. Please authorize via OAuth.",
                    },
                    status_code=401,
                    headers=headers,
                )

        return await call_next(request)


def _is_safe_redirect(redirect_uri: str) -> bool:
    """
    Whether a client may be sent to this redirect URI: an absolute URI with an http(s) or an
    application scheme (cursor://, vscode://), never javascript:, data: and the like.
    """
    parts = urlsplit(redirect_uri)
    if not parts.scheme or parts.scheme.lower() in _UNSAFE_REDIRECT_SCHEMES:
        return False
    if parts.scheme.lower() in ("http", "https"):
        return bool(parts.netloc)
    return True


def _with_query(url: str, params: dict[str, str]) -> str:
    """Add query parameters to a URL, keeping the ones it already has."""
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True) + list(params.items())
    return urlunsplit(parts._replace(query=urlencode(query)))


def _oauth_error(error: str, description: str, status_code: int = 400) -> JSONResponse:
    """An OAuth error response (RFC 6749 section 5.2)."""
    return JSONResponse(
        {"error": error, "error_description": description}, status_code=status_code
    )


def _token_response(grant_id: str) -> JSONResponse:
    """A new access token and refresh token for a grant (RFC 6749 section 5.1)."""
    access_token, refresh_token = get_token_store().issue_token_pair(grant_id)
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_LIFETIME,
            "refresh_token": refresh_token,
            "scope": "financetoolkit",
        },
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


def _pkce_problem(
    redirect_uri: str, code_challenge: str, code_challenge_method: str
) -> str:
    """Why an authorization request cannot be served, or "" when it can."""
    if not _is_safe_redirect(redirect_uri):
        return "The client sent no valid redirect_uri."
    if not code_challenge or code_challenge_method != "S256":
        return "The client must use PKCE with the S256 method."
    return ""


def register_auth_routes(mcp: Any) -> None:
    """Register custom OAuth 2.1 routes on the FastMCP instance."""
    if not hasattr(mcp, "custom_route"):
        return

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    _icon_path = pathlib.Path(__file__).parent / "mcpb" / "finance_toolkit_icon.png"

    @mcp.custom_route("/oauth/icon", methods=["GET"])
    async def oauth_icon(request: Request) -> Response:
        if not _icon_path.exists():
            return Response(status_code=404)
        return Response(content=_icon_path.read_bytes(), media_type="image/png")

    # Favicon lookups (claude.ai, Google's favicon service) read the root page and the
    # small sizes; a redirect to the website would make them pick up its icon instead.
    _static_path = pathlib.Path(__file__).parent / "static"
    _static_files = {
        "/favicon.ico": ("favicon.ico", "image/x-icon"),
        "/favicon-32.png": ("favicon-32.png", "image/png"),
        "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png"),
    }

    def _register_static(route: str, file_name: str, media_type: str) -> None:
        @mcp.custom_route(route, methods=["GET"])
        async def static_file(request: Request) -> Response:
            file_path = _static_path / file_name
            if not file_path.exists():
                return Response(status_code=404)
            return Response(
                content=file_path.read_bytes(),
                media_type=media_type,
                headers={"Cache-Control": "public, max-age=86400"},
            )

    for route, (file_name, media_type) in _static_files.items():
        _register_static(route, file_name, media_type)

    @mcp.custom_route("/", methods=["GET"])
    async def root(request: Request) -> HTMLResponse:
        return HTMLResponse(ROOT_PAGE)

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def oauth_protected_resource(request: Request) -> JSONResponse:
        base_url = str(request.base_url).rstrip("/")
        return JSONResponse(
            {
                "resource": base_url,
                "authorization_servers": [base_url],
                "bearer_methods_supported": ["header"],
                "scopes_supported": ["financetoolkit"],
                "resource_name": "Finance Toolkit Analyst",
            }
        )

    @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def oauth_authorization_server(request: Request) -> JSONResponse:
        base_url = str(request.base_url).rstrip("/")
        return JSONResponse(
            {
                "issuer": base_url,
                "authorization_endpoint": f"{base_url}/oauth/authorize",
                "token_endpoint": f"{base_url}/oauth/token",
                "registration_endpoint": f"{base_url}/oauth/register",
                "revocation_endpoint": f"{base_url}/oauth/revoke",
                "logo_uri": f"{base_url}/oauth/icon",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none"],
                "revocation_endpoint_auth_methods_supported": ["none"],
                "scopes_supported": ["financetoolkit"],
                "code_challenge_methods_supported": ["S256"],
            }
        )

    @mcp.custom_route("/oauth/register", methods=["POST"])
    async def oauth_register(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}

        client_name = body.get("client_name", "MCP Client")
        redirect_uris = body.get("redirect_uris", [])

        client_id = f"client_{secrets.token_hex(8)}"

        base_url = str(request.base_url).rstrip("/")
        return JSONResponse(
            {
                "client_id": client_id,
                "client_name": client_name,
                "redirect_uris": redirect_uris,
                "logo_uri": f"{base_url}/oauth/icon",
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
            status_code=201,
        )

    @mcp.custom_route("/oauth/authorize", methods=["GET"])
    async def authorize_get(request: Request) -> HTMLResponse:
        client_id = request.query_params.get("client_id", "")
        redirect_uri = request.query_params.get("redirect_uri", "")
        code_challenge = request.query_params.get("code_challenge", "")
        code_challenge_method = request.query_params.get("code_challenge_method", "")
        state = request.query_params.get("state", "")

        problem = _pkce_problem(redirect_uri, code_challenge, code_challenge_method)
        if problem:
            return HTMLResponse(html.escape(problem), status_code=400)

        # The badge names where the user is sent back to, which tells them who is asking.
        destination = urlsplit(redirect_uri).netloc or urlsplit(redirect_uri).scheme
        page = HTML_TEMPLATE.format(
            client_id=html.escape(client_id, quote=True),
            client_name=html.escape(destination or client_id or "MCP Client"),
            redirect_uri=html.escape(redirect_uri, quote=True),
            code_challenge=html.escape(code_challenge, quote=True),
            code_challenge_method=html.escape(code_challenge_method, quote=True),
            state=html.escape(state, quote=True),
        )
        return HTMLResponse(content=page)

    @mcp.custom_route("/oauth/authorize", methods=["POST"])
    async def authorize_post(request: Request) -> HTMLResponse | RedirectResponse:
        form = await request.form()
        fmp_api_key = str(form.get("fmp_api_key", "")).strip()
        fred_api_key = str(form.get("fred_api_key", "")).strip()
        client_id = str(form.get("client_id", ""))
        redirect_uri = str(form.get("redirect_uri", ""))
        code_challenge = str(form.get("code_challenge", ""))
        code_challenge_method = str(form.get("code_challenge_method", ""))
        state = str(form.get("state", ""))

        problem = _pkce_problem(redirect_uri, code_challenge, code_challenge_method)
        if problem:
            return HTMLResponse(html.escape(problem), status_code=400)
        if not fmp_api_key:
            return HTMLResponse("FMP API Key is required", status_code=400)

        # The keys stay on the server; the code that travels through the browser is a random
        # reference to them that works once, for this client, within five minutes.
        auth_code = get_token_store().create_authorization_code(
            {"fmp_api_key": fmp_api_key, "fred_api_key": fred_api_key},
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
        )

        params = {"code": auth_code}
        if state:
            params["state"] = state

        return RedirectResponse(url=_with_query(redirect_uri, params), status_code=303)

    @mcp.custom_route("/oauth/token", methods=["POST"])
    async def oauth_token(request: Request) -> JSONResponse:
        try:
            form = await request.form()
        except Exception:
            form = {}

        if not form:
            try:
                form = await request.json()
            except Exception:
                form = {}

        grant_type = form.get("grant_type")
        client_id = str(form.get("client_id") or "")

        if grant_type == "refresh_token":
            refresh_token = str(form.get("refresh_token") or "")
            grant_id = (
                get_token_store().redeem_refresh_token(refresh_token, client_id)
                if refresh_token
                else None
            )
            if grant_id is None:
                return _oauth_error(
                    "invalid_grant", "Refresh token is invalid, expired or revoked"
                )
            return _token_response(grant_id)

        if grant_type != "authorization_code":
            return _oauth_error(
                "unsupported_grant_type",
                "Only authorization_code and refresh_token are supported",
            )

        code = str(form.get("code") or "")
        if not code:
            return _oauth_error("invalid_request", "code is required")

        redeemed = get_token_store().redeem_authorization_code(code)
        if redeemed is None:
            return _oauth_error(
                "invalid_grant", "Authorization code is invalid, expired or used"
            )
        grant_id, code_client_id, code_meta = redeemed

        # The code belongs to the client and redirect URI it was issued for (RFC 6749 4.1.3).
        if client_id != code_client_id:
            get_token_store().revoke(code)
            return _oauth_error("invalid_grant", "Client ID mismatch")
        if str(form.get("redirect_uri") or "") != code_meta.get("redirect_uri"):
            get_token_store().revoke(code)
            return _oauth_error("invalid_grant", "Redirect URI mismatch")

        # PKCE (RFC 7636): only the client that started the flow knows the verifier.
        code_verifier = str(form.get("code_verifier") or "")
        expected_challenge = _base64url_encode(
            hashlib.sha256(code_verifier.encode("utf-8")).digest()
        )
        if not code_verifier or not hmac.compare_digest(
            expected_challenge, str(code_meta.get("code_challenge", ""))
        ):
            get_token_store().revoke(code)
            return _oauth_error("invalid_grant", "PKCE S256 verification failed")

        return _token_response(grant_id)

    @mcp.custom_route("/oauth/revoke", methods=["POST"])
    async def oauth_revoke(request: Request) -> Response:
        # RFC 7009: revoking a token ends its grant, which deletes the stored keys. The
        # answer is 200 whether or not the token was known, so it reveals nothing.
        try:
            form = await request.form()
        except Exception:
            form = {}
        token = str(form.get("token") or "").strip()
        if token:
            get_token_store().revoke(token)
        return Response(status_code=200)
