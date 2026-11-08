"""Per-request Financial Modeling Prep and FRED API key resolution and OAuth 2.1 authentication for hosted deployments."""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import ipaddress
import json
import os
import pathlib
import re
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlencode, urlsplit

import fastmcp.server.dependencies as deps
from cryptography.fernet import Fernet, InvalidToken
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from financetoolkit.mcp_server.setup_model import (
    get_global_env_path,
)
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

_JWT_PART_COUNT = 3

# Authorization codes are random strings that point to the submitted keys held here, so the
# keys never travel through the browser's address bar, history or the redirect target's logs.
# Each code is exchanged once and expires after five minutes. The codes live in this process
# only, so the server must run as a single process (as main() starts it): with several
# workers, a token request could reach a worker that never saw the code.
AUTH_CODE_LIFETIME = 300
_MAX_PENDING_CODES = 10_000
# The most codes one client address may have pending: a sign-in exchanges its code within
# seconds, so only a flood of authorizations comes near it.
_MAX_PENDING_CODES_PER_CLIENT = 20

# The longest value each authorization field may have; a form field may otherwise be up to
# a megabyte, and every pending code keeps its fields in memory.
FIELD_LIMITS = {
    "fmp_api_key": 128,
    "fred_api_key": 128,
    # A registered client_id carries its redirect URIs (see register_client).
    "client_id": 4096,
    "redirect_uri": 2048,
    "state": 1024,
}
# A S256 code challenge is the base64url SHA-256 of the verifier, without padding.
CODE_CHALLENGE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")
_auth_codes: dict[str, tuple[float, str, dict[str, Any]]] = {}
_auth_codes_lock = threading.Lock()

# The secret signing the access tokens, read once per process.
_secret_key: bytes | None = None
MINIMUM_SECRET_BYTES = 32

# Access and refresh tokens hold the user's API keys encrypted (Fernet: AES-128-CBC with an
# HMAC-SHA256) under a key derived from the server secret, so a token that shows up in a log,
# a proxy or another client doesn't reveal the keys. Nothing is stored per user: each request
# decrypts its token. An access token lasts a day and a refresh token, renewed with every
# refresh, 90 days, so a client in use stays signed in while a leaked access token soon stops
# working. Rotating FT_MCP_SECRET_KEY ends every token at once.
TOKEN_PREFIX = "ftk_"
ACCESS_TOKEN_LIFETIME = 24 * 60 * 60
REFRESH_TOKEN_LIFETIME = 90 * 24 * 60 * 60

# A registered client_id is "ftc_" with its name and redirect URIs, signed with a key derived
# from the server secret, so registration stores nothing and the redirect URIs can't be
# swapped: an authorization for a registered client only goes back to one of its own URIs.
REGISTERED_CLIENT_PREFIX = "ftc_"
MAXIMUM_REDIRECT_URIS = 10
MAXIMUM_CLIENT_NAME_LENGTH = 100

# Headers for every page: no framing by another site (so a visitor can't be tricked into
# typing a key into an invisible overlay), no MIME sniffing, and no Referer, which would
# carry the authorization request's address to the sites the page links to.
PAGE_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}

# Checked in order. FastMCP forwards `x-*`; `authorization` must be opted in.
_HEADER_NAMES: tuple[str, ...] = (
    "x-fmp-api-key",
    "x-financial-modeling-prep-api-key",
)

# Query-string parameter names checked in order. A key in the URL also lands in proxy and
# access logs, browser history and client configs, so the authorization page and the
# headers are the recommended ways; the query string is accepted for clients that can't
# set either.
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
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
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

        .redirect-note {{
            margin-top: 10px;
            font-size: 12.5px;
            color: var(--text-secondary);
            line-height: 1.5;
        }}

        .redirect-note strong {{
            color: var(--text-primary);
            word-break: break-all;
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
                Optionally add a FRED API key to also enable the ICE BofA and Moody's corporate bond series and
                the full history of US nonfarm payrolls.
            </p>
            <p class="redirect-note">
                After you authorize, your browser returns to <strong>{redirect_host}</strong> with
                access for this client. Only continue if you started connecting from there yourself.
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
                Your keys are encrypted into the access token your client keeps; the server doesn't store them.
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

    <script>{script}</script>
</body>
</html>
"""

# The authorization page's only script (showing and hiding the typed keys), allowed by its
# hash in the page's Content-Security-Policy so that no other script can run there.
AUTHORIZE_SCRIPT = """
(function () {
    function wireToggle(buttonId, inputId) {
        var btn = document.getElementById(buttonId);
        var inp = document.getElementById(inputId);
        btn.addEventListener('click', function () {
            var isHidden = inp.type === 'password';
            inp.type = isHidden ? 'text' : 'password';
            btn.classList.toggle('revealed', isHidden);
        });
    }
    wireToggle('toggleBtn', 'fmp_api_key');
    wireToggle('toggleBtnFred', 'fred_api_key');
})();
"""
AUTHORIZE_SCRIPT_HASH = base64.b64encode(
    hashlib.sha256(AUTHORIZE_SCRIPT.encode()).digest()
).decode()


def _restrict_to_owner(secret_file: pathlib.Path) -> None:
    """
    Make an existing secret file readable by its owner only, when others can read it.

    Only ever tightens: a file that is already private (or owned by someone else, such as a
    mounted secret) is left as it is, and a failure is logged rather than raised.

    Args:
        secret_file (pathlib.Path): The stored secret.
    """
    try:
        # Windows has no owner and group bits to clear: its files are guarded by access lists.
        if not hasattr(os, "getuid"):
            return
        status = secret_file.stat()
        if status.st_uid == os.getuid() and status.st_mode & 0o077:
            # Clear the group and other bits only, so a read-only file stays read-only.
            secret_file.chmod(status.st_mode & 0o700)
            logger.warning(
                "The MCP secret file %s was readable by others and is now private.",
                secret_file,
            )
    except OSError as error:
        logger.warning(
            "The MCP secret file could not be made private (%s).", type(error).__name__
        )


def get_secret_key() -> bytes:
    """Load or generate a stable secret key stored in the global configuration directory."""
    global _secret_key  # noqa: PLW0603

    env_key = os.environ.get("FT_MCP_SECRET_KEY")
    if env_key:
        return env_key.encode("utf-8")

    if _secret_key is not None:
        return _secret_key

    # A file in the global config directory keeps the tokens valid across restarts. When it
    # can't be written, a random key for this process only is safer than any fixed fallback:
    # tokens then stop working after a restart, but nobody can forge them.
    try:
        config_dir = get_global_env_path().parent
        config_dir.mkdir(parents=True, exist_ok=True)
        secret_file = config_dir / ".mcp_secret"
        stored = secret_file.read_bytes() if secret_file.exists() else b""
        # An empty or truncated file (an interrupted first write) counts as missing.
        if len(stored) >= MINIMUM_SECRET_BYTES:
            _restrict_to_owner(secret_file)
            _secret_key = stored
        else:
            key = secrets.token_bytes(32)
            # Created private, so the key is never readable by others, not even briefly.
            # The mode only applies to a new file, so a short file that already existed is
            # made private before the key goes in.
            descriptor = os.open(
                secret_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), 0o600)
                handle.write(key)
            _secret_key = key
    except OSError as error:
        logger.warning(
            "The MCP secret could not be stored (%s), so a key for this process is used and "
            "access tokens stop working after a restart. Set FT_MCP_SECRET_KEY to keep them.",
            type(error).__name__,
        )
        _secret_key = secrets.token_bytes(32)

    return _secret_key


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to a URL-safe base64 string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    """Decode a URL-safe base64 string without padding."""
    padding = "=" * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)


def _derived_key(purpose: bytes) -> bytes:
    """
    Derive a key for one purpose from the server secret, so the token encryption and the
    client registrations never share a key.

    Args:
        purpose (bytes): What the key is for.

    Returns:
        bytes: A 32-byte key.
    """
    return hmac.new(get_secret_key(), purpose, hashlib.sha256).digest()


def _token_cipher() -> Fernet:
    """The cipher that encrypts the access and refresh tokens."""
    return Fernet(base64.urlsafe_b64encode(_derived_key(b"financetoolkit-token")))


def issue_token(keys: dict[str, Any], kind: str, lifetime: int) -> str:
    """
    Create an encrypted access or refresh token that holds the user's API keys.

    Args:
        keys (dict[str, Any]): The FMP and FRED keys and the client they were given to.
        kind (str): "access" or "refresh"; a token only works as its own kind.
        lifetime (int): Seconds until the token expires.

    Returns:
        str: The token.
    """
    content = {**keys, "typ": kind, "exp": int(time.time()) + lifetime}

    return TOKEN_PREFIX + _token_cipher().encrypt(json.dumps(content).encode()).decode()


def read_token(token: str, kind: str = "access") -> dict[str, Any] | None:
    """
    Read the keys from a token this server issued, when it is valid and of the kind asked.

    A signed JWT (the unencrypted access token format) is read too until it expires, so
    connections made with one keep working.

    Args:
        token (str): The token.
        kind (str): "access" or "refresh". Defaults to "access".

    Returns:
        dict[str, Any] | None: The token's content, or None when it is not a valid token of
            this kind (expired, forged, issued under another secret or another kind).
    """
    if token.startswith(TOKEN_PREFIX):
        try:
            content = json.loads(_token_cipher().decrypt(token[len(TOKEN_PREFIX) :]))
        except (InvalidToken, ValueError):
            return None
        if content.get("typ") != kind or int(content.get("exp", 0)) < time.time():
            return None
        return content

    return _read_signed_jwt(token) if kind == "access" else None


def _read_signed_jwt(token: str) -> dict[str, Any] | None:
    """Verify a signed (HS256) JWT access token and its expiry. Returns its payload or None."""
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


def register_client(client_name: str, redirect_uris: list[str]) -> str:
    """
    Create the client_id of a registration: its name and redirect URIs, signed.

    Args:
        client_name (str): The name the client registered with.
        redirect_uris (list[str]): The redirect URIs it registered.

    Returns:
        str: The client_id.
    """
    body = _base64url_encode(
        json.dumps(
            {"name": client_name, "redirect_uris": redirect_uris},
            separators=(",", ":"),
        ).encode()
    )
    signature = hmac.new(
        _derived_key(b"financetoolkit-client"), body.encode(), hashlib.sha256
    ).digest()

    return f"{REGISTERED_CLIENT_PREFIX}{body}.{_base64url_encode(signature)}"


def read_client_registration(client_id: str) -> dict[str, Any] | None:
    """
    Read the name and redirect URIs from a client_id that register_client created.

    Args:
        client_id (str): The client_id.

    Returns:
        dict[str, Any] | None: The registration, or None for any other client_id, including
            one whose signature doesn't verify.
    """
    if not client_id.startswith(REGISTERED_CLIENT_PREFIX):
        return None

    body, _, signature = client_id[len(REGISTERED_CLIENT_PREFIX) :].partition(".")
    expected = hmac.new(
        _derived_key(b"financetoolkit-client"), body.encode(), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(signature, _base64url_encode(expected)):
        return None

    try:
        registration = json.loads(_base64url_decode(body))
    except ValueError:
        return None

    return registration if isinstance(registration, dict) else None


def get_api_key_from_headers(headers: Any) -> str:
    """Extract and validate the FMP API key from request headers."""
    # 1. Custom headers
    for name in _HEADER_NAMES:
        value = headers.get(name)
        if value:
            return key_from_credential(value, "fmp_api_key")

    # 2. Authorization header
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return key_from_credential(token, "fmp_api_key")

    return ""


def key_from_credential(credential: str, field: str) -> str:
    """Read an API key from a credential: an access token issued here, or the key itself.

    Args:
        credential (str): The header, bearer or query value.
        field (str): The key an access token holds it under ("fmp_api_key" or "fred_api_key").

    Returns:
        str: The key; "" for an access token that doesn't verify or doesn't hold it.
    """
    credential = credential.strip()
    payload = read_token(credential)
    if payload is not None:
        return str(payload.get(field) or "")
    # An access token that no longer verifies (expired, or signed with another secret) is
    # not an API key: MCPAuthMiddleware answers it with a 401 so the client signs in again,
    # and its contents are never sent on to FinancialModelingPrep.
    if is_invalid_access_token(credential):
        return ""
    return credential


def is_invalid_access_token(token: str) -> bool:
    """Whether a bearer token has the shape of an access token but is not a valid one.

    API keys are plain strings without dots or the token prefix, so a token with either is
    one this server (or an earlier secret) issued, and one that isn't valid is expired,
    forged or a refresh token.

    Args:
        token (str): The bearer token.

    Returns:
        bool: True for a token of this server that is not a valid access token.
    """
    issued_here = (
        token.startswith(TOKEN_PREFIX) or len(token.split(".")) == _JWT_PART_COUNT
    )
    return issued_here and read_token(token) is None


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
            return key_from_credential(value, "fmp_api_key")

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
                return key_from_credential(value, "fmp_api_key")
    except RuntimeError:
        pass

    return ""


def get_fred_api_key_from_headers(headers: Any) -> str:
    """Extract and validate the (optional) FRED API key from request headers."""
    for name in _FRED_HEADER_NAMES:
        value = headers.get(name)
        if value:
            return key_from_credential(value, "fred_api_key")

    # A signed FMP bearer token carries the FRED key alongside it, if authorized.
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            payload = read_token(token)
            if payload and "fred_api_key" in payload:
                return str(payload["fred_api_key"])

    return ""


def get_fred_api_key_from_request(request: Request) -> str:
    """Extract and validate the FRED API key from headers, auth header, or query params."""
    key = get_fred_api_key_from_headers(request.headers)
    if key:
        return key

    for name in _FRED_QUERY_NAMES:
        value = request.query_params.get(name)
        if value:
            return key_from_credential(value, "fred_api_key")

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
                return key_from_credential(value, "fred_api_key")
    except RuntimeError:
        pass

    return ""


def validate_redirect_uri(redirect_uri: str) -> str:
    """Check where the authorization may send the browser back to.

    The browser is sent to the redirect URI with the authorization code, so it must be an
    https address, or http on the user's own machine (loopback) for local clients.

    Args:
        redirect_uri (str): The redirect URI the client asked for.

    Returns:
        str: An explanation of what is wrong, or "" when the redirect URI is acceptable.
    """
    if not redirect_uri:
        return "redirect_uri is required."

    parts = urlsplit(redirect_uri)
    if not parts.scheme or parts.fragment:
        return "redirect_uri must be an absolute URL without a fragment."
    # Native apps register a private-use scheme (cursor://..., or com.example.app:/callback
    # without a host as in RFC 8252), which the operating system hands to the app itself.
    if parts.scheme not in {"http", "https"}:
        if parts.scheme in {"javascript", "data", "file", "vbscript"}:
            return "redirect_uri must use https (http is only allowed for localhost)."
        return ""
    if not parts.hostname:
        return "redirect_uri must be an absolute URL without a fragment."
    if parts.scheme == "https" or _is_loopback(redirect_uri):
        return ""
    return "redirect_uri must use https (http is only allowed for localhost)."


def _is_loopback(redirect_uri: str) -> bool:
    """Whether a redirect URI is http on the user's own machine (localhost or loopback)."""
    parts = urlsplit(redirect_uri)
    if parts.scheme != "http":
        return False
    try:
        return (
            parts.hostname == "localhost"
            or ipaddress.ip_address(parts.hostname or "").is_loopback
        )
    except ValueError:
        return False


def is_registered_redirect(redirect_uri: str, registered: list[str]) -> bool:
    """
    Whether a redirect URI is one the client registered.

    The match is exact, except for the port of a loopback redirect: a native app listens
    on a port it picks when the sign-in starts (RFC 8252, section 7.3).

    Args:
        redirect_uri (str): The redirect URI of the authorization request.
        registered (list[str]): The redirect URIs of the client's registration.

    Returns:
        bool: True when the redirect URI is registered.
    """
    if redirect_uri in registered:
        return True
    if not _is_loopback(redirect_uri):
        return False

    def without_port(uri: str) -> tuple:
        parts = urlsplit(uri)
        return parts.scheme, parts.hostname, parts.path, parts.query

    return any(
        _is_loopback(uri) and without_port(uri) == without_port(redirect_uri)
        for uri in registered
    )


def describe_redirect(redirect_uri: str) -> str:
    """
    Name where an authorization returns to, as shown on the authorization page: the host
    of a web address, or the scheme and host of an app's private-use scheme.

    Args:
        redirect_uri (str): The redirect URI.

    Returns:
        str: The host, e.g. "claude.ai" or "cursor://anysphere.cursor-retrieval".
    """
    parts = urlsplit(redirect_uri)
    if parts.scheme in {"http", "https"}:
        return parts.hostname or redirect_uri
    return (
        f"{parts.scheme}://{parts.hostname}" if parts.hostname else f"{parts.scheme}:"
    )


def authorize_page_headers(redirect_uri: str) -> dict[str, str]:
    """
    The headers of the authorization page: those of every page, with a policy that runs
    only the page's own script and submits the form only to this server, which then sends
    the browser on to the redirect URI (browsers check that step against form-action too).

    Args:
        redirect_uri (str): The validated redirect URI of the authorization request.

    Returns:
        dict[str, str]: The response headers.
    """
    parts = urlsplit(redirect_uri)
    destination = f"{parts.scheme}:"
    # Only a plain host name or IPv4 address is named; anything else allows the scheme.
    if parts.scheme in {"http", "https"} and re.fullmatch(
        r"[A-Za-z0-9.-]+", parts.hostname or ""
    ):
        try:
            port = f":{parts.port}" if parts.port else ""
            destination = f"{parts.scheme}://{parts.hostname}{port}"
        except ValueError:
            pass

    return {
        **PAGE_HEADERS,
        "Content-Security-Policy": (
            "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
            f"script-src 'sha256-{AUTHORIZE_SCRIPT_HASH}'; "
            f"form-action 'self' {destination}; base-uri 'none'; frame-ancestors 'none'"
        ),
    }


def build_redirect_url(redirect_uri: str, **params: str) -> str:
    """Add query parameters to a redirect URI, keeping any query it already has.

    Args:
        redirect_uri (str): The redirect URI.
        **params (str): The parameters to add; empty values are left out.

    Returns:
        str: The redirect URI with the parameters URL-encoded.
    """
    query = urlencode({key: value for key, value in params.items() if value})
    separator = "&" if urlsplit(redirect_uri).query else "?"
    return f"{redirect_uri}{separator}{query}"


def issue_authorization_code(payload: dict[str, Any], client: str = "") -> str | None:
    """Store the submitted keys and return a random code that refers to them.

    One client address may hold _MAX_PENDING_CODES_PER_CLIENT codes at a time. When the
    store is full, the oldest code makes way, so a flood of authorizations cannot keep
    others from signing in: a code is exchanged within seconds of being issued.

    Args:
        payload (dict[str, Any]): The keys and the client binding of the authorization.
        client (str): The address of the client asking for the code. Defaults to "".

    Returns:
        str | None: The single-use authorization code, or None when the client already
            has too many pending.
    """
    code = secrets.token_urlsafe(32)
    now = time.time()
    with _auth_codes_lock:
        for expired in [
            key for key, (expires, _, _) in _auth_codes.items() if expires < now
        ]:
            del _auth_codes[expired]
        if (
            sum(owner == client for _, owner, _ in _auth_codes.values())
            >= _MAX_PENDING_CODES_PER_CLIENT
        ):
            return None
        # The codes are kept in the order they were issued, so the first is the oldest.
        while len(_auth_codes) >= _MAX_PENDING_CODES:
            del _auth_codes[next(iter(_auth_codes))]
        _auth_codes[code] = (now + AUTH_CODE_LIFETIME, client, payload)
    return code


def redeem_authorization_code(code: str) -> dict[str, Any] | None:
    """Take an authorization code out of the store, so it can be exchanged only once.

    Args:
        code (str): The authorization code.

    Returns:
        dict[str, Any] | None: The stored payload, or None when the code is unknown,
            already used or expired.
    """
    with _auth_codes_lock:
        entry = _auth_codes.pop(code, None)
    if entry is None or entry[0] < time.time():
        return None
    return entry[2]


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
            return HTMLResponse(ROOT_PAGE, headers=PAGE_HEADERS)

        # Protect the MCP endpoints: SSE (/sse, /messages) and streamable-HTTP (/mcp).
        if path == "/sse" or path.startswith("/messages") or path == "/mcp":
            # If the server has a global default API key, allow access
            if os.environ.get("FINANCIAL_MODELING_PREP_API_KEY"):
                return await call_next(request)

            base_url = str(request.base_url).rstrip("/")
            metadata_url = f"{base_url}/.well-known/oauth-protected-resource"
            auth_header = request.headers.get("authorization", "")
            bearer = (
                auth_header[7:].strip()
                if auth_header.lower().startswith("bearer ")
                else ""
            )

            # An expired or foreign access token gets invalid_token, the signal for a client to
            # run the authorization again instead of failing every call with a dead key.
            if bearer and is_invalid_access_token(bearer):
                return JSONResponse(
                    {
                        "error": "invalid_token",
                        "error_description": "The access token is expired or invalid. Please authorize again.",
                    },
                    status_code=401,
                    headers={
                        "WWW-Authenticate": f'Bearer error="invalid_token", resource_metadata="{metadata_url}"'
                    },
                )

            api_key = get_api_key_from_request(request)
            if not api_key:
                # WWW-Authenticate header pointing to our protected resource metadata
                headers = {
                    "WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'
                }
                return JSONResponse(
                    {
                        "error": "unauthorized",
                        "error_description": "FMP API Key or Access Token is required. Please authorize via OAuth.",
                    },
                    status_code=401,
                    headers=headers,
                )

        return await call_next(request)


def _authorization_problem(
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    client_id: str,
    **fields: str,
) -> str:
    """Explain what makes an authorization request unacceptable, or return ""."""
    for name, value in {
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        **fields,
    }.items():
        if len(value) > FIELD_LIMITS[name]:
            return f"{name} is longer than {FIELD_LIMITS[name]} characters."
    problem = validate_redirect_uri(redirect_uri)
    if problem:
        return problem
    # A client that registered here only gets authorizations sent to its own redirect URIs.
    # Other client_ids (clients that don't register) are accepted, and the page names
    # where the browser returns to.
    if client_id.startswith(REGISTERED_CLIENT_PREFIX):
        registration = read_client_registration(client_id)
        if registration is None:
            return "client_id is not a registration of this server."
        if not is_registered_redirect(
            redirect_uri, list(registration.get("redirect_uris", []))
        ):
            return "redirect_uri is not registered for this client."
    if code_challenge_method != "S256" or not CODE_CHALLENGE_PATTERN.match(
        code_challenge
    ):
        return (
            "PKCE is required: send a S256 code_challenge (43 base64url characters) with "
            "code_challenge_method=S256."
        )
    return ""


def _registration_error(description: str) -> JSONResponse:
    """The answer to a client registration with unusable redirect URIs (RFC 7591)."""
    return JSONResponse(
        {"error": "invalid_redirect_uri", "error_description": description},
        status_code=400,
    )


def _token_response(grant: dict[str, Any]) -> JSONResponse:
    """
    Answer a token request with a new access token and refresh token for the keys of an
    authorization or of the refresh token presented.

    Args:
        grant (dict[str, Any]): The authorization's or refresh token's content.

    Returns:
        JSONResponse: The token response.
    """
    keys = {
        "fmp_api_key": grant.get("fmp_api_key"),
        "fred_api_key": grant.get("fred_api_key"),
        "client_id": grant.get("client_id"),
    }

    return JSONResponse(
        {
            "access_token": issue_token(keys, "access", ACCESS_TOKEN_LIFETIME),
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_LIFETIME,
            "refresh_token": issue_token(keys, "refresh", REFRESH_TOKEN_LIFETIME),
            "scope": "financetoolkit",
        }
    )


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
        return HTMLResponse(ROOT_PAGE, headers=PAGE_HEADERS)

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
                "logo_uri": f"{base_url}/oauth/icon",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none"],
                "scopes_supported": ["financetoolkit"],
                "code_challenge_methods_supported": ["S256"],
            }
        )

    @mcp.custom_route("/oauth/register", methods=["POST"])
    async def oauth_register(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = None

        redirect_uris = body.get("redirect_uris") if isinstance(body, dict) else None
        if not (
            isinstance(redirect_uris, list)
            and 0 < len(redirect_uris) <= MAXIMUM_REDIRECT_URIS
            and all(isinstance(uri, str) for uri in redirect_uris)
        ):
            return _registration_error(
                f"redirect_uris must be a list of 1 to {MAXIMUM_REDIRECT_URIS} URLs."
            )
        for uri in redirect_uris:
            problem = (
                "A redirect URI is too long."
                if len(uri) > FIELD_LIMITS["redirect_uri"]
                else validate_redirect_uri(uri)
            )
            if problem:
                return _registration_error(problem)

        client_name = str(body.get("client_name") or "MCP Client")[
            :MAXIMUM_CLIENT_NAME_LENGTH
        ]
        client_id = register_client(client_name, redirect_uris)
        if len(client_id) > FIELD_LIMITS["client_id"]:
            return _registration_error("The redirect URIs are too long together.")

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
        code_challenge_method = request.query_params.get(
            "code_challenge_method", "S256"
        )
        state = request.query_params.get("state", "")

        problem = _authorization_problem(
            redirect_uri,
            code_challenge,
            code_challenge_method,
            client_id=client_id,
            state=state,
        )
        if problem:
            return HTMLResponse(
                html.escape(problem), status_code=400, headers=PAGE_HEADERS
            )

        # Every value comes from the URL, so it is escaped: a crafted link must not be able
        # to add markup or scripts to the page where users enter their API keys. The badge
        # and the note name the site the browser returns to, so a link from elsewhere
        # stands out.
        registration = read_client_registration(client_id)
        name = registration.get("name") if registration else client_id
        redirect_host = describe_redirect(redirect_uri)
        page = HTML_TEMPLATE.format(
            client_id=html.escape(client_id, quote=True),
            client_name=html.escape(f"{name or 'MCP Client'} · {redirect_host}"),
            redirect_host=html.escape(redirect_host),
            script=AUTHORIZE_SCRIPT,
            redirect_uri=html.escape(redirect_uri, quote=True),
            code_challenge=html.escape(code_challenge, quote=True),
            code_challenge_method=html.escape(code_challenge_method, quote=True),
            state=html.escape(state, quote=True),
        )
        return HTMLResponse(content=page, headers=authorize_page_headers(redirect_uri))

    @mcp.custom_route("/oauth/authorize", methods=["POST"])
    async def authorize_post(request: Request) -> HTMLResponse | RedirectResponse:
        form = await request.form()
        fmp_api_key = str(form.get("fmp_api_key", "")).strip()
        fred_api_key = str(form.get("fred_api_key", "")).strip()
        client_id = str(form.get("client_id", ""))
        redirect_uri = str(form.get("redirect_uri", ""))
        code_challenge = str(form.get("code_challenge", ""))
        code_challenge_method = str(form.get("code_challenge_method", "S256"))
        state = str(form.get("state", ""))

        if not fmp_api_key:
            return HTMLResponse(
                "FMP API Key is required", status_code=400, headers=PAGE_HEADERS
            )

        problem = _authorization_problem(
            redirect_uri,
            code_challenge,
            code_challenge_method,
            client_id=client_id,
            state=state,
            fmp_api_key=fmp_api_key,
            fred_api_key=fred_api_key,
        )
        if problem:
            return HTMLResponse(
                html.escape(problem), status_code=400, headers=PAGE_HEADERS
            )

        # fred_api_key only unlocks the FRED subset.
        payload = {
            "fmp_api_key": fmp_api_key,
            "fred_api_key": fred_api_key,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }
        auth_code = issue_authorization_code(
            payload, request.client.host if request.client else ""
        )
        if auth_code is None:
            return HTMLResponse(
                "Too many authorizations are pending. Please try again in a few minutes.",
                status_code=429,
                headers=PAGE_HEADERS,
            )

        return RedirectResponse(
            url=build_redirect_url(redirect_uri, code=auth_code, state=state),
            status_code=303,
        )

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
        if grant_type == "refresh_token":
            refreshed = read_token(str(form.get("refresh_token") or ""), "refresh")
            if refreshed is None:
                return JSONResponse(
                    {
                        "error": "invalid_grant",
                        "error_description": "The refresh token is invalid or expired",
                    },
                    status_code=400,
                )
            client_id = form.get("client_id")
            if client_id and refreshed.get("client_id") != client_id:
                return JSONResponse(
                    {
                        "error": "invalid_grant",
                        "error_description": "Client ID mismatch",
                    },
                    status_code=400,
                )
            return _token_response(refreshed)
        if grant_type != "authorization_code":
            return JSONResponse(
                {
                    "error": "unsupported_grant_type",
                    "error_description": "Only authorization_code and refresh_token are supported",
                },
                status_code=400,
            )

        code = form.get("code")
        if not code:
            return JSONResponse(
                {"error": "invalid_request", "error_description": "code is required"},
                status_code=400,
            )

        redirect_uri = form.get("redirect_uri")
        client_id = form.get("client_id")
        code_verifier = form.get("code_verifier")

        payload = redeem_authorization_code(str(code))
        if not payload:
            return JSONResponse(
                {
                    "error": "invalid_grant",
                    "error_description": "Authorization code is invalid or expired",
                },
                status_code=400,
            )

        # Check client id and redirect uri
        if client_id and payload.get("client_id") != client_id:
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "Client ID mismatch"},
                status_code=400,
            )
        if payload.get("redirect_uri") != redirect_uri:
            return JSONResponse(
                {
                    "error": "invalid_grant",
                    "error_description": "Redirect URI mismatch",
                },
                status_code=400,
            )

        # PKCE (S256, required at /oauth/authorize): only the client that started the
        # authorization holds the verifier.
        if not code_verifier:
            return JSONResponse(
                {
                    "error": "invalid_grant",
                    "error_description": "code_verifier is required",
                },
                status_code=400,
            )
        sha256 = hashlib.sha256(str(code_verifier).encode("utf-8")).digest()
        if not hmac.compare_digest(
            _base64url_encode(sha256), str(payload.get("code_challenge", ""))
        ):
            return JSONResponse(
                {
                    "error": "invalid_grant",
                    "error_description": "PKCE S256 verification failed",
                },
                status_code=400,
            )

        return _token_response(payload)
