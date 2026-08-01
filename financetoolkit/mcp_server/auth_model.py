"""Per-request Financial Modeling Prep and FRED API key resolution and OAuth 2.1 authentication for hosted deployments."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import time
from typing import Any

import fastmcp.server.dependencies as deps
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from financetoolkit.mcp_server.setup_model import (
    get_global_env_path,
)
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

_JWT_PART_COUNT = 3

# Header names checked in order. Custom ``x-*`` headers are forwarded by
# FastMCP's ``get_http_headers`` by default; ``authorization`` must be opted in.
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

# Header names checked in order for the (optional) FRED API key, used by the
# Economics and FixedIncome modules' US labor/activity/rates indicators.
_FRED_HEADER_NAMES: tuple[str, ...] = ("x-fred-api-key",)

# Query-string parameter names checked in order for the FRED API key.
_FRED_QUERY_NAMES: tuple[str, ...] = (
    "fred_api_key",
    "fred_key",
)

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


def _get_secret_key() -> bytes:
    """Load or generate a stable secret key stored in the global configuration directory."""
    # First check environment variable
    env_key = os.environ.get("FT_MCP_SECRET_KEY")
    if env_key:
        return env_key.encode("utf-8")

    # Fallback to a file in the global config dir
    try:

        config_dir = get_global_env_path().parent
        config_dir.mkdir(parents=True, exist_ok=True)
        secret_file = config_dir / ".mcp_secret"
        if secret_file.exists():
            return secret_file.read_bytes()
        else:
            key = secrets.token_bytes(32)
            secret_file.write_bytes(key)
            return key
    except Exception:  # pragma: no cover  # noqa: S110
        return b"emergency_fallback_secret_key_12345"


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to a URL-safe base64 string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    """Decode a URL-safe base64 string without padding."""
    padding = "=" * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)


def sign_jwt(payload: dict[str, Any], expires_in: int = 2592000) -> str:
    """Create a signed JWT token containing payload, expiring in expires_in seconds."""
    header = {"alg": "HS256", "typ": "JWT"}

    payload_copy = payload.copy()
    payload_copy["exp"] = int(time.time()) + expires_in

    header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(payload_copy).encode("utf-8"))

    message = f"{header_b64}.{payload_b64}".encode()
    secret = _get_secret_key()
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
        secret = _get_secret_key()
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


def get_api_key_from_headers(headers: Any) -> str:
    """Extract and validate the FMP API key from request headers."""
    # 1. Custom headers
    for name in _HEADER_NAMES:
        value = headers.get(name)
        if value:
            payload = verify_jwt(value.strip())
            if payload and "fmp_api_key" in payload:
                return str(payload["fmp_api_key"])
            return value.strip()

    # 2. Authorization header
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            payload = verify_jwt(token)
            if payload and "fmp_api_key" in payload:
                return str(payload["fmp_api_key"])
            return token

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
            payload = verify_jwt(value.strip())
            if payload and "fmp_api_key" in payload:
                return str(payload["fmp_api_key"])
            return value.strip()

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
                payload = verify_jwt(value.strip())
                if payload and "fmp_api_key" in payload:
                    return str(payload["fmp_api_key"])
                return value.strip()
    except RuntimeError:
        pass

    return ""


def get_fred_api_key_from_headers(headers: Any) -> str:
    """Extract and validate the (optional) FRED API key from request headers."""
    for name in _FRED_HEADER_NAMES:
        value = headers.get(name)
        if value:
            payload = verify_jwt(value.strip())
            if payload and "fred_api_key" in payload:
                return str(payload["fred_api_key"])
            return value.strip()

    # A signed FMP bearer token carries the FRED key alongside it, if the
    # caller authorized with one via the OAuth flow.
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            payload = verify_jwt(token)
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
            payload = verify_jwt(value.strip())
            if payload and "fred_api_key" in payload:
                return str(payload["fred_api_key"])
            return value.strip()

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
                payload = verify_jwt(value.strip())
                if payload and "fred_api_key" in payload:
                    return str(payload["fred_api_key"])
                return value.strip()
    except RuntimeError:
        pass

    return ""


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware to protect MCP endpoints in hosted deployments."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        path = request.url.path

        # CORS preflight — always allow so CORSMiddleware can respond
        if request.method == "OPTIONS":
            return await call_next(request)

        # Bypass authorization checks for discovery, oauth, health and diagnostic endpoints
        if (
            path.startswith("/.well-known")
            or path.startswith("/oauth")
            or path in {"/health", "/diagnostics"}
        ):
            return await call_next(request)

        # Protect MCP endpoints — SSE transport (/sse, /messages) and
        # streamable-HTTP transport (/mcp)
        if path == "/sse" or path.startswith("/messages") or path == "/mcp":
            # If the server has a global default API key, allow access
            if os.environ.get("FINANCIAL_MODELING_PREP_API_KEY"):
                return await call_next(request)

            api_key = get_api_key_from_request(request)
            if not api_key:
                base_url = str(request.base_url).rstrip("/")
                metadata_url = f"{base_url}/.well-known/oauth-protected-resource"

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

    @mcp.custom_route("/favicon.ico", methods=["GET"])
    async def favicon(request: Request) -> Response:
        if not _icon_path.exists():
            return Response(status_code=404)
        return Response(content=_icon_path.read_bytes(), media_type="image/png")

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
                "grant_types_supported": ["authorization_code"],
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
                "grant_types": ["authorization_code"],
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

        html = HTML_TEMPLATE.format(
            client_id=client_id,
            client_name=client_id or "MCP Client",
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            state=state,
        )
        return HTMLResponse(content=html)

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
            return HTMLResponse("FMP API Key is required", status_code=400)

        # Generate PKCE code signed by server. fred_api_key is optional — it
        # only unlocks the subset of Economics/FixedIncome tools that source
        # US labor/activity/rates data from FRED.
        payload = {
            "fmp_api_key": fmp_api_key,
            "fred_api_key": fred_api_key,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }
        # Code lasts 5 minutes
        auth_code = sign_jwt(payload, expires_in=300)

        redirect_url = f"{redirect_uri}?code={auth_code}"
        if state:
            redirect_url += f"&state={state}"

        return RedirectResponse(url=redirect_url, status_code=303)

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
        if grant_type != "authorization_code":
            return JSONResponse(
                {
                    "error": "unsupported_grant_type",
                    "error_description": "Only authorization_code is supported",
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

        # Decode & verify code
        payload = verify_jwt(str(code))
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
        if redirect_uri and payload.get("redirect_uri") != redirect_uri:
            return JSONResponse(
                {
                    "error": "invalid_grant",
                    "error_description": "Redirect URI mismatch",
                },
                status_code=400,
            )

        # PKCE validation
        code_challenge = payload.get("code_challenge")
        code_challenge_method = payload.get("code_challenge_method", "S256")

        if code_challenge:
            if not code_verifier:
                return JSONResponse(
                    {
                        "error": "invalid_grant",
                        "error_description": "code_verifier is required",
                    },
                    status_code=400,
                )

            if code_challenge_method == "S256":
                sha256 = hashlib.sha256(str(code_verifier).encode("utf-8")).digest()
                expected_challenge = _base64url_encode(sha256)
                if not hmac.compare_digest(expected_challenge, str(code_challenge)):
                    return JSONResponse(
                        {
                            "error": "invalid_grant",
                            "error_description": "PKCE S256 verification failed",
                        },
                        status_code=400,
                    )
            # plain
            elif not hmac.compare_digest(str(code_verifier), str(code_challenge)):
                return JSONResponse(
                    {
                        "error": "invalid_grant",
                        "error_description": "PKCE plain verification failed",
                    },
                    status_code=400,
                )

        # Issue JWT Access Token
        token_payload = {
            "fmp_api_key": payload.get("fmp_api_key"),
            "fred_api_key": payload.get("fred_api_key"),
            "client_id": client_id,
        }
        # Access token lasts 1 year
        access_token = sign_jwt(token_payload, expires_in=31536000)

        return JSONResponse(
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": 31536000,
                "scope": "financetoolkit",
            }
        )
