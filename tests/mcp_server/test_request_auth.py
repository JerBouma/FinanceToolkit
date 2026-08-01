"""Tests for per-request API key resolution (hosted vs local transports)."""

import fastmcp.server.dependencies as deps
from starlette.requests import Request

from financetoolkit.mcp_server import auth_model


def _make_request(query: str = "") -> Request:
    """Build a minimal Starlette request carrying the given query string."""
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": query.encode(),
        "headers": [],
    }
    return Request(scope)


def test_stdio_returns_empty(monkeypatch):
    """No active HTTP request (stdio / uvx) yields an empty string for env fallback."""

    def _raise():
        raise RuntimeError("no http request")

    monkeypatch.setattr(deps, "get_http_headers", lambda include=None: {})
    monkeypatch.setattr(deps, "get_http_request", _raise)
    assert auth_model.resolve_api_key() == ""


def test_custom_header(monkeypatch):
    """The X-FMP-API-Key header is resolved."""
    monkeypatch.setattr(
        deps, "get_http_headers", lambda include=None: {"x-fmp-api-key": "HKEY"}
    )
    monkeypatch.setattr(deps, "get_http_request", _make_request)
    assert auth_model.resolve_api_key() == "HKEY"


def test_authorization_bearer(monkeypatch):
    """An Authorization: Bearer token is resolved."""
    monkeypatch.setattr(
        deps,
        "get_http_headers",
        lambda include=None: {"authorization": "Bearer BKEY"},
    )
    monkeypatch.setattr(deps, "get_http_request", _make_request)
    assert auth_model.resolve_api_key() == "BKEY"


def test_query_param(monkeypatch):
    """The ?fmp_api_key query parameter (Claude.ai path) is resolved."""
    monkeypatch.setattr(deps, "get_http_headers", lambda include=None: {})
    monkeypatch.setattr(
        deps, "get_http_request", lambda: _make_request("fmp_api_key=QKEY")
    )
    assert auth_model.resolve_api_key() == "QKEY"


def test_query_param_alias(monkeypatch):
    """The ?api_key alias is resolved."""
    monkeypatch.setattr(deps, "get_http_headers", lambda include=None: {})
    monkeypatch.setattr(deps, "get_http_request", lambda: _make_request("api_key=AKEY"))
    assert auth_model.resolve_api_key() == "AKEY"


def test_header_beats_query(monkeypatch):
    """A header takes priority over a query parameter when both are present."""
    monkeypatch.setattr(
        deps, "get_http_headers", lambda include=None: {"x-fmp-api-key": "HKEY"}
    )
    monkeypatch.setattr(
        deps, "get_http_request", lambda: _make_request("fmp_api_key=QKEY")
    )
    assert auth_model.resolve_api_key() == "HKEY"


def test_fred_stdio_returns_empty(monkeypatch):
    """No active HTTP request (stdio / uvx) yields an empty string for the FRED key too."""

    def _raise():
        raise RuntimeError("no http request")

    monkeypatch.setattr(deps, "get_http_headers", lambda include=None: {})
    monkeypatch.setattr(deps, "get_http_request", _raise)
    assert auth_model.resolve_fred_api_key() == ""


def test_fred_custom_header(monkeypatch):
    """The X-FRED-API-Key header is resolved."""
    monkeypatch.setattr(
        deps, "get_http_headers", lambda include=None: {"x-fred-api-key": "FHKEY"}
    )
    monkeypatch.setattr(deps, "get_http_request", _make_request)
    assert auth_model.resolve_fred_api_key() == "FHKEY"


def test_fred_query_param(monkeypatch):
    """The ?fred_api_key query parameter is resolved."""
    monkeypatch.setattr(deps, "get_http_headers", lambda include=None: {})
    monkeypatch.setattr(
        deps, "get_http_request", lambda: _make_request("fred_api_key=FQKEY")
    )
    assert auth_model.resolve_fred_api_key() == "FQKEY"


def test_fred_key_absent_returns_empty_not_fmp_key(monkeypatch):
    """An FMP key alone must not leak into FRED key resolution."""
    monkeypatch.setattr(
        deps, "get_http_headers", lambda include=None: {"x-fmp-api-key": "HKEY"}
    )
    monkeypatch.setattr(deps, "get_http_request", _make_request)
    assert auth_model.resolve_fred_api_key() == ""


def test_fred_header_beats_query(monkeypatch):
    """A FRED header takes priority over a FRED query parameter when both are present."""
    monkeypatch.setattr(
        deps, "get_http_headers", lambda include=None: {"x-fred-api-key": "FHKEY"}
    )
    monkeypatch.setattr(
        deps, "get_http_request", lambda: _make_request("fred_api_key=FQKEY")
    )
    assert auth_model.resolve_fred_api_key() == "FHKEY"
