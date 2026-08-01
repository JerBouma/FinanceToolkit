"""Tests for OAuth 2.1 authentication flow and middleware on the MCP server."""

import base64
import hashlib
import os
from collections.abc import Generator

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.testclient import TestClient

from financetoolkit.mcp_server.auth_model import (
    MCPAuthMiddleware,
    get_api_key_from_request,
    register_auth_routes,
    sign_jwt,
    verify_jwt,
)

_JWT_PART_COUNT = 3


@pytest.fixture(autouse=True)
def setup_env() -> Generator[None, None, None]:
    """Ensure FINANCIAL_MODELING_PREP_API_KEY is not set in the environment by default for tests."""
    old_key = os.environ.get("FINANCIAL_MODELING_PREP_API_KEY")
    if old_key is not None:
        del os.environ["FINANCIAL_MODELING_PREP_API_KEY"]
    yield
    if old_key is not None:
        os.environ["FINANCIAL_MODELING_PREP_API_KEY"] = old_key


def test_jwt_signing_and_verification() -> None:
    """Test that JWT encoding and decoding works with signatures and expirations."""
    payload = {"fmp_api_key": "test_fmp_key", "user_id": 42}

    # Test basic sign & verify
    token = sign_jwt(payload, expires_in=60)
    assert token is not None
    assert len(token.split(".")) == _JWT_PART_COUNT

    decoded = verify_jwt(token)
    assert decoded is not None
    assert decoded["fmp_api_key"] == "test_fmp_key"
    assert decoded["user_id"] == 42  # noqa

    # Test expiration
    expired_token = sign_jwt(payload, expires_in=-10)
    assert verify_jwt(expired_token) is None

    # Test invalid token format
    assert verify_jwt("invalid.token") is None
    assert verify_jwt("not_even_dots") is None
    assert verify_jwt("three.dots.but_bad_sig") is None


def test_get_api_key_from_request() -> None:
    """Test key extraction from various parts of a Starlette Request."""

    def make_req(headers: dict[str, str] | None = None, query: str = "") -> Request:
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "path": "/test",
            "raw_path": b"/test",
            "query_string": query.encode(),
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
            ],
        }
        return Request(scope)

    # 1. Custom header
    req = make_req(headers={"X-FMP-API-Key": "my-header-key"})
    assert get_api_key_from_request(req) == "my-header-key"

    # 2. Authorization Bearer (raw)
    req = make_req(headers={"Authorization": "Bearer my-bearer-key"})
    assert get_api_key_from_request(req) == "my-bearer-key"

    # 3. Authorization Bearer (JWT)
    jwt_token = sign_jwt({"fmp_api_key": "my-jwt-key"})
    req = make_req(headers={"Authorization": f"Bearer {jwt_token}"})
    assert get_api_key_from_request(req) == "my-jwt-key"

    # 4. Query param
    req = make_req(query="fmp_api_key=my-query-key")
    assert get_api_key_from_request(req) == "my-query-key"

    # 5. No fallback to environment key directly in get_api_key_from_request
    req = make_req()
    assert get_api_key_from_request(req) == ""


def test_oauth_routes_and_flow() -> None:
    """Perform a full test of the OAuth 2.1 authorization and token endpoints."""
    mcp = FastMCP("TestOAuthServer")
    register_auth_routes(mcp)
    app = mcp.sse_app()
    app.add_middleware(MCPAuthMiddleware)
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    # 1. Test Protected Resource Metadata Endpoint
    resp = client.get("/.well-known/oauth-protected-resource")
    assert resp.status_code == 200  # noqa
    data = resp.json()
    assert "authorization_servers" in data
    assert "bearer_methods_supported" in data

    # 2. Test Authorization Server Metadata Endpoint
    resp = client.get("/.well-known/oauth-authorization-server")
    assert resp.status_code == 200  # noqa
    data = resp.json()
    assert data["authorization_endpoint"].endswith("/oauth/authorize")
    assert data["token_endpoint"].endswith("/oauth/token")
    assert "S256" in data["code_challenge_methods_supported"]

    # 3. Test Dynamic Client Registration Endpoint
    reg_resp = client.post(
        "/oauth/register",
        json={
            "client_name": "AI Agent CLI",
            "redirect_uris": ["http://localhost/callback"],
        },
    )
    assert reg_resp.status_code == 201  # noqa
    reg_data = reg_resp.json()
    assert "client_id" in reg_data
    client_id = reg_data["client_id"]

    # 4. Test GET Authorize Page
    auth_get_resp = client.get(
        f"/oauth/authorize?client_id={client_id}&redirect_uri=http://localhost/callback"
        "&code_challenge=challenge123&code_challenge_method=plain&state=xyz"
    )
    assert auth_get_resp.status_code == 200  # noqa
    assert "Authorize Access" in auth_get_resp.text
    assert "fmp_api_key" in auth_get_resp.text

    # 5. Test POST Authorize (Plain Code Flow)
    auth_post_resp = client.post(
        "/oauth/authorize",
        data={
            "fmp_api_key": "my_fmp_api_key",
            "client_id": client_id,
            "redirect_uri": "http://localhost/callback",
            "code_challenge": "challenge123",
            "code_challenge_method": "plain",
            "state": "xyz",
        },
        follow_redirects=False,
    )
    assert auth_post_resp.status_code == 303  # noqa
    loc = auth_post_resp.headers["location"]
    assert loc.startswith("http://localhost/callback?code=")
    assert "state=xyz" in loc

    # Extract authorization code
    query_parts = loc.split("?")[1].split("&")
    auth_code = next(p.split("=")[1] for p in query_parts if p.startswith("code="))

    # 6. Test Token Endpoint (Plain Exchange)
    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": client_id,
            "redirect_uri": "http://localhost/callback",
            "code_verifier": "challenge123",
        },
    )
    assert token_resp.status_code == 200  # noqa
    token_data = token_resp.json()
    assert "access_token" in token_data
    access_token = token_data["access_token"]
    assert token_data["token_type"] == "Bearer"

    # Verify access token yields the original FMP key
    payload = verify_jwt(access_token)
    assert payload is not None
    assert payload["fmp_api_key"] == "my_fmp_api_key"

    # 7. Test S256 PKCE Flow
    code_verifier = "my_cryptographically_secure_code_verifier_123456"
    sha256_hash = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = base64.urlsafe_b64encode(sha256_hash).rstrip(b"=").decode("ascii")

    auth_post_s256 = client.post(
        "/oauth/authorize",
        data={
            "fmp_api_key": "my_s256_fmp_api_key",
            "client_id": client_id,
            "redirect_uri": "http://localhost/callback",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": "state123",
        },
        follow_redirects=False,
    )
    assert auth_post_s256.status_code == 303  # noqa
    loc_s256 = auth_post_s256.headers["location"]
    query_parts_s256 = loc_s256.split("?")[1].split("&")
    auth_code_s256 = next(
        p.split("=")[1] for p in query_parts_s256 if p.startswith("code=")
    )

    # 8. Test Token Endpoint (S256 Exchange)
    token_resp_s256 = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code_s256,
            "client_id": client_id,
            "redirect_uri": "http://localhost/callback",
            "code_verifier": code_verifier,
        },
    )
    assert token_resp_s256.status_code == 200  # noqa
    access_token_s256 = token_resp_s256.json()["access_token"]
    payload_s256 = verify_jwt(access_token_s256)
    assert payload_s256 is not None
    assert payload_s256["fmp_api_key"] == "my_s256_fmp_api_key"


def test_oauth_flow_with_fred_key() -> None:
    """FRED API key, when submitted on the authorize form, round-trips through the
    auth code and into the final access token alongside the FMP key."""
    mcp = FastMCP("TestOAuthFredServer")
    register_auth_routes(mcp)
    app = mcp.sse_app()
    app.add_middleware(MCPAuthMiddleware)
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    auth_get_resp = client.get(
        "/oauth/authorize?client_id=client_fred&redirect_uri=http://localhost/callback"
        "&code_challenge=challenge123&code_challenge_method=plain&state=xyz"
    )
    assert auth_get_resp.status_code == 200  # noqa
    assert "fred_api_key" in auth_get_resp.text
    assert "FRED API Key" in auth_get_resp.text

    auth_post_resp = client.post(
        "/oauth/authorize",
        data={
            "fmp_api_key": "my_fmp_api_key",
            "fred_api_key": "my_fred_api_key",
            "client_id": "client_fred",
            "redirect_uri": "http://localhost/callback",
            "code_challenge": "challenge123",
            "code_challenge_method": "plain",
            "state": "xyz",
        },
        follow_redirects=False,
    )
    assert auth_post_resp.status_code == 303  # noqa
    loc = auth_post_resp.headers["location"]
    auth_code = next(
        p.split("=")[1] for p in loc.split("?")[1].split("&") if p.startswith("code=")
    )

    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": "client_fred",
            "redirect_uri": "http://localhost/callback",
            "code_verifier": "challenge123",
        },
    )
    assert token_resp.status_code == 200  # noqa
    access_token = token_resp.json()["access_token"]

    payload = verify_jwt(access_token)
    assert payload is not None
    assert payload["fmp_api_key"] == "my_fmp_api_key"
    assert payload["fred_api_key"] == "my_fred_api_key"


def test_oauth_flow_without_fred_key_defaults_empty() -> None:
    """FRED API key is optional — omitting it from the authorize form must not
    block authorization, and it should resolve to an empty string in the token."""
    mcp = FastMCP("TestOAuthNoFredServer")
    register_auth_routes(mcp)
    app = mcp.sse_app()
    app.add_middleware(MCPAuthMiddleware)
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    auth_post_resp = client.post(
        "/oauth/authorize",
        data={
            "fmp_api_key": "my_fmp_api_key",
            "client_id": "client_no_fred",
            "redirect_uri": "http://localhost/callback",
            "code_challenge": "challenge123",
            "code_challenge_method": "plain",
            "state": "xyz",
        },
        follow_redirects=False,
    )
    assert auth_post_resp.status_code == 303  # noqa
    loc = auth_post_resp.headers["location"]
    auth_code = next(
        p.split("=")[1] for p in loc.split("?")[1].split("&") if p.startswith("code=")
    )

    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": "client_no_fred",
            "redirect_uri": "http://localhost/callback",
            "code_verifier": "challenge123",
        },
    )
    assert token_resp.status_code == 200  # noqa
    payload = verify_jwt(token_resp.json()["access_token"])
    assert payload is not None
    assert payload["fmp_api_key"] == "my_fmp_api_key"
    assert payload["fred_api_key"] == ""


def test_auth_middleware_protection() -> None:
    """Test that the MCPAuthMiddleware properly intercepts and restricts access."""
    mcp = FastMCP("TestMiddlewareServer")
    register_auth_routes(mcp)
    app = mcp.sse_app()
    app.add_middleware(MCPAuthMiddleware)
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    # Public endpoints should be accessible
    assert (
        client.get("/.well-known/oauth-protected-resource").status_code == 200  # noqa
    )
    assert (
        client.get("/.well-known/oauth-authorization-server").status_code == 200  # noqa
    )

    # SSE endpoint should be blocked with 401 when no token is provided
    sse_resp = client.get("/sse")
    assert sse_resp.status_code == 401  # noqa
    assert "WWW-Authenticate" in sse_resp.headers
    assert "resource_metadata=" in sse_resp.headers["WWW-Authenticate"]

    # SSE/Messages endpoints should allow access when valid headers or token are present.
    # We test with POST /messages/ instead of GET /sse to prevent the test client from hanging on a streaming connection.
    jwt_token = sign_jwt({"fmp_api_key": "my-fmp-key"})

    # We test with X-FMP-API-KEY header (should bypass middleware and proceed to FastMCP,
    # ielding 404 or 450 instead of 401)
    resp = client.post("/messages/", headers={"X-FMP-API-KEY": "my-key"})
    assert resp.status_code != 401  # noqa

    # We test with Bearer Token (should bypass middleware and proceed to FastMCP)
    resp = client.post("/messages/", headers={"Authorization": f"Bearer {jwt_token}"})
    assert resp.status_code != 401  # noqa
