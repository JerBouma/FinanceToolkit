"""Tests for OAuth 2.1 authentication flow and middleware on the MCP server."""

import base64
import hashlib
import os
import pathlib
import time
from collections.abc import Generator
from urllib.parse import parse_qs, urlsplit

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.testclient import TestClient

from financetoolkit.mcp_server import auth_model, token_store_model
from financetoolkit.mcp_server.auth_model import (
    MCPAuthMiddleware,
    get_api_key_from_request,
    get_fred_api_key_from_request,
    register_auth_routes,
    sign_jwt,
    verify_jwt,
)

_JWT_PART_COUNT = 3

VERIFIER = "my_cryptographically_secure_code_verifier_123456"


@pytest.fixture(autouse=True)
def setup_env(monkeypatch, tmp_path) -> Generator[None, None, None]:
    """No server-wide FMP key, a fresh grant store and a fixed secret for every test."""
    monkeypatch.delenv("FINANCIAL_MODELING_PREP_API_KEY", raising=False)
    monkeypatch.delenv("FT_MCP_ACCEPT_LEGACY_TOKENS", raising=False)
    monkeypatch.setenv("FT_MCP_SECRET_KEY", "test-secret-" + "x" * 40)
    monkeypatch.setenv("FT_MCP_AUTH_DB", str(tmp_path / "mcp_auth.db"))
    monkeypatch.setattr(auth_model, "_token_store", None)
    yield


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


def _make_request(headers: dict[str, str] | None = None, query: str = "") -> Request:
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


def _client() -> TestClient:
    mcp = FastMCP("TestOAuthServer")
    register_auth_routes(mcp)
    app = mcp.sse_app()
    app.add_middleware(MCPAuthMiddleware)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _authorize(
    client: TestClient,
    fmp_api_key: str = "my_fmp_api_key",
    fred_api_key: str | None = None,
    client_id: str = "client_test",
    redirect_uri: str = "http://localhost/callback",
) -> str:
    """Run the authorization step and return the code from the redirect."""
    data = {
        "fmp_api_key": fmp_api_key,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": _challenge(VERIFIER),
        "code_challenge_method": "S256",
        "state": "xyz",
    }
    if fred_api_key is not None:
        data["fred_api_key"] = fred_api_key
    response = client.post("/oauth/authorize", data=data, follow_redirects=False)
    assert response.status_code == 303  # noqa
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["state"] == ["xyz"]
    return query["code"][0]


def _exchange(
    client: TestClient, code: str, client_id: str = "client_test", **overrides: str
):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "redirect_uri": "http://localhost/callback",
        "code_verifier": VERIFIER,
    }
    data.update(overrides)
    return client.post("/oauth/token", data=data)


def _bearer(token: str) -> Request:
    return _make_request(headers={"Authorization": f"Bearer {token}"})


def test_get_api_key_from_request() -> None:
    """Test key extraction from various parts of a Starlette Request."""
    # 1. Custom header
    assert (
        get_api_key_from_request(
            _make_request(headers={"X-FMP-API-Key": "my-header-key"})
        )
        == "my-header-key"
    )

    # 2. Authorization Bearer with the API key itself
    assert get_api_key_from_request(_bearer("my-bearer-key")) == "my-bearer-key"

    # 3. Query param
    assert (
        get_api_key_from_request(_make_request(query="fmp_api_key=my-query-key"))
        == "my-query-key"
    )

    # 4. No fallback to environment key directly in get_api_key_from_request
    assert get_api_key_from_request(_make_request()) == ""

    # 5. An unknown token of this server is not taken for an API key.
    assert get_api_key_from_request(_bearer("fta_unknown")) == ""


def test_legacy_jwt_access_tokens_are_only_read_when_allowed(monkeypatch) -> None:
    """Legacy JWTs carry the keys in readable form; they work only while allowed."""
    legacy = sign_jwt({"fmp_api_key": "my-jwt-key", "fred_api_key": "my-fred"})

    assert get_api_key_from_request(_bearer(legacy)) == ""

    monkeypatch.setenv("FT_MCP_ACCEPT_LEGACY_TOKENS", "1")
    assert get_api_key_from_request(_bearer(legacy)) == "my-jwt-key"
    assert get_fred_api_key_from_request(_bearer(legacy)) == "my-fred"


def test_oauth_routes_and_flow() -> None:
    """Perform a full test of the OAuth 2.1 authorization and token endpoints."""
    client = _client()

    # 1. Protected Resource Metadata
    resp = client.get("/.well-known/oauth-protected-resource")
    assert resp.status_code == 200  # noqa
    assert "authorization_servers" in resp.json()

    # 2. Authorization Server Metadata
    data = client.get("/.well-known/oauth-authorization-server").json()
    assert data["authorization_endpoint"].endswith("/oauth/authorize")
    assert data["token_endpoint"].endswith("/oauth/token")
    assert data["revocation_endpoint"].endswith("/oauth/revoke")
    assert data["code_challenge_methods_supported"] == ["S256"]
    assert "refresh_token" in data["grant_types_supported"]

    # 3. Dynamic Client Registration
    reg_resp = client.post(
        "/oauth/register",
        json={
            "client_name": "AI Agent CLI",
            "redirect_uris": ["http://localhost/callback"],
        },
    )
    assert reg_resp.status_code == 201  # noqa
    client_id = reg_resp.json()["client_id"]

    # 4. The authorization page
    page = client.get(
        f"/oauth/authorize?client_id={client_id}&redirect_uri=http://localhost/callback"
        f"&code_challenge={_challenge(VERIFIER)}&code_challenge_method=S256&state=xyz"
    )
    assert page.status_code == 200  # noqa
    assert "Authorize Access" in page.text
    assert "fmp_api_key" in page.text

    # 5. The code is an opaque reference: the key is not in it.
    code = _authorize(client, "my_fmp_api_key", client_id=client_id)
    assert code.startswith("ftc_")
    assert "my_fmp_api_key" not in code

    # 6. The token exchange returns short-lived opaque tokens and a refresh token.
    token_resp = _exchange(client, code, client_id=client_id)
    assert token_resp.status_code == 200  # noqa
    assert token_resp.headers["cache-control"] == "no-store"
    token_data = token_resp.json()
    assert token_data["token_type"] == "Bearer"
    assert token_data["expires_in"] == 3600  # noqa
    access_token = token_data["access_token"]
    assert access_token.startswith("fta_")
    assert token_data["refresh_token"].startswith("ftr_")
    assert "." not in access_token
    assert verify_jwt(access_token) is None

    # 7. The access token resolves to the key on the server.
    assert get_api_key_from_request(_bearer(access_token)) == "my_fmp_api_key"


def test_oauth_flow_with_fred_key() -> None:
    """A FRED API key entered on the authorize form is available behind the access token."""
    client = _client()

    page = client.get(
        "/oauth/authorize?client_id=client_fred&redirect_uri=http://localhost/callback"
        f"&code_challenge={_challenge(VERIFIER)}&code_challenge_method=S256&state=xyz"
    )
    assert "fred_api_key" in page.text
    assert "FRED API Key" in page.text

    code = _authorize(client, fred_api_key="my_fred_api_key", client_id="client_fred")
    access_token = _exchange(client, code, client_id="client_fred").json()[
        "access_token"
    ]

    assert get_api_key_from_request(_bearer(access_token)) == "my_fmp_api_key"
    assert get_fred_api_key_from_request(_bearer(access_token)) == "my_fred_api_key"


def test_oauth_flow_without_fred_key_defaults_empty() -> None:
    """The FRED API key is optional: without it, the access token has no FRED key."""
    client = _client()
    access_token = _exchange(client, _authorize(client)).json()["access_token"]

    assert get_api_key_from_request(_bearer(access_token)) == "my_fmp_api_key"
    assert get_fred_api_key_from_request(_bearer(access_token)) == ""


def test_the_keys_are_stored_encrypted_and_tokens_only_as_hashes() -> None:
    """Neither the API key nor any token appears in readable form in the database."""
    client = _client()
    token_data = _exchange(client, _authorize(client, "fmp_secret_value_123")).json()

    stored = pathlib.Path(os.environ["FT_MCP_AUTH_DB"]).read_bytes()
    assert b"fmp_secret_value_123" not in stored
    assert token_data["access_token"].encode() not in stored
    assert token_data["refresh_token"].encode() not in stored


def test_an_authorization_code_works_once() -> None:
    """Reusing a code fails and revokes the grant, including the tokens it gave."""
    client = _client()
    code = _authorize(client)
    access_token = _exchange(client, code).json()["access_token"]

    assert _exchange(client, code).status_code == 400  # noqa
    assert get_api_key_from_request(_bearer(access_token)) == ""


def test_the_token_request_must_match_the_authorization() -> None:
    """A wrong verifier, client or redirect URI gets no tokens."""
    client = _client()

    assert (
        _exchange(client, _authorize(client), code_verifier="wrong").status_code == 400
    )  # noqa
    assert (
        _exchange(client, _authorize(client), client_id="client_other").status_code
        == 400
    )  # noqa
    assert (
        _exchange(client, _authorize(client), redirect_uri="http://evil/cb").status_code
        == 400
    )  # noqa


def test_refresh_tokens_rotate_and_a_reused_one_revokes_the_grant() -> None:
    """Every refresh gives a new pair; replaying an old refresh token ends the grant."""
    client = _client()
    first = _exchange(client, _authorize(client)).json()

    second = client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
            "client_id": "client_test",
        },
    )
    assert second.status_code == 200  # noqa
    second_data = second.json()
    assert second_data["refresh_token"] != first["refresh_token"]
    assert (
        get_api_key_from_request(_bearer(second_data["access_token"]))
        == "my_fmp_api_key"
    )

    replay = client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
            "client_id": "client_test",
        },
    )
    assert replay.status_code == 400  # noqa
    assert get_api_key_from_request(_bearer(second_data["access_token"])) == ""


def test_revoking_a_token_ends_the_grant() -> None:
    """After revocation the access token is rejected with invalid_token."""
    client = _client()
    token_data = _exchange(client, _authorize(client)).json()

    assert (
        client.post(
            "/oauth/revoke", data={"token": token_data["refresh_token"]}
        ).status_code
        == 200
    )  # noqa
    assert get_api_key_from_request(_bearer(token_data["access_token"])) == ""

    resp = client.post(
        "/messages/", headers={"Authorization": f"Bearer {token_data['access_token']}"}
    )
    assert resp.status_code == 401  # noqa
    assert 'error="invalid_token"' in resp.headers["WWW-Authenticate"]

    # Unknown tokens are answered the same way, so the endpoint reveals nothing.
    assert (
        client.post("/oauth/revoke", data={"token": "ftr_unknown"}).status_code == 200
    )  # noqa


def test_access_tokens_expire(monkeypatch) -> None:
    """An access token stops working after its lifetime."""
    client = _client()
    access_token = _exchange(client, _authorize(client)).json()["access_token"]

    later = time.time() + 3601
    monkeypatch.setattr(token_store_model.time, "time", lambda: later)
    assert get_api_key_from_request(_bearer(access_token)) == ""


def test_the_authorization_request_requires_s256_and_a_safe_redirect() -> None:
    """Plain PKCE, a missing challenge and unsafe redirect URIs are refused."""
    client = _client()
    base = "/oauth/authorize?client_id=c&state=s"

    assert (
        client.get(
            f"{base}&redirect_uri=http://localhost/cb&code_challenge=abc&code_challenge_method=plain"
        ).status_code
        == 400
    )  # noqa
    assert (
        client.get(f"{base}&redirect_uri=http://localhost/cb").status_code == 400
    )  # noqa
    assert (
        client.get(
            f"{base}&redirect_uri=javascript:alert(1)&code_challenge=abc&code_challenge_method=S256"
        ).status_code
        == 400
    )  # noqa


def test_the_authorization_page_escapes_its_parameters() -> None:
    """Query parameters are shown as text, never as markup."""
    client = _client()
    page = client.get(
        '/oauth/authorize?client_id="><script>alert(1)</script>&redirect_uri=http://localhost/cb'
        '&code_challenge=abc&code_challenge_method=S256&state="><img src=x onerror=alert(1)>'
    )
    assert page.status_code == 200  # noqa
    assert "<script>alert(1)</script>" not in page.text
    assert "<img src=x" not in page.text


def test_the_redirect_keeps_the_query_of_the_redirect_uri() -> None:
    """A redirect URI with its own query gets the code added to it."""
    client = _client()
    response = client.post(
        "/oauth/authorize",
        data={
            "fmp_api_key": "key",
            "client_id": "c",
            "redirect_uri": "https://example.com/cb?session=1",
            "code_challenge": _challenge(VERIFIER),
            "code_challenge_method": "S256",
            "state": "a b&c",
        },
        follow_redirects=False,
    )
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["session"] == ["1"]
    assert query["state"] == ["a b&c"]
    assert query["code"][0].startswith("ftc_")


def test_get_secret_key_has_no_built_in_fallback(monkeypatch, tmp_path) -> None:
    """Without FT_MCP_SECRET_KEY and a writable config directory, the server refuses."""
    monkeypatch.delenv("FT_MCP_SECRET_KEY", raising=False)
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monkeypatch.setattr(
        auth_model, "get_global_env_path", lambda: blocked / "sub" / ".env"
    )

    with pytest.raises(RuntimeError, match="FT_MCP_SECRET_KEY"):
        auth_model.get_secret_key()


def test_a_hosted_server_needs_a_secret(monkeypatch) -> None:
    """require_hosted_secret refuses a missing or short FT_MCP_SECRET_KEY."""
    monkeypatch.delenv("FT_MCP_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit):
        auth_model.require_hosted_secret()

    monkeypatch.setenv("FT_MCP_SECRET_KEY", "short")
    with pytest.raises(SystemExit):
        auth_model.require_hosted_secret()

    monkeypatch.setenv("FT_MCP_SECRET_KEY", "x" * 32)
    auth_model.require_hosted_secret()


def test_auth_middleware_protection() -> None:
    """Test that the MCPAuthMiddleware properly intercepts and restricts access."""
    client = _client()

    # Public endpoints should be accessible
    assert (
        client.get("/.well-known/oauth-protected-resource").status_code == 200
    )  # noqa
    assert (
        client.get("/.well-known/oauth-authorization-server").status_code == 200
    )  # noqa

    # SSE endpoint should be blocked with 401 when no token is provided
    sse_resp = client.get("/sse")
    assert sse_resp.status_code == 401  # noqa
    assert "resource_metadata=" in sse_resp.headers["WWW-Authenticate"]
    assert "invalid_token" not in sse_resp.headers["WWW-Authenticate"]

    # POST /messages/ rather than GET /sse, which would hang on the stream.
    # The X-FMP-API-KEY header should bypass the middleware and reach FastMCP.
    assert (
        client.post("/messages/", headers={"X-FMP-API-KEY": "my-key"}).status_code
        != 401
    )  # noqa

    # A valid access token passes.
    access_token = _exchange(client, _authorize(client)).json()["access_token"]
    resp = client.post(
        "/messages/", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert resp.status_code != 401  # noqa

    # A legacy JWT is not taken for an API key: the client is told to sign in again.
    legacy = sign_jwt({"fmp_api_key": "my-fmp-key"})
    resp = client.post("/messages/", headers={"Authorization": f"Bearer {legacy}"})
    assert resp.status_code == 401  # noqa


def test_root_page_and_favicons_are_served_without_authorization() -> None:
    """Favicon lookups read the root page and the small icon sizes, without a token."""
    mcp = FastMCP("TestIconServer")
    register_auth_routes(mcp)
    app = mcp.streamable_http_app()
    app.add_middleware(MCPAuthMiddleware)
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    root = client.get("/")
    assert root.status_code == 200  # noqa
    assert "text/html" in root.headers["content-type"]
    assert 'href="/favicon-32.png"' in root.text
    assert 'rel="apple-touch-icon"' in root.text

    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200  # noqa
    assert favicon.headers["content-type"] == "image/x-icon"
    # The ICO header: reserved 0, type 1 (icon), three images (16, 32 and 48 pixels).
    assert favicon.content[:6] == b"\x00\x00\x01\x00\x03\x00"

    for route in ("/favicon-32.png", "/apple-touch-icon.png", "/oauth/icon"):
        response = client.get(route)
        assert response.status_code == 200  # noqa
        assert response.content.startswith(b"\x89PNG")

    # The MCP endpoint itself stays protected.
    assert client.post("/mcp").status_code == 401  # noqa


def test_a_browser_opening_the_mcp_url_gets_the_setup_page() -> None:
    """A browser gets setup instructions at /mcp; MCP clients still get the OAuth challenge."""
    mcp = FastMCP("TestSetupPageServer")
    register_auth_routes(mcp)
    app = mcp.streamable_http_app()
    app.add_middleware(MCPAuthMiddleware)
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    page = client.get(
        "/mcp", headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}
    )
    assert page.status_code == 200  # noqa
    assert "Add custom connector" in page.text
    assert "https://www.jeroenbouma.com/projects/financetoolkit/mcp" in page.text

    # An MCP client asking for an event stream, or posting JSON, is still challenged.
    stream = client.get("/mcp", headers={"Accept": "text/event-stream"})
    assert stream.status_code == 401  # noqa
    assert "resource_metadata" in stream.headers["WWW-Authenticate"]
    assert client.post("/mcp", json={}).status_code == 401  # noqa
