"""Tests for OAuth 2.1 authentication flow and middleware on the MCP server."""

import base64
import contextlib
import hashlib
import os
from collections.abc import Generator
from urllib.parse import parse_qs, urlsplit

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

# PKCE S256 pair used by the flows below.
VERIFIER = "challenge123-verifier-that-is-long-enough-for-pkce-0123456789"
CHALLENGE = (
    base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest())
    .rstrip(b"=")
    .decode("ascii")
)


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
        f"&code_challenge={CHALLENGE}&code_challenge_method=S256&state=xyz"
    )
    assert auth_get_resp.status_code == 200  # noqa
    assert "Authorize Access" in auth_get_resp.text
    assert "fmp_api_key" in auth_get_resp.text

    # 5. Test POST Authorize
    auth_post_resp = client.post(
        "/oauth/authorize",
        data={
            "fmp_api_key": "my_fmp_api_key",
            "client_id": client_id,
            "redirect_uri": "http://localhost/callback",
            "code_challenge": CHALLENGE,
            "code_challenge_method": "S256",
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

    # 6. Test Token Endpoint
    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": client_id,
            "redirect_uri": "http://localhost/callback",
            "code_verifier": VERIFIER,
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
        f"&code_challenge={CHALLENGE}&code_challenge_method=S256&state=xyz"
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
            "code_challenge": CHALLENGE,
            "code_challenge_method": "S256",
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
            "code_verifier": VERIFIER,
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
            "code_challenge": CHALLENGE,
            "code_challenge_method": "S256",
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
            "code_verifier": VERIFIER,
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

    # POST /messages/ rather than GET /sse, which would hang on the stream.
    jwt_token = sign_jwt({"fmp_api_key": "my-fmp-key"})

    # The X-FMP-API-KEY header should bypass the middleware and reach FastMCP.
    resp = client.post("/messages/", headers={"X-FMP-API-KEY": "my-key"})
    assert resp.status_code != 401  # noqa

    # We test with Bearer Token (should bypass middleware and proceed to FastMCP)
    resp = client.post("/messages/", headers={"Authorization": f"Bearer {jwt_token}"})
    assert resp.status_code != 401  # noqa


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


def test_the_authorization_page_escapes_its_parameters() -> None:
    """Query parameters are shown as text, never as markup."""
    mcp = FastMCP("TestEscapeServer")
    register_auth_routes(mcp)
    client = TestClient(mcp.sse_app(), base_url="http://127.0.0.1:8000")

    page = client.get(
        '/oauth/authorize?client_id="><script>alert(1)</script>'
        f"&redirect_uri=http://localhost/callback&code_challenge={CHALLENGE}"
        '&code_challenge_method=S256&state="><img src=x onerror=alert(1)>'
    )

    assert page.status_code == 200  # noqa
    assert "<script>alert(1)</script>" not in page.text
    assert "<img src=x" not in page.text
    assert "&lt;script&gt;" in page.text


def _oauth_client() -> TestClient:
    mcp = FastMCP("TestHardeningServer")
    register_auth_routes(mcp)
    app = mcp.sse_app()
    app.add_middleware(MCPAuthMiddleware)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def _authorize(client: TestClient, redirect_uri: str, **extra: str):
    data = {
        "fmp_api_key": "SECRETFMPKEY123",
        "fred_api_key": "SECRETFRED",
        "client_id": "client_x",
        "redirect_uri": redirect_uri,
        "code_challenge": CHALLENGE,
        "code_challenge_method": "S256",
        "state": "xyz",
    }
    data.update(extra)
    return client.post("/oauth/authorize", data=data, follow_redirects=False)


def test_the_authorization_code_does_not_carry_the_keys() -> None:
    client = _oauth_client()
    response = _authorize(client, "https://claude.ai/api/mcp/auth_callback")

    assert response.status_code == 303  # noqa
    location = response.headers["location"]
    code = parse_qs(urlsplit(location).query)["code"][0]
    assert "SECRET" not in location
    assert "." not in code
    for part in code.split("_") + code.split("-"):
        with contextlib.suppress(Exception):
            assert b"SECRET" not in base64.urlsafe_b64decode(part + "==")


def test_an_authorization_code_is_exchanged_once() -> None:
    client = _oauth_client()
    location = _authorize(client, "http://localhost/callback").headers["location"]
    code = parse_qs(urlsplit(location).query)["code"][0]
    exchange = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": "client_x",
        "redirect_uri": "http://localhost/callback",
        "code_verifier": VERIFIER,
    }

    first = client.post("/oauth/token", data=exchange)
    second = client.post("/oauth/token", data=exchange)

    assert first.status_code == 200  # noqa
    assert verify_jwt(first.json()["access_token"])["fred_api_key"] == "SECRETFRED"
    assert second.status_code == 400  # noqa
    assert second.json()["error"] == "invalid_grant"


def test_the_token_request_must_repeat_the_redirect_uri_and_verifier() -> None:
    client = _oauth_client()

    for missing in ("redirect_uri", "code_verifier"):
        location = _authorize(client, "http://localhost/callback").headers["location"]
        exchange = {
            "grant_type": "authorization_code",
            "code": parse_qs(urlsplit(location).query)["code"][0],
            "redirect_uri": "http://localhost/callback",
            "code_verifier": VERIFIER,
        }
        del exchange[missing]
        assert client.post("/oauth/token", data=exchange).status_code == 400  # noqa


def test_authorization_requires_a_safe_redirect_and_s256() -> None:
    client = _oauth_client()

    for redirect_uri in ("http://evil.example/cb", "javascript:alert(1)", ""):
        assert _authorize(client, redirect_uri).status_code == 400  # noqa
        page = client.get(
            "/oauth/authorize",
            params={
                "redirect_uri": redirect_uri,
                "code_challenge": CHALLENGE,
                "code_challenge_method": "S256",
            },
        )
        assert page.status_code == 400  # noqa

    assert (
        _authorize(client, "http://localhost/cb", code_challenge="").status_code
        == 400  # noqa
    )
    assert (
        _authorize(
            client, "http://localhost/cb", code_challenge_method="plain"
        ).status_code
        == 400  # noqa
    )
    assert _authorize(client, "http://127.0.0.1:3000/cb").status_code == 303  # noqa
    assert _authorize(client, "cursor://anysphere.cursor/cb").status_code == 303  # noqa


def test_the_consent_page_names_the_redirect_host_and_cannot_be_framed() -> None:
    client = _oauth_client()
    page = client.get(
        "/oauth/authorize",
        params={
            "client_id": "client_x",
            "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
            "code_challenge": CHALLENGE,
            "code_challenge_method": "S256",
        },
    )

    assert page.status_code == 200  # noqa
    assert "claude.ai" in page.text.split("<h1>")[0]
    assert page.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]
    assert client.get("/").headers["X-Frame-Options"] == "DENY"


def test_state_and_an_existing_query_are_encoded_into_the_redirect() -> None:
    client = _oauth_client()
    location = _authorize(
        client, "https://app.example/cb?x=1", state="s&code=INJECTED"
    ).headers["location"]

    query = parse_qs(urlsplit(location).query)
    assert location.startswith("https://app.example/cb?x=1&")
    assert query["x"] == ["1"]
    assert query["state"] == ["s&code=INJECTED"]
    assert len(query["code"]) == 1
    assert query["code"] != ["INJECTED"]


def test_an_expired_or_foreign_access_token_gets_invalid_token() -> None:
    client = _oauth_client()
    expired = sign_jwt({"fmp_api_key": "OLDKEY"}, expires_in=-10)

    for token in (expired, "a.b.c"):
        response = client.post(
            "/messages/", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401  # noqa
        assert 'error="invalid_token"' in response.headers["WWW-Authenticate"]

    # A raw API key as bearer token is still accepted.
    response = client.post("/messages/", headers={"Authorization": "Bearer rawkey"})
    assert response.status_code != 401  # noqa


def test_the_secret_file_is_private_and_read_once(monkeypatch, tmp_path):
    from financetoolkit.mcp_server import auth_model

    monkeypatch.delenv("FT_MCP_SECRET_KEY", raising=False)
    monkeypatch.setattr(auth_model, "_secret_key", None)
    monkeypatch.setattr(auth_model, "get_global_env_path", lambda: tmp_path / ".env")

    key = auth_model.get_secret_key()
    secret_file = tmp_path / ".mcp_secret"

    assert secret_file.read_bytes() == key
    assert secret_file.stat().st_mode & 0o777 == 0o600
    secret_file.write_bytes(b"changed")
    assert auth_model.get_secret_key() == key


def test_authorization_fields_are_bounded_and_the_challenge_is_s256_shaped() -> None:
    client = _oauth_client()

    assert (
        _authorize(client, "http://localhost/cb", state="s" * 1025).status_code == 400
    )  # noqa
    assert (
        _authorize(client, "http://localhost/cb", fmp_api_key="k" * 129).status_code
        == 400  # noqa
    )
    assert (
        _authorize(client, "https://a.example/" + "p" * 2048).status_code == 400  # noqa
    )
    assert (
        _authorize(client, "http://localhost/cb", code_challenge="abc").status_code
        == 400  # noqa
    )
    assert _authorize(client, "http://localhost/cb").status_code == 303  # noqa


def test_a_full_code_store_refuses_new_codes_instead_of_evicting(monkeypatch) -> None:
    from financetoolkit.mcp_server import auth_model

    monkeypatch.setattr(auth_model, "_MAX_PENDING_CODES", 1)
    monkeypatch.setattr(auth_model, "_auth_codes", {})
    client = _oauth_client()

    first = _authorize(client, "http://localhost/callback")
    assert first.status_code == 303  # noqa
    assert _authorize(client, "http://localhost/callback").status_code == 503  # noqa

    code = parse_qs(urlsplit(first.headers["location"]).query)["code"][0]
    exchange = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": "http://localhost/callback",
        "code_verifier": VERIFIER,
    }
    assert client.post("/oauth/token", data=exchange).status_code == 200  # noqa


def test_private_use_schemes_without_a_host_are_accepted() -> None:
    from financetoolkit.mcp_server.auth_model import validate_redirect_uri

    assert validate_redirect_uri("com.example.app:/oauth2redirect") == ""
    assert validate_redirect_uri("cursor://anysphere.cursor/cb") == ""
    for refused in ("javascript:alert(1)", "data:text/html,x", "file:///etc/passwd"):
        assert validate_redirect_uri(refused)


def test_an_existing_secret_is_kept_and_a_short_one_is_replaced(monkeypatch, tmp_path):
    from financetoolkit.mcp_server import auth_model

    monkeypatch.delenv("FT_MCP_SECRET_KEY", raising=False)
    monkeypatch.setattr(auth_model, "get_global_env_path", lambda: tmp_path / ".env")
    secret_file = tmp_path / ".mcp_secret"

    # A file this process did not create is read as it is, never chmodded.
    secret_file.write_bytes(b"x" * 40)
    secret_file.chmod(0o640)
    monkeypatch.setattr(auth_model, "_secret_key", None)
    assert auth_model.get_secret_key() == b"x" * 40
    assert secret_file.stat().st_mode & 0o777 == 0o640

    secret_file.write_bytes(b"")
    monkeypatch.setattr(auth_model, "_secret_key", None)
    key = auth_model.get_secret_key()
    assert len(key) == 32 and secret_file.read_bytes() == key  # noqa: PLR2004
