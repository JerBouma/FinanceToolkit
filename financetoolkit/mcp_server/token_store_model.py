"""Server-side storage of the OAuth grants of a hosted MCP server.

A grant holds the API keys a user entered on the authorization page, encrypted with a key
derived from the server secret. Clients only ever receive opaque random tokens that refer to
a grant; the database stores a SHA-256 hash of each token, never the token itself. A leaked
token therefore reveals nothing, a copy of the database without the server secret reveals
nothing, and revoking a grant removes its keys.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

# An authorization code is exchanged within seconds; five minutes leaves room for a slow client.
CODE_LIFETIME = 300

# Access tokens are short-lived: a leaked one stops working within the hour.
ACCESS_TOKEN_LIFETIME = 3600

# A refresh token is replaced on every use. A grant whose refresh token goes unused for
# 90 days expires, and its keys are deleted.
REFRESH_TOKEN_LIFETIME = 90 * 86400

# The prefixes make the tokens recognisable, so an invalid or expired one is answered with a
# 401 (which makes the client refresh or sign in again) instead of being taken for an API key.
TOKEN_PREFIXES = {"code": "ftc_", "access": "fta_", "refresh": "ftr_"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS grants (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    keys BLOB NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    hash TEXT PRIMARY KEY,
    grant_id TEXT NOT NULL REFERENCES grants(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    expires REAL NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    meta TEXT
);
CREATE INDEX IF NOT EXISTS tokens_grant ON tokens(grant_id);
CREATE INDEX IF NOT EXISTS tokens_expires ON tokens(expires);
"""


def _hash(token: str) -> str:
    """The SHA-256 hash under which a token is stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_issued_token(value: str) -> bool:
    """Whether a value has the form of a token this server issues."""
    return value.startswith(tuple(TOKEN_PREFIXES.values()))


class TokenStore:
    """
    The grants and tokens of a hosted MCP server, in a SQLite database.

    Args:
        path (str | Path): The database file. It is created when it does not exist.
        secret (bytes): The server secret. The key that encrypts the API keys is derived
            from it, so the same secret is needed to read them back.
    """

    def __init__(self, path: str | Path, secret: bytes) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A separate key for encryption, so the secret itself is never used for two purposes.
        encryption_key = hmac.new(
            secret, b"financetoolkit-mcp-api-keys", hashlib.sha256
        ).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(encryption_key))
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
        # The database holds encrypted keys and token hashes only, but nobody else needs to read it.
        os.chmod(self.path, 0o600)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        """A connection inside a transaction, committed on success and closed afterwards."""
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _issue(
        self,
        connection: sqlite3.Connection,
        grant_id: str,
        kind: str,
        lifetime: int,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Create a token of the given kind for a grant and store its hash."""
        token = TOKEN_PREFIXES[kind] + secrets.token_urlsafe(32)
        connection.execute(
            "INSERT INTO tokens (hash, grant_id, kind, expires, meta) VALUES (?, ?, ?, ?, ?)",
            (
                _hash(token),
                grant_id,
                kind,
                time.time() + lifetime,
                json.dumps(meta) if meta else None,
            ),
        )
        return token

    def _purge(self, connection: sqlite3.Connection) -> None:
        """Delete expired tokens, and the grants (with their keys) that have no token left."""
        connection.execute("DELETE FROM tokens WHERE expires < ?", (time.time(),))
        connection.execute(
            "DELETE FROM grants WHERE id NOT IN (SELECT DISTINCT grant_id FROM tokens)"
        )

    def _delete_grant(self, connection: sqlite3.Connection, grant_id: str) -> None:
        """Delete a grant, its keys and every token that refers to it."""
        connection.execute("DELETE FROM tokens WHERE grant_id = ?", (grant_id,))
        connection.execute("DELETE FROM grants WHERE id = ?", (grant_id,))

    def create_authorization_code(
        self,
        keys: dict[str, str],
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
    ) -> str:
        """
        Store the keys a user entered in a new grant and issue an authorization code for it.

        Args:
            keys (dict[str, str]): The API keys, e.g. {"fmp_api_key": ..., "fred_api_key": ...}.
            client_id (str): The client the user authorized.
            redirect_uri (str): Where the code is sent; the token request must repeat it.
            code_challenge (str): The PKCE S256 challenge the token request must answer.

        Returns:
            str: The authorization code.
        """
        grant_id = secrets.token_urlsafe(16)
        encrypted = self._fernet.encrypt(json.dumps(keys).encode("utf-8"))
        with self._lock, self._connect() as connection:
            self._purge(connection)
            connection.execute(
                "INSERT INTO grants (id, client_id, keys, created) VALUES (?, ?, ?, ?)",
                (grant_id, client_id, encrypted, time.time()),
            )
            return self._issue(
                connection,
                grant_id,
                "code",
                CODE_LIFETIME,
                {"redirect_uri": redirect_uri, "code_challenge": code_challenge},
            )

    def redeem_authorization_code(
        self, code: str
    ) -> tuple[str, str, dict[str, Any]] | None:
        """
        Use up an authorization code.

        A code works once. A second attempt revokes the grant, as RFC 6749 section 4.1.2
        advises: someone else may have intercepted the code.

        Args:
            code (str): The authorization code.

        Returns:
            tuple[str, str, dict[str, Any]] | None: The grant id, the client id and the code's
            redirect_uri and code_challenge, or None when the code is unknown, expired or used.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT t.grant_id, t.expires, t.used, t.meta, g.client_id FROM tokens t "
                "JOIN grants g ON g.id = t.grant_id WHERE t.hash = ? AND t.kind = 'code'",
                (_hash(code),),
            ).fetchone()
            if row is None:
                return None
            grant_id, expires, used, meta, client_id = row
            if used:
                self._delete_grant(connection, grant_id)
                return None
            connection.execute(
                "UPDATE tokens SET used = 1 WHERE hash = ?", (_hash(code),)
            )
            if expires < time.time():
                return None
            return grant_id, client_id, json.loads(meta or "{}")

    def issue_token_pair(self, grant_id: str) -> tuple[str, str]:
        """
        Issue a new access token and refresh token for a grant.

        Args:
            grant_id (str): The grant.

        Returns:
            tuple[str, str]: The access token and the refresh token.
        """
        with self._lock, self._connect() as connection:
            access = self._issue(connection, grant_id, "access", ACCESS_TOKEN_LIFETIME)
            refresh = self._issue(
                connection, grant_id, "refresh", REFRESH_TOKEN_LIFETIME
            )
            return access, refresh

    def redeem_refresh_token(
        self, refresh_token: str, client_id: str = ""
    ) -> str | None:
        """
        Use up a refresh token, which is replaced by the next token pair.

        Presenting a refresh token that was already used revokes the grant: either the
        client or an attacker holds a stale copy, and the server cannot tell which.

        Args:
            refresh_token (str): The refresh token.
            client_id (str, optional): The client asking; when given, it must be the client
                the grant was issued to. Defaults to "".

        Returns:
            str | None: The grant id, or None when the token is unknown, expired or used.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT t.grant_id, t.expires, t.used, g.client_id FROM tokens t "
                "JOIN grants g ON g.id = t.grant_id WHERE t.hash = ? AND t.kind = 'refresh'",
                (_hash(refresh_token),),
            ).fetchone()
            if row is None:
                return None
            grant_id, expires, used, grant_client = row
            if used:
                self._delete_grant(connection, grant_id)
                return None
            if expires < time.time() or (client_id and client_id != grant_client):
                return None
            connection.execute(
                "UPDATE tokens SET used = 1 WHERE hash = ?", (_hash(refresh_token),)
            )
            # Access tokens issued earlier in the grant stay valid until they expire; the
            # old refresh token is kept, marked used, so a replay of it can be detected.
            return grant_id

    def keys_for_access_token(self, access_token: str) -> dict[str, str] | None:
        """
        The API keys behind an access token.

        Args:
            access_token (str): The access token.

        Returns:
            dict[str, str] | None: The keys, or None when the token is unknown or expired.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT g.keys FROM tokens t JOIN grants g ON g.id = t.grant_id "
                "WHERE t.hash = ? AND t.kind = 'access' AND t.expires >= ?",
                (_hash(access_token), time.time()),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(self._fernet.decrypt(row[0]))
        except InvalidToken:
            # Encrypted with another secret: the server secret changed since the grant.
            return None

    def revoke(self, token: str) -> bool:
        """
        Revoke the grant behind any of its tokens, deleting its keys and all its tokens.

        Args:
            token (str): An access token, refresh token or authorization code.

        Returns:
            bool: Whether a grant was revoked.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT grant_id FROM tokens WHERE hash = ?", (_hash(token),)
            ).fetchone()
            if row is None:
                return False
            self._delete_grant(connection, row[0])
            return True

    def revoke_all(self) -> int:
        """
        Revoke every grant, which signs every user out.

        Returns:
            int: The number of grants revoked.
        """
        with self._lock, self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM grants").fetchone()[0]
            connection.execute("DELETE FROM tokens")
            connection.execute("DELETE FROM grants")
            return int(count)
