"""Analytics Module

A small usage counter for a hosted MCP server: how often each tool is called and by
roughly how many different people, with the totals published at ``/stats``.

It is off unless ``FT_MCP_ANALYTICS=1`` is set, so a local installation never writes a
statistics file or exposes a ``/stats`` page. Nothing here depends on the Finance
Toolkit itself, so another MCP server (such as the Finance Database's) can use it with
its own name.

People are told apart by their API key, which is never stored: a user is the HMAC of
the key under the server's secret, so an identifier cannot be derived from a key by
anyone who lacks that secret. The counts live in one process; several worker processes
would each keep their own and overwrite each other's file.
"""

__docformat__ = "google"

import atexit
import functools
import hashlib
import hmac
import inspect
import json
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

ENABLED_ENVIRONMENT_VARIABLE = "FT_MCP_ANALYTICS"
LOCATION_ENVIRONMENT_VARIABLE = "FT_MCP_STATS_FILE"
SERVER_NAME_ENVIRONMENT_VARIABLE = "FT_MCP_SERVER_NAME"

ANONYMOUS = "anonymous"
SAVE_EVERY_CALLS = 5
DAILY_SERIES_DAYS = 30
TOP_TOOLS = 15


def is_enabled() -> bool:
    """
    Whether usage analytics are switched on, which they are only explicitly.

    Returns:
        bool: True when FT_MCP_ANALYTICS is 1, true, yes or on.
    """
    return os.environ.get(ENABLED_ENVIRONMENT_VARIABLE, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _today() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


class UsageAnalytics:
    """
    Counts the tool calls of an MCP server and keeps the counts in a JSON file.

    Every method catches its own errors: counting must never break a tool call.
    """

    def __init__(
        self,
        server_name: str,
        location: str | Path,
        secret: bytes,
        resolve_api_key: Callable[[], str] = lambda: "",
    ):
        """
        Initialize the counter. The stored counts are read with ``load``.

        Args:
            server_name (str): The name shown on the ``/stats`` page.
            location (str | Path): The JSON file the counts are kept in.
            secret (bytes): The secret the API keys are hashed with.
            resolve_api_key (Callable[[], str]): Returns the API key of the request
                being handled, empty without one.
        """
        self.server_name = server_name
        self._resolve_api_key = resolve_api_key
        self.location = Path(location)
        self._secret = secret
        self._lock = threading.Lock()
        self._started = time.time()
        self._unsaved_calls = 0

        self.total_calls = 0
        self.failed_calls = 0
        self.calls_per_tool: dict[str, int] = {}
        self.failures_per_tool: dict[str, int] = {}
        self.seconds_per_tool: dict[str, float] = {}
        self.calls_per_day: dict[str, int] = {}
        self.calls_per_user: dict[str, int] = {}
        self.first_seen: dict[str, str] = {}

    # ── Identity ──────────────────────────────────────────────────────────────
    def user_id(self, api_key: str) -> str:
        """
        The anonymous identifier of the person behind an API key.

        Args:
            api_key (str): The key the call was made with, empty without one.

        Returns:
            str: "u_" and 12 characters of the key's HMAC, or "anonymous".
        """
        if not api_key:
            return ANONYMOUS

        digest = hmac.new(self._secret, api_key.encode(), hashlib.sha256).hexdigest()

        return f"u_{digest[:12]}"

    def _adopt_unkeyed_identifier(self, api_key: str, user: str) -> None:
        """
        Moves the counts of a user stored under an unkeyed SHA-256 identifier, as an
        earlier version recorded them, to the keyed one, so nobody counts twice.
        """
        legacy = f"u_{hashlib.sha256(api_key.encode()).hexdigest()[:12]}"

        if legacy == user or legacy not in self.calls_per_user:
            return

        self.calls_per_user[user] = self.calls_per_user.get(
            user, 0
        ) + self.calls_per_user.pop(legacy)
        legacy_first_seen = self.first_seen.pop(legacy, None)

        if legacy_first_seen:
            self.first_seen[user] = min(
                legacy_first_seen, self.first_seen.get(user, legacy_first_seen)
            )

    # ── Counting ──────────────────────────────────────────────────────────────
    def record_call(
        self, tool: str, api_key: str, succeeded: bool = True, seconds: float = 0.0
    ) -> None:
        """
        Count a call of a tool, and save every few calls.

        Args:
            tool (str): The tool's name.
            api_key (str): The key the call was made with, empty without one.
            succeeded (bool): Whether the tool returned rather than raised.
            seconds (float): How long the call took.
        """
        try:
            user = self.user_id(api_key)
            today = _today()

            with self._lock:
                if api_key:
                    self._adopt_unkeyed_identifier(api_key, user)

                self.total_calls += 1
                self.calls_per_tool[tool] = self.calls_per_tool.get(tool, 0) + 1
                self.seconds_per_tool[tool] = (
                    self.seconds_per_tool.get(tool, 0.0) + seconds
                )
                self.calls_per_day[today] = self.calls_per_day.get(today, 0) + 1
                self.calls_per_user[user] = self.calls_per_user.get(user, 0) + 1
                self.first_seen.setdefault(user, today)

                if not succeeded:
                    self.failed_calls += 1
                    self.failures_per_tool[tool] = (
                        self.failures_per_tool.get(tool, 0) + 1
                    )

                self._unsaved_calls += 1
                due = self._unsaved_calls >= SAVE_EVERY_CALLS

            if due:
                self.save()
        except Exception as error:  # noqa: BLE001
            logger.debug("Could not count the call of %s: %s", tool, error)

    def counted(self, tool: str, function: Callable) -> Any:
        """
        Wrap a tool so every call is counted, keeping the signature and annotations
        FastMCP builds the tool's input schema from.

        Args:
            tool (str): The tool's name.
            function (Callable): The tool, synchronous or asynchronous.

        Returns:
            Any: The counting tool.
        """

        def key() -> str:
            try:
                return self._resolve_api_key() or ""
            except Exception:  # noqa: BLE001
                return ""

        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def counting_coroutine(*args, **kwargs):
                started, succeeded = time.perf_counter(), False
                try:
                    result = await function(*args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    self.record_call(
                        tool, key(), succeeded, time.perf_counter() - started
                    )

            wrapper: Any = counting_coroutine
        else:

            @functools.wraps(function)
            def counting_function(*args, **kwargs):
                started, succeeded = time.perf_counter(), False
                try:
                    result = function(*args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    self.record_call(
                        tool, key(), succeeded, time.perf_counter() - started
                    )

            wrapper = counting_function

        # The annotations are resolved here, against the tool's own module: FastMCP would
        # otherwise look string annotations up in this module and fail.
        signature = inspect.signature(function, eval_str=True)
        wrapper.__signature__ = signature
        wrapper.__annotations__ = {
            name: parameter.annotation
            for name, parameter in signature.parameters.items()
            if parameter.annotation is not inspect.Parameter.empty
        }

        if signature.return_annotation is not inspect.Signature.empty:
            wrapper.__annotations__["return"] = signature.return_annotation

        # The wrapper is described by __signature__, so FastMCP must not unwrap it.
        del wrapper.__wrapped__

        return wrapper

    # ── Storage ───────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """
        The counts as stored in the file.

        Returns:
            dict[str, Any]: The counts.
        """
        return {
            "server": self.server_name,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "calls_per_tool": self.calls_per_tool,
            "failures_per_tool": self.failures_per_tool,
            "seconds_per_tool": self.seconds_per_tool,
            "calls_per_day": self.calls_per_day,
            "calls_per_user": self.calls_per_user,
            "first_seen": self.first_seen,
        }

    def load(self) -> None:
        """Read the counts back from the file, if there is one."""
        try:
            if not self.location.exists():
                return

            stored = json.loads(self.location.read_text(encoding="utf-8"))

            def first(*names: str, default: Any) -> Any:
                # The counter the hosted server ran before (v2.2.1) named these by_tool,
                # by_day, user_calls and user_first_seen.
                return next((stored[name] for name in names if name in stored), default)

            with self._lock:
                self.total_calls = int(first("total_calls", default=0))
                self.failed_calls = int(first("failed_calls", default=0))
                self.calls_per_tool = dict(
                    first("calls_per_tool", "by_tool", default={})
                )
                self.failures_per_tool = dict(first("failures_per_tool", default={}))
                self.seconds_per_tool = dict(first("seconds_per_tool", default={}))
                self.calls_per_day = dict(first("calls_per_day", "by_day", default={}))
                self.calls_per_user = dict(
                    first("calls_per_user", "user_calls", default={})
                )
                self.first_seen = dict(
                    first("first_seen", "user_first_seen", default={})
                )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Could not read the usage statistics at %s: %s", self.location, error
            )

    def save(self) -> None:
        """Write the counts to the file, through a temporary file so it is never torn."""
        try:
            with self._lock:
                content = json.dumps(self.to_dict(), indent=1, sort_keys=True)
                self._unsaved_calls = 0

            self.location.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.location.with_suffix(".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(self.location)
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "Could not save the usage statistics at %s: %s", self.location, error
            )

    # ── Publishing ────────────────────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        """
        The totals published at ``/stats``, without any per-user figures.

        Returns:
            dict[str, Any]: The totals.
        """
        with self._lock:
            today = datetime.now(tz=UTC).date()
            days = [
                (today - timedelta(days=offset)).strftime("%Y-%m-%d")
                for offset in range(DAILY_SERIES_DAYS - 1, -1, -1)
            ]
            identified_users = [
                user for user in self.calls_per_user if user != ANONYMOUS
            ]
            identified_calls = sum(
                self.calls_per_user[user] for user in identified_users
            )
            top_tools = sorted(
                self.calls_per_tool.items(), key=lambda item: item[1], reverse=True
            )[:TOP_TOOLS]

            return {
                "server": self.server_name,
                "total_calls": self.total_calls,
                "unique_users": len(identified_users),
                "anonymous_calls": self.calls_per_user.get(ANONYMOUS, 0),
                "average_calls_per_user": (
                    round(identified_calls / len(identified_users), 1)
                    if identified_users
                    else 0
                ),
                "success_rate": (
                    round(1 - self.failed_calls / self.total_calls, 4)
                    if self.total_calls
                    else None
                ),
                "calls_today": self.calls_per_day.get(days[-1], 0),
                "calls_last_7_days": sum(
                    self.calls_per_day.get(day, 0) for day in days[-7:]
                ),
                "daily_calls": [
                    {"date": day, "calls": self.calls_per_day.get(day, 0)}
                    for day in days
                ],
                "top_tools": [
                    {
                        "tool": tool,
                        "calls": calls,
                        "average_seconds": round(
                            self.seconds_per_tool.get(tool, 0.0) / calls, 2
                        ),
                    }
                    for tool, calls in top_tools
                ],
                "uptime_seconds": int(time.time() - self._started),
            }

    def register_route(self, mcp: Any, path: str = "/stats") -> None:
        """
        Publish the totals at a public route of the server.

        Args:
            mcp (Any): The FastMCP server.
            path (str): The route. Defaults to "/stats".
        """
        if not hasattr(mcp, "custom_route"):
            return

        @mcp.custom_route(path, methods=["GET"])
        async def stats(request: Request) -> JSONResponse:  # noqa: ARG001
            return JSONResponse(self.summary())


def create_from_environment(
    default_server_name: str,
    default_location: Path,
    secret: bytes,
    resolve_api_key: Callable[[], str],
) -> UsageAnalytics | None:
    """
    Set up the counter when FT_MCP_ANALYTICS is on: read the stored counts back and
    save them once more when the server stops.

    Args:
        default_server_name (str): The server name, unless FT_MCP_SERVER_NAME is set.
        default_location (Path): The statistics file, unless FT_MCP_STATS_FILE is set.
        secret (bytes): The secret the API keys are hashed with.
        resolve_api_key (Callable[[], str]): Returns the API key of the request being
            handled, empty without one.

    Returns:
        UsageAnalytics | None: The counter, or None when analytics are off.
    """
    if not is_enabled():
        return None

    analytics = UsageAnalytics(
        server_name=os.environ.get(SERVER_NAME_ENVIRONMENT_VARIABLE)
        or default_server_name,
        location=os.environ.get(LOCATION_ENVIRONMENT_VARIABLE) or default_location,
        secret=secret,
        resolve_api_key=resolve_api_key,
    )
    analytics.load()
    atexit.register(analytics.save)
    logger.info(
        "Usage analytics are on and kept at %s (%d calls so far).",
        analytics.location,
        analytics.total_calls,
    )

    return analytics
