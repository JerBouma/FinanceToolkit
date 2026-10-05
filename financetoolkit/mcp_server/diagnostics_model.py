"""Diagnostics Model"""

__docformat__ = "google"

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from financetoolkit.utilities.logger_model import get_logger

# The warnings and errors the Finance Toolkit logs while one tool call runs, so that a call
# that returns no data can say why, such as a missing API key or an unreachable source,
# instead of only "No data available". A context variable keeps concurrent calls apart.
_CALL_MESSAGES: ContextVar[list[str] | None] = ContextVar(
    "financetoolkit_call_messages", default=None
)

# How many distinct reasons, and how many characters of each, a response includes.
MAXIMUM_REASONS = 3
MAXIMUM_REASON_LENGTH = 300

# Credentials that can appear in a logged URL or message, which must never reach the
# client: the value after any parameter whose name ends in "key" or is "apikey"/"token".
CREDENTIAL_PATTERN = re.compile(
    r"((?:api_?key|apikey|token|key)=)[^&\s'\"]+", re.IGNORECASE
)


# How a key is passed to the MCP server, which differs from the Python arguments the
# library's own messages name.
FMP_KEY_HINT = (
    "A FinancialModelingPrep API key is required for this tool. Local setup: set "
    "FINANCIAL_MODELING_PREP_API_KEY in your environment or .env file. Hosted setup: pass "
    "your key via the `X-FMP-API-Key` header or a `?fmp_api_key=...` URL parameter. Get a "
    "key with 15% off via https://www.jeroenbouma.com/fmp"
)
FRED_KEY_HINT = (
    "A FRED API key is required for this indicator. Local setup: set FRED_API_KEY in your "
    "environment or .env file. Hosted setup: pass your key via the `X-FRED-API-Key` header "
    "or a `?fred_api_key=...` URL parameter. A key is free at "
    "https://fred.stlouisfed.org/docs/api/api_key.html"
)

# The generic message the error handler logs for any failed metric, only worth passing on
# when nothing more specific was logged.
GENERIC_FAILURE_PATTERN = re.compile(r" could not be calculated for ")


class _CallMessageHandler(logging.Handler):
    """Collects the warnings and errors logged during the tool call that is running."""

    def emit(self, record: logging.LogRecord) -> None:
        messages = _CALL_MESSAGES.get()

        if messages is None or record.levelno < logging.WARNING:
            return

        try:
            messages.append(record.getMessage())
        except Exception:  # noqa: BLE001  # pylint: disable=broad-except
            # A message that cannot be formatted is not worth failing a tool call for.
            return


_HANDLER = _CallMessageHandler(level=logging.WARNING)


def _install_handler() -> None:
    """Attaches the collecting handler to the Finance Toolkit logger once."""
    toolkit_logger = get_logger()

    if _HANDLER not in toolkit_logger.handlers:
        toolkit_logger.addHandler(_HANDLER)


def redact(text: object) -> str:
    """
    Removes credentials from a message before it is returned to the client, such as the
    API key in a URL that an HTTP error repeats.

    Args:
        text (object): The message, or an exception whose text is the message.

    Returns:
        str: The message with every credential replaced by "***".
    """
    return CREDENTIAL_PATTERN.sub(r"\1***", str(text))


@contextmanager
def capture_call_messages() -> Iterator[list[str]]:
    """
    Collects the warnings and errors the Finance Toolkit logs while the block runs,
    including those of the worker threads it starts.

    Yields:
        list[str]: The collected messages, filled while the block runs.
    """
    _install_handler()
    messages: list[str] = []
    token = _CALL_MESSAGES.set(messages)

    try:
        yield messages
    finally:
        _CALL_MESSAGES.reset(token)


def summarize_reasons(messages: list[str] | None) -> str | None:
    """
    Turns the collected messages into a short reason for a response without data.

    A missing API key is answered with how to pass one to the MCP server. Otherwise the
    reason is the first line of each distinct message, with any credential redacted, and
    the generic "could not be calculated" message only when nothing specific was logged.

    Args:
        messages (list[str] | None): The collected messages.

    Returns:
        str | None: The reasons joined by " | ", or None when there are none.
    """
    messages = [message for message in messages or [] if message and message.strip()]
    text = " ".join(messages)

    if "FinancialModelingPrep API key" in text:
        return FMP_KEY_HINT
    if "FRED API key" in text:
        return FRED_KEY_HINT

    specific = [
        message for message in messages if not GENERIC_FAILURE_PATTERN.search(message)
    ]
    reasons: list[str] = []

    for message in specific or messages:
        first_line = redact(message.strip().splitlines()[0])

        if len(first_line) > MAXIMUM_REASON_LENGTH:
            first_line = first_line[: MAXIMUM_REASON_LENGTH - 3] + "..."

        if first_line not in reasons:
            reasons.append(first_line)

        if len(reasons) == MAXIMUM_REASONS:
            break

    return " | ".join(reasons) if reasons else None
