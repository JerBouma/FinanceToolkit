"""Logger Module"""

__docformat__ = "google"

import logging
import re
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

# Credentials that can appear in a logged URL or message: the value after a parameter named
# key, apikey, api_key, token, secret or password, or one of those after a prefix and an
# underscore (fmp_api_key, fred_key, access_token, client_secret, ...). The name has to start
# the parameter, so a name that merely ends in "key" (monkey, hotkey) keeps its value.
# A placeholder in angle brackets, as in the setup hints, is not a credential.
CREDENTIAL_PATTERN = re.compile(
    r"((?<!\w)(?:\w*_)?(?:api_?key|key|token|secret|password)=)[^&\s'\"<>]+",
    re.IGNORECASE,
)

# Renders the traceback of a record before it is redacted.
_TRACEBACK_FORMATTER = logging.Formatter()


def redact(text: object) -> str:
    """
    Removes credentials from a message, such as the API key in a URL that an HTTP error
    repeats.

    Args:
        text (object): The message, or an exception whose text is the message.

    Returns:
        str: The message with every credential replaced by "***".
    """
    return CREDENTIAL_PATTERN.sub(r"\1***", str(text))


class RedactingFilter(logging.Filter):
    """
    Removes credentials from a log record before any handler writes it: the message with
    its arguments filled in (an exception passed as an argument included), the traceback
    of a logged exception and the stack.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            # Arguments that don't fit the message are left to the handler to report.
            message = f"{record.msg} {record.args}"

        redacted = redact(message)
        if redacted != message:
            # Formatters such as uvicorn's access log read the arguments themselves, so a
            # credential inside an argument (a request path with ?apikey=...) is replaced in
            # that argument and the arguments are kept. A credential the arguments can't
            # hold apart, such as a key split over the message and an argument, makes the
            # redacted message replace both.
            arguments = None
            if isinstance(record.args, tuple) and redact(record.msg) == str(record.msg):
                arguments = tuple(
                    item if redact(item) == str(item) else redact(item)
                    for item in record.args
                )
                try:
                    if str(record.msg) % arguments != redacted:
                        arguments = None
                except (TypeError, ValueError):
                    arguments = None
            if arguments is None:
                record.msg, record.args = redacted, None
            else:
                record.args = arguments

        if record.exc_info and not record.exc_text:
            record.exc_text = _TRACEBACK_FORMATTER.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)

        return True


def add_redaction(logger: logging.Logger) -> None:
    """
    Attach a RedactingFilter to every handler of a logger, once.

    A handler's filter also applies to the records of the logger's children (such as
    "financetoolkit.cache"), which a filter on the logger itself would not see.

    Args:
        logger (logging.Logger): The logger whose handlers write the records.
    """
    for handler in logger.handlers:
        if not any(isinstance(item, RedactingFilter) for item in handler.filters):
            handler.addFilter(RedactingFilter())


def setup_logger(log_level=logging.INFO):
    """
    Set up and configure a logger with timestamp formatting.

    Credentials in the records of the Finance Toolkit's loggers, such as an API key in a
    request URL, are replaced by "***" before they are written.

    Args:
        log_level (int, optional): Logging level, by default logging.INFO

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("financetoolkit")

    # Don't override existing level if it's already been set
    if logger.level == 0:
        logger.setLevel(log_level)

    # Check if logger already has handlers to avoid duplicate handlers
    if not logger.handlers:
        # stderr always: stdout is reserved for JSON-RPC under stdio transport.
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # The handler added here writes every record, so passing them on to the root
        # logger as well would write them twice (or on stdout through a third-party root
        # handler). A logger that already had handlers was set up by the application,
        # whose choice to propagate is kept.
        logger.propagate = False

    add_redaction(logger)

    return logger


def get_logger():
    """
    Get the "financetoolkit" logger, set up through setup_logger when it is first asked
    for. Every module that logs gets it here, so its records are redacted whichever module
    is imported first (Economics, Fixed Income or Econometrics on their own included).

    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger()


# The messages written so far in the current log_each_message_once block. A ContextVar, so a
# block covers only its own call: the worker threads of run_in_parallel run in a copy of the
# caller's context and share its record, while another call running at the same time (an MCP
# request in another thread) keeps all of its own messages.
_written_messages: ContextVar[set[tuple[int, str]] | None] = ContextVar(
    "written_messages", default=None
)


def _first_time(record: logging.LogRecord) -> bool:
    """
    Lets a record through unless the current log_each_message_once block already wrote it.

    Args:
        record (logging.LogRecord): The record to write.

    Returns:
        bool: Whether to write the record.
    """
    written = _written_messages.get()
    if written is None:
        return True
    message = (record.levelno, record.getMessage())
    if message in written:
        return False
    written.add(message)
    return True


logging.getLogger("financetoolkit").addFilter(_first_time)


@contextmanager
def log_each_message_once() -> Generator[None]:
    """
    Write each distinct message of the Finance Toolkit's logger once while the block runs,
    for a method that calls many others which report the same thing (such as a date range
    without data) each time. It can also decorate a function, for every call of it. A block
    inside another shares the outer one's record, and the filter applies to this call only,
    including the worker threads it starts, never to another call running at the same time.

    Yields:
        None: Nothing; the messages are filtered until the block ends.
    """
    if _written_messages.get() is not None:
        yield
        return

    token = _written_messages.set(set())
    try:
        yield
    finally:
        _written_messages.reset(token)
