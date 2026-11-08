"""Logger Module"""

__docformat__ = "google"

import logging
import re
import sys

# Credentials that can appear in a logged URL or message: the value after any parameter
# whose name ends in "key" (apikey, api_key, fmp_api_key, fred_key, ...) or is "token".
# A placeholder in angle brackets, as in the setup hints, is not a credential.
CREDENTIAL_PATTERN = re.compile(
    r"((?:api_?key|apikey|token|key)=)[^&\s'\"<>]+", re.IGNORECASE
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
        record.msg = redact(message)
        record.args = None

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

    add_redaction(logger)

    # Prevents duplicate emission through third-party root handlers on stdout.
    logger.propagate = False

    return logger


def get_logger():
    """
    Get an existing logger or create a new one if it doesn't exist.

    Args:
        name (str, optional): Name of the logger, by default "financetoolkit"

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger("financetoolkit")
