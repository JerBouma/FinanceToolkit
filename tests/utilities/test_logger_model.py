"""Logger Model Tests"""

# ruff: noqa: PLR2004

import logging
import sys
from unittest.mock import MagicMock, patch

from financetoolkit.utilities import logger_model


def test_setup_logger_default_level():
    """Test setup_logger with default logging level."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.level = 0  # Not set
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch("logging.StreamHandler") as mock_stream_handler:
            mock_handler = MagicMock()
            mock_stream_handler.return_value = mock_handler

            result = logger_model.setup_logger()

            # Check logger was configured
            mock_get_logger.assert_called_once_with("financetoolkit")
            mock_logger.setLevel.assert_called_once_with(logging.INFO)
            mock_handler.setLevel.assert_called_once_with(logging.INFO)
            mock_logger.addHandler.assert_called_once_with(mock_handler)

            assert result == mock_logger


def test_setup_logger_custom_level():
    """Test setup_logger with custom logging level."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.level = 0  # Not set
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch("logging.StreamHandler") as mock_stream_handler:
            mock_handler = MagicMock()
            mock_stream_handler.return_value = mock_handler

            result = logger_model.setup_logger(logging.DEBUG)

            # Check logger was configured with custom level
            mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
            mock_handler.setLevel.assert_called_once_with(logging.DEBUG)

            assert result == mock_logger


def test_setup_logger_existing_level():
    """Test setup_logger when logger already has level set."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.level = logging.WARNING  # Already set
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch("logging.StreamHandler") as mock_stream_handler:
            mock_handler = MagicMock()
            mock_stream_handler.return_value = mock_handler

            result = logger_model.setup_logger()

            # Should not override existing level
            mock_logger.setLevel.assert_not_called()
            mock_handler.setLevel.assert_called_once_with(logging.INFO)

            assert result == mock_logger


def test_setup_logger_existing_handlers():
    """Test setup_logger when logger already has handlers."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.level = 0  # Not set
        mock_logger.handlers = [MagicMock()]  # Has handlers
        mock_get_logger.return_value = mock_logger

        result = logger_model.setup_logger()

        # Should not add new handlers
        mock_logger.addHandler.assert_not_called()

        assert result == mock_logger


def test_setup_logger_formatter_configuration():
    """Test setup_logger formatter configuration."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.level = 0
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch("logging.StreamHandler") as mock_stream_handler:
            mock_handler = MagicMock()
            mock_stream_handler.return_value = mock_handler

            with patch("logging.Formatter") as mock_formatter:
                mock_formatter_instance = MagicMock()
                mock_formatter.return_value = mock_formatter_instance

                result = logger_model.setup_logger()

                # Check formatter was configured correctly
                mock_formatter.assert_called_once_with(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                mock_handler.setFormatter.assert_called_once_with(
                    mock_formatter_instance
                )

                assert result == mock_logger


def test_setup_logger_stream_handler_uses_stderr():
    """Test setup_logger uses sys.stderr for StreamHandler."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.level = 0
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch("logging.StreamHandler") as mock_stream_handler:
            mock_handler = MagicMock()
            mock_stream_handler.return_value = mock_handler

            result = logger_model.setup_logger()

            # stderr, not stdout, so log output never corrupts the stdio JSON-RPC pipe.
            mock_stream_handler.assert_called_once_with(sys.stderr)

            assert result == mock_logger


def test_get_logger():
    """Test get_logger returns logger instance."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        result = logger_model.get_logger()

        mock_get_logger.assert_called_once_with("financetoolkit")
        assert result == mock_logger


def test_get_logger_returns_existing_logger():
    """Test get_logger returns existing logger instance."""
    # get_logger should return the same instance on every call.
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        result1 = logger_model.get_logger()
        result2 = logger_model.get_logger()

        # Should call getLogger multiple times but return same instance
        assert mock_get_logger.call_count == 2
        assert result1 == result2 == mock_logger


def test_setup_logger_integration():
    """Test setup_logger integration with actual logging module."""
    # Clean up any existing handlers
    logger = logging.getLogger("financetoolkit")
    logger.handlers.clear()
    logger.setLevel(0)

    result = logger_model.setup_logger(logging.DEBUG)

    # Check logger properties
    assert result.name == "financetoolkit"
    assert result.level == logging.DEBUG
    assert len(result.handlers) == 1

    # Check handler properties
    handler = result.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.level == logging.DEBUG
    assert handler.stream == sys.stderr

    # Check formatter
    formatter = handler.formatter
    assert formatter is not None
    assert "%(asctime)s - %(name)s - %(levelname)s - %(message)s" in formatter._fmt


def test_get_logger_integration():
    """Test get_logger integration with actual logging module."""
    result = logger_model.get_logger()

    assert result.name == "financetoolkit"
    assert isinstance(result, logging.Logger)


def test_logger_singleton_behavior():
    """Test that logger behaves as singleton."""
    # Setup logger first
    logger1 = logger_model.setup_logger()

    # Get logger should return same instance
    logger2 = logger_model.get_logger()

    assert logger1 is logger2


def test_setup_logger_multiple_calls():
    """Test setup_logger called multiple times doesn't duplicate handlers."""
    # Clean up any existing handlers
    logger = logging.getLogger("financetoolkit")
    logger.handlers.clear()
    logger.setLevel(0)

    # Setup logger multiple times
    logger1 = logger_model.setup_logger()
    logger2 = logger_model.setup_logger()
    logger3 = logger_model.setup_logger()

    # Should be same instance
    assert logger1 is logger2 is logger3

    # Should not duplicate handlers
    assert len(logger1.handlers) == 1


def test_setup_logger_different_levels():
    """Test setup_logger with different levels after initial setup."""
    # Clean up any existing handlers
    logger = logging.getLogger("financetoolkit")
    logger.handlers.clear()
    logger.setLevel(0)

    # Setup with INFO level
    logger1 = logger_model.setup_logger(logging.INFO)
    assert logger1.level == logging.INFO

    # Setup with DEBUG level - should not change existing level
    logger2 = logger_model.setup_logger(logging.DEBUG)
    assert logger2.level == logging.INFO  # Should remain INFO

    # Should be same instance
    assert logger1 is logger2


def test_logger_name_consistency():
    """Test that logger name is consistent across calls."""
    logger1 = logger_model.setup_logger()
    logger2 = logger_model.get_logger()

    assert logger1.name == "financetoolkit"
    assert logger2.name == "financetoolkit"
    assert logger1.name == logger2.name


def test_keys_in_logged_exceptions_and_tracebacks_are_redacted():
    """
    Test that an API key in a URL is redacted from everything the Finance Toolkit logs:
    an exception passed as an argument, the message of a child logger and the traceback.
    """
    import io  # noqa: PLC0415

    logger = logger_model.setup_logger()
    handler = logger.handlers[0]
    stream = io.StringIO()
    previous_stream = handler.setStream(stream)

    try:
        url = (
            "https://financialmodelingprep.com/stable/quote?symbol=AAPL&apikey=SECRET1"
        )
        logger.warning("Request failed: %s", ValueError(f"404 for {url}"))
        logging.getLogger("financetoolkit.cache").warning(
            "Inline %s https://api.example/x?fmp_api_key=SECRET2", "message"
        )
        try:
            raise ConnectionError(f"{url}&api_key=SECRET3")
        except ConnectionError:
            logger.exception("Could not reach the source")
    finally:
        handler.setStream(previous_stream)

    output = stream.getvalue()

    assert "SECRET" not in output
    assert "apikey=***" in output
    assert "fmp_api_key=***" in output
    assert "Traceback" in output
    assert "Inline message" in output


def test_redaction_keeps_the_arguments_formatters_read():
    """
    A record without a credential keeps its arguments, and a credential inside an argument is
    replaced there, so formatters that read the arguments themselves (uvicorn's access log
    unpacks client, method, path, version and status) still work.
    """
    redacting = logger_model.RedactingFilter()

    def record(msg, args):
        return logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, msg, args, None
        )

    access = '%s - "%s %s HTTP/%s" %d'
    clean = record(access, ("1.2.3.4:5", "GET", "/health", "1.1", 200))
    assert redacting.filter(clean)
    assert clean.args == ("1.2.3.4:5", "GET", "/health", "1.1", 200)
    assert clean.getMessage() == '1.2.3.4:5 - "GET /health HTTP/1.1" 200'

    keyed = record(access, ("1.2.3.4:5", "GET", "/mcp?apikey=secret123", "1.1", 200))
    assert redacting.filter(keyed)
    assert keyed.args == ("1.2.3.4:5", "GET", "/mcp?apikey=***", "1.1", 200)
    assert "secret123" not in keyed.getMessage()

    # A key split over the message and its argument can only be removed from the whole message.
    split = record("https://example.com/quote?apikey=%s", ("secret123",))
    assert redacting.filter(split)
    assert split.args is None
    assert split.getMessage() == "https://example.com/quote?apikey=***"


def test_get_logger_sets_up_redaction_before_any_toolkit_import():
    """
    Test that the logger a module gets is redacting even when nothing has called
    setup_logger yet, as when only Economics or Fixed Income is imported.
    """
    import io  # noqa: PLC0415

    logger = logging.getLogger("financetoolkit")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    for handler in handlers:
        logger.removeHandler(handler)

    try:
        stream = io.StringIO()
        with patch.object(sys, "stderr", stream):
            result = logger_model.get_logger()
            result.warning("Failed: https://api.example/x?series=GDP&api_key=SECRET")

        assert result is logger
        assert len(logger.handlers) == 1
        assert "SECRET" not in stream.getvalue()
        assert "api_key=***" in stream.getvalue()
        # Asking again adds nothing.
        logger_model.get_logger()
        assert len(logger.handlers) == 1
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


def test_setup_logger_keeps_propagation_of_a_logger_with_handlers():
    """Test that propagation is only switched off together with the handler it adds."""
    logger = logging.getLogger("financetoolkit")
    original_handlers, original_propagate = list(logger.handlers), logger.propagate

    try:
        logger.handlers = [logging.NullHandler()]
        logger.propagate = True
        logger_model.setup_logger()

        assert logger.propagate is True
        assert len(logger.handlers) == 1

        logger.handlers = []
        logger_model.setup_logger()

        assert logger.propagate is False
        assert len(logger.handlers) == 1
    finally:
        logger.handlers = original_handlers
        logger.propagate = original_propagate


def test_redact_only_hides_credential_names():
    """Test that a parameter that merely ends in "key" keeps its value."""
    assert (
        logger_model.redact("keyword=apple&monkey=1&hotkey=2&tokens=3")
        == "keyword=apple&monkey=1&hotkey=2&tokens=3"
    )
    assert (
        logger_model.redact(
            "key=A&apikey=B&API_KEY=C&fred_api_key=D&fred_key=E&access_token=F"
            "&token=G&client_secret=H&password=I"
        )
        == "key=***&apikey=***&API_KEY=***&fred_api_key=***&fred_key=***"
        "&access_token=***&token=***&client_secret=***&password=***"
    )


def test_log_each_message_once_writes_a_repeated_message_once(caplog):
    import logging  # noqa: PLC0415

    from financetoolkit.helpers import run_in_parallel  # noqa: PLC0415
    from financetoolkit.utilities.logger_model import (  # noqa: PLC0415
        log_each_message_once,
    )

    logger = logging.getLogger("financetoolkit")
    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        with log_each_message_once():
            logger.warning("No data for the range")
            with log_each_message_once():
                logger.warning("No data for the range")
            # The worker threads of the call share its record.
            run_in_parallel(lambda: logger.warning("No data for the range"), [()] * 4)
            logger.warning("Another message")
        logger.warning("No data for the range")

    assert caplog.text.count("No data for the range") == 2  # noqa: PLR2004
    assert caplog.text.count("Another message") == 1


def test_log_each_message_once_leaves_other_calls_alone(caplog):
    import logging  # noqa: PLC0415
    import threading  # noqa: PLC0415

    from financetoolkit.utilities.logger_model import (  # noqa: PLC0415
        log_each_message_once,
    )

    logger = logging.getLogger("financetoolkit")
    inside = threading.Event()
    done = threading.Event()

    def call():
        with log_each_message_once():
            logger.warning("Rate limit reached")
            inside.set()
            done.wait(5)

    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        thread = threading.Thread(target=call)
        thread.start()
        inside.wait(5)
        # Another call running at the same time still writes its own message.
        logger.warning("Rate limit reached")
        done.set()
        thread.join()

    assert caplog.text.count("Rate limit reached") == 2  # noqa: PLR2004
