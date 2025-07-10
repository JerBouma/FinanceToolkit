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


def test_setup_logger_stream_handler_uses_stdout():
    """Test setup_logger uses sys.stdout for StreamHandler."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.level = 0
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        with patch("logging.StreamHandler") as mock_stream_handler:
            mock_handler = MagicMock()
            mock_stream_handler.return_value = mock_handler

            result = logger_model.setup_logger()

            # Check StreamHandler was called with sys.stdout
            mock_stream_handler.assert_called_once_with(sys.stdout)

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
    # This test verifies that get_logger always returns the same logger instance
    # when called multiple times
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
    assert handler.stream == sys.stdout

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
