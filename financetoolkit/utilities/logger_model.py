"""Logger Module"""

__docformat__ = "google"

import logging
import sys

# pylint: disable=too-few-public-methods


def setup_logger(log_level=logging.INFO):
    """
    Set up and configure a logger with timestamp formatting.

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
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

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
