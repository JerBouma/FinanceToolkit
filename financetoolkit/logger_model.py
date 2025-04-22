"""Logger Module"""

__docformat__ = "google"

import logging
import os
import sys

# pylint: disable=too-few-public-methods


class DottedPathFilter(logging.Filter):
    """
    A logging filter that adds a dotted_path attribute to log records.

    The dotted_path represents the Python module path relative to the project root,
    which makes log messages more readable and consistent.
    """

    def filter(self, record):
        # Compute relative path from project root
        project_root = os.path.dirname(__file__)  # Adjust if needed
        relative_path = os.path.relpath(record.pathname, start=project_root)
        dotted_path = os.path.splitext(relative_path)[0].replace(os.path.sep, ".")
        record.dotted_path = dotted_path
        return True


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
            "%(asctime)s - %(name)s.%(dotted_path)s:%(lineno)d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        console_handler.addFilter(DottedPathFilter())

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
