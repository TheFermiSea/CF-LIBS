"""
Tests for logging configuration module.
"""

import pytest
import logging
from io import StringIO

from cflibs.core.logging_config import setup_logging, get_logger


def test_setup_logging_default():
    """Test setting up logging with default parameters."""
    stream = StringIO()
    setup_logging(level="INFO", stream=stream)

    logger = logging.getLogger("cflibs.test")
    logger.info("Test message")

    output = stream.getvalue()
    assert "Test message" in output
    assert "INFO" in output


def test_setup_logging_custom_level():
    """Test setting up logging with custom level."""
    stream = StringIO()
    setup_logging(level="DEBUG", stream=stream)

    logger = logging.getLogger("cflibs.test")
    logger.debug("Debug message")

    output = stream.getvalue()
    assert "Debug message" in output


def test_setup_logging_custom_format():
    """Test setting up logging with custom format."""
    stream = StringIO()
    custom_format = "%(levelname)s - %(message)s"
    setup_logging(level="INFO", format_string=custom_format, stream=stream)

    logger = logging.getLogger("cflibs.test")
    logger.info("Test message")

    output = stream.getvalue()
    assert "INFO - Test message" in output
    assert "Test message" in output


def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "cflibs.test.module"


def test_logger_hierarchy():
    """Test that loggers follow proper hierarchy."""
    parent_logger = get_logger("parent")
    child_logger = get_logger("parent.child")

    assert child_logger.parent == parent_logger or child_logger.name.startswith("cflibs.parent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
