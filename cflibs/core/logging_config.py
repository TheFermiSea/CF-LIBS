"""
Logging configuration for CF-LIBS.

Provides standardized logging setup for the library.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO", format_string: Optional[str] = None, stream: Optional[object] = None
) -> None:
    """
    Configure logging for CF-LIBS.

    Parameters
    ----------
    level : str
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    format_string : str, optional
        Custom format string. If None, uses default format.
    stream : file-like object, optional
        Stream to write logs to. If None, uses sys.stderr.
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

    if stream is None:
        stream = sys.stderr

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        stream=stream,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(f"cflibs.{name}")
