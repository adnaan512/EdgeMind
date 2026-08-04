"""
Centralized logging for EdgeMind AI.

Design Philosophy:
    Research frameworks never use ``print()`` for output. Proper logging
    provides:
    - Severity levels (DEBUG, INFO, WARNING, ERROR)
    - Timestamps for experiment tracking
    - File output for persistent records
    - Consistent formatting across all modules

    We use Python's built-in ``logging`` module enhanced with ``rich``
    for beautiful console output.

Usage:
    >>> from edgemind.core.logger import get_logger
    >>> logger = get_logger("training")
    >>> logger.info("Epoch 1/10 complete — accuracy: 92.3%")
    >>> logger.warning("Learning rate very low: 1e-7")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Keep track of loggers we've already configured to avoid duplicate handlers
_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Get or create a named logger for EdgeMind AI.

    Creates a logger with the prefix ``edgemind.`` to namespace all
    framework logs. If ``rich`` is installed, console output uses
    colorized formatting.

    Args:
        name: Logger name (e.g., "training", "data", "models").
            Will be prefixed with "edgemind.".
        level: Logging level (default: INFO).
        log_file: Optional path to a log file. If provided, logs are
            written to both console and file.

    Returns:
        A configured ``logging.Logger`` instance.

    Example:
        >>> logger = get_logger("training")
        >>> logger.info("Starting training...")
    """
    logger_name = f"edgemind.{name}"

    # Return existing logger if already configured
    if logger_name in _configured_loggers:
        return logging.getLogger(logger_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    if HAS_RICH:
        console_handler = RichHandler(
            level=level,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    _configured_loggers.add(logger_name)
    return logger
