"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "hrr",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup a logger with both console and file handlers.

    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
