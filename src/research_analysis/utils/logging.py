#!/usr/bin/env python3
"""
Logging configuration for the research analysis framework.
"""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure loguru logging with structured format and optional file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for persistent logging
        rotation: Log rotation size (e.g., "10 MB", "1 GB")
        retention: Log retention period (e.g., "1 week", "30 days")
    """
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Console handler with colors and emojis
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        colorize=True,
    )

    # File handler with rotation if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
        logger.info(f"📄 Logging to file: {log_path}")


def get_logger():
    """
    Returns the configured logger instance.

    This function provides an abstraction layer over the logging library,
    allowing other modules to get a logger without being directly coupled
    to loguru.

    Returns:
        The globally configured logger instance.
    """
    return logger 