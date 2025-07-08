#!/usr/bin/env python3
"""
Logging configuration for the research analysis framework.
"""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from research_analysis.config.models import LoggingConfig


def setup_logging(config: LoggingConfig, log_file: Optional[Union[str, Path]] = None) -> None:
    """
    Configure loguru logging with structured format and optional file output.

    Args:
        config: A Pydantic model containing logging settings (level, rotation, retention).
        log_file: Optional path to log file for persistent logging.
    """
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Console handler with colors and emojis
    logger.add(
        sys.stdout,
        level=config.level,
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
            level=config.level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=config.rotation,
            retention=config.retention,
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