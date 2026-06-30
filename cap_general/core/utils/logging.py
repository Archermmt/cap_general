"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def build_file_logger(
    log_dir: str | Path,
    *,
    logger_name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)s: %(message)s",
    console: bool = True,
) -> logging.Logger:
    """Build or reuse a logger that writes to file and optionally console."""
    record_dir = Path(log_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(fmt)
    log_path = record_dir / (logger_name + ".log")
    existing_handler = next(
        (
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path.resolve()
        ),
        None,
    )
    if existing_handler is None:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if console:
        existing_console_handler = next(
            (
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.StreamHandler)
                and not isinstance(handler, logging.FileHandler)
                and handler.stream is sys.stdout
            ),
            None,
        )
        if existing_console_handler is None:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    return logger


def close_file_handlers(logger: logging.Logger) -> None:
    """Close and detach all file handlers from a logger."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
