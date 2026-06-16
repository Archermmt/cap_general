"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def build_file_logger(
    log_dir: str | Path,
    *,
    logger_name: str,
    log_filename: str = "agent.log",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)s: %(message)s",
) -> logging.Logger:
    """Build or reuse a file logger under ``log_dir``."""
    record_dir = Path(log_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    log_path = record_dir / log_filename
    existing_handler = next(
        (
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == log_path.resolve()
        ),
        None,
    )
    if existing_handler is None:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
