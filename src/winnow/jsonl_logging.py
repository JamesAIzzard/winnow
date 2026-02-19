"""JSONL logging configuration for winnow exchange auditing."""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER_NAME = "winnow"


def configure_jsonl_logging(*, log_path: Path) -> None:
    """Attach a JSONL file handler to the winnow logger.

    If a handler is already attached, it is removed first to prevent
    accumulation on repeated calls.

    The handler uses a bare ``%(message)s`` formatter so that each
    pre-serialised JSON string is written as-is, one line per record.
    """
    logger = logging.getLogger(_LOGGER_NAME)

    _remove_existing_file_handlers(logger)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def _remove_existing_file_handlers(logger: logging.Logger) -> None:
    """Remove all FileHandlers currently attached to *logger*."""
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
