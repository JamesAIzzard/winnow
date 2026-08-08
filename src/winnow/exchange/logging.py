"""JSONL logging for prompt-response exchanges."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

_LOGGER_NAME = "winnow"
_logger = logging.getLogger(_LOGGER_NAME)


def configure_jsonl_logging(*, log_path: Path) -> None:
    """Attach a JSONL file handler to the winnow logger.

    If a handler is already attached, it is removed first to prevent
    accumulation on repeated calls.

    The handler uses a bare ``%(message)s`` formatter so that each
    pre-serialised JSON string is written as-is, one line per record.
    """
    _remove_existing_file_handlers(_logger)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.DEBUG)
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)


def record_exchange(*, uid: str, prompt: str, response: str) -> None:
    """Emit a JSONL record describing one prompt-response exchange."""
    record = json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "question_uid": uid,
        "prompt": prompt,
        "response": response,
    })
    _logger.debug(record)


def _remove_existing_file_handlers(logger: logging.Logger) -> None:
    """Remove all file handlers currently attached to the logger."""
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
