from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from winnow.exchange.logging import configure_jsonl_logging, record_exchange


@pytest.fixture
def isolated_winnow_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("winnow")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate

    for handler in original_handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True

    yield logger

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.setLevel(original_level)
    logger.propagate = original_propagate


class TestRecordExchange:
    def test_emits_json_record_with_utc_timestamp(
        self,
        isolated_winnow_logger: logging.Logger,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="winnow"):
            record_exchange(
                uid="protein",
                prompt="How much protein?",
                response="31",
            )

        assert len(caplog.records) == 1
        record = json.loads(caplog.records[0].message)
        assert record.keys() == {
            "timestamp",
            "question_uid",
            "prompt",
            "response",
        }
        assert record["question_uid"] == "protein"
        assert record["prompt"] == "How much protein?"
        assert record["response"] == "31"
        timestamp = datetime.fromisoformat(record["timestamp"])
        assert timestamp.utcoffset() == UTC.utcoffset(None)

    def test_is_silent_without_a_configured_file_handler(
        self, isolated_winnow_logger: logging.Logger,
    ) -> None:
        isolated_winnow_logger.propagate = False

        record_exchange(uid="protein", prompt="prompt", response="31")

        assert isolated_winnow_logger.handlers == []


class TestConfigureJsonlLogging:
    def test_replaces_existing_file_handler(
        self, isolated_winnow_logger: logging.Logger, tmp_path: Path,
    ) -> None:
        first_log = tmp_path / "first.jsonl"
        second_log = tmp_path / "second.jsonl"

        configure_jsonl_logging(log_path=first_log)
        record_exchange(uid="first", prompt="prompt", response="one")
        configure_jsonl_logging(log_path=second_log)
        record_exchange(uid="second", prompt="prompt", response="two")

        assert len(isolated_winnow_logger.handlers) == 1
        first_record = json.loads(first_log.read_text(encoding="utf-8"))
        second_record = json.loads(second_log.read_text(encoding="utf-8"))
        assert first_record["question_uid"] == "first"
        assert second_record["question_uid"] == "second"
