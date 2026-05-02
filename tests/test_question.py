from __future__ import annotations

import json
import logging

import pytest

from winnow.config import default_config
from winnow.estimator.numerical import NumericalEstimator
from winnow.parser.numerical import FloatParser
from winnow.question import Question
from winnow.stopping import StoppingCriterion


@pytest.fixture
def sample_question() -> Question[float]:
    return Question(
        uid="protein",
        query="How many grams of protein in 100g of chicken breast?",
        parser=FloatParser(),
        estimator=NumericalEstimator(),
        stopping_criterion=StoppingCriterion(min_samples=5, max_queries=20),
    )


class TestBuildPrompt:
    def test_includes_query(self, sample_question: Question[float]) -> None:
        """Verify the built prompt contains the original query string."""
        prompt = sample_question.build_prompt()
        assert sample_question.query in prompt

    def test_includes_decline_instruction(
        self, sample_question: Question[float],
    ) -> None:
        """Verify the built prompt instructs the model how to decline."""
        prompt = sample_question.build_prompt()
        assert default_config.decline_keyword in prompt
        assert "insufficient information" in prompt.lower()


class TestLogExchange:
    def test_emits_jsonl_record_to_winnow_logger(
        self,
        sample_question: Question[float],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify log_exchange writes a JSON record to the 'winnow' logger."""
        with caplog.at_level(logging.DEBUG, logger="winnow"):
            sample_question.log_exchange(
                prompt="prompt body", response="response body",
            )

        assert len(caplog.records) == 1
        record = json.loads(caplog.records[0].message)
        assert record["question_uid"] == sample_question.uid
        assert record["prompt"] == "prompt body"
        assert record["response"] == "response body"
        assert "timestamp" in record
