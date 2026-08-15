from __future__ import annotations

import pytest

from winnow import (
    FloatParser,
    NoEstimate,
    NumericalEstimator,
    Question,
    QuestionBank,
    SampleState,
    SampleStatus,
    StoppingCriterion,
)
from winnow.config import default_config


@pytest.fixture
def sample_question() -> Question[float]:
    return Question(
        uid="protein",
        query="How many grams of protein in 100g of chicken breast?",
        parser=FloatParser(),
        estimator=NumericalEstimator(),
        stopping_criterion=StoppingCriterion(min_samples=5, max_queries=20),
    )


class TestQuestionBuildPrompt:
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


class TestQuestionSurface:
    def test_owns_its_sampling_definition(
        self, sample_question: Question[float],
    ) -> None:
        """Verify a question holds identity, query, parser and strategy directly."""
        assert sample_question.uid == "protein"
        assert sample_question.query == (
            "How many grams of protein in 100g of chicken breast?"
        )
        assert isinstance(sample_question.parser, FloatParser)
        assert isinstance(sample_question.estimator, NumericalEstimator)
        assert sample_question.stopping_criterion.min_samples == 5

    def test_has_no_exchange_logging_method(
        self, sample_question: Question[float],
    ) -> None:
        """Verify question remains a pure sampling value."""
        assert not hasattr(sample_question, "log_exchange")


class TestSelectWave:
    def test_selects_up_to_wave_size(self) -> None:
        """Verify select_wave returns at most wave_size questions."""
        bank = QuestionBank(
            [
                Question(
                    uid=f"q{i}",
                    query=f"Question {i}?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(min_samples=3),
                )
                for i in range(5)
            ]
        )

        states = {
            f"q{i}": SampleState(
                samples=(),
                decline_count=0,
                parse_failure_count=0,
                consecutive_declines=0,
                current_estimate=NoEstimate,
                current_confidence=0.0,
                status=SampleStatus.PENDING,
                failure_reason=None,
            )
            for i in range(5)
        }

        wave = bank.select_wave(states, wave_size=3)

        assert len(wave) == 3

    def test_returns_empty_when_all_complete(self) -> None:
        """Verify select_wave returns empty tuple when no questions remain."""
        bank = QuestionBank(
            [
                Question(
                    uid="q0",
                    query="Question?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(min_samples=1),
                ),
            ]
        )

        states = {
            "q0": SampleState(
                samples=(31.0,),
                decline_count=0,
                parse_failure_count=0,
                consecutive_declines=0,
                current_estimate=31.0,
                current_confidence=1.0,
                status=SampleStatus.CONVERGED,
                failure_reason=None,
            ),
        }

        wave = bank.select_wave(states, wave_size=5)

        assert wave == ()

    def test_no_duplicate_questions_in_wave(self) -> None:
        """Verify each question appears at most once per wave."""
        bank = QuestionBank(
            [
                Question(
                    uid="q0",
                    query="Question?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(min_samples=3),
                ),
            ]
        )

        states = {
            "q0": SampleState(
                samples=(),
                decline_count=0,
                parse_failure_count=0,
                consecutive_declines=0,
                current_estimate=NoEstimate,
                current_confidence=0.0,
                status=SampleStatus.PENDING,
                failure_reason=None,
            ),
        }

        # wave_size=5 but only 1 question, should get 1 not 5
        wave = bank.select_wave(states, wave_size=5)

        assert len(wave) == 1
        assert wave[0].uid == "q0"
