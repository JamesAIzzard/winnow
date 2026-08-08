from __future__ import annotations

import pytest

from winnow.config import default_config
from winnow.estimator.numerical import NumericalEstimator
from winnow.parser.numerical import FloatParser
from winnow.question import Prompt, Question
from winnow.stopping import StoppingCriterion


@pytest.fixture
def sample_question() -> Question[float]:
    return Question(
        prompt=Prompt(
            uid="protein",
            query="How many grams of protein in 100g of chicken breast?",
            parser=FloatParser(),
        ),
        estimator=NumericalEstimator(),
        stopping_criterion=StoppingCriterion(min_samples=5, max_queries=20),
    )


class TestPromptBuildPrompt:
    def test_includes_query(self) -> None:
        """Verify the built prompt contains the original query string."""
        source = Prompt(
            uid="protein",
            query="How many grams of protein in 100g of chicken breast?",
            parser=FloatParser(),
        )

        prompt = source.build_prompt()

        assert source.query in prompt

    def test_includes_decline_instruction(self) -> None:
        """Verify the built prompt instructs the model how to decline."""
        source = Prompt(
            uid="protein",
            query="How many grams of protein in 100g of chicken breast?",
            parser=FloatParser(),
        )

        prompt = source.build_prompt()

        assert default_config.decline_keyword in prompt
        assert "insufficient information" in prompt.lower()


class TestQuestionComposition:
    def test_delegates_prompt_fields(self, sample_question: Question[float]) -> None:
        """Verify question exposes prompt fields through composition."""
        assert sample_question.uid == "protein"
        assert sample_question.query == (
            "How many grams of protein in 100g of chicken breast?"
        )
        assert isinstance(sample_question.parser, FloatParser)

    def test_has_no_exchange_logging_method(
        self, sample_question: Question[float],
    ) -> None:
        """Verify question remains a pure sampling value."""
        assert not hasattr(sample_question, "log_exchange")
