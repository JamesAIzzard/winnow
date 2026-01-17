from __future__ import annotations

import asyncio

import pytest

from winnow.collect import collect
from winnow.exceptions import EstimationFailedError
from winnow.estimator.boolean import BooleanEstimator
from winnow.estimator.numerical import NumericalEstimator
from winnow.parser.boolean import BooleanParser
from winnow.parser.numerical import FloatParser
from winnow.question import Question, QuestionBank
from winnow.stopping import StoppingCriterion


class TestCollectBasic:
    def test_collects_numerical_samples(self) -> None:
        """Verify collect gathers numerical samples and produces estimate."""
        responses = iter(["31", "30", "31", "32", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(min_samples=5, max_queries=100),
            ),
        ])

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert "protein" in results
        assert results["protein"].value == 31.0

    def test_collects_boolean_samples(self) -> None:
        """Verify collect gathers boolean samples and produces estimate."""
        responses = iter(["yes", "yes", "no", "yes", "yes"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="is_vegan",
                query="Is this vegan?",
                parser=BooleanParser(),
                estimator=BooleanEstimator(),
                stopping_criterion=StoppingCriterion(
                    min_samples=5, max_queries=5, confidence_threshold=1.0
                ),
            ),
        ])

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert "is_vegan" in results
        assert results["is_vegan"].value is True


class TestCollectMultipleQuestions:
    def test_collects_all_questions(self) -> None:
        """Verify collect handles multiple questions."""
        response_map = {
            "protein": iter(["31", "31", "30"]),
            "fat": iter(["3", "4", "3"]),
        }

        async def query_fn(prompt: str) -> str:
            if "protein" in prompt:
                return next(response_map["protein"])
            return next(response_map["fat"])

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(min_samples=3, max_queries=100),
            ),
            Question(
                uid="fat",
                query="How many grams of fat?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(min_samples=3, max_queries=100),
            ),
        ])

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert "protein" in results
        assert "fat" in results
        assert results["protein"].value is not None
        assert results["fat"].value is not None


class TestCollectDeclineHandling:
    def test_handles_declines(self) -> None:
        """Verify collect handles decline responses correctly."""
        responses = iter(["31", "DECLINE", "31", "DECLINE", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(min_samples=3, max_queries=100),
            ),
        ])

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        # Declines are skipped but valid samples still produce an estimate
        assert results["protein"].value == 31.0

    def test_raises_when_all_declines(self) -> None:
        """Verify EstimationFailedError raised when all responses are declines."""
        responses = iter(["DECLINE"] * 10)

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(min_samples=1, max_queries=5),
            ),
        ])

        with pytest.raises(EstimationFailedError) as exc_info:
            asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert exc_info.value.question_uid == "protein"


class TestCollectParseFailures:
    def test_handles_parse_failures(self) -> None:
        """Verify collect handles parse failures correctly."""
        responses = iter(["31", "invalid", "30", "garbage", "32"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(min_samples=3, max_queries=100),
            ),
        ])

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert results["protein"].value is not None
        # Parse failures don't penalise confidence (only declines do)
        assert results["protein"].confidence > 0.0


class TestCollectStoppingCriteria:
    def test_stops_at_max_queries(self) -> None:
        """Verify collect stops when MaxQueries is reached."""
        call_count = 0
        # Use varied responses to prevent early confidence stopping
        values = iter(["10", "20", "30", "40", "50"])

        async def query_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return next(values)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(
                    min_samples=1, max_queries=5, confidence_threshold=0.99
                ),
            ),
        ])

        asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert call_count == 5


class TestCollectConfidence:
    def test_confidence_based_on_sample_agreement(self) -> None:
        """Verify confidence is based purely on sample agreement."""
        # 3 identical samples - should give full confidence
        responses = iter(["31", "31", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=StoppingCriterion(min_samples=3, max_queries=100),
            ),
        ])

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        # Identical samples should give full confidence
        assert results["protein"].confidence == 1.0
