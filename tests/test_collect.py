from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING


from winnow.collect import collect
from winnow.estimator.boolean import BooleanEstimator
from winnow.estimator.numerical import NumericalEstimator
from winnow.parser.boolean import BooleanParser
from winnow.parser.numerical import FloatParser
from winnow.question import Question, QuestionBank
from winnow.stopping.primitives import MaxQueries, MinSamples
from winnow.types import Archetype

if TYPE_CHECKING:
    pass


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
                stopping_criterion=MinSamples(5),
            ),
        ])

        results = asyncio.run(collect(questions, query_fn=query_fn))

        assert "protein" in results
        assert results["protein"].value == 31.0
        assert results["protein"].sample_count == 5

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
                stopping_criterion=MinSamples(5),
            ),
        ])

        results = asyncio.run(collect(questions, query_fn=query_fn))

        assert "is_vegan" in results
        assert results["is_vegan"].value is True
        assert results["is_vegan"].sample_count == 5


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
                stopping_criterion=MinSamples(3),
            ),
            Question(
                uid="fat",
                query="How many grams of fat?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=MinSamples(3),
            ),
        ])

        results = asyncio.run(collect(questions, query_fn=query_fn))

        assert "protein" in results
        assert "fat" in results
        assert results["protein"].sample_count == 3
        assert results["fat"].sample_count == 3


class TestCollectDeclineHandling:
    def test_handles_declines(self) -> None:
        """Verify collect handles decline responses correctly."""
        responses = iter(["31", "UNKNOWN", "31", "UNKNOWN", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=MinSamples(3),
            ),
        ])

        results = asyncio.run(collect(questions, query_fn=query_fn))

        assert results["protein"].sample_count == 3
        assert results["protein"].decline_count == 2

    def test_insufficient_data_when_all_declines(self) -> None:
        """Verify INSUFFICIENT_DATA archetype when all responses are declines."""
        responses = iter(["UNKNOWN"] * 10)

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=MaxQueries(5),
            ),
        ])

        results = asyncio.run(collect(questions, query_fn=query_fn))

        assert results["protein"].archetype == Archetype.INSUFFICIENT_DATA
        assert results["protein"].value is None
        assert results["protein"].sample_count == 0


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
                stopping_criterion=MinSamples(3),
            ),
        ])

        results = asyncio.run(collect(questions, query_fn=query_fn))

        assert results["protein"].sample_count == 3
        # Parse failures don't affect decline count
        assert results["protein"].decline_count == 0


class TestCollectStoppingCriteria:
    def test_stops_at_max_queries(self) -> None:
        """Verify collect stops when MaxQueries is reached."""
        call_count = 0

        async def query_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "31"

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=MaxQueries(5),
            ),
        ])

        asyncio.run(collect(questions, query_fn=query_fn))

        assert call_count == 5


class TestCollectConfidence:
    def test_confidence_penalised_by_declines(self) -> None:
        """Verify confidence is reduced when there are many declines."""
        # 3 successful samples, 3 declines
        responses = iter(["31", "UNKNOWN", "31", "UNKNOWN", "31", "UNKNOWN"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank([
            Question(
                uid="protein",
                query="How many grams of protein?",
                parser=FloatParser(),
                estimator=NumericalEstimator(),
                stopping_criterion=MinSamples(3),
            ),
        ])

        results = asyncio.run(collect(questions, query_fn=query_fn))

        # Confidence should be penalised (raw confidence * decline_penalty)
        # With 3 samples and 3 declines, penalty = 1 - 3/6 = 0.5
        assert results["protein"].confidence < 1.0
