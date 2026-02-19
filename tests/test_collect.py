from __future__ import annotations

import asyncio

import pytest

from winnow.collect import collect
from winnow.estimator.boolean import BooleanEstimator
from winnow.estimator.numerical import NumericalEstimator
from winnow.exceptions import UnknownInitialStateError
from winnow.parser.boolean import BooleanParser
from winnow.parser.numerical import FloatParser
from winnow.question import Question, QuestionBank
from winnow.stopping import StoppingCriterion
from winnow.types import (
    Estimate,
    NeedsReview,
    NoEstimate,
    ReviewReason,
    SampleState,
    SampleStatus,
)


class TestCollectBasic:
    def test_collects_numerical_samples(self) -> None:
        """Verify collect gathers numerical samples and produces estimate."""
        responses = iter(["31", "30", "31", "32", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=5, max_queries=100
                    ),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert "protein" in results
        result = results["protein"]
        assert isinstance(result, Estimate)
        assert result.value == 31.0

    def test_collects_boolean_samples(self) -> None:
        """Verify collect gathers boolean samples and produces estimate."""
        responses = iter(["yes", "yes", "no", "yes", "yes"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                Question(
                    uid="is_vegan",
                    query="Is this vegan?",
                    parser=BooleanParser(),
                    estimator=BooleanEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=5, max_queries=5, confidence_threshold=0.5
                    ),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert "is_vegan" in results
        result = results["is_vegan"]
        assert isinstance(result, Estimate)
        assert result.value is True


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

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                Question(
                    uid="fat",
                    query="How many grams of fat?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert "protein" in results
        assert "fat" in results
        assert isinstance(results["protein"], Estimate)
        assert isinstance(results["fat"], Estimate)


class TestCollectDeclineHandling:
    def test_handles_declines(self) -> None:
        """Verify collect handles decline responses correctly."""
        responses = iter(["31", "DECLINE", "31", "DECLINE", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        # Declines are skipped but valid samples still produce an estimate
        result = results["protein"]
        assert isinstance(result, Estimate)
        assert result.value == 31.0

    def test_returns_needs_review_when_all_declines(self) -> None:
        """Verify NeedsReview returned when all responses are declines."""
        responses = iter(["DECLINE"] * 10)

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(min_samples=1, max_queries=5),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert isinstance(results["protein"], NeedsReview)
        assert results["protein"].reason is ReviewReason.MAX_CONSECUTIVE_DECLINES


class TestCollectParseFailures:
    def test_handles_parse_failures(self) -> None:
        """Verify collect handles parse failures correctly."""
        responses = iter(["31", "invalid", "30", "garbage", "32"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        result = results["protein"]
        assert isinstance(result, Estimate)
        # Parse failures don't prevent convergence when enough valid samples exist
        assert result.confidence > 0.0


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

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=1, max_queries=5, confidence_threshold=0.99
                    ),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        assert call_count == 5
        assert isinstance(results["protein"], NeedsReview)
        assert results["protein"].reason is ReviewReason.INSUFFICIENT_CONFIDENCE


class TestCollectConfidence:
    def test_confidence_based_on_sample_agreement(self) -> None:
        """Verify confidence is based purely on sample agreement."""
        # 3 identical samples - should give full confidence
        responses = iter(["31", "31", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        results = asyncio.run(collect(bank=questions, query_fn=query_fn))

        # Identical samples should give full confidence
        result = results["protein"]
        assert isinstance(result, Estimate)
        assert result.confidence == 1.0


class TestCollectProgressConvergence:
    def test_converged_state_in_progress_callback(self) -> None:
        """Verify converged is True in progress callback when threshold is met."""
        responses = iter(["31", "31", "31"])
        progress_states: list[dict[str, SampleState]] = []

        async def query_fn(prompt: str) -> str:
            return next(responses)

        def on_progress(states: dict[str, SampleState]) -> None:
            progress_states.append(dict(states))

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        asyncio.run(collect(bank=questions, query_fn=query_fn, on_progress=on_progress))

        # Final progress callback should show converged
        final_state = progress_states[-1]["protein"]
        assert final_state.status is SampleStatus.CONVERGED
        assert final_state.failure_reason is None

    def test_failure_reason_in_progress_callback(self) -> None:
        """Verify failure_reason is set when collection fails."""
        responses = iter(["DECLINE"] * 10)
        progress_states: list[dict[str, SampleState]] = []

        async def query_fn(prompt: str) -> str:
            return next(responses)

        def on_progress(states: dict[str, SampleState]) -> None:
            progress_states.append(dict(states))

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(min_samples=1, max_queries=10),
                ),
            ]
        )

        asyncio.run(collect(bank=questions, query_fn=query_fn, on_progress=on_progress))

        final_state = progress_states[-1]["protein"]
        assert final_state.status is SampleStatus.NEEDS_REVIEW
        assert final_state.failure_reason is ReviewReason.MAX_CONSECUTIVE_DECLINES

    def test_intermediate_states_not_converged(self) -> None:
        """Verify intermediate progress callbacks show not-yet-converged states."""
        responses = iter(["31", "31", "31"])
        progress_states: list[dict[str, SampleState]] = []

        async def query_fn(prompt: str) -> str:
            return next(responses)

        def on_progress(states: dict[str, SampleState]) -> None:
            progress_states.append(dict(states))

        questions = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        asyncio.run(collect(bank=questions, query_fn=query_fn, on_progress=on_progress))

        # First two callbacks should still be collecting (need min_samples=3)
        assert progress_states[0]["protein"].status is SampleStatus.COLLECTING
        assert progress_states[0]["protein"].failure_reason is None
        assert progress_states[1]["protein"].status is SampleStatus.COLLECTING
        assert progress_states[1]["protein"].failure_reason is None


class TestCollectInitialStates:
    def test_raises_on_unknown_initial_state_uids(self) -> None:
        """Verify UnknownInitialStateError raised when initial_states has UIDs not in bank."""

        async def query_fn(prompt: str) -> str:
            return "31"

        bank = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        bogus_state = SampleState(
            samples=(42.0,),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
            current_estimate=42.0,
            current_confidence=1.0,
            status=SampleStatus.CONVERGED,
            failure_reason=None,
        )

        with pytest.raises(UnknownInitialStateError) as exc_info:
            asyncio.run(
                collect(
                    bank=bank,
                    query_fn=query_fn,
                    initial_states={"not_in_bank": bogus_state},
                )
            )

        assert exc_info.value.unknown_uids == {"not_in_bank"}

    def test_raises_on_mix_of_known_and_unknown_uids(self) -> None:
        """Verify error reports only the unknown UIDs when some are valid."""

        async def query_fn(prompt: str) -> str:
            return "31"

        bank = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        converged_state = SampleState(
            samples=(31.0, 31.0, 31.0),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
            current_estimate=31.0,
            current_confidence=1.0,
            status=SampleStatus.CONVERGED,
            failure_reason=None,
        )

        with pytest.raises(UnknownInitialStateError) as exc_info:
            asyncio.run(
                collect(
                    bank=bank,
                    query_fn=query_fn,
                    initial_states={
                        "protein": converged_state,
                        "stale_uid": converged_state,
                        "another_stale": converged_state,
                    },
                )
            )

        assert exc_info.value.unknown_uids == {"stale_uid", "another_stale"}

    def test_skips_converged_initial_states(self) -> None:
        """Verify converged initial states are not re-queried."""
        query_count = 0

        async def query_fn(prompt: str) -> str:
            nonlocal query_count
            query_count += 1
            return "5"

        bank = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                Question(
                    uid="fat",
                    query="How many grams of fat?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        # Pre-converged state for protein; fat starts fresh
        cached_protein = SampleState(
            samples=(31.0, 31.0, 31.0),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
            current_estimate=31.0,
            current_confidence=1.0,
            status=SampleStatus.CONVERGED,
            failure_reason=None,
        )

        results = asyncio.run(
            collect(
                bank=bank,
                query_fn=query_fn,
                initial_states={"protein": cached_protein},
            )
        )

        # Protein should use the cached estimate without any queries
        assert isinstance(results["protein"], Estimate)
        assert results["protein"].value == 31.0

        # Fat should have been queried normally
        assert isinstance(results["fat"], Estimate)
        assert results["fat"].value == 5.0

        # Only fat was queried (3 samples), protein was skipped
        assert query_count == 3

    def test_skips_needs_review_initial_states(self) -> None:
        """Verify NEEDS_REVIEW initial states are not re-queried."""
        query_count = 0

        async def query_fn(prompt: str) -> str:
            nonlocal query_count
            query_count += 1
            return "5"

        bank = QuestionBank(
            [
                Question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                Question(
                    uid="fat",
                    query="How many grams of fat?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
            ]
        )

        # Protein previously failed; fat starts fresh
        failed_protein = SampleState(
            samples=(),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=5,
            current_estimate=NoEstimate,
            current_confidence=0.0,
            status=SampleStatus.NEEDS_REVIEW,
            failure_reason=ReviewReason.MAX_CONSECUTIVE_DECLINES,
        )

        results = asyncio.run(
            collect(
                bank=bank,
                query_fn=query_fn,
                initial_states={"protein": failed_protein},
            )
        )

        # Protein should be returned as NeedsReview without any queries
        assert isinstance(results["protein"], NeedsReview)
        assert results["protein"].reason is ReviewReason.MAX_CONSECUTIVE_DECLINES

        # Fat should have been queried normally
        assert isinstance(results["fat"], Estimate)
        assert results["fat"].value == 5.0

        # Only fat was queried (3 samples), protein was skipped
        assert query_count == 3
