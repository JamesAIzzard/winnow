from __future__ import annotations

import asyncio
from typing import Any

import pytest

from winnow.collect import collect
from winnow.estimator.boolean import BooleanEstimator
from winnow.estimator.numerical import NumericalEstimator
from winnow.estimator.optional_int import OptionalIntEstimator
from winnow.exceptions import UnknownInitialStateError
from winnow.parser.boolean import BooleanParser
from winnow.parser.numerical import FloatParser
from winnow.parser.optional_bounded import OptionalBoundedIntParser
from winnow.question import Prompt, Question, QuestionBank
from winnow.results import Estimate, NeedsReview
from winnow.state import NoEstimate, ReviewReason, SampleState, SampleStatus
from winnow.stopping import StoppingCriterion


def _question(
    *,
    estimator: Any,
    stopping_criterion: StoppingCriterion,
    prompt: Prompt[Any] | None = None,
    uid: str | None = None,
    query: str | None = None,
    parser: Any | None = None,
) -> Question[Any]:
    if prompt is None:
        assert uid is not None
        assert query is not None
        assert parser is not None
        prompt = Prompt(uid=uid, query=query, parser=parser)
    return Question(
        prompt=prompt,
        estimator=estimator,
        stopping_criterion=stopping_criterion,
    )


class TestCollectBasic:
    def test_collects_numerical_samples(self) -> None:
        """Verify collect gathers numerical samples and produces estimate."""
        responses = iter(["31", "30", "31", "32", "31"])

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                _question(
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
                _question(
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

    def test_optional_int_requires_min_samples_from_numeric_branch(self) -> None:
        """Verify optional integer convergence waits for enough numeric samples."""
        responses = iter(["None", "55", "55", "55", "55", "55"])
        progress: list[dict[str, SampleState]] = []

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                _question(
                    uid="glycaemic_index",
                    query="What is the GI?",
                    parser=OptionalBoundedIntParser(min_value=0, max_value=100),
                    estimator=OptionalIntEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=5, max_queries=10, confidence_threshold=0.5
                    ),
                ),
            ]
        )

        results = asyncio.run(
            collect(
                bank=questions,
                query_fn=query_fn,
                on_progress=lambda states, _wave: progress.append(states),
            )
        )

        result = results["glycaemic_index"]
        assert isinstance(result, Estimate)
        assert result.value == 55
        assert progress[-1]["glycaemic_index"].query_count == 6

    def test_optional_int_requires_min_samples_from_none_branch(self) -> None:
        """Verify optional integer convergence waits for enough None samples."""
        responses = iter(["55", "None", "None", "None", "None", "None"])
        progress: list[dict[str, SampleState]] = []

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                _question(
                    uid="glycaemic_index",
                    query="What is the GI?",
                    parser=OptionalBoundedIntParser(min_value=0, max_value=100),
                    estimator=OptionalIntEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=5, max_queries=10, confidence_threshold=0.5
                    ),
                ),
            ]
        )

        results = asyncio.run(
            collect(
                bank=questions,
                query_fn=query_fn,
                on_progress=lambda states, _wave: progress.append(states),
            )
        )

        result = results["glycaemic_index"]
        assert isinstance(result, Estimate)
        assert result.value is None
        assert progress[-1]["glycaemic_index"].query_count == 6


class TestCollectDeclineHandling:
    def test_returns_needs_review_when_all_declines(self) -> None:
        """Verify NeedsReview returned when all responses are declines."""
        responses = iter(["DECLINE"] * 10)

        async def query_fn(prompt: str) -> str:
            return next(responses)

        questions = QuestionBank(
            [
                _question(
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
                _question(
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
                _question(
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
                _question(
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

        def on_progress(states: dict[str, SampleState], _wave: frozenset[str]) -> None:
            progress_states.append(dict(states))

        questions = QuestionBank(
            [
                _question(
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

        def on_progress(states: dict[str, SampleState], _wave: frozenset[str]) -> None:
            progress_states.append(dict(states))

        questions = QuestionBank(
            [
                _question(
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

        def on_progress(states: dict[str, SampleState], _wave: frozenset[str]) -> None:
            progress_states.append(dict(states))

        questions = QuestionBank(
            [
                _question(
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
                _question(
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
                _question(
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
                _question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                _question(
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
                _question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                _question(
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


class TestCollectWaveBasic:
    def test_wave_collects_multiple_questions(self) -> None:
        """Verify wave-based collection produces estimates for all questions."""
        response_map = {
            "protein": iter(["31", "31", "31"]),
            "fat": iter(["3", "3", "3"]),
        }

        async def query_fn(prompt: str) -> str:
            if "protein" in prompt:
                return next(response_map["protein"])
            return next(response_map["fat"])

        bank = QuestionBank(
            [
                _question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                _question(
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

        results = asyncio.run(
            collect(bank=bank, query_fn=query_fn, wave_size=2)
        )

        assert isinstance(results["protein"], Estimate)
        assert results["protein"].value == 31.0
        assert isinstance(results["fat"], Estimate)
        assert results["fat"].value == 3.0


class TestCollectWaveConcurrency:
    def test_wave_dispatches_concurrently(self) -> None:
        """Verify queries within a wave are dispatched concurrently."""
        in_flight: list[str] = []
        max_concurrent = 0

        async def query_fn(prompt: str) -> str:
            nonlocal max_concurrent
            in_flight.append(prompt)
            max_concurrent = max(max_concurrent, len(in_flight))
            await asyncio.sleep(0)
            in_flight.pop()
            if "protein" in prompt:
                return "31"
            return "3"

        bank = QuestionBank(
            [
                _question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                _question(
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

        asyncio.run(collect(bank=bank, query_fn=query_fn, wave_size=2))

        assert max_concurrent == 2

    def test_wave_size_larger_than_bank_caps_at_bank_size(self) -> None:
        """Verify wave_size exceeding bank size dispatches one per question."""
        call_count = 0

        async def query_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "31"

        bank = QuestionBank(
            [
                _question(
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

        results = asyncio.run(
            collect(bank=bank, query_fn=query_fn, wave_size=10)
        )

        assert isinstance(results["protein"], Estimate)
        # Only 3 queries needed (min_samples=3), one per wave
        assert call_count == 3


class TestCollectWaveProgress:
    def test_progress_called_once_per_wave(self) -> None:
        """Verify on_progress is called once per wave, not once per query."""
        response_map = {
            "protein": iter(["31", "31", "31"]),
            "fat": iter(["3", "3", "3"]),
        }
        progress_call_count = 0

        async def query_fn(prompt: str) -> str:
            if "protein" in prompt:
                return next(response_map["protein"])
            return next(response_map["fat"])

        def on_progress(
            states: dict[str, SampleState], _wave: frozenset[str],
        ) -> None:
            nonlocal progress_call_count
            progress_call_count += 1

        bank = QuestionBank(
            [
                _question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                _question(
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

        asyncio.run(
            collect(
                bank=bank,
                query_fn=query_fn,
                on_progress=on_progress,
                wave_size=2,
            )
        )

        # 2 questions, 3 samples each, wave_size=2 => 3 waves (one per round)
        assert progress_call_count == 3


class TestCollectWaveStopping:
    def test_wave_respects_stopping_between_waves(self) -> None:
        """Verify a question that converges mid-run is dropped from subsequent waves."""
        call_count = 0

        async def query_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if "protein" in prompt:
                return "31"
            # fat gives varied answers to prevent early convergence
            return str(call_count * 10)

        bank = QuestionBank(
            [
                _question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                _question(
                    uid="fat",
                    query="How many grams of fat?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=5, confidence_threshold=0.99
                    ),
                ),
            ]
        )

        results = asyncio.run(
            collect(bank=bank, query_fn=query_fn, wave_size=2)
        )

        # Protein should converge quickly with identical answers
        assert isinstance(results["protein"], Estimate)
        assert results["protein"].value == 31.0
        # Fat should fail with varied answers and high threshold
        assert isinstance(results["fat"], NeedsReview)

    def test_wave_handles_mixed_declines_and_samples(self) -> None:
        """Verify wave correctly handles a mix of declines and valid samples."""
        response_map = {
            "protein": iter(["31", "DECLINE", "31", "31"]),
            "fat": iter(["DECLINE", "3", "3", "3"]),
        }

        async def query_fn(prompt: str) -> str:
            if "protein" in prompt:
                return next(response_map["protein"])
            return next(response_map["fat"])

        bank = QuestionBank(
            [
                _question(
                    uid="protein",
                    query="How many grams of protein?",
                    parser=FloatParser(),
                    estimator=NumericalEstimator(),
                    stopping_criterion=StoppingCriterion(
                        min_samples=3, max_queries=100
                    ),
                ),
                _question(
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

        results = asyncio.run(
            collect(bank=bank, query_fn=query_fn, wave_size=2)
        )

        assert isinstance(results["protein"], Estimate)
        assert results["protein"].value == 31.0
        assert isinstance(results["fat"], Estimate)
        assert results["fat"].value == 3.0


class TestSelectWave:
    def test_selects_up_to_wave_size(self) -> None:
        """Verify select_wave returns at most wave_size questions."""
        bank = QuestionBank(
            [
                _question(
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
                _question(
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
                _question(
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
