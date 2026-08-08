from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from winnow.exceptions import (
    ModelDeclinedError,
    ParseFailedError,
    UnknownInitialStateError,
)
from winnow.llm_client import LLMClient, LoggedLLMClient
from winnow.results import Estimate, NeedsReview
from winnow.state import NoEstimate, SampleState, SampleStatus

if TYPE_CHECKING:
    from winnow.question import Question, QuestionBank


class _RelevantSampleCounter(Protocol):
    def __call__(self, *, state: SampleState, estimate: object) -> int: ...


async def collect(
    *,
    bank: QuestionBank,
    query_fn: LLMClient,
    on_progress: Callable[[dict[str, SampleState], frozenset[str]], None] | None = None,
    initial_states: dict[str, SampleState] | None = None,
    wave_size: int = 1,
) -> dict[str, Estimate | NeedsReview]:
    """Collect estimates for all questions in the bank.

    Args:
        bank: The questions to answer.
        query_fn: Async function that sends a query string to the LLM
            and returns the raw response string.
        on_progress: Optional callback invoked after each wave with
            ``(states, wave_uids)`` — the current states and the set of
            question UIDs that were just dispatched in the wave that
            triggered the callback. Useful for displaying live progress
            in CLI applications.
        initial_states: Optional pre-populated states keyed by question UID.
            Useful for resuming from cached progress. States with
            CONVERGED status are automatically skipped by the collection loop.
        wave_size: Maximum number of queries dispatched concurrently per
            wave. Defaults to 1 (sequential behaviour).

    Returns:
        Mapping from question UID to either an Estimate (successful) or
        NeedsReview (collection failed and the item needs manual review).

    Raises:
        UnknownInitialStateError: If initial_states contains UIDs not
            present in the question bank.
    """
    if initial_states is not None:
        unknown_uids = initial_states.keys() - bank.questions.keys()
        if unknown_uids:
            raise UnknownInitialStateError(unknown_uids=unknown_uids)

    states = _initialise_states(bank, initial_states)
    logged_query_fn = LoggedLLMClient(query_fn=query_fn)

    while True:
        wave = bank.select_wave(states, wave_size=wave_size)
        if not wave:
            break

        responses = await asyncio.gather(
            *(logged_query_fn.query_prompt(q.prompt) for q in wave),
        )

        for question, response in zip(wave, responses):
            _process_response(question, response, states)

        if on_progress is not None:
            on_progress(states, frozenset(q.uid for q in wave))

    return _build_estimates(bank.questions, states)


def _initialise_states(
    bank: QuestionBank,
    initial_states: dict[str, SampleState] | None,
) -> dict[str, SampleState]:
    """Build initial state mapping for all questions in the bank."""
    return {
        q.uid: (
            initial_states[q.uid]
            if initial_states is not None and q.uid in initial_states
            else _make_empty_state()
        )
        for q in bank.questions.values()
    }


def _make_empty_state() -> SampleState:
    """Create a blank state for a question that has not yet been sampled."""
    return SampleState(
        samples=(),
        decline_count=0,
        parse_failure_count=0,
        consecutive_declines=0,
        current_estimate=NoEstimate,
        current_confidence=0.0,
        status=SampleStatus.PENDING,
        failure_reason=None,
    )


def _process_response(
    question: Question,
    response: str,
    states: dict[str, SampleState],
) -> None:
    """Parse a single response and update the question's state in place."""
    try:
        result = question.parser(response=response)
        new_samples = states[question.uid].samples + (result,)
        temp_state = _state_for_samples(new_samples)
        estimate = question.estimator.compute_estimate(state=temp_state)
        confidence = question.estimator.compute_confidence(
            state=temp_state,
            estimate=estimate,
        )
        effective_sample_count = _compute_effective_sample_count(
            question=question,
            state=temp_state,
            estimate=estimate,
        )
        states[question.uid] = states[question.uid].record_sample(
            result,
            estimate=estimate,
            confidence=confidence,
            effective_sample_count=effective_sample_count,
        )
    except ModelDeclinedError:
        states[question.uid] = states[question.uid].record_decline()
    except ParseFailedError:
        states[question.uid] = states[question.uid].record_parse_failure()

    states[question.uid] = states[question.uid].resolve_status(
        question.stopping_criterion,
    )


def _state_for_samples(samples: tuple[object, ...]) -> SampleState:
    """Create a minimal state for estimation from samples only."""
    return SampleState(
        samples=samples,
        decline_count=0,
        parse_failure_count=0,
        consecutive_declines=0,
        current_estimate=NoEstimate,
        current_confidence=0.0,
        status=SampleStatus.COLLECTING,
        failure_reason=None,
    )


def _compute_effective_sample_count(
    *,
    question: Question,
    state: SampleState,
    estimate: object,
) -> int | None:
    """Return estimator-specific sample count when one is defined."""
    counter = getattr(question.estimator, "count_relevant_samples", None)
    if not callable(counter):
        return None
    relevant_sample_counter = cast(_RelevantSampleCounter, counter)
    return relevant_sample_counter(state=state, estimate=estimate)


def _build_estimates(
    questions: dict[str, Question],
    states: dict[str, SampleState],
) -> dict[str, Estimate | NeedsReview]:
    """Build final estimates from collected states."""
    results: dict[str, Estimate | NeedsReview] = {}

    for q in questions.values():
        state = states[q.uid]

        if state.status is SampleStatus.CONVERGED:
            results[q.uid] = Estimate(
                value=state.current_estimate,
                confidence=state.current_confidence,
            )
        else:
            assert state.status is SampleStatus.NEEDS_REVIEW
            assert state.failure_reason is not None
            results[q.uid] = NeedsReview(reason=state.failure_reason)

    return results
