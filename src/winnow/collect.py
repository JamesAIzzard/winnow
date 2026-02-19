from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from winnow.config import default_config
from winnow.exceptions import ModelDeclinedError, ParseFailedError, UnknownInitialStateError
from winnow.types import (
    Estimate,
    NeedsReview,
    NoEstimate,
    ReviewReason,
    SampleState,
    SampleStatus,
)

from winnow.stopping import StoppingCriterion

if TYPE_CHECKING:
    from winnow.question import Question, QuestionBank


async def collect(
    *,
    bank: QuestionBank,
    query_fn: Callable[[str], Awaitable[str]],
    on_progress: Callable[[dict[str, SampleState]], None] | None = None,
    initial_states: dict[str, SampleState] | None = None,
    wave_size: int = 1,
) -> dict[str, Estimate | NeedsReview]:
    """Collect estimates for all questions in the bank.

    Args:
        bank: The questions to answer.
        query_fn: Async function that sends a query string to the LLM
            and returns the raw response string.
        on_progress: Optional callback invoked after each wave with the
            current states. Useful for displaying progress in CLI applications.
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

    while True:
        wave = bank.select_wave(states, wave_size=wave_size)
        if not wave:
            break

        prompts = [_build_prompt(q.query) for q in wave]
        responses = await asyncio.gather(*(query_fn(p) for p in prompts))

        for question, response in zip(wave, responses):
            _process_response(question, response, states)

        if on_progress is not None:
            on_progress(states)

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
        states[question.uid] = _record_sample(
            state=states[question.uid],
            value=result,
            current_estimate=estimate,
            current_confidence=confidence,
        )
    except ModelDeclinedError:
        states[question.uid] = _record_decline(states[question.uid])
    except ParseFailedError:
        states[question.uid] = _record_parse_failure(states[question.uid])

    states[question.uid] = _resolve_status(
        states[question.uid], question.stopping_criterion,
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


def _build_prompt(query: str) -> str:
    """Build the full prompt including decline instruction."""
    decline_instruction = (
        f"If you have insufficient information to answer, "
        f"respond with only: {default_config.decline_keyword}"
    )
    return f"{query}\n\n{decline_instruction}"


def _record_sample(
    *,
    state: SampleState,
    value: object,
    current_estimate: object,
    current_confidence: float,
) -> SampleState:
    """Record a successful sample."""
    return SampleState(
        samples=state.samples + (value,),
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=0,
        current_estimate=current_estimate,
        current_confidence=current_confidence,
        status=SampleStatus.COLLECTING,
        failure_reason=None,
    )


def _record_decline(state: SampleState) -> SampleState:
    """Record a decline from the model."""
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count + 1,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=state.consecutive_declines + 1,
        current_estimate=state.current_estimate,
        current_confidence=state.current_confidence,
        status=SampleStatus.COLLECTING,
        failure_reason=None,
    )


def _record_parse_failure(state: SampleState) -> SampleState:
    """Record a parse failure."""
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count + 1,
        consecutive_declines=0,
        current_estimate=state.current_estimate,
        current_confidence=state.current_confidence,
        status=SampleStatus.COLLECTING,
        failure_reason=None,
    )


def _resolve_status(
    state: SampleState,
    criterion: StoppingCriterion,
) -> SampleState:
    """Set converged and failure_reason based on the stopping criterion.

    If the criterion says sampling should stop, determines whether it stopped
    successfully (converged) or unsuccessfully (failure_reason set).
    Returns the state unchanged if sampling should continue.
    """
    if not criterion.should_stop(state):
        return state

    converged = (
        len(state.samples) >= criterion.min_samples
        and state.current_confidence >= criterion.confidence_threshold
    )

    if converged:
        status = SampleStatus.CONVERGED
        failure_reason = None
    else:
        status = SampleStatus.NEEDS_REVIEW
        failure_reason = _determine_failure_reason(state, criterion)

    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=state.consecutive_declines,
        current_estimate=state.current_estimate,
        current_confidence=state.current_confidence,
        status=status,
        failure_reason=failure_reason,
    )


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


def _determine_failure_reason(
    state: SampleState,
    criterion: StoppingCriterion,
) -> ReviewReason:
    """Determine why estimation failed for a question."""
    if state.consecutive_declines >= criterion.max_consecutive_declines:
        return ReviewReason.MAX_CONSECUTIVE_DECLINES
    if state.parse_failure_count >= criterion.max_parse_failures:
        return ReviewReason.MAX_PARSE_FAILURES
    if len(state.samples) == 0:
        return ReviewReason.MAX_QUERIES
    return ReviewReason.INSUFFICIENT_CONFIDENCE
