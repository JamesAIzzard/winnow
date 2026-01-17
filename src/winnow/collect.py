from __future__ import annotations

from typing import TYPE_CHECKING

from winnow.config import default_config
from winnow.exceptions import ParseFailedError
from winnow.types import Archetype, Estimate, SampleState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from winnow.question import Question, QuestionBank


async def collect(
    bank: QuestionBank,
    *,
    query_fn: Callable[[str], Awaitable[str]],
) -> dict[str, Estimate[object]]:
    """Collect estimates for all questions in the bank.

    Args:
        bank: The questions to answer.
        query_fn: Async function that sends a query string to the LLM
            and returns the raw response string.

    Returns:
        Mapping from question UID to its estimate.
    """
    states: dict[str, SampleState[object]] = {
        q.uid: SampleState(
            samples=(),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )
        for q in bank.questions
    }

    while (question := bank.select_next(states)) is not None:
        response = await query_fn(question.query)

        try:
            result = question.parser(response=response)
            if result is None:
                states[question.uid] = _record_decline(states[question.uid])
            else:
                states[question.uid] = _record_sample(states[question.uid], result)
        except ParseFailedError:
            states[question.uid] = _record_parse_failure(states[question.uid])

    return _build_estimates(bank.questions, states)


def _record_sample(state: SampleState[object], value: object) -> SampleState[object]:
    """Record a successful sample."""
    return SampleState(
        samples=state.samples + (value,),
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=0,
    )


def _record_decline(state: SampleState[object]) -> SampleState[object]:
    """Record a decline from the model."""
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count + 1,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=state.consecutive_declines + 1,
    )


def _record_parse_failure(state: SampleState[object]) -> SampleState[object]:
    """Record a parse failure."""
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count + 1,
        consecutive_declines=0,
    )


def _build_estimates(
    questions: list[Question[object]],
    states: dict[str, SampleState[object]],
) -> dict[str, Estimate[object]]:
    """Build final estimates from collected states."""
    estimates: dict[str, Estimate[object]] = {}

    for q in questions:
        state = states[q.uid]

        if len(state.samples) == 0:
            estimates[q.uid] = Estimate(
                value=None,
                confidence=0.0,
                archetype=Archetype.INSUFFICIENT_DATA,
                sample_count=0,
                decline_count=state.decline_count,
                samples=(),
            )
            continue

        value = q.estimator.compute_estimate(samples=state.samples)
        raw_confidence = q.estimator.compute_confidence(
            samples=state.samples, estimate=value
        )

        # Adjust confidence for decline rate
        total_attempts = len(state.samples) + state.decline_count
        if total_attempts > 0:
            decline_penalty = 1.0 - (state.decline_count / total_attempts)
        else:
            decline_penalty = 1.0
        confidence = raw_confidence * decline_penalty

        archetype = _classify_archetype(q, state, confidence)

        estimates[q.uid] = Estimate(
            value=value,
            confidence=confidence,
            archetype=archetype,
            sample_count=len(state.samples),
            decline_count=state.decline_count,
            samples=state.samples,
        )

    return estimates


def _classify_archetype(
    question: Question[object],
    state: SampleState[object],
    confidence: float,
) -> Archetype:
    """Classify the archetype of an estimate based on convergence behaviour."""
    threshold = question.stopping_criterion.confidence_threshold

    if confidence >= threshold:
        if len(state.samples) < default_config.confident_sample_threshold:
            return Archetype.CONFIDENT
        return Archetype.ACCEPTABLE

    if confidence >= default_config.acceptable_confidence_threshold:
        return Archetype.ACCEPTABLE

    return Archetype.UNCERTAIN
