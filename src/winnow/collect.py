from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from winnow.config import default_config
from winnow.exceptions import EstimationFailedError, ModelDeclinedError, ParseFailedError
from winnow.types import Estimate, SampleState

if TYPE_CHECKING:
    from winnow.question import Question, QuestionBank


async def collect(
    *,
    bank: QuestionBank,
    query_fn: Callable[[str], Awaitable[str]],
    on_progress: Callable[[dict[str, SampleState]], None] | None = None,
) -> dict[str, Estimate]:
    """Collect estimates for all questions in the bank.

    Args:
        bank: The questions to answer.
        query_fn: Async function that sends a query string to the LLM
            and returns the raw response string.
        on_progress: Optional callback invoked after each query with the
            current states. Useful for displaying progress in CLI applications.

    Returns:
        Mapping from question UID to its estimate.
    """
    states: dict[str, SampleState] = {
        q.uid: SampleState(
            samples=(),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
            current_estimate=None,
            current_confidence=0.0,
        )
        for q in bank.questions.values()
    }

    while True:
        question = bank.select_next(states)
        if question is None:
            break

        prompt = _build_prompt(question.query)
        response = await query_fn(prompt)

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

        if on_progress is not None:
            on_progress(states)

    return _build_estimates(bank.questions, states)


def _state_for_samples(samples: tuple[object, ...]) -> SampleState:
    """Create a minimal state for estimation from samples only."""
    return SampleState(
        samples=samples,
        decline_count=0,
        parse_failure_count=0,
        consecutive_declines=0,
        current_estimate=None,
        current_confidence=0.0,
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
    )


def _build_estimates(
    questions: dict[str, Question],
    states: dict[str, SampleState],
) -> dict[str, Estimate]:
    """Build final estimates from collected states."""
    estimates: dict[str, Estimate] = {}

    for q in questions.values():
        state = states[q.uid]

        if len(state.samples) == 0 or state.current_estimate is None:
            raise EstimationFailedError(question_uid=q.uid)

        estimates[q.uid] = Estimate(
            value=state.current_estimate,
            confidence=state.current_confidence,
        )

    return estimates
