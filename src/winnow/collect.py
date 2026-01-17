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
) -> dict[str, Estimate]:
    """Collect estimates for all questions in the bank.

    Args:
        bank: The questions to answer.
        query_fn: Async function that sends a query string to the LLM
            and returns the raw response string.

    Returns:
        Mapping from question UID to its estimate.
    """
    states: dict[str, SampleState] = {
        q.uid: SampleState(
            samples=(),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
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
            states[question.uid] = _record_sample(states[question.uid], result)
        except ModelDeclinedError:
            states[question.uid] = _record_decline(states[question.uid])
        except ParseFailedError:
            states[question.uid] = _record_parse_failure(states[question.uid])

    return _build_estimates(bank.questions, states)


def _build_prompt(query: str) -> str:
    """Build the full prompt including decline instruction."""
    decline_instruction = (
        f"If you have insufficient information to answer, "
        f"respond with only: {default_config.decline_keyword}"
    )
    return f"{query}\n\n{decline_instruction}"


def _record_sample(state: SampleState, value: object) -> SampleState:
    """Record a successful sample."""
    return SampleState(
        samples=state.samples + (value,),
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=0,
    )


def _record_decline(state: SampleState) -> SampleState:
    """Record a decline from the model."""
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count + 1,
        parse_failure_count=state.parse_failure_count,
        consecutive_declines=state.consecutive_declines + 1,
    )


def _record_parse_failure(state: SampleState) -> SampleState:
    """Record a parse failure."""
    return SampleState(
        samples=state.samples,
        decline_count=state.decline_count,
        parse_failure_count=state.parse_failure_count + 1,
        consecutive_declines=0,
    )


def _build_estimates(
    questions: dict[str, Question],
    states: dict[str, SampleState],
) -> dict[str, Estimate]:
    """Build final estimates from collected states."""
    estimates: dict[str, Estimate] = {}

    for q in questions.values():
        state = states[q.uid]

        if len(state.samples) == 0:
            raise EstimationFailedError(question_uid=q.uid)

        value = q.estimator.compute_estimate(state=state)
        confidence = q.estimator.compute_confidence(state=state, estimate=value)

        estimates[q.uid] = Estimate(value=value, confidence=confidence)

    return estimates
