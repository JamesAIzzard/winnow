from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from winnow.estimator.base import Estimator
    from winnow.parser.base import Parser
    from winnow.stopping import StoppingCriterion
    from winnow.types import SampleState

T = TypeVar("T")


@dataclass(frozen=True)
class Question(Generic[T]):
    """A query paired with its parsing and estimation strategy."""

    uid: str
    query: str
    parser: Parser[T]
    estimator: Estimator[T]
    stopping_criterion: StoppingCriterion


class QuestionBank:
    """A collection of questions to be answered."""

    def __init__(self, questions: Sequence[Question]) -> None:
        self._questions = {q.uid: q for q in questions}
        self._question_order = [q.uid for q in questions]
        self._current_question_uid: str | None = None
        self._next_index: int = 0

    @property
    def questions(self) -> dict[str, Question]:
        """The questions in this bank, keyed by uid."""
        return self._questions

    @property
    def current_question_uid(self) -> str | None:
        """The uid of the current question being asked, or None if complete."""
        return self._current_question_uid

    def num_pending_questions(self, states: dict[str, SampleState]) -> int:
        """Count questions that have not yet reached their stopping criterion."""
        return sum(
            1
            for q in self._questions.values()
            if not q.stopping_criterion.should_stop(states[q.uid])
        )

    def num_estimated_questions(self, states: dict[str, SampleState]) -> int:
        """Count questions that have reached their stopping criterion."""
        return sum(
            1
            for q in self._questions.values()
            if q.stopping_criterion.should_stop(states[q.uid])
        )

    def select_next(
        self,
        states: dict[str, SampleState],
    ) -> Question | None:
        """Select the next question to ask using round-robin.

        Cycles through questions in order, skipping any that have reached
        their stopping criterion. Returns None if all questions are complete.
        """
        num_questions = len(self._question_order)

        for _ in range(num_questions):
            uid = self._question_order[self._next_index]
            self._next_index = (self._next_index + 1) % num_questions
            question = self._questions[uid]

            if not question.stopping_criterion.should_stop(states[uid]):
                self._current_question_uid = uid
                return question

        self._current_question_uid = None
        return None
