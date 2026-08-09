from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import default_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .estimator.base import Estimator
    from .parser.base import Parser
    from .state import SampleState
    from .stopping import StoppingCriterion


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

    def select_wave(
        self,
        states: dict[str, SampleState],
        *,
        wave_size: int,
    ) -> tuple[Question, ...]:
        """Select up to wave_size incomplete questions in round-robin order.

        Each question appears at most once per wave. Returns an empty tuple
        when all questions are complete.
        """
        selected: list[Question] = []
        for _ in range(wave_size):
            question = self.select_next(states)
            if question is None:
                break
            # Each question appears at most once per wave.
            if any(q.uid == question.uid for q in selected):
                break
            selected.append(question)
        return tuple(selected)


@dataclass(frozen=True, kw_only=True)
class Question[T]:
    """A prompt paired with its sampling strategy."""

    prompt: Prompt[T]
    estimator: Estimator[T]
    stopping_criterion: StoppingCriterion

    @property
    def uid(self) -> str:
        return self.prompt.uid

    @property
    def query(self) -> str:
        return self.prompt.query

    @property
    def parser(self) -> Parser[T]:
        return self.prompt.parser

    def build_prompt(self) -> str:
        """Return the full prompt for this question."""
        return self.prompt.build_prompt()


@dataclass(frozen=True, kw_only=True)
class Prompt[T]:
    """A query paired with the parser needed to interpret one response."""

    uid: str
    query: str
    parser: Parser[T]

    def build_prompt(self) -> str:
        """Return the full prompt with the decline instruction appended."""
        decline_instruction = (
            f"If you have insufficient information to answer, "
            f"respond with only: {default_config.decline_keyword}"
        )
        return f"{self.query}\n\n{decline_instruction}"





