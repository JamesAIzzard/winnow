from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from winnow.estimator.base import ConsensusEstimator
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
    estimator: ConsensusEstimator[T]
    stopping_criterion: StoppingCriterion


class QuestionBank:
    """A collection of questions to be answered."""

    def __init__(self, questions: Sequence[Question[Any]]) -> None:
        self._questions = list(questions)

    @property
    def questions(self) -> list[Question[Any]]:
        """The questions in this bank."""
        return self._questions

    def select_next(
        self,
        states: dict[str, SampleState[Any]],
    ) -> Question[Any] | None:
        """Select the next question to ask.

        Returns an incomplete question at random, or None if all complete.
        Randomisation prevents the model from anchoring on repeated queries.
        """
        incomplete = [
            q
            for q in self._questions
            if not q.stopping_criterion.should_stop(states[q.uid], q.estimator)
        ]

        if not incomplete:
            return None

        return random.choice(incomplete)
