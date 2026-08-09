from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .question import QuestionUID
from .state import SampleState


@dataclass(frozen=True, kw_only=True)
class CollectionProgress:
    sample_states: SampleStates
    new_interactions: tuple[QuestionInteraction, ...]


@dataclass(frozen=True, kw_only=True)
class QuestionInteraction:
    question_uid: QuestionUID
    prompt: str
    raw_response: str


type SampleStates = Mapping[QuestionUID, SampleState[Any]]
type ProgressCallback = Callable[[CollectionProgress], None]
