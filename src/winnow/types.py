from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Archetype(Enum):
    """Classification of sampling convergence behaviour."""

    CONFIDENT = auto()
    ACCEPTABLE = auto()
    UNCERTAIN = auto()
    INSUFFICIENT_DATA = auto()


@dataclass(frozen=True)
class SampleState(Generic[T_co]):
    """Current sampling state for a single question."""

    samples: tuple[T_co, ...]
    decline_count: int
    parse_failure_count: int
    consecutive_declines: int

    @property
    def query_count(self) -> int:
        """Total number of queries made (successful + declined + failed)."""
        return len(self.samples) + self.decline_count + self.parse_failure_count


@dataclass(frozen=True)
class Estimate(Generic[T_co]):
    """A value estimated from repeated LLM queries."""

    value: T_co | None
    confidence: float
    archetype: Archetype
    sample_count: int
    decline_count: int
    samples: tuple[T_co, ...]
