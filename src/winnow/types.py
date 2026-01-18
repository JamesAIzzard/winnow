from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


@dataclass(frozen=True)
class SampleState(Generic[T_co]):
    """Current sampling state for a single question."""

    samples: tuple[T_co, ...]
    decline_count: int
    parse_failure_count: int
    consecutive_declines: int
    current_estimate: T_co | None
    current_confidence: float

    @property
    def query_count(self) -> int:
        """Total number of queries made (successful + declined + failed)."""
        return len(self.samples) + self.decline_count + self.parse_failure_count


@dataclass(frozen=True)
class Estimate(Generic[T]):
    """A value estimated from repeated LLM queries."""

    value: T
    confidence: float
