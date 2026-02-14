from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class _NoEstimateType:
    """Sentinel type indicating that no estimate has been computed yet.

    This is distinct from None, which may be a legitimate estimated value
    (e.g. for OptionalBoundedIntParser where None means 'not applicable').

    A single instance is exported as ``NoEstimate``. Comparison should use
    identity (``is``)::

        if state.current_estimate is NoEstimate:
            ...
    """

    def __repr__(self) -> str:
        return "NoEstimate"

    def __bool__(self) -> bool:
        return False

    def __hash__(self) -> int:
        return hash("NoEstimate")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NoEstimateType)


NoEstimate = _NoEstimateType()
"""Sentinel value indicating no estimate has been computed yet."""


@dataclass(frozen=True)
class SampleState(Generic[T_co]):
    """Current sampling state for a single question."""

    samples: tuple[T_co, ...]
    decline_count: int
    parse_failure_count: int
    consecutive_declines: int
    current_estimate: T_co | _NoEstimateType
    current_confidence: float
    converged: bool
    failure_reason: ReviewReason | None

    @property
    def query_count(self) -> int:
        """Total number of queries made (successful + declined + failed)."""
        return len(self.samples) + self.decline_count + self.parse_failure_count


@dataclass(frozen=True)
class Estimate(Generic[T]):
    """A value estimated from repeated LLM queries."""

    value: T
    confidence: float


class ReviewReason(Enum):
    """Reasons why estimation failed and manual review is required."""

    MAX_CONSECUTIVE_DECLINES = "max_consecutive_declines"
    MAX_PARSE_FAILURES = "max_parse_failures"
    MAX_QUERIES = "max_queries"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"


@dataclass(frozen=True)
class NeedsReview:
    """Returned when estimation failed and the item needs manual review.

    This is distinct from a low-confidence Estimate. A NeedsReview means
    no estimation was performed at all.
    """

    reason: ReviewReason
