from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Final, Generic, TypeVar

if TYPE_CHECKING:
    from .stopping import StoppingCriterion

T = TypeVar("T")


class _NoEstimateType:
    """Sentinel type for `NoEstimate`; compare with `is NoEstimate`.

    Distinct from `None`, which can be a legitimate estimate value
    (e.g. `OptionalBoundedIntParser` where `None` means 'not applicable').
    """


NoEstimate: Final = _NoEstimateType()
"""Sentinel value indicating no estimate has been computed yet."""


class SampleStatus(Enum):
    """Explicit lifecycle status of a sampling question."""

    PENDING = "pending"
    COLLECTING = "collecting"
    CONVERGED = "converged"
    NEEDS_REVIEW = "needs_review"


class ReviewReason(Enum):
    """Reasons why estimation failed and manual review is required."""

    MAX_CONSECUTIVE_DECLINES = "max_consecutive_declines"
    MAX_PARSE_FAILURES = "max_parse_failures"
    MAX_QUERIES = "max_queries"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"


@dataclass(frozen=True)
class SampleState(Generic[T]):
    """Current sampling state for a single question."""

    samples: tuple[T, ...]
    decline_count: int
    parse_failure_count: int
    consecutive_declines: int
    current_estimate: T | _NoEstimateType
    current_confidence: float
    status: SampleStatus
    failure_reason: ReviewReason | None
    effective_sample_count: int | None = None

    @property
    def query_count(self) -> int:
        """Total number of queries made (successful + declined + failed)."""
        return len(self.samples) + self.decline_count + self.parse_failure_count

    @property
    def stopping_sample_count(self) -> int:
        """Number of samples that count towards the minimum-sample floor."""
        return (
            self.effective_sample_count
            if self.effective_sample_count is not None
            else len(self.samples)
        )

    def record_sample(
        self,
        value: T,
        *,
        estimate: T,
        confidence: float,
        effective_sample_count: int | None,
    ) -> SampleState[T]:
        """Return the next state after a successful sample."""
        return replace(
            self,
            samples=self.samples + (value,),
            consecutive_declines=0,
            current_estimate=estimate,
            current_confidence=confidence,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
            effective_sample_count=effective_sample_count,
        )

    def record_decline(self) -> SampleState[T]:
        """Return the next state after a model decline."""
        return replace(
            self,
            decline_count=self.decline_count + 1,
            consecutive_declines=self.consecutive_declines + 1,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

    def record_parse_failure(self) -> SampleState[T]:
        """Return the next state after a parse failure."""
        return replace(
            self,
            parse_failure_count=self.parse_failure_count + 1,
            consecutive_declines=0,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

    def resolve_status(self, criterion: StoppingCriterion) -> SampleState[T]:
        """Return the state with terminal status applied when stopping."""
        if not criterion.should_stop(self):
            return self

        converged = (
            self.stopping_sample_count >= criterion.min_samples
            and self.current_confidence >= criterion.confidence_threshold
        )

        if converged:
            return replace(
                self,
                status=SampleStatus.CONVERGED,
                failure_reason=None,
            )

        return replace(
            self,
            status=SampleStatus.NEEDS_REVIEW,
            failure_reason=self._determine_failure_reason(criterion),
        )

    def _determine_failure_reason(
        self,
        criterion: StoppingCriterion,
    ) -> ReviewReason:
        """Determine why estimation failed for a question."""
        if self.consecutive_declines >= criterion.max_consecutive_declines:
            return ReviewReason.MAX_CONSECUTIVE_DECLINES
        if self.parse_failure_count >= criterion.max_parse_failures:
            return ReviewReason.MAX_PARSE_FAILURES
        if len(self.samples) == 0:
            return ReviewReason.MAX_QUERIES
        return ReviewReason.INSUFFICIENT_CONFIDENCE
