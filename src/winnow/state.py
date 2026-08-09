from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .stopping import StoppingCriterion


@dataclass(frozen=True, kw_only=True)
class SampleState[T]:
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

    @property
    def is_terminal(self) -> bool:
        """Whether collection has finished for this question."""
        return self.status in {SampleStatus.CONVERGED, SampleStatus.NEEDS_REVIEW}

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
        decision = criterion.evaluate(self)
        if decision is None:
            return self

        return replace(
            self,
            status=decision.status,
            failure_reason=decision.failure_reason,
        )


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


class _NoEstimateType(Enum):
    """Sentinel type for `NoEstimate`; compare with `is NoEstimate`.

    Distinct from `None`, which can be a legitimate estimate value
    (e.g. `OptionalBoundedIntParser` where `None` means 'not applicable').

    Defined as a single-member enum so that type checkers can narrow
    `T | _NoEstimateType` on an `is NoEstimate` comparison. A plain class
    cannot be narrowed, because nothing tells the checker it has only
    one instance.
    """

    TOKEN = auto()


NoEstimate: Final = _NoEstimateType.TOKEN
"""Sentinel value indicating no estimate has been computed yet."""
