from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import default_config
from .state import ReviewReason, SampleStatus

if TYPE_CHECKING:
    from .state import SampleState

_TERMINAL_STATUSES = frozenset({SampleStatus.CONVERGED, SampleStatus.NEEDS_REVIEW})


@dataclass(frozen=True, kw_only=True)
class StoppingCriterion:
    """Determines when sampling should stop for a question.

    Stops when any of:
    - State already has a terminal status (CONVERGED or NEEDS_REVIEW)
    - Confidence threshold reached (after min_samples collected)
    - Max consecutive declines reached
    - Max parse failures reached
    - Max queries reached
    """

    min_samples: int = default_config.standard_min_samples
    confidence_threshold: float = default_config.standard_confidence
    max_queries: int = default_config.standard_max_queries
    max_consecutive_declines: int = default_config.standard_max_consecutive_declines
    max_parse_failures: int = default_config.standard_max_parse_failures

    def evaluate(self, state: SampleState) -> StoppingDecision | None:
        """Return the terminal outcome, or None when sampling should continue."""
        if state.status in _TERMINAL_STATUSES:
            return StoppingDecision(
                status=state.status,
                failure_reason=state.failure_reason,
            )

        if (
            state.stopping_sample_count >= self.min_samples
            and state.current_confidence >= self.confidence_threshold
        ):
            return StoppingDecision(
                status=SampleStatus.CONVERGED,
                failure_reason=None,
            )

        if state.consecutive_declines >= self.max_consecutive_declines:
            return self._needs_review(ReviewReason.MAX_CONSECUTIVE_DECLINES)

        if state.parse_failure_count >= self.max_parse_failures:
            return self._needs_review(ReviewReason.MAX_PARSE_FAILURES)

        if state.query_count < self.max_queries:
            return None

        reason = (
            ReviewReason.MAX_QUERIES
            if state.stopping_sample_count < self.min_samples
            else ReviewReason.INSUFFICIENT_CONFIDENCE
        )
        return self._needs_review(reason)

    @staticmethod
    def _needs_review(reason: ReviewReason) -> StoppingDecision:
        return StoppingDecision(
            status=SampleStatus.NEEDS_REVIEW,
            failure_reason=reason,
        )


@dataclass(frozen=True, kw_only=True)
class StoppingDecision:
    """Terminal outcome selected by a stopping criterion."""

    status: SampleStatus
    failure_reason: ReviewReason | None
