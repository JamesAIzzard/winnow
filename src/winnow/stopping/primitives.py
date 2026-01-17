from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from winnow.stopping.base import StoppingCriterion

if TYPE_CHECKING:
    from winnow.estimator.base import ConsensusEstimator
    from winnow.types import SampleState


@dataclass(frozen=True)
class MinSamples(StoppingCriterion):
    """Don't stop until we have at least n successful samples."""

    n: int

    def should_stop(
        self,
        state: SampleState[Any],
        estimator: ConsensusEstimator[Any],
    ) -> bool:
        """Verify we have collected at least n samples."""
        return len(state.samples) >= self.n


@dataclass(frozen=True)
class MaxQueries(StoppingCriterion):
    """Stop after n total queries (success + decline + failure)."""

    n: int

    def should_stop(
        self,
        state: SampleState[Any],
        estimator: ConsensusEstimator[Any],
    ) -> bool:
        """Verify we have not exceeded the query budget."""
        return state.query_count >= self.n


@dataclass(frozen=True)
class ConfidenceReached(StoppingCriterion):
    """Stop when confidence exceeds threshold."""

    threshold: float

    def should_stop(
        self,
        state: SampleState[Any],
        estimator: ConsensusEstimator[Any],
    ) -> bool:
        """Verify confidence has reached the threshold."""
        if len(state.samples) < 2:
            return False
        estimate = estimator.compute_estimate(samples=state.samples)
        confidence = estimator.compute_confidence(samples=state.samples, estimate=estimate)
        return confidence >= self.threshold


@dataclass(frozen=True)
class ConsecutiveDeclines(StoppingCriterion):
    """Stop if model declines n times in a row."""

    n: int

    def should_stop(
        self,
        state: SampleState[Any],
        estimator: ConsensusEstimator[Any],
    ) -> bool:
        """Verify we have not hit too many consecutive declines."""
        return state.consecutive_declines >= self.n


@dataclass(frozen=True)
class UnanimousAgreement(StoppingCriterion):
    """Stop early if all samples agree (useful for categorical/boolean)."""

    min_samples: int = 3

    def should_stop(
        self,
        state: SampleState[Any],
        estimator: ConsensusEstimator[Any],
    ) -> bool:
        """Verify all samples agree after collecting minimum required."""
        if len(state.samples) < self.min_samples:
            return False
        return len(set(state.samples)) == 1
