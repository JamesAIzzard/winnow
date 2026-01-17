from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from winnow.config import default_config

if TYPE_CHECKING:
    from winnow.estimator.base import Estimator
    from winnow.types import SampleState


@dataclass(frozen=True)
class StoppingCriterion:
    """Determines when sampling should stop for a question.

    Stops when any of:
    - Confidence threshold reached (after min_samples collected)
    - Max queries reached
    - Max consecutive declines reached
    """

    min_samples: int = default_config.standard_min_samples
    confidence_threshold: float = default_config.standard_confidence
    max_queries: int = default_config.standard_max_queries
    max_consecutive_declines: int = default_config.standard_max_consecutive_declines

    def should_stop(
        self,
        state: SampleState,
        estimator: Estimator,
    ) -> bool:
        """Return True if sampling should stop."""
        total_queries = (
            len(state.samples) + state.decline_count + state.parse_failure_count
        )

        if total_queries >= self.max_queries:
            return True

        if state.consecutive_declines >= self.max_consecutive_declines:
            return True

        if len(state.samples) < self.min_samples:
            return False

        estimate = estimator.compute_estimate(samples=state.samples)
        
        confidence = estimator.compute_confidence(
            samples=state.samples, estimate=estimate
        )
        
        return confidence >= self.confidence_threshold
