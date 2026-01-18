from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from winnow.config import default_config

if TYPE_CHECKING:
    from winnow.types import SampleState


@dataclass(frozen=True)
class StoppingCriterion:
    """Determines when sampling should stop for a question.

    Stops when any of:
    - Confidence threshold reached (after min_samples collected)
    - Max queries reached
    - Max consecutive declines reached
    - Max parse failures reached
    """

    min_samples: int = default_config.standard_min_samples
    confidence_threshold: float = default_config.standard_confidence
    max_queries: int = default_config.standard_max_queries
    max_consecutive_declines: int = default_config.standard_max_consecutive_declines
    max_parse_failures: int = default_config.standard_max_parse_failures

    def should_stop(self, state: SampleState) -> bool:
        """Return True if sampling should stop."""
        if state.query_count >= self.max_queries:
            return True

        if state.consecutive_declines >= self.max_consecutive_declines:
            return True

        if state.parse_failure_count >= self.max_parse_failures:
            return True

        if len(state.samples) < self.min_samples:
            return False

        return state.current_confidence >= self.confidence_threshold
