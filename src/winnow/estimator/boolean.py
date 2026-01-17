from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from winnow.types import SampleState


class BooleanEstimator:
    """Consensus estimation for boolean values.

    Uses majority vote as the point estimate and agreement proportion
    for confidence.
    """

    def compute_estimate(self, *, state: SampleState[bool]) -> bool:
        """Return True if more than half of samples are True."""
        return sum(state.samples) > len(state.samples) / 2

    def compute_confidence(self, *, state: SampleState[bool], estimate: bool) -> float:
        """Compute confidence based on agreement proportion.

        For boolean values, the raw agreement proportion is intuitive
        as the confidence measure.
        """
        samples = state.samples
        if len(samples) == 0:
            return 0.0

        return sum(1 for s in samples if s == estimate) / len(samples)
