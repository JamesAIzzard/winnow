from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class BooleanEstimator:
    """Consensus estimation for boolean values.

    Uses majority vote as the point estimate and agreement proportion
    for confidence.
    """

    def compute_estimate(self, *, samples: Sequence[bool]) -> bool:
        """Return True if more than half of samples are True."""
        return sum(samples) > len(samples) / 2

    def compute_confidence(self, *, samples: Sequence[bool], estimate: bool) -> float:
        """Compute confidence based on agreement proportion.

        For boolean values, the raw agreement proportion is intuitive
        as the confidence measure.
        """
        if len(samples) == 0:
            return 0.0

        return sum(1 for s in samples if s == estimate) / len(samples)
