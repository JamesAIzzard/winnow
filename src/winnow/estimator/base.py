from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class ConsensusEstimator(Protocol[T]):
    """Strategy for deriving consensus from repeated samples."""

    def compute_estimate(self, *, samples: Sequence[T]) -> T:
        """Derive the best point estimate from collected samples."""
        ...

    def compute_confidence(self, *, samples: Sequence[T], estimate: T) -> float:
        """Compute confidence in the estimate, normalised to [0, 1]."""
        ...
