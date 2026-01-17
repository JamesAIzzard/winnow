from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from winnow.types import SampleState

T = TypeVar("T")


class Estimator(Protocol[T]):
    """Strategy for deriving consensus from repeated samples."""

    def compute_estimate(self, *, state: SampleState[T]) -> T:
        """Derive the best point estimate from collected samples."""
        ...

    def compute_confidence(self, *, state: SampleState[T], estimate: T) -> float:
        """Compute confidence in the estimate, normalised to [0, 1]."""
        ...
