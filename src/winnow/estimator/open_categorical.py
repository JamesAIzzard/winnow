from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, TypeVar

from .base import Estimator

if TYPE_CHECKING:
    from winnow.state import SampleState

T = TypeVar("T")


class OpenCategoricalEstimator(Estimator[T]):
    """Consensus estimation for categorical values from an unbounded option set.

    Uses the mode as the point estimate and raw agreement (the fraction of
    samples matching the mode) as the confidence metric.
    """

    def compute_estimate(self, *, state: SampleState[T]) -> T:
        """Return the mode (most common value) of the samples."""
        counts: Counter[T] = Counter(state.samples)
        return counts.most_common(1)[0][0]

    def compute_confidence(self, *, state: SampleState[T], estimate: T) -> float:
        """Compute confidence as the raw agreement with the mode.

        Unlike CategoricalEstimator, no normalisation against a random baseline
        is applied, because the number of valid options is not known. This is
        equivalent to the limit of normalised agreement as k approaches infinity.
        """
        samples = state.samples
        if len(samples) == 0:
            return 0.0

        return sum(1 for s in samples if s == estimate) / len(samples)
