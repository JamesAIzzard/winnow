from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from winnow.types import SampleState

T = TypeVar("T")


class CategoricalEstimator(Generic[T]):
    """Consensus estimation for categorical values.

    Uses the mode as the point estimate and normalised agreement for confidence.
    """

    def __init__(self, *, valid_options: frozenset[T]) -> None:
        """Initialise with the set of valid options.

        Args:
            valid_options: The set of valid categorical values.
        """
        self._valid_options = valid_options

    def compute_estimate(self, *, state: SampleState[T]) -> T:
        """Return the mode (most common value) of the samples."""
        counts: Counter[T] = Counter(state.samples)
        return counts.most_common(1)[0][0]

    def compute_confidence(self, *, state: SampleState[T], estimate: T) -> float:
        """Compute confidence based on normalised agreement.

        The confidence is normalised against random guessing:
        confidence = (agreement - 1/n) / (1 - 1/n)

        Where agreement is the proportion matching the mode and n is the
        number of valid options.
        """
        samples = state.samples
        if len(samples) == 0:
            return 0.0

        agreement = sum(1 for s in samples if s == estimate) / len(samples)
        baseline = 1.0 / len(self._valid_options)

        if baseline >= 1.0:
            return 1.0

        return (agreement - baseline) / (1.0 - baseline)
