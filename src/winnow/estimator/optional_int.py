from __future__ import annotations

from typing import TYPE_CHECKING

from winnow.util import mad, median

if TYPE_CHECKING:
    from winnow.types import SampleState


class OptionalIntEstimator:
    """Consensus estimation for optional integer values.

    Handles samples that may be integers or None. If the majority of samples
    are None, the estimate is None. Otherwise, uses median of numeric values,
    rounded to the nearest integer.
    """

    def compute_estimate(self, *, state: SampleState[int | None]) -> int | None:
        """Return None if majority are None, otherwise median of numeric values."""
        samples = state.samples
        none_count = sum(1 for s in samples if s is None)
        numeric_samples = [s for s in samples if s is not None]

        if none_count > len(numeric_samples):
            return None

        if not numeric_samples:
            return None

        median_value = median([float(s) for s in numeric_samples])
        return round(median_value)

    def compute_confidence(
        self, *, state: SampleState[int | None], estimate: int | None
    ) -> float:
        """Compute confidence based on agreement.

        For None estimates: confidence is proportion of None samples, normalised.
        For numeric estimates: combines applicability agreement with value agreement.
        """
        samples = state.samples
        if len(samples) < 2:
            return 0.0

        none_count = sum(1 for s in samples if s is None)
        numeric_samples = [float(s) for s in samples if s is not None]

        if estimate is None:
            agreement = none_count / len(samples)
            # Normalise against 50% baseline (binary choice: None vs numeric)
            return max(0.0, (agreement - 0.5) / 0.5)

        if len(numeric_samples) < 2:
            return 0.0

        # Confidence that a numeric answer applies (vs None)
        applicability = len(numeric_samples) / len(samples)
        applicability_confidence = max(0.0, (applicability - 0.5) / 0.5)

        # Confidence in the numeric value itself
        median_value = float(estimate)
        if estimate == 0:
            # Zero estimate with non-zero samples indicates high variability
            value_confidence = 0.0 if any(s != 0 for s in numeric_samples) else 1.0
        elif all(s == estimate for s in numeric_samples):
            value_confidence = 1.0
        else:
            mad_value = mad(numeric_samples, median_value)
            robust_cv = 1.4826 * mad_value / abs(median_value)
            value_confidence = 1.0 / (1.0 + robust_cv)

        # Combined confidence: both factors matter
        return applicability_confidence * value_confidence
