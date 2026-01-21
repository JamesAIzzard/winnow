from __future__ import annotations

from typing import TYPE_CHECKING

from winnow.util import mad, median

if TYPE_CHECKING:
    from winnow.types import SampleState


class NumericalEstimator:
    """Consensus estimation for continuous numerical values.

    Uses the median as the point estimate and robust coefficient of variation
    for confidence calculation.
    """

    def compute_estimate(self, *, state: SampleState[float]) -> float:
        """Return the median of the samples."""
        return median(state.samples)

    def compute_confidence(self, *, state: SampleState[float], estimate: float) -> float:
        """Compute confidence based on robust coefficient of variation.

        The confidence is calculated as 1 / (1 + robust_cv), where:
        robust_cv = 1.4826 * MAD / |median|

        The constant 1.4826 scales MAD to be comparable to standard deviation
        for normally distributed data.
        """
        samples = state.samples
        if len(samples) < 2:
            return 0.0

        # Handle all-zero case
        if all(s == 0.0 for s in samples):
            return 1.0

        # Zero median with non-zero samples indicates high variability
        if estimate == 0.0:
            return 0.0

        mad_value = mad(samples, estimate)
        robust_cv = 1.4826 * mad_value / abs(estimate)

        return 1.0 / (1.0 + robust_cv)
