from __future__ import annotations

from typing import TYPE_CHECKING

from winnow._util import _mad, _median

if TYPE_CHECKING:
    from collections.abc import Sequence


class NumericalEstimator:
    """Consensus estimation for continuous numerical values.

    Uses the median as the point estimate and robust coefficient of variation
    for confidence calculation.
    """

    def compute_estimate(self, *, samples: Sequence[float]) -> float:
        """Return the median of the samples."""
        return _median(samples)

    def compute_confidence(self, *, samples: Sequence[float], estimate: float) -> float:
        """Compute confidence based on robust coefficient of variation.

        The confidence is calculated as 1 / (1 + robust_cv), where:
        robust_cv = 1.4826 * MAD / |median|

        The constant 1.4826 scales MAD to be comparable to standard deviation
        for normally distributed data.
        """
        if len(samples) < 2:
            return 0.0

        # Handle all-zero case
        if all(s == 0.0 for s in samples):
            return 1.0

        # Zero median with non-zero samples indicates high variability
        if estimate == 0.0:
            return 0.0

        mad = _mad(samples, estimate)
        robust_cv = 1.4826 * mad / abs(estimate)

        return 1.0 / (1.0 + robust_cv)
