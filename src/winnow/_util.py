from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _median(values: Sequence[float]) -> float:
    """Compute the median of a sequence of values.

    Raises:
        ValueError: If the sequence is empty.
    """
    if not values:
        raise ValueError("Cannot compute median of empty sequence")

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def _mad(values: Sequence[float], median: float) -> float:
    """Compute the median absolute deviation from a given median.

    Args:
        values: The sequence of values.
        median: The precomputed median.

    Returns:
        The median absolute deviation.

    Raises:
        ValueError: If the sequence is empty.
    """
    if not values:
        raise ValueError("Cannot compute MAD of empty sequence")

    deviations = [abs(v - median) for v in values]
    return _median(deviations)
