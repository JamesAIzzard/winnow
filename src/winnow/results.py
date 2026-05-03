from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from winnow.state import ReviewReason

T = TypeVar("T")


@dataclass(frozen=True)
class Estimate(Generic[T]):
    """A value estimated from repeated LLM queries."""

    value: T
    confidence: float


@dataclass(frozen=True)
class NeedsReview:
    """Returned when estimation failed and the item needs manual review.

    This is distinct from a low-confidence Estimate. A NeedsReview means
    no estimation was performed at all.
    """

    reason: ReviewReason
