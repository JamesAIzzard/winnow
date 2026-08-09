from __future__ import annotations

from dataclasses import dataclass

from .state import ReviewReason


@dataclass(frozen=True, kw_only=True)
class Estimate[T]:
    """A value estimated from repeated LLM queries."""

    value: T
    confidence: float


@dataclass(frozen=True, kw_only=True)
class NeedsReview:
    """Returned when estimation failed and the item needs manual review."""

    reason: ReviewReason
