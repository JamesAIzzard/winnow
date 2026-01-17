from __future__ import annotations

from winnow.stopping.base import StoppingCriterion
from winnow.stopping.primitives import (
    ConfidenceReached,
    ConsecutiveDeclines,
    MaxQueries,
    MinSamples,
    UnanimousAgreement,
)


def standard_stopping(
    *,
    min_samples: int = 5,
    confidence: float = 0.90,
    max_queries: int = 20,
    max_consecutive_declines: int = 5,
) -> StoppingCriterion:
    """Standard stopping criterion for numerical fields.

    Stops when:
    - (min_samples reached AND confidence threshold met), OR
    - max_queries reached, OR
    - max_consecutive_declines reached
    """
    return (
        (MinSamples(min_samples) & ConfidenceReached(confidence))
        | MaxQueries(max_queries)
        | ConsecutiveDeclines(max_consecutive_declines)
    )


def categorical_stopping(
    *,
    unanimous_after: int = 3,
    min_samples: int = 5,
    confidence: float = 0.85,
    max_queries: int = 15,
) -> StoppingCriterion:
    """Stopping criterion for categorical fields with early unanimous exit.

    Stops when:
    - unanimous_after samples all agree, OR
    - (min_samples reached AND confidence threshold met), OR
    - max_queries reached
    """
    return (
        UnanimousAgreement(unanimous_after)
        | (MinSamples(min_samples) & ConfidenceReached(confidence))
        | MaxQueries(max_queries)
    )


def relaxed_stopping(
    *,
    min_samples: int = 5,
    confidence: float = 0.75,
    max_queries: int = 15,
    max_consecutive_declines: int = 3,
) -> StoppingCriterion:
    """Relaxed criterion for inherently variable data (e.g., trace nutrients).

    Stops when:
    - (min_samples reached AND lower confidence threshold met), OR
    - max_queries reached, OR
    - max_consecutive_declines reached
    """
    return (
        (MinSamples(min_samples) & ConfidenceReached(confidence))
        | MaxQueries(max_queries)
        | ConsecutiveDeclines(max_consecutive_declines)
    )
