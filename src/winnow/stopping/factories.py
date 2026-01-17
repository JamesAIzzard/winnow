from __future__ import annotations

from winnow.config import default_config
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
    min_samples: int = default_config.standard_min_samples,
    confidence: float = default_config.standard_confidence,
    max_queries: int = default_config.standard_max_queries,
    max_consecutive_declines: int = default_config.standard_max_consecutive_declines,
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
    unanimous_after: int = default_config.categorical_unanimous_after,
    min_samples: int = default_config.categorical_min_samples,
    confidence: float = default_config.categorical_confidence,
    max_queries: int = default_config.categorical_max_queries,
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
    min_samples: int = default_config.relaxed_min_samples,
    confidence: float = default_config.relaxed_confidence,
    max_queries: int = default_config.relaxed_max_queries,
    max_consecutive_declines: int = default_config.relaxed_max_consecutive_declines,
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
