from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WinnowConfig:
    """Centralised configuration for winnow defaults.

    All values have sensible defaults but can be overridden when instantiating.
    """

    # Parser settings
    decline_keyword: str = "DECLINE"

    # Stopping criteria defaults (standard)
    standard_min_samples: int = 5
    standard_confidence: float = 0.90
    standard_max_queries: int = 20
    standard_max_consecutive_declines: int = 5

    # Stopping criteria defaults (categorical)
    categorical_unanimous_after: int = 3
    categorical_min_samples: int = 5
    categorical_confidence: float = 0.85
    categorical_max_queries: int = 15

    # Stopping criteria defaults (relaxed)
    relaxed_min_samples: int = 5
    relaxed_confidence: float = 0.75
    relaxed_max_queries: int = 15
    relaxed_max_consecutive_declines: int = 3

    # Archetype classification thresholds
    confident_sample_threshold: int = 10
    acceptable_confidence_threshold: float = 0.7

    # UnanimousAgreement default
    unanimous_min_samples: int = 3

    @property
    def decline_keywords(self) -> frozenset[str]:
        """Return decline keyword as a frozenset for Parser compatibility."""
        return frozenset({self.decline_keyword})


# Default configuration instance
default_config = WinnowConfig()
