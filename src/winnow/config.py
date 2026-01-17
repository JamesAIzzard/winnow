from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WinnowConfig:
    """Centralised configuration for winnow defaults.

    All values have sensible defaults but can be overridden when instantiating.
    """

    # Parser settings
    decline_keyword: str = "DECLINE"

    # Stopping criteria defaults
    standard_min_samples: int = 5
    standard_confidence: float = 0.90
    standard_max_queries: int = 20
    standard_max_consecutive_declines: int = 5
    standard_max_parse_failures: int = 3

# Default configuration instance
default_config = WinnowConfig()
