from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from winnow.estimator.base import ConsensusEstimator
    from winnow.types import SampleState


class StoppingCriterion(ABC):
    """Determines when sampling should stop for a question."""

    @abstractmethod
    def should_stop(
        self,
        state: SampleState[Any],
        estimator: ConsensusEstimator[Any],
    ) -> bool:
        """Return True if sampling should stop."""
        ...

    def __and__(self, other: StoppingCriterion) -> StoppingCriterion:
        """Both criteria must agree to stop."""
        from winnow.stopping.combinators import All

        return All(self, other)

    def __or__(self, other: StoppingCriterion) -> StoppingCriterion:
        """Either criterion can trigger a stop."""
        from winnow.stopping.combinators import Any as AnyCriterion

        return AnyCriterion(self, other)
