from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any as AnyType

from winnow.stopping.base import StoppingCriterion

if TYPE_CHECKING:
    from winnow.estimator.base import ConsensusEstimator
    from winnow.types import SampleState


@dataclass(frozen=True)
class All(StoppingCriterion):
    """Stop only when all child criteria agree."""

    criteria: tuple[StoppingCriterion, ...]

    def __init__(self, *criteria: StoppingCriterion) -> None:
        object.__setattr__(self, "criteria", criteria)

    def should_stop(
        self,
        state: SampleState[AnyType],
        estimator: ConsensusEstimator[AnyType],
    ) -> bool:
        """All criteria must agree to stop."""
        return all(c.should_stop(state, estimator) for c in self.criteria)


@dataclass(frozen=True)
class Any(StoppingCriterion):
    """Stop when any child criterion is satisfied."""

    criteria: tuple[StoppingCriterion, ...]

    def __init__(self, *criteria: StoppingCriterion) -> None:
        object.__setattr__(self, "criteria", criteria)

    def should_stop(
        self,
        state: SampleState[AnyType],
        estimator: ConsensusEstimator[AnyType],
    ) -> bool:
        """Any criterion can trigger a stop."""
        return any(c.should_stop(state, estimator) for c in self.criteria)
