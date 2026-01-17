from __future__ import annotations

from winnow.estimator.categorical import CategoricalEstimator

_BOOLEAN_OPTIONS: frozenset[bool] = frozenset({True, False})


class BooleanEstimator(CategoricalEstimator[bool]):
    """Consensus estimation for boolean values.

    A specialisation of CategoricalEstimator for the two-option boolean case.
    Uses majority vote as the point estimate and normalised agreement for
    confidence.
    """

    def __init__(self) -> None:
        """Initialise with True/False as the valid options."""
        super().__init__(valid_options=_BOOLEAN_OPTIONS)
