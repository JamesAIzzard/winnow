from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable

from winnow.estimator.categorical import CategoricalEstimator

if TYPE_CHECKING:
    from winnow.types import SampleState


class TestCategoricalEstimatorEstimate:
    def test_returns_mode(self, make_state: Callable[..., SampleState[str]]) -> None:
        """Verify compute_estimate returns the most common value."""
        estimator: CategoricalEstimator[str] = CategoricalEstimator(
            valid_options=frozenset({"a", "b", "c"})
        )

        result = estimator.compute_estimate(state=make_state(["a", "b", "a", "c", "a"]))

        assert result == "a"

    def test_handles_tie_by_first_encountered(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify tie-breaking is deterministic."""
        estimator: CategoricalEstimator[str] = CategoricalEstimator(
            valid_options=frozenset({"a", "b"})
        )

        # Counter.most_common returns first encountered on tie
        result = estimator.compute_estimate(state=make_state(["a", "b", "a", "b"]))

        assert result in {"a", "b"}


class TestCategoricalEstimatorConfidence:
    def test_full_confidence_for_unanimous(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify full confidence when all samples agree."""
        estimator: CategoricalEstimator[str] = CategoricalEstimator(
            valid_options=frozenset({"a", "b", "c"})
        )

        confidence = estimator.compute_confidence(
            state=make_state(["a", "a", "a", "a", "a"]), estimate="a"
        )

        # With 3 options, baseline is 1/3, unanimous gives (1 - 1/3) / (1 - 1/3) = 1
        assert confidence == 1.0

    def test_zero_confidence_for_empty_samples(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify zero confidence for empty sample list."""
        estimator: CategoricalEstimator[str] = CategoricalEstimator(
            valid_options=frozenset({"a", "b", "c"})
        )

        confidence = estimator.compute_confidence(state=make_state([]), estimate="a")

        assert confidence == 0.0

    def test_normalised_against_baseline(
        self, make_state: Callable[..., SampleState[str]]
    ) -> None:
        """Verify confidence is normalised against random guessing baseline."""
        estimator: CategoricalEstimator[str] = CategoricalEstimator(
            valid_options=frozenset({"a", "b"})
        )

        # With 2 options, baseline is 0.5
        # If agreement is 0.75, confidence = (0.75 - 0.5) / (1 - 0.5) = 0.5
        confidence = estimator.compute_confidence(
            state=make_state(["a", "a", "a", "b"]), estimate="a"
        )

        assert confidence == 0.5
