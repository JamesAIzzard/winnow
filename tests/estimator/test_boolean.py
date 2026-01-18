from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import pytest

from winnow.estimator.boolean import BooleanEstimator

if TYPE_CHECKING:
    from winnow.types import SampleState


class TestBooleanEstimatorEstimate:
    def test_returns_true_for_majority_true(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify compute_estimate returns True when majority are True."""
        estimator = BooleanEstimator()

        result = estimator.compute_estimate(
            state=make_state([True, True, True, False, False])
        )

        assert result is True

    def test_returns_false_for_majority_false(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify compute_estimate returns False when majority are False."""
        estimator = BooleanEstimator()

        result = estimator.compute_estimate(
            state=make_state([True, False, False, False, False])
        )

        assert result is False

    def test_returns_first_seen_for_exact_tie(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify compute_estimate returns the first-seen value for exact tie.

        Counter.most_common returns items in insertion order when tied.
        """
        estimator = BooleanEstimator()

        result = estimator.compute_estimate(
            state=make_state([True, True, False, False])
        )

        assert result is True


class TestBooleanEstimatorConfidence:
    def test_full_confidence_for_unanimous_true(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify full confidence when all samples are True."""
        estimator = BooleanEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([True, True, True, True, True]), estimate=True
        )

        assert confidence == 1.0

    def test_full_confidence_for_unanimous_false(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify full confidence when all samples are False."""
        estimator = BooleanEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([False, False, False, False, False]), estimate=False
        )

        assert confidence == 1.0

    def test_zero_confidence_for_empty_samples(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify zero confidence for empty sample list."""
        estimator = BooleanEstimator()

        confidence = estimator.compute_confidence(state=make_state([]), estimate=True)

        assert confidence == 0.0

    def test_zero_confidence_for_even_split(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify zero confidence for 50/50 split (random chance baseline)."""
        estimator = BooleanEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([True, True, False, False]), estimate=True
        )

        assert confidence == 0.0

    def test_partial_confidence_for_mixed_samples(
        self, make_state: Callable[..., SampleState[bool]]
    ) -> None:
        """Verify partial confidence for mixed samples.

        With 3/5 agreement (60%), normalised confidence is:
        (0.6 - 0.5) / (1.0 - 0.5) = 0.1 / 0.5 = 0.2
        """
        estimator = BooleanEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([True, True, True, False, False]), estimate=True
        )

        assert confidence == pytest.approx(0.2)
