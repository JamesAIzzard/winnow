from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from winnow.estimator.optional_int import OptionalIntEstimator

if TYPE_CHECKING:
    from winnow.types import SampleState


class TestOptionalIntEstimatorEstimate:
    def test_returns_none_when_majority_none(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify compute_estimate returns None when majority are None."""
        estimator = OptionalIntEstimator()

        result = estimator.compute_estimate(
            state=make_state([None, None, None, 55, 60])
        )

        assert result is None

    def test_returns_median_when_majority_numeric(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify compute_estimate returns median when majority are numeric."""
        estimator = OptionalIntEstimator()

        result = estimator.compute_estimate(
            state=make_state([55, 60, 58, None, None])
        )

        assert result == 58

    def test_returns_numeric_when_equal_split(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify compute_estimate returns numeric when equal None and numeric."""
        estimator = OptionalIntEstimator()

        result = estimator.compute_estimate(
            state=make_state([55, 60, None, None])
        )

        # Equal split favours numeric (none_count not > numeric_count)
        assert result == 58  # median of [55, 60] rounded

    def test_returns_none_when_all_none(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify compute_estimate returns None when all samples are None."""
        estimator = OptionalIntEstimator()

        result = estimator.compute_estimate(
            state=make_state([None, None, None])
        )

        assert result is None

    def test_returns_median_when_all_numeric(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify compute_estimate returns median when all samples are numeric."""
        estimator = OptionalIntEstimator()

        result = estimator.compute_estimate(
            state=make_state([55, 60, 58, 62, 57])
        )

        assert result == 58

    def test_rounds_median_to_int(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify compute_estimate rounds median to nearest integer."""
        estimator = OptionalIntEstimator()

        result = estimator.compute_estimate(
            state=make_state([55, 56, 57, 58])
        )

        # Median of [55, 56, 57, 58] is 56.5, rounds to 56
        assert result == 56


class TestOptionalIntEstimatorConfidence:
    def test_full_confidence_all_numeric_identical(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify full confidence when all samples are identical numerics."""
        estimator = OptionalIntEstimator()
        state = make_state([55, 55, 55, 55, 55])

        confidence = estimator.compute_confidence(state=state, estimate=55)

        assert confidence == 1.0

    def test_low_confidence_narrow_numeric_majority(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify low confidence when numeric majority is slim despite value agreement."""
        estimator = OptionalIntEstimator()
        state = make_state([55, 55, 55, None, None])

        confidence = estimator.compute_confidence(state=state, estimate=55)

        # Applicability = 3/5 = 0.6 -> normalised = 0.2
        # Value confidence = 1.0 (all identical)
        # Combined = 0.2
        assert confidence == pytest.approx(0.2)

    def test_full_confidence_for_none_when_all_none(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify full confidence for None estimate when all samples are None."""
        estimator = OptionalIntEstimator()
        state = make_state([None, None, None, None, None])

        confidence = estimator.compute_confidence(state=state, estimate=None)

        assert confidence == 1.0

    def test_low_confidence_for_none_with_narrow_majority(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify low confidence for None when majority is slim."""
        estimator = OptionalIntEstimator()
        state = make_state([None, None, None, 55, 60])

        confidence = estimator.compute_confidence(state=state, estimate=None)

        # Agreement = 3/5 = 0.6 -> normalised = 0.2
        assert confidence == pytest.approx(0.2)

    def test_zero_confidence_for_single_sample(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify zero confidence for single sample."""
        estimator = OptionalIntEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([55]), estimate=55
        )

        assert confidence == 0.0

    def test_zero_confidence_single_none(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify zero confidence for single None sample."""
        estimator = OptionalIntEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([None]), estimate=None
        )

        assert confidence == 0.0

    def test_high_confidence_all_numeric_tight_cluster(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify high confidence for tightly clustered numeric samples."""
        estimator = OptionalIntEstimator()
        state = make_state([55, 56, 55, 54, 55])
        estimate = estimator.compute_estimate(state=state)

        confidence = estimator.compute_confidence(state=state, estimate=estimate)

        assert confidence > 0.9

    def test_lower_confidence_for_spread_numerics(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify lower confidence for widely spread numeric samples."""
        estimator = OptionalIntEstimator()
        wide_state = make_state([10, 50, 30, 70, 90])
        tight_state = make_state([50, 51, 49, 50, 50])

        wide_estimate = estimator.compute_estimate(state=wide_state)
        tight_estimate = estimator.compute_estimate(state=tight_state)

        wide_confidence = estimator.compute_confidence(
            state=wide_state, estimate=wide_estimate
        )
        tight_confidence = estimator.compute_confidence(
            state=tight_state, estimate=tight_estimate
        )

        assert wide_confidence < tight_confidence

    def test_zero_estimate_all_zeros(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify full confidence when all numeric samples are zero."""
        estimator = OptionalIntEstimator()
        state = make_state([0, 0, 0, 0, 0])

        confidence = estimator.compute_confidence(state=state, estimate=0)

        assert confidence == 1.0

    def test_zero_estimate_with_nonzero_samples(
        self, make_state: Callable[..., SampleState[int | None]]
    ) -> None:
        """Verify zero value confidence when estimate is zero but samples vary."""
        estimator = OptionalIntEstimator()
        state = make_state([-5, 0, 5, 0, 0])

        confidence = estimator.compute_confidence(state=state, estimate=0)

        # Applicability = 1.0, but value confidence = 0.0
        assert confidence == 0.0
