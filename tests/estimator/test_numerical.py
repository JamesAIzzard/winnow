from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING

from winnow.estimator.numerical import NumericalEstimator

if TYPE_CHECKING:
    from winnow.types import SampleState


class TestNumericalEstimatorEstimate:
    def test_returns_median_odd_count(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify compute_estimate returns median for odd sample count."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(
            state=make_state([1.0, 2.0, 3.0, 4.0, 5.0])
        )

        assert result == 3.0

    def test_returns_median_even_count(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify compute_estimate returns average of middle two for even count."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(
            state=make_state([1.0, 2.0, 3.0, 4.0])
        )

        assert result == 2.5

    def test_handles_unsorted_samples(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify median is computed correctly for unsorted input."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(
            state=make_state([5.0, 1.0, 3.0, 2.0, 4.0])
        )

        assert result == 3.0

    def test_robust_to_outliers(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify median is not affected by outliers."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(
            state=make_state(
                [31.0, 31.0, 29.0, 31.0, 280.0, 30.0, 31.0, 32.0, 31.0, 30.0]
            )
        )

        assert result == 31.0


class TestNumericalEstimatorConfidence:
    def test_high_confidence_for_consistent_samples(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify high confidence for tightly clustered samples."""
        estimator = NumericalEstimator()
        state = make_state([10.0, 10.1, 9.9, 10.0, 10.05])
        estimate = estimator.compute_estimate(state=state)

        confidence = estimator.compute_confidence(state=state, estimate=estimate)

        assert confidence > 0.9

    def test_lower_confidence_for_spread_samples(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify lower confidence for widely spread samples than tight clusters."""
        estimator = NumericalEstimator()
        wide_state = make_state([1.0, 100.0, 50.0, 25.0, 75.0])
        tight_state = make_state([50.0, 50.1, 49.9, 50.0, 50.05])

        wide_estimate = estimator.compute_estimate(state=wide_state)
        tight_estimate = estimator.compute_estimate(state=tight_state)

        wide_confidence = estimator.compute_confidence(
            state=wide_state, estimate=wide_estimate
        )
        tight_confidence = estimator.compute_confidence(
            state=tight_state, estimate=tight_estimate
        )

        assert wide_confidence < tight_confidence

    def test_zero_confidence_for_single_sample(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify zero confidence for single sample."""
        estimator = NumericalEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([10.0]), estimate=10.0
        )

        assert confidence == 0.0

    def test_full_confidence_for_identical_zeros(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify full confidence when all samples are zero."""
        estimator = NumericalEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([0.0, 0.0, 0.0]), estimate=0.0
        )

        assert confidence == 1.0

    def test_zero_confidence_when_median_is_zero(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify zero confidence when median is zero but samples are not."""
        estimator = NumericalEstimator()

        confidence = estimator.compute_confidence(
            state=make_state([-5.0, 0.0, 5.0]), estimate=0.0
        )

        assert confidence == 0.0

    def test_confidence_in_valid_range(
        self, make_state: Callable[..., SampleState[float]]
    ) -> None:
        """Verify confidence is always in [0, 1] range."""
        estimator = NumericalEstimator()
        state = make_state(
            [31.0, 31.0, 29.0, 31.0, 280.0, 30.0, 31.0, 32.0, 31.0, 30.0]
        )
        estimate = estimator.compute_estimate(state=state)

        confidence = estimator.compute_confidence(state=state, estimate=estimate)

        assert 0.0 <= confidence <= 1.0
