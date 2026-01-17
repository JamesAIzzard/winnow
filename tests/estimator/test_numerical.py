from __future__ import annotations


from winnow.estimator.numerical import NumericalEstimator


class TestNumericalEstimatorEstimate:
    def test_returns_median_odd_count(self) -> None:
        """Verify compute_estimate returns median for odd sample count."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(samples=[1.0, 2.0, 3.0, 4.0, 5.0])

        assert result == 3.0

    def test_returns_median_even_count(self) -> None:
        """Verify compute_estimate returns average of middle two for even count."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(samples=[1.0, 2.0, 3.0, 4.0])

        assert result == 2.5

    def test_handles_unsorted_samples(self) -> None:
        """Verify median is computed correctly for unsorted input."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(samples=[5.0, 1.0, 3.0, 2.0, 4.0])

        assert result == 3.0

    def test_robust_to_outliers(self) -> None:
        """Verify median is not affected by outliers."""
        estimator = NumericalEstimator()

        result = estimator.compute_estimate(
            samples=[31.0, 31.0, 29.0, 31.0, 280.0, 30.0, 31.0, 32.0, 31.0, 30.0]
        )

        assert result == 31.0


class TestNumericalEstimatorConfidence:
    def test_high_confidence_for_consistent_samples(self) -> None:
        """Verify high confidence for tightly clustered samples."""
        estimator = NumericalEstimator()
        samples = [10.0, 10.1, 9.9, 10.0, 10.05]
        estimate = estimator.compute_estimate(samples=samples)

        confidence = estimator.compute_confidence(samples=samples, estimate=estimate)

        assert confidence > 0.9

    def test_lower_confidence_for_spread_samples(self) -> None:
        """Verify lower confidence for widely spread samples than tight clusters."""
        estimator = NumericalEstimator()
        wide_samples = [1.0, 100.0, 50.0, 25.0, 75.0]
        tight_samples = [50.0, 50.1, 49.9, 50.0, 50.05]

        wide_estimate = estimator.compute_estimate(samples=wide_samples)
        tight_estimate = estimator.compute_estimate(samples=tight_samples)

        wide_confidence = estimator.compute_confidence(
            samples=wide_samples, estimate=wide_estimate
        )
        tight_confidence = estimator.compute_confidence(
            samples=tight_samples, estimate=tight_estimate
        )

        assert wide_confidence < tight_confidence

    def test_zero_confidence_for_single_sample(self) -> None:
        """Verify zero confidence for single sample."""
        estimator = NumericalEstimator()

        confidence = estimator.compute_confidence(samples=[10.0], estimate=10.0)

        assert confidence == 0.0

    def test_full_confidence_for_identical_zeros(self) -> None:
        """Verify full confidence when all samples are zero."""
        estimator = NumericalEstimator()

        confidence = estimator.compute_confidence(
            samples=[0.0, 0.0, 0.0], estimate=0.0
        )

        assert confidence == 1.0

    def test_zero_confidence_when_median_is_zero(self) -> None:
        """Verify zero confidence when median is zero but samples are not."""
        estimator = NumericalEstimator()

        confidence = estimator.compute_confidence(
            samples=[-5.0, 0.0, 5.0], estimate=0.0
        )

        assert confidence == 0.0

    def test_confidence_in_valid_range(self) -> None:
        """Verify confidence is always in [0, 1] range."""
        estimator = NumericalEstimator()
        samples = [31.0, 31.0, 29.0, 31.0, 280.0, 30.0, 31.0, 32.0, 31.0, 30.0]
        estimate = estimator.compute_estimate(samples=samples)

        confidence = estimator.compute_confidence(samples=samples, estimate=estimate)

        assert 0.0 <= confidence <= 1.0
