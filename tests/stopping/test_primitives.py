from __future__ import annotations

from winnow.estimator.numerical import NumericalEstimator
from winnow.stopping.primitives import (
    ConfidenceReached,
    ConsecutiveDeclines,
    MaxQueries,
    MinSamples,
    UnanimousAgreement,
)
from winnow.types import SampleState


class TestMinSamples:
    def test_does_not_stop_below_threshold(self) -> None:
        """Verify criterion does not stop when below minimum samples."""
        criterion = MinSamples(n=5)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False

    def test_stops_at_threshold(self) -> None:
        """Verify criterion stops when reaching minimum samples."""
        criterion = MinSamples(n=5)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0, 4.0, 5.0),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True

    def test_stops_above_threshold(self) -> None:
        """Verify criterion stops when exceeding minimum samples."""
        criterion = MinSamples(n=3)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0, 4.0, 5.0),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True


class TestMaxQueries:
    def test_does_not_stop_below_limit(self) -> None:
        """Verify criterion does not stop when below max queries."""
        criterion = MaxQueries(n=10)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0),
            decline_count=2,
            parse_failure_count=1,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False

    def test_stops_at_limit(self) -> None:
        """Verify criterion stops when reaching max queries."""
        criterion = MaxQueries(n=10)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0, 4.0, 5.0),
            decline_count=3,
            parse_failure_count=2,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True

    def test_counts_all_query_types(self) -> None:
        """Verify all query types contribute to count."""
        criterion = MaxQueries(n=5)
        state: SampleState[float] = SampleState(
            samples=(1.0,),
            decline_count=2,
            parse_failure_count=2,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True


class TestConfidenceReached:
    def test_does_not_stop_below_threshold(self) -> None:
        """Verify criterion does not stop when confidence is low."""
        criterion = ConfidenceReached(threshold=0.9)
        # Wide spread samples -> low confidence
        state: SampleState[float] = SampleState(
            samples=(1.0, 100.0, 50.0, 25.0, 75.0),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False

    def test_stops_at_threshold(self) -> None:
        """Verify criterion stops when confidence reaches threshold."""
        criterion = ConfidenceReached(threshold=0.9)
        # Tight cluster -> high confidence
        state: SampleState[float] = SampleState(
            samples=(10.0, 10.0, 10.0, 10.0, 10.0),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True

    def test_requires_at_least_two_samples(self) -> None:
        """Verify criterion requires minimum two samples."""
        criterion = ConfidenceReached(threshold=0.5)
        state: SampleState[float] = SampleState(
            samples=(10.0,),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False


class TestConsecutiveDeclines:
    def test_does_not_stop_below_threshold(self) -> None:
        """Verify criterion does not stop when below consecutive declines."""
        criterion = ConsecutiveDeclines(n=5)
        state: SampleState[float] = SampleState(
            samples=(),
            decline_count=10,
            parse_failure_count=0,
            consecutive_declines=3,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False

    def test_stops_at_threshold(self) -> None:
        """Verify criterion stops when reaching consecutive declines."""
        criterion = ConsecutiveDeclines(n=5)
        state: SampleState[float] = SampleState(
            samples=(),
            decline_count=10,
            parse_failure_count=0,
            consecutive_declines=5,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True


class TestUnanimousAgreement:
    def test_does_not_stop_below_min_samples(self) -> None:
        """Verify criterion requires minimum samples even if unanimous."""
        criterion = UnanimousAgreement(min_samples=5)
        state: SampleState[str] = SampleState(
            samples=("a", "a", "a"),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False

    def test_stops_for_unanimous_after_min_samples(self) -> None:
        """Verify criterion stops when unanimous after minimum samples."""
        criterion = UnanimousAgreement(min_samples=3)
        state: SampleState[str] = SampleState(
            samples=("a", "a", "a"),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True

    def test_does_not_stop_if_not_unanimous(self) -> None:
        """Verify criterion does not stop when samples differ."""
        criterion = UnanimousAgreement(min_samples=3)
        state: SampleState[str] = SampleState(
            samples=("a", "a", "b", "a", "a"),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False
