from __future__ import annotations

from winnow.estimator.numerical import NumericalEstimator
from winnow.stopping.combinators import All, Any
from winnow.stopping.primitives import MaxQueries, MinSamples
from winnow.types import SampleState


class TestAllCombinator:
    def test_stops_when_all_criteria_met(self) -> None:
        """Verify All stops when all child criteria agree."""
        criterion = All(MinSamples(3), MaxQueries(10))
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0, 4.0, 5.0),
            decline_count=3,
            parse_failure_count=2,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True

    def test_does_not_stop_when_one_criterion_not_met(self) -> None:
        """Verify All does not stop when any child criterion disagrees."""
        criterion = All(MinSamples(10), MaxQueries(5))
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=1,
            parse_failure_count=1,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        # MaxQueries met (5), but MinSamples not met (3 < 10)
        assert result is False

    def test_and_operator(self) -> None:
        """Verify & operator creates All combinator."""
        criterion = MinSamples(3) & MaxQueries(10)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0, 4.0, 5.0),
            decline_count=3,
            parse_failure_count=2,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True


class TestAnyCombinator:
    def test_stops_when_any_criterion_met(self) -> None:
        """Verify Any stops when any child criterion agrees."""
        criterion = Any(MinSamples(10), MaxQueries(5))
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=1,
            parse_failure_count=1,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        # MaxQueries met (5), MinSamples not met (3 < 10)
        assert result is True

    def test_does_not_stop_when_no_criteria_met(self) -> None:
        """Verify Any does not stop when no child criteria agree."""
        criterion = Any(MinSamples(10), MaxQueries(20))
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=1,
            parse_failure_count=1,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is False

    def test_or_operator(self) -> None:
        """Verify | operator creates Any combinator."""
        criterion = MinSamples(10) | MaxQueries(5)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=1,
            parse_failure_count=1,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        assert result is True


class TestComplexCombinations:
    def test_complex_composition(self) -> None:
        """Verify complex compositions work correctly."""
        # (MinSamples(3) & MaxQueries(10)) | MinSamples(20)
        criterion = (MinSamples(3) & MaxQueries(10)) | MinSamples(20)
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0, 4.0, 5.0),
            decline_count=3,
            parse_failure_count=2,
            consecutive_declines=0,
        )

        result = criterion.should_stop(state, NumericalEstimator())

        # First branch is satisfied: 5 samples >= 3 AND 10 queries >= 10
        assert result is True
