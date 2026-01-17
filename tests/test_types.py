from __future__ import annotations

from winnow.types import Archetype, Estimate, SampleState


class TestSampleState:
    def test_query_count_includes_all_attempts(self) -> None:
        """Verify query_count sums samples, declines, and parse failures."""
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=2,
            parse_failure_count=1,
            consecutive_declines=0,
        )

        assert state.query_count == 6

    def test_empty_state_has_zero_query_count(self) -> None:
        """Verify empty state has zero query count."""
        state: SampleState[float] = SampleState(
            samples=(),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        assert state.query_count == 0

    def test_frozen_dataclass_is_hashable(self) -> None:
        """Verify SampleState can be used as dict key."""
        state: SampleState[float] = SampleState(
            samples=(1.0,),
            decline_count=0,
            parse_failure_count=0,
            consecutive_declines=0,
        )

        # Should not raise
        hash(state)


class TestEstimate:
    def test_estimate_with_value(self) -> None:
        """Verify Estimate stores all fields correctly."""
        estimate: Estimate[float] = Estimate(
            value=31.0,
            confidence=0.94,
            archetype=Archetype.CONFIDENT,
            sample_count=7,
            decline_count=1,
            samples=(31.0, 30.0, 31.0, 29.0, 31.0, 32.0, 31.0),
        )

        assert estimate.value == 31.0
        assert estimate.confidence == 0.94
        assert estimate.archetype == Archetype.CONFIDENT
        assert estimate.sample_count == 7
        assert estimate.decline_count == 1
        assert len(estimate.samples) == 7

    def test_estimate_with_none_value(self) -> None:
        """Verify Estimate can have None value for insufficient data."""
        estimate: Estimate[float] = Estimate(
            value=None,
            confidence=0.0,
            archetype=Archetype.INSUFFICIENT_DATA,
            sample_count=0,
            decline_count=5,
            samples=(),
        )

        assert estimate.value is None
        assert estimate.archetype == Archetype.INSUFFICIENT_DATA


class TestArchetype:
    def test_all_archetypes_exist(self) -> None:
        """Verify all expected archetypes are defined."""
        assert Archetype.CONFIDENT is not None
        assert Archetype.ACCEPTABLE is not None
        assert Archetype.UNCERTAIN is not None
        assert Archetype.INSUFFICIENT_DATA is not None
