from __future__ import annotations

from winnow.types import Estimate, SampleState


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
        """Verify Estimate stores value and confidence."""
        estimate: Estimate[float] = Estimate(value=31.0, confidence=0.94)

        assert estimate.value == 31.0
        assert estimate.confidence == 0.94

    def test_frozen_dataclass_is_hashable(self) -> None:
        """Verify Estimate can be used as dict key."""
        estimate: Estimate[float] = Estimate(value=31.0, confidence=0.94)

        # Should not raise
        hash(estimate)
