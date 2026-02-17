from __future__ import annotations

from winnow.types import NoEstimate, SampleState, SampleStatus


class TestSampleState:
    def test_query_count_includes_all_attempts(self) -> None:
        """Verify query_count sums samples, declines, and parse failures."""
        state: SampleState[float] = SampleState(
            samples=(1.0, 2.0, 3.0),
            decline_count=2,
            parse_failure_count=1,
            consecutive_declines=0,
            current_estimate=NoEstimate,
            current_confidence=0.0,
            status=SampleStatus.COLLECTING,
            failure_reason=None,
        )

        assert state.query_count == 6
